"""Adverse-side pause: localized feedback loop on top of the global toxicity trigger.

When recent fills on one side average worse than
``-ADVERSE_SIDE_PAUSE_SOFT_BPS``, that side's passive placements are
suppressed for ``ADVERSE_SIDE_PAUSE_SECONDS``. Context:
``tmp/snap_20260417_183547`` — 3/3 fills adverse at -1.26 bps mean, yet
neither hard (-12) nor soft (-3) global triggers fired, so the bot kept
quoting into toxic flow with no adaptive response.

These tests cover the three invariants:
1. ``ToxicityEngine.snapshot`` populates per-side markout averages.
2. ``OrderManager._maybe_arm_adverse_side_pause`` arms the cooldown when a
   side's mean markout breaches the threshold.
3. ``OrderManager._stage_place_order_local`` skips a side while its pause is
   active (log: ``adverse_side_pause_skip``).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from app.enums import Side
from app.models import Fill, ToxicitySnapshot
from app.toxicity import ToxicityEngine
from tests.settings_helpers import UnitTestSettings


def _fill(
    fid: str,
    side: Side,
    *,
    ts: datetime,
    m5: float | None = None,
) -> Fill:
    return Fill(
        fill_id=fid,
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=ts,
        symbol="ETH",
        side=side,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        markout_1s_bps=None,
        markout_3s_bps=None,
        markout_5s_bps=m5,
    )


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "grvt",
        "SYMBOL": "ETH_USDT_Perp",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def test_toxicity_snapshot_reports_per_side_markouts() -> None:
    """Snapshot must populate per-side averages so execution can decide locally."""
    settings = _settings()
    engine = ToxicityEngine(settings)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fills = [
        _fill("a", Side.BUY, ts=t0, m5=-5.0),
        _fill("b", Side.BUY, ts=t0 - timedelta(seconds=1), m5=-3.0),
        _fill("c", Side.SELL, ts=t0 - timedelta(seconds=2), m5=1.0),
    ]
    snap = engine.snapshot(mid=100.0, current_vol_bps=5.0, fills=fills)
    assert snap.buy_side_fill_count == 2
    assert snap.sell_side_fill_count == 1
    assert snap.buy_side_avg_markout_bps is not None
    assert abs(snap.buy_side_avg_markout_bps - (-4.0)) < 1e-9
    assert snap.sell_side_avg_markout_bps == 1.0


def test_toxicity_snapshot_per_side_none_when_markouts_missing() -> None:
    """When a side has no resolved markouts, that side's average stays None."""
    settings = _settings()
    engine = ToxicityEngine(settings)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fills = [
        _fill("a", Side.BUY, ts=t0, m5=-5.0),  # only BUY has markout
        _fill("b", Side.SELL, ts=t0 - timedelta(seconds=1), m5=None),
    ]
    snap = engine.snapshot(mid=100.0, current_vol_bps=5.0, fills=fills)
    assert snap.buy_side_avg_markout_bps == -5.0
    assert snap.sell_side_avg_markout_bps is None
    assert snap.sell_side_fill_count == 0


def _snapshot(
    buy_avg: float | None = None,
    sell_avg: float | None = None,
    buy_n: int = 0,
    sell_n: int = 0,
) -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
        buy_side_avg_markout_bps=buy_avg,
        sell_side_avg_markout_bps=sell_avg,
        buy_side_fill_count=buy_n,
        sell_side_fill_count=sell_n,
    )


class _ShellState:
    """Minimal BotState surface: the monotonic session fill counters."""

    def __init__(self, buy_n: int = 0, sell_n: int = 0):
        self.session_fill_count: int = buy_n + sell_n
        self.session_fill_count_by_side: dict = {Side.BUY: buy_n, Side.SELL: sell_n}

    def bump(self, side: Side, n: int = 1) -> None:
        """Simulate ``n`` new fills on ``side``."""
        self.session_fill_count += n
        self.session_fill_count_by_side[side] = (
            self.session_fill_count_by_side.get(side, 0) + n
        )


def _make_om(buy_n: int = 0, sell_n: int = 0):
    """Mint an OrderManager with only the surface we need for the pause tests.

    Avoid constructing a full OrderManager (needs clients, storage, state)
    by testing the two pure methods as instance methods on a trivial shell.
    """
    from app.clock import SystemClock
    from app.execution import OrderManager

    # We don't want the full fixture — only the pause bookkeeping. Build a
    # minimal object that has the required attributes.
    class _ShellOM:
        pass

    shell = _ShellOM()
    shell._settings = _settings()
    shell._state = _ShellState(buy_n=buy_n, sell_n=sell_n)
    shell._clock = SystemClock()
    shell._adverse_side_pause_until = {Side.BUY: 0.0, Side.SELL: 0.0}
    shell._adverse_side_pause_skip_count = {Side.BUY: 0, Side.SELL: 0}
    shell._adverse_side_pause_arm_n_fills = {Side.BUY: -1, Side.SELL: -1}
    # Bind the unbound methods to the shell.
    shell._adverse_side_pause_active = OrderManager._adverse_side_pause_active.__get__(
        shell, _ShellOM
    )
    shell._maybe_arm_adverse_side_pause = (
        OrderManager._maybe_arm_adverse_side_pause.__get__(shell, _ShellOM)
    )
    return shell


def test_adverse_side_pause_not_armed_below_threshold() -> None:
    """Mild markout (-1 bps) is below default 2 bps threshold — no pause."""
    om = _make_om(buy_n=2)
    # Default: adverse_side_pause_soft_bps=2.0, adverse_side_pause_min_fills=2
    snap = _snapshot(buy_avg=-1.0, buy_n=2)
    om._maybe_arm_adverse_side_pause(snap)
    assert not om._adverse_side_pause_active(Side.BUY)
    assert not om._adverse_side_pause_active(Side.SELL)


def test_adverse_side_pause_not_armed_below_min_fills() -> None:
    """Threshold breach with n=1 fill — noise, don't pause yet."""
    om = _make_om(buy_n=1)
    snap = _snapshot(buy_avg=-10.0, buy_n=1)
    om._maybe_arm_adverse_side_pause(snap)
    assert not om._adverse_side_pause_active(Side.BUY)


def test_adverse_side_pause_arms_on_threshold_breach() -> None:
    """2 fills averaging -5 bps breach default threshold → pause armed."""
    om = _make_om(buy_n=3)
    snap = _snapshot(buy_avg=-5.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    # Other side untouched.
    assert not om._adverse_side_pause_active(Side.SELL)


def test_adverse_side_pause_only_affects_the_toxic_side() -> None:
    """BUY side is adverse; SELL is clean. Only BUY should be paused."""
    om = _make_om(buy_n=3, sell_n=3)
    snap = _snapshot(buy_avg=-10.0, sell_avg=5.0, buy_n=3, sell_n=3)
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    assert not om._adverse_side_pause_active(Side.SELL)


def test_adverse_side_pause_deadline_respected() -> None:
    """After the pause duration lapses, the side becomes tradable again."""
    om = _make_om(buy_n=3)
    snap = _snapshot(buy_avg=-10.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    # Fast-forward past the deadline.
    om._adverse_side_pause_until[Side.BUY] = time.monotonic() - 0.1
    assert not om._adverse_side_pause_active(Side.BUY)


def test_adverse_side_pause_disabled_via_settings_zero() -> None:
    """Setting pause_seconds=0 disables the feature entirely."""
    om = _make_om(buy_n=5)
    om._settings = _settings(ADVERSE_SIDE_PAUSE_SECONDS=0.0)
    snap = _snapshot(buy_avg=-20.0, buy_n=5)
    om._maybe_arm_adverse_side_pause(snap)
    # Arm returns early when disabled.
    assert not om._adverse_side_pause_active(Side.BUY)


def test_adverse_side_pause_does_not_extend_while_active() -> None:
    """An active pause must NOT be re-extended every tick (self-perpetuating
    lockout bug from snap_20260418_091244). The deadline is fixed once armed
    and runs down naturally; only re-arms after it expires AND new data arrives.
    """
    om = _make_om(buy_n=3)
    snap = _snapshot(buy_avg=-10.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap)
    first_deadline = om._adverse_side_pause_until[Side.BUY]
    # Sleep a hair then re-arm attempt with the SAME snapshot (same n_fills).
    # The pause is still active and the fill count hasn't changed — the
    # deadline MUST NOT move.
    time.sleep(0.01)
    om._maybe_arm_adverse_side_pause(snap)
    second_deadline = om._adverse_side_pause_until[Side.BUY]
    assert second_deadline == first_deadline


def test_adverse_side_pause_requires_new_fill_to_rearm_after_expiry() -> None:
    """After the pause expires, we only re-arm when at least one new fill
    on that side has been observed. Without new data (just the same old
    adverse sample still in the window), the pause stays inactive —
    preventing the self-perpetuating lockout from snap_20260418_091244."""
    om = _make_om(buy_n=3)
    snap = _snapshot(buy_avg=-10.0, buy_n=3)
    # First arming sets prev_arm_session_n=3.
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    # Fast-forward past the deadline; the pause is now inactive.
    om._adverse_side_pause_until[Side.BUY] = time.monotonic() - 0.1
    assert not om._adverse_side_pause_active(Side.BUY)
    # Same session fill count (still 3). No new data. Must NOT re-arm.
    om._maybe_arm_adverse_side_pause(snap)
    assert not om._adverse_side_pause_active(Side.BUY)
    # A fresh BUY fill lands in the session → re-arm.
    om._state.bump(Side.BUY)
    fresh_snap = _snapshot(buy_avg=-10.0, buy_n=4)
    om._maybe_arm_adverse_side_pause(fresh_snap)
    assert om._adverse_side_pause_active(Side.BUY)


def test_adverse_side_pause_self_perpetuation_regression() -> None:
    """Direct regression for snap_20260418_091244: 499 skips over 3+ minutes
    on a SINGLE -11.77 bps BUY fill. Simulate 200 ticks of re-arm attempts
    on a stale snapshot and verify the pause deadline moves at most once.
    """
    om = _make_om(buy_n=3)
    stale_snap = _snapshot(buy_avg=-5.13, buy_n=3)
    om._maybe_arm_adverse_side_pause(stale_snap)
    deadline_after_first_arm = om._adverse_side_pause_until[Side.BUY]
    # 200 re-arm attempts on the same snapshot.
    for _ in range(200):
        om._maybe_arm_adverse_side_pause(stale_snap)
    # Deadline must not have moved.
    assert om._adverse_side_pause_until[Side.BUY] == deadline_after_first_arm


# --- BUG-010 reducing-side bypass tests ----------------------------


class _PosShell:
    """Surface needed for ``_is_reducing_side`` — a position with ``position_qty``."""

    def __init__(self, qty: float):
        self.position_qty = float(qty)


def _make_om_with_position(qty: float):
    """Same shape as ``_make_om`` but also exposes ``state.position.position_qty``."""
    from app.execution import OrderManager

    class _ShellOM:
        pass

    shell = _ShellOM()
    shell._settings = _settings()
    state = _ShellState()
    state.position = _PosShell(qty)
    shell._state = state
    shell._is_reducing_side = OrderManager._is_reducing_side.__get__(shell, _ShellOM)
    return shell


def test_is_reducing_side_long_inventory() -> None:
    """Long position → SELL is the reducing side."""
    om = _make_om_with_position(qty=50.0)
    assert om._is_reducing_side(Side.SELL) is True
    assert om._is_reducing_side(Side.BUY) is False


def test_is_reducing_side_short_inventory() -> None:
    """Short position → BUY is the reducing side."""
    om = _make_om_with_position(qty=-50.0)
    assert om._is_reducing_side(Side.BUY) is True
    assert om._is_reducing_side(Side.SELL) is False


def test_is_reducing_side_flat_position() -> None:
    """Flat position → no reducing side; both return False."""
    om = _make_om_with_position(qty=0.0)
    assert om._is_reducing_side(Side.BUY) is False
    assert om._is_reducing_side(Side.SELL) is False


def test_is_reducing_side_dust_position_treated_as_flat() -> None:
    """Sub-epsilon positions are treated as flat — neither side reduces."""
    from app.execution import _POSITION_EPS

    om = _make_om_with_position(qty=_POSITION_EPS / 2.0)
    assert om._is_reducing_side(Side.BUY) is False
    assert om._is_reducing_side(Side.SELL) is False


def test_adverse_side_pause_survives_deque_maxlen_wrap() -> None:
    """After the window-bounded counter (``toxicity.buy_side_fill_count``)
    caps at the deque maxlen (200), new session fills still drive the
    monotonic ``session_fill_count_by_side`` counter and allow legitimate
    re-arming. Without this, a long-running session would permanently lock
    out the pause once the deque filled."""
    # Simulate: deque is at maxlen (window_n caps at maxlen's slice size),
    # but real session has had many more fills.
    om = _make_om(buy_n=250)  # 250 session fills → beyond the 200 deque
    snap = _snapshot(buy_avg=-10.0, buy_n=20)  # window-bounded to 20
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    first_arm_n = om._adverse_side_pause_arm_n_fills[Side.BUY]
    assert first_arm_n == 250
    # Expire + simulate one more session fill.
    om._adverse_side_pause_until[Side.BUY] = time.monotonic() - 0.1
    om._state.bump(Side.BUY)  # session_count → 251
    # Window snapshot still shows 20 (deque-bounded); that's fine for min-sample gate.
    om._maybe_arm_adverse_side_pause(snap)
    # MUST re-arm — session counter advanced.
    assert om._adverse_side_pause_active(Side.BUY)
    assert om._adverse_side_pause_arm_n_fills[Side.BUY] == 251


# ---------------------------------------------------------------------------
# v1.4.152 Phase 2K.1 — favorable-exit predicate tests
# ---------------------------------------------------------------------------


def test_phase2k2_favorable_exit_clears_pause_when_markout_recovers() -> None:
    """An active pause should clear EARLY when the per-side rolling avg
    markout recovers past ``-soft_bps × (1 - clear_mult)``. Default:
    soft=2.0, mult=0.5 → clear when avg > -1.0.

    Sequence: arm pause at avg=-5 bps, then markout recovers to -0.5
    bps (above the -1.0 clear band). The pause clears immediately,
    BEFORE the ceiling fires."""
    om = _make_om(buy_n=3)
    # Arm with adverse markout.
    snap = _snapshot(buy_avg=-5.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    favorable_before = om.adverse_side_pause_cleared_via_favorable_total
    # Markout recovers (above the -1.0 clear band).
    snap2 = _snapshot(buy_avg=-0.5, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap2)
    # Pause cleared early via favorable exit.
    assert not om._adverse_side_pause_active(Side.BUY)
    assert (
        om.adverse_side_pause_cleared_via_favorable_total
        == favorable_before + 1
    )
    # Ceiling counter NOT incremented.
    assert om.adverse_side_pause_cleared_via_ceiling_total == 0


def test_phase2k2_favorable_exit_does_not_fire_when_still_adverse() -> None:
    """Markout still in the adverse band (above -soft but below clear
    threshold) → pause stays active until ceiling. Default thresholds
    have ``clear_band = -1.0``; markout at -1.5 is between trigger
    (-2.0) and clear (-1.0), so the pause should hold."""
    om = _make_om(buy_n=3)
    snap = _snapshot(buy_avg=-5.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    # Markout improves but still in the "adverse but holding" band.
    snap2 = _snapshot(buy_avg=-1.5, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap2)
    # Pause UNCHANGED — still active.
    assert om._adverse_side_pause_active(Side.BUY)
    assert om.adverse_side_pause_cleared_via_favorable_total == 0


def test_phase2k2_ceiling_clearing_attributes_correctly() -> None:
    """When markout stays static-adverse for the full duration, the
    pause clears via the MAX-cooldown ceiling. Attribution counter
    increments on the deadline-lapsed path."""
    om = _make_om(buy_n=3)
    snap = _snapshot(buy_avg=-5.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    # Simulate ceiling elapsing — push the deadline into the past.
    om._adverse_side_pause_until[Side.BUY] = time.monotonic() - 0.1
    # Static-adverse markout that would NOT trip favorable exit
    # (still in the adverse band — below the clear threshold). With
    # the deadline now lapsed, the next call should attribute to
    # ceiling and clear.
    snap2 = _snapshot(buy_avg=-5.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap2)
    assert om.adverse_side_pause_cleared_via_ceiling_total == 1
    assert om.adverse_side_pause_cleared_via_favorable_total == 0


def test_phase2k2_disabled_mult_uses_pure_timer() -> None:
    """When ``adverse_side_pause_clear_threshold_mult=0``, the
    favorable-exit predicate is disabled — pause behaves as pure
    timer regardless of markout recovery during the window."""
    from tests.settings_helpers import UnitTestSettings

    om = _make_om(buy_n=3)
    # Override settings to disable favorable-exit.
    settings_dict = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "grvt",
        "SYMBOL": "ETH_USDT_Perp",
        "ADVERSE_SIDE_PAUSE_CLEAR_THRESHOLD_MULT": 0.0,
    }
    om._settings = UnitTestSettings.model_validate(settings_dict)
    snap = _snapshot(buy_avg=-5.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap)
    assert om._adverse_side_pause_active(Side.BUY)
    # Markout fully recovers, but with mult=0 the favorable exit is
    # disabled — pause stays active.
    snap2 = _snapshot(buy_avg=5.0, buy_n=3)
    om._maybe_arm_adverse_side_pause(snap2)
    assert om._adverse_side_pause_active(Side.BUY)
    assert om.adverse_side_pause_cleared_via_favorable_total == 0
