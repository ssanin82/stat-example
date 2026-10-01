"""v1.5.197 — defensive memory time-decay tests.

Pre-v1.5.197 the bot's defensive gates (toxicity, mae_gate,
at_touch_adverse_pause, realised_edge_side_suppress) had memory
that decayed ONLY by new-fill volume. When the gate engaged, it
suppressed fills; without new fills, the gate's input signal
(recent_fills) couldn't refresh; gate stayed engaged forever.

Failure mode documented in:
* snapshot v1.5.195-260527-161959 — toxicity_hard SF loop after a
  single -13 bp fill stayed in recent_fills for 10+ minutes
* snapshot v1.5.193-260527-155737 — structural_bias_throttle locked
  in QUOTE_SELL_ONLY for 23 minutes after a single asymmetric burst

v1.5.197 fixes three layers:

1. ``ToxicityEngine.snapshot(now=...)`` — filters recent_fills by
   ``ts_fill`` age before processing. Fills older than
   ``TOXICITY_RECENT_FILLS_MAX_AGE_SECONDS`` (default 600s) are
   excluded.

2. Idle-clear predicates on the three position-favorable gates
   (mae_gate, at_touch_adverse_pause, realised_edge_side_suppress)
   — clear when no fill has arrived for ``IDLE_CLEAR_SECONDS``.

3. ``inventory_exec_bias`` counter half-life decay — applied inside
   ``Bot._apply_structural_bias_throttle_gate`` when no fill has
   arrived for ``INVENTORY_EXEC_BIAS_IDLE_DECAY_SECONDS``.

Each layer is tested in isolation.
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from app.at_touch_adverse_pause import AtTouchAdversePause
from app.enums import Side
from app.mae_gate import MaeGateState, evaluate_idle_clear
from app.models import Fill
from app.realised_edge_side_suppress import RealisedEdgeSideSuppressGate


# ---------------------------------------------------------------------------
# Layer 1 — ToxicityEngine recent_fills time-decay
# ---------------------------------------------------------------------------


def _make_fill(ts: datetime, side: Side, markout: float) -> Fill:
    return Fill(
        fill_id=f"f-{ts.timestamp()}-{side.value}",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=ts,
        symbol="TEST",
        side=side,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=0.0,
        liquidity_flag="maker",
        mid_at_fill=1.0,
        markout_5s_bps=markout,
    )


class _StubSettings:
    """Minimal Settings stand-in for ToxicityEngine."""

    def __init__(self, max_age_seconds: float = 600.0):
        self.toxicity_enabled = True
        self.toxicity_markout_soft_bps = 3.0
        self.toxicity_markout_hard_bps = 8.0
        self.toxicity_one_sided_fill_ratio = 0.75
        self.toxicity_one_sided_min_fills = 4
        self.toxicity_recent_fills_max_age_seconds = max_age_seconds


def test_toxicity_engine_filters_stale_fills_by_age() -> None:
    """A -13 bp adverse fill that's 700s old should NOT trigger
    toxicity_hard when max_age=600. The engine drops it as stale."""
    from app.toxicity import ToxicityEngine

    settings = _StubSettings(max_age_seconds=600.0)
    engine = ToxicityEngine(settings)  # type: ignore[arg-type]
    engine.set_baseline_vol(5.0)

    now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
    stale_fill = _make_fill(
        ts=now - timedelta(seconds=700),  # 700s old > 600s cutoff
        side=Side.BUY,
        markout=-13.18,  # catastrophic pickoff
    )

    snap = engine.snapshot(
        mid=1.0,
        current_vol_bps=5.0,
        fills=[stale_fill],
        now=now,
    )
    # The -13 bp fill is filtered out; no fills remain → no triggers.
    assert snap.hard_trigger is False
    assert snap.delayed_markout_sample_count == 0


def test_toxicity_engine_keeps_fresh_fills() -> None:
    """A fresh adverse fill (within max_age) still triggers."""
    from app.toxicity import ToxicityEngine

    settings = _StubSettings(max_age_seconds=600.0)
    engine = ToxicityEngine(settings)  # type: ignore[arg-type]
    engine.set_baseline_vol(5.0)

    now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
    # 5 fresh fills, all 1-minute old, all averaging -10 bp adverse.
    fills = [
        _make_fill(
            ts=now - timedelta(seconds=60 + i),
            side=Side.BUY,
            markout=-10.0,
        )
        for i in range(5)
    ]
    snap = engine.snapshot(
        mid=1.0,
        current_vol_bps=5.0,
        fills=fills,
        now=now,
    )
    # avg adverse -10 <= -8 (hard threshold), engine fires.
    assert snap.hard_trigger is True
    assert snap.delayed_markout_sample_count == 5


def test_toxicity_engine_disabled_when_max_age_zero() -> None:
    """Setting max_age_seconds=0 disables the time filter
    (backward-compat for tests / pre-v1.5.197 behavior)."""
    from app.toxicity import ToxicityEngine

    settings = _StubSettings(max_age_seconds=0.0)
    engine = ToxicityEngine(settings)  # type: ignore[arg-type]
    engine.set_baseline_vol(5.0)

    now = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)
    # Stale fill — would be filtered if max_age were 600.
    stale_fill = _make_fill(
        ts=now - timedelta(seconds=10_000),
        side=Side.BUY,
        markout=-13.18,
    )
    snap = engine.snapshot(
        mid=1.0,
        current_vol_bps=5.0,
        fills=[stale_fill],
        now=now,
    )
    # No filtering — the stale fill is still processed.
    assert snap.delayed_markout_sample_count == 1


def test_toxicity_engine_no_now_falls_back_to_legacy() -> None:
    """When ``now`` is not passed (legacy callers / tests), the
    engine processes all fills regardless of age."""
    from app.toxicity import ToxicityEngine

    settings = _StubSettings(max_age_seconds=600.0)
    engine = ToxicityEngine(settings)  # type: ignore[arg-type]
    engine.set_baseline_vol(5.0)

    stale_fill = _make_fill(
        ts=datetime(2020, 1, 1, tzinfo=timezone.utc),  # very old
        side=Side.BUY,
        markout=-13.18,
    )
    # No ``now`` kwarg — fall back to legacy behavior.
    snap = engine.snapshot(mid=1.0, current_vol_bps=5.0, fills=[stale_fill])
    assert snap.delayed_markout_sample_count == 1


# ---------------------------------------------------------------------------
# Layer 2a — mae_gate idle-clear
# ---------------------------------------------------------------------------


def test_mae_gate_idle_clear_fires_after_threshold() -> None:
    """Gate active for >300s with no fills → idle-clear fires."""
    state = MaeGateState(lock=threading.Lock())
    state.cooldown_until_mono = 1000.0  # active
    state.was_active_last_call = True
    cleared = evaluate_idle_clear(
        state,
        now_mono=500.0,
        last_fill_mono=100.0,  # 400s ago > 300s threshold
        idle_clear_seconds=300.0,
    )
    assert cleared is True
    assert state.cooldown_until_mono == 0.0
    assert state.cleared_via_position_favorable_total == 1


def test_mae_gate_idle_clear_no_fire_before_threshold() -> None:
    """Gate active but fill was recent → idle-clear doesn't fire."""
    state = MaeGateState(lock=threading.Lock())
    state.cooldown_until_mono = 1000.0
    cleared = evaluate_idle_clear(
        state,
        now_mono=500.0,
        last_fill_mono=400.0,  # only 100s ago < 300s threshold
        idle_clear_seconds=300.0,
    )
    assert cleared is False
    assert state.cooldown_until_mono == 1000.0


def test_mae_gate_idle_clear_no_fire_when_inactive() -> None:
    """Gate already inactive (no cooldown) → idle-clear is a no-op."""
    state = MaeGateState(lock=threading.Lock())
    state.cooldown_until_mono = 0.0  # inactive
    cleared = evaluate_idle_clear(
        state,
        now_mono=500.0,
        last_fill_mono=100.0,
        idle_clear_seconds=300.0,
    )
    assert cleared is False


def test_mae_gate_idle_clear_no_fire_when_last_fill_none() -> None:
    """No fill recorded yet → idle-clear doesn't fire."""
    state = MaeGateState(lock=threading.Lock())
    state.cooldown_until_mono = 1000.0
    cleared = evaluate_idle_clear(
        state,
        now_mono=500.0,
        last_fill_mono=None,
        idle_clear_seconds=300.0,
    )
    assert cleared is False


# ---------------------------------------------------------------------------
# Layer 2b — at_touch_adverse_pause idle-clear
# ---------------------------------------------------------------------------


def test_at_touch_adverse_pause_idle_clear_fires() -> None:
    """Both sides paused, no fills for >300s → both clear via idle."""
    gate = AtTouchAdversePause(
        threshold_bps=-5.0,
        pause_seconds=30.0,
        min_fills=3,
    )
    gate._buy_paused_until_mono = 1000.0
    gate._sell_paused_until_mono = 1000.0
    cleared = gate.try_clear_via_idle(
        now_mono=500.0,
        last_fill_mono=100.0,  # 400s ago > 300s threshold
        idle_clear_seconds=300.0,
    )
    assert cleared is True
    assert gate._buy_paused_until_mono == 0.0
    assert gate._sell_paused_until_mono == 0.0


def test_at_touch_adverse_pause_idle_clear_no_fire_recent_fill() -> None:
    gate = AtTouchAdversePause(
        threshold_bps=-5.0,
        pause_seconds=30.0,
        min_fills=3,
    )
    gate._buy_paused_until_mono = 1000.0
    cleared = gate.try_clear_via_idle(
        now_mono=500.0,
        last_fill_mono=400.0,
        idle_clear_seconds=300.0,
    )
    assert cleared is False
    assert gate._buy_paused_until_mono == 1000.0


# ---------------------------------------------------------------------------
# Layer 2c — realised_edge_side_suppress idle-clear
# ---------------------------------------------------------------------------


def test_realised_edge_side_suppress_idle_clear_fires() -> None:
    gate = RealisedEdgeSideSuppressGate(
        threshold_bps=-4.0,
        cooldown_seconds=60.0,
        min_fills=4,
    )
    gate._buy_suppressed_until_mono = 1000.0
    gate._sell_suppressed_until_mono = 1000.0
    cleared = gate.try_clear_via_idle(
        now_mono=500.0,
        last_fill_mono=100.0,
        idle_clear_seconds=300.0,
    )
    assert cleared is True
    assert gate._buy_suppressed_until_mono == 0.0
    assert gate._sell_suppressed_until_mono == 0.0
