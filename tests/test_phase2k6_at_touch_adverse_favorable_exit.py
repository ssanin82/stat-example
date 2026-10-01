"""Phase 2K.6 — favorable-exit predicate for the
``at_touch_adverse_pause`` gate.

Architecture parity with Phase 2K.3 / 2K.4 / 2K.5: the legacy fixed
``AT_TOUCH_ADVERSE_PAUSE_SECONDS`` timer becomes the MAX-cooldown
ceiling; the gate also clears EARLY when the per-side median markout
has recovered past ``threshold_bps × clear_band_mult`` and held there
for ``favorable_exit_dwell_seconds``.

Re-flare semantics: a new at-touch fill that drags the median back
below the clear band resets the dwell timer (and may re-arm the pause).
Exit attribution: ``cleared_via_favorable_total[side]`` vs
``cleared_via_ceiling_total[side]`` lets the operator tune the knobs
from the dashboard.

These tests cover the pure predicate + counter behaviour.
Integration with the bot's tick loop is exercised indirectly by the
``test_at_touch_adverse_pause.py`` regression suite (which still
covers the pure-timer ceiling path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.at_touch_adverse_pause import AtTouchAdversePause
from app.enums import Side


@dataclass
class _FakeFill:
    side: Side
    markout_5s_bps: Optional[float]
    quote_aggressiveness: Optional[str]


def _gate(**overrides) -> AtTouchAdversePause:
    base = dict(
        threshold_bps=-5.0,
        pause_seconds=30.0,
        min_fills=3,
        window_size=20,
        favorable_exit_enabled=True,
        clear_band_mult=0.5,
        favorable_exit_dwell_seconds=5.0,
    )
    base.update(overrides)
    return AtTouchAdversePause(**base)


def _arm_buy(gate: AtTouchAdversePause, now: float) -> None:
    """Drive 3 adverse at_touch fills onto the BUY side so the median
    drops below -5 bps and arms the pause."""
    for mo in (-8.0, -10.0, -12.0):
        gate.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), now
        )


# ---------------------------------------------------------------------------
# Favorable-exit predicate: feeding recovery fills clears early
# ---------------------------------------------------------------------------


def test_favorable_exit_clears_when_median_recovers_above_clear_band() -> None:
    """threshold=-5, mult=0.5 → clear band -2.5. After arming with
    median -10, feed 3 fresh fills at median +1 (well above -2.5);
    after dwell, the pause clears via favorable-exit."""
    g = _gate(threshold_bps=-5.0, clear_band_mult=0.5,
              favorable_exit_dwell_seconds=5.0)
    _arm_buy(g, 100.0)
    assert g.is_paused(Side.BUY, 100.0)

    # Feed 3 favorable at_touch fills. Each call also re-evaluates the
    # favorable predicate inside observe_resolved_5s_markout.
    # With min_fills=3, the median of the LAST 3 samples is what counts.
    for mo in (+1.0, +2.0, +0.5):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.5
        )

    # Dwell hasn't elapsed yet (we're at 100.5; need 105.5).
    assert g.is_paused(Side.BUY, 100.5)
    # Now poll at 106.0 — past dwell. Should clear via favorable.
    assert not g.is_paused(Side.BUY, 106.0)
    snap = g.snapshot_dict(106.0)
    assert snap["buy"]["cleared_via_favorable_total"] == 1
    assert snap["buy"]["cleared_via_ceiling_total"] == 0


def test_favorable_exit_does_not_fire_before_dwell_completes() -> None:
    """The predicate flips True at 100.5 (recovery), but the gate
    must not clear before the 5 s dwell elapses."""
    g = _gate(threshold_bps=-5.0, clear_band_mult=0.5,
              favorable_exit_dwell_seconds=5.0)
    _arm_buy(g, 100.0)
    for mo in (+1.0, +2.0, +0.5):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.5
        )
    # Mid-dwell — still paused.
    for t in (100.5, 102.0, 104.5, 105.4):
        assert g.is_paused(Side.BUY, t), f"unexpected clear at t={t}"
    # Past dwell.
    assert not g.is_paused(Side.BUY, 105.6)


def test_reflare_resets_dwell() -> None:
    """If a fresh adverse fill drags the median back below the clear
    band mid-dwell, the dwell resets — the gate stays paused for the
    full ceiling."""
    g = _gate(threshold_bps=-5.0, clear_band_mult=0.5,
              favorable_exit_dwell_seconds=5.0, pause_seconds=30.0)
    _arm_buy(g, 100.0)
    # Recovery starts dwell at 100.5.
    for mo in (+1.0, +2.0, +0.5):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.5
        )
    assert g.is_paused(Side.BUY, 103.0)  # mid-dwell
    # Re-flare: a new adverse fill drags median back below the clear
    # band. min_fills=3 → last 3 samples = (+2.0, +0.5, -10.0) →
    # median = +0.5? No — sorted gives [-10, +0.5, +2.0] → median 0.5,
    # still above -2.5. So we need a stronger reflare: -10, -10, -10.
    for mo in (-10.0, -10.0, -10.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 103.0
        )
    # Now median = -10 → re-arms; new paused_until = 133.
    # Dwell must have been reset.
    snap_mid = g.snapshot_dict(103.0)
    assert snap_mid["buy"]["favorable_dwell_active"] is False
    # Even later, no favorable clear should have happened.
    assert g.is_paused(Side.BUY, 108.0)
    assert g.snapshot_dict(108.0)["buy"]["cleared_via_favorable_total"] == 0


def test_ceiling_fires_when_median_stays_adverse() -> None:
    """No recovery → timer expires naturally → ceiling counter
    increments exactly once on the active→cleared edge."""
    g = _gate(threshold_bps=-5.0, pause_seconds=30.0)
    _arm_buy(g, 100.0)
    assert g.is_paused(Side.BUY, 100.0)
    # Median stays at -10 throughout. No recovery fills.
    assert g.is_paused(Side.BUY, 129.0)
    # Past the 30 s ceiling.
    assert not g.is_paused(Side.BUY, 131.0)
    snap = g.snapshot_dict(131.0)
    assert snap["buy"]["cleared_via_ceiling_total"] == 1
    assert snap["buy"]["cleared_via_favorable_total"] == 0


def test_ceiling_attribution_only_fires_once_per_arming() -> None:
    """Polling ``is_paused`` repeatedly after the ceiling fires must
    not double-count. Edge detection uses ``was_paused_last``."""
    g = _gate(threshold_bps=-5.0, pause_seconds=30.0)
    _arm_buy(g, 100.0)
    # Trip through the timer.
    g.is_paused(Side.BUY, 100.0)
    g.is_paused(Side.BUY, 120.0)
    g.is_paused(Side.BUY, 131.0)  # ceiling fires
    g.is_paused(Side.BUY, 140.0)  # post-ceiling polls
    g.is_paused(Side.BUY, 200.0)
    snap = g.snapshot_dict(200.0)
    assert snap["buy"]["cleared_via_ceiling_total"] == 1


def test_partial_recovery_below_clear_band_no_favorable_exit() -> None:
    """threshold=-5, mult=0.5 → clear band -2.5. Median that recovers
    to only -4 (above trigger but BELOW clear band) does NOT fire
    favorable-exit — hysteresis requires fuller recovery."""
    g = _gate(threshold_bps=-5.0, clear_band_mult=0.5)
    _arm_buy(g, 100.0)
    for mo in (-4.0, -3.5, -4.5):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.5
        )
    # Median ~ -4 < -2.5 → predicate doesn't hold.
    assert g.snapshot_dict(110.0)["buy"]["favorable_dwell_active"] is False
    # Gate clears via ceiling at t=130+.
    assert not g.is_paused(Side.BUY, 131.0)
    snap = g.snapshot_dict(131.0)
    assert snap["buy"]["cleared_via_favorable_total"] == 0
    assert snap["buy"]["cleared_via_ceiling_total"] == 1


def test_clear_band_mult_zero_requires_positive_median() -> None:
    """mult=0 → clear band 0.0 → favorable-exit only fires when the
    median is strictly positive."""
    g = _gate(threshold_bps=-5.0, clear_band_mult=0.0,
              favorable_exit_dwell_seconds=5.0)
    _arm_buy(g, 100.0)
    # Median -0.5 is above trigger but NOT positive → no favorable.
    for mo in (-0.3, -0.7, -0.5):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.5
        )
    assert g.is_paused(Side.BUY, 110.0)
    # Now feed positive markouts.
    for mo in (+1.0, +2.0, +0.5):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 110.0
        )
    # Past dwell.
    assert not g.is_paused(Side.BUY, 115.5)
    assert g.snapshot_dict(115.5)["buy"]["cleared_via_favorable_total"] == 1


def test_clear_band_mult_one_clears_at_trigger_boundary() -> None:
    """mult=1.0 → clear band equals the trigger; favorable-exit fires
    as soon as the median rises just above the trigger value."""
    g = _gate(threshold_bps=-5.0, clear_band_mult=1.0,
              favorable_exit_dwell_seconds=5.0)
    _arm_buy(g, 100.0)
    # Median -4 > -5 → predicate holds. Dwell.
    for mo in (-4.0, -4.5, -3.5):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.5
        )
    assert not g.is_paused(Side.BUY, 106.0)
    assert g.snapshot_dict(106.0)["buy"]["cleared_via_favorable_total"] == 1


def test_favorable_exit_disabled_uses_pure_timer() -> None:
    """``favorable_exit_enabled=False`` → legacy behaviour: only the
    timer can clear the pause. Recovery fills don't fire favorable-exit."""
    g = _gate(threshold_bps=-5.0, favorable_exit_enabled=False,
              pause_seconds=30.0)
    _arm_buy(g, 100.0)
    for mo in (+5.0, +5.0, +5.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.5
        )
    # Even with strongly positive markouts, the gate stays paused
    # until the ceiling fires.
    assert g.is_paused(Side.BUY, 120.0)
    assert not g.is_paused(Side.BUY, 131.0)
    snap = g.snapshot_dict(131.0)
    assert snap["buy"]["cleared_via_favorable_total"] == 0
    assert snap["buy"]["cleared_via_ceiling_total"] == 1


def test_per_side_attribution_independence() -> None:
    """BUY favorable-exit doesn't affect SELL counters and vice-versa."""
    g = _gate(threshold_bps=-5.0, clear_band_mult=0.5,
              favorable_exit_dwell_seconds=5.0, pause_seconds=30.0)
    # Arm BUY adverse; SELL stays clean.
    _arm_buy(g, 100.0)
    # Recover BUY.
    for mo in (+1.0, +2.0, +0.5):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.5
        )
    assert not g.is_paused(Side.BUY, 106.0)
    # Arm SELL.
    for mo in (-8.0, -10.0, -12.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.SELL, mo, "at_touch"), 106.0
        )
    assert g.is_paused(Side.SELL, 106.0)
    # Let SELL ceiling fire.
    assert not g.is_paused(Side.SELL, 137.0)
    snap = g.snapshot_dict(137.0)
    assert snap["buy"]["cleared_via_favorable_total"] == 1
    assert snap["buy"]["cleared_via_ceiling_total"] == 0
    assert snap["sell"]["cleared_via_favorable_total"] == 0
    assert snap["sell"]["cleared_via_ceiling_total"] == 1


def test_snapshot_surfaces_new_attribution_fields() -> None:
    """The snapshot_dict shape grows Phase 2K.6 fields without breaking
    the existing keys other code reads."""
    g = _gate()
    snap = g.snapshot_dict(100.0)
    # Existing keys still present.
    for k in ("enabled", "threshold_bps", "min_fills", "pause_seconds"):
        assert k in snap
    # New Phase 2K.6 settings exposed for dashboard calibration.
    assert snap["favorable_exit_enabled"] is True
    assert snap["clear_band_mult"] == 0.5
    assert snap["favorable_exit_dwell_seconds"] == 5.0
    # Per-side counters initialised to zero.
    for side in ("buy", "sell"):
        assert snap[side]["cleared_via_favorable_total"] == 0
        assert snap[side]["cleared_via_ceiling_total"] == 0
        assert snap[side]["favorable_dwell_active"] is False
