"""Phase 2D (v1.5.26) -- residual-decay trigger for adaptive_widen.

Adds a composite-signal arming branch to the existing adaptive_widen
gate. The trigger reads the rolling mean of ``closed_pnl_bps +
rebate_bps - markout_5s_bps`` per fill and arms when the mean drops
below ``threshold_bps`` (negative) for ``dwell_seconds`` continuously.

Tests cover the pure tracker (``ResidualDecayTracker``) and the
helper ``compute_residual_bps``. The bot.py integration (live wiring
into the adaptive_widen arming block) is exercised by the existing
adaptive_widen tests via the same arming-on-OR-of-want_arm_* path.
"""

from __future__ import annotations

import pytest

from app.residual_decay_tracker import (
    ResidualDecayTracker,
    compute_residual_bps,
)


# ---------------------------------------------------------------------------
# compute_residual_bps
# ---------------------------------------------------------------------------


def test_compute_residual_bps_zero_when_balanced():
    """closed_pnl=0, fee=0, markout=0 -> residual = 0."""
    r = compute_residual_bps(
        closed_pnl=0.0, fee=0.0, markout_5s_bps=0.0, notional=100.0
    )
    assert r == pytest.approx(0.0)


def test_compute_residual_bps_rebate_only_positive():
    """Maker rebate (negative fee) shows as positive bps. Notional
    $100, fee=-$0.01 (1 bp rebate) -> +1 bp residual when markout=0.

    Conversion: rebate_bps = (-fee / notional) * 1e4. So a $0.01
    rebate on $100 notional = 0.01/100 * 1e4 = 1 bp."""
    r = compute_residual_bps(
        closed_pnl=0.0,
        fee=-0.01,
        markout_5s_bps=0.0,
        notional=100.0,
    )
    assert r == pytest.approx(1.0)


def test_compute_residual_bps_adverse_markout_subtracts():
    """Positive markout_5s_bps = mid moved against us. The formula
    SUBTRACTS markout so adverse selection drags residual negative."""
    r = compute_residual_bps(
        closed_pnl=0.0,
        fee=0.0,
        markout_5s_bps=3.0,  # 3 bp adverse
        notional=100.0,
    )
    assert r == pytest.approx(-3.0)


def test_compute_residual_bps_closed_pnl_offsets_adverse():
    """Realised PnL from closing a position offsets adverse markout.
    Notional $100, closed_pnl=$0.04 (4 bp), markout=3 bp adverse,
    no fee -> residual = 4 + 0 - 3 = +1 bp."""
    r = compute_residual_bps(
        closed_pnl=0.04,
        fee=0.0,
        markout_5s_bps=3.0,
        notional=100.0,
    )
    assert r == pytest.approx(1.0)


def test_compute_residual_bps_full_formula():
    """All four components contributing. notional=$100,
    closed_pnl=$0.02 (2 bp), fee=-$0.01 (1 bp rebate),
    markout=4 bp adverse -> residual = 2 + 1 - 4 = -1 bp."""
    r = compute_residual_bps(
        closed_pnl=0.02,
        fee=-0.01,
        markout_5s_bps=4.0,
        notional=100.0,
    )
    assert r == pytest.approx(-1.0)


def test_compute_residual_bps_none_when_markout_unresolved():
    """``markout_5s_bps=None`` (the fill hasn't aged to 5s yet) ->
    return None so the caller knows to skip."""
    r = compute_residual_bps(
        closed_pnl=0.0,
        fee=-0.001,
        markout_5s_bps=None,
        notional=100.0,
    )
    assert r is None


def test_compute_residual_bps_none_on_zero_notional():
    """Defensive: zero / negative notional -> None (division-by-zero
    would otherwise blow up)."""
    r = compute_residual_bps(
        closed_pnl=0.0, fee=0.0, markout_5s_bps=0.0, notional=0.0
    )
    assert r is None


def test_compute_residual_bps_missing_closed_pnl_treated_as_zero():
    """``closed_pnl=None`` is the common opening-fill case; treat as 0
    so the formula reduces to rebate - markout."""
    r = compute_residual_bps(
        closed_pnl=None,
        fee=-0.01,
        markout_5s_bps=2.0,
        notional=100.0,
    )
    assert r == pytest.approx(-1.0)  # 0 + 1 - 2


# ---------------------------------------------------------------------------
# ResidualDecayTracker -- enable / disable
# ---------------------------------------------------------------------------


def _tracker(
    *,
    threshold_bps: float = -0.5,
    dwell_seconds: float = 60.0,
    window_fills: int = 50,
    min_fills: int = 5,
) -> ResidualDecayTracker:
    return ResidualDecayTracker(
        threshold_bps=threshold_bps,
        dwell_seconds=dwell_seconds,
        window_fills=window_fills,
        min_fills=min_fills,
    )


def test_disabled_when_threshold_is_zero():
    """Default config has threshold=0.0 = disabled. The trigger only
    makes sense for negative thresholds."""
    t = _tracker(threshold_bps=0.0)
    assert t.enabled() is False
    assert t.is_armed(now_mono=1000.0) is False


def test_disabled_when_threshold_is_positive():
    """Defensive: positive threshold doesn't make sense (we'd never
    want the bot to widen because the residual is too GOOD)."""
    t = _tracker(threshold_bps=1.0)
    assert t.enabled() is False


def test_enabled_with_negative_threshold():
    t = _tracker(threshold_bps=-0.5, dwell_seconds=60.0)
    assert t.enabled() is True


# ---------------------------------------------------------------------------
# ResidualDecayTracker -- buffer fill + mean
# ---------------------------------------------------------------------------


def test_current_mean_none_below_min_fills():
    """Need at least ``min_fills`` samples before forming a verdict.
    A handful of -10 bp fills shouldn't trigger if min_fills=5 and
    we have 3."""
    t = _tracker(min_fills=5)
    for _ in range(3):
        t.note_fill(residual_bps=-10.0, now_mono=1000.0)
    assert t.current_mean_bps() is None
    assert t.is_armed(now_mono=1000.0) is False


def test_current_mean_computed_at_min_fills():
    t = _tracker(min_fills=5)
    for _ in range(5):
        t.note_fill(residual_bps=-2.0, now_mono=1000.0)
    assert t.current_mean_bps() == pytest.approx(-2.0)


def test_window_caps_buffer_size():
    """``window_fills=3`` -> only the most recent 3 samples retained."""
    t = _tracker(window_fills=3, min_fills=2)
    t.note_fill(residual_bps=10.0, now_mono=1000.0)  # would drop
    t.note_fill(residual_bps=-1.0, now_mono=1001.0)
    t.note_fill(residual_bps=-1.0, now_mono=1002.0)
    t.note_fill(residual_bps=-1.0, now_mono=1003.0)
    # The +10 sample is gone; mean of [-1, -1, -1] = -1.
    assert t.current_mean_bps() == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# ResidualDecayTracker -- dwell timer + is_armed
# ---------------------------------------------------------------------------


def test_not_armed_when_dwell_not_elapsed():
    """Mean is below threshold but dwell hasn't elapsed yet."""
    t = _tracker(
        threshold_bps=-0.5,
        dwell_seconds=60.0,
        min_fills=5,
    )
    # Cross below threshold at t=1000.
    for i in range(5):
        t.note_fill(residual_bps=-2.0, now_mono=1000.0 + i)
    # At t=1030 we're below threshold (mean=-2) but only 30s of dwell.
    assert t.is_armed(now_mono=1030.0) is False


def test_armed_when_dwell_elapsed():
    """Mean below threshold for >= dwell_seconds -> armed.

    The dwell starts at the FIRST below-threshold sample (which is
    the 5th note_fill, after min_fills=5 is satisfied). The 5th
    fill is at t=1004, so the dwell clock starts at 1004. To be
    armed, need now_mono >= 1004 + 60 = 1064.
    """
    t = _tracker(
        threshold_bps=-0.5,
        dwell_seconds=60.0,
        min_fills=5,
    )
    # Establish the "below" state. min_fills=5; first below at t=1004.
    for i in range(5):
        t.note_fill(residual_bps=-2.0, now_mono=1000.0 + i)
    # At t=1063 we're 1 second short of the 60s dwell.
    assert t.is_armed(now_mono=1063.0) is False
    # At t=1065 we're past it.
    assert t.is_armed(now_mono=1065.0) is True


def test_recovery_resets_dwell():
    """A sample that lifts the mean above threshold resets the dwell
    timer. Subsequent below-threshold samples must accumulate dwell
    afresh.

    Uses a small window (window_fills=3) so the recovery sample
    actually clears the buffer of the prior below-threshold samples
    -- otherwise a +20 outlier on a 50-sample buffer still leaves
    the average below threshold and the dwell never genuinely
    resets.
    """
    t = _tracker(
        threshold_bps=-0.5,
        dwell_seconds=60.0,
        window_fills=3,
        min_fills=3,
    )
    # 3 fills below at t=1000..1002. Mean=-2 at t=1002 (first time
    # we hit min_fills). below_since_mono = 1002.
    for i in range(3):
        t.note_fill(residual_bps=-2.0, now_mono=1000.0 + i)
    assert t.current_mean_bps() == pytest.approx(-2.0)
    # 30s past below-start, dwell not elapsed.
    assert t.is_armed(now_mono=1032.0) is False
    # Recovery: 3 strong-positive samples (window_fills=3 so these
    # fully replace the prior below-threshold samples).
    for i in range(3):
        t.note_fill(residual_bps=10.0, now_mono=1033.0 + i)
    # Mean=10 > -0.5 -> dwell reset.
    assert t.current_mean_bps() == pytest.approx(10.0)
    assert t.is_armed(now_mono=1036.0) is False
    # New below-threshold samples start a FRESH dwell. The 3rd
    # below-fill (t=1039) is the first sample where mean drops back
    # below threshold (since the buffer also rotates out the +10s).
    for i in range(3):
        t.note_fill(residual_bps=-5.0, now_mono=1037.0 + i)
    assert t.current_mean_bps() == pytest.approx(-5.0)
    # Fresh dwell from t=1039. Armed at t=1039 + 60 = 1099.
    assert t.is_armed(now_mono=1098.0) is False
    assert t.is_armed(now_mono=1100.0) is True


def test_dwell_zero_arms_immediately():
    """``dwell_seconds=0`` -> arm on the FIRST below-threshold mean."""
    t = _tracker(
        threshold_bps=-0.5,
        dwell_seconds=0.0,
        min_fills=3,
    )
    for i in range(3):
        t.note_fill(residual_bps=-2.0, now_mono=1000.0 + i)
    assert t.is_armed(now_mono=1003.0) is True


def test_none_residual_skipped():
    """``residual_bps=None`` (e.g. markout not resolved) silently
    skips -- doesn't pollute the buffer."""
    t = _tracker(min_fills=3)
    for _ in range(3):
        t.note_fill(residual_bps=None, now_mono=1000.0)
    assert t.current_mean_bps() is None
    # Buffer is empty so we're not armed.
    assert t.is_armed(now_mono=1500.0) is False


def test_consume_arm_bumps_fire_count():
    """Splitting the predicate (``is_armed``) from the side-effect
    (``consume_arm``) lets the caller poll without inflating
    counters; only the actual arming bumps fire_count."""
    t = _tracker(
        threshold_bps=-0.5,
        dwell_seconds=0.0,
        min_fills=3,
    )
    for i in range(3):
        t.note_fill(residual_bps=-2.0, now_mono=1000.0 + i)
    assert t.fire_count == 0  # predicate poll didn't bump
    assert t.is_armed(now_mono=1003.0) is True
    assert t.fire_count == 0  # still zero
    t.consume_arm(now_mono=1003.0)
    assert t.fire_count == 1
    assert t.last_trigger_mean_bps == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# snapshot_dict for operator visibility
# ---------------------------------------------------------------------------


def test_snapshot_dict_shape():
    t = _tracker(
        threshold_bps=-0.5,
        dwell_seconds=60.0,
        window_fills=50,
        min_fills=5,
    )
    snap = t.snapshot_dict()
    assert snap["enabled"] is True
    assert snap["threshold_bps"] == -0.5
    assert snap["dwell_seconds"] == 60.0
    assert snap["min_fills"] == 5
    assert snap["window_fills"] == 50
    assert snap["buffer_size"] == 0
    assert snap["current_mean_bps"] is None
    assert snap["fire_count"] == 0
    assert snap["last_trigger_mean_bps"] == 0.0


def test_snapshot_dict_after_fills():
    t = _tracker(threshold_bps=-0.5, dwell_seconds=60.0, min_fills=3)
    for i in range(5):
        t.note_fill(residual_bps=-1.0, now_mono=1000.0 + i)
    snap = t.snapshot_dict()
    assert snap["buffer_size"] == 5
    assert snap["current_mean_bps"] == pytest.approx(-1.0)
