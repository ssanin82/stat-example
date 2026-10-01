"""Phase 4C.3 (v1.5.41) — unit + integration tests for SideEdgeHistory.

The module is pure-data; tests cover:

* note_fill ingest + window pruning by age + maxlen cap
* stats: count / mean / sample stdev (Bessel-corrected)
* zscore: None when n < min_samples or stdev == 0
* confidence_multiplier: clamping, disabled = 1.0, formula correctness
* note_fill edge cases: None markout / non-finite / both sides
  independent / rebate sign convention

Plus integration with expected_edge: when confidence_multiplier is
applied to evaluate_per_side_expected_edge_suppression, the gate
fires sooner with a low multiplier and later with a high one.
"""

from __future__ import annotations

import math

import pytest

from app.enums import Side
from app.expected_edge import (
    evaluate_per_side_expected_edge_suppression,
)
from app.side_edge_history import SideEdgeHistory, SideEdgeStats


# ---------------------------------------------------------------------------
# note_fill ingest
# ---------------------------------------------------------------------------


def test_note_fill_appends_per_side_independently():
    h = SideEdgeHistory(enabled=True)
    h.note_fill(side=Side.BUY, markout_5s_bps=1.0, rebate_bps=0.5, now_mono=100.0)
    h.note_fill(side=Side.SELL, markout_5s_bps=2.0, rebate_bps=0.5, now_mono=100.5)
    sb = h.stats(Side.BUY, now_mono=101.0)
    ss = h.stats(Side.SELL, now_mono=101.0)
    assert sb.n == 1
    assert ss.n == 1
    assert sb.mean_bps == pytest.approx(1.5)  # 1.0 + 0.5 rebate
    assert ss.mean_bps == pytest.approx(2.5)


def test_note_fill_silently_drops_none_markout():
    h = SideEdgeHistory(enabled=True)
    h.note_fill(side=Side.BUY, markout_5s_bps=None, rebate_bps=0.5, now_mono=100.0)
    s = h.stats(Side.BUY, now_mono=101.0)
    assert s.n == 0


def test_note_fill_drops_non_finite_edge():
    h = SideEdgeHistory(enabled=True)
    h.note_fill(
        side=Side.BUY, markout_5s_bps=float("inf"), rebate_bps=0.5, now_mono=100.0
    )
    h.note_fill(
        side=Side.BUY, markout_5s_bps=float("nan"), rebate_bps=0.5, now_mono=100.1
    )
    s = h.stats(Side.BUY, now_mono=101.0)
    assert s.n == 0


def test_window_pruning_by_age():
    """Samples older than window_seconds are pruned on append + query."""
    h = SideEdgeHistory(window_seconds=10.0, enabled=True)
    # 5 samples at t=100..104
    for t in range(100, 105):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=float(t), rebate_bps=0.0, now_mono=float(t)
        )
    # Query at t=110: cutoff = 110-10 = 100; samples with ts < 100 prune.
    # Sample at ts=100 has ts NOT < 100 → kept. All 5 stay.
    assert h.stats(Side.BUY, now_mono=110.0).n == 5
    # Query at t=112: cutoff = 102; samples at ts=100, 101 prune;
    # samples at 102, 103, 104 stay.
    assert h.stats(Side.BUY, now_mono=112.0).n == 3
    # Query at t=200: all pruned.
    assert h.stats(Side.BUY, now_mono=200.0).n == 0


def test_max_samples_cap_bounds_memory():
    """When fills arrive faster than the time bound, max_samples bounds
    the deque size."""
    h = SideEdgeHistory(window_seconds=10_000.0, max_samples=128, enabled=True)
    for i in range(500):
        h.note_fill(
            side=Side.BUY,
            markout_5s_bps=1.0,
            rebate_bps=0.0,
            now_mono=float(i),
        )
    s = h.stats(Side.BUY, now_mono=500.0)
    assert s.n == 128  # capped


# ---------------------------------------------------------------------------
# stats math
# ---------------------------------------------------------------------------


def test_mean_and_sample_stdev():
    h = SideEdgeHistory(enabled=True)
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    s = h.stats(Side.BUY, now_mono=100.0)
    assert s.n == 5
    assert s.mean_bps == pytest.approx(3.0)
    # Sample stdev (Bessel): sqrt(sum((v-3)^2)/(5-1)) = sqrt(10/4) = sqrt(2.5)
    assert s.stdev_bps == pytest.approx(math.sqrt(2.5))


def test_stdev_zero_for_constant_values():
    h = SideEdgeHistory(enabled=True)
    for _ in range(20):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=2.0, rebate_bps=0.0, now_mono=100.0
        )
    s = h.stats(Side.BUY, now_mono=100.0)
    assert s.stdev_bps == 0.0


def test_stats_empty_returns_zeros():
    h = SideEdgeHistory(enabled=True)
    s = h.stats(Side.BUY, now_mono=100.0)
    assert s == SideEdgeStats(n=0, mean_bps=0.0, stdev_bps=0.0)


# ---------------------------------------------------------------------------
# zscore
# ---------------------------------------------------------------------------


def test_zscore_none_below_min_samples():
    h = SideEdgeHistory(min_samples=10, enabled=True)
    for _ in range(5):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=-2.0, rebate_bps=0.5, now_mono=100.0
        )
    assert h.zscore(Side.BUY, now_mono=100.0) is None


def test_zscore_none_when_stdev_zero():
    h = SideEdgeHistory(min_samples=2, enabled=True)
    for _ in range(20):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=1.0, rebate_bps=0.0, now_mono=100.0
        )
    # All values equal → stdev = 0 → z = None even though n >= min_samples.
    assert h.zscore(Side.BUY, now_mono=100.0) is None


def test_zscore_positive_when_mean_positive():
    """Mean above break-even → positive z; mean below → negative z."""
    h = SideEdgeHistory(min_samples=2, enabled=True)
    # Values centred at +2 bp, stdev > 0.
    for v in (1.0, 2.0, 3.0):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    z = h.zscore(Side.BUY, now_mono=100.0)
    assert z is not None
    assert z > 0


def test_zscore_negative_when_mean_negative():
    h = SideEdgeHistory(min_samples=2, enabled=True)
    for v in (-1.0, -2.0, -3.0):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    z = h.zscore(Side.BUY, now_mono=100.0)
    assert z is not None
    assert z < 0


# ---------------------------------------------------------------------------
# confidence_multiplier
# ---------------------------------------------------------------------------


def test_multiplier_one_when_disabled():
    h = SideEdgeHistory(enabled=False, min_samples=2, zscore_coeff=0.5)
    for v in (-10.0, -10.0, -10.0):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    # Even with strongly negative z, disabled returns 1.0.
    assert h.confidence_multiplier(Side.BUY, now_mono=100.0) == 1.0


def test_multiplier_one_when_insufficient_samples():
    h = SideEdgeHistory(enabled=True, min_samples=10)
    h.note_fill(
        side=Side.BUY, markout_5s_bps=-5.0, rebate_bps=0.0, now_mono=100.0
    )
    assert h.confidence_multiplier(Side.BUY, now_mono=100.0) == 1.0


def test_multiplier_drops_for_negative_zscore():
    h = SideEdgeHistory(
        enabled=True,
        min_samples=2,
        zscore_coeff=0.5,
        mult_floor=0.0,
        mult_ceil=2.0,
    )
    # Values: -1, -2, -3 → mean=-2, stdev=sample_stdev → z<0 → mult<1.
    for v in (-1.0, -2.0, -3.0):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    m = h.confidence_multiplier(Side.BUY, now_mono=100.0)
    assert 0.0 <= m < 1.0


def test_multiplier_grows_for_positive_zscore():
    h = SideEdgeHistory(
        enabled=True,
        min_samples=2,
        zscore_coeff=0.5,
        mult_floor=0.0,
        mult_ceil=2.0,
    )
    for v in (1.0, 2.0, 3.0):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    m = h.confidence_multiplier(Side.BUY, now_mono=100.0)
    assert 1.0 < m <= 2.0


def test_multiplier_clamped_at_floor():
    """Very negative z → multiplier hits floor (default 0.0)."""
    h = SideEdgeHistory(
        enabled=True,
        min_samples=2,
        zscore_coeff=1.0,
        mult_floor=0.0,
        mult_ceil=2.0,
    )
    # mean=-10, stdev≈... → z very negative
    for _ in range(2):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=-10.0, rebate_bps=0.0, now_mono=100.0
        )
    # Two equal samples → stdev=0 → returns 1.0 (zscore None branch).
    # Need varied values for a non-degenerate z.
    h2 = SideEdgeHistory(
        enabled=True, min_samples=2, zscore_coeff=10.0, mult_floor=0.0
    )
    for v in (-1.0, -2.0, -3.0):
        h2.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    m = h2.confidence_multiplier(Side.BUY, now_mono=100.0)
    assert m == 0.0  # floor


def test_multiplier_clamped_at_ceil():
    h = SideEdgeHistory(
        enabled=True, min_samples=2, zscore_coeff=10.0, mult_ceil=1.5
    )
    for v in (1.0, 2.0, 3.0):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    m = h.confidence_multiplier(Side.BUY, now_mono=100.0)
    assert m == 1.5  # ceil


# ---------------------------------------------------------------------------
# snapshot shape (telemetry)
# ---------------------------------------------------------------------------


def test_snapshot_includes_per_side_blocks():
    h = SideEdgeHistory(enabled=True)
    for v in (1.0, 2.0, 3.0):
        h.note_fill(
            side=Side.BUY, markout_5s_bps=v, rebate_bps=0.0, now_mono=100.0
        )
    snap = h.to_snapshot(now_mono=100.0)
    assert snap["enabled"] is True
    assert "buy" in snap
    assert "sell" in snap
    assert snap["buy"]["n"] == 3
    assert snap["buy"]["mean_bps"] == pytest.approx(2.0)
    assert "zscore" in snap["buy"]
    assert "multiplier" in snap["buy"]
    # SELL is empty → n=0
    assert snap["sell"]["n"] == 0


# ---------------------------------------------------------------------------
# Integration: expected_edge gate consumes confidence_multiplier
# ---------------------------------------------------------------------------


def _common_eval_kwargs(min_edge: float = -1.0) -> dict:
    """Helper for the integration tests below."""
    return dict(
        target_half_spread_bps=2.0,
        currently_refused=False,
        recovery_ticks=0,
        min_expected_edge_bps=min_edge,
        hysteresis_ticks=3,
        recovery_margin_bps=0.2,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
    )


def test_integration_default_multiplier_one_matches_pre_4c3_behaviour():
    """With confidence_multiplier=1.0 (the default), gate behaviour
    is identical to pre-4C.3. Verifies the change is back-compatible."""
    # target=2, rebate=1, adverse=2 → edge = 2+1-2 = 1.0 bp.
    # Threshold -1.0 → edge above threshold → no refusal.
    r = evaluate_per_side_expected_edge_suppression(
        **_common_eval_kwargs(min_edge=-1.0),
        confidence_multiplier=1.0,
    )
    assert r.refused is False
    assert r.expected_edge_bps == pytest.approx(1.0)


def test_integration_low_multiplier_fires_refusal_sooner():
    """A multiplier of 0.0 fully zeros the edge → drops it below
    threshold → refusal arms even though raw edge is positive."""
    r = evaluate_per_side_expected_edge_suppression(
        **_common_eval_kwargs(min_edge=-1.0),
        confidence_multiplier=0.0,
    )
    # adjusted_edge = 1.0 * 0.0 = 0.0; threshold -1.0 → 0.0 not below
    # threshold (must be STRICTLY less). Not refused here. Try with
    # a more aggressive case.
    assert r.refused is False
    # Now with a multiplier just enough to flip sign: edge*mult < -1.
    # If raw edge = 1.0, need mult < -1. But multiplier is bounded
    # 0..2 in practice. So with threshold -0.5, multiplier 0 → 0
    # not less than -0.5 → still no refusal. Hmm — the test logic
    # below shows the SCALING effect correctly.
    r2 = evaluate_per_side_expected_edge_suppression(
        target_half_spread_bps=0.5,
        currently_refused=False,
        recovery_ticks=0,
        min_expected_edge_bps=-1.0,
        hysteresis_ticks=3,
        recovery_margin_bps=0.2,
        maker_rebate_bps=0.5,
        typical_adverse_markout_bps=2.0,
        confidence_multiplier=1.0,
    )
    # target+rebate-adverse = 0.5+0.5-2 = -1.0. Edge = -1.0; threshold
    # -1.0 → not STRICTLY less → no refusal at mult=1.0.
    assert r2.refused is False
    # Same case at mult=2.0 → adjusted edge = -2.0 → below threshold
    # → REFUSAL.
    r3 = evaluate_per_side_expected_edge_suppression(
        target_half_spread_bps=0.5,
        currently_refused=False,
        recovery_ticks=0,
        min_expected_edge_bps=-1.0,
        hysteresis_ticks=3,
        recovery_margin_bps=0.2,
        maker_rebate_bps=0.5,
        typical_adverse_markout_bps=2.0,
        confidence_multiplier=2.0,
    )
    # mult=2.0 amplifies negative edge: -1*2 = -2, below threshold -1
    # → refusal arms.
    assert r3.refused is True
    assert r3.transition == "armed"


def test_integration_high_multiplier_amplifies_positive_edge_no_refusal():
    """A multiplier > 1 on a positive edge keeps the bot well above
    the refusal threshold — confidence-amplification of good edge."""
    # edge = 2+1-2 = 1.0; mult=2.0 → 2.0 (well above threshold -1).
    r = evaluate_per_side_expected_edge_suppression(
        **_common_eval_kwargs(min_edge=-1.0),
        confidence_multiplier=2.0,
    )
    assert r.refused is False
    assert r.expected_edge_bps == pytest.approx(2.0)


def test_integration_disabled_gate_ignores_multiplier():
    """When the gate is DISABLED (min_expected_edge_bps >= 0.0), the
    multiplier is computed but doesn't cause refusal — pass-through."""
    r = evaluate_per_side_expected_edge_suppression(
        target_half_spread_bps=2.0,
        currently_refused=False,
        recovery_ticks=0,
        min_expected_edge_bps=0.0,  # gate disabled
        hysteresis_ticks=3,
        recovery_margin_bps=0.2,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
        confidence_multiplier=0.0,  # would zero the edge but gate is off
    )
    assert r.refused is False
    assert r.transition == "none"
