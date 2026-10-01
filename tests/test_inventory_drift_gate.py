"""v1.4.106 Phase 1A — inventory_drift_gate unit tests.

Pin the gate's contract:

1. Disabled → no-op.
2. Util below threshold (low inventory) → no-op even if drift is huge.
3. Drift below thresholds (both windows) → no-op.
4. Aligned: LONG + drift UP → no-op (we're on the right side).
5. Aligned: SHORT + drift DOWN → no-op.
6. Anti-aligned: LONG + drift DOWN beyond 10s threshold →
   QUOTE_SELL_ONLY override + widen BID side.
7. Anti-aligned: SHORT + drift UP beyond 10s threshold →
   QUOTE_BUY_ONLY override + widen ASK side.
8. 30s window alone trips threshold → gate still fires.
9. Both windows trip → gate uses the larger-|drift| window.
10. Sentinel widen_bps=-1.0 → contribution = MAX_HALF_SPREAD_BPS.
11. widen_bps above cap → clamped to cap.
12. Drift inputs both None (cold start / empty deque) → no-op.
13. compute_short_window_drifts helper: empty deque → (None, None);
    populated deque → correctly anchored drifts.

The Codex Section 6 acceptance scenario (snapshot 260520-074215):
util=0.6, 30s drift -25 bp → gate must fire. Pre-06:00 util ≈ 0 →
gate must NOT fire even if same drift hits.
"""

from __future__ import annotations

import math

from app.enums import QuoteEligibility
from app.inventory_drift_gate import (
    compute_short_window_drifts,
    evaluate_inventory_drift_gate,
    widening_bps,
)


# ---------------------------------------------------------------------------
# Disabled / dormant cases
# ---------------------------------------------------------------------------


def test_disabled_returns_no_op() -> None:
    override, reason, drift = evaluate_inventory_drift_gate(
        position_qty=10.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-30.0,
        drift_bps_30s=-50.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=False,
    )
    assert override is None
    assert reason == "inventory_drift_disabled"
    assert drift is None


def test_util_below_threshold_returns_no_op() -> None:
    """Bot at 30 % util — gate dormant even if drift is severe."""
    override, reason, _ = evaluate_inventory_drift_gate(
        position_qty=3.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-30.0,
        drift_bps_30s=-50.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is None
    assert reason.startswith("inventory_drift_util_below:")


def test_drift_below_threshold_returns_no_op() -> None:
    """Util high enough; drift inside threshold band on both windows."""
    override, reason, _ = evaluate_inventory_drift_gate(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-5.0,
        drift_bps_30s=-10.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is None
    assert reason == "inventory_drift_ok"


# ---------------------------------------------------------------------------
# Aligned cases — no-op even when threshold cleared
# ---------------------------------------------------------------------------


def test_long_aligned_with_up_drift_no_op() -> None:
    """LONG + drift UP → bot is on the right side. No-op."""
    override, reason, _ = evaluate_inventory_drift_gate(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=+25.0,
        drift_bps_30s=+40.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is None
    assert reason == "inventory_drift_ok"


def test_short_aligned_with_down_drift_no_op() -> None:
    """SHORT + drift DOWN → bot is on the right side. No-op."""
    override, reason, _ = evaluate_inventory_drift_gate(
        position_qty=-8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-25.0,
        drift_bps_30s=-40.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is None
    assert reason == "inventory_drift_ok"


# ---------------------------------------------------------------------------
# Anti-aligned fires — eligibility + telemetry
# ---------------------------------------------------------------------------


def test_long_anti_aligned_10s_fires() -> None:
    """LONG + 10s drift DOWN beyond threshold → SELL_ONLY + bid widen."""
    override, reason, drift = evaluate_inventory_drift_gate(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-20.0,
        drift_bps_30s=-5.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert "down" in reason
    assert "10s" in reason
    assert drift == -20.0


def test_short_anti_aligned_10s_fires() -> None:
    """SHORT + 10s drift UP beyond threshold → BUY_ONLY + ask widen."""
    override, reason, drift = evaluate_inventory_drift_gate(
        position_qty=-8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=+20.0,
        drift_bps_30s=+5.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is QuoteEligibility.QUOTE_BUY_ONLY
    assert "up" in reason
    assert "10s" in reason
    assert drift == +20.0


def test_30s_only_threshold_clears_fires() -> None:
    """10s drift below its threshold but 30s drift clears 30s threshold."""
    override, reason, drift = evaluate_inventory_drift_gate(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-10.0,  # below 15.0 → 10s dormant
        drift_bps_30s=-35.0,  # above 30.0 → 30s fires
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert "30s" in reason
    assert drift == -35.0


def test_both_windows_fire_picks_larger_drift() -> None:
    """When both windows fire the gate selects whichever has larger |drift|
    (more informative signal in the telemetry)."""
    override, reason, drift = evaluate_inventory_drift_gate(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-20.0,  # clears 15.0
        drift_bps_30s=-40.0,  # clears 30.0, larger |drift|
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert "30s" in reason
    assert drift == -40.0


def test_cold_start_both_drifts_none_no_op() -> None:
    """Cold start — bot's mid deque is empty so both drift inputs are
    None. Gate stays dormant."""
    override, reason, drift = evaluate_inventory_drift_gate(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=None,
        drift_bps_30s=None,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
    )
    assert override is None
    assert reason == "inventory_drift_ok"
    assert drift is None


# ---------------------------------------------------------------------------
# Widening contribution
# ---------------------------------------------------------------------------


def test_widening_dormant_returns_zero_both_sides() -> None:
    bid, ask = widening_bps(
        position_qty=3.0,  # util 0.30 → below threshold
        effective_abs_cap=10.0,
        drift_bps_10s=-30.0,
        drift_bps_30s=-50.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
        max_half_spread_bps=100.0,
        widen_bps=15.0,
    )
    assert bid == 0.0
    assert ask == 0.0


def test_widening_long_anti_aligned_widens_bid_only() -> None:
    """LONG + drift DOWN → widen BID, leave ASK."""
    bid, ask = widening_bps(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-20.0,
        drift_bps_30s=-5.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
        max_half_spread_bps=100.0,
        widen_bps=15.0,
    )
    assert bid == 15.0
    assert ask == 0.0


def test_widening_short_anti_aligned_widens_ask_only() -> None:
    """SHORT + drift UP → widen ASK, leave BID."""
    bid, ask = widening_bps(
        position_qty=-8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=+20.0,
        drift_bps_30s=+5.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
        max_half_spread_bps=100.0,
        widen_bps=15.0,
    )
    assert bid == 0.0
    assert ask == 15.0


def test_widening_sentinel_falls_back_to_cap() -> None:
    """widen_bps=-1.0 → contribution = max_half_spread_bps (dark)."""
    bid, ask = widening_bps(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-20.0,
        drift_bps_30s=-5.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
        max_half_spread_bps=100.0,
        widen_bps=-1.0,
    )
    assert bid == 100.0
    assert ask == 0.0


def test_widening_above_cap_clamped() -> None:
    """widen_bps above max_half_spread_bps → clamped to cap."""
    bid, ask = widening_bps(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-20.0,
        drift_bps_30s=-5.0,
        inventory_pct_threshold=0.6,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=30.0,
        enabled=True,
        max_half_spread_bps=20.0,
        widen_bps=50.0,
    )
    assert bid == 20.0
    assert ask == 0.0


# ---------------------------------------------------------------------------
# compute_short_window_drifts helper
# ---------------------------------------------------------------------------


def test_compute_drifts_empty_deque_returns_none() -> None:
    d10, d30 = compute_short_window_drifts(
        [],
        now_mono=1000.0,
        mid_now=2.0,
    )
    assert d10 is None
    assert d30 is None


def test_compute_drifts_basic_anchors() -> None:
    """Build a deque with anchors at 30s, 10s, 0s.

    Samples: (970.0, 2.0), (990.0, 1.99), (1000.0, 1.98)
    mid_now = 1.97 at t=1000.0
    Expected:
        10s drift: anchor at oldest sample with ts >= 990.0 → 1.99
                   drift = (1.97 / 1.99 - 1) * 1e4 ≈ -100 bps
        30s drift: anchor at oldest sample with ts >= 970.0 → 2.00
                   drift = (1.97 / 2.00 - 1) * 1e4 ≈ -150 bps
    """
    samples = [(970.0, 2.00), (990.0, 1.99), (1000.0, 1.98)]
    d10, d30 = compute_short_window_drifts(
        samples,
        now_mono=1000.0,
        mid_now=1.97,
    )
    assert d10 is not None
    assert d30 is not None
    assert math.isclose(d10, -100.502512, rel_tol=1e-4)
    assert math.isclose(d30, -150.0, rel_tol=1e-4)


def test_compute_drifts_no_sample_in_window_returns_none() -> None:
    """All samples too old for the 10s window."""
    samples = [(900.0, 2.00), (950.0, 1.99)]
    d10, d30 = compute_short_window_drifts(
        samples,
        now_mono=1000.0,
        mid_now=1.97,
    )
    assert d10 is None  # nothing within 10s of now
    assert d30 is None  # nothing within 30s of now


# ---------------------------------------------------------------------------
# Replay acceptance — Codex Section 6 reference scenario
# ---------------------------------------------------------------------------


def test_replay_06_35_scenario_fires() -> None:
    """At ~06:35 in the 260520-074215 snapshot:
    util ≈ 0.6 (position +6 / cap +10), 30s drift ≈ -25 bps.
    Gate MUST fire under the TON profile thresholds."""
    override, reason, drift = evaluate_inventory_drift_gate(
        position_qty=+6.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-8.0,    # below 15.0 threshold — 10s dormant
        drift_bps_30s=-25.0,   # 30s window has accumulated the grind
        inventory_pct_threshold=0.60,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=20.0,  # tightened slightly per replay
        enabled=True,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert drift == -25.0
    assert "30s" in reason


def test_replay_pre_06_dormant() -> None:
    """Before 06:00 the bot is roughly flat. Even if a similar
    drift signature lands, the gate must NOT fire — that's the
    point of the position-aware design."""
    override, reason, _ = evaluate_inventory_drift_gate(
        position_qty=+0.5,  # near-flat
        effective_abs_cap=10.0,
        drift_bps_10s=-20.0,
        drift_bps_30s=-30.0,
        inventory_pct_threshold=0.60,
        drift_threshold_bps_10s=15.0,
        drift_threshold_bps_30s=20.0,
        enabled=True,
    )
    assert override is None
    assert reason.startswith("inventory_drift_util_below:")
