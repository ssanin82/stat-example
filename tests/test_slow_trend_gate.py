"""v1.4.102 — slow_trend_gate unit tests.

Pin the gate's contract:
1. Disabled (flag off OR threshold=0) → no-op.
2. Insufficient samples (cold start) → warmup, no-op.
3. Sustained DOWN drift beyond threshold → QUOTE_SELL_ONLY override +
   widen on BID side.
4. Sustained UP drift beyond threshold → QUOTE_BUY_ONLY override +
   widen on ASK side.
5. Drift within threshold → no-op (QUOTE_BOTH preserved).
6. Sentinel widen_bps=-1.0 → contribution = MAX_HALF_SPREAD_BPS
   (gate-equivalent dark).
7. Non-sentinel widen_bps → contribution = widen_bps (continuous).
8. Widen_bps above cap → clamped to MAX_HALF_SPREAD_BPS.

The gate's anchored-median logic mirrors the existing
``_long_drift_eligibility`` gate in ``quote_eligibility.py`` (same
shape, different window/threshold), so behaviour at boundary cases
(empty deque, single sample, anchor_fraction=0) is also exercised.
"""

from __future__ import annotations

import math

from app.enums import QuoteEligibility
from app.slow_trend_gate import evaluate_slow_trend_gate, widening_bps


def _synth_samples(
    *,
    n_samples: int,
    start_mid: float,
    end_mid: float,
    duration_s: float,
    now_mono: float,
) -> list[tuple[float, float]]:
    """Build a synthetic deque of (ts_mono, mid) pairs that linearly
    drift from ``start_mid`` to ``end_mid`` over ``duration_s``."""
    if n_samples < 2:
        return [(now_mono - duration_s, start_mid)]
    samples = []
    for i in range(n_samples):
        frac = i / (n_samples - 1)
        ts = now_mono - duration_s + frac * duration_s
        mid = start_mid + (end_mid - start_mid) * frac
        samples.append((ts, mid))
    return samples


# ---------------------------------------------------------------------------
# Disabled / warmup cases
# ---------------------------------------------------------------------------


def test_disabled_returns_no_op() -> None:
    """Disabled flag → None override, reason="slow_trend_disabled"."""
    override, reason, drift = evaluate_slow_trend_gate(
        samples_long=_synth_samples(
            n_samples=100, start_mid=2.0, end_mid=1.9,
            duration_s=900.0, now_mono=10000.0,
        ),
        now_mono=10000.0,
        mid_now=1.9,
        enabled=False,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
    )
    assert override is None
    assert reason == "slow_trend_disabled"
    assert drift is None


def test_zero_threshold_returns_no_op() -> None:
    """threshold_bps=0 → disabled (same path as flag-off)."""
    override, reason, drift = evaluate_slow_trend_gate(
        samples_long=_synth_samples(
            n_samples=100, start_mid=2.0, end_mid=1.9,
            duration_s=900.0, now_mono=10000.0,
        ),
        now_mono=10000.0,
        mid_now=1.9,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=0.0,
        min_samples=60,
        anchor_fraction=0.2,
    )
    assert override is None
    assert reason == "slow_trend_disabled"


def test_insufficient_samples_returns_warmup() -> None:
    """Below min_samples → warmup, no-op."""
    override, reason, drift = evaluate_slow_trend_gate(
        samples_long=_synth_samples(
            n_samples=10, start_mid=2.0, end_mid=1.9,
            duration_s=900.0, now_mono=10000.0,
        ),
        now_mono=10000.0,
        mid_now=1.9,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
    )
    assert override is None
    assert reason == "slow_trend_warmup"
    assert drift is None


def test_invalid_mid_returns_no_op() -> None:
    override, reason, _ = evaluate_slow_trend_gate(
        samples_long=_synth_samples(
            n_samples=100, start_mid=2.0, end_mid=1.9,
            duration_s=900.0, now_mono=10000.0,
        ),
        now_mono=10000.0,
        mid_now=0.0,  # invalid
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
    )
    assert override is None
    assert reason == "slow_trend_no_mid"


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_sustained_down_drift_fires_sell_only() -> None:
    """TON-style: mid drifts 1.985 → 1.961 over 15 min, threshold 25 bp.
    Expected drift = -(1 - 1.961/1.985) × 1e4 = -120.9 bp → fires SELL_ONLY."""
    samples = _synth_samples(
        n_samples=100, start_mid=1.985, end_mid=1.961,
        duration_s=900.0, now_mono=10000.0,
    )
    override, reason, drift = evaluate_slow_trend_gate(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=1.961,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
    )
    assert override == QuoteEligibility.QUOTE_SELL_ONLY
    assert "slow_trend_down" in reason
    assert drift is not None
    assert drift < -25.0  # Below threshold


def test_sustained_up_drift_fires_buy_only() -> None:
    samples = _synth_samples(
        n_samples=100, start_mid=2.0, end_mid=2.020,
        duration_s=900.0, now_mono=10000.0,
    )
    override, reason, drift = evaluate_slow_trend_gate(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=2.020,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
    )
    assert override == QuoteEligibility.QUOTE_BUY_ONLY
    assert "slow_trend_up" in reason
    assert drift is not None
    assert drift > 25.0


def test_drift_within_threshold_returns_ok() -> None:
    """5 bp drift, threshold 25 → no-op but drift surfaced in result."""
    samples = _synth_samples(
        n_samples=100, start_mid=2.000, end_mid=1.999,
        duration_s=900.0, now_mono=10000.0,
    )
    override, reason, drift = evaluate_slow_trend_gate(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=1.999,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
    )
    assert override is None
    assert reason == "slow_trend_ok"
    assert drift is not None
    assert abs(drift) < 25.0


# ---------------------------------------------------------------------------
# Widening contribution
# ---------------------------------------------------------------------------


def test_widening_bps_sentinel_returns_max() -> None:
    """widen_bps=-1.0 → falls back to max_half_spread_bps (dark)."""
    samples = _synth_samples(
        n_samples=100, start_mid=1.985, end_mid=1.961,
        duration_s=900.0, now_mono=10000.0,
    )
    bid_bps, ask_bps = widening_bps(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=1.961,
        max_half_spread_bps=30.0,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
        widen_bps=-1.0,
    )
    # DOWN drift → bid widened, ask zero.
    assert bid_bps == 30.0
    assert ask_bps == 0.0


def test_widening_bps_non_sentinel_returns_coefficient() -> None:
    """widen_bps=10 → contributes exactly 10 bp (continuous live mode)."""
    samples = _synth_samples(
        n_samples=100, start_mid=1.985, end_mid=1.961,
        duration_s=900.0, now_mono=10000.0,
    )
    bid_bps, ask_bps = widening_bps(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=1.961,
        max_half_spread_bps=30.0,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
        widen_bps=10.0,
    )
    assert bid_bps == 10.0
    assert ask_bps == 0.0


def test_widening_bps_above_cap_clamped() -> None:
    """widen_bps > max → clamps to max."""
    samples = _synth_samples(
        n_samples=100, start_mid=1.985, end_mid=1.961,
        duration_s=900.0, now_mono=10000.0,
    )
    bid_bps, ask_bps = widening_bps(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=1.961,
        max_half_spread_bps=30.0,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
        widen_bps=999.0,  # absurd
    )
    assert bid_bps == 30.0
    assert ask_bps == 0.0


def test_widening_bps_up_drift_widens_ask() -> None:
    """Asymmetric: up-drift → ASK suppressed → ask widened, bid zero."""
    samples = _synth_samples(
        n_samples=100, start_mid=2.0, end_mid=2.020,
        duration_s=900.0, now_mono=10000.0,
    )
    bid_bps, ask_bps = widening_bps(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=2.020,
        max_half_spread_bps=30.0,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
        widen_bps=10.0,
    )
    assert bid_bps == 0.0
    assert ask_bps == 10.0


def test_widening_bps_inactive_returns_zero() -> None:
    """Drift within threshold → zero on both sides."""
    samples = _synth_samples(
        n_samples=100, start_mid=2.000, end_mid=1.999,
        duration_s=900.0, now_mono=10000.0,
    )
    bid_bps, ask_bps = widening_bps(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=1.999,
        max_half_spread_bps=30.0,
        enabled=True,
        window_seconds=900.0,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
        widen_bps=10.0,
    )
    assert bid_bps == 0.0
    assert ask_bps == 0.0


def test_root_cause_scenario_replay() -> None:
    """Replay of snapshot v1.4.92-260520-074215 06:00-06:48 window.

    Mid drifted from 1.985 (06:00) to 1.961 (06:48) = -121 bp over
    48 min. That's a steady grind of -2.52 bp/min.

    Sampling the gate at 06:15 (with 15-min window: 06:00 → 06:15)
    sees -37.8 bp drift across the window. The anchored-median
    smoothing (20% from each end) shrinks the effective magnitude
    to ~80% of raw drift = ~30 bp. Above the 25-bp threshold → fires.

    If this gate had been live on Sunday night, BUY would have been
    suppressed starting around 06:15 local, BEFORE the position
    grew from +3 (06:14) to +6 (06:15) to +9 (06:43). The 06:50
    drop would still have hit, but with a much smaller position.
    """
    # 06:00 → 06:15 window: linear drift at -2.52 bp/min = -37.8 bp.
    # Anchored-median sees ~30 bp drift → fires SELL_ONLY.
    drift_per_min_bps = -2.52
    duration_s = 900.0  # 15 min
    start_mid = 1.985
    # Compute end-mid from cumulative grind.
    end_mid = start_mid * (1.0 + drift_per_min_bps * 15 / 10000.0)
    samples = _synth_samples(
        n_samples=100, start_mid=start_mid, end_mid=end_mid,
        duration_s=duration_s, now_mono=10000.0,
    )
    override, reason, drift = evaluate_slow_trend_gate(
        samples_long=samples,
        now_mono=10000.0,
        mid_now=end_mid,
        enabled=True,
        window_seconds=duration_s,
        threshold_bps=25.0,
        min_samples=60,
        anchor_fraction=0.2,
    )
    assert override == QuoteEligibility.QUOTE_SELL_ONLY, (
        f"slow_trend should fire on the 06:00-06:15 window of the "
        f"slow-grind scenario; got {override} reason={reason} drift={drift}"
    )
    assert drift is not None
    assert drift < -25.0, f"expected drift below -25 bp, got {drift}"
