"""Tests for the multi-minute drift gate (BUGS/bug-002.md).

Coverage:
* disabled-when-threshold-zero
* warmup (insufficient samples)
* drift below threshold (no restriction)
* rising-trend → QUOTE_BUY_ONLY (don't sell into a rising market)
* falling-trend → QUOTE_SELL_ONLY (don't catch falling knives)
* mid_now non-finite / non-positive → no decision
* stale samples (older than window) → ignored
* integration with full ``compute_quote_eligibility`` pipeline
* regression replay against the 2026-04-25 failure case
"""

from __future__ import annotations

from typing import Any

import pytest

from app.enums import QuoteEligibility
from app.quote_eligibility import (
    _long_drift_eligibility,
    compute_quote_eligibility,
)
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides: Any) -> UnitTestSettings:
    base = {
        "EXCHANGE": "bluefin",
        "SYMBOL": "SUI-PERP",
        "BLUEFIN_PRIVATE_KEY": "00" * 32,
        "BLUEFIN_ACCOUNT_ADDRESS": "0x" + "ab" * 32,
        "QUOTE_ELIGIBILITY_ENABLED": True,
        # Generous freshness/drift defaults so they don't interfere.
        "QUOTE_HOLD_MAX_BOOK_AGE_MS": 4000.0,
        "QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS": 4000.0,
        "QUOTE_HOLD_MAX_GAP_P95_MS": 100000.0,
        "QUOTE_ONE_SIDED_MAX_GAP_P95_MS": 100000.0,
        "DRIFT_BLOCK_100MS_BPS": 1000.0,
        "DRIFT_BLOCK_250MS_BPS": 1000.0,
        "DRIFT_HOLD_500MS_BPS": 1000.0,
        "JUMP_HOLD_250MS_BPS": 1000.0,
        # Long-drift defaults.
        "DRIFT_LONG_WINDOW_SECONDS": 300.0,
        "DRIFT_BLOCK_LONG_WINDOW_BPS": 50.0,
        "DRIFT_LONG_WINDOW_MAX_SAMPLES": 512,
        "DRIFT_LONG_WINDOW_MIN_SAMPLES": 5,  # easier to test
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _samples(prices: list[float], *, dt_s: float = 3.0, base_mono: float = 100.0) -> list[tuple[float, float]]:
    """Build (mono, mid) samples evenly spaced ``dt_s`` apart.

    The first sample is at ``base_mono``; the last is at
    ``base_mono + (len-1) * dt_s``. Use ``base_mono`` so tests can
    set ``now_mono`` to a value relative to the sample window.
    """
    return [(base_mono + i * dt_s, p) for i, p in enumerate(prices)]


# --- _long_drift_eligibility ---


def test_disabled_when_threshold_zero() -> None:
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=0.0)
    samples = _samples([100.0] * 10)
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=200.0, mid_now=100.0
    )
    assert elig == QuoteEligibility.QUOTE_BOTH
    assert "disabled" in reason
    assert drift is None


def test_warmup_when_insufficient_samples() -> None:
    s = _settings(DRIFT_LONG_WINDOW_MIN_SAMPLES=20)
    samples = _samples([100.0] * 5)
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=200.0, mid_now=100.0
    )
    assert elig == QuoteEligibility.QUOTE_BOTH
    assert "warmup" in reason
    assert drift is None


def test_drift_below_threshold_returns_quote_both() -> None:
    s = _settings()
    # 10 samples; endpoint drift +20 bps total. With 0.2 anchor
    # fraction (default), each anchor is the median of 3 samples;
    # smoothed drift is below the endpoint drift but well below the
    # 50 bps threshold either way.
    samples = _samples([100.0, 100.05, 100.10, 100.10, 100.15, 100.18, 100.18, 100.20, 100.20, 100.20])
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=130.0, mid_now=100.20
    )
    assert elig == QuoteEligibility.QUOTE_BOTH
    assert "ok" in reason
    assert drift is not None
    # Median(first 3) = 100.05; median(last 3) = 100.20. Drift ≈ 15 bps.
    assert drift == pytest.approx(15.0, abs=1.0)


def test_rising_trend_caps_to_buy_only() -> None:
    """Price up > 50 bps → ``QUOTE_BUY_ONLY``: don't sell into a
    rising market (selling would lock in losses against the trend)."""
    s = _settings()
    # 10 samples over 27s; price rises 80 bps total
    samples = _samples([100.0] * 5 + [100.40, 100.60, 100.70, 100.80, 100.80])
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=130.0, mid_now=100.80
    )
    assert elig == QuoteEligibility.QUOTE_BUY_ONLY
    assert "long_drift_up" in reason
    assert drift == pytest.approx(80.0, abs=1e-3)


def test_falling_trend_caps_to_sell_only() -> None:
    """Price down > 50 bps → ``QUOTE_SELL_ONLY``: don't buy into a
    falling market (don't catch falling knives — the regression case
    that motivated the gate)."""
    s = _settings()
    samples = _samples([100.0] * 5 + [99.60, 99.40, 99.30, 99.20, 99.20])
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=130.0, mid_now=99.20
    )
    assert elig == QuoteEligibility.QUOTE_SELL_ONLY
    assert "long_drift_down" in reason
    assert drift == pytest.approx(-80.0, abs=1e-3)


def test_non_finite_mid_returns_quote_both() -> None:
    s = _settings()
    samples = _samples([100.0] * 10)
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=130.0, mid_now=float("nan")
    )
    assert elig == QuoteEligibility.QUOTE_BOTH
    assert "no_mid" in reason
    assert drift is None


def test_stale_samples_outside_window_are_ignored() -> None:
    """Samples older than DRIFT_LONG_WINDOW_SECONDS are filtered out."""
    s = _settings(DRIFT_LONG_WINDOW_SECONDS=10.0)
    # First 5 samples timestamped well in the past — outside window.
    # Last 5 within window, all flat at 100.0 → no drift detected.
    old_samples = [(0.0 + i * 1.0, 90.0) for i in range(5)]
    fresh_samples = [(95.0 + i * 1.0, 100.0) for i in range(5)]
    samples = old_samples + fresh_samples
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=100.0, mid_now=100.0
    )
    # The old 90.0 samples are outside the 10 s window (now=100,
    # cutoff=90); only the fresh 100.0 samples are eligible →
    # 0 bps drift.
    assert elig == QuoteEligibility.QUOTE_BOTH
    assert "ok" in reason or "warmup" in reason  # depends on min_samples
    if "ok" in reason:
        assert drift == pytest.approx(0.0, abs=1e-6)


def test_threshold_boundary_just_above_50bps_fires() -> None:
    """Just above the threshold should fire. The default median
    anchoring requires the LAST anchor's median to be >50 bps above
    the FIRST anchor's median; build a sample that satisfies that
    cleanly."""
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=50.0)
    # 30 samples: first 12 at 100.00 (anchor1), last 12 at 100.55
    # (anchor2). Anchor median = 100.55 → drift = +55 bps, fires.
    flat_low = [100.0] * 12
    ramp = [100.10, 100.20, 100.30, 100.40, 100.50, 100.55]
    flat_high = [100.55] * 12
    samples = _samples(flat_low + ramp + flat_high)
    elig, _, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=samples[-1][0], mid_now=100.55
    )
    assert drift is not None and drift >= 50.0
    assert elig == QuoteEligibility.QUOTE_BUY_ONLY


def test_threshold_boundary_just_below_50bps_does_not_fire() -> None:
    """Just below the threshold should NOT fire."""
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=50.0)
    # +49 bps total at the late anchor median (100.49)
    flat_low = [100.0] * 12
    ramp = [100.10, 100.20, 100.30, 100.40, 100.45, 100.49]
    flat_high = [100.49] * 12
    samples = _samples(flat_low + ramp + flat_high)
    elig, _, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=samples[-1][0], mid_now=100.49
    )
    assert drift is not None and drift < 50.0
    assert elig == QuoteEligibility.QUOTE_BOTH


# --- spike-noise robustness ---


def test_single_old_spike_does_not_flip_gate() -> None:
    """The motivating case for median anchoring (BUGS/bug-002.md
    review): a single 5-min-ago aberration must NOT flip the gate
    for the full retention window. Build a stream where the OLDEST
    sample is wildly off but everything else is flat."""
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=50.0, DRIFT_LONG_WINDOW_MIN_SAMPLES=30)
    # Single old spike at $99 (-100 bps off mean), the other 29 anchor
    # samples at $100, all newer samples flat at $100. Mean trend = 0.
    spike = [99.0]
    flat = [100.0] * 35
    samples = _samples(spike + flat)
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=samples[-1][0], mid_now=100.0
    )
    # Median of first ~7 samples is 100.0 (the spike is overruled
    # by 6 flat samples). Median of last ~7 is 100.0. drift ≈ 0.
    assert drift is not None
    assert abs(drift) < 5.0, (
        f"single-sample spike must not flip the gate; got drift={drift:.2f} bps"
    )
    assert elig == QuoteEligibility.QUOTE_BOTH
    assert "ok" in reason


def test_single_recent_spike_does_not_flip_gate() -> None:
    """Symmetric to the previous test: a single recent aberration
    must not flip the gate either. The bot uses ``mid_now`` to
    populate the deque continuously; one bad reading shouldn't
    trigger 5 min of suppression."""
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=50.0, DRIFT_LONG_WINDOW_MIN_SAMPLES=30)
    flat = [100.0] * 35
    spike = [101.0]  # +100 bps spike on the most recent sample
    samples = _samples(flat + spike)
    elig, _, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=samples[-1][0], mid_now=101.0
    )
    # Median of last ~7 is dominated by the 6 flat 100.0 samples,
    # drift ≈ 0 even though `mid_now` is 101.
    assert drift is not None and abs(drift) < 5.0
    assert elig == QuoteEligibility.QUOTE_BOTH


def test_anchor_fraction_zero_falls_back_to_single_sample() -> None:
    """Setting anchor_fraction=0.0 disables median smoothing and
    reverts to the simple oldest-vs-mid_now comparison. Useful as
    an emergency fallback if the median anchoring has unexpected
    behaviour in production."""
    s = _settings(
        DRIFT_BLOCK_LONG_WINDOW_BPS=50.0,
        DRIFT_LONG_WINDOW_ANCHOR_FRACTION=0.0,
    )
    # Single old spike at $99: with anchor_fraction=0, the gate uses
    # the single oldest sample → drift = +101 bps → fires QUOTE_BUY_ONLY
    spike = [99.0]
    flat = [100.0] * 35
    samples = _samples(spike + flat)
    elig, _, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=samples[-1][0], mid_now=100.0
    )
    # Single-sample mode anchors on $99 → drift > +50 bps → BUY_ONLY
    assert drift is not None and drift > 50.0
    assert elig == QuoteEligibility.QUOTE_BUY_ONLY


def test_genuine_trend_still_caught_with_median_anchors() -> None:
    """The smoothing must NOT prevent the gate from firing on a real
    sustained trend. Linear -80 bps drift over the window should
    still be detected even with median anchoring."""
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=50.0, DRIFT_LONG_WINDOW_MIN_SAMPLES=30)
    n = 35
    start = 100.0
    end = 99.20  # -80 bps
    samples = _samples([
        start + (end - start) * (i / (n - 1)) for i in range(n)
    ])
    elig, reason, drift = _long_drift_eligibility(
        s, samples_long=samples, now_mono=samples[-1][0], mid_now=end
    )
    assert drift is not None and drift < -50.0
    assert elig == QuoteEligibility.QUOTE_SELL_ONLY
    assert "long_drift_down" in reason


# --- integration with compute_quote_eligibility ---


def test_full_pipeline_disabled_long_drift_does_not_affect_result() -> None:
    """When the gate is disabled, the pipeline behaves identically
    to pre-fix."""
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=0.0)
    long_samples = _samples([100.0] * 5 + [101.0] * 5)  # +100 bps would fire if enabled
    short_samples = [(199.99, 100.99), (200.0, 101.0)]
    result = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=101.0,
        now_mono=200.0,
        mid_samples=short_samples,
        seconds_since_public_bbo=0.5,
        gap_median_ms=500.0,
        gap_p95_ms=600.0,
        effective_staleness_ms=500.0,
        mid_samples_long=long_samples,
    )
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert result.mid_return_long_window_bps is None  # disabled = None


def test_full_pipeline_falling_trend_caps_to_sell_only() -> None:
    """Integration: a falling-trend long-window sample stream caps the
    final eligibility to QUOTE_SELL_ONLY even when the short-horizon
    drift/freshness gates are clean."""
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=50.0)
    long_samples = _samples([100.0] * 5 + [99.60, 99.40, 99.30, 99.20, 99.20])
    short_samples = [(199.99, 99.21), (200.0, 99.20)]
    result = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=99.20,
        now_mono=200.0,
        mid_samples=short_samples,
        seconds_since_public_bbo=0.5,
        gap_median_ms=500.0,
        gap_p95_ms=600.0,
        effective_staleness_ms=500.0,
        mid_samples_long=long_samples,
    )
    assert result.eligibility == QuoteEligibility.QUOTE_SELL_ONLY
    assert result.mid_return_long_window_bps is not None
    assert result.mid_return_long_window_bps < -50.0
    assert "long_drift_down" in result.reason
    assert "one_sided_due_long_drift" in result.counter_tags


def test_full_pipeline_rising_trend_caps_to_buy_only() -> None:
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=50.0)
    long_samples = _samples([100.0] * 5 + [100.40, 100.60, 100.70, 100.80, 100.80])
    short_samples = [(199.99, 100.79), (200.0, 100.80)]
    result = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=100.80,
        now_mono=200.0,
        mid_samples=short_samples,
        seconds_since_public_bbo=0.5,
        gap_median_ms=500.0,
        gap_p95_ms=600.0,
        effective_staleness_ms=500.0,
        mid_samples_long=long_samples,
    )
    assert result.eligibility == QuoteEligibility.QUOTE_BUY_ONLY
    assert result.mid_return_long_window_bps is not None
    assert result.mid_return_long_window_bps > 50.0
    assert "long_drift_up" in result.reason
    assert "one_sided_due_long_drift" in result.counter_tags


def test_full_pipeline_long_drift_more_restrictive_than_freshness() -> None:
    """If freshness says HOLD_ALL and long-drift says SELL_ONLY,
    the merge picks HOLD_ALL (more restrictive). The long-drift gate
    only ADDS restriction, never relaxes."""
    # Force HOLD_ALL via stale book age.
    s = _settings(
        QUOTE_HOLD_MAX_BOOK_AGE_MS=100.0,  # very tight
        QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS=50.0,
    )
    long_samples = _samples([100.0] * 5 + [99.60, 99.40, 99.30, 99.20, 99.20])
    short_samples = [(199.99, 99.21), (200.0, 99.20)]
    result = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=99.20,
        now_mono=200.0,
        mid_samples=short_samples,
        seconds_since_public_bbo=10.0,  # 10 s = 10000 ms; way over 100 ms threshold
        gap_median_ms=200.0,
        gap_p95_ms=200.0,
        effective_staleness_ms=10000.0,
        mid_samples_long=long_samples,
    )
    # HOLD_ALL wins (more restrictive than SELL_ONLY).
    assert result.eligibility == QuoteEligibility.HOLD_ALL


# --- regression replay against the 2026-04-25 failure case ---


def test_regression_replay_2026_04_25_falling_grind() -> None:
    """Synthesised mid-price stream matching the snapshot in
    `tmp/snap_20260425_174233`: 5-minute window inside a 4.8 h
    -100 bps trend on SUI. The gate must fire QUOTE_SELL_ONLY at the
    "right before the 9-BUY cluster" moment.

    Per the operator's snapshot analysis, the failure window centred
    at 16:00-16:33 UTC when the bot took 9 BUYs while the mid fell
    -25 bps. The 5-minute window leading into that moment had a
    sustained ~80-100 bps endpoint drift, which with the 0.2 anchor
    fraction smooths to ~64-80 bps median-anchored drift — comfortably
    above the 50 bps threshold.

    Note on the smoothing: median-anchored drift is approximately
    (1 - anchor_fraction) × endpoint_drift on a clean linear trend.
    With anchor_fraction=0.2 (default), the multiplier is 0.8. So an
    endpoint drift of 75 bps → ~60 bps median-anchored. Threshold of
    50 bps → fires comfortably.
    """
    s = _settings(DRIFT_BLOCK_LONG_WINDOW_BPS=50.0, DRIFT_LONG_WINDOW_MIN_SAMPLES=30)
    n = 100
    start_mid = 0.9425

    # Mild case: -45.6 bps endpoint, ~-37 bps median-anchored. Below
    # threshold; should NOT fire.
    end_mild = 0.9382
    long_mild = [
        (i * 3.0, start_mid + (end_mild - start_mid) * (i / (n - 1)))
        for i in range(n)
    ]
    now_mono = (n - 1) * 3.0
    result_mild = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=end_mild,
        now_mono=now_mono,
        mid_samples=[(now_mono - 0.5, end_mild + 0.0001), (now_mono, end_mild)],
        seconds_since_public_bbo=0.5,
        gap_median_ms=3000.0,
        gap_p95_ms=3500.0,
        effective_staleness_ms=500.0,
        mid_samples_long=long_mild,
    )
    assert result_mild.mid_return_long_window_bps is not None
    # Median-anchored ≈ 0.8 × -45.6 = -36.5 bps, below threshold.
    assert -45.0 < result_mild.mid_return_long_window_bps < -25.0
    assert result_mild.eligibility == QuoteEligibility.QUOTE_BOTH

    # Steep case: matches the actual 16 UTC hour of the snapshot
    # where the 5-min cumulative drift was substantial. End mid 0.9354
    # = -75 bps endpoint → ~-60 bps median-anchored, well below
    # the -50 bps threshold.
    end_steep = 0.9354
    long_steep = [
        (i * 3.0, start_mid + (end_steep - start_mid) * (i / (n - 1)))
        for i in range(n)
    ]
    result_steep = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=end_steep,
        now_mono=now_mono,
        mid_samples=[(now_mono - 0.5, end_steep + 0.0001), (now_mono, end_steep)],
        seconds_since_public_bbo=0.5,
        gap_median_ms=3000.0,
        gap_p95_ms=3500.0,
        effective_staleness_ms=500.0,
        mid_samples_long=long_steep,
    )
    assert result_steep.eligibility == QuoteEligibility.QUOTE_SELL_ONLY, (
        "drift gate must fire QUOTE_SELL_ONLY when 5-min median-"
        "anchored mid drift exceeds 50 bps (current value: "
        f"{result_steep.mid_return_long_window_bps:.2f} bps), preventing "
        "BUY-side accumulation into the falling-knife regime that "
        "produced -$0.71 of inventory-carry loss in "
        "tmp/snap_20260425_174233."
    )
    assert "one_sided_due_long_drift" in result_steep.counter_tags
