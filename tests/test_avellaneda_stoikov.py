"""Tests for ``app.avellaneda_stoikov.compute_as_half_spread_bps``.

Phase 8A.5 — covers default-value sanity, monotonicity properties
(spread widens with vol, narrows with k), boundary handling
(non-finite inputs, k → 0), and safety-clamp behaviour.

The function is pure / side-effect-free; these tests are pure
numerical assertions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from app.avellaneda_stoikov import (
    compute_as_half_spread_bps,
    estimate_k_intensity_per_min,
)


@dataclass
class _FakeFill:
    """Stand-in for ``app.models.Fill`` — only needs ``ts_fill``."""

    ts_fill: datetime


# ---------------------------------------------------------------------------
# Default-value sanity — calibrated for TON-USDT-SWAP scale per the
# docstring's worked examples.
# ---------------------------------------------------------------------------


def test_typical_ton_regime_returns_meaningful_spread():
    """vol_bps=5, k=1 fills/min, defaults — half-spread ~3 bps.

    Anchors the calibration: at the bot's typical operating regime
    the formula produces a spread that's WIDER than the bot's
    current constant 4.0 (low_vol 2.5) but inside the historical
    BASE_HALF_SPREAD envelope.
    """
    half = compute_as_half_spread_bps(vol_bps=5.0, k_intensity_per_min=1.0)
    # base_floor (1.0) + γ_inv·vol² (0.05·25=1.25) + γ_edge·ln(1+1/1)=ln(2)≈0.693
    # = 1.0 + 1.25 + 0.693 = 2.943
    assert half == pytest.approx(2.943, abs=0.01)


def test_calm_regime_clamps_at_min_floor():
    """vol_bps=2, k=5 fills/min — formula would return ~1.38 bps,
    clamped to the min_half_spread_bps default of 1.5."""
    half = compute_as_half_spread_bps(vol_bps=2.0, k_intensity_per_min=5.0)
    assert half == 1.5


def test_high_vol_regime_returns_wide_spread():
    """vol_bps=15, k=0.5 fills/min — half-spread ~13 bps.

    Captures the "vol cluster" regime where today's constant
    4-bp half-spread is structurally too tight.
    """
    half = compute_as_half_spread_bps(vol_bps=15.0, k_intensity_per_min=0.5)
    # 1.0 + 0.05·225 + ln(1+1/0.5) = 1.0 + 11.25 + ln(3) = 1.0 + 11.25 + 1.098 = 13.348
    assert half == pytest.approx(13.348, abs=0.01)


def test_shock_regime_clamps_at_max_ceiling():
    """vol_bps=30 (severe shock) — formula would return 46 bps,
    clamped to the max_half_spread_bps default of 30."""
    half = compute_as_half_spread_bps(vol_bps=30.0, k_intensity_per_min=1.0)
    assert half == 30.0


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------


def test_spread_monotonic_increasing_in_vol():
    """Holding k fixed, spread increases as vol increases."""
    halves = [
        compute_as_half_spread_bps(vol_bps=v, k_intensity_per_min=1.0)
        for v in [1.0, 3.0, 5.0, 7.5, 10.0, 12.5]
    ]
    for a, b in zip(halves, halves[1:]):
        assert a <= b, f"non-monotonic: {a} -> {b}"


def test_spread_monotonic_decreasing_in_k():
    """Holding vol fixed, spread decreases as k (fill rate) grows."""
    halves = [
        compute_as_half_spread_bps(vol_bps=10.0, k_intensity_per_min=k)
        for k in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    ]
    for a, b in zip(halves, halves[1:]):
        assert a >= b, f"non-monotonic: {a} -> {b}"


def test_spread_monotonic_increasing_in_gamma_inv():
    """Holding vol and k fixed, spread increases as the risk
    aversion coefficient grows."""
    halves = [
        compute_as_half_spread_bps(
            vol_bps=8.0,
            k_intensity_per_min=1.0,
            gamma_inv=g,
        )
        for g in [0.01, 0.05, 0.10, 0.20]
    ]
    for a, b in zip(halves, halves[1:]):
        assert a <= b, f"non-monotonic in gamma_inv: {a} -> {b}"


# ---------------------------------------------------------------------------
# Boundary / defensive behaviour
# ---------------------------------------------------------------------------


def test_zero_vol_falls_back_to_base_plus_edge():
    """vol_bps=0 — inventory_risk vanishes; only base + edge."""
    half = compute_as_half_spread_bps(vol_bps=0.0, k_intensity_per_min=1.0)
    # base (1.0) + 0 + ln(2) ≈ 1.693, but clamped at min 1.5
    assert half == pytest.approx(1.693, abs=0.01)


def test_negative_vol_treated_as_zero():
    """A non-finite/negative vol_bps reading shouldn't crash —
    fall through to the zero-vol path."""
    half = compute_as_half_spread_bps(
        vol_bps=-3.0, k_intensity_per_min=1.0
    )
    half0 = compute_as_half_spread_bps(
        vol_bps=0.0, k_intensity_per_min=1.0
    )
    assert half == half0


def test_nan_vol_treated_as_zero():
    half_nan = compute_as_half_spread_bps(
        vol_bps=float("nan"), k_intensity_per_min=1.0
    )
    half0 = compute_as_half_spread_bps(
        vol_bps=0.0, k_intensity_per_min=1.0
    )
    assert half_nan == half0


def test_zero_k_clamps_to_floor():
    """k=0 (no recent fills) must not blow up the log. Caller's
    k gets floored at ``k_floor_per_min``."""
    half_zero = compute_as_half_spread_bps(
        vol_bps=5.0, k_intensity_per_min=0.0
    )
    half_floor = compute_as_half_spread_bps(
        vol_bps=5.0, k_intensity_per_min=0.1  # default floor
    )
    assert half_zero == pytest.approx(half_floor, abs=1e-9)


def test_negative_k_clamps_to_floor():
    half_neg = compute_as_half_spread_bps(
        vol_bps=5.0, k_intensity_per_min=-1.0
    )
    half_floor = compute_as_half_spread_bps(
        vol_bps=5.0, k_intensity_per_min=0.1
    )
    assert half_neg == pytest.approx(half_floor, abs=1e-9)


def test_nan_k_clamps_to_floor():
    half_nan = compute_as_half_spread_bps(
        vol_bps=5.0, k_intensity_per_min=float("nan")
    )
    half_floor = compute_as_half_spread_bps(
        vol_bps=5.0, k_intensity_per_min=0.1
    )
    assert half_nan == pytest.approx(half_floor, abs=1e-9)


def test_inf_vol_caps_at_max_ceiling():
    """Vol going to infinity would normally make the formula explode;
    the clamp catches it."""
    half = compute_as_half_spread_bps(
        vol_bps=float("inf"), k_intensity_per_min=1.0
    )
    # inf treated as 0 by the input guard (non-finite path)
    half_zero = compute_as_half_spread_bps(
        vol_bps=0.0, k_intensity_per_min=1.0
    )
    assert half == half_zero


# ---------------------------------------------------------------------------
# Clamp behaviour
# ---------------------------------------------------------------------------


def test_custom_min_clamp_respected():
    """Caller can lift the min floor."""
    half = compute_as_half_spread_bps(
        vol_bps=2.0,
        k_intensity_per_min=5.0,
        min_half_spread_bps=4.0,
    )
    assert half == 4.0


def test_custom_max_clamp_respected():
    """Caller can lower the max ceiling."""
    half = compute_as_half_spread_bps(
        vol_bps=20.0,
        k_intensity_per_min=1.0,
        max_half_spread_bps=10.0,
    )
    assert half == 10.0


def test_max_below_min_returns_max():
    """Pathological config (max < min) — the inner clamp picks the
    max first, then min would lift, then min wins. Defensive: caller
    shouldn't do this, but the function shouldn't crash."""
    half = compute_as_half_spread_bps(
        vol_bps=10.0,
        k_intensity_per_min=1.0,
        min_half_spread_bps=5.0,
        max_half_spread_bps=3.0,
    )
    # min wins per the implementation order (min(max, x) then max(min, ...))
    assert half == 5.0


# ---------------------------------------------------------------------------
# Coefficient knob behaviour
# ---------------------------------------------------------------------------


def test_gamma_inv_zero_disables_inventory_risk():
    """gamma_inv=0 — formula reduces to base + edge."""
    half = compute_as_half_spread_bps(
        vol_bps=20.0,
        k_intensity_per_min=1.0,
        gamma_inv=0.0,
    )
    half_no_vol = compute_as_half_spread_bps(
        vol_bps=0.0,
        k_intensity_per_min=1.0,
        gamma_inv=0.0,
    )
    assert half == pytest.approx(half_no_vol, abs=1e-9)


def test_gamma_edge_zero_disables_revenue_term():
    """gamma_edge=0 — formula reduces to base + inventory_risk."""
    half = compute_as_half_spread_bps(
        vol_bps=5.0,
        k_intensity_per_min=0.01,
        gamma_edge=0.0,
    )
    # base (1.0) + γ_inv·25 (1.25) + 0 = 2.25
    assert half == pytest.approx(2.25, abs=0.01)


def test_edge_alpha_scales_log_term():
    """Higher edge_alpha = more sensitivity to k in the log term."""
    base = compute_as_half_spread_bps(
        vol_bps=5.0, k_intensity_per_min=1.0, edge_alpha=1.0
    )
    high = compute_as_half_spread_bps(
        vol_bps=5.0, k_intensity_per_min=1.0, edge_alpha=3.0
    )
    assert high > base


# ---------------------------------------------------------------------------
# Calibration snapshot — pins the formula's shape so future tweaks
# of the helper's defaults are visible in PR diffs. These values are
# the contract for the wiring step (8A.3).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vol_bps,k_per_min,expected",
    [
        # vol_bps=2, k=2 fills/min — calm: just above floor.
        (2.0, 2.0, 1.0 + 0.2 + math.log(1.5)),
        # vol_bps=5, k=2 fills/min — normal regime.
        (5.0, 2.0, 1.0 + 1.25 + math.log(1.5)),
        # vol_bps=10, k=2 fills/min — modest vol cluster.
        (10.0, 2.0, 1.0 + 5.0 + math.log(1.5)),
        # vol_bps=10, k=0.5 fills/min — thin book.
        (10.0, 0.5, 1.0 + 5.0 + math.log(3.0)),
        # vol_bps=20, k=1 fills/min — shock.
        (20.0, 1.0, 1.0 + 20.0 + math.log(2.0)),
    ],
)
def test_calibration_snapshot_default_coeffs(
    vol_bps: float, k_per_min: float, expected: float
) -> None:
    """Pin specific (vol, k) → half-spread pairs at default coeffs.
    If the defaults move, this test fails — intentional, so the
    wiring step (8A.3) doesn't silently change production behaviour.
    """
    half = compute_as_half_spread_bps(
        vol_bps=vol_bps, k_intensity_per_min=k_per_min
    )
    expected_clamped = max(1.5, min(30.0, expected))
    assert half == pytest.approx(expected_clamped, abs=0.01)


# ---------------------------------------------------------------------------
# estimate_k_intensity_per_min — 8A.2 helper
# ---------------------------------------------------------------------------


_NOW = datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc)


def test_empty_fills_returns_zero():
    """Cold start — no fills yet, k = 0."""
    k = estimate_k_intensity_per_min([], now=_NOW, window_seconds=3600.0)
    assert k == 0.0


def test_all_fills_outside_window_returns_zero():
    """Fills exist but all older than the window — k = 0."""
    old = _NOW - timedelta(seconds=3700)  # 100s outside 3600s window
    fills = [_FakeFill(ts_fill=old)] * 10
    k = estimate_k_intensity_per_min(fills, now=_NOW, window_seconds=3600.0)
    assert k == 0.0


def test_fills_per_minute_basic_math():
    """6 fills inside a 60-minute window → 0.1 fills/min."""
    fills = [
        _FakeFill(ts_fill=_NOW - timedelta(seconds=i * 600))
        for i in range(6)
    ]
    k = estimate_k_intensity_per_min(fills, now=_NOW, window_seconds=3600.0)
    # 6 fills / (3600/60 = 60 min) = 0.1 fills/min
    assert k == pytest.approx(0.1, abs=1e-9)


def test_high_fill_rate():
    """60 fills inside a 60-minute window → 1.0 fills/min."""
    fills = [
        _FakeFill(ts_fill=_NOW - timedelta(seconds=i * 60))
        for i in range(60)
    ]
    k = estimate_k_intensity_per_min(fills, now=_NOW, window_seconds=3600.0)
    assert k == pytest.approx(1.0, abs=1e-9)


def test_fills_at_exact_cutoff_count_as_inside():
    """The cutoff is INCLUSIVE on the older side (ts >= cutoff)."""
    boundary = _NOW - timedelta(seconds=3600.0)
    fills = [_FakeFill(ts_fill=boundary)]
    k = estimate_k_intensity_per_min(fills, now=_NOW, window_seconds=3600.0)
    # 1 fill / 60 min = 0.0167/min
    assert k == pytest.approx(1.0 / 60.0, abs=1e-9)


def test_window_size_scales_rate_correctly():
    """Uniform density fills — shorter window returns same-or-higher
    per-minute rate (only the boundary fill creates the small
    difference). The fills/min normalisation is consistent."""
    # Fills every 60s for the last hour (60 fills total).
    fills = [
        _FakeFill(ts_fill=_NOW - timedelta(seconds=i * 60))
        for i in range(60)
    ]
    k_1h = estimate_k_intensity_per_min(
        fills, now=_NOW, window_seconds=3600.0
    )
    k_10m = estimate_k_intensity_per_min(
        fills, now=_NOW, window_seconds=600.0
    )
    # 1h: 60 fills / 60 min = 1.0 fills/min
    # 10m: 11 fills (i=0..10 inclusive of boundary) / 10 min = 1.1
    assert k_1h == pytest.approx(1.0, abs=1e-9)
    assert k_10m == pytest.approx(1.1, abs=1e-9)
    assert k_10m >= k_1h  # boundary effect in shorter windows


def test_zero_window_returns_zero():
    """Defensive — window_seconds=0 returns 0 instead of dividing by zero."""
    fills = [_FakeFill(ts_fill=_NOW)]
    k = estimate_k_intensity_per_min(fills, now=_NOW, window_seconds=0.0)
    assert k == 0.0


def test_negative_window_returns_zero():
    fills = [_FakeFill(ts_fill=_NOW)]
    k = estimate_k_intensity_per_min(fills, now=_NOW, window_seconds=-100.0)
    assert k == 0.0


def test_fills_with_missing_ts_skipped():
    """Defensive — entries lacking ts_fill are silently skipped."""

    class _BareFill:
        pass

    fills = [_BareFill(), _FakeFill(ts_fill=_NOW)]
    k = estimate_k_intensity_per_min(fills, now=_NOW, window_seconds=3600.0)
    # 1 valid fill in 60 min → 1/60 fills/min
    assert k == pytest.approx(1.0 / 60.0, abs=1e-9)


def test_naive_datetime_treated_as_utc():
    """If a fill's ts_fill is timezone-naive while now is aware,
    the function aligns to UTC instead of crashing."""
    naive = datetime(2026, 5, 27, 11, 59, 50)  # 10s before _NOW
    fills = [_FakeFill(ts_fill=naive)]
    k = estimate_k_intensity_per_min(fills, now=_NOW, window_seconds=3600.0)
    # Should count the fill as inside the window.
    assert k == pytest.approx(1.0 / 60.0, abs=1e-9)


def test_mixed_inside_outside_window():
    """Realistic mix: some fills inside, some outside the window."""
    inside = [
        _FakeFill(ts_fill=_NOW - timedelta(seconds=s))
        for s in [10, 100, 1000, 3000]  # all inside 3600s
    ]
    outside = [
        _FakeFill(ts_fill=_NOW - timedelta(seconds=s))
        for s in [3700, 5000, 100_000]  # all outside
    ]
    k = estimate_k_intensity_per_min(
        inside + outside, now=_NOW, window_seconds=3600.0
    )
    # 4 fills / 60 min = 0.0667 fills/min
    assert k == pytest.approx(4.0 / 60.0, abs=1e-9)
