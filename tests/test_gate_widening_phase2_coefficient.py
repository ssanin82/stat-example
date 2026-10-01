"""Gate-widening Phase 2 — per-gate ``widen_bps`` coefficient knob.

Phase 2 adds operator-tunable widening magnitude per gate (v1.4.13).
This file locks in the coefficient semantics:

* Sentinel ``-1.0`` (default) → use ``max_half_spread_bps``
  (gate-equivalent magnitude, matches pre-cutover HOLD_ALL).
* Positive value <= cap → that value is the widening contribution.
* Positive value > cap → clamped to cap (defensive).
* Negative non-sentinel (e.g. ``-5.0``) → treated as sentinel
  (any negative value falls back to cap).
* Zero → zero widening (gate is effectively disabled at the widening
  layer; still fires for telemetry).

Each gate's widening_bps applies the coefficient with side-aware
semantics — symmetric gates apply it to both sides; asymmetric ones
to the side facing the signal.
"""

from __future__ import annotations

from app import (
    basis_regime_gate,
    microprice_gate,
    momentum_gate,
    post_swing_gate,
    vol_trend_gate,
)
from app.enums import QuoteEligibility
from app.quote_eligibility import (
    QuoteEligibilityResult,
    freshness_one_sided_widening_bps,
    recovery_cooldown_widening_bps,
)


MAX_BPS = 30.0


# ---------- vol_trend ---------------------------------------------

def test_vol_trend_default_sentinel_uses_max() -> None:
    """Sentinel -1.0 default falls back to MAX_HALF_SPREAD_BPS."""
    s = vol_trend_gate.VolTrendState()
    s.cooldown_until_mono = 110.0
    bid, ask = vol_trend_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS
    )
    assert bid == MAX_BPS
    assert ask == MAX_BPS


def test_vol_trend_explicit_coefficient_applied() -> None:
    """Operator sets a specific coefficient (e.g. 15 bps for partial
    widening). Both sides get that coefficient symmetrically."""
    s = vol_trend_gate.VolTrendState()
    s.cooldown_until_mono = 110.0
    bid, ask = vol_trend_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS, widen_bps=15.0
    )
    assert bid == 15.0
    assert ask == 15.0


def test_vol_trend_zero_coefficient_disables_widening() -> None:
    """Operator can effectively disable the gate by setting widen_bps=0.
    The gate still fires for telemetry but contributes zero spread."""
    s = vol_trend_gate.VolTrendState()
    s.cooldown_until_mono = 110.0
    bid, ask = vol_trend_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS, widen_bps=0.0
    )
    assert (bid, ask) == (0.0, 0.0)


def test_vol_trend_coefficient_clamped_to_cap() -> None:
    """A misconfigured coefficient larger than MAX_HALF_SPREAD_BPS
    gets clamped — composition's effective half-spread is bounded
    by the cap anyway, but the contribution shouldn't exceed it
    locally either (clean diagnostics)."""
    s = vol_trend_gate.VolTrendState()
    s.cooldown_until_mono = 110.0
    bid, ask = vol_trend_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS, widen_bps=100.0
    )
    assert bid == MAX_BPS
    assert ask == MAX_BPS


def test_vol_trend_inactive_zero_regardless_of_coefficient() -> None:
    s = vol_trend_gate.VolTrendState()
    s.cooldown_until_mono = 0.0  # not firing
    bid, ask = vol_trend_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS, widen_bps=15.0
    )
    assert (bid, ask) == (0.0, 0.0)


# ---------- microprice (asymmetric) -------------------------------

def test_microprice_coefficient_applies_to_suppressed_side_only() -> None:
    """Microprice fires ask_thin → widen ask. Operator's coefficient
    applies to the widened side only; the un-widened side stays at
    zero contribution regardless of the coefficient value."""
    bid, ask = microprice_gate.widening_bps(
        ob_imbalance_ewma=0.8,
        threshold=0.5,
        max_half_spread_bps=MAX_BPS,
        widen_bps=12.0,
    )
    assert bid == 0.0
    assert ask == 12.0


def test_microprice_default_sentinel_uses_max_asymmetric() -> None:
    bid, ask = microprice_gate.widening_bps(
        ob_imbalance_ewma=-0.8,
        threshold=0.5,
        max_half_spread_bps=MAX_BPS,
    )
    assert bid == MAX_BPS
    assert ask == 0.0


# ---------- basis_regime ------------------------------------------

def test_basis_regime_coefficient_symmetric() -> None:
    bid, ask = basis_regime_gate.widening_bps(
        last_ic=0.01,
        pair_count=100,
        ic_min_quote_threshold=0.05,
        min_pair_samples=50,
        max_half_spread_bps=MAX_BPS,
        widen_bps=20.0,
    )
    assert bid == 20.0
    assert ask == 20.0


# ---------- momentum (asymmetric) ---------------------------------

def test_momentum_coefficient_applies_to_aligned_side() -> None:
    """Long + up-drift → widen BID (suppresses BUY-into-trend).
    Operator coefficient applies only to that side."""
    bid, ask = momentum_gate.widening_bps(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps=5.0,
        drift_threshold_bps=2.0,
        inventory_pct_threshold=0.4,
        max_half_spread_bps=MAX_BPS,
        widen_bps=18.0,
    )
    assert bid == 18.0
    assert ask == 0.0


# ---------- post_swing --------------------------------------------

def test_post_swing_coefficient_symmetric() -> None:
    s = post_swing_gate.PostSwingState()
    s.cooldown_until_mono = 105.0
    bid, ask = post_swing_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS, widen_bps=10.0
    )
    assert bid == 10.0
    assert ask == 10.0


# ---------- freshness_one_sided / recovery_cooldown ---------------

def _elig(eligibility, reason: str, in_cooldown: bool = False) -> QuoteEligibilityResult:
    return QuoteEligibilityResult(
        eligibility=eligibility,
        reason=reason,
        seconds_since_last_public_book_update=0.0,
        effective_staleness_ms=0.0,
        market_data_gap_p95_ms=0.0,
        market_data_gap_median_ms=0.0,
        mid_return_100ms_bps=None,
        mid_return_250ms_bps=None,
        mid_return_500ms_bps=None,
        jump_100ms_bps=None,
        jump_250ms_bps=None,
        jump_500ms_bps=None,
        in_cooldown=in_cooldown,
    )


def test_freshness_coefficient_applies_to_suppressed_side() -> None:
    r = _elig(
        QuoteEligibility.QUOTE_BUY_ONLY,
        "freshness_only|fresh=freshness_one_sided:local_receipt_ms>1000",
    )
    bid, ask = freshness_one_sided_widening_bps(
        r, max_half_spread_bps=MAX_BPS, widen_bps=8.0
    )
    assert bid == 0.0
    assert ask == 8.0


def test_recovery_cooldown_coefficient_applies() -> None:
    from dataclasses import replace

    r = replace(
        _elig(QuoteEligibility.QUOTE_BUY_ONLY, "...|recovery_cooldown"),
        in_cooldown=True,
    )
    bid, ask = recovery_cooldown_widening_bps(
        r, max_half_spread_bps=MAX_BPS, widen_bps=5.0
    )
    assert bid == 0.0
    assert ask == 5.0
