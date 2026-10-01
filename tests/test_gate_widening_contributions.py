"""Gate-widening Phase 1 — per-gate ``widening_bps()`` contributions.

Each of the 7 regime-response gates publishes a function that
returns ``(bid_bps, ask_bps)`` representing how much the gate
widens each side of the half-spread.

Phase 1 invariants verified per gate:

* When the gate's underlying signal is clear, contribution is
  ``(0.0, 0.0)``.
* When the signal fires, contribution is ``MAX_HALF_SPREAD_BPS``
  on the side(s) the gate would have suppressed under the
  pre-migration binary eligibility model.
* Symmetric gates (vol_trend, post_swing, basis_regime) widen
  both sides equally.
* Asymmetric gates (microprice, momentum, freshness_one_sided,
  recovery_cooldown) widen only the side facing the signal.

The 0/MAX boundary is the "gate-equivalent" initial coefficient
per `plans/gate-to-widening.md` Phase 1. Phase 2 of the migration
replaces these with continuous functions of the underlying signal
magnitude.
"""

from __future__ import annotations

from dataclasses import replace

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


# ---------- vol_trend_gate ----------------------------------------

def test_vol_trend_widening_inactive_is_zero() -> None:
    s = vol_trend_gate.VolTrendState()
    s.cooldown_until_mono = 0.0  # never fired
    assert vol_trend_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS
    ) == (0.0, 0.0)


def test_vol_trend_widening_active_is_max_symmetric() -> None:
    s = vol_trend_gate.VolTrendState()
    s.cooldown_until_mono = 110.0
    bid, ask = vol_trend_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS
    )
    assert bid == MAX_BPS
    assert ask == MAX_BPS  # symmetric


# ---------- post_swing_gate ---------------------------------------

def test_post_swing_widening_inactive_is_zero() -> None:
    s = post_swing_gate.PostSwingState()
    s.cooldown_until_mono = 0.0
    assert post_swing_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS
    ) == (0.0, 0.0)


def test_post_swing_widening_active_is_max_symmetric() -> None:
    s = post_swing_gate.PostSwingState()
    s.cooldown_until_mono = 110.0
    bid, ask = post_swing_gate.widening_bps(
        s, now_mono=100.0, max_half_spread_bps=MAX_BPS
    )
    assert bid == MAX_BPS
    assert ask == MAX_BPS


# ---------- microprice_gate ---------------------------------------

def test_microprice_widening_neutral_book_is_zero() -> None:
    # imbalance below threshold = clear.
    bid, ask = microprice_gate.widening_bps(
        ob_imbalance_ewma=0.0,
        threshold=0.5,
        max_half_spread_bps=MAX_BPS,
    )
    assert (bid, ask) == (0.0, 0.0)


def test_microprice_widening_ask_thin_widens_ask_only() -> None:
    """Positive imbalance = bid-heavy = asks will be eaten. Widen
    the bot's ask quote (it'd be adversely selected if it sat there
    at normal width)."""
    bid, ask = microprice_gate.widening_bps(
        ob_imbalance_ewma=0.8,
        threshold=0.5,
        max_half_spread_bps=MAX_BPS,
    )
    assert bid == 0.0
    assert ask == MAX_BPS


def test_microprice_widening_bid_thin_widens_bid_only() -> None:
    """Negative imbalance = ask-heavy = bids will be eaten. Widen
    the bot's bid quote."""
    bid, ask = microprice_gate.widening_bps(
        ob_imbalance_ewma=-0.8,
        threshold=0.5,
        max_half_spread_bps=MAX_BPS,
    )
    assert bid == MAX_BPS
    assert ask == 0.0


def test_microprice_widening_disabled_returns_zero() -> None:
    bid, ask = microprice_gate.widening_bps(
        ob_imbalance_ewma=0.8,
        threshold=0.5,
        max_half_spread_bps=MAX_BPS,
        enabled=False,
    )
    assert (bid, ask) == (0.0, 0.0)


# ---------- basis_regime_gate -------------------------------------

def test_basis_regime_widening_signal_present_is_zero() -> None:
    """Strong basis IC = signal present = gate clear."""
    bid, ask = basis_regime_gate.widening_bps(
        last_ic=0.5,  # well above threshold
        pair_count=100,
        ic_min_quote_threshold=0.05,
        min_pair_samples=50,
        max_half_spread_bps=MAX_BPS,
    )
    assert (bid, ask) == (0.0, 0.0)


def test_basis_regime_widening_signal_absent_widens_both() -> None:
    """Weak basis IC = signal absent = gate fires symmetrically."""
    bid, ask = basis_regime_gate.widening_bps(
        last_ic=0.01,  # below threshold
        pair_count=100,
        ic_min_quote_threshold=0.05,
        min_pair_samples=50,
        max_half_spread_bps=MAX_BPS,
    )
    assert bid == MAX_BPS
    assert ask == MAX_BPS


# ---------- momentum_gate -----------------------------------------

def test_momentum_widening_flat_position_is_zero() -> None:
    bid, ask = momentum_gate.widening_bps(
        position_qty=0.0,
        effective_abs_cap=10.0,
        drift_bps=5.0,
        drift_threshold_bps=2.0,
        inventory_pct_threshold=0.4,
        max_half_spread_bps=MAX_BPS,
    )
    assert (bid, ask) == (0.0, 0.0)


def test_momentum_widening_long_uptrend_widens_bid() -> None:
    """Long position + up-drift → don't add to long → widen BID
    (suppresses BUY)."""
    bid, ask = momentum_gate.widening_bps(
        position_qty=8.0,  # 80% of cap
        effective_abs_cap=10.0,
        drift_bps=5.0,  # up-drift, above threshold
        drift_threshold_bps=2.0,
        inventory_pct_threshold=0.4,
        max_half_spread_bps=MAX_BPS,
    )
    assert bid == MAX_BPS
    assert ask == 0.0


def test_momentum_widening_short_downtrend_widens_ask() -> None:
    """Short position + down-drift → don't add to short → widen
    ASK (suppresses SELL)."""
    bid, ask = momentum_gate.widening_bps(
        position_qty=-8.0,
        effective_abs_cap=10.0,
        drift_bps=-5.0,
        drift_threshold_bps=2.0,
        inventory_pct_threshold=0.4,
        max_half_spread_bps=MAX_BPS,
    )
    assert bid == 0.0
    assert ask == MAX_BPS


def test_momentum_widening_anti_aligned_is_zero() -> None:
    """Long + down-drift = inventory already on wrong side of
    momentum. Other machinery (inventory_exec_bias, adverse_side_pause)
    handles this case; momentum gate stays out → zero widening."""
    bid, ask = momentum_gate.widening_bps(
        position_qty=8.0,
        effective_abs_cap=10.0,
        drift_bps=-5.0,  # ANTI-aligned
        drift_threshold_bps=2.0,
        inventory_pct_threshold=0.4,
        max_half_spread_bps=MAX_BPS,
    )
    assert (bid, ask) == (0.0, 0.0)


# ---------- freshness_one_sided (in quote_eligibility.py) ---------

def _elig(eligibility: QuoteEligibility, reason: str) -> QuoteEligibilityResult:
    """Minimal QuoteEligibilityResult for testing."""
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
        in_cooldown=False,
    )


def test_freshness_widening_clear_is_zero() -> None:
    r = _elig(QuoteEligibility.QUOTE_BOTH, "ok|fresh=freshness_ok")
    assert freshness_one_sided_widening_bps(
        r, max_half_spread_bps=MAX_BPS
    ) == (0.0, 0.0)


def test_freshness_widening_buy_only_widens_ask() -> None:
    """``QUOTE_BUY_ONLY`` with freshness_one_sided reason means the
    SELL side was suppressed → widen ask."""
    r = _elig(
        QuoteEligibility.QUOTE_BUY_ONLY,
        "freshness_only|fresh=freshness_one_sided:local_receipt_ms>1000",
    )
    bid, ask = freshness_one_sided_widening_bps(r, max_half_spread_bps=MAX_BPS)
    assert bid == 0.0
    assert ask == MAX_BPS


def test_freshness_widening_sell_only_widens_bid() -> None:
    r = _elig(
        QuoteEligibility.QUOTE_SELL_ONLY,
        "freshness_only|fresh=freshness_one_sided:local_receipt_ms>1000",
    )
    bid, ask = freshness_one_sided_widening_bps(r, max_half_spread_bps=MAX_BPS)
    assert bid == MAX_BPS
    assert ask == 0.0


def test_freshness_widening_no_signature_in_reason_is_zero() -> None:
    """A QUOTE_BUY_ONLY result that wasn't caused by freshness
    (some other gate) doesn't trigger freshness widening."""
    r = _elig(
        QuoteEligibility.QUOTE_BUY_ONLY,
        "ok|microprice_gate:ask_thin",  # different cause
    )
    assert freshness_one_sided_widening_bps(
        r, max_half_spread_bps=MAX_BPS
    ) == (0.0, 0.0)


# ---------- recovery_cooldown (in quote_eligibility.py) -----------

def test_recovery_cooldown_widening_clear_is_zero() -> None:
    r = _elig(QuoteEligibility.QUOTE_BOTH, "ok")
    # in_cooldown defaults False; widening should be zero.
    assert recovery_cooldown_widening_bps(
        r, max_half_spread_bps=MAX_BPS
    ) == (0.0, 0.0)


def test_recovery_cooldown_widening_buy_only_widens_ask() -> None:
    r = replace(
        _elig(QuoteEligibility.QUOTE_BUY_ONLY, "...|recovery_cooldown"),
        in_cooldown=True,
    )
    bid, ask = recovery_cooldown_widening_bps(r, max_half_spread_bps=MAX_BPS)
    assert bid == 0.0
    assert ask == MAX_BPS


def test_recovery_cooldown_widening_hold_all_widens_both() -> None:
    r = replace(
        _elig(QuoteEligibility.HOLD_ALL, "...|recovery_cooldown"),
        in_cooldown=True,
    )
    bid, ask = recovery_cooldown_widening_bps(r, max_half_spread_bps=MAX_BPS)
    assert bid == MAX_BPS
    assert ask == MAX_BPS
