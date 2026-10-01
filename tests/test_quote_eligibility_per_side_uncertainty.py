"""Per-side order-state uncertainty must cap eligibility to the OTHER side.

Context: ``tmp/snap_20260417_183547`` had BUY stuck in CANCEL_PENDING for 90+ s.
The previous logic treated ANY side-unresolved as global uncertainty → HOLD_ALL,
so SELL was silenced for the entire stall. Now:

- BUY uncertain, SELL clean → QUOTE_SELL_ONLY
- SELL uncertain, BUY clean → QUOTE_BUY_ONLY
- Both uncertain → HOLD_ALL
- Global desync (``order_state_uncertainty=True``) → HOLD_ALL regardless
"""

from __future__ import annotations

from app.enums import QuoteEligibility, Side
from app.quote_eligibility import compute_quote_eligibility
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    return UnitTestSettings.model_validate({"TRADING_ENABLED": False})


def _fresh_kwargs() -> dict:
    return dict(
        mid_now=100.0,
        now_mono=1000.0,
        mid_samples=[(999.5, 100.0), (999.0, 100.0)],
        seconds_since_public_bbo=0.05,
        gap_median_ms=50.0,
        gap_p95_ms=150.0,
        effective_staleness_ms=60.0,
    )


def test_buy_side_uncertain_caps_to_sell_only() -> None:
    r = compute_quote_eligibility(
        _settings(),
        order_state_uncertainty=False,
        uncertain_sides=frozenset({Side.BUY}),
        **_fresh_kwargs(),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_SELL_ONLY
    assert "buy_side_uncertain" in r.reason
    assert "one_sided_due_buy_uncertain" in r.counter_tags


def test_sell_side_uncertain_caps_to_buy_only() -> None:
    r = compute_quote_eligibility(
        _settings(),
        order_state_uncertainty=False,
        uncertain_sides=frozenset({Side.SELL}),
        **_fresh_kwargs(),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BUY_ONLY
    assert "sell_side_uncertain" in r.reason
    assert "one_sided_due_sell_uncertain" in r.counter_tags


def test_both_sides_uncertain_falls_back_to_hold_all() -> None:
    r = compute_quote_eligibility(
        _settings(),
        order_state_uncertainty=False,
        uncertain_sides=frozenset({Side.BUY, Side.SELL}),
        **_fresh_kwargs(),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL
    assert "both_sides" in r.reason
    assert "hold_order_uncertainty_both_sides" in r.counter_tags


def test_global_uncertainty_overrides_per_side_and_holds_all() -> None:
    """When desync is set, we HOLD_ALL regardless of per-side data."""
    r = compute_quote_eligibility(
        _settings(),
        order_state_uncertainty=True,
        uncertain_sides=frozenset(),
        **_fresh_kwargs(),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL
    assert r.reason == "order_state_uncertainty"


def test_no_uncertainty_yields_quote_both() -> None:
    r = compute_quote_eligibility(
        _settings(),
        order_state_uncertainty=False,
        uncertain_sides=frozenset(),
        **_fresh_kwargs(),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH


def test_uncertain_sides_default_none_preserves_legacy_quote_both() -> None:
    """Callers that don't pass ``uncertain_sides`` get the pre-refactor behaviour."""
    r = compute_quote_eligibility(
        _settings(),
        order_state_uncertainty=False,
        **_fresh_kwargs(),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH


def test_buy_uncertain_plus_stale_book_stays_hold_all() -> None:
    """Stale book is MORE restrictive than SELL_ONLY (HOLD_ALL wins)."""
    kw = _fresh_kwargs()
    # Force freshness HOLD: gap_p95 well beyond the default hold threshold.
    kw["gap_p95_ms"] = 10_000.0
    kw["seconds_since_public_bbo"] = 10.0
    r = compute_quote_eligibility(
        _settings(),
        order_state_uncertainty=False,
        uncertain_sides=frozenset({Side.BUY}),
        **kw,
    )
    # Merge takes the MORE restrictive of (freshness HOLD, SELL_ONLY) = HOLD_ALL.
    assert r.eligibility == QuoteEligibility.HOLD_ALL
