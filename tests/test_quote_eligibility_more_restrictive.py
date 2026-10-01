"""Tests for the side-set intersection semantics of
``quote_eligibility.more_restrictive`` (BUG-007 fix).

The previous rank-based implementation returned whichever arg had
higher rank in the order ``HOLD_ALL > one-sided > BOTH``. That
collapsed conflicting one-sided caps into "whichever ran first
wins" rather than the safe "no side allowed by both = HOLD_ALL".
"""

from __future__ import annotations

import pytest

from app.enums import QuoteEligibility
from app.quote_eligibility import more_restrictive


B = QuoteEligibility.QUOTE_BOTH
BUY = QuoteEligibility.QUOTE_BUY_ONLY
SELL = QuoteEligibility.QUOTE_SELL_ONLY
H = QuoteEligibility.HOLD_ALL


# --- HOLD_ALL absorbs ---


def test_hold_all_x_returns_hold_all() -> None:
    assert more_restrictive(H, B) is H
    assert more_restrictive(H, BUY) is H
    assert more_restrictive(H, SELL) is H
    assert more_restrictive(H, H) is H


def test_x_hold_all_returns_hold_all() -> None:
    assert more_restrictive(B, H) is H
    assert more_restrictive(BUY, H) is H
    assert more_restrictive(SELL, H) is H


# --- BOTH is identity ---


def test_both_with_anything_returns_anything() -> None:
    assert more_restrictive(B, B) is B
    assert more_restrictive(B, BUY) is BUY
    assert more_restrictive(B, SELL) is SELL


def test_anything_with_both_returns_anything() -> None:
    assert more_restrictive(BUY, B) is BUY
    assert more_restrictive(SELL, B) is SELL


# --- The BUG-007 regression case: opposing one-sided caps ---


def test_buy_only_intersect_sell_only_is_hold_all() -> None:
    """The headline regression: two safety gates that forbid
    opposite sides must collapse to HOLD_ALL, not 'whichever ran
    first'.

    Pre-fix this returned BUY_ONLY (or SELL_ONLY depending on arg
    order); post-fix it returns HOLD_ALL.
    """
    assert more_restrictive(BUY, SELL) is H
    assert more_restrictive(SELL, BUY) is H


def test_same_one_sided_is_idempotent() -> None:
    """Two safety gates that agree on a one-sided cap must still
    return that cap, not HOLD_ALL."""
    assert more_restrictive(BUY, BUY) is BUY
    assert more_restrictive(SELL, SELL) is SELL


# --- Symmetry / monotonicity properties ---


@pytest.mark.parametrize(
    "a,b",
    [
        (B, B), (B, BUY), (B, SELL), (B, H),
        (BUY, B), (BUY, BUY), (BUY, SELL), (BUY, H),
        (SELL, B), (SELL, BUY), (SELL, SELL), (SELL, H),
        (H, B), (H, BUY), (H, SELL), (H, H),
    ],
)
def test_symmetric(a: QuoteEligibility, b: QuoteEligibility) -> None:
    """Side-set intersection must be commutative."""
    assert more_restrictive(a, b) == more_restrictive(b, a)


@pytest.mark.parametrize(
    "a,b,c",
    [
        (B, BUY, SELL),
        (BUY, SELL, H),
        (BUY, BUY, SELL),
        (B, B, B),
    ],
)
def test_associative(
    a: QuoteEligibility, b: QuoteEligibility, c: QuoteEligibility
) -> None:
    """Order-independence under composition: applying caps in any
    order yields the same final cap."""
    left = more_restrictive(more_restrictive(a, b), c)
    right = more_restrictive(a, more_restrictive(b, c))
    assert left == right


# --- Integration: confirm the BUG-007 reproducer scenario yields HOLD_ALL ---


def test_per_side_cap_plus_long_drift_conflict_yields_hold_all() -> None:
    """The Codex repro: per-side uncertainty forbids one side AND
    the long-drift gate forbids the opposite side. After the fix,
    this collapses to HOLD_ALL — neither side allowed.
    """
    # Per-side uncertainty on BUY → caps to QUOTE_SELL_ONLY (sell still allowed)
    per_side_cap = SELL
    # Long-drift up (rising market) → caps to QUOTE_BUY_ONLY (don't sell into rising)
    long_drift_cap = BUY
    # Together, neither side is allowed by both gates.
    merged = more_restrictive(per_side_cap, long_drift_cap)
    assert merged is H
