"""Soft-flatten staged-pricing helper tests.

Phase 1: near-touch (best on the reduce side -- best_ask for SELL,
best_bid for BUY). Phase 2: one tick from the FAR touch into the
spread (best_bid+1 for SELL, best_ask-1 for BUY). On a 1-tick spread
the two phases collapse to the same level. Verified across long /
short and tight / wide spreads.
"""

from __future__ import annotations

import pytest

from app.enums import Side
from app.soft_flatten import compute_target_price


# ---------------------------------------------------------------------------
# Phase 1 -- near-touch
# ---------------------------------------------------------------------------


def test_phase_1_long_sells_at_best_ask() -> None:
    side, price = compute_target_price(
        pos_qty=10.0,
        best_bid=0.9712,
        best_ask=0.9715,
        tick_size=0.0001,
        in_phase_2=False,
    )
    assert side == Side.SELL
    assert price == pytest.approx(0.9715)


def test_phase_1_short_buys_at_best_bid() -> None:
    side, price = compute_target_price(
        pos_qty=-10.0,
        best_bid=0.9712,
        best_ask=0.9715,
        tick_size=0.0001,
        in_phase_2=False,
    )
    assert side == Side.BUY
    assert price == pytest.approx(0.9712)


# ---------------------------------------------------------------------------
# Phase 2 -- max-aggressive post-only (one tick from far touch)
# ---------------------------------------------------------------------------


def test_phase_2_long_sells_one_tick_above_bid_on_wide_spread() -> None:
    """3-tick spread: SELL phase-2 = best_bid + 1tick = 0.9713."""
    side, price = compute_target_price(
        pos_qty=10.0,
        best_bid=0.9712,
        best_ask=0.9715,
        tick_size=0.0001,
        in_phase_2=True,
    )
    assert side == Side.SELL
    assert price == pytest.approx(0.9713)


def test_phase_2_short_buys_one_tick_below_ask_on_wide_spread() -> None:
    """3-tick spread: BUY phase-2 = best_ask - 1tick = 0.9714."""
    side, price = compute_target_price(
        pos_qty=-10.0,
        best_bid=0.9712,
        best_ask=0.9715,
        tick_size=0.0001,
        in_phase_2=True,
    )
    assert side == Side.BUY
    assert price == pytest.approx(0.9714)


# ---------------------------------------------------------------------------
# Phase collapse on tight spreads
# ---------------------------------------------------------------------------


def test_phase_2_collapses_to_phase_1_on_one_tick_spread_long() -> None:
    """1-tick spread (best_bid=0.9712, best_ask=0.9713): phase-2
    target = best_bid+1tick = 0.9713 = phase_1. The cap-at-phase-1
    guard returns phase_1 unchanged so we don't drift backwards."""
    side, price_p1 = compute_target_price(
        pos_qty=10.0,
        best_bid=0.9712,
        best_ask=0.9713,
        tick_size=0.0001,
        in_phase_2=False,
    )
    _, price_p2 = compute_target_price(
        pos_qty=10.0,
        best_bid=0.9712,
        best_ask=0.9713,
        tick_size=0.0001,
        in_phase_2=True,
    )
    assert side == Side.SELL
    assert price_p1 == pytest.approx(0.9713)
    assert price_p2 == pytest.approx(0.9713)


def test_phase_2_collapses_to_phase_1_on_one_tick_spread_short() -> None:
    side, price_p1 = compute_target_price(
        pos_qty=-10.0,
        best_bid=0.9712,
        best_ask=0.9713,
        tick_size=0.0001,
        in_phase_2=False,
    )
    _, price_p2 = compute_target_price(
        pos_qty=-10.0,
        best_bid=0.9712,
        best_ask=0.9713,
        tick_size=0.0001,
        in_phase_2=True,
    )
    assert side == Side.BUY
    assert price_p1 == pytest.approx(0.9712)
    assert price_p2 == pytest.approx(0.9712)


# ---------------------------------------------------------------------------
# Phase 2 must not regress past phase 1
# ---------------------------------------------------------------------------


def test_phase_2_does_not_regress_past_phase_1_long() -> None:
    """Belt-and-braces: phase 2 SELL price must never be ABOVE
    phase 1 (best_ask). If best_bid drifts up unexpectedly such
    that best_bid+1 > best_ask, the cap forces it down to phase 1."""
    # Pathological book where bid > ask - 1 tick (exchange glitch).
    side, price = compute_target_price(
        pos_qty=10.0,
        best_bid=0.9716,  # higher than best_ask - 1tick
        best_ask=0.9715,
        tick_size=0.0001,
        in_phase_2=True,
    )
    assert side == Side.SELL
    assert price == pytest.approx(0.9715)  # capped at best_ask


def test_phase_2_does_not_regress_past_phase_1_short() -> None:
    """Same property for BUY: phase 2 must never be BELOW phase 1
    (best_bid)."""
    side, price = compute_target_price(
        pos_qty=-10.0,
        best_bid=0.9712,
        best_ask=0.9711,  # lower than best_bid + 1tick
        tick_size=0.0001,
        in_phase_2=True,
    )
    assert side == Side.BUY
    assert price == pytest.approx(0.9712)  # floored at best_bid


# ---------------------------------------------------------------------------
# Defensive: zero tick_size disables phase 2
# ---------------------------------------------------------------------------


def test_zero_tick_size_keeps_phase_1_price() -> None:
    """If symbol_spec didn't load a tick size (production: should
    never happen, defensive: tests + degenerate adapters), stay at
    phase 1 rather than nudging by an unknown amount."""
    _, price = compute_target_price(
        pos_qty=10.0,
        best_bid=0.9712,
        best_ask=0.9715,
        tick_size=0.0,
        in_phase_2=True,
    )
    assert price == pytest.approx(0.9715)
