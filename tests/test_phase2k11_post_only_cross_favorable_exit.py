"""Phase 2K.11 — favorable-exit predicate for the
``post_only_cross_cooldown``.

Lowest-priority of the Phase 2K series because the cooldown is
already short (2.5 s), but the favorable-exit pattern is now applied
consistently across every gate.

Pure-function tests for ``touch_moved_away_from_rejected_price``.
Integration with ``OrderManager._post_only_cross_cooldown_active``
is exercised by the existing execution.py test suite (which still
passes — the legacy timer-only path is preserved when the
favorable-exit knobs are disabled or inputs are missing).
"""

from __future__ import annotations

from app.enums import Side
from app.post_only_cross_cooldown import touch_moved_away_from_rejected_price


# ---------------------------------------------------------------------------
# BUY side — conflict was with best_ask
# ---------------------------------------------------------------------------


def test_buy_clears_when_ask_moves_up_one_tick() -> None:
    """Post-only BUY at P=2.00 rejected because best_ask was <= 2.00.
    Clears when best_ask >= P + 1 tick (e.g., 2.01)."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.01,
        tick_size=0.01,
    ) is True


def test_buy_does_not_clear_when_ask_at_rejected_price() -> None:
    """If best_ask == rejected_price, re-placement would STILL cross."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.00,
        tick_size=0.01,
    ) is False


def test_buy_does_not_clear_when_ask_moved_only_half_tick() -> None:
    """Need a FULL tick of separation by default (tick_multiplier=1.0).
    2.005 only halfway up."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.005,
        tick_size=0.01,
    ) is False


def test_buy_clears_when_ask_moved_more_than_one_tick() -> None:
    """Two ticks of separation — definitely clear."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.02,
        tick_size=0.01,
    ) is True


# ---------------------------------------------------------------------------
# SELL side — conflict was with best_bid
# ---------------------------------------------------------------------------


def test_sell_clears_when_bid_moves_down_one_tick() -> None:
    """Post-only SELL at P=2.00 rejected because best_bid was >= 2.00.
    Clears when best_bid <= P - 1 tick (e.g., 1.99)."""
    assert touch_moved_away_from_rejected_price(
        side=Side.SELL,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.01,
        tick_size=0.01,
    ) is True


def test_sell_does_not_clear_when_bid_at_rejected_price() -> None:
    assert touch_moved_away_from_rejected_price(
        side=Side.SELL,
        rejected_price=2.00,
        best_bid=2.00,
        best_ask=2.01,
        tick_size=0.01,
    ) is False


def test_sell_does_not_clear_when_bid_moved_only_half_tick() -> None:
    assert touch_moved_away_from_rejected_price(
        side=Side.SELL,
        rejected_price=2.00,
        best_bid=1.995,
        best_ask=2.01,
        tick_size=0.01,
    ) is False


# ---------------------------------------------------------------------------
# tick_multiplier
# ---------------------------------------------------------------------------


def test_tick_multiplier_two_requires_two_ticks() -> None:
    """tick_multiplier=2 → require ≥ 2 ticks of separation."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.01,  # only 1 tick away
        tick_size=0.01,
        tick_multiplier=2.0,
    ) is False
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.02,  # 2 ticks away
        tick_size=0.01,
        tick_multiplier=2.0,
    ) is True


def test_tick_multiplier_zero_disables_predicate() -> None:
    """Multiplier=0 → predicate never fires (margin is 0; would
    require touch BEYOND the rejected price, which means a cross
    that's already self-resolving, but we treat 0 as disabled)."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.10,  # very far above
        tick_size=0.01,
        tick_multiplier=0.0,
    ) is False


# ---------------------------------------------------------------------------
# Missing inputs — predicate must NOT fire (deferred to timer ceiling)
# ---------------------------------------------------------------------------


def test_missing_rejected_price_does_not_fire() -> None:
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=None,
        best_bid=1.99,
        best_ask=2.50,
        tick_size=0.01,
    ) is False


def test_missing_tick_size_does_not_fire() -> None:
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.50,
        tick_size=None,
    ) is False


def test_zero_tick_size_does_not_fire() -> None:
    """A zero tick is nonsensical for the predicate; defer to timer."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=2.50,
        tick_size=0.0,
    ) is False


def test_buy_missing_ask_does_not_fire() -> None:
    """BUY's conflict is with best_ask; if it's missing we can't
    evaluate."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=1.99,
        best_ask=None,
        tick_size=0.01,
    ) is False


def test_sell_missing_bid_does_not_fire() -> None:
    assert touch_moved_away_from_rejected_price(
        side=Side.SELL,
        rejected_price=2.00,
        best_bid=None,
        best_ask=2.01,
        tick_size=0.01,
    ) is False


def test_buy_missing_bid_irrelevant() -> None:
    """For BUY, best_bid isn't part of the predicate — its absence
    must not block firing."""
    assert touch_moved_away_from_rejected_price(
        side=Side.BUY,
        rejected_price=2.00,
        best_bid=None,
        best_ask=2.01,
        tick_size=0.01,
    ) is True
