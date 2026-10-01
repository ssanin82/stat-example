"""Regression for the ``max_allowed_size`` over-tightening bug.

Reproduced 2026-05-08 in snapshot 260507123509: when toxicity reduced
``decision.quoted_*_sz`` enough that the candidate, after lot-rounding
DOWN, produced a notional below ``min_quote_notional`` AND the
candidate was clipped by ``max_order_notional`` (not by the position
cap), ``_build_side``'s self-heal couldn't bump the size up to clear
``min_quote_notional`` because ``max_allowed_size`` was set to the
post-clip candidate value, not the position-cap-derived ceiling.

Result: engine returned ``mode="no_quote"`` silently; the bot wedged
until the position-drawdown gate fired or the deadlock watchdog hit.

1.1.35 fix: ``_clip_entry_sizes`` now also returns ``max_buy`` and
``max_sell`` (the position-cap-only ceilings), and ``build_quotes``
passes those as ``max_allowed_size`` to ``_build_side``. Self-heal
can now bump up to the actual position cap, while still preventing
the original 2026-04-18 cap-overshoot bug.
"""

from __future__ import annotations

import pytest

from app.enums import ActiveSides, RiskAction, Side
from app.exchange.symbol_spec import SymbolSpec
from app.models import QuoteDecision, ToxicitySnapshot, BestBidAsk
from app.quote_engine import QuoteBuildContext, QuoteEngine
from app.utils.time import utc_now
from tests.settings_helpers import UnitTestSettings


def _settings(**extra: object) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        # TON-shaped config from snapshot 260507123509.
        "QUOTE_NOTIONAL_USD": 10.0,
        "MIN_QUOTE_NOTIONAL_USD": 5.5,
        "MAX_ORDER_NOTIONAL_USD": 10.0,
        "MAX_ABS_POSITION": 4.0,
        "MAX_POSITION_NOTIONAL_USD": 11.0,
        "BASE_HALF_SPREAD_BPS": 2.0,
        "MIN_HALF_SPREAD_BPS": 0.1,
        "MAX_HALF_SPREAD_BPS": 30.0,
        "VOL_MULTIPLIER": 0.0,
        "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 0.0,
        "ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS": 0.0,
        "TOXICITY_SCORE_HALF_SPREAD_BPS": 0.0,
        "MICROPRICE_RESERVATION_ENABLED": False,
    }
    base.update(extra)  # type: ignore[arg-type]
    return UnitTestSettings.model_validate(base)


def _ton_spec() -> SymbolSpec:
    return SymbolSpec(
        price_tick=0.001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=5.0,
        sz_decimals=0,
        source="okx_v5",
    )


def _decision_with_quoted_sizes(*, mid: float, bid_sz: float, ask_sz: float) -> QuoteDecision:
    return QuoteDecision(
        ts=utc_now(),
        symbol="TON-USDT-SWAP",
        mid_price=mid,
        vol_estimate=0.0,
        inventory=0.0,
        reservation_price=mid,
        target_spread_bps=4.0,
        target_bid=mid - 0.001,
        target_ask=mid + 0.001,
        quoted_bid=mid - 0.001,
        quoted_ask=mid + 0.001,
        quoted_bid_sz=bid_sz,
        quoted_ask_sz=ask_sz,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="baseline",
        quote_cycle_id="qc-x",
    )


def _market(mid: float = 2.445) -> BestBidAsk:
    return BestBidAsk(
        symbol="TON-USDT-SWAP",
        best_bid=mid - 0.001,
        best_ask=mid + 0.001,
        mid_price=mid,
        spread_bps=8.0,
        bid_size=100.0,
        ask_size=100.0,
    )


def _ctx(*, position_qty: float, decision: QuoteDecision, market: BestBidAsk) -> QuoteBuildContext:
    return QuoteBuildContext(
        decision=decision,
        market=market,
        risk_action=RiskAction.ALLOW,
        bid_mult=1.0,
        ask_mult=1.0,
        spread_add_bps=0.0,
        position_qty=position_qty,
        position_notional=abs(position_qty) * market.mid_price,
        resting_bid=None,
        resting_ask=None,
        reprice_replace_pending_bid=False,
        reprice_replace_pending_ask=False,
    )


# ---------- Bug reproduction: pre-1.1.35 would return no_quote ---------------


def test_clip_returns_position_cap_maxes_separately() -> None:
    """``_clip_entry_sizes`` exposes the position-cap-derived
    ``max_buy`` / ``max_sell`` separately from the post-intent-clip
    sizes. This was previously implicit; pinning it down ensures
    ``build_quotes`` can pass it through to ``_build_side`` for
    self-heal upper-bound.
    """
    eng = QuoteEngine(_settings(), _ton_spec())
    bid_sz, ask_sz, max_buy, max_sell = eng._clip_entry_sizes(
        position_qty=-1.0,
        bid_sz=2.985,  # toxicity-reduced candidate
        ask_sz=2.985,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=2.445,
        ask_price=2.445,
    )
    # Candidate not clipped (2.985 < both max_buy=5 and max_sell=3 caps).
    assert bid_sz == pytest.approx(2.985)
    # Position cap maxes EXPOSED, not just the intersection.
    # max_buy = MAX_ABS_POSITION - position_qty = 4 - (-1) = 5
    assert max_buy == pytest.approx(5.0, abs=1e-6)
    # max_sell = position_qty + MAX_ABS_POSITION = -1 + 4 = 3
    assert max_sell == pytest.approx(3.0, abs=1e-6)


def test_self_heal_to_min_notional_within_position_cap_succeeds() -> None:
    """The exact scenario from snapshot 260507123509:

    - position = -1 short (sub-spec dust at $2.45/contract)
    - toxicity-reduced candidate bid_sz = 2.985
    - rounds DOWN to 2 contracts → notional $4.89 (below
      MIN_QUOTE_NOTIONAL=$5.5)
    - self-heal needs 3 contracts → $7.34
    - position cap allows up to 5 contracts (max_buy = 4 - (-1))
    - 1.1.35: self-heal succeeds, returns 3 contracts
    - Pre-1.1.35: max_allowed_size=2.985 blocked self-heal → returned
      None with "position_cap_forbids_min_notional", engine wedged.
    """
    eng = QuoteEngine(_settings(), _ton_spec())
    decision = _decision_with_quoted_sizes(mid=2.445, bid_sz=2.985, ask_sz=2.985)
    ctx = _ctx(position_qty=-1.0, decision=decision, market=_market(mid=2.445))
    build = eng.build_quotes(ctx)
    # Bot at -1 with bias active (util=25% > 12% min); SELL gets
    # suppressed because BUY (reducing side) needs to be maintained.
    # But BUY itself MUST be a valid order for the bot to make
    # progress toward zero.
    assert build.bid_order is not None, (
        "BUY side returned None — self-heal blocked by max_allowed_size bug"
    )
    # Self-heal bumped to 3 contracts (from 2 rounded-down) to clear
    # min_quote_notional ($5.5).
    assert build.bid_order.size == pytest.approx(3.0, abs=1e-6)
    assert build.bid_order.size * build.bid_order.price >= 5.5 - 1e-9
    # Position cap not violated: 3 contracts BUY from -1 → 0 ≤ +5 ≤ 4 ✓
    assert build.bid_order.size <= 5.0 + 1e-9


def test_self_heal_does_not_bump_above_max_order_notional() -> None:
    """Even with the position-cap-derived max_allowed_size, self-heal
    still respects MAX_ORDER_NOTIONAL_USD as the per-order ceiling.

    Scenario: tiny QUOTE_NOTIONAL config that produces a sub-min
    candidate; max_order_notional is also small. Self-heal should
    fail with the venue/order-cap mismatch error, not silently
    bump above max_order_notional.
    """
    s = _settings(
        QUOTE_NOTIONAL_USD=2.0,
        MAX_ORDER_NOTIONAL_USD=2.0,
        MIN_QUOTE_NOTIONAL_USD=5.5,  # higher than max_order → impossible
    )
    eng = QuoteEngine(s, _ton_spec())
    decision = _decision_with_quoted_sizes(mid=2.445, bid_sz=0.8, ask_sz=0.8)
    ctx = _ctx(position_qty=0.0, decision=decision, market=_market(mid=2.445))
    build = eng.build_quotes(ctx)
    # Both sides should be None due to min_notional > max_order_notional.
    assert build.bid_order is None
    assert build.ask_order is None
