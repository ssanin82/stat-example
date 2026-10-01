"""One-model invariant: when ``normal_mode`` is active, the market-anchored
spread it produces is NOT overridden by ``apply_profitability_spread_floor``.

Regression for ``tmp/snap_20260418_094415``. Observed: normal_mm produced
tight market-aware quotes (half ~0.13 bps / 3 ticks on GRVT's 1-tick ETH
book), then ``apply_profitability_spread_floor`` silently widened to half
~5.8 bps (138 ticks behind BBO), orders sat invisibly for 17 minutes. The
layered disagreement violated the user's "one model" directive.

Contract: if ``normal_mm`` fires (two-sided + ALLOW risk + market_spread_anchor
enabled + book fresh), the returned quote spread must track the market anchor,
not the economic floor.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

from app.enums import ActiveSides, RiskAction, Side
from app.models import BestBidAsk, QuoteDecision
from app.quote_engine import QuoteBuildContext, QuoteEngine
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _tight_book_settings() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_nmm_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            # Mimic the snap_20260418_094415 environment:
            "NORMAL_MM_USE_MARKET_SPREAD_ANCHOR": True,
            "NORMAL_MM_TOUCH_BUFFER_TICKS": 1.0,
            "NORMAL_MM_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "NORMAL_MM_MIN_COMPETITIVE_HALF_SPREAD_BPS": 0.3,
            # Economic floor set deliberately high to reveal whether the engine
            # respects normal_mm's tight spread (correct) or widens to the floor
            # (the layered-disagreement bug).
            "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 5.0,
            "ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS": 5.0,
            "MIN_HALF_SPREAD_BPS": 0.05,
            # Adverse overlay armed → pre-fix this widened to mid ± 5+ bps:
            "ADAPTIVE_SPREAD_ADVERSE_OVERLAY_HALF_SPREAD_BPS": 4.0,
            "ECONOMIC_TOXICITY_SCORE_HALF_SPREAD_BPS": 3.0,
            "MAX_ABS_POSITION": 10.0,
        }
    )
    return s, path


def _tight_book_decision(mid: float = 2375.725) -> QuoteDecision:
    """Strategy decision as produced by compute_quote_decision for a neutral,
    two-sided quote intent. The overlay field reflects what bot.py would inject
    when the adverse-spread-widen is armed."""
    return QuoteDecision(
        ts=utc_now(),
        symbol="ETH_USDT_Perp",
        mid_price=mid,
        vol_estimate=0.5,
        inventory=0.0,
        reservation_price=mid,
        target_spread_bps=12.0,  # wide, from the model math
        target_bid=mid - 1.0,
        target_ask=mid + 1.0,
        quoted_bid=mid - 1.0,
        quoted_ask=mid + 1.0,
        quoted_bid_sz=0.01,
        quoted_ask_sz=0.01,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.25,
        decision_reason="baseline,adaptive_spread_widen",
        quote_cycle_id="qc1",
        spread_floor_overlay_half_spread_bps=4.0,  # overlay armed
    )


def _tight_market(mid: float = 2375.725) -> BestBidAsk:
    """A 1-tick-wide book: best_bid = mid-0.005, best_ask = mid+0.005."""
    return BestBidAsk(
        symbol="ETH_USDT_Perp",
        best_bid=round(mid - 0.005, 2),  # 2375.72
        best_ask=round(mid + 0.005, 2),  # 2375.73
        mid_price=mid,
        spread_bps=0.042,
        ts_local=utc_now(),
    )


def _ctx(s: UnitTestSettings, *, pos_qty: float = 0.0) -> QuoteBuildContext:
    return QuoteBuildContext(
        decision=_tight_book_decision(),
        market=_tight_market(),
        risk_action=RiskAction.ALLOW,
        bid_mult=1.0,
        ask_mult=1.0,
        spread_add_bps=0.0,
        position_qty=pos_qty,
        position_notional=abs(pos_qty) * 2375.725,
        resting_bid=None,
        resting_ask=None,
    )


def test_normal_mm_quotes_hug_the_touch_not_the_economic_floor() -> None:
    """On a 1-tick book with overlay armed, engine output must stay within a
    few ticks of BBO — NOT at mid ± 5 bps = 138 ticks back."""
    s, path = _tight_book_settings()
    try:
        client = mock_mm_client()
        # Match the spec to a 1-tick book (GRVT ETH).
        from app.exchange.symbol_spec import SymbolSpec

        client.symbol_spec = SymbolSpec(
            price_tick=0.01,
            size_step=0.001,
            min_size=0.001,
            min_notional_usd=20.0,
            sz_decimals=9,
            source="grvt_meta",
        )
        eng = QuoteEngine(s, client.symbol_spec)
        result = eng.build_quotes(_ctx(s))

        assert result.bid_order is not None
        assert result.ask_order is not None
        mkt = _tight_market()
        mid = mkt.mid_price
        bid_dist_ticks = (mid - result.bid_order.price) / 0.01
        ask_dist_ticks = (result.ask_order.price - mid) / 0.01
        # The "bug" observed in snap_20260418_094415 put these at ~138 ticks.
        # Normal MM on a 1-tick book with 3-tick touch budget should be
        # within ~3 ticks of mid.
        assert bid_dist_ticks <= 5.0, f"bid {bid_dist_ticks} ticks from mid — widened past normal_mm cap"
        assert ask_dist_ticks <= 5.0, f"ask {ask_dist_ticks} ticks from mid — widened past normal_mm cap"
    finally:
        path.unlink(missing_ok=True)


def test_one_sided_mode_still_applies_economic_floor() -> None:
    """When normal_mode doesn't fire (e.g., one-sided intent), the economic
    floor IS the authoritative spread-setter. This preserves the safety gate
    for degraded modes while only exempting normal_mm."""
    s, path = _tight_book_settings()
    try:
        client = mock_mm_client()
        from app.exchange.symbol_spec import SymbolSpec

        client.symbol_spec = SymbolSpec(
            price_tick=0.01,
            size_step=0.001,
            min_size=0.001,
            min_notional_usd=20.0,
            sz_decimals=9,
            source="grvt_meta",
        )
        eng = QuoteEngine(s, client.symbol_spec)
        # Force one-sided via active_sides = ASK_ONLY → want_bid=False.
        ctx = _ctx(s)
        one_sided_decision = replace(ctx.decision, active_sides=ActiveSides.ASK_ONLY)
        ctx = replace(ctx, decision=one_sided_decision)
        result = eng.build_quotes(ctx)

        # One-sided: normal_mode does NOT fire (requires two_sided=True), so
        # the economic floor widens the ask to mid + economic_half_px. That's
        # 5 bps + 3 bps overlay + toxicity bump ≈ well above mid+3 ticks.
        assert result.bid_order is None
        assert result.ask_order is not None
        mid = ctx.market.mid_price
        ask_dist_bps = (result.ask_order.price - mid) / mid * 10_000.0
        # Economic floor enforced; >3 ticks (0.126 bps) from mid expected.
        assert ask_dist_bps > 0.5, f"economic floor not applied: ask at {ask_dist_bps:.3f} bps"
    finally:
        path.unlink(missing_ok=True)


def test_stale_book_falls_back_to_floor() -> None:
    """Book-not-fresh disables normal_mode → economic floor takes over."""
    s, path = _tight_book_settings()
    try:
        client = mock_mm_client()
        from app.exchange.symbol_spec import SymbolSpec

        client.symbol_spec = SymbolSpec(
            price_tick=0.01,
            size_step=0.001,
            min_size=0.001,
            min_notional_usd=20.0,
            sz_decimals=9,
            source="grvt_meta",
        )
        eng = QuoteEngine(s, client.symbol_spec)
        ctx = _ctx(s)
        # Mark the market as stale by backdating ts_local far beyond warn threshold.
        from datetime import timedelta

        stale_market = replace(
            ctx.market, ts_local=ctx.market.ts_local - timedelta(seconds=3600)
        )
        ctx = replace(ctx, market=stale_market)
        result = eng.build_quotes(ctx)
        # On stale book normal_mode is False, so economic floor applies.
        # We should still get widely-spaced orders (the floor path) not the
        # tight normal_mm path. The important invariant: the engine doesn't
        # crash on the stale path, and the spread respects the floor.
        if result.bid_order is not None and result.ask_order is not None:
            mid = ctx.market.mid_price
            gross_bps = (result.ask_order.price - result.bid_order.price) / mid * 10_000.0
            # floor = 2*5 = 10 bps minimum from economic floor
            assert gross_bps >= 5.0, f"economic floor not applied on stale book: {gross_bps} bps"
    finally:
        path.unlink(missing_ok=True)
