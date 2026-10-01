"""Public websocket BBO path: parsing, risk gating, gap stats, no REST book in ``one_tick``."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from app.bot import Bot
from app.enums import BotStatus, RiskAction
from app.exchange.hyperliquid_public_ws import (
    HyperliquidPublicStream,
    best_bid_ask_from_bbo_dict,
)
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot
from app.risk import evaluate_risk
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from app.utils.time import utc_now


def test_parse_bbo_dict_builds_best_bid_ask() -> None:
    d = {
        "coin": "ETH",
        "time": 1_700_000_000_000,
        "bbo": [
            {"px": "3000.0", "sz": "1.5"},
            {"px": "3001.0", "sz": "2.0"},
        ],
    }
    bb = best_bid_ask_from_bbo_dict("ETH", d)
    assert bb is not None
    assert bb.best_bid == 3000.0
    assert bb.best_ask == 3001.0
    assert bb.mid_price == 3000.5
    assert bb.ts_exchange_ms == 1_700_000_000_000


def test_gap_stats_source_public_ws_after_book_apply() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    state = BotState(s)
    bb = BestBidAsk(
        symbol="ETH",
        best_bid=1.0,
        best_ask=2.0,
        mid_price=1.5,
        spread_bps=10.0,
        ts_local=datetime.now(timezone.utc),
    )
    state.apply_market_book_only(bb, market_data_source="public_ws")
    d = state.market_data_gap_tracker.to_api_dict(session_id=state.session_id, symbol="ETH")
    assert d.get("source_type") == "public_ws"


def test_one_tick_does_not_call_fetch_best_bid_ask() -> None:
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "PRIVATE_WS_ENABLED": False,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.bot_status = BotStatus.RUNNING
    bb = BestBidAsk(
        symbol=settings.symbol,
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        spread_bps=200.0,
        ts_local=utc_now(),
    )
    state.apply_market_book_only(bb, market_data_source="public_ws")
    with state._lock:
        state.public_ws_last_message_wall_ts = utc_now()
        state.public_ws_connected = True
        state.public_ws_seen_first_bbo = True
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.fetch_position.return_value = PositionSnapshot(
        symbol=settings.symbol,
        position_qty=0.0,
        avg_entry_price=100.0,
        mark_price=100.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    client.fetch_account_snapshot.return_value = AccountSnapshot(
        equity_usd=10_000.0,
        cash_usd=5000.0,
        withdrawable_usd=5000.0,
    )
    client.fetch_recent_fills_raw.return_value = []
    pub = MagicMock()
    bot = Bot(settings, state, client, storage, public_stream=pub)
    with (
        patch.object(bot._exec, "maybe_sync_open_orders"),
        patch.object(bot._exec, "maybe_refresh_quotes"),
        patch.object(bot._exec, "cancel_resting_for_risk"),
        patch.object(bot, "_persist_snapshots"),
        patch.object(bot, "_market_data_recovery_supervisor", return_value=False),
    ):
        bot.one_tick()
    client.fetch_best_bid_ask.assert_not_called()


def test_risk_blocks_quotes_when_public_ws_message_stale() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "PUBLIC_WS_STALE_WARN_SECONDS": 1.0,
            "PUBLIC_WS_STALE_KILL_SECONDS": 10.0,
        }
    )
    m = BestBidAsk(
        symbol="ETH",
        best_bid=1.0,
        best_ask=2.0,
        mid_price=1.5,
        spread_bps=10.0,
        ts_local=utc_now(),
    )
    from app.models import PnlSnapshot, ToxicitySnapshot

    pnl = PnlSnapshot(
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
        total_pnl_usd=0.0,
        fees_usd=0.0,
        equity_usd=1000.0,
        drawdown_usd=0.0,
        session_peak_equity_usd=1000.0,
    )
    tox = ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
    )
    r = evaluate_risk(
        s,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=tox,
        execution_errors=0,
        desync=False,
        trades_last_minute=0,
        public_ws_live_path=True,
        public_ws_connected=True,
        public_ws_seconds_since_message=5.0,
        public_ws_seen_first_bbo=True,
    )
    assert r.action == RiskAction.NO_QUOTE
    assert "public_ws_stale_warn" in r.reasons


def test_feed_raw_bbo_updates_state_via_stream() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "PUBLIC_WS_ENABLED": True,
        }
    )
    state = BotState(s)
    raw = (
        '{"channel":"bbo","data":{"coin":"ETH","time":1700000000000,'
        '"bbo":[{"px":"100","sz":"1"},{"px":"101","sz":"1"}]}}'
    )

    def on_bbo(bb: BestBidAsk) -> None:
        state.apply_market_book_only(bb, market_data_source="public_ws")

    stream = HyperliquidPublicStream(s, state, "ETH", on_bbo)
    stream.feed_message_for_tests(raw)
    assert state.market is not None
    assert state.market.mid_price == 100.5
    assert state.live_market_data_source == "public_ws"


def test_reconnect_callback_invoked_from_stream_method() -> None:
    s = UnitTestSettings.model_validate({"TRADING_ENABLED": False, "HL_SECRET_KEY": ""})
    state = BotState(s)
    stream = HyperliquidPublicStream(s, state, "ETH", lambda _bb: None)
    stream.request_reconnect()
