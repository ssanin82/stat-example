from datetime import timedelta

from app.enums import BotStatus, RiskAction
from app.models import BestBidAsk, PnlSnapshot, ToxicitySnapshot
from app.risk import evaluate_risk
from app.utils.time import utc_now

from tests.settings_helpers import UnitTestSettings


def _s(trading: bool = False, **kw: float | bool | int) -> UnitTestSettings:
    # Use validation aliases so values are not dropped by pydantic-settings.
    data: dict = {
        "TRADING_ENABLED": trading,
        "HL_SECRET_KEY": "0x" + "11" * 32 if trading else "",
        "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20 if trading else "",
    }
    alias_map = {
        "stale_data_kill_seconds": "STALE_DATA_KILL_SECONDS",
        "stale_data_warn_seconds": "STALE_DATA_WARN_SECONDS",
        "max_session_loss_usd": "MAX_SESSION_LOSS_USD",
        "max_abs_position": "MAX_ABS_POSITION",
        "max_position_notional_usd": "MAX_POSITION_NOTIONAL_USD",
    }
    for k, v in kw.items():
        data[alias_map.get(k, k.upper())] = v
    return UnitTestSettings.model_validate(data)


def test_kill_when_session_loss() -> None:
    settings = _s(trading=True)
    pnl = PnlSnapshot(
        realized_pnl_usd=-600.0,
        unrealized_pnl_usd=0.0,
        total_pnl_usd=-600.0,
        fees_usd=1.0,
        equity_usd=1000.0,
        drawdown_usd=0.0,
        session_peak_equity_usd=2000.0,
        ts=utc_now(),
    )
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=None,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
    )
    assert r.action.value == "KILL"


def test_no_quote_trading_disabled() -> None:
    settings = _s()
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=None,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
    )
    assert r.action.value == "NO_QUOTE"


def test_no_quote_when_no_market_and_trading() -> None:
    settings = _s(trading=True)
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=None,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
    )
    assert r.action.value == "NO_QUOTE"
    assert "no_market_data" in r.reasons


def test_stale_data_kill() -> None:
    settings = _s(
        trading=True,
        stale_data_kill_seconds=5.0,
        market_data_recovery_enabled=False,
    )
    old = utc_now() - timedelta(seconds=10)
    m = BestBidAsk("ETH", 1.0, 1.1, 1.05, 10.0, ts_local=old)
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
    )
    assert r.action.value == "KILL"


def test_stale_data_past_kill_with_recovery_returns_no_quote_not_kill() -> None:
    settings = _s(trading=True, stale_data_kill_seconds=5.0, market_data_recovery_enabled=True)
    old = utc_now() - timedelta(seconds=10)
    m = BestBidAsk("ETH", 1.0, 1.1, 1.05, 10.0, ts_local=old)
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
    )
    assert r.action.value == "NO_QUOTE"
    assert "stale_market_recovery" in r.reasons


def test_over_notional_long_is_ask_only() -> None:
    settings = _s(
        trading=True,
        max_abs_position=1.0,
        max_position_notional_usd=250.0,
    )
    m = BestBidAsk(
        "ETH",
        3000.0,
        3001.0,
        3000.5,
        10.0,
        ts_local=utc_now(),
    )
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=0.02,
        position_notional=300.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
    )
    assert r.action == RiskAction.ASK_ONLY
    assert "max_position_notional" in r.reasons


def test_over_notional_short_is_bid_only() -> None:
    settings = _s(
        trading=True,
        max_abs_position=1.0,
        max_position_notional_usd=250.0,
    )
    m = BestBidAsk(
        "ETH",
        3000.0,
        3001.0,
        3000.5,
        10.0,
        ts_local=utc_now(),
    )
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=-0.02,
        position_notional=300.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
    )
    assert r.action == RiskAction.BID_ONLY
    assert "max_position_notional" in r.reasons


def test_trade_rate_limit_no_quote() -> None:
    settings = _s(
        trading=True,
        max_trades_per_minute=5,
    )
    m = BestBidAsk(
        "ETH",
        3000.0,
        3001.0,
        3000.5,
        10.0,
        ts_local=utc_now(),
    )
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
        trades_last_minute=10,
    )
    assert r.action.value == "NO_QUOTE"
    assert "trade_rate_limit" in r.reasons


def test_stale_data_warn_is_no_quote() -> None:
    settings = _s(
        trading=True,
        stale_data_warn_seconds=2.0,
        stale_data_kill_seconds=30.0,
    )
    old = utc_now() - timedelta(seconds=5)
    m = BestBidAsk("ETH", 3000.0, 3001.0, 3000.5, 10.0, ts_local=old)
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    r = evaluate_risk(
        settings,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
    )
    assert r.action.value == "NO_QUOTE"
    assert "stale_data_warn" in r.reasons
