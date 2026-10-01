"""Inventory execution bias is applied inside QuoteEngine (final desired orders only)."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import OrderStatus, RiskAction, Side
from app.models import WorkingOrder
from app.quote_engine import QuoteBuildContext, QuoteEngine
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_quote_reprice_maintenance import _decision, _fresh_market


def _engine() -> tuple[QuoteEngine, UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_qeinv_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 10.0,
            # ETH-priced mock symbol ~$3000; 2-unit position = $6000
            # notional. Test isolates the inventory-bias logic, not the
            # position-notional clip, so set the USD cap above $6000.
            "MAX_POSITION_NOTIONAL_USD": 100_000.0,
            "INVENTORY_EXEC_BIAS_RATIO": 0.01,
            "INVENTORY_EXEC_BIAS_MIN_UTIL_PCT": 0.0,
        }
    )
    client = mock_mm_client()
    return QuoteEngine(s, client.symbol_spec), s, path


def _ctx(s: UnitTestSettings, pos_qty: float, *, rb: WorkingOrder | None, ra: WorkingOrder | None) -> QuoteBuildContext:
    return QuoteBuildContext(
        decision=_decision(),
        market=_fresh_market(s),
        risk_action=RiskAction.ALLOW,
        bid_mult=1.0,
        ask_mult=1.0,
        spread_add_bps=0.0,
        position_qty=pos_qty,
        position_notional=abs(pos_qty) * 3000.5,
        resting_bid=rb,
        resting_ask=ra,
    )


def test_long_bias_suppresses_bid_until_sell_maintained() -> None:
    eng, s, path = _engine()
    out = eng.build_quotes(_ctx(s, 2.0, rb=None, ra=None))
    assert out.bid_order is None
    assert out.ask_order is not None
    assert out.ask_order.side == Side.SELL
    path.unlink(missing_ok=True)


def test_long_bias_two_sided_when_sell_resting() -> None:
    eng, s, path = _engine()
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol=s.symbol,
        side=Side.SELL,
        price=3001.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.ACKED,
        quote_cycle_id="q",
    )
    out = eng.build_quotes(_ctx(s, 2.0, rb=None, ra=wo))
    assert out.bid_order is not None
    assert out.ask_order is not None
    path.unlink(missing_ok=True)


def test_short_bias_suppresses_ask_until_buy_maintained() -> None:
    eng, s, path = _engine()
    out = eng.build_quotes(_ctx(s, -2.0, rb=None, ra=None))
    assert out.bid_order is not None
    assert out.ask_order is None
    path.unlink(missing_ok=True)


def test_bias_inactive_below_util_floor() -> None:
    path = Path(tempfile.gettempdir()) / f"mm_qeinv2_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 10.0,
            "MAX_POSITION_NOTIONAL_USD": 100_000.0,
            "INVENTORY_EXEC_BIAS_RATIO": 0.01,
            "INVENTORY_EXEC_BIAS_MIN_UTIL_PCT": 0.12,
        }
    )
    eng = QuoteEngine(s, mock_mm_client().symbol_spec)
    out = eng.build_quotes(_ctx(s, 0.5, rb=None, ra=None))
    assert out.bid_order is not None
    assert out.ask_order is not None
    path.unlink(missing_ok=True)


def test_bias_ratio_zero_is_two_sided() -> None:
    path = Path(tempfile.gettempdir()) / f"mm_qeinv3_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 10.0,
            "MAX_POSITION_NOTIONAL_USD": 100_000.0,
            "INVENTORY_EXEC_BIAS_RATIO": 0.0,
        }
    )
    eng = QuoteEngine(s, mock_mm_client().symbol_spec)
    out = eng.build_quotes(_ctx(s, 5.0, rb=None, ra=None))
    assert out.bid_order is not None
    assert out.ask_order is not None
    path.unlink(missing_ok=True)


def test_engine_outputs_executable_rounded_orders() -> None:
    eng, s, path = _engine()
    out = eng.build_quotes(_ctx(s, 0.0, rb=None, ra=None))
    if out.bid_order:
        assert out.bid_order.price > 0
        assert out.bid_order.size > 0
        assert out.bid_order.price * out.bid_order.size + 1e-9 >= float(s.min_quote_notional_usd)
    if out.ask_order:
        assert out.ask_order.price > 0
        assert out.ask_order.size > 0
    path.unlink(missing_ok=True)
