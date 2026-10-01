"""Cancel benign-missing vs execution_errors; SENT reconcile without spurious DESYNC; SENT timeout."""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from app.enums import OrderStatus, Side
from app.execution import (
    OrderManager,
    interpret_hl_cancel_response,
    make_deterministic_cloid_hex,
)
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings(**kw: object) -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_csh_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS": 60.0,
        "SENT_ORDER_UNRESOLVED_MAX_AMBIGUOUS_POLLS": 100,
    }
    for k, v in kw.items():
        data[k.upper() if k.islower() else k] = v
    return UnitTestSettings.model_validate(data), path


def _benign_cancel_resp() -> dict:
    return {
        "status": "ok",
        "response": {
            "type": "cancel",
            "data": {
                "statuses": [
                    {"error": "Order was never placed, already canceled, or filled."},
                ]
            },
        },
    }


def test_interpret_hl_cancel_benign_missing() -> None:
    k, d = interpret_hl_cancel_response(_benign_cancel_resp())
    assert k == "benign_missing"
    assert "already" in d.lower() or "placed" in d.lower()


def test_cancel_benign_missing_does_not_bump_execution_errors() -> None:
    settings, path = _settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.cancel_order.return_value = _benign_cancel_resp()
    om = OrderManager(settings, client, storage, state, private_event_queue=None)

    wo = WorkingOrder(
        order_id_local="L1",
        order_id_exchange=4242,
        client_order_id=None,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    state.execution_errors = 0
    assert om.cancel_order(wo) is True
    assert state.execution_errors == 0
    path.unlink(missing_ok=True)


def test_cancel_real_exchange_error_still_bumps_execution_errors() -> None:
    settings, path = _settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.cancel_order.return_value = {
        "status": "ok",
        "response": {
            "type": "cancel",
            "data": {"statuses": [{"error": "Some other fatal cancel error xyz"}]},
        },
    }
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    wo = WorkingOrder(
        order_id_local="L1",
        order_id_exchange=4242,
        client_order_id=None,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    state.execution_errors = 0
    assert om.cancel_order(wo) is False
    assert state.execution_errors == 1
    path.unlink(missing_ok=True)


def test_sent_transient_order_status_does_not_desync() -> None:
    settings, path = _settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    cloid = make_deterministic_cloid_hex(settings.symbol, Side.BUY, "q1", 100.0, 0.01)
    wo = WorkingOrder(
        order_id_local="local-sent",
        order_id_exchange=None,
        client_order_id=cloid,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.SENT,
    )
    state.working_bid = wo
    client.query_order_status_by_cloid.side_effect = RuntimeError("transient")
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    remote = HLOpenOrderRaw(999, settings.symbol, Side.BUY, 100.0, 0.01, 0, cloid=None)
    d = om._reconcile_side(Side.BUY, remote)
    assert d is False
    assert state.order_desync is False
    assert wo.status == OrderStatus.SENT
    assert state.working_bid is wo
    path.unlink(missing_ok=True)


def test_sent_unresolved_timeout_releases_slot() -> None:
    settings, path = _settings(
        SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS=1.0,
        SENT_ORDER_UNRESOLVED_MAX_AMBIGUOUS_POLLS=100,
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    cloid = make_deterministic_cloid_hex(settings.symbol, Side.BUY, "q1", 100.0, 0.01)
    past = utc_now() - timedelta(seconds=5)
    wo = WorkingOrder(
        order_id_local="local-old-sent",
        order_id_exchange=None,
        client_order_id=cloid,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.SENT,
        ts_sent=past,
    )
    state.working_bid = wo

    def bad_status(_addr: str, _cloid: str) -> dict:
        return {"status": "wat"}

    client.query_order_status_by_cloid.side_effect = bad_status
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    om._try_resolve_sent_order_by_cloid(wo)
    assert wo.status == OrderStatus.REJECTED
    assert "sent_unresolved" in (wo.cancel_reason or "")
    assert state.working_bid is None
    ev = storage.recent_bot_events(15)
    assert any(e["event_type"] == "sent_order_unresolved_timeout" for e in ev)
    path.unlink(missing_ok=True)


def test_sent_max_ambiguous_polls_releases_without_old_clock() -> None:
    settings, path = _settings(
        SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS=3600.0,
        SENT_ORDER_UNRESOLVED_MAX_AMBIGUOUS_POLLS=3,
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    cloid = make_deterministic_cloid_hex(settings.symbol, Side.SELL, "q1", 100.0, 0.01)
    wo = WorkingOrder(
        order_id_local="poll-cap",
        order_id_exchange=None,
        client_order_id=cloid,
        symbol=settings.symbol,
        side=Side.SELL,
        price=100.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.SENT,
    )
    state.working_ask = wo
    client.query_order_status_by_cloid.return_value = {"status": "nope"}
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    for _ in range(3):
        om._try_resolve_sent_order_by_cloid(wo)
    assert wo.status == OrderStatus.REJECTED
    assert state.working_ask is None
    path.unlink(missing_ok=True)
