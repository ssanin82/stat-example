"""Execution-layer order slotting, cancel retries, cloid reconcile, and position clip."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from app.enums import OrderStatus, Side
from app.execution import (
    OrderManager,
    interpret_hl_order_status_response,
    make_deterministic_cloid_hex,
)
from app.models import BestBidAsk, WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_settings() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_ordst_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 0.05,
        }
    )
    return s, path


def test_cancel_api_failure_leaves_cancel_pending_and_blocks_second_place() -> None:
    settings, path = _db_settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state, private_event_queue=None)

    wo = WorkingOrder(
        order_id_local="L1",
        order_id_exchange=999001,
        client_order_id="0x" + "ab" * 16,
        symbol=settings.symbol,
        side=Side.BUY,
        price=3000.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    state.working_bid = wo

    client.cancel_order.side_effect = RuntimeError("transport down")
    assert om.cancel_order(wo) is False
    assert wo.status == OrderStatus.CANCEL_PENDING
    assert state.working_bid is wo

    blocked = om.place_passive_order_manual_only(Side.BUY, 3000.0, 0.01, "c1")
    assert blocked is None
    client.cancel_order.assert_called()

    path.unlink(missing_ok=True)


def test_cancel_pending_retries_cancel_rpc() -> None:
    settings, path = _db_settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state, private_event_queue=None)

    wo = WorkingOrder(
        order_id_local="L1",
        order_id_exchange=999002,
        client_order_id=None,
        symbol=settings.symbol,
        side=Side.BUY,
        price=3000.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    _ok_cancel = {
        "status": "ok",
        "response": {"type": "cancel", "data": {"statuses": ["success"]}},
    }
    client.cancel_order.side_effect = [RuntimeError("fail once"), _ok_cancel]

    assert om.cancel_order(wo) is False
    assert wo.status == OrderStatus.CANCEL_PENDING
    assert om.cancel_order(wo) is True
    assert client.cancel_order.call_count == 2

    path.unlink(missing_ok=True)


def test_unconfirmed_place_same_cloid_reconcile_opens_without_duplicate_place() -> None:
    settings, path = _db_settings()
    # 2026-05-14 BUG-024: this test exercises the legacy HL
    # ``waitingForFill`` recovery path — reconcile-by-cloid resolves
    # the order to ACKED after the unconfirmed initial place. The
    # new strict-kill guard (default True) would CRITICAL-kill the
    # bot on the initial unconfirmed instead. Opt out per-test so
    # the legacy reconcile-recovery behaviour remains tested for HL.
    settings = settings.model_copy(update={"strict_place_unconfirmed_kill": False})
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True

    cloid = make_deterministic_cloid_hex(settings.symbol, Side.BUY, "qc", 3000.0, 0.01)

    unconfirmed = {
        "status": "ok",
        "response": {"type": "order", "data": {"statuses": ["waitingForFill"]}},
    }
    client.fetch_open_orders_raw.return_value = []
    client.place_post_only_limit.return_value = unconfirmed
    client.query_order_status_by_cloid.return_value = {
        "status": "order",
        "order": {
            "order": {"oid": 7711, "coin": settings.symbol, "side": "B"},
            "status": "open",
            "statusTimestamp": 1,
        },
    }

    state.market = BestBidAsk(
        symbol=settings.symbol,
        best_bid=2990.0,
        best_ask=3010.0,
        mid_price=3000.0,
        spread_bps=50.0,
    )
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    wo = om.place_passive_order_manual_only(Side.BUY, 3000.0, 0.01, "qc")
    assert wo is not None
    assert wo.status == OrderStatus.SENT
    assert wo.client_order_id == cloid
    assert state.working_bid is wo
    assert client.place_post_only_limit.call_count == 1

    blocked = om.place_passive_order_manual_only(Side.BUY, 3000.0, 0.01, "qc2")
    assert blocked is None

    om.sync_open_orders(force=True, emergency=True)
    assert wo.status == OrderStatus.ACKED
    assert wo.order_id_exchange == 7711
    assert client.query_order_status_by_cloid.call_count >= 1

    path.unlink(missing_ok=True)


def test_execution_layer_clip_applies_to_sent_working_bid() -> None:
    """SENT consumes headroom; ACKED slot alone does not (single-slot cancel/replace)."""
    settings, path = _db_settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state, private_event_queue=None)

    from app.enums import ActiveSides
    from app.models import QuoteDecision
    from app.utils.time import utc_now

    state.position.position_qty = 0.04
    sent = WorkingOrder(
        order_id_local="L",
        order_id_exchange=None,
        client_order_id="0x" + "cd" * 16,
        symbol=settings.symbol,
        side=Side.BUY,
        price=3000.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.SENT,
    )
    state.working_bid = sent

    d = QuoteDecision(
        ts=utc_now(),
        symbol=settings.symbol,
        mid_price=100.0,
        vol_estimate=1.0,
        inventory=0.04,
        reservation_price=100.0,
        target_spread_bps=5.0,
        target_bid=99.0,
        target_ask=101.0,
        quoted_bid=99.0,
        quoted_ask=101.0,
        quoted_bid_sz=0.02,
        quoted_ask_sz=0.02,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="t",
        quote_cycle_id="qc",
    )
    _, b_sent, _, _ = om._diagnostic_model_quote_prices_sizes(d, 1.0, 1.0, 0.0)
    assert b_sent <= 1e-12

    sent.status = OrderStatus.ACKED
    _, b_acked, _, _ = om._diagnostic_model_quote_prices_sizes(d, 1.0, 1.0, 0.0)
    assert abs(b_acked - 0.01) < 1e-9

    path.unlink(missing_ok=True)


def test_clip_entry_sizes_reduces_headroom_with_resting_working_orders() -> None:
    settings, path = _db_settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state, private_event_queue=None)

    b, a = om._clip_entry_sizes(
        0.04,
        0.02,
        0.02,
        resting_bid_sz=0.01,
        resting_ask_sz=0.0,
    )
    assert abs(b - 0.0) < 1e-9
    assert abs(a - 0.02) < 1e-9

    b2, a2 = om._clip_entry_sizes(
        0.04,
        0.02,
        0.02,
        resting_bid_sz=0.0,
        resting_ask_sz=0.015,
    )
    assert abs(b2 - 0.01) < 1e-9
    assert abs(a2 - 0.02) < 1e-9

    path.unlink(missing_ok=True)


def test_interpret_hl_order_status_unknown_oid() -> None:
    oid, out, _ = interpret_hl_order_status_response({"status": "unknownOid"})
    assert oid is None and out == "not_found"


def test_interpret_hl_order_status_open() -> None:
    oid, out, _ = interpret_hl_order_status_response(
        {
            "status": "order",
            "order": {
                "order": {"oid": 42},
                "status": "open",
                "statusTimestamp": 0,
            },
        }
    )
    assert oid == 42 and out == "open"
