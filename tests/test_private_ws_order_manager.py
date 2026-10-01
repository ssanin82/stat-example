"""OrderManager private queue, fills, order updates, reconcile gating."""

from __future__ import annotations

import os
import queue
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.exchange.hyperliquid_types import HLFillRaw
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
    PrivateWsConnectionEvent,
    PrivateWsConnectionKind,
)
from app.fill_ingestion import ingest_hl_fill_raw
from app.models import BestBidAsk, WorkingOrder
from app.pnl import PnlTracker
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_settings(**kw: object) -> UnitTestSettings:
    path = Path(tempfile.gettempdir()) / f"mm_privws_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "PRIVATE_WS_ENABLED": True,
        "PRIVATE_WS_RECOVERY_FILL_CATCHUP_TICKS": 3,
        "OPEN_ORDERS_RECONCILE_INTERVAL_TICKS": 100,
        "CANCEL_PENDING_REST_WATCHDOG_TICKS": 5,
        "REST_FILL_RECONCILE_INTERVAL_TICKS": 0,
    }
    for k, v in kw.items():
        data[k.upper() if k.islower() else k] = v
    s = UnitTestSettings.model_validate(data)
    return s


def _make_om(
    settings: UnitTestSettings,
    pq: queue.Queue | None = None,
) -> tuple[OrderManager, BotState, Storage, Path]:
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    # Tests use small historical time_ms values; treat them as in-session vs epoch start.
    state.session_started_at_utc = datetime(1970, 1, 1, tzinfo=timezone.utc)
    client = mock_mm_client()
    om = OrderManager(settings, client, storage, state, private_event_queue=pq)
    db_path = Path(settings.effective_sqlite_path())
    return om, state, storage, db_path


def _market() -> BestBidAsk:
    return BestBidAsk(
        symbol="ETH",
        best_bid=3000.0,
        best_ask=3001.0,
        mid_price=3000.5,
        spread_bps=3.0,
    )


def _wo(
    settings: UnitTestSettings,
    oid: int,
    *,
    side: Side = Side.BUY,
    status: OrderStatus = OrderStatus.ACKED,
    size: float = 0.1,
) -> WorkingOrder:
    return WorkingOrder(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=oid,
        client_order_id=None,
        symbol=settings.symbol,
        side=side,
        price=3000.0,
        size=size,
        post_only=True,
        status=status,
        quote_cycle_id="q1",
    )


def _order_update(
    *,
    coin: str = "ETH",
    oid: int,
    status: str,
    ts: int,
    rem: float,
    orig: float,
) -> PrivateOrderUpdateEvent:
    return PrivateOrderUpdateEvent(
        oid=oid,
        coin=coin,
        status=status,
        status_timestamp_ms=ts,
        side="B",
        limit_px=3000.0,
        remaining_sz=rem,
        orig_sz=orig,
        raw_status=status,
    )


def test_drain_empty_queue_noop() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    om.on_bot_tick_start()
    om.drain_private_events(None)
    path.unlink(missing_ok=True)


def test_drain_dispatches_connection_fill_order_events() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.CONNECTED, detail="t"))
    pq.put(
        PrivateFillEvent(
            fill_id="a_1",
            oid=1,
            coin=s.symbol,
            px=1.0,
            sz=2.0,
            side="B",
            time_ms=1,
            fee=0.0,
            closed_pnl=0.0,
            crossed=False,
            is_snapshot=False,
            raw={},
        )
    )
    pq.put(_order_update(oid=9, status="open", ts=1, rem=0.1, orig=0.1))
    om.on_bot_tick_start()
    state.apply_market_snapshot(_market(), state.position, None)
    om.drain_private_events(None)
    assert state.private_ws_connected is True
    assert state.private_ws_healthy is True
    path.unlink(missing_ok=True)


def test_drain_respects_per_tick_cap_leaves_remainder() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    for i in range(5):
        pq.put(
            PrivateFillEvent(
                fill_id=f"x_{i}",
                oid=i,
                coin=s.symbol,
                px=1.0,
                sz=1.0,
                side="B",
                time_ms=i,
                fee=0.0,
                closed_pnl=0.0,
                crossed=False,
                is_snapshot=False,
                raw={},
            )
        )
    om.on_bot_tick_start()
    state.apply_market_snapshot(_market(), state.position, None)
    with patch("app.execution._MAX_PRIVATE_EVENTS_PER_TICK", 2):
        om.drain_private_events(None)
    assert pq.qsize() == 3
    with patch("app.execution._MAX_PRIVATE_EVENTS_PER_TICK", 2):
        om.drain_private_events(None)
    assert pq.qsize() == 1
    path.unlink(missing_ok=True)


def test_private_fill_ingested_via_shared_path() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    om.on_bot_tick_start()
    state.apply_market_snapshot(_market(), state.position, None)
    pq.put(
        PrivateFillEvent(
            fill_id="h_500",
            oid=1,
            coin=s.symbol,
            px=3000.0,
            sz=0.01,
            side="B",
            time_ms=500_000,
            fee=-0.001,
            closed_pnl=0.0,
            crossed=True,
            is_snapshot=False,
            raw={"crossed": True},
        )
    )
    pnl = PnlTracker()
    om.drain_private_events(pnl)
    with state._lock:
        assert len(state.recent_fills) == 1
        assert state.recent_fills[0].fill_id == "h_500"
    path.unlink(missing_ok=True)


def test_private_fill_wrong_symbol_ignored() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    state.apply_market_snapshot(_market(), state.position, None)
    pq.put(
        PrivateFillEvent(
            fill_id="h_1",
            oid=1,
            coin="BTC",
            px=1.0,
            sz=1.0,
            side="B",
            time_ms=1,
            fee=0.0,
            closed_pnl=0.0,
            crossed=False,
            is_snapshot=False,
            raw={},
        )
    )
    om.drain_private_events(None)
    assert len(state.recent_fills) == 0
    path.unlink(missing_ok=True)


def test_private_fill_duplicate_idempotent() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    state.apply_market_snapshot(_market(), state.position, None)
    ev = PrivateFillEvent(
        fill_id="same",
        oid=1,
        coin=s.symbol,
        px=2.0,
        sz=3.0,
        side="B",
        time_ms=100,
        fee=0.0,
        closed_pnl=0.0,
        crossed=False,
        is_snapshot=False,
        raw={},
    )
    pq.put(ev)
    pq.put(ev)
    om.drain_private_events(None)
    with state._lock:
        assert len(state.recent_fills) == 1
    path.unlink(missing_ok=True)


def test_ws_and_rest_fill_same_downstream_state() -> None:
    s = _db_settings()
    path = Path(s.effective_sqlite_path())
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.session_started_at_utc = datetime(1970, 1, 1, tzinfo=timezone.utc)
    state.apply_market_snapshot(_market(), state.position, None)
    fr = HLFillRaw(
        fill_id="rest_ws",
        oid=10,
        coin=s.symbol,
        side=Side.BUY,
        px=100.0,
        sz=0.05,
        fee=0.01,
        time_ms=888_000,
        closed_pnl=0.0,
        raw={},
    )
    ingest_hl_fill_raw(
        state=state,
        storage=None,
        pnl=None,
        symbol=s.symbol,
        fr=fr,
        source="rest",
    )
    with state._lock:
        first = state.recent_fills[0]
        px1, sz1 = first.price, first.size
    ingest_hl_fill_raw(
        state=state,
        storage=None,
        pnl=None,
        symbol=s.symbol,
        fr=fr,
        source="rest",
    )
    with state._lock:
        assert len(state.recent_fills) == 1
        assert state.recent_fills[0].price == px1
    path.unlink(missing_ok=True)


def test_rest_after_ws_same_fill_id_no_double_count() -> None:
    s = _db_settings()
    pq = queue.Queue()
    om, state, storage, path = _make_om(s, pq)
    state.apply_market_snapshot(_market(), state.position, None)
    pq.put(
        PrivateFillEvent(
            fill_id="dup_1",
            oid=1,
            coin=s.symbol,
            px=1.0,
            sz=1.0,
            side="B",
            time_ms=100,
            fee=0.0,
            closed_pnl=0.0,
            crossed=False,
            is_snapshot=False,
            raw={},
        )
    )
    om.drain_private_events(None)
    fr = HLFillRaw(
        fill_id="dup_1",
        oid=1,
        coin=s.symbol,
        side=Side.BUY,
        px=1.0,
        sz=1.0,
        fee=0.0,
        time_ms=100,
        closed_pnl=0.0,
        raw={},
    )
    ingest_hl_fill_raw(
        state=state,
        storage=None,
        pnl=None,
        symbol=s.symbol,
        fr=fr,
        source="rest",
    )
    with state._lock:
        assert len(state.recent_fills) == 1
    path.unlink(missing_ok=True)


def test_order_update_open_acks_sent_order() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 100, status=OrderStatus.SENT)
    state.working_bid = wo
    pq.put(_order_update(oid=100, status="open", ts=10, rem=0.1, orig=0.1))
    om.drain_private_events(None)
    assert wo.status == OrderStatus.ACKED
    path.unlink(missing_ok=True)


def test_order_update_partial_reduces_size() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 200, status=OrderStatus.ACKED, size=0.1)
    state.working_bid = wo
    pq.put(_order_update(oid=200, status="open", ts=20, rem=0.03, orig=0.1))
    om.drain_private_events(None)
    assert wo.status == OrderStatus.PARTIAL
    assert abs(wo.size - 0.03) < 1e-9
    path.unlink(missing_ok=True)


def test_order_update_canceled_clears_cancel_pending() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 300, status=OrderStatus.CANCEL_PENDING)
    state.working_bid = wo
    pq.put(_order_update(oid=300, status="canceled", ts=30, rem=0.0, orig=0.1))
    om.drain_private_events(None)
    assert state.working_bid is None
    assert wo.status == OrderStatus.CANCELED
    path.unlink(missing_ok=True)


def test_order_update_filled_clears_working() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 400, status=OrderStatus.ACKED)
    state.working_bid = wo
    pq.put(_order_update(oid=400, status="filled", ts=40, rem=0.0, orig=0.1))
    om.drain_private_events(None)
    assert state.working_bid is None
    assert wo.status == OrderStatus.FILLED
    path.unlink(missing_ok=True)


def test_order_update_rejected_clears_working() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 500, status=OrderStatus.SENT, side=Side.SELL)
    state.working_ask = wo
    pq.put(_order_update(oid=500, status="rejected", ts=50, rem=0.0, orig=0.1))
    om.drain_private_events(None)
    assert state.working_ask is None
    assert wo.status == OrderStatus.REJECTED
    path.unlink(missing_ok=True)


def test_order_update_unknown_oid_ignored() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 600, status=OrderStatus.ACKED)
    state.working_bid = wo
    pq.put(_order_update(oid=9999, status="open", ts=60, rem=0.1, orig=0.1))
    om.drain_private_events(None)
    assert state.working_bid is wo
    assert wo.status == OrderStatus.ACKED
    path.unlink(missing_ok=True)


def test_order_update_wrong_symbol_ignored() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 700, status=OrderStatus.ACKED)
    state.working_bid = wo
    pq.put(_order_update(coin="BTC", oid=700, status="open", ts=70, rem=0.05, orig=0.1))
    om.drain_private_events(None)
    assert wo.size == 0.1
    path.unlink(missing_ok=True)


def test_order_update_stale_timestamp_skipped() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 800, status=OrderStatus.ACKED, size=0.1)
    state.working_bid = wo
    pq.put(_order_update(oid=800, status="open", ts=100, rem=0.08, orig=0.1))
    pq.put(_order_update(oid=800, status="open", ts=50, rem=0.05, orig=0.1))
    om.drain_private_events(None)
    assert abs(wo.size - 0.08) < 1e-9
    path.unlink(missing_ok=True)


def test_order_update_exact_duplicate_skipped() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 810, status=OrderStatus.ACKED, size=0.1)
    state.working_bid = wo
    ev = _order_update(oid=810, status="open", ts=200, rem=0.07, orig=0.1)
    pq.put(ev)
    pq.put(ev)
    om.drain_private_events(None)
    assert abs(wo.size - 0.07) < 1e-9
    path.unlink(missing_ok=True)


def test_terminal_then_stale_open_does_not_resurrect() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 820, status=OrderStatus.ACKED)
    state.working_bid = wo
    pq.put(_order_update(oid=820, status="filled", ts=300, rem=0.0, orig=0.1))
    pq.put(_order_update(oid=820, status="open", ts=150, rem=0.1, orig=0.1))
    om.drain_private_events(None)
    assert state.working_bid is None
    assert wo.status == OrderStatus.FILLED
    path.unlink(missing_ok=True)


def test_cancel_fill_race_converges_no_corruption() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    wo = _wo(s, 830, status=OrderStatus.CANCEL_PENDING)
    state.working_bid = wo
    pq.put(_order_update(oid=830, status="canceled", ts=400, rem=0.0, orig=0.1))
    pq.put(_order_update(oid=830, status="filled", ts=350, rem=0.0, orig=0.1))
    om.drain_private_events(None)
    assert state.working_bid is None
    assert wo.status in (OrderStatus.CANCELED, OrderStatus.FILLED)
    path.unlink(missing_ok=True)


def test_should_ingest_fills_true_when_periodic_interval_matches() -> None:
    pq = queue.Queue()
    s = _db_settings(
        PRIVATE_WS_RECOVERY_FILL_CATCHUP_TICKS=0,
        REST_FILL_RECONCILE_INTERVAL_TICKS=7,
    )
    om, state, storage, path = _make_om(s, pq)
    om._private_ws_healthy = True
    om.on_bot_tick_start()
    om._bot_tick_counter = 14
    assert 14 % 7 == 0
    assert om.should_ingest_fills_via_rest() is True
    path.unlink(missing_ok=True)


def test_should_ingest_fills_true_when_private_ws_disabled() -> None:
    s = _db_settings(PRIVATE_WS_ENABLED=False)
    pq = queue.Queue()
    om, state, storage, path = _make_om(s, pq)
    om._private_ws_healthy = True
    om.on_bot_tick_start()
    om._bot_tick_counter = 999
    assert om.should_ingest_fills_via_rest() is True
    path.unlink(missing_ok=True)


def test_should_ingest_fills_false_when_ws_healthy_past_catchup() -> None:
    pq = queue.Queue()
    s = _db_settings(PRIVATE_WS_RECOVERY_FILL_CATCHUP_TICKS=0)
    om, state, storage, path = _make_om(s, pq)
    om._private_ws_healthy = True
    for _ in range(5):
        om.on_bot_tick_start()
    assert om.should_ingest_fills_via_rest() is False
    path.unlink(missing_ok=True)


def test_reconnect_enables_rest_fill_catchup_window() -> None:
    """After a real reconnect (DISCONNECTED -> CONNECTED), the
    fill-catchup window opens for ``PRIVATE_WS_RECOVERY_FILL_CATCHUP_TICKS``
    ticks. We simulate the disconnect first so the next connect is
    treated as a reconnect (not first-time).
    """
    pq = queue.Queue()
    s = _db_settings(PRIVATE_WS_RECOVERY_FILL_CATCHUP_TICKS=4)
    om, state, storage, path = _make_om(s, pq)
    om.on_bot_tick_start()
    assert om._bot_tick_counter == 1
    # Mark that we've seen a disconnect so the next CONNECTED is a
    # reconnect rather than a first-time connect.
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.DISCONNECTED, detail=""))
    om.drain_private_events(None)
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.CONNECTED, detail=""))
    om.drain_private_events(None)
    assert om._rest_fill_catchup_until_tick == 1 + 4
    assert om.should_ingest_fills_via_rest() is True
    om.on_bot_tick_start()
    assert om._bot_tick_counter == 2
    assert om.should_ingest_fills_via_rest() is True
    om._bot_tick_counter = 10
    om._rest_fill_catchup_until_tick = 5
    assert om.should_ingest_fills_via_rest() is False
    path.unlink(missing_ok=True)


def test_first_connect_skips_rest_fill_catchup_window() -> None:
    """First-time CONNECTED (no prior DISCONNECTED) does NOT open the
    catchup window. Cold starts have no missed fills to recover, and
    the catchup REST burst was the trigger of bug-018 -- it 429'd
    OKX's per-endpoint rate limit at 2026-05-05 OKX bring-up. Only
    actual reconnects warrant catchup; a fresh process inheriting
    nothing should rely on the normal startup-reconcile path.
    """
    pq = queue.Queue()
    s = _db_settings(PRIVATE_WS_RECOVERY_FILL_CATCHUP_TICKS=4)
    om, state, storage, path = _make_om(s, pq)
    om.on_bot_tick_start()
    assert om._bot_tick_counter == 1
    # Fire CONNECTED without a preceding DISCONNECTED -- simulates
    # cold start.
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.CONNECTED, detail=""))
    om.drain_private_events(None)
    # Catchup window should NOT have been opened.
    assert om._rest_fill_catchup_until_tick == 0
    assert om._private_ws_healthy is True
    path.unlink(missing_ok=True)


def test_maybe_sync_triggers_on_forced_reconcile() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    om._client.fetch_open_orders_raw.return_value = []
    om.on_bot_tick_start()
    om._private_ws_healthy = True
    om._bot_tick_counter = 50
    om._force_open_orders_reconcile = True
    with patch.object(om, "_sync_open_orders_impl", wraps=om._sync_open_orders_impl) as spy:
        om.maybe_sync_open_orders()
        spy.assert_called_once()
    assert om._force_open_orders_reconcile is False
    path.unlink(missing_ok=True)


def test_maybe_sync_skips_when_healthy_no_other_reason() -> None:
    pq = queue.Queue()
    s = _db_settings(OPEN_ORDERS_RECONCILE_INTERVAL_TICKS=100)
    om, state, storage, path = _make_om(s, pq)
    om.on_bot_tick_start()
    om._private_ws_healthy = True
    om._bot_tick_counter = 37
    assert 37 % 100 != 0
    om._force_open_orders_reconcile = False
    with patch.object(om, "_sync_open_orders_impl") as spy:
        om.maybe_sync_open_orders()
        spy.assert_not_called()
    path.unlink(missing_ok=True)


def test_cancel_pending_watchdog_triggers_reconcile() -> None:
    pq = queue.Queue()
    s = _db_settings(
        OPEN_ORDERS_RECONCILE_INTERVAL_TICKS=1000,
        CANCEL_PENDING_REST_WATCHDOG_TICKS=5,
    )
    om, state, storage, path = _make_om(s, pq)
    om._client.fetch_open_orders_raw.return_value = []
    state.working_bid = _wo(s, 1, status=OrderStatus.CANCEL_PENDING)
    om._private_ws_healthy = True
    om._bot_tick_counter = 15
    assert 15 % 5 == 0
    with patch.object(om, "_sync_open_orders_impl") as spy:
        om.maybe_sync_open_orders()
        spy.assert_called_once()
    path.unlink(missing_ok=True)


def test_reconnect_restores_ws_health_flags() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.CONNECTED, detail=""))
    om.drain_private_events(None)
    assert state.private_ws_healthy is True
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.DISCONNECTED, detail=""))
    om.drain_private_events(None)
    assert state.private_ws_healthy is False
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.CONNECTED, detail=""))
    om.drain_private_events(None)
    assert state.private_ws_connected is True
    assert state.private_ws_healthy is True
    path.unlink(missing_ok=True)


def test_disconnected_clears_ws_health_flags() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.CONNECTED, detail=""))
    om.drain_private_events(None)
    assert state.private_ws_healthy is True
    pq.put(PrivateWsConnectionEvent(kind=PrivateWsConnectionKind.DISCONNECTED, detail="x"))
    om.drain_private_events(None)
    assert state.private_ws_healthy is False
    assert state.private_ws_connected is False
    assert om._private_ws_healthy is False
    path.unlink(missing_ok=True)


def test_unknown_dispatch_event_type_warns() -> None:
    pq = queue.Queue()
    s = _db_settings()
    om, state, storage, path = _make_om(s, pq)
    pq.put("not_an_event")
    with patch("app.execution.logger") as log:
        om.drain_private_events(None)
        log.warning.assert_called()
    path.unlink(missing_ok=True)
