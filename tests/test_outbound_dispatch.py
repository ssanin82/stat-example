"""Outbound dispatcher: coalescing, batch flush, unresolved suppression, materiality."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from app.enums import OrderStatus, RiskAction, Side
from app.execution import OrderManager
from app.outbound_dispatch import CancelTransportIntent, OutboundDispatchCoordinator, PlaceTransportIntent
from app.quote_engine import FinalQuoteOrder, QuoteBuildResult
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_quote_reprice_maintenance import _decision, _fresh_market, _ok_place


def _db() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_od_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 10.0,
            "REPRICE_THRESHOLD_BPS": 20.0,
            "ACTION_BATCH_INTERVAL_MS": 0,
            "ACTION_WS_ENABLED": False,
            "ACTION_REPRICE_EPSILON_TICKS": 0,
            "ACTION_RESIZE_EPSILON_RATIO": 0,
            "ACTION_MIN_REPLACE_INTERVAL_MS": 0,
        }
    )
    return s, path


def test_dispatch_coalesces_two_place_intents_same_side() -> None:
    s, path = _db()
    executed: list[PlaceTransportIntent] = []

    def exec_place(p: PlaceTransportIntent) -> None:
        executed.append(p)

    def exec_cancel(_c: CancelTransportIntent) -> None:
        pass

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
    )
    d.start()
    d.submit_place(
        PlaceTransportIntent("a", Side.BUY, 1, "q1", time.monotonic(), intent_created_perf=0.0)
    )
    d.submit_place(
        PlaceTransportIntent("b", Side.BUY, 2, "q2", time.monotonic(), intent_created_perf=0.0)
    )
    d.wait_until_idle(2.0)
    assert len(executed) == 1
    assert executed[0].wo_order_id_local == "b"
    assert executed[0].intent_seq == 2
    d.stop()
    path.unlink(missing_ok=True)


def test_zero_batch_interval_dispatches_on_notify_without_timeout_delay() -> None:
    s, path = _db()
    executed: list[PlaceTransportIntent] = []
    reached = threading.Event()

    def exec_place(p: PlaceTransportIntent) -> None:
        executed.append(p)
        reached.set()

    def exec_cancel(_c: CancelTransportIntent) -> None:
        pass

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
    )
    d.start()
    # Give the worker thread a brief chance to park in wait().
    time.sleep(0.02)
    t0 = time.perf_counter()
    d.submit_place(
        PlaceTransportIntent("a", Side.BUY, 1, "q1", time.monotonic(), intent_created_perf=0.0)
    )
    assert reached.wait(timeout=0.25)
    notify_to_exec_ms = (time.perf_counter() - t0) * 1000.0
    d.wait_until_idle(2.0)
    assert len(executed) == 1
    # Keep this loose to avoid timing fragility on busy CI hosts.
    assert notify_to_exec_ms < 250.0
    d.stop()
    path.unlink(missing_ok=True)


def test_no_place_while_side_unresolved() -> None:
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om._set_side_unresolved(Side.BUY, reason="test")

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=None,
            mode="bid",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle(timeout_s=2.0)
    assert client.place_post_only_limit.call_count == 0
    path.unlink(missing_ok=True)


def test_materiality_suppresses_redundant_replace() -> None:
    s, path = _db()
    s2 = s.model_copy(update={"ACTION_REPRICE_EPSILON_TICKS": 100.0, "ACTION_MIN_REPLACE_INTERVAL_MS": 60_000.0})
    storage = Storage(s2)
    storage.init_schema()
    state = BotState(s2)
    state.market = _fresh_market(s2)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s2, client, storage, state, private_event_queue=None)

    wo = om._stage_place_order_local(
        Side.BUY, price=3000.0, size=0.01, quote_cycle_id="x"
    )
    assert wo is not None
    wo.order_id_exchange = 999
    from app.execution import transition

    transition(wo, OrderStatus.ACKED)
    om.persist(wo)
    state.working_bid = wo
    om._last_emitted_fp[Side.BUY] = (3000.0, 0.01)
    om._last_outbound_replace_mono[Side.BUY] = time.monotonic()

    def fake_build_small_move(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.05, size=0.01),
            ask_order=None,
            mode="bid",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build_small_move  # type: ignore[method-assign]
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle(timeout_s=2.0)
    assert client.cancel_order.call_count == 0
    path.unlink(missing_ok=True)


def test_outbound_metrics_populated_on_place() -> None:
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.last_exchange_transport_mode = "http"
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=None,
            mode="bid",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle(timeout_s=3.0)
    flags = state.status_flags_dict()
    assert flags.get("ws_action_send_count", 0) + flags.get("http_action_send_count", 0) >= 1
    path.unlink(missing_ok=True)


def test_cancel_lane_not_starved_when_batch_interval_nonzero() -> None:
    s, path = _db()
    s2 = s.model_copy(update={"ACTION_BATCH_INTERVAL_MS": 50.0})
    storage = Storage(s2)
    storage.init_schema()
    state = BotState(s2)
    state.market = _fresh_market(s2)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s2, client, storage, state, private_event_queue=None)
    wo = om._stage_place_order_local(Side.BUY, price=3000.0, size=0.01, quote_cycle_id="c")
    assert wo is not None
    wo.order_id_exchange = 42
    from app.execution import transition

    transition(wo, OrderStatus.ACKED)
    om.persist(wo)
    state.working_bid = wo
    om._enqueue_cancel_quote_path(wo)
    t0 = time.perf_counter()
    om.wait_transport_idle(timeout_s=3.0)
    # v1.4.33+ routes all cancels through ``cancel_batch_orders`` so
    # they draw from the CANCEL_BATCH pool (300/2 s) instead of
    # CANCEL_SINGLE (60/2 s). Even a lone cancel uses the batch
    # endpoint with a 1-row body.
    assert client.cancel_batch_orders.call_count >= 1
    assert (time.perf_counter() - t0) < 2.5
    path.unlink(missing_ok=True)
