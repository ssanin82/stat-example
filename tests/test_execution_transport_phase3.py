"""Phase 3: non-blocking transport handoff, intent sequencing, and duplicate-wakeup guards."""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.enums import OrderStatus, RiskAction, Side
from app.execution import OrderManager, _PlaceTransportIntent
from app.models import WorkingOrder
from app.quote_engine import FinalQuoteOrder, QuoteBuildResult
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_quote_reprice_maintenance import _decision, _fresh_market, _ok_place


def _db() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_p3_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 10.0,
            "REPRICE_THRESHOLD_BPS": 20.0,
        }
    )
    return s, path


def test_maybe_refresh_returns_before_slow_transport_rtt_completes() -> None:
    """Strategy thread must not block on exchange HTTP RTT (worker runs concurrently)."""
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    allow_http = threading.Event()

    def slow_place(*_a, **_k):
        allow_http.wait(timeout=5.0)
        time.sleep(1.0)
        return _ok_place(1)

    client.place_post_only_limit.side_effect = slow_place
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=FinalQuoteOrder(side=Side.SELL, price=3001.0, size=0.01),
            mode="two_sided",
            telemetry={"quote_engine_mode": "two_sided"},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    t0 = time.perf_counter()
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert dt_ms < 400.0, "maybe_refresh_quotes blocked on transport RTT (would include per-side sleep)"
    allow_http.set()
    om.wait_transport_idle(timeout_s=5.0)
    assert client.place_post_only_limit.call_count == 2
    path.unlink(missing_ok=True)


def test_second_maybe_refresh_while_sent_does_not_enqueue_extra_places() -> None:
    """SENT/CANCEL_PENDING short-circuits orchestration — no duplicate place intents per side."""
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    gate = threading.Event()

    def gated_place(*_a, **_k):
        gate.wait(timeout=5.0)
        return _ok_place(1)

    client.place_post_only_limit.side_effect = gated_place
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=FinalQuoteOrder(side=Side.SELL, price=3001.0, size=0.01),
            mode="two_sided",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    enq = MagicMock(wraps=om._enqueue_place_transport)
    with patch.object(om, "_enqueue_place_transport", enq):
        om.maybe_refresh_quotes(_decision(cycle="c1"), RiskAction.ALLOW, 1.0, 1.0, 0.0)
        assert enq.call_count == 2
        om.maybe_refresh_quotes(_decision(cycle="c2"), RiskAction.ALLOW, 1.0, 1.0, 0.0)
        assert enq.call_count == 2
    gate.set()
    om.wait_transport_idle(timeout_s=3.0)
    path.unlink(missing_ok=True)


def test_stale_place_intent_skipped_when_transport_seq_advanced() -> None:
    """Worker must ignore completions whose intent_seq no longer matches the working order."""
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    wo = WorkingOrder(
        order_id_local="stale-test",
        order_id_exchange=None,
        client_order_id="0x" + "cd" * 16,
        symbol=s.symbol,
        side=Side.BUY,
        price=3000.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.SENT,
        transport_intent_seq=2,
    )
    state.working_bid = wo

    stale = _PlaceTransportIntent(
        wo_order_id_local=wo.order_id_local,
        side=Side.BUY,
        intent_seq=1,
        quote_cycle_id="x",
        enqueued_mono=time.monotonic(),
    )
    om._execute_place_intent(stale)
    assert client.place_post_only_limit.call_count == 0
    path.unlink(missing_ok=True)


def test_two_sided_enqueue_without_serial_submit_blocking() -> None:
    """BUY and SELL transports are both queued without waiting for first HTTP to finish."""
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    order: list[str] = []

    def record_side(sym, is_buy, *_rest, **_kw):
        order.append("buy" if is_buy else "sell")
        return _ok_place(1)

    client.place_post_only_limit.side_effect = record_side
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=FinalQuoteOrder(side=Side.SELL, price=3001.0, size=0.01),
            mode="two_sided",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle(timeout_s=2.0)
    assert order == ["buy", "sell"]
    path.unlink(missing_ok=True)
