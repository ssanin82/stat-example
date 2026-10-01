"""Phase 0.5 regression — cancel-latency decomposition stamps.

Three timestamps must land on the ``WorkingOrder`` (and persist to
``orders``) for every successful OKX cancel:

  * ``ts_cancel_sent`` — bot wall clock immediately before HTTP send.
  * ``ts_cancel_acked`` — bot wall clock after the HTTP cancel
    response interpreted as success.
  * ``venue_cancel_utime_ms`` — OKX server-side cancel timestamp
    from the private WS orders-channel CANCELED event (``uTime``).

Cancel-already-gone races (``benign_missing``) must NOT stamp
``ts_cancel_acked`` — the cancel never really acked, and including
those samples in the RTT distribution would skew it.

This file also asserts that the cancel-RTT tracker ingests a sample
on success (and only on success), with ``op="cancel"`` so the
heartbeat publisher can produce a clean place vs cancel RTT split.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from app.enums import OrderStatus, Side
from app.exchange.private_events import PrivateOrderUpdateEvent
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from app.execution import OrderManager
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings_db() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_cxl_ts_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH_USDT_Perp",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    return s, path


def _bootstrap() -> tuple[OrderManager, Path]:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state)
    return om, path


def _live_wo() -> WorkingOrder:
    wo = WorkingOrder(
        order_id_local="L-cxl-ts",
        order_id_exchange=987654321,
        client_order_id="MM-cxl-ts-1",
        symbol="ETH_USDT_Perp",
        side=Side.BUY,
        price=2400.00,
        size=0.01,
        post_only=True,
        status=OrderStatus.SENT,
        quote_cycle_id="qt",
    )
    wo.ts_sent = utc_now()
    wo.ts_ack = utc_now()
    return wo


def test_successful_cancel_stamps_all_three_local_timestamps() -> None:
    om, path = _bootstrap()
    try:
        wo = _live_wo()
        with om._state._lock:
            om._state.working_bid = wo
        # Mock the cancel HTTP to return success and the response
        # interpreter to classify it as ``success``.
        om._client.cancel_order.return_value = {"result": {"ack": True}}
        om._client.interpret_cancel_response.return_value = ("success", "")

        before_call = utc_now()
        ok = om.cancel_order(wo, trigger_reason="test_phase_0_5")
        assert ok is True
        after_call = utc_now()

        # ts_cancel_sent is stamped just before the HTTP call.
        assert wo.ts_cancel_sent is not None
        assert before_call <= wo.ts_cancel_sent <= after_call
        # ts_cancel_acked is stamped right after the response.
        assert wo.ts_cancel_acked is not None
        assert wo.ts_cancel_sent <= wo.ts_cancel_acked <= after_call
        # ts_cancel_requested is stamped at the very top of cancel_order.
        assert wo.ts_cancel_requested is not None
        assert wo.ts_cancel_requested <= wo.ts_cancel_sent
    finally:
        path.unlink(missing_ok=True)


def test_benign_missing_cancel_does_not_stamp_acked() -> None:
    """When OKX returns ``sCode=51400`` (order already gone — filled
    or canceled), the response classifies as ``benign_missing``. The
    bot still records the attempt (``cancel_response_outcome``) but
    MUST NOT stamp ``ts_cancel_acked`` — there's no real ack, and
    including the sample in the RTT distribution would skew it."""
    om, path = _bootstrap()
    try:
        wo = _live_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._client.cancel_order.return_value = {
            "code": "1",
            "data": [{"sCode": "51400", "sMsg": "order does not exist"}],
        }
        om._client.interpret_cancel_response.return_value = (
            "benign_missing",
            "okx_row_51400",
        )
        before = utc_now()
        om.cancel_order(wo, trigger_reason="test_phase_0_5")
        assert wo.ts_cancel_sent is not None
        assert wo.ts_cancel_sent >= before
        # Critical assertion: benign_missing means NO ack stamp.
        assert wo.ts_cancel_acked is None, (
            "ts_cancel_acked must NOT be stamped on benign_missing — "
            "the cancel never really acked. Including these samples in "
            "the cancel-RTT distribution would pollute it with order-"
            "already-gone race timings, not real wire RTT."
        )
        assert wo.cancel_response_outcome == "benign_missing"
    finally:
        path.unlink(missing_ok=True)


def test_cancel_rtt_tracker_ingests_only_on_success() -> None:
    """The OrderRttTracker receives one ``op='cancel'`` sample per
    successful cancel and zero on benign_missing / transport
    errors. Verified by inspecting the tracker's filtered summary."""
    om, path = _bootstrap()
    try:
        # First: a successful cancel — should produce 1 sample.
        wo1 = _live_wo()
        wo1.order_id_local = "L-success"
        with om._state._lock:
            om._state.working_bid = wo1
        om._client.cancel_order.return_value = {"result": {"ack": True}}
        om._client.interpret_cancel_response.return_value = ("success", "")
        om.cancel_order(wo1, trigger_reason="phase_0_5_test_success")

        # Second: a benign_missing — should NOT produce a sample.
        wo2 = _live_wo()
        wo2.order_id_local = "L-benign"
        wo2.order_id_exchange = 111111
        with om._state._lock:
            om._state.working_bid = wo2
        om._client.interpret_cancel_response.return_value = (
            "benign_missing",
            "okx_51400",
        )
        om.cancel_order(wo2, trigger_reason="phase_0_5_test_benign")

        # Tracker's cancel-summary should have exactly 1 sample.
        cancel_summary = om.cancel_rtt_summary()
        assert cancel_summary["sample_count"] == 1, (
            f"Expected 1 cancel RTT sample (success), got "
            f"{cancel_summary['sample_count']}. Benign-missing samples "
            f"must not be ingested."
        )
        # Place tracker unaffected (no place samples were submitted).
        place_summary = om.order_rtt_summary()
        assert place_summary["sample_count"] == 0
    finally:
        path.unlink(missing_ok=True)


def test_ws_canceled_event_stamps_venue_utime() -> None:
    """The private-WS CANCELED handler must copy ``uTime`` from the
    event onto the WO's ``venue_cancel_utime_ms`` field. Sanity-check
    anchor against bot clock skew + WS path length."""
    om, path = _bootstrap()
    try:
        wo = _live_wo()
        wo.status = OrderStatus.CANCEL_PENDING
        with om._state._lock:
            om._state.working_bid = wo
        # Synthesize a CANCELED event with a known uTime.
        utime_ms = 1700000000123
        ev = PrivateOrderUpdateEvent(
            oid=int(wo.order_id_exchange),
            coin=wo.symbol,
            status="canceled",
            status_timestamp_ms=utime_ms,
            side="B" if wo.side == Side.BUY else "A",
            limit_px=wo.price,
            remaining_sz=0.0,
            orig_sz=wo.size,
            raw_status="canceled",
            cloid=wo.client_order_id,
        )
        # Drive the handler.
        om._handle_private_order_update(ev)

        assert wo.venue_cancel_utime_ms == utime_ms, (
            f"venue_cancel_utime_ms should equal the event's uTime "
            f"({utime_ms}); got {wo.venue_cancel_utime_ms}"
        )
    finally:
        path.unlink(missing_ok=True)
