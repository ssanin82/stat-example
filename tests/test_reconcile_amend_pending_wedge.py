"""v1.4.24 — Reconcile must NOT silently skip AMEND_PENDING orphans.

Diagnosis from snapshot ``v1.4.23-260517-194714-prod.okx.ton.usdt.perp``:
the bot took a single fill at t=33s of the session, then went silent
for 5+ minutes. ``open_orders`` count at venue = 0; reconcile saw the
empty REST snapshot every 30s but did nothing. Root cause: the bot's
SELL working slot was stuck in ``AMEND_PENDING`` (an amend's response
was lost or the venue cancelled externally). The reconcile's
``gone_on_exchange`` branch at ``_reconcile_side`` listed only
``ACKED`` / ``PARTIAL`` / ``CANCEL_PENDING`` — **AMEND_PENDING was
missing**. So the stuck WO was never transitioned to CANCELED;
``_orchestrate`` early-returns on AMEND_PENDING; bot wedges silently.

Fix: add ``AMEND_PENDING`` to the gone_on_exchange list. Honor the
``rest_request_dispatched_at`` race guard using ``ts_amend_sent`` as
the anchor (an amend issued after the REST snapshot is still in
flight; treating it as gone races the response).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import BestBidAsk, WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[OrderManager, Path]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_apw_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "OKX_AMEND_ON_REPRICE_ENABLED": True,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=1.924,
        best_ask=1.925,
        mid_price=1.9245,
        spread_bps=5.0,
        ts_local=utc_now(),
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om._outbound.stop()  # prevent dispatch races during the test
    return om, path


def _amend_pending_wo(
    om: OrderManager,
    *,
    side: Side = Side.SELL,
    ts_amend_sent: datetime | None = None,
) -> WorkingOrder:
    """Build a WO that's mid-amend (status AMEND_PENDING)."""
    ts_amend = ts_amend_sent or (
        datetime.now(timezone.utc) - timedelta(seconds=10)
    )
    wo = WorkingOrder(
        order_id_local=f"local-{uuid.uuid4().hex[:8]}",
        order_id_exchange=12345,
        client_order_id="cl_" + uuid.uuid4().hex[:24],
        symbol=om._settings.symbol,
        side=side,
        price=1.924,
        size=3.0,
        post_only=True,
        status=OrderStatus.AMEND_PENDING,
        ts_created=datetime.now(timezone.utc) - timedelta(seconds=60),
        ts_sent=datetime.now(timezone.utc) - timedelta(seconds=60),
        ts_ack=datetime.now(timezone.utc) - timedelta(seconds=58),
    )
    wo.amend_intent_seq = 5
    wo.amend_target_px = 1.9245
    wo.amend_target_sz = 3.0
    wo.ts_amend_sent = ts_amend
    with om._state._lock:
        om._state.set_working_order(side, 0, wo)
    return wo


def test_amend_pending_orphan_marked_gone_on_exchange() -> None:
    """Wedge fix — AMEND_PENDING WO with NO matching remote should
    transition to CANCELED via gone_on_exchange. Without this fix the
    WO sticks forever and the bot wedges (see snapshot
    v1.4.23-260517-194714 diagnosis)."""
    om, path = _setup()
    try:
        wo = _amend_pending_wo(
            om,
            side=Side.SELL,
            ts_amend_sent=datetime.now(timezone.utc)
            - timedelta(seconds=30),  # well before any plausible REST snapshot
        )
        # REST snapshot taken NOW shows no remote orders.
        rest_request_dispatched_at = datetime.now(timezone.utc)
        result = om._reconcile_side(
            Side.SELL,
            None,  # remote is None — venue says no order
            rest_request_dispatched_at=rest_request_dispatched_at,
        )
        # WO should now be CANCELED (gone_on_exchange).
        assert wo.status == OrderStatus.CANCELED, (
            f"AMEND_PENDING WO with no remote must transition to "
            f"CANCELED; got {wo.status}. This is the v1.4.23 wedge "
            f"bug — pre-fix the reconcile silently skipped, leaving "
            f"the slot stuck forever."
        )
        # The cancel reason should reflect gone_on_exchange.
        assert wo.cancel_reason == "gone_on_exchange"
        # Slot should be cleared so the next quote tick can place fresh.
        with om._state._lock:
            assert om._state.get_working_order(Side.SELL, 0) is None
    finally:
        path.unlink(missing_ok=True)


def test_amend_pending_with_recent_amend_skips_gone_check() -> None:
    """Race guard — when ts_amend_sent post-dates the REST request
    dispatch, the amend's response is still in flight. Skip the
    gone_on_exchange decision; let the next reconcile cycle see the
    response-updated state."""
    om, path = _setup()
    try:
        # REST request was sent a moment ago; amend was sent AFTER.
        rest_request_dispatched_at = datetime.now(timezone.utc) - timedelta(
            seconds=1
        )
        wo = _amend_pending_wo(
            om,
            side=Side.SELL,
            ts_amend_sent=datetime.now(timezone.utc),  # just now (after REST)
        )
        result = om._reconcile_side(
            Side.SELL,
            None,
            rest_request_dispatched_at=rest_request_dispatched_at,
        )
        # WO should STILL be AMEND_PENDING — race guard fired.
        assert wo.status == OrderStatus.AMEND_PENDING
        # Slot still occupied.
        with om._state._lock:
            assert om._state.get_working_order(Side.SELL, 0) is wo
        # The race-guard counter incremented.
        assert om._state.reconcile_skip_snapshot_stale_total >= 1
    finally:
        path.unlink(missing_ok=True)


def test_acked_orphan_still_marked_gone_regression() -> None:
    """Regression: the existing ACKED → CANCELED path must continue
    to work after we added AMEND_PENDING to the same list."""
    om, path = _setup()
    try:
        wo = WorkingOrder(
            order_id_local=f"local-{uuid.uuid4().hex[:8]}",
            order_id_exchange=12345,
            client_order_id="cl_" + uuid.uuid4().hex[:24],
            symbol=om._settings.symbol,
            side=Side.BUY,
            price=1.924,
            size=3.0,
            post_only=True,
            status=OrderStatus.ACKED,
            ts_created=datetime.now(timezone.utc) - timedelta(seconds=60),
            ts_sent=datetime.now(timezone.utc) - timedelta(seconds=60),
            ts_ack=datetime.now(timezone.utc) - timedelta(seconds=58),
        )
        with om._state._lock:
            om._state.set_working_order(Side.BUY, 0, wo)
        rest_request_dispatched_at = datetime.now(timezone.utc)
        om._reconcile_side(
            Side.BUY,
            None,
            rest_request_dispatched_at=rest_request_dispatched_at,
        )
        assert wo.status == OrderStatus.CANCELED
        assert wo.cancel_reason == "gone_on_exchange"
    finally:
        path.unlink(missing_ok=True)
