"""Private WS order updates must late-bind the exchange oid via cloid.

The snapshot ``tmp/snap_20260417_164102`` showed every BUY stuck with
``order_id_exchange=null`` locally, because GRVT's synchronous
``create_order`` response does not reliably populate ``order_id``. The WS
sent status updates for the real oid, but our handler matched only by
oid — so updates for unbound WOs were silently discarded, and the order
state machine lost track of the order's real lifecycle.

These tests lock in:

- ``PrivateOrderUpdateEvent`` carries a ``cloid`` field.
- The GRVT ``v1.order`` parser extracts ``metadata.client_order_id`` into
  that field.
- ``_handle_private_order_update`` matches by oid first, but falls back to
  cloid when the local WO has no oid yet — binding the oid on the spot
  and applying the status change.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.enums import OrderStatus, Side
from app.exchange.private_events import PrivateOrderUpdateEvent
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from app.execution import OrderManager, transition
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[OrderManager, Any, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_ws_cloid_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH_USDT_Perp",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state)
    om._private_ws_healthy = True
    return om, state, path


def _make_wo(
    side: Side, *, cloid: str, oid: Any = None, status: OrderStatus = OrderStatus.SENT
) -> WorkingOrder:
    wo = WorkingOrder(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=oid,
        client_order_id=cloid,
        symbol="ETH_USDT_Perp",
        side=side,
        price=2440.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.NEW_LOCAL,
    )
    transition(wo, OrderStatus.SENT)
    wo.status = status
    return wo


def test_private_order_update_event_carries_cloid_field() -> None:
    ev = PrivateOrderUpdateEvent(
        oid=1,
        coin="ETH_USDT_Perp",
        status="OPEN",
        status_timestamp_ms=1,
        side="B",
        limit_px=0.0,
        remaining_sz=0.0,
        orig_sz=0.0,
        raw_status="",
        cloid="12345",
    )
    assert ev.cloid == "12345"


def test_grvt_ws_parser_captures_cloid_from_metadata() -> None:
    import json

    from app.exchange import grvt_ws as mod

    class _Q:
        def __init__(self) -> None:
            self.items: list[Any] = []

        def put(self, x: Any) -> None:
            self.items.append(x)

        def put_nowait(self, x: Any) -> None:
            self.items.append(x)

    s = mod.GrvtPrivateStream.__new__(mod.GrvtPrivateStream)
    s._q = _Q()  # type: ignore[attr-defined]
    s._on_queue_drop = None  # type: ignore[attr-defined]
    msg = json.dumps(
        {
            "stream": "v1.order",
            "feed": {
                "order_id": "1334440892959520532621925129411760553",
                "state": {"status": "OPEN", "update_time": "1000000"},
                "legs": [
                    {
                        "instrument": "ETH_USDT_Perp",
                        "is_buying_asset": True,
                        "limit_price": "2440",
                        "size": "0.01",
                    }
                ],
                "metadata": {"client_order_id": "6011131468247176399"},
            },
        }
    )
    s.feed_message_for_tests(msg)
    assert len(s._q.items) == 1  # type: ignore[attr-defined]
    ev = s._q.items[0]  # type: ignore[attr-defined]
    assert ev.cloid == "6011131468247176399"
    assert ev.oid == 1334440892959520532621925129411760553


def test_ws_update_binds_oid_when_local_unbound_and_cloid_matches() -> None:
    om, state, path = _setup()
    try:
        cloid = "6011131468247176399"
        remote_oid = 1334440892959520532621925129411760553
        wo = _make_wo(Side.BUY, cloid=cloid, oid=None, status=OrderStatus.SENT)
        with state._lock:
            state.working_bid = wo
        ev = PrivateOrderUpdateEvent(
            oid=remote_oid,
            coin="ETH_USDT_Perp",
            status="OPEN",
            status_timestamp_ms=1_000,
            side="B",
            limit_px=2440.0,
            remaining_sz=0.01,
            orig_sz=0.01,
            raw_status="",
            cloid=cloid,
        )
        om._handle_private_order_update(ev)
        assert wo.order_id_exchange == remote_oid, "oid should be late-bound by cloid"
        # OPEN with full remaining_sz → ACKED transition.
        assert wo.status == OrderStatus.ACKED
    finally:
        path.unlink(missing_ok=True)


def test_ws_update_cancel_binds_and_terminates_unbound_order() -> None:
    """The failure mode that produced phantom BUYs: cancel WS for an unbound
    local WO must now land, terminate the local, and drop the working slot."""
    om, state, path = _setup()
    try:
        cloid = "5040730717868104535"
        remote_oid = 1334440893116391496650798566679670493
        wo = _make_wo(Side.BUY, cloid=cloid, oid=None, status=OrderStatus.CANCEL_PENDING)
        with state._lock:
            state.working_bid = wo
        ev = PrivateOrderUpdateEvent(
            oid=remote_oid,
            coin="ETH_USDT_Perp",
            status="CANCELLED",
            status_timestamp_ms=2_000,
            side="B",
            limit_px=2440.0,
            remaining_sz=0.0,
            orig_sz=0.01,
            raw_status="CLIENT_CANCEL",
            cloid=cloid,
        )
        om._handle_private_order_update(ev)
        assert wo.order_id_exchange == remote_oid
        assert wo.status == OrderStatus.CANCELED
        # Working slot must be cleared once terminal status is applied.
        with state._lock:
            assert state.working_bid is None
    finally:
        path.unlink(missing_ok=True)


def test_ws_update_with_unknown_cloid_is_ignored() -> None:
    """If neither oid nor cloid match, the update must be dropped — we must
    not bind an arbitrary oid onto a working order on a whim."""
    om, state, path = _setup()
    try:
        wo = _make_wo(Side.BUY, cloid="MINE_1", oid=None, status=OrderStatus.SENT)
        with state._lock:
            state.working_bid = wo
        ev = PrivateOrderUpdateEvent(
            oid=999,
            coin="ETH_USDT_Perp",
            status="OPEN",
            status_timestamp_ms=1_000,
            side="B",
            limit_px=2440.0,
            remaining_sz=0.01,
            orig_sz=0.01,
            raw_status="",
            cloid="SOMEONE_ELSES_CLOID",
        )
        om._handle_private_order_update(ev)
        assert wo.order_id_exchange is None
        assert wo.status == OrderStatus.SENT
    finally:
        path.unlink(missing_ok=True)


def test_ws_update_oid_match_still_wins_when_both_could_match() -> None:
    """When oid matches but cloid is also present, oid path is authoritative."""
    om, state, path = _setup()
    try:
        cloid = "X"
        wo = _make_wo(
            Side.BUY, cloid=cloid, oid=1234567890, status=OrderStatus.ACKED
        )
        with state._lock:
            state.working_bid = wo
        ev = PrivateOrderUpdateEvent(
            oid=1234567890,
            coin="ETH_USDT_Perp",
            status="OPEN",
            status_timestamp_ms=1_000,
            side="B",
            limit_px=2440.0,
            remaining_sz=0.005,  # Will trigger PARTIAL
            orig_sz=0.01,
            raw_status="",
            cloid=cloid,
        )
        om._handle_private_order_update(ev)
        assert wo.order_id_exchange == 1234567890
        assert wo.status == OrderStatus.PARTIAL
    finally:
        path.unlink(missing_ok=True)
