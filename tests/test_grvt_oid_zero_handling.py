"""Regression guard for the GRVT oid-zero poisoning bug.

Observed in snap_20260417_144515: the bot placed a BUY on GRVT, the local
``WorkingOrder.order_id_exchange`` ended up set to ``0`` (the matching engine
assigns the real 128-bit id asynchronously, so the synchronous create_order
response can return ``order_id: null`` / ``"0"``). A subsequent cancel then
sent ``{"order_id": "0"}`` on the wire, and GRVT rejected with
    "Either order ID or client order ID must be supplied".

The reconcile path fell into the exchange-mismatch branch and DESYNC'd the
local order, even though the remote order's ``client_order_id`` still matched
— so we should have bound, not DESYNC'd.

These tests lock down:

1. ``_coerce_grvt_order_id`` and the two ``_grvt_oid_to_int`` helpers return
   ``None`` for any form of zero.
2. ``interpret_grvt_place_order_response`` does NOT accept a zero oid as an
   ack.
3. The GRVT private WS v1.order handler drops order updates that carry no
   real order id (instead of enqueueing ``oid=0``).
"""

from __future__ import annotations

from typing import Any

import pytest

from app.enums import Side
from app.exchange import grvt_ws as grvt_ws_module
from app.exchange import grvt_client as grvt_client_module
from app.exchange.grvt_client import _grvt_oid_to_int as _grvt_oid_to_int_client
from app.exchange.grvt_responses import (
    _coerce_grvt_order_id,
    interpret_grvt_cancel_response,
    interpret_grvt_place_order_response,
)
from app.exchange.grvt_ws import _grvt_oid_to_int as _grvt_oid_to_int_ws


@pytest.mark.parametrize("zero_like", [0, "0", "0x0", "0X0", "00"])
def test_coerce_grvt_order_id_rejects_zero(zero_like: Any) -> None:
    assert _coerce_grvt_order_id(zero_like) is None


@pytest.mark.parametrize("zero_like", [0, "0", "0x0"])
def test_grvt_oid_to_int_client_rejects_zero(zero_like: Any) -> None:
    assert _grvt_oid_to_int_client(zero_like) is None


@pytest.mark.parametrize("zero_like", [0, "0", "0x0", None])
def test_grvt_oid_to_int_ws_rejects_zero_and_none(zero_like: Any) -> None:
    assert _grvt_oid_to_int_ws(zero_like) is None


def test_coerce_grvt_order_id_accepts_real_128bit_id() -> None:
    real = "1334440892959520532621925129411760553"
    assert _coerce_grvt_order_id(real) == int(real)


def test_interpret_place_response_zero_oid_is_unconfirmed_not_accepted() -> None:
    """GRVT may return ``order_id: "0"`` on sync accept — must not ACK."""
    resp = {
        "result": {
            "order_id": "0",
            "state": {"status": "PENDING", "reject_reason": "UNSPECIFIED"},
        }
    }
    oid, outcome, _ = interpret_grvt_place_order_response(resp)
    assert oid is None
    assert outcome == "unconfirmed"


def test_interpret_place_response_null_order_id_is_unconfirmed() -> None:
    resp = {"result": {"order_id": None, "state": {"status": "OPEN"}}}
    oid, outcome, _ = interpret_grvt_place_order_response(resp)
    assert oid is None
    assert outcome == "unconfirmed"


def test_interpret_place_response_accepts_real_oid() -> None:
    real = "1334440892959520532621925129411760553"
    resp = {"result": {"order_id": real, "state": {"status": "OPEN"}}}
    oid, outcome, _ = interpret_grvt_place_order_response(resp)
    assert oid == int(real)
    assert outcome == "accepted"


# ---------------------------------------------------------------------------
# Private WS: v1.order updates without a real order_id must be dropped.
# ---------------------------------------------------------------------------


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def put(self, ev: Any) -> None:
        self.items.append(ev)

    def put_nowait(self, ev: Any) -> None:
        self.items.append(ev)


import json


def _make_stream() -> grvt_ws_module.GrvtPrivateStream:
    # Reach past __init__ (opens a WS) by instantiating a plain object and
    # manually wiring the queue; the message handler only touches ``self._q``
    # and ``self._on_queue_drop``.
    s = grvt_ws_module.GrvtPrivateStream.__new__(grvt_ws_module.GrvtPrivateStream)
    s._q = _FakeQueue()  # type: ignore[attr-defined]
    s._on_queue_drop = None  # type: ignore[attr-defined]
    return s


def _order_update_message(*, order_id: Any, status: str = "OPEN") -> str:
    return json.dumps(
        {
            "stream": "v1.order",
            "feed": {
                "order_id": order_id,
                "state": {
                    "status": status,
                    "update_time": "1000000",
                    "book_size": ["0.01"],
                },
                "legs": [
                    {
                        "instrument": "ETH_USDT_Perp",
                        "is_buying_asset": True,
                        "limit_price": "2500",
                        "size": "0.01",
                    }
                ],
            },
        }
    )


def test_private_ws_drops_order_update_with_missing_order_id() -> None:
    s = _make_stream()
    s.feed_message_for_tests(_order_update_message(order_id=None, status="PENDING"))
    assert s._q.items == []  # type: ignore[attr-defined]


def test_private_ws_drops_order_update_with_zero_order_id() -> None:
    s = _make_stream()
    s.feed_message_for_tests(_order_update_message(order_id="0"))
    assert s._q.items == []  # type: ignore[attr-defined]


def test_private_ws_enqueues_order_update_with_real_oid() -> None:
    s = _make_stream()
    real = "1334440892959520532621925129411760553"
    s.feed_message_for_tests(_order_update_message(order_id=real))
    assert len(s._q.items) == 1  # type: ignore[attr-defined]
    assert s._q.items[0].oid == int(real)  # type: ignore[attr-defined]
