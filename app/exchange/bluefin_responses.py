"""
Bluefin Pro wire-format interpreters and deterministic client-order-id helper.

Mirrors the shape of :mod:`app.exchange.grvt_responses` so the bot core
can treat place/cancel/status responses uniformly — the adapter boundary
reduces each to ``(oid, outcome, reason)`` / ``(kind, detail)`` tuples.

Pro-sdk specifics:

* ``POST /api/v1/trade/orders`` returns ``200`` or ``202`` with body
  ``{"orderHash":"0x..."}``. Any ``orderHash`` means "accepted" — the
  OpenOrders/WS stream will surface REJECTED reasons after the fact.

* ``PUT /api/v1/trade/orders/cancel`` returns ``202`` with (typically)
  empty body. We treat 2xx as ``success`` and 404-ish errors as
  ``benign_missing``. The adapter layer stamps the submitted
  ``orderHashes`` into the response dict so the interpreter always has
  something to report.

* ``GET /api/v1/trade/openOrders`` returns an array of
  ``OpenOrderResponse`` rows. The adapter narrows to a single row before
  calling :func:`interpret_bluefin_order_status_response`.

Order id mapping:

Bluefin identifies orders by ``orderHash`` (hex). We fold to a stable
63-bit int (the bot core's ``oid: int`` contract) and cache the reverse
mapping so cancel-by-oid can recover the hash.

Client order id:

The bot produces a deterministic 16-char hex derived from inputs and the
server passes it through via ``clientOrderId``. The server does **not**
prefix "bluefin-client: " on the pro-sdk API (that was a v2-era quirk).
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from app.enums import Side


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _http_code(resp: dict[str, Any]) -> Optional[int]:
    c = resp.get("code")
    if c is None:
        return None
    try:
        return int(c)
    except (TypeError, ValueError):
        return None


def _msg(resp: dict[str, Any]) -> str:
    m = resp.get("message")
    if m is None:
        m = resp.get("errorCode")
    return str(m or "")[:800]


# ------------------------------------------------------------------
# Order-id mapping
# ------------------------------------------------------------------


_OID_TO_HASH: dict[int, str] = {}
_HASH_TO_OID: dict[str, int] = {}


def hash_to_oid(order_hash_hex: str) -> int:
    """Return a stable 63-bit int id for a Bluefin order-hash hex string."""
    h = (order_hash_hex or "").lower().strip()
    if h.startswith("0x"):
        h = h[2:]
    cached = _HASH_TO_OID.get(h)
    if cached is not None:
        return cached
    digest = hashlib.sha256(h.encode("ascii")).digest()[:8]
    oid = int.from_bytes(digest, "big", signed=False) & ((1 << 63) - 1)
    if oid == 0:
        oid = 1
    _OID_TO_HASH[oid] = h
    _HASH_TO_OID[h] = oid
    return oid


def oid_to_hash(oid: int) -> Optional[str]:
    """Return the original hex hash for an oid, if we've seen it this session."""
    return _OID_TO_HASH.get(int(oid))


# ------------------------------------------------------------------
# Response interpreters
# ------------------------------------------------------------------


_BLUEFIN_OPEN_STATUSES = {
    "STANDBY",
    "OPEN",
    "PARTIALLY_FILLED_OPEN",
}


def interpret_bluefin_place_response(
    resp: Any,
) -> tuple[Optional[int], str, str]:
    """Normalise a POST /api/v1/trade/orders response.

    Outcome mapping:
      * 2xx body containing ``orderHash`` → ``accepted``
      * Non-2xx 4xx → ``exchange_rejected``
      * Non-2xx 5xx / transport → ``transport_rejected``
      * Missing hash on 2xx → ``unconfirmed``
    """
    if not isinstance(resp, dict):
        return None, "unconfirmed", "response_not_a_dict"
    code = _http_code(resp)
    if code is not None:
        msg = _msg(resp)
        if 400 <= code < 500:
            return None, "exchange_rejected", f"http_{code}:{msg}"[:800]
        return None, "transport_rejected", f"http_{code}:{msg}"[:800]
    # 2xx — body is the CreateOrderResponse (or nested under "data" if our
    # HTTP wrapper rewrapped a bare array; defensive).
    data = resp.get("data") if isinstance(resp.get("data"), dict) else resp
    order_hash = str(data.get("orderHash") or "").strip()
    if not order_hash:
        return None, "unconfirmed", "missing_order_hash"
    oid = hash_to_oid(order_hash)
    return oid, "accepted", ""


def interpret_bluefin_cancel_response(resp: Any) -> tuple[str, str]:
    """Normalise a PUT /api/v1/trade/orders/cancel response.

    Pro-sdk returns 202 + empty body on success. Our HTTP wrapper records
    ``_http_status`` on 2xx, ``code`` only on non-2xx. 404 / "not found"
    on this endpoint means the order was already closed.
    """
    if not isinstance(resp, dict):
        return "transport", "non_dict_response"
    code = _http_code(resp)
    if code is not None:
        msg = _msg(resp).lower()
        if code == 404 or "not found" in msg or "not open" in msg or "already" in msg:
            return "benign_missing", _msg(resp)[:500]
        if 500 <= code:
            return "transport", f"http_{code}:{_msg(resp)}"[:500]
        return "error", f"http_{code}:{_msg(resp)}"[:500]
    # 2xx path — treat as success regardless of body shape (202 is the
    # normal outcome; the async WS OrderCancellationUpdate confirms the
    # actual cancel).
    return "success", ""


def interpret_bluefin_order_status_response(
    resp: Any,
) -> tuple[Optional[int], str, str]:
    """Normalise a GET /api/v1/trade/openOrders row into (oid, outcome, detail)."""
    if not isinstance(resp, dict):
        return None, "invalid", "response_not_a_dict"
    code = _http_code(resp)
    if code is not None:
        msg = _msg(resp)
        if code == 404 or "not found" in msg.lower():
            return None, "not_found", msg[:200]
        return None, "transport", f"http_{code}:{msg}"[:200]
    data = resp.get("data")
    row: dict[str, Any]
    if isinstance(data, list) and data and isinstance(data[0], dict):
        row = data[0]
    elif isinstance(data, dict):
        row = data
    elif isinstance(resp.get("orderHash"), str):
        row = resp
    else:
        return None, "not_found", "no_rows"
    order_hash = str(row.get("orderHash") or row.get("hash") or "").strip()
    oid = hash_to_oid(order_hash) if order_hash else None
    status = str(row.get("status") or row.get("orderStatus") or "").upper()
    reason = str(
        row.get("cancellationReason")
        or row.get("rejectReason")
        or row.get("failureToCancelReason")
        or ""
    )
    if status in _BLUEFIN_OPEN_STATUSES:
        return oid, "open", ""
    if status == "FILLED":
        return oid, "filled", ""
    if status in {"CANCELLED", "PARTIALLY_FILLED_CANCELED"}:
        return oid, "canceled", reason[:200]
    if status in {"EXPIRED", "PARTIALLY_FILLED_EXPIRED"}:
        return oid, "canceled", f"EXPIRED:{reason}"[:200]
    # Pro-sdk doesn't use a "REJECTED" status in openOrders (rejection
    # surfaces via the AccountCommandFailureUpdate WS event). Treat any
    # unknown status defensively.
    return oid, "unknown_proc", status[:200]


# ------------------------------------------------------------------
# Client order id
# ------------------------------------------------------------------


def make_deterministic_bluefin_client_order_id(
    symbol: str, side: Side, quote_cycle_id: str, price: float, size: float
) -> str:
    """Deterministic 16-char hex CLOID, stable across retries."""
    payload = f"{symbol}\0{side.value}\0{quote_cycle_id}\0{price:.12g}\0{size:.12g}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()[:8]
    return digest.hex()


__all__ = [
    "hash_to_oid",
    "interpret_bluefin_cancel_response",
    "interpret_bluefin_order_status_response",
    "interpret_bluefin_place_response",
    "make_deterministic_bluefin_client_order_id",
    "oid_to_hash",
]
