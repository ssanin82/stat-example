"""
GRVT wire-format interpreters and deterministic client-order-id helper.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from app.enums import Side


def _coerce_grvt_order_id(value: Any) -> Optional[int]:
    """Parse a GRVT order id into an int, or return None if absent / invalid.

    GRVT order ids are 128-bit hashes assigned by the backend; a real id is
    never zero. Create_order responses may arrive with ``order_id: null`` /
    ``"0"`` before the matching engine assigns the real id (the field is
    documented as "[Filled by GRVT Backend]" in the pysdk schema). Treating
    those as a valid id caused the bot to bind a ``0`` to the local working
    order, so later cancels went out as ``{"order_id": "0"}`` and GRVT
    rejected them with "Either order ID or client order ID must be supplied".
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value if value != 0 else None
    s = str(value).strip()
    if not s:
        return None
    try:
        if s.lower().startswith("0x"):
            parsed = int(s, 16)
        else:
            parsed = int(s)
    except (TypeError, ValueError):
        return None
    return parsed if parsed != 0 else None


def interpret_grvt_place_order_response(resp: Any) -> tuple[Optional[int], str, str]:
    if not isinstance(resp, dict):
        return None, "unconfirmed", "response_not_a_dict"
    if "code" in resp:
        return None, "exchange_rejected", str(resp.get("message") or resp.get("code"))[:800]
    result = resp.get("result")
    if not isinstance(result, dict):
        return None, "unconfirmed", "missing_result_object"
    state = result.get("state")
    if isinstance(state, dict):
        status = str(state.get("status") or "").upper()
        if status == "REJECTED":
            reason = str(state.get("reject_reason") or "REJECTED")
            return None, "exchange_rejected", reason[:800]
    oid = _coerce_grvt_order_id(result.get("order_id"))
    if oid is None:
        return None, "unconfirmed", "missing_or_invalid_order_id"
    return oid, "accepted", ""


def interpret_grvt_cancel_response(resp: Any) -> tuple[str, str]:
    if not isinstance(resp, dict):
        return "transport", "non_dict_response"
    if "code" in resp:
        msg = str(resp.get("message") or resp.get("code"))
        m = msg.lower()
        if "not found" in m or "not open" in m:
            return "benign_missing", msg[:500]
        return "error", msg[:500]
    result = resp.get("result")
    if isinstance(result, dict) and bool(result.get("ack")):
        return "success", ""
    return "error", "missing_ack_true"


def interpret_grvt_order_status_response(resp: Any) -> tuple[Optional[int], str, str]:
    if not isinstance(resp, dict):
        return None, "invalid", "response_not_a_dict"
    if "code" in resp:
        msg = str(resp.get("message") or resp.get("code"))
        if "not found" in msg.lower():
            return None, "not_found", msg[:200]
        return None, "transport", msg[:200]
    result = resp.get("result")
    if not isinstance(result, dict):
        return None, "invalid", "missing_result"
    oid = _coerce_grvt_order_id(result.get("order_id"))
    state = result.get("state")
    status = ""
    reason = ""
    if isinstance(state, dict):
        status = str(state.get("status") or "").upper()
        reason = str(state.get("reject_reason") or "")
    if status in {"PENDING", "OPEN"}:
        return oid, "open", ""
    if status == "FILLED":
        return oid, "filled", ""
    if status == "CANCELLED":
        return oid, "canceled", reason[:200]
    if status == "REJECTED":
        return oid, "rejected", reason[:200]
    return oid, "unknown_proc", status[:200]


def make_deterministic_grvt_client_order_id(
    symbol: str, side: Side, quote_cycle_id: str, price: float, size: float
) -> str:
    payload = f"{symbol}\0{side.value}\0{quote_cycle_id}\0{price:.12g}\0{size:.12g}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()[:8]
    # UInt64 range keeps it compatible with GRVT client_order_id expectations.
    return str(int.from_bytes(digest, byteorder="big", signed=False))


__all__ = [
    "interpret_grvt_place_order_response",
    "interpret_grvt_cancel_response",
    "interpret_grvt_order_status_response",
    "make_deterministic_grvt_client_order_id",
]
