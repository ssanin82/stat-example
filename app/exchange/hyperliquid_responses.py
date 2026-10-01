"""
Hyperliquid wire-format interpreters and cloid generator.

Lives next to the Hyperliquid adapter — not in ``app.execution`` — because
this is venue-specific wire knowledge. The HL adapter exposes these as
methods on the :class:`~app.exchange.base.PerpExchangeAdapter` surface;
the bot core never imports from this module directly.

``app.execution`` re-exports these names for back-compat so external
scripts / existing tests that still import from there keep working.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from app.enums import Side


def _coerce_exchange_oid(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def interpret_hl_place_order_response(resp: Any) -> tuple[Optional[int], str, str]:
    """
    Hyperliquid POST /exchange order result (SDK returns parsed JSON).

    Returns (exchange_oid, outcome, reason):
    - outcome accepted: resting or filled with parsable oid (reason empty or 'filled_immediately')
    - exchange_rejected: per-order or batch error string from the exchange (HTTP may still be 200)
    - transport_rejected: top-level status != ok
    - unconfirmed: unknown shape, non-order response type, string token (e.g. waitingForFill), etc.
    """
    if not isinstance(resp, dict):
        return None, "unconfirmed", "response_not_a_dict"

    top = resp.get("status")
    if top != "ok":
        r = resp.get("response")
        if isinstance(r, str):
            detail = r
        elif isinstance(r, dict):
            detail = str(r.get("error") or r.get("msg") or r)[:500]
        else:
            detail = str(resp)[:500]
        return None, "transport_rejected", (detail or str(top))[:800]

    response = resp.get("response")
    if not isinstance(response, dict):
        return None, "unconfirmed", "missing_or_invalid_response_object"

    rtype = response.get("type")
    if rtype != "order":
        return None, "unconfirmed", f"unexpected_response_type:{rtype!r}"

    data = response.get("data")
    if not isinstance(data, dict):
        return None, "unconfirmed", "missing_response_data_object"

    batch_err = data.get("error")
    if isinstance(batch_err, str) and batch_err.strip():
        return None, "exchange_rejected", batch_err.strip()[:800]

    statuses = data.get("statuses")
    if not isinstance(statuses, list) or len(statuses) == 0:
        return None, "unconfirmed", "empty_or_missing_statuses"

    st0 = statuses[0]
    if isinstance(st0, str):
        return None, "unconfirmed", f"status_token:{st0}"

    if not isinstance(st0, dict):
        return None, "unconfirmed", f"unexpected_status_entry_type:{type(st0).__name__}"

    if "error" in st0:
        err = st0.get("error")
        return None, "exchange_rejected", str(err)[:800] if err is not None else "error_field_empty"

    if "resting" in st0:
        resting = st0["resting"]
        if isinstance(resting, dict):
            oid = _coerce_exchange_oid(resting.get("oid"))
            if oid is not None:
                return oid, "accepted", ""
        return None, "unconfirmed", "resting_without_parsable_oid"

    if "filled" in st0:
        filled = st0["filled"]
        if isinstance(filled, dict):
            oid = _coerce_exchange_oid(filled.get("oid"))
            if oid is not None:
                return oid, "accepted", "filled_immediately"
        return None, "unconfirmed", "filled_without_parsable_oid"

    keys = sorted(st0.keys())
    return None, "unconfirmed", f"unknown_status_shape:{keys}"


def interpret_hl_order_status_response(resp: Any) -> tuple[Optional[int], str, str]:
    """
    Hyperliquid POST /info type=orderStatus (by numeric oid or cloid string).

    Returns (exchange_oid_if_known, outcome, detail):
    - outcome: open | filled | canceled | rejected | not_found | invalid | transport | unknown_proc
    """
    if not isinstance(resp, dict):
        return None, "invalid", "response_not_a_dict"
    top = resp.get("status")
    if top == "unknownOid":
        return None, "not_found", ""
    if top != "order":
        return None, "transport", str(top)[:200]
    outer = resp.get("order")
    if not isinstance(outer, dict):
        return None, "invalid", "missing_order_object"
    inner = outer.get("order")
    if not isinstance(inner, dict):
        return None, "invalid", "missing_nested_order"
    proc = outer.get("status")
    proc_s = proc.strip().lower() if isinstance(proc, str) else ""
    oid = _coerce_exchange_oid(inner.get("oid"))
    if proc_s == "open":
        return oid, "open", ""
    if proc_s == "filled":
        return oid, "filled", ""
    if "cancel" in proc_s:
        return oid, "canceled", str(proc)[:200] if proc is not None else ""
    if proc_s == "rejected" or proc_s.endswith("rejected"):
        return oid, "rejected", str(proc)[:200] if proc is not None else ""
    if proc_s == "triggered":
        return oid, "open", ""
    return oid, "unknown_proc", str(proc)[:200] if proc is not None else ""


_HL_CANCEL_MISSING_ORDER_MARKERS = (
    "never placed",
    "already canceled",
    "already cancelled",
    "or filled",
    "missingorder",
)


def _cancel_error_is_benign_missing(msg: str) -> bool:
    m = (msg or "").lower()
    return any(x in m for x in _HL_CANCEL_MISSING_ORDER_MARKERS)


def interpret_hl_cancel_response(resp: Any) -> tuple[str, str]:
    """
    Hyperliquid POST /exchange cancel or cancelByCloid (SDK-returned JSON).

    Returns (kind, detail):
    - kind: success | benign_missing | error | transport
    """
    if not isinstance(resp, dict):
        return "transport", "non_dict_response"
    if resp.get("status") != "ok":
        tail = resp.get("error") or resp.get("response") or resp
        return "transport", str(tail)[:400]
    r = resp.get("response")
    if not isinstance(r, dict):
        return "error", "missing_response_object"
    rtype = r.get("type")
    if rtype not in ("cancel", "cancelByCloid"):
        return "error", f"unexpected_response_type:{rtype!r}"
    data = r.get("data")
    if not isinstance(data, dict):
        return "error", "missing_data_object"
    st = data.get("statuses")
    if not isinstance(st, list) or len(st) == 0:
        return "error", "empty_or_missing_statuses"
    s0 = st[0]
    if s0 == "success" or (isinstance(s0, str) and s0.lower() == "success"):
        return "success", ""
    if isinstance(s0, dict) and "error" in s0:
        err = s0.get("error")
        es = str(err) if err is not None else ""
        if _cancel_error_is_benign_missing(es):
            return "benign_missing", es[:500]
        return "error", es[:500]
    return "error", str(s0)[:200]


def make_deterministic_cloid_hex(
    symbol: str, side: Side, quote_cycle_id: str, price: float, size: float
) -> str:
    """16-byte Hyperliquid cloid (0x + 32 hex chars), stable for a given normalized quote intent."""
    payload = f"{symbol}\0{side.value}\0{quote_cycle_id}\0{price:.12g}\0{size:.12g}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()[:16]
    return "0x" + digest.hex()


__all__ = [
    "interpret_hl_place_order_response",
    "interpret_hl_order_status_response",
    "interpret_hl_cancel_response",
    "make_deterministic_cloid_hex",
]
