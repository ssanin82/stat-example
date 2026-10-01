"""Binance Futures USDM wire-format interpreters and deterministic
client-order-id helper.

Plan reference: ``plans/20260420-binance-move/plan.md`` Phase 2.3.

Mirrors the shape of :mod:`app.exchange.bluefin_responses` /
:mod:`app.exchange.grvt_responses` so the bot core can treat
place/cancel/status responses uniformly.

Binance error-code reference (subset we care about):

  *  -1000  generic transport / internal (we synthesise this from
            httpx exceptions so the interpreter has a code to branch on)
  *  -1003  too many requests (rate limit)
  *  -1021  timestamp outside recvWindow (clock skew)
  *  -1022  signature invalid
  *  -2010  insufficient balance / margin
  *  -2011  unknown order (already filled / cancelled)
  *  -2013  order does not exist
  *  -2019  margin insufficient
  *  -2020  order would immediately match (post-only / GTX would cross)
  *  -4131  PERCENT_PRICE filter violation
  *  -5022  post-only order would cross (the canonical "GTX rejected"
            error code; some endpoints use -2020 instead)

Client order id:

Binance Futures clientOrderId max length is 36 characters. The shape
matches Hyperliquid's (``0x`` + 32 hex = 34 chars), so the existing
make_deterministic generator pattern fits cleanly with no
modification beyond namespacing.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from app.enums import Side


# Binance error codes that map to "the order didn't take but the
# venue is healthy" (vs transport / 5xx errors which retry).
_POST_ONLY_WOULD_CROSS_CODES = frozenset({-5022, -2020})
_BENIGN_MISSING_CODES = frozenset({-2011, -2013})  # unknown / already gone


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _binance_code(resp: dict[str, Any]) -> Optional[int]:
    """Binance's own error code (negative int) lives at top-level
    ``code`` field. Non-error responses have no ``code``.
    """
    c = resp.get("code")
    if c is None:
        return None
    try:
        ic = int(c)
    except (TypeError, ValueError):
        return None
    # Binance reports 200 with no code on success; the only time a
    # positive int could appear here is from our HTTP wrapper stamping
    # ``_http_status`` into the dict — but that lives under
    # ``_http_status``, not ``code``. So treat positive code as "not
    # really an error" and return None.
    if ic >= 0:
        return None
    return ic


def _msg(resp: dict[str, Any]) -> str:
    m = resp.get("msg")
    if m is None:
        m = resp.get("message")
    return str(m or "")[:800]


def _http_status(resp: dict[str, Any]) -> Optional[int]:
    s = resp.get("_http_status")
    if s is None:
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------
# Place response
# ------------------------------------------------------------------


def interpret_binance_place_response(
    resp: Any,
) -> tuple[Optional[int], str, str]:
    """Normalise a ``POST /fapi/v1/order`` response.

    Outcome mapping:
      * 2xx body containing ``orderId`` → ``accepted`` (oid = orderId)
      * Binance code in {-5022, -2020}   → ``exchange_rejected/post_only_would_cross``
      * Binance code in {-2010, -2019}   → ``exchange_rejected/insufficient_margin``
      * Binance code in {-1021}          → ``exchange_rejected/timestamp_skew``
      * Binance code in {-1022}          → ``exchange_rejected/bad_signature``
      * Other negative Binance codes 4xx → ``exchange_rejected``
      * Negative -1000 / 5xx HTTP        → ``transport_rejected``
      * 2xx without orderId              → ``unconfirmed``
    """
    if not isinstance(resp, dict):
        return None, "unconfirmed", "response_not_a_dict"
    bcode = _binance_code(resp)
    if bcode is not None:
        msg = _msg(resp)
        if bcode in _POST_ONLY_WOULD_CROSS_CODES:
            return None, "exchange_rejected", f"post_only_would_cross:{msg}"[:800]
        if bcode in (-2010, -2019):
            return None, "exchange_rejected", f"insufficient_margin:{msg}"[:800]
        if bcode == -1021:
            return None, "exchange_rejected", f"timestamp_skew:{msg}"[:800]
        if bcode == -1022:
            return None, "exchange_rejected", f"bad_signature:{msg}"[:800]
        if bcode == -1003:
            return None, "transport_rejected", f"rate_limit:{msg}"[:800]
        if bcode == -1000:
            return None, "transport_rejected", f"transport:{msg}"[:800]
        # Other negative codes — assume venue rejection
        return None, "exchange_rejected", f"binance_code_{bcode}:{msg}"[:800]
    # No binance code → 2xx success
    body = resp.get("data") if isinstance(resp.get("data"), dict) else resp
    order_id = body.get("orderId")
    try:
        oid = int(order_id) if order_id is not None else None
    except (TypeError, ValueError):
        oid = None
    if oid is None or oid == 0:
        return None, "unconfirmed", "missing_order_id"
    return oid, "accepted", ""


# ------------------------------------------------------------------
# Cancel response
# ------------------------------------------------------------------


def interpret_binance_cancel_response(resp: Any) -> tuple[str, str]:
    """Normalise a ``DELETE /fapi/v1/order`` response.

    Outcome mapping:
      * 2xx with orderId   → ``success``
      * code -2011 / -2013 → ``benign_missing`` (already filled / unknown)
      * code -1000 / 5xx   → ``transport``
      * other negative     → ``error``
    """
    if not isinstance(resp, dict):
        return "transport", "non_dict_response"
    bcode = _binance_code(resp)
    if bcode is not None:
        msg = _msg(resp)
        if bcode in _BENIGN_MISSING_CODES:
            return "benign_missing", f"binance_code_{bcode}:{msg}"[:500]
        if bcode == -1000:
            return "transport", f"transport:{msg}"[:500]
        if bcode == -1003:
            return "transport", f"rate_limit:{msg}"[:500]
        return "error", f"binance_code_{bcode}:{msg}"[:500]
    return "success", ""


# ------------------------------------------------------------------
# Order-status response
# ------------------------------------------------------------------


_BINANCE_OPEN_STATUSES = frozenset({"NEW", "PARTIALLY_FILLED"})
_BINANCE_FILLED_STATUSES = frozenset({"FILLED"})
_BINANCE_CANCELED_STATUSES = frozenset(
    {"CANCELED", "EXPIRED", "EXPIRED_IN_MATCH"}
)
_BINANCE_REJECTED_STATUSES = frozenset({"REJECTED"})


def interpret_binance_order_status_response(
    resp: Any,
) -> tuple[Optional[int], str, str]:
    """Normalise ``GET /fapi/v1/order`` / single-row openOrders entry.

    Returns ``(oid, outcome, detail)`` per the Protocol contract:
    ``open / filled / canceled / rejected / not_found / invalid /
    transport / unknown_proc``.
    """
    if not isinstance(resp, dict):
        return None, "invalid", "response_not_a_dict"
    bcode = _binance_code(resp)
    if bcode is not None:
        msg = _msg(resp)
        if bcode in _BENIGN_MISSING_CODES:
            return None, "not_found", f"binance_code_{bcode}:{msg}"[:500]
        if bcode in (-1000, -1003):
            return None, "transport", f"binance_code_{bcode}:{msg}"[:500]
        return None, "invalid", f"binance_code_{bcode}:{msg}"[:500]
    # 2xx — extract orderId + status
    body = resp.get("data") if isinstance(resp.get("data"), dict) else resp
    try:
        oid = int(body.get("orderId") or 0)
    except (TypeError, ValueError):
        oid = 0
    status = str(body.get("status") or "").upper()
    if status in _BINANCE_OPEN_STATUSES:
        return (oid or None), "open", status
    if status in _BINANCE_FILLED_STATUSES:
        return (oid or None), "filled", status
    if status in _BINANCE_CANCELED_STATUSES:
        return (oid or None), "canceled", status
    if status in _BINANCE_REJECTED_STATUSES:
        return (oid or None), "rejected", status
    if not status:
        return None, "invalid", "missing_status"
    return (oid or None), "unknown_proc", status


# ------------------------------------------------------------------
# Deterministic client-order-id
# ------------------------------------------------------------------


def make_deterministic_binance_client_order_id(
    symbol: str,
    side: Side,
    quote_cycle_id: str,
    price: float,
    size: float,
) -> str:
    """Produce a stable 32-hex client-order-id for the given quote
    intent. Format: ``0x`` + 32 hex chars = 34 chars total, well under
    Binance's 36-char clientOrderId max.

    Determinism property: the same (symbol, side, quote_cycle_id,
    price, size) tuple always produces the same id, so retries don't
    accidentally place duplicates.
    """
    payload = (
        f"{symbol}|{side.value}|{quote_cycle_id}|"
        f"{price:.10g}|{size:.10g}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return "0x" + digest
