"""OKX V5 wire-format interpreters and deterministic client-order-id helper.

Plan reference: ``plans/20260504-okx-setup/plan.md`` Phase 2.3.

Mirrors the shape of :mod:`app.exchange.binance_responses` /
:mod:`app.exchange.grvt_responses` so the bot core can treat
place/cancel/status responses uniformly.

OKX V5 response envelope shape (different from Binance!):

    {
        "code": "0",                # "0" == success; non-zero string == venue error
        "msg":  "",                 # human-readable summary on error
        "data": [
            {
                "sCode": "0",       # per-row status code (string!)
                "sMsg":  "",        # per-row error message
                "ordId": "1234...", # only on success of place/status
                "clOrdId": "0xabc...",
                ...
            }
        ]
    }

Two layers of error reporting:

* The TOP-LEVEL ``code`` is non-"0" if the request itself was rejected
  (auth failure, malformed JSON, missing required field, etc.).
* The PER-ROW ``sCode`` is non-"0" if the request was structurally
  valid but the order-level operation failed (post-only would cross,
  insufficient margin, unknown order, etc.).

Most batch endpoints can return mixed outcomes (some rows OK, some
rejected). For the bot's single-order flow we only ever submit one
row per request, so we only inspect ``data[0]``.

Key OKX error codes we care about:

  * ``50011`` — too many requests (rate limit)
  * ``51000`` — parameter error / invalid input
  * ``51001`` — instrument not found
  * ``51008`` — order failed: insufficient balance
  * ``51202`` — Mass Cancel: no orders to cancel (benign on flatten)
  * ``51400`` — cancel order failed: order does not exist (benign)
  * ``51401`` — cancel order failed: order has been canceled (benign)
  * ``51402`` — cancel order failed: order has been filled (benign)
  * ``51403`` — cancel order failed: order is not allowed to cancel
  * ``51503`` — order modification failed: order does not exist
  * ``51604`` — post-only order would immediately cross (the canonical
                "post-only rejected" error, OKX's analog of Binance -5022)
  * ``50114`` — invalid authority (signing / passphrase wrong)
  * ``50102`` — timestamp expired (clock skew)

Client order id ("clOrdId" in OKX nomenclature):

OKX clOrdId max length is 32 characters, alphanumeric only (NO ``0x``
prefix or other special chars per the API docs as of 2024). We use
the same SHA-256-based deterministic generator pattern but produce
32 lowercase hex chars without the ``0x`` prefix.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional

from app.enums import Side


# OKX per-row status codes (sCode) that map to a "venue is healthy
# but the operation didn't take" outcome. These are common enough on
# normal flows that we surface them as benign rather than as errors.
_POST_ONLY_WOULD_CROSS_CODES = frozenset({"51604"})
# Benign cancel outcomes — order is verifiably gone VIA FILL. The
# inventory accounting already picks up the fill via the user-data
# WS, position changes accordingly, and the cancel attempt landed too
# late only because the matching engine completed the trade first.
# This is the "cancel-race-lost-to-fill" scenario and is fully
# expected on a busy MM book.
#
# 51402: order has been filled
#
# v1.3.120 split: previously this set also contained 51400 / 51401 /
# 51503 (operator pushback: "out of these 4 ONLY ONE - 51402 - is
# benign"). The other three mean "order is gone for a NON-FILL
# reason" — could be a stale-state cancel race, could be venue admin
# / risk action, could be a real bot bug (oid/cloid binding error).
# Treating them as silently benign hides anomalies. They now classify
# as ``unexpected_gone`` (see ``_UNEXPECTED_GONE_CODES`` below).
_BENIGN_CANCEL_CODES = frozenset({"51402"})

# Cancel-response codes that mean "order is gone, but NOT via fill".
# OKX returns these in several legitimate-but-suspicious scenarios:
#
# 51400: order does not exist
#        - Could be: cancel raced a fill but OKX mislabeled the
#          response (returns 51400 instead of 51402 — observed)
#        - Could be: stale-state race (we already cancelled, retry
#          hits this)
#        - Could be: real bot bug (wrong ordId/cloid)
# 51401: order has been canceled (already)
#        - Could be: our previous cancel succeeded, this is a retry
#        - Could be: venue admin / risk system cancelled it
# 51410: cancellation failed as the order is already in canceling
#        status or pending settlement
#        - The order's cancel is in flight at the venue (or it has
#          already started settling); a second cancel against it has
#          nothing to do. From the bot's point of view this is
#          semantically identical to "gone" — the slot is no longer
#          actionable. Snapshot v1.4.66-260518-192744 observed this
#          code on OID 3577425853826408448 with the previous code
#          classifying it as a hard cancel error and keeping the
#          local WO live; Codex Finding 13 / Bug 5.
#        - v1.4.72 wedge-elimination-cleanup Phase 1E adds 51410 to
#          this set so the WO terminalizes cleanly.
# 51503: cancel order failed (legacy generic)
#        - Catch-all for older OKX response paths; meaning unclear
#
# Treatment: bot transitions the WO to CANCELED locally (the order
# IS gone per OKX), logs at WARNING with full context, and bumps
# the dedicated ``cancel_unexpected_gone_total`` counter. By default
# ``execution_errors`` is NOT bumped (would self-kill the bot during
# legitimate cleanup-cancel-all races; incident 2026-05-05 22:35 UTC
# had 8 hits in 5 min). Operators wanting paranoid-mode alerting can
# set ``CANCEL_UNEXPECTED_GONE_STRICT_MODE=true`` to reclassify
# these as ``error`` (bumps ``execution_errors`` → self-kill on
# sustained occurrence within the rolling window).
_UNEXPECTED_GONE_CODES = frozenset({"51400", "51401", "51410", "51503"})
# Cancel-specific TRANSPORT codes — venue acknowledged the request
# but its response handler returned an ambiguous / misleading
# rejection. The order's actual state is unknown until the inbound
# user-data WS pushes a terminal event OR a fresh reconcile confirms
# the side. Treating as transport: cancel-pending watchdog retries
# with backoff, ``execution_errors`` is NOT bumped (no self-kill),
# and the bot does NOT lie about local WO state (stays in
# CANCEL_PENDING until a real terminal signal arrives).
#
# 50014: "Parameter X can not be empty" — observed 2026-05-17 on the
#        OKX colo trade-WS when retrying a cancel for an order that
#        was still resting at the matching engine. The original
#        v1.3.118 interpreter classified this as benign_missing,
#        which would have silently marked still-alive orders as
#        cancelled locally → unwanted fills under adverse price
#        moves. Reclassifying as transport lets the watchdog keep
#        retrying without claiming the order is gone. The inbound
#        user-data WS event remains the source of truth for terminal
#        state — when it eventually pushes CANCELED (or FILLED), the
#        oid-match path transitions the WO correctly.
_CANCEL_TRANSPORT_CODES = frozenset({"50014"})
# Top-level rate-limit code. Treated as transport so the retry layer
# backs off rather than surfacing it as a venue rejection.
_RATE_LIMIT_CODES = frozenset({"50011"})
# Top-level auth/signing codes — treat as transport so we don't
# silently keep placing orders against a venue that's rejecting them.
_AUTH_CODES = frozenset({"50113", "50114", "50102", "50104"})


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def _top_code(resp: dict[str, Any]) -> Optional[str]:
    """Top-level OKX ``code`` field. Always a STRING in OKX V5 (not int).
    ``"0"`` means request-level success. Anything else is a venue
    error. Missing means the response wasn't shaped like an OKX body.
    """
    c = resp.get("code")
    if c is None:
        return None
    return str(c)


def _top_msg(resp: dict[str, Any]) -> str:
    m = resp.get("msg")
    return str(m or "")[:800]


def _first_row(resp: dict[str, Any]) -> dict[str, Any]:
    """Pull ``data[0]`` from an OKX response, or empty dict if absent.
    The bot's single-order flow only ever submits one row per request.
    """
    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return {}


def _row_code(row: dict[str, Any]) -> Optional[str]:
    c = row.get("sCode")
    if c is None:
        return None
    return str(c)


def _row_msg(row: dict[str, Any]) -> str:
    m = row.get("sMsg")
    return str(m or "")[:800]


def _http_status(resp: dict[str, Any]) -> Optional[int]:
    s = resp.get("_http_status")
    if s is None:
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _maybe_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        return None
    if i == 0:
        return None
    return i


# ------------------------------------------------------------------
# Place response
# ------------------------------------------------------------------


def interpret_okx_place_response(
    resp: Any,
) -> tuple[Optional[int], str, str]:
    """Normalise a ``POST /api/v5/trade/order`` response.

    Outcome mapping:
      * top code "0" + row sCode "0" + ordId present → ``accepted``
      * row sCode 51604                              → ``exchange_rejected/post_only_would_cross``
      * row sCode 51008                              → ``exchange_rejected/insufficient_margin``
      * top code in auth-failure set                  → ``transport_rejected/auth``
      * top code in rate-limit set                    → ``transport_rejected/rate_limit``
      * any non-zero top code (other)                 → ``exchange_rejected``
      * any other non-zero row sCode                  → ``exchange_rejected``
      * 2xx but no data row                           → ``unconfirmed``
    """
    if not isinstance(resp, dict):
        return None, "unconfirmed", "response_not_a_dict"

    tcode = _top_code(resp)
    tmsg = _top_msg(resp)

    if tcode is None:
        # Not an OKX-shaped body — treat as transport.
        return None, "transport_rejected", f"non_okx_body:{str(resp)[:200]}"

    if tcode in _AUTH_CODES:
        return None, "transport_rejected", f"auth:{tcode}:{tmsg}"[:800]
    if tcode in _RATE_LIMIT_CODES:
        return None, "transport_rejected", f"rate_limit:{tcode}:{tmsg}"[:800]

    row = _first_row(resp)
    rcode = _row_code(row)
    rmsg = _row_msg(row)

    if tcode != "0":
        # Top-level rejection -- request itself was bad. Surface the
        # row message if we have one, else the top message.
        detail = rmsg or tmsg
        return None, "exchange_rejected", f"okx_top_{tcode}:{detail}"[:800]

    # Top code is "0" -- now check the per-row outcome.
    if rcode is None:
        return None, "unconfirmed", "missing_row_in_data"

    if rcode != "0":
        if rcode in _POST_ONLY_WOULD_CROSS_CODES:
            return None, "exchange_rejected", f"post_only_would_cross:{rmsg}"[:800]
        if rcode == "51008":
            return None, "exchange_rejected", f"insufficient_margin:{rmsg}"[:800]
        return None, "exchange_rejected", f"okx_row_{rcode}:{rmsg}"[:800]

    # Per-row success -- pull the order id.
    oid = _maybe_int(row.get("ordId"))
    if oid is None:
        return None, "unconfirmed", "missing_ord_id"
    return oid, "accepted", ""


# ------------------------------------------------------------------
# Cancel response
# ------------------------------------------------------------------


def interpret_okx_cancel_response(resp: Any) -> tuple[str, str]:
    """Normalise a ``POST /api/v5/trade/cancel-order`` response.

    Outcome mapping (v1.3.120 split):
      * top "0" + row "0"                      → ``success``
      * row 51402 (order has been filled)      → ``benign_missing``
        (true cancel-race-lost-to-fill — only THIS is fully benign)
      * row in {51400, 51401, 51503}            → ``unexpected_gone``
        (order is gone but NOT via fill — surfaced as WARNING +
        dedicated counter ``cancel_unexpected_gone_total``. WO
        transitions to CANCELED locally because the order IS gone
        per OKX; the operator should investigate WHY — could be
        stale-state retry, venue admin action, or a bot bug.
        Strict-mode operators can flip CANCEL_UNEXPECTED_GONE_STRICT
        to escalate to ``error`` and trigger self-kill on sustained
        occurrence.)
      * row in {50014}                          → ``transport``
        (OKX colo trade-WS misleading-rejection quirk; see
        ``_CANCEL_TRANSPORT_CODES``)
      * top "1" + msg keyword "has been filled" → ``benign_missing``
        (defensive fallback when row data is missing)
      * top "1" + other gone-msg keywords       → ``unexpected_gone``
      * top in auth / rate-limit sets           → ``transport``
      * any other non-zero top or row           → ``error``
    """
    if not isinstance(resp, dict):
        return "transport", "non_dict_response"

    tcode = _top_code(resp)
    tmsg = _top_msg(resp)

    if tcode is None:
        return "transport", f"non_okx_body:{str(resp)[:200]}"

    if tcode in _AUTH_CODES:
        return "transport", f"auth:{tcode}:{tmsg}"[:500]
    if tcode in _RATE_LIMIT_CODES:
        return "transport", f"rate_limit:{tcode}:{tmsg}"[:500]

    row = _first_row(resp)
    rcode = _row_code(row)
    rmsg = _row_msg(row)

    # Row sCode is the authoritative per-order outcome on OKX V5
    # batch endpoints. Check it BEFORE the top-level code so a row
    # classification doesn't get masked by the batch wrapper's top
    # code "1".
    #
    # v1.3.120 split: 51402 (filled) is the only TRULY benign code —
    # the order is gone because it filled, inventory accounting
    # picked it up via the user-data WS. 51400/51401/51503 are gone
    # for non-fill reasons (stale-state, venue admin, real bug) and
    # get the surfaced ``unexpected_gone`` classification.
    if rcode is not None and rcode in _BENIGN_CANCEL_CODES:
        return "benign_missing", f"okx_row_{rcode}:{rmsg}"[:500]
    if rcode is not None and rcode in _UNEXPECTED_GONE_CODES:
        return "unexpected_gone", f"okx_row_{rcode}:{rmsg}"[:500]
    # Cancel-specific transport reclassification (1.3.119) — sCode
    # 50014 on colo trade-WS is an ambiguous response, NOT proof the
    # order is gone. Bot retries without bumping execution_errors;
    # local WO state stays CANCEL_PENDING until a real terminal
    # signal (sCode 0 success on a later retry, or inbound user-data
    # WS event) arrives.
    if rcode is not None and rcode in _CANCEL_TRANSPORT_CODES:
        return "transport", f"colo_quirk_{rcode}:{rmsg}"[:500]

    if tcode != "0":
        # Defensive fallback: OKX has historically returned top code
        # "1" with an empty data array and a top-level msg like
        # "Order cancellation failed as the order has been filled,
        # canceled or does not exist." When that happens we have no
        # row to read, but the message itself flags the case. v1.3.120
        # applies the same fill-vs-non-fill split here:
        #   - "has been filled" → benign_missing (true fill race)
        #   - "has been canceled" / "cancelled" / "does not exist" →
        #     unexpected_gone (surfaced, dedicated counter, no
        #     execution_errors bump by default)
        # Treating ALL of these as benign trips silently; treating
        # all as error trips MAX_EXECUTION_ERRORS during racy
        # cancel-vs-fill conditions (incident 2026-05-05 22:35 UTC:
        # 8 hits in 5 min killed the bot at $0.28 drawdown). The
        # split surfaces the anomaly without crashing.
        msg_lc = tmsg.lower()
        if rcode is None:
            has_filled = "has been filled" in msg_lc
            has_canceled = (
                "has been canceled" in msg_lc
                or "has been cancelled" in msg_lc
            )
            has_missing = "does not exist" in msg_lc
            # v1.3.120: ONLY classify as benign_missing when OKX
            # specifically said "filled" and didn't also mention
            # canceled/missing. If OKX returns its ambiguous
            # "filled, canceled or does not exist" catch-all, treat
            # as unexpected_gone — we don't know which state it's
            # in and the operator should see the anomaly.
            if has_filled and not (has_canceled or has_missing):
                return "benign_missing", f"okx_top_{tcode}:{tmsg}"[:500]
            if has_filled or has_canceled or has_missing:
                # Either ambiguous multi-state message OR a clear
                # canceled / missing — both surface as unexpected_gone.
                return "unexpected_gone", f"okx_top_{tcode}:{tmsg}"[:500]
        # Surface the row sCode (when present) in the detail so the
        # operator can grep for it in logs/snapshots. Falls back to
        # the top-level msg when no row data is included.
        if rcode is not None:
            return "error", f"okx_top_{tcode}_row_{rcode}:{rmsg or tmsg}"[:500]
        return "error", f"okx_top_{tcode}:{tmsg}"[:500]

    if rcode is None:
        # Top-level success but no row -- treat as success (some
        # OKX cancel endpoints return empty data on success).
        return "success", ""
    if rcode == "0":
        return "success", ""
    return "error", f"okx_row_{rcode}:{rmsg}"[:500]


def interpret_okx_cancel_batch_response(
    resp: Any,
) -> tuple[str, str, list[tuple[str, str, Optional[str], Optional[str]]]]:
    """Normalise a ``POST /api/v5/trade/cancel-batch-orders`` response.

    Used by the outbound dispatcher's opportunistic batch-cancel path
    (1.4.0 cancel-prio Phase 1b). Returns:

      * ``top_kind`` — one of ``success`` / ``transport`` / ``error``;
        treats the top envelope code only. ``transport`` is returned
        for auth + rate-limit; ``error`` for any non-zero top code.
      * ``top_detail`` — venue-side detail string for the top kind.
      * ``rows`` — list of ``(kind, detail, ordId, clOrdId)`` tuples,
        one entry per submitted ref, IN THE ORDER OKX returned them.
        ``kind`` per row is one of ``success`` / ``benign_missing`` /
        ``error`` (same taxonomy as :func:`interpret_okx_cancel_response`).
        ``ordId`` and ``clOrdId`` are echoed back from OKX so the
        caller can correlate each row to its parent ``CancelTransportIntent``
        (use ``clOrdId`` since the intent carries the deterministic
        ``client_order_id`` but may not have a venue-side ``ordId``
        when cancel-by-cloid was used).

    Per-row semantics mirror the single-cancel interpreter:
      * row sCode 51402                 → ``benign_missing`` (filled)
      * row sCode in {51400/51401/51503} → ``unexpected_gone``
      * row sCode "0"                   → ``success``
      * any other row sCode             → ``error``
    """
    if not isinstance(resp, dict):
        return "transport", "non_dict_response", []

    tcode = _top_code(resp)
    tmsg = _top_msg(resp)

    if tcode is None:
        return "transport", f"non_okx_body:{str(resp)[:200]}", []

    if tcode in _AUTH_CODES:
        return "transport", f"auth:{tcode}:{tmsg}"[:500], []
    if tcode in _RATE_LIMIT_CODES:
        return "transport", f"rate_limit:{tcode}:{tmsg}"[:500], []

    # Parse the per-row data. OKX batch endpoints return top code "1"
    # when ANY row failed, with each row's actual sCode in the data
    # array. So the top kind alone isn't authoritative — we read each
    # row independently.
    data = resp.get("data") if isinstance(resp.get("data"), list) else []
    rows: list[tuple[str, str, Optional[str], Optional[str]]] = []
    for row in data:
        if not isinstance(row, dict):
            rows.append(("error", "non_dict_row", None, None))
            continue
        rcode = _row_code(row)
        rmsg = _row_msg(row)
        ord_id = row.get("ordId")
        cl_ord_id = row.get("clOrdId")
        ord_id_str = str(ord_id) if ord_id not in (None, "") else None
        cl_ord_id_str = str(cl_ord_id) if cl_ord_id not in (None, "") else None
        if rcode is not None and rcode in _BENIGN_CANCEL_CODES:
            rows.append((
                "benign_missing",
                f"okx_row_{rcode}:{rmsg}"[:500],
                ord_id_str,
                cl_ord_id_str,
            ))
        elif rcode is not None and rcode in _UNEXPECTED_GONE_CODES:
            # v1.3.120 split — order is gone for a non-fill reason.
            # Bot will transition the WO locally but log + count as
            # an anomaly worth investigating.
            rows.append((
                "unexpected_gone",
                f"okx_row_{rcode}:{rmsg}"[:500],
                ord_id_str,
                cl_ord_id_str,
            ))
        elif rcode == "0":
            rows.append(("success", "", ord_id_str, cl_ord_id_str))
        elif rcode is None:
            rows.append(("error", "missing_row_code", ord_id_str, cl_ord_id_str))
        else:
            rows.append((
                "error",
                f"okx_row_{rcode}:{rmsg}"[:500],
                ord_id_str,
                cl_ord_id_str,
            ))

    # Top kind: ``success`` only when EVERY row succeeded. ``error``
    # otherwise (with the venue's top-msg as detail). ``transport``
    # was already returned above for auth / rate-limit.
    if tcode == "0" and all(r[0] == "success" for r in rows):
        return "success", "", rows
    return "error", f"okx_top_{tcode}:{tmsg}"[:500], rows


# ------------------------------------------------------------------
# Batch-place response (v1.4.4)
# ------------------------------------------------------------------

# Row-level rate-limit indicators. When OKX returns top code "1"
# (batch envelope: at least one row failed) and an individual row's
# sMsg contains a rate-limit phrase, classify that row as
# transport_rejected so the retry layer absorbs it instead of
# permanently dropping the place. Observed shape from v1.4.2:
#   {"code":"1","data":[{"sCode":"...","sMsg":"Rate limit reached. ..."}]}
# Pre-1.4.4 the row was classified as ``exchange_rejected`` and the
# place intent died with it — operator saw 3 unrecoverable rejects /
# 714 placements that should have been transport_rejected.
_ROW_RATE_LIMIT_PHRASES: tuple[str, ...] = (
    "rate limit",
    "throttle",
    "too many request",
    "flow limit",
)


def _row_rate_limited(rcode: Optional[str], rmsg: str) -> bool:
    """True when the row indicates rate-limit pressure at the venue,
    either via a known sCode or by an sMsg matching one of the known
    phrases. Case-insensitive sMsg match — OKX is inconsistent about
    casing across endpoints."""
    if rcode is not None and rcode in _RATE_LIMIT_CODES:
        return True
    if not rmsg:
        return False
    lo = rmsg.lower()
    return any(p in lo for p in _ROW_RATE_LIMIT_PHRASES)


def interpret_okx_place_batch_response(
    resp: Any,
) -> tuple[
    str,
    str,
    list[tuple[Optional[int], str, str, Optional[str]]],
]:
    """Normalise a ``POST /api/v5/trade/batch-orders`` response.

    Returns ``(top_kind, top_detail, rows)`` where:

      * ``top_kind`` — ``success`` / ``transport`` / ``error``.
        ``success`` only when every row accepted; ``transport`` for
        auth / rate-limit at the envelope level; ``error`` otherwise.
      * ``top_detail`` — venue-side detail string for the top kind.
      * ``rows`` — list of ``(ord_id, kind, detail, cl_ord_id)`` per
        submitted order, IN THE ORDER OKX RETURNED THEM. Row kinds:

          - ``accepted``           — ``sCode == "0"``, ``ordId`` present
          - ``exchange_rejected``  — ``sCode`` indicates venue refusal
          - ``transport_rejected`` — row sMsg matches a rate-limit phrase
            (see ``_ROW_RATE_LIMIT_PHRASES``); caller must retry rather
            than drop. THIS IS THE 1.4.4 FIX for the v1.4.2 misclassify.
          - ``unconfirmed``        — row is missing required fields

    Caller correlates each row to its parent place intent via
    ``cl_ord_id`` — OKX echoes the request's ``clOrdId`` on every row,
    including failures. The deterministic clOrdId generator
    (:func:`make_deterministic_okx_client_order_id`) makes this
    correlation unambiguous.
    """
    if not isinstance(resp, dict):
        return "transport", "non_dict_response", []

    tcode = _top_code(resp)
    tmsg = _top_msg(resp)

    if tcode is None:
        return "transport", f"non_okx_body:{str(resp)[:200]}", []

    if tcode in _AUTH_CODES:
        return "transport", f"auth:{tcode}:{tmsg}"[:500], []
    if tcode in _RATE_LIMIT_CODES:
        return "transport", f"rate_limit:{tcode}:{tmsg}"[:500], []

    # Per-row parsing. OKX returns top code "1" when ANY row failed
    # (and "2" for partial success on some legacy paths); each row's
    # actual sCode is authoritative for that row.
    data = resp.get("data") if isinstance(resp.get("data"), list) else []
    rows: list[tuple[Optional[int], str, str, Optional[str]]] = []
    for row in data:
        if not isinstance(row, dict):
            rows.append((None, "unconfirmed", "non_dict_row", None))
            continue
        rcode = _row_code(row)
        rmsg = _row_msg(row)
        cl_ord_id = row.get("clOrdId")
        cl_ord_id_str = (
            str(cl_ord_id) if cl_ord_id not in (None, "") else None
        )

        if rcode == "0":
            oid = _maybe_int(row.get("ordId"))
            if oid is None:
                rows.append(
                    (None, "unconfirmed", "missing_ord_id", cl_ord_id_str)
                )
            else:
                rows.append((oid, "accepted", "", cl_ord_id_str))
            continue

        # Row failed. First check for row-level rate-limit indicators
        # — this is the 1.4.4 fix. Without it, the v1.4.2 batch path
        # would mis-classify e.g. ``sMsg="Rate limit reached..."`` as
        # exchange_rejected and the caller would drop the place
        # instead of letting the retry layer absorb it.
        if _row_rate_limited(rcode, rmsg):
            rows.append((
                None,
                "transport_rejected",
                f"rate_limit:okx_row_{rcode or '?'}:{rmsg}"[:500],
                cl_ord_id_str,
            ))
            continue

        if rcode in _POST_ONLY_WOULD_CROSS_CODES:
            rows.append((
                None,
                "exchange_rejected",
                f"post_only_would_cross:{rmsg}"[:500],
                cl_ord_id_str,
            ))
            continue
        if rcode == "51008":
            rows.append((
                None,
                "exchange_rejected",
                f"insufficient_margin:{rmsg}"[:500],
                cl_ord_id_str,
            ))
            continue
        if rcode is None:
            rows.append((
                None,
                "unconfirmed",
                "missing_row_code",
                cl_ord_id_str,
            ))
            continue
        rows.append((
            None,
            "exchange_rejected",
            f"okx_row_{rcode}:{rmsg}"[:500],
            cl_ord_id_str,
        ))

    # Top kind summarises the envelope. ``success`` requires EVERY row
    # accepted AND tcode "0"; anything else is ``error`` (transport
    # was already returned above for auth/rate-limit at the envelope).
    if tcode == "0" and all(r[1] == "accepted" for r in rows):
        return "success", "", rows
    return "error", f"okx_top_{tcode}:{tmsg}"[:500], rows


# ------------------------------------------------------------------
# Amend-batch response (v1.4.4 Pass 2)
# ------------------------------------------------------------------

# OKX amend-specific row codes worth special-casing:
#   51016: ``new size below filled qty`` — the requested newSz is
#          smaller than what's already filled, so the amend can't
#          shrink the remaining quantity that low. Callers should
#          fall back to cancel+place when they see this.
#   51400 / 51401 / 51503: order does not exist / canceled / not
#          found — same as the cancel case. The order is gone; the
#          local state should reconcile via the user-data WS or
#          reconcile loop.
_AMEND_BELOW_FILLED_CODES = frozenset({"51016"})


def interpret_okx_amend_batch_response(
    resp: Any,
) -> tuple[
    str,
    str,
    list[tuple[Optional[int], str, str, Optional[str]]],
]:
    """Normalise a ``POST /api/v5/trade/amend-batch-orders`` response.

    Returns ``(top_kind, top_detail, rows)`` with the same shape as
    :func:`interpret_okx_place_batch_response`. Row kinds:

      * ``accepted``           — sCode "0"; ord_id is echoed back
                                  (same id as before — amend preserves
                                  queue position so ordId is stable)
      * ``below_filled``       — sCode 51016; caller must fall back to
                                  cancel+place because the new size
                                  is below what's already filled
      * ``order_gone``         — sCode 51400/51401/51503; order is no
                                  longer on the venue (raced fill or
                                  cancel)
      * ``exchange_rejected``  — venue refused for another reason
                                  (post_only would cross, etc.)
      * ``transport_rejected`` — row-level rate-limit (the 1.4.4 fix
                                  applies here too)
      * ``unconfirmed``        — missing required fields

    Per-row correlation: the caller can match by ``clOrdId`` (echoed
    back by OKX) or by the optional ``reqId`` field. Both are
    included in the row tuple position [3].
    """
    if not isinstance(resp, dict):
        return "transport", "non_dict_response", []

    tcode = _top_code(resp)
    tmsg = _top_msg(resp)

    if tcode is None:
        return "transport", f"non_okx_body:{str(resp)[:200]}", []

    if tcode in _AUTH_CODES:
        return "transport", f"auth:{tcode}:{tmsg}"[:500], []
    if tcode in _RATE_LIMIT_CODES:
        return "transport", f"rate_limit:{tcode}:{tmsg}"[:500], []

    data = resp.get("data") if isinstance(resp.get("data"), list) else []
    rows: list[tuple[Optional[int], str, str, Optional[str]]] = []
    for row in data:
        if not isinstance(row, dict):
            rows.append((None, "unconfirmed", "non_dict_row", None))
            continue
        rcode = _row_code(row)
        rmsg = _row_msg(row)
        cl_ord_id = row.get("clOrdId") or row.get("reqId")
        cl_ord_id_str = (
            str(cl_ord_id) if cl_ord_id not in (None, "") else None
        )

        if rcode == "0":
            # Amend preserves ordId — echo it back so the caller can
            # confirm we're updating the right order.
            oid = _maybe_int(row.get("ordId"))
            rows.append((oid, "accepted", "", cl_ord_id_str))
            continue

        if _row_rate_limited(rcode, rmsg):
            rows.append((
                None,
                "transport_rejected",
                f"rate_limit:okx_row_{rcode or '?'}:{rmsg}"[:500],
                cl_ord_id_str,
            ))
            continue

        if rcode in _AMEND_BELOW_FILLED_CODES:
            rows.append((
                None,
                "below_filled",
                f"new_sz_below_filled:okx_row_{rcode}:{rmsg}"[:500],
                cl_ord_id_str,
            ))
            continue

        if rcode in _BENIGN_CANCEL_CODES or rcode in _UNEXPECTED_GONE_CODES:
            rows.append((
                None,
                "order_gone",
                f"okx_row_{rcode}:{rmsg}"[:500],
                cl_ord_id_str,
            ))
            continue

        if rcode in _POST_ONLY_WOULD_CROSS_CODES:
            rows.append((
                None,
                "exchange_rejected",
                f"post_only_would_cross:{rmsg}"[:500],
                cl_ord_id_str,
            ))
            continue

        if rcode is None:
            rows.append((
                None, "unconfirmed", "missing_row_code", cl_ord_id_str
            ))
            continue

        rows.append((
            None,
            "exchange_rejected",
            f"okx_row_{rcode}:{rmsg}"[:500],
            cl_ord_id_str,
        ))

    if tcode == "0" and all(r[1] == "accepted" for r in rows):
        return "success", "", rows
    return "error", f"okx_top_{tcode}:{tmsg}"[:500], rows


# ------------------------------------------------------------------
# Single-place response — row-level rate-limit fix (v1.4.4)
# ------------------------------------------------------------------
#
# The misclassify fix also applies to the single-order endpoint —
# ``interpret_okx_place_response`` above was originally written for
# the pre-batch flow where top code "1" with a "Rate limit reached..."
# row was treated as exchange_rejected. The check below is layered
# on top via a thin wrapper to avoid disturbing the existing tests
# that lock in the original return-shape; new callers can use the
# wrapped form when they want row-level rate-limit classification.


def interpret_okx_place_response_strict(
    resp: Any,
) -> tuple[Optional[int], str, str]:
    """Like :func:`interpret_okx_place_response` but re-classifies
    row-level rate-limit hits as ``transport_rejected`` instead of
    ``exchange_rejected``. Use this on new code paths; the unwrapped
    function stays for backward-compat callers that depend on the
    legacy taxonomy.
    """
    oid, outcome, detail = interpret_okx_place_response(resp)
    if outcome != "exchange_rejected":
        return oid, outcome, detail
    lo = detail.lower()
    if any(p in lo for p in _ROW_RATE_LIMIT_PHRASES):
        # Promote to transport so the retry layer absorbs it.
        return None, "transport_rejected", f"rate_limit:{detail}"[:800]
    return oid, outcome, detail


# ------------------------------------------------------------------
# Order-status response
# ------------------------------------------------------------------


# OKX order-state values (from /api/v5/trade/order ``state`` field):
#   live          — resting in the book
#   partially_filled
#   filled
#   canceled
#   mmp_canceled  — canceled by market-maker-protection
#   _failed       — never made it to the book (some flows)
_OKX_OPEN_STATES = frozenset({"live", "partially_filled"})
_OKX_FILLED_STATES = frozenset({"filled"})
_OKX_CANCELED_STATES = frozenset({"canceled", "mmp_canceled"})

# v1.4.64 wedge-elimination follow-up: OKX error codes that
# DEFINITIVELY mean "this order doesn't exist at the venue". For a
# SENT-status order on the bot side, this means the place was
# rejected at intake (rate-limit / validation / etc.) and the venue
# never accepted it. The bot's ``_try_resolve_sent_order_by_cloid``
# treats ``outcome="not_found"`` as REJECTED + slot release —
# whereas ``outcome="invalid"`` is treated as ambiguous (just bumps
# the poll counter), which produced a 136-second stuck-SENT in
# snapshot v1.4.61-260518-181059.
#
# Codes:
#   51603 — "Order does not exist" on the order-status endpoint.
#     The venue is unambiguous: nothing with this cloid/ordId is
#     known to them.
_OKX_ORDER_NOT_FOUND_CODES = frozenset({"51603"})


def interpret_okx_order_status_response(
    resp: Any,
) -> tuple[Optional[int], str, str]:
    """Normalise ``GET /api/v5/trade/order`` response.

    Returns ``(oid, outcome, detail)`` per the Protocol contract:
    ``open / filled / canceled / rejected / not_found / invalid /
    transport / unknown_proc``.
    """
    if not isinstance(resp, dict):
        return None, "invalid", "response_not_a_dict"

    tcode = _top_code(resp)
    tmsg = _top_msg(resp)

    if tcode is None:
        return None, "transport", f"non_okx_body:{str(resp)[:200]}"
    if tcode in _AUTH_CODES:
        return None, "transport", f"auth:{tcode}:{tmsg}"[:500]
    if tcode in _RATE_LIMIT_CODES:
        return None, "transport", f"rate_limit:{tcode}:{tmsg}"[:500]

    row = _first_row(resp)
    rcode = _row_code(row)
    rmsg = _row_msg(row)

    # OKX returns top "0" + empty data for "order not found" on the
    # status endpoint. So top non-zero or row non-zero usually means
    # something else.
    if tcode != "0":
        # v1.4.64: classify 51603 ("Order does not exist") as
        # ``not_found`` so the SENT-recovery path can REJECT the
        # WO immediately instead of polling indefinitely. Other
        # top-level non-zero codes stay as ``invalid`` (genuinely
        # ambiguous — auth failure, malformed request, etc.).
        if tcode in _OKX_ORDER_NOT_FOUND_CODES:
            return None, "not_found", f"okx_top_{tcode}:{tmsg or rmsg}"[:500]
        return None, "invalid", f"okx_top_{tcode}:{tmsg or rmsg}"[:500]

    if not row:
        return None, "not_found", "empty_data"

    if rcode is not None and rcode != "0":
        if rcode in _BENIGN_CANCEL_CODES:
            return None, "not_found", f"okx_row_{rcode}:{rmsg}"[:500]
        # v1.4.64: same classification at the row level.
        if rcode in _OKX_ORDER_NOT_FOUND_CODES:
            return None, "not_found", f"okx_row_{rcode}:{rmsg}"[:500]
        return None, "invalid", f"okx_row_{rcode}:{rmsg}"[:500]

    oid = _maybe_int(row.get("ordId"))
    state = str(row.get("state") or "").lower().strip()

    if state in _OKX_OPEN_STATES:
        return oid, "open", state
    if state in _OKX_FILLED_STATES:
        return oid, "filled", state
    if state in _OKX_CANCELED_STATES:
        return oid, "canceled", state
    if not state:
        return None, "invalid", "missing_state"
    return oid, "unknown_proc", state


# ------------------------------------------------------------------
# Deterministic client-order-id
# ------------------------------------------------------------------


def make_deterministic_okx_client_order_id(
    symbol: str,
    side: Side,
    quote_cycle_id: str,
    price: float,
    size: float,
) -> str:
    """Produce a stable 32-hex client-order-id for the given quote
    intent. Format: 32 lowercase hex chars (no ``0x`` prefix), well
    under OKX's 32-char clOrdId max and within OKX's
    alphanumeric-only constraint.

    Determinism property: the same (symbol, side, quote_cycle_id,
    price, size) tuple always produces the same id, so retries don't
    accidentally place duplicates.
    """
    payload = (
        f"{symbol}|{side.value}|{quote_cycle_id}|"
        f"{price:.10g}|{size:.10g}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
