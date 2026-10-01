"""v1.4.4 — ``interpret_okx_place_batch_response`` per-row outcomes.

The batch interpreter must:

  * Classify each row independently as ``accepted`` (with ord_id) /
    ``exchange_rejected`` / ``transport_rejected`` / ``unconfirmed``.
  * Echo back ``clOrdId`` per row so the caller can correlate to its
    parent intent — OKX preserves submission order but deterministic
    cloid-keyed correlation is what the caller actually uses.
  * Return ``success`` as the top-level kind ONLY when every row
    accepted AND envelope code "0".
  * **THE CORE 1.4.4 FIX**: a row whose ``sMsg`` matches a rate-limit
    phrase (e.g. "Rate limit reached. Refer to API documentation
    and throttle requests accordingly.") must classify as
    ``transport_rejected`` so the retry layer absorbs it. Pre-1.4.4
    these were silently mis-classified as ``exchange_rejected`` and
    permanently dropped — the source of the 3/714 rejections in the
    v1.4.2 snapshot that triggered this work-unit.

The auth and envelope-level rate-limit codes are top-level concerns
— interpreter returns those as ``transport`` with empty rows list,
same shape as the cancel-batch interpreter.
"""

from __future__ import annotations

from app.exchange.okx_responses import (
    interpret_okx_place_batch_response,
    interpret_okx_place_response_strict,
)


def test_all_rows_accepted() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [
            {"sCode": "0", "sMsg": "", "ordId": "1001", "clOrdId": "cloid-a"},
            {"sCode": "0", "sMsg": "", "ordId": "1002", "clOrdId": "cloid-b"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_place_batch_response(resp)
    assert top_kind == "success"
    assert len(rows) == 2
    assert rows[0] == (1001, "accepted", "", "cloid-a")
    assert rows[1] == (1002, "accepted", "", "cloid-b")


def test_row_level_rate_limit_reclassified_as_transport() -> None:
    """The 1.4.4 core fix. Top envelope code "1" + a row whose sMsg
    contains the OKX "Rate limit reached..." phrase should classify
    THAT row as transport_rejected — not exchange_rejected.

    The v1.4.2 snapshot had the exact wire shape below and showed
    3 of these dropped as permanent rejections."""
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {
                "sCode": "50061",
                "sMsg": "Rate limit reached. Refer to API documentation and throttle requests accordingly.",
                "clOrdId": "cloid-rate-limited",
            },
        ],
    }
    top_kind, _detail, rows = interpret_okx_place_batch_response(resp)
    assert top_kind == "error"  # top is "error" because not all rows accepted
    assert len(rows) == 1
    ord_id, kind, detail, cloid = rows[0]
    assert ord_id is None
    assert kind == "transport_rejected"
    assert "rate_limit" in detail
    assert cloid == "cloid-rate-limited"


def test_mixed_accept_and_post_only_cross() -> None:
    resp = {
        "code": "2",
        "msg": "Partial success",
        "data": [
            {"sCode": "0", "sMsg": "", "ordId": "2001", "clOrdId": "good"},
            {"sCode": "51604", "sMsg": "Post-only would cross", "clOrdId": "cross"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_place_batch_response(resp)
    assert top_kind == "error"
    assert rows[0] == (2001, "accepted", "", "good")
    ord_id, kind, detail, cloid = rows[1]
    assert ord_id is None
    assert kind == "exchange_rejected"
    assert "post_only_would_cross" in detail
    assert cloid == "cross"


def test_envelope_rate_limit_is_transport() -> None:
    """Top code 50011 is the per-UID rate-limit envelope. Whole batch
    returns transport (rows list is empty)."""
    resp = {"code": "50011", "msg": "Too Many Requests", "data": []}
    top_kind, detail, rows = interpret_okx_place_batch_response(resp)
    assert top_kind == "transport"
    assert "rate_limit" in detail
    assert rows == []


def test_envelope_auth_failure_is_transport() -> None:
    resp = {"code": "50113", "msg": "Invalid signature", "data": []}
    top_kind, detail, rows = interpret_okx_place_batch_response(resp)
    assert top_kind == "transport"
    assert "auth" in detail
    assert rows == []


def test_row_missing_ord_id_is_unconfirmed() -> None:
    """sCode "0" but ordId missing — the silent-phantom-place signature
    that triggers the strict-place-unconfirmed-kill flow."""
    resp = {
        "code": "0",
        "msg": "",
        "data": [
            {"sCode": "0", "sMsg": "", "clOrdId": "phantom"},  # no ordId
        ],
    }
    top_kind, _detail, rows = interpret_okx_place_batch_response(resp)
    # top is "error" because the row isn't accepted (we treat
    # missing ordId as unconfirmed, which is not accepted).
    assert top_kind == "error"
    ord_id, kind, detail, cloid = rows[0]
    assert ord_id is None
    assert kind == "unconfirmed"
    assert "missing_ord_id" in detail
    assert cloid == "phantom"


def test_row_insufficient_margin() -> None:
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {"sCode": "51008", "sMsg": "insufficient balance", "clOrdId": "broke"},
        ],
    }
    _top_kind, _detail, rows = interpret_okx_place_batch_response(resp)
    ord_id, kind, detail, cloid = rows[0]
    assert ord_id is None
    assert kind == "exchange_rejected"
    assert "insufficient_margin" in detail
    assert cloid == "broke"


def test_non_dict_response_is_transport() -> None:
    top_kind, _detail, rows = interpret_okx_place_batch_response(None)
    assert top_kind == "transport"
    assert rows == []


def test_strict_single_place_promotes_row_rate_limit() -> None:
    """The wrapper ``interpret_okx_place_response_strict`` promotes a
    row-level rate-limit hit to transport_rejected on the single-order
    endpoint too — pre-1.4.4 these were mis-classified as
    exchange_rejected on the single-order path (same root cause as
    the batch path, fixed at a different layer)."""
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {
                "sCode": "50061",
                "sMsg": "Rate limit reached. Refer to API documentation and throttle requests accordingly.",
                "clOrdId": "x",
            }
        ],
    }
    oid, outcome, detail = interpret_okx_place_response_strict(resp)
    assert oid is None
    assert outcome == "transport_rejected"
    assert "rate_limit" in detail


def test_strict_single_place_passes_through_non_rate_limit() -> None:
    """Non-rate-limit rejections stay classified as exchange_rejected.
    Don't over-promote — only the rate-limit phrases should flip.

    On the single-place endpoint with top code != "0" the legacy
    interpreter returns ``okx_top_{tcode}:{detail}`` without
    inspecting row sCode (different from the batch interpreter which
    is row-centric). The strict wrapper just promotes by sMsg match;
    here the detail is "Post-only would cross" — no rate-limit
    phrase — so it stays exchange_rejected with the original detail."""
    resp = {
        "code": "1",
        "msg": "Order failed",
        "data": [
            {"sCode": "51604", "sMsg": "Post-only would cross", "clOrdId": "x"},
        ],
    }
    oid, outcome, detail = interpret_okx_place_response_strict(resp)
    assert oid is None
    assert outcome == "exchange_rejected"
    # Detail carries the envelope-level prefix because top code is "1".
    assert "okx_top_1" in detail
    assert "Post-only would cross" in detail
