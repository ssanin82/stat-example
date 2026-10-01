"""v1.4.4 Pass 2 — ``interpret_okx_amend_batch_response`` per-row outcomes.

The amend interpreter must:

  * Accept ``accepted`` rows that include the (unchanged) ordId.
  * Distinguish ``below_filled`` (sCode 51016) as its own row kind
    — caller MUST fall back to cancel+place because the requested
    newSz is below what's already filled.
  * Distinguish ``order_gone`` (51400/51401/51402/51503) from other
    rejection reasons — order is on venue terms gone (filled or
    canceled) and local state should reconcile.
  * Apply the SAME row-level rate-limit reclassify as the place
    interpreter: a row whose sMsg matches a rate-limit phrase must
    classify as ``transport_rejected``, not ``exchange_rejected``.
"""

from __future__ import annotations

from app.exchange.okx_responses import interpret_okx_amend_batch_response


def test_all_rows_accepted() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [
            {"sCode": "0", "sMsg": "", "ordId": "5001", "clOrdId": "cloid-a"},
            {"sCode": "0", "sMsg": "", "ordId": "5002", "clOrdId": "cloid-b"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_amend_batch_response(resp)
    assert top_kind == "success"
    assert rows[0] == (5001, "accepted", "", "cloid-a")
    assert rows[1] == (5002, "accepted", "", "cloid-b")


def test_below_filled_is_distinct_row_kind() -> None:
    """sCode 51016 — caller must distinguish to fall back to
    cancel+place rather than re-trying the same amend forever."""
    resp = {
        "code": "1",
        "msg": "operation failed",
        "data": [
            {
                "sCode": "51016",
                "sMsg": "new size below filled qty",
                "clOrdId": "shrink-too-far",
            },
        ],
    }
    _top_kind, _detail, rows = interpret_okx_amend_batch_response(resp)
    _oid, kind, detail, cloid = rows[0]
    assert kind == "below_filled"
    assert "new_sz_below_filled" in detail
    assert cloid == "shrink-too-far"


def test_order_gone_codes() -> None:
    """Any of the cancel-side benign / unexpected-gone codes mean
    the order is no longer at the venue — local state must
    reconcile, but it's not a true error worthy of execution-error
    bumping."""
    for sc in ("51400", "51401", "51402", "51503"):
        resp = {
            "code": "1",
            "msg": "",
            "data": [
                {"sCode": sc, "sMsg": "order does not exist", "clOrdId": "x"},
            ],
        }
        _top, _detail, rows = interpret_okx_amend_batch_response(resp)
        assert rows[0][1] == "order_gone", f"sCode {sc}"


def test_row_rate_limit_reclassified_to_transport() -> None:
    """Same 1.4.4 fix as the place interpreter — row-level rate-limit
    must surface as transport so the retry layer absorbs it."""
    resp = {
        "code": "1",
        "msg": "",
        "data": [
            {
                "sCode": "50061",
                "sMsg": "Rate limit reached. Refer to API documentation and throttle requests accordingly.",
                "clOrdId": "rl",
            },
        ],
    }
    _top, _detail, rows = interpret_okx_amend_batch_response(resp)
    assert rows[0][1] == "transport_rejected"
    assert "rate_limit" in rows[0][2]


def test_post_only_would_cross_after_amend() -> None:
    """Amending a post_only order to a price that would cross should
    classify as exchange_rejected with the cross detail surfaced —
    caller arms the cross cooldown."""
    resp = {
        "code": "1",
        "msg": "",
        "data": [
            {"sCode": "51604", "sMsg": "post-only would cross", "clOrdId": "x"},
        ],
    }
    _top, _detail, rows = interpret_okx_amend_batch_response(resp)
    _oid, kind, detail, _cloid = rows[0]
    assert kind == "exchange_rejected"
    assert "post_only_would_cross" in detail


def test_mixed_accept_and_below_filled() -> None:
    resp = {
        "code": "2",
        "msg": "partial success",
        "data": [
            {"sCode": "0", "sMsg": "", "ordId": "5001", "clOrdId": "ok"},
            {
                "sCode": "51016",
                "sMsg": "new size below filled qty",
                "clOrdId": "shrink-too-far",
            },
        ],
    }
    top_kind, _detail, rows = interpret_okx_amend_batch_response(resp)
    assert top_kind == "error"  # not all accepted
    assert rows[0][1] == "accepted"
    assert rows[1][1] == "below_filled"


def test_envelope_auth_failure_is_transport() -> None:
    resp = {"code": "50113", "msg": "Invalid signature", "data": []}
    top_kind, detail, rows = interpret_okx_amend_batch_response(resp)
    assert top_kind == "transport"
    assert "auth" in detail
    assert rows == []


def test_envelope_rate_limit_is_transport() -> None:
    resp = {"code": "50011", "msg": "Too Many Requests", "data": []}
    top_kind, detail, rows = interpret_okx_amend_batch_response(resp)
    assert top_kind == "transport"
    assert "rate_limit" in detail
    assert rows == []


def test_non_dict_response_is_transport() -> None:
    top_kind, _detail, rows = interpret_okx_amend_batch_response(None)
    assert top_kind == "transport"
    assert rows == []


def test_req_id_used_when_clOrdId_absent() -> None:
    """Caller can pass ``req_id`` instead of relying on echoed
    clOrdId for correlation — useful when the WO's cloid is
    inconvenient to thread through."""
    resp = {
        "code": "0",
        "msg": "",
        "data": [
            {"sCode": "0", "sMsg": "", "ordId": "9001", "reqId": "amend-req-7"},
        ],
    }
    _top, _detail, rows = interpret_okx_amend_batch_response(resp)
    assert rows[0][3] == "amend-req-7"
