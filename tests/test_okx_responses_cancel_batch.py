"""Phase 1b — ``interpret_okx_cancel_batch_response`` per-row outcomes.

OKX V5 batch-cancel responses carry one row per submitted ref with
its own ``sCode`` / ``sMsg``. The interpreter must:

  * Classify each row independently as ``success`` / ``benign_missing``
    / ``error`` (same taxonomy as the single-cancel interpreter).
  * Echo back ``ordId`` and ``clOrdId`` per row so the caller can
    correlate to its parent intent.
  * Return ``success`` as the top-level kind ONLY when every row
    succeeded. Mixed-outcome batches return top ``error`` with
    individual row outcomes in the rows list.

Auth and rate-limit are top-level concerns — the interpreter returns
those as ``transport`` with an empty rows list.
"""

from __future__ import annotations

from app.exchange.okx_responses import interpret_okx_cancel_batch_response


def test_all_rows_succeed() -> None:
    resp = {
        "code": "0",
        "msg": "",
        "data": [
            {"sCode": "0", "sMsg": "", "ordId": "1001", "clOrdId": "cloid-a"},
            {"sCode": "0", "sMsg": "", "ordId": "1002", "clOrdId": "cloid-b"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "success"
    assert len(rows) == 2
    assert all(r[0] == "success" for r in rows)
    assert rows[0][2] == "1001"
    assert rows[0][3] == "cloid-a"
    assert rows[1][2] == "1002"
    assert rows[1][3] == "cloid-b"


def test_one_row_unexpected_gone_one_success() -> None:
    """v1.3.120 reclassification: a row with sCode 51400 ("order
    does not exist") classifies as ``unexpected_gone`` per the new
    fill-vs-non-fill split. Top is ``error`` because at least one
    row didn't succeed; individual rows preserve their kinds."""
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {"sCode": "51400", "sMsg": "order does not exist", "ordId": "1001", "clOrdId": "cloid-a"},
            {"sCode": "0", "sMsg": "", "ordId": "1002", "clOrdId": "cloid-b"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "error"
    assert len(rows) == 2
    assert rows[0][0] == "unexpected_gone"
    assert rows[1][0] == "success"


def test_one_row_benign_missing_one_success() -> None:
    """Only sCode 51402 (filled) classifies as ``benign_missing``
    after the v1.3.120 split — confirms the batch interpreter
    preserves the single-cancel split."""
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {"sCode": "51402", "sMsg": "order has been filled", "ordId": "1001", "clOrdId": "cloid-a"},
            {"sCode": "0", "sMsg": "", "ordId": "1002", "clOrdId": "cloid-b"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "error"
    assert len(rows) == 2
    assert rows[0][0] == "benign_missing"
    assert rows[1][0] == "success"


def test_all_rows_unexpected_gone() -> None:
    """v1.3.120 reclassification: 51400 / 51401 / 51503 all classify
    as ``unexpected_gone`` per row; the top-level still reports
    ``error`` because no row succeeded."""
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {"sCode": "51400", "sMsg": "does not exist", "clOrdId": "cloid-a"},
            {"sCode": "51401", "sMsg": "canceled", "clOrdId": "cloid-b"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "error"
    assert all(r[0] == "unexpected_gone" for r in rows)


def test_all_rows_benign_missing_filled() -> None:
    """All-rows-filled batch — every row is a true cancel-race-lost-
    to-fill. Per-row kind is ``benign_missing``."""
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {"sCode": "51402", "sMsg": "filled", "clOrdId": "cloid-a"},
            {"sCode": "51402", "sMsg": "filled", "clOrdId": "cloid-b"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "error"
    assert all(r[0] == "benign_missing" for r in rows)


def test_auth_failure_returns_transport_with_no_rows() -> None:
    resp = {"code": "50113", "msg": "Invalid sign", "data": []}
    top_kind, detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "transport"
    assert "auth" in detail.lower()
    assert rows == []


def test_rate_limit_returns_transport_with_no_rows() -> None:
    resp = {"code": "50011", "msg": "Rate limit reached", "data": []}
    top_kind, detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "transport"
    assert "rate_limit" in detail.lower()
    assert rows == []


def test_non_dict_response_handled_gracefully() -> None:
    top_kind, detail, rows = interpret_okx_cancel_batch_response("oops")
    assert top_kind == "transport"
    assert rows == []


def test_missing_row_code_treated_as_error() -> None:
    """A row with no ``sCode`` is a venue protocol violation. Treat
    as error rather than crashing."""
    resp = {
        "code": "0",
        "msg": "",
        "data": [
            {"ordId": "1001", "clOrdId": "cloid-a"},  # no sCode
        ],
    }
    top_kind, _detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "error"
    assert rows[0][0] == "error"
    assert "missing_row_code" in rows[0][1]


# ---------------------------------------------------------------------------
# v1.4.72 wedge-elimination-cleanup Phase 1E: OKX 51410 classification.
#
# Codex Finding 13 / Bug 5. Snapshot v1.4.66-260518-192744 caught
# ``okx_row_51410: Cancellation failed as the order is already in
# canceling status or pending settlement`` on OID 3577425853826408448
# with the pre-Phase-1E classifier treating it as a hard cancel
# error, keeping the local WO live indefinitely. Semantically the
# venue is telling us "this order is already going away" — the right
# classification is ``unexpected_gone`` so the WO terminalizes.
# ---------------------------------------------------------------------------


def test_v1_4_72_phase1e_51410_row_classifies_as_unexpected_gone() -> None:
    """v1.4.72 Phase 1E: per-row sCode 51410 ("cancellation failed
    as the order is already in canceling status or pending
    settlement") classifies as ``unexpected_gone`` — semantically
    "the order is gone (or going) for a non-fill reason".
    """
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {
                "sCode": "51410",
                "sMsg": "Cancellation failed as the order is already in canceling status or pending settlement.",
                "ordId": "1001",
                "clOrdId": "cloid-a",
            },
        ],
    }
    top_kind, _detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "error"
    assert rows[0][0] == "unexpected_gone", (
        f"51410 must classify as unexpected_gone, got {rows[0][0]}"
    )
    # The detail should preserve the row code for postmortem.
    assert "51410" in rows[0][1]


def test_v1_4_72_phase1e_51410_single_cancel_classifies_as_unexpected_gone() -> None:
    """v1.4.72 Phase 1E: the single-cancel interpreter also routes
    51410 to ``unexpected_gone``. The downstream WO handler
    transitions to CANCELED via the ``unexpected_gone`` path
    (``_reap_wo_after_cancel_unexpected_gone``), same as 51400/51401/51503.
    """
    from app.exchange.okx_responses import interpret_okx_cancel_response
    resp = {
        "code": "0",
        "msg": "",
        "data": [
            {
                "sCode": "51410",
                "sMsg": "Cancellation failed as the order is already in canceling status or pending settlement.",
                "ordId": "1001",
                "clOrdId": "cloid-x",
            },
        ],
    }
    kind, detail = interpret_okx_cancel_response(resp)
    assert kind == "unexpected_gone"
    assert "51410" in detail


def test_v1_4_72_phase1e_51410_mixed_batch() -> None:
    """v1.4.72 Phase 1E: a batch with 51410 + success row resolves
    correctly — top is ``error`` (one row failed), 51410 is
    unexpected_gone, success row is success.
    """
    resp = {
        "code": "1",
        "msg": "All operations failed",
        "data": [
            {"sCode": "51410", "sMsg": "already canceling", "ordId": "2001", "clOrdId": "cloid-a"},
            {"sCode": "0", "sMsg": "", "ordId": "2002", "clOrdId": "cloid-b"},
        ],
    }
    top_kind, _detail, rows = interpret_okx_cancel_batch_response(resp)
    assert top_kind == "error"
    assert rows[0][0] == "unexpected_gone"
    assert rows[1][0] == "success"


def test_v1_4_72_phase1e_51410_in_unexpected_gone_codes_set() -> None:
    """v1.4.72 Phase 1E: defensive — confirm 51410 is in the canonical
    set. Future code that imports ``_UNEXPECTED_GONE_CODES`` will
    pick it up automatically (no manual fan-out required).
    """
    from app.exchange.okx_responses import _UNEXPECTED_GONE_CODES
    assert "51410" in _UNEXPECTED_GONE_CODES
    # Sanity: pre-Phase-1E codes still in the set.
    assert "51400" in _UNEXPECTED_GONE_CODES
    assert "51401" in _UNEXPECTED_GONE_CODES
    assert "51503" in _UNEXPECTED_GONE_CODES
