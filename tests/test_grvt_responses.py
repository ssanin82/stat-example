from __future__ import annotations

from app.exchange.grvt_responses import (
    interpret_grvt_cancel_response,
    interpret_grvt_order_status_response,
    interpret_grvt_place_order_response,
)


def test_interpret_grvt_place_order_response_accepted() -> None:
    resp = {
        "result": {
            "order_id": "0x10",
            "state": {"status": "OPEN", "reject_reason": "UNSPECIFIED"},
        }
    }
    assert interpret_grvt_place_order_response(resp) == (16, "accepted", "")


def test_interpret_grvt_place_order_response_rejected() -> None:
    resp = {
        "result": {
            "order_id": "0x11",
            "state": {"status": "REJECTED", "reject_reason": "FAIL_POST_ONLY"},
        }
    }
    assert interpret_grvt_place_order_response(resp) == (
        None,
        "exchange_rejected",
        "FAIL_POST_ONLY",
    )


def test_interpret_grvt_cancel_response() -> None:
    assert interpret_grvt_cancel_response({"result": {"ack": True}}) == ("success", "")
    assert interpret_grvt_cancel_response({"code": 404, "message": "order not found"})[0] == (
        "benign_missing"
    )


def test_interpret_grvt_order_status_response() -> None:
    resp = {
        "result": {
            "order_id": "0x99",
            "state": {"status": "CANCELLED", "reject_reason": "CLIENT_CANCEL"},
        }
    }
    assert interpret_grvt_order_status_response(resp) == (153, "canceled", "CLIENT_CANCEL")
