"""Hyperliquid /exchange order response interpretation (placement ACK vs silent reject)."""

from __future__ import annotations

from app.execution import (
    interpret_hl_place_order_response,
    snapshot_hl_place_response,
)


def _resting(oid: int) -> dict:
    return {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {"statuses": [{"resting": {"oid": oid}}]},
        },
    }


def _filled(oid: int) -> dict:
    return {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {
                "statuses": [
                    {
                        "filled": {
                            "totalSz": "0.02",
                            "avgPx": "3000.0",
                            "oid": oid,
                        }
                    }
                ]
            },
        },
    }


def _err(msg: str) -> dict:
    return {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {"statuses": [{"error": msg}]},
        },
    }


def test_interpret_resting_oid() -> None:
    oid, out, r = interpret_hl_place_order_response(_resting(99))
    assert oid == 99 and out == "accepted" and r == ""


def test_interpret_filled_oid_string() -> None:
    d = _resting(100)
    d["response"]["data"]["statuses"][0]["resting"]["oid"] = "100"
    oid, out, r = interpret_hl_place_order_response(d)
    assert oid == 100 and out == "accepted"


def test_interpret_filled_branch() -> None:
    oid, out, r = interpret_hl_place_order_response(_filled(55))
    assert oid == 55 and out == "accepted" and r == "filled_immediately"


def test_interpret_post_only_error_is_exchange_rejected() -> None:
    msg = "Post only order would have immediately matched, bbo was 3000/3001."
    oid, out, r = interpret_hl_place_order_response(_err(msg))
    assert oid is None and out == "exchange_rejected" and msg in r


def test_interpret_batch_error() -> None:
    resp = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {"error": "Order must have minimum value of $10."},
        },
    }
    oid, out, r = interpret_hl_place_order_response(resp)
    assert oid is None and out == "exchange_rejected"


def test_interpret_transport_rejected() -> None:
    resp = {"status": "fail", "response": {"error": "nonce"}}
    oid, out, r = interpret_hl_place_order_response(resp)
    assert oid is None and out == "transport_rejected"


def test_interpret_waiting_token_unconfirmed() -> None:
    resp = {
        "status": "ok",
        "response": {"type": "order", "data": {"statuses": ["waitingForFill"]}},
    }
    oid, out, r = interpret_hl_place_order_response(resp)
    assert oid is None and out == "unconfirmed" and "waitingForFill" in r


def test_interpret_wrong_response_type() -> None:
    resp = {"status": "ok", "response": {"type": "cancel", "data": {"statuses": []}}}
    oid, out, r = interpret_hl_place_order_response(resp)
    assert oid is None and "unexpected_response_type" in r


def test_snapshot_error_kind() -> None:
    s = snapshot_hl_place_response(_err("BadAloPx"))
    assert s.get("first_status_kind") == "error"
    assert "BadAloPx" in (s.get("parsed_substatus") or "")
