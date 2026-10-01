"""Cloid-based cancel must include ``time_to_live_ms``.

Regression guard for the phantom-orders symptom observed in
``tmp/snap_20260417_164102``. With aggressive repricing, the bot frequently
cancels an order whose ``order_id_exchange`` is still ``None`` locally (the
synchronous ``create_order`` response did not carry the real oid yet —
GRVT assigns it asynchronously). The fallback is cancel-by-cloid.

GRVT's documented semantics for ``time_to_live_ms`` on cancel_order are:
"During this period, any order creation with a matching client_order_id
will be cancelled rather than added to the matching engine." Capped at
5000 ms. Setting it unconditionally closes the window where our cancel
arrives before the create has been fully indexed, which would otherwise
let the order materialise as a phantom.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.exchange import grvt_client as grvt_client_module
from app.exchange.grvt_client import GrvtClient


_ETH_ROW = {
    "instrument": "ETH_USDT_Perp",
    "tick_size": "0.01",
    "min_size": "0.001",
    "min_notional": "20",
    "base_decimals": 9,
    "instrument_hash": "197633",
}


class _FakeMetaResp:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._payload = {"result": rows}
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeMetaClient:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __enter__(self) -> "_FakeMetaClient":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(self, _url: str, *, json: dict[str, Any]) -> _FakeMetaResp:  # noqa: ARG002
        return _FakeMetaResp(self._rows)


class _CapturingResp:
    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self.status_code = 200
        self.text = ""
        self._body = body or {"result": {"ack": True}}

    def json(self) -> dict[str, Any]:
        return dict(self._body)


class _CapturingHttp:
    def __init__(self) -> None:
        self.last_payload: dict[str, Any] | None = None
        self.last_url: str | None = None

    def post(self, url: str, **kwargs: Any) -> _CapturingResp:
        self.last_url = url
        self.last_payload = kwargs.get("json")
        return _CapturingResp()


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> GrvtClient:
    monkeypatch.setattr(
        grvt_client_module.httpx,
        "Client",
        lambda *_args, **_kwargs: _FakeMetaClient([_ETH_ROW]),
    )
    c = GrvtClient(
        config={
            "symbol": "ETH_USDT_Perp",
            "api_key": "test-api-key",
            "api_secret": "0x" + "11" * 32,
            "sub_account_id": "42",
            "env": "prod",
        }
    )
    c._cookie_gravity = "fake-cookie"
    c._cookie_expiry_epoch = 1e18
    return c


def test_cancel_by_cloid_sends_time_to_live_ms(client: GrvtClient) -> None:
    capturer = _CapturingHttp()
    client._http = capturer  # type: ignore[assignment]
    client.cancel_order_by_cloid("ETH_USDT_Perp", "12345678901234567890")
    assert capturer.last_url.endswith("/full/v1/cancel_order")
    payload = capturer.last_payload or {}
    assert payload.get("client_order_id") == "12345678901234567890"
    # TTL is a string per the GRVT pysdk type ``time_to_live_ms: str | None``
    # and must be in GRVT's documented range (capped at 5000 ms).
    ttl_raw = payload.get("time_to_live_ms")
    assert ttl_raw is not None, "cloid cancel must carry time_to_live_ms"
    assert isinstance(ttl_raw, str)
    ttl = int(ttl_raw)
    assert 1000 <= ttl <= 5000, f"time_to_live_ms must be in [1000, 5000]; got {ttl}"


def test_cancel_by_oid_does_not_carry_ttl(client: GrvtClient) -> None:
    """``time_to_live_ms`` is a cloid-specific feature — oid cancels don't need it."""
    capturer = _CapturingHttp()
    client._http = capturer  # type: ignore[assignment]
    client.cancel_order("ETH_USDT_Perp", 1334440892959520532621925129411760553)
    payload = capturer.last_payload or {}
    assert payload.get("order_id")
    assert "time_to_live_ms" not in payload
    assert "client_order_id" not in payload
