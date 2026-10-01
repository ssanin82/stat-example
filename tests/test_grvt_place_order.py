"""Unit tests for GRVT create_order payload construction and error diagnostics.

Context: the first GRVT-live run rejected every ``POST /full/v1/create_order``
with HTTP 400. Root causes we now guard against here:

  1. The bot produced Hyperliquid-shaped cloids ("0x" + 32 hex) for GRVT, but
     GRVT validates ``metadata.client_order_id`` as a uint64 decimal string.
  2. The signature payload carried an extra ``chain_id`` field not in the pysdk
     schema (``grvt_raw_types.Signature``).
  3. Size / limit_price wire strings were floated through ``str(float)`` and
     could drift off the integer multiple encoded in the EIP-712 message.
  4. The bot's 400 path logged nothing, so we now assert the adapter emits a
     structured ``grvt_create_order_rejected`` line with business context.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pytest

from app.enums import Side
from app.exchange import grvt_client as grvt_client_module
from app.exchange.grvt_client import GrvtClient, _decimal_from_scaled_int


# ---------------------------------------------------------------------------
# Boot fixture: build a GrvtClient without actually hitting the network.
# ---------------------------------------------------------------------------

_ETH_INSTRUMENT_ROW = {
    "instrument": "ETH_USDT_Perp",
    "tick_size": "0.01",
    "min_size": "0.001",
    "min_notional": "20",
    "base_decimals": 9,
    "instrument_hash": "197633",
}


class _FakeMetaResponse:
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

    def post(self, _url: str, *, json: dict[str, Any]) -> _FakeMetaResponse:
        assert json == {"is_active": True}
        return _FakeMetaResponse(self._rows)


# Deterministic private key (0x…eadbeef…). Never used for real trading —
# eth_account just needs a well-formed 32-byte secret to sign the EIP-712
# order. The derived address is not sensitive.
_TEST_PRIVATE_KEY = "0x" + "11" * 32


@pytest.fixture()
def grvt_client(monkeypatch: pytest.MonkeyPatch) -> GrvtClient:
    monkeypatch.setattr(
        grvt_client_module.httpx,
        "Client",
        lambda *_args, **_kwargs: _FakeMetaClient([_ETH_INSTRUMENT_ROW]),
    )
    return GrvtClient(
        config={
            "symbol": "ETH_USDT_Perp",
            "api_key": "test-api-key",
            "api_secret": _TEST_PRIVATE_KEY,
            "sub_account_id": "2132029331314367",
            "env": "prod",
        }
    )


# ---------------------------------------------------------------------------
# EIP-712 <-> REST payload shape.
# ---------------------------------------------------------------------------


def test_signed_payload_matches_pysdk_schema(grvt_client: GrvtClient) -> None:
    """All top-level Order, OrderLeg and Signature keys match the pysdk spec."""
    payload = grvt_client._sign_order_payload(
        instrument="ETH_USDT_Perp",
        is_market=False,
        is_buy=True,
        sz=0.005,
        limit_px=2500.12,
        post_only=True,
        reduce_only=False,
        client_order_id="12345678901234567890",
        time_in_force="GOOD_TILL_TIME",
    )
    order = payload["order"]
    assert set(order.keys()) == {
        "sub_account_id",
        "is_market",
        "time_in_force",
        "post_only",
        "reduce_only",
        "legs",
        "signature",
        "metadata",
    }
    assert order["time_in_force"] == "GOOD_TILL_TIME"
    assert order["post_only"] is True
    assert order["reduce_only"] is False
    leg = order["legs"][0]
    assert set(leg.keys()) == {"instrument", "size", "limit_price", "is_buying_asset"}
    assert leg["instrument"] == "ETH_USDT_Perp"
    assert leg["is_buying_asset"] is True
    sig = order["signature"]
    # pysdk's Signature dataclass has exactly these fields. Extra fields
    # (previously ``chain_id``) were removed — they either confused strict
    # backend validation or at best were dead weight on the wire.
    assert set(sig.keys()) == {"signer", "r", "s", "v", "expiration", "nonce"}
    assert isinstance(sig["v"], int)
    assert isinstance(sig["nonce"], int)
    assert isinstance(sig["expiration"], str)
    assert sig["r"].startswith("0x") and len(sig["r"]) == 66
    assert sig["s"].startswith("0x") and len(sig["s"]) == 66
    assert order["metadata"] == {"client_order_id": "12345678901234567890"}


def test_signed_payload_size_matches_signed_integer(grvt_client: GrvtClient) -> None:
    """``leg.size`` serializes the signed contract_size back to its decimal form.

    Using ``str(float)`` drifts — e.g. ``str(0.007142857142857143)`` prints 17
    digits and does not necessarily round-trip back to the integer multiple the
    signature was computed over. We format from the signed integer instead.
    """
    payload = grvt_client._sign_order_payload(
        instrument="ETH_USDT_Perp",
        is_market=False,
        is_buy=True,
        sz=0.007142857142857143,
        limit_px=2500.0,
        post_only=True,
        reduce_only=False,
        client_order_id="1",
        time_in_force="GOOD_TILL_TIME",
    )
    leg = payload["order"]["legs"][0]
    # base_decimals=9 → signed contract_size = floor(0.007142857142857143 * 1e9)
    # = 7_142_857 → wire string "0.007142857".
    assert leg["size"] == "0.007142857"
    assert leg["limit_price"] == "2500"


def test_market_order_uses_zero_limit_price(grvt_client: GrvtClient) -> None:
    payload = grvt_client._sign_order_payload(
        instrument="ETH_USDT_Perp",
        is_market=True,
        is_buy=False,
        sz=0.1,
        limit_px=0.0,
        post_only=False,
        reduce_only=True,
        client_order_id="1",
        time_in_force="IMMEDIATE_OR_CANCEL",
    )
    leg = payload["order"]["legs"][0]
    assert leg["limit_price"] == "0"
    assert leg["is_buying_asset"] is False
    assert payload["order"]["is_market"] is True


# ---------------------------------------------------------------------------
# Decimal-from-scaled-int helper.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scaled,decimals,expected",
    [
        (5_000_000, 9, "0.005"),
        (7_142_857, 9, "0.007142857"),
        (2500_000_000_000, 9, "2500"),
        (0, 9, "0"),
        (123, 0, "123"),
    ],
)
def test_decimal_from_scaled_int(scaled: int, decimals: int, expected: str) -> None:
    assert _decimal_from_scaled_int(scaled, decimals) == expected


# ---------------------------------------------------------------------------
# Cloid format: GRVT expects uint64 decimal, NOT the HL 0x-hex form.
# ---------------------------------------------------------------------------


def test_grvt_make_client_order_id_is_uint64_decimal(grvt_client: GrvtClient) -> None:
    cloid = grvt_client.make_client_order_id(
        "ETH_USDT_Perp", Side.BUY, "q1", 2500.12, 0.005
    )
    assert re.fullmatch(r"\d+", cloid), f"expected decimal uint64 string, got {cloid!r}"
    assert 0 <= int(cloid) < (1 << 64)


# ---------------------------------------------------------------------------
# Diagnostics: both the transport and the business context are logged on 4xx.
# ---------------------------------------------------------------------------


def _patch_http_post_to_return(
    client: GrvtClient, *, status_code: int, body: dict[str, Any]
) -> None:
    class _Resp:
        def __init__(self) -> None:
            self.status_code = status_code
            self.text = ""

        def json(self) -> dict[str, Any]:
            return dict(body)

    class _Http:
        def post(self, _url: str, **_kwargs: Any) -> _Resp:
            return _Resp()

    client._http = _Http()  # type: ignore[assignment]
    # Short-circuit the cookie refresh: pretend we already have a fresh cookie.
    client._cookie_gravity = "fake-cookie"
    client._cookie_expiry_epoch = 1e18


def test_grvt_400_logs_http_and_rejection_context(
    grvt_client: GrvtClient, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_http_post_to_return(
        grvt_client,
        status_code=400,
        body={"code": 3, "message": "invalid client_order_id: expected uint64"},
    )
    with caplog.at_level(logging.WARNING, logger="app.exchange.grvt_client"):
        resp = grvt_client.place_post_only_limit(
            "ETH_USDT_Perp",
            is_buy=True,
            sz=0.005,
            limit_px=2500.12,
            client_order_id="12345678901234567890",
        )
    assert resp["code"] == 3
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "grvt_http_non_2xx op=create_order" in text
    assert "status=400" in text
    assert "grvt_create_order_rejected" in text
    assert "instrument=ETH_USDT_Perp" in text
    assert "side=BUY" in text
    assert "post_only=True" in text
    assert "time_in_force=GOOD_TILL_TIME" in text
    # Raw API key must never appear anywhere near the failure log — the cookie
    # refresh path is the only place it is allowed to leave the process.
    assert "test-api-key" not in text
    # Signer must be masked so public dashboards don't leak the full address.
    assert grvt_client._wallet.address not in text  # type: ignore[union-attr]
