"""Signing hot-path caches survive repeated calls.

The signing path runs per-place and per-replace, so anything expensive that
can be computed once is worth caching. These tests guard three caches:

- Per-instrument ``_InstrumentSignCache`` (instrument_hash + base_decimals +
  size_multiplier_decimal).
- Module-level ``_TIF_MAP`` + ``_ORDER_EIP712_TYPES`` + ``_PRICE_MULTIPLIER_DEC``.
- Per-client ``_eip712_domain`` + ``_sub_account_id_int`` + ``_wallet_address_str``.

They're narrow "the wiring is actually in place" tests — they do not claim
the signatures themselves change, only that the expensive bits aren't
rebuilt on every call.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from app.exchange import grvt_client as grvt_client_module
from app.exchange.grvt_client import (
    GrvtClient,
    _PRICE_MULTIPLIER_DEC,
    _TIF_MAP,
)


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


_TEST_PRIVATE_KEY = "0x" + "22" * 32


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> GrvtClient:
    monkeypatch.setattr(
        grvt_client_module.httpx,
        "Client",
        lambda *_args, **_kwargs: _FakeMetaClient([_ETH_ROW]),
    )
    return GrvtClient(
        config={
            "symbol": "ETH_USDT_Perp",
            "api_key": "test-api-key",
            "api_secret": _TEST_PRIVATE_KEY,
            "sub_account_id": "42",
            "env": "prod",
        }
    )


def test_module_level_constants_are_stable() -> None:
    assert _TIF_MAP["GOOD_TILL_TIME"] == 1
    assert _TIF_MAP["IMMEDIATE_OR_CANCEL"] == 3
    assert _PRICE_MULTIPLIER_DEC == Decimal(1_000_000_000)


def test_client_precomputes_sub_account_and_domain(client: GrvtClient) -> None:
    assert client._sub_account_id_str == "42"
    assert client._sub_account_id_int == 42
    assert client._chain_id == 325
    assert client._eip712_domain == {
        "name": "GRVT Exchange",
        "version": "0",
        "chainId": 325,
    }
    # Wallet-address string materialised once, used as the ``signature.signer``
    # on every signed order.
    assert client._wallet_address_str.startswith("0x")
    assert len(client._wallet_address_str) == 42


def test_instrument_sign_cache_populates_once_and_is_reused(
    client: GrvtClient,
) -> None:
    assert client._instrument_sign_cache == {}
    p1 = client._instrument_sign_params("ETH_USDT_Perp")
    p2 = client._instrument_sign_params("ETH_USDT_Perp")
    # Exact same cache object — dict lookup, not recomputation.
    assert p1 is p2
    assert p1.instrument_hash == 197633
    assert p1.size_decimals == 9
    assert p1.size_multiplier_dec == Decimal(10) ** 9


def test_instrument_sign_cache_is_invalidated_on_metadata_refetch(
    client: GrvtClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Populate cache.
    client._instrument_sign_params("ETH_USDT_Perp")
    assert client._instrument_sign_cache != {}
    # Simulate stale metadata triggering a reload — clear the loaded dict so
    # ``_instrument_row`` takes the re-fetch branch.
    client._instruments_by_symbol = {}
    monkeypatch.setattr(
        grvt_client_module.httpx,
        "Client",
        lambda *_args, **_kwargs: _FakeMetaClient([_ETH_ROW]),
    )
    # Accessing any instrument row forces re-fetch and cache invalidation.
    client._instrument_row("ETH_USDT_Perp")
    assert client._instrument_sign_cache == {}
