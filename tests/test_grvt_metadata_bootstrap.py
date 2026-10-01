from __future__ import annotations

from typing import Any

import pytest

from app.exchange.grvt_client import GrvtClient


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http status {self.status_code}")

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeHttpClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    def __enter__(self) -> _FakeHttpClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, _url: str, *, json: dict[str, Any]) -> _FakeResponse:
        assert json == {"is_active": True}
        return self._response


def _mock_httpx_client(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    def _factory(*args: Any, **kwargs: Any) -> _FakeHttpClient:
        _ = args, kwargs
        return _FakeHttpClient(_FakeResponse(payload))

    monkeypatch.setattr("app.exchange.grvt_client.httpx.Client", _factory)


def _instrument_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "instrument": "ETH_USDT_Perp",
        "tick_size": "0.1",
        "min_size": "0.001",
        "min_notional": "10",
        "base_decimals": 3,
        "instrument_hash": "0x1E240",
    }
    row.update(overrides)
    return row


def test_grvt_metadata_bootstrap_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_httpx_client(monkeypatch, {"result": [_instrument_row()]})
    c = GrvtClient(config={"symbol": "ETH_USDT_Perp"})
    assert c.symbol_spec_fetched_ok is True
    assert c.symbol_spec.price_tick == 0.1
    assert c.symbol_spec.size_step == 0.001
    assert c.symbol_spec.min_size == 0.001
    assert c.symbol_spec.min_notional_usd == 10.0
    assert c.symbol_spec.sz_decimals == 3
    assert c.symbol_spec.source == "grvt_meta"


def test_grvt_metadata_bootstrap_unknown_symbol_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_httpx_client(monkeypatch, {"result": [_instrument_row(instrument="BTC_USDT_Perp")]})
    with pytest.raises(RuntimeError, match="unknown symbol"):
        GrvtClient(config={"symbol": "ETH_USDT_Perp"})


def test_grvt_metadata_bootstrap_missing_required_field_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = _instrument_row()
    del bad["min_notional"]
    _mock_httpx_client(monkeypatch, {"result": [bad]})
    with pytest.raises(ValueError, match="min_notional"):
        GrvtClient(config={"symbol": "ETH_USDT_Perp"})


def test_grvt_metadata_bootstrap_malformed_numeric_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_httpx_client(
        monkeypatch,
        {"result": [_instrument_row(min_size="not-a-number")]},
    )
    with pytest.raises(ValueError, match="min_size"):
        GrvtClient(config={"symbol": "ETH_USDT_Perp"})


def test_grvt_metadata_bootstrap_no_fallback_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_httpx_client(monkeypatch, {"result": []})
    with pytest.raises(RuntimeError):
        GrvtClient(config={"symbol": "ETH_USDT_Perp"})
