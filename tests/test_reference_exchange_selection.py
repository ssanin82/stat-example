"""``REFERENCE_EXCHANGE`` validation and selection behaviour.

Production logic: ``app/main.py`` selects between
:class:`BinancePublicStream` and :class:`BybitPublicStream` based on
``settings.reference_exchange``. This module verifies the validator
and sanity-checks the factory path without spinning up the full
FastAPI lifespan.
"""

from __future__ import annotations

import pytest

from app.exchange.binance_public_ws import BinancePublicStream
from app.exchange.bybit_public_ws import BybitPublicStream
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "BINANCE_WS_ENABLED": True,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


# ---------- validator ----------

def test_reference_exchange_default_is_binance() -> None:
    s = _settings()
    assert s.reference_exchange == "binance"


@pytest.mark.parametrize("value", ["binance", "bybit", "off", "BYBIT", "Binance", " off "])
def test_reference_exchange_accepts_valid_values(value) -> None:
    s = _settings(REFERENCE_EXCHANGE=value)
    assert s.reference_exchange == value.strip().lower()


@pytest.mark.parametrize("value", ["okx", "coinbase", "random"])
def test_reference_exchange_rejects_invalid_values(value) -> None:
    with pytest.raises(Exception):
        _settings(REFERENCE_EXCHANGE=value)


def test_empty_reference_exchange_falls_back_to_binance() -> None:
    s = _settings(REFERENCE_EXCHANGE="")
    assert s.reference_exchange == "binance"


# ---------- factory path (mirror of app/main.py logic) ----------

def _make_reference_stream(settings, state):
    """Mirror of the selection block in ``app/main.py``. Kept here so the
    test doesn't spin up the full FastAPI lifespan."""
    if not settings.binance_ws_enabled or settings.reference_exchange == "off":
        return None
    if settings.reference_exchange == "bybit":
        return BybitPublicStream(settings, state, on_bbo_callback=None)
    return BinancePublicStream(settings, state, on_bbo_callback=None)


def test_factory_binance_default() -> None:
    settings = _settings()
    state = BotState(settings)
    stream = _make_reference_stream(settings, state)
    assert isinstance(stream, BinancePublicStream)


def test_factory_bybit_selected() -> None:
    settings = _settings(REFERENCE_EXCHANGE="bybit")
    state = BotState(settings)
    stream = _make_reference_stream(settings, state)
    assert isinstance(stream, BybitPublicStream)


def test_factory_off_returns_none() -> None:
    settings = _settings(REFERENCE_EXCHANGE="off")
    state = BotState(settings)
    stream = _make_reference_stream(settings, state)
    assert stream is None


def test_factory_legacy_kill_switch_overrides_selection() -> None:
    """``BINANCE_WS_ENABLED=false`` must disable any venue choice."""
    settings = _settings(BINANCE_WS_ENABLED=False, REFERENCE_EXCHANGE="bybit")
    state = BotState(settings)
    stream = _make_reference_stream(settings, state)
    assert stream is None
