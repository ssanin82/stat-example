from unittest.mock import patch

import pytest

from app.exchange.exchange_retry import (
    ServerError,
    _is_transient_exchange_error,
    exchange_call_with_retry,
)
from tests.settings_helpers import UnitTestSettings


def _settings(**kw: object) -> UnitTestSettings:
    data = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "EXCHANGE_RETRY_MAX_ATTEMPTS": 3,
        "EXCHANGE_RETRY_BASE_SECONDS": 0.01,
        "EXCHANGE_RETRY_MAX_BACKOFF_SECONDS": 0.05,
        "EXCHANGE_RATE_LIMIT_EXTRA_DELAY_SECONDS": 0.0,
    }
    for k, v in kw.items():
        data[k.upper() if k.islower() else k] = v
    return UnitTestSettings.model_validate(data)


def test_retry_succeeds_after_transient() -> None:
    s = _settings()
    n = {"i": 0}

    def fn():
        n["i"] += 1
        if n["i"] < 2:
            raise ConnectionError("boom")
        return "ok"

    with patch("app.exchange.exchange_retry.time.sleep"):
        assert exchange_call_with_retry("test_op", fn, s) == "ok"
    assert n["i"] == 2


def test_retry_exhausted_raises() -> None:
    s = _settings(EXCHANGE_RETRY_MAX_ATTEMPTS=2)

    def fn():
        raise ConnectionError("always")

    with patch("app.exchange.exchange_retry.time.sleep"):
        with pytest.raises(ConnectionError):
            exchange_call_with_retry("test_op", fn, s)


def test_non_transient_not_retried() -> None:
    s = _settings(EXCHANGE_RETRY_MAX_ATTEMPTS=5)
    n = {"i": 0}

    def fn():
        n["i"] += 1
        raise ValueError("logic")

    with patch("app.exchange.exchange_retry.time.sleep"):
        with pytest.raises(ValueError):
            exchange_call_with_retry("test_op", fn, s)
    assert n["i"] == 1


def test_server_error_transient() -> None:
    assert _is_transient_exchange_error(ServerError(503, "bad"))
