import os
import tempfile

import pytest
from pydantic import ValidationError

from app.config import require_trading_credentials_when_enabled
from tests.settings_helpers import UnitTestSettings


def _base() -> dict:
    return {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
    }


def test_inventory_soft_must_be_below_hard() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate(
            {
                **_base(),
                "INVENTORY_SOFT_LIMIT_PCT": 0.9,
                "INVENTORY_HARD_LIMIT_PCT": 0.5,
            }
        )


def test_min_spread_cannot_exceed_max() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate(
            {
                **_base(),
                "MIN_HALF_SPREAD_BPS": 50,
                "MAX_HALF_SPREAD_BPS": 10,
            }
        )


def test_quote_loop_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate({**_base(), "QUOTE_LOOP_SECONDS": 0})


def test_account_rest_min_interval_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate({**_base(), "ACCOUNT_REST_MIN_INTERVAL_SECONDS": 0})


def test_vol_window_at_least_four() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate({**_base(), "VOL_WINDOW_SAMPLES": 3})


def test_stale_kill_must_exceed_warn() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate(
            {
                **_base(),
                "STALE_DATA_WARN_SECONDS": 60,
                "STALE_DATA_KILL_SECONDS": 30,
            }
        )


def test_exchange_backoff_cap_must_cover_base() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate(
            {
                **_base(),
                "EXCHANGE_RETRY_BASE_SECONDS": 2,
                "EXCHANGE_RETRY_MAX_BACKOFF_SECONDS": 1,
            }
        )


def test_max_order_notional_must_not_exceed_position_cap() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate(
            {
                **_base(),
                "MAX_ORDER_NOTIONAL_USD": 500,
                "MAX_POSITION_NOTIONAL_USD": 100,
            }
        )


def test_hl_secret_key_from_env_used_when_trading() -> None:
    key = "0x" + "11" * 32
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": key,
            "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
        }
    )
    assert s.hl_secret_key == key


def test_hl_secret_key_file_used_when_hl_secret_key_empty() -> None:
    key = "0x" + "22" * 32
    fd, path = tempfile.mkstemp(suffix=".key")
    try:
        os.write(fd, f"  \n{key}\n  ".encode())
        os.close(fd)
        s = UnitTestSettings.model_validate(
            {
                "TRADING_ENABLED": True,
                "HL_SECRET_KEY": "",
                "HL_SECRET_KEY_FILE": path,
                "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
            }
        )
        assert s.hl_secret_key == key
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_hl_secret_key_env_wins_over_file() -> None:
    env_key = "0x" + "33" * 32
    file_key = "0x" + "44" * 32
    fd, path = tempfile.mkstemp(suffix=".key")
    try:
        os.write(fd, file_key.encode())
        os.close(fd)
        s = UnitTestSettings.model_validate(
            {
                "TRADING_ENABLED": True,
                "HL_SECRET_KEY": env_key,
                "HL_SECRET_KEY_FILE": path,
                "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
            }
        )
        assert s.hl_secret_key == env_key
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_trading_requires_hl_secret_key_or_file() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
        }
    )
    with pytest.raises(ValueError, match="HL_SECRET_KEY"):
        require_trading_credentials_when_enabled(s)


def test_trading_requires_hl_account_when_enabled() -> None:
    key = "0x" + "11" * 32
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": key,
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    with pytest.raises(ValueError, match="HL_ACCOUNT_ADDRESS"):
        require_trading_credentials_when_enabled(s)


def test_grvt_secret_file_used_when_secret_empty() -> None:
    key = "0x" + "55" * 32
    fd, path = tempfile.mkstemp(suffix=".grvt.key")
    try:
        os.write(fd, f"\n{key}\n".encode())
        os.close(fd)
        s = UnitTestSettings.model_validate(
            {
                **_base(),
                "EXCHANGE": "grvt",
                "GRVT_API_KEY": "k",
                "GRVT_API_SECRET": "",
                "GRVT_API_SECRET_FILE": path,
                "GRVT_SUB_ACCOUNT_ID": "123",
            }
        )
        assert s.grvt_api_secret == key
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def test_grvt_env_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError):
        UnitTestSettings.model_validate({**_base(), "GRVT_ENV": "invalid-env"})
