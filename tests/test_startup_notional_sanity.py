"""Startup emits a WARNING when local min_quote_notional_usd is below venue min,
and an ERROR when max_order_notional_usd is below venue min (impossible to place).

The QuoteEngine's self-heal treats the venue ``min_notional_usd`` as the true
floor — if our local ``MIN_QUOTE_NOTIONAL_USD`` is set below it, the local
setting is effectively ignored for GRVT (where venue min=20) while still being
the published "our floor" for HL (where venue min is lower). The warning makes
the config mismatch visible so operators can align the two.
"""

from __future__ import annotations

import logging

import pytest

from app.exchange.symbol_spec import SymbolSpec
from app.startup_report import log_startup_readiness
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _grvt_spec(min_notional: float = 20.0) -> SymbolSpec:
    return SymbolSpec(
        price_tick=0.01,
        size_step=0.001,
        min_size=0.001,
        min_notional_usd=min_notional,
        sz_decimals=9,
        source="grvt_meta",
    )


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "EXCHANGE": "grvt",
        "GRVT_API_KEY": "test-key",
        "GRVT_API_SECRET": "0x" + "11" * 32,
        "GRVT_SUB_ACCOUNT_ID": "1",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def test_warning_when_local_min_below_venue_min(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """MIN_QUOTE_NOTIONAL_USD=12 below GRVT's 20 → operator gets a warning."""
    settings = _settings(MIN_QUOTE_NOTIONAL_USD=12.0)
    client = mock_mm_client(symbol_spec=_grvt_spec(20.0))
    client.has_write_access.return_value = True
    client.fetch_account_snapshot.side_effect = RuntimeError("skip")

    with caplog.at_level(logging.WARNING, logger="app.startup_report"):
        log_startup_readiness(settings, client)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    msg = " ".join(r.message for r in warnings)
    assert "MIN_QUOTE_NOTIONAL_USD" in msg
    assert "venue min_notional_usd" in msg
    assert "12" in msg and "20" in msg


def test_no_warning_when_local_min_above_venue_min(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With the fix in prod.grvt.env (MIN_QUOTE_NOTIONAL_USD=22 > 20 venue min)
    no warning should fire."""
    settings = _settings(MIN_QUOTE_NOTIONAL_USD=22.0)
    client = mock_mm_client(symbol_spec=_grvt_spec(20.0))
    client.has_write_access.return_value = True
    client.fetch_account_snapshot.side_effect = RuntimeError("skip")

    with caplog.at_level(logging.WARNING, logger="app.startup_report"):
        log_startup_readiness(settings, client)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    msg = " ".join(r.message for r in warnings)
    assert "MIN_QUOTE_NOTIONAL_USD" not in msg


def test_error_when_max_order_below_venue_min(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """max_order_notional_usd < venue min_notional_usd makes every order
    infeasible. Operator must see an ERROR."""
    settings = _settings(MAX_ORDER_NOTIONAL_USD=10.0)  # below venue 20
    client = mock_mm_client(symbol_spec=_grvt_spec(20.0))
    client.has_write_access.return_value = True
    client.fetch_account_snapshot.side_effect = RuntimeError("skip")

    with caplog.at_level(logging.ERROR, logger="app.startup_report"):
        log_startup_readiness(settings, client)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    msg = " ".join(r.message for r in errors)
    assert "MAX_ORDER_NOTIONAL_USD" in msg
    assert "NO ORDER CAN BE PLACED" in msg


def test_no_error_when_max_order_above_venue_min(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(MAX_ORDER_NOTIONAL_USD=60.0)
    client = mock_mm_client(symbol_spec=_grvt_spec(20.0))
    client.has_write_access.return_value = True
    client.fetch_account_snapshot.side_effect = RuntimeError("skip")

    with caplog.at_level(logging.ERROR, logger="app.startup_report"):
        log_startup_readiness(settings, client)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    msg = " ".join(r.message for r in errors)
    assert "MAX_ORDER_NOTIONAL_USD" not in msg
