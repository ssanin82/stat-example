"""``startup_readiness`` log lines must be venue-accurate.

The bot used to emit HL-specific rule lines (``hl_limit_price_rules``,
``hl_price_normalize_pipeline=hl_perp_decimal_grid_plus_max_sig_figs_nonint``)
even when running against GRVT. Operators reading the startup banner would
reasonably conclude Hyperliquid's 5-sig-figs rule was active, which it isn't
on GRVT. The HL banner is preserved for the HL path (existing test guards it);
here we guard the GRVT path.
"""

from __future__ import annotations

import logging

import pytest

from app.exchange.symbol_spec import SymbolSpec
from app.startup_report import log_startup_readiness
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _grvt_spec() -> SymbolSpec:
    return SymbolSpec(
        price_tick=0.01,
        size_step=0.001,
        min_size=0.001,
        min_notional_usd=20.0,
        sz_decimals=9,
        source="grvt_meta",
    )


def test_grvt_startup_report_does_not_claim_hl_rules(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = UnitTestSettings.model_validate(
        {
            "EXCHANGE": "grvt",
            "GRVT_API_KEY": "test-key",
            "GRVT_API_SECRET": "0x" + "11" * 32,
            "GRVT_SUB_ACCOUNT_ID": "1",
        }
    )
    client = mock_mm_client(symbol_spec=_grvt_spec())
    client.has_write_access.return_value = True
    client.fetch_account_snapshot.side_effect = RuntimeError("skip-account-fetch")

    with caplog.at_level(logging.INFO, logger="app.startup_report"):
        log_startup_readiness(settings, client)

    text = "\n".join(r.message for r in caplog.records)
    assert "grvt_symbol_meta" in text
    assert "grvt_limit_price_rules" in text
    assert "grvt_order_sizing_floors" in text
    # The HL sig-figs rule doesn't apply to GRVT; the banner must not claim it.
    assert "hl_limit_price_rules" not in text
    assert "hl_max_sig_figs_nonint" not in text
    assert "hl_perp_decimal_grid_plus_max_sig_figs_nonint" not in text
