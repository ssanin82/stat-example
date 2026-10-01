"""Regression tests for execution / quote observability (ticks, pipeline, wire preview)."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

from app.enums import ActiveSides
from app.exchange.hyperliquid_precision import (
    HL_PERP_LIMIT_PRICE_PIPELINE_ID,
    HL_PERP_MAX_SIG_FIGS,
    wire_format_preview_limit_px,
)
from app.exchange.symbol_spec import symbol_spec_from_hyperliquid_meta
from app.execution import OrderManager
from app.models import QuoteDecision
from app.startup_report import log_runtime_config, log_startup_readiness
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _eth_spec():
    meta = {
        "universe": [
            {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
        ],
    }
    return symbol_spec_from_hyperliquid_meta(meta, "ETH")


def test_build_quote_execution_telemetry_eth_includes_meta_pipeline_and_wire() -> None:
    """Norm px uses sig-fig rule; wire string matches SDK preview for that float."""
    path = Path(tempfile.gettempdir()) / f"mm_obs_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 1.0,
            "QUOTE_NOTIONAL_USD": 500.0,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.symbol_spec = _eth_spec()
    om = OrderManager(settings, client, storage, state)

    spec = client.symbol_spec
    mid = 2245.0
    decision = QuoteDecision(
        ts=utc_now(),
        symbol=settings.symbol,
        mid_price=mid,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=mid,
        target_spread_bps=10.0,
        target_bid=2244.95,
        target_ask=2245.05,
        quoted_bid=2244.95,
        quoted_ask=2245.05,
        quoted_bid_sz=0.05,
        quoted_ask_sz=0.05,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="test",
        quote_cycle_id="obs1",
    )
    row = om.build_quote_execution_telemetry(decision, 1.0, 1.0, 0.0)

    assert abs(float(spec.price_tick) - 0.01) < 1e-12
    assert row["exec_meta_decimal_grid_price_tick"] == row["exec_price_tick"]
    assert row["exec_meta_decimal_size_step"] == row["exec_size_step"]
    assert row["exec_hl_max_sig_figs_nonint"] == int(HL_PERP_MAX_SIG_FIGS)
    assert row["exec_price_normalize_pipeline"] == HL_PERP_LIMIT_PRICE_PIPELINE_ID

    assert row["exec_raw_bid_px"] is not None and abs(row["exec_raw_bid_px"] - 2244.95) < 1e-9
    assert row["exec_norm_bid_px"] is not None and abs(row["exec_norm_bid_px"] - 2245.0) < 1e-9
    assert row["exec_wire_bid_limit_p"] == wire_format_preview_limit_px(row["exec_norm_bid_px"])
    assert row["exec_wire_ask_limit_p"] == wire_format_preview_limit_px(row["exec_norm_ask_px"])
    path.unlink(missing_ok=True)


def test_startup_readiness_logs_meta_tick_separately_from_sigfig_pipeline(caplog) -> None:
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    client = mock_mm_client()
    client.symbol_spec = _eth_spec()
    client.symbol_spec_fetched_ok = True
    client.has_write_access.return_value = False

    with caplog.at_level(logging.INFO, logger="app.startup_report"):
        log_startup_readiness(settings, client)

    text = " ".join(r.message for r in caplog.records)
    assert "meta_decimal_grid_price_tick=" in text
    assert "hl_limit_price_rules" in text
    assert HL_PERP_LIMIT_PRICE_PIPELINE_ID in text
    assert "hl_max_sig_figs_nonint=" in text


def test_schema_v6_quote_decisions_has_observability_columns() -> None:
    path = Path(tempfile.gettempdir()) / f"mm_obs_sch_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {"DATABASE_URL": f"sqlite:///{path.as_posix()}"}
    )
    storage = Storage(settings)
    storage.init_schema()
    with storage.connection() as conn:
        cur = conn.execute("PRAGMA table_info(quote_decisions)")
        cols = {r[1] for r in cur.fetchall()}
    assert "exec_wire_bid_limit_p" in cols
    assert "exec_price_normalize_pipeline" in cols
    assert "exec_meta_decimal_grid_price_tick" in cols
    assert "decision_to_submit_dispatch_ms" in cols
    assert "submit_transport_rtt_ms" in cols
    assert "quote_eligibility" in cols
    assert "quote_eligibility_reason" in cols
    path.unlink(missing_ok=True)


def test_startup_runtime_config_log_emitted(caplog) -> None:
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
        }
    )
    with caplog.at_level(logging.INFO, logger="app.startup_report"):
        log_runtime_config(settings)
    text = " ".join(r.message for r in caplog.records)
    assert "bot_runtime_config" in text
    assert "open_orders_reconcile_request_interval_seconds=" in text
    assert "account_rest_min_interval_seconds=" in text
