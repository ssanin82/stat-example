"""Execution must get its cloid from the adapter, not from an HL-specific helper.

Regression guard for the first-run GRVT 400 bug, where execution hard-coded
``make_deterministic_cloid_hex`` and sent Hyperliquid-shaped cloids to GRVT,
which validates ``metadata.client_order_id`` as a uint64 decimal string.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.enums import Side
from app.exchange.hyperliquid_responses import make_deterministic_cloid_hex
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _make_exec(cloid_fn) -> tuple[OrderManager, MagicMock]:
    settings = UnitTestSettings.model_validate(
        {"DATABASE_URL": "sqlite:///:memory:", "TRADING_ENABLED": "true"}
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = MagicMock()
    client.symbol_spec = FALLBACK_SYMBOL_SPEC
    client.symbol_spec_fetched_ok = True
    client.has_write_access.return_value = True
    client.make_client_order_id.side_effect = cloid_fn
    om = OrderManager(settings, client, storage, state)
    return om, client


def test_stage_place_order_uses_adapter_cloid() -> None:
    """``_stage_place_order_local`` must ask the adapter for the cloid."""
    captured = {}

    def _cloid(symbol, side, quote_cycle_id, price, size):
        captured["args"] = (symbol, side, quote_cycle_id, price, size)
        return "99999999"

    om, client = _make_exec(_cloid)
    wo = om._stage_place_order_local(
        Side.BUY, price=2500.12, size=0.011, quote_cycle_id="q-abc"
    )
    assert wo is not None
    client.make_client_order_id.assert_called_once()
    assert wo.client_order_id == "99999999"
    assert captured["args"][0] == om._settings.symbol
    assert captured["args"][1] == Side.BUY
    assert captured["args"][2] == "q-abc"


def test_stage_place_order_uses_hl_cloid_shape_on_hl_adapter() -> None:
    """Via the real HL helper the cloid must still be 0x + 32 hex chars."""
    om, _client = _make_exec(make_deterministic_cloid_hex)
    wo = om._stage_place_order_local(
        Side.SELL, price=2500.0, size=0.01, quote_cycle_id="q-abc"
    )
    assert wo is not None
    assert wo.client_order_id.startswith("0x")
    assert len(wo.client_order_id) == 34  # "0x" + 32 hex


def test_stage_place_order_uses_grvt_cloid_shape_on_grvt_adapter() -> None:
    """Via the GRVT helper the cloid must be a decimal uint64 string."""
    from app.exchange.grvt_responses import make_deterministic_grvt_client_order_id

    om, _client = _make_exec(make_deterministic_grvt_client_order_id)
    wo = om._stage_place_order_local(
        Side.SELL, price=2500.0, size=0.01, quote_cycle_id="q-abc"
    )
    assert wo is not None
    assert wo.client_order_id.isdigit()
    assert 0 <= int(wo.client_order_id) < (1 << 64)
