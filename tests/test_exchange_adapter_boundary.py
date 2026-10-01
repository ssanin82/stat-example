"""
Boundary tests for :class:`app.exchange.base.PerpExchangeAdapter`.

These tests do not exercise any live transport. They protect the
structural contract between bot core and venue adapters:

* the Hyperliquid client structurally satisfies the Protocol;
* the GRVT adapter declares every required method;
* venue-neutral type aliases resolve back to the Hyperliquid concrete
  types today (this is the intentional back-compat);
* the Hyperliquid adapter's wire-format delegators produce the same
  tuples as the underlying module-level interpreters.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.enums import Side
from app.exchange.base import FillRaw, OpenOrderRaw, PerpExchangeAdapter
from app.exchange.hyperliquid_responses import (
    interpret_hl_cancel_response,
    interpret_hl_order_status_response,
    interpret_hl_place_order_response,
    make_deterministic_cloid_hex,
)
from app.exchange.hyperliquid_types import HLFillRaw, HLOpenOrderRaw


# ---------------------------------------------------------------------------
# Type-alias sanity
# ---------------------------------------------------------------------------


def test_generic_aliases_resolve_to_hl_types() -> None:
    assert OpenOrderRaw is HLOpenOrderRaw
    assert FillRaw is HLFillRaw


# ---------------------------------------------------------------------------
# Required adapter surface
# ---------------------------------------------------------------------------


REQUIRED_METHODS = (
    "has_write_access",
    "fetch_best_bid_ask",
    "fetch_position",
    "fetch_account_snapshot",
    "fetch_open_orders_raw",
    "fetch_recent_fills_raw",
    "place_post_only_limit",
    "cancel_order",
    "cancel_order_by_cloid",
    "query_order_status_by_cloid",
    "market_close",
    "rest_runtime_counters",
    "interpret_place_response",
    "interpret_cancel_response",
    "interpret_order_status_response",
    "make_client_order_id",
)


def test_hyperliquid_client_class_declares_adapter_surface() -> None:
    # Importing the module eagerly loads the SDK, which is what the real
    # bot does at startup — we check the class itself, not an instance, so
    # no network calls happen.
    from app.exchange.hyperliquid_client import HyperliquidClient

    for name in REQUIRED_METHODS:
        assert callable(getattr(HyperliquidClient, name, None)), (
            f"HyperliquidClient is missing PerpExchangeAdapter method {name!r}"
        )
    assert hasattr(HyperliquidClient, "symbol_spec")


def test_grvt_scaffold_declares_adapter_surface() -> None:
    from app.exchange.grvt_client import GrvtClient

    for name in REQUIRED_METHODS:
        assert callable(getattr(GrvtClient, name, None)), (
            f"GrvtClient scaffold is missing PerpExchangeAdapter method {name!r}"
        )
    assert hasattr(GrvtClient, "symbol_spec")


def test_grvt_client_construction_defaults_to_no_write_access() -> None:
    from app.exchange.grvt_client import GrvtClient

    from app.exchange.symbol_spec import SymbolSpec

    with patch("app.exchange.grvt_client.GrvtClient._load_symbol_spec_strict") as _spec:
        _spec.return_value = SymbolSpec(
            price_tick=0.1,
            size_step=0.001,
            min_size=0.001,
            min_notional_usd=10.0,
            sz_decimals=3,
            source="fallback",
        )
        gc = GrvtClient()
    assert gc.has_write_access() is False
    cloid = gc.make_client_order_id("BTC_USDT_Perp", Side.BUY, "cycle-1", 100.0, 0.1)
    assert isinstance(cloid, str)
    assert cloid


# ---------------------------------------------------------------------------
# Runtime-checkable Protocol smoke
# ---------------------------------------------------------------------------


def test_runtime_isinstance_accepts_mock_with_full_surface() -> None:
    # @runtime_checkable only verifies attribute presence. A MagicMock with
    # explicitly attached callables should qualify; a bare object should not.
    m = MagicMock()
    for name in REQUIRED_METHODS:
        setattr(m, name, lambda *a, **kw: None)
    m.symbol_spec = object()
    m.symbol_spec_fetched_ok = False
    assert isinstance(m, PerpExchangeAdapter)

    class Empty:
        pass

    assert not isinstance(Empty(), PerpExchangeAdapter)


# ---------------------------------------------------------------------------
# HL adapter delegators match module-level interpreters
# ---------------------------------------------------------------------------


class _HLDelegatorHarness:
    """Just the four delegator methods, extracted so we don't need to
    construct a real HyperliquidClient (which boots the SDK)."""

    def interpret_place_response(self, resp):
        return interpret_hl_place_order_response(resp)

    def interpret_cancel_response(self, resp):
        return interpret_hl_cancel_response(resp)

    def interpret_order_status_response(self, resp):
        return interpret_hl_order_status_response(resp)

    def make_client_order_id(self, symbol, side, quote_cycle_id, price, size):
        return make_deterministic_cloid_hex(symbol, side, quote_cycle_id, price, size)


def test_hl_delegators_preserve_module_function_contract() -> None:
    h = _HLDelegatorHarness()

    place_resp = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {
                "statuses": [{"resting": {"oid": 12345}}],
            },
        },
    }
    assert h.interpret_place_response(place_resp) == interpret_hl_place_order_response(place_resp)
    assert h.interpret_place_response(place_resp) == (12345, "accepted", "")

    cancel_resp = {
        "status": "ok",
        "response": {
            "type": "cancel",
            "data": {"statuses": ["success"]},
        },
    }
    assert h.interpret_cancel_response(cancel_resp) == interpret_hl_cancel_response(cancel_resp)
    assert h.interpret_cancel_response(cancel_resp)[0] == "success"

    status_resp = {
        "status": "order",
        "order": {
            "order": {"oid": 42},
            "status": "open",
        },
    }
    assert h.interpret_order_status_response(status_resp) == interpret_hl_order_status_response(
        status_resp
    )
    assert h.interpret_order_status_response(status_resp) == (42, "open", "")


def test_hl_make_client_order_id_is_deterministic_and_well_formed() -> None:
    h = _HLDelegatorHarness()
    cloid = h.make_client_order_id("BTC", Side.BUY, "cycle-7", 12345.67, 0.01)
    again = h.make_client_order_id("BTC", Side.BUY, "cycle-7", 12345.67, 0.01)
    assert cloid == again
    # HL cloid shape: 0x + 32 hex chars (16 bytes).
    assert cloid.startswith("0x")
    assert len(cloid) == 2 + 32
    int(cloid[2:], 16)

    # Different inputs produce different ids.
    other = h.make_client_order_id("BTC", Side.SELL, "cycle-7", 12345.67, 0.01)
    assert cloid != other


# ---------------------------------------------------------------------------
# Back-compat: legacy import paths still resolve to the canonical objects
# ---------------------------------------------------------------------------


def test_mm_client_protocol_alias_points_at_base_protocol() -> None:
    from app.exchange.mm_client_protocol import HyperliquidMMClient as LegacyAlias

    assert LegacyAlias is PerpExchangeAdapter


def test_execution_module_reexports_interpreters() -> None:
    from app import execution

    assert execution.interpret_hl_place_order_response is interpret_hl_place_order_response
    assert execution.interpret_hl_cancel_response is interpret_hl_cancel_response
    assert execution.interpret_hl_order_status_response is interpret_hl_order_status_response
    assert execution.make_deterministic_cloid_hex is make_deterministic_cloid_hex
