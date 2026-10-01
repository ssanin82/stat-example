"""
Hyperliquid perp limit price: decimal grid + max 5 significant figures (non-integers).

Docs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.exchange.hyperliquid_precision import (
    normalize_hyperliquid_perp_limit_price,
    normalize_order_pair,
    price_tick_decimal,
    price_to_float_for_hyperliquid_wire,
    validate_hyperliquid_perp_limit_for_submit,
    wire_format_preview_limit_px,
)
from app.exchange.symbol_spec import SymbolSpec, symbol_spec_from_hyperliquid_meta


def _eth_spec() -> SymbolSpec:
    meta = {
        "universe": [
            {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
        ],
    }
    return symbol_spec_from_hyperliquid_meta(meta, "ETH")


def test_eth_decimal_grid_is_point_01_but_sig_figs_stricter() -> None:
    """``price_tick`` from meta is 0.01; live rejects 2244.95 (six significant figures)."""
    spec = _eth_spec()
    assert abs(spec.price_tick - 0.01) < 1e-18
    tick_dec = price_tick_decimal(spec.sz_decimals)
    assert tick_dec == Decimal("0.01")
    on_grid = Decimal("2244.95") % tick_dec == 0
    assert on_grid
    with pytest.raises(ValueError, match="significant figures"):
        validate_hyperliquid_perp_limit_for_submit(
            2244.95,
            spec.sz_decimals,
            source_tick=float(spec.price_tick),
        )


@pytest.mark.parametrize(
    "raw,expected_dec",
    [
        ("2244.95", "2245"),
        ("2248.55", "2248.60"),
        ("2245.05", "2245.10"),
        ("2248.65", "2248.70"),
    ],
)
def test_live_rejected_examples_normalize_to_exchange_shape(raw: str, expected_dec: str) -> None:
    spec = _eth_spec()
    out = normalize_hyperliquid_perp_limit_price(Decimal(raw), spec.sz_decimals)
    assert out == Decimal(expected_dec)
    px_f = price_to_float_for_hyperliquid_wire(out, spec.sz_decimals)
    validate_hyperliquid_perp_limit_for_submit(
        px_f,
        spec.sz_decimals,
        source_tick=float(spec.price_tick),
    )


def test_normalize_order_pair_applies_sig_fig_rule_not_just_tick() -> None:
    """Regression: metadata implies tick 0.01; pair normalizer must still fix sig figs."""
    spec = _eth_spec()
    pair, rej = normalize_order_pair(spec, 2244.95, 0.05)
    assert rej is None
    assert pair is not None
    px, sz = pair
    assert abs(px - 2245.0) < 1e-9
    assert sz > 0


def test_wire_payload_matches_sdk_float_to_wire() -> None:
    spec = _eth_spec()
    out = normalize_hyperliquid_perp_limit_price(Decimal("2244.95"), spec.sz_decimals)
    px_f = price_to_float_for_hyperliquid_wire(out, spec.sz_decimals)
    preview = wire_format_preview_limit_px(px_f)
    try:
        from hyperliquid.utils.signing import float_to_wire

        assert preview == float_to_wire(px_f)
    except ImportError:
        assert preview == "2245"


def test_btc_metadata_suggests_point_one_tick_coarser_grid() -> None:
    meta = {
        "universe": [
            {"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
        ],
    }
    spec = symbol_spec_from_hyperliquid_meta(meta, "BTC")
    assert abs(spec.price_tick - 0.1) < 1e-18
    out = normalize_hyperliquid_perp_limit_price(Decimal("97000.55"), spec.sz_decimals)
    # On 0.1 grid first, then six significant figures force a coarser 5-sf round-up to 97001.0.
    assert out == Decimal("97001.0")


def test_eth_three_thousand_point_oh_fifteen_rounds_for_hl() -> None:
    spec = _eth_spec()
    pair, rej = normalize_order_pair(spec, 3000.015, 0.01)
    assert rej is None and pair is not None
    px, _ = pair
    assert abs(px - 3000.0) < 1e-9


def test_validate_accepts_integer_despite_many_digits() -> None:
    spec = _eth_spec()
    validate_hyperliquid_perp_limit_for_submit(
        123456.0,
        spec.sz_decimals,
        source_tick=float(spec.price_tick),
    )
