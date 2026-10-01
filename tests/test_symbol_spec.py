import pytest

from app.exchange.symbol_spec import (
    SymbolSpec,
    symbol_spec_from_hyperliquid_meta,
)


def _meta_eth() -> dict:
    return {
        "universe": [
            {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
            {
                "name": "ETH",
                "szDecimals": 4,
                "maxLeverage": 25,
            },
        ],
    }


def test_eth_spec_matches_sdk_tick_and_step() -> None:
    s = symbol_spec_from_hyperliquid_meta(_meta_eth(), "ETH")
    assert isinstance(s, SymbolSpec)
    assert s.sz_decimals == 4
    assert abs(s.size_step - 1e-4) < 1e-18
    assert abs(s.price_tick - 0.01) < 1e-18
    assert s.min_size == s.size_step
    assert abs(s.min_notional_usd - 10.0) < 1e-9
    assert s.source == "hyperliquid_meta"


def test_btc_spec_fewer_price_decimals() -> None:
    s = symbol_spec_from_hyperliquid_meta(_meta_eth(), "BTC")
    assert s.sz_decimals == 5
    assert abs(s.size_step - 1e-5) < 1e-20
    assert abs(s.price_tick - 0.1) < 1e-18


def test_optional_min_fields_in_row() -> None:
    meta = {
        "universe": [
            {
                "name": "ZZZ",
                "szDecimals": 2,
                "minOrderSz": 0.05,
                "minNotionalUsd": 10.0,
            },
        ],
    }
    s = symbol_spec_from_hyperliquid_meta(meta, "ZZZ")
    assert s.min_size >= 0.05
    assert abs(s.min_notional_usd - 10.0) < 1e-9


def test_min_notional_meta_below_exchange_floor_clamped_to_10() -> None:
    """meta may advertise < $10; Hyperliquid perps still enforce $10 minimum."""
    meta = {
        "universe": [
            {"name": "ETH", "szDecimals": 4, "minNotionalUsd": 1.0},
        ],
    }
    s = symbol_spec_from_hyperliquid_meta(meta, "ETH")
    assert abs(s.min_notional_usd - 10.0) < 1e-9


def test_min_notional_meta_above_floor_preserved() -> None:
    meta = {
        "universe": [
            {"name": "ETH", "szDecimals": 4, "minNotionalUsd": 25.0},
        ],
    }
    s = symbol_spec_from_hyperliquid_meta(meta, "ETH")
    assert abs(s.min_notional_usd - 25.0) < 1e-9


def test_unknown_symbol_raises() -> None:
    with pytest.raises(ValueError, match="NOPE"):
        symbol_spec_from_hyperliquid_meta(_meta_eth(), "NOPE")


def test_normalize_uses_parsed_spec() -> None:
    from app.exchange.hyperliquid_precision import normalize_order_for_symbol

    s = symbol_spec_from_hyperliquid_meta(_meta_eth(), "ETH")
    out = normalize_order_for_symbol(s, 3000.015, 0.01)
    assert out is not None
    px, sz = out
    # On 0.01 grid 3000.02 has six significant figures; HL rule rounds to 3000.
    assert abs(px - 3000.0) < 1e-9
    assert abs(sz - 0.01) < 1e-12
