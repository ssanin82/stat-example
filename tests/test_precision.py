from app.exchange.hyperliquid_precision import (
    normalize_order_for_symbol,
    normalize_order_pair,
    normalize_price,
    normalize_size,
)
from app.exchange.symbol_spec import SymbolSpec


def test_normalize_price_rounds_to_tick() -> None:
    assert abs(normalize_price(3000.015, 0.01) - 3000.02) < 1e-9


def test_normalize_size_rounds_down_to_step() -> None:
    s = normalize_size(0.01015, 0.0001)
    assert abs(s - 0.0101) < 1e-12


def test_normalize_size_exact_lot_multiple_not_truncated_by_float_noise() -> None:
    """45 * 0.0001 is often 0.0044999... in binary float; must still be 45 lots."""
    step = 0.0001
    raw = 45 * step
    s = normalize_size(raw, step)
    assert abs(s - 0.0045) < 1e-15


def test_normalize_order_rejects_below_min_size() -> None:
    sp = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.01,
        min_notional_usd=1.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    assert normalize_order_for_symbol(sp, 100.0, 0.005) is None
    pair, reason = normalize_order_pair(sp, 100.0, 0.005)
    assert pair is None and reason is not None and "min_size" in reason


def test_normalize_order_rejects_below_min_notional() -> None:
    sp = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=100.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    assert normalize_order_for_symbol(sp, 10.0, 0.01) is None


def test_normalize_order_rejects_nonfinite_inputs() -> None:
    sp = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=1.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    assert normalize_order_for_symbol(sp, float("nan"), 0.01) is None
    assert normalize_order_for_symbol(sp, 100.0, float("nan")) is None


def test_normalize_order_accepts_valid() -> None:
    sp = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=1.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    out = normalize_order_for_symbol(sp, 3000.015, 0.01)
    assert out is not None
    px, sz = out
    assert px > 0 and sz > 0
    assert px * sz >= 1.0 - 1e-9
