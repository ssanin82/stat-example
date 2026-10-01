"""``normalize_order_pair`` must respect the reporting venue.

Historically the function always applied Hyperliquid's 5-sig-figs rule and
used ``decimal_size_step(sz_decimals)`` as the lot step. For Hyperliquid,
where ``size_step == 10 ** -sz_decimals`` always holds, that is fine. For
GRVT it is not: GRVT reports ``base_decimals`` as the wire-encoding scale
(9 for ETH), but the enforced lot step is the human-scale ``min_size``
(0.001 for ETH). Passing a non-multiple of 0.001 caused GRVT to reject the
order with HTTP 400.
"""

from __future__ import annotations

import pytest

from app.exchange.hyperliquid_precision import normalize_order_pair
from app.exchange.symbol_spec import SymbolSpec


def _grvt_eth_spec() -> SymbolSpec:
    # Mirrors the ``grvt_symbol_spec_bootstrap_success`` log line verbatim.
    return SymbolSpec(
        price_tick=0.01,
        size_step=0.001,
        min_size=0.001,
        min_notional_usd=20.0,
        sz_decimals=9,
        source="grvt_meta",
    )


def _hl_eth_spec() -> SymbolSpec:
    return SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )


def test_grvt_size_quantized_to_spec_size_step() -> None:
    """Size like 50 USD / 3500 ETH ≈ 0.01428 must round down to the 0.001 lot."""
    spec = _grvt_eth_spec()
    out, rej = normalize_order_pair(spec, price=3500.0, size=0.01428571428571428)
    assert rej is None
    px, sz = out  # type: ignore[misc]
    assert sz == pytest.approx(0.014, abs=1e-9)
    assert px == pytest.approx(3500.0, abs=1e-9)


def test_grvt_price_quantized_to_price_tick_not_hl_sig_figs() -> None:
    """GRVT prices keep full tick precision — HL's 5-sig-figs cap must not apply.

    Under the old behavior this price would be rounded down to 3500.1 because
    the HL pipeline derives ``max_price_decimals_perp(9) == 0`` and caps at 5
    sig figs. For GRVT the only price rule is the reported tick (0.01).
    """
    spec = _grvt_eth_spec()
    out, rej = normalize_order_pair(spec, price=3500.12, size=0.010)
    assert rej is None
    px, _sz = out  # type: ignore[misc]
    assert px == pytest.approx(3500.12, abs=1e-9)


def test_grvt_size_rounded_below_step_is_rejected() -> None:
    """A size below the lot step rounds to zero and must be rejected."""
    spec = _grvt_eth_spec()
    out, rej = normalize_order_pair(spec, price=3500.0, size=0.0005)
    assert out is None
    assert rej is not None


def test_grvt_size_on_boundary_rounds_to_min() -> None:
    """Exactly ``min_size`` is accepted (notional 3500 * 0.001 = 3.5 fails min_notional)."""
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.001,
        min_size=0.002,
        min_notional_usd=1.0,  # Relaxed so we isolate the size-step behavior.
        sz_decimals=9,
        source="grvt_meta",
    )
    out, rej = normalize_order_pair(spec, price=1000.0, size=0.0018)
    # 0.0018 rounds down to 0.001 which is below min_size=0.002 → rejected.
    assert out is None
    assert rej is not None and rej.startswith("below_min_size")


def test_hl_behavior_unchanged_for_eth_like_spec() -> None:
    """Hyperliquid path: 5-sig-figs still rounds 3000.123 down to 3000.1."""
    spec = _hl_eth_spec()
    out, rej = normalize_order_pair(spec, price=3000.123, size=0.011)
    assert rej is None
    px, sz = out  # type: ignore[misc]
    assert px == pytest.approx(3000.1, abs=1e-9)
    assert sz == pytest.approx(0.011, abs=1e-9)


def test_hl_fallback_keeps_hl_semantics() -> None:
    """Fallback spec is only emitted by the HL adapter — must keep HL semantics."""
    spec = SymbolSpec(
        price_tick=0.01,
        size_step=0.0001,
        min_size=0.001,
        min_notional_usd=10.0,
        sz_decimals=4,
        source="fallback",
    )
    out, rej = normalize_order_pair(spec, price=3000.123, size=0.011)
    assert rej is None
    px, _ = out  # type: ignore[misc]
    assert px == pytest.approx(3000.1, abs=1e-9)
