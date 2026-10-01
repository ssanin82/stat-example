"""QuoteEngine self-heals size when rounding pushes notional below venue minimum.

Context: ``tmp/snap_20260417_195747`` — bot stopped placing orders because:

  QUOTE_NOTIONAL_USD=25, toxicity_score=0.35, size_mult=0.825
  bid_sz_raw = 25 * 0.825 / 2426.73 = 0.00850
  size_step = 0.001 → floor(0.00850) = 0.008
  notional  = 0.008 * 2426.73 = 19.41 USD
  venue min = 20 USD  → BELOW — silently rejected

The rule (see ``app/quote_engine.py`` module docstring): the quote engine is a
single model; it must not reject for a reason it could repair. When size_step
rounding puts notional below the venue minimum, the engine bumps size to the
smallest step-multiple that clears the minimum — capped at
``max_order_notional_usd``. Only when the cap conflicts with the venue minimum
does it legitimately reject.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.enums import Side
from app.exchange.symbol_spec import SymbolSpec
from app.quote_engine import QuoteEngine
from tests.settings_helpers import UnitTestSettings


def _grvt_eth_spec(min_notional_usd: float = 20.0) -> SymbolSpec:
    return SymbolSpec(
        price_tick=0.01,
        size_step=0.001,
        min_size=0.001,
        min_notional_usd=min_notional_usd,
        sz_decimals=9,
        source="grvt_meta",
    )


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "grvt",
        "SYMBOL": "ETH_USDT_Perp",
        "QUOTE_NOTIONAL_USD": 25.0,
        "MAX_ORDER_NOTIONAL_USD": 75.0,
        "MIN_QUOTE_NOTIONAL_USD": 12.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _build_side(
    *,
    settings_overrides: dict | None = None,
    spec_min_notional: float = 20.0,
    candidate_px: float = 2426.73,
    candidate_sz: float = 0.00850,
    side: Side = Side.BUY,
):
    settings = _settings(**(settings_overrides or {}))
    spec = _grvt_eth_spec(min_notional_usd=spec_min_notional)
    eng = QuoteEngine(settings, spec)
    return eng._build_side(
        side=side,
        want_side=True,
        candidate_px=candidate_px,
        candidate_sz=candidate_sz,
        best_bid=2426.73,
        best_ask=2426.74,
        tick=0.01,
        min_half_spread_px=None,
        mid_ref=2426.735,
    )


def test_self_heal_bumps_size_to_clear_venue_minimum() -> None:
    """The exact snap_20260417_195747 scenario: 0.00850 → floor 0.008 below
    20 USD → bump to 0.009 which clears 20 USD."""
    order, reason, npx, nsz = _build_side(candidate_sz=0.00850)
    assert order is not None, f"unexpected rejection: reason={reason}"
    assert reason == "ok"
    # Size bumped from 0.008 (the floor rounding) to 0.009.
    assert nsz == pytest.approx(0.009, abs=1e-9)
    assert npx * nsz >= 20.0


def test_self_heal_respects_max_order_notional_cap() -> None:
    """If the cap is below the venue minimum, reject loudly — don't silently
    exceed the risk limit."""
    order, reason, _npx, _nsz = _build_side(
        settings_overrides={"MAX_ORDER_NOTIONAL_USD": 15.0},
        spec_min_notional=20.0,
    )
    assert order is None
    assert reason is not None
    assert "min_notional_exceeds_max_order_notional" in reason


def test_self_heal_rejects_when_intent_too_small_for_venue() -> None:
    """Guard against config drift: if clearing the venue min requires more
    than 4× the intent size, that's a config issue worth flagging."""
    # Intent size 0.001 (~2.4 USD) vs venue min 20 → would need 10× bump.
    order, reason, _npx, _nsz = _build_side(candidate_sz=0.001)
    assert order is None
    assert reason is not None
    assert "intent_too_small" in reason


def test_above_max_order_notional_is_rejected() -> None:
    """Legit risk-cap enforcement: intent too BIG must still reject."""
    order, reason, _npx, _nsz = _build_side(
        settings_overrides={"MAX_ORDER_NOTIONAL_USD": 75.0},
        candidate_sz=0.050,  # 0.05 * 2426 = 121 USD > 75 cap
    )
    assert order is None
    assert reason is not None
    assert "above_max_order_notional" in reason


def test_healthy_size_passes_through_unchanged() -> None:
    """When the intent already clears all floors, self-heal is a no-op."""
    order, reason, npx, nsz = _build_side(candidate_sz=0.015)
    assert order is not None
    assert reason == "ok"
    assert nsz == pytest.approx(0.015, abs=1e-9)
    assert npx == pytest.approx(2426.73, abs=1e-9)


def test_sell_side_self_heals_too() -> None:
    """Symmetry check: SELL-side self-heal works the same way."""
    order, reason, _npx, nsz = _build_side(
        side=Side.SELL,
        candidate_sz=0.00850,
    )
    assert order is not None
    assert reason == "ok"
    assert nsz == pytest.approx(0.009, abs=1e-9)


def test_self_heal_honours_both_min_quote_notional_and_venue_min() -> None:
    """required_min = max(local, venue). Both act as a single floor."""
    # Venue min 5, local min 30 → required = 30. Intent 0.00850 → 21 USD notional
    # → needs to bump to clear 30.
    order, reason, _npx, nsz = _build_side(
        settings_overrides={"MIN_QUOTE_NOTIONAL_USD": 30.0},
        spec_min_notional=5.0,
        candidate_sz=0.00850,
    )
    # Intent too small vs local 30 (30/2426/0.001 ≈ 0.013 → 1.5x bump)
    if order is not None:
        assert nsz * 2426.73 >= 30.0
    else:
        # If flagged as intent_too_small, that's the other correct outcome
        # (0.013 is > 4x the 0.00850 intent? 0.013/0.00850 = 1.53, no).
        # Should self-heal successfully.
        assert False, f"expected self-heal to succeed, got reject: {reason}"
