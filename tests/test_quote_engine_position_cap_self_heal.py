"""_build_side min-notional self-heal must respect the position cap.

Regression for ``tmp/snap_20260418_140635`` — the 1-hour ETH run where
``MAX_ABS_POSITION=0.05`` was silently exceeded by 20% (position reached
−0.060) because:

  1. ``_clip_entry_sizes`` correctly reduced ask_sz to 0.005 at pos=−0.045.
  2. ``_build_side`` received candidate_sz=0.005, computed notional
     0.005 × $2349 = $11.75, below MIN_QUOTE_NOTIONAL_USD=$22.
  3. Self-heal bumped size to 0.010 to clear the min-notional floor.
  4. Self-heal did NOT check the position cap → 0.010 placed,
     filled fully → pos went to −0.055 (over cap).

Fix: pass the clipped size as ``max_allowed_size`` into _build_side. Self-heal
rejects if ``needed_sz > max_allowed_size`` — the correct response is to skip
the placement, not to silently exceed ``MAX_ABS_POSITION``.

The invariant: for any ``_build_side`` call with a non-None ``max_allowed_size``,
the returned FinalQuoteOrder (when not None) MUST have ``size <=
max_allowed_size``. No exceptions.
"""

from __future__ import annotations

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
        "QUOTE_NOTIONAL_USD": 40.0,
        "MAX_ORDER_NOTIONAL_USD": 75.0,
        "MIN_QUOTE_NOTIONAL_USD": 22.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _build_ask(
    *,
    candidate_sz: float,
    max_allowed_size: float | None,
    px: float = 2349.23,
    settings_overrides: dict | None = None,
):
    settings = _settings(**(settings_overrides or {}))
    spec = _grvt_eth_spec()
    eng = QuoteEngine(settings, spec)
    return eng._build_side(
        side=Side.SELL,
        want_side=True,
        candidate_px=px,
        candidate_sz=candidate_sz,
        best_bid=px - 0.01,
        best_ask=px,
        tick=0.01,
        min_half_spread_px=None,
        mid_ref=px - 0.005,
        max_allowed_size=max_allowed_size,
    )


# ------------------- The specific bug regression ------------------------


def test_regression_140635_self_heal_must_not_exceed_position_cap() -> None:
    """Direct repro: clip reduced size to 0.005 at pos near cap, min-notional
    self-heal would want 0.010, but that would exceed the cap. Must reject."""
    order, reason, _npx, _nsz = _build_ask(
        candidate_sz=0.005,          # clipped by _clip_entry_sizes at pos=-0.045
        max_allowed_size=0.005,      # cap says we can sell at most 0.005 more
    )
    assert order is None
    assert reason is not None
    assert "position_cap_forbids_min_notional" in reason


def test_self_heal_fires_normally_when_cap_has_room() -> None:
    """When max_allowed_size is generous, self-heal proceeds as before:
    0.005 (below $22 notional) → bumped to 0.010 → accepted."""
    order, reason, _npx, nsz = _build_ask(
        candidate_sz=0.005,
        max_allowed_size=0.020,      # plenty of cap headroom
    )
    assert order is not None
    assert reason == "ok"
    assert nsz == pytest.approx(0.010, abs=1e-9)  # self-heal bumped to 10 from $22/$2349


def test_cap_equals_needed_size_accepts() -> None:
    """Edge case: max_allowed_size exactly matches needed_sz. Should accept."""
    order, reason, _npx, nsz = _build_ask(
        candidate_sz=0.005,
        max_allowed_size=0.010,      # exactly the self-heal target
    )
    assert order is not None, f"unexpected reject: {reason}"
    assert reason == "ok"
    assert nsz == pytest.approx(0.010, abs=1e-9)


def test_cap_one_step_below_needed_rejects() -> None:
    """Cap is one size_step below what self-heal requires → reject."""
    order, reason, _npx, _nsz = _build_ask(
        candidate_sz=0.005,
        max_allowed_size=0.009,       # one step shy of the 0.010 needed
    )
    assert order is None
    assert reason is not None
    assert "position_cap_forbids_min_notional" in reason


def test_max_allowed_none_preserves_legacy_behaviour() -> None:
    """``max_allowed_size=None`` → no cap enforcement (back-compat)."""
    order, reason, _npx, nsz = _build_ask(
        candidate_sz=0.005,
        max_allowed_size=None,
    )
    # Self-heal runs as before, output size 0.010.
    assert order is not None
    assert reason == "ok"
    assert nsz == pytest.approx(0.010, abs=1e-9)


def test_healthy_size_passes_with_cap_match() -> None:
    """When intent already clears all floors and equals cap, pass through."""
    order, reason, _npx, nsz = _build_ask(
        candidate_sz=0.015,           # ~$35 notional — above all floors
        max_allowed_size=0.015,
    )
    assert order is not None
    assert reason == "ok"
    assert nsz == pytest.approx(0.015, abs=1e-9)


def test_oversize_candidate_rejected_by_cap_guard() -> None:
    """Even if caller didn't clip properly and passes candidate > cap, the
    defensive cap guard at the end of _build_side catches it."""
    # Create the situation by passing max_allowed_size < candidate_sz.
    # This path is exercised by the second cap guard (after self-heal).
    order, reason, _npx, _nsz = _build_ask(
        candidate_sz=0.015,          # above cap
        max_allowed_size=0.008,      # cap allows only 0.008
    )
    assert order is None
    assert reason is not None
    # Either the self-heal rejects (if notional < min) or the final guard rejects.
    assert (
        "above_position_cap_allowed_size" in reason
        or "position_cap_forbids_min_notional" in reason
    )


# ------------------- Confirm compatibility with other guards ------------------------


def test_cap_and_max_order_notional_both_apply() -> None:
    """Self-heal respects BOTH max_allowed_size (position cap) AND
    max_order_notional_usd (risk cap). Whichever is tighter wins."""
    # Scenario: min_notional=$22 requires 0.010 but max_order_notional_usd=15
    # would only allow 0.006. In this case max_order_notional is tighter.
    order, reason, _npx, _nsz = _build_ask(
        candidate_sz=0.005,
        max_allowed_size=0.020,
        settings_overrides={"MAX_ORDER_NOTIONAL_USD": 15.0},
    )
    assert order is None
    assert reason is not None
    assert "min_notional_exceeds_max_order_notional" in reason


def test_intent_too_small_guard_still_fires() -> None:
    """The pre-existing 4×-intent guard still applies when intent is tiny."""
    order, reason, _npx, _nsz = _build_ask(
        candidate_sz=0.001,           # tiny intent, ~$2.35 notional
        max_allowed_size=0.020,
    )
    # needed_sz to clear $22 = 0.010. That's 10× the 0.001 intent — fails the 4× guard.
    assert order is None
    assert reason is not None
    assert "intent_too_small" in reason
