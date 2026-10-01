"""Gate-widening Phase 1 — ``build_spread_composition`` integration smoke.

The v1.4.9 deploy revealed a real bug that the unit tests missed:
``bot.py`` referenced ``ActiveSides`` without importing it, so the
``build_spread_composition`` call wrapped in try/except threw
NameError on every tick and the composition was silently never
captured (5052 swallowed exceptions in a 3-minute session).

The unit tests for SpreadComposition and per-gate widening_bps
passed because they tested the data layer in isolation, not the
caller's wiring. This test exercises the builder with the same
input shape ``bot.py`` uses, to catch the next class-of-bug like
this one without spinning up the full bot loop.
"""

from __future__ import annotations

from app.enums import ActiveSides, QuoteEligibility
from app.models import ToxicitySnapshot
from app.post_swing_gate import PostSwingState
from app.quote_eligibility import QuoteEligibilityResult
from app.quoting import build_spread_composition
from app.vol_trend_gate import VolTrendState
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "MAX_ABS_POSITION": 10.0,
            "MAX_POSITION_NOTIONAL_USD": 100.0,
            "MAX_HALF_SPREAD_BPS": 30.0,
        }
    )


def _clean_eligibility() -> QuoteEligibilityResult:
    return QuoteEligibilityResult(
        eligibility=QuoteEligibility.QUOTE_BOTH,
        reason="ok|fresh=freshness_ok|drift=drift_ok",
        seconds_since_last_public_book_update=0.0,
        effective_staleness_ms=0.0,
        market_data_gap_p95_ms=0.0,
        market_data_gap_median_ms=0.0,
        mid_return_100ms_bps=None,
        mid_return_250ms_bps=None,
        mid_return_500ms_bps=None,
        jump_100ms_bps=None,
        jump_250ms_bps=None,
        jump_500ms_bps=None,
        in_cooldown=False,
    )


def _tox() -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
    )


def test_builder_runs_with_clean_inputs() -> None:
    """The bot's call site exercises this combination of inputs every
    tick. v1.4.9 had a NameError because ``ActiveSides`` wasn't
    imported in bot.py; this test mirrors the call shape so the
    next missing-import / wrong-arg surfaces at test time, not at
    runtime."""
    s = _settings()
    comp = build_spread_composition(
        settings=s,
        toxicity=_tox(),
        active_sides=ActiveSides.BOTH,
        spread_floor_overlay_half_spread_bps=0.0,
        vol_trend_state=VolTrendState(),
        post_swing_state=PostSwingState(),
        ob_imbalance_ewma=0.0,
        basis_regime_last_ic=0.5,  # signal present
        basis_regime_pair_count=100,
        position_qty=0.0,
        effective_abs_cap=10.0,
        drift_bps=None,
        raw_eligibility=_clean_eligibility(),
        effective_eligibility=_clean_eligibility(),
        now_mono=100.0,
    )
    # Clean state — no gates fire. Only econ floor contributes.
    assert comp.total_bid_bps_uncapped() == comp.econ_floor_bps
    assert comp.total_ask_bps_uncapped() == comp.econ_floor_bps
    assert comp.capped_at_max_bid is False
    assert comp.capped_at_max_ask is False


def test_builder_runs_with_vol_trend_firing() -> None:
    """When vol_trend_gate is firing, composition contributes MAX
    on both sides → bid/ask spreads cap at MAX_HALF_SPREAD_BPS."""
    s = _settings()
    vt = VolTrendState()
    vt.cooldown_until_mono = 110.0  # active at now=100
    comp = build_spread_composition(
        settings=s,
        toxicity=_tox(),
        active_sides=ActiveSides.BOTH,
        spread_floor_overlay_half_spread_bps=0.0,
        vol_trend_state=vt,
        post_swing_state=PostSwingState(),
        ob_imbalance_ewma=0.0,
        basis_regime_last_ic=0.5,
        basis_regime_pair_count=100,
        position_qty=0.0,
        effective_abs_cap=10.0,
        drift_bps=None,
        raw_eligibility=_clean_eligibility(),
        effective_eligibility=_clean_eligibility(),
        now_mono=100.0,
    )
    assert comp.vol_trend_bid_bps == 30.0
    assert comp.vol_trend_ask_bps == 30.0
    assert comp.capped_at_max_bid is True
    assert comp.capped_at_max_ask is True


def test_builder_runs_with_microprice_ask_thin() -> None:
    """Asymmetric — microprice fires ask_thin → widens ASK only."""
    s = _settings()
    comp = build_spread_composition(
        settings=s,
        toxicity=_tox(),
        active_sides=ActiveSides.BOTH,
        spread_floor_overlay_half_spread_bps=0.0,
        vol_trend_state=VolTrendState(),
        post_swing_state=PostSwingState(),
        ob_imbalance_ewma=0.8,  # bid-heavy → ask thin → suppress asks
        basis_regime_last_ic=0.5,
        basis_regime_pair_count=100,
        position_qty=0.0,
        effective_abs_cap=10.0,
        drift_bps=None,
        raw_eligibility=_clean_eligibility(),
        effective_eligibility=_clean_eligibility(),
        now_mono=100.0,
    )
    # microprice widens the ask side only.
    assert comp.microprice_bid_bps == 0.0
    assert comp.microprice_ask_bps == 30.0
    assert comp.capped_at_max_bid is False
    assert comp.capped_at_max_ask is True


def test_builder_returns_dict_safely() -> None:
    """Heartbeat publisher calls ``.to_dict()`` on the composition.
    Verify the dict is JSON-serialisable and has all expected keys."""
    import json

    s = _settings()
    comp = build_spread_composition(
        settings=s,
        toxicity=_tox(),
        active_sides=ActiveSides.BOTH,
        spread_floor_overlay_half_spread_bps=0.0,
        vol_trend_state=VolTrendState(),
        post_swing_state=PostSwingState(),
        ob_imbalance_ewma=0.0,
        basis_regime_last_ic=0.5,
        basis_regime_pair_count=100,
        position_qty=0.0,
        effective_abs_cap=10.0,
        drift_bps=None,
        raw_eligibility=_clean_eligibility(),
        effective_eligibility=_clean_eligibility(),
        now_mono=100.0,
    )
    d = comp.to_dict()
    # Round-trip via JSON to confirm serialisability for the
    # heartbeat S3 payload.
    json.dumps(d)
    assert "vol_trend_bid_bps" in d
    assert "capped_at_max_bid" in d
    assert "econ_floor_bps" in d
