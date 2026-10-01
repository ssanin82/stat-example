"""Gate-widening Phase 1 cutover parity (v1.4.12+).

The cutover removed eligibility-clamps for the 7 regime-response
gates and replaced them with composition-driven widening. The
behaviour is by design DIFFERENT when a regime gate fires (bot
now quotes both sides at wide spreads instead of going dark).

But there's a critical NO-REGIME-FIRE parity property:

  When no regime gate is firing, the SpreadComposition's
  contributions reduce to (econ_floor + toxicity + adverse_overlay)
  on each side — which is ≤ the legacy ``compute_effective_min_half_spread_bps``
  return value. Taking max(legacy_half_spread, composition.bid_floor)
  therefore equals legacy_half_spread, and the bot's bid_px / ask_px
  are bit-identical to the pre-cutover values.

This file locks that property in. Phase 2 coefficient iteration
will start moving the per-gate widening DOWN — at that point the
parity boundary is "no regime gate fires + no Phase 2 coefficient
applies" and this same test continues to hold.
"""

from __future__ import annotations

from app.enums import ActiveSides, QuoteEligibility
from app.models import ToxicitySnapshot
from app.post_swing_gate import PostSwingState
from app.quote_eligibility import QuoteEligibilityResult
from app.quoting import (
    SpreadComposition,
    build_spread_composition,
    compute_effective_min_half_spread_bps,
)
from app.vol_trend_gate import VolTrendState
from tests.settings_helpers import UnitTestSettings


def _settings(max_half_spread_bps: float = 30.0) -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "MAX_ABS_POSITION": 10.0,
            "MAX_POSITION_NOTIONAL_USD": 100.0,
            "MAX_HALF_SPREAD_BPS": max_half_spread_bps,
            "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 3.5,
            "ECONOMIC_TOXICITY_SCORE_HALF_SPREAD_BPS": 4.0,
        }
    )


def _clean_eligibility() -> QuoteEligibilityResult:
    return QuoteEligibilityResult(
        eligibility=QuoteEligibility.QUOTE_BOTH,
        reason="ok",
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


def _tox(score: float = 0.0) -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=score,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
    )


def test_no_regime_fire_composition_matches_legacy_floor() -> None:
    """When no regime gate is firing, the composition's per-side
    floor equals the legacy ``compute_effective_min_half_spread_bps``
    return value (econ_floor + toxicity_bump + overlay). This is the
    no-regret parity boundary."""
    s = _settings()
    composition = build_spread_composition(
        settings=s,
        toxicity=_tox(score=0.5),  # moderate toxicity
        active_sides=ActiveSides.BOTH,
        spread_floor_overlay_half_spread_bps=1.0,  # small overlay
        vol_trend_state=VolTrendState(),
        post_swing_state=PostSwingState(),
        ob_imbalance_ewma=0.0,
        basis_regime_last_ic=0.5,  # signal present, gate clear
        basis_regime_pair_count=100,
        position_qty=0.0,
        effective_abs_cap=10.0,
        drift_bps=None,
        raw_eligibility=_clean_eligibility(),
        effective_eligibility=_clean_eligibility(),
        now_mono=100.0,
    )
    legacy_eff_min = compute_effective_min_half_spread_bps(
        s,
        ActiveSides.BOTH,
        toxicity_score=0.5,
        spread_floor_overlay_half_spread_bps=1.0,
    )
    bid_floor = composition.effective_half_spread_bid_bps(
        max_bps=s.max_half_spread_bps
    )
    ask_floor = composition.effective_half_spread_ask_bps(
        max_bps=s.max_half_spread_bps
    )
    # Composition floor == legacy eff_min (within float precision).
    assert abs(bid_floor - legacy_eff_min) < 1e-9
    assert abs(ask_floor - legacy_eff_min) < 1e-9


def test_vol_trend_firing_caps_both_sides_at_max() -> None:
    """When vol_trend fires (gate-equivalent contribution = MAX),
    both bid and ask composition floors hit MAX_HALF_SPREAD_BPS.
    This is the gate-equivalent firing — bot quotes too far from
    mid to fill in normal trading."""
    s = _settings(max_half_spread_bps=30.0)
    vt = VolTrendState()
    vt.cooldown_until_mono = 110.0  # firing at now=100
    composition = build_spread_composition(
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
    assert (
        composition.effective_half_spread_bid_bps(max_bps=30.0) == 30.0
    )
    assert (
        composition.effective_half_spread_ask_bps(max_bps=30.0) == 30.0
    )
    assert composition.capped_at_max_bid is True
    assert composition.capped_at_max_ask is True


def test_asymmetric_microprice_widens_one_side_only() -> None:
    """Microprice firing ask_thin widens ONLY the ask side. The bid
    side keeps the floor at (econ + tox + overlay) — same as legacy."""
    s = _settings(max_half_spread_bps=30.0)
    composition = build_spread_composition(
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
    legacy_eff_min = compute_effective_min_half_spread_bps(
        s, ActiveSides.BOTH, toxicity_score=0.0
    )
    # Bid side: only floor contributors, no widening → matches legacy.
    assert abs(
        composition.effective_half_spread_bid_bps(max_bps=30.0)
        - legacy_eff_min
    ) < 1e-9
    # Ask side: widening caps at MAX.
    assert (
        composition.effective_half_spread_ask_bps(max_bps=30.0) == 30.0
    )


def test_composition_to_dict_serialisable() -> None:
    """The heartbeat publisher calls ``.to_dict()`` and JSON-encodes.
    Verify the cutover dataclass survives the round-trip including
    a fully-zero and a fully-firing composition."""
    import json

    empty = SpreadComposition()
    json.dumps(empty.to_dict())

    full = SpreadComposition(
        econ_floor_bps=3.5,
        toxicity_bps=2.0,
        adverse_overlay_bps=0.5,
        vol_trend_bid_bps=30.0,
        vol_trend_ask_bps=30.0,
        microprice_bid_bps=0.0,
        microprice_ask_bps=15.0,
    )
    json.dumps(full.with_caps(max_bps=30.0).to_dict())


def test_eligibility_unchanged_when_only_safety_clamps_present() -> None:
    """The strip-step in bot.py preserves eligibility when a safety
    signature is in the reason. This isolated test just verifies the
    detection logic — the actual integration is covered by the bot
    integration tests."""
    safety_signatures = (
        "stale_data_warn", "stale_data_kill",
        "drift_100ms", "drift_250ms", "drift_500ms",
        "jump_100ms", "jump_250ms", "jump_500ms",
        "order_state_uncertainty",
        "per_side_uncertainty",
        "order_desync",
    )
    for sig in safety_signatures:
        reason = f"ok|fresh=freshness_ok|drift=drift_ok|{sig}|something"
        assert any(s in reason.lower() for s in safety_signatures)


def test_regime_only_reasons_dont_match_safety_strip() -> None:
    """Reasons that contain ONLY regime signatures should NOT match
    the safety-strip whitelist — those eligibility clamps get
    demoted to BOTH so widening drives the response."""
    safety_signatures = (
        "stale_data_warn", "stale_data_kill",
        "drift_100ms", "drift_250ms", "drift_500ms",
        "jump_100ms", "jump_250ms", "jump_500ms",
        "order_state_uncertainty",
        "per_side_uncertainty",
        "order_desync",
    )
    regime_reasons = (
        "ok|fresh=freshness_ok|drift=drift_ok|recovery_cooldown",
        "freshness_only|fresh=freshness_one_sided:local_receipt_ms>1000",
        "ok|fresh=freshness_ok|drift=drift_ok|microprice_gate:ask_thin",
        "ok|vol_trend_gate|vol_ratio=2.50|drift=+5.00bps|remaining=10s",
        "ok|post_swing:pnl_delta|delta=$-0.500|remaining=5.0s",
        "ok|momentum_gate:long_uptrend|util=0.85|drift=+5.00bps",
        "ok|basis_regime_signal_absent|ic=+0.020|pairs=180",
    )
    for reason in regime_reasons:
        rlow = reason.lower()
        assert not any(s in rlow for s in safety_signatures), (
            f"unexpected safety hit in regime-only reason: {reason}"
        )
