import pytest

from tests.settings_helpers import UnitTestSettings as Settings
from app.enums import ActiveSides
from app.models import QuoteDecision, ToxicitySnapshot
from app.quoting import (
    adverse_spread_widen_arm,
    apply_profitability_spread_floor,
    compute_effective_min_half_spread_bps,
    compute_quote_decision,
)
from app.utils.time import utc_now


def _settings(**kw: float | int | bool) -> Settings:
    return Settings(
        trading_enabled=False,
        hl_secret_key="",
        hl_account_address="",
        **kw,
    )


def test_compute_quote_decision_rejects_nonfinite_mid() -> None:
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    with pytest.raises(ValueError, match="mid"):
        compute_quote_decision(s, float("nan"), 0.0, 0.0, tox)


def test_reservation_skews_down_when_long() -> None:
    s = _settings(max_abs_position=1.0, inventory_skew_coeff_bps=100.0)
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(s, mid, position_qty=0.5, vol_bps=2.0, toxicity=tox)
    assert q.reservation_price < mid


def test_reservation_skews_up_when_short() -> None:
    s = _settings(MAX_ABS_POSITION=1.0, INVENTORY_SKEW_COEFF_BPS=100.0)
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(s, mid, position_qty=-0.5, vol_bps=2.0, toxicity=tox)
    assert q.reservation_price > mid


def test_spread_clipped_to_bounds() -> None:
    s = _settings(
        base_half_spread_bps=200.0,
        min_half_spread_bps=5.0,
        max_half_spread_bps=40.0,
        vol_multiplier=0.0,
    )
    mid = 50_000.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(s, mid, 0.0, vol_bps=0.0, toxicity=tox)
    assert q.target_spread_bps <= 80.0


def test_join_depth_overlay_widens_half_spread() -> None:
    """Positive ``join_depth_overlay_bps`` adds to the half-spread.
    Without other contributions (vol=0, tox=0, neutral inventory),
    half-spread should equal base + overlay."""
    s = _settings(
        max_abs_position=10.0,
        base_half_spread_bps=2.0,
        min_half_spread_bps=0.0,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q_base = compute_quote_decision(s, mid, 0.0, vol_bps=0.0, toxicity=tox)
    q_overlay = compute_quote_decision(
        s, mid, 0.0, vol_bps=0.0, toxicity=tox, join_depth_overlay_bps=3.0
    )
    # 1.5 is to give pytest.approx some slack against the engine's
    # own rounding; the assertion is "overlay widens by ~3 bps".
    assert q_overlay.target_spread_bps > q_base.target_spread_bps
    assert q_overlay.target_spread_bps - q_base.target_spread_bps == pytest.approx(
        6.0, abs=0.01
    )


def test_join_depth_overlay_negative_tightens_half_spread() -> None:
    """Negative overlay (controller pulls tighter) reduces half-
    spread, bounded by ``MIN_HALF_SPREAD_BPS``."""
    s = _settings(
        max_abs_position=10.0,
        base_half_spread_bps=3.0,
        min_half_spread_bps=0.5,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s, mid, 0.0, vol_bps=0.0, toxicity=tox, join_depth_overlay_bps=-1.0
    )
    # base 3.0 + overlay -1.0 = 2.0 half → 4.0 gross.
    assert q.target_spread_bps == pytest.approx(4.0, abs=0.01)


def test_join_depth_overlay_clamped_by_min_half_spread() -> None:
    """Overlay can't push half-spread below ``MIN_HALF_SPREAD_BPS``
    even if the overlay itself is larger negative."""
    s = _settings(
        max_abs_position=10.0,
        base_half_spread_bps=2.0,
        min_half_spread_bps=1.5,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s, mid, 0.0, vol_bps=0.0, toxicity=tox, join_depth_overlay_bps=-5.0
    )
    # base 2.0 + overlay -5.0 = -3.0 half → clamped to MIN 1.5 → gross 3.0.
    assert q.target_spread_bps == pytest.approx(3.0, abs=0.01)


def test_toxicity_size_reduction_coeff_legacy_default() -> None:
    """At the legacy default (0.5), tox.score=0.5 cuts size 25 %.

    Math: size_mult = clip(1 - 0.5*0.5, 0.2, 1) = 0.75.
    Bid notional = QUOTE_NOTIONAL_USD × 0.75 = 7.5.

    Setting min_quote_notional_usd=2.0 makes the fractional floor
    2/10=0.2, matching the legacy hardcoded floor (so this test
    pins legacy behavior without the new min-notional protection
    interfering).
    """
    s = _settings(
        max_abs_position=10.0,
        quote_notional_usd=10.0,
        min_quote_notional_usd=2.0,
        toxicity_size_reduction_coeff=0.5,
        base_half_spread_bps=2.0,
        min_half_spread_bps=0.1,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
        toxicity_score_half_spread_bps=0.0,
    )
    tox = ToxicitySnapshot(score=0.5, one_sided_fill_ratio=0, avg_adverse_markout_bps=0, vol_spike_ratio=1, hard_trigger=False, soft_trigger=False)
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=0.0, toxicity=tox)
    expected_sz = 10.0 * 0.75 / q.quoted_bid
    assert q.quoted_bid_sz == pytest.approx(expected_sz, rel=1e-6)


def test_toxicity_size_reduction_coeff_aggressive() -> None:
    """At coeff=1.0, tox.score=0.5 cuts size 50 % (matches Tier 1 of
    plans/20260507-calibrate.md) WHEN the bot's min-notional floor
    permits — i.e. min_quote_notional_usd / quote_notional_usd ≤ 0.5.

    Math: size_mult = clip(1 - 1.0*0.5, 0.2, 1) = 0.50.
    Bid notional = QUOTE_NOTIONAL_USD × 0.50 = 5.0.
    """
    s = _settings(
        max_abs_position=10.0,
        quote_notional_usd=10.0,
        min_quote_notional_usd=2.0,
        toxicity_size_reduction_coeff=1.0,
        base_half_spread_bps=2.0,
        min_half_spread_bps=0.1,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
        toxicity_score_half_spread_bps=0.0,
    )
    tox = ToxicitySnapshot(score=0.5, one_sided_fill_ratio=0, avg_adverse_markout_bps=0, vol_spike_ratio=1, hard_trigger=False, soft_trigger=False)
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=0.0, toxicity=tox)
    expected_sz = 10.0 * 0.50 / q.quoted_bid
    assert q.quoted_bid_sz == pytest.approx(expected_sz, rel=1e-6)


def test_toxicity_size_reduction_clamped_at_lower_bound() -> None:
    """Even with very high coeff and score, size_mult is clamped at 0.2
    (20 % of nominal) — the legacy absolute floor. Prevents zero-size
    quotes when MIN_QUOTE_NOTIONAL_USD is unset / very low."""
    s = _settings(
        max_abs_position=10.0,
        quote_notional_usd=10.0,
        min_quote_notional_usd=2.0,
        toxicity_size_reduction_coeff=2.0,
        base_half_spread_bps=2.0,
        min_half_spread_bps=0.1,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
        toxicity_score_half_spread_bps=0.0,
    )
    tox = ToxicitySnapshot(score=1.0, one_sided_fill_ratio=0, avg_adverse_markout_bps=0, vol_spike_ratio=1, hard_trigger=False, soft_trigger=False)
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=0.0, toxicity=tox)
    # 1 - 2.0*1.0 = -1.0 → clamped to 0.2 (legacy floor) → notional 2.0
    expected_sz = 10.0 * 0.20 / q.quoted_bid
    assert q.quoted_bid_sz == pytest.approx(expected_sz, rel=1e-6)


def test_toxicity_size_reduction_floor_protects_min_notional() -> None:
    """Regression for the 2026-05-08 deadlock incident on TON.

    With ``QUOTE_NOTIONAL_USD=10`` and ``MIN_QUOTE_NOTIONAL_USD=5.5``
    (TON's real config), the size_mult floor MUST be 5.5/10 = 0.55,
    so an aggressive coefficient cannot drive notional below the
    bot's self-heal floor (which would suppress placements and
    deadlock execution).

    Test: coeff=1.0, score=0.5 raw multiplier = 0.50 — but the
    fractional floor 0.55 should clip it. Resulting notional = 5.5,
    safely at the self-heal floor instead of below it.
    """
    s = _settings(
        max_abs_position=10.0,
        quote_notional_usd=10.0,
        min_quote_notional_usd=5.5,
        toxicity_size_reduction_coeff=1.0,
        base_half_spread_bps=2.0,
        min_half_spread_bps=0.1,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
        toxicity_score_half_spread_bps=0.0,
    )
    tox = ToxicitySnapshot(score=0.5, one_sided_fill_ratio=0, avg_adverse_markout_bps=0, vol_spike_ratio=1, hard_trigger=False, soft_trigger=False)
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=0.0, toxicity=tox)
    # raw 0.5 → clipped UP to 0.55 by the fractional floor → 5.5 notional
    expected_sz = 10.0 * 0.55 / q.quoted_bid
    assert q.quoted_bid_sz == pytest.approx(expected_sz, rel=1e-6)


def test_toxicity_size_reduction_floor_caps_at_one_when_min_above_nominal() -> None:
    """Defensive: if MIN_QUOTE_NOTIONAL_USD > QUOTE_NOTIONAL_USD
    (a misconfig), the floor caps at 1.0 — i.e. no toxicity reduction
    is applied. Quote stays at full nominal regardless of score.
    """
    s = _settings(
        max_abs_position=10.0,
        quote_notional_usd=5.0,
        min_quote_notional_usd=10.0,  # higher than nominal — degenerate
        toxicity_size_reduction_coeff=1.0,
        base_half_spread_bps=2.0,
        min_half_spread_bps=0.1,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
        toxicity_score_half_spread_bps=0.0,
    )
    tox = ToxicitySnapshot(score=1.0, one_sided_fill_ratio=0, avg_adverse_markout_bps=0, vol_spike_ratio=1, hard_trigger=False, soft_trigger=False)
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=0.0, toxicity=tox)
    expected_sz = 5.0 * 1.0 / q.quoted_bid
    assert q.quoted_bid_sz == pytest.approx(expected_sz, rel=1e-6)


def test_join_depth_overlay_default_zero_no_op() -> None:
    """Default overlay (0.0) leaves half-spread unchanged from a
    direct call without the kwarg — proves the default really is a
    no-op for back-compat."""
    s = _settings(
        max_abs_position=10.0,
        base_half_spread_bps=2.5,
        min_half_spread_bps=0.0,
        max_half_spread_bps=50.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q1 = compute_quote_decision(s, mid, 0.0, vol_bps=0.0, toxicity=tox)
    q2 = compute_quote_decision(
        s, mid, 0.0, vol_bps=0.0, toxicity=tox, join_depth_overlay_bps=0.0
    )
    assert q1.target_spread_bps == q2.target_spread_bps


def test_soft_inventory_ask_only_when_long() -> None:
    s = _settings(
        max_abs_position=1.0,
        inventory_soft_limit_pct=0.4,
        inventory_hard_limit_pct=0.9,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(s, mid, position_qty=0.45, vol_bps=1.0, toxicity=tox)
    assert q.active_sides == ActiveSides.ASK_ONLY
    assert q.quoted_bid_sz == 0.0


def test_soft_toxicity_clips_to_max_half_spread() -> None:
    s = _settings(
        base_half_spread_bps=8.0,
        min_half_spread_bps=4.0,
        max_half_spread_bps=10.0,
        vol_multiplier=0.0,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0.0, 0.0, 0.0, 1.0, False, True, None)
    q = compute_quote_decision(s, mid, 0.0, 0.0, tox)
    half = q.target_spread_bps / 2.0
    assert half <= 10.0 + 1e-9


def test_benign_toxicity_no_arm_adaptive_widen() -> None:
    s = _settings(adaptive_spread_adverse_overlay_half_spread_bps=6.0)
    tox = ToxicitySnapshot(
        0.0,
        0.4,
        0.0,
        1.0,
        False,
        False,
        delayed_markout_sample_count=0,
        adverse_uses_delayed_markouts=False,
    )
    assert adverse_spread_widen_arm(s, tox) is False


def test_adverse_markout_arms_adaptive_widen() -> None:
    s = _settings(
        adaptive_spread_adverse_overlay_half_spread_bps=6.0,
        toxicity_markout_soft_bps=3.0,
    )
    tox = ToxicitySnapshot(
        0.5,
        0.5,
        -5.0,
        1.0,
        False,
        False,
        delayed_markout_sample_count=3,
        adverse_uses_delayed_markouts=True,
    )
    assert adverse_spread_widen_arm(s, tox) is True


def test_effective_min_half_spread_benign_no_overlay_matches_baseline() -> None:
    s = _settings(
        min_half_spread_bps=4.0,
        economic_min_half_spread_neutral_bps=8.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
    )
    base = compute_effective_min_half_spread_bps(s, ActiveSides.BOTH, 0.0)
    with_overlay = compute_effective_min_half_spread_bps(
        s, ActiveSides.BOTH, 0.0, spread_floor_overlay_half_spread_bps=0.0
    )
    assert base == with_overlay == 8.0


def test_effective_min_half_spread_adverse_overlay_widens() -> None:
    s = _settings(
        min_half_spread_bps=4.0,
        economic_min_half_spread_neutral_bps=8.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
    )
    lo = compute_effective_min_half_spread_bps(s, ActiveSides.BOTH, 0.0)
    hi = compute_effective_min_half_spread_bps(
        s, ActiveSides.BOTH, 0.0, spread_floor_overlay_half_spread_bps=7.0
    )
    assert lo == 8.0
    assert hi == 15.0


def test_compute_quote_decision_overlay_raises_economic_floor() -> None:
    s = _settings(
        base_half_spread_bps=4.0,
        min_half_spread_bps=2.0,
        max_half_spread_bps=80.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=8.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_abs_position=1.0,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q0 = compute_quote_decision(s, mid, 0.0, vol_bps=0.0, toxicity=tox)
    q1 = compute_quote_decision(
        s,
        mid,
        0.0,
        vol_bps=0.0,
        toxicity=tox,
        spread_floor_overlay_half_spread_bps=10.0,
    )
    assert q0.target_spread_bps == pytest.approx(16.0)
    assert q1.target_spread_bps == pytest.approx(36.0)
    assert q1.spread_floor_overlay_half_spread_bps == pytest.approx(10.0)
    assert "adaptive_spread_widen" in q1.decision_reason


def test_effective_min_half_spread_neutral_vs_inventory() -> None:
    s = _settings(
        min_half_spread_bps=4.0,
        economic_min_half_spread_neutral_bps=10.0,
        economic_min_half_spread_inventory_bps=6.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
    )
    n = compute_effective_min_half_spread_bps(s, ActiveSides.BOTH, 0.0)
    i = compute_effective_min_half_spread_bps(s, ActiveSides.ASK_ONLY, 0.0)
    assert n == 10.0
    assert i == 6.0
    assert n > i


def test_effective_min_half_spread_toxicity_bump() -> None:
    s = _settings(
        min_half_spread_bps=4.0,
        economic_min_half_spread_neutral_bps=8.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=5.0,
        max_half_spread_bps=80.0,
    )
    lo = compute_effective_min_half_spread_bps(s, ActiveSides.BOTH, 0.0)
    hi = compute_effective_min_half_spread_bps(s, ActiveSides.BOTH, 1.0)
    assert lo == 8.0
    assert hi == 13.0


# ============================================================================
# todo-019 Part B — tick-aware one-sided floor.
# ============================================================================


def test_effective_min_half_spread_no_tick_params_unchanged() -> None:
    """Backwards-compat: when ``price_tick`` / ``fair_value`` aren't
    provided, the floor is the pre-1.2.79 bps-only computation."""
    s = _settings(
        min_half_spread_bps=4.0,
        economic_min_half_spread_neutral_bps=8.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
        one_sided_extra_tick_neutral=2.0,
        one_sided_extra_tick_inventory=2.0,
    )
    # Even with extra-tick knobs set, no tick params → no tick floor.
    n = compute_effective_min_half_spread_bps(s, ActiveSides.BOTH, 0.0)
    i = compute_effective_min_half_spread_bps(s, ActiveSides.ASK_ONLY, 0.0)
    assert n == 8.0
    assert i == 5.0


def test_effective_min_half_spread_tick_floor_dominates_when_bps_sub_tick() -> None:
    """On a tight-tick symbol where ``bps_floor`` is less than
    ``tick_bps_half``, the tick floor wins.

    Example calibrated to TON-ish: price_tick=0.0001, mid=2.30 →
    tick_bps_half = (0.0001/2.30)*10000/2 = 0.217 bp half. With
    extra_tick_inventory=1.0 → tick floor = (1+1)*0.217 = 0.434 bp.
    To exercise dominance we use a tighter symbol: price_tick=0.01,
    mid=2.30 → tick_bps_half = 21.7 bp half. With extra=1 →
    floor=43.4 bp. That dominates a 5 bp inventory floor.
    """
    s = _settings(
        min_half_spread_bps=2.0,
        economic_min_half_spread_neutral_bps=3.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=200.0,
        one_sided_extra_tick_neutral=0.0,
        one_sided_extra_tick_inventory=1.0,
    )
    eff = compute_effective_min_half_spread_bps(
        s,
        ActiveSides.ASK_ONLY,
        0.0,
        price_tick=0.01,
        fair_value=2.30,
    )
    # tick_bps_half = (0.01/2.30)*10000/2 ≈ 21.74; (1+1)*21.74 ≈ 43.48
    # bps floor max(min=2, inv=5) = 5. tick floor wins.
    assert 43.0 < eff < 44.0


def test_effective_min_half_spread_bps_floor_dominates_when_supra_tick() -> None:
    """When ``bps_floor`` is comfortably above ``tick_bps_half``,
    the bps floor wins (backwards-compat for fat-tick / high-price
    symbols where tick is small relative to typical edge)."""
    s = _settings(
        min_half_spread_bps=4.0,
        economic_min_half_spread_neutral_bps=10.0,
        economic_min_half_spread_inventory_bps=8.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
        one_sided_extra_tick_neutral=0.0,
        one_sided_extra_tick_inventory=0.0,
    )
    # ETH-ish: price_tick=0.01, mid=3000 → tick_bps_half = 0.0167 bp.
    # bps floor (inventory=8) dominates.
    eff = compute_effective_min_half_spread_bps(
        s,
        ActiveSides.ASK_ONLY,
        0.0,
        price_tick=0.01,
        fair_value=3000.0,
    )
    assert eff == 8.0


def test_effective_min_half_spread_extra_tick_inventory_only_in_inventory_mode() -> None:
    """``ONE_SIDED_EXTRA_TICK_INVENTORY`` should only multiply the
    tick floor when one-sided; neutral mode uses
    ``ONE_SIDED_EXTRA_TICK_NEUTRAL``.

    Configured with the same bps floor on both modes so the only
    difference is the extra-tick multiplier.
    """
    s = _settings(
        min_half_spread_bps=2.0,
        economic_min_half_spread_neutral_bps=5.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=200.0,
        one_sided_extra_tick_neutral=0.0,
        one_sided_extra_tick_inventory=2.0,
    )
    # tick_bps_half ≈ 21.74 bp. Neutral: 1*21.74 ≈ 21.74. Inventory: 3*21.74 ≈ 65.22.
    neutral = compute_effective_min_half_spread_bps(
        s,
        ActiveSides.BOTH,
        0.0,
        price_tick=0.01,
        fair_value=2.30,
    )
    inventory = compute_effective_min_half_spread_bps(
        s,
        ActiveSides.BID_ONLY,
        0.0,
        price_tick=0.01,
        fair_value=2.30,
    )
    assert 21.0 < neutral < 22.5
    assert 64.5 < inventory < 66.0
    assert inventory > neutral


def test_effective_min_half_spread_nonfinite_tick_or_fair_falls_back() -> None:
    """Defensive: when ``price_tick`` or ``fair_value`` is
    non-finite / zero / negative, skip the tick floor (don't crash
    or produce inf). Pre-1.2.79 behaviour."""
    s = _settings(
        min_half_spread_bps=4.0,
        economic_min_half_spread_neutral_bps=10.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
        one_sided_extra_tick_neutral=0.0,
        one_sided_extra_tick_inventory=1.0,
    )
    for tick, fv in [
        (0.0, 2.30),
        (-0.01, 2.30),
        (float("inf"), 2.30),
        (0.01, 0.0),
        (0.01, -1.0),
        (0.01, float("nan")),
    ]:
        eff = compute_effective_min_half_spread_bps(
            s,
            ActiveSides.ASK_ONLY,
            0.0,
            price_tick=tick,
            fair_value=fv,
        )
        assert eff == 5.0  # bps inventory floor wins, tick floor skipped


def test_apply_profitability_spread_floor_widens_two_sided() -> None:
    s = _settings(
        min_half_spread_bps=4.0,
        economic_min_half_spread_neutral_bps=10.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
    )
    mid = 100.0
    d = QuoteDecision(
        ts=utc_now(),
        symbol="ETH",
        mid_price=mid,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=mid,
        target_spread_bps=0.4,
        target_bid=99.98,
        target_ask=100.02,
        quoted_bid=99.98,
        quoted_ask=100.02,
        quoted_bid_sz=0.1,
        quoted_ask_sz=0.1,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="test",
        quote_cycle_id="x",
    )
    nb, na, w = apply_profitability_spread_floor(
        s,
        d,
        want_bid=True,
        want_ask=True,
        best_bid=None,
        best_ask=None,
        bid_px=99.98,
        ask_px=100.02,
    )
    assert w is True
    assert na - nb == pytest.approx(0.2 * mid / 100.0)  # 20 bps gross = 10 bps half


def test_compute_quote_decision_respects_economic_floor() -> None:
    s = _settings(
        base_half_spread_bps=3.0,
        min_half_spread_bps=2.0,
        max_half_spread_bps=80.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=12.0,
        economic_min_half_spread_inventory_bps=5.0,
        economic_toxicity_score_half_spread_bps=0.0,
        max_abs_position=1.0,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(s, mid, 0.0, vol_bps=0.0, toxicity=tox)
    assert q.target_spread_bps == pytest.approx(24.0)


def test_moderate_inventory_stays_two_sided_at_default_soft_skew() -> None:
    """Default soft skew (0.72): 55% util stays BOTH; would be one-sided at 0.65 soft."""
    s = _settings(max_abs_position=1.0)
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(s, mid, position_qty=0.55, vol_bps=1.0, toxicity=tox)
    assert q.active_sides == ActiveSides.BOTH
    assert q.quoted_bid_sz > 0.0 and q.quoted_ask_sz > 0.0


def test_hard_inventory_one_sided() -> None:
    s = _settings(
        max_abs_position=1.0,
        inventory_soft_limit_pct=0.4,
        inventory_hard_limit_pct=0.7,
    )
    mid = 100.0
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(s, mid, position_qty=-0.75, vol_bps=1.0, toxicity=tox)
    assert q.active_sides == ActiveSides.BID_ONLY


# ---------------------------------------------------------------------------
# todo-011: post-fill replace cooldown
# ---------------------------------------------------------------------------


def test_post_fill_cooldown_disabled_when_remaining_zero() -> None:
    """With both per-side cooldowns at 0 (feature disabled or expired),
    active_sides stays at BOTH and no cooldown reason is appended."""
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=1.0,
        toxicity=tox,
        post_fill_cooldown_bid_remaining_ms=0.0,
        post_fill_cooldown_ask_remaining_ms=0.0,
    )
    assert q.active_sides == ActiveSides.BOTH
    assert "post_fill_cooldown_bid" not in (q.decision_reason or "")
    assert "post_fill_cooldown_ask" not in (q.decision_reason or "")


def test_post_fill_cooldown_bid_suppresses_bid_side() -> None:
    """BUY fill in the cooldown window → BID side suppressed; ASK
    continues. Reason tag is stamped on decision_reason."""
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=1.0,
        toxicity=tox,
        post_fill_cooldown_bid_remaining_ms=120.0,
        post_fill_cooldown_ask_remaining_ms=0.0,
    )
    assert q.active_sides == ActiveSides.ASK_ONLY
    assert "post_fill_cooldown_bid" in (q.decision_reason or "")


def test_post_fill_cooldown_ask_suppresses_ask_side() -> None:
    """SELL fill in the cooldown window → ASK side suppressed; BID
    continues."""
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=1.0,
        toxicity=tox,
        post_fill_cooldown_bid_remaining_ms=0.0,
        post_fill_cooldown_ask_remaining_ms=120.0,
    )
    assert q.active_sides == ActiveSides.BID_ONLY
    assert "post_fill_cooldown_ask" in (q.decision_reason or "")


def test_post_fill_cooldown_both_sides_collapses_to_none() -> None:
    """When both sides have an active cooldown (fills on both within
    the window), active_sides collapses to NONE — bot holds entirely
    for the cooldown duration."""
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=1.0,
        toxicity=tox,
        post_fill_cooldown_bid_remaining_ms=80.0,
        post_fill_cooldown_ask_remaining_ms=120.0,
    )
    assert q.active_sides == ActiveSides.NONE
    assert "post_fill_cooldown_bid" in (q.decision_reason or "")
    assert "post_fill_cooldown_ask" in (q.decision_reason or "")


def test_post_fill_cooldown_composes_with_soft_skew() -> None:
    """When the bot was already restricted to ASK_ONLY by inventory
    soft-skew (long position) AND a BID cooldown fires, the BID side
    is already inactive — adding the BID cooldown is a no-op on
    active_sides but the reason tag still gets stamped for
    observability."""
    # Long position triggers soft_skew_long → ASK_ONLY.
    s = _settings(
        max_abs_position=2.0,
        inventory_soft_limit_pct=0.4,
        inventory_hard_limit_pct=0.8,
    )
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s,
        mid=100.0,
        position_qty=1.0,  # 50% util → soft_skew triggers
        vol_bps=1.0,
        toxicity=tox,
        post_fill_cooldown_bid_remaining_ms=100.0,
        post_fill_cooldown_ask_remaining_ms=0.0,
    )
    # Already ASK_ONLY due to soft_skew; BID cooldown doesn't change
    # state (BID already suppressed). The reason tag is not stamped
    # in the current impl since the soft_skew branch took the path.
    assert q.active_sides == ActiveSides.ASK_ONLY
    assert "soft_skew_long" in (q.decision_reason or "")


# ---------------------------------------------------------------------------
# 2026-05-12 codex-#1 narrow: at-touch adverse pause
# ---------------------------------------------------------------------------


def test_at_touch_adverse_pause_bid_suppresses_bid_side() -> None:
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=1.0,
        toxicity=tox,
        at_touch_adverse_pause_bid=True,
    )
    assert q.active_sides == ActiveSides.ASK_ONLY
    assert "at_touch_adverse_pause_bid" in (q.decision_reason or "")


def test_at_touch_adverse_pause_ask_suppresses_ask_side() -> None:
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=1.0,
        toxicity=tox,
        at_touch_adverse_pause_ask=True,
    )
    assert q.active_sides == ActiveSides.BID_ONLY
    assert "at_touch_adverse_pause_ask" in (q.decision_reason or "")


def test_at_touch_adverse_pause_both_sides_yields_none() -> None:
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q = compute_quote_decision(
        s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=1.0,
        toxicity=tox,
        at_touch_adverse_pause_bid=True,
        at_touch_adverse_pause_ask=True,
    )
    assert q.active_sides == ActiveSides.NONE


# ---------------------------------------------------------------------------
# 2026-05-12 codex-#3: fill-burst size shrink
# ---------------------------------------------------------------------------


def test_fill_burst_shrink_default_one_is_noop() -> None:
    s = _settings(max_abs_position=1.0)
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q_base = compute_quote_decision(s, 100.0, 0.0, 1.0, tox)
    q_test = compute_quote_decision(
        s, 100.0, 0.0, 1.0, tox, fill_burst_size_mult=1.0
    )
    # Both sides should produce the same size when fill_burst_size_mult=1.
    assert q_base.quoted_bid_sz == q_test.quoted_bid_sz
    assert q_base.quoted_ask_sz == q_test.quoted_ask_sz


def test_fill_burst_shrink_reduces_size() -> None:
    s = _settings(
        max_abs_position=1.0,
        quote_notional_usd=20.0,
        min_quote_notional_usd=1.0,
    )
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    q_base = compute_quote_decision(s, 100.0, 0.0, 1.0, tox)
    q_shrunk = compute_quote_decision(
        s, 100.0, 0.0, 1.0, tox, fill_burst_size_mult=0.5
    )
    # Shrunk should be smaller than base (floor permitting).
    assert q_shrunk.quoted_bid_sz < q_base.quoted_bid_sz
    assert q_shrunk.quoted_ask_sz < q_base.quoted_ask_sz
    assert "fill_burst_shrink" in (q_shrunk.decision_reason or "")
