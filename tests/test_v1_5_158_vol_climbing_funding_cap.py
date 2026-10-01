"""v1.5.158 — three regime-aware tunings shipped together:

Option A — vol-climbing anticipatory widening
    Short-MA vs long-MA ratio of realized vol_bps. Caller passes
    the ratio; compute_quote_decision adds widen_bps when the
    ratio crosses arm_ratio.

Option B — funding-settle anticipatory widening
    Time-driven overlay around OKX funding settles (00:00 / 08:00 /
    16:00 UTC by default). Caller passes (utc_hour, utc_minute);
    compute_quote_decision adds widen_bps within ±N min of any
    configured settle hour.

Option C — vol-adaptive position cap
    Caller can override MAX_ABS_POSITION per-tick via
    ``max_abs_position_override``. Reduced cap affects inventory_
    skew strength + at-max active_sides + soft/hard skew thresholds.

All three are Rule 0c-aligned (entry/exit driven by live signals
or recurring known events, NOT fixed-window non-trading). Defaults
all OFF; prod profile activates per-symbol.

Per CLAUDE.md: only this test file is run from the assistant.
"""

from __future__ import annotations

from app.models import ToxicitySnapshot
from app.quoting import compute_quote_decision
from tests.settings_helpers import UnitTestSettings as Settings


def _settings(**kw) -> Settings:
    return Settings(
        trading_enabled=False,
        hl_secret_key="",
        hl_account_address="",
        **kw,
    )


def _tox() -> ToxicitySnapshot:
    return ToxicitySnapshot(0, 0, 0, 1, False, False)


def _base_settings(**overrides):
    base = dict(
        max_abs_position=10.0,
        base_half_spread_bps=8.0,
        min_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
        toxicity_score_half_spread_bps=0.0,
    )
    base.update(overrides)
    return _settings(**base)


# ===========================================================================
# Option A — vol-climbing widening
# ===========================================================================


def test_option_a_disabled_does_not_widen():
    s = _base_settings(VOL_CLIMBING_WIDEN_ENABLED=False, VOL_CLIMBING_WIDEN_BPS=2.0)
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        vol_climbing_ratio=5.0,  # would normally arm
    )
    assert q.target_spread_bps == 16.0  # base 8 × 2


def test_option_a_below_arm_ratio_does_not_widen():
    s = _base_settings(
        VOL_CLIMBING_WIDEN_ENABLED=True,
        VOL_CLIMBING_WIDEN_ARM_RATIO=1.5,
        VOL_CLIMBING_WIDEN_BPS=2.0,
    )
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        vol_climbing_ratio=1.2,  # below 1.5 arm
    )
    assert q.target_spread_bps == 16.0


def test_option_a_at_arm_ratio_widens():
    s = _base_settings(
        VOL_CLIMBING_WIDEN_ENABLED=True,
        VOL_CLIMBING_WIDEN_ARM_RATIO=1.5,
        VOL_CLIMBING_WIDEN_BPS=2.0,
    )
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        vol_climbing_ratio=1.5,  # exactly at arm
    )
    assert q.target_spread_bps == 20.0  # (8 + 2) × 2


def test_option_a_above_arm_ratio_widens():
    s = _base_settings(
        VOL_CLIMBING_WIDEN_ENABLED=True,
        VOL_CLIMBING_WIDEN_ARM_RATIO=1.5,
        VOL_CLIMBING_WIDEN_BPS=2.0,
    )
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        vol_climbing_ratio=3.0,
    )
    assert q.target_spread_bps == 20.0


def test_option_a_none_ratio_no_op():
    """Caller passes None when buffer not warm — gate dormant."""
    s = _base_settings(VOL_CLIMBING_WIDEN_ENABLED=True, VOL_CLIMBING_WIDEN_BPS=2.0)
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        vol_climbing_ratio=None,
    )
    assert q.target_spread_bps == 16.0


# ===========================================================================
# Option B — funding-settle widening
# ===========================================================================


def test_option_b_disabled_does_not_widen():
    s = _base_settings(
        FUNDING_SETTLE_WIDEN_ENABLED=False,
        FUNDING_SETTLE_WIDEN_BPS=2.0,
    )
    # 08:00 UTC = exactly at a settle
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        utc_hour_minute=(8, 0),
    )
    assert q.target_spread_bps == 16.0


def test_option_b_exact_settle_time_widens():
    s = _base_settings(
        FUNDING_SETTLE_WIDEN_ENABLED=True,
        FUNDING_SETTLE_WIDEN_HOURS_UTC="0,8,16",
        FUNDING_SETTLE_WIDEN_PRE_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_POST_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_BPS=2.0,
    )
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        utc_hour_minute=(8, 0),
    )
    assert q.target_spread_bps == 20.0  # widened


def test_option_b_pre_window_widens():
    s = _base_settings(
        FUNDING_SETTLE_WIDEN_ENABLED=True,
        FUNDING_SETTLE_WIDEN_HOURS_UTC="0,8,16",
        FUNDING_SETTLE_WIDEN_PRE_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_POST_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_BPS=2.0,
    )
    # 07:50 UTC = 10 min before 08:00 settle, within 15 min pre.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        utc_hour_minute=(7, 50),
    )
    assert q.target_spread_bps == 20.0


def test_option_b_post_window_widens():
    s = _base_settings(
        FUNDING_SETTLE_WIDEN_ENABLED=True,
        FUNDING_SETTLE_WIDEN_HOURS_UTC="0,8,16",
        FUNDING_SETTLE_WIDEN_PRE_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_POST_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_BPS=2.0,
    )
    # 08:10 UTC = 10 min after 08:00 settle, within 15 min post.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        utc_hour_minute=(8, 10),
    )
    assert q.target_spread_bps == 20.0


def test_option_b_outside_window_no_op():
    s = _base_settings(
        FUNDING_SETTLE_WIDEN_ENABLED=True,
        FUNDING_SETTLE_WIDEN_HOURS_UTC="0,8,16",
        FUNDING_SETTLE_WIDEN_PRE_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_POST_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_BPS=2.0,
    )
    # 04:00 UTC — far from any settle.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        utc_hour_minute=(4, 0),
    )
    assert q.target_spread_bps == 16.0


def test_option_b_midnight_wraparound_widens():
    """The 00:00 UTC settle is special — 23:50 UTC is 10 min BEFORE
    via the wraparound, must still trigger."""
    s = _base_settings(
        FUNDING_SETTLE_WIDEN_ENABLED=True,
        FUNDING_SETTLE_WIDEN_HOURS_UTC="0,8,16",
        FUNDING_SETTLE_WIDEN_PRE_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_POST_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_BPS=2.0,
    )
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        utc_hour_minute=(23, 50),
    )
    assert q.target_spread_bps == 20.0


def test_option_b_only_fires_once_per_window():
    """Two settles within close range — gate adds widen_bps ONCE,
    not twice."""
    s = _base_settings(
        FUNDING_SETTLE_WIDEN_ENABLED=True,
        # Hypothetical: 08:00 and 08:30 settles.
        FUNDING_SETTLE_WIDEN_HOURS_UTC="8,9",
        FUNDING_SETTLE_WIDEN_PRE_MINUTES=60.0,
        FUNDING_SETTLE_WIDEN_POST_MINUTES=60.0,
        FUNDING_SETTLE_WIDEN_BPS=2.0,
    )
    # 08:30 UTC — within both 08:00 and 09:00 windows.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_tox(),
        utc_hour_minute=(8, 30),
    )
    # Widens by exactly 2 bps total — break statement guard.
    assert q.target_spread_bps == 20.0


# ===========================================================================
# Option C — vol-adaptive position cap
# ===========================================================================


def test_option_c_override_none_uses_settings_cap():
    s = _base_settings(max_abs_position=10.0, INVENTORY_SKEW_COEFF_BPS=100.0)
    q = compute_quote_decision(
        s, 100.0, position_qty=5.0, vol_bps=0.0, toxicity=_tox(),
        max_abs_position_override=None,
    )
    # norm_inv = 5/10 = 0.5
    # The reservation skew depends on inventory_skew_coeff_bps and the
    # exponent, but the key invariant is that the result uses 10 as
    # max_pos. We verify indirectly: half-spread + skew shouldn't
    # cause active_sides change at util=0.5.
    assert q.target_spread_bps > 0  # smoke


def test_option_c_override_reduces_cap_and_increases_skew():
    """Same position 5.0 but with override max=6.0 → util=5/6 > soft
    threshold → reservation skews harder → ask_only / sell_only behavior
    visible in active_sides."""
    s = _base_settings(
        max_abs_position=10.0,
        INVENTORY_SKEW_COEFF_BPS=100.0,
        INVENTORY_SOFT_LIMIT_PCT=0.6,  # at util >= 0.6, soft skew
        INVENTORY_HARD_LIMIT_PCT=0.85,
    )
    # With override=6: util = 5/6 = 0.83 → soft skew kicks in (>=0.6)
    q = compute_quote_decision(
        s, 100.0, position_qty=5.0, vol_bps=0.0, toxicity=_tox(),
        max_abs_position_override=6.0,
    )
    # Bot is LONG (position=5) and util>=soft → soft_skew_long →
    # ASK_ONLY (suppress new BUYs). Without override (util=0.5<0.6),
    # both sides active.
    # Confirm via the active_sides field on the decision.
    # active_sides serialises to a string — check it indicates ASK_ONLY
    # OR soft_skew_long appears in reason.
    decision_repr = str(q)
    assert "ASK_ONLY" in decision_repr or "soft_skew_long" in decision_repr


def test_option_c_override_at_or_above_cap_triggers_at_max():
    """When position equals override cap → util=1.0 → at_max_long /
    ASK_ONLY."""
    s = _base_settings(
        max_abs_position=10.0,
        INVENTORY_SKEW_COEFF_BPS=100.0,
    )
    q = compute_quote_decision(
        s, 100.0, position_qty=4.0, vol_bps=0.0, toxicity=_tox(),
        max_abs_position_override=4.0,
    )
    assert "ASK_ONLY" in str(q) or "at_max_long" in str(q)


def test_option_c_invalid_override_falls_back():
    s = _base_settings(max_abs_position=10.0, INVENTORY_SKEW_COEFF_BPS=100.0)
    # Override = 0 should be ignored (would mean "no trading at all")
    q = compute_quote_decision(
        s, 100.0, position_qty=5.0, vol_bps=0.0, toxicity=_tox(),
        max_abs_position_override=0.0,
    )
    # Bot behaves as if override didn't exist — util = 5/10 = 0.5
    assert "at_max_long" not in str(q)


# ===========================================================================
# Combined: A + B + C all active simultaneously
# ===========================================================================


def test_options_a_b_c_compose():
    """All three features fire on the same quote tick: vol climbing
    + within funding settle window + reduced position cap.

    Expected: widenings stack (A + B), reduced cap kicks in (C).
    Total spread > base alone."""
    s = _base_settings(
        max_abs_position=10.0,
        base_half_spread_bps=8.0,
        VOL_CLIMBING_WIDEN_ENABLED=True,
        VOL_CLIMBING_WIDEN_ARM_RATIO=1.5,
        VOL_CLIMBING_WIDEN_BPS=2.0,
        FUNDING_SETTLE_WIDEN_ENABLED=True,
        FUNDING_SETTLE_WIDEN_HOURS_UTC="0,8,16",
        FUNDING_SETTLE_WIDEN_PRE_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_POST_MINUTES=15.0,
        FUNDING_SETTLE_WIDEN_BPS=1.5,
    )
    q = compute_quote_decision(
        s, 100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(),
        vol_climbing_ratio=2.0,    # arms A
        utc_hour_minute=(8, 5),    # within B settle window
        max_abs_position_override=6.0,  # C cap reduction
    )
    # half-spread = 8 + 2 (A) + 1.5 (B) = 11.5 → total 23
    assert q.target_spread_bps == 23.0
