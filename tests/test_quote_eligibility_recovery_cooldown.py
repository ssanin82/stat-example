from __future__ import annotations

from dataclasses import replace

from app.enums import QuoteEligibility
from app.quote_eligibility import (
    QuoteEligibilityResult,
    apply_recovery_cooldown,
    maybe_arm_recovery_cooldown,
)
from tests.settings_helpers import UnitTestSettings


def _raw_result(e: QuoteEligibility) -> QuoteEligibilityResult:
    # Only fields used by apply_recovery_cooldown are set meaningfully.
    return QuoteEligibilityResult(
        eligibility=e,
        reason="raw",
        seconds_since_last_public_book_update=0.0,
        effective_staleness_ms=None,
        market_data_gap_p95_ms=None,
        market_data_gap_median_ms=None,
        mid_return_100ms_bps=None,
        mid_return_250ms_bps=None,
        mid_return_500ms_bps=None,
        jump_100ms_bps=None,
        jump_250ms_bps=None,
        jump_500ms_bps=None,
        in_cooldown=False,
        counter_tags=(),
    )


def test_hold_all_to_quote_both_cooldown_expires_without_rearming_forever() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "QUOTE_HOLD_COOLDOWN_MS": 2000,
            "QUOTE_ONE_SIDED_COOLDOWN_MS": 800,
        }
    )
    now = 100.0

    # Episode: last_effective was HOLD_ALL (due to order_state_uncertainty), raw improves to QUOTE_BOTH.
    ru, rf = maybe_arm_recovery_cooldown(
        s,
        last_effective=QuoteEligibility.HOLD_ALL,
        new_raw=QuoteEligibility.QUOTE_BOTH,
        now_mono=now,
        recovery_until_mono=0.0,
        recovery_floor=None,
    )
    assert rf == QuoteEligibility.HOLD_ALL
    assert ru == now + 2.0

    # Next healthy ticks: last_effective remains HOLD_ALL due to clamp, but raw is healthy.
    # The cooldown must NOT extend on every tick.
    now2 = now + 0.5
    ru2, rf2 = maybe_arm_recovery_cooldown(
        s,
        last_effective=QuoteEligibility.HOLD_ALL,
        new_raw=QuoteEligibility.QUOTE_BOTH,
        now_mono=now2,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    assert (ru2, rf2) == (ru, rf)

    # After expiry: apply_recovery_cooldown must release the clamp.
    eff = apply_recovery_cooldown(
        _raw_result(QuoteEligibility.QUOTE_BOTH),
        now_mono=ru + 0.001,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    assert eff.eligibility == QuoteEligibility.QUOTE_BOTH
    assert eff.in_cooldown is False


def test_one_sided_to_both_cooldown_expires_without_rearming_forever() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "QUOTE_HOLD_COOLDOWN_MS": 2000,
            "QUOTE_ONE_SIDED_COOLDOWN_MS": 800,
        }
    )
    now = 10.0

    ru, rf = maybe_arm_recovery_cooldown(
        s,
        last_effective=QuoteEligibility.QUOTE_BUY_ONLY,
        new_raw=QuoteEligibility.QUOTE_BOTH,
        now_mono=now,
        recovery_until_mono=0.0,
        recovery_floor=None,
    )
    assert rf == QuoteEligibility.QUOTE_BUY_ONLY
    assert ru == now + 0.8

    now2 = now + 0.1
    ru2, rf2 = maybe_arm_recovery_cooldown(
        s,
        last_effective=QuoteEligibility.QUOTE_BUY_ONLY,
        new_raw=QuoteEligibility.QUOTE_BOTH,
        now_mono=now2,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    assert (ru2, rf2) == (ru, rf)

    eff = apply_recovery_cooldown(
        _raw_result(QuoteEligibility.QUOTE_BOTH),
        now_mono=ru + 0.001,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    assert eff.eligibility == QuoteEligibility.QUOTE_BOTH


def test_new_restrictive_episode_rearms_after_recovery() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "QUOTE_HOLD_COOLDOWN_MS": 2000,
            "QUOTE_ONE_SIDED_COOLDOWN_MS": 800,
        }
    )
    t0 = 50.0

    # First restrictive episode: HOLD_ALL -> raw improves to BOTH => arm hold cooldown.
    ru, rf = maybe_arm_recovery_cooldown(
        s,
        last_effective=QuoteEligibility.HOLD_ALL,
        new_raw=QuoteEligibility.QUOTE_BOTH,
        now_mono=t0,
        recovery_until_mono=0.0,
        recovery_floor=None,
    )
    assert rf == QuoteEligibility.HOLD_ALL

    # Cooldown expires; effective becomes QUOTE_BOTH.
    eff = apply_recovery_cooldown(
        _raw_result(QuoteEligibility.QUOTE_BOTH),
        now_mono=ru + 0.01,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    assert eff.eligibility == QuoteEligibility.QUOTE_BOTH

    # Later, a NEW restrictive raw episode should clear recovery and then re-arm on improvement.
    t1 = ru + 1.0
    ru_c, rf_c = maybe_arm_recovery_cooldown(
        s,
        last_effective=QuoteEligibility.QUOTE_BOTH,
        new_raw=QuoteEligibility.HOLD_ALL,
        now_mono=t1,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    assert (ru_c, rf_c) == (0.0, None)

    # Improvement again should arm again (fresh episode).
    t2 = t1 + 0.1
    ru2, rf2 = maybe_arm_recovery_cooldown(
        s,
        last_effective=QuoteEligibility.HOLD_ALL,
        new_raw=QuoteEligibility.QUOTE_BOTH,
        now_mono=t2,
        recovery_until_mono=ru_c,
        recovery_floor=rf_c,
    )
    assert rf2 == QuoteEligibility.HOLD_ALL
    assert ru2 == t2 + 2.0


def test_end_to_end_state_machine_recovers_from_order_uncertainty_episode() -> None:
    """
    Model the same loop as Bot.one_tick:
      last_effective + raw -> maybe_arm -> apply_recovery -> last_effective updated.

    This regresses the reported failure mode where last_effective stays clamped and re-arms
    every tick, preventing recovery to QUOTE_BOTH.
    """
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "QUOTE_HOLD_COOLDOWN_MS": 2000,
            "QUOTE_ONE_SIDED_COOLDOWN_MS": 800,
        }
    )
    last_eff = QuoteEligibility.HOLD_ALL  # uncertainty episode
    ru = 0.0
    rf = None

    # Tick 0: uncertainty cleared; raw becomes QUOTE_BOTH => arm cooldown once.
    t = 0.0
    ru, rf = maybe_arm_recovery_cooldown(
        s,
        last_effective=last_eff,
        new_raw=QuoteEligibility.QUOTE_BOTH,
        now_mono=t,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    eff = apply_recovery_cooldown(
        _raw_result(QuoteEligibility.QUOTE_BOTH),
        now_mono=t,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    last_eff = eff.eligibility
    assert last_eff == QuoteEligibility.HOLD_ALL

    # Many healthy ticks while in cooldown: should not extend; should eventually recover.
    for step in range(1, 10):
        t = step * 0.25  # 250ms ticks
        ru2, rf2 = maybe_arm_recovery_cooldown(
            s,
            last_effective=last_eff,
            new_raw=QuoteEligibility.QUOTE_BOTH,
            now_mono=t,
            recovery_until_mono=ru,
            recovery_floor=rf,
        )
        assert (ru2, rf2) == (ru, rf)
        ru, rf = ru2, rf2
        eff = apply_recovery_cooldown(
            _raw_result(QuoteEligibility.QUOTE_BOTH),
            now_mono=t,
            recovery_until_mono=ru,
            recovery_floor=rf,
        )
        last_eff = eff.eligibility

    # After ~2s, effective must recover to QUOTE_BOTH.
    t = 2.01
    eff = apply_recovery_cooldown(
        _raw_result(QuoteEligibility.QUOTE_BOTH),
        now_mono=t,
        recovery_until_mono=ru,
        recovery_floor=rf,
    )
    assert eff.eligibility == QuoteEligibility.QUOTE_BOTH

