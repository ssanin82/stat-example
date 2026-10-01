"""Tests for ``app.session_drawdown_gate`` — tiered session-PnL
drawdown ladder. Targets the chronic-bleed pattern from snapshot
260510081612 where 15 minutes of slow grinding adverse markout
left $0.73 of damage with no single-event gate firing.
"""

from __future__ import annotations

import pytest

from app.session_drawdown_gate import (
    DrawdownTier,
    SessionDrawdownState,
    TierThresholds,
    is_killed,
    is_quote_paused,
    is_widen_active,
    observe_pause_expiry,
    observe_pnl,
    observe_test_resume_fill,
)


def _thresholds(**overrides) -> TierThresholds:
    base = dict(
        tier1_widen_usd=0.50,
        tier2_pause_short_usd=1.50,
        tier3_pause_long_usd=3.00,
        tier4_kill_usd=5.00,
        pause_short_seconds=300.0,
        pause_long_seconds=1800.0,
        test_resume_sample_fills=10,
        test_resume_max_adverse_bps=1.5,
    )
    base.update(overrides)
    return TierThresholds(**base)


# ---------------------------------------------------------------------------
# Tier ladder semantics — observe_pnl
# ---------------------------------------------------------------------------


def test_initial_state_is_clear() -> None:
    s = SessionDrawdownState()
    assert s.tier == DrawdownTier.CLEAR
    assert not is_quote_paused(s)
    assert not is_widen_active(s)
    assert not is_killed(s)


def test_clear_when_pnl_above_tier1() -> None:
    s = SessionDrawdownState()
    transition = observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-0.20, thresholds=_thresholds())
    assert transition is None
    assert s.tier == DrawdownTier.CLEAR


def test_tier1_widen_at_threshold() -> None:
    s = SessionDrawdownState()
    transition = observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-0.50, thresholds=_thresholds())
    assert transition == DrawdownTier.WIDEN
    assert is_widen_active(s)
    assert not is_quote_paused(s)


def test_tier2_pause_short_arms_cooldown() -> None:
    s = SessionDrawdownState()
    th = _thresholds()
    transition = observe_pnl(s, now_mono=100.0, now_iso="t", session_pnl_usd=-1.50, thresholds=th)
    assert transition == DrawdownTier.PAUSE_SHORT
    assert is_quote_paused(s)
    # Cooldown deadline = now + pause_short_seconds.
    assert s.cooldown_until_mono == pytest.approx(100.0 + th.pause_short_seconds)


def test_tier3_pause_long_arms_longer_cooldown() -> None:
    s = SessionDrawdownState()
    th = _thresholds()
    observe_pnl(s, now_mono=100.0, now_iso="t", session_pnl_usd=-3.00, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_LONG
    assert s.cooldown_until_mono == pytest.approx(100.0 + th.pause_long_seconds)


def test_tier4_kill_terminal() -> None:
    """KILLED is terminal — no automatic recovery, even if pnl
    bounces back. Mirrors MAX_DRAWDOWN_USD semantics."""
    s = SessionDrawdownState()
    observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-5.00, thresholds=_thresholds())
    assert s.tier == DrawdownTier.KILLED
    assert is_killed(s)
    # PnL recovers; gate should NOT auto-clear.
    transition = observe_pnl(s, now_mono=10.0, now_iso="t", session_pnl_usd=+0.10, thresholds=_thresholds())
    assert transition is None
    assert s.tier == DrawdownTier.KILLED


def test_escalation_during_pause_allowed() -> None:
    """While paused, a worsening pnl can escalate to a deeper
    tier (don't be stuck in PAUSE_SHORT while pnl plummets to
    tier-3)."""
    s = SessionDrawdownState()
    th = _thresholds()
    observe_pnl(s, now_mono=100.0, now_iso="t", session_pnl_usd=-1.50, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_SHORT
    # Worsen — drop into tier 3.
    observe_pnl(s, now_mono=120.0, now_iso="t", session_pnl_usd=-3.50, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_LONG


def test_no_de_escalation_during_pause() -> None:
    """While paused, a recovering pnl does NOT clear the gate —
    the cooldown must run its course; ``observe_pause_expiry``
    handles the transition out."""
    s = SessionDrawdownState()
    th = _thresholds()
    observe_pnl(s, now_mono=100.0, now_iso="t", session_pnl_usd=-1.50, thresholds=th)
    # PnL recovers — but cooldown still active.
    transition = observe_pnl(s, now_mono=150.0, now_iso="t", session_pnl_usd=+0.10, thresholds=th)
    assert transition is None
    assert s.tier == DrawdownTier.PAUSE_SHORT


# ---------------------------------------------------------------------------
# Pause-expiry → RESUME_TESTING
# ---------------------------------------------------------------------------


def test_pause_expiry_enters_resume_testing() -> None:
    s = SessionDrawdownState()
    th = _thresholds(pause_short_seconds=10.0)  # short for test
    observe_pnl(s, now_mono=100.0, now_iso="t", session_pnl_usd=-1.50, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_SHORT
    # Cooldown still active.
    transition = observe_pause_expiry(s, now_mono=105.0, now_iso="t", thresholds=th)
    assert transition is None
    assert s.tier == DrawdownTier.PAUSE_SHORT
    # Cooldown expired.
    transition = observe_pause_expiry(s, now_mono=111.0, now_iso="t", thresholds=th)
    assert transition == DrawdownTier.RESUME_TESTING
    assert s.tier == DrawdownTier.RESUME_TESTING
    assert s.test_resume_fills_remaining == th.test_resume_sample_fills
    # Widen is latched during RESUME_TESTING.
    assert is_widen_active(s)


# ---------------------------------------------------------------------------
# RESUME_TESTING → clear or escalate
# ---------------------------------------------------------------------------


def _put_in_resume_testing(th: TierThresholds) -> SessionDrawdownState:
    s = SessionDrawdownState()
    observe_pnl(s, now_mono=100.0, now_iso="t", session_pnl_usd=-1.50, thresholds=th)
    s.cooldown_until_mono = 100.0  # immediate expiry
    observe_pause_expiry(s, now_mono=101.0, now_iso="t", thresholds=th)
    return s


def test_resume_testing_clears_when_markout_ok() -> None:
    th = _thresholds(test_resume_sample_fills=3)
    s = _put_in_resume_testing(th)
    # Three non-adverse fills — gate clears.
    for mk in (-0.5, +0.2, -1.0):
        observe_test_resume_fill(
            s,
            fill_markout_5s_bps=mk,
            now_mono=200.0,
            now_iso="t",
            session_pnl_usd=-0.10,  # pnl recovered too
            thresholds=th,
        )
    assert s.tier == DrawdownTier.CLEAR


def test_resume_testing_lands_on_widen_when_pnl_still_below_tier1() -> None:
    """Even if markout clears, if session pnl is still below tier-1
    threshold the gate stays at WIDEN (the spreads stay wider until
    pnl actually recovers)."""
    th = _thresholds(test_resume_sample_fills=3)
    s = _put_in_resume_testing(th)
    for mk in (-0.5, +0.2, -1.0):
        observe_test_resume_fill(
            s,
            fill_markout_5s_bps=mk,
            now_mono=200.0,
            now_iso="t",
            session_pnl_usd=-1.20,  # still below tier-1 (-0.50)
            thresholds=th,
        )
    assert s.tier == DrawdownTier.WIDEN


def test_resume_testing_escalates_on_continued_adverse() -> None:
    """If markout is still adverse after the sample window,
    escalate to PAUSE_LONG (next tier from RESUME_TESTING)."""
    th = _thresholds(test_resume_sample_fills=3)
    s = _put_in_resume_testing(th)
    for mk in (-5.0, -8.0, -3.0):
        observe_test_resume_fill(
            s,
            fill_markout_5s_bps=mk,
            now_mono=200.0,
            now_iso="t",
            session_pnl_usd=-1.50,
            thresholds=th,
        )
    assert s.tier == DrawdownTier.PAUSE_LONG


def test_resume_testing_skips_none_markouts() -> None:
    """Fills with no markout sample (e.g. fills that haven't
    matured to 5s yet) shouldn't consume the sample budget."""
    th = _thresholds(test_resume_sample_fills=2)
    s = _put_in_resume_testing(th)
    observe_test_resume_fill(
        s, fill_markout_5s_bps=None, now_mono=200.0, now_iso="t",
        session_pnl_usd=-0.10, thresholds=th,
    )
    assert s.test_resume_fills_remaining == 2  # unchanged
    observe_test_resume_fill(
        s, fill_markout_5s_bps=-0.5, now_mono=200.0, now_iso="t",
        session_pnl_usd=-0.10, thresholds=th,
    )
    assert s.test_resume_fills_remaining == 1


def test_resume_testing_escalates_immediately_on_severe_pnl_drop() -> None:
    """If during RESUME_TESTING the pnl plummets further, don't
    insist on completing the test — escalate immediately."""
    th = _thresholds()
    s = _put_in_resume_testing(th)
    transition = observe_pnl(
        s, now_mono=200.0, now_iso="t", session_pnl_usd=-3.50, thresholds=th
    )
    assert transition == DrawdownTier.PAUSE_LONG
    assert s.tier == DrawdownTier.PAUSE_LONG


# ---------------------------------------------------------------------------
# Phase 2K.10 (v1.5.182) — continuous-test favorable-exit predicate
# ---------------------------------------------------------------------------


def _favorable_thresholds(**overrides) -> TierThresholds:
    base = dict(
        tier1_widen_usd=0.20,
        tier2_pause_short_usd=0.50,
        tier3_pause_long_usd=0.80,
        tier4_kill_usd=1.00,
        pause_short_seconds=600.0,
        pause_long_seconds=1800.0,
        test_resume_sample_fills=10,
        test_resume_max_adverse_bps=1.5,
        favorable_exit_enabled=True,
        favorable_exit_clear_band_ratio=0.5,
        favorable_exit_dwell_seconds=5.0,
    )
    base.update(overrides)
    return TierThresholds(**base)


def test_favorable_exit_disabled_keeps_pre_2k10_behaviour() -> None:
    """When ``favorable_exit_enabled=False`` (default), de-escalation
    from PAUSE_* mid-cooldown is still forbidden — preserves pre-2K.10
    semantics for callers that haven't migrated."""
    th = _thresholds()  # favorable_exit_enabled defaults False
    s = SessionDrawdownState()
    observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-1.60, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_SHORT
    # Pnl recovers above the clear band ($1.50 × 0.5 = $0.75 magnitude;
    # clear band is at -$0.75. -$0.30 is above that.)
    transition = observe_pnl(
        s, now_mono=10.0, now_iso="t2", session_pnl_usd=-0.30, thresholds=th
    )
    assert transition is None  # no early exit; still paused
    assert s.tier == DrawdownTier.PAUSE_SHORT


def test_favorable_exit_arms_at_first_recovery_tick() -> None:
    """Eligibility timestamp gets set the first tick PnL clears the
    clear band; dwell hasn't expired yet so no transition fires."""
    th = _favorable_thresholds()
    s = SessionDrawdownState()
    observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-0.60, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_SHORT
    assert s.favorable_exit_eligible_since_mono == 0.0
    # Clear band = -$0.50 × 0.5 = -$0.25. -$0.10 is above (better).
    transition = observe_pnl(
        s, now_mono=1.0, now_iso="t2", session_pnl_usd=-0.10, thresholds=th
    )
    assert transition is None
    assert s.tier == DrawdownTier.PAUSE_SHORT
    assert s.favorable_exit_eligible_since_mono == 1.0


def test_favorable_exit_transitions_after_dwell() -> None:
    """After ``dwell_seconds`` of sustained recovery above clear
    band, transition to RESUME_TESTING (skipping the cooldown)."""
    th = _favorable_thresholds()
    s = SessionDrawdownState()
    observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-0.60, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_SHORT

    # Tick 1: PnL clears the clear band. Eligibility starts.
    observe_pnl(s, now_mono=10.0, now_iso="t2", session_pnl_usd=-0.10, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_SHORT
    assert s.favorable_exit_eligible_since_mono == 10.0

    # Tick 2: still above clear band, dwell not yet expired (4s < 5s).
    transition = observe_pnl(
        s, now_mono=14.0, now_iso="t3", session_pnl_usd=-0.10, thresholds=th
    )
    assert transition is None
    assert s.tier == DrawdownTier.PAUSE_SHORT

    # Tick 3: dwell expired (5s reached) — transition.
    transition = observe_pnl(
        s, now_mono=15.0, now_iso="t4", session_pnl_usd=-0.10, thresholds=th
    )
    assert transition == DrawdownTier.RESUME_TESTING
    assert s.tier == DrawdownTier.RESUME_TESTING


def test_favorable_exit_dwell_resets_on_dip() -> None:
    """If PnL drops back below the clear band before dwell expires,
    the eligibility timer resets. Subsequent recovery starts the
    dwell from zero."""
    th = _favorable_thresholds()
    s = SessionDrawdownState()
    observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-0.60, thresholds=th)

    # Recovers at t=10 (eligibility starts)
    observe_pnl(s, now_mono=10.0, now_iso="t2", session_pnl_usd=-0.10, thresholds=th)
    assert s.favorable_exit_eligible_since_mono == 10.0

    # Dips back at t=12 (resets eligibility)
    observe_pnl(s, now_mono=12.0, now_iso="t3", session_pnl_usd=-0.40, thresholds=th)
    assert s.favorable_exit_eligible_since_mono == 0.0
    assert s.tier == DrawdownTier.PAUSE_SHORT  # still paused

    # Recovers again at t=14 (fresh eligibility)
    observe_pnl(s, now_mono=14.0, now_iso="t4", session_pnl_usd=-0.10, thresholds=th)
    assert s.favorable_exit_eligible_since_mono == 14.0

    # Still in dwell at t=18 (4s < 5s)
    transition = observe_pnl(
        s, now_mono=18.0, now_iso="t5", session_pnl_usd=-0.10, thresholds=th
    )
    assert transition is None

    # Dwell expires at t=19 (5s exactly)
    transition = observe_pnl(
        s, now_mono=19.0, now_iso="t6", session_pnl_usd=-0.10, thresholds=th
    )
    assert transition == DrawdownTier.RESUME_TESTING


def test_favorable_exit_eligibility_resets_on_escalation() -> None:
    """If pnl is in eligibility mid-PAUSE_SHORT but then plummets
    enough to escalate to PAUSE_LONG, the eligibility timer
    resets — the PAUSE_LONG dwell starts fresh."""
    th = _favorable_thresholds()
    s = SessionDrawdownState()
    observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-0.60, thresholds=th)
    observe_pnl(s, now_mono=10.0, now_iso="t2", session_pnl_usd=-0.10, thresholds=th)
    assert s.favorable_exit_eligible_since_mono == 10.0

    # Pnl plummets to PAUSE_LONG territory.
    transition = observe_pnl(
        s, now_mono=11.0, now_iso="t3", session_pnl_usd=-0.85, thresholds=th
    )
    assert transition == DrawdownTier.PAUSE_LONG
    assert s.favorable_exit_eligible_since_mono == 0.0


def test_favorable_exit_uses_correct_tier_threshold_for_clear_band() -> None:
    """The clear band is computed from the CURRENT tier's arm
    threshold — PAUSE_LONG's clear band uses Tier 3, not Tier 2.
    """
    th = _favorable_thresholds()
    s = SessionDrawdownState()
    # Push directly to PAUSE_LONG.
    observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-0.90, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_LONG
    # Tier 3 arm = $0.80; clear band = -$0.40. -$0.30 is above.
    # But Tier 2 arm = $0.50; if it mistakenly used Tier 2's clear
    # band (-$0.25), -$0.30 would NOT qualify.
    observe_pnl(s, now_mono=1.0, now_iso="t2", session_pnl_usd=-0.30, thresholds=th)
    assert s.favorable_exit_eligible_since_mono == 1.0
    # Dwell expires.
    transition = observe_pnl(
        s, now_mono=6.5, now_iso="t3", session_pnl_usd=-0.30, thresholds=th
    )
    assert transition == DrawdownTier.RESUME_TESTING


def test_favorable_exit_ceiling_still_binds_when_pnl_stuck() -> None:
    """Flat-bot scenario: pnl is locked below clear band (e.g.
    realized loss with no position). Favorable-exit never fires;
    ``observe_pause_expiry`` at the ceiling does its normal job.
    """
    th = _favorable_thresholds()
    s = SessionDrawdownState()
    observe_pnl(s, now_mono=0.0, now_iso="t", session_pnl_usd=-0.60, thresholds=th)
    assert s.tier == DrawdownTier.PAUSE_SHORT

    # Pnl locked at -$0.60, never crosses clear band (-$0.25). Tick
    # through several minutes; favorable-exit never arms.
    for t in [60.0, 120.0, 300.0, 500.0]:
        transition = observe_pnl(
            s, now_mono=t, now_iso="t", session_pnl_usd=-0.60, thresholds=th
        )
        assert transition is None
        assert s.tier == DrawdownTier.PAUSE_SHORT
        assert s.favorable_exit_eligible_since_mono == 0.0

    # Ceiling fires at the 600s mark (TIER2_PAUSE_SHORT_SECONDS).
    expiry = observe_pause_expiry(
        s, now_mono=601.0, now_iso="t", thresholds=th
    )
    assert expiry == DrawdownTier.RESUME_TESTING
    assert s.tier == DrawdownTier.RESUME_TESTING
