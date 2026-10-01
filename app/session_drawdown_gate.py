"""Tiered session-PnL drawdown gate (analysis-day 2026-05-10).

Targets the chronic-bleed pattern observed in snapshot
260510081612: 15 minutes into a fresh session, $0.73 drawdown,
markout asymmetric and adverse on SELL side, but NONE of the
existing gates fire. Toxicity score 0.04 (calm), position
utilisation 27% (below momentum-gate floor), no fast PnL spike
(post-swing miss), no catastrophic markout burst (toxicity_hard
miss). The bot continues quoting and bleeds.

Why a tiered escalation rather than a single threshold?

* A single hard threshold (= ``MAX_DRAWDOWN_USD`` kill) is too
  blunt — kills the whole session even when the regime would
  clear in 30 min.
* A single fixed-duration pause is naive — if the regime hasn't
  changed when the pause expires, the bot resumes and bleeds again
  in a periodic loop ("lose $1.50, pause, lose $1.50, pause").
* A tiered ladder escalates: warning → short pause → long pause →
  kill. Each tier ratchets the response, and a ``test_resume``
  sample at the end of each pause checks whether the regime
  actually cleared before scaling back up. A regime that won't
  clear pushes the bot toward the kill threshold instead of looping.

Tier ladder (defaults; tunable via ``SESSION_DRAWDOWN_TIER_*``):

  Tier 1 (-$0.50): widen spreads.
    Action: arm the existing adaptive_spread_widen overlay with
    a session-PnL trigger reason. Quoting continues with wider
    spreads. No pause. Clears automatically when session PnL
    recovers above the threshold.
  Tier 2 (-$1.50): soft-flatten + 5 min pause.
    Action: trip the SF worker (close inventory) and gate quoting
    HOLD_ALL for 5 minutes, then enter test_resume.
  Tier 3 (-$3.00): soft-flatten + 30 min pause.
    Action: same as tier 2 but with a longer pause.
  Tier 4 (-$5.00): hard kill.
    Action: stop. Operator restart required. Mirrors
    ``MAX_DRAWDOWN_USD``.

Test-resume protocol (after a tier-2 / tier-3 pause expires):

  1. Set state to ``RESUME_TESTING``. The quote engine quotes
     at the wider tier-3 spread floor (tier-1 widen still latched).
  2. Sample the next ``test_resume_sample_fills`` fills.
  3. If their mean markout is non-adverse (≥ ``-test_resume_max_adverse_bps``),
     scale back to normal — clear the gate.
  4. If still adverse, escalate to the next tier.

State is in-memory (``BotState.session_drawdown``). Single-threaded
ownership: the bot's main quote loop is the only writer; live_stats
publisher reads for the dashboard's Market tab.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DrawdownTier(str, Enum):
    """Five-state ladder. ``CLEAR`` is the normal/healthy state;
    each subsequent tier is a more severe response."""

    CLEAR = "CLEAR"
    WIDEN = "WIDEN"  # tier 1: spread floor widened
    PAUSE_SHORT = "PAUSE_SHORT"  # tier 2: 5-min pause
    PAUSE_LONG = "PAUSE_LONG"  # tier 3: 30-min pause
    RESUME_TESTING = "RESUME_TESTING"  # post-pause sample window
    KILLED = "KILLED"  # tier 4: hard kill


@dataclass
class SessionDrawdownState:
    """In-memory ladder state. Single-threaded ownership in the
    bot's main loop; live_stats publisher reads but doesn't
    mutate."""

    tier: DrawdownTier = DrawdownTier.CLEAR

    # 2K.10 (v1.5.182) — continuous-test favorable-exit tracking.
    # Monotonic timestamp at which session_pnl first crossed BACK
    # above the clear band during the current PAUSE_*. 0.0 when not
    # currently eligible. Paired with
    # ``favorable_exit_dwell_seconds`` to require sustained recovery
    # before transitioning out of PAUSE_*.
    favorable_exit_eligible_since_mono: float = 0.0

    cooldown_until_mono: float = 0.0
    """Monotonic deadline for the current pause tier. Zero when
    not paused."""

    test_resume_fills_remaining: int = 0
    """Countdown of fills the test_resume window must sample
    before deciding to escalate or clear. Set when entering
    ``RESUME_TESTING``."""

    test_resume_markouts: deque = field(default_factory=deque)
    """Markouts of fills observed during the test_resume window.
    Bounded by ``test_resume_sample_fills``."""

    last_trigger_pnl_usd: float = 0.0
    """Session PnL value when the most recent tier transition fired.
    Diagnostic only — surfaces in live_stats so the operator can
    correlate the gate's fires against the equity curve."""

    last_transition_iso: Optional[str] = None
    """ISO timestamp of the most recent tier transition. None when
    the gate has never moved off ``CLEAR``."""

    fire_count: int = 0
    """Cumulative count of tier transitions this session
    (excluding initial CLEAR). Useful for spotting an
    over-firing gate during postmortem."""


@dataclass(frozen=True)
class TierThresholds:
    """Configurable thresholds. Defaults match the SUI profile;
    other profiles can override via env vars (see
    ``app.config.SessionDrawdownSettings``)."""

    tier1_widen_usd: float
    tier2_pause_short_usd: float
    tier3_pause_long_usd: float
    tier4_kill_usd: float
    pause_short_seconds: float
    pause_long_seconds: float
    test_resume_sample_fills: int
    test_resume_max_adverse_bps: float
    # 2K.10 (v1.5.182) — favorable-exit predicate config. Defaults
    # preserve pre-2K.10 behaviour (no early exit; pause expires at
    # the ceiling, then RESUME_TESTING) so callers that haven't been
    # migrated work unchanged.
    favorable_exit_enabled: bool = False
    favorable_exit_clear_band_ratio: float = 0.5
    favorable_exit_dwell_seconds: float = 5.0


def _tier_for(pnl_usd: float, t: TierThresholds) -> DrawdownTier:
    """Map a session-PnL value to the deepest tier whose threshold
    it has crossed. Note: PnL is signed; thresholds are negative
    magnitudes (e.g. ``tier1_widen_usd = 0.50`` means "trip when
    pnl ≤ -$0.50").
    """
    if pnl_usd <= -float(t.tier4_kill_usd):
        return DrawdownTier.KILLED
    if pnl_usd <= -float(t.tier3_pause_long_usd):
        return DrawdownTier.PAUSE_LONG
    if pnl_usd <= -float(t.tier2_pause_short_usd):
        return DrawdownTier.PAUSE_SHORT
    if pnl_usd <= -float(t.tier1_widen_usd):
        return DrawdownTier.WIDEN
    return DrawdownTier.CLEAR


def observe_pnl(
    state: SessionDrawdownState,
    *,
    now_mono: float,
    now_iso: str,
    session_pnl_usd: float,
    thresholds: TierThresholds,
) -> Optional[DrawdownTier]:
    """Apply the tier ladder to a fresh session-PnL sample. Returns
    the new tier when a transition occurred, ``None`` otherwise.

    Transition rules:
    * Worsening (pnl drops further): escalate to the deepest tier
      the new pnl reaches.
    * Recovering: clear back to CLEAR ONLY when the pnl crosses
      back above the tier-1 threshold AND the pause cooldown (if
      any) has expired AND the test-resume window (if any) has
      finished.
    * RESUME_TESTING is a transient state — exited by
      ``observe_test_resume_fill`` (success → CLEAR; failure →
      next deeper tier).
    """
    cur = state.tier
    target = _tier_for(session_pnl_usd, thresholds)

    # Don't disturb a PAUSE_* while its cooldown is still active —
    # the deeper tier may have been entered above. ``observe_pause_expiry``
    # handles the ceiling-expiry → RESUME_TESTING transition.
    if cur in (DrawdownTier.PAUSE_SHORT, DrawdownTier.PAUSE_LONG):
        if now_mono < state.cooldown_until_mono:
            # Allow escalation but not de-escalation while paused.
            if _tier_rank(target) > _tier_rank(cur):
                state.favorable_exit_eligible_since_mono = 0.0
                _enter_tier(state, target, now_mono, now_iso, session_pnl_usd, thresholds)
                return target

            # 2K.10 (v1.5.182) — continuous-test favorable-exit
            # predicate. When session_pnl has recovered BACK above
            # the clear band (= arm threshold × clear_band_ratio)
            # AND has stayed there for ``favorable_exit_dwell_seconds``,
            # transition early to RESUME_TESTING. The fixed pause
            # ceiling stays as the safety net (handled by
            # ``observe_pause_expiry``). Disabled by default so
            # pre-migration callers see no behaviour change.
            if thresholds.favorable_exit_enabled:
                arm_threshold = (
                    float(thresholds.tier2_pause_short_usd)
                    if cur == DrawdownTier.PAUSE_SHORT
                    else float(thresholds.tier3_pause_long_usd)
                )
                clear_threshold = -arm_threshold * float(
                    thresholds.favorable_exit_clear_band_ratio
                )
                if session_pnl_usd >= clear_threshold:
                    # PnL is above the clear band — track or extend
                    # the eligibility timestamp.
                    if state.favorable_exit_eligible_since_mono <= 0.0:
                        state.favorable_exit_eligible_since_mono = float(
                            now_mono
                        )
                    elapsed = float(now_mono) - state.favorable_exit_eligible_since_mono
                    if elapsed >= float(thresholds.favorable_exit_dwell_seconds):
                        # Sustained recovery — exit pause early via
                        # RESUME_TESTING. Test-resume sampling still
                        # runs to confirm the markout regime cleared,
                        # so we don't immediately re-quote into an
                        # adverse environment.
                        state.favorable_exit_eligible_since_mono = 0.0
                        _enter_tier(
                            state,
                            DrawdownTier.RESUME_TESTING,
                            now_mono,
                            now_iso,
                            session_pnl_usd,
                            thresholds,
                        )
                        return DrawdownTier.RESUME_TESTING
                else:
                    # Dropped back below clear band — reset the dwell
                    # timer; the next eligible tick starts fresh.
                    state.favorable_exit_eligible_since_mono = 0.0

            return None

    # Transient RESUME_TESTING: only ``observe_test_resume_fill``
    # exits this state. PnL movement during the test window doesn't
    # trigger a fresh ladder transition (the sampling result owns
    # the decision).
    if cur == DrawdownTier.RESUME_TESTING:
        # Allow escalation if the pnl plummets further during testing —
        # don't insist on completing the test if the situation is
        # clearly worsening.
        if _tier_rank(target) >= _tier_rank(DrawdownTier.PAUSE_LONG):
            _enter_tier(state, target, now_mono, now_iso, session_pnl_usd, thresholds)
            return target
        return None

    if cur == DrawdownTier.KILLED:
        # Terminal. No automatic recovery; operator intervention
        # required (mirrors MAX_DRAWDOWN_USD semantics).
        return None

    if target == cur:
        return None

    _enter_tier(state, target, now_mono, now_iso, session_pnl_usd, thresholds)
    return target


def observe_pause_expiry(
    state: SessionDrawdownState,
    *,
    now_mono: float,
    now_iso: str,
    thresholds: TierThresholds,
) -> Optional[DrawdownTier]:
    """Called once per tick. When a PAUSE_* cooldown has expired,
    transition into RESUME_TESTING — quoting begins again at wider
    tier-3 spread floor, and the next ``test_resume_sample_fills``
    fills are observed before the gate decides.
    """
    if state.tier not in (DrawdownTier.PAUSE_SHORT, DrawdownTier.PAUSE_LONG):
        return None
    if now_mono < state.cooldown_until_mono:
        return None
    state.tier = DrawdownTier.RESUME_TESTING
    state.cooldown_until_mono = 0.0
    state.test_resume_fills_remaining = int(thresholds.test_resume_sample_fills)
    state.test_resume_markouts.clear()
    state.last_transition_iso = now_iso
    state.fire_count += 1
    return DrawdownTier.RESUME_TESTING


def observe_test_resume_fill(
    state: SessionDrawdownState,
    *,
    fill_markout_5s_bps: Optional[float],
    now_mono: float,
    now_iso: str,
    session_pnl_usd: float,
    thresholds: TierThresholds,
) -> Optional[DrawdownTier]:
    """Feed each fill that arrives during RESUME_TESTING into the
    sample buffer. When the buffer is full, decide:

    * If the buffer's mean markout >=
      ``-test_resume_max_adverse_bps`` → clear back to whichever
      tier the current PnL still warrants (CLEAR or WIDEN).
    * Otherwise → escalate to the next deeper tier.
    """
    if state.tier != DrawdownTier.RESUME_TESTING:
        return None
    if fill_markout_5s_bps is None:
        # Skip fills with no markout sample — the toxicity engine
        # has the same convention.
        return None

    state.test_resume_markouts.append(float(fill_markout_5s_bps))
    state.test_resume_fills_remaining -= 1
    if state.test_resume_fills_remaining > 0:
        return None

    # Sample window full — decide.
    samples = list(state.test_resume_markouts)
    mean_mk = sum(samples) / len(samples) if samples else 0.0
    if mean_mk >= -float(thresholds.test_resume_max_adverse_bps):
        # Markout cleared — exit gate to whatever tier current PnL
        # warrants (likely CLEAR or WIDEN).
        target = _tier_for(session_pnl_usd, thresholds)
        # If pnl has recovered too, we land on CLEAR; otherwise the
        # tier-1 widen latches again (no immediate re-escalation
        # since the markout cleared).
        if _tier_rank(target) >= _tier_rank(DrawdownTier.PAUSE_SHORT):
            target = DrawdownTier.WIDEN
        _enter_tier(state, target, now_mono, now_iso, session_pnl_usd, thresholds)
        return target

    # Markout still adverse — escalate.
    cur_rank = _tier_rank(DrawdownTier.RESUME_TESTING)
    next_tier = (
        DrawdownTier.PAUSE_LONG
        if cur_rank < _tier_rank(DrawdownTier.PAUSE_LONG)
        else DrawdownTier.KILLED
    )
    _enter_tier(state, next_tier, now_mono, now_iso, session_pnl_usd, thresholds)
    return next_tier


def is_quote_paused(state: SessionDrawdownState) -> bool:
    """True when the gate is in a state that should suppress
    quoting (HOLD_ALL eligibility). Read by the bot's quote loop."""
    return state.tier in (DrawdownTier.PAUSE_SHORT, DrawdownTier.PAUSE_LONG)


def is_widen_active(state: SessionDrawdownState) -> bool:
    """True when the gate is at tier 1 (widen) or in RESUME_TESTING
    (which keeps the tier-1 widen latched). The bot's spread floor
    overlay reads this."""
    return state.tier in (DrawdownTier.WIDEN, DrawdownTier.RESUME_TESTING)


def is_killed(state: SessionDrawdownState) -> bool:
    """Terminal kill state. Mirrors MAX_DRAWDOWN_USD semantics —
    the bot should halt and require operator intervention."""
    return state.tier == DrawdownTier.KILLED


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_TIER_RANK: dict[DrawdownTier, int] = {
    DrawdownTier.CLEAR: 0,
    DrawdownTier.WIDEN: 1,
    DrawdownTier.RESUME_TESTING: 2,
    DrawdownTier.PAUSE_SHORT: 3,
    DrawdownTier.PAUSE_LONG: 4,
    DrawdownTier.KILLED: 5,
}


def _tier_rank(t: DrawdownTier) -> int:
    return _TIER_RANK[t]


def _enter_tier(
    state: SessionDrawdownState,
    new_tier: DrawdownTier,
    now_mono: float,
    now_iso: str,
    session_pnl_usd: float,
    thresholds: TierThresholds,
) -> None:
    state.tier = new_tier
    state.last_trigger_pnl_usd = float(session_pnl_usd)
    state.last_transition_iso = now_iso
    state.fire_count += 1
    # 2K.10 — favorable-exit eligibility resets on every tier
    # transition so a previous PAUSE_*'s dwell history doesn't bleed
    # into a fresh PAUSE_*. Re-armed lazily inside observe_pnl.
    state.favorable_exit_eligible_since_mono = 0.0

    if new_tier == DrawdownTier.PAUSE_SHORT:
        state.cooldown_until_mono = now_mono + float(thresholds.pause_short_seconds)
        state.test_resume_fills_remaining = 0
        state.test_resume_markouts.clear()
    elif new_tier == DrawdownTier.PAUSE_LONG:
        state.cooldown_until_mono = now_mono + float(thresholds.pause_long_seconds)
        state.test_resume_fills_remaining = 0
        state.test_resume_markouts.clear()
    elif new_tier == DrawdownTier.RESUME_TESTING:
        state.cooldown_until_mono = 0.0
        state.test_resume_fills_remaining = int(thresholds.test_resume_sample_fills)
        state.test_resume_markouts.clear()
    else:
        state.cooldown_until_mono = 0.0
        state.test_resume_fills_remaining = 0
        state.test_resume_markouts.clear()
