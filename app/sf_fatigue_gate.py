"""v1.5.202 — SF-event-count fatigue ladder.

Tier-based brake that pauses quoting when the bot has been forced
into SF too many times in a rolling window. Composes alongside the
existing PnL-driven ``session_drawdown`` ladder — they're orthogonal:

* ``session_drawdown`` keys on cumulative session **PnL** (dollar
  loss). Catches "any one big SF was expensive."
* ``sf_fatigue_gate`` keys on SF **event count** in a rolling
  window. Catches "many small SFs in quick succession before any
  single one crosses the PnL tier" — the storm-cluster pattern.

Both produce the same EFFECTS (widen / pause / kill) but on
different signals. Either firing pauses quoting; both clearing is
required for normal-mode resume.

Lifecycle:

1. Bot calls ``note_sf_event(state, now_mono)`` every time SF
   actually enters (NOT every time SF would have triggered).
   v1.5.198's re-entry cooldown already gates that, so noted events
   correspond to real episodes.
2. ``evaluate_tier(state, settings, now_mono)`` runs each
   eligibility tick, returning the current tier label.
3. The tier label is used by ``Bot._apply_eligibility_engine`` to
   decide whether to WIDEN the spread (Tier 1), HOLD_ALL (Tiers 2-3),
   or KILL the bot (Tier 4).
4. As the rolling window passes events out, the count drops, and
   the tier auto-clears.

Tier definitions (config-driven, defaults documented in
``Settings``):

* ``CLEAR`` — under tier-1 threshold, normal behaviour
* ``WIDEN`` — tier-1 breached: apply a multiplicative widening to
  the half-spread (does NOT suspend quoting)
* ``PAUSE_SHORT`` — tier-2 breached: HOLD_ALL for
  ``pause_short_seconds`` (default 600 = 10 min)
* ``PAUSE_LONG`` — tier-3 breached: HOLD_ALL for ``pause_long_seconds``
  (default 1800 = 30 min)
* ``KILLED`` — tier-4 breached: bot stops quoting permanently,
  operator restart required

Why tiers (not a single binary): mirrors the ``session_drawdown``
ladder semantics so the operator can mentally model both with the
same vocabulary. Both ladders use the same color taxonomy in the
dashboard.

This is the third structural-fix-class shipped in this release
arc — same architectural lesson as v1.5.197 (recent_fills time-
decay) and v1.5.198 (SF episode hard-timeout): **defensive memory
should age out by time, not by external events**.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional


# Tier label constants. String constants (not enum) for easier JSON
# round-tripping and frontend interop — matches the session_drawdown
# pattern.
TIER_CLEAR = "CLEAR"
TIER_WIDEN = "WIDEN"
TIER_PAUSE_SHORT = "PAUSE_SHORT"
TIER_PAUSE_LONG = "PAUSE_LONG"
TIER_KILLED = "KILLED"

# Tier severity ordering (higher = more restrictive). Used by
# evaluate_tier to pick the highest tier whose threshold is breached.
_TIER_RANK: dict[str, int] = {
    TIER_CLEAR: 0,
    TIER_WIDEN: 1,
    TIER_PAUSE_SHORT: 2,
    TIER_PAUSE_LONG: 3,
    TIER_KILLED: 4,
}


@dataclass
class SfFatigueGateState:
    """Per-bot state. Lives on ``BotState`` as
    ``state.sf_fatigue``. Thread-safe via ``lock``; the SF-event
    note path and the eligibility-tick read path both touch this."""

    lock: Lock = field(default_factory=Lock)
    # Rolling deque of monotonic timestamps for recent SF entries.
    # Pruned on every read older than ``window_seconds`` ago.
    # Cap is generous (1000) since real SF rates are <1/min.
    event_timestamps_mono: deque[float] = field(
        default_factory=lambda: deque(maxlen=1000),
    )
    # Last-evaluated tier (string label). Read by the eligibility
    # engine; written by ``evaluate_tier``. Stays at last value
    # between ticks for snapshot publishing.
    current_tier: str = TIER_CLEAR
    # When the tier crossed into PAUSE_*, the monotonic time at
    # which the pause was armed. The eligibility engine uses
    # ``arm_mono + pause_seconds`` to decide if the pause is still
    # active. None when tier <= WIDEN.
    pause_armed_at_mono: Optional[float] = None
    # Total tier-level fires this session (each time we enter
    # WIDEN/PAUSE_*/KILLED for the first time from a lower tier).
    # Surfaced in snapshot for operator visibility.
    fire_count: int = 0
    # Last tier we surfaced — used to detect "tier raised this
    # tick" edge events for logging / counter bumping.
    _last_published_tier: str = TIER_CLEAR


def _prune_window(
    state: SfFatigueGateState,
    *,
    now_mono: float,
    window_seconds: float,
) -> None:
    """Drop event timestamps older than ``window_seconds`` ago.
    Mutates state.event_timestamps_mono in place."""
    if window_seconds <= 0:
        return
    cutoff = now_mono - window_seconds
    dq = state.event_timestamps_mono
    while dq and dq[0] < cutoff:
        dq.popleft()


def note_sf_event(state: SfFatigueGateState, *, now_mono: float) -> None:
    """Record one SF entry. Called by ``Bot._enter_soft_flatten``
    AFTER the v1.5.198 re-entry cooldown check passed (so we count
    REAL SF entries, not blocked ones)."""
    with state.lock:
        state.event_timestamps_mono.append(float(now_mono))


def evaluate_tier(
    state: SfFatigueGateState,
    *,
    now_mono: float,
    window_seconds: float,
    tier1_widen_count: int,
    tier2_pause_short_count: int,
    tier3_pause_long_count: int,
    tier4_kill_count: int,
    pause_short_seconds: float,
    pause_long_seconds: float,
) -> str:
    """Compute the current tier from the rolling window.

    Returns the tier string. Mutates ``state.current_tier`` +
    ``state.pause_armed_at_mono`` + ``state.fire_count`` as side
    effects (incrementing fire_count only on the rising edge — when
    a higher tier is entered for the first time since CLEAR).

    Pure-ish: no IO, no external state, just deque pruning + count
    comparison. Safe to call every eligibility tick.

    Tier resolution: highest tier whose threshold is satisfied. So
    if count >= tier4 then KILLED regardless of pause cooldowns;
    PAUSE_LONG vs PAUSE_SHORT picks the highest count threshold met.
    """
    with state.lock:
        _prune_window(state, now_mono=now_mono, window_seconds=window_seconds)
        n = len(state.event_timestamps_mono)

        # Highest-tier-first resolution.
        if tier4_kill_count > 0 and n >= tier4_kill_count:
            new_tier = TIER_KILLED
        elif tier3_pause_long_count > 0 and n >= tier3_pause_long_count:
            new_tier = TIER_PAUSE_LONG
        elif tier2_pause_short_count > 0 and n >= tier2_pause_short_count:
            new_tier = TIER_PAUSE_SHORT
        elif tier1_widen_count > 0 and n >= tier1_widen_count:
            new_tier = TIER_WIDEN
        else:
            new_tier = TIER_CLEAR

        # Once KILLED, stay KILLED. Operator-restart required.
        # (Mirrors session_drawdown semantics.)
        if state.current_tier == TIER_KILLED:
            new_tier = TIER_KILLED

        # Pause arming: when ENTERING a PAUSE_* tier from a lower
        # tier, stamp the arming time. When leaving PAUSE_* (back
        # down to WIDEN/CLEAR via window decay), clear the stamp.
        was_paused = state.current_tier in (TIER_PAUSE_SHORT, TIER_PAUSE_LONG)
        is_paused = new_tier in (TIER_PAUSE_SHORT, TIER_PAUSE_LONG)
        if is_paused and not was_paused:
            state.pause_armed_at_mono = float(now_mono)
        elif is_paused and was_paused and new_tier != state.current_tier:
            # Tier escalated PAUSE_SHORT → PAUSE_LONG mid-pause:
            # re-arm the timer so the longer pause window starts
            # fresh. Conservative — avoids the case where the
            # short-pause clock would otherwise expire mid-long-pause.
            state.pause_armed_at_mono = float(now_mono)
        elif not is_paused and was_paused:
            state.pause_armed_at_mono = None

        # PAUSE_* tiers also age out via their own cooldown timer
        # in addition to the rolling-window count decay. If
        # pause_armed + pause_seconds has elapsed AND the event
        # count is below the pause threshold, drop the tier.
        if is_paused and state.pause_armed_at_mono is not None:
            pause_budget = (
                pause_short_seconds
                if new_tier == TIER_PAUSE_SHORT
                else pause_long_seconds
            )
            elapsed = float(now_mono) - state.pause_armed_at_mono
            if elapsed >= pause_budget:
                # Pause budget expired. Drop to the count-driven
                # tier (which may be CLEAR / WIDEN / lower-PAUSE).
                # Re-evaluate without the pause filter.
                if tier3_pause_long_count > 0 and n >= tier3_pause_long_count:
                    # Still over long threshold → stay long-paused
                    # but with a fresh timer (operator visibility:
                    # this signals "fatigue persists").
                    state.pause_armed_at_mono = float(now_mono)
                    new_tier = TIER_PAUSE_LONG
                elif (
                    tier2_pause_short_count > 0
                    and n >= tier2_pause_short_count
                ):
                    state.pause_armed_at_mono = float(now_mono)
                    new_tier = TIER_PAUSE_SHORT
                elif (
                    tier1_widen_count > 0 and n >= tier1_widen_count
                ):
                    state.pause_armed_at_mono = None
                    new_tier = TIER_WIDEN
                else:
                    state.pause_armed_at_mono = None
                    new_tier = TIER_CLEAR

        # Fire-count: bump on RISING edge (entering a stricter tier
        # than we were at). Falling edges (window decay back to a
        # looser tier) don't bump — the operator wants to know how
        # many TIMES the fatigue ladder mattered.
        if _TIER_RANK[new_tier] > _TIER_RANK[state.current_tier]:
            state.fire_count += 1
        state.current_tier = new_tier
        return new_tier


def seconds_remaining_on_pause(
    state: SfFatigueGateState,
    *,
    now_mono: float,
    pause_short_seconds: float,
    pause_long_seconds: float,
) -> float:
    """Helper for snapshot publishing. Returns the remaining cooldown
    in seconds when in a PAUSE_* tier, else 0.0. Pure read."""
    with state.lock:
        if state.current_tier not in (TIER_PAUSE_SHORT, TIER_PAUSE_LONG):
            return 0.0
        if state.pause_armed_at_mono is None:
            return 0.0
        budget = (
            pause_short_seconds
            if state.current_tier == TIER_PAUSE_SHORT
            else pause_long_seconds
        )
        elapsed = float(now_mono) - state.pause_armed_at_mono
        return max(0.0, budget - elapsed)


def event_count_in_window(state: SfFatigueGateState) -> int:
    """Number of SF events currently in the rolling window. Read
    AFTER ``evaluate_tier`` has run this tick (which prunes the
    window). Pure read."""
    with state.lock:
        return len(state.event_timestamps_mono)


def snapshot_dict(
    state: SfFatigueGateState,
    *,
    now_mono: float,
    window_seconds: float,
    tier1_widen_count: int,
    tier2_pause_short_count: int,
    tier3_pause_long_count: int,
    tier4_kill_count: int,
    pause_short_seconds: float,
    pause_long_seconds: float,
    enabled: bool,
) -> dict:
    """Render the gate state as a JSON-serialisable dict, mirroring
    the session_drawdown shape. Used by both
    ``_behavioural_gates_snapshot`` and the live_stats top-level
    block. Read-only; caller must have already run
    ``evaluate_tier`` this tick to ensure the window is fresh."""
    # NOTE: compute the cooldown inline (not via seconds_remaining_on_pause)
    # so we only acquire state.lock once — the gate's Lock is non-reentrant.
    with state.lock:
        tier = state.current_tier
        if tier in (TIER_PAUSE_SHORT, TIER_PAUSE_LONG) and state.pause_armed_at_mono is not None:
            budget = (
                pause_short_seconds
                if tier == TIER_PAUSE_SHORT
                else pause_long_seconds
            )
            elapsed = float(now_mono) - state.pause_armed_at_mono
            cooldown_remaining = max(0.0, budget - elapsed)
        else:
            cooldown_remaining = 0.0
        return {
            "enabled": bool(enabled),
            "window_seconds": float(window_seconds),
            "tier": tier,
            "events_in_window": len(state.event_timestamps_mono),
            "tier1_widen_count": int(tier1_widen_count),
            "tier2_pause_short_count": int(tier2_pause_short_count),
            "tier3_pause_long_count": int(tier3_pause_long_count),
            "tier4_kill_count": int(tier4_kill_count),
            "cooldown_seconds_remaining": float(cooldown_remaining),
            "fire_count": int(state.fire_count),
        }
