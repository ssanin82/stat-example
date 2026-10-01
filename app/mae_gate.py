"""30-second MAE gate (analysis-day 2026-05-15).

A defensive-pause gate that operates at the 30s post-fill horizon,
where the existing toxicity engine's 5s markout signal does not
reach.

Background (snapshot 260515-095056):
* Session bled $0.09 over 2.3h, 51 fills.
* 5s markout was actually good — median -0.43 bp.
* 30s MAE was -7 to -8 bp (BUY -6.83 median, SELL -8.33 median).
* The bot couldn't see the bleed at the 5s horizon, so its existing
  toxicity engine never engaged its hard threshold.

Mechanism
---------

The bot's ``PostFillExcursionWatcher`` already computes per-fill
``mae_30s_bps`` after the 30s window elapses. This gate observes
those values as they finalise and maintains a rolling N-fill average
of ``min(0, mae_30s_bps)``. When the rolling average crosses
``-HARD_BPS`` (a negative threshold expressed as a positive number),
the gate arms a cooldown during which the eligibility path returns
HOLD_ALL.

It is parallel to the toxicity engine's ``hard`` trigger, just at a
different horizon. The two compose by side — either one firing is
enough to suppress quoting.

Threading
---------

The watcher daemon thread calls ``observe()`` after each 30s window
closes. The bot's main quote loop calls ``is_active()`` every cycle.
The shared state — a deque, a single float deadline, and integer
counters — is protected by a lock; reads outside the lock are
acceptable because the float deadline is single-attribute (atomic
under GIL for our purposes) and an inconsistent read at a quote-loop
boundary self-corrects within one cycle.

Pure-state design mirrors ``app.post_swing_gate``: the gate owns no
clock, takes ``now_mono`` from the caller. Easy to unit-test.

Phase 2K.7 (v1.4.158) — favorable-exit predicate
------------------------------------------------

The cooldown is no longer purely time-based. While
``MAE_GATE_COOLDOWN_SECONDS`` is still the MAX-cooldown ceiling, the
gate also clears EARLY when the rolling-N-fill MAE average has
recovered above ``-hard_threshold_bps × clear_band_mult`` (e.g.
hard=5.0 and mult=0.5 → clear band -2.5 bps) and held there for
``favorable_exit_dwell_seconds``. The dwell is intentionally short
(default 5 s): each new 30 s-resolved fill is a genuine signal
change, so we don't need a long hysteresis window — just enough that
a single noise sample can't flip the gate.

Re-flare: a fresh adverse fill that drags the rolling average back
below the clear band resets the dwell timer.

Exit attribution: ``cleared_via_favorable_total`` vs
``cleared_via_ceiling_total`` lets the operator see whether the
predicate is doing meaningful work. Ratio >50 % favorable suggests
the gate is correctly clearing on real recovery; ~100 % ceiling
suggests the clear-band knob is too tight (raise ``clear_band_mult``
toward 0.7-0.8) or the dwell is too long.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app import clock as _clock


@dataclass
class MaeGateState:
    """In-memory state owned by ``BotState``.

    Producer: the post-fill-excursion watcher daemon thread, calling
    ``observe()`` once per fill at the 30s deadline.
    Consumer: the bot's main quote loop, calling ``is_active()``
    every cycle.
    """

    samples: deque = field(default_factory=deque)
    """Rolling buffer of recent ``mae_30s_bps`` values (clipped to
    ``min(0, mae)``). Sized by ``fill_window`` parameter passed to
    ``observe()``."""

    cooldown_until_mono: float = 0.0
    """Monotonic deadline. While ``_clock.monotonic() < this``, the
    eligibility path returns HOLD_ALL with reason
    ``"mae_gate_cooldown"``. Set to 0.0 when favorable-exit clears
    the gate early so ``is_active`` edge detection can attribute the
    transition correctly."""

    last_trigger_avg_bps: float = 0.0
    """The rolling-average value that fired the most recent trigger.
    Diagnostic / dashboard."""

    fire_count: int = 0
    """Count of trigger fires this session — for operator visibility
    on whether the gate is engaging at all."""

    # Phase 2K.7 favorable-exit state.
    favorable_dwell_started_mono: Optional[float] = None
    """When non-None, marks the start of the favorable-exit dwell.
    Reset on re-flare (predicate flips back to not-holding) and on
    successful early-clear."""

    cleared_via_favorable_total: int = 0
    """Count of cooldowns cleared via the favorable-exit predicate
    this session — calibration signal for the clear_band_mult knob."""

    cleared_via_ceiling_total: int = 0
    """Count of cooldowns that hit the MAX ceiling (timer expired
    naturally) this session. Ratio favorable / (favorable + ceiling)
    is the operator-facing tuning signal."""

    cleared_via_position_favorable_total: int = 0
    """v1.5.155 — count of cooldowns cleared via the position-aware
    favorable exit (see ``evaluate_position_favorable_exit``). Tracks
    distinctly from ``cleared_via_favorable_total`` (markout-based
    Phase 2K.7 exit) so the operator can see which clearance path is
    doing the work. Operator instruction 2026-05-26 (CLAUDE.md
    Rule 0c): every timer-based gate must ALSO have a signal-driven
    conditional exit that considers current bot position. The Phase
    2K.7 markout exit doesn't satisfy this — it only fires when fill
    bleeding stops, which in a sustained adverse-trend regime
    requires waiting for the NEXT fill (and there may be none while
    paused). The position-aware exit fires when the bot has
    significant inventory and the market is moving favorably for
    that inventory."""

    was_active_last_call: bool = False
    """Set by ``is_active()`` for active→cleared edge detection so we
    can attribute ceiling expirations exactly once."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    """Protects ``samples``, ``cooldown_until_mono``, ``fire_count``,
    ``last_trigger_avg_bps``, the favorable-exit fields, and
    ``was_active_last_call``. Held briefly in ``observe()`` and during
    the counter-increment edge in ``is_active()``."""


def observe(
    state: MaeGateState,
    *,
    now_mono: float,
    mae_30s_bps: Optional[float],
    fill_window: int,
    hard_threshold_bps: float,
    cooldown_seconds: float,
    favorable_exit_enabled: bool = True,
    clear_band_mult: float = 0.5,
    favorable_exit_dwell_seconds: float = 5.0,
) -> None:
    """Record a finalised 30s MAE value and possibly arm the cooldown.

    * Appends ``min(0, mae_30s_bps)`` to the buffer (favourable
      excursions clip to zero — the gate measures adverse pressure
      only, not directional bleed).
    * Drops the oldest entries to keep ``len(samples) <= fill_window``.
    * Fires when the buffer is full AND the average crosses
      ``-hard_threshold_bps``. (``hard_threshold_bps`` is expressed
      as a positive number; the threshold is its negation.)
    * Does NOT re-fire while a cooldown is already active — caller
      gets one suppression window per crossing, then must observe a
      fresh non-firing window before the next trigger.

    Phase 2K.7: while in cooldown, also evaluates the favorable-exit
    predicate. If the rolling average has recovered above the clear
    band (``-hard_threshold_bps × clear_band_mult``) and held there
    for ``favorable_exit_dwell_seconds``, the cooldown is cleared
    early (``cooldown_until_mono`` reset to 0.0,
    ``cleared_via_favorable_total`` incremented, dwell reset).

    Idempotent against missing data: ``mae_30s_bps=None`` is a no-op
    (the watcher occasionally skips, e.g. degenerate mid_at_fill).
    """
    if mae_30s_bps is None:
        return
    clipped = min(0.0, float(mae_30s_bps))
    with state.lock:
        state.samples.append(clipped)
        # Evict from the left until we're within the window.
        while len(state.samples) > fill_window:
            state.samples.popleft()
        in_cooldown = now_mono < state.cooldown_until_mono
        if not in_cooldown:
            # Don't fire until the buffer is full — warmup guard so
            # a single bad fill at session start can't trigger a
            # 3-min pause.
            if len(state.samples) < fill_window:
                return
            avg = sum(state.samples) / len(state.samples)
            if avg <= -hard_threshold_bps:
                state.cooldown_until_mono = (
                    now_mono + max(0.0, cooldown_seconds)
                )
                state.last_trigger_avg_bps = avg
                state.fire_count += 1
                # Fresh arming cancels any in-flight favorable dwell.
                state.favorable_dwell_started_mono = None
            return

        # In cooldown — evaluate the Phase 2K.7 favorable-exit
        # predicate. Need a full window for a stable average.
        if not favorable_exit_enabled:
            return
        if len(state.samples) < fill_window:
            return
        avg = sum(state.samples) / len(state.samples)
        clear_band = -float(hard_threshold_bps) * float(clear_band_mult)
        # hard=5, mult=0.5 → clear_band=-2.5 → clear when avg > -2.5.
        # mult=1.0 → clear_band = -hard (the trigger boundary; weakest
        #   hysteresis — clear as soon as avg lifts above the trigger).
        # mult=0.0 → clear_band = 0 (strongest hysteresis — avg must
        #   be strictly non-adverse).
        if avg > clear_band:
            if state.favorable_dwell_started_mono is None:
                state.favorable_dwell_started_mono = now_mono
            elif (
                now_mono - state.favorable_dwell_started_mono
                >= favorable_exit_dwell_seconds
            ):
                # Dwell satisfied → clear cooldown early.
                state.cooldown_until_mono = 0.0
                state.cleared_via_favorable_total += 1
                state.favorable_dwell_started_mono = None
                # Reset edge-detection flag so the next is_active()
                # poll doesn't mis-attribute this transition to the
                # ceiling counter.
                state.was_active_last_call = False
        else:
            # Predicate not holding — reset dwell (re-flare).
            state.favorable_dwell_started_mono = None


def evaluate_position_favorable_exit(
    state: MaeGateState,
    *,
    now_mono: float,
    position_qty: float,
    drift_bps: Optional[float],
    inventory_threshold: float,
    drift_threshold_bps: float,
) -> bool:
    """v1.5.155 — position-aware favorable-exit predicate.

    Per CLAUDE.md Rule 0c (operator instruction 2026-05-26): every
    timer-based gate must ALSO have a signal-driven conditional exit
    that considers current bot position. The Phase 2K.7 markout-based
    favorable exit (``observe()``'s in-cooldown branch) only fires
    when recent fill bleeding stops — but during a cooldown there ARE
    no new fills (gate paused both sides), so observe() doesn't run
    and the markout-based exit never triggers. Hence the
    v1.5.154-260526-074029 snapshot's 78 cooldowns / 0 favorable / 78
    ceiling clearance ratio.

    This second clearance path fires when:

    * ``|position_qty| >= inventory_threshold`` — the bot has
      meaningful inventory to unwind, AND
    * ``sign(position_qty) * drift_bps >= drift_threshold_bps`` —
      current drift is in the SAME direction as inventory, meaning
      inventory is gaining value. This is exactly the moment when
      the bot should be unwinding (the favorable side fills at a
      good price). Pausing through this moment misses the
      mean-reversion / trend-with-inventory opportunity.

    Called by the bot's main quote loop every tick (cheap — no IO,
    no allocations). Idempotent: returns False without side effects
    when not in cooldown OR predicate doesn't hold.

    Returns True if this call cleared the cooldown.
    """
    import math

    if drift_bps is None or not math.isfinite(drift_bps):
        return False
    if abs(position_qty) < inventory_threshold:
        return False
    sign_pos = math.copysign(1.0, position_qty) if position_qty != 0.0 else 0.0
    if sign_pos * float(drift_bps) < drift_threshold_bps:
        return False
    with state.lock:
        if state.cooldown_until_mono <= now_mono:
            return False
        state.cooldown_until_mono = 0.0
        state.cleared_via_position_favorable_total += 1
        state.was_active_last_call = False
    return True


def evaluate_idle_clear(
    state: MaeGateState,
    *,
    now_mono: float,
    last_fill_mono: Optional[float],
    idle_clear_seconds: float,
) -> bool:
    """v1.5.197 — idle-decay exit predicate.

    The MAE-gate cooldown is driven by MAE statistics from recent
    fills. When the gate engages, it pauses quoting on both sides;
    by construction no NEW fills arrive while it's active, so the
    MAE-based exit predicate (``observe()``) is never re-evaluated
    and the gate can only exit via the time-ceiling. If something
    else (structural_bias_throttle, toxicity_hard, freshness holds)
    keeps preventing fills past the ceiling deadline, the gate
    re-arms on stale data the moment a fill does arrive.

    This third clearance path fires when:

    * The gate is active
    * No fill has arrived for at least ``idle_clear_seconds``

    Rationale: if the source signal (recent fill MAE) is stale by
    N minutes, the gate's reaction to it should also be stale. The
    idle-decay path mirrors v1.5.197's toxicity recent_fills
    time-decay — defensive memory should age out by time, not only
    by new-fill volume (since the defense itself blocks new fills).

    Returns True if this call cleared the cooldown via the idle
    path.
    """
    if last_fill_mono is None:
        # No fill ever recorded — engine hasn't bootstrapped. Don't
        # clear (gate may not even be active in this regime).
        return False
    if idle_clear_seconds <= 0.0:
        return False
    if (now_mono - float(last_fill_mono)) < idle_clear_seconds:
        return False
    with state.lock:
        if state.cooldown_until_mono <= now_mono:
            return False
        state.cooldown_until_mono = 0.0
        # Bump the position-favorable counter for now — semantically
        # this is "cleared by an adaptive predicate, not the time
        # ceiling." A dedicated cleared_via_idle_total counter could
        # be added in a future minor; for v1.5.197 we lump it with
        # position_favorable to keep the snapshot schema stable.
        state.cleared_via_position_favorable_total += 1
        state.was_active_last_call = False
    return True


def is_active(state: MaeGateState, now_mono: float) -> bool:
    """True while the cooldown deadline is in the future.

    Side effect (Phase 2K.7): detects the active→cleared edge for
    ceiling attribution. If the cooldown expired naturally (timer
    ran out, no favorable-exit clearing happened), bump
    ``cleared_via_ceiling_total`` exactly once. Locked because the
    counter increment must be coherent under contention with the
    watcher thread's ``observe()`` calls.
    """
    active = now_mono < state.cooldown_until_mono
    with state.lock:
        was = state.was_active_last_call
        if was and not active:
            # Active → cleared edge.
            # If ``cooldown_until_mono > 0`` here the timer ran out
            # naturally (favorable-exit would have set it to 0.0 and
            # reset ``was_active_last_call`` to False, so we wouldn't
            # be in this branch).
            if state.cooldown_until_mono > 0.0:
                state.cleared_via_ceiling_total += 1
        state.was_active_last_call = active
    return active


def seconds_remaining(state: MaeGateState, now_mono: float) -> float:
    """Wall-equivalent remaining cooldown. Zero when not active."""
    return max(0.0, state.cooldown_until_mono - now_mono)


def reset(state: MaeGateState) -> None:
    """Clear the rolling buffer and cooldown. Used at session start
    so a prior session's tail doesn't contaminate the new session.

    Also clears Phase 2K.7 favorable-exit attribution counters so
    snapshot ratios are session-scoped."""
    with state.lock:
        state.samples.clear()
        state.cooldown_until_mono = 0.0
        state.fire_count = 0
        state.last_trigger_avg_bps = 0.0
        state.favorable_dwell_started_mono = None
        state.cleared_via_favorable_total = 0
        state.cleared_via_ceiling_total = 0
        state.cleared_via_position_favorable_total = 0
        state.was_active_last_call = False
