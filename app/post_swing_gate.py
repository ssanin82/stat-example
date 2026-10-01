"""Post-swing PnL cooldown gate (analysis-day 2026-05-10 behavioural change).

Detects rapid PnL movement — in either direction — and pauses quoting
for a configurable cooldown so the bot doesn't immediately re-engage a
directional regime that just whipsawed inventory. Targets the snapshot
260510064549 pattern observed at 00:59-01:05 UTC: peak +$1.23 (long
inventory monetised on a rebound), then long-side fills walked into
the next downtick, dropping back to +$0.02 in 6 minutes.

Algorithm — pure function on a rolling sample buffer:

1. Caller samples ``total_pnl_usd`` every tick (cheap; ~once per
   second). Each sample is timestamped on the monotonic clock.
2. Stale samples (older than ``window_seconds``) are evicted.
3. Trigger fires when ``max(buffer) - min(buffer)`` exceeds
   ``pnl_delta_usd_threshold``.
4. Trigger sets a monotonic deadline ``cooldown_until_mono``; the
   quote-eligibility path reads that deadline and returns HOLD_ALL
   while the deadline is in the future.

The gate is symmetric: a +$0.50 spike *and* a −$0.50 drop both fire.
The intent is "regime just shifted — sit it out and let it settle"
either way; trying to capitalise on a spike is exactly the
inventory-flip pattern this gate exists to prevent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app import clock as _clock


@dataclass
class PostSwingState:
    """In-memory state owned by ``BotState``. Single-threaded usage —
    the bot's quote loop is the sole writer; the live_stats publisher
    only reads ``cooldown_until_mono`` and ``last_trigger_reason`` for
    surfacing on the dashboard.
    """

    samples: deque = field(default_factory=deque)
    """Rolling buffer of (mono_ts, total_pnl_usd) samples. Bounded by
    ``window_seconds`` (caller-provided), evicted on every observe()."""

    cooldown_until_mono: float = 0.0
    """Monotonic deadline. While ``_clock.monotonic() < this``, the
    quote-eligibility path returns HOLD_ALL with reason
    ``"post_swing_cooldown"``."""

    last_trigger_reason: Optional[str] = None
    """``"spike"`` (positive PnL delta) / ``"drop"`` (negative) /
    ``None`` if the gate has never fired this session. Surfaced via
    live_stats for the dashboard's Market tab."""

    last_trigger_delta_usd: float = 0.0
    """Magnitude (signed) of the PnL delta that fired the most recent
    trigger. Diagnostic only."""

    fire_count: int = 0
    """Count of trigger fires this session — useful for the operator
    to spot if the gate is over-firing on a quiet day."""

    # v1.4.154 Phase 2K.4 — favorable-exit support.
    #
    # When the cooldown is active and the rolling PnL-delta
    # (max(samples) - min(samples)) shrinks below
    # ``pnl_delta_usd_threshold × clear_band_mult``, start the
    # favorable-exit dwell timer. Once that dwell elapses without a
    # re-flare, clear the cooldown EARLY rather than waiting the
    # full ``cooldown_seconds`` deadline. Counters below attribute
    # which exit path fired so the operator can calibrate the knobs.
    favorable_dwell_started_mono: Optional[float] = None
    """Mono timestamp when the rolling PnL-delta first shrunk below
    the clear band during the active cooldown. None when the delta
    is still elevated, or when the gate is not currently in
    cooldown. Cleared back to None on a re-flare (delta widens past
    the clear band again) so a brief blip resets the dwell clock."""

    cleared_via_favorable_total: int = 0
    cleared_via_ceiling_total: int = 0
    """Exit-attribution counters. Increment on the
    ``cooldown_active`` → ``cooldown_cleared`` edge.
      * ``favorable``: PnL-delta shrunk below clear band AND held
        for the dwell window.
      * ``ceiling``: cooldown ran the full ``cooldown_seconds``.
    Operator calibration signal — ratio over a session indicates
    whether the favorable-exit predicate is doing meaningful work
    or whether the ceiling is the binding constraint."""

    was_active_last_call: bool = False
    """Tracks the active state from the prior observe() call so we
    can detect the ``active`` → ``cleared`` edge and attribute the
    clearing event correctly. observe() is the sole writer."""


def observe(
    state: PostSwingState,
    *,
    now_mono: float,
    total_pnl_usd: float,
    window_seconds: float,
    pnl_delta_usd_threshold: float,
    cooldown_seconds: float,
    # v1.4.154 Phase 2K.4 — favorable-exit predicate knobs.
    clear_band_mult: float = 0.5,
    favorable_exit_dwell_seconds: float = 15.0,
) -> None:
    """Record a PnL sample. Mutates ``state`` in place. Idempotent —
    safe to call from any tick rate; the buffer's eviction by wall-
    time keeps memory bounded.

    The trigger only fires when the buffer window genuinely covers
    ``window_seconds`` (i.e. has at least one sample older than that
    window, evicted on this call). Without that guard, a freshly-
    started session with sparse samples could spuriously fire on the
    very first PnL move.

    v1.4.154 Phase 2K.4: dual-track exit. When the cooldown is
    active AND the rolling PnL-delta (max-min over the visible
    sample window) shrinks below
    ``pnl_delta_usd_threshold × clear_band_mult``, start a dwell
    timer. After ``favorable_exit_dwell_seconds`` of continuous
    sub-threshold delta, the cooldown clears early. ``clear_mult=0``
    disables the predicate (pure timer behaviour). A re-widening
    of the delta past the clear band resets the dwell — both
    extremes must STAY tight for the entire window.
    """
    state.samples.append((now_mono, total_pnl_usd))

    # Evict samples older than the window. Keep one stale sample as
    # the "anchor" — that's what we measure against.
    cutoff = now_mono - window_seconds
    had_stale = False
    while len(state.samples) > 1 and state.samples[0][0] < cutoff:
        state.samples.popleft()
        had_stale = True

    # ------------------------------------------------------------------
    # COOLDOWN-ACTIVE branch — favorable-exit predicate runs here.
    # The PnL-delta is recomputed every tick from the current sample
    # window (newest sample just appended above). Same metric the
    # trigger uses, but compared against a tighter band.
    # ------------------------------------------------------------------
    cooldown_active = now_mono < state.cooldown_until_mono
    if cooldown_active:
        state.was_active_last_call = True
        if (
            clear_band_mult > 0.0
            and len(state.samples) >= 2
            and pnl_delta_usd_threshold > 0.0
        ):
            pnls = [p for _, p in state.samples]
            cur_delta = max(pnls) - min(pnls)
            clear_band = pnl_delta_usd_threshold * clear_band_mult
            if cur_delta < clear_band:
                # Delta tight — start (or continue) the dwell.
                if state.favorable_dwell_started_mono is None:
                    state.favorable_dwell_started_mono = now_mono
                elif (
                    now_mono - state.favorable_dwell_started_mono
                    >= favorable_exit_dwell_seconds
                ):
                    # Dwell satisfied — favorable exit fires.
                    state.cooldown_until_mono = 0.0
                    state.favorable_dwell_started_mono = None
                    state.cleared_via_favorable_total += 1
                    state.was_active_last_call = False
            else:
                # Delta re-widened past clear band — reset dwell.
                state.favorable_dwell_started_mono = None
        return

    # ------------------------------------------------------------------
    # COOLDOWN-CLEARING edge — attribute when the deadline has lapsed
    # without a favorable-exit. Reaches here only after the active
    # branch above has flipped off and we're in the post-cooldown
    # path. Single increment per cooldown cycle.
    # ------------------------------------------------------------------
    if state.was_active_last_call and state.cooldown_until_mono > 0.0:
        state.cleared_via_ceiling_total += 1
        state.cooldown_until_mono = 0.0
        state.favorable_dwell_started_mono = None
        state.was_active_last_call = False

    # Don't fire until the buffer covers the full window. This is
    # the warmup guard — at session start the buffer is empty and a
    # single PnL move shouldn't trigger.
    if not had_stale:
        return

    if len(state.samples) < 2:
        return

    pnls = [p for _, p in state.samples]
    pmax = max(pnls)
    pmin = min(pnls)
    delta = pmax - pmin
    if delta < pnl_delta_usd_threshold:
        return

    # Determine spike vs drop by which extreme is more recent. If the
    # latest sample is closer to the max → "spike" (PnL just popped
    # up); closer to the min → "drop".
    latest = state.samples[-1][1]
    reason = "spike" if (pmax - latest) <= (latest - pmin) else "drop"
    signed = (latest - state.samples[0][1])

    state.cooldown_until_mono = now_mono + max(0.0, cooldown_seconds)
    state.last_trigger_reason = reason
    state.last_trigger_delta_usd = float(signed)
    state.fire_count += 1
    state.favorable_dwell_started_mono = None
    state.was_active_last_call = True


def is_active(state: PostSwingState, now_mono: float) -> bool:
    """True while the cooldown deadline is in the future. Read-only;
    safe from any thread."""
    return now_mono < state.cooldown_until_mono


def seconds_remaining(state: PostSwingState, now_mono: float) -> float:
    """Wall-equivalent remaining cooldown. Zero when not active."""
    return max(0.0, state.cooldown_until_mono - now_mono)


# ------------------------------------------------------------------
# Widening contribution (gate-to-widening Phase 1, v1.4.8+)
# ------------------------------------------------------------------

def widening_bps(
    state: PostSwingState,
    now_mono: float,
    max_half_spread_bps: float,
    *,
    widen_bps: float = -1.0,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` widening contribution.

    Symmetric (swing affects both sides). ``widen_bps`` sentinel
    ``-1.0`` falls back to ``max_half_spread_bps`` (gate-equivalent
    magnitude). See ``plans/gate-to-widening.md`` Phase 2 for the
    operator iteration loop.
    """
    if not is_active(state, now_mono):
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    return (effective, effective)
