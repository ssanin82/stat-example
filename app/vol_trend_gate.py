"""Vol × trend conjunction gate (1.2.2 — analysis-day 2026-05-10).

Targets the directional-burst pattern observed in snapshot
260510082955: a +210 bp move in 10 minutes simultaneously triggered
high realised vol AND sustained directional drift, and the bot's
both-sided quotes got picked off as the price ran through them.
The post-swing PnL gate fired late (after damage was done); the
existing toxicity_hard gate doesn't see "vol + trend" as a single
signal.

This gate fires PRE-emptively on the conjunction of:

* High realised vol relative to baseline (vol_ratio ≥ multiplier)
* Sustained directional drift (|mid_return_500ms_bps| ≥ threshold)
* Conditions persisting ≥ persistence_seconds (don't fire on a
  single-tick spike that the noise floor produces routinely)

When all three are met, the gate enters a cooldown that suppresses
quoting (HOLD_ALL) for cooldown_seconds. The cooldown is sticky
even if the conditions clear — the post-burst whipsaw window is
where the bot bleeds inventory unwinds.

Inputs are already computed by the bot tick:

* ``vol_ratio`` ≈ ``state.toxicity.vol_spike_ratio`` (current vol /
  baseline). Toxicity engine maintains this continuously.
* ``drift_bps`` ≈ ``raw_q.mid_return_500ms_bps`` (per-tick).

The gate is symmetric: positive drift fires it the same as negative
drift (the bot is hurt either way by sustained directional moves).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class VolTrendState:
    """In-memory state owned by ``BotState``. Single-threaded
    ownership: bot's main quote loop is the sole writer; live_stats
    publisher reads ``cooldown_until_mono`` for the dashboard."""

    armed_at_mono: Optional[float] = None
    """Monotonic timestamp when the conjunction first became true.
    None when the conjunction is currently false. Used to enforce
    ``persistence_seconds`` before activating the cooldown."""

    cooldown_until_mono: float = 0.0
    """Monotonic deadline for the active cooldown. Zero when not
    in cooldown."""

    last_trigger_vol_ratio: float = 0.0
    last_trigger_drift_bps: float = 0.0
    """Diagnostics: values that caused the most recent activation.
    Surfaced via live_stats for the Market tab."""

    fire_count: int = 0

    # v1.4.153 Phase 2K.3 — favorable-exit support.
    #
    # When the cooldown is active and BOTH vol_ratio + |drift_bps|
    # fall back below their respective ``clear_band × threshold``
    # values, start a dwell timer. Once that dwell elapses without
    # a re-flare, clear the cooldown EARLY rather than waiting the
    # full ``cooldown_seconds`` deadline. Counters below attribute
    # which exit path fired so the operator can calibrate the
    # clear band + dwell knobs from telemetry.
    favorable_dwell_started_mono: Optional[float] = None
    """Monotonic timestamp when BOTH signals first fell below the
    clear band during the active cooldown. None when either signal
    is still elevated, or when the gate is not currently in
    cooldown. Cleared back to None on a re-flare (vol or drift
    pops above the clear band again) so a brief blip resets the
    dwell clock."""

    cleared_via_favorable_total: int = 0
    cleared_via_ceiling_total: int = 0
    """Exit-attribution counters. Increment on the
    ``cooldown_active`` → ``cooldown_cleared`` edge.
      * ``favorable``: both signals fell below clear band AND held
        for the dwell window.
      * ``ceiling``: cooldown ran the full ``cooldown_seconds``.
    Operator calibration signal — ratio of the two over a session
    indicates whether the favorable-exit predicate is doing
    meaningful work."""

    was_active_last_call: bool = False
    """Set to True while the cooldown is active, flipped back to
    False on the call that clears it. Used to detect the
    ``active`` → ``cleared`` edge for attribution. Single-call
    flag — observe() is the sole writer."""


def observe(
    state: VolTrendState,
    *,
    now_mono: float,
    vol_ratio: Optional[float],
    drift_bps: Optional[float],
    vol_multiplier: float,
    drift_threshold_bps: float,
    persistence_seconds: float,
    cooldown_seconds: float,
    # v1.4.153 Phase 2K.3 — favorable-exit predicate knobs.
    clear_band_mult: float = 0.7,
    favorable_exit_dwell_seconds: float = 10.0,
) -> None:
    """Apply the conjunction logic to a fresh tick. Mutates state
    in place. Idempotent — safe to call from every quote tick.

    Four states tracked implicitly:

    1. **Disarmed** (armed_at_mono = None, cooldown inactive): conj-
       unction false. No action. Sets armed_at_mono on a future tick
       when conjunction becomes true.
    2. **Arming** (armed_at_mono != None, persistence not yet
       elapsed): conjunction has been true for some duration but
       not long enough yet. Continue waiting; reset if it clears.
    3. **Cooldown active** (cooldown_until_mono > now_mono): the
       gate has fired; suppress quoting until either:
         a) the MAX-cooldown deadline elapses (ceiling exit), or
         b) BOTH ``vol_ratio < vol_multiplier × clear_band_mult``
            AND ``|drift_bps| < drift_threshold_bps × clear_band_mult``
            simultaneously for ``favorable_exit_dwell_seconds``
            (favorable-exit predicate, v1.4.153 Phase 2K.3).
    4. **Cooldown cleared** (cooldown_until_mono = 0 after edge):
       attribution counter incremented based on which exit fired.

    The favorable-exit predicate hysteresis (0.7× by default vs
    the 1.0× trigger) prevents a single-tick blip from prematurely
    clearing the gate — both signals must STAY below the lower band
    for the dwell to keep counting up.
    """
    if cooldown_seconds <= 0.0 or vol_multiplier <= 0.0 or drift_threshold_bps <= 0.0:
        # Effectively disabled; reset any stale state.
        state.armed_at_mono = None
        state.favorable_dwell_started_mono = None
        state.was_active_last_call = False
        return

    if vol_ratio is None or drift_bps is None:
        # Missing input — can't evaluate. Don't disturb in-flight
        # arming or favorable-exit dwell; the next tick will retry.
        return
    if not math.isfinite(vol_ratio) or not math.isfinite(drift_bps):
        return

    # ------------------------------------------------------------------
    # COOLDOWN-ACTIVE branch — favorable-exit predicate runs here.
    # ------------------------------------------------------------------
    cooldown_active = now_mono < state.cooldown_until_mono
    if cooldown_active:
        state.was_active_last_call = True
        # Compute hysteresis clear band (e.g. multiplier=1.5,
        # clear_mult=0.7 → vol must drop below 1.05).
        vol_clear = vol_multiplier * clear_band_mult
        drift_clear = drift_threshold_bps * clear_band_mult
        below_clear_band = (
            vol_ratio < vol_clear and abs(drift_bps) < drift_clear
        )
        if below_clear_band:
            # Both signals below clear band — start (or continue) dwell.
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
            # Either signal re-flared above clear band — reset dwell.
            state.favorable_dwell_started_mono = None
        return

    # ------------------------------------------------------------------
    # COOLDOWN-CLEARING edge — attribute the exit. Reaches here when
    # the deadline has lapsed AND favorable-exit didn't fire first.
    # ------------------------------------------------------------------
    if state.was_active_last_call and state.cooldown_until_mono > 0.0:
        state.cleared_via_ceiling_total += 1
        state.cooldown_until_mono = 0.0
        state.favorable_dwell_started_mono = None
        state.was_active_last_call = False

    # ------------------------------------------------------------------
    # Standard arming logic (unchanged from pre-2K.3).
    # ------------------------------------------------------------------
    conjunction = (
        vol_ratio >= vol_multiplier
        and abs(drift_bps) >= drift_threshold_bps
    )
    if not conjunction:
        # Lost the conjunction — disarm. Re-arm the next time it's true.
        state.armed_at_mono = None
        return

    # Conjunction is true.
    if state.armed_at_mono is None:
        state.armed_at_mono = now_mono
        return

    # Persistence check.
    if now_mono - state.armed_at_mono < persistence_seconds:
        return

    # Persistence satisfied — fire.
    state.cooldown_until_mono = now_mono + cooldown_seconds
    state.last_trigger_vol_ratio = float(vol_ratio)
    state.last_trigger_drift_bps = float(drift_bps)
    state.fire_count += 1
    state.armed_at_mono = None  # consume the arming
    state.favorable_dwell_started_mono = None
    state.was_active_last_call = True


def is_active(state: VolTrendState, now_mono: float) -> bool:
    return now_mono < state.cooldown_until_mono


def seconds_remaining(state: VolTrendState, now_mono: float) -> float:
    return max(0.0, state.cooldown_until_mono - now_mono)


# ------------------------------------------------------------------
# Widening contribution (gate-to-widening Phase 1, v1.4.8+)
# ------------------------------------------------------------------

def widening_bps(
    state: VolTrendState,
    now_mono: float,
    max_half_spread_bps: float,
    *,
    widen_bps: float = -1.0,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` widening contribution.

    Symmetric: vol-spike + drift suppression doesn't have a side
    preference (HOLD_ALL on both sides pre-cutover).

    ``widen_bps`` (1.4.13 Phase 2): the magnitude this gate
    contributes when firing. Sentinel ``-1.0`` means "use
    ``max_half_spread_bps``" — gate-equivalent magnitude that
    matches the pre-cutover HOLD_ALL behaviour. Operator iterates
    this knob DOWN per ``plans/gate-to-widening.md`` Phase 2 to
    find the operating point where the bot stays in the market
    under vol+drift stress at a wider but non-MAX spread.

    Bounds: clamps to ``[0, max_half_spread_bps]`` so a misconfig
    can't produce a contribution exceeding the venue cap.
    """
    if not is_active(state, now_mono):
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    return (effective, effective)
