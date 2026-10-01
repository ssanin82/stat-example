"""v1.5.248 — No-fill aggression escalator.

When the bot fails to fill for a sustained period (likely because
quotes are too defensive or too far from touch), this module
computes an "aggression level" in [0.0, 1.0] that several knobs
across compute_quote_decision then attenuate against:

* 0.0 — no aggression. Normal quoting. (seconds_since_last_fill
  below the trigger threshold.)
* 1.0 — full aggression. Spread compressed to MIN_HALF, microprice
  widening suppressed, toxicity bumps suppressed, reservation
  shifts attenuated toward mid.

The level ramps LINEARLY from 0 to 1 over the configured ramp
window, starting at the configured trigger threshold:

    seconds_since_last_fill < trigger        → 0.0
    seconds_since_last_fill in [trigger,
                                trigger+ramp] → linear interpolation
    seconds_since_last_fill > trigger+ramp   → 1.0

When a fill arrives, ``seconds_since_last_fill`` resets and the
level drops back to 0.0 on the next tick.

Replaces the single-dimension NO_FILL_COMPRESS feature from
v1.5.156 Option B (which only compressed spread). The escalator
is OPT-IN via ``NO_FILL_ESCALATOR_ENABLED``; when enabled, it
takes over the no-fill response and the legacy NO_FILL_COMPRESS
knobs are bypassed. When disabled, legacy compress behavior is
preserved exactly.

Design rationale (operator-stated 2026-05-29):

> "is there any way to adaptively increase the aggression if
>  there was no fill for too long?"

The answer was: yes, NO_FILL_COMPRESS exists but only adjusts
spread, takes 10+ min to trigger, and doesn't address other
fill-blocking knobs (microprice widening, defensive toxicity
bumps, alpha-driven reservation shifts that push quotes off mid).
This module is the response — a single continuous "aggression
level" that drives multiple dimensions simultaneously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EscalatorOutput:
    """The escalator's per-tick output. Consumed by
    ``compute_quote_decision`` to attenuate/suppress various
    aggression-blocking knobs.

    All fields are derived from the single ``aggression_level``
    in [0.0, 1.0] using config-driven multipliers. Splitting them
    out into a struct (instead of just passing the level) lets the
    caller A/B individual dimensions independently and gives the
    snapshot a structured view of what the escalator is doing.
    """

    # Headline metric: 0.0 = inactive, 1.0 = max aggression.
    aggression_level: float

    # Half-spread compression in bp. Subtracted from the half-spread
    # AFTER all other contributions. Floored by MIN_HALF_SPREAD_BPS
    # downstream.
    spread_compression_bps: float

    # Multiplier on the microprice widening contribution. 1.0 = no
    # attenuation; 0.0 = fully suppressed.
    microprice_widen_multiplier: float

    # Multiplier on the toxicity soft-bump contribution.
    toxicity_widen_multiplier: float

    # Multiplier on the reservation-shift sum (alpha contributions).
    # 1.0 = no change; 0.5 = halve the alpha pull.
    reservation_shift_multiplier: float

    # Diagnostic: which knob is doing the work at the current level?
    # Empty when level is 0.0; otherwise lists the active dimensions
    # for postmortem / snapshot readers.
    active_dimensions: tuple[str, ...]


def compute_aggression_level(
    *,
    seconds_since_last_fill: Optional[float],
    trigger_seconds: float,
    ramp_seconds: float,
) -> float:
    """Pure function: map seconds-since-last-fill → aggression level
    in [0.0, 1.0].

    * ``seconds_since_last_fill is None`` → 0.0 (no fill data yet —
      session just started, or first fill hasn't arrived; treat as
      "we don't know how stale, so don't escalate").
    * Below ``trigger_seconds`` → 0.0
    * In ``[trigger, trigger + ramp]`` → linear ramp 0.0 → 1.0
    * Above ``trigger + ramp`` → 1.0 (saturated)

    Degenerate inputs (negative/zero ramp, negative trigger): clamp
    sanely (ramp ≤ 0 → step function at trigger; trigger < 0 →
    treated as 0).
    """
    if seconds_since_last_fill is None:
        return 0.0
    try:
        ssf = float(seconds_since_last_fill)
    except (TypeError, ValueError):
        return 0.0
    if ssf != ssf:  # NaN
        return 0.0
    trigger = max(0.0, float(trigger_seconds))
    ramp = float(ramp_seconds)
    if ssf < trigger:
        return 0.0
    if ramp <= 0.0:
        return 1.0  # Step function at the trigger.
    elapsed_past_trigger = ssf - trigger
    if elapsed_past_trigger >= ramp:
        return 1.0
    return float(elapsed_past_trigger) / float(ramp)


def compute_escalator_output(
    *,
    seconds_since_last_fill: Optional[float],
    enabled: bool,
    trigger_seconds: float,
    ramp_seconds: float,
    spread_compress_max_bps: float,
    microprice_attenuate: bool,
    toxicity_attenuate: bool,
    reservation_shift_mult_at_full: float,
) -> EscalatorOutput:
    """Compute the full per-tick escalator output.

    When ``enabled=False`` returns the no-op output (all multipliers
    at 1.0, zero compression). When enabled but level is 0.0 (still
    below trigger), also returns the no-op output.

    Multipliers interpolate linearly with the aggression level:
    * microprice / toxicity: ``mult = 1.0 - level`` when the
      corresponding ``*_attenuate`` flag is True, else 1.0.
    * reservation shift: ``mult = 1.0 + level * (mult_at_full - 1.0)``
      so at level=0 mult is 1.0 and at level=1 mult is the
      configured ``mult_at_full``. Use 1.0 to disable this knob.
    """
    if not enabled:
        return _noop_output()
    level = compute_aggression_level(
        seconds_since_last_fill=seconds_since_last_fill,
        trigger_seconds=trigger_seconds,
        ramp_seconds=ramp_seconds,
    )
    if level <= 0.0:
        return _noop_output()
    spread_compress = float(level) * float(max(0.0, spread_compress_max_bps))
    if microprice_attenuate:
        microprice_mult = max(0.0, 1.0 - level)
    else:
        microprice_mult = 1.0
    if toxicity_attenuate:
        toxicity_mult = max(0.0, 1.0 - level)
    else:
        toxicity_mult = 1.0
    # Linear interpolation: at level=0 → 1.0, at level=1 →
    # mult_at_full. Clamp non-negative.
    reservation_mult = max(
        0.0,
        1.0 + float(level) * (float(reservation_shift_mult_at_full) - 1.0),
    )
    # Build the active-dimensions list for diagnostics.
    active: list[str] = []
    if spread_compress > 1e-9:
        active.append(f"spread_compress:{spread_compress:.2f}bp")
    if microprice_attenuate and microprice_mult < 0.999:
        active.append(f"microprice_mult:{microprice_mult:.2f}")
    if toxicity_attenuate and toxicity_mult < 0.999:
        active.append(f"toxicity_mult:{toxicity_mult:.2f}")
    if abs(reservation_mult - 1.0) > 1e-3:
        active.append(f"reservation_mult:{reservation_mult:.2f}")
    return EscalatorOutput(
        aggression_level=float(level),
        spread_compression_bps=float(spread_compress),
        microprice_widen_multiplier=float(microprice_mult),
        toxicity_widen_multiplier=float(toxicity_mult),
        reservation_shift_multiplier=float(reservation_mult),
        active_dimensions=tuple(active),
    )


def _noop_output() -> EscalatorOutput:
    """The "escalator inactive" output — passes everything through
    unchanged."""
    return EscalatorOutput(
        aggression_level=0.0,
        spread_compression_bps=0.0,
        microprice_widen_multiplier=1.0,
        toxicity_widen_multiplier=1.0,
        reservation_shift_multiplier=1.0,
        active_dimensions=(),
    )
