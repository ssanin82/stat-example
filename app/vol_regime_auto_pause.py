"""Phase 4F (v1.4.170) — Elevated-vol auto-pause.

When the trading venue's realised-vol ratio (current / baseline)
stays elevated above ``VOL_AUTO_PAUSE_ARM_RATIO`` for
``VOL_AUTO_PAUSE_ARM_DWELL_SECONDS``, force the whole bot's
eligibility to ``HOLD_ALL``. Self-clears when the ratio drops back
below ``VOL_AUTO_PAUSE_CLEAR_RATIO`` for
``VOL_AUTO_PAUSE_CLEAR_DWELL_SECONDS``, OR when
``VOL_AUTO_PAUSE_MAX_SECONDS`` passes (safety ceiling so a stale or
misconfigured signal can't pause forever).

Existing defenses (regime_controller, vol_spike, adaptive_widen,
shock_gate, 4C side-refusal, 4E fast-move cancel) keep the bot
trading while widening / shrinking / suppressing pieces of the
quote stack. 4F is the MACRO defense: when the regime ITSELF is
bad enough that quoting is net-negative-edge, sit out entirely.

Composition with 4C: 4C refuses a SIDE per tick when its expected
edge is < threshold. 4F refuses BOTH SIDES sustained when the
regime's vol ratio crosses a higher bar with dwell. Independent
triggers, can fire together or separately.

Naming. ``vol_auto_pause_*`` is exchange-agnostic per the v1.4.165
naming guidance — the gate consumes the toxicity engine's
``vol_spike_ratio`` which is itself derived from the target venue's
own realised-vol kinematic. No exchange-specific names appear in
this module.

Default DISABLED via ``arm_ratio = 0.0``. Opt-in per profile.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VolRegimeAutoPauseDecision:
    """Output of one per-tick evaluation. Caller round-trips the
    ``new_*`` fields back onto BotState so the dwell timers carry
    across ticks."""

    new_active: bool
    """Whether the pause is ARMED after this tick. ``True`` →
    ``OrderManager`` forces eligibility to ``HOLD_ALL`` this tick."""

    new_arm_dwell_started_mono: Optional[float]
    """When non-None, the bot is currently DWELLING toward arming
    (vol_ratio > arm_ratio but not yet for long enough). Reset on
    re-flare (predicate flips back to not-holding) and on successful
    arm."""

    new_clear_dwell_started_mono: Optional[float]
    """When non-None, the bot is currently DWELLING toward clearing
    (vol_ratio < clear_ratio but not yet for long enough). Reset on
    re-flare and on successful clear."""

    new_active_since_mono: float
    """Monotonic timestamp at which the current pause was armed.
    ``0.0`` when not active. Used by the MAX-ceiling check to bound
    the pause duration."""

    transition: str
    """One of ``"none"`` (no state change), ``"armed"`` (became
    active this tick), ``"cleared_favorable"`` (cleared via the
    vol-recovered predicate), ``"cleared_ceiling"`` (cleared via
    MAX-pause timeout). Caller increments the matching counter."""


def evaluate_vol_regime_auto_pause(
    *,
    now_mono: float,
    vol_spike_ratio: Optional[float],
    currently_active: bool,
    arm_dwell_started_mono: Optional[float],
    clear_dwell_started_mono: Optional[float],
    active_since_mono: float,
    arm_ratio: float,
    arm_dwell_seconds: float,
    clear_ratio: float,
    clear_dwell_seconds: float,
    max_pause_seconds: float,
) -> VolRegimeAutoPauseDecision:
    """Pure-function evaluator.

    Logic order:

    * Feature disabled (``arm_ratio <= 0``) → identity result, no
      state change. Preserves pre-v1.4.170 behaviour.
    * Missing / non-finite ``vol_spike_ratio`` (e.g., toxicity engine
      warmup) → identity result.
    * **When NOT active:**
        - If ratio > arm_ratio → start/continue arm dwell. Arm if
          ``now - dwell_started >= arm_dwell_seconds``.
        - Else → reset arm dwell.
    * **When ACTIVE:**
        - If ratio < clear_ratio → start/continue clear dwell. Clear
          (favorable) if ``now - dwell_started >= clear_dwell_seconds``.
        - Else → reset clear dwell.
        - MAX-ceiling: if ``now - active_since >= max_pause_seconds``
          → clear (ceiling). Fires regardless of vol — safety net.

    The favorable-clear check runs BEFORE the ceiling check this
    tick so a clean recovery doesn't get mis-attributed when both
    fire on the same call. The attribution counter receives whichever
    transition happened first.
    """
    # Feature disabled — passthrough.
    if arm_ratio <= 0.0:
        return VolRegimeAutoPauseDecision(
            new_active=currently_active,
            new_arm_dwell_started_mono=arm_dwell_started_mono,
            new_clear_dwell_started_mono=clear_dwell_started_mono,
            new_active_since_mono=active_since_mono,
            transition="none",
        )

    # Missing / NaN signal — passthrough (treat as warmup; no state
    # change).
    if vol_spike_ratio is None:
        return VolRegimeAutoPauseDecision(
            new_active=currently_active,
            new_arm_dwell_started_mono=arm_dwell_started_mono,
            new_clear_dwell_started_mono=clear_dwell_started_mono,
            new_active_since_mono=active_since_mono,
            transition="none",
        )
    try:
        vr = float(vol_spike_ratio)
    except (TypeError, ValueError):
        return VolRegimeAutoPauseDecision(
            new_active=currently_active,
            new_arm_dwell_started_mono=arm_dwell_started_mono,
            new_clear_dwell_started_mono=clear_dwell_started_mono,
            new_active_since_mono=active_since_mono,
            transition="none",
        )
    if not math.isfinite(vr):
        return VolRegimeAutoPauseDecision(
            new_active=currently_active,
            new_arm_dwell_started_mono=arm_dwell_started_mono,
            new_clear_dwell_started_mono=clear_dwell_started_mono,
            new_active_since_mono=active_since_mono,
            transition="none",
        )

    if not currently_active:
        # Not paused. Check arm condition.
        if vr > arm_ratio:
            new_arm_dwell = (
                arm_dwell_started_mono
                if arm_dwell_started_mono is not None
                else float(now_mono)
            )
            if (
                float(now_mono) - float(new_arm_dwell)
                >= float(arm_dwell_seconds)
            ):
                return VolRegimeAutoPauseDecision(
                    new_active=True,
                    new_arm_dwell_started_mono=None,
                    new_clear_dwell_started_mono=None,
                    new_active_since_mono=float(now_mono),
                    transition="armed",
                )
            return VolRegimeAutoPauseDecision(
                new_active=False,
                new_arm_dwell_started_mono=new_arm_dwell,
                new_clear_dwell_started_mono=None,
                new_active_since_mono=0.0,
                transition="none",
            )
        # Below arm threshold — reset arm dwell.
        return VolRegimeAutoPauseDecision(
            new_active=False,
            new_arm_dwell_started_mono=None,
            new_clear_dwell_started_mono=None,
            new_active_since_mono=0.0,
            transition="none",
        )

    # Currently active. Try favorable-clear first; then MAX-ceiling.
    if vr < clear_ratio:
        new_clear_dwell = (
            clear_dwell_started_mono
            if clear_dwell_started_mono is not None
            else float(now_mono)
        )
        if (
            float(now_mono) - float(new_clear_dwell)
            >= float(clear_dwell_seconds)
        ):
            return VolRegimeAutoPauseDecision(
                new_active=False,
                new_arm_dwell_started_mono=None,
                new_clear_dwell_started_mono=None,
                new_active_since_mono=0.0,
                transition="cleared_favorable",
            )
        # Still dwelling toward clear; check ceiling alongside.
        if (
            float(now_mono) - float(active_since_mono)
            >= float(max_pause_seconds)
        ):
            return VolRegimeAutoPauseDecision(
                new_active=False,
                new_arm_dwell_started_mono=None,
                new_clear_dwell_started_mono=None,
                new_active_since_mono=0.0,
                transition="cleared_ceiling",
            )
        return VolRegimeAutoPauseDecision(
            new_active=True,
            new_arm_dwell_started_mono=None,
            new_clear_dwell_started_mono=new_clear_dwell,
            new_active_since_mono=active_since_mono,
            transition="none",
        )
    # Vol still elevated — reset clear dwell; check ceiling.
    if (
        float(now_mono) - float(active_since_mono)
        >= float(max_pause_seconds)
    ):
        return VolRegimeAutoPauseDecision(
            new_active=False,
            new_arm_dwell_started_mono=None,
            new_clear_dwell_started_mono=None,
            new_active_since_mono=0.0,
            transition="cleared_ceiling",
        )
    return VolRegimeAutoPauseDecision(
        new_active=True,
        new_arm_dwell_started_mono=None,
        new_clear_dwell_started_mono=None,
        new_active_since_mono=active_since_mono,
        transition="none",
    )
