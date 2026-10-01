"""Vol-spike runtime adapter (BUGS/todo-009.md).

Reads ``ToxicityEngine``'s already-computed ``vol_spike_ratio``
(current realised vol / baseline vol) and produces:

- a ``shrink_factor`` ∈ [VOL_SHRINK_FLOOR, 1.0] that NEW orders
  multiply their size + notional by, automatically reverting as
  vol calms;
- an ``in_spike_window`` flag, latched for
  ``VOL_SPIKE_COOLDOWN_SECONDS`` after the ratio crosses
  ``VOL_SPIKE_THRESHOLD`` — captures the post-spike
  bounce-and-retest pattern that bites makers on the second leg;
- a ``half_spread_bump_bps`` to lift the spread floor while inside
  the spike window.

Pure function (no side effects). The caller writes any updated
``vol_spike_until_mono`` back to ``BotState`` so the persistence is
preserved across ticks.

Default OFF: ``VOL_SHRINK_COEFF=0.0`` returns
``VolRegimeAdjustment(1.0, False, 0.0)`` — no behaviour change.
Operators opt in per profile after the calmer config tuning has
had time to settle.

Phase 2K.8 (v1.4.160) — favorable-exit predicate
------------------------------------------------

The ``VOL_SPIKE_COOLDOWN_SECONDS`` latch is the MAX-cooldown ceiling.
On top of it, ``evaluate_vol_spike_favorable_exit`` clears the latch
EARLY when ``vol_ratio < threshold × clear_band_mult`` (default 0.7×
threshold; with VOL_SPIKE_THRESHOLD=1.5 and mult=0.7 the clear band
is at vol_ratio < 1.05 — basically calm) and held there for
``favorable_exit_dwell_seconds`` (default 5 s — short because the
vol_ratio is a per-tick signal, not fill-driven).

Exit attribution: BotState's ``vol_spike_cleared_via_favorable_total``
vs ``vol_spike_cleared_via_ceiling_total`` lets the operator see
whether the predicate is doing meaningful work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import Settings

from app import clock as _clock


@dataclass(frozen=True)
class VolRegimeAdjustment:
    """Per-tick output of ``compute_vol_regime_adjustment``."""

    shrink_factor: float
    """Multiplier in ``[vol_shrink_floor, 1.0]`` for NEW order size +
    quote notional. ``1.0`` = no adjustment (feature off / calm)."""

    in_spike_window: bool
    """True when ``_clock.monotonic() < vol_spike_until_mono``. While
    True, ``shrink_factor`` is held at no higher than the spike-time
    level (no premature relax) and ``half_spread_bump_bps`` is
    nonzero."""

    half_spread_bump_bps: float
    """Additive bps to lift the half-spread floor while in spike
    window. Zero outside the window."""

    tier_name: str = "off"
    """Operator-readable categorical label for the dashboard's Market
    tab. One of:

    * ``"off"`` — feature disabled (``VOL_SHRINK_COEFF=0``); identity
      adjustment, no shrink, no bump.
    * ``"spike"`` — within the post-trigger cooldown window; spread
      bump active, sizes shrunk.
    * ``"elevated"`` — current vol > baseline (shrink factor below
      0.95) but no spike-window latch.
    * ``"normal"`` — vol roughly at baseline (0.95 ≤ shrink ≤ 1.0).
    * ``"calm"`` — vol meaningfully below baseline (vr ≤ 1.0 means
      the engine has nothing to shrink; identity result with
      coeff > 0).

    Frontend renders this as a coloured pill so the operator can see
    at a glance whether the vol-spike defense is active and at what
    intensity."""


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def compute_vol_regime_adjustment(
    settings: Settings,
    *,
    vol_ratio: Optional[float],
    now_mono: float,
    vol_spike_until_mono: float,
) -> tuple[VolRegimeAdjustment, float]:
    """Return ``(adjustment, new_vol_spike_until_mono)``.

    Caller responsibilities:
    - Pass ``vol_ratio`` from the latest ``state.toxicity.vol_spike_ratio``;
      ``None`` (or ``< 1.0``) is treated as "no information / calm".
    - Pass the previous ``state.vol_spike_until_mono`` as the third arg.
    - Write the returned ``new_vol_spike_until_mono`` back to state.
    - Apply ``shrink_factor`` to NEW order sizes (NOT to existing
      position checks — see BUGS/todo-009.md "Risk: shrinking caps
      mid-trade").
    - Apply ``half_spread_bump_bps`` to ``eff_min_half_spread`` BEFORE
      the ``max_half_spread_bps`` clamp.
    """
    coeff = float(settings.vol_shrink_coeff)
    floor = float(settings.vol_shrink_floor)
    threshold = float(settings.vol_spike_threshold)
    cooldown_s = float(settings.vol_spike_cooldown_seconds)
    bump_bps = float(settings.vol_spike_half_spread_bump_bps)

    # Feature off: identity result. Don't even update the spike
    # timer — that's keyed off ``vol_shrink_coeff > 0`` so the
    # persistence semantics live and die together with the sizing
    # adjustment.
    if coeff <= 0.0:
        return (
            VolRegimeAdjustment(
                shrink_factor=1.0,
                in_spike_window=False,
                half_spread_bump_bps=0.0,
                tier_name="off",
            ),
            0.0,
        )

    # Normalize vol_ratio. ``< 1.0`` (calmer than baseline) is
    # treated as 1.0 — we never grow caps, only shrink them.
    vr = 1.0
    if vol_ratio is not None:
        try:
            v = float(vol_ratio)
        except (TypeError, ValueError):
            v = 1.0
        if v == v and v > 1.0:  # NaN guard + only positive deltas
            vr = v

    # Update persistence window if vol_ratio crossed threshold.
    new_until = float(vol_spike_until_mono)
    if cooldown_s > 0.0 and vr >= threshold:
        new_until = max(new_until, float(now_mono) + cooldown_s)

    in_spike = float(now_mono) < new_until

    # Current-vol-driven shrink factor. Smooth (not stepped) — every
    # extra unit of ``vr`` over 1.0 contributes ``-coeff`` to the
    # shrink, clipped at the configured floor.
    raw_shrink = 1.0 - coeff * (vr - 1.0)
    shrink = _clip(raw_shrink, floor, 1.0)

    # While in spike window, lift the spread floor by the configured
    # bump. (The shrink factor stays at its current vol-driven level —
    # we don't artificially hold the spike-time level; the cooldown's
    # job is to keep the spread defense up, not to memorize a sizing
    # snapshot. Operators who want stricter post-spike sizing can set
    # a higher ``vol_shrink_floor``.)
    bump = bump_bps if in_spike else 0.0

    # Operator-readable tier label. Order matters: spike beats
    # elevated (the spike latch holds even after vol calms back to
    # normal); elevated beats normal (shrink_factor < 0.95 means we
    # ARE shrinking sizes); calm flags below-baseline vol so the
    # dashboard shows a quiet-market state instead of "normal".
    if in_spike:
        tier = "spike"
    elif shrink < 0.95:
        tier = "elevated"
    elif vol_ratio is not None and vol_ratio < 0.9:
        tier = "calm"
    else:
        tier = "normal"

    return (
        VolRegimeAdjustment(
            shrink_factor=shrink,
            in_spike_window=in_spike,
            half_spread_bump_bps=bump,
            tier_name=tier,
        ),
        new_until,
    )


# ---------------------------------------------------------------------------
# Phase 2K.8 — favorable-exit predicate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VolSpikeExitResult:
    """Result of evaluating the Phase 2K.8 favorable-exit predicate
    for a single tick. Caller round-trips ``new_*`` fields back to
    ``BotState`` so the dwell timer carries across ticks."""

    new_until_mono: float
    """Possibly-updated spike deadline. Same as the input
    ``vol_spike_until_mono`` unless the favorable-exit predicate
    cleared the latch — in which case this is ``0.0`` so the next
    ``compute_vol_regime_adjustment`` call sees no spike active."""

    new_favorable_dwell_started_mono: Optional[float]
    """When non-None, the favorable-exit dwell is in progress.
    Reset on re-arm (vol re-spikes past threshold), on re-flare
    (predicate flips back to not-holding), and on successful clear."""

    new_was_active_last_call: bool
    """Set to ``True`` while the spike latch is active so the next
    call can detect the active→cleared edge for ceiling attribution."""

    cleared_via: str
    """One of ``"none"`` (no transition), ``"favorable"`` (predicate
    fired this call), ``"ceiling"`` (timer expired naturally). Caller
    increments the matching ``BotState`` counter."""


def evaluate_vol_spike_favorable_exit(
    settings: Settings,
    *,
    now_mono: float,
    vol_ratio: Optional[float],
    vol_spike_until_mono: float,
    favorable_dwell_started_mono: Optional[float],
    was_active_last_call: bool,
) -> VolSpikeExitResult:
    """Evaluate the Phase 2K.8 favorable-exit predicate.

    Call this AFTER ``compute_vol_regime_adjustment`` with the
    returned ``new_vol_spike_until_mono``. Honours the
    ``VOL_SPIKE_FAVORABLE_EXIT_ENABLED`` setting — when False, returns
    the input state unchanged (legacy pure-timer behaviour) but still
    tracks the active→cleared edge for ceiling attribution.

    Semantics:

    * **Not in spike**: nothing to clear. Counters track the edge if
      the timer expired between calls.
    * **In spike, predicate not enabled**: ceiling-only; track edge.
    * **In spike, predicate enabled**:
      * If ``vol_ratio < threshold × clear_band_mult``: start/continue
        dwell. When the dwell duration reaches
        ``favorable_exit_dwell_seconds``, clear the latch
        (``new_until_mono = 0``) and report ``cleared_via="favorable"``.
      * Else: reset the dwell (re-flare hysteresis).

    Pure function: no side effects on ``settings``. Caller manages
    state persistence on ``BotState``.
    """
    enabled = bool(
        getattr(settings, "vol_spike_favorable_exit_enabled", True)
    )
    clear_band_mult = float(
        getattr(settings, "vol_spike_clear_band_mult", 0.7)
    )
    dwell_seconds = float(
        getattr(settings, "vol_spike_favorable_exit_dwell_seconds", 5.0)
    )
    threshold = float(settings.vol_spike_threshold)

    in_spike = float(now_mono) < float(vol_spike_until_mono)

    # Default: no transition.
    new_until = float(vol_spike_until_mono)
    new_dwell = favorable_dwell_started_mono
    cleared_via = "none"

    # Favorable-exit clearing (only meaningful while in spike).
    if in_spike and enabled and threshold > 0.0:
        vr = 1.0
        if vol_ratio is not None:
            try:
                v = float(vol_ratio)
                if v == v:  # NaN guard
                    vr = max(0.0, v)
            except (TypeError, ValueError):
                vr = 1.0
        clear_band = threshold * clear_band_mult
        # threshold=1.5, mult=0.7 → clear_band=1.05. Predicate holds
        # when vr < 1.05 (vol has calmed back near baseline).
        if vr < clear_band:
            if new_dwell is None:
                new_dwell = float(now_mono)
            elif now_mono - new_dwell >= dwell_seconds:
                new_until = 0.0
                cleared_via = "favorable"
                new_dwell = None
        else:
            # Re-flare: predicate not holding, reset dwell.
            new_dwell = None
    elif not in_spike:
        # Outside the spike window the dwell is meaningless.
        new_dwell = None

    # Compute the post-predicate active flag for edge detection.
    new_active = float(now_mono) < new_until

    # Ceiling attribution: timer expired naturally between calls.
    # If favorable already cleared this call we returned above with
    # ``cleared_via=favorable`` and ``new_active=False``; in that case
    # we must NOT also bump ceiling. The check below catches the case
    # where ``was_active_last_call`` was True, the latch is now False,
    # AND no favorable clearing happened this call.
    if (
        cleared_via == "none"
        and was_active_last_call
        and not new_active
    ):
        cleared_via = "ceiling"

    return VolSpikeExitResult(
        new_until_mono=new_until,
        new_favorable_dwell_started_mono=new_dwell,
        new_was_active_last_call=new_active,
        cleared_via=cleared_via,
    )
