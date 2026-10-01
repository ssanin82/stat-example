"""Regime controller FSM — Phase 1C (v1.4.112).

Central architecture piece from Codex Section 4. A live overlay
controller that owns the bot's current "regime mode" label and
publishes a small set of knob overlays consumed downstream by
``compute_quote_decision``.

The controller is **not a profile swap** — it doesn't reload the
operator's config. It's a tick-level state-machine that reads
already-computed signals (util, vol_ratio, gate-active flags) and
maps them into one of three modes:

* **NORMAL** — steady-state. All knobs at 1.0, ladder at full
  configured depth, both sides quoted, base half-spread unchanged.
* **DEFENSIVE** — bot is loaded (util ≥ 0.5) and/or a slow trend
  has been detected and/or the inventory-drift gate is firing
  and/or vol is elevated. Knob overlay halves size, widens base
  half-spread by 1.5×, biases inventory skew harder. Both sides
  still quoted but degraded.
* **SHOCK** — `shock_gate` is locked. The reducing side stays in
  the market (so the bot can flatten quickly); the adding side is
  off via the existing shock_gate eligibility override. This
  controller's role at SHOCK is mostly observational — the binary
  protection lives in `shock_gate`; the regime mode label exists
  so the operator (and downstream telemetry) see a coherent
  story.

Hysteresis (the "15 s entry / 30 s exit" rule from the plan):

* **NORMAL → DEFENSIVE**: an entry-triggering condition must hold
  continuously for ``entry_dwell_seconds`` (default 15 s). A brief
  blip out of conditions resets the arming timer.
* **DEFENSIVE → NORMAL**: ALL defensive triggers must clear AND
  util must drop below ``defensive_util_exit`` (default 0.30)
  continuously for ``exit_dwell_seconds`` (default 30 s). Exit
  dwell is intentionally longer than entry dwell because the cost
  of leaving DEFENSIVE too early (catching a continuation move)
  is materially higher than the cost of staying 20 s longer than
  needed.
* **NORMAL → SHOCK**: instant (no dwell). When shock_gate fires
  the controller jumps straight to SHOCK regardless of current
  mode.
* **DEFENSIVE → SHOCK**: same — instant.
* **SHOCK → DEFENSIVE**: when shock_gate clears its lock. There
  is **no direct SHOCK → NORMAL** transition; the controller must
  transit through DEFENSIVE so the exit-dwell guard applies.

Knob overlay (the "controller publishes; consumers read" contract):

The controller publishes a frozen ``RegimeKnobs`` instance every
tick. Each consumer reads only the fields it cares about:

|                            | NORMAL  | DEFENSIVE | SHOCK |
|----------------------------|---------|-----------|-------|
| ``quote_notional_mult``    | 1.00    | 0.50      | 1.00¹ |
| ``inventory_skew_mult``    | 1.00    | 1.50      | 1.00² |
| ``base_half_spread_mult``  | 1.00    | 1.50      | 2.00  |
| ``extra_tick_adding_side`` | 0       | +1        | 0²    |
| ``adding_side_enabled``    | True    | True      | False²|
| ``ladder_levels_max``      | (cfg N) | min(N, 1) | 0     |

¹ The plan table says SHOCK = 0.00 ("adding side off"). In code
the value is 1.00 because shock_gate's binary eligibility override
already turns the adding side off, and we don't want to also shrink
the **reducing** side — quick flatten is the priority. The 0.00 in
the plan documents the conceptual intent, not the runtime value.

² ``inventory_skew_mult`` / ``extra_tick_adding_side`` /
``adding_side_enabled`` / ``ladder_levels_max`` at SHOCK are
labelled "n/a" in the plan because shock_gate's binary clamp
dominates. We still publish the conservative DEFENSIVE-like values
for telemetry consistency; downstream consumers respect the
binary clamp first.

Per-tick lifecycle:

1. Caller (`bot.py`) computes inputs: ``util``, ``vol_ratio``,
   gate-active booleans.
2. Caller calls ``evaluate_mode(state, ...)``. Function mutates
   ``state.mode`` / ``state.mode_since_mono`` / arming timers /
   transition log / session counters.
3. Caller calls ``compute_knobs_for_mode(state.mode, ...)`` to
   get the ``RegimeKnobs`` for this tick.
4. Knobs flow into ``compute_quote_decision`` via new kwargs
   (``regime_base_half_spread_mult`` and
   ``regime_quote_notional_mult`` are the load-bearing two
   wired in this phase; other knobs are published for telemetry
   and consumed in subsequent phases).

Config (TON profile):

    REGIME_CONTROLLER_ENABLED=true
    REGIME_ENTRY_DWELL_SECONDS=15.0
    REGIME_EXIT_DWELL_SECONDS=30.0
    REGIME_UTIL_ENTRY_THRESHOLD=0.50
    REGIME_UTIL_EXIT_THRESHOLD=0.30
    REGIME_VOL_RATIO_ENTRY_THRESHOLD=2.0
"""

from __future__ import annotations

import enum
import math
from collections import deque
from dataclasses import dataclass, field, replace as _replace
from datetime import datetime, timedelta, timezone
from typing import Deque, Optional, Tuple

from app import clock as _clock


class Mode(enum.Enum):
    """Five-mode FSM. Order in this enum reflects increasing
    aggressiveness on the calm side and increasing defensiveness on
    the stress side — useful for sorted-comparison invariants in
    tests, though the FSM uses explicit ``is`` checks, not ordering.

    * ``CALM`` (Phase 4G) — forward-classifier-driven UPSHIFT.
      Tighter spreads, larger size, full ladder. Engaged only when
      every leading indicator stays in its benign zone for the
      classifier's min-history window AND the FSM's entry dwell.
    * ``NORMAL`` — default behaviour. Today's bot.
    * ``CAUTIOUS`` (Phase 4G) — forward-classifier-driven DOWNSHIFT.
      Widened spreads, smaller size, 1-level ladder, reduced
      inventory budget. Engaged on LEADING indicators (vol slope,
      drift magnitude rising, OB imbalance widening, basis stretch)
      BEFORE the reactive DEFENSIVE / SHOCK gates would fire.
    * ``DEFENSIVE`` — reactive-trigger downshift. Existing pre-4G
      behaviour (util ≥ entry threshold, or vol_ratio ≥ 2.0, or
      slow_trend, or inventory_drift). Acts as a SAFETY NET below
      the forward-signal layer.
    * ``SHOCK`` — reactive-trigger lock. shock_gate fired; one side
      forced off. Same as pre-4G; preempts all other modes.
    """

    CALM = "CALM"
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"
    DEFENSIVE = "DEFENSIVE"
    SHOCK = "SHOCK"


@dataclass(frozen=True)
class RegimeKnobs:
    """Frozen per-tick knob overlay published by the controller and
    consumed downstream by ``compute_quote_decision`` and (later)
    the ladder builder.

    All fields default to "no overlay" so callers can pass the
    default value and get current behaviour back."""

    quote_notional_mult: float = 1.0
    inventory_skew_mult: float = 1.0
    base_half_spread_mult: float = 1.0
    extra_tick_adding_side: int = 0
    adding_side_enabled: bool = True
    ladder_levels_max: Optional[int] = None
    """``None`` means "use the configured value"; an int caps the
    ladder at that many levels (per side)."""
    inventory_budget_mult: float = 1.0
    """Phase 4G (v1.4.209+) — multiplier on ``MAX_ABS_POSITION``
    consumers. ``1.0`` = no overlay (use the configured cap as-is).
    ``0.7`` (CAUTIOUS default) effectively shrinks the inventory
    budget so the bot won't grow positions during rising-risk
    regimes. Wired into the place-time risk checks in 4G.4."""


@dataclass
class RegimeControllerState:
    """Per-bot mutable state, owned by ``BotState``.

    Single-threaded ownership: the bot's main quote loop is the
    sole writer (``evaluate_mode`` mutates in place). The dashboard
    publisher reads from it lock-free under the GIL — every field
    is either a primitive or an immutable enum value, so a torn
    read produces only an inconsistent snapshot, never undefined
    behaviour."""

    mode: Mode = Mode.NORMAL
    mode_since_mono: float = 0.0
    last_transition_reason: str = ""

    # Hysteresis timers. ``None`` means "condition not currently
    # true". When the condition first becomes true the timestamp
    # is recorded; when it stays true across the dwell window the
    # mode transitions.
    #
    # ``entry_arming_since_mono`` + ``exit_arming_since_mono`` track
    # the reactive NORMAL ↔ DEFENSIVE transitions (pre-4G; unchanged).
    entry_arming_since_mono: Optional[float] = None
    exit_arming_since_mono: Optional[float] = None
    #
    # Phase 4G (v1.4.209+) — four additional arming timers for the
    # forward-classifier-driven CALM ↔ NORMAL ↔ CAUTIOUS transitions.
    # Each timer records the monotonic timestamp at which the
    # corresponding entry/exit condition first became true. Cleared
    # when the condition stops holding (resets the dwell).
    cautious_entry_arming_since_mono: Optional[float] = None
    cautious_exit_arming_since_mono: Optional[float] = None
    calm_entry_arming_since_mono: Optional[float] = None
    calm_exit_arming_since_mono: Optional[float] = None

    # Session-cumulative time-in-mode. Updated each tick by
    # ``accumulate_time_in_mode()``. Surfaced in session_summary
    # for postmortem.
    time_in_normal_seconds: float = 0.0
    time_in_defensive_seconds: float = 0.0
    time_in_shock_seconds: float = 0.0
    # Phase 4G (v1.4.209+) — two new mode counters.
    time_in_calm_seconds: float = 0.0
    time_in_cautious_seconds: float = 0.0

    # Full session log of mode transitions. Bounded loosely at 1024
    # entries as a memory safety net; in practice a session has
    # << 100 transitions because the 15 s entry / 30 s exit
    # hysteresis prevents flapping. Operator wants the COMPLETE
    # session log on the dashboard's Mode-transitions feed (Phase
    # 1E.3 section 7) — not just the most recent few. The hot path
    # is unaffected: appending to a deque is O(1) and the bot only
    # touches this on actual mode changes, never per-tick.
    last_transitions: Deque[Tuple[str, str, float, str]] = field(
        default_factory=lambda: deque(maxlen=1024)
    )
    """Each entry: ``(from_mode, to_mode, ts_mono, reason)``. Bounded
    at 1024 as a safety net against a pathological-flap state; under
    normal operation this never gets close."""

    transition_count: int = 0

    # Telemetry: timestamp of the last tick observed. Used by
    # ``accumulate_time_in_mode`` to compute per-tick deltas. Set
    # on first tick; subsequent ticks add ``now - last_tick``.
    _last_tick_mono: Optional[float] = None
    # Phase 4G.9 (v1.4.219) — mode active at the previous tick. When
    # a transition happened between ticks (``_last_tick_mode != state.mode``),
    # ``accumulate_time_in_mode`` splits the inter-tick delta at
    # ``mode_since_mono`` so pre-transition time credits the OLD mode
    # and post-transition time credits the NEW mode. Without this,
    # the entire delta got credited to whichever mode the bot was
    # in WHEN THE TICK FIRED — biasing the counters toward the post-
    # transition mode (snapshot v1.4.214 showed SHOCK reported 32.4 s
    # vs. reconciled 126.8 s; DEFENSIVE +60 s overcount).
    _last_tick_mode: Optional[Mode] = None


# ---------------------------------------------------------------------
# Mode evaluation
# ---------------------------------------------------------------------


def _entry_condition_met(
    *,
    util: float,
    vol_ratio: Optional[float],
    slow_trend_active: bool,
    inventory_drift_active: bool,
    util_entry_threshold: float,
    vol_ratio_entry_threshold: float,
) -> Tuple[bool, str]:
    """Return ``(condition_met, reason_text)`` — DEFENSIVE entry.

    Any one of the four triggers is sufficient. Reason text records
    which trigger(s) fired so the transition log is self-explanatory.
    """
    triggers: list[str] = []
    if util >= util_entry_threshold:
        triggers.append(f"util={util:.3f}")
    if (
        vol_ratio is not None
        and math.isfinite(vol_ratio)
        and vol_ratio >= vol_ratio_entry_threshold
    ):
        triggers.append(f"vol_ratio={vol_ratio:.2f}")
    if slow_trend_active:
        triggers.append("slow_trend")
    if inventory_drift_active:
        triggers.append("inv_drift")
    if not triggers:
        return False, ""
    return True, ",".join(triggers)


def _exit_condition_met(
    *,
    util: float,
    slow_trend_active: bool,
    inventory_drift_active: bool,
    vol_ratio: Optional[float],
    util_exit_threshold: float,
    vol_ratio_entry_threshold: float,
) -> bool:
    """DEFENSIVE → NORMAL exit condition: util below exit threshold
    AND ALL defensive triggers clear."""
    if util >= util_exit_threshold:
        return False
    if slow_trend_active or inventory_drift_active:
        return False
    if (
        vol_ratio is not None
        and math.isfinite(vol_ratio)
        and vol_ratio >= vol_ratio_entry_threshold
    ):
        return False
    return True


def _record_transition(
    state: RegimeControllerState,
    *,
    from_mode: Mode,
    to_mode: Mode,
    now_mono: float,
    reason: str,
) -> None:
    """Mutate state to reflect a mode transition. Resets ALL arming
    timers so the new mode's hysteresis logic starts fresh."""
    state.mode = to_mode
    state.mode_since_mono = float(now_mono)
    state.last_transition_reason = reason
    state.entry_arming_since_mono = None
    state.exit_arming_since_mono = None
    # Phase 4G — clear the four forward-signal arming timers too so
    # a fresh dwell starts on whichever transition fires next.
    state.cautious_entry_arming_since_mono = None
    state.cautious_exit_arming_since_mono = None
    state.calm_entry_arming_since_mono = None
    state.calm_exit_arming_since_mono = None
    state.last_transitions.append(
        (from_mode.value, to_mode.value, float(now_mono), reason)
    )
    state.transition_count = int(state.transition_count) + 1


def evaluate_mode(
    state: RegimeControllerState,
    *,
    now_mono: float,
    util: float,
    vol_ratio: Optional[float],
    slow_trend_active: bool,
    inventory_drift_active: bool,
    shock_gate_locked: bool,
    enabled: bool,
    entry_dwell_seconds: float = 15.0,
    exit_dwell_seconds: float = 30.0,
    util_entry_threshold: float = 0.50,
    util_exit_threshold: float = 0.30,
    vol_ratio_entry_threshold: float = 2.0,
    forward_regime: Optional[object] = None,
    forward_reason: Optional[str] = None,
    cautious_entry_dwell_seconds: float = 3.0,
    cautious_exit_dwell_seconds: float = 60.0,
    calm_entry_dwell_seconds: float = 120.0,
    calm_exit_dwell_seconds: float = 1.0,
) -> Tuple[Mode, Optional[str]]:
    """Apply the FSM transition rules to one quote-loop tick.

    Returns ``(new_mode, transition_reason_or_None)``.

    * ``transition_reason`` is non-None ONLY on the tick that
      actually crossed a transition boundary. Idle ticks (mode
      unchanged) return ``(state.mode, None)``.
    * Mutates ``state`` in place (mode, mode_since_mono, timers,
      transition log, transition_count).
    * Idempotent under steady inputs: calling repeatedly with the
      same inputs after a transition just keeps the same mode.

    Disabled-flag behaviour: when ``enabled=False`` the controller
    forces ``Mode.NORMAL`` and clears any arming timers. This lets
    the operator hard-disable the FSM via an env flag without
    leaving the bot stuck in a stale DEFENSIVE / SHOCK label.

    Phase 4G transition matrix (priority highest-to-lowest):

    1. ``shock_gate_locked`` → ``SHOCK`` from any mode (immediate).
    2. ``SHOCK`` → ``DEFENSIVE`` when shock_gate clears (immediate).
    3. ``DEFENSIVE`` → ``NORMAL`` via the existing dwell-hysteresis
       exit (unchanged from pre-4G).
    4. Any non-SHOCK, non-DEFENSIVE mode → ``DEFENSIVE`` via the
       existing reactive entry path (util / vol_ratio / slow_trend /
       inventory_drift). Acts as the safety net BELOW the forward-
       signal layer.
    5. ``CALM`` → ``CAUTIOUS`` direct (skip NORMAL) when the forward
       classifier says ``CAUTIOUS`` — the asymmetric "drop fast on
       any rising signal" rule from the operator spec.
    6. ``NORMAL`` / ``CAUTIOUS`` / ``CALM`` ↔ each other based on
       the forward classifier output + per-transition dwell timers.

    ``forward_regime`` is intentionally typed as ``Optional[object]``
    so the FSM module doesn't need to import
    ``regime_forward_signals`` (avoids a circular dependency).
    ``None`` means "no forward signal available this tick" — the FSM
    falls back to pre-4G behaviour (only NORMAL ↔ DEFENSIVE ↔ SHOCK
    transitions). The caller is responsible for producing the
    classification.
    """
    # Lazy import of ForwardRegime — keeps this module independent.
    if forward_regime is not None:
        from app.regime_forward_signals import ForwardRegime
        if not isinstance(forward_regime, ForwardRegime):
            # Caller passed something unexpected; treat as no signal.
            forward_regime = None
    else:
        ForwardRegime = None  # type: ignore[assignment]

    if not enabled:
        if state.mode is not Mode.NORMAL:
            _record_transition(
                state,
                from_mode=state.mode,
                to_mode=Mode.NORMAL,
                now_mono=now_mono,
                reason="disabled",
            )
            return Mode.NORMAL, "disabled"
        return Mode.NORMAL, None

    cur = state.mode

    # ------------------------------------------------------------------
    # SHOCK is a hard preempt — any mode jumps to SHOCK the moment
    # the shock_gate locks. No dwell, no hysteresis.
    # ------------------------------------------------------------------
    if shock_gate_locked and cur is not Mode.SHOCK:
        reason = "shock_gate_locked"
        _record_transition(
            state,
            from_mode=cur,
            to_mode=Mode.SHOCK,
            now_mono=now_mono,
            reason=reason,
        )
        return Mode.SHOCK, reason

    # ------------------------------------------------------------------
    # SHOCK → DEFENSIVE on lock-clear. No direct SHOCK → NORMAL — the
    # exit-dwell guard via DEFENSIVE has to apply.
    # ------------------------------------------------------------------
    if cur is Mode.SHOCK:
        if not shock_gate_locked:
            reason = "shock_gate_cleared"
            _record_transition(
                state,
                from_mode=cur,
                to_mode=Mode.DEFENSIVE,
                now_mono=now_mono,
                reason=reason,
            )
            return Mode.DEFENSIVE, reason
        return Mode.SHOCK, None

    # ------------------------------------------------------------------
    # DEFENSIVE → NORMAL exit path. Existing reactive logic; the
    # forward-signal layer doesn't see DEFENSIVE — that's the safety
    # net. We exit to NORMAL; the next tick's forward signal will
    # then arm CAUTIOUS or CALM if appropriate.
    # ------------------------------------------------------------------
    if cur is Mode.DEFENSIVE:
        exit_met = _exit_condition_met(
            util=util,
            slow_trend_active=slow_trend_active,
            inventory_drift_active=inventory_drift_active,
            vol_ratio=vol_ratio,
            util_exit_threshold=util_exit_threshold,
            vol_ratio_entry_threshold=vol_ratio_entry_threshold,
        )
        if not exit_met:
            state.exit_arming_since_mono = None
            return Mode.DEFENSIVE, None
        if state.exit_arming_since_mono is None:
            state.exit_arming_since_mono = float(now_mono)
            return Mode.DEFENSIVE, None
        elapsed = float(now_mono) - float(state.exit_arming_since_mono)
        if elapsed >= float(exit_dwell_seconds):
            reason = (
                f"defensive_exit:util={util:.3f},dwell={elapsed:.1f}s"
            )
            _record_transition(
                state,
                from_mode=cur,
                to_mode=Mode.NORMAL,
                now_mono=now_mono,
                reason=reason,
            )
            return Mode.NORMAL, reason
        return Mode.DEFENSIVE, None

    # ------------------------------------------------------------------
    # Reactive DEFENSIVE-entry trigger — checked BEFORE the forward-
    # signal-driven CAUTIOUS / CALM transitions. If util / vol_ratio /
    # slow_trend / inventory_drift indicate "real defensive needed",
    # that's the safety net firing and it preempts the forward layer.
    #
    # Source modes: NORMAL, CAUTIOUS, CALM (anything that's not
    # already in DEFENSIVE / SHOCK reaches here).
    # ------------------------------------------------------------------
    entry_met, entry_reason = _entry_condition_met(
        util=util,
        vol_ratio=vol_ratio,
        slow_trend_active=slow_trend_active,
        inventory_drift_active=inventory_drift_active,
        util_entry_threshold=util_entry_threshold,
        vol_ratio_entry_threshold=vol_ratio_entry_threshold,
    )
    if entry_met:
        if state.entry_arming_since_mono is None:
            state.entry_arming_since_mono = float(now_mono)
        else:
            elapsed = float(now_mono) - float(state.entry_arming_since_mono)
            if elapsed >= float(entry_dwell_seconds):
                reason = (
                    f"defensive_entry:{entry_reason},"
                    f"dwell={elapsed:.1f}s"
                )
                _record_transition(
                    state,
                    from_mode=cur,
                    to_mode=Mode.DEFENSIVE,
                    now_mono=now_mono,
                    reason=reason,
                )
                return Mode.DEFENSIVE, reason
        # Still arming reactive DEFENSIVE entry — don't process
        # forward-signal transitions this tick.
        return cur, None
    else:
        # Reactive entry condition clear — release the arming timer.
        state.entry_arming_since_mono = None

    # ------------------------------------------------------------------
    # Phase 4G forward-signal transitions (NORMAL ↔ CAUTIOUS ↔ CALM).
    # Only reached when:
    #   * Not in SHOCK / DEFENSIVE
    #   * Reactive DEFENSIVE entry not arming
    # ``forward_regime is None`` (caller didn't pass one) → no
    # transitions; stay in current mode. Backward-compat with pre-4G
    # callers.
    # ------------------------------------------------------------------
    if forward_regime is None:
        return cur, None

    # Helper: arm or fire a dwell-gated transition. Returns the new
    # mode + reason if the transition fired this tick; (None, None)
    # otherwise (caller stays in ``cur`` mode).
    def _maybe_fire_dwell_transition(
        *,
        condition_holds: bool,
        arming_attr: str,
        dwell_seconds: float,
        target_mode: Mode,
        reason_prefix: str,
        reason_detail: str,
    ) -> Tuple[Optional[Mode], Optional[str]]:
        if not condition_holds:
            setattr(state, arming_attr, None)
            return None, None
        if getattr(state, arming_attr) is None:
            setattr(state, arming_attr, float(now_mono))
            return None, None
        elapsed = float(now_mono) - float(getattr(state, arming_attr))
        if elapsed >= float(dwell_seconds):
            reason = (
                f"{reason_prefix}:{reason_detail},dwell={elapsed:.1f}s"
            )
            _record_transition(
                state,
                from_mode=cur,
                to_mode=target_mode,
                now_mono=now_mono,
                reason=reason,
            )
            return target_mode, reason
        return None, None

    # v1.5.231 — fold the classifier's reason text (which already
    # names the winning criterion, e.g. "vol_slope:1.20bp/min |
    # basis_stretch:x2.30") into the FSM-side reason_detail. Falls
    # back to bare "forward_classifier" when forward_reason wasn't
    # passed in (older callers / tests).
    _classifier_tag = (
        f"forward_classifier[{forward_reason}]"
        if forward_reason
        else "forward_classifier"
    )

    # --- NORMAL: can step UP to CAUTIOUS or DOWN to CALM ---
    if cur is Mode.NORMAL:
        # CAUTIOUS has priority — rising-risk signals win over calm.
        new_mode, reason = _maybe_fire_dwell_transition(
            condition_holds=(forward_regime is ForwardRegime.CAUTIOUS),
            arming_attr="cautious_entry_arming_since_mono",
            dwell_seconds=cautious_entry_dwell_seconds,
            target_mode=Mode.CAUTIOUS,
            reason_prefix="cautious_entry",
            reason_detail=_classifier_tag,
        )
        if new_mode is not None:
            return new_mode, reason
        # CALM entry path — separate arming timer, longer dwell.
        new_mode, reason = _maybe_fire_dwell_transition(
            condition_holds=(forward_regime is ForwardRegime.CALM),
            arming_attr="calm_entry_arming_since_mono",
            dwell_seconds=calm_entry_dwell_seconds,
            target_mode=Mode.CALM,
            reason_prefix="calm_entry",
            reason_detail=_classifier_tag,
        )
        if new_mode is not None:
            return new_mode, reason
        return Mode.NORMAL, None

    # --- CAUTIOUS: can step DOWN to NORMAL when forward classifier
    # says NORMAL or CALM. We always step to NORMAL first (not direct
    # to CALM) so the CALM entry dwell still applies on the next tick.
    if cur is Mode.CAUTIOUS:
        exits_to_normal = forward_regime in (
            ForwardRegime.NORMAL,
            ForwardRegime.CALM,
        )
        # v1.5.231 — include the classifier's reason on exit too so
        # the operator can see WHY CAUTIOUS released (which criterion
        # receded). When exits_to_normal is True the existing
        # forward=<NEW> tag is preserved; we just append the criterion
        # detail when the classifier provided one.
        _exit_detail: str
        if exits_to_normal:
            _exit_detail = f"forward={forward_regime.value}"
            if forward_reason:
                _exit_detail += f"[{forward_reason}]"
        else:
            _exit_detail = _classifier_tag
        new_mode, reason = _maybe_fire_dwell_transition(
            condition_holds=exits_to_normal,
            arming_attr="cautious_exit_arming_since_mono",
            dwell_seconds=cautious_exit_dwell_seconds,
            target_mode=Mode.NORMAL,
            reason_prefix="cautious_exit",
            reason_detail=_exit_detail,
        )
        if new_mode is not None:
            return new_mode, reason
        return Mode.CAUTIOUS, None

    # --- CALM: drop fast on any rising signal. CALM → CAUTIOUS is
    # direct (skip NORMAL) to give the operator the "drop immediately"
    # behaviour. CALM → NORMAL uses the short 1 s exit dwell.
    if cur is Mode.CALM:
        if forward_regime is ForwardRegime.CAUTIOUS:
            # Immediate jump — no dwell. CALM → CAUTIOUS short-circuit
            # is the asymmetric "fast out of CALM" the operator spec
            # called for.
            reason = f"calm_to_cautious_direct:{_classifier_tag}"
            _record_transition(
                state,
                from_mode=cur,
                to_mode=Mode.CAUTIOUS,
                now_mono=now_mono,
                reason=reason,
            )
            return Mode.CAUTIOUS, reason
        # CALM → NORMAL on confirmed-NORMAL classification.
        new_mode, reason = _maybe_fire_dwell_transition(
            condition_holds=(forward_regime is ForwardRegime.NORMAL),
            arming_attr="calm_exit_arming_since_mono",
            dwell_seconds=calm_exit_dwell_seconds,
            target_mode=Mode.NORMAL,
            reason_prefix="calm_exit",
            reason_detail=_classifier_tag,
        )
        if new_mode is not None:
            return new_mode, reason
        return Mode.CALM, None

    # Unreachable — defensive default for static analysis.
    return cur, None


# ---------------------------------------------------------------------
# Knob computation
# ---------------------------------------------------------------------


# Knob tables — single source of truth for the per-mode overlay values.
# Tweaking these values is the most likely future calibration knob;
# isolated here so the operator can grep for a single constant.
_KNOBS_NORMAL = RegimeKnobs(
    quote_notional_mult=1.00,
    inventory_skew_mult=1.00,
    base_half_spread_mult=1.00,
    extra_tick_adding_side=0,
    adding_side_enabled=True,
    ladder_levels_max=None,
)

_KNOBS_DEFENSIVE = RegimeKnobs(
    quote_notional_mult=0.50,
    inventory_skew_mult=1.50,
    base_half_spread_mult=1.50,
    extra_tick_adding_side=1,
    adding_side_enabled=True,
    ladder_levels_max=1,
    # v1.5.26 Phase 2E -- mode-aware position cap. DEFENSIVE shrinks
    # the inventory budget to 60% of MAX_ABS_POSITION so the bot
    # can't grow the position much during a stressed regime. The
    # cap is consulted at PLACE time (in the quote engine via
    # QuoteBuildContext.effective_max_abs_position) so existing
    # inventory ABOVE the new cap is allowed to reduce naturally;
    # no forced flatten on mode transition. (2E.3 invariant.)
    inventory_budget_mult=0.60,
)

# SHOCK row: quote_notional_mult stays at 1.00 to keep the reducing
# side full-size (quick flatten). The adding side is suppressed by
# shock_gate's binary eligibility override, not by this knob. The
# plan documents the conceptual "0.00" intent which is preserved by
# the shock_gate clamp.
_KNOBS_SHOCK = RegimeKnobs(
    quote_notional_mult=1.00,
    inventory_skew_mult=1.50,
    base_half_spread_mult=2.00,
    extra_tick_adding_side=0,
    adding_side_enabled=False,
    ladder_levels_max=0,
    # v1.5.26 Phase 2E -- SHOCK shrinks the inventory budget to 30%
    # of MAX_ABS_POSITION. Tighter than DEFENSIVE (60%) -- under SHOCK
    # the bot has already hit a hard signal (shock_gate fire) and we
    # want the WORST-CASE position size that can be opened to be
    # minimal. Adding-side is also clamped off via ``adding_side_enabled
    # =False`` so in practice this mostly bounds reducing-side re-
    # opens after intra-SF state changes. (2E.3: existing inventory
    # above the new cap reduces naturally; no forced flatten.)
    # Pre-2E this was 0.50 (4G.4 default at v1.4.210); 2E specifies
    # 0.30 in the plan-spec.
    inventory_budget_mult=0.30,
)

# Phase 4G (v1.4.209+) — two new mode rows.
#
# CALM: more-aggressive-than-NORMAL UPSHIFT. Engaged when every
# leading indicator stays in its benign zone for the classifier's
# min-history window AND the FSM's 120 s entry dwell. This is where
# the "good PnL when regime is calm" comes from — tighter spread
# means more fills + larger size per fill = more rebate income.
_KNOBS_CALM = RegimeKnobs(
    quote_notional_mult=1.40,   # 40 % larger size per quote
    inventory_skew_mult=1.00,   # same skew shape — calm doesn't change inventory pressure
    base_half_spread_mult=0.75, # tighter spread — more fills
    extra_tick_adding_side=0,
    adding_side_enabled=True,
    ladder_levels_max=None,     # use configured (typically 2)
    inventory_budget_mult=1.00, # full position budget
)

# CAUTIOUS: between NORMAL and DEFENSIVE. Engaged on RISING leading
# indicators BEFORE reactive DEFENSIVE / SHOCK triggers fire. The
# proactive-defense layer the operator asked for.
#
# v1.5.231 (2026-05-29): `quote_notional_mult` raised from 0.70 to
# 1.00. The 0.70 value was calibrated assuming QUOTE_NOTIONAL_USD ≥
# ~$10. On TON we run at $7. CAUTIOUS computed $7 × 0.70 = $4.90,
# which lands BELOW MIN_QUOTE_NOTIONAL_USD=$5 and causes every rung
# (both levels, both sides) to be dropped — `desired_none` cancels
# all live orders the instant CAUTIOUS engages. v1.5.230 snapshot
# (260529-094214) recorded 235 regime transitions in 27 min, fill
# rate 0.0 /min, all cancels with trigger `desired_none`. Setting
# the multiplier to 1.00 keeps CAUTIOUS defensive via the spread
# widening (`base_half_spread_mult=1.30`) and ladder-level cap
# (`ladder_levels_max=1`) WITHOUT killing the rung outright. Both
# remaining defenses are still material:
#   * 1.30× wider spread → fewer fills, the ones that come are at
#     better edge.
#   * Single rung → reduced exposure during the proactive-defense
#     window without a complete quote blackout.
_KNOBS_CAUTIOUS = RegimeKnobs(
    quote_notional_mult=1.00,   # v1.5.231: was 0.70 — see comment block above
    inventory_skew_mult=1.20,   # somewhat stronger skew
    base_half_spread_mult=1.30, # widened spreads (between NORMAL=1.0 and DEFENSIVE=1.5)
    extra_tick_adding_side=0,
    adding_side_enabled=True,
    ladder_levels_max=1,        # one rung only
    inventory_budget_mult=0.70, # 70 % of MAX_ABS_POSITION
)


def compute_knobs_for_mode(
    mode: Mode,
    *,
    ladder_levels_max_config: Optional[int] = None,
) -> RegimeKnobs:
    """Return the frozen knob overlay for the given mode.

    ``ladder_levels_max_config`` is the operator-configured
    ``LADDER_NUM_LEVELS_PER_SIDE``; the DEFENSIVE / SHOCK / CAUTIOUS
    knobs cap the ladder at ``min(config, knob_value)``; NORMAL and
    CALM leave it unmodified."""
    if mode is Mode.SHOCK:
        base = _KNOBS_SHOCK
    elif mode is Mode.DEFENSIVE:
        base = _KNOBS_DEFENSIVE
    elif mode is Mode.CAUTIOUS:
        base = _KNOBS_CAUTIOUS
    elif mode is Mode.CALM:
        base = _KNOBS_CALM
    else:
        base = _KNOBS_NORMAL
    if ladder_levels_max_config is None or base.ladder_levels_max is None:
        return base
    capped = min(int(ladder_levels_max_config), int(base.ladder_levels_max))
    return _replace(base, ladder_levels_max=capped)


# ---------------------------------------------------------------------
# Time-in-mode accumulation
# ---------------------------------------------------------------------


def accumulate_time_in_mode(
    state: RegimeControllerState, now_mono: float
) -> None:
    """Update the per-mode session-cumulative second counters.

    Called once per quote tick AFTER ``evaluate_mode``. The delta
    between calls is added to whichever counter matches the
    current mode. First call seeds the timestamp without adding
    anything (no prior tick to measure from).

    Phase 4G.9 (v1.4.219) — split the delta at ``state.mode_since_mono``
    when a transition happened between ticks (``_last_tick_mode !=
    state.mode``). Pre-transition time credits the OLD mode
    (``_last_tick_mode``); post-transition time credits the NEW mode
    (``state.mode``). Pre-fix: the entire delta credited the new
    mode, producing the snapshot v1.4.214 discrepancy where SHOCK
    reported 32.4 s vs. reconciled 126.8 s (the 95 s shortfall
    leaked into whichever mode succeeded SHOCK on the FSM walk).

    Limitation: when MULTIPLE transitions happened in one tick gap
    (rare but possible during a SHOCK→DEFENSIVE→NORMAL flurry), only
    the final hop's split is applied — intermediate modes that
    started AND ended within the gap don't get any credit. The
    transition log preserves the exact monotonic timestamps so
    post-session reconciliation can compute exact counters; this
    function is a live-tick approximation that's correct on the
    common case (single transition per gap).
    """
    last = state._last_tick_mono
    last_mode = state._last_tick_mode
    if last is None:
        state._last_tick_mono = float(now_mono)
        state._last_tick_mode = state.mode
        return
    delta = float(now_mono) - float(last)
    if delta < 0 or not math.isfinite(delta):
        # Clock jump / corrupted input — reset without adding.
        state._last_tick_mono = float(now_mono)
        state._last_tick_mode = state.mode
        return
    # Cap absurd deltas (process pause / suspended VM / long sleep)
    # at 600 s. Pre-v1.4.219 this was 60 s, which was too aggressive
    # — a single 90 s SHOCK tick gap dropped ~30 s of accounted time.
    # 600 s catches genuine process-pause poisoning while preserving
    # accuracy across slower-cadence ticks.
    delta = min(delta, 600.0)

    if last_mode is not None and last_mode is not state.mode:
        # A transition happened in the gap. Split the delta at
        # mode_since_mono: anything before goes to last_mode,
        # anything after goes to state.mode.
        #
        # Fallback: when mode_since_mono is unset (==0) OR falls
        # outside the [last, now] window, assume the transition
        # happened at ``last`` — i.e. all delta credits the NEW mode.
        # This matches the pre-v1.4.219 behaviour for callers that
        # mutate state.mode without setting mode_since_mono (test
        # fixtures, replay scaffolding). Production callers update
        # mode_since_mono inside ``_record_transition`` so the split
        # is always exact on the live path.
        if state.mode_since_mono <= 0.0:
            ms = float(last)
        else:
            ms = float(state.mode_since_mono)
        # Clamp the split point to [last, now] so a malformed
        # mode_since_mono (e.g. snapshot replay with monotonic
        # discontinuity) doesn't double-count.
        ms = max(float(last), min(float(now_mono), ms))
        pre_delta = max(0.0, ms - float(last))
        post_delta = max(0.0, float(now_mono) - ms)
        # Re-clamp against the global cap so the sum doesn't exceed
        # 600 s (preserves the pause-poisoning guard).
        if pre_delta + post_delta > delta + 1e-9:
            # Should be impossible given the clamp above, but
            # defensive — proportionally rescale.
            scale = delta / max(pre_delta + post_delta, 1e-9)
            pre_delta *= scale
            post_delta *= scale
        _add_to_mode_counter(state, last_mode, pre_delta)
        _add_to_mode_counter(state, state.mode, post_delta)
    else:
        # No transition or this is the first tick where last_mode
        # equals the current mode — simple per-mode accumulation.
        _add_to_mode_counter(state, state.mode, delta)

    state._last_tick_mono = float(now_mono)
    state._last_tick_mode = state.mode


def _add_to_mode_counter(
    state: RegimeControllerState, mode: Mode, seconds: float
) -> None:
    """Route a per-tick second delta into the matching mode counter.
    Single-dispatch helper for ``accumulate_time_in_mode`` so the
    transition-split path doesn't duplicate the if/elif cascade."""
    if seconds <= 0:
        return
    if mode is Mode.SHOCK:
        state.time_in_shock_seconds += seconds
    elif mode is Mode.DEFENSIVE:
        state.time_in_defensive_seconds += seconds
    elif mode is Mode.CAUTIOUS:
        state.time_in_cautious_seconds += seconds
    elif mode is Mode.CALM:
        state.time_in_calm_seconds += seconds
    else:
        state.time_in_normal_seconds += seconds


# ---------------------------------------------------------------------
# Telemetry / publisher payload
# ---------------------------------------------------------------------


def snapshot_dict(
    state: RegimeControllerState,
    now_mono: float,
    forward_reading: Optional[object] = None,
) -> dict[str, object]:
    """JSON-safe payload for ``state_current.json`` + live_stats
    + heartbeat. Renders unconditionally (per memory note
    ``feedback_bot_stats_panels_always_render``).

    Phase 4G.5 (v1.4.211) — ``forward_reading`` (a
    ``ForwardSignalReading`` or ``None``) is surfaced as a nested
    ``forward_signal`` block so the dashboard / Telegram / postmortem
    can show which leading indicator triggered the current CAUTIOUS
    classification (or which one is keeping the bot out of CALM).
    When ``None`` (forward layer disabled, default), the block
    renders a stable placeholder per the always-render contract."""
    if state.mode_since_mono > 0:
        seconds_in_mode = max(0.0, float(now_mono) - float(state.mode_since_mono))
    else:
        seconds_in_mode = 0.0
    # Phase 4G.7 (v1.4.217) — derive a wall-clock ISO timestamp for each
    # historical transition so the dashboard's regime band can plot them
    # on its wall-clock X-axis. Conversion: ``ts_wall = now_wall - (now_mono
    # - ts_mono)``. Snapshot once at the top of the function for a
    # consistent mono ↔ wall reference; mirrors the ``mode_since_iso``
    # derivation that live_stats.py does for the current mode.
    now_wall = _clock.now_utc()
    return {
        "mode": state.mode.value,
        "seconds_in_mode": round(seconds_in_mode, 2),
        "last_transition_reason": state.last_transition_reason or None,
        "entry_arming_seconds": (
            None
            if state.entry_arming_since_mono is None
            else round(
                max(0.0, float(now_mono) - float(state.entry_arming_since_mono)),
                2,
            )
        ),
        "exit_arming_seconds": (
            None
            if state.exit_arming_since_mono is None
            else round(
                max(0.0, float(now_mono) - float(state.exit_arming_since_mono)),
                2,
            )
        ),
        "transition_count": int(state.transition_count),
        # Recent transitions, newest-last. Each as a small dict so
        # the frontend doesn't have to re-parse positional tuples.
        # ``ts_iso`` added v1.4.217 — wall-clock ISO 8601 derived from
        # ``ts_mono`` + the snapshot's ``now_wall``. Frontend regime
        # band plots rectangles on the wall-clock X-axis using this.
        "recent_transitions": [
            {
                "from": frm,
                "to": to,
                "ts_mono": float(ts),
                "ts_iso": (
                    now_wall - timedelta(seconds=max(0.0, float(now_mono) - float(ts)))
                ).isoformat(),
                "reason": reason,
            }
            for (frm, to, ts, reason) in list(state.last_transitions)
        ],
        # Session-cumulative — also surfaced in session_summary.
        "time_in_normal_seconds": round(state.time_in_normal_seconds, 2),
        "time_in_defensive_seconds": round(state.time_in_defensive_seconds, 2),
        "time_in_shock_seconds": round(state.time_in_shock_seconds, 2),
        # Phase 4G — new modes.
        "time_in_calm_seconds": round(state.time_in_calm_seconds, 2),
        "time_in_cautious_seconds": round(state.time_in_cautious_seconds, 2),
        # Phase 4G arming-timer diagnostics. Surfaced so the operator
        # can see "we're 2 s into the 3 s CAUTIOUS-entry dwell" at a
        # glance. Each is None when the corresponding condition isn't
        # currently true.
        "cautious_entry_arming_seconds": _arming_age(
            state.cautious_entry_arming_since_mono, now_mono
        ),
        "cautious_exit_arming_seconds": _arming_age(
            state.cautious_exit_arming_since_mono, now_mono
        ),
        "calm_entry_arming_seconds": _arming_age(
            state.calm_entry_arming_since_mono, now_mono
        ),
        "calm_exit_arming_seconds": _arming_age(
            state.calm_exit_arming_since_mono, now_mono
        ),
        # Phase 4G.5 (v1.4.211) — forward classifier diagnostics.
        # Renders ALWAYS (per the always-render contract); when the
        # forward layer is disabled OR hasn't produced a reading yet,
        # the inner fields are None / placeholder values.
        "forward_signal": _forward_signal_block(forward_reading),
    }


def _forward_signal_block(forward_reading: Optional[object]) -> dict[str, object]:
    """Render the always-on forward_signal block. When
    ``forward_reading`` is None (forward layer disabled / no tick yet),
    returns placeholder values so the dashboard card structure stays
    stable per ``feedback_bot_stats_panels_always_render``.

    The accepted type is ``ForwardSignalReading`` from
    ``app/regime_forward_signals.py``; typed as ``object`` here to
    avoid an import-cycle at module load time."""
    if forward_reading is None:
        return {
            "classification": None,
            "reason": None,
            "vol_slope_bps_per_min": None,
            "drift_magnitude_30s_bps": None,
            "drift_magnitude_rising_ratio_observed": None,
            "ob_imbalance_widening_delta_observed": None,
            "basis_stretch_ratio_observed": None,
            "history_span_seconds": 0.0,
        }
    # Duck-typed extraction — avoids the import cycle.
    cls = getattr(forward_reading, "classification", None)
    cls_value = getattr(cls, "value", None) if cls is not None else None
    return {
        "classification": cls_value,
        "reason": getattr(forward_reading, "reason", None),
        "vol_slope_bps_per_min": _round_or_none(
            getattr(forward_reading, "vol_slope_bps_per_min", None)
        ),
        "drift_magnitude_30s_bps": _round_or_none(
            getattr(forward_reading, "drift_magnitude_30s_bps", None)
        ),
        "drift_magnitude_rising_ratio_observed": _round_or_none(
            getattr(forward_reading, "drift_magnitude_rising_ratio_observed", None)
        ),
        "ob_imbalance_widening_delta_observed": _round_or_none(
            getattr(forward_reading, "ob_imbalance_widening_delta_observed", None)
        ),
        "basis_stretch_ratio_observed": _round_or_none(
            getattr(forward_reading, "basis_stretch_ratio_observed", None)
        ),
        "history_span_seconds": round(
            float(getattr(forward_reading, "history_span_seconds", 0.0) or 0.0),
            2,
        ),
    }


def _round_or_none(v: Optional[float]) -> Optional[float]:
    """Helper for ``_forward_signal_block``. JSON output rounds to 4 dp
    so floats stay readable in the dashboard / postmortem."""
    if v is None:
        return None
    try:
        return round(float(v), 4)
    except Exception:
        return None


def _arming_age(
    arming_since_mono: Optional[float], now_mono: float
) -> Optional[float]:
    """Helper for ``snapshot_dict`` — converts a per-transition
    arming timestamp into elapsed seconds (or ``None`` when the
    timer isn't armed). Kept inline so the snapshot logic isn't
    duplicated four times."""
    if arming_since_mono is None:
        return None
    return round(max(0.0, float(now_mono) - float(arming_since_mono)), 2)
