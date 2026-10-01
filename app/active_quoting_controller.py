"""v1.5.277 / AQC Phase 1 — Active Quoting Controller (observe-only).

Goal
====

Replace the operator's days-long parameter-by-parameter tuning loop
with a single PI controller that targets **net edge per minute**
(rebate income + markout impact) and outputs a continuous
``aggression_level`` in ``[0, 1]``.

Phase 1 (this module): the controller computes its state and
exposes diagnostics via ``snapshot_dict()``. **It does NOT drive
any quote-engine behavior yet.** Pure observability. Operator can
A/B the controller's would-be behavior against actual behavior on
the same snapshots before any code path consumes
``aggression_level``.

Phase 2 (separate ship): wire ``aggression_level`` to a single
effective multiplier (e.g. ``min_half_spread_bps`` floor) gated
behind a new env flag. A/B against the unmodified path.

Phase 3+ (later ships): expand to the full multiplier set
documented in ``plans/active_quoting_controller.md`` (when
written) — inventory_skew, inventory_exec_bias util floor,
inventory_drift thresholds, vol_climbing_widen arm gate.

Control law
===========

Standard PI controller with **conditional-integration** anti-windup
(v1.5.280) + a markout safety floor:

```
error = target_net_edge_per_min_usd - observed_net_edge_per_min_usd

# Conditional integration: only commit the integrator step if it would
# NOT push an already-saturated output deeper into saturation. This
# freezes the integrator at the edge where aggression clips (≈ 1/K_i),
# instead of letting it wind on uselessly toward the hard clip bound.
candidate   = clip(integrator + error × dt, [I_min, I_max])
raw_cand    = K_p × error + K_i × candidate
sat_high    = raw_cand > 1 and error > 0   # output pinned at 1, still pushing up
sat_low     = raw_cand < 0 and error < 0   # output pinned at 0, still pushing down
if not (sat_high or sat_low):
    integrator = candidate                 # else: hold — no windup

aggression = clip(K_p × error + K_i × integrator, [0, 1])

# Safety floor — overrides PI when realized markout is collapsing.
# v1.5.290: the floor compares the window MEDIAN (outlier-robust) and
# requires >= markout_floor_min_fills distinct fills. Consulted ONLY
# when both hold; when the median is absent or the window has too few
# fills the brake is skipped and the PI runs on net-edge alone
# (v1.5.279 decouple). The MEAN is no longer a floor input — a lone
# flash fill can no longer pin aggression for the whole window (the
# v1.5.289 incident).
if (median_bps < markout_floor_bps) and (sample_count >= min_fills):
    aggression = 0.0  # immediate revert to defensive
    integrator = 0.0  # reset; prevents fast snap-back when markout recovers
```

Why this target
---------------

* Pure fill-rate target drives toward adverse selection (bot
  quotes inside touch always, gets picked off).
* Pure markout target drives the bot to stop quoting entirely
  (the safest markout is no fill).
* **Net edge** = rebate income (positive, scales with fills) +
  markout impact (negative when adverse, positive when favorable).
  Maximizing net edge naturally balances "more fills" against
  "fewer toxic fills." Operator sets one target in cents/minute;
  the controller finds the regime-appropriate quote aggression.

Why PI not P-only
-----------------

* P-only would steady-state error: the bot would always run
  slightly below target because the controller only responds to
  instantaneous error.
* PI's integrator accumulates error over time and drives the
  steady-state error to zero.
* Anti-windup: with ``K_i = 0.05`` the I-term alone saturates
  aggression at ``integrator = 20`` (``K_i × 20 = 1.0``). A
  clip-only design would let the integrator keep winding to the
  ±300 hard bound — 15× past saturation — in a sustained
  below-target regime (flat market, no liquidity, dead book). When
  conditions recover, the bot stays pinned at aggression = 1.0
  while the over-wound integrator slowly unwinds (≈ 4 h at
  TON's error magnitudes). **Conditional integration** fixes this:
  the integrator step is committed only when it would NOT deepen an
  existing output saturation, so the integrator parks at ≈ 1/K_i
  and back-off begins the instant the error sign flips. The ±300
  clip is retained purely as a defensive backstop.

Why the markout safety floor
----------------------------

In a strong trend, the bot's quotes get picked off as the market
walks past them. Net edge collapses. The PI would respond by
pushing aggression UP (more aggressive → tries to maintain
target). That's exactly wrong — strong trends require LESS
aggression, not more. The markout safety floor short-circuits the
PI when realized markout falls below ``markout_floor_bps``,
forcing immediate revert to defensive baseline.

Threading
=========

Same model as ``VolAbsEwmaEstimator`` and ``OFIAccumulator``:
updated by the quote-loop thread on each tick (one call per
500 ms), read by the same thread for snapshot publication. Pure
float reads; GIL provides atomicity.

State
=====

The controller is a small dataclass holding:

* ``aggression_level`` — current output, in ``[0, 1]``
* ``integrator`` — accumulated error
* ``last_update_mono_seconds`` — for dt computation
* ``last_observed_net_edge_per_min_usd`` — diagnostic
* ``last_observed_markout_5s_median_bps`` — the floor-driving
  statistic (v1.5.290)
* ``last_observed_markout_sample_count`` — distinct fills in the
  window; gates the min-fill requirement (v1.5.290)
* ``last_observed_markout_5s_mean_bps`` — diagnostic only; NOT a
  floor input (kept to expose mean-vs-median divergence)
* ``safety_floor_engaged`` — bool, true when markout floor is
  forcing aggression = 0
* ``update_count`` / ``safety_floor_engagement_count`` — telemetry

The caller supplies the observed inputs via ``update(...)``. The
controller does NOT reach back into ``BotState`` directly —
keeps the module self-contained and testable.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


# Integrator bounds. Anti-windup clip range — wide enough that the
# integrator captures multi-minute accumulated error, narrow enough
# that recovery from boundary saturation doesn't lag by hours.
_INTEGRATOR_MIN: float = -300.0
_INTEGRATOR_MAX: float = 300.0


@dataclass(slots=True)
class AQCSettings:
    """Operator-facing tuning knobs for the AQC.

    The 5 knobs here replace the ~25 manually-tuned knobs scattered
    across the bot's defensive gates. Default values are conservative
    Phase-1 placeholders; the real calibration happens in Phase 4 of
    the AQC plan after replay-testing.
    """

    enabled: bool = False
    target_net_edge_per_min_usd: float = 0.020
    markout_floor_bps: float = -5.0
    # v1.5.290 — outlier-robust safety floor. The floor compares the
    # window MEDIAN markout (not the mean) to ``markout_floor_bps`` and
    # only engages with at least this many distinct fills in the window.
    # RATIONALE (v1.5.289 incident): a single flash fill (-63.6 bps)
    # dragged the 300 s MEAN to -12.3 < -5 and pinned aggression at 0 for
    # the whole window, though the MEDIAN was -2.8 (above the floor). The
    # median + min-fill gate make a lone outlier unable to trip the brake.
    markout_floor_min_fills: int = 5
    pi_kp: float = 0.4
    pi_ki: float = 0.05
    # Phase 2 (v1.5.281) — output wiring. ``wire_min_half_spread``
    # gates whether ``aggression_level`` modulates the economic
    # min-half-spread floor (default off → observe-only). At full
    # aggression the EFFECTIVE base econ floor is lerped down toward
    # ``min_half_spread_floor_at_full_aggression_bps`` (tighten-only).
    # The lerp lives in ``compute_effective_min_half_spread_bps`` so
    # the controller stays a pure signal source with no spread-pipeline
    # coupling. See plans/aqc-execute.md Phase 2.
    wire_min_half_spread: bool = False
    min_half_spread_floor_at_full_aggression_bps: float = 2.0
    # Phase 3 (v1.5.282) — output wiring. ``wire_inventory`` gates
    # whether ``aggression_level`` modulates the inventory exec-bias
    # util floor (default off → observe-only). At full aggression the
    # EFFECTIVE util floor is lerped DOWN toward
    # ``inventory_util_floor_at_full_aggression_pct`` (tighten-only:
    # exec-bias engages earlier → faster inventory churn). The lerp
    # lives in ``compute_effective_inventory_util_floor_pct`` so the
    # controller stays a pure signal source. See plans/aqc-execute.md
    # Phase 3.
    wire_inventory: bool = False
    inventory_util_floor_at_full_aggression_pct: float = 0.10
    # Phase 4 (v1.5.283) — output wiring. ``wire_skew`` gates whether
    # ``aggression_level`` modulates the inventory skew coefficient
    # (default off → observe-only). At full aggression the EFFECTIVE
    # skew coefficient is lerped UP toward
    # ``skew_coeff_at_full_aggression_bps`` (increase-only: the
    # reducing-side quote pulls closer to touch → faster, constructive
    # inventory flattening). The lerp lives in
    # ``compute_effective_inventory_skew_coeff_bps`` so the controller
    # stays a pure signal source. The plan's companion vol-climbing
    # DISABLE lever is intentionally NOT wired (Rule 0c — loosening a
    # defensive gate). See plans/aqc-execute.md Phase 4.
    wire_skew: bool = False
    skew_coeff_at_full_aggression_bps: float = 40.0


@dataclass(slots=True)
class ActiveQuotingController:
    """Per-symbol PI controller for quote aggression.

    Phase 1: observe-only. The ``aggression_level`` output is
    published in ``snapshot_dict()`` but no caller consumes it for
    quote-engine behavior changes.
    """

    settings: AQCSettings = field(default_factory=AQCSettings)

    # PI state.
    aggression_level: float = 0.0
    integrator: float = 0.0

    # Last-observation cache. Used for diagnostics + dt computation.
    last_update_mono_seconds: Optional[float] = None
    last_observed_net_edge_per_min_usd: Optional[float] = None
    last_observed_markout_5s_mean_bps: Optional[float] = None
    # v1.5.290 — the median is the floor-driving statistic; the mean is
    # retained alongside it purely as a diagnostic so the operator can
    # SEE a mean-vs-median divergence (the v1.5.289 pin signature).
    last_observed_markout_5s_median_bps: Optional[float] = None
    last_observed_markout_sample_count: Optional[int] = None
    safety_floor_engaged: bool = False

    # Telemetry counters.
    update_count: int = 0
    safety_floor_engagement_count: int = 0

    def update(
        self,
        *,
        observed_net_edge_per_min_usd: Optional[float],
        observed_markout_5s_median_bps: Optional[float] = None,
        markout_sample_count: Optional[int] = None,
        observed_markout_5s_mean_bps: Optional[float] = None,
        now_mono_seconds: Optional[float] = None,
    ) -> None:
        """Advance the controller one tick.

        Net-edge is the PRIMARY control signal and the only HARD
        requirement: when ``observed_net_edge_per_min_usd`` is None
        or non-finite the controller HOLDS its current state (no
        integrator advance) — there's nothing to drive the PI.

        Markout is an OPTIONAL safety-brake input (v1.5.279
        decouple). When ``observed_markout_5s_mean_bps`` is None or
        non-finite — e.g. the rolling window has too few markout
        samples for a trustworthy mean — the controller does NOT
        hold; it skips the safety-floor check and advances the PI on
        net-edge alone. This is deliberate: net-edge is computable
        even from a single fill (or zero fills → 0.0), so the PI
        stays alive in quiet markets where markout samples are
        sparse. Previously a missing markout froze the whole
        controller, leaving it idling at ``update_count=0`` for an
        entire 60-min session at TON's ~0.23 fills/min.

        ``now_mono_seconds`` defaults to ``time.monotonic()``;
        tests pass explicit values for deterministic dt.
        """
        if not self.settings.enabled:
            # Even when disabled we update the timestamp so a future
            # enable doesn't see a stale dt jump. But we don't
            # advance any control state.
            self.last_update_mono_seconds = (
                float(now_mono_seconds)
                if now_mono_seconds is not None
                else time.monotonic()
            )
            return

        ts = (
            float(now_mono_seconds)
            if now_mono_seconds is not None
            else time.monotonic()
        )

        # Cache observations for diagnostics regardless of whether
        # we advance the controller (operator may want to see
        # exactly what the AQC was being fed).
        if observed_net_edge_per_min_usd is not None and math.isfinite(
            float(observed_net_edge_per_min_usd)
        ):
            self.last_observed_net_edge_per_min_usd = float(
                observed_net_edge_per_min_usd
            )
        if observed_markout_5s_mean_bps is not None and math.isfinite(
            float(observed_markout_5s_mean_bps)
        ):
            self.last_observed_markout_5s_mean_bps = float(
                observed_markout_5s_mean_bps
            )
        if observed_markout_5s_median_bps is not None and math.isfinite(
            float(observed_markout_5s_median_bps)
        ):
            self.last_observed_markout_5s_median_bps = float(
                observed_markout_5s_median_bps
            )
        if markout_sample_count is not None:
            try:
                self.last_observed_markout_sample_count = int(
                    markout_sample_count
                )
            except (TypeError, ValueError):
                pass

        # === Net-edge is the PRIMARY signal; markout is an OPTIONAL
        # safety brake (v1.5.279 decouple). HOLD only when net-edge is
        # missing or non-finite — net-edge is computable even from a
        # sparse window (one fill, or zero fills → 0.0), so the PI can
        # advance in quiet markets. Previously a missing markout froze
        # the whole controller (idled at update_count=0 for a full
        # 60-min session at TON's ~0.23 fills/min). ===
        if observed_net_edge_per_min_usd is None:
            self.last_update_mono_seconds = ts
            return
        if not math.isfinite(float(observed_net_edge_per_min_usd)):
            self.last_update_mono_seconds = ts
            return

        # Markout availability gates ONLY the safety floor, never the
        # PI advance. A missing markout = "no brake this tick", not
        # "freeze the controller".
        #
        # v1.5.290 — the floor input is the window MEDIAN (outlier-robust)
        # AND requires at least ``markout_floor_min_fills`` distinct fills.
        # Both must hold for the brake to even be eligible to fire. The
        # MEAN is NOT consulted by the floor anymore — it is retained only
        # as a diagnostic (cached above, surfaced in snapshot_dict). This
        # closes the v1.5.289 pin: one flash fill (-63.6 bps) dragged the
        # MEAN to -12.3 < -5 and pinned aggression at 0 for the whole
        # window, though the MEDIAN was -2.8 (above the floor).
        min_fills = int(self.settings.markout_floor_min_fills)
        floor_input_available = (
            observed_markout_5s_median_bps is not None
            and math.isfinite(float(observed_markout_5s_median_bps))
            and markout_sample_count is not None
            and int(markout_sample_count) >= min_fills
        )

        # dt for integrator. Cold-start (first tick): use a small
        # nominal dt rather than 0 so the proportional response
        # still fires immediately. 0.5 matches the bot's quote-loop
        # tick rate; the choice doesn't affect steady-state, only
        # the first integrator step's magnitude.
        if self.last_update_mono_seconds is None:
            dt = 0.5
        else:
            dt = max(0.0, ts - float(self.last_update_mono_seconds))

        # === Safety floor check (median + min-fill gate, v1.5.290) ===
        # If the window MEDIAN markout is collapsing, force aggression to
        # zero immediately. This overrides the PI to prevent the
        # controller from pushing aggression UP during a trend where the
        # bot is being picked off. Skipped entirely when the floor input
        # is unavailable (too few distinct fills, or no median) — the PI
        # then runs unbraked on net-edge, which is the intended
        # quiet-market behavior (low fills → low net edge → push
        # aggression up). Using the MEDIAN (not the mean) means a lone
        # flash fill can no longer trip the brake for the whole window.
        if floor_input_available and (
            float(observed_markout_5s_median_bps)
            < self.settings.markout_floor_bps
        ):
            if not self.safety_floor_engaged:
                self.safety_floor_engagement_count += 1
            self.safety_floor_engaged = True
            self.aggression_level = 0.0
            self.integrator = 0.0  # reset so PI doesn't snap back hard when markout recovers
            self.last_update_mono_seconds = ts
            self.update_count += 1
            return

        # Safety floor not engaged this tick.
        self.safety_floor_engaged = False

        # === PI control with conditional-integration anti-windup
        # (v1.5.280) ===
        # Commit the integrator step ONLY when it would not push an
        # already-saturated output further into saturation. This is
        # proper anti-windup: it freezes the integrator at the edge
        # where the output clips, instead of letting it wind on
        # uselessly. The ±[I_min, I_max] hard clip below is retained
        # purely as a defensive backstop — with conditional
        # integration the integrator settles near 1/Ki and never
        # approaches it.
        #
        # Why this matters: with Ki=0.05 the I-term alone saturates
        # aggression at integrator=20. The old clip-only path let the
        # integrator wind to 300 (15× past saturation) in a sustained
        # below-target market, so when conditions recovered it took
        # ~4 h to unwind before aggression came off 1.0. The v1.5.279
        # live session reproduced this exactly (aggression 0.99 after
        # 16 min). Conditional integration freezes the integrator at
        # the saturation edge, so back-off begins the instant the
        # error sign flips.
        error = (
            float(self.settings.target_net_edge_per_min_usd)
            - float(observed_net_edge_per_min_usd)
        )
        kp = float(self.settings.pi_kp)
        ki = float(self.settings.pi_ki)

        # Tentative integrator step, bounded by the hard backstop.
        integrator_candidate = max(
            _INTEGRATOR_MIN,
            min(_INTEGRATOR_MAX, self.integrator + error * dt),
        )
        # Output the candidate WOULD produce (pre-clip).
        raw_with_candidate = kp * error + ki * integrator_candidate
        # Freeze the integrator if committing the step would deepen an
        # existing saturation (windup); otherwise commit it.
        saturating_high = raw_with_candidate > 1.0 and error > 0.0
        saturating_low = raw_with_candidate < 0.0 and error < 0.0
        if not (saturating_high or saturating_low):
            self.integrator = integrator_candidate
        # else: hold the integrator at its prior value — no windup.

        # Output computed from the committed integrator, then clipped.
        raw_aggression = kp * error + ki * self.integrator
        self.aggression_level = max(0.0, min(1.0, raw_aggression))

        self.last_update_mono_seconds = ts
        self.update_count += 1

    def reset(self) -> None:
        """Reset PI state to cold-start. Operator calls this when
        configuration changes mid-session (knob tweaks, regime
        recalibration) and the integrator's old accumulated error
        is no longer relevant."""
        self.aggression_level = 0.0
        self.integrator = 0.0
        self.safety_floor_engaged = False
        # Counters NOT reset — they're session-cumulative diagnostics.

    def snapshot_dict(self) -> dict:
        """Telemetry snapshot for live_stats / state_current.json.

        All fields are torn-read-safe under the GIL — caller does
        not need to lock state. Operators reading this during
        Phase 1 are looking at what the AQC WOULD HAVE DONE; no
        downstream consumer changes its behavior based on these
        values until Phase 2.
        """
        return {
            "enabled": bool(self.settings.enabled),
            "aggression_level": float(self.aggression_level),
            "integrator": float(self.integrator),
            "safety_floor_engaged": bool(self.safety_floor_engaged),
            "safety_floor_engagement_count": int(
                self.safety_floor_engagement_count
            ),
            "update_count": int(self.update_count),
            "last_observed_net_edge_per_min_usd": (
                float(self.last_observed_net_edge_per_min_usd)
                if self.last_observed_net_edge_per_min_usd is not None
                else None
            ),
            "last_observed_markout_5s_mean_bps": (
                float(self.last_observed_markout_5s_mean_bps)
                if self.last_observed_markout_5s_mean_bps is not None
                else None
            ),
            # v1.5.290 — the median is the floor-driving statistic; the
            # sample count gates the min-fill requirement. Surfaced so the
            # operator / acceptance script can confirm the brake fired (or
            # held) on the right input and SEE a mean-vs-median divergence.
            "last_observed_markout_5s_median_bps": (
                float(self.last_observed_markout_5s_median_bps)
                if self.last_observed_markout_5s_median_bps is not None
                else None
            ),
            "last_observed_markout_sample_count": (
                int(self.last_observed_markout_sample_count)
                if self.last_observed_markout_sample_count is not None
                else None
            ),
            "target_net_edge_per_min_usd": float(
                self.settings.target_net_edge_per_min_usd
            ),
            "markout_floor_bps": float(self.settings.markout_floor_bps),
            "markout_floor_min_fills": int(
                self.settings.markout_floor_min_fills
            ),
            "pi_kp": float(self.settings.pi_kp),
            "pi_ki": float(self.settings.pi_ki),
            # Phase 2 (v1.5.281) output-wiring telemetry. Lets the
            # operator / acceptance script confirm whether the
            # aggression→spread wire is live and what floor full
            # aggression targets, without needing a per-decision column.
            "wire_min_half_spread": bool(self.settings.wire_min_half_spread),
            "min_half_spread_floor_at_full_aggression_bps": float(
                self.settings.min_half_spread_floor_at_full_aggression_bps
            ),
            # Phase 3 (v1.5.282) output-wiring telemetry — same purpose
            # as the Phase 2 keys above for the inventory exec-bias floor.
            "wire_inventory": bool(self.settings.wire_inventory),
            "inventory_util_floor_at_full_aggression_pct": float(
                self.settings.inventory_util_floor_at_full_aggression_pct
            ),
            # Phase 4 (v1.5.283) output-wiring telemetry — same purpose
            # for the constructive inventory skew-coefficient lerp.
            "wire_skew": bool(self.settings.wire_skew),
            "skew_coeff_at_full_aggression_bps": float(
                self.settings.skew_coeff_at_full_aggression_bps
            ),
            "last_update_mono_seconds": (
                float(self.last_update_mono_seconds)
                if self.last_update_mono_seconds is not None
                else None
            ),
        }


def compute_rolling_net_edge_per_min(
    *,
    recent_fills_window_seconds: float,
    rebate_usd_in_window: float,
    markout_dollar_impact_usd_in_window: float,
) -> Optional[float]:
    """Pure helper that computes the observed net edge per minute.

    Net edge = rebate income + markout dollar impact, both summed
    over the rolling window. Returns ``None`` when the window is
    too short or non-positive (caller should hold the controller
    state in that case).

    The caller is responsible for the rolling-window bookkeeping
    (which fills count toward this minute's edge). The bot's
    existing ``state.recent_fills`` deque + the markout async
    queue make this straightforward — see the wiring in
    ``app/bot.py`` Phase 1 instrumentation.
    """
    if recent_fills_window_seconds <= 0:
        return None
    total_edge_usd = (
        float(rebate_usd_in_window)
        + float(markout_dollar_impact_usd_in_window)
    )
    minutes = float(recent_fills_window_seconds) / 60.0
    if minutes <= 0:
        return None
    return total_edge_usd / minutes
