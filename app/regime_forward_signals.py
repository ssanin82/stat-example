"""Phase 4G.1 — forward-looking regime classifier (pure functions).

Operator directive (2026-05-21, post v1.4.200-260521-180617 SF storm):

    "I really need a mechanism that will TEMPORARILY switch to more
    defensive quoting in this regime! NOT JUST TUNING, but auto-
    switch. Not one-off that will blunt the trading during high
    volatility, but strip off all the PnL when the regime is benign!
    I need behavior that gets good PnL when regime is calm, and
    stays really cautious in high risk regime!"

What this module does
=====================

Pure-function classifier that maps **current signal readings** to a
forward-looking regime label:

* ``CALM``     — exceptionally benign conditions (low vol, balanced
                 OB, low drift magnitude, stable spread). The FSM
                 layer uses this to UPSHIFT to a more aggressive
                 quoting profile.
* ``NORMAL``   — default. Today's bot behaviour.
* ``CAUTIOUS`` — leading indicators are rising even if absolute
                 thresholds haven't been crossed. The FSM uses this
                 to UPSHIFT to a more defensive quoting profile
                 BEFORE shock_gate / SF would fire.

``CAUTIOUS`` is the headline contribution. The existing reactive
defences (shock_gate, vol_auto_pause, SF position-drawdown) fire
when LAGGING indicators cross absolute thresholds — by which time
the bot has already taken a loss. ``CAUTIOUS`` fires on **LEADING**
indicators (vol slope, drift magnitude rising, OB imbalance
widening, binance basis stretching) so the bot can widen / shrink
BEFORE the storm fully lands.

Why pure functions
==================

The classifier is **stateless**. Inputs are histories + current
values; output is a classification + reason + diagnostic fields.
The FSM in ``app/regime_controller.py`` (4G.2) holds dwell state
and decides whether to commit a transition. This separation:

* Lets us exhaustively unit-test the classifier with synthesised
  inputs.
* Keeps the dwell / hysteresis logic in one place (the FSM).
* Makes the classifier reusable for backtesting / postmortem
  replay against historical snapshot data.

Priority order (CAUTIOUS wins over CALM)
========================================

If ANY leading indicator says CAUTIOUS, the classification is
CAUTIOUS regardless of how quiet the others are. CALM requires ALL
indicators in their benign zones simultaneously. NORMAL is the
default when the readings are neither uniformly calm nor
showing a single CAUTIOUS-triggering signal.

This asymmetry is intentional: false-positive CAUTIOUS costs a few
basis points of widened spread; false-negative ("missed CAUTIOUS,
stayed NORMAL into a storm") costs the SF-event tail we observed in
v1.4.200-260521-180617. The safety-biased default is to call
CAUTIOUS on weaker evidence.

DEFENSIVE and SHOCK are NOT produced by this classifier — those
remain reactive triggers in the existing FSM (shock_gate firing,
SF triggering). 4G.2 wires the forward-signal output into the FSM
as the entry path for CAUTIOUS / CALM ONLY; reactive triggers
keep their existing escalation path for DEFENSIVE / SHOCK.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ForwardRegime(enum.Enum):
    """The three modes this classifier can produce. DEFENSIVE and
    SHOCK are NOT in this enum — they're owned by the reactive
    triggers in ``app/regime_controller.py``."""

    CALM = "CALM"
    NORMAL = "NORMAL"
    CAUTIOUS = "CAUTIOUS"


@dataclass(frozen=True)
class ForwardSignalThresholds:
    """All tuning knobs in one place. Defaults chosen for TON-on-OKX
    based on the v1.4.200 snapshot's signal envelope; profiles can
    override via env (settings round-trip in 4G.5).

    v1.5.191 — Schmitt-trigger band hysteresis. Each criterion has
    TWO thresholds: ``*_enter_*`` (strict — apply when transitioning
    INTO the regime) and ``*_exit_*`` (loose — apply when ALREADY in
    the regime and considering exit). The classifier consults
    ``current_mode`` to pick the right threshold per tick. This
    replaces the v1.4.200 single-threshold + FSM-dwell pattern, which
    was a CLAUDE.md Rule 0c violation (fixed-time hysteresis on top
    of an already-smoothed signal). See ``plans/20260527-regime-
    band-hysteresis.md`` for the rationale.

    Hysteresis geometry (band width = exit − enter):
    * CALM bands are WIDER on exit (signals must drift well outside
      to leave): e.g., enter at vol_bps ≤ 4, exit at vol_bps > 8.
    * CAUTIOUS bands are LOWER on exit (signals must clearly clear):
      e.g., enter at vol_slope ≥ 0.8, exit at vol_slope < 0.4.
    * Setting enter == exit reverts to the v1.4.200 single-threshold
      behaviour for that criterion (no hysteresis).

    The defaults remain asymmetric in spirit: CAUTIOUS enter is
    intentionally low (false-positive CAUTIOUS is cheap; false-
    negative is expensive). CALM enter is tighter than CALM exit so
    we don't over-commit to CALM on the first quiet sample after
    noise."""

    # --- CAUTIOUS triggers (any one of these → CAUTIOUS) ---

    # Vol slope: bps/minute derivative of vol_bps over the lookback
    # window. v1.4.200-260521-180617's 13:50 vol spike showed a
    # slope of ~1-2 bps/min before SF#11175 fired.
    cautious_enter_vol_slope_bps_per_min: float = 0.5
    cautious_exit_vol_slope_bps_per_min: float = 0.25
    vol_slope_lookback_seconds: float = 60.0

    # Drift magnitude rising: abs(current drift_30s) divided by
    # abs(drift_30s at window start). Enter at ≥1.5 (≥50 % growth);
    # exit at <1.25 (≥25 % growth, but receding from the entry bar).
    cautious_enter_drift_magnitude_rising_ratio: float = 1.5
    cautious_exit_drift_magnitude_rising_ratio: float = 1.25
    drift_magnitude_lookback_seconds: float = 30.0
    drift_magnitude_floor_bps: float = 5.0

    # OB imbalance widening: delta of |ob_imbalance| over the
    # window. Bid-stack or ask-stack growing without offset moves
    # the imbalance EWMA. Enter at +0.3; exit at +0.15.
    cautious_enter_ob_imbalance_widening_delta: float = 0.3
    cautious_exit_ob_imbalance_widening_delta: float = 0.15
    ob_imbalance_lookback_seconds: float = 60.0

    # Basis stretch: abs(current binance_basis_bps) divided by
    # abs(30-min median basis). Enter at ≥2.0; exit at <1.5.
    cautious_enter_basis_stretch_ratio: float = 2.0
    cautious_exit_basis_stretch_ratio: float = 1.5
    basis_stretch_floor_bps: float = 1.0  # median below this → don't divide

    # --- CALM thresholds (ALL must hold) ---

    # Vol must be below this absolute level. Enter when ≤3, exit when >6.
    calm_enter_max_vol_bps: float = 3.0
    calm_exit_max_vol_bps: float = 6.0
    # Drift magnitude must be below this absolute level.
    calm_enter_max_drift_magnitude_bps: float = 5.0
    calm_exit_max_drift_magnitude_bps: float = 10.0
    # OB imbalance must be below this absolute level (balanced book).
    # The 0.15/0.30 default pair gives a 0.15-wide hysteresis band
    # which is the v1.5.190 snapshot's empirically-measured chatter
    # envelope for TON on OKX.
    calm_enter_max_ob_imbalance_magnitude: float = 0.15
    calm_exit_max_ob_imbalance_magnitude: float = 0.30
    # Need at least this much history of all-quiet before declaring
    # CALM (don't commit early on the first quiet sample after noise).
    calm_min_history_seconds: float = 60.0


@dataclass(frozen=True)
class ForwardSignalReading:
    """The classifier's output. ``classification`` is the
    headline result; ``reason`` is a short human-readable tag for
    log lines and the postmortem `regime_modes` section. The
    remaining fields are diagnostic — surfaced so the operator can
    see WHICH signal was the dominant driver."""

    classification: ForwardRegime
    reason: str

    # Diagnostic values. Each is the computed indicator value at the
    # classification moment, or ``None`` if the underlying history
    # was too short to compute. Operator reads these in the snapshot
    # / postmortem to validate the classification matches their
    # intuition about the regime.
    vol_slope_bps_per_min: Optional[float]
    drift_magnitude_30s_bps: Optional[float]
    drift_magnitude_rising_ratio_observed: Optional[float]
    ob_imbalance_widening_delta_observed: Optional[float]
    basis_stretch_ratio_observed: Optional[float]
    history_span_seconds: float


# ---------------------------------------------------------------------------
# Indicator helpers — pure functions, defensive on short histories
# ---------------------------------------------------------------------------


def _compute_vol_slope_bps_per_min(
    vol_bps_history: list[tuple[float, float]],
    lookback_seconds: float,
    now_mono: float,
) -> Optional[float]:
    """Linear-regression slope of vol_bps over the lookback window,
    expressed in bps per MINUTE. ``None`` if fewer than 2 samples
    inside the window."""
    if not vol_bps_history or lookback_seconds <= 0:
        return None
    cutoff = now_mono - lookback_seconds
    window = [
        (t, v) for (t, v) in vol_bps_history if t >= cutoff and v is not None
    ]
    if len(window) < 2:
        return None
    # Simple linear-regression slope using sums (no numpy dependency).
    n = len(window)
    sum_t = sum(t for t, _ in window)
    sum_v = sum(v for _, v in window)
    sum_tt = sum(t * t for t, _ in window)
    sum_tv = sum(t * v for t, v in window)
    denom = n * sum_tt - sum_t * sum_t
    if denom <= 0:
        return None
    slope_per_second = (n * sum_tv - sum_t * sum_v) / denom
    return slope_per_second * 60.0


def _compute_drift_magnitude_rising(
    drift_30s_history: list[tuple[float, float]],
    lookback_seconds: float,
    floor_bps: float,
    now_mono: float,
) -> tuple[Optional[float], Optional[float]]:
    """Returns ``(current_magnitude_bps, ratio_to_window_start)``.

    ``ratio_to_window_start`` = abs(current) / abs(window_start) — only
    meaningful if abs(window_start) > ``floor_bps`` (otherwise the
    division blows up from noise). When the floor isn't met, returns
    the magnitude but ``None`` for the ratio.
    """
    if not drift_30s_history:
        return None, None
    cutoff = now_mono - lookback_seconds
    window = [
        (t, v) for (t, v) in drift_30s_history if t >= cutoff and v is not None
    ]
    if not window:
        return None, None
    current_mag = abs(window[-1][1])
    start_mag = abs(window[0][1])
    if start_mag < floor_bps:
        return current_mag, None
    return current_mag, current_mag / start_mag


def _compute_ob_imbalance_widening(
    ob_imbalance_history: list[tuple[float, float]],
    lookback_seconds: float,
    now_mono: float,
) -> Optional[float]:
    """Magnitude of change in |ob_imbalance| over the window.
    Positive value = imbalance widening; we don't care about
    direction here (either side stacking is a leading indicator).
    ``None`` if fewer than 2 samples."""
    if not ob_imbalance_history:
        return None
    cutoff = now_mono - lookback_seconds
    window = [
        (t, v) for (t, v) in ob_imbalance_history
        if t >= cutoff and v is not None
    ]
    if len(window) < 2:
        return None
    return abs(window[-1][1]) - abs(window[0][1])


def _compute_basis_stretch_ratio(
    current_basis_bps: Optional[float],
    median_basis_30min_bps: Optional[float],
    floor_bps: float,
) -> Optional[float]:
    """abs(current) / abs(median). ``None`` when median magnitude
    is below the floor (the basis is genuinely tight around zero —
    no "stretch" to measure)."""
    if current_basis_bps is None or median_basis_30min_bps is None:
        return None
    med_mag = abs(median_basis_30min_bps)
    if med_mag < floor_bps:
        return None
    return abs(current_basis_bps) / med_mag


def _history_span_seconds(
    *histories: list[tuple[float, float]],
) -> float:
    """Longest (most-recent − oldest) span across the provided
    history lists. Used to gate CALM ("need at least N seconds of
    observed quiet")."""
    spans: list[float] = []
    for h in histories:
        if h and len(h) >= 2:
            spans.append(h[-1][0] - h[0][0])
    return max(spans) if spans else 0.0


# ---------------------------------------------------------------------------
# Classifier — the entry point
# ---------------------------------------------------------------------------


def classify_forward_regime(
    *,
    vol_bps_history: list[tuple[float, float]],
    drift_30s_history: list[tuple[float, float]],
    ob_imbalance_history: list[tuple[float, float]],
    current_ob_imbalance: Optional[float],
    current_vol_bps: Optional[float],
    current_drift_30s_bps: Optional[float],
    current_binance_basis_bps: Optional[float],
    binance_basis_30min_median_bps: Optional[float],
    settings: ForwardSignalThresholds,
    now_mono: float,
    current_mode: ForwardRegime = ForwardRegime.NORMAL,
) -> ForwardSignalReading:
    """Map current signal readings to a forward-looking regime.

    Priority order: **CAUTIOUS wins over CALM wins over NORMAL.**

    * ANY leading indicator in its CAUTIOUS zone → ``CAUTIOUS`` with
      a reason string identifying the winning indicator.
    * All current values in their CALM zone AND enough quiet
      history → ``CALM``.
    * Otherwise → ``NORMAL``.

    The classifier is **stateless** but consults ``current_mode`` for
    Schmitt-trigger band hysteresis (v1.5.191):

    * When ``current_mode != CAUTIOUS``, CAUTIOUS triggers use the
      *strict* enter thresholds.
    * When ``current_mode == CAUTIOUS``, CAUTIOUS triggers use the
      *loose* exit thresholds — i.e., the bot stays CAUTIOUS until
      indicators clearly recede past a lower bar.
    * Symmetrically for CALM: ``current_mode != CALM`` uses strict
      enter thresholds; ``current_mode == CALM`` uses loose exit
      thresholds.

    The bands replace the pre-v1.5.191 FSM-dwell pattern. They live
    at the signal layer (correct location for hysteresis) and don't
    consult elapsed time (CLAUDE.md Rule 0c clean).

    ``current_mode`` defaults to ``NORMAL`` so callers that haven't
    been updated still get a sensible classification (strict
    thresholds, same as the pre-v1.5.191 single-threshold path when
    enter == exit).

    Defensive: any indicator can be ``None`` (history too short, or
    upstream value missing). A missing indicator is treated as "no
    signal" — it cannot push toward CAUTIOUS, and it blocks CALM
    (CALM requires positive evidence from all four channels).
    """
    # Compute each indicator. Each helper returns None when its
    # underlying history is insufficient.
    vol_slope = _compute_vol_slope_bps_per_min(
        vol_bps_history,
        settings.vol_slope_lookback_seconds,
        now_mono,
    )
    drift_mag, drift_ratio = _compute_drift_magnitude_rising(
        drift_30s_history,
        settings.drift_magnitude_lookback_seconds,
        settings.drift_magnitude_floor_bps,
        now_mono,
    )
    ob_widening = _compute_ob_imbalance_widening(
        ob_imbalance_history,
        settings.ob_imbalance_lookback_seconds,
        now_mono,
    )
    basis_stretch = _compute_basis_stretch_ratio(
        current_binance_basis_bps,
        binance_basis_30min_median_bps,
        settings.basis_stretch_floor_bps,
    )
    history_span = _history_span_seconds(
        vol_bps_history,
        drift_30s_history,
        ob_imbalance_history,
    )

    # --- Band selection per regime (Schmitt-trigger hysteresis) ---
    # CAUTIOUS triggers use enter thresholds unless we're already
    # CAUTIOUS, in which case the lower exit thresholds apply (stay
    # CAUTIOUS until the signal clearly recedes).
    in_cautious = current_mode is ForwardRegime.CAUTIOUS
    in_calm = current_mode is ForwardRegime.CALM
    vol_slope_thr = (
        settings.cautious_exit_vol_slope_bps_per_min
        if in_cautious
        else settings.cautious_enter_vol_slope_bps_per_min
    )
    drift_ratio_thr = (
        settings.cautious_exit_drift_magnitude_rising_ratio
        if in_cautious
        else settings.cautious_enter_drift_magnitude_rising_ratio
    )
    ob_widening_thr = (
        settings.cautious_exit_ob_imbalance_widening_delta
        if in_cautious
        else settings.cautious_enter_ob_imbalance_widening_delta
    )
    basis_stretch_thr = (
        settings.cautious_exit_basis_stretch_ratio
        if in_cautious
        else settings.cautious_enter_basis_stretch_ratio
    )
    # CALM absolutes use enter thresholds unless we're already CALM,
    # in which case the larger exit thresholds apply (stay CALM until
    # the signal clearly drifts outside).
    calm_vol_thr = (
        settings.calm_exit_max_vol_bps
        if in_calm
        else settings.calm_enter_max_vol_bps
    )
    calm_drift_thr = (
        settings.calm_exit_max_drift_magnitude_bps
        if in_calm
        else settings.calm_enter_max_drift_magnitude_bps
    )
    calm_ob_thr = (
        settings.calm_exit_max_ob_imbalance_magnitude
        if in_calm
        else settings.calm_enter_max_ob_imbalance_magnitude
    )

    # --- CAUTIOUS check (any one of these wins) ---
    cautious_reasons: list[str] = []
    if vol_slope is not None and vol_slope >= vol_slope_thr:
        cautious_reasons.append(f"vol_slope:{vol_slope:.2f}bp/min")
    if drift_ratio is not None and drift_ratio >= drift_ratio_thr:
        cautious_reasons.append(f"drift_rising:x{drift_ratio:.2f}")
    if ob_widening is not None and ob_widening >= ob_widening_thr:
        cautious_reasons.append(f"ob_widening:+{ob_widening:.2f}")
    if basis_stretch is not None and basis_stretch >= basis_stretch_thr:
        cautious_reasons.append(f"basis_stretch:x{basis_stretch:.2f}")

    if cautious_reasons:
        return ForwardSignalReading(
            classification=ForwardRegime.CAUTIOUS,
            reason=" | ".join(cautious_reasons),
            vol_slope_bps_per_min=vol_slope,
            drift_magnitude_30s_bps=drift_mag,
            drift_magnitude_rising_ratio_observed=drift_ratio,
            ob_imbalance_widening_delta_observed=ob_widening,
            basis_stretch_ratio_observed=basis_stretch,
            history_span_seconds=history_span,
        )

    # --- CALM check (ALL must hold) ---
    # The CALM zone is defined on absolute CURRENT values (not
    # derivatives). The history-span gate prevents flipping to CALM
    # on the first quiet sample after a storm.
    calm_failures: list[str] = []
    if current_vol_bps is None:
        calm_failures.append("no_vol")
    elif current_vol_bps > calm_vol_thr:
        calm_failures.append(f"vol_high:{current_vol_bps:.2f}")
    if current_drift_30s_bps is None:
        calm_failures.append("no_drift")
    elif abs(current_drift_30s_bps) > calm_drift_thr:
        calm_failures.append(
            f"drift_mag:{abs(current_drift_30s_bps):.2f}"
        )
    if current_ob_imbalance is None:
        calm_failures.append("no_ob_imbalance")
    elif abs(current_ob_imbalance) > calm_ob_thr:
        calm_failures.append(
            f"ob_imbalance:{abs(current_ob_imbalance):.2f}"
        )
    if history_span < settings.calm_min_history_seconds:
        calm_failures.append(
            f"history_short:{history_span:.0f}s"
        )

    if not calm_failures:
        return ForwardSignalReading(
            classification=ForwardRegime.CALM,
            reason="all_quiet",
            vol_slope_bps_per_min=vol_slope,
            drift_magnitude_30s_bps=drift_mag,
            drift_magnitude_rising_ratio_observed=drift_ratio,
            ob_imbalance_widening_delta_observed=ob_widening,
            basis_stretch_ratio_observed=basis_stretch,
            history_span_seconds=history_span,
        )

    # --- Default: NORMAL ---
    # Reason carries the dominant calm-failure tag so the operator
    # can see at a glance which signal is keeping the bot out of
    # CALM (typically: "drift_mag:6.3" means "drift is 6.3 bps, just
    # over the 5 bps calm threshold — almost there").
    return ForwardSignalReading(
        classification=ForwardRegime.NORMAL,
        reason="not_calm:" + ",".join(calm_failures[:3]),
        vol_slope_bps_per_min=vol_slope,
        drift_magnitude_30s_bps=drift_mag,
        drift_magnitude_rising_ratio_observed=drift_ratio,
        ob_imbalance_widening_delta_observed=ob_widening,
        basis_stretch_ratio_observed=basis_stretch,
        history_span_seconds=history_span,
    )
