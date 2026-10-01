"""Phase 4G.1 tests — forward-looking regime classifier.

Pure-function tests against ``classify_forward_regime`` and its
helpers. Covers every classification branch plus the v1.4.200
incident replay (the SF#11175 timeline that fired straight from
NORMAL — Phase 4G's headline acceptance case).
"""

from __future__ import annotations

import pytest

from app.regime_forward_signals import (
    ForwardRegime,
    ForwardSignalReading,
    ForwardSignalThresholds,
    _compute_basis_stretch_ratio,
    _compute_drift_magnitude_rising,
    _compute_ob_imbalance_widening,
    _compute_vol_slope_bps_per_min,
    _history_span_seconds,
    classify_forward_regime,
)


# ---------------------------------------------------------------------------
# Helpers — build synthetic histories
# ---------------------------------------------------------------------------


def _linear_history(
    start_t: float,
    end_t: float,
    start_v: float,
    end_v: float,
    n: int = 60,
) -> list[tuple[float, float]]:
    """``n`` evenly-spaced (ts, value) samples interpolating
    linearly between (start_t, start_v) and (end_t, end_v).
    Useful for synthesising rising/falling signals."""
    if n < 2:
        n = 2
    step_t = (end_t - start_t) / (n - 1)
    step_v = (end_v - start_v) / (n - 1)
    return [
        (start_t + i * step_t, start_v + i * step_v) for i in range(n)
    ]


def _flat_history(
    start_t: float,
    end_t: float,
    value: float,
    n: int = 60,
) -> list[tuple[float, float]]:
    """Constant-value history for the "no signal change" case."""
    if n < 2:
        n = 2
    step_t = (end_t - start_t) / (n - 1)
    return [(start_t + i * step_t, value) for i in range(n)]


def _calm_defaults_thresholds() -> ForwardSignalThresholds:
    """Default settings — used unless a test explicitly tunes one
    threshold to exercise a specific branch."""
    return ForwardSignalThresholds()


# ===========================================================================
# Indicator helper tests
# ===========================================================================


# --- vol slope ---


def test_vol_slope_returns_none_on_empty_history() -> None:
    assert _compute_vol_slope_bps_per_min([], 60.0, 100.0) is None


def test_vol_slope_returns_none_with_only_one_sample() -> None:
    assert _compute_vol_slope_bps_per_min([(95.0, 2.0)], 60.0, 100.0) is None


def test_vol_slope_flat_history_returns_zero() -> None:
    history = _flat_history(start_t=40.0, end_t=100.0, value=2.0)
    slope = _compute_vol_slope_bps_per_min(history, 60.0, 100.0)
    assert slope is not None
    assert abs(slope) < 1e-9


def test_vol_slope_rising_history_returns_positive() -> None:
    """Rising from 1 bps → 5 bps over 60 s = 4 bps in 60 s = 4 bps/min."""
    history = _linear_history(start_t=40.0, end_t=100.0, start_v=1.0, end_v=5.0)
    slope = _compute_vol_slope_bps_per_min(history, 60.0, 100.0)
    assert slope is not None
    assert slope == pytest.approx(4.0, abs=0.01)


def test_vol_slope_falling_history_returns_negative() -> None:
    history = _linear_history(start_t=40.0, end_t=100.0, start_v=5.0, end_v=1.0)
    slope = _compute_vol_slope_bps_per_min(history, 60.0, 100.0)
    assert slope is not None
    assert slope == pytest.approx(-4.0, abs=0.01)


def test_vol_slope_only_uses_samples_in_window() -> None:
    """Samples older than ``now - lookback`` are excluded — old vol
    spikes don't pollute the current-window slope."""
    history = (
        [(0.0, 100.0)]  # old spike outside window
        + _flat_history(start_t=50.0, end_t=100.0, value=2.0)
    )
    slope = _compute_vol_slope_bps_per_min(history, 60.0, 100.0)
    assert slope is not None
    assert abs(slope) < 1e-9  # window is flat; old spike ignored


# --- drift magnitude rising ---


def test_drift_magnitude_rising_returns_ratio_above_floor() -> None:
    """abs(drift) at start = 6, at end = 12 → ratio 2.0. Floor = 5
    so the start value is above floor and the ratio is meaningful."""
    history = _linear_history(
        start_t=70.0, end_t=100.0, start_v=6.0, end_v=12.0
    )
    mag, ratio = _compute_drift_magnitude_rising(
        history, lookback_seconds=30.0, floor_bps=5.0, now_mono=100.0
    )
    assert mag == pytest.approx(12.0)
    assert ratio == pytest.approx(2.0)


def test_drift_magnitude_returns_none_ratio_below_floor() -> None:
    """abs(start) = 2 (below floor=5) → division would amplify noise →
    ratio returned as None. Magnitude still surfaced for diagnostic."""
    history = _linear_history(
        start_t=70.0, end_t=100.0, start_v=2.0, end_v=10.0
    )
    mag, ratio = _compute_drift_magnitude_rising(
        history, lookback_seconds=30.0, floor_bps=5.0, now_mono=100.0
    )
    assert mag == pytest.approx(10.0)
    assert ratio is None


def test_drift_magnitude_uses_absolute_value() -> None:
    """Negative drift treated as positive magnitude. -6 → -12 is a
    rising magnitude, not a falling one."""
    history = [(70.0, -6.0), (100.0, -12.0)]
    mag, ratio = _compute_drift_magnitude_rising(
        history, lookback_seconds=30.0, floor_bps=5.0, now_mono=100.0
    )
    assert mag == pytest.approx(12.0)
    assert ratio == pytest.approx(2.0)


def test_drift_magnitude_returns_none_none_on_empty_history() -> None:
    mag, ratio = _compute_drift_magnitude_rising(
        [], lookback_seconds=30.0, floor_bps=5.0, now_mono=100.0
    )
    assert mag is None and ratio is None


# --- OB imbalance widening ---


def test_ob_imbalance_widening_zero_when_flat() -> None:
    history = _flat_history(start_t=40.0, end_t=100.0, value=0.5)
    delta = _compute_ob_imbalance_widening(history, 60.0, 100.0)
    assert delta is not None
    assert abs(delta) < 1e-9


def test_ob_imbalance_widening_positive_when_growing() -> None:
    """abs(0.1) → abs(0.6) = +0.5 widening."""
    history = _linear_history(
        start_t=40.0, end_t=100.0, start_v=0.1, end_v=0.6
    )
    delta = _compute_ob_imbalance_widening(history, 60.0, 100.0)
    assert delta == pytest.approx(0.5, abs=0.01)


def test_ob_imbalance_widening_handles_sign_flip() -> None:
    """abs(-0.1) = 0.1 → abs(+0.6) = 0.6 → +0.5 widening despite
    sign change. The metric is "imbalance magnitude growth"
    regardless of which side stacks."""
    history = [(40.0, -0.1), (100.0, +0.6)]
    delta = _compute_ob_imbalance_widening(history, 60.0, 100.0)
    assert delta == pytest.approx(0.5, abs=0.01)


def test_ob_imbalance_widening_none_on_short_history() -> None:
    assert _compute_ob_imbalance_widening([(95.0, 0.5)], 60.0, 100.0) is None


# --- basis stretch ratio ---


def test_basis_stretch_typical_case() -> None:
    """Current basis 5 bps, 30-min median 2 bps → ratio 2.5."""
    ratio = _compute_basis_stretch_ratio(
        current_basis_bps=5.0,
        median_basis_30min_bps=2.0,
        floor_bps=1.0,
    )
    assert ratio == pytest.approx(2.5)


def test_basis_stretch_uses_absolute_values() -> None:
    """abs(-5) / abs(-2) = 2.5 — direction of basis irrelevant."""
    ratio = _compute_basis_stretch_ratio(
        current_basis_bps=-5.0,
        median_basis_30min_bps=-2.0,
        floor_bps=1.0,
    )
    assert ratio == pytest.approx(2.5)


def test_basis_stretch_none_when_median_below_floor() -> None:
    """median 0.5 < floor 1.0 → can't meaningfully compute stretch;
    return None to avoid amplifying noise."""
    ratio = _compute_basis_stretch_ratio(
        current_basis_bps=5.0,
        median_basis_30min_bps=0.5,
        floor_bps=1.0,
    )
    assert ratio is None


def test_basis_stretch_none_on_missing_inputs() -> None:
    assert (
        _compute_basis_stretch_ratio(None, 2.0, 1.0) is None
    )
    assert (
        _compute_basis_stretch_ratio(5.0, None, 1.0) is None
    )


# --- history span ---


def test_history_span_uses_longest_history() -> None:
    """Span = (most-recent_ts - oldest_ts) across all provided
    histories; the longest span wins."""
    short = [(95.0, 1.0), (100.0, 1.0)]    # 5 s span
    medium = [(70.0, 1.0), (100.0, 1.0)]   # 30 s span
    long = [(40.0, 1.0), (100.0, 1.0)]     # 60 s span
    assert _history_span_seconds(short, medium, long) == pytest.approx(60.0)


def test_history_span_zero_when_all_empty() -> None:
    assert _history_span_seconds([], []) == 0.0


def test_history_span_skips_short_lists() -> None:
    """Lists with < 2 samples don't contribute a span."""
    one_sample = [(95.0, 1.0)]
    multi = [(40.0, 1.0), (100.0, 1.0)]
    assert _history_span_seconds(one_sample, multi) == pytest.approx(60.0)


# ===========================================================================
# Classifier — branch coverage
# ===========================================================================


def _build_calm_inputs(now_mono: float = 100.0) -> dict:
    """All-quiet baseline: flat 2.0 bps vol, near-zero drift, balanced
    OB, tight basis, plenty of history. Classifier should return CALM."""
    return dict(
        vol_bps_history=_flat_history(start_t=0.0, end_t=now_mono, value=2.0),
        drift_30s_history=_flat_history(start_t=0.0, end_t=now_mono, value=1.0),
        ob_imbalance_history=_flat_history(
            start_t=0.0, end_t=now_mono, value=0.05
        ),
        current_ob_imbalance=0.05,
        current_vol_bps=2.0,
        current_drift_30s_bps=1.0,
        current_binance_basis_bps=0.1,
        binance_basis_30min_median_bps=0.1,
        settings=_calm_defaults_thresholds(),
        now_mono=now_mono,
    )


def test_calm_when_all_quiet() -> None:
    """Baseline: every signal in its CALM zone + enough history →
    CALM."""
    reading = classify_forward_regime(**_build_calm_inputs())
    assert reading.classification == ForwardRegime.CALM
    assert reading.reason == "all_quiet"


def test_normal_when_history_too_short() -> None:
    """Even with all-quiet readings, < calm_min_history_seconds of
    history means we don't trust the calm enough to commit. Stays
    NORMAL until the bot has observed at least N seconds of quiet."""
    inputs = _build_calm_inputs()
    # Shrink all histories to 30 s span (calm threshold = 60 s).
    inputs["vol_bps_history"] = _flat_history(
        start_t=70.0, end_t=100.0, value=2.0
    )
    inputs["drift_30s_history"] = _flat_history(
        start_t=70.0, end_t=100.0, value=1.0
    )
    inputs["ob_imbalance_history"] = _flat_history(
        start_t=70.0, end_t=100.0, value=0.05
    )
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.NORMAL
    assert "history_short" in reading.reason


# --- CAUTIOUS triggers, each in isolation ---


def test_cautious_on_vol_slope_alone() -> None:
    """Vol slope rising at 1 bps/min — above the 0.5 bps/min
    threshold. Other signals quiet. → CAUTIOUS with vol_slope reason."""
    inputs = _build_calm_inputs()
    # Rising vol: 1 → 2 over 60 s = 1 bps/min (above 0.5 threshold).
    inputs["vol_bps_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=1.0, end_v=2.0
    )
    inputs["current_vol_bps"] = 2.0
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.CAUTIOUS
    assert "vol_slope" in reading.reason
    assert reading.vol_slope_bps_per_min == pytest.approx(1.0, abs=0.01)


def test_cautious_on_drift_magnitude_rising_alone() -> None:
    """abs(drift) doubling over the 30 s window — above the 1.5
    ratio threshold. Other signals quiet. → CAUTIOUS."""
    inputs = _build_calm_inputs()
    inputs["drift_30s_history"] = _linear_history(
        start_t=70.0, end_t=100.0, start_v=6.0, end_v=12.0
    )
    inputs["current_drift_30s_bps"] = 12.0
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.CAUTIOUS
    assert "drift_rising" in reading.reason


def test_cautious_on_ob_imbalance_widening_alone() -> None:
    """OB imbalance growing from 0.05 → 0.5 over 60 s = +0.45
    widening (above 0.3 threshold)."""
    inputs = _build_calm_inputs()
    inputs["ob_imbalance_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=0.05, end_v=0.5
    )
    inputs["current_ob_imbalance"] = 0.5
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.CAUTIOUS
    assert "ob_widening" in reading.reason


def test_cautious_on_basis_stretch_alone() -> None:
    """Current basis = 5 bps, median = 2 bps → ratio 2.5 (above
    2.0 cautious threshold). All other signals quiet."""
    inputs = _build_calm_inputs()
    inputs["current_binance_basis_bps"] = 5.0
    inputs["binance_basis_30min_median_bps"] = 2.0
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.CAUTIOUS
    assert "basis_stretch" in reading.reason


def test_cautious_reason_lists_all_active_signals() -> None:
    """Multiple indicators tripping simultaneously → reason joins
    all of them with ' | ' so the operator sees the whole picture."""
    inputs = _build_calm_inputs()
    inputs["vol_bps_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=1.0, end_v=2.0
    )
    inputs["ob_imbalance_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=0.05, end_v=0.5
    )
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.CAUTIOUS
    assert "vol_slope" in reading.reason
    assert "ob_widening" in reading.reason
    assert " | " in reading.reason


def test_cautious_priority_over_calm() -> None:
    """If a CAUTIOUS-trigger fires AND the current absolute values
    are in the CALM zone, CAUTIOUS wins. Asymmetric safety bias."""
    inputs = _build_calm_inputs()
    # Vol slope says CAUTIOUS, but the CURRENT vol is still below
    # the calm threshold (rising fast from very low base).
    inputs["vol_bps_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=0.5, end_v=2.0
    )
    inputs["current_vol_bps"] = 2.0  # within calm_max_vol_bps=3.0
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.CAUTIOUS


def test_normal_when_current_vol_above_calm_threshold() -> None:
    """Vol absolute value above the CALM ceiling but no rising
    slope to trigger CAUTIOUS → NORMAL. The "elevated-but-stable"
    case lives between CALM and CAUTIOUS."""
    inputs = _build_calm_inputs()
    inputs["vol_bps_history"] = _flat_history(
        start_t=0.0, end_t=100.0, value=5.0  # above calm_max=3.0
    )
    inputs["current_vol_bps"] = 5.0
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.NORMAL
    assert "vol_high" in reading.reason


def test_normal_when_drift_magnitude_above_calm_threshold() -> None:
    """Drift magnitude above calm threshold (e.g. steady 10 bps
    drift) but not rising → NORMAL, not CAUTIOUS."""
    inputs = _build_calm_inputs()
    inputs["drift_30s_history"] = _flat_history(
        start_t=0.0, end_t=100.0, value=8.0
    )
    inputs["current_drift_30s_bps"] = 8.0
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.NORMAL
    assert "drift_mag" in reading.reason


def test_normal_when_ob_imbalance_above_calm_but_not_widening() -> None:
    """Stable but elevated OB imbalance (e.g. 0.4) — above calm
    threshold 0.2 but not widening from a lower baseline. NORMAL."""
    inputs = _build_calm_inputs()
    inputs["ob_imbalance_history"] = _flat_history(
        start_t=0.0, end_t=100.0, value=0.4
    )
    inputs["current_ob_imbalance"] = 0.4
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.NORMAL


# --- "no signal" handling ---


def test_normal_when_no_signals_at_all() -> None:
    """Cold start — all histories empty, all current values None.
    Classifier defaults to NORMAL (not CAUTIOUS, not CALM).
    CALM requires positive evidence; CAUTIOUS requires a triggering
    indicator. Neither hold → NORMAL."""
    reading = classify_forward_regime(
        vol_bps_history=[],
        drift_30s_history=[],
        ob_imbalance_history=[],
        current_ob_imbalance=None,
        current_vol_bps=None,
        current_drift_30s_bps=None,
        current_binance_basis_bps=None,
        binance_basis_30min_median_bps=None,
        settings=_calm_defaults_thresholds(),
        now_mono=100.0,
    )
    assert reading.classification == ForwardRegime.NORMAL


def test_missing_signal_blocks_calm_but_not_normal() -> None:
    """Three of four signals are calm; the fourth is None.
    Classifier returns NORMAL (CALM requires all four positive)."""
    inputs = _build_calm_inputs()
    inputs["current_ob_imbalance"] = None
    inputs["ob_imbalance_history"] = []
    reading = classify_forward_regime(**inputs)
    assert reading.classification == ForwardRegime.NORMAL
    assert "no_ob_imbalance" in reading.reason


# --- diagnostic surface ---


def test_diagnostic_fields_always_populated() -> None:
    """All diagnostic fields are surfaced on the reading even when
    they're None (lets the dashboard render an em-dash instead of
    crashing on a missing key)."""
    reading = classify_forward_regime(**_build_calm_inputs())
    # All these attributes exist on the dataclass — accessing them
    # without exception is the test.
    _ = reading.vol_slope_bps_per_min
    _ = reading.drift_magnitude_30s_bps
    _ = reading.drift_magnitude_rising_ratio_observed
    _ = reading.ob_imbalance_widening_delta_observed
    _ = reading.basis_stretch_ratio_observed
    _ = reading.history_span_seconds


def test_reading_is_frozen_dataclass() -> None:
    """``ForwardSignalReading`` is immutable — callers can't mutate
    the result after the classifier returns."""
    reading = classify_forward_regime(**_build_calm_inputs())
    with pytest.raises(Exception):
        reading.classification = ForwardRegime.CAUTIOUS  # type: ignore[misc]


# ===========================================================================
# Threshold customisation
# ===========================================================================


def test_custom_thresholds_can_tighten_calm() -> None:
    """A stricter CALM-enter threshold (e.g. require vol < 1 bps
    instead of 3 bps) makes a previously-CALM session NORMAL. Lets a
    profile dial in its own definition of 'exceptionally calm'."""
    inputs = _build_calm_inputs()
    inputs["settings"] = ForwardSignalThresholds(
        calm_enter_max_vol_bps=1.0,
        calm_exit_max_vol_bps=1.0,  # match enter — no hysteresis
    )
    reading = classify_forward_regime(**inputs)
    # Default vol_bps = 2.0 is above the tightened 1.0 ceiling.
    assert reading.classification == ForwardRegime.NORMAL


def test_custom_thresholds_can_loosen_cautious() -> None:
    """Raising the CAUTIOUS-enter vol-slope threshold from 0.5 to 5.0
    means a 1-bps/min slope no longer triggers. Profile lets you
    desensitise the gate if it's firing too often."""
    inputs = _build_calm_inputs()
    inputs["vol_bps_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=1.0, end_v=2.0  # 1 bps/min
    )
    inputs["settings"] = ForwardSignalThresholds(
        cautious_enter_vol_slope_bps_per_min=5.0,
        cautious_exit_vol_slope_bps_per_min=5.0,
    )
    reading = classify_forward_regime(**inputs)
    # Slope is 1 bps/min, below the 5 bps/min raised threshold.
    assert reading.classification != ForwardRegime.CAUTIOUS


# ===========================================================================
# v1.5.191 — Schmitt-trigger band hysteresis
# ===========================================================================


def test_band_hysteresis_calm_strict_entry_when_not_in_calm() -> None:
    """When current_mode is NOT CALM, classifier uses the strict ENTER
    threshold for OB imbalance. With current_ob_imbalance=0.25 and
    enter_max=0.15 / exit_max=0.30: signals fail the enter bar →
    NORMAL.
    """
    inputs = _build_calm_inputs()
    inputs["current_ob_imbalance"] = 0.25
    inputs["ob_imbalance_history"] = _flat_history(
        start_t=0.0, end_t=100.0, value=0.25
    )
    inputs["settings"] = ForwardSignalThresholds(
        calm_enter_max_ob_imbalance_magnitude=0.15,
        calm_exit_max_ob_imbalance_magnitude=0.30,
    )
    reading = classify_forward_regime(**inputs, current_mode=ForwardRegime.NORMAL)
    assert reading.classification == ForwardRegime.NORMAL
    assert "ob_imbalance" in reading.reason


def test_band_hysteresis_calm_loose_exit_when_in_calm() -> None:
    """When current_mode IS CALM, the SAME signals (0.25 OB imbalance)
    use the loose EXIT threshold (0.30) — bot stays CALM."""
    inputs = _build_calm_inputs()
    inputs["current_ob_imbalance"] = 0.25
    inputs["ob_imbalance_history"] = _flat_history(
        start_t=0.0, end_t=100.0, value=0.25
    )
    inputs["settings"] = ForwardSignalThresholds(
        calm_enter_max_ob_imbalance_magnitude=0.15,
        calm_exit_max_ob_imbalance_magnitude=0.30,
    )
    reading = classify_forward_regime(**inputs, current_mode=ForwardRegime.CALM)
    assert reading.classification == ForwardRegime.CALM


def test_band_hysteresis_calm_exit_when_signal_drifts_past_exit_threshold() -> None:
    """When current_mode IS CALM but a signal crosses the LOOSE exit
    threshold, classifier returns NORMAL (real regime change, not
    boundary chatter)."""
    inputs = _build_calm_inputs()
    inputs["current_ob_imbalance"] = 0.40  # past the 0.30 exit threshold
    inputs["ob_imbalance_history"] = _flat_history(
        start_t=0.0, end_t=100.0, value=0.40
    )
    inputs["settings"] = ForwardSignalThresholds(
        calm_enter_max_ob_imbalance_magnitude=0.15,
        calm_exit_max_ob_imbalance_magnitude=0.30,
    )
    reading = classify_forward_regime(**inputs, current_mode=ForwardRegime.CALM)
    assert reading.classification == ForwardRegime.NORMAL


def test_band_hysteresis_cautious_strict_entry_when_not_in_cautious() -> None:
    """When current_mode is NOT CAUTIOUS, the strict cautious-enter
    threshold gates entry. Slope=0.6 bps/min with enter=0.8 / exit=0.4:
    fails entry → NORMAL."""
    inputs = _build_calm_inputs()
    inputs["vol_bps_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=1.0, end_v=1.6  # 0.6 bps/min
    )
    inputs["current_vol_bps"] = 1.6
    inputs["settings"] = ForwardSignalThresholds(
        cautious_enter_vol_slope_bps_per_min=0.8,
        cautious_exit_vol_slope_bps_per_min=0.4,
    )
    reading = classify_forward_regime(**inputs, current_mode=ForwardRegime.NORMAL)
    assert reading.classification != ForwardRegime.CAUTIOUS


def test_band_hysteresis_cautious_loose_exit_when_in_cautious() -> None:
    """When current_mode IS CAUTIOUS, slope=0.6 bps/min STAYS CAUTIOUS
    via the lower exit threshold (0.4). Bot stays cautious until slope
    clearly recedes below 0.4."""
    inputs = _build_calm_inputs()
    inputs["vol_bps_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=1.0, end_v=1.6  # 0.6 bps/min
    )
    inputs["current_vol_bps"] = 1.6
    inputs["settings"] = ForwardSignalThresholds(
        cautious_enter_vol_slope_bps_per_min=0.8,
        cautious_exit_vol_slope_bps_per_min=0.4,
    )
    reading = classify_forward_regime(**inputs, current_mode=ForwardRegime.CAUTIOUS)
    assert reading.classification == ForwardRegime.CAUTIOUS


def test_band_hysteresis_cautious_exits_when_signal_clearly_clears() -> None:
    """When current_mode IS CAUTIOUS and slope falls below the EXIT
    threshold (0.4), classifier no longer reports CAUTIOUS — real
    recede."""
    inputs = _build_calm_inputs()
    # Slope ~0.2 bps/min — clearly below the 0.4 exit bar.
    inputs["vol_bps_history"] = _linear_history(
        start_t=40.0, end_t=100.0, start_v=1.0, end_v=1.2
    )
    inputs["current_vol_bps"] = 1.2
    inputs["settings"] = ForwardSignalThresholds(
        cautious_enter_vol_slope_bps_per_min=0.8,
        cautious_exit_vol_slope_bps_per_min=0.4,
    )
    reading = classify_forward_regime(**inputs, current_mode=ForwardRegime.CAUTIOUS)
    assert reading.classification != ForwardRegime.CAUTIOUS


def test_band_hysteresis_boundary_chatter_suppressed() -> None:
    """The core invariant: a signal sitting BETWEEN enter and exit
    thresholds produces a CONSTANT classification regardless of
    tick-to-tick noise around that band. Demonstrates the
    anti-chatter property the dwell was supposed to provide.

    Setup: OB imbalance = 0.22 — strictly between enter=0.15 and
    exit=0.30. The classifier should:
    * If current is NORMAL → stay NORMAL (failed enter)
    * If current is CALM → stay CALM (within exit envelope)
    """
    inputs = _build_calm_inputs()
    inputs["current_ob_imbalance"] = 0.22
    inputs["ob_imbalance_history"] = _flat_history(
        start_t=0.0, end_t=100.0, value=0.22
    )
    inputs["settings"] = ForwardSignalThresholds(
        calm_enter_max_ob_imbalance_magnitude=0.15,
        calm_exit_max_ob_imbalance_magnitude=0.30,
    )
    # Same inputs, different current_mode — opposite outcomes.
    r_from_normal = classify_forward_regime(
        **inputs, current_mode=ForwardRegime.NORMAL
    )
    r_from_calm = classify_forward_regime(
        **inputs, current_mode=ForwardRegime.CALM
    )
    assert r_from_normal.classification == ForwardRegime.NORMAL
    assert r_from_calm.classification == ForwardRegime.CALM


def test_band_hysteresis_default_current_mode_is_normal() -> None:
    """Backward-compat: callers that don't pass current_mode get
    NORMAL behaviour (strict enter thresholds). Verifies the default
    parameter contract."""
    inputs = _build_calm_inputs()
    # All-quiet inputs → would be CALM if current_mode were CALM.
    # With default NORMAL, strict-enter thresholds apply but the
    # all-quiet inputs are inside the strict bar too → CALM either way.
    reading = classify_forward_regime(**inputs)  # no current_mode kwarg
    assert reading.classification == ForwardRegime.CALM


def test_band_hysteresis_setting_enter_equals_exit_disables_hysteresis() -> None:
    """If a profile sets enter == exit for a criterion, that criterion
    reverts to the pre-v1.5.191 single-threshold behaviour. Useful for
    operators who want band hysteresis on some criteria but not others."""
    inputs = _build_calm_inputs()
    inputs["current_ob_imbalance"] = 0.18
    inputs["ob_imbalance_history"] = _flat_history(
        start_t=0.0, end_t=100.0, value=0.18
    )
    # enter == exit == 0.20 — single threshold behaviour
    inputs["settings"] = ForwardSignalThresholds(
        calm_enter_max_ob_imbalance_magnitude=0.20,
        calm_exit_max_ob_imbalance_magnitude=0.20,
    )
    # 0.18 < 0.20: passes the gate from both directions
    r_from_normal = classify_forward_regime(
        **inputs, current_mode=ForwardRegime.NORMAL
    )
    r_from_calm = classify_forward_regime(
        **inputs, current_mode=ForwardRegime.CALM
    )
    assert r_from_normal.classification == ForwardRegime.CALM
    assert r_from_calm.classification == ForwardRegime.CALM


# ===========================================================================
# Regression replay — the v1.4.200 SF#11175 incident
# ===========================================================================


def test_regression_v1_4_200_sf_11175_would_have_triggered_cautious() -> None:
    """Replays the v1.4.200-260521-180617 SF#11175 timeline. At
    ~13:50 the vol_bps history showed a sharp upward slope (~1-2
    bps/min) AND the OB imbalance started widening. With Phase 4G's
    classifier on the same inputs, ``CAUTIOUS`` would have
    classified the regime BEFORE the 13:52 SF fired — giving the
    FSM time to widen / shrink ahead of the storm.

    The actual v1.4.200 vol trajectory (from the snapshot
    inventory_history and the vol-SMA chart in the operator's
    screenshot): vol_bps held ~2-3 bps for most of the session,
    spiked to ~6 bps in the final ~5 minutes before SF#11175.
    That's a ~3-bps rise over ~300 s = ~0.6 bps/min — just above
    the default 0.5 bps/min CAUTIOUS threshold.

    Test fixture builds the same shape: vol 2.5 bps stable for
    240 s, then climbing to 6.0 bps over the final 60 s. CAUTIOUS
    should classify at the moment of evaluation."""
    now = 300.0
    # First 240 s: stable vol around 2.5 bps. Final 60 s: climbing
    # from 2.5 to 6.0. The classifier's 60-s lookback samples the
    # climb segment.
    stable_segment = _flat_history(start_t=0.0, end_t=240.0, value=2.5, n=24)
    climb_segment = _linear_history(
        start_t=240.0, end_t=300.0, start_v=2.5, end_v=6.0, n=12
    )
    vol_history = stable_segment + climb_segment
    # Drift was also rising in the snapshot but to keep this test
    # focused on the vol-slope signal alone, leave drift quiet.
    reading = classify_forward_regime(
        vol_bps_history=vol_history,
        drift_30s_history=_flat_history(start_t=0.0, end_t=now, value=1.0),
        ob_imbalance_history=_flat_history(start_t=0.0, end_t=now, value=0.05),
        current_ob_imbalance=0.05,
        current_vol_bps=6.0,
        current_drift_30s_bps=1.0,
        current_binance_basis_bps=0.1,
        binance_basis_30min_median_bps=0.1,
        settings=ForwardSignalThresholds(),
        now_mono=now,
    )
    assert reading.classification == ForwardRegime.CAUTIOUS, (
        f"Classifier should flag the pre-SF#11175 vol climb as "
        f"CAUTIOUS; got {reading.classification.value} "
        f"(reason={reading.reason!r}, slope="
        f"{reading.vol_slope_bps_per_min})"
    )
    # The reason should mention vol_slope (the dominant signal in
    # this fixture).
    assert "vol_slope" in reading.reason
    # Slope should be in the expected envelope (~3.5 bps over 60 s
    # = 3.5 bps/min; well above the 0.5 threshold).
    assert reading.vol_slope_bps_per_min is not None
    assert reading.vol_slope_bps_per_min >= 0.5
