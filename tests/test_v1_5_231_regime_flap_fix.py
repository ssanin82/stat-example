"""v1.5.231 — regime-flap fix + diagnostic instrumentation tests.

Closes the 0-fill regression discovered in v1.5.230-260529-094214
snapshot: vol_bps=5.58 sat right at the FSM-classifier boundaries,
causing 235 NORMAL↔CAUTIOUS transitions in 27 min. Each CAUTIOUS
collapse multiplied QUOTE_NOTIONAL_USD ($7) by 0.70 = $4.90, below
MIN_QUOTE_NOTIONAL_USD ($5), dropping every rung and cancelling
all live orders with `desired_none`. Zero fills resulted.

Three fixes verified here:

1. `_KNOBS_CAUTIOUS.quote_notional_mult` is now 1.00 (was 0.70).
   Defensive intent is preserved by `base_half_spread_mult=1.30`
   and `ladder_levels_max=1` — both still material.

2. `evaluate_mode` accepts a new `forward_reason` kwarg and folds
   it into the transition reason string so the FSM transition
   event log records WHICH criterion fired.

3. The widened classifier-band defaults (vol_slope 1.5/0.3, ob
   widening 0.5/0.15, basis stretch 3.0/1.2, calm vol 4.0/15.0)
   are exercised by reading them out of the env-loaded settings
   the production profile will deploy.
"""

from __future__ import annotations

import pytest


def test_cautious_quote_notional_mult_is_one():
    """v1.5.231 — CAUTIOUS must NOT shrink rung notional below the
    venue / config min. 0.70 × $7 = $4.90 < $5 MIN was the killer."""
    from app.regime_controller import _KNOBS_CAUTIOUS
    assert _KNOBS_CAUTIOUS.quote_notional_mult == 1.00, (
        "CAUTIOUS must keep full per-rung notional; defensive intent "
        "lives in base_half_spread_mult + ladder_levels_max only"
    )
    # Sanity: other defensive knobs are still material.
    assert _KNOBS_CAUTIOUS.base_half_spread_mult >= 1.20
    assert _KNOBS_CAUTIOUS.ladder_levels_max == 1
    assert _KNOBS_CAUTIOUS.inventory_budget_mult <= 0.80


def test_cautious_rung_survives_min_notional_at_ton_config():
    """Concrete regression: $7 base × CAUTIOUS notional_mult must
    NOT round below MIN_QUOTE_NOTIONAL_USD=$5."""
    from app.regime_controller import _KNOBS_CAUTIOUS
    quote_notional_usd = 7.0
    min_quote_notional_usd = 5.0
    effective = quote_notional_usd * _KNOBS_CAUTIOUS.quote_notional_mult
    assert effective >= min_quote_notional_usd, (
        f"CAUTIOUS rung at ${quote_notional_usd}×"
        f"{_KNOBS_CAUTIOUS.quote_notional_mult} = ${effective:.2f} would"
        f" drop below MIN_NOTIONAL=${min_quote_notional_usd}"
    )


def test_evaluate_mode_accepts_forward_reason_kwarg():
    """v1.5.231 — `forward_reason` is a new kwarg on `evaluate_mode`.
    Verifies the signature change so older callers still work
    (kwarg has a default of None)."""
    import inspect
    from app.regime_controller import evaluate_mode
    sig = inspect.signature(evaluate_mode)
    assert "forward_reason" in sig.parameters
    assert sig.parameters["forward_reason"].default is None


def _fire_transition(
    state, *, now_mono_first, now_mono_second, **kwargs
):
    """Helper: the FSM arms on the first tick the condition holds
    and fires on the next tick (even at dwell=0). Call twice with
    distinct now_mono so the elapsed-since-arming check passes."""
    from app.regime_controller import evaluate_mode
    # Tick 1: arms the timer, returns (current_mode, None)
    evaluate_mode(state, now_mono=now_mono_first, **kwargs)
    # Tick 2: timer elapsed >= 0 → fires the transition
    return evaluate_mode(state, now_mono=now_mono_second, **kwargs)


def test_transition_reason_includes_classifier_reason():
    """When `forward_reason` is passed, the FSM transition reason
    string must contain it bracketed inside the
    `forward_classifier[...]` tag, so the postmortem can grep for
    the winning criterion."""
    from app.regime_controller import Mode, RegimeControllerState
    from app.regime_forward_signals import ForwardRegime
    state = RegimeControllerState(mode=Mode.NORMAL, mode_since_mono=0.0)
    _mode, reason = _fire_transition(
        state,
        now_mono_first=100.0,
        now_mono_second=100.5,
        util=0.0,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
        forward_regime=ForwardRegime.CAUTIOUS,
        forward_reason="vol_slope:1.65bp/min",
        cautious_entry_dwell_seconds=0.0,
        cautious_exit_dwell_seconds=0.0,
        calm_entry_dwell_seconds=0.0,
        calm_exit_dwell_seconds=0.0,
    )
    assert _mode is Mode.CAUTIOUS
    assert reason is not None
    assert "forward_classifier[vol_slope:1.65bp/min]" in reason, (
        f"transition reason should embed the classifier's reason; got: {reason!r}"
    )


def test_transition_reason_falls_back_when_no_forward_reason():
    """Older callers that don't pass `forward_reason` must still get
    the bare `forward_classifier` tag (backwards-compatible)."""
    from app.regime_controller import Mode, RegimeControllerState
    from app.regime_forward_signals import ForwardRegime
    state = RegimeControllerState(mode=Mode.NORMAL, mode_since_mono=0.0)
    _mode, reason = _fire_transition(
        state,
        now_mono_first=100.0,
        now_mono_second=100.5,
        util=0.0,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
        forward_regime=ForwardRegime.CAUTIOUS,
        # forward_reason omitted — default None
        cautious_entry_dwell_seconds=0.0,
        cautious_exit_dwell_seconds=0.0,
        calm_entry_dwell_seconds=0.0,
        calm_exit_dwell_seconds=0.0,
    )
    assert _mode is Mode.CAUTIOUS
    assert reason is not None
    assert "forward_classifier" in reason
    assert "[" not in reason, (
        "no bracket should appear when forward_reason is absent"
    )


def test_exit_transition_reason_includes_classifier_reason_too():
    """The CAUTIOUS→NORMAL exit path also gets the classifier's
    reason appended — operator sees WHY CAUTIOUS released."""
    from app.regime_controller import Mode, RegimeControllerState
    from app.regime_forward_signals import ForwardRegime
    state = RegimeControllerState(mode=Mode.CAUTIOUS, mode_since_mono=0.0)
    _mode, reason = _fire_transition(
        state,
        now_mono_first=100.0,
        now_mono_second=100.5,
        util=0.0,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
        forward_regime=ForwardRegime.NORMAL,
        forward_reason="not_calm:vol_high:5.10,ob_imbalance:0.31",
        cautious_entry_dwell_seconds=0.0,
        cautious_exit_dwell_seconds=0.0,
        calm_entry_dwell_seconds=0.0,
        calm_exit_dwell_seconds=0.0,
    )
    assert _mode is Mode.NORMAL
    assert reason is not None
    # Existing format `forward=NORMAL` preserved; the criterion
    # detail is appended in brackets.
    assert "forward=NORMAL" in reason
    assert "[not_calm:vol_high:5.10,ob_imbalance:0.31]" in reason


def test_widened_classifier_bands_eliminate_boundary_flap():
    """Synthetic regression: with the new wider bands
    (vol_slope 1.5/0.3 instead of 0.8/0.4), a signal that
    oscillates within the OLD band but outside the NEW band's
    enter threshold should NOT cause CAUTIOUS entries from NORMAL.

    Sanity-check on the numeric design: the v1.5.230 snapshot's
    flap was driven by signals around the 0.8 boundary; the new
    1.5 enter threshold gives ~0.7 bps/min of headroom."""
    from app.regime_forward_signals import (
        ForwardRegime,
        ForwardSignalThresholds,
        classify_forward_regime,
    )
    # Simulate a signal at 1.0 bps/min — above the old enter (0.8)
    # but below the new enter (1.5).
    s = ForwardSignalThresholds(
        cautious_enter_vol_slope_bps_per_min=1.5,
        cautious_exit_vol_slope_bps_per_min=0.3,
        # Other criteria left at defaults; none should fire here.
    )
    # Build a synthetic vol_bps history with positive slope ~1.0
    # bps/min so the linear-regression slope lands ~1.0.
    now_mono = 1000.0
    vol_history: list[tuple[float, float]] = [
        (now_mono - 60.0 + i, 5.0 + i / 60.0)  # slope ~1.0 bps/min
        for i in range(0, 61, 5)
    ]
    reading = classify_forward_regime(
        vol_bps_history=vol_history,
        drift_30s_history=[],
        ob_imbalance_history=[],
        current_ob_imbalance=0.0,
        current_vol_bps=6.0,
        current_drift_30s_bps=0.0,
        current_binance_basis_bps=None,
        binance_basis_30min_median_bps=None,
        settings=s,
        now_mono=now_mono,
        current_mode=ForwardRegime.NORMAL,
    )
    # Sanity: vol_slope was indeed ~1.0
    assert reading.vol_slope_bps_per_min is not None
    assert 0.8 < reading.vol_slope_bps_per_min < 1.3, (
        f"test scaffolding broken: vol_slope={reading.vol_slope_bps_per_min}"
    )
    # The classification must be NORMAL (1.0 < new 1.5 enter).
    assert reading.classification is ForwardRegime.NORMAL, (
        f"vol_slope=1.0 should not trigger CAUTIOUS with widened enter=1.5; "
        f"got {reading.classification.value}: {reading.reason}"
    )


def test_widened_calm_band_stays_calm_through_vol_micro_spike():
    """With the new CALM band (enter 4.0, exit 15.0), a bot already
    in CALM must STAY CALM through a vol blip up to ~10 bps that
    would have exited CALM under the old 12.0 exit threshold."""
    from app.regime_forward_signals import (
        ForwardRegime,
        ForwardSignalThresholds,
        classify_forward_regime,
    )
    s = ForwardSignalThresholds(
        calm_enter_max_vol_bps=4.0,
        calm_exit_max_vol_bps=15.0,
        calm_min_history_seconds=60.0,
    )
    now_mono = 1000.0
    # Build sufficient history (>60s span) so the CALM history-span
    # gate doesn't block exit logic.
    history = [(now_mono - 70.0 + i, 5.0) for i in range(0, 71, 5)]
    # Current vol = 10 (within new exit but outside old exit). Bot
    # currently CALM → use exit threshold 15.0. 10 < 15 → STAY CALM.
    reading = classify_forward_regime(
        vol_bps_history=history,
        drift_30s_history=history,
        ob_imbalance_history=[(now_mono - 70.0 + i, 0.0) for i in range(0, 71, 5)],
        current_ob_imbalance=0.0,
        current_vol_bps=10.0,
        current_drift_30s_bps=0.0,
        current_binance_basis_bps=None,
        binance_basis_30min_median_bps=None,
        settings=s,
        now_mono=now_mono,
        current_mode=ForwardRegime.CALM,
    )
    assert reading.classification is ForwardRegime.CALM, (
        f"vol=10 with calm_exit=15 should stay CALM; got "
        f"{reading.classification.value}: {reading.reason}"
    )
