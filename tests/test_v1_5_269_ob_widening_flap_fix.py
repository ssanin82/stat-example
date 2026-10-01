"""v1.5.269 — ob_widening flap fix (raise enter threshold 0.8 → 1.5).

Diagnostic from snapshot v1.5.266-260529-192320-prod.okx.ton.usdt.perp:

* 326 regime_mode_transition events in 49.1 min = 6.64/min (>>2/min ceiling).
* 162 NORMAL→CAUTIOUS and 162 CAUTIOUS→NORMAL — perfectly symmetric flap.
* 140 of 162 (86%) CAUTIOUS entries triggered by ob_widening.
* All 140 ob_widening values fell in [0.80, 0.94] — a 14-cent-wide cluster
  sitting right above the 0.8 enter threshold. Distribution p10=0.81,
  p50=0.84, p90=0.90, max=0.94.

Conclusion: the ob_widening signal is fundamentally noise-dominated for
TON-USDT-SWAP. The orderbook imbalance EWMA naturally swings past 0.8
every ~20 s without any real risk event. Raising enter threshold to
1.5 puts it ABOVE the observed noise envelope so only genuinely large
imbalance shifts (e.g. flash spikes with >1.5 widening within the
60s lookback window) trip CAUTIOUS.

These tests lock in the v1.5.269 prod profile value so a future env
revert reintroduces the flap visibly (test failure) rather than
silently.
"""

from __future__ import annotations

from pathlib import Path


def test_prod_profile_ob_widening_enter_threshold_is_1_5():
    """Prod profile must ship the v1.5.269 enter threshold value."""
    env_path = (
        Path(__file__).parent.parent
        / "config"
        / "profiles"
        / "prod.okx.ton.usdt.perp.env"
    )
    text = env_path.read_text(encoding="utf-8")
    last_val = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(
            "REGIME_FORWARD_CAUTIOUS_ENTER_OB_IMBALANCE_WIDENING_DELTA="
        ):
            last_val = line.split("=", 1)[1].strip()
    assert last_val is not None, (
        "REGIME_FORWARD_CAUTIOUS_ENTER_OB_IMBALANCE_WIDENING_DELTA missing "
        "from prod profile"
    )
    assert last_val == "1.5", (
        f"v1.5.269 calibration: enter threshold should be 1.5 (was 0.8 "
        f"in v1.5.238, but observed natural noise envelope p90=0.90 / "
        f"max=0.94, so 0.8 produced 6.64 transitions/min). Got {last_val!r}."
    )


def test_prod_profile_ob_widening_exit_threshold_unchanged():
    """Exit threshold stays at 0.10 — the gap between enter (1.5) and
    exit (0.10) is now huge, but that's fine because the EXIT triggers
    fine when ob_widening naturally recedes. The flap pattern was
    entry-driven, not exit-driven."""
    env_path = (
        Path(__file__).parent.parent
        / "config"
        / "profiles"
        / "prod.okx.ton.usdt.perp.env"
    )
    text = env_path.read_text(encoding="utf-8")
    last_val = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(
            "REGIME_FORWARD_CAUTIOUS_EXIT_OB_IMBALANCE_WIDENING_DELTA="
        ):
            last_val = line.split("=", 1)[1].strip()
    assert last_val == "0.10", (
        f"exit threshold should remain 0.10 (the flap was entry-driven, "
        f"not exit-driven; exit threshold doesn't need to change). "
        f"Got {last_val!r}."
    )


def test_settings_picks_up_new_enter_threshold_via_env():
    """Settings reads the env knob correctly."""
    from app.config import Settings
    s = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
        REGIME_FORWARD_CAUTIOUS_ENTER_OB_IMBALANCE_WIDENING_DELTA=1.5,
    )
    assert s.regime_forward_cautious_enter_ob_imbalance_widening_delta == 1.5


def test_classifier_does_not_trigger_cautious_at_old_threshold():
    """Direct unit test on the classifier: with the new threshold (1.5),
    an ob_widening value of 0.84 (the median observed in the flap
    snapshot) does NOT trigger CAUTIOUS. Catches a future regression
    where someone silently lowers the threshold back to 0.8.
    """
    from app.regime_forward_signals import (
        classify_forward_regime, ForwardRegime, ForwardSignalThresholds,
    )
    # Build a settings object with the v1.5.269 threshold.
    settings = ForwardSignalThresholds(
        cautious_enter_ob_imbalance_widening_delta=1.5,
        cautious_exit_ob_imbalance_widening_delta=0.10,
    )
    # Construct an ob_imbalance history representing the flap: 60s ago
    # the imbalance was 0.0, now it's 0.84 → widening = 0.84.
    now_mono = 1000.0
    ob_history = [(now_mono - 60.0, 0.0), (now_mono, 0.84)]
    reading = classify_forward_regime(
        vol_bps_history=[],
        drift_30s_history=[],
        ob_imbalance_history=ob_history,
        current_ob_imbalance=0.84,
        current_vol_bps=4.0,  # below CALM, no CAUTIOUS trigger
        current_drift_30s_bps=0.0,
        current_binance_basis_bps=0.0,
        binance_basis_30min_median_bps=0.0,
        settings=settings,
        now_mono=now_mono,
        current_mode=ForwardRegime.NORMAL,
    )
    assert reading.classification is not ForwardRegime.CAUTIOUS, (
        f"v1.5.269 regression guard: ob_widening=0.84 must NOT trigger "
        f"CAUTIOUS at the new enter threshold of 1.5. Got "
        f"{reading.classification.name} with reason {reading.reason!r}."
    )


def test_classifier_still_triggers_cautious_on_genuine_widening():
    """Sanity: a TRUE shock event (widening >= 1.5) STILL trips CAUTIOUS.
    Confirms we haven't killed the signal entirely.
    """
    from app.regime_forward_signals import (
        classify_forward_regime, ForwardRegime, ForwardSignalThresholds,
    )
    settings = ForwardSignalThresholds(
        cautious_enter_ob_imbalance_widening_delta=1.5,
        cautious_exit_ob_imbalance_widening_delta=0.10,
    )
    # Genuine flash: imbalance went from 0 (60s ago) to +1.6 (now)
    # → widening = 1.6 > 1.5 enter threshold.
    now_mono = 1000.0
    ob_history = [(now_mono - 60.0, 0.0), (now_mono, 1.6)]
    reading = classify_forward_regime(
        vol_bps_history=[],
        drift_30s_history=[],
        ob_imbalance_history=ob_history,
        current_ob_imbalance=1.6,
        current_vol_bps=4.0,
        current_drift_30s_bps=0.0,
        current_binance_basis_bps=0.0,
        binance_basis_30min_median_bps=0.0,
        settings=settings,
        now_mono=now_mono,
        current_mode=ForwardRegime.NORMAL,
    )
    assert reading.classification is ForwardRegime.CAUTIOUS, (
        f"genuine widening >= 1.5 should STILL trigger CAUTIOUS. Got "
        f"{reading.classification.name} with reason {reading.reason!r}."
    )
    assert "ob_widening" in (reading.reason or ""), (
        f"expected ob_widening in reason; got {reading.reason!r}"
    )
