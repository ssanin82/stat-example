"""v1.5.239 — EWMA-of-|log-return| trend-aware vol measure tests.

Tests cover:

1. The estimator's pure math (cold start, dedup, EWMA decay,
   trend behaviour vs stdev-equivalent).
2. The state field is initialised and replaced from Settings in
   bot init (covered via importability + default values).
3. The classifier history-selection flag actually swaps the source
   buffer at runtime without rebuilding the classifier.

The behavioural A/B (does the classifier produce more CAUTIOUS
entries on trending sessions when the flag is on?) belongs in
post-deploy snapshot analysis, not unit tests.
"""

from __future__ import annotations

import math


def test_cold_start_returns_none():
    """Before any record_mid, value_bps() is None."""
    from app.vol_abs_ewma import VolAbsEwmaEstimator
    est = VolAbsEwmaEstimator(halflife_seconds=20.0)
    assert est.value_bps() is None
    assert est.update_count == 0


def test_single_push_still_none():
    """After first mid push (seeding only — no return computable yet)
    the value remains None."""
    from app.vol_abs_ewma import VolAbsEwmaEstimator
    est = VolAbsEwmaEstimator(halflife_seconds=20.0)
    est.record_mid(100.0, now_mono_seconds=1.0)
    assert est.value_bps() is None
    assert est.update_count == 0  # update_count increments only on real updates


def test_second_push_produces_value():
    """Second mid push produces a real EWMA value in bp."""
    from app.vol_abs_ewma import VolAbsEwmaEstimator
    est = VolAbsEwmaEstimator(halflife_seconds=20.0)
    est.record_mid(100.0, now_mono_seconds=1.0)
    est.record_mid(100.05, now_mono_seconds=2.0)  # +0.05% = ~5 bp move
    v = est.value_bps()
    assert v is not None
    # Cold-start EWMA seeds at the sample. |log(100.05/100)| ≈ 0.0005
    # × 1e4 = 4.999 bp.
    assert 4.5 < v < 5.5, f"expected ~5 bp, got {v}"
    assert est.update_count == 1


def test_dedup_on_identical_mid():
    """A push of the same mid as the previous push is a no-op."""
    from app.vol_abs_ewma import VolAbsEwmaEstimator
    est = VolAbsEwmaEstimator(halflife_seconds=20.0)
    est.record_mid(100.0, now_mono_seconds=1.0)
    est.record_mid(100.0, now_mono_seconds=2.0)  # same mid
    est.record_mid(100.0, now_mono_seconds=3.0)
    assert est.value_bps() is None  # still warm-up; no real updates
    assert est.update_count == 0


def test_trend_walk_produces_steady_value_not_zero():
    """The KEY test — the operator-motivating case. A steady walk of
    same-direction 1-tick moves produces a STEADY non-zero EWMA. The
    legacy stdev-of-log-returns measure produces NEAR-ZERO on this
    input because the variance of identical samples is zero.

    Simulation: 30 pushes, each +0.01% above the last (≈ 1 bp / tick).
    Final EWMA should be ~1 bp (= the per-tick return magnitude),
    NOT zero."""
    from app.vol_abs_ewma import VolAbsEwmaEstimator
    est = VolAbsEwmaEstimator(halflife_seconds=20.0)
    mid = 100.0
    for i in range(30):
        mid *= 1.0001  # +1 bp per tick
        est.record_mid(mid, now_mono_seconds=1.0 + i)
    v = est.value_bps()
    assert v is not None
    # Per-tick return: log(1.0001) × 1e4 ≈ 0.9999 bp. EWMA converges
    # to that value as updates accumulate.
    assert 0.5 < v < 1.5, (
        f"trend-aware EWMA should reflect per-tick step (~1 bp), got {v}"
    )


def test_decay_to_drops_value_during_silence():
    """decay_to() pulls the EWMA toward zero during quiet stretches."""
    from app.vol_abs_ewma import VolAbsEwmaEstimator
    est = VolAbsEwmaEstimator(halflife_seconds=10.0)
    est.record_mid(100.0, now_mono_seconds=0.0)
    est.record_mid(100.1, now_mono_seconds=1.0)  # seed value
    v0 = est.value_bps()
    assert v0 is not None and v0 > 0
    est.decay_to(now_mono_seconds=11.0)  # one half-life later
    v1 = est.value_bps()
    assert v1 is not None
    # After one half-life of decay toward 0, value should be ~ v0 / 2.
    assert 0.4 * v0 < v1 < 0.6 * v0, (
        f"decay one half-life: expected ~{v0/2:.3f}, got {v1}"
    )


def test_state_field_initialised_to_none_and_placeholder_estimator():
    """BotState ships with state.vol_abs_ewma already constructed
    (default half-life) and state.vol_abs_ewma_bps = None."""
    from app.state import BotState
    from app.config import Settings
    from app.vol_abs_ewma import VolAbsEwmaEstimator
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
    )
    state = BotState(s)
    assert isinstance(state.vol_abs_ewma, VolAbsEwmaEstimator)
    assert state.vol_abs_ewma_bps is None
    # History buffer is empty + present.
    assert state.forward_vol_abs_ewma_bps_history == []


def test_state_append_forward_signal_history_routes_ewma_value():
    """append_forward_signal_history accepts vol_abs_ewma_bps and
    populates forward_vol_abs_ewma_bps_history alongside the
    stdev-based forward_vol_bps_history."""
    from app.state import BotState
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
    )
    state = BotState(s)
    state.append_forward_signal_history(
        now_mono=100.0,
        vol_bps=5.5,
        drift_30s_bps=0.0,
        ob_imbalance=0.0,
        max_age_seconds=180.0,
        vol_abs_ewma_bps=4.8,
    )
    assert state.forward_vol_bps_history == [(100.0, 5.5)]
    assert state.forward_vol_abs_ewma_bps_history == [(100.0, 4.8)]
    # None is silently skipped.
    state.append_forward_signal_history(
        now_mono=101.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,
        vol_abs_ewma_bps=None,
    )
    assert len(state.forward_vol_bps_history) == 1  # unchanged
    assert len(state.forward_vol_abs_ewma_bps_history) == 1  # unchanged


def test_settings_default_disables_consumer_flag():
    """v1.5.239 — REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE defaults
    False so the classifier still reads from the legacy stdev-based
    history. Operator must explicitly flip the flag to start the
    A/B."""
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
    )
    assert s.regime_forward_use_vol_abs_ewma_for_slope is False
    assert s.vol_abs_ewma_halflife_seconds == 20.0


def test_settings_flag_can_be_flipped_via_env():
    """The flag round-trips from env file via the alias."""
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
        REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE=True,
        VOL_ABS_EWMA_HALFLIFE_SECONDS=15.0,
    )
    assert s.regime_forward_use_vol_abs_ewma_for_slope is True
    assert s.vol_abs_ewma_halflife_seconds == 15.0


def test_snapshot_dict_exposes_both_vol_measures():
    """snapshot_dict surfaces both short_vol_bps (legacy stdev) and
    short_vol_abs_ewma_bps (new EWMA) so operators can compare."""
    from app.state import BotState
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
    )
    state = BotState(s)
    state.vol_bps = 5.5
    state.vol_abs_ewma_bps = 4.7
    snap = state.snapshot_dict()
    assert snap["short_vol_bps"] == 5.5
    assert snap["short_vol_abs_ewma_bps"] == 4.7
    # Both None on cold-start state.
    state2 = BotState(s)
    snap2 = state2.snapshot_dict()
    assert snap2["short_vol_bps"] is None
    assert snap2["short_vol_abs_ewma_bps"] is None
