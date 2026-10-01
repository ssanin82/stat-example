"""v1.5.205 Phase 4C.4 — stale-resting-quote penalty tests.

Covers:

1. ``compute_stale_risk_penalty_bps`` pure-function semantics
   (zero-cases, positive-cases, monotonicity).
2. ``compute_expected_net_edge_bps`` correctly subtracts the penalty.
3. ``evaluate_per_side_expected_edge_suppression`` and
   ``evaluate_per_side_dampen_band`` forward the penalty into the
   formula.
4. BotState rolling-P50 helper warmup + median math.
5. End-to-end: a stale quote crosses the refuse threshold that a
   fresh quote wouldn't.
"""

from __future__ import annotations

from collections import deque

import pytest

from app.expected_edge import (
    compute_expected_net_edge_bps,
    compute_stale_risk_penalty_bps,
    evaluate_per_side_dampen_band,
    evaluate_per_side_expected_edge_suppression,
)


# ------------------------ compute_stale_risk_penalty_bps ----------------- #


def test_penalty_zero_when_coefficient_zero() -> None:
    assert (
        compute_stale_risk_penalty_bps(
            current_quote_age_seconds=10.0,
            p50_quote_age_seconds=1.0,
            coeff_bps_per_sec=0.0,
        )
        == 0.0
    )


def test_penalty_zero_when_age_below_or_at_p50() -> None:
    # Below P50 → no penalty.
    assert (
        compute_stale_risk_penalty_bps(
            current_quote_age_seconds=0.5,
            p50_quote_age_seconds=1.0,
            coeff_bps_per_sec=1.0,
        )
        == 0.0
    )
    # Exactly at P50 → no penalty (lower-tail symmetry).
    assert (
        compute_stale_risk_penalty_bps(
            current_quote_age_seconds=1.0,
            p50_quote_age_seconds=1.0,
            coeff_bps_per_sec=1.0,
        )
        == 0.0
    )


def test_penalty_zero_when_no_current_age_or_no_p50() -> None:
    assert (
        compute_stale_risk_penalty_bps(
            current_quote_age_seconds=None,
            p50_quote_age_seconds=1.0,
            coeff_bps_per_sec=1.0,
        )
        == 0.0
    )
    assert (
        compute_stale_risk_penalty_bps(
            current_quote_age_seconds=5.0,
            p50_quote_age_seconds=None,
            coeff_bps_per_sec=1.0,
        )
        == 0.0
    )


def test_penalty_positive_above_p50() -> None:
    # 2.5s above P50 of 1.0 at 0.5 bp/s → 1.25 bp.
    assert compute_stale_risk_penalty_bps(
        current_quote_age_seconds=3.5,
        p50_quote_age_seconds=1.0,
        coeff_bps_per_sec=0.5,
    ) == pytest.approx(1.25)


def test_penalty_monotonic_in_age() -> None:
    p50 = 1.0
    coeff = 1.0
    a = compute_stale_risk_penalty_bps(
        current_quote_age_seconds=1.5,
        p50_quote_age_seconds=p50,
        coeff_bps_per_sec=coeff,
    )
    b = compute_stale_risk_penalty_bps(
        current_quote_age_seconds=2.0,
        p50_quote_age_seconds=p50,
        coeff_bps_per_sec=coeff,
    )
    c = compute_stale_risk_penalty_bps(
        current_quote_age_seconds=5.0,
        p50_quote_age_seconds=p50,
        coeff_bps_per_sec=coeff,
    )
    assert a < b < c


# ------------------------ compute_expected_net_edge_bps ------------------ #


def test_edge_subtracts_penalty() -> None:
    """edge = ths + rebate − adverse − penalty"""
    base = compute_expected_net_edge_bps(
        target_half_spread_bps=4.0,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
    )
    assert base == pytest.approx(3.0)
    with_penalty = compute_expected_net_edge_bps(
        target_half_spread_bps=4.0,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
        stale_risk_penalty_bps=1.5,
    )
    assert with_penalty == pytest.approx(1.5)


def test_edge_default_penalty_zero_no_change() -> None:
    """Caller can omit stale_risk_penalty_bps and behaviour is unchanged."""
    assert compute_expected_net_edge_bps(
        target_half_spread_bps=4.0,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
    ) == compute_expected_net_edge_bps(
        target_half_spread_bps=4.0,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
        stale_risk_penalty_bps=0.0,
    )


# ---------- evaluator forwards penalty into formula ----------------------- #


def test_refuse_evaluator_forwards_penalty() -> None:
    """A stale quote can cross the refuse threshold that a fresh
    quote wouldn't. Setup: refuse threshold = -1.0 bp; ths=3, rebate=1,
    adverse=2, so fresh edge = +2 bp. With penalty +4 the adjusted
    edge = -2 → below threshold → refuses."""
    fresh = evaluate_per_side_expected_edge_suppression(
        target_half_spread_bps=3.0,
        currently_refused=False,
        recovery_ticks=0,
        min_expected_edge_bps=-1.0,
        hysteresis_ticks=3,
        recovery_margin_bps=0.5,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
        stale_risk_penalty_bps=0.0,
    )
    assert fresh.refused is False
    assert fresh.expected_edge_bps == pytest.approx(2.0)

    stale = evaluate_per_side_expected_edge_suppression(
        target_half_spread_bps=3.0,
        currently_refused=False,
        recovery_ticks=0,
        min_expected_edge_bps=-1.0,
        hysteresis_ticks=3,
        recovery_margin_bps=0.5,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
        stale_risk_penalty_bps=4.0,
    )
    assert stale.refused is True
    assert stale.expected_edge_bps == pytest.approx(-2.0)
    assert stale.transition == "armed"


def test_dampen_evaluator_forwards_penalty() -> None:
    """With a stale penalty, an edge that was above the dampen ceiling
    can fall into the dampen band and trigger widening."""
    # Refuse at -2.0, dampen at -0.5. Edge without penalty = +2.0
    # (above dampen ceiling, no widening). With penalty +2.5 the
    # adjusted edge = -0.5 — at the dampen ceiling, just inside.
    dampened = evaluate_per_side_dampen_band(
        target_half_spread_bps=3.0,
        refuse_threshold_bps=-2.0,
        dampen_max_bps=-0.5,
        dampen_widen_bps=1.0,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
        stale_risk_penalty_bps=2.5,
    )
    assert dampened.armed is True
    assert dampened.widen_bps == pytest.approx(1.0)


# ------------------------ BotState rolling P50 --------------------------- #


def test_botstate_p50_returns_none_before_warmup(tmp_path) -> None:
    from app.state import BotState
    from tests.settings_helpers import UnitTestSettings

    settings = UnitTestSettings.model_validate(
        {
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'mm.db').as_posix()}",
        }
    )
    state = BotState(settings)
    # Push fewer than 20 samples — P50 returns None.
    for i in range(10):
        state.record_quote_age_decision_sample_seconds(float(i) * 0.5)
    assert state.quote_age_decision_p50_seconds() is None


def test_botstate_p50_median_math(tmp_path) -> None:
    from app.state import BotState
    from tests.settings_helpers import UnitTestSettings

    settings = UnitTestSettings.model_validate(
        {
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'mm.db').as_posix()}",
        }
    )
    state = BotState(settings)
    # Push known sequence; median should be 10.0 for 21 samples
    # (0,1,...,20).
    for i in range(21):
        state.record_quote_age_decision_sample_seconds(float(i))
    p50 = state.quote_age_decision_p50_seconds()
    assert p50 == pytest.approx(10.0)


def test_botstate_p50_window_evicts_oldest(tmp_path) -> None:
    """maxlen=200 caps the deque; older samples drop off."""
    from app.state import BotState
    from tests.settings_helpers import UnitTestSettings

    settings = UnitTestSettings.model_validate(
        {
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'mm.db').as_posix()}",
        }
    )
    state = BotState(settings)
    # Push 100 samples of value 100, then 200 samples of value 1.
    for _ in range(100):
        state.record_quote_age_decision_sample_seconds(100.0)
    for _ in range(200):
        state.record_quote_age_decision_sample_seconds(1.0)
    # First 100 should have been evicted (maxlen=200 — only the
    # last 200 remain, which are all = 1.0).
    p50 = state.quote_age_decision_p50_seconds()
    assert p50 == pytest.approx(1.0)


def test_botstate_drops_non_finite_samples(tmp_path) -> None:
    from app.state import BotState
    from tests.settings_helpers import UnitTestSettings

    settings = UnitTestSettings.model_validate(
        {
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'mm.db').as_posix()}",
        }
    )
    state = BotState(settings)
    state.record_quote_age_decision_sample_seconds(float("nan"))
    state.record_quote_age_decision_sample_seconds(float("inf"))
    state.record_quote_age_decision_sample_seconds(-1.0)  # negative → drop
    assert len(state.quote_age_decision_samples_seconds) == 0
