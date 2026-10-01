"""v1.5.209 — Phase 8B Queue-position-aware sizing + inside-post tests.

Covers:
* ``estimate_queue_position_ratio`` — proxy bounds, None on degenerate
  inputs
* ``expected_wait_seconds`` — happy path, divide-by-zero guard
* ``queue_aware_size_multiplier`` — at ratio=0/0.5/1, None default,
  clamp at floor
* ``should_post_inside_spread`` — flag gate, threshold, None signal
* ``ArrivalRateEwma`` — record_trade, per-side independence,
  decay_to
* Integration with ``compute_quote_decision`` — flag-off no-op,
  flag-on shrink + inside-post applied + emitted in breakdown
"""

from __future__ import annotations

import pytest

from app.queue_model import (
    ArrivalRateEwma,
    estimate_queue_position_ratio,
    expected_wait_seconds,
    queue_aware_size_multiplier,
    should_post_inside_spread,
)


# ───────────────────── estimate_queue_position_ratio ────────────────────


def test_position_ratio_bot_only_no_queue_ahead():
    """Bot's own size IS the entire queue — ratio = 0 (nothing ahead)."""
    r = estimate_queue_position_ratio(own_resting_size=10.0, inside_total_size=10.0)
    assert r == pytest.approx(0.0)


def test_position_ratio_bot_half_of_queue():
    r = estimate_queue_position_ratio(own_resting_size=5.0, inside_total_size=10.0)
    assert r == pytest.approx(0.5)


def test_position_ratio_bot_tiny_fraction():
    r = estimate_queue_position_ratio(own_resting_size=1.0, inside_total_size=100.0)
    assert r == pytest.approx(0.99)


def test_position_ratio_zero_total_returns_none():
    assert estimate_queue_position_ratio(
        own_resting_size=0.0, inside_total_size=0.0
    ) is None


def test_position_ratio_clipped_at_one():
    """Own > total (shouldn't happen but guard against it)."""
    r = estimate_queue_position_ratio(own_resting_size=15.0, inside_total_size=10.0)
    assert r == pytest.approx(0.0)  # own clamped to total → queue_ahead=0


def test_position_ratio_nan_returns_none():
    assert estimate_queue_position_ratio(
        own_resting_size=float("nan"), inside_total_size=10.0
    ) is None


# ────────────────────────── expected_wait_seconds ───────────────────────


def test_expected_wait_happy_path():
    w = expected_wait_seconds(queue_ahead_size=20.0, arrival_rate_per_sec=2.0)
    assert w == pytest.approx(10.0)


def test_expected_wait_zero_ahead():
    w = expected_wait_seconds(queue_ahead_size=0.0, arrival_rate_per_sec=2.0)
    assert w == 0.0


def test_expected_wait_zero_rate_returns_none():
    assert expected_wait_seconds(
        queue_ahead_size=10.0, arrival_rate_per_sec=0.0
    ) is None


def test_expected_wait_negative_ahead_treated_as_zero():
    w = expected_wait_seconds(queue_ahead_size=-5.0, arrival_rate_per_sec=2.0)
    assert w == 0.0


# ───────────────────── queue_aware_size_multiplier ──────────────────────


def test_size_mult_ratio_zero_no_shrink():
    assert queue_aware_size_multiplier(queue_position_ratio=0.0) == pytest.approx(1.0)


def test_size_mult_ratio_half_default():
    """ratio=0.5 → 1 - 0.7*0.5 = 0.65."""
    assert queue_aware_size_multiplier(queue_position_ratio=0.5) == pytest.approx(0.65)


def test_size_mult_ratio_one_clamps_to_floor():
    """ratio=1 → 1 - 0.7 = 0.3 (matches floor default)."""
    assert queue_aware_size_multiplier(queue_position_ratio=1.0) == pytest.approx(0.3)


def test_size_mult_none_signal_no_shrink():
    assert queue_aware_size_multiplier(queue_position_ratio=None) == 1.0


def test_size_mult_custom_floor_and_decay():
    """Steeper decay floors out earlier."""
    m = queue_aware_size_multiplier(
        queue_position_ratio=0.5, floor=0.5, decay=1.0,
    )
    # 1 - 1.0*0.5 = 0.5 → at floor exactly
    assert m == pytest.approx(0.5)


def test_size_mult_clamps_above_one():
    """Defensive — negative ratio shouldn't make mult > 1."""
    m = queue_aware_size_multiplier(queue_position_ratio=-0.5)
    assert m <= 1.0


# ───────────────────── should_post_inside_spread ────────────────────────


def test_inside_post_disabled_flag_off():
    assert should_post_inside_spread(
        queue_position_ratio=0.9, threshold=0.7, enabled=False,
    ) is False


def test_inside_post_signal_none_skips():
    assert should_post_inside_spread(
        queue_position_ratio=None, threshold=0.7, enabled=True,
    ) is False


def test_inside_post_below_threshold_skips():
    assert should_post_inside_spread(
        queue_position_ratio=0.5, threshold=0.7, enabled=True,
    ) is False


def test_inside_post_at_threshold_fires():
    assert should_post_inside_spread(
        queue_position_ratio=0.7, threshold=0.7, enabled=True,
    ) is True


def test_inside_post_above_threshold_fires():
    assert should_post_inside_spread(
        queue_position_ratio=0.85, threshold=0.7, enabled=True,
    ) is True


# ─────────────────────────── ArrivalRateEwma ────────────────────────────


def test_arrival_rate_first_print_seeds():
    ewma = ArrivalRateEwma(halflife_seconds=10.0)
    ewma.record_trade(side="bid", qty=5.0, now_mono_seconds=0.0)
    assert ewma.bid_arrival_rate_per_sec == pytest.approx(5.0)
    assert ewma.ask_arrival_rate_per_sec is None
    assert ewma.update_count_bid == 1


def test_arrival_rate_per_side_independent():
    ewma = ArrivalRateEwma(halflife_seconds=10.0)
    ewma.record_trade(side="bid", qty=3.0, now_mono_seconds=0.0)
    ewma.record_trade(side="ask", qty=7.0, now_mono_seconds=0.0)
    assert ewma.bid_arrival_rate_per_sec == pytest.approx(3.0)
    assert ewma.ask_arrival_rate_per_sec == pytest.approx(7.0)


def test_arrival_rate_decay_to_drops_idle_rate():
    ewma = ArrivalRateEwma(halflife_seconds=5.0)
    ewma.record_trade(side="bid", qty=10.0, now_mono_seconds=0.0)
    assert ewma.bid_arrival_rate_per_sec == pytest.approx(10.0)
    ewma.decay_to(5.0)  # one halflife later → rate halved
    assert ewma.bid_arrival_rate_per_sec == pytest.approx(5.0)


def test_arrival_rate_rejects_invalid_inputs():
    ewma = ArrivalRateEwma()
    ewma.record_trade(side="bid", qty=-1.0, now_mono_seconds=0.0)
    ewma.record_trade(side="weird", qty=5.0, now_mono_seconds=0.0)
    ewma.record_trade(side="bid", qty=float("nan"), now_mono_seconds=0.0)
    assert ewma.update_count_bid == 0


# ─────────────── Integration with compute_quote_decision ────────────────


def _make_settings(
    queue_sizing: bool = False,
    queue_inside: bool = False,
):
    from app.config import Settings
    return Settings(
        VENUE="binance",
        SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0,
        MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
        QUEUE_AWARE_SIZING_ENABLED=queue_sizing,
        QUEUE_AWARE_INSIDE_POST_ENABLED=queue_inside,
        TOXICITY_ENABLED=False,
    )


def test_decision_queue_sizing_off_no_shrink():
    from app.quoting import compute_quote_decision
    from app.toxicity import ToxicitySnapshot

    settings = _make_settings(queue_sizing=False)
    decision = compute_quote_decision(
        settings=settings,
        mid=100.0, position_qty=0.0, vol_bps=5.0,
        toxicity=ToxicitySnapshot(score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0, vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False),
        queue_position_ratio_bid=0.9,  # would shrink if armed
        queue_position_ratio_ask=0.9,
    )
    bd = decision.breakdown
    assert bd.queue_size_mult_bid == 1.0
    assert bd.queue_size_mult_ask == 1.0


def test_decision_queue_sizing_on_applies_shrink():
    from app.quoting import compute_quote_decision
    from app.toxicity import ToxicitySnapshot

    settings = _make_settings(queue_sizing=True)
    decision = compute_quote_decision(
        settings=settings,
        mid=100.0, position_qty=0.0, vol_bps=5.0,
        toxicity=ToxicitySnapshot(score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0, vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False),
        queue_position_ratio_bid=0.5,  # → 0.65 mult expected
        queue_position_ratio_ask=0.0,
    )
    bd = decision.breakdown
    assert bd.queue_size_mult_bid == pytest.approx(0.65)
    assert bd.queue_size_mult_ask == pytest.approx(1.0)
    # Bid size should be smaller than ask size (asymmetric shrink).
    assert decision.quoted_bid_sz < decision.quoted_ask_sz


def test_decision_queue_inside_post_off_no_narrow():
    from app.quoting import compute_quote_decision
    from app.toxicity import ToxicitySnapshot

    settings = _make_settings(queue_inside=False)
    decision = compute_quote_decision(
        settings=settings, mid=100.0, position_qty=0.0, vol_bps=5.0,
        toxicity=ToxicitySnapshot(score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0, vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False),
        queue_position_ratio_bid=0.9,
        queue_position_ratio_ask=0.9,
    )
    bd = decision.breakdown
    assert bd.queue_inside_post_step_bid_bps == 0.0
    assert bd.queue_inside_post_step_ask_bps == 0.0


def test_decision_queue_inside_post_on_narrows_when_over_threshold():
    from app.quoting import compute_quote_decision
    from app.toxicity import ToxicitySnapshot

    settings = _make_settings(queue_inside=True)
    decision = compute_quote_decision(
        settings=settings, mid=100.0, position_qty=0.0, vol_bps=5.0,
        toxicity=ToxicitySnapshot(score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0, vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False),
        queue_position_ratio_bid=0.9,  # over default 0.7 → inside-post fires
        queue_position_ratio_ask=0.3,  # below threshold → no narrow
    )
    bd = decision.breakdown
    assert bd.queue_inside_post_step_bid_bps > 0.0
    assert bd.queue_inside_post_step_ask_bps == 0.0


def test_decision_queue_telemetry_emits_ratios():
    from app.quoting import compute_quote_decision
    from app.toxicity import ToxicitySnapshot

    settings = _make_settings(queue_sizing=False, queue_inside=False)
    decision = compute_quote_decision(
        settings=settings, mid=100.0, position_qty=0.0, vol_bps=5.0,
        toxicity=ToxicitySnapshot(score=0.0, one_sided_fill_ratio=0.5, avg_adverse_markout_bps=0.0, vol_spike_ratio=0.0, hard_trigger=False, soft_trigger=False),
        queue_position_ratio_bid=0.42,
        queue_position_ratio_ask=0.13,
    )
    bd = decision.breakdown
    assert bd.queue_position_ratio_bid == pytest.approx(0.42)
    assert bd.queue_position_ratio_ask == pytest.approx(0.13)
