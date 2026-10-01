"""Quote aging / distance-to-touch policy (pure helpers + WorkingOrder age)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.enums import ActiveSides, OrderStatus, RiskAction, Side, TouchPlacementMode
from app.models import QuoteDecision, WorkingOrder
from app.quote_aging import (
    adjust_ask_target_for_aging,
    adjust_bid_target_for_aging,
    at_touch_bid,
    compute_normal_mm_market_capped_half_spread_bps,
    distance_buy_to_touch_ticks,
    distance_sell_to_touch_ticks,
    finalize_ask_placement_touch_distance,
    finalize_buy_placement_touch_distance,
    inventory_pressure_active,
    preserve_ask_queue,
    preserve_bid_queue,
    touch_placement_mode_for_side,
)
from app.utils.time import utc_now
from tests.settings_helpers import UnitTestSettings


def _decision(mid: float = 100.0) -> QuoteDecision:
    return QuoteDecision(
        ts=utc_now(),
        symbol="ETH",
        mid_price=mid,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=mid,
        target_spread_bps=16.0,
        target_bid=mid * 0.999,
        target_ask=mid * 1.001,
        quoted_bid=mid * 0.999,
        quoted_ask=mid * 1.001,
        quoted_bid_sz=0.1,
        quoted_ask_sz=0.1,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="ok",
        quote_cycle_id="q1",
    )


def test_resting_age_seconds_on_working_order() -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=99.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_ack=t0,
    )
    assert wo.resting_age_seconds(t0 + timedelta(seconds=2.5)) == 2.5
    assert wo.resting_age_seconds(t0) == 0.0
    wo2 = WorkingOrder(
        order_id_local="y",
        order_id_exchange=None,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=99.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.SENT,
    )
    assert wo2.resting_age_seconds(utc_now()) is None


def test_distance_and_at_touch() -> None:
    assert distance_buy_to_touch_ticks(99.0, 100.0, 0.01) == 100.0
    assert at_touch_bid(100.0, 100.0, 0.01) is True
    assert at_touch_bid(99.995, 100.0, 0.01) is True


def test_adjust_bid_tightens_when_far_and_old() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_AGING_MAX_AGE_SECONDS": 2.0,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 2.0,
            "QUOTE_AGING_TIGHTEN_TICKS": 1.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(100.0)
    now = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    # One tick inside band: best 100, max_d=2 → floor 99.98; working 99.97 is 3 ticks back.
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=99.97,
        size=0.1,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_ack=now - timedelta(seconds=5.0),
    )
    raw, diag = adjust_bid_target_for_aging(
        s,
        d,
        bid_px_model=99.97,
        best_bid=100.0,
        tick=0.01,
        working=wo,
        now=now,
        position_qty=0.0,
    )
    assert diag.tighten_applied is True
    assert diag.aging_escalate_reprice == ()
    assert abs(raw - 99.98) < 1e-9
    assert raw <= 100.0 + 1e-9


def test_large_distance_does_not_cosmetic_tighten_escalates() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_AGING_MAX_AGE_SECONDS": 2.0,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 2.0,
            "QUOTE_AGING_TIGHTEN_TICKS": 1.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(100.0)
    now = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=99.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_ack=now - timedelta(seconds=0.5),
    )
    raw, diag = adjust_bid_target_for_aging(
        s,
        d,
        bid_px_model=99.0,
        best_bid=100.0,
        tick=0.01,
        working=wo,
        now=now,
        position_qty=0.0,
    )
    assert diag.tighten_applied is False
    assert raw == 99.0
    assert "distance_ticks" in diag.reasons
    assert diag.aging_escalate_reprice == ("exceeds_safe_aging_step",)


def test_age_only_many_ticks_to_touch_escalates() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_AGING_MAX_AGE_SECONDS": 2.0,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 50.0,
            "QUOTE_AGING_TIGHTEN_TICKS": 1.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(100.0)
    now = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=99.90,
        size=0.1,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_ack=now - timedelta(seconds=10.0),
    )
    raw, diag = adjust_bid_target_for_aging(
        s,
        d,
        bid_px_model=99.90,
        best_bid=100.0,
        tick=0.01,
        working=wo,
        now=now,
        position_qty=0.0,
    )
    assert "age_seconds" in diag.reasons
    assert "distance_ticks" not in diag.reasons
    assert diag.tighten_applied is False
    assert raw == 99.90
    assert diag.aging_escalate_reprice == ("exceeds_safe_aging_step",)


def test_escalation_merges_with_hard_reprice_reasons() -> None:
    from app.quote_aging import hard_reprice_reasons_buy

    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_AGING_MAX_AGE_SECONDS": 2.0,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 2.0,
            "QUOTE_AGING_TIGHTEN_TICKS": 1.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(100.0)
    now = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=99.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_ack=now - timedelta(seconds=0.5),
    )
    _, diag = adjust_bid_target_for_aging(
        s,
        d,
        bid_px_model=99.0,
        best_bid=100.0,
        tick=0.01,
        working=wo,
        now=now,
        position_qty=0.0,
    )
    hr = hard_reprice_reasons_buy(s, wo, best_bid=100.0, tick=0.01, now=now)
    merged = tuple(dict.fromkeys((*hr, *diag.aging_escalate_reprice)))
    assert "distance_to_touch_ticks" in merged
    assert "exceeds_safe_aging_step" in merged


def test_preserve_bid_queue_at_touch_no_aging_reason() -> None:
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    assert (
        preserve_bid_queue(
            wo,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=100.0,
            bid_px_model=99.95,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is True
    )


def test_preserve_false_when_aging_reason() -> None:
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    assert (
        preserve_bid_queue(
            wo,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=100.0,
            bid_px_model=99.95,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=("age_seconds",),
        )
        is False
    )


def _wo_sell(price: float, *, status: OrderStatus = OrderStatus.ACKED) -> WorkingOrder:
    return WorkingOrder(
        order_id_local="s",
        order_id_exchange=2,
        client_order_id=None,
        symbol="ETH",
        side=Side.SELL,
        price=price,
        size=0.1,
        post_only=True,
        status=status,
    )


def _wo_buy(price: float, *, status: OrderStatus = OrderStatus.ACKED) -> WorkingOrder:
    return WorkingOrder(
        order_id_local="b",
        order_id_exchange=3,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=price,
        size=0.1,
        post_only=True,
        status=status,
    )


def test_preserve_ask_queue_true_when_all_gates_pass() -> None:
    wo = _wo_sell(100.0)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=100.0,
            ask_px_model=100.05,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is True
    )


def test_preserve_ask_queue_true_when_partial_status() -> None:
    wo = _wo_sell(100.0, status=OrderStatus.PARTIAL)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=100.0,
            ask_px_model=100.02,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is True
    )


def test_preserve_ask_queue_true_when_model_far_from_working_price() -> None:
    """Model can sit far from touch under skew; must not block preservation alone."""
    wo = _wo_sell(100.0)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=100.0,
            ask_px_model=135.0,
            mid=100.0,
            reprice_threshold_bps=5.0,
            aging_reasons=(),
        )
        is True
    )


def test_preserve_ask_queue_false_when_wo_none() -> None:
    assert (
        preserve_ask_queue(
            None,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=100.0,
            ask_px_model=100.0,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_ask_queue_false_when_status_not_resting() -> None:
    wo = _wo_sell(100.0, status=OrderStatus.SENT)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=100.0,
            ask_px_model=100.0,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_ask_queue_false_when_aging_reasons_non_empty() -> None:
    wo = _wo_sell(100.0)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=100.0,
            ask_px_model=100.0,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=("order_age_seconds",),
        )
        is False
    )


def test_preserve_ask_queue_false_when_not_at_touch() -> None:
    wo = _wo_sell(100.10)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=100.10,
            ask_px_model=100.10,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_ask_queue_false_when_ask_target_norm_none() -> None:
    wo = _wo_sell(100.0)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=None,
            ask_px_model=100.0,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_ask_queue_false_when_mid_non_positive() -> None:
    wo = _wo_sell(100.0)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=100.0,
            tick=0.01,
            ask_target_norm=100.0,
            ask_px_model=100.0,
            mid=0.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_bid_queue_true_when_partial_status() -> None:
    wo = _wo_buy(100.0, status=OrderStatus.PARTIAL)
    assert (
        preserve_bid_queue(
            wo,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=100.0,
            bid_px_model=99.98,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is True
    )


def test_preserve_bid_queue_true_when_model_far_from_working_price() -> None:
    wo = _wo_buy(100.0)
    assert (
        preserve_bid_queue(
            wo,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=100.0,
            bid_px_model=65.0,
            mid=100.0,
            reprice_threshold_bps=5.0,
            aging_reasons=(),
        )
        is True
    )


def test_preserve_bid_queue_false_when_wo_none() -> None:
    assert (
        preserve_bid_queue(
            None,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=100.0,
            bid_px_model=100.0,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_bid_queue_false_when_status_not_resting() -> None:
    wo = _wo_buy(100.0, status=OrderStatus.CANCELED)
    assert (
        preserve_bid_queue(
            wo,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=100.0,
            bid_px_model=100.0,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_bid_queue_false_when_not_at_touch() -> None:
    wo = _wo_buy(99.90)
    assert (
        preserve_bid_queue(
            wo,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=99.90,
            bid_px_model=99.90,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_bid_queue_false_when_bid_target_norm_none() -> None:
    wo = _wo_buy(100.0)
    assert (
        preserve_bid_queue(
            wo,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=None,
            bid_px_model=100.0,
            mid=100.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_bid_queue_false_when_mid_non_positive() -> None:
    wo = _wo_buy(100.0)
    assert (
        preserve_bid_queue(
            wo,
            best_bid=100.0,
            tick=0.01,
            bid_target_norm=100.0,
            bid_px_model=100.0,
            mid=-1.0,
            reprice_threshold_bps=10.0,
            aging_reasons=(),
        )
        is False
    )


def test_preserve_bid_queue_regression_skew_guard_divergence_still_preserves() -> None:
    """At-touch bid with huge model vs touch gap (skew / guard-shaped model) must preserve."""
    wo = _wo_buy(3000.0)
    assert (
        preserve_bid_queue(
            wo,
            best_bid=3000.0,
            tick=0.01,
            bid_target_norm=3000.0,
            bid_px_model=2850.0,
            mid=3000.0,
            reprice_threshold_bps=1.0,
            aging_reasons=(),
        )
        is True
    )


def test_preserve_ask_queue_regression_skew_guard_divergence_still_preserves() -> None:
    """Symmetric: at-touch ask with large model divergence still preserves."""
    wo = _wo_sell(3000.0)
    assert (
        preserve_ask_queue(
            wo,
            best_ask=3000.0,
            tick=0.01,
            ask_target_norm=3000.0,
            ask_px_model=3150.0,
            mid=3000.0,
            reprice_threshold_bps=1.0,
            aging_reasons=(),
        )
        is True
    )


def test_inventory_pressure_active() -> None:
    s = UnitTestSettings.model_validate(
        {
            "MAX_ABS_POSITION": 0.1,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.5,
            "INVENTORY_FAIR_TOUCH_RELAX_MIN_UTIL_PCT": 0.0,
        }
    )
    assert inventory_pressure_active(s, 0.06) is True
    assert inventory_pressure_active(s, 0.04) is False


def test_inventory_pressure_uses_max_of_quote_threshold_and_relax_floor() -> None:
    s = UnitTestSettings.model_validate(
        {
            "MAX_ABS_POSITION": 1.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.4,
            "INVENTORY_FAIR_TOUCH_RELAX_MIN_UTIL_PCT": 0.55,
        }
    )
    # 50% util: below floor-only threshold
    assert inventory_pressure_active(s, 0.5) is False
    # 56% util: meets max(0.4, 0.55)
    assert inventory_pressure_active(s, 0.56) is True


def test_finalize_buy_skips_when_fair_cap_cannot_reach_max_touch_proximity() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(200.0)
    d.reservation_price = 100.0
    px, skip, tag, _ = finalize_buy_placement_touch_distance(
        s,
        d,
        90.0,
        None,
        best_bid=200.0,
        tick=1.0,
        position_qty=0.0,
    )
    assert px is None and skip == "distance_guard" and tag == ""


def test_adjust_ask_new_order_pulls_down_to_band_ceiling_symmetric_to_bid() -> None:
    """Far-above model ask uses min(ref, band_ceil), not max(floor, band_ceil)."""
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 5.0,
            "QUOTE_AGING_TIGHTEN_TICKS": 50.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(mid=3000.0)
    d.reservation_price = 3000.0
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    raw, diag = adjust_ask_target_for_aging(
        s,
        d,
        ask_px_model=3010.0,
        best_ask=3000.0,
        tick=1.0,
        working=None,
        now=now,
        position_qty=0.0,
    )
    assert "distance_ticks_new_order" in diag.reasons
    assert raw == 3005.0
    assert distance_sell_to_touch_ticks(raw, 3000.0, 1.0) == 5.0


def test_finalize_ask_symmetric_rescue_when_fair_floor_pins_far() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 5.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(mid=3000.0)
    d.reservation_price = 3200.0
    px, skip, tag, _ = finalize_ask_placement_touch_distance(
        s,
        d,
        3500.0,
        3500.0,
        best_ask=3000.0,
        tick=1.0,
        position_qty=0.0,
        best_bid=2999.0,
        symmetric_two_sided_rescue=True,
    )
    assert skip is None and tag == "symmetric_rescue"
    assert px == 3005.0
    assert distance_sell_to_touch_ticks(px, 3000.0, 1.0) == 5.0


def test_finalize_ask_no_symmetric_rescue_on_crossed_book() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 5.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(mid=3000.0)
    d.reservation_price = 3200.0
    px, skip, tag, _ = finalize_ask_placement_touch_distance(
        s,
        d,
        3500.0,
        3500.0,
        best_ask=3000.0,
        tick=1.0,
        position_qty=0.0,
        best_bid=3000.0,
        symmetric_two_sided_rescue=True,
    )
    assert px is None and skip == "distance_guard" and tag == ""


def test_finalize_symmetric_rescue_widens_vs_mid_when_min_half_spread_px_set() -> None:
    """Symmetric two-sided rescue cannot sit more aggressive than mid - half (economic floor)."""
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "POST_ONLY_TOUCH_BUFFER_TICKS": 1,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(mid=200.0)
    d.reservation_price = 80.0
    min_h = 2.0
    px, skip, tag, _ = finalize_buy_placement_touch_distance(
        s,
        d,
        raw_px=199.0,
        norm_px=199.0,
        best_bid=200.0,
        tick=1.0,
        position_qty=0.0,
        best_ask=201.0,
        symmetric_two_sided_rescue=True,
        min_half_spread_px=min_h,
        spread_floor_mid_px=200.0,
    )
    assert skip is None and tag == "symmetric_rescue"
    assert px is not None
    assert px == 198.0
    assert px <= 200.0 - min_h + 1e-9


def test_finalize_normal_two_sided_mm_preserves_wide_passive_bid() -> None:
    """Finalize does not pull toward touch; wide passive bid stays when spread contract allows it."""
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "POST_ONLY_TOUCH_BUFFER_TICKS": 1,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
            "NORMAL_MM_MAX_DISTANCE_TO_TOUCH_TICKS": 500.0,
        }
    )
    d = _decision(mid=3000.0)
    d.reservation_price = 3000.0
    px, skip, tag, diag = finalize_buy_placement_touch_distance(
        s,
        d,
        raw_px=2950.0,
        norm_px=2950.0,
        best_bid=3000.0,
        tick=1.0,
        position_qty=0.0,
        best_ask=3001.0,
        min_half_spread_px=0.5,
        spread_floor_mid_px=3000.5,
        placement_mode=TouchPlacementMode.NORMAL_TWO_SIDED_MM,
    )
    assert skip is None and tag == ""
    assert px == 2950.0
    assert diag["touch_band_rule_applied"] is False
    assert diag["touch_band_rule_reason"] == "normal_two_sided_mm_direct"
    assert diag["placement_mode"] == TouchPlacementMode.NORMAL_TWO_SIDED_MM.value


def test_finalize_normal_two_sided_mm_direct_no_touch_band_when_already_near_touch() -> None:
    """Touch distance is enforced upstream via capped half-spread; finalize only post-only + min-half vs mid."""
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "POST_ONLY_TOUCH_BUFFER_TICKS": 1,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
            "NORMAL_MM_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
        }
    )
    d = _decision(mid=3000.0)
    d.reservation_price = 3000.0
    # Bid already at mid - half consistent with capped spread (~3.5 px half -> 2997 bid vs mid 3000.5).
    px, skip, tag, diag = finalize_buy_placement_touch_distance(
        s,
        d,
        raw_px=2997.0,
        norm_px=2997.0,
        best_bid=3000.0,
        tick=1.0,
        position_qty=0.0,
        best_ask=3001.0,
        min_half_spread_px=3.5,
        spread_floor_mid_px=3000.5,
        placement_mode=TouchPlacementMode.NORMAL_TWO_SIDED_MM,
    )
    assert skip is None and tag == ""
    assert px == 2997.0
    assert diag["final_distance_to_touch_ticks_bid"] == 3.0
    assert diag["touch_band_rule_applied"] is False
    assert diag["touch_band_rule_reason"] == "normal_two_sided_mm_direct"
    assert diag["post_only_adjustment_applied"] is False
    assert diag["economic_floor_requested"] is False
    assert diag["economic_floor_blocked_by_post_only"] is False


def test_finalize_normal_two_sided_mm_records_economic_floor_request() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "POST_ONLY_TOUCH_BUFFER_TICKS": 1,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
            "NORMAL_MM_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
        }
    )
    d = _decision(mid=3000.0)
    d.reservation_price = 3000.0
    px, skip, tag, diag = finalize_buy_placement_touch_distance(
        s,
        d,
        raw_px=3000.0,
        norm_px=3000.0,
        best_bid=3000.0,
        tick=1.0,
        position_qty=0.0,
        best_ask=3001.0,
        min_half_spread_px=3.5,
        spread_floor_mid_px=3000.5,
        placement_mode=TouchPlacementMode.NORMAL_TWO_SIDED_MM,
    )
    assert skip is None and tag == ""
    assert px == 2997.0
    assert diag["economic_floor_requested"] is True
    assert diag["economic_floor_blocked_by_post_only"] is False
    assert diag["touch_band_rule_reason"] == "normal_two_sided_mm_direct"


def test_compute_normal_mm_touch_distance_cap_can_tighten_below_market_anchor() -> None:
    """Touch-geometry cap can bind below the market-anchor max when the book is tight vs min-competitive."""
    s = UnitTestSettings.model_validate(
        {
            "MIN_HALF_SPREAD_BPS": 0.5,
            "MAX_HALF_SPREAD_BPS": 200.0,
            "NORMAL_MM_TOUCH_BUFFER_TICKS": 0.0,
            "NORMAL_MM_MIN_COMPETITIVE_HALF_SPREAD_BPS": 5.0,
            "NORMAL_MM_MAX_COMPETITIVE_HALF_SPREAD_BPS": 0.0,
            "NORMAL_MM_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
        }
    )
    mid = 100.01
    tick = 0.01
    best_bid = 100.0
    best_ask = 100.02
    capped, diag = compute_normal_mm_market_capped_half_spread_bps(
        s,
        model_half_spread_bps=80.0,
        best_bid=best_bid,
        best_ask=best_ask,
        mid=mid,
        tick=tick,
    )
    assert diag["normal_mm_touch_distance_cap_applied"] is True
    assert capped + 1e-9 < 80.0
    # cap_px = min(0.01+0.03, 0.01+0.03) = 0.04 -> ~3.996 bps
    assert abs(capped - 0.04 / mid * 10_000.0) < 0.02
    assert diag["quoting_regime_reason"] == "touch_distance_cap"


def test_touch_placement_mode_for_side_risk_action_matrix() -> None:
    assert (
        touch_placement_mode_for_side(
            risk_action=RiskAction.ALLOW,
            active_sides=ActiveSides.BOTH,
            two_sided_quote=True,
            book_fresh_for_placement=True,
            skip_new_place_inventory_side=False,
            normal_mm_contract_active=True,
        )
        == TouchPlacementMode.NORMAL_TWO_SIDED_MM
    )
    assert (
        touch_placement_mode_for_side(
            risk_action=RiskAction.ALLOW,
            active_sides=ActiveSides.BOTH,
            two_sided_quote=True,
            book_fresh_for_placement=True,
            skip_new_place_inventory_side=False,
            normal_mm_contract_active=False,
        )
        == TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION
    )
    assert (
        touch_placement_mode_for_side(
            risk_action=RiskAction.ALLOW,
            active_sides=ActiveSides.BOTH,
            two_sided_quote=True,
            book_fresh_for_placement=False,
            skip_new_place_inventory_side=False,
            normal_mm_contract_active=True,
        )
        == TouchPlacementMode.DEGRADED_FALLBACK
    )
    assert (
        touch_placement_mode_for_side(
            risk_action=RiskAction.BID_ONLY,
            active_sides=ActiveSides.BOTH,
            two_sided_quote=True,
            book_fresh_for_placement=True,
            skip_new_place_inventory_side=False,
            normal_mm_contract_active=True,
        )
        == TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION
    )
    assert (
        touch_placement_mode_for_side(
            risk_action=RiskAction.ASK_ONLY,
            active_sides=ActiveSides.BOTH,
            two_sided_quote=True,
            book_fresh_for_placement=True,
            skip_new_place_inventory_side=False,
            normal_mm_contract_active=True,
        )
        == TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION
    )
    assert (
        touch_placement_mode_for_side(
            risk_action=RiskAction.NO_QUOTE,
            active_sides=ActiveSides.NONE,
            two_sided_quote=False,
            book_fresh_for_placement=True,
            skip_new_place_inventory_side=False,
            normal_mm_contract_active=True,
        )
        == TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION
    )
    for emergency_ra in (RiskAction.CANCEL_ALL, RiskAction.FLATTEN, RiskAction.KILL):
        assert (
            touch_placement_mode_for_side(
                risk_action=emergency_ra,
                active_sides=ActiveSides.BOTH,
                two_sided_quote=True,
                book_fresh_for_placement=True,
                skip_new_place_inventory_side=False,
                normal_mm_contract_active=True,
            )
            == TouchPlacementMode.EMERGENCY
        )


def test_compute_normal_mm_caps_model_when_wider_than_market_anchor() -> None:
    s = UnitTestSettings.model_validate(
        {
            "MIN_HALF_SPREAD_BPS": 0.5,
            "MAX_HALF_SPREAD_BPS": 200.0,
            "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 0.5,
            "NORMAL_MM_TOUCH_BUFFER_TICKS": 1.0,
            "NORMAL_MM_MIN_COMPETITIVE_HALF_SPREAD_BPS": 0.5,
            "NORMAL_MM_MAX_COMPETITIVE_HALF_SPREAD_BPS": 0.0,
        }
    )
    mid = 100.0
    tick = 0.01
    best_bid = 99.99
    best_ask = 100.01
    capped, diag = compute_normal_mm_market_capped_half_spread_bps(
        s,
        model_half_spread_bps=80.0,
        best_bid=best_bid,
        best_ask=best_ask,
        mid=mid,
        tick=tick,
    )
    assert diag["was_market_spread_cap_applied"] is True
    assert capped < 80.0
    assert diag["live_market_spread_bps"] is not None
    assert diag["quoting_regime_reason"] == "market_anchor_cap"


def test_compute_normal_mm_one_sided_mode_unchanged_when_not_used() -> None:
    """compute_normal_mm is only called from execution when anchor applies; helper is pure."""
    s = UnitTestSettings.model_validate(
        {
            "MIN_HALF_SPREAD_BPS": 0.5,
            "MAX_HALF_SPREAD_BPS": 200.0,
            "NORMAL_MM_TOUCH_BUFFER_TICKS": 0.0,
            "NORMAL_MM_MIN_COMPETITIVE_HALF_SPREAD_BPS": 0.5,
        }
    )
    capped, diag = compute_normal_mm_market_capped_half_spread_bps(
        s,
        model_half_spread_bps=5.0,
        best_bid=99.0,
        best_ask=101.0,
        mid=100.0,
        tick=1.0,
    )
    assert capped <= 5.0
    assert diag["model_half_spread_bps"] == 5.0


def test_adjust_bid_new_order_preserves_far_quote_when_enforce_touch_false() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 5.0,
            "QUOTE_AGING_TIGHTEN_TICKS": 50.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(mid=3000.0)
    d.reservation_price = 3000.0
    now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    raw, diag = adjust_bid_target_for_aging(
        s,
        d,
        bid_px_model=2990.0,
        best_bid=3000.0,
        tick=1.0,
        working=None,
        now=now,
        position_qty=0.0,
        enforce_touch_distance_band=False,
    )
    assert raw == 2990.0
    assert "distance_ticks_new_order" not in diag.reasons


def test_hard_reprice_distance_suppressed_when_enforce_touch_false() -> None:
    from app.quote_aging import hard_reprice_reasons_buy

    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_AGING_MAX_AGE_SECONDS": 2.0,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 2.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    now = datetime(2026, 1, 1, 12, 0, 10, tzinfo=timezone.utc)
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=1,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=99.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_ack=now - timedelta(seconds=0.5),
    )
    hr = hard_reprice_reasons_buy(
        s,
        wo,
        best_bid=100.0,
        tick=0.01,
        now=now,
        enforce_touch_distance_band=False,
    )
    assert "distance_to_touch_ticks" not in hr


def test_finalize_buy_symmetric_rescue_relaxes_cap_when_two_sided() -> None:
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(mid=200.0)
    d.reservation_price = 100.0
    px, skip, tag, _ = finalize_buy_placement_touch_distance(
        s,
        d,
        90.0,
        90.0,
        best_bid=200.0,
        tick=1.0,
        position_qty=0.0,
        best_ask=201.0,
        symmetric_two_sided_rescue=True,
    )
    assert skip is None and tag == "symmetric_rescue"
    assert px == 197.0
    assert distance_buy_to_touch_ticks(px, 200.0, 1.0) == 3.0


def test_live_pattern_far_sell_rescued_near_touch_while_buy_stays_valid() -> None:
    """Regression: far SELL model vs touch; aging + symmetric finalize keep ask inside band."""
    s = UnitTestSettings.model_validate(
        {
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "QUOTE_AGING_TIGHTEN_TICKS": 10.0,
            "QUOTE_INVENTORY_PRESSURE_PCT": 0.99,
            "MAX_ABS_POSITION": 1.0,
        }
    )
    d = _decision(mid=3000.0)
    d.reservation_price = 3000.0
    best_bid = 2999.0
    best_ask = 3001.0
    bid_px, _bd = adjust_bid_target_for_aging(
        s,
        d,
        bid_px_model=2998.0,
        best_bid=best_bid,
        tick=1.0,
        working=None,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        position_qty=0.0,
    )
    _, _ad = adjust_ask_target_for_aging(
        s,
        d,
        ask_px_model=3300.0,
        best_ask=best_ask,
        tick=1.0,
        working=None,
        now=datetime(2026, 1, 1, tzinfo=timezone.utc),
        position_qty=0.0,
    )
    assert distance_buy_to_touch_ticks(bid_px, best_bid, 1.0) <= 3.0 + 1e-9
    px_f, skip_f, tag_f, _ = finalize_ask_placement_touch_distance(
        s,
        d,
        3004.0,
        3004.0,
        best_ask=best_ask,
        tick=1.0,
        position_qty=0.0,
        best_bid=best_bid,
        symmetric_two_sided_rescue=True,
    )
    assert skip_f is None and tag_f == ""
    assert distance_sell_to_touch_ticks(px_f, best_ask, 1.0) <= 3.0 + 1e-9

    d2 = _decision(mid=3000.0)
    d2.reservation_price = 3200.0
    px_far, skip_far, tag_far, _ = finalize_ask_placement_touch_distance(
        s,
        d2,
        3300.0,
        3300.0,
        best_ask=best_ask,
        tick=1.0,
        position_qty=0.0,
        best_bid=best_bid,
        symmetric_two_sided_rescue=True,
    )
    assert skip_far is None and tag_far == "symmetric_rescue"
    assert distance_sell_to_touch_ticks(px_far, best_ask, 1.0) <= 3.0 + 1e-9


def test_widen_two_sided_if_collapsed_to_one_tick_steps_outward() -> None:
    from app.quote_aging import widen_two_sided_if_collapsed_to_one_tick

    nb, na = widen_two_sided_if_collapsed_to_one_tick(
        99.0,
        100.0,
        want_bid=True,
        want_ask=True,
        tick=1.0,
        best_bid=99.0,
        best_ask=100.0,
    )
    assert nb == 98.0
    assert na == 101.0


def test_widen_two_sided_if_collapsed_skips_when_spread_wider_than_one_tick() -> None:
    from app.quote_aging import widen_two_sided_if_collapsed_to_one_tick

    nb, na = widen_two_sided_if_collapsed_to_one_tick(
        98.0,
        100.0,
        want_bid=True,
        want_ask=True,
        tick=1.0,
        best_bid=99.0,
        best_ask=100.0,
    )
    assert nb == 98.0
    assert na == 100.0
