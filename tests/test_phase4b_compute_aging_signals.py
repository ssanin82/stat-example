"""Tests for v1.4.84 Phase 4B — ``compute_aging_signals`` module-level
pure function extracted from ``QuoteEngine.build_quotes``.

The function takes:
  * ``working_orders_by_slot`` — multi-rung snapshot (Phase 1C path)
  * ``resting_bid`` / ``resting_ask`` — backward-compat fallback
  * ``best_bid`` / ``best_ask`` / ``tick`` / ``now`` — venue state
  * ``settings`` — knob source (BEHIND_TOUCH_*, AT_TOUCH_*, etc.)

And returns ``dict[(Side, int), tuple[reason, ...]]`` — the per-slot
hard-cancel signals fed into the dispatcher.

Tests pin:

* Multi-rung path evaluates EVERY slot, not just inside-rung (Phase 1C
  invariant — Codex F6/B3).
* Backward-compat fallback when ``working_orders_by_slot`` is empty.
* No-op when no caps are configured.
* Returns reason tuples sourced from ``hard_reprice_reasons_{buy,sell}``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.quote_engine import compute_aging_signals
from tests.settings_helpers import UnitTestSettings


def _bid_wo(
    *,
    price: float = 2.000,
    age_seconds: float = 0.0,
    status: OrderStatus = OrderStatus.ACKED,
    level_idx: int = 0,
) -> WorkingOrder:
    t0 = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)
    ts = t0 - timedelta(seconds=age_seconds)
    return WorkingOrder(
        order_id_local=f"l-bid-{level_idx}",
        order_id_exchange=10 + level_idx,
        client_order_id=f"c-bid-{level_idx}",
        symbol="ETH",
        side=Side.BUY,
        price=price,
        size=1.0,
        post_only=True,
        status=status,
        ts_created=ts,
        ts_sent=ts,
        ts_ack=ts,
        level_idx=level_idx,
    )


def _ask_wo(
    *,
    price: float = 2.002,
    age_seconds: float = 0.0,
    status: OrderStatus = OrderStatus.ACKED,
    level_idx: int = 0,
) -> WorkingOrder:
    t0 = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)
    ts = t0 - timedelta(seconds=age_seconds)
    return WorkingOrder(
        order_id_local=f"l-ask-{level_idx}",
        order_id_exchange=20 + level_idx,
        client_order_id=f"c-ask-{level_idx}",
        symbol="ETH",
        side=Side.SELL,
        price=price,
        size=1.0,
        post_only=True,
        status=status,
        ts_created=ts,
        ts_sent=ts,
        ts_ack=ts,
        level_idx=level_idx,
    )


def _now() -> datetime:
    return datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


def _settings_caps_active(**overrides):
    """Settings with BEHIND_TOUCH and AT_TOUCH caps both active."""
    base = {
        "TRADING_ENABLED": False,
        "QUOTE_AGING_ENABLED": True,
        "QUOTE_AGING_MAX_AGE_SECONDS": 6.0,
        "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 1000.0,  # huge → distance check disabled
        "BEHIND_TOUCH_MAX_AGE_SECONDS": 1.0,
        "AT_TOUCH_MAX_AGE_SECONDS": 2.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


# ---------------------------------------------------------------------------
# No-cap / no-op cases
# ---------------------------------------------------------------------------


def test_phase4b_returns_empty_when_no_caps_active() -> None:
    s = UnitTestSettings.model_validate({"QUOTE_AGING_ENABLED": False})
    out = compute_aging_signals(
        s,
        {},
        resting_bid=_bid_wo(age_seconds=60.0),
        resting_ask=_ask_wo(age_seconds=60.0),
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    assert out == {}


def test_phase4b_returns_empty_when_no_orders() -> None:
    s = _settings_caps_active()
    out = compute_aging_signals(
        s,
        {},
        resting_bid=None,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Backward-compat fallback (empty slot map → resting_bid/ask path)
# ---------------------------------------------------------------------------


def test_phase4b_legacy_path_fires_behind_touch_cap_on_inside_bid() -> None:
    s = _settings_caps_active()
    # Behind-touch: bid 1.999, best_bid 2.000 → 1 tick behind. Age 1.5s
    # > BEHIND_TOUCH_MAX_AGE_SECONDS=1.0 → fires.
    wo = _bid_wo(price=1.999, age_seconds=1.5)
    out = compute_aging_signals(
        s,
        {},
        resting_bid=wo,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    assert (Side.BUY, 0) in out
    assert "behind_touch_order_age_seconds" in out[(Side.BUY, 0)]


def test_phase4b_legacy_path_fires_at_touch_cap_on_inside_ask() -> None:
    s = _settings_caps_active()
    # At-touch: ask 2.002 == best_ask 2.002. Age 2.5s > AT_TOUCH_MAX_AGE_SECONDS=2.0 → fires.
    wo = _ask_wo(price=2.002, age_seconds=2.5)
    out = compute_aging_signals(
        s,
        {},
        resting_bid=None,
        resting_ask=wo,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    assert (Side.SELL, 0) in out
    assert "at_touch_order_age_seconds" in out[(Side.SELL, 0)]


def test_phase4b_legacy_path_does_not_fire_when_under_threshold() -> None:
    s = _settings_caps_active()
    # Behind-touch but only 0.5s old (< 1.0s threshold).
    wo = _bid_wo(price=1.999, age_seconds=0.5)
    out = compute_aging_signals(
        s,
        {},
        resting_bid=wo,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Multi-rung path (Phase 1C invariant — every aged slot fires)
# ---------------------------------------------------------------------------


def test_phase4b_multi_rung_fires_outer_bid_rung() -> None:
    """Phase 1C invariant: an outer-rung order CAN trigger the cap.

    Pre-Phase-1C the inline block only checked inside rung → outer
    rungs aged forever (Codex F6/B3). After Phase 1C, every slot in
    the working_orders_by_slot snapshot is evaluated.
    """
    s = _settings_caps_active()
    inside_bid = _bid_wo(price=2.000, age_seconds=0.5, level_idx=0)  # fresh
    outer_bid = _bid_wo(price=1.998, age_seconds=3.0, level_idx=1)  # aged outer
    slot_map = {
        (Side.BUY, 0): inside_bid,
        (Side.BUY, 1): outer_bid,
    }
    out = compute_aging_signals(
        s,
        slot_map,
        resting_bid=inside_bid,  # unused once slot_map is non-empty
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    # Inside rung is fresh → no entry.
    assert (Side.BUY, 0) not in out
    # Outer rung is aged behind-touch → fires.
    assert (Side.BUY, 1) in out
    assert "behind_touch_order_age_seconds" in out[(Side.BUY, 1)]


def test_phase4b_multi_rung_both_sides() -> None:
    s = _settings_caps_active()
    slot_map = {
        (Side.BUY, 0): _bid_wo(price=1.999, age_seconds=1.5, level_idx=0),
        (Side.SELL, 0): _ask_wo(price=2.002, age_seconds=2.5, level_idx=0),
    }
    out = compute_aging_signals(
        s,
        slot_map,
        resting_bid=None,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    assert (Side.BUY, 0) in out
    assert (Side.SELL, 0) in out


def test_phase4b_multi_rung_path_ignores_resting_args_when_map_non_empty() -> None:
    """When ``working_orders_by_slot`` is provided, the fallback args
    ``resting_bid`` / ``resting_ask`` are intentionally IGNORED — the
    map is the canonical source. This pins the v1.4.70 Phase 1C invariant.
    """
    s = _settings_caps_active()
    aged_outer = _bid_wo(price=1.998, age_seconds=3.0, level_idx=1)
    slot_map = {(Side.BUY, 1): aged_outer}
    # Pass a separate, aged inside-rung as resting_bid — should be ignored.
    discarded = _bid_wo(price=1.999, age_seconds=10.0, level_idx=0)
    out = compute_aging_signals(
        s,
        slot_map,
        resting_bid=discarded,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    # Only the outer rung fires; the discarded inside rung is not consulted.
    assert (Side.BUY, 1) in out
    assert (Side.BUY, 0) not in out


def test_phase4b_multi_rung_skips_none_orders() -> None:
    """Slot map can contain stale ``None`` entries; the function skips them
    rather than raising."""
    s = _settings_caps_active()
    slot_map: dict = {
        (Side.BUY, 0): None,
        (Side.SELL, 0): _ask_wo(price=2.002, age_seconds=2.5),
    }
    out = compute_aging_signals(
        s,
        slot_map,
        resting_bid=None,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    assert (Side.BUY, 0) not in out
    assert (Side.SELL, 0) in out


# ---------------------------------------------------------------------------
# Reason content + tuple shape
# ---------------------------------------------------------------------------


def test_phase4b_returns_tuple_of_strings() -> None:
    s = _settings_caps_active()
    wo = _bid_wo(price=1.999, age_seconds=1.5)
    out = compute_aging_signals(
        s,
        {(Side.BUY, 0): wo},
        resting_bid=None,
        resting_ask=None,
        best_bid=2.000,
        best_ask=2.002,
        tick=0.001,
        now=_now(),
    )
    reasons = out[(Side.BUY, 0)]
    assert isinstance(reasons, tuple)
    assert all(isinstance(r, str) for r in reasons)
    assert len(reasons) >= 1
