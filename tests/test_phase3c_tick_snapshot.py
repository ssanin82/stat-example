"""Tests for v1.4.82 Phase 3C — TickSnapshot.

The snapshot is the per-tick immutable view that Phase 5A wires
into the hot path. Tests verify:

* Construction from a populated BotState
* Immutability (frozen dataclasses)
* Field math (mid, spread, headroom, reducing-side)
* Single-lock-acquire (point-in-time consistency)
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.enums import OrderStatus, Side
from app.models import BestBidAsk, PositionSnapshot, WorkingOrder
from app.state import BotState
from app.stores import (
    ImmutableMarketView,
    ImmutableOrderView,
    ImmutablePositionView,
    TickSnapshot,
    build_tick_snapshot,
)
from app.utils.time import utc_now
from tests.settings_helpers import UnitTestSettings


def _settings():
    return UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_phase3c_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
    })


def _state() -> BotState:
    return BotState(_settings())


def _make_wo(side=Side.BUY, oid=1, cloid="c1", price=1.0, size=1.0,
             status=OrderStatus.ACKED) -> WorkingOrder:
    now = utc_now()
    return WorkingOrder(
        order_id_local=f"l-{uuid.uuid4().hex[:8]}",
        order_id_exchange=oid,
        client_order_id=cloid,
        symbol="ETH",
        side=side,
        price=price,
        size=size,
        post_only=True,
        status=status,
        ts_created=now, ts_sent=now, ts_ack=now,
    )


# ---------------------------------------------------------------------------
# Construction + composition
# ---------------------------------------------------------------------------


def test_phase3c_snapshot_initial_empty() -> None:
    state = _state()
    snap = state.tick_snapshot()
    assert isinstance(snap, TickSnapshot)
    assert isinstance(snap.orders, ImmutableOrderView)
    assert isinstance(snap.position, ImmutablePositionView)
    assert isinstance(snap.market, ImmutableMarketView)
    assert snap.orders.is_empty()
    assert snap.position.is_flat()
    assert snap.market.is_present() is False


def test_phase3c_snapshot_captures_working_orders() -> None:
    state = _state()
    wo_buy = _make_wo(side=Side.BUY, oid=10, cloid="buyA")
    wo_sell_inside = _make_wo(side=Side.SELL, oid=20, cloid="sellA")
    wo_sell_outer = _make_wo(side=Side.SELL, oid=21, cloid="sellB")
    state.order_store.set(Side.BUY, 0, wo_buy)
    state.order_store.set(Side.SELL, 0, wo_sell_inside)
    state.order_store.set(Side.SELL, 1, wo_sell_outer)

    snap = state.tick_snapshot()
    assert not snap.orders.is_empty()
    assert snap.orders.count_live_per_side(Side.BUY) == 1
    assert snap.orders.count_live_per_side(Side.SELL) == 2
    # Slot lookup.
    assert snap.orders.slot(Side.BUY, 0) is wo_buy
    assert snap.orders.slot(Side.SELL, 1) is wo_sell_outer
    assert snap.orders.slot(Side.BUY, 99) is None
    # Per-side iter.
    sell_side = snap.orders.side(Side.SELL)
    assert [lvl for lvl, _ in sell_side] == [0, 1]


def test_phase3c_snapshot_captures_position() -> None:
    state = _state()
    state.position = PositionSnapshot(
        symbol="ETH",
        position_qty=3.5,
        avg_entry_price=2.0,
        mark_price=2.01,
        position_notional=7.035,
        unrealized_pnl_usd=0.035,
    )
    snap = state.tick_snapshot()
    assert snap.position.qty == 3.5
    assert snap.position.notional_abs_usd == 7.035
    assert snap.position.unrealized_pnl_usd == 0.035
    assert snap.position.avg_entry_price == 2.0
    assert snap.position.mark_price == 2.01
    assert snap.position.is_flat() is False
    assert snap.position.reducing_side() == Side.SELL


def test_phase3c_snapshot_captures_market() -> None:
    state = _state()
    state.market = BestBidAsk(
        symbol="ETH",
        best_bid=2.000,
        best_ask=2.002,
        mid_price=2.001,
        spread_bps=10.0,
        ts_local=datetime.now(timezone.utc),
    )
    snap = state.tick_snapshot()
    assert snap.market.best_bid == 2.000
    assert snap.market.best_ask == 2.002
    assert snap.market.mid_price == 2.001
    assert snap.market.spread_bps == 10.0
    assert snap.market.is_present() is True


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_phase3c_snapshot_is_frozen() -> None:
    state = _state()
    snap = state.tick_snapshot()
    # All four sub-views are frozen dataclasses.
    with pytest.raises((AttributeError, Exception)):
        snap.orders.by_slot = ()  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        snap.position.qty = 99.0  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        snap.market.best_bid = 99.0  # type: ignore[misc]
    with pytest.raises((AttributeError, Exception)):
        snap.captured_at = datetime.now(timezone.utc)  # type: ignore[misc]


def test_phase3c_snapshot_is_point_in_time() -> None:
    """A snapshot captures state at construction; mutations after
    the snapshot do NOT affect the snapshot's contents.
    """
    state = _state()
    wo1 = _make_wo(oid=100, cloid="t1")
    state.order_store.set(Side.BUY, 0, wo1)
    snap = state.tick_snapshot()
    # Mutate state AFTER taking the snapshot.
    state.order_store.set(Side.BUY, 0, None)
    state.order_store.set(Side.SELL, 0, _make_wo(side=Side.SELL, oid=200, cloid="t2"))
    # Snapshot still shows the original state.
    assert snap.orders.count_live_per_side(Side.BUY) == 1
    assert snap.orders.count_live_per_side(Side.SELL) == 0
    assert snap.orders.slot(Side.BUY, 0) is wo1


# ---------------------------------------------------------------------------
# Reducer side
# ---------------------------------------------------------------------------


def test_phase3c_immutable_position_reducer_buy_when_short() -> None:
    pv = ImmutablePositionView(
        qty=-2.5, notional_abs_usd=5.0, unrealized_pnl_usd=0.0,
        avg_entry_price=None, mark_price=None,
    )
    assert pv.reducing_side() == Side.BUY
    assert pv.is_flat() is False


def test_phase3c_immutable_position_reducer_none_when_flat() -> None:
    pv = ImmutablePositionView(
        qty=0.0, notional_abs_usd=0.0, unrealized_pnl_usd=0.0,
        avg_entry_price=None, mark_price=None,
    )
    assert pv.reducing_side() is None
    assert pv.is_flat() is True


# ---------------------------------------------------------------------------
# Lock policy — single acquire
# ---------------------------------------------------------------------------


def test_phase3c_build_tick_snapshot_takes_lock_once() -> None:
    """``build_tick_snapshot`` acquires state._lock exactly once.

    Instrument the lock by wrapping it and counting acquires.
    """
    state = _state()
    state.position = PositionSnapshot(
        symbol="ETH", position_qty=1.0, avg_entry_price=None,
        mark_price=None, position_notional=1.0, unrealized_pnl_usd=0.0,
    )
    state.order_store.set(Side.BUY, 0, _make_wo(oid=999, cloid="lock-test"))

    acquire_count = [0]
    real_lock = state._lock

    class CountingLock:
        def __enter__(self):
            acquire_count[0] += 1
            return real_lock.__enter__()
        def __exit__(self, *args):
            return real_lock.__exit__(*args)

    state._lock = CountingLock()  # type: ignore[assignment]
    try:
        _ = build_tick_snapshot(state)
        assert acquire_count[0] == 1, (
            f"build_tick_snapshot should acquire state._lock exactly once; "
            f"got {acquire_count[0]}"
        )
    finally:
        state._lock = real_lock
