"""Tests for v1.4.87 Phase 5A — single state-snapshot per tick.

The pre-Phase-5A ``maybe_refresh_quotes`` opened ``state._lock``
explicitly to copy out the 5 fields it needed for ``QuoteBuildContext``
(position_notional, position_qty, market, working_bid, working_ask,
working_orders_by_slot). Phase 3C shipped ``state.tick_snapshot()``
which does the same in ONE lock acquire and returns an immutable
composite. Phase 5A wires that factory into the hot path.

These tests pin the contract:

* ``state.tick_snapshot()`` acquires ``state._lock`` exactly once.
* The composite carries everything the hot path needs (position,
  market, orders, market_raw for BestBidAsk-typed consumers).
* No torn-read risk: mutations between field reads on the snapshot
  don't leak through (the snapshot is point-in-time).

Codex F9 ("7 lock acquisitions on the happy path") was the original
motivator. This is the regression test that catches future drift.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.enums import OrderStatus, Side
from app.models import BestBidAsk, PositionSnapshot, WorkingOrder
from app.state import BotState
from app.stores import TickSnapshot
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
            / f"mm_phase5a_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
    })


def _state() -> BotState:
    return BotState(_settings())


def _make_wo(side=Side.BUY, oid=1, cloid="c1", price=1.0, size=1.0) -> WorkingOrder:
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
        status=OrderStatus.ACKED,
        ts_created=now, ts_sent=now, ts_ack=now,
    )


# ---------------------------------------------------------------------------
# Lock-acquire count — the Codex F9 regression
# ---------------------------------------------------------------------------


def test_phase5a_tick_snapshot_takes_lock_exactly_once() -> None:
    """Construct a realistic state (WOs both sides, position, market),
    instrument ``state._lock`` with a counter, call
    ``state.tick_snapshot()``, and assert ONE acquire.

    Pre-Phase-5A the equivalent hot-path block took the lock once
    explicitly + the legacy ``working_bid`` / ``working_ask`` / etc.
    accessors each read state attributes (atomic via GIL but multiple
    re-reads). This test pins the new contract: ONE acquire returns
    a composite that has everything downstream needs.
    """
    state = _state()
    state.position = PositionSnapshot(
        symbol="ETH", position_qty=2.5, avg_entry_price=2.0,
        mark_price=2.01, position_notional=5.025, unrealized_pnl_usd=0.025,
    )
    state.market = BestBidAsk(
        symbol="ETH", best_bid=2.000, best_ask=2.002,
        mid_price=2.001, spread_bps=10.0, ts_local=datetime.now(timezone.utc),
    )
    state.order_store.set(Side.BUY, 0, _make_wo(side=Side.BUY, oid=10, cloid="b0"))
    state.order_store.set(Side.SELL, 0, _make_wo(side=Side.SELL, oid=20, cloid="s0"))
    state.order_store.set(Side.SELL, 1, _make_wo(side=Side.SELL, oid=21, cloid="s1"))

    count = [0]
    real = state._lock

    class CountingLock:
        def __enter__(self):
            count[0] += 1
            return real.__enter__()
        def __exit__(self, *args):
            return real.__exit__(*args)

    state._lock = CountingLock()  # type: ignore[assignment]
    try:
        snap = state.tick_snapshot()
        assert count[0] == 1, (
            f"state.tick_snapshot() should acquire state._lock exactly once "
            f"per the Phase 5A / Codex F9 contract; got {count[0]} acquires. "
            f"If this fails, somebody added a state attribute read inside "
            f"build_tick_snapshot that re-enters the lock — investigate "
            f"app/stores/tick_snapshot.py."
        )
    finally:
        state._lock = real

    # Sanity: snapshot is populated correctly.
    assert isinstance(snap, TickSnapshot)
    assert snap.position.qty == 2.5
    assert snap.market.best_bid == 2.000
    assert snap.orders.count_live_per_side(Side.BUY) == 1
    assert snap.orders.count_live_per_side(Side.SELL) == 2


# ---------------------------------------------------------------------------
# market_raw — BestBidAsk reference for legacy consumers
# ---------------------------------------------------------------------------


def test_phase5a_snapshot_carries_market_raw_reference() -> None:
    """``QuoteBuildContext.market`` takes a ``BestBidAsk`` object, not
    primitives. The snapshot must carry the raw reference so the hot
    path doesn't have to re-read ``state.market`` outside the lock
    (which would be a torn-read risk vs WS handlers)."""
    state = _state()
    bba = BestBidAsk(
        symbol="ETH", best_bid=2.000, best_ask=2.002,
        mid_price=2.001, spread_bps=10.0, ts_local=datetime.now(timezone.utc),
    )
    state.market = bba
    snap = state.tick_snapshot()
    assert snap.market_raw is bba, (
        "snapshot must reference the SAME BestBidAsk object captured "
        "under the lock, not a copy or reconstruction"
    )


def test_phase5a_snapshot_market_raw_is_none_when_market_absent() -> None:
    state = _state()
    state.market = None
    snap = state.tick_snapshot()
    assert snap.market_raw is None
    assert snap.market.is_present() is False


# ---------------------------------------------------------------------------
# Point-in-time invariant (re-pinned)
# ---------------------------------------------------------------------------


def test_phase5a_snapshot_decouples_from_subsequent_mutations() -> None:
    """After ``tick_snapshot()`` returns, downstream consumers must see
    a frozen view. WS-handler mutations to state DO NOT affect the
    snapshot's contents. This is the key safety property of Phase 5A —
    it eliminates the race window where the hot path's reads could
    interleave with WS state updates.
    """
    state = _state()
    state.position = PositionSnapshot(
        symbol="ETH", position_qty=1.0, avg_entry_price=None,
        mark_price=None, position_notional=2.0, unrealized_pnl_usd=0.0,
    )
    state.order_store.set(Side.BUY, 0, _make_wo(oid=100, cloid="t1"))
    snap = state.tick_snapshot()
    # Simulate a WS-handler mutation AFTER the snapshot was taken.
    state.position = PositionSnapshot(
        symbol="ETH", position_qty=99.0, avg_entry_price=None,
        mark_price=None, position_notional=198.0, unrealized_pnl_usd=0.0,
    )
    state.order_store.set(Side.BUY, 0, None)
    # Snapshot still reflects the pre-mutation state.
    assert snap.position.qty == 1.0
    assert snap.position.notional_abs_usd == 2.0
    assert snap.orders.count_live_per_side(Side.BUY) == 1
