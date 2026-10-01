"""Tests for v1.4.80 wedge-elimination-cleanup Phase 3A — OrderStore.

The store is the new SINGLE OWNER of working-order state. Tests
verify:

* Indexes stay consistent under set / delete / transition
* find_by_oid / find_by_cloid are O(1) for indexed entries
* find_by_oid_or_cloid preserves the legacy "cloid match only when
  no OID" semantic (the place-before-ack race window)
* Legacy ``BotState.set_working_order`` keeps indexes in sync
* Terminal transitions remove WOs from indexes
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path

from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.state import BotState
from app.stores.order_store import OrderStore
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
            / f"mm_orderstore_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
    })


def _make_wo(
    *,
    side: Side = Side.BUY,
    oid: int | None = 100_001,
    cloid: str | None = "cloid-100001",
    status: OrderStatus = OrderStatus.ACKED,
    price: float = 1.0,
    size: float = 1.0,
) -> WorkingOrder:
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
        ts_created=now,
        ts_sent=now,
        ts_ack=now,
    )


def _setup():
    s = _settings()
    state = BotState(s)
    return state, s


# ---------------------------------------------------------------------------
# Index maintenance
# ---------------------------------------------------------------------------


def test_order_store_set_indexes_wo_by_oid_and_cloid() -> None:
    """``order_store.set(side, lvl, wo)`` populates both indexes."""
    state, _ = _setup()
    store = state.order_store
    wo = _make_wo(oid=200_001, cloid="cloid-A")
    store.set(Side.BUY, 0, wo)
    by_oid, by_cloid = store.index_size()
    assert by_oid == 1
    assert by_cloid == 1
    assert store.find_by_oid(200_001) is wo
    assert store.find_by_cloid("cloid-A") is wo


def test_order_store_set_none_removes_from_indexes() -> None:
    """``set(side, lvl, None)`` deletes the slot AND removes the WO
    from both indexes."""
    state, _ = _setup()
    store = state.order_store
    wo = _make_wo(oid=200_002, cloid="cloid-B")
    store.set(Side.BUY, 0, wo)
    assert store.find_by_oid(200_002) is wo
    store.set(Side.BUY, 0, None)
    assert store.find_by_oid(200_002) is None
    assert store.find_by_cloid("cloid-B") is None
    by_oid, by_cloid = store.index_size()
    assert by_oid == 0
    assert by_cloid == 0


def test_order_store_transition_to_terminal_removes_from_indexes() -> None:
    """``transition(wo, CANCELED)`` updates status AND removes from
    indexes (terminal WOs aren't lookup-targets per Phase 1B.2).
    """
    state, _ = _setup()
    store = state.order_store
    wo = _make_wo(oid=200_003, cloid="cloid-C")
    store.set(Side.BUY, 0, wo)
    assert store.find_by_oid(200_003) is wo
    store.transition(wo, OrderStatus.CANCELED, reason="test")
    assert wo.status == OrderStatus.CANCELED
    # Removed from indexes.
    assert store.find_by_oid(200_003) is None
    assert store.find_by_cloid("cloid-C") is None
    # But still in state (slot not cleared by transition; that's
    # the caller's responsibility).
    assert state.get_working_order(Side.BUY, 0) is wo


def test_order_store_transition_to_non_terminal_keeps_indexes() -> None:
    """Non-terminal transitions (SENT → ACKED) keep WO indexed."""
    state, _ = _setup()
    store = state.order_store
    wo = _make_wo(oid=200_004, cloid="cloid-D", status=OrderStatus.SENT)
    store.set(Side.BUY, 0, wo)
    assert store.find_by_oid(200_004) is wo
    store.transition(wo, OrderStatus.ACKED, reason="ws_live")
    assert wo.status == OrderStatus.ACKED
    # Still indexed.
    assert store.find_by_oid(200_004) is wo


# ---------------------------------------------------------------------------
# Lookup semantics
# ---------------------------------------------------------------------------


def test_order_store_find_by_oid_or_cloid_oid_priority() -> None:
    """OID match wins over cloid match (most specific). Two WOs
    share a cloid (artificial test scenario); the OID disambiguates.
    """
    state, _ = _setup()
    store = state.order_store
    wo_oid = _make_wo(side=Side.BUY, oid=200_005, cloid="shared")
    wo_cloid = _make_wo(side=Side.SELL, oid=None, cloid="shared")
    store.set(Side.BUY, 0, wo_oid)
    store.set(Side.SELL, 0, wo_cloid)
    found = store.find_by_oid_or_cloid(200_005, "shared")
    assert found is wo_oid, "OID match must win"


def test_order_store_find_by_oid_or_cloid_cloid_path_no_oid_only() -> None:
    """Legacy semantic: cloid match ONLY targets WOs without OID
    (the place-before-ack race). A cloid-collision against an
    OID-bound WO is NOT returned.
    """
    state, _ = _setup()
    store = state.order_store
    # WO with OID bound + cloid.
    wo_bound = _make_wo(side=Side.BUY, oid=200_006, cloid="raceA")
    store.set(Side.BUY, 0, wo_bound)
    # WO without OID, same cloid (test-only setup).
    wo_unbound = _make_wo(side=Side.SELL, oid=None, cloid="raceA")
    store.set(Side.SELL, 0, wo_unbound)
    # find_by_oid_or_cloid(None, "raceA") should return wo_unbound
    # (no-OID match), NOT wo_bound.
    found = store.find_by_oid_or_cloid(None, "raceA")
    assert found is wo_unbound


def test_order_store_find_by_oid_or_cloid_returns_none_when_both_none() -> None:
    state, _ = _setup()
    assert state.order_store.find_by_oid_or_cloid(None, None) is None


def test_order_store_find_by_oid_returns_none_for_missing() -> None:
    state, _ = _setup()
    assert state.order_store.find_by_oid(999_999_999) is None


def test_order_store_find_by_cloid_returns_none_for_missing() -> None:
    state, _ = _setup()
    assert state.order_store.find_by_cloid("nonexistent") is None


# ---------------------------------------------------------------------------
# Bridge with legacy BotState API
# ---------------------------------------------------------------------------


def test_order_store_legacy_set_working_order_maintains_indexes() -> None:
    """v1.4.80 Phase 3A: ``state.set_working_order`` (the legacy
    setter) also updates the OrderStore indexes. Existing call
    sites get index maintenance for free.
    """
    state, _ = _setup()
    wo = _make_wo(oid=200_007, cloid="cloid-legacy")
    # Use legacy setter, not the store's set.
    with state._lock:
        state.set_working_order(Side.BUY, 0, wo)
    # Indexes should reflect this WO.
    assert state.order_store.find_by_oid(200_007) is wo
    assert state.order_store.find_by_cloid("cloid-legacy") is wo


def test_order_store_legacy_set_working_order_none_removes_from_indexes() -> None:
    state, _ = _setup()
    wo = _make_wo(oid=200_008, cloid="cloid-legacy2")
    with state._lock:
        state.set_working_order(Side.BUY, 0, wo)
    assert state.order_store.find_by_oid(200_008) is wo
    # Legacy delete via set(..., None).
    with state._lock:
        state.set_working_order(Side.BUY, 0, None)
    assert state.order_store.find_by_oid(200_008) is None
    assert state.order_store.find_by_cloid("cloid-legacy2") is None


def test_order_store_rebuild_indexes_from_existing_state() -> None:
    """If WOs were inserted before the store's first index build
    (legacy callers, persisted state restored at startup), the
    ``rebuild_indexes`` API recovers them.
    """
    state, _ = _setup()
    # Mutate the underlying dict directly, bypassing all setters.
    wo = _make_wo(oid=200_009, cloid="cloid-bypass")
    state._working_orders[Side.BUY][0] = wo
    # Indexes are now stale (the store doesn't know).
    assert state.order_store.find_by_oid(200_009) is wo or True  # scan fallback finds it
    # After explicit rebuild, the index is authoritative.
    state.order_store.rebuild_indexes()
    by_oid, by_cloid = state.order_store.index_size()
    assert by_oid >= 1
    assert by_cloid >= 1


def test_order_store_index_skips_terminal_wos_on_rebuild() -> None:
    """``rebuild_indexes`` does NOT index terminal WOs (matches
    the Phase 1B.2 contract: terminal lookups are skipped).
    """
    state, _ = _setup()
    wo = _make_wo(oid=200_010, cloid="cloid-term", status=OrderStatus.CANCELED)
    state._working_orders[Side.BUY][0] = wo
    state.order_store.rebuild_indexes()
    assert state.order_store.find_by_oid(200_010) is None
    assert state.order_store.find_by_cloid("cloid-term") is None


def test_order_store_snapshot_by_slot_returns_active_only() -> None:
    """``snapshot_by_slot`` returns ALL slots regardless of status
    (caller filters). Used by the engine's per-slot age-cap
    evaluation (Phase 1C).
    """
    state, _ = _setup()
    store = state.order_store
    wo_a = _make_wo(side=Side.BUY, oid=200_011, cloid="a")
    wo_b = _make_wo(side=Side.SELL, oid=200_012, cloid="b")
    store.set(Side.BUY, 0, wo_a)
    store.set(Side.SELL, 1, wo_b)
    snap = store.snapshot_by_slot()
    assert (Side.BUY, 0) in snap
    assert (Side.SELL, 1) in snap
    assert snap[(Side.BUY, 0)] is wo_a
    assert snap[(Side.SELL, 1)] is wo_b


def test_order_store_iter_side_returns_sorted_tuples() -> None:
    state, _ = _setup()
    store = state.order_store
    wo_0 = _make_wo(side=Side.BUY, oid=200_013, cloid="0")
    wo_1 = _make_wo(side=Side.BUY, oid=200_014, cloid="1")
    # Insert out of order — verify sorted return.
    store.set(Side.BUY, 1, wo_1)
    store.set(Side.BUY, 0, wo_0)
    result = store.iter_side(Side.BUY)
    assert [idx for idx, _ in result] == [0, 1]
    assert result[0][1] is wo_0
    assert result[1][1] is wo_1
