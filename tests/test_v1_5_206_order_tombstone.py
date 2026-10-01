"""v1.5.206 — Order-ID tombstone cache tests.

Covers the OrderStore-side tombstone bookkeeping. The execution.py
amend-response swallow path is exercised separately in the integration
tests; this file proves the foundation works in isolation.

Race scenario the cache closes: v1.5.204-260528-072247 session was
killed by `amend_response_unconfirmed (missing_row_for_amend)`. The
order was placed at T0, cancel was issued at T1, amend response
arrived at T1+δ with no matching row in the venue's batch reply. With
the tombstone cache, the lookup hits the recently-removed entry and
the resilience handler treats the unconfirmed amend as benign instead
of escalating to kill.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.state import BotState
from app.stores.order_store import (
    _TOMBSTONE_MAX,
    _TOMBSTONE_MAX_AGE_SECONDS,
    _TombstoneEntry,
)
from tests.settings_helpers import UnitTestSettings


def _make_state(tmp_path: Path) -> BotState:
    settings = UnitTestSettings.model_validate(
        {
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'mm.db').as_posix()}",
        }
    )
    return BotState(settings)


def _make_wo(*, oid: int, cloid: str, status: OrderStatus = OrderStatus.ACKED) -> WorkingOrder:
    return WorkingOrder(
        order_id_local=cloid,
        order_id_exchange=str(oid),
        client_order_id=cloid,
        symbol="TON-USDT-SWAP",
        side=Side.SELL,
        price=1.0,
        size=1.0,
        status=status,
        post_only=True,
    )


# --------------------- tombstone push on terminal transition --------------- #


def test_transition_to_terminal_pushes_tombstone(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    store = state.order_store
    wo = _make_wo(oid=12345, cloid="abc123")
    # Put the WO in a slot so it's indexed.
    store.set(Side.SELL, 0, wo)
    assert store.find_by_oid(12345) is wo
    # Transition to CANCELED — terminal.
    store.transition(wo, OrderStatus.CANCELED, "test")
    # Index removed.
    assert store.find_by_oid(12345) is None
    # Tombstone recorded with the right metadata.
    tomb = store.was_recently_removed(oid=12345)
    assert tomb is not None
    assert tomb.order_id_exchange == 12345
    assert tomb.client_order_id == "abc123"
    assert "CANCELED" in tomb.removed_reason


def test_transition_to_filled_pushes_tombstone(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    store = state.order_store
    wo = _make_wo(oid=999, cloid="fillme")
    store.set(Side.SELL, 0, wo)
    store.transition(wo, OrderStatus.FILLED, "test")
    tomb = store.was_recently_removed(oid=999)
    assert tomb is not None
    assert "FILLED" in tomb.removed_reason


def test_slot_replacement_tombstones_existing(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    store = state.order_store
    wo_old = _make_wo(oid=111, cloid="old")
    wo_new = _make_wo(oid=222, cloid="new")
    store.set(Side.SELL, 0, wo_old)
    # Replacing the slot with a different WO should tombstone the old one.
    store.set(Side.SELL, 0, wo_new)
    tomb_old = store.was_recently_removed(oid=111)
    assert tomb_old is not None
    # New WO is still in the index.
    assert store.find_by_oid(222) is wo_new
    # New WO is NOT tombstoned (it's live).
    assert store.was_recently_removed(oid=222) is None


def test_slot_clear_tombstones_existing(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    store = state.order_store
    wo = _make_wo(oid=333, cloid="clear")
    store.set(Side.SELL, 0, wo)
    store.set(Side.SELL, 0, None)
    tomb = store.was_recently_removed(oid=333)
    assert tomb is not None


# --------------------- lookup semantics ----------------------------------- #


def test_was_recently_removed_returns_none_when_unknown(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    store = state.order_store
    assert store.was_recently_removed(oid=999999) is None
    assert store.was_recently_removed(cloid="never_seen") is None


def test_was_recently_removed_by_cloid(tmp_path: Path) -> None:
    state = _make_state(tmp_path)
    store = state.order_store
    wo = _make_wo(oid=555, cloid="byname")
    store.set(Side.SELL, 0, wo)
    store.transition(wo, OrderStatus.CANCELED, "test")
    # OID match also works:
    assert store.was_recently_removed(oid=555) is not None
    # Cloid match works on a separate query:
    state2 = _make_state(tmp_path / "alt")
    wo2 = _make_wo(oid=777, cloid="byname2")
    state2.order_store.set(Side.SELL, 0, wo2)
    state2.order_store.transition(wo2, OrderStatus.CANCELED, "test")
    found = state2.order_store.was_recently_removed(cloid="byname2")
    assert found is not None
    assert found.client_order_id == "byname2"


def test_tombstone_age_window_enforced(tmp_path: Path) -> None:
    """Entries older than _TOMBSTONE_MAX_AGE_SECONDS are treated as
    not-found. The bounded deque can hold them but the lookup ignores
    them — the bot's "fresh tombstone" semantics are about RACE
    WINDOWS, not historical archaeology."""
    state = _make_state(tmp_path)
    store = state.order_store
    wo = _make_wo(oid=42, cloid="aged")
    store.set(Side.SELL, 0, wo)
    store.transition(wo, OrderStatus.CANCELED, "test")
    # Confirm it's present at now.
    assert store.was_recently_removed(oid=42) is not None
    # Look up with a future ``now_mono`` to simulate elapsed time.
    future = time.monotonic() + _TOMBSTONE_MAX_AGE_SECONDS + 1.0
    assert store.was_recently_removed(oid=42, now_mono=future) is None


def test_tombstone_deque_bounded(tmp_path: Path) -> None:
    """Pushing more than _TOMBSTONE_MAX entries evicts the oldest.
    Deque maxlen guarantees memory boundedness."""
    state = _make_state(tmp_path)
    store = state.order_store
    # Push 2× the max number of entries via repeated set/clear cycles.
    n = _TOMBSTONE_MAX * 2
    for i in range(n):
        wo = _make_wo(oid=10000 + i, cloid=f"c{i}")
        store.set(Side.SELL, 0, wo)
        store.transition(wo, OrderStatus.CANCELED, "loop")
    # Deque size stays at maxlen.
    assert len(store._recently_removed) == _TOMBSTONE_MAX
    # First entry was evicted; last entry is still there.
    assert store.was_recently_removed(oid=10000) is None
    assert store.was_recently_removed(oid=10000 + n - 1) is not None


def test_idempotent_no_tombstone_on_terminal_redundant_transition(
    tmp_path: Path,
) -> None:
    """If a WO is already terminal and we call transition again,
    don't push duplicate tombstones (would still be harmless but
    wastes deque slots)."""
    state = _make_state(tmp_path)
    store = state.order_store
    wo = _make_wo(oid=12, cloid="once")
    store.set(Side.SELL, 0, wo)
    store.transition(wo, OrderStatus.CANCELED, "first")
    n_after_first = len(store._recently_removed)
    # Second terminal transition — should NOT push another tombstone.
    store.transition(wo, OrderStatus.CANCELED, "redundant")
    assert len(store._recently_removed) == n_after_first


# --------------------- newest-first matching ------------------------------ #


def test_was_recently_removed_picks_newest_on_duplicate(tmp_path: Path) -> None:
    """If the same OID has been tombstoned twice (rare but possible
    via different code paths), the newest entry wins. Otherwise stale
    metadata could mask a more-recent removal."""
    state = _make_state(tmp_path)
    store = state.order_store
    # First removal:
    wo1 = _make_wo(oid=88, cloid="first")
    store.set(Side.SELL, 0, wo1)
    store.transition(wo1, OrderStatus.CANCELED, "first")
    # Second removal of the same OID via a different code path
    # (re-insert + clear). This is artificial but exercises the
    # deque-walk newest-first behaviour.
    wo2 = _make_wo(oid=88, cloid="second")
    store.set(Side.SELL, 0, wo2)
    store.transition(wo2, OrderStatus.FILLED, "second")
    found = store.was_recently_removed(oid=88)
    assert found is not None
    # The newest tombstone wins.
    assert found.client_order_id == "second"
    assert "FILLED" in found.removed_reason
