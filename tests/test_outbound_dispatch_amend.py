"""amend-prio Phase 2 (v1.4.16) — dispatcher amend lane.

The place lane carries both ``kind="place"`` and ``kind="amend"``
intents. The dispatcher partitions them by kind and routes to
separate executor callbacks (``execute_place_batch`` /
``execute_amend_batch``, or per-intent fallbacks). Coalesce key is
extended to ``(side, level_idx, kind)`` so place and amend on the
same slot don't collapse.

Cross-lane defer matrix (vs the pre-Phase-2 2x2):

  | Submitting | Cancel inflight | Place inflight | Amend inflight |
  |------------|-----------------|----------------|----------------|
  | Cancel     | -- (coalesces)  | OK             | DEFER          |
  | Place      | DEFER           | -- (coalesces) | impossible     |
  | Amend      | DISCARD         | impossible     | -- (coalesces) |

Tests cover:

  - kind=amend routes to execute_amend / execute_amend_batch
  - kind=place still routes to execute_place / execute_place_batch
  - coalesce: same (side, level, kind) collapses; (side, level) but
    different kind does NOT collapse
  - cross-lane DEFER for cancels while amend in-flight
  - cross-lane DISCARD for amends while cancel in-flight
  - counters increment correctly
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from app.enums import Side
from app.outbound_dispatch import (
    CancelTransportIntent,
    OutboundDispatchCoordinator,
    PlaceTransportIntent,
)
from tests.settings_helpers import UnitTestSettings


def _settings() -> tuple[UnitTestSettings, Path]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_amend_dispatch_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "ACTION_WS_ENABLED": False,
            # Force every place / amend through the batch path so the
            # tests deterministically check batch-routing behaviour.
            "BATCH_PLACES_ALWAYS": True,
            "BATCH_PLACES_ENABLED": True,
        }
    )
    return s, path


def _amend(local: str, side: Side, seq: int = 1, level: int = 0) -> PlaceTransportIntent:
    return PlaceTransportIntent(
        wo_order_id_local=local,
        side=side,
        intent_seq=seq,
        quote_cycle_id="qc",
        enqueued_mono=time.monotonic(),
        level_idx=level,
        kind="amend",
    )


def _place(local: str, side: Side, seq: int = 1, level: int = 0) -> PlaceTransportIntent:
    return PlaceTransportIntent(
        wo_order_id_local=local,
        side=side,
        intent_seq=seq,
        quote_cycle_id="qc",
        enqueued_mono=time.monotonic(),
        level_idx=level,
        kind="place",
    )


def test_amend_intent_routes_to_amend_batch_executor() -> None:
    """Two amend intents (BUY + SELL) queue together → single batch
    call to ``execute_amend_batch``. The place executors are never
    invoked."""
    s, path = _settings()
    amend_batch_calls: list[list[PlaceTransportIntent]] = []
    place_batch_calls: list[list[PlaceTransportIntent]] = []
    single_place_calls: list[PlaceTransportIntent] = []
    single_amend_calls: list[PlaceTransportIntent] = []
    done = threading.Event()

    def exec_place(p: PlaceTransportIntent) -> None:
        single_place_calls.append(p)

    def exec_cancel(_c: CancelTransportIntent) -> None:
        pass

    def exec_place_batch(ps: list[PlaceTransportIntent]) -> None:
        place_batch_calls.append(list(ps))

    def exec_amend(p: PlaceTransportIntent) -> None:
        single_amend_calls.append(p)

    def exec_amend_batch(ps: list[PlaceTransportIntent]) -> None:
        amend_batch_calls.append(list(ps))
        done.set()

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
        execute_place_batch=exec_place_batch,
        execute_amend=exec_amend,
        execute_amend_batch=exec_amend_batch,
    )
    d.start()
    try:
        d.submit_place(_amend("a", Side.BUY))
        d.submit_place(_amend("b", Side.SELL))
        assert done.wait(timeout=2.0), "amend batch never fired"
        d.wait_until_idle(2.0)
        assert len(amend_batch_calls) == 1
        assert {p.side for p in amend_batch_calls[0]} == {Side.BUY, Side.SELL}
        assert place_batch_calls == []
        assert single_place_calls == []
        assert single_amend_calls == []
        stats = d.snapshot_stats()
        assert stats["batch_amend_dispatch_count"] == 1
        assert stats["batch_place_dispatch_count"] == 0
    finally:
        d.stop()
        path.unlink(missing_ok=True)


def test_place_intent_still_routes_to_place_batch_executor() -> None:
    """Regression: pre-Phase-2 places are unaffected. With kind="place"
    explicit and amend executors wired, places still go via place
    batch."""
    s, path = _settings()
    amend_batch_calls: list[list[PlaceTransportIntent]] = []
    place_batch_calls: list[list[PlaceTransportIntent]] = []
    done = threading.Event()

    def exec_place(_p: PlaceTransportIntent) -> None:
        pass

    def exec_cancel(_c: CancelTransportIntent) -> None:
        pass

    def exec_place_batch(ps: list[PlaceTransportIntent]) -> None:
        place_batch_calls.append(list(ps))
        done.set()

    def exec_amend_batch(ps: list[PlaceTransportIntent]) -> None:
        amend_batch_calls.append(list(ps))

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
        execute_place_batch=exec_place_batch,
        execute_amend_batch=exec_amend_batch,
    )
    d.start()
    try:
        d.submit_place(_place("a", Side.BUY))
        d.submit_place(_place("b", Side.SELL))
        assert done.wait(timeout=2.0)
        d.wait_until_idle(2.0)
        assert len(place_batch_calls) == 1
        assert amend_batch_calls == []
        stats = d.snapshot_stats()
        assert stats["batch_place_dispatch_count"] == 1
        assert stats["batch_amend_dispatch_count"] == 0
    finally:
        d.stop()
        path.unlink(missing_ok=True)


def test_coalesce_same_kind_same_slot_collapses() -> None:
    """Two amends for (BUY, level=0) submitted back-to-back collapse
    to the LATEST one. The dispatcher's per-slot uniqueness rule still
    applies within a kind."""
    s, path = _settings()
    captured: list[PlaceTransportIntent] = []
    done = threading.Event()

    def exec_amend_batch(ps: list[PlaceTransportIntent]) -> None:
        captured.extend(ps)
        done.set()

    def exec_amend(p: PlaceTransportIntent) -> None:
        captured.append(p)
        done.set()

    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=lambda _c: None,
        execute_place_batch=lambda _ps: None,
        execute_amend=exec_amend,
        execute_amend_batch=exec_amend_batch,
    )
    # Submit while paused so both intents enqueue before the worker wakes.
    a1 = _amend("a1", Side.BUY, seq=1)
    a2 = _amend("a2", Side.BUY, seq=2)
    d.submit_place(a1)
    d.submit_place(a2)
    # Now start — only the second amend should survive.
    d.start()
    try:
        assert done.wait(timeout=2.0)
        d.wait_until_idle(2.0)
        # Latest amend (a2) wins coalesce.
        survivors = [p.wo_order_id_local for p in captured]
        assert "a2" in survivors
        assert "a1" not in survivors
    finally:
        d.stop()
        path.unlink(missing_ok=True)


def test_coalesce_different_kind_same_slot_does_not_collapse() -> None:
    """A place and an amend for (BUY, level=0) target DIFFERENT venue
    orders — the coalesce key (side, level, kind) keeps them separate.
    Both reach their respective executor.

    In production this scenario shouldn't normally arise (WO state
    machine gates which path is eligible), but the dispatcher's
    coalesce invariant must not silently drop one — that would be a
    latent state-drift bug.
    """
    s, path = _settings()
    place_intents: list[PlaceTransportIntent] = []
    amend_intents: list[PlaceTransportIntent] = []
    place_done = threading.Event()
    amend_done = threading.Event()

    def exec_place_batch(ps: list[PlaceTransportIntent]) -> None:
        place_intents.extend(ps)
        place_done.set()

    def exec_amend_batch(ps: list[PlaceTransportIntent]) -> None:
        amend_intents.extend(ps)
        amend_done.set()

    def exec_place(p: PlaceTransportIntent) -> None:
        place_intents.append(p)
        place_done.set()

    def exec_amend(p: PlaceTransportIntent) -> None:
        amend_intents.append(p)
        amend_done.set()

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=lambda _c: None,
        execute_place_batch=exec_place_batch,
        execute_amend=exec_amend,
        execute_amend_batch=exec_amend_batch,
    )
    # Submit both BEFORE start to lock the queue ordering.
    d.submit_place(_place("p1", Side.BUY, seq=1))
    d.submit_place(_amend("a1", Side.BUY, seq=1))
    d.start()
    try:
        assert place_done.wait(timeout=2.0)
        assert amend_done.wait(timeout=2.0)
        d.wait_until_idle(2.0)
        assert {p.wo_order_id_local for p in place_intents} == {"p1"}
        assert {p.wo_order_id_local for p in amend_intents} == {"a1"}
    finally:
        d.stop()
        path.unlink(missing_ok=True)


def test_cancel_defers_when_amend_inflight() -> None:
    """The cross-lane rule: a cancel arriving while an amend for the
    same slot is in-flight DEFERS. Counter
    ``cancel_deferred_amend_inflight_total`` increments.

    We simulate "amend in flight" by holding the amend executor on a
    threading.Event — the cancel submitted while it's blocked must
    leftover-requeue."""
    s, path = _settings()
    amend_release = threading.Event()
    amend_started = threading.Event()
    cancel_fired = threading.Event()
    cancel_calls: list[CancelTransportIntent] = []

    def exec_amend_batch(_ps: list[PlaceTransportIntent]) -> None:
        amend_started.set()
        amend_release.wait(timeout=3.0)

    def exec_amend(_p: PlaceTransportIntent) -> None:
        amend_started.set()
        amend_release.wait(timeout=3.0)

    def exec_cancel(c: CancelTransportIntent) -> None:
        cancel_calls.append(c)
        cancel_fired.set()

    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=exec_cancel,
        execute_place_batch=lambda _ps: None,
        execute_amend=exec_amend,
        execute_amend_batch=exec_amend_batch,
    )
    d.start()
    try:
        d.submit_place(_amend("a", Side.BUY))
        assert amend_started.wait(timeout=2.0), "amend never started"
        # Now amend is in flight. Submit a cancel for the SAME slot.
        d.submit_cancel(
            CancelTransportIntent(
                "a", Side.BUY, 1, time.monotonic(), level_idx=0
            )
        )
        # Give the dispatcher a moment to attempt + defer.
        time.sleep(0.1)
        stats_before_release = d.snapshot_stats()
        assert int(stats_before_release["cancel_deferred_amend_inflight_total"]) >= 1
        # Cancel must NOT have fired yet.
        assert not cancel_fired.is_set()
        # Release the amend — cancel should now drain.
        amend_release.set()
        assert cancel_fired.wait(timeout=2.0), "cancel never fired after amend release"
        d.wait_until_idle(2.0)
        assert len(cancel_calls) == 1
    finally:
        amend_release.set()
        d.stop()
        path.unlink(missing_ok=True)


def test_amend_discarded_when_cancel_inflight() -> None:
    """The cross-lane rule: an amend arriving while a cancel for the
    same slot is in-flight is DISCARDED (the WO is about to
    disappear; no target to amend). The amend executor never fires
    for the discarded amend. Counter
    ``amend_discarded_cancel_inflight_total`` increments."""
    s, path = _settings()
    cancel_release = threading.Event()
    cancel_started = threading.Event()
    amend_called = threading.Event()
    amend_batch_calls: list[list[PlaceTransportIntent]] = []

    def exec_cancel(_c: CancelTransportIntent) -> None:
        cancel_started.set()
        cancel_release.wait(timeout=3.0)

    def exec_amend(p: PlaceTransportIntent) -> None:
        amend_batch_calls.append([p])
        amend_called.set()

    def exec_amend_batch(ps: list[PlaceTransportIntent]) -> None:
        amend_batch_calls.append(list(ps))
        amend_called.set()

    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=exec_cancel,
        execute_place_batch=lambda _ps: None,
        execute_amend=exec_amend,
        execute_amend_batch=exec_amend_batch,
    )
    d.start()
    try:
        # Hold the cancel in flight.
        d.submit_cancel(
            CancelTransportIntent(
                "a", Side.BUY, 1, time.monotonic(), level_idx=0
            )
        )
        assert cancel_started.wait(timeout=2.0), "cancel never started"
        # Now submit an amend for the SAME slot — must be discarded.
        d.submit_place(_amend("a", Side.BUY))
        # Give the dispatcher a moment to drain + discard.
        time.sleep(0.1)
        stats = d.snapshot_stats()
        assert int(stats["amend_discarded_cancel_inflight_total"]) >= 1
        # Release the cancel; ensure the amend was NOT subsequently
        # called (true discard, not just deferred).
        cancel_release.set()
        d.wait_until_idle(2.0)
        # If the discard worked, the amend executor should never have
        # been invoked.
        assert not amend_called.is_set(), (
            "amend executor fired despite being discarded for "
            "cross-lane cancel-inflight"
        )
        assert amend_batch_calls == []
    finally:
        cancel_release.set()
        d.stop()
        path.unlink(missing_ok=True)


def test_amend_intent_default_kind_is_place() -> None:
    """Backward-compat: a PlaceTransportIntent constructed without
    ``kind=`` defaults to ``"place"``. Existing call-sites in
    execution.py do not specify ``kind`` and must keep going to the
    place executor."""
    intent = PlaceTransportIntent(
        wo_order_id_local="x",
        side=Side.BUY,
        intent_seq=1,
        quote_cycle_id="qc",
        enqueued_mono=time.monotonic(),
    )
    assert intent.kind == "place"


def test_amend_dispatch_counters_zero_when_no_amend_traffic() -> None:
    """Pre-amend deployment regression: when no amend intents queue,
    the amend counters stay at 0 and the place counters are
    unaffected."""
    s, path = _settings()

    def exec_place_batch(_ps: list[PlaceTransportIntent]) -> None:
        pass

    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=lambda _c: None,
        execute_place_batch=exec_place_batch,
        execute_amend_batch=lambda _ps: None,
    )
    d.start()
    try:
        d.submit_place(_place("p", Side.BUY))
        d.wait_until_idle(2.0)
        stats = d.snapshot_stats()
        assert stats["batch_amend_dispatch_count"] == 0
        assert stats["amend_discarded_cancel_inflight_total"] == 0
        assert stats["cancel_deferred_amend_inflight_total"] == 0
    finally:
        d.stop()
        path.unlink(missing_ok=True)
