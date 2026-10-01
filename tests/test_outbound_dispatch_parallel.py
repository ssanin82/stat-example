"""Phase 3 (v1.3.109) regression — parallel cancel + place worker
threads in ``OutboundDispatchCoordinator``.

Invariants under test:

1. **Concurrent dispatch**: a slow place HTTP must NOT block a cancel
   for a different side. With the legacy single-worker mode, the
   cancel HTTP would start only after the place HTTP completed; with
   parallel workers, the cancel starts immediately.

2. **Same-side cross-lane defer**: a place for side S held back
   while a cancel for side S is in flight. Restores the
   "cancel-before-place" ordering the single-worker mode achieved
   implicitly via sequential per-batch dispatch. Counted in
   ``cross_lane_deferrals``.

3. **Different-side independence**: cancel for BUY and place for SELL
   may run concurrently (no defer).

4. **Rollback knob**: ``OUTBOUND_CANCEL_WORKER_ENABLED=false`` reverts
   to the single-worker path and spawns exactly one thread.

5. **start/stop lifecycle**: ``stop()`` joins both worker threads;
   second ``stop()`` is a no-op.
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


def _settings(**overrides) -> UnitTestSettings:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_od_par_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "ACTION_BATCH_INTERVAL_MS": 0,
        "ACTION_MAX_BATCH_SIZE": 8,
        "ACTION_WS_ENABLED": False,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _place(local_id: str, side: Side, seq: int) -> PlaceTransportIntent:
    return PlaceTransportIntent(
        wo_order_id_local=local_id,
        side=side,
        intent_seq=seq,
        quote_cycle_id=f"cyc-{seq}",
        enqueued_mono=time.monotonic(),
        intent_created_perf=0.0,
    )


def _cancel(local_id: str, side: Side, seq: int) -> CancelTransportIntent:
    return CancelTransportIntent(
        wo_order_id_local=local_id,
        side=side,
        intent_seq=seq,
        enqueued_mono=time.monotonic(),
    )


# --------------------------------------------------------------------------
# Concurrent dispatch — cancel must not wait for an in-flight place
# --------------------------------------------------------------------------


def test_parallel_cancel_does_not_wait_for_in_flight_place() -> None:
    """A 100 ms place HTTP must not delay a cancel for a different
    side. Measures wall-clock time between submit and exec callback;
    on the legacy single-worker path the cancel would land at t≈100ms,
    on the parallel path it lands at t≈0 ms (just thread wake)."""
    place_started = threading.Event()
    place_done = threading.Event()
    cancel_started_at: list[float] = []
    place_started_at: list[float] = []

    def exec_place(_p: PlaceTransportIntent) -> None:
        place_started_at.append(time.monotonic())
        place_started.set()
        # Hold the place worker for 100 ms — simulates a slow OKX
        # place HTTP under load.
        time.sleep(0.1)
        place_done.set()

    def exec_cancel(_c: CancelTransportIntent) -> None:
        cancel_started_at.append(time.monotonic())

    s = _settings()
    d = OutboundDispatchCoordinator(
        s, execute_place=exec_place, execute_cancel=exec_cancel
    )
    d.start()
    try:
        t0 = time.monotonic()
        d.submit_place(_place("p1", Side.BUY, 1))
        # Wait until the place worker has actually entered its HTTP
        # callback so we know the place lane is "busy".
        assert place_started.wait(1.0), "place worker did not start"
        # Submit a cancel for the OPPOSITE side so cross-lane defer
        # doesn't apply.
        d.submit_cancel(_cancel("c1", Side.SELL, 2))
        assert place_done.wait(2.0), "place worker did not finish"

        # Cancel must have started before the place finished, NOT after.
        assert cancel_started_at, "cancel never executed"
        cancel_t = cancel_started_at[0] - t0
        place_t = place_started_at[0] - t0
        # Cancel must start within ~50 ms of place starting (well
        # under the 100 ms place HTTP duration). Single-worker mode
        # would put cancel at place_t + 100 ms.
        assert cancel_t < place_t + 0.08, (
            f"cancel started too late: cancel_t={cancel_t*1000:.1f}ms, "
            f"place_t={place_t*1000:.1f}ms (single-worker would be ~100ms gap)"
        )
    finally:
        d.stop()


def test_parallel_different_side_cancel_place_run_concurrently() -> None:
    """Cancel for BUY and place for SELL must run concurrently — no
    cross-lane defer applies for different sides."""
    in_flight = {"count": 0, "max": 0}
    lock = threading.Lock()

    def _enter():
        with lock:
            in_flight["count"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["count"])
        time.sleep(0.05)
        with lock:
            in_flight["count"] -= 1

    def exec_place(_p: PlaceTransportIntent) -> None:
        _enter()

    def exec_cancel(_c: CancelTransportIntent) -> None:
        _enter()

    s = _settings()
    d = OutboundDispatchCoordinator(
        s, execute_place=exec_place, execute_cancel=exec_cancel
    )
    d.start()
    try:
        d.submit_cancel(_cancel("c1", Side.BUY, 1))
        d.submit_place(_place("p1", Side.SELL, 2))
        d.wait_until_idle(2.0)
        assert in_flight["max"] >= 2, (
            f"expected cancel+place to overlap (max-in-flight=2); got "
            f"{in_flight['max']}"
        )
    finally:
        d.stop()


# --------------------------------------------------------------------------
# Cross-lane defer — place for side S waits while cancel for side S is live
# --------------------------------------------------------------------------


def test_parallel_same_side_place_defers_while_cancel_in_flight() -> None:
    """A 100 ms cancel for side BUY must hold back a place for BUY
    until the cancel completes — preserves the cancel-before-place
    invariant. Verifies the place's exec timestamp is AT OR AFTER the
    cancel's exec returned, not concurrent with it.

    Implementation note: the place worker uses a predicate-based wait
    (``_place_has_runnable_locked``) that holds the intent in the lane
    rather than popping-and-re-queuing when ALL candidates would be
    blocked by an active cancel. As a result, ``cross_lane_deferrals``
    does NOT increment in this single-intent scenario — the intent is
    never popped while blocked. The counter increments only in mixed-
    side flushes (covered by
    ``test_parallel_mixed_side_flush_counts_cross_lane_defer``)."""
    cancel_done_at: list[float] = []
    place_started_at: list[float] = []
    cancel_holding = threading.Event()

    def exec_cancel(_c: CancelTransportIntent) -> None:
        cancel_holding.set()
        time.sleep(0.1)
        cancel_done_at.append(time.monotonic())

    def exec_place(_p: PlaceTransportIntent) -> None:
        place_started_at.append(time.monotonic())

    s = _settings()
    d = OutboundDispatchCoordinator(
        s, execute_place=exec_place, execute_cancel=exec_cancel
    )
    d.start()
    try:
        d.submit_cancel(_cancel("c1", Side.BUY, 1))
        # Wait until the cancel worker is mid-HTTP so the cross-lane
        # state is set before we submit the place.
        assert cancel_holding.wait(1.0), "cancel did not start"
        d.submit_place(_place("p1", Side.BUY, 2))
        d.wait_until_idle(2.0)

        assert cancel_done_at and place_started_at
        # Place must start AT OR AFTER cancel's completion. ~5 ms
        # slack for thread-wake + lock-reacquire after the cancel
        # worker's notify_all.
        gap = place_started_at[0] - cancel_done_at[0]
        assert gap >= -0.005, (
            f"place started {gap*1000:.1f}ms BEFORE cancel completed — "
            f"cross-lane defer failed"
        )
    finally:
        d.stop()


def test_parallel_mixed_side_flush_counts_cross_lane_defer() -> None:
    """When the place lane carries BOTH a blocked side (BUY, with
    active cancel) AND an unblocked side (SELL), the predicate
    ``_place_has_runnable_locked`` returns True (SELL is runnable),
    the worker calls ``_flush_place_batch``, the SELL place runs and
    the BUY place is popped, found blocked, and re-queued. This is
    the path that increments ``cross_lane_deferrals`` in live
    parallel mode."""
    cancel_holding = threading.Event()
    placed: list[Side] = []

    def exec_cancel(_c: CancelTransportIntent) -> None:
        cancel_holding.set()
        time.sleep(0.1)  # hold BUY active long enough for the mixed flush

    def exec_place(p: PlaceTransportIntent) -> None:
        placed.append(p.side)

    s = _settings()
    d = OutboundDispatchCoordinator(
        s, execute_place=exec_place, execute_cancel=exec_cancel
    )
    d.start()
    try:
        d.submit_cancel(_cancel("c1", Side.BUY, 1))
        assert cancel_holding.wait(1.0), "cancel did not start"
        # Submit BOTH a same-side place (blocked) AND an opposite-side
        # place (unblocked) before the cancel completes.
        d.submit_place(_place("p_buy", Side.BUY, 2))
        d.submit_place(_place("p_sell", Side.SELL, 3))
        d.wait_until_idle(2.0)

        # Both places must eventually run.
        assert Side.BUY in placed and Side.SELL in placed
        # SELL must run before BUY (BUY waits for cancel).
        assert placed.index(Side.SELL) < placed.index(Side.BUY), (
            f"SELL place must dispatch before BUY (blocked by cancel); "
            f"got order {placed}"
        )
        # The mixed-side flush popped both and re-queued BUY, which
        # increments the counter.
        stats = d.snapshot_stats()
        assert int(stats["cross_lane_deferrals"]) >= 1, (
            f"cross_lane_deferrals must increment when a mixed-side "
            f"flush re-queues a blocked place; got "
            f"{stats['cross_lane_deferrals']}"
        )
    finally:
        d.stop()


def test_parallel_place_runs_immediately_when_no_cancel_in_flight() -> None:
    """Symmetric to the defer test: when there's NO cancel for the
    same side, the place runs without any cross-lane delay."""
    place_started_at: list[float] = []

    def exec_place(_p: PlaceTransportIntent) -> None:
        place_started_at.append(time.monotonic())

    def exec_cancel(_c: CancelTransportIntent) -> None:
        pass

    s = _settings()
    d = OutboundDispatchCoordinator(
        s, execute_place=exec_place, execute_cancel=exec_cancel
    )
    d.start()
    try:
        t0 = time.monotonic()
        d.submit_place(_place("p1", Side.BUY, 1))
        d.wait_until_idle(1.0)

        assert place_started_at
        latency_ms = (place_started_at[0] - t0) * 1000
        assert latency_ms < 80, (
            f"place latency {latency_ms:.1f}ms too high — no cancel "
            f"in flight should mean near-zero wake delay"
        )
        # No cross-lane defer should have fired.
        assert int(d.snapshot_stats()["cross_lane_deferrals"]) == 0
    finally:
        d.stop()


# --------------------------------------------------------------------------
# Topology + rollback knob
# --------------------------------------------------------------------------


def test_parallel_mode_spawns_two_threads_default() -> None:
    s = _settings()
    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=lambda _c: None,
    )
    d.start()
    try:
        assert d._cancel_thread is not None
        assert d._place_thread is not None
        assert d._thread is None
        assert d._cancel_thread.is_alive()
        assert d._place_thread.is_alive()
        # Thread names are exposed for operator visibility (e.g. py-spy).
        assert "cancel" in d._cancel_thread.name
        assert "place" in d._place_thread.name
    finally:
        d.stop()


def test_legacy_single_worker_when_flag_disabled() -> None:
    """``OUTBOUND_CANCEL_WORKER_ENABLED=false`` reverts to the v1.3.108-
    and-earlier single-worker path. Useful as a one-line operator
    rollback if a parallel-mode regression is suspected."""
    s = _settings(OUTBOUND_CANCEL_WORKER_ENABLED=False)
    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=lambda _c: None,
    )
    d.start()
    try:
        assert d._thread is not None
        assert d._cancel_thread is None
        assert d._place_thread is None
        assert d._thread.is_alive()
    finally:
        d.stop()


def test_stop_joins_both_workers_idempotently() -> None:
    s = _settings()
    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=lambda _c: None,
    )
    d.start()
    d.stop()
    # Both worker threads must have exited.
    assert d._cancel_thread is not None and not d._cancel_thread.is_alive()
    assert d._place_thread is not None and not d._place_thread.is_alive()
    # Second stop is a no-op (join on a completed thread returns).
    d.stop()


# --------------------------------------------------------------------------
# Stats surface
# --------------------------------------------------------------------------


def test_snapshot_stats_exposes_cross_lane_deferrals() -> None:
    """``snapshot_stats`` must surface the new Phase 3 counter so the
    operator can verify the cross-lane guard fires in production."""
    s = _settings()
    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=lambda _c: None,
    )
    stats = d.snapshot_stats()
    assert "cross_lane_deferrals" in stats
    assert int(stats["cross_lane_deferrals"]) == 0


# --------------------------------------------------------------------------
# Direct-call flush methods (synchronous, deterministic) — exercise the
# locked-state-machine code paths without involving thread scheduling
# --------------------------------------------------------------------------


def test_flush_place_batch_defers_when_cancel_side_active() -> None:
    """Synchronously seed ``_active_cancel_sides`` and a place lane,
    call ``_flush_place_batch`` directly. The place must be deferred
    and ``cross_lane_deferrals`` must increment."""
    s = _settings()
    executed: list[PlaceTransportIntent] = []
    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda p: executed.append(p),
        execute_cancel=lambda _c: None,
    )
    # 1.3.130 multi-rung Phase 2: active-cancel set is keyed on
    # (side, level_idx). For the legacy N=1 scenario this test models,
    # the key is (Side.BUY, 0).
    d._active_cancel_sides.add((Side.BUY, 0))
    d._place_lane.append(_place("p1", Side.BUY, 1))

    progress = d._flush_place_batch()

    assert progress == 0
    assert len(executed) == 0
    assert len(d._place_lane) == 1  # re-queued
    assert d._cross_lane_deferrals == 1

    # Now clear the cancel flag and re-flush — place must dispatch.
    d._active_cancel_sides.discard((Side.BUY, 0))
    progress = d._flush_place_batch()
    assert progress == 1
    assert len(executed) == 1
    assert len(d._place_lane) == 0


def test_flush_cancel_batch_does_not_check_place_side() -> None:
    """The cancel worker does NOT defer on ``_active_place_sides`` —
    cancel targets an old ordId, place creates a new one; no venue
    conflict. Verifies cancel for BUY runs even when a place for BUY
    is marked in-flight."""
    s = _settings()
    executed: list[CancelTransportIntent] = []
    d = OutboundDispatchCoordinator(
        s,
        execute_place=lambda _p: None,
        execute_cancel=lambda c: executed.append(c),
    )
    d._active_place_sides.add(Side.BUY)  # simulate place in flight
    d._cancel_lane.append(_cancel("c1", Side.BUY, 1))

    progress = d._flush_cancel_batch()

    assert progress == 1
    assert len(executed) == 1
    # No cross-lane defer should have been counted.
    assert d._cross_lane_deferrals == 0
