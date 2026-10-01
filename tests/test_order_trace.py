"""Tests for app.order_trace.OrderTraceBuffer.

Replay-style tests for the three branches Phase 2 distinguishes:

  1. Post-only cross at submit (terminal source = "place_response")
  2. Healthy ws-driven terminal (filled / cancel via WS)
  3. gone_on_exchange via reconcile, with vs without prior WS events
     — the diagnostic split that motivates Phase 2 in the first place.
"""

from __future__ import annotations

import threading
import time

from app.order_trace import OrderTraceBuffer


def test_basic_lifecycle_accepted_then_terminal_via_ws() -> None:
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="cloid-A", side="BUY", price=42.30, size_base=0.5)
    buf.record_place_response(
        client_order_id="cloid-A",
        outcome="accepted",
        detail="",
        order_id_exchange=1001,
    )
    buf.record_ws_event(order_id_exchange=1001, state="live")
    buf.record_ws_event(order_id_exchange=1001, state="filled")
    buf.record_terminal(
        order_id_exchange=1001,
        status="FILLED",
        reason="ws_order_status",
        source="ws",
    )
    [entry] = buf.to_list()
    assert entry["client_order_id"] == "cloid-A"
    assert entry["place_outcome"] == "accepted"
    assert entry["ws_event_count"] == 2
    assert [e["state"] for e in entry["ws_events"]] == ["live", "filled"]
    assert entry["terminal_source"] == "ws"
    assert entry["terminal_status"] == "FILLED"


def test_branch_1_post_only_cross_at_submit() -> None:
    """Place response says exchange_rejected. Terminal source must be
    place_response and ws_events stays empty (the order never made it
    to the venue).
    """
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="cloid-PO", side="SELL", price=42.30, size_base=0.5)
    buf.record_place_response(
        client_order_id="cloid-PO",
        outcome="exchange_rejected",
        detail="post_only_would_cross:cross_with_bid",
        order_id_exchange=None,
    )
    buf.record_terminal(
        client_order_id="cloid-PO",
        status="REJECTED",
        reason="post_only_would_cross:cross_with_bid",
        source="place_response",
    )
    [entry] = buf.to_list()
    assert entry["place_outcome"] == "exchange_rejected"
    assert "post_only_would_cross" in (entry["place_outcome_detail"] or "")
    assert entry["ws_event_count"] == 0
    assert entry["terminal_source"] == "place_response"
    assert entry["order_id_exchange"] is None


def test_branch_3a_gone_on_exchange_with_ws_events_state_machine_race() -> None:
    """The bug signature for a state-machine race: WS DID deliver a
    cancel event, but reconcile fired and called the order gone before
    the local state machine applied it. The trace shows ws_events
    populated AND terminal_source='reconcile'.
    """
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="cloid-RACE", side="BUY", price=42.30, size_base=0.5)
    buf.record_place_response(
        client_order_id="cloid-RACE",
        outcome="accepted",
        detail="",
        order_id_exchange=2002,
    )
    buf.record_ws_event(order_id_exchange=2002, state="live")
    buf.record_ws_event(order_id_exchange=2002, state="canceled")
    buf.record_terminal(
        order_id_exchange=2002,
        status="CANCELED",
        reason="gone_on_exchange",
        source="reconcile",
    )
    [entry] = buf.to_list()
    assert entry["place_outcome"] == "accepted"
    assert entry["ws_event_count"] == 2
    assert entry["terminal_source"] == "reconcile"
    assert entry["terminal_reason"] == "gone_on_exchange"


def test_branch_3b_gone_on_exchange_without_ws_events_silent_ws() -> None:
    """The bug signature for connectivity / silent-WS: place was
    accepted with an oid, but no orders-channel event ever arrived
    for it; reconcile cleaned up. ws_events stays empty.
    """
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="cloid-SILENT", side="SELL", price=42.30, size_base=0.5)
    buf.record_place_response(
        client_order_id="cloid-SILENT",
        outcome="accepted",
        detail="",
        order_id_exchange=3003,
    )
    buf.record_terminal(
        order_id_exchange=3003,
        status="CANCELED",
        reason="gone_on_exchange",
        source="reconcile",
    )
    [entry] = buf.to_list()
    assert entry["place_outcome"] == "accepted"
    assert entry["ws_event_count"] == 0
    assert entry["terminal_source"] == "reconcile"


def test_ring_eviction_by_age() -> None:
    """When the buffer hits its cap, the oldest entry is evicted and
    its index keys (cloid + oid) are removed so the indexes stay
    bounded.
    """
    buf = OrderTraceBuffer(max_entries=3)
    for i in range(5):
        buf.begin_order(
            client_order_id=f"c-{i}",
            side="BUY",
            price=42.0 + i,
            size_base=0.1,
        )
        buf.record_place_response(
            client_order_id=f"c-{i}",
            outcome="accepted",
            detail="",
            order_id_exchange=1000 + i,
        )
    out = buf.to_list()
    assert len(out) == 3
    assert [e["client_order_id"] for e in out] == ["c-2", "c-3", "c-4"]
    # Recording a WS event for an evicted oid should be a silent no-op.
    buf.record_ws_event(order_id_exchange=1000, state="live")  # c-0 long evicted
    out2 = buf.to_list()
    assert all(e["ws_event_count"] == 0 for e in out2)


def test_idempotent_begin_order() -> None:
    """Re-calling begin_order with the same cloid is a no-op so a
    transport retry doesn't create duplicate entries.
    """
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="dup", side="BUY", price=42.0, size_base=0.5)
    buf.begin_order(client_order_id="dup", side="BUY", price=42.0, size_base=0.5)
    assert len(buf) == 1


def test_terminal_only_records_first() -> None:
    """If multiple terminal sources race (e.g. reconcile fires, then
    a delayed WS canceled arrives), the FIRST terminal stays — that's
    the cause-of-death. Later events don't overwrite it.
    """
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="t", side="BUY", price=42.0, size_base=0.5)
    buf.record_place_response(
        client_order_id="t",
        outcome="accepted",
        detail="",
        order_id_exchange=99,
    )
    buf.record_terminal(
        order_id_exchange=99,
        status="CANCELED",
        reason="gone_on_exchange",
        source="reconcile",
    )
    buf.record_terminal(
        order_id_exchange=99,
        status="CANCELED",
        reason="ws_late",
        source="ws",
    )
    [entry] = buf.to_list()
    assert entry["terminal_reason"] == "gone_on_exchange"
    assert entry["terminal_source"] == "reconcile"


def test_thread_safety_under_concurrent_writers() -> None:
    """Sanity check: concurrent begin_order from multiple threads
    doesn't corrupt the buffer or its indexes.
    """
    buf = OrderTraceBuffer(max_entries=1000)

    def writer(start: int) -> None:
        for i in range(start, start + 100):
            buf.begin_order(
                client_order_id=f"w-{i}",
                side="BUY",
                price=42.0,
                size_base=0.1,
            )

    threads = [threading.Thread(target=writer, args=(i * 100,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = buf.to_list()
    assert len(out) == 500
    assert len({e["client_order_id"] for e in out}) == 500


def test_has_pending_ws_terminal_for_canceled() -> None:
    """The race-fix predicate: when the most recent WS event for an oid
    is a terminal state, ``has_pending_ws_terminal`` returns True so
    ``_reconcile_side`` defers gone_on_exchange to next tick.
    """
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="r", side="BUY", price=42.0, size_base=0.5)
    buf.record_place_response(
        client_order_id="r",
        outcome="accepted",
        detail="",
        order_id_exchange=42,
    )
    # No WS events yet — predicate False.
    assert buf.has_pending_ws_terminal(42) is False
    buf.record_ws_event(order_id_exchange=42, state="live")
    # 'live' isn't terminal — predicate still False.
    assert buf.has_pending_ws_terminal(42) is False
    buf.record_ws_event(order_id_exchange=42, state="canceled")
    # Now we have a pending terminal — predicate True.
    assert buf.has_pending_ws_terminal(42) is True


def test_has_pending_ws_terminal_for_filled() -> None:
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="f", side="SELL", price=42.0, size_base=0.5)
    buf.record_place_response(
        client_order_id="f",
        outcome="accepted",
        detail="",
        order_id_exchange=43,
    )
    buf.record_ws_event(order_id_exchange=43, state="live")
    buf.record_ws_event(order_id_exchange=43, state="partially_filled")
    # partially_filled is NOT terminal — order still active.
    assert buf.has_pending_ws_terminal(43) is False
    buf.record_ws_event(order_id_exchange=43, state="filled")
    assert buf.has_pending_ws_terminal(43) is True


def test_has_pending_ws_terminal_for_mmp_canceled() -> None:
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="m", side="BUY", price=42.0, size_base=0.5)
    buf.record_place_response(
        client_order_id="m",
        outcome="accepted",
        detail="",
        order_id_exchange=44,
    )
    buf.record_ws_event(order_id_exchange=44, state="mmp_canceled")
    assert buf.has_pending_ws_terminal(44) is True


def test_has_pending_ws_terminal_stale_by_age() -> None:
    """A WS terminal older than max_age_s must NOT count as pending —
    drain has had ample time to process it. Without this guard, the
    watchdog deadlock seen in snapshot 260508105856 reproduces:
    reconcile_skip_gone_pending_ws fires forever on a stale event.
    """
    import time
    from datetime import datetime, timezone, timedelta

    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="age", side="BUY", price=42.0, size_base=0.5)
    buf.record_place_response(
        client_order_id="age",
        outcome="accepted",
        detail="",
        order_id_exchange=77,
    )
    buf.record_ws_event(order_id_exchange=77, state="canceled")
    # Right after the WS event: predicate returns True (real race).
    assert buf.has_pending_ws_terminal(77, max_age_s=5.0) is True
    # Manually back-date the recorded event timestamp to simulate
    # 30 s of elapsed time since the event was recorded. (Reaching
    # into the private list is fine in tests — the field is the
    # whole point of this test.)
    with buf._lock:
        entry = buf._by_oid["77"]
        old_ts = (
            datetime.now(timezone.utc) - timedelta(seconds=30)
        ).isoformat()
        entry.ws_events[-1].ts = old_ts
    # 30 s old, max_age_s=5.0 → stale, predicate False, reconcile
    # may proceed with gone_on_exchange.
    assert buf.has_pending_ws_terminal(77, max_age_s=5.0) is False
    # With a more permissive window the same event still counts.
    assert buf.has_pending_ws_terminal(77, max_age_s=60.0) is True


def test_has_pending_ws_terminal_stale_by_lifecycle() -> None:
    """A WS terminal whose timestamp predates the working order's
    current ACK timestamp belongs to a PRIOR lifecycle of the same
    oid (re-hydrate scenario) and must NOT gate the new lifecycle.
    This is the specific bug behind the 1.1.44 watchdog deadlock.
    """
    from datetime import datetime, timezone, timedelta

    buf = OrderTraceBuffer()
    buf.begin_order(
        client_order_id="hydra", side="BUY", price=42.0, size_base=0.5,
    )
    buf.record_place_response(
        client_order_id="hydra",
        outcome="accepted",
        detail="",
        order_id_exchange=88,
    )
    buf.record_ws_event(order_id_exchange=88, state="canceled")
    # The WS event we just recorded — its ts is "now-ish".
    # Imagine the bot lost track of the order, re-hydrated it from
    # the venue 3 seconds later (fresh ts_ack on the new working
    # order). The trace's stale "canceled" event predates the new
    # ack and must be ignored by the predicate.
    fresh_ack = datetime.now(timezone.utc) + timedelta(seconds=3)
    assert (
        buf.has_pending_ws_terminal(88, ack_ts=fresh_ack, max_age_s=60.0)
        is False
    )
    # If reconcile had no ack_ts hint, the event would still pass
    # the age check (within 60 s). The lifecycle guard is the only
    # thing protecting us here.
    assert (
        buf.has_pending_ws_terminal(88, ack_ts=None, max_age_s=60.0)
        is True
    )
    # And conversely: if the ack is OLDER than the WS event (event
    # is real-race), predicate still returns True.
    older_ack = datetime.now(timezone.utc) - timedelta(seconds=10)
    assert (
        buf.has_pending_ws_terminal(88, ack_ts=older_ack, max_age_s=60.0)
        is True
    )


def test_has_pending_ws_terminal_unknown_oid() -> None:
    """Unknown oids / Nones / zeros all default to False (let reconcile
    proceed) — the guard should never block reconcile when we have no
    info on the order.
    """
    buf = OrderTraceBuffer()
    assert buf.has_pending_ws_terminal(None) is False
    assert buf.has_pending_ws_terminal(0) is False
    assert buf.has_pending_ws_terminal(99999) is False  # not in buffer


def test_to_list_is_a_copy_safe_to_iterate_outside_lock() -> None:
    buf = OrderTraceBuffer()
    buf.begin_order(client_order_id="x", side="BUY", price=42.0, size_base=0.5)
    snap = buf.to_list()
    buf.record_ws_event(order_id_exchange=1, state="live")  # missing oid, no-op
    buf.begin_order(client_order_id="y", side="SELL", price=42.0, size_base=0.5)
    # Older snap is unaffected by later mutations.
    assert len(snap) == 1
    assert snap[0]["client_order_id"] == "x"
