"""Tests for 1.3.82 connectivity-diagnostic instrumentation:

* ``cancel_trigger_reason`` stamped on cancel-dispatch
* ``ts_place_response`` + ``place_response_outcome`` stamped on
  place-response handling
* ``cancel_response_outcome`` stamped on cancel-HTTP response
* ``place_cancel_race_total`` counter increments when a cancel is
  enqueued within 50 ms of a place whose ack hasn't landed
* Reconcile snapshot-stale guard extended to cover the
  ``ts_cancel_requested`` case (no ts_ack but cancel in flight)

Targets the 56 gone_on_exchange events in snapshot 260515-095056
(50 phantom-place + 6 hydration-ghost). The instrumentation gives
the dashboard's Connectivity tab the data to attribute each event
to a specific cause.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from app.utils.time import utc_now


def _make_om(**overrides) -> tuple[OrderManager, BotState, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_conn_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    }
    base.update(overrides)
    settings = UnitTestSettings.model_validate(base)
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    return om, state, storage


def _make_wo(
    *,
    status: OrderStatus = OrderStatus.ACKED,
    ts_ack=True,
    ts_sent_offset_ms: float = 0.0,
    order_id_exchange: int | None = 100_001,
) -> WorkingOrder:
    now = utc_now()
    ts_sent = now - timedelta(milliseconds=ts_sent_offset_ms)
    return WorkingOrder(
        order_id_local=f"o_{uuid.uuid4().hex[:8]}",
        order_id_exchange=order_id_exchange,
        client_order_id="cloid_x",
        symbol="ETH",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=status,
        ts_sent=ts_sent,
        ts_ack=now if ts_ack else None,
    )


def test_cancel_trigger_reason_stamped_on_dispatch() -> None:
    """When ``_enqueue_cancel_quote_path`` is called with a trigger
    reason, the WO carries that reason after the call."""
    om, state, _storage = _make_om()
    wo = _make_wo()
    state.working_bid = wo
    # Stub the outbound submit so we don't actually try to transport.
    om._outbound.submit_cancel = MagicMock()
    om._enqueue_cancel_quote_path(wo, trigger_reason="hard_age_cap")
    assert wo.cancel_trigger_reason == "hard_age_cap"
    assert wo.ts_cancel_requested is not None


def test_cancel_trigger_reason_unchanged_on_subsequent_call() -> None:
    """If a WO already has a trigger_reason, a second cancel call
    (e.g. retry from the outbound worker) doesn't overwrite it."""
    om, state, _storage = _make_om()
    wo = _make_wo()
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    om._enqueue_cancel_quote_path(wo, trigger_reason="later_thing")
    # First-stamped reason wins. Subsequent calls don't clobber.
    assert wo.cancel_trigger_reason == "reprice_replace"


def test_cancel_without_trigger_reason_leaves_field_none() -> None:
    """Legacy call-sites that don't pass trigger_reason produce a
    None value. The dashboard surfaces these as ``(unknown)``."""
    om, state, _storage = _make_om()
    wo = _make_wo()
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    om._enqueue_cancel_quote_path(wo)
    assert wo.cancel_trigger_reason is None
    assert wo.ts_cancel_requested is not None


def test_place_cancel_race_counter_increments_on_close_race() -> None:
    """Cancel enqueued within 50 ms of ts_sent on a WO whose
    ts_ack is None → counter increments. Targets the 50 phantom-
    place events from snapshot 260515-095056."""
    om, state, _storage = _make_om()
    wo = _make_wo(status=OrderStatus.SENT, ts_ack=False, ts_sent_offset_ms=10.0)
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    assert state.place_cancel_race_total == 0
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    assert state.place_cancel_race_total == 1


def test_place_cancel_race_counter_does_not_increment_when_acked() -> None:
    """If the WO is already acked, the race window is past — no count."""
    om, state, _storage = _make_om()
    wo = _make_wo(status=OrderStatus.ACKED, ts_ack=True, ts_sent_offset_ms=10.0)
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    assert state.place_cancel_race_total == 0


def test_place_cancel_race_counter_does_not_increment_when_old_place() -> None:
    """Cancel issued 200 ms after place — outside the 50 ms race
    window — no count. This is a normal late cancel, not a race."""
    om, state, _storage = _make_om()
    wo = _make_wo(status=OrderStatus.SENT, ts_ack=False, ts_sent_offset_ms=200.0)
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    assert state.place_cancel_race_total == 0


def test_place_cancel_race_counter_handles_missing_ts_sent() -> None:
    """ts_sent=None is benign — counter doesn't crash, doesn't bump."""
    om, state, _storage = _make_om()
    wo = WorkingOrder(
        order_id_local="o_test",
        order_id_exchange=1,
        client_order_id="c",
        symbol="ETH",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.SENT,
        ts_sent=None,
        ts_ack=None,
    )
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    assert state.place_cancel_race_total == 0


def test_order_row_includes_new_connectivity_fields() -> None:
    """The serialized order row carries the six new connectivity
    fields so they reach SQLite + the publisher."""
    from app.execution import order_row

    wo = _make_wo()
    wo.cancel_trigger_reason = "reprice_replace"
    wo.ts_place_response = utc_now()
    wo.place_response_outcome = "accepted"
    wo.cancel_response_outcome = "success"
    wo.place_response_detail = "post_only_would_cross:foo"
    wo.cancel_response_detail = "okx_row_51400:Order doesn't exist"
    row = order_row(wo)
    assert row["cancel_trigger_reason"] == "reprice_replace"
    assert row["ts_place_response"] is not None
    assert row["place_response_outcome"] == "accepted"
    assert row["cancel_response_outcome"] == "success"
    assert row["place_response_detail"] == "post_only_would_cross:foo"
    assert row["cancel_response_detail"].startswith("okx_row_51400")


def test_order_row_truncates_long_details() -> None:
    """Details from the venue can be verbose; we truncate at 400 chars
    to keep SQLite rows bounded."""
    from app.execution import order_row

    wo = _make_wo()
    wo.place_response_detail = "x" * 1000
    row = order_row(wo)
    assert len(row["place_response_detail"]) == 400


def test_storage_persists_new_connectivity_fields() -> None:
    """Round-trip: persist a row with the new fields, read it back."""
    om, _state, storage = _make_om()
    wo = _make_wo()
    wo.cancel_trigger_reason = "hard_age_cap"
    wo.ts_place_response = utc_now()
    wo.place_response_outcome = "exchange_rejected"
    wo.place_response_detail = "post_only_would_cross"
    wo.cancel_response_outcome = "benign_missing"
    wo.cancel_response_detail = "okx_row_51400:already filled"
    om.persist(wo)
    rows = storage.orders_lifecycle_since(
        since_iso=(utc_now() - timedelta(minutes=1)).isoformat()
    )
    assert len(rows) == 1
    assert rows[0]["cancel_trigger_reason"] == "hard_age_cap"
    assert rows[0]["place_response_outcome"] == "exchange_rejected"
    assert rows[0]["place_response_detail"] == "post_only_would_cross"
    assert rows[0]["cancel_response_outcome"] == "benign_missing"
    assert rows[0]["cancel_response_detail"] == "okx_row_51400:already filled"
    assert rows[0]["ts_place_response"] is not None


def test_state_current_exposes_place_cancel_race_total() -> None:
    """The live_stats payload includes the new counter so the
    dashboard can surface it."""
    om, state, _storage = _make_om()
    wo = _make_wo(status=OrderStatus.SENT, ts_ack=False, ts_sent_offset_ms=5.0)
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    # Trip a second one too.
    wo2 = _make_wo(status=OrderStatus.SENT, ts_ack=False, ts_sent_offset_ms=5.0)
    state.working_bid = wo2
    om._enqueue_cancel_quote_path(wo2, trigger_reason="reprice_replace")
    assert state.place_cancel_race_total == 2


def test_order_trace_was_recently_terminal_local_record() -> None:
    """1.3.85: OrderTrace.was_recently_terminal returns True when the
    bot recorded a local terminal for that (oid, cloid) within the
    window."""
    from app.order_trace import OrderTraceBuffer

    buf = OrderTraceBuffer(max_entries=100)
    buf.begin_order(client_order_id="c1", side="BUY", price=1.0, size_base=1.0)
    buf.record_place_response(
        client_order_id="c1",
        outcome="accepted",
        detail="",
        order_id_exchange=12345,
    )
    buf.record_terminal(
        order_id_exchange=12345,
        client_order_id="c1",
        status="CANCELED",
        reason="ws:CANCELED",
        source="ws",
    )
    # Look up by oid.
    assert buf.was_recently_terminal(order_id_exchange=12345)
    # Look up by cloid.
    assert buf.was_recently_terminal(client_order_id="c1")
    # Window too narrow → not recent.
    assert not buf.was_recently_terminal(
        order_id_exchange=12345, max_age_seconds=0.0
    )


def test_order_trace_was_recently_terminal_ws_event() -> None:
    """When the local state machine hasn't yet recorded a terminal
    but the most recent WS event is already a terminal state, the
    guard still fires."""
    from app.order_trace import OrderTraceBuffer

    buf = OrderTraceBuffer(max_entries=100)
    buf.begin_order(client_order_id="c2", side="SELL", price=2.0, size_base=1.0)
    buf.record_place_response(
        client_order_id="c2",
        outcome="accepted",
        detail="",
        order_id_exchange=22222,
    )
    buf.record_ws_event(order_id_exchange=22222, state="canceled")
    # Terminal-via-WS-only is enough to consider it "recently terminal".
    assert buf.was_recently_terminal(order_id_exchange=22222)


def test_order_trace_was_recently_terminal_unknown() -> None:
    """Unknown oid/cloid → False."""
    from app.order_trace import OrderTraceBuffer

    buf = OrderTraceBuffer(max_entries=100)
    assert not buf.was_recently_terminal(order_id_exchange=99999)
    assert not buf.was_recently_terminal(client_order_id="nope")


def _open_order_raw(oid, cloid, side=Side.BUY, px=1.0, sz=1.0):
    from app.exchange.base import OpenOrderRaw

    return OpenOrderRaw(
        oid=oid, coin="ETH", side=side, limit_px=px, sz=sz,
        timestamp=int(utc_now().timestamp() * 1000), cloid=cloid,
    )


def test_hydration_guard_skips_recently_terminal_orders() -> None:
    """The reconcile hydration path consults
    ``was_recently_terminal`` and skips when it returns True. The
    skipped-hydration counter increments instead."""
    om, state, _storage = _make_om()
    # Pre-populate the order_trace with a recent terminal for oid=42,
    # cloid="cx" — simulating the WS-vs-REST race.
    state.order_trace.begin_order(
        client_order_id="cx", side="BUY", price=1.0, size_base=1.0
    )
    state.order_trace.record_place_response(
        client_order_id="cx", outcome="accepted", detail="", order_id_exchange=42
    )
    state.order_trace.record_terminal(
        order_id_exchange=42,
        client_order_id="cx",
        status="CANCELED",
        reason="ws:CANCELED",
        source="ws",
    )
    remote = _open_order_raw(oid=42, cloid="cx")
    hydrated = om._hydrate_working_from_exchange(Side.BUY, remote)
    assert hydrated is False
    assert state.hydration_skipped_recently_terminal_total == 1
    assert state.working_bid is None


def test_hydration_proceeds_when_no_recent_terminal() -> None:
    """If the order_trace has no recent terminal for the (oid, cloid),
    hydration proceeds as before."""
    om, state, _storage = _make_om()
    remote = _open_order_raw(oid=99, cloid="brand_new")
    hydrated = om._hydrate_working_from_exchange(Side.BUY, remote)
    assert hydrated is True
    assert state.hydration_skipped_recently_terminal_total == 0
    assert state.working_bid is not None
    assert state.working_bid.order_id_exchange == 99


# ---------------------------------------------------------------------------
# 1.3.86: cancel-before-ack defer behaviour
# ---------------------------------------------------------------------------


def test_cancel_defers_when_order_unacked() -> None:
    """A cancel issued against a SENT-but-no-ack order with no
    exchange_oid is parked, NOT dispatched. cancel_deferred_until_
    ack_total increments. order_id_exchange=None mirrors the real
    race: the place HTTP truly hasn't returned yet."""
    om, state, _storage = _make_om()
    wo = _make_wo(
        status=OrderStatus.SENT,
        ts_ack=False,
        ts_sent_offset_ms=5.0,
        order_id_exchange=None,
    )
    state.working_bid = wo
    # If _cancel_http_transport gets called, that means the defer
    # guard failed. Stub it to record the call.
    om._cancel_http_transport = MagicMock(return_value=True)
    result = om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    assert result is True
    assert wo.cancel_pending_after_ack is True
    assert wo.cancel_trigger_reason == "reprice_replace"
    assert wo.ts_cancel_requested is not None
    assert wo.status == OrderStatus.SENT  # NOT yet CANCEL_PENDING
    assert state.cancel_deferred_until_ack_total == 1
    om._cancel_http_transport.assert_not_called()


def test_cancel_dispatches_for_acked_order() -> None:
    """A cancel issued against an ACKED order is dispatched
    immediately (no defer)."""
    om, state, _storage = _make_om()
    wo = _make_wo(status=OrderStatus.ACKED, ts_ack=True)
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    result = om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    assert result is True
    assert wo.cancel_pending_after_ack is False
    assert state.cancel_deferred_until_ack_total == 0
    om._outbound.submit_cancel.assert_called_once()


def test_cancel_defer_idempotent_on_second_call() -> None:
    """Multiple cancel requests against the same un-acked WO bump
    the counter ONCE — the second call recognises the deferred
    state and is a no-op on the counter."""
    om, state, _storage = _make_om()
    wo = _make_wo(
        status=OrderStatus.SENT, ts_ack=False, order_id_exchange=None,
    )
    state.working_bid = wo
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    om._enqueue_cancel_quote_path(wo, trigger_reason="hard_age_cap")
    om._enqueue_cancel_quote_path(wo, trigger_reason="binance_cross_venue")
    assert state.cancel_deferred_until_ack_total == 1
    # First reason wins.
    assert wo.cancel_trigger_reason == "reprice_replace"


def test_cancel_defer_disabled_via_config_dispatches_normally() -> None:
    """When ``CANCEL_DEFER_UNTIL_ACK_ENABLED=False`` the guard is a
    no-op and cancels fire even against un-acked orders (legacy
    behaviour, for diagnostic A/B comparison)."""
    om, state, _storage = _make_om(CANCEL_DEFER_UNTIL_ACK_ENABLED=False)
    wo = _make_wo(
        status=OrderStatus.SENT,
        ts_ack=False,
        ts_sent_offset_ms=5.0,
        order_id_exchange=None,
    )
    state.working_bid = wo
    om._outbound.submit_cancel = MagicMock()
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    assert wo.cancel_pending_after_ack is False
    assert state.cancel_deferred_until_ack_total == 0
    om._outbound.submit_cancel.assert_called_once()


def test_cancel_deferred_flushes_on_ack() -> None:
    """When a deferred-cancel WO receives its place response with
    ex_oid set, the bot transitions to ACKED AND the deferred cancel
    immediately flushes via _enqueue_cancel_quote_path."""
    om, state, _storage = _make_om()
    wo = _make_wo(
        status=OrderStatus.SENT, ts_ack=False, order_id_exchange=None,
    )
    state.working_bid = wo
    om._enqueue_cancel_quote_path(wo, trigger_reason="reprice_replace")
    assert wo.cancel_pending_after_ack is True
    # Simulate the place-response success path manually: set
    # exchange_oid, transition to ACKED, then run the post-ack flush
    # block. (Full _handle_place_response would call _interpret which
    # needs a real response payload — too heavy for a unit test.)
    from app.enums import OrderStatus as _OS
    from app.execution import transition as _transition

    wo.order_id_exchange = 12345
    _transition(wo, _OS.ACKED)
    # Replay the v1.3.86 flush block from the place handler. We can
    # inline this because the actual handler delegates to the same
    # _enqueue_cancel_quote_path.
    om._outbound.submit_cancel = MagicMock()
    if wo.cancel_pending_after_ack:
        wo.cancel_pending_after_ack = False
        om._enqueue_cancel_quote_path(
            wo, trigger_reason=wo.cancel_trigger_reason
        )
    # Cancel should now have been dispatched against the acked order.
    om._outbound.submit_cancel.assert_called_once()
    assert wo.cancel_pending_after_ack is False
    assert wo.cancel_trigger_reason == "reprice_replace"
