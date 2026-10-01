"""v1.4.96 — three-tier connectivity classification tests.

Tests for ``OrderManager._bump_gone_on_exchange_counters`` and the
companion late-WS detection in ``_handle_private_order_update_v2``.

Three tiers (independent counters, independent severities):

* TIER 1 (FATAL) — ``gone_on_exchange_total``. Bot has no HTTP
  confirmation. Sub-signatures:
    - ``phantom_no_ack`` — ts_ack is None
    - ``acked_no_cancel`` — ts_ack set, ts_cancel_requested None
    - ``cancel_no_http_confirm`` — cancel sent, no HTTP success
    - ``other`` — none of the above

* TIER 2 (WARN) — ``http_acked_no_ws_total``. Cancel HTTP success
  but WS terminal didn't arrive. Tolerable when rare.

* TIER 3 (WARN) — ``ws_arrived_late_total``. WS terminal arrived
  AFTER local cleanup. Bumped from the unmatched-WS handler.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[Path, BotState, OrderManager]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_goe_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state)
    return path, state, om


def _wo(
    *,
    ts_ack: datetime | None,
    ts_cancel_requested: datetime | None = None,
    ts_cancel_acked: datetime | None = None,
    ts_closed: datetime | None = None,
    venue_cancel_utime_ms: int | None = None,
    cancel_response_outcome: str | None = None,
    side: Side = Side.SELL,
    order_id_exchange: str = "ex123",
) -> WorkingOrder:
    """Build a minimal WO with the timestamps required for classification."""
    wo = WorkingOrder(
        order_id_local=uuid.uuid4(),
        order_id_exchange=order_id_exchange,
        client_order_id="cl123",
        symbol="TON-USDT-SWAP",
        side=side,
        price=2.0,
        size=3.0,
        post_only=True,
        status=OrderStatus.CANCELED,
    )
    wo.ts_ack = ts_ack
    wo.ts_cancel_requested = ts_cancel_requested
    wo.ts_cancel_acked = ts_cancel_acked
    wo.ts_closed = ts_closed
    wo.venue_cancel_utime_ms = venue_cancel_utime_ms
    wo.cancel_response_outcome = cancel_response_outcome
    return wo


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_counters_start_at_zero() -> None:
    _path, state, _om = _setup()
    try:
        # TIER 1
        assert state.gone_on_exchange_total == 0
        assert state.gone_on_exchange_phantom_no_ack_total == 0
        assert state.gone_on_exchange_acked_no_cancel_total == 0
        assert state.gone_on_exchange_cancel_no_http_confirm_total == 0
        assert state.gone_on_exchange_other_total == 0
        assert len(state.gone_on_exchange_recent) == 0
        # TIER 2
        assert state.http_acked_no_ws_total == 0
        assert len(state.http_acked_no_ws_recent) == 0
        assert state.http_acked_no_ws_lateness_ms_min is None
        assert state.http_acked_no_ws_lateness_ms_max is None
        # TIER 3
        assert state.ws_arrived_late_total == 0
        assert len(state.ws_arrived_late_recent) == 0
        # Late-arrival detection map
        assert len(state.terminated_oids_recent) == 0
    finally:
        _path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TIER 1 (FATAL) — gone_on_exchange signature classification
# ---------------------------------------------------------------------------


def test_phantom_no_ack_signature_counts_as_red_gone() -> None:
    """ts_ack None → BUG-024 phantom, counted under gone_on_exchange_total."""
    path, state, om = _setup()
    try:
        wo = _wo(ts_ack=None)
        om._bump_gone_on_exchange_counters(wo)
        # TIER 1 counters bumped.
        assert state.gone_on_exchange_total == 1
        assert state.gone_on_exchange_phantom_no_ack_total == 1
        # Other tiers unaffected.
        assert state.http_acked_no_ws_total == 0
        assert state.ws_arrived_late_total == 0
        entry = state.gone_on_exchange_recent[0]
        assert entry["signature"] == "phantom_no_ack"
        assert entry["tier"] == "red"
    finally:
        path.unlink(missing_ok=True)


def test_acked_no_cancel_signature_counts_as_red_gone() -> None:
    """ts_ack set, ts_cancel_requested None → BUG-023, RED."""
    path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        wo = _wo(ts_ack=now, ts_cancel_requested=None)
        om._bump_gone_on_exchange_counters(wo)
        assert state.gone_on_exchange_total == 1
        assert state.gone_on_exchange_acked_no_cancel_total == 1
        assert state.http_acked_no_ws_total == 0
        entry = state.gone_on_exchange_recent[0]
        assert entry["signature"] == "acked_no_cancel"
        assert entry["tier"] == "red"
    finally:
        path.unlink(missing_ok=True)


def test_cancel_no_http_confirm_signature_counts_as_red_gone() -> None:
    """v1.4.96 new RED subcategory: cancel was sent but the venue's
    HTTP response was NOT a clean success (e.g. transport_rejected
    / timeout / no_response). The bot doesn't know whether the
    cancel landed."""
    path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        wo = _wo(
            ts_ack=now - timedelta(seconds=2),
            ts_cancel_requested=now - timedelta(seconds=1),
            ts_cancel_acked=None,  # no HTTP success ack
            ts_closed=now,
            cancel_response_outcome=None,
            venue_cancel_utime_ms=None,
        )
        om._bump_gone_on_exchange_counters(wo)
        assert state.gone_on_exchange_total == 1
        assert state.gone_on_exchange_cancel_no_http_confirm_total == 1
        assert state.http_acked_no_ws_total == 0
        entry = state.gone_on_exchange_recent[0]
        assert entry["signature"] == "cancel_no_http_confirm"
        assert entry["tier"] == "red"
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TIER 2 (WARN) — http_acked_no_ws (formerly the misclassified
# "ws_terminal_late" of v1.4.95). The 6 events in snapshot
# v1.4.92-260519-161411 all match this — they should NOT increment
# gone_on_exchange_total.
# ---------------------------------------------------------------------------


def test_http_acked_no_ws_does_not_count_as_red_gone() -> None:
    """Cancel HTTP returned success — bot knows venue canceled the
    order. Just the WS terminal didn't arrive in time. WARN-tier,
    NOT counted under gone_on_exchange_total."""
    path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        wo = _wo(
            ts_ack=now - timedelta(seconds=10),
            ts_cancel_requested=now - timedelta(seconds=8),
            ts_cancel_acked=now - timedelta(seconds=8)
            + timedelta(milliseconds=3),
            ts_closed=now,  # ~8 sec after cancel ack — extreme lateness
            cancel_response_outcome="success",  # HTTP confirmed
            venue_cancel_utime_ms=None,  # WS never arrived
        )
        om._bump_gone_on_exchange_counters(wo)
        # CRITICAL ASSERTION: NOT a gone_on_exchange event.
        assert state.gone_on_exchange_total == 0
        # TIER 2 counter bumped.
        assert state.http_acked_no_ws_total == 1
        # TIER 2 ring populated, TIER 1 ring empty.
        assert len(state.http_acked_no_ws_recent) == 1
        assert len(state.gone_on_exchange_recent) == 0
        # Lateness histogram updated.
        assert state.http_acked_no_ws_lateness_ms_min is not None
        assert 7990.0 <= state.http_acked_no_ws_lateness_ms_min <= 8010.0
        entry = state.http_acked_no_ws_recent[0]
        assert entry["signature"] == "http_acked_no_ws"
        assert entry["tier"] == "warn"
        assert entry["cancel_response_outcome"] == "success"
    finally:
        path.unlink(missing_ok=True)


def test_http_acked_no_ws_lateness_histogram_tracks_min_max() -> None:
    """Min/max persist across samples for the tier-2 histogram."""
    path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        for ms in (50, 5000, 200):
            wo = _wo(
                ts_ack=now - timedelta(seconds=10),
                ts_cancel_requested=now - timedelta(seconds=8),
                ts_cancel_acked=now - timedelta(milliseconds=ms),
                ts_closed=now,
                cancel_response_outcome="success",
                venue_cancel_utime_ms=None,
                order_id_exchange=f"ex-late-{ms}",
            )
            om._bump_gone_on_exchange_counters(wo)
        assert state.http_acked_no_ws_total == 3
        assert state.gone_on_exchange_total == 0  # NEVER red
        assert state.http_acked_no_ws_lateness_ms_min is not None
        assert state.http_acked_no_ws_lateness_ms_max is not None
        assert 40.0 <= state.http_acked_no_ws_lateness_ms_min <= 60.0
        assert 4990.0 <= state.http_acked_no_ws_lateness_ms_max <= 5010.0
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# TIER 3 (WARN) — ws_arrived_late. The unmatched-WS handler detects
# when a WS event arrives for an oid the bot previously cleaned up.
# ---------------------------------------------------------------------------


def test_late_ws_arrival_detected_when_oid_was_recently_terminated() -> None:
    """After a tier-1 or tier-2 terminal, the oid lives in
    ``terminated_oids_recent`` for late-arrival detection."""
    path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        # First: declare an oid gone via the tier-2 (http_acked_no_ws) path.
        wo = _wo(
            ts_ack=now - timedelta(seconds=5),
            ts_cancel_requested=now - timedelta(seconds=4),
            ts_cancel_acked=now - timedelta(seconds=4),
            ts_closed=now,
            cancel_response_outcome="success",
            venue_cancel_utime_ms=None,
            order_id_exchange="ex-late-1",
        )
        om._bump_gone_on_exchange_counters(wo)
        assert state.http_acked_no_ws_total == 1
        # The oid is registered for late-arrival detection.
        assert "ex-late-1" in state.terminated_oids_recent

        # Now: the WS terminal finally arrives — should bump tier 3.
        matched = om._check_ws_arrived_late_for_terminated_oid(
            "ex-late-1", "canceled"
        )
        assert matched is True
        assert state.ws_arrived_late_total == 1
        assert len(state.ws_arrived_late_recent) == 1
        late_entry = state.ws_arrived_late_recent[0]
        assert late_entry["order_id_exchange"] == "ex-late-1"
        assert late_entry["original_signature"] == "http_acked_no_ws"
        assert late_entry["original_tier"] == "warn"
        # And the oid is removed so a second WS event wouldn't double-count.
        assert "ex-late-1" not in state.terminated_oids_recent
    finally:
        path.unlink(missing_ok=True)


def test_late_ws_arrival_for_unknown_oid_does_not_bump() -> None:
    """If the oid was never recently terminated, the unmatched WS
    handler doesn't bump ws_arrived_late_total."""
    path, state, om = _setup()
    try:
        matched = om._check_ws_arrived_late_for_terminated_oid(
            "ex-never-seen", "canceled"
        )
        assert matched is False
        assert state.ws_arrived_late_total == 0
    finally:
        path.unlink(missing_ok=True)


def test_late_arrival_after_red_gone_also_detected() -> None:
    """The detector works for TIER 1 (RED) terminals too, not just
    TIER 2. After a phantom_no_ack, if the WS terminal eventually
    arrives, we count it."""
    path, state, om = _setup()
    try:
        wo = _wo(ts_ack=None, order_id_exchange="ex-red-1")
        om._bump_gone_on_exchange_counters(wo)
        assert state.gone_on_exchange_total == 1
        assert "ex-red-1" in state.terminated_oids_recent

        matched = om._check_ws_arrived_late_for_terminated_oid(
            "ex-red-1", "canceled"
        )
        assert matched is True
        assert state.ws_arrived_late_total == 1
        late_entry = state.ws_arrived_late_recent[0]
        assert late_entry["original_signature"] == "phantom_no_ack"
        assert late_entry["original_tier"] == "red"
    finally:
        path.unlink(missing_ok=True)


def test_terminated_oids_recent_is_bounded() -> None:
    """FIFO eviction at terminated_oids_recent_max prevents unbounded
    memory growth on long sessions."""
    path, state, om = _setup()
    try:
        state.terminated_oids_recent_max = 5  # shrink for the test
        for i in range(10):
            wo = _wo(ts_ack=None, order_id_exchange=f"ex-{i}")
            om._bump_gone_on_exchange_counters(wo)
        # Only the last 5 oids survive.
        assert len(state.terminated_oids_recent) == 5
        for i in range(5):
            assert f"ex-{i}" not in state.terminated_oids_recent
        for i in range(5, 10):
            assert f"ex-{i}" in state.terminated_oids_recent
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Ring buffer behavior
# ---------------------------------------------------------------------------


def test_ring_buffer_caps_at_100_entries() -> None:
    """Tier-1 ring is FIFO-bounded at 100."""
    path, state, om = _setup()
    try:
        for i in range(105):
            wo = _wo(ts_ack=None, order_id_exchange=f"ex-ring-{i}")
            om._bump_gone_on_exchange_counters(wo)
        assert state.gone_on_exchange_total == 105
        assert len(state.gone_on_exchange_recent) == 100
        assert all(
            e["signature"] == "phantom_no_ack"
            for e in state.gone_on_exchange_recent
        )
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_bump_never_raises_on_malformed_wo() -> None:
    """A best-effort diagnostic must NEVER break the trading loop."""
    path, state, om = _setup()
    try:
        # Pass an object with the wrong shape — should swallow.
        class Bogus:
            pass

        om._bump_gone_on_exchange_counters(Bogus())  # type: ignore[arg-type]
        # Counter is still consistent (didn't half-bump).
        assert state.gone_on_exchange_total >= 0
    finally:
        path.unlink(missing_ok=True)
