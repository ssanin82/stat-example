"""Three log-flood fixes for INFO/WARNING lines that previously re-fired every tick.

Motivation: post-Binance 4-minute run produced 25,000 log lines
(15.4 MB, ~230 MB/hour). Top three offenders were **88.7%** of the
total volume:

  - ``same_side_place_suppressed``       — 10,390 lines (42.7%)
  - ``reconcile_requested``               —  9,773 lines (33.2%)
  - ``cancel_pending_timeout_recovery``   —  3,167 lines (12.7%)

All three were "log the intent, not the outcome" antipatterns: the
underlying ACTION was correctly debounced (counter rate-limits,
cooldown gates, retry intervals), but the LOG line fired on every
evaluation. These tests pin the fix without changing any counter
semantics — telemetry / gauge values remain identical.

Invariants pinned here:

  1. ``same_side_place_suppressed`` logs exactly ONCE per
     side-unresolved episode (state transition). Counter still
     increments on every suppression so gauge values are unchanged.
  2. Clearing the side then re-entering unresolved re-enables the
     transition log for the new episode.
  3. ``reconcile_requested`` no longer emits at INFO — rejected
     requests are silent counter bumps; accepted requests still
     produce ``reconcile_started`` / ``reconcile_completed`` INFO
     lines (not pinned here — pinned by existing tests).
  4. ``cancel_pending_timeout_recovery`` logs once per stuck
     ``order_id_exchange``. Subsequent timeout evaluations for the
     SAME oid are silent. A different oid (new cancel episode)
     re-enables logging.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"mm_lfr_{os.getpid()}_{uuid.uuid4().hex}.db"


def _setup(**overrides) -> tuple[UnitTestSettings, OrderManager]:
    path = _db_path()
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "PRIVATE_WS_ENABLED": False,
        # Tests use size=0.1 with ETH-priced symbol; new position-
        # aware central gate (Codex MED #1, 2026-05-06) refuses
        # orders past max_abs_position, so raise it to ensure the
        # log-flood / suppression path being tested can actually run.
        "MAX_ABS_POSITION": 100.0,
        "MAX_POSITION_NOTIONAL_USD": 100_000.0,
    }
    for k, v in overrides.items():
        data[k.upper() if k.islower() else k] = v
    s = UnitTestSettings.model_validate(data)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state)
    # Clean slots so the unresolved logic paths are straightforward.
    state.working_bid = None
    state.working_ask = None
    return s, om


def _count_records(caplog, message_substring: str) -> int:
    return sum(1 for r in caplog.records if message_substring in r.getMessage())


# ==================== Fix 1a: same_side_place_suppressed ====================


def test_same_side_suppressed_logs_once_per_episode(caplog) -> None:
    """Enter unresolved, call the suppress-log site 50 times; expect
    exactly 1 log line (transition) and 50 counter increments."""
    _s, om = _setup()
    caplog.set_level(logging.INFO, logger="app.execution")

    om._set_side_unresolved(Side.BUY, reason="cancel_pending_wait")

    # Call the place-stage path 50 times; every call hits the
    # "is_side_unresolved" guard and the log should fire only once.
    for _ in range(50):
        result = om._stage_place_order_local(
            Side.BUY, price=100.0, size=0.1, quote_cycle_id="qc"
        )
        assert result is None  # suppressed

    suppress_logs = _count_records(caplog, "same_side_place_suppressed")
    assert suppress_logs == 1, (
        f"expected exactly one suppress log per episode, got {suppress_logs} "
        f"(regression: the per-tick log flood is back)"
    )
    # Counter semantics unchanged — still increments every call.
    assert om._suppress_place_due_unresolved_count == 50


def test_same_side_suppressed_relogs_after_clear_and_reenter(caplog) -> None:
    """Clear the side (resolving the episode), enter unresolved again
    → the next suppression MUST log once more (new episode, new
    transition). Without this, a long-running bot that goes through
    multiple unresolved episodes would lose visibility into each."""
    _s, om = _setup()
    caplog.set_level(logging.INFO, logger="app.execution")

    om._set_side_unresolved(Side.BUY, reason="cancel_pending_wait")
    om._stage_place_order_local(Side.BUY, price=100.0, size=0.1, quote_cycle_id="a")
    om._stage_place_order_local(Side.BUY, price=100.0, size=0.1, quote_cycle_id="a")
    assert _count_records(caplog, "same_side_place_suppressed") == 1

    # New episode.
    om._clear_side_unresolved(Side.BUY, reason="reconcile_confirmed_gone")
    om._set_side_unresolved(Side.BUY, reason="orphan_confirmation_pending")

    om._stage_place_order_local(Side.BUY, price=100.0, size=0.1, quote_cycle_id="b")
    # Now we should have 2 suppress logs total (one per episode).
    assert _count_records(caplog, "same_side_place_suppressed") == 2


def test_same_side_suppressed_independent_per_side(caplog) -> None:
    """BUY and SELL have independent transition flags — each logs once
    per its own episode even if both sides are simultaneously stuck."""
    _s, om = _setup()
    caplog.set_level(logging.INFO, logger="app.execution")

    om._set_side_unresolved(Side.BUY, reason="cancel_pending_wait")
    om._set_side_unresolved(Side.SELL, reason="cancel_pending_wait")

    om._stage_place_order_local(Side.BUY, price=100.0, size=0.1, quote_cycle_id="q")
    om._stage_place_order_local(Side.SELL, price=100.0, size=0.1, quote_cycle_id="q")
    om._stage_place_order_local(Side.BUY, price=100.0, size=0.1, quote_cycle_id="q")
    om._stage_place_order_local(Side.SELL, price=100.0, size=0.1, quote_cycle_id="q")

    # Two log lines total — one per side (each a transition), not four.
    assert _count_records(caplog, "same_side_place_suppressed") == 2


# ==================== Fix 1b: reconcile_requested ====================


def test_reconcile_requested_is_not_logged_at_info(caplog) -> None:
    """request_open_orders_reconcile must NOT emit at INFO anymore —
    the accepted path already logs ``reconcile_started`` with the
    same reason. Pre-fix: 9,773 INFO-level reconcile_requested lines
    on a 4-minute run for only 37 actual reconciles started (99.6%
    redundant)."""
    _s, om = _setup()
    caplog.set_level(logging.DEBUG, logger="app.execution")

    # Many calls with the reasons we saw flooding the log in production.
    for _ in range(20):
        om.request_open_orders_reconcile(
            reason="side_unresolved_or_desync", force=True, emergency=True
        )
    for _ in range(20):
        om.request_open_orders_reconcile(
            reason="cancel_pending_timeout_buy", force=True, emergency=True
        )

    # Zero INFO-level records carrying "reconcile_requested".
    info_reqs = [
        r for r in caplog.records
        if r.levelno >= logging.INFO and "reconcile_requested" in r.getMessage()
    ]
    assert info_reqs == [], (
        f"expected zero INFO-level reconcile_requested logs, got {len(info_reqs)}"
    )
    # DEBUG-level entries are still present for debugging; not asserted on
    # (too fragile across logger configs). The telemetry counter is what
    # callers should read.
    assert om._reconcile_requests_total == 40


def test_reconcile_requested_counter_still_increments(caplog) -> None:
    """Dropping the INFO log must not break counter bookkeeping —
    ``/state/current`` and monitoring dashboards read
    ``reconcile_requests_total`` from exec_runtime_counters."""
    _s, om = _setup()
    caplog.set_level(logging.INFO, logger="app.execution")

    for _ in range(7):
        om.request_open_orders_reconcile(
            reason="side_unresolved_or_desync", force=True, emergency=True
        )
    assert om._reconcile_requests_total == 7


# ==================== Fix 1c: cancel_pending_timeout_recovery ====================


def _make_working_order(side: Side, oid: int, cloid: str = "c") -> WorkingOrder:
    return WorkingOrder(
        order_id_local=f"wo_{uuid.uuid4().hex[:8]}",
        order_id_exchange=oid,
        client_order_id=cloid,
        symbol="ETH",
        side=side,
        price=100.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.CANCEL_PENDING,
    )


def test_cancel_pending_timeout_logs_once_per_oid(caplog) -> None:
    """After a 30 s-stuck CANCEL_PENDING, the timeout recovery WARNING
    fires on first detection. Subsequent tick evaluations for the
    SAME order must NOT re-fire the log (the retry cadence in the
    same function is already rate-limited). Pre-fix: 1,020 duplicate
    log lines observed for a single stuck oid."""
    s, om = _setup(CANCEL_PENDING_UNRESOLVED_TIMEOUT_SECONDS=0.01)
    caplog.set_level(logging.WARNING, logger="app.execution")
    om._enqueue_cancel_quote_path = MagicMock(return_value=True)  # stub
    # Stub the reconcile request path too; we only care about the log.
    om.request_open_orders_reconcile = MagicMock()

    wo = _make_working_order(Side.BUY, oid=111_111)
    # Force "already past timeout": rewrite cancel-pending-since.
    om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 10.0

    # Call the timeout handler 5 times; log should fire on the FIRST
    # only (since the oid doesn't change).
    for _ in range(5):
        om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)

    logs = _count_records(caplog, "cancel_pending_timeout_recovery")
    assert logs == 1, (
        f"expected exactly one timeout log per stuck oid, got {logs}"
    )


def test_cancel_pending_timeout_relogs_when_oid_changes(caplog) -> None:
    """A second stuck cancel on the same side (new oid, e.g. after
    placement → reprice → new cancel) is a distinct episode and
    deserves its own log line. Without this the operator would never
    see a second stuck order until the first clears."""
    s, om = _setup(CANCEL_PENDING_UNRESOLVED_TIMEOUT_SECONDS=0.01)
    caplog.set_level(logging.WARNING, logger="app.execution")
    om._enqueue_cancel_quote_path = MagicMock(return_value=True)
    om.request_open_orders_reconcile = MagicMock()

    wo_a = _make_working_order(Side.BUY, oid=111_111)
    wo_b = _make_working_order(Side.BUY, oid=222_222)
    om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 10.0

    om._maybe_handle_cancel_pending_timeout(Side.BUY, wo_a)
    om._maybe_handle_cancel_pending_timeout(Side.BUY, wo_a)
    om._maybe_handle_cancel_pending_timeout(Side.BUY, wo_b)  # new oid
    om._maybe_handle_cancel_pending_timeout(Side.BUY, wo_b)

    logs = _count_records(caplog, "cancel_pending_timeout_recovery")
    assert logs == 2, (
        f"expected 2 logs (one per distinct stuck oid), got {logs}"
    )


def test_cancel_pending_timeout_dedup_resets_on_side_clear(caplog) -> None:
    """Once the side is fully resolved (reconcile confirmed gone),
    the per-oid dedup must reset — a subsequent stuck cancel (even if
    it happens to land on the same oid) is a fresh episode."""
    s, om = _setup(CANCEL_PENDING_UNRESOLVED_TIMEOUT_SECONDS=0.01)
    caplog.set_level(logging.WARNING, logger="app.execution")
    om._enqueue_cancel_quote_path = MagicMock(return_value=True)
    om.request_open_orders_reconcile = MagicMock()

    wo = _make_working_order(Side.BUY, oid=333_333)
    om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 10.0

    om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
    om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)  # silent
    assert _count_records(caplog, "cancel_pending_timeout_recovery") == 1

    # Resolve the side (simulating the reconcile confirming "gone").
    om._clear_side_unresolved(Side.BUY, reason="reconcile_confirmed_gone")

    # Engineered pathology: the SAME oid shows up again as stuck.
    # In practice it'd be a new oid, but the invariant is about
    # the dedup state cleanly resetting with the side.
    wo2 = _make_working_order(Side.BUY, oid=333_333)
    om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 10.0
    om._maybe_handle_cancel_pending_timeout(Side.BUY, wo2)

    assert _count_records(caplog, "cancel_pending_timeout_recovery") == 2


def test_cancel_pending_timeout_count_counter_still_increments(caplog) -> None:
    """Dedup guards the LOG only. The ``_cancel_pending_timeout_count``
    counter — surfaced in exec_runtime_counters and used by the
    operator to spot chronically-stuck sides — must still increment
    on every timeout detection."""
    s, om = _setup(CANCEL_PENDING_UNRESOLVED_TIMEOUT_SECONDS=0.01)
    caplog.set_level(logging.WARNING, logger="app.execution")
    om._enqueue_cancel_quote_path = MagicMock(return_value=True)
    om.request_open_orders_reconcile = MagicMock()

    wo = _make_working_order(Side.BUY, oid=444_444)
    om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 10.0

    for _ in range(6):
        om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)

    # Log deduped (expect 1), but counter incremented each call.
    assert _count_records(caplog, "cancel_pending_timeout_recovery") == 1
    assert om._cancel_pending_timeout_count == 6
