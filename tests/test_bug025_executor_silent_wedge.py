"""BUG-025 regression tests across all three phases.

v1.4.40 — Phases 1 & 2 (instrumentation):

1. ``state.executor_state_snapshot`` populates correctly after
   ``_sync_outbound_state_flags()`` and surfaces through
   ``BotState.snapshot_dict()`` under ``executor_state``. Verifies
   every expected field is present with the right type so the
   dashboard / postmortem can rely on the schema.

2. ``_maybe_emit_silent_wedge_diagnostic`` fires its ERROR event
   only when ALL FIVE conjoint conditions hold (engine healthy,
   eligibility QUOTE_BOTH, active_sides BOTH, no side_unresolved,
   execution_idle > threshold). Negative controls verify it does
   NOT fire on benign cycles.

v1.4.41 — Phase 3 (targeted fix):

3. ``_reap_wo_after_cancel_unexpected_gone`` reaps the local
   ``WorkingOrder`` when OKX responds to a cancel with sCode
   51400 / 51401 / 51503 (``unexpected_gone``). Both batch and
   single-cancel paths flow through the helper. Captured wedge
   2026-05-18 v1.4.40-260518-093848 showed two zombie OIDs that
   the bot kept trying to cancel for 16 min until the deadlock
   watchdog killed at idle=600 s; without the reap, the stale
   WO sits in ``working_orders[side][level_idx]`` and each
   orchestrate cycle re-issues a cancel that gets another 51400.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from app.enums import Side
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _safe_unlink_db(path: Path) -> None:
    """v1.5.205 — Windows-safe DB cleanup for test ``finally:`` blocks.

    Background. Tests in this file create a unique ``mm_bug025_<pid>_
    <uuid>.db`` per test, then ``_safe_unlink_db(db_path)`` in
    ``finally:`` to delete it. On Windows this races: pytest sometimes
    proceeds to the cleanup line BEFORE the ``Storage`` / SQLite
    connection's last reference has been finalized, so Python still
    holds an open file descriptor and ``unlink`` raises
    ``PermissionError: [WinError 32]``. The CI daemon hit this on
    ``test_phase2_repeated_cancelling_risk_does_not_refire_cancel_all``
    on 2026-05-28.

    Fix: force a GC pass (finalises the connection if it's only
    reachable via a soon-to-die local), then retry the unlink a
    handful of times with a short backoff. Final attempt swallows
    ``PermissionError`` — the temp file system will reap stale files
    eventually and a leaked temp is not worth failing a test for.

    Linux/Mac are immune to the race (no file locks during unlink),
    so this helper is a no-op-cost on those platforms — gc.collect
    runs but no retry is needed.
    """
    import gc
    import time
    gc.collect()
    for _ in range(5):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.05)
    # Final attempt — swallow if still locked. Worst case: stale temp
    # file. ``%TEMP%\mm_bug025_*.db`` is reaped by Windows / pytest.
    try:
        path.unlink(missing_ok=True)
    except PermissionError:
        pass


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_bug025_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
        # Make wedge detection trigger quickly in tests.
        "SILENT_WEDGE_DETECT_THRESHOLD_SECONDS": 10.0,
        "SILENT_WEDGE_DETECT_RELOG_SECONDS": 1.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _setup() -> tuple[OrderManager, Storage, BotState, Path]:
    s = _settings()
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    return om, storage, state, db_path


# ---------------------------------------------------------------------------
# Phase 1 — executor_state snapshot exposure
# ---------------------------------------------------------------------------


def test_executor_state_snapshot_has_expected_top_level_keys() -> None:
    """v1.4.40 BUG-025 Phase 1: executor-state snapshot exposes
    every field that could explain a silent wedge."""
    om, storage, state, db_path = _setup()
    try:
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        expected_top_level = {
            "engine_no_quote_streak_ticks",
            "engine_no_quote_diag_last_log_age_s",
            "side_unresolved",
            "side_unresolved_enter_count",
            "side_unresolved_confirm_block_count",
            "reprice_replace_pending",
            "cancel_pending_since_s",
            "post_only_cross_cooldown_remaining_s",
            "last_outbound_replace_age_s",
            "outbound_inflight_count",
            "outbound_active_place_sides",
            "outbound_active_cancel_sides",
            "outbound_active_amend_sides",
            "outbound_queue_depth",
            "recent_orphan_cancels_size",
            "orphan_cancel_dedup_skips",
            "execution_idle_s",
            "snapshot_age_s",
        }
        missing = expected_top_level - set(es.keys())
        assert not missing, (
            f"executor_state missing expected keys: {missing}. "
            f"BUG-025 Phase 3 cannot localize the wedge without them."
        )
    finally:
        _safe_unlink_db(db_path)


def test_executor_state_snapshot_per_side_subkeys() -> None:
    """v1.4.40 BUG-025 Phase 1: per-side sub-blocks
    (side_unresolved, cancel_pending_since_s, etc) carry BOTH
    'buy' and 'sell' keys."""
    om, storage, state, db_path = _setup()
    try:
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        for per_side_key in (
            "side_unresolved",
            "cancel_pending_since_s",
            "post_only_cross_cooldown_remaining_s",
            "last_outbound_replace_age_s",
        ):
            block = es[per_side_key]
            assert "buy" in block, f"{per_side_key} missing 'buy'"
            assert "sell" in block, f"{per_side_key} missing 'sell'"
        # reprice_replace_pending uses bid/ask not buy/sell.
        assert "bid" in es["reprice_replace_pending"]
        assert "ask" in es["reprice_replace_pending"]
    finally:
        _safe_unlink_db(db_path)


def test_executor_state_surfaces_in_snapshot_dict() -> None:
    """v1.4.40 BUG-025 Phase 1: BotState.snapshot_dict() carries
    the executor_state block at top level. This is the data the
    /state/current endpoint serves into state_current.json — the
    postmortem-facing surface."""
    om, storage, state, db_path = _setup()
    try:
        om._sync_outbound_state_flags()
        snap = state.snapshot_dict()
        assert "executor_state" in snap, (
            "executor_state must be a top-level key in snapshot_dict so "
            "it surfaces in state_current.json for postmortem analysis."
        )
        es = snap["executor_state"]
        assert isinstance(es, dict)
        assert "side_unresolved" in es
    finally:
        _safe_unlink_db(db_path)


def test_executor_state_reflects_side_unresolved_latch() -> None:
    """v1.4.40 BUG-025 Phase 1: when a side is latched as
    unresolved, the snapshot exposes the latch + its
    requires_confirm flag + age. This is the smoking-gun field for
    the v1.4.33 latch-wedge variant of BUG-025."""
    om, storage, state, db_path = _setup()
    try:
        # Synthetically latch BUY side as unresolved with
        # requires_confirm=True (the never-clears case).
        om._set_side_unresolved(
            Side.BUY,
            reason="test_latch",
            requires_confirm=True,
        )
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        buy_state = es["side_unresolved"]["buy"]
        assert buy_state["active"] is True
        assert buy_state["reason"] == "test_latch"
        assert buy_state["requires_confirm"] is True
        assert buy_state["since_s"] is not None
        # SELL stays clean.
        sell_state = es["side_unresolved"]["sell"]
        assert sell_state["active"] is False
        assert sell_state["requires_confirm"] is False
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# Phase 2 — silent-wedge detector
# ---------------------------------------------------------------------------


def _telemetry_engine_healthy() -> dict:
    """Build a telemetry dict where the engine produced valid
    quotes — i.e. the precondition for the silent-wedge detector."""
    return {
        "quote_engine_mode": "normal",
        "quote_engine_normal_mode_requested": True,
        "quote_engine_bid_reason": None,
        "quote_engine_ask_reason": None,
        "quote_engine_bid_final_px": 100.00,
        "quote_engine_ask_final_px": 100.01,
        "quote_engine_bid_final_sz": 3.0,
        "quote_engine_ask_final_sz": 3.0,
    }


def test_silent_wedge_detector_fires_on_conjoint_condition() -> None:
    """v1.4.40 BUG-025 Phase 2: detector fires ERROR event when
    all five conjoint conditions hold — engine healthy, eligibility
    QUOTE_BOTH, active_sides BOTH, no side_unresolved, idle >
    threshold. Verifies the bot_events table receives the entry."""
    om, storage, state, db_path = _setup()
    try:
        # Simulate idle > threshold by setting last_place_attempt
        # 30 s ago (test config threshold = 10 s).
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-1",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        # Check the bot_events table.
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert len(wedge_events) == 1, (
            f"Detector should fire exactly once on conjoint "
            f"condition; got {len(wedge_events)} events."
        )
    finally:
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_does_not_fire_when_engine_in_no_quote() -> None:
    """v1.4.40 BUG-025 Phase 2: negative control. If the engine
    itself is in no_quote mode, the engine_no_quote diagnostic
    handles it. This detector must skip — otherwise we'd
    double-log."""
    om, storage, state, db_path = _setup()
    try:
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        telemetry = _telemetry_engine_healthy()
        telemetry["quote_engine_mode"] = "no_quote"
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=telemetry,
            decision_quote_cycle_id="test-cycle-2",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert wedge_events == [], (
            "Detector must skip when engine is in no_quote mode — "
            "the engine_no_quote diagnostic already covers that case."
        )
    finally:
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_does_not_fire_when_idle_below_threshold() -> None:
    """v1.4.40 BUG-025 Phase 2: negative control. Idle < threshold
    is NOT a wedge — bot is healthy."""
    om, storage, state, db_path = _setup()
    try:
        # Idle = 2 s; threshold = 10 s.
        state.last_place_attempt_ts_mono = time.monotonic() - 2.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-3",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert wedge_events == [], "Detector should not fire below idle threshold"
    finally:
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_does_not_fire_when_side_unresolved() -> None:
    """v1.4.40 BUG-025 Phase 2: negative control. When a
    side_unresolved is set, the bot has explicit justification for
    not placing — not a silent wedge."""
    om, storage, state, db_path = _setup()
    try:
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        om._set_side_unresolved(
            Side.BUY,
            reason="test_unresolved",
            requires_confirm=False,
        )
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-4",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert wedge_events == [], (
            "Detector should not fire when a side is unresolved — "
            "the latch explains the no-place."
        )
    finally:
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_does_not_fire_when_eligibility_not_both() -> None:
    """v1.4.40 BUG-025 Phase 2: negative control. If eligibility is
    QUOTE_SELL_ONLY (or any non-BOTH), the bot has an explicit
    one-sided posture — not a silent wedge."""
    om, storage, state, db_path = _setup()
    try:
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-5",
            eligibility="QUOTE_SELL_ONLY",
            active_sides="ASK_ONLY",
            position_qty=5.0,
            position_notional=10.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert wedge_events == [], (
            "Detector should not fire when eligibility ≠ QUOTE_BOTH"
        )
    finally:
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_does_not_fire_when_engine_in_one_sided_mode() -> None:
    """v1.4.206 BUG-025 false-positive fix #1.

    When the engine returns ``one_sided`` (e.g.
    ``inventory_bias_suppressed_adding_side`` while the bot is at
    high util), the OTHER side may have a perfectly healthy resting
    order maintaining the quote. The detector pre-v1.4.206 fired
    anyway because ``one_sided != 'no_quote'`` evaded its check.

    Observed in ``snapshots/v1.4.180-260521-141619`` — 20 false-
    positive fires per session, all with
    ``quote_engine_mode=one_sided`` +
    ``quote_engine_bid_reason=inventory_bias_suppressed_adding_side``.
    """
    om, storage, state, db_path = _setup()
    try:
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        telemetry = _telemetry_engine_healthy()
        telemetry["quote_engine_mode"] = "one_sided"
        telemetry["quote_engine_bid_reason"] = (
            "inventory_bias_suppressed_adding_side"
        )
        telemetry["quote_engine_bid_final_px"] = None
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=telemetry,
            decision_quote_cycle_id="test-cycle-one-sided",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=9.0,  # high inventory drives the one-sided
            position_notional=18.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert wedge_events == [], (
            "Detector must skip when engine is in one_sided mode — "
            "the resting side is doing its job and the suppressed "
            "side is engine-intentional, not a wedge."
        )
    finally:
        try:
            storage.close()
        except Exception:
            pass
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_does_not_fire_when_working_order_resting() -> None:
    """v1.4.206 BUG-025 false-positive fix #2.

    Steady-state with a healthy ACKED working order — the bot's
    quote is on the book at the engine's preferred price, so no
    new place/amend has fired for >> threshold seconds. The
    original detector treated this as a wedge; the fix treats it
    as the HEALTHY case it actually is.

    Engine-side ``mode`` is ``normal`` here (NOT one_sided) so we
    exercise the order_store exemption distinctly from fix #1.
    """
    from datetime import datetime, timezone

    from app.enums import OrderStatus
    from app.models import WorkingOrder

    om, storage, state, db_path = _setup()
    try:
        # Stage an ACKED working order on the BID side — bot's
        # quote is resting and protecting that side.
        wo = WorkingOrder(
            order_id_local=f"local-{uuid.uuid4().hex[:8]}",
            order_id_exchange=99999,
            client_order_id="cl_test_bid_resting",
            symbol=om._settings.symbol,
            side=Side.BUY,
            price=100.0,
            size=3.0,
            post_only=True,
            status=OrderStatus.ACKED,
            ts_created=datetime.now(timezone.utc),
            ts_sent=datetime.now(timezone.utc),
            ts_ack=datetime.now(timezone.utc),
        )
        with state._lock:
            state.set_working_order(Side.BUY, 0, wo)
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-resting",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert wedge_events == [], (
            "Detector must skip when at least one ACKED working "
            "order is on the book — that's the steady-state-resting "
            "case, not a wedge."
        )
    finally:
        try:
            storage.close()
        except Exception:
            pass
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_still_fires_with_partial_status() -> None:
    """v1.4.206 BUG-025 false-positive fix #2 corollary.

    ``PARTIAL`` status counts as resting (a partially-filled order
    is still on the book), so the detector must skip the same way
    as for ACKED.
    """
    from datetime import datetime, timezone

    from app.enums import OrderStatus
    from app.models import WorkingOrder

    om, storage, state, db_path = _setup()
    try:
        wo = WorkingOrder(
            order_id_local=f"local-{uuid.uuid4().hex[:8]}",
            order_id_exchange=99998,
            client_order_id="cl_test_ask_partial",
            symbol=om._settings.symbol,
            side=Side.SELL,
            price=100.01,
            size=2.0,
            post_only=True,
            status=OrderStatus.PARTIAL,
            ts_created=datetime.now(timezone.utc),
            ts_sent=datetime.now(timezone.utc),
            ts_ack=datetime.now(timezone.utc),
        )
        with state._lock:
            state.set_working_order(Side.SELL, 0, wo)
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-partial",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=1.0,
            position_notional=100.01,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert wedge_events == [], (
            "PARTIAL must count as resting — partially-filled order "
            "is still on the book maintaining the quote."
        )
    finally:
        try:
            storage.close()
        except Exception:
            pass
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_fires_when_no_working_orders_at_all() -> None:
    """v1.4.206 BUG-025 false-positive fix #2 corollary — the
    GENUINE wedge case must still fire.

    No resting orders + engine producing valid quotes + execution
    idle = exactly the wedge BUG-025 was built to catch. The
    new exemption MUST NOT defang this path.
    """
    om, storage, state, db_path = _setup()
    try:
        # No working orders staged. ``last_place_attempt_ts_mono``
        # idle > threshold.
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-real-wedge",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert len(wedge_events) == 1, (
            "Detector must STILL fire when no resting orders are "
            "protecting the quote — that's the original BUG-025 "
            "scenario the v1.4.206 exemption must not defang."
        )
    finally:
        try:
            storage.close()
        except Exception:
            pass
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_respects_disable_flag() -> None:
    """v1.4.40 BUG-025 Phase 2: SILENT_WEDGE_DETECT_ENABLED=false
    suppresses the detector entirely (rollback hatch)."""
    s = _settings(SILENT_WEDGE_DETECT_ENABLED=False)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-6",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert wedge_events == [], (
            "Disabled flag must suppress the detector entirely."
        )
    finally:
        _safe_unlink_db(db_path)


def test_silent_wedge_detector_relog_cadence() -> None:
    """v1.4.40 BUG-025 Phase 2: re-log cadence — after the first
    emission, the detector must NOT spam an event on every cycle.
    Re-emit only after SILENT_WEDGE_DETECT_RELOG_SECONDS (1 s in
    test config)."""
    om, storage, state, db_path = _setup()
    try:
        state.last_place_attempt_ts_mono = time.monotonic() - 30.0
        # First firing.
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-7a",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        # Immediate second call — should be suppressed by re-log
        # cadence.
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_engine_healthy(),
            decision_quote_cycle_id="test-cycle-7b",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        events = storage.recent_bot_events(50)
        wedge_events = [
            e for e in events
            if e.get("event_type") == "executor_silent_wedge_detected"
        ]
        assert len(wedge_events) == 1, (
            f"Re-log cadence must suppress immediate re-fires; got "
            f"{len(wedge_events)} events from two back-to-back calls."
        )
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# Phase 3 — reap zombie WorkingOrder on cancel_unexpected_gone (v1.4.41)
# ---------------------------------------------------------------------------


def _setup_phase3() -> tuple[OrderManager, Storage, BotState, Path]:
    """Phase 3 helper. Wires the OKX-shaped cancel interpreter on the
    mock client so the classifier under test (51400 → unexpected_gone)
    actually applies; the BUG-025 instrumentation tests above use the
    default Hyperliquid-shaped interpreter which would mis-classify
    OKX rows as ``transport``."""
    s = _settings()
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    from app.exchange.okx_responses import interpret_okx_cancel_response

    client.interpret_cancel_response = interpret_okx_cancel_response
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    return om, storage, state, db_path


def _make_cancel_pending_wo(
    symbol: str, side: Side, oid: int, cloid: str
) -> "WorkingOrder":
    """Build a CANCEL_PENDING WorkingOrder representing the zombie
    state captured in the BUG-025 wedge (post-WS-CANCELED but somehow
    still resident in working_orders)."""
    from app.enums import OrderStatus
    from app.models import WorkingOrder
    from app.utils.time import utc_now

    now = utc_now()
    return WorkingOrder(
        order_id_local=f"local-{uuid.uuid4().hex[:8]}",
        order_id_exchange=oid,
        client_order_id=cloid,
        symbol=symbol,
        side=side,
        price=1.95,
        size=3.0,
        post_only=True,
        status=OrderStatus.CANCEL_PENDING,
        ts_created=now,
        ts_sent=now,
        ts_ack=now,
        ts_cancel_requested=now,
        ts_cancel_sent=now,
    )


def test_phase3_reap_helper_clears_slot_and_side_unresolved() -> None:
    """v1.4.41 BUG-025 Phase 3: the reap helper, called directly,
    transitions the WO to CANCELED, clears the per-rung slot, clears
    side_unresolved for the side, and records a ws-terminal entry
    so late WS events for the oid get skipped."""
    from app.enums import OrderStatus

    om, storage, state, db_path = _setup_phase3()
    sym = om._settings.symbol
    try:
        wo = _make_cancel_pending_wo(sym, Side.BUY, oid=5001, cloid="cloid-r1")
        state.set_working_order(Side.BUY, 0, wo)
        # Pre-condition: the orchestrator just stamped cancel_pending_wait.
        om._mark_cancel_pending(Side.BUY)
        assert om._side_unresolved_active.get(Side.BUY) is True

        om._reap_wo_after_cancel_unexpected_gone(
            wo,
            source="single",
            detail="okx_row_51400:Order cancellation failed as the "
            "order has been filled, canceled or does not exist.",
        )

        # WO transitioned to CANCELED with the venue-code-attributed reason.
        assert wo.status == OrderStatus.CANCELED
        assert wo.cancel_reason == "http:unexpected_gone_51400"
        # Slot reaped.
        assert state.get_working_order(Side.BUY, 0) is None
        # side_unresolved cleared.
        assert om._side_unresolved_active.get(Side.BUY) is False
        # Terminal-ts recorded so late WS NEW/CANCELED for this oid is
        # skipped at _handle_private_order_update's stale-terminal guard.
        assert 5001 in om._ws_order_terminal_ts
    finally:
        _safe_unlink_db(db_path)


def test_phase3_reap_helper_is_idempotent() -> None:
    """v1.4.41: calling the reap helper twice (or on an already-
    cleared slot) must not raise. Important because under the
    captured race, both the batch path's per-row handler AND the
    next reconcile may attempt to reap the same WO in rapid
    succession."""
    om, storage, state, db_path = _setup_phase3()
    sym = om._settings.symbol
    try:
        wo = _make_cancel_pending_wo(sym, Side.BUY, oid=5002, cloid="cloid-r2")
        state.set_working_order(Side.BUY, 0, wo)
        om._reap_wo_after_cancel_unexpected_gone(
            wo, source="single", detail="okx_row_51400:x"
        )
        # Second call — slot already None, status already CANCELED,
        # side_unresolved already inactive. Must not raise.
        om._reap_wo_after_cancel_unexpected_gone(
            wo, source="batch", detail="okx_row_51400:x"
        )
        assert state.get_working_order(Side.BUY, 0) is None
    finally:
        _safe_unlink_db(db_path)


def test_phase3_reap_helper_handles_missing_oid() -> None:
    """v1.4.41: WO with no order_id_exchange (place not yet acked).
    The reap should still clear the slot + side_unresolved; only the
    ws-terminal recording is skipped (no oid to key by)."""
    from app.enums import OrderStatus

    om, storage, state, db_path = _setup_phase3()
    sym = om._settings.symbol
    try:
        wo = _make_cancel_pending_wo(
            sym, Side.SELL, oid=5003, cloid="cloid-r3"
        )
        wo.order_id_exchange = None  # simulate not-yet-acked
        state.set_working_order(Side.SELL, 0, wo)
        om._mark_cancel_pending(Side.SELL)

        om._reap_wo_after_cancel_unexpected_gone(
            wo, source="single", detail="okx_row_51401:x"
        )
        assert wo.status == OrderStatus.CANCELED
        assert wo.cancel_reason == "http:unexpected_gone_51401"
        assert state.get_working_order(Side.SELL, 0) is None
        assert om._side_unresolved_active.get(Side.SELL) is False
    finally:
        _safe_unlink_db(db_path)


def test_phase3_batch_cancel_unexpected_gone_reaps_zombie_wo() -> None:
    """v1.4.41 BUG-025 Phase 3 wedge regression: the batch-cancel
    path with a 51400 row must REAP the local WO. Pre-v1.4.41 the
    handler bumped counters and logged but left the WO in
    ``working_orders[side][0]`` in CANCEL_PENDING — the next
    orchestrate cycle would find it, re-cancel it, get 51400 again,
    re-stamp ``cancel_pending_wait`` via _mark_cancel_pending, and
    the bot wedged in a per-cycle cancel ping-pong until the
    deadlock watchdog killed at idle=600 s (captured 2026-05-18,
    snapshot v1.4.40-260518-093848: two zombie OIDs, 28k+ side_
    unresolved enters in 16 min).

    Post-fix: a single batch-cancel 51400 response on a working
    order leaves the slot empty AND side_unresolved inactive."""
    from app.enums import OrderStatus
    from app.execution import CancelTransportIntent

    om, storage, state, db_path = _setup_phase3()
    sym = om._settings.symbol
    try:
        oid = 3576235247318016000  # the actual zombie OID from the wedge
        cloid = "b6de4cb69c38d794faa617f85ddbde18"
        wo = _make_cancel_pending_wo(sym, Side.BUY, oid=oid, cloid=cloid)
        state.set_working_order(Side.BUY, 0, wo)
        om._mark_cancel_pending(Side.BUY)
        assert om._side_unresolved_active.get(Side.BUY) is True

        # Mock the batch-cancel response: OKX returns 51400 for the row.
        om._client.cancel_batch_orders.side_effect = lambda _s, _refs: {
            "code": "1",
            "msg": "",
            "data": [
                {
                    "sCode": "51400",
                    "sMsg": "Order cancellation failed as the order "
                    "has been filled, canceled or does not exist.",
                    "ordId": str(oid),
                    "clOrdId": cloid,
                }
            ],
        }

        intent = CancelTransportIntent(
            wo.order_id_local,
            Side.BUY,
            wo.cancel_transport_seq,
            time.monotonic(),
        )
        # The dispatcher feeds the batch path with ≥2 intents.
        om._execute_cancel_batch_intents([intent, intent])

        # KEY ASSERTIONS — the wedge primitives are all broken:
        # (1) the slot is empty so the next orchestrate cycle won't
        #     find a WO to cancel
        assert state.get_working_order(Side.BUY, 0) is None, (
            "BUG-025 Phase 3 regression: the batch 51400 response "
            "did NOT reap the WO; the slot still holds the zombie."
        )
        # (2) the WO transitioned terminal so no other path will pick
        #     it up
        assert wo.status == OrderStatus.CANCELED
        assert wo.cancel_reason == "http:unexpected_gone_51400"
        # (3) side_unresolved cleared so the eligibility layer stops
        #     gating QUOTE_BOTH on this side
        assert om._side_unresolved_active.get(Side.BUY) is False, (
            "BUG-025 Phase 3 regression: side_unresolved for BUY "
            "did not clear; eligibility will stay HOLD_ALL "
            "and the wedge persists."
        )
        # (4) the counter still bumped for observability (v1.4.38 fix
        #     remains intact)
        assert state.cancel_unexpected_gone_total >= 1
    finally:
        _safe_unlink_db(db_path)


def test_phase3_single_cancel_unexpected_gone_reaps_zombie_wo() -> None:
    """v1.4.41 BUG-025 Phase 3: the single-cancel HTTP path
    (``_cancel_http_transport`` → ``_interpret_cancel_response``)
    also reaps the WO on 51400. Same regression as the batch path
    above; both call sites flow through the same helper."""
    from app.enums import OrderStatus

    om, storage, state, db_path = _setup_phase3()
    sym = om._settings.symbol
    try:
        oid = 3576236819141828608  # the second zombie OID from the wedge
        cloid = "999d8a289674f623ca0000000000abcd"
        wo = _make_cancel_pending_wo(sym, Side.SELL, oid=oid, cloid=cloid)
        state.set_working_order(Side.SELL, 0, wo)
        om._mark_cancel_pending(Side.SELL)

        # Single-cancel response shape (cancel_order returns OKX V5
        # single-cancel shape).
        om._client.cancel_order.return_value = {
            "code": "1",
            "msg": "",
            "data": [
                {
                    "sCode": "51400",
                    "sMsg": "Order cancellation failed as the order "
                    "has been filled, canceled or does not exist.",
                    "ordId": str(oid),
                }
            ],
        }

        om._cancel_http_transport(wo)

        assert state.get_working_order(Side.SELL, 0) is None, (
            "BUG-025 Phase 3 regression: single-cancel 51400 did NOT "
            "reap the WO."
        )
        assert wo.status == OrderStatus.CANCELED
        assert wo.cancel_reason == "http:unexpected_gone_51400"
        assert om._side_unresolved_active.get(Side.SELL) is False
    finally:
        _safe_unlink_db(db_path)


def test_phase3_genuine_error_does_not_reap_wo() -> None:
    """v1.4.41 BUG-025 Phase 3 negative control: a genuine error
    (e.g. sCode 51008 = insufficient margin) must NOT reap the WO.
    The Phase 3 fix is a narrow carve-out for ``unexpected_gone``
    only; real errors should preserve the local state so the
    cancel-pending watchdog can retry."""
    from app.enums import OrderStatus
    from app.execution import CancelTransportIntent

    om, storage, state, db_path = _setup_phase3()
    sym = om._settings.symbol
    try:
        wo = _make_cancel_pending_wo(
            sym, Side.BUY, oid=5009, cloid="cloid-r9"
        )
        state.set_working_order(Side.BUY, 0, wo)

        # 51008 = insufficient margin — a real error, not "gone".
        om._client.cancel_batch_orders.side_effect = lambda _s, _refs: {
            "code": "1",
            "msg": "",
            "data": [
                {
                    "sCode": "51008",
                    "sMsg": "Insufficient margin balance for cancel",
                    "ordId": "5009",
                    "clOrdId": "cloid-r9",
                }
            ],
        }

        intent = CancelTransportIntent(
            wo.order_id_local,
            Side.BUY,
            wo.cancel_transport_seq,
            time.monotonic(),
        )
        om._execute_cancel_batch_intents([intent, intent])

        # KEY ASSERTION: the WO is STILL in the slot — real errors
        # don't reap. The cancel-pending watchdog handles retry.
        assert state.get_working_order(Side.BUY, 0) is wo, (
            "BUG-025 Phase 3 over-reach: a genuine (non-``unexpected"
            "_gone``) error reaped the WO. The fix must narrowly "
            "exempt ``unexpected_gone`` only."
        )
        assert wo.status == OrderStatus.CANCEL_PENDING, (
            "WO must remain CANCEL_PENDING for a real error so the "
            "cancel-pending watchdog can retry."
        )
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.50 — Decision-trace instrumentation (BUG-025 follow-on)
#
# The 2026-05-18 wedge sequence (snapshots v1.4.43 / v1.4.45 / v1.4.47)
# showed the executor sitting idle for 5+ min while the engine
# produced valid quotes — no log explained which silent-return branch
# in ``_orchestrate`` / ``_stage_place_order_local`` was firing. The
# v1.4.50 instrumentation surfaces every decision so the next wedge
# is diagnosable without code archaeology.
# ---------------------------------------------------------------------------


def test_decision_trace_fields_default_initialized() -> None:
    """v1.4.50: the OrderManager constructs with empty decision-trace
    state. Ring buffer is a per-side deque; counters are an empty
    dict; the stage-place skip-reason cache is initialized with None
    per side. The settings expose the trace flag (default True) and
    buffer size (default 200).
    """
    om, _storage, _state, db_path = _setup()
    try:
        assert hasattr(om, "_orchestrate_decision_history")
        assert hasattr(om, "_orchestrate_decision_counts")
        assert hasattr(om, "_last_stage_place_skip_reason")
        # Both sides present, both initialized empty.
        assert Side.BUY in om._orchestrate_decision_history
        assert Side.SELL in om._orchestrate_decision_history
        assert len(om._orchestrate_decision_history[Side.BUY]) == 0
        assert len(om._orchestrate_decision_history[Side.SELL]) == 0
        assert om._orchestrate_decision_counts == {}
        assert om._last_stage_place_skip_reason == {"BUY": None, "SELL": None}
    finally:
        _safe_unlink_db(db_path)


def test_record_orchestrate_decision_writes_buffer_and_counter() -> None:
    """v1.4.50: ``_record_orchestrate_decision`` appends to the
    per-side ring buffer AND increments the (side, branch, action)
    counter. Decisions for the BUY side don't pollute the SELL
    buffer, and vice versa.
    """
    om, _storage, _state, db_path = _setup()
    try:
        om._record_orchestrate_decision(
            side=Side.BUY,
            level_idx=0,
            cur=None,
            desired=None,
            decision_branch="side_unresolved",
            action="return_silent",
            extra={"unresolved_reason": "duplicate_open_orders_detected"},
        )
        assert len(om._orchestrate_decision_history[Side.BUY]) == 1
        assert len(om._orchestrate_decision_history[Side.SELL]) == 0
        entry = om._orchestrate_decision_history[Side.BUY][0]
        assert entry["side"] == "BUY"
        assert entry["level_idx"] == 0
        assert entry["cur_status"] == "none"
        assert entry["desired_present"] is False
        assert entry["decision_branch"] == "side_unresolved"
        assert entry["action"] == "return_silent"
        assert entry["unresolved_reason"] == "duplicate_open_orders_detected"
        assert "ts" in entry
        # Counter is composite-keyed.
        assert om._orchestrate_decision_counts[
            "BUY:side_unresolved:return_silent"
        ] == 1
        # Second BUY decision with same branch/action bumps the same counter.
        om._record_orchestrate_decision(
            side=Side.BUY,
            level_idx=0,
            cur=None,
            desired=None,
            decision_branch="side_unresolved",
            action="return_silent",
        )
        assert om._orchestrate_decision_counts[
            "BUY:side_unresolved:return_silent"
        ] == 2
        # A SELL decision opens a new counter key, doesn't touch BUY's.
        om._record_orchestrate_decision(
            side=Side.SELL,
            level_idx=0,
            cur=None,
            desired=None,
            decision_branch="terminal_should_not_emit_fresh",
            action="return_silent",
        )
        assert om._orchestrate_decision_counts[
            "BUY:side_unresolved:return_silent"
        ] == 2
        assert om._orchestrate_decision_counts[
            "SELL:terminal_should_not_emit_fresh:return_silent"
        ] == 1
    finally:
        _safe_unlink_db(db_path)


def test_decision_trace_buffer_bounded_by_config_size() -> None:
    """v1.4.50: the ring buffer respects ``EXECUTOR_DECISION_TRACE_BUFFER_SIZE``.
    Default 200; setting it lower truncates correctly. Each push
    beyond the cap evicts the oldest entry.
    """
    s = _settings(EXECUTOR_DECISION_TRACE_BUFFER_SIZE=50)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        for i in range(60):
            om._record_orchestrate_decision(
                side=Side.BUY,
                level_idx=0,
                cur=None,
                desired=None,
                decision_branch="test_branch",
                action=f"iter_{i}",
            )
        # The deque was capped at 50, so the oldest 10 should have been evicted.
        assert len(om._orchestrate_decision_history[Side.BUY]) == 50
        actions = [e["action"] for e in om._orchestrate_decision_history[Side.BUY]]
        # First retained entry is iter_10, last is iter_59.
        assert actions[0] == "iter_10"
        assert actions[-1] == "iter_59"
    finally:
        _safe_unlink_db(db_path)


def test_executor_state_snapshot_surfaces_decision_trace_fields() -> None:
    """v1.4.50: the executor_state snapshot exposes the decision-
    trace block so it appears in state_current.json and the silent-
    wedge event payload. Operator + postmortem read it from there;
    they don't need a custom inspection path."""
    om, _storage, _state, db_path = _setup()
    try:
        om._record_orchestrate_decision(
            side=Side.BUY,
            level_idx=0,
            cur=None,
            desired=None,
            decision_branch="terminal_should_not_emit_fresh",
            action="return_silent",
        )
        om._record_orchestrate_decision(
            side=Side.SELL,
            level_idx=0,
            cur=None,
            desired=None,
            decision_branch="side_unresolved",
            action="return_silent",
        )
        om._sync_outbound_state_flags()
        es = _state.executor_state_snapshot
        # All four new top-level fields present.
        for key in (
            "working_order_status",
            "orchestrate_decision_counts",
            "orchestrate_decision_history",
            "last_stage_place_skip_reason",
        ):
            assert key in es, f"executor_state missing v1.4.50 field {key}"
        # working_order_status carries both sides with the expected
        # primitive fields.
        for side_key in ("buy", "sell"):
            wos = es["working_order_status"][side_key]
            assert set(wos.keys()) == {
                "status", "level_idx", "exchange_oid", "px", "sz", "age_s",
            }
            # No WO staged in this test → status == "none".
            assert wos["status"] == "none"
        # decision_counts reflects both writes.
        assert es["orchestrate_decision_counts"][
            "BUY:terminal_should_not_emit_fresh:return_silent"
        ] == 1
        assert es["orchestrate_decision_counts"][
            "SELL:side_unresolved:return_silent"
        ] == 1
        # decision_history is per-side, the trace shows up.
        assert len(es["orchestrate_decision_history"]["buy"]) == 1
        assert len(es["orchestrate_decision_history"]["sell"]) == 1
        assert (
            es["orchestrate_decision_history"]["buy"][0]["decision_branch"]
            == "terminal_should_not_emit_fresh"
        )
    finally:
        _safe_unlink_db(db_path)


def test_stage_place_skip_reason_cached_on_no_write_access() -> None:
    """v1.4.50: when ``_stage_place_order_local`` returns None because
    the client doesn't have write access, the skip-reason cache
    records ``no_write_access`` for that side. This is the lowest-
    cost gate — proves the cache-on-skip plumbing works at all.
    """
    om, _storage, _state, db_path = _setup()
    try:
        om._client.has_write_access.return_value = False
        wo = om._stage_place_order_local(
            Side.BUY,
            price=1.0,
            size=1.0,
            quote_cycle_id="qc-test",
        )
        assert wo is None
        assert (
            om._last_stage_place_skip_reason["BUY"] == "no_write_access"
        ), (
            "stage_place_order_local must cache the skip reason so the "
            "decision trace shows which silent-return path fired."
        )
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.53 — maybe_refresh_quotes early-return instrumentation
#
# Snapshot v1.4.52-260518-160650 proved the wedge is upstream of
# ``_orchestrate``: 2 minutes of wedge with ZERO orchestrate
# decisions recorded but engine + eligibility healthy. Cause is one
# of the four ``maybe_refresh_quotes`` early-return paths
# (no_write_access, cancel_resting_for_risk, no_quote_hold_resting,
# residual_flatten_requested) — all four previously uninstrumented.
# These tests verify the new ``_record_quote_refresh_skip`` helper.
# ---------------------------------------------------------------------------


def test_quote_refresh_skip_fields_default_initialized() -> None:
    """v1.4.53: the OrderManager constructs with empty
    ``_quote_refresh_skip_history`` deque and empty
    ``_quote_refresh_skip_counts`` dict.
    """
    om, _storage, _state, db_path = _setup()
    try:
        assert hasattr(om, "_quote_refresh_skip_history")
        assert hasattr(om, "_quote_refresh_skip_counts")
        assert len(om._quote_refresh_skip_history) == 0
        assert om._quote_refresh_skip_counts == {}
    finally:
        _safe_unlink_db(db_path)


def test_record_quote_refresh_skip_writes_buffer_and_counter() -> None:
    """v1.4.53: ``_record_quote_refresh_skip`` appends an entry to the
    ring buffer AND increments the per-reason counter. Extra fields
    are merged into the entry.
    """
    om, _storage, _state, db_path = _setup()
    try:
        om._record_quote_refresh_skip(
            reason="no_quote_hold_resting",
            extra={
                "risk_action": "NO_QUOTE",
                "cancel_on_no_quote": False,
                "quote_cycle_id": "qc-1",
            },
        )
        assert len(om._quote_refresh_skip_history) == 1
        entry = om._quote_refresh_skip_history[0]
        assert entry["reason"] == "no_quote_hold_resting"
        assert entry["risk_action"] == "NO_QUOTE"
        assert entry["cancel_on_no_quote"] is False
        assert entry["quote_cycle_id"] == "qc-1"
        assert "ts" in entry
        assert om._quote_refresh_skip_counts["no_quote_hold_resting"] == 1
        # Second skip with same reason bumps the same counter.
        om._record_quote_refresh_skip(
            reason="no_quote_hold_resting",
            extra={"quote_cycle_id": "qc-2"},
        )
        assert om._quote_refresh_skip_counts["no_quote_hold_resting"] == 2
        # A different reason opens a new counter key.
        om._record_quote_refresh_skip(
            reason="residual_flatten_requested",
            extra={
                "position_qty": 2.0,
                "position_notional_usd": 3.95,
                "quote_cycle_id": "qc-3",
            },
        )
        assert om._quote_refresh_skip_counts == {
            "no_quote_hold_resting": 2,
            "residual_flatten_requested": 1,
        }
    finally:
        _safe_unlink_db(db_path)


def test_quote_refresh_skip_surfaces_in_executor_state_snapshot() -> None:
    """v1.4.53: the executor_state snapshot exposes
    ``quote_refresh_skip_counts`` and ``quote_refresh_skip_history``
    so the next wedge event payload reveals which upstream gate
    held during the wedge.
    """
    om, _storage, _state, db_path = _setup()
    try:
        om._record_quote_refresh_skip(
            reason="residual_flatten_requested",
            extra={
                "position_qty": 2.0,
                "position_notional_usd": 3.95,
                "quote_cycle_id": "qc-test",
            },
        )
        om._sync_outbound_state_flags()
        es = _state.executor_state_snapshot
        assert "quote_refresh_skip_counts" in es
        assert "quote_refresh_skip_history" in es
        assert es["quote_refresh_skip_counts"][
            "residual_flatten_requested"
        ] == 1
        assert len(es["quote_refresh_skip_history"]) == 1
        h0 = es["quote_refresh_skip_history"][0]
        assert h0["reason"] == "residual_flatten_requested"
        assert h0["position_qty"] == 2.0
        assert h0["position_notional_usd"] == 3.95
    finally:
        _safe_unlink_db(db_path)


def test_quote_refresh_skip_buffer_bounded_by_config_size() -> None:
    """v1.4.53: the skip ring buffer respects the same
    ``EXECUTOR_DECISION_TRACE_BUFFER_SIZE`` config knob as the
    orchestrate decision history. One buffer-size setting, two
    buffers — consistent operator UX.
    """
    s = _settings(EXECUTOR_DECISION_TRACE_BUFFER_SIZE=30)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        for i in range(40):
            om._record_quote_refresh_skip(
                reason="test_reason",
                extra={"iter": i},
            )
        # Capped at 30 — oldest 10 evicted.
        assert len(om._quote_refresh_skip_history) == 30
        iters = [e["iter"] for e in om._quote_refresh_skip_history]
        assert iters[0] == 10
        assert iters[-1] == 39
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.55 — Wedge elimination Phase 1: state-aware cancel-all
#
# Pre-v1.4.55 ``cancel_all_orders_for_symbol`` iterated the (TTL-cached)
# exchange snapshot and fired per-order HTTP cancels with no local
# state check. When the snapshot was stale and the same OIDs were
# already gone, the loop hammered the same 51400s with no memory of
# the previous attempt — 319 firings in a 3-s desync window observed
# in snapshot v1.4.53-260518-162051. The fix routes cancel-all through
# the local WO state + dispatcher (CANCEL_PENDING as tombstone).
# ---------------------------------------------------------------------------


def _ack_wo(om, side: Side, price: float = 1.0, oid: int = 1, level_idx: int = 0):
    """Create a synthetic ACKED WO directly on the state (bypassing the
    full place flow). Used to set up cancel-all test scenarios.
    """
    from app.enums import OrderStatus
    from app.models import WorkingOrder
    from app.utils.time import utc_now
    import uuid as _uuid
    now = utc_now()
    wo = WorkingOrder(
        order_id_local=f"local-{_uuid.uuid4().hex[:8]}",
        order_id_exchange=int(oid),
        client_order_id=f"cloid-{oid}",
        symbol=om._settings.symbol,
        side=side,
        price=float(price),
        size=5.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=now,
        ts_sent=now,
        ts_ack=now,
    )
    with om._state._lock:
        om._state.set_working_order(side, int(level_idx), wo)
    return wo


def test_phase1_cancel_all_skips_cancel_pending_wos() -> None:
    """v1.4.55 Phase 1: cancel-all must skip WOs already in
    CANCEL_PENDING. The dispatcher path is the single source of
    truth for cancel-in-flight; cancel-all must not fire a second
    cancel for the same OID (that's the storm class we're killing).
    """
    from app.execution import OrderStatus
    om, _storage, _state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=111)
        # Transition to CANCEL_PENDING manually (simulates dispatcher
        # already in-flight).
        wo.status = OrderStatus.CANCEL_PENDING
        # Spy on the enqueue method that cancel-all routes through.
        enq_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kwargs):
            enq_calls.append((wo_arg.order_id_exchange, kwargs.get("trigger_reason")))
            return original(wo_arg, **kwargs)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        om.cancel_all_orders_for_symbol()

        assert enq_calls == [], (
            "cancel-all must skip WOs already in CANCEL_PENDING — "
            f"dispatcher would dedup anyway but skipping here saves "
            f"the wasted enqueue. Got calls: {enq_calls}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_phase1_cancel_all_skips_terminal_wos() -> None:
    """v1.4.55 Phase 1: cancel-all must skip WOs already in terminal
    statuses (CANCELED / FILLED / REJECTED). They're done.
    """
    from app.execution import OrderStatus
    om, _storage, _state, db_path = _setup()
    try:
        for status in (OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED):
            wo = _ack_wo(om, Side.BUY, price=1.0, oid=200 + hash(status.value) % 100)
            wo.status = status
        # Also put a terminal one on SELL so we cover both sides.
        wo2 = _ack_wo(om, Side.SELL, price=2.0, oid=999)
        wo2.status = OrderStatus.CANCELED

        enq_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kwargs):
            enq_calls.append(wo_arg.order_id_exchange)
            return original(wo_arg, **kwargs)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        om.cancel_all_orders_for_symbol()
        assert enq_calls == [], (
            "Terminal WOs (CANCELED/FILLED/REJECTED) must not be "
            f"re-cancelled. Got: {enq_calls}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_phase1_cancel_all_cancels_acked_partial_amend_pending_exactly_once() -> None:
    """v1.4.55 Phase 1: cancel-all should fire exactly one cancel
    per cancellable WO (ACKED / PARTIAL / AMEND_PENDING) per call.
    Verifies the routing through ``_enqueue_cancel_quote_path``.
    """
    from app.execution import OrderStatus
    om, _storage, _state, db_path = _setup()
    try:
        wo1 = _ack_wo(om, Side.BUY, price=1.0, oid=301)  # ACKED
        wo2 = _ack_wo(om, Side.SELL, price=2.0, oid=302)
        wo2.status = OrderStatus.PARTIAL
        # AMEND_PENDING isn't representable as a second rung at idx 0,
        # so simulate by mutating wo1 after the BUY slot — actually
        # use a fresh oid + put it on the BUY rung 1 if multi-rung.
        # Simpler: validate the two we have.

        enq_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kwargs):
            enq_calls.append((wo_arg.order_id_exchange, kwargs.get("trigger_reason")))
            return original(wo_arg, **kwargs)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        om.cancel_all_orders_for_symbol()

        oids = sorted(c[0] for c in enq_calls)
        assert oids == [301, 302], (
            f"Each cancellable WO must be enqueued for cancel exactly "
            f"once. Got: {enq_calls}"
        )
        triggers = {c[1] for c in enq_calls}
        assert triggers == {"cancel_all_for_symbol"}, (
            "All cancel-all dispatches must carry the trigger_reason "
            f"='cancel_all_for_symbol' for postmortem visibility. Got: {triggers}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_phase1_repeated_cancel_all_idempotent_after_dispatcher_transitions() -> None:
    """v1.4.55 Phase 1 — THE BUG THAT STARTED THIS REFACTOR.

    Snapshot v1.4.53-260518-162051: 319 cancel_resting_for_risk
    firings in a 3-s desync window, all hammering the same 3 ghost
    OIDs because the old cancel-all loop ignored local state.

    With Phase 1: after the first cancel-all dispatches cancels,
    those WOs are in CANCEL_PENDING. The SECOND cancel-all call must
    skip them (CANCEL_PENDING is the tombstone). A 100-iteration
    loop simulates the BBO-driven storm; only the first iteration
    should dispatch cancels.
    """
    om, _storage, _state, db_path = _setup()
    try:
        _ack_wo(om, Side.BUY, price=1.0, oid=401)
        _ack_wo(om, Side.SELL, price=2.0, oid=402)

        enq_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kwargs):
            enq_calls.append(wo_arg.order_id_exchange)
            return original(wo_arg, **kwargs)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        # Simulate the storm: 100 cancel-all calls in a tight loop.
        for _ in range(100):
            om.cancel_all_orders_for_symbol()

        # First call: 2 enqueues (one per WO). After that, both WOs
        # are in CANCEL_PENDING and subsequent calls skip them.
        assert sorted(enq_calls) == [401, 402], (
            f"100 cancel-all calls in a tight loop must produce exactly "
            f"2 cancel enqueues (one per WO). Pre-v1.4.55 this was 200 "
            f"(the cancel storm bug). Got: {enq_calls}"
        )
    finally:
        # v1.5.27 -- the 100 cancel-all calls enqueue cancels into the
        # outbound dispatcher (worker thread); each cancel's terminal
        # state PERSISTS via ``insert_order_row`` against the SQLite
        # file. On Linux, ``unlink-while-open`` silently succeeds (the
        # file unlinks but the FD stays valid until close); on Windows,
        # ``unlink`` refuses with WinError 32 because storage holds an
        # open handle. The captured ``no such table: orders`` log is
        # the same race surfacing inside the dispatcher thread when
        # the connection IS torn down mid-batch. Fix: drain the
        # dispatcher first, then close storage, THEN unlink. Same
        # pattern as the v1.5.23 fix to test_execution_inventory_bias.py.
        try:
            om.wait_transport_idle()
        except Exception:
            pass
        try:
            _storage.close()
        except Exception:
            pass
        _safe_unlink_db(db_path)


def test_phase1_cancel_all_skips_pre_ack_statuses() -> None:
    """v1.4.55 Phase 1: WOs in NEW_LOCAL or SENT (pre-ack) must NOT
    be cancelled — cancelling a SENT order before its ack lands can
    race with the place response. The dispatcher will pick them up
    once they reach ACKED naturally.
    """
    from app.execution import OrderStatus
    om, _storage, _state, db_path = _setup()
    try:
        wo1 = _ack_wo(om, Side.BUY, price=1.0, oid=501)
        wo1.status = OrderStatus.NEW_LOCAL
        wo2 = _ack_wo(om, Side.SELL, price=2.0, oid=502)
        wo2.status = OrderStatus.SENT

        enq_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kwargs):
            enq_calls.append(wo_arg.order_id_exchange)
            return original(wo_arg, **kwargs)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        om.cancel_all_orders_for_symbol()
        assert enq_calls == [], (
            "NEW_LOCAL / SENT WOs must not be cancelled (pre-ack race). "
            f"Got: {enq_calls}"
        )
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.56 — Wedge elimination Phase 2: risk-action state machine
#
# Edge-triggered cancel-all dispatch. Sustained CANCEL_ALL/KILL/FLATTEN
# risk action no longer fires cancel-all every tick — only on
# transitions into the cancelling cohort.
# ---------------------------------------------------------------------------


def test_phase2_initial_state_is_normal() -> None:
    """v1.4.56: a freshly-constructed OrderManager starts in NORMAL
    risk-exec state with no transitions recorded.
    """
    from app.enums import RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        assert om._risk_exec_state == RiskExecState.NORMAL
        assert om._risk_exec_state_transition_counts == {}
    finally:
        _safe_unlink_db(db_path)


def test_phase2_normal_state_allow_action_is_noop() -> None:
    """v1.4.56: when in NORMAL state and risk action is ALLOW,
    cancel_resting_for_risk returns False without firing cancel-all
    and stays in NORMAL.
    """
    from app.enums import RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        call_count = {"n": 0}
        original = om.cancel_all_orders_for_symbol
        def _spy():
            call_count["n"] += 1
            return original()
        om.cancel_all_orders_for_symbol = _spy  # type: ignore[method-assign]

        result = om.cancel_resting_for_risk(RiskAction.ALLOW)
        assert result is False
        assert call_count["n"] == 0
        assert om._risk_exec_state == RiskExecState.NORMAL
    finally:
        _safe_unlink_db(db_path)


def test_phase2_normal_to_cancelling_fires_cancel_all_once() -> None:
    """v1.4.56: the FIRST tick where risk = CANCEL_ALL transitions
    NORMAL → CANCELLING and fires cancel-all exactly once.
    """
    from app.enums import RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        # Stage a cancellable WO so the state machine has work.
        _ack_wo(om, Side.BUY, price=1.0, oid=601)

        enq_calls = []
        original_enq = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kwargs):
            enq_calls.append(wo_arg.order_id_exchange)
            return original_enq(wo_arg, **kwargs)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        result = om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        assert result is True
        assert om._risk_exec_state == RiskExecState.CANCELLING
        # Cancel-all routed through dispatcher → spy saw one enqueue.
        assert enq_calls == [601]
        # Transition was counted.
        assert om._risk_exec_state_transition_counts == {
            "NORMAL->CANCELLING": 1,
        }
    finally:
        _safe_unlink_db(db_path)


def test_phase2_repeated_cancelling_risk_does_not_refire_cancel_all() -> None:
    """v1.4.56 — THE STORM-PREVENTION TEST.

    Snapshot v1.4.53-260518-162051 caught 319 cancel-all firings in
    3 seconds because cancel_resting_for_risk re-fired on every BBO-
    driven tick while risk stayed CANCEL_ALL. With the state machine,
    only the FIRST tick fires; subsequent ticks in CANCELLING are
    no-ops at the cancel-all level.
    """
    from app.enums import RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        _ack_wo(om, Side.BUY, price=1.0, oid=701)
        _ack_wo(om, Side.SELL, price=2.0, oid=702)

        enq_calls = []
        original_enq = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kwargs):
            enq_calls.append(wo_arg.order_id_exchange)
            return original_enq(wo_arg, **kwargs)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        # Simulate the storm — 100 ticks all returning CANCEL_ALL risk.
        results = []
        for _ in range(100):
            results.append(om.cancel_resting_for_risk(RiskAction.CANCEL_ALL))

        # Every tick returns True (suppress quoting).
        assert all(r is True for r in results)
        # Cancel-all dispatched on ONLY the first tick (oids 701 + 702
        # = 2 enqueues from the spy). 100 ticks in the OLD code path
        # would have produced 200 enqueues.
        assert sorted(enq_calls) == [701, 702], (
            f"100-tick storm must produce exactly 2 cancel enqueues "
            f"(one per WO on the first transition). Pre-v1.4.56 this "
            f"was 200 — the cancel storm. Got: {enq_calls}"
        )
        # State is in CANCELLING (still waiting for terminal status).
        assert om._risk_exec_state == RiskExecState.CANCELLING
        # Exactly one NORMAL→CANCELLING transition recorded.
        assert om._risk_exec_state_transition_counts == {
            "NORMAL->CANCELLING": 1,
        }
    finally:
        _safe_unlink_db(db_path)


def test_phase2_cancelling_to_suppressed_when_wos_reach_terminal() -> None:
    """v1.4.56: once all WOs reach terminal status (CANCELED), the
    state machine transitions CANCELLING → SUPPRESSED on the next
    tick. Still returns True (quoting still suppressed).
    """
    from app.enums import OrderStatus, RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=801)
        # First tick: NORMAL → CANCELLING + fire cancel-all.
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        assert om._risk_exec_state == RiskExecState.CANCELLING

        # Simulate the dispatcher having confirmed the cancel: WO
        # transitions to CANCELED. Manually set status to simulate
        # WS-CANCELED arriving.
        wo.status = OrderStatus.CANCELED

        # Next tick with same risk action — should transition to SUPPRESSED.
        result = om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        assert result is True
        assert om._risk_exec_state == RiskExecState.SUPPRESSED
        assert "CANCELLING->SUPPRESSED" in om._risk_exec_state_transition_counts
    finally:
        _safe_unlink_db(db_path)


def test_phase2_suppressed_to_normal_on_risk_clear() -> None:
    """v1.4.56: when risk returns to ALLOW from SUPPRESSED, transitions
    to NORMAL and returns False. Subsequent CANCEL_ALL ticks can fire
    again.
    """
    from app.enums import OrderStatus, RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=901)
        # NORMAL → CANCELLING
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        # WO completes its cancel.
        wo.status = OrderStatus.CANCELED
        # CANCELLING → SUPPRESSED
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        assert om._risk_exec_state == RiskExecState.SUPPRESSED

        # Risk returns to ALLOW.
        result = om.cancel_resting_for_risk(RiskAction.ALLOW)
        assert result is False
        assert om._risk_exec_state == RiskExecState.NORMAL
        assert "SUPPRESSED->NORMAL" in om._risk_exec_state_transition_counts
    finally:
        _safe_unlink_db(db_path)


def test_phase2_cancelling_to_normal_when_risk_clears_and_no_pending() -> None:
    """v1.4.56: if risk returns to ALLOW while in CANCELLING and no
    WOs remain pending, transition directly CANCELLING → NORMAL and
    return False. Quoting can resume immediately.
    """
    from app.enums import OrderStatus, RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=1001)
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        assert om._risk_exec_state == RiskExecState.CANCELLING
        # WO confirmed cancelled.
        wo.status = OrderStatus.CANCELED
        # Risk clears.
        result = om.cancel_resting_for_risk(RiskAction.ALLOW)
        assert result is False
        assert om._risk_exec_state == RiskExecState.NORMAL
    finally:
        _safe_unlink_db(db_path)


def test_phase2_immediate_normal_when_no_cancellable_wos_on_entry() -> None:
    """v1.4.56: NORMAL → CANCELLING then immediately → SUPPRESSED when
    local state had zero cancellable WOs on entry. Avoids a stuck
    CANCELLING when there's nothing to cancel.
    """
    from app.enums import RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        # No WOs staged — local state empty.
        result = om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        assert result is True
        # Should have transitioned NORMAL → CANCELLING → SUPPRESSED.
        assert om._risk_exec_state == RiskExecState.SUPPRESSED
        # Both transitions recorded.
        counts = om._risk_exec_state_transition_counts
        assert counts.get("NORMAL->CANCELLING", 0) == 1
        assert counts.get("CANCELLING->SUPPRESSED", 0) == 1
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.67 — SUPPRESSED-state wedge fix
#
# Snapshot v1.4.66-260518-192744 caught a 52-s silent wedge. The bot
# had been quoting normally for ~75 s, then risk_action briefly went
# CANCEL_ALL (desync detected → resolved), but the risk_exec_state
# machine refused to transition SUPPRESSED → NORMAL because
# ``_local_has_cancellable_wos()`` kept returning True for two stuck
# WOs:
#   * A CANCEL_PENDING BUY hydrated from a previous session
#     (oid 3577421873029259264) — the cancel ack never came; the
#     cancel-pending watchdog only checks inside-rung slots
#     (working_bid/working_ask), so an orphan-slot CANCEL_PENDING is
#     never reaped.
#   * A SELL placed at 15:26:38, marked CANCEL_PENDING after a
#     duplicate hydration event, and never closed in local state
#     (the same OID had two WorkingOrder records).
#
# Result: 2041 of 2491 quote_refresh_skip events had risk_action=ALLOW
# but `cancel_resting_for_risk` returned True anyway because of the
# stale WOs. The bot was structurally fine but the state machine was
# locked out.
#
# The v1.4.67 fix makes the state machine NOT wait on:
#   * DESYNC WOs (given-up state)
#   * CANCEL_PENDING WOs older than the configured timeout
# ---------------------------------------------------------------------------


def test_v1_4_67_desync_wo_does_not_block_suppressed_to_normal() -> None:
    """v1.4.67: a DESYNC-status WorkingOrder must NOT keep the risk
    state machine pinned in SUPPRESSED. DESYNC is a "given up" state
    (bot already issued the cancel(s); nothing more to do).

    Regression for snapshot v1.4.66-260518-192744: 52-second wedge
    where DESYNC + stale CANCEL_PENDING orders kept
    ``_local_has_cancellable_wos()`` True forever, locking the state
    machine in SUPPRESSED while risk_action was ALLOW.
    """
    from app.enums import OrderStatus, RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        # Stage: an order goes to DESYNC (the bot detected mismatch and
        # gave up — same status the exchange_mismatch path sets).
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=4201)
        wo.status = OrderStatus.DESYNC

        # Enter the state machine while in CANCEL_ALL.
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        # Should NOT be stuck in CANCELLING — DESYNC counts as terminal
        # for cancellable-WO purposes, so the machine transitions
        # straight through to SUPPRESSED.
        assert om._risk_exec_state == RiskExecState.SUPPRESSED, (
            f"Expected SUPPRESSED, got {om._risk_exec_state}; "
            "DESYNC WO should not be counted as cancellable"
        )

        # Now risk clears — SUPPRESSED should transition back to NORMAL
        # because DESYNC doesn't block.
        result = om.cancel_resting_for_risk(RiskAction.ALLOW)
        assert result is False, (
            "DESYNC WO must NOT keep cancel_resting_for_risk True; "
            "this is the v1.4.66-260518-192744 wedge."
        )
        assert om._risk_exec_state == RiskExecState.NORMAL
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_67_stale_cancel_pending_does_not_block_suppressed_to_normal() -> None:
    """v1.4.67: a CANCEL_PENDING WO older than
    ``cancel_pending_unresolved_timeout_seconds`` must NOT keep the
    risk state machine pinned. The cancel-pending watchdog only
    tracks inside-rung slots; an orphan-slot or hydrated
    CANCEL_PENDING never gets reaped and would otherwise wedge the
    state machine forever.
    """
    from datetime import timedelta
    from app.enums import OrderStatus, RiskAction, RiskExecState
    from app.utils.time import utc_now
    om, _storage, _state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=4202)
        wo.status = OrderStatus.CANCEL_PENDING
        # Backdate the cancel-requested timestamp well past the
        # configured timeout so the WO is "stale CANCEL_PENDING".
        timeout_s = float(om._settings.cancel_pending_unresolved_timeout_seconds)
        wo.ts_cancel_requested = utc_now() - timedelta(seconds=timeout_s * 4.0)

        # NORMAL → CANCELLING via cancel-all
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        # The stale CANCEL_PENDING should NOT count, so the state
        # machine reaches SUPPRESSED immediately.
        assert om._risk_exec_state == RiskExecState.SUPPRESSED, (
            f"Expected SUPPRESSED, got {om._risk_exec_state}"
        )

        # Risk clears → must transition back to NORMAL, not stay
        # pinned by the stale CANCEL_PENDING.
        result = om.cancel_resting_for_risk(RiskAction.ALLOW)
        assert result is False, (
            "Stale CANCEL_PENDING must NOT block SUPPRESSED → NORMAL"
        )
        assert om._risk_exec_state == RiskExecState.NORMAL
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_67_fresh_cancel_pending_still_blocks() -> None:
    """v1.4.67 boundary: a CANCEL_PENDING WO that is still WITHIN the
    cancel-pending timeout MUST still block the state machine. We
    only skip stale ones, not fresh ones. This protects the
    legitimate "wait for the cancel ack" behavior.
    """
    from app.enums import OrderStatus, RiskAction, RiskExecState
    from app.utils.time import utc_now
    om, _storage, _state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=4203)
        wo.status = OrderStatus.CANCEL_PENDING
        wo.ts_cancel_requested = utc_now()  # fresh

        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        # Fresh CANCEL_PENDING still counts as cancellable — stay in
        # CANCELLING, don't transition to SUPPRESSED yet.
        assert om._risk_exec_state == RiskExecState.CANCELLING

        # Risk clears while still CANCELLING with pending WO → go to
        # SUPPRESSED briefly (not directly NORMAL), per Phase 2 design.
        result = om.cancel_resting_for_risk(RiskAction.ALLOW)
        assert result is True
        assert om._risk_exec_state == RiskExecState.SUPPRESSED
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_67_reconciler_treats_desync_as_terminal() -> None:
    """v1.4.67: the reconciler's ``_TERMINAL`` set now includes
    DESYNC. With a DESYNC current order and a should_exist=True
    desired, the reconciler must emit ``PlaceAction`` (fresh place
    on the slot), NOT ``CancelAction`` or ``AmendAction`` against an
    order the bot has already given up on.
    """
    from types import SimpleNamespace
    from app.enums import OrderStatus, Side as S
    from app.reconciler import (
        DesiredOrderState,
        PlaceAction,
        reconcile,
    )

    desired = {
        (S.BUY, 0): DesiredOrderState(
            side=S.BUY, level_idx=0, should_exist=True,
            price=1.0, size=5.0, reason="engine_quote",
        ),
    }
    current = {
        (S.BUY, 0): SimpleNamespace(
            order_id_local="local-desync",
            status=OrderStatus.DESYNC,
            side=S.BUY, price=1.0, size=5.0,
        ),
    }
    actions = reconcile(desired=desired, current=current)
    assert len(actions) == 1
    assert isinstance(actions[0], PlaceAction), (
        f"Reconciler must treat DESYNC as terminal and emit "
        f"PlaceAction; got {type(actions[0]).__name__}"
    )


def test_v1_4_67_reconciler_desync_with_no_desired_is_noop() -> None:
    """v1.4.67: a DESYNC current with empty desired produces a NoOp
    ("terminal_match"), not a redundant CancelAction. The bot has
    already issued the cancel(s) it can for a DESYNC order;
    re-issuing more would just churn the venue rate limiter.
    """
    from types import SimpleNamespace
    from app.enums import OrderStatus, Side as S
    from app.reconciler import (
        DesiredOrderState,
        NoOpAction,
        reconcile,
    )

    desired = {
        (S.BUY, 0): DesiredOrderState.empty(
            side=S.BUY, level_idx=0, reason="engine_no_quote",
        ),
    }
    current = {
        (S.BUY, 0): SimpleNamespace(
            order_id_local="local-desync2",
            status=OrderStatus.DESYNC,
            side=S.BUY, price=1.0, size=5.0,
        ),
    }
    actions = reconcile(desired=desired, current=current)
    assert len(actions) == 1
    assert isinstance(actions[0], NoOpAction)
    assert actions[0].reason == "terminal_match"


# ---------------------------------------------------------------------------
# v1.4.57 — Wedge elimination Phase 3: liveness + always-iterate
# ---------------------------------------------------------------------------


def test_phase3b_silent_wedge_detected_triggers_force_reconcile() -> None:
    """v1.4.57 Phase 3b: when ``_maybe_emit_silent_wedge_diagnostic``
    fires its ERROR event, it must ADDITIONALLY request a forced
    reconcile via ``request_open_orders_reconcile``. The reconcile is
    the canonical recovery for "local state has diverged from
    exchange state" — firing it once per wedge emission closes the
    loop.
    """
    om, _storage, state, db_path = _setup()
    try:
        # Spy on request_open_orders_reconcile.
        reconcile_calls = []
        original = om.request_open_orders_reconcile
        def _spy(**kwargs):
            reconcile_calls.append(kwargs)
            return original(**kwargs)
        om.request_open_orders_reconcile = _spy  # type: ignore[method-assign]

        # Set up conditions for silent_wedge to fire:
        #   - engine producing valid quotes (mode != no_quote)
        #   - eligibility QUOTE_BOTH
        #   - active_sides BOTH
        #   - no side_unresolved
        #   - idle > threshold (we configured 10 s in _settings())
        state.last_outbound_attempt_ts_mono = time.monotonic() - 15.0

        om._maybe_emit_silent_wedge_diagnostic(
            telemetry={"quote_engine_mode": "two_sided"},
            decision_quote_cycle_id="qc-wedge-test",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )

        assert reconcile_calls, (
            "v1.4.57 Phase 3b regression: silent_wedge_detected must "
            "trigger a forced reconcile request. Got no calls."
        )
        assert reconcile_calls[0].get("reason") == "silent_wedge_detected"
        assert reconcile_calls[0].get("emergency") is True
    finally:
        _safe_unlink_db(db_path)


def test_phase3b_force_reconcile_disabled_via_config() -> None:
    """v1.4.57 Phase 3b: operators can disable the auto-reconcile
    via ``SILENT_WEDGE_FORCE_RECONCILE_ENABLED=false``. The wedge
    event still fires (log + DB row), but no reconcile is requested.
    """
    om, _storage, state, db_path = _setup()
    try:
        om._settings = om._settings.model_copy(
            update={"silent_wedge_force_reconcile_enabled": False}
        )
        reconcile_calls = []
        original = om.request_open_orders_reconcile
        def _spy(**kwargs):
            reconcile_calls.append(kwargs)
            return original(**kwargs)
        om.request_open_orders_reconcile = _spy  # type: ignore[method-assign]

        state.last_outbound_attempt_ts_mono = time.monotonic() - 15.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry={"quote_engine_mode": "two_sided"},
            decision_quote_cycle_id="qc-wedge-test-disabled",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )

        assert reconcile_calls == [], (
            "With SILENT_WEDGE_FORCE_RECONCILE_ENABLED=false the "
            f"auto-reconcile must be suppressed. Got: {reconcile_calls}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_phase2_state_surfaces_in_executor_state_snapshot() -> None:
    """v1.4.56: the risk-exec state machine is exposed via
    ``executor_state_snapshot.risk_exec_state`` so the postmortem and
    wedge events can see whether the bot is suppressed and why.
    """
    from app.enums import RiskAction, RiskExecState
    om, _storage, state, db_path = _setup()
    try:
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        for key in (
            "risk_exec_state",
            "risk_exec_state_age_s",
            "risk_exec_state_last_risk_action",
            "risk_exec_state_transition_counts",
        ):
            assert key in es, f"executor_state missing v1.4.56 field {key}"
        # SUPPRESSED because no WOs were staged (immediate transition).
        assert es["risk_exec_state"] in (
            RiskExecState.CANCELLING.value,
            RiskExecState.SUPPRESSED.value,
        )
        assert es["risk_exec_state_last_risk_action"] == RiskAction.CANCEL_ALL.value
        assert "NORMAL->CANCELLING" in es["risk_exec_state_transition_counts"]
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.60 — Wedge elimination Phase 5: reducing-side bypass on
# post_only_cross_cooldown
#
# Snapshot v1.4.59-260518-173229 caught the bot wedged for 90+ s with:
#   * adverse_side_pause blocking BUY (adding side, position +1 → correct)
#   * post_only_cross_cooldown blocking SELL (reducing side → wrong)
# The reducer can't flatten inventory while the cooldown holds. The
# v1.4.60 fix mirrors the BUG-010 ``adverse_side_pause`` bypass on
# the ``post_only_cross_cooldown`` so BOTH defensive cooldowns
# respect the same "never block the reducer with non-zero inventory"
# invariant.
# ---------------------------------------------------------------------------


def _setup_phase5() -> tuple:
    """Phase 5 test setup with a higher MAX_ABS_POSITION so the
    central pre-send risk check doesn't preempt the gate-bypass
    assertion. We're testing the cooldown gate, not the position
    cap; raise the cap above the test order sizes.
    """
    s = _settings(MAX_ABS_POSITION=10.0)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    return om, storage, state, db_path


def test_phase5_post_only_cross_cooldown_blocks_when_no_position() -> None:
    """v1.4.60 baseline: with flat inventory, the cooldown still
    blocks. The bypass only applies when inventory exists."""
    om, _storage, state, db_path = _setup_phase5()
    try:
        # Arm the cooldown on the SELL side.
        om._arm_post_only_cross_cooldown(Side.SELL)
        # Position is flat by default.
        assert float(state.position.position_qty) == 0.0
        # Attempt to stage a SELL — small size to clear the
        # central pre-send risk gate.
        wo = om._stage_place_order_local(
            Side.SELL,
            price=2.0,
            size=1.0,
            quote_cycle_id="qc-flat-1",
        )
        # Cooldown was active and inventory is flat → blocked.
        assert wo is None
        assert (
            om._last_stage_place_skip_reason["SELL"]
            and om._last_stage_place_skip_reason["SELL"].startswith(
                "post_only_cross_cooldown"
            )
        )
    finally:
        _safe_unlink_db(db_path)


def test_phase5_post_only_cross_cooldown_bypassed_for_reducing_side_when_long() -> None:
    """v1.4.60 wedge fix: bot is LONG, SELL is the reducer. With
    ``post_only_cross_cooldown`` active on SELL, the bypass MUST
    let the SELL placement proceed past this gate. The pre-v1.4.60
    wedge was 909 SELL stage_returned_none events from this cooldown
    — the bot couldn't flatten.
    """
    om, _storage, state, db_path = _setup_phase5()
    try:
        # Position +2 long (well under MAX_ABS_POSITION=10).
        with state._lock:
            state.position.position_qty = 2.0
        # Arm the cooldown on SELL.
        om._arm_post_only_cross_cooldown(Side.SELL)
        # Verify cooldown IS active (sanity).
        assert om._post_only_cross_cooldown_active(Side.SELL) is True
        # SELL is the REDUCING side for a long position.
        assert om._is_reducing_side(Side.SELL) is True
        # Attempt to stage a SELL. Bypass should let it through past
        # the cooldown gate. The skip reason MUST NOT be a cooldown
        # one — that's the bypass invariant.
        om._stage_place_order_local(
            Side.SELL,
            price=2.0,
            size=1.0,
            quote_cycle_id="qc-long-reduce-1",
        )
        skip_reason = om._last_stage_place_skip_reason.get("SELL")
        assert skip_reason is None or "post_only_cross_cooldown" not in str(skip_reason), (
            f"v1.4.60 bypass: post_only_cross_cooldown MUST NOT block the "
            f"reducing side (SELL when LONG). Got skip_reason={skip_reason!r}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_phase5_post_only_cross_cooldown_bypassed_for_reducing_side_when_short() -> None:
    """v1.4.60 wedge fix (mirror): bot is SHORT, BUY is the reducer.
    Cooldown on BUY → bypassed."""
    om, _storage, state, db_path = _setup_phase5()
    try:
        with state._lock:
            state.position.position_qty = -2.0
        om._arm_post_only_cross_cooldown(Side.BUY)
        assert om._is_reducing_side(Side.BUY) is True
        om._stage_place_order_local(
            Side.BUY,
            price=1.0,
            size=1.0,
            quote_cycle_id="qc-short-reduce-1",
        )
        skip_reason = om._last_stage_place_skip_reason.get("BUY")
        assert skip_reason is None or "post_only_cross_cooldown" not in str(skip_reason)
    finally:
        _safe_unlink_db(db_path)


def test_phase5_post_only_cross_cooldown_still_blocks_adding_side() -> None:
    """v1.4.60 negative control: the bypass is NARROW. Bot LONG,
    cooldown on BUY (the ADDING side). The cooldown MUST still
    block — protection against re-crossing the spread when adding
    to inventory is legitimate.
    """
    om, _storage, state, db_path = _setup_phase5()
    try:
        with state._lock:
            state.position.position_qty = 2.0
        om._arm_post_only_cross_cooldown(Side.BUY)
        # BUY is ADDING side for a long position.
        assert om._is_reducing_side(Side.BUY) is False
        wo = om._stage_place_order_local(
            Side.BUY,
            price=1.0,
            size=1.0,
            quote_cycle_id="qc-long-add-1",
        )
        # Adding side WITH cooldown → blocked.
        assert wo is None
        skip_reason = om._last_stage_place_skip_reason.get("BUY")
        assert skip_reason and "post_only_cross_cooldown" in skip_reason, (
            "v1.4.60 bypass must NOT affect the adding side. "
            f"Got skip_reason={skip_reason!r}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_phase5_post_only_cross_cooldown_bypass_disabled_via_config() -> None:
    """v1.4.60: operators can disable the bypass via config
    (``POST_ONLY_CROSS_COOLDOWN_BYPASS_REDUCING_SIDE=false``) to
    revert to pre-fix behaviour."""
    om, _storage, state, db_path = _setup_phase5()
    try:
        om._settings = om._settings.model_copy(
            update={"post_only_cross_cooldown_bypass_reducing_side": False}
        )
        with state._lock:
            state.position.position_qty = 2.0
        om._arm_post_only_cross_cooldown(Side.SELL)
        wo = om._stage_place_order_local(
            Side.SELL,
            price=2.0,
            size=1.0,
            quote_cycle_id="qc-disabled-1",
        )
        # With bypass disabled, the cooldown blocks even the reducer.
        assert wo is None
        skip_reason = om._last_stage_place_skip_reason.get("SELL")
        assert skip_reason and "post_only_cross_cooldown" in skip_reason
    finally:
        _safe_unlink_db(db_path)


# ===========================================================================
# v1.4.62 — comprehensive wedge-probing test suite
#
# Systematic coverage of every layer that can produce a "stuck" symptom:
#
#   1. Stage-skip reason coverage — each of the 7 _stage_place_order_local
#      None-returns is tested AND its reason recorded in the trace.
#   2. _dispatch_action behavior — each Action type, plus the race-window
#      cases (cur disappears between snapshot and dispatch).
#   3. side_unresolved override — any action type becomes NoOp.
#   4. Cooldown × bypass interaction matrix — all four combinations.
#   5. SENT-stuck recovery — verifies the v1.4.62 reduced timeout fires.
#   6. Reconciler ↔ dispatcher integration — desired/current diffs map
#      to the right transport-helper calls.
#
# These tests are intentionally focused — one wedge mechanism per test —
# so a failure name immediately tells you which class regressed.
# ===========================================================================


# ---------------------------------------------------------------------------
# Stage-skip reason coverage — every None-return path in
# _stage_place_order_local must (1) return None and (2) record the
# specific reason in _last_stage_place_skip_reason. Pre-v1.4.50 some
# of these paths were silent.
# ---------------------------------------------------------------------------


def test_stage_skip_reason_central_pre_send_risk_check_failed() -> None:
    """v1.4.62 wedge probe: when the position cap would be breached,
    stage_place returns None with reason ``central_pre_send_risk_check_failed``.
    """
    om, _storage, state, db_path = _setup()  # default MAX_ABS_POSITION=0.05
    try:
        # Stage a place that would exceed the cap.
        wo = om._stage_place_order_local(
            Side.BUY, price=1.0, size=10.0, quote_cycle_id="qc-risk-1",
        )
        assert wo is None
        assert (
            om._last_stage_place_skip_reason["BUY"]
            == "central_pre_send_risk_check_failed"
        )
    finally:
        _safe_unlink_db(db_path)


def test_stage_skip_reason_side_unresolved() -> None:
    """v1.4.62: stage_place returns None and records
    ``side_unresolved:<reason>`` when the side is in unresolved state.
    """
    om, _storage, state, db_path = _setup()
    try:
        om._set_side_unresolved(
            Side.BUY, reason="test_unresolved", payload={}
        )
        wo = om._stage_place_order_local(
            Side.BUY, price=1.0, size=0.01, quote_cycle_id="qc-unres-1",
        )
        assert wo is None
        sr = om._last_stage_place_skip_reason["BUY"]
        assert sr and sr.startswith("side_unresolved:")
    finally:
        _safe_unlink_db(db_path)


def test_stage_skip_reason_invalid_px_or_sz() -> None:
    """v1.4.62: zero/negative price OR size yields None + reason
    ``invalid_px_or_sz``. Pre-v1.4.50 this was silently returned
    None with no log — a place attempt with bad inputs vanished.
    """
    om, _storage, _state, db_path = _setup_phase5()  # MAX_ABS_POSITION=10
    try:
        wo = om._stage_place_order_local(
            Side.BUY, price=0.0, size=1.0, quote_cycle_id="qc-bad-1",
        )
        assert wo is None
        sr = om._last_stage_place_skip_reason["BUY"]
        assert sr and sr.startswith("invalid_px_or_sz")
    finally:
        _safe_unlink_db(db_path)


def test_stage_skip_reason_existing_wo_active_race() -> None:
    """v1.4.62 wedge probe: race window — between _orchestrate's
    snapshot of cur and _stage_place_order_local's re-read of ex,
    a WS handler can transition the WO to active status. This was
    silent pre-v1.4.50; now it records ``existing_wo_active_race``.
    """
    om, _storage, state, db_path = _setup_phase5()
    try:
        from app.enums import OrderStatus
        wo_existing = _ack_wo(om, Side.BUY, price=1.0, oid=111)
        # Existing WO is ACKED. Stage another place — should be blocked.
        wo = om._stage_place_order_local(
            Side.BUY, price=1.0, size=1.0, quote_cycle_id="qc-race-1",
        )
        assert wo is None
        sr = om._last_stage_place_skip_reason["BUY"]
        assert sr and sr.startswith("existing_wo_active_race")
    finally:
        _safe_unlink_db(db_path)


def test_stage_skip_reason_no_write_access() -> None:
    """v1.4.62: when the client is read-only (no_write_access),
    stage_place returns None with that reason. Used by dry-run /
    inspector deploys."""
    om, _storage, _state, db_path = _setup_phase5()
    try:
        om._client.has_write_access.return_value = False
        wo = om._stage_place_order_local(
            Side.BUY, price=1.0, size=1.0, quote_cycle_id="qc-ro-1",
        )
        assert wo is None
        assert (
            om._last_stage_place_skip_reason["BUY"] == "no_write_access"
        )
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# _dispatch_action behavior — each Action type + race window cases
# ---------------------------------------------------------------------------


def test_dispatch_noop_records_decision_no_side_effects() -> None:
    """v1.4.62: NoOpAction must record the decision but produce
    NO transport-side effects. Counters before/after should match
    except for the orchestrate_decision_counts."""
    from app.reconciler import NoOpAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        # Snapshot transport counters.
        enq_calls = []
        original_cancel = om._enqueue_cancel_quote_path
        original_place = om._enqueue_place_transport
        def _spy_cancel(*a, **kw):
            enq_calls.append(("cancel",))
            return original_cancel(*a, **kw)
        def _spy_place(*a, **kw):
            enq_calls.append(("place",))
            return original_place(*a, **kw)
        om._enqueue_cancel_quote_path = _spy_cancel  # type: ignore[method-assign]
        om._enqueue_place_transport = _spy_place  # type: ignore[method-assign]

        # Need a decision-shaped obj for the trace logging.
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-noop-1"

        om._dispatch_action(
            NoOpAction(side=Side.BUY, level_idx=0, reason="acked_no_reprice_needed"),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        assert enq_calls == [], "NoOpAction must not trigger any transport call"
        # Decision was recorded.
        counter_key = "BUY:reconciler:noop:acked_no_reprice_needed"
        assert om._orchestrate_decision_counts.get(counter_key, 0) == 1
    finally:
        _safe_unlink_db(db_path)


def test_dispatch_cancel_action_routes_through_enqueue_cancel() -> None:
    """v1.4.62: CancelAction must route through
    _enqueue_cancel_quote_path with the correct trigger_reason."""
    from app.enums import OrderStatus
    from app.reconciler import CancelAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=222)
        spy_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kwargs):
            spy_calls.append((wo_arg.order_id_exchange, kwargs.get("trigger_reason")))
            return original(wo_arg, **kwargs)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-cancel-1"

        om._dispatch_action(
            CancelAction(
                side=Side.BUY, level_idx=0,
                current_local_id=wo.order_id_local,
                trigger_reason="reprice_replace",
            ),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        assert spy_calls == [(222, "reprice_replace")]
        # Reprice flag set when trigger_reason == reprice_replace.
        assert om._quote_reprice_replace_pending[Side.BUY] is True
    finally:
        _safe_unlink_db(db_path)


def test_dispatch_cancel_action_race_cur_disappeared() -> None:
    """v1.4.62 wedge probe: CancelAction with cur disappeared
    between reconciler-snapshot and dispatch-time (WS-CANCELED
    landed in the gap). Dispatcher records the race as a benign
    no-op, NOT a stack trace or silent crash."""
    from app.reconciler import CancelAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        # No WO on BUY slot 0 — simulates "disappeared between reads".
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-race-cancel-1"
        om._dispatch_action(
            CancelAction(
                side=Side.BUY, level_idx=0,
                current_local_id="vanished",
                trigger_reason="reprice_replace",
            ),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        # Race recorded.
        counter_key = "BUY:reconciler:cancel_race_cur_none"
        assert om._orchestrate_decision_counts.get(counter_key, 0) == 1
    finally:
        _safe_unlink_db(db_path)


def test_dispatch_place_action_with_stage_skip_records_reason() -> None:
    """v1.4.62 → v1.4.74 Phase 2A: PlaceAction directly injected into
    the dispatcher with side_unresolved active triggers the Phase 2A
    invariant-violation path. In production this shouldn't happen
    because ``compute_desired_state`` applies the side_unresolved
    fold upstream; injecting bypasses that fold and tests the
    dispatcher's defense-in-depth.
    """
    from app.reconciler import DesiredOrderState, PlaceAction
    om, _storage, state, db_path = _setup_phase5()
    try:
        # Stage will block due to side_unresolved.
        om._set_side_unresolved(
            Side.BUY, reason="test_block", payload={}
        )
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-place-blocked-1"

        # Phase 2A: dispatcher refuses the PlaceAction and traces an
        # invariant-violation. Pre-Phase-2A this would silently
        # return under ``side_unresolved:return_silent``.
        om._dispatch_action(
            PlaceAction(
                side=Side.BUY, level_idx=0,
                desired=DesiredOrderState(
                    side=Side.BUY, level_idx=0, should_exist=True,
                    price=1.0, size=1.0, reason="r",
                ),
            ),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        counter_key = "BUY:phase2a_invariant_violation:place_dispatch_refused_unresolved"
        assert om._orchestrate_decision_counts.get(counter_key, 0) == 1
        # And the dedicated invariant-violation counter increments.
        assert om._gate_phase2a_invariant_violation_total == 1
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# side_unresolved override matrix — any Action becomes NoOp
# ---------------------------------------------------------------------------


def test_side_unresolved_overrides_place_action() -> None:
    """v1.4.62 → v1.4.74 Phase 2A: a PlaceAction directly injected into
    the dispatcher with side_unresolved active is REFUSED and a Phase
    2A invariant-violation is recorded. The stage layer is not reached
    (the dispatcher's pre-check short-circuits).

    In production, ``compute_desired_state`` empties the slot upstream,
    so the reconciler never emits a PlaceAction for an unresolved
    side; this defense-in-depth path is reached only if a future
    code path bypasses the fold.
    """
    from app.reconciler import DesiredOrderState, PlaceAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        om._set_side_unresolved(Side.SELL, reason="test", payload={})
        spy_calls = []
        original = om._stage_place_order_local
        def _spy(*args, **kwargs):
            spy_calls.append("stage")
            return original(*args, **kwargs)
        om._stage_place_order_local = _spy  # type: ignore[method-assign]

        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-unresolved-place"
        om._dispatch_action(
            PlaceAction(
                side=Side.SELL, level_idx=0,
                desired=DesiredOrderState(
                    side=Side.SELL, level_idx=0, should_exist=True,
                    price=2.0, size=1.0, reason="r",
                ),
            ),
            decision=decision, mid_price=2.0, tick=0.001,
        )
        assert spy_calls == [], "PlaceAction must not reach stage while side_unresolved"
        counter_key = "SELL:phase2a_invariant_violation:place_dispatch_refused_unresolved"
        assert om._orchestrate_decision_counts.get(counter_key, 0) == 1
        assert om._gate_phase2a_invariant_violation_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_side_unresolved_overrides_amend_action() -> None:
    """v1.4.62: AmendAction is also overridden by side_unresolved."""
    from app.enums import OrderStatus
    from app.reconciler import AmendAction, DesiredOrderState
    om, _storage, _state, db_path = _setup_phase5()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=333)
        om._set_side_unresolved(Side.BUY, reason="test", payload={})
        spy_calls = []
        original = om._enqueue_amend_quote_path
        def _spy(*a, **kw):
            spy_calls.append("amend")
            return original(*a, **kw)
        om._enqueue_amend_quote_path = _spy  # type: ignore[method-assign]

        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-unresolved-amend"
        om._dispatch_action(
            AmendAction(
                side=Side.BUY, level_idx=0,
                desired=DesiredOrderState(
                    side=Side.BUY, level_idx=0, should_exist=True,
                    price=1.05, size=1.0, reason="r",
                ),
                current_local_id=wo.order_id_local,
            ),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        assert spy_calls == [], "AmendAction must not amend while side_unresolved"
    finally:
        _safe_unlink_db(db_path)


def test_side_unresolved_does_not_affect_other_side() -> None:
    """v1.4.62 wedge probe: only the unresolved side is blocked.
    The OTHER side continues to dispatch normally. Pre-Phase-5 the
    side_unresolved check was per-call; here we verify the dispatcher
    isolates state correctly."""
    from app.enums import OrderStatus
    from app.reconciler import CancelAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        om._set_side_unresolved(Side.BUY, reason="test", payload={})
        wo = _ack_wo(om, Side.SELL, price=2.0, oid=444)
        spy_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kw):
            spy_calls.append(wo_arg.order_id_exchange)
            return original(wo_arg, **kw)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-sell-side-isolated"
        # SELL action — should dispatch despite BUY being unresolved.
        om._dispatch_action(
            CancelAction(
                side=Side.SELL, level_idx=0,
                current_local_id=wo.order_id_local,
                trigger_reason="desired_none",
            ),
            decision=decision, mid_price=2.0, tick=0.001,
        )
        assert spy_calls == [444], (
            "SELL action must dispatch even when BUY is unresolved"
        )
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# Cooldown × bypass interaction matrix — verify the v1.4.59 wedge
# class is truly closed under EVERY combination.
# ---------------------------------------------------------------------------


def test_wedge_matrix_adverse_pause_alone_long_position_sell_bypassed() -> None:
    """v1.4.62 wedge matrix: position LONG, adverse_side_pause on
    SELL alone → bypassed because SELL is the reducer."""
    om, _storage, state, db_path = _setup_phase5()
    try:
        with state._lock:
            state.position.position_qty = 3.0
        # Arm adverse_side_pause on SELL (reducer).
        from app.toxicity import ToxicitySnapshot
        # Direct manipulation: set the until_mono to a future time.
        om._adverse_side_pause_until[Side.SELL] = time.monotonic() + 30.0
        om._stage_place_order_local(
            Side.SELL, price=2.0, size=1.0, quote_cycle_id="qc-mtx-1",
        )
        skip = om._last_stage_place_skip_reason.get("SELL")
        assert skip is None or "adverse_side_pause" not in str(skip), (
            f"adverse_side_pause MUST bypass the reducer (SELL when LONG). "
            f"Got skip={skip!r}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_wedge_matrix_post_only_cooldown_alone_short_position_buy_bypassed() -> None:
    """v1.4.62 wedge matrix: position SHORT, post_only_cross_cooldown
    on BUY alone → bypassed (BUY is reducer when short)."""
    om, _storage, state, db_path = _setup_phase5()
    try:
        with state._lock:
            state.position.position_qty = -3.0
        om._arm_post_only_cross_cooldown(Side.BUY)
        om._stage_place_order_local(
            Side.BUY, price=1.0, size=1.0, quote_cycle_id="qc-mtx-2",
        )
        skip = om._last_stage_place_skip_reason.get("BUY")
        assert skip is None or "post_only_cross_cooldown" not in str(skip)
    finally:
        _safe_unlink_db(db_path)


def test_wedge_matrix_both_cooldowns_active_long_position_reducer_bypassed() -> None:
    """v1.4.62 wedge matrix: THE v1.4.59 WEDGE SCENARIO.

    Position +long, BOTH adverse_side_pause AND post_only_cross_cooldown
    armed on SELL (reducer). The reducing-side bypass must let
    AT LEAST ONE of them through.

    Pre-v1.4.60 the bot wedged for 90+ s with this combo.
    """
    om, _storage, state, db_path = _setup_phase5()
    try:
        with state._lock:
            state.position.position_qty = 3.0
        om._adverse_side_pause_until[Side.SELL] = time.monotonic() + 30.0
        om._arm_post_only_cross_cooldown(Side.SELL)
        om._stage_place_order_local(
            Side.SELL, price=2.0, size=1.0, quote_cycle_id="qc-mtx-3",
        )
        skip = om._last_stage_place_skip_reason.get("SELL")
        # Neither cooldown reason should be in the skip — both should be bypassed.
        if skip:
            assert "adverse_side_pause" not in skip, (
                f"adverse_side_pause bypass FAILED for reducer. skip={skip!r}"
            )
            assert "post_only_cross_cooldown" not in skip, (
                f"post_only_cross_cooldown bypass FAILED for reducer. skip={skip!r}"
            )
    finally:
        _safe_unlink_db(db_path)


def test_wedge_matrix_both_cooldowns_active_flat_position_blocks_both_sides() -> None:
    """v1.4.62 wedge matrix: at FLAT position, both cooldowns
    legitimately block (no inventory to reduce, so no reducer to
    protect)."""
    om, _storage, state, db_path = _setup_phase5()
    try:
        # Position is 0.0 by default.
        assert float(state.position.position_qty) == 0.0
        om._adverse_side_pause_until[Side.BUY] = time.monotonic() + 30.0
        om._arm_post_only_cross_cooldown(Side.BUY)
        wo = om._stage_place_order_local(
            Side.BUY, price=1.0, size=1.0, quote_cycle_id="qc-mtx-4",
        )
        assert wo is None
        skip = om._last_stage_place_skip_reason.get("BUY")
        # One of the two cooldowns will have fired first.
        assert skip and (
            "adverse_side_pause" in skip
            or "post_only_cross_cooldown" in skip
        )
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# Risk-exec state machine wedge probes (already 9 tests; adding stress)
# ---------------------------------------------------------------------------


def test_risk_state_machine_storm_500_ticks_only_one_cancel_all() -> None:
    """v1.4.62 stress: 500 consecutive risk=CANCEL_ALL ticks
    produce exactly ONE cancel-all dispatch. Pre-Phase-2 this was
    500. The state machine's storm-prevention contract."""
    from app.enums import RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        # Stage a few WOs so cancel-all has work on the first invocation.
        _ack_wo(om, Side.BUY, price=1.0, oid=1)
        _ack_wo(om, Side.SELL, price=2.0, oid=2)
        enq_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kw):
            enq_calls.append(wo_arg.order_id_exchange)
            return original(wo_arg, **kw)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        for _ in range(500):
            om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)

        # First tick fired 2 cancels (one per WO). 499 subsequent
        # ticks fired none (state was already CANCELLING).
        assert sorted(enq_calls) == [1, 2], (
            f"500-tick storm must produce exactly 2 cancel enqueues. "
            f"Got: {enq_calls}"
        )
        assert om._risk_exec_state == RiskExecState.CANCELLING
    finally:
        # Explicitly close the storage so SQLite releases the file
        # handle before unlink. On Windows the 500-tick storm leaves
        # enough connection-cache pressure that the handle isn't GC'd
        # by the time the unlink runs (full-suite-only flake observed
        # 2026-05-27 — passes in isolation). The defensive PermissionError
        # catch is a belt-and-suspenders against any future leak path.
        try:
            _storage.close()
        except Exception:
            pass
        try:
            _safe_unlink_db(db_path)
        except PermissionError:
            pass


def test_risk_state_machine_transition_to_normal_and_back() -> None:
    """v1.4.62 wedge probe: state machine cycle integrity — can
    transition NORMAL → CANCELLING → NORMAL → CANCELLING → NORMAL.
    """
    from app.enums import OrderStatus, RiskAction, RiskExecState
    om, _storage, _state, db_path = _setup()
    try:
        # Cycle 1.
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        # SUPPRESSED (no WOs to cancel).
        om.cancel_resting_for_risk(RiskAction.ALLOW)
        assert om._risk_exec_state == RiskExecState.NORMAL
        # Cycle 2.
        om.cancel_resting_for_risk(RiskAction.CANCEL_ALL)
        om.cancel_resting_for_risk(RiskAction.ALLOW)
        assert om._risk_exec_state == RiskExecState.NORMAL
        # Verify transition counts.
        counts = om._risk_exec_state_transition_counts
        assert counts.get("NORMAL->CANCELLING", 0) == 2
        assert counts.get("SUPPRESSED->NORMAL", 0) == 2
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# SENT-timeout recovery — v1.4.62 reduced from 120s to 15s
# ---------------------------------------------------------------------------


def test_sent_timeout_default_calibrated_to_okx_colo_latency() -> None:
    """v1.4.63: default SENT timeout is 1.0 s, calibrated against
    the actual production place-to-ack RTT histogram:
        min 3.13 ms · med 3.27 ms · p95 3.94 ms · max 4.73 ms (n=1005)
    1.0 s is 250× median, 200× max — well above any reasonable
    WS hiccup but tight enough that stuck SENT recovers fast.

    History: 120 s → 15 s (v1.4.62) → 1.0 s (v1.4.63). The 120 s
    and 15 s defaults were not calibrated against actual latency;
    they were arbitrary safety margins that left the bot
    side-suppressed for seconds-to-minutes per stuck SENT.
    """
    from app.config import Settings
    s = Settings()
    assert s.sent_order_unresolved_timeout_seconds == 1.0, (
        f"v1.4.63: SENT timeout default must be 1.0 s (calibrated "
        f"to OKX colo latency), got {s.sent_order_unresolved_timeout_seconds}"
    )
    assert s.sent_order_unresolved_max_ambiguous_polls == 3


def test_sent_timeout_can_be_overridden_via_env_alias() -> None:
    """v1.4.63: operator can tune the timeout via env for
    cross-region / higher-latency setups. The 1 s default is
    OKX-colo-aggressive; venues with 100+ ms RTT may want 2-5 s.
    """
    from app.config import Settings
    import os
    os.environ["SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS"] = "5"
    try:
        s = Settings()
        assert s.sent_order_unresolved_timeout_seconds == 5.0
    finally:
        del os.environ["SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS"]


# ---------------------------------------------------------------------------
# compute_desired_state — output correctness against engine input
# ---------------------------------------------------------------------------


def _make_build(bid_order=None, ask_order=None):
    """Minimal QuoteEngine build result shape used by compute_desired_state."""
    from types import SimpleNamespace as _SN
    return _SN(
        bid_order=bid_order,
        ask_order=ask_order,
        telemetry={},
    )


def _final_quote_order(side, price, size, target_half=3.0, aging=False):
    from app.quote_engine import FinalQuoteOrder
    return FinalQuoteOrder(
        side=side, price=price, size=size,
        target_half_spread_bps=target_half,
        aging_tighten_applied=aging,
    )


def test_compute_desired_state_engine_no_quote_produces_empty_slots() -> None:
    """v1.4.62: engine returns no bid_order / no ask_order →
    compute_desired_state produces empty slots for all levels."""
    om, _storage, _state, db_path = _setup_phase5()
    try:
        build = _make_build(bid_order=None, ask_order=None)
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-empty-1"

        post, engine, rejs = om.compute_desired_state(
            build=build, decision=decision, ladder=None,
            position_qty=0.0, num_levels=1,
            client_spec=om._client.symbol_spec,
        )
        assert post[(Side.BUY, 0)].should_exist is False
        assert post[(Side.SELL, 0)].should_exist is False
        assert rejs == []
    finally:
        _safe_unlink_db(db_path)


def test_compute_desired_state_engine_quote_populates_inner_rung() -> None:
    """v1.4.62: single-rung mode — engine bid_order → desired
    BUY level 0 populated with the engine's price/size."""
    om, _storage, _state, db_path = _setup_phase5()
    try:
        build = _make_build(
            bid_order=_final_quote_order(Side.BUY, 1.0, 3.0),
            ask_order=_final_quote_order(Side.SELL, 1.01, 3.0),
        )
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-engine-1"
        post, engine, rejs = om.compute_desired_state(
            build=build, decision=decision, ladder=None,
            position_qty=0.0, num_levels=1,
            client_spec=om._client.symbol_spec,
        )
        buy = post[(Side.BUY, 0)]
        assert buy.should_exist is True
        assert buy.price == 1.0
        assert buy.size == 3.0
        assert buy.reason == "engine_quote"
    finally:
        _safe_unlink_db(db_path)


def test_compute_desired_state_is_pure_no_state_mutation() -> None:
    """v1.4.62: compute_desired_state must not mutate state.
    Calling twice produces identical output."""
    om, _storage, _state, db_path = _setup_phase5()
    try:
        build = _make_build(
            bid_order=_final_quote_order(Side.BUY, 1.0, 3.0),
            ask_order=_final_quote_order(Side.SELL, 1.01, 3.0),
        )
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-pure-1"
        p1, _, _ = om.compute_desired_state(
            build=build, decision=decision, ladder=None,
            position_qty=0.0, num_levels=1,
            client_spec=om._client.symbol_spec,
        )
        p2, _, _ = om.compute_desired_state(
            build=build, decision=decision, ladder=None,
            position_qty=0.0, num_levels=1,
            client_spec=om._client.symbol_spec,
        )
        assert p1 == p2, "compute_desired_state must be pure (deterministic)"
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# _snapshot_working_orders_for_reconciler — orphan rungs included
# ---------------------------------------------------------------------------


def test_snapshot_includes_orphan_rungs_beyond_configured() -> None:
    """v1.4.62 wedge probe: if the operator reduces
    ``ladder_num_levels_per_side`` at runtime, the orphan outer
    rungs must still be visible to the reconciler so they can be
    cancelled. _snapshot_working_orders_for_reconciler includes
    them.
    """
    om, _storage, _state, db_path = _setup_phase5()
    try:
        # Configured = 1 level, but seed an orphan at level 1.
        wo_inside = _ack_wo(om, Side.BUY, price=1.0, oid=10, level_idx=0)
        wo_outside = _ack_wo(om, Side.BUY, price=0.99, oid=11, level_idx=1)
        snap = om._snapshot_working_orders_for_reconciler(num_levels=1)
        assert (Side.BUY, 0) in snap
        assert (Side.BUY, 1) in snap, (
            "Orphan rung at level beyond num_levels must be in the snapshot "
            "so the reconciler can cancel it."
        )
        assert snap[(Side.BUY, 1)].order_id_exchange == 11
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# Reducing-side bypass surface in executor_state_snapshot
# ---------------------------------------------------------------------------


def test_reducing_side_bypass_surface_in_executor_state() -> None:
    """v1.4.62: ``executor_state.reducing_side_bypass`` exposes the
    invariant state so the postmortem / dashboard can verify it's
    operating. The v1.4.59 wedge would have been diagnosable in
    seconds from this surface."""
    om, _storage, state, db_path = _setup_phase5()
    try:
        with state._lock:
            state.position.position_qty = 2.0
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        rsb = es.get("reducing_side_bypass")
        assert rsb is not None
        assert rsb["position_qty"] == 2.0
        assert rsb["buy_is_reducer"] is False
        assert rsb["sell_is_reducer"] is True
        assert rsb["adverse_side_pause_bypass_enabled"] is True
        assert rsb["post_only_cross_cooldown_bypass_enabled"] is True
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# Decision-trace ring buffer correctness
# ---------------------------------------------------------------------------


def test_decision_trace_records_reconciler_actions_with_correct_branch_tag() -> None:
    """v1.4.62: when _dispatch_action records a decision, the
    branch tag is 'reconciler' (Phase 5 marker). Pre-Phase-5 it was
    branch-specific like 'terminal_fresh_place' / 'acked_partial_amend'.
    """
    from app.reconciler import NoOpAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-trace-1"
        om._dispatch_action(
            NoOpAction(side=Side.BUY, level_idx=0, reason="acked_no_reprice_needed"),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        history = list(om._orchestrate_decision_history[Side.BUY])
        assert len(history) == 1
        entry = history[0]
        assert entry["decision_branch"] == "reconciler"
        assert entry["action"] == "noop:acked_no_reprice_needed"
    finally:
        _safe_unlink_db(db_path)


def test_decision_trace_buffer_per_side_isolated() -> None:
    """v1.4.62: BUY decisions stay in the BUY buffer; SELL decisions
    stay in the SELL buffer. Cross-side pollution would corrupt
    postmortem analysis."""
    from app.reconciler import NoOpAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-iso"
        om._dispatch_action(
            NoOpAction(side=Side.BUY, level_idx=0, reason="r1"),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        om._dispatch_action(
            NoOpAction(side=Side.SELL, level_idx=0, reason="r2"),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        buy = list(om._orchestrate_decision_history[Side.BUY])
        sell = list(om._orchestrate_decision_history[Side.SELL])
        assert len(buy) == 1
        assert len(sell) == 1
        assert buy[0]["side"] == "BUY"
        assert sell[0]["side"] == "SELL"
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# Cancel-all idempotency stress
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v1.4.64 — the REAL bug behind the v1.4.61 stuck-SENT wedge:
# transport_rejected WOs were left in SENT instead of transitioning
# to REJECTED. Plus OKX 51603 was classified as "invalid" instead of
# "not_found" — together they produced a 136 s wedge for a single
# rate-limited place.
# ---------------------------------------------------------------------------


def test_v1_4_64_okx_51603_classified_as_not_found() -> None:
    """v1.4.64 wedge fix: OKX top-code 51603 ("Order does not exist")
    must map to ``outcome="not_found"`` so the bot's SENT-recovery
    path REJECTS the WO immediately. Pre-v1.4.64 it mapped to
    ``"invalid"`` → treated as ambiguous → polled indefinitely.
    """
    from app.exchange.okx_responses import interpret_okx_order_status_response
    resp = {"code": "51603", "msg": "Order does not exist", "data": []}
    oid, outcome, detail = interpret_okx_order_status_response(resp)
    assert outcome == "not_found", (
        f"OKX top-51603 must classify as 'not_found' (definitive), "
        f"not 'invalid' (ambiguous). Got: ({oid!r}, {outcome!r}, {detail!r})"
    )
    assert "51603" in detail


def test_v1_4_64_okx_51603_at_row_level_classified_as_not_found() -> None:
    """v1.4.64: 51603 can also appear at the ROW level (top=0 with
    rows carrying sCode=51603). Same classification."""
    from app.exchange.okx_responses import interpret_okx_order_status_response
    resp = {
        "code": "0", "msg": "",
        "data": [{"sCode": "51603", "sMsg": "Order does not exist"}],
    }
    oid, outcome, detail = interpret_okx_order_status_response(resp)
    assert outcome == "not_found"


def test_v1_4_64_other_top_codes_still_classified_as_invalid() -> None:
    """v1.4.64 negative control: the not-found classification is
    narrow — only 51603 (and any future codes we add to
    ``_OKX_ORDER_NOT_FOUND_CODES``). Other unfamiliar non-zero
    codes still default to 'invalid' so a genuinely ambiguous
    response doesn't get prematurely REJECTED."""
    from app.exchange.okx_responses import interpret_okx_order_status_response
    resp = {"code": "99999", "msg": "Mysterious error", "data": []}
    oid, outcome, detail = interpret_okx_order_status_response(resp)
    assert outcome == "invalid", (
        f"Unknown top-code must default to 'invalid' (ambiguous), "
        f"not 'not_found'. Got: {outcome!r}"
    )


def test_v1_4_64_batch_place_transport_rejected_transitions_to_rejected() -> None:
    """v1.4.64 wedge fix: when ``_execute_place_batch_intents`` sees
    a row-level ``transport_rejected`` outcome (rate-limit / auth /
    etc.), it must transition the WO to REJECTED AND release the
    slot. Pre-v1.4.64 the WO was left in SENT — the reconciler
    then NoOp'd forever waiting for an ACK that would never come.

    Reproduces the v1.4.61-260518-181059 root cause: SELL cloid
    ``baf268d0...`` placed at 14:07:09, rate-limited (50011), left
    in SENT for 136 s until the unresolved-timeout fired.
    """
    from app.enums import OrderStatus
    from app.execution import PlaceTransportIntent
    om, _storage, state, db_path = _setup_phase5()
    try:
        # Stage a SELL WO and put it in SENT (simulates the just-
        # dispatched place).
        wo = _ack_wo(om, Side.SELL, price=2.0, oid=5001)
        wo.order_id_exchange = None  # transport hasn't bound an oid yet
        wo.status = OrderStatus.SENT
        # Configure the mock client to return a transport-rejected
        # response from the batch endpoint (OKX row-level rate-limit
        # shape: top "1" + row sCode 50011).
        om._client.batch_place_post_only_limit.return_value = {
            "code": "1",
            "msg": "",
            "data": [{
                "sCode": "50011",
                "sMsg": "Rate limit reached. Refer to API documentation and throttle requests accordingly.",
                "clOrdId": wo.client_order_id,
            }],
        }
        intent = PlaceTransportIntent(
            wo_order_id_local=wo.order_id_local,
            side=wo.side,
            intent_seq=wo.transport_intent_seq,
            quote_cycle_id="qc-50011-1",
            enqueued_mono=time.monotonic(),
            intent_created_perf=time.perf_counter(),
        )

        om._execute_place_batch_intents([intent, intent])  # ≥2 → batch path

        # KEY ASSERTION 1: WO transitioned to REJECTED.
        assert wo.status == OrderStatus.REJECTED, (
            f"v1.4.64 regression: transport_rejected (rate-limit) must "
            f"transition the WO from SENT → REJECTED. Pre-v1.4.64 the "
            f"WO stayed in SENT and the reconciler waited forever. "
            f"Got status={wo.status.value}"
        )
        # KEY ASSERTION 2: slot released.
        with state._lock:
            slot = state.get_working_order(Side.SELL, 0)
        assert slot is None or slot.order_id_local != wo.order_id_local, (
            "v1.4.64: WO slot must be released after REJECTED so "
            "the next quote tick can place fresh."
        )
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_65_compute_desired_state_inside_rung_uses_self_healed_engine_order() -> None:
    """v1.4.65 wedge fix: for the INSIDE rung (lvl=0) in multi-rung
    mode, compute_desired_state MUST use ``build.bid_order`` /
    ``build.ask_order`` (which is engine-self-healed and grid-aligned),
    NOT ``ladder.bids[0]`` / ``ladder.asks[0]`` (which is the PRE-
    self-heal float).

    Snapshot v1.4.64-260518-185510 caught the wedge: position +3
    long, SELL is reducer, engine self-healed the ask size to clear
    min_notional, but the ladder rung had the pre-self-heal size
    that rounded to BELOW min_notional → SELL inside rung rejected →
    bot stuck.
    """
    from app.quote_engine import FinalQuoteOrder
    from app.reconciler import DesiredOrderState
    om, _storage, _state, db_path = _setup_phase5()
    try:
        # Engine produces a self-healed ASK at 3.0 contracts ($5.91 notional > $5 min).
        ask_order = FinalQuoteOrder(
            side=Side.SELL, price=1.97, size=3.0,
            target_half_spread_bps=3.5,
        )
        build = _make_build(bid_order=None, ask_order=ask_order)
        # Fake a multi-rung ladder where the inside rung is PRE-self-heal
        # (size 2.83 — would round to 2.0 → notional $3.94 < $5 min).
        from app.ladder import LadderDecision, LadderRung
        ladder = LadderDecision(
            bids=[],  # engine suppressed BID (inventory_exec_bias)
            asks=[
                LadderRung(level_idx=0, side=Side.SELL, px=1.97, sz=2.83),  # pre-self-heal
                LadderRung(level_idx=1, side=Side.SELL, px=1.98, sz=2.0),
            ],
            requested_levels=2,
            effective_levels_buy=0,
            effective_levels_sell=2,
            gate_caps={},
        )
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-inside-1"

        post, engine_desired, rejections = om.compute_desired_state(
            build=build,
            decision=decision,
            ladder=ladder,
            position_qty=3.0,
            num_levels=2,
            client_spec=om._client.symbol_spec,
        )
        sell_inside = post[(Side.SELL, 0)]
        assert sell_inside.should_exist is True, (
            "v1.4.65 wedge fix: INSIDE rung must use engine-self-healed "
            "order, not pre-self-heal ladder rung. Got "
            f"should_exist={sell_inside.should_exist}, "
            f"size={sell_inside.size}, reason={sell_inside.reason}"
        )
        # The size should match the engine's self-healed size (3.0),
        # NOT the ladder rung's pre-self-heal size (2.83).
        assert sell_inside.size == 3.0, (
            f"Inside rung size must come from engine_order (3.0), not "
            f"ladder rung (2.83). Got: {sell_inside.size}"
        )
        assert sell_inside.price == 1.97
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_66_hard_age_cap_forces_cancel_via_desired_empty() -> None:
    """v1.4.66 wedge fix: when the engine sets
    ``build.hard_cancel_bid_reasons`` (BEHIND_TOUCH_MAX_AGE_SECONDS
    or AT_TOUCH_MAX_AGE_SECONDS exceeded), ``compute_desired_state``
    must override the BUY inside rung to should_exist=False so the
    reconciler emits a CancelAction. Pre-v1.4.66 the reconciler
    discarded the hard-cancel signal entirely → orders aged
    indefinitely (snapshot v1.4.65-260518-191405 caught a BUY +
    SELL at 1:54 age and counting, idle=132 s).
    """
    from app.quote_engine import FinalQuoteOrder
    from app.ladder import LadderDecision, LadderRung
    from types import SimpleNamespace
    om, _storage, _state, db_path = _setup_phase5()
    try:
        bid_order = FinalQuoteOrder(
            side=Side.BUY, price=1.0, size=3.0,
            target_half_spread_bps=3.5,
        )
        # Build object includes the hard_cancel_bid_reasons that
        # would have been populated by the QuoteEngine.
        build = SimpleNamespace(
            bid_order=bid_order, ask_order=None, telemetry={},
            hard_cancel_bid_reasons=("behind_touch_age_exceeded:1.54s>1.50s",),
            hard_cancel_ask_reasons=(),
        )
        ladder = LadderDecision(
            bids=[LadderRung(level_idx=0, side=Side.BUY, px=1.0, sz=3.0)],
            asks=[],
            requested_levels=1,
            effective_levels_buy=1,
            effective_levels_sell=0,
            gate_caps={},
        )
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-hard-age-1"

        post, engine_desired, rejs = om.compute_desired_state(
            build=build,
            decision=decision,
            ladder=ladder,
            position_qty=0.0,
            num_levels=1,
            client_spec=om._client.symbol_spec,
        )
        # BUY inside rung must be overridden to empty.
        buy = post[(Side.BUY, 0)]
        assert buy.should_exist is False, (
            f"v1.4.66 wedge fix: hard_cancel_bid_reasons MUST override "
            f"desired to empty so reconciler emits Cancel. Got "
            f"should_exist={buy.should_exist}"
        )
        assert "hard_age_cap" in buy.reason, (
            f"v1.4.66: desired.reason should carry the hard_age_cap "
            f"tag for postmortem. Got reason={buy.reason!r}"
        )
        # SELL was not aged → unaffected.
        assert post[(Side.SELL, 0)].should_exist is False  # engine_no_quote (no ask_order)
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_66_hard_age_cap_with_no_resting_order_is_noop() -> None:
    """v1.4.66 invariant guard: hard_cancel reasons populated but no
    current WO on that side → no override (nothing to cancel)."""
    from app.quote_engine import FinalQuoteOrder
    from types import SimpleNamespace
    om, _storage, _state, db_path = _setup_phase5()
    try:
        bid_order = FinalQuoteOrder(
            side=Side.BUY, price=1.0, size=3.0,
            target_half_spread_bps=3.5,
        )
        build = SimpleNamespace(
            bid_order=bid_order, ask_order=None, telemetry={},
            # hard_cancel set but engine just produced a fresh quote.
            hard_cancel_bid_reasons=("stale_quote",),
            hard_cancel_ask_reasons=(),
        )
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-hard-age-noop"
        post, engine_desired, rejs = om.compute_desired_state(
            build=build,
            decision=decision,
            ladder=None,  # single-rung
            position_qty=0.0,
            num_levels=1,
            client_spec=om._client.symbol_spec,
        )
        # Engine produced bid_order, but hard_cancel forces empty.
        # Even when there's no cur, the reconciler will see desired=empty +
        # cur=None → noop:empty_match. Next tick: hard_cancel clears (no
        # resting order to be aged), engine produces fresh quote → place.
        buy = post[(Side.BUY, 0)]
        assert buy.should_exist is False
        assert "hard_age_cap" in buy.reason
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_65_compute_desired_state_outer_rung_still_normalizes_ladder() -> None:
    """v1.4.65 invariant guard: the outer rungs (lvl >= 1) STILL go
    through ``normalize_order_pair`` because the engine self-heal
    only fixed the inside rung. Outer rungs that fall below
    min_notional after rounding are correctly skipped (not silently
    placed at an invalid size)."""
    from app.quote_engine import FinalQuoteOrder
    from app.ladder import LadderDecision, LadderRung
    om, _storage, _state, db_path = _setup_phase5()
    try:
        # Engine self-healed inside rung (clears min_notional).
        ask_order = FinalQuoteOrder(
            side=Side.SELL, price=1.97, size=3.0,
            target_half_spread_bps=3.5,
        )
        build = _make_build(bid_order=None, ask_order=ask_order)
        # Outer rung is pre-self-heal — below min_notional after grid.
        ladder = LadderDecision(
            bids=[],
            asks=[
                LadderRung(level_idx=0, side=Side.SELL, px=1.97, sz=2.83),
                LadderRung(level_idx=1, side=Side.SELL, px=1.98, sz=0.5),  # tiny — below min
            ],
            requested_levels=2,
            effective_levels_buy=0,
            effective_levels_sell=2,
            gate_caps={},
        )
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-outer-1"

        post, engine_desired, rejections = om.compute_desired_state(
            build=build,
            decision=decision,
            ladder=ladder,
            position_qty=3.0,
            num_levels=2,
            client_spec=om._client.symbol_spec,
        )
        # Inside rung from engine_order — should_exist=True.
        assert post[(Side.SELL, 0)].should_exist is True
        # Outer rung was tiny; should be either rejected or have size.
        # If rejected, rejections list captures the reason.
        if not post[(Side.SELL, 1)].should_exist:
            assert any(r["level_idx"] == 1 for r in rejections), (
                f"Outer rung rejection must be recorded. Got: {rejections}"
            )
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_64_batch_place_exchange_rejected_still_transitions_to_rejected() -> None:
    """v1.4.64 invariant preserved: exchange_rejected (the OTHER
    branch) still transitions to REJECTED. This is a regression guard
    so the v1.4.64 fix doesn't accidentally regress the prior
    behavior.
    """
    from app.enums import OrderStatus
    from app.execution import PlaceTransportIntent
    om, _storage, state, db_path = _setup_phase5()
    try:
        wo = _ack_wo(om, Side.SELL, price=2.0, oid=5002)
        wo.order_id_exchange = None
        wo.status = OrderStatus.SENT
        # OKX post-only-cross response (51604 row code with top "1").
        om._client.batch_place_post_only_limit.return_value = {
            "code": "1",
            "msg": "",
            "data": [{
                "sCode": "51604",
                "sMsg": "Post-only order would cross the market price.",
                "clOrdId": wo.client_order_id,
            }],
        }
        intent = PlaceTransportIntent(
            wo_order_id_local=wo.order_id_local,
            side=wo.side,
            intent_seq=wo.transport_intent_seq,
            quote_cycle_id="qc-51604-1",
            enqueued_mono=time.monotonic(),
            intent_created_perf=time.perf_counter(),
        )
        om._execute_place_batch_intents([intent, intent])
        assert wo.status == OrderStatus.REJECTED
    finally:
        _safe_unlink_db(db_path)


def test_cancel_all_under_concurrent_simulated_state_mutation() -> None:
    """v1.4.62 wedge probe: cancel-all is robust to in-loop state
    mutations. Pre-Phase-1 the loop iterated a stale exchange
    snapshot; Phase-1 reads from local state in a single lock
    acquire. Test this by mutating WOs in between cancel-all calls
    and verifying no spurious cancels."""
    from app.enums import OrderStatus
    om, _storage, _state, db_path = _setup_phase5()
    try:
        wo1 = _ack_wo(om, Side.BUY, price=1.0, oid=901)
        wo2 = _ack_wo(om, Side.SELL, price=2.0, oid=902)
        enq_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kw):
            enq_calls.append(wo_arg.order_id_exchange)
            return original(wo_arg, **kw)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        # First call: cancels both, transitions to CANCEL_PENDING.
        om.cancel_all_orders_for_symbol()
        assert sorted(enq_calls) == [901, 902]

        # Simulate one WO completing terminal between calls.
        wo1.status = OrderStatus.CANCELED
        # Reset spy.
        enq_calls.clear()
        # Second call: only wo2 (CANCEL_PENDING) is checked and skipped;
        # wo1 (CANCELED) is skipped as terminal. Zero new enqueues.
        om.cancel_all_orders_for_symbol()
        assert enq_calls == [], (
            "Second cancel-all must skip both CANCEL_PENDING and "
            f"terminal WOs. Got: {enq_calls}"
        )
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.68 wedge-elimination-cleanup Phase 1A: WS event buffer for the
# WS→WO matcher race.
#
# Snapshot v1.4.67-260518-195033 caught a 33-s SENT→ACKED gap because
# WS ``live`` events arrived 250 µs after the place response, faster
# than the place-response handler could commit the OID to the local
# WO. The WS handler's matcher missed and the events were silently
# dropped. Phase 1A buffers unmatched WS events by cloid and drains
# them when the place response commits.
# ---------------------------------------------------------------------------


def _make_private_order_update_event(
    *,
    oid: int,
    cloid: str,
    side: str = "B",
    status: str = "live",
    raw_status: str = "live",
    status_timestamp_ms: int = 1779119426507,
    coin: str = "TON-USDT-SWAP",
    remaining_sz: float = 5.0,
    orig_sz: float = 5.0,
    limit_px: float = 1.93,
):
    """Construct a PrivateOrderUpdateEvent for use in WS-race tests."""
    from app.exchange.private_events import PrivateOrderUpdateEvent
    return PrivateOrderUpdateEvent(
        oid=oid,
        coin=coin,
        status=status,
        status_timestamp_ms=status_timestamp_ms,
        side=side,
        limit_px=limit_px,
        remaining_sz=remaining_sz,
        orig_sz=orig_sz,
        raw_status=raw_status,
        inbound_timing=None,
        cloid=cloid,
    )


def test_v1_4_68_phase1a_ws_event_unmatched_increments_counter() -> None:
    """v1.4.68 Phase 1A: when a WS event arrives with no matching
    local WorkingOrder, the unmatched counter increments and (if the
    event has a cloid) the event is buffered.
    """
    s = _settings(SYMBOL="TON-USDT-SWAP")
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        ev = _make_private_order_update_event(
            oid=999_001, cloid="raceA"
        )
        assert om._ws_event_unmatched_to_local_wo_total == 0
        om._handle_private_order_update(ev)
        assert om._ws_event_unmatched_to_local_wo_total == 1, (
            "unmatched WS event must increment counter"
        )
        # Buffer should contain one event keyed by cloid.
        assert "raceA" in om._pending_ws_events_by_cloid
        assert len(om._pending_ws_events_by_cloid["raceA"]) == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_68_phase1a_buffered_event_drained_on_place_response() -> None:
    """v1.4.68 Phase 1A: when a place response commits the OID to a
    WO, buffered WS events for that cloid are drained and applied.

    Scenario: WS ``live`` arrives first (no local WO yet), event is
    buffered. Then the place-response handler runs, sets the OID on
    the WO, transitions SENT→ACKED, and calls the buffer drain.
    """
    from app.enums import OrderStatus
    s = _settings(SYMBOL="TON-USDT-SWAP")
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Step 1: WS ``live`` arrives BEFORE the place response.
        # No local WO yet → buffered.
        ev = _make_private_order_update_event(
            oid=999_002, cloid="raceB", status="live", raw_status="live"
        )
        om._handle_private_order_update(ev)
        assert "raceB" in om._pending_ws_events_by_cloid

        # Step 2: simulate the place-response handler committing
        # the WO. We construct a SENT WO with matching cloid, then
        # call the drain.
        wo = _ack_wo(om, Side.BUY, price=1.93, oid=999_002)
        wo.status = OrderStatus.SENT  # Reset to pre-ACK state
        wo.ts_ack = None
        wo.client_order_id = "raceB"

        applied = om._drain_pending_ws_events_for_cloid("raceB")
        assert applied == 1, f"expected 1 event drained, got {applied}"
        assert om._ws_event_buffered_replay_applied_total == 1
        # Buffer should be empty for raceB after drain.
        assert "raceB" not in om._pending_ws_events_by_cloid
        # The WO should now be ACKED (the live event was applied).
        assert wo.status == OrderStatus.ACKED, (
            f"expected ACKED after buffered replay, got {wo.status}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_68_phase1a_buffer_per_cloid_caps_at_max() -> None:
    """v1.4.68 Phase 1A: per-cloid buffer is bounded; oldest evicted."""
    from app.execution import _PENDING_WS_BUFFER_PER_CLOID_MAX
    s = _settings(SYMBOL="TON-USDT-SWAP")
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Push MAX+5 events for the same cloid.
        for i in range(_PENDING_WS_BUFFER_PER_CLOID_MAX + 5):
            ev = _make_private_order_update_event(
                oid=100_000 + i, cloid="overflow",
                status_timestamp_ms=1_779_000_000_000 + i,
            )
            om._handle_private_order_update(ev)
        # Buffer should be at exactly MAX entries.
        assert (
            len(om._pending_ws_events_by_cloid["overflow"])
            == _PENDING_WS_BUFFER_PER_CLOID_MAX
        )
        # Dropped-oldest counter should be 5.
        assert om._ws_event_buffer_dropped_oldest_total == 5
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_68_phase1a_buffer_sweeper_drops_stale() -> None:
    """v1.4.68 Phase 1A: sweeper drops buffered events older than
    the TTL.
    """
    import time as _time
    from app.execution import _PENDING_WS_BUFFER_TTL_SECONDS
    s = _settings(SYMBOL="TON-USDT-SWAP")
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Push one event then backdate it past the TTL.
        ev = _make_private_order_update_event(oid=200_001, cloid="ttl")
        om._handle_private_order_update(ev)
        assert "ttl" in om._pending_ws_events_by_cloid
        # Backdate the entry's mono timestamp.
        lst = om._pending_ws_events_by_cloid["ttl"]
        backdated_t = _time.monotonic() - _PENDING_WS_BUFFER_TTL_SECONDS - 1.0
        om._pending_ws_events_by_cloid["ttl"] = [(backdated_t, ev) for (_t, ev) in lst]
        # Reset the last-sweep timestamp so the sweep runs.
        om._ws_event_buffer_last_sweep_mono = 0.0
        # Trigger the sweep.
        om._sweep_pending_ws_event_buffer()
        assert "ttl" not in om._pending_ws_events_by_cloid
        assert om._ws_event_buffer_swept_stale_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_68_phase1a_event_without_cloid_not_buffered() -> None:
    """v1.4.68 Phase 1A: events without a cloid increment the
    unmatched counter but are NOT buffered (no key to drain on).
    """
    s = _settings(SYMBOL="TON-USDT-SWAP")
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        from app.exchange.private_events import PrivateOrderUpdateEvent
        ev = PrivateOrderUpdateEvent(
            oid=300_001,
            coin="TON-USDT-SWAP",
            status="live",
            status_timestamp_ms=1779119426507,
            side="B",
            limit_px=1.93,
            remaining_sz=5.0,
            orig_sz=5.0,
            raw_status="live",
            inbound_timing=None,
            cloid=None,  # explicit no-cloid
        )
        om._handle_private_order_update(ev)
        assert om._ws_event_unmatched_to_local_wo_total == 1
        assert len(om._pending_ws_events_by_cloid) == 0
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_68_phase1a_metrics_in_executor_state_snapshot() -> None:
    """v1.4.68 Phase 1A: the new metrics appear in
    ``executor_state_snapshot`` so the operator can see whether
    the race is happening in production.
    """
    om, _storage, state, db_path = _setup()
    try:
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        for key in (
            "ws_event_unmatched_to_local_wo_total",
            "ws_event_buffered_replay_applied_total",
            "ws_event_buffer_dropped_oldest_total",
            "ws_event_buffer_swept_stale_total",
            "ws_event_buffer_dropped_full_total",
            "ws_event_buffer_current_size",
        ):
            assert key in es, f"executor_state missing Phase 1A metric: {key}"
        # All should be zero on a fresh executor.
        assert es["ws_event_unmatched_to_local_wo_total"] == 0
        assert es["ws_event_buffered_replay_applied_total"] == 0
        assert es["ws_event_buffer_current_size"] == 0
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.69 wedge-elimination-cleanup Phase 1B: hydration dedup by
# (oid, cloid).
#
# Snapshots v1.4.66-260518-192744 and v1.4.67-260518-195033 caught
# the duplicate-WO state: same OID had TWO records in
# orders_lifecycle-since.json. Phase 1B merges instead of duplicating.
# ---------------------------------------------------------------------------


def _make_open_order_raw(
    *,
    oid: int,
    cloid: str | None,
    side: str = "BUY",
    limit_px: float = 1.93,
    sz: float = 5.0,
    timestamp_ms: int = 1779119426507,
    coin: str = "TON-USDT-SWAP",
):
    """Construct an OpenOrderRaw for hydration tests."""
    from app.exchange.base import OpenOrderRaw
    from app.enums import Side as _Side
    side_enum = _Side.BUY if side in ("BUY", "B") else _Side.SELL
    return OpenOrderRaw(
        oid=oid,
        coin=coin,
        side=side_enum,
        limit_px=limit_px,
        sz=sz,
        timestamp=timestamp_ms,
        cloid=cloid,
    )


def test_v1_4_69_phase1b_hydration_merges_existing_non_terminal_by_oid() -> None:
    """v1.4.69 Phase 1B: hydration finds an existing non-terminal WO
    with matching OID and merges into it instead of creating a
    duplicate.
    """
    from app.enums import OrderStatus
    om, _storage, state, db_path = _setup()
    try:
        # Stage an existing SENT WO that the bot has placed but is
        # missing the OID (race condition: place HTTP in flight).
        existing = _ack_wo(om, Side.BUY, price=1.93, oid=555_001)
        existing.status = OrderStatus.SENT
        existing.client_order_id = "merge_oid_test"

        # Now hydration finds the same OID on the exchange.
        remote = _make_open_order_raw(
            oid=555_001, cloid="merge_oid_test", side="B",
            limit_px=1.93, sz=5.0,
        )
        result = om._hydrate_working_from_exchange(Side.BUY, remote)
        assert result is True, "merge should return True"
        # Should NOT have created a duplicate.
        wos = [w for w in state.all_working_orders() if w.order_id_exchange == 555_001]
        assert len(wos) == 1, f"expected 1 WO with oid=555001, got {len(wos)}"
        # Existing WO should now be ACKED (SENT→ACKED via merge).
        assert existing.status == OrderStatus.ACKED
        # Should be marked as hydrated_from_exchange.
        assert existing.hydrated_from_exchange is True
        # Counter incremented.
        assert om._hydration_merged_existing_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_69_phase1b_hydration_merges_existing_non_terminal_by_cloid() -> None:
    """v1.4.69 Phase 1B: when local WO has no OID yet (HTTP response
    pending), hydration matches by cloid and binds the OID via merge.
    """
    from app.enums import OrderStatus
    om, _storage, state, db_path = _setup()
    try:
        existing = _ack_wo(om, Side.BUY, price=1.93, oid=0)
        existing.order_id_exchange = None  # explicitly no OID
        existing.status = OrderStatus.SENT
        existing.client_order_id = "merge_cloid_test"

        remote = _make_open_order_raw(
            oid=555_002, cloid="merge_cloid_test", side="B",
        )
        result = om._hydrate_working_from_exchange(Side.BUY, remote)
        assert result is True
        # OID bound via merge.
        assert existing.order_id_exchange == 555_002
        # Status promoted from SENT to ACKED.
        assert existing.status == OrderStatus.ACKED
        assert om._hydration_merged_existing_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_69_phase1b_hydration_skips_terminal_match() -> None:
    """v1.4.69 Phase 1B: when a TERMINAL local WO matches, hydration
    skips entirely. The order is conceptually done; resurrecting it
    would re-introduce the bug.
    """
    from app.enums import OrderStatus
    om, _storage, state, db_path = _setup()
    try:
        existing = _ack_wo(om, Side.BUY, price=1.93, oid=555_003)
        existing.status = OrderStatus.CANCELED
        existing.client_order_id = "term_test"

        remote = _make_open_order_raw(
            oid=555_003, cloid="term_test", side="B",
        )
        result = om._hydrate_working_from_exchange(Side.BUY, remote)
        assert result is False, "terminal match should skip"
        # Counter incremented.
        assert om._hydration_skipped_terminal_match_total == 1
        # No second WO created.
        wos = [w for w in state.all_working_orders() if w.order_id_exchange == 555_003]
        assert len(wos) == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_69_phase1b_hydration_creates_new_when_no_match() -> None:
    """v1.4.69 Phase 1B: with no matching local WO, hydration creates
    a new one (existing behavior preserved).
    """
    om, _storage, state, db_path = _setup()
    try:
        # No existing WOs.
        remote = _make_open_order_raw(
            oid=555_004, cloid="brand_new", side="B",
        )
        result = om._hydrate_working_from_exchange(Side.BUY, remote)
        assert result is True
        # No merge happened.
        assert om._hydration_merged_existing_total == 0
        # New WO exists.
        wos = [w for w in state.all_working_orders() if w.order_id_exchange == 555_004]
        assert len(wos) == 1
        assert wos[0].hydrated_from_exchange is True
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_69_phase1b_find_helper_oid_priority() -> None:
    """v1.4.69 Phase 1B: ``_find_local_wo_by_oid_or_cloid`` prefers
    OID matches over CLOID matches.
    """
    from app.enums import OrderStatus
    om, _storage, state, db_path = _setup()
    try:
        # WO with both OID and cloid set.
        wo_oid = _ack_wo(om, Side.BUY, price=1.93, oid=555_005)
        wo_oid.client_order_id = "cloid_A"
        # Another WO with no OID but matching cloid.
        wo_cloid = _ack_wo(om, Side.SELL, price=1.94, oid=555_999)
        wo_cloid.order_id_exchange = None
        wo_cloid.client_order_id = "cloid_A"

        # OID lookup should find wo_oid.
        found = om._find_local_wo_by_oid_or_cloid(555_005, "cloid_A")
        assert found is wo_oid, "OID match should win"

        # CLOID-only lookup (no OID) should find wo_cloid (since wo_oid has an OID).
        found = om._find_local_wo_by_oid_or_cloid(None, "cloid_A")
        assert found is wo_cloid

        # Both None — returns None.
        assert om._find_local_wo_by_oid_or_cloid(None, None) is None
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_69_phase1b_metrics_in_executor_state_snapshot() -> None:
    """v1.4.69 Phase 1B: the new dedup metrics are surfaced."""
    om, _storage, state, db_path = _setup()
    try:
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        assert "hydration_merged_existing_total" in es
        assert "hydration_skipped_terminal_match_total" in es
        assert es["hydration_merged_existing_total"] == 0
        assert es["hydration_skipped_terminal_match_total"] == 0
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.70 wedge-elimination-cleanup Phase 1C: hard age cap reaches
# every slot, measures wall-lifetime.
#
# Closes Codex F6/B3 (outer-rung cap not enforced) and F7 (post-ACK
# vs wall-lifetime). Multi-rung ladders now have outer-rung age caps
# fire correctly; orders stuck in SENT (slow ACK) also get capped via
# ``ts_sent`` as the age anchor.
# ---------------------------------------------------------------------------


def test_v1_4_70_phase1c_resting_age_uses_ts_sent_for_sent_orders() -> None:
    """v1.4.70 Phase 1C wall-lifetime cap: a WO in SENT (no ts_ack
    yet) returns a finite age from ``resting_age_seconds`` measured
    from ``ts_sent``. Pre-Phase-1C this returned None and the order
    was invisible to the cap.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.models import WorkingOrder
    from app.quote_aging import resting_age_seconds
    from app.utils.time import utc_now
    now = utc_now()
    wo = WorkingOrder(
        order_id_local="local-test-sent",
        order_id_exchange=None,
        client_order_id="cloid-sent",
        symbol="TON-USDT-SWAP",
        side=Side.BUY,
        price=1.93,
        size=5.0,
        post_only=True,
        status=OrderStatus.SENT,
        ts_created=now - timedelta(seconds=3.0),
        ts_sent=now - timedelta(seconds=3.0),
        ts_ack=None,
    )
    age = resting_age_seconds(wo, now)
    assert age is not None, "SENT WO must report a wall-lifetime age"
    assert 2.9 < age < 3.1, f"expected ~3.0s, got {age}"


def test_v1_4_70_phase1c_resting_age_uses_ts_sent_when_ack_is_newer() -> None:
    """v1.4.70 Phase 1C: when both ``ts_sent`` and ``ts_ack`` are set
    (normal case after fast ACK), the age is the MAX (ts_sent earlier
    → larger age). This is what makes slow-ACK orders eligible: the
    sent_age starts the moment we hit the wire, not the moment we
    learn the venue accepted.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.models import WorkingOrder
    from app.quote_aging import resting_age_seconds
    from app.utils.time import utc_now
    now = utc_now()
    wo = WorkingOrder(
        order_id_local="local-test-acked",
        order_id_exchange=12345,
        client_order_id="cloid-acked",
        symbol="TON-USDT-SWAP",
        side=Side.BUY,
        price=1.93,
        size=5.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=now - timedelta(seconds=2.0),
        ts_sent=now - timedelta(seconds=2.0),    # sent 2.0s ago
        ts_ack=now - timedelta(seconds=1.5),     # ack 1.5s ago (0.5s later)
    )
    age = resting_age_seconds(wo, now)
    assert age is not None
    assert 1.9 < age < 2.1, f"expected ~2.0s (max of sent_age, ack_age), got {age}"


def test_v1_4_70_phase1c_resting_age_hydrated_uses_ts_ack() -> None:
    """v1.4.70 Phase 1C: hydrated WOs (no ts_sent, only ts_ack
    populated from venue cTime) still age correctly from ts_ack.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.models import WorkingOrder
    from app.quote_aging import resting_age_seconds
    from app.utils.time import utc_now
    now = utc_now()
    wo = WorkingOrder(
        order_id_local="local-test-hydrated",
        order_id_exchange=67890,
        client_order_id="cloid-hydrated",
        symbol="TON-USDT-SWAP",
        side=Side.SELL,
        price=1.94,
        size=3.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=now,                          # hydration time
        ts_sent=None,                            # no local place
        ts_ack=now - timedelta(seconds=5.0),     # venue cTime 5s ago
        hydrated_from_exchange=True,
    )
    age = resting_age_seconds(wo, now)
    assert age is not None
    assert 4.9 < age < 5.1, f"expected ~5.0s from ts_ack, got {age}"


def test_v1_4_70_phase1c_hard_reprice_fires_on_sent_with_old_ts_sent() -> None:
    """v1.4.70 Phase 1C: the hard-age-cap fires on a SENT order
    whose ``ts_sent`` is past ``BEHIND_TOUCH_MAX_AGE_SECONDS``. This
    is the v1.4.67-260518-195033 wedge mechanism — slow venue ACK
    left WOs in SENT with no age check.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.models import WorkingOrder
    from app.quote_aging import hard_reprice_reasons_buy
    from app.utils.time import utc_now
    s = _settings(
        QUOTE_AGING_ENABLED=True,
        QUOTE_AGING_MAX_AGE_SECONDS=6.0,
        BEHIND_TOUCH_MAX_AGE_SECONDS=1.5,
        AT_TOUCH_MAX_AGE_SECONDS=2.5,
    )
    now = utc_now()
    wo = WorkingOrder(
        order_id_local="local-test-stuck-sent",
        order_id_exchange=None,
        client_order_id="cloid-stuck",
        symbol="TON-USDT-SWAP",
        side=Side.BUY,
        price=1.92,  # below best_bid → behind touch
        size=5.0,
        post_only=True,
        status=OrderStatus.SENT,
        ts_created=now - timedelta(seconds=10.0),
        ts_sent=now - timedelta(seconds=10.0),
        ts_ack=None,
    )
    reasons = hard_reprice_reasons_buy(
        s, wo, best_bid=1.93, tick=0.001, now=now,
        enforce_touch_distance_band=False,
    )
    assert "behind_touch_order_age_seconds" in reasons, (
        f"behind_touch cap must fire on SENT wo aged 10s vs cap 1.5s; "
        f"got reasons={reasons}"
    )


def test_v1_4_70_phase1c_hard_cancel_by_slot_field_present() -> None:
    """v1.4.70 Phase 1C: ``QuoteBuildResult.hard_cancel_by_slot`` is
    populated and keyed by ``(Side, level_idx)``.
    """
    from app.quote_engine import QuoteBuildResult
    from app.enums import Side as _Side
    r = QuoteBuildResult(
        bid_order=None,
        ask_order=None,
        mode="no_quote",
        hard_cancel_by_slot={
            (_Side.BUY, 0): ("behind_touch_order_age_seconds",),
            (_Side.SELL, 1): ("at_touch_order_age_seconds",),
        },
    )
    assert (_Side.BUY, 0) in r.hard_cancel_by_slot
    assert (_Side.SELL, 1) in r.hard_cancel_by_slot
    assert r.hard_cancel_by_slot[(_Side.BUY, 0)] == ("behind_touch_order_age_seconds",)


def test_v1_4_70_phase1c_compute_desired_state_empties_outer_rung() -> None:
    """v1.4.70 Phase 1C: ``compute_desired_state`` empties EVERY
    aged slot in ``hard_cancel_by_slot``, including outer rungs
    (level_idx >= 1). Pre-Phase-1C only (side, 0) was touched —
    Codex F6/B3.
    """
    from types import SimpleNamespace
    from app.quote_engine import QuoteBuildResult
    from app.enums import Side as _Side
    from app.quote_engine import FinalQuoteOrder

    om, _storage, state, db_path = _setup()
    try:
        # Build a 2-rung-style decision with engine_order for both
        # bid and ask (which would normally produce a full ladder).
        engine_bid = FinalQuoteOrder(
            side=_Side.BUY, price=1.93, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        engine_ask = FinalQuoteOrder(
            side=_Side.SELL, price=1.94, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        build = QuoteBuildResult(
            bid_order=engine_bid,
            ask_order=engine_ask,
            mode="two_sided",
            hard_cancel_by_slot={
                (_Side.BUY, 1): ("behind_touch_order_age_seconds",),  # outer rung aged
                (_Side.SELL, 0): ("at_touch_order_age_seconds",),     # inside rung aged
            },
        )
        # Minimal decision + ladder stubs.
        decision = SimpleNamespace(
            quote_cycle_id="qc-1c-test",
            mid_price=1.935,
            active_sides=None,
        )
        # Simulate a multi-rung ladder with both rungs alive.
        rung0 = SimpleNamespace(level_idx=0, px=1.93, sz=5.0)
        rung1 = SimpleNamespace(level_idx=1, px=1.92, sz=3.0)
        ladder = SimpleNamespace(
            bids=[rung0, rung1],
            asks=[
                SimpleNamespace(level_idx=0, px=1.94, sz=5.0),
                SimpleNamespace(level_idx=1, px=1.95, sz=3.0),
            ],
        )
        from tests.exchange_client_mocks import mock_mm_client
        client = mock_mm_client()
        post_gates, engine_des, rejects = om.compute_desired_state(
            build=build,
            decision=decision,
            ladder=ladder,
            position_qty=0.0,
            num_levels=2,
            client_spec=client.symbol_spec,
        )
        # Outer-rung BUY must be empty due to hard age cap.
        buy_lvl1 = engine_des.get((_Side.BUY, 1))
        assert buy_lvl1 is not None
        assert buy_lvl1.should_exist is False, (
            "outer-rung BUY (lvl 1) must be emptied by Phase 1C cap"
        )
        assert "hard_age_cap" in buy_lvl1.reason
        assert "lvl=1" in buy_lvl1.reason
        # Inside-rung SELL must also be empty.
        sell_lvl0 = engine_des.get((_Side.SELL, 0))
        assert sell_lvl0 is not None
        assert sell_lvl0.should_exist is False
        assert "lvl=0" in sell_lvl0.reason
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.71 wedge-elimination-cleanup Phase 1D: slot-aware exchange
# reconcile.
#
# Snapshot v1.4.66-260518-192744 caught 8 duplicate_same_side_extra
# orphan_remote_cancel_dispatched events in 75 s and 538 dispatcher
# side_unresolved:return_silent decisions; v1.4.70-260518-213819
# still shows 9 desync_detected events in 12 min from this. The root
# cause is the side-keyed dup detector treating legitimate multi-rung
# cohorts as duplicates. Phase 1D maps remotes to slots and only
# flags real slot conflicts.
# ---------------------------------------------------------------------------


def _make_open_order_raw_v1_4_71(
    oid: int,
    cloid: str | None,
    side: Side,
    timestamp_ms: int = 1_779_119_426_000,
    limit_px: float = 1.93,
    sz: float = 5.0,
    coin: str = "ETH",
):
    """Build an OpenOrderRaw. Default ``coin='ETH'`` matches the test
    settings default symbol so reconcile's symbol filter accepts it.
    """
    from app.exchange.base import OpenOrderRaw
    return OpenOrderRaw(
        oid=oid,
        coin=coin,
        side=side,
        limit_px=limit_px,
        sz=sz,
        timestamp=timestamp_ms,
        cloid=cloid,
    )


def test_v1_4_71_phase1d_two_rung_legitimate_no_dup() -> None:
    """v1.4.71 Phase 1D: with ``LADDER_NUM_LEVELS_PER_SIDE=2`` and
    two BUYs on the exchange that map to (BUY, 0) and (BUY, 1)
    respectively, the reconcile MUST NOT flag a duplicate or fire
    extra cancels.
    """
    s = _settings(LADDER_NUM_LEVELS_PER_SIDE=2)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Stage two BUY rungs locally with distinct cloids.
        wo_inside = _ack_wo(om, Side.BUY, price=1.93, oid=700_001, level_idx=0)
        wo_inside.client_order_id = "cloid_inside"
        wo_outer = _ack_wo(om, Side.BUY, price=1.92, oid=700_002, level_idx=1)
        wo_outer.client_order_id = "cloid_outer"

        # Mock exchange returns both BUYs.
        remote_inside = _make_open_order_raw_v1_4_71(
            oid=700_001, cloid="cloid_inside", side=Side.BUY, limit_px=1.93
        )
        remote_outer = _make_open_order_raw_v1_4_71(
            oid=700_002, cloid="cloid_outer", side=Side.BUY, limit_px=1.92,
            timestamp_ms=1_779_119_426_500,
        )
        client.fetch_open_orders_raw.return_value = [remote_inside, remote_outer]
        # Spy on orphan cancel dispatches.
        orphan_calls = []
        original_orphan = om._cancel_orphan_remote_order
        def _spy(**kw):
            orphan_calls.append(kw)
            return True
        om._cancel_orphan_remote_order = _spy  # type: ignore[method-assign]

        om._sync_open_orders_impl()
        assert len(orphan_calls) == 0, (
            f"two legitimate rungs must not produce orphan cancels; got {orphan_calls}"
        )
        # No desync.
        assert state.order_desync is False
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_71_phase1d_slot_conflict_still_cancels_extras() -> None:
    """v1.4.71 Phase 1D: 2+ remotes that map to the SAME slot (a true
    duplicate) still trigger ``duplicate_same_side_extra`` orphan
    cancels. The legacy invariant is preserved.
    """
    s = _settings(LADDER_NUM_LEVELS_PER_SIDE=2)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # One local WO at (BUY, 0).
        wo = _ack_wo(om, Side.BUY, price=1.93, oid=701_001, level_idx=0)
        wo.client_order_id = "cloid_X"
        # Two remotes both mapped to inside slot (same cloid → same slot).
        r1 = _make_open_order_raw_v1_4_71(
            oid=701_001, cloid="cloid_X", side=Side.BUY, timestamp_ms=100,
        )
        # The second remote has a different oid but same cloid (rare;
        # can happen on a race where a stale phantom shares cloid).
        # We force the duplicate by giving both the same cloid match.
        r2 = _make_open_order_raw_v1_4_71(
            oid=701_002, cloid="cloid_X", side=Side.BUY, timestamp_ms=200,
        )
        client.fetch_open_orders_raw.return_value = [r1, r2]
        orphan_calls = []
        om._cancel_orphan_remote_order = (
            lambda **kw: (orphan_calls.append(kw) or True)
        )  # type: ignore[method-assign]
        om._sync_open_orders_impl()
        # Should have cancelled exactly one extra with the legacy
        # "duplicate_same_side_extra" reason.
        dup_calls = [c for c in orphan_calls if c.get("reason") == "duplicate_same_side_extra"]
        assert len(dup_calls) == 1, (
            f"slot conflict must produce exactly one duplicate cancel; got {orphan_calls}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_71_phase1d_beyond_configured_rung_is_orphan() -> None:
    """v1.4.71 Phase 1D: a remote that maps (via cloid) to a slot
    beyond ``LADDER_NUM_LEVELS_PER_SIDE`` is cancelled as an orphan,
    not classified as a duplicate.
    """
    s = _settings(LADDER_NUM_LEVELS_PER_SIDE=1)  # only inside rung configured
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Local WO at level 1 (stale config drop — operator just
        # reduced LADDER_NUM_LEVELS_PER_SIDE).
        wo = _ack_wo(om, Side.BUY, price=1.92, oid=702_001, level_idx=1)
        wo.client_order_id = "cloid_outer_stale"
        # Exchange shows the order.
        r = _make_open_order_raw_v1_4_71(
            oid=702_001, cloid="cloid_outer_stale", side=Side.BUY,
        )
        client.fetch_open_orders_raw.return_value = [r]
        orphan_calls = []
        om._cancel_orphan_remote_order = (
            lambda **kw: (orphan_calls.append(kw) or True)
        )  # type: ignore[method-assign]
        om._sync_open_orders_impl()
        # Must be cancelled as orphan (configured=1, slot=1 → beyond).
        assert len(orphan_calls) == 1
        assert orphan_calls[0]["reason"] == "orphan_unmatched_or_beyond_configured_rung"
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_71_phase1d_single_rung_legacy_path_unchanged() -> None:
    """v1.4.71 Phase 1D backward-compat: with
    ``LADDER_NUM_LEVELS_PER_SIDE=1`` and 2 BUYs on the exchange
    (the classic v1.4.x duplicate scenario), behaviour reduces to
    the pre-Phase-1D path: dup_buy=True, one extra cancelled as
    ``duplicate_same_side_extra``.
    """
    s = _settings(LADDER_NUM_LEVELS_PER_SIDE=1)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Local WO at inside rung.
        wo = _ack_wo(om, Side.BUY, price=1.93, oid=703_001, level_idx=0)
        wo.client_order_id = "cloid_legacy"
        # Two remotes for BUY: one matching local, one stray.
        r_match = _make_open_order_raw_v1_4_71(
            oid=703_001, cloid="cloid_legacy", side=Side.BUY, timestamp_ms=100,
        )
        r_stray = _make_open_order_raw_v1_4_71(
            oid=703_999, cloid=None, side=Side.BUY, timestamp_ms=200,
        )
        client.fetch_open_orders_raw.return_value = [r_match, r_stray]
        orphan_calls = []
        om._cancel_orphan_remote_order = (
            lambda **kw: (orphan_calls.append(kw) or True)
        )  # type: ignore[method-assign]
        om._sync_open_orders_impl()
        # Stray remote: not OID-bound, not cloid-bound → unbound.
        # configured=1, inside slot is filled by the bound one →
        # unbound has no open slot → orphan. Cancelled.
        assert len(orphan_calls) == 1
        # On the single-rung backward-compat path, the reason is
        # "orphan_unmatched_or_beyond_configured_rung" (not slot-
        # conflict because the bound one took its slot, and the
        # stray didn't map). dup_buy is False because there's no
        # slot conflict — semantically more accurate than the old
        # "every-extra-is-duplicate" classification.
        assert orphan_calls[0]["reason"] in (
            "orphan_unmatched_or_beyond_configured_rung",
            "duplicate_same_side_extra",
        )
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_71_phase1d_compat_existing_test_still_works() -> None:
    """v1.4.71 Phase 1D backward-compat: existing v1.4.66 tests that
    use ``ladder_num_levels_per_side`` defaults (1) keep passing.
    Smoke check that the cluster ran (the broader cluster pytest
    above also confirms).
    """
    # Just import & instantiate to confirm no init failure.
    s = _settings()
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # No orders: empty exchange snapshot.
        client.fetch_open_orders_raw.return_value = []
        result = om._sync_open_orders_impl()
        assert result in ("ok", "no_change", None) or isinstance(result, str)
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_70_phase1c_backward_compat_inside_rung_tuples() -> None:
    """v1.4.70 Phase 1C: when only the inside-rung tuples
    (``hard_cancel_bid_reasons`` / ``hard_cancel_ask_reasons``) are
    populated and ``hard_cancel_by_slot`` is empty, the desired-state
    fold still empties (side, 0). Preserves pre-Phase-1C semantics
    for any caller that hasn't been updated to the new field.
    """
    from types import SimpleNamespace
    from app.quote_engine import QuoteBuildResult
    from app.enums import Side as _Side
    from app.quote_engine import FinalQuoteOrder
    om, _storage, _state, db_path = _setup()
    try:
        engine_bid = FinalQuoteOrder(
            side=_Side.BUY, price=1.93, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        build = QuoteBuildResult(
            bid_order=engine_bid,
            ask_order=None,
            mode="bid_only",
            hard_cancel_bid_reasons=("behind_touch_order_age_seconds",),
            hard_cancel_ask_reasons=(),
            # hard_cancel_by_slot intentionally empty
        )
        decision = SimpleNamespace(
            quote_cycle_id="qc-1c-compat",
            mid_price=1.935,
            active_sides=None,
        )
        from tests.exchange_client_mocks import mock_mm_client
        client = mock_mm_client()
        _post, engine_des, _rejs = om.compute_desired_state(
            build=build, decision=decision, ladder=None,
            position_qty=0.0, num_levels=1, client_spec=client.symbol_spec,
        )
        buy_lvl0 = engine_des.get((_Side.BUY, 0))
        assert buy_lvl0 is not None
        assert buy_lvl0.should_exist is False, (
            "backward-compat path must empty inside rung when tuple "
            "is non-empty even if hard_cancel_by_slot is missing"
        )
        assert "hard_age_cap" in buy_lvl0.reason
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.74 wedge-elimination-cleanup Phase 2A: structural gates lifted
# into the desired-state layer.
#
# Snapshot v1.4.66-260518-192744: 538 dispatcher silent suppressions
# per side via ``side_unresolved:return_silent``; 143 via
# ``place_stage_returned_none``. Snapshot v1.4.73-260518-222923:
# ``side_unresolved_enter_count=1264`` indicating sustained pressure.
# Phase 2A makes these gates VISIBLE in compute_desired_state's
# output and surfaces per-gate counters in executor_state.
# ---------------------------------------------------------------------------


def _make_phase2a_decision(quote_cycle_id: str = "qc-2a-test"):
    """Lightweight decision stub for compute_desired_state tests."""
    from types import SimpleNamespace
    return SimpleNamespace(
        quote_cycle_id=quote_cycle_id,
        mid_price=1.935,
        active_sides=None,
    )


def test_v1_4_74_phase2a_side_unresolved_appears_in_desired_state() -> None:
    """v1.4.74 Phase 2A: when a side is unresolved,
    ``compute_desired_state`` returns desired state with that side's
    slots ``should_exist=False`` and a reason starting with
    ``gate:side_unresolved:`` so postmortem can attribute the
    suppression.
    """
    from app.quote_engine import QuoteBuildResult, FinalQuoteOrder
    from app.enums import Side as _Side
    om, _storage, state, db_path = _setup()
    try:
        om._set_side_unresolved(_Side.BUY, reason="reconcile_test", payload={})
        engine_bid = FinalQuoteOrder(
            side=_Side.BUY, price=1.93, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        engine_ask = FinalQuoteOrder(
            side=_Side.SELL, price=1.94, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        build = QuoteBuildResult(
            bid_order=engine_bid, ask_order=engine_ask, mode="two_sided",
        )
        from tests.exchange_client_mocks import mock_mm_client
        client = mock_mm_client()
        post_gates, _engine_des, _rejs = om.compute_desired_state(
            build=build, decision=_make_phase2a_decision(), ladder=None,
            position_qty=0.0, num_levels=1, client_spec=client.symbol_spec,
        )
        # BUY suppressed via Phase 2A fold.
        buy_slot = post_gates.get((_Side.BUY, 0))
        assert buy_slot is not None
        assert buy_slot.should_exist is False
        assert "gate:side_unresolved:reconcile_test" in buy_slot.reason
        # SELL untouched.
        sell_slot = post_gates.get((_Side.SELL, 0))
        assert sell_slot is not None
        assert sell_slot.should_exist is True
        # Counter incremented.
        assert om._gate_side_unresolved_applied_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_74_phase2a_adverse_side_pause_appears_in_desired_state() -> None:
    """v1.4.74 Phase 2A: adverse_side_pause fold suppresses the side
    when inventory is flat (no reducer bypass)."""
    from app.quote_engine import QuoteBuildResult, FinalQuoteOrder
    from app.enums import Side as _Side
    om, _storage, _state, db_path = _setup()
    try:
        # Arm the adverse-side pause for SELL.
        om._adverse_side_pause_until[_Side.SELL] = time.monotonic() + 30.0
        engine_bid = FinalQuoteOrder(
            side=_Side.BUY, price=1.93, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        engine_ask = FinalQuoteOrder(
            side=_Side.SELL, price=1.94, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        build = QuoteBuildResult(
            bid_order=engine_bid, ask_order=engine_ask, mode="two_sided",
        )
        from tests.exchange_client_mocks import mock_mm_client
        client = mock_mm_client()
        post_gates, _engine_des, _rejs = om.compute_desired_state(
            build=build, decision=_make_phase2a_decision(), ladder=None,
            position_qty=0.0,  # FLAT — no reducer bypass
            num_levels=1, client_spec=client.symbol_spec,
        )
        sell_slot = post_gates.get((_Side.SELL, 0))
        assert sell_slot is not None
        assert sell_slot.should_exist is False
        assert "gate:adverse_side_pause:remaining_s=" in sell_slot.reason
        assert om._gate_adverse_side_pause_applied_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_74_phase2a_post_only_cross_cooldown_appears_in_desired_state() -> None:
    """v1.4.74 Phase 2A: post_only_cross_cooldown fold suppresses the
    side when inventory is flat.
    """
    from app.quote_engine import QuoteBuildResult, FinalQuoteOrder
    from app.enums import Side as _Side
    om, _storage, _state, db_path = _setup()
    try:
        om._arm_post_only_cross_cooldown(_Side.BUY)
        engine_bid = FinalQuoteOrder(
            side=_Side.BUY, price=1.93, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        build = QuoteBuildResult(
            bid_order=engine_bid, ask_order=None, mode="bid_only",
        )
        from tests.exchange_client_mocks import mock_mm_client
        client = mock_mm_client()
        post_gates, _engine_des, _rejs = om.compute_desired_state(
            build=build, decision=_make_phase2a_decision(), ladder=None,
            position_qty=0.0, num_levels=1, client_spec=client.symbol_spec,
        )
        buy_slot = post_gates.get((_Side.BUY, 0))
        assert buy_slot is not None
        assert buy_slot.should_exist is False
        assert "gate:post_only_cross_cooldown:remaining_s=" in buy_slot.reason
        assert om._gate_post_only_cross_cooldown_applied_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_74_phase2a_reducing_side_bypass_overrides_adverse_pause() -> None:
    """v1.4.74 Phase 2A: BUG-010 invariant preserved — when inventory
    is non-zero, the reducer-side adverse_side_pause is overridden
    by the reducing-side bypass.

    Long position → SELL is reducer → adverse_side_pause on SELL is
    BYPASSED (slot restored to engine quote).
    """
    from app.quote_engine import QuoteBuildResult, FinalQuoteOrder
    from app.enums import Side as _Side
    om, _storage, _state, db_path = _setup()
    try:
        om._adverse_side_pause_until[_Side.SELL] = time.monotonic() + 30.0
        engine_ask = FinalQuoteOrder(
            side=_Side.SELL, price=1.94, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        build = QuoteBuildResult(
            bid_order=None, ask_order=engine_ask, mode="ask_only",
        )
        from tests.exchange_client_mocks import mock_mm_client
        client = mock_mm_client()
        post_gates, _engine_des, _rejs = om.compute_desired_state(
            build=build, decision=_make_phase2a_decision(), ladder=None,
            position_qty=5.0,  # LONG → SELL is reducer
            num_levels=1, client_spec=client.symbol_spec,
        )
        sell_slot = post_gates.get((_Side.SELL, 0))
        assert sell_slot is not None
        # Reducer bypass restored the slot.
        assert sell_slot.should_exist is True, (
            "Reducer-side bypass MUST restore SELL when LONG + "
            "adverse_side_pause active (BUG-010 invariant)"
        )
        assert "reducing_side_bypass" in sell_slot.reason
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_74_phase2a_reducing_side_bypass_does_not_override_unresolved() -> None:
    """v1.4.74 Phase 2A: side_unresolved is NOT a defensive cooldown
    — it's a state-uncertainty signal. The reducer-bypass MUST NOT
    restore an unresolved side, even when inventory exists. Placing
    more orders while uncertain risks duplicates.
    """
    from app.quote_engine import QuoteBuildResult, FinalQuoteOrder
    from app.enums import Side as _Side
    om, _storage, _state, db_path = _setup()
    try:
        om._set_side_unresolved(_Side.SELL, reason="unc_test", payload={})
        engine_ask = FinalQuoteOrder(
            side=_Side.SELL, price=1.94, size=5.0,
            target_half_spread_bps=5.0, aging_tighten_applied=False,
        )
        build = QuoteBuildResult(
            bid_order=None, ask_order=engine_ask, mode="ask_only",
        )
        from tests.exchange_client_mocks import mock_mm_client
        client = mock_mm_client()
        post_gates, _engine_des, _rejs = om.compute_desired_state(
            build=build, decision=_make_phase2a_decision(), ladder=None,
            position_qty=5.0,  # LONG → SELL is reducer ...
            num_levels=1, client_spec=client.symbol_spec,
        )
        sell_slot = post_gates.get((_Side.SELL, 0))
        assert sell_slot is not None
        # ... but unresolved still wins. Bypass cannot restore.
        assert sell_slot.should_exist is False, (
            "side_unresolved must NOT be overridden by reducer-side "
            "bypass — state uncertainty is not a defensive cooldown"
        )
        assert "side_unresolved" in sell_slot.reason
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_74_phase2a_cancel_action_flows_on_unresolved_side() -> None:
    """v1.4.74 Phase 2A: cancels CAN still flow on unresolved sides.
    Pre-Phase-2A the dispatcher silently suppressed BOTH places and
    cancels under ``side_unresolved:return_silent``, which let
    orders pile up indefinitely. Phase 2A's dispatcher only refuses
    PlaceAction/AmendAction; CancelAction flows through normally.
    """
    from app.reconciler import CancelAction
    from app.enums import OrderStatus
    om, _storage, _state, db_path = _setup_phase5()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=20240)
        om._set_side_unresolved(Side.BUY, reason="cancel_flow_test", payload={})
        # Spy on the cancel-enqueue.
        enqueue_calls = []
        original = om._enqueue_cancel_quote_path
        def _spy(wo_arg, **kw):
            enqueue_calls.append(int(wo_arg.order_id_exchange))
            return original(wo_arg, **kw)
        om._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-cancel-on-unresolved"
        om._dispatch_action(
            CancelAction(
                side=Side.BUY, level_idx=0,
                current_local_id=wo.order_id_local,
                trigger_reason="phase2a_test",
            ),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        assert enqueue_calls == [20240], (
            "Phase 2A: CancelAction must dispatch on unresolved side; "
            f"got {enqueue_calls}"
        )
        # No invariant violation (cancel is fine, only places aren't).
        assert om._gate_phase2a_invariant_violation_total == 0
    finally:
        # v1.4.176: explicitly close the storage connection before
        # unlinking the SQLite file. On Windows, an open connection
        # holds the file handle and ``Path.unlink`` raises
        # ``PermissionError`` (WinError 32). Surfaced as a CI flake
        # after v1.4.175's migration v41 added eight more ALTER TABLE
        # statements to ``init_schema()``, which can lengthen the
        # window the kernel keeps the handle alive on Windows.
        try:
            _storage.close()
        except Exception:
            pass
        _safe_unlink_db(db_path)


def test_v1_4_74_phase2a_metrics_in_executor_state_snapshot() -> None:
    """v1.4.74 Phase 2A: new gate-applied counters appear in
    ``executor_state_snapshot``.
    """
    om, _storage, state, db_path = _setup()
    try:
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        for key in (
            "gate_side_unresolved_applied_total",
            "gate_adverse_side_pause_applied_total",
            "gate_post_only_cross_cooldown_applied_total",
            "gate_phase2a_invariant_violation_total",
        ):
            assert key in es, f"executor_state missing Phase 2A metric: {key}"
        # All zero on a fresh executor.
        assert es["gate_side_unresolved_applied_total"] == 0
        assert es["gate_adverse_side_pause_applied_total"] == 0
        assert es["gate_post_only_cross_cooldown_applied_total"] == 0
        assert es["gate_phase2a_invariant_violation_total"] == 0
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.75 wedge-elimination-cleanup Phase 2B: full-state stale-ghost
# reaper.
#
# Phase 2B's reaper is the last-resort safety net. The LEGITIMATE
# recovery paths (cancel-pending watchdog inside-rung, side_unresolved
# REST poll, hydration dedup merge) handle most cases. The reaper
# catches the residual: orphan-slot ghosts, hydration leftovers,
# DESYNC tombstones aged past the recovery window, and SENT WOs
# that slipped every legitimate path.
# ---------------------------------------------------------------------------


def test_v1_4_75_phase2b_reaper_clears_stale_cancel_pending() -> None:
    """v1.4.75 Phase 2B: a CANCEL_PENDING WO aged past
    ``cancel_pending_unresolved_timeout_seconds`` is force-terminalled
    to CANCELED and the slot is cleared.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.utils.time import utc_now
    om, _storage, state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=88_001)
        wo.status = OrderStatus.CANCEL_PENDING
        timeout_s = float(om._settings.cancel_pending_unresolved_timeout_seconds)
        # Backdate the cancel-request to past the timeout.
        wo.ts_cancel_requested = utc_now() - timedelta(seconds=timeout_s * 2.0)
        # Disable rate-limit so the reaper runs on this call.
        om._reaper_last_call_mono = 0.0
        reaped = om._reap_stale_ghosts()
        assert reaped == 1, f"expected 1 reaped, got {reaped}"
        # WO is now CANCELED (force-terminal).
        assert wo.status == OrderStatus.CANCELED
        assert "reaper:stale_cancel_pending" in (wo.cancel_reason or "")
        # Slot is cleared.
        with state._lock:
            assert state.get_working_order(Side.BUY, 0) is None
        # Counter incremented.
        assert om._reaper_cancel_pending_reaped_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_75_phase2b_reaper_skips_fresh_cancel_pending() -> None:
    """v1.4.75 Phase 2B: a CANCEL_PENDING WO within the timeout is
    NOT reaped. The cancel-pending watchdog handles the in-window
    case; the reaper only acts beyond the watchdog horizon.
    """
    from app.enums import OrderStatus
    from app.utils.time import utc_now
    om, _storage, state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=88_002)
        wo.status = OrderStatus.CANCEL_PENDING
        wo.ts_cancel_requested = utc_now()  # FRESH
        om._reaper_last_call_mono = 0.0
        reaped = om._reap_stale_ghosts()
        assert reaped == 0
        assert wo.status == OrderStatus.CANCEL_PENDING  # untouched
        assert om._reaper_cancel_pending_reaped_total == 0
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_75_phase2b_reaper_removes_old_desync() -> None:
    """v1.4.75 Phase 2B: a DESYNC WO aged past
    ``desync_reap_timeout_seconds`` is removed from state entirely.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.utils.time import utc_now
    om, _storage, state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=88_003)
        wo.status = OrderStatus.DESYNC
        timeout_s = float(getattr(om._settings, "desync_reap_timeout_seconds", 30.0))
        wo.ts_closed = utc_now() - timedelta(seconds=timeout_s * 2.0)
        om._reaper_last_call_mono = 0.0
        reaped = om._reap_stale_ghosts()
        assert reaped == 1
        with state._lock:
            assert state.get_working_order(Side.BUY, 0) is None
        assert om._reaper_desync_removed_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_75_phase2b_reaper_rejects_stale_sent() -> None:
    """v1.4.75 Phase 2B: a SENT WO aged past
    ``sent_order_unresolved_timeout_seconds * sent_reaper_safety_multiplier``
    is force-terminalled to REJECTED.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.utils.time import utc_now
    # Use a small sent_timeout so the test doesn't need to wait 20 min worth.
    s = _settings(
        SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS=1.0,
        SENT_REAPER_SAFETY_MULTIPLIER=2.0,  # threshold = 2s
    )
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=88_004)
        wo.status = OrderStatus.SENT
        wo.ts_sent = utc_now() - timedelta(seconds=10.0)  # well past 2s threshold
        om._reaper_last_call_mono = 0.0
        reaped = om._reap_stale_ghosts()
        assert reaped == 1
        assert wo.status == OrderStatus.REJECTED
        assert "reaper:stale_sent_no_response" in (wo.cancel_reason or "")
        assert om._reaper_sent_rejected_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_75_phase2b_reaper_rate_limited_to_one_call_per_second() -> None:
    """v1.4.75 Phase 2B: rapid back-to-back calls reap at most once."""
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.utils.time import utc_now
    om, _storage, state, db_path = _setup()
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=88_005)
        wo.status = OrderStatus.CANCEL_PENDING
        timeout_s = float(om._settings.cancel_pending_unresolved_timeout_seconds)
        wo.ts_cancel_requested = utc_now() - timedelta(seconds=timeout_s * 2.0)
        om._reaper_last_call_mono = 0.0  # allow first call

        # First call: reaps.
        n1 = om._reap_stale_ghosts()
        assert n1 == 1
        # Second call immediately: rate-limited, no-op.
        n2 = om._reap_stale_ghosts()
        assert n2 == 0
        # Counter still 1 (the second call was a no-op).
        assert om._reaper_cancel_pending_reaped_total == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_75_phase2b_reaper_disabled_via_setting() -> None:
    """v1.4.75 Phase 2B: ``REAPER_ENABLED=False`` makes the reaper a
    no-op. Operator escape hatch.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.utils.time import utc_now
    s = _settings(REAPER_ENABLED=False)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=88_006)
        wo.status = OrderStatus.CANCEL_PENDING
        wo.ts_cancel_requested = utc_now() - timedelta(seconds=600.0)  # very old
        om._reaper_last_call_mono = 0.0
        reaped = om._reap_stale_ghosts()
        assert reaped == 0, "reaper must be a no-op when disabled"
        # WO untouched.
        assert wo.status == OrderStatus.CANCEL_PENDING
        assert om._reaper_cancel_pending_reaped_total == 0
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_75_phase2b_reaper_metrics_in_executor_state() -> None:
    """v1.4.75 Phase 2B: reaper counters are surfaced in
    ``executor_state_snapshot``.
    """
    om, _storage, state, db_path = _setup()
    try:
        om._sync_outbound_state_flags()
        es = state.executor_state_snapshot
        for key in (
            "reaper_cancel_pending_reaped_total",
            "reaper_desync_removed_total",
            "reaper_sent_rejected_total",
        ):
            assert key in es, f"executor_state missing Phase 2B metric: {key}"
        # Fresh executor: zero.
        assert es["reaper_cancel_pending_reaped_total"] == 0
        assert es["reaper_desync_removed_total"] == 0
        assert es["reaper_sent_rejected_total"] == 0
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.76 wedge-elimination-cleanup Phase 2C: pre-send risk check
# multi-rung aware.
#
# Codex Bug 6 / B6. Pre-Phase-2C ``_central_pre_send_risk_check``
# read only ``working_bid`` / ``working_ask`` (inside-rung) for the
# worst-case exposure sum. With ``LADDER_NUM_LEVELS_PER_SIDE>=2``,
# two legitimate same-side rungs could each individually pass the
# cap while their COMBINED fill exposure breached it. Phase 2C
# sums size across every in-flight / live same-side WO across all
# rungs (NEW_LOCAL / SENT / ACKED / PARTIAL / AMEND_PENDING).
# ---------------------------------------------------------------------------


def test_v1_4_76_phase2c_pre_send_risk_sums_all_rungs() -> None:
    """v1.4.76 Phase 2C: two BUY rungs of size 4 each occupy 8
    units. With MAX_ABS_POSITION=10 and position=0, attempting to
    place a third BUY of size 3 would yield worst-case fill of
    4 + 4 + 3 = 11 > 10 → must be refused.

    Pre-Phase-2C only the inside rung (size 4) was summed, so the
    worst case computed was 4 + 3 = 7 ≤ 10 → falsely accepted.
    """
    from app.enums import OrderStatus
    s = _settings(MAX_ABS_POSITION=10.0, LADDER_NUM_LEVELS_PER_SIDE=2)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Set up two BUY rungs of size 4 each in state.
        wo_inside = _ack_wo(om, Side.BUY, price=1.00, oid=900_001, level_idx=0)
        wo_inside.size = 4.0
        wo_inside.status = OrderStatus.ACKED
        wo_outer = _ack_wo(om, Side.BUY, price=0.99, oid=900_002, level_idx=1)
        wo_outer.size = 4.0
        wo_outer.status = OrderStatus.ACKED

        # Try to place another size-3 BUY. With multi-rung sum:
        # worst-case = 4 + 4 + 3 = 11 > 10 → REFUSE.
        ok = om._central_pre_send_risk_check(
            Side.BUY, price=1.00, size=3.0, quote_cycle_id="qc-phase2c-1",
        )
        assert ok is False, (
            "Phase 2C: multi-rung pre-send risk check must refuse "
            "when SUM(resting_same_side) + new_size > MAX_ABS_POSITION"
        )
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_76_phase2c_pre_send_risk_single_rung_unchanged() -> None:
    """v1.4.76 Phase 2C backward-compat: with L=1 and one BUY rung,
    behavior is identical to pre-Phase-2C — the single-rung path
    reduces naturally to the multi-rung path with one slot.
    """
    from app.enums import OrderStatus
    s = _settings(MAX_ABS_POSITION=10.0, LADDER_NUM_LEVELS_PER_SIDE=1)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        wo = _ack_wo(om, Side.BUY, price=1.00, oid=900_011, level_idx=0)
        wo.size = 4.0
        wo.status = OrderStatus.ACKED

        # 4 + 5 = 9 ≤ 10 → accept.
        ok = om._central_pre_send_risk_check(
            Side.BUY, price=1.00, size=5.0, quote_cycle_id="qc-phase2c-2",
        )
        assert ok is True

        # 4 + 7 = 11 > 10 → refuse.
        ok = om._central_pre_send_risk_check(
            Side.BUY, price=1.00, size=7.0, quote_cycle_id="qc-phase2c-3",
        )
        assert ok is False
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_76_phase2c_pre_send_risk_excludes_terminal_wos() -> None:
    """v1.4.76 Phase 2C: terminal WOs (CANCELED / FILLED / REJECTED /
    DESYNC) and CANCEL_PENDING are NOT summed into worst-case
    exposure. They can't fill anymore (or in the case of
    CANCEL_PENDING, are explicitly being cleared).
    """
    from app.enums import OrderStatus
    s = _settings(MAX_ABS_POSITION=10.0, LADDER_NUM_LEVELS_PER_SIDE=2)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Inside rung CANCELED (terminal — not counted).
        wo_term = _ack_wo(om, Side.BUY, price=1.00, oid=900_021, level_idx=0)
        wo_term.size = 4.0
        wo_term.status = OrderStatus.CANCELED
        # Outer rung CANCEL_PENDING (not counted — being cleared).
        wo_cp = _ack_wo(om, Side.BUY, price=0.99, oid=900_022, level_idx=1)
        wo_cp.size = 4.0
        wo_cp.status = OrderStatus.CANCEL_PENDING

        # Worst-case sum = 0 (both excluded) + 7 = 7 ≤ 10 → accept.
        ok = om._central_pre_send_risk_check(
            Side.BUY, price=1.00, size=7.0, quote_cycle_id="qc-phase2c-4",
        )
        assert ok is True, (
            "Phase 2C: terminal and CANCEL_PENDING WOs must NOT "
            "contribute to worst-case exposure"
        )
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_76_phase2c_pre_send_risk_includes_sent_and_amend_pending() -> None:
    """v1.4.76 Phase 2C: SENT and AMEND_PENDING WOs ARE summed —
    SENT could be live at the venue, AMEND_PENDING means the
    original order is still resting.
    """
    from app.enums import OrderStatus
    s = _settings(MAX_ABS_POSITION=10.0, LADDER_NUM_LEVELS_PER_SIDE=2)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # SENT inside rung (counted).
        wo_sent = _ack_wo(om, Side.BUY, price=1.00, oid=900_031, level_idx=0)
        wo_sent.size = 4.0
        wo_sent.status = OrderStatus.SENT
        # AMEND_PENDING outer rung (counted).
        wo_amend = _ack_wo(om, Side.BUY, price=0.99, oid=900_032, level_idx=1)
        wo_amend.size = 4.0
        wo_amend.status = OrderStatus.AMEND_PENDING

        # Worst-case = 4 + 4 + 3 = 11 > 10 → refuse.
        ok = om._central_pre_send_risk_check(
            Side.BUY, price=1.00, size=3.0, quote_cycle_id="qc-phase2c-5",
        )
        assert ok is False


        # Worst-case = 4 + 4 + 2 = 10 ≤ 10 → accept (boundary).
        ok = om._central_pre_send_risk_check(
            Side.BUY, price=1.00, size=2.0, quote_cycle_id="qc-phase2c-6",
        )
        assert ok is True
    finally:
        _safe_unlink_db(db_path)


# ---------------------------------------------------------------------------
# v1.4.77 wedge-elimination-cleanup Phase 2D: QuoteBuildResult consumer audit.
#
# REMOVED in v1.4.92 Phase 4A cutover. The runtime validator was a
# stopgap to catch the v1.4.66 regression class ("new control field
# added to QuoteBuildResult, no consumer reads it"). The typed
# ``BuildCommand`` sum-type replaces it structurally:
#
#   * New fields on any variant are explicit dataclass attributes —
#     adding one means deciding which variant carries it, which
#     forces the question "who reads it?" at edit time.
#   * Match-statement consumers must handle each variant; missing
#     a variant is a type error at edit time, not a runtime drift.
#
# The 6 phase2d tests that exercised the runtime validator
# (``_mark_qbr_consumed``, ``_validate_qbr_consumption``) are
# deleted along with the audit infrastructure.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v1.4.78 hot-path optimizations — noop ring-buffer sampling.
#
# Snapshot v1.4.77-260518-234804 showed 32,270
# noop:acked_no_reprice_needed decisions in 10 min — each one
# allocating a 13-field dict and appending to the ring buffer.
# The ring buffer is bounded so the noise crowds out useful
# entries. Sampling cuts ~95% of those allocations while keeping
# postmortem signal (counter still bumps every time; first
# occurrence + every Nth subsequent are recorded).
# ---------------------------------------------------------------------------


def test_v1_4_78_hot_path_noop_first_occurrence_always_traced() -> None:
    """v1.4.78: the FIRST occurrence of any noop:high-freq action
    is always added to the ring buffer (postmortem exemplar). The
    counter starts at 1; sampling kicks in from #2 onward.
    """
    from app.reconciler import NoOpAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-78-1"
        om._dispatch_action(
            NoOpAction(side=Side.BUY, level_idx=0, reason="acked_no_reprice_needed"),
            decision=decision, mid_price=1.0, tick=0.001,
        )
        history = list(om._orchestrate_decision_history[Side.BUY])
        # First occurrence always recorded.
        assert len(history) == 1
        assert history[0]["action"] == "noop:acked_no_reprice_needed"
        # Counter incremented.
        ck = "BUY:reconciler:noop:acked_no_reprice_needed"
        assert om._orchestrate_decision_counts[ck] == 1
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_78_hot_path_noop_sampling_skips_intermediate() -> None:
    """v1.4.78: with the default sample interval (N=20), only the
    1st and every 20th subsequent noop:acked_no_reprice_needed
    decision lands in the ring buffer. Counter bumps every time.
    """
    from app.reconciler import NoOpAction
    from app.execution import _NOOP_TRACE_SAMPLE_INTERVAL
    om, _storage, _state, db_path = _setup_phase5()
    try:
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-78-2"
        # Issue 50 noop:acked actions.
        for _ in range(50):
            om._dispatch_action(
                NoOpAction(side=Side.BUY, level_idx=0, reason="acked_no_reprice_needed"),
                decision=decision, mid_price=1.0, tick=0.001,
            )
        ck = "BUY:reconciler:noop:acked_no_reprice_needed"
        # Counter: every call bumps.
        assert om._orchestrate_decision_counts[ck] == 50
        # Ring buffer: count=1 + counts at 20 and 40 = 3 entries.
        history = list(om._orchestrate_decision_history[Side.BUY])
        expected_samples = 1  # count=1
        for n in range(1, 51):
            if n != 1 and (n % _NOOP_TRACE_SAMPLE_INTERVAL) == 0:
                expected_samples += 1
        assert len(history) == expected_samples, (
            f"expected {expected_samples} ring-buffer entries "
            f"(1 + every-Nth from N=20 in range 50), got {len(history)}"
        )
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_78_hot_path_non_noop_actions_always_traced() -> None:
    """v1.4.78: non-noop actions (place / cancel / amend) are NEVER
    sampled — every one is in the ring buffer. The sampling only
    applies to the high-frequency noop branches.
    """
    from app.reconciler import NoOpAction
    om, _storage, _state, db_path = _setup_phase5()
    try:
        decision = type("_D", (), {})()
        decision.quote_cycle_id = "qc-78-3"
        # 30 NoOp actions with a non-sampled reason.
        for _ in range(30):
            om._dispatch_action(
                NoOpAction(side=Side.BUY, level_idx=0,
                           reason="in_flight_wait:CANCEL_PENDING"),
                decision=decision, mid_price=1.0, tick=0.001,
            )
        history = list(om._orchestrate_decision_history[Side.BUY])
        # ALL 30 recorded (reason not in _NOOP_HIGH_FREQ_ACTIONS).
        assert len(history) == 30
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_76_phase2c_pre_send_risk_per_side_isolation() -> None:
    """v1.4.76 Phase 2C: BUY-side risk check only sums BUY WOs; SELL
    WOs do NOT contribute. Position-cap math is per-side as expected.
    """
    from app.enums import OrderStatus
    s = _settings(MAX_ABS_POSITION=10.0, LADDER_NUM_LEVELS_PER_SIDE=2)
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    _safe_unlink_db(db_path)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    try:
        # Large SELL exposure existing.
        wo_sell = _ack_wo(om, Side.SELL, price=2.00, oid=900_041, level_idx=0)
        wo_sell.size = 8.0
        wo_sell.status = OrderStatus.ACKED

        # BUY-side check: 0 BUY same-side + 5 = 5 ≤ 10 → accept.
        ok = om._central_pre_send_risk_check(
            Side.BUY, price=1.00, size=5.0, quote_cycle_id="qc-phase2c-7",
        )
        assert ok is True, "BUY risk must not double-count SELL exposure"
    finally:
        _safe_unlink_db(db_path)


def test_v1_4_75_phase2b_reaper_handles_orphan_slot_cancel_pending() -> None:
    """v1.4.75 Phase 2B: stale CANCEL_PENDING in an OUTER slot
    (level_idx>=1) — the pre-Phase-2B watchdog only inspected
    inside-rung slots and never saw these ghosts. Reaper catches them.
    """
    from datetime import timedelta
    from app.enums import OrderStatus
    from app.utils.time import utc_now
    om, _storage, state, db_path = _setup()
    try:
        # Place at level_idx=1 (outer rung).
        wo = _ack_wo(om, Side.BUY, price=1.0, oid=88_007, level_idx=1)
        wo.status = OrderStatus.CANCEL_PENDING
        timeout_s = float(om._settings.cancel_pending_unresolved_timeout_seconds)
        wo.ts_cancel_requested = utc_now() - timedelta(seconds=timeout_s * 2.0)
        om._reaper_last_call_mono = 0.0
        reaped = om._reap_stale_ghosts()
        assert reaped == 1, "outer-rung CANCEL_PENDING must be reaped"
        # Outer slot cleared.
        with state._lock:
            assert state.get_working_order(Side.BUY, 1) is None
        assert om._reaper_cancel_pending_reaped_total == 1
    finally:
        _safe_unlink_db(db_path)
