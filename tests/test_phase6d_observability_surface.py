"""v1.4.93 wedge-elimination-cleanup Phase 6D — observability
surface tests.

The plan's section 6D promises tests that pin the operator-facing
observability surface:

* The executor_state_snapshot exposes the Phase 1/2/3 wedge metrics
  the postmortem needs.
* The v1.4.92 Phase 4A cutover removed Phase 2D fields cleanly.
* `TelemetryStore.is_healthy()` aggregates the right signals.

These tests are the contract pin for "what fields must remain
present in the executor surface". If a future refactor renames or
removes one of these, the test fails LOUD — protecting the dashboard
/ postmortem tools that depend on them.

Deferred to a future session:
* 6D.2 (postmortem payload on silent_wedge fire) — needs a higher-
  level harness that triggers the silent_wedge detector with full
  payload assertion. Out of minimal scope.
* 6D.3 (Telegram /status includes wedge_episode_count_session) —
  needs Telegram surface mocking + a wedge counter to be added
  (none of the existing counters is named ``wedge_episode_count``;
  this would be a new metric).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings():
    return UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_p6d_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
        "MAX_ABS_POSITION": 10.0,
    })


def _setup() -> tuple[OrderManager, BotState]:
    s = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om._sync_outbound_state_flags()
    return om, state


# ---------------------------------------------------------------------------
# 6D.1 — Phase 1/2/3 metrics are present in executor_state_snapshot
# ---------------------------------------------------------------------------


# The fields that MUST be present in the executor surface. Each maps to
# a specific phase / wedge-class that the operator-facing telemetry
# tracks. Removing or renaming one breaks the postmortem.
REQUIRED_PHASE_1_2_3_METRIC_KEYS = {
    # Phase 1A — WS-event matcher race detection.
    "ws_event_unmatched_to_local_wo_total",
    # Phase 1B — hydration dedup counter.
    "hydration_merged_existing_total",
    # Phase 2B — stale-ghost reaper.
    "reaper_cancel_pending_reaped_total",
    "reaper_desync_removed_total",
    "reaper_sent_rejected_total",
    # Phase 2A — bypass-invariant detector.
    "gate_phase2a_invariant_violation_total",
    # Risk state surface (used by silent-wedge detector, postmortem).
    "risk_exec_state",
    # Phase 2A — side-unresolved tracking.
    "side_unresolved_enter_count",
    # Phase 5C trace-mode-related: execution_idle_s feeds the
    # silent-wedge detector's "should we be quoting?" guard.
    "execution_idle_s",
}


def test_phase6d_executor_state_exposes_required_phase_1_2_3_metrics() -> None:
    """The executor_state_snapshot dict surface MUST expose every
    metric the postmortem / dashboard depends on. If a future refactor
    drops or renames one, this test fails LOUD."""
    om, state = _setup()
    es = state.executor_state_snapshot or {}
    missing = REQUIRED_PHASE_1_2_3_METRIC_KEYS - set(es.keys())
    assert not missing, (
        f"executor_state_snapshot is missing required keys: {sorted(missing)}. "
        f"Got keys: {sorted(es.keys())}. "
        f"Dashboards / postmortem tools depend on these — removing or renaming "
        f"breaks them. If intentional, update REQUIRED_PHASE_1_2_3_METRIC_KEYS "
        f"in this test."
    )


def test_phase6d_executor_state_metrics_are_zero_on_fresh_setup() -> None:
    """A freshly-constructed OrderManager has all wedge counters at 0
    and risk_exec_state in a healthy state. If non-zero on init, the
    constructor is leaking state."""
    om, state = _setup()
    es = state.executor_state_snapshot or {}
    for k in (
        "ws_event_unmatched_to_local_wo_total",
        "hydration_merged_existing_total",
        "reaper_cancel_pending_reaped_total",
        "reaper_desync_removed_total",
        "reaper_sent_rejected_total",
        "gate_phase2a_invariant_violation_total",
    ):
        assert int(es.get(k, 0)) == 0, (
            f"fresh OrderManager has nonzero {k}={es.get(k)}; "
            f"constructor is leaking state"
        )
    assert str(es.get("risk_exec_state", "")).upper() in ("NORMAL", "UNKNOWN", ""), (
        f"fresh OrderManager risk_exec_state should be NORMAL/UNKNOWN, "
        f"got {es.get('risk_exec_state')!r}"
    )


# ---------------------------------------------------------------------------
# 6D.1 (negative pin) — v1.4.92 Phase 4A cutover removed the Phase 2D field
# ---------------------------------------------------------------------------


def test_phase6d_phase2d_quote_build_result_unconsumed_field_total_removed() -> None:
    """v1.4.92 Phase 4A cutover deleted the Phase 2D audit
    infrastructure. The ``quote_build_result_unconsumed_field_total``
    counter is gone from the executor surface. Dashboards / postmortem
    tools that consumed this key must drop it.

    This test pins the REMOVAL — if a future change accidentally
    re-introduces the field, the test fails LOUD so the operator
    knows the v1.4.92 cleanup was undone."""
    om, state = _setup()
    es = state.executor_state_snapshot or {}
    assert "quote_build_result_unconsumed_field_total" not in es, (
        "quote_build_result_unconsumed_field_total was removed in v1.4.92 "
        "(Phase 4A cutover). Its presence indicates the Phase 2D audit "
        "was re-introduced — investigate."
    )


# ---------------------------------------------------------------------------
# 6D — TelemetryStore.is_healthy() aggregate
# ---------------------------------------------------------------------------


def test_phase6d_telemetry_store_is_healthy_on_fresh_state() -> None:
    """A fresh BotState (no counters non-zero, no risk events) is
    healthy according to TelemetryStore.is_healthy()."""
    om, state = _setup()
    assert state.telemetry_store.is_healthy() is True


def test_phase6d_telemetry_store_unhealthy_when_ws_unmatched_nonzero() -> None:
    """Mark a wedge counter non-zero — is_healthy() must return False.
    This is the Phase 1A wedge signal."""
    om, state = _setup()
    state.executor_state_snapshot = dict(state.executor_state_snapshot or {})
    state.executor_state_snapshot["ws_event_unmatched_to_local_wo_total"] = 1
    assert state.telemetry_store.is_healthy() is False


def test_phase6d_telemetry_store_unhealthy_when_phase2a_violation() -> None:
    """Phase 2A invariant violation triggers unhealthy."""
    om, state = _setup()
    state.executor_state_snapshot = dict(state.executor_state_snapshot or {})
    state.executor_state_snapshot["gate_phase2a_invariant_violation_total"] = 1
    assert state.telemetry_store.is_healthy() is False


def test_phase6d_telemetry_store_unhealthy_when_risk_not_normal() -> None:
    """Risk state non-NORMAL → unhealthy. UNKNOWN is fine (startup);
    SUPPRESSED / CANCELLING / DEGRADED are not."""
    om, state = _setup()
    state.executor_state_snapshot = dict(state.executor_state_snapshot or {})
    state.executor_state_snapshot["risk_exec_state"] = "SUPPRESSED"
    assert state.telemetry_store.is_healthy() is False


def test_phase6d_telemetry_store_unhealthy_when_reaper_fired() -> None:
    """Any reaper firing → unhealthy. Phase 2B reaper counters
    aggregate (cancel_pending + desync_removed + sent_rejected)."""
    om, state = _setup()
    state.executor_state_snapshot = dict(state.executor_state_snapshot or {})
    state.executor_state_snapshot["reaper_cancel_pending_reaped_total"] = 1
    assert state.telemetry_store.is_healthy() is False

    om, state = _setup()
    state.executor_state_snapshot = dict(state.executor_state_snapshot or {})
    state.executor_state_snapshot["reaper_desync_removed_total"] = 1
    assert state.telemetry_store.is_healthy() is False


# ---------------------------------------------------------------------------
# 6D — TelemetryStore facade exposes the right reader API
# ---------------------------------------------------------------------------


def test_phase6d_telemetry_store_exposes_required_accessors() -> None:
    """The TelemetryStore facade exposes named accessors for each
    metric — callers shouldn't poke at the raw dict. This test pins
    the accessor API contract."""
    om, state = _setup()
    ts = state.telemetry_store
    # Each accessor returns a value of the right type without raising.
    assert isinstance(ts.ws_unmatched_total(), int)
    assert isinstance(ts.hydration_merged_total(), int)
    assert isinstance(ts.reaper_total(), int)
    assert isinstance(ts.phase2a_invariant_violations(), int)
    assert isinstance(ts.risk_exec_state(), str)
    assert isinstance(ts.is_healthy(), bool)


def test_phase6d_telemetry_store_qbr_accessor_removed() -> None:
    """v1.4.92 Phase 4A cutover removed ``TelemetryStore.qbr_unconsumed_total()``.
    Verify the method is gone — callers of the old API must update."""
    om, state = _setup()
    ts = state.telemetry_store
    assert not hasattr(ts, "qbr_unconsumed_total"), (
        "TelemetryStore.qbr_unconsumed_total() should be removed in v1.4.92 "
        "(Phase 4A cutover); the typed BuildCommand replaces the runtime audit"
    )
