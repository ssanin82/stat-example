"""Tests for v1.4.89 Phase 5C — trace sampling modes.

Phase 5C adds the ``executor_trace_mode`` operator knob:

  * ``always``        — record every decision (cumulative counters +
                        ring buffer + log line); v1.4.78 noop-sampling
                        still applies.
  * ``incident_only`` — same as ``always``, EXCEPT high-frequency noop
                        actions are 100% SKIPPED when no incident is
                        active. Cancels / places / amends / errors /
                        non-NORMAL risk_exec_state always trace.
  * ``off``           — skip ring buffer + log for ALL actions
                        (counter still increments).

Auto-elevation: when a silent_wedge fires, ``incident_only`` behaves
as ``always`` for 5 minutes after the fire. Restores full visibility
during incidents.

Tests pin:

* The setting accepts the three valid values + rejects others.
* The `_trace_mode_incident_active` helper returns True only when
  silent_wedge is recent OR risk_exec_state is non-NORMAL.
* The trace-mode gate fires correctly in each mode for noop vs
  non-noop actions.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> "UnitTestSettings":  # type: ignore[name-defined]
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_phase5c_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


# ---------------------------------------------------------------------------
# Setting validation
# ---------------------------------------------------------------------------


def test_phase5c_setting_accepts_always() -> None:
    s = _settings(EXECUTOR_TRACE_MODE="always")
    assert s.executor_trace_mode == "always"


def test_phase5c_setting_accepts_incident_only() -> None:
    s = _settings(EXECUTOR_TRACE_MODE="incident_only")
    assert s.executor_trace_mode == "incident_only"


def test_phase5c_setting_accepts_off() -> None:
    s = _settings(EXECUTOR_TRACE_MODE="off")
    assert s.executor_trace_mode == "off"


def test_phase5c_setting_default_is_always() -> None:
    """Default preserves v1.4.78 behavior — operator opts INTO
    incident_only for prod."""
    s = _settings()
    assert s.executor_trace_mode == "always"


def test_phase5c_setting_rejects_invalid_value() -> None:
    with pytest.raises(Exception):  # pydantic ValidationError
        _settings(EXECUTOR_TRACE_MODE="verbose")


# ---------------------------------------------------------------------------
# Incident-active detector
# ---------------------------------------------------------------------------


def _stub_om(risk_state: str = "NORMAL", last_wedge_mono: float = 0.0):
    """Build a minimal stub with just the fields _trace_mode_incident_active reads.
    Avoids the full OrderManager constructor which is heavy."""
    from app.execution import OrderManager

    om = OrderManager.__new__(OrderManager)
    om._silent_wedge_diag_last_log_mono = last_wedge_mono
    # Mock the telemetry store path
    telemetry_store = MagicMock()
    telemetry_store.risk_exec_state.return_value = risk_state
    state = MagicMock()
    state.telemetry_store = telemetry_store
    om._state = state
    om._TRACE_MODE_INCIDENT_WINDOW_S = 300.0
    # Phase 1c (v1.4.231) — OrderManager.__init__ normally installs
    # self._clock, but ``__new__`` bypass requires we set it manually.
    from app.clock import SystemClock
    om._clock = SystemClock()
    return om


def test_phase5c_incident_inactive_in_normal_state() -> None:
    om = _stub_om(risk_state="NORMAL", last_wedge_mono=0.0)
    assert om._trace_mode_incident_active() is False


def test_phase5c_incident_active_when_risk_state_non_normal() -> None:
    om = _stub_om(risk_state="SUPPRESSED", last_wedge_mono=0.0)
    assert om._trace_mode_incident_active() is True
    om2 = _stub_om(risk_state="DEGRADED", last_wedge_mono=0.0)
    assert om2._trace_mode_incident_active() is True


def test_phase5c_incident_inactive_when_state_is_unknown() -> None:
    """UNKNOWN is treated as healthy — telemetry store returns UNKNOWN
    when executor_state_snapshot is empty, which happens during
    startup. Don't auto-elevate trace during startup."""
    om = _stub_om(risk_state="UNKNOWN", last_wedge_mono=0.0)
    assert om._trace_mode_incident_active() is False


def test_phase5c_incident_active_when_silent_wedge_recent() -> None:
    """Wedge fired 60s ago → still in 300s window → active."""
    now = time.monotonic()
    om = _stub_om(risk_state="NORMAL", last_wedge_mono=now - 60.0)
    assert om._trace_mode_incident_active() is True


def test_phase5c_incident_inactive_when_silent_wedge_outside_window() -> None:
    """Wedge fired 400s ago → outside 300s window → inactive."""
    now = time.monotonic()
    om = _stub_om(risk_state="NORMAL", last_wedge_mono=now - 400.0)
    assert om._trace_mode_incident_active() is False


def test_phase5c_incident_active_at_window_boundary() -> None:
    """Wedge fired exactly 299s ago — still active. At 301s → inactive."""
    now = time.monotonic()
    om_in = _stub_om(risk_state="NORMAL", last_wedge_mono=now - 299.0)
    assert om_in._trace_mode_incident_active() is True
    om_out = _stub_om(risk_state="NORMAL", last_wedge_mono=now - 301.0)
    assert om_out._trace_mode_incident_active() is False


# ---------------------------------------------------------------------------
# Gate behavior — counter always increments, ring buffer respects mode
# ---------------------------------------------------------------------------


def _stub_om_for_record(trace_mode: str, risk_state: str = "NORMAL",
                        last_wedge_mono: float = 0.0):
    """Stub an OrderManager just enough that _record_orchestrate_decision
    runs through the early gating logic without needing real exchange
    state."""
    from collections import deque
    from app.clock import SystemClock
    from app.execution import OrderManager

    om = OrderManager.__new__(OrderManager)
    om._orchestrate_decision_counts = {}
    om._orchestrate_decision_history = {
        "BUY": deque(maxlen=200),
        "SELL": deque(maxlen=200),
    }
    om._silent_wedge_diag_last_log_mono = last_wedge_mono
    om._TRACE_MODE_INCIDENT_WINDOW_S = 300.0
    settings = MagicMock()
    settings.executor_trace_mode = trace_mode
    settings.executor_decision_trace_enabled = False  # skip the log path
    om._settings = settings
    telemetry_store = MagicMock()
    telemetry_store.risk_exec_state.return_value = risk_state
    state = MagicMock()
    state.telemetry_store = telemetry_store
    om._state = state
    # Phase 1c (v1.4.231) — OrderManager.__init__ normally installs
    # self._clock, but ``__new__`` bypass requires we set it manually.
    om._clock = SystemClock()
    return om


def _ring_buffer_size(om) -> int:
    return sum(len(d) for d in om._orchestrate_decision_history.values())


def test_phase5c_off_mode_skips_ring_buffer_for_all_actions() -> None:
    """In ``off`` mode, no decisions hit the ring buffer regardless of
    action. The counter still increments (postmortem aggregates
    preserved)."""
    from app.enums import Side
    om = _stub_om_for_record(trace_mode="off")
    # Try a non-noop action (placed_fresh) — should still skip in off mode.
    om._record_orchestrate_decision(
        side=Side.BUY, level_idx=0, cur=None, desired=None,
        decision_branch="terminal_fresh_place", action="placed_fresh",
    )
    assert _ring_buffer_size(om) == 0
    # Counter MUST still increment.
    assert om._orchestrate_decision_counts.get("BUY:terminal_fresh_place:placed_fresh") == 1


def test_phase5c_off_mode_counter_increments() -> None:
    from app.enums import Side
    om = _stub_om_for_record(trace_mode="off")
    for _ in range(5):
        om._record_orchestrate_decision(
            side=Side.SELL, level_idx=0, cur=None, desired=None,
            decision_branch="any_branch", action="noop:empty_match",
        )
    assert om._orchestrate_decision_counts.get("SELL:any_branch:noop:empty_match") == 5
    assert _ring_buffer_size(om) == 0


def test_phase5c_incident_only_skips_noop_when_no_incident() -> None:
    """In ``incident_only`` mode with risk=NORMAL and no recent wedge,
    high-freq noop actions are 100% skipped (not even sampled)."""
    from app.enums import Side
    om = _stub_om_for_record(trace_mode="incident_only", risk_state="NORMAL")
    for _ in range(50):
        om._record_orchestrate_decision(
            side=Side.BUY, level_idx=0, cur=None, desired=None,
            decision_branch="branch", action="noop:empty_match",
        )
    # Counter saw all 50, but ring buffer skipped them all.
    assert om._orchestrate_decision_counts.get("BUY:branch:noop:empty_match") == 50
    assert _ring_buffer_size(om) == 0


def test_phase5c_incident_only_elevates_to_always_during_wedge() -> None:
    """When silent_wedge is recent, ``incident_only`` mode behaves as
    ``always`` — the v1.4.78 noop sampling kicks back in (first
    occurrence + every Nth)."""
    from app.enums import Side
    now = time.monotonic()
    om = _stub_om_for_record(
        trace_mode="incident_only",
        risk_state="NORMAL",
        last_wedge_mono=now - 30.0,  # wedge fired 30s ago — well within 5min window
    )
    om._record_orchestrate_decision(
        side=Side.BUY, level_idx=0, cur=None, desired=None,
        decision_branch="branch", action="noop:empty_match",
    )
    # First occurrence always recorded under v1.4.78 sampling.
    assert _ring_buffer_size(om) == 1


def test_phase5c_incident_only_traces_non_noop_actions() -> None:
    """Cancels / places / amends / errors must always trace regardless
    of incident state in ``incident_only`` mode."""
    from app.enums import Side
    om = _stub_om_for_record(trace_mode="incident_only", risk_state="NORMAL")
    om._record_orchestrate_decision(
        side=Side.SELL, level_idx=0, cur=None, desired=None,
        decision_branch="terminal_fresh_place", action="placed_fresh",
    )
    assert _ring_buffer_size(om) == 1


def test_phase5c_always_mode_uses_v1_4_78_sampling() -> None:
    """In ``always`` mode (the default), the v1.4.78 sampling logic is
    active: first noop occurrence is recorded; subsequent ones at the
    sample interval."""
    from app.enums import Side
    om = _stub_om_for_record(trace_mode="always")
    # Fire 25 noop:empty_match — sample interval is 20, so first + 20th
    # should hit the ring buffer = 2 entries.
    for _ in range(25):
        om._record_orchestrate_decision(
            side=Side.BUY, level_idx=0, cur=None, desired=None,
            decision_branch="branch", action="noop:empty_match",
        )
    # First (count==1) + 20th (count==20) hit the buffer.
    assert _ring_buffer_size(om) == 2
