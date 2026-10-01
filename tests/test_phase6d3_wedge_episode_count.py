"""v1.4.93 wedge-elimination-cleanup Phase 6D.3 —
``wedge_episode_count_session`` metric + Telegram surface.

The metric counts entries into a non-trivial risk-exec state
(CANCELLING / SUPPRESSED) from NORMAL. Re-arming transitions
(staying in CANCELLING across ticks) do NOT count. Operator's
one-line answer to "did the bot wedge at all this session?".

Wired:
* Derived in ``OrderManager._build_executor_state_snapshot`` from
  the existing ``_risk_exec_state_transition_counts`` dict. No new
  state field on OrderManager.
* Surfaced in ``BotState.status_flags_dict`` (reads from
  ``executor_state_snapshot``).
* Rendered as a row in Telegram ``/status`` (``wedge_episodes:N``).

Tests verify:
* Fresh session: counter is 0.
* NORMAL→CANCELLING transition: counter increments to 1.
* CANCELLING→CANCELLING re-arm: counter stays at 1.
* CANCELLING→NORMAL→CANCELLING: counter increments to 2.
* NORMAL→SUPPRESSED also counts.
* Surfaced in executor_state_snapshot AND status_flags_dict.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import RiskAction, RiskExecState
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
            / f"mm_p6d3_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
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


def _refresh_snapshot(om: OrderManager) -> None:
    """Re-sync the executor_state_snapshot after manipulating
    OrderManager state. The Telegram surface and status_flags_dict
    read from the snapshot, not from OrderManager directly."""
    om._sync_outbound_state_flags()


def test_phase6d3_fresh_session_wedge_count_is_zero() -> None:
    """A freshly-constructed bot has zero wedge episodes."""
    om, state = _setup()
    es = state.executor_state_snapshot or {}
    assert es.get("wedge_episode_count_session", -1) == 0


def test_phase6d3_normal_to_cancelling_increments_count() -> None:
    """A NORMAL→CANCELLING transition is one wedge episode."""
    om, state = _setup()
    om._transition_risk_exec_state(
        RiskExecState.CANCELLING,
        reason="test_cancel_all",
        risk_action=RiskAction.CANCEL_ALL,
    )
    _refresh_snapshot(om)
    es = state.executor_state_snapshot or {}
    assert es.get("wedge_episode_count_session") == 1, (
        f"NORMAL→CANCELLING should bump wedge count to 1; "
        f"got {es.get('wedge_episode_count_session')}"
    )


def test_phase6d3_cancelling_to_cancelling_re_arm_does_not_count() -> None:
    """Re-arming the cancelling state (CANCELLING→CANCELLING, which
    is a no-op in the transition function) must NOT count."""
    om, state = _setup()
    om._transition_risk_exec_state(
        RiskExecState.CANCELLING, reason="first", risk_action=RiskAction.CANCEL_ALL,
    )
    # The transition function early-returns if new_state == current.
    # No counter bump, even if called repeatedly.
    om._transition_risk_exec_state(
        RiskExecState.CANCELLING, reason="re-arm", risk_action=RiskAction.CANCEL_ALL,
    )
    om._transition_risk_exec_state(
        RiskExecState.CANCELLING, reason="re-arm-2", risk_action=RiskAction.CANCEL_ALL,
    )
    _refresh_snapshot(om)
    es = state.executor_state_snapshot or {}
    assert es.get("wedge_episode_count_session") == 1


def test_phase6d3_cycle_back_to_normal_then_cancelling_again_counts_two() -> None:
    """NORMAL → CANCELLING → NORMAL → CANCELLING is two episodes."""
    om, state = _setup()
    om._transition_risk_exec_state(
        RiskExecState.CANCELLING, reason="ep1", risk_action=RiskAction.CANCEL_ALL,
    )
    om._transition_risk_exec_state(
        RiskExecState.NORMAL, reason="back_to_normal", risk_action=RiskAction.ALLOW,
    )
    om._transition_risk_exec_state(
        RiskExecState.CANCELLING, reason="ep2", risk_action=RiskAction.CANCEL_ALL,
    )
    _refresh_snapshot(om)
    es = state.executor_state_snapshot or {}
    assert es.get("wedge_episode_count_session") == 2


def test_phase6d3_normal_to_suppressed_also_counts() -> None:
    """SUPPRESSED is also a wedge episode (the risk state machine's
    other non-NORMAL state per the plan)."""
    om, state = _setup()
    om._transition_risk_exec_state(
        RiskExecState.SUPPRESSED,
        reason="ws_unhealthy",
        risk_action=None,
    )
    _refresh_snapshot(om)
    es = state.executor_state_snapshot or {}
    assert es.get("wedge_episode_count_session") == 1


def test_phase6d3_status_flags_dict_exposes_wedge_count() -> None:
    """The metric is in ``status_flags_dict`` so Telegram /status
    can render it without a second snapshot call."""
    om, state = _setup()
    om._transition_risk_exec_state(
        RiskExecState.CANCELLING, reason="t", risk_action=RiskAction.CANCEL_ALL,
    )
    _refresh_snapshot(om)
    flags = state.status_flags_dict()
    assert "wedge_episode_count_session" in flags
    assert flags["wedge_episode_count_session"] == 1


def test_phase6d3_status_flags_dict_default_zero_when_snapshot_empty() -> None:
    """Defensive: status_flags_dict returns 0 when executor_state_snapshot
    is empty / not yet populated (e.g. startup window)."""
    om, state = _setup()
    # Clear the snapshot to simulate "not yet populated".
    state.executor_state_snapshot = {}
    flags = state.status_flags_dict()
    assert flags.get("wedge_episode_count_session") == 0
