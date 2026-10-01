"""v1.4.90 wedge-elimination-cleanup FU-2 regression.

The ``_maybe_emit_engine_no_quote_diagnostic`` warning fires when the
engine returns ``mode="no_quote"`` for a persistent streak of ticks
(default 20). It's meant to catch silent wedges — "we SHOULD be
quoting but aren't".

But during ``desync_phase != OK`` the engine LEGITIMATELY holds
quotes (both sides return ``side_not_requested``) while the reconciler
finishes recovery. That's correct behavior, NOT a silent wedge.

Pre-FU-2 the warning fired in that window — snapshot
v1.4.85-260519-104553 caught this at 06:26:44 during legitimate
desync recovery from the FU-1 drift event. False positive.

FU-2 adds a one-liner guard: when ``state.desync_phase != "OK"``,
suppress the warning AND reset the streak counter so fresh
no-quote ticks need to accumulate from zero after recovery clears.

Tests verify:

* Streak resets to 0 and warning is suppressed when desync_phase != OK.
* The warning fires normally when desync_phase == OK.
* After desync clears, the streak starts fresh (no carry-over).
"""

from __future__ import annotations

from collections import deque
from unittest.mock import MagicMock

from app.enums import DesyncPhase


def _stub_om(desync_phase: DesyncPhase | str | None):
    """Stub the minimum surface ``_maybe_emit_engine_no_quote_diagnostic``
    reads. Skips the full OrderManager constructor."""
    from app.clock import SystemClock
    from app.execution import OrderManager
    om = OrderManager.__new__(OrderManager)
    om._clock = SystemClock()
    om._engine_no_quote_streak_ticks = 0
    om._engine_no_quote_diag_last_log_mono = 0.0
    om._side_unresolved_active = {}
    om._side_unresolved_reason = {}
    om._side_unresolved_requires_confirm = {}
    settings = MagicMock()
    settings.engine_no_quote_diag_streak_ticks = 5  # low threshold for fast test
    settings.engine_no_quote_diag_relog_seconds = 30.0
    om._settings = settings
    state = MagicMock()
    state.desync_phase = desync_phase
    om._state = state
    om._silent_wedge_event_payload_persisted = False
    return om


def _telemetry_no_quote() -> dict:
    return {"quote_engine_mode": "no_quote"}


def test_fu2_warning_fires_in_desync_ok_state() -> None:
    """Baseline: with desync_phase=OK, the warning fires after 5 ticks
    of no_quote (the configured threshold for this test)."""
    om = _stub_om(desync_phase=DesyncPhase.OK)
    for _ in range(5):
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_no_quote(),
            decision_quote_cycle_id="cycle-x",
            position_qty=0.0, position_notional=0.0,
        )
    # Streak reached threshold; the diag fire timestamp is set.
    assert om._engine_no_quote_streak_ticks == 5
    assert om._engine_no_quote_diag_last_log_mono > 0.0


def test_fu2_warning_suppressed_during_desync_detected() -> None:
    """FU-2 contract: during DESYNC_PHASE=DETECTED, no_quote ticks do
    NOT advance the streak — the warning never fires."""
    om = _stub_om(desync_phase=DesyncPhase.DETECTED)
    for _ in range(20):
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_no_quote(),
            decision_quote_cycle_id="cycle-x",
            position_qty=0.0, position_notional=0.0,
        )
    assert om._engine_no_quote_streak_ticks == 0
    assert om._engine_no_quote_diag_last_log_mono == 0.0


def test_fu2_warning_suppressed_during_desync_reconciling() -> None:
    om = _stub_om(desync_phase=DesyncPhase.RECONCILING)
    for _ in range(20):
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_no_quote(),
            decision_quote_cycle_id="cycle-x",
            position_qty=0.0, position_notional=0.0,
        )
    assert om._engine_no_quote_streak_ticks == 0


def test_fu2_warning_suppressed_during_desync_recovered() -> None:
    """RECOVERED is a transitional state on the way back to OK. The
    engine may still be re-aligning. Suppress for safety."""
    om = _stub_om(desync_phase=DesyncPhase.RECOVERED)
    for _ in range(20):
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_no_quote(),
            decision_quote_cycle_id="cycle-x",
            position_qty=0.0, position_notional=0.0,
        )
    assert om._engine_no_quote_streak_ticks == 0


def test_fu2_streak_starts_fresh_after_desync_clears() -> None:
    """After desync clears (DETECTED → OK), the next no_quote streak
    must start from 0 — pre-FU-2 the streak silently accumulated
    during recovery and then fired immediately on transition back to
    OK, which is a false positive."""
    om = _stub_om(desync_phase=DesyncPhase.DETECTED)
    # 100 no_quote ticks during desync — streak stays 0.
    for _ in range(100):
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_no_quote(),
            decision_quote_cycle_id="cycle-x",
            position_qty=0.0, position_notional=0.0,
        )
    assert om._engine_no_quote_streak_ticks == 0
    # Now flip to OK — streak should resume from 0, NOT 100.
    om._state.desync_phase = DesyncPhase.OK
    for _ in range(3):
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_no_quote(),
            decision_quote_cycle_id="cycle-x",
            position_qty=0.0, position_notional=0.0,
        )
    # 3 ticks of fresh no_quote, threshold is 5 — should NOT have fired.
    assert om._engine_no_quote_streak_ticks == 3
    assert om._engine_no_quote_diag_last_log_mono == 0.0


def test_fu2_handles_string_desync_phase() -> None:
    """Defensive: some legacy paths may store desync_phase as a raw
    string. The guard accepts either the enum or a str."""
    om = _stub_om(desync_phase="DETECTED")  # raw string
    for _ in range(20):
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_no_quote(),
            decision_quote_cycle_id="cycle-x",
            position_qty=0.0, position_notional=0.0,
        )
    assert om._engine_no_quote_streak_ticks == 0


def test_fu2_handles_missing_desync_phase_attribute() -> None:
    """Defensive: if state has no desync_phase attribute at all,
    treat as OK (don't crash, don't silently suppress everything)."""
    from app.clock import SystemClock
    from app.execution import OrderManager
    om = OrderManager.__new__(OrderManager)
    om._clock = SystemClock()
    om._engine_no_quote_streak_ticks = 0
    om._engine_no_quote_diag_last_log_mono = 0.0
    om._side_unresolved_active = {}
    om._side_unresolved_reason = {}
    om._side_unresolved_requires_confirm = {}
    settings = MagicMock()
    settings.engine_no_quote_diag_streak_ticks = 5
    settings.engine_no_quote_diag_relog_seconds = 30.0
    om._settings = settings
    state = MagicMock(spec=[])  # no attributes at all
    om._state = state
    for _ in range(5):
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_no_quote(),
            decision_quote_cycle_id="cycle-x",
            position_qty=0.0, position_notional=0.0,
        )
    # Treated as OK → streak advances and warning fires.
    assert om._engine_no_quote_streak_ticks == 5
