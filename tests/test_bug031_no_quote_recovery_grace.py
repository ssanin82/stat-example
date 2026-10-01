"""BUG-031 fix — silent-wedge false-positive after engine no_quote
recovery.

Driving snapshot: ``v1.5.157-260526-113525`` showed 14 wedges in
1.31 h (vs v1.5.154-260526-100625's 30 in 10.95 h = 4x rate spike).
Forensics:

* Every wedge was preceded by exactly 3 ``engine_no_quote_persistent``
  events.
* The first ``no_quote_persistent`` payload shows ``streak_ticks=60,
  position_qty=-3``, both bid+ask reason ``"side_not_requested"``.
  Bot was in an extended period (~16 min in the longest case) where
  the quote engine produced no orders because the upstream
  active_sides + risk_action combination resolved to "neither side
  wanted".
* During that period the executor's ``last_outbound_attempt_ts_mono``
  did NOT advance (no place attempts; nothing to place). So
  ``execution_idle_s`` accumulated to many minutes.
* When the engine recovered to ``two_sided``, eligibility was once
  again ``QUOTE_BOTH``, no resting orders existed, and idle was
  large — the wedge detector's conjoint condition tripped and
  fired a false-positive event WHILE the executor was still
  catching up to the freshly-recovered engine.

The fix: capture the engine-recovery timestamp (when the streak
transitions from > 0 back to 0). The wedge detector adds a grace
window after that recovery — within those N seconds, the wedge
check is skipped so the executor has time to issue a fresh
place/amend. After the grace window, if the executor STILL hasn't
acted, the wedge fires (the genuine wedge class).

Tests verify:
1. Within grace window after recovery → no wedge (false positive
   prevented).
2. After grace window expires with executor still idle → wedge
   DOES fire (real wedges are still caught).
3. No recovery has ever happened (clean session start) → wedge
   fires normally (no regression on the legacy detector behaviour).

Per CLAUDE.md: only this test file is run from the assistant;
full-suite verification is the CI daemon's job.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_bug031_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
        "SILENT_WEDGE_DETECT_THRESHOLD_SECONDS": 10.0,
        "SILENT_WEDGE_DETECT_RELOG_SECONDS": 1.0,
        "SILENT_WEDGE_NO_QUOTE_RECOVERY_GRACE_SECONDS": 10.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _setup() -> tuple[OrderManager, Storage, BotState, Path]:
    s = _settings()
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    db_path.unlink(missing_ok=True)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    return om, storage, state, db_path


def _telemetry_healthy() -> dict:
    return {
        "quote_engine_mode": "two_sided",
        "quote_engine_normal_mode_requested": True,
        "quote_engine_bid_reason": None,
        "quote_engine_ask_reason": None,
        "quote_engine_bid_final_px": 100.00,
        "quote_engine_ask_final_px": 100.01,
        "quote_engine_bid_final_sz": 3.0,
        "quote_engine_ask_final_sz": 3.0,
    }


def _wedge_event_count(storage) -> int:
    events = storage.recent_bot_events(50)
    return sum(
        1 for e in events
        if e.get("event_type") == "executor_silent_wedge_detected"
    )


# ---------------------------------------------------------------------------
# The fix — within the grace window, skip the wedge check
# ---------------------------------------------------------------------------


def test_wedge_skipped_within_grace_window_after_recovery():
    """The fix: when the engine has just recovered from a no_quote
    stretch, the wedge detector skips for the configured grace
    period. Reproduces the v1.5.157-260526-113525 false-positive
    pattern."""
    om, storage, state, db_path = _setup()
    try:
        # 1. Simulate the engine being in no_quote for many ticks.
        om._engine_no_quote_streak_ticks = 1653
        # 2. Simulate the executor having been idle for a long time
        #    (longer than the 10 s threshold). This is what happens
        #    when no_quote prevented all place attempts.
        state.last_outbound_attempt_ts_mono = time.monotonic() - 30.0
        # 3. Engine recovers: a healthy telemetry arrives. The
        #    no_quote diagnostic captures the recovery moment.
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_healthy(),
            decision_quote_cycle_id="test",
            position_qty=0.0,
            position_notional=0.0,
        )
        # 4. Recovery timestamp must be set (sentinel was 0.0).
        assert om._engine_no_quote_recovery_ts_mono > 0.0
        assert om._engine_no_quote_streak_ticks == 0  # reset
        # 5. Now the wedge detector runs. It should SKIP (within
        #    the grace window) even though all conjoint conditions
        #    look like a wedge.
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_healthy(),
            decision_quote_cycle_id="test-cycle-grace",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        assert _wedge_event_count(storage) == 0, (
            "Within grace window, wedge must not fire — "
            "false-positive prevention is the whole point of the fix."
        )
    finally:
        db_path.unlink(missing_ok=True)


def test_wedge_fires_after_grace_window_expires():
    """The grace period is short enough that genuine wedges are
    still caught. After the configured grace seconds elapse, if the
    executor still hasn't placed, the wedge fires as before."""
    om, storage, state, db_path = _setup()
    try:
        # Recovery happened > grace_seconds ago.
        om._engine_no_quote_recovery_ts_mono = time.monotonic() - 60.0
        om._engine_no_quote_streak_ticks = 0
        # Executor STILL hasn't placed (last_outbound idle > threshold).
        state.last_outbound_attempt_ts_mono = time.monotonic() - 30.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_healthy(),
            decision_quote_cycle_id="test-cycle-real-wedge",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        assert _wedge_event_count(storage) == 1, (
            "Past the grace window, the wedge detector must still "
            "catch a genuine wedge (executor truly stuck)."
        )
    finally:
        db_path.unlink(missing_ok=True)


def test_wedge_fires_when_no_recovery_has_ever_happened():
    """Sentinel ``_engine_no_quote_recovery_ts_mono == 0.0`` means
    "no recovery yet this session". The wedge detector must NOT
    skip in that case — regression guard against turning the
    detector off entirely."""
    om, storage, state, db_path = _setup()
    try:
        # Initial state: recovery_ts is the sentinel 0.0.
        assert om._engine_no_quote_recovery_ts_mono == 0.0
        # Idle is > threshold.
        state.last_outbound_attempt_ts_mono = time.monotonic() - 30.0
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=_telemetry_healthy(),
            decision_quote_cycle_id="test-cycle-clean-start",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        assert _wedge_event_count(storage) == 1, (
            "Without a no_quote recovery to skip from, the wedge "
            "detector must fire normally."
        )
    finally:
        db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Recovery-capture invariant
# ---------------------------------------------------------------------------


def test_recovery_ts_captured_on_streak_transition():
    """The recovery timestamp is set ONLY when the streak
    transitions from > 0 to 0. If streak was already 0 (e.g. first
    healthy tick after session start), no recovery is recorded."""
    om, storage, state, db_path = _setup()
    try:
        # Case 1: streak was 0, stays 0 → no recovery ts set.
        om._engine_no_quote_streak_ticks = 0
        assert om._engine_no_quote_recovery_ts_mono == 0.0
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_healthy(),
            decision_quote_cycle_id="test",
            position_qty=0.0,
            position_notional=0.0,
        )
        assert om._engine_no_quote_recovery_ts_mono == 0.0, (
            "No transition (streak was already 0); recovery ts "
            "must not be set."
        )
        # Case 2: streak was > 0, then becomes 0 → recovery ts set.
        om._engine_no_quote_streak_ticks = 50
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_healthy(),
            decision_quote_cycle_id="test",
            position_qty=0.0,
            position_notional=0.0,
        )
        assert om._engine_no_quote_recovery_ts_mono > 0.0
        assert om._engine_no_quote_streak_ticks == 0
    finally:
        db_path.unlink(missing_ok=True)


def test_recovery_ts_persists_across_subsequent_healthy_ticks():
    """The recovery timestamp must persist across multiple healthy
    ticks within the grace window — it's only updated on the
    transition, not on every tick."""
    om, storage, state, db_path = _setup()
    try:
        # Drive recovery.
        om._engine_no_quote_streak_ticks = 50
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_telemetry_healthy(),
            decision_quote_cycle_id="test",
            position_qty=0.0,
            position_notional=0.0,
        )
        first_ts = om._engine_no_quote_recovery_ts_mono
        assert first_ts > 0.0
        # Simulate some additional healthy ticks.
        time.sleep(0.05)  # microscopic delay to ensure clock advance
        for _ in range(5):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_telemetry_healthy(),
                decision_quote_cycle_id="test",
                position_qty=0.0,
                position_notional=0.0,
            )
        # Recovery ts must NOT have been bumped by the subsequent
        # healthy ticks (streak was already 0 each time).
        assert om._engine_no_quote_recovery_ts_mono == first_ts
    finally:
        db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Reproducer of the actual snapshot pattern
# ---------------------------------------------------------------------------


def test_v1_5_157_snapshot_reproducer_no_false_positive():
    """Direct reproducer of the v1.5.157-260526-113525 false-
    positive: engine had been in no_quote_persistent for ~16 min
    (streak ticks 1653), then recovered, executor's
    last_outbound_attempt_ts_mono was ~17 min stale. Pre-fix the
    wedge fired immediately on recovery. Post-fix the grace window
    suppresses the false positive."""
    om, storage, state, db_path = _setup()
    try:
        # Reproduce the exact pre-recovery state.
        om._engine_no_quote_streak_ticks = 1653
        # Executor idle ~17 min (last place attempt long before
        # no_quote started).
        state.last_outbound_attempt_ts_mono = time.monotonic() - 1020.0  # 17 min
        # Engine recovers — telemetry now reports two_sided.
        telemetry = _telemetry_healthy()
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=telemetry,
            decision_quote_cycle_id="snapshot-repro",
            position_qty=0.0,
            position_notional=0.0,
        )
        # Immediately after recovery (within grace window), wedge
        # detector runs.
        om._maybe_emit_silent_wedge_diagnostic(
            telemetry=telemetry,
            decision_quote_cycle_id="snapshot-repro",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )
        assert _wedge_event_count(storage) == 0, (
            "Snapshot reproducer must NOT fire a wedge — that "
            "would be the v1.5.157 false-positive class this fix "
            "is designed to prevent."
        )
    finally:
        db_path.unlink(missing_ok=True)
