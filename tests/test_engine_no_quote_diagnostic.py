"""Diagnostic event for persistent ``mode="no_quote"`` from QuoteEngine.

Reproduced 2026-05-08 in snapshot 260507114312: the bot's quote engine
produced no orders for 4.5 minutes between fill 11:38:48 and the
deadlock watchdog firing at 11:48. No event in the event log surfaced
the cause — only the watchdog kill at the 600 s ceiling.

These tests pin the new diagnostic that catches such silent wedges
within ~10 seconds:

  * On ``streak_ticks == threshold`` the diagnostic fires once with
    a payload that names the most-actionable telemetry (engine
    reasons, side_unresolved status, position state).
  * While stuck past the threshold, re-emits at a slower cadence
    (``ENGINE_NO_QUOTE_DIAG_RELOG_SECONDS``) — not every tick.
  * On any tick that produces orders, the streak counter resets.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.enums import Side
from app.execution import OrderManager
from app.exchange.symbol_spec import symbol_spec_from_hyperliquid_meta
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _make_om(
    *,
    threshold_ticks: int = 5,
    relog_seconds: float = 30.0,
) -> tuple[OrderManager, BotState, Storage, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_engineno_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "ENGINE_NO_QUOTE_DIAG_STREAK_TICKS": threshold_ticks,
            "ENGINE_NO_QUOTE_DIAG_RELOG_SECONDS": relog_seconds,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    meta = {"universe": [{"name": "ETH", "szDecimals": 4, "maxLeverage": 25}]}
    client.symbol_spec = symbol_spec_from_hyperliquid_meta(meta, "ETH")
    om = OrderManager(settings, client, storage, state)
    return om, state, storage, path


def _no_quote_telemetry(
    *,
    inventory_bias_active: bool = False,
    bid_reason: str = "side_not_requested",
    ask_reason: str = "side_not_requested",
) -> dict:
    return {
        "quote_engine_mode": "no_quote",
        "quote_engine_inventory_bias_active": inventory_bias_active,
        "quote_engine_inventory_bias_suppressed_bid": False,
        "quote_engine_inventory_bias_suppressed_ask": False,
        "quote_engine_bid_reason": bid_reason,
        "quote_engine_ask_reason": ask_reason,
        "quote_engine_normal_mode_requested": True,
        "exec_raw_bid_sz": 0.0,
        "exec_raw_ask_sz": 0.0,
        "downgraded_cycle_skipped_reason": None,
    }


def _ok_telemetry() -> dict:
    return {**_no_quote_telemetry(), "quote_engine_mode": "two_sided"}


def _bot_events(storage: Storage, event_type: str | None = None) -> list[dict]:
    rows = storage.bot_events_since("1970-01-01T00:00:00Z")
    if event_type is not None:
        rows = [r for r in rows if r.get("event_type") == event_type]
    return rows


# ---------- Streak-and-fire ----------------------------------------------------


def test_diag_does_not_fire_below_threshold() -> None:
    """Below the streak threshold, no event fires regardless of how many
    consecutive no-quote ticks. Avoids spam during normal one-tick blips."""
    om, _state, storage, path = _make_om(threshold_ticks=5)
    try:
        for _ in range(4):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_no_quote_telemetry(),
                decision_quote_cycle_id="qc-1",
                position_qty=0.0,
                position_notional=0.0,
            )
        events = _bot_events(storage, "engine_no_quote_persistent")
        assert events == []
        assert om._engine_no_quote_streak_ticks == 4
    finally:
        path.unlink(missing_ok=True)


def test_diag_fires_at_threshold_with_payload() -> None:
    """Reaching ``streak_ticks == threshold`` emits one event with the
    full diagnostic payload (engine reasons, side_unresolved status,
    position state)."""
    om, _state, storage, path = _make_om(threshold_ticks=3)
    try:
        for _ in range(3):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_no_quote_telemetry(
                    inventory_bias_active=True,
                    bid_reason="below_min_notional_intent_too_small",
                    ask_reason="inventory_bias_suppressed_adding_side",
                ),
                decision_quote_cycle_id="qc-stuck",
                position_qty=-1.0,
                position_notional=2.48,
            )
        events = _bot_events(storage, "engine_no_quote_persistent")
        assert len(events) == 1
        ev = events[0]
        assert ev["severity"] == "WARNING"
        # Payload may be a JSON string in the storage row depending on
        # backend — accept either dict or stringified.
        payload_raw = ev.get("payload_json") or ev.get("payload")
        assert payload_raw is not None
        if isinstance(payload_raw, str):
            import json as _json
            payload = _json.loads(payload_raw)
        else:
            payload = payload_raw
        assert payload["streak_ticks"] == 3
        assert payload["position_qty"] == -1.0
        assert payload["position_notional_usd"] == pytest.approx(2.48)
        assert payload["first_emission"] is True
        assert payload["quote_engine_bid_reason"] == "below_min_notional_intent_too_small"
        # Side-unresolved fields included for both sides.
        assert "side_unresolved_active_buy" in payload
        assert "side_unresolved_active_sell" in payload
        assert "side_unresolved_requires_confirm_buy" in payload
        assert "side_unresolved_requires_confirm_sell" in payload
    finally:
        path.unlink(missing_ok=True)


def test_diag_resets_on_quote_producing_tick() -> None:
    """A single tick that produces orders resets the streak counter to
    zero. The next no-quote streak starts fresh — must reach the
    threshold again before re-emitting."""
    om, _state, storage, path = _make_om(threshold_ticks=3)
    try:
        # Build up to threshold-1.
        for _ in range(2):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_no_quote_telemetry(),
                decision_quote_cycle_id="qc",
                position_qty=0.0,
                position_notional=0.0,
            )
        assert om._engine_no_quote_streak_ticks == 2
        # One successful tick.
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_ok_telemetry(),
            decision_quote_cycle_id="qc",
            position_qty=0.0,
            position_notional=0.0,
        )
        assert om._engine_no_quote_streak_ticks == 0
        # Two more no-quote ticks — still under threshold, no event.
        for _ in range(2):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_no_quote_telemetry(),
                decision_quote_cycle_id="qc",
                position_qty=0.0,
                position_notional=0.0,
            )
        assert _bot_events(storage, "engine_no_quote_persistent") == []
    finally:
        path.unlink(missing_ok=True)


def test_diag_relog_cadence_during_persistent_streak() -> None:
    """While stuck past the threshold, the diagnostic re-emits at the
    configured cadence — not on every tick. Avoids flooding the event
    log when the bot is wedged for minutes."""
    om, _state, storage, path = _make_om(threshold_ticks=2, relog_seconds=10.0)
    try:
        # First emission on threshold hit.
        for _ in range(2):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_no_quote_telemetry(),
                decision_quote_cycle_id="qc",
                position_qty=0.0,
                position_notional=0.0,
            )
        assert len(_bot_events(storage, "engine_no_quote_persistent")) == 1
        first_log_mono = om._engine_no_quote_diag_last_log_mono
        # Many subsequent ticks. Without manipulation of monotonic clock,
        # they all happen "now" and should NOT trigger relog (relog
        # requires elapsed >= relog_seconds = 10 s).
        for _ in range(50):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_no_quote_telemetry(),
                decision_quote_cycle_id="qc",
                position_qty=0.0,
                position_notional=0.0,
            )
        events = _bot_events(storage, "engine_no_quote_persistent")
        assert len(events) == 1, (
            "Diagnostic re-fired despite < relog_seconds elapsed"
        )
        # Simulate enough time for the relog window to elapse.
        om._engine_no_quote_diag_last_log_mono = first_log_mono - 11.0
        om._maybe_emit_engine_no_quote_diagnostic(
            telemetry=_no_quote_telemetry(),
            decision_quote_cycle_id="qc",
            position_qty=0.0,
            position_notional=0.0,
        )
        events = _bot_events(storage, "engine_no_quote_persistent")
        assert len(events) == 2
        # Re-emission flagged not-first.
        import json as _json
        payload2 = _json.loads(events[-1]["payload_json"])
        assert payload2["first_emission"] is False
    finally:
        path.unlink(missing_ok=True)


def test_diag_payload_captures_side_unresolved_latch() -> None:
    """The classic silent-wedge scenario: a side is latched as unresolved
    with ``requires_confirm=True`` and the bot is gated indefinitely.
    The diagnostic payload must surface this so the operator can see
    immediately which side is locked and why."""
    om, _state, storage, path = _make_om(threshold_ticks=2)
    try:
        # Simulate the latched-side scenario.
        om._side_unresolved_active[Side.BUY] = True
        om._side_unresolved_reason[Side.BUY] = "exchange_mismatch"
        om._side_unresolved_requires_confirm[Side.BUY] = True
        for _ in range(2):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_no_quote_telemetry(),
                decision_quote_cycle_id="qc",
                position_qty=0.0,
                position_notional=0.0,
            )
        events = _bot_events(storage, "engine_no_quote_persistent")
        assert len(events) == 1
        import json as _json
        payload = _json.loads(events[0]["payload_json"])
        assert payload["side_unresolved_active_buy"] is True
        assert payload["side_unresolved_reason_buy"] == "exchange_mismatch"
        assert payload["side_unresolved_requires_confirm_buy"] is True
        assert payload["side_unresolved_active_sell"] is False
    finally:
        path.unlink(missing_ok=True)


def test_diag_inactive_when_engine_produces_orders() -> None:
    """Sanity: when build_quotes returns ``two_sided`` or ``one_sided``,
    the diagnostic is fully inert."""
    om, _state, storage, path = _make_om(threshold_ticks=2)
    try:
        for _ in range(20):
            om._maybe_emit_engine_no_quote_diagnostic(
                telemetry=_ok_telemetry(),
                decision_quote_cycle_id="qc",
                position_qty=0.0,
                position_notional=0.0,
            )
        assert _bot_events(storage, "engine_no_quote_persistent") == []
        assert om._engine_no_quote_streak_ticks == 0
    finally:
        path.unlink(missing_ok=True)
