"""v1.4.43 heartbeat fix: ``execution_idle_seconds`` reads the new
``last_outbound_attempt_ts_mono`` (place + amend) instead of the
place-only ``last_place_attempt_ts_mono``.

Context: v1.4.42 BUG-025-adjacent fix added
``last_outbound_attempt_ts_mono`` and wired the watchdog +
silent-wedge detector to read it. The heartbeat was missed —
the dashboard's "NEAR-DEADLOCK" / "LOW-ACTIVITY" chip reads
``bot_execution_idle_seconds`` from the heartbeat, so the chip
kept false-firing during amend-heavy quoting even though the
bot itself was healthy (no watchdog kill, no silent-wedge
event).

Observed 2026-05-18, snapshot v1.4.42-260518-131017:
``executor_state.execution_idle_s = 0.016`` (16 ms — amend
just fired) while the dashboard chip displayed
``idle=305s`` (the place-only counter).

This test pins the heartbeat field to read the new counter so
that future refactors can't silently regress the dashboard
chip back to false-firing.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock

from app.heartbeat import HeartbeatPublisher
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides: Any) -> UnitTestSettings:
    base: dict[str, Any] = {
        "EXCHANGE": "okx",
        "SYMBOL": "TON-USDT-SWAP",
        "OKX_API_KEY": "x",
        "OKX_SECRET_KEY": "y",
        "OKX_PASSPHRASE": "z",
        "OKX_ACCOUNT_ADDRESS": "0xabc",
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "TRADING_ENABLED": True,
        "HEARTBEAT_INTERVAL_SECONDS": 30.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _publisher_with_state(state: BotState) -> tuple[HeartbeatPublisher, MagicMock]:
    """Construct a HeartbeatPublisher with a mocked S3 client so
    ``_publish_once`` runs end-to-end and we can intercept the
    JSON body via the put_object call."""
    s = _settings()
    pub = HeartbeatPublisher(
        s, state, bucket="test-bucket", profile_name="test-profile"
    )
    mock_s3 = MagicMock()
    pub._client = mock_s3
    return pub, mock_s3


def _published_body(s3_mock: MagicMock) -> dict[str, Any]:
    """Pull the JSON body out of the last put_object call. The
    publisher uses ``s3.put_object(Bucket=..., Key=..., Body=...)``
    keyword form."""
    s3_mock.put_object.assert_called()
    call = s3_mock.put_object.call_args
    body = call.kwargs.get("Body")
    if body is None and call.args:
        # Defensive: handle positional form if the publisher ever
        # changes (it currently uses kwargs).
        body = call.args[-1]
    assert body is not None, (
        f"heartbeat publisher did not include Body in put_object call. "
        f"args={call.args!r} kwargs={list(call.kwargs.keys())!r}"
    )
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def test_heartbeat_execution_idle_uses_outbound_attempt_field_v1_4_43() -> None:
    """v1.4.43 fix: when only ``last_outbound_attempt_ts_mono`` is
    fresh (i.e. the bot is alive via amends, not places), the
    heartbeat must report a small idle. Pre-fix the place-only
    counter went stale and the dashboard chip showed false-alarm
    "NEAR-DEADLOCK"."""
    s = _settings()
    state = BotState(s)
    now = time.monotonic()
    # Place counter is STALE (no fresh place in 5 min — typical
    # amend-on-reprice regime).
    state.last_place_attempt_ts_mono = now - 305.0
    # Outbound counter is FRESH (amend just fired).
    state.last_outbound_attempt_ts_mono = now - 0.02

    pub, s3 = _publisher_with_state(state)
    pub._publish_once()
    body = _published_body(s3)

    idle = body.get("execution_idle_seconds")
    assert idle is not None, "field must be published when bot has attempted"
    assert idle < 5.0, (
        f"v1.4.43: heartbeat must read the outbound counter (fresh, "
        f"~0.02 s) — not the place-only counter (stale, 305 s). "
        f"Got idle={idle}. Dashboard chip would still false-fire "
        f"NEAR-DEADLOCK if this regresses."
    )


def test_heartbeat_falls_back_to_place_field_pre_v1_4_42_state() -> None:
    """v1.4.43 back-compat: state objects from before v1.4.42 don't
    have ``last_outbound_attempt_ts_mono`` populated (it stays at the
    init default 0.0). The heartbeat must fall back to the legacy
    place-only counter so the chip still works for pre-fix sessions
    in hot-upgrade scenarios."""
    s = _settings()
    state = BotState(s)
    now = time.monotonic()
    state.last_place_attempt_ts_mono = now - 42.0
    state.last_outbound_attempt_ts_mono = 0.0  # never set (legacy)

    pub, s3 = _publisher_with_state(state)
    pub._publish_once()
    body = _published_body(s3)

    idle = body.get("execution_idle_seconds")
    assert idle is not None
    assert 40.0 <= idle <= 45.0, (
        f"v1.4.43 back-compat: fall back to place-only counter when "
        f"outbound counter is unset. Expected ~42 s, got {idle}."
    )


def test_heartbeat_idle_none_when_never_attempted() -> None:
    """v1.4.43 invariant preserved: until ANY outbound attempt has
    happened (neither place nor amend), the field is ``None``.
    Distinguishes "warm-up / never-traded" from "actively idle"."""
    s = _settings()
    state = BotState(s)
    state.last_place_attempt_ts_mono = 0.0
    state.last_outbound_attempt_ts_mono = 0.0

    pub, s3 = _publisher_with_state(state)
    pub._publish_once()
    body = _published_body(s3)

    assert body.get("execution_idle_seconds") is None, (
        "Until the bot makes its first outbound attempt, the chip "
        "should be hidden (None), not show idle=∞ or similar."
    )


def test_heartbeat_amend_only_session_chip_shows_fresh_idle_v1_4_43() -> None:
    """v1.4.43 end-to-end: simulate the exact amend-heavy regime
    that triggered the false-positive in snapshot v1.4.42-260518-131017.
    Place counter stale 305 s, amend counter fresh 0.016 s. Heartbeat
    publishes idle in single-digit-ms range."""
    s = _settings()
    state = BotState(s)
    now = time.monotonic()
    state.last_place_attempt_ts_mono = now - 305.0
    state.last_outbound_attempt_ts_mono = now - 0.016

    pub, s3 = _publisher_with_state(state)
    pub._publish_once()
    body = _published_body(s3)

    idle = body.get("execution_idle_seconds")
    assert idle is not None
    # Heartbeat rounds to 0.1s; 0.016 may round to 0.0.
    assert idle < 1.0, (
        f"v1.4.43 false-positive regression test: snapshot "
        f"v1.4.42-260518-131017 had amend at 16 ms but chip read "
        f"place at 305 s. The fix must show <1 s idle here. "
        f"Got idle={idle}."
    )
