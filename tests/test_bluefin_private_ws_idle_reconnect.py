"""Bluefin private-WS idle watcher: two-tier warn / reconnect semantics.

History:
  ``tmp/snap_20260424_152226`` — Bluefin private WS silent for 62 minutes
  after the last fill, cancel-confirmation gate latched, quote loop
  stopped placing orders. Root cause: Bluefin's server does NOT send
  heartbeat frames on idle connections, so the websocket-client's
  transport-level ``ping_interval`` alone could not detect that the
  server-side had gone quiet at the application layer. The client
  reported ``private_ws_connected=True, healthy=True`` while zero
  messages had arrived for an hour.

  Fix (ported from ``app/exchange/grvt_ws.py``): a dedicated keepalive
  worker thread monitors ``state.private_ws_last_message_wall_ts`` on
  every ``PRIVATE_WS_APP_KEEPALIVE_SECONDS`` tick and has two
  independent tiers:

  * ``PRIVATE_WS_IDLE_WARN_SECONDS`` — log-only warning, no side effect.
  * ``PRIVATE_WS_IDLE_RECONNECT_SECONDS`` — force-close the WS so the
    outer reconnect loop re-mints auth and re-subscribes.

Tests exercise the worker directly (no real WS) to stay deterministic
and not sleep for the keepalive interval.
"""

from __future__ import annotations

import queue as _queue
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

from app.exchange.bluefin_ws import (
    BluefinPrivateStream,
    _BLUEFIN_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S,
)
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


_DUMMY_HEX64 = "0x" + "ab" * 32


def _make_stream(
    *,
    idle_warn: float = 10.0,
    idle_reconnect: float = 30.0,
    keepalive: float = 1.0,
) -> tuple[BluefinPrivateStream, BotState]:
    settings = UnitTestSettings.model_validate(
        {
            "EXCHANGE": "bluefin",
            "SYMBOL": "SUI-PERP",
            "BLUEFIN_PRIVATE_KEY": "00" * 32,
            "BLUEFIN_ACCOUNT_ADDRESS": _DUMMY_HEX64,
            "PRIVATE_WS_APP_KEEPALIVE_SECONDS": keepalive,
            "PRIVATE_WS_IDLE_WARN_SECONDS": idle_warn,
            "PRIVATE_WS_IDLE_RECONNECT_SECONDS": idle_reconnect,
        }
    )
    state = BotState(settings)
    s = BluefinPrivateStream(
        settings,
        user_address=_DUMMY_HEX64,
        out_queue=_queue.Queue(maxsize=64),
        state=state,
    )
    return s, state


def _install_fake_ws(s: BluefinPrivateStream) -> MagicMock:
    """Attach a fake WebSocketApp so the worker can call ``.close()`` on it."""
    fake_ws = MagicMock()
    fake_ws.close = MagicMock()
    with s._ws_lock:
        s._ws_app = fake_ws
    return fake_ws


def _force_one_worker_iteration(
    s: BluefinPrivateStream, *, iterations: int = 1
) -> None:
    """Run the worker body in-thread for N simulated ticks.

    The real worker uses ``Event.wait(timeout=interval)`` as its clock.
    We monkey-patch that so each tick returns immediately; after N ticks
    ``wait`` returns True (stopping event set) and the loop exits.
    """
    real_event = s._keepalive_stop
    real_event.clear()
    calls = {"n": 0}

    def _instant_wait(timeout: float) -> bool:  # noqa: ARG001
        calls["n"] += 1
        return calls["n"] > iterations

    real_event.wait = _instant_wait  # type: ignore[method-assign]
    s._keepalive_worker()


def test_idle_below_warn_does_not_log_or_close(caplog) -> None:
    """Below both thresholds → no warn log, no force close."""
    import logging

    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=2.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.bluefin_ws"):
        _force_one_worker_iteration(s)
    assert [r for r in caplog.records if "bluefin_private_ws_idle" in r.getMessage()] == []
    fake_ws.close.assert_not_called()


def test_idle_past_warn_below_reconnect_logs_only(caplog) -> None:
    """Warn tier fires, reconnect tier does not.

    Observability only — must NOT tear down the socket on natural fill
    silence, which would false-trigger the reconnect loop.
    """
    import logging

    s, state = _make_stream(idle_warn=10.0, idle_reconnect=60.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=30.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.bluefin_ws"):
        _force_one_worker_iteration(s)
    warn_hits = [r for r in caplog.records if "bluefin_private_ws_idle_warn" in r.getMessage()]
    force_hits = [r for r in caplog.records if "forcing_reconnect" in r.getMessage()]
    assert len(warn_hits) == 1, f"expected exactly one warn log, got {len(warn_hits)}"
    assert force_hits == [], "reconnect-tier must not fire below the reconnect threshold"
    fake_ws.close.assert_not_called()


def test_idle_past_reconnect_forces_close_and_logs(caplog) -> None:
    """Reconnect tier fires → force close + ``forcing_reconnect`` log."""
    import logging

    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.bluefin_ws"):
        _force_one_worker_iteration(s)
    fake_ws.close.assert_called_once()
    assert any("forcing_reconnect" in r.getMessage() for r in caplog.records), (
        "``bluefin_private_ws_idle_too_long ... forcing_reconnect`` log must fire"
    )


def test_reconnect_threshold_zero_disables_force_close(caplog) -> None:
    """``PRIVATE_WS_IDLE_RECONNECT_SECONDS=0`` disables the reconnect tier.

    Warn tier still fires (if set), but the socket is never force-closed.
    Pure-observability configuration for ops that want visibility without
    the recovery side-effect.
    """
    import logging

    s, state = _make_stream(idle_warn=10.0, idle_reconnect=0.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=600.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.bluefin_ws"):
        _force_one_worker_iteration(s)
    fake_ws.close.assert_not_called()
    assert any("bluefin_private_ws_idle_warn" in r.getMessage() for r in caplog.records)


def test_warn_threshold_zero_still_allows_force_close() -> None:
    """Inverse: WARN=0, RECONNECT>0 keeps recovery, silences the warn log."""
    s, state = _make_stream(idle_warn=0.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    _force_one_worker_iteration(s)
    fake_ws.close.assert_called_once()


def test_worker_exits_after_forcing_close() -> None:
    """After one force-close the worker must break out of its loop.

    The next ``_connect_once`` will spawn a fresh worker for the new
    connection; if the current one kept running it would double-close
    during the reconnect.
    """
    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    _force_one_worker_iteration(s, iterations=5)
    assert fake_ws.close.call_count == 1


def test_worker_disabled_when_keepalive_interval_non_positive() -> None:
    """Guardrail: if ``PRIVATE_WS_APP_KEEPALIVE_SECONDS`` ever reaches
    zero (via a hot-reload path that bypassed pydantic validation, say),
    the worker must no-op rather than busy-loop or crash.
    """
    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0, keepalive=1.0)
    # Patch the setting directly on the in-memory Settings object.
    object.__setattr__(s._settings, "private_ws_app_keepalive_seconds", 0.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=600.0)
    _force_one_worker_iteration(s, iterations=3)
    fake_ws.close.assert_not_called()


def test_log_throttle_constant_matches_grvt() -> None:
    """Documentation guard: the min-interval between warn logs is kept at
    the same value as the GRVT analogue so ops dashboards don't have to
    special-case Bluefin.
    """
    assert _BLUEFIN_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S == 60.0
