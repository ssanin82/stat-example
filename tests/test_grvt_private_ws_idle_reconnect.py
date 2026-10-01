"""GRVT private-WS idle watcher: two-tier warn / reconnect semantics.

History:
  ``tmp/snap_20260417_173125`` — TCP ping/pong kept the GRVT private WS
  "connected=true / healthy=true" for 68+ s with zero inbound messages
  after a SELL fill; the bot missed the terminal WS event for the next
  BUY cancel. First fix: force a reconnect at
  ``PRIVATE_WS_IDLE_WARN_SECONDS``, single-threshold.

  ``tmp/snap_20260418_153202`` — on thin-book AXS the single-threshold
  design mis-fired: normal 30-60 s fill-silence triggered repeated
  forced reconnects with exponential backoff, freezing the bot. Second
  fix (**this file**): split the threshold into a warn tier
  (observability only, no reconnect) and a reconnect tier (genuine
  stuck-subscription recovery), so the warn tier can fire on natural
  silence without the recovery side-effect.

The two tiers are independent:
  * ``PRIVATE_WS_IDLE_WARN_SECONDS`` — log only, rate-limited.
  * ``PRIVATE_WS_IDLE_RECONNECT_SECONDS`` — close the socket, triggering
    ``_run_forever`` to reconnect / re-subscribe.

Tests exercise the worker directly (no real WS) so they are
deterministic and don't sleep for the keepalive interval.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

from app.exchange.grvt_ws import (
    GrvtPrivateStream,
    _GRVT_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S,
)
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _make_stream(
    *,
    idle_warn: float = 10.0,
    idle_reconnect: float = 30.0,
    keepalive: float = 1.0,
) -> tuple[GrvtPrivateStream, BotState]:
    settings = UnitTestSettings.model_validate(
        {
            "EXCHANGE": "grvt",
            "GRVT_API_KEY": "x",
            "GRVT_API_SECRET": "x",
            "GRVT_SUB_ACCOUNT_ID": "42",
            "SYMBOL": "ETH_USDT_Perp",
            "PRIVATE_WS_APP_KEEPALIVE_SECONDS": keepalive,
            "PRIVATE_WS_IDLE_WARN_SECONDS": idle_warn,
            "PRIVATE_WS_IDLE_RECONNECT_SECONDS": idle_reconnect,
        }
    )
    state = BotState(settings)
    import queue as _queue

    s = GrvtPrivateStream(
        settings,
        user_address="42",
        out_queue=_queue.Queue(maxsize=64),
        state=state,
    )
    return s, state


def _install_fake_ws(s: GrvtPrivateStream) -> MagicMock:
    """Attach a fake websocket-app so the worker can call .close() on it."""
    fake_ws = MagicMock()
    fake_ws.close = MagicMock()
    with s._ws_lock:
        s._ws_app = fake_ws
    return fake_ws


def _force_one_worker_iteration(
    s: GrvtPrivateStream, *, iterations: int = 1, wait_timeout: float = 1.0
) -> None:
    """Run the worker loop in-thread for the configured number of iterations.

    The real worker uses ``Event.wait(timeout=interval)`` as its clock. We
    set the event after each simulated tick so the loop exits cleanly once
    the test has observed the side effect.
    """
    # Monkeypatch the worker's stop-event.wait so it "ticks" immediately.
    real_event = s._keepalive_stop
    real_event.clear()
    calls = {"n": 0}

    def _instant_wait(timeout: float) -> bool:  # noqa: ARG001
        calls["n"] += 1
        # First ``iterations`` calls return False (not stopping); after that
        # return True so the loop exits.
        return calls["n"] > iterations

    real_event.wait = _instant_wait  # type: ignore[method-assign]
    s._keepalive_worker()


def test_idle_below_warn_does_not_log_or_close(caplog) -> None:
    """Below both thresholds → no warn log, no force close."""
    import logging

    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=2.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.grvt_ws"):
        _force_one_worker_iteration(s)
    assert [r for r in caplog.records if "grvt_private_ws_idle" in r.getMessage()] == []
    fake_ws.close.assert_not_called()


def test_idle_past_warn_below_reconnect_logs_only(caplog) -> None:
    """Warn-tier fires, reconnect-tier does not. Observability only.

    This is the thin-book scenario: natural 30-60 s silence on AXS/NEAR
    should produce a warn log for ops visibility but MUST NOT tear down
    the socket — the old behaviour caused a reconnect loop that froze
    the bot (``tmp/snap_20260418_153202``).
    """
    import logging

    s, state = _make_stream(idle_warn=10.0, idle_reconnect=60.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=30.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.grvt_ws"):
        _force_one_worker_iteration(s)
    warn_hits = [r for r in caplog.records if "grvt_private_ws_idle_warn" in r.getMessage()]
    force_hits = [r for r in caplog.records if "forcing_reconnect" in r.getMessage()]
    assert len(warn_hits) == 1, f"expected one warn log, got {len(warn_hits)}"
    assert force_hits == [], "reconnect-tier log must not fire below the reconnect threshold"
    fake_ws.close.assert_not_called()


def test_idle_past_reconnect_forces_close_and_logs(caplog) -> None:
    """Reconnect-tier fires → force close + legacy ``forcing_reconnect`` log."""
    import logging

    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.grvt_ws"):
        _force_one_worker_iteration(s)
    fake_ws.close.assert_called_once()
    assert any("forcing_reconnect" in r.getMessage() for r in caplog.records), (
        "legacy ``grvt_private_ws_idle_too_long ... forcing_reconnect`` log "
        "must fire so existing log-scraping dashboards keep matching"
    )


def test_reconnect_threshold_zero_disables_force_close(caplog) -> None:
    """With ``PRIVATE_WS_IDLE_RECONNECT_SECONDS=0`` the reconnect-tier is
    disabled: warn-tier still fires (if set), but the socket is never
    force-closed. This is the "pure observability" configuration."""
    import logging

    s, state = _make_stream(idle_warn=10.0, idle_reconnect=0.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=600.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.grvt_ws"):
        _force_one_worker_iteration(s)
    fake_ws.close.assert_not_called()
    assert any("grvt_private_ws_idle_warn" in r.getMessage() for r in caplog.records)


def test_warn_threshold_zero_still_allows_force_close() -> None:
    """Inverse of the above: warn-tier disabled, reconnect-tier still works.

    Operators who don't want the warn log in their dashboards but still
    want stuck-subscription recovery can set WARN=0 and RECONNECT=300.
    """
    s, state = _make_stream(idle_warn=0.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    _force_one_worker_iteration(s)
    fake_ws.close.assert_called_once()


def test_worker_exits_after_forcing_close() -> None:
    """After one force-close the worker must stop — the next connect will
    spawn a fresh one. Otherwise we'd double-close during the reconnect."""
    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    # Up to 5 iterations available, but the worker should break out after the first close.
    _force_one_worker_iteration(s, iterations=5)
    assert fake_ws.close.call_count == 1


def test_worker_disabled_when_keepalive_interval_non_positive() -> None:
    """Guardrail: even if the validated settings somehow reach 0/negative
    (e.g. via a hot-reload path that bypasses the pydantic constraint),
    the worker must no-op rather than busy-looping."""
    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0, keepalive=1.0)
    # Bypass pydantic validation — directly mutate the cached value the
    # worker reads. Simulates "operator tried to disable the worker".
    s._settings.private_ws_app_keepalive_seconds = 0.0  # type: ignore[misc]
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    _force_one_worker_iteration(s)
    fake_ws.close.assert_not_called()


def test_worker_disabled_when_both_thresholds_non_positive() -> None:
    """Both warn and reconnect tiers at 0 → worker is effectively disabled."""
    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0, keepalive=1.0)
    s._settings.private_ws_idle_warn_seconds = 0.0  # type: ignore[misc]
    s._settings.private_ws_idle_reconnect_seconds = 0.0  # type: ignore[misc]
    fake_ws = _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=600.0)
    _force_one_worker_iteration(s)
    fake_ws.close.assert_not_called()


def test_worker_without_last_message_does_not_trip() -> None:
    """On a fresh connection before the first inbound message,
    ``private_ws_last_message_wall_ts`` is None — the worker must wait for
    a real baseline rather than tripping immediately."""
    s, state = _make_stream(idle_warn=10.0, idle_reconnect=30.0)
    fake_ws = _install_fake_ws(s)
    assert state.private_ws_last_message_wall_ts is None
    _force_one_worker_iteration(s)
    fake_ws.close.assert_not_called()


def test_worker_rate_limits_warn_log(caplog) -> None:
    """The warn-tier log is rate-limited so a dead session doesn't log-flood.

    We simulate two worker iterations both above the warn threshold but
    below the reconnect threshold; only the first should produce the
    warning because the rate-limit window
    (``_GRVT_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S`` = 60 s) is still open.
    """
    import logging

    # Pick idle_reconnect high enough that it never trips in this test —
    # we're measuring warn-tier log rate-limiting in isolation.
    s, state = _make_stream(idle_warn=10.0, idle_reconnect=10000.0)
    _install_fake_ws(s)
    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    with caplog.at_level(logging.WARNING, logger="app.exchange.grvt_ws"):
        _force_one_worker_iteration(s)
    first = [r for r in caplog.records if "grvt_private_ws_idle_warn" in r.getMessage()]
    assert len(first) == 1

    state.private_ws_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=60.0)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="app.exchange.grvt_ws"):
        _force_one_worker_iteration(s)
    second = [r for r in caplog.records if "grvt_private_ws_idle_warn" in r.getMessage()]
    assert second == [], "warn log must be rate-limited within the window"
    # Sanity: the rate-limit constant is documented at 60 s.
    assert _GRVT_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S == 60.0


def test_stop_tears_down_worker_thread() -> None:
    """``stop()`` must not leak the keepalive thread on shutdown."""
    s, _state = _make_stream(idle_warn=10.0, idle_reconnect=30.0, keepalive=30.0)
    # Start a real worker (it will block on the 30 s wait, plenty of time).
    s._start_keepalive()
    t = s._keepalive_thread
    assert t is not None
    assert t.is_alive()
    # Simulate stop() semantics for the keepalive portion only, without
    # touching the real WS connection machinery.
    s._stop_keepalive_worker()
    # Give the thread a moment to tear down; normally ``join(timeout=2.5)``
    # takes care of this inside ``_stop_keepalive_worker``.
    for _ in range(20):
        if not t.is_alive():
            break
        time.sleep(0.05)
    assert not t.is_alive()
    assert s._keepalive_thread is None
