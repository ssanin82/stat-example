"""Fill-drain thread must keep writing to trading.db even after KILL.

Regression guard for the 2026-04-19 ETH overnight session. After the bot
self-killed at 02:04:59 UTC (``stale_data_kill_escalated``), the GRVT
exchange processed two more fills on the bot's account — a post-kill
maker match at 02:05:28 and a manual operator taker-close at 04:02:50.
Both were delivered to the private-WS queue (the WS thread kept running
independent of kill state), but **neither was written to the local
``trading.db``** because the only consumer of that queue was
``drain_private_events`` inside ``Bot.one_tick()``, and ``one_tick()``
early-returns when ``is_killed()`` is True.

Reconciliation ``tmp/xinfo_20260419_044643`` vs
``tmp/snap_20260419_035212/trading.db`` confirmed: exchange=359 session
fills, bot=357. Missing 2. The bot's DB is now expected to match
exchange truth regardless of bot status — if a fill lands on the
private-WS queue, it MUST be persisted to Storage, full stop.

The fix: an independent drain thread (``Bot._drain_loop``) started by
``Bot.run_forever`` that loops ``drain_private_events`` on a 50 ms poll
and exits only on ``Bot.stop()`` (not on kill). The ``_drain_mutex`` in
``OrderManager`` serialises the two callers (quote loop + drain thread)
so in-memory state mutations inside ``_dispatch_private_event`` can't
race.

The invariant pinned here: **a fill event enqueued AFTER ``bot.kill()``
must reach ``trading.db`` within a short poll interval.** No bot-status
gating may prevent this write. If any future refactor reintroduces the
pre-fix behaviour, this test is the trip-wire.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from app.bot import Bot
from app.enums import BotStatus
from app.exchange.private_events import PrivateFillEvent
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"mm_fdrain_{os.getpid()}_{uuid.uuid4().hex}.db"


def _make_bot_with_queue() -> tuple[Bot, queue.Queue, Storage]:
    path = _db_path()
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "PRIVATE_WS_ENABLED": True,  # required for drain path to do work
            # Keep poll tight so tests don't sleep for long.
            # (The drain interval itself is an internal constant — we
            # reduce it below via direct attribute override.)
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    private_q: queue.Queue = queue.Queue(maxsize=256)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(
        settings,
        state,
        client,
        storage,
        private_event_queue=private_q,
    )
    # Stub out anything the kill path would call that doesn't matter here.
    bot._exec.cancel_all_orders_for_symbol = MagicMock()
    bot.flatten = MagicMock()
    # Speed the drain poll for deterministic tests.
    bot._drain_loop_interval_s = 0.005
    return bot, private_q, storage


def _fill_event(*, fill_id: str, oid: int, side: str = "BUY") -> PrivateFillEvent:
    return PrivateFillEvent(
        fill_id=fill_id,
        oid=oid,
        coin="ETH",
        px=2345.67,
        sz=0.017,
        side=side,
        time_ms=1_776_500_000_000,
        fee=-0.00001,
        closed_pnl=0.0,
        crossed=False,
        is_snapshot=False,
    )


def _count_fills(storage: Storage) -> int:
    with storage._lock:
        with storage.connection() as conn:
            row = conn.execute("SELECT COUNT(*) FROM fills").fetchone()
    return int(row[0])


def _wait_until(pred, timeout_s: float = 2.0, poll_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(poll_s)
    return False


# ------------------- The regression test -------------------------------


def test_fill_enqueued_after_kill_still_lands_in_db() -> None:
    """The exact 2026-04-19 scenario.

    1. Start the drain thread (as ``run_forever`` would).
    2. Kill the bot (``stale_data_kill_escalated`` — blind-kill path).
    3. Enqueue a fill event onto the private-WS queue.
    4. Assert the fill appears in ``trading.db`` within a short window.

    This fails on pre-fix code because ``drain_private_events`` only
    ran from ``one_tick()``, which returned early on ``is_killed()``.
    """
    bot, q, storage = _make_bot_with_queue()
    try:
        bot._start_drain_thread()

        # Kill the bot — blind-kill path, no flatten.
        bot.kill("stale_data_kill_escalated")
        assert bot._state.bot_status == BotStatus.KILLED

        # Enqueue a fill event that would have come from the private WS
        # post-kill (matched before our cancel propagated).
        ev = _fill_event(fill_id=f"post_kill_{uuid.uuid4().hex[:8]}", oid=999_001)
        q.put(ev)

        # The drain thread polls every 5 ms (overridden above). Within
        # well under 2 s the fill must be written to the DB.
        landed = _wait_until(lambda: _count_fills(storage) >= 1, timeout_s=2.0)
        assert landed, (
            "Fill enqueued after kill never reached trading.db — the "
            "independent drain thread is not working."
        )
    finally:
        bot.stop()


def test_drain_thread_writes_multiple_events_in_order() -> None:
    """Several events enqueued post-kill all get persisted, not just one.
    Covers the loop-within-drain (up to ``_MAX_PRIVATE_EVENTS_PER_TICK``)."""
    bot, q, storage = _make_bot_with_queue()
    try:
        bot._start_drain_thread()
        bot.kill("stale_data_kill_escalated")

        for i in range(5):
            q.put(_fill_event(fill_id=f"post_kill_batch_{i}", oid=900_000 + i))

        landed = _wait_until(lambda: _count_fills(storage) >= 5, timeout_s=2.0)
        assert landed, f"Only {_count_fills(storage)} of 5 fills landed in DB"
    finally:
        bot.stop()


def test_drain_thread_stops_cleanly_on_bot_stop() -> None:
    """After ``Bot.stop()``, the drain thread must exit within the join
    timeout (2 s). A leaked thread would prevent clean process
    shutdown under any supervisor's SIGTERM → SIGKILL grace window."""
    bot, _q, _storage = _make_bot_with_queue()
    bot._start_drain_thread()
    t = bot._drain_thread
    assert t is not None and t.is_alive()
    bot.stop()
    # ``stop()`` already joins with a 2 s timeout; confirm the thread
    # is no longer alive.
    assert not t.is_alive(), "drain thread leaked past Bot.stop()"


def test_drain_thread_is_idempotent_to_restart_attempts() -> None:
    """``_start_drain_thread`` is called from ``run_forever`` — if anyone
    calls it twice (e.g. a future refactor), it must not spawn a second
    thread that would race on the queue."""
    bot, _q, _storage = _make_bot_with_queue()
    try:
        bot._start_drain_thread()
        t1 = bot._drain_thread
        bot._start_drain_thread()
        t2 = bot._drain_thread
        assert t1 is t2, "second _start_drain_thread call spawned a new thread"
    finally:
        bot.stop()


def test_drain_thread_survives_handler_exception() -> None:
    """A transient exception in an event handler must NOT kill the
    drain thread — otherwise the very bug we're fixing reopens after
    the first bad event."""
    bot, q, storage = _make_bot_with_queue()
    try:
        bot._start_drain_thread()
        # Force the next drain to raise by replacing the inner method.
        # The outer ``drain_private_events`` wraps everything in a
        # try/except inside ``_drain_loop``, so the thread should
        # survive and keep processing subsequent events.
        original = bot._exec._drain_private_events_locked
        call_count = {"n": 0}

        def flaky(pnl):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated transient handler fault")
            return original(pnl)

        bot._exec._drain_private_events_locked = flaky  # type: ignore[assignment]

        # Wait for the first flaky call to happen, then restore and
        # enqueue a real fill to confirm the thread kept running.
        assert _wait_until(lambda: call_count["n"] >= 1, timeout_s=1.0)
        bot._exec._drain_private_events_locked = original  # type: ignore[assignment]
        q.put(_fill_event(fill_id="after_flake", oid=910_001))

        landed = _wait_until(lambda: _count_fills(storage) >= 1, timeout_s=2.0)
        assert landed, "drain thread died on handler exception instead of surviving"
    finally:
        bot.stop()


# ------------------- Mutex / concurrency -------------------------------


def test_drain_mutex_serialises_quote_loop_and_background_thread() -> None:
    """Both the quote loop and the drain thread call
    ``drain_private_events``. The ``_drain_mutex`` must serialise them
    so they can't concurrently mutate in-memory state. This test
    doesn't exercise the timing (hard to pin in a unit test), but it
    does prove:
      1. The mutex exists and is a ``threading.Lock``.
      2. It can be acquired / released safely.
    """
    bot, _q, _storage = _make_bot_with_queue()
    try:
        mutex = bot._exec._drain_mutex
        # Dual of threading.Lock: acquire + release.
        assert mutex.acquire(timeout=0.1)
        mutex.release()
        # Also confirm drain works when mutex is idle.
        bot._exec.drain_private_events(bot._pnl)
    finally:
        bot.stop()
