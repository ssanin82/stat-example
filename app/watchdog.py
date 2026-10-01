"""Deadlock watchdog: force process exit when the bot is stuck.

Why
---
The bot has many internal gates that each have their own timeout, but
collectively can latch longer than any single gate's timeout:

* cancel-confirmation gate (waits for WS ``OrderCancellationUpdate``)
* uncertain-order-state gate (waits for ``AccountOrderUpdate``)
* cross-venue-cancel cascade (Binance drift > N bps)
* POST_ONLY_WOULD_TRADE retry backoff
* pending-cancel state in the bluefin_client adapter

Observed 2026-04-24: the quote engine emitted non-NONE decisions
continuously for ~1 hour while the execution layer placed zero
orders — silent deadlock requiring manual supervisor restart.

This watchdog breaks the deadlock by exiting the process cleanly
(status code 42) when it detects the stuck pattern:

    quote engine HAS been emitting non-NONE decisions recently
    AND execution has NOT dispatched an order attempt for a long time

Process supervisors (systemd ``Restart=on-failure``, container
managers, Kubernetes, ECS, etc.) bring the process back with clean
in-memory state, clearing all the latched gates.

Why exit instead of trying to "unstick" in place
-------------------------------------------------
Writing code that correctly untangles every possible latched-gate
combination is hard and high-risk (bugs in that code are worse than
the deadlock). A process restart is platform-agnostic, verified by
every container manager, and guaranteed to clear every in-memory
gate. This is the strategy used by every serious long-lived service
I'm aware of (beanstalkd, nginx workers, systemd service policy).

Tuning
------
See ``WATCHDOG_*`` settings in ``app/config.py``. Defaults are
conservative (10 min of no-place-attempts, 2 min quote activity
window) to avoid false positives in genuinely quiet markets.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

from app.config import Settings
from app.state import BotState

from app import clock as _clock

logger = logging.getLogger(__name__)


# Exit code the watchdog uses when it force-exits. Distinct from common
# codes (0 clean, 1 generic error, 2 misuse, 130 sigint) so operators
# can grep container logs for "exit code 42" to spot watchdog-triggered
# restarts specifically. Also used by tests to assert the right path.
WATCHDOG_EXIT_CODE = 42


class Watchdog:
    """Background thread that polls state and exits the process on deadlock.

    Construct once at bot startup, call ``start()``. The thread is a
    daemon so it won't block normal shutdown; ``stop()`` is still
    provided for orderly teardown in tests.

    ``on_pre_exit`` (optional) is invoked synchronously just before
    ``exit_fn`` fires. Use it for short, time-bounded hooks like a
    Telegram alert (sync HTTP POST with a 2 s budget). Any exception
    raised by the hook is logged and swallowed; the exit proceeds
    regardless.
    """

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        *,
        exit_fn: Optional[object] = None,
        on_pre_exit: Optional[Callable[[], None]] = None,
    ) -> None:
        self._settings = settings
        self._state = state
        self._enabled = bool(getattr(settings, "watchdog_enabled", True))
        self._no_place_s = float(
            getattr(settings, "watchdog_no_place_attempt_seconds", 600.0)
        )
        self._quote_window_s = float(
            getattr(settings, "watchdog_quote_activity_window_seconds", 120.0)
        )
        self._check_interval_s = float(
            getattr(settings, "watchdog_check_interval_seconds", 30.0)
        )
        # Injected for tests — production uses ``os._exit`` so we bypass
        # the stdlib ``sys.exit`` atexit handlers (some of which could
        # re-enter paused threads) and genuinely terminate immediately.
        self._exit_fn = exit_fn or (lambda code: os._exit(code))
        self._on_pre_exit = on_pre_exit
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._enabled:
            logger.info("watchdog_disabled")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(target=self._run, name="watchdog", daemon=True)
        self._thread = t
        t.start()
        logger.info(
            "watchdog_started no_place_s=%.0f quote_window_s=%.0f check_s=%.0f",
            self._no_place_s,
            self._quote_window_s,
            self._check_interval_s,
        )

    def stop(self) -> None:
        """Signal the thread to stop. Used by tests; production lets it die with the daemon."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)

    def _run(self) -> None:
        # Give the bot a full quote-activity window on startup before the
        # first check. Otherwise a slow startup (auth mint, venue info,
        # first public-WS message arriving late) would race the watchdog
        # into a false-positive exit on the first tick.
        initial_grace_s = max(
            self._no_place_s, self._quote_window_s, self._check_interval_s * 2
        )
        if self._stop.wait(timeout=initial_grace_s):
            return
        while not self._stop.wait(timeout=self._check_interval_s):
            try:
                self._check_once()
            except Exception:  # noqa: BLE001
                # Never let the watchdog itself crash the watchdog. Log
                # and continue; the next tick re-checks.
                logger.exception("watchdog_check_failed")

    def _check_once(self) -> None:
        """Evaluate the deadlock condition once; exit if it fires."""
        # Respect the enabled flag here too, not only in the thread
        # loop — operators occasionally flip config at runtime via env
        # reloads, and callers (tests, /status endpoints) might invoke
        # _check_once directly.
        if not self._enabled:
            return
        # Don't fire while the bot is intentionally paused / killed /
        # flattening. Those are states the operator (or safety logic)
        # put us in deliberately; bouncing the container would undo
        # their action.
        if getattr(self._state, "killed", False):
            return
        if getattr(self._state, "manual_pause", False):
            return
        if getattr(self._state, "flatten_mode", False):
            return
        if getattr(self._state, "flatten_incomplete", False):
            return

        now_mono = _clock.monotonic()

        # Signal 1: the quote engine has been saying "quote at least one
        # side" recently. If it hasn't, we're not stuck — we're just in
        # a regime where the eligibility gates are holding and no new
        # orders are expected. That's the correct behaviour, not a
        # deadlock.
        last_quote_mono = float(
            getattr(self._state, "last_quote_engine_non_hold_ts_mono", 0.0) or 0.0
        )
        if last_quote_mono <= 0:
            # Quote engine has never emitted non-NONE. Likely still
            # warming up or in a persistent HOLD_ALL regime. Not a
            # deadlock in either case.
            return
        quote_idle_s = now_mono - last_quote_mono
        if quote_idle_s > self._quote_window_s:
            # Quote engine itself is idle. Not the stuck-execution
            # pattern. Don't fire.
            return

        # Signal 2: the execution layer has NOT dispatched an order
        # attempt for a long time. v1.4.42: read the broader
        # ``last_outbound_attempt_ts_mono`` which counts PLACES + AMENDS
        # (but NOT cancels). The original ``last_place_attempt_ts_mono``
        # was place-only and produced false-positive kills in amend-
        # heavy regimes — the v1.4.16 amend-on-reprice path keeps an
        # order alive via continuous amends without dispatching a
        # fresh place, so the place-only counter went stale and the
        # watchdog killed the process every ~10 min of amend-only
        # quoting (observed 2026-05-18 in
        # `snapshots/v1.4.41-260518-103610` line 232:
        # `execution_last_attempt_s_ago=605.8` while amends were
        # firing every 670 ms). Cancels remain excluded — a cancels-
        # only loop is the BUG-025 wedge pattern and SHOULD trip.
        last_attempt_mono = float(
            getattr(self._state, "last_outbound_attempt_ts_mono", 0.0) or 0.0
        )
        # Fallback to the legacy place-only field for back-compat with
        # any state that pre-dates the new field (e.g. tests that
        # construct BotState manually). Removable in v1.5+.
        if last_attempt_mono <= 0:
            last_attempt_mono = float(
                getattr(self._state, "last_place_attempt_ts_mono", 0.0) or 0.0
            )
        if last_attempt_mono <= 0:
            # Bot has never placed AND never amended. Warm-up or
            # never-traded scenario — don't fire. A session with
            # literally zero fills is not a deadlock; it may be a
            # misconfiguration or a quiet market.
            return
        place_idle_s = now_mono - last_attempt_mono
        if place_idle_s < self._no_place_s:
            return

        # Both signals fired: quote engine is active, execution isn't.
        logger.error(
            "watchdog_deadlock_detected "
            "quote_engine_last_non_hold_s_ago=%.1f "
            "execution_last_attempt_s_ago=%.1f "
            "thresholds=no_place_s>=%.0f,quote_window_s<=%.0f "
            "action=process_exit code=%d",
            quote_idle_s,
            place_idle_s,
            self._no_place_s,
            self._quote_window_s,
            WATCHDOG_EXIT_CODE,
        )
        # Pre-exit hook (Telegram alert, etc.). Time-bounded; any
        # exception is swallowed so the exit always proceeds.
        if self._on_pre_exit is not None:
            try:
                self._on_pre_exit()
            except Exception:  # noqa: BLE001
                logger.exception("watchdog_on_pre_exit_failed")
        # Use os._exit to skip atexit / other shutdown hooks that could
        # themselves be blocked on whatever's causing the deadlock.
        self._exit_fn(WATCHDOG_EXIT_CODE)
