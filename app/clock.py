"""Clock abstraction — Phase 1a of the backtesting build (v1.4.229).

The bot was originally written calling ``time.monotonic()`` /
``time.time()`` / ``datetime.now(timezone.utc)`` directly from ~250
sites across ``app/``. For BACKTESTING REPLAY the strategy code
must drive its "current time" from REPLAY DATA (the recorded
timestamps in the captured market-data stream), not from the
operating system clock — otherwise per-tick deltas, EWMA decays,
cooldown timers etc. all read the wall-clock running while the
replay loop processes a 2-hour fixture in 30 seconds.

This module defines a ``Clock`` protocol with two implementations:

* ``SystemClock`` — the production default. Delegates to the
  stdlib ``time`` / ``datetime`` modules. **Behaviourally identical
  to a direct ``time.monotonic()`` call.** Live bot uses this; CI
  daemon catches any drift.

* ``ReplayClock`` — the backtester's driven clock. Time advances
  ONLY when ``advance_to(ts_ns)`` is called. The replay engine
  pumps captured-data timestamps into this, and every strategy
  module that reads time gets the captured-data time, not the
  wall-clock.

Phase 1a (THIS COMMIT) ships the module + tests. It does NOT
migrate any existing call sites — all 250+ ``time.monotonic`` /
``time.time`` / ``datetime.now`` calls in ``app/`` still use the
stdlib directly. ``Bot.__init__`` accepts an optional ``clock``
parameter that defaults to ``SystemClock()`` so callers can opt
in without forcing every site to change at once.

Phase 1b will migrate the hot path (``bot.py`` + ``execution.py``)
to ``self._clock.<method>``. Phase 1c finishes the remaining sites
(state.py, gate evaluators, etc.).

Why the cautious 3-step split: 250+ sites is too many to migrate
in one PR without losing reviewability. Each cut is independently
testable and reversible. Production keeps shipping ``SystemClock``
through all three cuts; backtesting wiring depends on Phase 1c
being complete.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    """Protocol the bot's strategy code reads time through.

    Three accessor methods, mirroring the stdlib ones the bot
    currently uses:

    * ``time()`` — wall-clock seconds since the Unix epoch. Used
      by code that needs absolute timestamps (e.g. log lines,
      session_started_at_utc).
    * ``monotonic()`` — monotonically-non-decreasing seconds since
      an arbitrary origin. Used by ALL cooldown timers, dwell
      counters, age caps, etc. — anything that measures elapsed
      time and must be immune to wall-clock jumps (NTP sync,
      operator-set system clock, etc.).
    * ``now_utc()`` — tz-aware ``datetime`` in UTC. Used by code
      that needs human-readable timestamps (heartbeat publishes,
      event log entries).

    All three implementations of these three methods must agree
    on the underlying time. ``monotonic()`` is the most-used and
    its monotonicity invariant is load-bearing for many gate
    evaluators (a backward jump would corrupt EWMA decays, age
    counts, etc.).
    """

    def time(self) -> float: ...
    def monotonic(self) -> float: ...
    def now_utc(self) -> datetime: ...


class SystemClock:
    """Production clock — delegates to the stdlib.

    Behaviourally identical to direct ``time.*`` / ``datetime.*``
    calls. Used by the live bot via ``Bot.__init__(clock=...)``'s
    default. Zero-overhead wrapper; the stdlib calls themselves
    are already optimised.
    """

    __slots__ = ()

    def time(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)


class ReplayClock:
    """Backtester-driven clock — time advances ONLY on
    ``advance_to(ts_ns)``.

    Internal storage is nanosecond integers to avoid floating-point
    drift across long replays (a 24h fixture × 0.001 s tick =
    8.64e7 ticks; accumulated float drift would visibly accumulate
    after a few hundred thousand). The ``time()`` / ``monotonic()``
    accessors convert to seconds at read time.

    The replay engine drives this clock by reading each captured
    event's timestamp, calling ``advance_to(event_ts_ns)``, then
    invoking the appropriate strategy handler. Time never advances
    "ambiently" — only on explicit calls.

    ``monotonic()`` returns elapsed seconds since the replay's
    start (set on construction). This means ``monotonic()`` starts
    at 0 at the beginning of every replay run. Strategy code that
    stores monotonic timestamps and compares them later (typical
    pattern: ``state.last_seen_mono = clock.monotonic()``; later:
    ``elapsed = clock.monotonic() - state.last_seen_mono``) works
    correctly because BOTH reads happen against the same
    ``ReplayClock`` instance.
    """

    __slots__ = ("_t_ns", "_mono_origin_ns")

    def __init__(self, start_t_ns: int) -> None:
        """Construct with the replay's start timestamp (Unix
        nanoseconds). The monotonic origin is pinned to this same
        instant — so ``monotonic()`` at the start of the replay
        returns 0.0.
        """
        if start_t_ns < 0:
            raise ValueError(
                f"ReplayClock start_t_ns must be non-negative, got {start_t_ns}"
            )
        self._t_ns: int = int(start_t_ns)
        self._mono_origin_ns: int = int(start_t_ns)

    def advance_to(self, t_ns: int) -> None:
        """Set the current time to ``t_ns`` (Unix nanoseconds).

        Rejects backward jumps with a ``ValueError`` — strategy
        code's monotonic invariants would break under backward
        time. Equal timestamps (same-instant events processed in
        order) are allowed.
        """
        if t_ns < self._t_ns:
            raise ValueError(
                f"ReplayClock cannot go backward: current {self._t_ns} ns, "
                f"requested {t_ns} ns (delta = {t_ns - self._t_ns} ns)"
            )
        self._t_ns = int(t_ns)

    def time(self) -> float:
        """Wall-clock seconds since the Unix epoch at the current
        replay instant."""
        return self._t_ns / 1e9

    def monotonic(self) -> float:
        """Seconds since this replay's start. Always non-negative,
        non-decreasing across calls within one replay."""
        return (self._t_ns - self._mono_origin_ns) / 1e9

    def now_utc(self) -> datetime:
        """Tz-aware UTC ``datetime`` at the current replay instant."""
        return datetime.fromtimestamp(self._t_ns / 1e9, tz=timezone.utc)


# ---------------------------------------------------------------------
# Module-level clock proxy (Phase 1c, v1.4.231).
#
# The bot has ~144 ``time.monotonic`` / ``time.time`` / ``utc_now``
# call sites in free functions and standalone helper classes (state.py,
# live_stats.py, gate evaluators, etc.) that CANNOT use ``self._clock``
# because there's no ``self`` in scope. Phase 1a/1b handled the Bot
# and OrderManager class methods; this layer handles everything else.
#
# Architecture:
#   * ``_module_clock`` is the active clock for free-function callers.
#     Default = ``SystemClock()`` → production behaviour identical
#     to direct stdlib calls.
#   * ``set_module_clock(clock)`` swaps the active clock atomically.
#     Bot startup calls this with its own ``self._clock`` so the
#     module + bot share the SAME clock instance (essential — they
#     exchange monotonic timestamps via shared state).
#   * Module-level ``monotonic()`` / ``time_seconds()`` /
#     ``now_utc()`` functions delegate to ``_module_clock``. Free
#     functions import these directly as drop-in replacements for
#     stdlib calls.
#
# Threading note: assignment to a module-level name is atomic in
# CPython. Calls into the active clock are stateless (SystemClock)
# or single-threaded by design (ReplayClock — only the replay
# driver mutates its time, all other readers are pure read-only).
# No locks required.
#
# Why a module-level proxy not a context-manager or dependency-
# injection container: the bot is a single-process singleton — it
# never runs two clocks in one process. Test scaffolding that
# wants a custom clock per-test can call ``set_module_clock`` in
# ``setUp`` / pytest fixture and restore in ``tearDown`` /
# fixture teardown.
# ---------------------------------------------------------------------

_module_clock: Clock = SystemClock()


def set_module_clock(clock: Clock) -> None:
    """Install ``clock`` as the active module-level clock.

    Called by ``Bot.__init__`` at startup with the bot's own
    ``self._clock`` so the module-level free-function callers and
    the bot's method callers share a single clock instance.

    Backtesting replay driver calls this with a ``ReplayClock``
    before importing any bot code that would observe time.
    """
    global _module_clock
    _module_clock = clock


def get_module_clock() -> Clock:
    """Return the currently active module-level clock. Mostly for
    tests that want to save / restore the prior clock around an
    intentional ``set_module_clock`` substitution."""
    return _module_clock


def monotonic() -> float:
    """Module-level ``monotonic()`` — drop-in replacement for
    ``time.monotonic()`` in free-function consumers."""
    return _module_clock.monotonic()


def time_seconds() -> float:
    """Module-level wall-clock seconds — drop-in for ``time.time()``.

    Named ``time_seconds()`` to avoid shadowing the stdlib ``time``
    module which many call sites still import for ``perf_counter()``
    / ``sleep()`` / etc.
    """
    return _module_clock.time()


def now_utc() -> datetime:
    """Module-level tz-aware UTC ``datetime`` — drop-in for
    ``datetime.now(timezone.utc)`` and the old
    ``app.utils.time.utc_now()`` helper."""
    return _module_clock.now_utc()
