"""Startup-time exchange cleanup — REST-only, runs BEFORE WS subscribe.

Operator directive (2026-05-21):

    "Initial all cancellations must fully complete before the trade
    starts and any counting begins. Do it from a separate endpoint if
    needed. The most reliable way. It doesn't have to be the MOST
    efficient one, it does NOT have to use websockets. Startup cleanup
    must be reliable and complete, ONLY AFTER that bot starts,
    validation counters start."

Problem this module solves
==========================

Pre-fix startup flow in ``app/main.py``:

  1. ``private_stream.start()`` — WS subscribes; queue starts receiving
     order-update events from the venue.
  2. ``cancel_all_orders_for_symbol_bulk_or_fallback()`` — REST
     cancel-all of pre-existing orders left over from the previous
     process.
  3. Bot trading loop starts.

The race: the WS subscription in step 1 receives the cancel-terminals
that fire in response to step 2's REST cancels. Those terminals
carry exchange-oids the new process has no local record of, so they
land in the executor's ``ws_event_unmatched_to_local_wo_total``
counter — which the wedge-acceptance gate's strict-zero contract
flags as a regression even though it's just startup orphan bookkeeping.

Post-fix flow:

  1. Build streams **but do not start them.**
  2. Build Bot.
  3. REST cancel-all + REST poll-until-book-empty (this module).
  4. Reset the executor's startup-grace counters.
  5. **Now** subscribe to private + public WS streams.
  6. Bot trading loop starts.

After this reorder, no WS events from pre-existing orders can land
because we don't subscribe until the book is empty. The validation
counters genuinely start at zero with a clean book.

Why REST is the reliable path
-----------------------------

REST ``fetch_open_orders_raw(addr)`` is a synchronous read that
returns the matching engine's current open-orders state. It's the
authoritative source: when REST shows zero open orders for our
symbol, the venue has fully applied all cancels (the bulk-cancel
REST returns when the request is *accepted*; the cancels are
*applied* microseconds later). Polling REST until empty is the
"slow but correct" gate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

from app import clock as _clock


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — the minimum surface the helper consumes from client/settings
# ---------------------------------------------------------------------------


class _OpenOrderProtocol(Protocol):
    """Minimum WO row shape returned by ``fetch_open_orders_raw``.

    The real venue row (``OpenOrderRaw`` = ``HLOpenOrderRaw``) carries
    the instrument under the venue-neutral attribute ``coin`` (NOT
    ``symbol``) — the same field the live reconcile filters on
    (``_sync_open_orders_impl``: ``o.coin == settings.symbol``).
    ``wait_for_clean_book`` reads ``coin`` first and falls back to
    ``symbol`` only for adapters / fakes that still use the old name.
    """

    coin: str


class _ClientProtocol(Protocol):
    def fetch_open_orders_raw(self, address: str) -> list[_OpenOrderProtocol]: ...


class _SettingsProtocol(Protocol):
    symbol: str


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CleanBookResult:
    """Outcome of ``wait_for_clean_book``.

    ``success`` — True if the symbol's open-orders count reached zero
    before the timeout.
    ``final_count`` — open-orders count for the configured symbol on
    the LAST REST read. 0 on success; >=1 on timeout; -1 if every REST
    call failed (no readable count was ever obtained).
    ``polls_taken`` — number of REST calls issued.
    ``elapsed_seconds`` — wall-clock time spent draining.
    """

    success: bool
    final_count: int
    polls_taken: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


def wait_for_clean_book(
    *,
    client: _ClientProtocol,
    settings: _SettingsProtocol,
    address: str,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.5,
) -> CleanBookResult:
    """Poll the venue's open-orders REST endpoint until zero orders
    remain for ``settings.symbol``, or until ``timeout_seconds`` elapses.

    Used at startup AFTER ``cancel_all_orders_for_symbol_bulk_or_fallback()``
    and BEFORE ``private_stream.start()`` to guarantee the cancel-all
    has FULLY landed on the venue before the bot subscribes to private
    WS. Without this fence, cancel-terminals from pre-existing orders
    arrive via WS into a queue with no local working orders to match
    them against → ``ws_event_unmatched_to_local_wo_total`` ticks up
    on every restart even on a clean session.

    Returns a ``CleanBookResult`` with the outcome. The caller decides
    what to do on timeout — typically log a WARNING and continue (the
    bot can still trade with a non-empty book; subsequent orders will
    work fine — only the strict-zero acceptance counter would tick).

    Defensive choices:

    * REST exceptions during polling are caught + logged + retried
      until timeout — a transient REST error shouldn't crash startup.
    * Each loop iteration sleeps ``poll_interval_seconds`` after a
      non-zero read. The first read is immediate (no pre-sleep) — if
      cancel-all already drained synchronously, we return without
      blocking.
    * Counted symbol matches use case-insensitive comparison on the
      row's ``coin`` attribute (falling back to ``symbol``) to absorb
      venue casing differences. NOTE: the production ``OpenOrderRaw``
      row exposes the instrument as ``coin``; an earlier revision
      filtered on ``symbol`` — a field the real row does NOT carry —
      so the comprehension matched nothing and the fence reported the
      book clean on its first poll regardless of resting orders (a
      silent no-op). Reading ``coin`` first restores the real
      poll-until-empty behaviour.
    """
    deadline = _clock.monotonic() + timeout_seconds
    start = _clock.monotonic()
    polls = 0
    target_symbol = str(settings.symbol or "").upper()
    last_count: int = -1

    while True:
        polls += 1
        try:
            orders = client.fetch_open_orders_raw(address)
        except Exception:
            logger.exception(
                "startup_clean_book_poll_failed polls=%d", polls
            )
            # Don't update last_count — leave at last known value.
            if _clock.monotonic() >= deadline:
                return CleanBookResult(
                    success=False,
                    final_count=last_count,
                    polls_taken=polls,
                    elapsed_seconds=_clock.monotonic() - start,
                )
            time.sleep(poll_interval_seconds)
            continue

        # Count orders for OUR symbol only. A second bot on a different
        # symbol under the same account is NOT disturbed by the drain.
        # Prefer the venue-neutral ``coin`` attribute (what the real
        # OpenOrderRaw row carries and what the live reconcile filters
        # on); fall back to ``symbol`` for adapters / fakes still on the
        # old name.
        sym_orders = [
            o for o in (orders or [])
            if str(getattr(o, "coin", None) or getattr(o, "symbol", "") or "")
            .upper()
            == target_symbol
        ]
        last_count = len(sym_orders)
        if last_count == 0:
            return CleanBookResult(
                success=True,
                final_count=0,
                polls_taken=polls,
                elapsed_seconds=_clock.monotonic() - start,
            )
        if _clock.monotonic() >= deadline:
            return CleanBookResult(
                success=False,
                final_count=last_count,
                polls_taken=polls,
                elapsed_seconds=_clock.monotonic() - start,
            )
        time.sleep(poll_interval_seconds)


# ---------------------------------------------------------------------------
# Public orchestration helper — what main.py calls
# ---------------------------------------------------------------------------


def run_startup_cleanup(
    *,
    bot: Any,
    client: _ClientProtocol,
    settings: _SettingsProtocol,
    address: str,
    storage: Any,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.5,
) -> CleanBookResult:
    """Full startup-cleanup sequence: cancel-all + poll-until-empty +
    reset grace counters. Called from ``app/main.py`` right after the
    Bot is constructed and BEFORE any WS subscription happens.

    Steps:

      1. Issue ``cancel_all_orders_for_symbol_bulk_or_fallback()`` via
         the bot's executor.
      2. Persist a ``cancel_all_on_startup_executed`` event for
         postmortem / dashboard visibility.
      3. Poll ``client.fetch_open_orders_raw(addr)`` until zero open
         orders remain for ``settings.symbol``, or timeout.
      4. Persist a ``startup_book_clean`` event (or
         ``startup_book_drain_timeout`` on failure) with the poll
         count + elapsed time + final order count.
      5. Reset ``bot._exec`` startup-grace counters so subsequent
         ``ws_event_unmatched_to_local_wo_total`` increments
         genuinely reflect mid-session matcher misses.

    Returns the ``CleanBookResult`` so the caller can log / decide.
    """
    from app.enums import EventSeverity
    from app.utils.time import utc_now_iso

    # Step 1: REST cancel-all (via the executor's bulk-or-fallback path).
    try:
        outcome = bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback()
    except Exception:
        logger.exception("startup_cleanup_cancel_all_failed")
        outcome = "cancel_exception"

    # Step 2: event for visibility.
    try:
        storage.insert_bot_event(
            utc_now_iso(),
            EventSeverity.INFO.value,
            "cancel_all_on_startup_executed",
            f"cancel_all_on_startup executed (symbol-scoped, outcome={outcome})",
            {"symbol": settings.symbol, "outcome": outcome},
        )
    except Exception:
        logger.exception("startup_cleanup_event_insert_failed")

    # Step 3: poll-until-empty.
    result = wait_for_clean_book(
        client=client,
        settings=settings,
        address=address,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )

    # Step 4: event with the drain outcome.
    if result.success:
        logger.info(
            "startup_book_clean polls=%d elapsed_s=%.2f symbol=%s",
            result.polls_taken,
            result.elapsed_seconds,
            settings.symbol,
        )
        try:
            storage.insert_bot_event(
                utc_now_iso(),
                EventSeverity.INFO.value,
                "startup_book_clean",
                (
                    f"book clean after startup cancel-all "
                    f"(polls={result.polls_taken}, "
                    f"elapsed={result.elapsed_seconds:.2f}s)"
                ),
                {
                    "symbol": settings.symbol,
                    "polls": result.polls_taken,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            )
        except Exception:
            logger.exception("startup_clean_event_insert_failed")
    else:
        logger.warning(
            "startup_book_drain_timeout symbol=%s polls=%d "
            "elapsed_s=%.2f remaining=%d",
            settings.symbol,
            result.polls_taken,
            result.elapsed_seconds,
            result.final_count,
        )
        try:
            storage.insert_bot_event(
                utc_now_iso(),
                EventSeverity.WARNING.value,
                "startup_book_drain_timeout",
                (
                    f"book NOT clean after cancel-all "
                    f"(remaining={result.final_count}, "
                    f"polls={result.polls_taken}, "
                    f"elapsed={result.elapsed_seconds:.2f}s) — "
                    f"continuing anyway; validation counters may "
                    f"tick on residual orphan terminals"
                ),
                {
                    "symbol": settings.symbol,
                    "remaining": result.final_count,
                    "polls": result.polls_taken,
                    "elapsed_seconds": result.elapsed_seconds,
                },
            )
        except Exception:
            logger.exception("startup_drain_timeout_event_insert_failed")

    # Step 5: reset the executor's startup-grace counters. Defense in
    # depth — even though no WS events from pre-existing orders should
    # arrive (we haven't subscribed yet), a fresh counter at the
    # session boundary is the cleanest contract with the wedge-
    # acceptance gate's strict-zero check.
    try:
        bot._exec.reset_startup_grace_counters()
    except AttributeError:
        # Method added in v1.4.203; defensive for older executor shapes.
        logger.warning(
            "reset_startup_grace_counters not present on executor "
            "(old build?)"
        )
    except Exception:
        logger.exception("startup_cleanup_counter_reset_failed")

    return result
