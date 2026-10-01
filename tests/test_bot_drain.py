"""``Bot.drain`` — the finalized-stop de-risk primitive (v1.5.317).

Context
=======
``ops.ps1 <profile> stop`` (WITHOUT ``-Urgent``) is a *finalized* stop:

  1. drain the bot IN PLACE (this method),
  2. capture the full HTTP diagnostic snapshot from the STILL-RUNNING
     process (its live half — ``state_current.json``,
     ``session_summary.json``, venue REST cross-checks — only exists
     while the process is up),
  3. stop the service,
  4. pull the recording.

``Bot.drain`` is step 1. It must:

  * PAUSE quoting (``set_manual_pause(True)``) so no new orders go out,
  * let the outbound dispatcher settle (``wait_transport_idle``),
  * CANCEL all resting orders via
    ``cancel_all_orders_for_symbol_bulk_or_fallback`` — on OKX this is
    ASYNC (no venue bulk endpoint; the fallback ENQUEUES per-order
    cancels on the dispatcher), so drain waits idle then CONFIRMS via
    ``verify_book_clean_on_venue`` (a venue REST poll-until-empty), and
    retries up to ``max_cancel_passes``,
  * be **cancel-only**: the inventory position is intentionally LEFT
    INTACT and carried to the next session. drain NEVER flattens and
    NEVER kills.

The invariant this file pins hardest: **drain de-risks but never
liquidates.** A drain that market-flattened on the way out (like the
blind-kill regression in ``test_blind_kill_skip_flatten.py``) would
realise inventory at whatever price the book happened to show — the
opposite of a clean, position-preserving stop.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from app.bot import Bot
from app.startup_cleanup import CleanBookResult
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


# ---------------------------------------------------------------------------
# Harness — mirrors tests/test_blind_kill_skip_flatten.py::_make_bot
# ---------------------------------------------------------------------------


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"mm_drain_{os.getpid()}_{uuid.uuid4().hex}.db"


def _clean(final_count: int = 0) -> CleanBookResult:
    """Venue REST says the book is empty for our symbol."""
    return CleanBookResult(
        success=True,
        final_count=final_count,
        polls_taken=1,
        elapsed_seconds=0.01,
    )


def _dirty(final_count: int = 2) -> CleanBookResult:
    """Venue REST still shows ``final_count`` resting orders."""
    return CleanBookResult(
        success=False,
        final_count=final_count,
        polls_taken=1,
        elapsed_seconds=0.01,
    )


def _make_bot() -> Bot:
    """Build a real ``Bot`` over a real ``BotState`` + ``Storage`` (so
    ``set_manual_pause`` and ``_log_event`` exercise production code),
    then replace the executor's venue-touching methods with mocks so we
    can observe the drain sequence without a live exchange.
    """
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "PRIVATE_WS_ENABLED": False,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)

    # Executor surface drain consumes — all stubbed.
    bot._exec.wait_transport_idle = MagicMock()
    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback = MagicMock(
        return_value="okx_async_fallback_enqueued"
    )
    bot._exec.verify_book_clean_on_venue = MagicMock(return_value=_clean())

    # Cancel-only invariant guards: drain must touch NONE of these.
    bot._exec.cancel_all_orders_for_symbol = MagicMock()  # kill-path cancel
    bot.flatten = MagicMock()
    bot.kill = MagicMock()
    return bot


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_drain_happy_path_clean_first_pass() -> None:
    """Book drains on the first cancel pass:
      * quoting is paused (``manual_pause`` set True),
      * cancel-all + verify each run exactly once,
      * the loop breaks early (only 1 pass),
      * payload reports clean / 0 remaining / paused.
    """
    bot = _make_bot()

    result = bot.drain()

    # Paused in place — HTTP API stays up, but the risk gate now blocks.
    assert bot._state.manual_pause is True

    # Exactly one cancel + one verify (clean → break after pass 1).
    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback.assert_called_once()
    bot._exec.verify_book_clean_on_venue.assert_called_once()

    assert result["clean"] is True
    assert result["remaining_open_orders"] == 0
    assert result["cancel_passes"] == 1
    assert result["paused"] is True
    assert result["cancel_outcomes"] == ["okx_async_fallback_enqueued"]


def test_drain_waits_for_transport_idle_before_reading_venue() -> None:
    """OKX cancel-all enqueues on the dispatcher; drain MUST
    ``wait_transport_idle`` (once before the loop to settle in-flight
    batches, once per pass after enqueueing the cancels) so the venue
    REST read sees the cancels actually applied."""
    bot = _make_bot()

    bot.drain()

    # 1 pre-loop settle + 1 post-cancel wait on the single (clean) pass.
    assert bot._exec.wait_transport_idle.call_count == 2


# ---------------------------------------------------------------------------
# Retry / multi-pass behaviour
# ---------------------------------------------------------------------------


def test_drain_retries_until_clean() -> None:
    """First verify shows the book still dirty (async cancels not yet
    applied); the second pass confirms clean. drain must loop and report
    success with 2 passes."""
    bot = _make_bot()
    bot._exec.verify_book_clean_on_venue = MagicMock(
        side_effect=[_dirty(final_count=2), _clean()]
    )

    result = bot.drain()

    assert bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback.call_count == 2
    assert bot._exec.verify_book_clean_on_venue.call_count == 2
    assert result["clean"] is True
    assert result["remaining_open_orders"] == 0
    assert result["cancel_passes"] == 2


def test_drain_reports_incomplete_when_book_never_clears() -> None:
    """Cancels never land (stuck matching engine / venue fault). drain
    exhausts ``max_cancel_passes`` and reports INCOMPLETE with the
    residual count — it does NOT escalate to flatten/kill, it leaves the
    SIGTERM shutdown cancel-all as the backstop."""
    bot = _make_bot()
    bot._exec.verify_book_clean_on_venue = MagicMock(return_value=_dirty(3))

    result = bot.drain(max_cancel_passes=2)

    assert bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback.call_count == 2
    assert result["clean"] is False
    assert result["remaining_open_orders"] == 3
    assert result["cancel_passes"] == 2
    # Still cancel-only even on failure.
    bot.flatten.assert_not_called()
    bot.kill.assert_not_called()


def test_drain_respects_max_cancel_passes_floor() -> None:
    """``max_cancel_passes`` is floored at 1 — a caller passing 0 (or a
    negative) still gets exactly one cancel attempt, never zero."""
    bot = _make_bot()
    bot._exec.verify_book_clean_on_venue = MagicMock(return_value=_dirty(1))

    result = bot.drain(max_cancel_passes=0)

    assert bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback.call_count == 1
    assert result["cancel_passes"] == 1


# ---------------------------------------------------------------------------
# Cancel-only invariant — the load-bearing safety property
# ---------------------------------------------------------------------------


def test_drain_never_flattens_or_kills() -> None:
    """The whole point of the finalized stop: de-risk by cancelling our
    resting orders, but LEAVE THE POSITION INTACT for the next session.
    drain must never call ``flatten`` (market-close = realising
    inventory at the prevailing price) nor ``kill`` (that's the stop's
    job, AFTER the snapshot)."""
    bot = _make_bot()

    bot.drain()

    bot.flatten.assert_not_called()
    bot.kill.assert_not_called()
    # And drain uses the bulk-or-fallback cancel, NOT the kill-path
    # plain ``cancel_all_orders_for_symbol``.
    bot._exec.cancel_all_orders_for_symbol.assert_not_called()


def test_drain_leaves_position_untouched() -> None:
    """A non-zero inventory position carried into drain is still present
    afterwards — drain is cancel-only, so nothing reduces the qty."""
    bot = _make_bot()
    bot._state.position.position_qty = 5.0

    bot.drain()

    assert bot._state.position.position_qty == 5.0


# ---------------------------------------------------------------------------
# Robustness — best-effort, never raises into the operator's stop path
# ---------------------------------------------------------------------------


def test_drain_swallows_pause_failure_and_still_cancels() -> None:
    """If ``set_manual_pause`` raises, drain logs it and PRESSES ON to
    the cancel phase (a wedged pause must not block de-risking). The
    payload reports ``paused=False`` so the operator sees the pause
    didn't take."""
    bot = _make_bot()
    bot._state.set_manual_pause = MagicMock(side_effect=RuntimeError("boom"))

    result = bot.drain()

    assert result["paused"] is False
    # Cancel still ran despite the pause failure.
    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback.assert_called_once()
    assert result["clean"] is True


def test_drain_records_cancel_exception_and_continues() -> None:
    """A cancel that throws is recorded as ``cancel_exception`` in the
    per-pass outcomes; drain still proceeds to verify (the throw may
    have been a partial that still drained the book) and does not
    propagate the exception."""
    bot = _make_bot()
    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback = MagicMock(
        side_effect=RuntimeError("transport down")
    )

    result = bot.drain(max_cancel_passes=1)

    assert result["cancel_outcomes"] == ["cancel_exception"]
    # verify still consulted after the failed cancel.
    bot._exec.verify_book_clean_on_venue.assert_called_once()


def test_drain_forwards_clean_book_timeout() -> None:
    """The ``clean_book_timeout_s`` kwarg is forwarded to
    ``verify_book_clean_on_venue(timeout_s=...)`` so the operator can
    bound the venue-REST poll from the ops script."""
    bot = _make_bot()

    bot.drain(clean_book_timeout_s=42.0)

    bot._exec.verify_book_clean_on_venue.assert_called_once_with(timeout_s=42.0)


def test_drain_verify_exception_marks_not_clean() -> None:
    """If the venue-REST verify itself throws, drain treats the pass as
    NOT clean (we can't prove the book drained) and reports incomplete
    rather than falsely claiming success."""
    bot = _make_bot()
    bot._exec.verify_book_clean_on_venue = MagicMock(
        side_effect=RuntimeError("rest 500")
    )

    result = bot.drain(max_cancel_passes=1)

    assert result["clean"] is False
    # -1 is the "never got a readable count" sentinel.
    assert result["remaining_open_orders"] == -1
