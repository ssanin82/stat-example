"""Regression tests for the three 1.1.37 fixes from the 2026-05-07
Codex review (deferred from the 1.1.36 batch).

Findings + fixes:

  - HIGH-1: ``_maybe_save_persistent_runtime_state`` only ran on the
    long normal-path tail of ``one_tick``. Soft-flatten paths return
    before that tail, so a crash during ``SOFT_FLATTENING`` could come
    back without the persisted SF intent. Fix: an immediate
    ``_save_persistent_runtime_state_now`` call inside
    ``_enter_soft_flatten`` and ``_exit_soft_flatten``.

  - HIGH-3: ``record_fill`` updated ``position_qty`` and
    ``position_notional`` but left ``unrealized_pnl_usd`` stale, so
    drawdown / session-loss gates ran against the pre-fill unrealized
    until the next REST refresh. Fix: shadow-recompute
    ``unrealized_pnl_usd`` from ``mark_price`` + ``avg_entry_price``
    against the new shadow qty. ``test_shadow_position.py`` covers
    the behavior; this file pins the remaining wiring guards.

  - MEDIUM: ``_run_soft_flatten_tick`` did not advance the execution
    tick counter, so ``should_ingest_fills_via_rest`` periodic-modulo
    gate stopped firing during long SF episodes. Fix: SF tick now
    calls ``OrderManager.on_bot_tick_start()`` at the top.
"""

from __future__ import annotations

import inspect

from app.bot import Bot


# ---------------------------------------------------------------------------
# HIGH-1: persistent runtime state is saved at SF entry/exit transitions
# ---------------------------------------------------------------------------


def test_save_persistent_runtime_state_now_helper_exists() -> None:
    """The helper bypasses the throttle interval so SF transitions
    are persisted even when the regular tail save would be skipped."""
    assert hasattr(Bot, "_save_persistent_runtime_state_now")
    src = inspect.getsource(Bot._save_persistent_runtime_state_now)
    # Bypass: the helper must call the underlying save directly.
    assert "try_save_persistent_runtime_state" in src
    # The helper must update the throttle timestamp so a subsequent
    # tail save in the same window is suppressed (avoids double-write
    # at SF-entry-then-immediate-tail).
    assert "_last_persistent_save_monotonic" in src


def test_enter_soft_flatten_saves_persistent_state_now() -> None:
    """Without this call, a crash during SOFT_FLATTENING would come
    back with the adverse inventory but the flatten *intent* lost —
    the bot would resume normal quoting on adverse position until
    the drawdown gate re-fires (another 30+ s of breach)."""
    src = inspect.getsource(Bot._enter_soft_flatten)
    assert "_save_persistent_runtime_state_now" in src


def test_exit_soft_flatten_saves_persistent_state_now() -> None:
    """Symmetric to entry: without an immediate save on exit, a
    restart between SF-completed and the next throttled tail save
    would come back believing SF is still active."""
    src = inspect.getsource(Bot._exit_soft_flatten)
    assert "_save_persistent_runtime_state_now" in src


# ---------------------------------------------------------------------------
# MEDIUM: SF tick advances the execution tick counter
# ---------------------------------------------------------------------------


def test_run_soft_flatten_tick_calls_on_bot_tick_start() -> None:
    """Without this, the periodic-modulo gate inside
    ``OrderManager.should_ingest_fills_via_rest`` is driven by a
    counter that stops advancing during SF, so the periodic REST
    fill-reconcile path never fires until SF exits.

    Source-substring sentinel: if a future refactor removes the
    ``on_bot_tick_start()`` call from ``_run_soft_flatten_tick``,
    this guard catches it before SF gates regress.
    """
    src = inspect.getsource(Bot._run_soft_flatten_tick)
    # The call must precede ``should_ingest_fills_via_rest`` so the
    # counter is advanced before the gate is consulted. Check by
    # searching for the call sites (``.on_bot_tick_start(`` /
    # ``.should_ingest_fills_via_rest(``) rather than bare names —
    # otherwise the comment block referencing the function names
    # would falsely satisfy the order check.
    on_tick_idx = src.find(".on_bot_tick_start(")
    ingest_idx = src.find(".should_ingest_fills_via_rest(")
    assert on_tick_idx >= 0, "on_bot_tick_start() not called in SF tick"
    assert ingest_idx >= 0, "should_ingest_fills_via_rest() not called in SF tick"
    assert on_tick_idx < ingest_idx, (
        "on_bot_tick_start must precede should_ingest_fills_via_rest "
        "or the periodic-modulo gate sees a stale counter"
    )
