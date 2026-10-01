"""v1.5.275 / BUG-039 — kill path forces a final position+equity snapshot.

Pre-fix: ``Bot.kill()`` marked the bot KILLED and ran cancel /
flatten ops, but never explicitly flushed a final
``position_snapshots`` or ``equity_snapshots`` row. The periodic
``_persist_snapshots(force=False)`` writer skips writes when
``now_m - self._last_snapshot_monotonic < interval``. On a kill
that fires partway through that interval, the LAST persisted
row predates the just-before-kill fill. The dashboard's
inventory / unrealized-PnL sub-bands then display stale values
at the right edge of the chart even though
``state_current.json`` (the authoritative state surface)
correctly reflects the post-kill state.

Reproduced on snapshot v1.5.273-260530-075807: SF#25 taker fill
at 07:35:54 SELL 2.0 cleared inventory +2 → 0. Then
``sf_fatigue_tier4`` killed the bot ~1 second later. The
dashboard chart showed +2 at the right edge; state_current.json
correctly showed 0.

Fix: ``Bot.kill()`` now calls ``_persist_snapshots(force=True)``
right after marking KILLED, before any cleanup ops. The flush
captures the post-fill / pre-cleanup state — same value that
state_current.json carries — so the chart and the JSON agree.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


def test_kill_invokes_persist_snapshots_with_force_true():
    """The kill() method must call _persist_snapshots(force=True)
    immediately after marking KILLED. This is the regression guard
    that locks in the BUG-039 fix."""
    # We don't want to construct a full Bot (lots of dependencies);
    # we patch _persist_snapshots on the Bot CLASS, run a minimal
    # subset of kill() that triggers the flush, and verify the call.
    from app.bot import Bot
    from app.enums import BotStatus

    # Build a mock self with the minimum surface kill() touches.
    bot = MagicMock(spec=Bot)
    bot._state = MagicMock()
    bot._state._lock = MagicMock()
    bot._state._lock.__enter__ = MagicMock(return_value=None)
    bot._state._lock.__exit__ = MagicMock(return_value=False)
    bot._clock = MagicMock()
    bot._clock.now_utc.return_value = "2026-05-30T07:35:55+00:00"
    # _log_event, _maybe_write_shutdown_manifest, _dump_crash_snapshot,
    # _exec.cancel_all_orders_for_symbol — all stubs.
    bot._log_event = MagicMock()
    bot._maybe_write_shutdown_manifest = MagicMock()
    bot._dump_crash_snapshot = MagicMock()
    bot._exec = MagicMock()
    bot._exec.cancel_all_orders_for_symbol = MagicMock()
    bot._state.join_depth_controller = MagicMock()
    bot._settings = MagicMock()
    bot._settings.flatten_on_kill = False  # skip flatten path
    bot._settings.kill_auto_restart_grace_seconds = 0  # skip auto-restart
    bot._persist_snapshots = MagicMock()

    # Invoke the unbound kill() with our mock as self.
    Bot.kill(bot, "test_kill_reason")

    # The fix: _persist_snapshots was called exactly once with
    # force=True, AFTER bot_status was marked KILLED. We can't
    # easily verify the ordering with a MagicMock, but we CAN
    # verify the force argument and that it ran without raising.
    assert bot._persist_snapshots.called, (
        "BUG-039 regression: Bot.kill() must call _persist_snapshots() "
        "to flush a final position/equity row so the dashboard's "
        "inventory sub-band shows the post-kill state, not the "
        "pre-fill stale value."
    )
    call_args = bot._persist_snapshots.call_args
    # Accept either force=True kwarg or positional True.
    if call_args.kwargs:
        assert call_args.kwargs.get("force") is True, (
            f"_persist_snapshots called with kwargs={call_args.kwargs}; "
            f"force=True is required."
        )
    elif call_args.args:
        assert call_args.args[0] is True, (
            f"_persist_snapshots called with args={call_args.args}; "
            f"force=True is required."
        )
    else:
        pytest.fail(
            "_persist_snapshots called with no arguments; "
            "force=True is required."
        )


def test_kill_handles_persist_snapshots_exception_gracefully():
    """If the snapshot flush raises (disk full, DB locked, etc.),
    kill() must continue with the rest of the cleanup. The flush
    is best-effort observability, not a precondition for kill."""
    from app.bot import Bot

    bot = MagicMock(spec=Bot)
    bot._state = MagicMock()
    bot._state._lock = MagicMock()
    bot._state._lock.__enter__ = MagicMock(return_value=None)
    bot._state._lock.__exit__ = MagicMock(return_value=False)
    bot._clock = MagicMock()
    bot._clock.now_utc.return_value = "2026-05-30T07:35:55+00:00"
    bot._log_event = MagicMock()
    bot._maybe_write_shutdown_manifest = MagicMock()
    bot._dump_crash_snapshot = MagicMock()
    bot._exec = MagicMock()
    bot._exec.cancel_all_orders_for_symbol = MagicMock()
    bot._state.join_depth_controller = MagicMock()
    bot._settings = MagicMock()
    bot._settings.flatten_on_kill = False
    bot._settings.kill_auto_restart_grace_seconds = 0
    # Make the snapshot flush raise.
    bot._persist_snapshots = MagicMock(
        side_effect=RuntimeError("disk full")
    )

    # kill() must NOT propagate the exception — the cleanup ops
    # downstream are more important than the snapshot flush.
    Bot.kill(bot, "test_kill_reason")

    # _exec.cancel_all_orders_for_symbol was still called.
    assert bot._exec.cancel_all_orders_for_symbol.called, (
        "kill() must continue past a snapshot-flush failure — the "
        "cancel/flatten cleanup is the priority."
    )


def test_persist_snapshots_force_true_bypasses_interval_check():
    """The fix relies on ``_persist_snapshots(force=True)`` actually
    bypassing the interval check. This test pins that contract so
    a future refactor can't silently remove force-mode."""
    from app.bot import Bot
    import inspect

    # Walk the source of _persist_snapshots; the first interval
    # check must be guarded by ``not force``.
    src = inspect.getsource(Bot._persist_snapshots)
    # Look for the canonical pattern.
    assert "not force" in src, (
        "BUG-039 fix relies on Bot._persist_snapshots respecting "
        "force=True to bypass the interval skip. The current "
        "implementation must guard the early-return with `not force`."
    )
    assert "force" in src, (
        "_persist_snapshots must accept a `force` parameter for the "
        "BUG-039 fix path."
    )
