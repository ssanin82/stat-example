"""BUG-013: ``Bot.kill()`` must auto-restart the process on
environmental kill reasons (Bluefin WS stall, account-data stale, etc.)
so the process supervisor (systemd / container manager / k8s) bounces it.

Reproducer: ``tmp/snap_20260427_040436``. Bluefin's public WS went silent
for ~9 minutes; the bot's market-data recovery loop tried for 10 minutes
and self-killed with reason ``stale_data_kill_escalated``. The kill
itself was correct (don't quote blind). What was wrong: ``Bot.kill()``
set ``state.killed = True`` and stopped, never calling ``os._exit``. The
process stayed alive in KILLED state; the host's restart-on-failure
policy (Railway era at the time) only fires on non-zero exit; so the
bot sat dead-but-alive for 2.5 h until the operator noticed.

Fix: when the kill reason is in ``_AUTO_RESTART_KILL_REASONS`` (a
deliberate allow-list of environmental / transient causes), schedule a
delayed ``os._exit(42)`` so the container manager bounces the process.

Out-of-scope reasons (manual ``/kill``, drawdown, session loss, desync)
keep the bot KILLED-alive forever — those are real safety stops where
the operator should investigate before resuming.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.bot import (
    Bot,
    _AUTO_RESTART_KILL_REASONS,
    _kill_reason_should_auto_restart,
)
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"mm_kar_{os.getpid()}_{uuid.uuid4().hex}.db"


def _make_bot(
    *,
    grace_seconds: float = 0.05,
    flatten_on_kill: bool = False,
) -> tuple[Bot, list[int]]:
    """Build a Bot with the exit_fn replaced by a list-recorder. Returns
    ``(bot, exits)``; ``exits`` collects the codes the bot tried to exit
    with so tests can assert on them.

    ``grace_seconds`` is intentionally tiny (50 ms) so tests don't hang —
    production default is 5 s.
    """
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "PRIVATE_WS_ENABLED": False,
            "FLATTEN_ON_KILL": flatten_on_kill,
            "KILL_AUTO_RESTART_GRACE_SECONDS": grace_seconds,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    exits: list[int] = []
    bot = Bot(
        settings,
        state,
        client,
        storage,
        exit_fn=lambda code: exits.append(code),
    )
    bot._exec.cancel_all_orders_for_symbol = MagicMock()
    bot.flatten = MagicMock()
    return bot, exits


# --- pure helper coverage --------------------------------------------------


def test_helper_recognises_every_auto_restart_reason() -> None:
    """Every reason in the allow-list must round-trip through the helper.
    Catches typo regressions when reasons are added.
    """
    for reason in _AUTO_RESTART_KILL_REASONS:
        assert _kill_reason_should_auto_restart(reason), (
            f"{reason!r} should auto-restart"
        )


def test_helper_rejects_non_environmental_reasons() -> None:
    """Manual / drawdown / session-loss / desync / execution-errors are
    real safety stops — must NOT auto-restart.
    """
    for reason in (
        "manual_kill_via_telegram",
        "max_drawdown",
        "max_session_loss",
        "execution_errors",
        "desync_unrecoverable",
        "killed",
        "",
    ):
        assert not _kill_reason_should_auto_restart(reason), (
            f"{reason!r} should NOT auto-restart"
        )


def test_helper_mixed_reason_with_non_recoverable_token_does_not_restart() -> None:
    """A kill that bundles ``stale_data_kill`` WITH a drawdown breach
    must NOT auto-restart — the operator should look at the
    non-recoverable cause first.
    """
    assert not _kill_reason_should_auto_restart(
        "stale_data_kill,max_drawdown"
    )
    assert not _kill_reason_should_auto_restart(
        "max_session_loss,public_ws_stale_kill"
    )


def test_helper_mixed_reason_all_recoverable_does_restart() -> None:
    """A kill where every comma-separated token is recoverable does
    auto-restart. Mirrors how risk.py joins reasons before kill().
    """
    assert _kill_reason_should_auto_restart(
        "stale_data_kill,public_ws_stale_kill"
    )


# --- end-to-end on Bot.kill -----------------------------------------------


def test_environmental_kill_schedules_exit_42() -> None:
    """The reproducer scenario: stale_data_kill_escalated must result
    in the bot exiting with code 42 after the grace period.
    """
    bot, exits = _make_bot(grace_seconds=0.05)
    bot.kill("stale_data_kill_escalated")
    # Grace 50 ms — wait up to 1 s for the timer to fire.
    deadline = time.time() + 1.0
    while time.time() < deadline and not exits:
        time.sleep(0.01)
    assert exits == [42], (
        f"expected [42], got {exits!r} — auto-restart didn't fire"
    )


def test_manual_kill_does_not_exit() -> None:
    """Operator-triggered ``/kill`` (reason ``manual_kill_via_telegram``)
    must NOT auto-restart. The operator wants the bot dead.
    """
    bot, exits = _make_bot(grace_seconds=0.05)
    bot.kill("manual_kill_via_telegram")
    time.sleep(0.15)  # wait past the grace window
    assert exits == [], (
        f"manual kill must not auto-restart; got exits={exits!r}"
    )


def test_drawdown_kill_does_not_exit() -> None:
    """Capital-protection kill (drawdown / session loss) must NOT
    auto-restart — those are real risk stops.
    """
    bot, exits = _make_bot(grace_seconds=0.05)
    bot.kill("max_drawdown")
    time.sleep(0.15)
    assert exits == []


def test_grace_zero_disables_auto_restart() -> None:
    """``KILL_AUTO_RESTART_GRACE_SECONDS=0`` is the operator's escape
    hatch back to the legacy "stay-KILLED-alive" behaviour. Verify it
    actually disables the exit even on environmental reasons.
    """
    bot, exits = _make_bot(grace_seconds=0.0)
    bot.kill("stale_data_kill_escalated")
    time.sleep(0.15)
    assert exits == []


def test_blind_kill_skips_flatten_but_still_auto_restarts() -> None:
    """The blind-kill guard (no taker-flatten when book is unreliable)
    composes correctly with the auto-restart path. Position is left for
    the operator on the next process run, but the process DOES exit so
    the supervisor brings up a fresh instance.
    """
    bot, exits = _make_bot(grace_seconds=0.05, flatten_on_kill=True)
    bot.kill("stale_data_kill_escalated")
    # Flatten skipped (kill reason is blind).
    bot.flatten.assert_not_called()
    deadline = time.time() + 1.0
    while time.time() < deadline and not exits:
        time.sleep(0.01)
    assert exits == [42]


def test_auto_restart_does_not_block_kill_call() -> None:
    """The auto-restart timer is daemon-non-blocking — ``kill()`` itself
    returns promptly so the caller's ``try/except`` semantics stay
    intact.
    """
    bot, _exits = _make_bot(grace_seconds=2.0)  # long grace
    t0 = time.monotonic()
    bot.kill("stale_data_kill_escalated")
    elapsed = time.monotonic() - t0
    # kill() returns far before the 2 s grace elapses.
    assert elapsed < 0.5, (
        f"kill() blocked for {elapsed:.2f}s — should be near-zero"
    )


def test_state_killed_set_before_exit_scheduled() -> None:
    """Even though we exit shortly after, ``state.killed`` is True from
    the moment the kill returns — the cancel-all path and any in-flight
    quote tick must observe the killed flag.
    """
    bot, _exits = _make_bot(grace_seconds=0.5)
    assert bot._state.killed is False
    bot.kill("stale_data_kill_escalated")
    assert bot._state.killed is True
    assert bot._state.kill_reason == "stale_data_kill_escalated"
