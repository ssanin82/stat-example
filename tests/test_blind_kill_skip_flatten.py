"""Blind-kill guard: kill reasons that indicate public market data is
unreliable must NOT trigger flatten-on-kill.

Context:
  ``tmp/snap_20260418_170340`` — during an AXS flash crash the public
  WS fell behind, stale-data recovery exhausted its 15 s budget, and
  the bot self-killed with reason ``stale_data_kill_escalated``.
  ``FLATTEN_ON_KILL`` was the default (True) so ``Bot.flatten`` ran,
  and flatten's exit path uses ``client.market_close`` (a TAKER order)
  because its mission is "exit regardless of price." The taker-flatten
  issued into an un-priceable book, paying a $0.025 taker fee on $41
  notional and realising the crash trough as the fill price.

  The kill itself was correct (don't quote while blind). The follow-on
  flatten was self-harm. This file pins the invariant: if we can't see
  the market, we cancel outstanding orders but we don't market-flatten.

Invariant (``app.bot._kill_reason_is_blind`` + ``Bot.kill``):
  * Any kill whose reason contains a token in ``_BLIND_KILL_REASONS``
    → flatten skipped, ``blind_kill_skip_flatten`` event logged.
  * All other kills → flatten runs iff ``FLATTEN_ON_KILL`` is true.
  * ``cancel_all_orders_for_symbol`` is called in BOTH cases — cancel
    by order-id needs no market data.
  * Mixed reasons (blind + non-blind) → blind dominates, flatten skipped.

The operator regains control with the position intact and can call
``/control/flatten`` once public market data has recovered.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.bot import Bot, _BLIND_KILL_REASONS, _kill_reason_is_blind
from app.state import BotState
from app.storage import Storage
from app.enums import BotStatus
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"mm_bk_{os.getpid()}_{uuid.uuid4().hex}.db"


def _make_bot(*, flatten_on_kill: bool) -> Bot:
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "PRIVATE_WS_ENABLED": False,
            "FLATTEN_ON_KILL": flatten_on_kill,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)
    # Stub out the internals we want to observe / block
    bot._exec.cancel_all_orders_for_symbol = MagicMock()
    bot.flatten = MagicMock()
    return bot


# ---------------------- Pure helper ------------------------------------


def test_is_blind_kill_recognises_every_advertised_reason() -> None:
    """Every reason advertised in ``_BLIND_KILL_REASONS`` must be matched
    by the helper. If a new blind reason is added to the set, either it
    works here or this test yells."""
    for reason in _BLIND_KILL_REASONS:
        assert _kill_reason_is_blind(reason), f"{reason!r} should be blind"


def test_is_blind_kill_rejects_non_blind_reasons() -> None:
    for reason in (
        "manual_kill",
        "max_drawdown",
        "max_session_loss",
        "execution_errors",
        "desync_unrecoverable",
        "killed",
        "",
    ):
        assert not _kill_reason_is_blind(reason), f"{reason!r} should NOT be blind"


def test_is_blind_kill_tokenises_comma_joined_reasons() -> None:
    """Risk-module call sites pass ``",".join(reasons)`` — the helper
    must tokenise and match any one token."""
    assert _kill_reason_is_blind("stale_data_kill_threshold,max_drawdown")
    assert _kill_reason_is_blind("max_drawdown,stale_data_kill")
    # Whitespace tolerance around commas.
    assert _kill_reason_is_blind("foo , stale_data_kill , bar")
    # None of the tokens blind.
    assert not _kill_reason_is_blind("manual_kill,max_drawdown")


# ---------------------- Bot.kill behaviour -----------------------------


def test_manual_kill_flattens_when_flatten_on_kill_true() -> None:
    """Baseline: a non-blind kill with FLATTEN_ON_KILL=true still flattens.
    We only want to suppress the flatten on *blind* kills."""
    bot = _make_bot(flatten_on_kill=True)
    bot.kill("manual_kill")
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()
    bot.flatten.assert_called_once_with(blocking=True)
    assert bot._state.bot_status == BotStatus.KILLED


def test_risk_breach_kill_flattens() -> None:
    """Risk-breach reasons (max_drawdown, max_session_loss, etc.) imply
    we should exit, not stay open. Flatten must run."""
    bot = _make_bot(flatten_on_kill=True)
    bot.kill("max_drawdown", {"drawdown_usd": 100.0})
    bot.flatten.assert_called_once_with(blocking=True)


def test_stale_data_kill_escalated_does_not_flatten() -> None:
    """The snap_20260418_170340 regression. Kill fires from
    ``stale_data_kill_escalated`` → flatten must be SKIPPED. Orders are
    still cancelled (safe) and the suppression event is emitted."""
    bot = _make_bot(flatten_on_kill=True)
    bot.kill("stale_data_kill_escalated")
    # Orders cancelled — always safe.
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()
    # Flatten NOT called — we're blind.
    bot.flatten.assert_not_called()


@pytest.mark.parametrize(
    "reason",
    sorted(_BLIND_KILL_REASONS),
)
def test_every_blind_reason_suppresses_flatten(reason: str) -> None:
    """Parametrised: each individual blind reason, in isolation,
    suppresses the flatten."""
    bot = _make_bot(flatten_on_kill=True)
    bot.kill(reason)
    bot.flatten.assert_not_called()
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()


def test_mixed_reasons_with_any_blindness_suppresses_flatten() -> None:
    """If a kill lists both blind and non-blind reasons, the blind flag
    wins — a market-flatten into an un-priceable book is self-harm
    regardless of the accompanying risk-breach reason."""
    bot = _make_bot(flatten_on_kill=True)
    bot.kill("max_drawdown,stale_data_kill_threshold")
    bot.flatten.assert_not_called()
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()


def test_flatten_on_kill_false_never_flattens_regardless_of_reason() -> None:
    """Operator config FLATTEN_ON_KILL=false still suppresses flatten for
    non-blind reasons too — orthogonal to the blind-kill guard."""
    bot = _make_bot(flatten_on_kill=False)
    bot.kill("max_drawdown")
    bot.flatten.assert_not_called()
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()


def test_blind_kill_still_transitions_state_to_killed() -> None:
    """Skipping the flatten must not leak into the state machine:
    ``bot_status`` still moves to KILLED, and ``kill_reason`` is still
    recorded, so ``/kill-state`` tells the truth."""
    bot = _make_bot(flatten_on_kill=True)
    bot.kill("stale_data_kill_escalated")
    assert bot._state.killed is True
    assert bot._state.kill_reason == "stale_data_kill_escalated"
    assert bot._state.bot_status == BotStatus.KILLED
    assert bot._state.kill_timestamp is not None


def test_blind_kill_logs_skip_event(caplog) -> None:
    """The suppression must be visible to operators — we emit a WARNING
    event ``blind_kill_skip_flatten`` so the operator sees that a) the
    bot killed, b) flatten was skipped on purpose, c) they need to
    decide when to unwind."""
    import logging

    bot = _make_bot(flatten_on_kill=True)
    with caplog.at_level(logging.WARNING):
        bot.kill("stale_data_kill_escalated")
    # We log via _log_event which also writes to storage — check any
    # WARNING record mentions the skip event key.
    assert any(
        "blind_kill_skip_flatten" in (r.getMessage() or "")
        for r in caplog.records
    ), "blind_kill_skip_flatten event must be logged"


def test_empty_reason_is_not_blind() -> None:
    """Edge case: empty string reason is not blind (we can't tell — but
    caller shouldn't be calling kill() without a reason anyway). Default
    behaviour is to flatten if FLATTEN_ON_KILL=true."""
    bot = _make_bot(flatten_on_kill=True)
    bot.kill("")
    bot.flatten.assert_called_once_with(blocking=True)
