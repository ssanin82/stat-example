"""Flatten result types, events, residual flags, and bot status."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.bot import Bot
from app.enums import BotStatus, FlattenResult
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _bot(overrides: dict | None = None) -> tuple[Bot, BotState, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_flatten_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    data = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "FLATTEN_TIMEOUT_SECONDS": 120.0,
    }
    if overrides:
        data.update(overrides)
    settings = UnitTestSettings.model_validate(data)
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    return Bot(settings, state, client, storage), state, path


def test_flatten_skipped_already_flat() -> None:
    bot, state, path = _bot()

    def _refresh(*_a, **_k) -> None:
        state.position.position_qty = 0.0

    with patch("app.bot.refresh_account_only", side_effect=_refresh):
        r = bot.flatten(blocking=True)
    assert r == FlattenResult.SKIPPED_ALREADY_FLAT
    assert state.bot_status == BotStatus.RUNNING
    assert state.flatten_incomplete is False
    assert state.flatten_residual_abs_qty == 0.0
    ev = bot._storage.recent_bot_events(20)
    assert any(
        e["event_type"] == "flatten_completed"
        and "SKIPPED_ALREADY_FLAT" in (e.get("payload_json") or "")
        for e in ev
    )
    path.unlink(missing_ok=True)


def test_flatten_completed_via_refresh() -> None:
    bot, state, path = _bot()
    state.position.position_qty = 0.1
    n = {"c": 0}

    def _refresh(*_a, **_k) -> None:
        n["c"] += 1
        state.position.position_qty = 0.1 if n["c"] == 1 else 0.0

    with patch("app.bot.refresh_account_only", side_effect=_refresh):
        r = bot.flatten(blocking=True)
    assert r == FlattenResult.COMPLETED
    assert state.bot_status == BotStatus.RUNNING
    assert state.flatten_incomplete is False
    ev = bot._storage.recent_bot_events(20)
    assert any(
        e["event_type"] == "flatten_completed"
        and "COMPLETED" in (e.get("payload_json") or "")
        for e in ev
    )
    path.unlink(missing_ok=True)


def test_flatten_timed_out_sets_residual() -> None:
    bot, state, path = _bot(
        {"FLATTEN_TIMEOUT_SECONDS": 0.0, "FLATTEN_ON_KILL": False}
    )
    state.position.position_qty = 0.05

    def _refresh(*_a, **_k) -> None:
        state.position.position_qty = 0.05

    with patch("app.bot.refresh_account_only", side_effect=_refresh):
        r = bot.flatten(blocking=True)
    assert r == FlattenResult.TIMED_OUT
    assert state.bot_status == BotStatus.PAUSED
    assert state.flatten_incomplete is True
    assert state.flatten_residual_abs_qty > 1e-9
    ev = bot._storage.recent_bot_events(20)
    assert any(e["event_type"] == "flatten_timed_out" for e in ev)
    path.unlink(missing_ok=True)


def test_flatten_failed_no_write_access() -> None:
    bot, state, path = _bot()
    bot._client.has_write_access.return_value = False
    state.position.position_qty = 0.02
    with patch("app.bot.refresh_account_only"):
        r = bot.flatten(blocking=True)
    assert r == FlattenResult.FAILED
    assert state.bot_status == BotStatus.PAUSED
    assert state.flatten_incomplete is True
    ev = bot._storage.recent_bot_events(20)
    assert any(e["event_type"] == "flatten_failed" for e in ev)
    path.unlink(missing_ok=True)


def test_flatten_drain_skips_reconcile_when_flat_and_no_open_orders() -> None:
    bot, state, path = _bot()
    state.position.position_qty = 0.0
    state.working_bid = None
    state.working_ask = None
    with patch.object(bot._exec, "request_open_orders_reconcile") as req:
        bot._flatten_drain_private_ws()
    req.assert_not_called()
    path.unlink(missing_ok=True)


def test_flatten_drain_requests_reconcile_when_inventory_or_orders_exist() -> None:
    bot, state, path = _bot()
    state.position.position_qty = 0.01
    with patch.object(bot._exec, "request_open_orders_reconcile") as req:
        bot._flatten_drain_private_ws()
    req.assert_called_once()
    path.unlink(missing_ok=True)


def test_flatten_drain_requested_reconcile_still_obeys_cooldown() -> None:
    bot, state, path = _bot()
    state.position.position_qty = 0.01
    # Call twice; first triggers reconcile, second should be cooldown-skipped.
    bot._flatten_drain_private_ws()
    bot._flatten_drain_private_ws()
    assert bot._client.fetch_open_orders_raw.call_count == 1
    path.unlink(missing_ok=True)
