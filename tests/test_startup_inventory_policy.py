"""
Targeted tests for post-restart inventory policy wired to ``Bot.one_tick`` and ``main`` startup.

These exercise the same control paths as production: ``refresh_account_only`` + public BBO → snapshot
health / reconcile stall → ``evaluate_risk`` → optional ``flatten`` / quote persistence.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.bot import Bot
from app.enums import BotStatus, FlattenResult, Side
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot, ToxicitySnapshot
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"mm_sip_{os.getpid()}_{uuid.uuid4().hex}.db"


def _bba(sym: str, mid: float = 100.0) -> BestBidAsk:
    return BestBidAsk(
        symbol=sym,
        best_bid=mid - 1.0,
        best_ask=mid + 1.0,
        mid_price=mid,
        spread_bps=200.0,
    )


def _pos(sym: str, qty: float, mid: float = 100.0) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=sym,
        position_qty=qty,
        avg_entry_price=mid,
        mark_price=mid,
        position_notional=abs(qty) * mid,
        unrealized_pnl_usd=0.0,
    )


def _acct() -> AccountSnapshot:
    return AccountSnapshot(
        equity_usd=10_000.0,
        cash_usd=5000.0,
        withdrawable_usd=5000.0,
    )


def _wire_client(
    client: MagicMock,
    settings: UnitTestSettings,
    *,
    position_qty: float = 0.0,
    account_fn=None,
) -> None:
    """REST doubles for a single-symbol live account (mirrors ``refresh_account_only``)."""
    sym = settings.symbol
    client.has_write_access.return_value = True
    client.fetch_best_bid_ask.return_value = _bba(sym, 100.0)
    client.fetch_position.return_value = _pos(sym, position_qty, 100.0)
    if account_fn is None:
        client.fetch_account_snapshot.return_value = _acct()
    else:
        client.fetch_account_snapshot.side_effect = account_fn
    client.fetch_recent_fills_raw.return_value = []
    client.fetch_open_orders_raw.return_value = []


def _bot_with_mocks(
    *,
    position_qty: float = 0.0,
    account_fn=None,
    from_starting: bool = False,
    **settings_kw: object,
) -> tuple[Bot, BotState, Path, UnitTestSettings, MagicMock]:
    path = _db_path()
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 0.05,
        "MAX_POSITION_NOTIONAL_USD": 50_000.0,
        "INVENTORY_SKEW_COEFF_BPS": 100.0,
        "PRIVATE_WS_ENABLED": False,
    }
    data.update(settings_kw)
    settings = UnitTestSettings.model_validate(data)
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    bb = _bba(settings.symbol, 100.0)
    state.apply_market_book_only(bb, market_data_source="test_seed")
    with state._lock:
        state.public_ws_last_message_wall_ts = utc_now()
        state.public_ws_connected = True
        state.public_ws_seen_first_bbo = True
    if not from_starting:
        state.bot_status = BotStatus.RUNNING
    client = mock_mm_client()
    _wire_client(client, settings, position_qty=position_qty, account_fn=account_fn)
    bot = Bot(settings, state, client, storage)
    return bot, state, path, settings, client


def _tick_no_exchange_io(bot: Bot) -> None:
    """Avoid placing/cancelling on the mock; still runs real reconcile + risk + quote path."""
    with (
        patch.object(bot._exec, "maybe_sync_open_orders"),
        patch.object(bot._exec, "maybe_refresh_quotes"),
        patch.object(bot._exec, "cancel_resting_for_risk"),
        patch.object(bot, "_persist_snapshots"),
    ):
        bot.one_tick()


def test_starting_promotes_to_running_after_first_healthy_reconcile() -> None:
    """Fresh process state is STARTING; first good tick reconciles and promotes (no startup flatten)."""
    bot, state, path, _, _ = _bot_with_mocks(
        position_qty=0.03,
        from_starting=True,
    )
    assert state.bot_status == BotStatus.STARTING
    with (
        patch.object(bot._exec, "maybe_sync_open_orders"),
        patch.object(bot._exec, "maybe_refresh_quotes"),
        patch.object(bot._exec, "cancel_resting_for_risk"),
        patch.object(bot, "_persist_snapshots"),
        patch.object(bot, "flatten") as flat_m,
    ):
        bot.one_tick()
        flat_m.assert_not_called()
    assert state.bot_status == BotStatus.RUNNING
    assert abs(state.position.position_qty - 0.03) < 1e-9
    ev = bot._storage.recent_bot_events(15)
    assert any(e.get("event_type") == "startup_reconciled" for e in ev)
    path.unlink(missing_ok=True)


def test_inherited_nonzero_position_one_tick_no_flatten_stays_running() -> None:
    """After REST reconciliation, non-zero inventory does not trigger flatten or PAUSED."""
    bot, state, path, settings, _client = _bot_with_mocks(position_qty=0.02)
    with patch.object(bot, "flatten") as flat_m:
        _tick_no_exchange_io(bot)
        flat_m.assert_not_called()
    assert state.bot_status == BotStatus.RUNNING
    assert not state.reconcile_auto_pause
    assert abs(state.position.position_qty - 0.02) < 1e-9
    assert state.order_desync is False
    path.unlink(missing_ok=True)


def test_flat_position_healthy_one_tick_stays_running_not_paused() -> None:
    bot, state, path, _, _ = _bot_with_mocks(position_qty=0.0)
    _tick_no_exchange_io(bot)
    assert state.bot_status == BotStatus.RUNNING
    assert not state.reconcile_auto_pause
    assert abs(state.position.position_qty) < 1e-9
    path.unlink(missing_ok=True)


def test_reconcile_stall_pauses_after_sustained_unhealthy_snapshot() -> None:
    """
    Mirrors ``Bot.one_tick`` reconcile stall: account snapshot missing while address is configured
    counts as unhealthy until ``exchange_reconcile_stall_ticks`` consecutive bad ticks.
    """
    n_acct = {"i": 0}

    def acct(addr: str) -> AccountSnapshot | None:
        n_acct["i"] += 1
        if n_acct["i"] <= 2:
            return _acct()
        return None

    bot, state, path, _, _ = _bot_with_mocks(
        position_qty=0.0,
        account_fn=acct,
        EXCHANGE_RECONCILE_STALL_TICKS=2,
        # Each tick must run account refresh so None from acct() is applied. With PRIVATE_WS off,
        # REST fill ingest forces the unhealthy min interval (else 12s) unless lowered here.
        ACCOUNT_REST_MIN_INTERVAL_SECONDS=0.001,
        UNHEALTHY_ACCOUNT_REST_MIN_INTERVAL_SECONDS=0.001,
    )
    with (
        patch.object(bot._exec, "maybe_sync_open_orders"),
        patch.object(bot._exec, "maybe_refresh_quotes"),
        patch.object(bot._exec, "cancel_resting_for_risk"),
        patch.object(bot, "_persist_snapshots"),
    ):
        bot.one_tick()
        bot.one_tick()
        assert state.bot_status == BotStatus.RUNNING
        bot.one_tick()
        assert state.exchange_snapshot_unhealthy_streak >= 1
        bot.one_tick()
    assert state.bot_status == BotStatus.PAUSED
    assert state.reconcile_auto_pause is True
    ev = bot._storage.recent_bot_events(20)
    assert any(e.get("event_type") == "reconcile_stall_pause" for e in ev)
    path.unlink(missing_ok=True)


def test_order_desync_triggers_cancel_all_path_not_flatten() -> None:
    """Persistent duplicate opens → ``order_desync`` / CANCEL_ALL; not startup flatten."""
    bot, state, path, settings, client = _bot_with_mocks(position_qty=0.0)
    client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(1, settings.symbol, Side.BUY, 99.0, 0.01, 0),
        HLOpenOrderRaw(2, settings.symbol, Side.BUY, 99.5, 0.01, 0),
    ]
    with patch.object(bot, "flatten") as flat_m:
        with (
            patch.object(bot._exec, "maybe_refresh_quotes"),
            patch.object(bot, "_persist_snapshots"),
        ):
            bot.one_tick()
        flat_m.assert_not_called()
    assert state.order_desync is True
    path.unlink(missing_ok=True)


def test_flatten_mode_triggers_flatten_exceptional_not_inventory() -> None:
    """``evaluate_risk`` returns FLATTEN when ``flatten_mode`` is set (operational flatten)."""
    bot, state, path, _, _ = _bot_with_mocks(position_qty=0.01)
    state.flatten_mode = True
    with patch.object(bot, "flatten") as flat_m:
        _tick_no_exchange_io(bot)
        flat_m.assert_called_once()
    path.unlink(missing_ok=True)


def test_toxicity_hard_with_position_triggers_soft_flatten_not_taker() -> None:
    """Risk path: toxicity hard + non-zero inventory → SOFT_FLATTEN
    (post-only worker via ``_enter_soft_flatten``), NOT FLATTEN
    (taker market_close).

    History: this used to call ``bot.flatten()`` (the taker path),
    gated on ``FLATTEN_ON_KILL=True``. Operator hit it on TON
    2026-05-07 and lost ~$0.13 across three taker exits in a 30-min
    window. Redesigned so toxicity-trigger always uses the post-
    only soft-flatten worker (with optional taker fallback after
    adverse drift exceeds a threshold). ``FLATTEN_ON_KILL`` no
    longer gates this path — it only affects kill-event flattening.
    See plans/20260507-sf-frontend.md Phase 6.
    """
    bot, state, path, _, _ = _bot_with_mocks(
        position_qty=0.02,
        FLATTEN_ON_KILL=True,  # explicitly: DOES NOT re-enable taker on toxicity
        TOXICITY_ENABLED=True,
    )
    hard = ToxicitySnapshot(
        score=1.0,
        one_sided_fill_ratio=1.0,
        avg_adverse_markout_bps=-50.0,
        vol_spike_ratio=1.0,
        hard_trigger=True,
        soft_trigger=True,
        toxic_side=Side.BUY,
        delayed_markout_sample_count=1,
        adverse_uses_delayed_markouts=True,
    )
    with patch.object(bot._tox, "snapshot", return_value=hard):
        with patch.object(bot, "flatten") as flat_m:
            with patch.object(bot, "_enter_soft_flatten") as sf_m:
                _tick_no_exchange_io(bot)
                # Taker path NOT called.
                flat_m.assert_not_called()
                # Soft-flatten path IS called, with force_phase=2
                # (toxicity wants aggressive starting price).
                sf_m.assert_called_once()
                kw = sf_m.call_args.kwargs
                assert kw.get("force_phase") == 2
                assert kw.get("trigger_reason") == "toxicity_hard_trigger"
    path.unlink(missing_ok=True)


def test_toxicity_hard_on_flat_position_suppresses_side_does_not_flatten() -> None:
    """Regression guard for the flatten-loop bug observed 2026-04-25:
    when toxicity hard-triggers on an already-flat position, the bot
    must NOT call ``flatten()``. Calling flatten on a flat position
    creates a tight loop (cancel-all → orphan WS events → reconcile →
    flatten → ...) that produced 3915 flatten_started events in 13
    minutes and triggered the deadlock watchdog every 20-30 minutes.

    Correct behaviour: fall through to side-suppression
    (BID_ONLY / ASK_ONLY depending on which side is toxic). Same
    risk-control intent, no busy loop.
    """
    bot, state, path, _, _ = _bot_with_mocks(
        position_qty=0.0,
        FLATTEN_ON_KILL=True,
        TOXICITY_ENABLED=True,
    )
    hard = ToxicitySnapshot(
        score=1.0,
        one_sided_fill_ratio=1.0,
        avg_adverse_markout_bps=-50.0,
        vol_spike_ratio=1.0,
        hard_trigger=True,
        soft_trigger=True,
        toxic_side=Side.BUY,
        delayed_markout_sample_count=1,
        adverse_uses_delayed_markouts=True,
    )
    with patch.object(bot._tox, "snapshot", return_value=hard):
        with patch.object(bot, "flatten") as flat_m:
            _tick_no_exchange_io(bot)
            flat_m.assert_not_called()
    path.unlink(missing_ok=True)


def test_max_abs_position_exceeded_one_sided_not_flatten() -> None:
    """Inventory cap → one-sided quoting, not flatten (post-policy safety)."""
    bot, state, path, _, _ = _bot_with_mocks(position_qty=0.06)
    with patch.object(bot, "flatten") as flat_m:
        _tick_no_exchange_io(bot)
        flat_m.assert_not_called()
    path.unlink(missing_ok=True)


def test_flatten_completed_auto_resumes_running_not_paused() -> None:
    """After successful flatten, ``finally`` restores ``RUNNING`` (single-bot continuity)."""
    bot, state, path, _, _ = _bot_with_mocks(position_qty=0.1)
    n = {"c": 0}

    def _refresh(*_a, **_k) -> None:
        n["c"] += 1
        state.position.position_qty = 0.1 if n["c"] == 1 else 0.0

    with patch("app.bot.refresh_account_only", side_effect=_refresh):
        r = bot.flatten(blocking=True)
    assert r == FlattenResult.COMPLETED
    assert state.bot_status == BotStatus.RUNNING
    path.unlink(missing_ok=True)


def test_cancel_all_on_startup_main_snippet_does_not_invoke_flatten() -> None:
    """
    Mirrors ``app.main`` lifespan: ``cancel_all_on_startup`` calls the
    bulk-or-fallback cancel wrapper. Same boolean guard as production;
    cancel-all alone must not call ``flatten``.
    """
    path = _db_path()
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "CANCEL_ALL_ON_STARTUP": True,
            "PRIVATE_WS_ENABLED": False,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)
    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback = MagicMock(return_value="bulk_ok")
    bot.flatten = MagicMock()

    if (
        settings.cancel_all_on_startup
        and settings.trading_enabled
        and client.has_write_access()
    ):
        bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback()

    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback.assert_called_once()
    bot.flatten.assert_not_called()
    path.unlink(missing_ok=True)


def test_persisted_quote_uses_reconciled_position_for_inventory_and_reservation() -> None:
    """
    ``compute_quote_decision`` receives ``pos.position_qty`` from the same tick's snapshot;
    persisted row must reflect inherited inventory skew vs flat.
    """
    mid = 100.0
    skew_bps = 100.0
    max_abs = 0.05

    def run_tick(qty: float) -> dict:
        path = _db_path()
        path.unlink(missing_ok=True)
        settings = UnitTestSettings.model_validate(
            {
                "TRADING_ENABLED": True,
                "HL_SECRET_KEY": "x",
                "HL_ACCOUNT_ADDRESS": "0xabc",
                "DATABASE_URL": f"sqlite:///{path.as_posix()}",
                "MAX_ABS_POSITION": max_abs,
                "INVENTORY_SKEW_COEFF_BPS": skew_bps,
                "PRIVATE_WS_ENABLED": False,
            }
        )
        storage = Storage(settings)
        storage.init_schema()
        state = BotState(settings)
        state.bot_status = BotStatus.RUNNING
        bb = _bba(settings.symbol, mid)
        state.apply_market_book_only(bb, market_data_source="test_seed")
        with state._lock:
            state.public_ws_last_message_wall_ts = utc_now()
            state.public_ws_connected = True
            state.public_ws_seen_first_bbo = True
        client = mock_mm_client()
        _wire_client(client, settings, position_qty=qty)
        bot = Bot(settings, state, client, storage)
        with (
            patch.object(bot._exec, "maybe_sync_open_orders"),
            patch.object(bot._exec, "maybe_refresh_quotes"),
            patch.object(bot, "_persist_snapshots"),
        ):
            bot.one_tick()
        rows = storage.recent_quote_decisions(1)
        path.unlink(missing_ok=True)
        assert rows, "expected quote_decisions row for ALLOW path"
        return rows[0]

    flat = run_tick(0.0)
    long = run_tick(0.02)
    assert abs(flat["inventory"]) < 1e-9
    assert abs(long["inventory"] - 0.02) < 1e-9
    assert abs(flat["reservation_price"] - mid) < 1e-6
    # Long inventory skews reservation down (see ``app.quoting.compute_quote_decision``).
    assert long["reservation_price"] < flat["reservation_price"]


def test_reconcile_recovered_auto_resumes_from_stall_pause() -> None:
    """After PAUSED(reconcile_stall), a healthy tick clears stall and returns to RUNNING."""
    n_acct = {"i": 0}

    def acct(addr: str) -> AccountSnapshot | None:
        n_acct["i"] += 1
        if n_acct["i"] <= 2:
            return _acct()
        return None

    bot, state, path, _, client = _bot_with_mocks(
        position_qty=0.0,
        account_fn=acct,
        EXCHANGE_RECONCILE_STALL_TICKS=2,
        ACCOUNT_REST_MIN_INTERVAL_SECONDS=0.001,
        UNHEALTHY_ACCOUNT_REST_MIN_INTERVAL_SECONDS=0.001,
    )
    with (
        patch.object(bot._exec, "maybe_sync_open_orders"),
        patch.object(bot._exec, "maybe_refresh_quotes"),
        patch.object(bot._exec, "cancel_resting_for_risk"),
        patch.object(bot, "_persist_snapshots"),
    ):
        bot.one_tick()
        bot.one_tick()
        bot.one_tick()
        bot.one_tick()
    assert state.bot_status == BotStatus.PAUSED
    client.fetch_account_snapshot.side_effect = None
    client.fetch_account_snapshot.return_value = _acct()
    with (
        patch.object(bot._exec, "maybe_sync_open_orders"),
        patch.object(bot._exec, "maybe_refresh_quotes"),
        patch.object(bot._exec, "cancel_resting_for_risk"),
        patch.object(bot, "_persist_snapshots"),
    ):
        bot.one_tick()
    assert state.bot_status == BotStatus.RUNNING
    assert state.reconcile_auto_pause is False
    ev = bot._storage.recent_bot_events(30)
    assert any(e.get("event_type") == "reconcile_recovered" for e in ev)
    path.unlink(missing_ok=True)
