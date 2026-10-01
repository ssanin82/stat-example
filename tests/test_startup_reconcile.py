"""Startup / reconciliation: hydrate resting orders from exchange; risk pause reasons."""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from app.enums import BotStatus, DesyncPhase, RiskAction, Side
from app.execution import OrderManager
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.models import BestBidAsk, PnlSnapshot, ToxicitySnapshot
from app.risk import evaluate_risk
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _mk_settings() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_st_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    return s, path


def test_sync_hydrates_resting_buy_when_local_empty() -> None:
    s, path = _mk_settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(901, s.symbol, Side.BUY, 3000.0, 0.02, 0),
    ]
    om = OrderManager(s, client, storage, state)
    om.sync_open_orders(force=True, emergency=True)
    assert state.working_bid is not None
    assert state.working_bid.order_id_exchange == 901
    assert abs(state.working_bid.price - 3000.0) < 1e-9
    assert state.order_desync is False
    path.unlink(missing_ok=True)


def test_sync_still_desync_on_duplicate_buy() -> None:
    s, path = _mk_settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(1, s.symbol, Side.BUY, 3000.0, 0.01, 0),
        HLOpenOrderRaw(2, s.symbol, Side.BUY, 3001.0, 0.01, 0),
    ]
    om = OrderManager(s, client, storage, state)
    om.sync_open_orders(force=True, emergency=True)
    assert state.order_desync is True
    path.unlink(missing_ok=True)


def test_sync_open_orders_force_respects_global_cooldown() -> None:
    s, path = _mk_settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = []
    om = OrderManager(s, client, storage, state)
    om.sync_open_orders(force=True)
    om.sync_open_orders(force=True)
    assert client.fetch_open_orders_raw.call_count == 1
    path.unlink(missing_ok=True)


def test_sync_open_orders_emergency_also_respects_global_cooldown() -> None:
    s, path = _mk_settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = []
    om = OrderManager(s, client, storage, state)
    om.sync_open_orders(force=True, emergency=True)
    om.sync_open_orders(force=True, emergency=True)
    assert client.fetch_open_orders_raw.call_count == 1
    path.unlink(missing_ok=True)


def test_sync_open_orders_rate_limited_sets_backoff() -> None:
    s, path = _mk_settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.fetch_open_orders_raw.side_effect = RuntimeError("HTTP 429 Too Many Requests")
    om = OrderManager(s, client, storage, state)
    om.sync_open_orders(force=True)
    om.sync_open_orders(force=True)
    assert client.fetch_open_orders_raw.call_count == 1
    path.unlink(missing_ok=True)


def test_maybe_sync_open_orders_tight_loop_hits_reconcile_once() -> None:
    s, path = _mk_settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = []
    om = OrderManager(s, client, storage, state)
    # Make maybe_sync eligible every call.
    om._private_q = object()
    om._private_ws_healthy = True
    om._bot_tick_counter = 99
    s.open_orders_reconcile_request_interval_seconds = 0.01
    om._next_interval_reconcile_request_mono = 0.0
    for _ in range(8):
        om.maybe_sync_open_orders()
    assert client.fetch_open_orders_raw.call_count == 1
    path.unlink(missing_ok=True)


def test_maybe_sync_open_orders_interval_is_wall_clock_not_tick_modulo() -> None:
    s, path = _mk_settings()
    s.open_orders_reconcile_request_interval_seconds = 5.0
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = []
    om = OrderManager(s, client, storage, state)
    om._private_q = object()
    om._private_ws_healthy = True
    om._bot_tick_counter = 10_000

    with patch("app.execution.time.monotonic", return_value=100.0):
        om._next_interval_reconcile_request_mono = 104.0
        om.maybe_sync_open_orders()
    assert client.fetch_open_orders_raw.call_count == 0

    with patch("app.execution.time.monotonic", return_value=105.0):
        om.maybe_sync_open_orders()
    assert client.fetch_open_orders_raw.call_count == 1
    path.unlink(missing_ok=True)


def test_interval_cooldown_skip_not_info_spam(caplog) -> None:
    s, path = _mk_settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = []
    om = OrderManager(s, client, storage, state)
    om._open_orders_reconcile_next_allowed_mono = 999_999.0
    with caplog.at_level(logging.INFO, logger="app.execution"):
        om.request_open_orders_reconcile(reason="interval_time", force=False, emergency=False)
    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert not any("reconcile_skipped reason=interval_time cause=cooldown" in m for m in info_msgs)
    path.unlink(missing_ok=True)


def test_risk_reconcile_stall_reason() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "0x" + "11" * 32,
            "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
        }
    )
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    m = BestBidAsk(
        symbol="ETH",
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        spread_bps=200.0,
    )
    r = evaluate_risk(
        s,
        bot_status=BotStatus.PAUSED,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=tox,
        execution_errors=0,
        desync=False,
        desync_phase=DesyncPhase.OK,
        desync_quarantine_remaining=0,
        trades_last_minute=0,
        reconcile_auto_pause=True,
    )
    assert r.action == RiskAction.NO_QUOTE
    assert "reconcile_stall" in r.reasons


def test_risk_paused_without_reconcile_flag_uses_paused_reason() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "0x" + "11" * 32,
            "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
        }
    )
    pnl = PnlSnapshot(0, 0, 0, 0, 1000.0, 0, 1000.0, utc_now())
    tox = ToxicitySnapshot(0, 0, 0, 1, False, False)
    m = BestBidAsk(
        symbol="ETH",
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        spread_bps=200.0,
    )
    r = evaluate_risk(
        s,
        bot_status=BotStatus.PAUSED,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=m,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl,
        toxicity=tox,
        execution_errors=0,
        desync=False,
        desync_phase=DesyncPhase.OK,
        desync_quarantine_remaining=0,
        trades_last_minute=0,
        reconcile_auto_pause=False,
    )
    assert r.action == RiskAction.NO_QUOTE
    assert "paused" in r.reasons
    assert "reconcile_stall" not in r.reasons
