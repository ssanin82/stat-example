"""Desync phases, events, quarantine, and unrecoverable path."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from app.enums import BotStatus, DesyncPhase, RiskAction, Side
from app.execution import OrderManager
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.models import PnlSnapshot, ToxicitySnapshot
from app.risk import evaluate_risk
from app.state import BotState
from app.utils.time import utc_now
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup(
    **settings_kw: object,
) -> tuple[UnitTestSettings, Path, BotState, OrderManager, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_desync_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "DESYNC_UNRECOVERABLE_AFTER_TICKS": 3,
        "DESYNC_QUARANTINE_TICKS": 1,
    }
    for k, v in settings_kw.items():
        key = k.upper() if k.islower() else k
        data[key] = v
    s = UnitTestSettings.model_validate(data)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    om = OrderManager(s, client, storage, state)
    return s, path, state, om, storage


def _pnl() -> PnlSnapshot:
    return PnlSnapshot(
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
        total_pnl_usd=0.0,
        fees_usd=0.0,
        equity_usd=1000.0,
        drawdown_usd=0.0,
        session_peak_equity_usd=1000.0,
        ts=utc_now(),
    )


def test_desync_detected_then_reconciling_events() -> None:
    s, path, state, om, storage = _setup()
    sym = s.symbol
    dup = [
        HLOpenOrderRaw(1, sym, Side.BUY, 100.0, 0.1, 0),
        HLOpenOrderRaw(2, sym, Side.BUY, 100.0, 0.1, 0),
    ]
    om._client.fetch_open_orders_raw.return_value = dup
    om.sync_open_orders(force=True, emergency=True)
    assert state.order_desync is True
    assert state.desync_phase == DesyncPhase.DETECTED
    ev = storage.recent_bot_events(5)
    assert any(e["event_type"] == "desync_detected" for e in ev)

    # Reconcile calls are globally cooldown-gated; simulate time passing.
    om._open_orders_reconcile_next_allowed_mono = 0.0
    om.sync_open_orders(force=True, emergency=True)
    assert state.desync_phase == DesyncPhase.RECONCILING
    ev2 = storage.recent_bot_events(10)
    assert any(e["event_type"] == "desync_reconciling" for e in ev2)
    path.unlink(missing_ok=True)


def test_recovery_quarantine_then_ok() -> None:
    s, path, state, om, storage = _setup()
    sym = s.symbol
    client = om._client
    client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(1, sym, Side.BUY, 100.0, 0.1, 0),
        HLOpenOrderRaw(2, sym, Side.BUY, 100.0, 0.1, 0),
    ]
    om.sync_open_orders(force=True, emergency=True)
    client.fetch_open_orders_raw.return_value = []
    # Reconcile calls are globally cooldown-gated; simulate time passing.
    om._open_orders_reconcile_next_allowed_mono = 0.0
    om.sync_open_orders(force=True, emergency=True)
    assert state.order_desync is False
    assert state.desync_phase == DesyncPhase.RECOVERED
    assert state.desync_quarantine_remaining == 1

    r = evaluate_risk(
        s,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=None,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=_pnl(),
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=False,
        desync_phase=state.desync_phase,
        desync_quarantine_remaining=state.desync_quarantine_remaining,
    )
    assert r.action == RiskAction.NO_QUOTE
    assert "desync_quarantine" in r.reasons

    # Reconcile calls are globally cooldown-gated; simulate time passing.
    om._open_orders_reconcile_next_allowed_mono = 0.0
    om.sync_open_orders(force=True, emergency=True)
    assert state.desync_quarantine_remaining == 0
    assert state.desync_phase == DesyncPhase.OK
    path.unlink(missing_ok=True)


def test_unrecoverable_after_ticks() -> None:
    s, path, state, om, storage = _setup(DESYNC_UNRECOVERABLE_AFTER_TICKS=1)
    sym = s.symbol
    om._client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(1, sym, Side.BUY, 100.0, 0.1, 0),
        HLOpenOrderRaw(2, sym, Side.BUY, 100.0, 0.1, 0),
    ]
    om.sync_open_orders(force=True, emergency=True)
    assert state.desync_phase == DesyncPhase.UNRECOVERABLE
    ev = storage.recent_bot_events(5)
    assert any(e["event_type"] == "desync_unrecoverable" for e in ev)

    r = evaluate_risk(
        s,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=None,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=_pnl(),
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=0,
        desync=state.order_desync,
        desync_phase=state.desync_phase,
        desync_quarantine_remaining=state.desync_quarantine_remaining,
    )
    assert r.action == RiskAction.KILL
    assert "desync_unrecoverable" in r.reasons
    path.unlink(missing_ok=True)
