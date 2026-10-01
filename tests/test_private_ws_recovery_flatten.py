"""Private WS queue overflow recovery and flatten interaction with event draining."""

from __future__ import annotations

import os
import queue
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from app.bot import Bot
from app.enums import BotStatus, FlattenResult, Side
from app.execution import OrderManager
from app.exchange.hyperliquid_types import HLFillRaw
from app.fill_ingestion import ingest_hl_fill_raw
from app.state import BotState
from app.storage import Storage
from app.pnl import PnlTracker
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings_db(**kw: object) -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_pws_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "PRIVATE_WS_ENABLED": True,
        "PRIVATE_WS_RECOVERY_FILL_CATCHUP_TICKS": 5,
    }
    for k, v in kw.items():
        data[k.upper() if k.islower() else k] = v
    return UnitTestSettings.model_validate(data), path


def test_queue_drop_makes_rest_fill_ingest_same_tick_before_drain() -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    pq = queue.Queue()
    om = OrderManager(settings, client, storage, state, private_event_queue=pq)
    om._private_ws_healthy = True
    om.on_bot_tick_start()
    state.note_private_ws_queue_drop("fill")
    assert om.should_ingest_fills_via_rest() is True
    om.drain_private_events(None)
    with state._lock:
        assert state.private_ws_recovery_pending is False
        assert state.private_ws_queue_drops == 1
    rows = storage.recent_bot_events(10)
    assert any(r["event_type"] == "private_ws_queue_overflow_recovery" for r in rows)
    path.unlink(missing_ok=True)


def test_flatten_calls_drain_private_events_while_looping() -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage, private_event_queue=None)
    state.position.position_qty = 0.05
    n_drain = 0
    orig = bot._exec.drain_private_events

    def track(pnl):
        nonlocal n_drain
        n_drain += 1
        return orig(pnl)

    k = {"i": 0}

    def refresh(*_a, **_k):
        k["i"] += 1
        if k["i"] >= 3:
            state.position.position_qty = 0.0

    with patch.object(bot._exec, "drain_private_events", side_effect=track):
        with patch("app.bot.refresh_account_only", side_effect=refresh):
            with patch.object(bot._client, "market_close", return_value={"status": "ok"}):
                r = bot.flatten(blocking=True)
    assert r == FlattenResult.COMPLETED
    assert n_drain >= 3
    path.unlink(missing_ok=True)


def test_rest_reingest_same_fill_id_does_not_double_session_pnl() -> None:
    """Simulates WS drop + REST catchup: duplicate fill_id must not double-count."""
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    pnl = PnlTracker()
    sym = settings.symbol
    fr = HLFillRaw(
        fill_id="h_1000",
        oid=1,
        coin=sym,
        side=Side.BUY,
        px=100.0,
        sz=0.01,
        fee=0.0,
        time_ms=2_000_000_000_000,
        closed_pnl=0.0,
        raw={},
    )
    ingest_hl_fill_raw(
        state=state,
        storage=storage,
        pnl=pnl,
        symbol=sym,
        fr=fr,
        source="rest_catchup",
    )
    r1 = pnl.build_snapshot(state.position, None).total_pnl_usd
    ingest_hl_fill_raw(
        state=state,
        storage=storage,
        pnl=pnl,
        symbol=sym,
        fr=fr,
        source="rest",
    )
    r2 = pnl.build_snapshot(state.position, None).total_pnl_usd
    assert abs(r1 - r2) < 1e-12
    path.unlink(missing_ok=True)


def test_bot_status_flags_include_queue_drop_counters() -> None:
    settings, path = _settings_db()
    state = BotState(settings)
    state.note_private_ws_queue_drop("order_update")
    flags = state.status_flags_dict()
    assert flags["private_ws_queue_drops"] == 1
    assert flags["private_ws_recovery_pending"] is True
    path.unlink(missing_ok=True)
