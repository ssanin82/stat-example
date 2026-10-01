"""Session boundary: replay fills must not affect session PnL / risk inputs."""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.enums import Side
from app.exchange.hyperliquid_types import HLFillRaw
from app.fill_ingestion import ingest_hl_fill_raw
from app.models import PositionSnapshot
from app.pnl import PnlTracker
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_sess_{os.getpid()}_{uuid.uuid4().hex}.db"
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


def _raw(
    settings: UnitTestSettings,
    *,
    fill_id: str,
    time_ms: int,
    closed_pnl: float = -50.0,
    fee: float = 0.02,
) -> HLFillRaw:
    return HLFillRaw(
        fill_id=fill_id,
        oid=1,
        coin=settings.symbol,
        side=Side.BUY,
        px=100.0,
        sz=0.1,
        fee=fee,
        time_ms=time_ms,
        closed_pnl=closed_pnl,
        raw={},
    )


def test_replay_fill_before_session_start_does_not_move_pnl() -> None:
    settings, path = _settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    t_session = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    state.session_started_at_utc = t_session
    t_old = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    old_ms = int(t_old.timestamp() * 1000)

    pnl = PnlTracker()
    fr = _raw(settings, fill_id="hist_a", time_ms=old_ms, closed_pnl=-99.0, fee=0.05)
    assert ingest_hl_fill_raw(
        state=state,
        storage=storage,
        pnl=pnl,
        symbol=settings.symbol,
        fr=fr,
        source="rest",
    )
    pos = PositionSnapshot(settings.symbol, 0.0, None, None, 0.0, 0.0)
    snap = pnl.build_snapshot(pos, 10_000.0)
    assert abs(snap.realized_pnl_usd) < 1e-9
    assert abs(snap.fees_usd) < 1e-9
    with state._lock:
        assert len(state.recent_fills) == 0
    rows = storage.recent_fills(5)
    assert len(rows) == 1
    path.unlink(missing_ok=True)


def test_fill_after_session_start_updates_pnl_normally() -> None:
    settings, path = _settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    t_session = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    state.session_started_at_utc = t_session
    t_new = datetime(2026, 4, 1, 13, 0, 0, tzinfo=timezone.utc)
    new_ms = int(t_new.timestamp() * 1000)

    pnl = PnlTracker()
    fr = _raw(settings, fill_id="live_a", time_ms=new_ms, closed_pnl=1.25, fee=0.03)
    assert ingest_hl_fill_raw(
        state=state,
        storage=storage,
        pnl=pnl,
        symbol=settings.symbol,
        fr=fr,
        source="rest",
    )
    pos = PositionSnapshot(settings.symbol, 0.0, None, None, 0.0, 0.0)
    snap = pnl.build_snapshot(pos, 10_000.0)
    assert abs(snap.realized_pnl_usd - 1.25) < 1e-9
    assert abs(snap.fees_usd - 0.03) < 1e-9
    with state._lock:
        assert len(state.recent_fills) == 1
    path.unlink(missing_ok=True)


def test_replay_fill_idempotent_no_double_pnl() -> None:
    settings, path = _settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    t_session = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    state.session_started_at_utc = t_session
    t_new = datetime(2026, 4, 1, 14, 0, 0, tzinfo=timezone.utc)
    new_ms = int(t_new.timestamp() * 1000)

    pnl = PnlTracker()
    fr = _raw(settings, fill_id="live_dup", time_ms=new_ms, closed_pnl=2.0, fee=0.01)
    assert ingest_hl_fill_raw(
        state=state,
        storage=storage,
        pnl=pnl,
        symbol=settings.symbol,
        fr=fr,
        source="rest",
    )
    assert not ingest_hl_fill_raw(
        state=state,
        storage=storage,
        pnl=pnl,
        symbol=settings.symbol,
        fr=fr,
        source="rest_catchup",
    )
    pos = PositionSnapshot(settings.symbol, 0.0, None, None, 0.0, 0.0)
    snap = pnl.build_snapshot(pos, 10_000.0)
    assert abs(snap.realized_pnl_usd - 2.0) < 1e-9
    assert abs(snap.fees_usd - 0.01) < 1e-9
    path.unlink(missing_ok=True)


def test_mixed_replay_and_session_totals_only_session_component() -> None:
    settings, path = _settings()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    t_session = datetime(2026, 4, 10, 0, 0, 0, tzinfo=timezone.utc)
    state.session_started_at_utc = t_session
    old_ms = int(datetime(2026, 4, 9, 0, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    new_ms = int(datetime(2026, 4, 10, 1, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)

    pnl = PnlTracker()
    ingest_hl_fill_raw(
        state=state,
        storage=storage,
        pnl=pnl,
        symbol=settings.symbol,
        fr=_raw(settings, fill_id="old", time_ms=old_ms, closed_pnl=-500.0),
        source="rest",
    )
    ingest_hl_fill_raw(
        state=state,
        storage=storage,
        pnl=pnl,
        symbol=settings.symbol,
        fr=_raw(settings, fill_id="new", time_ms=new_ms, closed_pnl=-3.0),
        source="rest",
    )
    pos = PositionSnapshot(settings.symbol, 0.0, None, None, 0.0, 0.0)
    snap = pnl.build_snapshot(pos, 10_000.0)
    assert abs(snap.realized_pnl_usd - (-3.0)) < 1e-9
    path.unlink(missing_ok=True)
