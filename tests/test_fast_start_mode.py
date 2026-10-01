from __future__ import annotations

import os
import queue
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from app.enums import BotStatus, Side
from app.exchange.hyperliquid_types import HLFillRaw
from app.exchange.private_events import PrivateFillEvent
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_startup_inventory_policy import _bba, _bot_with_mocks


def _hist_fill_raw(*, sym: str, fill_id: str, time_ms: int) -> HLFillRaw:
    return HLFillRaw(
        fill_id=fill_id,
        oid=None,
        coin=sym,
        side=Side.BUY,
        px=100.0,
        sz=0.01,
        fee=0.0,
        time_ms=time_ms,
        closed_pnl=0.0,
        raw={},
    )


def test_fast_start_skips_rest_fill_replay_and_reaches_running() -> None:
    bot, state, path, settings, client = _bot_with_mocks(
        position_qty=0.0,
        from_starting=True,
        FAST_START_SKIP_HISTORICAL_FILL_REPLAY=True,
        CANCEL_ALL_ON_STARTUP=True,
    )
    assert state.bot_status == BotStatus.STARTING

    t_session = datetime(2026, 4, 16, 0, 0, 0, tzinfo=timezone.utc)
    state.session_started_at_utc = t_session

    # Historical (pre-session) fill rows should never be fetched/ingested during STARTING
    # in fast-start mode.
    t_old_ms = int((t_session - timedelta(days=3)).timestamp() * 1000.0)
    client.fetch_recent_fills_raw.return_value = [
        _hist_fill_raw(sym=settings.symbol, fill_id="hist1", time_ms=t_old_ms)
    ]

    with (
        patch.object(bot._exec, "maybe_refresh_quotes"),
        patch.object(bot._exec, "cancel_resting_for_risk"),
        patch.object(bot, "_persist_snapshots"),
    ):
        bot.one_tick()

    assert state.bot_status == BotStatus.RUNNING
    assert client.fetch_recent_fills_raw.call_count == 0
    assert state.fast_start_mode_enabled is True
    assert state.startup_historical_fill_replay_skipped is True
    assert state.startup_ready_without_fill_replay is True
    assert state.startup_rest_fill_replay_skipped_count >= 1

    path.unlink(missing_ok=True)


def _mk_unit_settings(*, fast_start: bool) -> tuple[UnitTestSettings, Path]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_faststart_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "PRIVATE_WS_ENABLED": True,
            "FAST_START_SKIP_HISTORICAL_FILL_REPLAY": fast_start,
        }
    )
    return settings, path


def test_fast_start_defers_private_ws_snapshot_fills_until_running() -> None:
    settings, path = _mk_unit_settings(fast_start=True)
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()

    # Seed a quote eligibility snapshot so deferred ingest (if any) can resolve.
    bb = _bba(settings.symbol, mid=100.0)
    state.apply_market_book_only(bb, market_data_source="test_seed")
    with state._lock:
        state.public_ws_last_message_wall_ts = utc_now()
        state.public_ws_connected = True
        state.public_ws_seen_first_bbo = True

    q: queue.Queue = queue.Queue()
    om = OrderManager(settings, client, storage, state, private_event_queue=q)

    snapshot_time_ms = int((state.session_started_at_utc - timedelta(days=1)).timestamp() * 1000.0)
    snap_ev = PrivateFillEvent(
        fill_id="snap1",
        oid=None,
        coin=settings.symbol,
        px=100.0,
        sz=0.01,
        side="B",
        time_ms=snapshot_time_ms,
        fee=0.0,
        closed_pnl=0.0,
        crossed=False,
        is_snapshot=True,
        raw={},
        inbound_timing=None,
    )

    q.put(snap_ev)

    with (
        patch("app.execution.private_fill_event_to_hl_raw") as p_to_hl,
        patch("app.execution.ingest_hl_fill_raw") as p_ingest,
    ):
        # During STARTING: isSnapshot fills are deferred/skipped (no ingest).
        om.drain_private_events(pnl=None)
        assert p_ingest.call_count == 0
        assert p_to_hl.call_count == 0
        assert state.startup_private_snapshot_fills_skipped_count == 1
        assert state.startup_historical_fill_replay_skipped is True

        # After RUNNING: deferred historical snapshot fills may be ingested for analytics.
        state.bot_status = BotStatus.RUNNING
        om.drain_private_events(pnl=None)

        assert p_ingest.call_count == 1
        assert p_to_hl.call_count == 1

        # Live private fills (isSnapshot=False) always work normally.
        p_to_hl.reset_mock()
        p_ingest.reset_mock()

        live_time_ms = int((state.session_started_at_utc + timedelta(seconds=1)).timestamp() * 1000.0)
        live_ev = PrivateFillEvent(
            fill_id="live1",
            oid=None,
            coin=settings.symbol,
            px=101.0,
            sz=0.02,
            side="B",
            time_ms=live_time_ms,
            fee=0.0,
            closed_pnl=0.0,
            crossed=False,
            is_snapshot=False,
            raw={},
            inbound_timing=None,
        )
        q.put(live_ev)
        om.drain_private_events(pnl=None)

        assert p_ingest.call_count == 1
        assert p_to_hl.call_count == 1
        assert state.startup_private_snapshot_fills_skipped_count == 1

    path.unlink(missing_ok=True)


def test_fast_start_flag_disabled_preserves_private_snapshot_replay() -> None:
    settings, path = _mk_unit_settings(fast_start=False)
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()

    bb = _bba(settings.symbol, mid=100.0)
    state.apply_market_book_only(bb, market_data_source="test_seed")
    with state._lock:
        state.public_ws_last_message_wall_ts = utc_now()
        state.public_ws_connected = True
        state.public_ws_seen_first_bbo = True

    q: queue.Queue = queue.Queue()
    om = OrderManager(settings, client, storage, state, private_event_queue=q)

    snapshot_time_ms = int((state.session_started_at_utc - timedelta(days=1)).timestamp() * 1000.0)
    snap_ev = PrivateFillEvent(
        fill_id="snap1",
        oid=None,
        coin=settings.symbol,
        px=100.0,
        sz=0.01,
        side="B",
        time_ms=snapshot_time_ms,
        fee=0.0,
        closed_pnl=0.0,
        crossed=False,
        is_snapshot=True,
        raw={},
        inbound_timing=None,
    )
    q.put(snap_ev)

    with (
        patch("app.execution.private_fill_event_to_hl_raw") as p_to_hl,
        patch("app.execution.ingest_hl_fill_raw") as p_ingest,
    ):
        om.drain_private_events(pnl=None)
        assert p_ingest.call_count == 1
        assert p_to_hl.call_count == 1
        assert state.startup_private_snapshot_fills_skipped_count == 0

    path.unlink(missing_ok=True)

