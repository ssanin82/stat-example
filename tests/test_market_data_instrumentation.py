"""Account REST refresh metrics; public BBO stall → optional public WS reconnect."""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

from app.bot import Bot
from app.enums import BotStatus
from app.market_data import instrument_book_update_success, refresh_account_only
from app.models import BestBidAsk
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from app.utils.time import utc_now


def _bb(mid: float = 100.0, ts_ms: int | None = 1) -> BestBidAsk:
    return BestBidAsk(
        symbol="ETH",
        best_bid=mid - 1,
        best_ask=mid + 1,
        mid_price=mid,
        spread_bps=20.0,
        ts_exchange_ms=ts_ms,
        ts_local=datetime.now(timezone.utc),
    )


def test_repeated_account_refresh_failures_visible_in_state_and_events() -> None:
    path = Path(tempfile.gettempdir()) / f"mm_mdi_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = MagicMock()
    client.fetch_position.side_effect = RuntimeError("down")
    refresh_account_only(client, state, "0xabc", storage, None)
    refresh_account_only(client, state, "0xabc", storage, None)
    assert state.market_data_failed_refresh_streak == 2
    assert state.market_data_last_refresh_latency_ms is not None
    rows = storage.recent_bot_events(20)
    fails = [r for r in rows if r.get("event_type") == "market_data_refresh_failure"]
    assert len(fails) >= 2
    path.unlink(missing_ok=True)


def test_account_success_after_failure_resets_streak_and_records_latency() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "0xabc",
        }
    )
    state = BotState(s)
    client = MagicMock()
    client.fetch_position.side_effect = [RuntimeError("x"), MagicMock()]
    client.fetch_account_snapshot.return_value = MagicMock()
    client.fetch_recent_fills_raw.return_value = []
    refresh_account_only(client, state, "0xabc", None, None)
    assert state.market_data_failed_refresh_streak == 1
    refresh_account_only(client, state, "0xabc", None, None)
    assert state.market_data_failed_refresh_streak == 0


def test_unchanged_bbo_stall_triggers_event_and_optional_public_ws_reconnect() -> None:
    path = Path(tempfile.gettempdir()) / f"mm_mdi2_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MARKET_DATA_STALL_UNCHANGED_THRESHOLD": 3,
            "MARKET_DATA_SUCCESS_LOG_INTERVAL_SECONDS": 0,
            "MARKET_DATA_RESET_TRANSPORT_ON_STALL": True,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    same = _bb(100.0, ts_ms=42)
    on_rc = MagicMock()
    for _ in range(4):
        state.apply_market_book_only(same, market_data_source="public_ws", storage=storage)
        out = state.market_refresh_note_success(same, 0.0)
        instrument_book_update_success(
            state,
            storage,
            sym="ETH",
            bb=same,
            latency_ms=0.0,
            out=out,
            on_stall_reconnect=on_rc,
        )
    assert state.market_data_unchanged_snapshot_streak >= 3
    assert state.market_data_stall_latched is True
    on_rc.assert_called_once()
    rows = storage.recent_bot_events(30)
    assert any(r.get("event_type") == "market_data_snapshot_stalled" for r in rows)
    assert any(r.get("event_type") == "market_data_task_restarted" for r in rows)
    path.unlink(missing_ok=True)


def test_recovery_burst_increments_episode_counter() -> None:
    path = Path(tempfile.gettempdir()) / f"mm_mdi3_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "PUBLIC_WS_STALE_WARN_SECONDS": 2.0,
            "PUBLIC_WS_STALE_KILL_SECONDS": 5.0,
            "MARKET_DATA_RECOVERY_MAX_ATTEMPTS": 2,
            "MARKET_DATA_RECOVERY_BACKOFF_SECONDS": 0.0,
            "MARKET_DATA_RECOVERY_MAX_DURATION_SECONDS": 60.0,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.bot_status = BotStatus.RUNNING
    state.public_ws_connected = True
    state.public_ws_seen_first_bbo = True
    client = mock_mm_client()
    client.has_write_access.return_value = True
    stale_ts = utc_now() - timedelta(seconds=30)
    stale_b = BestBidAsk(
        symbol="ETH",
        best_bid=2999.0,
        best_ask=3001.0,
        mid_price=3000.0,
        spread_bps=10.0,
        ts_exchange_ms=1,
        ts_local=stale_ts,
    )
    state.apply_market_book_only(stale_b, market_data_source="public_ws")
    state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    pub = MagicMock()
    bot = Bot(s, state, client, storage, public_stream=pub)

    fresh = _bb(3000.0, ts_ms=99)
    book_seq = [stale_b, fresh, fresh]

    def refresh_side_effect(*args, **kwargs):
        b = book_seq.pop(0) if len(book_seq) > 1 else book_seq[-1]
        state.apply_market_book_only(b, market_data_source="public_ws")
        if b is stale_b:
            state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)
        else:
            state.public_ws_last_message_wall_ts = utc_now()

    from unittest.mock import patch

    with (
        patch("app.bot.refresh_account_only", side_effect=refresh_side_effect),
        patch.object(bot._exec, "maybe_refresh_quotes"),
        patch.object(bot, "_exchange_snapshot_healthy", return_value=True),
    ):
        bot.one_tick()
    assert state.bot_status == BotStatus.RUNNING
    assert not state.killed
    assert pub.request_reconnect.call_count >= 2
    path.unlink(missing_ok=True)
