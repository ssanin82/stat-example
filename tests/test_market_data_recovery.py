"""Public WS stale kill → bounded reconnect + fresh BBO → RUNNING or escalated kill."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.bot import Bot
from app.enums import BotStatus, RiskAction
from app.models import BestBidAsk, RiskDecision
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from app.utils.time import utc_now


def _settings(**kw: object) -> UnitTestSettings:
    path = Path(tempfile.gettempdir()) / f"mm_mdr_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "PUBLIC_WS_STALE_WARN_SECONDS": 2.0,
        "PUBLIC_WS_STALE_KILL_SECONDS": 5.0,
        "MARKET_DATA_RECOVERY_ENABLED": True,
        "MARKET_DATA_RECOVERY_MAX_ATTEMPTS": 2,
        "MARKET_DATA_RECOVERY_BACKOFF_SECONDS": 0.0,
        "MARKET_DATA_RECOVERY_MAX_DURATION_SECONDS": 30.0,
    }
    for k, v in kw.items():
        key = k.upper() if k.islower() else k
        data[key] = v
    return UnitTestSettings.model_validate(data)


def _stale_book() -> BestBidAsk:
    old = utc_now() - timedelta(seconds=30)
    return BestBidAsk(
        symbol="ETH",
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        spread_bps=200.0,
        ts_local=old,
    )


def _fresh_book() -> BestBidAsk:
    return BestBidAsk(
        symbol="ETH",
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        spread_bps=200.0,
        ts_local=utc_now(),
    )


def test_recovery_success_returns_running_and_stops_quoting_while_recovering() -> None:
    s = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.bot_status = BotStatus.RUNNING
    state.public_ws_connected = True
    state.public_ws_seen_first_bbo = True
    client = mock_mm_client()
    client.has_write_access.return_value = True

    stale = _stale_book()
    fresh = _fresh_book()
    state.apply_market_book_only(stale, market_data_source="public_ws")
    state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    pub = MagicMock()
    bot = Bot(s, state, client, storage, public_stream=pub)
    book_q = [stale, fresh, fresh]

    def refresh_side_effect(*args, **kwargs):
        b = book_q.pop(0) if len(book_q) > 1 else book_q[-1]
        state.apply_market_book_only(b, market_data_source="public_ws")
        if b is stale:
            state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)
        else:
            state.public_ws_last_message_wall_ts = utc_now()

    with patch("app.bot.refresh_account_only", side_effect=refresh_side_effect):
        bot.one_tick()

    assert state.bot_status == BotStatus.RUNNING
    assert not state.killed
    assert pub.request_reconnect.call_count >= 2


def test_recovery_exits_when_market_already_fresh_without_running_burst() -> None:
    """Regression for the 2026-05-09 stuck-in-RECOVERING bug.

    Sequence reproduced from snapshot_prod.okx.hype.usdt.perp_260509054855:
      1. Bot enters ``RECOVERING_MARKET_DATA`` after a transient stale event.
      2. Bot's regular maintenance path brings the public WS book back to
         fresh BEFORE the next backoff-gated burst window opens
         (``market_data_refresh_success`` events fire).
      3. Recovery branch is re-entered on a subsequent tick.
      4. Expected: bot transitions back to RUNNING immediately,
         WITHOUT running another ``_market_recovery_burst`` (which
         would call ``request_reconnect`` and tear the now-healthy
         WS down again).

    Pre-fix the bot stayed in RECOVERING for 6+ minutes because the
    only recovery-exit path was the post-burst freshness check, and
    the burst itself disturbed the WS enough that the check failed.
    """
    s = _settings(
        # A non-zero backoff so the test meaningfully exercises the
        # "exit-while-gate-open" behaviour. If the gate is at 0, the
        # post-burst path would be reached on the same tick anyway.
        MARKET_DATA_RECOVERY_BURST_BACKOFF_INITIAL_SECONDS=10.0,
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.bot_status = BotStatus.RECOVERING_MARKET_DATA
    state.public_ws_connected = True
    state.public_ws_seen_first_bbo = True
    # Simulate "we're inside the recovery branch, the burst has fired
    # at least once already, the backoff gate is closed for the next
    # 10 s" — and the regular MD path has brought the book back to
    # fresh in the meantime.
    state.market_data_recovery_started_monotonic = time.monotonic() - 1.0
    state.market_data_recovery_next_burst_mono = time.monotonic() + 10.0
    state.market_data_recovery_current_burst_backoff_s = 10.0
    state.market_data_recovery_refresh_attempts_episode = 3
    fresh = _fresh_book()
    state.apply_market_book_only(fresh, market_data_source="public_ws")
    state.public_ws_last_message_wall_ts = utc_now()

    client = mock_mm_client()
    client.has_write_access.return_value = True
    pub = MagicMock()
    bot = Bot(s, state, client, storage, public_stream=pub)

    with patch("app.bot.refresh_account_only"):
        bot.one_tick()

    assert state.bot_status == BotStatus.RUNNING, (
        "bot should exit recovery when market is already fresh"
    )
    assert not state.killed
    # The critical assertion: the recovery BURST was not run.
    # ``_market_recovery_burst`` is the path that calls
    # ``request_reconnect`` on the public WS, and that's the
    # disruptive behaviour we're avoiding. Other code paths in
    # ``one_tick`` may call ``refresh_account_only`` for unrelated
    # reasons (regular maintenance, etc.) which is why we don't
    # assert on the refresh mock here.
    assert pub.request_reconnect.call_count == 0, (
        "non-disruptive exit must not call request_reconnect"
    )
    # Recovery state was reset cleanly.
    assert state.market_data_recovery_started_monotonic is None
    assert state.market_data_recovery_refresh_attempts_episode == 0


def test_recovery_does_not_exit_early_when_market_still_stale() -> None:
    """Negative case: the non-disruptive early-exit must NOT fire
    when the market data is still stale. The burst path should run
    instead (modulo the backoff gate)."""
    s = _settings(
        MARKET_DATA_RECOVERY_BURST_BACKOFF_INITIAL_SECONDS=0.0,
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.bot_status = BotStatus.RECOVERING_MARKET_DATA
    state.public_ws_connected = True
    state.public_ws_seen_first_bbo = True
    state.market_data_recovery_started_monotonic = time.monotonic() - 1.0
    # Stale book + stale ws timestamp — early-exit must not trigger.
    stale = _stale_book()
    state.apply_market_book_only(stale, market_data_source="public_ws")
    state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    client = mock_mm_client()
    client.has_write_access.return_value = True
    pub = MagicMock()
    bot = Bot(s, state, client, storage, public_stream=pub)

    def refresh_keeps_stale(*args, **kwargs):
        state.apply_market_book_only(stale, market_data_source="public_ws")
        state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    with patch("app.bot.refresh_account_only", side_effect=refresh_keeps_stale):
        bot.one_tick()

    # Still recovering (no early-exit triggered) and the burst DID run.
    assert state.bot_status == BotStatus.RECOVERING_MARKET_DATA
    assert pub.request_reconnect.call_count >= 1, (
        "burst must still run when data is genuinely stale"
    )


def test_recovery_duration_exceeded_escalates_kill() -> None:
    s = _settings(
        MARKET_DATA_RECOVERY_MAX_DURATION_SECONDS=0.01,
        MARKET_DATA_RECOVERY_MAX_ATTEMPTS=1,
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.bot_status = BotStatus.RUNNING
    state.public_ws_connected = True
    state.public_ws_seen_first_bbo = True
    client = mock_mm_client()
    client.has_write_access.return_value = True

    stale = _stale_book()
    state.apply_market_book_only(stale, market_data_source="public_ws")
    state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    pub = MagicMock()
    bot = Bot(s, state, client, storage, public_stream=pub)

    def refresh_always_stale(*args, **kwargs):
        state.apply_market_book_only(stale, market_data_source="public_ws")
        state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    with patch("app.bot.refresh_account_only", side_effect=refresh_always_stale):
        bot.one_tick()
        assert state.bot_status == BotStatus.RECOVERING_MARKET_DATA

        time.sleep(0.05)
        bot.one_tick()

    assert state.killed
    assert state.bot_status == BotStatus.KILLED
    assert "stale_data_kill_escalated" in (state.kill_reason or "")


def test_stale_market_data_warn_log_includes_actionable_context(caplog) -> None:
    s = _settings(MARKET_DATA_RECOVERY_ENABLED=False, PUBLIC_WS_ENABLED=False)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.bot_status = BotStatus.RUNNING
    state.market = _stale_book()
    state.public_ws_connected = False
    state.market_data_failed_refresh_streak = 3
    state.market_data_unchanged_snapshot_streak = 9
    state.market_data_last_success_wall_ts = utc_now() - timedelta(seconds=22)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(s, state, client, storage)
    fake_risk = RiskDecision(action=RiskAction.NO_QUOTE, reasons=["stale_data_warn"])
    with caplog.at_level(logging.WARNING, logger="app.bot"), patch(
        "app.bot.refresh_account_only"
    ), patch("app.bot.evaluate_risk", return_value=fake_risk):
        bot.one_tick()
    recs = [r for r in caplog.records if r.getMessage() == "stale market data — quoting disabled"]
    assert recs
    extra = getattr(recs[0], "extra_data", {})
    assert "seconds_since_last_market_data_refresh" in extra
    assert "public_ws_connected" in extra
    assert "market_data_failed_refresh_streak" in extra


def test_stale_market_data_warn_log_throttled_once_per_episode(caplog) -> None:
    """v1.5.301 throttle: a sustained stale-book episode re-flags
    ``stale_data_warn`` on EVERY tick, but the WARNING must be emitted
    only ONCE on episode entry — not once per 500 ms tick. When the book
    goes fresh and a distinct stale episode later begins, a fresh entry
    line is logged.

    Regression guard for the per-tick log spam observed replaying the
    genuine 2–5 s OKX bbo-tbt gaps. Quoting is unaffected: the action is
    NO_QUOTE every tick regardless of whether we log."""
    s = _settings(MARKET_DATA_RECOVERY_ENABLED=False, PUBLIC_WS_ENABLED=False)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.bot_status = BotStatus.RUNNING
    state.market = _stale_book()
    state.public_ws_connected = False
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(s, state, client, storage)

    stale = RiskDecision(action=RiskAction.NO_QUOTE, reasons=["stale_data_warn"])
    fresh = RiskDecision(action=RiskAction.NO_QUOTE, reasons=[])

    def _count() -> int:
        return len(
            [
                r
                for r in caplog.records
                if r.getMessage() == "stale market data — quoting disabled"
            ]
        )

    with caplog.at_level(logging.WARNING, logger="app.bot"), patch(
        "app.bot.refresh_account_only"
    ):
        # Episode 1: three consecutive stale ticks → exactly ONE log.
        with patch("app.bot.evaluate_risk", return_value=stale):
            bot.one_tick()
            bot.one_tick()
            bot.one_tick()
        assert _count() == 1, "sustained stale episode must log once, not per tick"

        # Book fresh again → throttle re-arms (and emits no stale line).
        with patch("app.bot.evaluate_risk", return_value=fresh):
            bot.one_tick()
        assert _count() == 1

        # Episode 2: stale again → a distinct entry line is logged.
        with patch("app.bot.evaluate_risk", return_value=stale):
            bot.one_tick()
            bot.one_tick()
        assert _count() == 2, "a new stale episode must log its own entry line"
