"""Exponential-backoff between recovery bursts + raised default max-duration.

Context
-------
The overnight 2026-04-19 ETH session was killed by a ~38 s GRVT public-feed
outage. The OLD behaviour: bursts ran back-to-back on every tick (~4 per
second), the 15 s ``MARKET_DATA_RECOVERY_MAX_DURATION_SECONDS`` budget
exhausted, and ``stale_data_kill_escalated`` fired — just 1 s before the
feed came back. That single-second-too-late kill cost us an hour of
downtime and an open position that had to be flattened manually.

The fix has two parts, tested here:

1. **Default MAX_DURATION raised from 15 s → 600 s (10 min).** A 38 s
   vendor blip no longer escalates; only a truly unrecoverable outage
   does. See ``test_default_max_duration_is_ten_minutes``.

2. **Exponential backoff between bursts.** Previously every tick ran a
   full burst. Now a failed burst schedules the next one ``backoff``
   seconds later, where ``backoff`` doubles on each subsequent failure
   (capped). This stops us hammering the venue during outages. See
   ``test_burst_backoff_ramps_and_caps``.

3. **Backoff resets on recovery success** so the next outage starts
   clean. See ``test_complete_recovery_resets_backoff``.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.bot import Bot
from app.config import Settings
from app.enums import BotStatus
from app.models import BestBidAsk
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from app.utils.time import utc_now


def _settings(**kw: object) -> UnitTestSettings:
    path = Path(tempfile.gettempdir()) / f"mm_mdrb_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "PUBLIC_WS_STALE_WARN_SECONDS": 2.0,
        "PUBLIC_WS_STALE_KILL_SECONDS": 5.0,
        "MARKET_DATA_RECOVERY_ENABLED": True,
        "MARKET_DATA_RECOVERY_MAX_ATTEMPTS": 1,
        "MARKET_DATA_RECOVERY_BACKOFF_SECONDS": 0.0,
        # Per-test defaults (fast so we can exercise the state machine without
        # real sleeping); the one "default knob" test calls the production
        # Settings() directly.
        "MARKET_DATA_RECOVERY_MAX_DURATION_SECONDS": 600.0,
        "MARKET_DATA_RECOVERY_BURST_BACKOFF_INITIAL_SECONDS": 1.0,
        "MARKET_DATA_RECOVERY_BURST_BACKOFF_MAX_SECONDS": 8.0,
        "MARKET_DATA_RECOVERY_BURST_BACKOFF_MULTIPLIER": 2.0,
    }
    for k, v in kw.items():
        data[k.upper() if k.islower() else k] = v
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


# -------- 1. Default raised: 10 min, not 15 s ----------------------------


def test_default_max_duration_is_ten_minutes() -> None:
    """The prod default must be 600 s — not the old 15 s. Locks in the
    operational calibration so a future silent ``Field(default=15)``
    revert fails here rather than overnight in prod."""
    # Construct production Settings (no UnitTest override) — uses plain
    # Pydantic defaults. We only care about the default, not credentials.
    import os as _os
    saved = _os.environ.pop("MARKET_DATA_RECOVERY_MAX_DURATION_SECONDS", None)
    try:
        # Also unset any ETH-profile-style inherited value.
        saved2 = _os.environ.pop("GRVT_API_SECRET", None)
        _os.environ["GRVT_API_SECRET"] = "x" * 64
        s = Settings()
    finally:
        if saved is not None:
            _os.environ["MARKET_DATA_RECOVERY_MAX_DURATION_SECONDS"] = saved
        if saved2 is None:
            _os.environ.pop("GRVT_API_SECRET", None)
        else:
            _os.environ["GRVT_API_SECRET"] = saved2
    assert s.market_data_recovery_max_duration_seconds == 600.0
    assert s.market_data_recovery_burst_backoff_initial_seconds == 3.0
    assert s.market_data_recovery_burst_backoff_max_seconds == 30.0
    assert s.market_data_recovery_burst_backoff_multiplier == 2.0


# -------- 2. A 38-s outage survives (the actual regression) --------------


def test_thirty_eight_second_outage_does_not_escalate_to_kill() -> None:
    """Pin the exact 2026-04-19 scenario. With default 600 s max-duration
    and exponential backoff, a ~38 s vendor outage must not kill."""
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
    state.apply_market_book_only(stale, market_data_source="public_ws")
    state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    pub = MagicMock()
    bot = Bot(s, state, client, storage, public_stream=pub)

    # Simulate 38 s of real time by monkey-patching time.monotonic to
    # return advancing values as the supervisor polls it. We also keep
    # the book stale on refresh so we stay in RECOVERING_MARKET_DATA.
    def refresh_still_stale(*args, **kwargs):
        state.apply_market_book_only(stale, market_data_source="public_ws")
        state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    fake_now = [0.0]
    def fake_mono():
        return fake_now[0]

    with patch("app.bot.refresh_account_only", side_effect=refresh_still_stale), \
         patch("app.bot.time.monotonic", side_effect=fake_mono):
        # Tick 0: transition to RECOVERING; initial burst.
        fake_now[0] = 0.0
        bot.one_tick()
        assert state.bot_status == BotStatus.RECOVERING_MARKET_DATA
        # Advance 38 s in small increments, ticking periodically — this
        # simulates the outage-window pattern. The backoff ramp means
        # most ticks are skipped (not enough time elapsed); only a few
        # bursts actually run. None of it should escalate to kill
        # because 38 s is well under the 600 s max.
        for dt in (0.5, 1.0, 2.0, 4.0, 8.0, 10.0, 12.0):
            fake_now[0] += dt
            bot.one_tick()
            if state.killed:
                break

    assert fake_now[0] >= 37.0, "fake clock advanced past 38 s"
    assert not state.killed, (
        f"38-s outage must not kill with default 600 s max-duration; "
        f"kill_reason={state.kill_reason}"
    )
    assert state.bot_status == BotStatus.RECOVERING_MARKET_DATA


# -------- 3. Exponential backoff ramps + caps ----------------------------


def test_burst_backoff_ramps_and_caps() -> None:
    """Each failed burst doubles the next-burst delay, capped at the
    configured max. We use ``initial=1, max=8, multiplier=2`` so the
    ramp is ``1 → 2 → 4 → 8 → 8 → 8`` across failed bursts.

    We count BURSTS via ``pub.request_reconnect.call_count`` (the first
    thing every burst does), because ``refresh_account_only`` is ALSO
    called on the normal market-data path every tick and would inflate
    the count.
    """
    s = _settings(
        MARKET_DATA_RECOVERY_BURST_BACKOFF_INITIAL_SECONDS=1.0,
        MARKET_DATA_RECOVERY_BURST_BACKOFF_MAX_SECONDS=8.0,
        MARKET_DATA_RECOVERY_BURST_BACKOFF_MULTIPLIER=2.0,
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

    def refresh_still_stale(*args, **kwargs):
        state.apply_market_book_only(stale, market_data_source="public_ws")
        state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    fake_now = [0.0]
    def fake_mono():
        return fake_now[0]

    backoffs_observed: list[float] = []
    bursts_observed: list[int] = []

    with patch("app.bot.refresh_account_only", side_effect=refresh_still_stale), \
         patch("app.bot.time.monotonic", side_effect=fake_mono):
        # Tick 0: transition + initial burst runs (1 reconnect). Fails →
        # backoff=1.0, next_burst=1.0.
        fake_now[0] = 0.0
        bot.one_tick()
        backoffs_observed.append(state.market_data_recovery_current_burst_backoff_s)
        bursts_observed.append(pub.request_reconnect.call_count)

        # Tick at 1.01: gate open, 2nd burst → fails → backoff doubles to 2.0.
        fake_now[0] = 1.01
        bot.one_tick()
        backoffs_observed.append(state.market_data_recovery_current_burst_backoff_s)
        bursts_observed.append(pub.request_reconnect.call_count)

        # Next burst scheduled at 1.01 + 2.0 = 3.01. Tick at 3.02 → fire.
        fake_now[0] = 3.02
        bot.one_tick()
        backoffs_observed.append(state.market_data_recovery_current_burst_backoff_s)
        bursts_observed.append(pub.request_reconnect.call_count)

        # Next = 3.02 + 4.0 = 7.02. Tick at 7.03 → fire.
        fake_now[0] = 7.03
        bot.one_tick()
        backoffs_observed.append(state.market_data_recovery_current_burst_backoff_s)
        bursts_observed.append(pub.request_reconnect.call_count)

        # Next = 7.03 + 8.0 = 15.03. Tick at 15.04 → fire.
        fake_now[0] = 15.04
        bot.one_tick()
        backoffs_observed.append(state.market_data_recovery_current_burst_backoff_s)
        bursts_observed.append(pub.request_reconnect.call_count)

    # Backoff ramp: 1 → 2 → 4 → 8 → 8 (capped at MAX_SECONDS=8).
    assert backoffs_observed == [1.0, 2.0, 4.0, 8.0, 8.0], (
        f"expected [1,2,4,8,8], got {backoffs_observed}"
    )
    # Cumulative burst count grows by 1 per tick (each tick fires a burst
    # because we advance the clock past each scheduled next-burst).
    assert bursts_observed == [1, 2, 3, 4, 5], (
        f"expected cumulative bursts [1,2,3,4,5], got {bursts_observed}"
    )


# -------- 4. Burst is SKIPPED while still waiting on backoff -------------


def test_ticks_inside_backoff_window_are_skipped() -> None:
    """Between bursts, if ``time.monotonic() < next_burst_mono``, ticks
    MUST NOT run another burst. Pin this so the "every tick = burst"
    behaviour can't silently return.

    We count BURSTS via ``pub.request_reconnect.call_count`` because
    ``refresh_account_only`` is also called on the normal market-data
    path every tick regardless of recovery state.
    """
    s = _settings(
        MARKET_DATA_RECOVERY_BURST_BACKOFF_INITIAL_SECONDS=5.0,
        MARKET_DATA_RECOVERY_BURST_BACKOFF_MAX_SECONDS=30.0,
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

    fake_now = [0.0]
    def fake_mono():
        return fake_now[0]

    with patch("app.bot.refresh_account_only", side_effect=refresh_always_stale), \
         patch("app.bot.time.monotonic", side_effect=fake_mono):
        # Initial tick: transition + initial burst → 1 reconnect.
        fake_now[0] = 0.0
        bot.one_tick()
        bursts_after_first = pub.request_reconnect.call_count
        assert bursts_after_first >= 1
        # Backoff = 5 s. Ticks at t=1, 2, 3, 4 must NOT burst.
        for t in (1.0, 2.0, 3.0, 4.0):
            fake_now[0] = t
            bot.one_tick()
        assert pub.request_reconnect.call_count == bursts_after_first, (
            f"No new burst should have run while waiting on the 5-s backoff; "
            f"burst count went {bursts_after_first} -> {pub.request_reconnect.call_count}"
        )
        # At t=5.01, the gate opens — the next tick bursts again.
        fake_now[0] = 5.01
        bot.one_tick()
        assert pub.request_reconnect.call_count > bursts_after_first


# -------- 5. Recovery success resets backoff state -----------------------


def test_complete_recovery_resets_backoff_state() -> None:
    """After a successful recovery, the next outage must start with a
    fresh backoff (initial=1 s here) — not resume from wherever the
    previous outage left off. Otherwise a flappy hour would ramp backoff
    to the max and never come down.

    We flip a ``book_mode`` flag from 'stale' to 'fresh' between ticks so
    the multiple refresh calls per tick (normal path + burst path) all
    see the same mode. Simpler and more deterministic than a pop-based
    sequence.
    """
    s = _settings(
        MARKET_DATA_RECOVERY_BURST_BACKOFF_INITIAL_SECONDS=1.0,
        MARKET_DATA_RECOVERY_BURST_BACKOFF_MAX_SECONDS=8.0,
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
    fresh = _fresh_book()

    state.apply_market_book_only(stale, market_data_source="public_ws")
    state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)

    pub = MagicMock()
    bot = Bot(s, state, client, storage, public_stream=pub)

    book_mode = ["stale"]  # closure-mutable
    def refresh_by_mode(*args, **kwargs):
        if book_mode[0] == "stale":
            state.apply_market_book_only(stale, market_data_source="public_ws")
            state.public_ws_last_message_wall_ts = utc_now() - timedelta(seconds=30)
        else:
            state.apply_market_book_only(fresh, market_data_source="public_ws")
            state.public_ws_last_message_wall_ts = utc_now()

    fake_now = [0.0]
    def fake_mono():
        return fake_now[0]

    with patch("app.bot.refresh_account_only", side_effect=refresh_by_mode), \
         patch("app.bot.time.monotonic", side_effect=fake_mono):
        # Tick 1 (mode=stale): transition + initial burst fails → backoff 1.0.
        fake_now[0] = 0.0
        bot.one_tick()
        assert state.bot_status == BotStatus.RECOVERING_MARKET_DATA
        assert state.market_data_recovery_current_burst_backoff_s == 1.0

        # Tick 2 (mode=stale): burst fails again → backoff doubles to 2.0.
        fake_now[0] = 1.01
        bot.one_tick()
        assert state.bot_status == BotStatus.RECOVERING_MARKET_DATA
        assert state.market_data_recovery_current_burst_backoff_s == 2.0

        # Flip to fresh. Tick 3 (mode=fresh, past gate at t=3.01): burst
        # succeeds → _complete_market_recovery resets state.
        book_mode[0] = "fresh"
        fake_now[0] = 3.03
        bot.one_tick()

    assert state.bot_status == BotStatus.RUNNING
    assert state.market_data_recovery_current_burst_backoff_s == 0.0, (
        "backoff must reset to 0 on recovery success"
    )
    assert state.market_data_recovery_next_burst_mono is None
