"""Integration: inventory bias is expressed via QuoteEngine output, not execution reordering."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import RiskAction
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_quote_reprice_maintenance import _decision, _fresh_market, _ok_place


def _settings(**extra: object) -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_invbias_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 10.0,
        # ETH-priced mock symbol; raise USD cap so the new
        # max_position_notional_usd clip (added 2026-05-06) doesn't
        # zero the test's 2-unit positions ($6000 at $3000/unit).
        "MAX_POSITION_NOTIONAL_USD": 100_000.0,
        "QUOTE_AGING_ENABLED": True,
        "QUOTE_AGING_MAX_AGE_SECONDS": 300.0,
        "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 50.0,
        "STALE_DATA_WARN_SECONDS": 30.0,
        "STALE_DATA_KILL_SECONDS": 120.0,
        "REPRICE_THRESHOLD_BPS": 0.5,
        "INVENTORY_EXEC_BIAS_RATIO": 0.01,
        "INVENTORY_EXEC_BIAS_MIN_UTIL_PCT": 0.0,
        "INVENTORY_EXEC_BIAS_NONPREFERRED_REPRICE_MULT": 3.0,
    }
    base.update(extra)
    return UnitTestSettings.model_validate(base), path


def test_neutral_inventory_symmetric_both_sides_place() -> None:
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = 0.0
    state.position.position_notional = 0.0
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()
    assert client.place_post_only_limit.call_count == 2
    path.unlink(missing_ok=True)


def test_positive_inventory_long_places_sell_only_when_reducer_absent() -> None:
    """Engine drops bid (adding); execution may only submit resting SELL intent."""
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = 1.0
    state.position.position_notional = 3000.0
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()
    assert client.place_post_only_limit.call_count == 1
    side_is_buy = client.place_post_only_limit.call_args[0][1]
    assert side_is_buy is False
    path.unlink(missing_ok=True)


def test_negative_inventory_short_places_buy_only_when_reducer_absent() -> None:
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = -1.5
    state.position.position_notional = 4500.0
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()
    assert client.place_post_only_limit.call_count == 1
    assert client.place_post_only_limit.call_args[0][1] is True
    path.unlink(missing_ok=True)


def test_execution_bias_suppressed_when_util_below_min_floor() -> None:
    s, path = _settings(
        INVENTORY_EXEC_BIAS_MIN_UTIL_PCT=0.12,
        INVENTORY_EXEC_BIAS_RATIO=0.01,
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = 0.5
    state.position.position_notional = 1500.0
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()
    assert client.place_post_only_limit.call_count == 2
    path.unlink(missing_ok=True)


def test_bias_ratio_zero_disables_priority() -> None:
    s, path = _settings(INVENTORY_EXEC_BIAS_RATIO=0.0)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = 5.0
    state.position.position_notional = 15000.0
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()
    assert client.place_post_only_limit.call_count == 2
    path.unlink(missing_ok=True)


def test_preferred_unavailable_stale_book_skips_deprioritized_near_touch_without_crash() -> None:
    s, path = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    from app.models import BestBidAsk

    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=3000.0,
        best_ask=3001.0,
        mid_price=3000.5,
        spread_bps=10.0,
        ts_local=None,
    )
    state.position.position_qty = 3.0
    state.position.position_notional = 9000.0
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    # v1.5.23 -- pre-fix this test was the only one in the file that
    # skipped ``wait_transport_idle()`` + ``storage.close()`` before
    # unlink. The stale-book scenario triggers a place-path response
    # check that logs ``place_response_unconfirmed_critical`` (the
    # MagicMock returns a non-dict) and writes the unconfirmed-order
    # row to SQLite. On Windows, ``unlink`` then fails with
    # WinError 32 ("file in use") because Storage's SQLite handle is
    # still open. Linux's unlink-while-open semantics hid this until
    # the CI daemon ran the suite on Windows.
    om.wait_transport_idle()
    storage.close()
    path.unlink(missing_ok=True)
