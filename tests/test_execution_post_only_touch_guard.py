"""place_passive_order_manual_only: grid-normalized manual path (not runtime quote refresh)."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import Side
from app.execution import OrderManager
from app.models import BestBidAsk
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[OrderManager, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_touch_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=3000.0,
        best_ask=3001.0,
        mid_price=3000.5,
        spread_bps=10.0,
        ts_local=utc_now(),
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = {
        "status": "ok",
        "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 1}}]}},
    }
    return OrderManager(s, client, storage, state, private_event_queue=None), path


def test_place_passive_order_submits_verbatim_buy_price() -> None:
    om, path = _setup()
    om.place_passive_order_manual_only(Side.BUY, 3000.123, 0.011, "q1")
    args = om._client.place_post_only_limit.call_args[0]
    # FALLBACK_SYMBOL_SPEC price_tick=0.01
    assert args[3] == 3000.1
    assert args[2] == 0.011
    path.unlink(missing_ok=True)


def test_place_passive_order_submits_verbatim_sell_price() -> None:
    om, path = _setup()
    om.place_passive_order_manual_only(Side.SELL, 3001.987, 0.013, "q1")
    args = om._client.place_post_only_limit.call_args[0]
    assert args[3] == 3002.0
    assert args[2] == 0.013
    path.unlink(missing_ok=True)
