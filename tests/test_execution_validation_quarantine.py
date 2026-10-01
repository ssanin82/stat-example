"""Execution no longer mutates quotes with validation quarantine."""

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
    path = Path(tempfile.gettempdir()) / f"mm_vq_{os.getpid()}_{uuid.uuid4().hex}.db"
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


def test_place_passive_order_no_validation_quarantine_suppression() -> None:
    om, path = _setup()
    r1 = om.place_passive_order_manual_only(Side.BUY, 3000.0, 0.01, "q1")
    assert r1 is not None
    assert om._vk_strikes == {}
    assert om._vk_until == {}
    path.unlink(missing_ok=True)
