"""No post-only cross mutation stack in execution path."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import ActiveSides, RiskAction
from app.execution import OrderManager
from app.models import BestBidAsk, QuoteDecision
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _decision() -> QuoteDecision:
    mid = 3000.0
    return QuoteDecision(
        ts=utc_now(),
        symbol="ETH",
        mid_price=mid,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=mid,
        target_spread_bps=20.0,
        target_bid=mid - 2.0,
        target_ask=mid + 2.0,
        quoted_bid=mid - 2.0,
        quoted_ask=mid + 2.0,
        quoted_bid_sz=0.02,
        quoted_ask_sz=0.02,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="ok",
        quote_cycle_id="c1",
    )


def _setup() -> tuple[OrderManager, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_pocc_{os.getpid()}_{uuid.uuid4().hex}.db"
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


def test_maybe_refresh_quotes_does_not_emit_legacy_cross_cooldown_events() -> None:
    om, path = _setup()
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    ev = om._storage.recent_bot_events(40)
    names = {e["event_type"] for e in ev}
    assert "post_only_cross_cooldown_skip_side" not in names
    assert "post_only_cross_rejection" not in names
    path.unlink(missing_ok=True)
