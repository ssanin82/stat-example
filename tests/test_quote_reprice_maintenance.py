"""Single-layer quote engine behavior in OrderManager maybe_refresh_quotes."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import ActiveSides, OrderStatus, RiskAction, Side
from app.execution import OrderManager
from app.models import BestBidAsk, QuoteDecision, WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_qrm_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 10.0,
            "QUOTE_AGING_ENABLED": True,
            "QUOTE_AGING_MAX_AGE_SECONDS": 3.0,
            "QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS": 3.0,
            "STALE_DATA_WARN_SECONDS": 30.0,
            "STALE_DATA_KILL_SECONDS": 120.0,
            "REPRICE_THRESHOLD_BPS": 20.0,
        }
    )
    return s, path


def _decision(mid: float = 3000.0, cycle: str = "qc1") -> QuoteDecision:
    return QuoteDecision(
        ts=utc_now(),
        symbol="ETH",
        mid_price=mid,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=mid,
        target_spread_bps=16.0,
        target_bid=mid - 1.0,
        target_ask=mid + 1.0,
        quoted_bid=mid - 1.0,
        quoted_ask=mid + 1.0,
        quoted_bid_sz=0.01,
        quoted_ask_sz=0.01,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="ok",
        quote_cycle_id=cycle,
    )


def _fresh_market(s: UnitTestSettings) -> BestBidAsk:
    return BestBidAsk(
        symbol=s.symbol,
        best_bid=3000.0,
        best_ask=3001.0,
        mid_price=3000.5,
        spread_bps=10.0,
        ts_local=utc_now(),
    )


def _ok_place(oid: int = 1) -> dict:
    return {
        "status": "ok",
        "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": oid}}]}},
    }


def test_single_layer_quote_engine_sets_final_prices_and_no_legacy_events() -> None:
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(oid=101)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    t = om._quote_exec_telemetry
    assert t["final_submitted_bid_px"] is not None
    assert t["final_submitted_ask_px"] is not None
    assert t["quote_engine_mode"] == "two_sided"
    ev = storage.recent_bot_events(50)
    names = {e["event_type"] for e in ev}
    assert "quote_reprice_skipped_distance_guard" not in names
    assert "quote_placement_symmetric_rescue_impossible" not in names
    path.unlink(missing_ok=True)


def test_reprice_orchestrator_cancels_when_desired_differs() -> None:
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.working_bid = WorkingOrder(
        order_id_local="L1",
        order_id_exchange=900001,
        client_order_id="0x" + "ab" * 16,
        symbol=s.symbol,
        side=Side.BUY,
        price=2990.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(oid=102)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()
    # v1.4.33+ routes all cancels through ``cancel_batch_orders``
    # (CANCEL_BATCH pool, 300/2 s) instead of CANCEL_SINGLE.
    client.cancel_batch_orders.assert_called()
    path.unlink(missing_ok=True)


def test_quote_engine_can_emit_one_sided_under_constraints() -> None:
    s, path = _db()
    s.min_quote_notional_usd = 1000.0
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(oid=103)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    om.maybe_refresh_quotes(_decision(), RiskAction.BID_ONLY, 1.0, 1.0, 0.0)
    t = om._quote_exec_telemetry
    assert t["quote_engine_mode"] in {"one_sided", "no_quote"}
    assert t["ask_executable"] is False
    path.unlink(missing_ok=True)
