from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from app.enums import ActiveSides, RiskAction
from app.execution import OrderManager
from app.models import BestBidAsk, QuoteDecision
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[UnitTestSettings, Path, Storage, BotState, OrderManager]:
    path = Path(tempfile.gettempdir()) / f"mm_hp_{os.getpid()}_{uuid.uuid4().hex}.db"
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
            "REPRICE_THRESHOLD_BPS": 50.0,
            "EXECUTION_LATENCY_WARN_MS": 10.0,
            "EXECUTION_LATENCY_DEGRADE_MS": 20.0,
            "EXECUTION_LATENCY_WINDOW_SAMPLES": 5,
            "EXECUTION_LATENCY_DEGRADE_MIN_BREACHES": 2,
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
    c = mock_mm_client()
    c.has_write_access.return_value = True
    c.place_post_only_limit.return_value = {
        "status": "ok",
        "response": {"type": "order", "data": {"statuses": [{"resting": {"oid": 1}}]}},
    }
    om = OrderManager(s, c, storage, state, private_event_queue=None)
    return s, path, storage, state, om


def _decision(mid: float = 3000.0) -> QuoteDecision:
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
        quote_cycle_id="perf1",
    )


def test_compute_executable_quote_size_budget() -> None:
    _s, path, _stg, _state, om = _setup()
    t0 = time.perf_counter()
    for _ in range(600):
        _sz, ok, _ntn, _blk = om._compute_executable_quote_size(
            price=3000.0, desired_size=0.01, min_quote_notional_usd=10.0
        )
        assert ok is True
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert elapsed_ms < 180.0
    path.unlink(missing_ok=True)


def test_maybe_refresh_quotes_hot_path_budget_and_breakdown_fields() -> None:
    _s, path, _stg, _state, om = _setup()
    t0 = time.perf_counter()
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    t = om._quote_exec_telemetry
    # Loose guard against pathological slowdowns; Windows/CI cold runs often land ~1.2–1.5s.
    _budget_ms = 2000.0
    assert elapsed_ms < _budget_ms
    assert float(t["quote_contract_build_ms"]) < _budget_ms
    assert isinstance(t["quote_contract_build_ms"], float)
    assert isinstance(t["executable_size_check_ms"], float)
    assert isinstance(t["placement_mode_eval_ms"], float)
    assert isinstance(t["finalize_ms"], float)
    assert isinstance(t["order_submit_prep_ms"], float)
    for k in (
        "decision_to_submit_dispatch_ms",
        "submit_queue_wait_ms",
        "submit_transport_rtt_ms",
        "decision_to_first_submit_dispatch_ms",
        "ack_resolution_ms",
    ):
        assert k in t
    path.unlink(missing_ok=True)


def test_latency_guardrail_events_emit_on_thresholds() -> None:
    _s, path, stg, state, om = _setup()
    insert = MagicMock(wraps=stg.insert_bot_event)
    stg.insert_bot_event = insert
    with state._lock:
        state.last_latency_decision_to_first_place_ms = 650.0
    om._maybe_emit_latency_guardrails(quote_cycle_id="perf-guard-1")
    om._maybe_emit_latency_guardrails(quote_cycle_id="perf-guard-2")
    names = [c.args[2] for c in insert.call_args_list]
    assert "execution_latency_warning" in names
    assert "execution_latency_sustained_high" in names
    path.unlink(missing_ok=True)
