"""Tests for v1.4.82 Phase 3B — PositionStore, MarketStore,
TelemetryStore.

All three are additive facades over ``BotState`` state. Phase 3D
will move ownership; for now these tests pin the API surface so
future refactors don't break callers.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.models import BestBidAsk, PositionSnapshot
from app.state import BotState
from app.stores import MarketStore, OrderStore, PositionStore, TelemetryStore
from tests.settings_helpers import UnitTestSettings


def _settings():
    return UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_phase3b_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
    })


def _state() -> BotState:
    return BotState(_settings())


# ---------------------------------------------------------------------------
# PositionStore
# ---------------------------------------------------------------------------


def test_phase3b_position_store_flat_initial() -> None:
    state = _state()
    ps = state.position_store
    assert ps.qty() == 0.0
    assert ps.is_flat()
    assert ps.reducing_side() is None
    assert ps.notional_abs_usd() == 0.0


def test_phase3b_position_store_long_reducer_is_sell() -> None:
    state = _state()
    state.position = PositionSnapshot(
        symbol="ETH",
        position_qty=5.0,
        avg_entry_price=2.0,
        mark_price=2.0,
        position_notional=10.0,
        unrealized_pnl_usd=0.0,
    )
    ps = state.position_store
    assert ps.qty() == 5.0
    assert ps.reducing_side() == "SELL"
    assert ps.is_flat() is False
    assert ps.notional_abs_usd() == 10.0


def test_phase3b_position_store_short_reducer_is_buy() -> None:
    state = _state()
    state.position = PositionSnapshot(
        symbol="ETH",
        position_qty=-3.0,
        avg_entry_price=2.0,
        mark_price=2.0,
        position_notional=6.0,
        unrealized_pnl_usd=0.0,
    )
    ps = state.position_store
    assert ps.qty() == -3.0
    assert ps.reducing_side() == "BUY"


def test_phase3b_position_store_headroom() -> None:
    state = _state()
    state.position = PositionSnapshot(
        symbol="ETH", position_qty=3.0, avg_entry_price=None,
        mark_price=None, position_notional=0.0, unrealized_pnl_usd=0.0,
    )
    ps = state.position_store
    # MAX_ABS_POSITION=10: long 3 → 7 long headroom + 13 short headroom.
    assert ps.headroom_long(10.0) == 7.0
    assert ps.headroom_short(10.0) == 13.0


# ---------------------------------------------------------------------------
# MarketStore
# ---------------------------------------------------------------------------


def test_phase3b_market_store_empty_initial() -> None:
    state = _state()
    ms = state.market_store
    assert ms.bbo is None
    assert ms.best_bid() is None
    assert ms.best_ask() is None
    assert ms.mid_price() is None
    assert ms.spread_abs() is None
    assert ms.spread_bps() is None


def test_phase3b_market_store_with_bbo() -> None:
    from datetime import datetime, timezone
    state = _state()
    state.market = BestBidAsk(
        symbol="ETH",
        best_bid=2.000,
        best_ask=2.002,
        mid_price=2.001,
        spread_bps=10.0,
        ts_local=datetime.now(timezone.utc),
    )
    ms = state.market_store
    assert ms.best_bid() == 2.000
    assert ms.best_ask() == 2.002
    assert ms.mid_price() == 2.001
    assert abs(ms.spread_abs() - 0.002) < 1e-9
    # spread_bps = (ask-bid)/bid * 10000 = 0.002/2.000 * 10000 = 10
    assert abs(ms.spread_bps() - 10.0) < 1e-6


# ---------------------------------------------------------------------------
# TelemetryStore
# ---------------------------------------------------------------------------


def test_phase3b_telemetry_store_default_zero_when_no_snapshot() -> None:
    state = _state()
    ts = state.telemetry_store
    # No executor_state_snapshot populated yet.
    assert ts.ws_unmatched_total() == 0
    assert ts.hydration_merged_total() == 0
    assert ts.reaper_total() == 0
    assert ts.phase2a_invariant_violations() == 0
    # v1.4.92 Phase 4A cutover: ``qbr_unconsumed_total`` removed along
    # with the Phase 2D runtime audit. Typed BuildCommand replaces it.
    # Risk state defaults to UNKNOWN when snapshot is empty.
    assert ts.risk_exec_state() == "UNKNOWN"


def test_phase3b_telemetry_store_reads_executor_state() -> None:
    state = _state()
    state.executor_state_snapshot = {
        "ws_event_unmatched_to_local_wo_total": 3,
        "hydration_merged_existing_total": 12,
        "reaper_cancel_pending_reaped_total": 1,
        "reaper_desync_removed_total": 2,
        "reaper_sent_rejected_total": 0,
        "gate_phase2a_invariant_violation_total": 0,
        "risk_exec_state": "NORMAL",
        "side_unresolved_enter_count": 50,
        "execution_idle_s": 0.123,
    }
    ts = state.telemetry_store
    assert ts.ws_unmatched_total() == 3
    assert ts.hydration_merged_total() == 12
    assert ts.reaper_total() == 3  # 1 + 2 + 0
    assert ts.risk_exec_state() == "NORMAL"
    assert ts.side_unresolved_enter_count() == 50
    assert ts.execution_idle_s() == 0.123


def test_phase3b_telemetry_store_is_healthy_aggregate() -> None:
    state = _state()
    state.executor_state_snapshot = {
        "ws_event_unmatched_to_local_wo_total": 0,
        "hydration_merged_existing_total": 0,
        "reaper_cancel_pending_reaped_total": 0,
        "reaper_desync_removed_total": 0,
        "reaper_sent_rejected_total": 0,
        "gate_phase2a_invariant_violation_total": 0,
        "risk_exec_state": "NORMAL",
    }
    assert state.telemetry_store.is_healthy() is True

    # Flip an invariant violation → unhealthy.
    state.executor_state_snapshot["gate_phase2a_invariant_violation_total"] = 1
    assert state.telemetry_store.is_healthy() is False


def test_phase3b_telemetry_store_is_unhealthy_when_suppressed() -> None:
    state = _state()
    state.executor_state_snapshot = {
        "ws_event_unmatched_to_local_wo_total": 0,
        "hydration_merged_existing_total": 0,
        "reaper_cancel_pending_reaped_total": 0,
        "reaper_desync_removed_total": 0,
        "reaper_sent_rejected_total": 0,
        "gate_phase2a_invariant_violation_total": 0,
        "risk_exec_state": "SUPPRESSED",
    }
    assert state.telemetry_store.is_healthy() is False


# ---------------------------------------------------------------------------
# BotState wiring
# ---------------------------------------------------------------------------


def test_phase3b_botstate_exposes_all_four_stores() -> None:
    state = _state()
    assert isinstance(state.order_store, OrderStore)
    assert isinstance(state.position_store, PositionStore)
    assert isinstance(state.market_store, MarketStore)
    assert isinstance(state.telemetry_store, TelemetryStore)
