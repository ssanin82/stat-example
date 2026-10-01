"""Inbound private/public WS timing, backlog metrics, and decision/fill attribution."""

from __future__ import annotations

from datetime import datetime, timezone

from app.inbound_timing import (
    InboundPrivateTiming,
    InboundPublicTiming,
    private_timing_derived_ms,
    public_timing_derived_ms,
)
from app.models import BestBidAsk, Fill
from app.enums import Side
from app.fill_ingestion import fill_row
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def test_private_timing_derived_ms_populated() -> None:
    t = InboundPrivateTiming(
        ws_recv_mono=10.0,
        ws_recv_wall=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        parse_start_mono=10.001,
        parse_end_mono=10.003,
        enqueue_mono=10.004,
        dequeue_mono=10.05,
        handler_start_mono=10.05,
        handler_end_mono=10.08,
        state_apply_mono=10.081,
        exchange_event_ms=1_735_689_600_000,
    )
    d = private_timing_derived_ms(t)
    assert d["private_ws_receive_to_parse_ms"] is not None
    assert d["private_ws_queue_wait_ms"] is not None
    assert d["private_ws_receive_to_state_apply_ms"] is not None
    assert d["exchange_to_local_private_receive_ms"] is not None


def test_public_timing_derived_ms_populated() -> None:
    t = InboundPublicTiming(
        ws_recv_mono=1.0,
        ws_recv_wall=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        parse_start_mono=1.001,
        parse_end_mono=1.004,
        callback_start_mono=1.005,
        apply_mono=1.02,
        exchange_ts_ms=1_735_689_600_000,
    )
    d = public_timing_derived_ms(t)
    assert d["public_ws_receive_to_apply_ms"] is not None
    assert d["public_ws_exchange_to_local_receive_ms"] is not None


def test_apply_market_book_only_sets_public_last_derived() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    st = BotState(s)
    it = InboundPublicTiming(
        ws_recv_mono=100.0,
        ws_recv_wall=datetime.now(timezone.utc),
        parse_start_mono=100.0,
        parse_end_mono=100.002,
        callback_start_mono=100.003,
        apply_mono=0.0,
        exchange_ts_ms=1_000,
    )
    bb = BestBidAsk(
        symbol="ETH",
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        spread_bps=20.0,
        ts_exchange_ms=1_000,
        ts_local=datetime.now(timezone.utc),
        inbound_public_timing=it,
    )
    st.apply_market_book_only(bb, market_data_source="public_ws")
    assert "public_ws_receive_to_apply_ms" in st.public_ws_last_inbound_derived_ms


def test_note_private_inbound_metrics_updates_queue_wait_latency() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    st = BotState(s)
    t = InboundPrivateTiming(
        ws_recv_mono=1.0,
        parse_start_mono=1.0,
        parse_end_mono=1.001,
        enqueue_mono=1.002,
        dequeue_mono=1.1,
        handler_start_mono=1.1,
        handler_end_mono=1.11,
        state_apply_mono=1.12,
    )
    d = private_timing_derived_ms(t)
    st.note_private_inbound_metrics(t, d)
    assert st.latency_private_queue_wait_ms is not None


def test_fill_row_includes_latency_attribution_columns() -> None:
    f = Fill(
        fill_id="x",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="ETH",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=0.0,
        liquidity_flag="resting",
        mid_at_fill=1.0,
        effective_book_age_at_last_decision_ms=12.0,
        effective_book_age_at_fill_ms=34.0,
        private_ws_receive_to_state_apply_ms=5.0,
        quote_cycle_to_first_transport_send_ms=6.0,
        quote_cycle_to_first_ack_ms=7.0,
        ack_to_private_ws_lifecycle_ms=8.0,
    )
    row = fill_row(f)
    assert row["effective_book_age_at_last_decision_ms"] == 12.0
    assert row["private_ws_receive_to_state_apply_ms"] == 5.0


def test_fill_row_includes_closed_pnl() -> None:
    """Storage schema v18 added a ``closed_pnl`` column on the
    ``fills`` table so postmortem reports can compute win-rate /
    profit-factor / per-fill PnL distributions. The value is
    already on every Fill (every venue adapter populates it);
    this test guards the persistence wiring."""
    now = datetime.now(timezone.utc)
    f_with_realized = Fill(
        fill_id="closing-leg",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=now,
        symbol="SUI-USDT-SWAP",
        side=Side.SELL,
        price=1.10,
        size=10.0,
        notional=11.0,
        fee=-0.005,
        liquidity_flag="resting",
        mid_at_fill=1.10,
        closed_pnl=0.42,
    )
    f_opening = Fill(
        fill_id="opening-leg",
        order_id_exchange=2,
        client_order_id=None,
        ts_fill=now,
        symbol="SUI-USDT-SWAP",
        side=Side.BUY,
        price=1.00,
        size=10.0,
        notional=10.0,
        fee=-0.005,
        liquidity_flag="resting",
        mid_at_fill=1.00,
        # closed_pnl=None by default — opening fill carries no realized
    )
    row_with = fill_row(f_with_realized)
    row_open = fill_row(f_opening)
    assert row_with["closed_pnl"] == 0.42
    assert row_open["closed_pnl"] is None


def test_status_flags_include_private_backlog_keys() -> None:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    st = BotState(s)
    st.private_ws_queue_depth_after_drain = 3
    st.private_ws_drain_time_ms_last_tick = 1.5
    st.private_ws_events_drained_last_tick = 7
    flags = st.status_flags_dict()
    assert flags["private_ws_queue_depth_after_drain"] == 3
    assert flags["private_ws_drain_time_ms_last_tick"] == 1.5
    assert flags["private_ws_events_drained_last_tick"] == 7
