"""Unit tests for quote-quality / spread-capture telemetry rollups."""

from datetime import datetime, timezone

import pytest

from app.enums import OrderStatus, Side
from app.models import Fill, WorkingOrder
from app.quote_quality_telemetry import QuoteQualityRollup, build_delayed_markout_summary


def _fill(
    *,
    side: Side,
    price: float,
    size: float,
    mid: float,
    m5: float | None = -2.0,
    fill_id: str = "x",
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="ETH",
        side=side,
        price=price,
        size=size,
        notional=abs(price * size),
        fee=0.0,
        liquidity_flag="maker",
        mid_at_fill=mid,
        markout_5s_bps=m5,
    )


def test_rollup_spread_and_two_sided_percentages() -> None:
    r = QuoteQualityRollup(window_samples=10)
    r.record_quote_cycle(
        quoted_spread_bps=10.0,
        one_tick_wide=False,
        two_sided_effective=True,
        intended_two_sided=True,
    )
    r.record_quote_cycle(
        quoted_spread_bps=None,
        one_tick_wide=False,
        two_sided_effective=False,
        intended_two_sided=False,
    )
    d = r.to_dict(realized_pnl_usd=1.5, fills_for_markout=[], markout_window=50)
    assert d["quote_cycle_samples_in_window"] == 2
    assert d["avg_quoted_spread_bps"] == 10.0
    assert d["median_quoted_spread_bps"] == 10.0
    assert d["pct_time_two_sided_effective"] == 50.0
    assert d["pct_time_one_sided_effective"] == 50.0
    assert d["pct_time_intended_two_sided"] == 50.0
    assert d["pct_time_one_tick_wide_when_two_sided"] == 0.0
    assert d["realized_pnl_usd"] == 1.5


def test_one_tick_pct_only_over_two_sided_spreads() -> None:
    r = QuoteQualityRollup(window_samples=10)
    r.record_quote_cycle(
        quoted_spread_bps=5.0,
        one_tick_wide=True,
        two_sided_effective=True,
        intended_two_sided=True,
    )
    r.record_quote_cycle(
        quoted_spread_bps=20.0,
        one_tick_wide=False,
        two_sided_effective=True,
        intended_two_sided=True,
    )
    d = r.to_dict(realized_pnl_usd=0.0, fills_for_markout=[], markout_window=50)
    assert d["pct_time_one_tick_wide_when_two_sided"] == 50.0


def test_spread_capture_buy_sell() -> None:
    r = QuoteQualityRollup(window_samples=10)
    r.note_fill_spread_capture_usd(
        _fill(side=Side.BUY, price=99.0, size=1.0, mid=100.0, m5=None, fill_id="a")
    )
    r.note_fill_spread_capture_usd(
        _fill(side=Side.SELL, price=101.0, size=1.0, mid=100.0, m5=None, fill_id="b")
    )
    d = r.to_dict(realized_pnl_usd=0.0, fills_for_markout=[], markout_window=50)
    assert d["estimated_gross_spread_capture_usd_session"] == pytest.approx(2.0)


def test_spread_capture_capped_when_delayed_markout_clearly_adverse() -> None:
    r = QuoteQualityRollup(window_samples=20)
    r.note_fill_spread_capture_usd(
        _fill(side=Side.BUY, price=99.0, size=1.0, mid=100.0, m5=-3.0, fill_id="a")
    )
    fills = [
        _fill(
            side=Side.BUY,
            price=1.0,
            size=1.0,
            mid=1.0,
            m5=-3.0,
            fill_id=f"f{i}",
        )
        for i in range(8)
    ]
    d = r.to_dict(realized_pnl_usd=0.0, fills_for_markout=fills, markout_window=50)
    assert d["estimated_gross_spread_capture_usd_session"] == 0.0


def test_spread_widen_signal_high_one_tick_share() -> None:
    r = QuoteQualityRollup(window_samples=80)
    for i in range(50):
        r.record_quote_cycle(
            quoted_spread_bps=0.5,
            one_tick_wide=True,
            two_sided_effective=True,
            intended_two_sided=True,
        )
    assert r.spread_widen_signal(
        fills_for_markout=[],
        markout_window=50,
        min_quote_cycles=45,
    )


def test_spread_widen_signal_adverse_markout() -> None:
    r = QuoteQualityRollup(window_samples=80)
    for _ in range(45):
        r.record_quote_cycle(
            quoted_spread_bps=None,
            one_tick_wide=False,
            two_sided_effective=False,
            intended_two_sided=False,
        )
    fills = [
        _fill(side=Side.BUY, price=1.0, size=1.0, mid=1.0, m5=-2.0, fill_id=f"m{i}")
        for i in range(6)
    ]
    assert r.spread_widen_signal(
        fills_for_markout=fills,
        markout_window=50,
        min_quote_cycles=45,
    )


def test_counters_and_lifetime() -> None:
    r = QuoteQualityRollup(window_samples=10)
    r.note_post_only_cross_rejection()
    r.note_quote_reprice_required()
    r.note_order_lifetime_seconds(3.0)
    r.note_order_lifetime_seconds(1.0)
    d = r.to_dict(realized_pnl_usd=0.0, fills_for_markout=[], markout_window=50)
    assert d["post_only_cross_rejection_count_session"] == 1
    assert d["quote_reprice_required_count_session"] == 1
    assert d["avg_passive_order_lifetime_seconds"] == 2.0
    assert d["passive_order_lifetime_samples"] == 2


def test_delayed_markout_summary_prefers_5s() -> None:
    f0 = _fill(side=Side.BUY, price=1.0, size=1.0, mid=1.0, m5=-1.5)
    f1 = _fill(side=Side.BUY, price=1.0, size=1.0, mid=1.0, m5=2.0)
    s = build_delayed_markout_summary([f0, f1], window=10)
    assert s["delayed_markout_sample_count"] == 2
    assert s["mean_delayed_markout_bps"] == pytest.approx(0.25)
    assert s["adverse_delayed_markout_count"] == 1
    assert s["favorable_delayed_markout_count"] == 1


def test_note_passive_order_lifetime_idempotent() -> None:
    from app.state import BotState

    from tests.settings_helpers import UnitTestSettings

    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "QUOTE_QUALITY_WINDOW_SAMPLES": 100,
        }
    )
    st = BotState(s)
    wo = WorkingOrder(
        order_id_local="a",
        order_id_exchange=1,
        client_order_id="c",
        symbol="ETH",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    wo.ts_ack = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    wo.ts_closed = datetime(2024, 1, 1, 0, 0, 2, tzinfo=timezone.utc)
    wo.status = OrderStatus.FILLED
    with st._lock:
        st.note_passive_order_lifetime_if_new(wo)
        st.note_passive_order_lifetime_if_new(wo)
    d = st.quote_quality_dict()
    assert d["passive_order_lifetime_samples"] == 1
    assert d["avg_passive_order_lifetime_seconds"] == 2.0
