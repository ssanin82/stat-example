import pytest

from app.models import Fill, PositionSnapshot
from app.pnl import PnlTracker
from app.enums import Side
from app.utils.time import utc_now


def test_pnl_total_is_net_of_fees() -> None:
    """BUG-011: total_pnl_usd must be NET of fees so MAX_SESSION_LOSS_USD
    and MAX_DRAWDOWN_USD bound NET loss, not gross.
    """
    tr = PnlTracker()
    f = Fill(
        fill_id="x",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=utc_now(),
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.01,
        liquidity_flag="x",
        mid_at_fill=100.0,
    )
    tr.on_fill(f, closed_pnl_component=0.5)
    pos = PositionSnapshot("ETH", 0.1, 100.0, 101.0, 10.0, 0.2)
    snap = tr.build_snapshot(pos, 1000.0)
    assert snap.fees_usd == pytest.approx(0.01)
    expected = snap.realized_pnl_usd + snap.unrealized_pnl_usd - snap.fees_usd
    assert snap.total_pnl_usd == pytest.approx(expected)


def test_pnl_fee_drag_can_trigger_session_loss_kill() -> None:
    """BUG-011 reproducer: gross PnL at zero, fees accumulate; total_pnl_usd
    must reflect the fee drag so the runtime kill fires.
    """
    tr = PnlTracker()
    # Simulate $20 of fees with zero gross realized PnL.
    for i in range(20):
        f = Fill(
            fill_id=f"f{i}",
            order_id_exchange=i,
            client_order_id=None,
            ts_fill=utc_now(),
            symbol="ETH",
            side=Side.BUY,
            price=100.0,
            size=0.1,
            notional=10.0,
            fee=1.0,
            liquidity_flag="x",
            mid_at_fill=100.0,
        )
        tr.on_fill(f, closed_pnl_component=0.0)
    pos = PositionSnapshot("ETH", 0.0, None, None, 0.0, 0.0)
    snap = tr.build_snapshot(pos, 1000.0)
    assert snap.fees_usd == pytest.approx(20.0)
    assert snap.total_pnl_usd == pytest.approx(-20.0)


def test_pnl_credits_maker_rebate_to_total() -> None:
    """BUG-018 follow-up: maker rebates (negative fee per the bot's
    signed-fee convention) must INCREASE total_pnl_usd, not decrease
    it. Pre-2026-05-05 the OKX adapter abs()-stripped the fee sign,
    making rebates indistinguishable from fees-paid and undercounting
    net PnL by 2× the rebate.
    """
    tr = PnlTracker()
    # Simulate $5 of maker REBATE income (negative fee per convention)
    # against a flat realized-PnL session.
    for i in range(20):
        f = Fill(
            fill_id=f"rb{i}",
            order_id_exchange=i,
            client_order_id=None,
            ts_fill=utc_now(),
            symbol="ETH",
            side=Side.BUY,
            price=100.0,
            size=0.1,
            notional=10.0,
            # -0.25 means we received 0.25 USD as maker rebate per fill.
            fee=-0.25,
            liquidity_flag="resting",
            mid_at_fill=100.0,
        )
        tr.on_fill(f, closed_pnl_component=0.0)
    pos = PositionSnapshot("ETH", 0.0, None, None, 0.0, 0.0)
    snap = tr.build_snapshot(pos, 1000.0)
    # Sum of -0.25 × 20 = -5.0 (negative fees = net rebate received).
    assert snap.fees_usd == pytest.approx(-5.0)
    # total = realized(0) + unreal(0) - fees(-5) = +5 NET INCOME
    # from rebates. This is the correctness assertion.
    assert snap.total_pnl_usd == pytest.approx(5.0)


def test_drawdown_tracks_peak_equity() -> None:
    tr = PnlTracker()
    pos = PositionSnapshot("ETH", 0.0, None, None, 0.0, 0.0)
    s1 = tr.build_snapshot(pos, 1000.0)
    assert s1.drawdown_usd == 0.0
    s2 = tr.build_snapshot(pos, 900.0)
    assert s2.drawdown_usd == 100.0
