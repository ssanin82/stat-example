"""TODO-004: cumulative ``blind_resting_seconds_total`` metric.

The metric measures wall-clock seconds where ``risk.evaluate_risk()``
returned ``NO_QUOTE`` AND at least one passive order was still on the
book — quantifying BUG-009-class exposure. Should be ~0 in healthy
operation; non-zero values are operator-visible regressions.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "EXCHANGE": "grvt",
            "SYMBOL": "ETH_USDT_Perp",
        }
    )


def _make_working(side: Side, *, status: OrderStatus) -> WorkingOrder:
    return WorkingOrder(
        order_id_local="o-x",
        order_id_exchange=None,
        client_order_id="c-x",
        symbol="ETH_USDT_Perp",
        side=side,
        price=100.0,
        size=0.1,
        post_only=True,
        status=status,
        ts_created=datetime.now(timezone.utc),
        ts_sent=datetime.now(timezone.utc),
        quote_cycle_id="q",
    )


def test_has_resting_passive_order_true_when_acked() -> None:
    state = BotState(_settings())
    state.working_bid = _make_working(Side.BUY, status=OrderStatus.ACKED)
    assert state.has_resting_passive_order() is True


def test_has_resting_passive_order_false_when_only_sent() -> None:
    """SENT means in-flight, not yet on the book — does not count."""
    state = BotState(_settings())
    state.working_bid = _make_working(Side.BUY, status=OrderStatus.SENT)
    assert state.has_resting_passive_order() is False


def test_has_resting_passive_order_includes_cancel_pending() -> None:
    """CANCEL_PENDING orders are still on the book until cancel-confirms.
    They count toward 'blind but resting' exposure."""
    state = BotState(_settings())
    state.working_ask = _make_working(Side.SELL, status=OrderStatus.CANCEL_PENDING)
    assert state.has_resting_passive_order() is True


def test_has_resting_passive_order_false_when_terminal() -> None:
    state = BotState(_settings())
    state.working_bid = _make_working(Side.BUY, status=OrderStatus.CANCELED)
    state.working_ask = _make_working(Side.SELL, status=OrderStatus.FILLED)
    assert state.has_resting_passive_order() is False


def test_blind_resting_seconds_accumulates_only_when_blind_and_resting() -> None:
    state = BotState(_settings())
    # First sample establishes the anchor; no delta accumulated yet.
    state.note_blind_resting_sample(blind=True, now_mono=100.0)
    assert state.blind_resting_seconds_total == 0.0
    assert state.blind_resting_tick_count == 0

    # Blind for 0.5s — should accumulate.
    state.note_blind_resting_sample(blind=True, now_mono=100.5)
    assert abs(state.blind_resting_seconds_total - 0.5) < 1e-9
    assert state.blind_resting_tick_count == 1

    # Not blind for 1.0s — accumulator does NOT advance.
    state.note_blind_resting_sample(blind=False, now_mono=101.5)
    assert abs(state.blind_resting_seconds_total - 0.5) < 1e-9
    assert state.blind_resting_tick_count == 1

    # Blind again for 0.2s.
    state.note_blind_resting_sample(blind=True, now_mono=101.7)
    assert abs(state.blind_resting_seconds_total - 0.7) < 1e-9
    assert state.blind_resting_tick_count == 2


def test_blind_resting_skips_giant_clock_jumps() -> None:
    """A 5-minute jump (sleep / suspend) must not be charged as blind exposure."""
    state = BotState(_settings())
    state.note_blind_resting_sample(blind=True, now_mono=100.0)
    state.note_blind_resting_sample(blind=True, now_mono=100.0 + 300.0)
    assert state.blind_resting_seconds_total == 0.0


def test_status_flags_dict_surfaces_blind_resting_metric() -> None:
    state = BotState(_settings())
    state.note_blind_resting_sample(blind=True, now_mono=100.0)
    state.note_blind_resting_sample(blind=True, now_mono=100.25)
    flags = state.status_flags_dict()
    assert "blind_resting_seconds_total" in flags
    assert "blind_resting_tick_count" in flags
    assert flags["blind_resting_seconds_total"] == 0.25
    assert flags["blind_resting_tick_count"] == 1
