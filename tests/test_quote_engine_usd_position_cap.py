"""Regression tests for ``QuoteEngine._clip_entry_sizes`` with
``MAX_POSITION_NOTIONAL_USD`` clipping.

Pre-2026-05-06, only ``MAX_ABS_POSITION`` (base units) was applied.
At low-priced assets like SUI ($1) the base + USD caps are similar,
but at higher prices the gap grows: a 25-SUI base cap at SUI=$5
permits $125 notional, way over the $20 USD cap. These tests pin
the corrected behaviour.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.quote_engine import QuoteEngine
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _engine(**overrides):
    path = Path(tempfile.gettempdir()) / f"mm_usdcap_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 25.0,
        "MAX_POSITION_NOTIONAL_USD": 20.0,
        # Settings validator requires MAX_ORDER_NOTIONAL_USD <= MAX_POSITION_NOTIONAL_USD.
        "MAX_ORDER_NOTIONAL_USD": 10.0,
    }
    base.update(overrides)
    s = UnitTestSettings.model_validate(base)
    eng = QuoteEngine(s, mock_mm_client().symbol_spec)
    return eng, s, path


def test_usd_cap_clips_long_buy_when_position_at_cap() -> None:
    """Long position already at $20: no further BUY allowed."""
    eng, s, _ = _engine(MAX_POSITION_NOTIONAL_USD=20.0, MAX_ABS_POSITION=1000.0)
    bid, ask, _max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=20.0,  # 20 units × $1 = $20 = at cap
        bid_sz=10.0,
        ask_sz=10.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
    )
    # Buy clipped to 0 (already at cap)
    assert bid == pytest.approx(0.0, abs=1e-6)
    # Sell allowed: max_sell = 20 + 20/1.0 - 0 = 40 units. Capped to ask_sz=10.
    assert ask == pytest.approx(10.0)


def test_usd_cap_at_higher_prices_tightens_clip() -> None:
    """At $5 / unit, the same $20 USD cap = 4 units, NOT 25."""
    eng, s, _ = _engine(MAX_POSITION_NOTIONAL_USD=20.0, MAX_ABS_POSITION=25.0)
    bid, ask, _max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=5.0,
        ask_price=5.0,
    )
    # Allowed long base = $20 / $5 = 4 units. Bid clipped to 4.
    # MAX_ABS_POSITION (25) is looser, but USD cap wins.
    assert bid == pytest.approx(4.0, abs=1e-6)
    assert ask == pytest.approx(4.0, abs=1e-6)


def test_base_units_cap_wins_at_low_prices() -> None:
    """SUI at $0.10: $20 / $0.10 = 200 units allowed by USD cap;
    base cap of 25 wins."""
    eng, s, _ = _engine(MAX_POSITION_NOTIONAL_USD=20.0, MAX_ABS_POSITION=25.0)
    bid, ask, _max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=0.10,
        ask_price=0.10,
    )
    # Tighter cap = 25. Bid/ask each clipped to 25.
    assert bid == pytest.approx(25.0)
    assert ask == pytest.approx(25.0)


def test_resting_orders_count_against_usd_cap() -> None:
    """Long 5 units already + 10 units resting BID + USD cap of $20:
    new BID can only add what fits in remaining USD headroom."""
    eng, s, _ = _engine(MAX_POSITION_NOTIONAL_USD=20.0, MAX_ABS_POSITION=1000.0)
    bid, ask, _max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=5.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=10.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
    )
    # USD cap allows 20 long. Already 5 + 10 resting = 15. Headroom = 5.
    assert bid == pytest.approx(5.0, abs=1e-6)


def test_zero_max_position_notional_disables_usd_cap() -> None:
    """``MAX_POSITION_NOTIONAL_USD=0`` disables USD cap; only base
    cap applies. Settings validator requires order_notional <= pos_notional,
    so we set both to 0 to mean "uncapped"."""
    eng, s, _ = _engine(
        MAX_POSITION_NOTIONAL_USD=0.0,
        MAX_ABS_POSITION=25.0,
        MAX_ORDER_NOTIONAL_USD=0.0,
    )
    bid, ask, _max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=5.0,  # would trigger USD cap if enabled
        ask_price=5.0,
    )
    assert bid == pytest.approx(25.0)
    assert ask == pytest.approx(25.0)


def test_zero_prices_skip_usd_cap_safely() -> None:
    """Backward-compat: callers that pass 0 for prices (legacy
    diagnostics paths) get the old base-only behaviour rather than
    a divide-by-zero or false-zero clip."""
    eng, s, _ = _engine(MAX_POSITION_NOTIONAL_USD=20.0, MAX_ABS_POSITION=25.0)
    bid, ask, _max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        # No prices supplied → USD cap skipped
    )
    assert bid == pytest.approx(25.0)
    assert ask == pytest.approx(25.0)


def test_short_side_usd_cap_symmetric() -> None:
    """Short -20 units already at $20 cap: no further SELL."""
    eng, s, _ = _engine(MAX_POSITION_NOTIONAL_USD=20.0, MAX_ABS_POSITION=1000.0)
    bid, ask, _max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=-20.0,  # -20 units × $1 = -$20 = at short cap
        bid_sz=10.0,
        ask_sz=10.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
    )
    # Sell clipped to 0 (already at short cap)
    assert ask == pytest.approx(0.0, abs=1e-6)


def test_long_position_allows_sell_to_unwind() -> None:
    """Long 30 units (over cap due to fills): SELL should be allowed
    (close-direction), not blocked by the cap. The cap restricts
    GROWING beyond the cap, not unwinding from past it."""
    eng, s, _ = _engine(MAX_POSITION_NOTIONAL_USD=20.0, MAX_ABS_POSITION=25.0)
    bid, ask, _max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=30.0,  # already past 25 base + $20 USD caps
        bid_sz=10.0,
        ask_sz=10.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
    )
    # BUY: blocked (would grow further long)
    assert bid == pytest.approx(0.0, abs=1e-6)
    # SELL: should permit unwind
    assert ask > 0
