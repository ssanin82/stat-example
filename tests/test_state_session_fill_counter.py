"""BotState maintains a monotonic ``session_fill_count`` (and per-side variant)
that is NOT bounded by the ``recent_fills`` deque maxlen.

Rationale: self-perpetuation guards in ``adaptive_spread_widen`` and
``adverse_side_pause`` require a "has new data arrived?" gate. Using
``len(recent_fills)`` or per-side window counts from the toxicity engine
breaks once the deque hits its maxlen (default 1000) — the length stops
growing, so the gate permanently blocks re-arming.

The monotonic counter solves this by growing indefinitely, independent of
any window or buffer cap.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.enums import Side
from app.models import Fill
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _fill(fill_id: str, side: Side) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="ETH_USDT_Perp",
        side=side,
        price=2375.0,
        size=0.01,
        notional=23.75,
        fee=-0.0001,
        liquidity_flag="resting",
        mid_at_fill=2375.0,
    )


def _state() -> BotState:
    s = UnitTestSettings.model_validate({"TRADING_ENABLED": False, "SYMBOL": "ETH_USDT_Perp"})
    return BotState(s)


def test_counter_starts_at_zero() -> None:
    st = _state()
    assert st.session_fill_count == 0
    assert st.session_fill_count_by_side == {}


def test_each_new_fill_bumps_total_and_per_side() -> None:
    st = _state()
    st.record_fill(_fill("a", Side.BUY))
    assert st.session_fill_count == 1
    assert st.session_fill_count_by_side[Side.BUY] == 1
    st.record_fill(_fill("b", Side.SELL))
    assert st.session_fill_count == 2
    assert st.session_fill_count_by_side[Side.SELL] == 1
    st.record_fill(_fill("c", Side.BUY))
    assert st.session_fill_count == 3
    assert st.session_fill_count_by_side[Side.BUY] == 2
    assert st.session_fill_count_by_side[Side.SELL] == 1


def test_dedup_does_not_double_count() -> None:
    """``record_fill`` is idempotent on fill_id — the counter must respect that."""
    st = _state()
    f = _fill("x", Side.BUY)
    st.record_fill(f)
    st.record_fill(f)  # same fill_id
    st.record_fill(f)
    assert st.session_fill_count == 1
    assert st.session_fill_count_by_side[Side.BUY] == 1


def test_non_session_scoped_record_does_not_count() -> None:
    """Replay path (``session_scoped=False``) must not bump the counter —
    those are historical fills from the REST catchup, not session activity."""
    st = _state()
    f = _fill("x", Side.BUY)
    ok = st.record_fill(f, session_scoped=False)
    assert ok is True  # dedup registered the ID
    assert st.session_fill_count == 0
    assert st.session_fill_count_by_side == {}


def test_counter_grows_past_deque_maxlen() -> None:
    """The deque maxlen must NOT cap the monotonic counter — this is
    the whole point of the counter. Reads maxlen at runtime so the
    test is immune to future widenings."""
    st = _state()
    maxlen = st.recent_fills.maxlen
    n = maxlen + 150  # well past the deque cap, both sides included
    for i in range(n):
        side = Side.BUY if i % 2 == 0 else Side.SELL
        st.record_fill(_fill(f"f{i}", side))
    # Counter keeps growing past the deque limit.
    assert st.session_fill_count == n
    assert st.session_fill_count_by_side[Side.BUY] == (n + 1) // 2
    assert st.session_fill_count_by_side[Side.SELL] == n // 2
    # recent_fills is capped at maxlen.
    assert len(st.recent_fills) == maxlen
