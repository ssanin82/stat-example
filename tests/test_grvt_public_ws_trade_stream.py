"""Trade-stream message parsing on :class:`GrvtPublicStream` (Priority #3 v1).

Pinned invariants:
- A well-formed ``v1.trade`` message pushes a ``TradePrint`` onto
  ``state.recent_trades`` and updates ``state.flow_score``.
- ``is_taker_buyer`` → Side.BUY; ``False`` → Side.SELL.
- Malformed messages (missing price/size, wrong instrument, unknown
  stream) are silently ignored — the parser must not crash the WS
  thread.
- The BBO stream continues to work alongside the trade stream (one
  shared connection, two subscriptions).
"""

from __future__ import annotations

import json
from typing import Any

from app.enums import Side
from app.exchange.grvt_public_ws import GrvtPublicStream
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _make_stream(applied: list[Any]) -> tuple[GrvtPublicStream, BotState]:
    settings = UnitTestSettings.model_validate(
        {
            "EXCHANGE": "grvt",
            "GRVT_API_KEY": "x",
            "GRVT_API_SECRET": "x",
            "GRVT_SUB_ACCOUNT_ID": "1",
            "SYMBOL": "ETH_USDT_Perp",
            "GRVT_TRADE_STREAM_ENABLED": True,
            "GRVT_TRADE_STREAM_NAME": "v1.trade",
        }
    )
    state = BotState(settings)
    stream = GrvtPublicStream(
        settings,
        state,
        "ETH_USDT_Perp",
        on_bbo=lambda bb: applied.append(bb),
    )
    return stream, state


def _trade_msg(**feed_overrides: Any) -> str:
    feed: dict[str, Any] = {
        "instrument": "ETH_USDT_Perp",
        "event_time": "1700000000000000000",
        "price": "86.12",
        "size": "0.5",
        "is_taker_buyer": True,
        "trade_id": "abc123",
    }
    feed.update(feed_overrides)
    return json.dumps({"stream": "v1.trade", "feed": feed})


# ---------- Happy path ----------

def test_buy_aggressor_trade_lands_in_state() -> None:
    stream, state = _make_stream([])
    stream.feed_message_for_tests(_trade_msg(is_taker_buyer=True))
    assert len(state.recent_trades) == 1
    tp = state.recent_trades[0]
    assert tp.aggressor_side == Side.BUY
    assert tp.price == 86.12
    assert tp.size == 0.5
    assert tp.trade_id == "abc123"


def test_sell_aggressor_trade_maps_correctly() -> None:
    stream, state = _make_stream([])
    stream.feed_message_for_tests(_trade_msg(is_taker_buyer=False, trade_id="xyz"))
    assert state.recent_trades[0].aggressor_side == Side.SELL


def test_trade_updates_flow_score_accumulator() -> None:
    """A BUY aggressor should immediately register in the flow score's
    TFI / streak — confirming that record_trade is called end-to-end
    from the message handler."""
    stream, state = _make_stream([])
    # Send 3 BUY aggressor prints.
    for i in range(3):
        stream.feed_message_for_tests(
            _trade_msg(
                is_taker_buyer=True,
                trade_id=f"t_{i}",
                event_time=f"{1700000000000000000 + i * 10_000_000}",
            )
        )
    snap = state.flow_score.snapshot_dict()
    # All 3 prints recorded; streak should be 3 on the BUY side.
    assert snap["trade_history_count"] == 3
    assert snap["streak_buy_count"] == 3
    assert snap["streak_sell_count"] == 0


# ---------- Malformed ----------

def test_wrong_instrument_is_dropped() -> None:
    stream, state = _make_stream([])
    stream.feed_message_for_tests(_trade_msg(instrument="BTC_USDT_Perp"))
    assert len(state.recent_trades) == 0


def test_missing_price_is_dropped() -> None:
    stream, state = _make_stream([])
    feed = {"instrument": "ETH_USDT_Perp", "size": "0.5", "is_taker_buyer": True}
    stream.feed_message_for_tests(
        json.dumps({"stream": "v1.trade", "feed": feed})
    )
    assert len(state.recent_trades) == 0


def test_zero_size_is_dropped() -> None:
    stream, state = _make_stream([])
    stream.feed_message_for_tests(_trade_msg(size="0"))
    assert len(state.recent_trades) == 0


def test_missing_aggressor_flag_is_dropped() -> None:
    stream, state = _make_stream([])
    feed = {"instrument": "ETH_USDT_Perp", "price": "86.12", "size": "0.5"}
    stream.feed_message_for_tests(
        json.dumps({"stream": "v1.trade", "feed": feed})
    )
    assert len(state.recent_trades) == 0


def test_unknown_stream_is_silently_dropped() -> None:
    stream, state = _make_stream([])
    # Stream name doesn't match v1.trade and isn't our BBO stream either.
    stream.feed_message_for_tests(
        json.dumps({"stream": "v1.somethingelse", "feed": {}})
    )
    assert len(state.recent_trades) == 0


def test_fallback_taker_side_string_parsing() -> None:
    """If is_taker_buyer is absent but taker_side is present as string,
    the parser falls back to that. Guards against GRVT field-name drift."""
    stream, state = _make_stream([])
    feed = {
        "instrument": "ETH_USDT_Perp",
        "price": "86.12",
        "size": "0.5",
        "taker_side": "SELL",
    }
    stream.feed_message_for_tests(
        json.dumps({"stream": "v1.trade", "feed": feed})
    )
    assert len(state.recent_trades) == 1
    assert state.recent_trades[0].aggressor_side == Side.SELL


# ---------- Co-existence with BBO stream ----------

def test_bbo_stream_still_works_alongside_trades() -> None:
    """Sending a BBO message after a trade should still invoke on_bbo
    and not be mis-routed to the trade handler."""
    applied: list[Any] = []
    stream, state = _make_stream(applied)
    stream.feed_message_for_tests(_trade_msg())
    bbo_msg = json.dumps(
        {
            "stream": "v1.mini.s",
            "feed": {
                "instrument": "ETH_USDT_Perp",
                "best_bid_price": "100.00",
                "best_ask_price": "100.05",
                "best_bid_size": "10",
                "best_ask_size": "10",
                "event_time": "1700000000000000000",
            },
        }
    )
    stream.feed_message_for_tests(bbo_msg)
    assert len(state.recent_trades) == 1  # the trade
    assert len(applied) == 1             # the BBO emission
