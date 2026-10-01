"""Mini-delta BBO parsing lives on :class:`GrvtPublicStream`.

The old stateless ``_best_bid_ask_from_grvt_feed`` helper was removed when
we switched from ``v1.book.s@500-10`` (full-snapshot, stateless) to
``v1.mini.d@50`` (tick delta, stateful). Parsing now keeps the last-seen
BBO on the stream object and emits whenever the composed bid+ask is known.
"""

from __future__ import annotations

import json
from typing import Any

from app.exchange.grvt_public_ws import (
    GrvtPublicStream,
    _GRVT_PUBLIC_RATE_MS,
    _GRVT_PUBLIC_STREAM,
)
from tests.settings_helpers import UnitTestSettings
from app.state import BotState


def _make_stream(applied: list[Any]) -> GrvtPublicStream:
    settings = UnitTestSettings.model_validate(
        {
            "EXCHANGE": "grvt",
            "GRVT_API_KEY": "x",
            "GRVT_API_SECRET": "x",
            "GRVT_SUB_ACCOUNT_ID": "1",
            "SYMBOL": "ETH_USDT_Perp",
        }
    )
    state = BotState(settings)
    return GrvtPublicStream(
        settings,
        state,
        "ETH_USDT_Perp",
        on_bbo=lambda bb: applied.append(bb),
    )


def _msg(**feed_overrides: Any) -> str:
    feed: dict[str, Any] = {
        "instrument": "ETH_USDT_Perp",
        "event_time": "1700000000000000000",
    }
    feed.update(feed_overrides)
    return json.dumps({"stream": _GRVT_PUBLIC_STREAM, "feed": feed})


def test_constants_pin_mini_snapshot_at_fastest_rate() -> None:
    """Guards against a silent revert to ``v1.book.s`` or a cadence regression.

    Snapshot (``.s``) not delta (``.d``) because GRVT's delta feed goes silent
    when the touch doesn't change — which on a tight book kept ``gap_p95_ms``
    above the freshness gate. Snapshot at 200 ms forces a heartbeat.
    """
    assert _GRVT_PUBLIC_STREAM == "v1.mini.s"
    assert _GRVT_PUBLIC_RATE_MS == 200


def test_mini_delta_initial_full_snapshot_emits_bbo() -> None:
    applied: list[Any] = []
    s = _make_stream(applied)
    s.feed_message_for_tests(_msg(best_bid_price="100", best_ask_price="101"))
    assert len(applied) == 1
    bb = applied[0]
    assert bb.best_bid == 100.0
    assert bb.best_ask == 101.0
    assert bb.mid_price == 100.5
    assert bb.ts_exchange_ms == 1_700_000_000_000


def test_mini_delta_bid_only_update_uses_cached_ask() -> None:
    applied: list[Any] = []
    s = _make_stream(applied)
    s.feed_message_for_tests(_msg(best_bid_price="100", best_ask_price="101"))
    s.feed_message_for_tests(
        _msg(event_time="1700000000050000000", best_bid_price="100.5")
    )
    assert [round(bb.mid_price, 4) for bb in applied] == [100.5, 100.75]
    assert applied[1].best_ask == 101.0


def test_mini_delta_first_half_snapshot_waits_for_other_side() -> None:
    """A bid-only delta before we've ever seen an ask must not emit."""
    applied: list[Any] = []
    s = _make_stream(applied)
    s.feed_message_for_tests(_msg(best_bid_price="100"))
    assert applied == []
    s.feed_message_for_tests(_msg(best_ask_price="101"))
    assert len(applied) == 1


def test_other_streams_are_ignored() -> None:
    applied: list[Any] = []
    s = _make_stream(applied)
    s.feed_message_for_tests(
        json.dumps(
            {
                "stream": "v1.book.s",
                "feed": {
                    "instrument": "ETH_USDT_Perp",
                    "event_time": "1700000000000000000",
                    "bids": [{"price": "100", "size": "1"}],
                    "asks": [{"price": "101", "size": "1"}],
                },
            }
        )
    )
    assert applied == []


def test_subscribe_ack_is_not_parsed_as_bbo() -> None:
    applied: list[Any] = []
    s = _make_stream(applied)
    s.feed_message_for_tests(
        '{"jsonrpc":"2.0","id":1,"result":{"subscribed":true}}'
    )
    assert applied == []


def test_reset_cache_drops_last_prices() -> None:
    """A reconnect must not cross a stale cached BBO boundary."""
    applied: list[Any] = []
    s = _make_stream(applied)
    s.feed_message_for_tests(_msg(best_bid_price="100", best_ask_price="101"))
    s._reset_cache()
    # Bid-only delta after reset should not emit (no ask yet).
    s.feed_message_for_tests(_msg(best_bid_price="105"))
    assert len(applied) == 1  # only the pre-reset emission
