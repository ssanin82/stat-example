"""BUG-028: books5 liveness heartbeat (``BotState.note_book_heartbeat``)
and the OKX dual-subscribe routing that feeds it.

The live OKX feed runs on the event-driven ``bbo-tbt`` channel, which is
silent when the touch is stable. In quiet markets that silence used to
grow the freshness gate's book-age clock past ``QUOTE_HOLD_MAX_BOOK_AGE_MS``
→ ``freshness_drift_hold`` false-fired and suppressed ~70% of quoting.

The fix subscribes ``books5`` on the same socket as a liveness heartbeat.
These tests pin the heartbeat's contract:

* It ALWAYS refreshes the book-age clock + feeds the gap ring (the fix).
* It drives the touch only when at least as fresh as the held touch (the
  frozen-touch safety net) and never clobbers a fresher held touch.
* An ineligible frame is a no-op (proves nothing about liveness).
* The WS layer routes ``books5`` to the heartbeat (NOT ``_on_bbo``) and
  does NOT let a heartbeat set ``_first_book_data_event`` (so the bug-019
  silent-subscribe fallback timer still works).
"""

from __future__ import annotations

import time

from app.exchange.okx_public_ws import OkxPublicStream
from app.models import BestBidAsk
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _state() -> BotState:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    return BotState(s)


def _bba(bid: float, ask: float, ts_exchange_ms: int | None) -> BestBidAsk:
    return BestBidAsk(
        symbol="TON-USDT-SWAP",
        best_bid=bid,
        best_ask=ask,
        mid_price=(bid + ask) / 2.0,
        spread_bps=(ask - bid) / ((bid + ask) / 2.0) * 10_000.0,
        ts_exchange_ms=ts_exchange_ms,
        bid_size=10.0,
        ask_size=10.0,
    )


def _update_count(state: BotState) -> int:
    d = state.market_data_gap_tracker.to_api_dict(
        session_id=state.session_id, symbol="TON-USDT-SWAP"
    )
    return int(d.get("update_count") or 0)


def test_heartbeat_refreshes_age_clock_and_feeds_gap_tracker() -> None:
    state = _state()
    assert state.public_last_bbo_receipt_monotonic is None
    assert _update_count(state) == 0

    state.note_book_heartbeat(_bba(1.0, 2.0, 1000), market_data_source="public_ws")

    # THE FIX: a received book frame refreshes the age clock and feeds
    # the gap ring even on the very first frame.
    assert state.public_last_bbo_receipt_monotonic is not None
    assert _update_count(state) == 1
    # With no prior touch the heartbeat adopts it (frozen-touch safety).
    assert state.market is not None
    assert state.market.best_bid == 1.0
    assert state.market.best_ask == 2.0
    assert state.public_ws_seen_first_bbo is True


def test_heartbeat_drives_touch_when_at_least_as_fresh() -> None:
    state = _state()
    # Held bbo-tbt touch at ts=1000.
    state.apply_market_book_only(_bba(1.0, 2.0, 1000), market_data_source="public_ws")
    assert state.market is not None and state.market.best_bid == 1.0

    # books5 heartbeat arrives FRESHER (ts=2000) with a moved touch.
    state.note_book_heartbeat(_bba(3.0, 4.0, 2000), market_data_source="public_ws")

    # Frozen-touch safety net: the heartbeat drives the touch when its
    # exchange ts is at least as fresh as the held touch.
    assert state.market.best_bid == 3.0
    assert state.market.best_ask == 4.0


def test_heartbeat_does_not_clobber_a_fresher_held_touch() -> None:
    state = _state()
    # Held bbo-tbt touch at ts=2000 (fresh).
    state.apply_market_book_only(_bba(1.0, 2.0, 2000), market_data_source="public_ws")
    before_clock = state.public_last_bbo_receipt_monotonic
    assert before_clock is not None
    before_updates = _update_count(state)

    # A LAGGING books5 snapshot (ts=1000, older) must not move the touch.
    state.note_book_heartbeat(_bba(3.0, 4.0, 1000), market_data_source="public_ws")

    # Touch unchanged (held was fresher) ...
    assert state.market is not None
    assert state.market.best_bid == 1.0
    assert state.market.best_ask == 2.0
    # ... but the age clock + gap ring STILL advanced: an older book
    # frame is still proof the feed is alive.
    assert state.public_last_bbo_receipt_monotonic is not None
    assert state.public_last_bbo_receipt_monotonic > before_clock
    assert _update_count(state) == before_updates + 1


def test_heartbeat_adopts_when_ts_missing() -> None:
    state = _state()
    # Held touch with no exchange ts.
    state.apply_market_book_only(_bba(1.0, 2.0, None), market_data_source="public_ws")
    # Incoming with a ts: can't compare against a None held ts → adopt
    # (bias toward the fresher data).
    state.note_book_heartbeat(_bba(3.0, 4.0, 5000), market_data_source="public_ws")
    assert state.market is not None
    assert state.market.best_bid == 3.0


def test_heartbeat_ignores_ineligible_frame() -> None:
    state = _state()
    # No valid bid/ask → proves nothing about liveness → full no-op.
    ineligible = BestBidAsk(
        symbol="TON-USDT-SWAP",
        best_bid=0.0,
        best_ask=0.0,
        mid_price=0.0,
        spread_bps=None,
        ts_exchange_ms=1000,
    )
    state.note_book_heartbeat(ineligible, market_data_source="public_ws")
    assert state.public_last_bbo_receipt_monotonic is None
    assert _update_count(state) == 0
    assert state.market is None


def test_heartbeat_does_not_bloat_book_snapshots_on_stable_touch() -> None:
    state = _state()
    # First frame establishes the touch (one snapshot row).
    state.note_book_heartbeat(_bba(1.0, 2.0, 1000), market_data_source="public_ws")
    with state._lock:
        n_after_first = len(state._book_snapshots)
    assert n_after_first == 1

    # 50 same-touch re-confirms at advancing ts must NOT append rows —
    # otherwise 100 ms heartbeats would flush the maxlen=400 deque that
    # feeds fill-time book matching.
    for i in range(50):
        state.note_book_heartbeat(
            _bba(1.0, 2.0, 1000 + i + 1), market_data_source="public_ws"
        )
    with state._lock:
        n_after_reconfirms = len(state._book_snapshots)
    assert n_after_reconfirms == 1
    # But the age clock kept advancing across all of them (liveness).
    assert state.public_last_bbo_receipt_monotonic is not None
    assert _update_count(state) == 51


# ---------------------------------------------------------------------------
# WS-layer routing (OkxPublicStream): primary → _on_bbo, books5 → heartbeat.
# ---------------------------------------------------------------------------


def _okx_book_frame(channel: str, bid: float, ask: float, ts: int) -> dict:
    return {
        "arg": {"channel": channel, "instId": "TON-USDT-SWAP"},
        "data": [
            {
                "bids": [[str(bid), "10", "0", "1"]],
                "asks": [[str(ask), "10", "0", "1"]],
                "ts": str(ts),
            }
        ],
    }


def test_okx_routing_books5_to_heartbeat_bbo_tbt_to_primary() -> None:
    state = _state()
    on_bbo_calls: list[BestBidAsk] = []
    stream = OkxPublicStream(
        UnitTestSettings.model_validate({"TRADING_ENABLED": False}),
        state,
        "TON-USDT-SWAP",
        on_bbo_calls.append,
    )
    # Simulate a live dual-subscribe: bbo-tbt primary + books5 heartbeat.
    stream._active_book_channel = "bbo-tbt"
    stream._heartbeat_channel = "books5"

    # A books5 frame must go to the heartbeat path (NOT _on_bbo) and must
    # NOT set _first_book_data_event (else the bug-019 fallback timer
    # would be masked by a non-primary frame).
    stream._handle_raw_message(_okx_book_frame("books5", 3.0, 4.0, 1000))
    assert on_bbo_calls == []
    assert stream._first_book_data_event.is_set() is False
    assert state.market is not None
    assert state.market.best_bid == 3.0  # heartbeat adopted (held was None)

    # A bbo-tbt frame must go to the primary _on_bbo path and set the
    # first-book-data flag.
    stream._handle_raw_message(_okx_book_frame("bbo-tbt", 1.0, 2.0, 2000))
    assert len(on_bbo_calls) == 1
    assert on_bbo_calls[0].best_bid == 1.0
    assert stream._first_book_data_event.is_set() is True


def test_okx_routing_books5_is_primary_after_fallback() -> None:
    state = _state()
    on_bbo_calls: list[BestBidAsk] = []
    stream = OkxPublicStream(
        UnitTestSettings.model_validate({"TRADING_ENABLED": False}),
        state,
        "TON-USDT-SWAP",
        on_bbo_calls.append,
    )
    # After a fallback, books5 IS the primary and there is no heartbeat.
    stream._active_book_channel = "books5"
    stream._heartbeat_channel = None

    stream._handle_raw_message(_okx_book_frame("books5", 1.0, 2.0, 1000))
    assert len(on_bbo_calls) == 1
    assert stream._first_book_data_event.is_set() is True
