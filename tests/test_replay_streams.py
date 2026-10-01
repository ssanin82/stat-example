"""Tests for ``app.backtest.streams`` — Phase 3 (v1.4.234)."""

from __future__ import annotations

from queue import Queue

import pytest

from app.backtest import RecordedEvent, ReplayPrivateStream, ReplayPublicStream
from app.clock import ReplayClock
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
)
from app.models import BestBidAsk


def _clock() -> ReplayClock:
    return ReplayClock(start_t_ns=1_700_000_000_000_000_000)


# ---------------------------------------------------------------------------
# ReplayPublicStream — OKX bbo-tbt / books5
# ---------------------------------------------------------------------------


def _okx_bbo_event(*, bid: str, bid_sz: str, ask: str, ask_sz: str, t_ns: int = 100) -> RecordedEvent:
    return RecordedEvent(
        t_recv_ns=t_ns,
        source="okx_public",
        msg={
            "arg": {"channel": "bbo-tbt", "instId": "TON-USDT-SWAP"},
            "data": [{
                "asks": [[ask, ask_sz, "0", "1"]],
                "bids": [[bid, bid_sz, "0", "1"]],
                "ts": "1779353641000",
                "seqId": 1,
            }],
        },
    )


def test_okx_bbo_parsed_into_bestbidask() -> None:
    received: list[BestBidAsk] = []
    s = ReplayPublicStream(clock=_clock(), on_bbo=received.append)
    s.deliver(_okx_bbo_event(bid="2.041", bid_sz="100", ask="2.042", ask_sz="200"))
    assert len(received) == 1
    bba = received[0]
    assert bba.best_bid == 2.041
    assert bba.best_ask == 2.042
    assert bba.bid_size == 100.0
    assert bba.ask_size == 200.0
    assert bba.symbol == "TON-USDT-SWAP"
    assert bba.mid_price == pytest.approx(2.0415)
    assert s.bbo_dispatched == 1


def test_okx_books5_uses_top_level() -> None:
    received: list[BestBidAsk] = []
    s = ReplayPublicStream(clock=_clock(), on_bbo=received.append)
    msg = {
        "arg": {"channel": "books5", "instId": "TON-USDT-SWAP"},
        "data": [{
            "asks": [
                ["2.042", "200", "0", "2"],  # top
                ["2.043", "300", "0", "1"],
            ],
            "bids": [
                ["2.041", "100", "0", "1"],  # top
                ["2.040", "50", "0", "1"],
            ],
            "ts": "1779353641000",
            "seqId": 2,
        }],
    }
    s.deliver(RecordedEvent(t_recv_ns=1, source="okx_public", msg=msg))
    assert len(received) == 1
    assert received[0].best_bid == 2.041
    assert received[0].best_ask == 2.042


def test_okx_subscribe_ack_skipped_silently() -> None:
    """Subscribe acks have no ``data`` key — must be skipped, not crash."""
    received: list[BestBidAsk] = []
    s = ReplayPublicStream(clock=_clock(), on_bbo=received.append)
    s.deliver(RecordedEvent(
        t_recv_ns=1,
        source="okx_public",
        msg={"event": "subscribe", "arg": {"channel": "bbo-tbt"}},
    ))
    assert received == []
    assert s.events_skipped == 1


def test_okx_malformed_book_data_skipped() -> None:
    received: list[BestBidAsk] = []
    s = ReplayPublicStream(clock=_clock(), on_bbo=received.append)
    # Empty asks → cannot derive a touch.
    s.deliver(RecordedEvent(
        t_recv_ns=1,
        source="okx_public",
        msg={"arg": {"channel": "bbo-tbt"}, "data": [{"asks": [], "bids": []}]},
    ))
    assert received == []


def test_okx_negative_or_zero_prices_skipped() -> None:
    received: list[BestBidAsk] = []
    s = ReplayPublicStream(clock=_clock(), on_bbo=received.append)
    s.deliver(_okx_bbo_event(bid="0", bid_sz="100", ask="2.042", ask_sz="100"))
    assert received == []


def test_okx_last_bba_tracked() -> None:
    s = ReplayPublicStream(clock=_clock(), on_bbo=lambda b: None)
    s.deliver(_okx_bbo_event(bid="2.041", bid_sz="100", ask="2.042", ask_sz="100"))
    s.deliver(_okx_bbo_event(bid="2.044", bid_sz="50", ask="2.045", ask_sz="50"))
    assert s.last_okx_bba is not None
    assert s.last_okx_bba.best_bid == 2.044


# ---------------------------------------------------------------------------
# ReplayPublicStream — OKX trades
# ---------------------------------------------------------------------------


def test_okx_trades_dispatched() -> None:
    trades: list[tuple[float, float, str, str]] = []
    s = ReplayPublicStream(clock=_clock(), on_trade=lambda *a: trades.append(a))
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_public",
        msg={
            "arg": {"channel": "trades", "instId": "TON-USDT-SWAP"},
            "data": [{"px": "2.042", "sz": "10", "side": "buy", "ts": "1779353641000"}],
        },
    ))
    assert len(trades) == 1
    assert trades[0] == (2.042, 10.0, "BUY", "okx")
    assert s.trades_dispatched == 1


# ---------------------------------------------------------------------------
# ReplayPublicStream — Binance bookTicker
# ---------------------------------------------------------------------------


def test_binance_book_ticker_parsed() -> None:
    received: list[BestBidAsk] = []
    s = ReplayPublicStream(clock=_clock(), on_bbo=received.append, binance_symbol="TONUSDT")
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="binance_public",
        msg={
            "e": "bookTicker",
            "s": "TONUSDT",
            "b": "2.0482",
            "B": "335.7",
            "a": "2.0483",
            "A": "555.5",
            "T": 1779353641206,
        },
    ))
    assert len(received) == 1
    assert received[0].best_bid == 2.0482
    assert received[0].symbol == "TONUSDT"


def test_binance_non_book_ticker_skipped() -> None:
    received: list[BestBidAsk] = []
    s = ReplayPublicStream(clock=_clock(), on_bbo=received.append)
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="binance_public",
        msg={"e": "depthUpdate"},  # not a bookTicker
    ))
    assert received == []
    assert s.events_skipped == 1


# ---------------------------------------------------------------------------
# ReplayPublicStream — unknown source counted as skip
# ---------------------------------------------------------------------------


def test_unknown_source_counted_as_skip() -> None:
    s = ReplayPublicStream(clock=_clock())
    s.deliver(RecordedEvent(t_recv_ns=1, source="some_other_venue", msg={}))
    assert s.events_seen == 1
    assert s.events_skipped == 1


def test_lifecycle_methods_are_noop() -> None:
    """Production code may call start/stop/request_reconnect — must
    not raise."""
    s = ReplayPublicStream(clock=_clock())
    s.start()
    s.stop()
    s.request_reconnect()


# ---------------------------------------------------------------------------
# ReplayPrivateStream — OKX orders + fills
# ---------------------------------------------------------------------------


def test_okx_order_update_dispatched() -> None:
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_private",
        msg={
            "arg": {"channel": "orders"},
            "data": [{
                "ordId": "1234567890",
                "clOrdId": "paper-xyz",
                "state": "live",
                "side": "buy",
                "px": "2.041",
                "sz": "10",
                "accFillSz": "0",
                "cTime": "1779353641000",
                "uTime": "1779353641100",
            }],
        },
    ))
    ev = sink.get_nowait()
    assert isinstance(ev, PrivateOrderUpdateEvent)
    assert ev.oid == 1234567890
    assert ev.status == "live"
    assert ev.side == "B"
    assert ev.limit_px == 2.041
    assert ev.orig_sz == 10.0
    assert ev.remaining_sz == 10.0
    assert ev.cloid == "paper-xyz"


def test_okx_fill_event_dispatched_with_okx_fee_convention_flipped() -> None:
    """OKX fills carry signed fees: positive=rebate, negative=paid.
    The bot-canonical convention is the opposite — adapter negates.
    Replay must do the same flip so consumers see the bot's view."""
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_private",
        msg={
            "arg": {"channel": "fills"},
            "data": [{
                "billId": "fill-123",
                "ordId": "999",
                "fillPx": "2.041",
                "fillSz": "5",
                "side": "buy",
                "ts": "1779353641000",
                "fee": "0.002",  # positive in OKX → rebate
                "pnl": "0",
                "execType": "M",
            }],
        },
    ))
    ev = sink.get_nowait()
    assert isinstance(ev, PrivateFillEvent)
    assert ev.fill_id == "fill-123"
    assert ev.oid == 999
    assert ev.px == 2.041
    assert ev.sz == 5.0
    # Flipped: bot sees negative = rebate received.
    assert ev.fee == -0.002
    assert ev.crossed is False  # 'M' = maker


def test_okx_unknown_channel_skipped() -> None:
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_private",
        msg={"arg": {"channel": "positions"}, "data": [{}]},
    ))
    assert sink.empty()
    assert s.events_skipped == 1


def test_non_okx_private_source_ignored() -> None:
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    s.deliver(RecordedEvent(t_recv_ns=1, source="okx_public", msg={}))
    assert sink.empty()
    assert s.events_skipped == 1


def test_okx_malformed_order_row_skipped() -> None:
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_private",
        msg={
            "arg": {"channel": "orders"},
            "data": [{"ordId": "not-a-number"}],  # bad oid
        },
    ))
    assert sink.empty()


def test_forward_to_sink_false_suppresses_order_update() -> None:
    """--with-bot mode: recorded order-updates are parsed + counted as
    suppressed but NOT pushed onto the bot's queue (paper-exec is the
    authoritative private-event source). Prevents the shadow-position
    divergence storm diagnosed in the v1.5.294 audit."""
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink, forward_to_sink=False)
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_private",
        msg={
            "arg": {"channel": "orders"},
            "data": [{
                "ordId": "1234567890",
                "clOrdId": "foreign-live-session",
                "state": "live",
                "side": "buy",
                "px": "2.041",
                "sz": "10",
                "accFillSz": "0",
                "uTime": "1779353641100",
            }],
        },
    ))
    assert sink.empty()  # nothing reaches the bot's queue
    assert s.order_updates_dispatched == 0
    assert s.order_updates_suppressed == 1
    assert s.events_seen == 1


def test_forward_to_sink_false_suppresses_fill() -> None:
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink, forward_to_sink=False)
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_private",
        msg={
            "arg": {"channel": "fills"},
            "data": [{
                "billId": "fill-123",
                "ordId": "999",
                "fillPx": "2.041",
                "fillSz": "5",
                "side": "buy",
                "ts": "1779353641000",
                "fee": "0.002",
                "pnl": "0",
                "execType": "M",
            }],
        },
    ))
    assert sink.empty()
    assert s.fills_dispatched == 0
    assert s.fills_suppressed == 1


def test_forward_to_sink_default_true_still_dispatches() -> None:
    """Paper-only mode (no bot) keeps the default: events ARE forwarded
    so the driver can inspect synthetic + recorded events together."""
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    assert s.forward_to_sink is True
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_private",
        msg={
            "arg": {"channel": "orders"},
            "data": [{
                "ordId": "42", "state": "live", "side": "sell",
                "px": "2.0", "sz": "1", "accFillSz": "0",
                "uTime": "1779353641100",
            }],
        },
    ))
    assert not sink.empty()
    assert s.order_updates_dispatched == 1
    assert s.order_updates_suppressed == 0


def test_private_stream_lifecycle_noop() -> None:
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    s.start()
    s.stop()
    s.request_reconnect()


# ---------------------------------------------------------------------------
# BUG-E (v1.5.303) — events_skipped breakdown by reason
# ---------------------------------------------------------------------------


def test_public_skip_reasons_breakdown_and_invariant() -> None:
    """BUG-E fix: every skipped public event is attributed to a reason
    key, and the per-reason counts sum to events_skipped — so a large
    skip total is explainable (benign acks / off-channel) rather than a
    context-free number the operator can't judge."""
    s = ReplayPublicStream(clock=_clock())
    assert s.skip_reasons == {}  # fresh stream
    # Unknown source.
    s.deliver(RecordedEvent(t_recv_ns=1, source="some_other_venue", msg={}))
    # OKX subscribe ack (no "data" key).
    s.deliver(RecordedEvent(
        t_recv_ns=2, source="okx_public",
        msg={"event": "subscribe", "arg": {"channel": "bbo-tbt"}},
    ))
    # OKX off-channel frame.
    s.deliver(RecordedEvent(
        t_recv_ns=3, source="okx_public",
        msg={"arg": {"channel": "candle1m"}, "data": [{}]},
    ))
    # Binance non-bookTicker, twice — proves per-reason counts accumulate.
    for t in (4, 5):
        s.deliver(RecordedEvent(
            t_recv_ns=t, source="binance_public", msg={"e": "depthUpdate"},
        ))
    assert s.skip_reasons == {
        "unknown_source:some_other_venue": 1,
        "okx_subscribe_ack": 1,
        "okx_unknown_channel:candle1m": 1,
        "binance_unknown_event:depthUpdate": 2,
    }
    # Invariant: per-reason counts reconcile to the grand total.
    assert sum(s.skip_reasons.values()) == s.events_skipped == 5


def test_public_skip_reasons_untouched_by_successful_dispatch() -> None:
    """A successfully dispatched BBO must NOT add a skip reason."""
    s = ReplayPublicStream(clock=_clock(), on_bbo=lambda b: None)
    s.deliver(_okx_bbo_event(bid="2.041", bid_sz="100", ask="2.042", ask_sz="100"))
    assert s.bbo_dispatched == 1
    assert s.events_skipped == 0
    assert s.skip_reasons == {}


def test_private_skip_reasons_breakdown_and_invariant() -> None:
    """BUG-E fix: private-stream skips (off-instrument orders, wrong-
    source frames, acks) are attributed, and the counts reconcile to
    events_skipped. This is what turns a scary ``events_skipped=12241``
    into an explainable 'benign positions/account filtering'."""
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    assert s.skip_reasons == {}
    # Wrong source.
    s.deliver(RecordedEvent(t_recv_ns=1, source="okx_public", msg={}))
    # Unknown channels (positions ×2, account ×1).
    for ch, t in (("positions", 2), ("positions", 3), ("account", 4)):
        s.deliver(RecordedEvent(
            t_recv_ns=t, source="okx_private",
            msg={"arg": {"channel": ch}, "data": [{}]},
        ))
    assert s.skip_reasons == {
        "wrong_source:okx_public": 1,
        "unknown_channel:positions": 2,
        "unknown_channel:account": 1,
    }
    assert sum(s.skip_reasons.values()) == s.events_skipped == 4


def test_private_skip_reasons_untouched_by_dispatched_order() -> None:
    """A dispatched order-update must NOT add a skip reason."""
    sink: Queue = Queue()
    s = ReplayPrivateStream(clock=_clock(), event_sink=sink)
    s.deliver(RecordedEvent(
        t_recv_ns=1, source="okx_private",
        msg={
            "arg": {"channel": "orders"},
            "data": [{
                "ordId": "42", "state": "live", "side": "sell",
                "px": "2.0", "sz": "1", "accFillSz": "0",
                "uTime": "1779353641100",
            }],
        },
    ))
    assert s.order_updates_dispatched == 1
    assert s.events_skipped == 0
    assert s.skip_reasons == {}
