"""Bybit public-WS subscriber — message parsing + state updates.

Production code: ``app/exchange/bybit_public_ws.py``. The Bybit stream
is an alternative to Binance selected via ``REFERENCE_EXCHANGE=bybit``;
it writes to the same ``state.binance_*`` fields so downstream
cross-venue-cancel logic is venue-agnostic.

Invariants pinned here:

  1. Snapshot messages populate ``state.binance_*`` top-of-book fields.
  2. Delta messages preserve the last-seen side when only one side updates.
  3. Messages without a top-of-book payload (subscribe ack, pong) are ignored.
  4. Malformed / crossed / non-orderbook messages are silently dropped.
  5. Basis EWMA seeds from the first Binance tick once GRVT mid is present
     and smooths over subsequent samples (same math as Binance path).
  6. ``exchange_ts_ms`` is read from ``ts`` (Bybit v5 server send time).
"""

from __future__ import annotations

import pytest

from app.exchange.bybit_public_ws import BybitPublicStream
from app.models import BestBidAsk
from app.state import BotState
from app.utils.time import utc_now
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "REFERENCE_EXCHANGE": "bybit",
        "BINANCE_WS_ENABLED": True,
        "BINANCE_BASIS_EWMA_ALPHA": 0.5,
        "BYBIT_SYMBOL": "ETHUSDT",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_stream_with_state(**overrides) -> tuple[BybitPublicStream, BotState]:
    settings = _settings(**overrides)
    state = BotState(settings)
    stream = BybitPublicStream(settings, state, on_bbo_callback=None)
    return stream, state


# ---------- Snapshot ----------

def test_snapshot_populates_top_of_book() -> None:
    stream, state = _make_stream_with_state()
    msg = (
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":1744024600123,'
        '"data":{"s":"ETHUSDT","b":[["2340.10","10.5"]],"a":[["2340.20","7.2"]],'
        '"u":1,"seq":1}}'
    )
    stream._handle_raw_message(msg)
    assert state.binance_best_bid == pytest.approx(2340.10)
    assert state.binance_best_ask == pytest.approx(2340.20)
    assert state.binance_mid == pytest.approx(2340.15)
    assert state.binance_bid_size == pytest.approx(10.5)
    assert state.binance_ask_size == pytest.approx(7.2)
    assert state.binance_last_message_wall_ts is not None


def test_snapshot_with_empty_side_is_dropped() -> None:
    """A snapshot is expected to carry both sides. Missing side ⇒ drop."""
    stream, state = _make_stream_with_state()
    msg = (
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":1,'
        '"data":{"s":"ETHUSDT","b":[],"a":[["2340.20","7.2"]],"u":1}}'
    )
    stream._handle_raw_message(msg)
    assert state.binance_best_bid is None
    assert state.binance_best_ask is None


# ---------- Delta ----------

def test_delta_preserves_untouched_side() -> None:
    """Delta with only ``a`` updated keeps the last-seen bid."""
    stream, state = _make_stream_with_state()
    # Seed from a snapshot.
    stream._handle_raw_message(
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":1,'
        '"data":{"s":"ETHUSDT","b":[["2340.10","10.5"]],"a":[["2340.20","7.2"]],"u":1}}'
    )
    # Delta updates only the ask.
    stream._handle_raw_message(
        '{"topic":"orderbook.1.ETHUSDT","type":"delta","ts":2,'
        '"data":{"s":"ETHUSDT","b":[],"a":[["2340.25","4.0"]],"u":2}}'
    )
    assert state.binance_best_bid == pytest.approx(2340.10)
    assert state.binance_bid_size == pytest.approx(10.5)
    assert state.binance_best_ask == pytest.approx(2340.25)
    assert state.binance_ask_size == pytest.approx(4.0)


def test_delta_before_snapshot_is_noop() -> None:
    """Before a snapshot establishes baseline, a delta-only side update
    leaves state untouched."""
    stream, state = _make_stream_with_state()
    stream._handle_raw_message(
        '{"topic":"orderbook.1.ETHUSDT","type":"delta","ts":1,'
        '"data":{"s":"ETHUSDT","b":[["2340.10","5.0"]],"a":[],"u":1}}'
    )
    # Only bid was updated; no last-seen ask — drop silently.
    assert state.binance_best_bid is None
    assert state.binance_best_ask is None


# ---------- Non-orderbook messages ----------

@pytest.mark.parametrize("payload", [
    '{"op":"subscribe","success":true,"ret_msg":"subscribe"}',
    '{"op":"pong"}',
    '{"op":"ping"}',
    '{"topic":"tickers.ETHUSDT","data":{}}',  # wrong topic family
])
def test_control_messages_ignored(payload) -> None:
    stream, state = _make_stream_with_state()
    stream._handle_raw_message(payload)
    assert state.binance_best_bid is None
    assert state.binance_best_ask is None


@pytest.mark.parametrize("payload", [
    b"not json",
    "",
    "[]",
    '{"topic":"orderbook.1.ETHUSDT"}',  # no data
    '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","data":[]}',  # data wrong type
])
def test_malformed_payload_is_dropped(payload) -> None:
    stream, state = _make_stream_with_state()
    stream._handle_raw_message(payload)
    assert state.binance_best_bid is None


def test_crossed_book_is_dropped() -> None:
    """ask < bid indicates a bad message or a race; drop rather than trust."""
    stream, state = _make_stream_with_state()
    msg = (
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":1,'
        '"data":{"s":"ETHUSDT","b":[["2340.50","1"]],"a":[["2340.10","1"]],"u":1}}'
    )
    stream._handle_raw_message(msg)
    assert state.binance_best_bid is None
    assert state.binance_best_ask is None


# ---------- Basis EWMA ----------

def test_basis_requires_grvt_mid() -> None:
    stream, state = _make_stream_with_state()
    stream._handle_raw_message(
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":1,'
        '"data":{"s":"ETHUSDT","b":[["2340.0","1"]],"a":[["2340.10","1"]],"u":1}}'
    )
    assert state.binance_basis_ewma is None

    state.market = BestBidAsk(
        symbol="ETH", best_bid=2340.20, best_ask=2340.30, mid_price=2340.25,
        spread_bps=0.4, ts_local=utc_now(),
    )
    stream._handle_raw_message(
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":2,'
        '"data":{"s":"ETHUSDT","b":[["2340.0","1"]],"a":[["2340.10","1"]],"u":2}}'
    )
    # Binance mid 2340.05, GRVT mid 2340.25, raw basis = +0.20.
    assert state.binance_basis_ewma == pytest.approx(0.20, abs=1e-6)


def test_basis_ewma_smooths() -> None:
    stream, state = _make_stream_with_state(BINANCE_BASIS_EWMA_ALPHA=0.5)
    state.market = BestBidAsk(
        symbol="ETH", best_bid=2340.0, best_ask=2340.0, mid_price=2340.0,
        spread_bps=0.0, ts_local=utc_now(),
    )
    # Bybit mid 2339 → raw basis +1.0
    stream._handle_raw_message(
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":1,'
        '"data":{"s":"ETHUSDT","b":[["2338.90","1"]],"a":[["2339.10","1"]],"u":1}}'
    )
    assert state.binance_basis_ewma == pytest.approx(1.0, abs=1e-6)
    # Bybit mid 2341 → raw basis −1.0. EWMA = 0.5 * -1 + 0.5 * 1 = 0.
    stream._handle_raw_message(
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":2,'
        '"data":{"s":"ETHUSDT","b":[["2340.90","1"]],"a":[["2341.10","1"]],"u":2}}'
    )
    assert state.binance_basis_ewma == pytest.approx(0.0, abs=1e-6)


# ---------- Timing ingest ----------

def test_exchange_ts_read_from_ts_field() -> None:
    """When the ``binance_public_ws_timing`` tracker is present it should
    receive the ``ts`` field as ``exchange_ts_ms``."""
    from app.market_data_timing import PublicWsTimingTracker

    stream, state = _make_stream_with_state()
    state.binance_public_ws_timing = PublicWsTimingTracker(
        symbol="ETHUSDT", max_samples=100
    )
    msg = (
        '{"topic":"orderbook.1.ETHUSDT","type":"snapshot","ts":1744024600123,'
        '"data":{"s":"ETHUSDT","b":[["2340.10","1"]],"a":[["2340.20","1"]],"u":1}}'
    )
    stream._handle_raw_message(msg)
    summary = state.binance_public_ws_timing.summary()
    assert summary["total_samples"] == 1
    assert summary["samples_with_exchange_ts"] == 1
