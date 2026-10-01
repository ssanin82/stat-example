"""Hyperliquid private WS parsing and stream test hooks (no live socket)."""

from __future__ import annotations

import json
import queue

import pytest

from app.exchange.hyperliquid_ws import (
    HyperliquidPrivateStream,
    _normalize_fill_row,
    _normalize_order_row,
    classify_private_ws_inactive_close,
)
from app.state import BotState
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
    PrivateWsConnectionEvent,
    PrivateWsConnectionKind,
)
from tests.settings_helpers import UnitTestSettings


def _ws_settings() -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "HL_WS_URL": "wss://unit.test/ws",
            "PRIVATE_WS_ENABLED": True,
        }
    )


def test_normalize_fill_row_valid() -> None:
    f = {
        "coin": "ETH",
        "px": "3000.5",
        "sz": "0.02",
        "side": "B",
        "time": 1_700_000_000_000,
        "hash": "0xabc",
        "oid": 99,
        "fee": "-0.01",
        "closedPnl": "0",
        "crossed": False,
    }
    ev = _normalize_fill_row(f, is_snapshot=True)
    assert ev is not None
    assert isinstance(ev, PrivateFillEvent)
    assert ev.fill_id == "0xabc_1700000000000"
    assert ev.coin == "ETH"
    assert ev.px == 3000.5
    assert ev.sz == 0.02
    assert ev.oid == 99
    assert ev.is_snapshot is True
    assert ev.crossed is False


def test_normalize_fill_row_malformed_returns_none() -> None:
    assert _normalize_fill_row({}, False) is None
    assert _normalize_fill_row({"coin": "ETH"}, False) is None


def test_normalize_order_row_valid() -> None:
    row = {
        "order": {
            "coin": "ETH",
            "side": "A",
            "limitPx": "3010",
            "sz": "0.05",
            "origSz": "0.1",
            "oid": 42,
            "timestamp": 1,
        },
        "status": "open",
        "statusTimestamp": 12345,
    }
    ev = _normalize_order_row(row)
    assert ev is not None
    assert isinstance(ev, PrivateOrderUpdateEvent)
    assert ev.oid == 42
    assert ev.coin == "ETH"
    assert ev.status == "open"
    assert ev.status_timestamp_ms == 12345
    assert ev.remaining_sz == 0.05
    assert ev.orig_sz == 0.1
    assert ev.limit_px == 3010.0


def test_normalize_order_row_malformed_returns_none() -> None:
    assert _normalize_order_row({}) is None
    assert _normalize_order_row({"order": "bad"}) is None


def test_feed_message_user_fills_enqueues_private_fill_events() -> None:
    q: queue.Queue = queue.Queue()
    s = _ws_settings()
    stream = HyperliquidPrivateStream(s, "0xabc", q)
    payload = {
        "channel": "userFills",
        "data": {
            "isSnapshot": False,
            "fills": [
                {
                    "coin": "ETH",
                    "px": "1",
                    "sz": "2",
                    "side": "B",
                    "time": 100,
                    "hash": "h1",
                    "oid": 1,
                    "fee": "0",
                    "closedPnl": "0",
                    "crossed": True,
                }
            ],
        },
    }
    stream.feed_message_for_tests(json.dumps(payload))
    ev = q.get_nowait()
    assert isinstance(ev, PrivateFillEvent)
    assert ev.fill_id == "h1_100"


def test_feed_message_order_updates_enqueues_order_events() -> None:
    q: queue.Queue = queue.Queue()
    s = _ws_settings()
    stream = HyperliquidPrivateStream(s, "0xabc", q)
    payload = {
        "channel": "orderUpdates",
        "data": [
            {
                "order": {
                    "coin": "ETH",
                    "side": "B",
                    "limitPx": "100",
                    "sz": "0.1",
                    "origSz": "0.1",
                    "oid": 7,
                    "timestamp": 1,
                },
                "status": "open",
                "statusTimestamp": 999,
            }
        ],
    }
    stream.feed_message_for_tests(json.dumps(payload))
    ev = q.get_nowait()
    assert isinstance(ev, PrivateOrderUpdateEvent)
    assert ev.oid == 7


def test_feed_message_malformed_json_does_not_crash() -> None:
    q: queue.Queue = queue.Queue()
    s = _ws_settings()
    stream = HyperliquidPrivateStream(s, "0xabc", q)
    stream.feed_message_for_tests("not json {{{")
    assert q.empty()


def test_feed_message_user_fills_bad_shape_no_enqueue() -> None:
    q: queue.Queue = queue.Queue()
    s = _ws_settings()
    stream = HyperliquidPrivateStream(s, "0xabc", q)
    stream.feed_message_for_tests(
        json.dumps({"channel": "userFills", "data": "not_a_dict"})
    )
    assert q.empty()


def test_connection_lifecycle_emit_conn_enqueues_events() -> None:
    q: queue.Queue = queue.Queue()
    s = _ws_settings()
    stream = HyperliquidPrivateStream(s, "0xabc", q)
    stream._emit_conn(PrivateWsConnectionKind.CONNECTED, "test")
    stream._emit_conn(PrivateWsConnectionKind.DISCONNECTED, "bye")
    a = q.get_nowait()
    b = q.get_nowait()
    assert isinstance(a, PrivateWsConnectionEvent)
    assert a.kind == PrivateWsConnectionKind.CONNECTED
    assert isinstance(b, PrivateWsConnectionEvent)
    assert b.kind == PrivateWsConnectionKind.DISCONNECTED


def test_stream_stop_without_start_is_clean() -> None:
    q: queue.Queue = queue.Queue()
    s = _ws_settings()
    stream = HyperliquidPrivateStream(s, "0xabc", q)
    stream.stop()  # should not raise
    assert stream._thread is None


@pytest.mark.parametrize(
    "channel,data",
    [
        ("subscriptionResponse", {"type": "userFills"}),
        ("pong", None),
    ],
)
def test_feed_message_control_channels_no_enqueue(channel: str, data: object) -> None:
    q: queue.Queue = queue.Queue()
    s = _ws_settings()
    stream = HyperliquidPrivateStream(s, "0xabc", q)
    stream.feed_message_for_tests(json.dumps({"channel": channel, "data": data}))
    assert q.empty()


def test_classify_private_ws_inactive_close() -> None:
    assert classify_private_ws_inactive_close(1000, "Inactive")
    assert classify_private_ws_inactive_close(None, "connection inactive")
    assert not classify_private_ws_inactive_close(1000, "going away")


def test_feed_message_pong_updates_state_timestamps() -> None:
    q: queue.Queue = queue.Queue()
    s = _ws_settings()
    state = BotState(s)
    stream = HyperliquidPrivateStream(s, "0xabc", q, state=state)
    stream.feed_message_for_tests(json.dumps({"channel": "pong", "data": None}))
    assert state.private_ws_last_pong_wall_ts is not None
    assert state.private_ws_last_message_wall_ts is not None
    assert q.empty()
