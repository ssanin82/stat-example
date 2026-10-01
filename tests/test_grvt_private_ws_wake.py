"""GRVT private-WS must wake the main quote loop on fill/order events.

Ports the HL private-WS ``_after_queue_put`` pattern. Without this wake,
fills and order updates sit in ``queue.Queue`` for up to
``quote_loop_seconds`` before the main tick drains them — the 405 ms
``private_ws_queue_wait_ms`` stall observed in
``tmp/snap_20260417_183547``.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.exchange.grvt_ws import GrvtPrivateStream
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "EXCHANGE": "grvt",
            "TRADING_ENABLED": False,
            "PRIVATE_WS_ENABLED": True,
            "SYMBOL": "ETH_USDT_Perp",
        }
    )


def _mock_state() -> SimpleNamespace:
    state = SimpleNamespace()
    state.wake_quote_loop = MagicMock()
    return state


def test_fill_event_wakes_main_loop() -> None:
    q: queue.Queue = queue.Queue()
    state = _mock_state()
    stream = GrvtPrivateStream(_settings(), "123", q, state=state)
    stream.feed_message_for_tests(
        '{"stream":"v1.fill","feed":{"event_time":"1700000000000000000",'
        '"trade_id":"t1","order_id":"0x15","instrument":"ETH_USDT_Perp",'
        '"is_buyer":true,"size":"0.01","price":"3000","fee":"-0.002","realized_pnl":"0"}}'
    )
    # The wake MUST fire on the enqueue so the main loop doesn't wait for
    # the next ``quote_loop_seconds`` tick to drain this fill.
    assert state.wake_quote_loop.call_count == 1


def test_order_update_wakes_main_loop() -> None:
    q: queue.Queue = queue.Queue()
    state = _mock_state()
    stream = GrvtPrivateStream(_settings(), "123", q, state=state)
    stream.feed_message_for_tests(
        '{"stream":"v1.order","feed":{"order_id":"0x10",'
        '"legs":[{"instrument":"ETH_USDT_Perp","size":"0.02","limit_price":"3001",'
        '"is_buying_asset":false}],'
        '"state":{"status":"OPEN","reject_reason":"UNSPECIFIED",'
        '"book_size":["0.02"],"update_time":"1700000001000000000"}}}'
    )
    assert state.wake_quote_loop.call_count == 1


def test_connection_events_do_not_wake_main_loop() -> None:
    """Connection events land on the queue but don't need to wake the loop —
    the quote engine only cares about fills and order updates. Matches HL
    behaviour (``label in ("fill", "order_update")``)."""
    q: queue.Queue = queue.Queue()
    state = _mock_state()
    stream = GrvtPrivateStream(_settings(), "123", q, state=state)
    # ``_emit_conn`` is the internal path connection lifecycle events take.
    from app.exchange.private_events import PrivateWsConnectionKind

    stream._emit_conn(PrivateWsConnectionKind.CONNECTED, detail="test")
    assert state.wake_quote_loop.call_count == 0
    # But the event did land on the queue.
    assert q.qsize() == 1


def test_multiple_fills_coalesce_into_multiple_wakes_but_one_tick_drains_all() -> None:
    """HFT-grade coalescing: each fill sets the wake event (idempotent) so a
    burst of events doesn't produce a tick-storm, but one tick draining the
    queue still processes all of them. This mirrors HL behaviour and keeps
    CPU usage flat under high fill rates."""
    q: queue.Queue = queue.Queue()
    state = _mock_state()
    stream = GrvtPrivateStream(_settings(), "123", q, state=state)
    for i in range(5):
        stream.feed_message_for_tests(
            '{"stream":"v1.fill","feed":{"event_time":"1700000000000000000",'
            f'"trade_id":"t{i}","order_id":"0x15","instrument":"ETH_USDT_Perp",'
            '"is_buyer":true,"size":"0.01","price":"3000","fee":"-0.002","realized_pnl":"0"}}'
        )
    # 5 wakes fired (1 per event) — the real ``threading.Event`` would be
    # idempotently set, which is what the mocked method simulates.
    assert state.wake_quote_loop.call_count == 5
    # 5 events in the queue — one drain cycle processes them all.
    assert q.qsize() == 5


def test_wake_not_called_if_state_missing() -> None:
    """Streams constructed without state (e.g. in low-level tests) must not
    crash on the wake path."""
    q: queue.Queue = queue.Queue()
    stream = GrvtPrivateStream(_settings(), "123", q, state=None)
    # Should not raise.
    stream.feed_message_for_tests(
        '{"stream":"v1.fill","feed":{"event_time":"1700000000000000000",'
        '"trade_id":"t1","order_id":"0x15","instrument":"ETH_USDT_Perp",'
        '"is_buyer":true,"size":"0.01","price":"3000","fee":"-0.002","realized_pnl":"0"}}'
    )
    assert q.qsize() == 1
