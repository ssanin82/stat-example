from __future__ import annotations

import queue

from app.exchange.grvt_ws import GrvtPrivateStream
from app.exchange.private_events import PrivateFillEvent, PrivateOrderUpdateEvent
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


def test_grvt_private_ws_fill_feed_parses() -> None:
    q: queue.Queue = queue.Queue()
    stream = GrvtPrivateStream(_settings(), "123", q)
    stream.feed_message_for_tests(
        '{"stream":"v1.fill","feed":{"event_time":"1700000000000000000","trade_id":"t1","order_id":"0x15","instrument":"ETH_USDT_Perp","is_buyer":true,"size":"0.01","price":"3000","fee":"-0.002","realized_pnl":"0"}}'
    )
    ev = q.get_nowait()
    assert isinstance(ev, PrivateFillEvent)
    assert ev.oid == 21
    assert ev.coin == "ETH_USDT_Perp"


def test_grvt_private_ws_order_feed_parses() -> None:
    q: queue.Queue = queue.Queue()
    stream = GrvtPrivateStream(_settings(), "123", q)
    stream.feed_message_for_tests(
        '{"stream":"v1.order","feed":{"order_id":"0x10","legs":[{"instrument":"ETH_USDT_Perp","size":"0.02","limit_price":"3001","is_buying_asset":false}],"state":{"status":"OPEN","reject_reason":"UNSPECIFIED","book_size":["0.02"],"update_time":"1700000001000000000"}}}'
    )
    ev = q.get_nowait()
    assert isinstance(ev, PrivateOrderUpdateEvent)
    assert ev.oid == 16
    assert ev.status == "OPEN"
