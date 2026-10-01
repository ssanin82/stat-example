"""Smoke tests for the Binance Futures USDM adapter boundary.

Plan reference: ``plans/20260420-binance-move/plan.md`` Phase 2.6.

Goals:

* The adapter is constructible with mocked REST responses and
  structurally satisfies :class:`PerpExchangeAdapter`.
* HMAC signing is deterministic and signs over the actual urlencoded
  query string.
* Wire-format interpreters map canonical Binance response shapes to
  the bot core's normalised tuples (place / cancel / status).
* WS event parser maps ORDER_TRADE_UPDATE → PrivateOrderUpdateEvent +
  PrivateFillEvent shape.
* The factory wires ``EXCHANGE=binance`` to BinanceClient and the
  trading-public-WS module.
* Credential validation gates fire when ``EXCHANGE=binance`` and
  ``TRADING_ENABLED=true``.
"""

from __future__ import annotations

import hashlib
import hmac
import queue
import time
from typing import Any
from unittest.mock import patch

import pytest

from app.config import Settings, require_trading_credentials_when_enabled
from app.enums import Side
from app.exchange.binance_client import BinanceClient
from app.exchange.binance_responses import (
    interpret_binance_cancel_response,
    interpret_binance_order_status_response,
    interpret_binance_place_response,
    make_deterministic_binance_client_order_id,
)
from app.exchange.binance_ws import BinancePrivateStream
from app.exchange.factory import (
    build_adapter,
    venue_account_address,
)
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
)
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> Settings:
    base = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "binance",
        "REFERENCE_EXCHANGE": "off",
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "GRVT_API_KEY": "",
        "GRVT_API_SECRET": "",
        "GRVT_ACCOUNT_ADDRESS": "",
        "GRVT_SUB_ACCOUNT_ID": "",
        "BLUEFIN_PRIVATE_KEY": "",
        "BLUEFIN_ACCOUNT_ADDRESS": "",
        "BINANCE_API_KEY": "k_test",
        "BINANCE_API_SECRET": "s_test_secret",
        "BINANCE_REST_URL": "https://fapi.binance.com",
        "BINANCE_PRIVATE_WS_URL": "wss://fstream.binance.com",
        "SYMBOL": "DOGEUSDT",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _exchange_info_response() -> dict[str, Any]:
    return {
        "timezone": "UTC",
        "serverTime": 1700000000000,
        "symbols": [
            {
                "symbol": "DOGEUSDT",
                "pair": "DOGEUSDT",
                "contractType": "PERPETUAL",
                "status": "TRADING",
                "quotePrecision": 4,
                "quantityPrecision": 0,
                "filters": [
                    {"filterType": "PRICE_FILTER", "tickSize": "0.00001"},
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "1",
                        "minQty": "1",
                    },
                    {"filterType": "MIN_NOTIONAL", "notional": "5"},
                ],
            }
        ],
    }


def _make_client(s: Settings) -> BinanceClient:
    """Build a BinanceClient with REST mocked at construction so the
    exchangeInfo bootstrap returns deterministic data.
    """
    with patch.object(BinanceClient, "_request") as m:
        m.return_value = _exchange_info_response()
        return BinanceClient(s)


# ---------------------------------------------------------------------------
# Settings / credentials
# ---------------------------------------------------------------------------


def test_credential_validation_fails_when_trading_enabled_without_keys() -> None:
    s = _settings(TRADING_ENABLED=True, BINANCE_API_KEY="", BINANCE_API_SECRET="")
    with pytest.raises(ValueError, match="BINANCE_API_KEY"):
        require_trading_credentials_when_enabled(s)


def test_credential_validation_fails_when_secret_missing() -> None:
    s = _settings(TRADING_ENABLED=True, BINANCE_API_KEY="k", BINANCE_API_SECRET="")
    with pytest.raises(ValueError, match="BINANCE_API_SECRET"):
        require_trading_credentials_when_enabled(s)


def test_credential_validation_passes_with_full_creds() -> None:
    s = _settings(TRADING_ENABLED=True, BINANCE_API_KEY="k", BINANCE_API_SECRET="sec")
    require_trading_credentials_when_enabled(s)  # no raise


# ---------------------------------------------------------------------------
# Adapter constructible + symbol-spec bootstrap
# ---------------------------------------------------------------------------


def test_adapter_constructs_and_bootstraps_symbol_spec() -> None:
    s = _settings()
    client = _make_client(s)
    assert client.symbol_spec_fetched_ok is True
    spec = client.symbol_spec
    assert spec.source == "binance_meta"
    assert spec.price_tick == pytest.approx(0.00001)
    assert spec.size_step == pytest.approx(1.0)
    assert spec.min_size == pytest.approx(1.0)
    # MIN_NOTIONAL = 5 in the synthetic response — adapter floor at $5
    assert spec.min_notional_usd == pytest.approx(5.0)


def test_adapter_satisfies_perp_protocol_isinstance_check() -> None:
    """``PerpExchangeAdapter`` is a runtime-checkable Protocol; the
    Binance client should satisfy it structurally.
    """
    from app.exchange.base import PerpExchangeAdapter

    s = _settings()
    client = _make_client(s)
    assert isinstance(client, PerpExchangeAdapter)


def test_has_write_access_requires_both_key_and_secret() -> None:
    s = _settings(BINANCE_API_KEY="k", BINANCE_API_SECRET="")
    client = _make_client(s)
    assert client.has_write_access() is False
    s2 = _settings(BINANCE_API_KEY="k", BINANCE_API_SECRET="sec")
    client2 = _make_client(s2)
    assert client2.has_write_access() is True


# ---------------------------------------------------------------------------
# HMAC signing
# ---------------------------------------------------------------------------


def test_sign_appends_timestamp_recvwindow_signature() -> None:
    """The signed-params dict must contain timestamp, recvWindow, and a
    valid HMAC-SHA256 signature over the urlencoded query string.
    """
    s = _settings(BINANCE_RECV_WINDOW_MS=5000)
    client = _make_client(s)
    params = {"symbol": "DOGEUSDT", "side": "BUY", "quantity": "10"}
    signed = client._sign(params)
    assert "timestamp" in signed
    assert signed["recvWindow"] == 5000
    assert "signature" in signed
    # Recompute and verify
    from urllib.parse import urlencode

    body = {k: v for k, v in signed.items() if k != "signature"}
    expected = hmac.new(
        b"s_test_secret", urlencode(body, doseq=True).encode(), hashlib.sha256
    ).hexdigest()
    assert signed["signature"] == expected


def test_sign_does_not_mutate_input() -> None:
    s = _settings()
    client = _make_client(s)
    params = {"symbol": "DOGEUSDT"}
    snapshot = dict(params)
    client._sign(params)
    assert params == snapshot


# ---------------------------------------------------------------------------
# Place / cancel response interpreters
# ---------------------------------------------------------------------------


def test_place_response_accepted_extracts_oid() -> None:
    oid, outcome, reason = interpret_binance_place_response(
        {"orderId": 12345, "status": "NEW", "_http_status": 200}
    )
    assert oid == 12345
    assert outcome == "accepted"
    assert reason == ""


def test_place_response_post_only_would_cross_5022() -> None:
    oid, outcome, reason = interpret_binance_place_response(
        {"code": -5022, "msg": "Post-only order would cross"}
    )
    assert oid is None
    assert outcome == "exchange_rejected"
    assert "post_only_would_cross" in reason


def test_place_response_post_only_would_cross_2020() -> None:
    """Some Binance endpoints surface -2020 instead of -5022 for the
    same condition. Both must map to ``post_only_would_cross``.
    """
    oid, outcome, reason = interpret_binance_place_response(
        {"code": -2020, "msg": "Order would immediately match"}
    )
    assert oid is None
    assert outcome == "exchange_rejected"
    assert "post_only_would_cross" in reason


def test_place_response_insufficient_margin() -> None:
    oid, outcome, reason = interpret_binance_place_response(
        {"code": -2010, "msg": "Account has insufficient balance"}
    )
    assert oid is None
    assert outcome == "exchange_rejected"
    assert "insufficient_margin" in reason


def test_place_response_clock_skew() -> None:
    oid, outcome, reason = interpret_binance_place_response(
        {"code": -1021, "msg": "Timestamp outside recvWindow"}
    )
    assert outcome == "exchange_rejected"
    assert "timestamp_skew" in reason


def test_place_response_transport_500() -> None:
    oid, outcome, reason = interpret_binance_place_response(
        {"code": -1000, "msg": "transport_error:DNS lookup failed"}
    )
    assert outcome == "transport_rejected"


def test_cancel_response_success() -> None:
    kind, detail = interpret_binance_cancel_response(
        {"orderId": 12345, "status": "CANCELED", "_http_status": 200}
    )
    assert kind == "success"


def test_cancel_response_unknown_order_is_benign() -> None:
    kind, detail = interpret_binance_cancel_response(
        {"code": -2011, "msg": "Unknown order sent."}
    )
    assert kind == "benign_missing"


def test_cancel_response_transport() -> None:
    kind, detail = interpret_binance_cancel_response(
        {"code": -1000, "msg": "transport_error:..."}
    )
    assert kind == "transport"


def test_status_response_open() -> None:
    oid, outcome, detail = interpret_binance_order_status_response(
        {"orderId": 99, "status": "NEW"}
    )
    assert oid == 99 and outcome == "open"


def test_status_response_filled() -> None:
    oid, outcome, _ = interpret_binance_order_status_response(
        {"orderId": 99, "status": "FILLED"}
    )
    assert outcome == "filled"


def test_status_response_canceled() -> None:
    oid, outcome, _ = interpret_binance_order_status_response(
        {"orderId": 99, "status": "CANCELED"}
    )
    assert outcome == "canceled"


def test_status_response_not_found() -> None:
    _, outcome, _ = interpret_binance_order_status_response(
        {"code": -2011, "msg": "Unknown order sent."}
    )
    assert outcome == "not_found"


# ---------------------------------------------------------------------------
# Deterministic client order id
# ---------------------------------------------------------------------------


def test_cloid_is_36_chars_or_less_and_deterministic() -> None:
    a = make_deterministic_binance_client_order_id(
        "DOGEUSDT", Side.BUY, "q1", 0.4231, 50.0
    )
    b = make_deterministic_binance_client_order_id(
        "DOGEUSDT", Side.BUY, "q1", 0.4231, 50.0
    )
    assert a == b
    assert len(a) <= 36
    assert a.startswith("0x")


def test_cloid_changes_when_inputs_change() -> None:
    a = make_deterministic_binance_client_order_id(
        "DOGEUSDT", Side.BUY, "q1", 0.4231, 50.0
    )
    b = make_deterministic_binance_client_order_id(
        "DOGEUSDT", Side.BUY, "q1", 0.4231, 51.0  # different size
    )
    assert a != b


# ---------------------------------------------------------------------------
# WS parser
# ---------------------------------------------------------------------------


def test_ws_order_trade_update_parses_to_order_update_and_fill() -> None:
    s = _settings()
    q: queue.Queue = queue.Queue(maxsize=10)
    stream = BinancePrivateStream(
        s,
        out_queue=q,
        listen_key_provider=lambda: "fake_listenkey",
        listen_key_keepalive=lambda: True,
    )
    msg = {
        "e": "ORDER_TRADE_UPDATE",
        "E": 1700000000000,
        "T": 1700000000001,
        "o": {
            "s": "DOGEUSDT",
            "c": "0xabcd",
            "S": "BUY",
            "o": "LIMIT",
            "x": "TRADE",
            "X": "PARTIALLY_FILLED",
            "i": 12345,
            "l": "10",
            "z": "20",
            "q": "50",
            "p": "0.4231",
            "L": "0.4230",
            "n": "0.0042",
            "rp": "0.001",
            "T": 1700000000002,
            "t": 99887766,
            "m": True,
        },
    }
    stream._handle_raw_message(msg)
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    types = [type(ev).__name__ for ev in events]
    assert "PrivateOrderUpdateEvent" in types
    assert "PrivateFillEvent" in types
    fill = next(ev for ev in events if isinstance(ev, PrivateFillEvent))
    assert fill.fill_id == "99887766"
    assert fill.oid == 12345
    assert fill.coin == "DOGEUSDT"
    assert fill.px == pytest.approx(0.4230)
    assert fill.sz == pytest.approx(10.0)
    assert fill.fee == pytest.approx(0.0042)
    assert fill.crossed is False  # m=True → maker → not crossed
    order_update = next(
        ev for ev in events if isinstance(ev, PrivateOrderUpdateEvent)
    )
    assert order_update.oid == 12345
    assert order_update.coin == "DOGEUSDT"
    assert order_update.status == "PARTIALLY_FILLED"
    assert order_update.cloid == "0xabcd"
    assert order_update.orig_sz == pytest.approx(50.0)
    assert order_update.remaining_sz == pytest.approx(30.0)


def test_ws_listen_key_expired_closes_connection() -> None:
    """``listenKeyExpired`` event should trigger ws.close() so the
    outer reconnect loop respawns a fresh key.
    """
    s = _settings()
    q: queue.Queue = queue.Queue(maxsize=10)
    stream = BinancePrivateStream(
        s,
        out_queue=q,
        listen_key_provider=lambda: "fake_listenkey",
        listen_key_keepalive=lambda: True,
    )

    closed = {"hit": False}

    class _FakeWS:
        def close(self) -> None:
            closed["hit"] = True

    stream._ws_app = _FakeWS()
    stream._handle_raw_message({"e": "listenKeyExpired"})
    assert closed["hit"] is True


def test_ws_account_update_does_not_emit_events() -> None:
    """ACCOUNT_UPDATE just wakes the quote loop — no fill/order events
    are emitted from it.
    """
    s = _settings()
    q: queue.Queue = queue.Queue(maxsize=10)
    stream = BinancePrivateStream(
        s,
        out_queue=q,
        listen_key_provider=lambda: "fake_listenkey",
        listen_key_keepalive=lambda: True,
    )
    stream._handle_raw_message({"e": "ACCOUNT_UPDATE", "a": {}})
    assert q.empty()


# ---------------------------------------------------------------------------
# Factory wiring
# ---------------------------------------------------------------------------


def test_factory_build_adapter_returns_binance_client() -> None:
    s = _settings()
    with patch.object(BinanceClient, "_request") as m:
        m.return_value = _exchange_info_response()
        client = build_adapter(s)
    assert isinstance(client, BinanceClient)


def test_factory_venue_account_address_uses_api_key_prefix() -> None:
    s = _settings(BINANCE_API_KEY="ABCDEFGHIJKLMNOP")
    addr = venue_account_address(s)
    assert addr == "ABCDEFGH..."
    # short keys (under 8 chars) should not be padded with "..."
    s2 = _settings(BINANCE_API_KEY="ABC")
    assert venue_account_address(s2) == "ABC"


def test_factory_venue_account_address_empty_when_no_key() -> None:
    s = _settings(BINANCE_API_KEY="")
    assert venue_account_address(s) == ""
