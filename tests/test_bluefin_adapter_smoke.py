"""Smoke tests for the Bluefin Pro adapter boundary.

Goals:

* The adapter is constructible with mocked env vars / symbol metadata and
  structurally satisfies :class:`PerpExchangeAdapter` (no import cycles, no
  missing methods).
* Wire-format interpreters map canonical Bluefin Pro response shapes to the
  bot core's normalised tuples (place / cancel / status).
* Public + private WS parsers can ingest a synthetic message without
  raising — the fill/order-update events land on the out-queue.
* Order signing is deterministic for fixed inputs (guards against
  accidental drift in the pretty-printed JSON format or BCS intent bytes).
* Credential-validation gates fire for EXCHANGE=bluefin.

We do NOT connect real sockets or hit real HTTP in these tests — the
HTTP client is patched and WS streams use ``feed_message_for_tests``.
"""

from __future__ import annotations

import queue
from typing import Any
from unittest.mock import patch

import pytest

from app.config import Settings, require_trading_credentials_when_enabled
from app.exchange.base import PerpExchangeAdapter
from app.exchange.bluefin_auth import (
    BluefinOrderSignPayload,
    default_order_expiration_ms,
    fresh_salt,
    parse_ed25519_secret,
    sign_login_request,
    sign_order_payload,
    signable_create_order,
    signable_login,
)
from app.exchange.bluefin_responses import (
    hash_to_oid,
    interpret_bluefin_cancel_response,
    interpret_bluefin_order_status_response,
    interpret_bluefin_place_response,
    make_deterministic_bluefin_client_order_id,
)
from app.exchange.factory import (
    build_adapter,
    build_private_stream,
    build_public_stream,
    venue_account_address,
)
from app.state import BotState
from app.enums import Side
from tests.settings_helpers import UnitTestSettings


_DUMMY_HEX64 = "0x" + "01" * 32
_DUMMY_PRIVKEY = "0x" + ("00" * 31) + "01"


def _settings(**overrides) -> Settings:
    """Bluefin-profile UnitTestSettings with blank credentials by default."""
    base = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "bluefin",
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "GRVT_API_KEY": "",
        "GRVT_API_SECRET": "",
        "GRVT_ACCOUNT_ADDRESS": "",
        "GRVT_SUB_ACCOUNT_ID": "",
        "BLUEFIN_PRIVATE_KEY": _DUMMY_PRIVKEY,
        "BLUEFIN_ACCOUNT_ADDRESS": _DUMMY_HEX64,
        "BLUEFIN_NETWORK": "SUI_PROD",
        "BLUEFIN_ONE_CT_ENABLED": True,
        "BLUEFIN_ONE_CT_DURATION_HOURS": 24.0,
        "BLUEFIN_REST_URL": "https://api.sui-prod.bluefin.io",
        "BLUEFIN_AUTH_URL": "https://auth.api.sui-prod.bluefin.io",
        "BLUEFIN_TRADE_URL": "https://trade.api.sui-prod.bluefin.io",
        "BLUEFIN_PUBLIC_WS_URL": "wss://stream.api.sui-prod.bluefin.io",
        "BLUEFIN_PRIVATE_WS_URL": "wss://stream.api.sui-prod.bluefin.io",
        "SYMBOL": "SUI-PERP",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _state(s: Settings) -> BotState:
    return BotState(s)


def _patch_exchange_info_response() -> Any:
    """Short-circuit the /v1/exchange/info REST call used at bootstrap.

    Matches the pro-sdk ``ExchangeInfoResponse`` schema: a top-level
    object with ``markets: [Market, ...]`` and ``contractsConfig: {idsId: ...}``.
    Numeric fields are in 1e9 base.
    """
    return {
        "markets": [
            {
                "symbol": "SUI-PERP",
                "marketAddress": _DUMMY_HEX64,
                "tickSizeE9": "100000",  # 0.0001 * 1e9
                "stepSizeE9": "100000000",  # 0.1 * 1e9
                "minOrderQuantityE9": "100000000",  # 0.1 * 1e9
                "minTradePriceE9": "100000",  # 0.0001 * 1e9
                "minTradeQuantityE9": "100000000",
            }
        ],
        "contractsConfig": {"idsId": _DUMMY_HEX64},
        "tradingGasFeeE9": "0",
        "serverTimeAtMillis": 1700000000000,
        "timezone": "UTC",
    }


# ---------------------------------------------------------------------------
# Settings / credentials
# ---------------------------------------------------------------------------


def test_exchange_accepts_bluefin() -> None:
    s = _settings()
    assert s.exchange == "bluefin"


def test_credentials_check_bluefin_requires_private_key() -> None:
    s = _settings(TRADING_ENABLED=True, BLUEFIN_PRIVATE_KEY="")
    with pytest.raises(ValueError, match="BLUEFIN_PRIVATE_KEY"):
        require_trading_credentials_when_enabled(s)


def test_credentials_check_bluefin_requires_account_address() -> None:
    s = _settings(
        TRADING_ENABLED=True,
        BLUEFIN_PRIVATE_KEY="0x" + "00" * 31 + "01",
        BLUEFIN_ACCOUNT_ADDRESS="",
    )
    with pytest.raises(ValueError, match="BLUEFIN_ACCOUNT_ADDRESS"):
        require_trading_credentials_when_enabled(s)


def test_sanitized_dict_masks_bluefin_privkey() -> None:
    s = _settings(BLUEFIN_PRIVATE_KEY="top-secret-privkey")
    d = s.sanitized_dict()
    assert d.get("BLUEFIN_PRIVATE_KEY") == "***"


# ---------------------------------------------------------------------------
# Adapter construction
# ---------------------------------------------------------------------------


def test_build_adapter_returns_bluefin_client() -> None:
    s = _settings()
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _patch_exchange_info_response()
        client = build_adapter(s)
    from app.exchange.bluefin_client import BluefinClient

    assert isinstance(client, BluefinClient)
    assert isinstance(client, PerpExchangeAdapter)


def test_bluefin_adapter_has_all_protocol_methods() -> None:
    s = _settings()
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _patch_exchange_info_response()
        client = build_adapter(s)
    expected = [
        "symbol_spec",
        "symbol_spec_fetched_ok",
        "has_write_access",
        "fetch_best_bid_ask",
        "fetch_position",
        "fetch_account_snapshot",
        "fetch_open_orders_raw",
        "fetch_recent_fills_raw",
        "place_post_only_limit",
        "cancel_order",
        "cancel_order_by_cloid",
        "query_order_status_by_cloid",
        "market_close",
        "rest_runtime_counters",
        "interpret_place_response",
        "interpret_cancel_response",
        "interpret_order_status_response",
        "make_client_order_id",
    ]
    for name in expected:
        assert hasattr(client, name), f"BluefinClient missing {name}"


def test_bluefin_adapter_loads_ids_id_from_contracts_config() -> None:
    s = _settings()
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _patch_exchange_info_response()
        client = build_adapter(s)
    assert client.ids_id() == _DUMMY_HEX64


def test_venue_account_address_picks_bluefin_field() -> None:
    s = _settings(BLUEFIN_ACCOUNT_ADDRESS=_DUMMY_HEX64)
    assert venue_account_address(s) == _DUMMY_HEX64


def test_build_public_stream_selects_bluefin() -> None:
    s = _settings()
    stream = build_public_stream(s, _state(s), "SUI-PERP", on_bbo=lambda _bb: None)
    from app.exchange.bluefin_public_ws import BluefinPublicStream

    assert isinstance(stream, BluefinPublicStream)


def test_build_private_stream_selects_bluefin() -> None:
    s = _settings()
    q: queue.Queue = queue.Queue(maxsize=16)
    stream = build_private_stream(s, _state(s), _DUMMY_HEX64, q, on_queue_drop=None)
    from app.exchange.bluefin_ws import BluefinPrivateStream

    assert isinstance(stream, BluefinPrivateStream)


# ---------------------------------------------------------------------------
# Wire-format interpretation
# ---------------------------------------------------------------------------


def test_interpret_place_response_accepted() -> None:
    # pro-sdk CreateOrderResponse: {"orderHash": "0x..."}.
    resp = {"orderHash": "0x" + "aa" * 32}
    oid, outcome, reason = interpret_bluefin_place_response(resp)
    assert outcome == "accepted"
    assert isinstance(oid, int) and oid > 0
    assert reason == ""


def test_interpret_place_response_http_error() -> None:
    resp = {"code": 400, "message": "bad signature"}
    oid, outcome, _reason = interpret_bluefin_place_response(resp)
    assert outcome == "exchange_rejected"
    assert oid is None


def test_interpret_place_response_missing_hash_on_2xx() -> None:
    resp = {"_http_status": 202}
    oid, outcome, reason = interpret_bluefin_place_response(resp)
    assert outcome == "unconfirmed"
    assert oid is None
    assert "missing_order_hash" in reason


def test_interpret_cancel_response_success() -> None:
    # pro-sdk cancel returns 202 with empty body.
    resp = {"_http_status": 202, "orderHashes": ["0x" + "aa" * 32]}
    kind, _detail = interpret_bluefin_cancel_response(resp)
    assert kind == "success"


def test_interpret_cancel_response_not_found_benign() -> None:
    resp = {"code": 404, "message": "order not found"}
    kind, _detail = interpret_bluefin_cancel_response(resp)
    assert kind == "benign_missing"


def test_interpret_order_status_response_open() -> None:
    resp = {"data": {"orderHash": "0x" + "bb" * 32, "status": "OPEN"}}
    oid, outcome, _ = interpret_bluefin_order_status_response(resp)
    assert outcome == "open"
    assert isinstance(oid, int) and oid > 0


def test_interpret_order_status_response_filled() -> None:
    resp = {"data": {"orderHash": "0x" + "cc" * 32, "status": "FILLED"}}
    oid, outcome, _ = interpret_bluefin_order_status_response(resp)
    assert outcome == "filled"


def test_interpret_order_status_response_canceled_preserves_reason() -> None:
    resp = {
        "data": {
            "orderHash": "0x" + "dd" * 32,
            "status": "CANCELLED",
            "cancellationReason": "USER_CANCELLED",
        }
    }
    oid, outcome, detail = interpret_bluefin_order_status_response(resp)
    assert outcome == "canceled"
    assert "USER_CANCELLED" in detail


# ---------------------------------------------------------------------------
# Deterministic cloid + oid mapping
# ---------------------------------------------------------------------------


def test_make_deterministic_bluefin_client_order_id_is_stable() -> None:
    a = make_deterministic_bluefin_client_order_id(
        "SUI-PERP", Side.BUY, "cycle-1", 0.9712, 10.5
    )
    b = make_deterministic_bluefin_client_order_id(
        "SUI-PERP", Side.BUY, "cycle-1", 0.9712, 10.5
    )
    assert a == b
    assert len(a) == 16


def test_hash_to_oid_is_stable_and_nonzero() -> None:
    h = "ab" * 32
    assert hash_to_oid(h) == hash_to_oid(h)
    assert hash_to_oid(h) > 0
    assert hash_to_oid("ab" * 32) != hash_to_oid("cd" * 32)


# ---------------------------------------------------------------------------
# Order signing (pro-sdk Sui UserSignature format)
# ---------------------------------------------------------------------------


def test_signable_create_order_has_canonical_field_order() -> None:
    """The JSON field order is load-bearing: the server re-serializes with
    Rust's ``serde_json::to_string_pretty`` and the digests must match byte
    for byte.
    """
    payload = signable_create_order(
        symbol="SUI-PERP",
        account_address=_DUMMY_HEX64,
        price_e9="970000000",
        quantity_e9="10000000000",
        leverage_e9="3000000000",
        side="LONG",
        is_isolated=False,
        expires_at_millis=1_700_000_000_000,
        salt="12345",
        ids_id=_DUMMY_HEX64,
        signed_at_millis=1_700_000_000_000,
    )
    keys = list(payload.keys())
    assert keys == [
        "type",
        "ids",
        "account",
        "market",
        "price",
        "quantity",
        "leverage",
        "side",
        "positionType",
        "expiration",
        "salt",
        "signedAt",
    ]
    assert payload["type"] == "Bluefin Pro Order"
    # Must match Rust `PositionType::Display` output (ALL CAPS) — Bluefin's
    # server re-serializes this value via `serde_json::to_string_pretty`
    # and any byte-level divergence turns the request into `401 Invalid
    # Signature`. See pro-sdk/rust/src/signature.rs:165.
    assert payload["positionType"] == "CROSS"


def test_sign_order_payload_deterministic_for_fixed_inputs() -> None:
    from nacl.signing import SigningKey

    sk = SigningKey(parse_ed25519_secret(_DUMMY_PRIVKEY))
    p = BluefinOrderSignPayload(
        symbol="SUI-PERP",
        account_address=_DUMMY_HEX64,
        ids_id=_DUMMY_HEX64,
        price_e9="970000000",
        quantity_e9="10000000000",
        leverage_e9="3000000000",
        side="LONG",
        is_isolated=False,
        expires_at_millis=1_700_000_000_000,
        salt="12345",
        signed_at_millis=1_700_000_000_000,
    )
    sig1 = sign_order_payload(p, signing_key=sk)
    sig2 = sign_order_payload(p, signing_key=sk)
    assert sig1 == sig2
    # Sui UserSignature base64: 1-byte flag + 64-byte sig + 32-byte pubkey
    # = 97 bytes → 132 base64 chars (ceil(97*4/3), padded).
    import base64

    raw = base64.b64decode(sig1)
    assert len(raw) == 97
    assert raw[0] == 0x00  # Ed25519 scheme flag


def test_sign_order_payload_byte_parity_with_pro_sdk_rust() -> None:
    """Regression guard: byte-exact equivalence with pro-sdk Rust reference.

    The expected values below were captured by building and running a
    tiny Rust binary that mirrors
    ``signature::conversion::signable::CreateOrderRequest`` and calls
    ``serde_json::to_string_pretty`` + the Sui PersonalMessage
    digest/sign pipeline on the fixed input below. See
    ``.refgen/src/main.rs`` in the dev worktree.

    Any byte-level drift in the Python serializer (field order,
    separators, position-type casing, intent prefix, ULEB128 length,
    Blake2b digest, or base64 envelope layout) will surface here before
    it hits the live Bluefin server and trips ``401 Invalid Signature``.
    """
    from hashlib import blake2b

    from app.exchange.bluefin_auth import (
        _bcs_vector_u8,
        _serialize_pretty,
        build_signable_order,
    )
    from nacl.signing import SigningKey

    p = BluefinOrderSignPayload(
        symbol="SUI-PERP",
        account_address=_DUMMY_HEX64,
        ids_id=_DUMMY_HEX64,
        price_e9="970000000",
        quantity_e9="10000000000",
        leverage_e9="3000000000",
        side="LONG",
        is_isolated=False,
        expires_at_millis=1_700_000_000_000,
        salt="12345",
        signed_at_millis=1_700_000_000_000,
    )

    signable = build_signable_order(p)
    json_str = _serialize_pretty(signable)
    msg = json_str.encode("utf-8")

    expected_json = (
        "{\n"
        '  "type": "Bluefin Pro Order",\n'
        '  "ids": "0x0101010101010101010101010101010101010101010101010101010101010101",\n'
        '  "account": "0x0101010101010101010101010101010101010101010101010101010101010101",\n'
        '  "market": "SUI-PERP",\n'
        '  "price": "970000000",\n'
        '  "quantity": "10000000000",\n'
        '  "leverage": "3000000000",\n'
        '  "side": "LONG",\n'
        '  "positionType": "CROSS",\n'
        '  "expiration": "1700000000000",\n'
        '  "salt": "12345",\n'
        '  "signedAt": "1700000000000"\n'
        "}"
    )
    assert json_str == expected_json

    intent = b"\x03\x00\x00" + _bcs_vector_u8(msg)
    assert (
        intent.hex()
        == "030000ac03" + msg.hex()
    )

    digest = blake2b(intent, digest_size=32).digest()
    assert (
        digest.hex()
        == "138ce0e7b19ab5d4ddc0e1c11e8eda0cdfbeec149d7211146980fa1bd2420a48"
    )

    sk = SigningKey(parse_ed25519_secret(_DUMMY_PRIVKEY))
    sig_b64 = sign_order_payload(p, signing_key=sk)
    assert sig_b64 == (
        "AFBbgH6lq2uZq71A2X5dJkIZ18Gg9VealZdJMmEJYZtK829VYZqqFUeSTp73"
        "/q7iZrZoNXVp4bfb6sUK5GVf6gpMtav2rXn79au8yvzCadhc0mUe1LiFtYaf"
        "JBrt8KW6KQ=="
    )


def test_sign_login_request_deterministic_and_compact() -> None:
    from nacl.signing import SigningKey

    sk = SigningKey(parse_ed25519_secret(_DUMMY_PRIVKEY))
    payload = signable_login(
        account_address=_DUMMY_HEX64,
        audience="api",
        signed_at_millis=1_700_000_000_000,
    )
    sig1 = sign_login_request(payload, signing_key=sk)
    sig2 = sign_login_request(payload, signing_key=sk)
    assert sig1 == sig2
    import base64

    raw = base64.b64decode(sig1)
    assert len(raw) == 97  # same Sui UserSignature layout


def test_fresh_salt_is_decimal_string() -> None:
    s = fresh_salt()
    assert isinstance(s, str)
    assert s.isdigit()


def test_default_order_expiration_ms_matches_pro_sdk_windows() -> None:
    import time

    now_ms = int(time.time() * 1000)
    market_exp = default_order_expiration_ms(is_market=True)
    limit_exp = default_order_expiration_ms(is_market=False)
    assert 5 * 60_000 <= (market_exp - now_ms) <= 7 * 60_000
    assert 29 * 86_400_000 <= (limit_exp - now_ms) <= 31 * 86_400_000


# ---------------------------------------------------------------------------
# Public WS parser (pro-sdk TickerUpdate)
# ---------------------------------------------------------------------------


def test_public_ws_parses_ticker_update() -> None:
    s = _settings()
    applied: list[float] = []
    from app.exchange.bluefin_public_ws import BluefinPublicStream

    stream = BluefinPublicStream(
        s,
        _state(s),
        "SUI-PERP",
        on_bbo=lambda bb: applied.append(float(bb.mid_price or 0.0)),
    )
    # Pro-sdk TickerUpdate: bestBid/bestAsk in 1e9 base.
    msg = (
        '{"event":"TickerUpdate","payload":{'
        '"symbol":"SUI-PERP",'
        '"bestBidPriceE9":"970000000",'
        '"bestBidQuantityE9":"100000000000",'
        '"bestAskPriceE9":"971000000",'
        '"bestAskQuantityE9":"200000000000",'
        '"updatedAtMillis":1700000000123}}'
    )
    stream.feed_message_for_tests(msg)
    assert len(applied) == 1
    # mid = (0.97 + 0.971) / 2 = 0.9705
    assert abs(applied[0] - 0.9705) < 1e-9


# ---------------------------------------------------------------------------
# Private WS parser (pro-sdk AccountStreamMessage)
# ---------------------------------------------------------------------------


def test_private_ws_parses_account_trade_update() -> None:
    s = _settings()
    q: queue.Queue = queue.Queue(maxsize=16)
    from app.exchange.bluefin_ws import BluefinPrivateStream

    stream = BluefinPrivateStream(s, _DUMMY_HEX64, q)
    # Pro-sdk envelope: {event, reason, payload:{trade:{...Trade...}}}
    msg = (
        '{"event":"AccountTradeUpdate","reason":"OrderMatched","payload":{'
        '"trade":{'
        '"id":"trade-123",'
        '"symbol":"SUI-PERP","side":"LONG",'
        '"orderHash":"0xaa","priceE9":"970000000",'
        '"quantityE9":"10000000000",'
        '"tradingFeeE9":"-100000",'
        '"realizedPnlE9":"0","executedAtMillis":1700000000000,'
        '"isMaker":true,"quoteQuantityE9":"9700000000"'
        '}}}'
    )
    stream.feed_message_for_tests(msg)
    ev = q.get_nowait()
    assert ev.fill_id.startswith("trade-123_")
    assert ev.coin == "SUI-PERP"
    assert abs(ev.px - 0.97) < 1e-9
    assert abs(ev.sz - 10.0) < 1e-9
    # Fee comes through as abs USDC (1e-4 USDC here) — confirms the
    # zero-fee-promo telemetry channel is intact.
    assert abs(ev.fee - 0.0001) < 1e-12
    # Maker side → crossed=False.
    assert ev.crossed is False


def test_private_ws_parses_account_order_update_active() -> None:
    s = _settings()
    q: queue.Queue = queue.Queue(maxsize=16)
    from app.exchange.bluefin_ws import BluefinPrivateStream

    stream = BluefinPrivateStream(s, _DUMMY_HEX64, q)
    msg = (
        '{"event":"AccountOrderUpdate","reason":"OrderCreated","payload":{'
        '"orderHash":"0xbb","symbol":"SUI-PERP","status":"OPEN","side":"SHORT",'
        '"priceE9":"970000000","quantityE9":"10000000000",'
        '"filledQuantityE9":"0","updatedAtMillis":1700000000000}}'
    )
    stream.feed_message_for_tests(msg)
    ev = q.get_nowait()
    assert ev.status == "OPEN"
    assert ev.side == "A"  # SHORT → A(sk)
    assert ev.orig_sz == 10.0
    assert ev.remaining_sz == 10.0


def test_private_ws_parses_account_order_update_cancellation() -> None:
    s = _settings()
    q: queue.Queue = queue.Queue(maxsize=16)
    from app.exchange.bluefin_ws import BluefinPrivateStream

    stream = BluefinPrivateStream(s, _DUMMY_HEX64, q)
    # OrderCancellationUpdate payload shape (no status / quantity fields).
    msg = (
        '{"event":"AccountOrderUpdate","reason":"OrderCancelled","payload":{'
        '"orderHash":"0xcc","symbol":"SUI-PERP",'
        '"accountAddress":"0xabc",'
        '"createdAtMillis":1700000000000,'
        '"cancellationReason":"USER_CANCELLED",'
        '"remainingQuantityE9":"10000000000"}}'
    )
    stream.feed_message_for_tests(msg)
    ev = q.get_nowait()
    assert ev.status == "CANCELLED"
    assert ev.raw_status == "USER_CANCELLED"


# ---------------------------------------------------------------------------
# Stream lifecycle safety (start/stop without real WS)
# ---------------------------------------------------------------------------


def test_bluefin_public_stream_disabled_is_noop() -> None:
    s = _settings(PUBLIC_WS_ENABLED=False)
    from app.exchange.bluefin_public_ws import BluefinPublicStream

    stream = BluefinPublicStream(s, _state(s), "SUI-PERP", on_bbo=lambda _bb: None)
    stream.start()
    stream.stop()


def test_bluefin_private_stream_no_address_is_noop() -> None:
    s = _settings()
    q: queue.Queue = queue.Queue(maxsize=16)
    from app.exchange.bluefin_ws import BluefinPrivateStream

    stream = BluefinPrivateStream(s, "", q)
    stream.start()  # must not raise
    stream.stop()


# ---------------------------------------------------------------------------
# Cancel-confirmation gating (async-cancel race-condition fix, 2026-04-23)
#
# Bluefin Pro's PUT /api/v1/trade/orders/cancel returns 202 Accepted but
# the matching-engine effect is asynchronous. The adapter now tracks
# in-flight cancels via ``_pending_cancels`` and exposes
# ``has_pending_cancel(symbol, side)`` so the execution layer can skip
# same-side replacement placement while a cancel is pending. Confirmation
# arrives via the private-WS AccountOrderUpdate/OrderCancellationUpdate
# event. A per-entry timeout (``BLUEFIN_CANCEL_CONFIRM_TIMEOUT_SECONDS``)
# prevents a dropped WS frame from blocking quoting indefinitely.
# ---------------------------------------------------------------------------


def _bluefin_client_with_mocked_rest():
    """Build a BluefinClient whose REST calls are mocked out.

    Returns (client, request_mock). The mock is installed so ``_request``
    returns the exchange-info response at construction time and can be
    reconfigured by individual tests before issuing cancel calls.
    """
    from app.exchange.bluefin_client import BluefinClient

    s = _settings()
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _patch_exchange_info_response()
        client = BluefinClient(s)
    # Replace the mock with a fresh one on the instance for test-time calls;
    # the constructor-time exchange-info call has already completed.
    client._request = lambda *args, **kwargs: {"_http_status": 202}  # type: ignore[method-assign]
    return client


def _register_pending_cancel_direct(
    client, *, order_hash: str, symbol: str = "SUI-PERP", side: Side = Side.BUY
) -> None:
    """Register a pending cancel without routing through the REST layer.

    The full path (cancel_order → _cancel_by_hashes) exercises _request
    and _retry; the unit tests here only need the gating state, so we
    call the internal helper directly. This keeps the tests focused on
    the state machine rather than HTTP plumbing.
    """
    client._register_pending_cancels(symbol, [order_hash], side_hint=side)


def test_bluefin_cancel_registers_pending_and_gate_blocks_same_side() -> None:
    client = _bluefin_client_with_mocked_rest()
    h = "0x" + "aa" * 32
    _register_pending_cancel_direct(client, order_hash=h, side=Side.BUY)

    # has_pending_cancel reports True for the matching (symbol, side) and
    # False for the opposite side. Symbol normalisation accepts the legacy
    # underscore form as well.
    assert client.has_pending_cancel("SUI-PERP", Side.BUY) is True
    assert client.has_pending_cancel("SUI_PERP", Side.BUY) is True
    assert client.has_pending_cancel("SUI-PERP", Side.SELL) is False

    # Snapshot carries the registered entry so operators/tests can inspect.
    snap = client.pending_cancel_snapshot()
    assert len(snap) == 1
    assert snap[0]["order_hash"] == h
    assert snap[0]["side_value"] == Side.BUY.value


def test_bluefin_on_cancel_confirmed_clears_pending_entry() -> None:
    client = _bluefin_client_with_mocked_rest()
    h = "0x" + "bb" * 32
    _register_pending_cancel_direct(client, order_hash=h, side=Side.SELL)
    assert client.has_pending_cancel("SUI-PERP", Side.SELL) is True

    # Simulate the private-WS OrderCancellationUpdate callback firing.
    client.on_cancel_confirmed(h)
    assert client.has_pending_cancel("SUI-PERP", Side.SELL) is False

    # Idempotent: a second confirmation (or a confirmation for an unknown
    # hash) must not raise and must leave state empty.
    client.on_cancel_confirmed(h)
    client.on_cancel_confirmed("0x" + "cd" * 32)
    assert client.pending_cancel_snapshot() == []

    # The WS-level parser path also fires the callback. Verify end-to-end
    # using ``feed_message_for_tests`` on a synthetic cancellation event.
    _register_pending_cancel_direct(client, order_hash=h, side=Side.SELL)
    assert client.has_pending_cancel("SUI-PERP", Side.SELL) is True
    from app.exchange.bluefin_ws import BluefinPrivateStream

    q: queue.Queue = queue.Queue(maxsize=16)
    stream = BluefinPrivateStream(
        _settings(),
        _DUMMY_HEX64,
        q,
        on_cancel_confirmed=client.on_cancel_confirmed,
    )
    raw_hash = h[2:]  # the WS payload carries the bare-hex form
    msg = (
        '{"event":"AccountOrderUpdate","reason":"OrderCancelled","payload":{'
        f'"orderHash":"0x{raw_hash}","symbol":"SUI-PERP",'
        '"accountAddress":"0xabc",'
        '"createdAtMillis":1700000000000,'
        '"cancellationReason":"USER_CANCELLED",'
        '"remainingQuantityE9":"10000000000"}}'
    )
    stream.feed_message_for_tests(msg)
    assert client.has_pending_cancel("SUI-PERP", Side.SELL) is False


def test_bluefin_pending_cancel_timeout_clears_entry() -> None:
    client = _bluefin_client_with_mocked_rest()
    h = "0x" + "ee" * 32
    _register_pending_cancel_direct(client, order_hash=h, side=Side.BUY)
    assert client.has_pending_cancel("SUI-PERP", Side.BUY) is True

    # Advance a synthetic clock past the configured timeout. We don't
    # sleep in tests — ``expire_stale_pending_cancels`` accepts an
    # explicit ``now`` parameter. The default timeout is 3.0s; we jump
    # forward by 10s to be sure.
    import time as _time

    future = _time.time() + 10.0
    dropped = client.expire_stale_pending_cancels(now=future)
    assert dropped == 1
    assert client.has_pending_cancel("SUI-PERP", Side.BUY) is False


def test_bluefin_cancel_gate_disabled_flag_bypasses_pending_state() -> None:
    """With ``BLUEFIN_CANCEL_CONFIRM_GATE_ENABLED=false`` the gate is off.

    Operator escape hatch: the adapter still tracks pending cancels (they
    clear on WS confirm or on timeout as usual) but ``has_pending_cancel``
    always reports False so the execution layer falls back to the old
    cancel-then-replace behaviour. Used to rule out the gate as a cause
    of unexpected quoting stalls.
    """
    from app.exchange.bluefin_client import BluefinClient

    s = _settings(BLUEFIN_CANCEL_CONFIRM_GATE_ENABLED=False)
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _patch_exchange_info_response()
        client = BluefinClient(s)
    client._request = lambda *args, **kwargs: {"_http_status": 202}  # type: ignore[method-assign]
    h = "0x" + "ff" * 32
    # Registration itself is a no-op when the gate is disabled (we never
    # reach _register_pending_cancels from _cancel_by_hashes). Even if a
    # direct registration were to occur, has_pending_cancel must still
    # return False.
    _register_pending_cancel_direct(client, order_hash=h, side=Side.BUY)
    assert client.has_pending_cancel("SUI-PERP", Side.BUY) is False
    assert client.has_pending_cancel("SUI-PERP", Side.SELL) is False


# =====================================================================
# Rate-limit defense tests — BLUEFIN_MIN_PLACE_INTERVAL_SECONDS +
# BLUEFIN_REST_429_* (see app/config.py and app/exchange/bluefin_client.py).
# Guards against regression in the throttle / 429-retry logic added in
# response to the 2026-04-23 CloudFront rate-limit incident where 98% of
# POST /orders returned HTTP 429.
# =====================================================================


class _MockHttpResp:
    """Minimal httpx.Response stand-in for bluefin_client._request tests.

    Only the attributes/methods the adapter touches: ``status_code``,
    ``headers``, ``content``, ``text``, and ``json()``.
    """

    def __init__(
        self,
        status_code: int,
        body_json: Any = None,
        headers: dict | None = None,
    ) -> None:
        self.status_code = status_code
        self._json = body_json if body_json is not None else {}
        self.text = str(self._json)
        self.content = self.text.encode("utf-8")
        self.headers = headers or {}

    def json(self) -> Any:
        return self._json


def _bluefin_client_with_mocked_httpx():
    """Build a ``BluefinClient`` with the REAL ``_request`` method intact but
    its underlying ``httpx.Client`` swapped for a ``MagicMock``.

    Construction-time ``_request`` is patched (so the /v1/exchange/info
    bootstrap returns canned data), then reverted — the returned client
    exercises the real retry / throttle logic when tests call ``_request``.
    """
    from unittest.mock import MagicMock

    from app.exchange.bluefin_client import BluefinClient

    s = _settings()
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _patch_exchange_info_response()
        client = BluefinClient(s)
    mock_http = MagicMock()
    client._http = mock_http
    return client, mock_http


def test_bluefin_throttle_sleeps_when_within_min_interval() -> None:
    """Second place-call within the min interval must sleep the remainder."""
    import time as _t

    client, _ = _bluefin_client_with_mocked_httpx()
    client._min_place_interval_s = 0.1
    client._last_place_order_mono = _t.monotonic()
    client._place_throttle_waits_total = 0
    client._place_throttle_wait_seconds_total = 0.0

    start = _t.monotonic()
    client._throttle_place_order()
    elapsed = _t.monotonic() - start

    # Sleep approximately the full interval — allow a generous lower
    # bound for scheduling slack on noisy CI.
    assert elapsed >= 0.08, f"expected >=80 ms sleep, got {elapsed*1000:.1f} ms"
    assert client._place_throttle_waits_total == 1
    assert client._place_throttle_wait_seconds_total > 0.0


def test_bluefin_throttle_no_sleep_when_spaced_out() -> None:
    """If the last place call is older than min_interval, throttle is a no-op."""
    import time as _t

    client, _ = _bluefin_client_with_mocked_httpx()
    client._min_place_interval_s = 0.1
    client._last_place_order_mono = _t.monotonic() - 1.0
    client._place_throttle_waits_total = 0

    start = _t.monotonic()
    client._throttle_place_order()
    elapsed = _t.monotonic() - start

    assert elapsed < 0.02, f"expected ~0 ms sleep, got {elapsed*1000:.1f} ms"
    assert client._place_throttle_waits_total == 0


def test_bluefin_throttle_disabled_with_zero_interval() -> None:
    """``BLUEFIN_MIN_PLACE_INTERVAL_SECONDS=0`` disables the throttle entirely."""
    import time as _t

    client, _ = _bluefin_client_with_mocked_httpx()
    client._min_place_interval_s = 0.0
    client._last_place_order_mono = _t.monotonic()

    start = _t.monotonic()
    client._throttle_place_order()
    elapsed = _t.monotonic() - start

    assert elapsed < 0.01
    assert client._place_throttle_waits_total == 0


def test_bluefin_request_retries_on_429_then_succeeds() -> None:
    """_request retries on HTTP 429 up to max_retries, returns 2xx body on success."""
    client, http = _bluefin_client_with_mocked_httpx()
    client._rest_429_base_backoff_s = 0.001  # fast for test
    client._rest_429_max_retries = 3

    http.request.side_effect = [
        _MockHttpResp(429, {"message": "rate limited"}, headers={"Retry-After": "0.002"}),
        _MockHttpResp(429, {"message": "rate limited"}),
        _MockHttpResp(200, {"orderHash": "0xabc"}),
    ]

    result = client._request(
        "create_order",
        "POST",
        "https://trade.api.sui-prod.bluefin.io",
        "/api/v1/trade/orders",
        json_body={},
    )
    assert result.get("orderHash") == "0xabc"
    assert http.request.call_count == 3
    # Two retries fired before the third attempt succeeded.
    assert client._rest_429_retries_total == 2


def test_bluefin_request_gives_up_after_max_429_retries() -> None:
    """After exhausting retries on sustained 429s, the 429 body is returned."""
    client, http = _bluefin_client_with_mocked_httpx()
    client._rest_429_base_backoff_s = 0.001
    client._rest_429_max_retries = 2

    http.request.side_effect = [
        _MockHttpResp(429, {"message": "rate limited"}),
        _MockHttpResp(429, {"message": "rate limited"}),
        _MockHttpResp(429, {"message": "rate limited"}),
    ]

    result = client._request(
        "create_order",
        "POST",
        "https://trade.api.sui-prod.bluefin.io",
        "/api/v1/trade/orders",
        json_body={},
    )
    assert result.get("code") == 429
    # Initial attempt + 2 retries.
    assert http.request.call_count == 3
    assert client._rest_429_retries_total == 2


def test_bluefin_request_caps_retry_wait_at_10_seconds() -> None:
    """An absurd Retry-After value from the server is clamped to 10s.

    Protects the quote loop from a server-side bug or adversarial
    Retry-After that would otherwise block placement for an hour.
    """
    client, http = _bluefin_client_with_mocked_httpx()
    client._rest_429_base_backoff_s = 0.5
    client._rest_429_max_retries = 1

    http.request.side_effect = [
        _MockHttpResp(429, {"message": "rate limited"}, headers={"Retry-After": "3600"}),
        _MockHttpResp(200, {"orderHash": "0xabc"}),
    ]

    with patch("app.exchange.bluefin_client.time.sleep") as sleep_mock:
        result = client._request(
            "create_order",
            "POST",
            "https://trade.api.sui-prod.bluefin.io",
            "/api/v1/trade/orders",
        )

    assert result.get("orderHash") == "0xabc"
    # Sleep must have been called at least once (for the retry) and
    # every call must be <= 10.0s.
    assert sleep_mock.call_count >= 1
    for call in sleep_mock.call_args_list:
        wait = call[0][0]
        assert wait <= 10.0, f"sleep called with {wait}s; 10s cap breached"


def test_bluefin_request_passes_through_non_429_errors() -> None:
    """400 / 500 responses do NOT trigger the 429-retry path."""
    client, http = _bluefin_client_with_mocked_httpx()
    client._rest_429_max_retries = 3

    http.request.side_effect = [
        _MockHttpResp(400, {"message": "bad request"}),
    ]

    result = client._request(
        "op",
        "POST",
        "https://trade.api.sui-prod.bluefin.io",
        "/api/v1/trade/orders",
    )
    assert result.get("code") == 400
    assert http.request.call_count == 1
    assert client._rest_429_retries_total == 0


def test_bluefin_rest_runtime_counters_surface_rate_limit_metrics() -> None:
    """New counters (retries, throttle waits) appear in ``rest_runtime_counters``.

    Operators read these via /state/current to spot rate-limit episodes
    without having to grep logs.
    """
    client, _ = _bluefin_client_with_mocked_httpx()
    client._rest_429_retries_total = 5
    client._place_throttle_waits_total = 3
    client._place_throttle_wait_seconds_total = 0.42

    counters = client.rest_runtime_counters()
    assert counters.get("bluefin_rest_429_retries_total") == 5
    assert counters.get("bluefin_place_throttle_waits_total") == 3
    # wait-seconds -> ms conversion (0.42s -> 420ms)
    assert counters.get("bluefin_place_throttle_wait_ms_total") == 420


# =====================================================================
# Residual-order audit log test — asserts that _sync_open_orders_impl
# emits the audit event when local vs server state disagree.
# =====================================================================


def test_residual_order_audit_log_fires_on_server_extra_order(caplog) -> None:
    """``_sync_open_orders_impl`` emits ``residual_order_audit`` when the server
    holds an order the bot's local state doesn't know about (orphan residual).

    The existing reconcile machinery then cancels the extra; this test covers
    only the telemetry emission (belt-and-suspenders visibility for the
    cancel-confirmation gating added 2026-04-23).
    """
    import logging as _logging
    import os
    import tempfile
    from pathlib import Path

    from app.enums import Side as _Side
    from app.execution import OrderManager
    from app.exchange.hyperliquid_types import HLOpenOrderRaw
    from app.state import BotState
    from app.storage import Storage
    from tests.exchange_client_mocks import mock_mm_client
    from tests.settings_helpers import UnitTestSettings

    # Construct an OrderManager via the same pattern other
    # execution-layer tests use (see tests/test_desync_lifecycle.py).
    # BotState and Storage need to be real because _sync_open_orders_impl
    # reaches into them for locking / side-unresolved state.
    db_path = Path(tempfile.gettempdir()) / f"mm_residual_audit_{os.getpid()}.db"
    db_path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
            "SYMBOL": "SUI-PERP",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    # Server holds two orders; local state has none. Delta is +1 on both
    # sides, which should trigger the audit log.
    client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(1, "SUI-PERP", _Side.BUY, 0.93, 10.0, 1700000000000),
        HLOpenOrderRaw(2, "SUI-PERP", _Side.SELL, 0.94, 10.0, 1700000000000),
    ]
    om = OrderManager(s, client, storage, state)

    with caplog.at_level(_logging.INFO, logger="app.execution"):
        om._sync_open_orders_impl()

    matching = [
        r for r in caplog.records
        if "residual_order_audit" in r.getMessage()
    ]
    assert matching, (
        "expected a residual_order_audit log event; "
        f"got messages: {[r.getMessage() for r in caplog.records[:10]]}"
    )


def test_residual_order_audit_log_silent_when_local_and_server_match(caplog) -> None:
    """Happy-path: zero delta, no duplicates — no audit log emitted.

    Prevents noise on every reconcile cycle when there's nothing interesting
    to report. The log only fires when there's actual news.
    """
    import logging as _logging
    import os
    import tempfile
    from pathlib import Path

    from app.execution import OrderManager
    from app.state import BotState
    from app.storage import Storage
    from tests.exchange_client_mocks import mock_mm_client
    from tests.settings_helpers import UnitTestSettings

    db_path = Path(tempfile.gettempdir()) / f"mm_residual_audit_silent_{os.getpid()}.db"
    db_path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
            "SYMBOL": "SUI-PERP",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    # Zero orders on the server, zero tracked locally — perfect agreement.
    client.fetch_open_orders_raw.return_value = []
    om = OrderManager(s, client, storage, state)

    with caplog.at_level(_logging.INFO, logger="app.execution"):
        om._sync_open_orders_impl()

    residual_logs = [
        r for r in caplog.records
        if "residual_order_audit" in r.getMessage()
    ]
    assert not residual_logs, (
        f"expected no residual_order_audit on happy path; got {len(residual_logs)} records"
    )


# ---------------------------------------------------------------------------
# signedAtMillis clock-skew safety backoff (BLUEFIN_SIGNED_AT_BACKOFF_MS).
# Bluefin's server enforces an undocumented upper bound
# (signedAtMillis <= server_now). Without the backoff, hosts with forward
# clock drift (notably un-NTP'd Windows) silently 400 on every placement
# with a misleading "must be no earlier than 1 minute in the past" error.
# Verified empirically 2026-04-24 via scripts/bluefin_place_cancel_probe.py.
# ---------------------------------------------------------------------------


def _captured_place_body(client, monkeypatch) -> dict:
    """Intercept the HTTP body sent by ``_place_order`` without making a
    real request. Returns the captured body dict for assertions.
    """
    captured: dict = {}

    def _fake_request(op, method, base, path, *, json_body=None, auth=False):
        captured.update({
            "op": op, "method": method, "base": base, "path": path,
            "body": json_body, "auth": auth,
        })
        # Return a 202 with an orderHash so _place_order doesn't error.
        return {"_http_status": 202, "orderHash": "0x" + "ab" * 32}

    # Throttle is not what we're testing — disable it for this test.
    client._min_place_interval_s = 0.0
    client._request = _fake_request  # type: ignore[method-assign]
    # Stub out the signing so we don't need a real private key.
    monkeypatch.setattr(
        "app.exchange.bluefin_client.sign_order_payload",
        lambda payload, *, signing_key: "stub-sig",
    )
    # Use the real BluefinSession dataclass so `_require_session` and its
    # expiry-healthy check see the fields they need. ``expires_at_epoch_s=0``
    # means "no expiry tracking" and is accepted by session_expiry_healthy.
    import time as _time
    from app.exchange.bluefin_auth import BluefinSession
    addr_stub = "0x" + "00" * 32
    client._session = BluefinSession(
        signing_key=object(),
        parent_address=addr_stub,
        signing_address=addr_stub,
        one_ct_enabled=False,
        expires_at_epoch_s=_time.time() + 3600,  # safely in the future
    )
    # _require_session may also check auth-token expiry; make it look fresh.
    client._auth_token = "stub-bearer"
    client._auth_token_expiry_epoch = _time.time() + 3600
    # ids_id must be set for placement to proceed.
    client._ids_id = "0x" + "11" * 32
    return captured


def test_bluefin_signed_at_has_configured_backoff(monkeypatch) -> None:
    """``BLUEFIN_SIGNED_AT_BACKOFF_MS`` must be subtracted from
    ``time.time()*1000`` before being used as ``signedAtMillis``.

    Regression guard: without this the Bluefin server enforces
    ``signedAtMillis <= server_now`` and silently 400s every placement
    under forward local clock drift.
    """
    import time as _time

    client, _ = _bluefin_client_with_mocked_httpx()
    client._signed_at_backoff_ms = 2000
    captured = _captured_place_body(client, monkeypatch)

    t_before_ms = int(_time.time() * 1000)
    client._place_order(
        symbol="SUI-PERP",
        is_buy=True,
        sz=1.0,
        limit_px=0.9,
        post_only=True,
        reduce_only=False,
        ioc=False,
        order_type="LIMIT",
        client_order_id=None,
    )
    t_after_ms = int(_time.time() * 1000)

    signed_at = captured["body"]["signedFields"]["signedAtMillis"]
    # Must be in the server's past vs local_now by ~2000ms, with some slack
    # for scheduling / _time.time() granularity.
    assert t_before_ms - 2050 <= signed_at <= t_after_ms - 1950, (
        f"signedAt={signed_at} not within expected backoff window "
        f"[{t_before_ms - 2050}, {t_after_ms - 1950}]"
    )
    # expiresAtMillis is computed independently (not backed off) and stays
    # in the far future — the 30-day default in default_order_expiration_ms.
    expires_at = captured["body"]["signedFields"]["expiresAtMillis"]
    assert expires_at > t_before_ms + (29 * 24 * 3600 * 1000)


def test_bluefin_signed_at_backoff_zero_disables(monkeypatch) -> None:
    """Operator escape hatch: setting the backoff to 0 lets ``signedAtMillis``
    match local ``time.time()`` exactly. Intended for hosts with known
    clock sync (well-managed PaaS, NTP-synced colo) where the backoff
    is unnecessary latency, and for turning the safeguard off once
    Bluefin fixes the upstream bug.
    """
    import time as _time

    client, _ = _bluefin_client_with_mocked_httpx()
    client._signed_at_backoff_ms = 0
    captured = _captured_place_body(client, monkeypatch)

    t_before_ms = int(_time.time() * 1000)
    client._place_order(
        symbol="SUI-PERP", is_buy=True, sz=1.0, limit_px=0.9,
        post_only=True, reduce_only=False, ioc=False,
        order_type="LIMIT", client_order_id=None,
    )
    t_after_ms = int(_time.time() * 1000)

    signed_at = captured["body"]["signedFields"]["signedAtMillis"]
    assert t_before_ms <= signed_at <= t_after_ms, (
        f"expected signedAt ~= now with backoff=0, got {signed_at} vs "
        f"[{t_before_ms}, {t_after_ms}]"
    )


# ---------------------------------------------------------------------------
# cancel-by-hash workaround (BLUEFIN_CANCEL_BY_HASH_WORKAROUND). Bluefin's
# selective cancel-by-hash path returns HTTP 202 but silently drops the
# request as of 2026-04-24. Omitting the orderHashes key triggers the
# documented cancel-all-for-symbol behaviour which works correctly.
# ---------------------------------------------------------------------------


def _capture_cancel_body(client) -> dict:
    """Intercept the HTTP body sent by ``_cancel_by_hashes`` without making
    a real request. Returns the captured body dict for assertions.
    """
    captured: dict = {}

    def _fake_request(op, method, base, path, *, json_body=None, auth=False):
        captured.update({
            "op": op, "method": method, "path": path,
            "body": json_body, "auth": auth,
        })
        return {"_http_status": 202}

    client._request = _fake_request  # type: ignore[method-assign]
    return captured


def test_bluefin_cancel_workaround_omits_orderhashes_by_default() -> None:
    """Default behaviour: ``_cancel_by_hashes`` must send the cancel-all
    shape (``{"symbol": sym}`` with no ``orderHashes`` key) even when the
    caller passed specific hashes. See BLUEFIN_CANCEL_BY_HASH_WORKAROUND
    in app/config.py for the full rationale.
    """
    client = _bluefin_client_with_mocked_rest()
    assert client._cancel_by_hash_workaround_enabled is True
    captured = _capture_cancel_body(client)

    h = "0x" + "aa" * 32
    client._cancel_by_hashes("SUI-PERP", [h], side_hint=Side.BUY)

    assert captured["path"] == "/api/v1/trade/orders/cancel"
    assert captured["body"] == {"symbol": "SUI-PERP"}
    assert "orderHashes" not in captured["body"], (
        "workaround must omit orderHashes to trigger cancel-all-for-symbol"
    )


def test_bluefin_cancel_workaround_disabled_sends_orderhashes() -> None:
    """When ``BLUEFIN_CANCEL_BY_HASH_WORKAROUND=false``, the adapter
    reverts to the spec-documented selective path — useful for the day
    Bluefin fixes the upstream bug and we want to flip the knob off.
    """
    client = _bluefin_client_with_mocked_rest()
    client._cancel_by_hash_workaround_enabled = False
    captured = _capture_cancel_body(client)

    h1 = "0x" + "aa" * 32
    h2 = "0x" + "bb" * 32
    client._cancel_by_hashes("SUI-PERP", [h1, h2], side_hint=Side.BUY)

    assert captured["body"]["symbol"] == "SUI-PERP"
    assert captured["body"]["orderHashes"] == [h1, h2]


def test_bluefin_cancel_workaround_counter_increments() -> None:
    """Counter ``bluefin_cancel_workaround_fanouts_total`` must reflect
    each fanout so operators can see the workaround's footprint in
    rest_runtime_counters.
    """
    client = _bluefin_client_with_mocked_rest()
    _capture_cancel_body(client)
    start = client._cancel_workaround_fanout_total
    client._cancel_by_hashes("SUI-PERP", ["0x" + "aa" * 32], side_hint=Side.BUY)
    client._cancel_by_hashes("SUI-PERP", ["0x" + "bb" * 32], side_hint=Side.SELL)
    assert client._cancel_workaround_fanout_total == start + 2
    counters = client.rest_runtime_counters()
    assert counters["bluefin_cancel_workaround_fanouts_total"] == start + 2


def test_bluefin_cancel_workaround_still_registers_pending_cancel() -> None:
    """The fanout must not bypass the cancel-confirmation gate: the
    pending-cancel entries for the caller's hashes must still be
    registered so the execution layer can block replacement placement
    until WS confirms.
    """
    client = _bluefin_client_with_mocked_rest()
    _capture_cancel_body(client)

    h = "0x" + "aa" * 32
    client._cancel_by_hashes("SUI-PERP", [h], side_hint=Side.BUY)
    assert client.has_pending_cancel("SUI-PERP", Side.BUY) is True
    snap = client.pending_cancel_snapshot()
    assert any(e["order_hash"] == h for e in snap)


# ---------------------------------------------------------------------------
# BUG-000 regression: workaround fanout pre-registers EVERY currently-open
# hash, not just the caller's. Pre-fix, the side-effect cancel WS events
# (cancellation events for hashes the caller never asked to cancel)
# arrived at the bot and were dropped because no _pending_cancels entry
# existed for them — the OrderManager's WorkingOrder lifecycle never
# observed the cancellation, the side stayed "pending acknowledgement",
# the quote loop refused to replace, and eventually the deadlock watchdog
# fired. ~once every few hours of trading on SUI-PERP.
# ---------------------------------------------------------------------------


def test_bluefin_cancel_workaround_preregisters_all_open_hashes() -> None:
    """When the workaround fans out to cancel-all, every currently-open
    hash for the symbol (not just the caller's) must be in
    ``_pending_cancels`` so the side-effect WS cancellation events can
    clear cleanly.
    """
    from app.exchange.hyperliquid_types import HLOpenOrderRaw

    client = _bluefin_client_with_mocked_rest()
    # Open-orders set on the venue: caller asks to cancel one, but two
    # other orders are also live and will be cancelled as a side effect.
    h_caller = "0x" + "aa" * 32
    h_other_buy = "0x" + "bb" * 32
    h_other_sell = "0x" + "cc" * 32

    open_orders = [
        HLOpenOrderRaw(
            oid=int(h_other_buy[2:][:15], 16),
            coin="SUI-PERP",
            side=Side.BUY,
            limit_px=0.94,
            sz=10.0,
            timestamp=0,
        ),
        HLOpenOrderRaw(
            oid=int(h_other_sell[2:][:15], 16),
            coin="SUI-PERP",
            side=Side.SELL,
            limit_px=0.95,
            sz=10.0,
            timestamp=0,
        ),
    ]

    # We patch fetch_open_orders_raw to return the side-effect set, AND
    # patch oid_to_hash so the synthetic oids round-trip back to our
    # synthetic hashes.
    from app.exchange import bluefin_client as _bc

    real_oid_to_hash = _bc.oid_to_hash
    h_by_oid = {oo.oid: h for oo, h in zip(open_orders, [h_other_buy, h_other_sell])}

    def fake_oid_to_hash(oid):
        return h_by_oid.get(int(oid)) or real_oid_to_hash(int(oid))

    client.fetch_open_orders_raw = lambda _addr: open_orders  # type: ignore[assignment]
    _capture_cancel_body(client)

    with patch("app.exchange.bluefin_client.oid_to_hash", fake_oid_to_hash):
        client._cancel_by_hashes("SUI-PERP", [h_caller], side_hint=Side.BUY)

    snap = client.pending_cancel_snapshot()
    snap_hashes = {e["order_hash"] for e in snap}

    # All three: caller + both side-effect hashes.
    assert h_caller in snap_hashes, "caller hash must still be tracked"
    assert h_other_buy in snap_hashes, (
        "side-effect BUY hash must be pre-registered (BUG-000 fix)"
    )
    assert h_other_sell in snap_hashes, (
        "side-effect SELL hash must be pre-registered (BUG-000 fix)"
    )


def test_bluefin_cancel_workaround_preregister_survives_open_orders_fetch_failure() -> None:
    """Defensive: a transient REST failure on the open-orders pre-fetch
    must NOT break the cancel itself. We still fan out to cancel-all and
    register the caller's hash; we just lose the side-effect coverage
    for that one fanout.
    """
    client = _bluefin_client_with_mocked_rest()
    _capture_cancel_body(client)

    def boom(_addr):
        raise RuntimeError("simulated open_orders fetch failure")

    client.fetch_open_orders_raw = boom  # type: ignore[assignment]

    h = "0x" + "aa" * 32
    # Should not raise:
    client._cancel_by_hashes("SUI-PERP", [h], side_hint=Side.BUY)

    snap = client.pending_cancel_snapshot()
    assert any(e["order_hash"] == h for e in snap), (
        "caller's hash must still be registered even when open-orders pre-fetch fails"
    )


def test_bluefin_cancel_orphan_event_logs_warning(caplog) -> None:
    """BUG-000 defensive layer: a cancellation event for a hash the bot
    never registered must produce a ``bluefin_cancel_orphan_event``
    WARNING log line (previously these events were silently dropped,
    which made the BUG-000 deadlock invisible for weeks).
    """
    import logging

    client = _bluefin_client_with_mocked_rest()
    orphan_hash = "0x" + "ee" * 32
    with caplog.at_level(logging.WARNING, logger="app.exchange.bluefin_client"):
        client.on_cancel_confirmed(orphan_hash)
    msgs = [
        rec.getMessage()
        for rec in caplog.records
        if "orphan" in rec.getMessage().lower()
    ]
    assert msgs, "expected at least one bluefin_cancel_orphan_event warning"
    assert orphan_hash in msgs[0]


def test_bluefin_cancel_confirmed_for_registered_hash_does_not_log_orphan(
    caplog,
) -> None:
    """Counter-test: a cancel confirmation for a hash that WAS registered
    (caller path) must NOT produce an orphan log line — it's a normal
    confirmation and should log ``bluefin_cancel_confirmed`` at INFO.
    """
    import logging

    client = _bluefin_client_with_mocked_rest()
    h = "0x" + "ff" * 32
    _register_pending_cancel_direct(client, order_hash=h, side=Side.BUY)
    with caplog.at_level(logging.WARNING, logger="app.exchange.bluefin_client"):
        client.on_cancel_confirmed(h)
    orphan_msgs = [
        rec.getMessage()
        for rec in caplog.records
        if "orphan" in rec.getMessage().lower()
    ]
    assert not orphan_msgs, (
        f"registered hash should not log orphan; got {orphan_msgs!r}"
    )


# ---------------------------------------------------------------------------
# Auth token refresh on expiry. Observed production bug 2026-04-24: bot
# minted token at startup, token expired server-side after ~14 min, and
# 828 subsequent create_order requests all rejected with HTTP 401
# "ExpiredSignature" — zero refresh attempts made. Root cause: the hot
# path gated the refresh on `if not token`, so a cached-but-expired
# token string was reused forever.
# ---------------------------------------------------------------------------


def test_bluefin_request_refreshes_expired_auth_token_before_send() -> None:
    """Every auth'd request must go through ``_ensure_auth_token`` so its
    expiry check can fire. Previously the hot path reused any cached
    token string without checking expiry, which 401'd every subsequent
    request after the server-side token TTL lapsed.
    """
    import time as _time
    client, http = _bluefin_client_with_mocked_httpx()
    # Pretend we have a token but it expired 10 s ago.
    client._auth_token = "stale-token"
    client._auth_token_expiry_epoch = _time.time() - 10.0
    # Stub the auth-mint path so _ensure_auth_token returns a fresh token
    # without actually hitting the network. mint is indirectly triggered
    # via self._request for /auth/v2/token; the mocked http fields the
    # mint call and the create_order call in sequence.
    http.request.side_effect = [
        _MockHttpResp(200, {
            'accessToken': 'fresh-token',
            'accessTokenValidForSeconds': 3600,
        }),
        _MockHttpResp(200, {'orderHash': '0xbeef'}),
    ]
    # Required by _ensure_auth_token: a valid session.
    from app.exchange.bluefin_auth import BluefinSession
    addr = '0x' + '00' * 32
    client._session = BluefinSession(
        signing_key=object(),
        parent_address=addr, signing_address=addr,
        one_ct_enabled=False, expires_at_epoch_s=_time.time() + 3600,
    )
    import unittest.mock
    with unittest.mock.patch(
        'app.exchange.bluefin_client.sign_login_request',
        return_value='sig',
    ):
        result = client._request(
            'create_order', 'POST',
            'https://trade.api.sui-prod.bluefin.io',
            '/api/v1/trade/orders',
            json_body={}, auth=True,
        )
    assert result.get('orderHash') == '0xbeef'
    assert client._auth_token == 'fresh-token', (
        'expected stale token to be replaced by fresh one on pre-send refresh'
    )
    # Two HTTP calls: one mint, one create_order.
    assert http.request.call_count == 2


def test_bluefin_request_retries_on_401_expired_signature() -> None:
    """If the server returns 401 ExpiredSignature despite our expiry
    tracking saying the token is fresh (clock skew between us and the
    server), invalidate the cached token and retry once. Second 401
    means structural auth failure; don't loop forever.
    """
    import time as _time
    client, http = _bluefin_client_with_mocked_httpx()
    # Cache a token with a future expiry — our local check says it's
    # fresh, so _ensure_auth_token on the hot path would normally reuse
    # it. The server disagrees.
    client._auth_token = 'will-get-401'
    client._auth_token_expiry_epoch = _time.time() + 3600
    http.request.side_effect = [
        _MockHttpResp(401, {'message': 'Invalid token ExpiredSignature'}),
        # Auth retry invalidates; _ensure_auth_token re-mints:
        _MockHttpResp(200, {
            'accessToken': 'fresh-after-401',
            'accessTokenValidForSeconds': 3600,
        }),
        # Then the original create_order retries with the new token:
        _MockHttpResp(202, {'orderHash': '0xc0de'}),
    ]
    from app.exchange.bluefin_auth import BluefinSession
    addr = '0x' + '00' * 32
    client._session = BluefinSession(
        signing_key=object(),
        parent_address=addr, signing_address=addr,
        one_ct_enabled=False, expires_at_epoch_s=_time.time() + 3600,
    )
    import unittest.mock
    with unittest.mock.patch(
        'app.exchange.bluefin_client.sign_login_request',
        return_value='sig',
    ):
        result = client._request(
            'create_order', 'POST',
            'https://trade.api.sui-prod.bluefin.io',
            '/api/v1/trade/orders',
            json_body={}, auth=True,
        )
    assert result.get('orderHash') == '0xc0de', result
    assert client._auth_token == 'fresh-after-401'
    # Three HTTP calls: 401 create_order, mint, 202 create_order.
    assert http.request.call_count == 3


def test_bluefin_request_gives_up_after_second_401() -> None:
    """Auth retry fires once. A second 401 on the retried request
    returns the error body to the caller so the quote loop can log the
    structural failure rather than spinning forever.
    """
    import time as _time
    client, http = _bluefin_client_with_mocked_httpx()
    client._auth_token = 'bad-token'
    client._auth_token_expiry_epoch = _time.time() + 3600
    http.request.side_effect = [
        _MockHttpResp(401, {'message': 'Invalid token ExpiredSignature'}),
        _MockHttpResp(200, {
            'accessToken': 'still-bad',
            'accessTokenValidForSeconds': 3600,
        }),
        _MockHttpResp(401, {'message': 'Invalid token ExpiredSignature'}),
    ]
    from app.exchange.bluefin_auth import BluefinSession
    addr = '0x' + '00' * 32
    client._session = BluefinSession(
        signing_key=object(),
        parent_address=addr, signing_address=addr,
        one_ct_enabled=False, expires_at_epoch_s=_time.time() + 3600,
    )
    import unittest.mock
    with unittest.mock.patch(
        'app.exchange.bluefin_client.sign_login_request',
        return_value='sig',
    ):
        result = client._request(
            'create_order', 'POST',
            'https://trade.api.sui-prod.bluefin.io',
            '/api/v1/trade/orders',
            json_body={}, auth=True,
        )
    assert result.get('code') == 401
    assert 'ExpiredSignature' in str(result.get('message', ''))
    # 3 HTTP calls: 401, mint, 401 — no fourth.
    assert http.request.call_count == 3


# ---------------------------------------------------------------------------
# fetch_account_volume_usd (used by Telegram /status 7d/30d cache)
# ---------------------------------------------------------------------------


def test_fetch_account_volume_usd_aggregates_single_page() -> None:
    """One-page response: trades aggregate to (price * qty) volume and
    abs(fee) in fees. priceE9 / quantityE9 / tradingFeeE9 are all 1e9-scaled
    on the wire so the product overflows to 1e18 — divided back via
    ``_from_e9`` (twice) inside the aggregator."""
    client = _bluefin_client_with_mocked_rest()
    # Two trades: 100 SUI @ $1.00 and 50 SUI @ $2.00 → $100 + $100 = $200.
    client._request = lambda *a, **kw: {  # type: ignore[method-assign]
        "data": [
            {
                "symbol": "SUI-PERP",
                "priceE9": str(int(1.0 * 1e9)),
                "quantityE9": str(int(100.0 * 1e9)),
                "tradingFeeE9": str(int(0.05 * 1e9)),
            },
            {
                "symbol": "SUI-PERP",
                "priceE9": str(int(2.0 * 1e9)),
                "quantityE9": str(int(50.0 * 1e9)),
                "tradingFeeE9": str(int(-0.10 * 1e9)),  # negative → fee paid; abs()
            },
        ]
    }
    out = client.fetch_account_volume_usd("SUI-PERP", 0, 1_000_000_000_000)
    assert out["volume_usd"] == pytest.approx(200.0)
    assert out["fees_usd"] == pytest.approx(0.15)
    assert out["trade_count"] == 2
    assert out["pages_fetched"] == 1
    assert out["pagination_truncated"] is False


def test_fetch_account_volume_usd_paginates_when_full_page_returned() -> None:
    """A page at exactly ``limit`` rows triggers a follow-up call.
    Aggregates correctly across pages.
    """
    client = _bluefin_client_with_mocked_rest()
    page_full = [
        {
            "symbol": "SUI-PERP",
            "priceE9": str(int(1.0 * 1e9)),
            "quantityE9": str(int(1.0 * 1e9)),
            "tradingFeeE9": str(int(0.001 * 1e9)),
        }
    ] * 1000  # exactly the page size
    page_short = [
        {
            "symbol": "SUI-PERP",
            "priceE9": str(int(1.0 * 1e9)),
            "quantityE9": str(int(2.0 * 1e9)),
            "tradingFeeE9": str(int(0.002 * 1e9)),
        }
    ]
    pages = [page_full, page_short]
    call_count = {"n": 0}

    def fake_request(*args, **kwargs):
        i = call_count["n"]
        call_count["n"] += 1
        return {"data": pages[i] if i < len(pages) else []}

    client._request = fake_request  # type: ignore[method-assign]
    out = client.fetch_account_volume_usd("SUI-PERP", 0, 1_000_000_000_000)
    # 1000 trades of 1 SUI @ $1 = $1000; one trade of 2 SUI @ $1 = $2 → $1002.
    assert out["volume_usd"] == pytest.approx(1002.0)
    assert out["trade_count"] == 1001
    assert out["pages_fetched"] == 2
    assert out["pagination_truncated"] is False


def test_fetch_account_volume_usd_handles_empty_window() -> None:
    """An inverted or empty window returns zeros without making any REST
    calls — defensive guard for stale operator inputs.
    """
    client = _bluefin_client_with_mocked_rest()
    n_calls = {"n": 0}
    def boom(*a, **kw):
        n_calls["n"] += 1
        raise RuntimeError("should not be called for inverted window")
    client._request = boom  # type: ignore[method-assign]
    out = client.fetch_account_volume_usd("SUI-PERP", 1000, 500)
    assert out["volume_usd"] == 0.0
    assert out["trade_count"] == 0
    assert n_calls["n"] == 0


def test_fetch_account_volume_usd_filters_by_symbol_defensively() -> None:
    """If the server ignores the ``symbol`` query param and returns trades
    on other symbols, the aggregator must filter them out at the row level.
    """
    client = _bluefin_client_with_mocked_rest()
    client._request = lambda *a, **kw: {  # type: ignore[method-assign]
        "data": [
            {
                "symbol": "SUI-PERP",
                "priceE9": str(int(1.0 * 1e9)),
                "quantityE9": str(int(10.0 * 1e9)),
                "tradingFeeE9": "0",
            },
            {
                "symbol": "ETH-PERP",  # different symbol — must be filtered
                "priceE9": str(int(2000.0 * 1e9)),
                "quantityE9": str(int(1.0 * 1e9)),
                "tradingFeeE9": "0",
            },
        ]
    }
    out = client.fetch_account_volume_usd("SUI-PERP", 0, 1_000_000_000_000)
    assert out["volume_usd"] == pytest.approx(10.0)
    assert out["trade_count"] == 1
