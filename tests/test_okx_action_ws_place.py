"""Phase 4b (v1.3.111) regression — OKX trade-via-WS for places.

Layered on the Phase 4a action-WS infrastructure (already shipped +
unit-tested in :mod:`tests.test_okx_action_ws`). This module focuses on:

1. **Place frame shape** — ``op=order`` with the same body the HTTP
   path uses (``instId / tdMode / side / ordType / sz / clOrdId / px``).
2. **ordId binding via response** — the WS response envelope mirrors
   the HTTP envelope (``data[0].ordId``), so the
   ``interpret_okx_place_response`` interpreter the caller uses for
   Phase 1a sync binding works transparently for both transports.
3. **Rejection routing** — OKX V5 returns place rejections as ``sCode``
   on the data row regardless of transport; ``51604`` (post-only-cross)
   and ``51008`` (insufficient margin) are the common ones. The
   shared interpreter handles them identically.
4. **HTTP fallback** — on any ``OkxActionWsError`` the place
   automatically falls back to HTTP when
   ``action_http_fallback_enabled`` is True (default).
5. **Hard pre-flight cap is transport-agnostic** — the
   notional safety check runs before either transport; refusal raises
   ``RuntimeError`` regardless of WS state.
6. **Flag isolation** — ``OKX_ACTION_WS_PLACE_ENABLED`` is independent
   of the cancel flag; turning one on does not turn the other on.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.exchange import okx_client as okx_client_module
from app.exchange.okx_action_ws import (
    OkxActionWsDisconnected,
    OkxActionWsError,
    OkxActionWsTimeout,
)
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides):
    base = {
        "TRADING_ENABLED": True,
        "EXCHANGE": "okx",
        "SYMBOL": "SUI-USDT-SWAP",
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "OKX_API_KEY": "k",
        "OKX_API_SECRET": "s",
        "OKX_API_PASSPHRASE": "p",
        "OKX_ACTION_WS_PLACE_ENABLED": True,
        "ACTION_WS_ENABLED": True,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_client(settings=None):
    if settings is None:
        settings = _settings()
    with patch.object(
        okx_client_module.OkxClient,
        "_bootstrap_symbol_spec",
        return_value=(okx_client_module.FALLBACK_SYMBOL_SPEC, 1.0, None),
    ):
        client = okx_client_module.OkxClient(settings)
    # Release the real httpx pools allocated during construction.
    client.close()
    client._http_place = MagicMock(spec=httpx.Client)
    client._http_place.timeout = httpx.Timeout(8.0)
    client._http_cancel = MagicMock(spec=httpx.Client)
    client._http_cancel.timeout = httpx.Timeout(1.5)
    return client


def _ws_place_ack(ord_id: str = "42", cloid: str = "cloid-abc"):
    """WS place response envelope shaped per OKX V5 trade-WS docs.
    Identical to the HTTP ``POST /trade/order`` envelope so a single
    interpreter handles both transports."""
    return {
        "id": "1",
        "op": "order",
        "code": "0",
        "msg": "",
        "data": [
            {
                "ordId": ord_id,
                "clOrdId": cloid,
                "tag": "",
                "sCode": "0",
                "sMsg": "",
            }
        ],
    }


def _ws_place_reject(sCode: str, sMsg: str = ""):
    return {
        "id": "1",
        "op": "order",
        "code": "0",
        "msg": "",
        "data": [
            {
                "ordId": "",
                "clOrdId": "cloid-abc",
                "tag": "",
                "sCode": sCode,
                "sMsg": sMsg,
            }
        ],
    }


def _http_place_response(ord_id: str = "42", cloid: str = "cloid-abc"):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "code": "0",
        "msg": "",
        "data": [
            {
                "ordId": ord_id,
                "clOrdId": cloid,
                "tag": "",
                "sCode": "0",
                "sMsg": "",
            }
        ],
    }
    r.headers = {}
    return r


# --------------------------------------------------------------------------
# Frame shape + routing
# --------------------------------------------------------------------------


def test_place_routes_through_ws_with_op_order() -> None:
    """``place_post_only_limit`` must use the OKX V5 trade-WS op
    name ``order`` (NOT ``cancel-order``, NOT ``place-order``)."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = _ws_place_ack()
        client._action_ws = ws_mock

        client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=10.0,
            limit_px=1.0,
            client_order_id="cloid-abc",
        )

        ws_mock.send_and_await.assert_called_once()
        call = ws_mock.send_and_await.call_args
        op = call.args[0]
        assert op == "order", f"WS place op must be 'order', got {op!r}"
        assert not client._http_place.request.called
    finally:
        client.close()


def test_place_frame_args_carry_full_order_body() -> None:
    """The args list passed to ``send_and_await`` is a SINGLE dict
    carrying the exact body shape the HTTP path POSTs to
    ``/trade/order``: instId, tdMode, side, ordType, sz, clOrdId, px.
    Catches a regression where a WS-only field shape would drift
    away from the HTTP shape."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = _ws_place_ack()
        client._action_ws = ws_mock

        client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=False,  # SELL
            sz=25.0,
            limit_px=1.2345,
            client_order_id="cloid-xyz",
        )

        args_passed = ws_mock.send_and_await.call_args.args[1]
        assert isinstance(args_passed, list) and len(args_passed) == 1
        body = args_passed[0]
        assert body["instId"] == "SUI-USDT-SWAP"
        assert body["tdMode"] == "cross"
        assert body["side"] == "sell"
        assert body["ordType"] == "post_only"
        assert body["clOrdId"] == "cloid-xyz"
        # Numbers stringified at the wire boundary (OKX expects strings).
        assert isinstance(body["sz"], str)
        assert isinstance(body["px"], str)
        # No HTTP-only fields slipped in (e.g. headers, query params).
        assert set(body.keys()) <= {
            "instId",
            "tdMode",
            "side",
            "ordType",
            "sz",
            "clOrdId",
            "px",
            "reduceOnly",
        }
    finally:
        client.close()


# --------------------------------------------------------------------------
# Response carries ordId; interpreter sees identical shape vs HTTP
# --------------------------------------------------------------------------


def test_ws_place_response_envelope_matches_http_shape() -> None:
    """The dict returned from ``place_post_only_limit`` (via WS) must
    pass the same ``interpret_okx_place_response`` test the HTTP
    response does — same envelope keys, same ``data[0].ordId`` /
    ``sCode``. This is the contract Phase 1a's sync ordId binding
    relies on."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = _ws_place_ack(
            ord_id="987654", cloid="cloid-A"
        )
        client._action_ws = ws_mock

        resp = client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=5.0,
            limit_px=2.0,
            client_order_id="cloid-A",
        )

        # The interpreter is the production contract used by
        # OrderManager — call it explicitly and confirm it returns
        # the ordId.
        ex_oid, outcome, reason = client.interpret_place_response(resp)
        assert outcome == "accepted", (
            f"WS place response must be interpreted as accepted; got "
            f"outcome={outcome!r} reason={reason!r}"
        )
        assert ex_oid == 987654
    finally:
        client.close()


def test_ws_place_post_only_cross_rejection_routed_through_interpreter() -> None:
    """A WS response carrying ``sCode=51604`` (post-only-would-cross)
    must be interpreted identically to the HTTP path's same sCode —
    so the post-only-cross cooldown engages on WS rejections too.
    Verifies the rejection contract is transport-agnostic."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = _ws_place_reject(
            sCode="51604", sMsg="Order would cross post-only"
        )
        client._action_ws = ws_mock

        resp = client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=5.0,
            limit_px=2.0,
            client_order_id="cloid-cross",
        )

        ex_oid, outcome, reason = client.interpret_place_response(resp)
        assert ex_oid is None
        assert outcome == "exchange_rejected"
        # The interpreter's reason string carries the post-only-cross
        # signal; downstream cooldown logic keys off this.
        assert "post_only" in reason.lower() or "51604" in reason
    finally:
        client.close()


def test_ws_place_insufficient_margin_rejection_routed() -> None:
    """sCode=51008 (insufficient margin) — also a routine rejection
    that the bot's existing logic handles. Same interpreter path."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = _ws_place_reject(
            sCode="51008", sMsg="Insufficient balance"
        )
        client._action_ws = ws_mock

        resp = client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=False,
            sz=1.0,
            limit_px=1.0,
            client_order_id="cloid-im",
        )

        ex_oid, outcome, _ = client.interpret_place_response(resp)
        assert ex_oid is None
        assert outcome == "exchange_rejected"
    finally:
        client.close()


# --------------------------------------------------------------------------
# Transport stamping + HTTP fallback
# --------------------------------------------------------------------------


def test_place_stamps_transport_mode_ws_on_success() -> None:
    """The successful WS place path must stamp
    ``last_exchange_transport_mode = 'ws'`` so execution.py picks it
    up for the per-place transport attribution (read at
    execution.py:4305)."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = _ws_place_ack()
        client._action_ws = ws_mock

        client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=1.0,
            limit_px=1.0,
            client_order_id="cloid-1",
        )

        assert client.last_exchange_transport_mode == "ws"
        assert client._place_ws_send_count == 1
        assert client._place_ws_fallback_count == 0
    finally:
        client.close()


def test_place_falls_back_to_http_on_ws_timeout() -> None:
    """When the WS path raises ``OkxActionWsTimeout``,
    ``place_post_only_limit`` falls through to the HTTP path
    automatically (default ``action_http_fallback_enabled=True``).
    Transport stamp becomes 'http'; fallback counter increments."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.side_effect = OkxActionWsTimeout("synthetic")
        client._action_ws = ws_mock
        client._http_place.request.return_value = _http_place_response()

        client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=1.0,
            limit_px=1.0,
            client_order_id="cloid-1",
        )

        assert ws_mock.send_and_await.called
        assert client._http_place.request.called
        assert client.last_exchange_transport_mode == "http"
        assert client._place_ws_fallback_count == 1
    finally:
        client.close()


def test_place_falls_back_to_http_on_ws_disconnect_mid_request() -> None:
    """Socket drop during an in-flight place → WS layer raises
    ``OkxActionWsDisconnected`` (from ``_fail_all_pending``) →
    caller falls back to HTTP. Same path as timeout but the exception
    differs."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.side_effect = OkxActionWsDisconnected("drop")
        client._action_ws = ws_mock
        client._http_place.request.return_value = _http_place_response()

        client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=1.0,
            limit_px=1.0,
            client_order_id="cloid-1",
        )

        assert client._http_place.request.called
        assert client.last_exchange_transport_mode == "http"
        assert client._place_ws_fallback_count == 1
    finally:
        client.close()


def test_place_uses_http_when_ws_disconnected_no_attempt() -> None:
    """If the WS coordinator reports NOT connected, the place skips
    the WS layer entirely (no send attempted, no fallback counter
    increment because no WS call happened). Avoids paying the
    immediate ``OkxActionWsNotConnected`` exception cost on every
    place during a reconnect storm."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = False
        client._action_ws = ws_mock
        client._http_place.request.return_value = _http_place_response()

        client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=1.0,
            limit_px=1.0,
            client_order_id="cloid-1",
        )

        assert not ws_mock.send_and_await.called
        assert client._http_place.request.called
        assert client.last_exchange_transport_mode == "http"
        # No WS attempt → no fallback counter increment.
        assert client._place_ws_fallback_count == 0
        assert client._place_ws_send_count == 0
    finally:
        client.close()


def test_place_no_http_fallback_when_disabled_propagates_error() -> None:
    """With ``ACTION_HTTP_FALLBACK_ENABLED=false`` AND the WS path
    failing, the exception propagates so the operator can see WS
    failures explicitly. Mirror of the cancel-side test."""
    client = _make_client(
        _settings(ACTION_HTTP_FALLBACK_ENABLED=False)
    )
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.side_effect = OkxActionWsTimeout("test")
        client._action_ws = ws_mock

        with pytest.raises(OkxActionWsError):
            client.place_post_only_limit(
                symbol="SUI-USDT-SWAP",
                is_buy=True,
                sz=1.0,
                limit_px=1.0,
                client_order_id="cloid-1",
            )
        # HTTP must NOT have been touched.
        assert not client._http_place.request.called
    finally:
        client.close()


# --------------------------------------------------------------------------
# Flag isolation between cancel WS and place WS
# --------------------------------------------------------------------------


def test_place_flag_disabled_bypasses_ws_even_when_cancel_flag_on() -> None:
    """``OKX_ACTION_WS_PLACE_ENABLED=false`` keeps places on HTTP even
    when the cancel-WS flag is on. The two flags are independent so
    the operator can validate them sequentially in the staged
    release (cancel WS in Stage 3, place WS in Stage 4)."""
    client = _make_client(
        _settings(
            OKX_ACTION_WS_CANCEL_ENABLED=True,
            OKX_ACTION_WS_PLACE_ENABLED=False,
        )
    )
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.side_effect = AssertionError(
            "place must NOT touch WS when its flag is disabled"
        )
        client._action_ws = ws_mock
        client._http_place.request.return_value = _http_place_response()

        client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=1.0,
            limit_px=1.0,
            client_order_id="cloid-1",
        )

        assert client._http_place.request.called
        assert client.last_exchange_transport_mode == "http"
    finally:
        client.close()


def test_cancel_flag_disabled_does_not_block_place_via_ws() -> None:
    """Symmetric to the previous test: cancel-WS OFF must NOT prevent
    place-WS from working. ``_get_action_ws`` starts the coordinator
    when EITHER flag is on; the per-action flag gates the actual
    send."""
    client = _make_client(
        _settings(
            OKX_ACTION_WS_CANCEL_ENABLED=False,
            OKX_ACTION_WS_PLACE_ENABLED=True,
        )
    )
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = _ws_place_ack()
        client._action_ws = ws_mock

        client.place_post_only_limit(
            symbol="SUI-USDT-SWAP",
            is_buy=True,
            sz=1.0,
            limit_px=1.0,
            client_order_id="cloid-1",
        )

        assert ws_mock.send_and_await.called
        assert client.last_exchange_transport_mode == "ws"
    finally:
        client.close()


# --------------------------------------------------------------------------
# Pre-flight notional cap fires before BOTH transports
# --------------------------------------------------------------------------


def test_pre_flight_notional_cap_fires_before_ws() -> None:
    """The hard notional safety check runs in ``_place_order`` BEFORE
    the WS/HTTP transport dispatch. A cap-exceeding order must raise
    ``RuntimeError`` without touching either transport — guarantees
    the 2026-05-06 SUI-incident-style runaway can't slip through
    just because the WS path is enabled."""
    client = _make_client(
        _settings(
            MAX_ORDER_NOTIONAL_USD=10.0,
            MAX_ORDER_NOTIONAL_HARD_MULTIPLIER=2.0,
        )
    )
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.side_effect = AssertionError(
            "WS must NOT be touched when pre-flight cap fires"
        )
        client._action_ws = ws_mock

        with pytest.raises(RuntimeError, match="HARD_NOTIONAL_CAP"):
            client.place_post_only_limit(
                symbol="SUI-USDT-SWAP",
                is_buy=True,
                sz=1000.0,  # notional = 1000 * 1 = 1000 USD > 20 hard cap
                limit_px=1.0,
                client_order_id="cloid-runaway",
            )

        assert not ws_mock.send_and_await.called
        assert not client._http_place.request.called
    finally:
        client.close()


# --------------------------------------------------------------------------
# Runtime counters surface the WS place stats
# --------------------------------------------------------------------------


def test_runtime_counters_expose_place_ws_send_and_fallback() -> None:
    """``rest_runtime_counters`` must surface the new Phase 4b
    counters alongside the cancel ones, so the operator dashboard
    has a single source of truth for WS transport health."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.snapshot_stats.return_value = {
            "okx_ws_action_send_count": 0,
            "okx_ws_action_rtt_ms_median": 0,
            "okx_ws_action_rate_limit_total": 0,
        }
        client._action_ws = ws_mock

        counters = client.rest_runtime_counters()
        assert "okx_ws_place_send_count" in counters
        assert "okx_ws_place_fallback_count" in counters
        assert counters["okx_ws_place_send_count"] == 0
        assert counters["okx_ws_place_fallback_count"] == 0
    finally:
        client.close()
