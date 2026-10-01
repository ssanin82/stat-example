"""Phase 4a (v1.3.110) regression — OKX trade-via-WS for cancels.

Two layers under test:

1. ``OkxActionWs`` (low-level): login flow, request-id allocation,
   response routing by id, timeout + disconnect failure paths.
   Mocked via a controlled ``FakeWs`` that the daemon thread reads
   from / writes to in place of a real socket.

2. ``OkxClient`` integration: WS-first dispatch for cancel ops,
   automatic HTTP fallback when the WS path raises, transport-mode
   stamping. Mocked via the existing pattern (httpx Clients replaced
   by ``MagicMock``) plus an ``OkxActionWs`` mock injected directly.

We do NOT spin up real WS connections — running tests against
``wss://ws.okx.com`` would be flaky in CI and require live creds.
The unit-test surface verifies the dispatch contract; an integration
smoke (one session at OKX_ACTION_WS_CANCEL_ENABLED=true) is the
human-driven acceptance per the plan.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.exchange import okx_action_ws as ws_module
from app.exchange import okx_client as okx_client_module
from app.exchange.okx_action_ws import (
    OkxActionWs,
    OkxActionWsDisconnected,
    OkxActionWsError,
    OkxActionWsNotConnected,
    OkxActionWsTimeout,
)
from tests.settings_helpers import UnitTestSettings


# --------------------------------------------------------------------------
# FakeWs — drives the daemon thread's recv loop deterministically
# --------------------------------------------------------------------------


class FakeWs:
    """Minimal stand-in for a ``websocket-client`` socket. Tests
    push incoming frames via ``inject_message``; outgoing frames the
    daemon thread sends land in ``sent`` for inspection.

    ``recv`` blocks until a frame is injected, ``close`` was called,
    or the optional timeout elapses. Multiple injected messages are
    drained FIFO."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = threading.Event()
        self._inbox: deque[str] = deque()
        self._cond = threading.Condition()
        self._timeout: Optional[float] = None

    def send(self, payload: str) -> None:
        if self.closed.is_set():
            raise OSError("fake_ws_send_after_close")
        self.sent.append(payload)

    def recv(self) -> str:
        with self._cond:
            while not self._inbox and not self.closed.is_set():
                if not self._cond.wait(timeout=self._timeout or 5.0):
                    # Bounded wait so a buggy test doesn't hang the
                    # whole pytest session — surface a clear error.
                    raise OSError("fake_ws_recv_idle_timeout")
            if self.closed.is_set():
                raise OSError("fake_ws_closed_during_recv")
            return self._inbox.popleft()

    def close(self) -> None:
        self.closed.set()
        with self._cond:
            self._cond.notify_all()

    def settimeout(self, t: Optional[float]) -> None:
        self._timeout = t

    # Test-driver helpers ------------------------------------------------

    def inject_message(self, payload: str) -> None:
        with self._cond:
            self._inbox.append(payload)
            self._cond.notify_all()

    def inject_json(self, obj: dict[str, Any]) -> None:
        self.inject_message(json.dumps(obj))

    def wait_for_send(self, timeout_s: float = 2.0) -> str:
        """Block until the daemon thread has issued at least one send;
        return the most recent send payload."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.sent:
                return self.sent[-1]
            time.sleep(0.005)
        raise AssertionError("daemon thread never called send")


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": True,
        "EXCHANGE": "okx",
        "SYMBOL": "SUI-USDT-SWAP",
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "OKX_API_KEY": "test-key",
        "OKX_API_SECRET": "test-secret",
        "OKX_API_PASSPHRASE": "test-pass",
        "OKX_PRIVATE_WS_URL": "wss://ws.okx.test/ws/v5/private",
        # Tighter timeouts so tests don't wait the production defaults.
        "OKX_ACTION_WS_REQUEST_TIMEOUT_SECONDS": 1.0,
        "OKX_ACTION_WS_RECONNECT_INITIAL_SECONDS": 0.05,
        "OKX_ACTION_WS_RECONNECT_CAP_SECONDS": 0.5,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


@pytest.fixture
def fake_ws() -> FakeWs:
    return FakeWs()


@pytest.fixture
def patched_websocket(fake_ws: FakeWs):
    """Patch ``okx_action_ws.websocket.create_connection`` to return
    our FakeWs. The patch survives the test scope."""
    fake_module = MagicMock()
    fake_module.create_connection.return_value = fake_ws
    with patch.object(ws_module, "websocket", fake_module):
        yield fake_module


def _login_ok(fake_ws: FakeWs) -> None:
    """Helper: drive a successful OKX login handshake from the
    test's perspective. The daemon thread sends the login frame,
    waits for the ack, then proceeds to the recv loop."""
    fake_ws.inject_json({"event": "login", "code": "0"})


def _await_connected(ws: OkxActionWs, timeout_s: float = 2.0) -> None:
    assert ws._connected.wait(timeout=timeout_s), (
        "OkxActionWs did not transition to connected within "
        f"{timeout_s}s — check login handshake"
    )


# --------------------------------------------------------------------------
# Login flow
# --------------------------------------------------------------------------


def test_login_frame_shape_matches_okx_v5_spec(
    fake_ws: FakeWs, patched_websocket
) -> None:
    """The first send on the socket must be a signed login frame with
    ``op=login`` and ``args=[{apiKey, passphrase, timestamp, sign}]``.
    Matches the read-only OKX private WS convention so the same
    signing helper works for both channels."""
    ws = OkxActionWs(_settings())
    ws.start()
    try:
        # Daemon thread sends login synchronously after connect.
        login_payload = fake_ws.wait_for_send(timeout_s=1.5)
        frame = json.loads(login_payload)
        assert frame["op"] == "login"
        assert isinstance(frame["args"], list) and len(frame["args"]) == 1
        creds = frame["args"][0]
        assert creds["apiKey"] == "test-key"
        assert creds["passphrase"] == "test-pass"
        # Timestamp must be Unix epoch seconds as a STRING (OKX spec).
        ts = creds["timestamp"]
        assert isinstance(ts, str) and ts.isdigit()
        # sign is a non-empty base64-ish string
        assert isinstance(creds["sign"], str) and len(creds["sign"]) > 10
        # Drive login success so the daemon doesn't hang
        _login_ok(fake_ws)
        _await_connected(ws)
    finally:
        ws.stop()


def test_login_failure_keeps_disconnected(
    fake_ws: FakeWs, patched_websocket
) -> None:
    """A non-zero login ack code triggers the reconnect loop. The WS
    stays in disconnected state until a fresh attempt succeeds."""
    ws = OkxActionWs(_settings())
    ws.start()
    try:
        fake_ws.wait_for_send()
        fake_ws.inject_json(
            {"event": "login", "code": "60001", "msg": "invalid_sign"}
        )
        # Daemon will fail this attempt and start the backoff. Connected
        # event must never set during the failure window.
        assert not ws._connected.wait(timeout=0.2)
        assert not ws.is_connected()
    finally:
        ws.stop()


# --------------------------------------------------------------------------
# Send + response routing
# --------------------------------------------------------------------------


def test_send_and_await_routes_response_by_id(
    fake_ws: FakeWs, patched_websocket
) -> None:
    """The daemon thread routes incoming frames by ``id`` to the
    correct pending entry. Verifies the frame the caller sends has
    the right shape (op + args) and that the awaiting call returns
    the matching response payload."""
    ws = OkxActionWs(_settings())
    ws.start()
    try:
        fake_ws.wait_for_send()  # login frame
        _login_ok(fake_ws)
        _await_connected(ws)

        # Issue the cancel; the daemon will send and we'll inject the
        # matching response from a helper thread.
        responses = {
            "expected_response": {
                "id": None,  # filled below from observed send
                "op": "cancel-order",
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "ordId": "777",
                        "clOrdId": "",
                        "sCode": "0",
                        "sMsg": "",
                    }
                ],
            }
        }

        def responder():
            # Wait for the cancel send (login is sent[0], cancel is sent[1])
            for _ in range(200):
                if len(fake_ws.sent) >= 2:
                    break
                time.sleep(0.005)
            assert len(fake_ws.sent) >= 2
            outbound = json.loads(fake_ws.sent[-1])
            assert outbound["op"] == "cancel-order"
            assert outbound["args"] == [
                {"instId": "SUI-USDT-SWAP", "ordId": "777"}
            ]
            req_id = outbound["id"]
            # Echo the response with the same id
            responses["expected_response"]["id"] = req_id
            fake_ws.inject_json(responses["expected_response"])

        t = threading.Thread(target=responder, daemon=True)
        t.start()

        resp = ws.send_and_await(
            op="cancel-order",
            args=[{"instId": "SUI-USDT-SWAP", "ordId": "777"}],
            timeout_ms=2000,
        )
        t.join(timeout=2.0)
        assert resp["op"] == "cancel-order"
        assert resp["code"] == "0"
        assert resp["data"][0]["ordId"] == "777"
    finally:
        ws.stop()


def test_send_and_await_unique_ids_per_call(
    fake_ws: FakeWs, patched_websocket
) -> None:
    """Sequential ``send_and_await`` calls must use distinct ids so
    response routing remains unambiguous. Verifies the id allocator
    increments monotonically across calls."""
    ws = OkxActionWs(_settings())
    ws.start()
    try:
        fake_ws.wait_for_send()
        _login_ok(fake_ws)
        _await_connected(ws)

        # Drive 3 cancel responses back to the daemon thread; each
        # carries the matching id observed on the send.
        seen_ids: list[str] = []

        def responder():
            for expected in range(3):
                for _ in range(400):
                    if len(fake_ws.sent) >= 2 + expected:
                        break
                    time.sleep(0.005)
                outbound = json.loads(fake_ws.sent[1 + expected])
                req_id = outbound["id"]
                seen_ids.append(req_id)
                fake_ws.inject_json(
                    {
                        "id": req_id,
                        "op": "cancel-order",
                        "code": "0",
                        "msg": "",
                        "data": [],
                    }
                )

        t = threading.Thread(target=responder, daemon=True)
        t.start()

        for i in range(3):
            ws.send_and_await(
                "cancel-order",
                [{"instId": "SUI-USDT-SWAP", "ordId": str(100 + i)}],
                timeout_ms=2000,
            )

        t.join(timeout=2.0)
        assert len(seen_ids) == 3
        assert len(set(seen_ids)) == 3, (
            f"ids must be unique across calls; got {seen_ids}"
        )
    finally:
        ws.stop()


def test_send_and_await_timeout_raises(
    fake_ws: FakeWs, patched_websocket
) -> None:
    """When the response does not arrive within ``timeout_ms``, the
    call raises ``OkxActionWsTimeout`` and the pending entry is
    cleaned up so it doesn't leak."""
    ws = OkxActionWs(_settings())
    ws.start()
    try:
        fake_ws.wait_for_send()
        _login_ok(fake_ws)
        _await_connected(ws)

        with pytest.raises(OkxActionWsTimeout):
            ws.send_and_await(
                "cancel-order",
                [{"instId": "SUI-USDT-SWAP", "ordId": "1"}],
                timeout_ms=50,
            )
        # Pending registry must be empty post-timeout.
        with ws._pending_lock:
            assert ws._pending == {}, (
                f"timed-out pending entry leaked: {ws._pending!r}"
            )
        # Stats must record the timeout.
        stats = ws.snapshot_stats()
        assert int(stats["okx_ws_action_timeout_count"]) >= 1
    finally:
        ws.stop()


def test_send_and_await_not_connected_raises(
    patched_websocket,
) -> None:
    """Before login completes, ``send_and_await`` must raise
    immediately so the caller can fall back to HTTP without waiting
    on a doomed timeout."""
    # Don't drive the login ack — the daemon will stay pre-connected.
    ws = OkxActionWs(_settings())
    ws.start()
    try:
        with pytest.raises(OkxActionWsNotConnected):
            ws.send_and_await(
                "cancel-order",
                [{"instId": "SUI-USDT-SWAP", "ordId": "1"}],
                timeout_ms=500,
            )
    finally:
        ws.stop()


def test_disconnect_fails_in_flight_requests(
    fake_ws: FakeWs, patched_websocket
) -> None:
    """When the WS drops mid-request, any in-flight pendings must
    fail with ``OkxActionWsDisconnected`` so the caller wakes and
    can fall back to HTTP."""
    ws = OkxActionWs(_settings())
    ws.start()
    try:
        fake_ws.wait_for_send()
        _login_ok(fake_ws)
        _await_connected(ws)

        # Issue a send that will never be answered; close the socket
        # from another thread to trigger the disconnect path.
        def killer():
            for _ in range(200):
                if len(fake_ws.sent) >= 2:
                    break
                time.sleep(0.005)
            # Close the FakeWs — the daemon's recv() returns an OSError,
            # the run loop fails all pendings.
            fake_ws.close()

        threading.Thread(target=killer, daemon=True).start()

        with pytest.raises(OkxActionWsDisconnected):
            ws.send_and_await(
                "cancel-order",
                [{"instId": "SUI-USDT-SWAP", "ordId": "1"}],
                timeout_ms=2000,
            )
    finally:
        ws.stop()


# --------------------------------------------------------------------------
# Rate-limit detection
# --------------------------------------------------------------------------


def test_rate_limit_response_increments_counter(
    fake_ws: FakeWs, patched_websocket
) -> None:
    """Response carrying envelope code=50011 (OKX rate-limit) must
    bump ``okx_ws_action_rate_limit_total`` so the operator dashboard
    surfaces a throttling regime forming."""
    ws = OkxActionWs(_settings())
    ws.start()
    try:
        fake_ws.wait_for_send()
        _login_ok(fake_ws)
        _await_connected(ws)

        def responder():
            for _ in range(200):
                if len(fake_ws.sent) >= 2:
                    break
                time.sleep(0.005)
            req_id = json.loads(fake_ws.sent[-1])["id"]
            fake_ws.inject_json(
                {
                    "id": req_id,
                    "op": "cancel-order",
                    "code": "50011",  # OKX rate-limit
                    "msg": "Too Many Requests",
                    "data": [],
                }
            )

        threading.Thread(target=responder, daemon=True).start()
        resp = ws.send_and_await(
            "cancel-order",
            [{"instId": "SUI-USDT-SWAP", "ordId": "1"}],
            timeout_ms=2000,
        )
        assert resp["code"] == "50011"
        stats = ws.snapshot_stats()
        assert int(stats["okx_ws_action_rate_limit_total"]) >= 1
    finally:
        ws.stop()


# --------------------------------------------------------------------------
# OkxClient integration — WS-first dispatch + HTTP fallback
# --------------------------------------------------------------------------


def _client_settings(**overrides):
    base = {
        "TRADING_ENABLED": True,
        "EXCHANGE": "okx",
        "SYMBOL": "SUI-USDT-SWAP",
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "OKX_API_KEY": "k",
        "OKX_API_SECRET": "s",
        "OKX_API_PASSPHRASE": "p",
        "OKX_ACTION_WS_CANCEL_ENABLED": True,
        "ACTION_WS_ENABLED": True,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_client(settings=None):
    if settings is None:
        settings = _client_settings()
    with patch.object(
        okx_client_module.OkxClient,
        "_bootstrap_symbol_spec",
        return_value=(okx_client_module.FALLBACK_SYMBOL_SPEC, 1.0, None),
    ):
        client = okx_client_module.OkxClient(settings)
    # Replace HTTP clients with mocks for assertion convenience.
    client.close()
    client._http_place = MagicMock(spec=httpx.Client)
    client._http_place.timeout = httpx.Timeout(8.0)
    client._http_cancel = MagicMock(spec=httpx.Client)
    client._http_cancel.timeout = httpx.Timeout(1.5)
    return client


def _ok_cancel_http_response():
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {
        "code": "0",
        "msg": "",
        "data": [{"ordId": "1", "sCode": "0", "sMsg": ""}],
    }
    r.headers = {}
    return r


def test_okx_client_cancel_uses_ws_when_enabled_and_connected() -> None:
    """With the flag ON and the WS coordinator reporting connected,
    cancel_order routes through the WS path; HTTP is NOT touched.
    Transport mode is stamped 'ws'."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = {
            "id": "1",
            "op": "cancel-order",
            "code": "0",
            "msg": "",
            "data": [{"ordId": "42", "sCode": "0", "sMsg": ""}],
        }
        # Inject the mock directly (bypass lazy construction).
        client._action_ws = ws_mock

        resp = client.cancel_order("SUI-USDT-SWAP", oid=42)

        ws_mock.send_and_await.assert_called_once()
        call = ws_mock.send_and_await.call_args
        assert call.kwargs.get("op", call.args[0] if call.args else None) == "cancel-order" or (
            "cancel-order" in str(call)
        )
        # Frame's args list
        args_passed = call.args[1] if len(call.args) >= 2 else call.kwargs.get("args")
        assert args_passed == [{"instId": "SUI-USDT-SWAP", "ordId": "42"}]
        # HTTP path not taken
        assert not client._http_cancel.request.called
        # Transport stamp
        assert client.last_exchange_transport_mode == "ws"
        # Counter incremented
        assert client._cancel_ws_send_count == 1
        assert resp["data"][0]["ordId"] == "42"
    finally:
        client.close()


def test_okx_client_cancel_falls_back_to_http_on_ws_error() -> None:
    """When the WS path raises (timeout / disconnect / not-connected),
    OkxClient falls back to the HTTP path automatically (because
    action_http_fallback_enabled is True by default) and stamps the
    transport mode 'http'."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.side_effect = OkxActionWsTimeout(
            "synthetic timeout"
        )
        client._action_ws = ws_mock
        client._http_cancel.request.return_value = _ok_cancel_http_response()

        client.cancel_order("SUI-USDT-SWAP", oid=99)

        assert ws_mock.send_and_await.called
        assert client._http_cancel.request.called
        assert client.last_exchange_transport_mode == "http"
        assert client._cancel_ws_fallback_count == 1
    finally:
        client.close()


def test_okx_client_cancel_uses_http_when_ws_disconnected() -> None:
    """If the WS coordinator reports NOT connected (login still
    pending, in backoff), OkxClient skips the WS send entirely and
    goes directly to HTTP. No fallback counter increment because no
    WS call was actually attempted."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = False
        client._action_ws = ws_mock
        client._http_cancel.request.return_value = _ok_cancel_http_response()

        client.cancel_order("SUI-USDT-SWAP", oid=99)

        assert not ws_mock.send_and_await.called
        assert client._http_cancel.request.called
        assert client.last_exchange_transport_mode == "http"
        assert client._cancel_ws_fallback_count == 0
        assert client._cancel_ws_send_count == 0
    finally:
        client.close()


def test_okx_client_cancel_uses_http_when_flag_disabled() -> None:
    """``OKX_ACTION_WS_CANCEL_ENABLED=false`` (default) means cancel
    ops bypass the WS layer entirely, even if the coordinator
    happens to be connected. The flag is the staged-release lever."""
    client = _make_client(_client_settings(OKX_ACTION_WS_CANCEL_ENABLED=False))
    try:
        # We don't even allocate the WS — but pretend we did, with a
        # mock that would error if called.
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.side_effect = AssertionError(
            "WS must not be called when flag disabled"
        )
        client._action_ws = ws_mock
        client._http_cancel.request.return_value = _ok_cancel_http_response()

        client.cancel_order("SUI-USDT-SWAP", oid=99)

        assert client._http_cancel.request.called
        assert client.last_exchange_transport_mode == "http"
    finally:
        client.close()


def test_okx_client_cancel_by_cloid_routes_through_ws() -> None:
    """The cloid variant must also use the WS path when enabled; the
    frame's ``args`` carries ``clOrdId`` instead of ``ordId``."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = {
            "id": "1",
            "op": "cancel-order",
            "code": "0",
            "msg": "",
            "data": [{"clOrdId": "cloid-abc", "sCode": "0", "sMsg": ""}],
        }
        client._action_ws = ws_mock

        client.cancel_order_by_cloid("SUI-USDT-SWAP", "cloid-abc")

        args_passed = ws_mock.send_and_await.call_args.args[1]
        assert args_passed == [
            {"instId": "SUI-USDT-SWAP", "clOrdId": "cloid-abc"}
        ]
    finally:
        client.close()


def test_okx_client_batch_cancel_uses_batch_op_on_ws() -> None:
    """``cancel_batch_orders`` must use the ``batch-cancel-orders``
    OKX WS op (NOT the single-cancel op) so multiple refs ride the
    same frame and matching-engine dispatch."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = {
            "id": "1",
            "op": "batch-cancel-orders",
            "code": "0",
            "msg": "",
            "data": [
                {"ordId": "1", "sCode": "0", "sMsg": ""},
                {"ordId": "2", "sCode": "0", "sMsg": ""},
            ],
        }
        client._action_ws = ws_mock

        client.cancel_batch_orders(
            "SUI-USDT-SWAP",
            [
                {"instId": "SUI-USDT-SWAP", "ordId": "1"},
                {"instId": "SUI-USDT-SWAP", "ordId": "2"},
            ],
        )

        call = ws_mock.send_and_await.call_args
        op = call.args[0]
        args_passed = call.args[1]
        assert op == "batch-cancel-orders"
        assert len(args_passed) == 2
    finally:
        client.close()


# ---------------------------------------------------------------------
# 1.3.123 Phase 4a v2 — instIdCode injection on colo trade-WS frames
# ---------------------------------------------------------------------


def test_inst_id_code_stamped_on_cancel_order_ws_args() -> None:
    """When the bootstrap captured a numeric ``instIdCode``, the
    cancel-order WS args carry it alongside ``instId``/``ordId``.
    Required by the OKX colo trade-WS endpoint (sCode 50014 without)."""
    client = _make_client()
    client._inst_id_code = 120850  # SUI-USDT-SWAP per OKX instruments resp
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = {
            "id": "1",
            "op": "cancel-order",
            "code": "0",
            "msg": "",
            "data": [{"ordId": "42", "sCode": "0", "sMsg": ""}],
        }
        client._action_ws = ws_mock

        client.cancel_order("SUI-USDT-SWAP", oid=42)

        args_passed = ws_mock.send_and_await.call_args.args[1]
        assert args_passed == [
            {"instId": "SUI-USDT-SWAP", "ordId": "42", "instIdCode": 120850}
        ]
    finally:
        client.close()


def test_inst_id_code_stamped_on_cancel_by_cloid_ws_args() -> None:
    """Same instIdCode injection on the cloid-cancel path."""
    client = _make_client()
    client._inst_id_code = 120850
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = {
            "id": "1",
            "op": "cancel-order",
            "code": "0",
            "msg": "",
            "data": [{"clOrdId": "cloid-x", "sCode": "0", "sMsg": ""}],
        }
        client._action_ws = ws_mock

        client.cancel_order_by_cloid("SUI-USDT-SWAP", "cloid-x")

        args_passed = ws_mock.send_and_await.call_args.args[1]
        assert args_passed == [
            {
                "instId": "SUI-USDT-SWAP",
                "clOrdId": "cloid-x",
                "instIdCode": 120850,
            }
        ]
    finally:
        client.close()


def test_inst_id_code_stamped_on_batch_cancel_ws_args() -> None:
    """Batch-cancel rides the same decoration: every ref carries
    instIdCode independently."""
    client = _make_client()
    client._inst_id_code = 120850
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = {
            "id": "1",
            "op": "batch-cancel-orders",
            "code": "0",
            "msg": "",
            "data": [
                {"ordId": "1", "sCode": "0", "sMsg": ""},
                {"ordId": "2", "sCode": "0", "sMsg": ""},
            ],
        }
        client._action_ws = ws_mock

        client.cancel_batch_orders(
            "SUI-USDT-SWAP",
            [
                {"instId": "SUI-USDT-SWAP", "ordId": "1"},
                {"instId": "SUI-USDT-SWAP", "ordId": "2"},
            ],
        )

        args_passed = ws_mock.send_and_await.call_args.args[1]
        assert args_passed == [
            {"instId": "SUI-USDT-SWAP", "ordId": "1", "instIdCode": 120850},
            {"instId": "SUI-USDT-SWAP", "ordId": "2", "instIdCode": 120850},
        ]
    finally:
        client.close()


def test_inst_id_code_absent_when_bootstrap_returned_none() -> None:
    """Defensive: if bootstrap didn't capture the code (offline test,
    older instrument), cancel frames omit the field — standard
    (non-colo) endpoints still accept them."""
    client = _make_client()
    # _make_client patches bootstrap with inst_id_code=None
    assert client._inst_id_code is None
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.return_value = {
            "id": "1",
            "op": "cancel-order",
            "code": "0",
            "msg": "",
            "data": [{"ordId": "42", "sCode": "0", "sMsg": ""}],
        }
        client._action_ws = ws_mock

        client.cancel_order("SUI-USDT-SWAP", oid=42)

        args_passed = ws_mock.send_and_await.call_args.args[1]
        assert args_passed == [{"instId": "SUI-USDT-SWAP", "ordId": "42"}]
        assert "instIdCode" not in args_passed[0]
    finally:
        client.close()


def test_okx_client_cancel_no_fallback_when_disabled_raises() -> None:
    """When ``ACTION_HTTP_FALLBACK_ENABLED=false`` AND the WS path
    fails, the exception propagates to the caller — no silent HTTP
    safety net. This is the operator's "I want to see WS failures"
    knob (default is True for safety)."""
    client = _make_client(
        _client_settings(ACTION_HTTP_FALLBACK_ENABLED=False)
    )
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.send_and_await.side_effect = OkxActionWsTimeout("test")
        client._action_ws = ws_mock

        with pytest.raises(OkxActionWsError):
            client.cancel_order("SUI-USDT-SWAP", oid=1)
        # HTTP must NOT have been touched
        assert not client._http_cancel.request.called
    finally:
        client.close()


def test_okx_client_runtime_counters_include_ws_stats() -> None:
    """``rest_runtime_counters`` merges the action-WS coordinator's
    snapshot so the operator sees one unified view."""
    client = _make_client()
    try:
        ws_mock = MagicMock()
        ws_mock.is_connected.return_value = True
        ws_mock.snapshot_stats.return_value = {
            "okx_ws_action_send_count": 5,
            "okx_ws_action_rtt_ms_median": 2.5,
            "okx_ws_action_rate_limit_total": 0,
        }
        client._action_ws = ws_mock

        counters = client.rest_runtime_counters()

        assert counters["okx_ws_action_send_count"] == 5
        # Floats are int-cast in the runtime counters surface.
        assert counters["okx_ws_action_rtt_ms_median"] == 2
        assert counters["okx_ws_action_rate_limit_total"] == 0
        assert counters["okx_ws_cancel_send_count"] == 0
        assert counters["okx_ws_cancel_fallback_count"] == 0
    finally:
        client.close()
