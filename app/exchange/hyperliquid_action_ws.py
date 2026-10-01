"""Hyperliquid exchange WebSocket `post` transport for signed actions (mirror of HTTP POST /exchange)."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import websocket  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    websocket = None  # type: ignore[assignment]


def normalize_ws_post_action_response(raw_msg: dict[str, Any]) -> dict[str, Any]:
    """
    Map WS `channel=post` payload to the same JSON shape as HTTP POST /exchange returns.

    HTTP: ``{"status":"ok","response":{...}}``
    WS wraps in ``channel/post/data/response`` with optional nested ``type=action/payload``.
    """
    if raw_msg.get("channel") != "post":
        return {"status": "transport_rejected", "response": f"unexpected_channel:{raw_msg.get('channel')!r}"}
    data = raw_msg.get("data")
    if not isinstance(data, dict):
        return {"status": "transport_rejected", "response": "missing_data"}
    inner = data.get("response")
    if isinstance(inner, dict) and inner.get("type") == "action":
        payload = inner.get("payload")
        if isinstance(payload, dict) and payload.get("status") == "ok":
            # Mirror HTTP: top-level status + inner response object
            resp_body = payload.get("response")
            if isinstance(resp_body, dict):
                return {"status": "ok", "response": resp_body}
            return {"status": "ok", "response": resp_body}
        if isinstance(payload, dict):
            return payload
        return {"status": "transport_rejected", "response": str(payload)[:800]}
    if isinstance(inner, dict) and inner.get("status") == "ok":
        return inner
    if isinstance(inner, dict) and "status" in inner:
        return inner
    return {"status": "transport_rejected", "response": str(inner)[:800]}


class HyperliquidExchangeActionWs:
    """
    One persistent WS connection; synchronous send/recv with request id matching.

    Not thread-safe across concurrent sends — the outbound dispatcher serializes calls.
    """

    def __init__(self, ws_url: str, *, timeout_s: float = 15.0) -> None:
        self._ws_url = ws_url
        self._timeout_s = timeout_s
        self._lock = threading.Lock()
        self._ws: Any = None
        self._next_id = 1

    def close(self) -> None:
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:
                    logger.debug("action_ws_close_failed", exc_info=True)
                self._ws = None

    def _ensure_ws(self) -> Any:
        if websocket is None:
            raise RuntimeError("websocket-client not installed")
        if self._ws is not None:
            return self._ws
        self._ws = websocket.create_connection(self._ws_url, timeout=self._timeout_s)
        return self._ws

    def send_signed_exchange_payload(self, exchange_payload: dict[str, Any]) -> dict[str, Any]:
        """
        `exchange_payload` is the JSON body for POST /exchange: action, nonce, signature, vaultAddress, expiresAfter.
        """
        with self._lock:
            return self._send_locked(exchange_payload)

    def _send_locked(self, exchange_payload: dict[str, Any]) -> dict[str, Any]:
        ws = self._ensure_ws()
        req_id = self._next_id
        self._next_id += 1
        envelope = {
            "method": "post",
            "id": req_id,
            "request": {"type": "action", "payload": exchange_payload},
        }
        t0 = time.perf_counter()
        try:
            ws.send(json.dumps(envelope))
        except Exception:
            self._reset_ws()
            raise
        deadline = time.perf_counter() + float(self._timeout_s)
        while time.perf_counter() < deadline:
            try:
                raw = ws.recv()
            except Exception:
                self._reset_ws()
                raise
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("channel") == "pong" or msg.get("method") == "pong":
                continue
            if msg.get("channel") == "post":
                data = msg.get("data")
                if isinstance(data, dict) and data.get("id") == req_id:
                    _ = (time.perf_counter() - t0) * 1000.0
                    return normalize_ws_post_action_response(msg)
            # Ignore subscription noise; keep reading until our id arrives
        self._reset_ws()
        return {"status": "transport_rejected", "response": "action_ws_recv_timeout"}

    def _reset_ws(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
