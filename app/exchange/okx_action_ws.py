"""OKX V5 trade-via-WebSocket transport for signed actions (cancel + place).

Phase 4a (v1.3.110) — ship the WS infrastructure and the CANCEL wire-up;
places continue on REST until Phase 4b adds the wire-up on top of this
same coordinator.

Design — daemon thread + request/response routing by ``id``:

  1. ``start()`` spawns the WS daemon thread.
  2. The thread connects to ``/ws/v5/private``, sends the signed OKX
     login frame (reuses :func:`app.exchange.okx_ws._okx_login_sign`),
     and waits for the login ack.
  3. After login, the thread enters a recv loop. Each incoming frame
     with an ``id`` field is dispatched to the matching in-flight
     pending response by setting its ``threading.Event`` and result
     slot — the caller in :meth:`send_and_await` wakes and returns.
  4. ``send_and_await(op, args, timeout_ms)`` allocates a fresh id,
     registers a :class:`_PendingResponse`, sends the frame, and blocks
     on the Event until the matching response arrives (or the timeout
     elapses, in which case the caller can fall back to HTTP).
  5. On disconnect / read error, the thread fails all in-flight pending
     responses with ``OkxActionWsDisconnected`` (callers fall back to
     HTTP), closes the socket, sleeps the exponential-backoff interval,
     and reconnects.
  6. ``stop()`` sets the stop event, breaks the recv loop, joins the
     thread.

Why daemon thread + futures vs the simpler HL-style synchronous-lock
pattern? With Phase 3's parallel cancel + place workers, BOTH workers
may call ``send_and_await`` concurrently. The lock pattern would
serialize them (defeating part of the Phase 3 win); the daemon-thread
demuxing lets multiple in-flight requests share one socket.

Failure modes that trigger HTTP fallback (when
``action_http_fallback_enabled=True``):
  * ``OkxActionWsNotConnected`` — WS is mid-reconnect / has never connected
  * ``OkxActionWsSendError`` — socket-level error during send (broken pipe)
  * ``OkxActionWsTimeout`` — response did not arrive within ``timeout_ms``
  * ``OkxActionWsDisconnected`` — socket dropped while waiting for response

The plan reference is ``plans/cxl-optimize.md`` Phase 4a.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

from app.config import Settings
from app.exchange.okx_ws import _okx_login_sign

logger = logging.getLogger(__name__)

try:
    import websocket  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    websocket = None  # type: ignore[assignment]


# OKX V5 rate-limit code on the trade-WS envelope. Same code as the
# REST path uses; surfaced here as a class-level constant so callers
# can match on it without importing the WS module just for the literal.
OKX_RATE_LIMIT_CODE = "50011"


class OkxActionWsError(Exception):
    """Base class for action-WS transport failures."""


class OkxActionWsNotConnected(OkxActionWsError):
    """The daemon thread has not yet completed login, or the socket is
    currently disconnected and waiting on the backoff. Callers should
    fall back to HTTP."""


class OkxActionWsSendError(OkxActionWsError):
    """Low-level socket error during frame send (broken pipe, etc.).
    Callers should fall back to HTTP and let the daemon thread
    reconnect."""


class OkxActionWsTimeout(OkxActionWsError):
    """The response for an in-flight request did not arrive within the
    caller-supplied timeout. Callers should fall back to HTTP."""


class OkxActionWsDisconnected(OkxActionWsError):
    """The WS socket dropped while a request was in flight; the
    pending response was force-failed by the daemon thread. Callers
    should fall back to HTTP."""


class _PendingResponse:
    """One slot per in-flight ``send_and_await`` call. The daemon
    thread sets ``result`` / ``error`` and the event when the matching
    response arrives (or on socket drop)."""

    __slots__ = ("event", "result", "error", "t0_perf")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.result: Optional[dict[str, Any]] = None
        self.error: Optional[Exception] = None
        self.t0_perf = time.perf_counter()


class OkxActionWs:
    """OKX V5 trade-via-WebSocket sender (cancel + place).

    Lifetime: one instance per OkxClient. Construct in the bot's
    bootstrap path; call :meth:`start` to spawn the daemon thread;
    call :meth:`send_and_await` per action; call :meth:`stop` at
    shutdown.

    Thread-safety: ``send_and_await`` is safe to call from multiple
    threads concurrently (Phase 3's cancel + place workers). The
    ``_id_lock`` and ``_pending_lock`` are short-held; the heavy
    blocking happens on the per-call ``Event.wait``.

    Telemetry: :meth:`snapshot_stats` returns the rolling RTT (last
    ``_rtt_window`` samples), last-frame-sent / last-response-received
    timestamps, and the rate-limit counter. Surfaced to the dashboard
    Latency panel via ``okx_ws_action_*`` metric names.
    """

    # OKX recommends a literal "ping" text frame every <30s on the
    # private socket. The read loop also accepts WebSocket protocol-
    # level ping/pong via websocket-client's ping_interval kwarg.
    _PING_INTERVAL_SEC = 25.0
    # Login ack must arrive within this window or the connection is
    # reset and we backoff. OKX typically logs in within ~50ms post-
    # TCP-handshake on colo.
    _LOGIN_TIMEOUT_SEC = 10.0
    # Rolling RTT window for the snapshot. 64 samples is enough to see
    # the median + a tail bump without growing unbounded.
    _RTT_WINDOW = 64

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._ws_url = (settings.okx_private_ws_url or "").strip()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws: Any = None
        self._connected = threading.Event()
        # Request-id allocator. OKX accepts arbitrary string ids; we
        # use a monotonically increasing integer formatted as decimal
        # for compactness on the wire (typical id ~5 chars).
        self._id_lock = threading.Lock()
        self._next_id = 1
        # In-flight registry: req_id (str) -> _PendingResponse. Mutated
        # under ``_pending_lock``; the daemon-thread reader holds the
        # lock only long enough to ``pop`` the entry, so producer
        # threads adding new entries are not blocked by message
        # dispatch.
        self._pending_lock = threading.Lock()
        self._pending: dict[str, _PendingResponse] = {}
        # Telemetry.
        self._stats_lock = threading.Lock()
        self._rtt_samples: deque[float] = deque(maxlen=self._RTT_WINDOW)
        self._last_frame_sent_ms: int = 0
        self._last_response_received_ms: int = 0
        self._send_count: int = 0
        self._timeout_count: int = 0
        self._disconnect_count: int = 0
        # 1.3.110: WS-side rate-limit (sCode 50011 on the envelope).
        # Incremented whenever a response carries the rate-limit code
        # — surfaced to the dashboard so the operator can spot a
        # throttling regime forming.
        self._rate_limit_count: int = 0

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if websocket is None:
            logger.warning(
                "okx_action_ws_disabled websocket-client missing"
            )
            return
        api_key = (self._settings.okx_api_key or "").strip()
        api_secret = (self._settings.okx_api_secret or "").strip()
        passphrase = (self._settings.okx_api_passphrase or "").strip()
        if not (api_key and api_secret and passphrase):
            logger.warning(
                "okx_action_ws_disabled: write credentials missing"
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="okx-action-ws",
            daemon=True,
        )
        self._ping_thread = threading.Thread(
            target=self._ping_worker,
            name="okx-action-ws-ping",
            daemon=True,
        )
        self._thread.start()
        self._ping_thread.start()

    def stop(self) -> None:
        """Stop the daemon thread; fail any in-flight pending requests
        with ``OkxActionWsDisconnected`` so blocked callers wake and
        can fall back to HTTP. Idempotent; safe at process shutdown."""
        self._stop.set()
        self._connected.clear()
        with self._ws_lock:
            ws = self._ws
            self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.debug("okx_action_ws_close_failed", exc_info=True)
        self._fail_all_pending(
            OkxActionWsDisconnected("action_ws_stopped")
        )
        for t in (self._thread, self._ping_thread):
            if t is not None:
                t.join(timeout=5.0)

    def is_connected(self) -> bool:
        """True when the WS is connected AND login has completed.
        Callers use this as a precondition for ``send_and_await`` to
        skip a doomed send and fall straight to HTTP."""
        return self._connected.is_set()

    # ------------------------------------------------------------------
    # Public send + await
    # ------------------------------------------------------------------

    def send_and_await(
        self,
        op: str,
        args: list[dict[str, Any]],
        *,
        timeout_ms: Optional[int] = None,
    ) -> dict[str, Any]:
        """Send one action frame and block until the matching response
        arrives. Returns the OKX response envelope, shape:
            {"id": "<req_id>", "op": "<op>", "code": "0", "msg": "",
             "data": [...]}
        which the caller can pass through the same
        ``interpret_okx_*_response`` helpers the HTTP path uses.

        Raises:
            OkxActionWsNotConnected — login not yet complete / socket
                dropped before send. Caller should HTTP-fallback.
            OkxActionWsSendError — socket-level send failure.
            OkxActionWsTimeout — no response within ``timeout_ms``.
            OkxActionWsDisconnected — socket dropped after send,
                before response.
        """
        if not self._connected.is_set():
            raise OkxActionWsNotConnected("login_pending_or_disconnected")
        if timeout_ms is None:
            timeout_ms = int(
                float(
                    getattr(
                        self._settings,
                        "okx_action_ws_request_timeout_seconds",
                        1.5,
                    )
                )
                * 1000
            )
        # Allocate id + register pending BEFORE sending — the daemon
        # thread could theoretically demux a response before we return
        # from ``ws.send`` on a very fast connection.
        with self._id_lock:
            req_id = str(self._next_id)
            self._next_id += 1
        pending = _PendingResponse()
        with self._pending_lock:
            self._pending[req_id] = pending
        frame = {"id": req_id, "op": op, "args": args}
        try:
            self._send_frame_json(frame)
        except OkxActionWsError:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise
        except Exception as exc:
            with self._pending_lock:
                self._pending.pop(req_id, None)
            raise OkxActionWsSendError(str(exc)) from exc
        with self._stats_lock:
            self._send_count += 1
            self._last_frame_sent_ms = int(time.time() * 1000)
        if not pending.event.wait(timeout=timeout_ms / 1000.0):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            with self._stats_lock:
                self._timeout_count += 1
            raise OkxActionWsTimeout(
                f"req_id={req_id} op={op} timeout_ms={timeout_ms}"
            )
        # Pending entry was already popped by ``_route_message`` on
        # response arrival, OR by ``_fail_all_pending`` on disconnect.
        if pending.error is not None:
            raise pending.error
        assert pending.result is not None  # event set => one of the two slots is filled
        rtt_ms = (time.perf_counter() - pending.t0_perf) * 1000.0
        with self._stats_lock:
            self._rtt_samples.append(rtt_ms)
            self._last_response_received_ms = int(time.time() * 1000)
            # Detect rate-limit on the response envelope OR on data
            # rows (OKX puts the limit code at the envelope level for
            # blanket throttling; per-row sCode 51000 series can also
            # apply but those are validation, not throttling).
            code = str(pending.result.get("code") or "")
            if code == OKX_RATE_LIMIT_CODE:
                self._rate_limit_count += 1
        return pending.result

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def snapshot_stats(self) -> dict[str, float | int]:
        """Compact stats for the Latency / Connectivity dashboard
        panels. Median RTT plus the most recent send/recv timestamps."""
        with self._stats_lock:
            samples = list(self._rtt_samples)
            stats = {
                "okx_ws_action_send_count": int(self._send_count),
                "okx_ws_action_timeout_count": int(self._timeout_count),
                "okx_ws_action_disconnect_count": int(self._disconnect_count),
                "okx_ws_action_rate_limit_total": int(self._rate_limit_count),
                "okx_ws_action_last_frame_sent_ms": int(
                    self._last_frame_sent_ms
                ),
                "okx_ws_action_last_response_received_ms": int(
                    self._last_response_received_ms
                ),
                "okx_ws_action_connected": int(self._connected.is_set()),
            }
        # Compute median outside the lock to keep its critical section
        # tiny; ``samples`` is already a snapshot.
        if samples:
            samples.sort()
            n = len(samples)
            median = (
                samples[n // 2]
                if n % 2 == 1
                else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
            )
            stats["okx_ws_action_rtt_ms_median"] = float(median)
            stats["okx_ws_action_rtt_ms_max"] = float(samples[-1])
        else:
            stats["okx_ws_action_rtt_ms_median"] = 0.0
            stats["okx_ws_action_rtt_ms_max"] = 0.0
        return stats

    # ------------------------------------------------------------------
    # Daemon thread internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Connect → login → recv loop → on disconnect, fail pendings
        and backoff. Repeat until ``_stop`` is set."""
        backoff_initial = float(
            getattr(
                self._settings,
                "okx_action_ws_reconnect_initial_seconds",
                0.5,
            )
        )
        backoff_cap = float(
            getattr(
                self._settings,
                "okx_action_ws_reconnect_cap_seconds",
                30.0,
            )
        )
        attempt = 0
        while not self._stop.is_set():
            try:
                self._connect_and_loop()
                # Successful run reset
                attempt = 0
            except Exception:
                logger.exception("okx_action_ws_loop_exception")
            # Drop any in-flight pendings — the socket is gone.
            self._fail_all_pending(
                OkxActionWsDisconnected("ws_disconnect_pending_failed")
            )
            self._connected.clear()
            if self._stop.is_set():
                break
            attempt += 1
            backoff = min(backoff_cap, backoff_initial * (2 ** (attempt - 1)))
            with self._stats_lock:
                self._disconnect_count += 1
            logger.warning(
                "okx_action_ws_reconnect attempt=%d backoff_s=%.2f",
                attempt,
                backoff,
            )
            # Wake early on stop; otherwise sleep the backoff.
            self._stop.wait(timeout=backoff)

    def _connect_and_loop(self) -> None:
        if websocket is None:  # pragma: no cover
            return
        ws = websocket.create_connection(self._ws_url, timeout=10.0)
        with self._ws_lock:
            self._ws = ws
        try:
            self._send_login_frame(ws)
            self._await_login_ack(ws)
            self._connected.set()
            logger.info("okx_action_ws_connected")
            self._recv_loop(ws)
        finally:
            self._connected.clear()
            with self._ws_lock:
                self._ws = None
            try:
                ws.close()
            except Exception:
                pass

    def _send_login_frame(self, ws: Any) -> None:
        api_key = (self._settings.okx_api_key or "").strip()
        api_secret = (self._settings.okx_api_secret or "").strip()
        passphrase = (self._settings.okx_api_passphrase or "").strip()
        ts = str(int(time.time()))
        sign = _okx_login_sign(api_secret, ts)
        frame = {
            "op": "login",
            "args": [
                {
                    "apiKey": api_key,
                    "passphrase": passphrase,
                    "timestamp": ts,
                    "sign": sign,
                }
            ],
        }
        ws.send(json.dumps(frame))

    def _await_login_ack(self, ws: Any) -> None:
        """Block reads until either the login event arrives (success
        or failure) or the timeout elapses. OKX sends the login ack
        as ``{"event": "login", "code": "0"}`` (code != "0" carries
        the error message in ``msg``)."""
        deadline = time.time() + self._LOGIN_TIMEOUT_SEC
        ws.settimeout(self._LOGIN_TIMEOUT_SEC)
        while time.time() < deadline:
            raw = ws.recv()
            if not raw or raw == "pong":
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("event") == "login":
                code = str(msg.get("code") or "")
                if code == "0":
                    return
                raise OkxActionWsError(
                    f"login_failed code={code} msg={msg.get('msg')!r}"
                )
        raise OkxActionWsError("login_timeout")

    def _recv_loop(self, ws: Any) -> None:
        ws.settimeout(None)
        while not self._stop.is_set():
            try:
                raw = ws.recv()
            except Exception:
                # Socket-level error → break out of recv loop; the
                # outer ``_run`` will reconnect.
                return
            if not raw:
                # Empty frame; OKX uses these on socket close. The
                # next ``recv`` will raise.
                continue
            if raw == "pong":
                # Application-level pong (response to our text "ping").
                # No demux needed.
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("okx_action_ws_unparseable_frame raw=%r", raw[:200])
                continue
            self._route_message(msg)

    def _route_message(self, msg: dict[str, Any]) -> None:
        """Dispatch a parsed incoming frame. Messages with a matching
        ``id`` are routed to the pending entry; everything else
        (login acks observed late, error events, etc.) is logged."""
        if not isinstance(msg, dict):
            return
        req_id = msg.get("id")
        if req_id is None:
            # Event frames: error, pong, etc. The login ack is handled
            # by ``_await_login_ack`` before this loop runs, so any
            # login frame here is a re-emit and benign.
            event = msg.get("event")
            if event == "error":
                logger.warning("okx_action_ws_error_event msg=%s", msg)
            return
        with self._pending_lock:
            pending = self._pending.pop(str(req_id), None)
        if pending is None:
            # Late response after the caller already timed out — benign.
            return
        pending.result = msg
        pending.event.set()

    def _send_frame_json(self, frame: dict[str, Any]) -> None:
        with self._ws_lock:
            ws = self._ws
        if ws is None or not self._connected.is_set():
            raise OkxActionWsNotConnected("ws_not_ready_at_send")
        ws.send(json.dumps(frame))

    def _fail_all_pending(self, exc: Exception) -> None:
        with self._pending_lock:
            pendings = list(self._pending.items())
            self._pending.clear()
        for _req_id, pending in pendings:
            pending.error = exc
            pending.event.set()

    def _ping_worker(self) -> None:
        """OKX requires a text frame "ping" every <30s on the private
        socket. Send opportunistically; failures are silent (the recv
        loop will pick up the broken socket on its next read and
        trigger a reconnect)."""
        while not self._stop.wait(self._PING_INTERVAL_SEC):
            with self._ws_lock:
                ws = self._ws
            if ws is None or not self._connected.is_set():
                continue
            try:
                ws.send("ping")
            except Exception:
                logger.debug("okx_action_ws_ping_send_failed", exc_info=True)
