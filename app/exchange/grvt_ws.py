"""GRVT private websocket stream: fills + order updates."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from typing import Any, Callable, Optional

import httpx

from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
    PrivateWsConnectionEvent,
    PrivateWsConnectionKind,
)
from app.inbound_timing import InboundPrivateTiming

from app.config import Settings
from app.state import BotState

logger = logging.getLogger(__name__)

DropCallback = Optional[Callable[[int], None]]

# Rate-limit how often the keepalive worker logs the same "idle too long"
# warning. Stops one dead session from producing a log-flood while the
# reconnect loop spins back up.
_GRVT_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S = 60.0

try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error = e
else:
    _websocket_import_error = None


def _default_private_ws_url(env: str) -> str:
    env_lc = (env or "prod").strip().lower()
    if env_lc == "prod":
        return "wss://trades.grvt.io/ws/full"
    if env_lc == "testnet":
        return "wss://trades.testnet.grvt.io/ws/full"
    return f"wss://trades.{env_lc}.gravitymarkets.io/ws/full"


def _default_edge_url(env: str) -> str:
    env_lc = (env or "prod").strip().lower()
    if env_lc == "prod":
        return "https://edge.grvt.io"
    if env_lc == "testnet":
        return "https://edge.testnet.grvt.io"
    return f"https://edge.{env_lc}.gravitymarkets.io"


def _normalize_instrument(symbol: str) -> str:
    s = (symbol or "").strip()
    if "_" in s:
        return s
    return f"{s}_USDT_Perp"


AfterPutCallback = Optional[Callable[[str], None]]


def _safe_put(
    q: queue.Queue,
    item: Any,
    drop_label: str,
    on_queue_drop: DropCallback = None,
    on_after_put: AfterPutCallback = None,
) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        logger.error("grvt_private_ws_queue_full dropped=%s", drop_label)
        if on_queue_drop is not None:
            try:
                on_queue_drop(drop_label)
            except Exception:
                logger.exception("grvt_private_ws_queue_drop_callback_failed")
        return
    # Successful enqueue: wake the main loop immediately so it drains this event
    # on the next cycle without waiting for the tick interval. This closes the
    # ~400 ms private_ws_queue_wait_ms latency we were observing — fills and
    # order updates propagate to state.apply_* in sub-millisecond time rather
    # than after the next ``quote_loop_seconds`` timeout.
    if on_after_put is not None:
        try:
            on_after_put(drop_label)
        except Exception:
            logger.exception("grvt_private_ws_on_after_put_failed label=%s", drop_label)


class GrvtPrivateStream:
    """Dedicated thread: connect, subscribe to ``v1.fill`` + ``v1.order``."""

    def __init__(
        self,
        settings: Settings,
        user_address: str,
        out_queue: queue.Queue,
        on_queue_drop: DropCallback = None,
        state: Optional[BotState] = None,
    ) -> None:
        self._settings = settings
        self._addr = (user_address or "").strip()
        self._q = out_queue
        self._on_queue_drop = on_queue_drop
        self._state = state
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None
        self._http = httpx.Client(timeout=8.0)
        self._cookie_gravity = ""
        self._cookie_expiry_epoch = 0.0
        self._grvt_account_id_header = ""
        self._instrument = _normalize_instrument(settings.symbol)
        # Keepalive / idle-reconnect worker. See ``_keepalive_worker``. Mirrors
        # the HL private-WS pattern (``app/exchange/hyperliquid_ws.py``) with
        # one important addition: if no inbound message lands for
        # ``PRIVATE_WS_IDLE_WARN_SECONDS``, the worker force-closes the socket
        # to trigger a fresh subscription. Observed in
        # ``tmp/snap_20260417_173125``: TCP ping/pong kept the connection
        # "alive" for 68+ s after the SELL fill's terminal WS event, but the
        # bot received no further order updates — most importantly it missed
        # the CANCELLED event for a BUY we were trying to kill. A forced
        # reconnect recovers the subscription without relying on the
        # transport layer noticing the silence first.
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._last_idle_log_mono: float = 0.0

    def _after_queue_put(self, label: str) -> None:
        """Wake the main quote loop as soon as a private event lands.

        Mirrors ``app/exchange/hyperliquid_ws.py::_after_queue_put``. Without
        this, fills and order updates sat in the queue for up to
        ``quote_loop_seconds`` (1.5 s default) before the main loop drained
        them — observed in ``snap_20260417_183547`` as
        ``private_ws_queue_wait_ms=404.96``. Waking on enqueue pulls queue
        wait down to sub-millisecond because the main loop blocks on
        ``quote_wake_event.wait(timeout=interval)`` and we set the event
        immediately.
        """
        if self._state is not None and label in ("fill", "order_update"):
            try:
                self._state.wake_quote_loop()
            except Exception:
                logger.exception("grvt_private_ws_wake_failed label=%s", label)

    def _emit_conn(
        self,
        kind: PrivateWsConnectionKind,
        detail: str = "",
        backoff_seconds: float = 0.0,
    ) -> None:
        _safe_put(
            self._q,
            PrivateWsConnectionEvent(kind=kind, detail=detail, backoff_seconds=backoff_seconds),
            "connection",
            self._on_queue_drop,
            self._after_queue_put,
        )

    # ------------------------------------------------------------------
    # Keepalive / idle-reconnect worker
    # ------------------------------------------------------------------

    def _start_keepalive(self) -> None:
        """(Re)start the keepalive/idle-watch thread attached to this session."""
        self._stop_keepalive_worker()
        self._keepalive_stop.clear()
        t = threading.Thread(
            target=self._keepalive_worker,
            name="grvt-private-ws-keepalive",
            daemon=True,
        )
        self._keepalive_thread = t
        t.start()

    def _stop_keepalive_worker(self) -> None:
        self._keepalive_stop.set()
        t = self._keepalive_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.5)
        self._keepalive_thread = None

    def _keepalive_worker(self) -> None:
        """Two-tier idle detection.

        The worker runs every ``PRIVATE_WS_APP_KEEPALIVE_SECONDS`` and
        checks how long the inbound private stream has been silent.
        It has two independent thresholds, with different side-effects:

          *  ``PRIVATE_WS_IDLE_WARN_SECONDS`` (warn tier): emit a rate-
             limited log so ops can see silent windows. **No reconnect.**
             Natural silence (no fills on a slow-trading symbol) should
             cross this threshold without triggering recovery.

          *  ``PRIVATE_WS_IDLE_RECONNECT_SECONDS`` (reconnect tier):
             close the socket so ``_run_forever`` reconnects. This is
             the real "stuck subscription" recovery — first observed in
             ``tmp/snap_20260417_173125`` (68+ s of silence while TCP
             ping/pong was healthy, the bot missed a cancel event).

        Why split them:
          Previously both behaviours shared one threshold. On thin-book
          symbols (AXS, NEAR) genuine fill-free windows of 30–60 s are
          normal, and the single-threshold design mis-read them as a
          stuck subscription and reconnect-looped with exponential
          backoff. The warn tier preserves ops visibility without the
          false-positive reconnects; the reconnect tier keeps the
          stuck-subscription recovery, just at a threshold tuned to the
          symbol's expected fill cadence.

        Both thresholds can be set to ``0`` to disable their respective
        tier; setting both to 0 (or ``PRIVATE_WS_APP_KEEPALIVE_SECONDS``
        to 0) is the way to turn the worker off entirely. HL's private
        WS uses a different keepalive strategy (app-level pings) and is
        unchanged; this two-tier logic is GRVT-only.
        """
        interval = float(self._settings.private_ws_app_keepalive_seconds)
        idle_warn = float(self._settings.private_ws_idle_warn_seconds)
        idle_reconnect = float(self._settings.private_ws_idle_reconnect_seconds)
        if interval <= 0:
            # Worker check-interval disabled → no-op.
            return
        if idle_warn <= 0 and idle_reconnect <= 0:
            # Both tiers disabled → no-op.
            return
        while not self._keepalive_stop.wait(timeout=interval):
            if self._stop.is_set():
                break
            with self._ws_lock:
                ws = self._ws_app
            if ws is None:
                continue
            if self._state is None:
                continue
            with self._state._lock:
                last_msg = self._state.private_ws_last_message_wall_ts
            if last_msg is None:
                continue
            idle_s = max(0.0, (datetime.now(timezone.utc) - last_msg).total_seconds())

            # -------- Warn tier (observability only, no reconnect) --------
            if idle_warn > 0 and idle_s >= idle_warn:
                now_mono = time.monotonic()
                if now_mono - self._last_idle_log_mono >= _GRVT_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S:
                    self._last_idle_log_mono = now_mono
                    logger.warning(
                        "grvt_private_ws_idle_warn idle_s=%.1f "
                        "warn_threshold_s=%.1f reconnect_threshold_s=%.1f",
                        idle_s,
                        idle_warn,
                        idle_reconnect,
                    )

            # -------- Reconnect tier (stuck-subscription recovery) --------
            if idle_reconnect > 0 and idle_s >= idle_reconnect:
                # Preserve the legacy log key so existing log-scraping
                # dashboards keep matching the stuck-subscription event.
                logger.warning(
                    "grvt_private_ws_idle_too_long idle_s=%.1f threshold_s=%.1f "
                    "forcing_reconnect",
                    idle_s,
                    idle_reconnect,
                )
                try:
                    ws.close()
                except Exception:
                    logger.debug("grvt_private_ws_idle_close_failed", exc_info=True)
                # Exit: the next _connect_once will spawn a fresh worker.
                break

    def _refresh_cookie(self) -> None:
        if self._cookie_gravity and (self._cookie_expiry_epoch - time.time() > 5.0):
            return
        api_key = (self._settings.grvt_api_key or "").strip()
        if not api_key:
            raise RuntimeError("GRVT_API_KEY missing for private ws")
        edge_url = self._settings.grvt_edge_url or _default_edge_url(self._settings.grvt_env)
        resp = self._http.post(
            f"{edge_url.rstrip('/')}/auth/api_key/login",
            headers={"Content-Type": "application/json", "Cookie": "rm=true;"},
            json={"api_key": api_key},
        )
        resp.raise_for_status()
        jar = SimpleCookie()
        jar.load(resp.headers.get("set-cookie", ""))
        if "gravity" in jar:
            self._cookie_gravity = jar["gravity"].value
            expires_s = jar["gravity"].get("expires", "")
            if expires_s:
                try:
                    self._cookie_expiry_epoch = time.mktime(
                        time.strptime(expires_s, "%a, %d %b %Y %H:%M:%S %Z")
                    )
                except ValueError:
                    self._cookie_expiry_epoch = time.time() + 300.0
        else:
            self._cookie_gravity = resp.cookies.get("gravity", "")
            self._cookie_expiry_epoch = time.time() + 300.0
        self._grvt_account_id_header = str(resp.headers.get("X-Grvt-Account-Id") or "").strip()

    def start(self) -> None:
        if websocket is None:
            logger.error(
                "grvt_private_ws_disabled missing_dependency err=%s",
                _websocket_import_error,
            )
            return
        if not self._settings.private_ws_enabled:
            logger.info("grvt_private_ws_start_skipped PRIVATE_WS_ENABLED=false")
            return
        if not self._addr:
            logger.info("grvt_private_ws_start_skipped empty_sub_account_id")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="grvt-private-ws",
            daemon=True,
        )
        self._thread.start()
        logger.info("grvt_private_ws_thread_started sub_account_id=%s", self._addr)

    def stop(self) -> None:
        self._stop.set()
        # Tear down the keepalive worker first so it can't race against a
        # freshly-closed socket and log a bogus "idle" warning after stop.
        self._stop_keepalive_worker()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("grvt_private_ws_close_error")
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("grvt_private_ws_stopped")

    def _run_forever(self) -> None:
        initial = float(self._settings.private_ws_reconnect_initial_seconds)
        max_b = float(self._settings.private_ws_reconnect_max_seconds)
        delay_s = 0.0
        first = True
        attempt = 0
        while not self._stop.is_set():
            if not first and delay_s > 0:
                self._emit_conn(
                    PrivateWsConnectionKind.RECONNECT_SCHEDULED,
                    f"backoff_s={delay_s:.3f}",
                    delay_s,
                )
                self._stop.wait(timeout=delay_s)
            first = False
            if self._stop.is_set():
                break
            attempt += 1
            if attempt > 1 and self._state is not None:
                with self._state._lock:
                    self._state.private_ws_reconnect_count += 1
            try:
                self._connect_once()
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception("grvt_private_ws_session_failed")
                self._emit_conn(PrivateWsConnectionKind.ERROR, "session_exception", 0.0)
            delay_s = initial if delay_s <= 0 else min(max_b, max(delay_s * 2.0, initial))

    def _connect_once(self) -> None:
        assert websocket is not None
        self._refresh_cookie()
        url = (
            self._settings.grvt_private_ws_url
            or _default_private_ws_url(self._settings.grvt_env)
        )
        headers = [f"Cookie: gravity={self._cookie_gravity}"]
        if self._grvt_account_id_header:
            headers.append(f"X-Grvt-Account-Id: {self._grvt_account_id_header}")
        sub_account = self._addr
        selector_instrument = f"{sub_account}-{self._instrument}"

        def on_open(ws: Any) -> None:
            for stream, selector in (
                ("v1.fill", selector_instrument),
                ("v1.order", selector_instrument),
            ):
                ws.send(
                    json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": "subscribe",
                            "params": {"stream": stream, "selectors": [selector]},
                            "id": int(time.time() * 1000) % 1_000_000,
                        }
                    )
                )
            self._emit_conn(PrivateWsConnectionKind.CONNECTED, "open_and_subscribe_sent")
            if self._state is not None:
                with self._state._lock:
                    self._state.private_ws_last_connect_wall_ts = datetime.now(timezone.utc)
            # Spawn the idle-watch worker now that we have a live socket
            # and an active subscription. ``_start_keepalive`` tears down
            # any prior worker first.
            self._start_keepalive()

        def on_message(_ws: Any, message: str) -> None:
            recv_mono = time.perf_counter()
            recv_wall = datetime.now(timezone.utc)
            if self._state is not None:
                with self._state._lock:
                    self._state.private_ws_last_message_wall_ts = recv_wall
            self._handle_raw_message(message, recv_mono=recv_mono, recv_wall=recv_wall)

        def on_error(_ws: Any, error: Any) -> None:
            logger.warning("grvt_private_ws_on_error %s", error)

        def on_close(_ws: Any, close_status_code: Any, close_msg: Any) -> None:
            # Stop the idle watcher for this session; the next _connect_once
            # will spawn a fresh one. Called before emitting DISCONNECTED so
            # a just-closed worker can't linger and double-force-close.
            self._stop_keepalive_worker()
            detail = f"code={close_status_code} msg={str(close_msg or '')[:300]}"
            self._emit_conn(PrivateWsConnectionKind.DISCONNECTED, detail)

        with self._ws_lock:
            self._ws_app = websocket.WebSocketApp(
                url,
                header=headers,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            app = self._ws_app
        app.run_forever(ping_interval=25, ping_timeout=15)

    def _handle_raw_message(
        self, message: str, *, recv_mono: float, recv_wall: datetime
    ) -> None:
        parse_start = time.perf_counter()
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(
                "grvt_private_ws_malformed_json len=%s snippet=%s",
                len(message),
                message[:240].replace("\n", " "),
            )
            return
        parse_end = time.perf_counter()
        if not isinstance(obj, dict):
            return
        if obj.get("jsonrpc") == "2.0" and "result" in obj:
            return
        stream = str(obj.get("stream") or "")
        feed = obj.get("feed")
        if not isinstance(feed, dict):
            return
        if stream == "v1.fill":
            event_ms = int(int(str(feed.get("event_time") or "0")) // 1_000_000)
            timing = InboundPrivateTiming(
                ws_recv_mono=recv_mono,
                ws_recv_wall=recv_wall,
                parse_start_mono=parse_start,
                parse_end_mono=parse_end,
                enqueue_mono=time.perf_counter(),
                exchange_event_ms=event_ms,
            )
            fill_id = f"{feed.get('trade_id', '')}_{event_ms}"
            ev = PrivateFillEvent(
                fill_id=fill_id,
                oid=_grvt_oid_to_int(feed.get("order_id")),
                coin=str(feed.get("instrument") or ""),
                px=_to_float(feed.get("price")),
                sz=_to_float(feed.get("size")),
                side="B" if bool(feed.get("is_buyer")) else "A",
                time_ms=event_ms,
                fee=_to_float(feed.get("fee")),
                closed_pnl=_to_float(feed.get("realized_pnl")),
                crossed=False,
                is_snapshot=False,
                raw=dict(feed),
                inbound_timing=timing,
            )
            _safe_put(self._q, ev, "fill", self._on_queue_drop, self._after_queue_put)
            return
        if stream == "v1.order":
            # GRVT may emit bootstrap/snapshot order updates with ``order_id``
            # absent or "0" — those are placeholders, not a real identifier.
            # Previously we enqueued them with ``oid=0``; the execution layer
            # then matched against any local WO whose ``order_id_exchange`` was
            # also 0 (after a bogus place-ack) and applied stale status, which
            # poisoned the order-state machine.
            parsed_oid = _grvt_oid_to_int(feed.get("order_id"))
            if parsed_oid is None:
                return
            state = feed.get("state") if isinstance(feed.get("state"), dict) else {}
            update_ns = state.get("update_time")
            event_ms = int(int(str(update_ns or "0")) // 1_000_000)
            legs = feed.get("legs")
            leg0 = legs[0] if isinstance(legs, list) and legs and isinstance(legs[0], dict) else {}
            timing = InboundPrivateTiming(
                ws_recv_mono=recv_mono,
                ws_recv_wall=recv_wall,
                parse_start_mono=parse_start,
                parse_end_mono=parse_end,
                enqueue_mono=time.perf_counter(),
                exchange_event_ms=event_ms,
            )
            metadata = feed.get("metadata") if isinstance(feed.get("metadata"), dict) else {}
            cloid_raw = metadata.get("client_order_id") if isinstance(metadata, dict) else None
            cloid = str(cloid_raw) if cloid_raw not in (None, "") else None
            ev = PrivateOrderUpdateEvent(
                oid=parsed_oid,
                coin=str(leg0.get("instrument") or ""),
                status=str(state.get("status") or ""),
                status_timestamp_ms=event_ms,
                side="B" if bool(leg0.get("is_buying_asset")) else "A",
                limit_px=_to_float(leg0.get("limit_price")),
                remaining_sz=_to_float((state.get("book_size") or [0])[0]),
                orig_sz=_to_float(leg0.get("size")),
                raw_status=str(state.get("reject_reason") or ""),
                inbound_timing=timing,
                cloid=cloid,
            )
            _safe_put(self._q, ev, "order_update", self._on_queue_drop, self._after_queue_put)

    def feed_message_for_tests(self, message: str) -> None:
        self._handle_raw_message(
            message,
            recv_mono=time.perf_counter(),
            recv_wall=datetime.now(timezone.utc),
        )


def _grvt_oid_to_int(v: Any) -> Optional[int]:
    """Parse a GRVT order id into a non-zero int. See :func:`app.exchange.grvt_client._grvt_oid_to_int`."""
    if v is None:
        return None
    try:
        s = str(v)
        if s.lower().startswith("0x"):
            parsed = int(s, 16)
        else:
            parsed = int(s)
    except (TypeError, ValueError):
        return None
    return parsed if parsed != 0 else None


def _to_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["GrvtPrivateStream"]
