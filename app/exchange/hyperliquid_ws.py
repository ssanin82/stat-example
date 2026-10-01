"""Hyperliquid private websocket: userFills + orderUpdates only (network I/O → bounded queue)."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.config import Settings
from app.inbound_timing import InboundPrivateTiming
from app.state import BotState
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
    PrivateWsConnectionEvent,
    PrivateWsConnectionKind,
)

logger = logging.getLogger(__name__)

try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error = e
else:
    _websocket_import_error = None


DropCallback = Optional[Callable[[str], None]]
AfterPutCallback = Optional[Callable[[str], None]]

# Hyperliquid websocket application keepalive (JSON text frame).
_PRIVATE_WS_APP_PING_PAYLOAD = json.dumps({"method": "ping"})

# Rate limits for INFO logs (wall clock is fine; monotonic for intervals).
_PRIVATE_WS_PING_LOG_MIN_INTERVAL_S = 60.0
_PRIVATE_WS_PONG_LOG_MIN_INTERVAL_S = 60.0
_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S = 120.0


def classify_private_ws_inactive_close(close_status_code: Any, close_msg: Any) -> bool:
    """Server idle policy: close message often contains ``Inactive`` (case-insensitive)."""
    _ = close_status_code
    return "inactive" in str(close_msg or "").lower()


def _parse_close_code(close_status_code: Any) -> Optional[int]:
    if close_status_code is None:
        return None
    try:
        return int(close_status_code)
    except (TypeError, ValueError):
        return None


def _next_session_backoff(prev: float, initial: float, max_b: float) -> float:
    if prev <= 0:
        return initial
    return min(max_b, max(prev * 2.0, initial))


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
        logger.error(
            "private_ws_queue_full dropped=%s (increase PRIVATE_WS_QUEUE_MAX); "
            "REST fill/open-order catchup will run on next bot drain",
            drop_label,
        )
        if on_queue_drop is not None:
            try:
                on_queue_drop(drop_label)
            except Exception:
                logger.exception("private_ws_on_queue_drop_callback_failed label=%s", drop_label)
        return
    if on_after_put is not None:
        try:
            on_after_put(drop_label)
        except Exception:
            logger.exception("private_ws_on_after_put_failed label=%s", drop_label)


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize_fill_row(
    f: dict[str, Any],
    is_snapshot: bool,
    *,
    inbound_timing: Optional[InboundPrivateTiming] = None,
) -> Optional[PrivateFillEvent]:
    try:
        coin = str(f["coin"])
        px = _parse_float(f.get("px"))
        sz = _parse_float(f.get("sz"))
        if px is None or sz is None:
            return None
        side = str(f.get("side", "A"))
        t = f.get("time")
        time_ms = int(t) if t is not None else 0
        oid_raw = f.get("oid")
        try:
            oid = int(oid_raw) if oid_raw is not None else None
        except (TypeError, ValueError):
            oid = None
        fee = _parse_float(f.get("fee")) or 0.0
        try:
            cp = float(f.get("closedPnl", 0) or 0)
        except (TypeError, ValueError):
            cp = 0.0
        h = str(f.get("hash", ""))
        fill_id = f"{h}_{time_ms}"
        crossed = bool(f.get("crossed", False))
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "private_ws_malformed_fill keys_sample=%s",
            list(f.keys())[:24] if isinstance(f, dict) else type(f).__name__,
        )
        return None
    return PrivateFillEvent(
        fill_id=fill_id,
        oid=oid,
        coin=coin,
        px=px,
        sz=sz,
        side=side,
        time_ms=time_ms,
        fee=fee,
        closed_pnl=cp,
        crossed=crossed,
        is_snapshot=is_snapshot,
        raw=dict(f),
        inbound_timing=inbound_timing,
    )


def _normalize_order_row(
    row: dict[str, Any],
    *,
    inbound_timing: Optional[InboundPrivateTiming] = None,
) -> Optional[PrivateOrderUpdateEvent]:
    try:
        order = row["order"]
        if not isinstance(order, dict):
            return None
        oid = int(order["oid"])
        coin = str(order["coin"])
        side = str(order.get("side", "A"))
        limit_px = _parse_float(order.get("limitPx"))
        sz = _parse_float(order.get("sz"))
        orig_sz = _parse_float(order.get("origSz"))
        if limit_px is None or sz is None:
            return None
        if orig_sz is None:
            orig_sz = sz
        status = str(row.get("status", ""))
        st = row.get("statusTimestamp")
        status_ts = int(st) if st is not None else 0
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "private_ws_malformed_order_update keys_sample=%s",
            list(row.keys())[:24] if isinstance(row, dict) else type(row).__name__,
        )
        return None
    return PrivateOrderUpdateEvent(
        oid=oid,
        coin=coin,
        status=status,
        status_timestamp_ms=status_ts,
        side=side,
        limit_px=limit_px,
        remaining_sz=sz,
        orig_sz=orig_sz,
        raw_status=status,
        inbound_timing=inbound_timing,
    )


class HyperliquidPrivateStream:
    """
    Dedicated thread: connect, subscribe to userFills + orderUpdates, parse → queue.

    Optional ``BotState`` mirrors transport timestamps, reconnect stats, and disconnect metadata
    for health endpoints (bot thread still owns ``private_ws_connected`` / healthy flags via events).
    """

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
        self._close_was_inactive = False
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._last_ping_log_mono = 0.0
        self._last_pong_log_mono = 0.0
        self._last_idle_log_mono = 0.0

    def start(self) -> None:
        if websocket is None:
            logger.error(
                "private_ws_disabled missing_dependency err=%s",
                _websocket_import_error,
            )
            return
        if not self._settings.private_ws_enabled:
            logger.info("private_ws_start_skipped PRIVATE_WS_ENABLED=false")
            return
        if not self._addr:
            logger.info("private_ws_start_skipped empty_HL_ACCOUNT_ADDRESS")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="hl-private-ws",
            daemon=True,
        )
        self._thread.start()
        logger.info("private_ws_thread_started user=%s…", self._addr[:12])

    def stop(self) -> None:
        self._stop.set()
        self._stop_keepalive_worker()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("private_ws_close_error")
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("private_ws_stopped")

    def _stop_keepalive_worker(self) -> None:
        self._keepalive_stop.set()
        t = self._keepalive_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.5)
        self._keepalive_thread = None

    def _start_keepalive(self) -> None:
        self._stop_keepalive_worker()
        self._keepalive_stop.clear()
        t = threading.Thread(
            target=self._keepalive_worker,
            name="hl-private-ws-keepalive",
            daemon=True,
        )
        self._keepalive_thread = t
        t.start()

    def _keepalive_worker(self) -> None:
        interval = float(self._settings.private_ws_app_keepalive_seconds)
        idle_warn = float(self._settings.private_ws_idle_warn_seconds)
        while not self._keepalive_stop.wait(timeout=interval):
            if self._stop.is_set():
                break
            with self._ws_lock:
                ws = self._ws_app
            if ws is None:
                continue
            now_mono = time.monotonic()
            if self._state is not None and idle_warn > 0:
                with self._state._lock:
                    last_msg = self._state.private_ws_last_message_wall_ts
                if last_msg is not None:
                    idle_s = max(0.0, (datetime.now(timezone.utc) - last_msg).total_seconds())
                    if idle_s >= idle_warn:
                        if now_mono - self._last_idle_log_mono >= _PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S:
                            self._last_idle_log_mono = now_mono
                            logger.warning(
                                "private_ws_idle_too_long idle_s=%.1f threshold_s=%.1f",
                                idle_s,
                                idle_warn,
                            )
            try:
                ws.send(_PRIVATE_WS_APP_PING_PAYLOAD)
            except Exception:
                logger.debug("private_ws_keepalive_send_failed", exc_info=True)
                continue
            wall = datetime.now(timezone.utc)
            if self._state is not None:
                with self._state._lock:
                    self._state.private_ws_last_ping_sent_wall_ts = wall
            now_mono = time.monotonic()
            if now_mono - self._last_ping_log_mono >= _PRIVATE_WS_PING_LOG_MIN_INTERVAL_S:
                self._last_ping_log_mono = now_mono
                logger.info(
                    "private_ws_ping_sent interval_s=%.1f payload=%s",
                    interval,
                    _PRIVATE_WS_APP_PING_PAYLOAD,
                )

    def _touch_last_message_wall_ts(self) -> None:
        wall = datetime.now(timezone.utc)
        st = self._state
        if st is None:
            return
        with st._lock:
            st.private_ws_last_message_wall_ts = wall

    def _note_disconnect_state(
        self,
        *,
        code: Optional[int],
        reason_s: str,
        hist_key: str,
    ) -> None:
        st = self._state
        if st is None:
            return
        with st._lock:
            st.private_ws_disconnect_close_code_last = code
            st.private_ws_disconnect_reason_last = reason_s[:500] if reason_s else None
            h = st.private_ws_disconnect_histogram
            h[hist_key] = h.get(hist_key, 0) + 1

    def _bump_reconnect_counters(self, reason_class: str) -> None:
        st = self._state
        if st is None:
            return
        with st._lock:
            st.private_ws_reconnect_count += 1
            rc = st.private_ws_reconnect_reason_counts
            rc[reason_class] = rc.get(reason_class, 0) + 1

    def _after_queue_put(self, label: str) -> None:
        if self._state is not None and label in ("fill", "order_update"):
            self._state.wake_quote_loop()

    def _emit_conn(
        self,
        kind: PrivateWsConnectionKind,
        detail: str = "",
        backoff_seconds: float = 0.0,
    ) -> None:
        ev = PrivateWsConnectionEvent(
            kind=kind, detail=detail, backoff_seconds=backoff_seconds
        )
        _safe_put(self._q, ev, "connection", self._on_queue_drop, self._after_queue_put)

        if kind == PrivateWsConnectionKind.RECONNECT_SCHEDULED:
            log_fn = logger.debug
        else:
            log_fn = logger.info
        log_fn(
            "private_ws_connection_event kind=%s detail=%s backoff_s=%s",
            kind.value,
            detail[:300] if detail else "",
            backoff_seconds,
        )

    def _run_forever(self) -> None:
        initial = float(self._settings.private_ws_reconnect_initial_seconds)
        max_b = float(self._settings.private_ws_reconnect_max_seconds)
        inactive_sleep = float(self._settings.private_ws_inactive_reconnect_seconds)
        failure_backoff = 0.0
        pending_sleep = 0.0
        pending_reason = "session_closed"
        first = True

        while not self._stop.is_set():
            if not first:
                if pending_sleep > 0:
                    self._emit_conn(
                        PrivateWsConnectionKind.RECONNECT_SCHEDULED,
                        f"reason={pending_reason} backoff_s={pending_sleep:.3f}",
                        pending_sleep,
                    )
                    logger.info(
                        "private_ws_reconnect_scheduled reason=%s backoff_s=%.3f",
                        pending_reason,
                        pending_sleep,
                    )
                    self._stop.wait(timeout=pending_sleep)
                self._bump_reconnect_counters(pending_reason)
                logger.info("private_ws_reconnect_started reason=%s", pending_reason)

            if self._stop.is_set():
                break

            try:
                self._connect_once()
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception("private_ws_session_failed")
                self._emit_conn(
                    PrivateWsConnectionKind.ERROR,
                    "session_exception",
                    0.0,
                )
                self._close_was_inactive = False
                failure_backoff = _next_session_backoff(failure_backoff, initial, max_b)
                pending_sleep = failure_backoff
                pending_reason = "exception"
                first = False
                continue

            if self._close_was_inactive:
                pending_sleep = inactive_sleep
                pending_reason = "inactive"
                failure_backoff = 0.0
            else:
                failure_backoff = _next_session_backoff(failure_backoff, initial, max_b)
                pending_sleep = failure_backoff
                pending_reason = "session_closed"
            self._close_was_inactive = False
            first = False

    def _connect_once(self) -> None:
        assert websocket is not None
        url = self._settings.hl_ws_url
        user = self._addr
        if user.startswith("0x"):
            user_lc = "0x" + user[2:].lower()
        else:
            user_lc = user

        stream_self = self

        def on_open(ws: Any) -> None:
            logger.info("private_ws_raw_connected url=%s", url)
            wall = datetime.now(timezone.utc)
            if stream_self._state is not None:
                with stream_self._state._lock:
                    stream_self._state.private_ws_last_connect_wall_ts = wall
            for sub_type in ("userFills", "orderUpdates"):
                payload = {
                    "method": "subscribe",
                    "subscription": {"type": sub_type, "user": user_lc},
                }
                ws.send(json.dumps(payload))
            stream_self._emit_conn(
                PrivateWsConnectionKind.CONNECTED, "open_and_subscribe_sent"
            )
            stream_self._start_keepalive()

        def on_message(_ws: Any, message: str) -> None:
            stream_self._touch_last_message_wall_ts()
            recv_mono = time.perf_counter()
            recv_wall = datetime.now(timezone.utc)
            stream_self._handle_raw_message(message, recv_mono=recv_mono, recv_wall=recv_wall)

        def on_error(_ws: Any, error: Any) -> None:
            logger.warning("private_ws_on_error %s", error)

        def on_close(_ws: Any, close_status_code: Any, close_msg: Any) -> None:
            stream_self._stop_keepalive_worker()
            reason_s = str(close_msg or "")
            code_int = _parse_close_code(close_status_code)
            detail = f"code={close_status_code} msg={reason_s}"[:500]
            inactive = classify_private_ws_inactive_close(close_status_code, close_msg)
            stream_self._close_was_inactive = inactive
            hist_key = f"{close_status_code}:{reason_s[:120]}"
            stream_self._note_disconnect_state(
                code=code_int,
                reason_s=reason_s,
                hist_key=hist_key,
            )
            logger.info(
                "private_ws_closed code=%s reason=%s inactive=%s",
                close_status_code,
                reason_s[:300],
                inactive,
            )
            stream_self._emit_conn(PrivateWsConnectionKind.DISCONNECTED, detail)

        with self._ws_lock:
            self._ws_app = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            app = self._ws_app

        logger.debug("private_ws_run_forever_start url=%s", url)
        app.run_forever(ping_interval=25, ping_timeout=15)

    def _handle_raw_message(
        self,
        message: str,
        *,
        recv_mono: float,
        recv_wall: datetime,
    ) -> None:
        self._touch_last_message_wall_ts()
        parse_start = time.perf_counter()
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(
                "private_ws_malformed_json len=%s snippet=%s",
                len(message),
                message[:240].replace("\n", " "),
            )
            return
        parse_end = time.perf_counter()
        if not isinstance(obj, dict):
            logger.warning("private_ws_malformed_top_level type=%s", type(obj).__name__)
            return
        ch = obj.get("channel")
        data = obj.get("data")
        if ch == "subscriptionResponse":
            logger.info("private_ws_subscription_ack data=%s", data)
            return
        if ch == "pong":
            wall = datetime.now(timezone.utc)
            if self._state is not None:
                with self._state._lock:
                    self._state.private_ws_last_pong_wall_ts = wall
            now_mono = time.monotonic()
            if now_mono - self._last_pong_log_mono >= _PRIVATE_WS_PONG_LOG_MIN_INTERVAL_S:
                self._last_pong_log_mono = now_mono
                logger.info("private_ws_pong_received channel=pong")
            return

        if ch == "userFills":
            if not isinstance(data, dict):
                logger.warning("private_ws_userFills_data_not_dict")
                return
            is_snap = bool(data.get("isSnapshot"))
            fills = data.get("fills")
            if not isinstance(fills, list):
                logger.warning("private_ws_userFills_missing_fills_list")
                return
            logger.debug(
                "private_ws_raw channel=userFills n=%s isSnapshot=%s",
                len(fills),
                is_snap,
            )
            logger.info(
                "private_ws_normalized_batch channel=userFills count=%s isSnapshot=%s",
                len(fills),
                is_snap,
            )
            for f in fills:
                if not isinstance(f, dict):
                    continue
                t = f.get("time")
                ex_ms = int(t) if t is not None else 0
                enq = time.perf_counter()
                it = InboundPrivateTiming(
                    ws_recv_mono=recv_mono,
                    ws_recv_wall=recv_wall,
                    parse_start_mono=parse_start,
                    parse_end_mono=parse_end,
                    enqueue_mono=enq,
                    exchange_event_ms=ex_ms,
                )
                ev = _normalize_fill_row(f, is_snap, inbound_timing=it)
                if ev:
                    _safe_put(self._q, ev, "fill", self._on_queue_drop, self._after_queue_put)
            return

        if ch == "orderUpdates":
            if not isinstance(data, list):
                logger.warning("private_ws_orderUpdates_data_not_list")
                return
            logger.debug("private_ws_raw channel=orderUpdates n=%s", len(data))
            logger.info(
                "private_ws_normalized_batch channel=orderUpdates count=%s",
                len(data),
            )
            for row in data:
                if not isinstance(row, dict):
                    continue
                st = row.get("statusTimestamp") if isinstance(row, dict) else None
                ex_ms = int(st) if st is not None else 0
                enq = time.perf_counter()
                it = InboundPrivateTiming(
                    ws_recv_mono=recv_mono,
                    ws_recv_wall=recv_wall,
                    parse_start_mono=parse_start,
                    parse_end_mono=parse_end,
                    enqueue_mono=enq,
                    exchange_event_ms=ex_ms,
                )
                ev = _normalize_order_row(row, inbound_timing=it)
                if ev:
                    _safe_put(
                        self._q, ev, "order_update", self._on_queue_drop, self._after_queue_put
                    )
            return

        # Other channels ignored here by design.
        logger.debug("private_ws_raw_ignored channel=%s", ch)

    def feed_message_for_tests(self, message: str) -> None:
        """
        Parse one raw WS text frame and enqueue events (unit tests only; no socket).
        """
        recv_mono = time.perf_counter()
        recv_wall = datetime.now(timezone.utc)
        self._handle_raw_message(message, recv_mono=recv_mono, recv_wall=recv_wall)


# Backwards-compatible name used in docs / imports
HyperliquidWsPlaceholder = HyperliquidPrivateStream
