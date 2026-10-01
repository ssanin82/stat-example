"""Hyperliquid public websocket: BBO (best bid/offer) for live top-of-book — no REST polling on the quote path."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.config import Settings
from app.inbound_timing import InboundPublicTiming
from app.models import BestBidAsk
from app.state import BotState

logger = logging.getLogger(__name__)

try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error = e
else:
    _websocket_import_error = None

BboCallback = Callable[[BestBidAsk], None]


def _parse_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def best_bid_ask_from_bbo_dict(
    symbol: str, data: dict[str, Any]
) -> Optional[BestBidAsk]:
    """
    Parse Hyperliquid ``bbo`` channel payload: ``{coin, time, bbo: [bid_level|null, ask_level|null]}``.
    Each level is ``{px, sz}`` strings.
    """
    try:
        coin = str(data.get("coin", ""))
        if coin != symbol:
            return None
        t = data.get("time")
        ts_exchange_ms = int(t) if t is not None else None
        levels = data.get("bbo")
        if not isinstance(levels, list) or len(levels) < 2:
            return None
        bid_l, ask_l = levels[0], levels[1]
        best_bid = None
        best_ask = None
        bid_size: Optional[float] = None
        ask_size: Optional[float] = None
        if isinstance(bid_l, dict):
            best_bid = _parse_float(bid_l.get("px"))
            bid_size = _parse_float(bid_l.get("sz"))
        if isinstance(ask_l, dict):
            best_ask = _parse_float(ask_l.get("px"))
            ask_size = _parse_float(ask_l.get("sz"))
        if (
            best_bid is None
            or best_ask is None
            or not (best_bid > 0 and best_ask > 0)
        ):
            return None
        mid = (best_bid + best_ask) / 2.0
        spread_bps = (best_ask - best_bid) / mid * 10_000.0 if mid > 0 else None
        ts_local: datetime
        if ts_exchange_ms is not None:
            ts_local = datetime.fromtimestamp(
                ts_exchange_ms / 1000.0, tz=timezone.utc
            )
        else:
            ts_local = datetime.now(timezone.utc)
        return BestBidAsk(
            symbol=symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread_bps=spread_bps,
            ts_exchange_ms=ts_exchange_ms,
            ts_local=ts_local,
            bid_size=bid_size,
            ask_size=ask_size,
        )
    except (TypeError, ValueError, KeyError):
        return None


class HyperliquidPublicStream:
    """
    Dedicated thread: connect to HL WS, subscribe to ``bbo`` for ``coin``, parse → ``on_bbo``.
    Connection health and last message time are mirrored into ``BotState`` under the state lock.
    """

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        symbol: str,
        on_bbo: BboCallback,
    ) -> None:
        self._settings = settings
        self._state = state
        self._symbol = (symbol or "").strip()
        self._on_bbo = on_bbo
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None

    def start(self) -> None:
        if websocket is None:
            logger.error(
                "public_ws_disabled missing_dependency err=%s",
                _websocket_import_error,
            )
            return
        if not self._settings.public_ws_enabled:
            logger.info("public_ws_start_skipped PUBLIC_WS_ENABLED=false")
            return
        if not self._symbol:
            logger.info("public_ws_start_skipped empty_SYMBOL")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="hl-public-ws",
            daemon=True,
        )
        self._thread.start()
        logger.info("public_ws_thread_started symbol=%s", self._symbol)

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("public_ws_close_error")
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        with self._state._lock:
            self._state.public_ws_connected = False
        logger.info("public_ws_stopped")

    def request_reconnect(self) -> None:
        """Close the socket; the run loop reconnects with backoff."""
        with self._ws_lock:
            ws = self._ws_app
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("public_ws_request_reconnect_close_failed")

    def _run_forever(self) -> None:
        initial = float(self._settings.public_ws_reconnect_initial_seconds)
        max_b = float(self._settings.public_ws_reconnect_max_seconds)
        delay_s = 0.0
        first = True
        attempt = 0
        while not self._stop.is_set():
            if not first and delay_s > 0:
                logger.debug("public_ws_backoff sleep_s=%.3f", delay_s)
                self._stop.wait(timeout=delay_s)
            first = False
            if self._stop.is_set():
                break
            attempt += 1
            if attempt > 1:
                with self._state._lock:
                    self._state.public_ws_reconnect_count += 1
            try:
                self._connect_once()
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception("public_ws_session_failed")
            if delay_s <= 0:
                delay_s = initial
            else:
                delay_s = min(max_b, max(delay_s * 2, initial))

    def _connect_once(self) -> None:
        assert websocket is not None
        url = self._settings.hl_ws_url
        sym = self._symbol

        def on_open(ws: Any) -> None:
            logger.info("public_ws_raw_connected url=%s symbol=%s", url, sym)
            payload = {
                "method": "subscribe",
                "subscription": {"type": "bbo", "coin": sym},
            }
            ws.send(json.dumps(payload))
            with self._state._lock:
                self._state.public_ws_connected = True
            # Timing window: avoid reconnect holes poisoning gap metrics.
            try:
                tr = getattr(self._state, "public_ws_timing", None)
                if tr is not None:
                    tr.note_stream_reset("public_ws_open")
            except Exception:
                pass

        def on_message(_ws: Any, message: str) -> None:
            self._handle_raw_message(message)

        def on_error(_ws: Any, error: Any) -> None:
            logger.warning("public_ws_on_error %s", error)

        def on_close(_ws: Any, close_status_code: Any, close_msg: Any) -> None:
            logger.info(
                "public_ws_disconnected code=%s msg=%s",
                close_status_code,
                (str(close_msg)[:200] if close_msg is not None else ""),
            )
            with self._state._lock:
                self._state.public_ws_connected = False
            # Timing window: avoid reconnect holes poisoning gap metrics.
            try:
                tr = getattr(self._state, "public_ws_timing", None)
                if tr is not None:
                    tr.note_stream_reset("public_ws_close")
            except Exception:
                pass

        with self._ws_lock:
            self._ws_app = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            app = self._ws_app

        logger.debug("public_ws_run_forever_start url=%s", url)
        app.run_forever(ping_interval=25, ping_timeout=15)

    def _handle_raw_message(self, message: str) -> None:
        recv_mono = time.perf_counter()
        recv_wall = datetime.now(timezone.utc)
        parse_start = time.perf_counter()
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(
                "public_ws_malformed_json len=%s snippet=%s",
                len(message),
                message[:240].replace("\n", " "),
            )
            return
        parse_end = time.perf_counter()
        if not isinstance(obj, dict):
            return
        ch = obj.get("channel")
        if ch == "subscriptionResponse":
            logger.info("public_ws_subscription_ack data=%s", obj.get("data"))
            return
        if ch == "pong":
            return
        if ch != "bbo":
            logger.debug("public_ws_ignored channel=%s", ch)
            return
        data = obj.get("data")
        if not isinstance(data, dict):
            return
        cb_start = time.perf_counter()
        bb = best_bid_ask_from_bbo_dict(self._symbol, data)
        if bb is None:
            logger.debug("public_ws_bbo_parse_skip data_keys=%s", list(data.keys())[:12])
            return
        it = InboundPublicTiming(
            ws_recv_mono=recv_mono,
            ws_recv_wall=recv_wall,
            parse_start_mono=parse_start,
            parse_end_mono=parse_end,
            callback_start_mono=cb_start,
            apply_mono=0.0,
            exchange_ts_ms=bb.ts_exchange_ms,
        )
        bb = replace(bb, inbound_public_timing=it)
        with self._state._lock:
            self._state.public_ws_last_message_wall_ts = recv_wall
        try:
            self._on_bbo(bb)
        except Exception:
            logger.exception("public_ws_on_bbo_callback_failed")

    def feed_message_for_tests(self, message: str) -> None:
        """Parse one raw WS text frame (unit tests; no socket)."""
        self._handle_raw_message(message)
