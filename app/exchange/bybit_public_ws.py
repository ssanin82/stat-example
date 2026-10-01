"""Bybit public-WS subscriber for cross-venue fair-value reference.

Alternative to :mod:`app.exchange.binance_public_ws`. Selected via
``REFERENCE_EXCHANGE=bybit``. Writes to the same ``BotState.binance_*``
fields (kept under that name for back-compat — the cross-venue-cancel
execution path doesn't care which venue produced the reference).

Why Bybit
---------
Measured one-way latency from a Singapore host (Railway-era baseline,
2026-04; geographic measurement remains a useful upper bound for any
SG-hosted deployment)
(``C:/Work/DTC/xch-md-measure/tmp/logs.singapore.log``):

    venue=bybit    p50=3 ms  p95=4 ms  max=31 ms
    venue=binance  p50=44 ms p95=79 ms max=153 ms

The ~40 ms p50 / ~75 ms p95 advantage is material for cutting adverse
selection on GRVT's ETH book — latency-advantaged takers win the
stale-quote race in exactly this window.

Message format (Bybit v5 ``orderbook.1`` stream, from
https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook):

    {
      "topic": "orderbook.1.ETHUSDT",
      "type":  "snapshot" | "delta",
      "ts":    1744024600123,        // server send time (ms)
      "data": {
        "s":   "ETHUSDT",
        "b":   [["2300.50", "12.345"]],   // [price, size][]; empty on delta with no bid change
        "a":   [["2300.51", "5.678"]],    // [price, size][]
        "u":   123456,
        "seq": 789
      },
      "cts":   1744024600100         // creation time (optional)
    }

For depth 1, ``snapshot`` carries a populated ``b`` and ``a`` each. A
``delta`` only contains the side(s) that changed — if ``b`` is empty
the best bid hasn't moved; the previous value stands. We maintain the
last-seen bid/ask in this class so we always emit a consistent
top-of-book snapshot into state, even when Bybit sends a one-sided
delta.

Heartbeat
---------
Bybit v5 requires an *application-level* ping every 20 s (``{"op":"ping"}``)
and considers the connection dead if no ping arrives within that
window. We send JSON pings on a sidecar thread so WS-protocol ping
settings are irrelevant; we leave ``run_forever`` without
``ping_interval`` so nothing interferes with our pings.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.config import Settings
from app.state import BotState

logger = logging.getLogger(__name__)

try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error = e
else:
    _websocket_import_error = None


_BYBIT_PING_INTERVAL_SECONDS = 20.0


class BybitPublicStream:
    """Dedicated thread: subscribe to Bybit ``orderbook.1`` and update state.

    Interface mirrors :class:`BinancePublicStream` so ``app/main.py``
    can pick between them with a single env switch.
    """

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        on_bbo_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self._settings = settings
        self._state = state
        self._on_bbo = on_bbo_callback
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None
        self._url = settings.bybit_ws_base_url
        self._symbol = settings.bybit_symbol.strip().upper()
        self._topic = f"orderbook.1.{self._symbol}"
        self._basis_alpha = float(settings.binance_basis_ewma_alpha)
        # Track last-seen top-of-book between messages so deltas with an
        # empty ``b`` or ``a`` can still emit a complete snapshot.
        self._last_bid: Optional[float] = None
        self._last_ask: Optional[float] = None
        self._last_bid_size: Optional[float] = None
        self._last_ask_size: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if websocket is None:
            logger.error(
                "bybit_public_ws_disabled missing_dependency err=%s",
                _websocket_import_error,
            )
            return
        # Respect the generic kill-switch: ``BINANCE_WS_ENABLED=false`` or
        # ``REFERENCE_EXCHANGE=off`` should disable both streams. Caller
        # in ``main.py`` already gates on ``reference_exchange``; this
        # check is belt-and-braces.
        if not self._settings.binance_ws_enabled:
            logger.info(
                "bybit_public_ws_disabled BINANCE_WS_ENABLED=false"
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(target=self._run_forever, name="bybit-public-ws", daemon=True)
        self._thread = t
        t.start()
        logger.info(
            "bybit_public_ws_thread_started url=%s topic=%s",
            self._url,
            self._topic,
        )

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.debug("bybit_public_ws_close_failed", exc_info=True)
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5.0)
        self._thread = None
        pt = self._ping_thread
        if pt is not None and pt.is_alive() and pt is not threading.current_thread():
            pt.join(timeout=2.0)
        self._ping_thread = None
        logger.info("bybit_public_ws_stopped")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def _run_forever(self) -> None:
        initial = 0.5
        max_backoff = 30.0
        delay = 0.0
        first = True
        attempt = 0
        while not self._stop.is_set():
            if not first and delay > 0:
                self._stop.wait(timeout=delay)
            first = False
            if self._stop.is_set():
                break
            attempt += 1
            if attempt > 1:
                with self._state._lock:
                    self._state.binance_ws_reconnect_count += 1
            try:
                self._connect_once()
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception("bybit_public_ws_session_failed")
            with self._state._lock:
                self._state.binance_ws_connected = False
            delay = initial if delay <= 0 else min(max_backoff, delay * 2.0)

    def _connect_once(self) -> None:
        assert websocket is not None

        def on_open(ws: Any) -> None:
            with self._state._lock:
                self._state.binance_ws_connected = True
                self._state.binance_ws_last_connect_ts = datetime.now(timezone.utc)
                # 2026-05-12 codex-#6: record which venue is feeding
                # the reference fields. Bybit writes into the same
                # ``binance_*`` state slots today (legacy naming;
                # see ``BUGS/todo-014-generic-reference-fields.md``).
                self._state.reference_venue_name = "bybit"
                tracker = self._state.binance_public_ws_timing
            if tracker is not None:
                tracker.note_stream_reset("bybit_public_ws_open")
            logger.info(
                "bybit_public_ws_open url=%s topic=%s",
                self._url,
                self._topic,
            )
            # Subscribe to the top-of-book stream. Bybit expects a JSON
            # payload, not a URL-encoded path (unlike Binance).
            try:
                ws.send(json.dumps({"op": "subscribe", "args": [self._topic]}))
            except Exception:
                logger.exception("bybit_public_ws_subscribe_send_failed")
            # Start application-level ping loop (Bybit requires every 20 s).
            self._start_ping_loop(ws)

        def on_message(_ws: Any, raw: Any) -> None:
            self._handle_raw_message(raw)

        def on_error(_ws: Any, err: Any) -> None:
            logger.warning("bybit_public_ws_error err=%s", err)

        def on_close(_ws: Any, close_status_code: Any, close_msg: Any) -> None:
            with self._state._lock:
                self._state.binance_ws_connected = False
            logger.info(
                "bybit_public_ws_close code=%s msg=%s",
                close_status_code,
                close_msg,
            )

        with self._ws_lock:
            self._ws_app = websocket.WebSocketApp(
                self._url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            app = self._ws_app
        # Run without protocol-level ping — Bybit uses app-level JSON pings
        # which we drive ourselves on a sidecar thread.
        app.run_forever()

    def _start_ping_loop(self, ws: Any) -> None:
        def ping_loop() -> None:
            while not self._stop.is_set():
                if self._stop.wait(timeout=_BYBIT_PING_INTERVAL_SECONDS):
                    return
                try:
                    ws.send(json.dumps({"op": "ping"}))
                except Exception:
                    logger.debug("bybit_public_ws_ping_send_failed", exc_info=True)
                    return
        pt = threading.Thread(target=ping_loop, name="bybit-public-ws-ping", daemon=True)
        self._ping_thread = pt
        pt.start()

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_raw_message(self, raw: Any) -> None:
        if not isinstance(raw, (str, bytes, bytearray)):
            return
        local_recv_mono_ns = time.monotonic_ns()
        local_recv_wall_ms = int(time.time() * 1000.0)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(obj, dict):
            return

        # Connection acks / subscribe confirms / pong replies — ignore.
        # These arrive as ``{"op":"subscribe","success":true,...}`` or
        # ``{"op":"pong",...}`` without a ``topic``.
        topic = obj.get("topic")
        if not isinstance(topic, str) or not topic.startswith("orderbook."):
            return
        data = obj.get("data")
        if not isinstance(data, dict):
            return
        msg_type = obj.get("type")

        # Extract best bid / ask. For snapshots both sides are populated;
        # for deltas only changed sides appear. When a side is missing we
        # fall back to the last-seen value so state always reflects a
        # consistent top-of-book pair.
        def _first_level(side_key: str) -> tuple[Optional[float], Optional[float]]:
            rows = data.get(side_key)
            if not isinstance(rows, list) or not rows:
                return None, None
            row = rows[0]
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                return None, None
            try:
                return float(row[0]), float(row[1])
            except (TypeError, ValueError):
                return None, None

        new_bid, new_bid_size = _first_level("b")
        new_ask, new_ask_size = _first_level("a")

        # Snapshots reset last-seen; deltas preserve it where the side is
        # absent. A snapshot with an empty side is malformed; drop it.
        if msg_type == "snapshot":
            if new_bid is None or new_ask is None:
                return
            self._last_bid = new_bid
            self._last_ask = new_ask
            self._last_bid_size = new_bid_size
            self._last_ask_size = new_ask_size
        else:
            # delta (or missing type — treat as delta)
            if new_bid is not None:
                self._last_bid = new_bid
                self._last_bid_size = new_bid_size
            if new_ask is not None:
                self._last_ask = new_ask
                self._last_ask_size = new_ask_size

        best_bid = self._last_bid
        best_ask = self._last_ask
        if best_bid is None or best_ask is None:
            return
        if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
            return

        mid = (best_bid + best_ask) / 2.0
        now = datetime.now(timezone.utc)

        # Bybit v5 carries ``ts`` (server send time in ms) on every message.
        # Older revisions also emit ``cts`` (creation time); prefer ``ts``.
        exchange_ts_ms: Optional[int] = None
        for key in ("ts", "cts"):
            raw_ts = obj.get(key)
            if raw_ts is None:
                continue
            try:
                candidate = int(raw_ts)
            except (TypeError, ValueError):
                continue
            if candidate > 0:
                exchange_ts_ms = candidate
                break

        grvt_mid: Optional[float] = None
        with self._state._lock:
            self._state.binance_best_bid = best_bid
            self._state.binance_best_ask = best_ask
            self._state.binance_mid = mid
            self._state.binance_bid_size = self._last_bid_size
            self._state.binance_ask_size = self._last_ask_size
            self._state.binance_last_message_wall_ts = now
            if self._state.market is not None and self._state.market.mid_price is not None:
                try:
                    grvt_mid = float(self._state.market.mid_price)
                except (TypeError, ValueError):
                    grvt_mid = None
            if grvt_mid is not None and mid > 0:
                basis_raw = grvt_mid - mid
                cur = self._state.binance_basis_ewma
                if cur is None:
                    self._state.binance_basis_ewma = basis_raw
                else:
                    self._state.binance_basis_ewma = (
                        self._basis_alpha * basis_raw
                        + (1.0 - self._basis_alpha) * cur
                    )
            tracker = self._state.binance_public_ws_timing
        if tracker is not None:
            local_apply_mono_ns = time.monotonic_ns()
            local_apply_wall_ms = int(time.time() * 1000.0)
            try:
                tracker.ingest(
                    local_receive_wall_ms=local_recv_wall_ms,
                    local_receive_mono_ns=local_recv_mono_ns,
                    exchange_ts_ms=exchange_ts_ms,
                    local_apply_wall_ms=local_apply_wall_ms,
                    local_apply_mono_ns=local_apply_mono_ns,
                    seq=None,
                )
            except Exception:
                logger.exception("bybit_public_ws_timing_ingest_failed")

        if self._on_bbo is not None:
            try:
                self._on_bbo()
            except Exception:
                logger.exception("bybit_on_bbo_callback_failed")
