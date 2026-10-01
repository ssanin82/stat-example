"""Binance Futures USDM private user-data WebSocket stream.

Plan reference: ``plans/20260420-binance-move/plan.md`` Phase 2.2.

Binance's user-data WS layout:

  * Spawn the listenKey via REST: ``POST /fapi/v1/listenKey`` (uses
    only the API key header, no signature).
  * WS URL: ``wss://fstream.binance.com/ws/<listenKey>``.
  * Server silently expires listenKeys after 60 min of no PUT
    keepalive. We PUT every 30 min from a daemon thread to double
    the safety margin.
  * Events arrive as JSON objects with ``e`` (event type):
      - ``ORDER_TRADE_UPDATE`` — order placed / filled / cancelled
      - ``ACCOUNT_UPDATE`` — balance + position changes
      - ``MARGIN_CALL`` — pre-liquidation alert (CRITICAL)
      - ``listenKeyExpired`` — listen key invalidated; respawn

Compared to Bluefin's private WS:

  * Auth is dramatically simpler — no JWT minting, no expiry
    tracking on the WS side. The listenKey IS the credential.
  * The keepalive thread does REST PUT, not WS application-level
    pings. Binance's WS protocol-level ping/pong is handled by
    websocket-client.
  * No cancel-confirmation gate (Binance cancels are synchronous
    via REST DELETE, not async WS events).

Out of this stream's scope:
  * The ``_construct_fill`` event-shape mapping converts the
    Binance order-trade-update into a ``PrivateFillEvent`` that
    the bot core ingests via the same path as HL/GRVT/Bluefin.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.config import Settings
from app.enums import Side
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
    PrivateWsConnectionEvent,
    PrivateWsConnectionKind,
)
from app.inbound_timing import InboundPrivateTiming
from app.state import BotState

logger = logging.getLogger(__name__)


DropCallback = Optional[Callable[[int], None]]
ListenKeyProvider = Callable[[], str]
ListenKeyKeepalive = Callable[[], bool]


try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error: Optional[Exception] = e
else:
    _websocket_import_error = None


# Float-coercion helper. Binance returns numbers as decimal strings on
# the WS feed.
def _coerce_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_put(
    q: queue.Queue,
    item: Any,
    label: str,
    on_drop: DropCallback,
) -> None:
    try:
        q.put_nowait(item)
    except queue.Full:
        if on_drop is not None:
            try:
                on_drop(1)
            except Exception:
                logger.exception("binance_private_ws_drop_callback_failed")
        logger.warning("binance_private_ws_queue_drop label=%s", label)


class BinancePrivateStream:
    """Daemon-thread WS subscriber for the Binance user-data stream.

    Lifecycle:
      1. ``start()`` spawns the keepalive thread + the WS thread.
      2. The WS thread spawns a fresh listenKey, connects, parses
         messages, enqueues PrivateFillEvent / PrivateOrderUpdateEvent.
      3. The keepalive thread PUTs ``/fapi/v1/listenKey`` every
         ``BINANCE_LISTEN_KEY_KEEPALIVE_SECONDS`` (default 1800).
      4. On ``listenKeyExpired`` event OR on connection drop, the WS
         thread respawns the key and reconnects with backoff.
      5. ``stop()`` signals both threads to exit and closes the WS.
    """

    def __init__(
        self,
        settings: Settings,
        out_queue: queue.Queue,
        listen_key_provider: ListenKeyProvider,
        listen_key_keepalive: ListenKeyKeepalive,
        on_queue_drop: DropCallback = None,
        state: Optional[BotState] = None,
    ) -> None:
        self._settings = settings
        self._q = out_queue
        self._provider = listen_key_provider
        self._keepalive_fn = listen_key_keepalive
        self._on_queue_drop = on_queue_drop
        self._state = state
        self._stop = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._ws_app: Any = None
        self._ws_lock = threading.Lock()
        self._listen_key: Optional[str] = None
        # Reconnect backoff state (capped exponential).
        self._reconnect_attempt: int = 0
        self._connected_at: float = 0.0

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        if websocket is None:
            logger.warning(
                "binance_private_ws_disabled websocket-client missing: %s",
                _websocket_import_error,
            )
            return
        self._stop.clear()
        self._ws_thread = threading.Thread(
            target=self._run_forever,
            name="binance-private-ws",
            daemon=True,
        )
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_worker,
            name="binance-private-ws-keepalive",
            daemon=True,
        )
        self._ws_thread.start()
        self._keepalive_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("binance_private_ws_close_failed")

    # ------------------------------------------------------------------
    # WS thread
    # ------------------------------------------------------------------

    def _run_forever(self) -> None:
        """Outer reconnect loop. Each iteration spawns a fresh
        listenKey, connects, runs until disconnect, then backs off.
        """
        backoff_initial = float(
            self._settings.private_ws_reconnect_initial_seconds
        )
        backoff_max = float(self._settings.private_ws_reconnect_max_seconds)
        while not self._stop.is_set():
            try:
                self._listen_key = self._provider()
            except Exception:
                logger.exception("binance_listen_key_spawn_failed")
                self._sleep_with_backoff(backoff_initial, backoff_max)
                continue
            url = self._build_ws_url(self._listen_key)
            self._connect_once(url)
            if self._stop.is_set():
                return
            self._reconnect_attempt += 1
            self._sleep_with_backoff(backoff_initial, backoff_max)

    def _build_ws_url(self, listen_key: str) -> str:
        base = (self._settings.binance_private_ws_url or "").rstrip("/")
        if not base:
            base = "wss://fstream.binance.com"
        # Binance's user-data URL is {base}/ws/{listenKey}
        return f"{base}/ws/{listen_key}"

    def _sleep_with_backoff(self, initial: float, cap: float) -> None:
        attempt = max(0, self._reconnect_attempt)
        delay = min(cap, initial * (2 ** attempt))
        if self._stop.wait(delay):
            return

    def _connect_once(self, url: str) -> None:
        emit = self._emit_conn
        try:
            emit(
                PrivateWsConnectionKind.CONNECTING,
                detail=f"url={url[:80]}",
            )

            def on_open(_ws: Any) -> None:
                self._connected_at = time.time()
                self._reconnect_attempt = 0
                emit(PrivateWsConnectionKind.CONNECTED)
                logger.info("binance_private_ws_connected")

            def on_message(_ws: Any, raw: Any) -> None:
                self._handle_raw_message(raw)

            def on_error(_ws: Any, err: Any) -> None:
                logger.warning(
                    "binance_private_ws_error err=%s", str(err)[:200]
                )

            def on_close(_ws: Any, code: Any, reason: Any) -> None:
                emit(
                    PrivateWsConnectionKind.DISCONNECTED,
                    detail=f"code={code} reason={reason}",
                )
                logger.info(
                    "binance_private_ws_disconnected code=%s reason=%s",
                    code,
                    str(reason)[:200] if reason else None,
                )

            ws_app = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            with self._ws_lock:
                self._ws_app = ws_app
            ws_app.run_forever(ping_interval=30, ping_timeout=10)
        except Exception:
            logger.exception("binance_private_ws_connect_failed")
        finally:
            with self._ws_lock:
                self._ws_app = None

    # ------------------------------------------------------------------
    # Keepalive thread
    # ------------------------------------------------------------------

    def _keepalive_worker(self) -> None:
        interval = float(
            self._settings.binance_listen_key_keepalive_seconds
        )
        if interval <= 0:
            return
        while not self._stop.wait(interval):
            try:
                ok = self._keepalive_fn()
                if not ok:
                    logger.warning(
                        "binance_listen_key_keepalive_failed; "
                        "WS thread will respawn on next disconnect"
                    )
                else:
                    logger.debug("binance_listen_key_keepalive_ok")
            except Exception:
                logger.exception("binance_listen_key_keepalive_exception")

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _emit_conn(
        self,
        kind: PrivateWsConnectionKind,
        detail: str = "",
        backoff_seconds: float = 0.0,
    ) -> None:
        _safe_put(
            self._q,
            PrivateWsConnectionEvent(
                kind=kind,
                detail=detail,
                backoff_seconds=backoff_seconds,
            ),
            "connection",
            self._on_queue_drop,
        )

    def _handle_raw_message(self, raw: Any) -> None:
        try:
            msg = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            logger.warning(
                "binance_private_ws_unparseable raw=%s", str(raw)[:200]
            )
            return
        if not isinstance(msg, dict):
            return
        event = msg.get("e")
        if event == "ORDER_TRADE_UPDATE":
            self._handle_order_trade_update(msg)
        elif event == "ACCOUNT_UPDATE":
            # ACCOUNT_UPDATE carries balance + position deltas. The bot
            # already periodically refreshes account state via REST
            # (refresh_account_only); we treat the WS event as a
            # liveness heartbeat and a wake signal but don't separately
            # parse it. If the operator wants real-time account-side
            # bookkeeping, this is the seam for it.
            self._wake_quote_loop_safely()
        elif event == "MARGIN_CALL":
            logger.critical(
                "binance_margin_call_received: %s", str(msg)[:500]
            )
        elif event == "listenKeyExpired":
            logger.warning(
                "binance_listen_key_expired — reconnecting with fresh key"
            )
            with self._ws_lock:
                ws = self._ws_app
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    logger.exception("binance_listen_key_expired_close_failed")
        # Unknown events: ignore silently (Binance adds new events
        # over time; spamming logs on every one would be noisy).

    def _handle_order_trade_update(self, msg: dict[str, Any]) -> None:
        """Parse an ORDER_TRADE_UPDATE event and emit a
        PrivateOrderUpdateEvent (always) plus a PrivateFillEvent when
        the execution type is TRADE.

        Wire shape (subset we use):

            {
              "e": "ORDER_TRADE_UPDATE",
              "E": <event time ms>,
              "T": <transaction time ms>,
              "o": {
                "s": "DOGEUSDT",
                "c": "<clientOrderId>",
                "S": "BUY"|"SELL",
                "o": "LIMIT"|"MARKET"|...,
                "x": "TRADE"|"NEW"|"CANCELED"|"EXPIRED"|...,
                "X": "NEW"|"PARTIALLY_FILLED"|"FILLED"|"CANCELED"|...,
                "i": <orderId int>,
                "l": "<last fill qty>",
                "z": "<cumulative fill qty>",
                "L": "<last fill price>",
                "n": "<commission>",
                "N": "<commission asset>",
                "T": <trade time ms>,
                "t": <tradeId int>,
                "rp": "<realizedPnl>"
              }
            }
        """
        o = msg.get("o") or {}
        if not isinstance(o, dict):
            return
        # Coin field on the event dataclass mirrors the bot core's
        # venue-neutral naming (HL legacy). For Binance the symbol is
        # ``DOGEUSDT`` etc; we pass it through verbatim.
        coin = str(o.get("s") or "").upper()
        cloid = str(o.get("c") or "")
        side_str = str(o.get("S") or "").upper()
        # The dataclass takes a plain string side; existing parsers
        # accept "BUY"/"SELL" or HL's "B"/"A". We preserve Binance's
        # native form so logging is unambiguous.
        try:
            order_id = int(o.get("i") or 0)
        except (TypeError, ValueError):
            order_id = 0
        exec_type = str(o.get("x") or "").upper()
        order_status = str(o.get("X") or "").upper()
        ts_ms = int(msg.get("E") or 0)
        # InboundPrivateTiming carries lifecycle clocks; we set the
        # exchange event timestamp + the WS-receive monotonic so
        # downstream latency telemetry has something to plot.
        timing = InboundPrivateTiming(
            ws_recv_mono=time.monotonic(),
            exchange_event_ms=ts_ms,
        )

        # Order state fields — Binance's WS gives us the cumulative
        # filled qty (``z``) and the original qty (``q``).
        cum_filled = _coerce_float(o.get("z"), default=0.0)
        orig_sz = _coerce_float(o.get("q"), default=0.0)
        remaining = max(0.0, orig_sz - cum_filled)
        limit_px = _coerce_float(o.get("p"), default=0.0)
        status_ts_ms = int(o.get("T") or ts_ms)

        order_update = PrivateOrderUpdateEvent(
            oid=order_id,
            coin=coin,
            status=order_status,
            status_timestamp_ms=status_ts_ms,
            side=side_str,
            limit_px=limit_px,
            remaining_sz=remaining,
            orig_sz=orig_sz,
            raw_status=order_status,
            inbound_timing=timing,
            cloid=cloid or None,
        )
        _safe_put(
            self._q,
            order_update,
            "order_update",
            self._on_queue_drop,
        )

        # Fill event — only when execType is TRADE.
        if exec_type == "TRADE":
            try:
                trade_id = int(o.get("t") or 0)
            except (TypeError, ValueError):
                trade_id = 0
            last_qty = _coerce_float(o.get("l"), default=0.0)
            last_px = _coerce_float(o.get("L"), default=0.0)
            commission = abs(_coerce_float(o.get("n"), default=0.0))
            realized = _coerce_float(o.get("rp"), default=0.0)
            trade_time = int(o.get("T") or ts_ms)
            # Binance reports m=true for maker, m=false for taker on the
            # trade leg; our PrivateFillEvent.crossed is "did this fill
            # cross the book", which is the inverse of maker.
            maker = bool(o.get("m"))
            crossed = not maker
            if last_qty > 0 and last_px > 0:
                fill = PrivateFillEvent(
                    fill_id=str(trade_id),
                    oid=order_id,
                    coin=coin,
                    px=last_px,
                    sz=last_qty,
                    side=side_str,
                    time_ms=trade_time,
                    fee=commission,
                    closed_pnl=realized,
                    crossed=crossed,
                    is_snapshot=False,
                    raw=dict(msg),
                    inbound_timing=timing,
                )
                _safe_put(
                    self._q,
                    fill,
                    "fill",
                    self._on_queue_drop,
                )
        self._wake_quote_loop_safely()

    def _wake_quote_loop_safely(self) -> None:
        if self._state is None:
            return
        try:
            self._state.wake_quote_loop()
        except Exception:
            logger.exception("binance_private_ws_wake_failed")
