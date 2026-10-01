"""Bluefin Pro public WebSocket stream (BBO ticker + trade prints).

Migrated to pro-sdk layout:

  * URL: ``wss://stream.api.{env}.bluefin.io/ws/market``
  * Subscribe message shape::

        {
          "method": "Subscribe",
          "dataStreams": [
            {"symbol": "SUI-PERP", "streams": ["Ticker", "Recent_Trade"]}
          ]
        }

  * Message envelope::

        {"event": "TickerUpdate", "payload": {...TickerUpdate fields...}}

The ``TickerUpdate`` payload carries ``bestBidPriceE9`` / ``bestAskPriceE9`` +
corresponding ``*QuantityE9`` fields — everything we need for BBO without
keeping a client-side orderbook. ``RecentTradesUpdates`` carries a
``trades`` array per the shared ``Trade`` schema.

All numeric fields are in 1e9 base as decimal strings.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Optional

from app.config import Settings
from app.exchange.bluefin_client import _normalize_bluefin_symbol
from app.inbound_timing import InboundPublicTiming
from app.models import BestBidAsk
from app.state import BotState

logger = logging.getLogger(__name__)

BboCallback = Callable[[BestBidAsk], None]

try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error: Optional[Exception] = e
else:
    _websocket_import_error = None


_DEFAULT_HOST = "wss://stream.api.sui-prod.bluefin.io"
_MARKET_PATH = "/ws/market"


def _from_e9(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        return float(Decimal(str(raw)) / (Decimal(10) ** 9))
    except Exception:
        return 0.0


def _resolve_market_ws_url(settings: Settings) -> str:
    raw = (settings.bluefin_public_ws_url or _DEFAULT_HOST).rstrip("/")
    if raw.endswith(_MARKET_PATH):
        return raw
    return raw + _MARKET_PATH


class BluefinPublicStream:
    """Dedicated thread; subscribes to Ticker + Recent_Trade for one symbol."""

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        symbol: str,
        on_bbo: BboCallback,
    ) -> None:
        self._settings = settings
        self._state = state
        self._symbol = _normalize_bluefin_symbol(symbol)
        self._on_bbo = on_bbo
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None
        self._last_bid_px: Optional[float] = None
        self._last_ask_px: Optional[float] = None
        self._last_bid_size: Optional[float] = None
        self._last_ask_size: Optional[float] = None

    def _reset_cache(self) -> None:
        self._last_bid_px = None
        self._last_ask_px = None
        self._last_bid_size = None
        self._last_ask_size = None

    def start(self) -> None:
        if websocket is None:
            logger.error(
                "bluefin_public_ws_disabled missing_dependency err=%s",
                _websocket_import_error,
            )
            return
        if not self._settings.public_ws_enabled:
            logger.info("bluefin_public_ws_start_skipped PUBLIC_WS_ENABLED=false")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="bluefin-public-ws",
            daemon=True,
        )
        self._thread.start()
        logger.info("bluefin_public_ws_thread_started symbol=%s", self._symbol)

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("bluefin_public_ws_close_error")
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        with self._state._lock:
            self._state.public_ws_connected = False
        logger.info("bluefin_public_ws_stopped")

    def request_reconnect(self) -> None:
        with self._ws_lock:
            ws = self._ws_app
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("bluefin_public_ws_request_reconnect_close_failed")

    def _run_forever(self) -> None:
        initial = float(self._settings.public_ws_reconnect_initial_seconds)
        max_b = float(self._settings.public_ws_reconnect_max_seconds)
        delay_s = 0.0
        first = True
        attempt = 0
        while not self._stop.is_set():
            if not first and delay_s > 0:
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
                logger.exception("bluefin_public_ws_session_failed")
            delay_s = initial if delay_s <= 0 else min(max_b, max(delay_s * 2.0, initial))

    def _connect_once(self) -> None:
        assert websocket is not None
        url = _resolve_market_ws_url(self._settings)
        symbol = self._symbol
        self._reset_cache()

        def on_open(ws: Any) -> None:
            subscribe = {
                "method": "Subscribe",
                "dataStreams": [
                    {
                        "symbol": symbol,
                        # Ticker carries BBO + mark/oracle/open-interest;
                        # Recent_Trade carries trade prints.
                        "streams": ["Ticker", "Recent_Trade"],
                    }
                ],
            }
            try:
                ws.send(json.dumps(subscribe))
            except Exception:
                logger.exception("bluefin_public_ws_subscribe_failed")
            with self._state._lock:
                self._state.public_ws_connected = True
            try:
                tr = getattr(self._state, "public_ws_timing", None)
                if tr is not None:
                    tr.note_stream_reset("bluefin_public_ws_open")
            except Exception:
                pass
            logger.info(
                "bluefin_public_ws_connected url=%s symbol=%s",
                url,
                symbol,
            )

        def on_message(_ws: Any, message: str) -> None:
            self._handle_raw_message(message)

        def on_error(_ws: Any, error: Any) -> None:
            logger.warning("bluefin_public_ws_on_error %s", error)

        def on_close(_ws: Any, close_status_code: Any, close_msg: Any) -> None:
            with self._state._lock:
                self._state.public_ws_connected = False
            try:
                tr = getattr(self._state, "public_ws_timing", None)
                if tr is not None:
                    tr.note_stream_reset("bluefin_public_ws_close")
            except Exception:
                pass
            logger.info(
                "bluefin_public_ws_disconnected code=%s msg=%s",
                close_status_code,
                str(close_msg or "")[:200],
            )

        with self._ws_lock:
            self._ws_app = websocket.WebSocketApp(
                url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            app = self._ws_app

        app.run_forever(ping_interval=280, ping_timeout=30)

    def _handle_raw_message(self, message: str) -> None:
        recv_mono = time.perf_counter()
        recv_wall = datetime.now(timezone.utc)
        parse_start = time.perf_counter()
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(
                "bluefin_public_ws_malformed_json len=%s snippet=%s",
                len(message),
                message[:240].replace("\n", " "),
            )
            return
        parse_end = time.perf_counter()
        if not isinstance(obj, dict):
            return

        # Subscription-ack frames carry {"success":true|false,"message":"..."}
        if "event" not in obj and "success" in obj:
            if obj.get("success") is False:
                logger.warning(
                    "bluefin_public_ws_subscription_failed message=%s",
                    str(obj.get("message") or "")[:500],
                )
            return

        event_name = str(obj.get("event") or obj.get("eventName") or "")
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return

        sym_from_event = str(payload.get("symbol") or "").upper()
        if sym_from_event and sym_from_event != self._symbol:
            return

        if event_name == "TickerUpdate":
            self._handle_ticker(payload, recv_mono, recv_wall, parse_start, parse_end)
            return

        if event_name in ("OrderbookPartialDepthUpdate", "OrderbookDiffDepthUpdate"):
            self._handle_depth(payload, recv_mono, recv_wall, parse_start, parse_end)
            return

        if event_name == "RecentTradesUpdates":
            self._handle_trades(payload)
            return

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_ticker(
        self,
        data: dict[str, Any],
        recv_mono: float,
        recv_wall: datetime,
        parse_start: float,
        parse_end: float,
    ) -> None:
        bid = _from_e9(data.get("bestBidPriceE9"))
        ask = _from_e9(data.get("bestAskPriceE9"))
        bid_size = _from_e9(data.get("bestBidQuantityE9"))
        ask_size = _from_e9(data.get("bestAskQuantityE9"))
        if bid > 0:
            self._last_bid_px = bid
            self._last_bid_size = bid_size if bid_size > 0 else self._last_bid_size
        if ask > 0:
            self._last_ask_px = ask
            self._last_ask_size = ask_size if ask_size > 0 else self._last_ask_size
        self._maybe_emit_bbo(data, recv_mono, recv_wall, parse_start, parse_end)

    def _handle_depth(
        self,
        data: dict[str, Any],
        recv_mono: float,
        recv_wall: datetime,
        parse_start: float,
        parse_end: float,
    ) -> None:
        bids = data.get("bidsE9")
        asks = data.get("asksE9")
        try:
            if isinstance(bids, list) and bids and isinstance(bids[0], (list, tuple)):
                px = _from_e9(bids[0][0])
                if px > 0:
                    self._last_bid_px = px
                    if len(bids[0]) > 1:
                        sz = _from_e9(bids[0][1])
                        if sz > 0:
                            self._last_bid_size = sz
            if isinstance(asks, list) and asks and isinstance(asks[0], (list, tuple)):
                px = _from_e9(asks[0][0])
                if px > 0:
                    self._last_ask_px = px
                    if len(asks[0]) > 1:
                        sz = _from_e9(asks[0][1])
                        if sz > 0:
                            self._last_ask_size = sz
        except (TypeError, ValueError, IndexError):
            return
        self._maybe_emit_bbo(data, recv_mono, recv_wall, parse_start, parse_end)

    def _maybe_emit_bbo(
        self,
        data: dict[str, Any],
        recv_mono: float,
        recv_wall: datetime,
        parse_start: float,
        parse_end: float,
    ) -> None:
        bid = self._last_bid_px
        ask = self._last_ask_px
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) * 0.5
        spread_bps = (ask - bid) / mid * 10_000.0 if mid > 0 else None
        ts_raw = data.get("updatedAtMillis") or data.get("lastTimeAtMillis")
        ts_ms: Optional[int]
        try:
            ts_ms = int(ts_raw) if ts_raw is not None else None
        except (TypeError, ValueError):
            ts_ms = None
        ts_local = (
            datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            if (ts_ms and ts_ms > 0)
            else recv_wall
        )
        bb = BestBidAsk(
            symbol=self._symbol,
            best_bid=bid,
            best_ask=ask,
            mid_price=mid,
            spread_bps=spread_bps,
            ts_exchange_ms=ts_ms,
            ts_local=ts_local,
            inbound_public_timing=InboundPublicTiming(
                ws_recv_mono=recv_mono,
                ws_recv_wall=recv_wall,
                parse_start_mono=parse_start,
                parse_end_mono=parse_end,
                callback_start_mono=time.perf_counter(),
                apply_mono=0.0,
                exchange_ts_ms=ts_ms,
            ),
            bid_size=self._last_bid_size,
            ask_size=self._last_ask_size,
        )
        with self._state._lock:
            self._state.public_ws_last_message_wall_ts = recv_wall
        try:
            self._on_bbo(bb)
        except Exception:
            logger.exception("bluefin_public_ws_on_bbo_callback_failed")

    def _handle_trades(self, data: dict[str, Any]) -> None:
        from app.enums import Side
        from app.models import TradePrint

        trades = data.get("trades")
        if not isinstance(trades, list):
            return
        local_ms = int(time.time() * 1000.0)
        for t in trades:
            if not isinstance(t, dict):
                continue
            try:
                price = _from_e9(t.get("priceE9"))
                size = _from_e9(t.get("quantityE9"))
            except Exception:
                continue
            if not (price > 0 and size > 0):
                continue
            # Pro-sdk Trade.side is LONG / SHORT (per TradeSide enum).
            side_str = str(t.get("side") or "").upper()
            if side_str == "LONG":
                aggressor = Side.BUY
            elif side_str == "SHORT":
                aggressor = Side.SELL
            else:
                continue
            ts_raw = t.get("executedAtMillis") or 0
            try:
                ts_ex_ms = int(ts_raw)
            except (TypeError, ValueError):
                ts_ex_ms = 0
            if ts_ex_ms == 0:
                ts_ex_ms = local_ms
            tp = TradePrint(
                ts_exchange_ms=ts_ex_ms,
                ts_local_ms=local_ms,
                price=price,
                size=size,
                aggressor_side=aggressor,
                trade_id=str(t.get("id") or ""),
            )
            with self._state._lock:
                self._state.recent_trades.append(tp)
                try:
                    self._state.flow_score.record_trade(tp)
                except Exception:
                    logger.exception("bluefin_public_ws_flow_score_record_failed")

    def feed_message_for_tests(self, message: str) -> None:
        self._handle_raw_message(message)


__all__ = ["BluefinPublicStream"]
