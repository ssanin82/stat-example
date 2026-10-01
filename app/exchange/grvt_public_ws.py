"""GRVT public websocket stream: top-of-book via ``v1.mini.d`` tick-feed.

We deliberately subscribe to the mini-ticker *delta* stream at ``@50`` ms
cadence rather than ``v1.book.s@500-10``. A snapshot every 500 ms puts
``gap_median_ms`` right at 500 ms and ``gap_p95_ms`` comfortably above the
``QUOTE_HOLD_MAX_GAP_P95_MS=600`` freshness gate, which kept the bot pinned
at ``HOLD_ALL`` on GRVT (see ``code_reports/tuning.md``). ``v1.mini.d`` gives
us BBO-only updates, so each message is ~5 fields instead of a 10-level
book — faster to parse and produces a tighter freshness curve.

The delta format sends an initial message with all fields populated and then
only the fields that changed; the stream therefore has to be parsed
statefully. The parser caches the last-seen BBO on the stream object and
emits a new ``BestBidAsk`` whenever either price changes. Size-only deltas
refresh the cached sizes but do not wake the quote loop — quote decisions
key off price.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.inbound_timing import InboundPublicTiming

from app.config import Settings
from app.models import BestBidAsk
from app.state import BotState

logger = logging.getLogger(__name__)

BboCallback = Callable[[BestBidAsk], None]


try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error = e
else:
    _websocket_import_error = None


# Stream + cadence constants. Exposed at module level so the (very small)
# negotiated parameters are visible in one place and easy to tune later.
#
#   stream ``v1.mini.s``: mini-ticker *snapshot* feed — every message carries
#                         all BBO fields. Chosen over ``v1.mini.d`` because
#                         delta feeds go silent whenever the top-of-book does
#                         not change, which on a tight GRVT market
#                         (0.04 bps spread, deep depth) meant ``gap_p95_ms``
#                         drifted above the freshness gate despite the @50 ms
#                         tick we asked for. Snapshot at 200 ms forces a
#                         heartbeat regardless of activity and keeps
#                         ``gap_p95_ms`` pinned around 200–250 ms, well under
#                         ``QUOTE_HOLD_MAX_GAP_P95_MS=600``.
#   cadence ``200`` ms  : fastest throttled rate GRVT supports for ``mini.s``
#                         (valid values: 200, 500, 1000, 5000).
_GRVT_PUBLIC_STREAM = "v1.mini.s"
_GRVT_PUBLIC_RATE_MS = 200


def _default_public_ws_url(env: str) -> str:
    env_lc = (env or "prod").strip().lower()
    if env_lc == "prod":
        return "wss://market-data.grvt.io/ws/full"
    if env_lc == "testnet":
        return "wss://market-data.testnet.grvt.io/ws/full"
    return f"wss://market-data.{env_lc}.gravitymarkets.io/ws/full"


def _normalize_instrument(symbol: str) -> str:
    s = (symbol or "").strip()
    if "_" in s:
        return s
    return f"{s}_USDT_Perp"


class GrvtPublicStream:
    """Dedicated thread: connect to GRVT WS, subscribe to :data:`_GRVT_PUBLIC_STREAM`.

    Holds the stateful last-seen BBO fields for the mini-delta feed. Cache is
    reset on every connect so we never cross a reconnect boundary with stale
    data (a delta right after reconnect is treated as incremental; GRVT
    follows up with an initial snapshot but we don't want to quote against
    partially populated state in the interim).
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
        self._symbol = _normalize_instrument(symbol)
        self._on_bbo = on_bbo
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None
        # Last-seen mini-ticker fields (delta state). None until the first
        # ``best_*_price`` lands. Size fields are tracked for microprice
        # computation downstream (see ``app/quoting.py::compute_microprice``)
        # but don't gate emission — the BBO can emit without sizes, which
        # makes the reservation gracefully fall back to midprice.
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
                "grvt_public_ws_disabled missing_dependency err=%s",
                _websocket_import_error,
            )
            return
        if not self._settings.public_ws_enabled:
            logger.info("grvt_public_ws_start_skipped PUBLIC_WS_ENABLED=false")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="grvt-public-ws",
            daemon=True,
        )
        self._thread.start()
        logger.info("grvt_public_ws_thread_started symbol=%s", self._symbol)

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("grvt_public_ws_close_error")
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        with self._state._lock:
            self._state.public_ws_connected = False
        logger.info("grvt_public_ws_stopped")

    def request_reconnect(self) -> None:
        with self._ws_lock:
            ws = self._ws_app
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("grvt_public_ws_request_reconnect_close_failed")

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
                logger.exception("grvt_public_ws_session_failed")
            delay_s = initial if delay_s <= 0 else min(max_b, max(delay_s * 2.0, initial))

    def _connect_once(self) -> None:
        assert websocket is not None
        url = (
            self._settings.grvt_public_ws_url
            or _default_public_ws_url(self._settings.grvt_env)
        )
        symbol = self._symbol
        # Mini selector carries no depth token. Format is ``instrument@rate``
        # (matches gravity-technologies/grvt-pysdk ``grvt_ccxt_ws.py``).
        selector = f"{symbol}@{_GRVT_PUBLIC_RATE_MS}"
        self._reset_cache()

        # Trade-print subscription (Priority #3). On the SAME connection
        # — GRVT's JSON-RPC accepts multiple subscribe messages per
        # session. Trade selector uses the bare instrument name with no
        # rate suffix; rate is inherent to the feed (each aggressor
        # trade is its own event).
        trade_stream_enabled = bool(
            self._settings.grvt_trade_stream_enabled
            and getattr(self._settings, "grvt_trade_stream_name", "")
        )
        trade_stream_name = str(getattr(self._settings, "grvt_trade_stream_name", "") or "")

        def on_open(ws: Any) -> None:
            ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "subscribe",
                        "params": {"stream": _GRVT_PUBLIC_STREAM, "selectors": [selector]},
                        "id": 1,
                    }
                )
            )
            if trade_stream_enabled:
                try:
                    ws.send(
                        json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "method": "subscribe",
                                "params": {
                                    "stream": trade_stream_name,
                                    "selectors": [symbol],
                                },
                                "id": 2,
                            }
                        )
                    )
                except Exception:
                    logger.exception("grvt_public_ws_trade_subscribe_failed")
            with self._state._lock:
                self._state.public_ws_connected = True
            try:
                tr = getattr(self._state, "public_ws_timing", None)
                if tr is not None:
                    tr.note_stream_reset("grvt_public_ws_open")
            except Exception:
                pass
            logger.info(
                "grvt_public_ws_connected url=%s bbo_stream=%s selector=%s "
                "trade_stream=%s trade_enabled=%s",
                url,
                _GRVT_PUBLIC_STREAM,
                selector,
                trade_stream_name or "(none)",
                trade_stream_enabled,
            )

        def on_message(_ws: Any, message: str) -> None:
            self._handle_raw_message(message)

        def on_error(_ws: Any, error: Any) -> None:
            logger.warning("grvt_public_ws_on_error %s", error)

        def on_close(_ws: Any, close_status_code: Any, close_msg: Any) -> None:
            with self._state._lock:
                self._state.public_ws_connected = False
            try:
                tr = getattr(self._state, "public_ws_timing", None)
                if tr is not None:
                    tr.note_stream_reset("grvt_public_ws_close")
            except Exception:
                pass
            logger.info(
                "grvt_public_ws_disconnected code=%s msg=%s",
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

        app.run_forever(ping_interval=25, ping_timeout=15)

    def _handle_raw_message(self, message: str) -> None:
        recv_mono = time.perf_counter()
        recv_wall = datetime.now(timezone.utc)
        parse_start = time.perf_counter()
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(
                "grvt_public_ws_malformed_json len=%s snippet=%s",
                len(message),
                message[:240].replace("\n", " "),
            )
            return
        parse_end = time.perf_counter()
        if obj.__class__ is not dict:
            return
        # JSON-RPC subscribe-ack. The method + result shape is identical across
        # GRVT WS endpoints; skip before anything else so acks never touch the
        # hot BBO path.
        if obj.get("jsonrpc") == "2.0" and "result" in obj:
            return
        # Priority #3 — trade-print stream. Branched before the BBO
        # early-return because trades arrive on a different stream name.
        incoming_stream = obj.get("stream")
        if incoming_stream == str(getattr(self._settings, "grvt_trade_stream_name", "") or ""):
            self._handle_trade_message(obj)
            return
        if incoming_stream != _GRVT_PUBLIC_STREAM:
            return
        feed = obj.get("feed")
        if not isinstance(feed, dict):
            return
        if feed.get("instrument") != self._symbol:
            return

        # Mini-delta parse: any of the price fields may be absent on a given
        # message (delta semantics — only changed fields are sent). Merge into
        # cached state and emit iff both bid+ask are known.
        bid_raw = feed.get("best_bid_price")
        ask_raw = feed.get("best_ask_price")
        try:
            if bid_raw is not None:
                self._last_bid_px = float(bid_raw)
            if ask_raw is not None:
                self._last_ask_px = float(ask_raw)
        except (TypeError, ValueError):
            return
        # Best bid/ask sizes — used downstream for the microprice reservation.
        # Not required for emission; malformed or missing values just leave
        # the cache at its previous value, and the reservation falls back to
        # midprice when either side's size is None.
        bid_sz_raw = feed.get("best_bid_size")
        ask_sz_raw = feed.get("best_ask_size")
        try:
            if bid_sz_raw is not None:
                self._last_bid_size = float(bid_sz_raw)
            if ask_sz_raw is not None:
                self._last_ask_size = float(ask_sz_raw)
        except (TypeError, ValueError):
            pass
        bid = self._last_bid_px
        ask = self._last_ask_px
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            # Initial deltas may only set one side; wait until we have both.
            return
        mid = (bid + ask) * 0.5
        spread_bps = (ask - bid) / mid * 10_000.0 if mid > 0 else None
        ts_ns_raw = feed.get("event_time")
        ts_ms: Optional[int]
        if ts_ns_raw is None:
            ts_ms = None
            ts_local = recv_wall
        else:
            try:
                ts_ns = int(ts_ns_raw) if isinstance(ts_ns_raw, int) else int(str(ts_ns_raw))
            except (TypeError, ValueError):
                ts_ns = 0
            if ts_ns > 0:
                ts_ms = ts_ns // 1_000_000
                ts_local = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            else:
                ts_ms = None
                ts_local = recv_wall

        cb_start = time.perf_counter()
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
                callback_start_mono=cb_start,
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
            logger.exception("grvt_public_ws_on_bbo_callback_failed")

    def _handle_trade_message(self, obj: Any) -> None:
        """Parse a ``v1.trade``-stream message and push a ``TradePrint``
        onto ``state.recent_trades`` + feed the flow-score accumulator.
        Silently drops messages that don't match our symbol or have
        malformed / missing required fields (price, size, aggressor).

        Expected GRVT payload shape (subject to verification against
        the production feed — parser is tolerant):

            {
              "stream": "v1.trade",
              "feed": {
                "event_time": "1745000000123456789",   # ns
                "instrument": "SOL_USDT_Perp",
                "is_taker_buyer": true,
                "price": "86.12",
                "size": "0.5",
                "trade_id": "abc123"
              }
            }
        """
        from app.enums import Side
        from app.models import TradePrint

        feed = obj.get("feed")
        if not isinstance(feed, dict):
            return
        if feed.get("instrument") != self._symbol:
            return
        try:
            price = float(feed["price"])
            size = float(feed["size"])
        except (KeyError, TypeError, ValueError):
            return
        if not (price > 0 and size > 0):
            return
        is_taker_buyer = feed.get("is_taker_buyer")
        if is_taker_buyer is None:
            # Some GRVT environments use alternative field names; fall
            # back to "taker_side" if present.
            taker_side_raw = feed.get("taker_side") or feed.get("side")
            if isinstance(taker_side_raw, str):
                is_taker_buyer = taker_side_raw.upper() in ("BUY", "B", "BID")
            else:
                return
        aggressor = Side.BUY if bool(is_taker_buyer) else Side.SELL
        ts_ns_raw = feed.get("event_time")
        ts_ex_ms = 0
        if ts_ns_raw is not None:
            try:
                ts_ns = int(ts_ns_raw) if isinstance(ts_ns_raw, int) else int(str(ts_ns_raw))
                if ts_ns > 0:
                    ts_ex_ms = ts_ns // 1_000_000
            except (TypeError, ValueError):
                ts_ex_ms = 0
        if ts_ex_ms == 0:
            ts_ex_ms = int(time.time() * 1000.0)
        ts_local_ms = int(time.time() * 1000.0)
        trade_id = str(feed.get("trade_id") or "")
        tp = TradePrint(
            ts_exchange_ms=ts_ex_ms,
            ts_local_ms=ts_local_ms,
            price=price,
            size=size,
            aggressor_side=aggressor,
            trade_id=trade_id,
        )
        # Push to state under the same lock used for other state
        # mutations. The flow_score accumulator is updated synchronously
        # so TFI at event time is always available to the bot loop.
        with self._state._lock:
            self._state.recent_trades.append(tp)
            try:
                self._state.flow_score.record_trade(tp)
            except Exception:
                logger.exception("grvt_public_ws_flow_score_record_failed")

    def feed_message_for_tests(self, message: str) -> None:
        self._handle_raw_message(message)


__all__ = ["GrvtPublicStream"]
