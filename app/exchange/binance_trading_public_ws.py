"""Binance Futures USDM public-WS subscriber for the TRADING venue.

Plan reference: ``plans/20260420-binance-move/plan.md`` Phase 2.

Distinct from :mod:`app.exchange.binance_public_ws` (which serves
cross-venue REFERENCE pricing — it writes to ``state.binance_mid`` /
``state.binance_basis_ewma``). When ``EXCHANGE=binance`` we ARE the
venue, so we need a trading-venue public stream that:

* subscribes to ``<symbol>@bookTicker`` on Binance Futures,
* emits a :class:`BestBidAsk` to the on-bbo callback so
  ``state.market`` updates the same way Bluefin / HL / GRVT do,
* reconnects with exponential backoff,
* implements ``request_reconnect()`` so the bot's market-data
  recovery supervisor can force a reconnect on stalls.

It deliberately does NOT duplicate the cross-venue basis bookkeeping;
that path stays in ``binance_public_ws.py`` and is disabled in the
Binance trading profile via ``REFERENCE_EXCHANGE=off`` (Phase 1).
"""

from __future__ import annotations

import json
import logging
import threading
import time
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
    _websocket_import_error: Optional[Exception] = e
else:
    _websocket_import_error = None


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class BinanceTradingPublicStream:
    """Daemon-thread bookTicker subscriber for the trading venue.

    Lifecycle: ``start()`` spawns the WS thread, which loops on
    connect / receive / disconnect with backoff until ``stop()``.

    Each bookTicker message becomes a ``BestBidAsk`` and is delivered
    via ``on_bbo`` synchronously on the WS thread — the callback must
    be cheap (the bot's ``_on_public_bbo`` does an in-place state
    update + an event-set; same pattern as the other venues).
    """

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        symbol: str,
        on_bbo: Callable[[BestBidAsk], None],
    ) -> None:
        self._settings = settings
        self._state = state
        self._symbol = (symbol or "").strip().upper()
        self._on_bbo = on_bbo
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None
        self._reconnect_attempt: int = 0
        self._force_reconnect = threading.Event()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if websocket is None:
            logger.warning(
                "binance_trading_public_ws_disabled "
                "websocket-client missing: %s",
                _websocket_import_error,
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="binance-trading-public-ws",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("binance_trading_public_ws_close_failed")

    def request_reconnect(self) -> None:
        """Force a reconnect — used by the bot's market-data recovery
        supervisor when the BBO has been stale past the configured
        threshold.
        """
        self._force_reconnect.set()
        with self._ws_lock:
            ws = self._ws_app
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception(
                    "binance_trading_public_ws_force_reconnect_close_failed"
                )

    # ------------------------------------------------------------------
    # WS thread
    # ------------------------------------------------------------------

    def _run_forever(self) -> None:
        backoff_initial = float(
            self._settings.public_ws_reconnect_initial_seconds
        )
        backoff_max = float(self._settings.public_ws_reconnect_max_seconds)
        # Binance accepts the bookTicker stream at a per-symbol path.
        # Base is the trading-venue WS host; ``BINANCE_WS_BASE_URL``
        # already names the public futures stream.
        base = (self._settings.binance_ws_base_url or "").rstrip("/")
        if not base:
            base = "wss://fstream.binance.com/ws"
        url = f"{base}/{self._symbol.lower()}@bookTicker"
        while not self._stop.is_set():
            self._connect_once(url)
            if self._stop.is_set():
                return
            self._reconnect_attempt += 1
            attempt = self._reconnect_attempt
            delay = min(backoff_max, backoff_initial * (2 ** attempt))
            if self._stop.wait(delay):
                return
            self._force_reconnect.clear()

    def _connect_once(self, url: str) -> None:
        try:
            def on_open(_ws: Any) -> None:
                self._reconnect_attempt = 0
                # Mirror the GRVT / HL / Bluefin / OKX contract: flip
                # the ``public_ws_connected`` flag on the bot state so
                # the market-data recovery supervisor sees the WS as
                # live. See ``app/bot.py::_market_data_recovery_supervisor``
                # for the read site. Adding this on 2026-05-05 to close
                # the same omission that bit OKX -- the Binance bot has
                # been trading happily without it (likely because
                # something else in the recovery path didn't trip in
                # practice), but flag accuracy is purely additive and
                # protects against future drift in the supervisor.
                with self._state._lock:
                    self._state.public_ws_connected = True
                logger.info(
                    "binance_trading_public_ws_connected url=%s", url
                )

            def on_message(_ws: Any, raw: Any) -> None:
                self._handle_raw_message(raw)

            def on_error(_ws: Any, err: Any) -> None:
                logger.warning(
                    "binance_trading_public_ws_error err=%s",
                    str(err)[:200],
                )

            def on_close(_ws: Any, code: Any, reason: Any) -> None:
                with self._state._lock:
                    self._state.public_ws_connected = False
                logger.info(
                    "binance_trading_public_ws_disconnected "
                    "code=%s reason=%s",
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
            logger.exception("binance_trading_public_ws_connect_failed")
        finally:
            with self._ws_lock:
                self._ws_app = None
            # Defensive: if run_forever raised before on_close fired,
            # mark disconnected explicitly so the supervisor sees a
            # truthful flag. Mirrors the OKX adapter.
            with self._state._lock:
                self._state.public_ws_connected = False

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_raw_message(self, raw: Any) -> None:
        # Capture the receive-side timestamps as early as possible so the
        # PublicWsTimingTracker's ``exchange_to_local_receive_ms`` and
        # ``receive_to_apply_ms`` measurements include the entire local
        # processing path (parse + on_bbo callback + state-apply).
        local_recv_mono_ns = time.monotonic_ns()
        local_recv_wall_ms = int(time.time() * 1000.0)
        # Stamp the bot state's last-message wall clock so the
        # market-data recovery supervisor can compute book-age. Mirrors
        # the GRVT / HL / Bluefin / OKX contract; see header comment
        # in on_open for the broader rationale.
        recv_wall_dt = datetime.fromtimestamp(
            local_recv_wall_ms / 1000.0, tz=timezone.utc
        )
        with self._state._lock:
            self._state.public_ws_last_message_wall_ts = recv_wall_dt
        try:
            msg = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            logger.debug(
                "binance_trading_public_ws_unparseable raw=%s",
                str(raw)[:200],
            )
            return
        if not isinstance(msg, dict):
            return
        # Binance bookTicker shape:
        # {
        #   "e": "bookTicker", "u": <updateId>, "s": "DOGEUSDT",
        #   "b": "<bidPrice>", "B": "<bidQty>",
        #   "a": "<askPrice>", "A": "<askQty>",
        #   "T": <transaction time ms>, "E": <event time ms>
        # }
        if msg.get("s") != self._symbol:
            return
        bid = _coerce_float(msg.get("b"))
        ask = _coerce_float(msg.get("a"))
        bid_sz = _coerce_float(msg.get("B"))
        ask_sz = _coerce_float(msg.get("A"))
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10_000.0 if mid > 0 else None
        # Prefer ``E`` (event time on Binance's side) for the one-way
        # latency measurement; ``T`` (transaction/match time) is also
        # exchange-side but a few μs earlier in the pipeline. Fall back
        # to whichever is present.
        ts_ms = msg.get("E") or msg.get("T")
        try:
            ts_exchange_ms = int(ts_ms) if ts_ms is not None else None
        except (TypeError, ValueError):
            ts_exchange_ms = None
        bbo = BestBidAsk(
            symbol=self._symbol,
            best_bid=bid,
            best_ask=ask,
            mid_price=mid,
            spread_bps=spread_bps,
            ts_exchange_ms=ts_exchange_ms,
            bid_size=bid_sz,
            ask_size=ask_sz,
        )
        try:
            self._on_bbo(bbo)
        except Exception:
            logger.exception("binance_trading_public_ws_callback_failed")
        # Feed the public-WS timing tracker so /latency + /health can
        # report rolling p50/p95/max for both
        # ``exchange_to_local_receive_ms`` (network + serialisation
        # one-way latency) and ``receive_to_apply_ms`` (our local
        # processing). Tracker has its own lock, lock-free from the
        # BotState perspective.
        tracker = getattr(self._state, "binance_public_ws_timing", None)
        if tracker is not None:
            try:
                tracker.ingest(
                    local_receive_wall_ms=local_recv_wall_ms,
                    local_receive_mono_ns=local_recv_mono_ns,
                    exchange_ts_ms=ts_exchange_ms,
                    local_apply_wall_ms=int(time.time() * 1000.0),
                    local_apply_mono_ns=time.monotonic_ns(),
                    seq=None,
                )
            except Exception:
                logger.exception(
                    "binance_trading_public_ws_timing_ingest_failed"
                )
