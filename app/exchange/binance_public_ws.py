"""Binance public-WS subscriber for cross-venue fair-value reference.

Subscribes to Binance's ``@bookTicker`` stream for a single symbol and
maintains the latest best bid/ask + a rolling EWMA of the GRVT-vs-Binance
basis on ``BotState``. The subscriber is public (no auth), runs on its
own thread with exponential-backoff reconnect, and is independent of
the bot's kill / running state.

Motivation
----------
Binance volume dominates ETH (and most mid-cap perps). On GRVT, BBO
updates lag Binance by 100-500 ms during active tape; arb bots watch
Binance and cross our GRVT quote at the old price before our book
catches up. The overnight 2026-04-19 ETH session showed a clean 1:1
BUY/SELL symmetry and −1.53 bps mean markout — textbook
cross-venue-lag adverse selection.

The operator's friend described the mitigation directly: *"their bot
was cancelling orders on these perp DEXs by looking at market data
from Binance."* This module implements exactly that (Level 1 of the
cross-reference proposal in ``proposals/cross-reference.md``) — the
module itself just maintains the state; the cancel trigger lives in
``app/execution.py::maybe_refresh_quotes``.

Message format (Binance **Futures** ``@bookTicker``, from
https://binance-docs.github.io/apidocs/futures/en/#individual-symbol-book-ticker-streams):

    {
      "e": "bookTicker",    // event type
      "u": 400900217,       // order book updateId
      "s": "ETHUSDT",       // symbol
      "b": "2345.67",       // best bid price (string)
      "B": "10.50",         // best bid qty (string)
      "a": "2345.68",       // best ask price (string)
      "A": "8.20",          // best ask qty (string)
      "T": 1700000000000,   // transaction time (ms)
      "E": 1700000000000    // event time (ms)
    }

The default URL is ``wss://fstream.binance.com/ws`` — the **Futures**
endpoint, which always includes ``E`` (event time) and ``T`` (transaction
time). We parse ``E`` per message and feed it to
``state.binance_public_ws_timing`` so ``/market-data/binance-timing-summary``
can report the one-way latency (Binance engine → our bot host).
Empirically we expected 30-50 ms p50 between Binance Tokyo and a
Singapore host (Railway-era baseline); if the measurement shows much
higher, that's a signal to consider a lower-latency reference (OKX has
Singapore + HK servers). Deploy targets: AWS EC2 is the default
release platform; one OKX-adjacent instance currently runs on Alibaba
HK colo for proximity.
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


def _build_ws_url(base_url: str, symbol: str) -> str:
    """Construct the full bookTicker URL.

    Binance uses lowercase symbols in stream paths (``ethusdt@bookTicker``).
    Callers pass the Binance symbol (e.g. ``ETHUSDT``) separately from
    the GRVT / HL symbol so we can keep operator config clean.
    """
    return f"{base_url.rstrip('/')}/{symbol.strip().lower()}@bookTicker"


class BinancePublicStream:
    """Dedicated thread: connect to Binance ``@bookTicker``, update state."""

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        on_bbo_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self._settings = settings
        self._state = state
        # Typically set to ``state.wake_quote_loop`` so the main quote
        # loop runs promptly after each Binance tick — enables the
        # cancel-on-move trigger to fire within one loop interval.
        self._on_bbo = on_bbo_callback
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None
        self._url = _build_ws_url(
            settings.binance_ws_base_url,
            settings.binance_symbol,
        )
        # Alpha for EWMA of (grvt_mid - binance_mid) basis. At 1 update/sec
        # a typical alpha=0.05 gives ~14-sample (~14 s) effective window —
        # tracks slow funding-premium drift without over-reacting to
        # single-tick jitter.
        self._basis_alpha = float(settings.binance_basis_ewma_alpha)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WS thread if the feature is enabled + dependency
        present. No-op when already running."""
        if websocket is None:
            logger.error(
                "binance_public_ws_disabled missing_dependency err=%s",
                _websocket_import_error,
            )
            return
        if not self._settings.binance_ws_enabled:
            logger.info("binance_public_ws_disabled BINANCE_WS_ENABLED=false")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(target=self._run_forever, name="binance-public-ws", daemon=True)
        self._thread = t
        t.start()
        logger.info("binance_public_ws_thread_started url=%s", self._url)

    def stop(self) -> None:
        """Signal shutdown and join the thread (bounded wait). Safe to
        call multiple times; no-op if never started."""
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.debug("binance_public_ws_close_failed", exc_info=True)
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5.0)
        self._thread = None
        logger.info("binance_public_ws_stopped")

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    def _run_forever(self) -> None:
        """Reconnect with exponential backoff. First connect attempt has
        no delay; subsequent attempts double up to a cap."""
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
                logger.exception("binance_public_ws_session_failed")
            # Mark disconnected between sessions.
            with self._state._lock:
                self._state.binance_ws_connected = False
            delay = initial if delay <= 0 else min(max_backoff, delay * 2.0)

    def _connect_once(self) -> None:
        assert websocket is not None

        def on_open(_ws: Any) -> None:
            with self._state._lock:
                self._state.binance_ws_connected = True
                self._state.binance_ws_last_connect_ts = datetime.now(timezone.utc)
                # 2026-05-12 codex-#6: record which venue is feeding
                # the ``binance_*``/reference fields. Cleared on
                # disconnect by the WS-close path.
                self._state.reference_venue_name = "binance"
                tracker = self._state.binance_public_ws_timing
            # Reset the timing window boundary so the first post-reconnect
            # message doesn't compute a stale multi-second gap against the
            # message from the prior (now defunct) session. Without this,
            # reconnect events would pollute ``exchange_gap_ms.p95`` with
            # huge but meaningless values.
            if tracker is not None:
                tracker.note_stream_reset("binance_public_ws_open")
            logger.info("binance_public_ws_open url=%s", self._url)

        def on_message(_ws: Any, raw: Any) -> None:
            self._handle_raw_message(raw)

        def on_error(_ws: Any, err: Any) -> None:
            logger.warning("binance_public_ws_error err=%s", err)

        def on_close(_ws: Any, close_status_code: Any, close_msg: Any) -> None:
            with self._state._lock:
                self._state.binance_ws_connected = False
            logger.info(
                "binance_public_ws_close code=%s msg=%s",
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
        # ping_interval=20 sends a WebSocket-protocol ping every 20 s;
        # Binance replies with pong. If no pong in 10 s → ``run_forever``
        # returns and the outer reconnect loop retries.
        app.run_forever(ping_interval=20, ping_timeout=10)

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_raw_message(self, raw: Any) -> None:
        """Parse a ``@bookTicker`` payload and push the top-of-book into
        state. Malformed messages are silently dropped (no raises — the
        WS is read-only telemetry, no value in crashing the thread)."""
        if not isinstance(raw, (str, bytes, bytearray)):
            return
        # Capture receive-side time markers as early as possible so the
        # latency figure reflects *arrival* age, not parse/decode age.
        # ``local_apply_mono_ns`` is captured below right before we
        # finish the state update, giving a tight process-internal
        # receive-to-apply window.
        local_recv_mono_ns = time.monotonic_ns()
        local_recv_wall_ms = int(time.time() * 1000.0)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(obj, dict):
            return
        try:
            best_bid = float(obj.get("b"))
            best_ask = float(obj.get("a"))
        except (TypeError, ValueError):
            return
        if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
            return
        # Sizes are informational (captured for later microprice-on-Binance
        # experiments); not required for the cancel trigger.
        try:
            b_sz = obj.get("B")
            a_sz = obj.get("A")
            bid_size = float(b_sz) if b_sz is not None else None
            ask_size = float(a_sz) if a_sz is not None else None
        except (TypeError, ValueError):
            bid_size = None
            ask_size = None
        mid = (best_bid + best_ask) / 2.0
        now = datetime.now(timezone.utc)

        # Binance Futures bookTicker messages carry ``E`` (event time)
        # and ``T`` (transaction time) in ms since epoch. ``E`` is the
        # moment Binance emitted the update; diff to our local receive
        # wall is the one-way delay. Fall back to ``T`` if ``E`` is
        # absent (spot-shaped payload, shouldn't happen on fstream but
        # guards against a future routing change). If neither field is
        # present or parseable, pass ``None`` so the tracker increments
        # its ``missing_exchange_ts_count`` — that counter then tells
        # operators the feed suddenly stopped providing timestamps.
        exchange_ts_ms: Optional[int] = None
        for key in ("E", "T"):
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

        # Update state atomically. Compute basis against the CURRENT
        # GRVT mid (if available) and fold into the EWMA.
        grvt_mid: Optional[float] = None
        with self._state._lock:
            self._state.binance_best_bid = best_bid
            self._state.binance_best_ask = best_ask
            self._state.binance_mid = mid
            self._state.binance_bid_size = bid_size
            self._state.binance_ask_size = ask_size
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
        # ``local_apply`` is captured RIGHT HERE — after the lock is
        # released — so the receive-to-apply measurement includes the
        # whole state-update critical section. Tracker has its own
        # internal lock, so calling ``.ingest()`` here is lock-free
        # from BotState's perspective.
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
                logger.exception("binance_public_ws_timing_ingest_failed")

        # Wake the quote loop so the cancel-on-move check runs within
        # one loop interval. Without the wake, the check runs only on
        # the next normal quote tick (up to ``quote_loop_seconds`` of
        # extra lag), defeating the point of the low-latency Binance
        # feed.
        if self._on_bbo is not None:
            try:
                self._on_bbo()
            except Exception:
                logger.exception("binance_on_bbo_callback_failed")
