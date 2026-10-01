"""OKX V5 public-WS subscriber for the TRADING venue (BBO).

Plan reference: ``plans/20260504-okx-setup/plan.md`` Phase 2.

Mirrors :mod:`app.exchange.binance_trading_public_ws` shape: a small
daemon-thread WS subscriber that yields :class:`BestBidAsk` to a
callback. Subscribes to OKX's ``books5`` channel (top-5 levels,
throttled to 100ms cadence). Available to all users regardless of
VIP level.

Channel choice rationale (2026-05-05): the tighter ``bbo-tbt``
channel (per-tick BBO, 1-3ms cadence) requires VIP5+ trading
volume -- a gate separate from the MM-tier rate-limit
upgrade. Subscribing to bbo-tbt without VIP5 returns a SUBSCRIBE
ack but no data frames; the bot sits forever in STARTING with no
visible error (Telegram only fires on kill events). Reverted to
books5 + freshness-gate widening in the OKX profile. See bug-019
for the full incident.

The handler accepts both books5 and bbo-tbt frames -- the JSON
shape is identical for the fields we consume (bids[0], asks[0],
ts). If/when the operator's OKX account reaches VIP5, switching
back to bbo-tbt is a one-line change.

OKX BBO frame shape (subset):

    {
      "arg": {"channel": "books5", "instId": "DOGE-USDT-SWAP"},
      "data": [
        {
          "asks": [["px", "sz", "...", "..."], ...],   # 5 levels for books5
          "bids": [["px", "sz", "...", "..."], ...],
          "ts": "1727328000000",
          "instId": "DOGE-USDT-SWAP",
          "seqId": 12345
        }
      ]
    }

Sizes are in OKX CONTRACTS, so the public stream needs the same
contract-value -> base-units conversion the OKX REST adapter uses.
We accept it via ``set_contract_value`` after the OkxClient bootstraps
its symbol spec (avoids a circular import).

Keepalive: same as the private WS -- send literal text ``"ping"`` every
~25s; server replies ``"pong"``. websocket-client's protocol-level
``ping_interval`` is also enabled.
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


class OkxPublicStream:
    """Daemon-thread books5 subscriber for the trading venue.

    Lifecycle: ``start()`` spawns the WS thread, which loops on
    connect / receive / disconnect with backoff until ``stop()``.

    Each books5 message becomes a ``BestBidAsk`` and is delivered via
    ``on_bbo`` synchronously on the WS thread -- the callback must be
    cheap (matches the bot's contract for the other venues).
    """

    _PING_INTERVAL_SEC = 25.0

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
        self._ping_thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None
        self._reconnect_attempt: int = 0
        self._force_reconnect = threading.Event()
        # Set externally by main.py after OkxClient bootstrap.
        self._contract_value: float = 1.0
        # Book-channel auto-detection (1.1.46). On every fresh
        # connection we optimistically subscribe to ``bbo-tbt`` —
        # event-driven L1 BBO with ~38ms median cadence on HYPE per
        # the 2026-05-08 ``scripts/okx_probe_book_channel.py`` probe.
        # If no data frame arrives within ``_FALLBACK_TIMEOUT_SEC``,
        # we silently fall back to ``books5`` (100ms throttle,
        # universally available). The fallback covers the bug-019
        # silent-failure pattern: a VIP-gated channel may subscribe-ack
        # without ever streaming data; the timer catches that.
        #
        # The deeper L2-tbt channels (``books-l2-tbt`` /
        # ``books50-l2-tbt``) would give us more depth at similar
        # cadence, but they require an authenticated public WS
        # (``code=60011 "Please log in"`` on anonymous subscribes per
        # the same probe). That's a bigger code change — deferred.
        #
        # Once we've fallen back in a session, we stay on ``books5``
        # for the rest of the session — no point retrying the fast
        # channel on every reconnect when we know it doesn't stream
        # for this account+IP.
        self._desired_book_channel = "bbo-tbt"
        self._active_book_channel: str = ""
        # BUG-028: when the primary channel is the event-driven bbo-tbt,
        # we ADDITIONALLY subscribe to books5 on the SAME socket as a
        # liveness heartbeat (routed to ``state.note_book_heartbeat``) so
        # the freshness gate's book-age clock keeps ticking in quiet
        # markets when bbo-tbt goes silent. ``None`` when there is no
        # separate heartbeat channel (i.e. we've fallen back to books5 as
        # the primary feed). Read lock-free on the WS thread — same GIL-
        # atomic treatment as ``_active_book_channel``.
        self._heartbeat_channel: Optional[str] = None
        self._first_book_data_event = threading.Event()
        self._fallback_attempted: bool = False
        self._fallback_timer: Optional[threading.Timer] = None

    _FALLBACK_TIMEOUT_SEC = 5.0
    # Channel names we accept BBO data frames on. ``bbo-tbt`` kept for
    # historical reasons; we don't subscribe to it but the parser
    # accepts its frame shape.
    _BOOK_CHANNELS = ("books5", "books50-l2-tbt", "books-l2-tbt", "bbo-tbt")

    def set_contract_value(self, contract_value: float) -> None:
        if contract_value > 0:
            self._contract_value = float(contract_value)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if websocket is None:
            logger.warning(
                "okx_public_ws_disabled websocket-client missing: %s",
                _websocket_import_error,
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="okx-public-ws",
            daemon=True,
        )
        self._ping_thread = threading.Thread(
            target=self._ping_worker,
            name="okx-public-ws-ping",
            daemon=True,
        )
        self._thread.start()
        self._ping_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._fallback_timer is not None:
            try:
                self._fallback_timer.cancel()
            except Exception:
                pass
            self._fallback_timer = None
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("okx_public_ws_close_failed")

    def request_reconnect(self) -> None:
        """Force a reconnect -- used by the bot's market-data recovery
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
                logger.exception("okx_public_ws_force_reconnect_close_failed")

    # ------------------------------------------------------------------
    # WS thread
    # ------------------------------------------------------------------

    def _run_forever(self) -> None:
        backoff_initial = float(
            self._settings.public_ws_reconnect_initial_seconds
        )
        backoff_max = float(self._settings.public_ws_reconnect_max_seconds)
        url = (self._settings.okx_public_ws_url or "").strip()
        if not url:
            logger.warning("okx_public_ws_no_url -- not starting")
            return
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
                # Mirror the GRVT / HL / Bluefin contract: flip the
                # ``public_ws_connected`` flag on the bot state so the
                # market-data recovery supervisor (``app/bot.py``) sees
                # the WS as live. Missing this update was the cause of
                # the OKX bot getting stuck in RECOVERING_MARKET_DATA
                # the moment TRADING_ENABLED flipped to true.
                with self._state._lock:
                    self._state.public_ws_connected = True
                logger.info("okx_public_ws_connected url=%s", url)
                self._send_subscribe()

            def on_message(_ws: Any, raw: Any) -> None:
                self._handle_raw_message(raw)

            def on_error(_ws: Any, err: Any) -> None:
                logger.warning(
                    "okx_public_ws_error err=%s", str(err)[:200]
                )

            def on_close(_ws: Any, code: Any, reason: Any) -> None:
                with self._state._lock:
                    self._state.public_ws_connected = False
                logger.info(
                    "okx_public_ws_disconnected code=%s reason=%s",
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
            logger.exception("okx_public_ws_connect_failed")
        finally:
            with self._ws_lock:
                self._ws_app = None
            # Defensive: if run_forever raised before on_close fired
            # (e.g. transport-level exception), mark disconnected here
            # so the supervisor sees an accurate flag.
            with self._state._lock:
                self._state.public_ws_connected = False

    def _send_subscribe(self) -> None:
        with self._ws_lock:
            ws = self._ws_app
        if ws is None:
            return
        # Channel selection (1.1.46):
        #
        #  - ``books50-l2-tbt`` — top-50 L2 tick-by-tick, ~10ms cadence.
        #    Gated by VIP4+ on OKX. the partner's MM-tier sub-accounts may or
        #    may not have inheritance enabled; we don't know without
        #    asking, so we test by subscribing optimistically.
        #  - ``books5`` — top-5 L2, 100ms-throttled. Available to all.
        #
        # First connection of a session: try ``books50-l2-tbt``. If no
        # data frame arrives within ``_FALLBACK_TIMEOUT_SEC`` (the
        # bug-019 silent-failure pattern: subscribe-ack with no data
        # for VIP-gated channels), fall back to ``books5`` and remember
        # that for the rest of the session.
        #
        # We also subscribe to ``trades`` (public prints) on the same
        # WS — feeds ``BotState.flow_score``. Trades has no VIP gate.
        primary = (
            "books5" if self._fallback_attempted else self._desired_book_channel
        )
        self._active_book_channel = primary
        self._first_book_data_event.clear()
        args = [{"channel": primary, "instId": self._symbol}]
        # BUG-028: when the primary is the event-driven bbo-tbt (which
        # goes silent in quiet markets), ALSO subscribe books5 on this
        # same socket as a liveness heartbeat. books5 pushes on ANY top-5
        # change every 100 ms — far more often than top-1-only bbo-tbt —
        # so it keeps the freshness gate's book-age clock fresh even when
        # the touch is stable. When we've already fallen back to books5 as
        # the primary feed there is no separate heartbeat channel.
        if primary != "books5":
            args.append({"channel": "books5", "instId": self._symbol})
            self._heartbeat_channel = "books5"
        else:
            self._heartbeat_channel = None
        # Public prints (feeds ``BotState.flow_score``); no VIP gate.
        args.append({"channel": "trades", "instId": self._symbol})
        frame = {"op": "subscribe", "args": args}
        try:
            ws.send(json.dumps(frame))
        except Exception:
            logger.exception("okx_public_ws_subscribe_send_failed")
            return
        logger.info(
            "okx_public_ws_subscribe_sent channel=%s heartbeat=%s instId=%s",
            primary,
            self._heartbeat_channel,
            self._symbol,
        )
        # Arm fallback timer ONLY if we just tried the fast channel
        # and haven't yet fallen back this session.
        if primary != "books5" and not self._fallback_attempted:
            self._arm_fallback_timer()

    def _arm_fallback_timer(self) -> None:
        """Schedule a one-shot fallback check. If no book-data frame
        arrives within the timeout, switch to ``books5``.
        """
        if self._fallback_timer is not None:
            try:
                self._fallback_timer.cancel()
            except Exception:
                pass
        t = threading.Timer(
            self._FALLBACK_TIMEOUT_SEC, self._on_fallback_check
        )
        t.daemon = True
        self._fallback_timer = t
        t.start()

    def _on_fallback_check(self) -> None:
        """Fires ~5s after subscribe. If we got data, no-op. If not,
        we've hit the bug-019 silent-failure pattern — unsubscribe the
        fast channel and re-subscribe to ``books5``.
        """
        if self._stop.is_set():
            return
        if self._first_book_data_event.is_set():
            return
        if self._fallback_attempted:
            return
        logger.warning(
            "okx_public_ws_fallback_to_books5 reason=no_data_within_%.1fs "
            "tried_channel=%s — VIP gate likely not satisfied for this account",
            self._FALLBACK_TIMEOUT_SEC,
            self._active_book_channel,
        )
        self._fallback_attempted = True
        with self._ws_lock:
            ws = self._ws_app
        if ws is None:
            return
        # Best-effort unsubscribe; if it fails, books5 subscribe still
        # arrives and OKX will route both. Harmless.
        try:
            ws.send(
                json.dumps(
                    {
                        "op": "unsubscribe",
                        "args": [
                            {
                                "channel": self._active_book_channel,
                                "instId": self._symbol,
                            }
                        ],
                    }
                )
            )
        except Exception:
            logger.exception("okx_public_ws_unsubscribe_failed")
        # Re-subscribe; ``_fallback_attempted=True`` makes
        # ``_send_subscribe`` pick books5.
        self._send_subscribe()

    def _ping_worker(self) -> None:
        while not self._stop.wait(self._PING_INTERVAL_SEC):
            with self._ws_lock:
                ws = self._ws_app
            if ws is None:
                continue
            try:
                ws.send("ping")
            except Exception:
                logger.debug("okx_public_ws_ping_send_failed")

    # ------------------------------------------------------------------
    # Message handling
    # ------------------------------------------------------------------

    def _handle_raw_message(self, raw: Any) -> None:
        # Server's pong is a literal text frame.
        if isinstance(raw, str) and raw.strip() == "pong":
            return
        # Capture receive-side timestamps as early as possible so the
        # PublicWsTimingTracker's exchange_to_local_receive_ms and
        # receive_to_apply_ms cover the full local pipeline (parse +
        # callback + state apply). Mirrors the GRVT / HL / Bluefin
        # pattern -- attach an InboundPublicTiming to the BBO and let
        # ``state.apply_market_book_only`` route it into
        # ``state.public_ws_timing``. No new tracker / endpoint needed.
        recv_mono = time.perf_counter()
        recv_wall = datetime.now(timezone.utc)
        parse_start = time.perf_counter()
        try:
            msg = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            logger.debug(
                "okx_public_ws_unparseable raw=%s", str(raw)[:200]
            )
            return
        parse_end = time.perf_counter()
        if not isinstance(msg, dict):
            return
        # Subscribe ack carries no data.
        if msg.get("event") == "subscribe":
            arg = msg.get("arg") or {}
            logger.info(
                "okx_public_ws_subscribed channel=%s instId=%s",
                arg.get("channel"),
                arg.get("instId"),
            )
            return
        if msg.get("event") == "error":
            logger.error(
                "okx_public_ws_server_error code=%s msg=%s",
                msg.get("code"),
                msg.get("msg"),
            )
            return

        arg = msg.get("arg") or {}
        # Accept any of the L2 / BBO order-book channels — frame shape
        # for the fields we read (bids[0], asks[0], ts, instId) is
        # identical across them; only cadence and depth differ.
        ch = str(arg.get("channel") or "")
        if ch == "trades":
            self._handle_trades_message(msg)
            return
        if ch not in self._BOOK_CHANNELS:
            return
        # 1.1.47 FIX: ``bbo-tbt`` data rows do NOT include ``instId`` —
        # only the parent ``arg`` does. ``books5`` includes it in both
        # places. Check the arg-level instId (always present) rather
        # than the row-level one (missing on bbo-tbt → every frame
        # silently rejected, bot stuck in STARTING).
        if str(arg.get("instId") or "") != self._symbol:
            return
        data = msg.get("data") or []
        if not isinstance(data, list) or not data:
            return
        row = data[0]
        if not isinstance(row, dict):
            return
        bids = row.get("bids") or []
        asks = row.get("asks") or []
        if not bids or not asks:
            return
        bid = _coerce_float(bids[0][0]) if bids[0] else None
        ask = _coerce_float(asks[0][0]) if asks[0] else None
        bid_sz_contracts = _coerce_float(bids[0][1]) if bids[0] else None
        ask_sz_contracts = _coerce_float(asks[0][1]) if asks[0] else None
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return
        # Convert top-of-book sizes from contracts to base units.
        bid_sz = (
            bid_sz_contracts * self._contract_value
            if bid_sz_contracts is not None
            else None
        )
        ask_sz = (
            ask_sz_contracts * self._contract_value
            if ask_sz_contracts is not None
            else None
        )
        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10_000.0 if mid > 0 else None
        ts_raw = row.get("ts")
        try:
            ts_exchange_ms = int(ts_raw) if ts_raw is not None else None
        except (TypeError, ValueError):
            ts_exchange_ms = None
        cb_start = time.perf_counter()
        bbo = BestBidAsk(
            symbol=self._symbol,
            best_bid=bid,
            best_ask=ask,
            mid_price=mid,
            spread_bps=spread_bps,
            ts_exchange_ms=ts_exchange_ms,
            bid_size=bid_sz,
            ask_size=ask_sz,
            inbound_public_timing=InboundPublicTiming(
                ws_recv_mono=recv_mono,
                ws_recv_wall=recv_wall,
                parse_start_mono=parse_start,
                parse_end_mono=parse_end,
                callback_start_mono=cb_start,
                apply_mono=0.0,
                exchange_ts_ms=ts_exchange_ms,
            ),
        )
        # Mirror the GRVT / HL / Bluefin contract: stamp the last-message
        # wall time so ``app/bot.py``'s recovery supervisor can compute
        # ``w_age`` and decide whether the feed is stale. Without this,
        # ``state.public_ws_last_message_wall_ts`` stays None forever
        # and the supervisor forces RECOVERING_MARKET_DATA the moment
        # ``TRADING_ENABLED=true``.
        # Stamped for BOTH the primary channel and the books5 heartbeat
        # (BUG-028): a heartbeat frame is also proof the feed is alive, so
        # it must keep ``w_age`` fresh too — otherwise the recovery
        # supervisor would force a reconnect in a quiet market exactly as
        # the freshness gate used to false-fire.
        with self._state._lock:
            self._state.public_ws_last_message_wall_ts = recv_wall
        # BUG-028 routing. The primary channel (``bbo-tbt`` until a
        # fallback, then ``books5``) drives the full
        # ``apply_market_book_only`` path via ``_on_bbo``. The ``books5``
        # heartbeat (only present while bbo-tbt is primary) drives the
        # narrow ``note_book_heartbeat`` path. After a fallback the primary
        # IS books5 and ``_heartbeat_channel`` is None, so books5 takes the
        # primary branch.
        is_primary = ch == self._active_book_channel
        if is_primary:
            try:
                self._on_bbo(bbo)
            except Exception:
                logger.exception("okx_public_ws_callback_failed")
                return
            # 1.1.47: only flag "first book data" AFTER a successful apply.
            # Setting it earlier (1.1.46) caused the auto-fallback timer
            # to disarm based on parsing alone — but when the parser
            # rejected bbo-tbt frames silently (missing row.instId), the
            # state never updated and the bot stayed in STARTING. By
            # gating on a real apply, the fallback correctly catches
            # "frames arrive but never reach state." Gated to the PRIMARY
            # channel ONLY (BUG-028): a books5 heartbeat must NOT disarm
            # the fallback timer, or a silently-non-streaming bbo-tbt would
            # go undetected (the bug-019 pattern).
            if not self._first_book_data_event.is_set():
                self._first_book_data_event.set()
                logger.info(
                    "okx_public_ws_first_book_data channel=%s instId=%s",
                    ch,
                    self._symbol,
                )
        elif ch == self._heartbeat_channel:
            try:
                self._state.note_book_heartbeat(
                    bbo, market_data_source="public_ws"
                )
            except Exception:
                logger.exception("okx_public_ws_heartbeat_failed")
                return
        else:
            # A book channel we are not actively routing (e.g. a late
            # frame from the bbo-tbt we unsubscribed during fallback, or
            # a books5 frame that arrived before the heartbeat sub was
            # torn down). Ignore — neither feed nor heartbeat.
            return

    def _handle_trades_message(self, msg: dict) -> None:
        """Parse OKX V5 ``trades`` channel frames into ``TradePrint``s
        and feed them into ``BotState.flow_score`` + ``recent_trades``.

        OKX trade payload shape:
            {"arg": {"channel": "trades", "instId": "TON-USDT-SWAP"},
             "data": [{
                "instId": "TON-USDT-SWAP",
                "tradeId": "...",
                "px": "2.700",
                "sz": "1",
                "side": "buy" | "sell",   # taker / aggressor side
                "ts": "1700000000000"      # exchange time, ms
             }, ...]}

        Mirrors GRVT / Bluefin trade-stream feeders: append to the
        bounded deque and the FlowScoreAccumulator. Failures per row
        are swallowed (logged at debug) so a single malformed row does
        not break BBO processing.
        """
        from app.enums import Side
        from app.models import TradePrint

        data = msg.get("data") or []
        if not isinstance(data, list) or not data:
            return
        local_ms = int(time.time() * 1000.0)
        for row in data:
            if not isinstance(row, dict):
                continue
            if row.get("instId") != self._symbol:
                continue
            try:
                px = _coerce_float(row.get("px"))
                sz_contracts = _coerce_float(row.get("sz"))
                if px is None or sz_contracts is None or px <= 0 or sz_contracts <= 0:
                    continue
                # Trade size on OKX is in CONTRACTS — convert to base
                # units for consistency with the BBO sizes used by the
                # rest of the bot.
                size_base = sz_contracts * float(self._contract_value)
                side_raw = str(row.get("side") or "").lower()
                if side_raw == "buy":
                    aggressor = Side.BUY
                elif side_raw == "sell":
                    aggressor = Side.SELL
                else:
                    continue
                ts_raw = row.get("ts")
                try:
                    ts_ex_ms = int(ts_raw) if ts_raw is not None else local_ms
                except (TypeError, ValueError):
                    ts_ex_ms = local_ms
                trade_id = str(row.get("tradeId") or "")
                tp = TradePrint(
                    ts_exchange_ms=ts_ex_ms,
                    ts_local_ms=local_ms,
                    price=px,
                    size=size_base,
                    aggressor_side=aggressor,
                    trade_id=trade_id,
                )
                with self._state._lock:
                    try:
                        self._state.recent_trades.append(tp)
                    except Exception:
                        pass
                    try:
                        self._state.flow_score.record_trade(tp)
                    except Exception:
                        pass
            except Exception:
                logger.debug(
                    "okx_public_ws_trade_parse_failed row=%s", str(row)[:200]
                )
