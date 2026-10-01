"""OKX V5 private user-data WebSocket stream.

Plan reference: ``plans/20260504-okx-setup/plan.md`` Phase 2.2.

OKX private WS layout (different from Binance's listenKey model):

  * Single endpoint per environment:
    ``wss://ws.okx.com:8443/ws/v5/private`` (production).
    Demo trading uses the same host with a path/header marker
    (handled via ``OKX_DEMO_TRADING``).
  * Auth happens via a SIGNED LOGIN FRAME sent over the open WS.
    The sign payload is fixed: timestamp + "GET" + "/users/self/verify".
    Once the server returns ``{"event": "login", "code": "0"}`` the
    connection is authenticated for the lifetime of the socket.
  * After login success, we send one SUBSCRIBE frame asking for the
    ``orders`` + ``positions`` channels filtered to our SWAP symbol.
  * Server pushes:
      - login / subscribe ack frames (``event`` field present)
      - data frames: ``{arg: {channel: ...}, data: [...]}``
      - error frames: ``{event: "error", code: "...", msg: "..."}``
  * Keepalive: the application sends a literal text frame ``"ping"``
    every ~25s; the server replies with ``"pong"``. WebSocket
    protocol-level ping/pong is also enabled via websocket-client's
    ``ping_interval`` arg.

Compared to Binance's private WS:

  * No listenKey lifecycle. WS auth is self-contained on the socket.
  * No keepalive REST PUT thread. The keepalive is application-level
    text frames on the WS itself.
  * No cancel-confirmation gate (OKX cancels are synchronous like
    Binance's).
  * Errors are surfaced inline as event frames rather than through
    a separate channel.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from app.config import Settings
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


try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error: Optional[Exception] = e
else:
    _websocket_import_error = None


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
                logger.exception("okx_private_ws_drop_callback_failed")
        logger.warning("okx_private_ws_queue_drop label=%s", label)


def _okx_login_sign(api_secret: str, timestamp_seconds: str) -> str:
    """OKX login-frame signature.

    The signed payload is the literal string
    ``<timestamp><method=GET><path=/users/self/verify>`` with no body.
    Result is base64-encoded HMAC-SHA256.
    """
    prehash = f"{timestamp_seconds}GET/users/self/verify"
    digest = hmac.new(
        api_secret.encode("utf-8"),
        prehash.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(digest).decode("ascii")


# OKX channels we subscribe to. ``orders`` carries lifecycle updates
# (placed / partially filled / filled / cancelled) AND fill rows
# embedded in each update; ``positions`` carries position-mode deltas
# we treat as a wake signal for the bot's reconcile loop.
_OKX_CHANNEL_ORDERS = "orders"
_OKX_CHANNEL_POSITIONS = "positions"


class OkxPrivateStream:
    """Daemon-thread WS subscriber for the OKX V5 user-data stream.

    Lifecycle:
      1. ``start()`` spawns the WS thread.
      2. The WS thread connects, sends the signed login frame, waits
         for the login ack, then sends the channel subscribe frame.
      3. Incoming data frames are parsed into PrivateFillEvent /
         PrivateOrderUpdateEvent and put on ``out_queue``.
      4. A small ping thread sends ``"ping"`` text frames every 25s
         to keep the server-side idle timer fresh.
      5. On disconnect (server kick, error), the WS thread reconnects
         with capped exponential backoff. Stop set via ``stop()``.
    """

    _PING_INTERVAL_SEC = 25.0

    def __init__(
        self,
        settings: Settings,
        out_queue: queue.Queue,
        on_queue_drop: DropCallback = None,
        state: Optional[BotState] = None,
    ) -> None:
        self._settings = settings
        self._q = out_queue
        self._on_queue_drop = on_queue_drop
        self._state = state
        self._stop = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None
        self._ping_thread: Optional[threading.Thread] = None
        self._ws_app: Any = None
        self._ws_lock = threading.Lock()
        self._reconnect_attempt: int = 0
        self._connected_at: float = 0.0
        self._logged_in = threading.Event()
        # Symbol-spec for parsing OKX's CONTRACT-quantity wire format
        # back to base units. We don't import OkxClient here (would
        # cause a circular dep); the contract value is set externally
        # via set_contract_value() right after factory wiring.
        self._contract_value: float = 1.0
        self._symbol = (settings.symbol or "").strip().upper()

    def set_contract_value(self, contract_value: float) -> None:
        """Inject the OKX contract -> base-unit conversion factor.
        Called once at bot bootstrap by main.py / factory wiring after
        the OkxClient adapter has bootstrapped its symbol spec.
        """
        if contract_value > 0:
            self._contract_value = float(contract_value)

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._ws_thread is not None and self._ws_thread.is_alive():
            return
        if websocket is None:
            logger.warning(
                "okx_private_ws_disabled websocket-client missing: %s",
                _websocket_import_error,
            )
            return
        self._stop.clear()
        self._ws_thread = threading.Thread(
            target=self._run_forever,
            name="okx-private-ws",
            daemon=True,
        )
        self._ping_thread = threading.Thread(
            target=self._ping_worker,
            name="okx-private-ws-ping",
            daemon=True,
        )
        self._ws_thread.start()
        self._ping_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("okx_private_ws_close_failed")

    # ------------------------------------------------------------------
    # WS thread
    # ------------------------------------------------------------------

    def _run_forever(self) -> None:
        backoff_initial = float(
            self._settings.private_ws_reconnect_initial_seconds
        )
        backoff_max = float(self._settings.private_ws_reconnect_max_seconds)
        url = (self._settings.okx_private_ws_url or "").strip()
        if not url:
            logger.warning("okx_private_ws_no_url -- not starting")
            return
        while not self._stop.is_set():
            self._logged_in.clear()
            self._connect_once(url)
            if self._stop.is_set():
                return
            self._reconnect_attempt += 1
            delay = min(backoff_max, backoff_initial * (2 ** self._reconnect_attempt))
            self._sleep_with_backoff(delay)

    def _sleep_with_backoff(self, delay: float) -> None:
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
                # v1.4.95 — match the bluefin_ws / grvt_ws /
                # hyperliquid_ws convention: bump
                # ``private_ws_reconnect_count`` when this isn't the
                # very first connect of the session. Pre-v1.4.95 the
                # OKX path stamped ``private_ws_last_connect_wall_ts``
                # on every login (see ``_handle_raw_message`` event=login
                # branch) but never incremented the counter. Snapshot
                # ``v1.4.92-260519-161411`` exposed the lie:
                # ``private_ws_last_connect_ts`` = 2026-05-19T09:26:23
                # (mid-session) while ``private_ws_reconnect_count``
                # = 0. Operator can no longer tell from the dashboard
                # whether the private WS reconnected during the
                # session. Fixed by checking ``_reconnect_attempt``
                # BEFORE it gets reset: any value >0 means we just
                # came back from a disconnect.
                if self._reconnect_attempt > 0:
                    st = self._state
                    if st is not None:
                        with st._lock:
                            st.private_ws_reconnect_count += 1
                self._reconnect_attempt = 0
                emit(PrivateWsConnectionKind.CONNECTED)
                logger.info("okx_private_ws_connected")
                self._send_login_frame()

            def on_message(_ws: Any, raw: Any) -> None:
                # Stamp last-message timestamp on EVERY frame (including
                # pongs) so a silent-but-still-connected socket is
                # detectable. Mirror of the bluefin_ws on_message hook.
                # Done before _handle_raw_message so even unparseable
                # garbage still bumps the heartbeat clock.
                self._touch_last_message_wall_ts()
                self._handle_raw_message(raw)

            def on_error(_ws: Any, err: Any) -> None:
                logger.warning(
                    "okx_private_ws_error err=%s", str(err)[:200]
                )

            def on_close(_ws: Any, code: Any, reason: Any) -> None:
                emit(
                    PrivateWsConnectionKind.DISCONNECTED,
                    detail=f"code={code} reason={reason}",
                )
                logger.info(
                    "okx_private_ws_disconnected code=%s reason=%s",
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
            logger.exception("okx_private_ws_connect_failed")
        finally:
            with self._ws_lock:
                self._ws_app = None

    # ------------------------------------------------------------------
    # Login + subscribe
    # ------------------------------------------------------------------

    def _send_login_frame(self) -> None:
        api_key = (self._settings.okx_api_key or "").strip()
        api_secret = (self._settings.okx_api_secret or "").strip()
        passphrase = (self._settings.okx_api_passphrase or "").strip()
        if not (api_key and api_secret and passphrase):
            logger.warning(
                "okx_private_ws_login_skipped: credentials missing"
            )
            return
        # OKX login timestamp is Unix epoch seconds (NOT ms, NOT ISO).
        ts = str(int(time.time()))
        sign = _okx_login_sign(api_secret, ts)
        frame = {
            "op": "login",
            "args": [
                {
                    "apiKey": api_key,
                    "passphrase": passphrase,
                    "timestamp": ts,
                    "sign": sign,
                }
            ],
        }
        self._send_frame(frame, label="login")

    def _send_subscribe_frame(self) -> None:
        sym = self._symbol
        frame = {
            "op": "subscribe",
            "args": [
                {
                    "channel": _OKX_CHANNEL_ORDERS,
                    "instType": "SWAP",
                    "instId": sym,
                },
                {
                    "channel": _OKX_CHANNEL_POSITIONS,
                    "instType": "SWAP",
                    "instId": sym,
                },
            ],
        }
        self._send_frame(frame, label="subscribe")

    def _send_frame(self, frame: dict[str, Any], *, label: str) -> None:
        with self._ws_lock:
            ws = self._ws_app
        if ws is None:
            return
        try:
            ws.send(json.dumps(frame))
        except Exception:
            logger.exception("okx_private_ws_send_failed label=%s", label)

    # ------------------------------------------------------------------
    # Ping thread
    # ------------------------------------------------------------------

    def _ping_worker(self) -> None:
        # OKX expects a literal text "ping" every <30s.
        while not self._stop.wait(self._PING_INTERVAL_SEC):
            with self._ws_lock:
                ws = self._ws_app
            if ws is None:
                continue
            try:
                ws.send("ping")
            except Exception:
                logger.debug("okx_private_ws_ping_send_failed")
                continue
            # Stamp ping-sent wall ts so the operator can compare ping
            # vs pong gaps in snapshots. A pong arriving > a few seconds
            # after each ping (OKX's normal RTT is ~50-200 ms from
            # ap-east-1) suggests upstream socket congestion.
            wall = datetime.now(timezone.utc)
            st = self._state
            if st is not None:
                with st._lock:
                    st.private_ws_last_ping_sent_wall_ts = wall

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

    def _touch_last_message_wall_ts(self) -> None:
        """Stamp ``private_ws_last_message_wall_ts`` on every inbound
        frame so the operator's snapshot can compute
        ``private_ws_seconds_since_last_message``. Called from the WS
        ``on_message`` hook before parsing — so even malformed frames
        still register as "the socket is alive."
        """
        st = self._state
        if st is None:
            return
        wall = datetime.now(timezone.utc)
        with st._lock:
            st.private_ws_last_message_wall_ts = wall

    def _bump_orders_msg_counter(self, state_value: str) -> None:
        """Bump per-state OKX orders-channel counters in BotState.
        ``state_value`` is the lowercase OKX state ("live",
        "partially_filled", "filled", "canceled", "mmp_canceled", or
        the raw value if unrecognised). Used by the operator to verify
        the socket is actually delivering order updates — a silent
        socket vs an active one looks identical without these counters.
        """
        st = self._state
        if st is None:
            return
        key = state_value or "unknown"
        with st._lock:
            st.okx_ws_orders_msgs_session += 1
            st.okx_ws_orders_msgs_by_state[key] = (
                st.okx_ws_orders_msgs_by_state.get(key, 0) + 1
            )

    def _record_pong_gap(self) -> None:
        """Record the largest seen ping->pong gap. A gap > a few seconds
        is suspicious on OKX from ap-east-1.
        """
        st = self._state
        if st is None:
            return
        wall = datetime.now(timezone.utc)
        with st._lock:
            st.private_ws_last_pong_wall_ts = wall
            ping_ts = st.private_ws_last_ping_sent_wall_ts
            if ping_ts is not None:
                gap = max(0.0, (wall - ping_ts).total_seconds())
                if gap > st.okx_ws_pong_gap_seconds_max:
                    st.okx_ws_pong_gap_seconds_max = gap

    def _handle_raw_message(self, raw: Any) -> None:
        # Server's pong is a literal text frame, not JSON.
        if isinstance(raw, str) and raw.strip() == "pong":
            self._record_pong_gap()
            return
        try:
            msg = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            logger.warning(
                "okx_private_ws_unparseable raw=%s", str(raw)[:200]
            )
            return
        if not isinstance(msg, dict):
            return

        event = msg.get("event")
        if event == "login":
            code = str(msg.get("code") or "")
            if code == "0":
                logger.info("okx_private_ws_login_ok")
                self._logged_in.set()
                # Mark the WS as genuinely usable. This is the moment
                # we'd actually start receiving order updates, not the
                # bare on_open handshake. Mirrors the convention
                # bluefin_ws uses (it stamps after subscribe-sent).
                wall = datetime.now(timezone.utc)
                st = self._state
                if st is not None:
                    with st._lock:
                        st.private_ws_last_connect_wall_ts = wall
                self._send_subscribe_frame()
            else:
                logger.error(
                    "okx_private_ws_login_failed code=%s msg=%s",
                    code,
                    msg.get("msg"),
                )
            return
        if event == "subscribe":
            arg = msg.get("arg") or {}
            logger.info(
                "okx_private_ws_subscribed channel=%s instId=%s",
                arg.get("channel"),
                arg.get("instId"),
            )
            return
        if event == "error":
            logger.error(
                "okx_private_ws_server_error code=%s msg=%s",
                msg.get("code"),
                msg.get("msg"),
            )
            return

        # Data frame: routed by `arg.channel`.
        arg = msg.get("arg") or {}
        channel = str(arg.get("channel") or "")
        data = msg.get("data") or []
        if channel == _OKX_CHANNEL_ORDERS:
            for row in data:
                if isinstance(row, dict):
                    self._handle_order_row(row)
        elif channel == _OKX_CHANNEL_POSITIONS:
            # Treat as wake signal; the bot reconciles positions
            # periodically via REST. We don't separately parse
            # position deltas here (matches the Binance shape).
            self._wake_quote_loop_safely()

    def _handle_order_row(self, row: dict[str, Any]) -> None:
        """Parse an OKX orders-channel row.

        Wire shape (subset):

            {
              "instId":   "DOGE-USDT-SWAP",
              "ordId":    "1234567890",
              "clOrdId":  "<deterministic>",
              "side":     "buy"|"sell",
              "ordType":  "limit"|"post_only"|"market"|"ioc",
              "px":       "<limit price>",
              "sz":       "<original size in CONTRACTS>",
              "fillSz":   "<last fill size in contracts>",
              "fillPx":   "<last fill price>",
              "tradeId":  "<last fill trade id>",
              "fillFee":  "<last fill fee, signed>",
              "fee":      "<accumulated fee>",
              "fillPnl":  "<realized pnl on this fill>",
              "accFillSz":"<cumulative filled in contracts>",
              "state":    "live"|"partially_filled"|"filled"|"canceled"|"mmp_canceled",
              "uTime":    "<update ts ms>",
              "cTime":    "<create ts ms>",
              "execType": "T"|"M"  -- T=taker, M=maker (per fill)
            }
        """
        try:
            order_id = int(row.get("ordId") or 0)
        except (TypeError, ValueError):
            order_id = 0
        coin = str(row.get("instId") or "").upper()
        cloid = str(row.get("clOrdId") or "")
        side_str = str(row.get("side") or "").lower()
        # Normalise to BUY/SELL upper for the bot's downstream parsers.
        side_norm = "BUY" if side_str == "buy" else "SELL"
        state = str(row.get("state") or "").lower()
        # Diagnostic: per-state OKX orders-channel message counter so
        # the operator can verify the socket is delivering updates
        # commensurate with placed orders. A "silent" socket and a
        # healthy one look identical without this.
        self._bump_orders_msg_counter(state)
        # Phase 2 trace: record this WS event onto the matching order's
        # lifecycle entry. If reconcile later fires gone_on_exchange
        # for this oid, the trace will show whether we received any
        # cancel/fill events first (state-machine race) or none at all
        # (silent venue cancel / connectivity gap).
        if self._state is not None and order_id != 0:
            try:
                self._state.order_trace.record_ws_event(
                    order_id_exchange=order_id,
                    state=state,
                )
            except Exception:
                logger.exception("order_trace_ws_event_failed")
        try:
            uts = int(row.get("uTime") or 0)
        except (TypeError, ValueError):
            uts = 0

        timing = InboundPrivateTiming(
            ws_recv_mono=time.monotonic(),
            exchange_event_ms=uts,
        )

        sz_contracts = _coerce_float(row.get("sz"))
        acc_filled_contracts = _coerce_float(row.get("accFillSz"))
        sz_base = sz_contracts * self._contract_value
        acc_filled_base = acc_filled_contracts * self._contract_value
        remaining_base = max(0.0, sz_base - acc_filled_base)
        limit_px = _coerce_float(row.get("px"))

        # Map OKX state to the bot's expected uppercase status string.
        # The bot's parsers mostly consume "open"/"filled"/"canceled".
        state_upper_map = {
            "live": "NEW",
            "partially_filled": "PARTIALLY_FILLED",
            "filled": "FILLED",
            "canceled": "CANCELED",
            "mmp_canceled": "CANCELED",
        }
        order_status = state_upper_map.get(state, state.upper())

        order_update = PrivateOrderUpdateEvent(
            oid=order_id,
            coin=coin,
            status=order_status,
            status_timestamp_ms=uts,
            side=side_norm,
            limit_px=limit_px,
            remaining_sz=remaining_base,
            orig_sz=sz_base,
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

        # Fill event -- only when this row carries a fresh fill.
        # OKX repeats the row on every state change, so we gate on
        # ``fillSz > 0`` AND a fresh tradeId.
        last_qty_contracts = _coerce_float(row.get("fillSz"))
        if last_qty_contracts > 0:
            try:
                trade_id = int(row.get("tradeId") or 0)
            except (TypeError, ValueError):
                trade_id = 0
            last_px = _coerce_float(row.get("fillPx"))
            # OKX V5 ``fillFee`` on private WS orders: same convention as
            # /trade/fills REST -- POSITIVE = rebate received, NEGATIVE =
            # fee paid. Bot canonical convention is the opposite, so
            # negate. See HLFillRaw docstring for the full spec.
            # BUG-018 fix 2026-05-05.
            commission = -_coerce_float(row.get("fillFee"))
            realized = _coerce_float(row.get("fillPnl"))
            try:
                trade_time = int(row.get("fillTime") or row.get("uTime") or 0)
            except (TypeError, ValueError):
                trade_time = uts
            # OKX execType: "T"=taker, "M"=maker. crossed = took liquidity.
            exec_type = str(row.get("execType") or "").upper()
            crossed = exec_type == "T"
            last_qty_base = last_qty_contracts * self._contract_value
            if last_qty_base > 0 and last_px > 0 and trade_id > 0:
                fill = PrivateFillEvent(
                    fill_id=str(trade_id),
                    oid=order_id,
                    coin=coin,
                    px=last_px,
                    sz=last_qty_base,
                    side=side_norm,
                    time_ms=trade_time,
                    fee=commission,
                    closed_pnl=realized,
                    crossed=crossed,
                    is_snapshot=False,
                    raw=dict(row),
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
            logger.exception("okx_private_ws_wake_loop_failed")
