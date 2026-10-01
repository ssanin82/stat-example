"""Bluefin Pro private WebSocket stream (fills + order updates).

Migrated to pro-sdk layout:

  * URL: ``wss://stream.api.{env}.bluefin.io/ws/account``
  * Upgrade handshake carries ``Authorization: Bearer <jwt>``.
  * Subscribe message shape::

        {
          "authToken": "<jwt>",
          "method": "Subscribe",
          "dataStreams": ["AccountTradeUpdate", "AccountOrderUpdate"]
        }

  * Message envelope::

        {
          "event": "AccountTradeUpdate" | "AccountOrderUpdate" | ...,
          "reason": "OrderMatched" | "OrderCancelled" | ...,
          "payload": { ...event-specific... }
        }

All numeric fields arrive in 1e9 base (``*E9`` string decimals). The fill
payload's ``tradingFeeE9`` goes into :attr:`PrivateFillEvent.fee` verbatim
(after abs + e9 descale) so post-session telemetry can confirm whether
the SUI-PERP zero-fee promo is still active.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Optional

import httpx

# Minimum seconds between consecutive "idle warn" log lines, to keep
# logs readable when the connection stays idle for long stretches.
# Kept at 60 s to match the GRVT analogue.
_BLUEFIN_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S = 60.0

from app.config import Settings
from app.exchange.bluefin_auth import (
    load_or_init_session,
    sign_login_request,
    signable_login,
)
from app.exchange.bluefin_responses import hash_to_oid
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
AfterPutCallback = Optional[Callable[[str], None]]
# Invoked for every AccountOrderUpdate whose payload carries a
# ``cancellationReason`` (pro-sdk OrderCancellationUpdate). The WS layer
# forwards the lowercase 0x-prefixed order hash; the BluefinClient adapter
# uses it to clear its ``_pending_cancels`` state and release the
# replacement-placement gate. See app/exchange/base.py docstring on
# ``has_pending_cancel``.
CancelConfirmCallback = Optional[Callable[[str], None]]

try:
    import websocket  # type: ignore[import-untyped]
except ImportError as e:  # pragma: no cover
    websocket = None  # type: ignore[assignment]
    _websocket_import_error: Optional[Exception] = e
else:
    _websocket_import_error = None


_DEFAULT_HOST = "wss://stream.api.sui-prod.bluefin.io"
_ACCOUNT_PATH = "/ws/account"


def _from_e9(raw: Any) -> float:
    if raw is None:
        return 0.0
    try:
        return float(Decimal(str(raw)) / (Decimal(10) ** 9))
    except Exception:
        return 0.0


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
        logger.error("bluefin_private_ws_queue_full dropped=%s", drop_label)
        if on_queue_drop is not None:
            try:
                on_queue_drop(drop_label)
            except Exception:
                logger.exception("bluefin_private_ws_queue_drop_callback_failed")
        return
    if on_after_put is not None:
        try:
            on_after_put(drop_label)
        except Exception:
            logger.exception("bluefin_private_ws_on_after_put_failed label=%s", drop_label)


def _resolve_account_ws_url(settings: Settings) -> str:
    raw = (settings.bluefin_private_ws_url or _DEFAULT_HOST).rstrip("/")
    # Tolerate env profiles that specify either the host-only form
    # (``wss://stream.api.../``) or the fully-qualified path form.
    if raw.endswith(_ACCOUNT_PATH):
        return raw
    return raw + _ACCOUNT_PATH


def _audience_for_network(network: str) -> str:
    _ = network
    return "api"


def _resolve_auth_url(settings: Settings) -> str:
    if settings.bluefin_auth_url:
        return settings.bluefin_auth_url.rstrip("/")
    network = (settings.bluefin_network or "SUI_PROD").strip().lower()
    if network in ("sui_prod", "production", "prod", "mainnet"):
        env = "sui-prod"
    elif network in ("sui_staging", "staging", "testnet"):
        env = "sui-staging"
    else:
        env = "sui-dev"
    return f"https://auth.api.{env}.bluefin.io"


class BluefinPrivateStream:
    """Dedicated thread; subscribes to AccountTradeUpdate + AccountOrderUpdate."""

    def __init__(
        self,
        settings: Settings,
        user_address: str,
        out_queue: queue.Queue,
        on_queue_drop: DropCallback = None,
        state: Optional[BotState] = None,
        on_cancel_confirmed: CancelConfirmCallback = None,
    ) -> None:
        self._settings = settings
        self._addr = (user_address or "").strip()
        self._q = out_queue
        self._on_queue_drop = on_queue_drop
        self._state = state
        self._on_cancel_confirmed = on_cancel_confirmed
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws_lock = threading.Lock()
        self._ws_app: Any = None
        self._auth_token: Optional[str] = None
        self._auth_token_expiry_epoch: float = 0.0
        self._http = httpx.Client(timeout=8.0)
        # Keepalive / idle-watch worker — ports the two-tier pattern from
        # app/exchange/grvt_ws.py (which GRVT needed for the same reason:
        # the server does NOT send heartbeat frames on idle, so the
        # websocket-client's transport-level ping_interval alone leaves
        # the bot susceptible to zombie connections. Observed 2026-04-24
        # on Bluefin: private WS silent for 62 minutes after the last
        # fill, cancel-confirmation gate latched, quote loop stopped
        # placing. The worker logs at idle_warn and force-closes (triggers
        # reconnect) at idle_reconnect.
        self._keepalive_stop = threading.Event()
        self._keepalive_thread: Optional[threading.Thread] = None
        self._last_idle_log_mono: float = 0.0

    def set_on_cancel_confirmed(self, cb: CancelConfirmCallback) -> None:
        """Register (or replace) the cancel-confirmation callback after construction.

        Used by ``app/main.py`` when the private stream is constructed
        before the adapter's ``on_cancel_confirmed`` method is known to
        the composition root, and by tests that want to observe cancel
        confirmations without going through the adapter.
        """
        self._on_cancel_confirmed = cb

    def _after_queue_put(self, label: str) -> None:
        if self._state is not None and label in ("fill", "order_update"):
            try:
                self._state.wake_quote_loop()
            except Exception:
                logger.exception("bluefin_private_ws_wake_failed label=%s", label)

    def _emit_conn(
        self,
        kind: PrivateWsConnectionKind,
        detail: str = "",
        backoff_seconds: float = 0.0,
    ) -> None:
        _safe_put(
            self._q,
            PrivateWsConnectionEvent(kind=kind, detail=detail, backoff_seconds=backoff_seconds),
            "connection",
            self._on_queue_drop,
            self._after_queue_put,
        )

    # ------------------------------------------------------------------
    # Auth token minting (/auth/v2/token)
    # ------------------------------------------------------------------

    def _ensure_auth_token(self) -> Optional[str]:
        now = time.time()
        if self._auth_token and self._auth_token_expiry_epoch - now > 30.0:
            return self._auth_token
        pk = (self._settings.bluefin_private_key or "").strip()
        if not pk:
            return None
        try:
            session = load_or_init_session(
                private_key_raw=pk,
                account_address=self._settings.bluefin_account_address,
                one_ct_enabled=self._settings.bluefin_one_ct_enabled,
                one_ct_duration_hours=self._settings.bluefin_one_ct_duration_hours,
                symbol=self._settings.symbol,
                network=self._settings.bluefin_network,
            )
        except Exception:
            logger.exception("bluefin_private_ws_session_load_failed")
            return None
        payload = signable_login(
            account_address=session.parent_address,
            audience=_audience_for_network(self._settings.bluefin_network),
            signed_at_millis=int(now * 1000),
        )
        try:
            signature = sign_login_request(payload, signing_key=session.signing_key)
        except Exception:
            logger.exception("bluefin_private_ws_sign_failed")
            return None
        auth_url = _resolve_auth_url(self._settings)
        try:
            resp = self._http.post(
                f"{auth_url}/auth/v2/token",
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "payloadSignature": signature,
                },
            )
            resp.raise_for_status()
            body = resp.json() if resp.content else {}
        except Exception as e:
            logger.warning("bluefin_private_ws_auth_token_failed err=%s", e)
            return None
        if not isinstance(body, dict):
            return None
        token = str(body.get("accessToken") or "").strip()
        if not token:
            logger.warning("bluefin_private_ws_no_auth_token")
            return None
        try:
            valid_for = int(body.get("accessTokenValidForSeconds") or 0)
        except (TypeError, ValueError):
            valid_for = 0
        if valid_for <= 0:
            valid_for = 12 * 3600
        self._auth_token = token
        self._auth_token_expiry_epoch = now + float(valid_for)
        return token

    # ------------------------------------------------------------------
    # Thread lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if websocket is None:
            logger.error(
                "bluefin_private_ws_disabled missing_dependency err=%s",
                _websocket_import_error,
            )
            return
        if not self._settings.private_ws_enabled:
            logger.info("bluefin_private_ws_start_skipped PRIVATE_WS_ENABLED=false")
            return
        if not self._addr:
            logger.info("bluefin_private_ws_start_skipped empty_address")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_forever,
            name="bluefin-private-ws",
            daemon=True,
        )
        self._thread.start()
        logger.info("bluefin_private_ws_thread_started address=%s", self._addr)

    def stop(self) -> None:
        self._stop.set()
        with self._ws_lock:
            ws = self._ws_app
            self._ws_app = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception("bluefin_private_ws_close_error")
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None
        logger.info("bluefin_private_ws_stopped")

    def _run_forever(self) -> None:
        initial = float(self._settings.private_ws_reconnect_initial_seconds)
        max_b = float(self._settings.private_ws_reconnect_max_seconds)
        delay_s = 0.0
        first = True
        attempt = 0
        while not self._stop.is_set():
            if not first and delay_s > 0:
                self._emit_conn(
                    PrivateWsConnectionKind.RECONNECT_SCHEDULED,
                    f"backoff_s={delay_s:.3f}",
                    delay_s,
                )
                self._stop.wait(timeout=delay_s)
            first = False
            if self._stop.is_set():
                break
            attempt += 1
            if attempt > 1 and self._state is not None:
                with self._state._lock:
                    self._state.private_ws_reconnect_count += 1
            try:
                self._connect_once()
            except Exception:
                if self._stop.is_set():
                    break
                logger.exception("bluefin_private_ws_session_failed")
                self._emit_conn(PrivateWsConnectionKind.ERROR, "session_exception", 0.0)
            delay_s = initial if delay_s <= 0 else min(max_b, max(delay_s * 2.0, initial))

    def _connect_once(self) -> None:
        assert websocket is not None
        token = self._ensure_auth_token() or ""
        url = _resolve_account_ws_url(self._settings)

        def on_open(ws: Any) -> None:
            subscribe = {
                "authToken": token,
                "method": "Subscribe",
                "dataStreams": [
                    "AccountTradeUpdate",
                    "AccountOrderUpdate",
                ],
            }
            try:
                ws.send(json.dumps(subscribe))
            except Exception:
                logger.exception("bluefin_private_ws_subscribe_failed")
                return
            self._emit_conn(PrivateWsConnectionKind.CONNECTED, "open_and_subscribe_sent")
            logger.info("bluefin_private_ws_connected url=%s", url)
            if self._state is not None:
                with self._state._lock:
                    self._state.private_ws_last_connect_wall_ts = datetime.now(timezone.utc)

        def on_message(_ws: Any, message: str) -> None:
            recv_mono = time.perf_counter()
            recv_wall = datetime.now(timezone.utc)
            if self._state is not None:
                with self._state._lock:
                    self._state.private_ws_last_message_wall_ts = recv_wall
            self._handle_raw_message(message, recv_mono=recv_mono, recv_wall=recv_wall)

        def on_error(_ws: Any, error: Any) -> None:
            logger.warning("bluefin_private_ws_on_error %s", error)

        def on_close(_ws: Any, close_status_code: Any, close_msg: Any) -> None:
            detail = f"code={close_status_code} msg={str(close_msg or '')[:300]}"
            self._emit_conn(PrivateWsConnectionKind.DISCONNECTED, detail)

        # Authorization header during upgrade — pro-sdk docs imply either
        # header auth OR in-message authToken works; we send both for
        # belt-and-braces, since a stray proxy might strip one.
        headers = []
        if token:
            headers.append(f"Authorization: Bearer {token}")

        with self._ws_lock:
            self._ws_app = websocket.WebSocketApp(
                url,
                header=headers or None,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            app = self._ws_app
        # Start the keepalive/idle-watch worker. It runs in parallel with
        # run_forever and is torn down in the ``finally`` below so a
        # reconnect cycle doesn't leak threads.
        self._start_keepalive()
        try:
            # ping_interval/timeout are transport-level pings which
            # Bluefin's server does NOT treat as activity (observed: no
            # private-WS messages for 62 minutes despite ping_interval=280
            # keeping the TCP/WS connection alive). The idle-watch worker
            # above is the actual recovery mechanism; these transport
            # pings only catch full TCP dropouts. Reduced interval from
            # 280 s to 60 s so a dropped TCP connection is detected
            # faster, and the idle-watch worker provides defense in depth
            # for the application-level silence case.
            app.run_forever(ping_interval=60, ping_timeout=20)
        finally:
            self._stop_keepalive_worker()

    def _start_keepalive(self) -> None:
        """Spawn the idle-watch worker thread for this connection."""
        self._stop_keepalive_worker()
        self._keepalive_stop.clear()
        t = threading.Thread(
            target=self._keepalive_worker,
            name="bluefin-private-ws-keepalive",
            daemon=True,
        )
        self._keepalive_thread = t
        t.start()

    def _stop_keepalive_worker(self) -> None:
        self._keepalive_stop.set()
        t = self._keepalive_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.5)
        self._keepalive_thread = None

    def _keepalive_worker(self) -> None:
        """Two-tier idle detection for the private WS.

        Runs every ``PRIVATE_WS_APP_KEEPALIVE_SECONDS`` (default 22 s) and
        measures time since ``state.private_ws_last_message_wall_ts``:

        * If idle ≥ ``PRIVATE_WS_IDLE_WARN_SECONDS`` (default 120 s): log
          a throttled warning. Observability only, no state change.
        * If idle ≥ ``PRIVATE_WS_IDLE_RECONNECT_SECONDS`` (default 300 s):
          force-close the WS. ``run_forever`` exits, the outer reconnect
          loop (``_run_reconnect_loop``-style caller) re-mints auth and
          reconnects. This is the actual recovery step.

        Either threshold set to ``0`` disables that tier.
        ``PRIVATE_WS_APP_KEEPALIVE_SECONDS = 0`` disables the worker
        entirely.

        This worker is the Bluefin equivalent of the GRVT worker at
        ``grvt_ws.py:_keepalive_worker``; Bluefin had none prior to
        2026-04-24, which caused the bot to silently stop quoting ~1 h
        after the last fill when the exchange stopped sending frames.
        """
        interval = float(self._settings.private_ws_app_keepalive_seconds)
        idle_warn = float(self._settings.private_ws_idle_warn_seconds)
        idle_reconnect = float(self._settings.private_ws_idle_reconnect_seconds)
        if interval <= 0:
            return
        if idle_warn <= 0 and idle_reconnect <= 0:
            return
        while not self._keepalive_stop.wait(timeout=interval):
            if self._stop.is_set():
                break
            with self._ws_lock:
                ws = self._ws_app
            if ws is None:
                continue
            if self._state is None:
                continue
            with self._state._lock:
                last_msg = self._state.private_ws_last_message_wall_ts
            if last_msg is None:
                continue
            idle_s = max(0.0, (datetime.now(timezone.utc) - last_msg).total_seconds())

            # Warn tier (observability only).
            if idle_warn > 0 and idle_s >= idle_warn:
                now_mono = time.monotonic()
                if now_mono - self._last_idle_log_mono >= _BLUEFIN_PRIVATE_WS_IDLE_LOG_MIN_INTERVAL_S:
                    self._last_idle_log_mono = now_mono
                    logger.warning(
                        "bluefin_private_ws_idle_warn idle_s=%.1f "
                        "warn_threshold_s=%.1f reconnect_threshold_s=%.1f",
                        idle_s,
                        idle_warn,
                        idle_reconnect,
                    )

            # Reconnect tier (stuck-subscription / zombie-connection recovery).
            if idle_reconnect > 0 and idle_s >= idle_reconnect:
                logger.warning(
                    "bluefin_private_ws_idle_too_long idle_s=%.1f threshold_s=%.1f "
                    "forcing_reconnect",
                    idle_s,
                    idle_reconnect,
                )
                try:
                    ws.close()
                except Exception:
                    logger.debug("bluefin_private_ws_idle_close_failed", exc_info=True)
                # Exit: run_forever will unwind, the caller's reconnect
                # loop spawns a fresh worker for the new connection.
                break

    def _handle_raw_message(
        self, message: str, *, recv_mono: float, recv_wall: datetime
    ) -> None:
        parse_start = time.perf_counter()
        try:
            obj = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(
                "bluefin_private_ws_malformed_json len=%s snippet=%s",
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
                    "bluefin_private_ws_subscription_failed message=%s",
                    str(obj.get("message") or "")[:500],
                )
            return

        event_name = str(obj.get("event") or obj.get("eventName") or "")
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return

        if event_name in ("AccountTradeUpdate", "AccountAggregatedTradeUpdate"):
            trade = payload.get("trade")
            if isinstance(trade, dict):
                self._emit_fill(trade, recv_mono, recv_wall, parse_start, parse_end)
            return
        if event_name == "AccountOrderUpdate":
            # payload is either an ActiveOrderUpdate or an OrderCancellationUpdate;
            # the two are distinguished by which fields are present.
            self._emit_order_update(payload, recv_mono, recv_wall, parse_start, parse_end)
            return

    def _emit_fill(
        self,
        data: dict[str, Any],
        recv_mono: float,
        recv_wall: datetime,
        parse_start: float,
        parse_end: float,
    ) -> None:
        ts_raw = data.get("executedAtMillis") or 0
        try:
            event_ms = int(ts_raw)
        except (TypeError, ValueError):
            event_ms = int(time.time() * 1000.0)
        timing = InboundPrivateTiming(
            ws_recv_mono=recv_mono,
            ws_recv_wall=recv_wall,
            parse_start_mono=parse_start,
            parse_end_mono=parse_end,
            enqueue_mono=time.perf_counter(),
            exchange_event_ms=event_ms,
        )
        order_hash = str(data.get("orderHash") or "")
        trade_id = str(data.get("id") or "")
        fill_id = f"{trade_id}_{event_ms}"
        side_str = str(data.get("side") or "").upper()
        # tradingFeeE9 is negative when the account paid a fee. Fold to
        # abs() — downstream telemetry treats ``fee`` as unsigned USDC.
        fee_usdc = abs(_from_e9(data.get("tradingFeeE9") or "0"))
        ev = PrivateFillEvent(
            fill_id=fill_id,
            oid=hash_to_oid(order_hash) if order_hash else None,
            coin=str(data.get("symbol") or "").upper(),
            px=_from_e9(data.get("priceE9")),
            sz=_from_e9(data.get("quantityE9") or "0"),
            side="B" if side_str == "LONG" else "A",
            time_ms=event_ms,
            fee=fee_usdc,
            closed_pnl=_from_e9(data.get("realizedPnlE9") or "0"),
            crossed=not bool(data.get("isMaker", False)),
            is_snapshot=False,
            raw=dict(data),
            inbound_timing=timing,
        )
        _safe_put(self._q, ev, "fill", self._on_queue_drop, self._after_queue_put)

    def _emit_order_update(
        self,
        data: dict[str, Any],
        recv_mono: float,
        recv_wall: datetime,
        parse_start: float,
        parse_end: float,
    ) -> None:
        order_hash = str(data.get("orderHash") or "")
        if not order_hash:
            return
        oid = hash_to_oid(order_hash)
        ts_raw = (
            data.get("updatedAtMillis")
            or data.get("createdAtMillis")
            or 0
        )
        try:
            event_ms = int(ts_raw)
        except (TypeError, ValueError):
            event_ms = int(time.time() * 1000.0)
        timing = InboundPrivateTiming(
            ws_recv_mono=recv_mono,
            ws_recv_wall=recv_wall,
            parse_start_mono=parse_start,
            parse_end_mono=parse_end,
            enqueue_mono=time.perf_counter(),
            exchange_event_ms=event_ms,
        )
        # OrderCancellationUpdate carries ``cancellationReason``/``remainingQuantityE9``
        # but no ``status``; synthesise a "CANCELLED" status for it.
        is_cancellation_update = "cancellationReason" in data and "status" not in data
        if is_cancellation_update:
            status = "CANCELLED"
            orig_sz = 0.0
            remaining = _from_e9(data.get("remainingQuantityE9") or "0")
            reason = str(data.get("cancellationReason") or "")
            # Fire the cancel-confirmation callback before enqueuing the
            # event so the adapter's pending-cancel state clears as soon
            # as possible. Callback failures must not drop the event —
            # log and keep going. Hash is normalised to the
            # lowercase-with-``0x`` form the adapter keys its state on.
            cb = self._on_cancel_confirmed
            if cb is not None and order_hash:
                normalised = order_hash.lower().strip()
                if not normalised.startswith("0x"):
                    normalised = "0x" + normalised
                try:
                    cb(normalised)
                except Exception:
                    logger.exception(
                        "bluefin_cancel_confirm_callback_failed hash=%s", normalised
                    )
        else:
            status = str(data.get("status") or "").upper()
            orig_sz = _from_e9(data.get("quantityE9") or "0")
            filled_sz = _from_e9(data.get("filledQuantityE9") or "0")
            remaining = max(0.0, orig_sz - filled_sz) if orig_sz > 0 else 0.0
            reason = ""
        side_str = str(data.get("side") or "").upper()
        cloid_raw = data.get("clientOrderId")
        cloid = str(cloid_raw) if cloid_raw not in (None, "") else None
        ev = PrivateOrderUpdateEvent(
            oid=oid,
            coin=str(data.get("symbol") or "").upper(),
            status=status,
            status_timestamp_ms=event_ms,
            side="B" if side_str == "LONG" else "A",
            limit_px=_from_e9(data.get("priceE9")),
            remaining_sz=remaining,
            orig_sz=orig_sz,
            raw_status=reason,
            inbound_timing=timing,
            cloid=cloid,
        )
        _safe_put(self._q, ev, "order_update", self._on_queue_drop, self._after_queue_put)

    def feed_message_for_tests(self, message: str) -> None:
        self._handle_raw_message(
            message,
            recv_mono=time.perf_counter(),
            recv_wall=datetime.now(timezone.utc),
        )


__all__ = ["BluefinPrivateStream"]
