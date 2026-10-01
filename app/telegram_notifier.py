"""Outbound Telegram notifier (async fire-and-forget).

Two channels, configured via env:

* **Ops** (``TELEGRAM_OPS_CHAT_ID``) — bot lifecycle, recovery, drawdown,
  watchdog, WS reconnect, errors. Severity-prefixed.
* **Trades** (``TELEGRAM_TRADES_CHAT_ID``) — every fill (coalesced into
  short windows to cut noise) and notable trade events.

Disabled when ``TELEGRAM_BOT_TOKEN`` is empty: every public method becomes
a cheap no-op so local dev / unit tests never need real Telegram setup.

Architecture
------------
One daemon thread reads from a bounded queue; the public methods enqueue
and return immediately so callers in the trading hot path never block on
the network. HTTP via stdlib ``urllib`` (no extra dependency).

Coalescing for trades: fills within
``TELEGRAM_FILL_COALESCE_SECONDS`` are batched into one message. A token
bucket caps the trade-channel rate; drops over the cap accumulate into a
"+N more fills suppressed" suffix on the next emitted message so nothing
is silently lost.

The notifier is intentionally **best-effort**: any HTTP failure is logged
and dropped. The trading loop must not be coupled to Telegram availability.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# Telegram message length cap is 4096 chars. Leave headroom for our
# decorations (severity prefix, code blocks) and any auto-appended
# "+N more" suffix.
_MAX_MESSAGE_LEN = 3800

# How often the worker thread wakes when idle. Short enough to flush a
# coalesced batch promptly; long enough to be cheap.
_WORKER_TICK_SECONDS = 0.25

# HTTP request timeout for sending a Telegram message. Keep generous —
# Telegram occasionally takes 2-3 s to respond; bot is fire-and-forget
# so a slow send can't block trading.
_SEND_TIMEOUT_SECONDS = 10.0


@dataclass
class _QueueItem:
    channel: str  # "ops" or "trade"
    text: str
    parse_mode: Optional[str] = None  # "Markdown" / "HTML" / None
    enqueued_mono: float = field(default_factory=time.monotonic)


class TelegramNotifier:
    """Outbound notifier. Construct once, call ``start()``, then call
    ``notify_ops(...)`` / ``notify_trade_fill(...)`` from anywhere.

    Lifecycle:
        notifier = TelegramNotifier(settings)
        notifier.start()
        ... bot runs ...
        notifier.stop()  # flushes pending, closes worker

    Thread-safety: every public method is thread-safe.

    When disabled (no token / no chat configured), every method is a
    cheap no-op that doesn't even enqueue — there is no thread, no
    state mutation, no risk of stalling startup if Telegram is down.
    """

    def __init__(
        self,
        settings: Any,
        *,
        http_post: Optional[Callable[[str, dict], tuple[int, str]]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        token = (getattr(settings, "telegram_bot_token", "") or "").strip()
        ops_chat = (getattr(settings, "telegram_ops_chat_id", "") or "").strip()
        trades_chat = (
            getattr(settings, "telegram_trades_chat_id", "") or ""
        ).strip()

        self._token = token
        self._ops_chat = ops_chat
        self._trades_chat = trades_chat
        # Notifier is enabled only if we have a token AND at least one
        # destination. With token but no chats, there's nothing to do.
        self._enabled = bool(token) and (bool(ops_chat) or bool(trades_chat))

        self._coalesce_s = float(
            getattr(settings, "telegram_fill_coalesce_seconds", 5.0)
        )
        self._trade_rate = float(
            getattr(settings, "telegram_trade_rate_per_second", 0.5)
        )
        self._trade_burst = int(getattr(settings, "telegram_trade_burst", 3))

        # Injected for tests; production uses the real Telegram HTTP API.
        self._http_post = http_post or self._default_http_post
        self._clock = clock or time.monotonic

        # Bounded queue prevents pathological producer blowing up memory
        # if the notifier worker dies. Drops over the cap are logged.
        self._queue: queue.Queue[_QueueItem] = queue.Queue(maxsize=2000)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Token bucket state for the trades channel.
        self._trade_tokens: float = float(self._trade_burst)
        self._trade_tokens_last_refill_mono: float = self._clock()
        # Counter of trade messages dropped because the bucket was
        # empty; appended as a suffix on the next successful trade send.
        self._trade_suppressed_count: int = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # --- lifecycle ---

    def start(self) -> None:
        if not self._enabled:
            logger.info("telegram_notifier_disabled (no token or no chat configured)")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(
            target=self._run, name="telegram-notifier", daemon=True
        )
        self._thread = t
        t.start()
        logger.info(
            "telegram_notifier_started ops=%s trades=%s coalesce_s=%.1f rate=%.2f/s burst=%d",
            bool(self._ops_chat),
            bool(self._trades_chat),
            self._coalesce_s,
            self._trade_rate,
            self._trade_burst,
        )

    def stop(self, *, drain_timeout_s: float = 3.0) -> None:
        """Signal stop. The worker tries to flush any pending batch
        before exiting; capped at ``drain_timeout_s``."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=drain_timeout_s)

    # --- public emit API ---

    SEVERITY_PREFIX = {
        "INFO": "\u2705 INFO",
        "WARNING": "\u26a0\ufe0f WARN",
        "WARN": "\u26a0\ufe0f WARN",
        "ERROR": "\U0001f534 ERROR",
        "CRITICAL": "\U0001f534 CRITICAL",
    }

    def notify_ops(
        self,
        severity: str,
        event_type: str,
        message: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Enqueue an ops-channel message. Returns immediately.

        ``severity`` is one of INFO / WARNING / ERROR / CRITICAL. The
        prefix and Telegram formatting are added here so callers stay
        decoupled from Telegram details.
        """
        if not self._enabled or not self._ops_chat:
            return
        prefix = self.SEVERITY_PREFIX.get(
            (severity or "INFO").upper(), self.SEVERITY_PREFIX["INFO"]
        )
        body = self._format_ops_text(prefix, event_type, message, payload)
        self._enqueue(_QueueItem(channel="ops", text=body, parse_mode=None))

    def notify_trade_fill(self, fill: dict[str, Any]) -> None:
        """Enqueue a trade-channel fill. Coalesced with other fills
        inside ``TELEGRAM_FILL_COALESCE_SECONDS``."""
        if not self._enabled or not self._trades_chat:
            return
        text = self._format_fill_one(fill)
        self._enqueue(_QueueItem(channel="trade", text=text, parse_mode=None))

    def notify_trade_event(self, message: str) -> None:
        """Enqueue an arbitrary trades-channel message (e.g. position
        change, daily PnL summary). Same throttle as fills."""
        if not self._enabled or not self._trades_chat:
            return
        self._enqueue(_QueueItem(channel="trade", text=message, parse_mode=None))

    def notify_ops_blocking(
        self,
        severity: str,
        event_type: str,
        message: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Synchronous ops notification — bypasses the queue and worker.

        Use only for pre-exit hooks (watchdog, kill, ``/restart``) where
        the process is about to terminate and the worker thread may not
        get a chance to drain. HTTP timeout is short (default 2 s) so
        a Telegram outage doesn't block the shutdown path.
        """
        if not self._enabled or not self._ops_chat:
            return
        prefix = self.SEVERITY_PREFIX.get(
            (severity or "INFO").upper(), self.SEVERITY_PREFIX["INFO"]
        )
        body = self._format_ops_text(prefix, event_type, message, payload)
        try:
            self._send(self._ops_chat, body, None)
        except Exception:  # noqa: BLE001
            logger.exception("telegram_notify_ops_blocking_failed")

    # --- internal: formatting ---

    @staticmethod
    def _format_ops_text(
        prefix: str,
        event_type: str,
        message: str,
        payload: Optional[dict[str, Any]],
    ) -> str:
        head = f"{prefix} {event_type}"
        body = (message or "").strip()
        out = f"{head}\n{body}" if body else head
        if payload:
            try:
                payload_text = json.dumps(payload, default=str, sort_keys=True)
            except Exception:
                payload_text = str(payload)
            if len(payload_text) > 800:
                payload_text = payload_text[:800] + "..."
            out = f"{out}\n{payload_text}"
        if len(out) > _MAX_MESSAGE_LEN:
            out = out[: _MAX_MESSAGE_LEN - 20] + "...[truncated]"
        return out

    @staticmethod
    def _format_fill_one(fill: dict[str, Any]) -> str:
        side = fill.get("side", "?")
        sym = fill.get("symbol", "")
        price = fill.get("price")
        size = fill.get("size")
        notional = fill.get("notional")
        fee = fill.get("fee")
        closed_pnl = fill.get("closed_pnl")
        markout_5s = fill.get("markout_5s_bps")
        head = f"FILL {side} {sym}"
        parts = []
        if price is not None:
            parts.append(f"px={price}")
        if size is not None:
            parts.append(f"sz={size}")
        if notional is not None:
            try:
                parts.append(f"notional=${float(notional):.2f}")
            except (TypeError, ValueError):
                pass
        if fee is not None:
            try:
                parts.append(f"fee=${float(fee):.4f}")
            except (TypeError, ValueError):
                pass
        if closed_pnl is not None:
            # Shown unconditionally (including $0.00) so opening vs
            # closing fills are both legible at a glance: opening fills
            # have closed_pnl=0; reducing/closing fills carry the
            # exchange-reported realised PnL component for this trade.
            try:
                parts.append(f"closed_pnl=${float(closed_pnl):+.4f}")
            except (TypeError, ValueError):
                pass
        if markout_5s is not None:
            try:
                parts.append(f"mk5s={float(markout_5s):.2f}bps")
            except (TypeError, ValueError):
                pass
        return f"{head}  {' '.join(parts)}"

    # --- internal: enqueue ---

    def _enqueue(self, item: _QueueItem) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Silently dropping a notification is preferable to blocking
            # the trading hot path. Log so it's diagnosable.
            logger.warning(
                "telegram_notifier_queue_full dropped channel=%s", item.channel
            )

    # --- internal: worker loop ---

    def _run(self) -> None:
        coalesce_buf: list[_QueueItem] = []
        coalesce_oldest_mono: Optional[float] = None
        while not self._stop.is_set():
            timeout = _WORKER_TICK_SECONDS
            if coalesce_buf and coalesce_oldest_mono is not None:
                elapsed = self._clock() - coalesce_oldest_mono
                wait = max(0.05, self._coalesce_s - elapsed)
                timeout = min(timeout, wait)
            try:
                item: Optional[_QueueItem] = self._queue.get(timeout=timeout)
            except queue.Empty:
                item = None

            if item is not None:
                try:
                    if item.channel == "ops":
                        self._send_ops(item)
                    elif item.channel == "trade":
                        if self._coalesce_s <= 0:
                            self._send_trade(item.text)
                        else:
                            if not coalesce_buf:
                                coalesce_oldest_mono = self._clock()
                            coalesce_buf.append(item)
                except Exception:
                    logger.exception("telegram_notifier_send_failed")

            # Flush coalesce buffer if window elapsed.
            if (
                coalesce_buf
                and coalesce_oldest_mono is not None
                and (self._clock() - coalesce_oldest_mono) >= self._coalesce_s
            ):
                try:
                    self._flush_trade_batch(coalesce_buf)
                except Exception:
                    logger.exception("telegram_notifier_flush_failed")
                coalesce_buf = []
                coalesce_oldest_mono = None

        # Drain on stop: send any pending batch.
        if coalesce_buf:
            try:
                self._flush_trade_batch(coalesce_buf)
            except Exception:
                logger.exception("telegram_notifier_drain_failed")

    # --- internal: send ---

    def _send_ops(self, item: _QueueItem) -> None:
        self._send(self._ops_chat, item.text, item.parse_mode)

    def _send_trade(self, text: str) -> None:
        if not self._try_acquire_trade_token():
            with self._lock:
                self._trade_suppressed_count += 1
            return
        # Append the suppressed-count suffix from prior drops, if any.
        with self._lock:
            suppressed = self._trade_suppressed_count
            self._trade_suppressed_count = 0
        if suppressed > 0:
            text = f"{text}\n(+{suppressed} prior fills suppressed)"
        self._send(self._trades_chat, text, None)

    def _flush_trade_batch(self, items: list[_QueueItem]) -> None:
        if not items:
            return
        if len(items) == 1:
            self._send_trade(items[0].text)
            return
        # Multi-fill batch: one message with each fill on its own line.
        head = f"FILLS x{len(items)}"
        body = "\n".join(item.text for item in items)
        text = f"{head}\n{body}"
        if len(text) > _MAX_MESSAGE_LEN:
            text = text[: _MAX_MESSAGE_LEN - 20] + "...[truncated]"
        self._send_trade(text)

    def _try_acquire_trade_token(self) -> bool:
        now = self._clock()
        with self._lock:
            elapsed = max(0.0, now - self._trade_tokens_last_refill_mono)
            self._trade_tokens = min(
                float(self._trade_burst),
                self._trade_tokens + elapsed * self._trade_rate,
            )
            self._trade_tokens_last_refill_mono = now
            if self._trade_tokens >= 1.0:
                self._trade_tokens -= 1.0
                return True
            return False

    def _send(self, chat_id: str, text: str, parse_mode: Optional[str]) -> None:
        if not self._token or not chat_id:
            return
        body = {"chat_id": chat_id, "text": text}
        if parse_mode:
            body["parse_mode"] = parse_mode
        try:
            status, resp = self._http_post(
                f"https://api.telegram.org/bot{self._token}/sendMessage", body
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("telegram_send_exception channel_chat=%s err=%s", chat_id, exc)
            return
        if status != 200:
            # Telegram returns JSON with description. Truncate noisy bodies.
            r = (resp or "")[:300]
            logger.warning(
                "telegram_send_failed status=%d chat=%s body=%s", status, chat_id, r
            )

    # --- default HTTP impl (stdlib) ---

    @staticmethod
    def _default_http_post(url: str, body: dict[str, Any]) -> tuple[int, str]:
        data = urllib.parse.urlencode(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                req, timeout=_SEND_TIMEOUT_SECONDS
            ) as resp:
                status = int(resp.status)
                payload = resp.read().decode("utf-8", errors="replace")
                return status, payload
        except urllib.error.HTTPError as e:
            try:
                payload = e.read().decode("utf-8", errors="replace")
            except Exception:
                payload = ""
            return int(e.code), payload
