"""Inbound Telegram command poller.

Long-poll-based receiver for operator commands. Designed to be portable
across any deploy host (AWS EC2 default release target, the
OKX-adjacent Alibaba HK colo exception, any PaaS container) without a
webhook-capable public ingress: the bot calls Telegram's ``getUpdates``
with a long timeout and acts on the results, so the only network
requirement is *outbound* HTTPS, which is already present.

Authentication: two-layer whitelist.

* ``TELEGRAM_ALLOWED_USER_IDS`` — comma-separated Telegram user_ids
  permitted to issue commands. Without this, the poller refuses to
  start (no auth → no commands).
* ``TELEGRAM_ALLOWED_CHAT_IDS`` — optional chat_id allowlist. Empty
  means "any chat where an allowed user writes is fine"; for DM-only
  setups this is the natural default because chat_id == user_id in
  private chats.

Rate limit: ``TELEGRAM_COMMAND_MIN_INTERVAL_SECONDS`` enforced per user.

Two-step confirmation: destructive commands (``/kill``, ``/flatten``,
``/restart``) require a follow-up ``/<cmd> confirm`` within
``TELEGRAM_CONFIRM_TIMEOUT_SECONDS`` or the request is dropped. Read-only
and reversible commands (``/status``, ``/pause``, etc.) execute immediately.

The poller is intentionally **best-effort**: any HTTP failure is logged
and dropped. Trading correctness must not depend on Telegram availability.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from app import __version__

from app import clock as _clock

logger = logging.getLogger(__name__)


# Code passed to ``os._exit`` on ``/restart``. Distinct from the watchdog
# (42) so operators grepping container logs can tell user-triggered
# restarts from deadlock-triggered ones.
RESTART_EXIT_CODE = 43


# Long-poll request timeout. Telegram's API caps at 50; the local timeout
# must be slightly larger to cover the round trip.
_HTTP_TIMEOUT_HEADROOM_SECONDS = 5.0


def _fmt_vol_with_threshold(
    vol_bps: Optional[float], *, threshold_bps: float
) -> str:
    """Format the bot's short-window vol estimate alongside the reprice
    threshold so the operator can read at a glance whether the market is
    hot or cool relative to the bot's reprice gate.

    Returns ``"?"`` when vol isn't yet computable (warmup), otherwise
    ``"<vol>.XX bps (vs <threshold>.XX reprice)"`` with a regime hint.
    """
    if vol_bps is None:
        return "?"
    v = float(vol_bps)
    if threshold_bps <= 0:
        return f"{v:.2f} bps"
    ratio = v / threshold_bps
    if ratio < 0.5:
        regime = "quiet"
    elif ratio < 1.5:
        regime = "normal"
    elif ratio < 3.0:
        regime = "active"
    else:
        regime = "volatile"
    return (
        f"{v:.2f} bps (vs {threshold_bps:.1f} reprice → {regime})"
    )


@dataclass
class _PendingConfirm:
    command: str
    user_id: int
    chat_id: int
    deadline_mono: float


class TelegramCommandPoller:
    """Background thread that polls Telegram ``getUpdates`` and dispatches
    commands.

    Construct once at startup, call ``start()``; ``stop()`` for orderly
    teardown. Disabled when token is empty OR allowed_user_ids is empty.
    """

    def __init__(
        self,
        settings: Any,
        *,
        bot: Any,
        state: Any,
        storage: Any,
        notifier: Optional[Any] = None,
        client: Optional[Any] = None,
        http_get: Optional[Callable[[str, dict, float], tuple[int, str]]] = None,
        http_post: Optional[Callable[[str, dict], tuple[int, str]]] = None,
        clock: Optional[Callable[[], float]] = None,
        exit_fn: Optional[Callable[[int], None]] = None,
    ) -> None:
        token = (getattr(settings, "telegram_bot_token", "") or "").strip()
        user_ids_raw = (
            getattr(settings, "telegram_allowed_user_ids", "") or ""
        ).strip()
        chat_ids_raw = (
            getattr(settings, "telegram_allowed_chat_ids", "") or ""
        ).strip()

        self._settings = settings
        self._bot = bot
        self._state = state
        self._storage = storage
        self._notifier = notifier
        self._client = client

        # Account volume cache for ``/status``. Refreshed by a background
        # daemon thread (started in ``start()``) every
        # ``TELEGRAM_VOLUME_REFRESH_SECONDS`` (default 1800 = 30 min). Doing
        # this in the background keeps ``/status`` snappy — the REST
        # round-trip for 30 days of fills is 1-3 seconds and we don't want
        # the operator command path blocked on it.
        self._volume_lock = threading.Lock()
        self._volume_7d_usd: Optional[float] = None
        self._volume_30d_usd: Optional[float] = None
        self._volume_refreshed_at_mono: Optional[float] = None
        self._volume_last_error: Optional[str] = None
        self._volume_thread: Optional[threading.Thread] = None
        self._volume_stop = threading.Event()

        self._token = token
        self._allowed_user_ids = self._parse_int_csv(user_ids_raw)
        self._allowed_chat_ids = self._parse_int_csv(chat_ids_raw)
        self._enabled = bool(token) and bool(self._allowed_user_ids)

        self._poll_timeout_s = int(
            getattr(settings, "telegram_long_poll_timeout_seconds", 30)
        )
        self._min_interval_s = float(
            getattr(settings, "telegram_command_min_interval_seconds", 2.0)
        )
        self._confirm_timeout_s = float(
            getattr(settings, "telegram_confirm_timeout_seconds", 30.0)
        )

        self._http_get = http_get or self._default_http_get
        self._http_post = http_post or self._default_http_post
        self._clock = clock or time.monotonic
        self._exit_fn = exit_fn or (lambda code: os._exit(code))

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._offset: int = 0  # Telegram update_id cursor

        # Per-user rate limiter: user_id -> last-command monotonic ts.
        self._last_cmd_mono: dict[int, float] = {}

        # Pending two-step-confirm slots: keyed by (user_id, command).
        self._pending_confirms: dict[tuple[int, str], _PendingConfirm] = {}

        # Lock for shared dicts + cursor.
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self._enabled

    # --- lifecycle ---

    def start(self) -> None:
        if not self._enabled:
            logger.info(
                "telegram_commands_disabled (no token or no allowed_user_ids)"
            )
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(
            target=self._run, name="telegram-commands", daemon=True
        )
        self._thread = t
        t.start()
        # Spin up the volume cache refresher iff a Bluefin client is
        # available AND the refresh interval is positive. Failures inside
        # the refresher are caught and logged; they never break the
        # command-poll thread.
        self._maybe_start_volume_refresher()
        logger.info(
            "telegram_commands_started users=%d chats=%d poll_timeout=%ds",
            len(self._allowed_user_ids),
            len(self._allowed_chat_ids),
            self._poll_timeout_s,
        )

    def stop(self) -> None:
        self._stop.set()
        self._volume_stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            # Long-poll request may take up to poll_timeout_s + headroom
            # to return; bound the join so shutdown isn't held hostage.
            t.join(timeout=self._poll_timeout_s + 5.0)
        vt = self._volume_thread
        if vt is not None and vt.is_alive() and vt is not threading.current_thread():
            vt.join(timeout=5.0)

    # --- volume cache refresher ---

    def _maybe_start_volume_refresher(self) -> None:
        """Launch the background thread that refreshes 7d/30d account volume.

        No-op if (a) no client is wired in (e.g. dev / unit tests),
        (b) the client doesn't expose ``fetch_account_volume_usd`` (e.g.
        non-Bluefin venues), or (c) the refresh interval is 0 (disabled).
        """
        if self._client is None:
            return
        interval = float(
            getattr(self._settings, "telegram_volume_refresh_seconds", 0) or 0
        )
        if interval <= 0:
            return
        if not hasattr(self._client, "fetch_account_volume_usd"):
            return
        if self._volume_thread is not None and self._volume_thread.is_alive():
            return
        self._volume_stop.clear()
        vt = threading.Thread(
            target=self._volume_refresh_loop,
            name="telegram-volume-refresh",
            daemon=True,
        )
        self._volume_thread = vt
        vt.start()
        logger.info(
            "telegram_volume_refresher_started interval=%.0fs", interval
        )

    def _volume_refresh_loop(self) -> None:
        """Background loop. First iteration runs immediately so the cache
        is populated by the time the operator sends ``/status``; subsequent
        iterations sleep ``telegram_volume_refresh_seconds`` between runs.
        """
        interval = float(
            getattr(self._settings, "telegram_volume_refresh_seconds", 1800.0)
            or 1800.0
        )
        # First refresh: immediate, with a tiny delay so we don't fight
        # the bot's own startup REST surge.
        if self._volume_stop.wait(timeout=2.0):
            return
        while not self._volume_stop.is_set():
            try:
                self._refresh_volume_cache_once()
            except Exception:  # noqa: BLE001
                # Don't let any failure kill the daemon thread — the cache
                # just stays at its last good value, and the operator sees
                # a stale-marker on /status.
                logger.exception("telegram_volume_refresh_failed")
            if self._volume_stop.wait(timeout=interval):
                return

    def _refresh_volume_cache_once(self) -> None:
        """Fetch 7d and 30d volume from Bluefin and update the cache.

        Both windows hit the same paginated endpoint with different start
        timestamps. Failures on either window are caught individually so
        a transient REST hiccup on one doesn't blank out the other.
        """
        symbol = getattr(self._settings, "symbol", None)
        # Compute window bounds in milliseconds (Bluefin's units).
        now = _clock.now_utc()
        end_ms = int(now.timestamp() * 1000)
        windows = {
            "7d": int((now - timedelta(days=7)).timestamp() * 1000),
            "30d": int((now - timedelta(days=30)).timestamp() * 1000),
        }
        results: dict[str, Optional[float]] = {"7d": None, "30d": None}
        last_err: Optional[str] = None
        for label, since_ms in windows.items():
            try:
                out = self._client.fetch_account_volume_usd(
                    symbol, since_ms, end_ms
                )
                results[label] = float(out.get("volume_usd") or 0.0)
            except Exception as exc:  # noqa: BLE001
                last_err = f"{label}: {exc}"
                logger.warning("telegram_volume_refresh_window_failed window=%s err=%s", label, exc)
        with self._volume_lock:
            if results["7d"] is not None:
                self._volume_7d_usd = results["7d"]
            if results["30d"] is not None:
                self._volume_30d_usd = results["30d"]
            if results["7d"] is not None or results["30d"] is not None:
                self._volume_refreshed_at_mono = self._clock()
            self._volume_last_error = last_err

    # --- main loop ---

    def _run(self) -> None:
        # Backoff state for transient HTTP errors. Exponential with cap.
        backoff_s = 1.0
        max_backoff_s = 60.0
        while not self._stop.is_set():
            try:
                updates = self._fetch_updates()
            except Exception as exc:  # noqa: BLE001
                logger.warning("telegram_get_updates_failed err=%s", exc)
                # Sleep with stop-awareness so shutdown is responsive.
                if self._stop.wait(backoff_s):
                    return
                backoff_s = min(max_backoff_s, backoff_s * 2.0)
                continue
            backoff_s = 1.0  # reset on success
            for upd in updates:
                if self._stop.is_set():
                    return
                try:
                    self._handle_update(upd)
                except Exception:
                    logger.exception("telegram_handle_update_failed")

    def _fetch_updates(self) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "timeout": self._poll_timeout_s,
            # Skip channel_post / edited_*; we only react to messages
            # in DMs / groups from allowed users.
            "allowed_updates": json.dumps(["message"]),
        }
        with self._lock:
            if self._offset:
                params["offset"] = self._offset
        url = f"https://api.telegram.org/bot{self._token}/getUpdates"
        timeout = self._poll_timeout_s + _HTTP_TIMEOUT_HEADROOM_SECONDS
        status, body = self._http_get(url, params, timeout)
        if status != 200:
            logger.warning(
                "telegram_get_updates_status=%d body=%s",
                status,
                (body or "")[:300],
            )
            return []
        try:
            j = json.loads(body or "{}")
        except json.JSONDecodeError:
            return []
        if not j.get("ok"):
            logger.warning("telegram_get_updates_not_ok body=%s", (body or "")[:300])
            return []
        return list(j.get("result") or [])

    # --- update handling ---

    def _handle_update(self, upd: dict[str, Any]) -> None:
        update_id = int(upd.get("update_id") or 0)
        # Always advance the cursor — even rejected messages must not
        # be re-delivered, otherwise an attacker could spam the auth log.
        with self._lock:
            self._offset = max(self._offset, update_id + 1)

        msg = upd.get("message")
        if not isinstance(msg, dict):
            return

        text = (msg.get("text") or "").strip()
        if not text or not text.startswith("/"):
            return

        from_obj = msg.get("from") or {}
        chat_obj = msg.get("chat") or {}
        try:
            user_id = int(from_obj.get("id"))
        except (TypeError, ValueError):
            return
        try:
            chat_id = int(chat_obj.get("id"))
        except (TypeError, ValueError):
            return

        if not self._is_authorized(user_id=user_id, chat_id=chat_id):
            logger.warning(
                "telegram_command_rejected user=%s chat=%s",
                user_id,
                chat_id,
            )
            # Don't reply — silent rejection denies attackers a probe oracle.
            return

        if not self._check_rate_limit(user_id):
            self._reply(chat_id, "rate-limited; slow down")
            return

        self._dispatch(user_id=user_id, chat_id=chat_id, text=text)

    def _is_authorized(self, *, user_id: int, chat_id: int) -> bool:
        if user_id not in self._allowed_user_ids:
            return False
        if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids:
            return False
        return True

    def _check_rate_limit(self, user_id: int) -> bool:
        now = self._clock()
        with self._lock:
            last = self._last_cmd_mono.get(user_id, 0.0)
            if now - last < self._min_interval_s:
                return False
            self._last_cmd_mono[user_id] = now
            return True

    # --- dispatch ---

    _DESTRUCTIVE_COMMANDS = {"kill", "flatten", "restart"}

    def _dispatch(self, *, user_id: int, chat_id: int, text: str) -> None:
        parts = text.split()
        cmd_full = parts[0].lstrip("/").lower()
        # Telegram allows ``/cmd@botname`` form in groups; strip the
        # bot suffix so handlers don't have to care.
        if "@" in cmd_full:
            cmd_full = cmd_full.split("@", 1)[0]
        args = parts[1:]

        # Two-step confirmation handling.
        if cmd_full in self._DESTRUCTIVE_COMMANDS:
            if args and args[0].lower() == "confirm":
                if not self._consume_pending_confirm(user_id, cmd_full):
                    self._reply(
                        chat_id,
                        f"no pending /{cmd_full} (or it expired). re-send /{cmd_full} to start over",
                    )
                    return
                # Authorized confirmation — execute.
                self._handle_destructive(
                    user_id=user_id, chat_id=chat_id, cmd=cmd_full
                )
                return
            # First request — arm the confirm slot.
            self._arm_pending_confirm(
                user_id=user_id, chat_id=chat_id, command=cmd_full
            )
            self._reply(
                chat_id,
                f"\u26a0\ufe0f /{cmd_full} requested. Confirm with `/{cmd_full} confirm` "
                f"within {int(self._confirm_timeout_s)}s, or this request will expire.",
            )
            return

        handlers: dict[str, Callable[[int, int, list[str]], None]] = {
            "help": self._cmd_help,
            "start": self._cmd_help,  # Telegram's default greeting on first DM
            "version": self._cmd_version,
            "status": self._cmd_status,
            "stats": self._cmd_stats,
            "fills": self._cmd_fills,
            "position": self._cmd_position,
            "equity": self._cmd_equity,
            "health": self._cmd_health,
            "latency": self._cmd_latency,
            "config": self._cmd_config,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
        }
        h = handlers.get(cmd_full)
        if h is None:
            self._reply(chat_id, f"unknown command /{cmd_full}. Try /help")
            return
        try:
            h(user_id, chat_id, args)
        except Exception as exc:
            logger.exception("telegram_command_handler_failed cmd=%s", cmd_full)
            self._reply(chat_id, f"error handling /{cmd_full}: {exc}")

    # --- two-step confirm helpers ---

    def _arm_pending_confirm(
        self, *, user_id: int, chat_id: int, command: str
    ) -> None:
        deadline = self._clock() + self._confirm_timeout_s
        with self._lock:
            self._pending_confirms[(user_id, command)] = _PendingConfirm(
                command=command,
                user_id=user_id,
                chat_id=chat_id,
                deadline_mono=deadline,
            )

    def _consume_pending_confirm(self, user_id: int, command: str) -> bool:
        now = self._clock()
        with self._lock:
            pending = self._pending_confirms.pop((user_id, command), None)
        if pending is None:
            return False
        if now > pending.deadline_mono:
            return False
        return True

    def _handle_destructive(
        self, *, user_id: int, chat_id: int, cmd: str
    ) -> None:
        if cmd == "kill":
            try:
                self._bot.kill(reason="manual_kill_via_telegram")
            except Exception as exc:
                self._reply(chat_id, f"kill failed: {exc}")
                return
            self._reply(chat_id, "\U0001f534 bot killed (manual via Telegram)")
            return
        if cmd == "flatten":
            try:
                result = self._bot.flatten(blocking=True)
            except Exception as exc:
                self._reply(chat_id, f"flatten failed: {exc}")
                return
            txt = result.value if result is not None and hasattr(result, "value") else str(result)
            self._reply(chat_id, f"flatten complete result={txt}")
            return
        if cmd == "restart":
            self._reply(
                chat_id,
                "\u267b\ufe0f restarting now (process will exit; container "
                "manager will bring it back)",
            )
            # Best-effort flush of any in-flight notifier batch before
            # we hard-exit, so the trades / ops channels see this restart.
            try:
                if self._notifier is not None:
                    self._notifier.notify_ops(
                        "WARNING",
                        "manual_restart",
                        "operator-triggered restart via Telegram",
                        {"user_id": user_id},
                    )
                    # Allow the worker thread a brief moment to drain.
                    time.sleep(0.5)
            except Exception:
                pass
            self._exit_fn(RESTART_EXIT_CODE)
            return

    # --- read-only / reversible command handlers ---

    def _cmd_help(self, user_id: int, chat_id: int, args: list[str]) -> None:
        text = (
            "Commands:\n"
            "  /status — bot status, position, drawdown, last fill\n"
            "  /stats — session PnL summary\n"
            "  /stats day — today's UTC totals\n"
            "  /fills [N] — last N fills (default 10)\n"
            "  /position — current position\n"
            "  /equity — current equity / cash\n"
            "  /health — WS connections + toxicity\n"
            "  /latency — L2 tick + order RTT distributions (μs/ms)\n"
            "  /config [filter] — current settings (secrets redacted)\n"
            "  /pause — pause quoting (reversible)\n"
            "  /resume — resume quoting\n"
            "  /flatten — close position via taker \u26a0\ufe0f confirm\n"
            "  /kill — hard kill \u26a0\ufe0f confirm\n"
            "  /restart — soft restart \u26a0\ufe0f confirm\n"
            "  /version — git version\n"
            "  /help — this message\n"
        )
        self._reply(chat_id, text)

    def _cmd_version(self, user_id: int, chat_id: int, args: list[str]) -> None:
        self._reply(chat_id, f"version {__version__}")

    def _cmd_status(self, user_id: int, chat_id: int, args: list[str]) -> None:
        flags = self._safe(self._state.status_flags_dict, default={})
        pos = self._safe(self._state.position_dict, default={})
        pnl = self._safe(self._state.pnl_dict, default={})
        sid = str(
            self._safe(lambda: getattr(self._state, "session_id", "?"), default="?")
        )
        fills = int(
            self._safe(
                lambda: int(getattr(self._state, "session_fill_count", 0)),
                default=0,
            )
        )
        # Session runtime as hh:mm:ss. ``session_started_at_utc`` is a
        # tz-aware datetime; produce the delta against ``utc_now`` and
        # format. Negative deltas (clock skew) clamp to 0.
        runtime_str = "?"
        started = self._safe(
            lambda: getattr(self._state, "session_started_at_utc", None),
            default=None,
        )
        if started is not None:
            try:
                delta_seconds = int(
                    max(
                        0.0,
                        (_clock.now_utc() - started).total_seconds(),
                    )
                )
                h, rem = divmod(delta_seconds, 3600)
                m, s = divmod(rem, 60)
                runtime_str = f"{h:02d}:{m:02d}:{s:02d}"
            except Exception:  # noqa: BLE001
                pass
        # New-order placements only (NOT cancels). Bumped every time
        # the execution layer dispatches ``place_post_only_limit``,
        # whether the call succeeds or rejects. Use this as the
        # liveliness heartbeat: low-but-growing means the bot is
        # actively quoting; flat means execution is stuck.
        new_orders = int(
            self._safe(
                lambda: int(
                    getattr(self._state, "session_place_attempt_count", 0)
                ),
                default=0,
            )
        )
        # Combined exchange-action count (placements + cancels). Useful
        # as a second heartbeat showing how many round-trips the bot
        # has issued, including the cancel half of each cancel-and-
        # replace cycle. The dispatcher does not separate place vs
        # cancel here, so this number > new_orders by design.
        ws_actions = int(
            self._safe(
                lambda: int(
                    getattr(self._state, "outbound_ws_action_send_count", 0)
                ),
                default=0,
            )
        )
        http_actions = int(
            self._safe(
                lambda: int(
                    getattr(self._state, "outbound_http_action_send_count", 0)
                ),
                default=0,
            )
        )
        actions_total = ws_actions + http_actions
        # Multi-minute drift gate reading (BUGS/bug-002.md). None
        # before the long window has warmed up; signed bps once active.
        long_drift = flags.get("mid_return_long_window_bps")
        long_drift_str = self._fmt_bps(long_drift) if long_drift is not None else "warmup"

        # v1.5.26 Phase 2A.3 -- regime-mode line on /status. Pulls
        # mode + dwell + last-transition-reason directly from the
        # controller so a stale ``snapshot_dict`` cache can't lag
        # the live mode by a tick. ``mode_dwell_str`` reads
        # human-friendly ("12m 47s" / "8h 47m") so the line fits the
        # phone screen. Format example:
        #   regime: CAUTIOUS · 2m 14s · vol_slope:1.20bp/min
        regime_mode_str = "?"
        try:
            rc = getattr(self._state, "regime_controller", None)
            if rc is not None:
                mode_val = getattr(getattr(rc, "mode", None), "value", "?")
                # Dwell since ``mode_since_mono`` (per snapshot_dict's
                # ``seconds_in_mode`` derivation). Defensive: 0.0 when
                # the controller hasn't initialised yet.
                mode_since = float(getattr(rc, "mode_since_mono", 0.0) or 0.0)
                if mode_since > 0:
                    dwell_s = max(0.0, _clock.monotonic() - mode_since)
                else:
                    dwell_s = 0.0
                if dwell_s < 60.0:
                    dwell_str = f"{int(dwell_s)}s"
                elif dwell_s < 3600.0:
                    m, s = divmod(int(dwell_s), 60)
                    dwell_str = f"{m}m {s}s"
                else:
                    h, rem = divmod(int(dwell_s), 3600)
                    m = rem // 60
                    dwell_str = f"{h}h {m}m"
                reason = (
                    str(getattr(rc, "last_transition_reason", "") or "").strip()
                )
                if reason:
                    regime_mode_str = f"{mode_val} · {dwell_str} · {reason}"
                else:
                    regime_mode_str = f"{mode_val} · {dwell_str}"
        except Exception:  # noqa: BLE001 -- defensive; /status mustn't crash
            pass

        # v1.5.33 — take-profit row. Format examples:
        #   tp: off
        #   tp: armed 0/0/0 (active uPnL=+47.3bps · dwell=4.2s)
        #   tp: armed 12/8/3 · avg-fill=+52.1bps
        # Order: armed_total / filled_total / exited_unfilled_total.
        tp_str = "off"
        try:
            tp_enabled = bool(
                getattr(self._settings, "upnl_harvest_enabled", False)
            )
            if tp_enabled:
                armed = int(getattr(self._state, "tp_armed_total", 0))
                filled = int(getattr(self._state, "tp_filled_total", 0))
                unfilled = int(
                    getattr(self._state, "tp_exited_unfilled_total", 0)
                )
                tp_str = f"armed {armed}/{filled}/{unfilled}"
                # Active-now decoration.
                if bool(getattr(self._state, "tp_active", False)):
                    entry_upnl = getattr(
                        self._state, "tp_entry_upnl_bps", None
                    )
                    armed_at = float(
                        getattr(self._state, "tp_armed_at_mono", 0.0)
                        or 0.0
                    )
                    dwell = max(
                        0.0,
                        _clock.monotonic() - armed_at,
                    ) if armed_at > 0 else 0.0
                    tp_str += (
                        f" (active uPnL="
                        f"{entry_upnl:+.1f}bps · dwell={dwell:.1f}s)"
                        if entry_upnl is not None
                        else f" (active dwell={dwell:.1f}s)"
                    )
                elif filled > 0:
                    avg_fill = (
                        float(
                            getattr(
                                self._state,
                                "tp_sum_upnl_bps_at_fill",
                                0.0,
                            )
                        )
                        / max(filled, 1)
                    )
                    tp_str += f" · avg-fill={avg_fill:+.1f}bps"
        except Exception:  # noqa: BLE001 -- defensive
            pass

        rows = [
            ("version", str(__version__)),
            ("status", str(flags.get("bot_status", "?"))),
            ("regime", regime_mode_str),
            ("tp", tp_str),
            ("killed", str(flags.get("killed", False))),
            ("pause", str(flags.get("manual_pause", False))),
            ("flatten", str(flags.get("flatten_mode", False))),
            ("session", sid[:8] if sid else "?"),
            ("runtime", runtime_str),
            ("symbol", str(pos.get("symbol", ""))),
            ("position", str(pos.get("position_qty", 0))),
            ("entry", str(pos.get("avg_entry_price"))),
            ("mark", str(pos.get("mark_price"))),
            ("realized", f"${self._fmt_money(pnl.get('realized_pnl_usd'))}"),
            ("fees", f"${self._fmt_money(pnl.get('fees_usd'))}"),
            ("drawdown", f"${self._fmt_money(pnl.get('drawdown_usd'))}"),
            # Account-value snapshot. ``equity`` = total account
            # value (collateral + unrealized PnL); ``collateral`` =
            # free / withdrawable balance not currently locked as
            # initial margin. Same fields as the dedicated ``/equity``
            # command, surfaced here so the operator doesn't have to
            # switch commands to see whether the account is healthy.
            ("equity", f"${self._fmt_money(pnl.get('equity_usd'))}"),
            ("collateral", f"${self._fmt_money(pnl.get('withdrawable_usd'))}"),
            ("trend_5m", long_drift_str),
            ("fills", str(fills)),
            ("new_orders", str(new_orders)),
            ("actions", f"{actions_total} (place+cancel)"),
            (
                "blind_resting_s",
                f"{flags.get('blind_resting_seconds_total', 0.0):.1f}",
            ),
            # v1.4.93 wedge-elimination-cleanup Phase 6D.3 — "did the
            # bot wedge at all this session?" Operator-facing summary:
            # 0 = clean session; non-zero = N entries into a non-trivial
            # risk state (CANCELLING / SUPPRESSED) this session.
            (
                "wedge_episodes",
                str(int(flags.get("wedge_episode_count_session", 0))),
            ),
        ]
        # amend-prio Phase 4 (v1.4.17): amend rollout summary. Hidden
        # entirely when the knob is off (= zero emitted intents) so
        # pre-rollout deployments don't show a misleading "0/0/0" row.
        # The format is compact — one line, fits a phone screen:
        #   amends: 124 ok / 3 below-filled / 1 cross-rejected / 0 other
        amend_emitted = int(flags.get("amend_intents_emitted_total", 0))
        if amend_emitted > 0:
            ok = int(flags.get("amend_success_total", 0))
            bf = int(flags.get("amend_below_filled_total", 0))
            cr = int(flags.get("amend_post_only_cross_total", 0))
            other = int(flags.get("amend_exchange_rejected_other_total", 0))
            tr = int(flags.get("amend_transport_rejected_total", 0))
            gone = int(flags.get("amend_order_gone_total", 0))
            hwm = int(flags.get("amend_pending_high_watermark", 0))
            rows.append(
                (
                    "amends",
                    f"{ok} ok / {bf} below-filled / {cr} cross-rejected / "
                    f"{other} other-rej / {tr} transport-rej / {gone} gone "
                    f"(emitted {amend_emitted}, hwm {hwm})",
                )
            )
        # TODO-003: rolling net-edge-after-fees on /status.
        # Show only when a meaningful sample exists (the underlying summary
        # returns None below 5 fills with markouts) so the channel doesn't
        # advertise noise during the warmup window of a fresh session.
        ne = flags.get("net_edge_summary") or {}
        if ne.get("net_edge_bps") is not None:
            rows.append(
                (
                    "net_edge",
                    f"{ne['net_edge_bps']:+.2f} bps "
                    f"(gross {ne.get('mean_markout_bps', 0):+.2f} − fees "
                    f"{ne.get('mean_round_trip_fee_bps', 0):.2f}, "
                    f"n={ne.get('net_edge_sample_count', 0)})",
                )
            )

        # Market-regime block. Enough state for the operator (or me, when
        # asked verbally) to reason about why the bot is or isn't trading
        # without pulling klines. Placed at the bottom so the human PnL/
        # liveness numbers above are reached first on small screens.
        rows.append(("--- market ---", ""))
        rows.append(
            (
                "vol",
                _fmt_vol_with_threshold(
                    flags.get("short_vol_bps"),
                    threshold_bps=float(
                        getattr(self._settings, "reprice_threshold_bps", 0) or 0
                    ),
                ),
            )
        )
        spread_bps = flags.get("venue_spread_bps")
        rows.append(
            (
                "spread",
                "?" if spread_bps is None else f"{float(spread_bps):.2f} bps",
            )
        )
        basis = flags.get("basis_to_binance_bps")
        rows.append(
            (
                "basis",
                "?"
                if basis is None
                else f"{float(basis):+.2f} bps (vs Binance)",
            )
        )
        active_sides = flags.get("active_sides") or "?"
        rows.append(("active_sides", str(active_sides)))
        book_age = flags.get("book_age_ms")
        rows.append(
            (
                "book_age",
                "?" if book_age is None else f"{float(book_age):.0f} ms",
            )
        )
        pos_pct = flags.get("position_notional_pct_of_cap")
        pos_notional = flags.get("position_notional_usd")
        cap = flags.get("max_position_notional_usd")
        if pos_pct is not None and pos_notional is not None and cap is not None:
            rows.append(
                (
                    "pos_cap",
                    f"${pos_notional:.2f} / ${cap:.0f} ({pos_pct:.0f}%)",
                )
            )
        tox = flags.get("toxicity_score")
        if tox is not None:
            rows.append(("toxicity", f"{float(tox):.3f}"))

        # 7d / 30d account volume from the cache (refreshed every
        # ``TELEGRAM_VOLUME_REFRESH_SECONDS`` in the background). Hidden
        # entirely when no client is wired in or the cache is empty
        # (first refresh hasn't completed yet) so an unconfigured
        # environment doesn't surface a misleading "$0".
        with self._volume_lock:
            v7 = self._volume_7d_usd
            v30 = self._volume_30d_usd
            v_age = self._volume_refreshed_at_mono
            v_err = self._volume_last_error
        if v7 is not None or v30 is not None:
            rows.append(("--- volume ---", ""))
            if v7 is not None:
                rows.append(("vol_7d", f"${v7:,.2f}"))
            if v30 is not None:
                rows.append(("vol_30d", f"${v30:,.2f}"))
            if v_age is not None:
                age_min = max(0.0, (self._clock() - v_age) / 60.0)
                rows.append(("vol_age", f"{age_min:.0f} min"))
            if v_err:
                rows.append(("vol_err", v_err[:60]))

        self._reply(chat_id, self._format_block(rows), parse_mode="Markdown")

    def _cmd_stats(self, user_id: int, chat_id: int, args: list[str]) -> None:
        scope = (args[0].lower() if args else "session")
        if scope == "day":
            d_anchor = self._safe(
                lambda: self._state.operator_day_anchor_utc.isoformat(),
                default="?",
            )
            rows = [
                ("day (UTC)", str(d_anchor)),
                (
                    "realized",
                    f"${self._fmt_money(getattr(self._state, 'daily_realized_pnl', None))}",
                ),
                ("trades", str(getattr(self._state, "daily_trade_count", 0))),
                (
                    "notional",
                    f"${self._fmt_money(getattr(self._state, 'daily_traded_notional', None))}",
                ),
                ("buys", str(getattr(self._state, "recent_buy_fill_count", 0))),
                ("sells", str(getattr(self._state, "recent_sell_fill_count", 0))),
            ]
            self._reply(
                chat_id, self._format_block(rows), parse_mode="Markdown"
            )
            return
        # session scope
        pnl = self._safe(self._state.pnl_dict, default={})
        sid = str(
            self._safe(lambda: getattr(self._state, "session_id", "?"), default="?")
        )
        started_dt = self._safe(
            lambda: getattr(self._state, "session_started_at_utc", None),
            default=None,
        )
        started_str = started_dt.isoformat() if started_dt is not None else "?"
        runtime_str = "?"
        if started_dt is not None:
            try:
                delta_seconds = int(
                    max(
                        0.0,
                        (_clock.now_utc() - started_dt).total_seconds(),
                    )
                )
                h, rem = divmod(delta_seconds, 3600)
                m, s = divmod(rem, 60)
                runtime_str = f"{h:02d}:{m:02d}:{s:02d}"
            except Exception:  # noqa: BLE001
                pass
        fills = int(
            self._safe(
                lambda: int(getattr(self._state, "session_fill_count", 0)),
                default=0,
            )
        )
        rows = [
            ("session", sid[:8] if sid else "?"),
            ("started", started_str),
            ("runtime", runtime_str),
            ("fills", str(fills)),
            ("realized", f"${self._fmt_money(pnl.get('realized_pnl_usd'))}"),
            ("fees", f"${self._fmt_money(pnl.get('fees_usd'))}"),
            ("total", f"${self._fmt_money(pnl.get('total_pnl_usd'))}"),
            (
                "peak equity",
                f"${self._fmt_money(pnl.get('session_peak_equity_usd'))}",
            ),
            ("drawdown", f"${self._fmt_money(pnl.get('drawdown_usd'))}"),
        ]
        self._reply(chat_id, self._format_block(rows), parse_mode="Markdown")

    def _cmd_fills(self, user_id: int, chat_id: int, args: list[str]) -> None:
        n = 10
        if args:
            try:
                n = max(1, min(50, int(args[0])))
            except (TypeError, ValueError):
                pass
        rows = self._safe(lambda: self._storage.recent_fills(limit=n), default=[])
        if not rows:
            self._reply(chat_id, "no fills")
            return
        lines = [f"last {len(rows)} fills:"]
        for r in rows:
            try:
                lines.append(
                    f"  {r.get('ts_fill','')} {r.get('side','?'):>4} {r.get('size','?')} @ {r.get('price','?')}"
                    f"  fee=${self._fmt_money(r.get('fee'))}  mk5s={self._fmt_bps(r.get('markout_5s_bps'))}"
                )
            except Exception:
                pass
        self._reply(chat_id, "\n".join(lines))

    def _cmd_position(self, user_id: int, chat_id: int, args: list[str]) -> None:
        pos = self._safe(self._state.position_dict, default={})
        rows = [
            ("symbol", str(pos.get("symbol", ""))),
            ("qty", str(pos.get("position_qty", 0))),
            ("entry", str(pos.get("avg_entry_price"))),
            ("mark", str(pos.get("mark_price"))),
            ("notional", f"${self._fmt_money(pos.get('position_notional'))}"),
            ("unrealized", f"${self._fmt_money(pos.get('unrealized_pnl_usd'))}"),
        ]
        self._reply(chat_id, self._format_block(rows), parse_mode="Markdown")

    def _cmd_equity(self, user_id: int, chat_id: int, args: list[str]) -> None:
        pnl = self._safe(self._state.pnl_dict, default={})
        rows = [
            ("equity", f"${self._fmt_money(pnl.get('equity_usd'))}"),
            ("withdrawable", f"${self._fmt_money(pnl.get('withdrawable_usd'))}"),
            ("realized", f"${self._fmt_money(pnl.get('realized_pnl_usd'))}"),
            ("unrealized", f"${self._fmt_money(pnl.get('unrealized_pnl_usd'))}"),
            ("fees", f"${self._fmt_money(pnl.get('fees_usd'))}"),
            ("drawdown", f"${self._fmt_money(pnl.get('drawdown_usd'))}"),
            ("peak", f"${self._fmt_money(pnl.get('session_peak_equity_usd'))}"),
        ]
        self._reply(chat_id, self._format_block(rows), parse_mode="Markdown")

    def _cmd_health(self, user_id: int, chat_id: int, args: list[str]) -> None:
        s = self._state
        rows = [
            (
                "private_ws",
                f"connected={getattr(s, 'private_ws_connected', False)} "
                f"healthy={getattr(s, 'private_ws_healthy', False)} "
                f"reconnects={getattr(s, 'private_ws_reconnect_count', 0)}",
            ),
            (
                "public_ws",
                f"connected={getattr(s, 'public_ws_connected', False)} "
                f"reconnects={getattr(s, 'public_ws_reconnect_count', 0)}",
            ),
            (
                "binance_ws",
                f"connected={getattr(s, 'binance_ws_connected', False)} "
                f"reconnects={getattr(s, 'binance_ws_reconnect_count', 0)}",
            ),
            (
                "tox_5s_bps",
                self._fmt_bps(
                    getattr(s, "operator_rolling_toxicity_markout_bps", None)
                ),
            ),
        ]
        self._reply(chat_id, self._format_block(rows), parse_mode="Markdown")

    def _cmd_latency(self, user_id: int, chat_id: int, args: list[str]) -> None:
        """Two distributions:

        1. **L2 tick latency** — exchange timestamp → local receive
           timestamp, computed in the trading-public-WS layer. Captures
           one-way "age at receipt" plus our local processing time
           (parse + on_bbo callback + state apply).
        2. **Order place-to-ack RTT** — ``time.perf_counter()`` delta
           from transport-send to transport-done in the execution
           layer. Microsecond-resolved under the hood, displayed in
           millisecond units with 3 decimal places (i.e. μs precision).

        Both expose **min / median / p95 / p99 / max** over a rolling
        window. Targets per the operator:
          * L2 tick: well below 10 ms one-way (typically 2-5 ms on
            AWS Tokyo same-region to Binance).
          * Order RTT: under 10 ms place-to-ack (typically 5-15 ms
            on AWS Tokyo same-region).
        """
        rows: list[tuple[str, str]] = []

        # --- L2 / market-data tick latency -----------------------------
        tracker = getattr(self._state, "binance_public_ws_timing", None)
        if tracker is None:
            rows.append(("l2_tick", "n/a (timing tracker not initialised)"))
        else:
            try:
                summary = tracker.summary()
            except Exception as exc:  # noqa: BLE001
                rows.append(("l2_tick_error", str(exc)[:100]))
                summary = None
            if summary is not None:
                exch_to_recv = (
                    summary.get("exchange_to_local_receive_ms") or {}
                )
                rows.append(("--- L2 tick (exchange → local) ---", ""))
                rows.append(
                    (
                        "samples",
                        f"{summary.get('total_samples') or 0} "
                        f"(window={summary.get('configured_window_size') or 0})",
                    )
                )
                rows.append(
                    (
                        "min",
                        self._fmt_ms_us(exch_to_recv.get("min")),
                    )
                )
                rows.append(
                    (
                        "median",
                        self._fmt_ms_us(exch_to_recv.get("median")),
                    )
                )
                rows.append(
                    (
                        "p95",
                        self._fmt_ms_us(exch_to_recv.get("p95")),
                    )
                )
                rows.append(
                    (
                        "max",
                        self._fmt_ms_us(exch_to_recv.get("max")),
                    )
                )
                # Local processing slice (receive → state apply)
                r2a = summary.get("receive_to_apply_ms") or {}
                if r2a.get("median") is not None:
                    rows.append(("--- local proc (recv → apply) ---", ""))
                    rows.append(
                        ("median", self._fmt_ms_us(r2a.get("median")))
                    )
                    rows.append(
                        ("p95", self._fmt_ms_us(r2a.get("p95")))
                    )
                    rows.append(
                        ("max", self._fmt_ms_us(r2a.get("max")))
                    )

        # --- Order place-to-ack RTT ------------------------------------
        bot = self._bot
        exec_layer = getattr(bot, "_exec", None) if bot is not None else None
        rtt_summary = None
        if exec_layer is not None:
            try:
                rtt_summary = exec_layer.order_rtt_summary()
            except Exception as exc:  # noqa: BLE001
                rows.append(("order_rtt_error", str(exc)[:100]))
        rows.append(("--- order RTT (place → ack) ---", ""))
        if rtt_summary is None or rtt_summary.get("sample_count", 0) == 0:
            rows.append(
                (
                    "samples",
                    f"0 (window={rtt_summary.get('window_size') if rtt_summary else 1024})",
                )
            )
            rows.append(("status", "no order placements observed yet"))
        else:
            rows.append(
                (
                    "samples",
                    f"{rtt_summary['sample_count']} "
                    f"(window={rtt_summary['window_size']}, "
                    f"total={rtt_summary['total_observed_count']})",
                )
            )
            rows.append(("min", self._fmt_ms_us(rtt_summary.get("min_ms"))))
            rows.append(
                ("median", self._fmt_ms_us(rtt_summary.get("median_ms")))
            )
            rows.append(("p95", self._fmt_ms_us(rtt_summary.get("p95_ms"))))
            rows.append(("p99", self._fmt_ms_us(rtt_summary.get("p99_ms"))))
            rows.append(("max", self._fmt_ms_us(rtt_summary.get("max_ms"))))
            rows.append(
                ("mean", self._fmt_ms_us(rtt_summary.get("mean_ms")))
            )

        self._reply(chat_id, self._format_block(rows), parse_mode="Markdown")

    @staticmethod
    def _fmt_ms_us(v: Any) -> str:
        """Render a millisecond value with microsecond precision when
        the magnitude is small enough that whole-ms rounding would
        hide variation. Strategy:

          * < 1 ms → show in μs (``"347 μs"``)
          * 1-100 ms → show with 2 decimals (``"5.31 ms"``)
          * ≥ 100 ms → show with 1 decimal (``"123.5 ms"``)
          * None / NaN → ``"n/a"``
        """
        if v is None:
            return "n/a"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "n/a"
        if f != f:  # NaN
            return "n/a"
        if f < 1.0:
            return f"{f * 1000:.0f} μs"
        if f < 100.0:
            return f"{f:.2f} ms"
        return f"{f:.1f} ms"

    # Field-name fragments that mark a value as a secret to redact in
    # ``/config``. Substring match on the lower-cased field name; exact
    # matches in the explicit set always redact regardless of substring.
    _CONFIG_SECRET_FRAGMENTS = (
        "private_key",
        "secret_key",
        "api_secret",
        "api_key",
        "_token",
        "password",
    )

    @classmethod
    def _is_config_secret_field(cls, name: str) -> bool:
        """True if ``name`` is a Settings field whose value must be redacted
        in ``/config`` output. Public addresses (``*_account_address``) and
        URLs are NOT secret. The exact name is matched against a fragment
        list so we don't have to hand-maintain a per-field allowlist as new
        venues land.
        """
        n = name.lower()
        return any(frag in n for frag in cls._CONFIG_SECRET_FRAGMENTS)

    @staticmethod
    def _redact_config_value(value: object) -> str:
        """Render a redacted secret as ``"<redacted N chars>"`` so the
        operator can confirm a value is set without exposing the secret.
        Empty string renders as ``"<empty>"`` so the operator can spot
        missing credentials at a glance.
        """
        if value is None:
            return "<unset>"
        s = str(value)
        if not s:
            return "<empty>"
        return f"<redacted {len(s)} chars>"

    def _cmd_config(self, user_id: int, chat_id: int, args: list[str]) -> None:
        """Dump the current Settings model with secrets redacted.

        Optional positional arg is a substring filter applied case-insensitively
        to field names — useful because the full dump runs to ~150 fields.
        Telegram messages cap at 4096 characters; the dump is split into
        multiple replies if needed.

        Public addresses (``hl_account_address``, ``bluefin_account_address``,
        etc.) are NOT redacted — they're public-chain identifiers and the
        operator needs them to verify the configured account.

        Rendered with HTML ``<pre>`` blocks rather than legacy Markdown.
        Markdown's underscore-as-italic rule misfires on the 100+ underscored
        field names (``MAX_POSITION_NOTIONAL_USD`` etc.) — Telegram rejects
        any message with an unbalanced ``_`` pair with HTTP 400, which
        manifests as "no reply" to the operator. HTML escaping needs only
        ``<``, ``>``, ``&`` (which we'd also need for the redacted-marker
        rendering ``<redacted N chars>``), so it's both more reliable and
        no more code.
        """
        try:
            data = dict(self._settings.model_dump())
        except Exception as exc:  # noqa: BLE001
            self._reply(chat_id, f"config dump failed: {exc}")
            return
        flt = (args[0].strip().lower() if args else "")
        rows: list[tuple[str, str]] = []
        for key in sorted(data.keys()):
            if flt and flt not in key.lower():
                continue
            value = data[key]
            if self._is_config_secret_field(key):
                rendered = self._redact_config_value(value)
            else:
                rendered = "<unset>" if value is None else str(value)
            rows.append((key.upper(), rendered))
        if not rows:
            self._reply(
                chat_id,
                f"no settings match filter {flt!r}" if flt else "no settings found",
            )
            return
        chunks = self._chunk_rows_for_telegram(rows)
        header_prefix = (
            f"config (filter={flt!r}, {len(rows)} fields)"
            if flt
            else f"config ({len(rows)} fields, secrets redacted)"
        )
        for i, chunk in enumerate(chunks, start=1):
            header = (
                f"{header_prefix} — part {i}/{len(chunks)}"
                if len(chunks) > 1
                else header_prefix
            )
            body_html = self._format_block_html(chunk)
            # Escape the header line too (``filter='...'`` could in theory
            # contain HTML-meaningful chars).
            from html import escape as _html_escape

            self._reply(
                chat_id,
                f"{_html_escape(header)}\n{body_html}",
                parse_mode="HTML",
            )

    @staticmethod
    def _format_block_html(rows: list[tuple[str, str]]) -> str:
        """Same alignment as ``_format_block`` but in HTML ``<pre>`` so
        Telegram's HTML parse mode can render it as monospace without the
        underscore-as-italic landmine that legacy Markdown has.
        """
        from html import escape as _html_escape

        if not rows:
            return "<pre>(empty)</pre>"
        max_k = max(len(str(k)) for k, _ in rows)
        lines = [
            _html_escape(f"{str(k):<{max_k}}  {v}") for k, v in rows
        ]
        return "<pre>" + "\n".join(lines) + "</pre>"

    @staticmethod
    def _chunk_rows_for_telegram(
        rows: list[tuple[str, str]],
        *,
        max_chars: int = 3500,
    ) -> list[list[tuple[str, str]]]:
        """Split rows into chunks that each render under Telegram's 4096-char
        cap AND under ``_reply``'s 3900-char truncation guardrail.

        Earlier versions of this estimator under-counted per-row size by
        using ``len(key)`` directly — but ``_format_block_html`` pads every
        key to the longest one in the chunk. With field names ranging from
        ~6 chars to ~50 chars, that's 30+ invisible bytes per short row,
        and the full Settings dump overran the 3900-char cap by ~15
        bytes — manifesting in production as silent truncation that cut
        secrets mid-line and made ``/config`` reply look broken.

        Fix: compute the longest key length up front and budget every row
        as ``max_k + 2 + len(value) + 1`` (the actual rendered line
        length), plus an HTML-escaping inflation allowance for values
        containing ``<``, ``>``, ``&``.
        """
        if not rows:
            return []
        max_k = max(len(str(k)) for k, _ in rows)
        chunks: list[list[tuple[str, str]]] = []
        cur: list[tuple[str, str]] = []
        cur_len = 0
        for row in rows:
            v = str(row[1])
            # HTML-escaping inflation: <,>,& each → 4 chars (&lt; / &gt; /
            # &amp;). Count occurrences to size accurately rather than
            # estimating an average.
            escape_inflation = sum(v.count(c) * 3 for c in "<>&")
            row_len = max_k + 2 + len(v) + 1 + escape_inflation
            if cur_len + row_len > max_chars and cur:
                chunks.append(cur)
                cur = []
                cur_len = 0
            cur.append(row)
            cur_len += row_len
        if cur:
            chunks.append(cur)
        return chunks

    def _cmd_pause(self, user_id: int, chat_id: int, args: list[str]) -> None:
        try:
            self._state.set_manual_pause(True)
            from app.utils.time import utc_now_iso  # local import to avoid hot reload

            self._storage.insert_bot_event(
                utc_now_iso(),
                "INFO",
                "control_pause_telegram",
                "manual pause via Telegram",
                {"user_id": user_id},
            )
        except Exception as exc:
            self._reply(chat_id, f"pause failed: {exc}")
            return
        self._reply(chat_id, "paused")

    def _cmd_resume(self, user_id: int, chat_id: int, args: list[str]) -> None:
        try:
            flags = self._state.status_flags_dict()
            if flags.get("killed"):
                self._reply(chat_id, "bot is killed; restart service first")
                return
            rebaselined = self._state.set_manual_pause(False)
            from app.utils.time import utc_now_iso

            self._storage.insert_bot_event(
                utc_now_iso(),
                "INFO",
                "control_resume_telegram",
                "manual resume via Telegram",
                {"user_id": user_id},
            )
            if rebaselined:
                # Audit event for the inventory-consistency rebaseline so the
                # operator can correlate "I closed manually on the venue" with
                # the bot's new anchor on grep / Telegram CRITICAL channels.
                baseline = float(getattr(self._state, "inventory_baseline_qty", 0.0) or 0.0)
                self._storage.insert_bot_event(
                    utc_now_iso(),
                    "INFO",
                    "control_resume_inventory_rebaselined",
                    f"inventory baseline rebaselined to venue qty {baseline:+.4f}",
                    {"user_id": user_id, "baseline_qty": baseline},
                )
        except Exception as exc:
            self._reply(chat_id, f"resume failed: {exc}")
            return
        suffix = ""
        if rebaselined:
            try:
                baseline = float(self._state.inventory_baseline_qty or 0.0)
                suffix = f" (inventory rebaselined to venue qty {baseline:+.4f})"
            except (TypeError, ValueError):
                suffix = " (inventory rebaselined)"
        self._reply(chat_id, f"resumed{suffix}")

    # --- helpers ---

    @staticmethod
    def _parse_int_csv(raw: str) -> set[int]:
        out: set[int] = set()
        for token in (raw or "").split(","):
            token = token.strip()
            if not token:
                continue
            try:
                out.add(int(token))
            except ValueError:
                continue
        return out

    @staticmethod
    def _safe(fn: Callable[[], Any], *, default: Any = None) -> Any:
        try:
            return fn()
        except Exception:
            return default

    @staticmethod
    def _fmt_money(v: Any) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    @staticmethod
    def _fmt_bps(v: Any) -> str:
        if v is None:
            return "n/a"
        try:
            return f"{float(v):.2f}bps"
        except (TypeError, ValueError):
            return str(v)

    @staticmethod
    def _format_block(rows: list[tuple[str, str]]) -> str:
        """Render ``[(key, value), ...]`` as a Markdown code block with
        keys aligned to a column.

        Telegram's "Markdown" parse mode renders triple-backtick blocks
        as monospaced text, which gives clean key/value alignment in
        DM chats. The keys are padded to the longest key length so
        every value lines up.
        """
        if not rows:
            return "```\n(empty)\n```"
        max_k = max(len(str(k)) for k, _ in rows)
        lines = [f"{str(k):<{max_k}}  {v}" for k, v in rows]
        body = "\n".join(lines)
        return f"```\n{body}\n```"

    def _reply(
        self,
        chat_id: int,
        text: str,
        parse_mode: Optional[str] = None,
    ) -> None:
        if not self._token:
            return
        # Telegram caps at 4096; truncate.
        if len(text) > 3900:
            text = text[:3900] + "\n...[truncated]"
        body: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if parse_mode:
            body["parse_mode"] = parse_mode
        try:
            status, resp = self._http_post(
                f"https://api.telegram.org/bot{self._token}/sendMessage",
                body,
            )
        except Exception as exc:
            logger.warning("telegram_reply_exception err=%s", exc)
            return
        if status != 200:
            logger.warning(
                "telegram_reply_failed status=%d chat=%s body=%s",
                status,
                chat_id,
                (resp or "")[:200],
            )

    @staticmethod
    def _default_http_get(
        url: str, params: dict[str, Any], timeout: float
    ) -> tuple[int, str]:
        # Stringify all values; long_polling expects ints, json strings, etc.
        encoded = urllib.parse.urlencode(
            {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in params.items()}
        )
        full = f"{url}?{encoded}" if encoded else url
        req = urllib.request.Request(full, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = int(resp.status)
                payload = resp.read().decode("utf-8", errors="replace")
                return status, payload
        except urllib.error.HTTPError as e:
            try:
                payload = e.read().decode("utf-8", errors="replace")
            except Exception:
                payload = ""
            return int(e.code), payload

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
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                status = int(resp.status)
                payload = resp.read().decode("utf-8", errors="replace")
                return status, payload
        except urllib.error.HTTPError as e:
            try:
                payload = e.read().decode("utf-8", errors="replace")
            except Exception:
                payload = ""
            return int(e.code), payload
