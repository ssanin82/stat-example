"""Periodic S3 heartbeat publisher.

Writes a tiny JSON object describing the bot's current session state
to ``s3://<logs_bucket>/heartbeat/<profile>.json`` every
``HEARTBEAT_INTERVAL_SECONDS`` seconds. The operator dashboard reads
this file to display "bot session uptime" -- distinct from EC2 host
uptime, which doesn't reset on ``ops deploy`` (the deploy restarts
the bot but leaves the EC2 instance up).

Why S3 and not the bot's HTTP API directly?

The bot binds its FastAPI server to ``0.0.0.0:8000`` but the EC2's
security group only allows the loopback path (no inbound from the
internet). The dashboard runs on the operator's laptop, so it can't
reach the bot directly without an SSM tunnel or a security-group
hole. S3 is already in the data-flow path (snapshots / log dumps go
there), the bot's IAM role already has ``s3:PutObject`` on the
target bucket, and reads from S3 are cheap + reliable.

Failure semantics: the heartbeat writer is strictly best-effort.
Boto3 ImportError, missing IAM permissions, network glitches, etc.
all log a warning and quietly disable the publisher. Trading is
never blocked or impacted by heartbeat failures.

Wire-up: ``app.main`` constructs a ``HeartbeatPublisher`` after
``BotState`` exists, calls ``start()`` once, and ``stop()`` during
shutdown. Same lifecycle as other daemon-thread workers (telegram
notifier, ws subscribers, etc.).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from app import __version__
from app.config import Settings
from app.state import BotState

from app import clock as _clock

logger = logging.getLogger(__name__)


def _safe_recording_status(settings: Any) -> dict[str, Any]:
    """Defensive wrapper around ``app.recording_status.recording_status``.

    The function itself is safe-by-construction, but the import is
    wrapped here too so a missing module on older deploys doesn't
    crash the heartbeat publisher.
    """
    try:
        from app.recording_status import recording_status
        return recording_status(settings)
    except Exception:
        return {
            "enabled": bool(getattr(settings, "recording_enabled", False)),
            "active": False,
            "session_name": None,
            "session_path": None,
            "bytes": 0,
            "files": 0,
            "profile_resolved": "",
        }


def _safe_gate_attribution(
    state: BotState, now_mono: float
) -> dict[str, dict[str, Any]]:
    """Best-effort wrapper around ``BotState.gate_attribution_snapshot``.
    Returns an empty dict if anything goes wrong so the heartbeat
    publisher never fails over a telemetry-only field.
    """
    try:
        return state.gate_attribution_snapshot(now_mono=now_mono)
    except Exception:
        logger.exception("gate_attribution_snapshot_failed")
        return {}


def _safe_spread_composition(state: BotState) -> Optional[dict[str, Any]]:
    """Best-effort wrapper around ``state.last_spread_composition.to_dict()``.
    Returns ``None`` until the first quote-cycle tick populates the
    composition, or on any unexpected error."""
    try:
        comp = getattr(state, "last_spread_composition", None)
        if comp is None:
            return None
        return comp.to_dict()
    except Exception:
        logger.exception("spread_composition_snapshot_failed")
        return None


def _now_iso() -> str:
    return _clock.now_utc().isoformat()


class HeartbeatPublisher:
    """Daemon-thread S3 heartbeat writer. Idempotent re-publishes
    every ``interval_seconds``; nothing else stores state worth
    persisting between iterations.
    """

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        bucket: str,
        profile_name: str,
    ) -> None:
        self._settings = settings
        self._state = state
        self._bucket = bucket.strip()
        self._profile_name = (profile_name or "unknown").strip() or "unknown"
        self._key = f"heartbeat/{self._profile_name}.json"
        self._interval_s = float(settings.heartbeat_interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_errors = 0
        self._client: Any = None  # boto3.client("s3"), lazily set

    @classmethod
    def maybe_create(
        cls,
        settings: Settings,
        state: BotState,
        profile_name: str,
    ) -> Optional["HeartbeatPublisher"]:
        """Construct iff the operator configured a logs bucket. Returns
        ``None`` (with a one-line log) when disabled so ``main`` can
        treat heartbeat as optional.
        """
        bucket = (settings.logs_bucket or "").strip()
        if not bucket:
            logger.info(
                "heartbeat_disabled reason=no_logs_bucket "
                "set_LOGS_BUCKET_to_enable"
            )
            return None
        return cls(settings, state, bucket, profile_name)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # Lazy boto3 import so the bot starts even if boto3 is missing
        # (pip-install lag, dev environments, etc.). Heartbeat is
        # never load-bearing for trading.
        try:
            import boto3  # type: ignore[import-untyped]

            self._client = boto3.client("s3")
        except Exception as e:
            logger.warning(
                "heartbeat_disabled reason=boto3_init_failed err=%s",
                str(e)[:200],
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="heartbeat",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "heartbeat_started bucket=%s key=%s interval_s=%.1f",
            self._bucket,
            self._key,
            self._interval_s,
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Publish once immediately on startup so the dashboard's first
        # poll lands a fresh heartbeat instead of waiting up to
        # ``interval_s`` for the first write.
        self._safe_publish_once()
        while not self._stop.wait(self._interval_s):
            self._safe_publish_once()

    def _safe_publish_once(self) -> None:
        try:
            self._publish_once()
            self._consecutive_errors = 0
        except Exception as e:
            self._consecutive_errors += 1
            # Log every failure for the first few; then back off to
            # debug-level to avoid log-spam if the bucket / IAM is
            # permanently misconfigured.
            level = (
                logging.WARNING if self._consecutive_errors <= 3 else logging.DEBUG
            )
            logger.log(
                level,
                "heartbeat_publish_failed errs=%d err=%s",
                self._consecutive_errors,
                str(e)[:200],
            )

    def _publish_once(self) -> None:
        if self._client is None:
            return
        # Snapshot state under the lock to avoid mid-mutation reads.
        with self._state._lock:
            session_id = self._state.session_id
            session_started = self._state.session_started_at_utc
            # ``BotStatus`` mixes ``str`` + ``Enum``, so ``str(member)``
            # returns "BotStatus.RUNNING" rather than "RUNNING". Use
            # ``.value`` (with a defensive fallback) so the heartbeat
            # JSON carries the bare enum value the dashboard compares
            # against.
            _bs = self._state.bot_status
            bot_status = getattr(_bs, "value", str(_bs))
            killed = self._state.killed
            kill_reason = self._state.kill_reason
            kill_ts = self._state.kill_timestamp
            pause_reason = self._state.pause_reason
            pause_ts = self._state.pause_timestamp
            manual_pause = self._state.manual_pause
            symbol = self._state.symbol
            venue_leverage = self._state.venue_leverage
            venue_margin_mode = self._state.venue_margin_mode
            venue_position_mode = self._state.venue_position_mode
            ws_timing_tracker = self._state.public_ws_timing
            rtt_provider = self._state.order_rtt_summary_provider
            # 1.4.0 cancel-prio Phase 0.5: sibling cancel-RTT provider.
            cancel_rtt_provider = self._state.cancel_rtt_summary_provider
            # v1.4.58 todo-037: unified TX → ack provider across all
            # three op kinds (place + amend + cancel). Read by the
            # dashboard's single combined LATENCY row.
            tx_rtt_provider = getattr(
                self._state, "tx_rtt_summary_provider", None
            )
            # Session activity counters — same fields shown by
            # Telegram /status (``app/telegram_commands.py``). Cheap
            # int reads under the same lock; published every 30s so
            # the operator dashboard can show them next to the
            # Session PnL chip without polling Telegram.
            session_fill_count = int(
                getattr(self._state, "session_fill_count", 0) or 0
            )
            session_place_attempt_count = int(
                getattr(self._state, "session_place_attempt_count", 0) or 0
            )
            ws_actions = int(
                getattr(self._state, "outbound_ws_action_send_count", 0) or 0
            )
            http_actions = int(
                getattr(self._state, "outbound_http_action_send_count", 0) or 0
            )
            session_action_count = ws_actions + http_actions
            session_traded_notional_usd = float(
                getattr(self._state, "session_traded_notional_usd", 0.0)
                or 0.0
            )
            # Phase 5 ("healthy but not trading" indicator): elapsed
            # since the last outbound execution ATTEMPT (place or
            # amend, NOT cancel). The deadlock watchdog uses the same
            # field; here we publish it for the operator dashboard
            # to surface BEFORE the watchdog fires (typically at 600 s).
            # ``None`` until the bot has issued its first attempt.
            #
            # v1.4.43 (BUG-025-adjacent follow-up): switched from the
            # place-only ``last_place_attempt_ts_mono`` to the broader
            # ``last_outbound_attempt_ts_mono`` (bumped on place AND
            # amend; cancels stay excluded so a cancels-only zombie
            # loop still trips). Pre-v1.4.43 the dashboard
            # "NEAR-DEADLOCK" chip false-fired during amend-heavy
            # quoting because the v1.4.16 amend-on-reprice path keeps
            # an order alive via continuous amends without dispatching
            # a fresh place — observed 2026-05-18, snapshot
            # v1.4.42-260518-131017: dashboard chip showed idle=305s
            # while ``executor_state.execution_idle_s = 0.016`` (16 ms!).
            # The watchdog + silent-wedge detector both got the new
            # field in v1.4.42; the heartbeat (this code) was the
            # missing third reader. Fallback to the legacy field
            # preserves back-compat with pre-v1.4.42 state objects.
            last_outbound_mono = float(
                getattr(self._state, "last_outbound_attempt_ts_mono", 0.0) or 0.0
            )
            if last_outbound_mono <= 0.0:
                last_outbound_mono = float(
                    getattr(self._state, "last_place_attempt_ts_mono", 0.0) or 0.0
                )
            now_mono = _clock.monotonic()
            execution_idle_seconds: Optional[float] = (
                round(now_mono - last_outbound_mono, 1)
                if last_outbound_mono > 0.0
                else None
            )
            # 60 s rolling suppression-rate from quote_quality. Used
            # together with ``execution_idle_seconds`` to distinguish
            # "calm market, nothing to do" (low suppression, long idle
            # — green) from "engine alive but suppression gates eating
            # decisions" (high suppression, long idle — amber/red).
            suppression_rate_60s: Optional[float] = None
            try:
                qq = getattr(self._state, "quote_quality", None)
                if qq is not None and hasattr(qq, "suppression_rate_60s"):
                    suppression_rate_60s = qq.suppression_rate_60s()
            except Exception:
                suppression_rate_60s = None
        body = {
            "schema_version": 1,
            "profile": self._profile_name,
            "venue": (self._settings.exchange or "").strip().lower(),
            "symbol": symbol,
            "version": __version__,
            "session_id": session_id,
            "session_started_at_utc": (
                session_started.isoformat() if session_started else None
            ),
            "bot_status": bot_status,
            "trading_enabled": bool(self._settings.trading_enabled),
            "killed": bool(killed),
            "kill_reason": kill_reason,
            "kill_timestamp_utc": kill_ts.isoformat() if kill_ts else None,
            "pause_reason": pause_reason,
            "pause_timestamp_utc": pause_ts.isoformat() if pause_ts else None,
            "manual_pause": bool(manual_pause),
            "venue_leverage": venue_leverage,
            "venue_margin_mode": venue_margin_mode,
            "venue_position_mode": venue_position_mode,
            "interval_seconds": self._interval_s,
            "heartbeat_at_utc": _now_iso(),
            "latency": _build_latency_block(
                ws_timing_tracker,
                rtt_provider,
                cancel_rtt_provider,
                tx_rtt_provider,
            ),
            # Session activity counters — match Telegram /status fields.
            # Display: dashboard renders ``fills / new_orders /
            # amend_intents / actions`` next to the Session PnL chip.
            # ``session_amend_intents_emitted_total`` was added in
            # v1.4.27 so the dashboard can show amend volume as a
            # SEPARATE column from new orders / total actions — the
            # combined ``actions`` was uninformative after amend
            # rollout dominated traffic.
            "session_fill_count": session_fill_count,
            "session_place_attempt_count": session_place_attempt_count,
            "session_amend_intents_emitted_total": int(
                getattr(self._state, "amend_intents_emitted_total", 0)
                or 0
            ),
            # v1.4.28: per-intent cancel count for the dashboard's
            # FILLS|ORDERS|CANCEL|VOLUME card. Bumped in
            # ``_enqueue_cancel_quote_path``. Parallel to
            # ``session_place_attempt_count``.
            "session_cancel_attempt_count": int(
                getattr(self._state, "session_cancel_attempt_count", 0)
                or 0
            ),
            "session_action_count": session_action_count,
            # Cumulative traded notional (USD) since session start.
            # Operator wants visibility on volume — useful for
            # estimating rebate income (volume × rebate_bps) and
            # for reasoning about exposure / churn.
            "session_traded_notional_usd": round(
                session_traded_notional_usd, 4
            ),
            # Phase 5 dashboard fields. ``execution_idle_seconds`` is
            # the time since the last place attempt (None when bot just
            # started). ``suppression_rate_60s`` is the fraction of the
            # last 60 s of quote cycles where any suppressor fired
            # (None when the rolling window is empty). Together they
            # let the dashboard show an early-warning chip before the
            # 600 s deadlock watchdog kills the bot. See
            # ``plans/20260507-sf-frontend.md`` Phase 5 for the chip
            # color rules.
            "execution_idle_seconds": execution_idle_seconds,
            "suppression_rate_60s": (
                round(suppression_rate_60s, 4)
                if suppression_rate_60s is not None
                else None
            ),
            # 1.4.7 Phase 0 gate-attribution: per-gate fire_count +
            # fire_seconds_total. Plain dict so the dashboard reads
            # it without schema changes. Empty when no gate has
            # fired yet (fresh session). See
            # ``BotState.gate_attribution_snapshot`` for the per-gate
            # shape and ``plans/gate-to-widening.md`` Phase 0 for
            # the dark-time baseline the operator computes from it.
            "gate_attribution": _safe_gate_attribution(
                self._state, now_mono
            ),
            # 1.4.9 gate-to-widening Phase 1: per-tick spread
            # composition (each contributor's bps, plus capped flags).
            # Operator dashboard reads this for the new "Spread
            # composition" widget specced in
            # plans/20260517-ui-correction-gates.md. ``None`` until
            # the first quote tick completes.
            "spread_composition": _safe_spread_composition(self._state),
            # v1.5.2: recording status (icon + size in the dashboard
            # header). Safe-by-construction; ``recording_status`` never
            # raises. Always present so the dashboard schema is
            # stable. ``active=False`` shape when recording is
            # disabled or the pointer file isn't on disk.
            "recording": _safe_recording_status(self._settings),
        }
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key,
            Body=json.dumps(body, indent=2).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-store, max-age=0",
        )


def _build_latency_block(
    ws_timing_tracker: Any,
    rtt_provider: Optional[Any],
    cancel_rtt_provider: Optional[Any] = None,
    tx_rtt_provider: Optional[Any] = None,
) -> dict[str, Any]:
    """Compact ``latency`` block for the heartbeat JSON.

    Four distributions are surfaced for the operator dashboard:
      * ``exchange_to_local_receive_ms`` — public-WS one-way delay
        (exchange-stamped event time → local receive wall time).
      * ``tx_submit_rtt_ms`` — v1.4.58 unified outbound transport RTT
        (place + amend + cancel pooled). The dashboard's headline
        LATENCY metric; replaces the per-op rows below.
      * ``order_submit_rtt_ms`` — place + amend RTT (kept for the
        postmortem path / Telegram ``/latency`` command). Pre-v1.4.58
        was the dashboard's "Place send → ack" row.
      * ``cancel_submit_rtt_ms`` — cancel-only RTT (kept for
        postmortem). Pre-v1.4.58 was the dashboard's "Cancel send →
        ack" row.

    All four fall through to ``None`` when their tracker isn't
    available (e.g. heartbeat fires before the executor has wired up
    its provider, or ``MARKET_DATA_TIMING_WINDOW_ENABLED=false``).
    """
    out: dict[str, Any] = {
        "exchange_to_local_receive_ms": None,
        "tx_submit_rtt_ms": None,
        "order_submit_rtt_ms": None,
        "cancel_submit_rtt_ms": None,
    }
    try:
        if ws_timing_tracker is not None:
            ws_summary = ws_timing_tracker.summary()
            stats = ws_summary.get("exchange_to_local_receive_ms")
            sample_count = int(
                ws_summary.get("samples_with_valid_exchange_to_local_receive") or 0
            )
            if stats and sample_count > 0:
                out["exchange_to_local_receive_ms"] = {
                    "min_ms": _round_or_none(stats.get("min")),
                    "median_ms": _round_or_none(stats.get("median")),
                    "p95_ms": _round_or_none(stats.get("p95")),
                    "max_ms": _round_or_none(stats.get("max")),
                    "sample_count": sample_count,
                }
    except Exception as e:
        logger.warning(
            "heartbeat_ws_latency_summary_failed err=%s", str(e)[:200]
        )
    try:
        if rtt_provider is not None:
            rtt_summary = rtt_provider() or {}
            sample_count = int(rtt_summary.get("sample_count") or 0)
            if sample_count > 0:
                out["order_submit_rtt_ms"] = {
                    "min_ms": rtt_summary.get("min_ms"),
                    "median_ms": rtt_summary.get("median_ms"),
                    "p95_ms": rtt_summary.get("p95_ms"),
                    "max_ms": rtt_summary.get("max_ms"),
                    "sample_count": sample_count,
                }
    except Exception as e:
        logger.warning(
            "heartbeat_rtt_summary_failed err=%s", str(e)[:200]
        )
    try:
        if cancel_rtt_provider is not None:
            crtt_summary = cancel_rtt_provider() or {}
            sample_count = int(crtt_summary.get("sample_count") or 0)
            if sample_count > 0:
                out["cancel_submit_rtt_ms"] = {
                    "min_ms": crtt_summary.get("min_ms"),
                    "median_ms": crtt_summary.get("median_ms"),
                    "p95_ms": crtt_summary.get("p95_ms"),
                    "max_ms": crtt_summary.get("max_ms"),
                    "sample_count": sample_count,
                }
    except Exception as e:
        logger.warning(
            "heartbeat_cancel_rtt_summary_failed err=%s", str(e)[:200]
        )
    # v1.4.58 todo-037: unified TX → ack across place + amend + cancel.
    # Single dashboard row instead of two per-op rows; ~16k samples
    # per session instead of ~1k + ~19.
    try:
        if tx_rtt_provider is not None:
            txrtt_summary = tx_rtt_provider() or {}
            sample_count = int(txrtt_summary.get("sample_count") or 0)
            if sample_count > 0:
                out["tx_submit_rtt_ms"] = {
                    "min_ms": txrtt_summary.get("min_ms"),
                    "median_ms": txrtt_summary.get("median_ms"),
                    "p95_ms": txrtt_summary.get("p95_ms"),
                    "max_ms": txrtt_summary.get("max_ms"),
                    "sample_count": sample_count,
                }
    except Exception as e:
        logger.warning(
            "heartbeat_tx_rtt_summary_failed err=%s", str(e)[:200]
        )
    return out


def _round_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None
