"""Periodic S3 publisher for the session's equity-history time series.

Distinct from ``live_stats.py`` (which publishes a fast-cadence point-
in-time snapshot) and ``heartbeat.py`` (slow-cadence operator-status):
this module exports the **session-wide equity samples** so the
dashboard's Bot Stats panel can render the PnL path, high-water mark,
and drawdown trough.

S3 path: ``s3://<logs_bucket>/equity_history/<profile>.json``.

Cadence: 60 s default — each publish reads the last ~6 h of equity
snapshots from storage and uploads as JSON. Payload typically 10-50 KB
depending on session length.

Trading impact: zero. Same daemon-thread pattern as the rest of the
publishers; reads from storage (which has its own lock) and writes
to S3 on a separate connection. The trading loop never waits on this.

Failure semantics: best-effort. boto3 ImportError, IAM denial, S3
network glitches all log a warning (with backoff) and the publisher
keeps trying; the bot continues trading regardless.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from app import __version__
from app.config import Settings
from app.state import BotState

from app import clock as _clock

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return _clock.now_utc().isoformat()


class EquityHistoryPublisher:
    """Daemon-thread S3 writer for session equity history.

    Reads ``storage.equity_history_since(session_started_at_utc)`` on
    each publish — gives us the full set of in-session samples in
    chronological order. The dashboard renders all of them as a
    single line chart with HWM and drawdown markers.
    """

    # 60 s default: equity samples are written by the bot at ~10 s
    # cadence, so 60 s here means each publish carries up to ~6 new
    # samples on top of the prior set. S3 PUT cost stays bounded
    # (one PUT per minute per profile) regardless of session length.
    _DEFAULT_INTERVAL_S = 60.0
    # Cap how many samples we serialise per publish. At the bot's
    # 10 s equity-snapshot cadence, 10000 samples covers ~28 h —
    # plenty for a typical operator session. For longer-running
    # sessions, ``most_recent=True`` in the storage query means we
    # ship the most recent 10000 samples (the live tail) rather
    # than the oldest 10000 (which used to freeze the dashboard
    # chart at 6h40min once the previous 2400 cap was hit).
    _MAX_SAMPLES = 10000

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        bucket: str,
        profile_name: str,
        storage: Any,
    ) -> None:
        self._settings = settings
        self._state = state
        self._bucket = bucket.strip()
        self._profile_name = (profile_name or "unknown").strip() or "unknown"
        self._key = f"equity_history/{self._profile_name}.json"
        self._interval_s = self._DEFAULT_INTERVAL_S
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_errors = 0
        self._client: Any = None
        self._storage = storage

    @classmethod
    def maybe_create(
        cls,
        settings: Settings,
        state: BotState,
        profile_name: str,
        storage: Any,
    ) -> Optional["EquityHistoryPublisher"]:
        # Reuse the same gate as live_stats: if S3 publishing is
        # disabled / unconfigured, this publisher is also off. Keeps
        # the operator's "log destination" decision in one place.
        if not bool(settings.live_stats_enabled):
            logger.info(
                "equity_history_disabled reason=LIVE_STATS_ENABLED=false"
            )
            return None
        bucket = (settings.logs_bucket or "").strip()
        if not bucket:
            logger.info(
                "equity_history_disabled reason=no_logs_bucket "
                "set_LOGS_BUCKET_to_enable"
            )
            return None
        if storage is None:
            logger.info("equity_history_disabled reason=no_storage")
            return None
        return cls(settings, state, bucket, profile_name, storage=storage)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import boto3  # type: ignore[import-untyped]

            self._client = boto3.client("s3")
        except Exception as e:
            logger.warning(
                "equity_history_disabled reason=boto3_init_failed err=%s",
                str(e)[:200],
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="equity_history", daemon=True
        )
        self._thread.start()
        logger.info(
            "equity_history_started bucket=%s key=%s interval_s=%.1f",
            self._bucket,
            self._key,
            self._interval_s,
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Publish once on startup so the dashboard's first poll has
        # data even before the first 60 s tick, then on the regular
        # cadence.
        self._safe_publish_once()
        while not self._stop.wait(self._interval_s):
            self._safe_publish_once()

    def _safe_publish_once(self) -> None:
        try:
            self._publish_once()
            self._consecutive_errors = 0
        except Exception as e:
            self._consecutive_errors += 1
            level = (
                logging.WARNING
                if self._consecutive_errors <= 3
                else logging.DEBUG
            )
            logger.log(
                level,
                "equity_history_publish_failed errs=%d err=%s",
                self._consecutive_errors,
                str(e)[:200],
            )

    def _publish_once(self) -> None:
        if self._client is None:
            return
        body = self._build_payload()
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key,
            Body=json.dumps(body, indent=None, separators=(",", ":")).encode(
                "utf-8"
            ),
            ContentType="application/json",
            CacheControl="no-store, max-age=0",
        )

    def _build_payload(self) -> dict[str, Any]:
        # Read session start under lock so we always pull the right
        # window even if the session rolls during publish.
        with self._state._lock:
            session_id = self._state.session_id
            session_started_at_utc = self._state.session_started_at_utc
            killed = bool(self._state.killed)
            kill_reason = self._state.kill_reason
            kill_ts = self._state.kill_timestamp
            symbol = self._state.symbol

        since_iso = session_started_at_utc.isoformat()
        try:
            samples = self._storage.equity_history_since(
                since_iso,
                limit=self._MAX_SAMPLES,
                most_recent=True,
            )
        except Exception:
            logger.exception("equity_history_storage_read_failed")
            samples = []

        # Compact each row: storage returns full DB columns; the
        # dashboard only needs the time + four PnL components +
        # drawdown + (1.2.13) the three sub-band feeds (mid, vol,
        # session-cumulative traded notional). Drop everything else
        # so the S3 file stays small. Per-row size grows ~24 bytes
        # for the three new numerics — at 10 000 samples = ~240 KB
        # added to the cap, still well under typical S3 / dashboard
        # bandwidth budgets.
        compact: list[dict[str, Any]] = []
        for s in samples:
            compact.append(
                {
                    "ts": s.get("ts"),
                    "equity_usd": s.get("equity_usd"),
                    "realized_pnl_usd": s.get("realized_pnl_usd"),
                    "unrealized_pnl_usd": s.get("unrealized_pnl_usd"),
                    "fees_usd": s.get("fees_usd"),
                    "drawdown_usd": s.get("drawdown_usd"),
                    # 1.2.13 sub-band feeds. ``None`` on samples
                    # written by pre-1.2.9 bot builds (column was
                    # NULL on disk); the dashboard handles missing
                    # values per-band.
                    "mid_price": s.get("mid_price"),
                    "vol_bps": s.get("vol_bps"),
                    "session_traded_notional_usd": s.get(
                        "session_traded_notional_usd"
                    ),
                    # 1.2.15 inventory sub-band feed. Signed
                    # position quantity; NULL on pre-1.2.15 rows.
                    "position_qty": s.get("position_qty"),
                    # 1.3.31 (todo-030) basis sub-band feed. The
                    # cross-venue basis EWMA captured at equity-
                    # snapshot time. NULL on pre-1.3.31 rows and
                    # whenever the bot was running with Binance
                    # disabled / pre-warmup. Dashboard's Session-PnL
                    # chart renders this as an additional sub-band
                    # below the inventory trace.
                    "binance_basis_ewma": s.get("binance_basis_ewma"),
                    # Phase 8A Option B (v1.5.189) — AS attribution
                    # time-series. NULL on pre-v1.5.189 rows and on
                    # rows written with AS disabled. Dashboard /
                    # postmortem read these to chart AS-derived
                    # base half-spread + k-intensity alongside vol.
                    "base_half_spread_bps": s.get("base_half_spread_bps"),
                    "as_k_intensity_per_min": s.get("as_k_intensity_per_min"),
                }
            )

        # 1.2.22 (v1.4.162 rewrite): aggregate soft-flatten events for
        # dashboard markers. Source-of-truth is the
        # ``soft_flatten_events`` table itself — every SF entry/exit
        # writes a row there, INDEPENDENT of whether the closing fills
        # carried the ``soft_flatten_event_id`` stamp.
        #
        # Pre-v1.4.162 we aggregated by grouping ``fills`` rows on
        # ``soft_flatten_event_id``. That worked for episodes closed
        # via the passive post-only path (the ``WorkingOrder``
        # constructor in ``execution.py`` stamps SF orders, fill
        # ingestion propagates the tag), but FAILED for episodes
        # closed via the taker-fallback path in ``bot.py`` (around
        # line 2369-2374): ``self._client.market_close(...)`` bypasses
        # ``ExecutionEngine.place_order`` entirely, so no row is
        # inserted into the local ``orders`` SQLite table, so
        # ``order_metadata_for_fill_ingest`` returns the empty dict at
        # ingest time and the taker fills end up with
        # ``soft_flatten_event_id=None``.
        #
        # See snapshot ``v1.4.157-260520-213540-prod.okx.ton.usdt.perp``
        # for the canonical reproduction: SF#11167 closed via the
        # taker fallback, 3 fills at 17:15:00.757 — none of them
        # tagged — and the dashboard renders NO SF marker for the
        # episode that drove the entire $0.20 equity drop.
        #
        # Fix: query the events table directly. Pull
        # ``attributed_fills_count`` + ``attributed_fills_notional_usd``
        # via the LEFT JOIN the storage helper already does, so when
        # fills ARE correctly tagged the marker tooltip carries the
        # same info as before. When fills aren't tagged (taker fallback)
        # the counts are zero but the marker still renders, anchored
        # to ``ts_start`` so the operator sees WHERE the flatten
        # happened on the timeline.
        sf_events: list[dict[str, Any]] = []
        try:
            sf_rows = self._storage.recent_soft_flatten_events(
                limit=200,
                since_ts=since_iso,
            )
            for row in sf_rows:
                event_id_raw = row.get("id")
                if event_id_raw is None:
                    continue
                ts_start = row.get("ts_start")
                ts_end = row.get("ts_end") or ts_start
                fill_count = int(row.get("attributed_fills_count") or 0)
                notional = float(
                    row.get("attributed_fills_notional_usd") or 0.0
                )
                # Side inference: the BOT flattens by trading OPPOSITE
                # to its existing inventory. ``entry_position_qty`` is
                # the position at SF entry; positive → bot was long →
                # closes via SELL; negative → bot was short → closes
                # via BUY. Falls back to None if the column wasn't
                # captured.
                entry_qty = row.get("entry_position_qty")
                side: Optional[str] = None
                if entry_qty is not None:
                    try:
                        q = float(entry_qty)
                        if q > 0:
                            side = "SELL"
                        elif q < 0:
                            side = "BUY"
                    except (TypeError, ValueError):
                        side = None
                # v1.4.173 (Phase 4D.4): per-phase fill breakdown +
                # notional-weighted average taker spread paid. Computed
                # in ``_exit_soft_flatten`` and persisted on the row;
                # NULL on episodes that ran before this migration or
                # while the ladder was disabled (`fills_by_phase` null
                # → dashboard renders the legacy SF tooltip).
                fills_by_phase: Optional[dict[str, int]] = None
                phase_json = row.get("fills_by_phase_json")
                if phase_json:
                    try:
                        parsed = json.loads(str(phase_json))
                        if isinstance(parsed, dict):
                            # Coerce values to int; tolerate non-int
                            # entries by dropping them.
                            tmp: dict[str, int] = {}
                            for k, v in parsed.items():
                                try:
                                    tmp[str(k)] = int(v)
                                except (TypeError, ValueError):
                                    continue
                            fills_by_phase = tmp
                    except (TypeError, ValueError, json.JSONDecodeError):
                        fills_by_phase = None
                taker_bps_raw = row.get("taker_spread_bps_paid")
                taker_spread_bps_paid: Optional[float] = None
                if taker_bps_raw is not None:
                    try:
                        taker_spread_bps_paid = round(float(taker_bps_raw), 4)
                    except (TypeError, ValueError):
                        taker_spread_bps_paid = None
                sf_events.append(
                    {
                        "event_id": int(event_id_raw),
                        "ts_first_fill": ts_start,
                        "ts_last_fill": ts_end,
                        "fill_count": fill_count,
                        "side": side,
                        "notional_usd": round(notional, 4),
                        "fills_by_phase": fills_by_phase,
                        "taker_spread_bps_paid": taker_spread_bps_paid,
                    }
                )
            sf_events = sorted(
                sf_events, key=lambda x: x["ts_first_fill"] or ""
            )
        except Exception:
            logger.exception("equity_history_sf_events_aggregate_failed")
            sf_events = []

        # v1.5.33 — take-profit episode markers. Same shape as
        # ``sf_events`` so the frontend can render them with a
        # parallel ``TpMarkers`` component. Anchored on ``ts_start``
        # (TP arming time) — unlike SF, the TP marker is meaningful
        # even when the post-only never filled (operator wants to
        # see "we tried to take profit here and price escaped").
        tp_events: list[dict[str, Any]] = []
        try:
            tp_rows = self._storage.recent_tp_events(
                limit=200,
                since_ts=since_iso,
            )
            for row in tp_rows:
                event_id_raw = row.get("id")
                if event_id_raw is None:
                    continue
                fill_count = int(row.get("attributed_fills_count") or 0)
                trigger_upnl = row.get("trigger_upnl_bps")
                exit_reason = row.get("exit_reason")
                exit_upnl = row.get("exit_upnl_bps")
                side = row.get("close_side")
                tp_events.append(
                    {
                        "event_id": int(event_id_raw),
                        "ts_anchor": row.get("ts_start"),
                        "fill_count": fill_count,
                        "trigger_upnl_bps": (
                            None
                            if trigger_upnl is None
                            else round(float(trigger_upnl), 3)
                        ),
                        "exit_reason": exit_reason,
                        "exit_upnl_bps": (
                            None
                            if exit_upnl is None
                            else round(float(exit_upnl), 3)
                        ),
                        "side": side,
                    }
                )
            tp_events = sorted(
                tp_events, key=lambda x: x["ts_anchor"] or ""
            )
        except Exception:
            logger.exception("equity_history_tp_events_aggregate_failed")
            tp_events = []

        return {
            "schema_version": 1,
            "profile": self._profile_name,
            "symbol": symbol,
            "version": __version__,
            "captured_at_utc": _now_iso(),
            "session_id": session_id,
            "session_started_at_utc": since_iso,
            "killed": killed,
            "kill_reason": kill_reason,
            "kill_timestamp_utc": (
                kill_ts.isoformat() if isinstance(kill_ts, datetime) else kill_ts
            ),
            "max_drawdown_usd_cap": float(self._settings.max_drawdown_usd),
            "samples": compact,
            "sf_events": sf_events,
            # v1.5.33 — take-profit episode markers.
            "tp_events": tp_events,
        }
