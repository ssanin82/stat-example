"""Periodic S3 publisher for the dashboard's session-scoped tables.

Companion to ``live_stats.py`` (point-in-time snapshot at ~5s) and
``equity_history_publisher.py`` (equity samples at ~60s). This module
publishes the **session-scoped tables** that the dashboard's
Inventory / Gates / Execution-quality / Regimes panels read from:

- ``dashboard/exposure_since_<profile>.json``
  Source: ``storage.exposure_bars_since`` (5s cadence rows from the
  exposure-bar emitter). Powers the inventory/gate aggregations.

- ``dashboard/fills_since_<profile>.json``
  Source: ``storage.fills_since`` (all Phase 1-4c enrichment fields
  per fill). Powers the inventory-conditional markout panels and the
  execution-quality leakage cards.

- ``dashboard/orders_lifecycle_since_<profile>.json``
  Source: ``storage.orders_lifecycle_since`` (decision-state stamps
  + derived ``placement_to_ack_ms`` / ``cancel_to_close_ms`` /
  ``lifetime_ms``). Powers the execution-quality latency histograms
  and cancel-race diagnostics.

Cadence: 30s default. Trading impact: zero (daemon thread, storage
reads under storage's own lock, S3 writes on a separate connection).
Failure semantics: best-effort (boto3 ImportError / IAM denial / S3
network glitch → warning with backoff; the bot keeps trading
regardless).

Spec sources:

- ``issues/todo-028-frontend-inventory-behavior.md`` § Prerequisites
- ``issues/todo-032-frontend-execution-card.md`` § Backend work
  (the orders_lifecycle extension)
- ``plans/20260514-execution.md`` § Tomorrow morning: the publisher
  deploy
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


class DashboardStatePublisher:
    """Daemon-thread S3 writer for the session-scoped dashboard tables.

    One thread, one cadence, three S3 PUTs per tick (one per table).
    Reading happens under storage's own lock; serialisation + S3 PUT
    happen outside the lock so trading-hot-path latency is unaffected.

    Disable via ``OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED=false`` if
    any unforeseen impact appears. The other publishers
    (live_stats, equity_history, heartbeat) remain unaffected.
    """

    _S3_KEY_EXPOSURE = "dashboard/exposure_since_{profile}.json"
    _S3_KEY_FILLS = "dashboard/fills_since_{profile}.json"
    _S3_KEY_ORDERS_LIFECYCLE = "dashboard/orders_lifecycle_since_{profile}.json"

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        bucket: str,
        profile_name: str,
        storage: Any,
        *,
        interval_s: float,
        exposure_limit: int,
        fills_limit: int,
        orders_lifecycle_limit: int,
    ) -> None:
        self._settings = settings
        self._state = state
        self._bucket = bucket.strip()
        self._profile_name = (profile_name or "unknown").strip() or "unknown"
        self._interval_s = float(interval_s)
        self._exposure_limit = int(exposure_limit)
        self._fills_limit = int(fills_limit)
        self._orders_lifecycle_limit = int(orders_lifecycle_limit)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Per-table consecutive-error counters so one failing PUT
        # doesn't suppress logs for the other two healthy PUTs.
        self._consecutive_errors: dict[str, int] = {
            "exposure": 0,
            "fills": 0,
            "orders_lifecycle": 0,
        }
        self._client: Any = None
        self._storage = storage

    @classmethod
    def maybe_create(
        cls,
        settings: Settings,
        state: BotState,
        profile_name: str,
        storage: Any,
    ) -> Optional["DashboardStatePublisher"]:
        """Construct an instance if the publisher is enabled and has
        the required dependencies; return ``None`` otherwise.

        Disable conditions:
        - ``OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED=false`` (explicit
          operator opt-out)
        - ``LIVE_STATS_ENABLED=false`` (treats this publisher as a
          sibling of live_stats — same global S3 kill-switch)
        - ``LOGS_BUCKET`` empty (no S3 destination configured)
        - ``storage`` is None (no SQLite handle available)

        Each disable condition logs a one-line ``info`` so the
        operator can tell from the log why the publisher isn't
        running. Same pattern as the other publishers.
        """
        if not bool(settings.observability_dashboard_publish_enabled):
            logger.info(
                "dashboard_state_disabled reason=OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED=false"
            )
            return None
        if not bool(settings.live_stats_enabled):
            logger.info(
                "dashboard_state_disabled reason=LIVE_STATS_ENABLED=false"
            )
            return None
        bucket = (settings.logs_bucket or "").strip()
        if not bucket:
            logger.info(
                "dashboard_state_disabled reason=no_logs_bucket "
                "set_LOGS_BUCKET_to_enable"
            )
            return None
        if storage is None:
            logger.info("dashboard_state_disabled reason=no_storage")
            return None
        return cls(
            settings,
            state,
            bucket,
            profile_name,
            storage=storage,
            interval_s=float(
                settings.observability_dashboard_publish_interval_seconds
            ),
            exposure_limit=int(
                settings.observability_dashboard_exposure_bars_limit
            ),
            fills_limit=int(settings.observability_dashboard_fills_limit),
            orders_lifecycle_limit=int(
                settings.observability_dashboard_orders_lifecycle_limit
            ),
        )

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import boto3  # type: ignore[import-untyped]

            self._client = boto3.client("s3")
        except Exception as e:
            logger.warning(
                "dashboard_state_disabled reason=boto3_init_failed err=%s",
                str(e)[:200],
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="dashboard_state", daemon=True
        )
        self._thread.start()
        logger.info(
            "dashboard_state_started bucket=%s profile=%s interval_s=%.1f "
            "exposure_limit=%d fills_limit=%d orders_lifecycle_limit=%d",
            self._bucket,
            self._profile_name,
            self._interval_s,
            self._exposure_limit,
            self._fills_limit,
            self._orders_lifecycle_limit,
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Publish once on startup so the dashboard's first poll has
        # data even before the first 30s tick. Then on the regular
        # cadence.
        self._safe_publish_once()
        while not self._stop.wait(self._interval_s):
            self._safe_publish_once()

    def _safe_publish_once(self) -> None:
        """Publish all three tables, isolating failures per-table.

        We deliberately do NOT short-circuit on the first failure —
        a transient S3 error for one PUT shouldn't suppress the
        other two healthy PUTs. Each table is wrapped in its own
        try/except with its own consecutive-error counter for log
        rate-limiting.
        """
        # Snapshot session metadata once so all three publishes see
        # the same window even if the session rolls during this tick.
        with self._state._lock:
            session_id = self._state.session_id
            session_started_at_utc = self._state.session_started_at_utc
            symbol = self._state.symbol
        since_iso = session_started_at_utc.isoformat()
        common_meta = {
            "schema_version": 1,
            "profile": self._profile_name,
            "symbol": symbol,
            "version": __version__,
            "captured_at_utc": _now_iso(),
            "session_id": session_id,
            "session_started_at_utc": since_iso,
        }

        self._safe_publish_table(
            "exposure",
            self._S3_KEY_EXPOSURE,
            self._build_exposure_payload,
            common_meta,
            since_iso,
        )
        self._safe_publish_table(
            "fills",
            self._S3_KEY_FILLS,
            self._build_fills_payload,
            common_meta,
            since_iso,
        )
        self._safe_publish_table(
            "orders_lifecycle",
            self._S3_KEY_ORDERS_LIFECYCLE,
            self._build_orders_lifecycle_payload,
            common_meta,
            since_iso,
        )

    def _safe_publish_table(
        self,
        table_name: str,
        key_template: str,
        builder: Any,
        common_meta: dict[str, Any],
        since_iso: str,
    ) -> None:
        try:
            payload = builder(since_iso, common_meta)
            self._put_object(
                key_template.format(profile=self._profile_name), payload
            )
            self._consecutive_errors[table_name] = 0
        except Exception as e:
            self._consecutive_errors[table_name] += 1
            level = (
                logging.WARNING
                if self._consecutive_errors[table_name] <= 3
                else logging.DEBUG
            )
            logger.log(
                level,
                "dashboard_state_publish_failed table=%s errs=%d err=%s",
                table_name,
                self._consecutive_errors[table_name],
                str(e)[:200],
            )

    def _put_object(self, key: str, body: dict[str, Any]) -> None:
        if self._client is None:
            return
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=json.dumps(
                body, indent=None, separators=(",", ":"), default=str
            ).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-store, max-age=0",
        )

    # ------------------------------------------------------------------
    # Per-table builders. Each returns a dict ready to JSON-serialise.
    # Builders read from storage (which uses its own lock) — no need
    # to hold BotState's lock during the read.
    # ------------------------------------------------------------------

    def _build_exposure_payload(
        self, since_iso: str, common_meta: dict[str, Any]
    ) -> dict[str, Any]:
        # 1.3.99: ``most_recent=True`` so long sessions clip at the
        # OLDEST rows beyond the cap, not the newest. The dashboard's
        # Gate-activity strip + Eligibility-state band track the live
        # tail; without this flag they freeze at the first ~2h45m of
        # a session (at the 5s exposure-bar cadence × default 2000-row
        # cap). Operator screenshot 2026-05-16 showed both bands
        # empty across most of the timeline despite an active session.
        rows = self._storage.exposure_bars_since(
            since_iso, limit=self._exposure_limit, most_recent=True
        )
        return {
            **common_meta,
            "table": "exposure_bars",
            "row_limit": self._exposure_limit,
            "row_count": len(rows),
            "rows": rows,
        }

    def _build_fills_payload(
        self, since_iso: str, common_meta: dict[str, Any]
    ) -> dict[str, Any]:
        # 1.3.99: ``most_recent=True`` — same rationale as the exposure
        # publisher above. The Regime / Fill-Drilldown / Bot Stats
        # tabs all read this artifact; long sessions need the live
        # tail, not the oldest fills.
        rows = self._storage.fills_since(
            since_iso, limit=self._fills_limit, most_recent=True
        )
        return {
            **common_meta,
            "table": "fills",
            "row_limit": self._fills_limit,
            "row_count": len(rows),
            "rows": rows,
        }

    def _build_orders_lifecycle_payload(
        self, since_iso: str, common_meta: dict[str, Any]
    ) -> dict[str, Any]:
        # 1.3.99: ``most_recent=True`` — the Places/min and Cancels/min
        # sub-bands (introduced v1.3.98) bucket this artifact; long
        # sessions beyond the 5000-row default would otherwise show
        # the bars glued to the left edge of the chart.
        rows = self._storage.orders_lifecycle_since(
            since_iso,
            limit=self._orders_lifecycle_limit,
            most_recent=True,
        )
        # 1.4.6: STICKY rejection summary — survives the
        # ``most_recent`` row-buffer ring eviction. The frontend's
        # "Place rejects by venue reason" + the total-reject count
        # MUST be sourced from this rather than from ``rows`` so
        # operator never loses visibility on exchange errors after
        # the buffer rolls. Cardinality is small (~10-50 distinct
        # detail strings even on long sessions) so the inline JSON
        # payload doesn't bloat the S3 artifact.
        try:
            reject_summary = self._state.reject_summary_snapshot()
        except Exception:  # pragma: no cover - defensive
            reject_summary = {
                "place": {},
                "cancel": {},
                "place_total": 0,
                "cancel_total": 0,
            }
        # 1.4.6: per-outcome cumulative aggregates (count + latency
        # moments). Source of truth for total-placements / ack-rate
        # in the dashboard — replaces the row-count-based derivation
        # which silently capped at 5000.
        try:
            outcome_aggregates = self._state.outcome_aggregates_snapshot()
        except Exception:  # pragma: no cover - defensive
            outcome_aggregates = {"place": {}, "cancel": {}}
        return {
            **common_meta,
            "table": "orders_lifecycle",
            "row_limit": self._orders_lifecycle_limit,
            "row_count": len(rows),
            "rows": rows,
            "sticky_reject_summary": reject_summary,
            "outcome_aggregates": outcome_aggregates,
        }
