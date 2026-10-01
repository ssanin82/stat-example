"""
In-memory market-data update gap metrics (update-to-update, not REST latency).

Median and p95 are computed on demand by sorting a bounded ring buffer of recent gap
samples (see Settings.market_data_gap_ring_buffer_samples). When the number of gaps
in the run exceeds the ring size, quantiles describe the recent window only — not
the full history.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

from app.config import Settings
from app.storage import Storage
from app.utils.logfmt import log_extra

from app import clock as _clock

logger = logging.getLogger(__name__)


def market_snapshot_eligible_for_gap_stats(market: Any) -> bool:
    if market is None:
        return False
    if market.mid_price is not None and market.mid_price > 0:
        return True
    return bool(
        market.best_bid
        and market.best_ask
        and market.best_bid > 0
        and market.best_ask > 0
    )


def _percentile_sorted(sorted_vals: list[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


class MarketDataGapTracker:
    """Thread-safe gap stats for consecutive successful market snapshot applies."""

    def __init__(self, ring_maxlen: int) -> None:
        self._lock = threading.Lock()
        self._ring_max = max(64, int(ring_maxlen))
        self._updates = 0
        self._gaps = 0
        self._sum_gap_ms = 0.0
        self._min_gap_ms: Optional[float] = None
        self._max_gap_ms: Optional[float] = None
        self._last_gap_ms: Optional[float] = None
        self._first_ts: Optional[datetime] = None
        self._last_ts: Optional[datetime] = None
        self._last_source: Optional[str] = None
        self._last_perf: Optional[float] = None
        self._gap_ring: deque[float] = deque(maxlen=self._ring_max)
        self._last_large_gap_log_perf: float = 0.0

    def note_successful_update(
        self,
        *,
        perf_now: float,
        wall_now: datetime,
        source: str,
        symbol: str,
        recovery_state: str,
        settings: Settings,
        storage: Optional[Storage],
        session_id: str,
    ) -> None:
        gap_ms: Optional[float] = None
        with self._lock:
            self._updates += 1
            self._last_source = source
            if self._first_ts is None:
                self._first_ts = wall_now
            self._last_ts = wall_now
            if self._last_perf is not None:
                gap_ms = max(0.0, (perf_now - self._last_perf) * 1000.0)
                self._gaps += 1
                self._sum_gap_ms += gap_ms
                self._min_gap_ms = (
                    gap_ms if self._min_gap_ms is None else min(self._min_gap_ms, gap_ms)
                )
                self._max_gap_ms = (
                    gap_ms if self._max_gap_ms is None else max(self._max_gap_ms, gap_ms)
                )
                self._last_gap_ms = gap_ms
                self._gap_ring.append(gap_ms)
            self._last_perf = perf_now

        if gap_ms is None:
            return

        thr = float(settings.market_data_gap_large_log_threshold_ms)
        if thr > 0.0 and gap_ms >= thr:
            interval = float(settings.market_data_gap_large_log_interval_seconds)
            now_m = _clock.monotonic()
            with self._lock:
                if interval > 0.0 and (now_m - self._last_large_gap_log_perf) < interval:
                    return
                self._last_large_gap_log_perf = now_m
            pl = {
                "event": "market_data_large_gap_detected",
                "gap_ms": round(gap_ms, 3),
                "symbol": symbol,
                "update_ts": wall_now.isoformat(),
                "market_data_recovery_state": recovery_state,
                "source_type": source,
            }
            log_extra(logger, logging.WARNING, "market_data_large_gap_detected", pl)

        if settings.market_data_gap_persist_samples and storage is not None:
            storage.insert_market_data_gap_sample(
                session_id=session_id,
                ts_utc=wall_now.isoformat(),
                gap_ms=gap_ms,
                symbol=symbol,
                source=source,
                max_rows_per_session=int(settings.market_data_gap_persist_max_rows),
            )

    def to_api_dict(self, *, session_id: str, symbol: str) -> dict[str, Any]:
        with self._lock:
            n_up = self._updates
            n_gap = self._gaps
            first_ts = self._first_ts.isoformat() if self._first_ts else None
            last_ts = self._last_ts.isoformat() if self._last_ts else None
            src = self._last_source
            last_gap = self._last_gap_ms
            min_g = self._min_gap_ms
            max_g = self._max_gap_ms
            mean_g = (self._sum_gap_ms / n_gap) if n_gap > 0 else None
            ring = list(self._gap_ring)

        quantile_note = (
            f"median and p95_gap_ms are from the last up to {self._ring_max} gap samples "
            "in memory; older gaps are omitted from quantiles when the run exceeds the ring size."
        )

        if n_up < 2:
            return {
                "session_id": session_id,
                "symbol": symbol,
                "source_type": src,
                "update_count": n_up,
                "gap_count": 0,
                "min_gap_ms": None,
                "max_gap_ms": None,
                "mean_gap_ms": None,
                "median_gap_ms": None,
                "p95_gap_ms": None,
                "last_gap_ms": None,
                "first_update_ts": first_ts,
                "last_update_ts": last_ts,
                "quantiles_note": quantile_note,
            }

        sorted_ring = sorted(ring)
        med = _percentile_sorted(sorted_ring, 0.5)
        p95 = _percentile_sorted(sorted_ring, 0.95)

        return {
            "session_id": session_id,
            "symbol": symbol,
            "source_type": src,
            "update_count": n_up,
            "gap_count": n_gap,
            "min_gap_ms": None if min_g is None else round(float(min_g), 3),
            "max_gap_ms": None if max_g is None else round(float(max_g), 3),
            "mean_gap_ms": None if mean_g is None else round(float(mean_g), 3),
            "median_gap_ms": None if med is None else round(float(med), 3),
            "p95_gap_ms": None if p95 is None else round(float(p95), 3),
            "last_gap_ms": None if last_gap is None else round(float(last_gap), 3),
            "first_update_ts": first_ts,
            "last_update_ts": last_ts,
            "quantiles_note": quantile_note,
        }

    def recent_gap_median_p95_ms(self) -> tuple[Optional[float], Optional[float]]:
        """Median and p95 of recent inter-update gaps (ms); None if not enough samples."""
        with self._lock:
            ring = list(self._gap_ring)
        if not ring:
            return None, None
        sorted_ring = sorted(ring)
        med = _percentile_sorted(sorted_ring, 0.5)
        p95 = _percentile_sorted(sorted_ring, 0.95)
        return (
            None if med is None else round(float(med), 3),
            None if p95 is None else round(float(p95), 3),
        )

    def recent_gap_stats_for_gate(
        self,
    ) -> tuple[Optional[float], Optional[float], Optional[float], int]:
        """One-shot snapshot of (median, p95, last, count) under a single lock.

        Used by :func:`compute_quote_eligibility` to:

        - feed p95/median into the freshness hold/one-sided gates, and
        - use ``last_gap_ms`` as a "feed is alive NOW" override signal,
        - gate p95-hold on sample count (skip when the ring hasn't warmed up).

        Single-lock atomicity matters: without it, the percentile computation
        and the ``last_gap_ms`` read could disagree during a WS-write burst,
        producing inconsistent gate decisions.
        """
        with self._lock:
            ring = list(self._gap_ring)
            last_gap = self._last_gap_ms
        count = len(ring)
        if not ring:
            return None, None, (None if last_gap is None else round(float(last_gap), 3)), 0
        sorted_ring = sorted(ring)
        med = _percentile_sorted(sorted_ring, 0.5)
        p95 = _percentile_sorted(sorted_ring, 0.95)
        return (
            None if med is None else round(float(med), 3),
            None if p95 is None else round(float(p95), 3),
            None if last_gap is None else round(float(last_gap), 3),
            count,
        )
