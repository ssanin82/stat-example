from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from app import clock as _clock


# Tags describing how ``receive_to_apply_ms`` was computed for a given sample.
# Kept as module-level constants so summary counts and serialization never drift.
RECEIVE_TO_APPLY_SOURCE_MONOTONIC = "monotonic"
RECEIVE_TO_APPLY_SOURCE_WALL_FALLBACK = "wall_fallback"

ReceiveToApplySource = Literal["monotonic", "wall_fallback"]


def _percentile_sorted(sorted_vals: list[float], p: float) -> Optional[float]:
    """Nearest-rank percentile on a pre-sorted list; returns None for empty input."""
    if not sorted_vals:
        return None
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 1:
        return float(sorted_vals[-1])
    # Linear interpolation between closest ranks.
    n = len(sorted_vals)
    pos = (n - 1) * p
    lo = int(pos)
    hi = min(n - 1, lo + 1)
    if hi == lo:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _stats(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"min": None, "median": None, "p95": None, "max": None}
    s = sorted(values)
    return {
        "min": float(s[0]),
        "median": _percentile_sorted(s, 0.5),
        "p95": _percentile_sorted(s, 0.95),
        "max": float(s[-1]),
    }


@dataclass(frozen=True, slots=True)
class PublicWsTimingSample:
    """
    Compact, in-memory-only timing sample for one applied public WS book update.

    Semantics:
    - exchange_gap_ms: delta between successive exchange timestamps (if both present).
    - local_receive_gap_ms: delta between successive local receive wall timestamps.
    - exchange_to_local_receive_ms: local wall age of book at receipt (local_wall_ms - exchange_ts_ms).
    - receive_to_apply_ms: local processing delay between receive and state apply (perf-counter deltas).
    """

    local_receive_wall_time_ms: Optional[int]
    local_receive_mono_ns: Optional[int]
    local_apply_wall_time_ms: Optional[int]
    local_apply_mono_ns: Optional[int]
    exchange_ts_ms: Optional[int]
    receive_to_apply_ms: Optional[float]
    receive_to_apply_source: Optional[ReceiveToApplySource]
    exchange_to_local_receive_ms: Optional[float]
    local_receive_gap_ms: Optional[int]
    exchange_gap_ms: Optional[int]
    seq: Optional[int]
    segment_id: int
    exchange_ts_missing: bool
    local_receive_wall_missing: bool
    local_receive_mono_missing: bool
    invalid_exchange_gap: bool
    invalid_local_receive_gap: bool
    negative_one_way_delay: bool
    negative_receive_to_apply: bool


class PublicWsTimingTracker:
    """Rolling window for public WS market-data timing (bounded; summary computed on demand)."""

    def __init__(self, *, symbol: str, max_samples: int) -> None:
        self._symbol = symbol
        self._max = int(max_samples)
        self._lock = threading.Lock()
        self._samples: deque[PublicWsTimingSample] = deque(maxlen=self._max)
        self._last_recv_wall_ms: Optional[int] = None
        self._last_exchange_ts_ms: Optional[int] = None
        self._segment_id: int = 0
        self._boundary_count: int = 0
        self._missing_exchange_ts_count = 0
        self._missing_local_receive_wall_count = 0
        self._missing_local_receive_mono_count = 0
        self._invalid_exchange_gap_count = 0
        self._invalid_local_receive_gap_count = 0
        self._negative_one_way_delay_count = 0
        self._negative_receive_to_apply_count = 0
        self._last_boundary_reason: Optional[str] = None
        self._last_boundary_wall_time_ms: Optional[int] = None

        # Interpretation guide (mirrors API docs):
        # - exchange_gap_ms: exchange-side cadence (how often the exchange says it produced updates).
        # - local_receive_gap_ms: arrival cadence (how often *we* receive updates).
        # - exchange_to_local_receive_ms: one-way “age at receipt” (network/exchange/system time effects).
        # - receive_to_apply_ms: local processing delay after receipt (our process).

    @property
    def symbol(self) -> str:
        return self._symbol

    def note_stream_reset(self, reason: str = "") -> None:
        """
        Mark a reset/reconnect boundary for gap calculations.

        After a reset, the next ingested sample will not compute exchange_gap_ms or
        local_receive_gap_ms against the previous sample (avoids reconnect holes poisoning gaps).
        """
        _ = reason  # reserved for future inclusion in structured payloads
        with self._lock:
            self._segment_id += 1
            self._boundary_count += 1
            self._last_boundary_reason = str(reason or "")
            self._last_boundary_wall_time_ms = int(_clock.time_seconds() * 1000.0)
            self._last_recv_wall_ms = None
            self._last_exchange_ts_ms = None

    def ingest(
        self,
        *,
        local_receive_wall_ms: Optional[int],
        local_receive_mono_ns: Optional[int],
        exchange_ts_ms: Optional[int],
        local_apply_wall_ms: Optional[int],
        local_apply_mono_ns: Optional[int],
        seq: Optional[int] = None,
    ) -> None:
        """Ingest a single applied update sample (lightweight; O(1))."""
        recv_gap: Optional[int] = None
        exch_gap: Optional[int] = None
        exch_to_recv: Optional[float] = None
        recv_to_apply: Optional[float] = None
        exch_missing = exchange_ts_ms is None
        recv_wall_missing = local_receive_wall_ms is None
        recv_mono_missing = local_receive_mono_ns is None
        invalid_exch_gap = False
        invalid_recv_gap = False
        negative_one_way = False
        negative_r2a = False
        r2a_source: Optional[ReceiveToApplySource] = None

        if recv_wall_missing:
            self._missing_local_receive_wall_count += 1
        else:
            if int(local_receive_wall_ms) < 0:
                # Defensive; should not happen.
                local_receive_wall_ms = 0
            if self._last_recv_wall_ms is not None:
                recv_gap = int(int(local_receive_wall_ms) - int(self._last_recv_wall_ms))
                if recv_gap < 0:
                    invalid_recv_gap = True
                    self._invalid_local_receive_gap_count += 1
                    recv_gap = None
            self._last_recv_wall_ms = int(local_receive_wall_ms)

        if exchange_ts_ms is None:
            self._missing_exchange_ts_count += 1
        else:
            if local_receive_wall_ms is not None:
                try:
                    exch_to_recv = float(int(local_receive_wall_ms) - int(exchange_ts_ms))
                    if exch_to_recv < 0:
                        negative_one_way = True
                        self._negative_one_way_delay_count += 1
                except Exception:
                    exch_to_recv = None
            if self._last_exchange_ts_ms is not None:
                exch_gap = int(int(exchange_ts_ms) - int(self._last_exchange_ts_ms))
                if exch_gap < 0:
                    invalid_exch_gap = True
                    self._invalid_exchange_gap_count += 1
                    exch_gap = None
            self._last_exchange_ts_ms = int(exchange_ts_ms)

        if recv_mono_missing:
            self._missing_local_receive_mono_count += 1

        # receive_to_apply_ms:
        # - Prefer monotonic deltas when both are present (truth-preserving, no wall-clock skew).
        # - Fall back to wall-clock only when both receive/apply wall timestamps exist.
        # - Never mix wall + monotonic, and never synthesize missing receive times.
        if local_receive_mono_ns is not None and local_apply_mono_ns is not None:
            if int(local_apply_mono_ns) >= int(local_receive_mono_ns):
                recv_to_apply = (
                    int(local_apply_mono_ns) - int(local_receive_mono_ns)
                ) / 1_000_000.0
                r2a_source = RECEIVE_TO_APPLY_SOURCE_MONOTONIC
            else:
                negative_r2a = True
                self._negative_receive_to_apply_count += 1
        elif local_receive_wall_ms is not None and local_apply_wall_ms is not None:
            delta = float(int(local_apply_wall_ms) - int(local_receive_wall_ms))
            if delta < 0:
                negative_r2a = True
                self._negative_receive_to_apply_count += 1
            else:
                recv_to_apply = delta
                r2a_source = RECEIVE_TO_APPLY_SOURCE_WALL_FALLBACK

        sample = PublicWsTimingSample(
            local_receive_wall_time_ms=(
                int(local_receive_wall_ms) if local_receive_wall_ms is not None else None
            ),
            local_receive_mono_ns=(
                int(local_receive_mono_ns) if local_receive_mono_ns is not None else None
            ),
            local_apply_wall_time_ms=(
                int(local_apply_wall_ms) if local_apply_wall_ms is not None else None
            ),
            local_apply_mono_ns=(
                int(local_apply_mono_ns) if local_apply_mono_ns is not None else None
            ),
            exchange_ts_ms=int(exchange_ts_ms) if exchange_ts_ms is not None else None,
            receive_to_apply_ms=recv_to_apply,
            receive_to_apply_source=r2a_source,
            exchange_to_local_receive_ms=exch_to_recv,
            local_receive_gap_ms=recv_gap,
            exchange_gap_ms=exch_gap,
            seq=seq,
            segment_id=self._segment_id,
            exchange_ts_missing=exch_missing,
            local_receive_wall_missing=recv_wall_missing,
            local_receive_mono_missing=recv_mono_missing,
            invalid_exchange_gap=invalid_exch_gap,
            invalid_local_receive_gap=invalid_recv_gap,
            negative_one_way_delay=negative_one_way,
            negative_receive_to_apply=negative_r2a,
        )

        with self._lock:
            self._samples.append(sample)

    def recent_samples(self, *, limit: int) -> list[PublicWsTimingSample]:
        with self._lock:
            n = max(0, min(int(limit), len(self._samples)))
            # newest-first
            out = list(self._samples)[-n:][::-1]
        return out

    def summary(self) -> dict[str, Any]:
        """Compute summary stats over the current rolling buffer (O(N log N) when called)."""
        with self._lock:
            samples = list(self._samples)
            missing = int(self._missing_exchange_ts_count)
            missing_recv_wall = int(self._missing_local_receive_wall_count)
            missing_recv_mono = int(self._missing_local_receive_mono_count)
            invalid_gap = int(self._invalid_exchange_gap_count)
            invalid_recv_gap = int(self._invalid_local_receive_gap_count)
            neg_one_way = int(self._negative_one_way_delay_count)
            neg_r2a = int(self._negative_receive_to_apply_count)
            ring_max = int(self._max)
            boundary_count = int(self._boundary_count)
            last_boundary_reason = self._last_boundary_reason
            last_boundary_wall_ms = self._last_boundary_wall_time_ms

        now_wall_ms = int(_clock.time_seconds() * 1000.0)
        newest_age_ms: Optional[int] = None
        oldest_age_ms: Optional[int] = None
        if samples:
            newest_wall = samples[-1].local_receive_wall_time_ms or samples[-1].local_apply_wall_time_ms
            oldest_wall = samples[0].local_receive_wall_time_ms or samples[0].local_apply_wall_time_ms
            if newest_wall is not None:
                newest_age_ms = max(0, now_wall_ms - int(newest_wall))
            if oldest_wall is not None:
                oldest_age_ms = max(0, now_wall_ms - int(oldest_wall))

        window_span_ms: Optional[int] = None
        if samples:
            # Prefer receive wall span when both ends have it; else apply wall span; else None.
            o_recv = samples[0].local_receive_wall_time_ms
            n_recv = samples[-1].local_receive_wall_time_ms
            if o_recv is not None and n_recv is not None:
                window_span_ms = max(0, int(n_recv) - int(o_recv))
            else:
                o_app = samples[0].local_apply_wall_time_ms
                n_app = samples[-1].local_apply_wall_time_ms
                if o_app is not None and n_app is not None:
                    window_span_ms = max(0, int(n_app) - int(o_app))

        exch_gaps = [
            float(s.exchange_gap_ms)
            for s in samples
            if s.exchange_gap_ms is not None
            and not s.invalid_exchange_gap
        ]
        recv_gaps = [
            float(s.local_receive_gap_ms)
            for s in samples
            if s.local_receive_gap_ms is not None
            and not s.invalid_local_receive_gap
        ]
        one_way = [
            float(s.exchange_to_local_receive_ms)
            for s in samples
            if s.exchange_to_local_receive_ms is not None
            and not s.negative_one_way_delay
        ]
        recv_to_apply = [
            float(s.receive_to_apply_ms)
            for s in samples
            if s.receive_to_apply_ms is not None
        ]

        total = len(samples)
        with_exch = sum(1 for s in samples if s.exchange_ts_ms is not None)
        with_recv_wall = sum(1 for s in samples if s.local_receive_wall_time_ms is not None)
        with_recv_mono = sum(1 for s in samples if s.local_receive_mono_ns is not None)
        valid_exch_gap = len(exch_gaps)
        valid_recv_gap = len(recv_gaps)
        valid_one_way = len(one_way)
        valid_r2a = len(recv_to_apply)
        valid_r2a_mono = sum(
            1
            for s in samples
            if s.receive_to_apply_ms is not None
            and s.receive_to_apply_source == RECEIVE_TO_APPLY_SOURCE_MONOTONIC
        )
        valid_r2a_wall = sum(
            1
            for s in samples
            if s.receive_to_apply_ms is not None
            and s.receive_to_apply_source == RECEIVE_TO_APPLY_SOURCE_WALL_FALLBACK
        )

        def _pct(num: int, den: int) -> Optional[float]:
            if den <= 0:
                return None
            return round((num / den) * 100.0, 3)

        # Fields are grouped logically below:
        #   window/meta -> coverage -> anomalies -> metrics.
        # Field order is preserved for API consumers; do not reorder without care.
        return {
            # --- window / meta -------------------------------------------------
            "symbol": self._symbol,
            "source_type": "public_ws",
            "total_samples": total,
            "configured_window_size": ring_max,
            "current_buffer_size": len(samples),
            "window_span_ms": window_span_ms,
            "boundary_event_count": boundary_count,
            "boundary_count_semantics": (
                "Counts stream boundary events (e.g. open/close markers), "
                "not unique reconnect episodes."
            ),
            "last_boundary_reason": last_boundary_reason,
            "last_boundary_wall_time_ms": last_boundary_wall_ms,
            "last_boundary_wall_time_iso": (
                datetime.fromtimestamp(last_boundary_wall_ms / 1000.0, tz=timezone.utc).isoformat()
                if last_boundary_wall_ms is not None
                else None
            ),
            # --- coverage ------------------------------------------------------
            "samples_with_exchange_ts": with_exch,
            "samples_with_local_receive_wall": with_recv_wall,
            "samples_with_local_receive_mono": with_recv_mono,
            "samples_with_valid_exchange_gap": valid_exch_gap,
            "samples_with_valid_local_receive_gap": valid_recv_gap,
            "samples_with_valid_exchange_to_local_receive": valid_one_way,
            "samples_with_valid_receive_to_apply": valid_r2a,
            "samples_with_valid_receive_to_apply_monotonic": valid_r2a_mono,
            "samples_with_valid_receive_to_apply_wall_fallback": valid_r2a_wall,
            "exchange_ts_coverage_pct": _pct(with_exch, total),
            "local_receive_wall_coverage_pct": _pct(with_recv_wall, total),
            "valid_one_way_delay_coverage_pct": _pct(valid_one_way, total),
            # --- anomalies -----------------------------------------------------
            "missing_exchange_ts_count": missing,
            "missing_local_receive_wall_count": missing_recv_wall,
            "missing_local_receive_mono_count": missing_recv_mono,
            "invalid_exchange_gap_count": invalid_gap,
            "invalid_local_receive_gap_count": invalid_recv_gap,
            "negative_one_way_delay_count": neg_one_way,
            "negative_receive_to_apply_count": neg_r2a,
            "oldest_sample_age_ms": oldest_age_ms,
            "newest_sample_age_ms": newest_age_ms,
            # --- metrics -------------------------------------------------------
            "exchange_gap_ms": _stats(exch_gaps),
            "local_receive_gap_ms": _stats(recv_gaps),
            "exchange_to_local_receive_ms": _stats(one_way),
            "receive_to_apply_ms": _stats(recv_to_apply),
        }

    @staticmethod
    def serialize_sample(sample: PublicWsTimingSample) -> dict[str, Any]:
        d = asdict(sample)
        lr_ms = sample.local_receive_wall_time_ms
        if lr_ms is not None:
            d["local_receive_wall_time_iso"] = datetime.fromtimestamp(
                int(lr_ms) / 1000.0, tz=timezone.utc
            ).isoformat()
        else:
            d["local_receive_wall_time_iso"] = None
        la_ms = sample.local_apply_wall_time_ms
        if la_ms is not None:
            d["local_apply_wall_time_iso"] = datetime.fromtimestamp(
                int(la_ms) / 1000.0, tz=timezone.utc
            ).isoformat()
        else:
            d["local_apply_wall_time_iso"] = None
        if sample.exchange_ts_ms is not None:
            d["exchange_ts_iso"] = datetime.fromtimestamp(
                int(sample.exchange_ts_ms) / 1000.0, tz=timezone.utc
            ).isoformat()
        else:
            d["exchange_ts_iso"] = None
        return d

