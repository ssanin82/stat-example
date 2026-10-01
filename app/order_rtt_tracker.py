"""Rolling order place-to-ack RTT tracker.

Plan reference: `plans/20260420-binance-move/plan.md` latency-toolkit
addition.

The bot already records the per-tick max RTT in
``state.latency_order_submit_rtt_ms`` (last value only, reset each
tick). This module adds a rolling window of samples so the operator
can see distributional stats (min / median / p95 / p99 / max) and
reason about regime-vs-tier latency rather than just last-tick.

Sample source: ``execution.py`` measures RTT via
``time.perf_counter()`` deltas between ``transport_send`` and
``transport_done``. ``perf_counter`` has nanosecond resolution on
typical platforms so the float-millisecond storage already preserves
microsecond precision (e.g. 2.347 ms = 2347 μs).

Bounded memory: deque(maxlen=N), default 1024 samples.

Thread-safety: a single lock protects ingest + summary. Both are O(N)
in the worst case (sorting for percentiles); at N=1024 this is < 50 μs
per call so it's fine on a 0.5 Hz quote loop.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional


def _percentile_sorted(s: list[float], p: float) -> Optional[float]:
    if not s:
        return None
    if p <= 0:
        return float(s[0])
    if p >= 1:
        return float(s[-1])
    n = len(s)
    pos = (n - 1) * p
    lo = int(pos)
    hi = min(n - 1, lo + 1)
    if hi == lo:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


@dataclass(frozen=True, slots=True)
class OrderRttSample:
    """One place-to-ack RTT sample. ``rtt_ms`` carries microsecond
    precision (perf_counter is ns on most platforms; we just store as
    float ms for ergonomics).
    """

    rtt_ms: float
    when_mono_ns: int
    op: str  # "place" / "cancel" — kept generic so future cancel-RTT can fold in
    outcome: str  # "accepted" / "rejected" / "transport_error"


class OrderRttTracker:
    """Rolling window of place-to-ack RTT samples.

    Used by ``execution.py`` (ingest side) and the ``/latency``
    Telegram command + ``/health`` API (read side).
    """

    def __init__(self, *, max_samples: int = 1024) -> None:
        self._lock = threading.Lock()
        self._samples: deque[OrderRttSample] = deque(maxlen=max_samples)
        self._total_count: int = 0  # monotonic; survives the deque rotation

    def ingest(
        self,
        *,
        rtt_ms: float,
        op: str = "place",
        outcome: str = "accepted",
    ) -> None:
        """Record one RTT sample. ``rtt_ms`` should be the
        ``perf_counter`` delta in milliseconds; preserve microsecond
        precision (don't pre-round).
        """
        try:
            v = float(rtt_ms)
        except (TypeError, ValueError):
            return
        if v < 0 or v > 60_000.0:
            # Defensive: a 60-second RTT is a transport hang; capture
            # but flag in the outcome so summary stats can exclude.
            outcome = "transport_error"
            v = max(0.0, min(v, 60_000.0))
        sample = OrderRttSample(
            rtt_ms=v,
            when_mono_ns=time.monotonic_ns(),
            op=str(op or "place"),
            outcome=str(outcome or "accepted"),
        )
        with self._lock:
            self._samples.append(sample)
            self._total_count += 1

    def summary(
        self,
        *,
        outcome_filter: Optional[str] = "accepted",
        op_filter: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Compute distributional stats over the rolling window.

        ``outcome_filter='accepted'`` (default) excludes transport
        errors from the percentile calculation — those are not
        representative of the venue's normal RTT. Pass ``None`` to
        include everything.

        ``op_filter`` (1.4.0 cancel-prio Phase 0.5; v1.4.27 extension):
        filter to one OR MORE op kinds. Accepts:
          * ``None``                — include all ops (default)
          * ``"place"``             — single op (back-compat)
          * ``("place", "amend")``  — tuple/list/set for combined view

        v1.4.27 use case: ``OrderManager.order_rtt_summary()`` now
        passes ``("place", "amend")`` so the dashboard's "Order send
        → ack" panel shows the combined place + amend distribution
        (previously only place; amend RTT was tracked but unsurfaced).
        """
        if isinstance(op_filter, str):
            op_filter_set: Optional[frozenset[str]] = frozenset({op_filter})
        elif op_filter is None:
            op_filter_set = None
        else:
            op_filter_set = frozenset(op_filter)
        with self._lock:
            samples = [
                s for s in self._samples
                if (outcome_filter is None or s.outcome == outcome_filter)
                and (op_filter_set is None or s.op in op_filter_set)
            ]
            window_size = self._samples.maxlen
            total_count = self._total_count
        if not samples:
            return {
                "sample_count": 0,
                "window_size": window_size,
                "total_observed_count": total_count,
                "min_ms": None,
                "median_ms": None,
                "p95_ms": None,
                "p99_ms": None,
                "max_ms": None,
                "mean_ms": None,
            }
        vals = sorted(s.rtt_ms for s in samples)
        return {
            "sample_count": len(samples),
            "window_size": window_size,
            "total_observed_count": total_count,
            "min_ms": round(float(vals[0]), 3),
            "median_ms": round(_percentile_sorted(vals, 0.5) or 0.0, 3),
            "p95_ms": round(_percentile_sorted(vals, 0.95) or 0.0, 3),
            "p99_ms": round(_percentile_sorted(vals, 0.99) or 0.0, 3),
            "max_ms": round(float(vals[-1]), 3),
            "mean_ms": round(sum(vals) / len(vals), 3),
        }
