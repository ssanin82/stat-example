"""Quote-age fill bucketing (todo-006).

For each fill, computes the **quote age at fill** — i.e. how long the
order rested on the venue between ACK and being hit — and aggregates
markout-per-bucket so the operator can see at a glance whether the
bot's edge lives in fast (toxic) or slow (passive) fills.

Buckets (matched to BUGS/todo-006.md):

  < 100 ms    — instant fills; almost always toxic / picked-off
  100-500 ms  — still fast; toxicity tail
  500-2000 ms — borderline; mixed signal
  2-10 s      — passive; where market-maker edge lives
  10 s +      — very slow; rare, but the cleanest income

Two integration points:

  1. ``note_ack(order_id_exchange, ts_ack)`` — called from
     ``app/execution.py`` immediately after a place-response transitions
     a WorkingOrder to ACKED. Records the venue ack timestamp in a
     small bounded LRU so the fill side can compute the age.

  2. ``note_fill(fill)`` — called from ``app/fill_ingestion.py`` after
     ``state.record_fill`` succeeds. Looks up the matching ack ts,
     computes ``quote_age_ms = ts_fill - ts_ack``, buckets, and
     accumulates the markout into the bucket's running aggregate.

The aggregator publishes a snapshot dict via ``to_dict()`` consumed by
``live_stats`` (5s cadence). Frontend Bot Stats reads from there.

Bounded memory: ack cache 500 entries (~5-10 minutes of order activity
at HYPE rates), fill window 100 entries. Worst case ~20 KB. Lock-held
work is O(1) for both note_* methods; ``to_dict`` is O(N) over the
fill window.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.models import Fill


# Bucket boundaries in milliseconds. Lower bound inclusive, upper bound
# exclusive. Last bucket is open-ended (10s+).
_BUCKET_BOUNDS_MS: tuple[tuple[float, float], ...] = (
    (0.0, 100.0),
    (100.0, 500.0),
    (500.0, 2000.0),
    (2000.0, 10_000.0),
    (10_000.0, float("inf")),
)
_BUCKET_LABELS: tuple[str, ...] = (
    "<100ms",
    "100-500ms",
    "500ms-2s",
    "2-10s",
    "10s+",
)


def _bucket_index_for(age_ms: float) -> int:
    for i, (lo, hi) in enumerate(_BUCKET_BOUNDS_MS):
        if lo <= age_ms < hi:
            return i
    return len(_BUCKET_BOUNDS_MS) - 1


@dataclass
class _BucketedFill:
    """Per-fill derived data we keep in the rolling window.

    Holds a *reference* to the Fill object (not a copy) so that the
    delayed markout values populated by app/markout.py (at +1s/+3s/+5s
    after the fill) are picked up automatically when ``to_dict()`` is
    called. The Fill stays alive in ``state.recent_fills`` until that
    deque rolls it out, which is many minutes longer than our 100-fill
    window — so the reference is safe to hold.
    """

    age_ms: float
    bucket_index: int
    side: str  # "BUY" / "SELL"
    notional_usd: float
    fill_ref: "Fill"  # reference; markout_5s_bps read lazily
    # todo-006: place-time aggressiveness category copied from the
    # parent ``WorkingOrder``; one of ``at_touch`` / ``inside`` /
    # ``aged_tightened`` / ``behind_touch`` / ``unknown``. ``None``
    # on legacy fills (pre-1.1.57). ``behind_touch`` was added in
    # 1.1.72 after the operator noticed every fill on tight-tick
    # books was being lumped into ``at_touch`` regardless of where
    # it actually sat — the new category disambiguates "exactly at
    # the venue's best touch" from "1+ tick worse than the touch".
    quote_aggressiveness: Optional[str] = None

    def current_markout_5s_bps(self) -> Optional[float]:
        """Read the latest markout from the live Fill ref. Returns None
        if the delayed markout hasn't landed yet (within first 5s).
        """
        return getattr(self.fill_ref, "markout_5s_bps", None)

    def current_markout_1s_bps(self) -> Optional[float]:
        """1-second markout — lands earliest. Useful as a fallback for
        the bucket display when 5s hasn't accumulated yet.
        """
        return getattr(self.fill_ref, "markout_1s_bps", None)


class FillBucketAggregator:
    """Thread-safe bounded aggregator for quote-age fill buckets.

    Lifecycle:
      * ack hook records (oid, ts_ack) into a 500-entry LRU dict
      * fill hook looks up ts_ack, drops the entry from the LRU, derives
        per-fill metrics, appends to the rolling window
      * ``to_dict()`` aggregates the rolling window into per-bucket
        counts and mean markouts

    Drop-on-miss semantics: if a fill arrives for an oid whose ack we
    never recorded (e.g. fill arrived before our ack-handler ran, or
    bot restarted), the fill is bucketed as ``unknown`` rather than
    fabricating an age. We track the unknown count so the operator
    can see when this is happening.
    """

    def __init__(
        self,
        *,
        fill_window: int = 1000,
        ack_cache_size: int = 1500,
    ) -> None:
        self._lock = threading.Lock()
        self._ack_cache: OrderedDict[str, datetime] = OrderedDict()
        self._ack_cache_size = ack_cache_size
        self._fills: deque[_BucketedFill] = deque(maxlen=fill_window)
        self._unknown_age_count: int = 0
        self._session_total_fills: int = 0

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def note_ack(
        self,
        *,
        order_id_exchange: Optional[int],
        ts_ack: datetime,
    ) -> None:
        """Record an order's venue ACK timestamp. Idempotent on oid:
        re-noting an existing oid updates the timestamp (matters for
        partial fills where the order is acked once and lives across
        multiple fills).
        """
        if order_id_exchange is None or order_id_exchange == 0:
            return
        oid = str(order_id_exchange)
        with self._lock:
            if oid in self._ack_cache:
                self._ack_cache.move_to_end(oid)
            self._ack_cache[oid] = ts_ack
            # FIFO eviction once over the cap (LRU semantics via
            # move_to_end on lookup keeps active oids alive).
            while len(self._ack_cache) > self._ack_cache_size:
                self._ack_cache.popitem(last=False)

    def note_fill(self, fill: "Fill") -> Optional[float]:
        """Bucket this fill. If we have an ack record for its oid,
        compute quote_age_ms; otherwise count it as unknown-age (still
        contributes to the unknown counter but not to a bucket).

        Holds a reference to the Fill so delayed markouts (1s/3s/5s)
        are picked up automatically when the markout job populates them
        in-place on the same Fill object.

        Returns the computed quote-age in milliseconds, or ``None`` when
        no ack record exists. Callers (currently
        ``app.fill_ingestion``) use the return value to stamp
        ``Fill.quote_age_at_fill_ms`` for DB persistence — the
        in-memory aggregate and the persisted column always agree.
        """
        side_str = (
            fill.side.value if hasattr(fill.side, "value") else str(fill.side)
        )
        with self._lock:
            self._session_total_fills += 1
            ts_ack: Optional[datetime] = None
            oid = fill.order_id_exchange
            if oid is not None and oid != 0:
                ts_ack = self._ack_cache.get(str(oid))
            if ts_ack is None:
                self._unknown_age_count += 1
                return None
            age_ms = max(0.0, (fill.ts_fill - ts_ack).total_seconds() * 1000.0)
            self._fills.append(
                _BucketedFill(
                    age_ms=age_ms,
                    bucket_index=_bucket_index_for(age_ms),
                    side=side_str,
                    notional_usd=float(fill.notional or 0.0),
                    fill_ref=fill,
                    quote_aggressiveness=getattr(fill, "quote_aggressiveness", None),
                )
            )
            return age_ms

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Per-bucket aggregates for the live_stats payload. Frontend
        Bot Stats panel renders this as a bucketed bar chart.

        Shape::

            {
              "window_size": 100,
              "fills_in_window": 32,
              "unknown_age_count": 0,
              "session_total_fills": 32,
              "buckets": [
                {"label": "<100ms",   "count": 5,  "mean_markout_bps": -3.1, "median_markout_bps": -2.8, "buy_count": 3, "sell_count": 2, "notional_usd": 105.20},
                ...
              ]
            }

        Empty markouts (None) are excluded from the mean/median; if
        every fill in a bucket has missing markouts, the bucket's
        ``mean_markout_bps`` field is ``None``.
        """
        with self._lock:
            buckets_data: list[dict[str, Any]] = []
            grouped: dict[int, list[_BucketedFill]] = {
                i: [] for i in range(len(_BUCKET_BOUNDS_MS))
            }
            for f in self._fills:
                grouped[f.bucket_index].append(f)
            for i, label in enumerate(_BUCKET_LABELS):
                rows = grouped[i]
                count = len(rows)
                buy_count = sum(1 for r in rows if r.side == "BUY")
                sell_count = sum(1 for r in rows if r.side == "SELL")
                notional = sum(r.notional_usd for r in rows)
                # Read markouts lazily: the Fill object's markout_5s_bps
                # gets populated by the delayed-markout job ~5s after the
                # fill, in-place. Reading here picks up whatever is
                # currently set; fills younger than 5s have None.
                mks = [
                    m for r in rows
                    if (m := r.current_markout_5s_bps()) is not None
                ]
                mean_mk = (sum(mks) / len(mks)) if mks else None
                med_mk = median(mks) if mks else None
                buckets_data.append(
                    {
                        "label": label,
                        "count": count,
                        "buy_count": buy_count,
                        "sell_count": sell_count,
                        "notional_usd": round(notional, 4),
                        "mean_markout_bps": (
                            round(mean_mk, 4) if mean_mk is not None else None
                        ),
                        "median_markout_bps": (
                            round(med_mk, 4) if med_mk is not None else None
                        ),
                    }
                )
            # todo-006 second axis: place-time aggressiveness slice over
            # the same rolling window. Three real categories
            # (``at_touch`` / ``inside`` / ``aged_tightened``) plus
            # ``unknown`` for fills whose parent order isn't tagged
            # (legacy / non-MM placement / lookup miss). Operator can
            # answer "is the quote-aging path paying for itself?" by
            # comparing markouts across categories.
            agg_categories = (
                "at_touch",
                "behind_touch",
                "inside",
                "aged_tightened",
                "unknown",
            )
            agg_groups: dict[str, list[_BucketedFill]] = {
                k: [] for k in agg_categories
            }
            for f in self._fills:
                key = f.quote_aggressiveness or "unknown"
                if key not in agg_groups:
                    key = "unknown"
                agg_groups[key].append(f)
            agg_data: list[dict[str, Any]] = []
            for label in agg_categories:
                rows = agg_groups[label]
                count = len(rows)
                buy_count = sum(1 for r in rows if r.side == "BUY")
                sell_count = sum(1 for r in rows if r.side == "SELL")
                notional = sum(r.notional_usd for r in rows)
                mks_a = [
                    m for r in rows
                    if (m := r.current_markout_5s_bps()) is not None
                ]
                mean_mk_a = (sum(mks_a) / len(mks_a)) if mks_a else None
                med_mk_a = median(mks_a) if mks_a else None
                agg_data.append(
                    {
                        "label": label,
                        "count": count,
                        "buy_count": buy_count,
                        "sell_count": sell_count,
                        "notional_usd": round(notional, 4),
                        "mean_markout_bps": (
                            round(mean_mk_a, 4)
                            if mean_mk_a is not None
                            else None
                        ),
                        "median_markout_bps": (
                            round(med_mk_a, 4)
                            if med_mk_a is not None
                            else None
                        ),
                    }
                )
            return {
                "window_size": self._fills.maxlen,
                "fills_in_window": len(self._fills),
                "unknown_age_count": self._unknown_age_count,
                "session_total_fills": self._session_total_fills,
                "buckets": buckets_data,
                "aggressiveness": agg_data,
            }

    # Test / debug helper.
    def _ack_cache_size_now(self) -> int:
        with self._lock:
            return len(self._ack_cache)
