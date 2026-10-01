"""Tests for app.fill_bucket_metrics.FillBucketAggregator (todo-006)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from app.fill_bucket_metrics import FillBucketAggregator


class _MockFill:
    """Stand-in for app.models.Fill — only the attributes the aggregator
    reads. Keeps tests independent of the real Fill dataclass shape so
    a refactor to Fill doesn't break these.
    """

    def __init__(
        self,
        *,
        side: str,
        order_id_exchange: int,
        ts_fill: datetime,
        notional: float,
        markout_1s_bps: Optional[float] = None,
        markout_5s_bps: Optional[float] = None,
    ) -> None:
        self.side = side
        self.order_id_exchange = order_id_exchange
        self.ts_fill = ts_fill
        self.notional = notional
        self.markout_1s_bps = markout_1s_bps
        self.markout_5s_bps = markout_5s_bps


def _fill_at(ack_ts: datetime, age_ms: int, oid: int, side: str = "BUY",
             markout_5s: Optional[float] = None) -> _MockFill:
    return _MockFill(
        side=side,
        order_id_exchange=oid,
        ts_fill=ack_ts + timedelta(milliseconds=age_ms),
        notional=21.34,
        markout_5s_bps=markout_5s,
    )


def test_buckets_partition_by_age_correctly() -> None:
    """Each fill lands in the bucket whose [lo, hi) bracket contains
    its age_ms. Boundary cases: age=100ms → 100-500ms bucket (lo
    inclusive, hi exclusive).
    """
    agg = FillBucketAggregator()
    ack = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
    for i, age_ms in enumerate([50, 100, 499, 500, 1999, 2000, 9999, 10_000]):
        oid = 1000 + i
        agg.note_ack(order_id_exchange=oid, ts_ack=ack)
        agg.note_fill(_fill_at(ack, age_ms, oid))
    out = agg.to_dict()
    counts = {b["label"]: b["count"] for b in out["buckets"]}
    # Boundaries: 100 falls into 100-500ms (lo inclusive); 500 into 500ms-2s; etc.
    assert counts["<100ms"] == 1     # age 50
    assert counts["100-500ms"] == 2  # 100, 499
    assert counts["500ms-2s"] == 2   # 500, 1999
    assert counts["2-10s"] == 2      # 2000, 9999
    assert counts["10s+"] == 1       # 10000


def test_unknown_age_increments_when_ack_missing() -> None:
    """If a fill arrives for an oid we never recorded an ack for (e.g.
    bot restarted between place and fill), the bucket counters skip
    it but ``unknown_age_count`` increments.
    """
    agg = FillBucketAggregator()
    ack = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
    agg.note_fill(_fill_at(ack, 200, 999))  # no note_ack called
    out = agg.to_dict()
    assert out["unknown_age_count"] == 1
    assert out["session_total_fills"] == 1
    assert sum(b["count"] for b in out["buckets"]) == 0


def test_note_fill_returns_age_ms_for_persistence() -> None:
    """``note_fill`` returns the computed age so the fill ingestion
    path can stamp it onto ``Fill.quote_age_at_fill_ms`` for DB
    persistence. None when the ack is missing.
    """
    agg = FillBucketAggregator()
    ack = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
    agg.note_ack(order_id_exchange=42, ts_ack=ack)
    age = agg.note_fill(_fill_at(ack, 750, 42))
    assert age is not None
    assert abs(age - 750.0) < 1e-6
    # No-ack fill returns None — caller leaves Fill.quote_age_at_fill_ms NULL.
    assert agg.note_fill(_fill_at(ack, 200, 999)) is None


def test_lazy_markout_picks_up_in_place_updates() -> None:
    """The aggregator holds a Fill *reference*, not a snapshot of
    ``markout_5s_bps`` at note_fill time. Mutating the field after
    bucketing must be reflected in the next ``to_dict()``.
    """
    agg = FillBucketAggregator()
    ack = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
    f = _fill_at(ack, 500, 1, markout_5s=None)
    agg.note_ack(order_id_exchange=1, ts_ack=ack)
    agg.note_fill(f)
    out1 = agg.to_dict()
    bucket = next(b for b in out1["buckets"] if b["label"] == "500ms-2s")
    assert bucket["count"] == 1
    assert bucket["mean_markout_bps"] is None  # markout hasn't landed
    # Simulate the markout job populating the field 5s later:
    f.markout_5s_bps = -2.5
    out2 = agg.to_dict()
    bucket2 = next(b for b in out2["buckets"] if b["label"] == "500ms-2s")
    assert bucket2["mean_markout_bps"] == -2.5


def test_window_evicts_oldest_at_capacity() -> None:
    agg = FillBucketAggregator(fill_window=3)
    ack = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        agg.note_ack(order_id_exchange=i, ts_ack=ack)
        agg.note_fill(_fill_at(ack, 50, i, markout_5s=float(i)))
    out = agg.to_dict()
    assert out["fills_in_window"] == 3
    assert out["session_total_fills"] == 5
    bucket = next(b for b in out["buckets"] if b["label"] == "<100ms")
    assert bucket["count"] == 3
    # Window holds fills 2, 3, 4 → mean markout = 3.0
    assert bucket["mean_markout_bps"] == 3.0


def test_per_side_count_split() -> None:
    agg = FillBucketAggregator()
    ack = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
    for i, side in enumerate(["BUY", "BUY", "SELL", "BUY", "SELL"]):
        oid = 100 + i
        agg.note_ack(order_id_exchange=oid, ts_ack=ack)
        agg.note_fill(_fill_at(ack, 50, oid, side=side))
    out = agg.to_dict()
    bucket = next(b for b in out["buckets"] if b["label"] == "<100ms")
    assert bucket["count"] == 5
    assert bucket["buy_count"] == 3
    assert bucket["sell_count"] == 2


def test_ack_cache_lru_eviction() -> None:
    """Once the ack cache hits its cap, oldest ACKs are evicted; fills
    arriving for those orders count as unknown-age.
    """
    agg = FillBucketAggregator(ack_cache_size=3)
    ack = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
    for i in range(5):
        agg.note_ack(order_id_exchange=i, ts_ack=ack)
    # Acks 0 and 1 should be evicted by now (cache holds 2, 3, 4).
    assert agg._ack_cache_size_now() == 3
    agg.note_fill(_fill_at(ack, 50, 0))  # evicted ack
    out = agg.to_dict()
    assert out["unknown_age_count"] == 1


def test_zero_oid_treated_as_no_ack() -> None:
    """Some venues return ordId=0 transiently. Treat it as absent for
    both note_ack and note_fill so we never index by 0.
    """
    agg = FillBucketAggregator()
    ack = datetime(2026, 5, 8, 10, 0, 0, tzinfo=timezone.utc)
    agg.note_ack(order_id_exchange=0, ts_ack=ack)  # silently ignored
    agg.note_fill(_fill_at(ack, 50, 0))  # no ack recorded → unknown
    out = agg.to_dict()
    assert out["unknown_age_count"] == 1


def test_to_dict_shape_complete() -> None:
    """Sanity: to_dict always returns all 5 buckets, even when empty,
    so the frontend can render a stable layout from the first second.
    """
    agg = FillBucketAggregator()
    out = agg.to_dict()
    assert [b["label"] for b in out["buckets"]] == [
        "<100ms",
        "100-500ms",
        "500ms-2s",
        "2-10s",
        "10s+",
    ]
    for b in out["buckets"]:
        assert b["count"] == 0
        assert b["buy_count"] == 0
        assert b["sell_count"] == 0
        assert b["notional_usd"] == 0.0
        assert b["mean_markout_bps"] is None
        assert b["median_markout_bps"] is None
