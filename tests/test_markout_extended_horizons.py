"""v1.4.98 — diagnostic markout horizons (15s / 30s / 60s / 120s).

The 1s / 3s / 5s endpoints already exist (see ``tests/test_markout.py``).
This file covers ONLY the v1.4.98 additions:

* New ``Fill`` fields are wired through the resolution loop.
* Each horizon resolves at its own age threshold (independent gates).
* Jobs survive in the pending deque until ALL seven horizons resolve.
* The storage update path receives the 8-tuple ``(fill_id, m1, m3, m5,
  m15, m30, m60, m120)`` shape.
* Insufficient-age cases (e.g., 20s after fill) only resolve through
  15s; 30s/60s/120s remain None.

None of these changes touch gate logic. The 5s endpoint remains
canonical for sizing / aging / soft-flatten control surfaces.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

from app.enums import Side
from app.markout import (
    process_pending_markouts,
    register_fill_for_delayed_markouts,
)
from app.models import Fill


class _TrackingLock:
    def __init__(self) -> None:
        self.held = False

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self.held = False
        return False


class _TrackingStorage:
    """Captures the 8-tuple shape produced by v1.4.98's markout
    resolution loop. Tests slice individual horizons by index."""

    def __init__(self, lock: _TrackingLock) -> None:
        self.lock = lock
        self.calls: list[tuple] = []

    def update_fill_markouts(
        self,
        fill_id: str,
        m1: float | None,
        m3: float | None,
        m5: float | None,
        m15: float | None = None,
        m30: float | None = None,
        m60: float | None = None,
        m120: float | None = None,
    ) -> None:
        assert self.lock.held is False, "storage I/O must be outside lock"
        self.calls.append((fill_id, m1, m3, m5, m15, m30, m60, m120))


def _make_fill(fill_id: str, t_fill: datetime, *, side: Side = Side.BUY, px: float = 100.0) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=t_fill,
        symbol="ETH",
        side=side,
        price=px,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=px,
        best_bid_at_fill=px - 0.1,
        best_ask_at_fill=px + 0.1,
        book_snapshot_quality="full",
    )


def _make_state() -> object:
    lock = _TrackingLock()
    st = type("S", (), {})()
    st._lock = lock
    st.pending_markout_jobs = deque(maxlen=8000)
    st.recent_fills = deque(maxlen=200)
    return st


# ---------------------------------------------------------------------------
# New-horizon fields exist + start as None
# ---------------------------------------------------------------------------


def test_fill_has_extended_horizon_fields_defaulting_to_none() -> None:
    f = _make_fill("f1", datetime.now(timezone.utc))
    assert f.markout_15s_bps is None
    assert f.markout_30s_bps is None
    assert f.markout_60s_bps is None
    assert f.markout_120s_bps is None


# ---------------------------------------------------------------------------
# Each horizon resolves at its own age threshold
# ---------------------------------------------------------------------------


def test_15s_horizon_resolves_when_age_exceeds_15_seconds() -> None:
    st = _make_state()
    sink = _TrackingStorage(st._lock)
    now = datetime.now(timezone.utc) - timedelta(seconds=20)
    f = _make_fill("f1", now, side=Side.BUY, px=100.0)
    st.recent_fills.append(f)
    register_fill_for_delayed_markouts(st, f)
    # mid moved AGAINST a BUY (adverse): 100 → 99.9.
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 99.9)
    # 1s, 3s, 5s, 15s should all be set; 30s/60s/120s still None.
    assert f.markout_1s_bps is not None
    assert f.markout_5s_bps is not None
    assert f.markout_15s_bps is not None
    assert f.markout_15s_bps < 0  # adverse for BUY when mid drops
    assert f.markout_30s_bps is None
    assert f.markout_60s_bps is None
    assert f.markout_120s_bps is None


def test_30s_horizon_resolves_when_age_exceeds_30_seconds() -> None:
    st = _make_state()
    sink = _TrackingStorage(st._lock)
    now = datetime.now(timezone.utc) - timedelta(seconds=35)
    f = _make_fill("f1", now, side=Side.SELL, px=100.0)
    st.recent_fills.append(f)
    register_fill_for_delayed_markouts(st, f)
    # mid moved AGAINST a SELL (adverse): 100 → 100.1.
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 100.1)
    assert f.markout_15s_bps is not None
    assert f.markout_30s_bps is not None
    assert f.markout_30s_bps < 0
    assert f.markout_60s_bps is None
    assert f.markout_120s_bps is None


def test_60s_horizon_resolves_when_age_exceeds_60_seconds() -> None:
    st = _make_state()
    sink = _TrackingStorage(st._lock)
    now = datetime.now(timezone.utc) - timedelta(seconds=65)
    f = _make_fill("f1", now, side=Side.BUY, px=100.0)
    st.recent_fills.append(f)
    register_fill_for_delayed_markouts(st, f)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 100.5)
    # 1s..60s set; 120s still pending.
    assert f.markout_60s_bps is not None
    assert f.markout_60s_bps > 0  # favorable for BUY when mid rises
    assert f.markout_120s_bps is None


def test_120s_horizon_resolves_when_age_exceeds_120_seconds() -> None:
    st = _make_state()
    sink = _TrackingStorage(st._lock)
    now = datetime.now(timezone.utc) - timedelta(seconds=125)
    f = _make_fill("f1", now, side=Side.BUY, px=100.0)
    st.recent_fills.append(f)
    register_fill_for_delayed_markouts(st, f)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 101.0)
    # All seven horizons set.
    assert f.markout_1s_bps is not None
    assert f.markout_3s_bps is not None
    assert f.markout_5s_bps is not None
    assert f.markout_15s_bps is not None
    assert f.markout_30s_bps is not None
    assert f.markout_60s_bps is not None
    assert f.markout_120s_bps is not None


# ---------------------------------------------------------------------------
# Job retention: stays in deque until ALL horizons resolve
# ---------------------------------------------------------------------------


def test_job_retained_until_all_horizons_resolve() -> None:
    """Pre-v1.4.98 the loop dropped jobs once 5s resolved; v1.4.98
    keeps them until 120s also resolves so the longer-horizon columns
    can be filled in subsequent ticks."""
    st = _make_state()
    sink = _TrackingStorage(st._lock)
    # First tick: fill is 20s old → resolves 1s/3s/5s/15s only.
    now_first = datetime.now(timezone.utc) - timedelta(seconds=20)
    f = _make_fill("f1", now_first, side=Side.BUY, px=100.0)
    st.recent_fills.append(f)
    register_fill_for_delayed_markouts(st, f)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 99.9)
    assert f.markout_15s_bps is not None
    assert f.markout_120s_bps is None
    assert len(st.pending_markout_jobs) == 1, (
        "job must be retained when later horizons are still pending"
    )

    # Second tick: we simulate 130 seconds elapsing by manipulating the
    # job's t_fill backward. Now all horizons resolve in one go.
    job = st.pending_markout_jobs[0]
    job.t_fill = datetime.now(timezone.utc) - timedelta(seconds=130)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 99.5)
    assert f.markout_30s_bps is not None
    assert f.markout_60s_bps is not None
    assert f.markout_120s_bps is not None
    assert len(st.pending_markout_jobs) == 0, (
        "job must be dropped once all seven horizons have resolved"
    )


# ---------------------------------------------------------------------------
# Storage shape — 8-tuple, in the order the schema expects
# ---------------------------------------------------------------------------


def test_storage_update_carries_8_tuple_for_extended_horizons() -> None:
    st = _make_state()
    sink = _TrackingStorage(st._lock)
    now = datetime.now(timezone.utc) - timedelta(seconds=125)
    f = _make_fill("f1", now, side=Side.BUY, px=100.0)
    st.recent_fills.append(f)
    register_fill_for_delayed_markouts(st, f)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 100.5)

    assert len(sink.calls) == 1
    call = sink.calls[0]
    assert len(call) == 8, f"expected 8-tuple (fill_id + 7 horizons), got {len(call)}"
    fill_id, m1, m3, m5, m15, m30, m60, m120 = call
    assert fill_id == "f1"
    for h, name in zip(
        (m1, m3, m5, m15, m30, m60, m120),
        ("1s", "3s", "5s", "15s", "30s", "60s", "120s"),
    ):
        assert h is not None, f"horizon {name} should have resolved"


# ---------------------------------------------------------------------------
# Idempotence: re-running on a fully-resolved fill is a no-op
# ---------------------------------------------------------------------------


def test_re_resolution_does_not_overwrite_or_double_update() -> None:
    st = _make_state()
    sink = _TrackingStorage(st._lock)
    now = datetime.now(timezone.utc) - timedelta(seconds=125)
    f = _make_fill("f1", now, side=Side.BUY, px=100.0)
    st.recent_fills.append(f)
    register_fill_for_delayed_markouts(st, f)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 100.5)
    # All seven horizons set; job evicted from pending.
    assert len(sink.calls) == 1
    original = (
        f.markout_1s_bps, f.markout_3s_bps, f.markout_5s_bps,
        f.markout_15s_bps, f.markout_30s_bps, f.markout_60s_bps,
        f.markout_120s_bps,
    )

    # Try to re-register and re-resolve at a different mid. Since the
    # fill's markout_*_bps are non-None, the conditional `is None`
    # guards in the resolution loop should prevent any update.
    register_fill_for_delayed_markouts(st, f)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 110.0)

    # No new storage updates (none of the seven u_* would be non-None).
    assert len(sink.calls) == 1
    # Values unchanged.
    after = (
        f.markout_1s_bps, f.markout_3s_bps, f.markout_5s_bps,
        f.markout_15s_bps, f.markout_30s_bps, f.markout_60s_bps,
        f.markout_120s_bps,
    )
    assert after == original
