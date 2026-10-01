from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.enums import Side
from app.markout import (
    delayed_markout_bps,
    process_pending_markouts,
    register_fill_for_delayed_markouts,
)
from app.models import BestBidAsk, Fill, PositionSnapshot
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


class _FailOnEnterLock:
    def __enter__(self):
        raise AssertionError("lock should not be entered")

    def __exit__(self, exc_type, exc, tb):
        return False


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
    def __init__(self, lock: _TrackingLock) -> None:
        self.lock = lock
        # v1.4.98 — calls are 8-tuples carrying all 7 horizons + fill_id.
        # Tests that only care about 1s/3s/5s should slice [:4]; the new
        # 15s/30s/60s/120s slots will be None on short-window jobs.
        self.calls: list[
            tuple[
                str,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
                float | None,
            ]
        ] = []

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
        # Regression guard: writes should happen after state lock scope exits.
        assert self.lock.held is False
        self.calls.append((fill_id, m1, m3, m5, m15, m30, m60, m120))


def test_delayed_markout_bps_buy_adverse_when_mid_drops() -> None:
    assert delayed_markout_bps(Side.BUY, 100.0, 99.0) < 0


def test_delayed_markout_bps_sell_adverse_when_mid_rises() -> None:
    assert delayed_markout_bps(Side.SELL, 100.0, 101.0) < 0


def test_delayed_markout_bps_nonfinite_returns_zero() -> None:
    import math

    assert delayed_markout_bps(Side.BUY, float("nan"), 100.0) == 0.0
    assert delayed_markout_bps(Side.BUY, 100.0, float("nan")) == 0.0
    assert delayed_markout_bps(Side.BUY, float("inf"), 100.0) == 0.0
    assert math.isfinite(delayed_markout_bps(Side.BUY, 100.0, 101.0))


def test_process_pending_markouts_empty_queue_returns_without_lock() -> None:
    class _State:
        def __init__(self) -> None:
            from collections import deque

            self.pending_markout_jobs = deque(maxlen=400)
            self.recent_fills = deque(maxlen=200)
            self._lock = _FailOnEnterLock()

    st = _State()
    fake_storage = object()
    process_pending_markouts(st, fake_storage, datetime.now(timezone.utc), 100.0)  # type: ignore[arg-type]


def test_process_pending_markouts_persists_after_unlock_and_honors_max_jobs() -> None:
    from collections import deque

    lock = _TrackingLock()
    st = type("S", (), {})()
    st._lock = lock
    st.pending_markout_jobs = deque(maxlen=8000)
    # v1.4.98 — fills are timestamped 125 seconds in the past so all
    # SEVEN horizons (1s/3s/5s/15s/30s/60s/120s) resolve in one pass
    # and the job is removed from the deque. Pre-v1.4.98 this used
    # 6 seconds, which only resolved 1s/3s/5s — fine when those were
    # the only horizons.
    now = datetime.now(timezone.utc) - timedelta(seconds=125)
    f1 = Fill(
        fill_id="f1",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=now,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        best_bid_at_fill=99.9,
        best_ask_at_fill=100.1,
        book_snapshot_quality="full",
    )
    f2 = Fill(
        fill_id="f2",
        order_id_exchange=2,
        client_order_id=None,
        ts_fill=now,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        best_bid_at_fill=99.9,
        best_ask_at_fill=100.1,
        book_snapshot_quality="full",
    )
    st.recent_fills = deque([f1, f2], maxlen=200)
    register_fill_for_delayed_markouts(st, f1)
    register_fill_for_delayed_markouts(st, f2)
    sink = _TrackingStorage(lock)

    process_pending_markouts(st, sink, datetime.now(timezone.utc), 100.5, max_jobs=1)  # type: ignore[arg-type]
    assert len(sink.calls) == 1
    assert sink.calls[0][0] == "f1"
    assert len(st.pending_markout_jobs) == 1
    assert st.pending_markout_jobs[0].fill_id == "f2"

    process_pending_markouts(st, sink, datetime.now(timezone.utc), 100.5, max_jobs=1)  # type: ignore[arg-type]
    assert len(sink.calls) == 2
    assert sink.calls[1][0] == "f2"
    assert len(st.pending_markout_jobs) == 0


def test_markout_integration_snapshot_then_fill_then_resolution() -> None:
    """Apply a market snapshot, ingest a fill, then resolve delayed markouts on a later tick."""
    path = Path(tempfile.gettempdir()) / f"mm_markout_int_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    # v1.4.98 — 125 s in the past so all seven horizons (incl. 120s)
    # resolve in one pass and the job is evicted. Pre-v1.4.98 this used
    # 6 s, valid when only 1s/3s/5s were tracked.
    t0 = datetime.now(timezone.utc) - timedelta(seconds=125)
    bb = BestBidAsk(
        symbol=settings.symbol,
        best_bid=99.9,
        best_ask=100.1,
        mid_price=100.0,
        spread_bps=20.0,
        ts_local=t0,
    )
    pos = PositionSnapshot(
        symbol=settings.symbol,
        position_qty=0.0,
        avg_entry_price=None,
        mark_price=100.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
        ts_local=t0,
    )
    state.apply_market_snapshot(bb, pos, None)
    assert state.market is not None and state.market.mid_price == 100.0

    f = Fill(
        fill_id=str(uuid.uuid4()),
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=t0,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        best_bid_at_fill=99.9,
        best_ask_at_fill=100.1,
        book_snapshot_quality="full",
    )
    assert state.record_fill(f) is True
    register_fill_for_delayed_markouts(state, f)
    storage.insert_fill_row(
        {
            "fill_id": f.fill_id,
            "order_id_exchange": None,
            "client_order_id": None,
            "ts_fill": f.ts_fill.isoformat(),
            "symbol": f.symbol,
            "side": f.side.value,
            "price": f.price,
            "size": f.size,
            "notional": f.notional,
            "fee": f.fee,
            "liquidity_flag": f.liquidity_flag,
            "mid_at_fill": f.mid_at_fill,
            "best_bid_at_fill": f.best_bid_at_fill,
            "best_ask_at_fill": f.best_ask_at_fill,
            "book_snapshot_quality": f.book_snapshot_quality,
            "book_reference_quality": f.book_reference_quality,
            "markout_1s_bps": None,
            "markout_3s_bps": None,
            "markout_5s_bps": None,
        }
    )
    process_pending_markouts(state, storage, datetime.now(timezone.utc), 100.5)
    assert f.markout_1s_bps is not None
    assert f.markout_3s_bps is not None
    assert f.markout_5s_bps is not None
    assert len(state.pending_markout_jobs) == 0
    path.unlink(missing_ok=True)


def test_process_pending_sets_markouts_after_delay() -> None:
    path = Path(tempfile.gettempdir()) / f"mm_markout_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    # v1.4.98 — 125 s past so all seven horizons resolve and deque drains.
    t0 = datetime.now(timezone.utc) - timedelta(seconds=125)
    f = Fill(
        fill_id=str(uuid.uuid4()),
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=t0,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        best_bid_at_fill=99.9,
        best_ask_at_fill=100.1,
        book_snapshot_quality="full",
    )
    assert state.record_fill(f) is True
    register_fill_for_delayed_markouts(state, f)
    storage.insert_fill_row(
        {
            "fill_id": f.fill_id,
            "order_id_exchange": None,
            "client_order_id": None,
            "ts_fill": f.ts_fill.isoformat(),
            "symbol": f.symbol,
            "side": f.side.value,
            "price": f.price,
            "size": f.size,
            "notional": f.notional,
            "fee": f.fee,
            "liquidity_flag": f.liquidity_flag,
            "mid_at_fill": f.mid_at_fill,
            "best_bid_at_fill": f.best_bid_at_fill,
            "best_ask_at_fill": f.best_ask_at_fill,
            "book_snapshot_quality": f.book_snapshot_quality,
            "markout_1s_bps": None,
            "markout_3s_bps": None,
            "markout_5s_bps": None,
        }
    )
    process_pending_markouts(state, storage, datetime.now(timezone.utc), 100.5)
    assert f.markout_1s_bps is not None
    assert f.markout_3s_bps is not None
    assert f.markout_5s_bps is not None
    assert len(state.pending_markout_jobs) == 0
    path.unlink(missing_ok=True)


def test_markout_resolves_after_recent_fills_deque_evicts_burst_fills() -> None:
    """BUG-012: under bursty fill conditions, fills age out of the
    ``recent_fills`` deque (bounded by maxlen) before all delayed markout
    horizons resolve. Pre-fix, ``_find_fill`` returned None and the
    markout was silently dropped. Post-fix, the job holds a Fill reference
    so the horizon resolves regardless of deque membership.

    The exact maxlen is read from the deque at runtime, so this test is
    immune to future widenings (200 → 1000 in 1.1.128, etc.).
    """
    path = Path(tempfile.gettempdir()) / f"mm_markout_burst_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    t0 = datetime.now(timezone.utc) - timedelta(seconds=6)

    victim = Fill(
        fill_id="victim-burst",
        order_id_exchange=999,
        client_order_id=None,
        ts_fill=t0,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        best_bid_at_fill=99.9,
        best_ask_at_fill=100.1,
        book_snapshot_quality="full",
    )
    state.recent_fills.append(victim)
    register_fill_for_delayed_markouts(state, victim)

    # Burst: enough fills to push `victim` off the deque.
    for i in range(state.recent_fills.maxlen):
        burst = Fill(
            fill_id=f"burst-{i}",
            order_id_exchange=i,
            client_order_id=None,
            ts_fill=t0,
            symbol="ETH",
            side=Side.BUY,
            price=100.0,
            size=0.1,
            notional=10.0,
            fee=0.0,
            liquidity_flag="x",
            mid_at_fill=100.0,
        )
        state.recent_fills.append(burst)

    assert all(x.fill_id != "victim-burst" for x in state.recent_fills)

    process_pending_markouts(state, storage, datetime.now(timezone.utc), 100.5)

    assert victim.markout_1s_bps is not None
    assert victim.markout_3s_bps is not None
    assert victim.markout_5s_bps is not None
    assert getattr(state, "markout_jobs_orphaned_count", 0) == 0
    path.unlink(missing_ok=True)


def test_markout_orphan_counter_increments_on_legacy_jobs_without_fill_ref() -> None:
    """If a legacy ``PendingMarkoutJob`` (pre-BUG-012) lacks a ``fill`` ref
    AND the fill is no longer in ``recent_fills``, the orphan counter must
    bump so operators can spot regressions.
    """
    path = Path(tempfile.gettempdir()) / f"mm_markout_orphan_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    t0 = datetime.now(timezone.utc) - timedelta(seconds=6)

    from app.markout import PendingMarkoutJob

    state.pending_markout_jobs.append(
        PendingMarkoutJob(
            fill_id="legacy-orphan",
            side=Side.BUY,
            fill_price=100.0,
            t_fill=t0,
            fill=None,
        )
    )
    process_pending_markouts(state, storage, datetime.now(timezone.utc), 100.5)
    assert getattr(state, "markout_jobs_orphaned_count", 0) >= 1
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# v1.4.36 Codex #4 — pending-markout-jobs overflow counter
# ---------------------------------------------------------------------------


def test_markout_overflow_counter_increments_when_deque_full() -> None:
    """Codex #4 fix: ``pending_markout_jobs`` is a bounded
    ``deque(maxlen=400)``; pre-v1.4.36 appending while full silently
    evicted the oldest entry (a 1s/3s/5s markout completion was
    lost). v1.4.36 detects the overflow in
    ``register_fill_for_delayed_markouts`` and increments
    ``state.markout_jobs_dropped_overflow_count`` so the silent loss
    becomes observable in ``state_current`` snapshots.

    Shape: append ``maxlen + 5`` fills; expect counter == 5 and
    deque length capped at ``maxlen``.
    """
    from collections import deque
    from app.markout import PendingMarkoutJob

    st = type("S", (), {})()

    class _NoOpLock:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    st._lock = _NoOpLock()
    # Shrink maxlen to keep the test fast / RAM-light. ``maxlen`` is
    # a constructor argument so we re-create with our test size; the
    # production deque uses 400 (set in BotState.__init__).
    st.pending_markout_jobs = deque(maxlen=8)
    st.markout_jobs_dropped_overflow_count = 0
    base_ts = datetime.now(timezone.utc)
    # Build 13 fills (5 over the maxlen of 8).
    for i in range(13):
        f = Fill(
            fill_id=f"f{i}",
            order_id_exchange=i,
            client_order_id=None,
            ts_fill=base_ts + timedelta(milliseconds=i),
            symbol="ETH",
            side=Side.BUY,
            price=100.0,
            size=0.01,
            notional=1.0,
            fee=0.0,
            liquidity_flag="x",
            mid_at_fill=100.0,
            best_bid_at_fill=99.9,
            best_ask_at_fill=100.1,
            book_snapshot_quality="full",
        )
        register_fill_for_delayed_markouts(st, f)
    # 13 appends into a maxlen=8 deque → 5 overflows
    assert len(st.pending_markout_jobs) == 8
    assert st.markout_jobs_dropped_overflow_count == 5


def test_process_pending_markouts_uses_batch_method_when_available() -> None:
    """v1.4.37 Codex #6 fix: ``process_pending_markouts`` prefers
    ``update_fill_markouts_many`` when the storage exposes it, so all
    finalisations land in ONE transaction instead of N. Falls back to
    the per-fill ``update_fill_markouts`` when only that method exists
    (e.g. legacy / test doubles)."""
    from collections import deque

    lock = _TrackingLock()
    st = type("S", (), {})()
    st._lock = lock
    st.pending_markout_jobs = deque(maxlen=400)
    now = datetime.now(timezone.utc) - timedelta(seconds=6)

    # Two due fills → two updates in one tick.
    f1 = Fill(
        fill_id="f-batch-1",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=now,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        best_bid_at_fill=99.9,
        best_ask_at_fill=100.1,
        book_snapshot_quality="full",
    )
    f2 = Fill(
        fill_id="f-batch-2",
        order_id_exchange=2,
        client_order_id=None,
        ts_fill=now,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        best_bid_at_fill=99.9,
        best_ask_at_fill=100.1,
        book_snapshot_quality="full",
    )
    st.recent_fills = deque([f1, f2], maxlen=200)
    register_fill_for_delayed_markouts(st, f1)
    register_fill_for_delayed_markouts(st, f2)

    class _BatchAwareStorage:
        def __init__(self, lock: _TrackingLock) -> None:
            self.lock = lock
            self.many_calls: list[
                list[tuple[str, float | None, float | None, float | None]]
            ] = []
            self.single_calls: list[
                tuple[str, float | None, float | None, float | None]
            ] = []

        def update_fill_markouts_many(
            self,
            rows: list[
                tuple[str, float | None, float | None, float | None]
            ],
        ) -> None:
            # Regression guard same as single-update path.
            assert self.lock.held is False
            self.many_calls.append(list(rows))

        def update_fill_markouts(
            self,
            fill_id: str,
            m1: float | None,
            m3: float | None,
            m5: float | None,
        ) -> None:
            self.single_calls.append((fill_id, m1, m3, m5))

    sink = _BatchAwareStorage(lock)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 100.5)  # type: ignore[arg-type]
    # Codex #6 contract: batch helper invoked exactly once with both
    # finalisations bundled; per-fill helper never invoked.
    assert len(sink.many_calls) == 1, (
        f"expected exactly 1 batch call, got {len(sink.many_calls)}"
    )
    batch_rows = sink.many_calls[0]
    fill_ids = {r[0] for r in batch_rows}
    assert fill_ids == {"f-batch-1", "f-batch-2"}, (
        f"expected both fills in batch, got {fill_ids}"
    )
    assert sink.single_calls == [], (
        "Codex #6 regression: per-fill update_fill_markouts called "
        "even though update_fill_markouts_many was available."
    )


def test_process_pending_markouts_falls_back_to_per_fill_when_batch_missing() -> None:
    """v1.4.37 Codex #6 negative control: when the storage exposes
    only the legacy single-update method, ``process_pending_markouts``
    falls back to the per-fill path. This preserves back-compat with
    test doubles like ``_TrackingStorage`` and with any external
    callers that haven't been upgraded."""
    from collections import deque

    lock = _TrackingLock()
    st = type("S", (), {})()
    st._lock = lock
    st.pending_markout_jobs = deque(maxlen=400)
    now = datetime.now(timezone.utc) - timedelta(seconds=6)
    f1 = Fill(
        fill_id="f-fallback-1",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=now,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        best_bid_at_fill=99.9,
        best_ask_at_fill=100.1,
        book_snapshot_quality="full",
    )
    st.recent_fills = deque([f1], maxlen=200)
    register_fill_for_delayed_markouts(st, f1)

    sink = _TrackingStorage(lock)
    process_pending_markouts(st, sink, datetime.now(timezone.utc), 100.5)  # type: ignore[arg-type]
    # Legacy single-call path took the write.
    assert len(sink.calls) == 1
    assert sink.calls[0][0] == "f-fallback-1"


def test_update_fill_markouts_many_batches_into_one_transaction() -> None:
    """v1.4.37 Codex #6 fix: ``Storage.update_fill_markouts_many``
    persists N updates in a single ``self.connection()`` context.
    This unit-test exercises the storage method end-to-end against
    real SQLite and confirms all rows are committed."""
    path = (
        Path(tempfile.gettempdir())
        / f"mm_markout_batch_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    # Seed three fill rows with NULL markouts.
    for i in range(3):
        storage.insert_fill_row(
            {
                "fill_id": f"fb-{i}",
                "ts_fill": (
                    datetime.now(timezone.utc)
                    + timedelta(milliseconds=i)
                ).isoformat(),
                "symbol": "ETH",
                "side": "BUY",
                "price": 100.0,
                "size": 0.1,
                "notional": 10.0,
                "fee": 0.0,
                "liquidity_flag": "resting",
            }
        )
    # Batch update: write distinct markouts on each fill.
    storage.update_fill_markouts_many(
        [
            ("fb-0", 1.0, 2.0, 3.0),
            ("fb-1", 4.0, 5.0, 6.0),
            ("fb-2", 7.0, 8.0, 9.0),
        ]
    )
    # Verify via direct DB read.
    fills = storage.recent_fills(limit=10)
    by_id = {f["fill_id"]: f for f in fills}
    assert by_id["fb-0"]["markout_1s_bps"] == 1.0
    assert by_id["fb-0"]["markout_3s_bps"] == 2.0
    assert by_id["fb-0"]["markout_5s_bps"] == 3.0
    assert by_id["fb-1"]["markout_1s_bps"] == 4.0
    assert by_id["fb-2"]["markout_5s_bps"] == 9.0
    # Empty input is a no-op.
    storage.update_fill_markouts_many([])
    path.unlink(missing_ok=True)


def test_markout_overflow_counter_stays_zero_under_capacity() -> None:
    """Negative control: appending below capacity does NOT bump the
    overflow counter."""
    from collections import deque

    st = type("S", (), {})()

    class _NoOpLock:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    st._lock = _NoOpLock()
    st.pending_markout_jobs = deque(maxlen=8)
    st.markout_jobs_dropped_overflow_count = 0
    base_ts = datetime.now(timezone.utc)
    # 5 fills → well under the 8-slot maxlen.
    for i in range(5):
        f = Fill(
            fill_id=f"f{i}",
            order_id_exchange=i,
            client_order_id=None,
            ts_fill=base_ts + timedelta(milliseconds=i),
            symbol="ETH",
            side=Side.BUY,
            price=100.0,
            size=0.01,
            notional=1.0,
            fee=0.0,
            liquidity_flag="x",
            mid_at_fill=100.0,
            best_bid_at_fill=99.9,
            best_ask_at_fill=100.1,
            book_snapshot_quality="full",
        )
        register_fill_for_delayed_markouts(st, f)
    assert len(st.pending_markout_jobs) == 5
    assert st.markout_jobs_dropped_overflow_count == 0
