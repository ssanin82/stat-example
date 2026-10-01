"""Time-range ``*_since`` helpers on the ``Storage`` class.

Phase 1 of the long-run observability work (see
``~/.claude/plans/jazzy-crunching-quokka.md``). Each helper must:

* return rows with ``ts* >= since_ts`` in ascending timestamp order
* honour the ``limit`` parameter
* honour the new keyword-only ``until_ts`` (exclusive upper bound)
* filter by ``symbol`` where the table has one (orders,
  quote_decisions, position_snapshots)

The helpers wrap a small amount of SQL; these tests are intentionally
table-level rather than mocking out sqlite, because the whole value
of the helpers is exercising the actual indexes + query shape.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _db() -> tuple[Storage, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_since_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    # Pass DATABASE_URL (not SQLITE_PATH) — Settings.effective_sqlite_path
    # prefers the URL when it starts with ``sqlite:///``; setting only
    # SQLITE_PATH is ignored while the default DATABASE_URL still points
    # at ``./data/mm.db`` (the shared prod path).
    settings = UnitTestSettings.model_validate(
        {"DATABASE_URL": f"sqlite:///{path.as_posix()}"}
    )
    storage = Storage(settings)
    storage.init_schema()
    return storage, path


def _cleanup(storage: Storage, path: Path) -> None:
    storage.close()
    path.unlink(missing_ok=True)


def _ts(hours: int = 0, minutes: int = 0, seconds: int = 0) -> str:
    """Deterministic ISO-8601 timestamp: ``2026-04-18T{HH:MM:SS}+00:00``."""
    return f"2026-04-18T{hours:02d}:{minutes:02d}:{seconds:02d}+00:00"


# --------------------- orders_since ----------------------------------


def _insert_order(storage: Storage, **overrides) -> None:
    row = {
        "order_id_local": overrides.get("order_id_local") or uuid.uuid4().hex[:16],
        "order_id_exchange": None,
        "client_order_id": None,
        "ts_created": overrides["ts_created"],
        "ts_sent": None,
        "ts_ack": None,
        "ts_closed": None,
        "symbol": overrides.get("symbol", "ETH_USDT_Perp"),
        "side": "BUY",
        "price": 1.0,
        "size": 1.0,
        "post_only": 1,
        "status": "ACKED",
        "cancel_reason": None,
        "replace_group_id": None,
        "quote_cycle_id": None,
    }
    storage.insert_order_row(row)


def test_orders_since_returns_rows_after_threshold_in_ts_asc() -> None:
    storage, path = _db()
    try:
        _insert_order(storage, ts_created=_ts(10, 0, 0))
        _insert_order(storage, ts_created=_ts(12, 0, 0))
        _insert_order(storage, ts_created=_ts(14, 0, 0))
        rows = storage.orders_since(since_ts=_ts(11, 0, 0))
        tss = [r["ts_created"] for r in rows]
        assert tss == [_ts(12, 0, 0), _ts(14, 0, 0)]
    finally:
        _cleanup(storage, path)


def test_orders_since_boundary_is_inclusive() -> None:
    storage, path = _db()
    try:
        _insert_order(storage, ts_created=_ts(12, 0, 0))
        rows = storage.orders_since(since_ts=_ts(12, 0, 0))
        assert len(rows) == 1
    finally:
        _cleanup(storage, path)


def test_orders_since_until_ts_is_exclusive_upper_bound() -> None:
    storage, path = _db()
    try:
        _insert_order(storage, ts_created=_ts(10, 0, 0))
        _insert_order(storage, ts_created=_ts(12, 0, 0))
        _insert_order(storage, ts_created=_ts(14, 0, 0))
        rows = storage.orders_since(
            since_ts=_ts(10, 0, 0),
            until_ts=_ts(14, 0, 0),
        )
        tss = [r["ts_created"] for r in rows]
        # 14:00:00 must be excluded (half-open interval).
        assert tss == [_ts(10, 0, 0), _ts(12, 0, 0)]
    finally:
        _cleanup(storage, path)


def test_orders_since_symbol_filter() -> None:
    storage, path = _db()
    try:
        _insert_order(storage, ts_created=_ts(12, 0, 0), symbol="ETH_USDT_Perp")
        _insert_order(storage, ts_created=_ts(12, 0, 30), symbol="AXS_USDT_Perp")
        _insert_order(storage, ts_created=_ts(12, 1, 0), symbol="ETH_USDT_Perp")
        rows = storage.orders_since(
            since_ts=_ts(11, 0, 0),
            symbol="ETH_USDT_Perp",
        )
        assert len(rows) == 2
        assert all(r["symbol"] == "ETH_USDT_Perp" for r in rows)
    finally:
        _cleanup(storage, path)


def test_orders_since_limit_caps_returned_rows() -> None:
    storage, path = _db()
    try:
        for i in range(5):
            _insert_order(storage, ts_created=_ts(12, 0, i))
        rows = storage.orders_since(since_ts=_ts(12, 0, 0), limit=3)
        assert len(rows) == 3
        # Ordered ascending — first three by ts are returned.
        assert [r["ts_created"] for r in rows] == [_ts(12, 0, 0), _ts(12, 0, 1), _ts(12, 0, 2)]
    finally:
        _cleanup(storage, path)


# --------------------- quote_decisions_since -------------------------


def _insert_quote_decision(storage: Storage, **overrides) -> None:
    row = {
        "ts": overrides["ts"],
        "symbol": overrides.get("symbol", "ETH_USDT_Perp"),
        "mid_price": overrides.get("mid_price", 100.0),
        "quote_cycle_id": uuid.uuid4().hex,
    }
    with storage._lock:
        with storage.connection() as conn:
            cols = ", ".join(row.keys())
            placeholders = ", ".join("?" * len(row))
            conn.execute(
                f"INSERT INTO quote_decisions ({cols}) VALUES ({placeholders})",
                tuple(row.values()),
            )


def test_quote_decisions_since_basic_ordering() -> None:
    storage, path = _db()
    try:
        _insert_quote_decision(storage, ts=_ts(14, 0, 2))
        _insert_quote_decision(storage, ts=_ts(14, 0, 0))
        _insert_quote_decision(storage, ts=_ts(14, 0, 1))
        rows = storage.quote_decisions_since(since_ts=_ts(14, 0, 0))
        assert [r["ts"] for r in rows] == [_ts(14, 0, 0), _ts(14, 0, 1), _ts(14, 0, 2)]
    finally:
        _cleanup(storage, path)


def test_quote_decisions_since_window_and_symbol_filter() -> None:
    storage, path = _db()
    try:
        _insert_quote_decision(storage, ts=_ts(14, 0, 0), symbol="ETH_USDT_Perp")
        _insert_quote_decision(storage, ts=_ts(14, 0, 5), symbol="AXS_USDT_Perp")
        _insert_quote_decision(storage, ts=_ts(14, 0, 8), symbol="ETH_USDT_Perp")
        _insert_quote_decision(storage, ts=_ts(14, 1, 0), symbol="ETH_USDT_Perp")
        rows = storage.quote_decisions_since(
            since_ts=_ts(14, 0, 0),
            until_ts=_ts(14, 1, 0),  # excluded
            symbol="ETH_USDT_Perp",
        )
        assert [r["ts"] for r in rows] == [_ts(14, 0, 0), _ts(14, 0, 8)]
    finally:
        _cleanup(storage, path)


# --------------------- position_snapshots_since ----------------------


def _insert_position_snapshot(storage: Storage, **overrides) -> None:
    row = {
        "ts": overrides["ts"],
        "symbol": overrides.get("symbol", "ETH_USDT_Perp"),
        "position_qty": overrides.get("position_qty", 0.0),
        "avg_entry_price": overrides.get("avg_entry_price"),
        "mark_price": overrides.get("mark_price", 100.0),
        "position_notional": overrides.get("position_notional", 0.0),
        "unrealized_pnl_usd": overrides.get("unrealized_pnl_usd", 0.0),
    }
    with storage._lock:
        with storage.connection() as conn:
            cols = ", ".join(row.keys())
            placeholders = ", ".join("?" * len(row))
            conn.execute(
                f"INSERT INTO position_snapshots ({cols}) VALUES ({placeholders})",
                tuple(row.values()),
            )


def test_position_snapshots_since_basic() -> None:
    storage, path = _db()
    try:
        _insert_position_snapshot(storage, ts=_ts(15, 0, 0), position_qty=10.0)
        _insert_position_snapshot(storage, ts=_ts(15, 1, 0), position_qty=12.0)
        _insert_position_snapshot(storage, ts=_ts(15, 2, 0), position_qty=8.0)
        rows = storage.position_snapshots_since(since_ts=_ts(15, 0, 30))
        assert [r["position_qty"] for r in rows] == [12.0, 8.0]
    finally:
        _cleanup(storage, path)


def test_position_snapshots_since_symbol_filter() -> None:
    storage, path = _db()
    try:
        _insert_position_snapshot(storage, ts=_ts(15, 0, 0), symbol="ETH_USDT_Perp", position_qty=1.0)
        _insert_position_snapshot(storage, ts=_ts(15, 0, 1), symbol="AXS_USDT_Perp", position_qty=2.0)
        _insert_position_snapshot(storage, ts=_ts(15, 0, 2), symbol="ETH_USDT_Perp", position_qty=3.0)
        rows = storage.position_snapshots_since(
            since_ts=_ts(15, 0, 0),
            symbol="AXS_USDT_Perp",
        )
        assert len(rows) == 1
        assert rows[0]["position_qty"] == 2.0
    finally:
        _cleanup(storage, path)


# --------------------- extensions to existing helpers ----------------


def test_fills_since_accepts_until_ts_without_breaking_positional_contract() -> None:
    """``fills_since`` historically has positional ``(since_ts, limit)``.
    Keyword-only ``until_ts`` must not break the existing two-arg calls
    in ``api.py`` (e.g. ``/pnl/attribution``)."""
    storage, path = _db()
    try:
        # Insert three fills via the public API — last stamp crosses
        # the minute boundary so the ``until=16:01:00`` window excludes
        # the third row.
        stamps = [_ts(16, 0, 0), _ts(16, 0, 30), _ts(16, 1, 0)]
        for stamp in stamps:
            storage.insert_fill_row(
                {
                    "fill_id": uuid.uuid4().hex,
                    "order_id_exchange": None,
                    "client_order_id": None,
                    "ts_fill": stamp,
                    "symbol": "ETH_USDT_Perp",
                    "side": "BUY",
                    "price": 1.0,
                    "size": 1.0,
                    "notional": 1.0,
                    "fee": 0.0,
                    "liquidity_flag": "resting",
                    "mid_at_fill": 1.0,
                    "best_bid_at_fill": 0.999,
                    "best_ask_at_fill": 1.001,
                    "book_snapshot_quality": "full",
                    "markout_1s_bps": 0.0,
                    "markout_3s_bps": 0.0,
                    "markout_5s_bps": 0.0,
                    "book_reference_quality": "exact_or_prior",
                }
            )
        # Positional call (existing api.py contract).
        rows_all = storage.fills_since(_ts(16, 0, 0), 10000)
        assert len(rows_all) == 3
        # Keyword-only until_ts (new).
        rows_window = storage.fills_since(
            _ts(16, 0, 0), 10000, until_ts=_ts(16, 1, 0)
        )
        assert len(rows_window) == 2
    finally:
        _cleanup(storage, path)


def test_bot_events_since_until_ts_excludes_upper() -> None:
    storage, path = _db()
    try:
        storage.insert_bot_event(_ts(17, 0, 0), "INFO", "e1", "", None)
        storage.insert_bot_event(_ts(17, 0, 5), "INFO", "e2", "", None)
        storage.insert_bot_event(_ts(17, 0, 10), "INFO", "e3", "", None)
        rows = storage.bot_events_since(
            since_ts=_ts(17, 0, 0),
            until_ts=_ts(17, 0, 10),
        )
        assert [r["event_type"] for r in rows] == ["e1", "e2"]
    finally:
        _cleanup(storage, path)


def test_equity_history_since_until_ts_excludes_upper() -> None:
    storage, path = _db()
    try:
        # Use valid seconds (0, 30, plus a second-level timestamp in the
        # next minute) — ``_ts(18, 0, 60)`` would be string-compared as
        # less than "18:01:00" and slip past the until bound.
        stamps = [_ts(18, 0, 0), _ts(18, 0, 30), _ts(18, 1, 0)]
        for stamp in stamps:
            with storage._lock:
                with storage.connection() as conn:
                    conn.execute(
                        "INSERT INTO equity_snapshots (ts, equity_usd, cash_usd, realized_pnl_usd, "
                        "unrealized_pnl_usd, fees_usd, drawdown_usd) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (stamp, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0),
                    )
        rows = storage.equity_history_since(
            since_ts=_ts(18, 0, 0),
            until_ts=_ts(18, 1, 0),
        )
        # 18:01:00 excluded; 18:00:00 and 18:00:30 included.
        assert len(rows) == 2
    finally:
        _cleanup(storage, path)


# --------------------- schema v12 index sanity -----------------------


def test_v12_adds_timestamp_indexes() -> None:
    """Freshly-initialised DB at v12 has indexes on the tables that
    previously lacked them."""
    storage, path = _db()
    try:
        with storage._lock:
            with storage.connection() as conn:
                rows = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
                names = {r[0] for r in rows}
        assert "idx_orders_ts_created" in names
        assert "idx_positions_ts" in names
        assert "idx_equity_ts" in names
        # Pre-existing indexes must also still be present.
        assert "idx_fills_ts" in names
        assert "idx_quotes_ts" in names
        assert "idx_events_ts" in names
    finally:
        _cleanup(storage, path)
