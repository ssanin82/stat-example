"""Tests for todo-027 Tier 2 backend instrumentation.

The Tier 2 work adds per-bar + per-fill snapshots for three gates
that previously only exposed a session-cumulative ``fire_count``
via ``live_stats``:

- ``vol_trend_gate``    — bar flag + ``*_at_decision`` on orders/fills
- ``post_swing``        — bar flag + ``*_at_decision`` on orders/fills
- ``session_drawdown``  — bar tier label + ``*_at_decision`` on
                          orders/fills (string because the ladder has
                          6 tiers, not a binary active flag)

Coverage:

1. v32 migration applies the 9 new columns (3 tables × 3 fields).
2. ``_DECISION_STATE_COLS`` includes the three new propagation fields.
3. ``WorkingOrder`` and ``Fill`` dataclasses carry the fields.
4. ``order_row`` serialises the fields to the DB-row dict.
5. ``ExposureBarEmitter`` reads gate state into the bar dict.
"""

from __future__ import annotations

import os
import tempfile
import time as _time
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.execution import order_row
from app.models import Fill, WorkingOrder
from app.enums import OrderStatus, Side
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _db() -> tuple[Storage, Path]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_t27t2_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {"DATABASE_URL": f"sqlite:///{path.as_posix()}"}
    )
    storage = Storage(settings)
    storage.init_schema()
    return storage, path


def _cleanup(storage: Storage, path: Path) -> None:
    storage.close()
    path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# 1. v32 migration adds the 9 new columns
# ----------------------------------------------------------------------


def test_migration_v32_adds_exposure_bars_columns() -> None:
    """All three new columns on exposure_bars must round-trip cleanly,
    including the TEXT ``session_drawdown_tier``."""
    storage, path = _db()
    try:
        storage.insert_exposure_bar(
            {
                "session_id": "test-session",
                "ts_bar": "2026-05-14T12:00:00+00:00",
                "symbol": "TEST",
                "mid": 1.0,
                "vol_trend_active": 1,
                "post_swing_active": 0,
                "session_drawdown_tier": "WIDEN",
            }
        )
        rows = storage.exposure_bars_since("2026-05-14T00:00:00+00:00")
        assert len(rows) == 1
        assert rows[0]["vol_trend_active"] == 1
        assert rows[0]["post_swing_active"] == 0
        assert rows[0]["session_drawdown_tier"] == "WIDEN"
    finally:
        _cleanup(storage, path)


def test_migration_v32_adds_orders_at_decision_columns() -> None:
    """Three new ``*_at_decision`` columns on orders. Inserting a
    row that references them must succeed."""
    storage, path = _db()
    try:
        wo = WorkingOrder(
            order_id_local="o1",
            order_id_exchange=None,
            client_order_id=None,
            symbol="TEST",
            side=Side.BUY,
            price=1.0,
            size=1.0,
            post_only=True,
            status=OrderStatus.NEW_LOCAL,
            vol_trend_active_at_decision=True,
            post_swing_active_at_decision=False,
            session_drawdown_tier_at_decision="PAUSE_SHORT",
        )
        storage.insert_order_row(order_row(wo))
        decision = storage.order_decision_state_for_order(None)
        # No order_id_exchange means decision lookup returns empty —
        # but storage didn't raise. Use a direct fetch instead:
        with storage._lock:
            with storage.connection() as conn:
                row = conn.execute(
                    "SELECT vol_trend_active_at_decision, "
                    "post_swing_active_at_decision, "
                    "session_drawdown_tier_at_decision "
                    "FROM orders WHERE order_id_local = ?",
                    ("o1",),
                ).fetchone()
        assert row is not None
        assert row[0] == 1
        assert row[1] == 0
        assert row[2] == "PAUSE_SHORT"
    finally:
        _cleanup(storage, path)


def test_migration_v32_adds_fills_at_decision_columns() -> None:
    """Three new ``*_at_decision`` columns on fills."""
    storage, path = _db()
    try:
        with storage._lock:
            with storage.connection() as conn:
                conn.execute(
                    "INSERT INTO fills (fill_id, ts_fill, symbol, side, "
                    "price, size, notional, fee, liquidity_flag, mid_at_fill, "
                    "vol_trend_active_at_decision, "
                    "post_swing_active_at_decision, "
                    "session_drawdown_tier_at_decision) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "f1",
                        "2026-05-14T12:00:00+00:00",
                        "TEST",
                        "BUY",
                        1.0,
                        1.0,
                        1.0,
                        0.0,
                        "MAKER",
                        1.0,
                        1,
                        0,
                        "CLEAR",
                    ),
                )
        rows = storage.fills_since("2026-05-14T00:00:00+00:00")
        assert len(rows) == 1
        assert rows[0]["vol_trend_active_at_decision"] == 1
        assert rows[0]["post_swing_active_at_decision"] == 0
        assert rows[0]["session_drawdown_tier_at_decision"] == "CLEAR"
    finally:
        _cleanup(storage, path)


# ----------------------------------------------------------------------
# 2. _DECISION_STATE_COLS propagation tuple
# ----------------------------------------------------------------------


def test_decision_state_cols_includes_three_new_propagation_fields() -> None:
    """Fill ingestion uses ``_DECISION_STATE_COLS`` to look up parent-
    order decision state at fill time. The three new tier-2 fields
    must be in the tuple so they propagate to fills."""
    assert (
        "vol_trend_active_at_decision" in Storage._DECISION_STATE_COLS
    )
    assert (
        "post_swing_active_at_decision" in Storage._DECISION_STATE_COLS
    )
    assert (
        "session_drawdown_tier_at_decision"
        in Storage._DECISION_STATE_COLS
    )


def test_decision_state_propagates_from_order_to_lookup() -> None:
    """End-to-end: insert an order with the three new fields stamped,
    look it up via order_decision_state_for_order, confirm the values
    round-trip."""
    storage, path = _db()
    try:
        wo = WorkingOrder(
            order_id_local="o2",
            order_id_exchange=12345,
            client_order_id=None,
            symbol="TEST",
            side=Side.SELL,
            price=2.0,
            size=1.0,
            post_only=True,
            status=OrderStatus.ACKED,
            vol_trend_active_at_decision=True,
            post_swing_active_at_decision=False,
            session_drawdown_tier_at_decision="RESUME_TESTING",
        )
        storage.insert_order_row(order_row(wo))
        decision = storage.order_decision_state_for_order("12345")
        assert decision["vol_trend_active_at_decision"] == 1
        assert decision["post_swing_active_at_decision"] == 0
        assert (
            decision["session_drawdown_tier_at_decision"]
            == "RESUME_TESTING"
        )
    finally:
        _cleanup(storage, path)


# ----------------------------------------------------------------------
# 3. WorkingOrder + Fill dataclass fields
# ----------------------------------------------------------------------


def test_working_order_has_tier2_fields_default_none() -> None:
    wo = WorkingOrder(
        order_id_local="o3",
        order_id_exchange=None,
        client_order_id=None,
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.NEW_LOCAL,
    )
    assert wo.vol_trend_active_at_decision is None
    assert wo.post_swing_active_at_decision is None
    assert wo.session_drawdown_tier_at_decision is None


def test_fill_has_tier2_fields_default_none() -> None:
    from datetime import datetime, timezone

    f = Fill(
        fill_id="f3",
        order_id_exchange=None,
        client_order_id=None,
        ts_fill=datetime(2026, 5, 14, tzinfo=timezone.utc),
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=0.0,
        liquidity_flag="MAKER",
        mid_at_fill=1.0,
    )
    assert f.vol_trend_active_at_decision is None
    assert f.post_swing_active_at_decision is None
    assert f.session_drawdown_tier_at_decision is None


# ----------------------------------------------------------------------
# 4. order_row serialisation
# ----------------------------------------------------------------------


def test_order_row_serialises_tier2_bools_as_int_and_tier_as_str() -> None:
    """SQLite stores INTEGER for bools (1/0/NULL); the string tier is
    stored as TEXT verbatim."""
    wo = WorkingOrder(
        order_id_local="o4",
        order_id_exchange=None,
        client_order_id=None,
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.NEW_LOCAL,
        vol_trend_active_at_decision=True,
        post_swing_active_at_decision=False,
        session_drawdown_tier_at_decision="WIDEN",
    )
    row = order_row(wo)
    assert row["vol_trend_active_at_decision"] == 1
    assert row["post_swing_active_at_decision"] == 0
    assert row["session_drawdown_tier_at_decision"] == "WIDEN"


def test_order_row_handles_none_tier2_fields() -> None:
    wo = WorkingOrder(
        order_id_local="o5",
        order_id_exchange=None,
        client_order_id=None,
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.NEW_LOCAL,
        # all three Tier 2 fields default to None
    )
    row = order_row(wo)
    assert row["vol_trend_active_at_decision"] is None
    assert row["post_swing_active_at_decision"] is None
    assert row["session_drawdown_tier_at_decision"] is None


# ----------------------------------------------------------------------
# 5. ExposureBarEmitter reads gate state
# ----------------------------------------------------------------------


class _FakeGate:
    """Minimal gate stub with the attributes the emitter reads."""

    def __init__(self, cooldown_until_mono: float = 0.0) -> None:
        self.cooldown_until_mono = cooldown_until_mono


class _FakeDrawdown:
    """Minimal session-drawdown stub with a tier enum-like value."""

    def __init__(self, tier_value: str = "CLEAR") -> None:
        self.tier = _TierStub(tier_value)


class _TierStub:
    """Enum-like .value access used by the emitter."""

    def __init__(self, value: str) -> None:
        self.value = value


def test_exposure_bar_emitter_captures_three_tier2_fields() -> None:
    """Direct test of the emitter's bar dict builder. Stub a minimal
    state with the three gate objects active and assert the bar dict
    has the right values.

    We don't need a full BotState — only the attributes the emitter
    actually touches. ExposureBarEmitter is forgiving by design.
    """
    from app.exposure_bar_emitter import ExposureBarEmitter

    # Stub state with gate objects active.
    class _MinimalMarket:
        mid_price = 1.0
        best_bid = 0.99
        best_ask = 1.01
        bid_size = 100.0
        ask_size = 100.0

    class _MinimalPosition:
        position_qty = 0.0

    class _MinimalState:
        session_id = "test-session"
        market = _MinimalMarket()
        position = _MinimalPosition()
        # Active gates: vol_trend deadline in the future.
        vol_trend_gate = _FakeGate(
            cooldown_until_mono=_time.monotonic() + 60.0
        )
        # Inactive gate: post_swing deadline in the past.
        post_swing = _FakeGate(cooldown_until_mono=0.0)
        # Tier value: WIDEN.
        session_drawdown = _FakeDrawdown(tier_value="WIDEN")
        # Other attributes the emitter pokes (None / safe defaults).
        toxicity = None
        vol_bps = None
        last_active_sides = None
        binance_basis_ewma = None
        basis_regime = None
        working_bid = None
        working_ask = None
        adaptive_spread_widen_until_mono = 0.0
        quote_elig_recovery_remaining_ms = None
        observability_gate_flags = {}
        quote_eligibility_snapshot_dict = {}

    settings = UnitTestSettings.model_validate({})
    emitter = ExposureBarEmitter(settings, _MinimalState(), storage=None)  # type: ignore[arg-type]
    bar = emitter._capture_bar()
    assert bar is not None
    assert bar["vol_trend_active"] == 1
    assert bar["post_swing_active"] == 0
    assert bar["session_drawdown_tier"] == "WIDEN"
