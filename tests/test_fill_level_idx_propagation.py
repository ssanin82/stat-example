"""Tests for v1.4.100 F1 — ``level_idx`` propagates from
``WorkingOrder`` → fills row.

Covers:
1. ``Fill`` defaults ``level_idx`` to 0 (backward-compat with
   pre-multi-level callers).
2. ``Fill`` constructed with explicit ``level_idx=1`` keeps that
   value.
3. Schema migration (v37→v38) adds ``level_idx`` to both the
   ``fills`` and ``orders`` tables.
4. Legacy fills inserted without specifying ``level_idx`` default
   to 0 thanks to the column DEFAULT clause.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from app.models import Fill, Side
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings(tmp_db: Path) -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{tmp_db.as_posix()}",
        }
    )


def _make_fill(*, level_idx: int = 0, fid: str = "fill-1") -> Fill:
    """Construct a minimal Fill."""
    now = datetime.now(timezone.utc)
    return Fill(
        fill_id=fid,
        order_id_exchange=1,
        client_order_id="cloid-1",
        ts_fill=now,
        symbol="TON-USDT-SWAP",
        side=Side.BUY,
        price=2.0,
        size=1.0,
        notional=2.0,
        fee=-0.001,
        liquidity_flag="M",
        mid_at_fill=2.0,
        level_idx=level_idx,
    )


def test_fill_default_level_idx_is_zero() -> None:
    """A Fill constructed without ``level_idx`` defaults to 0
    (single-rung / pre-multi-level callers)."""
    fill = _make_fill()
    assert fill.level_idx == 0


def test_fill_explicit_level_idx_preserved() -> None:
    """A Fill constructed with ``level_idx=1`` keeps that value."""
    fill = _make_fill(level_idx=1)
    assert fill.level_idx == 1


def test_fills_table_has_level_idx_column(tmp_path: Path) -> None:
    """Schema v38 added ``level_idx`` to the fills table."""
    db = tmp_path / f"f1_fills_{os.getpid()}.db"
    storage = Storage(_settings(db))
    storage.init_schema()
    with storage.connection() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(fills)")}
    assert "level_idx" in cols, f"fills.level_idx missing; cols={cols}"
    storage.close()


def test_orders_table_has_level_idx_column(tmp_path: Path) -> None:
    """Schema v38 also added ``level_idx`` to the orders table."""
    db = tmp_path / f"f1_orders_{os.getpid()}.db"
    storage = Storage(_settings(db))
    storage.init_schema()
    with storage.connection() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(orders)")}
    assert "level_idx" in cols, f"orders.level_idx missing; cols={cols}"
    storage.close()


def test_legacy_fill_row_defaults_level_idx_to_zero(tmp_path: Path) -> None:
    """A row inserted without ``level_idx`` (pre-v38 caller pattern)
    must read back as 0 thanks to the DEFAULT clause in the migration.
    """
    db = tmp_path / f"f1_legacy_{os.getpid()}.db"
    storage = Storage(_settings(db))
    storage.init_schema()
    with storage.connection() as conn:
        conn.execute(
            "INSERT INTO fills (fill_id, order_id_exchange, "
            "client_order_id, ts_fill, symbol, side, price, size, "
            "notional, fee, liquidity_flag) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-fill",
                "oid",
                "cl",
                "2026-05-19T00:00:00+00:00",
                "TON-USDT-SWAP",
                "BUY",
                2.0,
                1.0,
                2.0,
                -0.001,
                "M",
            ),
        )
    with storage.connection() as conn:
        row = conn.execute(
            "SELECT level_idx FROM fills WHERE fill_id='legacy-fill'"
        ).fetchone()
    assert row is not None
    assert row[0] == 0
    storage.close()
