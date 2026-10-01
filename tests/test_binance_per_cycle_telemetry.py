"""Per-cycle Binance L1 reference on ``quote_decisions`` (schema v14).

Closes the last Gap-3 audit item: until this ships, the only DB trace
of cross-venue state was the ``binance_cross_venue_cancel`` event —
written ONLY when the cancel-on-move threshold is breached. Every
quote cycle in between (hundreds per minute) had no cross-venue
context. Now:

  * ``QuoteDecision.binance_mid`` — Binance mid snapped under the
    ``BotState`` lock at decision time (None if feed hasn't delivered
    its first message or ``BINANCE_WS_ENABLED=false``).
  * ``QuoteDecision.binance_basis_ewma`` — basis EWMA at same instant.
  * Both persisted on the ``quote_decisions`` row, flow through
    ``/quotes/recent`` and ``/quotes/since`` automatically (``SELECT *``
    readers + dynamic-column writer), captured by
    ``scripts/stats_snapshot.py``.

Invariants pinned here:

  1. Schema v14 adds exactly two REAL columns to ``quote_decisions``;
     an existing v13 DB (with historical rows) upgrades cleanly and
     the prior rows remain queryable with NULLs in the new columns.
  2. Idempotent on re-migration (running init_schema twice doesn't
     raise "duplicate column" from the safety wrapper).
  3. ``QuoteDecision`` default-constructs with both fields == None
     (clean shape-stable contract).
  4. ``quote_decision_row`` includes the two new keys so
     ``insert_quote_decision`` writes them.
  5. End-to-end: when the bot's one_tick path reads ``state.binance_*``
     under lock, those values land on the decision and on the DB row.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from app.bot import quote_decision_row
from app.enums import ActiveSides
from app.models import QuoteDecision
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.settings_helpers import UnitTestSettings


# --------------------- Schema migration ------------------------------


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"mm_v14_{os.getpid()}_{uuid.uuid4().hex}.db"


def _settings_for(path: Path) -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "PRIVATE_WS_ENABLED": False,
        }
    )


def _quote_decisions_columns(path: Path) -> set[str]:
    conn = sqlite3.connect(path)
    try:
        return {row[1] for row in conn.execute("PRAGMA table_info(quote_decisions)")}
    finally:
        conn.close()


def test_fresh_schema_includes_v14_binance_columns() -> None:
    """A brand-new DB initialised by ``Storage.init_schema()`` must end
    up at ``user_version=14`` with ``binance_mid`` and
    ``binance_basis_ewma`` columns present on ``quote_decisions``."""
    path = _db_path()
    path.unlink(missing_ok=True)
    try:
        storage = Storage(_settings_for(path))
        storage.init_schema()

        conn = sqlite3.connect(path)
        try:
            user_v = conn.execute("PRAGMA user_version").fetchone()[0]
            assert user_v == Storage.SCHEMA_VERSION, user_v
            assert user_v >= 14, f"schema version must be >= 14, got {user_v}"
        finally:
            conn.close()

        cols = _quote_decisions_columns(path)
        assert "binance_mid" in cols, cols
        assert "binance_basis_ewma" in cols, cols
        # v13 (microprice) still present — migrations are cumulative.
        assert "microprice" in cols, cols
    finally:
        path.unlink(missing_ok=True)


def test_v13_database_migrates_to_v14_and_preserves_data() -> None:
    """Simulate a DB left at ``user_version=13`` from a prior bot run
    (with real quote_decisions rows). On next bot start, init_schema
    must add the two new columns without touching existing rows."""
    path = _db_path()
    path.unlink(missing_ok=True)
    try:
        # Step 1: build a v13 DB by running init_schema, then rolling
        # ``user_version`` back to 13 and dropping the v14 columns.
        # This faithfully mimics an existing Railway-volume DB.
        storage = Storage(_settings_for(path))
        storage.init_schema()

        conn = sqlite3.connect(path)
        try:
            # SQLite has no "DROP COLUMN" prior to 3.35 — use a
            # rebuild that lists the exact v13 columns explicitly.
            # (We re-check by PRAGMA user_version after.)
            conn.execute("PRAGMA foreign_keys=OFF")
            conn.execute(
                """
                CREATE TABLE quote_decisions_v13 AS
                SELECT
                    ts, symbol, mid_price, vol_estimate, inventory,
                    reservation_price, target_spread_bps,
                    target_bid, target_ask, quoted_bid, quoted_ask,
                    quoted_bid_sz, quoted_ask_sz, active_sides,
                    toxicity_score, decision_reason, quote_cycle_id,
                    spread_floor_overlay_half_spread_bps, microprice
                FROM quote_decisions
                """
            )
            conn.execute("DROP TABLE quote_decisions")
            conn.execute("ALTER TABLE quote_decisions_v13 RENAME TO quote_decisions")
            conn.execute("PRAGMA user_version = 13")
            # Seed a historical row so we can prove it survives.
            conn.execute(
                """
                INSERT INTO quote_decisions
                  (ts, symbol, mid_price, quote_cycle_id, microprice)
                VALUES (?, 'ETH', 100.0, 'hist-v13', 100.05)
                """,
                (utc_now().isoformat(),),
            )
            conn.commit()
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 13
        finally:
            conn.close()

        # Sanity check — before migration, the new columns must NOT exist.
        cols_before = _quote_decisions_columns(path)
        assert "binance_mid" not in cols_before
        assert "binance_basis_ewma" not in cols_before
        assert "microprice" in cols_before  # v13 column was preserved

        # Step 2: a fresh Storage on the same file runs init_schema
        # again. The ``from_v<14`` block must kick in.
        storage2 = Storage(_settings_for(path))
        storage2.init_schema()

        cols_after = _quote_decisions_columns(path)
        assert "binance_mid" in cols_after, cols_after
        assert "binance_basis_ewma" in cols_after, cols_after

        conn = sqlite3.connect(path)
        try:
            user_v = conn.execute("PRAGMA user_version").fetchone()[0]
            assert user_v == Storage.SCHEMA_VERSION

            # Historical v13 row still readable, new columns NULL.
            row = conn.execute(
                "SELECT mid_price, microprice, binance_mid, binance_basis_ewma "
                "FROM quote_decisions WHERE quote_cycle_id='hist-v13'"
            ).fetchone()
            assert row is not None, "historical v13 row was lost during migration"
            assert row[0] == 100.0
            assert row[1] == 100.05
            assert row[2] is None, "binance_mid must be NULL for pre-v14 rows"
            assert row[3] is None, "binance_basis_ewma must be NULL for pre-v14 rows"
        finally:
            conn.close()
    finally:
        path.unlink(missing_ok=True)


def test_init_schema_is_idempotent_at_v14() -> None:
    """Running init_schema twice on the same file must not raise — the
    ``duplicate column`` swallow in the v14 migration guarantees this."""
    path = _db_path()
    path.unlink(missing_ok=True)
    try:
        storage = Storage(_settings_for(path))
        storage.init_schema()
        # Second call is the real test — if the ALTER block doesn't
        # swallow duplicate-column OperationalError we'd see it here.
        storage.init_schema()
        cols = _quote_decisions_columns(path)
        assert "binance_mid" in cols
        assert "binance_basis_ewma" in cols
    finally:
        path.unlink(missing_ok=True)


# --------------------- QuoteDecision dataclass + row ---------------


def _qd(**overrides) -> QuoteDecision:
    """Minimal QuoteDecision with required fields, for row-serialisation tests."""
    base = dict(
        ts=utc_now(),
        symbol="ETH",
        mid_price=100.0,
        vol_estimate=5.0,
        inventory=0.0,
        reservation_price=100.0,
        target_spread_bps=2.0,
        target_bid=99.99,
        target_ask=100.01,
        quoted_bid=99.99,
        quoted_ask=100.01,
        quoted_bid_sz=0.1,
        quoted_ask_sz=0.1,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="baseline",
        quote_cycle_id="test-cycle",
    )
    base.update(overrides)
    return QuoteDecision(**base)


def test_quote_decision_binance_fields_default_to_none() -> None:
    """Default construction — Binance fields are None. Shape contract
    for downstream tooling (``explain_moment.py``, stats_snapshot)."""
    d = _qd()
    assert d.binance_mid is None
    assert d.binance_basis_ewma is None


def test_quote_decision_row_includes_binance_keys_when_set() -> None:
    """When QuoteDecision carries concrete Binance values, they must
    land in the row dict under their canonical keys — otherwise the
    insert_quote_decision SQL drops them."""
    d = _qd(binance_mid=2340.10, binance_basis_ewma=0.05)
    row = quote_decision_row(d, "qc-001")
    assert row["binance_mid"] == pytest.approx(2340.10)
    assert row["binance_basis_ewma"] == pytest.approx(0.05)


def test_quote_decision_row_includes_binance_keys_as_none_when_unset() -> None:
    """Fields must be present in the row even when value is None —
    otherwise a cycle that fires before the Binance stream warms up
    would produce a row missing these columns entirely (confusing
    downstream JSON dumps / analysis queries)."""
    d = _qd()  # Binance fields default to None
    row = quote_decision_row(d, "qc-002")
    assert "binance_mid" in row
    assert "binance_basis_ewma" in row
    assert row["binance_mid"] is None
    assert row["binance_basis_ewma"] is None


# --------------------- End-to-end DB persistence -------------------


def test_insert_quote_decision_round_trips_binance_fields() -> None:
    """Insert a row with Binance fields populated; read it back via
    ``recent_quote_decisions``. Values must survive the round trip
    through SQLite (REAL column → float)."""
    path = _db_path()
    path.unlink(missing_ok=True)
    try:
        storage = Storage(_settings_for(path))
        storage.init_schema()

        d = _qd(binance_mid=2340.10, binance_basis_ewma=0.05)
        row = quote_decision_row(d, "qc-rt")
        storage.insert_quote_decision(row)

        rows = storage.recent_quote_decisions(limit=5)
        hits = [r for r in rows if r["quote_cycle_id"] == "qc-rt"]
        assert len(hits) == 1
        assert hits[0]["binance_mid"] == pytest.approx(2340.10)
        assert hits[0]["binance_basis_ewma"] == pytest.approx(0.05)
    finally:
        path.unlink(missing_ok=True)


def test_insert_quote_decision_stores_null_when_binance_off() -> None:
    """With BINANCE_WS_ENABLED=false, state has None for both fields.
    The row must persist NULLs and read back as None — NOT 0.0 or
    any other sentinel that would corrupt downstream analysis."""
    path = _db_path()
    path.unlink(missing_ok=True)
    try:
        storage = Storage(_settings_for(path))
        storage.init_schema()

        # Simulate Binance-disabled run: state.binance_mid is None.
        state = BotState(_settings_for(path))
        assert state.binance_mid is None
        assert state.binance_basis_ewma is None

        d = _qd()
        d = replace(
            d,
            binance_mid=state.binance_mid,
            binance_basis_ewma=state.binance_basis_ewma,
        )
        row = quote_decision_row(d, "qc-off")
        storage.insert_quote_decision(row)

        rows = storage.recent_quote_decisions(limit=5)
        hits = [r for r in rows if r["quote_cycle_id"] == "qc-off"]
        assert len(hits) == 1
        assert hits[0]["binance_mid"] is None
        assert hits[0]["binance_basis_ewma"] is None
    finally:
        path.unlink(missing_ok=True)


def test_replace_attaches_snapshot_from_state_under_lock() -> None:
    """Mirror of the ``app/bot.py`` one_tick snippet: read Binance state
    under lock, attach via ``replace()``. This pins the call pattern
    so a future refactor doesn't break the "read-under-lock" contract
    that the bot relies on for thread-safety against the
    BinancePublicStream writer thread."""
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "BINANCE_WS_ENABLED": True,
            "PRIVATE_WS_ENABLED": False,
        }
    )
    state = BotState(settings)
    # Simulate BinancePublicStream having written some state.
    state.binance_mid = 2340.10
    state.binance_basis_ewma = 0.05

    d = _qd()
    # Exactly the call shape used in app/bot.py one_tick.
    with state._lock:
        bmid = state.binance_mid
        bbas = state.binance_basis_ewma
    d2 = replace(d, binance_mid=bmid, binance_basis_ewma=bbas)
    assert d2.binance_mid == pytest.approx(2340.10)
    assert d2.binance_basis_ewma == pytest.approx(0.05)
    # Original unchanged (replace returns a new instance).
    assert d.binance_mid is None
    assert d.binance_basis_ewma is None
