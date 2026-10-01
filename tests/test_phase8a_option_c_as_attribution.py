"""Phase 8A Option C (v1.5.190) — per-fill Avellaneda-Stoikov attribution.

Tests cover the same three layers as the Phase 3F reservation-alpha
plumbing tests (`tests/test_phase3f_reservation_alpha_attribution.py`):

1. **Schema** — migration v44 adds the
   ``as_base_half_spread_bps_at_decision`` column to both ``orders``
   and ``fills``, and the user_version is bumped accordingly.

2. **Decision-state contract** — the column is present in
   :data:`Storage._DECISION_STATE_COLS` so the
   ``order_metadata_for_fill_ingest`` lookup propagates the value from
   the parent order row onto the resulting fill at ingest time.

3. **Row-mapper stamping** — ``order_row`` / ``fill_row`` serialise
   the new field; a parent order persisted with a non-NULL AS-base is
   readable via the metadata lookup; a fill constructed from that
   metadata carries the same value.

No new postmortem section ships with Option C — the field is
recorded for offline analysis. The pure AS-formula math is covered
by ``tests/test_avellaneda_stoikov.py``.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.enums import OrderStatus, Side
from app.execution import order_row
from app.fill_ingestion import fill_row
from app.models import Fill, WorkingOrder
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_8a_optc_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "LOGS_BUCKET": "fake-bucket",
            "LIVE_STATS_ENABLED": True,
        }
    )


# ---------------------------------------------------------------------------
# Schema migration v44
# ---------------------------------------------------------------------------


def test_schema_v44_adds_as_base_column_to_orders_and_fills() -> None:
    s = _settings()
    Storage(s).init_schema()
    conn = sqlite3.connect(s.database_url.split("sqlite:///", 1)[-1])
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v >= 44
        for tbl in ("orders", "fills"):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({tbl})")}
            assert "as_base_half_spread_bps_at_decision" in cols, (
                f"missing as_base_half_spread_bps_at_decision in {tbl}"
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Decision-state propagation contract
# ---------------------------------------------------------------------------


def test_decision_state_cols_includes_as_base() -> None:
    """The new column MUST be present in ``_DECISION_STATE_COLS`` or
    ``order_metadata_for_fill_ingest`` won't SELECT it, and fill
    ingest won't propagate the value from parent order → fill."""
    cols = set(Storage._DECISION_STATE_COLS)
    assert "as_base_half_spread_bps_at_decision" in cols


# ---------------------------------------------------------------------------
# Row-mapper contracts
# ---------------------------------------------------------------------------


def test_order_row_includes_as_base_field() -> None:
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=None,
        client_order_id="c",
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.SENT,
        as_base_half_spread_bps_at_decision=4.86,
    )
    row = order_row(wo)
    assert row["as_base_half_spread_bps_at_decision"] == 4.86


def test_order_row_as_base_is_none_when_unset() -> None:
    """Legacy / SF / manual paths construct WO without a breakdown —
    the field must serialise as None (NULL in SQLite)."""
    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=None,
        client_order_id="c",
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.SENT,
    )
    row = order_row(wo)
    assert row["as_base_half_spread_bps_at_decision"] is None


def test_fill_row_includes_as_base_field() -> None:
    f = Fill(
        fill_id="x",
        order_id_exchange="0",
        client_order_id=None,
        ts_fill=datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc),
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=-0.0001,
        liquidity_flag="resting",
        mid_at_fill=1.0,
        as_base_half_spread_bps_at_decision=3.21,
    )
    r = fill_row(f)
    assert r["as_base_half_spread_bps_at_decision"] == 3.21


def test_fill_row_as_base_is_none_when_unset() -> None:
    f = Fill(
        fill_id="x",
        order_id_exchange="0",
        client_order_id=None,
        ts_fill=datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc),
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=-0.0001,
        liquidity_flag="resting",
        mid_at_fill=1.0,
    )
    r = fill_row(f)
    assert r["as_base_half_spread_bps_at_decision"] is None


# ---------------------------------------------------------------------------
# End-to-end round-trip via storage
# ---------------------------------------------------------------------------


def test_order_metadata_lookup_returns_as_base_field() -> None:
    """Persist an order with a non-NULL AS-base, then read it back
    via the same lookup ``fill_ingestion`` uses at fill ingest time.
    Verifies the SELECT in ``order_metadata_for_fill_ingest`` picks
    up the v44 column."""
    s = _settings()
    st = Storage(s)
    st.init_schema()
    wo = WorkingOrder(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=1234567890,
        client_order_id="cloid-1",
        symbol="TEST",
        side=Side.BUY,
        price=2.030,
        size=2.0,
        post_only=True,
        status=OrderStatus.SENT,
        ts_created=datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc),
        as_base_half_spread_bps_at_decision=5.42,
    )
    st.insert_order_row(order_row(wo))
    meta = st.order_metadata_for_fill_ingest("1234567890")
    assert "as_base_half_spread_bps_at_decision" in meta
    assert meta["as_base_half_spread_bps_at_decision"] == 5.42


def test_order_metadata_lookup_as_base_is_none_on_legacy_row() -> None:
    """A WO with no AS-base set persists as NULL, and the lookup
    returns None (not raises) — same contract as the other
    *_at_decision fields for pre-v1.5.190 rows."""
    s = _settings()
    st = Storage(s)
    st.init_schema()
    wo = WorkingOrder(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=999,
        client_order_id="cloid-legacy",
        symbol="TEST",
        side=Side.SELL,
        price=2.035,
        size=2.0,
        post_only=True,
        status=OrderStatus.SENT,
        ts_created=datetime(2026, 5, 27, 12, 0, 0, tzinfo=timezone.utc),
    )
    st.insert_order_row(order_row(wo))
    meta = st.order_metadata_for_fill_ingest("999")
    assert meta["as_base_half_spread_bps_at_decision"] is None


def test_order_metadata_lookup_missing_order_returns_empty_dict() -> None:
    """Same behaviour as the other decision-state fields: a fill
    whose parent order isn't in the local DB returns a dict with
    every value None — the caller treats this as "no parent
    recorded", which on the Fill side becomes a NULL stamp."""
    s = _settings()
    st = Storage(s)
    st.init_schema()
    meta = st.order_metadata_for_fill_ingest("nonexistent-oid")
    assert meta["as_base_half_spread_bps_at_decision"] is None
