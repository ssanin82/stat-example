"""Phase 0.5 regression — ``orders_lifecycle_since`` derived columns.

The query adds two new SELECT-time projections (v1.4.0):
  * ``cancel_decision_to_send_ms`` = ts_cancel_sent - ts_cancel_requested
  * ``cancel_send_to_ack_ms``      = ts_cancel_acked - ts_cancel_sent

Both must:
  * Match the expected diff (in ms) on synthetic rows with known deltas
  * Return NULL when either input timestamp is NULL (the row was
    written before v1.4.0, or the cancel HTTP path wasn't reached)
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings_db() -> tuple[UnitTestSettings, Path, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_lifecycle_ms_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH_USDT_Perp",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    return s, path, storage


def _insert_row(
    storage: Storage,
    *,
    order_id_local: str,
    ts_created: datetime,
    ts_cancel_requested: datetime | None,
    ts_cancel_sent: datetime | None,
    ts_cancel_acked: datetime | None,
    ts_closed: datetime | None = None,
) -> None:
    row = {
        "order_id_local": order_id_local,
        "order_id_exchange": "12345",
        "client_order_id": f"cid-{order_id_local}",
        "ts_created": ts_created.isoformat(),
        "ts_sent": None,
        "ts_ack": None,
        "ts_closed": ts_closed.isoformat() if ts_closed else None,
        "ts_cancel_requested": (
            ts_cancel_requested.isoformat() if ts_cancel_requested else None
        ),
        "symbol": "ETH_USDT_Perp",
        "side": "BUY",
        "price": 2400.0,
        "size": 0.01,
        "post_only": 1,
        "status": "CANCELED",
        "ts_cancel_sent": (
            ts_cancel_sent.isoformat() if ts_cancel_sent else None
        ),
        "ts_cancel_acked": (
            ts_cancel_acked.isoformat() if ts_cancel_acked else None
        ),
        "venue_cancel_utime_ms": None,
    }
    storage.insert_order_row(row)


def test_derived_cancel_ms_columns_correct_on_full_data() -> None:
    """Synthetic row with known deltas: 7 ms decision→send, 5 ms
    send→ack. The derived columns should match within 1 ms (SQLite's
    julianday rounding can introduce a sub-ms error on small
    intervals)."""
    s, path, storage = _settings_db()
    try:
        t0 = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
        _insert_row(
            storage,
            order_id_local="row-full",
            ts_created=t0,
            ts_cancel_requested=t0 + timedelta(milliseconds=10),
            ts_cancel_sent=t0 + timedelta(milliseconds=17),  # +7 ms
            ts_cancel_acked=t0 + timedelta(milliseconds=22),  # +5 ms
            ts_closed=t0 + timedelta(milliseconds=30),
        )
        rows = storage.orders_lifecycle_since(t0.isoformat())
        assert len(rows) == 1
        r = rows[0]
        # Allow ±1 ms tolerance for julianday rounding.
        assert abs(int(r["cancel_decision_to_send_ms"]) - 7) <= 1
        assert abs(int(r["cancel_send_to_ack_ms"]) - 5) <= 1
    finally:
        path.unlink(missing_ok=True)


def test_derived_cancel_ms_columns_null_when_inputs_missing() -> None:
    """Pre-v1.4.0 rows have NULL for the new timestamps. The derived
    columns must also return NULL — not 0, not a garbage value."""
    s, path, storage = _settings_db()
    try:
        t0 = datetime(2026, 5, 16, 12, 0, 0, tzinfo=timezone.utc)
        # Legacy-shaped row: cancel_requested stamped, but
        # ts_cancel_sent / ts_cancel_acked were never written
        # (pre-v1.4.0 code).
        _insert_row(
            storage,
            order_id_local="row-legacy",
            ts_created=t0,
            ts_cancel_requested=t0 + timedelta(milliseconds=10),
            ts_cancel_sent=None,
            ts_cancel_acked=None,
        )
        # Also: a row where send was stamped but ack wasn't (benign-
        # missing cancel race — v1.4.0 path explicitly skips the ack
        # stamp). The send leg should compute; the ack leg stays NULL.
        _insert_row(
            storage,
            order_id_local="row-benign",
            ts_created=t0,
            ts_cancel_requested=t0 + timedelta(milliseconds=10),
            ts_cancel_sent=t0 + timedelta(milliseconds=15),
            ts_cancel_acked=None,
        )
        rows = storage.orders_lifecycle_since(t0.isoformat())
        rows_by_id = {r["order_id_local"]: r for r in rows}
        legacy = rows_by_id["row-legacy"]
        benign = rows_by_id["row-benign"]
        # Legacy: both NULL.
        assert legacy["cancel_decision_to_send_ms"] is None
        assert legacy["cancel_send_to_ack_ms"] is None
        # Benign: send leg computes, ack leg NULL.
        assert benign["cancel_decision_to_send_ms"] is not None
        assert benign["cancel_send_to_ack_ms"] is None
    finally:
        path.unlink(missing_ok=True)


def test_schema_version_is_35() -> None:
    """v35 added the three new columns. Confirm the bump landed."""
    assert Storage.SCHEMA_VERSION >= 35, (
        f"Schema version must be >= 35 (v1.4.0 Phase 0.5 added the "
        f"cancel-timestamp columns); got {Storage.SCHEMA_VERSION}"
    )
