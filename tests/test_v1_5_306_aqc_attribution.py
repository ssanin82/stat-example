"""v1.5.306 (audit §5 P0 #2) — Active-Quoting-Controller attribution.

TWO independent stamping mechanisms, both observability-only (the AQC's
effect on quoting already flows through the half-spread floor / inventory
gates / skew — these columns just RECORD the controller's state):

* **§4.2 per-fill** — ``aqc_aggression_level_at_decision`` (REAL) +
  ``aqc_safety_floor_engaged_at_decision`` (INTEGER 0/1) added to
  ``orders`` + ``fills`` and registered in
  :data:`Storage._DECISION_STATE_COLS`. Stamped on the WorkingOrder at
  place-time, copied to each resulting fill via
  ``order_metadata_for_fill_ingest``. Same shape as the v1.5.190 AS-base
  / v1.5.204 microprice per-fill plumbing
  (``tests/test_phase8a_option_c_as_attribution.py``).

* **§4.1 per-tick** — five columns on ``quote_decisions``
  (``aqc_aggression_level``, ``aqc_integrator``,
  ``aqc_safety_floor_engaged``, ``aqc_observed_net_edge_per_min_usd``,
  ``aqc_observed_markout_5s_mean_bps``) written EVERY tick (incl.
  no-order ticks) from ``state.active_quoting_controller`` via
  ``Bot._aqc_quote_decision_fields()``.

Both share schema migration v46.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.active_quoting_controller import ActiveQuotingController
from app.bot import Bot
from app.enums import OrderStatus, Side
from app.execution import order_row
from app.fill_ingestion import fill_row
from app.models import Fill, WorkingOrder
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_aqc_attr_{os.getpid()}_{uuid.uuid4().hex}.db"
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


def _db_path(s: UnitTestSettings) -> str:
    return s.database_url.split("sqlite:///", 1)[-1]


# ---------------------------------------------------------------------------
# Schema migration v46
# ---------------------------------------------------------------------------


def test_schema_v46_bumps_user_version() -> None:
    s = _settings()
    Storage(s).init_schema()
    conn = sqlite3.connect(_db_path(s))
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v >= 46
    finally:
        conn.close()


def test_schema_v46_adds_aqc_columns_to_orders_and_fills() -> None:
    s = _settings()
    Storage(s).init_schema()
    conn = sqlite3.connect(_db_path(s))
    try:
        for tbl in ("orders", "fills"):
            cols = {r[1] for r in conn.execute(f"PRAGMA table_info({tbl})")}
            assert "aqc_aggression_level_at_decision" in cols, (
                f"missing aqc_aggression_level_at_decision in {tbl}"
            )
            assert "aqc_safety_floor_engaged_at_decision" in cols, (
                f"missing aqc_safety_floor_engaged_at_decision in {tbl}"
            )
    finally:
        conn.close()


def test_schema_v46_adds_aqc_columns_to_quote_decisions() -> None:
    s = _settings()
    Storage(s).init_schema()
    conn = sqlite3.connect(_db_path(s))
    try:
        cols = {
            r[1] for r in conn.execute("PRAGMA table_info(quote_decisions)")
        }
        for c in (
            "aqc_aggression_level",
            "aqc_integrator",
            "aqc_safety_floor_engaged",
            "aqc_observed_net_edge_per_min_usd",
            "aqc_observed_markout_5s_mean_bps",
        ):
            assert c in cols, f"missing {c} in quote_decisions"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# §4.2 — decision-state propagation contract
# ---------------------------------------------------------------------------


def test_decision_state_cols_includes_aqc() -> None:
    """Both per-fill columns MUST be in ``_DECISION_STATE_COLS`` or
    ``order_metadata_for_fill_ingest`` won't SELECT them, and fill
    ingest won't propagate the value from parent order → fill."""
    cols = set(Storage._DECISION_STATE_COLS)
    assert "aqc_aggression_level_at_decision" in cols
    assert "aqc_safety_floor_engaged_at_decision" in cols


# ---------------------------------------------------------------------------
# §4.2 — row-mapper contracts (float passthrough + SQLite-bool encode)
# ---------------------------------------------------------------------------


def _wo(**kw) -> WorkingOrder:
    base = dict(
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
    base.update(kw)
    return WorkingOrder(**base)


def _fill(**kw) -> Fill:
    base = dict(
        fill_id="x",
        order_id_exchange="0",
        client_order_id=None,
        ts_fill=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=-0.0001,
        liquidity_flag="resting",
        mid_at_fill=1.0,
    )
    base.update(kw)
    return Fill(**base)


def test_order_row_includes_aqc_fields() -> None:
    row = order_row(
        _wo(
            aqc_aggression_level_at_decision=0.73,
            aqc_safety_floor_engaged_at_decision=True,
        )
    )
    assert row["aqc_aggression_level_at_decision"] == 0.73
    # Bool encodes to 1 for SQLite storage.
    assert row["aqc_safety_floor_engaged_at_decision"] == 1


def test_order_row_aqc_floor_false_encodes_zero() -> None:
    row = order_row(
        _wo(
            aqc_aggression_level_at_decision=0.0,
            aqc_safety_floor_engaged_at_decision=False,
        )
    )
    assert row["aqc_aggression_level_at_decision"] == 0.0
    assert row["aqc_safety_floor_engaged_at_decision"] == 0


def test_order_row_aqc_none_when_unset() -> None:
    """Legacy / SF / manual paths + AQC-disabled construct WO without
    controller state — both fields serialise as None (NULL)."""
    row = order_row(_wo())
    assert row["aqc_aggression_level_at_decision"] is None
    assert row["aqc_safety_floor_engaged_at_decision"] is None


def test_fill_row_includes_aqc_fields() -> None:
    row = fill_row(
        _fill(
            aqc_aggression_level_at_decision=0.42,
            aqc_safety_floor_engaged_at_decision=True,
        )
    )
    assert row["aqc_aggression_level_at_decision"] == 0.42
    assert row["aqc_safety_floor_engaged_at_decision"] == 1


def test_fill_row_aqc_floor_false_encodes_zero() -> None:
    row = fill_row(
        _fill(
            aqc_aggression_level_at_decision=0.0,
            aqc_safety_floor_engaged_at_decision=False,
        )
    )
    assert row["aqc_safety_floor_engaged_at_decision"] == 0


def test_fill_row_aqc_none_when_unset() -> None:
    row = fill_row(_fill())
    assert row["aqc_aggression_level_at_decision"] is None
    assert row["aqc_safety_floor_engaged_at_decision"] is None


# ---------------------------------------------------------------------------
# §4.2 — end-to-end round-trip via storage metadata lookup
# ---------------------------------------------------------------------------


def test_order_metadata_lookup_returns_aqc_fields() -> None:
    """Persist an order with non-NULL AQC state, then read it back via
    the same lookup ``fill_ingestion`` uses at fill ingest time. The
    float survives as a float; the bool survives as the stored int (the
    Fill reconstruction wraps it in ``_opt_bool``)."""
    s = _settings()
    st = Storage(s)
    st.init_schema()
    wo = _wo(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=1234567890,
        client_order_id="cloid-1",
        side=Side.BUY,
        price=2.030,
        size=2.0,
        ts_created=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        aqc_aggression_level_at_decision=0.61,
        aqc_safety_floor_engaged_at_decision=True,
    )
    st.insert_order_row(order_row(wo))
    meta = st.order_metadata_for_fill_ingest("1234567890")
    assert meta["aqc_aggression_level_at_decision"] == 0.61
    # Stored as INTEGER 1; truthy round-trips to True via _opt_bool.
    assert bool(int(meta["aqc_safety_floor_engaged_at_decision"])) is True


def test_order_metadata_lookup_aqc_floor_false_round_trips() -> None:
    s = _settings()
    st = Storage(s)
    st.init_schema()
    wo = _wo(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=222,
        client_order_id="cloid-floor-off",
        ts_created=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        aqc_aggression_level_at_decision=0.95,
        aqc_safety_floor_engaged_at_decision=False,
    )
    st.insert_order_row(order_row(wo))
    meta = st.order_metadata_for_fill_ingest("222")
    assert meta["aqc_aggression_level_at_decision"] == 0.95
    assert bool(int(meta["aqc_safety_floor_engaged_at_decision"])) is False


def test_order_metadata_lookup_aqc_none_on_legacy_row() -> None:
    """A WO with no AQC state persists as NULL; the lookup returns None
    (not raises) — same contract as the other *_at_decision fields for
    pre-v1.5.306 rows + AQC-disabled placements."""
    s = _settings()
    st = Storage(s)
    st.init_schema()
    wo = _wo(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=999,
        client_order_id="cloid-legacy",
        side=Side.SELL,
        price=2.035,
        size=2.0,
        ts_created=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    st.insert_order_row(order_row(wo))
    meta = st.order_metadata_for_fill_ingest("999")
    assert meta["aqc_aggression_level_at_decision"] is None
    assert meta["aqc_safety_floor_engaged_at_decision"] is None


def test_order_metadata_lookup_missing_order_returns_none() -> None:
    s = _settings()
    st = Storage(s)
    st.init_schema()
    meta = st.order_metadata_for_fill_ingest("nonexistent-oid")
    assert meta["aqc_aggression_level_at_decision"] is None
    assert meta["aqc_safety_floor_engaged_at_decision"] is None


# ---------------------------------------------------------------------------
# §4.1 — per-tick trace helper Bot._aqc_quote_decision_fields()
# ---------------------------------------------------------------------------


def _stub_bot_with_controller(aqc) -> types.SimpleNamespace:
    """A minimal stand-in for ``self`` so we can exercise the helper
    without constructing a full Bot. The helper only touches
    ``self._state.active_quoting_controller``."""
    return types.SimpleNamespace(
        _state=types.SimpleNamespace(active_quoting_controller=aqc)
    )


def test_aqc_quote_decision_fields_from_controller() -> None:
    aqc = ActiveQuotingController(
        aggression_level=0.42,
        integrator=0.13,
        safety_floor_engaged=True,
        last_observed_net_edge_per_min_usd=1.5,
        last_observed_markout_5s_mean_bps=-0.8,
    )
    fields = Bot._aqc_quote_decision_fields(_stub_bot_with_controller(aqc))
    assert fields["aqc_aggression_level"] == 0.42
    assert fields["aqc_integrator"] == 0.13
    assert fields["aqc_safety_floor_engaged"] == 1
    assert fields["aqc_observed_net_edge_per_min_usd"] == 1.5
    assert fields["aqc_observed_markout_5s_mean_bps"] == -0.8


def test_aqc_quote_decision_fields_floor_false_is_zero() -> None:
    aqc = ActiveQuotingController(
        aggression_level=0.0,
        integrator=0.0,
        safety_floor_engaged=False,
        last_observed_net_edge_per_min_usd=None,
        last_observed_markout_5s_mean_bps=None,
    )
    fields = Bot._aqc_quote_decision_fields(_stub_bot_with_controller(aqc))
    assert fields["aqc_safety_floor_engaged"] == 0
    # Missing observation inputs propagate as None (not 0.0).
    assert fields["aqc_observed_net_edge_per_min_usd"] is None
    assert fields["aqc_observed_markout_5s_mean_bps"] is None


def test_aqc_quote_decision_fields_none_when_controller_absent() -> None:
    """AQC disabled → controller is None → every field None (and the
    helper must never raise)."""
    fields = Bot._aqc_quote_decision_fields(_stub_bot_with_controller(None))
    assert fields == {
        "aqc_aggression_level": None,
        "aqc_integrator": None,
        "aqc_safety_floor_engaged": None,
        "aqc_observed_net_edge_per_min_usd": None,
        "aqc_observed_markout_5s_mean_bps": None,
    }


def test_aqc_quote_decision_fields_non_finite_coerced_none() -> None:
    aqc = ActiveQuotingController(
        aggression_level=float("nan"),
        integrator=float("inf"),
        safety_floor_engaged=True,
        last_observed_net_edge_per_min_usd=float("-inf"),
        last_observed_markout_5s_mean_bps=2.0,
    )
    fields = Bot._aqc_quote_decision_fields(_stub_bot_with_controller(aqc))
    assert fields["aqc_aggression_level"] is None
    assert fields["aqc_integrator"] is None
    assert fields["aqc_observed_net_edge_per_min_usd"] is None
    # The finite value still comes through.
    assert fields["aqc_observed_markout_5s_mean_bps"] == 2.0
    # The bool flag is unaffected by the float coercion.
    assert fields["aqc_safety_floor_engaged"] == 1


def test_aqc_quote_decision_fields_missing_state_attr_safe() -> None:
    """If ``_state`` has no ``active_quoting_controller`` attribute at
    all (getattr default None), the helper returns all-None rather than
    raising — observability must never break the hot path."""
    stub = types.SimpleNamespace(_state=types.SimpleNamespace())
    fields = Bot._aqc_quote_decision_fields(stub)
    assert all(v is None for v in fields.values())
    assert set(fields) == {
        "aqc_aggression_level",
        "aqc_integrator",
        "aqc_safety_floor_engaged",
        "aqc_observed_net_edge_per_min_usd",
        "aqc_observed_markout_5s_mean_bps",
    }


# ---------------------------------------------------------------------------
# §4.1 — quote_decisions row insert round-trip
# ---------------------------------------------------------------------------


def test_quote_decisions_insert_round_trip_with_aqc() -> None:
    """A quote_decisions row carrying the five AQC fields inserts
    cleanly (the v46 columns exist) and reads back unchanged."""
    s = _settings()
    st = Storage(s)
    st.init_schema()
    row = {
        "ts": "2026-06-01T12:00:00+00:00",
        "symbol": "TEST",
        "quote_cycle_id": "cyc-1",
        "aqc_aggression_level": 0.55,
        "aqc_integrator": 0.22,
        "aqc_safety_floor_engaged": 1,
        "aqc_observed_net_edge_per_min_usd": 3.14,
        "aqc_observed_markout_5s_mean_bps": -1.2,
    }
    st.insert_quote_decision(row)
    conn = sqlite3.connect(_db_path(s))
    try:
        got = conn.execute(
            "SELECT aqc_aggression_level, aqc_integrator, "
            "aqc_safety_floor_engaged, aqc_observed_net_edge_per_min_usd, "
            "aqc_observed_markout_5s_mean_bps FROM quote_decisions"
        ).fetchone()
    finally:
        conn.close()
    assert got == (0.55, 0.22, 1, 3.14, -1.2)


def test_quote_decisions_insert_round_trip_with_aqc_nulls() -> None:
    """AQC-disabled / no-controller ticks write all-None AQC fields;
    they persist as SQL NULL and read back as None."""
    s = _settings()
    st = Storage(s)
    st.init_schema()
    row = {
        "ts": "2026-06-01T12:00:01+00:00",
        "symbol": "TEST",
        "quote_cycle_id": "cyc-2",
        "aqc_aggression_level": None,
        "aqc_integrator": None,
        "aqc_safety_floor_engaged": None,
        "aqc_observed_net_edge_per_min_usd": None,
        "aqc_observed_markout_5s_mean_bps": None,
    }
    st.insert_quote_decision(row)
    conn = sqlite3.connect(_db_path(s))
    try:
        got = conn.execute(
            "SELECT aqc_aggression_level, aqc_integrator, "
            "aqc_safety_floor_engaged, aqc_observed_net_edge_per_min_usd, "
            "aqc_observed_markout_5s_mean_bps FROM quote_decisions"
        ).fetchone()
    finally:
        conn.close()
    assert got == (None, None, None, None, None)
