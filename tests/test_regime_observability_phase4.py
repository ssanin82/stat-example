"""Regression tests for regime-observability Phase 4 (a/b/c).

Phase 4a — ``expected_net_edge_bps_at_decision``:
- Schema migration v28 adds the column to ``orders`` + ``fills``.
- Field present on WorkingOrder + Fill dataclasses.
- Propagation channel includes it.
- STAMP ONLY guard: bot's quote-construction path never reads it
  (verified indirectly — the field is computed in
  ``_stage_place_order_local`` and never consumed).

Phase 4b — ``orders_lifecycle_since`` view:
- Returns existing order rows plus three derived columns:
  ``lifetime_ms``, ``cancel_to_close_ms``, ``placement_to_ack_ms``.

Phase 4c — ``PostFillExcursionWatcher``:
- Schema migration v29 adds MAE/MFE columns to fills.
- ``maybe_create`` gates on the feature flag.
- ``track_fill`` records initial state.
- 5s/30s deadline crossing fires ``Storage.update_fill_excursion``
  with correctly-signed MAE/MFE values.
- ``_compute_mae_mfe_bps`` returns the expected formula outputs for
  BUY and SELL.

See ``plans/regime-observability.md`` Phase 4.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.post_fill_excursion_watcher import PostFillExcursionWatcher
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _make_storage() -> Storage:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_p4_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    s = Storage(settings)
    s.init_schema()
    return s


def _make_settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "SYMBOL": "TON-USDT-SWAP",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


# --------------------------------------------------------------------- #
# Phase 4a: expected_net_edge_bps                                        #
# --------------------------------------------------------------------- #


def test_phase4a_schema_v28_columns_exist() -> None:
    s = _make_storage()
    with s.connection() as conn:
        order_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(orders)").fetchall()
        }
        fill_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(fills)").fetchall()
        }
    assert "expected_net_edge_bps_at_decision" in order_cols
    assert "expected_net_edge_bps_at_decision" in fill_cols


def test_phase4a_field_on_working_order_dataclass() -> None:
    from app.models import WorkingOrder
    from app.enums import OrderStatus, Side

    wo = WorkingOrder(
        order_id_local="x",
        order_id_exchange=None,
        client_order_id=None,
        symbol="X",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.ACKED,
        expected_net_edge_bps_at_decision=2.5,
    )
    assert wo.expected_net_edge_bps_at_decision == 2.5


def test_phase4a_decision_state_cols_includes_expected_edge() -> None:
    assert (
        "expected_net_edge_bps_at_decision"
        in Storage._DECISION_STATE_COLS
    )


def test_phase4a_propagation_via_storage_lookup() -> None:
    """End-to-end Phase 4a: insert an order with the expected-edge
    stamp, look it up via the decision-state propagation channel,
    confirm the value is returned."""
    from app.execution import order_row
    from app.models import WorkingOrder
    from app.enums import OrderStatus, Side

    s = _make_storage()
    wo = WorkingOrder(
        order_id_local="wo-a",
        order_id_exchange=999999,
        client_order_id="c",
        symbol="X",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.ACKED,
        expected_net_edge_bps_at_decision=3.7,
    )
    s.insert_order_row(order_row(wo))
    state = s.order_decision_state_for_order("999999")
    assert state["expected_net_edge_bps_at_decision"] == 3.7


# --------------------------------------------------------------------- #
# Phase 4b: orders_lifecycle_since                                       #
# --------------------------------------------------------------------- #


def test_phase4b_orders_lifecycle_since_returns_derived_columns() -> None:
    s = _make_storage()
    # Insert a closed order: ts_created → ts_sent → ts_ack → ts_closed.
    created = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
    s.insert_order_row(
        {
            "order_id_local": "olc-1",
            "order_id_exchange": "111",
            "client_order_id": "c",
            "ts_created": created.isoformat(),
            "ts_sent": (created + timedelta(milliseconds=5)).isoformat(),
            "ts_ack": (created + timedelta(milliseconds=25)).isoformat(),
            "ts_closed": (created + timedelta(seconds=10)).isoformat(),
            "ts_cancel_requested": (
                created + timedelta(seconds=9)
            ).isoformat(),
            "symbol": "X",
            "side": "BUY",
            "price": 1.0,
            "size": 1.0,
            "post_only": 1,
            "status": "CANCELED",
        }
    )
    rows = s.orders_lifecycle_since(
        (created - timedelta(hours=1)).isoformat()
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["lifetime_ms"] is not None
    assert 9_500 <= row["lifetime_ms"] <= 10_500  # ~10s
    assert row["cancel_to_close_ms"] is not None
    assert 500 <= row["cancel_to_close_ms"] <= 1_500  # ~1s
    assert row["placement_to_ack_ms"] is not None
    assert 10 <= row["placement_to_ack_ms"] <= 30  # ~20ms


def test_phase4b_orders_lifecycle_since_handles_open_order() -> None:
    """Order still open (ts_closed = None) → lifetime_ms = NULL,
    cancel_to_close_ms = NULL."""
    s = _make_storage()
    created = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)
    s.insert_order_row(
        {
            "order_id_local": "olc-2",
            "order_id_exchange": "222",
            "client_order_id": "c",
            "ts_created": created.isoformat(),
            "ts_sent": None,
            "ts_ack": None,
            "ts_closed": None,
            "symbol": "X",
            "side": "BUY",
            "price": 1.0,
            "size": 1.0,
            "post_only": 1,
            "status": "ACKED",
        }
    )
    rows = s.orders_lifecycle_since(
        (created - timedelta(hours=1)).isoformat()
    )
    assert len(rows) == 1
    assert rows[0]["lifetime_ms"] is None
    assert rows[0]["cancel_to_close_ms"] is None


# --------------------------------------------------------------------- #
# Phase 4c: PostFillExcursionWatcher                                     #
# --------------------------------------------------------------------- #


def test_phase4c_schema_v29_columns_exist() -> None:
    s = _make_storage()
    with s.connection() as conn:
        cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(fills)").fetchall()
        }
    for col in ("mae_5s_bps", "mfe_5s_bps", "mae_30s_bps", "mfe_30s_bps"):
        assert col in cols


def test_phase4c_watcher_disabled_by_default() -> None:
    settings = _make_settings()  # OBSERVABILITY_MAE_MFE_ENABLED defaults False
    state = BotState(settings)
    storage = _make_storage()
    w = PostFillExcursionWatcher.maybe_create(settings, state, storage)
    assert w is None


def test_phase4c_watcher_enabled_with_flag() -> None:
    settings = _make_settings(OBSERVABILITY_MAE_MFE_ENABLED=True)
    state = BotState(settings)
    storage = _make_storage()
    w = PostFillExcursionWatcher.maybe_create(settings, state, storage)
    assert w is not None


def test_phase4c_track_fill_records_state() -> None:
    settings = _make_settings(OBSERVABILITY_MAE_MFE_ENABLED=True)
    state = BotState(settings)
    storage = _make_storage()
    w = PostFillExcursionWatcher(settings, state, storage)
    w.track_fill("f1", "BUY", 100.0)
    assert "f1" in w._records
    rec = w._records["f1"]
    assert rec["mid_at_fill"] == 100.0
    assert rec["min_mid"] == 100.0
    assert rec["max_mid"] == 100.0


def test_phase4c_compute_mae_mfe_buy() -> None:
    """BUY: bot is long after fill. Adverse = price down.
    mid_at_fill=100, min=99, max=102 → MAE=-100bp, MFE=+200bp."""
    rec = {"mid_at_fill": 100.0, "min_mid": 99.0, "max_mid": 102.0, "side": "BUY"}
    mae, mfe = PostFillExcursionWatcher._compute_mae_mfe_bps(rec)
    assert mae is not None and abs(mae - (-100.0)) < 0.01
    assert mfe is not None and abs(mfe - 200.0) < 0.01


def test_phase4c_compute_mae_mfe_sell() -> None:
    """SELL: bot is short after fill. Adverse = price up.
    mid_at_fill=100, min=99, max=102 → MAE=-200bp (max moved against us),
    MFE=+100bp (min moved in favor)."""
    rec = {"mid_at_fill": 100.0, "min_mid": 99.0, "max_mid": 102.0, "side": "SELL"}
    mae, mfe = PostFillExcursionWatcher._compute_mae_mfe_bps(rec)
    assert mae is not None and abs(mae - (-200.0)) < 0.01
    assert mfe is not None and abs(mfe - 100.0) < 0.01


def test_phase4c_update_fill_excursion_roundtrip() -> None:
    """Storage UPDATE writes MAE/MFE columns; SELECT returns them."""
    from app.models import Fill
    from app.enums import Side
    from app.fill_ingestion import fill_row

    s = _make_storage()
    f = Fill(
        fill_id="exf-1",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        symbol="X",
        side=Side.BUY,
        price=100.0,
        size=1.0,
        notional=100.0,
        fee=-0.01,
        liquidity_flag="resting",
        mid_at_fill=100.0,
    )
    s.insert_fill_row(fill_row(f))
    s.update_fill_excursion("exf-1", mae_5s_bps=-2.5, mfe_5s_bps=1.0)
    s.update_fill_excursion("exf-1", mae_30s_bps=-5.0, mfe_30s_bps=3.0)
    with s.connection() as conn:
        row = conn.execute(
            "SELECT mae_5s_bps, mfe_5s_bps, mae_30s_bps, mfe_30s_bps "
            "FROM fills WHERE fill_id='exf-1'"
        ).fetchone()
    assert row[0] == -2.5
    assert row[1] == 1.0
    assert row[2] == -5.0
    assert row[3] == 3.0


def test_phase4c_watcher_end_to_end_writes_mae_mfe() -> None:
    """End-to-end: track a fill, manipulate state.market.mid_price
    to simulate post-fill price path, advance time, verify
    storage.update_fill_excursion is called with the right values."""
    from app.models import BestBidAsk, Fill
    from app.enums import Side
    from app.fill_ingestion import fill_row

    settings = _make_settings(
        OBSERVABILITY_MAE_MFE_ENABLED=True,
        OBSERVABILITY_MAE_MFE_POLL_INTERVAL_SECONDS=0.05,
    )
    state = BotState(settings)
    storage = _make_storage()

    # Pre-insert the fill so the UPDATE has a row to target.
    f = Fill(
        fill_id="e2e-1",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        symbol="X",
        side=Side.BUY,
        price=100.0,
        size=1.0,
        notional=100.0,
        fee=-0.01,
        liquidity_flag="resting",
        mid_at_fill=100.0,
    )
    storage.insert_fill_row(fill_row(f))

    # Patch the watcher's window constants to small values so the
    # test runs in ~1s.
    import app.post_fill_excursion_watcher as mod

    original_windows = mod._WATCH_WINDOWS_SECONDS
    original_max = mod._MAX_WATCH_SECONDS
    mod._WATCH_WINDOWS_SECONDS = (0.2, 0.6)
    mod._MAX_WATCH_SECONDS = 0.6
    try:
        w = PostFillExcursionWatcher(settings, state, storage)
        # Inject initial market.
        state.market = BestBidAsk(
            symbol="X",
            best_bid=99.9,
            best_ask=100.1,
            mid_price=100.0,
            spread_bps=20.0,
        )
        w.start()
        w.track_fill("e2e-1", "BUY", 100.0)
        # Drive price down to 99 (MAE), then up to 101 (MFE) over time.
        time.sleep(0.1)
        state.market = BestBidAsk(
            symbol="X",
            best_bid=98.9,
            best_ask=99.1,
            mid_price=99.0,
            spread_bps=20.0,
        )
        time.sleep(0.15)  # crosses 0.2s mark
        state.market = BestBidAsk(
            symbol="X",
            best_bid=100.9,
            best_ask=101.1,
            mid_price=101.0,
            spread_bps=20.0,
        )
        time.sleep(0.5)  # crosses 0.6s mark
        w.stop()
        if w._thread:
            w._thread.join(timeout=1.0)

        with storage.connection() as conn:
            row = conn.execute(
                "SELECT mae_5s_bps, mfe_5s_bps, mae_30s_bps, mfe_30s_bps "
                "FROM fills WHERE fill_id='e2e-1'"
            ).fetchone()
        # 5s window saw price drop to 99 → MAE ≈ -100bp.
        # MFE in the 5s window may or may not have caught 101 yet,
        # depending on timing. Just check MAE was captured.
        assert row[0] is not None, "MAE 5s should have been written"
        assert row[0] < 0, f"MAE 5s should be negative, got {row[0]}"
        # 30s window saw both extremes → MFE ≈ +100bp.
        assert row[2] is not None, "MAE 30s should have been written"
        assert row[3] is not None, "MFE 30s should have been written"
        assert row[3] > 0, f"MFE 30s should be positive, got {row[3]}"
    finally:
        mod._WATCH_WINDOWS_SECONDS = original_windows
        mod._MAX_WATCH_SECONDS = original_max
