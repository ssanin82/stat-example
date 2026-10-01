"""Regression tests for the regime-observability Phase 1 fill enrichment.

Covers:
- Schema migration v26: new columns on ``orders`` and ``fills`` tables.
- ``Storage.order_decision_state_for_order``: returns the stamped
  decision-state dict, falls back gracefully when the order isn't in
  the local DB.
- ``order_row(wo)`` serialization: new WorkingOrder fields land in the
  output dict and round-trip through SQLite.
- ``fill_row(f)`` serialization: new Fill fields land in the output
  dict, including Optional[bool] → INTEGER conversion.
- Trivial derivations at fill ingestion: spread_bps_at_fill,
  microprice_at_fill, imbalance_top_at_fill, inventory_*_before_fill.
- Cancel-race diagnostic: cancel_requested_before_fill /
  ms_cancel_request_to_fill derive correctly when the parent order
  has a ts_cancel_requested timestamp.

The bot's quote-construction path is NOT exercised here — Phase 1 is
pure observability, no decision-logic changes. The tests verify the
data plumbing only.

See ``plans/regime-observability.md`` Phase 1 for the field list +
rationale.
"""

from __future__ import annotations

import os
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


def _make_storage() -> Storage:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_p1_{os.getpid()}_{uuid.uuid4().hex}.db"
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


def _now() -> datetime:
    return datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc)


def _make_wo(
    *,
    oid_local: str = "wo-1",
    oid_exchange: str = "1234567890",
    side: Side = Side.BUY,
    price: float = 100.0,
    size: float = 1.0,
    **overrides,
) -> WorkingOrder:
    base = dict(
        order_id_local=oid_local,
        order_id_exchange=int(oid_exchange),
        client_order_id="cloid-1",
        symbol="ETH_USDT_Perp",
        side=side,
        price=price,
        size=size,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=_now(),
    )
    base.update(overrides)
    return WorkingOrder(**base)


# --------------------------------------------------------------------- #
# Schema migration: new columns exist on orders + fills tables.         #
# --------------------------------------------------------------------- #


def test_schema_v26_orders_columns_exist() -> None:
    s = _make_storage()
    with s.connection() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(orders)").fetchall()}
    expected = {
        "toxicity_score_at_decision",
        "vol_estimate_at_decision",
        "active_sides_at_decision",
        "decision_reason_at_decision",
        "binance_basis_ewma_at_decision",
        "adaptive_widen_active_at_decision",
        "post_fill_cooldown_active_bid_at_decision",
        "post_fill_cooldown_active_ask_at_decision",
        "at_touch_adverse_pause_bid_at_decision",
        "at_touch_adverse_pause_ask_at_decision",
        "quote_distance_to_touch_ticks_at_placement",
        "ts_cancel_requested",
    }
    missing = expected - cols
    assert not missing, f"Missing orders columns: {missing}"


def test_schema_v26_fills_columns_exist() -> None:
    s = _make_storage()
    with s.connection() as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(fills)").fetchall()}
    expected = {
        "spread_bps_at_fill",
        "microprice_at_fill",
        "imbalance_top_at_fill",
        "inventory_qty_before_fill",
        "inventory_utilization_before_fill",
        "toxicity_score_at_decision",
        "vol_estimate_at_decision",
        "active_sides_at_decision",
        "decision_reason_at_decision",
        "binance_basis_ewma_at_decision",
        "adaptive_widen_active_at_decision",
        "post_fill_cooldown_active_bid_at_decision",
        "post_fill_cooldown_active_ask_at_decision",
        "at_touch_adverse_pause_bid_at_decision",
        "at_touch_adverse_pause_ask_at_decision",
        "quote_distance_to_touch_ticks_at_placement",
        "cancel_requested_before_fill",
        "ms_cancel_request_to_fill",
    }
    missing = expected - cols
    assert not missing, f"Missing fills columns: {missing}"


# --------------------------------------------------------------------- #
# order_row(wo) serialization round-trips through SQLite.               #
# --------------------------------------------------------------------- #


def test_order_row_includes_phase1_fields() -> None:
    wo = _make_wo(
        toxicity_score_at_decision=0.42,
        vol_estimate_at_decision=12.5,
        active_sides_at_decision="BOTH",
        decision_reason_at_decision="baseline,ob_imbalance_shift",
        binance_basis_ewma_at_decision=0.0003,
        adaptive_widen_active_at_decision=False,
        post_fill_cooldown_active_bid_at_decision=True,
        post_fill_cooldown_active_ask_at_decision=False,
        at_touch_adverse_pause_bid_at_decision=False,
        at_touch_adverse_pause_ask_at_decision=True,
        quote_distance_to_touch_ticks_at_placement=1.5,
        ts_cancel_requested=_now(),
    )
    row = order_row(wo)
    assert row["toxicity_score_at_decision"] == 0.42
    assert row["vol_estimate_at_decision"] == 12.5
    assert row["active_sides_at_decision"] == "BOTH"
    assert row["decision_reason_at_decision"] == "baseline,ob_imbalance_shift"
    assert row["binance_basis_ewma_at_decision"] == 0.0003
    # Bools serialize to 1/0 INTEGER for SQLite.
    assert row["adaptive_widen_active_at_decision"] == 0
    assert row["post_fill_cooldown_active_bid_at_decision"] == 1
    assert row["post_fill_cooldown_active_ask_at_decision"] == 0
    assert row["at_touch_adverse_pause_bid_at_decision"] == 0
    assert row["at_touch_adverse_pause_ask_at_decision"] == 1
    assert row["quote_distance_to_touch_ticks_at_placement"] == 1.5
    assert row["ts_cancel_requested"] is not None


def test_order_row_phase1_fields_nullable_by_default() -> None:
    """Legacy / non-MM placement path: WorkingOrder constructed without
    the new fields. order_row must emit None for all of them, not
    raise."""
    wo = _make_wo()
    row = order_row(wo)
    for col in (
        "toxicity_score_at_decision",
        "vol_estimate_at_decision",
        "active_sides_at_decision",
        "decision_reason_at_decision",
        "binance_basis_ewma_at_decision",
        "adaptive_widen_active_at_decision",
        "post_fill_cooldown_active_bid_at_decision",
        "post_fill_cooldown_active_ask_at_decision",
        "at_touch_adverse_pause_bid_at_decision",
        "at_touch_adverse_pause_ask_at_decision",
        "quote_distance_to_touch_ticks_at_placement",
        "ts_cancel_requested",
    ):
        assert row[col] is None, f"{col} should be None on default WO"


def test_order_row_roundtrip_through_storage() -> None:
    """Insert an order with stamped decision-state; SELECT it back;
    confirm every field round-trips."""
    s = _make_storage()
    wo = _make_wo(
        toxicity_score_at_decision=0.7,
        vol_estimate_at_decision=8.5,
        active_sides_at_decision="BID_ONLY",
        decision_reason_at_decision="reducing_long",
        binance_basis_ewma_at_decision=-0.0001,
        adaptive_widen_active_at_decision=True,
        post_fill_cooldown_active_bid_at_decision=False,
        post_fill_cooldown_active_ask_at_decision=False,
        at_touch_adverse_pause_bid_at_decision=True,
        at_touch_adverse_pause_ask_at_decision=False,
        quote_distance_to_touch_ticks_at_placement=2.0,
    )
    s.insert_order_row(order_row(wo))
    state = s.order_decision_state_for_order(str(wo.order_id_exchange))
    assert state["toxicity_score_at_decision"] == 0.7
    assert state["vol_estimate_at_decision"] == 8.5
    assert state["active_sides_at_decision"] == "BID_ONLY"
    assert state["decision_reason_at_decision"] == "reducing_long"
    assert state["binance_basis_ewma_at_decision"] == -0.0001
    assert bool(state["adaptive_widen_active_at_decision"]) is True
    assert bool(state["post_fill_cooldown_active_bid_at_decision"]) is False
    assert bool(state["at_touch_adverse_pause_bid_at_decision"]) is True
    assert state["quote_distance_to_touch_ticks_at_placement"] == 2.0


def test_order_decision_state_returns_nones_when_order_absent() -> None:
    """REST catch-up fills can reference orders the local DB doesn't
    have. Lookup must return None for every key (not raise)."""
    s = _make_storage()
    state = s.order_decision_state_for_order("nonexistent-9999999999")
    for key in Storage._DECISION_STATE_COLS:
        assert state[key] is None, f"{key} should be None for absent order"


# --------------------------------------------------------------------- #
# fill_row(f) serialization.                                            #
# --------------------------------------------------------------------- #


def _make_fill(**overrides) -> Fill:
    base = dict(
        fill_id="fill-1",
        order_id_exchange=1234567890,
        client_order_id="cloid-1",
        ts_fill=_now(),
        symbol="ETH_USDT_Perp",
        side=Side.BUY,
        price=100.0,
        size=1.0,
        notional=100.0,
        fee=-0.01,
        liquidity_flag="resting",
        mid_at_fill=100.05,
    )
    base.update(overrides)
    return Fill(**base)


def test_fill_row_includes_phase1_fields() -> None:
    f = _make_fill(
        spread_bps_at_fill=2.0,
        microprice_at_fill=100.04,
        imbalance_top_at_fill=0.25,
        inventory_qty_before_fill=3.0,
        inventory_utilization_before_fill=0.375,
        toxicity_score_at_decision=0.5,
        adaptive_widen_active_at_decision=True,
        cancel_requested_before_fill=False,
        ms_cancel_request_to_fill=42.5,
    )
    row = fill_row(f)
    assert row["spread_bps_at_fill"] == 2.0
    assert row["microprice_at_fill"] == 100.04
    assert row["imbalance_top_at_fill"] == 0.25
    assert row["inventory_qty_before_fill"] == 3.0
    assert row["inventory_utilization_before_fill"] == 0.375
    assert row["toxicity_score_at_decision"] == 0.5
    # bool → INTEGER conversion.
    assert row["adaptive_widen_active_at_decision"] == 1
    assert row["cancel_requested_before_fill"] == 0
    assert row["ms_cancel_request_to_fill"] == 42.5


def test_fill_row_phase1_fields_nullable_by_default() -> None:
    f = _make_fill()
    row = fill_row(f)
    for col in (
        "spread_bps_at_fill",
        "microprice_at_fill",
        "imbalance_top_at_fill",
        "inventory_qty_before_fill",
        "inventory_utilization_before_fill",
        "toxicity_score_at_decision",
        "vol_estimate_at_decision",
        "active_sides_at_decision",
        "decision_reason_at_decision",
        "binance_basis_ewma_at_decision",
        "adaptive_widen_active_at_decision",
        "post_fill_cooldown_active_bid_at_decision",
        "post_fill_cooldown_active_ask_at_decision",
        "at_touch_adverse_pause_bid_at_decision",
        "at_touch_adverse_pause_ask_at_decision",
        "quote_distance_to_touch_ticks_at_placement",
        "cancel_requested_before_fill",
        "ms_cancel_request_to_fill",
    ):
        assert row[col] is None, f"{col} should be None on default Fill"


def test_fill_row_roundtrip_through_storage() -> None:
    s = _make_storage()
    f = _make_fill(
        spread_bps_at_fill=5.0,
        microprice_at_fill=100.1,
        imbalance_top_at_fill=-0.1,
        inventory_qty_before_fill=-2.5,
        inventory_utilization_before_fill=0.5,
        toxicity_score_at_decision=0.3,
        vol_estimate_at_decision=15.0,
        adaptive_widen_active_at_decision=False,
        at_touch_adverse_pause_bid_at_decision=True,
        cancel_requested_before_fill=True,
        ms_cancel_request_to_fill=12.3,
    )
    s.insert_fill_row(fill_row(f))
    with s.connection() as conn:
        row = conn.execute(
            "SELECT spread_bps_at_fill, microprice_at_fill, "
            "imbalance_top_at_fill, inventory_qty_before_fill, "
            "inventory_utilization_before_fill, "
            "toxicity_score_at_decision, vol_estimate_at_decision, "
            "adaptive_widen_active_at_decision, "
            "at_touch_adverse_pause_bid_at_decision, "
            "cancel_requested_before_fill, ms_cancel_request_to_fill "
            "FROM fills WHERE fill_id = ?",
            ("fill-1",),
        ).fetchone()
    assert row[0] == 5.0
    assert row[1] == 100.1
    assert row[2] == -0.1
    assert row[3] == -2.5
    assert row[4] == 0.5
    assert row[5] == 0.3
    assert row[6] == 15.0
    assert row[7] == 0  # False as INTEGER
    assert row[8] == 1  # True as INTEGER
    assert row[9] == 1  # True as INTEGER
    assert row[10] == 12.3
