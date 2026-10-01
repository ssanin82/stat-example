"""SF episode attribution end-to-end (Phase 2 of plans/20260507-sf-frontend.md).

The shipped pieces:

- Schema: ``soft_flatten_events`` table + ``soft_flatten_event_id``
  FK column on ``orders`` and ``fills``.
- ``BotState.soft_flatten_event_id`` (set on enter, cleared on exit).
- ``WorkingOrder.soft_flatten_event_id`` (stamped at construction
  from state).
- ``Fill.soft_flatten_event_id`` (looked up by parent order at
  ingest time).
- Storage helpers: ``insert_soft_flatten_event``,
  ``update_soft_flatten_event_end``,
  ``recent_soft_flatten_events`` (with attribution counts via JOIN),
  ``soft_flatten_event_id_for_order``.
- Live-stats publisher: ``soft_flatten_attribution`` block
  (events list + order_tags + fill_tags) for the dashboard.

These tests cover the data layer end-to-end without touching the
SF tick worker (which has heavy execution-engine dependencies).
"""

from __future__ import annotations

import inspect
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.enums import OrderStatus, Side
from app.execution import order_row
from app.fill_ingestion import fill_row
from app.live_stats import (
    LiveStatsPublisher,
    _soft_flatten_attribution_block,
)
from app.models import Fill, WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_sfattr_{os.getpid()}_{uuid.uuid4().hex}.db"
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
# Schema
# ---------------------------------------------------------------------------


def test_schema_has_soft_flatten_table_and_fk_columns() -> None:
    """Migration creates the events table and adds NULL-safe FK
    columns to orders + fills. user_version should be ≥ 15."""
    import sqlite3

    s = _settings()
    db_url = s.database_url
    db_path = db_url.split("sqlite:///", 1)[-1]
    Storage(s).init_schema()
    conn = sqlite3.connect(db_path)
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v >= 15
        cols = {r[1] for r in conn.execute("PRAGMA table_info(orders)")}
        assert "soft_flatten_event_id" in cols
        cols = {r[1] for r in conn.execute("PRAGMA table_info(fills)")}
        assert "soft_flatten_event_id" in cols
        # Episodes table
        rows = list(conn.execute("PRAGMA table_info(soft_flatten_events)"))
        assert any(r[1] == "id" for r in rows)
        assert any(r[1] == "ts_start" for r in rows)
        assert any(r[1] == "ts_end" for r in rows)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def test_insert_event_returns_id_and_recent_join_returns_attribution_counts() -> None:
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = storage.insert_soft_flatten_event(
        {
            "ts_start": "2026-05-07T17:30:00+00:00",
            "trigger_reason": "position_drawdown_gate",
            "entry_position_qty": -3.0,
            "entry_mid_price": 0.973,
        }
    )
    assert isinstance(ev_id, int) and ev_id > 0
    storage.insert_order_row(
        {
            "order_id_local": "oA",
            "order_id_exchange": "111",
            "ts_created": "2026-05-07T17:30:01+00:00",
            "symbol": "TEST",
            "side": "BUY",
            "price": 0.973,
            "size": 1.0,
            "post_only": 1,
            "status": "ACKED",
            "soft_flatten_event_id": ev_id,
        }
    )
    storage.insert_fill_row(
        {
            "fill_id": "fA",
            "order_id_exchange": "111",
            "ts_fill": "2026-05-07T17:30:02+00:00",
            "symbol": "TEST",
            "side": "BUY",
            "price": 0.973,
            "size": 2.0,
            "notional": 1.946,
            "fee": -0.0001,
            "liquidity_flag": "resting",
            "soft_flatten_event_id": ev_id,
        }
    )
    storage.update_soft_flatten_event_end(
        ev_id, {"ts_end": "2026-05-07T17:31:00+00:00", "exit_reason": "flat"}
    )
    events = storage.recent_soft_flatten_events(limit=10)
    assert len(events) == 1
    e = events[0]
    assert e["id"] == ev_id
    assert e["exit_reason"] == "flat"
    assert e["attributed_orders_count"] == 1
    assert e["attributed_fills_count"] == 1
    assert e["attributed_fills_notional_usd"] == pytest.approx(1.946)


def test_recent_join_excludes_non_sf_orders_and_fills() -> None:
    """Rows with NULL soft_flatten_event_id must NOT be counted in
    the attributed totals — the LEFT JOIN must filter on the FK."""
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = storage.insert_soft_flatten_event(
        {"ts_start": "2026-05-07T17:30:00+00:00"}
    )
    # SF-tagged
    storage.insert_order_row(
        {
            "order_id_local": "oA",
            "order_id_exchange": "111",
            "ts_created": "2026-05-07T17:30:01+00:00",
            "symbol": "TEST",
            "side": "BUY",
            "price": 1.0,
            "size": 1.0,
            "post_only": 1,
            "status": "ACKED",
            "soft_flatten_event_id": ev_id,
        }
    )
    # Non-SF (NULL FK)
    storage.insert_order_row(
        {
            "order_id_local": "oB",
            "order_id_exchange": "222",
            "ts_created": "2026-05-07T17:32:00+00:00",
            "symbol": "TEST",
            "side": "SELL",
            "price": 1.0,
            "size": 1.0,
            "post_only": 1,
            "status": "ACKED",
        }
    )
    events = storage.recent_soft_flatten_events(limit=10)
    assert events[0]["attributed_orders_count"] == 1


def test_soft_flatten_event_id_for_order_returns_fk_or_none() -> None:
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = storage.insert_soft_flatten_event(
        {"ts_start": "2026-05-07T17:30:00+00:00"}
    )
    storage.insert_order_row(
        {
            "order_id_local": "oA",
            "order_id_exchange": "111",
            "ts_created": "2026-05-07T17:30:01+00:00",
            "symbol": "TEST",
            "side": "BUY",
            "price": 1.0,
            "size": 1.0,
            "post_only": 1,
            "status": "ACKED",
            "soft_flatten_event_id": ev_id,
        }
    )
    storage.insert_order_row(
        {
            "order_id_local": "oB",
            "order_id_exchange": "222",
            "ts_created": "2026-05-07T17:32:00+00:00",
            "symbol": "TEST",
            "side": "SELL",
            "price": 1.0,
            "size": 1.0,
            "post_only": 1,
            "status": "ACKED",
        }
    )
    assert storage.soft_flatten_event_id_for_order("111") == ev_id
    assert storage.soft_flatten_event_id_for_order("222") is None
    assert storage.soft_flatten_event_id_for_order("999") is None


# ---------------------------------------------------------------------------
# Model + row builders
# ---------------------------------------------------------------------------


def test_working_order_default_sf_event_id_is_none() -> None:
    wo = WorkingOrder(
        order_id_local="oid",
        order_id_exchange=None,
        client_order_id=None,
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.NEW_LOCAL,
    )
    assert wo.soft_flatten_event_id is None


def test_order_row_serializes_sf_event_id() -> None:
    wo = WorkingOrder(
        order_id_local="oid",
        order_id_exchange=None,
        client_order_id=None,
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        post_only=True,
        status=OrderStatus.NEW_LOCAL,
        soft_flatten_event_id=42,
    )
    row = order_row(wo)
    assert row["soft_flatten_event_id"] == 42


def test_fill_row_serializes_sf_event_id() -> None:
    f = Fill(
        fill_id="fA",
        order_id_exchange=111,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=0.0,
        liquidity_flag="resting",
        mid_at_fill=1.0,
        soft_flatten_event_id=42,
    )
    row = fill_row(f)
    assert row["soft_flatten_event_id"] == 42


# ---------------------------------------------------------------------------
# State field
# ---------------------------------------------------------------------------


def test_botstate_sf_event_id_default_none() -> None:
    state = BotState(_settings())
    assert state.soft_flatten_event_id is None


# ---------------------------------------------------------------------------
# Bot enter/exit wiring (source-substring sentinels: full integration
# requires execution engine + market data; the source check is the
# de facto regression guard for the wiring)
# ---------------------------------------------------------------------------


def test_enter_soft_flatten_inserts_event_and_captures_id() -> None:
    """Source-level sentinel: ``Bot._enter_soft_flatten`` must call
    ``insert_soft_flatten_event`` and assign the result to
    ``state.soft_flatten_event_id``. Catches a future refactor that
    silently drops the wiring."""
    from app.bot import Bot

    src = inspect.getsource(Bot._enter_soft_flatten)
    assert "insert_soft_flatten_event" in src
    assert "soft_flatten_event_id" in src


def test_exit_soft_flatten_updates_event_end_and_clears_id() -> None:
    from app.bot import Bot

    src = inspect.getsource(Bot._exit_soft_flatten)
    assert "update_soft_flatten_event_end" in src
    assert "soft_flatten_event_id = None" in src


def test_stage_place_order_local_stamps_sf_event_id() -> None:
    """Source-level sentinel: every WorkingOrder built by the
    placement path picks up ``state.soft_flatten_event_id``. During
    SF this stamps SF orders; outside SF it's None (NULL FK)."""
    from app.execution import OrderManager

    src = inspect.getsource(OrderManager._stage_place_order_local)
    assert "soft_flatten_event_id" in src
    assert "self._state.soft_flatten_event_id" in src


def test_fill_ingest_looks_up_parent_sf_event_id() -> None:
    """Fill ingestion reads the parent order's FK and stamps it on
    the Fill before storage write. Without this, fills wouldn't be
    attributed even when their parent order is.

    v1.4.37 (Codex #5): the lookup method changed from the three
    separate calls (``soft_flatten_event_id_for_order`` /
    ``order_quote_quality_for_order`` /
    ``order_decision_state_for_order``) to a single combined call
    ``order_metadata_for_fill_ingest`` that returns all three field
    sets in one SELECT. The FK still gets stamped on the Fill via
    ``soft_flatten_event_id=sf_event_id``; only the source of
    ``sf_event_id`` changed (now ``meta.get("soft_flatten_event_id")``).
    """
    import app.fill_ingestion as fi

    src = inspect.getsource(fi.ingest_hl_fill_raw)
    # v1.4.37+: combined lookup is what fill ingestion uses now.
    assert "order_metadata_for_fill_ingest" in src
    # FK still propagates through ``sf_event_id`` to the Fill row.
    assert "soft_flatten_event_id=sf_event_id" in src


# ---------------------------------------------------------------------------
# Live stats publisher block
# ---------------------------------------------------------------------------


def test_live_stats_attribution_block_with_storage_returns_events_and_tags() -> None:
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = storage.insert_soft_flatten_event(
        {
            "ts_start": "2026-05-07T17:30:00+00:00",
            "trigger_reason": "position_drawdown_gate",
            "entry_position_qty": -3.0,
            "entry_mid_price": 0.973,
        }
    )
    storage.insert_order_row(
        {
            "order_id_local": "oA",
            "order_id_exchange": "111",
            "ts_created": "2026-05-07T17:30:01+00:00",
            "symbol": "TEST",
            "side": "BUY",
            "price": 1.0,
            "size": 1.0,
            "post_only": 1,
            "status": "ACKED",
            "soft_flatten_event_id": ev_id,
        }
    )
    storage.insert_fill_row(
        {
            "fill_id": "fA",
            "order_id_exchange": "111",
            "ts_fill": "2026-05-07T17:30:02+00:00",
            "symbol": "TEST",
            "side": "BUY",
            "price": 1.0,
            "size": 1.0,
            "notional": 1.0,
            "fee": -0.0001,
            "liquidity_flag": "resting",
            "soft_flatten_event_id": ev_id,
        }
    )
    block = _soft_flatten_attribution_block(storage)
    assert len(block["events"]) == 1
    e = block["events"][0]
    assert e["id"] == ev_id
    assert e["attributed_orders_count"] == 1
    assert e["attributed_fills_count"] == 1
    assert block["order_tags"] == {"111": ev_id}
    assert block["fill_tags"] == {"fA": ev_id}


def test_live_stats_attribution_block_without_storage_returns_empty() -> None:
    block = _soft_flatten_attribution_block(None)
    assert block == {"events": [], "order_tags": {}, "fill_tags": {}}


# ---------------------------------------------------------------------------
# v1.4.36 Codex #3 + #7 — current-session scope
# ---------------------------------------------------------------------------


def test_recent_soft_flatten_events_filters_by_since_ts() -> None:
    """Codex #3 fix: ``recent_soft_flatten_events(since_ts=...)``
    returns only episodes with ``ts_start >= since_ts``. Without the
    filter the method returns all rows (existing behaviour, used by
    postmortem + equity-history callers)."""
    storage = Storage(_settings())
    storage.init_schema()
    # Yesterday's episode — must be excluded when since_ts = today.
    old_id = storage.insert_soft_flatten_event(
        {
            "ts_start": "2026-05-07T17:30:00+00:00",
            "trigger_reason": "yesterday_episode",
            "entry_position_qty": -3.0,
            "entry_mid_price": 1.0,
        }
    )
    # Today's episode — must be included.
    new_id = storage.insert_soft_flatten_event(
        {
            "ts_start": "2026-05-17T17:30:00+00:00",
            "trigger_reason": "today_episode",
            "entry_position_qty": -2.0,
            "entry_mid_price": 1.1,
        }
    )
    # No filter → both rows returned (back-compat for postmortem).
    all_events = storage.recent_soft_flatten_events(limit=10)
    all_ids = {e["id"] for e in all_events}
    assert {old_id, new_id} <= all_ids
    # Filter to today's session → only today's row.
    today_events = storage.recent_soft_flatten_events(
        limit=10, since_ts="2026-05-17T00:00:00+00:00"
    )
    today_ids = {e["id"] for e in today_events}
    assert new_id in today_ids
    assert old_id not in today_ids


def test_order_metadata_for_fill_ingest_returns_combined_fields() -> None:
    """v1.4.37 Codex #5 fix: ``order_metadata_for_fill_ingest``
    returns soft_flatten_event_id + quote-quality + decision-state
    fields from a SINGLE indexed lookup, replacing the pre-v1.4.37
    three-call sequence in fill ingestion."""
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = storage.insert_soft_flatten_event(
        {
            "ts_start": "2026-05-17T17:00:00+00:00",
            "trigger_reason": "test",
            "entry_position_qty": -1.0,
            "entry_mid_price": 1.0,
        }
    )
    storage.insert_order_row(
        {
            "order_id_local": "oM",
            "order_id_exchange": "999",
            "ts_created": "2026-05-17T17:00:01+00:00",
            "symbol": "TEST",
            "side": "BUY",
            "price": 1.0,
            "size": 1.0,
            "post_only": 1,
            "status": "ACKED",
            "soft_flatten_event_id": ev_id,
            "target_half_spread_bps": 3.5,
            "quote_aggressiveness": "at_touch",
            "decision_reason_at_decision": "baseline",
            "toxicity_score_at_decision": 0.5,
            "vol_estimate_at_decision": 1.2,
            "active_sides_at_decision": "BOTH",
        }
    )
    meta = storage.order_metadata_for_fill_ingest("999")
    # SF + quality fields
    assert meta["soft_flatten_event_id"] == ev_id
    assert meta["target_half_spread_bps"] == 3.5
    assert meta["quote_aggressiveness"] == "at_touch"
    # Decision-state fields
    assert meta["decision_reason_at_decision"] == "baseline"
    assert meta["toxicity_score_at_decision"] == 0.5
    assert meta["vol_estimate_at_decision"] == 1.2
    assert meta["active_sides_at_decision"] == "BOTH"


def test_order_metadata_for_fill_ingest_unknown_oid_returns_none_dict() -> None:
    """v1.4.37 Codex #5: when no parent order is recorded for the
    oid (REST-catch-up fill, legacy row, race), every value is None
    — same fallback as the three legacy methods produced."""
    storage = Storage(_settings())
    storage.init_schema()
    meta = storage.order_metadata_for_fill_ingest("unknown-oid")
    assert meta["soft_flatten_event_id"] is None
    assert meta["target_half_spread_bps"] is None
    assert meta["quote_aggressiveness"] is None
    assert meta["decision_reason_at_decision"] is None
    # Spot-check a few decision-state cols to confirm they're keyed.
    assert "toxicity_score_at_decision" in meta
    assert meta["toxicity_score_at_decision"] is None
    assert "vol_estimate_at_decision" in meta


def test_attribution_block_scopes_events_to_session_started_at() -> None:
    """Codex #3 fix: ``_soft_flatten_attribution_block(...,
    session_started_at_iso=X)`` returns only episodes with
    ``ts_start >= X``. The DB still contains the historical
    episodes — they're just hidden from the live-stats publication."""
    storage = Storage(_settings())
    storage.init_schema()
    old_id = storage.insert_soft_flatten_event(
        {
            "ts_start": "2026-05-07T17:30:00+00:00",
            "trigger_reason": "prior_session",
            "entry_position_qty": -1.0,
            "entry_mid_price": 1.0,
        }
    )
    new_id = storage.insert_soft_flatten_event(
        {
            "ts_start": "2026-05-17T19:00:00+00:00",
            "trigger_reason": "current_session",
            "entry_position_qty": -2.0,
            "entry_mid_price": 1.1,
        }
    )
    # No session filter → both events (regression check — old callers
    # like postmortem still see the full history).
    full = _soft_flatten_attribution_block(storage)
    full_ids = {e["id"] for e in full["events"]}
    assert {old_id, new_id} <= full_ids
    # With session filter → only the current-session event.
    scoped = _soft_flatten_attribution_block(
        storage, session_started_at_iso="2026-05-17T00:00:00+00:00"
    )
    scoped_ids = {e["id"] for e in scoped["events"]}
    assert new_id in scoped_ids
    assert old_id not in scoped_ids, (
        "Codex #3 regression: prior-session SF event leaking into "
        "current-session live-stats block."
    )


def test_live_stats_publisher_payload_contains_attribution_block() -> None:
    s = _settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.symbol = "TEST"
    pub = LiveStatsPublisher(
        s, state, bucket="fake", profile_name="test", storage=storage
    )
    payload = pub._build_payload()
    assert "soft_flatten_attribution" in payload
    sfa = payload["soft_flatten_attribution"]
    assert sfa["events"] == []
    assert sfa["order_tags"] == {}
    assert sfa["fill_tags"] == {}
