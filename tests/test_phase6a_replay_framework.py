"""v1.4.93 wedge-elimination-cleanup Phase 6A — replay framework
unit tests (NO filesystem dependency).

The actual snapshot regression tests live under
``tests/integration/test_snapshot_replay.py`` (excluded from default
pytest collection) because they depend on operator-local
``snapshots/`` artifacts that aren't in git.

This file unit-tests the framework code itself by constructing
``SnapshotFixture`` objects inline — same code paths exercised, no
filesystem needed. Runs in default CI.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.enums import Side
from tests.integration.snapshot_replay import (
    SnapshotFixture,
    invariant_no_duplicate_oid_or_cloid,
    invariant_no_wedge_counters_nonzero,
    invariant_risk_state_normal,
    invariant_desync_phase_ok,
    load_snapshot,
    market_from_snapshot,
    position_from_snapshot,
    working_orders_from_snapshot,
)


def _make_inline_fixture(
    *,
    open_orders_raw: list[dict] | None = None,
    state_dict: dict | None = None,
) -> SnapshotFixture:
    """Build a SnapshotFixture from inline dicts. No filesystem access."""
    return SnapshotFixture(
        path=Path("/inline/test/fixture"),
        bot_version="1.4.93-test",
        bot_profile="prod.okx.ton.usdt.perp",
        captured_at_utc=datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc),
        state_dict=state_dict or {"symbol": "TON-USDT-SWAP"},
        open_orders_raw=open_orders_raw or [],
        config={},
        session_summary={},
    )


# ---------------------------------------------------------------------------
# working_orders_from_snapshot — reconstruction logic
# ---------------------------------------------------------------------------


def test_phase6a_framework_reconstructs_empty_orders_returns_empty() -> None:
    fixture = _make_inline_fixture(open_orders_raw=[])
    assert working_orders_from_snapshot(fixture) == []


def test_phase6a_framework_reconstructs_single_buy_at_level_0() -> None:
    fixture = _make_inline_fixture(open_orders_raw=[
        {
            "ordId": "3579105000000000001",
            "clOrdId": "abc123",
            "side": "buy",
            "px": "2.001",
            "sz": "3",
            "instId": "TON-USDT-SWAP",
            "state": "live",
            "ordType": "post_only",
        },
    ])
    wos = working_orders_from_snapshot(fixture)
    assert len(wos) == 1
    side, lvl, wo = wos[0]
    assert side == Side.BUY
    assert lvl == 0
    assert wo.order_id_exchange == 3579105000000000001
    assert wo.client_order_id == "abc123"
    assert wo.price == 2.001
    assert wo.size == 3.0
    assert wo.post_only is True


def test_phase6a_framework_assigns_inside_rung_by_price() -> None:
    """For BID, higher price = inside (lvl 0). For ASK, lower price = inside.
    The framework synthesizes level_idx from per-side price sort."""
    fixture = _make_inline_fixture(open_orders_raw=[
        # BID side: 2.000 (outer) and 2.001 (inside)
        {"ordId": "1", "clOrdId": "b_outer", "side": "buy",
         "px": "2.000", "sz": "3", "instId": "TON-USDT-SWAP", "ordType": "post_only"},
        {"ordId": "2", "clOrdId": "b_inside", "side": "buy",
         "px": "2.001", "sz": "3", "instId": "TON-USDT-SWAP", "ordType": "post_only"},
        # ASK side: 2.005 (outer) and 2.003 (inside)
        {"ordId": "3", "clOrdId": "a_outer", "side": "sell",
         "px": "2.005", "sz": "3", "instId": "TON-USDT-SWAP", "ordType": "post_only"},
        {"ordId": "4", "clOrdId": "a_inside", "side": "sell",
         "px": "2.003", "sz": "3", "instId": "TON-USDT-SWAP", "ordType": "post_only"},
    ])
    wos = working_orders_from_snapshot(fixture)
    by_oid = {wo.order_id_exchange: (side, lvl, wo) for side, lvl, wo in wos}
    # BID: 2.001 should be inside (lvl 0)
    assert by_oid[2][1] == 0  # b_inside
    assert by_oid[1][1] == 1  # b_outer
    # ASK: 2.003 should be inside (lvl 0)
    assert by_oid[4][1] == 0  # a_inside
    assert by_oid[3][1] == 1  # a_outer


def test_phase6a_framework_skips_orders_with_invalid_side() -> None:
    fixture = _make_inline_fixture(open_orders_raw=[
        {"ordId": "1", "clOrdId": "x", "side": "weird",
         "px": "2.0", "sz": "1", "instId": "TON-USDT-SWAP"},
        {"ordId": "2", "clOrdId": "y", "side": "buy",
         "px": "2.0", "sz": "1", "instId": "TON-USDT-SWAP"},
    ])
    wos = working_orders_from_snapshot(fixture)
    # Only the valid 'buy' row.
    assert len(wos) == 1
    assert wos[0][2].order_id_exchange == 2


def test_phase6a_framework_raises_on_malformed_order() -> None:
    """A row missing required fields (ordId, px, sz) should raise so
    the operator notices a schema change."""
    fixture = _make_inline_fixture(open_orders_raw=[
        {"clOrdId": "x", "side": "buy", "px": "2.0", "sz": "1"},  # no ordId
    ])
    with pytest.raises(ValueError, match="failed to reconstruct"):
        working_orders_from_snapshot(fixture)


# ---------------------------------------------------------------------------
# position_from_snapshot
# ---------------------------------------------------------------------------


def test_phase6a_framework_position_defaults_to_flat() -> None:
    """If state_dict has no position fields, return a flat-zero
    PositionSnapshot. Defensive against snapshots taken pre-bot-startup."""
    fixture = _make_inline_fixture(state_dict={"symbol": "TON-USDT-SWAP"})
    pos = position_from_snapshot(fixture)
    assert pos.position_qty == 0.0
    assert pos.position_notional == 0.0


def test_phase6a_framework_position_reads_flat_keys_or_nested() -> None:
    """State recorder may put position fields at top-level or nested
    under 'position'. Both shapes should work."""
    # Top-level
    fixture = _make_inline_fixture(state_dict={
        "symbol": "TON-USDT-SWAP",
        "position_qty": 2.5,
        "position_notional": 5.0,
    })
    pos = position_from_snapshot(fixture)
    assert pos.position_qty == 2.5
    assert pos.position_notional == 5.0

    # Nested
    fixture2 = _make_inline_fixture(state_dict={
        "symbol": "TON-USDT-SWAP",
        "position": {"position_qty": -1.0, "position_notional": 2.0},
    })
    pos2 = position_from_snapshot(fixture2)
    assert pos2.position_qty == -1.0
    assert pos2.position_notional == 2.0


# ---------------------------------------------------------------------------
# market_from_snapshot
# ---------------------------------------------------------------------------


def test_phase6a_framework_market_returns_none_when_absent() -> None:
    fixture = _make_inline_fixture(state_dict={"symbol": "TON-USDT-SWAP"})
    assert market_from_snapshot(fixture) is None


def test_phase6a_framework_market_reconstructs_bba() -> None:
    fixture = _make_inline_fixture(state_dict={
        "symbol": "TON-USDT-SWAP",
        "best_bid": 2.000,
        "best_ask": 2.002,
        "mid_price": 2.001,
    })
    market = market_from_snapshot(fixture)
    assert market is not None
    assert market.best_bid == 2.000
    assert market.best_ask == 2.002
    assert market.mid_price == 2.001


# ---------------------------------------------------------------------------
# Invariant predicates — operating on SnapshotFixture, no filesystem needed
# ---------------------------------------------------------------------------


def test_phase6a_framework_invariant_wedge_counters_pass_when_zero() -> None:
    fixture = _make_inline_fixture(state_dict={
        "symbol": "TON-USDT-SWAP",
        "executor_state": {
            "ws_event_unmatched_to_local_wo_total": 0,
            "gate_phase2a_invariant_violation_total": 0,
        },
    })
    ok, reason = invariant_no_wedge_counters_nonzero(fixture)
    assert ok, f"expected pass; got {reason}"


def test_phase6a_framework_invariant_wedge_counters_fail_when_nonzero() -> None:
    fixture = _make_inline_fixture(state_dict={
        "symbol": "TON-USDT-SWAP",
        "executor_state": {
            "ws_event_unmatched_to_local_wo_total": 4,
            "gate_phase2a_invariant_violation_total": 0,
        },
    })
    ok, reason = invariant_no_wedge_counters_nonzero(fixture)
    assert not ok
    assert "ws_event_unmatched_to_local_wo_total" in reason


def test_phase6a_framework_invariant_risk_state_pass_when_normal() -> None:
    for state_val in ("NORMAL", "UNKNOWN", ""):
        fixture = _make_inline_fixture(state_dict={
            "symbol": "TON-USDT-SWAP",
            "executor_state": {"risk_exec_state": state_val},
        })
        ok, reason = invariant_risk_state_normal(fixture)
        assert ok, f"expected pass for risk_exec_state={state_val!r}; got {reason}"


def test_phase6a_framework_invariant_risk_state_fail_when_suppressed() -> None:
    for bad in ("SUPPRESSED", "CANCELLING", "DEGRADED"):
        fixture = _make_inline_fixture(state_dict={
            "symbol": "TON-USDT-SWAP",
            "executor_state": {"risk_exec_state": bad},
        })
        ok, reason = invariant_risk_state_normal(fixture)
        assert not ok
        assert bad in reason


def test_phase6a_framework_invariant_desync_phase_pass_when_ok() -> None:
    for dp in ("OK", "", None):
        sd = {"symbol": "TON-USDT-SWAP"}
        if dp is not None:
            sd["desync_phase"] = dp
        fixture = _make_inline_fixture(state_dict=sd)
        ok, reason = invariant_desync_phase_ok(fixture)
        assert ok, f"expected pass for desync_phase={dp!r}; got {reason}"


def test_phase6a_framework_invariant_desync_phase_fail_when_recovering() -> None:
    for bad in ("DETECTED", "RECONCILING", "UNRECOVERABLE"):
        fixture = _make_inline_fixture(state_dict={
            "symbol": "TON-USDT-SWAP",
            "desync_phase": bad,
        })
        ok, reason = invariant_desync_phase_ok(fixture)
        assert not ok
        assert bad in reason.upper()


# ---------------------------------------------------------------------------
# load_snapshot — error paths (no fixture data needed)
# ---------------------------------------------------------------------------


def test_phase6a_framework_load_raises_on_nonexistent_path() -> None:
    with pytest.raises(FileNotFoundError):
        load_snapshot("/path/that/does/not/exist/anywhere")


def test_phase6a_framework_load_raises_on_missing_meta_json(tmp_path) -> None:
    """If the directory exists but has no meta.json, raise loud
    rather than silently fall through."""
    empty_dir = tmp_path / "fake_snapshot_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        load_snapshot(empty_dir)


def test_phase6a_framework_load_handles_minimal_valid_snapshot(tmp_path) -> None:
    """Construct a minimal valid snapshot directory with just
    meta.json and verify the loader returns a usable fixture
    (state/orders default to empty)."""
    snap = tmp_path / "v1.4.93-test"
    snap.mkdir()
    (snap / "meta.json").write_text(json.dumps({
        "bot_version": "1.4.93-test",
        "bot_profile": "prod.okx.ton.usdt.perp",
        "captured_at_utc": "2026-05-19T12:00:00Z",
    }))
    (snap / "stats").mkdir()
    # No JSON files in stats/ — loader should tolerate.

    fixture = load_snapshot(snap)
    assert fixture.bot_version == "1.4.93-test"
    assert fixture.bot_profile == "prod.okx.ton.usdt.perp"
    assert fixture.state_dict == {}
    assert fixture.open_orders_raw == []


def test_phase6a_framework_load_parses_open_orders_data_array(tmp_path) -> None:
    """OKX open-orders payload is wrapped in {'code':..., 'data': [...]}.
    The loader extracts the data array."""
    snap = tmp_path / "v1.4.93-orders"
    (snap / "stats").mkdir(parents=True)
    (snap / "meta.json").write_text(json.dumps({
        "bot_version": "1.4.93", "bot_profile": "p",
        "captured_at_utc": "2026-05-19T12:00:00Z",
    }))
    (snap / "stats" / "open_orders.json").write_text(json.dumps({
        "code": "0", "msg": "",
        "data": [
            {"ordId": "1", "side": "buy", "px": "2.0", "sz": "1",
             "instId": "TON-USDT-SWAP", "clOrdId": "x"},
        ],
    }))
    fixture = load_snapshot(snap)
    assert len(fixture.open_orders_raw) == 1
    assert fixture.open_orders_raw[0]["ordId"] == "1"
