"""Phase 4D.4 (v1.4.173) — SF phase-ladder observability.

Three behaviours pinned end-to-end:

1. **Migration v40** adds three columns:
   - ``fills.sf_force_phase`` (integer, nullable)
   - ``soft_flatten_events.fills_by_phase_json`` (text, JSON)
   - ``soft_flatten_events.taker_spread_bps_paid`` (real, notional-
     weighted average cross-spread cost in bps for non-passive phases)

2. **Storage rollup** (``compute_sf_episode_phase_totals``) computes
   the per-phase fill counts + the notional-weighted taker spread
   paid for an SF episode, treating NULL phases as phase 0 (legacy /
   ladder-disabled episodes).

3. **Synthetic ORDERS rows** for the IOC + market_close paths so the
   dashboard's ORDERS list reflects every SF placement (previously
   fire-and-forget calls bypassed the orders table entirely).

The fill-ingestion ``sf_force_phase`` stamp is exercised
indirectly via ``fill_row`` (the row dict produced by ingest is
the persistence contract; the stamp value itself is just a field
copy from ``state.sf_phase_ladder_phase``).
"""

from __future__ import annotations

import inspect
import json
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

import pytest

from app.fill_ingestion import fill_row
from app.models import Fill
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_sfp4d4_{os.getpid()}_{uuid.uuid4().hex}.db"
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


def _open_db(storage: Storage) -> sqlite3.Connection:
    db_path = storage._settings.database_url.split("sqlite:///", 1)[-1]
    return sqlite3.connect(db_path)


# ---------------------------------------------------------------------------
# Migration v40
# ---------------------------------------------------------------------------


def test_schema_v40_adds_sf_phase_columns() -> None:
    s = _settings()
    Storage(s).init_schema()
    conn = sqlite3.connect(s.database_url.split("sqlite:///", 1)[-1])
    try:
        v = conn.execute("PRAGMA user_version").fetchone()[0]
        assert v >= 40
        fill_cols = {r[1] for r in conn.execute("PRAGMA table_info(fills)")}
        assert "sf_force_phase" in fill_cols
        ev_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(soft_flatten_events)")
        }
        assert "fills_by_phase_json" in ev_cols
        assert "taker_spread_bps_paid" in ev_cols
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Rollup: compute_sf_episode_phase_totals
# ---------------------------------------------------------------------------


def _make_episode(
    storage: Storage,
    *,
    entry_mid: float = 2.040,
    entry_qty: float = -9.0,
) -> int:
    return storage.insert_soft_flatten_event(
        {
            "ts_start": "2026-05-21T07:00:00+00:00",
            "trigger_reason": "test",
            "entry_position_qty": entry_qty,
            "entry_mid_price": entry_mid,
        }
    )


def _insert_fill(
    storage: Storage,
    *,
    fill_id: str,
    ev_id: int,
    price: float,
    size: float,
    phase: int | None,
) -> None:
    storage.insert_fill_row(
        {
            "fill_id": fill_id,
            "order_id_exchange": f"oid-{fill_id}",
            "ts_fill": "2026-05-21T07:00:01+00:00",
            "symbol": "TON-USDT-SWAP",
            "side": "BUY",
            "price": price,
            "size": size,
            "notional": price * size,
            "fee": -0.0001,
            "liquidity_flag": (
                "resting" if phase is not None and phase <= 1 else "crossed"
            ),
            "soft_flatten_event_id": ev_id,
            "sf_force_phase": phase,
        }
    )


def test_rollup_counts_fills_by_phase() -> None:
    """Mixed-phase episode: 3 × p0, 1 × p1, 2 × p2, 1 × p4.
    Rollup returns the right counts; phases with no fills come back
    as 0."""
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = _make_episode(storage)
    for i in range(3):
        _insert_fill(
            storage, fill_id=f"p0-{i}", ev_id=ev_id,
            price=2.040, size=1.0, phase=0,
        )
    _insert_fill(
        storage, fill_id="p1", ev_id=ev_id,
        price=2.041, size=1.0, phase=1,
    )
    for i in range(2):
        _insert_fill(
            storage, fill_id=f"p2-{i}", ev_id=ev_id,
            price=2.052, size=1.0, phase=2,
        )
    _insert_fill(
        storage, fill_id="p4", ev_id=ev_id,
        price=2.059, size=1.0, phase=4,
    )

    out = storage.compute_sf_episode_phase_totals(ev_id)
    counts = out["fills_by_phase"]
    assert counts == {"0": 3, "1": 1, "2": 2, "3": 0, "4": 1}


def test_rollup_treats_null_phase_as_phase_0() -> None:
    """Legacy fills (no ladder, sf_force_phase=NULL) bucket as phase
    0 — back-compat with pre-v1.4.172 episodes whose fills don't
    carry a phase but came from the legacy near-touch worker."""
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = _make_episode(storage)
    _insert_fill(
        storage, fill_id="legacy", ev_id=ev_id,
        price=2.040, size=1.0, phase=None,
    )
    out = storage.compute_sf_episode_phase_totals(ev_id)
    assert out["fills_by_phase"]["0"] == 1


def test_rollup_taker_spread_bps_paid_weights_by_notional() -> None:
    """Three taker fills: 1×p2 @ 2.052 (12 bp), 1×p3 @ 2.054 (≈ 19.6 bp),
    1×p4 @ 2.060 (≈ 49 bp). Notional-weighted average against entry
    mid 2.040, sizes 1.0 each so each fill weights equally → average
    of the per-fill bps.

    Computed in SQL by the rollup; reference values here from the
    same formula: ABS(price - entry_mid) / entry_mid * 10000."""
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = _make_episode(storage, entry_mid=2.040)
    _insert_fill(
        storage, fill_id="p0", ev_id=ev_id,
        price=2.040, size=1.0, phase=0,  # passive, not counted
    )
    _insert_fill(
        storage, fill_id="p2", ev_id=ev_id,
        price=2.052, size=1.0, phase=2,  # 58.82 bp
    )
    _insert_fill(
        storage, fill_id="p3", ev_id=ev_id,
        price=2.054, size=1.0, phase=3,  # 68.63 bp
    )
    _insert_fill(
        storage, fill_id="p4", ev_id=ev_id,
        price=2.060, size=1.0, phase=4,  # 98.04 bp
    )
    out = storage.compute_sf_episode_phase_totals(ev_id)
    # Per-fill bps: ABS(p - 2.040) / 2.040 * 10000
    # p2: 58.823..., p3: 68.627..., p4: 98.039...
    # Notionals: 2.052, 2.054, 2.060; weights ≈ equal
    expected_num = (
        2.052 * abs(2.052 - 2.040) / 2.040 * 10000
        + 2.054 * abs(2.054 - 2.040) / 2.040 * 10000
        + 2.060 * abs(2.060 - 2.040) / 2.040 * 10000
    )
    expected_den = 2.052 + 2.054 + 2.060
    expected = expected_num / expected_den
    assert out["taker_spread_bps_paid"] == pytest.approx(expected, rel=1e-4)


def test_rollup_taker_spread_none_when_no_taker_fills() -> None:
    """Pure post-only episode → no taker fills → taker_spread_bps_paid
    is None (dashboard renders the legacy SF tooltip without the
    'taker spread paid' line)."""
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = _make_episode(storage)
    _insert_fill(
        storage, fill_id="p0a", ev_id=ev_id,
        price=2.040, size=1.0, phase=0,
    )
    _insert_fill(
        storage, fill_id="p1a", ev_id=ev_id,
        price=2.041, size=1.0, phase=1,
    )
    out = storage.compute_sf_episode_phase_totals(ev_id)
    assert out["taker_spread_bps_paid"] is None


def test_rollup_empty_episode_returns_zero_counts() -> None:
    """No fills attributed → all phases at 0, taker spread None.
    Idempotent / safe to call on episodes that never landed a fill."""
    storage = Storage(_settings())
    storage.init_schema()
    ev_id = _make_episode(storage)
    out = storage.compute_sf_episode_phase_totals(ev_id)
    assert out["fills_by_phase"] == {
        "0": 0, "1": 0, "2": 0, "3": 0, "4": 0,
    }
    assert out["taker_spread_bps_paid"] is None


# ---------------------------------------------------------------------------
# Fill row contract
# ---------------------------------------------------------------------------


def test_fill_row_includes_sf_force_phase() -> None:
    """The persistence dict produced by ``fill_row`` MUST include
    ``sf_force_phase`` so the new column is populated on insert."""
    from datetime import datetime, timezone
    from app.enums import Side
    f = Fill(
        fill_id="x",
        order_id_exchange="0",
        client_order_id=None,
        ts_fill=datetime(2026, 5, 21, 7, 0, 0, tzinfo=timezone.utc),
        symbol="TEST",
        side=Side.BUY,
        price=2.05,
        size=1.0,
        notional=2.05,
        fee=-0.0001,
        liquidity_flag="crossed",
        mid_at_fill=2.05,
        sf_force_phase=2,
    )
    row = fill_row(f)
    assert "sf_force_phase" in row
    assert row["sf_force_phase"] == 2


def test_fill_row_sf_force_phase_defaults_to_none() -> None:
    """Default-constructed Fill → sf_force_phase stays None (column
    NULL in DB)."""
    from datetime import datetime, timezone
    from app.enums import Side
    f = Fill(
        fill_id="y",
        order_id_exchange="0",
        client_order_id=None,
        ts_fill=datetime(2026, 5, 21, 7, 0, 0, tzinfo=timezone.utc),
        symbol="TEST",
        side=Side.BUY,
        price=2.05,
        size=1.0,
        notional=2.05,
        fee=-0.0001,
        liquidity_flag="resting",
        mid_at_fill=2.05,
    )
    row = fill_row(f)
    assert row["sf_force_phase"] is None


# ---------------------------------------------------------------------------
# Synthetic ORDERS rows for IOC + market_close
# ---------------------------------------------------------------------------


def test_dispatcher_persists_synthetic_orders_row_for_ioc() -> None:
    """Source contract: the IOC branch of the phase-ladder dispatcher
    writes a stub orders row BEFORE firing the IOC, so the ORDERS
    history captures the placement even when the venue fills + the
    fill-ingestion runs before the row write would otherwise land."""
    from app.bot import Bot

    src = inspect.getsource(Bot._run_sf_phase_ladder_dispatch)
    # IOC path persists with the limit price (post_only=False).
    assert '_persist_synthetic_sf_order_row' in src
    assert 'order_kind="ioc"' in src


def test_dispatcher_persists_synthetic_orders_row_for_market_close() -> None:
    """Phase-4 terminal market_close ALSO gets a stub orders row so
    the ORDERS list reflects the close. (Without this, the dashboard
    sees a fill out of nowhere with no parent order.)"""
    from app.bot import Bot

    src = inspect.getsource(Bot._run_sf_phase_ladder_dispatch)
    assert 'order_kind="market_close"' in src


def test_legacy_taker_fallback_persists_synthetic_orders_row() -> None:
    """The legacy 2-phase post-only worker's adverse-drift fallback
    ALSO writes a stub orders row (Phase 4D.4 extends the fix to the
    pre-ladder path so v1.4.163 SF tagging on the FILLS table has a
    matching ORDERS row for the dashboard's per-episode views)."""
    from app.bot import Bot

    src = inspect.getsource(Bot._run_soft_flatten_tick)
    assert 'market_close_legacy_drift' in src


def test_synthetic_row_helper_signature_is_stable() -> None:
    """Sentinel: the synthetic-row helper exists with the documented
    parameter set. A refactor that drops it will break the ORDERS-
    list attribution + needs to be flagged."""
    from app.bot import Bot

    assert hasattr(Bot, "_persist_synthetic_sf_order_row")
    sig = inspect.signature(Bot._persist_synthetic_sf_order_row)
    for p in ("phase", "side", "price", "size", "post_only", "order_kind"):
        assert p in sig.parameters, f"missing param {p}"
