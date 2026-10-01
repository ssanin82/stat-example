"""Integration tests for the new analytics endpoints.

- ``GET /fills/since``: session-scoped unbounded-ish fill history.
- ``GET /pnl/attribution``: fee / markout / residual decomposition.
- ``GET /session/summary``: one-shot session rollup.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router
from app.enums import Side
from app.models import Fill
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _client() -> tuple[TestClient, BotState, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_analytics_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    app = FastAPI()
    app.include_router(router)
    app.state.settings = settings
    app.state.bot_state = state
    app.state.storage = storage
    return TestClient(app), state, storage


def _make_fill(
    idx: int,
    side: Side,
    *,
    ts: datetime,
    notional: float = 25.0,
    fee: float = -3.5e-5,
    markout_5s: float | None = -2.0,
) -> Fill:
    return Fill(
        fill_id=f"f_{idx}",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=ts,
        symbol="ETH_USDT_Perp",
        side=side,
        price=2500.0,
        size=notional / 2500.0,
        notional=notional,
        fee=fee,
        liquidity_flag="resting",
        mid_at_fill=2500.0,
        markout_1s_bps=markout_5s,
        markout_3s_bps=markout_5s,
        markout_5s_bps=markout_5s,
    )


def _persist_fill(storage: Storage, f: Fill) -> None:
    """Persist a :class:`Fill` to SQLite via the same path the ingestion
    pipeline uses (column-explicit INSERT)."""
    storage.insert_fill_row(
        {
            "fill_id": f.fill_id,
            "order_id_exchange": str(f.order_id_exchange) if f.order_id_exchange else None,
            "client_order_id": f.client_order_id,
            "ts_fill": f.ts_fill.isoformat(),
            "symbol": f.symbol,
            "side": f.side.value,
            "price": f.price,
            "size": f.size,
            "notional": f.notional,
            "fee": f.fee,
            "liquidity_flag": f.liquidity_flag,
            "mid_at_fill": f.mid_at_fill,
            "best_bid_at_fill": None,
            "best_ask_at_fill": None,
            "book_snapshot_quality": "full",
            "markout_1s_bps": f.markout_1s_bps,
            "markout_3s_bps": f.markout_3s_bps,
            "markout_5s_bps": f.markout_5s_bps,
            "book_reference_quality": "exact_or_prior",
        }
    )


# ------------------- /fills/since ----------------------


def test_fills_since_defaults_to_session_start() -> None:
    c, state, storage = _client()
    # Seed one fill BEFORE the session starts — it must NOT be returned.
    # Seed three fills AFTER session start.
    session_start = state.session_started_at_utc
    before = session_start - timedelta(minutes=30)
    after1 = session_start + timedelta(seconds=30)
    after2 = session_start + timedelta(minutes=1)
    after3 = session_start + timedelta(minutes=2)

    _persist_fill(storage, _make_fill(0, Side.BUY, ts=before))
    _persist_fill(storage, _make_fill(1, Side.BUY, ts=after1))
    _persist_fill(storage, _make_fill(2, Side.SELL, ts=after2))
    _persist_fill(storage, _make_fill(3, Side.BUY, ts=after3))

    r = c.get("/fills/since")
    assert r.status_code == 200
    rows = r.json()
    # Only the 3 fills after session start.
    assert len(rows) == 3
    # Oldest-first.
    assert rows[0]["fill_id"] == "f_1"
    assert rows[1]["fill_id"] == "f_2"
    assert rows[2]["fill_id"] == "f_3"


def test_fills_since_explicit_ts_overrides_default() -> None:
    c, state, storage = _client()
    start = state.session_started_at_utc
    _persist_fill(storage, _make_fill(1, Side.BUY, ts=start + timedelta(seconds=10)))
    _persist_fill(storage, _make_fill(2, Side.SELL, ts=start + timedelta(minutes=5)))
    _persist_fill(storage, _make_fill(3, Side.SELL, ts=start + timedelta(minutes=10)))
    # since halfway through → only f_2 and f_3.
    cutoff = (start + timedelta(minutes=3)).isoformat()
    r = c.get(f"/fills/since?ts={cutoff}")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert [row["fill_id"] for row in rows] == ["f_2", "f_3"]


def test_fills_since_limit_is_respected() -> None:
    c, state, storage = _client()
    start = state.session_started_at_utc
    for i in range(25):
        _persist_fill(storage, 
            _make_fill(i, Side.BUY, ts=start + timedelta(seconds=i))
        )
    r = c.get("/fills/since?limit=10")
    assert r.status_code == 200
    assert len(r.json()) == 10


# ------------------- /pnl/attribution ----------------------


def test_pnl_attribution_session_default_returns_expected_structure() -> None:
    c, state, storage = _client()
    start = state.session_started_at_utc
    # 4 fills at ~$25 notional, -2 bps markout, rebate of 3.5e-5 each.
    for i in range(4):
        _persist_fill(storage, 
            _make_fill(i, Side.BUY if i % 2 == 0 else Side.SELL, ts=start + timedelta(seconds=i))
        )
    # Bump realized PnL to simulate a partial losing session.
    state.pnl.realized_pnl_usd = -0.02

    r = c.get("/pnl/attribution")
    assert r.status_code == 200
    out = r.json()
    assert out["horizon_s"] == 5
    assert out["fills"]["total"] == 4
    assert out["fills"]["buy_count"] == 2
    assert out["fills"]["sell_count"] == 2
    # markout_dollar ≈ 4 * -2 * 25 / 10000 = -0.02
    assert out["markout"]["dollar_impact_usd"] < -0.01
    # rebate ≈ 4 * 3.5e-5 = 0.00014
    assert out["fees"]["rebate_earned_usd"] > 0
    assert out["pnl_attribution_usd"]["realized_pnl_total"] == -0.02
    assert out["pnl_attribution_usd"]["residual"] is not None


def test_pnl_attribution_rejects_invalid_horizon() -> None:
    c, _, _ = _client()
    r = c.get("/pnl/attribution?horizon_s=7")
    assert r.status_code == 400


def test_pnl_attribution_horizons_accepts_1_3_5() -> None:
    c, state, storage = _client()
    start = state.session_started_at_utc
    _persist_fill(storage, _make_fill(1, Side.BUY, ts=start + timedelta(seconds=1)))
    for h in (1, 3, 5):
        r = c.get(f"/pnl/attribution?horizon_s={h}")
        assert r.status_code == 200
        assert r.json()["horizon_s"] == h


# ------------------- /session/summary ----------------------


def test_session_summary_returns_rollup() -> None:
    c, state, storage = _client()
    start = state.session_started_at_utc
    for i in range(5):
        _persist_fill(storage, 
            _make_fill(i, Side.BUY if i % 2 == 0 else Side.SELL, ts=start + timedelta(seconds=i))
        )
    state.pnl.realized_pnl_usd = -0.05

    r = c.get("/session/summary")
    assert r.status_code == 200
    out = r.json()
    assert out["session_id"] == state.session_id
    assert out["throughput"]["fills_count"] == 5
    assert out["attribution"]["fills"]["total"] == 5
    assert out["attribution"]["pnl_attribution_usd"]["realized_pnl_total"] == -0.05
    assert out["duration_seconds"] is not None
    assert "key_counters" in out


def test_session_summary_key_counters_include_session_fill_counts() -> None:
    """After :meth:`record_fill`, the monotonic counters should show up in
    ``key_counters`` so operators can see session throughput without reading
    the DB."""
    c, state, storage = _client()
    start = state.session_started_at_utc
    # Record a fill through BotState so ``session_fill_count`` advances.
    state.record_fill(_make_fill(1, Side.BUY, ts=start + timedelta(seconds=1)))
    state.record_fill(_make_fill(2, Side.SELL, ts=start + timedelta(seconds=2)))
    state.record_fill(_make_fill(3, Side.BUY, ts=start + timedelta(seconds=3)))
    # Also persist to the DB so /fills/since returns something.
    for i, side in ((1, Side.BUY), (2, Side.SELL), (3, Side.BUY)):
        _persist_fill(storage, _make_fill(i, side, ts=start + timedelta(seconds=i)))

    r = c.get("/session/summary")
    assert r.status_code == 200
    out = r.json()
    assert out["key_counters"]["session_fill_count"] == 3
    by_side = out["key_counters"]["session_fill_count_by_side"]
    assert by_side["BUY"] == 2
    assert by_side["SELL"] == 1


def test_session_summary_empty_session_is_valid_json() -> None:
    """Freshly-started session with no fills yet — should return zeros, not 500."""
    c, _, _ = _client()
    r = c.get("/session/summary")
    assert r.status_code == 200
    out = r.json()
    assert out["throughput"]["fills_count"] == 0
    assert out["attribution"]["fills"]["total"] == 0
    assert out["attribution"]["pnl_attribution_usd"]["markout_dollar_impact"] == 0.0


# ------------------- URL registration ----------------------


def test_monitor_get_urls_includes_new_endpoints() -> None:
    """The monitoring bundle picks up the new endpoints."""
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from monitor_get_urls import URLS

    assert "/fills/since" in URLS
    assert "/pnl/attribution" in URLS
    assert "/session/summary" in URLS
