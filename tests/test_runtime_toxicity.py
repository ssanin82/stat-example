from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router
from app.enums import Side
from app.models import Fill
from app.runtime_toxicity_aggregate import build_runtime_toxicity_summary
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _fill(
    fid: str,
    side: Side,
    *,
    ts: datetime,
    m1: float | None = None,
    m3: float | None = None,
    m5: float | None = None,
) -> Fill:
    return Fill(
        fill_id=fid,
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=ts,
        symbol="ETH",
        side=side,
        price=100.0,
        size=0.1,
        notional=10.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
        markout_1s_bps=m1,
        markout_3s_bps=m3,
        markout_5s_bps=m5,
    )


def test_no_fills_summary_is_emptyish() -> None:
    s = build_runtime_toxicity_summary([], window=50, session_id="sid-1")
    assert s["session_id"] == "sid-1"
    assert s["fill_count_in_window"] == 0
    assert s["buy_fill_count"] == 0
    assert s["sell_fill_count"] == 0
    assert s["mean_markout_1s_bps"] is None
    assert s["mean_markout_3s_bps"] is None
    assert s["mean_markout_5s_bps"] is None
    assert s["one_sided_fill_ratio"] is None
    assert s["toxicity_score"] is None
    assert s["last_fill_ts"] is None


def test_balanced_benign_fills_low_toxicity() -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Newest first: index 0 is latest ts
    fills = [
        _fill("d", Side.SELL, ts=t0, m5=5.0),
        _fill("c", Side.BUY, ts=t0 - timedelta(seconds=1), m5=5.0),
        _fill("b", Side.SELL, ts=t0 - timedelta(seconds=2), m5=5.0),
        _fill("a", Side.BUY, ts=t0 - timedelta(seconds=3), m5=5.0),
    ]
    s = build_runtime_toxicity_summary(fills, window=50, session_id="x")
    assert s["fill_count_in_window"] == 4
    assert s["one_sided_fill_ratio"] == 0.5
    assert s["mean_markout_5s_bps"] == 5.0
    assert s["toxicity_score"] is not None
    assert s["toxicity_score"] < 0.05
    assert s["last_fill_ts"] == t0.isoformat()


def test_one_sided_adverse_fills_higher_toxicity() -> None:
    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    fills = [
        _fill("d", Side.SELL, ts=t0 - timedelta(seconds=i), m5=-25.0)
        for i in range(4)
    ]
    s = build_runtime_toxicity_summary(fills, window=50, session_id="x")
    assert s["one_sided_fill_ratio"] == 1.0
    assert s["mean_markout_5s_bps"] == -25.0
    assert s["toxicity_score"] is not None
    assert s["toxicity_score"] > 0.7


def test_score_null_without_enough_fills_or_markouts() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    three = [
        _fill("c", Side.BUY, ts=t0),
        _fill("b", Side.SELL, ts=t0),
        _fill("a", Side.BUY, ts=t0),
    ]
    s = build_runtime_toxicity_summary(three, window=50, session_id="x")
    assert s["fill_count_in_window"] == 3
    assert s["toxicity_score"] is None

    four_no_m = [
        _fill("d", Side.BUY, ts=t0),
        _fill("c", Side.SELL, ts=t0),
        _fill("b", Side.BUY, ts=t0),
        _fill("a", Side.SELL, ts=t0),
    ]
    s2 = build_runtime_toxicity_summary(four_no_m, window=50, session_id="x")
    assert s2["toxicity_score"] is None


def test_toxicity_score_populated_at_two_fills_with_markouts() -> None:
    """With MIN_FILLS_FOR_TOXICITY_SCORE lowered to 2, a session with as few
    as 2 adverse fills shows a non-null toxicity_score to the operator — the
    previous 4-fill gate hid a genuinely adverse session
    (``tmp/snap_20260417_183547``: 3/3 adverse fills, score=null)."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    two_adverse = [
        _fill("b", Side.SELL, ts=t0, m5=-10.0),
        _fill("a", Side.SELL, ts=t0 - timedelta(seconds=1), m5=-10.0),
    ]
    s = build_runtime_toxicity_summary(two_adverse, window=50, session_id="x")
    assert s["fill_count_in_window"] == 2
    assert s["toxicity_score"] is not None
    # Both fills sell + adverse → high one-sided score, meaningful markout.
    assert s["toxicity_score"] > 0.5


def test_toxicity_score_null_at_one_fill() -> None:
    """One fill is too noisy — score must stay null."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    one = [_fill("a", Side.BUY, ts=t0, m5=-10.0)]
    s = build_runtime_toxicity_summary(one, window=50, session_id="x")
    assert s["fill_count_in_window"] == 1
    assert s["toxicity_score"] is None


def test_preferred_markout_prefers_longer_horizon() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fills = [
        _fill("d", Side.BUY, ts=t0, m1=-1.0, m3=-2.0, m5=-10.0),
        _fill("c", Side.SELL, ts=t0, m1=-1.0, m3=-2.0, m5=-10.0),
        _fill("b", Side.BUY, ts=t0, m1=-1.0, m3=-2.0, m5=-10.0),
        _fill("a", Side.SELL, ts=t0, m1=-1.0, m3=-2.0, m5=-10.0),
    ]
    s = build_runtime_toxicity_summary(fills, window=50, session_id="x")
    assert s["toxicity_score"] is not None
    # Per-fill preferred is 5s (-10); means for columns still independent
    assert s["mean_markout_1s_bps"] == -1.0
    assert s["mean_markout_5s_bps"] == -10.0


def test_window_truncates_oldest() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fills = [
        _fill(f"{i}", Side.BUY if i % 2 == 0 else Side.SELL, ts=t0 + timedelta(seconds=i), m5=1.0)
        for i in range(6)
    ]
    fills.reverse()  # newest first (high i first)
    s = build_runtime_toxicity_summary(fills, window=2, session_id="x")
    assert s["fill_count_in_window"] == 2
    assert s["buy_fill_count"] + s["sell_fill_count"] == 2


def _api_client() -> TestClient:
    path = Path(tempfile.gettempdir()) / f"mm_rtox_test_{os.getpid()}.db"
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
    return TestClient(app)


def test_toxicity_current_endpoint_shape() -> None:
    c = _api_client()
    r = c.get("/toxicity/current")
    assert r.status_code == 200
    body = r.json()
    keys = {
        "session_id",
        "fill_count_in_window",
        "mean_markout_1s_bps",
        "mean_markout_3s_bps",
        "mean_markout_5s_bps",
        "buy_fill_count",
        "sell_fill_count",
        "one_sided_fill_ratio",
        "toxicity_score",
        "last_fill_ts",
    }
    assert keys == set(body.keys())
    assert body["fill_count_in_window"] == 0
    assert body["toxicity_score"] is None

    st = c.app.state.bot_state
    t0 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    for i in range(4):
        st.record_fill(
            _fill(
                f"fx{i}",
                Side.BUY if i % 2 == 0 else Side.SELL,
                ts=t0 + timedelta(seconds=i),
                m5=3.0,
            )
        )
    r2 = c.get("/toxicity/current")
    assert r2.status_code == 200
    b2 = r2.json()
    assert b2["fill_count_in_window"] == 4
    assert b2["toxicity_score"] is not None

    snap = c.get("/state/current").json()
    assert snap["toxicity_score"] == b2["toxicity_score"]
