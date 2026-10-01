"""Market-data update-to-update gap metrics (in-memory tracker + API)."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router
from app.market_data_gap_stats import MarketDataGapTracker
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _make_settings(**extra: object) -> UnitTestSettings:
    path = Path(tempfile.gettempdir()) / f"mm_gap_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    }
    base.update(extra)
    return UnitTestSettings.model_validate(base)


def test_first_update_no_gap_second_has_gap() -> None:
    s = _make_settings()
    tr = MarketDataGapTracker(256)
    t0 = 1000.0
    tr.note_successful_update(
        perf_now=t0,
        wall_now=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        source="test",
        symbol="ETH",
        recovery_state="OK",
        settings=s,
        storage=None,
        session_id="sid",
    )
    d = tr.to_api_dict(session_id="sid", symbol="ETH")
    assert d["update_count"] == 1
    assert d["gap_count"] == 0
    assert d["last_gap_ms"] is None
    assert d["mean_gap_ms"] is None
    assert d["first_update_ts"] is not None
    tr.note_successful_update(
        perf_now=t0 + 0.1,
        wall_now=datetime(2026, 1, 1, 12, 0, 0, 500000, tzinfo=timezone.utc),
        source="test",
        symbol="ETH",
        recovery_state="OK",
        settings=s,
        storage=None,
        session_id="sid",
    )
    d2 = tr.to_api_dict(session_id="sid", symbol="ETH")
    assert d2["update_count"] == 2
    assert d2["gap_count"] == 1
    assert d2["last_gap_ms"] == 100.0
    assert d2["min_gap_ms"] == 100.0
    assert d2["max_gap_ms"] == 100.0
    assert d2["mean_gap_ms"] == 100.0


def test_min_max_mean_median_p95_known_sequence() -> None:
    s = _make_settings()
    tr = MarketDataGapTracker(256)
    base = 0.0
    gaps = [10.0, 20.0, 30.0, 40.0, 50.0]
    tr.note_successful_update(
        perf_now=base,
        wall_now=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        source="u",
        symbol="ETH",
        recovery_state="OK",
        settings=s,
        storage=None,
        session_id="sid",
    )
    for i, g in enumerate(gaps, start=1):
        base += g / 1000.0
        tr.note_successful_update(
            perf_now=base,
            wall_now=datetime(2026, 1, 1, 12, 0, i, tzinfo=timezone.utc),
            source="u",
            symbol="ETH",
            recovery_state="OK",
            settings=s,
            storage=None,
            session_id="sid",
        )
    out = tr.to_api_dict(session_id="sid", symbol="ETH")
    assert out["gap_count"] == 5
    assert out["min_gap_ms"] == 10.0
    assert out["max_gap_ms"] == 50.0
    assert out["mean_gap_ms"] == 30.0
    assert out["median_gap_ms"] == 30.0
    # n=5 → 95th percentile interpolates between 40 and 50 ms
    assert out["p95_gap_ms"] == 48.0


def test_gap_stats_endpoint_structure() -> None:
    s = _make_settings()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    app = FastAPI()
    app.include_router(router)
    app.state.settings = s
    app.state.bot_state = state
    app.state.storage = storage
    c = TestClient(app)
    r = c.get("/market-data/gap-stats")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == state.session_id
    assert body["symbol"] == s.symbol
    assert "quantiles_note" in body
    assert body["update_count"] == 0
    assert body["gap_count"] == 0
    assert body["min_gap_ms"] is None


def test_persist_sample_inserts_and_prunes() -> None:
    path = Path(tempfile.gettempdir()) / f"mm_gap_p_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MARKET_DATA_GAP_PERSIST_SAMPLES": True,
            "MARKET_DATA_GAP_PERSIST_MAX_ROWS": 100,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    tr = MarketDataGapTracker(64)
    # 102 applies → 101 gaps → prune to 100 rows for this session
    for i in range(102):
        tr.note_successful_update(
            perf_now=float(i) * 0.001,
            wall_now=datetime(2026, 1, 1, 12, 0, 0, i * 1000, tzinfo=timezone.utc),
            source="t",
            symbol="ETH",
            recovery_state="OK",
            settings=s,
            storage=storage,
            session_id="sess-a",
        )
    with storage.connection() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM market_data_gap_samples WHERE session_id = ?",
            ("sess-a",),
        ).fetchone()[0]
    assert int(n) == 100
    path.unlink(missing_ok=True)
