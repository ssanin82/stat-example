"""API endpoint tests for Binance WS one-way latency measurement.

Tests the two routes that expose the Binance tracker:

  * ``GET /market-data/binance-timing-summary``
  * ``GET /market-data/binance-timing-recent``

These land in ``scripts/stats_snapshot.py`` captures as
``market-data_binance-timing-summary.json`` +
``market-data_binance-timing-recent.json``, which is how the operator
will read the one-way latency after a live session.

Invariants pinned:

  1. When BINANCE_WS_ENABLED=true and the timing tracker has samples,
     summary returns ``enabled: true`` with all the histogram metrics
     that match the GRVT ``/market-data/timing-summary`` contract.
  2. ``source_type`` is explicitly ``binance_public_ws`` (not the
     tracker's default ``public_ws``) so a snapshot folder can
     distinguish the two venues.
  3. When BINANCE_WS_ENABLED=false, both routes return ``enabled:
     false`` with a clear ``notes`` string — rather than empty /
     400 — so snapshot consumers see a definitive "feature disabled"
     marker in the JSON rather than silent absence.
  4. The recent-samples endpoint honours ``limit`` and clamps to the
     configured max.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _client(**settings_overrides) -> TestClient:
    path = Path(tempfile.gettempdir()) / f"mm_bn_ts_{os.getpid()}_{uuid.uuid4().hex}.db"
    data: dict = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "SYMBOL": "ETH",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        # Defaults for the two flags the new endpoints test.
        "BINANCE_WS_ENABLED": True,
        "BINANCE_SYMBOL": "ETHUSDT",
        "MARKET_DATA_TIMING_WINDOW_ENABLED": True,
    }
    for k, v in settings_overrides.items():
        data[k.upper() if k.islower() else k] = v
    settings = UnitTestSettings.model_validate(data)
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    app = FastAPI()
    app.include_router(router)
    app.state.settings = settings
    app.state.bot_state = state
    app.state.storage = storage
    return TestClient(app)


def _seed_samples(tc: TestClient, n: int = 5) -> None:
    """Push ``n`` synthetic samples into the Binance tracker so the
    summary is non-empty. Not going through ``BinancePublicStream``
    (which needs a live websocket) — calling tracker directly."""
    state = tc.app.state.bot_state
    tr = state.binance_public_ws_timing
    assert tr is not None, "test fixture must enable the tracker"
    now_ms = int(time.time() * 1000.0)
    for i in range(n):
        wall_ms = now_ms + i * 100
        exch_ms = wall_ms - 40  # synthetic 40 ms one-way
        tr.ingest(
            local_receive_wall_ms=wall_ms,
            local_receive_mono_ns=(wall_ms * 1_000_000),
            exchange_ts_ms=exch_ms,
            local_apply_wall_ms=wall_ms + 1,
            local_apply_mono_ns=(wall_ms * 1_000_000 + 1_000_000),
            seq=i,
        )


# --------------------- Summary endpoint ---------------------


def test_binance_timing_summary_enabled_returns_metrics() -> None:
    tc = _client(BINANCE_WS_ENABLED=True)
    _seed_samples(tc, n=5)
    r = tc.get("/market-data/binance-timing-summary")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["enabled"] is True
    assert body["source_type"] == "binance_public_ws"
    # Symbol is the Binance-side one, not GRVT's.
    assert body["symbol"] == "ETHUSDT"
    # Must carry the one-way-delay metric that the whole endpoint exists for.
    assert body["total_samples"] == 5
    assert body["samples_with_exchange_ts"] == 5
    # Histograms present (tracker populates with real numbers).
    for field in (
        "exchange_gap_ms", "local_receive_gap_ms",
        "exchange_to_local_receive_ms", "receive_to_apply_ms",
    ):
        assert field in body, f"summary missing metric: {field}"
    # Operator-facing explanation strings are present so someone
    # scanning the JSON knows what each metric means.
    assert "notes" in body
    assert "exchange_to_local_receive_ms" in body["notes"]


def test_binance_timing_summary_disabled_when_binance_ws_off() -> None:
    """When BINANCE_WS_ENABLED=false the route still returns 200 with
    a definitive ``enabled: false`` marker. Crucial — silent absence
    would make the snapshot look complete when it's missing data."""
    tc = _client(BINANCE_WS_ENABLED=False)
    r = tc.get("/market-data/binance-timing-summary")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["source_type"] == "binance_public_ws"
    assert "BINANCE_WS_ENABLED" in body.get("notes", "")


def test_binance_timing_summary_disabled_when_timing_window_off() -> None:
    """Timing tracker itself can be disabled at the framework level
    via MARKET_DATA_TIMING_WINDOW_ENABLED=false. Endpoint reports this
    distinctly from the 'feature off' case."""
    tc = _client(
        BINANCE_WS_ENABLED=True,
        MARKET_DATA_TIMING_WINDOW_ENABLED=False,
    )
    r = tc.get("/market-data/binance-timing-summary")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert "MARKET_DATA_TIMING_WINDOW_ENABLED" in body.get("notes", "")


# --------------------- Recent-samples endpoint ---------------------


def test_binance_timing_recent_returns_newest_first() -> None:
    tc = _client(BINANCE_WS_ENABLED=True)
    _seed_samples(tc, n=3)
    r = tc.get("/market-data/binance-timing-recent")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["source_type"] == "binance_public_ws"
    assert body["newest_first"] is True
    assert body["limit"] == 200  # default
    assert len(body["samples"]) == 3


def test_binance_timing_recent_honours_limit_query_param() -> None:
    tc = _client(BINANCE_WS_ENABLED=True)
    _seed_samples(tc, n=10)
    r = tc.get("/market-data/binance-timing-recent?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert body["limit"] == 5
    assert len(body["samples"]) == 5


def test_binance_timing_recent_clamps_limit_to_max() -> None:
    """Operator-passed ``limit`` is clamped by
    MARKET_DATA_TIMING_RAW_ENDPOINT_MAX_LIMIT (default 1000). Without
    this guard, a ?limit=10000000 would pull the entire ring buffer
    and bloat the response. Same contract as the GRVT endpoint."""
    tc = _client(BINANCE_WS_ENABLED=True)
    _seed_samples(tc, n=3)
    r = tc.get("/market-data/binance-timing-recent?limit=999999")
    assert r.status_code == 200
    body = r.json()
    # Clamp is in effect — limit returned should be the max, not the
    # requested 999999.
    assert body["limit"] == body["max_limit"]


def test_binance_timing_recent_disabled_when_binance_ws_off() -> None:
    tc = _client(BINANCE_WS_ENABLED=False)
    r = tc.get("/market-data/binance-timing-recent")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is False
    assert body["samples"] == []


# --------------------- stats_snapshot URL list contract -----------


def test_binance_timing_routes_are_in_stats_snapshot_url_list() -> None:
    """``scripts/monitor_get_urls.py::URLS`` is a manual list. If a new
    ``/market-data/*`` route is added to the API but not appended here,
    ``stats_snapshot.py`` won't capture it — and the operator will
    silently lose visibility. Pin the invariant so a future add
    can't forget to register."""
    import sys
    _root = Path(__file__).resolve().parents[1]
    _scripts = _root / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from monitor_get_urls import URLS  # type: ignore
    assert "/market-data/binance-timing-summary" in URLS
    assert "/market-data/binance-timing-recent" in URLS
