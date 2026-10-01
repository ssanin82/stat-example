import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router
from app.market_data_timing import RECEIVE_TO_APPLY_SOURCE_MONOTONIC
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _client() -> TestClient:
    path = Path(tempfile.gettempdir()) / f"mm_api_test_{os.getpid()}_{uuid.uuid4().hex}.db"
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


def test_status_includes_control_and_safety_flags() -> None:
    c = _client()
    r = c.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["control_endpoints_enabled"] is False
    assert "order_desync" in body
    assert body["desync_quarantine_remaining"] == 0
    assert "flatten_incomplete" in body
    assert body["trades_last_minute"] == 0
    assert "latency_fill_to_process_ms" in body


def test_health_ok() -> None:
    c = _client()
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["trading_enabled"] is False
    assert body["market_data_available"] is False
    assert body["account_data_available"] is False
    assert body["order_desync"] is False
    assert body["desync_phase"] == "OK"
    assert body["desync_quarantine_remaining"] == 0
    assert body["trades_last_minute"] == 0
    assert body["exchange_ts_to_decision_ms"] is None
    assert body["latency_tick_preamble_ms"] is None


def test_state_position_thread_safe_shape() -> None:
    c = _client()
    r = c.get("/state/position")
    assert r.status_code == 200
    assert "position_qty" in r.json()


def test_recent_limits_clamped() -> None:
    c = _client()
    r = c.get("/orders/recent?limit=99999")
    assert r.status_code == 200
    assert r.json() == []


def test_kill_state_shape() -> None:
    c = _client()
    r = c.get("/kill-state")
    assert r.status_code == 200
    body = r.json()
    assert body["killed"] is False
    assert body["kill_reason"] is None
    assert body["kill_timestamp"] is None
    assert "flatten_on_kill" in body
    assert "toxicity_cooldown_remaining_seconds" not in body
    assert body["order_desync"] is False
    assert body["desync_phase"] == "OK"
    assert body["desync_quarantine_remaining"] == 0
    assert body["flatten_incomplete"] is False
    assert body["flatten_residual_abs_qty"] == 0.0


def test_state_current_includes_quote_quality() -> None:
    c = _client()
    r = c.get("/state/current")
    assert r.status_code == 200
    qq = r.json().get("quote_quality")
    assert isinstance(qq, dict)
    assert "avg_quoted_spread_bps" in qq
    assert "delayed_markout_summary" in qq
    assert "realized_pnl_usd" in qq


def test_state_current_includes_flow_score() -> None:
    """Priority #3 v1 — the flow-direction toxicity score is surfaced
    in /state/current. v1 is observability-only (threshold unreachable)
    so presence of the telemetry is the test; score values default to
    zero until trades arrive."""
    c = _client()
    r = c.get("/state/current")
    assert r.status_code == 200
    fs = r.json().get("flow_score")
    assert isinstance(fs, dict)
    # Cold-start values: no trades yet.
    assert fs["buy_toxic_score"] == 0.0
    assert fs["sell_toxic_score"] == 0.0
    assert fs["tfi_signed_normalised"] == 0.0
    assert fs["tfi_window_trade_count"] == 0
    assert fs["streak_buy_count"] == 0
    assert fs["streak_sell_count"] == 0
    assert fs["last_trade_ts_ms"] is None
    assert fs["trade_history_count"] == 0
    # Config values flow through as diagnostics.
    assert fs["tfi_window_seconds"] > 0
    assert fs["streak_window_prints"] >= 2


def test_state_current_includes_basis_regime_snapshot() -> None:
    """Priority #2 v2 — the basis regime classifier's current state is
    surfaced in /state/current so the operator can verify the detector
    is producing sensible IC / regime-sign calls before re-enabling the
    basis-deviation alpha term."""
    c = _client()
    r = c.get("/state/current")
    assert r.status_code == 200
    br = r.json().get("basis_regime")
    assert isinstance(br, dict)
    # Cold-start values: no pairs yet, no IC, undecided sign.
    assert br["last_regime_sign"] == 0.0
    assert br["pair_count"] == 0
    assert br["last_ic"] is None
    # Configuration bounds are reflected (confirms wiring, not just a
    # stale / dummy stub in the response).
    assert br["horizon_seconds"] > 0
    assert br["window_samples"] >= 10
    assert 0.0 <= br["ic_threshold"] < 1.0
    assert br["min_pair_samples"] >= 2


def test_telemetry_quote_quality_endpoint() -> None:
    c = _client()
    r = c.get("/telemetry/quote-quality")
    assert r.status_code == 200
    qq = r.json()
    assert "post_only_cross_rejection_count_session" in qq
    assert "estimated_gross_spread_capture_usd_session" in qq


def test_market_data_timing_endpoints_shape() -> None:
    c = _client()
    r = c.get("/market-data/timing-summary")
    assert r.status_code == 200
    body = r.json()
    assert "symbol" in body
    assert "enabled" in body

    r2 = c.get("/market-data/timing-recent?limit=5")
    assert r2.status_code == 200
    b2 = r2.json()
    assert "enabled" in b2
    assert "samples" in b2


def test_market_data_timing_endpoints_values_and_nulls() -> None:
    c = _client()
    st = c.app.state.bot_state
    tr = st.public_ws_timing
    assert tr is not None
    tr.note_stream_reset("seed")
    tr.ingest(
        local_receive_wall_ms=None,
        local_receive_mono_ns=None,
        exchange_ts_ms=1000,
        local_apply_wall_ms=2000,
        local_apply_mono_ns=2_000_000,
        seq=1,
    )
    tr.ingest(
        local_receive_wall_ms=2100,
        local_receive_mono_ns=10_000,
        exchange_ts_ms=2000,
        local_apply_wall_ms=2105,
        local_apply_mono_ns=15_000,
        seq=2,
    )
    summ = c.get("/market-data/timing-summary").json()
    assert summ["total_samples"] >= 2
    assert summ["samples_with_local_receive_wall"] >= 1
    assert summ["missing_local_receive_wall_count"] >= 1
    assert "boundary_event_count" in summ
    assert "boundary_count_semantics" in summ
    assert summ["last_boundary_reason"] == "seed"

    recent = c.get("/market-data/timing-recent?limit=2").json()["samples"]
    # newest-first: sample with receive wall present should compute one-way delay.
    assert recent[0]["exchange_to_local_receive_ms"] == 100.0
    assert recent[0]["receive_to_apply_source"] == RECEIVE_TO_APPLY_SOURCE_MONOTONIC
    # older sample missing receive should keep nulls (no synthetic backfill).
    assert recent[1]["local_receive_wall_time_ms"] is None
    assert recent[1]["exchange_to_local_receive_ms"] is None
    assert recent[1]["receive_to_apply_source"] is None
