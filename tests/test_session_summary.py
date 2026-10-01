"""Pure-function tests for the session rollup."""

from __future__ import annotations

import pytest

from app.session_summary import build_session_summary


def _fill(side: str, notional: float, fee: float, markout_5s: float | None) -> dict:
    return {
        "fill_id": f"f{notional}",
        "side": side,
        "price": notional,
        "size": 1.0,
        "notional": notional,
        "fee": fee,
        "liquidity_flag": "resting",
        "markout_1s_bps": markout_5s,
        "markout_3s_bps": markout_5s,
        "markout_5s_bps": markout_5s,
    }


def test_empty_session_returns_zero_fills_but_valid_structure() -> None:
    out = build_session_summary(
        session_id="sess",
        session_started_at_utc="2026-04-18T10:00:00+00:00",
        now_utc="2026-04-18T10:30:00+00:00",
        fills=[],
        equity_history=[],
        realized_pnl_total_usd=0.0,
    )
    assert out["session_id"] == "sess"
    assert out["duration_seconds"] == pytest.approx(1800.0)
    assert out["throughput"]["fills_count"] == 0
    assert out["attribution"]["fills"]["total"] == 0


def test_fill_rate_per_min() -> None:
    """30 fills over 10 min = 3 fills/min."""
    fills = [_fill("BUY", 100.0, -0.001, -1.0) for _ in range(30)]
    out = build_session_summary(
        session_id="s",
        session_started_at_utc="2026-04-18T10:00:00+00:00",
        now_utc="2026-04-18T10:10:00+00:00",
        fills=fills,
        equity_history=[],
        realized_pnl_total_usd=-0.1,
    )
    assert out["throughput"]["fills_count"] == 30
    assert out["throughput"]["fill_rate_per_min"] == pytest.approx(3.0)


def test_equity_and_drawdown_from_history() -> None:
    eq = [
        {"ts": "2026-04-18T10:00:00+00:00", "equity_usd": 500.0, "drawdown_usd": 0.0, "fees_usd": 0.0},
        {"ts": "2026-04-18T10:05:00+00:00", "equity_usd": 499.8, "drawdown_usd": 0.2, "fees_usd": -0.05},
        {"ts": "2026-04-18T10:10:00+00:00", "equity_usd": 499.5, "drawdown_usd": 0.5, "fees_usd": -0.11},
        {"ts": "2026-04-18T10:15:00+00:00", "equity_usd": 499.9, "drawdown_usd": 0.3, "fees_usd": -0.15},
    ]
    out = build_session_summary(
        session_id="s",
        session_started_at_utc="2026-04-18T10:00:00+00:00",
        now_utc="2026-04-18T10:20:00+00:00",
        fills=[],
        equity_history=eq,
        realized_pnl_total_usd=-0.1,
    )
    assert out["equity"]["start_usd"] == pytest.approx(500.0)
    assert out["equity"]["end_usd"] == pytest.approx(499.9)
    assert out["equity"]["change_usd"] == pytest.approx(-0.1)
    assert out["equity"]["max_drawdown_usd"] == pytest.approx(0.5)
    assert out["equity"]["fees_cumulative_usd"] == pytest.approx(-0.15)


def test_attribution_block_flows_through() -> None:
    """build_session_summary wraps compute_attribution and surfaces its output."""
    fills = [
        _fill("BUY", 100.0, -0.001, -2.0),
        _fill("SELL", 100.0, -0.001, +1.0),
    ]
    out = build_session_summary(
        session_id="s",
        session_started_at_utc="2026-04-18T10:00:00+00:00",
        now_utc="2026-04-18T10:30:00+00:00",
        fills=fills,
        equity_history=[],
        realized_pnl_total_usd=-0.01,
        markout_horizon_s=5,
    )
    attr = out["attribution"]
    assert attr["fills"]["total"] == 2
    assert attr["markout"]["mean_bps"] == pytest.approx(-0.5)
    assert attr["fees"]["rebate_earned_usd"] == pytest.approx(0.002)
    assert attr["pnl_attribution_usd"]["realized_pnl_total"] == pytest.approx(-0.01)


def test_key_counters_passthrough() -> None:
    out = build_session_summary(
        session_id="s",
        session_started_at_utc="2026-04-18T10:00:00+00:00",
        now_utc="2026-04-18T10:30:00+00:00",
        fills=[],
        equity_history=[],
        realized_pnl_total_usd=0.0,
        key_counters={"private_ws_reconnect_count": 2, "custom": "value"},
    )
    assert out["key_counters"]["private_ws_reconnect_count"] == 2
    assert out["key_counters"]["custom"] == "value"


def test_malformed_timestamps_do_not_crash() -> None:
    """Robustness against missing/bad timestamps."""
    out = build_session_summary(
        session_id="s",
        session_started_at_utc="not-a-timestamp",
        now_utc="",
        fills=[],
        equity_history=[],
        realized_pnl_total_usd=0.0,
    )
    assert out["duration_seconds"] is None
    assert out["throughput"]["fill_rate_per_min"] is None
