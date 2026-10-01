"""Tests for ``app.backtest.driver`` — Phase 3 (v1.4.234).

Covers:

* Driver runs to completion on a synthetic fixture.
* Same fixture replayed twice produces byte-identical reports.
* ``--override``-like config injection works through ``ReplayConfig``.
* Driver refuses to load a fixture with gaps unless ``allow_gaps=True``.
* Acceptance §3.4: real fixture (laptop-smoke-1) runs to completion;
  determinism check; wall-time under target.
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import pytest

from app.backtest import (
    PaperExecutorConfig,
    ReplayConfig,
    ReplayHarness,
    ReplayReport,
    replay,
    write_report,
)


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _write_jsonl_gz(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")


def _make_bbo_row(*, bid: str, ask: str, bid_sz: str = "100", ask_sz: str = "100") -> dict:
    return {
        "arg": {"channel": "bbo-tbt", "instId": "TON-USDT-SWAP"},
        "data": [{
            "asks": [[ask, ask_sz, "0", "1"]],
            "bids": [[bid, bid_sz, "0", "1"]],
            "ts": "1779353641000",
            "seqId": 1,
        }],
    }


def _build_synthetic_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "synth"
    fixture.mkdir(parents=True, exist_ok=True)
    # 1 second of data with 4 bbo updates @ 0.25s spacing.
    base_t = 1_700_000_000_000_000_000
    okx_events = [
        {"t_recv_ns": base_t + i * 250_000_000,
         "source": "okx_public",
         "msg": _make_bbo_row(bid=f"2.04{i}", ask=f"2.05{i}")}
        for i in range(4)
    ]
    _write_jsonl_gz(fixture / "okx_public.jsonl.gz", okx_events)
    manifest = {
        "schema_version": 1,
        "session_id": "synth-1",
        "recorder_version": "0.1.0",
        "is_finalized": True,
        "files": [{
            "path": "okx_public.jsonl.gz",
            "source": "okx_public",
            "lines": 4,
            "first_t_recv_ns": okx_events[0]["t_recv_ns"],
            "last_t_recv_ns": okx_events[-1]["t_recv_ns"],
            "gaps": 0,
        }],
    }
    (fixture / "manifest.json").write_text(json.dumps(manifest))
    return fixture


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_non_positive_tick_interval() -> None:
    with pytest.raises(ValueError):
        ReplayConfig(tick_interval_s=0.0)
    with pytest.raises(ValueError):
        ReplayConfig(tick_interval_s=-1.0)


def test_config_rejects_negative_warmup() -> None:
    with pytest.raises(ValueError):
        ReplayConfig(warmup_seconds=-1.0)


# ---------------------------------------------------------------------------
# Basic synthetic replay
# ---------------------------------------------------------------------------


def test_synthetic_replay_runs_to_completion(tmp_path: Path) -> None:
    fixture = _build_synthetic_fixture(tmp_path)
    report = replay(fixture, bot_version="test")
    assert isinstance(report, ReplayReport)
    assert report.total_events == 4
    assert report.events_by_source["okx_public"] == 4
    assert report.public_stream["bbo_dispatched"] == 4
    assert report.first_t_ns == 1_700_000_000_000_000_000


def test_synthetic_replay_advances_paper_book(tmp_path: Path) -> None:
    """The paper executor's last-seen book should match the last
    BBO from the replay."""
    fixture = _build_synthetic_fixture(tmp_path)
    # Sneak a peek by capturing the harness through on_tick.
    captured: list[ReplayHarness] = []

    def on_tick(_t_ns: int, h: ReplayHarness) -> None:
        captured.append(h)

    replay(fixture, on_tick=on_tick)
    # At least one tick was scheduled (1 second of data at 0.5s
    # interval → 2 ticks at minimum).
    assert len(captured) >= 1
    paper = captured[-1].paper
    assert paper._best_bid == pytest.approx(2.043)
    assert paper._best_ask == pytest.approx(2.053)


def test_tick_callbacks_fire_at_interval(tmp_path: Path) -> None:
    fixture = _build_synthetic_fixture(tmp_path)
    cfg = ReplayConfig(tick_interval_s=0.25)  # tick every BBO
    tick_times: list[int] = []
    replay(fixture, config=cfg, on_tick=lambda t_ns, _: tick_times.append(t_ns))
    # 1 second of events, tick every 250ms → at least 4 scheduled ticks.
    assert len(tick_times) >= 4
    # Monotonically increasing.
    assert tick_times == sorted(tick_times)


def test_warmup_skips_ticks(tmp_path: Path) -> None:
    """During the warmup window, tick callbacks shouldn't fire."""
    fixture = _build_synthetic_fixture(tmp_path)
    cfg = ReplayConfig(tick_interval_s=0.1, warmup_seconds=0.5)
    callbacks: list[int] = []
    report = replay(fixture, config=cfg, on_tick=lambda t, _: callbacks.append(t))
    # Ticks scheduled span the full 1 second; half should be in warmup.
    assert report.ticks_skipped_warmup > 0
    assert report.ticks_executed == len(callbacks)


# ---------------------------------------------------------------------------
# Determinism — byte-identical reports across reruns
# ---------------------------------------------------------------------------


def test_two_runs_produce_byte_identical_report(tmp_path: Path) -> None:
    fixture = _build_synthetic_fixture(tmp_path)
    r1 = replay(fixture)
    r2 = replay(fixture)
    # Strip wall_time_s (intentionally non-deterministic).
    j1 = r1.to_json(exclude_wall_time=True)
    j2 = r2.to_json(exclude_wall_time=True)
    assert j1 == j2


def test_determinism_holds_with_two_sources(tmp_path: Path) -> None:
    """Add a Binance file and verify the merge order is stable."""
    fixture = _build_synthetic_fixture(tmp_path)
    base_t = 1_700_000_000_000_000_000
    binance_events = [
        {"t_recv_ns": base_t + i * 200_000_000,
         "source": "binance_public",
         "msg": {"e": "bookTicker", "s": "TONUSDT",
                 "b": f"2.04{i}", "B": "100", "a": f"2.05{i}", "A": "100",
                 "T": 1779353641000 + i * 200}}
        for i in range(5)
    ]
    _write_jsonl_gz(fixture / "binance_public.jsonl.gz", binance_events)
    # Update manifest.
    manifest = json.loads((fixture / "manifest.json").read_text())
    manifest["files"].append({
        "path": "binance_public.jsonl.gz",
        "source": "binance_public",
        "lines": 5,
        "first_t_recv_ns": binance_events[0]["t_recv_ns"],
        "last_t_recv_ns": binance_events[-1]["t_recv_ns"],
        "gaps": 0,
    })
    (fixture / "manifest.json").write_text(json.dumps(manifest))

    r1 = replay(fixture)
    r2 = replay(fixture)
    assert r1.to_json(exclude_wall_time=True) == r2.to_json(exclude_wall_time=True)


# ---------------------------------------------------------------------------
# Config overrides thread through
# ---------------------------------------------------------------------------


def test_paper_executor_config_overrides_apply(tmp_path: Path) -> None:
    """Custom PaperExecutorConfig propagates into the harness."""
    fixture = _build_synthetic_fixture(tmp_path)
    cfg = ReplayConfig(
        paper=PaperExecutorConfig(
            sim_place_latency_s=0.050,
            sim_cancel_latency_s=0.030,
            fee_maker_bps=-1.0,
            fee_taker_bps=2.0,
        ),
    )
    report = replay(fixture, config=cfg)
    assert report.config["paper_executor"]["sim_place_latency_s"] == 0.050
    assert report.config["paper_executor"]["fee_maker_bps"] == -1.0


# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------


def test_replay_refuses_gaps_without_opt_in(tmp_path: Path) -> None:
    fixture = _build_synthetic_fixture(tmp_path)
    manifest = json.loads((fixture / "manifest.json").read_text())
    manifest["files"][0]["gaps"] = 3
    (fixture / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="gap"):
        replay(fixture)


def test_replay_proceeds_with_allow_gaps(tmp_path: Path) -> None:
    fixture = _build_synthetic_fixture(tmp_path)
    manifest = json.loads((fixture / "manifest.json").read_text())
    manifest["files"][0]["gaps"] = 3
    (fixture / "manifest.json").write_text(json.dumps(manifest))
    cfg = ReplayConfig(allow_gaps=True)
    report = replay(fixture, config=cfg)
    assert report.gaps_reported == 3


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------


def test_report_writes_to_disk(tmp_path: Path) -> None:
    fixture = _build_synthetic_fixture(tmp_path)
    report = replay(fixture)
    out_path = tmp_path / "report.json"
    write_report(report, out_path)
    assert out_path.exists()
    parsed = json.loads(out_path.read_text())
    assert parsed["schema_version"] == 2  # bumped in Phase 4 (v1.4.235)
    assert parsed["fixture"]["session_id"] == "synth-1"
    assert parsed["events"]["total"] == 4


def test_report_excludes_wall_time_when_requested(tmp_path: Path) -> None:
    fixture = _build_synthetic_fixture(tmp_path)
    report = replay(fixture)
    raw = json.loads(report.to_json(exclude_wall_time=False))
    det = json.loads(report.to_json(exclude_wall_time=True))
    assert "wall_time_s" in raw
    assert "wall_time_s" not in det


# ---------------------------------------------------------------------------
# BUG-B + BUG-H (v1.5.303) — honesty caveats; BUG-G fill_attribution and
# BUG-E skip_reasons surfaced in the report.
# ---------------------------------------------------------------------------


def test_report_caveats_present_paper_only(tmp_path: Path) -> None:
    """BUG-B + BUG-H: every report carries an honesty caveats block so
    downstream readers don't mistake paper-sim figures for venue-exact
    truth."""
    fixture = _build_synthetic_fixture(tmp_path)
    report = replay(fixture)
    cav = report.caveats
    # BUG-B: paper executor is authoritative; recorded private fills are
    # NOT cross-checked against simulated fills.
    assert cav["authoritative_fill_source"] == "paper_executor"
    assert cav["recorded_private_fills_cross_checked"] is False
    # Paper-only mode → no bot, nothing suppressed.
    assert cav["with_bot_mode"] is False
    assert cav["recorded_private_fills_suppressed"] == 0
    assert cav["recorded_private_order_updates_suppressed"] == 0
    # BUG-H: mark price is a mid proxy → drawdown is not venue-exact.
    assert cav["mark_price_source"] == "mid_proxy"
    assert cav["drawdown_is_mid_proxy"] is True


def test_report_caveats_serialize_in_to_dict(tmp_path: Path) -> None:
    """Caveats survive the JSON round-trip under a stable top-level key."""
    fixture = _build_synthetic_fixture(tmp_path)
    report = replay(fixture)
    parsed = json.loads(report.to_json())
    assert parsed["caveats"]["mark_price_source"] == "mid_proxy"
    assert parsed["caveats"]["authoritative_fill_source"] == "paper_executor"


def test_report_fill_attribution_present(tmp_path: Path) -> None:
    """BUG-G: the report carries the fill-attribution diagnostic so a
    low fill count is explainable. The synthetic fixture has BBO only
    (no trades) → nothing seen / addressable, ratios safely 0."""
    fixture = _build_synthetic_fixture(tmp_path)
    report = replay(fixture)
    fa = report.fill_attribution
    assert fa["trade_prints_seen"] == 0
    assert fa["trade_base_size_addressable"] == 0.0
    assert fa["queue_block_ratio"] == 0.0
    assert fa["fill_ratio"] == 0.0
    # Stable key set downstream (dashboard / acceptance) can rely on.
    assert "filled_base_from_trades" in fa
    assert "fills_blocked_by_queue_only" in fa
    parsed = json.loads(report.to_json())
    assert "fill_attribution" in parsed


def test_report_skip_reasons_present(tmp_path: Path) -> None:
    """BUG-E: the report's stream blocks carry the per-reason skip
    breakdown (empty here — the synthetic fixture dispatches cleanly)."""
    fixture = _build_synthetic_fixture(tmp_path)
    report = replay(fixture)
    assert report.public_stream["skip_reasons"] == {}
    assert report.private_stream["skip_reasons"] == {}
    parsed = json.loads(report.to_json())
    assert "skip_reasons" in parsed["streams"]["public"]
    assert "skip_reasons" in parsed["streams"]["private"]


def test_report_per_regime_and_net_edge_present_paper_only(tmp_path: Path) -> None:
    """Audit §4.4 (P0 #4+#5): every report carries per_regime +
    net_edge_per_min_series blocks. Paper-only replay has no bot, so the
    regime FSM / AQC controller never stamp a tick → both stay empty,
    but the keys exist and serialize for the dashboard to rely on."""
    fixture = _build_synthetic_fixture(tmp_path)
    report = replay(fixture)
    assert report.per_regime == {}
    assert report.net_edge_per_min_series == []
    parsed = json.loads(report.to_json())
    assert parsed["per_regime"] == {}
    assert parsed["net_edge_per_min_series"] == []


# ---------------------------------------------------------------------------
# Real fixture acceptance
# ---------------------------------------------------------------------------


_REAL_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "backtesting/data/sessions/laptop-smoke-1"
)


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="laptop-smoke-1 fixture not present",
)
def test_real_fixture_replay_runs_to_completion() -> None:
    report = replay(_REAL_FIXTURE, bot_version="test")
    assert report.total_events > 0
    # Acceptance §3.4: a 4-min fixture should complete in well
    # under 30 seconds wall time on operator's laptop.
    assert report.wall_time_s < 30.0, f"too slow: {report.wall_time_s}s"


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="laptop-smoke-1 fixture not present",
)
def test_real_fixture_two_runs_byte_identical() -> None:
    r1 = replay(_REAL_FIXTURE)
    r2 = replay(_REAL_FIXTURE)
    assert r1.to_json(exclude_wall_time=True) == r2.to_json(exclude_wall_time=True)


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="laptop-smoke-1 fixture not present",
)
def test_real_fixture_dispatches_both_venues() -> None:
    report = replay(_REAL_FIXTURE)
    # laptop-smoke-1 captures OKX + Binance public streams.
    assert "okx_public" in report.events_by_source
    assert "binance_public" in report.events_by_source
    assert report.public_stream["bbo_dispatched"] > 0


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


def test_cli_runs_against_synthetic_fixture(tmp_path: Path) -> None:
    """End-to-end CLI invocation: load fixture, write report."""
    import subprocess
    import sys

    fixture = _build_synthetic_fixture(tmp_path)
    out_path = tmp_path / "cli-report.json"
    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "backtest" / "replay.py"),
            "--fixture", str(fixture),
            "--report-out", str(out_path),
            "--deterministic",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()
    parsed = json.loads(out_path.read_text())
    assert "wall_time_s" not in parsed  # --deterministic stripped it
    assert parsed["events"]["total"] == 4
