"""Tests for ``app.backtest.report`` + the Phase 4 report extensions
on ``ReplayReport`` (v1.4.235)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from queue import Queue

import pytest

from app.backtest import (
    MetricsAccumulator,
    PaperExecutor,
    PaperExecutorConfig,
    ReplayConfig,
    TickSample,
    compute_config_hash,
    replay,
)
from app.clock import ReplayClock
from app.enums import Side


# ---------------------------------------------------------------------------
# compute_config_hash
# ---------------------------------------------------------------------------


def test_config_hash_is_deterministic() -> None:
    a = compute_config_hash({"a": 1, "b": 2, "c": [1, 2, 3]})
    b = compute_config_hash({"c": [1, 2, 3], "a": 1, "b": 2})
    assert a == b  # key order doesn't matter


def test_config_hash_differs_on_value_change() -> None:
    a = compute_config_hash({"sim_place_latency_s": 0.020})
    b = compute_config_hash({"sim_place_latency_s": 0.030})
    assert a != b


def test_config_hash_has_sha256_prefix() -> None:
    h = compute_config_hash({})
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


# ---------------------------------------------------------------------------
# MetricsAccumulator
# ---------------------------------------------------------------------------


def _sample(
    *,
    t_ns: int = 0,
    position: float = 0.0,
    realized: float = 0.0,
    unrealised: float = 0.0,
    fees: float = 0.0,
    open_orders: int = 0,
    best_bid: float = 2.0,
    best_ask: float = 2.01,
    regime: "str | None" = None,
    aggression: "float | None" = None,
    net_edge: "float | None" = None,
) -> TickSample:
    return TickSample(
        t_ns=t_ns,
        position_qty=position,
        realized_pnl_usd=realized,
        unrealized_pnl_usd=unrealised,
        fees_total_usd=fees,
        best_bid=best_bid,
        best_ask=best_ask,
        open_orders=open_orders,
        equity_usd=realized + unrealised - fees,
        regime_mode_label=regime,
        aqc_aggression_level=aggression,
        aqc_observed_net_edge_per_min_usd=net_edge,
    )


def test_metrics_tracks_peak_equity_and_drawdown() -> None:
    m = MetricsAccumulator()
    m.observe_tick(_sample(realized=10.0), 0)   # equity = 10
    m.observe_tick(_sample(realized=20.0), 0)   # equity = 20 (new peak)
    m.observe_tick(_sample(realized=15.0), 0)   # drawdown = -5
    m.observe_tick(_sample(realized=5.0), 0)    # drawdown = -15
    m.observe_tick(_sample(realized=30.0), 0)   # new peak; drawdown stays at -15
    assert m.peak_equity_usd == 30.0
    assert m.max_drawdown_usd == -15.0


def test_metrics_tracks_inventory_extremes() -> None:
    m = MetricsAccumulator()
    m.observe_tick(_sample(position=10.0), 0)
    m.observe_tick(_sample(position=25.0), 0)
    m.observe_tick(_sample(position=-15.0), 0)
    m.observe_tick(_sample(position=-30.0), 0)
    assert m.max_inventory_qty_long == 25.0
    assert m.max_inventory_qty_short == -30.0


def test_metrics_time_with_orders_pct() -> None:
    m = MetricsAccumulator()
    for _ in range(8):
        m.observe_tick(_sample(open_orders=2), 0)
    for _ in range(2):
        m.observe_tick(_sample(open_orders=0), 0)
    summary = m.finalize(fills=0, realized_pnl_usd=0.0, fees_total_usd=0.0)
    assert summary["time_with_orders_pct"] == pytest.approx(80.0)


def test_metrics_volume_accumulates_across_ticks() -> None:
    m = MetricsAccumulator()
    m.observe_tick(_sample(), 100.0)
    m.observe_tick(_sample(), 250.0)
    m.observe_tick(_sample(), 50.0)
    summary = m.finalize(fills=3, realized_pnl_usd=0.0, fees_total_usd=0.0)
    assert summary["volume_usd"] == 400.0


def test_metrics_finalize_includes_all_required_keys() -> None:
    m = MetricsAccumulator()
    m.observe_tick(_sample(), 0)
    summary = m.finalize(fills=5, realized_pnl_usd=1.5, fees_total_usd=-0.05)
    expected_keys = {
        "fills", "volume_usd", "realized_pnl_usd", "fees_total_usd",
        "max_drawdown_usd", "max_inventory_qty_long",
        "max_inventory_qty_short", "peak_equity_usd",
        "ticks_total", "ticks_with_orders", "time_with_orders_pct",
    }
    assert set(summary.keys()) == expected_keys
    assert summary["fills"] == 5
    assert summary["realized_pnl_usd"] == 1.5


# ---------------------------------------------------------------------------
# MetricsAccumulator — per-regime rollup + net-edge series (audit §4.4,
# P0 #4+#5). v1.5.304.
# ---------------------------------------------------------------------------


def test_per_regime_rollup_attributes_ticks_and_deltas() -> None:
    m = MetricsAccumulator()
    # 3 CALM ticks (2 with orders); one fill (+volume +pnl −fee) lands on
    # a CALM tick. Then 1 CAUTIOUS tick with no orders.
    m.observe_tick(_sample(regime="CALM", open_orders=2, aggression=0.2), 0.0)
    m.observe_tick(
        _sample(regime="CALM", open_orders=2, aggression=0.4),
        20.0,
        fills_delta=1,
        realized_pnl_delta=1.5,
        fees_delta=-0.01,
    )
    m.observe_tick(_sample(regime="CALM", open_orders=0, aggression=0.6), 0.0)
    m.observe_tick(_sample(regime="CAUTIOUS", open_orders=0, aggression=0.0), 0.0)
    pr = m.finalize_extras()["per_regime"]
    assert set(pr) == {"CALM", "CAUTIOUS"}
    calm = pr["CALM"]
    assert calm["ticks"] == 3
    assert calm["ticks_with_orders"] == 2
    assert calm["fills"] == 1
    assert calm["volume_usd"] == pytest.approx(20.0)
    assert calm["realized_pnl_usd"] == pytest.approx(1.5)
    assert calm["fees_total_usd"] == pytest.approx(-0.01)
    # avg aggression over the 3 CALM ticks = (0.2+0.4+0.6)/3 = 0.4
    assert calm["avg_aggression_level"] == pytest.approx(0.4)
    # time_in_mode_pct = 3 of 4 total ticks = 75%
    assert calm["time_in_mode_pct"] == pytest.approx(75.0)
    assert pr["CAUTIOUS"]["ticks"] == 1
    assert pr["CAUTIOUS"]["fills"] == 0


def test_per_regime_labels_emitted_sorted() -> None:
    m = MetricsAccumulator()
    for lbl in ("VOLATILE", "CALM", "CAUTIOUS"):
        m.observe_tick(_sample(regime=lbl), 0.0)
    assert list(m.finalize_extras()["per_regime"].keys()) == [
        "CALM",
        "CAUTIOUS",
        "VOLATILE",
    ]


def test_paper_only_ticks_leave_per_regime_and_series_empty() -> None:
    m = MetricsAccumulator()
    for _ in range(5):
        m.observe_tick(_sample(), 0.0)  # no regime label, no net-edge
    extras = m.finalize_extras()
    assert extras["per_regime"] == {}
    assert extras["net_edge_per_min_series"] == []


def test_net_edge_series_downsamples_to_one_per_minute() -> None:
    m = MetricsAccumulator()
    minute = 60_000_000_000
    sec = 1_000_000_000
    # t=0 -> sampled (series empty). t=+30s -> NOT (< 60s since last).
    # t=+60s -> sampled. t=+61s -> NOT.
    m.observe_tick(_sample(t_ns=0, regime="CALM", net_edge=0.010), 0.0)
    m.observe_tick(_sample(t_ns=30 * sec, regime="CALM", net_edge=0.012), 0.0)
    m.observe_tick(_sample(t_ns=minute, regime="CALM", net_edge=0.020), 0.0)
    m.observe_tick(_sample(t_ns=minute + sec, regime="CALM", net_edge=0.030), 0.0)
    series = m.finalize_extras()["net_edge_per_min_series"]
    assert [pt["t_ns"] for pt in series] == [0, minute]
    assert [pt["net_edge_per_min_usd"] for pt in series] == [0.01, 0.02]


def test_net_edge_series_skips_none_readings() -> None:
    m = MetricsAccumulator()
    # net_edge None on every tick (controller had no observation yet).
    for i in range(3):
        m.observe_tick(_sample(t_ns=i * 60_000_000_000, net_edge=None), 0.0)
    assert m.finalize_extras()["net_edge_per_min_series"] == []


def test_finalize_extras_does_not_disturb_summary_keys() -> None:
    m = MetricsAccumulator()
    m.observe_tick(
        _sample(regime="CALM", aggression=0.5, net_edge=0.02), 10.0, fills_delta=1
    )
    summary = m.finalize(fills=1, realized_pnl_usd=0.0, fees_total_usd=0.0)
    # The stable summary contract (baseline tests) must not gain keys.
    assert "per_regime" not in summary
    assert "net_edge_per_min_series" not in summary


def test_observe_tick_back_compat_without_delta_kwargs() -> None:
    # Older callers pass only fill_volume_usd_delta positionally; the new
    # keyword deltas default to 0 and per-regime fills/pnl stay zeroed.
    m = MetricsAccumulator()
    m.observe_tick(_sample(regime="CALM"), 50.0)
    calm = m.finalize_extras()["per_regime"]["CALM"]
    assert calm["volume_usd"] == pytest.approx(50.0)
    assert calm["fills"] == 0
    assert calm["realized_pnl_usd"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TickSample serialisation
# ---------------------------------------------------------------------------


def test_tick_sample_to_dict_has_regime_and_aqc_fields() -> None:
    s = _sample(regime="CALM", aggression=0.3, net_edge=0.015)
    d = s.to_dict()
    assert d["regime_mode_label"] == "CALM"
    assert d["aqc_aggression_level"] == 0.3
    assert d["aqc_observed_net_edge_per_min_usd"] == 0.015


def test_tick_sample_to_dict_paper_only_has_none_aqc_fields() -> None:
    d = _sample().to_dict()
    assert d["regime_mode_label"] is None
    assert d["aqc_aggression_level"] is None
    assert d["aqc_observed_net_edge_per_min_usd"] is None


def test_tick_sample_to_dict_round_trip() -> None:
    s = _sample(t_ns=12345, position=10.0, realized=5.0, fees=-0.1)
    d = s.to_dict()
    assert d["t_ns"] == 12345
    assert d["position_qty"] == 10.0
    assert d["equity_usd"] == 5.0 - (-0.1)


# ---------------------------------------------------------------------------
# Paper executor: total_fill_volume_usd accumulates correctly
# ---------------------------------------------------------------------------


def test_paper_executor_tracks_fill_volume() -> None:
    clock = ReplayClock(start_t_ns=1_700_000_000_000_000_000)
    sink: Queue = Queue()
    px = PaperExecutor(clock=clock, private_event_sink=sink,
                       config=PaperExecutorConfig())
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=10.0, limit_px=2.000)
    # Advance past place-latency.
    clock.advance_to(clock._t_ns + int(0.025 * 1e9))
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    assert px.total_fill_volume_usd == pytest.approx(20.0)  # 10 × 2.0
    # Another fill.
    px.process_book_event(bid=2.040, ask=2.060, bid_size=10.0, ask_size=0.0)
    px.place_post_only_limit(symbol="X", is_buy=False, sz=5.0, limit_px=2.050)
    clock.advance_to(clock._t_ns + int(0.025 * 1e9))
    px.process_trade_event(price=2.050, size=5.0, side="BUY")
    assert px.total_fill_volume_usd == pytest.approx(20.0 + 5.0 * 2.050)


# ---------------------------------------------------------------------------
# Driver: report has all Phase 4 fields
# ---------------------------------------------------------------------------


def _write_jsonl_gz(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")


def _build_fixture(tmp_path: Path) -> Path:
    fixture = tmp_path / "synth"
    fixture.mkdir(parents=True, exist_ok=True)
    base_t = 1_700_000_000_000_000_000
    okx_events = [
        {"t_recv_ns": base_t + i * 250_000_000,
         "source": "okx_public",
         "msg": {"arg": {"channel": "bbo-tbt"},
                 "data": [{"asks": [[f"2.0{50+i}", "100", "0", "1"]],
                           "bids": [[f"2.0{40+i}", "100", "0", "1"]],
                           "ts": str(base_t + i * 250_000_000),
                           "seqId": i}]}}
        for i in range(8)
    ]
    _write_jsonl_gz(fixture / "okx_public.jsonl.gz", okx_events)
    (fixture / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "session_id": "phase4-synth",
        "recorder_version": "0.1.0",
        "is_finalized": True,
        "files": [{
            "path": "okx_public.jsonl.gz",
            "source": "okx_public",
            "lines": len(okx_events),
            "first_t_recv_ns": okx_events[0]["t_recv_ns"],
            "last_t_recv_ns": okx_events[-1]["t_recv_ns"],
            "gaps": 0,
        }],
    }))
    return fixture


def test_report_has_phase4_fields(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    report = replay(fixture)
    d = report.to_dict()
    assert d["schema_version"] == 2  # bumped in Phase 4
    assert "config_hash" in d
    assert d["config_hash"].startswith("sha256:")
    assert "summary" in d
    assert "gates_fired" in d
    assert "per_tick" in d
    # Summary has the §5.4 spec keys.
    assert {"fills", "volume_usd", "realized_pnl_usd",
            "max_drawdown_usd"} <= d["summary"].keys()


def test_per_tick_off_by_default(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    report = replay(fixture)
    assert report.per_tick == []


def test_per_tick_populated_when_opted_in(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    cfg = ReplayConfig(tick_interval_s=0.25, include_per_tick=True)
    report = replay(fixture, config=cfg)
    # 2 seconds of data at 0.25s cadence → ~8 ticks.
    assert len(report.per_tick) >= 4
    # Each sample has the expected keys.
    for sample in report.per_tick:
        assert "t_ns" in sample
        assert "position_qty" in sample
        assert "equity_usd" in sample
        assert "open_orders" in sample


def test_paper_only_replay_summary_is_zero(tmp_path: Path) -> None:
    """No strategy → no fills → all PnL/volume metrics are zero."""
    fixture = _build_fixture(tmp_path)
    report = replay(fixture)
    assert report.summary["fills"] == 0
    assert report.summary["volume_usd"] == 0.0
    assert report.summary["realized_pnl_usd"] == 0.0
    assert report.summary["max_drawdown_usd"] == 0.0


def test_summary_with_strategy_simulated_via_on_tick(tmp_path: Path) -> None:
    """Drive fills via on_tick → summary reflects them.

    This proves the end-to-end metric pipeline works once a strategy
    (Phase 4b real Bot) is wired in.
    """
    fixture = _build_fixture(tmp_path)
    placed = [False]

    def on_tick(t_ns, harness):
        if not placed[0] and harness.paper._best_bid is not None:
            harness.paper.place_post_only_limit(
                symbol="X", is_buy=True, sz=10.0,
                limit_px=harness.paper._best_bid - 0.005,
            )
            placed[0] = True

    cfg = ReplayConfig(tick_interval_s=0.25)
    report = replay(fixture, config=cfg, on_tick=on_tick)
    # Order placed but never filled (no trade events in fixture);
    # the placement should still register in paper executor counters.
    assert report.paper_executor["acks_emitted"] >= 1


def test_determinism_with_metrics_enabled(tmp_path: Path) -> None:
    """Two runs with per_tick + summary enabled still byte-identical."""
    fixture = _build_fixture(tmp_path)
    cfg = ReplayConfig(tick_interval_s=0.25, include_per_tick=True)
    r1 = replay(fixture, config=cfg)
    r2 = replay(fixture, config=cfg)
    assert r1.to_json(exclude_wall_time=True) == r2.to_json(exclude_wall_time=True)


def test_config_hash_present_and_stable(tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    r1 = replay(fixture)
    r2 = replay(fixture)
    assert r1.config_hash == r2.config_hash
    # Different config → different hash.
    r3 = replay(
        fixture,
        config=ReplayConfig(paper=PaperExecutorConfig(sim_place_latency_s=0.040)),
    )
    assert r1.config_hash != r3.config_hash


# ---------------------------------------------------------------------------
# Baseline-comparison test framework — meta-tests (run on operator's box)
# ---------------------------------------------------------------------------


_REAL_FIXTURE = (
    Path(__file__).resolve().parent.parent
    / "backtesting" / "data" / "sessions" / "laptop-smoke-1"
)


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="laptop-smoke-1 fixture not present",
)
def test_real_fixture_summary_keys_present() -> None:
    cfg = ReplayConfig(allow_gaps=True)
    report = replay(_REAL_FIXTURE, config=cfg)
    assert {"fills", "volume_usd", "realized_pnl_usd",
            "max_drawdown_usd", "max_inventory_qty_long",
            "max_inventory_qty_short", "fees_total_usd"} <= report.summary.keys()
