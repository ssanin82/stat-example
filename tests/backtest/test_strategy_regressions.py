"""Strategy regression tests — Phase 4 (v1.4.235).

These tests replay every available fixture and assert that the
captured baseline still matches. They follow the §4.2 spec from
``backtesting/docs/execution-plan.md``:

* ``test_no_catastrophic_loss`` — replay any fixture; the summary
  must show ``max_drawdown_usd > -100``. Catches accidental
  strategy-breaking changes that would have produced extreme
  drawdown.
* ``test_no_regression_vs_baseline`` — replay against a captured
  baseline (``tests/backtest/baselines/<fixture>.json``). Asserts:
  - ``config_hash`` matches (config drift invalidates the comparison)
  - ``summary.realized_pnl_usd`` within ±$0.50 of baseline
  - ``summary.max_drawdown_usd`` within ±$0.50 of baseline
  - ``summary.fills`` exactly matches (paper-only replay is fully
    deterministic)
  - ``gates_fired`` exactly matches (when populated — empty in
    Phase 4a; populated by Phase 4b real-Bot integration)

Auto-skip when fixtures absent: CI daemon's worktree has no
fixtures (gitignored), so the whole module skips → green. Operator's
laptop has fixtures → exercises the framework.

Baseline generation: ``python scripts/backtest/capture_baseline.py
--fixture <path>``. Operator runs this after intentional strategy
changes; commits the new baseline alongside the code change so the
CI worktree refresh keeps things in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.backtest import ReplayConfig, replay
from tests.backtest.conftest import (
    baseline_path,
    has_baseline,
    list_fixtures,
)


_FIXTURES = list_fixtures()


@pytest.mark.skipif(
    not _FIXTURES,
    reason="no backtest fixtures present (CI worktree expected)",
)
@pytest.mark.parametrize(
    "fixture", _FIXTURES, ids=lambda p: p.name if isinstance(p, Path) else str(p),
)
def test_replay_runs_to_completion(fixture: Path) -> None:
    """Smoke check: every fixture replays without raising."""
    cfg = ReplayConfig(allow_gaps=True)  # tolerate gaps in raw sessions
    report = replay(fixture, config=cfg)
    assert report.total_events > 0
    assert report.first_t_ns > 0
    assert report.last_t_ns >= report.first_t_ns


@pytest.mark.skipif(
    not _FIXTURES,
    reason="no backtest fixtures present (CI worktree expected)",
)
@pytest.mark.parametrize(
    "fixture", _FIXTURES, ids=lambda p: p.name if isinstance(p, Path) else str(p),
)
def test_no_catastrophic_loss(fixture: Path) -> None:
    """The replay must not produce extreme drawdown.

    Paper-only replay without a strategy trades nothing — drawdown
    is always 0. The test still runs to prove the framework works;
    once Phase 4b wires real Bot in, this assertion meaningfully
    catches catastrophic-loss regressions.
    """
    cfg = ReplayConfig(allow_gaps=True)
    report = replay(fixture, config=cfg)
    drawdown = report.summary.get("max_drawdown_usd", 0.0)
    assert drawdown > -100.0, (
        f"catastrophic loss on {fixture.name}: max_drawdown_usd = {drawdown}"
    )


@pytest.mark.skipif(
    not _FIXTURES,
    reason="no backtest fixtures present (CI worktree expected)",
)
@pytest.mark.parametrize(
    "fixture", _FIXTURES, ids=lambda p: p.name if isinstance(p, Path) else str(p),
)
def test_no_regression_vs_baseline(fixture: Path) -> None:
    """Compare current replay output against the captured baseline.

    Skipped per-fixture if the baseline file doesn't exist — run
    ``scripts/backtest/capture_baseline.py --fixture <path>`` to
    create one.
    """
    if not has_baseline(fixture):
        pytest.skip(
            f"no baseline for {fixture.name}; run "
            f"scripts/backtest/capture_baseline.py to create"
        )

    baseline = json.loads(baseline_path(fixture).read_text())
    # Replay with the SAME config as the baseline was captured under
    # — any divergence in config is a different regression class than
    # strategy drift and is asserted on separately via config_hash.
    base_cfg = baseline.get("config", {})
    from app.backtest import PaperExecutorConfig
    base_paper = base_cfg.get("paper_executor", {})
    cfg = ReplayConfig(
        tick_interval_s=base_cfg.get("tick_interval_s", 0.5),
        warmup_seconds=base_cfg.get("warmup_seconds", 0.0),
        okx_symbol=base_cfg.get("okx_symbol", "TON-USDT-SWAP"),
        binance_symbol=base_cfg.get("binance_symbol", "TONUSDT"),
        allow_gaps=base_cfg.get("allow_gaps", False),
        include_per_tick=base_cfg.get("include_per_tick", False),
        paper=PaperExecutorConfig(
            sim_place_latency_s=base_paper.get("sim_place_latency_s", 0.020),
            sim_cancel_latency_s=base_paper.get("sim_cancel_latency_s", 0.015),
            fee_maker_bps=base_paper.get("fee_maker_bps", -0.5),
            fee_taker_bps=base_paper.get("fee_taker_bps", 1.0),
            queue_policy=base_paper.get("queue_policy", "back_of_queue"),
        ),
    )
    report = replay(fixture, config=cfg)

    # Config-drift guard: if the user changed how the config gets
    # built without updating the baseline, the comparison is
    # meaningless — flag it.
    assert report.config_hash == baseline["config_hash"], (
        f"config_hash mismatch for {fixture.name}: "
        f"current={report.config_hash} baseline={baseline['config_hash']}. "
        f"Update the baseline with capture_baseline.py or restore the "
        f"original config."
    )

    cur_summary = report.summary
    base_summary = baseline.get("summary", {})

    # PnL tolerance: ±$0.50 (matches §4.2 spec).
    pnl_diff = abs(
        cur_summary.get("realized_pnl_usd", 0.0)
        - base_summary.get("realized_pnl_usd", 0.0)
    )
    assert pnl_diff < 0.50, (
        f"realized_pnl drift on {fixture.name}: "
        f"current={cur_summary.get('realized_pnl_usd')} "
        f"baseline={base_summary.get('realized_pnl_usd')} "
        f"diff={pnl_diff}"
    )

    # Drawdown tolerance: ±$0.50.
    dd_diff = abs(
        cur_summary.get("max_drawdown_usd", 0.0)
        - base_summary.get("max_drawdown_usd", 0.0)
    )
    assert dd_diff < 0.50, (
        f"drawdown drift on {fixture.name}: "
        f"current={cur_summary.get('max_drawdown_usd')} "
        f"baseline={base_summary.get('max_drawdown_usd')} "
        f"diff={dd_diff}"
    )

    # Fills + volume are exact (paper replay is fully deterministic).
    assert cur_summary.get("fills") == base_summary.get("fills"), (
        f"fill count drift on {fixture.name}: "
        f"current={cur_summary.get('fills')} "
        f"baseline={base_summary.get('fills')}"
    )

    # Gate-firing counts: exact match. Empty in Phase 4a;
    # populated once Phase 4b wires real Bot.
    assert dict(report.gates_fired) == dict(baseline.get("gates_fired", {})), (
        f"gate-firing drift on {fixture.name}: "
        f"current={dict(report.gates_fired)} "
        f"baseline={baseline.get('gates_fired')}"
    )


@pytest.mark.skipif(
    not _FIXTURES,
    reason="no backtest fixtures present (CI worktree expected)",
)
@pytest.mark.parametrize(
    "fixture", _FIXTURES, ids=lambda p: p.name if isinstance(p, Path) else str(p),
)
def test_determinism_byte_identical(fixture: Path) -> None:
    """Two consecutive replays of the same fixture produce byte-
    identical reports (modulo wall_time_s)."""
    cfg = ReplayConfig(allow_gaps=True)
    r1 = replay(fixture, config=cfg)
    r2 = replay(fixture, config=cfg)
    assert r1.to_json(exclude_wall_time=True, exclude_per_tick=False) == \
        r2.to_json(exclude_wall_time=True, exclude_per_tick=False)
