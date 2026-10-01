"""Tests for ``tools.postmortem.sections.counterfactual``.

The framework is descriptive — single-gate, snapshot-bound. These
tests assert the mechanics: windows align with fills, savings sum
correctly, the per-gate replays produce the right shape on a
synthetic snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from tools.postmortem.sections.counterfactual import (
    GateActiveWindow,
    GateResult,
    apply_active_windows_to_fills,
    render_markdown_section,
    replay_momentum_gate,
    replay_post_swing_gate,
    replay_toxicity_hard_gate,
)


@dataclass
class _FakeSnap:
    """Minimal snapshot stub — enough to feed the replay functions
    without needing real JSON files."""

    fills: pd.DataFrame = field(default_factory=pd.DataFrame)
    inventory: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict[str, Any] = field(default_factory=dict)


def _ts(seconds_offset: float, *, base: datetime = datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)) -> pd.Timestamp:
    return pd.Timestamp(base + timedelta(seconds=seconds_offset))


# ---------------------------------------------------------------------------
# apply_active_windows_to_fills — the framework core
# ---------------------------------------------------------------------------


def test_apply_windows_empty_fills_returns_empty_result() -> None:
    res = apply_active_windows_to_fills(
        pd.DataFrame(),
        [],
        gate_name="t",
        description="t",
    )
    assert res.suppressed_fill_count == 0
    assert res.fire_count == 0


def test_apply_windows_no_windows_yields_no_suppression() -> None:
    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(0), _ts(60), _ts(120)],
            "side": ["BUY", "SELL", "BUY"],
            "size": [1.0, 1.0, 1.0],
            "price": [1.0, 1.0, 1.0],
            "notional": [1.0, 1.0, 1.0],
            "fee": [-0.0005, -0.0005, -0.0005],
            "markout_5s_bps": [-1.0, -2.0, -3.0],
        }
    )
    res = apply_active_windows_to_fills(
        fills, [], gate_name="t", description="t"
    )
    assert res.suppressed_fill_count == 0
    assert res.total_session_fills == 3


def test_apply_windows_suppresses_fills_inside_window() -> None:
    """Two of three fills fall inside the [10s, 70s] window. Their
    notional, markout_dollar, and rebate sum into the result."""
    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(5), _ts(30), _ts(60), _ts(120)],
            "side": ["BUY", "SELL", "BUY", "SELL"],
            "size": [10.0, 10.0, 10.0, 10.0],
            "price": [1.0, 1.0, 1.0, 1.0],
            "notional": [10.0, 10.0, 10.0, 10.0],
            "fee": [-0.005, -0.005, -0.005, -0.005],
            "markout_5s_bps": [+5.0, -10.0, -20.0, +1.0],  # bps
        }
    )
    windows = [GateActiveWindow(start_ts=_ts(10), end_ts=_ts(70), reason="x")]
    res = apply_active_windows_to_fills(
        fills, windows, gate_name="t", description="t"
    )
    assert res.suppressed_fill_count == 2  # 30s + 60s fills
    assert res.suppressed_fill_indices == [1, 2]
    assert res.suppressed_fill_notional_usd == pytest.approx(20.0)
    # markout dollars: -10 bps × $10 / 10000 + -20 × $10 / 10000 = -0.03
    assert res.suppressed_markout_dollar_usd == pytest.approx(-0.03)
    # rebate: -fees * 2 fills = +0.01
    assert res.suppressed_rebate_usd == pytest.approx(0.01)
    # net = markout + (-rebate) = -0.03 - 0.01 = -0.04 (gate would have helped)
    assert res.net_impact_usd == pytest.approx(-0.04)
    assert res.fire_count == 1


def test_apply_windows_active_fraction() -> None:
    """Active seconds / session seconds. Half-open intervals: a
    [10, 70] window inside a [0, 120] session → 60/120 = 50%."""
    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(0), _ts(120)],
            "side": ["BUY", "SELL"],
            "size": [1.0, 1.0],
            "price": [1.0, 1.0],
            "notional": [1.0, 1.0],
            "fee": [-0.0005, -0.0005],
            "markout_5s_bps": [0.0, 0.0],
        }
    )
    windows = [GateActiveWindow(start_ts=_ts(10), end_ts=_ts(70))]
    res = apply_active_windows_to_fills(
        fills, windows, gate_name="t", description="t",
        session_start=_ts(0), session_end=_ts(120),
    )
    assert res.active_fraction_of_session == pytest.approx(0.5)


def test_apply_windows_closed_pnl_only_when_nonzero() -> None:
    """All-zero closed_pnl (pre-v18 placeholder) → suppressed_closed_pnl_usd is None."""
    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(30)],
            "side": ["BUY"],
            "size": [1.0], "price": [1.0],
            "notional": [1.0], "fee": [-0.0005],
            "markout_5s_bps": [0.0],
            "closed_pnl": [0.0],  # placeholder — pre-v18 behaviour
        }
    )
    windows = [GateActiveWindow(start_ts=_ts(0), end_ts=_ts(60))]
    res = apply_active_windows_to_fills(
        fills, windows, gate_name="t", description="t"
    )
    assert res.suppressed_closed_pnl_usd is None  # signals "data not available"

    # With at least one non-zero closed_pnl, the field is populated.
    fills2 = fills.copy()
    fills2.loc[0, "closed_pnl"] = 0.05
    res2 = apply_active_windows_to_fills(
        fills2, windows, gate_name="t", description="t"
    )
    assert res2.suppressed_closed_pnl_usd == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# replay_post_swing_gate — drives the live gate logic on equity series
# ---------------------------------------------------------------------------


def test_post_swing_replay_no_equity_returns_empty() -> None:
    snap = _FakeSnap()
    res = replay_post_swing_gate(snap)
    assert res.gate_name == "post_swing"
    assert res.fire_count == 0


def test_post_swing_replay_fires_on_synthetic_spike() -> None:
    """Synthetic equity series: 70s of flat, then a $1 spike. Gate
    should fire (default threshold $0.50 over 60s)."""
    rows = []
    for i in range(0, 71, 5):
        rows.append({"ts": _ts(i), "equity_usd": 1000.0, "realized_pnl_usd": 0.0, "unrealized_pnl_usd": 0.0})
    rows.append({"ts": _ts(72), "equity_usd": 1001.0, "realized_pnl_usd": 1.0, "unrealized_pnl_usd": 0.0})
    eq = pd.DataFrame(rows)

    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(75), _ts(100)],
            "side": ["BUY", "SELL"],
            "size": [10.0, 10.0],
            "price": [1.0, 1.0],
            "notional": [10.0, 10.0],
            "fee": [-0.005, -0.005],
            "markout_5s_bps": [-5.0, +1.0],
        }
    )
    snap = _FakeSnap(fills=fills, equity=eq)
    res = replay_post_swing_gate(snap)
    assert res.fire_count >= 1
    # Both fills (75s, 100s) should fall inside the 120s cooldown
    # starting at the spike (~72s).
    assert res.suppressed_fill_count >= 1


# ---------------------------------------------------------------------------
# replay_momentum_gate — per-fill drift × inventory check
# ---------------------------------------------------------------------------


def test_momentum_replay_long_uptrend_suppresses_buy() -> None:
    """Long inventory + positive drift, BUY fill arrives → gate
    fires (suppresses the BUY because adding to long during
    up-trend is FOMO)."""
    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(100)],
            "side": ["BUY"],
            "size": [10.0],
            "price": [1.0],
            "notional": [10.0],
            "fee": [-0.005],
            "markout_5s_bps": [-5.0],
            "mid_at_fill": [1.0],
            "mid_return_500ms_bps_at_fill": [+5.0],  # uptrend
        }
    )
    inv = pd.DataFrame(
        {
            "ts": [_ts(50)],
            "position_qty": [50.0],  # 50/100 = 50% util > 40% threshold
        }
    )
    snap = _FakeSnap(
        fills=fills,
        inventory=inv,
        config={"MAX_ABS_POSITION": 100.0, "MAX_POSITION_NOTIONAL_USD": 0.0},
    )
    res = replay_momentum_gate(snap)
    assert res.suppressed_fill_count == 1
    assert res.fire_count == 1


def test_momentum_replay_anti_aligned_does_not_fire() -> None:
    """Long + downtrend = anti-aligned. Gate stays out (existing
    inventory_exec_bias / adverse_side_pause cover this)."""
    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(100)],
            "side": ["BUY"],
            "size": [10.0],
            "price": [1.0],
            "notional": [10.0],
            "fee": [-0.005],
            "markout_5s_bps": [-5.0],
            "mid_at_fill": [1.0],
            "mid_return_500ms_bps_at_fill": [-5.0],  # downtrend (anti-aligned)
        }
    )
    inv = pd.DataFrame({"ts": [_ts(50)], "position_qty": [50.0]})
    snap = _FakeSnap(
        fills=fills,
        inventory=inv,
        config={"MAX_ABS_POSITION": 100.0},
    )
    res = replay_momentum_gate(snap)
    assert res.suppressed_fill_count == 0


# ---------------------------------------------------------------------------
# replay_toxicity_hard_gate — rolling-N-fill markout window
# ---------------------------------------------------------------------------


def test_toxicity_hard_replay_fires_on_sustained_adverse_burst() -> None:
    """N consecutive adverse fills with avg ≤ -hard_threshold → trigger fires."""
    # 25 fills, all -10 bp markout → 25-fill avg is -10 bp, trips hard=8.0.
    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(i * 10) for i in range(30)],
            "side": ["BUY"] * 30,
            "size": [10.0] * 30,
            "price": [1.0] * 30,
            "notional": [10.0] * 30,
            "fee": [-0.005] * 30,
            "markout_5s_bps": [-10.0] * 30,
        }
    )
    snap = _FakeSnap(fills=fills)
    res = replay_toxicity_hard_gate(
        snap, markout_hard_bps=8.0, fill_window=25, cooldown_seconds=180.0
    )
    assert res.fire_count >= 1
    assert res.suppressed_fill_count > 0


def test_toxicity_hard_replay_does_not_fire_on_calm_session() -> None:
    """Low-magnitude markouts → no trigger."""
    fills = pd.DataFrame(
        {
            "ts_fill": [_ts(i * 10) for i in range(50)],
            "side": ["BUY"] * 50,
            "size": [10.0] * 50,
            "price": [1.0] * 50,
            "notional": [10.0] * 50,
            "fee": [-0.005] * 50,
            "markout_5s_bps": [-1.0] * 50,
        }
    )
    snap = _FakeSnap(fills=fills)
    res = replay_toxicity_hard_gate(snap, markout_hard_bps=8.0, fill_window=25)
    assert res.fire_count == 0


# ---------------------------------------------------------------------------
# render_markdown_section — output shape sanity
# ---------------------------------------------------------------------------


def test_render_markdown_section_empty() -> None:
    assert render_markdown_section([]) == ""


def test_render_markdown_section_includes_caveats_and_table() -> None:
    res = GateResult(
        gate_name="test_gate",
        description="test description",
        suppressed_fill_count=3,
        suppressed_markout_dollar_usd=-0.10,
        suppressed_rebate_usd=+0.02,
        total_session_fills=100,
        fire_count=2,
        active_seconds=60.0,
        total_session_seconds=600.0,
    )
    out = render_markdown_section([res])
    assert "Counterfactual gate analysis" in out
    assert "Caveats" in out
    assert "test_gate" in out
    assert "test description" in out
    # Table column header
    assert "| Gate | Fires |" in out
