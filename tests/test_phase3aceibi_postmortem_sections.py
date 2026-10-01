"""Phase 3A + 3B + 3C + 3E + 3I (v1.4.187) — postmortem-only batch.

Covers four new postmortem surfaces shipped in v1.4.187:

* **Phase 3I** — inventory edge-support proof. Buckets fills by
  ``util_at_fill`` into low/mid/high and verdicts the inventory-
  skew levers as OK vs. under-tuned based on the high-vs-low markout
  gap.
* **Phase 3C** — ``position_vs_drift`` axis classifier on
  ``annotate_fills_with_regimes`` + the new
  ``per_position_vs_drift_table`` aggregator.
* **Phase 3A + 3E** — regime-mode summary section: per-mode time,
  fills, notional, rebate, markout, net edge, win rate, plus the
  full transitions log with wall-clock interpolation.
* **Phase 3B** — regime-FSM counterfactual (PnL decomposition).
  Compares actual-mode markouts to NORMAL baseline.

Each section is exercised through:
* Pure math (verdict thresholds, bucket assignments).
* Renderer smoke (markdown + HTML emit cleanly on all branches).
* Edge cases (empty fills, missing columns, insufficient samples).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tools.postmortem.sections.inventory_edge_support import (
    VERDICT_INSUFFICIENT_DATA as IES_INSUFFICIENT,
    VERDICT_SKEW_OK,
    VERDICT_SKEW_UNDERTUNED,
    detect_inventory_edge_support_findings,
    render_html_section as render_ies_html,
    render_markdown_section as render_ies_md,
)
from tools.postmortem.sections.regime_modes import (
    detect_regime_modes_findings,
    render_html_section as render_rm_html,
    render_markdown_section as render_rm_md,
)
from tools.postmortem.sections.regime_mode_counterfactual import (
    detect_regime_fsm_counterfactual_findings,
    render_html_section as render_rmc_html,
    render_markdown_section as render_rmc_md,
)
from tools.postmortem.sections.aggregator import (
    per_position_vs_drift_table,
)


# ===========================================================================
# Phase 3I — inventory edge-support
# ===========================================================================


def _ies_fills(
    *,
    n_low: int,
    n_mid: int,
    n_high: int,
    mo_low: float,
    mo_mid: float,
    mo_high: float,
) -> pd.DataFrame:
    rows = (
        [{"util_at_fill": 0.1, "markout_5s_bps": mo_low}] * n_low
        + [{"util_at_fill": 0.5, "markout_5s_bps": mo_mid}] * n_mid
        + [{"util_at_fill": 0.85, "markout_5s_bps": mo_high}] * n_high
    )
    return pd.DataFrame(rows)


def test_ies_verdict_skew_ok_when_high_close_to_low() -> None:
    """High-bucket markout within 2 bp of low → skew OK."""
    df = _ies_fills(
        n_low=30, n_mid=30, n_high=30,
        mo_low=0.0, mo_mid=-1.0, mo_high=-1.5,
    )
    f = detect_inventory_edge_support_findings(
        annotated_fills=df,
        snapshot_name="t", bot_version="v", captured_at="t",
    )
    # high − low = -1.5 − 0.0 = -1.5 → above -2.0 threshold → OK.
    assert f.verdict == VERDICT_SKEW_OK
    assert f.high_minus_low_bps == pytest.approx(-1.5, rel=1e-6)


def test_ies_verdict_skew_undertuned_when_high_much_worse() -> None:
    """High-bucket markout > 2 bp worse than low → under-tuned."""
    df = _ies_fills(
        n_low=30, n_mid=30, n_high=30,
        mo_low=0.0, mo_mid=-2.0, mo_high=-5.0,
    )
    f = detect_inventory_edge_support_findings(
        annotated_fills=df,
        snapshot_name="t", bot_version="v", captured_at="t",
    )
    # high − low = -5.0 → below -2.0 threshold → under-tuned.
    assert f.verdict == VERDICT_SKEW_UNDERTUNED
    assert f.high_minus_low_bps == pytest.approx(-5.0, rel=1e-6)


def test_ies_insufficient_when_high_bucket_thin() -> None:
    """< 20 fills in high bucket → insufficient_data verdict."""
    df = _ies_fills(
        n_low=30, n_mid=30, n_high=5,
        mo_low=0.0, mo_mid=-1.0, mo_high=-10.0,
    )
    f = detect_inventory_edge_support_findings(
        annotated_fills=df,
        snapshot_name="t", bot_version="v", captured_at="t",
    )
    assert f.verdict == IES_INSUFFICIENT
    # No delta computed.
    assert f.high_minus_low_bps is None


def test_ies_empty_fills_returns_insufficient() -> None:
    f = detect_inventory_edge_support_findings(
        annotated_fills=pd.DataFrame(),
        snapshot_name="t", bot_version="v", captured_at="t",
    )
    assert f.verdict == IES_INSUFFICIENT


def test_ies_missing_columns_returns_insufficient() -> None:
    """Missing ``util_at_fill`` or ``markout_5s_bps`` columns →
    insufficient_data (defensive)."""
    f = detect_inventory_edge_support_findings(
        annotated_fills=pd.DataFrame({"x": [1, 2, 3]}),
        snapshot_name="t", bot_version="v", captured_at="t",
    )
    assert f.verdict == IES_INSUFFICIENT


def test_ies_render_markdown_smoke() -> None:
    df = _ies_fills(
        n_low=30, n_mid=30, n_high=30,
        mo_low=0.0, mo_mid=-1.0, mo_high=-5.0,
    )
    f = detect_inventory_edge_support_findings(
        annotated_fills=df,
        snapshot_name="t", bot_version="v", captured_at="t",
    )
    md = render_ies_md(f)
    assert "Inventory edge-support" in md
    assert "under-tuned" in md


def test_ies_render_html_smoke() -> None:
    df = _ies_fills(
        n_low=30, n_mid=30, n_high=30,
        mo_low=0.0, mo_mid=-1.0, mo_high=-1.5,
    )
    f = detect_inventory_edge_support_findings(
        annotated_fills=df,
        snapshot_name="t", bot_version="v", captured_at="t",
    )
    html = render_ies_html(f)
    assert "inventory-edge-support" in html
    assert "<table>" in html


# ===========================================================================
# Phase 3C — position_vs_drift axis + aggregator
# ===========================================================================


def _annotated_for_pvd(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal annotated-fills DataFrame for the
    position_vs_drift classifier. Adds the columns the aggregator
    table expects but doesn't touch."""
    df = pd.DataFrame(rows)
    if "fee" not in df.columns:
        df["fee"] = -0.0006
    if "notional" not in df.columns:
        df["notional"] = 6.0
    if "side" not in df.columns:
        df["side"] = "BUY"
    return df


def test_pvd_aggregator_groups_buckets_correctly() -> None:
    """5 fills per bucket; aggregator returns one row per bucket
    that had ≥ 1 fill."""
    rows = (
        [{"position_vs_drift": "ALIGNED_LONG_UP",
          "markout_5s_bps": 1.0}] * 5
        + [{"position_vs_drift": "ANTI_LONG_DOWN",
            "markout_5s_bps": -5.0}] * 5
        + [{"position_vs_drift": "FLAT_OR_NEUTRAL_DRIFT",
            "markout_5s_bps": 0.0}] * 5
    )
    df = _annotated_for_pvd(rows)
    out = per_position_vs_drift_table(df)
    assert not out.empty
    labels = set(out["position_vs_drift"].tolist())
    assert "ALIGNED_LONG_UP" in labels
    assert "ANTI_LONG_DOWN" in labels
    assert "FLAT_OR_NEUTRAL_DRIFT" in labels


def test_pvd_axis_classifier_via_annotate_fills_with_regimes() -> None:
    """End-to-end: ``annotate_fills_with_regimes`` produces the
    ``position_vs_drift`` column on real-shaped input."""
    from tools.postmortem.sections.regimes import (
        annotate_fills_with_regimes,
    )
    fills = pd.DataFrame(
        {
            "ts_fill": pd.to_datetime(
                ["2026-05-21T10:00:00Z", "2026-05-21T10:01:00Z"]
            ),
            "side": ["BUY", "SELL"],
            "price": [2.05, 2.05],
            "size": [3.0, 3.0],
            "notional": [6.15, 6.15],
            "fee": [-0.0006, -0.0006],
            "markout_5s_bps": [1.0, -1.0],
            "mid_return_500ms_bps_at_fill": [+8.0, -8.0],
            # mid_at_fill required by annotate_fills_with_regimes to
            # compute util_at_fill from the effective-cap formula.
            "mid_at_fill": [2.05, 2.05],
        }
    )
    inventory = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2026-05-21T09:59:59Z", "2026-05-21T10:00:59Z"]
            ),
            "position_qty": [5.0, -5.0],
            "position_notional": [10.25, -10.25],
            "mark_price": [2.05, 2.05],
            "unrealized_pnl_usd": [0.0, 0.0],
        }
    )
    quotes = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                ["2026-05-21T09:59:59Z", "2026-05-21T10:00:59Z"]
            ),
            "vol_estimate": [0.5, 0.5],
            "toxicity_score": [0.1, 0.1],
        }
    )
    out = annotate_fills_with_regimes(
        fills, quotes, inventory,
        max_abs_position=10.0,
        max_position_notional_usd=20.0,
    )
    assert "position_vs_drift" in out.columns
    # Fill 1: long +5 + up drift → ALIGNED_LONG_UP.
    # Fill 2: short -5 + down drift → ALIGNED_SHORT_DOWN.
    labels = set(out["position_vs_drift"].tolist())
    assert "ALIGNED_LONG_UP" in labels or "ALIGNED_SHORT_DOWN" in labels


# ===========================================================================
# Phase 3A + 3E — regime modes
# ===========================================================================


def _snap_with_regime(
    *,
    time_normal: float = 1000.0,
    time_defensive: float = 200.0,
    time_shock: float = 100.0,
    transitions: list[dict] = None,
    seconds_in_mode: float = 50.0,
    mode: str = "NORMAL",
    duration_seconds: float = 1300.0,
    now_utc: str = "2026-05-21T10:00:00+00:00",
) -> SimpleNamespace:
    return SimpleNamespace(
        meta={"bot_version": "1.4.187", "captured_at_utc": now_utc},
        snapshot_dir=SimpleNamespace(name="snap-t"),
        session_summary={
            "duration_seconds": duration_seconds,
            "now_utc": now_utc,
        },
        state_current={
            "behavioural_gates": {
                "regime_mode": {
                    "mode": mode,
                    "seconds_in_mode": seconds_in_mode,
                    "transition_count": len(transitions or []),
                    "time_in_normal_seconds": time_normal,
                    "time_in_defensive_seconds": time_defensive,
                    "time_in_shock_seconds": time_shock,
                    "recent_transitions": transitions or [],
                }
            }
        },
    )


def test_regime_modes_per_mode_times_populated() -> None:
    snap = _snap_with_regime(
        time_normal=900.0, time_defensive=300.0, time_shock=100.0
    )
    f = detect_regime_modes_findings(
        snap=snap, annotated_fills=pd.DataFrame()
    )
    times = {m.mode: m.time_seconds for m in f.per_mode}
    assert times == {"NORMAL": 900.0, "DEFENSIVE": 300.0, "SHOCK": 100.0}
    pcts = {m.mode: m.time_pct for m in f.per_mode}
    # 1300s duration; normal/total ≈ 0.692
    assert pcts["NORMAL"] == pytest.approx(900.0 / 1300.0, rel=1e-3)


def test_regime_modes_per_fill_assignment_walks_transition_timeline() -> None:
    """Build a session with one DEFENSIVE window and assert fills
    landing inside it are tagged DEFENSIVE."""
    transitions = [
        # Bot started NORMAL, went DEFENSIVE 600 s before snapshot,
        # exited 300 s before snapshot.
        {"from": "NORMAL", "to": "DEFENSIVE",
         "ts_mono": 1000.0, "reason": "test"},
        {"from": "DEFENSIVE", "to": "NORMAL",
         "ts_mono": 1300.0, "reason": "test_exit"},
    ]
    # snapshot now_utc is the anchor — set so that:
    # last transition ts_mono=1300, seconds_in_mode=300 → now_mono=1600
    # delta from now_mono to first transition (1000) = 600s ago.
    snap = _snap_with_regime(
        transitions=transitions,
        seconds_in_mode=300.0,
        mode="NORMAL",
        now_utc="2026-05-21T10:00:00+00:00",
    )
    # Fills: one before first transition (NORMAL), one inside the
    # DEFENSIVE window (450 s before now), one after exit (NORMAL).
    fills = pd.DataFrame(
        {
            "ts_fill": pd.to_datetime(
                [
                    "2026-05-21T09:48:00+00:00",  # 720 s before now → before transition
                    "2026-05-21T09:52:30+00:00",  # 450 s before now → DEFENSIVE
                    "2026-05-21T09:58:00+00:00",  # 120 s before now → NORMAL again
                ]
            ),
            "notional": [6.0, 6.0, 6.0],
            "fee": [-0.0006, -0.0006, -0.0006],
            "markout_5s_bps": [1.0, -2.0, 0.5],
            "side": ["BUY", "SELL", "BUY"],
        }
    )
    f = detect_regime_modes_findings(snap=snap, annotated_fills=fills)
    counts = {m.mode: m.n_fills for m in f.per_mode}
    assert counts["DEFENSIVE"] == 1
    # 2 fills in NORMAL (one before transition, one after exit).
    assert counts["NORMAL"] == 2


def test_regime_modes_render_markdown_no_transitions() -> None:
    snap = _snap_with_regime()  # no transitions
    f = detect_regime_modes_findings(snap=snap, annotated_fills=pd.DataFrame())
    md = render_rm_md(f)
    assert "Regime modes" in md
    assert "No mode transitions" in md


def test_regime_modes_render_html_smoke() -> None:
    snap = _snap_with_regime()
    f = detect_regime_modes_findings(snap=snap, annotated_fills=pd.DataFrame())
    html = render_rm_html(f)
    assert "regime-modes" in html
    assert "<table>" in html


# ===========================================================================
# Phase 3B — regime-FSM counterfactual
# ===========================================================================


def test_counterfactual_skipped_on_pure_normal_session() -> None:
    """No DEFENSIVE/SHOCK windows → counterfactual table is empty
    + the renderer emits the "stayed in NORMAL" note."""
    snap = _snap_with_regime()  # no transitions
    # Sufficient NORMAL fills.
    fills = pd.DataFrame(
        {
            "ts_fill": pd.to_datetime(
                ["2026-05-21T09:50:00+00:00"] * 10
            ),
            "notional": [6.0] * 10,
            "fee": [-0.0006] * 10,
            "markout_5s_bps": [0.5] * 10,
        }
    )
    f = detect_regime_fsm_counterfactual_findings(
        snap=snap, annotated_fills=fills
    )
    # No DEFENSIVE/SHOCK fills → per_mode list is empty.
    assert f.per_mode == []
    md = render_rmc_md(f)
    assert "stayed in NORMAL" in md


def test_counterfactual_insufficient_normal_baseline() -> None:
    """< 5 NORMAL-mode fills → baseline can't be computed → section
    renders the "insufficient baseline" note."""
    snap = _snap_with_regime()
    fills = pd.DataFrame(
        {
            "ts_fill": pd.to_datetime(
                ["2026-05-21T09:50:00+00:00"] * 3
            ),
            "notional": [6.0] * 3,
            "fee": [-0.0006] * 3,
            "markout_5s_bps": [0.5] * 3,
        }
    )
    f = detect_regime_fsm_counterfactual_findings(
        snap=snap, annotated_fills=fills
    )
    assert f.normal_baseline_mean_bps is None
    md = render_rmc_md(f)
    assert "Insufficient NORMAL-mode fills" in md


def test_counterfactual_computes_delta_when_data_sufficient() -> None:
    """NORMAL baseline +0.5 bp, DEFENSIVE fills at +2.0 bp → Δ per
    fill = 0.5 − 2.0 = -1.5 bp (negative → FSM helped)."""
    transitions = [
        {"from": "NORMAL", "to": "DEFENSIVE",
         "ts_mono": 1000.0, "reason": "test"},
        {"from": "DEFENSIVE", "to": "NORMAL",
         "ts_mono": 1300.0, "reason": "test_exit"},
    ]
    snap = _snap_with_regime(
        transitions=transitions, seconds_in_mode=300.0,
    )
    # 10 NORMAL fills at +0.5; 10 DEFENSIVE fills at +2.0.
    fills = pd.DataFrame(
        {
            "ts_fill": pd.to_datetime(
                ["2026-05-21T09:48:00+00:00"] * 10
                + ["2026-05-21T09:52:30+00:00"] * 10
            ),
            "notional": [6.0] * 20,
            "fee": [-0.0006] * 20,
            "markout_5s_bps": [0.5] * 10 + [2.0] * 10,
        }
    )
    f = detect_regime_fsm_counterfactual_findings(
        snap=snap, annotated_fills=fills
    )
    assert f.normal_baseline_mean_bps == pytest.approx(0.5)
    defensive = next(m for m in f.per_mode if m.mode == "DEFENSIVE")
    assert defensive.actual_mean_markout_5s_bps == pytest.approx(2.0)
    assert defensive.delta_per_fill_bps == pytest.approx(-1.5, rel=1e-6)
    # Negative delta → FSM helped.
    md = render_rmc_md(f)
    assert "FSM helped" in md


def test_counterfactual_render_html_smoke() -> None:
    transitions = [
        {"from": "NORMAL", "to": "DEFENSIVE",
         "ts_mono": 1000.0, "reason": "test"},
        {"from": "DEFENSIVE", "to": "NORMAL",
         "ts_mono": 1300.0, "reason": "test_exit"},
    ]
    snap = _snap_with_regime(
        transitions=transitions, seconds_in_mode=300.0,
    )
    fills = pd.DataFrame(
        {
            "ts_fill": pd.to_datetime(
                ["2026-05-21T09:48:00+00:00"] * 10
                + ["2026-05-21T09:52:30+00:00"] * 10
            ),
            "notional": [6.0] * 20,
            "fee": [-0.0006] * 20,
            "markout_5s_bps": [0.5] * 10 + [-3.0] * 10,
        }
    )
    f = detect_regime_fsm_counterfactual_findings(
        snap=snap, annotated_fills=fills
    )
    html = render_rmc_html(f)
    assert "regime-fsm-counterfactual" in html


# ===========================================================================
# Violin chart height bump (v1.4.187)
# ===========================================================================


def test_violin_chart_height_is_3x_taller() -> None:
    """Sentinel: the violin row in the multi-row plotly chart was
    bumped from 0.2 (240 px @ height=1200) to a 6/14 row-height share
    (720 px @ height=1680). Source-text check — behaviour requires
    rendering plotly which is heavyweight for a unit test."""
    import inspect
    from tools.postmortem.render import plotly as plt_render

    src = inspect.getsource(plt_render.render_html)
    # The row_heights tuple holds the violin row weight; the new
    # value is 6 / 14 of total → expressed as the literal "6" in the
    # row_heights list. Anchor on the height literal too.
    assert "row_heights=[3, 2.5, 2.5, 6]" in src
    assert "height=1680" in src
