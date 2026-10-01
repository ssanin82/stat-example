"""Tests for postmortem per-axis breakdown tables —
``plans/telemetry.md`` Steps 2 and 3.

Covers:
  * ``per_basis_regime_table`` (basis_regime axis)
  * ``per_hour_regime_table`` (hour_regime axis)
  * ``per_quote_age_table`` (quote-age bucket axis)
  * empty-input handling (pre-v19 snapshots / all-UNKNOWN data)
  * render integration: tables appear in the markdown output.
"""

from __future__ import annotations

import pandas as pd

from tools.postmortem.sections.aggregator import (
    per_basis_regime_table,
    per_hour_regime_table,
    per_quote_age_table,
)


def _fills_with_basis_regimes() -> pd.DataFrame:
    """Synth fills annotated with the basis_regime axis already
    populated (as ``regimes.annotate_fills_with_regimes`` would)."""
    return pd.DataFrame(
        {
            "fill_id": [f"f{i}" for i in range(6)],
            "side": ["buy", "sell"] * 3,
            "notional": [7.0] * 6,
            "fee": [-0.0007] * 6,  # rebate
            "markout_5s_bps": [-2.0, -4.0, -1.0, -8.0, 2.0, -3.0],
            "regime": ["NORMAL_VOL-LOW_TOX-FLAT-NO_TREND"] * 6,
            "basis_regime": [
                "POSITIVE_IC",
                "POSITIVE_IC",
                "NEGATIVE_IC",
                "NEGATIVE_IC",
                "WEAK_IC",
                "WEAK_IC",
            ],
            "hour_regime": [
                "ASIA_AM",
                "ASIA_AM",
                "EU_AM",
                "EU_AM",
                "US_AM",
                "US_AM",
            ],
            "net_edge_bps": [-1.0, -3.0, 0.0, -7.0, 3.0, -2.0],
            "quote_age_at_fill_ms": [50, 300, 1200, 5000, 20000, 80],
        }
    )


# ---------------------------------------------------------------------------
# per_basis_regime_table
# ---------------------------------------------------------------------------


def test_per_basis_regime_table_groups_correctly() -> None:
    df = _fills_with_basis_regimes()
    out = per_basis_regime_table(df)
    assert not out.empty
    # 3 distinct basis_regime values → 3 rows.
    assert len(out) == 3
    assert set(out["basis_regime"].tolist()) == {
        "POSITIVE_IC",
        "NEGATIVE_IC",
        "WEAK_IC",
    }
    # POSITIVE_IC: 2 fills, markouts -2 and -4 → median -3, mean -3.
    pos = out[out["basis_regime"] == "POSITIVE_IC"].iloc[0]
    assert pos["fill_count"] == 2
    assert pos["median_markout_5s_bps"] == -3.0
    assert pos["mean_markout_5s_bps"] == -3.0


def test_per_basis_regime_table_empty_input() -> None:
    out = per_basis_regime_table(pd.DataFrame())
    assert out.empty


def test_per_basis_regime_table_missing_column() -> None:
    """Pre-v19 snapshots have no basis_regime column → empty result."""
    df = pd.DataFrame({"fill_id": ["x"], "side": ["buy"], "notional": [7.0]})
    out = per_basis_regime_table(df)
    assert out.empty


def test_per_basis_regime_table_all_unknown_skipped() -> None:
    df = pd.DataFrame(
        {
            "fill_id": ["f1", "f2"],
            "side": ["buy", "sell"],
            "notional": [7.0, 7.0],
            "fee": [-0.0007, -0.0007],
            "markout_5s_bps": [-2.0, -3.0],
            "basis_regime": ["UNKNOWN", "UNKNOWN"],
        }
    )
    out = per_basis_regime_table(df)
    assert out.empty


# ---------------------------------------------------------------------------
# per_hour_regime_table
# ---------------------------------------------------------------------------


def test_per_hour_regime_table_groups_correctly() -> None:
    df = _fills_with_basis_regimes()
    out = per_hour_regime_table(df)
    assert not out.empty
    assert set(out["hour_regime"].tolist()) == {"ASIA_AM", "EU_AM", "US_AM"}


def test_per_hour_regime_table_empty() -> None:
    assert per_hour_regime_table(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# per_quote_age_table
# ---------------------------------------------------------------------------


def test_per_quote_age_table_bucket_assignment() -> None:
    df = _fills_with_basis_regimes()
    out = per_quote_age_table(df)
    assert not out.empty
    # Buckets expected:
    #   50ms  → <100ms
    #   300ms → 100-500ms
    #   1200ms → 500ms-2s
    #   5000ms → 2-10s
    #   20000ms → 10s+
    #   80ms → <100ms
    counts = dict(zip(out["quote_age_bucket"], out["fill_count"]))
    assert counts.get("<100ms") == 2
    assert counts.get("100-500ms") == 1
    assert counts.get("500ms-2s") == 1
    assert counts.get("2-10s") == 1
    assert counts.get("10s+") == 1


def test_per_quote_age_table_missing_column() -> None:
    df = pd.DataFrame({"fill_id": ["x"], "side": ["buy"], "notional": [7.0]})
    out = per_quote_age_table(df)
    assert out.empty


def test_per_quote_age_table_all_null_column() -> None:
    df = pd.DataFrame(
        {
            "fill_id": ["a", "b"],
            "side": ["buy", "sell"],
            "notional": [7.0, 7.0],
            "fee": [-0.0007, -0.0007],
            "markout_5s_bps": [-2.0, -3.0],
            "quote_age_at_fill_ms": [None, None],
        }
    )
    out = per_quote_age_table(df)
    assert out.empty


# ---------------------------------------------------------------------------
# Render integration — markdown
# ---------------------------------------------------------------------------


def test_render_markdown_includes_per_axis_tables() -> None:
    """Smoke test: when the per-axis tables are non-empty, the
    markdown rendered output includes their section headers."""
    from tools.postmortem.render.plotly import render_markdown
    from tools.postmortem.sections.aggregator import SessionTotals

    df = _fills_with_basis_regimes()

    # Minimal SessionTotals stub.
    totals = SessionTotals(
        fills=6,
        fill_rate_per_min=1.0,
        duration_minutes=6.0,
        realized_pnl_usd=-0.01,
        rebate_income_usd=0.004,
        markout_dollar_impact_usd=-0.01,
        residual_usd=-0.004,
        median_5s_markout_bps=-2.5,
        adverse_pct_5s=80.0,
        pct_time_one_sided=50.0,
        suppression_top=[],
    )
    # Minimal SnapshotData stub.
    from types import SimpleNamespace
    snap = SimpleNamespace(
        meta={"bot_profile": "test", "bot_version": "1.2.99"},
        session_summary={},
        state_current={},
    )
    out = render_markdown(
        snap,
        totals,
        regime_table=pd.DataFrame(),
        basis_regime_table=per_basis_regime_table(df),
        hour_regime_table=per_hour_regime_table(df),
        quote_age_table=per_quote_age_table(df),
    )
    assert "Per-axis breakdowns" in out
    assert "PnL by basis regime" in out
    assert "PnL by hour-of-day" in out
    assert "PnL by quote age at fill" in out


def test_render_markdown_skips_empty_tables() -> None:
    """Backward compat: when the new tables are empty / not passed,
    no per-axis section is emitted."""
    from tools.postmortem.render.plotly import render_markdown
    from tools.postmortem.sections.aggregator import SessionTotals
    from types import SimpleNamespace

    totals = SessionTotals(
        fills=0,
        fill_rate_per_min=None,
        duration_minutes=None,
        realized_pnl_usd=None,
        rebate_income_usd=None,
        markout_dollar_impact_usd=None,
        residual_usd=None,
        median_5s_markout_bps=None,
        adverse_pct_5s=None,
        pct_time_one_sided=None,
        suppression_top=[],
    )
    snap = SimpleNamespace(
        meta={"bot_profile": "test", "bot_version": "1.2.99"},
        session_summary={},
        state_current={},
    )
    out = render_markdown(snap, totals, regime_table=pd.DataFrame())
    assert "Per-axis breakdowns" not in out
