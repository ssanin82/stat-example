"""Tests for ``tools.postmortem.sections.basis_ic_tuning``.

The section is purely retrospective — joins fills to quote_decisions
by timestamp, bins by |basis_ic|, reports markout per bin and per
candidate threshold. These tests synthesise small fill / quote
frames and assert the joining + binning logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from tools.postmortem.sections.basis_ic_tuning import (
    BasisIcTuningAnalysis,
    compute_analysis,
    render_html_section,
    render_markdown_section,
)


@dataclass
class _FakeSnapshot:
    """Minimal snapshot stub — only the fields ``compute_analysis``
    actually reads. Mirrors the shape of ``SnapshotData`` for the
    purposes of this section."""

    fills: pd.DataFrame
    quotes: pd.DataFrame


def _make_quotes(ic_series: list[float], step_s: float = 1.0) -> pd.DataFrame:
    """Build a quote_decisions frame with monotonically-increasing ts
    and the supplied basis_ic values. ts is in UTC."""
    base = pd.Timestamp("2026-05-16T00:00:00", tz="UTC")
    return pd.DataFrame({
        "ts": [base + pd.Timedelta(seconds=i * step_s) for i in range(len(ic_series))],
        "basis_ic": ic_series,
    })


def _make_fills(ts_offsets_s: list[float], markouts: list[float]) -> pd.DataFrame:
    """Build a fills frame where each fill is offset_s seconds AFTER
    the quotes base ts. markouts list is 1:1 with ts_offsets."""
    assert len(ts_offsets_s) == len(markouts)
    base = pd.Timestamp("2026-05-16T00:00:00", tz="UTC")
    return pd.DataFrame({
        "ts_fill": [base + pd.Timedelta(seconds=o) for o in ts_offsets_s],
        "markout_5s_bps": markouts,
        "side": ["BUY"] * len(ts_offsets_s),
    })


def test_no_fills_returns_unavailable() -> None:
    snap = _FakeSnapshot(
        fills=pd.DataFrame(),
        quotes=_make_quotes([0.05, 0.10, 0.15]),
    )
    analysis = compute_analysis(snap)
    assert analysis.available is False
    assert "no fills" in analysis.reason_unavailable


def test_missing_basis_ic_column_returns_unavailable() -> None:
    """Pre-1.1.129 snapshots have no basis_ic column → graceful skip."""
    quotes_no_ic = pd.DataFrame({
        "ts": [pd.Timestamp("2026-05-16T00:00:00", tz="UTC")],
        "some_other_col": [1.0],
    })
    fills = _make_fills([1.0], [-0.5])
    snap = _FakeSnapshot(fills=fills, quotes=quotes_no_ic)
    analysis = compute_analysis(snap)
    assert analysis.available is False
    assert "basis_ic" in analysis.reason_unavailable


def test_missing_markout_returns_unavailable() -> None:
    quotes = _make_quotes([0.1, 0.2])
    fills = pd.DataFrame({
        "ts_fill": [pd.Timestamp("2026-05-16T00:00:01", tz="UTC")],
        "side": ["BUY"],
        # No markout_5s_bps column.
    })
    snap = _FakeSnapshot(fills=fills, quotes=quotes)
    analysis = compute_analysis(snap)
    assert analysis.available is False


def test_asof_join_attaches_most_recent_ic() -> None:
    """A fill at t=2.5s should pick up the quote at t=2s (most recent
    backward match), not the quote at t=3s (future)."""
    # Use 0.21 (strictly > 0.20) so the bin-edge semantics of pd.cut
    # (right=True, so 0.20 lands in 0.15-0.20) don't confuse the test.
    quotes = _make_quotes([0.0, 0.1, 0.21, 0.3], step_s=1.0)  # ts=0,1,2,3
    # Fill at t=2.5 → should match the quote at t=2 (basis_ic=0.21)
    fills = _make_fills([2.5], [-1.0])
    snap = _FakeSnapshot(fills=fills, quotes=quotes)
    analysis = compute_analysis(snap)
    assert analysis.available is True
    assert analysis.joined_fills == 1
    # The single fill landed in the 0.20-0.30 bin.
    bin_row = analysis.bin_table[analysis.bin_table["ic_bin"] == "0.20-0.30"]
    assert int(bin_row["n"].iloc[0]) == 1


def test_fill_beyond_lookback_window_is_dropped() -> None:
    """asof tolerance is 30s. A fill that's 60s past the latest quote
    should be dropped from the join — we don't attribute a stale IC
    to a fresh fill."""
    quotes = _make_quotes([0.2], step_s=1.0)  # one quote at t=0
    fills = _make_fills([60.0], [-1.0])  # fill 60s after the quote
    snap = _FakeSnapshot(fills=fills, quotes=quotes)
    analysis = compute_analysis(snap)
    # Either available=False (no joined fills) or joined_fills=0.
    if analysis.available:
        assert analysis.joined_fills == 0
    else:
        assert "joined" in analysis.reason_unavailable or "no fills" in analysis.reason_unavailable


def test_per_bin_aggregation_correct() -> None:
    """Three fills land in three different IC bins; check counts and
    means per bin."""
    # ICs: 0.02, 0.12, 0.25 → bins 0.00-0.05, 0.10-0.15, 0.20-0.30
    quotes = _make_quotes([0.02, 0.12, 0.25], step_s=1.0)
    # Markouts: bin 0.00-0.05 → -1.0, bin 0.10-0.15 → -2.0, bin 0.20-0.30 → -3.0
    fills = _make_fills([0.5, 1.5, 2.5], [-1.0, -2.0, -3.0])
    snap = _FakeSnapshot(fills=fills, quotes=quotes)
    analysis = compute_analysis(snap)
    assert analysis.available is True
    bt = analysis.bin_table
    assert int(bt[bt["ic_bin"] == "0.00-0.05"]["n"].iloc[0]) == 1
    assert bt[bt["ic_bin"] == "0.00-0.05"]["mean_markout_5s_bps"].iloc[0] == pytest.approx(-1.0)
    assert int(bt[bt["ic_bin"] == "0.10-0.15"]["n"].iloc[0]) == 1
    assert bt[bt["ic_bin"] == "0.10-0.15"]["mean_markout_5s_bps"].iloc[0] == pytest.approx(-2.0)
    assert int(bt[bt["ic_bin"] == "0.20-0.30"]["n"].iloc[0]) == 1
    assert bt[bt["ic_bin"] == "0.20-0.30"]["mean_markout_5s_bps"].iloc[0] == pytest.approx(-3.0)
    # All bins are below the 30-fill threshold → none trustworthy.
    assert not bt["trustworthy"].any()


def test_threshold_partition_correct() -> None:
    """Build a 100-fill dataset where fills above |IC|=0.15 have
    systematically worse markout than fills below. Check that the
    threshold table reflects this."""
    # 60 fills at IC=0.05 with markout 0 (low-IC, benign).
    # 40 fills at IC=0.25 with markout -5 (high-IC, adverse).
    ics = [0.05] * 60 + [0.25] * 40
    markouts = [0.0] * 60 + [-5.0] * 40
    quotes = _make_quotes(ics, step_s=0.5)
    # Each fill 0.1s after its matching quote.
    offsets = [i * 0.5 + 0.1 for i in range(100)]
    fills = _make_fills(offsets, markouts)
    snap = _FakeSnapshot(fills=fills, quotes=quotes)
    analysis = compute_analysis(snap)
    assert analysis.available is True
    assert analysis.joined_fills == 100

    tt = analysis.threshold_table
    # At threshold 0.15: above = 40 fills (IC=0.25), below = 60 (IC=0.05).
    row_015 = tt[abs(tt["threshold"] - 0.15) < 1e-9].iloc[0]
    assert int(row_015["above_n"]) == 40
    assert int(row_015["below_n"]) == 60
    assert row_015["above_mean_markout_5s_bps"] == pytest.approx(-5.0)
    assert row_015["below_mean_markout_5s_bps"] == pytest.approx(0.0)
    # Spread = above - below = -5.0 - 0.0 = -5.0 (above is worse)
    assert row_015["spread_bps"] == pytest.approx(-5.0)
    # Trustworthy because both partitions have >=30.
    assert row_015["trustworthy"] is True or row_015["trustworthy"] == 1


def test_suggested_threshold_maximises_spread() -> None:
    """A clean dataset where the discrimination is sharpest at 0.20
    should produce suggested_threshold=0.20."""
    # 50 fills at IC=0.05, markout 0.
    # 50 fills at IC=0.25, markout -10.
    # At threshold 0.05: above=50 IC=0.25, below=50 IC=0.05, spread=-10
    # At threshold 0.20: above=50 IC=0.25, below=50 IC=0.05, spread=-10
    # At threshold 0.30: above=0, below=100 → no partition.
    # Tied at 0.05-0.20. Implementation picks the first largest (0.05).
    # So this test verifies the suggestion is among the maximum-spread
    # candidates rather than a specific value.
    ics = [0.05] * 50 + [0.25] * 50
    markouts = [0.0] * 50 + [-10.0] * 50
    quotes = _make_quotes(ics, step_s=0.5)
    offsets = [i * 0.5 + 0.1 for i in range(100)]
    fills = _make_fills(offsets, markouts)
    snap = _FakeSnapshot(fills=fills, quotes=quotes)
    analysis = compute_analysis(snap)
    assert analysis.available is True
    assert analysis.suggested_threshold is not None
    # The suggested threshold should be one of the ones that captures
    # the boundary between 0.05 and 0.25 (so 0.05-0.20 all valid).
    assert 0.05 <= analysis.suggested_threshold <= 0.20


def test_render_markdown_skipped_when_unavailable() -> None:
    """The renderer returns the skip-stub when analysis is
    unavailable so the report doesn't grow an empty section."""
    analysis = BasisIcTuningAnalysis(
        available=False, reason_unavailable="no quote_decisions in snapshot"
    )
    md = render_markdown_section(analysis)
    assert "Skipped" in md
    assert "no quote_decisions" in md


def test_render_markdown_includes_all_sections_when_available() -> None:
    quotes = _make_quotes([0.05] * 50 + [0.25] * 50, step_s=0.5)
    offsets = [i * 0.5 + 0.1 for i in range(100)]
    fills = _make_fills(offsets, [0.0] * 50 + [-10.0] * 50)
    snap = _FakeSnapshot(fills=fills, quotes=quotes)
    analysis = compute_analysis(snap)
    md = render_markdown_section(analysis)
    assert "Basis-IC threshold tuning" in md
    assert "gradient view" in md.lower() or "|IC| bin" in md
    assert "Threshold candidates" in md
    assert "current" in md.lower()
    assert "Interpretation" in md


def test_render_html_skipped_when_unavailable() -> None:
    analysis = BasisIcTuningAnalysis(available=False)
    assert render_html_section(analysis) == ""


def test_render_html_emits_section_when_available() -> None:
    quotes = _make_quotes([0.05] * 50 + [0.25] * 50, step_s=0.5)
    offsets = [i * 0.5 + 0.1 for i in range(100)]
    fills = _make_fills(offsets, [0.0] * 50 + [-10.0] * 50)
    snap = _FakeSnapshot(fills=fills, quotes=quotes)
    analysis = compute_analysis(snap)
    html = render_html_section(analysis)
    assert "<section" in html
    assert "Basis-IC threshold tuning" in html
    assert "<table" in html
