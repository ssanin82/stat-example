"""v1.4.100 O4 + O5 — effective-N dwell + gate-cap dwell postmortem section.

The section reads the snapshot's ``quote_decisions`` DataFrame
directly. Tests use a small inline-constructed DataFrame so the
storage layer is not exercised here.

Coverage:
* Empty DataFrame → ``available=False`` (skipped stub).
* Missing required columns → ``available=False`` with a clear reason.
* Per-side dwell counters add up to total rows.
* Dominant requested-levels is the mode of the column.
* Gate-cap aggregation: parses JSON-encoded ``ladder_gate_caps``,
  counts fires per gate, distribution per cap value.
* Markdown + HTML renderers don't raise on either the available or
  the unavailable path.
"""

from __future__ import annotations

import json

import pandas as pd

from tools.postmortem.sections.ladder_effective_n import (
    detect_ladder_effective_n_findings,
    render_html_section,
    render_markdown_section,
)


def _findings(qd: pd.DataFrame | None):
    return detect_ladder_effective_n_findings(
        snapshot_name="test",
        bot_version="1.4.100",
        captured_at="2026-05-19T00:00:00Z",
        quote_decisions=qd,
    )


def test_empty_dataframe_marks_unavailable() -> None:
    out = _findings(pd.DataFrame())
    assert out.available is False
    assert "no quote_decisions" in (out.reason_unavailable or "").lower()


def test_missing_required_columns_marks_unavailable() -> None:
    """Pre-schema-v23 snapshots lack the ladder columns; section
    reports unavailable instead of crashing."""
    qd = pd.DataFrame({"ts": ["2026-05-19T00:00:00Z"], "mid_price": [2.0]})
    out = _findings(qd)
    assert out.available is False
    assert "missing required columns" in (out.reason_unavailable or "").lower()


def test_per_side_dwell_counts() -> None:
    qd = pd.DataFrame(
        {
            "ladder_effective_levels_buy": [2, 2, 1, 1, 0],
            "ladder_effective_levels_sell": [2, 2, 2, 1, 0],
            "ladder_requested_levels": [2, 2, 2, 2, 2],
        }
    )
    out = _findings(qd)
    assert out.available is True
    assert out.total_cycles == 5
    assert out.dwell_buy is not None
    assert out.dwell_sell is not None
    # 2 cycles at N=2, 2 at N=1, 1 at N=0 on BUY side.
    assert out.dwell_buy.by_level == {2: 2, 1: 2, 0: 1}
    # 3 at N=2, 1 at N=1, 1 at N=0 on SELL side.
    assert out.dwell_sell.by_level == {2: 3, 1: 1, 0: 1}
    # Dominant requested = 2 across all rows.
    assert out.dwell_buy.requested_levels_dominant == 2
    assert out.dwell_sell.requested_levels_dominant == 2


def test_dwell_counts_handle_null_as_zero() -> None:
    """A NULL in ladder_effective_levels_* should bucket to N=0
    (no rungs)."""
    qd = pd.DataFrame(
        {
            "ladder_effective_levels_buy": [2, None, 1],
            "ladder_effective_levels_sell": [None, 2, 1],
            "ladder_requested_levels": [2, 2, 2],
        }
    )
    out = _findings(qd)
    assert out.available is True
    assert out.dwell_buy.by_level.get(0, 0) == 1
    assert out.dwell_sell.by_level.get(0, 0) == 1


def test_gate_cap_aggregation() -> None:
    """Three cycles where two gates published caps. Verify per-gate
    fire count + cap distribution."""
    qd = pd.DataFrame(
        {
            "ladder_effective_levels_buy": [1, 1, 1],
            "ladder_effective_levels_sell": [1, 1, 1],
            "ladder_requested_levels": [2, 2, 2],
            "ladder_gate_caps": [
                json.dumps({"microprice_cap": 1, "vol_trend_cap": 2}),
                json.dumps({"microprice_cap": 1}),
                json.dumps({"vol_trend_cap": 2}),
            ],
        }
    )
    out = _findings(qd)
    assert out.available is True
    by_name = {g.gate_name: g for g in out.gate_caps}
    assert "microprice_cap" in by_name
    assert "vol_trend_cap" in by_name
    assert by_name["microprice_cap"].fire_count == 2
    assert by_name["vol_trend_cap"].fire_count == 2
    assert by_name["microprice_cap"].cap_value_distribution == {1: 2}
    assert by_name["vol_trend_cap"].cap_value_distribution == {2: 2}


def test_malformed_gate_caps_cell_does_not_crash() -> None:
    """A garbage cell in ladder_gate_caps must be silently dropped."""
    qd = pd.DataFrame(
        {
            "ladder_effective_levels_buy": [1],
            "ladder_effective_levels_sell": [1],
            "ladder_requested_levels": [2],
            "ladder_gate_caps": ["not-json{"],
        }
    )
    out = _findings(qd)
    assert out.available is True
    assert out.gate_caps == []


def test_markdown_renderer_smoke_available() -> None:
    qd = pd.DataFrame(
        {
            "ladder_effective_levels_buy": [2, 1],
            "ladder_effective_levels_sell": [2, 2],
            "ladder_requested_levels": [2, 2],
            "ladder_gate_caps": [
                json.dumps({}),
                json.dumps({"microprice_cap": 1}),
            ],
        }
    )
    md = render_markdown_section(_findings(qd))
    assert "Ladder effective-N" in md


def test_html_renderer_smoke_unavailable() -> None:
    html = render_html_section(_findings(None))
    assert "<section" in html
    assert "skipped" in html.lower()
