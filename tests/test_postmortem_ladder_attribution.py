"""v1.4.100 O1 — per-rung attribution postmortem section.

The section is shaped as compute_per_rung → findings → render
markdown / HTML. Tests pin the compute path with inline-constructed
fill dicts (the storage layer hands the section list-of-dicts via
``to_dict(orient="records")``), so the assertions are independent of
SQLite + the full snapshot loader.

Coverage:
* compute_attribution_per_rung() correctly buckets fills by level_idx
* per-rung notional, rebate, and counts add up
* missing markout values are gracefully tolerated
* zero fills → empty list
* find_findings + render markdown wrap without raising on a typical
  multi-rung session
"""

from __future__ import annotations

from typing import Any

from tools.postmortem.sections.ladder_attribution import (
    compute_attribution_per_rung,
    detect_ladder_attribution_findings,
    render_markdown_section,
    render_html_section,
)


def _fill(
    *,
    level_idx: int = 0,
    side: str = "BUY",
    notional: float = 10.0,
    fee: float = -0.001,
    closed_pnl: float = 0.0,
    m1: float | None = None,
    m5: float | None = None,
    m30: float | None = None,
    m60: float | None = None,
    m120: float | None = None,
) -> dict[str, Any]:
    return {
        "level_idx": level_idx,
        "side": side,
        "notional": notional,
        "fee": fee,
        "closed_pnl": closed_pnl,
        "markout_1s_bps": m1,
        "markout_5s_bps": m5,
        "markout_30s_bps": m30,
        "markout_60s_bps": m60,
        "markout_120s_bps": m120,
    }


def test_empty_fills_returns_empty_list() -> None:
    out = compute_attribution_per_rung([])
    assert out == []


def test_single_rung_groups_into_one_bucket() -> None:
    """All fills at L0 — one entry, n_fills counts everyone."""
    fills = [_fill(level_idx=0) for _ in range(8)]
    out = compute_attribution_per_rung(fills)
    assert len(out) == 1
    assert out[0].level_idx == 0
    assert out[0].n_fills == 8


def test_multi_rung_groups_correctly() -> None:
    """Mixed L0 + L1 fills: two entries, sorted by level_idx."""
    fills = [
        _fill(level_idx=0, side="BUY"),
        _fill(level_idx=0, side="SELL"),
        _fill(level_idx=1, side="BUY"),
        _fill(level_idx=1, side="BUY"),
        _fill(level_idx=1, side="SELL"),
    ]
    out = compute_attribution_per_rung(fills)
    assert [r.level_idx for r in out] == [0, 1]
    by_lvl = {r.level_idx: r for r in out}
    assert by_lvl[0].n_fills == 2
    assert by_lvl[0].n_buy == 1
    assert by_lvl[0].n_sell == 1
    assert by_lvl[1].n_fills == 3
    assert by_lvl[1].n_buy == 2
    assert by_lvl[1].n_sell == 1


def test_notional_and_rebate_sum() -> None:
    """Notional sums per rung; rebate = notional × rebate_bps / 10_000."""
    fills = [
        _fill(level_idx=0, notional=100.0),
        _fill(level_idx=0, notional=50.0),
        _fill(level_idx=1, notional=30.0),
    ]
    out = compute_attribution_per_rung(fills, rebate_bps=1.0)
    by_lvl = {r.level_idx: r for r in out}
    assert by_lvl[0].notional_usd == 150.0
    assert abs(by_lvl[0].rebate_usd - 0.015) < 1e-12  # 150 * 1bp = 0.015 USD
    assert by_lvl[1].notional_usd == 30.0
    assert abs(by_lvl[1].rebate_usd - 0.003) < 1e-12


def test_missing_markouts_tolerated() -> None:
    """Fills with no markout fields produce None per-horizon means
    (n < 5 threshold). Should not raise."""
    fills = [_fill(level_idx=0) for _ in range(4)]  # all-None markouts
    out = compute_attribution_per_rung(fills)
    assert out[0].n_fills == 4
    # All horizons should report None (below n=5 floor).
    for v in out[0].mean_markout_bps_by_horizon_s.values():
        assert v is None


def test_per_horizon_mean_computed_when_sample_large_enough() -> None:
    """With n_with_horizon ≥ 5, mean is computed."""
    fills = [_fill(level_idx=0, m5=1.0) for _ in range(6)]
    out = compute_attribution_per_rung(fills)
    assert out[0].mean_markout_bps_by_horizon_s[5] == 1.0


def test_unknown_level_idx_buckets_to_zero() -> None:
    """A fill with no ``level_idx`` (pre-v1.4.100 row) lands at L0."""
    fills = [{"side": "BUY", "notional": 10.0, "fee": -0.001}]
    out = compute_attribution_per_rung(fills)
    assert len(out) == 1
    assert out[0].level_idx == 0
    assert out[0].n_fills == 1


def test_findings_wrapper_renders_markdown_without_raising() -> None:
    """Smoke: the public entrypoint detect_ladder_attribution_findings
    + render_markdown_section round-trip cleanly for a typical multi-
    rung sample."""
    fills = [_fill(level_idx=0, m5=1.0) for _ in range(10)] + [
        _fill(level_idx=1, m5=-0.5) for _ in range(8)
    ]
    findings = detect_ladder_attribution_findings(
        snapshot_name="test",
        bot_version="1.4.100",
        captured_at="2026-05-19T00:00:00Z",
        fills=fills,
    )
    md = render_markdown_section(findings)
    assert "Ladder per-rung attribution" in md or "ladder" in md.lower()
    html = render_html_section(findings)
    assert "<section" in html
    # has_multi_rung_data should reflect "saw L1 in the session"
    assert findings.has_multi_rung_data is True


def test_findings_empty_renders_skip_stub() -> None:
    """No fills → renderer emits a skipped/empty stub, no exception."""
    findings = detect_ladder_attribution_findings(
        snapshot_name="test",
        bot_version="1.4.100",
        captured_at="2026-05-19T00:00:00Z",
        fills=[],
    )
    md = render_markdown_section(findings)
    html = render_html_section(findings)
    assert isinstance(md, str) and isinstance(html, str)
