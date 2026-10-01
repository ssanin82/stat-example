"""Phase 3H (v1.4.186) — positive-regime expectancy proof.

Three layers:

1. **Filter logic** — fills are filtered by ``vol_regime == "LOW_VOL"``;
   when the column is missing the function falls back gracefully.

2. **Verdict math** — three thresholds (n ≥ min_fills, mean > +0.5 bp,
   95 % bootstrap CI lower bound > 0). Each verdict label is exercised:
   ``positive_expectancy_proven``, ``positive_expectancy_unproven``,
   ``negative_expectancy``, ``insufficient_fills``, ``no_data``.

3. **Renderer smoke** — markdown + HTML render without error for each
   verdict label.

Plus a small reproducibility test: the bootstrap CI is deterministic
under a fixed seed (no flaky test runs).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.postmortem.sections.positive_regime_expectancy import (
    DEFAULT_REBATE_BPS,
    PositiveRegimeFindings,
    VERDICT_INSUFFICIENT_FILLS,
    VERDICT_NEGATIVE,
    VERDICT_NO_DATA,
    VERDICT_PROVEN,
    VERDICT_UNPROVEN,
    _bootstrap_ci,
    _classify_verdict,
    detect_positive_regime_expectancy_findings,
    render_html_section,
    render_markdown_section,
)


def _fills(
    n: int,
    *,
    markout_5s: float = -1.0,
    vol_regime: str = "LOW_VOL",
    fee_per_fill: float = -0.0006,
    notional: float = 6.0,
) -> pd.DataFrame:
    """Build an annotated-fills DataFrame for tests."""
    return pd.DataFrame(
        {
            "markout_5s_bps": [markout_5s] * n,
            "vol_regime": [vol_regime] * n,
            "fee": [fee_per_fill] * n,
            "notional": [notional] * n,
        }
    )


# ---------------------------------------------------------------------------
# Filter logic
# ---------------------------------------------------------------------------


def test_filters_to_low_vol_only() -> None:
    """Mixed regime — only LOW_VOL counts."""
    df = pd.concat(
        [
            _fills(50, markout_5s=+0.5, vol_regime="LOW_VOL"),
            _fills(50, markout_5s=-5.0, vol_regime="HIGH_VOL"),
        ],
        ignore_index=True,
    )
    out = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        min_fills_for_verdict=50,
    )
    assert out.n_fills_total == 100
    assert out.n_fills_positive_regime == 50
    # Mean edge ~ +0.5 + 1.0 (rebate) = +1.5 bp on LOW_VOL fills only.
    assert out.mean_net_edge_bps == pytest.approx(1.5, rel=1e-2)


def test_missing_vol_regime_column_uses_all_fills() -> None:
    """Defensive fallback: no vol_regime column → use all fills, do
    not crash."""
    df = pd.DataFrame(
        {
            "markout_5s_bps": [0.0] * 10,
            "fee": [-0.0006] * 10,
            "notional": [6.0] * 10,
        }
    )
    out = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        min_fills_for_verdict=50,
    )
    # All 10 used (column missing); insufficient for verdict.
    assert out.n_fills_positive_regime == 10
    assert out.verdict == VERDICT_INSUFFICIENT_FILLS


def test_empty_fills_returns_no_data() -> None:
    out = detect_positive_regime_expectancy_findings(
        annotated_fills=pd.DataFrame(),
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
    )
    assert out.verdict == VERDICT_NO_DATA
    assert out.n_fills_positive_regime == 0


def test_all_fills_high_vol_returns_no_data() -> None:
    """No LOW_VOL fills landed → no_data even though session had
    fills."""
    df = _fills(200, vol_regime="HIGH_VOL")
    out = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
    )
    assert out.verdict == VERDICT_NO_DATA


# ---------------------------------------------------------------------------
# Verdict math
# ---------------------------------------------------------------------------


def test_verdict_proven_when_all_three_thresholds_clear() -> None:
    """200 fills, markout +1.5 bp/fill, rebate +1 bp → net +2.5
    bp/fill. Mean clearly > +0.5 bp; CI lower bound > 0 with
    200 samples + 1.0 bp signal. Should be proven."""
    # Use markout +1.5; rebate from fee = 1.0 bp; total = +2.5 bp/fill.
    df = _fills(200, markout_5s=1.5)
    out = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        min_fills_for_verdict=100,
    )
    assert out.verdict == VERDICT_PROVEN
    assert out.mean_net_edge_bps == pytest.approx(2.5, rel=1e-2)
    assert out.ci_low_bps is not None and out.ci_low_bps > 0


def test_verdict_negative_when_ci_upper_below_zero() -> None:
    """200 fills, markout -8 bp/fill, rebate +1 bp → net -7 bp/fill.
    Strongly negative. Bootstrap CI upper bound stays below 0 → the
    bot is provably losing alpha in positive regime."""
    df = _fills(200, markout_5s=-8.0)
    out = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        min_fills_for_verdict=100,
    )
    assert out.verdict == VERDICT_NEGATIVE
    assert out.mean_net_edge_bps is not None and out.mean_net_edge_bps < 0
    assert out.ci_high_bps is not None and out.ci_high_bps < 0


def test_verdict_unproven_when_mean_below_threshold() -> None:
    """200 fills, markout -0.7 bp, rebate +1 bp → net +0.3 bp/fill.
    Just barely positive, but the mean doesn't clear the +0.5 bp pass
    threshold AND the CI may include zero. Should be unproven."""
    df = _fills(200, markout_5s=-0.7)
    out = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        min_fills_for_verdict=100,
    )
    # Net +0.3 — not enough to pass.
    assert out.verdict == VERDICT_UNPROVEN


def test_verdict_insufficient_fills_when_n_below_threshold() -> None:
    """50 fills, would otherwise be proven, but n < min_fills."""
    df = _fills(50, markout_5s=2.0)
    out = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
        min_fills_for_verdict=100,
    )
    assert out.verdict == VERDICT_INSUFFICIENT_FILLS


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_classify_verdict_pure() -> None:
    """Pure verdict-classification rule, exercised at every boundary."""
    # n < min_fills → insufficient regardless of edge.
    assert (
        _classify_verdict(
            n=50,
            mean_edge_bps=5.0,
            ci_low=4.0,
            ci_high=6.0,
            min_fills=100,
            pass_threshold_bps=0.5,
        )
        == VERDICT_INSUFFICIENT_FILLS
    )
    # CI upper < 0 → negative.
    assert (
        _classify_verdict(
            n=200,
            mean_edge_bps=-5.0,
            ci_low=-7.0,
            ci_high=-3.0,
            min_fills=100,
            pass_threshold_bps=0.5,
        )
        == VERDICT_NEGATIVE
    )
    # Mean > threshold AND CI lower > 0 → proven.
    assert (
        _classify_verdict(
            n=200,
            mean_edge_bps=2.0,
            ci_low=1.0,
            ci_high=3.0,
            min_fills=100,
            pass_threshold_bps=0.5,
        )
        == VERDICT_PROVEN
    )
    # Mean below threshold → unproven.
    assert (
        _classify_verdict(
            n=200,
            mean_edge_bps=0.3,
            ci_low=0.1,
            ci_high=0.5,
            min_fills=100,
            pass_threshold_bps=0.5,
        )
        == VERDICT_UNPROVEN
    )
    # Mean above threshold but CI low ≤ 0 → unproven.
    assert (
        _classify_verdict(
            n=200,
            mean_edge_bps=0.8,
            ci_low=-0.1,
            ci_high=1.5,
            min_fills=100,
            pass_threshold_bps=0.5,
        )
        == VERDICT_UNPROVEN
    )


def test_bootstrap_ci_is_deterministic_under_fixed_seed() -> None:
    """Same input + same seed → identical CI. Prevents flaky tests."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 20)
    lo1, hi1 = _bootstrap_ci(values, n_resamples=500, ci_pct=95.0, seed=42)
    lo2, hi2 = _bootstrap_ci(values, n_resamples=500, ci_pct=95.0, seed=42)
    assert lo1 == lo2
    assert hi1 == hi2


# ---------------------------------------------------------------------------
# Renderer smoke
# ---------------------------------------------------------------------------


def _findings(verdict: str) -> PositiveRegimeFindings:
    return PositiveRegimeFindings(
        snapshot_name="snap-x",
        bot_version="1.4.186",
        captured_at="2026-05-21T10:00:00Z",
        n_fills_total=200,
        n_fills_positive_regime=150,
        rebate_bps_per_fill=DEFAULT_REBATE_BPS,
        mean_net_edge_bps=1.5,
        median_net_edge_bps=1.4,
        stdev_net_edge_bps=2.0,
        ci_low_bps=1.0,
        ci_high_bps=2.0,
        verdict=verdict,
    )


@pytest.mark.parametrize(
    "verdict",
    [
        VERDICT_PROVEN,
        VERDICT_NEGATIVE,
        VERDICT_UNPROVEN,
        VERDICT_INSUFFICIENT_FILLS,
        VERDICT_NO_DATA,
    ],
)
def test_render_markdown_section_for_each_verdict(verdict: str) -> None:
    md = render_markdown_section(_findings(verdict))
    assert "Positive-regime expectancy" in md
    # Verdict chip text always present.
    chip_words = {
        VERDICT_PROVEN: "proven",
        VERDICT_NEGATIVE: "negative",
        VERDICT_UNPROVEN: "unproven",
        VERDICT_INSUFFICIENT_FILLS: "insufficient",
        VERDICT_NO_DATA: "no data",
    }[verdict]
    assert chip_words in md


@pytest.mark.parametrize(
    "verdict",
    [
        VERDICT_PROVEN,
        VERDICT_NEGATIVE,
        VERDICT_UNPROVEN,
        VERDICT_INSUFFICIENT_FILLS,
        VERDICT_NO_DATA,
    ],
)
def test_render_html_section_for_each_verdict(verdict: str) -> None:
    html = render_html_section(_findings(verdict))
    assert "positive-regime-expectancy" in html
    assert "<table>" in html


# ---------------------------------------------------------------------------
# Reproducibility — same fills, same verdict
# ---------------------------------------------------------------------------


def test_reproducible_on_same_inputs() -> None:
    """Same fills DataFrame + default seed → same verdict + same
    headline numbers across two invocations."""
    df = _fills(200, markout_5s=1.5)
    a = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
    )
    b = detect_positive_regime_expectancy_findings(
        annotated_fills=df,
        snapshot_name="t",
        bot_version="v",
        captured_at="t",
    )
    assert a.verdict == b.verdict
    assert a.mean_net_edge_bps == b.mean_net_edge_bps
    assert a.ci_low_bps == b.ci_low_bps
    assert a.ci_high_bps == b.ci_high_bps
