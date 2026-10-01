"""Regression tests for the regime-observability Phase 3 regime labels
+ summary tables.

Pure post-processing — runs on fills_since.json + exposure_since.json
without touching the bot. Tests cover:

- Bucket assignment functions: basis, inventory, quote-age, aggressiveness,
  mode, trend, spread, quintile.
- Per-axis aggregation: ``regime_summary_fill``.
- Exposure-time denominator: ``regime_summary_exposure``.
- Cross-tabs: ``regime_summary_cross``.
- Net-edge rollup: ``regime_summary_net_edge``.
- Block-bootstrap CI: ``block_bootstrap_ci``.
- End-to-end ``generate_summaries`` on a fixture stats dir.

See ``plans/regime-observability.md`` Phase 3.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

# Resolve ``scripts/`` module path for direct import.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from regime_summary import (  # noqa: E402
    aggressiveness_bucket,
    assign_buckets_to_exposure_bars,
    assign_buckets_to_fills,
    assign_quintile,
    basis_bucket_from_sign_and_ewma,
    block_bootstrap_ci,
    generate_summaries,
    inventory_bucket,
    mode_bucket,
    quintile_breakpoints,
    quote_age_bucket,
    regime_summary_cross,
    regime_summary_exposure,
    regime_summary_fill,
    regime_summary_net_edge,
    sample_size_label,
    spread_bucket,
    trend_bucket,
)


# --------------------------------------------------------------------- #
# Bucket helpers                                                         #
# --------------------------------------------------------------------- #


def test_basis_bucket_categorical_mapping() -> None:
    assert basis_bucket_from_sign_and_ewma(0, 0.0) == "FLAT"
    assert basis_bucket_from_sign_and_ewma(-1, 0.0001) == "NEG"
    assert basis_bucket_from_sign_and_ewma(-1, 0.0010) == "NEG_STRETCHED"
    assert basis_bucket_from_sign_and_ewma(+1, 0.0001) == "POS"
    assert basis_bucket_from_sign_and_ewma(+1, 0.0010) == "POS_STRETCHED"
    assert basis_bucket_from_sign_and_ewma(None, None) is None
    # Fallback when ewma is missing — direction-only.
    assert basis_bucket_from_sign_and_ewma(-1, None) == "NEG"


def test_inventory_bucket_thresholds() -> None:
    assert inventory_bucket(0.0) == "FLAT"
    assert inventory_bucket(0.05) == "FLAT"
    assert inventory_bucket(0.20) == "LOW"
    assert inventory_bucket(0.50) == "MED"
    assert inventory_bucket(0.80) == "HIGH"
    assert inventory_bucket(-0.20) == "LOW"  # uses abs
    assert inventory_bucket(None) is None


def test_quote_age_bucket_thresholds() -> None:
    assert quote_age_bucket(50) == "<250ms"
    assert quote_age_bucket(500) == "250-1000ms"
    assert quote_age_bucket(1500) == "1-2s"
    assert quote_age_bucket(3000) == "2-5s"
    assert quote_age_bucket(10000) == "5s+"
    assert quote_age_bucket(None) is None


def test_aggressiveness_bucket_uses_distance() -> None:
    assert aggressiveness_bucket("at_touch", None) == "AT_TOUCH"
    assert aggressiveness_bucket("inside", None) == "INSIDE"
    assert aggressiveness_bucket("aged_tightened", None) == "AGED_TIGHTENED"
    assert aggressiveness_bucket("behind_touch", 1.0) == "BEHIND_1"
    assert aggressiveness_bucket("behind_touch", 2.5) == "BEHIND_2PLUS"
    assert aggressiveness_bucket("behind_touch", None) == "BEHIND_1"
    assert aggressiveness_bucket("unknown", None) == "UNKNOWN"
    assert aggressiveness_bucket(None, None) is None


def test_mode_bucket_eligibility_takes_priority() -> None:
    assert mode_bucket("BOTH", "HOLD_ALL") == "HOLD_ALL"
    assert mode_bucket(None, "QUOTE_BID_ONLY") == "BID_ONLY"
    assert mode_bucket(None, "QUOTE_ASK_ONLY") == "ASK_ONLY"
    assert mode_bucket(None, "QUOTE_BOTH") == "BOTH"
    # Fall back to active_sides.
    assert mode_bucket("BID_ONLY", None) == "BID_ONLY"
    assert mode_bucket("NONE", None) == "HOLD_ALL"


def test_trend_bucket_thresholds() -> None:
    assert trend_bucket(5.0) == "UP_DRIFT"
    assert trend_bucket(-5.0) == "DOWN_DRIFT"
    assert trend_bucket(0.5) == "FLAT"
    assert trend_bucket(None) is None


def test_spread_bucket_tick_aware() -> None:
    # tick = 1.0 bp; spread = 2.0 → 2 ticks → 2tick.
    assert spread_bucket(2.0, 1.0) == "2tick"
    assert spread_bucket(0.5, 1.0) == "1tick"
    assert spread_bucket(5.0, 1.0) == "3tick+"
    # No tick info: fallback to absolute bps bands.
    assert spread_bucket(3.0, None) == "1tick"
    assert spread_bucket(7.0, None) == "2tick"
    assert spread_bucket(15.0, None) == "3tick+"


def test_quintile_breakpoints_too_few_samples() -> None:
    assert quintile_breakpoints([1.0, 2.0, 3.0]) is None  # n < 10
    bps = quintile_breakpoints(list(range(20)))
    assert bps is not None
    assert len(bps) == 4


def test_assign_quintile() -> None:
    bps = [10.0, 20.0, 30.0, 40.0]
    assert assign_quintile(5, bps) == "Q1_LOW"
    assert assign_quintile(15, bps) == "Q2"
    assert assign_quintile(25, bps) == "Q3"
    assert assign_quintile(35, bps) == "Q4"
    assert assign_quintile(50, bps) == "Q5_HIGH"
    assert assign_quintile(None, bps) is None
    assert assign_quintile(15, None) is None


# --------------------------------------------------------------------- #
# Block-bootstrap CI                                                     #
# --------------------------------------------------------------------- #


def test_block_bootstrap_ci_returns_none_for_small_n() -> None:
    assert block_bootstrap_ci([1.0, 2.0, 3.0]) is None  # n < 30


def test_block_bootstrap_ci_returns_triple_for_large_n() -> None:
    vals = [1.0] * 50
    ci = block_bootstrap_ci(vals)
    assert ci is not None
    mean, lo, hi = ci
    # Constant input → CI degenerate but well-defined.
    assert abs(mean - 1.0) < 1e-9
    assert abs(lo - 1.0) < 1e-9
    assert abs(hi - 1.0) < 1e-9


def test_block_bootstrap_ci_widens_with_variance() -> None:
    import random as _r

    _r.seed(42)
    vals = [_r.uniform(-10, 10) for _ in range(100)]
    ci = block_bootstrap_ci(vals)
    assert ci is not None
    mean, lo, hi = ci
    # 95% CI for n=100 uniform(-10,10) should be reasonably tight,
    # but non-trivial.
    assert hi > lo
    assert hi - lo > 0.5


def test_sample_size_labels() -> None:
    assert sample_size_label(10) == "anecdote"
    assert sample_size_label(50) == "hypothesis"
    assert sample_size_label(100) == "directional"
    assert sample_size_label(500) == "trustworthy"


# --------------------------------------------------------------------- #
# Aggregations                                                           #
# --------------------------------------------------------------------- #


def _make_fill(**overrides) -> dict:
    base = {
        "fill_id": "f1",
        "side": "BUY",
        "price": 100.0,
        "size": 1.0,
        "notional": 100.0,
        "fee": -0.01,
        "closed_pnl": 0.0,
        "markout_1s_bps": -1.0,
        "markout_5s_bps": -2.0,
        "quote_age_at_fill_ms": 500,
        "basis_regime_sign": -1,
        "binance_basis_ewma_at_decision": 0.0003,
        "inventory_utilization_before_fill": 0.4,
        "quote_aggressiveness": "behind_touch",
        "quote_distance_to_touch_ticks_at_placement": 1.0,
        "active_sides_at_decision": "BOTH",
        "quote_eligibility_state": "QUOTE_BOTH",
        "mid_return_500ms_bps_at_fill": 0.0,
        "spread_bps_at_fill": 4.0,
        "vol_estimate_at_decision": 10.0,
        "toxicity_score_at_decision": 0.3,
    }
    base.update(overrides)
    return base


def _make_bar(**overrides) -> dict:
    base = {
        "session_id": "s",
        "ts_bar": "2026-05-13T12:00:00+00:00",
        "mid": 100.0,
        "spread_bps": 4.0,
        "inventory_qty": 1.0,
        "inventory_utilization": 0.125,
        "active_sides": "BOTH",
        "quote_eligibility": "QUOTE_BOTH",
        "toxicity_score": 0.3,
        "vol_estimate": 10.0,
        "binance_basis_ewma": 0.0003,
        "basis_regime_sign": -1,
    }
    base.update(overrides)
    return base


def test_assign_buckets_to_fills_populates_axes() -> None:
    fills = [_make_fill()]
    out = assign_buckets_to_fills(
        fills, vol_breaks=[5.0, 10.0, 15.0, 20.0], toxicity_breaks=[0.1, 0.2, 0.3, 0.4]
    )
    assert len(out) == 1
    f = out[0]
    assert f["basis_bucket"] == "NEG"
    assert f["inventory_bucket"] == "MED"
    assert f["quote_age_bucket"] == "250-1000ms"
    assert f["aggressiveness_bucket"] == "BEHIND_1"
    assert f["mode_bucket"] == "BOTH"
    assert f["trend_bucket"] == "FLAT"
    assert f["spread_bucket"] in {"1tick", "2tick", "3tick+"}
    assert f["vol_bucket"] in {"Q1_LOW", "Q2", "Q3", "Q4", "Q5_HIGH"}
    assert f["toxicity_bucket"] in {"Q1_LOW", "Q2", "Q3", "Q4", "Q5_HIGH"}


def test_regime_summary_fill_per_side_per_axis() -> None:
    fills = [
        _make_fill(side="BUY", markout_5s_bps=-3.0),
        _make_fill(side="BUY", markout_5s_bps=-1.0),
        _make_fill(side="SELL", markout_5s_bps=0.5),
    ]
    fills = assign_buckets_to_fills(fills)
    summary = regime_summary_fill(fills, axes=["basis_bucket"])
    by_axis = summary["by_axis"]
    assert "basis_bucket" in by_axis
    buy_buckets = by_axis["basis_bucket"]["BUY"]
    assert "NEG" in buy_buckets
    assert buy_buckets["NEG"]["n"] == 2
    assert buy_buckets["NEG"]["mean_markout_5s_bps"] == -2.0
    assert by_axis["basis_bucket"]["SELL"]["NEG"]["n"] == 1


def test_regime_summary_exposure_minutes_calculation() -> None:
    bars = assign_buckets_to_exposure_bars([_make_bar()] * 12)
    summary = regime_summary_exposure(bars, cadence_seconds=5.0)
    assert summary["total_bars"] == 12
    assert summary["total_minutes"] == 1.0  # 12 * 5 / 60
    basis_buckets = summary["by_axis"]["basis_bucket"]
    assert "NEG" in basis_buckets
    assert basis_buckets["NEG"]["minutes_exposed"] == 1.0
    assert basis_buckets["NEG"]["pct_of_session"] == 100.0


def test_regime_summary_cross_pairs_fills_with_exposure() -> None:
    fills = assign_buckets_to_fills([_make_fill()])
    bars = assign_buckets_to_exposure_bars([_make_bar()] * 6)
    summary = regime_summary_cross(
        fills, bars, cross_axes=[("side", "basis_bucket")]
    )
    cross = summary["by_cross"]["side__x__basis_bucket"]
    assert "BUY" in cross
    assert "NEG" in cross["BUY"]
    assert cross["BUY"]["NEG"]["n"] == 1
    # Exposure minutes attached when applicable.
    assert "minutes_exposed" in cross["BUY"]["NEG"]
    assert cross["BUY"]["NEG"]["minutes_exposed"] == 0.5  # 6 * 5 / 60


def test_regime_summary_net_edge_combines_rebate_markout_pnl() -> None:
    fills = assign_buckets_to_fills(
        [
            _make_fill(markout_5s_bps=-2.0, notional=100.0, fee=-0.01),
            _make_fill(markout_5s_bps=-2.0, notional=100.0, fee=-0.01),
        ]
    )
    bars = assign_buckets_to_exposure_bars([_make_bar()] * 12)
    summary = regime_summary_net_edge(fills, bars)
    assert len(summary["rows"]) >= 1
    row = summary["rows"][0]
    assert row["side"] == "BUY"
    assert row["n_fills"] == 2
    assert row["rebate_usd"] == 0.02  # 2 * 0.01
    # markout_5s_dollars: -2 bp on $200 notional total = -$0.04;
    # inverted → +0.04 net contribution = 0.04 from rebate + (-0.04) markout.
    assert abs(row["markout_5s_dollars"] - 0.04) < 1e-6 or row["markout_5s_dollars"] == 0.04
    # net = rebate + (-markout_dollars) + closed_pnl
    # = 0.02 + 0.04 + 0 = 0.06
    assert abs(row["net_dollars"] - 0.06) < 1e-6
    assert row["minutes_in_basis_regime"] == 1.0


# --------------------------------------------------------------------- #
# End-to-end: generate_summaries on a fixture stats dir                  #
# --------------------------------------------------------------------- #


def test_generate_summaries_writes_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stats_dir = Path(tmp) / "stats"
        stats_dir.mkdir()
        # Minimal fixture: 50 fills, 100 exposure bars (enough for
        # bootstrap CI + quintile breaks).
        fills = [_make_fill(fill_id=f"f{i}") for i in range(50)]
        bars = [_make_bar() for _ in range(100)]
        (stats_dir / "fills_since.json").write_text(
            json.dumps(fills), encoding="utf-8"
        )
        (stats_dir / "exposure_since.json").write_text(
            json.dumps(bars), encoding="utf-8"
        )
        written = generate_summaries(stats_dir)
        assert "regime_summary_fill" in written
        assert "regime_summary_exposure" in written
        assert "regime_summary_cross" in written
        assert "regime_summary_net_edge" in written
        # All files exist.
        for path in written.values():
            assert path.is_file()
            payload = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(payload, dict)


def test_generate_summaries_handles_missing_input() -> None:
    """Pre-Phase-2 snapshots have no exposure_since.json. Function
    should skip the exposure-dependent outputs and still produce
    fill summary."""
    with tempfile.TemporaryDirectory() as tmp:
        stats_dir = Path(tmp) / "stats"
        stats_dir.mkdir()
        fills = [_make_fill(fill_id=f"f{i}") for i in range(50)]
        (stats_dir / "fills_since.json").write_text(
            json.dumps(fills), encoding="utf-8"
        )
        written = generate_summaries(stats_dir)
        # fill / cross / net_edge present (rely only on fills);
        # exposure absent because there are no bars.
        assert "regime_summary_fill" in written
        assert "regime_summary_exposure" not in written
        assert "regime_summary_cross" in written
        assert "regime_summary_net_edge" in written


def test_generate_summaries_handles_empty_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stats_dir = Path(tmp) / "stats"
        stats_dir.mkdir()
        (stats_dir / "fills_since.json").write_text("[]", encoding="utf-8")
        (stats_dir / "exposure_since.json").write_text(
            "[]", encoding="utf-8"
        )
        written = generate_summaries(stats_dir)
        # Empty inputs → no outputs.
        assert written == {}
