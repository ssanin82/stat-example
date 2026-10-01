"""v1.4.100 O2 — ``rung_regime`` axis in postmortem regime classification.

The regimes layer adds a new per-fill column ``rung_regime`` derived
from ``level_idx``. Tests verify:

* L0/L1/L2/L3 bucket to "L0"/"L1"/"L2"/"L3" labels respectively.
* level_idx ≥ 4 rolls into "L4+".
* level_idx missing → "UNKNOWN".
* The aggregator's per_rung_regime_table groups fills correctly and
  drops UNKNOWN rows.
"""

from __future__ import annotations

import pandas as pd

from tools.postmortem.sections.aggregator import per_rung_regime_table
from tools.postmortem.sections.regimes import (
    RegimeThresholds,
    annotate_fills_with_regimes,
)


def _fills(level_idxs: list[int | None]) -> pd.DataFrame:
    rows = []
    for i, lvl in enumerate(level_idxs):
        row = {
            "fill_id": f"f{i}",
            "ts_fill": pd.Timestamp("2026-05-19T00:00:00Z") + pd.Timedelta(seconds=i),
            "side": "BUY" if i % 2 == 0 else "SELL",
            "price": 2.0,
            "size": 1.0,
            "notional": 2.0,
            "fee": -0.001,
            "markout_5s_bps": 1.0,
            "mid_at_fill": 2.0,
        }
        if lvl is not None:
            row["level_idx"] = lvl
        rows.append(row)
    return pd.DataFrame(rows)


def _empty_quotes() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "vol_estimate", "toxicity_score"])


def _empty_inventory() -> pd.DataFrame:
    return pd.DataFrame(columns=["ts", "position_qty"])


def test_rung_regime_column_added_with_correct_buckets() -> None:
    fills = _fills([0, 1, 2, 3, 5, None])
    out = annotate_fills_with_regimes(
        fills,
        _empty_quotes(),
        _empty_inventory(),
        max_abs_position=10.0,
        thresholds=RegimeThresholds(),
    )
    assert "rung_regime" in out.columns
    labels = list(out["rung_regime"])
    # Two fills hit the L0/L1/L2/L3 inner buckets; one hits L4+; one
    # unknown (no level_idx — coerces NaN through fillna(0)? in our
    # implementation we use ``getattr(...,'level_idx',...)`` so a
    # missing column COL routes to UNKNOWN.
    # Note: the fill ordering may change after annotate_fills_with_regimes
    # sorts by ts_fill — preserve that.
    assert "L0" in labels
    assert "L1" in labels
    assert "L2" in labels
    assert "L3" in labels
    assert "L4+" in labels


def test_rung_regime_unknown_when_column_absent() -> None:
    """When the fills DataFrame has no ``level_idx`` column at all,
    every fill buckets to UNKNOWN — and the per_rung_regime_table
    drops UNKNOWN rows, so the resulting table is empty."""
    fills = _fills([0, 1])
    fills = fills.drop(columns=["level_idx"])
    out = annotate_fills_with_regimes(
        fills,
        _empty_quotes(),
        _empty_inventory(),
        max_abs_position=10.0,
        thresholds=RegimeThresholds(),
    )
    assert (out["rung_regime"] == "UNKNOWN").all()
    table = per_rung_regime_table(out)
    assert table.empty


def test_per_rung_regime_table_groups_fills_correctly() -> None:
    """The aggregator emits one row per non-UNKNOWN rung_regime bucket
    with the correct fill_count."""
    fills = _fills([0, 0, 0, 1, 1])
    out = annotate_fills_with_regimes(
        fills,
        _empty_quotes(),
        _empty_inventory(),
        max_abs_position=10.0,
        thresholds=RegimeThresholds(),
    )
    table = per_rung_regime_table(out)
    assert not table.empty
    by_rung = dict(zip(table["rung_regime"], table["fill_count"]))
    assert by_rung.get("L0") == 3
    assert by_rung.get("L1") == 2
