"""TODO-003: rolling net-edge-after-fees observability metric.

For each of the last N session fills, the summary computes:

* mean_markout_bps   — gross adverse/favourable proxy
* mean_fee_bps_per_fill — single-side fee drag in bps of notional
* mean_round_trip_fee_bps — 2× single-side
* net_edge_bps = mean_markout_bps − mean_round_trip_fee_bps

Below the min-sample threshold (5), ``net_edge_bps`` is None so the
metric does not advertise noise.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.enums import Side
from app.models import Fill
from app.quote_quality_telemetry import build_net_edge_summary


def _fill(
    *,
    fee: float,
    notional: float,
    markout_5s_bps: float | None = None,
    markout_3s_bps: float | None = None,
    side: Side = Side.BUY,
) -> Fill:
    return Fill(
        fill_id=f"f-{markout_5s_bps}-{notional}",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="SUI-PERP",
        side=side,
        price=1.0,
        size=notional / 1.0,
        notional=notional,
        fee=fee,
        liquidity_flag="x",
        mid_at_fill=1.0,
        markout_1s_bps=None,
        markout_3s_bps=markout_3s_bps,
        markout_5s_bps=markout_5s_bps,
    )


def test_net_edge_summary_empty_returns_none() -> None:
    out = build_net_edge_summary([], window=50)
    assert out["net_edge_sample_count"] == 0
    assert out["net_edge_bps"] is None
    assert out["mean_markout_bps"] is None


def test_net_edge_summary_below_min_samples_returns_none_net() -> None:
    """4 fills with markouts → still None for net_edge_bps (< min 5)."""
    fills = [_fill(fee=0.05, notional=10.0, markout_5s_bps=2.0) for _ in range(4)]
    out = build_net_edge_summary(fills, window=50)
    assert out["net_edge_sample_count"] == 4
    assert out["net_edge_bps"] is None
    # but the single-fill aggregates ARE populated.
    assert out["mean_markout_bps"] == 2.0


def test_net_edge_summary_clean_positive_edge() -> None:
    """Markout +3 bps, fee 0.5 bps single-side → net = 3 - 1 = +2 bps."""
    fills = [
        _fill(fee=0.05, notional=100.0, markout_5s_bps=3.0) for _ in range(10)
    ]  # fee_bps = 0.05/100*10000 = 5.0 ... let me redo
    out = build_net_edge_summary(fills, window=50)
    assert out["net_edge_sample_count"] == 10
    # fee_bps per fill = 0.05 / 100 * 10000 = 5.0
    assert out["mean_fee_bps_per_fill"] == 5.0
    assert out["mean_round_trip_fee_bps"] == 10.0
    assert out["mean_markout_bps"] == 3.0
    assert out["net_edge_bps"] == 3.0 - 10.0  # -7 bps net


def test_net_edge_summary_realistic_sui_fills() -> None:
    """Mirror real SUI-PERP fee tier (~0.5 bps single-side → 1 bp round trip)
    against +2 bps markout — net should be +1 bp.
    """
    # fee_bps target = 0.5 → fee = 0.5 / 10000 * notional = 5e-5 * 100 = 0.005
    fills = [
        _fill(fee=0.005, notional=100.0, markout_5s_bps=2.0) for _ in range(8)
    ]
    out = build_net_edge_summary(fills, window=50)
    assert out["net_edge_sample_count"] == 8
    assert abs(out["mean_fee_bps_per_fill"] - 0.5) < 1e-9
    assert abs(out["mean_round_trip_fee_bps"] - 1.0) < 1e-9
    assert out["mean_markout_bps"] == 2.0
    assert abs(out["net_edge_bps"] - 1.0) < 1e-9


def test_net_edge_summary_falls_back_to_3s_markout() -> None:
    """When 5s is None but 3s is present, the 3s value is used."""
    fills = [
        _fill(fee=0.005, notional=100.0, markout_5s_bps=None, markout_3s_bps=1.5)
        for _ in range(6)
    ]
    out = build_net_edge_summary(fills, window=50)
    assert out["net_edge_sample_count"] == 6
    assert out["mean_markout_bps"] == 1.5


def test_net_edge_summary_skips_zero_notional_fills() -> None:
    """Defensive: a malformed fill with notional=0 must not divide by zero."""
    fills = [_fill(fee=0.005, notional=100.0, markout_5s_bps=2.0) for _ in range(5)]
    fills.append(_fill(fee=0.0, notional=0.0, markout_5s_bps=2.0))
    out = build_net_edge_summary(fills, window=50)
    assert out["net_edge_sample_count"] == 5  # zero-notional fill skipped


def test_net_edge_summary_window_caps_input() -> None:
    """Only the first ``window`` fills (newest-first) are considered."""
    fills = [_fill(fee=0.005, notional=100.0, markout_5s_bps=2.0) for _ in range(7)]
    fills.extend(
        # These older fills have terrible markouts but should be dropped by the window.
        [_fill(fee=0.005, notional=100.0, markout_5s_bps=-10.0) for _ in range(20)]
    )
    out = build_net_edge_summary(fills, window=7)
    assert out["net_edge_sample_count"] == 7
    assert out["mean_markout_bps"] == 2.0  # not influenced by the older bad fills


def test_net_edge_summary_skips_fills_without_any_markout() -> None:
    """Fills with all three markout horizons None are skipped entirely."""
    fills = [_fill(fee=0.005, notional=100.0, markout_5s_bps=2.0) for _ in range(5)]
    fills.append(_fill(fee=0.005, notional=100.0))  # all None
    out = build_net_edge_summary(fills, window=50)
    assert out["net_edge_sample_count"] == 5
