"""Pure-function tests for P&L attribution.

The module ``app.pnl_attribution`` is deliberately stateless — we feed it
fill-row dicts and check the arithmetic.
"""

from __future__ import annotations

import pytest

from app.pnl_attribution import compute_attribution


def _fill(
    *,
    side: str,
    notional: float,
    fee: float,
    markout_5s: float | None = None,
    markout_3s: float | None = None,
    markout_1s: float | None = None,
    liquidity_flag: str = "resting",
) -> dict:
    return {
        "fill_id": f"f_{side}_{notional}",
        "side": side,
        "price": notional,  # dummy; notional-based math doesn't use price
        "size": 1.0,
        "notional": notional,
        "fee": fee,
        "liquidity_flag": liquidity_flag,
        "markout_1s_bps": markout_1s,
        "markout_3s_bps": markout_3s,
        "markout_5s_bps": markout_5s,
    }


def test_empty_fills_produces_zero_attribution() -> None:
    out = compute_attribution(fills=[], horizon_s=5)
    assert out["fills"]["total"] == 0
    assert out["fees"]["total_fee_usd"] == 0.0
    assert out["markout"]["dollar_impact_usd"] == 0.0


def test_single_adverse_buy_fill_math() -> None:
    """BUY 100 USD notional, markout_5s = -2 bps, fee = -0.001 (rebate).

    Expected:
      - fee_usd = -0.001 (rebate)
      - rebate_earned = +0.001
      - markout dollar = -2 * 100 / 10000 = -0.02
    """
    fills = [_fill(side="BUY", notional=100.0, fee=-0.001, markout_5s=-2.0)]
    out = compute_attribution(fills=fills, horizon_s=5)
    assert out["fills"]["total"] == 1
    assert out["fills"]["buy_count"] == 1
    assert out["fees"]["total_fee_usd"] == pytest.approx(-0.001)
    assert out["fees"]["rebate_earned_usd"] == pytest.approx(+0.001)
    assert out["markout"]["dollar_impact_usd"] == pytest.approx(-0.02)
    assert out["markout"]["adverse_count"] == 1
    assert out["markout"]["favorable_count"] == 0
    assert out["markout"]["win_rate"] == pytest.approx(0.0)
    assert out["markout"]["mean_bps"] == pytest.approx(-2.0)


def test_per_side_split() -> None:
    fills = [
        _fill(side="BUY", notional=100.0, fee=-0.001, markout_5s=-3.0),
        _fill(side="BUY", notional=100.0, fee=-0.001, markout_5s=-1.0),
        _fill(side="SELL", notional=100.0, fee=-0.001, markout_5s=+1.0),
    ]
    out = compute_attribution(fills=fills, horizon_s=5)
    assert out["per_side"]["BUY"]["fills"] == 2
    assert out["per_side"]["SELL"]["fills"] == 1
    assert out["per_side"]["BUY"]["mean_markout_bps"] == pytest.approx(-2.0)
    assert out["per_side"]["SELL"]["mean_markout_bps"] == pytest.approx(+1.0)
    assert out["per_side"]["BUY"]["markout_dollar_usd"] == pytest.approx(-0.04)
    assert out["per_side"]["SELL"]["markout_dollar_usd"] == pytest.approx(+0.01)


def test_horizon_selection_uses_correct_markout_field() -> None:
    """Different horizons → different markout values."""
    fills = [
        _fill(
            side="BUY",
            notional=100.0,
            fee=0.0,
            markout_1s=-1.0,
            markout_3s=-2.0,
            markout_5s=-3.0,
        )
    ]
    out_1s = compute_attribution(fills=fills, horizon_s=1)
    out_3s = compute_attribution(fills=fills, horizon_s=3)
    out_5s = compute_attribution(fills=fills, horizon_s=5)
    assert out_1s["markout"]["mean_bps"] == pytest.approx(-1.0)
    assert out_3s["markout"]["mean_bps"] == pytest.approx(-2.0)
    assert out_5s["markout"]["mean_bps"] == pytest.approx(-3.0)
    assert out_1s["markout"]["dollar_impact_usd"] == pytest.approx(-0.01)
    assert out_5s["markout"]["dollar_impact_usd"] == pytest.approx(-0.03)


def test_invalid_horizon_raises() -> None:
    with pytest.raises(ValueError):
        compute_attribution(fills=[], horizon_s=2)
    with pytest.raises(ValueError):
        compute_attribution(fills=[], horizon_s=0)


def test_residual_computed_when_total_pnl_provided() -> None:
    """residual = realized_total − rebate − markout_dollar."""
    fills = [_fill(side="BUY", notional=1000.0, fee=-0.01, markout_5s=-2.0)]
    out = compute_attribution(
        fills=fills,
        horizon_s=5,
        realized_pnl_total_usd=-0.25,
    )
    # rebate = +0.01, markout dollar = -2*1000/10000 = -0.2
    # residual = -0.25 - 0.01 - (-0.2) = -0.25 - 0.01 + 0.2 = -0.06
    assert out["pnl_attribution_usd"]["fee_income"] == pytest.approx(+0.01)
    assert out["pnl_attribution_usd"]["markout_dollar_impact"] == pytest.approx(-0.2)
    assert out["pnl_attribution_usd"]["residual"] == pytest.approx(-0.06)
    assert out["pnl_attribution_usd"]["realized_pnl_total"] == pytest.approx(-0.25)


def test_residual_is_null_when_total_pnl_missing() -> None:
    fills = [_fill(side="BUY", notional=100.0, fee=0.0, markout_5s=0.0)]
    out = compute_attribution(fills=fills, horizon_s=5)
    assert out["pnl_attribution_usd"]["residual"] is None
    assert out["pnl_attribution_usd"]["realized_pnl_total"] is None


def test_liquidity_flag_breakdown() -> None:
    """maker_count and taker_count reflect the liquidity_flag field."""
    fills = [
        _fill(side="BUY", notional=100.0, fee=-0.001, markout_5s=-1.0, liquidity_flag="resting"),
        _fill(side="BUY", notional=100.0, fee=-0.001, markout_5s=-1.0, liquidity_flag="resting"),
        _fill(side="SELL", notional=100.0, fee=+0.001, markout_5s=-1.0, liquidity_flag="taking"),
    ]
    out = compute_attribution(fills=fills, horizon_s=5)
    assert out["fills"]["maker_count"] == 2
    assert out["fills"]["taker_count"] == 1
    assert out["fills"]["liquidity_flag_breakdown"]["resting"] == 2
    assert out["fills"]["liquidity_flag_breakdown"]["taking"] == 1


def test_rebate_bps_of_notional_scales_correctly() -> None:
    """rebate_bps = rebate_usd / notional * 10000."""
    # 0.1 bps rebate on $10000 notional = $0.1
    fills = [_fill(side="BUY", notional=10000.0, fee=-0.1, markout_5s=0.0)]
    out = compute_attribution(fills=fills, horizon_s=5)
    assert out["fees"]["rebate_bps_of_notional"] == pytest.approx(0.1)


def test_missing_markout_is_excluded_from_stats_not_notional() -> None:
    """Fills without a resolved markout count for notional/fee but not markout stats."""
    fills = [
        _fill(side="BUY", notional=100.0, fee=-0.001, markout_5s=-1.0),
        _fill(side="BUY", notional=100.0, fee=-0.001, markout_5s=None),
    ]
    out = compute_attribution(fills=fills, horizon_s=5)
    assert out["fills"]["total"] == 2
    assert out["fills"]["total_notional_usd"] == pytest.approx(200.0)
    assert out["markout"]["sample_count"] == 1  # only one has resolved markout
    assert out["markout"]["dollar_impact_usd"] == pytest.approx(-0.01)


def test_regression_session_113453_shape() -> None:
    """Sanity: with the approximate shape of snap_20260418_113453
    (70 fills, balanced, ~$2471 notional, mean markout_5s ≈ -2.25 bps,
    tiny rebate), the attribution should produce markout_dollar ≈ -$0.56
    and rebate ≈ $0.002.
    """
    # 70 fills at avg ~$35 each, mean markout -2.25 bps, fee -3.4e-5 each.
    fills = []
    for i in range(70):
        side = "BUY" if i % 2 == 0 else "SELL"
        markout = -2.5 if side == "BUY" else -2.0
        fills.append(_fill(side=side, notional=35.0, fee=-3.4e-5, markout_5s=markout))
    out = compute_attribution(fills=fills, horizon_s=5, realized_pnl_total_usd=-0.589)
    # Expected: rebate = 70 * 3.4e-5 = ~0.00238; markout dollar ≈ -2.25 * 70 * 35 / 10000 ≈ -0.551
    assert out["fees"]["rebate_earned_usd"] == pytest.approx(70 * 3.4e-5, abs=1e-5)
    # Markout dollar should be within a few cents of the observed -$0.56.
    assert -0.60 < out["markout"]["dollar_impact_usd"] < -0.50
    # Residual = -0.589 - rebate - markout_dollar — small but signed.
    assert out["pnl_attribution_usd"]["residual"] is not None
