"""Pure-function P&L attribution over a set of fills.

Decomposes realized P&L over a window into three components:

- **fee_income_usd**: sum of ``−fee`` (negative fees are rebates; market-maker
  exchanges pay you for adding liquidity). Positive means we earned rebate.
- **markout_dollar_impact_usd**: sum of ``markout_bps × notional / 10_000``
  over fills with resolved markouts at the chosen horizon. Negative means
  the mid moved against us after our fills (adverse selection).
- **residual_usd**: whatever doesn't fit fees+markout. Captures things the
  markout horizon doesn't see — longer-dated inventory drift, mark-to-market
  on held inventory, edge cases.

The key identity:

    realized_pnl_total = fee_income_usd
                      + markout_dollar_impact_usd (at horizon)
                      + residual_usd

The residual exists because markout at a finite horizon (5 s) doesn't capture
PnL from inventory held longer than that. For an MM running near-flat, residual
is small; for a bot that accumulates significant inventory, residual can be
the dominant term.

This module is deliberately stateless and reads only fill-like dicts. The API
wrapper in ``app/api.py`` does the DB query and passes rows in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

# v1.4.98 — extended to seven horizons. The 1s/3s/5s endpoints
# capture HFT-immediate-pickoff range (control surfaces still anchor
# on 5s); 15s/30s/60s/120s align with the bot's gate time-constants
# (post-quote-life, soft-flatten window, post-swing, long-tail).
_SUPPORTED_HORIZONS = (1, 3, 5, 15, 30, 60, 120)


@dataclass(frozen=True, slots=True)
class _PerSideStats:
    fills: int
    notional_usd: float
    fee_usd: float
    rebate_usd: float
    mean_markout_bps: Optional[float]
    median_markout_bps: Optional[float]
    stdev_markout_bps: Optional[float]
    adverse_count: int
    favorable_count: int
    markout_dollar_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fills": self.fills,
            "notional_usd": round(self.notional_usd, 4),
            "fee_usd": round(self.fee_usd, 6),
            "rebate_usd": round(self.rebate_usd, 6),
            "mean_markout_bps": None if self.mean_markout_bps is None else round(self.mean_markout_bps, 4),
            "median_markout_bps": None if self.median_markout_bps is None else round(self.median_markout_bps, 4),
            "stdev_markout_bps": None if self.stdev_markout_bps is None else round(self.stdev_markout_bps, 4),
            "adverse_count": self.adverse_count,
            "favorable_count": self.favorable_count,
            "markout_dollar_usd": round(self.markout_dollar_usd, 6),
        }


def _markout_field(horizon_s: int) -> str:
    if horizon_s not in _SUPPORTED_HORIZONS:
        raise ValueError(f"unsupported horizon_s={horizon_s}; use one of {_SUPPORTED_HORIZONS}")
    return f"markout_{horizon_s}s_bps"


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN check
        return None
    return v


def _mean(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / len(xs)


def _median(xs: list[float]) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _stdev(xs: list[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def _side_stats(fills: list[dict[str, Any]], markout_field: str) -> _PerSideStats:
    n = len(fills)
    notional = sum((_safe_float(f.get("notional")) or 0.0) for f in fills)
    fee_sum = sum((_safe_float(f.get("fee")) or 0.0) for f in fills)
    rebate_sum = -fee_sum  # rebate is the negative of fee
    markouts = [m for f in fills if (m := _safe_float(f.get(markout_field))) is not None]
    adverse = sum(1 for m in markouts if m < 0)
    favorable = sum(1 for m in markouts if m > 0)
    dollar = 0.0
    for f in fills:
        m = _safe_float(f.get(markout_field))
        notn = _safe_float(f.get("notional"))
        if m is None or notn is None:
            continue
        dollar += m * notn / 10_000.0
    return _PerSideStats(
        fills=n,
        notional_usd=notional,
        fee_usd=fee_sum,
        rebate_usd=rebate_sum,
        mean_markout_bps=_mean(markouts),
        median_markout_bps=_median(markouts),
        stdev_markout_bps=_stdev(markouts),
        adverse_count=adverse,
        favorable_count=favorable,
        markout_dollar_usd=dollar,
    )


def compute_attribution(
    *,
    fills: Iterable[dict[str, Any]],
    horizon_s: int = 5,
    window_since_iso: Optional[str] = None,
    window_until_iso: Optional[str] = None,
    realized_pnl_total_usd: Optional[float] = None,
) -> dict[str, Any]:
    """Decompose P&L into fee / markout / residual components.

    Args:
        fills: Iterable of fill-row dicts. Each row must have at least
            ``side``, ``price``, ``size``, ``notional``, ``fee``, and the
            chosen ``markout_{N}s_bps`` field (may be None).
        horizon_s: Which delayed-markout horizon to attribute against.
            Must be one of {1, 3, 5}. Default 5.
        window_since_iso / window_until_iso: Cosmetic — copied into the
            returned ``window`` dict for display. No filtering here; that's
            the caller's job.
        realized_pnl_total_usd: If provided, ``pnl_attribution.residual_usd``
            is computed as ``total − fee_income − markout_dollar``. Without
            it the residual is reported as ``null``.

    Returns:
        A JSON-serialisable dict with keys:
        ``window``, ``fills``, ``fees``, ``markout``, ``per_side``,
        ``pnl_attribution_usd``, ``horizon_s``.
    """
    markout_field = _markout_field(horizon_s)
    fills_list = list(fills)

    overall = _side_stats(fills_list, markout_field)
    buys = _side_stats([f for f in fills_list if f.get("side") == "BUY"], markout_field)
    sells = _side_stats([f for f in fills_list if f.get("side") == "SELL"], markout_field)

    liquidity_flags: dict[str, int] = {}
    for f in fills_list:
        flag = str(f.get("liquidity_flag") or "unknown")
        liquidity_flags[flag] = liquidity_flags.get(flag, 0) + 1

    maker_fills = liquidity_flags.get("resting", 0) + liquidity_flags.get("maker", 0)
    taker_fills = liquidity_flags.get("taking", 0) + liquidity_flags.get("taker", 0)
    other_fills = overall.fills - maker_fills - taker_fills

    rebate_bps_of_notional: Optional[float] = None
    if overall.notional_usd > 0:
        rebate_bps_of_notional = overall.rebate_usd / overall.notional_usd * 10_000.0

    win_rate: Optional[float] = None
    total_resolved = overall.adverse_count + overall.favorable_count
    if total_resolved > 0:
        win_rate = overall.favorable_count / total_resolved

    residual: Optional[float] = None
    if realized_pnl_total_usd is not None:
        residual = float(realized_pnl_total_usd) - overall.rebate_usd - overall.markout_dollar_usd

    explanation_note = (
        "realized_pnl_total = rebate_income + markout_dollar_impact + residual. "
        "residual captures PnL from inventory held past the markout horizon "
        f"({horizon_s}s) and mark-to-market drift outside the measurement window."
    )

    return {
        "horizon_s": horizon_s,
        "window": {
            "since": window_since_iso,
            "until": window_until_iso,
        },
        "fills": {
            "total": overall.fills,
            "buy_count": buys.fills,
            "sell_count": sells.fills,
            "maker_count": maker_fills,
            "taker_count": taker_fills,
            "other_count": other_fills,
            "liquidity_flag_breakdown": liquidity_flags,
            "total_notional_usd": round(overall.notional_usd, 4),
            "avg_notional_usd": (
                round(overall.notional_usd / overall.fills, 4) if overall.fills else 0.0
            ),
        },
        "fees": {
            "total_fee_usd": round(overall.fee_usd, 6),
            "rebate_earned_usd": round(overall.rebate_usd, 6),
            "rebate_bps_of_notional": (
                None if rebate_bps_of_notional is None else round(rebate_bps_of_notional, 4)
            ),
        },
        "markout": {
            "horizon_s": horizon_s,
            "sample_count": overall.adverse_count + overall.favorable_count,
            "mean_bps": None if overall.mean_markout_bps is None else round(overall.mean_markout_bps, 4),
            "median_bps": None if overall.median_markout_bps is None else round(overall.median_markout_bps, 4),
            "stdev_bps": None if overall.stdev_markout_bps is None else round(overall.stdev_markout_bps, 4),
            "adverse_count": overall.adverse_count,
            "favorable_count": overall.favorable_count,
            "win_rate": None if win_rate is None else round(win_rate, 4),
            "dollar_impact_usd": round(overall.markout_dollar_usd, 6),
        },
        "per_side": {
            "BUY": buys.to_dict(),
            "SELL": sells.to_dict(),
        },
        "pnl_attribution_usd": {
            "realized_pnl_total": (
                None if realized_pnl_total_usd is None else round(float(realized_pnl_total_usd), 6)
            ),
            "fee_income": round(overall.rebate_usd, 6),
            "markout_dollar_impact": round(overall.markout_dollar_usd, 6),
            "residual": None if residual is None else round(residual, 6),
            "explanation": explanation_note,
        },
    }
