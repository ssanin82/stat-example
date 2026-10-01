from __future__ import annotations


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def bps_from_prices(mid: float, price: float) -> float:
    if mid <= 0:
        return 0.0
    return (price - mid) / mid * 10_000.0
