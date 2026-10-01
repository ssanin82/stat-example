"""Markout-tier size scaler (1.2.3 — analysis-day 2026-05-10).

The toxicity-driven size reduction (``TOXICITY_SIZE_REDUCTION_COEFF``)
is gated on the COMPOSITE toxicity score, which requires multiple
conditions (markout AND one-sided fill ratio AND vol spike) each
contributing partial credit. On real sessions the dashboard's
"heavy adverse" markout tier (median_5s ≤ −3 bp) frequently shows
red while the toxicity score sits at 0.04 — because the score
hasn't summed past its 0.45 trigger.

This scaler uses the **direct** rolling-median 5s markout signal
(same one the dashboard tier label reads) and produces a size
multiplier ∈ [floor, 1.0]. The bot takes ``min(toxicity_size_mult,
markout_size_mult)`` so either signal can shrink, neither can grow.

Tier defaults match the dashboard's pill colours:

* ≥ 0 (clean) → 1.0 (no shrink)
* −1 to 0 (mild adverse) → 0.85
* −3 to −1 (moderate adverse) → 0.5
* < −3 (heavy adverse) → 0.25 (down to floor)

Stateless; per-tick evaluation. Floor is clamped to the same
``size_mult_floor`` the toxicity scaler uses (typically
``MIN_QUOTE_NOTIONAL_USD / QUOTE_NOTIONAL_USD``) so the venue
min-notional gate doesn't suppress the order.
"""

from __future__ import annotations

import math
from typing import Optional


def markout_size_mult(
    median_5s_bps: Optional[float],
    *,
    mild_threshold_bps: float = 0.0,
    moderate_threshold_bps: float = -1.0,
    heavy_threshold_bps: float = -3.0,
    mild_mult: float = 0.85,
    moderate_mult: float = 0.5,
    heavy_mult: float = 0.25,
    floor: float = 0.2,
) -> float:
    """Return a size multiplier in [floor, 1.0] based on the
    rolling-median 5s markout. ``None`` / NaN / clean tier returns
    1.0 (no shrink).

    Threshold ordering must satisfy
    ``mild >= moderate >= heavy`` (more-negative is more-adverse).
    """
    if median_5s_bps is None:
        return 1.0
    try:
        v = float(median_5s_bps)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    if v >= mild_threshold_bps:
        return 1.0
    if v >= moderate_threshold_bps:
        return max(floor, mild_mult)
    if v >= heavy_threshold_bps:
        return max(floor, moderate_mult)
    return max(floor, heavy_mult)
