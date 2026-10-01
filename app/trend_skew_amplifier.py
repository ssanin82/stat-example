"""Trend-aware inventory-skew amplifier (analysis-day 2026-05-15).

Companion to ``momentum_gate``. Same trigger geometry; different
intervention.

* ``momentum_gate`` is preventive: when ``sign(position) ==
  sign(drift)`` and the position is large, it blocks the adding side
  ("don't make my long bigger when the market is rising while I'm
  already long").
* ``trend_skew_amplifier`` is corrective: under the same trigger, it
  multiplies the inventory_skew coefficient so the REDUCING side is
  more aggressive (pulls the ask closer to touch when long, pulls the
  bid closer to touch when short). The intent: get the wrong-sided
  inventory off the book FASTER during the kind of one-way market
  that produces it in the first place.

Why both:
* On their own, ``momentum_gate`` stops the bleed from growing but
  leaves the existing position to ride out the trend.
* On its own, an amplifier without the gate could pull the reducing
  side aggressively while the adding side still adds — net inventory
  doesn't move much.
* Together: adding side blocked, reducing side amplified, inventory
  walks down to zero faster than under either alone.

The amplifier returns a multiplier (>= 1.0). When inactive it returns
1.0 so the caller's existing skew computation is unchanged. Pure
function — no state; safe to call from any quote cycle.

Driven by the 260515-095056 snapshot: 86.6% long during a -2.45%
move, 30s MAE -7 to -8 bp. The 5s markout was fine (-0.43 bp) but the
inventory bled outside the markout horizon. The bot couldn't reduce
because its ask was 2 ticks behind a touch that kept walking away;
amplifying the skew pulls the ask closer (or inside the touch) when
the alignment is hot.
"""

from __future__ import annotations

import math
from typing import Optional


def compute_trend_skew_multiplier(
    *,
    position_qty: float,
    effective_abs_cap: float,
    drift_bps: Optional[float],
    drift_threshold_bps: float,
    inventory_pct_threshold: float,
    amplification_factor: float,
    enabled: bool = True,
) -> tuple[float, str]:
    """Return ``(multiplier, reason)``.

    * ``multiplier=1.0`` when the gate is inactive (most common case
      — drift below threshold, inventory below threshold, feature
      disabled, or trend not aligned with inventory). Caller's
      ``inventory_skew_coeff_bps`` is unchanged.
    * ``multiplier=amplification_factor`` when the alignment trigger
      fires. The caller should multiply its skew coefficient by this
      value. ``amplification_factor`` ≤ 1.0 is treated as disabled
      (the amplifier is strictly a *boost*, never a reduction).

    ``effective_abs_cap`` should mirror the same denominator as the
    momentum gate (``min(MAX_ABS_POSITION, MAX_POSITION_NOTIONAL_USD /
    mid)``) so the two features arm together under identical
    conditions. The reason string is parallel to momentum_gate's so
    they can be co-displayed in the dashboard's gate-activity strip.
    """
    if not enabled:
        return 1.0, ""
    if amplification_factor <= 1.0:
        return 1.0, ""
    if drift_bps is None or not math.isfinite(drift_bps):
        return 1.0, ""
    if effective_abs_cap <= 0:
        return 1.0, ""

    util = abs(position_qty) / effective_abs_cap
    if util < inventory_pct_threshold:
        return 1.0, ""

    if abs(drift_bps) < drift_threshold_bps:
        return 1.0, ""

    if position_qty > 0 and drift_bps > 0:
        return (
            amplification_factor,
            f"trend_skew_amp:long_uptrend|util={util:.2f}|drift={drift_bps:+.2f}bps|x{amplification_factor:.2f}",
        )
    if position_qty < 0 and drift_bps < 0:
        return (
            amplification_factor,
            f"trend_skew_amp:short_downtrend|util={util:.2f}|drift={drift_bps:+.2f}bps|x{amplification_factor:.2f}",
        )
    # Anti-aligned (long+downtrend, short+uptrend): inventory is
    # against momentum; this is a fast-recovery scenario, not the
    # case the amplifier is built for. Leave skew unchanged.
    return 1.0, ""
