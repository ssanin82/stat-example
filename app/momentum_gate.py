"""Inventory-aligned-with-momentum gate (analysis-day 2026-05-10).

Refuses to ADD to a position when the position is already aligned
with recent price drift. Targets the snapshot 260510064549 pattern:
the bot was long +64 SUI at peak (00:59) after monetising a short
through a down-and-rebound move, then the next leg down filled
fresh BUYs into the existing long inventory at adverse prices.

The intent: if the market has already moved your way and you're
sitting on the resulting inventory, *don't double down*. Take the
unrealized gain and let the regime clarify before adding more.

This is an EXTENSION of the existing ``inventory_exec_bias``
suppression in ``app.quoting``. That gate already pulls quotes on
the adding side when |position| is high; this one adds the
*momentum direction* as a second condition. Together they form:

* `inventory_exec_bias` alone: "I'm long, don't make my long bigger"
* `inventory_exec_bias` + this: "I'm long AND the market just rose,
  don't make my long bigger" (FOMO defense)

The gate is intentionally narrower than `inventory_exec_bias` —
firing only on the aligned-with-momentum case — so it kicks in
when the pure inventory gate is too lax (small position, big move)
without being trigger-happy on routine refilling.
"""

from __future__ import annotations

import math
from typing import Optional

from app.enums import QuoteEligibility


def evaluate_momentum_gate(
    *,
    position_qty: float,
    effective_abs_cap: float,
    drift_bps: Optional[float],
    drift_threshold_bps: float,
    inventory_pct_threshold: float,
    enabled: bool = True,
) -> tuple[Optional[QuoteEligibility], str]:
    """Return ``(override_eligibility, reason)``.

    * ``override_eligibility=None`` when the gate is inactive (most
      common case — drift below threshold, inventory below threshold,
      or feature disabled). Caller should keep its existing
      eligibility.
    * ``QuoteEligibility.QUOTE_SELL_ONLY`` when the gate forbids
      adding to a long (sign(qty) > 0 and drift > +threshold).
    * ``QuoteEligibility.QUOTE_BUY_ONLY`` when the gate forbids
      adding to a short (sign(qty) < 0 and drift < -threshold).

    The caller should ``intersect`` (more_restrictive) the override
    with the existing eligibility — the gate is a *cap*, never a
    *promotion* (it never widens a HOLD_ALL into BUY_ONLY).

    ``effective_abs_cap`` is ``min(MAX_ABS_POSITION,
    MAX_POSITION_NOTIONAL_USD / mid)`` — same denominator as the
    postmortem inventory-regime axis, so behaviour and reporting
    align.
    """
    if not enabled:
        return None, ""
    if drift_bps is None or not math.isfinite(drift_bps):
        return None, ""
    if effective_abs_cap <= 0:
        return None, ""

    util = abs(position_qty) / effective_abs_cap
    if util < inventory_pct_threshold:
        return None, ""

    if abs(drift_bps) < drift_threshold_bps:
        return None, ""

    # Aligned: long while drift is positive, OR short while drift is
    # negative. Refuse the add-side.
    if position_qty > 0 and drift_bps > 0:
        # Long with up-drift: gate the BUY side (BUY adds to long).
        return (
            QuoteEligibility.QUOTE_SELL_ONLY,
            f"momentum_gate:long_uptrend|util={util:.2f}|drift={drift_bps:+.2f}bps",
        )
    if position_qty < 0 and drift_bps < 0:
        # Short with down-drift: gate the SELL side (SELL adds to short).
        return (
            QuoteEligibility.QUOTE_BUY_ONLY,
            f"momentum_gate:short_downtrend|util={util:.2f}|drift={drift_bps:+.2f}bps",
        )
    # Anti-aligned (long+downtrend, short+uptrend): inventory is on
    # the wrong side of momentum already. The existing
    # ``inventory_exec_bias`` and adverse-side-pause logic handles
    # this case; the momentum gate stays out.
    return None, ""


# ------------------------------------------------------------------
# Widening contribution (gate-to-widening Phase 1, v1.4.8+)
# ------------------------------------------------------------------

def widening_bps(
    *,
    position_qty: float,
    effective_abs_cap: float,
    drift_bps: Optional[float],
    drift_threshold_bps: float,
    inventory_pct_threshold: float,
    max_half_spread_bps: float,
    widen_bps: float = -1.0,
    enabled: bool = True,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` widening contribution.

    Asymmetric — widens the SIDE the bot would otherwise use to add
    to an aligned-momentum position:
      * Long with up-drift: widen BID side (suppresses BUY).
      * Short with down-drift: widen ASK side (suppresses SELL).
      * Otherwise zero on both.

    ``widen_bps`` sentinel ``-1.0`` falls back to
    ``max_half_spread_bps`` (gate-equivalent). Operator iterates
    DOWN per ``plans/gate-to-widening.md`` Phase 2.
    """
    override, _ = evaluate_momentum_gate(
        position_qty=position_qty,
        effective_abs_cap=effective_abs_cap,
        drift_bps=drift_bps,
        drift_threshold_bps=drift_threshold_bps,
        inventory_pct_threshold=inventory_pct_threshold,
        enabled=enabled,
    )
    if override is None:
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    if override == QuoteEligibility.QUOTE_SELL_ONLY:
        # BUY suppressed → widen bid.
        return (effective, 0.0)
    if override == QuoteEligibility.QUOTE_BUY_ONLY:
        # SELL suppressed → widen ask.
        return (0.0, effective)
    return (0.0, 0.0)
