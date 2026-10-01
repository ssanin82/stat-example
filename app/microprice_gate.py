"""Microprice / OB-imbalance passive-adverse-selection gate
(1.2.2 — analysis-day 2026-05-10).

When the order book has visibly more depth on one side than the
other, the *thin* side is the side aggressive flow is most likely
to clear next. A passive maker quoting on the thin side is making
a high-probability adverse-selection bet — the next aggressive
print eats the thin side, the price moves, and the maker is
holding inventory at the worst possible mark.

The bot already smooths an EWMA of OB imbalance into
``state.ob_imbalance_ewma``:

  imbalance = (bid_size - ask_size) / (bid_size + ask_size)

Range: [-1, +1]. Positive = bid is heavier (asks will be eaten
first — uptrend pressure). Negative = ask is heavier (bids will
be eaten — downtrend pressure).

This gate consumes that signal and suppresses the side facing
adverse-selection risk:

* ``imbalance > +threshold`` (bid-heavy book → ask side is thin
  → aggressive buyers will sweep asks): suppress ASK quoting,
  return ``QUOTE_BUY_ONLY``.
* ``imbalance < -threshold`` (ask-heavy book → bid side is thin):
  suppress BID quoting, return ``QUOTE_SELL_ONLY``.
* Otherwise: no gate.

This is orthogonal to the existing inventory_skew (which adjusts
quote PRICES) and microprice reservation (which moves the
midpoint reference). Those still run; this just decides whether
to quote at all on the imbalanced side.

Stateless — single-tick evaluation. Default threshold 0.5 is a
conservative starting point ("clear visible asymmetry"); in
flatter books the gate rarely fires, in stressed books it fires
appropriately.
"""

from __future__ import annotations

import math
from typing import Optional

from app.enums import QuoteEligibility


def evaluate_gate(
    *,
    ob_imbalance_ewma: Optional[float],
    threshold: float,
    enabled: bool = True,
) -> tuple[Optional[QuoteEligibility], str]:
    """Return ``(override_eligibility, reason)`` tuple.

    * ``None, ""`` when the gate is silent (most ticks).
    * ``QUOTE_BUY_ONLY, "..."`` when the ask side faces adverse
      selection (bid-heavy book).
    * ``QUOTE_SELL_ONLY, "..."`` when the bid side faces adverse
      selection (ask-heavy book).

    Caller should intersect the override with its existing
    eligibility via ``more_restrictive`` — the gate never widens.
    """
    if not enabled:
        return None, ""
    if ob_imbalance_ewma is None or not math.isfinite(ob_imbalance_ewma):
        return None, ""
    if threshold <= 0.0:
        return None, ""

    if ob_imbalance_ewma >= threshold:
        # Bid-heavy → ask side is thin → suppress asks (quote bid only).
        return (
            QuoteEligibility.QUOTE_BUY_ONLY,
            f"microprice_gate:ask_thin|ob_imb={ob_imbalance_ewma:+.3f}",
        )
    if ob_imbalance_ewma <= -threshold:
        # Ask-heavy → bid side is thin → suppress bids (quote ask only).
        return (
            QuoteEligibility.QUOTE_SELL_ONLY,
            f"microprice_gate:bid_thin|ob_imb={ob_imbalance_ewma:+.3f}",
        )
    return None, ""


# ------------------------------------------------------------------
# Widening contribution (gate-to-widening Phase 1, v1.4.8+)
# ------------------------------------------------------------------

def widening_bps(
    *,
    ob_imbalance_ewma: Optional[float],
    threshold: float,
    max_half_spread_bps: float,
    widen_bps: float = -1.0,
    enabled: bool = True,
    # v1.4.42 BUG-026 inventory-aware suppression:
    position_qty: float = 0.0,
    effective_abs_cap: float = 0.0,
    reducing_side_widen_suppress_pct: float = 0.0,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` widening contribution.

    Asymmetric — widens only the side facing adverse-selection risk:
      * ``imbalance >= +threshold`` (ask thin, sweepers will eat asks):
        widen ASK side; bid stays at zero contribution.
      * ``imbalance <= -threshold`` (bid thin):
        widen BID side; ask stays at zero contribution.
      * otherwise zero on both.

    ``widen_bps`` sentinel ``-1.0`` falls back to
    ``max_half_spread_bps`` (gate-equivalent magnitude). Operator
    iterates this DOWN per ``plans/gate-to-widening.md`` Phase 2.

    BUG-026 inventory-direction suppression (v1.4.47 structural):
    the microprice gate's "adverse selection" protection only makes
    sense for the ADDING side of the bot's current inventory.

    Reasoning. The gate fires on order-book imbalance:
      * bid-heavy book (``ob_imb >= +threshold``) → ask is thin →
        aggressive buyers will eat asks next → price likely to rise.
      * ask-heavy book (``ob_imb <= -threshold``) → bid is thin →
        aggressive sellers will eat bids next → price likely to drop.

    The "widen the thin side" rule protects against being adversely
    picked off — getting filled on the side that's about to lose.
    That logic assumes NEUTRAL inventory: a fill on either side is
    equally adversarial.

    When inventory is non-zero the calculus inverts on one side:

      * Bot is SHORT, book signals drop (bid thin): a fill on the
        bid REDUCES the short at current price. Yes, price might
        drop further and the bot would have done better waiting —
        but the alternative (staying short indefinitely while the
        gate widens the bid out of reach) is far worse. The bid is
        the REDUCING side. Don't widen it.
      * Bot is LONG, book signals rise (ask thin): symmetric. Ask
        is the reducing side. Don't widen it.

    Rule (no threshold, no magic number):

      * ``position_qty > 0`` (long) AND override = ``QUOTE_BUY_ONLY``
        (gate would widen ask): ask is reducing side → no widening.
      * ``position_qty < 0`` (short) AND override = ``QUOTE_SELL_ONLY``
        (gate would widen bid): bid is reducing side → no widening.
      * ``position_qty == 0``: neutral inventory → gate's normal
        adversarial-protection logic applies on whichever side is
        thin.
      * Any other combination (gate would widen the ADDING side):
        normal widening; adverse selection on the adding side is
        legitimate to protect against.

    The ``reducing_side_widen_suppress_pct`` parameter is kept for
    signature back-compat but its semantics changed:

      * ``< 0`` (DEFAULT in caller via config sentinel): apply the
        structural direction-based rule above. No threshold.
      * ``== 0``: disable the protection entirely (pre-v1.4.42
        behaviour — gate widens reducing side too).
      * ``> 0``: LEGACY threshold mode. Suppression fires when
        ``|position| / abs_cap >= reducing_side_widen_suppress_pct``.
        Preserved so operators who pinned a threshold pre-v1.4.47
        keep working; not recommended for new deployments.

    History: v1.4.42 introduced this protection with a fixed 0.30
    threshold default (operator override available). v1.4.46
    "fixed" it by coupling the default to
    ``INVENTORY_EXECUTION_BIAS_MIN_UTIL_PCT`` (0.12 by default).
    Operator pushback ("no magic numbers") motivated the v1.4.47
    structural rewrite: drop the threshold entirely and gate by
    position direction only. Position direction is the actual
    signal — inventory size determines how BADLY the gate's widening
    hurts but the SIGN of inventory determines whether the gate's
    widening helps or hurts at all.
    """
    override, _ = evaluate_gate(
        ob_imbalance_ewma=ob_imbalance_ewma,
        threshold=threshold,
        enabled=enabled,
    )
    if override is None:
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    mode = float(reducing_side_widen_suppress_pct)
    if override == QuoteEligibility.QUOTE_BUY_ONLY:
        # ASK is thin → gate wants to widen ASK side.
        # If bot is LONG, ask is the inventory-reducing side.
        if mode < 0.0:
            # Structural mode: direction-only, no threshold.
            if float(position_qty) > 0.0:
                return (0.0, 0.0)
        elif mode > 0.0:
            # Legacy threshold mode (v1.4.42 - v1.4.46).
            if float(effective_abs_cap) > 1e-12:
                util_signed = float(position_qty) / float(effective_abs_cap)
                if util_signed >= mode:
                    return (0.0, 0.0)
        return (0.0, effective)
    if override == QuoteEligibility.QUOTE_SELL_ONLY:
        # BID is thin → gate wants to widen BID side.
        # If bot is SHORT, bid is the inventory-reducing side.
        if mode < 0.0:
            if float(position_qty) < 0.0:
                return (0.0, 0.0)
        elif mode > 0.0:
            if float(effective_abs_cap) > 1e-12:
                util_signed = float(position_qty) / float(effective_abs_cap)
                if util_signed <= -mode:
                    return (0.0, 0.0)
        return (effective, 0.0)
    return (0.0, 0.0)


# ------------------------------------------------------------------
# Recorder-fed variant: microprice-deviation z-score (M8 Candidate B)
# ------------------------------------------------------------------

def microprice_z_widening_bps(
    *,
    microprice_dev_z: Optional[float],
    z_threshold: float,
    widen_bps: float,
    max_half_spread_bps: float,
    enabled: bool = True,
    position_qty: float = 0.0,
    effective_abs_cap: float = 0.0,
    reducing_side_widen_suppress_pct: float = 0.0,
) -> tuple[float, float]:
    """``(bid_bps, ask_bps)`` widening from the recorder's microprice
    deviation z-score instead of the in-process OB-imbalance EWMA.

    This is the M8 Candidate-B consumer of the tape runtime feed's
    ``microprice_dev_z_24h`` field (a *true* z-score: ``(last - mean) / sd``
    over a 24 h EWMA, clamped to ±10, published only after the recorder's
    deviation aggregator has ≥2000 samples; ``None`` until then). The sign
    convention is IDENTICAL to ``ob_imbalance_ewma``:

      * ``z >= +z_threshold`` → microprice above mid → bid-heavy book →
        ask side is thin → widen ASK.
      * ``z <= -z_threshold`` → microprice below mid → ask-heavy book →
        bid side is thin → widen BID.

    so we delegate to :func:`widening_bps` with the z-score in the
    imbalance slot and ``z_threshold`` as the firing threshold. The
    BUG-026 inventory-direction suppression carries over unchanged (same
    ``position_qty`` / ``effective_abs_cap`` / ``reducing_side_widen_
    suppress_pct`` semantics) so this contributor behaves like its
    OB-imbalance sibling w.r.t. the bot's inventory.

    Unlike the sibling, ``widen_bps`` here is a real per-side bps amount
    (NOT the ``-1.0`` → full-cap sentinel): the z-gate fires on a much
    larger fraction of ticks than the rare binary gates, so a full-cap
    default would distort quoting. The caller passes
    ``MICROPRICE_Z_WIDEN_BPS`` (default 4.0). ``None`` ``microprice_dev_z``
    (feed off / field dark / stale — resolved by the caller) yields a
    zero contribution, so the feature is byte-identical to today when the
    recorder feed is disabled.
    """
    if not enabled:
        return (0.0, 0.0)
    if microprice_dev_z is None or not math.isfinite(microprice_dev_z):
        return (0.0, 0.0)
    return widening_bps(
        ob_imbalance_ewma=float(microprice_dev_z),
        threshold=float(z_threshold),
        max_half_spread_bps=float(max_half_spread_bps),
        widen_bps=float(widen_bps),
        enabled=True,
        position_qty=float(position_qty),
        effective_abs_cap=float(effective_abs_cap),
        reducing_side_widen_suppress_pct=float(reducing_side_widen_suppress_pct),
    )
