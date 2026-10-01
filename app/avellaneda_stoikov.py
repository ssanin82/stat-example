"""Phase 8A (v1.5.184) — Avellaneda-Stoikov-inspired adaptive half-spread.

The original Avellaneda-Stoikov 2008 closed-form optimal half-spread
for an inventory-averse market maker:

    δ*(t) = γ · σ² · (T − t) + (2/γ) · ln(1 + γ/k)

where:
    γ      — risk-aversion coefficient
    σ²     — variance of mid-price returns
    T − t  — time remaining in the trading session
    k      — order-arrival intensity at touch (fills per unit time)

The literal formula has dimensional issues for crypto perp MM
(mixes price units, time units, and bps in awkward ways). This
module adapts it to a bot-friendly form that:

* preserves the qualitative structure (vol²-scaled inventory-risk
  term + log-shaped revenue-per-trade term);
* uses bps directly so the result composes with the bot's
  existing spread arithmetic (no unit conversions at call sites);
* exposes operator-tunable coefficients (γ_inv, γ_edge,
  edge_alpha) and safety clamps (min/max half-spread, base floor,
  k floor) so the function is calibratable per profile;
* is pure — no state, no side effects, no IO — so callers can
  cache the result with TTL or recompute per-tick as needed.

See ``plans/20260520-defense-action-plan.md`` § Phase 8 for the
broader context: 8A is the structural fix for the bot's currently-
constant ``BASE_HALF_SPREAD_BPS`` (= 4.0 in the TON prod profile),
which is structurally too tight in vol regimes where median 5-second
drift exceeds the configured spread.

**Default-off** — the wiring step (8A.3, separate PR) will gate the
new formula behind ``AVELLANEDA_STOIKOV_ENABLED`` so a config-only
deploy can opt in and rollback is a one-knob revert.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


def compute_as_half_spread_bps(
    *,
    vol_bps: float,
    k_intensity_per_min: float,
    gamma_inv: float = 0.05,
    gamma_edge: float = 1.0,
    edge_alpha: float = 1.0,
    k_floor_per_min: float = 0.1,
    base_floor_bps: float = 1.0,
    min_half_spread_bps: float = 1.5,
    max_half_spread_bps: float = 30.0,
) -> float:
    """Compute optimal half-spread (bps) per a simplified
    Avellaneda-Stoikov framework adapted for crypto MM.

    Returns a half-spread in bps, clamped to
    ``[min_half_spread_bps, max_half_spread_bps]``.

    The formula is:

        half_bps  =  base_floor_bps
                  +  γ_inv · vol_bps²
                  +  γ_edge · ln(1 + edge_alpha / max(k, k_floor))

    Three additive terms:

    1. ``base_floor_bps`` — the minimum half-spread the bot will
       ever ask for, regardless of vol or fill rate. Typically set
       to the rebate breakeven (∼ 1 bp on OKX with a maker rebate).

    2. ``γ_inv · vol_bps²`` — inventory-risk term. Quadratic in
       vol because AS's original closed form has γσ²; we use bps²
       directly so the term scales naturally with the
       ``vol_bps`` BotState already publishes (32-sample stdev
       of mid returns). The ``T − t`` horizon factor of the
       original AS is folded into ``γ_inv`` as a single dial —
       at the bot's continuous-trading cadence the explicit time
       term is more confusing than illuminating.

       Calibration tips:
       * γ_inv = 0.05 (default) gives vol=5 → 1.25 bp, vol=10 →
         5 bp, vol=20 → 20 bp — covers the typical TON regime.
       * Operators wanting a tighter spread in calm regimes can
         drop γ_inv to 0.02; for storm-aversion, push to 0.10.

    3. ``γ_edge · ln(1 + edge_alpha / max(k, k_floor))`` — revenue-
       per-trade term. Log-shaped per AS so it saturates rather
       than blowing up at low k. Narrows the spread as fill rate
       grows (each fill is cheap → can run thinner). Widens when
       fills are slow (need to capture more per trade to make it
       worthwhile).

       Calibration tips:
       * γ_edge = 1.0, edge_alpha = 1.0 (defaults) → at k = 1
         fill/min the term is ln(2) ≈ 0.69 bp; at k = 0.1/min
         (the floor) it's ln(11) ≈ 2.4 bp.
       * Drop γ_edge to 0.5 for less spread-tightening in
         high-fill regimes.

    Parameters
    ----------
    vol_bps : float
        Current per-tick volatility in bps (the bot's
        ``BotState.short_vol_bps``). Treated as σ in the
        inventory-risk term. Non-finite or negative → 0.
    k_intensity_per_min : float
        Recent fill rate at touch, in fills per minute. Estimated
        upstream from the bot's fill history (see Phase 8A.2 — the
        ``estimate_k_intensity`` helper, separate PR). Caller is
        responsible for the sampling window; this function is pure.
    gamma_inv : float
        Inventory-risk coefficient. Operator dial. Default 0.05.
    gamma_edge : float
        Revenue-per-trade coefficient. Operator dial. Default 1.0.
    edge_alpha : float
        Shape coefficient inside the log. Higher = more sensitivity
        to k. Default 1.0.
    k_floor_per_min : float
        Minimum k to use in the denominator (avoids the log blowing
        up at k → 0). Default 0.1 fills/min.
    base_floor_bps : float
        Minimum half-spread to always include. Default 1.0 bp.
    min_half_spread_bps : float
        Hard clamp floor. Default 1.5 bp. The bot's existing
        ``MIN_HALF_SPREAD_BPS`` is the source of truth; this
        argument lets callers override per profile.
    max_half_spread_bps : float
        Hard clamp ceiling. Default 30.0 bp.

    Returns
    -------
    float
        Half-spread in bps, clamped to ``[min, max]``.

    Examples
    --------
    Typical TON regime (vol_bps ≈ 5, k ≈ 1 fill/min) with defaults:

    >>> round(compute_as_half_spread_bps(vol_bps=5.0, k_intensity_per_min=1.0), 3)
    2.943

    Calm regime (vol_bps ≈ 2, k ≈ 5 fills/min):

    >>> round(compute_as_half_spread_bps(vol_bps=2.0, k_intensity_per_min=5.0), 3)
    1.5

    High-vol regime (vol_bps ≈ 15, k ≈ 0.5 fills/min):

    >>> round(compute_as_half_spread_bps(vol_bps=15.0, k_intensity_per_min=0.5), 3)
    13.348
    """
    # Defensive normalisation. Crypto vol can briefly read 0 (deque
    # not yet warm) or a huge number (single-tick outlier on a thin
    # book); guard both.
    if not math.isfinite(vol_bps) or vol_bps < 0.0:
        sigma = 0.0
    else:
        sigma = float(vol_bps)

    if not math.isfinite(k_intensity_per_min):
        k_per_min = float(k_floor_per_min)
    else:
        k_per_min = max(float(k_floor_per_min), float(k_intensity_per_min))

    # Term 1 — inventory-risk. Quadratic in vol. Coefficient
    # γ_inv absorbs the original AS horizon factor (T-t).
    inventory_risk_bps = float(gamma_inv) * sigma * sigma

    # Term 2 — revenue-per-trade. Log-shaped so the term stays
    # bounded as k → 0 (via the floor) and decays as k grows.
    edge_per_trade_bps = float(gamma_edge) * math.log(
        1.0 + float(edge_alpha) / k_per_min
    )

    # Sum + base floor.
    half_bps = float(base_floor_bps) + inventory_risk_bps + edge_per_trade_bps

    # Safety clamps.
    if not math.isfinite(half_bps):
        half_bps = float(min_half_spread_bps)
    return max(
        float(min_half_spread_bps),
        min(float(max_half_spread_bps), half_bps),
    )


def estimate_k_intensity_per_min(
    fills: Iterable[Any],
    *,
    now: datetime,
    window_seconds: float = 3600.0,
) -> float:
    """Count fills inside [now − window_seconds, now] → fills/minute.

    Phase 8A.2 helper. Reads ``ts_fill`` from each entry (the
    ``Fill`` dataclass field) and counts how many fall inside the
    rolling window. Returns 0.0 when no usable timestamps are
    present (cold start, empty deque, or all entries outside the
    window).

    Pure / side-effect-free — caller (bot.py per-tick slow path)
    handles cache + TTL invalidation via
    ``state.as_k_intensity_last_refresh_mono`` so the O(N) scan
    over ``state.recent_fills`` (cap 1000) doesn't run every tick.

    Parameters
    ----------
    fills : iterable
        Any iterable yielding objects with a ``ts_fill`` attribute
        (``datetime``). Typically ``state.recent_fills`` (deque[Fill]).
    now : datetime
        Reference moment for the window. Caller passes
        ``_clock.now_utc()`` or equivalent.
    window_seconds : float
        Window width. Default 3600 s (1 hour) — long enough to
        smooth tick-to-tick variation, short enough that a
        regime shift in fill rate registers within a few minutes.

    Returns
    -------
    float
        Fills per minute over the window. Always ≥ 0.0.
    """
    if window_seconds <= 0.0:
        return 0.0
    # tolerate naive ``now`` by aligning to UTC if a tz is
    # missing on either side; we only compare deltas so the
    # absolute zone doesn't matter as long as both are consistent.
    cutoff = now - timedelta(seconds=window_seconds)
    n = 0
    for f in fills:
        ts = getattr(f, "ts_fill", None)
        if ts is None:
            continue
        # Both ts and cutoff should be timezone-aware datetimes
        # in practice (Fill.ts_fill is utc_now()-stamped). Guard
        # against accidental naive datetimes mixed with aware.
        if (ts.tzinfo is None) != (cutoff.tzinfo is None):
            # Normalise — treat naive as UTC.
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if cutoff.tzinfo is None:
                cutoff = cutoff.replace(tzinfo=timezone.utc)
        if ts >= cutoff:
            n += 1
    return float(n) / (float(window_seconds) / 60.0)


__all__ = [
    "compute_as_half_spread_bps",
    "estimate_k_intensity_per_min",
]
