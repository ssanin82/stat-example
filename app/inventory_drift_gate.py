"""Inventory-drift defence gate — Phase 1A (v1.4.106).

Companion to ``slow_trend_gate`` (15-min sustained drift) and the
existing ``long_drift_eligibility`` (5-min acute moves). Where those
are *position-blind*, this gate is **position-aware** and reacts on a
**short** window (10–30 s):

* Fires only when bot inventory is non-trivial (``util ≥ threshold``).
* Fires only when short-window drift is **anti-aligned** with the
  bot's current position (i.e. price is moving in the direction that
  *hurts* the held inventory).
* Faster reaction than slow_trend — designed to detect mid-grind
  episodes within seconds of them turning against held inventory,
  not wait the 5-15 min the longer gates need.

Why a separate gate from slow_trend / momentum:

* ``slow_trend_gate`` is position-blind and slow (15 min). It catches
  the trend itself, regardless of where the bot is sitting. Good
  early-warning but its widening fires the same way whether the bot
  is +9 long or flat.
* ``momentum_gate`` is position-aware but uses a single drift input
  (the long-window). Designed for sustained moves, not the
  10-30 s reaction window this gate covers.
* ``long_drift_eligibility`` is position-blind, 5-min window, 50 bp
  threshold — only fires on acute moves.

This gate fills the **short-window × position-aware** corner. It is
the "simplest high-value rule" called out in the Codex Section 6
recommendation: when the bot is already loaded the wrong way and the
mid is moving the wrong way *right now*, stop adding.

Anti-aligned semantics:

* LONG inventory + DOWN drift  → suppress BUY  → ``QUOTE_SELL_ONLY``,
  widen BID side.
* SHORT inventory + UP   drift → suppress SELL → ``QUOTE_BUY_ONLY``,
  widen ASK side.
* Aligned (LONG + UP, or SHORT + DOWN) → no-op. The bot is on the
  right side of the move; the inventory-skew machinery already
  prefers reducing the position. No need to widen.
* Util below threshold → no-op. The whole point is to defend held
  inventory, not to react to drift in isolation.

Drift source: same multi-second mid-history deque the
``_long_drift_eligibility`` and ``slow_trend_gate`` use (``BotState
._mid_price_samples_long``). The gate evaluates 10 s and 30 s
windows from that deque on every tick. Either window crossing its
threshold (with anti-aligned sign) trips the gate.

Config (TON profile):

    INVENTORY_DRIFT_GATE_ENABLED=true
    INVENTORY_DRIFT_INVENTORY_PCT_THRESHOLD=0.60
    INVENTORY_DRIFT_THRESHOLD_BPS_10S=15.0
    INVENTORY_DRIFT_THRESHOLD_BPS_30S=30.0
    INVENTORY_DRIFT_WIDEN_BPS=15.0

The widening default (15 bp) is intentionally **wider than
slow_trend's 10 bp**: the inventory-drift gate's trigger is more
specific (position-aware AND short-window AND anti-aligned), so
when it fires the conviction that "don't add here" is higher and a
larger backoff is warranted.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from app.enums import QuoteEligibility
# v1.5.181 (Phase 2B.2) -- single source of truth for drift math
# lives in ``app/mid_drift_windows.py``. This module re-exports the
# helper so existing imports (including the public-API name
# ``_drift_bps_over_window`` used by tests) keep working unchanged,
# while production paths go through ``compute_mid_drift_windows`` +
# ``state.mid_drift_windows`` cache.
from app.mid_drift_windows import _drift_bps_over_window  # noqa: F401  (re-export for tests)


def compute_short_window_drifts(
    samples: Sequence[tuple[float, float]],
    *,
    now_mono: float,
    mid_now: float,
    window_10s: float = 10.0,
    window_30s: float = 30.0,
) -> tuple[Optional[float], Optional[float]]:
    """Return ``(drift_bps_10s, drift_bps_30s)``.

    Helper for the bot's tick path. Either value is ``None`` when the
    deque doesn't yet have a sample inside that window — gate stays
    dormant until both windows have warmed up. (Either being None
    plus the other being present is still actionable; the gate uses
    whichever is available.)
    """
    return (
        _drift_bps_over_window(
            samples,
            now_mono=now_mono,
            mid_now=mid_now,
            window_seconds=window_10s,
        ),
        _drift_bps_over_window(
            samples,
            now_mono=now_mono,
            mid_now=mid_now,
            window_seconds=window_30s,
        ),
    )


def evaluate_inventory_drift_gate(
    *,
    position_qty: float,
    effective_abs_cap: float,
    drift_bps_10s: Optional[float],
    drift_bps_30s: Optional[float],
    inventory_pct_threshold: float,
    drift_threshold_bps_10s: float,
    drift_threshold_bps_30s: float,
    enabled: bool,
) -> tuple[Optional[QuoteEligibility], str, Optional[float]]:
    """Return ``(override_eligibility, reason, drift_used_bps)``.

    Gate semantics:

    * ``override_eligibility=None`` when the gate is dormant
      (disabled, util below threshold, drift below threshold, or
      drift aligned with inventory). Caller should keep its
      existing eligibility.
    * ``QuoteEligibility.QUOTE_SELL_ONLY`` when LONG + drift DOWN.
      Suppresses BUY → don't add to the bad-side position.
    * ``QuoteEligibility.QUOTE_BUY_ONLY`` when SHORT + drift UP.
      Suppresses SELL → don't add to the bad-side position.

    The third return value is the signed drift that triggered the
    gate (in bps) — surfaced in telemetry. ``None`` when the gate
    stays dormant.

    Threshold convention: each window has its own threshold
    (``drift_threshold_bps_10s`` for 10s, ``drift_threshold_bps_30s``
    for 30s). The gate fires when **either** window's drift clears
    **its own** threshold AND is anti-aligned with inventory. This
    lets the operator tune sensitivity per window independently.
    """
    if not enabled:
        return None, "inventory_drift_disabled", None
    if effective_abs_cap <= 0 or not math.isfinite(effective_abs_cap):
        return None, "inventory_drift_no_cap", None
    if not math.isfinite(position_qty):
        return None, "inventory_drift_no_position", None

    # Util check — gate is position-defence, dormant at low inventory.
    util = abs(float(position_qty)) / float(effective_abs_cap)
    if util < float(inventory_pct_threshold):
        return None, f"inventory_drift_util_below:{util:.3f}", None

    # Position sign — needed for the anti-aligned check. Util > 0
    # guarantees we have a nonzero position here.
    pos_sign = 1.0 if position_qty > 0 else -1.0

    # Try each window in order — the SHORTER window (10s) first so the
    # gate reacts as fast as the data lets it. The longer (30s) window
    # is the fallback when the 10s window is below its threshold but
    # the 30s window has accumulated enough motion to clear its.
    candidates: list[tuple[float, str, float]] = []
    if (
        drift_bps_10s is not None
        and math.isfinite(drift_bps_10s)
        and abs(drift_bps_10s) >= float(drift_threshold_bps_10s)
        and (drift_bps_10s * pos_sign) < 0
    ):
        candidates.append(
            (
                float(drift_bps_10s),
                "10s",
                float(drift_threshold_bps_10s),
            )
        )
    if (
        drift_bps_30s is not None
        and math.isfinite(drift_bps_30s)
        and abs(drift_bps_30s) >= float(drift_threshold_bps_30s)
        and (drift_bps_30s * pos_sign) < 0
    ):
        candidates.append(
            (
                float(drift_bps_30s),
                "30s",
                float(drift_threshold_bps_30s),
            )
        )

    if not candidates:
        return None, "inventory_drift_ok", None

    # If both windows fired, pick the one with the larger |drift| — it
    # carries the stronger signal and is more informative in the
    # telemetry trace.
    chosen = max(candidates, key=lambda c: abs(c[0]))
    drift_used, window_label, threshold = chosen
    direction = "down" if drift_used < 0 else "up"
    override = (
        QuoteEligibility.QUOTE_SELL_ONLY
        if drift_used < 0
        else QuoteEligibility.QUOTE_BUY_ONLY
    )
    reason = (
        f"inventory_drift_{direction}:"
        f"util={util:.3f},drift_{window_label}={drift_used:+.2f}bps"
        f">={threshold:.2f}"
    )
    return override, reason, drift_used


# ---------------------------------------------------------------------
# Widening contribution (live from day one — non-sentinel default)
# ---------------------------------------------------------------------


def widening_bps(
    *,
    position_qty: float,
    effective_abs_cap: float,
    drift_bps_10s: Optional[float],
    drift_bps_30s: Optional[float],
    inventory_pct_threshold: float,
    drift_threshold_bps_10s: float,
    drift_threshold_bps_30s: float,
    enabled: bool,
    max_half_spread_bps: float,
    widen_bps: float,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` widening contribution.

    Asymmetric — widens the SIDE the bot would otherwise use to add
    to its position when drift fights it:

    * LONG  + drift DOWN: widen BID (suppresses BUY → no piling on).
    * SHORT + drift UP:   widen ASK (suppresses SELL → no piling on).
    * Otherwise zero on both sides.

    ``widen_bps`` magnitude semantics (mirroring slow_trend_gate):

    * ``-1.0`` (sentinel): falls back to ``max_half_spread_bps``
      (gate-equivalent — effectively dark on the suppressed side).
    * ``>= 0``: clamped to ``[0, max_half_spread_bps]``; bot widens
      the suppressed side by exactly this many bps while staying in
      the market on the other.

    **Recommended starting value: 15.0 bps** — wider than
    slow_trend's 10 bp because this gate's trigger is more specific
    (position-aware AND short-window AND anti-aligned). When it
    fires, the conviction "don't add here" is higher and the
    backoff should reflect that.
    """
    override, _, _ = evaluate_inventory_drift_gate(
        position_qty=position_qty,
        effective_abs_cap=effective_abs_cap,
        drift_bps_10s=drift_bps_10s,
        drift_bps_30s=drift_bps_30s,
        inventory_pct_threshold=inventory_pct_threshold,
        drift_threshold_bps_10s=drift_threshold_bps_10s,
        drift_threshold_bps_30s=drift_threshold_bps_30s,
        enabled=enabled,
    )
    if override is None:
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    if override == QuoteEligibility.QUOTE_SELL_ONLY:
        # LONG + drift down: BUY suppressed → widen bid.
        return (effective, 0.0)
    if override == QuoteEligibility.QUOTE_BUY_ONLY:
        # SHORT + drift up: SELL suppressed → widen ask.
        return (0.0, effective)
    return (0.0, 0.0)
