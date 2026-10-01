"""Slow-trend defence gate — v1.4.102.

Complements the existing ``_long_drift_eligibility`` gate in
``quote_eligibility.py``. The existing gate watches the **5-minute
window** with a **50-bps threshold** — it catches *acute* moves
(sharp drops/spikes within 5 min) but **misses slow grinds**.

Root cause for needing this gate: snapshot
``v1.4.92-260520-074215`` showed the bot accumulating LONG +9 over
48 minutes (06:00 → 06:48 local) while TON drifted from 1.985 →
1.961 (-121 bp total). Each individual 5-minute window had only
~10-12 bp of drift — well below the existing gate's 50-bp threshold
— so nothing fired. By the time the price snapped to 1.94 at 06:50
local, the position was already maxed long; the existing
``long_drift`` gate finally fired but the damage was done.

**This gate fills the slow-grind gap** with a longer window
(default 15 min) and a lower threshold (default 25 bp). The two
gates compose: ``long_drift`` for acute moves, ``slow_trend`` for
sustained grinds.

Design choices vs. ``long_drift_eligibility``:

* **Longer window** — 15 min default (vs 5 min) catches drift
  patterns that the shorter window averages away.
* **Lower threshold** — 25 bp default (vs 50 bp) makes the gate
  responsive to grinds, not just acute moves. The lower threshold
  is safe because the wider window already filters out
  high-frequency noise.
* **Widening framework from day one** — unlike the existing
  ``long_drift`` gate which is still binary, this gate's
  ``widening_bps()`` accepts a coefficient parameter. With
  ``slow_trend_widen_bps=10.0`` (the recommended TON-profile
  starting value), the gate produces a moderate +10 bp widening
  on the adding side when active — wider quotes, fewer fills,
  but never dark. No coefficient-iteration phase needed; the
  starting value IS the operating value.
* **Anti-aligned semantics** — same as ``long_drift``: when drift
  is DOWN, suppress BUY (don't add to long in falling market).
  When drift is UP, suppress SELL (don't add to short in rising
  market).

The gate is independent of position. It fires purely on the drift
signal. The reasoning: even at flat position, opening a new long
into a sustained downtrend is bad. Pairing with the inventory-skew
machinery (which fights against the resulting position) gives
defence-in-depth.

Config (TON profile):
    SLOW_TREND_GATE_ENABLED=true
    SLOW_TREND_WINDOW_SECONDS=900           (15 min)
    SLOW_TREND_THRESHOLD_BPS=25.0
    SLOW_TREND_MIN_SAMPLES=60
    SLOW_TREND_ANCHOR_FRACTION=0.2
    SLOW_TREND_WIDEN_BPS=10.0               (off sentinel; live widening from day one)
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

from app.enums import QuoteEligibility


def _median(xs: Sequence[float]) -> float:
    """Pure-stdlib median to avoid an import for one call."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def evaluate_slow_trend_gate(
    *,
    samples_long: Sequence[tuple[float, float]],
    now_mono: float,
    mid_now: float,
    enabled: bool,
    window_seconds: float,
    threshold_bps: float,
    min_samples: int,
    anchor_fraction: float,
) -> tuple[Optional[QuoteEligibility], str, Optional[float]]:
    """Return ``(override_eligibility, reason, drift_bps)``.

    * ``override_eligibility=None`` when the gate is inactive
      (disabled, insufficient data, drift below threshold, etc.).
      Caller should keep its existing eligibility.
    * ``QuoteEligibility.QUOTE_SELL_ONLY`` when drift is DOWN —
      don't add to long in falling market. Bot is allowed to SELL
      (reduce long / open short).
    * ``QuoteEligibility.QUOTE_BUY_ONLY`` when drift is UP — don't
      add to short in rising market. Bot is allowed to BUY (reduce
      short / open long).

    Mirrors the anchored-median logic of
    ``quote_eligibility._long_drift_eligibility`` so the two gates
    share the same noise-robustness properties; only the window /
    threshold differ.

    The third return value is the observed drift in bps (signed)
    even when the gate stays at QUOTE_BOTH — surfaced in telemetry
    for operator visibility.
    """
    if not enabled:
        return None, "slow_trend_disabled", None
    if threshold_bps <= 0.0:
        return None, "slow_trend_disabled", None
    if mid_now <= 0 or not math.isfinite(mid_now):
        return None, "slow_trend_no_mid", None

    cutoff = now_mono - window_seconds
    in_window = [(t, m) for t, m in samples_long if t >= cutoff and m > 0]
    if len(in_window) < min_samples:
        return None, "slow_trend_warmup", None

    n = len(in_window)
    if anchor_fraction <= 0.0:
        old_prices = [in_window[0][1]]
        new_prices = [mid_now]
    else:
        anchor_count = max(3, int(n * anchor_fraction))
        anchor_count = min(anchor_count, max(3, n // 2))
        old_prices = [m for _, m in in_window[:anchor_count]]
        new_prices = [m for _, m in in_window[-anchor_count:]]

    oldest_anchor = _median(old_prices)
    newest_anchor = _median(new_prices)
    if (
        oldest_anchor <= 0
        or not math.isfinite(oldest_anchor)
        or not math.isfinite(newest_anchor)
    ):
        return None, "slow_trend_no_anchor", None

    drift_bps = (newest_anchor / oldest_anchor - 1.0) * 1e4

    if drift_bps >= threshold_bps:
        return (
            QuoteEligibility.QUOTE_BUY_ONLY,
            f"slow_trend_up:{drift_bps:+.2f}bps>={threshold_bps:.2f}",
            float(drift_bps),
        )
    if drift_bps <= -threshold_bps:
        return (
            QuoteEligibility.QUOTE_SELL_ONLY,
            f"slow_trend_down:{drift_bps:+.2f}bps<=-{threshold_bps:.2f}",
            float(drift_bps),
        )
    return None, "slow_trend_ok", float(drift_bps)


# ------------------------------------------------------------------
# Widening contribution (live from day one — non-sentinel default)
# ------------------------------------------------------------------


def widening_bps(
    *,
    samples_long: Sequence[tuple[float, float]],
    now_mono: float,
    mid_now: float,
    max_half_spread_bps: float,
    enabled: bool,
    window_seconds: float,
    threshold_bps: float,
    min_samples: int,
    anchor_fraction: float,
    widen_bps: float,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` widening contribution.

    Asymmetric — widens the SIDE the bot would otherwise use to
    add to a position that fights the trend:

    * Drift DOWN beyond threshold: widen BID side (suppresses
      BUY — don't catch falling knife).
    * Drift UP beyond threshold: widen ASK side (suppresses SELL
      — don't sell into a rally).
    * Otherwise zero on both.

    ``widen_bps`` magnitude semantics:

    * ``-1.0`` (sentinel): falls back to ``max_half_spread_bps``
      (gate-equivalent, effectively dark on the suppressed side).
    * ``>= 0``: clamped to ``[0, max_half_spread_bps]``; bot
      widens by exactly this many bps on the suppressed side
      while staying in the market on the other.

    **Recommended starting value: 10.0 bps.** Empirically wide
    enough to deter fills on the suppressed side during a real
    trend, while not so wide that the bot disappears for ages
    after a single false positive. Operator can tune via
    ``SLOW_TREND_WIDEN_BPS`` env knob.
    """
    override, _, _ = evaluate_slow_trend_gate(
        samples_long=samples_long,
        now_mono=now_mono,
        mid_now=mid_now,
        enabled=enabled,
        window_seconds=window_seconds,
        threshold_bps=threshold_bps,
        min_samples=min_samples,
        anchor_fraction=anchor_fraction,
    )
    if override is None:
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    if override == QuoteEligibility.QUOTE_SELL_ONLY:
        # Drift down: BUY suppressed → widen bid.
        return (effective, 0.0)
    if override == QuoteEligibility.QUOTE_BUY_ONLY:
        # Drift up: SELL suppressed → widen ask.
        return (0.0, effective)
    return (0.0, 0.0)
