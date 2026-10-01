"""
Pre-inventory quote eligibility: freshness (local receipt + gaps) and short-horizon mid drift/jump.

This layer runs after hard risk/safety checks and optional order-state uncertainty, and before
:class:`~app.quoting.compute_quote_decision` inventory/spread logic. It only caps which sides may
quote; it does not reshape prices inside QuoteEngine.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Sequence

from app.config import Settings
from app.enums import QuoteEligibility, Side
from app.utils.time import utc_now


def _rank(e: QuoteEligibility) -> int:
    if e == QuoteEligibility.HOLD_ALL:
        return 3
    if e in (QuoteEligibility.QUOTE_BUY_ONLY, QuoteEligibility.QUOTE_SELL_ONLY):
        return 2
    return 1


def effective_staleness_ms_from_exchange_ts(ts_exchange_ms: Optional[int]) -> Optional[float]:
    """Wall-clock lag vs exchange timestamp (ms), when ``ts_exchange_ms`` is present on the book."""
    if ts_exchange_ms is None:
        return None
    try:
        wall_ms = utc_now().timestamp() * 1000.0
        return max(0.0, wall_ms - float(ts_exchange_ms))
    except (TypeError, ValueError, OverflowError):
        return None


def more_restrictive(
    a: QuoteEligibility, b: QuoteEligibility
) -> QuoteEligibility:
    """Side-set intersection.

    BUG-007 fix: previously this was a rank comparison
    (HOLD_ALL > one-sided > BOTH), so ``more_restrictive(BUY_ONLY,
    SELL_ONLY)`` returned whichever was the first arg and the bot
    quoted that side even though both safety gates explicitly
    forbade their counterparty. The fail-open semantics are
    dangerous: "BUY only forbidden + SELL only forbidden" must mean
    HOLD_ALL, not "whichever cap arrived first wins".

    The new behaviour treats each enum value as a side set:
        HOLD_ALL       = ∅
        QUOTE_BUY_ONLY = {BUY}
        QUOTE_SELL_ONLY = {SELL}
        QUOTE_BOTH     = {BUY, SELL}
    and returns the intersection. ``BUY_ONLY ∩ SELL_ONLY = ∅ = HOLD_ALL``.

    Other transitions still work as before:
        BOTH ∩ X      = X
        HOLD_ALL ∩ X  = HOLD_ALL
        X ∩ X         = X
    """
    if a == QuoteEligibility.HOLD_ALL or b == QuoteEligibility.HOLD_ALL:
        return QuoteEligibility.HOLD_ALL
    if a == QuoteEligibility.QUOTE_BOTH:
        return b
    if b == QuoteEligibility.QUOTE_BOTH:
        return a
    if a == b:
        return a
    # Both are one-sided AND different → no side allowed by both
    return QuoteEligibility.HOLD_ALL


def merge_eligibility_freshness_drift(
    fresh: QuoteEligibility,
    drift: QuoteEligibility,
) -> tuple[QuoteEligibility, str]:
    """
    Combine independent freshness and drift caps. Conflicting one-sided directions => HOLD_ALL.
    """
    if fresh == QuoteEligibility.HOLD_ALL or drift == QuoteEligibility.HOLD_ALL:
        return QuoteEligibility.HOLD_ALL, "freshness_or_drift_hold"
    if fresh == QuoteEligibility.QUOTE_BOTH:
        return drift, "drift_only" if drift != QuoteEligibility.QUOTE_BOTH else "ok"
    if drift == QuoteEligibility.QUOTE_BOTH:
        return fresh, "freshness_only" if fresh != QuoteEligibility.QUOTE_BOTH else "ok"
    if fresh != drift:
        return QuoteEligibility.HOLD_ALL, "freshness_drift_conflict"
    return fresh, "aligned"


@dataclass(frozen=True, slots=True)
class QuoteEligibilitySnapshot:
    """Immutable snapshot for fills and telemetry."""

    quote_eligibility_state: str
    quote_eligibility_reason: str
    seconds_since_last_public_book_update: Optional[float]
    effective_staleness_ms: Optional[float]
    market_data_gap_p95_ms: Optional[float]
    market_data_gap_median_ms: Optional[float]
    mid_return_100ms_bps: Optional[float]
    mid_return_250ms_bps: Optional[float]
    mid_return_500ms_bps: Optional[float]
    jump_100ms_bps: Optional[float]
    jump_250ms_bps: Optional[float]
    jump_500ms_bps: Optional[float]
    in_cooldown: bool
    # Multi-minute mid drift in bps over ``DRIFT_LONG_WINDOW_SECONDS``;
    # None until the long-window deque has accumulated enough samples
    # (see BUGS/bug-002.md).
    mid_return_long_window_bps: Optional[float] = None


@dataclass(slots=True)
class QuoteEligibilityResult:
    eligibility: QuoteEligibility
    reason: str
    seconds_since_last_public_book_update: Optional[float]
    effective_staleness_ms: Optional[float]
    market_data_gap_p95_ms: Optional[float]
    market_data_gap_median_ms: Optional[float]
    mid_return_100ms_bps: Optional[float]
    mid_return_250ms_bps: Optional[float]
    mid_return_500ms_bps: Optional[float]
    jump_100ms_bps: Optional[float]
    jump_250ms_bps: Optional[float]
    jump_500ms_bps: Optional[float]
    in_cooldown: bool
    counter_tags: tuple[str, ...] = ()
    mid_return_long_window_bps: Optional[float] = None

    def to_snapshot(self) -> QuoteEligibilitySnapshot:
        return QuoteEligibilitySnapshot(
            quote_eligibility_state=self.eligibility.value,
            quote_eligibility_reason=self.reason,
            seconds_since_last_public_book_update=self.seconds_since_last_public_book_update,
            effective_staleness_ms=self.effective_staleness_ms,
            market_data_gap_p95_ms=self.market_data_gap_p95_ms,
            market_data_gap_median_ms=self.market_data_gap_median_ms,
            mid_return_100ms_bps=self.mid_return_100ms_bps,
            mid_return_250ms_bps=self.mid_return_250ms_bps,
            mid_return_500ms_bps=self.mid_return_500ms_bps,
            jump_100ms_bps=self.jump_100ms_bps,
            jump_250ms_bps=self.jump_250ms_bps,
            jump_500ms_bps=self.jump_500ms_bps,
            in_cooldown=self.in_cooldown,
            mid_return_long_window_bps=self.mid_return_long_window_bps,
        )


def _mid_at_or_before(samples: Sequence[tuple[float, float]], cut_mono: float) -> Optional[float]:
    """Samples ordered oldest->newest; return last mid with timestamp <= cut_mono."""
    last: Optional[float] = None
    for t, m in samples:
        if t <= cut_mono:
            last = m
        else:
            break
    return last


def _mid_return_bps(
    samples: Sequence[tuple[float, float]],
    now_mono: float,
    mid_now: float,
    window_s: float,
) -> Optional[float]:
    if mid_now <= 0 or not math.isfinite(mid_now):
        return None
    old = _mid_at_or_before(samples, now_mono - window_s)
    if old is None or old <= 0 or not math.isfinite(old):
        return None
    return (mid_now / old - 1.0) * 10_000.0


def _jump_bps(ret: Optional[float]) -> Optional[float]:
    if ret is None:
        return None
    return abs(float(ret))


def compute_mid_kinematics(
    samples: Sequence[tuple[float, float]],
    now_mono: float,
    mid_now: float,
) -> dict[str, Optional[float]]:
    """Returns signed returns and absolute jumps for 100/250/500ms windows."""
    out: dict[str, Optional[float]] = {}
    for label, w in (("100ms", 0.1), ("250ms", 0.25), ("500ms", 0.5)):
        r = _mid_return_bps(samples, now_mono, mid_now, w)
        out[f"mid_return_{label}_bps"] = r
        out[f"jump_{label}_bps"] = _jump_bps(r)
    return out


def _freshness_eligibility(
    settings: Settings,
    *,
    age_ms: Optional[float],
    gap_p95_ms: Optional[float],
    gap_median_ms: Optional[float],
    eff_staleness_ms: Optional[float],
    last_gap_ms: Optional[float] = None,
    gap_sample_count: Optional[int] = None,
    position_qty: Optional[float] = None,
) -> tuple[QuoteEligibility, str]:
    """Map book age / gap p95 into a freshness cap.

    Clock semantics (intentionally asymmetric):

    * ``age_ms`` is derived from the local monotonic receipt of the last public book
      update — it measures how long *we* have been without a fresh book sample. This is
      the correct input for eligibility gating: it is independent of exchange geography.

    * ``eff_staleness_ms`` is ``wall_now - exchange_timestamp``. It embeds the one-way
      network delay from the exchange to our host and is therefore permanently ~220ms
      even for books that just arrived (for hosts far from the matching engine). Using
      it for gating means "gate harder the farther you are from the exchange", which is
      a deployment-geography artifact, not a market-data quality signal. It is kept as
      a **monitoring / telemetry** metric only (flows through to ``QuoteEligibilityResult``
      and thereby into snapshots / fills) — NOT into the gating threshold comparisons.

    ``gap_p95_ms`` is the rolling p95 of *local* inter-update gaps, which is also
    deployment-independent: it measures the worst-case silence window we actually
    observed locally, irrespective of one-way network delay. This remains a gating input.

    Legacy behavior — where ``eff_staleness_ms`` fed the gate via ``max(age, eff_stal)``
    — is available behind the opt-in ``QUOTE_FRESHNESS_USE_EXCHANGE_STALENESS_FOR_GATING``
    setting for operators who explicitly want it.
    """
    hold_age = float(settings.quote_hold_max_book_age_ms)
    one_age = float(settings.quote_one_sided_max_book_age_ms)
    hold_gap = float(settings.quote_hold_max_gap_p95_ms)
    one_gap = float(settings.quote_one_sided_max_gap_p95_ms)

    use_exchange_gate = bool(
        getattr(settings, "quote_freshness_use_exchange_staleness_for_gating", False)
    )
    age_use: Optional[float]
    if use_exchange_gate:
        ages = [
            x for x in (age_ms, eff_staleness_ms)
            if x is not None and math.isfinite(float(x))
        ]
        age_use = max(ages) if ages else None
        age_tag = "receipt_or_eff_staleness_ms"
    else:
        if age_ms is not None and math.isfinite(float(age_ms)):
            age_use = float(age_ms)
        else:
            age_use = None
        age_tag = "local_receipt_ms"

    if age_use is None and gap_p95_ms is None:
        return QuoteEligibility.HOLD_ALL, "freshness_missing_age_and_gap"

    age_hold = age_use is not None and age_use > hold_age
    p95_hold = gap_p95_ms is not None and gap_p95_ms > hold_gap
    age_one = age_use is not None and age_use > one_age
    p95_one = gap_p95_ms is not None and gap_p95_ms > one_gap

    # --- p95-hold cold-start guard (Fix B) ---
    # Skip the p95-based hold/one-sided when the gap ring hasn't warmed up yet.
    # A single startup/reconnect outlier in a small sample can pin p95 above the
    # threshold even though the current feed is delivering fine. Age-based gating
    # is UNAFFECTED (age is a current-state read, not a rolling statistic).
    # Observed in ``tmp/snap_20260418_132744``: 74 samples including one 3.9 s
    # reconnect outlier → p95=1813 ms, bot held in HOLD_ALL for the first 5+ min
    # of every restart until the ring rolled the outlier out.
    p95_warmup_skip = False
    min_samples = int(getattr(settings, "quote_freshness_p95_min_gap_samples", 0) or 0)
    if (
        min_samples > 0
        and (p95_hold or p95_one)
        and gap_sample_count is not None
        and gap_sample_count < min_samples
    ):
        p95_warmup_skip = True

    # --- Live-book-fresh override (Fix A, extended) ---
    # p95 is a rolling statistic about the past. A single outlier gap can keep the
    # gate tripped even when the current feed is demonstrably live. The override
    # fires when BOTH:
    #
    #   1. the book is currently fresh (age_ms <= threshold), AND
    #   2. the feed is producing recent gaps at healthy cadence — either
    #      (a) the rolling median is healthy, OR
    #      (b) the MOST RECENT gap (``last_gap_ms``) is healthy.
    #
    # The ``last_gap_ms`` path (A) handles the case where the ring-buffer median
    # is still polluted by an old outlier but new samples are arriving fine.
    # Observed in ``tmp/snap_20260418_132744``: median=748 ms (polluted by startup
    # jitter), last_gap=416 ms (feed fine right now). Median-only override failed;
    # last_gap path would have fired immediately.
    #
    # Only applies to the p95 portion of the gates — an age-based trigger means
    # the feed is stale RIGHT NOW, and no rolling statistic can make that safe.
    p95_override_fired = False
    if (
        (p95_hold or p95_one)
        and bool(getattr(settings, "quote_freshness_live_override_enabled", True))
    ):
        override_max_age = float(
            getattr(settings, "quote_freshness_live_override_max_book_age_ms", 150.0)
        )
        override_max_median = float(
            getattr(settings, "quote_freshness_live_override_max_gap_median_ms", 500.0)
        )
        override_max_last = float(
            getattr(settings, "quote_freshness_live_override_max_last_gap_ms", 500.0)
        )
        age_is_live = age_use is not None and age_use <= override_max_age
        median_is_healthy = (
            gap_median_ms is not None
            and math.isfinite(float(gap_median_ms))
            and float(gap_median_ms) <= override_max_median
        )
        last_is_healthy = (
            last_gap_ms is not None
            and math.isfinite(float(last_gap_ms))
            and float(last_gap_ms) <= override_max_last
        )
        if age_is_live and (median_is_healthy or last_is_healthy):
            p95_override_fired = True

    # The p95 portion is skipped if EITHER the warmup guard or the live override
    # fires. Age-based holds are never skipped.
    p95_skipped = p95_warmup_skip or p95_override_fired

    hold_reasons: list[str] = []
    if age_hold:
        hold_reasons.append(f"{age_tag}>{hold_age}")
    if p95_hold and not p95_skipped:
        hold_reasons.append(f"gap_p95_ms>{hold_gap}")

    if hold_reasons:
        return QuoteEligibility.HOLD_ALL, "freshness_hold:" + ",".join(hold_reasons)

    one_reasons: list[str] = []
    if age_one:
        one_reasons.append(f"{age_tag}>{one_age}")
    if p95_one and not p95_skipped:
        one_reasons.append(f"gap_p95_ms>{one_gap}")

    if one_reasons:
        # 2026-05-12 codex-#5: inventory-aware fallback. When the bot
        # is carrying a position, prefer the INVENTORY-REDUCING side
        # (asks when long, bids when short) — that side closes
        # exposure even if the book is stale. The static config
        # ``QUOTE_FRESHNESS_ONE_SIDED_PREFERENCE`` is kept as a tie-
        # breaker (used when flat) or as an explicit operator override
        # via ``QUOTE_FRESHNESS_ONE_SIDED_PREFER_REDUCING=false``.
        prefer_reducing = bool(
            getattr(settings, "quote_freshness_one_sided_prefer_reducing", True)
        )
        if (
            prefer_reducing
            and position_qty is not None
            and abs(position_qty) > 1e-12
        ):
            if position_qty > 0:
                # Long → ask is the reducing side.
                return (
                    QuoteEligibility.QUOTE_SELL_ONLY,
                    "freshness_one_sided:" + ",".join(one_reasons) + "|reducing_long",
                )
            else:
                # Short → bid is the reducing side.
                return (
                    QuoteEligibility.QUOTE_BUY_ONLY,
                    "freshness_one_sided:" + ",".join(one_reasons) + "|reducing_short",
                )
        # Flat (or reducing-preference disabled): fall back to the
        # static config preference. Codex's stronger recommendation
        # was "prefer HOLD_ALL when flat", but that's a behavioural
        # change with potential rebate-income downside; left as a
        # follow-up if the conservative fallback proves insufficient.
        pref = (settings.quote_freshness_one_sided_preference or "BUY").upper().strip()
        if pref == "SELL":
            return QuoteEligibility.QUOTE_SELL_ONLY, "freshness_one_sided:" + ",".join(one_reasons)
        return QuoteEligibility.QUOTE_BUY_ONLY, "freshness_one_sided:" + ",".join(one_reasons)

    if p95_warmup_skip and not p95_override_fired:
        return QuoteEligibility.QUOTE_BOTH, "freshness_ok_p95_warmup"
    if p95_override_fired:
        return QuoteEligibility.QUOTE_BOTH, "freshness_ok_p95_override_live"
    return QuoteEligibility.QUOTE_BOTH, "freshness_ok"


def _effective_drift_threshold(
    *,
    abs_bps: float,
    vol_multiplier: float,
    vol_bps: Optional[float],
) -> float:
    """Blend the absolute bps floor with a vol-scaled multiplier.

    The effective threshold is ``max(abs_bps, multiplier × vol_bps)``. This
    gives the drift filter TWO knobs per horizon:

    - ``abs_bps``: absolute floor — prevents hair-trigger pauses in near-zero
      vol conditions where a tiny noise move would otherwise fire the gate.
    - ``vol_multiplier``: adaptive scaling — at higher observed volatility
      the gate relaxes automatically, so normal volatility doesn't keep the
      bot paused.

    When ``vol_multiplier=0`` (default), the threshold is just ``abs_bps`` —
    preserves legacy behaviour. When ``vol_bps`` is missing or non-finite,
    the multiplier path is ignored.
    """
    if vol_multiplier <= 0.0 or vol_bps is None:
        return float(abs_bps)
    try:
        v = float(vol_bps)
    except (TypeError, ValueError):
        return float(abs_bps)
    if not math.isfinite(v) or v < 0.0:
        return float(abs_bps)
    return max(float(abs_bps), float(vol_multiplier) * v)


def _drift_eligibility(
    settings: Settings,
    *,
    kin: dict[str, Optional[float]],
    vol_bps: Optional[float] = None,
) -> tuple[QuoteEligibility, str]:
    """
    Use signed short-horizon returns: rising mid -> vulnerable ask (BUY-only); falling -> SELL-only.
    Large magnitude => HOLD. Abrupt jump triggers hold.

    Each threshold blends an absolute bps floor with a vol-scaled multiplier
    (see :func:`_effective_drift_threshold`). This adapts the filter to the
    symbol's observed volatility instead of locking it at HL-calibrated
    bps constants.
    """
    r100 = kin.get("mid_return_100ms_bps")
    r250 = kin.get("mid_return_250ms_bps")
    r500 = kin.get("mid_return_500ms_bps")
    j250 = kin.get("jump_250ms_bps")

    j_thr = _effective_drift_threshold(
        abs_bps=float(settings.jump_hold_250ms_bps),
        vol_multiplier=float(getattr(settings, "jump_hold_250ms_vol_multiplier", 0.0)),
        vol_bps=vol_bps,
    )
    if j250 is not None and j250 >= j_thr:
        return QuoteEligibility.HOLD_ALL, f"jump_250ms_bps>={j_thr:.3f}"

    h500 = _effective_drift_threshold(
        abs_bps=float(settings.drift_hold_500ms_bps),
        vol_multiplier=float(getattr(settings, "drift_hold_500ms_vol_multiplier", 0.0)),
        vol_bps=vol_bps,
    )
    if r500 is not None and abs(r500) >= h500:
        return QuoteEligibility.HOLD_ALL, f"abs_mid_return_500ms_bps>={h500:.3f}"

    b100 = _effective_drift_threshold(
        abs_bps=float(settings.drift_block_100ms_bps),
        vol_multiplier=float(getattr(settings, "drift_block_100ms_vol_multiplier", 0.0)),
        vol_bps=vol_bps,
    )
    b250 = _effective_drift_threshold(
        abs_bps=float(settings.drift_block_250ms_bps),
        vol_multiplier=float(getattr(settings, "drift_block_250ms_vol_multiplier", 0.0)),
        vol_bps=vol_bps,
    )

    bias: Optional[str] = None  # "buy_side" | "sell_side"
    reasons: list[str] = []

    def _add_bias(want: str, msg: str) -> None:
        nonlocal bias
        if want == "buy_side":
            if bias == "sell_side":
                bias = "conflict"
            elif bias != "conflict":
                bias = "buy_side"
        elif want == "sell_side":
            if bias == "buy_side":
                bias = "conflict"
            elif bias != "conflict":
                bias = "sell_side"
        reasons.append(msg)

    if r100 is not None:
        if r100 > b100:
            _add_bias("buy_side", f"up_drift_100ms>{b100:.3f}")
        elif r100 < -b100:
            _add_bias("sell_side", f"down_drift_100ms>{b100:.3f}")
    if r250 is not None:
        if r250 > b250:
            _add_bias("buy_side", f"up_drift_250ms>{b250:.3f}")
        elif r250 < -b250:
            _add_bias("sell_side", f"down_drift_250ms>{b250:.3f}")

    if bias == "conflict":
        return QuoteEligibility.HOLD_ALL, "drift_direction_conflict:" + ",".join(reasons)
    if bias == "buy_side":
        return QuoteEligibility.QUOTE_BUY_ONLY, "drift:" + ",".join(reasons)
    if bias == "sell_side":
        return QuoteEligibility.QUOTE_SELL_ONLY, "drift:" + ",".join(reasons)
    return QuoteEligibility.QUOTE_BOTH, "drift_ok"


def _median(values: Sequence[float]) -> float:
    """Lightweight median of a non-empty numeric sequence."""
    s = sorted(float(v) for v in values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _long_drift_eligibility(
    settings: Settings,
    *,
    samples_long: Sequence[tuple[float, float]],
    now_mono: float,
    mid_now: float,
) -> tuple[QuoteEligibility, str, Optional[float]]:
    """Multi-minute drift gate. See BUGS/bug-002.md.

    Anchored on **smoothed endpoints** rather than single samples:

    * The "5-min-ago reference" is the median of the first
      ``anchor_count`` samples in the window.
    * The "now reference" is the median of the last ``anchor_count``
      samples in the window.

    ``anchor_count = max(3, int(len(in_window) * anchor_fraction))``.
    With ``anchor_fraction=0.2`` (default) and 30 samples in window,
    each anchor is the median of 6 samples. A single transient
    aberration in either anchor period contributes only 1/anchor_count
    to the median — robust to single-sample spikes that would otherwise
    flip the gate for the full retention window.

    Set ``anchor_fraction=0.0`` to fall back to single-sample anchoring
    (oldest in-window sample vs ``mid_now``).

    Behaviour:

    * Threshold ``DRIFT_BLOCK_LONG_WINDOW_BPS = 0`` → gate disabled,
      returns ``(QUOTE_BOTH, "long_drift_disabled", None)``.
    * Insufficient samples (cold start; fewer than
      ``DRIFT_LONG_WINDOW_MIN_SAMPLES`` inside the window) →
      ``(QUOTE_BOTH, "long_drift_warmup", None)``.
    * Drift below threshold → ``(QUOTE_BOTH, "long_drift_ok",
      <bps>)``.
    * Drift above +threshold (rising) → ``(QUOTE_BUY_ONLY,
      "long_drift_up:N.NNbps>=T", <bps>)``. Don't sell into a rising
      market; wait for the trend to cool or for inventory to fade.
    * Drift below -threshold (falling) → ``(QUOTE_SELL_ONLY,
      "long_drift_down:-N.NNbps<=-T", <bps>)``. Don't buy into a
      falling market (don't catch falling knives — the regression
      case in BUGS/bug-002.md).

    The third return value is the observed drift in bps (signed)
    even when the gate stays at QUOTE_BOTH — surfaced in
    ``QuoteEligibilityResult.mid_return_long_window_bps`` for
    operator visibility (Telegram /status, snapshot dumps).
    """
    threshold_bps = float(
        getattr(settings, "drift_block_long_window_bps", 0.0) or 0.0
    )
    if threshold_bps <= 0.0:
        return QuoteEligibility.QUOTE_BOTH, "long_drift_disabled", None
    if mid_now <= 0 or not math.isfinite(mid_now):
        return QuoteEligibility.QUOTE_BOTH, "long_drift_no_mid", None

    window_s = float(
        getattr(settings, "drift_long_window_seconds", 300.0) or 300.0
    )
    min_samples = int(
        getattr(settings, "drift_long_window_min_samples", 30) or 30
    )
    anchor_fraction = float(
        getattr(settings, "drift_long_window_anchor_fraction", 0.2) or 0.0
    )
    cutoff = now_mono - window_s
    # Only count samples actually inside the window. The deque's
    # retention policy already prunes by age at append time, but we
    # filter here defensively against test fixtures and stale
    # snapshots.
    in_window = [(t, m) for t, m in samples_long if t >= cutoff and m > 0]
    if len(in_window) < min_samples:
        return QuoteEligibility.QUOTE_BOTH, "long_drift_warmup", None

    # Determine anchor sizes. With anchor_fraction == 0.0 we fall back
    # to single-sample anchoring on both ends (oldest sample vs
    # ``mid_now``). With anchor_fraction > 0 we take the median of the
    # first / last N samples for noise robustness.
    n = len(in_window)
    if anchor_fraction <= 0.0:
        old_prices = [in_window[0][1]]
        new_prices = [mid_now]
    else:
        anchor_count = max(3, int(n * anchor_fraction))
        # Cap anchor_count so the two anchors don't overlap the middle
        # entirely (need at least 1 sample between them on a small
        # window).
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
        return QuoteEligibility.QUOTE_BOTH, "long_drift_no_anchor", None
    drift_bps = (newest_anchor / oldest_anchor - 1.0) * 1e4

    if drift_bps >= threshold_bps:
        return (
            QuoteEligibility.QUOTE_BUY_ONLY,
            f"long_drift_up:{drift_bps:+.2f}bps>={threshold_bps:.2f}",
            float(drift_bps),
        )
    if drift_bps <= -threshold_bps:
        return (
            QuoteEligibility.QUOTE_SELL_ONLY,
            f"long_drift_down:{drift_bps:+.2f}bps<=-{threshold_bps:.2f}",
            float(drift_bps),
        )
    return QuoteEligibility.QUOTE_BOTH, "long_drift_ok", float(drift_bps)


def compute_quote_eligibility(
    settings: Settings,
    *,
    order_state_uncertainty: bool,
    mid_now: float,
    now_mono: float,
    mid_samples: Sequence[tuple[float, float]],
    seconds_since_public_bbo: Optional[float],
    gap_median_ms: Optional[float],
    gap_p95_ms: Optional[float],
    effective_staleness_ms: Optional[float],
    uncertain_sides: Optional[frozenset[Side]] = None,
    vol_bps: Optional[float] = None,
    gap_last_ms: Optional[float] = None,
    gap_sample_count: Optional[int] = None,
    mid_samples_long: Sequence[tuple[float, float]] = (),
    # 2026-05-12 codex-#5: current position qty, used to pick the
    # inventory-reducing side when freshness degrades to one-sided.
    # Defaults to None (caller didn't supply) → fallback to the static
    # ``QUOTE_FRESHNESS_ONE_SIDED_PREFERENCE`` config.
    position_qty: Optional[float] = None,
) -> QuoteEligibilityResult:
    """
    Compute eligibility cap from order uncertainty, freshness, drift/jump, and (externally) recovery.

    Missing mid history for a window leaves that window's metrics None; drift does not relax
    freshness HOLD.

    ``order_state_uncertainty`` is reserved for *global* conditions (desync, etc.) — those
    halt both sides. ``uncertain_sides`` is the per-side signal: one side stuck in
    ``CANCEL_PENDING`` shouldn't silence the *other* side from capturing rebates, so a
    single-side entry in this set caps to the OPPOSITE-side QUOTE_*_ONLY. Two entries
    falls through to HOLD_ALL.
    """
    if not settings.quote_eligibility_enabled:
        return QuoteEligibilityResult(
            eligibility=QuoteEligibility.QUOTE_BOTH,
            reason="disabled",
            seconds_since_last_public_book_update=seconds_since_public_bbo,
            effective_staleness_ms=effective_staleness_ms,
            market_data_gap_p95_ms=gap_p95_ms,
            market_data_gap_median_ms=gap_median_ms,
            mid_return_100ms_bps=None,
            mid_return_250ms_bps=None,
            mid_return_500ms_bps=None,
            jump_100ms_bps=None,
            jump_250ms_bps=None,
            jump_500ms_bps=None,
            in_cooldown=False,
            counter_tags=(),
        )

    if order_state_uncertainty:
        return QuoteEligibilityResult(
            eligibility=QuoteEligibility.HOLD_ALL,
            reason="order_state_uncertainty",
            seconds_since_last_public_book_update=seconds_since_public_bbo,
            effective_staleness_ms=effective_staleness_ms,
            market_data_gap_p95_ms=gap_p95_ms,
            market_data_gap_median_ms=gap_median_ms,
            mid_return_100ms_bps=None,
            mid_return_250ms_bps=None,
            mid_return_500ms_bps=None,
            jump_100ms_bps=None,
            jump_250ms_bps=None,
            jump_500ms_bps=None,
            in_cooldown=False,
            counter_tags=("hold_all", "hold_order_uncertainty"),
        )

    # Per-side uncertainty: one side's stuck CANCEL_PENDING etc. should not suppress
    # the other side. Two uncertain sides → HOLD_ALL. One uncertain side → cap the
    # eligibility to the opposite side's QUOTE_*_ONLY. The rest of the eligibility
    # pipeline (freshness/drift) still runs after — the merge intersects with the
    # per-side cap.
    per_side_cap: Optional[QuoteEligibility] = None
    per_side_reason: Optional[str] = None
    per_side_tags: list[str] = []
    if uncertain_sides:
        uset = set(uncertain_sides)
        if Side.BUY in uset and Side.SELL in uset:
            return QuoteEligibilityResult(
                eligibility=QuoteEligibility.HOLD_ALL,
                reason="order_state_uncertainty_both_sides",
                seconds_since_last_public_book_update=seconds_since_public_bbo,
                effective_staleness_ms=effective_staleness_ms,
                market_data_gap_p95_ms=gap_p95_ms,
                market_data_gap_median_ms=gap_median_ms,
                mid_return_100ms_bps=None,
                mid_return_250ms_bps=None,
                mid_return_500ms_bps=None,
                jump_100ms_bps=None,
                jump_250ms_bps=None,
                jump_500ms_bps=None,
                in_cooldown=False,
                counter_tags=("hold_all", "hold_order_uncertainty_both_sides"),
            )
        if Side.BUY in uset:
            per_side_cap = QuoteEligibility.QUOTE_SELL_ONLY
            per_side_reason = "buy_side_uncertain"
            per_side_tags.append("one_sided_due_buy_uncertain")
        elif Side.SELL in uset:
            per_side_cap = QuoteEligibility.QUOTE_BUY_ONLY
            per_side_reason = "sell_side_uncertain"
            per_side_tags.append("one_sided_due_sell_uncertain")

    kin = compute_mid_kinematics(mid_samples, now_mono, mid_now)
    age_ms = (
        seconds_since_public_bbo * 1000.0
        if seconds_since_public_bbo is not None and math.isfinite(seconds_since_public_bbo)
        else None
    )

    fr, fr_reason = _freshness_eligibility(
        settings,
        age_ms=age_ms,
        gap_p95_ms=gap_p95_ms,
        gap_median_ms=gap_median_ms,
        eff_staleness_ms=effective_staleness_ms,
        last_gap_ms=gap_last_ms,
        gap_sample_count=gap_sample_count,
        position_qty=position_qty,
    )
    dr, dr_reason = _drift_eligibility(settings, kin=kin, vol_bps=vol_bps)
    merged, merge_reason = merge_eligibility_freshness_drift(fr, dr)

    # Long-window drift gate (multi-minute trend). Independent axis
    # from the sub-second freshness/drift gates above; applies as an
    # additional ``more_restrictive`` cap on the merged result. Same
    # treatment as ``per_side_cap`` below — it never UN-restricts,
    # only adds restriction.
    long_drift_elig, long_drift_reason, long_drift_bps = _long_drift_eligibility(
        settings,
        samples_long=mid_samples_long,
        now_mono=now_mono,
        mid_now=mid_now,
    )

    parts = [merge_reason, f"fresh={fr_reason}", f"drift={dr_reason}"]
    # Apply the per-side uncertainty cap (if any) by taking the more restrictive of
    # freshness+drift merged result and the per-side cap. This means a stuck BUY
    # AND a freshness hold simultaneously yields HOLD_ALL, while a stuck BUY alone
    # on a fresh book yields QUOTE_SELL_ONLY.
    if per_side_cap is not None:
        merged = more_restrictive(merged, per_side_cap)
        parts.append(f"per_side_uncertainty:{per_side_reason}")
    # Apply the long-window drift cap. Disabled / warmup / ok states
    # all return QUOTE_BOTH (more_restrictive(X, QUOTE_BOTH) = X), so
    # the gate never UN-restricts. Only fires on a true multi-minute
    # trend.
    if long_drift_elig != QuoteEligibility.QUOTE_BOTH:
        merged = more_restrictive(merged, long_drift_elig)
        parts.append(f"long_drift={long_drift_reason}")
    reason = "|".join(parts)

    tags: list[str] = []
    if merged == QuoteEligibility.HOLD_ALL:
        tags.append("hold_all")
        if "freshness_hold" in fr_reason or "freshness_missing" in fr_reason:
            tags.append("hold_due_stale")
        if "jump" in dr_reason or "drift_direction_conflict" in dr_reason:
            tags.append("hold_due_jump")
    elif merged in (QuoteEligibility.QUOTE_BUY_ONLY, QuoteEligibility.QUOTE_SELL_ONLY):
        if "drift:" in dr_reason and "drift_ok" not in dr_reason:
            tags.append("one_sided_due_drift")
        if "freshness_one_sided" in fr_reason:
            tags.append("one_sided_due_freshness")
        if long_drift_elig != QuoteEligibility.QUOTE_BOTH:
            tags.append("one_sided_due_long_drift")
    # Per-side uncertainty tags fire regardless of whether the cap was binding;
    # they're informational for the operator.
    tags.extend(per_side_tags)

    return QuoteEligibilityResult(
        eligibility=merged,
        reason=reason[:2000],
        seconds_since_last_public_book_update=seconds_since_public_bbo,
        effective_staleness_ms=effective_staleness_ms,
        market_data_gap_p95_ms=gap_p95_ms,
        market_data_gap_median_ms=gap_median_ms,
        mid_return_100ms_bps=kin.get("mid_return_100ms_bps"),
        mid_return_250ms_bps=kin.get("mid_return_250ms_bps"),
        mid_return_500ms_bps=kin.get("mid_return_500ms_bps"),
        jump_100ms_bps=kin.get("jump_100ms_bps"),
        jump_250ms_bps=kin.get("jump_250ms_bps"),
        jump_500ms_bps=kin.get("jump_500ms_bps"),
        in_cooldown=False,
        counter_tags=tuple(tags),
        mid_return_long_window_bps=long_drift_bps,
    )


# ------------------------------------------------------------------
# Widening contributions (gate-to-widening Phase 1, v1.4.8+)
# ------------------------------------------------------------------
#
# Freshness one-sided and recovery_cooldown both live inside the
# eligibility-result reason string / in_cooldown flag. Rather than
# pulling the state out into a separate module, we expose two
# helpers that take an eligibility result and produce the
# corresponding widening contribution. Callers (bot.py) invoke
# them when building the SpreadComposition each tick.


def freshness_one_sided_widening_bps(
    raw: QuoteEligibilityResult,
    *,
    max_half_spread_bps: float,
    widen_bps: float = -1.0,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` for the freshness one-sided
    trigger inside ``compute_quote_eligibility``.

    Fires when ``raw.reason`` contains ``freshness_one_sided`` (set
    by the kinematic gate when ``local_receipt_ms`` exceeds the
    one-sided threshold). Mirrors the pre-cutover eligibility-clamp
    direction.

    ``widen_bps`` sentinel ``-1.0`` falls back to
    ``max_half_spread_bps`` (gate-equivalent magnitude).
    """
    reason = raw.reason or ""
    if "freshness_one_sided" not in reason:
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    if raw.eligibility == QuoteEligibility.QUOTE_BUY_ONLY:
        # ASK suppressed → widen ask.
        return (0.0, effective)
    if raw.eligibility == QuoteEligibility.QUOTE_SELL_ONLY:
        # BID suppressed → widen bid.
        return (effective, 0.0)
    # HOLD_ALL fallthrough (e.g. freshness on both): widen both.
    if raw.eligibility == QuoteEligibility.HOLD_ALL:
        return (effective, effective)
    return (0.0, 0.0)


def recovery_cooldown_widening_bps(
    eff: QuoteEligibilityResult,
    *,
    max_half_spread_bps: float,
    widen_bps: float = -1.0,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` for the post-transition
    recovery cooldown clamp applied by :func:`apply_recovery_cooldown`.

    Fires when ``eff.in_cooldown`` is True. Widens the side(s) that
    match the recovery floor's one-sided direction.

    ``widen_bps`` sentinel ``-1.0`` falls back to
    ``max_half_spread_bps`` (gate-equivalent magnitude).
    """
    if not eff.in_cooldown:
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    if eff.eligibility == QuoteEligibility.QUOTE_BUY_ONLY:
        return (0.0, effective)
    if eff.eligibility == QuoteEligibility.QUOTE_SELL_ONLY:
        return (effective, 0.0)
    if eff.eligibility == QuoteEligibility.HOLD_ALL:
        return (effective, effective)
    return (0.0, 0.0)


def apply_recovery_cooldown(
    raw: QuoteEligibilityResult,
    *,
    now_mono: float,
    recovery_until_mono: float,
    recovery_floor: Optional[QuoteEligibility],
) -> QuoteEligibilityResult:
    """Clamp result to recovery_floor while now < recovery_until_mono."""
    if recovery_floor is None or recovery_until_mono <= 0.0:
        return raw
    if now_mono + 1e-12 >= recovery_until_mono:
        return raw
    clamped = more_restrictive(raw.eligibility, recovery_floor)
    return QuoteEligibilityResult(
        eligibility=clamped,
        reason=raw.reason + "|recovery_cooldown",
        seconds_since_last_public_book_update=raw.seconds_since_last_public_book_update,
        effective_staleness_ms=raw.effective_staleness_ms,
        market_data_gap_p95_ms=raw.market_data_gap_p95_ms,
        market_data_gap_median_ms=raw.market_data_gap_median_ms,
        mid_return_100ms_bps=raw.mid_return_100ms_bps,
        mid_return_250ms_bps=raw.mid_return_250ms_bps,
        mid_return_500ms_bps=raw.mid_return_500ms_bps,
        jump_100ms_bps=raw.jump_100ms_bps,
        jump_250ms_bps=raw.jump_250ms_bps,
        jump_500ms_bps=raw.jump_500ms_bps,
        in_cooldown=True,
        counter_tags=raw.counter_tags,
    )


def maybe_arm_recovery_cooldown(
    settings: Settings,
    *,
    last_effective: QuoteEligibility,
    new_raw: QuoteEligibility,
    now_mono: float,
    recovery_until_mono: float,
    recovery_floor: Optional[QuoteEligibility],
) -> tuple[float, Optional[QuoteEligibility]]:
    """
    When raw conditions improve vs last tick's effective eligibility, start or extend recovery.

    When conditions worsen, clear the recovery timer.
    """
    r_new, r_last = _rank(new_raw), _rank(last_effective)
    if r_new > r_last:
        return 0.0, None
    if r_new == r_last:
        return recovery_until_mono, recovery_floor
    # Improved (less restrictive raw than last tick's effective cap).
    #
    # Important: ``last_effective`` is the *post-clamp* value (apply_recovery_cooldown). If we
    # blindly key off it, we can re-arm on every healthy tick while effective remains clamped,
    # extending the cooldown forever. Only arm/extend when entering recovery, not while already
    # in the same active recovery floor.
    if (
        recovery_floor == last_effective
        and recovery_floor is not None
        and recovery_until_mono > 0.0
    ):
        return recovery_until_mono, recovery_floor
    if last_effective == QuoteEligibility.HOLD_ALL:
        until = now_mono + float(settings.quote_hold_cooldown_ms) / 1000.0
        return max(until, recovery_until_mono), last_effective
    if last_effective in (
        QuoteEligibility.QUOTE_BUY_ONLY,
        QuoteEligibility.QUOTE_SELL_ONLY,
    ):
        until = now_mono + float(settings.quote_one_sided_cooldown_ms) / 1000.0
        return max(until, recovery_until_mono), last_effective
    return recovery_until_mono, recovery_floor
