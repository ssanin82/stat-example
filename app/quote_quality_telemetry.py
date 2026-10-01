"""
Session-scoped quote / execution quality rollups for API observability (not strategy input).

Rolling window of recent quote cycles; counters are monotonic for the process lifetime.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Optional, Sequence

from app.enums import Side
from app.models import Fill

from app import clock as _clock

# Soft widen signal: require enough two-sided spread samples before reacting to one-tick pressure.
_SPREAD_WIDEN_MIN_TWO_SIDED_SPREAD_SAMPLES = 25
_SPREAD_WIDEN_PCT_ONE_TICK = 58.0
_DELAYED_MARKOUT_MIN_SAMPLES = 6
_DELAYED_MARKOUT_MEAN_BPS = -1.5


def _median(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return (s[m - 1] + s[m]) / 2.0


def build_delayed_markout_summary(
    fills_newest_first: Sequence[Fill],
    *,
    window: int,
) -> dict[str, Any]:
    """Summarize delayed markouts over the same fill window as runtime toxicity."""
    w = max(0, int(window))
    buf = list(fills_newest_first[:w]) if w else []
    prefs: list[float] = []
    for f in buf:
        v = f.markout_5s_bps
        if v is None:
            v = f.markout_3s_bps
        if v is None:
            v = f.markout_1s_bps
        if v is not None:
            prefs.append(float(v))
    adverse = sum(1 for p in prefs if p < 0)
    favorable = sum(1 for p in prefs if p > 0)
    return {
        "delayed_markout_sample_count": len(prefs),
        "mean_delayed_markout_bps": None
        if not prefs
        else round(sum(prefs) / len(prefs), 4),
        "median_delayed_markout_bps": None
        if not prefs
        else round(float(_median(prefs)), 4),
        "adverse_delayed_markout_count": adverse,
        "favorable_delayed_markout_count": favorable,
    }


def build_net_edge_summary(
    fills_newest_first: Sequence[Fill],
    *,
    window: int,
) -> dict[str, Any]:
    """TODO-003: rolling net-edge-after-fees summary.

    For each of the last ``window`` fills, compute:

    * ``markout_bps`` — the same preference order as
      ``build_delayed_markout_summary`` (5s → 3s → 1s).
    * ``fee_bps_per_fill`` — ``fee_usd / notional * 10000``.

    Returns:

    * ``mean_markout_bps`` — gross edge proxy (signed, positive = favourable).
    * ``mean_fee_bps_per_fill`` — single-side fee drag.
    * ``mean_round_trip_fee_bps`` — 2 × single-side (entry + exit assumed
      symmetric — close enough for live observability; precise per-pair
      attribution lives in offline reporting).
    * ``net_edge_bps = mean_markout_bps - mean_round_trip_fee_bps`` —
      the operator's "are we monetising the configured edge?" number.

    All fields ``None`` when sample count is too low (sample_count < 5).
    Net-edge is meaningful only with reasonable averaging.
    """
    w = max(0, int(window))
    buf = list(fills_newest_first[:w]) if w else []
    markouts: list[float] = []
    fees_bps: list[float] = []
    for f in buf:
        v = f.markout_5s_bps
        if v is None:
            v = f.markout_3s_bps
        if v is None:
            v = f.markout_1s_bps
        if v is None:
            continue
        if not isinstance(f.notional, (int, float)) or float(f.notional) <= 0:
            continue
        # SIGNED fee bps: positive = cost, negative = rebate received.
        # The downstream net-edge calculation (mean_m - mean_rt) then
        # correctly CREDITS rebates instead of debiting them. Pre-
        # 2026-05-05 this used abs() which made rebates look like fees
        # in net-edge math. See HLFillRaw docstring for the convention.
        fee_bps = float(f.fee) / float(f.notional) * 10_000.0
        markouts.append(float(v))
        fees_bps.append(fee_bps)

    n = len(markouts)
    mean_m = None if n == 0 else sum(markouts) / n
    mean_f = None if n == 0 else sum(fees_bps) / n
    mean_rt = None if mean_f is None else 2.0 * mean_f
    net_edge = (
        None
        if mean_m is None or mean_rt is None or n < 5
        else mean_m - mean_rt
    )
    return {
        "net_edge_sample_count": n,
        "mean_markout_bps": None if mean_m is None else round(mean_m, 4),
        "mean_fee_bps_per_fill": None if mean_f is None else round(mean_f, 4),
        "mean_round_trip_fee_bps": None if mean_rt is None else round(mean_rt, 4),
        "net_edge_bps": None if net_edge is None else round(net_edge, 4),
    }


class QuoteQualityRollup:
    """
    Thread-safe updates assumed from the bot / OrderManager thread only (same as most BotState).
    """

    def __init__(self, *, window_samples: int, order_lifetime_max: int = 200) -> None:
        self._window = max(1, int(window_samples))
        self._spreads_bps: deque[Optional[float]] = deque(maxlen=self._window)
        self._one_tick: deque[bool] = deque(maxlen=self._window)
        self._two_sided_effective: deque[bool] = deque(maxlen=self._window)
        self._intended_two_sided: deque[bool] = deque(maxlen=self._window)
        self._post_only_cross_rejections: int = 0
        self._quote_reprice_required: int = 0
        self._order_lifetime_s: deque[float] = deque(maxlen=max(1, int(order_lifetime_max)))
        self._spread_capture_session_usd: float = 0.0
        # Session-monotonic counter of quote-cycle suppression reasons. A cycle may
        # increment multiple keys if several suppressors fire simultaneously (e.g.
        # freshness + inventory bias). Used for diagnosing why the bot spent time
        # one-sided or held without relying on event log scrapes.
        self._suppression_counts: dict[str, int] = {}
        # Phase 5 ("healthy but not trading" indicator): 60 s rolling
        # suppression-rate bookkeeping. Each ``record_quote_cycle`` call
        # appends a monotonic timestamp; ``record_suppression`` appends
        # one too IF the cycle was suppressed at all (any reason fired).
        # ``suppression_rate_60s()`` returns the ratio over the last
        # 60 seconds. The dashboard chip uses this together with
        # execution_idle_seconds to detect "engine alive but execution
        # stalled" before the deadlock watchdog fires (snapshot
        # 260507114312 spent 4.5 min in that state with no operator
        # signal). Bounded deque keeps memory flat; entries older than
        # 60 s are discarded on read.
        self._cycle_ts_mono: deque[float] = deque(maxlen=2048)
        self._suppressed_cycle_ts_mono: deque[float] = deque(maxlen=2048)
        self._last_cycle_was_suppressed: bool = False
        self._suppression_rate_window_seconds: float = 60.0

    def record_quote_cycle(
        self,
        *,
        quoted_spread_bps: Optional[float],
        one_tick_wide: bool,
        two_sided_effective: bool,
        intended_two_sided: bool,
    ) -> None:
        self._spreads_bps.append(quoted_spread_bps)
        self._one_tick.append(one_tick_wide)
        self._two_sided_effective.append(two_sided_effective)
        self._intended_two_sided.append(intended_two_sided)
        # Phase 5: 60 s suppression-rate bookkeeping. Stamp this cycle.
        # If any suppression was recorded for THIS cycle (caller called
        # ``record_suppression`` since the last ``record_quote_cycle``),
        # the cycle counts as suppressed. Reset the per-cycle latch
        # whether or not a suppression fired.
        now_m = _clock.monotonic()
        self._cycle_ts_mono.append(now_m)
        if self._last_cycle_was_suppressed:
            self._suppressed_cycle_ts_mono.append(now_m)
        self._last_cycle_was_suppressed = False

    def record_suppression(self, reason: str) -> None:
        if not reason:
            return
        self._suppression_counts[reason] = self._suppression_counts.get(reason, 0) + 1
        # Phase 5: any suppressor firing in this cycle marks the cycle
        # as suppressed for the rolling-rate calculation. Multiple
        # reasons in one cycle still count as one suppressed cycle.
        self._last_cycle_was_suppressed = True

    def suppression_rate_60s(self) -> Optional[float]:
        """Fraction of the last 60 s of quote cycles that were
        suppressed. Returns ``None`` when there are no cycles in the
        window (bot just started, or no quote loop activity).

        Used by the Phase 5 dashboard indicator to detect the
        "engine alive but execution stalled" pattern. A cycle counts
        as suppressed if any suppressor fired during its execution
        (multiple reasons in the same cycle still count once).
        """
        now_m = _clock.monotonic()
        cutoff = now_m - float(self._suppression_rate_window_seconds)
        # Discard timestamps older than the window from the LEFT of
        # both deques. O(N) worst-case at first call after a long
        # idle, but typically constant per call (deque heads turn
        # over fast at 2 Hz quote loop).
        while self._cycle_ts_mono and self._cycle_ts_mono[0] < cutoff:
            self._cycle_ts_mono.popleft()
        while (
            self._suppressed_cycle_ts_mono
            and self._suppressed_cycle_ts_mono[0] < cutoff
        ):
            self._suppressed_cycle_ts_mono.popleft()
        total = len(self._cycle_ts_mono)
        if total <= 0:
            return None
        suppressed = len(self._suppressed_cycle_ts_mono)
        if suppressed > total:
            # Defensive — same cycle should at most count once. Clamp.
            suppressed = total
        return float(suppressed) / float(total)

    def note_post_only_cross_rejection(self) -> None:
        self._post_only_cross_rejections += 1

    @property
    def post_only_cross_rejection_count(self) -> int:
        """Public, read-only view of the session counter so external
        controllers (``JoinDepthController``) can sample without
        reaching into a private attribute."""
        return self._post_only_cross_rejections

    def note_quote_reprice_required(self) -> None:
        self._quote_reprice_required += 1

    def note_order_lifetime_seconds(self, seconds: float) -> None:
        if seconds >= 0 and seconds < 86400.0 * 7:
            self._order_lifetime_s.append(float(seconds))

    def note_fill_spread_capture_usd(self, f: Fill) -> None:
        m = f.mid_at_fill
        if m is None or not isinstance(m, (int, float)) or m <= 0:
            return
        px = float(f.price)
        sz = abs(float(f.size))
        if sz <= 0:
            return
        if f.side == Side.BUY:
            self._spread_capture_session_usd += (float(m) - px) * sz
        else:
            self._spread_capture_session_usd += (px - float(m)) * sz

    def spread_widen_signal(
        self,
        *,
        fills_for_markout: Sequence[Fill],
        markout_window: int,
        min_quote_cycles: int,
    ) -> bool:
        """
        True when quote_quality suggests briefly widening via spread_floor overlay
        (stuck one-tick two-sided or clearly adverse delayed markout).
        """
        if len(self._spreads_bps) < max(1, int(min_quote_cycles)):
            return False
        dm = build_delayed_markout_summary(fills_for_markout, window=markout_window)
        mean_m = dm.get("mean_delayed_markout_bps")
        n_m = int(dm.get("delayed_markout_sample_count") or 0)
        if (
            n_m >= _DELAYED_MARKOUT_MIN_SAMPLES
            and mean_m is not None
            and float(mean_m) <= _DELAYED_MARKOUT_MEAN_BPS
        ):
            return True
        sp_seq = list(self._spreads_bps)
        tk_seq = list(self._one_tick)
        n_sp = sum(1 for s in sp_seq if s is not None)
        if n_sp < _SPREAD_WIDEN_MIN_TWO_SIDED_SPREAD_SAMPLES:
            return False
        one_tick_hits = sum(1 for s, t in zip(sp_seq, tk_seq) if s is not None and t)
        pct = 100.0 * one_tick_hits / n_sp
        return pct + 1e-9 >= _SPREAD_WIDEN_PCT_ONE_TICK

    def to_dict(
        self,
        *,
        realized_pnl_usd: float,
        fills_for_markout: Sequence[Fill],
        markout_window: int,
    ) -> dict[str, Any]:
        spreads = [x for x in self._spreads_bps if x is not None]
        n = len(self._spreads_bps)
        n_sp = len(spreads)
        sp_seq = list(self._spreads_bps)
        tk_seq = list(self._one_tick)
        one_tick_hits = sum(1 for s, t in zip(sp_seq, tk_seq) if s is not None and t)
        pct_one_tick = None if n_sp == 0 else round(100.0 * one_tick_hits / n_sp, 2)

        eff_two = sum(1 for x in self._two_sided_effective if x)
        int_two = sum(1 for x in self._intended_two_sided if x)
        pct_two_sided_effective = None if n == 0 else round(100.0 * eff_two / n, 2)
        pct_one_sided_effective = (
            None if n == 0 else round(100.0 * (n - eff_two) / n, 2)
        )
        pct_intended_two_sided = None if n == 0 else round(100.0 * int_two / n, 2)

        lifetimes = list(self._order_lifetime_s)
        avg_life = None if not lifetimes else sum(lifetimes) / len(lifetimes)

        dm = build_delayed_markout_summary(
            fills_for_markout, window=markout_window
        )

        raw_cap = float(self._spread_capture_session_usd)
        # Mid-at-fill edge is a gross proxy; when delayed markouts are clearly adverse it can
        # stay positive while fills lose to subsequent movement. Cap at zero in that regime so
        # the metric is not systematically misleading (does not revalue inventory — observational only).
        mean_dm = dm.get("mean_delayed_markout_bps")
        n_dm = int(dm.get("delayed_markout_sample_count") or 0)
        spread_capture_usd = raw_cap
        if (
            n_dm >= 8
            and mean_dm is not None
            and float(mean_dm) <= -2.0
        ):
            spread_capture_usd = min(raw_cap, 0.0)

        return {
            "quote_cycle_samples_in_window": n,
            "avg_quoted_spread_bps": None
            if not spreads
            else round(sum(spreads) / len(spreads), 4),
            "median_quoted_spread_bps": None
            if not spreads
            else round(float(_median(spreads)), 4),
            "pct_time_one_tick_wide_when_two_sided": pct_one_tick,
            "pct_time_two_sided_effective": pct_two_sided_effective,
            "pct_time_one_sided_effective": pct_one_sided_effective,
            "pct_time_intended_two_sided": pct_intended_two_sided,
            "post_only_cross_rejection_count_session": self._post_only_cross_rejections,
            "quote_reprice_required_count_session": self._quote_reprice_required,
            "avg_passive_order_lifetime_seconds": None
            if avg_life is None
            else round(avg_life, 3),
            "passive_order_lifetime_samples": len(lifetimes),
            "realized_pnl_usd": round(float(realized_pnl_usd), 6),
            "estimated_gross_spread_capture_usd_session": round(spread_capture_usd, 6),
            "delayed_markout_summary": dm,
            "suppression_reason_counts_session": dict(self._suppression_counts),
        }
