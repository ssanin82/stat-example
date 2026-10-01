from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

from app.config import Settings
from app.enums import Side
from app.models import Fill, ToxicitySnapshot


def _preferred_delayed_markout_bps(f: Fill) -> Optional[float]:
    """Prefer longer horizons when multiple are resolved (more stable signal)."""
    if f.markout_5s_bps is not None:
        return f.markout_5s_bps
    if f.markout_3s_bps is not None:
        return f.markout_3s_bps
    if f.markout_1s_bps is not None:
        return f.markout_1s_bps
    return None


class ToxicityEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._baseline_vol_bps: Optional[float] = None

    def set_baseline_vol(self, vol_bps: float) -> None:
        if self._baseline_vol_bps is None and vol_bps > 0:
            self._baseline_vol_bps = vol_bps

    def snapshot(
        self,
        mid: Optional[float],
        current_vol_bps: float,
        fills: Sequence[Fill],
        now: Optional[datetime] = None,
    ) -> ToxicitySnapshot:
        """Compute toxicity-engine outputs for the current tick.

        v1.5.197 — fills are FILTERED BY TIME-AGE before processing.
        Pre-v1.5.197 the engine consumed the count-based recent_fills
        deque (last 1000 fills regardless of age). When defensive
        gates suppressed trading, the deque didn't decay and stale
        adverse fills kept the toxicity score elevated permanently —
        a defensive-deadlock pattern documented in v1.5.195 snapshot
        post-mortem. v1.5.197 filters out fills older than
        ``TOXICITY_RECENT_FILLS_MAX_AGE_SECONDS`` (default 600s)
        BEFORE computing the engine output, so the toxicity score
        auto-decays to zero after 10 minutes of no fills regardless
        of how adverse the recent burst was.

        ``now`` is optional for backward compat with callers (notably
        tests) that don't pass it. When None, the engine falls back
        to the pre-v1.5.197 behaviour (process all provided fills).
        Production callers in bot.py pass ``self._clock.now_utc()``.
        """
        if not self._settings.toxicity_enabled:
            return ToxicitySnapshot(
                score=0.0,
                one_sided_fill_ratio=0.0,
                avg_adverse_markout_bps=0.0,
                vol_spike_ratio=1.0,
                hard_trigger=False,
                soft_trigger=False,
                delayed_markout_sample_count=0,
                adverse_uses_delayed_markouts=False,
            )

        # v1.5.197 — time-decay filter. Fills older than the
        # configured max-age get dropped BEFORE any windowing /
        # averaging downstream. Defensive: ``now`` may be None
        # (legacy callers / tests); in that case use no filter.
        fl_all = list(fills)
        max_age_s = float(
            getattr(
                self._settings,
                "toxicity_recent_fills_max_age_seconds",
                600.0,
            )
            or 0.0
        )
        if now is not None and max_age_s > 0.0:
            cutoff = now - timedelta(seconds=max_age_s)
            # Filter defensively — Fill.ts_fill may be naive on some
            # legacy paths. Normalize to UTC for comparison.
            def _is_fresh(f: Fill) -> bool:
                ts = f.ts_fill
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts >= cutoff
            fl = [f for f in fl_all if _is_fresh(f)]
        else:
            fl = fl_all

        one_sided = 0.0
        min_one_sided_fills = int(self._settings.toxicity_one_sided_min_fills)
        if len(fl) >= min_one_sided_fills:
            window = fl[:12]
            buys = sum(1 for x in window if x.side == Side.BUY)
            sells = len(window) - buys
            dominant = max(buys, sells)
            one_sided = dominant / len(window)

        adverse: list[float] = []
        # Per-side running markout — feeds the adverse-side pause decision.
        # Window matches the toxicity one: latest 20 fills.
        buy_markouts: list[float] = []
        sell_markouts: list[float] = []
        for f in fl[:20]:
            m = _preferred_delayed_markout_bps(f)
            if m is not None:
                adverse.append(m)
                if f.side == Side.BUY:
                    buy_markouts.append(m)
                else:
                    sell_markouts.append(m)

        delayed_count = len(adverse)
        adverse_from_delayed = delayed_count > 0
        avg_adv = sum(adverse) / len(adverse) if adverse else 0.0

        base = self._baseline_vol_bps or max(current_vol_bps, 1e-6)
        vol_ratio = current_vol_bps / base if base > 0 else 1.0

        score = 0.0
        if one_sided >= self._settings.toxicity_one_sided_fill_ratio:
            score += 0.35
        if adverse_from_delayed and avg_adv <= -self._settings.toxicity_markout_soft_bps:
            score += 0.35
        if vol_ratio >= 2.5:
            score += 0.2
        score = min(1.0, score)

        # v1.5.291 — min-fill gate on the MARKOUT hard path. The
        # markout average ``avg_adv`` is computed over ``delayed_count``
        # resolved-markout fills; averaging 1-2 fills is pure noise.
        # Require at least ``toxicity_markout_hard_min_fills`` samples
        # before the markout path can declare a hard trigger. Default 0
        # => gate disabled (legacy behaviour: any single adverse fill
        # past the threshold trips hard). The prod profile sets 4. The
        # SECOND clause (one-sided ratio) keeps its own ``len(fl) >= 8``
        # floor unchanged. See config.toxicity_markout_hard_min_fills for
        # the incident rationale (the residual_below_min_notional /
        # sf_fatigue tier-4 kill loop on a 2-fill window).
        markout_hard_min_fills = int(
            getattr(self._settings, "toxicity_markout_hard_min_fills", 0) or 0
        )
        markout_hard = (
            adverse_from_delayed
            and delayed_count >= markout_hard_min_fills
            and avg_adv <= -self._settings.toxicity_markout_hard_bps
        )
        one_sided_hard = one_sided >= 0.9 and len(fl) >= 8
        hard = markout_hard or one_sided_hard
        soft = score >= 0.45

        toxic_side: Optional[Side] = None
        if hard and fl:
            tail = fl[:8]
            buys = sum(1 for x in tail if x.side == Side.BUY)
            sells = len(tail) - buys
            if buys > sells * 2:
                toxic_side = Side.BUY
            elif sells > buys * 2:
                toxic_side = Side.SELL

        buy_avg = sum(buy_markouts) / len(buy_markouts) if buy_markouts else None
        sell_avg = sum(sell_markouts) / len(sell_markouts) if sell_markouts else None

        return ToxicitySnapshot(
            score=score,
            one_sided_fill_ratio=one_sided,
            avg_adverse_markout_bps=avg_adv,
            vol_spike_ratio=vol_ratio,
            hard_trigger=hard,
            soft_trigger=soft,
            toxic_side=toxic_side,
            delayed_markout_sample_count=delayed_count,
            adverse_uses_delayed_markouts=adverse_from_delayed,
            buy_side_avg_markout_bps=buy_avg,
            sell_side_avg_markout_bps=sell_avg,
            buy_side_fill_count=len(buy_markouts),
            sell_side_fill_count=len(sell_markouts),
        )
