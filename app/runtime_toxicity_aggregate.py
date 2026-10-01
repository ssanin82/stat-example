"""
Session-scoped rolling toxicity summary from recent fills (observational only).

``recent_fills`` is newest-first (``deque.appendleft``). Means use only fills where the
corresponding markout column is non-null. ``toxicity_score`` is null unless there are
enough fills and at least one resolved delayed markout in the window — see module constants.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from app.enums import Side
from app.models import Fill

# Score needs a minimum window size; below this, toxicity_score stays null.
# Lowered from 4 → 2: observed in ``tmp/snap_20260417_183547`` that a session
# with 3 fills — all three adversely marked out (-1.25 bps mean 5s markout) —
# showed ``toxicity_score=null`` to the operator. The underlying signal was
# unambiguous (3/3 adverse, one-sided ratio 0.67) but the 4-fill gate hid it.
# This is an observational metric for /toxicity/current, not a strategy
# driver; the quote engine's ToxicityEngine has its own independent threshold.
# Two is the lowest statistically meaningful sample — "both fills adverse"
# carries real information; one fill does not.
MIN_FILLS_FOR_TOXICITY_SCORE = 2
# Maps average adverse markout (bps, negative = bad) into ~[0, 1] before blending.
ADVERSARY_MARKOUT_SCALE_BPS = 15.0


def _preferred_markout_bps(f: Fill) -> Optional[float]:
    """Prefer longer horizons when present (same ordering as strategy toxicity)."""
    if f.markout_5s_bps is not None:
        return float(f.markout_5s_bps)
    if f.markout_3s_bps is not None:
        return float(f.markout_3s_bps)
    if f.markout_1s_bps is not None:
        return float(f.markout_1s_bps)
    return None


def _mean(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / len(vals)


def build_runtime_toxicity_summary(
    fills_newest_first: Sequence[Fill],
    *,
    window: int,
    session_id: str,
) -> dict[str, Any]:
    """
    Build API payload for GET /toxicity/current.

    Median/p95 are not computed here — only simple means over non-null markout columns.
    """
    w = max(0, int(window))
    buf = list(fills_newest_first[:w]) if w else []
    n = len(buf)
    buys = sum(1 for f in buf if f.side == Side.BUY)
    sells = n - buys
    total = n

    m1 = _mean([float(f.markout_1s_bps) for f in buf if f.markout_1s_bps is not None])
    m3 = _mean([float(f.markout_3s_bps) for f in buf if f.markout_3s_bps is not None])
    m5 = _mean([float(f.markout_5s_bps) for f in buf if f.markout_5s_bps is not None])

    os_ratio: Optional[float] = None
    if total > 0:
        os_ratio = max(buys, sells) / total

    last_ts: Optional[str] = None
    if buf:
        last_ts = buf[0].ts_fill.isoformat()

    prefs: list[float] = []
    for f in buf:
        p = _preferred_markout_bps(f)
        if p is not None:
            prefs.append(p)

    score: Optional[float] = None
    if n >= MIN_FILLS_FOR_TOXICITY_SCORE and prefs and os_ratio is not None:
        avg_pref = sum(prefs) / len(prefs)
        # Negative markout bps = adverse → positive severity
        adverse_mag = max(0.0, -avg_pref)
        markout_comp = min(1.0, adverse_mag / ADVERSARY_MARKOUT_SCALE_BPS)
        # one_sided 0.5 (balanced) → 0, 1.0 → 1
        onesided_comp = min(1.0, max(0.0, (os_ratio - 0.5) / 0.5))
        raw = 0.55 * markout_comp + 0.45 * onesided_comp
        score = round(min(1.0, max(0.0, raw)), 4)

    return {
        "session_id": session_id,
        "fill_count_in_window": n,
        "mean_markout_1s_bps": None if m1 is None else round(m1, 4),
        "mean_markout_3s_bps": None if m3 is None else round(m3, 4),
        "mean_markout_5s_bps": None if m5 is None else round(m5, 4),
        "buy_fill_count": buys,
        "sell_fill_count": sells,
        "one_sided_fill_ratio": None if os_ratio is None else round(os_ratio, 4),
        "toxicity_score": score,
        "last_fill_ts": last_ts,
    }
