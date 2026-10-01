"""
Rolling top-of-book memory for honest fill-time references (not exchange co-timestamped).

Match uses local snapshot timestamps from the bot refresh loop; alignment with the
exchange fill clock is best-effort only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence


@dataclass(frozen=True)
class TopOfBookRow:
    ts: datetime
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid: Optional[float]
    # Top-of-book sizes captured alongside the prices (added 2026-05-10
    # for analysis-day instrumentation N3 — queue-imbalance / depth-
    # conditional markout decomposition). Optional because some venue
    # adapters publish only price (not size) on partial / book-recovery
    # events. ``None`` is forward-compatible: legacy snapshots that
    # don't carry size leave the column NULL on the fills table.
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def snapshot_fullness(bid: Optional[float], ask: Optional[float], mid: Optional[float]) -> str:
    if bid and ask and bid > 0 and ask > 0:
        return "full"
    if mid and mid > 0:
        return "mid_only"
    return "unknown"


def match_top_of_book(
    rows: Sequence[TopOfBookRow],
    ts_fill: datetime,
    max_skew: timedelta,
) -> tuple[Optional[TopOfBookRow], str]:
    """
    Match a book row to a fill time with a hard recency bound.

    1. Prefer the latest snapshot at or before ts_fill; use it only if
       (ts_fill - row.ts) <= max_skew.
    2. Otherwise, if no valid prior, consider the earliest snapshot at or after
       ts_fill; use it only if (row.ts - ts_fill) <= max_skew.
    3. If neither is within max_skew, return missing_reference (no far-away rows).
    """
    if not rows:
        return None, "missing_reference"
    if max_skew < timedelta(0):
        raise ValueError("max_skew must be non-negative")
    t = _aware(ts_fill)
    prior: Optional[TopOfBookRow] = None
    for row in reversed(rows):
        if _aware(row.ts) <= t:
            prior = row
            break
    if prior is not None:
        delta_prior = t - _aware(prior.ts)
        if delta_prior <= max_skew:
            return prior, "exact_or_prior"
    for row in rows:
        tr = _aware(row.ts)
        if tr >= t:
            delta_after = tr - t
            if delta_after <= max_skew:
                return row, "approximate_after"
            break
    return None, "missing_reference"
