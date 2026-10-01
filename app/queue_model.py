"""v1.5.209 — Phase 8B Queue-position-aware sizing & repricing.

Background
----------

Stoikov, Avellaneda and Guéant all converge on the same point: a
passive market-maker that ignores queue position pays a 20-40 %
"queue tax" on top of theoretical edge. The bot today quotes at the
touch with no model of how much size is already resting ahead of it.
When the bot joins a 50-lot bid stack with 49 lots already ahead,
its fill probability is dramatically lower than the 1-lot scenario
— but the price + edge calculation is identical, so the bot leaves
size + reservation unchanged in both cases.

This module provides the queue model. It is intentionally simple:
the bot is observing only L1 (best bid/ask + sizes), and the day
plan defers L2-aware queue estimation. The estimate uses public
print volume per side as a proxy for arrival rate and ``own_size /
total_l1_size`` as a position ratio.

What the helpers do
-------------------

``estimate_queue_position_ratio`` — what fraction of the inside
queue is ahead of (or equal to) the bot's own resting order, on a
given side. Range: ``[0, 1]``. ``0.0`` = nothing ahead (bot is the
only resting order), ``1.0`` = bot is the last one in.

``expected_wait_seconds`` — given queue-ahead size + arrival rate,
how long until the bot's order would be at the front. Used as a
diagnostic only in v1.5.209; not in the hot path's gating logic.

``queue_aware_size_multiplier`` — clamp(min, max, 1 - decay *
ratio). At decay=0.7, min=0.3: ratio=0 → mult=1.0 (no shrink),
ratio=1.0 → mult=0.3 (70 % shrink). Pure function.

``ArrivalRateEwma`` — bounded-memory EWMA on trade-print volume per
side. Updated from the public-WS handler thread (one ``record_trade``
call per print), read by the quote loop. Same threading pattern as
``OFIAccumulator``.

Default-off in v1.5.209
-----------------------

Two independent flags:

* ``QUEUE_AWARE_SIZING_ENABLED=false`` — enables the size multiplier
  in ``compute_quote_decision``. When off, the multiplier is bypassed
  entirely and per-side sizing matches pre-v1.5.209 behaviour.

* ``QUEUE_AWARE_INSIDE_POST_ENABLED=false`` — enables the optional
  inside-the-spread post: when the bot would otherwise quote at the
  touch but the queue ahead is too long, it posts one tick INSIDE
  the spread (touch + 1 tick) to skip the queue. This is a stronger
  intervention than sizing — it affects price, not just size — and
  ships off by default.

These flags can be toggled independently: operator can A/B-test
sizing alone, then sizing + inside-post.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple


def estimate_queue_position_ratio(
    *,
    own_resting_size: float,
    inside_total_size: float,
) -> Optional[float]:
    """Estimate what fraction of the inside queue is ahead of own order.

    Bot's-own resting size on this side is ``own_resting_size``;
    the inside (L1) total size on this side is ``inside_total_size``.
    Since L1-only observability hides queue ordering, we use a simple
    upper-bound proxy: assume the bot is the LAST one in the queue
    (worst case). Then queue_ahead = inside_total_size − own_resting_size
    and ratio = queue_ahead / inside_total_size.

    Returns ``None`` if ``inside_total_size`` is non-positive (no
    visible queue to assess) or inputs are non-finite.

    The "last in queue" assumption is intentionally conservative.
    Once L2-aware estimation lands (Phase 8B.7 follow-up), this
    helper gets replaced with an L2 walk — but the API stays the
    same.
    """
    for v in (own_resting_size, inside_total_size):
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return None
    own = float(own_resting_size)
    total = float(inside_total_size)
    if total <= 0:
        return None
    if own < 0:
        own = 0.0
    if own > total:
        own = total
    queue_ahead = total - own
    return max(0.0, min(1.0, queue_ahead / total))


def expected_wait_seconds(
    *,
    queue_ahead_size: float,
    arrival_rate_per_sec: float,
) -> Optional[float]:
    """Estimate seconds-to-front-of-queue.

    ``queue_ahead_size`` = how much volume is in front of own order.
    ``arrival_rate_per_sec`` = trade-print volume EWMA on this side
    (units of "shares of aggressor flow consumed from this side per
    second"). Both must be finite + positive for a meaningful answer.

    Returns ``None`` when the rate is too low to compute (would
    divide by ~0). Caller treats as "queue effectively frozen, don't
    bother quoting".
    """
    for v in (queue_ahead_size, arrival_rate_per_sec):
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            return None
    qa = float(queue_ahead_size)
    rate = float(arrival_rate_per_sec)
    if qa <= 0:
        return 0.0
    if rate <= 1e-9:
        return None
    return qa / rate


def queue_aware_size_multiplier(
    *,
    queue_position_ratio: Optional[float],
    floor: float = 0.3,
    decay: float = 0.7,
) -> float:
    """Pure mapping from ratio → size multiplier.

    Formula: ``clamp(floor, 1.0, 1.0 − decay × ratio)``.

    * ratio = 0.0 (no one ahead)  → mult = 1.0
    * ratio = 0.5                 → mult = 1.0 − 0.7×0.5 = 0.65
    * ratio = 1.0 (whole queue)   → mult = 0.3 (floor)
    * ratio = None                → mult = 1.0 (no signal, no shrink)

    ``floor`` keeps the size from collapsing below venue self-heal
    minimums; ``decay`` controls aggressiveness of the shrink. Both
    are env-tunable (``QUEUE_AWARE_SIZE_FLOOR`` /
    ``QUEUE_AWARE_SIZE_DECAY``) but the defaults match the day-plan
    spec.
    """
    if queue_position_ratio is None:
        return 1.0
    try:
        r = float(queue_position_ratio)
    except (TypeError, ValueError):
        return 1.0
    if r != r:  # NaN
        return 1.0
    if r < 0:
        r = 0.0
    elif r > 1:
        r = 1.0
    return max(float(floor), min(1.0, 1.0 - float(decay) * r))


def should_post_inside_spread(
    *,
    queue_position_ratio: Optional[float],
    threshold: float,
    enabled: bool,
) -> bool:
    """Inside-spread gate.

    Returns ``True`` when the operator has enabled inside-post AND
    the queue ratio exceeds the threshold. ``False`` in every other
    case (warmup / dormant / queue not too bad).
    """
    if not enabled:
        return False
    if queue_position_ratio is None:
        return False
    try:
        r = float(queue_position_ratio)
    except (TypeError, ValueError):
        return False
    if r != r:  # NaN
        return False
    return r >= float(threshold)


@dataclass(slots=True)
class ArrivalRateEwma:
    """60s rolling EWMA of trade-print volume per side.

    Updated by the public-WS trade-print handler (one
    ``record_trade(side, qty, ts)`` per print); read by the quote
    loop. Halflife defaults to 30 s — covers the 1-2 min print
    bursts typical on TON without smoothing across regime changes.

    Per-side EWMAs are kept independent: a trade that lifted the
    offer consumes ask-side supply; a trade that hit the bid
    consumes bid-side supply. Same convention as the existing
    flow-score TFI accumulator.
    """

    halflife_seconds: float = 30.0

    # Last-update timestamps + EWMA values (per side).
    _bid_last_ts_mono: Optional[float] = None
    _ask_last_ts_mono: Optional[float] = None
    bid_arrival_rate_per_sec: Optional[float] = None
    ask_arrival_rate_per_sec: Optional[float] = None

    update_count_bid: int = 0
    update_count_ask: int = 0

    def record_trade(
        self,
        *,
        side: str,  # "bid" (sell-to-bid) or "ask" (buy-from-ask)
        qty: float,
        now_mono_seconds: Optional[float] = None,
    ) -> None:
        """Record a single trade print on the given side."""
        if not isinstance(qty, (int, float)) or not math.isfinite(float(qty)) or float(qty) <= 0:
            return
        if side not in ("bid", "ask"):
            return
        ts = float(now_mono_seconds) if now_mono_seconds is not None else time.monotonic()

        if side == "bid":
            last_ts = self._bid_last_ts_mono
            prev_rate = self.bid_arrival_rate_per_sec
        else:
            last_ts = self._ask_last_ts_mono
            prev_rate = self.ask_arrival_rate_per_sec

        if last_ts is None or prev_rate is None:
            # First print on this side. Seed the rate as qty / 1s
            # (conservative cold-start); halflife will smooth it out.
            new_rate = float(qty)
        else:
            dt = max(1e-6, ts - last_ts)
            instantaneous_rate = float(qty) / dt
            if self.halflife_seconds <= 0:
                new_rate = instantaneous_rate
            else:
                alpha = 1.0 - 0.5 ** (dt / float(self.halflife_seconds))
                new_rate = float(prev_rate) + alpha * (instantaneous_rate - float(prev_rate))

        if side == "bid":
            self._bid_last_ts_mono = ts
            self.bid_arrival_rate_per_sec = new_rate
            self.update_count_bid += 1
        else:
            self._ask_last_ts_mono = ts
            self.ask_arrival_rate_per_sec = new_rate
            self.update_count_ask += 1

    def decay_to(self, now_mono_seconds: float) -> None:
        """Decay both side-rates toward 0 if no new prints arrived.

        Without this call, a quiet 30s window leaves the rate stuck
        at its last value, which over-estimates current arrival flow.
        Bot's quote loop should call this once per tick before
        reading the rates. Pure decay: applies the halflife to the
        gap (now − last_ts) on each side independently.
        """
        if self.halflife_seconds <= 0:
            return
        for side in ("bid", "ask"):
            last_ts = self._bid_last_ts_mono if side == "bid" else self._ask_last_ts_mono
            prev_rate = (
                self.bid_arrival_rate_per_sec if side == "bid"
                else self.ask_arrival_rate_per_sec
            )
            if last_ts is None or prev_rate is None:
                continue
            dt = float(now_mono_seconds) - last_ts
            if dt <= 0:
                continue
            decay_factor = 0.5 ** (dt / float(self.halflife_seconds))
            new_rate = float(prev_rate) * decay_factor
            if side == "bid":
                self.bid_arrival_rate_per_sec = new_rate
                self._bid_last_ts_mono = float(now_mono_seconds)
            else:
                self.ask_arrival_rate_per_sec = new_rate
                self._ask_last_ts_mono = float(now_mono_seconds)

    def snapshot_dict(self) -> dict:
        return {
            "halflife_seconds": float(self.halflife_seconds),
            "bid_arrival_rate_per_sec": (
                float(self.bid_arrival_rate_per_sec)
                if self.bid_arrival_rate_per_sec is not None else None
            ),
            "ask_arrival_rate_per_sec": (
                float(self.ask_arrival_rate_per_sec)
                if self.ask_arrival_rate_per_sec is not None else None
            ),
            "update_count_bid": int(self.update_count_bid),
            "update_count_ask": int(self.update_count_ask),
        }
