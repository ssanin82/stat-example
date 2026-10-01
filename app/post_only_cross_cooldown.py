"""Phase 2K.11 — favorable-exit predicate for the
``post_only_cross_cooldown`` (v1.4.160).

Background. When a post-only order is rejected by the venue because
it would cross (price ≥ best_ask for a BUY, or ≤ best_bid for a
SELL), ``OrderManager`` arms a short side-suppression cooldown
(``POST_ONLY_CROSS_COOLDOWN_SECONDS``, default 2.5 s). The intent
is to avoid re-placing into the same micro-state and getting
rejected again.

The 2.5 s blanket is conservative: as soon as the touch on the
relevant side has moved by ≥ 1 tick AWAY from the rejected price, a
re-place cannot cross. Phase 2K.11 adds that early-clear check so
the cooldown isn't burning quote-time after the market has already
resolved the conflict.

This module is a small pure helper used by ``OrderManager``. The
state machine itself lives in execution.py — we just expose the
predicate as a unit-testable function.

The cooldown is the shortest of the Phase 2K family (2.5 s), so the
cumulative impact is modest — but the favorable-exit pattern is now
applied consistently across every gate.
"""

from __future__ import annotations

from typing import Optional

from app.enums import Side

_EPS = 1e-12


def touch_moved_away_from_rejected_price(
    *,
    side: Side,
    rejected_price: Optional[float],
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick_size: Optional[float],
    tick_multiplier: float = 1.0,
) -> bool:
    """True when the touch on the conflicting side has moved at
    least ``tick_multiplier × tick_size`` AWAY from ``rejected_price``,
    so a re-place at the original price wouldn't cross again.

    Semantics by side (recall: post-only BUY at P crosses when
    P ≥ best_ask; post-only SELL at P crosses when P ≤ best_bid):

    * **BUY** (rejected buy): the conflict is with best_ask. Clear
      when ``best_ask >= rejected_price + tick_size × tick_multiplier``.
    * **SELL** (rejected sell): the conflict is with best_bid. Clear
      when ``best_bid <= rejected_price - tick_size × tick_multiplier``.

    Returns ``False`` (= don't clear) for any missing input — the
    timer ceiling continues to govern.
    """
    if rejected_price is None or tick_size is None or tick_size <= 0:
        return False
    if tick_multiplier <= 0:
        return False
    margin = float(tick_size) * float(tick_multiplier)
    if side == Side.BUY:
        if best_ask is None:
            return False
        return float(best_ask) >= float(rejected_price) + margin - _EPS
    if side == Side.SELL:
        if best_bid is None:
            return False
        return float(best_bid) <= float(rejected_price) - margin + _EPS
    return False
