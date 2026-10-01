"""v1.4.82 wedge-elimination-cleanup Phase 3B — PositionStore.

Owns the bot's position-related state:

* ``PositionSnapshot`` — symbol, qty, avg entry, mark, notional, uPnL
* Session-scoped fill accounting baselines
* Position-cap helpers

Phase 3B is additive — the store shares state with ``BotState`` for
the migration window. Phase 3D moves ownership entirely.

The store provides a focused API for callers that want to reason
about position WITHOUT pulling in the rest of the god-class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models import PositionSnapshot

if TYPE_CHECKING:
    from app.state import BotState


class PositionStore:
    """Thin facade over ``BotState.position`` and the per-session
    fill-accounting fields.

    Read API is the most common use. Write paths
    (``apply_fill`` / ``set_position_from_venue``) stay on
    ``BotState`` until Phase 3D — they touch multiple subsystems
    (PnL tracker, fill-burst counter, etc.) and re-housing them
    requires coordinating those side effects.
    """

    def __init__(self, state: "BotState") -> None:
        self._state = state

    @property
    def snapshot(self) -> PositionSnapshot:
        """Current position snapshot. Same object as
        ``state.position`` (not a copy) — callers MUST treat as
        read-only.
        """
        return self._state.position

    def qty(self) -> float:
        """Signed position quantity (contracts)."""
        try:
            return float(self._state.position.position_qty)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def notional_abs_usd(self) -> float:
        """Absolute position notional in USD."""
        try:
            return float(self._state.position.position_notional)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def unrealized_pnl_usd(self) -> float:
        try:
            return float(self._state.position.unrealized_pnl_usd)
        except (AttributeError, TypeError, ValueError):
            return 0.0

    def is_flat(self, eps: float = 1e-10) -> bool:
        """True when position is at or below the configured epsilon."""
        return abs(self.qty()) <= eps

    def reducing_side(self) -> str | None:
        """Returns ``"BUY"`` when SHORT (BUY reduces), ``"SELL"``
        when LONG (SELL reduces), or ``None`` when flat.
        """
        q = self.qty()
        if q > 0:
            return "SELL"
        if q < 0:
            return "BUY"
        return None

    def headroom_long(self, max_abs_position: float) -> float:
        """Remaining capacity to grow LONG before hitting the cap."""
        return max(0.0, float(max_abs_position) - self.qty())

    def headroom_short(self, max_abs_position: float) -> float:
        """Remaining capacity to grow SHORT before hitting the cap."""
        return max(0.0, float(max_abs_position) + self.qty())
