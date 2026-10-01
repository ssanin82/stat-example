"""v1.4.82 wedge-elimination-cleanup Phase 3B — MarketStore.

Owns the bot's view of market data:

* ``BestBidAsk`` — best bid/ask + freshness clock
* Last-trade / micro-price (when present)
* Public-WS freshness diagnostics
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from app.models import BestBidAsk

if TYPE_CHECKING:
    from app.state import BotState


class MarketStore:
    """Thin facade over ``BotState.market`` and the public-WS
    freshness state. Reads only — market writes happen via the
    public-WS message handler in ``app/exchange``.
    """

    def __init__(self, state: "BotState") -> None:
        self._state = state

    @property
    def bbo(self) -> Optional[BestBidAsk]:
        return self._state.market

    def best_bid(self) -> Optional[float]:
        m = self._state.market
        if m is None:
            return None
        try:
            return float(m.best_bid) if m.best_bid is not None else None
        except (TypeError, ValueError):
            return None

    def best_ask(self) -> Optional[float]:
        m = self._state.market
        if m is None:
            return None
        try:
            return float(m.best_ask) if m.best_ask is not None else None
        except (TypeError, ValueError):
            return None

    def mid_price(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return (b + a) / 2.0

    def spread_abs(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None:
            return None
        return a - b

    def spread_bps(self) -> Optional[float]:
        b, a = self.best_bid(), self.best_ask()
        if b is None or a is None or b <= 0:
            return None
        return (a - b) / b * 10_000.0

    def is_fresh(self) -> bool:
        """True when market data exists and is non-stale.

        Phase 3B uses the same freshness semantic as
        ``BotState.market_data_available`` (which is the
        aggregate boolean produced by the public-WS handler).
        """
        return bool(getattr(self._state, "market_data_available", False))
