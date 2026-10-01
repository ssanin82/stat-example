"""v1.4.80 wedge-elimination-cleanup Phase 3 — store package.

The historical ``BotState`` god-class (3,480+ lines, 81 methods)
owned every piece of mutable state in the bot: working orders,
position, market data, telemetry counters, lifecycle traces, ws-
queue stats, executor-state surfaces. Three structural problems
from that shape:

1. **No single point of order-state mutation** — callers reach into
   ``state._working_orders`` directly, bypassing any invariant
   the future-self might want to enforce (e.g., index consistency,
   transition legality).

2. **One mega-lock** — ``state._lock`` serializes everything across
   threads. WS handlers, quote-loop, snapshot-writer all contend.

3. **Untestable in isolation** — to test order-state behaviour you
   must instantiate the full god-class with database connections,
   pnl trackers, telemetry deques, etc.

Phase 3 splits the god-class into focused stores:

  * ``OrderStore``      — working orders (this phase, 3A)
  * ``PositionStore``   — position state (3B)
  * ``MarketStore``     — best-bid/ask, freshness clocks (3B)
  * ``TelemetryStore``  — counters, ring buffers (3B)

Each store has its own lock. ``BotState`` becomes a thin composite
that holds the four stores plus the bot-status flags.

The refactor is INCREMENTAL — each sub-phase ships independently
and the legacy API is preserved until Phase 3D explicitly removes
it. No big-bang rewrites.
"""

from app.stores.market_store import MarketStore
from app.stores.order_store import OrderStore
from app.stores.position_store import PositionStore
from app.stores.telemetry_store import TelemetryStore
from app.stores.tick_snapshot import (
    ImmutableMarketView,
    ImmutableOrderView,
    ImmutablePositionView,
    TickSnapshot,
    build_tick_snapshot,
)

__all__ = [
    "OrderStore",
    "PositionStore",
    "MarketStore",
    "TelemetryStore",
    "TickSnapshot",
    "ImmutableOrderView",
    "ImmutablePositionView",
    "ImmutableMarketView",
    "build_tick_snapshot",
]
