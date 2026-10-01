"""v1.4.82 wedge-elimination-cleanup Phase 3C — TickSnapshot.

Frozen per-tick view of state. Consumers (engine build, decision
layer, reconciler, dispatcher) read from the snapshot instead of
re-reading mutable state through the locks.

Why this matters
================

Pre-Phase-3C ``maybe_refresh_quotes`` acquired ``state._lock``
multiple times per tick (Codex F9 flagged "7 acquisitions on the
happy path"). Each acquisition is a cross-thread serialization
point with the WS handlers. A single snapshot at tick start, then
all downstream reads from the immutable view, removes the
contention without changing semantics.

Phase 3C is the INFRASTRUCTURE for the optimization. Phase 5A is
the cutover that actually replaces the legacy read paths. Until
then the snapshot is available for new callers (postmortem
extractors, integration test harnesses) that benefit from a
coherent point-in-time view.

Design
======

Each store contributes a typed ``Immutable*View`` dataclass.
``TickSnapshot`` composes them. All fields are primitive or
deep-frozen (tuples, immutable dicts, dataclasses with ``frozen=True``).

Mutability: callers MUST treat the snapshot as read-only. Python
doesn't fully enforce immutability on contained Python objects
(e.g., a WorkingOrder reference can still be mutated by another
thread), so the contract is: don't mutate anything reachable from
a TickSnapshot. If you need to mutate state, go through the
proper write path (``OrderStore.transition`` / etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from app.enums import Side
from app.models import BestBidAsk, PositionSnapshot, WorkingOrder

if TYPE_CHECKING:
    from app.state import BotState


@dataclass(frozen=True, slots=True)
class ImmutableOrderView:
    """Frozen view of working orders at one moment.

    ``by_slot`` is a tuple-of-tuples (sortable, hashable) rather
    than a dict — supports deterministic iteration and slice
    operations without forcing the caller to handle dict-ordering
    edge cases.
    """

    # ``((side, level_idx, wo), ...)`` for every live WO.
    by_slot: tuple[tuple[Side, int, WorkingOrder], ...] = ()

    def slot(self, side: Side, level_idx: int) -> Optional[WorkingOrder]:
        for s, lvl, wo in self.by_slot:
            if s == side and lvl == int(level_idx):
                return wo
        return None

    def side(self, side: Side) -> tuple[tuple[int, WorkingOrder], ...]:
        return tuple(
            (lvl, wo) for s, lvl, wo in self.by_slot if s == side
        )

    def count_live_per_side(self, side: Side) -> int:
        return sum(1 for s, _, _ in self.by_slot if s == side)

    def is_empty(self) -> bool:
        return len(self.by_slot) == 0


@dataclass(frozen=True, slots=True)
class ImmutablePositionView:
    """Frozen view of position state."""

    qty: float
    notional_abs_usd: float
    unrealized_pnl_usd: float
    avg_entry_price: Optional[float]
    mark_price: Optional[float]

    def is_flat(self, eps: float = 1e-10) -> bool:
        return abs(self.qty) <= eps

    def reducing_side(self) -> Optional[Side]:
        if self.qty > 0:
            return Side.SELL
        if self.qty < 0:
            return Side.BUY
        return None


@dataclass(frozen=True, slots=True)
class ImmutableMarketView:
    """Frozen view of market data."""

    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread_bps: Optional[float]
    ts_local: Optional[datetime]

    def is_present(self) -> bool:
        return (
            self.best_bid is not None
            and self.best_ask is not None
            and self.best_bid > 0
            and self.best_ask > 0
        )


@dataclass(frozen=True, slots=True)
class TickSnapshot:
    """Composite per-tick view across all four stores.

    Build once at the top of ``maybe_refresh_quotes`` (Phase 5A);
    pass through to engine / decision / reconciler / dispatcher.

    v1.4.87 Phase 5A: wired into the hot path. ``maybe_refresh_quotes``
    now calls ``state.tick_snapshot()`` once per tick and reads all
    downstream state from this object.
    """

    orders: ImmutableOrderView
    position: ImmutablePositionView
    market: ImmutableMarketView
    # Wall clock of when the snapshot was built. Useful for
    # downstream age math + audit trail.
    captured_at: datetime
    # Bot status string at snapshot time (RUNNING, STARTING, etc.)
    bot_status: str = ""
    # v1.4.87 Phase 5A: the raw ``BestBidAsk`` reference captured
    # under the same lock as the rest of the snapshot. Needed by
    # consumers (``QuoteBuildContext.market``) that take the object
    # rather than primitives. ``BestBidAsk`` itself is a frozen
    # dataclass so the reference is effectively immutable; callers
    # treat it as read-only.
    market_raw: Optional[BestBidAsk] = None


def build_tick_snapshot(state: "BotState") -> TickSnapshot:
    """Construct a ``TickSnapshot`` from current state. Acquires
    ``state._lock`` once and reads everything inside it; the result
    is immutable and safe to pass between threads.
    """
    from app.utils.time import utc_now

    with state._lock:
        # Order view — iterate the order_store's underlying dict
        # while holding the lock so we don't see a half-mutated
        # state.
        order_tuples: list[tuple[Side, int, WorkingOrder]] = []
        for side in (Side.BUY, Side.SELL):
            for lvl, wo in state.iter_working_orders(side):
                if wo is not None:
                    order_tuples.append((side, int(lvl), wo))
        orders = ImmutableOrderView(by_slot=tuple(order_tuples))

        # Position view.
        pos = state.position
        position = ImmutablePositionView(
            qty=float(pos.position_qty),
            notional_abs_usd=float(pos.position_notional),
            unrealized_pnl_usd=float(pos.unrealized_pnl_usd or 0.0),
            avg_entry_price=(
                float(pos.avg_entry_price)
                if pos.avg_entry_price is not None
                else None
            ),
            mark_price=(
                float(pos.mark_price)
                if pos.mark_price is not None
                else None
            ),
        )

        # Market view.
        m = state.market
        if m is None:
            market = ImmutableMarketView(
                best_bid=None, best_ask=None, mid_price=None,
                spread_bps=None, ts_local=None,
            )
        else:
            bb = float(m.best_bid) if m.best_bid is not None else None
            ba = float(m.best_ask) if m.best_ask is not None else None
            mid = float(m.mid_price) if m.mid_price is not None else None
            sb = float(m.spread_bps) if m.spread_bps is not None else None
            market = ImmutableMarketView(
                best_bid=bb, best_ask=ba, mid_price=mid,
                spread_bps=sb, ts_local=m.ts_local,
            )

        bot_status_str = ""
        bs = getattr(state, "bot_status", None)
        if bs is not None:
            bot_status_str = (
                bs.value if hasattr(bs, "value") else str(bs)
            )

    return TickSnapshot(
        orders=orders,
        position=position,
        market=market,
        captured_at=utc_now(),
        bot_status=bot_status_str,
        market_raw=m,  # captured under the same lock as the rest
    )
