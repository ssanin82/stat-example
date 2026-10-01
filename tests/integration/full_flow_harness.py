"""v1.4.93 wedge-elimination-cleanup Phase 6B —
full-flow integration test harness.

Constructs a real OrderManager wired to a stateful MockOkxClient, lets
tests drive ``maybe_refresh_quotes`` with controlled inputs, and
provides assertion helpers.

Architectural sketch::

    +---------+      QuoteDecision     +-----------+
    |  test   |  --------------------> | OrderMgr  | --HTTP--> MockOkxClient
    | scenario|     RiskAction          |           |               |
    |         |                          |           | <--WS--+      |
    |         | <-- assert state -------|           |       |      |
    +---------+                          +-----------+       +------+
                                                            event_queue
                                                          (test injects)

The test:
1. Configures `MockOkxClient.config` for the scenario's failure mode.
2. Calls `harness.tick_once(...)` with a `QuoteDecision` + `RiskAction`.
3. Inspects `client.order_book` for resulting venue state.
4. Inspects `state.order_store` for resulting bot state.
5. Optionally calls `client.emit_ws_*` to simulate WS events.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.enums import ActiveSides, OrderStatus, RiskAction, Side
from app.execution import OrderManager
from app.models import BestBidAsk, PositionSnapshot, QuoteDecision
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.integration.mock_okx_client import MockOkxClient
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///"
        + (
            Path(tempfile.gettempdir())
            / f"mm_p6b_{os.getpid()}_{uuid.uuid4().hex}.db"
        ).as_posix(),
        "SYMBOL": "TON-USDT-SWAP",
        "MAX_ABS_POSITION": 100.0,
        "QUOTE_NOTIONAL_USD": 6.0,
        "MAX_ORDER_NOTIONAL_USD": 20.0,
        # Loose floors so the engine accepts our synthesized decisions.
        "MIN_HALF_SPREAD_BPS": 1.5,
        "BASE_HALF_SPREAD_BPS": 3.0,
        "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 3.5,
        "LADDER_NUM_LEVELS_PER_SIDE": 1,  # single-rung default for simplicity
        # Disable the cancel-pending escalation timer for cleaner scenarios.
        "CANCEL_PENDING_CLOID_ESCALATION_ENABLED": False,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


@dataclass
class FullFlowHarness:
    """Glues OrderManager, MockOkxClient, BotState, Storage into a
    test fixture. Constructed via ``new_harness()``."""

    settings: UnitTestSettings
    client: MockOkxClient
    storage: Storage
    state: BotState
    om: OrderManager
    private_q: queue.Queue
    db_path: Path

    def seed_market(
        self,
        best_bid: float = 2.000,
        best_ask: float = 2.002,
        mid_price: Optional[float] = None,
    ) -> None:
        self.state.market = BestBidAsk(
            symbol=self.settings.symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid_price if mid_price is not None else (best_bid + best_ask) / 2,
            spread_bps=None,
            ts_local=datetime.now(timezone.utc),
        )

    def seed_position(self, qty: float = 0.0, mark: Optional[float] = None) -> None:
        self.state.position = PositionSnapshot(
            symbol=self.settings.symbol,
            position_qty=float(qty),
            avg_entry_price=None,
            mark_price=mark,
            position_notional=abs(qty) * (mark or 2.001),
            unrealized_pnl_usd=0.0,
        )

    def make_decision(
        self,
        *,
        mid: float = 2.001,
        target_spread_bps: float = 6.0,
        quoted_bid: Optional[float] = None,
        quoted_ask: Optional[float] = None,
        quoted_bid_sz: float = 3.0,
        quoted_ask_sz: float = 3.0,
        active_sides: ActiveSides = ActiveSides.BOTH,
        cycle_id: Optional[str] = None,
    ) -> QuoteDecision:
        """Synthesize a QuoteDecision for a tick. Sensible defaults
        for a healthy two-sided quote."""
        hs_bps = target_spread_bps / 2.0
        bid = quoted_bid if quoted_bid is not None else mid * (1.0 - hs_bps / 10_000.0)
        ask = quoted_ask if quoted_ask is not None else mid * (1.0 + hs_bps / 10_000.0)
        return QuoteDecision(
            ts=utc_now(),
            symbol=self.settings.symbol,
            mid_price=mid,
            vol_estimate=1.0,
            inventory=self.state.position.position_qty,
            reservation_price=mid,
            target_spread_bps=target_spread_bps,
            target_bid=bid,
            target_ask=ask,
            quoted_bid=bid,
            quoted_ask=ask,
            quoted_bid_sz=quoted_bid_sz,
            quoted_ask_sz=quoted_ask_sz,
            active_sides=active_sides,
            toxicity_score=0.0,
            decision_reason="test",
            quote_cycle_id=cycle_id or f"cycle-{uuid.uuid4().hex[:8]}",
        )

    def tick_once(
        self,
        *,
        decision: Optional[QuoteDecision] = None,
        risk_action: RiskAction = RiskAction.ALLOW,
        bid_mult: float = 1.0,
        ask_mult: float = 1.0,
        spread_add_bps: float = 0.0,
        cancel_on_no_quote: bool = False,
        wait_outbound_idle_s: float = 1.0,
    ) -> None:
        """Run one tick of ``maybe_refresh_quotes`` with the given
        inputs (synthesizes a healthy default decision if absent).

        After dispatch, waits up to ``wait_outbound_idle_s`` for the
        outbound dispatcher's worker thread to drain queued intents
        (places/cancels go through an async queue; without the wait
        the test sees half-completed state)."""
        if decision is None:
            decision = self.make_decision()
        self.om.maybe_refresh_quotes(
            decision=decision,
            risk_action=risk_action,
            bid_mult=bid_mult,
            ask_mult=ask_mult,
            spread_add_bps=spread_add_bps,
            cancel_on_no_quote=cancel_on_no_quote,
        )
        # Wait for the async outbound dispatcher to finish processing
        # queued place/cancel intents this tick produced.
        if wait_outbound_idle_s > 0:
            try:
                self.om._outbound.wait_until_idle(timeout_s=wait_outbound_idle_s)
            except Exception:
                pass

    def drain_private_events(self, max_events: int = 100) -> int:
        """Drain all queued WS events through the bot's private-event
        handler. Returns approximate count processed (the bot's
        drain method doesn't return a count, so we estimate by queue
        depth before vs after).

        The signature of the bot's ``drain_private_events`` is
        ``drain(pnl)`` — it processes everything in the queue. We
        call it with a None PnL tracker since these tests don't
        exercise the fill→PnL path."""
        before = self.private_q.qsize()
        try:
            self.om.drain_private_events(None)
        except TypeError:
            # Older signature may differ; tolerate.
            self.om.drain_private_events()  # type: ignore[call-arg]
        after = self.private_q.qsize()
        return max(0, before - after)

    def cleanup(self) -> None:
        try:
            self.db_path.unlink(missing_ok=True)
        except PermissionError:
            pass


def new_harness(**settings_overrides) -> FullFlowHarness:
    """Construct a fresh FullFlowHarness. Caller is responsible for
    calling `.cleanup()` (or use the `harness_ctx()` context manager).
    """
    settings = _settings(**settings_overrides)
    db_path = Path(settings.database_url.split("sqlite:///", 1)[-1])
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = MockOkxClient(symbol=settings.symbol)
    private_q: queue.Queue = queue.Queue()
    om = OrderManager(settings, client, storage, state, private_event_queue=private_q)
    client.attach_event_queue(private_q)
    return FullFlowHarness(
        settings=settings,
        client=client,
        storage=storage,
        state=state,
        om=om,
        private_q=private_q,
        db_path=db_path,
    )


@contextmanager
def harness_ctx(**settings_overrides):
    """Context manager — guarantees DB cleanup even on test failure."""
    h = new_harness(**settings_overrides)
    try:
        yield h
    finally:
        h.cleanup()
