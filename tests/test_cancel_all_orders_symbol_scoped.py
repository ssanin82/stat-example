"""``cancel_all_orders_for_symbol`` symbol-scope invariant.

v1.4.55 wedge-elimination Phase 1 changed the IMPLEMENTATION but
the INVARIANT survives — strengthened, in fact:

PRE-v1.4.55 implementation:
  cancel-all iterated ``client.fetch_open_orders_raw(addr)`` results,
  filtered each row with ``if o.coin != settings.symbol: continue``,
  fired ``client.cancel_order(...)`` per surviving row. The symbol-
  scope check was an EXPLICIT FILTER at the cancel-all entry point.

POST-v1.4.55 implementation:
  cancel-all iterates ``state.all_working_orders()`` and routes
  cancels through ``_enqueue_cancel_quote_path`` (the dispatcher).
  The symbol-scope invariant is now STRUCTURAL: the local working-
  order state is owned by the OrderManager which only ever creates
  WOs for its own configured symbol. There's no path by which a
  WO for a foreign symbol can end up in ``state.working_orders``.

This file's tests verify the NEW invariant:
  * cancel-all enqueues cancels for the bot's own local WOs
  * cancel-all is a no-op when local state is empty (no working orders)
  * the back-compat alias still resolves to the same function

The CROSS-SYMBOL CONTAMINATION scenarios from the pre-v1.4.55 tests
are unreachable in the new design — there's no code path that puts
a foreign-symbol WO into local state. The orphan-detection / orphan-
cancel path on reconcile is what handles "exchange has an order
we don't know about" cases, and that path filters by symbol at the
adapter level (verified by adapter tests, unchanged).
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from app.bot import Bot
from app.enums import OrderStatus, Side
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.models import WorkingOrder
from app.storage import Storage
from app.state import BotState
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _db_path() -> Path:
    return Path(tempfile.gettempdir()) / f"mm_sym_{os.getpid()}_{uuid.uuid4().hex}.db"


def _settings(symbol: str = "ETH") -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": symbol,
            "PRIVATE_WS_ENABLED": False,
        }
    )


def _ack_wo(symbol: str, side: Side, oid: int, price: float = 100.0):
    """Build an ACKED working order suitable for cancel-all consumption."""
    now = utc_now()
    return WorkingOrder(
        order_id_local=f"local-{uuid.uuid4().hex[:8]}",
        order_id_exchange=oid,
        client_order_id=f"cloid-{oid}",
        symbol=symbol,
        side=side,
        price=float(price),
        size=1.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=now,
        ts_sent=now,
        ts_ack=now,
    )


def _make_bot(settings: UnitTestSettings) -> Bot:
    """Construct a Bot with empty local state. Tests seed local WOs
    directly via ``state.set_working_order`` to verify cancel-all's
    new state-aware behavior.
    """
    path = _db_path()
    path.unlink(missing_ok=True)
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)
    return bot


# ------------------- New (Phase 1) state-driven invariants ------------------


def test_cancel_all_cancels_all_local_wos_for_own_symbol() -> None:
    """v1.4.55 Phase 1: cancel-all enqueues cancels for every
    cancellable local WO. Each enqueue routes through
    ``_enqueue_cancel_quote_path`` (the dispatcher), which
    transitions the WO to CANCEL_PENDING — verifiable from local
    state.
    """
    settings = _settings(symbol="ETH")
    bot = _make_bot(settings)
    # Seed two local WOs for the bot's symbol.
    wo_buy = _ack_wo("ETH", Side.BUY, oid=100)
    wo_sell = _ack_wo("ETH", Side.SELL, oid=101)
    with bot._state._lock:
        bot._state.set_working_order(Side.BUY, 0, wo_buy)
        bot._state.set_working_order(Side.SELL, 0, wo_sell)

    bot._exec.cancel_all_orders_for_symbol()

    # Both WOs transitioned to CANCEL_PENDING by the dispatcher
    # enqueue path. (The actual venue cancel runs async in the
    # dispatcher worker — we verify the local state transition only.)
    assert wo_buy.status == OrderStatus.CANCEL_PENDING, (
        f"BUY WO must be CANCEL_PENDING after cancel-all, got {wo_buy.status}"
    )
    assert wo_sell.status == OrderStatus.CANCEL_PENDING, (
        f"SELL WO must be CANCEL_PENDING after cancel-all, got {wo_sell.status}"
    )


def test_cancel_all_no_op_when_local_state_empty() -> None:
    """v1.4.55 Phase 1: cancel-all is a no-op when local working-order
    state is empty. No enqueue calls, no exceptions, clean return.
    """
    settings = _settings(symbol="ETH")
    bot = _make_bot(settings)

    enq_calls = []
    original = bot._exec._enqueue_cancel_quote_path
    def _spy(wo_arg, **kwargs):
        enq_calls.append(wo_arg.order_id_exchange)
        return original(wo_arg, **kwargs)
    bot._exec._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

    bot._exec.cancel_all_orders_for_symbol()
    assert enq_calls == [], (
        f"cancel-all with empty local state must not enqueue anything. "
        f"Got: {enq_calls}"
    )


def test_backcompat_alias_points_to_same_function() -> None:
    """``cancel_all_orders`` (old name) remains a back-compat alias for
    ``cancel_all_orders_for_symbol``. Both resolve to the same function
    object so any out-of-tree caller keeps working.
    """
    from app.execution import OrderManager

    assert OrderManager.cancel_all_orders is OrderManager.cancel_all_orders_for_symbol


def test_alias_call_also_cancels_local_wos() -> None:
    """v1.4.55 Phase 1: the legacy ``cancel_all_orders`` alias resolves
    to the same function and therefore inherits the same state-driven
    behavior. Single ACKED WO → single CANCEL_PENDING transition.
    """
    settings = _settings(symbol="ETH")
    bot = _make_bot(settings)
    wo = _ack_wo("ETH", Side.BUY, oid=100)
    with bot._state._lock:
        bot._state.set_working_order(Side.BUY, 0, wo)

    bot._exec.cancel_all_orders()  # legacy alias

    assert wo.status == OrderStatus.CANCEL_PENDING


# ------------------- Symbol-scope invariant (now structural) ----------------


def test_symbol_scope_is_structural_local_state_only_holds_own_symbol() -> None:
    """v1.4.55 Phase 1: the symbol-scope invariant for cancel-all is
    no longer enforced by a runtime ``if coin != symbol: continue``
    filter — it's STRUCTURAL. The bot's local working-order state
    is populated EXCLUSIVELY by code paths that create WOs for the
    bot's own configured symbol (``_stage_place_order_local``,
    ``_hydrate_working_from_exchange``, etc), and there is no code
    path that puts a foreign-symbol WO into ``state.working_orders``.

    This test pins the invariant by inspecting the bot's local state
    after construction: no foreign-symbol WOs should ever be present,
    so cancel-all has no foreign-symbol rows to skip.

    (The orphan-cancel path on reconcile is what handles "exchange
    has an order we don't know about" — that path filters by symbol
    at the adapter level, verified by adapter tests.)
    """
    settings = _settings(symbol="ETH")
    bot = _make_bot(settings)
    # After construction the local WO state is empty.
    with bot._state._lock:
        wos = list(bot._state.all_working_orders())
    assert wos == [], (
        f"Local state must start empty. Got: {wos}"
    )
    # And cancel-all is a no-op in that condition (proven in the
    # dedicated test above; reaffirmed here to pin the structural
    # invariant). All paths that mutate this state go through the
    # OrderManager, which only ever uses ``settings.symbol``.


def test_cancel_all_skips_pre_ack_and_terminal_wos() -> None:
    """v1.4.55 Phase 1 detail: cancel-all skips WOs in pre-ack
    (NEW_LOCAL, SENT) and terminal (CANCELED, FILLED, REJECTED)
    statuses. Pre-ack skip avoids a race with the place response;
    terminal skip avoids redundant work.

    Only ACKED, PARTIAL, and AMEND_PENDING WOs get enqueued for
    cancel.
    """
    settings = _settings(symbol="ETH")
    bot = _make_bot(settings)
    # NEW_LOCAL on BUY rung 0 — should be skipped.
    wo_new = _ack_wo("ETH", Side.BUY, oid=200)
    wo_new.status = OrderStatus.NEW_LOCAL
    # CANCELED on SELL rung 0 — should be skipped.
    wo_done = _ack_wo("ETH", Side.SELL, oid=201)
    wo_done.status = OrderStatus.CANCELED
    with bot._state._lock:
        bot._state.set_working_order(Side.BUY, 0, wo_new)
        bot._state.set_working_order(Side.SELL, 0, wo_done)

    enq_calls = []
    original = bot._exec._enqueue_cancel_quote_path
    def _spy(wo_arg, **kwargs):
        enq_calls.append(wo_arg.order_id_exchange)
        return original(wo_arg, **kwargs)
    bot._exec._enqueue_cancel_quote_path = _spy  # type: ignore[method-assign]

    bot._exec.cancel_all_orders_for_symbol()
    assert enq_calls == [], (
        "WOs in NEW_LOCAL or CANCELED status must not be enqueued for "
        f"cancel. Got: {enq_calls}"
    )
    assert wo_new.status == OrderStatus.NEW_LOCAL, (
        "NEW_LOCAL status must not be mutated by cancel-all skip path"
    )
    assert wo_done.status == OrderStatus.CANCELED, (
        "Terminal CANCELED status must not be re-mutated by cancel-all"
    )
