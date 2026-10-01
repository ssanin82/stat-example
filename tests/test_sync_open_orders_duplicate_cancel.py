"""
Invariant: when sync_open_orders sees multiple remote orders on the same side,
it MUST dispatch cancels for every extra (non-keeper) one — not merely record
the duplicate in telemetry.

Regression bug: before the fix, duplicate detection set order_desync and
recorded the condition, but the losing duplicate(s) stayed live on the exchange
until the next reconcile cycle (or until a later code path cleaned them up).
With a 30-second reconcile cooldown, that meant phantoms could persist for
tens of seconds producing duplicate fills.

Keeper selection rule: prefer the remote order whose oid matches our local
WorkingOrder's order_id_exchange (so hydrate/match paths still work). Otherwise
deterministic fallback = newest timestamp (last placed wins) with oid tiebreak.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from app.enums import OrderStatus, Side
from app.execution import OrderManager, transition
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _ok_cancel() -> dict:
    return {
        "status": "ok",
        "response": {"type": "cancel", "data": {"statuses": ["success"]}},
    }


def _payload(ev: dict) -> dict:
    raw = ev.get("payload_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _setup() -> tuple[UnitTestSettings, Path, BotState, OrderManager, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_dup_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            # Central pre-send gate (Codex MED #1, 2026-05-06)
            # checks position caps. Raise so this test's size=0.1
            # mock orders aren't refused.
            "MAX_ABS_POSITION": 100.0,
            "MAX_POSITION_NOTIONAL_USD": 100_000.0,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.cancel_order.return_value = _ok_cancel()
    client.cancel_order_by_cloid.return_value = _ok_cancel()
    om = OrderManager(s, client, storage, state)
    return s, path, state, om, storage


def test_three_same_side_remotes_cancel_two_losers() -> None:
    """3 BUY remote orders (no local WorkingOrder) → keeper=newest, 2 cancels dispatched."""
    s, path, state, om, storage = _setup()
    sym = s.symbol
    # Three same-side remote orders with ascending timestamps; newest (ts=300) is keeper.
    remotes = [
        HLOpenOrderRaw(oid=101, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=100),
        HLOpenOrderRaw(oid=102, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=200),
        HLOpenOrderRaw(oid=103, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=300),
    ]
    om._client.fetch_open_orders_raw.return_value = remotes

    om.sync_open_orders(force=True, emergency=True)

    # Exactly 2 cancel_order dispatches — for the two losing oids 101 and 102.
    canceled_oids = {int(c.args[1]) for c in om._client.cancel_order.call_args_list}
    assert canceled_oids == {101, 102}, (
        f"must cancel both non-keeper extras; got {canceled_oids}"
    )

    # Every cancel must be persisted as orphan_remote_cancel_dispatched with
    # reason=duplicate_same_side_extra.
    ev = storage.recent_bot_events(50)
    dup_events = [
        e for e in ev
        if e.get("event_type") == "orphan_remote_cancel_dispatched"
        and _payload(e).get("reason") == "duplicate_same_side_extra"
    ]
    assert len(dup_events) == 2, (
        f"two orphan cancel events must be persisted; got {len(dup_events)} "
        f"events={[_payload(e) for e in dup_events]}"
    )
    path.unlink(missing_ok=True)


def test_duplicate_with_local_match_keeps_locally_owned() -> None:
    """Among duplicates, the keeper must be the one matching our local order_id_exchange."""
    s, path, state, om, storage = _setup()
    sym = s.symbol

    # Seed a local working_bid whose order_id_exchange=555 — keeper must be oid=555.
    wo = om._stage_place_order_local(
        Side.BUY, price=100.0, size=0.1, quote_cycle_id="cyc-local"
    )
    assert wo is not None
    wo.order_id_exchange = 555
    transition(wo, OrderStatus.ACKED)
    om.persist(wo)
    state.working_bid = wo

    remotes = [
        # Newer ts than keeper — must NOT win (local match overrides newest-wins).
        HLOpenOrderRaw(oid=999, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=1000),
        HLOpenOrderRaw(oid=555, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=500),
    ]
    om._client.fetch_open_orders_raw.return_value = remotes

    om.sync_open_orders(force=True, emergency=True)

    canceled_oids = {int(c.args[1]) for c in om._client.cancel_order.call_args_list}
    assert canceled_oids == {999}, (
        f"locally-owned 555 must be keeper; 999 must be canceled. got {canceled_oids}"
    )
    path.unlink(missing_ok=True)


def test_duplicates_on_both_sides_dispatch_both_lanes() -> None:
    """Duplicates present on BOTH sides → cancels for extras on BOTH sides."""
    s, path, state, om, storage = _setup()
    sym = s.symbol
    remotes = [
        HLOpenOrderRaw(oid=1, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=10),
        HLOpenOrderRaw(oid=2, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=20),
        HLOpenOrderRaw(oid=3, coin=sym, side=Side.SELL, limit_px=110.0, sz=0.1, timestamp=30),
        HLOpenOrderRaw(oid=4, coin=sym, side=Side.SELL, limit_px=110.0, sz=0.1, timestamp=40),
    ]
    om._client.fetch_open_orders_raw.return_value = remotes

    om.sync_open_orders(force=True, emergency=True)

    # Keepers = 2 (buy newest) and 4 (sell newest); losers = 1 (buy) and 3 (sell).
    canceled_oids = {int(c.args[1]) for c in om._client.cancel_order.call_args_list}
    assert canceled_oids == {1, 3}, (
        f"both-side duplicates must dispatch cancels for all losers; got {canceled_oids}"
    )
    path.unlink(missing_ok=True)


def test_no_duplicates_no_cancel_dispatched() -> None:
    """Baseline: one remote per side → zero orphan cancels dispatched."""
    s, path, state, om, storage = _setup()
    sym = s.symbol
    remotes = [
        HLOpenOrderRaw(oid=1, coin=sym, side=Side.BUY, limit_px=100.0, sz=0.1, timestamp=10),
        HLOpenOrderRaw(oid=2, coin=sym, side=Side.SELL, limit_px=110.0, sz=0.1, timestamp=20),
    ]
    om._client.fetch_open_orders_raw.return_value = remotes
    om.sync_open_orders(force=True, emergency=True)
    assert om._client.cancel_order.call_count == 0
    path.unlink(missing_ok=True)
