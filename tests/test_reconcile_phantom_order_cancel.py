"""
Invariant: on exchange_mismatch in _reconcile_side, a cancel MUST be dispatched
for the observed remote oid BEFORE the local slot is nulled.

Regression bug: before the fix, _reconcile_side transitioned the local working
order to DESYNC and set working_bid/ask = None, but did NOT cancel the remote
oid. That left a phantom order live on the exchange; subsequent same-side
placements would race duplicates (observed in snap_20260416_154159 oid=384103722786).
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


def _payload(ev: dict) -> dict:
    raw = ev.get("payload_json")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def _ok_cancel() -> dict:
    return {
        "status": "ok",
        "response": {"type": "cancel", "data": {"statuses": ["success"]}},
    }


def _setup() -> tuple[UnitTestSettings, Path, BotState, OrderManager, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_phantom_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
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


def _seed_working_ask(om: OrderManager, *, local_oid: int) -> None:
    """Place a local WorkingOrder in ACKED status owning working_ask slot."""
    wo = om._stage_place_order_local(
        Side.SELL, price=3100.0, size=0.01, quote_cycle_id="cyc-mismatch"
    )
    assert wo is not None
    wo.order_id_exchange = local_oid
    transition(wo, OrderStatus.ACKED)
    om.persist(wo)
    om._state.working_ask = wo


def test_exchange_mismatch_dispatches_cancel_for_remote_oid() -> None:
    """
    The primary regression: when remote oid != local oid for the same side,
    _reconcile_side MUST dispatch a cancel for the remote oid.
    """
    s, path, state, om, storage = _setup()
    local_oid = 1001
    remote_oid = 2002
    _seed_working_ask(om, local_oid=local_oid)

    remote = HLOpenOrderRaw(
        oid=remote_oid,
        coin=s.symbol,
        side=Side.SELL,
        limit_px=3100.0,
        sz=0.01,
        timestamp=0,
        cloid=None,
    )
    changed = om._reconcile_side(Side.SELL, remote)
    assert changed is True

    # Core invariant: remote_oid cancel MUST have been dispatched.
    client = om._client
    called_oids = [int(c.args[1]) for c in client.cancel_order.call_args_list]
    assert remote_oid in called_oids, (
        f"exchange_mismatch must dispatch cancel for remote oid={remote_oid}; "
        f"cancel_order got called with oids={called_oids}"
    )

    # Local slot cleared; local order transitioned to DESYNC.
    assert state.working_ask is None
    # Side-unresolved flag set with remote_cancel_dispatched=True in payload.
    assert om._is_side_unresolved(Side.SELL) is True

    # Telemetry: orphan_remote_cancel_dispatched event persisted for the remote cancel.
    ev = storage.recent_bot_events(50)
    evs_dispatched = [e for e in ev if e.get("event_type") == "orphan_remote_cancel_dispatched"]
    assert evs_dispatched, (
        "orphan_remote_cancel_dispatched event must be persisted for the remote cancel"
    )
    reasons = {_payload(e).get("reason") for e in evs_dispatched}
    assert "exchange_mismatch_remote" in reasons, (
        f"remote cancel must carry reason=exchange_mismatch_remote; got {reasons}"
    )
    path.unlink(missing_ok=True)


def test_exchange_mismatch_also_cancels_local_ghost_oid() -> None:
    """
    When the local WorkingOrder has a different order_id_exchange (the local
    ghost), the path should also dispatch a cancel for the local oid — it may
    still be alive on the exchange. A benign_missing response is acceptable.
    """
    s, path, state, om, storage = _setup()
    local_oid = 1001
    remote_oid = 2002
    _seed_working_ask(om, local_oid=local_oid)

    remote = HLOpenOrderRaw(
        oid=remote_oid,
        coin=s.symbol,
        side=Side.SELL,
        limit_px=3100.0,
        sz=0.01,
        timestamp=0,
    )
    om._reconcile_side(Side.SELL, remote)

    client = om._client
    called_oids = [int(c.args[1]) for c in client.cancel_order.call_args_list]
    assert remote_oid in called_oids
    assert local_oid in called_oids, (
        f"local ghost cancel must also be dispatched; called_oids={called_oids}"
    )

    ev = storage.recent_bot_events(50)
    reasons = {_payload(e).get("reason") for e in ev if e.get("event_type") == "orphan_remote_cancel_dispatched"}
    assert "exchange_mismatch_local_ghost" in reasons
    path.unlink(missing_ok=True)


def test_exchange_mismatch_cancel_failure_is_recorded_not_swallowed() -> None:
    """
    If the remote cancel fails (non-benign), the failure MUST be recorded
    (orphan_cancel_failed event) and reflected in the side-unresolved payload.
    This prevents the phantom from being silently forgotten.
    """
    s, path, state, om, storage = _setup()
    client = om._client
    # Exchange rejects the cancel.
    client.cancel_order.return_value = {
        "status": "ok",
        "response": {
            "type": "cancel",
            "data": {"statuses": [{"error": "Order no longer a valid resting order"}]},
        },
    }
    # That error string is benign_missing (see _cancel_error_is_benign_missing).
    # Replace with something genuinely non-benign.
    client.cancel_order.return_value = {
        "status": "ok",
        "response": {
            "type": "cancel",
            "data": {"statuses": [{"error": "internal server error"}]},
        },
    }

    _seed_working_ask(om, local_oid=1001)
    remote = HLOpenOrderRaw(
        oid=2002, coin=s.symbol, side=Side.SELL, limit_px=3100.0, sz=0.01, timestamp=0
    )
    om._reconcile_side(Side.SELL, remote)

    ev = storage.recent_bot_events(50)
    failed_events = [e for e in ev if e.get("event_type") == "orphan_cancel_failed"]
    assert failed_events, (
        "orphan_cancel_failed must be persisted when exchange rejects the cancel"
    )
    path.unlink(missing_ok=True)
