"""Execution must not send ``order_id=0`` to GRVT and must recover via cloid.

Regression guard for snap_20260417_144515: the bot sat in CANCEL_PENDING for
110+ seconds because every cancel went out as ``{"order_id": "0"}`` and GRVT
rejected it with "Either order ID or client order ID must be supplied".
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.enums import OrderStatus, Side
from app.exchange.base import OpenOrderRaw
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from app.execution import OrderManager, transition
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[OrderManager, Any, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_grvt_cancel_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH_USDT_Perp",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    client.interpret_cancel_response.return_value = ("success", "")
    om = OrderManager(settings, client, storage, state)
    return om, client, path


def _make_sent_wo(
    *,
    cloid: str,
    order_id_exchange: Any = None,
    symbol: str = "ETH_USDT_Perp",
) -> WorkingOrder:
    return WorkingOrder(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=order_id_exchange,
        client_order_id=cloid,
        symbol=symbol,
        side=Side.BUY,
        price=2442.42,
        size=0.01,
        post_only=True,
        status=OrderStatus.SENT,
    )


def test_cancel_with_zero_oid_falls_back_to_cloid() -> None:
    """``wo.order_id_exchange == 0`` must not produce ``{"order_id": "0"}``."""
    om, client, path = _setup()
    try:
        wo = _make_sent_wo(cloid="6011131468247176399", order_id_exchange=0)
        om.cancel_order(wo)
        client.cancel_order.assert_not_called()
        client.cancel_order_by_cloid.assert_called_once_with(
            "ETH_USDT_Perp", "6011131468247176399"
        )
    finally:
        path.unlink(missing_ok=True)


def test_cancel_with_real_oid_uses_oid_path() -> None:
    om, client, path = _setup()
    try:
        wo = _make_sent_wo(
            cloid="6011131468247176399",
            order_id_exchange=1334440892959520532621925129411760553,
        )
        om.cancel_order(wo)
        client.cancel_order.assert_called_once_with(
            "ETH_USDT_Perp", 1334440892959520532621925129411760553
        )
        client.cancel_order_by_cloid.assert_not_called()
    finally:
        path.unlink(missing_ok=True)


def test_orphan_cancel_with_zero_oid_falls_back_to_cloid() -> None:
    om, client, path = _setup()
    try:
        ok = om._cancel_orphan_remote_order(
            symbol="ETH_USDT_Perp",
            oid=0,
            cloid="6011131468247176399",
            reason="exchange_mismatch_local_ghost",
        )
        assert ok is True
        client.cancel_order.assert_not_called()
        client.cancel_order_by_cloid.assert_called_once_with(
            "ETH_USDT_Perp", "6011131468247176399"
        )
    finally:
        path.unlink(missing_ok=True)


def test_orphan_cancel_with_zero_oid_and_no_cloid_refuses() -> None:
    """Don't dispatch a zero-oid, no-cloid cancel — that was the 400 path."""
    om, client, path = _setup()
    try:
        ok = om._cancel_orphan_remote_order(
            symbol="ETH_USDT_Perp",
            oid=0,
            cloid=None,
            reason="exchange_mismatch_local_ghost",
        )
        assert ok is False
        client.cancel_order.assert_not_called()
        client.cancel_order_by_cloid.assert_not_called()
    finally:
        path.unlink(missing_ok=True)


def test_reconcile_binds_oid_by_cloid_when_local_has_none() -> None:
    """If remote cloid == local cloid, bind oid instead of DESYNC'ing."""
    om, client, path = _setup()
    try:
        wo = _make_sent_wo(cloid="6011131468247176399", order_id_exchange=None)
        # Simulate post-cancel state — wo was transitioned to CANCEL_PENDING
        # by the quote engine, but the real id had never been assigned.
        transition(wo, OrderStatus.CANCEL_PENDING)
        with om._state._lock:
            om._state.working_bid = wo
        remote = OpenOrderRaw(
            oid=1334440892959520532621925129411760553,
            coin="ETH_USDT_Perp",
            side=Side.BUY,
            limit_px=2442.42,
            sz=0.01,
            timestamp=0,
            cloid="6011131468247176399",
        )
        desync = om._reconcile_side(Side.BUY, remote)
        assert desync is False
        # oid has been bound; status preserved (still CANCEL_PENDING — the
        # in-flight cancel is allowed to complete rather than being torn down).
        assert wo.order_id_exchange == 1334440892959520532621925129411760553
        assert wo.status == OrderStatus.CANCEL_PENDING
        # No orphan cancel for "local_ghost" — we recognized the same order.
        client.cancel_order.assert_not_called()
    finally:
        path.unlink(missing_ok=True)


def test_reconcile_still_desyncs_on_true_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Guard that the cloid-binding recovery doesn't swallow real mismatches."""
    om, client, path = _setup()
    try:
        wo = _make_sent_wo(
            cloid="6011131468247176399",
            order_id_exchange=555555555555555555,  # a real but different oid
        )
        transition(wo, OrderStatus.ACKED)
        with om._state._lock:
            om._state.working_bid = wo
        remote = OpenOrderRaw(
            oid=1334440892959520532621925129411760553,
            coin="ETH_USDT_Perp",
            side=Side.BUY,
            limit_px=2442.42,
            sz=0.01,
            timestamp=0,
            cloid="DIFFERENT_CLOID_9999",
        )
        desync = om._reconcile_side(Side.BUY, remote)
        assert desync is True
        # Both orphan cancels dispatched: remote.oid and local_oid (both real).
        assert client.cancel_order.call_count >= 1
        assert wo.status == OrderStatus.DESYNC
    finally:
        path.unlink(missing_ok=True)


def test_reconcile_mismatch_skips_local_ghost_cancel_when_local_oid_is_zero() -> None:
    """A zero local_oid is a bogus ack, not a separate order to cancel."""
    om, client, path = _setup()
    try:
        wo = _make_sent_wo(
            cloid="6011131468247176399",
            order_id_exchange=0,  # bogus zero left from a placeholder ack
        )
        transition(wo, OrderStatus.CANCEL_PENDING)
        with om._state._lock:
            om._state.working_bid = wo
        remote = OpenOrderRaw(
            oid=1334440892959520532621925129411760553,
            coin="ETH_USDT_Perp",
            side=Side.BUY,
            limit_px=2442.42,
            sz=0.01,
            timestamp=0,
            cloid="DIFFERENT_CLOID_9999",  # forces mismatch branch
        )
        om._reconcile_side(Side.BUY, remote)
        # Exactly ONE orphan cancel: the remote phantom. The "local_ghost"
        # path must be skipped because local_oid=0 is not a real second order.
        assert client.cancel_order.call_count == 1
        client.cancel_order.assert_called_with(
            "ETH_USDT_Perp", 1334440892959520532621925129411760553
        )
    finally:
        path.unlink(missing_ok=True)


def test_place_response_with_zero_oid_does_not_ack_local_wo(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """GRVT may return ``order_id: "0"`` synchronously — we must stay SENT.

    2026-05-14 BUG-024: this is GRVT's legitimate PENDING-state path.
    The new strict-kill guard (default True) would CRITICAL-kill the
    bot on this response. Opt out per-test so the legacy GRVT
    reconcile-by-cloid recovery path remains tested.
    """
    om, client, path = _setup()
    om._settings = om._settings.model_copy(
        update={"strict_place_unconfirmed_kill": False}
    )
    try:
        client.place_post_only_limit.return_value = {
            "result": {"order_id": "0", "state": {"status": "PENDING"}}
        }
        client.interpret_place_response.return_value = (0, "accepted", "")
        wo = om.place_passive_order_manual_only(Side.BUY, 2442.42, 0.01, "q1")
        assert wo is not None
        # Must NOT have been transitioned to ACKED with a bogus 0 oid.
        assert wo.order_id_exchange is None
        # Must remain SENT (pending real oid via reconcile/WS by cloid).
        assert wo.status == OrderStatus.SENT
    finally:
        path.unlink(missing_ok=True)
