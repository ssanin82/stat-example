"""Regression test for the 2026-05-14 orphan-fill race.

Reproduction (from the live SUI session):

    Local order acked at  T = 08:28:37.425
    Local order CANCELED at T+1ms with reason "gone_on_exchange"
        ts_cancel_requested = NULL (the bot never sent a cancel)
    Same order_id_exchange FILLED on the venue at T+3.3 s
        (because the bot believed it canceled the order, but never
        actually told OKX, so the order stayed alive on the venue
        and was hit by a market buy)

Root cause:

    The bot's open-orders REST snapshot was DISPATCHED BEFORE the
    WS ack landed. OKX's response was built from a state that did
    not yet include the freshly-acked order. Reconcile saw
    ``local says alive, remote doesn't list it`` and concluded
    ``gone_on_exchange`` — orphan.

Fix:

    Capture the REST request's dispatch timestamp; in
    ``_reconcile_side`` skip the ``gone_on_exchange`` decision when
    the working order's ts_ack (ACKED branch) or ts_sent (SENT
    branch) post-dates that timestamp. The snapshot cannot possibly
    have contained the order.

Tests below assert:

    1. ACKED branch — freshly-acked order is NOT marked gone when
       its ts_ack is AFTER the REST dispatch instant. Slot retained.
       Counter ``reconcile_skip_snapshot_stale_total`` increments.
    2. SENT branch — freshly-sent order is NOT marked gone when its
       ts_sent is AFTER the REST dispatch instant.
    3. Inverse — an order whose ts_ack is BEFORE the REST dispatch
       instant DOES go through the existing gone_on_exchange path
       (we haven't broken the legitimate "order really vanished"
       behaviour).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.enums import OrderStatus, Side
from app.execution import OrderManager, transition
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[UnitTestSettings, Path, BotState, OrderManager]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_orph_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
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
    om = OrderManager(s, client, storage, state)
    return s, path, state, om


def _seed_acked_ask(
    om: OrderManager, *, local_oid: int, ts_ack: datetime
) -> None:
    """Stage a local SELL with ACKED status and explicit ts_ack."""
    wo = om._stage_place_order_local(
        Side.SELL, price=3100.0, size=0.01, quote_cycle_id="orph-test"
    )
    assert wo is not None
    wo.order_id_exchange = local_oid
    transition(wo, OrderStatus.ACKED)
    wo.ts_ack = ts_ack
    om.persist(wo)
    om._state.working_ask = wo


def _seed_sent_ask(
    om: OrderManager, *, local_oid: int, ts_sent: datetime
) -> None:
    """Stage a local SELL stuck in SENT (no ack yet) with explicit ts_sent."""
    wo = om._stage_place_order_local(
        Side.SELL, price=3100.0, size=0.01, quote_cycle_id="orph-test-sent"
    )
    assert wo is not None
    wo.order_id_exchange = local_oid
    transition(wo, OrderStatus.SENT)
    wo.ts_sent = ts_sent
    om.persist(wo)
    om._state.working_ask = wo


# ---------------------------------------------------------------- ACKED
def test_reconcile_skips_gone_when_ts_ack_after_rest_dispatch() -> None:
    """ACKED branch: freshly-acked order must not be marked gone."""
    _s, path, state, om = _setup()
    try:
        # Dispatch happened 50 ms ago; the order acked 25 ms ago — so
        # ts_ack is AFTER the snapshot's earliest possible build time.
        now = datetime.now(timezone.utc)
        rest_dispatch = now - timedelta(milliseconds=50)
        ts_ack = now - timedelta(milliseconds=25)
        _seed_acked_ask(om, local_oid=12345, ts_ack=ts_ack)
        assert state.working_ask is not None
        assert state.working_ask.status == OrderStatus.ACKED

        changed = om._reconcile_side(
            Side.SELL,
            remote=None,
            rest_request_dispatched_at=rest_dispatch,
        )
        # Guard fired → no state change reported.
        assert changed is False
        # Slot retained — order is NOT marked gone.
        assert state.working_ask is not None
        assert state.working_ask.status == OrderStatus.ACKED
        # Telemetry counter incremented.
        assert state.reconcile_skip_snapshot_stale_total == 1
    finally:
        path.unlink(missing_ok=True)


def test_reconcile_skips_gone_when_ts_ack_equals_rest_dispatch() -> None:
    """Boundary: equal timestamps must also skip (snapshot can't have it)."""
    _s, path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        # ts_ack EXACTLY equals rest_dispatch — still a race, skip.
        ts_ack = now - timedelta(milliseconds=10)
        rest_dispatch = ts_ack
        _seed_acked_ask(om, local_oid=12346, ts_ack=ts_ack)

        changed = om._reconcile_side(
            Side.SELL,
            remote=None,
            rest_request_dispatched_at=rest_dispatch,
        )
        assert changed is False
        assert state.working_ask is not None
        assert state.working_ask.status == OrderStatus.ACKED
        assert state.reconcile_skip_snapshot_stale_total == 1
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------- SENT
def test_reconcile_skips_gone_when_ts_sent_after_rest_dispatch() -> None:
    """SENT branch: freshly-sent (not yet acked) order must not be marked gone."""
    _s, path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        rest_dispatch = now - timedelta(milliseconds=50)
        ts_sent = now - timedelta(milliseconds=20)
        _seed_sent_ask(om, local_oid=22222, ts_sent=ts_sent)
        # Clear the cloid so the SENT-cloid-resolution branch is skipped
        # and we reach the gone_on_exchange decision point.
        om._state.working_ask.client_order_id = None  # type: ignore[union-attr]

        changed = om._reconcile_side(
            Side.SELL,
            remote=None,
            rest_request_dispatched_at=rest_dispatch,
        )
        assert changed is False
        assert state.working_ask is not None
        assert state.working_ask.status == OrderStatus.SENT
        assert state.reconcile_skip_snapshot_stale_total == 1
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------- Inverse
def test_reconcile_still_marks_gone_when_ts_ack_predates_rest_dispatch() -> None:
    """Inverse: order acked BEFORE REST dispatch is a legitimate orphan —
    the existing gone_on_exchange path must still fire (the fix MUST NOT
    suppress every reconcile, only the freshly-acked race window)."""
    _s, path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        # Order acked 5 seconds ago; REST dispatched 100 ms ago. The
        # snapshot SHOULD have contained the order; the fact that
        # remote=None means OKX really did lose it.
        rest_dispatch = now - timedelta(milliseconds=100)
        ts_ack = now - timedelta(seconds=5)
        _seed_acked_ask(om, local_oid=33333, ts_ack=ts_ack)

        # cancel_order_by_cloid may be invoked downstream; give it a
        # benign response so we don't crash.
        om._client.cancel_order.return_value = {  # type: ignore[attr-defined]
            "status": "ok",
            "response": {"type": "cancel", "data": {"statuses": ["success"]}},
        }
        om._client.cancel_order_by_cloid.return_value = {  # type: ignore[attr-defined]
            "status": "ok",
            "response": {"type": "cancel", "data": {"statuses": ["success"]}},
        }

        om._reconcile_side(
            Side.SELL,
            remote=None,
            rest_request_dispatched_at=rest_dispatch,
        )
        # Slot cleared — legitimate gone_on_exchange fired.
        assert state.working_ask is None
        # Telemetry counter NOT incremented (the snapshot-stale guard
        # didn't fire; the legitimate path did).
        assert state.reconcile_skip_snapshot_stale_total == 0
    finally:
        path.unlink(missing_ok=True)


# ------------------------------------------------- Backward compatibility
def test_reconcile_works_without_rest_dispatch_arg() -> None:
    """Calling _reconcile_side WITHOUT the new kwarg must still work
    (backward compat with any code paths that haven't been updated to
    pass the timestamp yet). Falls through to the existing behaviour."""
    _s, path, state, om = _setup()
    try:
        now = datetime.now(timezone.utc)
        ts_ack = now - timedelta(seconds=2)
        _seed_acked_ask(om, local_oid=44444, ts_ack=ts_ack)
        om._client.cancel_order.return_value = {  # type: ignore[attr-defined]
            "status": "ok",
            "response": {"type": "cancel", "data": {"statuses": ["success"]}},
        }
        om._client.cancel_order_by_cloid.return_value = {  # type: ignore[attr-defined]
            "status": "ok",
            "response": {"type": "cancel", "data": {"statuses": ["success"]}},
        }

        # No rest_request_dispatched_at — guard inactive, legitimate
        # gone_on_exchange path runs.
        om._reconcile_side(Side.SELL, remote=None)
        assert state.working_ask is None
        assert state.reconcile_skip_snapshot_stale_total == 0
    finally:
        path.unlink(missing_ok=True)
