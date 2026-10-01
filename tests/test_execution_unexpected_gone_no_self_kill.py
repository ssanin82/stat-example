"""v1.4.38 — cancel classifier completeness on `unexpected_gone`.

The v1.3.120 OKX response interpreter introduced ``unexpected_gone``
for cancel responses where the order is gone but NOT via fill (sCode
51400 / 51401 / 51503). The single-cancel HTTP path
(``_cancel_http_transport``) honors this by bumping a dedicated
counter ``cancel_unexpected_gone_total`` and NOT bumping
``execution_errors`` — see comments in ``app/exchange/okx_responses.py``
near line 109 for the rationale ("cleanup-cancel-all races would
otherwise self-kill on sustained occurrence; 2026-05-05 incident").

But two SIBLING cancel paths missed this rule and continued to bump
``execution_errors`` on every ``unexpected_gone``:

1. ``_execute_cancel_batch_intents`` — the batch-cancel response
   handler at ``app/execution.py:1313`` had ``elif kind ==
   "benign_missing": ... else: bump_execution_errors("cancel_http
   _exchange_reject")`` — falling ``unexpected_gone`` into the error
   bucket.
2. ``_orphan_remote_cancel`` — the orphan-cancel path at
   ``app/execution.py:4442`` had ``ok = kind in ("success",
   "benign_missing")`` — ``unexpected_gone`` falls outside ok and
   bumps ``orphan_cancel_exchange_reject``.

Pre-v1.4.38 these two bugs were latent. The v1.4.33 + v1.4.35
cancel-pool routing fix (every cancel now flows through
``_execute_cancel_batch_intents`` instead of the single-cancel
path) exposed bug (1). The v1.4.34 ``hydrated_from_exchange`` flag
(removed an accidental fast-cancel-cooldown rate-limiter on the
orphan-cancel loop) increased the orphan-cancel rate, exposing
bug (2). Production snapshot
``snapshots/v1.4.37-260518-075317-prod.okx.ton.usdt.perp/`` shows
the bot self-killed at 66 errors in 5 min (threshold 20):
``orphan_cancel_exchange_reject: 56`` + ``cancel_http_exchange
_reject: 10``, all underlying ``okx_row_51400``.

v1.4.38 fix: both paths mirror the single-cancel HTTP path's
handling — bump ``cancel_unexpected_gone_total`` (visibility),
DON'T bump ``execution_errors`` (no self-kill on the venue's
"order already gone" responses, which are not bot errors).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": "sqlite:///"
            + (
                Path(tempfile.gettempdir())
                / f"mm_uxgone_{os.getpid()}_{uuid.uuid4().hex}.db"
            ).as_posix(),
        }
    )


def _setup() -> tuple[OrderManager, Storage, BotState, Path]:
    s = _settings()
    db_path = Path(s.database_url.split("sqlite:///", 1)[-1])
    db_path.unlink(missing_ok=True)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    # Wire the real OKX cancel interpreter on the mock so the
    # classifier semantics under test (``unexpected_gone`` mapping
    # of OKX sCode 51400 / 51401 / 51503) actually apply. Without
    # this, ``OrderManager._interpret_cancel_response`` would fall
    # back to the Hyperliquid interpreter (since the MagicMock's
    # ``interpret_cancel_response`` is itself a MagicMock and
    # ``isinstance(parsed, tuple)`` returns False), and an
    # OKX-shaped response would mis-classify as ``transport``.
    from app.exchange.okx_responses import interpret_okx_cancel_response
    client.interpret_cancel_response = interpret_okx_cancel_response
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    return om, storage, state, db_path


# ---------------------------------------------------------------------------
# Batch-cancel path (_execute_cancel_batch_intents)
# ---------------------------------------------------------------------------


def _make_acked_wo(symbol: str, side: Side, oid: int, cloid: str) -> WorkingOrder:
    """Build a CANCEL_PENDING working order ready to be batch-cancelled."""
    now = utc_now()
    wo = WorkingOrder(
        order_id_local=f"local-{uuid.uuid4().hex[:8]}",
        order_id_exchange=oid,
        client_order_id=cloid,
        symbol=symbol,
        side=side,
        price=1.95,
        size=3.0,
        post_only=True,
        status=OrderStatus.CANCEL_PENDING,
        ts_created=now,
        ts_sent=now,
        ts_ack=now,
        ts_cancel_requested=now,
        ts_cancel_sent=now,
    )
    return wo


def test_batch_cancel_unexpected_gone_does_not_bump_execution_errors() -> None:
    """v1.4.38 fix: a batch-cancel response row with sCode=51400
    classifies as ``unexpected_gone`` — the bot must bump
    ``cancel_unexpected_gone_total`` but MUST NOT bump
    ``execution_errors``. Pre-v1.4.38 this row fell into the
    ``else`` bucket and bumped ``cancel_http_exchange_reject``,
    which under load (v1.4.33 routing change → every cancel hits
    this path) drove the v1.4.37 self-kill."""
    from app.execution import CancelTransportIntent

    om, storage, state, db_path = _setup()
    sym = om._settings.symbol
    try:
        wo = _make_acked_wo(sym, Side.BUY, oid=1001, cloid="cloid-uxg-1")
        state.set_working_order(Side.BUY, 0, wo)

        # Mock the batch-cancel to return an OKX-shaped response where
        # the row carries sCode 51400 (order doesn't exist) — the
        # ``unexpected_gone`` case.
        om._client.cancel_batch_orders.side_effect = lambda _sym, _refs: {
            "code": "1",  # top "1" since a row failed
            "msg": "",
            "data": [
                {
                    "sCode": "51400",
                    "sMsg": "Order cancellation failed as the order "
                    "has been filled, canceled or does not exist.",
                    "ordId": "1001",
                    "clOrdId": "cloid-uxg-1",
                }
            ],
        }

        errors_before = state.execution_errors_window_snapshot(300.0)["total"]
        unexpected_before = state.cancel_unexpected_gone_total

        intent = CancelTransportIntent(
            wo.order_id_local,
            Side.BUY,
            wo.cancel_transport_seq,
            __import__("time").monotonic(),
        )
        om._execute_cancel_batch_intents([intent, intent])  # ≥2 → batch path

        errors_after = state.execution_errors_window_snapshot(300.0)
        # KEY ASSERTION: execution_errors did NOT climb.
        assert errors_after["total"] == errors_before, (
            f"v1.4.38 regression: ``unexpected_gone`` in the batch-"
            f"cancel response bumped execution_errors "
            f"({errors_before} → {errors_after['total']}). The bot "
            f"would self-kill on sustained occurrence under load. "
            f"Sources after: {errors_after['sources']}"
        )
        # AND the dedicated counter DID climb.
        assert state.cancel_unexpected_gone_total > unexpected_before, (
            f"v1.4.38: cancel_unexpected_gone_total must bump on "
            f"unexpected_gone (was {unexpected_before}, now "
            f"{state.cancel_unexpected_gone_total})"
        )
    finally:
        db_path.unlink(missing_ok=True)


def test_batch_cancel_real_error_still_bumps_execution_errors() -> None:
    """v1.4.38 negative control: a GENUINE error row (e.g. sCode
    51008 = insufficient margin, not a venue "gone" response) MUST
    still bump ``execution_errors``. This pins that the v1.4.38
    fix is a NARROW carve-out for ``unexpected_gone`` and doesn't
    disable error detection wholesale."""
    from app.execution import CancelTransportIntent

    om, storage, state, db_path = _setup()
    sym = om._settings.symbol
    try:
        wo = _make_acked_wo(sym, Side.BUY, oid=2001, cloid="cloid-err-1")
        state.set_working_order(Side.BUY, 0, wo)

        # 51008 = insufficient balance — a real error, not "gone".
        om._client.cancel_batch_orders.side_effect = lambda _sym, _refs: {
            "code": "1",
            "msg": "",
            "data": [
                {
                    "sCode": "51008",
                    "sMsg": "Insufficient margin balance for cancel",
                    "ordId": "2001",
                    "clOrdId": "cloid-err-1",
                }
            ],
        }

        errors_before = state.execution_errors_window_snapshot(300.0)["total"]

        intent = CancelTransportIntent(
            wo.order_id_local,
            Side.BUY,
            wo.cancel_transport_seq,
            __import__("time").monotonic(),
        )
        om._execute_cancel_batch_intents([intent, intent])

        errors_after = state.execution_errors_window_snapshot(300.0)
        # KEY ASSERTION: a real error still bumps.
        assert errors_after["total"] > errors_before, (
            f"v1.4.38 over-reach: a genuine (non-``unexpected_gone``) "
            f"error row did NOT bump execution_errors "
            f"({errors_before} → {errors_after['total']}). The fix "
            f"should narrowly exempt ``unexpected_gone`` only."
        )
        assert "cancel_http_exchange_reject" in errors_after["sources"], (
            f"genuine errors should still tag as cancel_http_exchange_reject; "
            f"got sources {errors_after['sources']}"
        )
    finally:
        db_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Orphan-cancel path (_orphan_remote_cancel via _sync_open_orders_impl)
# ---------------------------------------------------------------------------


def test_orphan_cancel_unexpected_gone_does_not_bump_execution_errors() -> None:
    """v1.4.38 fix: when an orphan cancel returns sCode=51400
    (order doesn't exist), the orphan-cancel path must classify
    that as confirmation that the order is gone — bump
    ``cancel_unexpected_gone_total`` but NOT
    ``orphan_cancel_exchange_reject``.

    Rationale: for an orphan cancel, "the order is gone" IS the
    desired outcome. We wanted it gone. It's gone. That's a
    successful reconcile, not an error. Pre-v1.4.38 this bumped
    ``orphan_cancel_exchange_reject`` and drove the v1.4.37 self-
    kill (56 of 66 errors in the kill payload).

    Test calls ``_cancel_orphan_remote_order`` directly to exercise
    the classifier in isolation. The outer
    ``_sync_open_orders_impl`` path that USES this function has
    a lot of pre-conditions (dup detection, side-mismatch
    handling, dedup TTL); the unit-level test bypasses those so
    the classifier semantics are pinned without coupling to the
    reconcile orchestration.
    """
    om, storage, state, db_path = _setup()
    sym = om._settings.symbol
    try:
        # 51400 single-cancel response shape. The orphan-cancel path
        # uses ``cancel_order`` (oid-based) or ``cancel_order_by_cloid``
        # — both share the same response shape via the OKX V5 single-
        # cancel endpoint.
        ok_response_51400 = {
            "code": "1",
            "msg": "",
            "data": [
                {
                    "sCode": "51400",
                    "sMsg": "Order cancellation failed as the order "
                    "has been filled, canceled or does not exist.",
                    "ordId": "9001",
                }
            ],
        }
        om._client.cancel_order.return_value = ok_response_51400
        om._client.cancel_order_by_cloid.return_value = ok_response_51400

        errors_before = state.execution_errors_window_snapshot(300.0)["total"]
        unexpected_before = state.cancel_unexpected_gone_total

        # Direct invocation — bypass the reconcile orchestration so
        # the classifier semantics are tested in isolation.
        ok = om._cancel_orphan_remote_order(
            symbol=sym,
            oid=9001,
            cloid="cloid-orphan-uxg",
            reason="test_unexpected_gone",
        )

        # The function must report success: for an orphan cancel,
        # "the order is gone" is the desired outcome.
        assert ok is True, (
            "v1.4.38: orphan-cancel of an already-gone order must "
            "return ok=True (the order being gone IS the desired "
            "outcome for an orphan cancel)"
        )

        errors_after = state.execution_errors_window_snapshot(300.0)
        # KEY ASSERTION 1: orphan-cancel of an already-gone order
        # does NOT bump execution_errors.
        assert errors_after["total"] == errors_before, (
            f"v1.4.38 regression: orphan-cancel ``unexpected_gone`` "
            f"bumped execution_errors ({errors_before} → "
            f"{errors_after['total']}). For orphan cancels, the order "
            f"being gone IS the desired outcome — should not count as "
            f"an error. Sources: {errors_after['sources']}"
        )
        # KEY ASSERTION 2: the dedicated counter climbed for
        # observability.
        assert state.cancel_unexpected_gone_total > unexpected_before, (
            f"v1.4.38: cancel_unexpected_gone_total must bump on "
            f"orphan-cancel unexpected_gone too (was "
            f"{unexpected_before}, now "
            f"{state.cancel_unexpected_gone_total})"
        )
    finally:
        db_path.unlink(missing_ok=True)


def test_orphan_cancel_real_error_still_bumps_execution_errors() -> None:
    """v1.4.38 orphan-cancel negative control: a non-``unexpected_
    gone`` error response (e.g. sCode 51008) MUST still bump
    ``orphan_cancel_exchange_reject``. The fix is narrow: only
    ``unexpected_gone`` is exempted, because that case IS the
    desired outcome for an orphan cancel."""
    om, storage, state, db_path = _setup()
    sym = om._settings.symbol
    try:
        # 51008 = insufficient margin — a real error, not "gone".
        error_response = {
            "code": "1",
            "msg": "",
            "data": [
                {
                    "sCode": "51008",
                    "sMsg": "Insufficient margin balance for cancel",
                    "ordId": "9002",
                }
            ],
        }
        om._client.cancel_order.return_value = error_response

        errors_before = state.execution_errors_window_snapshot(300.0)["total"]

        ok = om._cancel_orphan_remote_order(
            symbol=sym,
            oid=9002,
            cloid="cloid-err-2",
            reason="test_real_error",
        )

        # A real error should NOT count as ok.
        assert ok is False, (
            "v1.4.38 over-reach: a real (non-``unexpected_gone``) "
            "cancel error must NOT report ok=True from the orphan "
            "path"
        )

        errors_after = state.execution_errors_window_snapshot(300.0)
        # Real errors DO bump.
        assert errors_after["total"] > errors_before, (
            f"v1.4.38 over-reach: a genuine error did NOT bump "
            f"execution_errors ({errors_before} → "
            f"{errors_after['total']}). The fix should narrowly "
            f"exempt ``unexpected_gone`` only."
        )
        assert "orphan_cancel_exchange_reject" in errors_after["sources"], (
            f"genuine orphan-cancel errors should still tag as "
            f"orphan_cancel_exchange_reject; got sources "
            f"{errors_after['sources']}"
        )
    finally:
        db_path.unlink(missing_ok=True)
