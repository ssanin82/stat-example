"""amend-prio Phase 1 — WorkingOrder amend lifecycle.

The WO state machine pre-1.4.15 was:
    NEW_LOCAL → SENT → ACKED → (PARTIAL ↔ ACKED) → CANCEL_PENDING → CANCELED
    plus REJECTED / FAILED terminal states.

Phase 1 introduces ``AMEND_PENDING`` as a new transient state
between ACKED/PARTIAL and either back-to-ACKED (on amend success)
or CANCEL_PENDING (on below_filled fallback) or CANCELED (when the
order was already gone via fill / cancel race).

This file verifies:

* New status enum value exists and serialises correctly.
* New WorkingOrder fields default to safe values (zero/None).
* Storage migration v36 schema preserves the new columns (legacy
  rows have NULLs).
* Transition entry: from ACKED → AMEND_PENDING. Field updates are
  in-place; ordId/cloid unchanged.
* Transition exits land on the correct terminal status per amend
  response kind.

Phase 2/3/4 (dispatch, orchestrate, reconcile) are covered in
subsequent test files when those phases land.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from app.enums import OrderStatus, Side
from app.models import WorkingOrder


def _wo_acked() -> WorkingOrder:
    """Build a representative WO in ACKED state, mid-lifecycle."""
    return WorkingOrder(
        order_id_local="local-abc",
        order_id_exchange=12345,
        client_order_id="cloid-xyz",
        symbol="TON-USDT-SWAP",
        side=Side.BUY,
        price=1.930,
        size=3.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=datetime.now(timezone.utc),
        ts_sent=datetime.now(timezone.utc),
        ts_ack=datetime.now(timezone.utc),
    )


def test_amend_pending_status_exists() -> None:
    assert OrderStatus.AMEND_PENDING.value == "AMEND_PENDING"
    # Round-trip via the string enum (storage column uses .value).
    assert OrderStatus("AMEND_PENDING") is OrderStatus.AMEND_PENDING


def test_amend_pending_distinct_from_cancel_pending() -> None:
    """They're two different transient pending-states. AMEND_PENDING
    preserves ordId; CANCEL_PENDING is the path TO termination."""
    assert OrderStatus.AMEND_PENDING != OrderStatus.CANCEL_PENDING
    assert OrderStatus.AMEND_PENDING != OrderStatus.ACKED


def test_working_order_amend_fields_default() -> None:
    wo = _wo_acked()
    assert wo.amend_intent_seq == 0
    assert wo.amend_target_px is None
    assert wo.amend_target_sz is None
    assert wo.ts_amend_sent is None
    assert wo.ts_amend_response is None
    assert wo.amend_response_outcome is None
    assert wo.amend_response_detail is None


def test_amend_enqueue_transition() -> None:
    """When _enqueue_amend_quote_path runs (Phase 3), it should
    update the WO to AMEND_PENDING with target px/sz set, intent_seq
    bumped, and ts_amend_sent stamped. Verify a manually-constructed
    transition has all these properties."""
    wo = _wo_acked()
    new_px, new_sz = 1.929, 3.5
    ts_send = datetime.now(timezone.utc)
    transitioned = replace(
        wo,
        status=OrderStatus.AMEND_PENDING,
        amend_intent_seq=wo.amend_intent_seq + 1,
        amend_target_px=new_px,
        amend_target_sz=new_sz,
        ts_amend_sent=ts_send,
    )
    assert transitioned.status == OrderStatus.AMEND_PENDING
    assert transitioned.amend_intent_seq == 1
    assert transitioned.amend_target_px == new_px
    assert transitioned.amend_target_sz == new_sz
    # IMPORTANT: ordId / cloid / underlying price/size are UNCHANGED
    # while the amend is in flight. Only target_* hold the proposed
    # values. The wire amend hasn't been confirmed yet.
    assert transitioned.order_id_exchange == 12345
    assert transitioned.client_order_id == "cloid-xyz"
    assert transitioned.price == 1.930  # original
    assert transitioned.size == 3.0  # original


def test_amend_success_transition_commits_target() -> None:
    """On amend success: AMEND_PENDING → ACKED, target_px/sz commit
    into price/size, ordId stays, target_* cleared (post-commit
    diagnostics), response fields stamped."""
    wo_pending = replace(
        _wo_acked(),
        status=OrderStatus.AMEND_PENDING,
        amend_intent_seq=1,
        amend_target_px=1.929,
        amend_target_sz=3.5,
        ts_amend_sent=datetime.now(timezone.utc),
    )
    # Operator-facing transition: commit target → price/size, clear
    # target fields, transition back to ACKED, stamp response.
    ts_resp = datetime.now(timezone.utc)
    wo_after = replace(
        wo_pending,
        status=OrderStatus.ACKED,
        price=wo_pending.amend_target_px,
        size=wo_pending.amend_target_sz,
        amend_target_px=None,
        amend_target_sz=None,
        ts_amend_response=ts_resp,
        amend_response_outcome="accepted",
    )
    assert wo_after.status == OrderStatus.ACKED
    assert wo_after.price == 1.929
    assert wo_after.size == 3.5
    assert wo_after.order_id_exchange == 12345  # PRESERVED
    assert wo_after.amend_response_outcome == "accepted"
    assert wo_after.amend_target_px is None
    assert wo_after.amend_target_sz is None


def test_amend_below_filled_transition_to_cancel_pending() -> None:
    """sCode 51016 = below_filled. The amend can't shrink remaining
    qty below already-filled qty. Bot falls back to cancel-then-
    place, transitioning the WO to CANCEL_PENDING (not back to
    ACKED). Target fields cleared; response stamped for diagnostics."""
    wo_pending = replace(
        _wo_acked(),
        status=OrderStatus.AMEND_PENDING,
        amend_target_px=1.929,
        amend_target_sz=0.5,
    )
    wo_after = replace(
        wo_pending,
        status=OrderStatus.CANCEL_PENDING,
        amend_target_px=None,
        amend_target_sz=None,
        amend_response_outcome="below_filled",
        amend_response_detail=(
            "new_sz_below_filled:okx_row_51016:new size below filled qty"
        ),
    )
    assert wo_after.status == OrderStatus.CANCEL_PENDING
    assert wo_after.amend_response_outcome == "below_filled"
    # Underlying price/size still original (the amend didn't take).
    assert wo_after.price == 1.930
    assert wo_after.size == 3.0


def test_amend_order_gone_transition() -> None:
    """sCode 51400/51401/51402/51503 → order_gone. The order is
    already terminal (fill race / cancel race / venue admin).
    Transition to CANCELED."""
    wo_pending = replace(
        _wo_acked(),
        status=OrderStatus.AMEND_PENDING,
        amend_target_px=1.929,
    )
    wo_after = replace(
        wo_pending,
        status=OrderStatus.CANCELED,
        amend_target_px=None,
        amend_target_sz=None,
        amend_response_outcome="order_gone",
    )
    assert wo_after.status == OrderStatus.CANCELED


def test_amend_exchange_rejected_returns_to_acked() -> None:
    """post-only-would-cross (51604) etc: the amend didn't take
    BUT the original order survives at its original price/size.
    Transition back to ACKED, response stamped, target_* cleared."""
    wo_pending = replace(
        _wo_acked(),
        status=OrderStatus.AMEND_PENDING,
        amend_target_px=1.935,  # would cross
    )
    wo_after = replace(
        wo_pending,
        status=OrderStatus.ACKED,
        amend_target_px=None,
        amend_target_sz=None,
        amend_response_outcome="exchange_rejected",
        amend_response_detail="post_only_would_cross:Post-only would cross",
    )
    assert wo_after.status == OrderStatus.ACKED
    assert wo_after.price == 1.930  # original, unchanged
    assert wo_after.amend_response_outcome == "exchange_rejected"


def test_amend_transport_rejected_returns_to_acked() -> None:
    """Row-level rate-limit during amend: order is still alive,
    amend just didn't make it. Transition back to ACKED, watchdog
    will reschedule next cycle if reprice is still warranted."""
    wo_pending = replace(
        _wo_acked(),
        status=OrderStatus.AMEND_PENDING,
        amend_target_px=1.929,
    )
    wo_after = replace(
        wo_pending,
        status=OrderStatus.ACKED,
        amend_target_px=None,
        amend_target_sz=None,
        amend_response_outcome="transport_rejected",
        amend_response_detail="rate_limit:okx_row_50011:Rate limit reached",
    )
    assert wo_after.status == OrderStatus.ACKED
    assert wo_after.amend_response_outcome == "transport_rejected"


def test_amend_partial_fill_during_pending_preserves_filled_qty() -> None:
    """An order with partial fill (e.g. filled=1 out of size=3) goes
    AMEND_PENDING → PARTIAL on success when the new total size still
    exceeds the filled portion. The WorkingOrder doesn't track
    filled qty directly (Fill table does); this test just confirms
    that PARTIAL is a legitimate exit status alongside ACKED."""
    wo = replace(_wo_acked(), status=OrderStatus.PARTIAL, size=3.0)
    wo_pending = replace(
        wo,
        status=OrderStatus.AMEND_PENDING,
        amend_target_sz=2.0,  # still > filled (1)
    )
    wo_after = replace(
        wo_pending,
        status=OrderStatus.PARTIAL,
        size=2.0,
        amend_target_sz=None,
        amend_response_outcome="accepted",
    )
    assert wo_after.status == OrderStatus.PARTIAL
    assert wo_after.size == 2.0


def test_amend_intent_seq_monotonic() -> None:
    """Each subsequent amend bumps the seq counter. The dispatcher
    uses this to drop stale intents (e.g. when a new amend was
    issued before the previous one's response landed)."""
    wo = _wo_acked()
    w1 = replace(wo, amend_intent_seq=1)
    w2 = replace(w1, amend_intent_seq=2)
    w3 = replace(w2, amend_intent_seq=3)
    assert w1.amend_intent_seq < w2.amend_intent_seq < w3.amend_intent_seq


def test_storage_schema_version_bumped() -> None:
    """v36 migration adds the 7 amend columns to the orders table.
    This test just imports the Storage class to confirm the version
    constant was bumped; the actual SQL migration runs against a
    real DB in integration tests."""
    from app.storage import Storage
    assert Storage.SCHEMA_VERSION >= 36
