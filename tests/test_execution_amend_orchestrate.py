"""amend-prio Phase 3 (v1.4.16) — orchestrate decision + per-row
outcome handling.

Tests cover the predicate ``_amend_viable`` (knob, status, ordId,
cooldown, adapter capability) and the per-row outcome dispatch in
``_execute_amend_batch_intents`` (accepted / below_filled /
order_gone / exchange_rejected / transport_rejected).

The orchestrate-decision wiring is tested via a tight loop:
construct an ACKED WorkingOrder, call ``_amend_viable`` /
``_enqueue_amend_quote_path``, assert the dispatched intent has
``kind="amend"`` and the WO transitions to AMEND_PENDING with
``amend_target_*`` stamped.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import BestBidAsk, WorkingOrder
from app.quote_engine import FinalQuoteOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup(*, amend_enabled: bool = True) -> tuple[OrderManager, Path, Any]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_amendph3_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "OKX_AMEND_ON_REPRICE_ENABLED": amend_enabled,
            "STRICT_PLACE_UNCONFIRMED_KILL": False,
            # v1.4.160 Phase 2K.11: this test uses a synthetic
            # "post-only would cross" rejection where the market
            # state (best_ask=3001) does NOT actually cross the amend
            # target (2999.5). Phase 2K.11's favorable-exit predicate
            # would correctly conclude "touch is already past the
            # rejected price" and clear the cooldown immediately —
            # which would defeat THIS test's purpose (verifying that
            # arming sets the cooldown). Disable favorable-exit here
            # so the legacy arming-only behaviour is testable.
            # Phase 2K.11 favorable-exit itself is covered by
            # ``tests/test_phase2k11_post_only_cross_favorable_exit.py``.
            "POST_ONLY_CROSS_COOLDOWN_FAVORABLE_EXIT_ENABLED": False,
            # v1.4.169 Phase 2I: this test suite exercises back-to-back
            # amends on the same order to verify the intent-seq /
            # response handling. The new amend rate-defence guard
            # would correctly suppress those rapid amends (its job is
            # to stop the v1.4.102 244-amends-in-1-sec runaway), which
            # would defeat THIS test's purpose. Disable the flicker
            # gap + raise the per-order rate cap so the guard is
            # dormant during these unit tests. Phase 2I itself is
            # covered by ``tests/test_phase2i_amend_rate_defence.py``.
            "AMEND_TICK_FLICKER_MIN_MS": 0.0,
            "AMEND_PER_ORDER_MAX_PER_SEC": 200,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=3000.0,
        best_ask=3001.0,
        mid_price=3000.5,
        spread_bps=10.0,
        ts_local=utc_now(),
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    return om, path, client


def _wo_acked(om: OrderManager, *, ord_id: int = 12345) -> WorkingOrder:
    wo = WorkingOrder(
        order_id_local=f"local-{uuid.uuid4().hex[:8]}",
        order_id_exchange=ord_id,
        client_order_id="cl_" + uuid.uuid4().hex[:24],
        symbol=om._settings.symbol,
        side=Side.BUY,
        price=3000.0,
        size=0.02,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=datetime.now(timezone.utc),
        ts_sent=datetime.now(timezone.utc),
        ts_ack=datetime.now(timezone.utc),
    )
    # Drop the WO into state's working slot so executor lookups find it.
    with om._state._lock:
        om._state.set_working_order(Side.BUY, 0, wo)
    return wo


def _desired(price: float, size: float, side: Side = Side.BUY) -> FinalQuoteOrder:
    return FinalQuoteOrder(side=side, price=price, size=size)


# ----------------------------------------------------------------------
# _amend_viable predicate
# ----------------------------------------------------------------------


def test_amend_viable_when_enabled_acked_with_ord_id() -> None:
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om)
        assert om._amend_viable(wo, _desired(2999.0, 0.02)) is True
    finally:
        path.unlink(missing_ok=True)


def test_amend_not_viable_when_knob_off() -> None:
    om, path, _client = _setup(amend_enabled=False)
    try:
        wo = _wo_acked(om)
        assert om._amend_viable(wo, _desired(2999.0, 0.02)) is False
    finally:
        path.unlink(missing_ok=True)


def test_amend_not_viable_when_no_ord_id() -> None:
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om)
        wo.order_id_exchange = None
        assert om._amend_viable(wo, _desired(2999.0, 0.02)) is False
    finally:
        path.unlink(missing_ok=True)


def test_amend_not_viable_when_partial() -> None:
    """Phase 3 stays conservative: PARTIAL falls through to
    cancel-then-place. WO doesn't track filled qty so we can't
    pre-check OKX's below_filled rule."""
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om)
        wo.status = OrderStatus.PARTIAL
        assert om._amend_viable(wo, _desired(2999.0, 0.02)) is False
    finally:
        path.unlink(missing_ok=True)


def test_amend_not_viable_when_post_only_cross_cooldown_armed() -> None:
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om)
        om._arm_post_only_cross_cooldown(Side.BUY)
        assert om._amend_viable(wo, _desired(2999.0, 0.02)) is False
    finally:
        path.unlink(missing_ok=True)


def test_amend_not_viable_when_adapter_missing_method() -> None:
    """If the adapter doesn't expose ``amend_batch_orders``, the
    predicate refuses — the bot falls back to cancel-then-place
    rather than raising AttributeError later."""
    om, path, client = _setup(amend_enabled=True)
    try:
        # Remove the auto-mocked method.
        del client.amend_batch_orders
        wo = _wo_acked(om)
        assert om._amend_viable(wo, _desired(2999.0, 0.02)) is False
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# _enqueue_amend_quote_path
# ----------------------------------------------------------------------


def test_enqueue_amend_transitions_to_amend_pending() -> None:
    om, path, _client = _setup(amend_enabled=True)
    try:
        # Stop the dispatcher so the submitted intent stays in the
        # lane and we can inspect the intermediate AMEND_PENDING state
        # before the executor consumes it.
        om._outbound.stop()
        wo = _wo_acked(om)
        ok = om._enqueue_amend_quote_path(
            wo, _desired(2999.0, 0.03), trigger_reason="reprice_amend"
        )
        assert ok is True
        assert wo.status == OrderStatus.AMEND_PENDING
        assert wo.amend_target_px == 2999.0
        assert wo.amend_target_sz == 0.03
        assert wo.amend_intent_seq == 1
        # Original price/size unchanged while amend in flight.
        assert wo.price == 3000.0
        assert wo.size == 0.02
        # ordId / cloid preserved.
        assert wo.order_id_exchange == 12345
    finally:
        path.unlink(missing_ok=True)


def test_enqueue_amend_bumps_intent_seq_monotonically() -> None:
    om, path, _client = _setup(amend_enabled=True)
    try:
        om._outbound.stop()
        wo = _wo_acked(om)
        om._enqueue_amend_quote_path(wo, _desired(2999.0, 0.03))
        # Pretend response came back ACKED-OK then enqueue another amend.
        wo.status = OrderStatus.ACKED
        wo.amend_target_px = None
        wo.amend_target_sz = None
        om._enqueue_amend_quote_path(wo, _desired(2998.0, 0.04))
        assert wo.amend_intent_seq == 2
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# _execute_amend_batch_intents — per-row outcomes
# ----------------------------------------------------------------------


def _amend_pending(om: OrderManager, *, ord_id: int = 12345) -> WorkingOrder:
    wo = _wo_acked(om, ord_id=ord_id)
    wo.amend_intent_seq = 1
    wo.amend_target_px = 2999.5
    wo.amend_target_sz = 0.03
    wo.status = OrderStatus.AMEND_PENDING
    return wo


def _amend_intent(wo: WorkingOrder):
    from app.outbound_dispatch import PlaceTransportIntent
    import time as time_mod
    return PlaceTransportIntent(
        wo_order_id_local=wo.order_id_local,
        side=wo.side,
        intent_seq=int(wo.amend_intent_seq),
        quote_cycle_id="amend",
        enqueued_mono=time_mod.monotonic(),
        level_idx=0,
        kind="amend",
    )


def test_amend_reqid_format_is_strictly_alphanumeric() -> None:
    """OKX V5 ``reqId`` regression — OKX accepts ONLY alphanumeric
    characters [a-zA-Z0-9] up to 32 chars. Two earlier deploys
    confirmed this empirically:
      v1.4.17: ``{ordId}-{seq}``  → sCode 51000 (dash rejected)
      v1.4.18: ``a{ordId}_{seq}`` → sCode 51000 (underscore rejected)
    Final: ``a{ordId}s{seq}`` — letter ``s`` as separator.

    This test pins:
      - alphanumeric-only character set (no -, _, ., or other punctuation)
      - leading ``a`` keeps the value non-numeric
      - length ≤ 32 chars
      - deterministic (same wo + same seq → same reqId)
    """
    import re
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om, ord_id=3574241245467189248)
        intent = _amend_intent(wo)
        client.amend_batch_orders.return_value = {
            "code": "0", "msg": "", "data": [
                {"sCode": "0", "ordId": str(wo.order_id_exchange)},
            ],
        }
        om._execute_amend_batch_intents([intent])
        assert client.amend_batch_orders.call_count == 1
        symbol, amends = client.amend_batch_orders.call_args.args
        assert len(amends) == 1
        req_id = amends[0]["req_id"]
        # STRICT alphanumeric — no dashes, no underscores, no dots.
        assert re.fullmatch(r"[a-zA-Z0-9]{1,32}", req_id), (
            f"reqId {req_id!r} must match OKX V5 regex "
            "[a-zA-Z0-9]{1,32} — OKX rejects any other character "
            "with sCode 51000 (verified empirically v1.4.17 and v1.4.18)"
        )
        # Specific format pin.
        assert req_id == f"a{wo.order_id_exchange}s1"
        # Ensure we never sneak in punctuation via field changes.
        for forbidden in "-_.,/+: ":
            assert forbidden not in req_id, (
                f"reqId must not contain {forbidden!r}; got {req_id!r}"
            )
    finally:
        path.unlink(missing_ok=True)


def test_amend_outcome_accepted_commits_target() -> None:
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        intent = _amend_intent(wo)
        # OKX-shaped success response.
        client.amend_batch_orders.return_value = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "sCode": "0",
                    "sMsg": "",
                    "ordId": str(wo.order_id_exchange),
                    "reqId": f"a{wo.order_id_exchange}s1",
                    "clOrdId": wo.client_order_id,
                }
            ],
        }
        om._execute_amend_batch_intents([intent])
        assert wo.status == OrderStatus.ACKED
        assert wo.price == 2999.5  # committed
        assert wo.size == 0.03  # committed
        assert wo.amend_target_px is None
        assert wo.amend_target_sz is None
        assert wo.amend_response_outcome == "accepted"
        assert wo.order_id_exchange == 12345  # preserved
    finally:
        path.unlink(missing_ok=True)


def test_amend_outcome_below_filled_falls_back_to_cancel() -> None:
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        intent = _amend_intent(wo)
        client.amend_batch_orders.return_value = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "sCode": "51016",
                    "sMsg": "new size below filled qty",
                    "reqId": f"a{wo.order_id_exchange}s1",
                    "clOrdId": wo.client_order_id,
                }
            ],
        }
        om._execute_amend_batch_intents([intent])
        # Transitions to CANCEL_PENDING via the fallback cancel-enqueue.
        assert wo.status == OrderStatus.CANCEL_PENDING
        assert wo.amend_target_px is None
        assert wo.amend_target_sz is None
        assert wo.amend_response_outcome == "below_filled"
        # Original underlying price/size unchanged.
        assert wo.price == 3000.0
        assert wo.size == 0.02
    finally:
        path.unlink(missing_ok=True)


def test_amend_outcome_order_gone_transitions_canceled() -> None:
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        intent = _amend_intent(wo)
        client.amend_batch_orders.return_value = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "sCode": "51400",
                    "sMsg": "order does not exist",
                    "reqId": f"a{wo.order_id_exchange}s1",
                    "clOrdId": wo.client_order_id,
                }
            ],
        }
        om._execute_amend_batch_intents([intent])
        assert wo.status == OrderStatus.CANCELED
        assert wo.amend_response_outcome == "order_gone"
    finally:
        path.unlink(missing_ok=True)


def test_amend_outcome_post_only_cross_arms_cooldown() -> None:
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        intent = _amend_intent(wo)
        client.amend_batch_orders.return_value = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "sCode": "51604",
                    "sMsg": "Post-only would cross",
                    "reqId": f"a{wo.order_id_exchange}s1",
                    "clOrdId": wo.client_order_id,
                }
            ],
        }
        om._execute_amend_batch_intents([intent])
        # Reverts to ACKED — original survives unchanged.
        assert wo.status == OrderStatus.ACKED
        assert wo.price == 3000.0  # original
        assert wo.size == 0.02  # original
        assert wo.amend_target_px is None
        assert wo.amend_target_sz is None
        assert wo.amend_response_outcome == "exchange_rejected"
        # Cooldown armed.
        assert om._post_only_cross_cooldown_active(Side.BUY) is True
    finally:
        path.unlink(missing_ok=True)


def test_amend_outcome_transport_rejected_reverts_to_acked() -> None:
    """Row-level rate-limit during amend: original survives, revert
    to ACKED, next cycle re-evaluates."""
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        intent = _amend_intent(wo)
        client.amend_batch_orders.return_value = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "sCode": "50011",
                    "sMsg": "rate_limit",
                    "reqId": f"a{wo.order_id_exchange}s1",
                    "clOrdId": wo.client_order_id,
                }
            ],
        }
        om._execute_amend_batch_intents([intent])
        assert wo.status == OrderStatus.ACKED
        assert wo.price == 3000.0
        assert wo.size == 0.02
        assert wo.amend_response_outcome == "transport_rejected"
        # No cooldown armed.
        assert om._post_only_cross_cooldown_active(Side.BUY) is False
    finally:
        path.unlink(missing_ok=True)


def test_amend_stale_intent_seq_drops() -> None:
    """An intent whose intent_seq is older than the WO's current
    amend_intent_seq must drop without firing the executor — that
    seq represents a superseded amend (new one in flight)."""
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        # Build intent with seq=1, then bump the WO to seq=2.
        intent = _amend_intent(wo)
        wo.amend_intent_seq = 2
        om._execute_amend_batch_intents([intent])
        # Adapter call should not have been issued.
        assert client.amend_batch_orders.call_count == 0
        # WO still AMEND_PENDING (no state change).
        assert wo.status == OrderStatus.AMEND_PENDING
    finally:
        path.unlink(missing_ok=True)


def test_amend_adapter_attribute_error_reverts_to_acked() -> None:
    """If the venue adapter lacks amend_batch_orders at dispatch
    time (e.g. swapped venue), the executor reverts the WO without
    crashing."""
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        intent = _amend_intent(wo)
        client.amend_batch_orders.side_effect = AttributeError("no amend")
        om._execute_amend_batch_intents([intent])
        assert wo.status == OrderStatus.ACKED
        assert wo.amend_response_outcome == "exchange_rejected"
    finally:
        path.unlink(missing_ok=True)


def test_amend_transport_exception_reverts_to_acked() -> None:
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        intent = _amend_intent(wo)
        client.amend_batch_orders.side_effect = RuntimeError("network down")
        om._execute_amend_batch_intents([intent])
        assert wo.status == OrderStatus.ACKED
        assert wo.amend_response_outcome == "transport_rejected"
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# BUG-040 — price-changing amend re-bases the fill-age clock
# ----------------------------------------------------------------------


class _AgeProbeFill:
    """Minimal stand-in for app.models.Fill — only the attributes
    FillBucketAggregator.note_fill reads. Used to probe what age the
    aggregator would compute for a fill landing on ``oid`` at
    ``ts_fill``.
    """

    def __init__(self, *, oid: int, ts_fill: datetime) -> None:
        self.side = "BUY"
        self.order_id_exchange = oid
        self.ts_fill = ts_fill
        self.notional = 21.0
        self.markout_1s_bps = None
        self.markout_5s_bps = None
        self.quote_aggressiveness = "at_touch"


def test_amend_accepted_px_change_rebases_fill_age_clock() -> None:
    """BUG-040: an accepted amend that CHANGES THE PRICE must re-stamp
    the fill-bucket ack clock so ``quote_age_at_fill_ms`` measures from
    the LAST reprice, not from original placement.

    OKX amend-in-place preserves ``order_id_exchange``; the fill-bucket
    LRU is keyed by that oid. Without the re-stamp the cache keeps the
    ORIGINAL placement ts forever, so the computed age spans every
    reprice (the prod session that surfaced this ran ~12 amends/fill,
    pushing P99 quote_age_at_fill to ~5 s — the AT_TOUCH cap — even
    though the bot was actively repricing).
    """
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)  # ord_id=12345, price 3000 → target 2999.5
        # Seed the bucket cache with the ORIGINAL place ack 10 s ago.
        old_ack = utc_now() - timedelta(seconds=10)
        om._state.fill_buckets.note_ack(
            order_id_exchange=wo.order_id_exchange, ts_ack=old_ack
        )
        intent = _amend_intent(wo)
        client.amend_batch_orders.return_value = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "sCode": "0",
                    "sMsg": "",
                    "ordId": str(wo.order_id_exchange),
                    "reqId": f"a{wo.order_id_exchange}s1",
                    "clOrdId": wo.client_order_id,
                }
            ],
        }
        om._execute_amend_batch_intents([intent])
        assert wo.status == OrderStatus.ACKED
        assert wo.price == 2999.5  # price changed → re-base must fire
        assert wo.ts_ack is not None

        # A fill landing 300 ms after the reprice must read ~300 ms of
        # age (from the amend), NOT ~10.3 s (from original placement).
        probe = _AgeProbeFill(
            oid=wo.order_id_exchange,
            ts_fill=wo.ts_ack + timedelta(milliseconds=300),
        )
        age_ms = om._state.fill_buckets.note_fill(probe)
        assert age_ms is not None
        assert age_ms < 1000.0, (
            f"age {age_ms:.0f}ms should be ~300ms (from reprice), not "
            "~10300ms (from original placement) — BUG-040 re-base failed"
        )
    finally:
        path.unlink(missing_ok=True)


def test_amend_accepted_size_only_does_not_rebase_fill_age_clock() -> None:
    """BUG-040 gate: a size-only amend (price unchanged) must NOT reset
    the fill-age clock. The quote's price-exposure (toxicity) clock keeps
    running because the order is still resting at the same price — only a
    PRICE change re-bases.
    """
    om, path, client = _setup(amend_enabled=True)
    try:
        wo = _amend_pending(om)
        wo.amend_target_px = None  # size-only amend
        wo.amend_target_sz = 0.05
        old_ack = utc_now() - timedelta(seconds=10)
        om._state.fill_buckets.note_ack(
            order_id_exchange=wo.order_id_exchange, ts_ack=old_ack
        )
        intent = _amend_intent(wo)
        client.amend_batch_orders.return_value = {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "sCode": "0",
                    "sMsg": "",
                    "ordId": str(wo.order_id_exchange),
                    "reqId": f"a{wo.order_id_exchange}s1",
                    "clOrdId": wo.client_order_id,
                }
            ],
        }
        om._execute_amend_batch_intents([intent])
        assert wo.status == OrderStatus.ACKED
        assert wo.size == 0.05  # size committed
        assert wo.price == 3000.0  # price unchanged

        # Fill 300 ms later still reads ~10.3 s of age — the clock was
        # NOT reset because the price never changed.
        probe = _AgeProbeFill(
            oid=wo.order_id_exchange,
            ts_fill=utc_now() + timedelta(milliseconds=300),
        )
        age_ms = om._state.fill_buckets.note_fill(probe)
        assert age_ms is not None
        assert age_ms > 9000.0, (
            f"size-only amend must NOT reset the age clock; got "
            f"{age_ms:.0f}ms (expected ~10300ms from original placement)"
        )
    finally:
        path.unlink(missing_ok=True)
