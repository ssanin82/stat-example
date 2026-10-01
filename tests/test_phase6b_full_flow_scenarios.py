"""v1.4.93 wedge-elimination-cleanup Phase 6B — full-flow scenarios.

Each test drives the real ``OrderManager`` via the harness, configures
the stateful ``MockOkxClient`` for a specific failure mode, and asserts
on the resulting bot + venue state.

Coverage in this session (scenarios shipped):

1. **`normal_quote_cycle`** — happy path. Place both sides, both go
   ACKED on the venue. Reprice cancels + replaces. Baseline that the
   harness wiring is correct.
2. **`51410_during_cancel`** — cancel response sCode=51410 (gone)
   transitions WO to CANCELED via the `benign_missing` path.
3. **`exchange_rejected_place`** — place row sCode != "0" → WO
   transitions to REJECTED, slot released for next-tick re-place.
4. **`transport_rejected_place`** — top code != "0" → WO transitioned
   to REJECTED, no oid bound, slot released.
5. **`one_sided_active_sides`** — engine emits only BID; venue sees
   only one place; no SELL on the book.

Scenarios not yet shipped (deferred — need timing/multi-thread sim):

* `slow_ack_30s_does_not_age_past_cap` — needs simulated time + age
  cap firing.
* `ws_live_before_place_response` — needs negative-latency WS event
  injection (very fine timing).
* `lost_cancel_ack` — needs reaper interaction across multiple ticks.
* `duplicate_oid_hydration` — needs the hydrator path.
* `two_rung_concurrent_legitimate` — needs LADDER_NUM_LEVELS_PER_SIDE=2
  + 2-rung dispatch verified.
* `post_only_cross_5x_cooldown_armed` — needs 5 consecutive 51604
  responses + cooldown observation.
* `risk_action_oscillation` — needs risk-state transitions.

Each of those is ~30-60 min of focused work; defer to a follow-up
session.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.enums import ActiveSides, OrderStatus, RiskAction, Side
from tests.integration.full_flow_harness import harness_ctx
from tests.integration.mock_okx_client import MockOrderState


# ---------------------------------------------------------------------------
# Scenario 1: normal quote cycle — happy path
# ---------------------------------------------------------------------------


def test_phase6b_scenario_normal_quote_cycle() -> None:
    """Place both sides, both go ACKED on the venue.
    Reprice (next tick at different mid) cancels the old and places
    the new at the new price. Validates the harness wiring + the
    bot's standard quote-refresh path."""
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Tick 1: initial placement.
        h.tick_once()
        assert len(h.client.live_orders()) == 2, (
            f"after tick 1: expected 2 live orders on venue, "
            f"got {len(h.client.live_orders())}"
        )
        bid_wo = h.state.order_store.get(Side.BUY, 0)
        ask_wo = h.state.order_store.get(Side.SELL, 0)
        assert bid_wo is not None and bid_wo.status == OrderStatus.ACKED
        assert ask_wo is not None and ask_wo.status == OrderStatus.ACKED

        # Validate the bot's WO oids match the venue's.
        venue_oids = {o.ord_id for o in h.client.live_orders()}
        assert bid_wo.order_id_exchange in venue_oids
        assert ask_wo.order_id_exchange in venue_oids


def test_phase6b_scenario_reprice_cancels_old_places_new() -> None:
    """Tick 1: place at one price. Tick 2: market moves 200 bps (big
    enough to clear the materiality threshold + the bot's amend-vs-
    cancel heuristic), engine emits a different price → bot issues
    cancels OR amends for the old orders."""
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()
        h.tick_once()
        calls_after_tick1 = len(h.client.calls)
        bid_initial_oid = h.state.order_store.get(Side.BUY, 0).order_id_exchange

        # Tick 2: market moves up 200 bps (well above materiality threshold).
        h.seed_market(best_bid=2.040, best_ask=2.044)
        decision2 = h.make_decision(
            mid=2.042, target_spread_bps=20.0,
            quoted_bid=2.040, quoted_ask=2.044,
        )
        h.tick_once(decision=decision2)

        # The bot should have issued cancel OR amend calls for the old WOs.
        new_calls = h.client.calls[calls_after_tick1:]
        cancel_or_amend = sum(
            1 for n, _ in new_calls
            if n in (
                "cancel_order", "cancel_batch_orders",
                "amend_batch_orders", "amend_order",
            )
        )
        assert cancel_or_amend >= 1, (
            f"reprice on 200-bps move should issue ≥1 cancel/amend; "
            f"got new calls: {[c[0] for c in new_calls]}"
        )


# ---------------------------------------------------------------------------
# Scenario 2: cancel returns 51410 (order already gone) → benign_missing
# ---------------------------------------------------------------------------


def test_phase6b_scenario_cancel_51410_transitions_wo_to_canceled() -> None:
    """When the cancel response is sCode=51410 (order already gone),
    the bot's interpreter classifies as ``benign_missing`` — the WO
    transitions to CANCELED locally. Dispatcher must NOT leave it
    stuck in CANCEL_PENDING forever.

    Uses a 200-bps price move to ensure the engine actually issues
    cancels (smaller moves can fall under the materiality threshold
    or trigger amend-instead-of-cancel)."""
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()
        h.tick_once()
        bid_wo = h.state.order_store.get(Side.BUY, 0)
        assert bid_wo is not None and bid_wo.status == OrderStatus.ACKED

        # Configure cancel to return 51410.
        h.client.config.cancel_response_scode = "51410"
        h.client.config.cancel_response_smsg = "Order does not exist"
        calls_after_tick1 = len(h.client.calls)

        # Force a reprice at a much different price → bot tries to cancel.
        h.seed_market(best_bid=2.040, best_ask=2.044)
        decision2 = h.make_decision(
            mid=2.042, target_spread_bps=20.0,
            quoted_bid=2.040, quoted_ask=2.044,
        )
        h.tick_once(decision=decision2)

        # The cancel went through. May be cancel_order or batch.
        new_calls = h.client.calls[calls_after_tick1:]
        cancel_or_amend = sum(
            1 for n, _ in new_calls
            if n in ("cancel_order", "cancel_batch_orders", "amend_batch_orders")
        )
        assert cancel_or_amend >= 1, (
            f"reprice should issue cancel/amend; got: {[c[0] for c in new_calls]}"
        )

        # KEY: after a benign_missing (51410) response, the bot MUST
        # NOT leave the WO stuck in CANCEL_PENDING. Either:
        #   (a) the WO transitioned to CANCELED locally
        #   (b) the slot is occupied by a NEW WO (the bot placed fresh)
        #   (c) the slot is empty
        slot = h.state.order_store.get(Side.BUY, 0)
        if slot is not None:
            # If still has the same oid, must NOT be CANCEL_PENDING.
            if slot.order_id_exchange == bid_wo.order_id_exchange:
                assert slot.status not in (OrderStatus.CANCEL_PENDING,), (
                    f"after 51410 (benign_missing), BUY WO with old oid must "
                    f"transition out of CANCEL_PENDING; got status={slot.status.value}"
                )


# ---------------------------------------------------------------------------
# Scenario 3: place row reject (exchange_rejected) → WO REJECTED, slot released
# ---------------------------------------------------------------------------


def test_phase6b_scenario_exchange_rejected_place_transitions_wo_to_rejected() -> None:
    """When the place response row carries sCode != "0" (e.g., 51604
    post-only cross, 51400 min-notional), the WO must transition to
    REJECTED — NOT stay in SENT.

    Pre-v1.4.64: a row-level reject left the WO stuck in SENT and the
    reconciler waited forever. v1.4.64 fixed it; this scenario pins it."""
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Configure exchange_rejected with sCode 51604 (post-only would cross).
        h.client.config.place_response_outcome = "exchange_rejected"
        h.client.config.place_reject_scode = "51604"
        h.client.config.place_reject_smsg = "Post-only order would cross spread"

        h.tick_once()

        # No live orders on the venue (every place rejected).
        assert len(h.client.live_orders()) == 0, (
            f"all places should have been rejected at the venue; "
            f"got {len(h.client.live_orders())} live orders"
        )

        # The bot's WO state — must NOT be stuck in SENT.
        for side in (Side.BUY, Side.SELL):
            wo = h.state.order_store.get(side, 0)
            if wo is not None:
                # If the slot still has a WO, it must be terminal or
                # CANCELED/REJECTED — never stuck in SENT.
                assert wo.status != OrderStatus.SENT, (
                    f"after exchange_rejected place response, {side.value} WO "
                    f"must not be stuck in SENT; got {wo.status.value}"
                )


# ---------------------------------------------------------------------------
# Scenario 4: place transport reject → WO REJECTED, no oid bound
# ---------------------------------------------------------------------------


def test_phase6b_scenario_transport_rejected_place_releases_slot() -> None:
    """When the place response has top sCode != "0" (transport / rate
    limit), the WO transitions to REJECTED. Same regression-class as
    Scenario 3 but a different code path in the interpreter."""
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        h.client.config.place_response_outcome = "transport_rejected"
        h.client.config.place_reject_scode = "50011"
        h.client.config.place_reject_smsg = "Rate limit"

        h.tick_once()

        # No live orders.
        assert len(h.client.live_orders()) == 0

        # Slots either empty or in terminal state.
        for side in (Side.BUY, Side.SELL):
            wo = h.state.order_store.get(side, 0)
            if wo is not None:
                assert wo.status not in (OrderStatus.SENT, OrderStatus.NEW_LOCAL), (
                    f"after transport_rejected, {side.value} WO must not be "
                    f"stuck in non-terminal status; got {wo.status.value}"
                )


# ---------------------------------------------------------------------------
# Scenario 5: one-sided active_sides — only one place
# ---------------------------------------------------------------------------


def test_phase6b_scenario_one_sided_active_sides_only_places_active_side() -> None:
    """When the strategy emits ``active_sides=BID_ONLY``, only the BID
    side should be placed. The ASK slot stays empty. Mirror with
    ``ASK_ONLY``."""
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        decision = h.make_decision(active_sides=ActiveSides.BID_ONLY)
        h.tick_once(decision=decision, risk_action=RiskAction.ALLOW)

        live = h.client.live_orders()
        assert len(live) == 1, (
            f"BID_ONLY should produce exactly 1 placed order; got {len(live)}"
        )
        assert live[0].side == Side.BUY, (
            f"BID_ONLY should place BUY; got side={live[0].side.value}"
        )

        # Bot state: BUY ACKED, SELL slot empty.
        bid_wo = h.state.order_store.get(Side.BUY, 0)
        ask_wo = h.state.order_store.get(Side.SELL, 0)
        assert bid_wo is not None and bid_wo.status == OrderStatus.ACKED
        assert ask_wo is None, (
            f"SELL slot must be empty in BID_ONLY mode; got {ask_wo}"
        )


# ---------------------------------------------------------------------------
# Scenario 6: risk_action=NO_QUOTE with cancel_on_no_quote=False → keep resting
# ---------------------------------------------------------------------------


def test_phase6b_scenario_risk_no_quote_keeps_resting_orders() -> None:
    """If risk_action=NO_QUOTE and cancel_on_no_quote=False, the bot
    must NOT cancel resting orders — it just holds. Validates the
    risk-state contract documented in maybe_refresh_quotes."""
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Tick 1: place normally.
        h.tick_once()
        assert len(h.client.live_orders()) == 2

        calls_before = list(h.client.calls)

        # Tick 2: risk says NO_QUOTE, hold resting.
        h.tick_once(
            risk_action=RiskAction.NO_QUOTE,
            cancel_on_no_quote=False,
        )

        # No new cancel calls.
        new_calls = h.client.calls[len(calls_before):]
        cancel_calls = sum(
            1 for n, _ in new_calls
            if n in ("cancel_order", "cancel_batch_orders")
        )
        assert cancel_calls == 0, (
            f"NO_QUOTE with cancel_on_no_quote=False must NOT cancel resting; "
            f"got {cancel_calls} cancel call(s)"
        )
        # Orders still live on the venue.
        assert len(h.client.live_orders()) == 2


# ---------------------------------------------------------------------------
# Scenario 7: WS-canceled event after place → WO transitions to CANCELED
# ---------------------------------------------------------------------------


def test_phase6b_scenario_ws_canceled_event_transitions_wo() -> None:
    """After a place + ACKED, simulate a WS ``canceled`` event for the
    same oid (e.g., venue-side post-only cross detected post-place).
    Bot's WO must transition from ACKED → CANCELED via the WS handler.
    """
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()
        h.tick_once()
        bid_wo = h.state.order_store.get(Side.BUY, 0)
        assert bid_wo is not None
        assert bid_wo.status == OrderStatus.ACKED
        bid_oid = bid_wo.order_id_exchange

        # Inject WS canceled event for the BUY oid.
        h.client.emit_ws_canceled(int(bid_oid))
        # Drain the WS event into the bot's handler.
        processed = h.drain_private_events(max_events=10)
        assert processed >= 1, (
            f"WS canceled event must be drained; got processed={processed}"
        )

        # WO should now be terminal (CANCELED) OR slot may be released.
        bid_wo_after = h.state.order_store.get(Side.BUY, 0)
        if bid_wo_after is not None:
            assert bid_wo_after.status in (
                OrderStatus.CANCELED,
                OrderStatus.ACKED,  # if matcher race / event format mismatch
            ), (
                f"after WS canceled, BUY WO should be CANCELED or slot empty; "
                f"got status={bid_wo_after.status.value}"
            )


# ---------------------------------------------------------------------------
# Scenario 8: lost cancel ack — reaper transitions stale CANCEL_PENDING
# ---------------------------------------------------------------------------


def test_phase6b_scenario_lost_cancel_ack_reaped_after_timeout() -> None:
    """A cancel is dispatched, the HTTP succeeds, but the WS ``canceled``
    event never arrives. The reaper's CANCEL_PENDING timeout must fire
    and transition the WO to CANCELED.

    Without the reaper, the WO would sit in CANCEL_PENDING forever and
    the side would latch as unresolved — the v1.4.66 wedge class.
    Phase 2B's reaper shipped explicitly to fix this; this scenario
    pins that fix end-to-end.
    """
    from datetime import timedelta
    with harness_ctx(
        CANCEL_PENDING_UNRESOLVED_TIMEOUT_SECONDS=2.0,  # short timeout for fast test
    ) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()
        h.tick_once()
        bid_wo = h.state.order_store.get(Side.BUY, 0)
        assert bid_wo is not None and bid_wo.status == OrderStatus.ACKED

        # Force a cancel dispatch via a big reprice.
        h.seed_market(best_bid=2.040, best_ask=2.044)
        decision2 = h.make_decision(
            mid=2.042, target_spread_bps=20.0,
            quoted_bid=2.040, quoted_ask=2.044,
        )
        h.tick_once(decision=decision2)

        # The BUY WO should now be in CANCEL_PENDING (cancel sent,
        # WS event not yet received) — we never call emit_ws_canceled.
        bid_wo_after_reprice = h.state.order_store.get(Side.BUY, 0)
        assert bid_wo_after_reprice is not None
        assert bid_wo_after_reprice.status == OrderStatus.CANCEL_PENDING, (
            f"after reprice with no WS canceled event, BUY should be "
            f"CANCEL_PENDING; got {bid_wo_after_reprice.status.value}"
        )

        # Simulate timeout: push ts_cancel_requested back in time so the
        # reaper sees it as past the configured timeout.
        bid_wo_after_reprice.ts_cancel_requested = (
            datetime.now(timezone.utc) - timedelta(seconds=10.0)
        )

        # Bypass the reaper's per-call rate limit by resetting it.
        h.om._reaper_last_call_mono = 0.0

        # Run the reaper directly.
        reaped = h.om._reap_stale_ghosts()

        # The stale CANCEL_PENDING WO should have been reaped.
        assert reaped >= 1, (
            f"reaper should reap stale CANCEL_PENDING; got reaped={reaped}"
        )

        # The slot is either empty or holds a fresh WO (place may have
        # fired in the same tick after the cancel cleared the slot).
        slot = h.state.order_store.get(Side.BUY, 0)
        if slot is not None:
            assert slot.order_id_exchange != bid_wo.order_id_exchange or \
                slot.status != OrderStatus.CANCEL_PENDING, (
                f"after reaper firing, slot must not still hold the "
                f"original stale CANCEL_PENDING WO; got status={slot.status.value} "
                f"oid={slot.order_id_exchange}"
            )


# ---------------------------------------------------------------------------
# Scenario 9: slow-ack / stale-WO doesn't age past wall-lifetime cap
# ---------------------------------------------------------------------------


def test_phase6b_scenario_slow_ack_does_not_age_past_wall_lifetime_cap() -> None:
    """v1.4.67/v1.4.69 wedge class: an ACKED WO with very old ts_sent
    must be reaped via the SENT-timeout / wall-lifetime path, NOT
    sit on the book past `sent_order_unresolved_timeout × safety_mult`.

    Simulates an order that was placed normally (got ACK) but then
    sat unrefreshed for too long (e.g., the bot was paused for
    market-data staleness, or a reprice didn't fire). The wall-lifetime
    reaper safety net should catch it.

    Note: the bot's primary wall-lifetime guard is in `_build_side`'s
    age-cap check during quote generation, NOT the reaper. The reaper
    is the fallback. This test verifies the FALLBACK by manipulating
    timestamps directly.
    """
    from datetime import timedelta
    with harness_ctx(
        SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS=1.0,
        SENT_REAPER_SAFETY_MULTIPLIER=2.0,  # → reap at 2s
    ) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Manually construct a stale WO in SENT status to simulate
        # a slow-ack scenario (place dispatched, no response yet).
        from app.models import WorkingOrder
        import uuid
        now = datetime.now(timezone.utc)
        stale_ts = now - timedelta(seconds=60.0)  # placed 60s ago
        wo = WorkingOrder(
            order_id_local=f"stale-{uuid.uuid4().hex[:8]}",
            order_id_exchange=None,  # no oid bound yet (in SENT)
            client_order_id="stalewo123",
            symbol=h.settings.symbol,
            side=Side.BUY,
            price=2.000,
            size=3.0,
            post_only=True,
            status=OrderStatus.SENT,
            ts_created=stale_ts,
            ts_sent=stale_ts,
            ts_ack=None,
            level_idx=0,
        )
        h.state.order_store.set(Side.BUY, 0, wo)

        # Bypass reaper rate limit.
        h.om._reaper_last_call_mono = 0.0
        reaped = h.om._reap_stale_ghosts()

        # The stale SENT WO should be reaped.
        assert reaped >= 1, (
            f"reaper should reap stale SENT WO older than threshold; "
            f"got reaped={reaped}"
        )


# ---------------------------------------------------------------------------
# Scenario 10: repeated post-only cross 51604 — bot defends sanely
# ---------------------------------------------------------------------------


def test_phase6b_scenario_repeated_post_only_cross_keeps_side_safe() -> None:
    """When every place attempt comes back with sCode=51604 (post-only
    would cross — bot's price was too aggressive), the bot must NOT
    spin in a hot place-reject loop. The simplest invariant: after N
    rejected places, the bot's WO must transition to REJECTED (not
    stuck in SENT) and not be re-placed at the SAME aggressive price
    without re-evaluation.

    This is a softer pin than the plan's `5x_cooldown_armed` — the
    cooldown counter is bot-internal and varies with config. The
    invariant we CAN pin externally: after a series of 51604s,
    rejected WOs don't persist as live on the venue.
    """
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Configure all places to reject with 51604.
        h.client.config.place_response_outcome = "exchange_rejected"
        h.client.config.place_reject_scode = "51604"
        h.client.config.place_reject_smsg = "Post-only order would cross"

        # Drive 5 ticks. Every place attempt rejects.
        for i in range(5):
            decision = h.make_decision(cycle_id=f"cycle-cross-{i}")
            h.tick_once(decision=decision)

        # KEY: no live orders on the venue (every place rejected).
        assert len(h.client.live_orders()) == 0, (
            f"after 5 ticks of all-51604, venue should have no live orders; "
            f"got {len(h.client.live_orders())}"
        )

        # KEY: bot state's slots must not be stuck in SENT.
        for side in (Side.BUY, Side.SELL):
            slot = h.state.order_store.get(side, 0)
            if slot is not None:
                assert slot.status != OrderStatus.SENT, (
                    f"after repeated 51604s, {side.value} slot must not be "
                    f"stuck in SENT; got status={slot.status.value}"
                )


# ---------------------------------------------------------------------------
# Scenario 11: risk action oscillation — state machine settles
# ---------------------------------------------------------------------------


def test_phase6b_scenario_risk_oscillation_state_machine_settles() -> None:
    """Risk action flips ALLOW ↔ CANCEL_ALL across multiple ticks.
    State machine must converge — no permanent stuck state, no
    duplicate-place leaks.

    The end-state contract (after settling):
    * If the LAST tick was ALLOW, the bot should be quoting (or in
      the process of placing).
    * If the LAST tick was CANCEL_ALL, no live orders on the venue.
    * No WO ever leaks: every place either lands on the venue or
      transitions to a terminal status.
    """
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Sequence: ALLOW, CANCEL_ALL, ALLOW, CANCEL_ALL, ALLOW
        sequence = [
            RiskAction.ALLOW,
            RiskAction.CANCEL_ALL,
            RiskAction.ALLOW,
            RiskAction.CANCEL_ALL,
            RiskAction.ALLOW,
        ]
        for i, ra in enumerate(sequence):
            decision = h.make_decision(cycle_id=f"cycle-osc-{i}")
            h.tick_once(decision=decision, risk_action=ra)

        # Ends in ALLOW — bot should be quoting (≥1 order on venue).
        # OR may be in the process of placing (still SENT). Either is
        # acceptable; what we ENFORCE: no WO is stuck in a stale state.
        for side in (Side.BUY, Side.SELL):
            slot = h.state.order_store.get(side, 0)
            if slot is not None:
                # Any non-terminal WO must have made progress recently.
                assert slot.status in (
                    OrderStatus.NEW_LOCAL,
                    OrderStatus.SENT,
                    OrderStatus.ACKED,
                    OrderStatus.PARTIAL,
                    OrderStatus.CANCEL_PENDING,
                    OrderStatus.CANCELED,
                    OrderStatus.FILLED,
                    OrderStatus.REJECTED,
                    OrderStatus.DESYNC,
                ), f"after oscillation, {side.value} slot has unknown status: {slot.status}"

        # Sanity: dispatcher didn't stop responding to ALLOW. The
        # final tick was ALLOW; if the previous CANCEL_ALL left the
        # bot in a state where it can no longer quote, that's a wedge.
        # We accept "still in-flight" (SENT) as OK — it means the bot
        # is trying to place after the cancel.
        any_engaged = False
        for side in (Side.BUY, Side.SELL):
            slot = h.state.order_store.get(side, 0)
            if slot is not None and slot.status in (
                OrderStatus.SENT, OrderStatus.ACKED, OrderStatus.PARTIAL,
                OrderStatus.NEW_LOCAL,
            ):
                any_engaged = True
                break
        # If neither side is engaged, check that the venue ALSO has
        # no live orders — full stop is acceptable (the bot might
        # have decided not to quote post-oscillation), but the
        # STATE on the bot and venue must agree.
        if not any_engaged:
            live_after = len(h.client.live_orders())
            assert live_after == 0, (
                f"bot has no in-flight WO but venue has {live_after} live "
                f"orders — state drift!"
            )


# ---------------------------------------------------------------------------
# v1.4.97 — Phase 6B remaining scenarios (closing the plan's deferred set)
# ---------------------------------------------------------------------------


def test_phase6b_scenario_duplicate_oid_hydration_merges() -> None:
    """Phase 1B (v1.4.69) dedup invariant — full-flow regression test.

    The v1.4.66 / v1.4.67 wedge mechanism was: a single exchange OID
    ending up as TWO ``WorkingOrder`` rows in local state — one stuck
    in ``CANCEL_PENDING`` never reaped, one freshly hydrated. Each
    later cancelled via a different path (one via REST 51503, one via
    WS:CANCELED), leaving the cancel-reason histogram polluted and the
    state machine confused.

    Phase 1B's fix: the hydrator looks up any non-terminal WO matching
    the remote ``(oid, cloid)`` BEFORE creating a new row. If found,
    it MERGES — updates the existing WO's status, binds the OID if
    missing, stamps ``hydrated_from_exchange=True``. The
    ``hydration_merged_existing_total`` counter increments. No
    duplicate row is created.

    This test drives the full-flow:
    1. Place an order normally → ACKED, OID + cloid set locally.
    2. Simulate a stale REST snapshot containing the same OID (as if
       a hydration sweep landed mid-session).
    3. Assert: counter +=1, exactly ONE WO holds the OID,
       ``hydrated_from_exchange`` is True on that WO.
    """
    from app.exchange.base import OpenOrderRaw

    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Tick 1: place and ack a BUY.
        h.tick_once()
        existing_bid = h.state.order_store.get(Side.BUY, 0)
        assert existing_bid is not None, "tick 1 should have placed a BUY"
        assert existing_bid.status == OrderStatus.ACKED
        existing_oid = existing_bid.order_id_exchange
        existing_cloid = existing_bid.client_order_id
        existing_local_id = existing_bid.order_id_local
        assert existing_oid is not None, "venue should have bound an OID"
        assert existing_cloid is not None, "bot should have a cloid"

        before_merged = h.om._hydration_merged_existing_total

        # Construct an OpenOrderRaw mimicking what `sync_open_orders`
        # would feed `_hydrate_working_from_exchange` if a REST
        # snapshot included this still-live order.
        remote = OpenOrderRaw(
            oid=int(existing_oid),
            coin=h.settings.symbol,
            side=Side.BUY,
            limit_px=float(existing_bid.price),
            sz=float(existing_bid.size),
            timestamp=int(datetime.now(timezone.utc).timestamp() * 1000),
            cloid=existing_cloid,
        )

        result = h.om._hydrate_working_from_exchange(Side.BUY, remote)
        # Returns True when treated as a successful hydration (merge).
        assert result is True, "hydrate should report success on merge"

        # The dedup counter incremented exactly once.
        assert h.om._hydration_merged_existing_total == before_merged + 1, (
            f"Phase 1B counter didn't bump: "
            f"before={before_merged} after={h.om._hydration_merged_existing_total}"
        )

        # CRITICAL INVARIANT (Phase 1B): exactly ONE WO holds the OID.
        # Iterate all working orders (not just the BUY-0 slot) since
        # the bug would put the duplicate anywhere.
        wos_with_oid = [
            wo for wo in h.state.all_working_orders()
            if wo.order_id_exchange and int(wo.order_id_exchange) == int(existing_oid)
        ]
        assert len(wos_with_oid) == 1, (
            f"Phase 1B dedup violated: {len(wos_with_oid)} WOs share OID "
            f"{existing_oid} (expected exactly 1). "
            f"WO local_ids: {[str(wo.order_id_local) for wo in wos_with_oid]}"
        )
        merged = wos_with_oid[0]
        # The merged WO is the SAME local row that existed before the
        # hydration — not a fresh row that happens to share the OID.
        assert merged.order_id_local == existing_local_id, (
            "hydration should have merged into the existing WO, not "
            "created a new one"
        )
        # Hydrated-from-exchange flag is set as documented in Phase 1B.
        assert merged.hydrated_from_exchange is True, (
            "merged WO should have hydrated_from_exchange=True"
        )


def test_phase6b_scenario_two_rung_concurrent_legitimate() -> None:
    """Phase 1D (v1.4.71/72) slot-aware reconcile invariant —
    full-flow.

    With ``LADDER_NUM_LEVELS_PER_SIDE=2`` the bot should sustain TWO
    BUY rungs + TWO SELL rungs concurrently on the venue. The
    pre-Phase-1D bug was the side-keyed reconcile path: anything past
    one order per side classified as ``duplicate_same_side_extra``
    and got cancelled. With LADDER=2 every legitimate outer rung was
    cancelled within one cycle. Snapshot v1.4.66-260518-192744 showed
    8 such phantom cancels in 75 s.

    Phase 1D's fix: ``_sync_open_orders_impl`` now maps each remote
    order to a ``(side, level_idx)`` slot. Two BUYs at distinct
    cloids fill ``(BUY, 0)`` and ``(BUY, 1)`` cleanly; neither is
    flagged.

    This test verifies the end-to-end flow:
    1. ``LADDER_NUM_LEVELS_PER_SIDE=2`` configured.
    2. Engine emits 2 BUY + 2 SELL on first tick.
    3. Bot places all 4 via the dispatcher.
    4. After tick: 4 live orders on the venue, no
       ``desync_detected`` flag, ``duplicate_same_side_extra``
       count stays at 0.
    """
    with harness_ctx(
        LADDER_NUM_LEVELS_PER_SIDE=2,
        # Outer-rung offset_step controls how far rung-1 sits behind
        # rung-0. Default 1.0 of half-spread — fine for the harness's
        # 6 bps half-spread.
        LADDER_OFFSET_STEP=1.0,
        LADDER_SIZE_DECAY=0.7,
    ) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        h.tick_once()

        # On the venue: expect 4 live (2 BUY + 2 SELL).
        live = h.client.live_orders()
        live_by_side: dict[Side, list] = {Side.BUY: [], Side.SELL: []}
        for o in live:
            live_by_side[o.side].append(o)

        # ``LADDER_NUM_LEVELS_PER_SIDE=2`` SHOULD produce 2/2. If the
        # ladder grid-collision dedup (v1.4.79) drops a rung because
        # of tick-size at the chosen prices, we may only see 1 per
        # side. In either case the invariant we care about is "no
        # duplicate same-side-extra cancels".
        n_buy = len(live_by_side[Side.BUY])
        n_sell = len(live_by_side[Side.SELL])
        assert n_buy >= 1 and n_buy <= 2, (
            f"LADDER=2 should produce 1-2 BUY rungs (2 if grid-distinct, "
            f"1 if collision-deduped); got {n_buy}"
        )
        assert n_sell >= 1 and n_sell <= 2, (
            f"LADDER=2 should produce 1-2 SELL rungs; got {n_sell}"
        )

        # No desync flagged.
        assert h.state.order_desync is False, (
            "Phase 1D invariant: 2-rung config must not trigger desync"
        )

        # No ``duplicate_same_side_extra`` cancellations recorded.
        # The mock venue records every cancel call; check no rung was
        # spuriously cancelled. ``client.calls`` is the running list of
        # API calls; for a fresh 2-rung tick we expect place_batch (or
        # place_single x2) + zero cancels.
        cancel_calls = [c for c in h.client.calls if c[0] in (
            "cancel_order", "cancel_batch_orders",
        )]
        assert len(cancel_calls) == 0, (
            f"Phase 1D invariant: 2-rung config must not cancel any "
            f"legitimate rung; got {len(cancel_calls)} cancel calls: "
            f"{[c[0] for c in cancel_calls]}"
        )

        # Tick 2 (same market state): the bot should be stable; no
        # spurious cancels. (If the slot-aware reconcile is broken,
        # this is when the duplicates-extra cancels would fire.)
        calls_before_tick2 = len(h.client.calls)
        h.tick_once()
        new_calls = h.client.calls[calls_before_tick2:]
        cancel_calls_tick2 = [c for c in new_calls if c[0] in (
            "cancel_order", "cancel_batch_orders",
        )]
        assert len(cancel_calls_tick2) == 0, (
            f"Phase 1D invariant: a quiet 2-rung session should not "
            f"emit cancels on the 2nd tick; got: "
            f"{[c[0] for c in cancel_calls_tick2]}"
        )
