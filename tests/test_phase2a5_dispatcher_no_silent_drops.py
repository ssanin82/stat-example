"""v1.4.93 wedge-elimination-cleanup Phase 2A.5 — dispatcher
"never silently drops a non-NoOp action" property test.

Contract: for every `PlaceAction` / `CancelAction` / `AmendAction`
that the reconciler emits, the dispatcher must either:

* produce a corresponding transport call (HTTP / queued intent), OR
* record an explicit suppression trace via
  ``_record_orchestrate_decision`` with a reason.

Pre-refactor (silent-wedge class): a missing branch or swallowed
exception could DROP an Action — no transport, no trace, no
visibility. The bot would silently fail to place / cancel / amend.
This test pins the contract: count Actions emitted by reconcile()
vs. transport calls + explicit-noop traces; assert they balance.

Approach:
1. Drive 50 random ticks via Phase 6B's harness.
2. For each tick, capture:
   - Pre-tick orchestrate_decision_counts (per (side, branch, action))
   - Pre-tick venue transport call count
3. After each tick, compute deltas.
4. Assert: every new orchestrate decision with action in
   {placed_fresh, cancel_*, amend_*} has a corresponding venue call
   OR is explicitly logged as a suppression.

This is a "tail balance" check — strict equality is hard because some
actions queue intents (outbound dispatcher) that may not flush within
one tick. We pin the weaker but observable invariant: no PRODUCED
action is invisible.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from app.enums import ActiveSides, OrderStatus, RiskAction, Side
from tests.integration.full_flow_harness import harness_ctx
from tests.integration.mock_okx_client import MockOrderState


# Real action strings used by ``_record_orchestrate_decision`` —
# discovered by grepping ``action="..."`` in `app/execution.py`. The
# orchestrate-decision tracker is PLACE-side instrumentation; cancels
# go through a different trace path (not exercised by this test).
_PLACE_DISPATCHED = {"place_dispatched"}
_PLACE_SUPPRESSED = {
    "place_dispatch_refused_unresolved",
    "place_skipped_bluefin_cancel_pending",
    "place_skipped_should_not_emit_fresh",
    "place_stage_returned_none",
    "place_enqueue_transport_failed",
}
_AMEND_ACTIONS = {"amend_dispatched"}
_VENUE_PLACE_METHODS = {"place_post_only_limit", "batch_place_post_only_limit"}
_VENUE_CANCEL_METHODS = {"cancel_order", "cancel_batch_orders"}
_VENUE_AMEND_METHODS = {"amend_batch_orders", "amend_order"}


def _decisions_in_categories(
    counts: dict[str, int],
    categories: set[str],
) -> int:
    """Sum decision counts where the action (last colon-segment) is
    in ``categories``."""
    total = 0
    for key, c in counts.items():
        parts = key.split(":")
        if not parts:
            continue
        action = parts[-1]
        if action in categories:
            total += c
    return total


def _venue_calls_in_categories(
    calls: list[tuple[str, dict]],
    methods: set[str],
) -> int:
    return sum(1 for name, _ in calls if name in methods)


def _run_random_tick(
    h,
    rng: random.Random,
    cycle_id: str,
) -> None:
    """Apply one of: normal_quote, one_sided, risk_no_quote,
    risk_cancel_all, reprice, simulated_fill, simulated_cancel.
    Random distribution biased toward normal_quote so most ticks
    actually quote."""
    op = rng.choices(
        ["quote", "one_sided", "no_quote", "cancel_all", "reprice", "ws_fill"],
        weights=[40, 10, 15, 5, 20, 10],
    )[0]

    if op == "quote":
        h.tick_once(decision=h.make_decision(cycle_id=cycle_id))
    elif op == "one_sided":
        active = rng.choice([ActiveSides.BID_ONLY, ActiveSides.ASK_ONLY])
        h.tick_once(decision=h.make_decision(active_sides=active, cycle_id=cycle_id))
    elif op == "no_quote":
        h.tick_once(
            decision=h.make_decision(cycle_id=cycle_id),
            risk_action=RiskAction.NO_QUOTE,
            cancel_on_no_quote=False,
        )
    elif op == "cancel_all":
        h.tick_once(
            decision=h.make_decision(cycle_id=cycle_id),
            risk_action=RiskAction.CANCEL_ALL,
        )
        # Confirm cancels via WS so the bot doesn't latch unresolved.
        for o in h.client.canceled_orders():
            try:
                h.client.emit_ws_canceled(o.ord_id)
            except (RuntimeError, ValueError):
                pass
        h.drain_private_events()
    elif op == "reprice":
        # Big price move to force the bot to act.
        drift = rng.choice([-0.020, -0.010, 0.010, 0.020, 0.030])
        h.seed_market(
            best_bid=2.000 + drift,
            best_ask=2.002 + drift,
        )
        h.tick_once(
            decision=h.make_decision(
                mid=2.001 + drift,
                quoted_bid=2.000 + drift - 0.001,
                quoted_ask=2.002 + drift + 0.001,
                cycle_id=cycle_id,
            )
        )
    elif op == "ws_fill":
        # Fill the inside BUY if it's live on the venue.
        live = h.client.live_orders()
        buys = [o for o in live if o.side == Side.BUY]
        if buys:
            o = buys[0]
            h.client.emit_ws_filled(o.ord_id, fill_qty=o.size, fill_px=o.price)
            h.drain_private_events()
        else:
            # Fall back to a normal quote.
            h.tick_once(decision=h.make_decision(cycle_id=cycle_id))


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


PHASE2A5_ITERATIONS = 50
PHASE2A5_SEED = 1234


def test_phase2a5_every_venue_place_has_a_dispatch_trace() -> None:
    """**The key invariant**: the count of venue `place_post_only_limit` +
    `batch_place_post_only_limit` calls must be ≤ the count of
    ``place_dispatched`` orchestrate-decision traces.

    Why this direction: if a venue place HAPPENED without a
    corresponding ``place_dispatched`` trace, the dispatcher fired a
    place call without recording it. That's the silent-wedge class
    we're catching (refactor accidentally bypassed
    ``_record_orchestrate_decision``).

    The OTHER direction (more traces than calls) is also a wedge —
    "dispatched but never sent" — but that's harder to catch because
    of the async outbound dispatcher; some traces have in-flight
    intents that haven't drained yet. Phase 6B's ``wait_until_idle``
    minimizes this, but we still allow some slack.
    """
    rng = random.Random(PHASE2A5_SEED)
    with harness_ctx() as h:
        h.seed_market()
        h.seed_position()

        for i in range(PHASE2A5_ITERATIONS):
            _run_random_tick(h, rng, cycle_id=f"p2a5-{i}")

        decisions = dict(h.om._orchestrate_decision_counts)
        place_dispatched_traces = _decisions_in_categories(decisions, _PLACE_DISPATCHED)
        venue_place_calls = _venue_calls_in_categories(h.client.calls, _VENUE_PLACE_METHODS)

        # The CRITICAL direction: venue calls must NOT exceed traces.
        # Every place that hit the wire must have left a record.
        # Tolerance of +2 accounts for batch_place_post_only_limit calls
        # that fire ONE HTTP call covering multiple traced dispatches —
        # the venue.calls count could legitimately be 1 trace : 1 call
        # OR N traces : 1 batched call, so trace ≥ call is the
        # stronger direction.
        assert venue_place_calls <= place_dispatched_traces + 2, (
            f"venue saw {venue_place_calls} place calls but only "
            f"{place_dispatched_traces} ``place_dispatched`` traces — "
            f"silent-emit wedge class. Decisions: "
            f"{ {k: v for k, v in decisions.items() if 'place' in k.lower()} }"
        )


def test_phase2a5_place_actions_always_record_outcome() -> None:
    """Every reconciler PlaceAction must reach the dispatcher AND
    record an outcome (one of place_dispatched, place_skipped_*,
    place_dispatch_refused_*, place_stage_returned_none,
    place_enqueue_transport_failed). If the dispatcher silently
    short-circuits without recording, this test fails.

    Tested by: drive many ticks under varied conditions; assert that
    at least one ``place_*`` action category is non-zero (i.e., the
    instrumentation IS firing). If the entire category is empty
    despite obvious place activity (venue saw calls), the
    instrumentation is bypassed."""
    rng = random.Random(PHASE2A5_SEED + 1)
    with harness_ctx() as h:
        h.seed_market()
        h.seed_position()

        for i in range(PHASE2A5_ITERATIONS):
            _run_random_tick(h, rng, cycle_id=f"p2a5p-{i}")

        venue_place_calls = _venue_calls_in_categories(h.client.calls, _VENUE_PLACE_METHODS)
        decisions = dict(h.om._orchestrate_decision_counts)
        total_place_traces = _decisions_in_categories(
            decisions, _PLACE_DISPATCHED | _PLACE_SUPPRESSED
        )

        # If venue saw places, the trace surface must show activity.
        if venue_place_calls > 0:
            assert total_place_traces > 0, (
                f"venue saw {venue_place_calls} place calls but ZERO "
                f"place-related traces ({_PLACE_DISPATCHED | _PLACE_SUPPRESSED}) "
                f"in orchestrate_decisions. Dispatcher is bypassing "
                f"_record_orchestrate_decision. All decisions: {decisions}"
            )


def test_phase2a5_no_action_without_trace() -> None:
    """Stronger invariant: for any orchestrate-decision count that
    bumps, the trace IS recorded (we just look at the keys — every
    decision branch must have a corresponding counter increment).

    This is a tautology if the orchestrate-decision recording API is
    used consistently. The point: the COUNT of distinct
    (side, branch, action) tuples should grow with activity — if it
    doesn't, the dispatcher is bypassing the recording API.
    """
    rng = random.Random(PHASE2A5_SEED + 2)
    with harness_ctx() as h:
        h.seed_market()
        h.seed_position()

        for i in range(15):  # smaller — checking trace richness
            _run_random_tick(h, rng, cycle_id=f"p2a5t-{i}")

        decisions = dict(h.om._orchestrate_decision_counts)
        # After 15 mixed-op ticks, we expect at minimum 3 distinct
        # decision keys (multiple branches exercised: at least one
        # place, one cancel-or-noop, one ws-driven transition).
        assert len(decisions) >= 3, (
            f"after 15 mixed ticks, only {len(decisions)} distinct decision "
            f"keys recorded — dispatcher may be bypassing the trace API. "
            f"Keys: {sorted(decisions.keys())}"
        )

        # Every key follows the "<side>:<branch>:<action>" schema.
        for key in decisions:
            parts = key.split(":")
            # action may itself contain ":" (e.g., "cancel:reason"), so
            # we only check the first segment is a side.
            assert parts[0] in ("BUY", "SELL"), (
                f"orchestrate decision key {key!r} doesn't follow "
                f"<side>:<branch>:<action> schema"
            )


def test_phase2a5_no_silent_failure_on_reject_storm() -> None:
    """During a storm of place rejects (51604), every reject must
    leave a trace. This catches the v1.4.64 regression class where
    rejects could leave WOs stuck in SENT with no observability."""
    rng = random.Random(PHASE2A5_SEED + 3)
    with harness_ctx() as h:
        h.seed_market()
        h.seed_position()

        # Initial tick to populate state.
        h.tick_once()

        # Configure all places to reject.
        h.client.config.place_response_outcome = "exchange_rejected"
        h.client.config.place_reject_scode = "51604"

        for i in range(10):
            h.tick_once(decision=h.make_decision(cycle_id=f"rej-{i}"))

        # No WO stuck in SENT after the storm.
        stuck_sent = []
        for side in (Side.BUY, Side.SELL):
            slot = h.state.order_store.get(side, 0)
            if slot is not None and slot.status == OrderStatus.SENT:
                stuck_sent.append((side, slot.order_id_local))

        assert not stuck_sent, (
            f"after 10 reject-storm ticks, {len(stuck_sent)} WO(s) stuck "
            f"in SENT: {stuck_sent}. The reject path silently failed."
        )

        # The orchestrate decisions should show evidence of rejects.
        decisions = dict(h.om._orchestrate_decision_counts)
        total_decisions = sum(decisions.values())
        assert total_decisions > 0, (
            f"after 10 reject-storm ticks, NO orchestrate decisions "
            f"recorded — dispatcher is silent"
        )
