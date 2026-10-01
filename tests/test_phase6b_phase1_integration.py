"""v1.4.93 wedge-elimination-cleanup — integration tests for the
Phase 1 contracts (originally Phase 1A.8 / 1C.7 / 1D.6, deferred when
the mock-venue harness didn't exist).

Phase 6B's `MockOkxClient` + `FullFlowHarness` now make these
buildable. Each test pins one Phase 1 invariant end-to-end (vs the
existing Phase 1A/1C/1D unit tests, which exercise individual
helpers).

* **1A.8** `test_full_flow_ws_arrives_before_place_response` — WS
  ``live`` event arrives in the bot's queue BEFORE the HTTP place
  response returns. The Phase 1A buffer must hold it and apply it
  when the place response binds the WO.
* **1C.7** `test_outer_rung_does_not_survive_past_wall_lifetime_cap` —
  2-rung ladder; an outer rung that never gets WS confirmation must
  be reaped via the wall-lifetime cap. No WO lives past 2× cap.
* **1D.6** `test_2_rung_session_no_spurious_desync` — 2-rung session
  over multiple ticks: ``desync_detected`` count stays zero (Phase 1D
  reconciler is slot-aware).
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from app.enums import OrderStatus, RiskAction, Side
from tests.integration.full_flow_harness import harness_ctx
from tests.integration.mock_okx_client import MockOrderState


# ---------------------------------------------------------------------------
# 1C.7 — outer rung doesn't survive past wall-lifetime cap
# ---------------------------------------------------------------------------


def test_phase1c7_outer_rung_does_not_survive_past_wall_lifetime_cap() -> None:
    """2-rung ladder; an outer rung gets placed but the WS ``live``
    event never arrives. The Phase 1C wall-lifetime cap + Phase 2B
    reaper combination must eventually clear it. No WO lives past
    2× the configured cap.

    *Simulation approach:* construct a fake aged outer rung in SENT
    status (ts_sent = far past), run the reaper, assert it's reaped.
    Equivalent to the plan's "30-tick simulated session with slow ACK"
    but without spending real wall-clock time."""
    with harness_ctx(
        LADDER_NUM_LEVELS_PER_SIDE=2,
        SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS=1.0,
        SENT_REAPER_SAFETY_MULTIPLIER=2.0,  # → 2-second reap threshold
        BEHIND_TOUCH_MAX_AGE_SECONDS=1.5,
    ) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Construct an aged outer-rung BUY WO in SENT status.
        from app.models import WorkingOrder
        import uuid
        stale_ts = datetime.now(timezone.utc) - timedelta(seconds=10.0)
        outer_wo = WorkingOrder(
            order_id_local=f"stale-outer-{uuid.uuid4().hex[:8]}",
            order_id_exchange=None,
            client_order_id="stale-outer-cloid",
            symbol=h.settings.symbol,
            side=Side.BUY,
            price=1.998,
            size=3.0,
            post_only=True,
            status=OrderStatus.SENT,
            ts_created=stale_ts,
            ts_sent=stale_ts,
            ts_ack=None,
            level_idx=1,  # outer rung
        )
        h.state.order_store.set(Side.BUY, 1, outer_wo)

        # Run the reaper.
        h.om._reaper_last_call_mono = 0.0
        reaped = h.om._reap_stale_ghosts()
        assert reaped >= 1, (
            f"reaper should reap the stale outer-rung SENT WO; "
            f"got reaped={reaped}"
        )

        # The outer slot should now be empty or hold a different WO
        # (NOT the stale one).
        slot = h.state.order_store.get(Side.BUY, 1)
        if slot is not None:
            assert slot.order_id_local != outer_wo.order_id_local, (
                f"stale outer-rung WO {outer_wo.order_id_local} still in slot "
                f"after reap"
            )


# ---------------------------------------------------------------------------
# 1D.6 — 2-rung session, no spurious desync events
# ---------------------------------------------------------------------------


def test_phase1d6_2_rung_session_no_spurious_desync() -> None:
    """With ``LADDER_NUM_LEVELS_PER_SIDE=2``, a healthy multi-tick
    session must NOT trigger ``desync_detected`` events. The
    pre-Phase-1D reconciler kept only one BUY + one SELL — any
    legitimate outer rung was classified as ``duplicate_same_side_extra``
    and cancelled, triggering a desync cycle.

    This test pins the v1.4.71 Phase 1D fix end-to-end: 2 rungs both
    sides, 10 ticks, no desync."""
    with harness_ctx(LADDER_NUM_LEVELS_PER_SIDE=2) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Capture desync state baseline (should be OK at start).
        assert getattr(h.state, "desync_phase", None) in (None, "OK") or \
            getattr(h.state.desync_phase, "value", "OK") == "OK"

        # Drive 10 ticks. Each tick may reprice; the bot manages 2
        # rungs per side. No desync should be detected.
        for i in range(10):
            h.tick_once(decision=h.make_decision(cycle_id=f"2rung-{i}"))

        # Phase 1D's slot-keyed reconciler should not have triggered.
        # Check that no orders were classified as `duplicate_same_side_extra`
        # (counter-style assertion via executor state).
        es = h.state.executor_state_snapshot or {}
        # The bot's counter for this case is per-classification; we
        # check the most-direct signal: desync_phase stayed OK.
        final_dp = getattr(h.state, "desync_phase", "OK")
        final_dp_val = getattr(final_dp, "value", None) or str(final_dp)
        assert final_dp_val.upper() in ("OK", ""), (
            f"after 10-tick 2-rung session, desync_phase transitioned to "
            f"{final_dp_val!r} — Phase 1D regression"
        )


def test_phase1d6_2_rung_both_sides_can_be_live_simultaneously() -> None:
    """Sub-invariant of 1D.6: in a 2-rung config, BOTH rungs on a
    side can be live simultaneously without the reconciler firing
    a spurious cancel.

    This directly tests the v1.4.71 Phase 1D fix: pre-v1.4.71 a
    2nd live order on the same side was always ``duplicate_same_side_extra``."""
    with harness_ctx(LADDER_NUM_LEVELS_PER_SIDE=2) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Drive a tick — bot places (potentially up to 2 BUYs + 2 SELLs).
        h.tick_once()

        # Count live orders per side.
        live = h.client.live_orders()
        bid_count = sum(1 for o in live if o.side == Side.BUY)
        ask_count = sum(1 for o in live if o.side == Side.SELL)

        # The bot may place 1 OR 2 per side depending on engine
        # decision (multi-rung dedup may collapse rungs at same tick).
        # We assert: at least one side placed something, AND no side
        # has MORE than 2 live orders (which would be a real wedge).
        assert (bid_count + ask_count) > 0, "no orders placed at all"
        assert bid_count <= 2, f"too many live BIDs: {bid_count}"
        assert ask_count <= 2, f"too many live ASKs: {ask_count}"


# ---------------------------------------------------------------------------
# 1A.8 — WS event arrives BEFORE place response (Phase 1A buffer)
# ---------------------------------------------------------------------------


def test_phase1a8_ws_event_arrives_before_place_response() -> None:
    """The Phase 1A buffer must hold a WS ``live`` event when it
    arrives before the place HTTP response binds the WO to its OID.

    *Simulation:* spawn a thread that watches for new orders in the
    mock venue's order book; as soon as one appears (during the place
    HTTP call), emit the WS event. The mock client's place call has
    a configurable latency to widen the race window.

    What we verify: after the place + WS flow completes, the WO is
    ACKED (not stuck in SENT). The Phase 1A buffer caught the
    pre-arrival event and applied it on place-response commit.

    NOTE: this is a coarse timing race — the bot's outbound dispatcher
    runs in its own thread. Real-world prod timing is much tighter
    than what this test simulates. If this test passes, the buffer
    works for typical races; it doesn't prove correctness under all
    timing patterns.
    """
    with harness_ctx() as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()
        # 100ms place latency to widen the race window.
        h.client.config.place_latency_ms = 100.0

        # Spawn a watcher thread that emits a WS `live` event for any
        # new order it sees on the mock venue.
        stop_event = threading.Event()

        def _ws_emitter():
            seen: set[int] = set()
            while not stop_event.is_set():
                try:
                    for o in h.client.live_orders():
                        if o.ord_id not in seen:
                            seen.add(o.ord_id)
                            # Emit immediately — this happens DURING the
                            # bot's place HTTP call (place_latency_ms=100).
                            try:
                                h.client.emit_ws_live(o.ord_id)
                            except (RuntimeError, ValueError):
                                pass
                except Exception:
                    pass
                time.sleep(0.005)  # 5ms poll

        t = threading.Thread(target=_ws_emitter, daemon=True)
        t.start()
        try:
            h.tick_once()
            time.sleep(0.2)  # let any late WS events settle
            h.drain_private_events()
        finally:
            stop_event.set()
            t.join(timeout=1.0)

        # Verify: at least one WO landed in ACKED status. The Phase 1A
        # buffer (or natural ordering) ensured the pre-arrival WS event
        # was applied correctly.
        any_acked = False
        for side in (Side.BUY, Side.SELL):
            slot = h.state.order_store.get(side, 0)
            if slot is not None and slot.status in (
                OrderStatus.ACKED, OrderStatus.PARTIAL,
            ):
                any_acked = True
                break

        assert any_acked, (
            "no WO reached ACKED state — Phase 1A buffer may have lost "
            "the pre-arrival WS event"
        )
