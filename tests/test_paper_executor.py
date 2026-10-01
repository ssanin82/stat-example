"""Tests for Phase 2 paper executor (v1.4.232).

Covers the §2.4 acceptance suite from
``backtesting/docs/execution-plan.md``:

* Single-order fill on touching trade.
* No fill if queue ahead is unconsumed.
* Cancel race (cancel + trade within cancel latency → fill wins).
* Amend resets queue position.
* Self-trade prevention (self-cross rejected at placement).
* Position + realized PnL accumulate correctly.

Plus:

* Post-only cross-reject vs touch.
* Place-latency gating (trades before active_at don't fill).
* Sweep pending cancels emits cancellation event.
* Adapter Protocol conformance.
* Deterministic cloid generation.
"""

from __future__ import annotations

from queue import Queue

import pytest

from app.backtest import PaperExecutor, PaperExecutorConfig
from app.clock import ReplayClock
from app.enums import Side
from app.exchange.base import PerpExchangeAdapter
from app.exchange.private_events import PrivateFillEvent, PrivateOrderUpdateEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _executor(*, start_ns: int = 1_700_000_000_000_000_000) -> tuple[
    PaperExecutor, ReplayClock, "Queue"
]:
    """Build a paper executor + clock + sink for a fresh test."""
    clock = ReplayClock(start_t_ns=start_ns)
    sink: Queue = Queue()
    cfg = PaperExecutorConfig(
        sim_place_latency_s=0.020,
        sim_cancel_latency_s=0.015,
        fee_maker_bps=-0.5,
        fee_taker_bps=1.0,
    )
    px = PaperExecutor(
        clock=clock,
        private_event_sink=sink,
        config=cfg,
        symbol="TON-USDT-SWAP",
    )
    return px, clock, sink


def _advance(clock: ReplayClock, delta_s: float) -> None:
    new_ns = int(clock._t_ns + delta_s * 1e9)
    clock.advance_to(new_ns)


def _drain(sink: Queue) -> list:
    out = []
    while not sink.empty():
        out.append(sink.get_nowait())
    return out


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_rejects_negative_latency() -> None:
    with pytest.raises(ValueError):
        PaperExecutorConfig(sim_place_latency_s=-0.001)
    with pytest.raises(ValueError):
        PaperExecutorConfig(sim_cancel_latency_s=-0.001)


def test_config_rejects_unknown_queue_policy() -> None:
    with pytest.raises(ValueError):
        PaperExecutorConfig(queue_policy="fifo_v2")


# ---------------------------------------------------------------------------
# Adapter Protocol conformance
# ---------------------------------------------------------------------------


def test_satisfies_perpexchangeadapter_protocol() -> None:
    px, _, _ = _executor()
    # ``@runtime_checkable`` only validates method presence (not
    # signatures), but that's the smoke check we want at import time.
    assert isinstance(px, PerpExchangeAdapter)
    # symbol_spec_fetched_ok must be settable per the Protocol.
    assert px.symbol_spec_fetched_ok is True


def test_has_write_access_returns_true() -> None:
    px, _, _ = _executor()
    assert px.has_write_access() is True


# ---------------------------------------------------------------------------
# Place: cross-reject + self-cross
# ---------------------------------------------------------------------------


def test_place_post_only_cross_reject_against_ask() -> None:
    """BUY at price >= best_ask → reject as post-only cross."""
    px, clock, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    resp = px.place_post_only_limit(
        symbol="TON-USDT-SWAP", is_buy=True, sz=10.0, limit_px=2.015
    )
    assert resp["status"] == "rejected"
    assert resp["code"] == "post_only_cross"
    # Detector recognizes the reject reason.
    from app.execution import is_post_only_immediate_match_rejection
    assert is_post_only_immediate_match_rejection(resp["msg"])
    assert px.rejects_emitted == 1
    assert px.acks_emitted == 0


def test_place_post_only_cross_reject_against_bid() -> None:
    """SELL at price <= best_bid → reject as post-only cross."""
    px, _, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    resp = px.place_post_only_limit(
        symbol="TON-USDT-SWAP", is_buy=False, sz=10.0, limit_px=2.000
    )
    assert resp["status"] == "rejected"
    assert resp["code"] == "post_only_cross"


def test_place_self_cross_rejected() -> None:
    """If a SELL is already resting at 2.05, a BUY at 2.05 self-crosses."""
    px, _, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.100, bid_size=100.0, ask_size=100.0)
    sell = px.place_post_only_limit(
        symbol="X", is_buy=False, sz=5.0, limit_px=2.050
    )
    assert sell["status"] == "ok"
    buy = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=5.0, limit_px=2.050
    )
    assert buy["status"] == "rejected"
    assert buy["code"] == "post_only_cross"


def test_place_returns_deterministic_oid_sequence() -> None:
    """Two consecutive places get oid 1, 2 in order."""
    px, _, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.100, bid_size=10.0, ask_size=10.0)
    r1 = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=1.0, limit_px=1.999
    )
    r2 = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=1.0, limit_px=1.998
    )
    assert r1["oid"] == 1
    assert r2["oid"] == 2


# ---------------------------------------------------------------------------
# Fill matching — single order, simple case
# ---------------------------------------------------------------------------


def test_single_order_fill_on_touching_trade() -> None:
    """Place a BUY at 2.00 (alone at the top) → a SELL trade at 2.00 fills."""
    px, clock, sink = _executor()
    # No existing depth at the price (we'll be the only resting BUY at 2.00).
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    assert resp["status"] == "ok"
    # Advance past the place-latency.
    _advance(clock, 0.025)
    # SELL aggressor at 2.00, size 10.
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    events = _drain(sink)
    fills = [e for e in events if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 1
    assert fills[0].sz == 10.0
    assert fills[0].px == 2.000
    assert fills[0].side == "B"
    # Position bookkeeping.
    assert px.current_position_qty() == 10.0
    assert px.current_avg_entry() == pytest.approx(2.000)


def test_no_fill_when_queue_ahead_unconsumed() -> None:
    """Order behind 100 lots of bid — a 10-lot SELL doesn't reach us."""
    px, clock, sink = _executor()
    # 100 lots resting at 2.00 ahead of us.
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    assert resp["status"] == "ok"
    _advance(clock, 0.025)
    # SELL aggressor only 10 lots — gets eaten by the queue ahead.
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    events = _drain(sink)
    fills = [e for e in events if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 0
    assert px.current_position_qty() == 0.0


def test_fill_after_queue_drains() -> None:
    """100 lots queue ahead → after 100 lots traded, the next trade fills us."""
    px, clock, sink = _executor()
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    px.place_post_only_limit(
        symbol="X", is_buy=True, sz=5.0, limit_px=2.000
    )
    _advance(clock, 0.025)
    # Drain the queue.
    px.process_trade_event(price=2.000, size=100.0, side="SELL")
    # Now next trade should fill us.
    px.process_trade_event(price=2.000, size=5.0, side="SELL")
    events = _drain(sink)
    fills = [e for e in events if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 1
    assert fills[0].sz == 5.0


# ---------------------------------------------------------------------------
# Place latency — trades before active_at don't fill
# ---------------------------------------------------------------------------


def test_place_latency_gates_fills() -> None:
    """A trade arriving inside the 20ms place-latency window doesn't fill."""
    px, clock, sink = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    # Only 10ms — less than 20ms place latency.
    _advance(clock, 0.010)
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    fills = [e for e in _drain(sink) if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 0
    # Advance past latency, send another trade — now it fills.
    _advance(clock, 0.020)
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    fills = [e for e in _drain(sink) if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 1


# ---------------------------------------------------------------------------
# Cancel race
# ---------------------------------------------------------------------------


def test_cancel_race_fill_wins_inside_cancel_latency() -> None:
    """Cancel submitted, then a trade arrives inside the 15ms cancel window
    → fill happens (the cancel hasn't applied at the engine yet)."""
    px, clock, sink = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    _advance(clock, 0.025)  # past place-latency
    # Submit cancel.
    cancel = px.cancel_order(symbol="X", oid=oid)
    assert cancel["status"] == "ok"
    # 5ms later, trade arrives before cancel completes (15ms latency).
    _advance(clock, 0.005)
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    fills = [e for e in _drain(sink) if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 1, "fill must win the cancel race"
    assert px.current_position_qty() == 10.0


def test_cancel_completes_after_latency() -> None:
    """No trade during the cancel window → the order is gone after it lapses."""
    px, clock, sink = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    _advance(clock, 0.025)
    px.cancel_order(symbol="X", oid=oid)
    _advance(clock, 0.020)  # past cancel-latency
    # Send a book event so the sweep runs.
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    # Order should be gone; even a touch trade now finds no resting order.
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    fills = [e for e in _drain(sink) if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 0
    # And a "canceled" order-update was emitted.
    # (Drain already happened; we trust cancels_emitted counter.)
    assert px.cancels_emitted == 1


def test_cancel_missing_order_returns_benign() -> None:
    px, _, _ = _executor()
    resp = px.cancel_order(symbol="X", oid=9999)
    assert resp["status"] == "benign_missing"


# ---------------------------------------------------------------------------
# Amend resets queue position
# ---------------------------------------------------------------------------


def test_amend_resets_queue_position() -> None:
    """Amend = cancel+place at new price → new queue position computed
    from current top-of-book at amend time."""
    px, clock, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.010, bid_size=50.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    # Initial queue ahead = 50 (sat behind 50-lot bid).
    assert px._orders[oid].queue_ahead == 50.0
    # Now bid_size moves to 200; amend.
    px.process_book_event(bid=2.000, ask=2.010, bid_size=200.0, ask_size=50.0)
    px.amend_order(symbol="X", oid=oid, new_price=2.000)
    # Queue ahead refreshed to 200.
    assert px._orders[oid].queue_ahead == 200.0


def test_amend_rejected_when_crosses_book() -> None:
    px, _, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.010, bid_size=10.0, ask_size=10.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    out = px.amend_order(symbol="X", oid=oid, new_price=2.020)  # crosses ask
    assert out["status"] == "rejected"
    assert out["code"] == "post_only_cross"


# ---------------------------------------------------------------------------
# Position + realised PnL accumulation
# ---------------------------------------------------------------------------


def test_position_pnl_open_then_close_long() -> None:
    """BUY 10 @ 2.00, then SELL 10 @ 2.05 → realised PnL = +0.50 USD/lot
    × 10 = +5.00."""
    px, clock, _ = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    assert px.current_position_qty() == 10.0
    assert px.current_avg_entry() == pytest.approx(2.000)
    # Now place a SELL at 2.05 against an ask of 2.040 (so SELL doesn't cross).
    px.process_book_event(bid=2.040, ask=2.060, bid_size=10.0, ask_size=0.0)
    px.place_post_only_limit(
        symbol="X", is_buy=False, sz=10.0, limit_px=2.050
    )
    _advance(clock, 0.025)
    # BUY aggressor at 2.05 lifts our offer.
    px.process_trade_event(price=2.050, size=10.0, side="BUY")
    assert px.current_position_qty() == 0.0
    assert px.current_realized_pnl() == pytest.approx(0.50)


def test_position_pnl_open_then_flip_short() -> None:
    """BUY 10 @ 2.00, then SELL 15 @ 2.10 → realised = (2.10-2.00)*10 = 1.00;
    residual 5 short at 2.10."""
    px, clock, _ = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=10.0, side="SELL")

    px.process_book_event(bid=2.090, ask=2.110, bid_size=20.0, ask_size=0.0)
    px.place_post_only_limit(
        symbol="X", is_buy=False, sz=15.0, limit_px=2.100
    )
    _advance(clock, 0.025)
    px.process_trade_event(price=2.100, size=15.0, side="BUY")
    assert px.current_position_qty() == -5.0
    assert px.current_avg_entry() == pytest.approx(2.100)
    assert px.current_realized_pnl() == pytest.approx(1.00)


def test_partial_fill_leaves_remainder_resting() -> None:
    """A trade smaller than our size partials us; remainder rests on."""
    px, clock, _ = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=3.0, side="SELL")
    assert px.current_position_qty() == 3.0
    assert px._orders[oid].remaining_size == 7.0


def test_partial_fill_oversized_trade_caps_at_remaining() -> None:
    """Trade larger than our remaining size only fills what we have."""
    px, clock, _ = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(
        symbol="X", is_buy=True, sz=5.0, limit_px=2.000
    )
    _advance(clock, 0.025)
    # Trade of 50 — we only get 5.
    px.process_trade_event(price=2.000, size=50.0, side="SELL")
    assert px.current_position_qty() == 5.0


def test_maker_rebate_credits_negative_fees() -> None:
    """Default fee_maker_bps = -0.5 → negative fee (rebate) per fill."""
    px, clock, _ = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    # fee = 10 * 2.0 * -0.5 / 1e4 = -0.001
    assert px.current_fees() == pytest.approx(-0.001)


# ---------------------------------------------------------------------------
# Self-trade prevention
# ---------------------------------------------------------------------------


def test_self_trade_prevention_via_self_cross_rejection() -> None:
    """We resting BUY @ 2.05 + we resting SELL @ 2.04 would self-trade.
    The SELL placement is rejected before the cross can happen."""
    px, _, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.100, bid_size=10.0, ask_size=10.0)
    buy = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=5.0, limit_px=2.050
    )
    assert buy["status"] == "ok"
    sell = px.place_post_only_limit(
        symbol="X", is_buy=False, sz=5.0, limit_px=2.040
    )
    assert sell["status"] == "rejected"


# ---------------------------------------------------------------------------
# Smoke harness — verify every adapter method is callable
# ---------------------------------------------------------------------------


def test_full_protocol_surface_smoke() -> None:
    """Call every method on the adapter — no exceptions allowed.

    Acceptance criterion: 'feeding it through a small smoke harness
    that calls every method'."""
    px, clock, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)

    # symbol_spec property
    spec = px.symbol_spec
    assert spec is not None

    # market / account reads
    bba = px.fetch_best_bid_ask("X")
    assert bba.best_bid == 2.000
    assert px.fetch_position("0xabc", "X") is not None
    assert px.fetch_account_snapshot("0xabc") is not None
    assert px.fetch_open_orders_raw("0xabc") == []
    assert px.fetch_recent_fills_raw("0xabc", "X") == []

    # order lifecycle
    r = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=1.0, limit_px=1.999
    )
    assert r["status"] == "ok"
    oid = r["oid"]
    cloid = px.make_client_order_id(
        symbol="X", side=Side.BUY, quote_cycle_id="q1", price=1.999, size=1.0
    )
    assert cloid.startswith("paper-")

    r2 = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=1.0, limit_px=1.998,
        client_order_id=cloid,
    )
    assert r2["status"] == "ok"

    px.amend_order(symbol="X", oid=oid, new_price=1.997)
    px.cancel_order(symbol="X", oid=oid)
    px.cancel_order_by_cloid(symbol="X", client_order_id=cloid)

    status = px.query_order_status_by_cloid("0xabc", cloid)
    assert status is not None

    # Interpret-* round-trips
    oid_p, outcome_p, _ = px.interpret_place_response(r)
    assert oid_p == oid and outcome_p == "accepted"
    cancel_kind, _ = px.interpret_cancel_response({"status": "ok"})
    assert cancel_kind == "success"
    _, status_kind, _ = px.interpret_order_status_response({"status": "open", "oid": 1})
    assert status_kind == "open"

    # rest_runtime_counters
    counters = px.rest_runtime_counters()
    assert counters["place_total"] >= 2

    # market_close (no position open at this point → no_position)
    out = px.market_close("X")
    assert out["status"] == "no_position"


def test_market_close_against_long_position() -> None:
    px, clock, _ = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    assert px.current_position_qty() == 10.0
    # Now market-close against the bid.
    px.process_book_event(bid=2.020, ask=2.030, bid_size=100.0, ask_size=100.0)
    out = px.market_close("X")
    assert out["status"] == "ok"
    assert out["filled_sz"] == 10.0
    assert px.current_position_qty() == 0.0


# ---------------------------------------------------------------------------
# Cloid stability
# ---------------------------------------------------------------------------


def test_make_client_order_id_deterministic() -> None:
    """Same inputs → byte-identical cloid (replay determinism)."""
    px1, _, _ = _executor()
    px2, _, _ = _executor()
    a = px1.make_client_order_id(
        symbol="TON-USDT-SWAP", side=Side.BUY,
        quote_cycle_id="q-42", price=2.041, size=7.0,
    )
    b = px2.make_client_order_id(
        symbol="TON-USDT-SWAP", side=Side.BUY,
        quote_cycle_id="q-42", price=2.041, size=7.0,
    )
    assert a == b


def test_make_client_order_id_varies_by_input() -> None:
    px, _, _ = _executor()
    a = px.make_client_order_id(
        symbol="X", side=Side.BUY, quote_cycle_id="q1", price=2.0, size=1.0
    )
    b = px.make_client_order_id(
        symbol="X", side=Side.SELL, quote_cycle_id="q1", price=2.0, size=1.0
    )
    assert a != b


# ---------------------------------------------------------------------------
# Order-update events emitted on every state change
# ---------------------------------------------------------------------------


def test_place_emits_open_order_update() -> None:
    px, _, sink = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    events = _drain(sink)
    opens = [
        e for e in events
        if isinstance(e, PrivateOrderUpdateEvent) and e.status == "open"
    ]
    assert len(opens) == 1
    assert opens[0].limit_px == 2.000
    assert opens[0].orig_sz == 10.0


def test_cancel_lapse_emits_canceled_event() -> None:
    px, clock, sink = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    _advance(clock, 0.025)
    px.cancel_order(symbol="X", oid=oid)
    _advance(clock, 0.020)
    # Trigger sweep via a book event.
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    events = _drain(sink)
    canceled = [
        e for e in events
        if isinstance(e, PrivateOrderUpdateEvent) and e.status == "canceled"
    ]
    assert len(canceled) == 1


# ---------------------------------------------------------------------------
# Fill order-update faithfulness (regression: emit-before-decrement bug)
#
# The bot retires a WorkingOrder ONLY on a terminal order-update
# (``_map_hl_ws_order_status`` → filled/canceled/rejected). The
# PrivateFillEvent path (``ingest_hl_fill_raw``) updates position/PnL
# but never touches the order_store. So a full fill MUST surface as a
# terminal ``filled`` order-update with remaining 0 — otherwise the WO
# lingers, the next REST reconcile sees it missing on the (already
# reaped) exchange, and the bot enters the
# exchange_mismatch → side_unresolved → orphan-cancel cascade that
# floods the replay log. The old code emitted the order-update from
# inside ``_apply_fill`` BEFORE the size was decremented, so a full
# fill went out as ``open`` carrying the full pre-fill size and a
# terminal ``filled`` was NEVER produced.
# ---------------------------------------------------------------------------


def test_full_fill_emits_terminal_filled_order_update() -> None:
    """A full fill emits exactly one terminal ``filled`` with remaining 0."""
    px, clock, sink = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=10.0, limit_px=2.000)
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    updates = [
        e for e in _drain(sink) if isinstance(e, PrivateOrderUpdateEvent)
    ]
    # Place-time "open" then the terminal "filled".
    assert [u.status for u in updates] == ["open", "filled"]
    filled = updates[-1]
    assert filled.remaining_sz == 0.0
    assert filled.orig_sz == 10.0


def test_fill_event_precedes_terminal_order_update() -> None:
    """Ordering contract: the PrivateFillEvent is enqueued BEFORE the
    terminal order-update, mirroring the venue (fill lands, then the
    order goes terminal). The bot drains FIFO, so it applies the fill
    to position first, then retires the WO."""
    px, clock, sink = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=10.0, limit_px=2.000)
    _advance(clock, 0.025)
    _drain(sink)  # discard the place-time "open"
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    events = _drain(sink)
    assert isinstance(events[0], PrivateFillEvent)
    assert isinstance(events[1], PrivateOrderUpdateEvent)
    assert events[1].status == "filled"


def test_partial_fills_report_true_remaining_then_filled() -> None:
    """A sequence of partials carries the TRUE post-decrement remaining
    (open@7 → open@4 → filled@0), not the stale pre-fill size, and the
    completing trade emits a terminal ``filled``."""
    px, clock, sink = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=10.0, limit_px=2.000)
    _advance(clock, 0.025)
    _drain(sink)  # discard the place-time "open"
    px.process_trade_event(price=2.000, size=3.0, side="SELL")  # remaining 7
    px.process_trade_event(price=2.000, size=3.0, side="SELL")  # remaining 4
    px.process_trade_event(price=2.000, size=4.0, side="SELL")  # remaining 0
    updates = [
        e for e in _drain(sink) if isinstance(e, PrivateOrderUpdateEvent)
    ]
    assert [(u.status, u.remaining_sz) for u in updates] == [
        ("open", 7.0),
        ("open", 4.0),
        ("filled", 0.0),
    ]


# ---------------------------------------------------------------------------
# No network sockets opened during a synthetic run
# ---------------------------------------------------------------------------


def test_no_sockets_opened_during_synthetic_run() -> None:
    """Acceptance §2.5: 'No network sockets opened during a 1000-event
    synthetic run.' We can't easily prove a negative across every
    OS-level fd, but we can patch socket.socket to fail and verify
    a 1000-event run completes."""
    import socket as _socket
    original = _socket.socket
    sockets_opened: list = []

    def _tracking_socket(*args, **kwargs):
        sockets_opened.append((args, kwargs))
        return original(*args, **kwargs)

    _socket.socket = _tracking_socket  # type: ignore[assignment]
    try:
        px, clock, _ = _executor()
        px.process_book_event(bid=1.999, ask=2.010, bid_size=10.0, ask_size=10.0)
        # 1000 events: 500 book updates + 500 trade events.
        for i in range(500):
            _advance(clock, 0.001)
            px.process_book_event(
                bid=1.999 + 0.001 * (i % 5),
                ask=2.010 + 0.001 * (i % 5),
                bid_size=10.0,
                ask_size=10.0,
            )
            px.process_trade_event(price=2.000, size=0.1, side="SELL")
    finally:
        _socket.socket = original  # type: ignore[assignment]
    assert sockets_opened == [], "paper executor must not open sockets"


# ---------------------------------------------------------------------------
# BUG-G (v1.5.303) — fill-attribution diagnostic
# ---------------------------------------------------------------------------


def test_fill_attribution_quiet_market_not_addressable() -> None:
    """A trade whose price does NOT cross our resting order counts as
    'seen' but not 'addressable' — the quiet-market signature. With no
    addressable volume, queue_block_ratio / fill_ratio stay 0."""
    px, clock, _ = _executor()
    px.process_book_event(bid=1.999, ask=2.010, bid_size=0.0, ask_size=50.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=10.0, limit_px=2.000)
    _advance(clock, 0.025)
    # SELL prints ABOVE our bid (2.005 > 2.000) — never reaches our order.
    px.process_trade_event(price=2.005, size=7.0, side="SELL")
    snap = px.fill_attribution_snapshot()
    assert snap["trade_prints_seen"] == 1
    assert snap["trade_base_size_seen"] == pytest.approx(7.0)
    assert snap["trade_base_size_addressable"] == pytest.approx(0.0)
    assert snap["queue_ahead_absorbed_base"] == pytest.approx(0.0)
    assert snap["filled_base_from_trades"] == pytest.approx(0.0)
    assert snap["fills_blocked_by_queue_only"] == 0
    assert snap["queue_block_ratio"] == pytest.approx(0.0)
    assert snap["fill_ratio"] == pytest.approx(0.0)


def test_fill_attribution_back_of_queue_blocks_fill() -> None:
    """A crossing trade fully absorbed by queue-ahead is 'addressable'
    but yields zero fill — the back-of-queue signature: queue_block_ratio
    ≈ 1.0, fill_ratio ≈ 0, and one fills_blocked_by_queue_only tick."""
    px, clock, _ = _executor()
    # 100 lots resting ahead of us at 2.000.
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=10.0, limit_px=2.000)
    _advance(clock, 0.025)
    # SELL crosses our price but only 10 lots — eaten by the 100-lot queue.
    px.process_trade_event(price=2.000, size=10.0, side="SELL")
    snap = px.fill_attribution_snapshot()
    assert snap["trade_base_size_addressable"] == pytest.approx(10.0)
    assert snap["queue_ahead_absorbed_base"] == pytest.approx(10.0)
    assert snap["filled_base_from_trades"] == pytest.approx(0.0)
    assert snap["fills_blocked_by_queue_only"] == 1
    assert snap["queue_block_ratio"] == pytest.approx(1.0)
    assert snap["fill_ratio"] == pytest.approx(0.0)
    assert px.current_position_qty() == 0.0


def test_fill_attribution_records_reaching_fills() -> None:
    """After the queue drains, the next crossing trade reaches us:
    filled_base_from_trades accrues, fill_ratio climbs, and the filling
    trade does NOT increment fills_blocked_by_queue_only."""
    px, clock, _ = _executor()
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=5.0, limit_px=2.000)
    _advance(clock, 0.025)
    # Drain the 100-lot queue (blocked, no fill).
    px.process_trade_event(price=2.000, size=100.0, side="SELL")
    # Next 5-lot SELL now reaches us and fills.
    px.process_trade_event(price=2.000, size=5.0, side="SELL")
    snap = px.fill_attribution_snapshot()
    # Both trades crossed our price → both addressable.
    assert snap["trade_base_size_addressable"] == pytest.approx(105.0)
    assert snap["queue_ahead_absorbed_base"] == pytest.approx(100.0)
    assert snap["filled_base_from_trades"] == pytest.approx(5.0)
    # Only the first (queue-draining) trade produced zero fill.
    assert snap["fills_blocked_by_queue_only"] == 1
    assert snap["queue_block_ratio"] == pytest.approx(100.0 / 105.0)
    assert snap["fill_ratio"] == pytest.approx(5.0 / 105.0)
    assert px.current_position_qty() == 5.0


def test_fill_attribution_snapshot_keys_stable() -> None:
    """The snapshot dict must expose the full, stable key set the replay
    report's fill_attribution block depends on."""
    px, _, _ = _executor()
    snap = px.fill_attribution_snapshot()
    assert set(snap) == {
        "trade_prints_seen",
        "trade_base_size_seen",
        "trade_base_size_addressable",
        "queue_ahead_absorbed_base",
        "filled_base_from_trades",
        "fills_blocked_by_queue_only",
        "queue_block_ratio",
        "fill_ratio",
    }
    # Fresh executor → all zeros, no divide-by-zero on empty addressable.
    assert snap["queue_block_ratio"] == pytest.approx(0.0)
    assert snap["fill_ratio"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# queue_ahead_fraction — backtest-only fill-rate lever (v1.5.316)
#
# Scales the displayed top-of-book size that ``_estimate_queue_ahead``
# reports as the "queue ahead of us" floor. 1.0 (default) = the full
# pessimistic baseline (byte-identical to the pre-knob behaviour);
# <1.0 = closer to the front of the queue = higher fill rate; 0.0 =
# front-of-queue. Fills stay gated on REAL recorded trades crossing our
# price — only the assumed queue position moves.
# ---------------------------------------------------------------------------


def _executor_frac(
    frac: float, *, start_ns: int = 1_700_000_000_000_000_000
) -> tuple[PaperExecutor, ReplayClock, "Queue"]:
    """Like ``_executor`` but with an explicit ``queue_ahead_fraction``."""
    clock = ReplayClock(start_t_ns=start_ns)
    sink: Queue = Queue()
    cfg = PaperExecutorConfig(
        sim_place_latency_s=0.020,
        sim_cancel_latency_s=0.015,
        fee_maker_bps=-0.5,
        fee_taker_bps=1.0,
        queue_ahead_fraction=frac,
    )
    px = PaperExecutor(
        clock=clock,
        private_event_sink=sink,
        config=cfg,
        symbol="TON-USDT-SWAP",
    )
    return px, clock, sink


def test_queue_ahead_fraction_defaults_to_one() -> None:
    """Default config keeps the pessimistic full-size-ahead baseline."""
    cfg = PaperExecutorConfig()
    assert cfg.queue_ahead_fraction == 1.0


def test_config_rejects_negative_queue_ahead_fraction() -> None:
    with pytest.raises(ValueError):
        PaperExecutorConfig(queue_ahead_fraction=-0.1)


def test_queue_ahead_fraction_zero_is_allowed() -> None:
    """0.0 (front-of-queue) is a valid extreme, not rejected."""
    cfg = PaperExecutorConfig(queue_ahead_fraction=0.0)
    assert cfg.queue_ahead_fraction == 0.0


def test_queue_ahead_fraction_default_byte_identical_queue_position() -> None:
    """At the default 1.0 the queue floor equals the full displayed size —
    proving ``x * 1.0`` leaves the pre-knob queue position unchanged."""
    px, _, _ = _executor_frac(1.0)
    px.process_book_event(bid=2.000, ask=2.010, bid_size=50.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    assert px._orders[oid].queue_ahead == 50.0


def test_queue_ahead_fraction_scales_queue_position() -> None:
    """fraction=0.5 halves the modelled queue-ahead floor (50 → 25)."""
    px, _, _ = _executor_frac(0.5)
    px.process_book_event(bid=2.000, ask=2.010, bid_size=50.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    assert px._orders[oid].queue_ahead == 25.0


def test_queue_ahead_fraction_scales_sell_side() -> None:
    """The fraction applies symmetrically to the ask side (ask_size * frac)."""
    px, _, _ = _executor_frac(0.5)
    px.process_book_event(bid=2.000, ask=2.010, bid_size=50.0, ask_size=80.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=False, sz=10.0, limit_px=2.010
    )
    oid = resp["oid"]
    assert px._orders[oid].queue_ahead == 40.0


def test_queue_ahead_fraction_zero_front_of_queue_position() -> None:
    """fraction=0.0 puts us at the front of the queue (floor 0)."""
    px, _, _ = _executor_frac(0.0)
    px.process_book_event(bid=2.000, ask=2.010, bid_size=50.0, ask_size=50.0)
    resp = px.place_post_only_limit(
        symbol="X", is_buy=True, sz=10.0, limit_px=2.000
    )
    oid = resp["oid"]
    assert px._orders[oid].queue_ahead == 0.0


def test_queue_ahead_fraction_default_blocks_small_trade() -> None:
    """Baseline reference for the contrast test below: behind 100 lots at
    fraction=1.0, a 60-lot SELL is fully absorbed by the queue → no fill."""
    px, clock, sink = _executor_frac(1.0)
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=5.0, limit_px=2.000)
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=60.0, side="SELL")
    fills = [e for e in _drain(sink) if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 0
    assert px.current_position_qty() == 0.0


def test_queue_ahead_fraction_half_fills_sooner() -> None:
    """Same book + same 60-lot SELL as the baseline, but fraction=0.5 drops
    the queue floor to 50, so the trade breaks through and fills us. This is
    the core lever behaviour: identical recorded trades, higher fill rate —
    fills stay gated on the REAL trade crossing our price (adverse-selection
    correlation preserved); only our assumed queue position moves."""
    px, clock, sink = _executor_frac(0.5)
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=5.0, limit_px=2.000)
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=60.0, side="SELL")
    fills = [e for e in _drain(sink) if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 1
    assert fills[0].sz == 5.0
    assert px.current_position_qty() == 5.0


def test_queue_ahead_fraction_zero_fills_immediately() -> None:
    """fraction=0.0 = front of queue: the first addressable trade fills us
    with no draining, even sitting behind 100 displayed lots."""
    px, clock, sink = _executor_frac(0.0)
    px.process_book_event(bid=2.000, ask=2.010, bid_size=100.0, ask_size=100.0)
    px.place_post_only_limit(symbol="X", is_buy=True, sz=5.0, limit_px=2.000)
    _advance(clock, 0.025)
    px.process_trade_event(price=2.000, size=5.0, side="SELL")
    fills = [e for e in _drain(sink) if isinstance(e, PrivateFillEvent)]
    assert len(fills) == 1
    assert px.current_position_qty() == 5.0
