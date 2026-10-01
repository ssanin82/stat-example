"""Binance Level 1 cross-venue cancel trigger + WS message handling.

Motivation: proposals/cross-reference.md. The overnight ETH session
showed clean MM symmetry but −1.53 bps mean markout — the informed
arb flow crosses us when GRVT's BBO hasn't yet caught up to Binance.
This module's production code lives in:
  * ``app/exchange/binance_public_ws.py`` — WS subscriber + state updates
  * ``app/execution.py::_maybe_cancel_on_binance_move`` — cancel trigger

Invariants pinned here:

  1. ``_build_ws_url`` renders a legal URL for any symbol case.
  2. Message handler parses valid payloads into ``state.binance_*`` fields.
  3. Malformed / partial / negative-priced messages are dropped (the
     thread survives; the bot doesn't crash on bad telemetry).
  4. Basis EWMA is computed when GRVT mid is present, not otherwise.
  5. The cancel trigger:
     a. Does nothing when disabled via ``BINANCE_WS_ENABLED=false``.
     b. Does nothing when Binance state is missing (not yet connected).
     c. Does nothing when the feed is stale beyond
        ``BINANCE_WS_FAIR_VALUE_MAX_AGE_SECONDS``.
     d. Does nothing when the resting-order price is within the
        configured bps band.
     e. Enqueues a cancel when an order is further than the threshold
        from the fair value.
     f. Applies the basis adjustment (``binance_mid + basis_ewma``).
     g. Only cancels orders in ACKED / PARTIAL status — not SENT /
        CANCEL_PENDING.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.exchange.binance_public_ws import (
    BinancePublicStream,
    _build_ws_url,
)
from app.enums import OrderStatus, RiskAction, Side
from app.execution import OrderManager
from app.models import BestBidAsk, WorkingOrder, QuoteDecision
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from app.utils.time import utc_now


# --------------------- URL construction ------------------------------


def test_build_ws_url_lowercases_symbol() -> None:
    """Binance uses lowercase symbols in stream paths."""
    assert (
        _build_ws_url("wss://fstream.binance.com/ws", "ETHUSDT")
        == "wss://fstream.binance.com/ws/ethusdt@bookTicker"
    )


def test_build_ws_url_strips_trailing_slash_in_base() -> None:
    assert (
        _build_ws_url("wss://fstream.binance.com/ws/", "ETHUSDT")
        == "wss://fstream.binance.com/ws/ethusdt@bookTicker"
    )


def test_build_ws_url_preserves_mixed_case_after_lowercasing_symbol() -> None:
    """Base URL is unchanged (may contain cased path segments); only
    the trailing symbol is downcased."""
    url = _build_ws_url("wss://foo.example.com/Path", "BtCuSdT")
    assert url == "wss://foo.example.com/Path/btcusdt@bookTicker"


# --------------------- Message parsing -------------------------------


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "BINANCE_WS_ENABLED": True,
        "BINANCE_BASIS_EWMA_ALPHA": 0.5,  # fast convergence in tests
        "BINANCE_CANCEL_ON_MOVE_BPS": 5.0,
        "BINANCE_WS_FAIR_VALUE_MAX_AGE_SECONDS": 15.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_stream_with_state(**settings_overrides) -> tuple[BinancePublicStream, BotState]:
    settings = _settings(**settings_overrides)
    state = BotState(settings)
    stream = BinancePublicStream(settings, state, on_bbo_callback=None)
    return stream, state


def test_valid_payload_updates_state_top_of_book() -> None:
    stream, state = _make_stream_with_state()
    payload = '{"u":1,"s":"ETHUSDT","b":"2340.12","B":"10.5","a":"2340.20","A":"7.2"}'
    stream._handle_raw_message(payload)
    assert state.binance_best_bid == pytest.approx(2340.12)
    assert state.binance_best_ask == pytest.approx(2340.20)
    assert state.binance_mid == pytest.approx((2340.12 + 2340.20) / 2.0)
    assert state.binance_bid_size == pytest.approx(10.5)
    assert state.binance_ask_size == pytest.approx(7.2)
    assert state.binance_last_message_wall_ts is not None


def test_basis_ewma_requires_grvt_mid() -> None:
    """Basis can only be computed when GRVT has a mid too. Before the
    public WS reports its first BBO, basis_ewma stays None."""
    stream, state = _make_stream_with_state()
    # No GRVT market yet → basis stays None.
    stream._handle_raw_message(
        '{"u":1,"s":"ETHUSDT","b":"2340.0","B":"1","a":"2340.10","A":"1"}'
    )
    assert state.binance_basis_ewma is None

    # Now set a GRVT mid — next Binance tick seeds basis_ewma.
    state.market = BestBidAsk(
        symbol="ETH", best_bid=2340.20, best_ask=2340.30, mid_price=2340.25,
        spread_bps=0.4, ts_local=utc_now(),
    )
    stream._handle_raw_message(
        '{"u":2,"s":"ETHUSDT","b":"2340.0","B":"1","a":"2340.10","A":"1"}'
    )
    # GRVT mid 2340.25, Binance mid 2340.05 → basis = +0.20.
    assert state.binance_basis_ewma == pytest.approx(0.20, abs=1e-6)


def test_basis_ewma_smooths_multiple_samples() -> None:
    """With alpha=0.5, the second sample is the average of the first
    sample and the new raw basis. Third sample is further smoothed."""
    stream, state = _make_stream_with_state(BINANCE_BASIS_EWMA_ALPHA=0.5)
    state.market = BestBidAsk(
        symbol="ETH", best_bid=2340.0, best_ask=2340.0, mid_price=2340.0,
        spread_bps=0.0, ts_local=utc_now(),
    )
    # Binance mid 2339.0 → raw basis = +1.0, seed.
    stream._handle_raw_message(
        '{"u":1,"s":"ETHUSDT","b":"2338.90","B":"1","a":"2339.10","A":"1"}'
    )
    assert state.binance_basis_ewma == pytest.approx(1.0, abs=1e-6)
    # Binance mid 2341.0 → raw basis = -1.0. EWMA = 0.5 * -1 + 0.5 * 1 = 0.
    stream._handle_raw_message(
        '{"u":2,"s":"ETHUSDT","b":"2340.90","B":"1","a":"2341.10","A":"1"}'
    )
    assert state.binance_basis_ewma == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",                                              # non-JSON bytes
        "",                                                       # empty
        "{}",                                                     # missing fields
        '{"b":"abc","a":"101"}',                                  # non-numeric
        '{"b":"-10","a":"5"}',                                    # negative bid
        '{"b":"10","a":"-5"}',                                    # negative ask
        '{"b":"0","a":"5"}',                                      # zero bid
        '{"b":"10","a":"5"}',                                     # ask < bid (crossed book)
        '["not","a","dict"]',                                     # wrong type
    ],
)
def test_malformed_payload_is_silently_dropped(payload) -> None:
    stream, state = _make_stream_with_state()
    # No raise, no state mutation.
    stream._handle_raw_message(payload)
    assert state.binance_best_bid is None
    assert state.binance_best_ask is None
    assert state.binance_mid is None


# --------------------- Cancel trigger --------------------------------


def _make_order_manager_with_orders(
    *,
    binance_enabled: bool = True,
    binance_mid: float = 2340.10,
    binance_best_bid: float | None = None,
    binance_best_ask: float | None = None,
    basis_ewma: float = 0.05,
    feed_age_s: float = 1.0,
    bid_px: float = 2340.0,
    ask_px: float = 2340.20,
    bid_status: OrderStatus = OrderStatus.ACKED,
    ask_status: OrderStatus = OrderStatus.ACKED,
    cancel_on_move_bps: float = 5.0,
    max_age_s: float = 15.0,
) -> tuple[OrderManager, BotState]:
    """Build a real OrderManager with the specified state fixture."""
    path = Path(tempfile.gettempdir()) / f"mm_bxc_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "BINANCE_WS_ENABLED": binance_enabled,
            "BINANCE_CANCEL_ON_MOVE_BPS": cancel_on_move_bps,
            "BINANCE_WS_FAIR_VALUE_MAX_AGE_SECONDS": max_age_s,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    # Stub out the cancel enqueue so we can observe call count without
    # spinning up the outbound dispatcher.
    om._enqueue_cancel_quote_path = MagicMock(return_value=True)

    # Populate Binance state. v1.5.46: bid/ask are now required by the
    # cancel gate (side-aware reference price). Default to a symmetric
    # half-cent spread around ``binance_mid`` when the caller didn't
    # override — that produces a ~0.4 bp ETH spread, narrower than the
    # threshold used in all current tests, so the new gate behaves
    # equivalently to the pre-v1.5.46 mid-only behaviour they were
    # written against.
    state.binance_mid = binance_mid
    state.binance_best_bid = (
        binance_best_bid if binance_best_bid is not None else binance_mid - 0.05
    )
    state.binance_best_ask = (
        binance_best_ask if binance_best_ask is not None else binance_mid + 0.05
    )
    state.binance_basis_ewma = basis_ewma
    state.binance_last_message_wall_ts = datetime.now(timezone.utc) - timedelta(seconds=feed_age_s)

    # Populate working orders.
    state.working_bid = WorkingOrder(
        order_id_local=f"b_{uuid.uuid4().hex[:8]}",
        order_id_exchange=100_001,
        client_order_id="cloid_b",
        symbol="ETH", side=Side.BUY,
        price=bid_px, size=0.05, post_only=True, status=bid_status,
    )
    state.working_ask = WorkingOrder(
        order_id_local=f"a_{uuid.uuid4().hex[:8]}",
        order_id_exchange=100_002,
        client_order_id="cloid_a",
        symbol="ETH", side=Side.SELL,
        price=ask_px, size=0.05, post_only=True, status=ask_status,
    )
    return om, state


def test_disabled_trigger_never_fires() -> None:
    om, _state = _make_order_manager_with_orders(binance_enabled=False)
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_not_called()


def test_missing_binance_state_noop() -> None:
    """First run, Binance hasn't connected yet → no state → no cancel."""
    om, state = _make_order_manager_with_orders()
    # Clear Binance state.
    state.binance_mid = None
    state.binance_basis_ewma = None
    state.binance_last_message_wall_ts = None
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_not_called()


def test_stale_feed_noop() -> None:
    """Feed older than the max-age threshold → don't trust it."""
    om, _state = _make_order_manager_with_orders(feed_age_s=60.0, max_age_s=15.0)
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_not_called()


def test_orders_within_band_are_not_cancelled() -> None:
    """Mid=2340.15 (after +0.05 basis), orders at 2340.00/2340.20 → within
    5 bps = $1.17 at $2340. No cancels."""
    om, _state = _make_order_manager_with_orders(
        binance_mid=2340.10,
        basis_ewma=0.05,       # fair value = 2340.15
        bid_px=2340.00,        # 2340.15 - 2340.00 = 0.15 → 0.64 bps
        ask_px=2340.20,        # 2340.20 - 2340.15 = 0.05 → 0.21 bps
        cancel_on_move_bps=5.0,
    )
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_not_called()


def test_bid_outside_band_triggers_cancel() -> None:
    """Fair value moved UP (now 2350.15). Our stale resting BID at
    2340.00 is 43 bps below — cancel it. Ask at 2350.20 is on the
    right side and close enough — left alone."""
    om, state = _make_order_manager_with_orders(
        binance_mid=2350.10,
        basis_ewma=0.05,       # fair value = 2350.15
        bid_px=2340.00,        # |2340.00 - 2350.15| / 2350.15 * 10000 ≈ 43.2 bps
        ask_px=2350.20,        # |2350.20 - 2350.15| / 2350.15 * 10000 ≈ 0.2 bps
        cancel_on_move_bps=5.0,
    )
    om._maybe_cancel_on_binance_move()
    # Only one cancel — the stale BID.
    assert om._enqueue_cancel_quote_path.call_count == 1
    called_wo = om._enqueue_cancel_quote_path.call_args[0][0]
    assert called_wo.side == Side.BUY
    # v1.4.200: read via the new per-rung accessor instead of the
    # deprecated ``state.working_bid`` shim (the warning was firing
    # in CI). Inside-rung is ``level_idx=0``.
    assert called_wo is state.get_working_order(Side.BUY, 0)


def test_ask_outside_band_triggers_cancel() -> None:
    """Fair value moved DOWN (now 2330.05). Our resting ASK at 2340.20
    is 43 bps above — cancel it. BID at 2330.00 is within threshold."""
    om, state = _make_order_manager_with_orders(
        binance_mid=2330.0,
        basis_ewma=0.05,       # fair value ≈ 2330.05
        bid_px=2330.00,
        ask_px=2340.20,        # ≈ 43.5 bps high
        cancel_on_move_bps=5.0,
    )
    om._maybe_cancel_on_binance_move()
    assert om._enqueue_cancel_quote_path.call_count == 1
    called_wo = om._enqueue_cancel_quote_path.call_args[0][0]
    assert called_wo.side == Side.SELL
    # v1.4.200: read via the new per-rung accessor (see BUY-side
    # twin test above for rationale).
    assert called_wo is state.get_working_order(Side.SELL, 0)


def test_both_sides_trigger_when_both_stale() -> None:
    """A big sudden move can leave BOTH sides stale. Both get cancelled."""
    om, _state = _make_order_manager_with_orders(
        binance_mid=2400.0,
        basis_ewma=0.0,        # fair value = 2400 — large shift from the orders
        bid_px=2340.0,         # way below
        ask_px=2340.20,        # way below too (a BIG up-move leaves both low)
        cancel_on_move_bps=5.0,
    )
    om._maybe_cancel_on_binance_move()
    assert om._enqueue_cancel_quote_path.call_count == 2
    called_sides = {c.args[0].side for c in om._enqueue_cancel_quote_path.call_args_list}
    assert called_sides == {Side.BUY, Side.SELL}


def test_basis_adjustment_is_applied() -> None:
    """A GRVT premium of +10 over Binance means fair value = mid + 10.
    An order at binance_mid is NOT within band (10-bp off fair value);
    an order at fair value IS within band. Pin the basis application."""
    # Case A: fair value = 2340.00 + 10 = 2350.00.
    # Order at 2340.00 is ~42 bps below fair value → cancel.
    om, _state = _make_order_manager_with_orders(
        binance_mid=2340.0,
        basis_ewma=10.0,
        bid_px=2340.0,   # at Binance mid, NOT at fair value
        ask_px=2350.0,   # at fair value
        cancel_on_move_bps=5.0,
    )
    om._maybe_cancel_on_binance_move()
    sides_cancelled = {c.args[0].side for c in om._enqueue_cancel_quote_path.call_args_list}
    assert Side.BUY in sides_cancelled
    assert Side.SELL not in sides_cancelled


def test_in_flight_status_is_skipped() -> None:
    """Orders in SENT or CANCEL_PENDING are NOT cancelled by this
    trigger — those states are handled by other paths (ack-wait watchdog,
    reconcile). The Binance trigger only ever touches resting orders."""
    om, _state = _make_order_manager_with_orders(
        binance_mid=2400.0, basis_ewma=0.0,
        bid_px=2340.0, ask_px=2340.20,  # both far from fair
        bid_status=OrderStatus.SENT,
        ask_status=OrderStatus.CANCEL_PENDING,
    )
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_not_called()


def test_no_working_orders_noop() -> None:
    om, state = _make_order_manager_with_orders()
    state.working_bid = None
    state.working_ask = None
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_not_called()


def test_side_aware_reference_eliminates_half_spread_artefact() -> None:
    """v1.5.46 regression pin: bot's SELL at OKX's ask must NOT trigger
    on the half-OKX-spread artefact when there is no real cross-venue
    dislocation.

    Setup mimics a wide-OKX regime (Binance spread = 10 bps), the bot's
    SELL sitting at Binance's ask: cross-venue dislocation = 0. Under
    the pre-v1.5.46 mid-only comparison, ``delta_bps`` = half of the
    Binance spread = 5 bps (a structural artefact). With a 4-bp
    threshold the gate would CANCEL the order despite no genuine
    drift — the production wedge observed on
    snapshot ``v1.5.45-260523-104811-prod.okx.ton.usdt.perp`` (100
    cancels in 22 min, zero trades).

    Post-fix: SELL vs ``ref_ask = binance_best_ask + basis`` → delta = 0
    → no cancel. The same setup with the SELL pulled meaningfully past
    the ask STILL fires the gate; this is just the artefact case.
    """
    # Binance has a 10-bp spread; basis zero for simplicity.
    # bid = 2340.00, ask = 2342.34, mid = 2341.17 → spread ≈ 10 bps.
    om, _state = _make_order_manager_with_orders(
        binance_mid=2341.17,
        binance_best_bid=2340.00,
        binance_best_ask=2342.34,
        basis_ewma=0.0,
        # Bot's SELL sitting AT Binance ask (= ref_ask post-fix).
        # Under v1.5.45 logic this was 5 bps above fair_value (mid).
        ask_px=2342.34,
        # BUY also AT touch (= ref_bid).
        bid_px=2340.00,
        cancel_on_move_bps=4.0,  # tight threshold the bug fired through
    )
    om._maybe_cancel_on_binance_move()
    om._enqueue_cancel_quote_path.assert_not_called()


def test_real_dislocation_still_fires_under_side_aware_reference() -> None:
    """Symmetric companion to the artefact pin: when there IS a real
    cross-venue dislocation beyond half-spread, the gate still fires.

    Same Binance BBO as the artefact test, but the SELL is now 8 bps
    above the Binance ask (genuine drift in the bot's favour-resisting
    direction). Threshold 4 bps → cancel.
    """
    om, _state = _make_order_manager_with_orders(
        binance_mid=2341.17,
        binance_best_bid=2340.00,
        binance_best_ask=2342.34,
        basis_ewma=0.0,
        # 8 bps above Binance ask = 2342.34 * (1 + 8/10000) ≈ 2344.21.
        ask_px=2344.21,
        bid_px=2340.00,
        cancel_on_move_bps=4.0,
    )
    om._maybe_cancel_on_binance_move()
    assert om._enqueue_cancel_quote_path.call_count == 1
    called_wo = om._enqueue_cancel_quote_path.call_args[0][0]
    assert called_wo.side == Side.SELL


def test_threshold_is_configurable() -> None:
    """Tighter threshold = more sensitive = more cancels."""
    # Same order prices as test_orders_within_band_are_not_cancelled
    # but pulled half a cent past each side's reference touch, with a
    # 0.1-bp threshold (sub-cent): both sides trigger.
    #
    # v1.5.46: the gate is now side-aware — BUY vs ref_bid (mid - 0.05),
    # SELL vs ref_ask (mid + 0.05). The previous test setup placed
    # ask_px exactly at the synthetic Binance ask (2340.20 ≈ 2340.10 +
    # 0.05 + basis 0.05), so delta_ask = 0 bps and the gate wouldn't
    # fire even at a 0.1-bp threshold. Move both sides one tick OUT of
    # the touch so each is ~0.43 bp away — comfortably > 0.1 bp.
    om, _state = _make_order_manager_with_orders(
        binance_mid=2340.10, basis_ewma=0.05,
        bid_px=2339.90, ask_px=2340.30,
        cancel_on_move_bps=0.1,
    )
    om._maybe_cancel_on_binance_move()
    assert om._enqueue_cancel_quote_path.call_count == 2


# --------------------- DB event persistence --------------------------
#
# The cancel trigger must leave a durable trace in ``bot_events`` so
# post-hoc analysis (``explain_moment.py``, DBeaver) can count / align
# cross-venue cancels with fills and P&L. stdout-only logging vanishes
# after process restart — the DB is the source of truth for the
# overnight observability workflow.


def test_trigger_writes_bot_event_to_db() -> None:
    """When the trigger fires, a row lands in ``bot_events``
    with event_type ``binance_cross_venue_cancel`` and the full
    payload (fair_value, delta_bps, order price, etc.)."""
    om, _state = _make_order_manager_with_orders(
        binance_mid=2350.10,
        basis_ewma=0.05,       # fair value = 2350.15
        bid_px=2340.00,        # ~43 bps below → cancel
        ask_px=2350.20,        # close enough → no cancel
        cancel_on_move_bps=5.0,
    )
    om._maybe_cancel_on_binance_move()

    events = om._storage.recent_bot_events(limit=50)
    cancels = [e for e in events if e["event_type"] == "binance_cross_venue_cancel"]
    assert len(cancels) == 1, (
        f"expected exactly 1 binance_cross_venue_cancel bot_event row, got "
        f"{len(cancels)}; full event list: {[e['event_type'] for e in events]}"
    )
    row = cancels[0]
    # Payload is stored as JSON in ``payload_json``; shape must carry
    # every field ``explain_moment.py`` needs for reconstruction.
    # v1.5.46: ``binance_best_bid`` / ``binance_best_ask`` / ``reference_price``
    # are new; ``fair_value`` / ``delta_bps`` are retained but ``delta_bps``
    # now measures order-price vs side-aware reference (ref_bid for BUY,
    # ref_ask for SELL), not vs ``fair_value``.
    import json as _json
    payload = _json.loads(row["payload_json"])
    for required in (
        "side", "order_price", "binance_mid", "binance_best_bid",
        "binance_best_ask", "basis_ewma", "fair_value", "reference_price",
        "delta_bps", "threshold_bps", "binance_feed_age_s",
        "order_id_local", "order_id_exchange",
    ):
        assert required in payload, f"missing field {required!r} in payload: {payload}"
    assert payload["side"] == "BUY"
    assert payload["order_price"] == pytest.approx(2340.00)
    assert payload["fair_value"] == pytest.approx(2350.15)
    # v1.5.46: reference_price for BUY = binance_best_bid + basis.
    # Default fixture sets best_bid = mid - 0.05 = 2350.05; + basis 0.05
    # → 2350.10. Pin the new field so downstream consumers can rely on
    # the per-side anchor being present and well-formed.
    assert payload["reference_price"] == pytest.approx(2350.10)
    assert payload["delta_bps"] > 5.0


def test_noop_does_not_write_bot_event() -> None:
    """When the trigger does NOT fire (orders within band), no
    DB row is written. Prevents false-positive cancel counts in
    post-hoc analysis."""
    om, _state = _make_order_manager_with_orders(
        binance_mid=2340.10,
        basis_ewma=0.05,
        bid_px=2340.00,        # within 5 bps
        ask_px=2340.20,        # within 5 bps
        cancel_on_move_bps=5.0,
    )
    om._maybe_cancel_on_binance_move()
    events = om._storage.recent_bot_events(limit=50)
    cancels = [e for e in events if e["event_type"] == "binance_cross_venue_cancel"]
    assert cancels == []


def test_both_sides_trigger_writes_two_bot_events() -> None:
    """Two triggers in one pass → two DB rows. Each cancel is
    independently recorded so post-hoc analysis can count sides."""
    om, _state = _make_order_manager_with_orders(
        binance_mid=2400.0,
        basis_ewma=0.0,        # fair value = 2400 — both sides stale low
        bid_px=2340.0,
        ask_px=2340.20,
        cancel_on_move_bps=5.0,
    )
    om._maybe_cancel_on_binance_move()
    events = om._storage.recent_bot_events(limit=50)
    cancels = [e for e in events if e["event_type"] == "binance_cross_venue_cancel"]
    assert len(cancels) == 2
    import json as _json
    sides = {_json.loads(e["payload_json"])["side"] for e in cancels}
    assert sides == {"BUY", "SELL"}


# --------------------- State/API surface -----------------------------
#
# Binance fields must be exposed through ``snapshot_dict`` (served by
# ``GET /state/current`` and captured by ``scripts/stats_snapshot.py``
# as ``state_current.json``) and through ``status_flags_dict`` (served
# by ``GET /status`` and ``GET /health``). These tests pin the key
# names and shape so a future refactor doesn't silently drop the
# fields — breaking the overnight diagnostic workflow.


_EXPECTED_BINANCE_KEYS = {
    "binance_ws_connected",
    "binance_ws_reconnect_count",
    "binance_ws_last_connect_ts",
    "binance_ws_last_message_ts",
    "binance_ws_seconds_since_last_message",
    "binance_best_bid",
    "binance_best_ask",
    "binance_bid_size",
    "binance_ask_size",
    "binance_mid",
    "binance_basis_ewma",
}


def _fresh_state(**overrides) -> BotState:
    settings = _settings(**overrides)
    return BotState(settings)


def test_snapshot_dict_exposes_binance_keys_with_nulls_when_empty() -> None:
    """Before the first Binance message, all fields are present but null
    (shape-stable contract for downstream tooling)."""
    state = _fresh_state()
    snap = state.snapshot_dict()
    missing = _EXPECTED_BINANCE_KEYS - snap.keys()
    assert not missing, f"snapshot_dict missing Binance keys: {missing}"
    assert snap["binance_ws_connected"] is False
    assert snap["binance_ws_reconnect_count"] == 0
    assert snap["binance_best_bid"] is None
    assert snap["binance_best_ask"] is None
    assert snap["binance_mid"] is None
    assert snap["binance_basis_ewma"] is None
    assert snap["binance_ws_last_message_ts"] is None
    assert snap["binance_ws_seconds_since_last_message"] is None


def test_status_flags_dict_exposes_binance_keys_with_nulls_when_empty() -> None:
    state = _fresh_state()
    flags = state.status_flags_dict()
    missing = _EXPECTED_BINANCE_KEYS - flags.keys()
    assert not missing, f"status_flags_dict missing Binance keys: {missing}"


def test_snapshot_dict_reflects_populated_binance_state() -> None:
    """After a Binance message lands, snapshot_dict reports live values
    and ``binance_ws_seconds_since_last_message`` is a non-negative float."""
    state = _fresh_state()
    state.binance_best_bid = 2340.10
    state.binance_best_ask = 2340.20
    state.binance_mid = 2340.15
    state.binance_bid_size = 12.5
    state.binance_ask_size = 8.3
    state.binance_basis_ewma = 0.05
    state.binance_ws_connected = True
    state.binance_ws_reconnect_count = 3
    msg_ts = datetime.now(timezone.utc) - timedelta(seconds=2.0)
    state.binance_last_message_wall_ts = msg_ts
    conn_ts = datetime.now(timezone.utc) - timedelta(seconds=120.0)
    state.binance_ws_last_connect_ts = conn_ts

    snap = state.snapshot_dict()
    assert snap["binance_best_bid"] == pytest.approx(2340.10)
    assert snap["binance_best_ask"] == pytest.approx(2340.20)
    assert snap["binance_mid"] == pytest.approx(2340.15)
    assert snap["binance_bid_size"] == pytest.approx(12.5)
    assert snap["binance_ask_size"] == pytest.approx(8.3)
    assert snap["binance_basis_ewma"] == pytest.approx(0.05)
    assert snap["binance_ws_connected"] is True
    assert snap["binance_ws_reconnect_count"] == 3
    assert snap["binance_ws_last_message_ts"] == msg_ts.isoformat()
    assert snap["binance_ws_last_connect_ts"] == conn_ts.isoformat()
    # 2 s ago → freshness between 1.5 and 4.0 s (wallclock slop tolerance).
    assert 1.5 <= snap["binance_ws_seconds_since_last_message"] <= 4.0


# ================== One-way latency measurement (Phase A) ==================
#
# The Binance Futures ``@bookTicker`` stream carries ``E`` (event time ms)
# and ``T`` (transaction time ms). Until these tests shipped we only
# recorded local receive wall time, so we couldn't measure the
# Binance-Tokyo → Railway-Singapore transport latency. The tracker
# field ``state.binance_public_ws_timing`` now receives a sample per
# message; these tests pin the capture + fallback behaviour so a future
# refactor of the handler doesn't silently regress the measurement.


def test_binance_ws_captures_exchange_timestamp_from_E_field() -> None:
    """Futures payload with ``E``: the tracker receives ``exchange_ts_ms``
    matching ``E`` exactly, and the one-way-delay metric gets populated
    (negative values would indicate a parsing bug / units mismatch)."""
    stream, state = _make_stream_with_state()
    # Tracker only exists when MARKET_DATA_TIMING_WINDOW_ENABLED is
    # truthy — default in our settings is true, so it should be
    # non-None. If the default ever flips, this test catches it.
    assert state.binance_public_ws_timing is not None, (
        "Binance timing tracker must be instantiated when "
        "MARKET_DATA_TIMING_WINDOW_ENABLED=true (the default)"
    )
    # E = now - 40 ms (simulating Binance Tokyo → Railway ~40ms).
    now_ms = int(time.time() * 1000.0)
    e_ms = now_ms - 40
    payload = (
        '{"e":"bookTicker","u":1,"s":"ETHUSDT",'
        '"b":"2340.10","B":"1","a":"2340.20","A":"1",'
        f'"T":{e_ms - 1},"E":{e_ms}' + "}"
    )
    stream._handle_raw_message(payload)

    samples = state.binance_public_ws_timing.recent_samples(limit=10)
    assert len(samples) == 1
    sample = samples[0]
    assert sample.exchange_ts_ms == e_ms, (
        f"tracker must receive E={e_ms}, got {sample.exchange_ts_ms}"
    )
    # One-way delay must be populated and non-negative.
    assert sample.exchange_to_local_receive_ms is not None
    assert sample.exchange_to_local_receive_ms >= 0, (
        f"one-way delay should be >= 0 for E in the past; "
        f"got {sample.exchange_to_local_receive_ms} — likely units or sign bug"
    )
    # Tolerance: our synthetic 40 ms gap plus test-harness slack.
    assert sample.exchange_to_local_receive_ms <= 500, (
        f"synthetic 40 ms gap, delay should be <500ms even under load; "
        f"got {sample.exchange_to_local_receive_ms}"
    )
    # No missing-exchange-ts anomaly when E is present.
    summary = state.binance_public_ws_timing.summary()
    assert summary["samples_with_exchange_ts"] == 1
    # Delay histogram should have at least one valid observation.
    assert summary["valid_one_way_delay_coverage_pct"] is not None


def test_binance_ws_falls_back_to_T_when_E_missing() -> None:
    """If ``E`` is missing but ``T`` is present (unusual, but guards
    against future feed-routing changes), the tracker uses ``T``."""
    stream, state = _make_stream_with_state()
    assert state.binance_public_ws_timing is not None
    t_ms = int(time.time() * 1000.0) - 25
    payload = (
        '{"u":1,"s":"ETHUSDT",'
        '"b":"2340.10","B":"1","a":"2340.20","A":"1",'
        f'"T":{t_ms}' + "}"
    )
    stream._handle_raw_message(payload)
    samples = state.binance_public_ws_timing.recent_samples(limit=10)
    assert len(samples) == 1
    assert samples[0].exchange_ts_ms == t_ms


def test_binance_ws_records_missing_exchange_ts_when_both_E_and_T_absent() -> None:
    """Spot-shaped payload (no ``E``, no ``T``): tracker still receives
    the sample (so the receive-cadence metric is not broken) but flags
    the exchange-ts as missing. The one-way-delay metric is not
    polluted with a fake value."""
    stream, state = _make_stream_with_state()
    assert state.binance_public_ws_timing is not None
    payload = (
        '{"u":1,"s":"ETHUSDT",'
        '"b":"2340.10","B":"1","a":"2340.20","A":"1"}'
    )
    stream._handle_raw_message(payload)
    samples = state.binance_public_ws_timing.recent_samples(limit=10)
    assert len(samples) == 1, "tracker should still log the sample for cadence"
    assert samples[0].exchange_ts_ms is None
    assert samples[0].exchange_to_local_receive_ms is None, (
        "one-way delay must be None when exchange ts is missing — "
        "do NOT substitute a zero or wall-clock fallback (that would "
        "silently corrupt the latency histogram)"
    )
    summary = state.binance_public_ws_timing.summary()
    assert summary["samples_with_exchange_ts"] == 0
    assert summary["total_samples"] == 1


def test_binance_ws_rejects_zero_or_negative_exchange_timestamps() -> None:
    """Defensive: ``E=0`` is a placeholder from a broken feed and must
    not appear as a valid sample. The tracker treats it as missing."""
    stream, state = _make_stream_with_state()
    assert state.binance_public_ws_timing is not None
    payload = (
        '{"u":1,"s":"ETHUSDT",'
        '"b":"2340.10","B":"1","a":"2340.20","A":"1",'
        '"E":0,"T":0}'
    )
    stream._handle_raw_message(payload)
    samples = state.binance_public_ws_timing.recent_samples(limit=10)
    assert len(samples) == 1
    assert samples[0].exchange_ts_ms is None


def test_binance_ws_latency_summary_has_shape_for_stats_snapshot() -> None:
    """The ``summary()`` shape must carry the same fields that the
    ``/market-data/timing-summary`` endpoint exposes. Feed 3 messages
    and confirm the metrics render — this is what
    ``stats_snapshot.py`` will capture as
    ``market-data_binance-timing-summary.json``."""
    stream, state = _make_stream_with_state()
    assert state.binance_public_ws_timing is not None
    now_ms = int(time.time() * 1000.0)
    for i in range(3):
        payload = (
            '{"e":"bookTicker","u":'
            f'{100 + i}'
            ',"s":"ETHUSDT",'
            f'"b":"{2340.10 + i * 0.01}","B":"1",'
            f'"a":"{2340.20 + i * 0.01}","A":"1",'
            f'"T":{now_ms - 40 + i * 100},"E":{now_ms - 40 + i * 100}' + "}"
        )
        stream._handle_raw_message(payload)
    summary = state.binance_public_ws_timing.summary()
    # Fields expected by the API notes (pin them explicitly so a
    # tracker refactor can't silently drop one).
    for field in (
        "symbol", "source_type", "total_samples",
        "current_buffer_size", "samples_with_exchange_ts",
        "exchange_gap_ms", "local_receive_gap_ms",
        "exchange_to_local_receive_ms", "receive_to_apply_ms",
    ):
        assert field in summary, f"summary missing field: {field}"
    assert summary["total_samples"] == 3
    assert summary["samples_with_exchange_ts"] == 3
