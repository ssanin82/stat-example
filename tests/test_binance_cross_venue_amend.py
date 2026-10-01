"""v1.4.21 — Binance cross-venue trigger routes through amend path.

Until v1.4.20 ``_maybe_cancel_on_binance_move`` called
``_enqueue_cancel_quote_path`` directly, bypassing the amend
decision tree in ``_orchestrate``. v1.4.19 deploy snapshot showed
this accounted for **84% of all venue actions** (2349 binance
cancels / 87s vs 28 amends from the regular reprice path).

v1.4.21 fix: when the Binance cross-venue trigger fires AND amend
is viable AND the last quote breakdown supplies a target price
within threshold of fair_value, route through ``_enqueue_amend_quote_path``
instead. Cancel falls through when:
  - amend not viable (knob off, no ordId, partial fill, cooldown, etc.)
  - no last quote breakdown available
  - target price would also trigger the binance threshold (stale
    breakdown — let the next quote loop tick refresh)

Counter ``session_cross_venue_amend_count`` tracks the amend wins.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import BestBidAsk, WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup(*, amend_enabled: bool = True) -> tuple[OrderManager, Path, Any]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_xvenue_{os.getpid()}_{uuid.uuid4().hex}.db"
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
            "BINANCE_WS_ENABLED": True,
            "BINANCE_CANCEL_ON_MOVE_BPS": 10.0,
            "BINANCE_WS_FAIR_VALUE_MAX_AGE_SECONDS": 5.0,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=1.946,
        best_ask=1.947,
        mid_price=1.9465,
        spread_bps=5.0,
        ts_local=utc_now(),
    )
    # Binance fair-value-side state (used by _maybe_cancel_on_binance_move).
    # v1.5.46: bid/ask required by the side-aware cancel gate. Symmetric
    # 5-bp spread around mid (≈ TON-like behaviour on a $1.95 price level
    # with a 0.001 tick: 0.001 / 1.9475 ≈ 5 bps).
    state.binance_mid = 1.9475
    state.binance_best_bid = 1.9470
    state.binance_best_ask = 1.9480
    state.binance_basis_ewma = 0.0
    state.binance_last_message_wall_ts = datetime.now(timezone.utc)
    # Use a TON-like SymbolSpec (tick=0.001) so the v1.4.27 no-op skip
    # (which compares target_px to wo.price against ``price_tick``)
    # uses a realistic tick size rather than the fallback 0.01 which
    # would mask everything at the test's 1.94 price level.
    from app.exchange.symbol_spec import SymbolSpec
    ton_spec = SymbolSpec(
        price_tick=0.001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=5.0,
        sz_decimals=0,
        source="hyperliquid_meta",
    )
    client = mock_mm_client(symbol_spec=ton_spec)
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    # Stop the dispatcher so submitted intents stay in the lane for
    # inspection.
    om._outbound.stop()
    return om, path, client


def _wo_acked(om: OrderManager, *, price: float, side: Side = Side.BUY) -> WorkingOrder:
    wo = WorkingOrder(
        order_id_local=f"local-{uuid.uuid4().hex[:8]}",
        order_id_exchange=12345,
        client_order_id="cl_" + uuid.uuid4().hex[:24],
        symbol=om._settings.symbol,
        side=side,
        price=price,
        size=3.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=datetime.now(timezone.utc),
        ts_sent=datetime.now(timezone.utc),
        ts_ack=datetime.now(timezone.utc),
    )
    with om._state._lock:
        om._state.set_working_order(side, 0, wo)
    return wo


def _set_breakdown(
    om: OrderManager,
    *,
    mid_price: float,
    quoted_bid_px: float,
    quoted_ask_px: float,
    target_half_spread_bps: float = 3.5,
) -> None:
    """Set a minimal last_quote_breakdown that the cross-venue handler
    will read.

    v1.4.27: the handler now uses ``target_half_spread_bps`` from the
    breakdown (a simple, bounded value — not the full bid/ask offset
    that can carry 30+ bps of inventory_skew when the bot is loaded).
    Tests supply this directly; ``mid_price`` / ``quoted_bid_px`` /
    ``quoted_ask_px`` are kept on the mock for backward-compat with
    other code paths that might read them.
    """
    bd = MagicMock()
    bd.mid_price = mid_price
    bd.quoted_bid_px = quoted_bid_px
    bd.quoted_ask_px = quoted_ask_px
    bd.target_half_spread_bps = target_half_spread_bps
    om._state.last_quote_breakdown = bd


def test_buy_far_from_fair_uses_amend_at_half_spread() -> None:
    """BUY at 1.941 vs fair 1.9475 — delta = 33 bps (over 10 bps
    threshold). Breakdown's target_half_spread_bps = 3.5. The
    v1.4.27 fix produces target_px = fair × (1 − 3.5/10000) =
    1.9475 × 0.99965 = 1.94682. Should AMEND, not cancel."""
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om, price=1.941, side=Side.BUY)
        _set_breakdown(
            om,
            mid_price=1.9475,
            quoted_bid_px=1.9465,
            quoted_ask_px=1.9485,
            target_half_spread_bps=3.5,
        )
        om._maybe_cancel_on_binance_move()
        assert wo.status == OrderStatus.AMEND_PENDING
        # fair_value = 1.9475 (binance_mid 1.9475 + basis 0.0 from _setup)
        # target_px = 1.9475 * (1 - 3.5/10000) ≈ 1.946818
        assert abs(wo.amend_target_px - 1.94681813) < 1e-6
        assert om._state.session_cross_venue_amend_count == 1
        assert om._state.session_cross_venue_cancel_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_sell_far_from_fair_uses_amend_at_half_spread() -> None:
    """Symmetric SELL case. target_px = fair × (1 + 3.5/10000)."""
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om, price=1.954, side=Side.SELL)
        _set_breakdown(
            om,
            mid_price=1.9475,
            quoted_bid_px=1.9465,
            quoted_ask_px=1.9485,
            target_half_spread_bps=3.5,
        )
        om._maybe_cancel_on_binance_move()
        assert wo.status == OrderStatus.AMEND_PENDING
        # target_px = 1.9475 * (1 + 3.5/10000) ≈ 1.948181
        assert abs(wo.amend_target_px - 1.94818163) < 1e-6
        assert om._state.session_cross_venue_amend_count == 1
    finally:
        path.unlink(missing_ok=True)


def test_okx_lag_still_amends_v1_4_23_regression() -> None:
    """Regression — when OKX mid lags Binance fair, the v1.4.27 fix
    anchors target_px to Binance fair (not the breakdown's lagged
    quoted_bid/ask). Target is always ~target_half_spread_bps from
    fair, regardless of how lagged the breakdown is."""
    om, path, _client = _setup(amend_enabled=True)
    om._state.binance_mid = 1.9295
    om._state.binance_basis_ewma = 0.0
    try:
        wo = _wo_acked(om, price=1.932, side=Side.BUY)
        _set_breakdown(
            om,
            mid_price=1.9315,  # OKX mid lagged 10 bps from Binance fair
            quoted_bid_px=1.9308,
            quoted_ask_px=1.9322,
            target_half_spread_bps=3.5,
        )
        om._maybe_cancel_on_binance_move()
        assert wo.status == OrderStatus.AMEND_PENDING
        # target_px = 1.9295 * (1 - 3.5/10000) ≈ 1.928824675
        # Pre-v1.4.27 (v1.4.23) this anchored to breakdown's lagged
        # bid_offset and produced 1.9288; post-v1.4.27 it anchors to
        # fair × (1 - half_spread) and produces 1.928824675 — same idea,
        # just no longer carries the inventory-skew baggage.
        assert abs(wo.amend_target_px - 1.928824675) < 1e-6
        assert om._state.session_cross_venue_amend_count == 1
        assert om._state.session_cross_venue_cancel_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_no_op_amend_skipped_when_target_within_tick() -> None:
    """v1.4.27 — when target_px is within ONE PRICE TICK of the
    current wo.price, the amend would be a wire-level no-op after
    grid rounding. Skip it (fall through to cancel) so the bot
    doesn't burn one venue call per Binance tick on a price change
    too small to land on the grid.

    Diagnosed in snapshot v1.4.26-260517-210319: 94 amends/sec, every
    amend's ``amend_target_px == wo.order.price`` exactly. With the
    no-op skip, the bot uses the cross-venue trigger as a SIGNAL but
    only fires an amend when there's actual grid-level movement.
    """
    om, path, _client = _setup(amend_enabled=True)
    om._state.binance_mid = 1.9295
    om._state.binance_basis_ewma = 0.0
    try:
        # Place the WO at a price within 1 tick of what target_px
        # would compute to. fair × (1 + 3.5/10000) ≈ 1.93018. WO at
        # 1.930 (one tick below the computed target). Difference is
        # 0.000175 — less than tick=0.001 — would round to same grid.
        wo = _wo_acked(om, price=1.930, side=Side.SELL)
        # wo_px=1.930 is far from fair=1.9295 → delta = 2.6 bps
        # which is UNDER the 10 bps threshold — trigger wouldn't fire.
        # Override fair to be further away to force a trigger.
        # Actually 2.6 bps is under threshold so trigger DOESN'T fire.
        # To test the no-op path we need delta > threshold + target
        # within 1 tick of wo. Move binance further away from wo.
        om._state.binance_mid = 1.928  # fair = 1.928, wo = 1.930 → delta ≈ 10.4 bps
        _set_breakdown(
            om,
            mid_price=1.928,
            quoted_bid_px=1.9270,
            quoted_ask_px=1.9290,
            target_half_spread_bps=3.5,
        )
        # target_px = 1.928 * (1 + 3.5/10000) ≈ 1.928675
        # wo_px = 1.930. Diff = 0.001325 > tick (0.001).
        # So the no-op skip would NOT fire here. Amend proceeds.
        # → Need a different setup: wo_px closer to target_px.
        # target_px ≈ 1.9287. Place wo at 1.9290 (within 1 tick).
        wo.price = 1.929
        wo_px = 1.929
        # wo_px=1.929 vs fair=1.928 → delta ≈ 5.2 bps. Still under.
        # Need bigger delta. Move binance further down.
        om._state.binance_mid = 1.926  # fair=1.926, wo=1.929 → delta ≈ 15.6 bps
        _set_breakdown(
            om,
            mid_price=1.926,
            quoted_bid_px=1.9250,
            quoted_ask_px=1.9270,
            target_half_spread_bps=3.5,
        )
        # target_px = 1.926 * (1 + 3.5/10000) ≈ 1.926674
        # wo_px = 1.929. Diff ≈ 0.002326 → bigger than 1 tick.
        # Still would amend. Need target_px close to wo_px.
        # Just hard-set the WO close to the computed target:
        wo.price = 1.927  # target ≈ 1.9267 → diff = 0.000326 < tick (0.001)
        wo_px = 1.927
        om._state.binance_mid = 1.920  # fair = 1.920, wo = 1.927 → delta ≈ 36 bps
        _set_breakdown(
            om,
            mid_price=1.920,
            quoted_bid_px=1.9190,
            quoted_ask_px=1.9210,
            target_half_spread_bps=3.5,
        )
        # target_px = 1.920 * (1 + 3.5/10000) ≈ 1.920672
        # wo_px = 1.927. Diff ≈ 0.006 — too big. amend would proceed.
        # OK final attempt: place wo at the EXACT target so no-op
        # skip definitely fires.
        wo.price = 1.9207
        om._state.binance_mid = 1.920
        _set_breakdown(
            om,
            mid_price=1.920,
            quoted_bid_px=1.9190,
            quoted_ask_px=1.9210,
            target_half_spread_bps=3.5,
        )
        # target_px ≈ 1.920672. wo_px = 1.9207. Diff ≈ 0.000028 < tick.
        # No-op skip should fire → cancel route taken.
        om._state.session_cross_venue_amend_count = 0
        om._state.session_cross_venue_cancel_count = 0
        # The WO needs to be ACKED for trigger to consider it.
        wo.status = OrderStatus.ACKED
        # We need delta > threshold to trigger at all. wo=1.9207
        # vs fair=1.920 → 3.6 bps, UNDER threshold. Force a bigger
        # delta by moving the WO further from fair.
        wo.price = 1.924  # delta ≈ 20.8 bps, over threshold
        # target_px ≈ 1.920672. wo=1.924. Diff = 0.003328 > tick.
        # So no-op skip won't fire here.
        #
        # The honest truth: in a real scenario, the bot AMENDS the WO
        # to target_px on tick N. Then on tick N+1, fair barely moves
        # so target_px is essentially the SAME (within sub-tick), and
        # wo_px (just amended) ≈ target_px. THAT's when no-op fires.
        # We simulate this by placing wo AT target_px from a prior tick.
        wo.price = 1.92067
        om._state.binance_mid = 1.918  # Slightly different fair to ensure delta > threshold
        # delta = (1.92067 - 1.918) / 1.918 * 10000 ≈ 13.9 bps > 10
        _set_breakdown(
            om,
            mid_price=1.918,
            quoted_bid_px=1.917,
            quoted_ask_px=1.919,
            target_half_spread_bps=3.5,
        )
        # target_px = 1.918 * (1 + 3.5/10000) ≈ 1.918671
        # wo_px = 1.92067. Diff = 0.001998 ≈ 2 ticks. Amend WOULD fire.
        #
        # OK simpler: keep fair very close to wo so target is within tick:
        wo.price = 1.918671  # exactly the target
        om._state.binance_mid = 1.918  # fair = 1.918
        # delta = (1.918671 - 1.918) / 1.918 * 10000 ≈ 3.5 bps under threshold
        # Need to bypass the threshold. Use a forced setup: just
        # mock the delta_bps check to be skipped. Or set threshold low.
        # Actually let's lower threshold for this test to force trigger:
        # NOTE: the threshold check uses `binance_cancel_on_move_bps`
        # which defaults to 10. We can't change it at runtime easily.
        #
        # Pragmatic: the no-op skip is also testable by verifying
        # the BEHAVIOR — count amend_intents_emitted before/after a
        # single trigger with target ≈ wo_px.
        #
        # v1.4.45 made the cross-venue threshold dynamic
        # (``max(static_floor, half_spread + buffer)``). With the
        # default buffer of 5.0 and the test's half_spread=3.5, the
        # dynamic threshold would be 8.5 bps — exceeding the 3.5 bps
        # delta this test stages and silently suppressing the trigger.
        # Disable the buffer to keep this test focused on the v1.4.27
        # no-op-skip behavior, which is what it's actually asserting.
        om._settings = om._settings.model_copy(
            update={
                "binance_cancel_on_move_bps": 1.0,
                "binance_cancel_on_move_buffer_bps": 0.0,
            }
        )
        # Now threshold=1bps. delta=3.5bps > 1 → trigger fires.
        # target_px ≈ 1.918671 ≈ wo_px → no-op skip → cancel route.
        om._maybe_cancel_on_binance_move()
        # The amend should NOT have fired (no-op).
        assert om._state.session_cross_venue_amend_count == 0, (
            "No-op amend (target_px within tick of wo_px) should "
            "have been skipped"
        )
        # Cancel route taken as fallback.
        assert om._state.session_cross_venue_cancel_count == 1
    finally:
        path.unlink(missing_ok=True)


def test_missing_target_half_spread_falls_back_to_cancel() -> None:
    """v1.4.27: when the breakdown lacks ``target_half_spread_bps``
    (e.g. older bot data, engine no-quote result), the handler can't
    compute a target. Fall back to cancel."""
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om, price=1.941, side=Side.BUY)
        # Breakdown with no half_spread.
        bd = MagicMock()
        bd.mid_price = 1.9475
        bd.quoted_bid_px = 1.9465
        bd.quoted_ask_px = 1.9485
        bd.target_half_spread_bps = None
        om._state.last_quote_breakdown = bd
        om._maybe_cancel_on_binance_move()
        # Falls back to cancel.
        assert wo.status == OrderStatus.CANCEL_PENDING
        assert om._state.session_cross_venue_cancel_count == 1
        assert om._state.session_cross_venue_amend_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_no_breakdown_falls_back_to_cancel() -> None:
    """First-cycle case — no quote breakdown yet. Cancel."""
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om, price=1.941, side=Side.BUY)
        # last_quote_breakdown stays None (default).
        om._maybe_cancel_on_binance_move()
        assert wo.status == OrderStatus.CANCEL_PENDING
        assert om._state.session_cross_venue_cancel_count == 1
        assert om._state.session_cross_venue_amend_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_knob_off_falls_back_to_cancel() -> None:
    """When OKX_AMEND_ON_REPRICE_ENABLED=false, amend is never
    viable — should always cancel even with a good target."""
    om, path, _client = _setup(amend_enabled=False)
    try:
        wo = _wo_acked(om, price=1.941, side=Side.BUY)
        _set_breakdown(
            om,
            mid_price=1.9475,
            quoted_bid_px=1.9465,
            quoted_ask_px=1.9485,
        )
        om._maybe_cancel_on_binance_move()
        assert wo.status == OrderStatus.CANCEL_PENDING
        assert om._state.session_cross_venue_cancel_count == 1
        assert om._state.session_cross_venue_amend_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_below_threshold_does_not_trigger_at_all() -> None:
    """When the WO is within threshold of fair_value, neither cancel
    nor amend should fire. The trigger only fires on the out-of-band
    case."""
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om, price=1.9475, side=Side.BUY)  # at fair
        _set_breakdown(
            om,
            mid_price=1.9475,
            quoted_bid_px=1.9465,
            quoted_ask_px=1.9485,
        )
        om._maybe_cancel_on_binance_move()
        # Neither path fired.
        assert wo.status == OrderStatus.ACKED
        assert om._state.session_cross_venue_cancel_count == 0
        assert om._state.session_cross_venue_amend_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_partial_status_falls_back_to_cancel() -> None:
    """``_amend_viable`` excludes PARTIAL (Phase 3 conservative
    behaviour — bot doesn't track filled qty). Cross-venue trigger
    on PARTIAL should cancel."""
    om, path, _client = _setup(amend_enabled=True)
    try:
        wo = _wo_acked(om, price=1.941, side=Side.BUY)
        wo.status = OrderStatus.PARTIAL
        _set_breakdown(
            om,
            mid_price=1.9475,
            quoted_bid_px=1.9465,
            quoted_ask_px=1.9485,
        )
        om._maybe_cancel_on_binance_move()
        assert wo.status == OrderStatus.CANCEL_PENDING
        assert om._state.session_cross_venue_amend_count == 0
        assert om._state.session_cross_venue_cancel_count == 1
    finally:
        path.unlink(missing_ok=True)


def test_counter_in_snapshot_dict() -> None:
    """The new ``session_cross_venue_amend_count`` must appear in
    state_current.json for dashboard consumption."""
    om, path, _client = _setup(amend_enabled=True)
    try:
        snap = om._state.snapshot_dict()
        assert "session_cross_venue_amend_count" in snap
        assert snap["session_cross_venue_amend_count"] == 0
        # Existing cancel counter still present (back-compat).
        assert "session_cross_venue_cancel_count" in snap
    finally:
        path.unlink(missing_ok=True)
