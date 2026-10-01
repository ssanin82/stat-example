"""v1.4.26 — Fix the crossing-place hot loop.

Diagnosed in snapshot v1.4.25-260517-202432:
  * 1,688 places + 1,608 amends in 58 seconds (29 places/sec, 121/sec peak)
  * 1,687 of those orders lived <5ms each (silent venue cancel)
  * Place prices were AT the opposite touch (BUY at best_ask = 1.925,
    SELL at best_bid = 1.923)
  * OKX accepted the place, then silently cancelled at match time via
    user-data-WS (no sCode 51604 in the place response)
  * No `post_only_cross_cooldown` armed → bot retried immediately

Three coordinated fixes:

  Fix A+B (engine): final defensive cross-check in ``_build_side``. If
    rounded price would cross the touch, retreat by 1 tick. Catches
    everything the earlier ``min_half_spread`` retreat misses (which is
    everything, in normal_mm at-touch mode).

  Fix C (execution): WS user-data-WS CANCELED handler detects "ack→cancel
    within FAST_CANCEL_POST_ONLY_CROSS_THRESHOLD_MS" pattern and arms
    ``post_only_cross_cooldown`` for that side. Even when Fix A+B miss
    a case (e.g. book moves between engine emit and venue match), the
    cooldown prevents the next-tick retry.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import BestBidAsk, WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


# ----------------------------------------------------------------------
# Fix A+B — engine final cross-check
# ----------------------------------------------------------------------


def _build_quote_engine(monkeypatch=None):
    """Build a QuoteEngine with a configurable best_bid/best_ask
    market context. Returns (engine, build_quotes_fn) where
    build_quotes_fn is a thin wrapper accepting market overrides."""
    from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
    from app.quote_engine import QuoteEngine
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
        }
    )
    eng = QuoteEngine(s, FALLBACK_SYMBOL_SPEC)
    return eng


def test_build_side_buy_retreats_when_would_cross_best_ask() -> None:
    """Engine's final defensive check: if BUY price would cross
    best_ask, retreat by one tick. Without this fix, normal_mm
    at-touch + inventory_skew can push BUY to best_ask (= crossing)."""
    eng = _build_quote_engine()
    # Best bid 100, best ask 100.01, tick 0.01.
    # Candidate BUY at 100.01 (= best_ask) would cross. Expected
    # behaviour: retreat to 100.00.
    order, reason, npx, nsz = eng._build_side(
        side=Side.BUY,
        want_side=True,
        candidate_px=100.01,
        candidate_sz=0.5,
        best_bid=100.00,
        best_ask=100.01,
        tick=0.01,
        min_half_spread_px=None,
        mid_ref=100.005,
        max_allowed_size=None,
    )
    assert order is not None, f"reason={reason}"
    # Retreated to 100.00 (best_bid) — passive.
    assert order.price < 100.01, (
        f"BUY at {order.price} should have retreated below "
        f"best_ask=100.01 to avoid cross"
    )
    assert order.price <= 100.00 + 1e-9


def test_build_side_sell_advances_when_would_cross_best_bid() -> None:
    """Symmetric: SELL crossing best_bid should advance by one tick."""
    eng = _build_quote_engine()
    # SELL at 100.00 (= best_bid) would cross. Should advance to 100.01.
    order, reason, npx, nsz = eng._build_side(
        side=Side.SELL,
        want_side=True,
        candidate_px=100.00,
        candidate_sz=0.5,
        best_bid=100.00,
        best_ask=100.01,
        tick=0.01,
        min_half_spread_px=None,
        mid_ref=100.005,
        max_allowed_size=None,
    )
    assert order is not None, f"reason={reason}"
    assert order.price > 100.00, (
        f"SELL at {order.price} should have advanced above "
        f"best_bid=100.00 to avoid cross"
    )
    assert order.price >= 100.01 - 1e-9


def test_build_side_passive_buy_not_modified() -> None:
    """Regression: a BUY clearly below best_ask should NOT be touched
    by the cross-check."""
    eng = _build_quote_engine()
    order, reason, npx, nsz = eng._build_side(
        side=Side.BUY,
        want_side=True,
        candidate_px=99.50,  # well below best_ask
        candidate_sz=0.5,
        best_bid=100.00,
        best_ask=100.10,
        tick=0.01,
        min_half_spread_px=None,
        mid_ref=100.05,
        max_allowed_size=None,
    )
    assert order is not None, f"reason={reason}"
    # Price should be unchanged (modulo grid rounding to 99.50 exactly).
    assert abs(order.price - 99.50) < 1e-6


def test_build_side_refuses_when_no_passive_room() -> None:
    """When best_bid + tick == best_ask (1-tick-wide market), a BUY
    crossing best_ask cannot retreat to a passive price (retreated
    price would equal best_bid, still in-spread or equal). The
    engine should REFUSE rather than emit a price that still crosses."""
    eng = _build_quote_engine()
    # Best bid 100.00, best ask 100.01 (1-tick wide). Candidate at
    # 100.01 retreats to 100.00 = best_bid, which is passive (just
    # joining the bid). That's fine — passive_room available.
    # Now: best_bid 100.00, best_ask 100.00 (zero-spread, impossible
    # in practice but defensive). Retreat to 99.99 should succeed.
    order, reason, _, _ = eng._build_side(
        side=Side.BUY,
        want_side=True,
        candidate_px=100.00,
        candidate_sz=0.5,
        best_bid=100.00,
        best_ask=100.00,
        tick=0.01,
        min_half_spread_px=None,
        mid_ref=100.00,
        max_allowed_size=None,
    )
    # In a zero-spread market the engine should retreat below best_ask.
    # Either refuse OR retreat below — both are correct. Pin: result
    # must not be a crossing price.
    if order is not None:
        assert order.price < 100.00, f"BUY at {order.price} crosses"
    else:
        assert "cross" in (reason or "").lower() or "retreat" in (reason or "").lower()


# ----------------------------------------------------------------------
# Fix C — WS-detected silent venue cancel arms cooldown
# ----------------------------------------------------------------------


def _setup_ws() -> tuple[OrderManager, Path]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_xpc_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "POST_ONLY_CROSS_COOLDOWN_SECONDS": 2.5,
            "FAST_CANCEL_POST_ONLY_CROSS_THRESHOLD_MS": 100.0,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=1.924,
        best_ask=1.925,
        mid_price=1.9245,
        spread_bps=5.0,
        ts_local=utc_now(),
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om._outbound.stop()
    return om, path


def _wo_just_acked(
    om: OrderManager,
    side: Side,
    *,
    ack_age_ms: float = 5.0,
) -> WorkingOrder:
    """Build a WO that was acked ``ack_age_ms`` ago and is now about to
    be cancelled by a WS event. ``ts_cancel_requested`` is None — the
    bot never requested the cancel."""
    ts_ack = datetime.now(timezone.utc) - timedelta(milliseconds=ack_age_ms)
    wo = WorkingOrder(
        order_id_local=f"local-{uuid.uuid4().hex[:8]}",
        order_id_exchange=12345,
        client_order_id="cl_" + uuid.uuid4().hex[:24],
        symbol=om._settings.symbol,
        side=side,
        price=1.925 if side == Side.BUY else 1.923,
        size=3.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=datetime.now(timezone.utc) - timedelta(seconds=1),
        ts_sent=datetime.now(timezone.utc) - timedelta(seconds=1),
        ts_ack=ts_ack,
    )
    with om._state._lock:
        om._state.set_working_order(side, 0, wo)
    return wo


def _make_ws_canceled_event(wo: WorkingOrder):
    """Build a synthetic PrivateOrderUpdateEvent for a CANCELED state."""
    from app.exchange.private_events import PrivateOrderUpdateEvent
    return PrivateOrderUpdateEvent(
        oid=int(wo.order_id_exchange),
        coin=wo.symbol,
        status="canceled",
        status_timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
        side="B" if wo.side == Side.BUY else "A",
        limit_px=float(wo.price),
        remaining_sz=float(wo.size),
        orig_sz=float(wo.size),
        raw_status="canceled",
        cloid=wo.client_order_id,
    )


def test_ws_cancel_within_threshold_arms_cooldown() -> None:
    """Fix C — when a WS CANCELED arrives within
    FAST_CANCEL_POST_ONLY_CROSS_THRESHOLD_MS of ts_ack AND the bot
    never requested the cancel, infer post-only-cross-at-match-time
    and arm the cooldown."""
    om, path = _setup_ws()
    try:
        wo = _wo_just_acked(om, Side.BUY, ack_age_ms=10.0)
        ev = _make_ws_canceled_event(wo)
        # Pre-condition: cooldown not armed.
        assert not om._post_only_cross_cooldown_active(Side.BUY)
        # Apply the WS event (private dispatch path).
        om._handle_private_order_update(ev)
        # Cooldown should now be armed.
        assert om._post_only_cross_cooldown_active(Side.BUY)
        assert wo.status == OrderStatus.CANCELED
    finally:
        path.unlink(missing_ok=True)


def test_ws_cancel_beyond_threshold_does_not_arm_cooldown() -> None:
    """Slow cancel (>threshold ms after ts_ack) is treated as a
    normal cancel — no cooldown."""
    om, path = _setup_ws()
    try:
        wo = _wo_just_acked(om, Side.BUY, ack_age_ms=500.0)  # 500ms > 100ms threshold
        ev = _make_ws_canceled_event(wo)
        om._handle_private_order_update(ev)
        # Cooldown NOT armed — the cancel was too old to be venue-silent.
        assert not om._post_only_cross_cooldown_active(Side.BUY)
        assert wo.status == OrderStatus.CANCELED
    finally:
        path.unlink(missing_ok=True)


def test_ws_cancel_with_bot_initiated_cancel_does_not_arm() -> None:
    """When ts_cancel_requested is set, the bot DID request this
    cancel. Even if it lands within the threshold window, it's not
    a venue-silent cancel — don't arm the cooldown."""
    om, path = _setup_ws()
    try:
        wo = _wo_just_acked(om, Side.BUY, ack_age_ms=10.0)
        wo.ts_cancel_requested = datetime.now(timezone.utc)
        ev = _make_ws_canceled_event(wo)
        om._handle_private_order_update(ev)
        # Cooldown NOT armed — bot-initiated cancel, not venue silent.
        assert not om._post_only_cross_cooldown_active(Side.BUY)
    finally:
        path.unlink(missing_ok=True)


def test_fast_cancel_threshold_zero_disables_detector() -> None:
    """Operator can disable Fix C entirely by setting threshold to 0."""
    path = (
        Path(tempfile.gettempdir())
        / f"mm_xpc_dis_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "POST_ONLY_CROSS_COOLDOWN_SECONDS": 2.5,
            "FAST_CANCEL_POST_ONLY_CROSS_THRESHOLD_MS": 0.0,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=1.924,
        best_ask=1.925,
        mid_price=1.9245,
        spread_bps=5.0,
        ts_local=utc_now(),
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    om._outbound.stop()
    try:
        wo = _wo_just_acked(om, Side.BUY, ack_age_ms=10.0)
        ev = _make_ws_canceled_event(wo)
        om._handle_private_order_update(ev)
        # Detector disabled → no cooldown even with fast cancel.
        assert not om._post_only_cross_cooldown_active(Side.BUY)
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# v1.4.34 Codex #1 — hydrated orders MUST be exempt from the fast-cancel
# heuristic. Their ``ts_ack`` is venue-derived (or, in degraded paths,
# the bot's clock at hydrate time) and does NOT represent a fresh local
# place — a cancel within the fast-cancel window after process start is
# almost certainly an external trigger (Binance-cross-venue, operator,
# exchange sweep), not a post-only cross reject.
# ----------------------------------------------------------------------


def test_ws_cancel_within_threshold_skips_cooldown_for_hydrated_order() -> None:
    """Codex #1 regression: a hydrated order that gets cancelled
    within the fast-cancel threshold of its synthetic ``ts_ack`` must
    NOT arm the post-only-cross cooldown.

    Pre-v1.4.34: every restart-with-open-orders would falsely arm the
    cooldown on whichever side a cancel landed for, suppressing
    quoting for the cooldown window. Fixed by gating the heuristic
    on ``not wo.hydrated_from_exchange``.
    """
    om, path = _setup_ws()
    try:
        wo = _wo_just_acked(om, Side.BUY, ack_age_ms=10.0)
        # Mark the WO as hydrated (the only behavioural difference vs
        # the existing ``test_ws_cancel_within_threshold_arms_cooldown``).
        wo.hydrated_from_exchange = True
        ev = _make_ws_canceled_event(wo)
        assert not om._post_only_cross_cooldown_active(Side.BUY)
        om._handle_private_order_update(ev)
        assert wo.status == OrderStatus.CANCELED
        # The hydrated flag MUST exempt this WO from the heuristic
        # even though every other condition (ack_age < threshold,
        # bot didn't request cancel) is met.
        assert not om._post_only_cross_cooldown_active(Side.BUY), (
            "Hydrated order cancelled within fast-cancel window "
            "must NOT arm post_only_cross_cooldown (Codex #1)."
        )
    finally:
        path.unlink(missing_ok=True)


def test_ws_cancel_arms_cooldown_for_non_hydrated_order() -> None:
    """Codex #1 regression — negative control: a non-hydrated WO
    (the default ``hydrated_from_exchange=False``) must still arm
    the cooldown. This pins that the v1.4.34 fix is a NARROW carve-
    out and doesn't disable the legitimate detector for orders the
    bot placed in this process lifetime."""
    om, path = _setup_ws()
    try:
        wo = _wo_just_acked(om, Side.BUY, ack_age_ms=10.0)
        # Default hydrated_from_exchange is False — explicit for clarity.
        assert wo.hydrated_from_exchange is False
        ev = _make_ws_canceled_event(wo)
        om._handle_private_order_update(ev)
        assert om._post_only_cross_cooldown_active(Side.BUY), (
            "Non-hydrated WO cancelled within fast-cancel window "
            "MUST still arm the cooldown (legitimate detector path)."
        )
    finally:
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# v1.4.34 Codex #2 — after the cross-retreat, the resulting price must
# still respect ``min_half_spread_px``. The previous code only checked
# "still crosses" but missed the case where retreat lands inside the
# economic spread floor.
# ----------------------------------------------------------------------


def test_violates_min_half_spread_floor_buy_above_floor_is_violation() -> None:
    """Codex #2 helper: BUY above ``mid - min_half_spread_px`` is a
    floor violation (too aggressive on the BUY side).

    mid=100.00, mhs=0.05  →  BUY floor = 99.95 (must be ≤ 99.95).
    A BUY at 100.00 is 0.05 above the floor → violation.
    """
    from app.quote_engine import _violates_min_half_spread_floor

    assert _violates_min_half_spread_floor(
        Side.BUY, 100.00, mid_ref=100.00, min_half_spread_px=0.05
    ) is True


def test_violates_min_half_spread_floor_buy_at_or_below_floor_passes() -> None:
    """Codex #2 helper: BUY at the floor or below is NOT a violation.

    mid=100.00, mhs=0.05  →  BUY floor = 99.95.
    Exactly at floor (99.95) → not a violation.
    Below floor (99.90)    → not a violation.
    """
    from app.quote_engine import _violates_min_half_spread_floor

    assert _violates_min_half_spread_floor(
        Side.BUY, 99.95, mid_ref=100.00, min_half_spread_px=0.05
    ) is False
    assert _violates_min_half_spread_floor(
        Side.BUY, 99.90, mid_ref=100.00, min_half_spread_px=0.05
    ) is False


def test_violates_min_half_spread_floor_sell_below_floor_is_violation() -> None:
    """Codex #2 helper: SELL below ``mid + min_half_spread_px`` is a
    floor violation (too aggressive on the SELL side).

    mid=100.00, mhs=0.05  →  SELL floor = 100.05 (must be ≥ 100.05).
    A SELL at 100.00 is 0.05 below the floor → violation.
    """
    from app.quote_engine import _violates_min_half_spread_floor

    assert _violates_min_half_spread_floor(
        Side.SELL, 100.00, mid_ref=100.00, min_half_spread_px=0.05
    ) is True


def test_violates_min_half_spread_floor_sell_at_or_above_floor_passes() -> None:
    """Codex #2 helper: SELL at the floor or above is NOT a violation."""
    from app.quote_engine import _violates_min_half_spread_floor

    assert _violates_min_half_spread_floor(
        Side.SELL, 100.05, mid_ref=100.00, min_half_spread_px=0.05
    ) is False
    assert _violates_min_half_spread_floor(
        Side.SELL, 100.10, mid_ref=100.00, min_half_spread_px=0.05
    ) is False


def test_violates_min_half_spread_floor_disabled_when_inputs_missing() -> None:
    """Codex #2 helper: when the floor isn't configured (any of
    ``mid_ref`` / ``min_half_spread_px`` is None / 0 / negative /
    NaN), the helper returns False — the floor only exists when
    both are present and positive. This is the safety guard that
    keeps the cross-retreat path from refusing quotes the operator
    hasn't asked to gate.
    """
    from app.quote_engine import _violates_min_half_spread_floor

    # mid_ref missing
    assert _violates_min_half_spread_floor(
        Side.BUY, 100.00, mid_ref=None, min_half_spread_px=0.05
    ) is False
    # min_half_spread_px missing
    assert _violates_min_half_spread_floor(
        Side.BUY, 100.00, mid_ref=100.00, min_half_spread_px=None
    ) is False
    # zero min_half_spread_px
    assert _violates_min_half_spread_floor(
        Side.BUY, 100.00, mid_ref=100.00, min_half_spread_px=0.0
    ) is False
    # negative min_half_spread_px
    assert _violates_min_half_spread_floor(
        Side.BUY, 100.00, mid_ref=100.00, min_half_spread_px=-0.01
    ) is False
    # zero mid_ref
    assert _violates_min_half_spread_floor(
        Side.BUY, 100.00, mid_ref=0.0, min_half_spread_px=0.05
    ) is False


def test_build_side_buy_cross_retreat_passes_when_no_floor_configured() -> None:
    """Codex #2 negative control: when ``min_half_spread_px=None``
    (no economic floor configured), the cross-retreat path emits
    normally with no extra floor check. Pins that v1.4.34's check
    is gated on ``min_half_spread_px > 0`` and doesn't accidentally
    refuse quotes when the floor is unset.

    Note: this is the same shape as
    ``test_build_side_buy_retreats_when_would_cross_best_ask``; we
    re-assert here as a positive control for the v1.4.34 floor-
    check branch (to confirm it doesn't fire when the floor is
    None).
    """
    eng = _build_quote_engine()
    order, reason, _, _ = eng._build_side(
        side=Side.BUY,
        want_side=True,
        candidate_px=100.01,  # = best_ask, crosses
        candidate_sz=0.5,
        best_bid=100.00,
        best_ask=100.01,
        tick=0.01,
        min_half_spread_px=None,  # no floor configured
        mid_ref=100.005,
        max_allowed_size=None,
    )
    assert order is not None, (
        f"Expected emit (no floor configured), got refusal: {reason}"
    )
    assert order.price < 100.01, "should have retreated below best_ask"
