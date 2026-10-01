"""BUG-027 — exclude the about-to-be-cancelled slot from the
worst-case post-fill check in ``_central_pre_send_risk_check``.

Background: the v1.5.148-260525-201857 snapshot showed 6,043 risk
refusals (``worst_post_fill_qty_over_max_abs_position``) producing
3 silent-wedge episodes. The Phase 2C multi-rung sum (v1.4.76)
counts every same-side resting working order's size against the
cap, including a slot whose order is about to be cancelled by the
SAME orchestrator action (the cancel-replace pattern).

v1.5.150 adds a ``replacing_slot`` kwarg to
``_central_pre_send_risk_check`` (and the stage-place wrapper)
so the call site can declare "this slot's resting WO is going
to be cancelled — don't count it against worst-case".

These tests pin:
* default ``replacing_slot=None`` preserves pre-v1.5.150 behaviour
  bit-identically (Phase 2C multi-rung sum unchanged)
* ``replacing_slot=(side, idx)`` excludes only the named slot's
  WO, not other slots / sides
* The exclusion is no-op when ``reduce_only=True`` (the position
  check is bypassed entirely on reduce-only orders)
* The exclusion makes the v1.5.148 reproducer pattern PASS
  (pos_qty=0 + resting slot to be cancelled + new same-size place)
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import PositionSnapshot, WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _make_om(**overrides):
    path = Path(tempfile.gettempdir()) / (
        f"mm_bug027_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "PRIVATE_WS_ENABLED": False,
        "MAX_ABS_POSITION": 5.0,
        "MAX_POSITION_NOTIONAL_USD": 10_000.0,
        "MAX_ORDER_NOTIONAL_USD": 50.0,
        "MAX_ORDER_NOTIONAL_HARD_MULTIPLIER": 2.0,
        "LADDER_NUM_LEVELS_PER_SIDE": 2,
    }
    base.update(overrides)
    s = UnitTestSettings.model_validate(base)
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=0.0,
        avg_entry_price=None,
        mark_price=10.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    return OrderManager(s, client, storage, state), state


def _set_wo(state: BotState, side: Side, level_idx: int, size: float) -> None:
    """Attach an ACKED working order at the given slot. Used to seed
    the resting-same-side condition the v1.5.148 wedge reproduced."""
    wo = WorkingOrder(
        order_id_local=f"x-{side.value.lower()}-{level_idx}",
        order_id_exchange=1000 + level_idx,
        client_order_id=None,
        symbol="ETH",
        side=side,
        price=10.0,
        size=size,
        post_only=True,
        status=OrderStatus.ACKED,
        level_idx=level_idx,
    )
    wos = state._working_orders  # noqa: SLF001
    wos.setdefault(side, {})[level_idx] = wo


# -------------------------------------------------------------------
# Default behaviour preserved (replacing_slot=None)
# -------------------------------------------------------------------


def test_default_replacing_slot_none_preserves_phase2c_sum() -> None:
    """Without ``replacing_slot``, the worst-case sum still includes
    every same-side ACKED WO. Pin the pre-v1.5.150 behaviour."""
    om, state = _make_om()
    # Rung-0 BUY ACKED at 3 contracts; want to place rung-1 BUY at 3.
    _set_wo(state, Side.BUY, 0, size=3.0)
    # cur_qty=0, resting=3, new=3 -> worst=6 > cap=5 -> REFUSE.
    assert om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="c"
    ) is False


def test_default_no_replacing_slot_same_as_no_kwarg() -> None:
    """Explicit ``replacing_slot=None`` is identical to omitting the
    kwarg. Pinning the default-arg semantics."""
    om, state = _make_om()
    _set_wo(state, Side.BUY, 0, size=3.0)
    a = om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="c"
    )
    b = om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="c",
        replacing_slot=None,
    )
    assert a == b == False  # noqa: E712 -- explicit equality desired


# -------------------------------------------------------------------
# Same-slot exclusion (the BUG-027 fix)
# -------------------------------------------------------------------


def test_replacing_slot_excludes_named_slots_resting() -> None:
    """``replacing_slot=(BUY, 0)`` removes the (BUY, 0) WO's size
    from the worst-case sum. Same scenario as the default-behaviour
    test above, but with the exclusion the check PASSES."""
    om, state = _make_om()
    _set_wo(state, Side.BUY, 0, size=3.0)
    # cur_qty=0, resting=3 (excluded), new=3 -> worst=3 <= cap=5 -> PASS.
    assert om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="c",
        replacing_slot=(Side.BUY, 0),
    ) is True


def test_replacing_slot_other_slot_not_excluded() -> None:
    """``replacing_slot=(BUY, 0)`` excludes only the (BUY, 0) slot.
    A (BUY, 1) ACKED order still contributes to worst-case. Pin
    the slot-targeting precision."""
    om, state = _make_om()
    _set_wo(state, Side.BUY, 0, size=3.0)
    _set_wo(state, Side.BUY, 1, size=3.0)
    # Replacing rung-0; rung-1 still counts.
    # cur_qty=0, resting=3 (rung-1 only), new=3 -> worst=6 > cap=5 -> REFUSE.
    assert om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="c",
        replacing_slot=(Side.BUY, 0),
    ) is False


def test_replacing_slot_other_side_not_excluded() -> None:
    """``replacing_slot`` matches BOTH side and level_idx. A
    (SELL, 0) WO is unaffected by ``replacing_slot=(BUY, 0)``
    because the loop only iterates the SAME side anyway, but pin
    the constraint explicitly."""
    om, state = _make_om()
    # SELL on the other side has no effect on BUY's pre-send check
    # because iter_working_orders(BUY) doesn't touch SELL anyway.
    _set_wo(state, Side.SELL, 0, size=3.0)
    # No BUY rest -> fresh BUY place at 3 contracts always passes
    # the multi-rung sum (cur=0, resting=0, new=3, worst=3 <= 5).
    assert om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="c",
        replacing_slot=(Side.BUY, 0),
    ) is True


def test_replacing_slot_with_nonzero_position() -> None:
    """The exclusion composes with non-zero current position. Bot
    short -3 (inherited), rung-0 BUY of 3 about to be cancelled;
    re-placing rung-0 at 3 contracts. cur=-3, resting=3 (excluded),
    new=3 -> worst=-3+3=0 -> within cap."""
    om, state = _make_om()
    state.position = PositionSnapshot(
        symbol="ETH",
        position_qty=-3.0,
        avg_entry_price=10.0,
        mark_price=10.0,
        position_notional=-300.0,
        unrealized_pnl_usd=0.0,
    )
    _set_wo(state, Side.BUY, 0, size=3.0)
    assert om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="c",
        replacing_slot=(Side.BUY, 0),
    ) is True


# -------------------------------------------------------------------
# Reduce-only compatibility
# -------------------------------------------------------------------


def test_reduce_only_bypass_makes_replacing_slot_a_noop() -> None:
    """Reduce-only orders skip the position check entirely, so
    ``replacing_slot`` has no effect on the outcome. Pin the
    interaction so a future refactor doesn't accidentally re-couple
    them."""
    om, state = _make_om(MAX_ABS_POSITION=0.5)
    state.position = PositionSnapshot(
        symbol="ETH",
        position_qty=0.4,
        avg_entry_price=10.0,
        mark_price=10.0,
        position_notional=40.0,
        unrealized_pnl_usd=0.0,
    )
    _set_wo(state, Side.SELL, 0, size=0.3)
    # Without reduce_only: cur=0.4, resting=0.3, new=1.0; SELL ->
    # worst = 0.4 - (1.0 + 0.3) = -0.9; |0.9| > 0.5 -> REFUSE.
    # With reduce_only: the position-check branch is skipped, only
    # the order-notional hard cap applies. $50 base x 2 hard = $100.
    # 1.0 x $50 = $50 -> PASS.
    assert om._central_pre_send_risk_check(
        Side.SELL, price=50.0, size=1.0, quote_cycle_id="c",
        reduce_only=True, replacing_slot=(Side.SELL, 0),
    ) is True
    # Same call without reduce_only refuses (position check fires).
    assert om._central_pre_send_risk_check(
        Side.SELL, price=50.0, size=1.0, quote_cycle_id="c",
        reduce_only=False,
    ) is False


# -------------------------------------------------------------------
# v1.5.148 reproducer regression test
# -------------------------------------------------------------------


def test_v1_5_148_reproducer_pattern_passes_with_replacing_slot() -> None:
    """Direct reproducer of the v1.5.148-260525-201857 wedge pattern.
    Without the fix: REFUSED (the observed bug). With
    ``replacing_slot=(BUY, 0)``: PASSES (the cancel-replace can
    proceed normally)."""
    om, state = _make_om(MAX_ABS_POSITION=5.0)
    state.position = PositionSnapshot(
        symbol="ETH",
        position_qty=0.0,
        avg_entry_price=None,
        mark_price=10.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    # Bot has ACKED BUY rung-0 at 3 contracts; engine wants to
    # update the price (cancel-replace flow with same size).
    _set_wo(state, Side.BUY, 0, size=3.0)
    # Pre-v1.5.150 behaviour: refused.
    assert om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="reproducer",
    ) is False
    # Post-v1.5.150 with the slot-being-replaced declared: passes.
    assert om._central_pre_send_risk_check(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="reproducer",
        replacing_slot=(Side.BUY, 0),
    ) is True


# -------------------------------------------------------------------
# Stage-place wrapper threads replacing_slot through
#
# NOTE: ``_stage_place_order_local`` has a SECOND guard
# (``existing_wo_active_race`` at lines ~7460-7486) that refuses the
# stage when the SPECIFIC slot being placed has an active WO at
# that level_idx. That guard fires AFTER the risk check and CANNOT
# be bypassed by ``replacing_slot`` -- the cancel-replace flow in
# the current reconciler enqueues the cancel and emits the
# PlaceAction on SEPARATE ticks. By the time the PlaceAction
# reaches stage_place, either:
#   * cur.status is terminal (CANCELED/FILLED/REJECTED/DESYNC):
#     no active WO -> guard passes -> stage proceeds.
#   * cur.status is CANCEL_PENDING/SENT/ACKED/PARTIAL: guard
#     refuses with ``existing_wo_active_race`` regardless of
#     ``replacing_slot`` -- this is a deliberate race guard from
#     v1.4.50 BUG-025 and unrelated to BUG-027.
#
# Therefore ``replacing_slot`` is **forward-looking correctness**
# for a future atomic-cancel-replace code path, not the actual
# unsticker for today's BUG-027 wedge. The wedge is unstuck by the
# accompanying config change ``MAX_ABS_POSITION 5 -> 6`` (see the
# BUG-027 issue doc + v1.5.150 release entry).
#
# The pure-helper tests above on ``_central_pre_send_risk_check``
# are sufficient coverage for the code change itself. A test for
# the stage-place wrapper would require the same-slot guard to NOT
# fire, which requires a code path that doesn't currently exist.
# -------------------------------------------------------------------


def test_stage_place_accepts_replacing_slot_kwarg() -> None:
    """Smoke test that ``replacing_slot`` is in the public signature
    of ``_stage_place_order_local`` and doesn't TypeError when
    passed. Behavioural assertion (it actually staging a place when
    set) requires a code path that bypasses ``existing_wo_active_race``
    -- not exercised today."""
    om, _state = _make_om()
    # Empty slot — neither guard fires; place should succeed
    # regardless of replacing_slot. The point of THIS test is just
    # that the kwarg is accepted without raising.
    wo = om._stage_place_order_local(
        Side.BUY, price=10.0, size=3.0, quote_cycle_id="c",
        replacing_slot=(Side.BUY, 0),
    )
    assert wo is not None
    assert wo.side == Side.BUY
    assert wo.size == 3.0
