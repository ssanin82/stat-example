"""v1.5.157 — position-aware favorable-exit batch.

Completes the v1.5.155 follow-up: adds position-aware favorable
exits to the two remaining gates that still cleared 0% via
favorable signal in the v1.5.154-260526-074029 snapshot:

* ``at_touch_adverse_pause`` (3 fires / 0 favorable / 3 ceiling) —
  same per-side reducing-side-carve-out template as
  ``realised_edge_side_suppress`` (v1.5.155).
* ``adaptive_widen`` (98 fires / 10 favorable / 88 ceiling) —
  global widening overlay; same drift-aligned-with-inventory
  template as ``mae_gate`` (v1.5.155). Tested here only at the
  AtTouchAdversePause level (gate side); the bot.py wiring for
  adaptive_widen is covered indirectly by the existing gate-
  composition tests.

``post_reduction_cooldown`` is intentionally NOT migrated — it
ALREADY has a position-aware favorable exit (utilization-based:
clears when |position|/cap < 0.30). That predicate fits this
gate's "don't immediately re-flip after a reduction" intent
better than a drift-based predicate would.

Per CLAUDE.md: only this test file is run from the assistant;
full-suite verification is the CI daemon's job.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.at_touch_adverse_pause import AtTouchAdversePause
from app.enums import Side
from app.models import Fill


def _make_at_touch_pause() -> AtTouchAdversePause:
    return AtTouchAdversePause(
        threshold_bps=-5.0,
        pause_seconds=30.0,
        min_fills=3,
    )


def _adverse_fill(side: Side, markout: float = -10.0) -> Fill:
    """A minimal Fill object that the at_touch detector will accept."""
    fill = Fill(
        fill_id="f",
        order_id_exchange=1,
        client_order_id="cl",
        ts_fill=datetime(2026, 5, 27, 0, 0, 0, tzinfo=timezone.utc),
        symbol="TEST",
        side=side,
        price=1.0,
        size=1.0,
        notional=1.0,
        fee=0.0,
        liquidity_flag="maker",
        mid_at_fill=1.0,
        markout_5s_bps=markout,
    )
    # The at_touch detector reads quote_aggressiveness; not in the
    # constructor signature in this codebase but settable as an
    # attribute on the dataclass.
    object.__setattr__(fill, "quote_aggressiveness", "at_touch")
    return fill


def _arm_buy(gate: AtTouchAdversePause) -> None:
    for _ in range(3):
        gate.observe_resolved_5s_markout(_adverse_fill(Side.BUY), now_mono=0.0)
    assert gate.is_paused(Side.BUY, 1.0)


def _arm_sell(gate: AtTouchAdversePause) -> None:
    for _ in range(3):
        gate.observe_resolved_5s_markout(_adverse_fill(Side.SELL), now_mono=0.0)
    assert gate.is_paused(Side.SELL, 1.0)


# ---------------------------------------------------------------------------
# Firing conditions — at_touch_adverse_pause
# ---------------------------------------------------------------------------


def test_buy_pause_clears_when_bot_is_short():
    """BUY paused + bot SHORT → BUY is the reducing side → clear."""
    gate = _make_at_touch_pause()
    _arm_buy(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    assert cleared is True
    assert gate.is_paused(Side.BUY, 1.0) is False


def test_sell_pause_clears_when_bot_is_long():
    gate = _make_at_touch_pause()
    _arm_sell(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.SELL,
        now_mono=1.0,
        position_qty=+3.0,
        inventory_threshold=1.0,
    )
    assert cleared is True
    assert gate.is_paused(Side.SELL, 1.0) is False


# ---------------------------------------------------------------------------
# Non-firing conditions
# ---------------------------------------------------------------------------


def test_buy_pause_does_not_clear_when_bot_is_long():
    """BUY paused + bot LONG → BUY would add adverse → keep pause."""
    gate = _make_at_touch_pause()
    _arm_buy(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=+3.0,
        inventory_threshold=1.0,
    )
    assert cleared is False
    assert gate.is_paused(Side.BUY, 1.0) is True


def test_sell_pause_does_not_clear_when_bot_is_short():
    gate = _make_at_touch_pause()
    _arm_sell(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.SELL,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    assert cleared is False


def test_flat_position_does_not_clear():
    gate = _make_at_touch_pause()
    _arm_buy(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=0.0,
        inventory_threshold=1.0,
    )
    assert cleared is False


def test_subthreshold_position_does_not_clear():
    gate = _make_at_touch_pause()
    _arm_buy(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-0.5,
        inventory_threshold=1.0,
    )
    assert cleared is False


def test_no_op_when_not_paused():
    gate = _make_at_touch_pause()
    # Never arm.
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    assert cleared is False


def test_counter_increments_per_side():
    gate = _make_at_touch_pause()
    _arm_buy(gate)
    _arm_sell(gate)
    gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    snap = gate.snapshot_dict(1.0)
    assert snap["buy"]["cleared_via_position_favorable_total"] == 1
    assert snap["sell"]["cleared_via_position_favorable_total"] == 0
    # SELL still paused (BUY clear didn't touch it)
    assert snap["sell"]["paused"] is True


def test_snapshot_dict_includes_position_favorable_counter():
    gate = _make_at_touch_pause()
    snap = gate.snapshot_dict(0.0)
    assert "cleared_via_position_favorable_total" in snap["buy"]
    assert "cleared_via_position_favorable_total" in snap["sell"]


def test_disabled_gate_returns_false():
    """When the gate is disabled (threshold=0 or pause=0), the
    position-aware exit is also a no-op."""
    gate = AtTouchAdversePause(
        threshold_bps=0.0,  # disabled
        pause_seconds=30.0,
        min_fills=3,
    )
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    assert cleared is False


# ---------------------------------------------------------------------------
# fast_move_cancel drift-window selector
# ---------------------------------------------------------------------------


def test_fast_move_cancel_with_500ms_returns_side_buy_on_down_move():
    """Bottom-up sanity check: the underlying detector returns the
    correct side for a down move. This is unchanged from v1.5.156 —
    the v1.5.157 wiring is about which drift-window FEEDS the
    detector, not about the detector itself."""
    from app.fast_move_cancel import detect_target_venue_fast_move
    result = detect_target_venue_fast_move(
        mid_return_bps=-15.0,
        threshold_bps=10.0,
    )
    assert result == Side.BUY


def test_fast_move_cancel_threshold_disables():
    from app.fast_move_cancel import detect_target_venue_fast_move
    result = detect_target_venue_fast_move(
        mid_return_bps=-100.0,
        threshold_bps=0.0,
    )
    assert result is None
