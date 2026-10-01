"""v1.5.155 — position-aware favorable-exit for
``realised_edge_side_suppress``.

Per CLAUDE.md Rule 0c: timer-based gates must have a signal-driven
conditional exit that considers current bot position. The existing
markout-based favorable exit re-evaluates only on new fills — but
on the suppressed side there ARE no fills, so it never re-evaluates
to favorable. v1.5.154-260526-074029 showed 30 fires / 0 favorable /
29 ceiling.

The position-aware exit clears a per-side suppression when that
side is the REDUCING side for current adverse inventory. Operator's
"reducing side must always be available" principle.

Per CLAUDE.md: only this test file is run from the assistant; full-
suite verification is the CI daemon's job.
"""

from __future__ import annotations

from app.enums import Side
from app.realised_edge_side_suppress import RealisedEdgeSideSuppressGate


def _make_gate() -> RealisedEdgeSideSuppressGate:
    return RealisedEdgeSideSuppressGate(
        threshold_bps=-4.0,
        cooldown_seconds=60.0,
        min_fills=4,
        window_size=20,
    )


def _arm_buy(gate: RealisedEdgeSideSuppress) -> None:
    """Drive BUY into suppression via 4 adverse fills."""
    for _ in range(4):
        gate.note_fill(
            side=Side.BUY,
            markout_5s_bps=-10.0,
            rebate_bps=0.5,
            now_mono=0.0,
        )
    assert gate.is_suppressed(Side.BUY, 1.0)


def _arm_sell(gate: RealisedEdgeSideSuppress) -> None:
    for _ in range(4):
        gate.note_fill(
            side=Side.SELL,
            markout_5s_bps=-10.0,
            rebate_bps=0.5,
            now_mono=0.0,
        )
    assert gate.is_suppressed(Side.SELL, 1.0)


# ---------------------------------------------------------------------------
# Firing conditions
# ---------------------------------------------------------------------------


def test_buy_suppression_clears_when_bot_is_short():
    """BUY is suppressed but bot is SHORT → BUY is the reducing side
    → clear the suppression."""
    gate = _make_gate()
    _arm_buy(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    assert cleared is True
    assert gate.is_suppressed(Side.BUY, 1.0) is False


def test_sell_suppression_clears_when_bot_is_long():
    """SELL is suppressed but bot is LONG → SELL is the reducing side
    → clear the suppression."""
    gate = _make_gate()
    _arm_sell(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.SELL,
        now_mono=1.0,
        position_qty=+3.0,
        inventory_threshold=1.0,
    )
    assert cleared is True
    assert gate.is_suppressed(Side.SELL, 1.0) is False


# ---------------------------------------------------------------------------
# Non-firing conditions
# ---------------------------------------------------------------------------


def test_buy_suppression_does_not_clear_when_bot_is_long():
    """BUY suppressed + bot LONG → BUY would ADD adverse → keep
    suppression (don't add to long when buy edges bleeding)."""
    gate = _make_gate()
    _arm_buy(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=+3.0,
        inventory_threshold=1.0,
    )
    assert cleared is False
    assert gate.is_suppressed(Side.BUY, 1.0) is True


def test_sell_suppression_does_not_clear_when_bot_is_short():
    """SELL suppressed + bot SHORT → SELL would ADD adverse → keep
    suppression."""
    gate = _make_gate()
    _arm_sell(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.SELL,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    assert cleared is False


def test_flat_position_does_not_clear():
    """Bot flat → no adverse inventory to unwind → no need for
    position-aware exit."""
    gate = _make_gate()
    _arm_buy(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=0.0,
        inventory_threshold=1.0,
    )
    assert cleared is False


def test_subthreshold_inventory_does_not_clear():
    """Small residual position (0.5 contract) — too small to warrant
    clearing the gate."""
    gate = _make_gate()
    _arm_buy(gate)
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-0.5,
        inventory_threshold=1.0,
    )
    assert cleared is False


def test_no_op_when_not_currently_suppressed():
    """Suppression already expired → no-op, no counter increment."""
    gate = _make_gate()
    # Never arm.
    cleared = gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    assert cleared is False


def test_counter_increments():
    """Counter increments on each successful clear."""
    gate = _make_gate()
    _arm_buy(gate)
    gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    snap = gate.snapshot_dict(1.0)
    assert snap["buy"]["cleared_via_position_favorable_total"] == 1
    assert snap["sell"]["cleared_via_position_favorable_total"] == 0


# ---------------------------------------------------------------------------
# Both sides independence
# ---------------------------------------------------------------------------


def test_clearing_buy_does_not_affect_sell_suppression():
    """The two sides' suppressions are independent."""
    gate = _make_gate()
    _arm_buy(gate)
    _arm_sell(gate)
    gate.try_clear_via_position_favorable(
        side=Side.BUY,
        now_mono=1.0,
        position_qty=-3.0,
        inventory_threshold=1.0,
    )
    # BUY cleared, SELL still suppressed.
    assert gate.is_suppressed(Side.BUY, 1.0) is False
    assert gate.is_suppressed(Side.SELL, 1.0) is True


# ---------------------------------------------------------------------------
# Snapshot contract
# ---------------------------------------------------------------------------


def test_snapshot_dict_includes_position_favorable_counter():
    """New v1.5.155 counter surfaces in the snapshot for dashboard
    visibility."""
    gate = _make_gate()
    snap = gate.snapshot_dict(0.0)
    assert "cleared_via_position_favorable_total" in snap["buy"]
    assert "cleared_via_position_favorable_total" in snap["sell"]
    assert snap["buy"]["cleared_via_position_favorable_total"] == 0
    assert snap["sell"]["cleared_via_position_favorable_total"] == 0
