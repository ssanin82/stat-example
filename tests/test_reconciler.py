"""v1.4.60 wedge-elimination Phase 5 — reconciler tests.

Three layers covered:

  1. ``DesiredOrderState`` + ``Action`` types — construction and
     ``empty()`` factory.
  2. Gate functions — pure ``DesiredLadderState → DesiredLadderState``
     folds, especially ``apply_reducing_side_bypass``.
  3. ``reconcile(desired, current)`` — pure diff producing the
     correct ``Action`` for every (desired-present?, current-status)
     combination.

The reconciler is a pure function. Tests set up minimal fixtures
(stub working orders via SimpleNamespace) — no state, no storage,
no clock. If a test needs threading or a database, it's testing
the wrong layer.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.enums import OrderStatus, Side
from app.reconciler import (
    AmendAction,
    CancelAction,
    DesiredOrderState,
    NoOpAction,
    PlaceAction,
    apply_reducing_side_bypass,
    apply_side_suppression,
    reconcile,
)


def _wo(
    side: Side,
    status: OrderStatus,
    price: float = 1.0,
    size: float = 1.0,
    local_id: str = "abc",
) -> SimpleNamespace:
    return SimpleNamespace(
        order_id_local=local_id,
        status=status,
        side=side,
        price=float(price),
        size=float(size),
    )


# ---------------------------------------------------------------------------
# Desired-state types
# ---------------------------------------------------------------------------


def test_desired_order_state_empty_factory() -> None:
    """``DesiredOrderState.empty(...)`` produces a should_exist=False
    state with reason captured."""
    d = DesiredOrderState.empty(Side.BUY, 0, reason="engine_no_quote")
    assert d.side == Side.BUY
    assert d.level_idx == 0
    assert d.should_exist is False
    assert d.reason == "engine_no_quote"
    assert d.price == 0.0
    assert d.size == 0.0


def test_desired_order_state_frozen() -> None:
    """``DesiredOrderState`` is frozen — accidental mutation raises."""
    import dataclasses
    d = DesiredOrderState(
        side=Side.BUY, level_idx=0, should_exist=True, price=1.0, size=5.0
    )
    try:
        d.price = 2.0  # type: ignore[misc]
    except (dataclasses.FrozenInstanceError, AttributeError):
        return
    raise AssertionError("DesiredOrderState should be frozen")


# ---------------------------------------------------------------------------
# Reducing-side bypass — THE STRUCTURAL INVARIANT
# ---------------------------------------------------------------------------


def test_reducing_side_bypass_restores_sell_when_long() -> None:
    """v1.4.60: when bot is LONG and a gate suppressed the SELL
    side (the reducer), the bypass restores it from the original
    engine output. This is what makes the v1.4.59 wedge class
    impossible: SELL can never be fully suppressed by defensive
    gates while a long position exists.
    """
    engine_desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.988, size=3.0, reason="engine_quote",
        ),
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.996, size=3.0, reason="engine_quote",
        ),
    }
    # A gate suppressed the SELL side.
    after_gates = {
        (Side.BUY, 0): engine_desired[(Side.BUY, 0)],
        (Side.SELL, 0): DesiredOrderState.empty(
            Side.SELL, 0, reason="post_only_cross_cooldown"
        ),
    }
    # Position +1 long → SELL is reducing.
    bypassed = apply_reducing_side_bypass(
        after_gates, position_qty=1.0, engine_desired=engine_desired
    )
    sell = bypassed[(Side.SELL, 0)]
    assert sell.should_exist is True, (
        "Reducing-side bypass MUST restore the SELL when position > 0 "
        "and a gate had suppressed it. This is the v1.4.59 wedge fix."
    )
    assert sell.price == 1.996
    assert sell.size == 3.0
    assert "reducing_side_bypass_overrode" in sell.reason
    # The BUY (adding side) stays untouched.
    assert bypassed[(Side.BUY, 0)] == engine_desired[(Side.BUY, 0)]


def test_reducing_side_bypass_restores_buy_when_short() -> None:
    """v1.4.60: short position → BUY is reducer. Same bypass logic
    in the opposite direction."""
    engine_desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.988, size=3.0, reason="engine_quote",
        ),
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.996, size=3.0, reason="engine_quote",
        ),
    }
    after_gates = {
        (Side.BUY, 0): DesiredOrderState.empty(
            Side.BUY, 0, reason="adverse_side_pause"
        ),
        (Side.SELL, 0): engine_desired[(Side.SELL, 0)],
    }
    bypassed = apply_reducing_side_bypass(
        after_gates, position_qty=-1.0, engine_desired=engine_desired
    )
    buy = bypassed[(Side.BUY, 0)]
    assert buy.should_exist is True
    assert buy.price == 1.988


def test_reducing_side_bypass_no_op_when_flat() -> None:
    """Position == 0: no "reducing side" — bypass is a no-op even
    if a gate suppressed something. With flat inventory, defensive
    gates are correctly applied to both sides; neither is "the
    reducer we must protect"."""
    engine_desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.988, size=3.0, reason="engine_quote",
        ),
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.996, size=3.0, reason="engine_quote",
        ),
    }
    after_gates = {
        (Side.BUY, 0): engine_desired[(Side.BUY, 0)],
        (Side.SELL, 0): DesiredOrderState.empty(
            Side.SELL, 0, reason="some_gate"
        ),
    }
    bypassed = apply_reducing_side_bypass(
        after_gates, position_qty=0.0, engine_desired=engine_desired
    )
    # Flat → bypass unchanged.
    assert bypassed[(Side.SELL, 0)].should_exist is False


def test_reducing_side_bypass_does_not_protect_adding_side() -> None:
    """Position +1 long, SELL is reducer. If a gate suppresses
    the BUY (adding side), bypass should NOT restore it — defending
    against growing inventory is legitimate."""
    engine_desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.988, size=3.0, reason="engine_quote",
        ),
    }
    after_gates = {
        (Side.BUY, 0): DesiredOrderState.empty(
            Side.BUY, 0, reason="adverse_side_pause"
        ),
    }
    bypassed = apply_reducing_side_bypass(
        after_gates, position_qty=1.0, engine_desired=engine_desired
    )
    # BUY is adding side when long → stays suppressed.
    assert bypassed[(Side.BUY, 0)].should_exist is False


def test_reducing_side_bypass_does_not_invent_quotes() -> None:
    """If the ENGINE didn't produce a quote for the reducing side,
    the bypass must not invent one. It only protects engine output
    from downstream suppression."""
    engine_desired = {
        (Side.SELL, 0): DesiredOrderState.empty(
            Side.SELL, 0, reason="engine_no_quote"
        ),
    }
    after_gates = dict(engine_desired)
    bypassed = apply_reducing_side_bypass(
        after_gates, position_qty=1.0, engine_desired=engine_desired
    )
    # No quote to restore.
    assert bypassed[(Side.SELL, 0)].should_exist is False


def test_reducing_side_bypass_applies_to_all_levels() -> None:
    """Multi-rung ladder: the bypass applies to every reducer-side
    level, not just level 0."""
    engine_desired = {
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.996, size=3.0, reason="engine_quote",
        ),
        (Side.SELL, 1): DesiredOrderState(
            side=Side.SELL, level_idx=1, should_exist=True,
            price=1.998, size=2.0, reason="engine_quote",
        ),
    }
    after_gates = {
        (Side.SELL, 0): DesiredOrderState.empty(
            Side.SELL, 0, reason="some_gate"
        ),
        (Side.SELL, 1): DesiredOrderState.empty(
            Side.SELL, 1, reason="some_gate"
        ),
    }
    bypassed = apply_reducing_side_bypass(
        after_gates, position_qty=1.0, engine_desired=engine_desired
    )
    assert bypassed[(Side.SELL, 0)].should_exist is True
    assert bypassed[(Side.SELL, 1)].should_exist is True


def test_reducing_side_bypass_disabled_via_flag() -> None:
    """Operators can disable the bypass for rollback / testing via
    ``enabled=False``. With disabled, the gate suppression sticks
    even on the reducer."""
    engine_desired = {
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.996, size=3.0, reason="engine_quote",
        ),
    }
    after_gates = {
        (Side.SELL, 0): DesiredOrderState.empty(
            Side.SELL, 0, reason="some_gate"
        ),
    }
    bypassed = apply_reducing_side_bypass(
        after_gates,
        position_qty=1.0,
        engine_desired=engine_desired,
        enabled=False,
    )
    assert bypassed[(Side.SELL, 0)].should_exist is False


# ---------------------------------------------------------------------------
# Side-suppression gate
# ---------------------------------------------------------------------------


def test_apply_side_suppression_marks_all_levels_empty() -> None:
    base = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True, price=1.0, size=3.0,
            reason="engine_quote",
        ),
        (Side.BUY, 1): DesiredOrderState(
            side=Side.BUY, level_idx=1, should_exist=True, price=0.99, size=2.0,
            reason="engine_quote",
        ),
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True, price=1.01, size=3.0,
            reason="engine_quote",
        ),
    }
    out = apply_side_suppression(base, side=Side.BUY, reason="test_gate")
    assert out[(Side.BUY, 0)].should_exist is False
    assert out[(Side.BUY, 0)].reason == "gate:test_gate"
    assert out[(Side.BUY, 1)].should_exist is False
    # SELL untouched.
    assert out[(Side.SELL, 0)].should_exist is True


def test_apply_side_suppression_respects_levels_filter() -> None:
    """``levels`` filter — only those level indices get suppressed."""
    base = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True, price=1.0, size=3.0,
            reason="engine_quote",
        ),
        (Side.BUY, 1): DesiredOrderState(
            side=Side.BUY, level_idx=1, should_exist=True, price=0.99, size=2.0,
            reason="engine_quote",
        ),
    }
    out = apply_side_suppression(base, side=Side.BUY, reason="test", levels=[0])
    assert out[(Side.BUY, 0)].should_exist is False
    # Level 1 not in filter → unchanged.
    assert out[(Side.BUY, 1)].should_exist is True


# ---------------------------------------------------------------------------
# Reconciler — pure diff
# ---------------------------------------------------------------------------


def test_reconcile_empty_desired_and_current() -> None:
    """Both sides empty → NoOp."""
    actions = reconcile(desired={}, current={})
    assert actions == []


def test_reconcile_desired_should_exist_no_current_emits_place() -> None:
    """Desired=ACTIVE, current=None → PlaceAction."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
    }
    actions = reconcile(desired=desired, current={})
    assert len(actions) == 1
    a = actions[0]
    assert isinstance(a, PlaceAction)
    assert a.side == Side.BUY
    assert a.level_idx == 0
    assert a.desired.price == 1.0
    assert a.desired.size == 3.0


def test_reconcile_desired_should_not_exist_active_current_emits_cancel() -> None:
    """Desired=EMPTY, current=ACKED → CancelAction."""
    desired = {
        (Side.BUY, 0): DesiredOrderState.empty(Side.BUY, 0, reason="gate_suppressed"),
    }
    current = {
        (Side.BUY, 0): _wo(Side.BUY, OrderStatus.ACKED, price=1.0, local_id="local-1"),
    }
    actions = reconcile(desired=desired, current=current)
    assert len(actions) == 1
    a = actions[0]
    assert isinstance(a, CancelAction)
    assert a.current_local_id == "local-1"


def test_reconcile_in_flight_current_emits_noop_wait() -> None:
    """Current in SENT/CANCEL_PENDING/AMEND_PENDING → wait."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
    }
    for status in (OrderStatus.SENT, OrderStatus.CANCEL_PENDING, OrderStatus.AMEND_PENDING):
        current = {(Side.BUY, 0): _wo(Side.BUY, status, local_id="local-1")}
        actions = reconcile(desired=desired, current=current)
        assert len(actions) == 1
        a = actions[0]
        assert isinstance(a, NoOpAction), (
            f"In-flight status {status} must produce NoOpAction, got {type(a)}"
        )
        assert "in_flight_wait" in a.reason


def test_reconcile_terminal_current_treats_as_absent() -> None:
    """Current in CANCELED/FILLED/REJECTED is treated as absent.
    Desired=ACTIVE → PlaceAction (fresh)."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
    }
    for status in (OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED):
        current = {(Side.BUY, 0): _wo(Side.BUY, status, local_id="local-1")}
        actions = reconcile(desired=desired, current=current)
        assert len(actions) == 1
        assert isinstance(actions[0], PlaceAction), (
            f"Terminal status {status} should be treated as absent → "
            f"PlaceAction. Got: {type(actions[0])}"
        )


def test_reconcile_acked_matches_desired_emits_noop_no_materiality_check() -> None:
    """Both want and have an order. Materiality check returns False
    (no reprice needed) → NoOp."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
    }
    current = {
        (Side.BUY, 0): _wo(Side.BUY, OrderStatus.ACKED, price=1.0, size=3.0, local_id="L1"),
    }
    actions = reconcile(
        desired=desired,
        current=current,
        materiality_check=lambda c, d: False,  # no reprice
    )
    assert len(actions) == 1
    a = actions[0]
    assert isinstance(a, NoOpAction)
    assert a.reason == "acked_no_reprice_needed"


def test_reconcile_acked_needs_reprice_amend_viable_emits_amend() -> None:
    """Reprice needed AND amend viable → AmendAction."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="engine_quote",
        ),
    }
    current = {
        (Side.BUY, 0): _wo(Side.BUY, OrderStatus.ACKED, price=1.0, size=3.0, local_id="L1"),
    }
    actions = reconcile(
        desired=desired,
        current=current,
        materiality_check=lambda c, d: True,
        amend_viable_check=lambda c, d: True,
    )
    assert len(actions) == 1
    a = actions[0]
    assert isinstance(a, AmendAction)
    assert a.current_local_id == "L1"
    assert a.desired.price == 1.01


def test_reconcile_acked_needs_reprice_amend_not_viable_emits_cancel() -> None:
    """Reprice needed AND amend NOT viable → CancelAction
    (cancel-then-place fallback handled at dispatcher)."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="engine_quote",
        ),
    }
    current = {
        (Side.BUY, 0): _wo(Side.BUY, OrderStatus.ACKED, price=1.0, size=3.0, local_id="L1"),
    }
    actions = reconcile(
        desired=desired,
        current=current,
        materiality_check=lambda c, d: True,
        amend_viable_check=lambda c, d: False,
    )
    assert len(actions) == 1
    a = actions[0]
    assert isinstance(a, CancelAction)
    assert a.trigger_reason == "reprice_replace"


def test_reconcile_partial_treated_same_as_acked() -> None:
    """PARTIAL fills are still active orders; same handling as ACKED."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
    }
    current = {
        (Side.BUY, 0): _wo(Side.BUY, OrderStatus.PARTIAL, price=1.0, size=3.0, local_id="L1"),
    }
    actions = reconcile(
        desired=desired,
        current=current,
        materiality_check=lambda c, d: False,
    )
    assert len(actions) == 1
    assert isinstance(actions[0], NoOpAction)


def test_reconcile_handles_multi_rung_ladder() -> None:
    """Multi-rung: 2 sides × 2 levels = 4 slots. Each diffs
    independently."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
        (Side.BUY, 1): DesiredOrderState(
            side=Side.BUY, level_idx=1, should_exist=True,
            price=0.99, size=2.0, reason="engine_quote",
        ),
        (Side.SELL, 0): DesiredOrderState.empty(
            Side.SELL, 0, reason="gate_suppressed"
        ),
        # SELL level 1 missing → defaults to empty
    }
    current = {
        (Side.SELL, 0): _wo(Side.SELL, OrderStatus.ACKED, price=1.01, local_id="L1"),
        # All others absent
    }
    actions = reconcile(desired=desired, current=current)
    by_slot = {(a.side, a.level_idx): a for a in actions}
    assert isinstance(by_slot[(Side.BUY, 0)], PlaceAction)
    assert isinstance(by_slot[(Side.BUY, 1)], PlaceAction)
    assert isinstance(by_slot[(Side.SELL, 0)], CancelAction)


def test_reconcile_is_pure() -> None:
    """Two reconciler calls with identical inputs produce identical
    outputs. No hidden state, no side effects."""
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
    }
    a1 = reconcile(desired=desired, current={})
    a2 = reconcile(desired=desired, current={})
    assert a1 == a2


# ---------------------------------------------------------------------------
# Wedge-probing — pure reconciler edge cases
# ---------------------------------------------------------------------------


def test_reconcile_mixed_states_across_rungs() -> None:
    """v1.4.62 wedge probe: reconciler handles a realistic
    multi-rung snapshot where every slot has a different status.
    Verifies each slot gets the right Action independently — no
    cross-slot leakage.
    """
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
        (Side.BUY, 1): DesiredOrderState.empty(Side.BUY, 1, reason="gate"),
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="engine_quote",
        ),
        (Side.SELL, 1): DesiredOrderState(
            side=Side.SELL, level_idx=1, should_exist=True,
            price=1.02, size=2.0, reason="engine_quote",
        ),
    }
    current = {
        # BUY 0: no current → Place
        # BUY 1: ACKED with desired=empty → Cancel
        (Side.BUY, 1): _wo(Side.BUY, OrderStatus.ACKED, price=0.99, local_id="L-b1"),
        # SELL 0: ACKED, desired needs reprice (px diff) → Amend or Cancel
        (Side.SELL, 0): _wo(Side.SELL, OrderStatus.ACKED, price=1.005, local_id="L-s0"),
        # SELL 1: SENT in flight → NoOp wait
        (Side.SELL, 1): _wo(Side.SELL, OrderStatus.SENT, local_id="L-s1"),
    }
    actions = reconcile(
        desired=desired,
        current=current,
        materiality_check=lambda c, d: True,
        amend_viable_check=lambda c, d: True,
    )
    by_slot = {(a.side, a.level_idx): a for a in actions}
    assert isinstance(by_slot[(Side.BUY, 0)], PlaceAction)
    assert isinstance(by_slot[(Side.BUY, 1)], CancelAction)
    assert isinstance(by_slot[(Side.SELL, 0)], AmendAction)
    assert isinstance(by_slot[(Side.SELL, 1)], NoOpAction)
    assert "in_flight_wait:SENT" in by_slot[(Side.SELL, 1)].reason


def test_reconcile_action_order_is_deterministic() -> None:
    """v1.4.62 wedge probe: reconciler produces actions in a stable
    order across runs. The dispatcher and decision-trace assume
    determinism. Implementation sorts by (side.value, level_idx).
    """
    desired = {
        (Side.SELL, 1): DesiredOrderState(
            side=Side.SELL, level_idx=1, should_exist=True,
            price=2.0, size=1.0, reason="r",
        ),
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=1.0, reason="r",
        ),
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.9, size=1.0, reason="r",
        ),
        (Side.BUY, 1): DesiredOrderState(
            side=Side.BUY, level_idx=1, should_exist=True,
            price=0.9, size=1.0, reason="r",
        ),
    }
    actions = reconcile(desired=desired, current={})
    slot_order = [(a.side, a.level_idx) for a in actions]
    assert slot_order == [
        (Side.BUY, 0),
        (Side.BUY, 1),
        (Side.SELL, 0),
        (Side.SELL, 1),
    ]


def test_reconcile_no_materiality_check_always_reprices() -> None:
    """v1.4.62: with ``materiality_check=None``, every ACKED slot
    triggers reprice. Used by reconciler tests; production always
    passes a check.
    """
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="r",
        ),
    }
    current = {(Side.BUY, 0): _wo(Side.BUY, OrderStatus.ACKED, price=1.0, size=3.0)}
    # Default amend_viable_check=None means falls back to CancelAction.
    actions = reconcile(desired=desired, current=current)
    assert len(actions) == 1
    assert isinstance(actions[0], CancelAction)
    assert actions[0].trigger_reason == "reprice_replace"


def test_reconcile_treats_new_local_as_in_flight() -> None:
    """v1.4.62 wedge probe: NEW_LOCAL is a latent-wedge status
    (staged but transport not enqueued). The reconciler treats it
    as in-flight (NoOp wait) so we don't dispatch a duplicate place.
    The stage-place idempotency check + watchdog handle stuck cases.
    """
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="r",
        ),
    }
    current = {(Side.BUY, 0): _wo(Side.BUY, OrderStatus.NEW_LOCAL, local_id="L1")}
    actions = reconcile(desired=desired, current=current)
    assert isinstance(actions[0], NoOpAction)
    assert "in_flight_wait" in actions[0].reason


def test_reconcile_processes_current_slots_without_desired_entry() -> None:
    """v1.4.62 wedge probe: orphan rungs (current state has level_idx
    beyond configured) get included in the action list. Default
    desired = empty → CancelAction for any active orphan.

    This is the operator-reduced-num-levels-per-side recovery path.
    """
    # No desired entry for SELL 1 — only current has it.
    desired = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="r",
        ),
    }
    current = {
        (Side.SELL, 1): _wo(Side.SELL, OrderStatus.ACKED, price=2.0, local_id="orphan"),
    }
    actions = reconcile(desired=desired, current=current)
    by_slot = {(a.side, a.level_idx): a for a in actions}
    # SELL 1 orphan must be cancelled.
    assert isinstance(by_slot[(Side.SELL, 1)], CancelAction)
    assert by_slot[(Side.SELL, 1)].current_local_id == "orphan"


def test_reconcile_purity_no_input_mutation() -> None:
    """v1.4.62 wedge probe: reconciler MUST NOT mutate its input
    dicts. Pre-/post-call equality verifies the pure-function
    contract — critical for snapshot-replay use cases.
    """
    import copy
    desired_orig = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="r",
        ),
    }
    current_orig = {
        (Side.BUY, 0): _wo(Side.BUY, OrderStatus.ACKED, price=1.0, local_id="L1"),
    }
    desired_snap = copy.deepcopy(desired_orig)
    current_snap = copy.copy(current_orig)  # WO is SimpleNamespace; shallow OK
    reconcile(
        desired=desired_orig, current=current_orig,
        materiality_check=lambda c, d: True,
    )
    assert desired_orig == desired_snap, "reconciler mutated desired"
    assert set(current_orig.keys()) == set(current_snap.keys())


# ---------------------------------------------------------------------------
# Gate composition wedge probes
# ---------------------------------------------------------------------------


def test_gate_composition_suppression_then_bypass_restores_reducer() -> None:
    """v1.4.62: end-to-end gate composition. Engine produces both
    sides. A side-suppression gate kills SELL. Reducing-side bypass
    (last fold) restores SELL because position is long.

    This is the v1.4.59 wedge fix expressed as a pure pipeline.
    """
    engine = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="engine_quote",
        ),
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="engine_quote",
        ),
    }
    # Gate fires on SELL.
    suppressed = apply_side_suppression(
        engine, side=Side.SELL, reason="post_only_cross_cooldown"
    )
    assert suppressed[(Side.SELL, 0)].should_exist is False
    # Bypass restores it because position is long (SELL is reducer).
    final = apply_reducing_side_bypass(
        suppressed, position_qty=1.0, engine_desired=engine
    )
    assert final[(Side.SELL, 0)].should_exist is True
    assert "reducing_side_bypass_overrode" in final[(Side.SELL, 0)].reason
    # BUY untouched (it's the adding side; suppression didn't fire on it).
    assert final[(Side.BUY, 0)] == engine[(Side.BUY, 0)]


def test_gate_composition_suppression_no_bypass_when_flat() -> None:
    """v1.4.62: at zero inventory, the bypass is a no-op. Defensive
    cooldowns correctly apply to both sides when flat (no inventory
    to "reduce", so no reducer needs protection)."""
    engine = {
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="engine_quote",
        ),
    }
    suppressed = apply_side_suppression(
        engine, side=Side.SELL, reason="cooldown"
    )
    final = apply_reducing_side_bypass(
        suppressed, position_qty=0.0, engine_desired=engine
    )
    assert final[(Side.SELL, 0)].should_exist is False


def test_gate_composition_two_gates_then_bypass() -> None:
    """v1.4.62: TWO cooldown gates fire on the same side. Bypass
    still restores the reducer (it consults engine_desired, not
    the cumulative gate output)."""
    engine = {
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="engine_quote",
        ),
    }
    # Gate 1.
    s1 = apply_side_suppression(engine, side=Side.SELL, reason="adverse_pause")
    # Gate 2 (no-op because already suppressed; defensive).
    s2 = apply_side_suppression(s1, side=Side.SELL, reason="post_only_cooldown")
    # Bypass (final).
    final = apply_reducing_side_bypass(
        s2, position_qty=2.0, engine_desired=engine
    )
    assert final[(Side.SELL, 0)].should_exist is True


def test_gate_composition_bypass_uses_engine_not_intermediate_state() -> None:
    """v1.4.62 wedge probe: bypass must read from the ENGINE
    snapshot, not from intermediate post-gate state. Otherwise a
    chain of gates with reason-overwriting could mask the original
    quote."""
    engine = {
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="engine_quote",
        ),
    }
    # Intermediate state has SELL absent entirely (not just suppressed).
    intermediate = {}
    # Bypass should still work — it reads from engine_desired.
    final = apply_reducing_side_bypass(
        intermediate, position_qty=2.0, engine_desired=engine
    )
    # Bypass only restores slots that EXIST in the intermediate
    # (it's a fold, not an injection). Verifies the bypass is
    # conservative.
    assert (Side.SELL, 0) not in final


def test_apply_side_suppression_idempotent() -> None:
    """v1.4.62: applying the same gate twice is a no-op. Important
    because gates can re-fire over multiple ticks; the desired
    state should be stable."""
    engine = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="r",
        ),
    }
    once = apply_side_suppression(engine, side=Side.BUY, reason="gate")
    twice = apply_side_suppression(once, side=Side.BUY, reason="gate")
    assert once == twice


def test_apply_side_suppression_preserves_other_side() -> None:
    """v1.4.62: gate on BUY must not affect SELL slots."""
    engine = {
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="r",
        ),
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="r",
        ),
    }
    out = apply_side_suppression(engine, side=Side.BUY, reason="gate")
    assert out[(Side.BUY, 0)].should_exist is False
    assert out[(Side.SELL, 0)] == engine[(Side.SELL, 0)]


# ---------------------------------------------------------------------------
# Action-equality wedge probes
# ---------------------------------------------------------------------------


def test_action_types_are_distinct_classes() -> None:
    """v1.4.62: each Action subclass is its own type — dispatcher's
    isinstance checks must be unambiguous."""
    p = PlaceAction(side=Side.BUY, level_idx=0, desired=DesiredOrderState.empty(Side.BUY, 0))
    a = AmendAction(side=Side.BUY, level_idx=0, desired=DesiredOrderState.empty(Side.BUY, 0), current_local_id="L")
    c = CancelAction(side=Side.BUY, level_idx=0, current_local_id="L", trigger_reason="r")
    n = NoOpAction(side=Side.BUY, level_idx=0, reason="r")
    assert isinstance(p, PlaceAction)
    assert not isinstance(p, AmendAction)
    assert isinstance(a, AmendAction)
    assert not isinstance(a, PlaceAction)
    assert isinstance(c, CancelAction)
    assert isinstance(n, NoOpAction)


def test_actions_are_frozen() -> None:
    """v1.4.62: Action dataclasses are frozen — accidental mutation
    raises. Prevents shared-state bugs in the dispatcher."""
    import dataclasses
    p = PlaceAction(side=Side.BUY, level_idx=0, desired=DesiredOrderState.empty(Side.BUY, 0))
    try:
        p.level_idx = 99  # type: ignore[misc]
    except (dataclasses.FrozenInstanceError, AttributeError):
        return
    raise AssertionError("PlaceAction should be frozen")


# ---------------------------------------------------------------------------
# Reconciler ordering across sides (BUY before SELL) — sanity
# ---------------------------------------------------------------------------


def test_reconcile_buy_processed_before_sell_within_same_level() -> None:
    """v1.4.62: action ordering — BUY level 0 before SELL level 0.
    The dispatcher relies on this for consistent ack-timing
    measurements and tests assert on it."""
    desired = {
        (Side.SELL, 0): DesiredOrderState(
            side=Side.SELL, level_idx=0, should_exist=True,
            price=1.01, size=3.0, reason="r",
        ),
        (Side.BUY, 0): DesiredOrderState(
            side=Side.BUY, level_idx=0, should_exist=True,
            price=1.0, size=3.0, reason="r",
        ),
    }
    actions = reconcile(desired=desired, current={})
    assert actions[0].side == Side.BUY
    assert actions[1].side == Side.SELL
