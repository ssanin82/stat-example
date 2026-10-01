"""v1.4.60 wedge-elimination Phase 5 — reconciler architecture.

Phase 0 contract (v1.4.68 wedge-elimination-cleanup)
----------------------------------------------------

**Concurrency model.** The bot supports MULTI-RUNG ladders: an order's
identity is ``(symbol, side, level_idx)``. At most ONE live order per
slot, but multiple slots per side are independent. The config knob
``LADDER_NUM_LEVELS_PER_SIDE`` (default 1) sets the number of slots
per side.

Every layer in the codebase MUST speak this model:

* ``BotState.order_store`` (Phase 3A) — keyed by ``(side, level_idx)``
* Exchange reconcile (Phase 1D) — matches remote orders to local slots
  by ``client_order_id``, not just by side
* Pre-send risk check (Phase 2C) — sums size across all rungs per side
* Hard age cap (Phase 1C) — fires per slot, not per side
* Reconciler diff (this module) — already slot-keyed since v1.4.60

The historical "one live order per side" model (``working_bid`` /
``working_ask`` accessors) is DEPRECATED and removed in Phase 3D.

**Layer-mutation rule.** Every ``OrderStatus`` change goes through
``OrderStore.transition(wo, new_status, reason)`` (Phase 3A). Direct
``wo.status = X`` assignments are contract violations.

----

The bot's order management was historically organised as:

  decision (engine + gates) ──► orchestrate (decide + dispatch in one)

The "decide" and "dispatch" steps were tangled in ``_orchestrate``,
which made the gate-composition story fragile: each gate had its
own attachment point (some in the engine, some in ``stage_place``,
some in ``orchestrate``), and the "reducing-side bypass" rule that
prevents the legitimate-defenses-aligned-across-both-sides wedge
was only present for ``adverse_side_pause``, not for any other
defensive cooldown.

This module formalises the reconciler architecture:

  decision (pure) ──► DesiredLadderState ──► reconcile (pure) ──► Actions ──► dispatch

Every gate is now a pure ``DesiredLadderState → DesiredLadderState``
fold. The reducing-side bypass is the LAST fold and applies to
every gate uniformly. Result: defensive gates can never fully
suppress the reducing side while inventory exists. The
v1.4.59-260518-173229 wedge class (adverse_side_pause on adding
side + post_only_cross_cooldown on reducing side = zero orders) is
structurally impossible by construction.

Module shape:

  * Types: ``DesiredOrderState``, ``DesiredLadderState``, ``Action``
    + subclasses (frozen dataclasses)
  * Pure decision layer: ``compute_desired_state(...)``
  * Pure reconciler: ``reconcile(desired, current)``
  * Gate functions: ``apply_reducing_side_bypass(...)``,
    ``apply_post_only_cross_cooldown(...)``, etc.

The dispatcher (executing actions) stays in ``app/execution.py`` —
the side-effect layer is unchanged. Only the decision + diff
become explicit, pure, testable layers.

Phase 5 of ``plans/wedge-elimination.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional

from app.enums import OrderStatus, Side


# ---------------------------------------------------------------------------
# Desired-state types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DesiredOrderState:
    """What the bot WANTS one (side, level_idx) slot to look like
    on the venue this tick. Pure data; no side effects.

    ``should_exist=False`` is the "I want NO order at this slot"
    state — the reconciler will produce a CancelAction if there's
    a live working order, or NoOpAction if the slot is already
    empty.

    ``should_exist=True`` requires ``price > 0`` and ``size > 0``
    — the values the bot wants on the wire.

    ``reason`` is a short, machine-readable tag describing WHY the
    decision layer produced this state. Carried into the decision-
    trace buffer for postmortem. Examples:
      * ``"engine_quote"`` — the engine produced this quote
      * ``"gate:adverse_side_pause:reducer_bypassed"`` — a gate
        wanted to suppress but the reducing-side bypass overrode
      * ``"gate:post_only_cross_cooldown:suppressed"`` — a gate
        suppressed this side
      * ``"engine_no_quote"`` — engine returned no quote for this side
    """

    side: Side
    level_idx: int
    should_exist: bool
    price: float = 0.0
    size: float = 0.0
    target_half_spread_bps: Optional[float] = None
    aging_tighten_applied: bool = False
    reason: str = ""

    @classmethod
    def empty(cls, side: Side, level_idx: int, reason: str = "") -> "DesiredOrderState":
        """Construct a ``should_exist=False`` slot. Used by the
        decision layer when the engine produced no quote or a gate
        suppressed the side.
        """
        return cls(
            side=side,
            level_idx=level_idx,
            should_exist=False,
            reason=reason,
        )


# A DesiredLadderState is a mapping keyed by (Side, level_idx) →
# DesiredOrderState. Using a plain dict keeps the structure
# composable (gates produce new dicts by ``replace()``).
DesiredLadderState = dict[tuple[Side, int], DesiredOrderState]


# ---------------------------------------------------------------------------
# Action types — what the dispatcher should DO
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlaceAction:
    """Place a fresh order for the given (side, level_idx) slot.

    ``desired`` carries the price/size/reason; the dispatcher uses
    this to call ``_stage_place_order_local`` + ``_enqueue_place_transport``.
    """

    side: Side
    level_idx: int
    desired: DesiredOrderState


@dataclass(frozen=True, slots=True)
class AmendAction:
    """Amend the existing working order for this slot to match
    ``desired`` price/size. The reconciler emits this when
    amend-on-reprice is viable (per ``_amend_viable``); the
    dispatcher confirms viability and falls back to CancelAction
    if not.

    Carries ``current_local_id`` so the dispatcher can address the
    correct WorkingOrder.
    """

    side: Side
    level_idx: int
    desired: DesiredOrderState
    current_local_id: str


@dataclass(frozen=True, slots=True)
class CancelAction:
    """Cancel the existing working order for this slot. Emitted
    when:
      * ``desired.should_exist=False`` AND there's a live WO
      * ``desired.should_exist=True`` AND the price/size diff
        from current is large enough to require cancel-then-place
        (because ``_amend_viable`` returned False at the dispatcher)
      * a hard age cap fired (the WO is past its
        ``BEHIND_TOUCH_MAX_AGE_SECONDS`` / similar)

    Carries the cancel ``trigger_reason`` for postmortem visibility.
    """

    side: Side
    level_idx: int
    current_local_id: str
    trigger_reason: str


@dataclass(frozen=True, slots=True)
class NoOpAction:
    """No action needed this tick.

    Emitted when:
      * current matches desired (within materiality)
      * an action is already in-flight (SENT, CANCEL_PENDING, AMEND_PENDING)
      * desired and current are both empty

    ``reason`` is the short tag explaining why the reconciler
    chose to do nothing. Drives the decision-trace buffer.
    """

    side: Side
    level_idx: int
    reason: str


Action = PlaceAction | AmendAction | CancelAction | NoOpAction


# ---------------------------------------------------------------------------
# Gate functions — pure DesiredLadderState → DesiredLadderState folds
# ---------------------------------------------------------------------------


def _is_reducing_side(side: Side, position_qty: float) -> bool:
    """True iff trading the given ``side`` would REDUCE the bot's
    inventory.

    Long position (qty > 0): SELL is reducing.
    Short position (qty < 0): BUY is reducing.
    Flat (qty == 0): neither side reduces anything.
    """
    if position_qty > 0:
        return side == Side.SELL
    if position_qty < 0:
        return side == Side.BUY
    return False


def apply_reducing_side_bypass(
    desired: DesiredLadderState,
    *,
    position_qty: float,
    engine_desired: DesiredLadderState,
    enabled: bool = True,
) -> DesiredLadderState:
    """The structural invariant that makes the v1.4.59 wedge class
    impossible: when inventory is non-zero, the reducing side is
    NEVER fully suppressed by defensive gates.

    Why: defensive gates (``adverse_side_pause``,
    ``post_only_cross_cooldown``, etc.) back off after adverse
    signals — they're protective for the ADDING side (avoid
    growing inventory in toxic conditions, avoid re-crossing the
    spread). Applied to the REDUCING side, they actively HURT —
    the bot can't flatten the inventory it already holds. The
    venue's 51604 rejection cost is small compared to "bot can't
    reduce a +5 long during a downward move".

    Mechanism: if a downstream gate set ``should_exist=False`` on
    the reducing side, this fold un-suppresses it by restoring
    the engine's original desired-state for that slot.

    ``engine_desired`` is the BEFORE-gates desired-state from the
    decision layer — used as the restoration source so we never
    invent prices/sizes here.

    ``enabled`` defaults to True. Operators can disable via
    ``RECONCILER_REDUCING_SIDE_BYPASS_ENABLED=false`` for testing
    or rollback.

    The bypass applies to ALL levels of the reducing side, not
    just level 0 — multi-rung ladders should reduce in full when
    inventory exists.

    Position == 0 (flat): bypass is a no-op (no "reducing side").
    """
    if not enabled:
        return desired
    if position_qty == 0.0:
        return desired
    result = dict(desired)
    for (side, lvl), state in list(desired.items()):
        if not _is_reducing_side(side, position_qty):
            continue
        if state.should_exist:
            continue
        # Gate suppressed the reducing side. Restore.
        original = engine_desired.get((side, lvl))
        if original is None or not original.should_exist:
            # Engine itself didn't produce a quote for this slot —
            # nothing to restore. Bypass doesn't INVENT quotes; it
            # only protects engine output from being suppressed by
            # downstream gates.
            continue
        restored_reason = (
            f"{original.reason}|reducing_side_bypass_overrode:{state.reason}"
            if state.reason and original.reason
            else "reducing_side_bypass_overrode"
        )
        result[(side, lvl)] = replace(original, reason=restored_reason)
    return result


def apply_side_suppression(
    desired: DesiredLadderState,
    *,
    side: Side,
    reason: str,
    levels: Optional[Iterable[int]] = None,
) -> DesiredLadderState:
    """Mark all slots on ``side`` as ``should_exist=False``. Generic
    "this side is suppressed" fold used by gates that want to block
    placement on one side wholesale (adverse_side_pause,
    post_only_cross_cooldown, microprice_gate side suppression, etc.).

    ``levels`` is an optional whitelist — if provided, only those
    level_idx values are suppressed. Default = all levels for the
    side.

    ``reason`` is the gate identifier (e.g.,
    ``"adverse_side_pause:remaining_s=26.5"``). Used by postmortem.
    """
    result = dict(desired)
    for (s, lvl), state in list(desired.items()):
        if s != side:
            continue
        if levels is not None and lvl not in set(levels):
            continue
        if not state.should_exist:
            # Already suppressed by an earlier gate.
            continue
        result[(s, lvl)] = DesiredOrderState.empty(
            side=s, level_idx=lvl, reason=f"gate:{reason}"
        )
    return result


# ---------------------------------------------------------------------------
# Reconciler — pure diff DesiredLadderState vs current WorkingOrders
# ---------------------------------------------------------------------------


def reconcile(
    desired: DesiredLadderState,
    current: dict[tuple[Side, int], "_WorkingOrderView"],
    *,
    materiality_check: Optional[
        callable  # type: ignore[valid-type]
    ] = None,
    amend_viable_check: Optional[
        callable  # type: ignore[valid-type]
    ] = None,
) -> list[Action]:
    """Diff the bot's DESIRED ladder state against its CURRENT
    working orders, produce a list of actions for the dispatcher.

    Pure function: no side effects, no state mutation, no I/O.
    Same input → same output, every time.

    For each (side, level_idx) slot present in EITHER desired OR
    current:

      desired.should_exist  current_status              → action
      ────────────────────  ──────────────────────────  ─────────────
      False                 None                         NoOpAction("empty_match")
      False                 CANCELED/FILLED/REJECTED     NoOpAction("terminal_match")
      False                 ACKED/PARTIAL/AMEND_PENDING  CancelAction("desired_none")
      False                 SENT/CANCEL_PENDING          NoOpAction("in_flight_wait")
      True                  None                         PlaceAction
      True                  CANCELED/FILLED/REJECTED     PlaceAction (terminal → fresh)
      True                  ACKED/PARTIAL + same px/sz   NoOpAction("acked_no_reprice")
      True                  ACKED/PARTIAL + reprice OK   AmendAction (if viable) or CancelAction
      True                  SENT/CANCEL_PENDING/AMEND_PENDING  NoOpAction("in_flight_wait")

    ``materiality_check(current_wo, desired_state) -> bool`` — caller
    provides; returns True iff the price/size delta is large enough
    to warrant a reprice. NULL → reprice always (test default).

    ``amend_viable_check(current_wo, desired_state) -> bool`` —
    caller provides; returns True iff the dispatcher can amend in-
    place (vs cancel+place). NULL → never amend-viable (test
    default; falls back to CancelAction for reprices).
    """
    actions: list[Action] = []
    # All slots from BOTH the desired ladder and the current state.
    all_slots = set(desired.keys()) | set(current.keys())
    for slot in sorted(all_slots, key=lambda s: (s[0].value, s[1])):
        side, lvl = slot
        d = desired.get(slot)
        c = current.get(slot)
        # Default desired = empty slot. Mirrors the engine's
        # "no quote for this rung" case.
        if d is None:
            d = DesiredOrderState.empty(
                side=side, level_idx=lvl, reason="desired_absent"
            )
        # No current working order.
        if c is None or c.status in _TERMINAL:
            if d.should_exist:
                actions.append(PlaceAction(side=side, level_idx=lvl, desired=d))
            else:
                actions.append(
                    NoOpAction(
                        side=side,
                        level_idx=lvl,
                        reason="empty_match" if c is None else "terminal_match",
                    )
                )
            continue
        # Current is in-flight. Wait for response.
        if c.status in _IN_FLIGHT:
            actions.append(
                NoOpAction(
                    side=side,
                    level_idx=lvl,
                    reason=f"in_flight_wait:{c.status.value}",
                )
            )
            continue
        # Current is ACKED or PARTIAL — eligible for cancel / amend.
        if not d.should_exist:
            actions.append(
                CancelAction(
                    side=side,
                    level_idx=lvl,
                    current_local_id=c.order_id_local,
                    trigger_reason="desired_none",
                )
            )
            continue
        # Both want and have an order. Compare.
        if materiality_check is not None and not materiality_check(c, d):
            actions.append(
                NoOpAction(
                    side=side,
                    level_idx=lvl,
                    reason="acked_no_reprice_needed",
                )
            )
            continue
        # Need a reprice. Try amend first (if dispatcher signals
        # viability), else cancel-and-place.
        if amend_viable_check is not None and amend_viable_check(c, d):
            actions.append(
                AmendAction(
                    side=side,
                    level_idx=lvl,
                    desired=d,
                    current_local_id=c.order_id_local,
                )
            )
        else:
            actions.append(
                CancelAction(
                    side=side,
                    level_idx=lvl,
                    current_local_id=c.order_id_local,
                    trigger_reason="reprice_replace",
                )
            )
    return actions


# Status partitions used by the reconciler. Frozen sets at module
# load so the per-tick hot path doesn't reallocate.
#
# v1.4.67: DESYNC is included as terminal here. DESYNC means the bot
# detected an exchange↔local mismatch and has already issued cancels
# for the orphan(s) — the WO is a tombstone, not actionable. Treating
# DESYNC as terminal makes the reconciler emit ``PlaceAction`` for a
# fresh order on the slot (when desired) instead of ``CancelAction``
# against an order it has already given up on.
_TERMINAL = frozenset(
    {OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.DESYNC}
)
_IN_FLIGHT = frozenset(
    {OrderStatus.SENT, OrderStatus.CANCEL_PENDING, OrderStatus.AMEND_PENDING, OrderStatus.NEW_LOCAL}
)


# ---------------------------------------------------------------------------
# Type alias for the reconciler's "current state" view. Loose
# protocol so tests can pass minimal stubs; in production the
# OrderManager passes its WorkingOrder objects which match.
# ---------------------------------------------------------------------------


class _WorkingOrderView:
    """Structural protocol for the reconciler's view of a
    WorkingOrder. The real ``app.models.WorkingOrder`` matches by
    duck-typing; tests pass simple namespace objects.
    """

    order_id_local: str
    status: OrderStatus
    side: Side
    price: float
    size: float
