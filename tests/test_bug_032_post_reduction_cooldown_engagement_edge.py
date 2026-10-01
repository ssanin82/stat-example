"""BUG-032 — ``post_reduction_cooldown.fire_count`` must count
engagement events, not potential-arm events.

Pre-v1.5.191 behaviour:

* ``BotState._note_inventory_reduction_for_cooldown`` (called on
  every reducing fill, regime-blind) bumped ``fire_count`` AND set
  the arming timestamp.
* The cooldown's actual engagement + clear logic lives in
  ``Bot._apply_eligibility_engine``, gated to DEFENSIVE/SHOCK regimes.
* ``cleared_via_favorable_total`` / ``cleared_via_ceiling_total`` /
  ``cleared_via_position_favorable_total`` only increment inside
  that regime-gated block.

Result: NORMAL/CAUTIOUS-only sessions reported
``fire_count: 7, cleared_via_*: 0`` — semantically inconsistent
because the gate had never actually engaged. The acceptance script's
position-favorable-exit metric (``cleared_via_position_favorable /
fire_count``) was silently broken for this gate.

v1.5.191 fix: ``fire_count`` increment moved out of the state-arming
method into the actual engagement edge in
``Bot._apply_eligibility_engine`` — alongside the WARNING log on the
``was_active_last_tick`` False→True edge. Now ``fire_count`` matches
``cleared_via_*`` semantics: both count active engagements.

This file pins:

1. The state-arming method does NOT bump ``fire_count`` (the bug).
2. The arming method still sets timestamp + suppressed_side
   (the rest of the contract is unchanged).
3. A NORMAL-only session shows ``fire_count == 0`` regardless of how
   many reducing fills occurred (was non-zero pre-fix).

Per CLAUDE.md: only this test file is run from the assistant.
"""

from __future__ import annotations

from app.config import Settings
from app.enums import QuoteEligibility, Side
from app.state import BotState


def _bs(*, cd_seconds: float = 60.0) -> BotState:
    return BotState(Settings(POST_REDUCTION_COOLDOWN_SECONDS=cd_seconds))


def test_state_arming_does_not_bump_fire_count() -> None:
    """The headline regression guard. Pre-v1.5.191 this assertion
    would fail (fire_count would be 1)."""
    bs = _bs()
    bs._note_inventory_reduction_for_cooldown(
        now_mono=1000.0,
        fill_side=Side.SELL,
        prev_qty=+5.0,
        new_qty=+3.0,
    )
    assert bs.post_reduction_cooldown_fire_count == 0


def test_state_arming_still_sets_timestamp_and_side() -> None:
    """The arming method retains its core responsibility: record
    when the reducing fill happened + which side to suppress."""
    bs = _bs()
    bs._note_inventory_reduction_for_cooldown(
        now_mono=1000.0,
        fill_side=Side.SELL,
        prev_qty=+5.0,
        new_qty=+3.0,
    )
    assert bs.last_inventory_reduction_at_mono == 1000.0
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_SELL_ONLY
    )


def test_many_reducing_fills_do_not_inflate_fire_count() -> None:
    """The smoking-gun simulation: 7 reducing fills in a NORMAL-only
    session. Pre-v1.5.191 this reported fire_count=7 with
    cleared_via_*=0 (the v1.5.187 snapshot pattern). Post-fix it's
    fire_count=0 because the cooldown never engaged in a
    DEFENSIVE/SHOCK regime."""
    bs = _bs()
    # Simulate seven reducing-fill arming events.
    for i, (prev, new) in enumerate(
        [
            (+5.0, +4.0),
            (+4.0, +3.0),
            (+3.0, +2.0),
            (+2.0, +1.0),
            (-5.0, -4.0),
            (-4.0, -3.0),
            (-3.0, -2.0),
        ]
    ):
        side = Side.SELL if prev > 0 else Side.BUY
        bs._note_inventory_reduction_for_cooldown(
            now_mono=1000.0 + i * 5.0,
            fill_side=side,
            prev_qty=prev,
            new_qty=new,
        )
    # Pre-v1.5.191: fire_count=7. Post-v1.5.191: 0 (gate never
    # engaged because we never simulated DEFENSIVE/SHOCK).
    assert bs.post_reduction_cooldown_fire_count == 0
    # The LAST arming's timestamp wins (the cooldown extends
    # naturally on each new reducing fill).
    assert bs.last_inventory_reduction_at_mono == 1030.0


def test_cleared_via_counters_remain_consistent_with_fire_count() -> None:
    """The invariant we want to preserve: fire_count counts
    engagements; cleared_via_* counters count exits from those
    engagements. Both should be 0 in a NORMAL-only session.

    This pins the structural property that's the entire point of the
    fix — fire_count and cleared_via_* should always have matching
    semantics (both count active-window events)."""
    bs = _bs()
    bs._note_inventory_reduction_for_cooldown(
        now_mono=1000.0,
        fill_side=Side.SELL,
        prev_qty=+5.0,
        new_qty=+3.0,
    )
    assert bs.post_reduction_cooldown_fire_count == 0
    assert bs.post_reduction_cooldown_cleared_via_favorable_total == 0
    assert bs.post_reduction_cooldown_cleared_via_ceiling_total == 0
    # Pre-v1.5.191 invariant violation: fire_count > 0 but all
    # cleared_via_* = 0. This assertion now holds trivially.
    fc = bs.post_reduction_cooldown_fire_count
    total_cleared = (
        bs.post_reduction_cooldown_cleared_via_favorable_total
        + bs.post_reduction_cooldown_cleared_via_ceiling_total
    )
    # Either both zero (never engaged) or fire_count >= total_cleared
    # (some still active). Never fire_count > 0 with total_cleared = 0
    # PERMANENTLY (which was the pre-fix bug shape).
    assert fc >= total_cleared
