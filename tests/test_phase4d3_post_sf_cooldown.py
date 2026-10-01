"""Phase 4D.3 -- post-SF cooldown gate tests.

History:

* v1.5.2 — original design: after SF exit, suppress the side that
  would re-add to the pre-SF position direction (pre-SF LONG →
  QUOTE_SELL_ONLY for the cooldown window). Targeted the v1.4.219
  pattern of 3 SFs in 27 min where the bot kept re-leaning into the
  same direction.

* v1.5.154 (this rewrite) — convert to **HOLD_ALL pause**. The
  side-suppression design backfired in sustained trending regimes:
  after a long-closing SF in an uptrend, the 5-min SELL_ONLY window
  guaranteed the bot kept getting adversely selected on SELL until
  the cooldown cleared OR another SF re-armed it the other way.
  Snapshot ``v1.5.150-260525-220850`` captured 8 -3 SHORT vs 2 +3 LONG
  transitions over 23 min in an uptrend session under the v1.5.39
  300 s + side-force combination.

  The fix: HOLD_ALL pause regardless of pre-SF direction. The bot
  rests, then resumes two-sided quoting when the timer expires. The
  ``lean_side`` field is still populated by the arming code for
  dashboard / postmortem visibility ("which direction did we just
  SF from?") but does NOT drive the eligibility decision anymore.

  v1.5.153 already dropped POST_SF_COOLDOWN_SECONDS 300 → 60 to
  contain the magnitude; v1.5.154 removes the directional asymmetry
  itself.

Direction logic (v1.5.154):
  * Pre-SF LONG (pre_qty > 0)  → arm cooldown; gate forces HOLD_ALL
  * Pre-SF SHORT (pre_qty < 0) → arm cooldown; gate forces HOLD_ALL
  * Pre-SF FLAT (pre_qty == 0) → no arm (SF would have been a no-op)
  * pre_qty unknown (None)     → no arm

The gate composes via ``more_restrictive`` — it can only narrow
eligibility, never widen it. HOLD_ALL is the most restrictive
eligibility, so the cooldown deterministically wins the
composition.

These tests directly exercise the state-mutation contract and the
``_apply_post_sf_cooldown_gate`` method via a minimal stub harness.
The arming side of the contract (SF entry captures pre-qty, SF exit
populates post_sf_cooldown_until_mono + lean_side) is exercised by
state-direct manipulation -- driving a real Bot through SF would
require an order-execution harness out of scope here.

Per CLAUDE.md: only this test file is run from the assistant; full-
suite verification is the CI daemon's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from app.bot import Bot
from app.config import Settings
from app.enums import QuoteEligibility, Side
from app.quote_eligibility import QuoteEligibilityResult
from app.state import BotState


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


@dataclass
class _FakeClock:
    """Drive-able clock so tests can advance time deterministically.
    The gate uses ``self._clock.monotonic()`` exclusively; nothing
    else is exercised."""

    t: float = 0.0

    def monotonic(self) -> float:
        return self.t


@dataclass
class _GateHarness:
    """Minimal stub providing ``_settings``, ``_state``, ``_clock``
    -- the only attributes ``_apply_post_sf_cooldown_gate`` touches.
    Letting tests invoke the bound method without spinning up a real
    ``Bot`` (which needs an exchange client, storage, etc.)."""

    _settings: Settings
    _state: BotState
    _clock: _FakeClock


def _make_harness(
    *,
    enabled: bool = True,
    cooldown_s: float = 60.0,
) -> _GateHarness:
    settings = Settings(
        POST_SF_COOLDOWN_ENABLED=enabled,
        POST_SF_COOLDOWN_SECONDS=cooldown_s,
    )
    return _GateHarness(
        _settings=settings,
        _state=BotState(settings),
        _clock=_FakeClock(t=1000.0),
    )


def _eff_q(
    eligibility: QuoteEligibility = QuoteEligibility.QUOTE_BOTH,
    reason: str = "base",
) -> QuoteEligibilityResult:
    """Fresh eligibility result -- the gate only touches
    ``eligibility`` and ``reason``; the rest of the dataclass fields
    are filled with neutral placeholders so dataclass.replace works."""
    return QuoteEligibilityResult(
        eligibility=eligibility,
        reason=reason,
        seconds_since_last_public_book_update=0.0,
        effective_staleness_ms=0.0,
        market_data_gap_p95_ms=None,
        market_data_gap_median_ms=None,
        mid_return_100ms_bps=None,
        mid_return_250ms_bps=None,
        mid_return_500ms_bps=None,
        jump_100ms_bps=None,
        jump_250ms_bps=None,
        jump_500ms_bps=None,
        in_cooldown=False,
    )


def _arm(
    h: _GateHarness,
    *,
    lean_side: Optional[Side],
    cooldown_s: float = 60.0,
) -> None:
    """Synthesise an SF-exit having armed the gate. Tests don't need
    to drive the real SF entry/exit code path -- the gate only reads
    ``post_sf_cooldown_until_mono`` + ``post_sf_cooldown_lean_side``."""
    h._state.post_sf_cooldown_until_mono = h._clock.t + cooldown_s
    h._state.post_sf_cooldown_lean_side = lean_side


# ---------------------------------------------------------------------------
# Dormant cases (no-op pass-through)
# ---------------------------------------------------------------------------


def test_disabled_setting_passes_through_unchanged():
    """When ``POST_SF_COOLDOWN_ENABLED=False`` the gate is a no-op
    even if state has stale armed values. Lets the operator toggle
    the gate off via env without restart-state-cleaning."""
    h = _make_harness(enabled=False)
    _arm(h, lean_side=Side.BUY)
    eff_in = _eff_q(QuoteEligibility.QUOTE_BOTH)
    eff_out = Bot._apply_post_sf_cooldown_gate(h, eff_in)
    assert eff_out.eligibility == QuoteEligibility.QUOTE_BOTH
    assert eff_out.reason == "base"
    # active_last_tick stays False when disabled.
    assert h._state.post_sf_cooldown_active_last_tick is False
    # fire_count must not bump under disabled.
    assert h._state.post_sf_cooldown_fire_count_total == 0


def test_unarmed_passes_through_unchanged():
    """Default state: gate enabled, no SF has fired -- deadline=0,
    lean_side=None. Gate is a no-op. This is the steady-state of
    a NORMAL-quoting bot."""
    h = _make_harness()
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert eff_out.eligibility == QuoteEligibility.QUOTE_BOTH
    assert h._state.post_sf_cooldown_active_last_tick is False
    assert h._state.post_sf_cooldown_fire_count_total == 0


# ---------------------------------------------------------------------------
# Armed cases -- HOLD_ALL semantics (v1.5.154)
# ---------------------------------------------------------------------------


def test_pre_sf_long_forces_hold_all():
    """v1.5.154: Pre-SF LONG arms the cooldown → gate forces
    HOLD_ALL (no quoting either side). The original v1.5.2 design
    forced QUOTE_SELL_ONLY; the snapshot ``v1.5.150-260525-220850``
    proved that mechanism mis-fired in trending markets by guaranteeing
    adverse selection on the forced side."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY)  # pre-SF LONG
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL
    assert "post_sf_cooldown:HOLD_ALL" in eff_out.reason
    assert "prev=LONG" in eff_out.reason
    assert h._state.post_sf_cooldown_active_last_tick is True
    assert h._state.post_sf_cooldown_fire_count_total == 1


def test_pre_sf_short_forces_hold_all():
    """Mirror: Pre-SF SHORT also forces HOLD_ALL. Same logic as the
    LONG case — the v1.5.154 design is direction-symmetric: both
    pre-SF directions produce HOLD_ALL for the cooldown window."""
    h = _make_harness()
    _arm(h, lean_side=Side.SELL)
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL
    assert "post_sf_cooldown:HOLD_ALL" in eff_out.reason
    assert "prev=SHORT" in eff_out.reason
    assert h._state.post_sf_cooldown_active_last_tick is True


def test_pre_sf_long_no_longer_forces_quote_sell_only():
    """v1.5.154 regression pin: the gate must NOT produce
    QUOTE_SELL_ONLY under any input. The mis-fire from
    v1.5.150-260525-220850 was forced SELL_ONLY on a LONG
    pre-SF. With HOLD_ALL this cannot recur regardless of
    upstream eligibility."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY)
    eff_out = Bot._apply_post_sf_cooldown_gate(
        h, _eff_q(QuoteEligibility.QUOTE_BOTH)
    )
    assert eff_out.eligibility != QuoteEligibility.QUOTE_SELL_ONLY
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL


def test_pre_sf_short_no_longer_forces_quote_buy_only():
    """Mirror regression pin."""
    h = _make_harness()
    _arm(h, lean_side=Side.SELL)
    eff_out = Bot._apply_post_sf_cooldown_gate(
        h, _eff_q(QuoteEligibility.QUOTE_BOTH)
    )
    assert eff_out.eligibility != QuoteEligibility.QUOTE_BUY_ONLY
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL


def test_lean_side_none_still_fires_hold_all():
    """Defensive: if a future code path arms the timer without
    populating lean_side (or if lean_side gets cleared mid-window),
    the timer is the authority — gate still forces HOLD_ALL.
    Direction string is ``UNKNOWN`` for operator-visible
    debuggability.

    v1.5.154 semantics change vs v1.5.2: the old design fell through
    to pass-through when lean_side was None (since it couldn't pick
    a side). HOLD_ALL doesn't need a direction, so the timer alone
    is enough."""
    h = _make_harness()
    h._state.post_sf_cooldown_until_mono = h._clock.t + 60.0
    h._state.post_sf_cooldown_lean_side = None
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL
    assert "prev=UNKNOWN" in eff_out.reason
    assert h._state.post_sf_cooldown_active_last_tick is True


def test_reason_string_includes_remaining_seconds():
    """The reason string carries the time remaining so the operator
    can see "how much longer" in /status output. Format:
    ``post_sf_cooldown:HOLD_ALL_prev=<DIR>_<N>s``."""
    h = _make_harness(cooldown_s=60.0)
    _arm(h, lean_side=Side.BUY, cooldown_s=45.0)
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    # 45 s remaining; reason carries the integer second count.
    assert "_45s" in eff_out.reason


# ---------------------------------------------------------------------------
# Composition with other gates -- more_restrictive semantics
#
# With v1.5.154 forcing HOLD_ALL, composition becomes trivial:
# HOLD_ALL is the most restrictive eligibility, so any upstream
# eligibility gets clamped to HOLD_ALL. The old test names
# (e.g. "conflicting one-sided yields HOLD_ALL") are renamed to
# reflect the new semantics: ALL upstream eligibilities yield
# HOLD_ALL when the cooldown is armed.
# ---------------------------------------------------------------------------


def test_composes_with_quote_both_yields_hold_all():
    """The common case: upstream eligibility is QUOTE_BOTH (no other
    gate firing), cooldown is armed → HOLD_ALL wins."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY)
    eff_in = _eff_q(QuoteEligibility.QUOTE_BOTH, "upstream_clean")
    eff_out = Bot._apply_post_sf_cooldown_gate(h, eff_in)
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL


def test_composes_with_hold_all_stays_hold_all():
    """HOLD_ALL is its own intersection. Idempotent."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY)
    eff_in = _eff_q(QuoteEligibility.HOLD_ALL, "higher_priority_gate")
    eff_out = Bot._apply_post_sf_cooldown_gate(h, eff_in)
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL


def test_composes_with_quote_buy_only_collapses_to_hold_all():
    """Upstream was QUOTE_BUY_ONLY (say, shock_gate locking BID-side
    only); cooldown adds HOLD_ALL → intersection = HOLD_ALL."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY)
    eff_in = _eff_q(QuoteEligibility.QUOTE_BUY_ONLY, "shock_gate_long")
    eff_out = Bot._apply_post_sf_cooldown_gate(h, eff_in)
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL


def test_composes_with_quote_sell_only_collapses_to_hold_all():
    """Mirror: upstream QUOTE_SELL_ONLY + cooldown HOLD_ALL = HOLD_ALL."""
    h = _make_harness()
    _arm(h, lean_side=Side.SELL)
    eff_in = _eff_q(QuoteEligibility.QUOTE_SELL_ONLY, "shock_gate_short")
    eff_out = Bot._apply_post_sf_cooldown_gate(h, eff_in)
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL


# ---------------------------------------------------------------------------
# Time-based clearing
# ---------------------------------------------------------------------------


def test_deadline_passed_clears_state_and_passes_through():
    """At deadline-or-past, the gate clears state (deadline=0,
    lean_side=None) and passes the eligibility through unchanged.
    Subsequent ticks short-circuit on the deadline<=0 check."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY, cooldown_s=60.0)
    # Advance clock past deadline.
    h._clock.t += 90.0
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert eff_out.eligibility == QuoteEligibility.QUOTE_BOTH
    assert h._state.post_sf_cooldown_until_mono == 0.0
    assert h._state.post_sf_cooldown_lean_side is None


def test_just_before_deadline_still_active():
    """At deadline - epsilon, gate is still active. Sanity boundary.
    v1.5.154: result is HOLD_ALL, not QUOTE_SELL_ONLY as in v1.5.2."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY, cooldown_s=60.0)
    h._clock.t += 59.5
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL


def test_exactly_at_deadline_clears():
    """At deadline (now >= deadline), gate clears -- inclusive
    boundary. Documents the exact semantics so future refactors
    don't accidentally make it exclusive."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY, cooldown_s=60.0)
    h._clock.t += 60.0  # now == deadline
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert eff_out.eligibility == QuoteEligibility.QUOTE_BOTH
    assert h._state.post_sf_cooldown_until_mono == 0.0


# ---------------------------------------------------------------------------
# Fire-count accumulation
# ---------------------------------------------------------------------------


def test_fire_count_bumps_each_active_tick():
    """The counter is session-cumulative number of ticks the gate
    was actively clamping. Three consecutive arming ticks = 3.
    Surfaces in snapshots so the operator can see how often the
    gate is engaging."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY, cooldown_s=60.0)
    for _ in range(3):
        Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert h._state.post_sf_cooldown_fire_count_total == 3


def test_fire_count_does_not_bump_after_deadline():
    """Once the deadline passes, the gate clears state and the
    counter is FROZEN at its pre-deadline value -- no further
    bumps until a new SF arms again."""
    h = _make_harness()
    _arm(h, lean_side=Side.BUY, cooldown_s=60.0)
    Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert h._state.post_sf_cooldown_fire_count_total == 2
    h._clock.t += 90.0
    for _ in range(5):
        Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    # Still 2 (no further bumps after clear).
    assert h._state.post_sf_cooldown_fire_count_total == 2


# ---------------------------------------------------------------------------
# Cooldown of zero seconds == disabled
# ---------------------------------------------------------------------------


def test_zero_cooldown_seconds_armed_but_immediately_expires():
    """When ``POST_SF_COOLDOWN_SECONDS=0``, even if arming somehow
    happened, the deadline equals "now" which the >= check
    interprets as expired. Tests the boundary."""
    h = _make_harness(cooldown_s=0.0)
    # Simulate an SF exit that armed with cooldown_s=0.
    h._state.post_sf_cooldown_until_mono = h._clock.t  # deadline == now
    h._state.post_sf_cooldown_lean_side = Side.BUY
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())
    assert eff_out.eligibility == QuoteEligibility.QUOTE_BOTH
    assert h._state.post_sf_cooldown_until_mono == 0.0


# ---------------------------------------------------------------------------
# v1.5.154 — trend-direction neutrality
#
# These tests pin the *symmetry* property the rewrite was designed
# for: cooldown behaviour is identical regardless of pre-SF direction
# OR upstream eligibility one-sidedness. The old design produced
# asymmetric outcomes that systematically penalised the bot in
# trends; the new design must not.
# ---------------------------------------------------------------------------


def test_long_and_short_pre_sf_produce_identical_eligibility():
    """The defining property of the v1.5.154 rewrite: BOTH pre-SF
    directions yield the same eligibility (HOLD_ALL). The old
    side-suppression design produced opposite eligibilities
    (LONG → SELL_ONLY vs SHORT → BUY_ONLY), creating the
    direction-asymmetric adverse-selection that drove the
    v1.5.150-260525-220850 incident."""
    # LONG branch:
    h_long = _make_harness()
    _arm(h_long, lean_side=Side.BUY)  # pre-SF LONG
    out_long = Bot._apply_post_sf_cooldown_gate(h_long, _eff_q())

    # SHORT branch:
    h_short = _make_harness()
    _arm(h_short, lean_side=Side.SELL)  # pre-SF SHORT
    out_short = Bot._apply_post_sf_cooldown_gate(h_short, _eff_q())

    # Eligibility must be identical (the asymmetry is gone):
    assert out_long.eligibility == out_short.eligibility
    assert out_long.eligibility == QuoteEligibility.HOLD_ALL

    # Both branches bump the same counter and latch active:
    assert h_long._state.post_sf_cooldown_active_last_tick is True
    assert h_short._state.post_sf_cooldown_active_last_tick is True

    # Reason strings differ only in the ``prev=`` direction tag:
    assert "prev=LONG" in out_long.reason
    assert "prev=SHORT" in out_short.reason
    # The HOLD_ALL portion is identical:
    assert "post_sf_cooldown:HOLD_ALL" in out_long.reason
    assert "post_sf_cooldown:HOLD_ALL" in out_short.reason


def test_v1_5_150_260525_220850_reproducer_no_longer_forces_sell_only():
    """Direct reproducer of the snapshot scenario.
    Pre-SF was LONG, market in continuing uptrend. Under v1.5.2 the
    gate forced QUOTE_SELL_ONLY → bot got adversely selected on the
    one side it could quote. Under v1.5.154 the gate forces HOLD_ALL
    → bot pauses; cannot be adversely selected because it's not
    quoting either side."""
    h = _make_harness(cooldown_s=60.0)
    _arm(h, lean_side=Side.BUY)  # pre-SF LONG (reproducer)
    eff_out = Bot._apply_post_sf_cooldown_gate(h, _eff_q())

    # The defining assertions of the fix:
    assert eff_out.eligibility == QuoteEligibility.HOLD_ALL
    # The reason string lets the operator see what happened:
    assert "HOLD_ALL" in eff_out.reason
    assert "prev=LONG" in eff_out.reason
    # Bot is NOT being forced into SELL_ONLY (the pre-fix mis-fire):
    assert eff_out.eligibility != QuoteEligibility.QUOTE_SELL_ONLY
