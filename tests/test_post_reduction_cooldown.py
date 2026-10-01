"""v1.4.113 Phase 1D — post-reduction re-entry cooldown unit tests.

Pin the cooldown's contract:

* Feature disabled (``post_reduction_cooldown_seconds == 0``) → no
  state mutation on any fill, no eligibility clamp ever.
* Reducing fill arms the cooldown: timestamp + suppressed side set.
  (Pre-v1.5.191 also bumped ``post_reduction_cooldown_fire_count``
  here; v1.5.191 moved that increment to the actual engagement edge
  in ``Bot._apply_eligibility_engine`` per BUG-032. The arming
  state-method now only sets the timestamp + side — the counter
  bumps when the cooldown becomes ACTIVE in a DEFENSIVE/SHOCK
  regime, which is when ``cleared_via_*`` counters bump too.)
* Extending fill (|new| > |prev|) does NOT arm.
* Flat-opening fill (prev=0, new=±X) does NOT arm.
* Suppressed side semantics:
  - prev=+5, SELL 2 → new=+3 (LONG) → suppress BUY (QUOTE_SELL_ONLY)
  - prev=-5, BUY 2 → new=-3 (SHORT) → suppress SELL (QUOTE_BUY_ONLY)
  - prev=+5, SELL 5 → new=0 (flat) → suppress BUY (opposite of fill side)
  - prev=+5, SELL 8 → new=-3 (flipped to SHORT) → suppress BUY
    (new direction's adding side is SELL, but we use post-fill
     direction-based suppression: new < 0 → suppress SELL → QUOTE_BUY_ONLY)
* Behavioural-gates snapshot exposes the cooldown block whether
  dormant or armed (always-render contract).

The full eligibility-clamp interaction with `regime_controller.mode`
lives in the bot's `_apply_regime_gates` block (not directly
testable here without a heavier harness); these tests pin the
*state-arming* contract that those higher-layer guards depend on.
The 06:50:05 replay scenario is verified in the cross-cutting 1F
tests once 1F lands.
"""

from __future__ import annotations

import math

from app.config import Settings
from app.enums import QuoteEligibility, Side
from app.state import BotState


def _bs(*, cd_seconds: float = 60.0) -> BotState:
    """Fresh BotState with the cooldown feature enabled."""
    s = Settings(POST_REDUCTION_COOLDOWN_SECONDS=cd_seconds)
    return BotState(s)


def _note(
    bs: BotState,
    *,
    now_mono: float,
    fill_side: Side,
    prev_qty: float,
    new_qty: float,
) -> None:
    bs._note_inventory_reduction_for_cooldown(
        now_mono=now_mono,
        fill_side=fill_side,
        prev_qty=prev_qty,
        new_qty=new_qty,
    )


# ---------------------------------------------------------------------------
# Disabled / non-reducing cases
# ---------------------------------------------------------------------------


def test_feature_disabled_no_arming() -> None:
    bs = _bs(cd_seconds=0.0)
    _note(bs, now_mono=1000.0, fill_side=Side.SELL, prev_qty=+5.0, new_qty=+3.0)
    assert bs.last_inventory_reduction_at_mono is None
    assert bs.last_inventory_reduction_suppressed_side is None
    assert bs.post_reduction_cooldown_fire_count == 0


def test_extending_fill_does_not_arm() -> None:
    """Position +5 → +7 (BUY of 2) is an EXTENSION, not a reduction."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.BUY, prev_qty=+5.0, new_qty=+7.0)
    assert bs.last_inventory_reduction_at_mono is None
    assert bs.last_inventory_reduction_suppressed_side is None
    assert bs.post_reduction_cooldown_fire_count == 0


def test_opening_from_flat_does_not_arm() -> None:
    """Position 0 → +5 (BUY of 5 from flat) is OPENING, not reducing."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.BUY, prev_qty=0.0, new_qty=+5.0)
    assert bs.last_inventory_reduction_at_mono is None
    assert bs.last_inventory_reduction_suppressed_side is None
    assert bs.post_reduction_cooldown_fire_count == 0


def test_nan_input_silently_ignored() -> None:
    """Defensive: non-finite inputs (corrupt position state) don't
    crash, just skip the arming."""
    bs = _bs()
    _note(
        bs,
        now_mono=1000.0,
        fill_side=Side.SELL,
        prev_qty=float("nan"),
        new_qty=+3.0,
    )
    assert bs.last_inventory_reduction_at_mono is None


# ---------------------------------------------------------------------------
# Reducing-fill cases — suppressed-side semantics
# ---------------------------------------------------------------------------


def test_long_partial_reduction_suppresses_buy() -> None:
    """+5 → +3 (SELL of 2). Post-fill LONG → suppress BUY → QUOTE_SELL_ONLY.

    v1.5.191 BUG-032: ``fire_count`` no longer bumped in the
    state-arming method — only at the cooldown-engagement edge in
    Bot._apply_eligibility_engine. So this state-level test asserts
    ``fire_count == 0`` even though the arming did happen (timestamp +
    suppressed_side set)."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.SELL, prev_qty=+5.0, new_qty=+3.0)
    assert bs.last_inventory_reduction_at_mono == 1000.0
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_SELL_ONLY
    )
    # v1.5.191 BUG-032 — counter moved to engagement edge.
    assert bs.post_reduction_cooldown_fire_count == 0


def test_short_partial_reduction_suppresses_sell() -> None:
    """-5 → -3 (BUY of 2). Post-fill SHORT → suppress SELL → QUOTE_BUY_ONLY."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.BUY, prev_qty=-5.0, new_qty=-3.0)
    assert bs.last_inventory_reduction_at_mono == 1000.0
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_BUY_ONLY
    )


def test_long_closed_to_flat_suppresses_buy() -> None:
    """+5 → 0 (SELL of 5 to flat). Post-fill FLAT → suppress side
    OPPOSITE the fill direction = suppress BUY (don't re-establish
    the LONG we just exited). 06:50:05 fast-flip protection."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.SELL, prev_qty=+5.0, new_qty=0.0)
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_SELL_ONLY
    )


def test_short_closed_to_flat_suppresses_sell() -> None:
    """-5 → 0 (BUY of 5 to flat). Post-fill FLAT → suppress side
    OPPOSITE the fill direction = suppress SELL."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.BUY, prev_qty=-5.0, new_qty=0.0)
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_BUY_ONLY
    )


def test_long_flipped_to_short_suppresses_sell() -> None:
    """+5 → -3 (SELL of 8 through flat). |new|=3 < |prev|=5 →
    qualifies as reduction. Post-fill SHORT → suppress SELL
    (adding to SHORT). Allows BUY to flatten the unintended
    SHORT back to neutral."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.SELL, prev_qty=+5.0, new_qty=-3.0)
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_BUY_ONLY
    )


def test_short_flipped_to_long_suppresses_buy() -> None:
    """-5 → +3 (BUY of 8 through flat). Post-fill LONG → suppress
    BUY (adding to LONG)."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.BUY, prev_qty=-5.0, new_qty=+3.0)
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_SELL_ONLY
    )


# ---------------------------------------------------------------------------
# Repeated arming — re-arms refresh the timestamp + count
# ---------------------------------------------------------------------------


def test_repeated_reduction_re_arms_cooldown() -> None:
    """Two consecutive reducing fills → the LATER timestamp wins (so
    the cooldown extends naturally).

    v1.5.191 BUG-032: ``fire_count`` is no longer bumped here in the
    state-arming path; it only increments when the cooldown actually
    engages in DEFENSIVE/SHOCK at the bot layer. The state-level
    contract is now strictly about timestamp + side."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.SELL, prev_qty=+5.0, new_qty=+3.0)
    _note(bs, now_mono=1010.0, fill_side=Side.SELL, prev_qty=+3.0, new_qty=+1.0)
    assert bs.last_inventory_reduction_at_mono == 1010.0
    # v1.5.191 BUG-032 — counter moved to engagement edge.
    assert bs.post_reduction_cooldown_fire_count == 0
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_SELL_ONLY
    )


def test_arming_then_extending_does_not_clear() -> None:
    """Reducing fill arms; subsequent EXTENDING fill leaves the arming
    intact (the cooldown is still running from the prior reduction)."""
    bs = _bs()
    _note(bs, now_mono=1000.0, fill_side=Side.SELL, prev_qty=+5.0, new_qty=+3.0)
    arm_ts = bs.last_inventory_reduction_at_mono
    _note(bs, now_mono=1010.0, fill_side=Side.BUY, prev_qty=+3.0, new_qty=+5.0)
    # Timestamp unchanged — the extending BUY did not re-arm.
    assert bs.last_inventory_reduction_at_mono == arm_ts
    # v1.5.191 BUG-032 — counter moved to engagement edge.
    assert bs.post_reduction_cooldown_fire_count == 0
    # Suppressed side preserved from the original arming.
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_SELL_ONLY
    )


# ---------------------------------------------------------------------------
# Behavioural-gates snapshot shape
# ---------------------------------------------------------------------------


def test_snapshot_dormant_renders_full_block() -> None:
    """Fresh BotState with cooldown enabled: no fill ever → block
    still present (always-render), active=False, fire_count=0."""
    bs = _bs(cd_seconds=60.0)
    from app.state import _behavioural_gates_snapshot

    snap = _behavioural_gates_snapshot(bs)
    block = snap["post_reduction_cooldown"]
    assert block["enabled"] is True
    assert block["cooldown_seconds"] == 60.0
    assert block["active"] is False
    assert block["seconds_remaining"] == 0.0
    assert block["suppressed_side"] is None
    assert block["fire_count"] == 0


def test_snapshot_armed_state() -> None:
    """After a reducing fill the snapshot reports the suppressed
    side. ``active`` / ``seconds_remaining`` depend on monotonic
    clock; we just assert the structural fields.

    v1.5.191 BUG-032: ``fire_count`` no longer bumps on the
    state-arming path — only on the cooldown-engagement edge in
    bot.py. The snapshot still renders fire_count=0 until that
    engagement edge fires."""
    bs = _bs(cd_seconds=60.0)
    _note(bs, now_mono=1000.0, fill_side=Side.SELL, prev_qty=+5.0, new_qty=+3.0)
    from app.state import _behavioural_gates_snapshot

    snap = _behavioural_gates_snapshot(bs)
    block = snap["post_reduction_cooldown"]
    assert block["enabled"] is True
    assert block["suppressed_side"] == "QUOTE_SELL_ONLY"
    # v1.5.191 BUG-032 — counter moved to engagement edge.
    assert block["fire_count"] == 0


def test_snapshot_includes_phase2k1_exit_attribution_fields() -> None:
    """v1.4.144 Phase 2K.1 — snapshot exposes the new exit-attribution
    counters AND the configured clear_util_pct so the operator can:
      * see whether favorable-exit predicate or MAX-ceiling is the
        binding clearing condition (calibration signal for the knob)
      * confirm the configured threshold from the dashboard without
        opening the env file.
    Fields render at zero on a fresh state (no clearing events yet).
    """
    bs = _bs(cd_seconds=60.0)
    from app.state import _behavioural_gates_snapshot

    snap = _behavioural_gates_snapshot(bs)
    block = snap["post_reduction_cooldown"]
    # New fields present and defaulted.
    assert "clear_util_pct" in block
    assert isinstance(block["clear_util_pct"], float)
    assert 0.0 <= block["clear_util_pct"] <= 1.0
    assert "cleared_via_favorable_total" in block
    assert "cleared_via_ceiling_total" in block
    assert block["cleared_via_favorable_total"] == 0
    assert block["cleared_via_ceiling_total"] == 0


def test_state_counters_increment_independently() -> None:
    """v1.4.144 Phase 2K.1 — the two exit-attribution counters live on
    BotState and increment independently. Verifies they exist with
    the correct names + types so the bot-side eligibility code can
    target them without needing a runtime check."""
    bs = _bs(cd_seconds=60.0)
    # Direct counter mutation — the eligibility code reads + writes
    # these fields. This test pins the field names + initial 0 value.
    assert bs.post_reduction_cooldown_cleared_via_favorable_total == 0
    assert bs.post_reduction_cooldown_cleared_via_ceiling_total == 0
    bs.post_reduction_cooldown_cleared_via_favorable_total += 1
    assert bs.post_reduction_cooldown_cleared_via_favorable_total == 1
    assert bs.post_reduction_cooldown_cleared_via_ceiling_total == 0
    bs.post_reduction_cooldown_cleared_via_ceiling_total += 2
    assert bs.post_reduction_cooldown_cleared_via_ceiling_total == 2


def test_snapshot_disabled_shows_block_with_enabled_false() -> None:
    """Cooldown disabled (seconds=0). Block still rendered with
    enabled=False so the operator can tell at a glance whether
    the gate is off vs. dormant-but-on."""
    bs = _bs(cd_seconds=0.0)
    from app.state import _behavioural_gates_snapshot

    snap = _behavioural_gates_snapshot(bs)
    block = snap["post_reduction_cooldown"]
    assert block["enabled"] is False
    assert block["active"] is False
    assert block["fire_count"] == 0


# ---------------------------------------------------------------------------
# 06:50:05 replay scenario — the exact pattern that motivated 1D
# ---------------------------------------------------------------------------


def test_replay_06_50_05_fast_flip_pattern() -> None:
    """Snapshot 260520-074215 06:50:05 era:
      * Bot LONG +9
      * One SELL fill unwinds to +6 (reducing fill)
      * Within seconds the bot dispatches a BUY that would re-add to LONG
      * Plan 1D: the BUY must be SUPPRESSED for 60 s after the SELL,
        under DEFENSIVE / SHOCK regime mode.

    This test pins the STATE-arming half (which side gets suppressed +
    timestamp). The eligibility-clamp integration under DEFENSIVE is
    covered by the bot-level integration tests in Phase 1F.
    """
    bs = _bs(cd_seconds=60.0)
    _note(bs, now_mono=1000.0, fill_side=Side.SELL, prev_qty=+9.0, new_qty=+6.0)
    # After the SELL reduction:
    #   * Cooldown timestamp recorded
    #   * Suppressed side = QUOTE_SELL_ONLY (suppress BUY → blocks the
    #     re-add-to-LONG pattern the 06:50:05 incident demonstrated)
    #   * Fire count = 1
    assert bs.last_inventory_reduction_at_mono == 1000.0
    assert (
        bs.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_SELL_ONLY
    )
    # v1.5.191 BUG-032 — counter moved to engagement edge.
    assert bs.post_reduction_cooldown_fire_count == 0


# ---------------------------------------------------------------------------
# Sanity: float-noise tolerance
# ---------------------------------------------------------------------------


def test_floating_point_noise_below_tolerance_not_a_reduction() -> None:
    """Position effectively unchanged due to size_step rounding noise
    (|delta| ~ 1e-12) is NOT a reduction — the 1e-9 tolerance in the
    helper guards against spurious arming."""
    bs = _bs()
    _note(
        bs,
        now_mono=1000.0,
        fill_side=Side.SELL,
        prev_qty=+5.000000001,
        new_qty=+5.000000000,
    )
    assert bs.last_inventory_reduction_at_mono is None


def test_tiny_real_reduction_does_arm() -> None:
    """A reduction of size_step magnitude (e.g. 0.001 on TON) ARMS
    the cooldown. The tolerance is sub-tick — well below any real
    fill.

    v1.5.191 BUG-032: arming sets timestamp + side only; fire_count
    moved to engagement edge in bot.py."""
    bs = _bs()
    _note(
        bs,
        now_mono=1000.0,
        fill_side=Side.SELL,
        prev_qty=+5.0,
        new_qty=+4.999,
    )
    assert bs.last_inventory_reduction_at_mono == 1000.0
    # v1.5.191 BUG-032 — counter moved to engagement edge.
    assert bs.post_reduction_cooldown_fire_count == 0
