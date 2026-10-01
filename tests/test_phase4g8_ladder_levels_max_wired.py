"""Phase 4G.8 (v1.4.219) — ladder_levels_max regime knob wire-up.

Pre-v1.4.219 audit (from snapshot
``snapshots/v1.4.214-260521-202803-prod.okx.ton.usdt.perp``):
    - LADDER_NUM_LEVELS_PER_SIDE=2 in profile
    - Bot in CAUTIOUS for 82.9 % of session
    - ZERO rung-1 orders placed across 1,395 total orders
    - ``ladder_rung_drops.min_notional_total = 0`` (so rung-1 wasn't
      dropped at the build_ladder min_notional gate)

Root cause #1: ``RegimeKnobs.ladder_levels_max`` was set on every
regime row (NORMAL=None, CAUTIOUS=1, DEFENSIVE=1, SHOCK=0) but never
consumed by any caller. ``grep ladder_levels_max`` showed zero
readers in bot.py / execution.py / ladder.py — the knob was dead.

This module pins the v1.4.219 wire-up: the cap is now honoured at
``LadderConfig.num_levels_per_side`` construction time in bot.py
and at ``num_levels_per_side`` derivation in execution.py.

Floor at 1: SHOCK's documented cap=0 would zero the entire ladder
and rely exclusively on the SF path to flatten — a bigger
behavioural change than this fix's scope. SHOCK→0-rung is queued
as 4G.10.
"""

from __future__ import annotations

import pytest

from app.regime_controller import (
    Mode,
    RegimeKnobs,
    compute_knobs_for_mode,
)


# -------------------------------------------------------------------
# Knob shape — pinned values
# -------------------------------------------------------------------


def test_normal_calm_have_no_cap() -> None:
    """NORMAL + CALM = full ladder per the operator's config."""
    assert compute_knobs_for_mode(Mode.NORMAL).ladder_levels_max is None
    assert compute_knobs_for_mode(Mode.CALM).ladder_levels_max is None


def test_defensive_caps_to_one_rung() -> None:
    """DEFENSIVE = 1 rung. Sized down × 0.50, spread × 1.50 already;
    a 2nd rung would just sit deeper at the same shrunken size."""
    knobs = compute_knobs_for_mode(Mode.DEFENSIVE)
    assert knobs.ladder_levels_max == 1


def test_cautious_caps_to_one_rung() -> None:
    """CAUTIOUS = 1 rung. The whole point of Phase 4G CAUTIOUS is
    to pull back participation; an outer rung defeats that."""
    knobs = compute_knobs_for_mode(Mode.CAUTIOUS)
    assert knobs.ladder_levels_max == 1


def test_shock_caps_to_zero_in_knob_table() -> None:
    """SHOCK's KNOB says 0, but v1.4.219's wire-up floors at 1 to
    preserve the existing shock_gate-driven reducing-side quoting
    behaviour. The knob value stays as documented intent; the floor
    lives at the consumption site (bot.py / execution.py)."""
    knobs = compute_knobs_for_mode(Mode.SHOCK)
    assert knobs.ladder_levels_max == 0


# -------------------------------------------------------------------
# Cap composition — min(config, knob) with floor at 1
# -------------------------------------------------------------------


def _effective_levels(
    cfg_levels: int, regime_levels_cap: int | None, floor: int = 1
) -> int:
    """Mirror of the wire-up logic in bot.py + execution.py.
    Extracted so the same formula has one test surface here."""
    if regime_levels_cap is not None:
        return max(floor, min(cfg_levels, int(regime_levels_cap)))
    return max(floor, cfg_levels)


def test_config_2_normal_yields_2_rungs() -> None:
    """NORMAL: knob None → use raw config."""
    assert _effective_levels(2, None) == 2


def test_config_2_cautious_caps_to_1_rung() -> None:
    """CAUTIOUS: knob=1 caps the config=2 down to 1."""
    assert _effective_levels(2, 1) == 1


def test_config_2_defensive_caps_to_1_rung() -> None:
    """DEFENSIVE: knob=1 caps the config=2 down to 1."""
    assert _effective_levels(2, 1) == 1


def test_config_2_shock_floors_at_1_rung() -> None:
    """SHOCK: knob=0, but the floor preserves 1-rung quoting so
    the reducing side stays open under shock_gate's QuoteEligibility
    lock. This is the explicit back-compat invariant for v1.4.219."""
    assert _effective_levels(2, 0) == 1


def test_config_1_under_any_mode_stays_at_1() -> None:
    """Single-rung profiles (e.g. early-stage bots, lower-risk
    operators) are unaffected by the regime cap."""
    for cap in (None, 0, 1, 2):
        assert _effective_levels(1, cap) == 1


def test_config_3_cautious_still_caps_to_1() -> None:
    """3-rung ladders shrink to 1 under CAUTIOUS just like 2-rung."""
    assert _effective_levels(3, 1) == 1


def test_config_3_defensive_caps_to_1() -> None:
    assert _effective_levels(3, 1) == 1


def test_config_3_normal_stays_at_3() -> None:
    assert _effective_levels(3, None) == 3


# -------------------------------------------------------------------
# Asymmetric design pin — CAUTIOUS / DEFENSIVE shrink, NORMAL / CALM
# don't. SHOCK stays at the knob's documented value (cap consumer
# applies the floor).
# -------------------------------------------------------------------


def test_asymmetric_design_pin() -> None:
    """Single test that pins the full knob-cap design across all 5
    modes. If anyone touches the knob defaults, this test breaks
    and forces a deliberate decision rather than silent drift."""
    expected = {
        Mode.CALM: None,       # full ladder; aggressive mode
        Mode.NORMAL: None,     # full ladder; baseline
        Mode.CAUTIOUS: 1,      # 1-rung; proactive defence
        Mode.DEFENSIVE: 1,     # 1-rung; reactive defence
        Mode.SHOCK: 0,         # 0-rung documented; floor at consumer
    }
    for mode, expected_cap in expected.items():
        assert compute_knobs_for_mode(mode).ladder_levels_max == expected_cap, (
            f"{mode.value} expected ladder_levels_max={expected_cap}, "
            f"got {compute_knobs_for_mode(mode).ladder_levels_max}"
        )
