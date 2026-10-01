"""BUG-033 — adaptive_widen position-favorable exit must engage when
position is flat.

Pre-v1.5.191 behaviour (snapshot v1.5.187-260527-130257-prod.okx.ton.usdt.perp):

* ``adaptive_widen`` fired 10× over 39 min
* 0× cleared via position-favorable
* 9× cleared via time ceiling
* Bot was flat-position 73 % of the session — but the position-
  favorable predicate required ``|pos| >= 1`` to even attempt a
  clear → structurally impossible on most ticks.

The v1.5.191 fix: when ``|pos| < 1e-9`` (effectively flat), clear the
widen immediately because there's no inventory to defend and the
widen is pure fill-rate cost. The drift-aligned-with-inventory branch
below the flat-clear remains intact for the non-flat case.

The actual mutation block lives inline in ``Bot.one_tick``
(app/bot.py:7081+). We test the predicate by replicating the block
in a minimal shell — same approach as
``tests/test_adaptive_spread_widen_no_self_perpetuate.py``.

Per CLAUDE.md: only this test file is run from the assistant;
full-suite verification is the CI daemon's job.
"""

from __future__ import annotations

import math


class _ShellState:
    """Minimal surface of BotState needed for the position-favorable
    clear block."""

    def __init__(self, *, position_qty: float, until_mono: float):
        self.position = _Pos(position_qty)
        self.adaptive_spread_widen_until_mono: float = until_mono
        self.adaptive_spread_widen_favorable_dwell_started_mono: float | None = None
        self.quote_quality_widen_latched: bool = True
        self.adaptive_spread_widen_cleared_via_position_favorable_total: int = 0
        self.adaptive_spread_widen_was_active_last_tick: bool = True


class _Pos:
    def __init__(self, qty: float):
        self.position_qty = qty


class _ShellSettings:
    """The knobs consumed by the clear block."""

    def __init__(self, **overrides):
        self.adaptive_spread_widen_position_favorable_exit_enabled = overrides.get(
            "enabled", True
        )
        self.adaptive_spread_widen_position_favorable_inventory_threshold = overrides.get(
            "inv_threshold", 1.0
        )
        self.adaptive_spread_widen_position_favorable_drift_threshold_bps = overrides.get(
            "drift_threshold", 5.0
        )


def _evaluate_position_favorable_clear(
    state: _ShellState,
    settings: _ShellSettings,
    *,
    drift_10s_bps: float | None,
    widen_active: bool,
) -> bool:
    """Mirror of the v1.5.191 position-favorable clear logic in
    ``Bot.one_tick`` (app/bot.py:7081-7140). Returns the new
    ``widen_active`` value after the clear evaluation.

    Mutates ``state`` to match the bot's side-effects (counter bump,
    deadline reset, etc.) when a clear fires.
    """
    if not (
        widen_active
        and bool(getattr(settings, "adaptive_spread_widen_position_favorable_exit_enabled", True))
    ):
        return widen_active
    pos_qty = float(getattr(state.position, "position_qty", 0.0) or 0.0)
    inv_thresh = float(
        getattr(
            settings,
            "adaptive_spread_widen_position_favorable_inventory_threshold",
            1.0,
        )
    )
    drift_thresh = float(
        getattr(
            settings,
            "adaptive_spread_widen_position_favorable_drift_threshold_bps",
            5.0,
        )
    )
    # v1.5.191 BUG-033 fix — flat-position fast-clear.
    if abs(pos_qty) < 1e-9:
        state.adaptive_spread_widen_until_mono = 0.0
        state.adaptive_spread_widen_favorable_dwell_started_mono = None
        state.quote_quality_widen_latched = False
        state.adaptive_spread_widen_cleared_via_position_favorable_total += 1
        state.adaptive_spread_widen_was_active_last_tick = False
        return False
    elif abs(pos_qty) >= inv_thresh:
        if drift_10s_bps is None or not math.isfinite(float(drift_10s_bps)):
            return widen_active
        sign_pos = math.copysign(1.0, pos_qty) if pos_qty != 0.0 else 0.0
        if sign_pos * float(drift_10s_bps) >= drift_thresh:
            state.adaptive_spread_widen_until_mono = 0.0
            state.adaptive_spread_widen_favorable_dwell_started_mono = None
            state.quote_quality_widen_latched = False
            state.adaptive_spread_widen_cleared_via_position_favorable_total += 1
            state.adaptive_spread_widen_was_active_last_tick = False
            return False
    return widen_active


# ---------------------------------------------------------------------------
# BUG-033 — flat-position fast-clear (the actual fix)
# ---------------------------------------------------------------------------


def test_flat_position_clears_widen_immediately() -> None:
    """The headline fix: with |pos| ≈ 0, the widen should clear on
    this tick regardless of drift. The flat case is unambiguously
    "no inventory to defend, widen is pure fill-rate cost."
    """
    state = _ShellState(position_qty=0.0, until_mono=1000.0)
    settings = _ShellSettings()
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=0.0, widen_active=True
    )
    assert new_active is False
    assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 1
    assert state.adaptive_spread_widen_until_mono == 0.0
    assert state.quote_quality_widen_latched is False
    assert state.adaptive_spread_widen_was_active_last_tick is False


def test_flat_position_clears_regardless_of_drift_direction() -> None:
    """Drift direction is irrelevant when flat — no inventory means
    no inventory-defense rationale."""
    for drift_bps in (-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0):
        state = _ShellState(position_qty=0.0, until_mono=1000.0)
        settings = _ShellSettings()
        new_active = _evaluate_position_favorable_clear(
            state, settings, drift_10s_bps=drift_bps, widen_active=True
        )
        assert new_active is False, f"failed at drift={drift_bps}"
        assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 1


def test_near_flat_position_inside_epsilon_still_clears() -> None:
    """Sub-1e-9 absolute position is treated as flat. The 1e-9
    epsilon matches the rest of the codebase's flat-position guard
    pattern (state.py uses the same in _note_inventory_reduction)."""
    state = _ShellState(position_qty=5e-10, until_mono=1000.0)
    settings = _ShellSettings()
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=0.0, widen_active=True
    )
    assert new_active is False


# ---------------------------------------------------------------------------
# Regression guards — the original (non-flat) drift-aligned path
# must still work.
# ---------------------------------------------------------------------------


def test_long_position_drift_up_clears() -> None:
    """Long + drift positive (≥ 5 bps) → inventory aligned with
    drift → clear via the v1.5.157 drift-aligned predicate."""
    state = _ShellState(position_qty=3.0, until_mono=1000.0)
    settings = _ShellSettings()
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=10.0, widen_active=True
    )
    assert new_active is False
    assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 1


def test_long_position_drift_down_does_not_clear() -> None:
    """Long + drift negative → inventory misaligned with drift
    (bleeding case). The widen stays active — this is the case the
    v1.5.157 predicate was designed for."""
    state = _ShellState(position_qty=3.0, until_mono=1000.0)
    settings = _ShellSettings()
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=-10.0, widen_active=True
    )
    assert new_active is True
    assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 0


def test_short_position_drift_down_clears() -> None:
    """Short + drift negative → inventory aligned with drift →
    clear. Mirror of the long-drift-up case."""
    state = _ShellState(position_qty=-3.0, until_mono=1000.0)
    settings = _ShellSettings()
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=-10.0, widen_active=True
    )
    assert new_active is False
    assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 1


def test_long_position_drift_below_threshold_does_not_clear() -> None:
    """Long + drift positive but below the 5 bps threshold → no
    clear (drift-aligned but not magnitude-sufficient)."""
    state = _ShellState(position_qty=3.0, until_mono=1000.0)
    settings = _ShellSettings()
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=2.0, widen_active=True
    )
    assert new_active is True
    assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 0


# ---------------------------------------------------------------------------
# Feature-flag + edge-case guards
# ---------------------------------------------------------------------------


def test_disabled_feature_flag_blocks_all_clears() -> None:
    """When ``adaptive_spread_widen_position_favorable_exit_enabled``
    is False, no clearance happens — neither flat nor drift-aligned."""
    state = _ShellState(position_qty=0.0, until_mono=1000.0)
    settings = _ShellSettings(enabled=False)
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=0.0, widen_active=True
    )
    assert new_active is True
    assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 0


def test_not_widen_active_short_circuits() -> None:
    """If widen isn't active, the function returns immediately
    without inspecting state. Defensive against re-entrant calls."""
    state = _ShellState(position_qty=0.0, until_mono=0.0)
    settings = _ShellSettings()
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=0.0, widen_active=False
    )
    assert new_active is False
    assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 0


def test_non_finite_drift_with_non_flat_position_does_not_clear() -> None:
    """Drift = NaN / inf with non-flat position → no clear via the
    drift-aligned branch. (The flat branch doesn't consult drift.)"""
    state = _ShellState(position_qty=3.0, until_mono=1000.0)
    settings = _ShellSettings()
    new_active = _evaluate_position_favorable_clear(
        state, settings, drift_10s_bps=float("nan"), widen_active=True
    )
    assert new_active is True
    assert state.adaptive_spread_widen_cleared_via_position_favorable_total == 0
