"""Phase 4G.2 — CALM + CAUTIOUS FSM transitions.

Tests the five-mode FSM extension (CALM + CAUTIOUS added alongside
NORMAL / DEFENSIVE / SHOCK) and the asymmetric dwell rules driven by
the forward-signal classifier.

Transition matrix verified:

| From | To | Trigger | Dwell |
|---|---|---|---|
| NORMAL | CAUTIOUS | forward = CAUTIOUS | 3 s |
| NORMAL | CALM | forward = CALM | 120 s |
| CAUTIOUS | NORMAL | forward = NORMAL or CALM | 60 s |
| CALM | NORMAL | forward = NORMAL | 1 s |
| CALM | CAUTIOUS | forward = CAUTIOUS | direct (no dwell) |
| any | DEFENSIVE | reactive entry (util / vol / drift) | 15 s |
| any | SHOCK | shock_gate_locked | immediate |

Backward-compat coverage (``forward_regime=None``) is in
``tests/test_regime_controller.py`` (75 tests still passing).
"""

from __future__ import annotations

from typing import Optional

import pytest

from app.regime_controller import (
    Mode,
    RegimeControllerState,
    accumulate_time_in_mode,
    compute_knobs_for_mode,
    evaluate_mode,
    snapshot_dict,
)
from app.regime_forward_signals import ForwardRegime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_state(mode: Mode = Mode.NORMAL, now_mono: float = 100.0):
    """Build a state already in ``mode``, with no arming timers
    set. Convenience for the "what does the FSM do from THIS mode?"
    tests."""
    state = RegimeControllerState()
    state.mode = mode
    state.mode_since_mono = now_mono
    return state


def _call(
    state,
    *,
    now_mono: float,
    forward_regime: Optional[ForwardRegime] = None,
    util: float = 0.0,
    vol_ratio: Optional[float] = 1.0,
    slow_trend_active: bool = False,
    inventory_drift_active: bool = False,
    shock_gate_locked: bool = False,
    enabled: bool = True,
    entry_dwell_seconds: float = 15.0,
    exit_dwell_seconds: float = 30.0,
    cautious_entry_dwell_seconds: float = 3.0,
    cautious_exit_dwell_seconds: float = 60.0,
    calm_entry_dwell_seconds: float = 120.0,
    calm_exit_dwell_seconds: float = 1.0,
):
    """Tick wrapper with defaults matching the spec.

    Defaults are deliberately benign (util=0, vol_ratio=1, no
    triggers) — each test changes only what it's exercising."""
    return evaluate_mode(
        state,
        now_mono=now_mono,
        util=util,
        vol_ratio=vol_ratio,
        slow_trend_active=slow_trend_active,
        inventory_drift_active=inventory_drift_active,
        shock_gate_locked=shock_gate_locked,
        enabled=enabled,
        entry_dwell_seconds=entry_dwell_seconds,
        exit_dwell_seconds=exit_dwell_seconds,
        forward_regime=forward_regime,
        cautious_entry_dwell_seconds=cautious_entry_dwell_seconds,
        cautious_exit_dwell_seconds=cautious_exit_dwell_seconds,
        calm_entry_dwell_seconds=calm_entry_dwell_seconds,
        calm_exit_dwell_seconds=calm_exit_dwell_seconds,
    )


# ===========================================================================
# Backward compat — forward_regime=None preserves pre-4G behaviour
# ===========================================================================


def test_no_forward_signal_keeps_normal() -> None:
    """``forward_regime=None`` → FSM doesn't engage the 4G layer; stays
    in NORMAL on benign inputs. The 75 existing regime_controller
    tests would catch any regression in pre-4G behaviour; this is a
    quick smoke check that the new code path doesn't fire when the
    forward signal is absent."""
    state = _fresh_state(Mode.NORMAL)
    mode, reason = _call(state, now_mono=100.0, forward_regime=None)
    assert mode is Mode.NORMAL
    assert reason is None


def test_no_forward_signal_does_not_transition_to_calm_or_cautious() -> None:
    """Without a forward signal, the FSM cannot reach CALM or
    CAUTIOUS regardless of how long we tick. Backward-compat
    guarantee: pre-4G call sites that don't pass forward_regime
    see exactly the old behaviour."""
    state = _fresh_state(Mode.NORMAL)
    # Tick many times across many minutes — no CALM/CAUTIOUS appears.
    for t in (100.0, 250.0, 300.0, 500.0):
        mode, _ = _call(state, now_mono=t, forward_regime=None)
        assert mode is Mode.NORMAL


def test_invalid_forward_regime_treated_as_none() -> None:
    """If a buggy caller passes something that isn't a
    ``ForwardRegime`` enum (e.g. a string), the FSM defensively
    treats it as "no signal" — better than crashing the quote loop."""
    state = _fresh_state(Mode.NORMAL)
    mode, reason = _call(
        state, now_mono=100.0, forward_regime="CAUTIOUS"  # type: ignore[arg-type]
    )
    assert mode is Mode.NORMAL
    assert reason is None


# ===========================================================================
# NORMAL → CAUTIOUS transition (3 s dwell)
# ===========================================================================


def test_normal_to_cautious_arms_on_first_tick() -> None:
    """First tick with forward=CAUTIOUS arms the dwell timer; stays
    in NORMAL until the dwell expires."""
    state = _fresh_state(Mode.NORMAL)
    mode, reason = _call(
        state, now_mono=100.0, forward_regime=ForwardRegime.CAUTIOUS
    )
    assert mode is Mode.NORMAL
    assert reason is None
    assert state.cautious_entry_arming_since_mono == 100.0


def test_normal_to_cautious_fires_after_dwell() -> None:
    """3 s dwell — at 102 s elapsed still NORMAL; at 103 s the
    transition fires."""
    state = _fresh_state(Mode.NORMAL)
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.CAUTIOUS)
    mode, _ = _call(
        state, now_mono=102.0, forward_regime=ForwardRegime.CAUTIOUS
    )
    assert mode is Mode.NORMAL  # still arming
    mode, reason = _call(
        state, now_mono=103.5, forward_regime=ForwardRegime.CAUTIOUS
    )
    assert mode is Mode.CAUTIOUS
    assert reason is not None
    assert "cautious_entry" in reason
    assert "dwell=" in reason


def test_normal_to_cautious_dwell_resets_on_signal_clear() -> None:
    """If the forward signal goes back to NORMAL mid-dwell, the
    arming timer clears. The next CAUTIOUS tick starts a fresh dwell."""
    state = _fresh_state(Mode.NORMAL)
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.CAUTIOUS)
    assert state.cautious_entry_arming_since_mono == 100.0
    # Signal clears → timer resets.
    _call(state, now_mono=101.0, forward_regime=ForwardRegime.NORMAL)
    assert state.cautious_entry_arming_since_mono is None
    # Re-arms on next CAUTIOUS tick.
    _call(state, now_mono=102.0, forward_regime=ForwardRegime.CAUTIOUS)
    assert state.cautious_entry_arming_since_mono == 102.0


# ===========================================================================
# CAUTIOUS → NORMAL transition (60 s exit dwell)
# ===========================================================================


def test_cautious_to_normal_requires_long_dwell() -> None:
    """60 s exit dwell — much longer than entry dwell (asymmetric,
    slow out)."""
    state = _fresh_state(Mode.CAUTIOUS, now_mono=100.0)
    # Forward signal says NORMAL — start arming exit.
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.NORMAL)
    assert state.cautious_exit_arming_since_mono == 100.0
    # 30 s in — still CAUTIOUS.
    mode, _ = _call(
        state, now_mono=130.0, forward_regime=ForwardRegime.NORMAL
    )
    assert mode is Mode.CAUTIOUS
    # 60 s in — exit fires.
    mode, reason = _call(
        state, now_mono=161.0, forward_regime=ForwardRegime.NORMAL
    )
    assert mode is Mode.NORMAL
    assert "cautious_exit" in reason


def test_cautious_exit_also_arms_on_calm_classifier() -> None:
    """forward=CALM also counts as "exit CAUTIOUS" — but we land in
    NORMAL first (not direct CALM); the NORMAL → CALM dwell then
    fires on subsequent ticks if calm persists."""
    state = _fresh_state(Mode.CAUTIOUS, now_mono=100.0)
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.CALM)
    mode, reason = _call(
        state, now_mono=161.0, forward_regime=ForwardRegime.CALM
    )
    assert mode is Mode.NORMAL
    assert "cautious_exit" in reason


def test_cautious_dwell_resets_on_reflare() -> None:
    """If forward signal flares back to CAUTIOUS mid-exit-dwell, the
    arming timer clears — bot stays CAUTIOUS."""
    state = _fresh_state(Mode.CAUTIOUS, now_mono=100.0)
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.NORMAL)
    assert state.cautious_exit_arming_since_mono == 100.0
    # Re-flare.
    _call(state, now_mono=130.0, forward_regime=ForwardRegime.CAUTIOUS)
    assert state.cautious_exit_arming_since_mono is None


# ===========================================================================
# NORMAL → CALM transition (120 s entry dwell — slow in)
# ===========================================================================


def test_normal_to_calm_requires_120s_dwell() -> None:
    state = _fresh_state(Mode.NORMAL)
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.CALM)
    # 60 s in — still NORMAL.
    mode, _ = _call(
        state, now_mono=160.0, forward_regime=ForwardRegime.CALM
    )
    assert mode is Mode.NORMAL
    # 119 s in — still NORMAL.
    mode, _ = _call(
        state, now_mono=219.0, forward_regime=ForwardRegime.CALM
    )
    assert mode is Mode.NORMAL
    # 121 s in — transitions.
    mode, reason = _call(
        state, now_mono=221.0, forward_regime=ForwardRegime.CALM
    )
    assert mode is Mode.CALM
    assert "calm_entry" in reason


def test_normal_to_calm_does_not_arm_on_cautious_signal() -> None:
    """Forward=CAUTIOUS while in NORMAL must NOT arm the CALM-entry
    timer (those are separate dwells; CAUTIOUS path wins anyway)."""
    state = _fresh_state(Mode.NORMAL)
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.CAUTIOUS)
    assert state.calm_entry_arming_since_mono is None


# ===========================================================================
# CALM → NORMAL transition (1 s exit dwell — fast out)
# ===========================================================================


def test_calm_to_normal_with_1s_dwell() -> None:
    """CALM exit is fast — 1 s of confirmed NORMAL flips the FSM
    back. This is the "drop immediately on any rising indicator"
    operator spec."""
    state = _fresh_state(Mode.CALM, now_mono=100.0)
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.NORMAL)
    # 0.5 s in — still CALM.
    mode, _ = _call(
        state, now_mono=100.5, forward_regime=ForwardRegime.NORMAL
    )
    assert mode is Mode.CALM
    # 1.0 s in — exit fires.
    mode, reason = _call(
        state, now_mono=101.0, forward_regime=ForwardRegime.NORMAL
    )
    assert mode is Mode.NORMAL
    assert "calm_exit" in reason


# ===========================================================================
# CALM → CAUTIOUS direct (no dwell — fastest escape)
# ===========================================================================


def test_calm_to_cautious_direct() -> None:
    """forward=CAUTIOUS while in CALM → immediate transition to
    CAUTIOUS (skip NORMAL). Asymmetric "fast out of CALM on rising
    risk" — the headline safety property of the asymmetric dwell."""
    state = _fresh_state(Mode.CALM, now_mono=100.0)
    mode, reason = _call(
        state, now_mono=100.5, forward_regime=ForwardRegime.CAUTIOUS
    )
    assert mode is Mode.CAUTIOUS
    assert "calm_to_cautious_direct" in reason


def test_calm_to_cautious_direct_no_arming_required() -> None:
    """No arming timer is set or checked — the very first
    CAUTIOUS classification while in CALM fires the transition."""
    state = _fresh_state(Mode.CALM, now_mono=500.0)
    # No prior calls — state has no arming timers set.
    assert state.cautious_entry_arming_since_mono is None
    mode, reason = _call(
        state, now_mono=500.001, forward_regime=ForwardRegime.CAUTIOUS
    )
    assert mode is Mode.CAUTIOUS
    assert "calm_to_cautious_direct" in reason


# ===========================================================================
# Reactive DEFENSIVE / SHOCK preempts the forward-signal layer
# ===========================================================================


def test_shock_preempts_calm() -> None:
    """SHOCK gate locking while in CALM → instant SHOCK. The forward
    signal layer is irrelevant in a real shock event."""
    state = _fresh_state(Mode.CALM, now_mono=100.0)
    mode, reason = _call(
        state,
        now_mono=100.0,
        forward_regime=ForwardRegime.CALM,
        shock_gate_locked=True,
    )
    assert mode is Mode.SHOCK
    assert reason == "shock_gate_locked"


def test_shock_preempts_cautious() -> None:
    """SHOCK preempts CAUTIOUS the same way it preempts CALM."""
    state = _fresh_state(Mode.CAUTIOUS, now_mono=100.0)
    mode, reason = _call(
        state,
        now_mono=100.0,
        forward_regime=ForwardRegime.CAUTIOUS,
        shock_gate_locked=True,
    )
    assert mode is Mode.SHOCK


def test_defensive_entry_preempts_forward_signal_from_calm() -> None:
    """If reactive DEFENSIVE entry conditions hold (e.g. util ≥ 0.5)
    while in CALM, the reactive path wins. We arm DEFENSIVE entry
    AND do NOT process the forward-signal-driven transition this
    tick. After dwell expires, CALM → DEFENSIVE directly."""
    state = _fresh_state(Mode.CALM, now_mono=100.0)
    # util=0.6 ≥ default entry threshold 0.5 → reactive arming fires.
    mode, _ = _call(
        state,
        now_mono=100.0,
        forward_regime=ForwardRegime.CALM,  # ignored
        util=0.6,
    )
    assert mode is Mode.CALM
    assert state.entry_arming_since_mono == 100.0
    # 16 s later (entry dwell = 15 s) — DEFENSIVE fires.
    mode, reason = _call(
        state,
        now_mono=116.0,
        forward_regime=ForwardRegime.CALM,
        util=0.6,
    )
    assert mode is Mode.DEFENSIVE
    assert "defensive_entry" in reason


def test_defensive_entry_from_cautious() -> None:
    """Same path: reactive DEFENSIVE entry from CAUTIOUS."""
    state = _fresh_state(Mode.CAUTIOUS, now_mono=100.0)
    _call(
        state,
        now_mono=100.0,
        forward_regime=ForwardRegime.CAUTIOUS,
        vol_ratio=3.0,  # above 2.0 default entry threshold
    )
    mode, _ = _call(
        state,
        now_mono=120.0,
        forward_regime=ForwardRegime.CAUTIOUS,
        vol_ratio=3.0,
    )
    assert mode is Mode.DEFENSIVE


# ===========================================================================
# Disabled flag forces NORMAL regardless of mode or forward signal
# ===========================================================================


def test_disabled_forces_normal_from_calm() -> None:
    state = _fresh_state(Mode.CALM, now_mono=100.0)
    mode, reason = _call(
        state,
        now_mono=100.0,
        enabled=False,
        forward_regime=ForwardRegime.CALM,
    )
    assert mode is Mode.NORMAL
    assert reason == "disabled"


def test_disabled_forces_normal_from_cautious() -> None:
    state = _fresh_state(Mode.CAUTIOUS, now_mono=100.0)
    mode, reason = _call(
        state,
        now_mono=100.0,
        enabled=False,
        forward_regime=ForwardRegime.CAUTIOUS,
    )
    assert mode is Mode.NORMAL


# ===========================================================================
# Time-in-mode accumulation for new modes
# ===========================================================================


def test_time_in_calm_accumulates() -> None:
    state = _fresh_state(Mode.CALM, now_mono=100.0)
    accumulate_time_in_mode(state, now_mono=100.0)  # seeds
    accumulate_time_in_mode(state, now_mono=130.0)  # +30 s in CALM
    assert state.time_in_calm_seconds == pytest.approx(30.0)
    assert state.time_in_normal_seconds == 0.0


def test_time_in_cautious_accumulates() -> None:
    state = _fresh_state(Mode.CAUTIOUS, now_mono=100.0)
    accumulate_time_in_mode(state, now_mono=100.0)
    accumulate_time_in_mode(state, now_mono=145.0)
    assert state.time_in_cautious_seconds == pytest.approx(45.0)


# ===========================================================================
# Knob tables — CALM more aggressive, CAUTIOUS more defensive
# ===========================================================================


def test_calm_knobs_are_more_aggressive_than_normal() -> None:
    """CALM should have tighter spread, larger size, full ladder."""
    calm = compute_knobs_for_mode(Mode.CALM)
    normal = compute_knobs_for_mode(Mode.NORMAL)
    assert calm.base_half_spread_mult < normal.base_half_spread_mult
    assert calm.quote_notional_mult > normal.quote_notional_mult
    assert calm.inventory_budget_mult == normal.inventory_budget_mult
    assert calm.adding_side_enabled is True


def test_cautious_knobs_are_between_normal_and_defensive() -> None:
    """CAUTIOUS sits between NORMAL and DEFENSIVE on the load-bearing
    axes (spread, inventory budget, ladder depth).

    v1.5.231 (2026-05-29) — the notional axis was intentionally LIFTED
    OUT of CAUTIOUS's defensive stack: `quote_notional_mult` is now
    1.00 (same as NORMAL) instead of 0.70. Reason: on TON with
    `QUOTE_NOTIONAL_USD=$7` and `MIN_QUOTE_NOTIONAL_USD=$5`, the old
    0.70 multiplier collapsed effective notional to $4.90 — below the
    min-notional floor — and dropped every rung whenever CAUTIOUS
    engaged. v1.5.230-260529-094214 snapshot recorded 0 fills in
    27 min because of this. The defensive intent of CAUTIOUS is now
    carried by spread widening (1.30×) + ladder-depth cap (1 rung)
    alone, both of which remain strictly between NORMAL and DEFENSIVE.
    """
    cautious = compute_knobs_for_mode(Mode.CAUTIOUS)
    normal = compute_knobs_for_mode(Mode.NORMAL)
    defensive = compute_knobs_for_mode(Mode.DEFENSIVE)
    # Spread widens (CAUTIOUS strictly between normal and defensive).
    assert normal.base_half_spread_mult < cautious.base_half_spread_mult < defensive.base_half_spread_mult
    # Notional axis: CAUTIOUS no longer shrinks (v1.5.231). DEFENSIVE
    # still shrinks. Invariant is now <= on the normal side, < on the
    # defensive side.
    assert defensive.quote_notional_mult < cautious.quote_notional_mult <= normal.quote_notional_mult
    assert cautious.quote_notional_mult == pytest.approx(1.0)  # v1.5.231
    # Inventory budget shrinks — 4G adds this axis. Still load-bearing
    # for CAUTIOUS (0.70×) — unchanged in v1.5.231.
    assert cautious.inventory_budget_mult < normal.inventory_budget_mult
    assert cautious.ladder_levels_max == 1


def test_inventory_budget_mult_default_is_one() -> None:
    """NORMAL and CALM leave the inventory budget at 1.0 — no
    overlay. Only CAUTIOUS (0.7), DEFENSIVE (1.0 - kept-default),
    and SHOCK (0.5) downsize."""
    assert compute_knobs_for_mode(Mode.NORMAL).inventory_budget_mult == 1.0
    assert compute_knobs_for_mode(Mode.CALM).inventory_budget_mult == 1.0


def test_shock_knobs_disable_adding_side() -> None:
    """SHOCK still has adding_side_enabled=False — unchanged from
    pre-4G."""
    shock = compute_knobs_for_mode(Mode.SHOCK)
    assert shock.adding_side_enabled is False
    assert shock.ladder_levels_max == 0


# ===========================================================================
# Snapshot dict exposes new fields
# ===========================================================================


def test_snapshot_dict_includes_new_time_in_mode_counters() -> None:
    state = _fresh_state(Mode.CALM, now_mono=100.0)
    state.time_in_calm_seconds = 42.5
    state.time_in_cautious_seconds = 17.3
    snap = snapshot_dict(state, now_mono=200.0)
    assert snap["time_in_calm_seconds"] == 42.5
    assert snap["time_in_cautious_seconds"] == 17.3


def test_snapshot_dict_includes_new_arming_timers() -> None:
    state = _fresh_state(Mode.NORMAL, now_mono=100.0)
    state.cautious_entry_arming_since_mono = 198.0
    state.calm_entry_arming_since_mono = 150.0
    snap = snapshot_dict(state, now_mono=200.0)
    assert snap["cautious_entry_arming_seconds"] == pytest.approx(2.0)
    assert snap["calm_entry_arming_seconds"] == pytest.approx(50.0)
    # Inactive timers render as None.
    assert snap["cautious_exit_arming_seconds"] is None
    assert snap["calm_exit_arming_seconds"] is None


# ===========================================================================
# Transition log includes the new mode names
# ===========================================================================


def test_transition_log_records_calm_and_cautious_strings() -> None:
    """Recent-transitions log uses Mode.value strings — the new
    modes appear as 'CALM' / 'CAUTIOUS' so the dashboard / postmortem
    render the same way as NORMAL / DEFENSIVE / SHOCK."""
    state = _fresh_state(Mode.NORMAL)
    # NORMAL → CAUTIOUS
    _call(state, now_mono=100.0, forward_regime=ForwardRegime.CAUTIOUS)
    _call(state, now_mono=104.0, forward_regime=ForwardRegime.CAUTIOUS)
    assert state.transition_count == 1
    last = state.last_transitions[-1]
    assert last[0] == "NORMAL"
    assert last[1] == "CAUTIOUS"


# ===========================================================================
# Regression replay — v1.4.200 SF#11175 with FSM wired
# ===========================================================================


def test_regression_v1_4_200_fsm_would_have_engaged_cautious() -> None:
    """The v1.4.200 SF#11175 incident — bot was in NORMAL for 41 min
    while vol climbed, then SF fired straight from NORMAL with no
    upstream defensive activity.

    With Phase 4G wired, the bot would have:

      1. Stayed in NORMAL during the 41-min stable period (vol_bps
         in CALM zone but history not long enough yet → NORMAL).
      2. Transitioned to CAUTIOUS within 3 s once the vol slope
         crossed the classifier's threshold.
      3. Quoted at CAUTIOUS multipliers (1.3 spread × 0.7 size,
         1-level ladder, 0.7 inventory budget) BEFORE position
         drawdown could build to 35 bps.

    This test replays steps 1-2 (the FSM transitions); 4G.3-4G.6
    wire the multipliers into compute_quote_decision so the
    behaviour reaches the order placements.
    """
    state = _fresh_state(Mode.NORMAL, now_mono=0.0)
    # 41 min stable — forward signal returns NORMAL (vol present
    # but not slope-rising; not all-calm enough for CALM).
    for t in range(0, 2460, 30):  # 41 min, every 30 s
        mode, _ = _call(
            state, now_mono=float(t), forward_regime=ForwardRegime.NORMAL
        )
    assert state.mode is Mode.NORMAL

    # Now the vol-slope spike begins. Forward signal returns CAUTIOUS
    # for 3+ consecutive seconds (the entry dwell window). At 2463 s
    # the FSM is still arming; at 2464 s it transitions.
    _call(state, now_mono=2460.0, forward_regime=ForwardRegime.CAUTIOUS)
    assert state.mode is Mode.NORMAL
    assert state.cautious_entry_arming_since_mono == 2460.0
    mode, reason = _call(
        state, now_mono=2464.0, forward_regime=ForwardRegime.CAUTIOUS
    )
    assert mode is Mode.CAUTIOUS, (
        f"FSM should have transitioned to CAUTIOUS within 3 s of the "
        f"classifier's first CAUTIOUS read; got {mode.value}"
    )
    assert "cautious_entry" in reason


# ===========================================================================
# Phase 4G.7 (v1.4.217) — wall-clock ts_iso on recent_transitions
# ===========================================================================


def test_snapshot_dict_recent_transitions_include_wall_clock_ts_iso() -> None:
    """v1.4.217 — each transition in the snapshot dict carries a
    wall-clock ISO 8601 timestamp derived from ``ts_mono`` and the
    snapshot's now-wall reference. The dashboard regime band plots
    transitions on the wall-clock X-axis using this field; the
    monotonic ``ts_mono`` alone wouldn't suffice because the frontend
    doesn't have the bot's monotonic origin."""
    state = _fresh_state(Mode.NORMAL, now_mono=1000.0)
    # Drive one transition by feeding CAUTIOUS for the dwell window.
    _call(state, now_mono=1000.0, forward_regime=ForwardRegime.CAUTIOUS)
    _call(state, now_mono=1004.0, forward_regime=ForwardRegime.CAUTIOUS)
    assert state.mode is Mode.CAUTIOUS
    snap = snapshot_dict(state, now_mono=1010.0)
    transitions = snap["recent_transitions"]
    assert isinstance(transitions, list) and len(transitions) >= 1
    t = transitions[-1]
    # New v1.4.217 keys.
    assert "ts_iso" in t, "snapshot transitions must carry wall-clock ts_iso"
    assert isinstance(t["ts_iso"], str)
    assert t["ts_iso"].endswith("+00:00") or "Z" in t["ts_iso"], (
        "ts_iso should be a timezone-aware UTC ISO 8601 string"
    )
    # ts_iso must be roughly "(snapshot time) - (now_mono - ts_mono)" s
    # in the past. Since now_mono=1010 and the transition fired at
    # ts_mono≈1004, ts_iso should be ~6 s before now_wall.
    from datetime import datetime, timezone
    parsed = datetime.fromisoformat(t["ts_iso"])
    now_wall = datetime.now(timezone.utc)
    # Allow generous slack — this is verifying the conversion direction,
    # not microsecond precision.
    delta_seconds = (now_wall - parsed).total_seconds()
    # ts_mono = 1004; now_mono = 1010 → ts_iso should be ~6 s in the
    # past relative to snapshot wall time. Wall ≈ snapshot wall, so the
    # delta we observe should be ≥ ~6 s minus test-execution drift.
    assert 5.0 <= delta_seconds <= 30.0, (
        f"ts_iso delta = {delta_seconds:.2f}s — expected ~6 s based on "
        f"the mono offset (1010 - 1004 = 6)"
    )
    # Other expected keys still present (back-compat).
    assert "from" in t
    assert "to" in t
    assert "ts_mono" in t
    assert "reason" in t
