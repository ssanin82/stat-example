"""v1.4.112 Phase 1C — regime_controller FSM unit tests.

Pin the controller's contract:

* Pure NORMAL session → never leaves NORMAL.
* Slow-grind scenario → NORMAL → DEFENSIVE after the 15 s entry
  dwell; stays in DEFENSIVE while util / drift signals persist;
  returns to NORMAL after the 30 s exit dwell.
* Shock scenario → NORMAL → SHOCK directly (no dwell); SHOCK →
  DEFENSIVE on lock-clear; DEFENSIVE → NORMAL after exit dwell.
  No direct SHOCK → NORMAL transition.
* Hysteresis: brief blip out of the entry condition resets the
  arming timer; the controller does NOT flip on a transient that
  doesn't satisfy dwell.
* Knob table: NORMAL / DEFENSIVE / SHOCK rows match the plan's
  spec (base_half_spread_mult, quote_notional_mult, etc).
* Time-in-mode accumulator updates the correct counter per tick
  and is robust to clock jumps / suspended-VM gaps.
* Disabled flag forces NORMAL even when triggers are hot.
* Telemetry snapshot dict shape: every expected field present
  (renders unconditionally per the always-render contract).
"""

from __future__ import annotations

import math

from app.regime_controller import (
    Mode,
    RegimeControllerState,
    RegimeKnobs,
    accumulate_time_in_mode,
    compute_knobs_for_mode,
    evaluate_mode,
    snapshot_dict,
)


def _tick_normal(state: RegimeControllerState, now_mono: float) -> Mode:
    """All-clear inputs — exercises the NORMAL-stays-NORMAL path."""
    mode, _ = evaluate_mode(
        state,
        now_mono=now_mono,
        util=0.1,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
    )
    return mode


def _tick_defensive_entry(
    state: RegimeControllerState, now_mono: float, **overrides
) -> Mode:
    """Default: util high enough to entry-arm DEFENSIVE."""
    kwargs: dict = dict(
        now_mono=now_mono,
        util=0.7,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
    )
    kwargs.update(overrides)
    mode, _ = evaluate_mode(state, **kwargs)
    return mode


# ---------------------------------------------------------------------------
# Disabled / dormant cases
# ---------------------------------------------------------------------------


def test_disabled_forces_normal() -> None:
    state = RegimeControllerState()
    mode, reason = evaluate_mode(
        state,
        now_mono=1000.0,
        util=0.9,
        vol_ratio=5.0,
        slow_trend_active=True,
        inventory_drift_active=True,
        shock_gate_locked=True,
        enabled=False,
    )
    assert mode is Mode.NORMAL
    assert reason is None  # first-tick / unchanged-mode → no transition log
    assert state.mode is Mode.NORMAL


def test_disabled_returns_to_normal_from_defensive() -> None:
    """If the operator disables the FSM mid-session while it's in
    DEFENSIVE, the controller force-transitions back to NORMAL."""
    state = RegimeControllerState()
    state.mode = Mode.DEFENSIVE
    mode, reason = evaluate_mode(
        state,
        now_mono=1000.0,
        util=0.9,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=False,
    )
    assert mode is Mode.NORMAL
    assert reason == "disabled"
    assert state.transition_count == 1


# ---------------------------------------------------------------------------
# Pure NORMAL — never transitions when triggers stay clear.
# ---------------------------------------------------------------------------


def test_pure_normal_stays_normal() -> None:
    state = RegimeControllerState()
    for t in range(0, 600, 5):  # 10 minutes of clear ticks
        assert _tick_normal(state, float(t)) is Mode.NORMAL
    assert state.mode is Mode.NORMAL
    assert state.transition_count == 0
    assert state.entry_arming_since_mono is None


# ---------------------------------------------------------------------------
# NORMAL → DEFENSIVE entry dwell hysteresis
# ---------------------------------------------------------------------------


def test_defensive_entry_after_dwell() -> None:
    """Util above threshold for 15 s → mode flips."""
    state = RegimeControllerState()
    # 0–14 s: util high but dwell not satisfied.
    for t in range(0, 15):
        m = _tick_defensive_entry(state, float(t))
        assert m is Mode.NORMAL, f"flipped at t={t}"
    # At t=15 s the dwell window is met — transition fires.
    m = _tick_defensive_entry(state, 15.0)
    assert m is Mode.DEFENSIVE
    assert state.transition_count == 1
    assert "defensive_entry" in state.last_transition_reason
    # ``util=0.700`` should appear in the reason text.
    assert "util=0.700" in state.last_transition_reason


def test_defensive_entry_blip_resets_arming() -> None:
    """Brief blip out of entry condition resets the arming timer —
    the FSM does NOT flip on a transient."""
    state = RegimeControllerState()
    # 0–10 s: util high (arming).
    for t in range(0, 11):
        _tick_defensive_entry(state, float(t))
    assert state.mode is Mode.NORMAL
    assert state.entry_arming_since_mono is not None
    # t=11 s: util drops below entry threshold → arming clears.
    _tick_defensive_entry(state, 11.0, util=0.3)
    assert state.entry_arming_since_mono is None
    # 12–24 s: util high again, but only 13 s elapsed since RE-arming.
    for t in range(12, 25):
        m = _tick_defensive_entry(state, float(t))
        assert m is Mode.NORMAL, f"flipped at t={t}"
    assert state.transition_count == 0


def test_defensive_entry_via_slow_trend_only() -> None:
    """Slow-trend signal alone (no util) can entry-arm DEFENSIVE."""
    state = RegimeControllerState()
    for t in range(0, 15):
        m = _tick_defensive_entry(
            state, float(t), util=0.1, slow_trend_active=True
        )
        assert m is Mode.NORMAL
    m = _tick_defensive_entry(
        state, 15.0, util=0.1, slow_trend_active=True
    )
    assert m is Mode.DEFENSIVE
    assert "slow_trend" in state.last_transition_reason


def test_defensive_entry_via_vol_ratio() -> None:
    """Vol ratio above threshold alone can entry-arm DEFENSIVE."""
    state = RegimeControllerState()
    for t in range(0, 15):
        m = _tick_defensive_entry(
            state, float(t), util=0.1, vol_ratio=3.0
        )
        assert m is Mode.NORMAL
    m = _tick_defensive_entry(state, 15.0, util=0.1, vol_ratio=3.0)
    assert m is Mode.DEFENSIVE
    assert "vol_ratio" in state.last_transition_reason


def test_defensive_entry_via_inventory_drift() -> None:
    state = RegimeControllerState()
    for t in range(0, 15):
        m = _tick_defensive_entry(
            state, float(t), util=0.1, inventory_drift_active=True
        )
        assert m is Mode.NORMAL
    m = _tick_defensive_entry(
        state, 15.0, util=0.1, inventory_drift_active=True
    )
    assert m is Mode.DEFENSIVE
    assert "inv_drift" in state.last_transition_reason


# ---------------------------------------------------------------------------
# DEFENSIVE → NORMAL exit dwell hysteresis (30 s)
# ---------------------------------------------------------------------------


def test_defensive_exit_after_long_dwell() -> None:
    """Once in DEFENSIVE, util drops below exit threshold AND all
    triggers clear → 30 s exit dwell then back to NORMAL."""
    state = RegimeControllerState()
    state.mode = Mode.DEFENSIVE
    state.mode_since_mono = 0.0
    # 0–29 s: exit condition met but dwell not satisfied.
    for t in range(0, 30):
        m, _ = evaluate_mode(
            state,
            now_mono=float(t),
            util=0.1,
            vol_ratio=1.0,
            slow_trend_active=False,
            inventory_drift_active=False,
            shock_gate_locked=False,
            enabled=True,
        )
        assert m is Mode.DEFENSIVE
    m, _ = evaluate_mode(
        state,
        now_mono=30.0,
        util=0.1,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
    )
    assert m is Mode.NORMAL
    assert state.transition_count == 1
    assert "defensive_exit" in state.last_transition_reason


def test_defensive_exit_blip_resets_arming() -> None:
    """Brief re-trigger of an entry condition during exit-arming
    resets the timer — the FSM does NOT exit on a transient clear."""
    state = RegimeControllerState()
    state.mode = Mode.DEFENSIVE
    # 0–20 s: exit-armed.
    for t in range(0, 21):
        evaluate_mode(
            state,
            now_mono=float(t),
            util=0.1,
            vol_ratio=1.0,
            slow_trend_active=False,
            inventory_drift_active=False,
            shock_gate_locked=False,
            enabled=True,
        )
    assert state.exit_arming_since_mono is not None
    # t=21 s: a slow-trend re-fire resets exit arming.
    evaluate_mode(
        state,
        now_mono=21.0,
        util=0.1,
        vol_ratio=1.0,
        slow_trend_active=True,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
    )
    assert state.exit_arming_since_mono is None
    assert state.mode is Mode.DEFENSIVE


def test_defensive_exit_requires_all_triggers_clear() -> None:
    """If util drops below threshold but a defensive trigger is
    still hot, exit-arming doesn't start."""
    state = RegimeControllerState()
    state.mode = Mode.DEFENSIVE
    for t in range(0, 40):
        evaluate_mode(
            state,
            now_mono=float(t),
            util=0.1,  # below exit threshold
            vol_ratio=1.0,
            slow_trend_active=True,  # but slow_trend still hot
            inventory_drift_active=False,
            shock_gate_locked=False,
            enabled=True,
        )
    assert state.mode is Mode.DEFENSIVE
    assert state.exit_arming_since_mono is None


# ---------------------------------------------------------------------------
# SHOCK preemption + recovery via DEFENSIVE
# ---------------------------------------------------------------------------


def test_normal_to_shock_instant_no_dwell() -> None:
    state = RegimeControllerState()
    m, reason = evaluate_mode(
        state,
        now_mono=1000.0,
        util=0.1,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=True,
        enabled=True,
    )
    assert m is Mode.SHOCK
    assert reason == "shock_gate_locked"
    assert state.transition_count == 1


def test_defensive_to_shock_instant() -> None:
    state = RegimeControllerState()
    state.mode = Mode.DEFENSIVE
    m, reason = evaluate_mode(
        state,
        now_mono=1000.0,
        util=0.9,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=True,
        enabled=True,
    )
    assert m is Mode.SHOCK
    assert reason == "shock_gate_locked"


def test_shock_to_defensive_on_lock_clear() -> None:
    """SHOCK → DEFENSIVE the moment shock_gate clears its lock.
    There is no direct SHOCK → NORMAL — the exit-dwell guard must
    apply via DEFENSIVE."""
    state = RegimeControllerState()
    state.mode = Mode.SHOCK
    m, reason = evaluate_mode(
        state,
        now_mono=1000.0,
        util=0.1,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
    )
    assert m is Mode.DEFENSIVE
    assert reason == "shock_gate_cleared"


def test_shock_does_not_skip_to_normal_directly() -> None:
    """Even when conditions for NORMAL are met, SHOCK transits to
    DEFENSIVE first. The exit-dwell on DEFENSIVE then governs the
    return to NORMAL."""
    state = RegimeControllerState()
    state.mode = Mode.SHOCK
    # First tick: SHOCK → DEFENSIVE.
    m1, _ = evaluate_mode(
        state,
        now_mono=0.0,
        util=0.0,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
    )
    assert m1 is Mode.DEFENSIVE
    # Walk forward through ticks 1..30. The exit-arming timer starts
    # on the FIRST tick after the SHOCK → DEFENSIVE transition (t=1)
    # since the SHOCK transition itself clears arming. Exit dwell
    # of 30 s is satisfied at t=31, not t=30.
    for t in range(1, 31):
        m, _ = evaluate_mode(
            state,
            now_mono=float(t),
            util=0.0,
            vol_ratio=1.0,
            slow_trend_active=False,
            inventory_drift_active=False,
            shock_gate_locked=False,
            enabled=True,
        )
        assert m is Mode.DEFENSIVE, f"flipped at t={t}"
    # At t=31 s the exit dwell is met.
    m, _ = evaluate_mode(
        state,
        now_mono=31.0,
        util=0.0,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
    )
    assert m is Mode.NORMAL


# ---------------------------------------------------------------------------
# Knob table
# ---------------------------------------------------------------------------


def test_knobs_normal() -> None:
    k = compute_knobs_for_mode(Mode.NORMAL, ladder_levels_max_config=3)
    assert isinstance(k, RegimeKnobs)
    assert k.quote_notional_mult == 1.0
    assert k.inventory_skew_mult == 1.0
    assert k.base_half_spread_mult == 1.0
    assert k.extra_tick_adding_side == 0
    assert k.adding_side_enabled is True
    assert k.ladder_levels_max is None  # NORMAL leaves ladder unmodified


def test_knobs_defensive() -> None:
    k = compute_knobs_for_mode(Mode.DEFENSIVE, ladder_levels_max_config=3)
    assert k.quote_notional_mult == 0.5
    assert k.inventory_skew_mult == 1.5
    assert k.base_half_spread_mult == 1.5
    assert k.extra_tick_adding_side == 1
    assert k.adding_side_enabled is True
    assert k.ladder_levels_max == 1  # min(config 3, knob 1)


def test_knobs_defensive_respects_config_lower_than_knob() -> None:
    """If config says 1 level/side, DEFENSIVE doesn't push UP to its
    own value — min() applies."""
    k = compute_knobs_for_mode(Mode.DEFENSIVE, ladder_levels_max_config=1)
    assert k.ladder_levels_max == 1


def test_knobs_shock() -> None:
    """SHOCK: half-spread widens 2×, size stays full (shock_gate's
    binary clamp handles the adding-side suppression), ladder off."""
    k = compute_knobs_for_mode(Mode.SHOCK, ladder_levels_max_config=3)
    assert k.quote_notional_mult == 1.0  # NOT 0.0 — see module docstring
    assert k.base_half_spread_mult == 2.0
    assert k.adding_side_enabled is False
    assert k.ladder_levels_max == 0


# ---------------------------------------------------------------------------
# Time-in-mode accumulator
# ---------------------------------------------------------------------------


def test_accumulate_time_in_mode_seeds_then_adds() -> None:
    state = RegimeControllerState()
    # First call seeds the last-tick timestamp without adding.
    accumulate_time_in_mode(state, now_mono=1000.0)
    assert state.time_in_normal_seconds == 0.0
    # Second call adds the delta.
    accumulate_time_in_mode(state, now_mono=1005.0)
    assert math.isclose(state.time_in_normal_seconds, 5.0)


def test_accumulate_routes_to_current_mode() -> None:
    state = RegimeControllerState()
    accumulate_time_in_mode(state, now_mono=0.0)  # seed
    accumulate_time_in_mode(state, now_mono=10.0)
    assert math.isclose(state.time_in_normal_seconds, 10.0)
    state.mode = Mode.DEFENSIVE
    accumulate_time_in_mode(state, now_mono=15.0)
    assert math.isclose(state.time_in_defensive_seconds, 5.0)
    state.mode = Mode.SHOCK
    accumulate_time_in_mode(state, now_mono=18.0)
    assert math.isclose(state.time_in_shock_seconds, 3.0)


def test_accumulate_caps_absurd_delta() -> None:
    """Process pause / suspended VM with a 3600 s gap shouldn't add
    3600 s to the counter — cap at 600 s.

    v1.4.219 raised the cap from 60 s → 600 s. The old 60 s cap was
    eating real SHOCK time (a 90 s SHOCK tick gap was losing 30 s
    of accounted time — snapshot v1.4.214 showed this as the
    SHOCK-counter shortfall). 600 s catches genuine process-pause
    poisoning while preserving accuracy across slower-cadence ticks
    (CALM mode at the 120 s entry dwell, for instance)."""
    state = RegimeControllerState()
    accumulate_time_in_mode(state, now_mono=0.0)
    accumulate_time_in_mode(state, now_mono=3600.0)
    assert state.time_in_normal_seconds == 600.0


def test_accumulate_resets_on_clock_backwards() -> None:
    """A clock that goes backwards (impossible with monotonic, but
    handle defensively) resets the last-tick without adding."""
    state = RegimeControllerState()
    accumulate_time_in_mode(state, now_mono=1000.0)
    accumulate_time_in_mode(state, now_mono=900.0)
    assert state.time_in_normal_seconds == 0.0
    # Subsequent tick from the new baseline works normally.
    accumulate_time_in_mode(state, now_mono=905.0)
    assert math.isclose(state.time_in_normal_seconds, 5.0)


# ---------------------------------------------------------------------------
# Snapshot dict shape
# ---------------------------------------------------------------------------


def test_snapshot_dict_fresh_state() -> None:
    state = RegimeControllerState()
    snap = snapshot_dict(state, now_mono=0.0)
    assert snap["mode"] == "NORMAL"
    assert snap["seconds_in_mode"] == 0.0
    assert snap["transition_count"] == 0
    assert snap["entry_arming_seconds"] is None
    assert snap["exit_arming_seconds"] is None
    assert snap["recent_transitions"] == []
    assert "time_in_normal_seconds" in snap
    assert "time_in_defensive_seconds" in snap
    assert "time_in_shock_seconds" in snap


def test_snapshot_dict_after_transition() -> None:
    state = RegimeControllerState()
    # Drive a NORMAL → DEFENSIVE transition.
    for t in range(0, 16):
        _tick_defensive_entry(state, float(t))
    snap = snapshot_dict(state, now_mono=20.0)
    assert snap["mode"] == "DEFENSIVE"
    assert snap["seconds_in_mode"] == 5.0  # 20 - 15
    assert snap["transition_count"] == 1
    assert len(snap["recent_transitions"]) == 1
    last = snap["recent_transitions"][-1]
    assert last["from"] == "NORMAL"
    assert last["to"] == "DEFENSIVE"
    assert "defensive_entry" in last["reason"]


# ---------------------------------------------------------------------------
# Full scenario: NORMAL → DEFENSIVE → SHOCK → DEFENSIVE → NORMAL
# ---------------------------------------------------------------------------


def test_full_session_arc() -> None:
    """Walks the full state-machine: clear NORMAL → util ramps to
    trigger DEFENSIVE → shock fires → shock clears → util drops →
    NORMAL again. Asserts every transition logged in order."""
    state = RegimeControllerState()

    # Phase 1 — clear, NORMAL.
    for t in range(0, 30):
        _tick_normal(state, float(t))
    assert state.mode is Mode.NORMAL

    # Phase 2 — util ramps, after 15 s in DEFENSIVE.
    for t in range(30, 50):
        _tick_defensive_entry(state, float(t))
    assert state.mode is Mode.DEFENSIVE

    # Phase 3 — shock_gate fires.
    m, _ = evaluate_mode(
        state,
        now_mono=55.0,
        util=0.9,
        vol_ratio=3.0,
        slow_trend_active=True,
        inventory_drift_active=True,
        shock_gate_locked=True,
        enabled=True,
    )
    assert m is Mode.SHOCK

    # Phase 4 — shock_gate clears → DEFENSIVE.
    m, _ = evaluate_mode(
        state,
        now_mono=100.0,
        util=0.4,
        vol_ratio=1.0,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
    )
    assert m is Mode.DEFENSIVE

    # Phase 5 — util drops, exit dwell elapses, NORMAL.
    for t in range(100, 131):
        m, _ = evaluate_mode(
            state,
            now_mono=float(t),
            util=0.1,
            vol_ratio=1.0,
            slow_trend_active=False,
            inventory_drift_active=False,
            shock_gate_locked=False,
            enabled=True,
        )
    assert m is Mode.NORMAL

    # Transitions: NORMAL → DEFENSIVE → SHOCK → DEFENSIVE → NORMAL (4)
    assert state.transition_count == 4
    transitions = list(state.last_transitions)
    assert [(frm, to) for (frm, to, _, _) in transitions] == [
        ("NORMAL", "DEFENSIVE"),
        ("DEFENSIVE", "SHOCK"),
        ("SHOCK", "DEFENSIVE"),
        ("DEFENSIVE", "NORMAL"),
    ]
