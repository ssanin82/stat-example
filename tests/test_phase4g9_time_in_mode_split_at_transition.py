"""Phase 4G.9 (v1.4.219) — time_in_mode counter mis-attribution fix.

Pre-v1.4.219 bug (caught in snapshot
``v1.4.214-260521-202803-prod.okx.ton.usdt.perp``):
    SHOCK reported 32.4 s but reconciled 126.8 s (-94.5 s)
    DEFENSIVE reported 90.0 s but reconciled 30.0 s (+60.0 s)

Root cause: ``accumulate_time_in_mode`` attributed the entire
inter-tick delta to the CURRENT mode at the time the tick fired.
If a transition happened in the gap, ALL of the gap time credited
the post-transition mode — biasing the counters toward whatever
mode the bot was in AT THE TICK MOMENT, not the time-weighted
average across the gap. Also the ``delta = min(delta, 60.0)`` cap
dropped any gap >60 s entirely.

Fix: track ``_last_tick_mode`` alongside ``_last_tick_mono``. When
the modes differ, split the delta at ``state.mode_since_mono``.
Raised the pause-poisoning cap to 600 s (was 60 s).
"""

from __future__ import annotations

import pytest

from app.regime_controller import (
    Mode,
    RegimeControllerState,
    accumulate_time_in_mode,
)


def test_first_tick_seeds_without_accumulating() -> None:
    """First call has no prior tick → seeds the timestamp but
    doesn't credit any counter."""
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    accumulate_time_in_mode(state, now_mono=100.0)
    assert state.time_in_normal_seconds == 0.0
    assert state._last_tick_mono == 100.0
    assert state._last_tick_mode is Mode.NORMAL


def test_simple_accumulation_no_transition() -> None:
    """Two ticks in the same mode — full delta credits the mode."""
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    accumulate_time_in_mode(state, now_mono=100.0)
    accumulate_time_in_mode(state, now_mono=105.0)
    assert state.time_in_normal_seconds == pytest.approx(5.0)


def test_split_at_transition_within_gap() -> None:
    """The headline fix: a transition happens between two
    accumulator calls; pre-transition time credits the OLD mode,
    post-transition time credits the NEW mode."""
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    state.mode_since_mono = 100.0
    accumulate_time_in_mode(state, now_mono=100.0)
    # Simulate 10s in NORMAL, then transition to CAUTIOUS at t=110,
    # then 5s in CAUTIOUS — next tick at t=115.
    state.mode = Mode.CAUTIOUS
    state.mode_since_mono = 110.0
    accumulate_time_in_mode(state, now_mono=115.0)
    assert state.time_in_normal_seconds == pytest.approx(10.0)
    assert state.time_in_cautious_seconds == pytest.approx(5.0)


def test_v1_4_214_snapshot_pattern_reproduced_correctly() -> None:
    """Reproduce the snapshot v1.4.214 SHOCK→DEFENSIVE transition
    and confirm the split now credits each mode the right
    proportion.

    Snapshot's actual SHOCK phase: 126.8 s, then 30 s DEFENSIVE.
    Simulate two ticks: one INSIDE SHOCK (no transition yet),
    one AFTER the transition to DEFENSIVE."""
    state = RegimeControllerState()
    state.mode = Mode.SHOCK
    state.mode_since_mono = 0.0
    # Seed at SHOCK start.
    accumulate_time_in_mode(state, now_mono=0.0)
    # First tick inside SHOCK at t=60.
    accumulate_time_in_mode(state, now_mono=60.0)
    assert state.time_in_shock_seconds == pytest.approx(60.0)
    # Now SHOCK transitions to DEFENSIVE at t=126.8, then
    # accumulator fires again at t=130 (3.2 s after transition).
    state.mode = Mode.DEFENSIVE
    state.mode_since_mono = 126.8
    accumulate_time_in_mode(state, now_mono=130.0)
    # SHOCK should get the additional 126.8 - 60.0 = 66.8 s
    # → total SHOCK: 60.0 + 66.8 = 126.8
    assert state.time_in_shock_seconds == pytest.approx(126.8)
    # DEFENSIVE should get 130.0 - 126.8 = 3.2 s.
    assert state.time_in_defensive_seconds == pytest.approx(3.2)


def test_pause_poisoning_cap_at_600s() -> None:
    """A 1-hour tick gap (process pause / VM suspend) is capped at
    600 s, not 60 s. Raised from 60 → 600 in v1.4.219 because the
    60 s cap was eating real SHOCK time (a 90 s SHOCK gap was
    losing 30 s)."""
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    accumulate_time_in_mode(state, now_mono=0.0)
    accumulate_time_in_mode(state, now_mono=3600.0)  # 1 hour gap
    assert state.time_in_normal_seconds == pytest.approx(600.0), (
        "Pause-poisoning cap should clamp at 600 s, not credit 3600 s"
    )


def test_negative_delta_resets_without_crediting() -> None:
    """Defensive: clock jumps backward → reset without adding."""
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    accumulate_time_in_mode(state, now_mono=100.0)
    accumulate_time_in_mode(state, now_mono=50.0)  # backward
    assert state.time_in_normal_seconds == 0.0
    # State should have re-seeded to now_mono.
    assert state._last_tick_mono == 50.0


def test_mode_since_mono_clamping_against_pathological_input() -> None:
    """Defensive: if mode_since_mono is OUTSIDE [last, now], the
    function clamps to that range. Without the clamp, a malformed
    timestamp could attribute negative seconds."""
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    accumulate_time_in_mode(state, now_mono=100.0)
    # Now switch to CAUTIOUS but with mode_since_mono BEFORE last
    # (= 50, predates the accumulator's seed). Clamp should pull
    # ms to 100 → pre_delta=0 → all 5s credits CAUTIOUS.
    state.mode = Mode.CAUTIOUS
    state.mode_since_mono = 50.0  # pathological — before last tick
    accumulate_time_in_mode(state, now_mono=105.0)
    assert state.time_in_normal_seconds == pytest.approx(0.0)
    assert state.time_in_cautious_seconds == pytest.approx(5.0)


def test_mode_since_after_now_clamped() -> None:
    """Defensive: mode_since_mono AFTER now_mono → clamp to now,
    crediting all delta to the OLD mode."""
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    accumulate_time_in_mode(state, now_mono=100.0)
    # CAUTIOUS at t=110 but accumulator fires at t=105 (mode_since
    # is in the future relative to now). Clamp ms to 105.
    state.mode = Mode.CAUTIOUS
    state.mode_since_mono = 110.0
    accumulate_time_in_mode(state, now_mono=105.0)
    # ms clamped to 105 → pre_delta=5, post_delta=0
    assert state.time_in_normal_seconds == pytest.approx(5.0)
    assert state.time_in_cautious_seconds == pytest.approx(0.0)


def test_multi_transition_in_gap_approximates_last_hop() -> None:
    """Documented limitation: when MULTIPLE transitions happen in
    one tick gap (e.g. NORMAL→CAUTIOUS→SHOCK in <1s), only the
    last hop's split is applied; intermediate modes get no credit.
    This is acceptable because the transition log preserves the
    exact monotonic timestamps for post-session reconciliation.

    Test pins this approximation so future code changes don't
    silently widen the gap.
    """
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    accumulate_time_in_mode(state, now_mono=0.0)
    # NORMAL→CAUTIOUS at t=4, CAUTIOUS→SHOCK at t=6. Accumulator
    # fires at t=10. We expect: NORMAL gets some pre-time,
    # SHOCK gets some post-time, CAUTIOUS gets nothing (its
    # 2 s phase fell entirely within the gap).
    state.mode = Mode.SHOCK
    state.mode_since_mono = 6.0
    accumulate_time_in_mode(state, now_mono=10.0)
    # The function only sees last_mode=NORMAL and current=SHOCK.
    # It splits at mode_since_mono=6 → 6s NORMAL + 4s SHOCK.
    # CAUTIOUS time (4→6) is lost. Acceptable; pinned.
    assert state.time_in_normal_seconds == pytest.approx(6.0)
    assert state.time_in_shock_seconds == pytest.approx(4.0)
    assert state.time_in_cautious_seconds == pytest.approx(0.0)


def test_all_five_modes_correct_counter_routing() -> None:
    """Each mode should route to its own counter."""
    for mode, counter_name in [
        (Mode.CALM, "time_in_calm_seconds"),
        (Mode.NORMAL, "time_in_normal_seconds"),
        (Mode.CAUTIOUS, "time_in_cautious_seconds"),
        (Mode.DEFENSIVE, "time_in_defensive_seconds"),
        (Mode.SHOCK, "time_in_shock_seconds"),
    ]:
        state = RegimeControllerState()
        state.mode = mode
        accumulate_time_in_mode(state, now_mono=0.0)
        accumulate_time_in_mode(state, now_mono=10.0)
        assert getattr(state, counter_name) == pytest.approx(10.0), (
            f"mode={mode.value} should have credited {counter_name}"
        )
