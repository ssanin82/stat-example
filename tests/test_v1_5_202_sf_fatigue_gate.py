"""v1.5.202 — SF-event-count fatigue ladder tests.

Covers the pure-helper module ``app/sf_fatigue_gate.py``:

* Tier-resolution thresholds (CLEAR → WIDEN → PAUSE_SHORT → PAUSE_LONG → KILLED)
* Rolling-window decay (events ageing out lowers the tier)
* Pause-budget cooldown expiry behaviour
* Sticky KILLED tier (cannot fall back)
* Rising-edge fire_count semantics
* Pause re-arm on PAUSE_SHORT → PAUSE_LONG escalation
* snapshot_dict shape + JSON-serialisable fields
* seconds_remaining_on_pause math
"""

from __future__ import annotations

import json

import pytest

from app.sf_fatigue_gate import (
    SfFatigueGateState,
    TIER_CLEAR,
    TIER_WIDEN,
    TIER_PAUSE_SHORT,
    TIER_PAUSE_LONG,
    TIER_KILLED,
    evaluate_tier,
    event_count_in_window,
    note_sf_event,
    seconds_remaining_on_pause,
    snapshot_dict,
)


# Reusable thresholds. tier4=10 keeps the KILLED ceiling reachable in
# tests without needing 12+ events.
T1, T2, T3, T4 = 3, 5, 8, 10
WINDOW = 1800.0  # 30 min
PAUSE_S = 600.0  # 10 min
PAUSE_L = 1800.0  # 30 min


def _eval(state: SfFatigueGateState, now: float, **overrides) -> str:
    """Evaluator with the standard thresholds; overrides any kwarg."""
    kwargs: dict = dict(
        now_mono=now,
        window_seconds=WINDOW,
        tier1_widen_count=T1,
        tier2_pause_short_count=T2,
        tier3_pause_long_count=T3,
        tier4_kill_count=T4,
        pause_short_seconds=PAUSE_S,
        pause_long_seconds=PAUSE_L,
    )
    kwargs.update(overrides)
    return evaluate_tier(state, **kwargs)


# ------------------------ tier resolution ---------------------------------


def test_clear_when_no_events() -> None:
    st = SfFatigueGateState()
    assert _eval(st, 100.0) == TIER_CLEAR
    assert st.current_tier == TIER_CLEAR
    assert st.fire_count == 0


def test_widen_at_tier1_threshold() -> None:
    st = SfFatigueGateState()
    now = 1000.0
    for _ in range(T1):
        note_sf_event(st, now_mono=now)
    assert _eval(st, now) == TIER_WIDEN
    assert st.fire_count == 1  # rising edge


def test_widen_one_below_threshold_is_clear() -> None:
    st = SfFatigueGateState()
    now = 1000.0
    for _ in range(T1 - 1):
        note_sf_event(st, now_mono=now)
    assert _eval(st, now) == TIER_CLEAR


def test_pause_short_at_tier2() -> None:
    st = SfFatigueGateState()
    now = 1000.0
    for _ in range(T2):
        note_sf_event(st, now_mono=now)
    assert _eval(st, now) == TIER_PAUSE_SHORT
    assert st.pause_armed_at_mono == pytest.approx(now)


def test_pause_long_at_tier3() -> None:
    st = SfFatigueGateState()
    now = 1000.0
    for _ in range(T3):
        note_sf_event(st, now_mono=now)
    assert _eval(st, now) == TIER_PAUSE_LONG


def test_killed_at_tier4() -> None:
    st = SfFatigueGateState()
    now = 1000.0
    for _ in range(T4):
        note_sf_event(st, now_mono=now)
    assert _eval(st, now) == TIER_KILLED


# ------------------------ stickiness + decay ------------------------------


def test_killed_is_sticky_even_after_decay() -> None:
    """Once KILLED, the tier stays KILLED even when events age out
    of the rolling window. Operator-restart-required semantics."""
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(T4):
        note_sf_event(st, now_mono=t0)
    assert _eval(st, t0) == TIER_KILLED
    # Advance past the rolling-window expiration; all events should
    # have aged out. Tier still KILLED.
    t1 = t0 + WINDOW + 10.0
    assert _eval(st, t1) == TIER_KILLED


def test_widen_decays_to_clear_via_window() -> None:
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(T1):
        note_sf_event(st, now_mono=t0)
    assert _eval(st, t0) == TIER_WIDEN
    # All events age out past the window.
    t1 = t0 + WINDOW + 1.0
    assert _eval(st, t1) == TIER_CLEAR
    assert event_count_in_window(st) == 0


def test_pause_clears_after_budget_expires_and_count_drops() -> None:
    """PAUSE_SHORT must clear when (a) the pause-budget elapsed AND
    (b) the rolling-window count is back below tier-2 threshold."""
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(T2):
        note_sf_event(st, now_mono=t0)
    assert _eval(st, t0) == TIER_PAUSE_SHORT
    # Past the pause budget AND past the rolling-window expiration
    # (so count drops to 0). Should clear.
    t1 = t0 + max(PAUSE_S, WINDOW) + 1.0
    assert _eval(st, t1) == TIER_CLEAR


# ------------------------ fire_count semantics ----------------------------


def test_fire_count_is_rising_edge_only() -> None:
    """fire_count bumps when the tier escalates to a stricter level,
    NOT when it stays at the same tier or falls back."""
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(T1):
        note_sf_event(st, now_mono=t0)
    assert _eval(st, t0) == TIER_WIDEN
    assert st.fire_count == 1
    # Re-evaluate same state at same time — no bump.
    _eval(st, t0)
    assert st.fire_count == 1
    # Add more events → escalate to PAUSE_SHORT. Bump.
    for _ in range(T2 - T1):
        note_sf_event(st, now_mono=t0)
    assert _eval(st, t0) == TIER_PAUSE_SHORT
    assert st.fire_count == 2
    # Decay back to WIDEN via window expiration of the recent batch.
    # Falling-edge transition: NO bump.
    # (We need to age out enough of the recent events.)


# ------------------------ pause re-arm on escalation ----------------------


def test_pause_short_to_long_re_arms_timer() -> None:
    """Escalating PAUSE_SHORT → PAUSE_LONG mid-pause re-arms the
    timer so the LONG window starts fresh. Otherwise the SHORT clock
    might expire mid-LONG-pause."""
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(T2):
        note_sf_event(st, now_mono=t0)
    assert _eval(st, t0) == TIER_PAUSE_SHORT
    arm_short = st.pause_armed_at_mono
    assert arm_short == pytest.approx(t0)
    # Some time later, more events push us to PAUSE_LONG.
    t1 = t0 + 100.0
    for _ in range(T3 - T2):
        note_sf_event(st, now_mono=t1)
    assert _eval(st, t1) == TIER_PAUSE_LONG
    # Re-armed to the NEW time.
    assert st.pause_armed_at_mono == pytest.approx(t1)


# ------------------------ seconds_remaining_on_pause ----------------------


def test_seconds_remaining_zero_when_clear() -> None:
    st = SfFatigueGateState()
    _eval(st, 1000.0)
    assert seconds_remaining_on_pause(
        st,
        now_mono=1000.0,
        pause_short_seconds=PAUSE_S,
        pause_long_seconds=PAUSE_L,
    ) == 0.0


def test_seconds_remaining_matches_pause_budget_minus_elapsed() -> None:
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(T2):
        note_sf_event(st, now_mono=t0)
    _eval(st, t0)
    # Right at arming → full budget remaining.
    assert seconds_remaining_on_pause(
        st,
        now_mono=t0,
        pause_short_seconds=PAUSE_S,
        pause_long_seconds=PAUSE_L,
    ) == pytest.approx(PAUSE_S)
    # 60 s later → 60 s less.
    assert seconds_remaining_on_pause(
        st,
        now_mono=t0 + 60.0,
        pause_short_seconds=PAUSE_S,
        pause_long_seconds=PAUSE_L,
    ) == pytest.approx(PAUSE_S - 60.0)


# ------------------------ snapshot_dict -----------------------------------


def test_snapshot_dict_shape_and_json_serialisable() -> None:
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(T1):
        note_sf_event(st, now_mono=t0)
    _eval(st, t0)
    snap = snapshot_dict(
        st,
        now_mono=t0,
        window_seconds=WINDOW,
        tier1_widen_count=T1,
        tier2_pause_short_count=T2,
        tier3_pause_long_count=T3,
        tier4_kill_count=T4,
        pause_short_seconds=PAUSE_S,
        pause_long_seconds=PAUSE_L,
        enabled=True,
    )
    # All expected keys present.
    expected_keys = {
        "enabled", "window_seconds", "tier", "events_in_window",
        "tier1_widen_count", "tier2_pause_short_count",
        "tier3_pause_long_count", "tier4_kill_count",
        "cooldown_seconds_remaining", "fire_count",
    }
    assert set(snap.keys()) == expected_keys
    assert snap["tier"] == TIER_WIDEN
    assert snap["events_in_window"] == T1
    assert snap["enabled"] is True
    # JSON round-trip.
    assert json.loads(json.dumps(snap)) == snap


def test_snapshot_dict_cooldown_zero_when_not_paused() -> None:
    st = SfFatigueGateState()
    snap = snapshot_dict(
        st,
        now_mono=1000.0,
        window_seconds=WINDOW,
        tier1_widen_count=T1,
        tier2_pause_short_count=T2,
        tier3_pause_long_count=T3,
        tier4_kill_count=T4,
        pause_short_seconds=PAUSE_S,
        pause_long_seconds=PAUSE_L,
        enabled=True,
    )
    assert snap["cooldown_seconds_remaining"] == 0.0


def test_snapshot_dict_cooldown_nonzero_when_paused() -> None:
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(T2):
        note_sf_event(st, now_mono=t0)
    _eval(st, t0)
    snap = snapshot_dict(
        st,
        now_mono=t0,
        window_seconds=WINDOW,
        tier1_widen_count=T1,
        tier2_pause_short_count=T2,
        tier3_pause_long_count=T3,
        tier4_kill_count=T4,
        pause_short_seconds=PAUSE_S,
        pause_long_seconds=PAUSE_L,
        enabled=True,
    )
    assert snap["tier"] == TIER_PAUSE_SHORT
    assert snap["cooldown_seconds_remaining"] == pytest.approx(PAUSE_S)


# ------------------------ zero / disabled handling ------------------------


def test_tier_threshold_zero_is_disabled() -> None:
    """Setting a tier count to 0 disables that tier — even with many
    events the gate won't escalate past whichever next-higher tier
    is also non-zero."""
    st = SfFatigueGateState()
    t0 = 1000.0
    for _ in range(20):
        note_sf_event(st, now_mono=t0)
    # Disable tier4 → bot won't get KILLED even with 20 events; falls
    # back to PAUSE_LONG (next-highest configured tier).
    tier = _eval(
        st, t0,
        tier4_kill_count=0,
    )
    assert tier == TIER_PAUSE_LONG


def test_window_zero_disables_pruning() -> None:
    """window_seconds=0 means events never age out (a NOP pruner)."""
    st = SfFatigueGateState()
    note_sf_event(st, now_mono=1.0)
    note_sf_event(st, now_mono=2.0)
    _eval(st, 1_000_000.0, window_seconds=0.0)
    assert event_count_in_window(st) == 2


# ------------------------ event_count_in_window ---------------------------


def test_event_count_in_window_reflects_pruning() -> None:
    st = SfFatigueGateState()
    note_sf_event(st, now_mono=0.0)
    note_sf_event(st, now_mono=100.0)
    note_sf_event(st, now_mono=200.0)
    # Window = 50s, evaluating at t=250s → only the t=200 event
    # should survive.
    _eval(st, 250.0, window_seconds=50.0)
    assert event_count_in_window(st) == 1
