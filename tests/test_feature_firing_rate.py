"""v1.4.116 Phase 1E.3.f — FeatureFiringRateTracker unit tests."""

from __future__ import annotations

from app.feature_firing_rate import (
    FeatureFiringRateTracker,
    MAX_RETENTION_SECONDS,
    detect_edge_transitions,
)


def test_empty_tracker_returns_zero_for_unknown_feature() -> None:
    t = FeatureFiringRateTracker()
    assert t.rate_in_window("never_seen", 60.0, 1000.0) == 0


def test_single_fire_counted_in_window() -> None:
    t = FeatureFiringRateTracker()
    t.note_fire("post_fill_cooldown_bid", 1000.0)
    assert t.rate_in_window("post_fill_cooldown_bid", 60.0, 1010.0) == 1
    assert t.rate_in_window("post_fill_cooldown_bid", 60.0, 1059.5) == 1


def test_fire_outside_window_excluded() -> None:
    t = FeatureFiringRateTracker()
    t.note_fire("post_fill_cooldown_bid", 1000.0)
    # At t=1061 the fire at t=1000 is just outside the 60 s window.
    assert t.rate_in_window("post_fill_cooldown_bid", 60.0, 1061.0) == 0


def test_window_boundary_is_half_open() -> None:
    """A fire exactly at ``now − window`` is excluded; a fire at
    ``now − window + epsilon`` is included."""
    t = FeatureFiringRateTracker()
    t.note_fire("x", 940.0)
    # At now=1000, window=60 → cutoff is exactly 940.0. Fire AT
    # cutoff is excluded.
    assert t.rate_in_window("x", 60.0, 1000.0) == 0
    # Move now BACKWARD 1 ms (cutoff slides earlier to 939.999): the
    # fire at 940 is now strictly inside the half-open window.
    assert t.rate_in_window("x", 60.0, 999.999) == 1


def test_multiple_fires_counted() -> None:
    t = FeatureFiringRateTracker()
    for ts in [1000.0, 1005.0, 1010.0, 1020.0, 1045.0]:
        t.note_fire("at_touch_adverse_pause_bid", ts)
    assert (
        t.rate_in_window("at_touch_adverse_pause_bid", 60.0, 1050.0) == 5
    )
    # Tighter window (15 s) at now=1050 → cutoff=1035. Only the
    # fire at 1045 is past the cutoff.
    assert (
        t.rate_in_window("at_touch_adverse_pause_bid", 15.0, 1050.0) == 1
    )
    # Wider window (35 s) at now=1050 → cutoff=1015. Fires at 1020
    # and 1045 are past the cutoff.
    assert (
        t.rate_in_window("at_touch_adverse_pause_bid", 35.0, 1050.0) == 2
    )


def test_old_fires_pruned_at_note() -> None:
    """Entries older than ``MAX_RETENTION_SECONDS`` are pruned at
    append time so the deque stays bounded."""
    t = FeatureFiringRateTracker()
    t.note_fire("f", 0.0)
    t.note_fire("f", 10.0)
    # Now fire 200 s later — entries at 0.0 and 10.0 are past the
    # 120 s retention cap and get pruned at append.
    t.note_fire("f", 210.0)
    assert t.rate_in_window("f", MAX_RETENTION_SECONDS + 10, 211.0) == 1


def test_deque_maxlen_caps_growth_under_pathological_firing() -> None:
    """Even if a feature fires every monotonic-tick (impossible in
    real bot but defends against state-flap bugs), the deque is
    capped at the maxlen safety net."""
    t = FeatureFiringRateTracker()
    for i in range(500):
        t.note_fire("flap", float(i) * 0.01)  # 5 s of 100 Hz firing
    # Window=60 should hit the maxlen cap, not 500.
    rate = t.rate_in_window("flap", 60.0, 5.0)
    assert rate <= 200  # safety-net cap


def test_all_rates_returns_dict_of_every_known_feature() -> None:
    t = FeatureFiringRateTracker()
    t.note_fire("a", 1000.0)
    t.note_fire("b", 1000.0)
    t.note_fire("b", 1010.0)
    rates = t.all_rates(60.0, 1020.0)
    assert rates == {"a": 1, "b": 2}


def test_empty_name_ignored() -> None:
    t = FeatureFiringRateTracker()
    t.note_fire("", 1000.0)
    assert t.feature_names() == []


# ---------------------------------------------------------------------------
# detect_edge_transitions helper
# ---------------------------------------------------------------------------


def test_edge_detection_records_false_to_true() -> None:
    t = FeatureFiringRateTracker()
    new_snap = detect_edge_transitions(
        tracker=t,
        now_mono=1000.0,
        prior_flags={"a": False, "b": False},
        current_flags={"a": True, "b": False},
    )
    assert t.rate_in_window("a", 60.0, 1001.0) == 1
    assert t.rate_in_window("b", 60.0, 1001.0) == 0
    assert new_snap == {"a": True, "b": False}


def test_edge_detection_ignores_already_active() -> None:
    """Feature stays True across ticks — do NOT record a duplicate
    fire. The tracker should only count edge transitions."""
    t = FeatureFiringRateTracker()
    # Tick 1: a fires.
    detect_edge_transitions(
        tracker=t,
        now_mono=1000.0,
        prior_flags={"a": False},
        current_flags={"a": True},
    )
    # Tick 2: a still active — no new fire.
    detect_edge_transitions(
        tracker=t,
        now_mono=1005.0,
        prior_flags={"a": True},
        current_flags={"a": True},
    )
    assert t.rate_in_window("a", 60.0, 1010.0) == 1


def test_edge_detection_none_treated_as_false() -> None:
    """``None`` prior (uninitialised / disabled feature) followed by
    True current → records one fire."""
    t = FeatureFiringRateTracker()
    detect_edge_transitions(
        tracker=t,
        now_mono=1000.0,
        prior_flags={"a": None},
        current_flags={"a": True},
    )
    assert t.rate_in_window("a", 60.0, 1001.0) == 1


def test_edge_detection_returns_independent_snapshot() -> None:
    """The returned snapshot is a copy — mutating the current_flags
    after the call doesn't affect the stashed snapshot."""
    t = FeatureFiringRateTracker()
    current = {"a": True}
    snap = detect_edge_transitions(
        tracker=t,
        now_mono=1000.0,
        prior_flags={},
        current_flags=current,
    )
    current["a"] = False  # mutate live
    assert snap == {"a": True}  # snap unchanged
