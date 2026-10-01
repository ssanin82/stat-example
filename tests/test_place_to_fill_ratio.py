"""Tests for the place-to-fill ratio tracker.

The risk gate that originally consumed this tracker (kill/pause when
fills/places dropped below a threshold) was removed 2026-05-06 after
a false-positive KILL on a calm-market no-fill stretch — the strategy
was healthy, the gate just didn't model that fills come in bursty
clusters separated by quiet windows. Tracker itself is kept for
observability (heartbeat / dashboard) so the original gate-test
stubs were dropped.
"""

from __future__ import annotations

from app.place_to_fill_ratio_tracker import PlaceToFillRatioTracker


# ---------------------------------------------------------------------------
# Tracker unit tests
# ---------------------------------------------------------------------------


def test_tracker_snapshot_empty() -> None:
    t = PlaceToFillRatioTracker(window_seconds=60.0)
    snap = t.snapshot()
    assert snap.places == 0
    assert snap.fills == 0
    assert snap.ratio_pct is None  # 0/0 => None, not 0.0


def test_tracker_counts_places_and_fills() -> None:
    t = PlaceToFillRatioTracker(window_seconds=60.0)
    for _ in range(100):
        t.note_place()
    for _ in range(5):
        t.note_fill()
    snap = t.snapshot()
    assert snap.places == 100
    assert snap.fills == 5
    assert snap.ratio_pct == 5.0


def test_tracker_evaluate_holdoff() -> None:
    """Below the min-places-before-gate threshold the verdict must
    stay 'ok' even at a 0/N ratio -- otherwise a fresh session would
    trip the gate on its first 10 places before any fill could land."""
    t = PlaceToFillRatioTracker(window_seconds=60.0)
    for _ in range(50):
        t.note_place()
    verdict, snap = t.evaluate(
        min_places_before_gate=100, pause_pct=1.0, kill_pct=0.3
    )
    assert verdict == "ok"
    assert snap.places == 50
    assert snap.fills == 0


def test_tracker_evaluate_below_pause() -> None:
    t = PlaceToFillRatioTracker(window_seconds=60.0)
    for _ in range(200):
        t.note_place()
    # 0.5% — below 1.0 pause threshold but above 0.3 kill threshold.
    t.note_fill()
    verdict, snap = t.evaluate(
        min_places_before_gate=100, pause_pct=1.0, kill_pct=0.3
    )
    assert verdict == "below_pause"
    assert snap.places == 200
    assert snap.fills == 1
    assert snap.ratio_pct == 0.5


def test_tracker_evaluate_below_kill() -> None:
    t = PlaceToFillRatioTracker(window_seconds=60.0)
    for _ in range(1000):
        t.note_place()
    # 0/1000 = 0% — well below the 0.3% kill threshold.
    verdict, snap = t.evaluate(
        min_places_before_gate=100, pause_pct=1.0, kill_pct=0.3
    )
    assert verdict == "below_kill"
    assert snap.places == 1000
    assert snap.fills == 0


def test_tracker_evaluate_ok_above_pause() -> None:
    """Ratio at or above the pause threshold returns 'ok'. Confirms
    the gate doesn't fire on a well-behaved session."""
    t = PlaceToFillRatioTracker(window_seconds=60.0)
    for _ in range(100):
        t.note_place()
    for _ in range(15):
        t.note_fill()
    verdict, _ = t.evaluate(
        min_places_before_gate=100, pause_pct=1.0, kill_pct=0.3
    )
    assert verdict == "ok"


def test_tracker_window_eviction() -> None:
    """Old samples outside the window must be pruned on read."""
    t = PlaceToFillRatioTracker(window_seconds=0.05)  # 50 ms window
    for _ in range(10):
        t.note_place()
    snap1 = t.snapshot()
    assert snap1.places == 10
    import time as _time
    _time.sleep(0.1)  # > window
    snap2 = t.snapshot()
    assert snap2.places == 0
    assert snap2.fills == 0


