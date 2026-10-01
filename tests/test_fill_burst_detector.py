"""Tests for ``app/fill_burst_detector.py`` (codex-#3 burst shrink)."""

from __future__ import annotations

from app.fill_burst_detector import FillBurstDetector


def _detector(**overrides) -> FillBurstDetector:
    defaults = dict(
        threshold=3,
        window_seconds=30.0,
        size_mult=0.5,
        cooldown_seconds=60.0,
    )
    defaults.update(overrides)
    return FillBurstDetector(**defaults)


def test_disabled_when_threshold_zero() -> None:
    d = _detector(threshold=0)
    assert not d.enabled()
    d.note_fill(100.0)
    d.note_fill(100.5)
    d.note_fill(101.0)
    assert d.current_size_mult(101.0) == 1.0


def test_disabled_when_size_mult_one() -> None:
    d = _detector(size_mult=1.0)
    assert not d.enabled()


def test_does_not_arm_below_threshold() -> None:
    d = _detector(threshold=3)
    d.note_fill(100.0)
    d.note_fill(101.0)
    assert d.current_size_mult(101.0) == 1.0
    assert not d.active(101.0)


def test_arms_at_threshold_count_within_window() -> None:
    d = _detector(threshold=3, window_seconds=30.0, size_mult=0.5)
    d.note_fill(100.0)
    d.note_fill(105.0)
    d.note_fill(110.0)  # threshold reached
    assert d.current_size_mult(110.0) == 0.5
    assert d.active(110.0)


def test_evicts_fills_outside_window() -> None:
    """Fills older than ``window_seconds`` don't count toward threshold."""
    d = _detector(threshold=3, window_seconds=30.0)
    d.note_fill(100.0)
    d.note_fill(105.0)
    # The third fill comes 60 s later — the first two are out of window.
    d.note_fill(160.0)
    # Only 1 fill in the 30 s window ending at 160 → no arm.
    assert d.current_size_mult(160.0) == 1.0
    assert not d.active(160.0)


def test_cooldown_expires() -> None:
    d = _detector(threshold=3, cooldown_seconds=60.0)
    for t in (100.0, 105.0, 110.0):
        d.note_fill(t)
    assert d.active(110.0)
    assert d.active(169.9)
    assert not d.active(170.1)
    assert d.current_size_mult(170.1) == 1.0


def test_cooldown_extends_on_repeat_arm() -> None:
    d = _detector(threshold=3, cooldown_seconds=60.0)
    for t in (100.0, 105.0, 110.0):
        d.note_fill(t)
    # Second burst at t=140 (still inside the window of the first
    # arming): cooldown should extend.
    d.note_fill(135.0)
    d.note_fill(138.0)
    # The 4th fill (138) sees 4 fills in the last 30s window (100,
    # 105, 110, 135, 138) so threshold still met → arm extends.
    # Cooldown was 110+60=170; new is 138+60=198.
    assert d.remaining_seconds(140.0) > 50.0


def test_recent_fill_count() -> None:
    d = _detector(threshold=3, window_seconds=30.0)
    d.note_fill(100.0)
    d.note_fill(110.0)
    d.note_fill(120.0)
    assert d.recent_fill_count(125.0) == 3
    # By t=131, the t=100 fill should be evicted (30s window).
    assert d.recent_fill_count(131.0) == 2


def test_snapshot_dict_shape() -> None:
    d = _detector(threshold=3, cooldown_seconds=60.0, size_mult=0.5)
    for t in (100.0, 105.0, 110.0):
        d.note_fill(t)
    snap = d.snapshot_dict(110.0)
    assert snap["enabled"] is True
    assert snap["active"] is True
    assert snap["fire_count"] >= 1
    assert 50.0 < snap["seconds_remaining"] <= 60.0
    assert snap["recent_fill_count"] == 3
