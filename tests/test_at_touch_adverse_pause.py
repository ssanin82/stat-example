"""Tests for ``app/at_touch_adverse_pause.py`` (codex-#1 narrow)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.at_touch_adverse_pause import AtTouchAdversePause
from app.enums import Side


@dataclass
class _FakeFill:
    """Minimal Fill stand-in covering only the fields the gate reads."""

    side: Side
    markout_5s_bps: Optional[float]
    quote_aggressiveness: Optional[str]


def _gate(**overrides) -> AtTouchAdversePause:
    defaults = dict(
        threshold_bps=-5.0,
        pause_seconds=30.0,
        min_fills=3,
        window_size=20,
    )
    defaults.update(overrides)
    return AtTouchAdversePause(**defaults)


def test_disabled_when_threshold_zero() -> None:
    g = _gate(threshold_bps=0.0)
    assert not g.enabled()
    f = _FakeFill(Side.BUY, markout_5s_bps=-50.0, quote_aggressiveness="at_touch")
    g.observe_resolved_5s_markout(f, 100.0)
    assert not g.is_paused(Side.BUY, 100.0)


def test_ignores_behind_touch_fills() -> None:
    g = _gate()
    f = _FakeFill(Side.BUY, markout_5s_bps=-100.0, quote_aggressiveness="behind_touch")
    for _ in range(5):
        g.observe_resolved_5s_markout(f, 100.0)
    # Behind-touch fills shouldn't feed the at_touch median.
    assert not g.is_paused(Side.BUY, 100.0)


def test_ignores_fills_without_markout() -> None:
    g = _gate()
    f = _FakeFill(Side.BUY, markout_5s_bps=None, quote_aggressiveness="at_touch")
    for _ in range(5):
        g.observe_resolved_5s_markout(f, 100.0)
    assert not g.is_paused(Side.BUY, 100.0)


def test_arms_pause_when_median_below_threshold() -> None:
    g = _gate(threshold_bps=-5.0, min_fills=3, pause_seconds=30.0)
    # 3 fills with median -10 (< -5).
    for mo in (-8.0, -10.0, -12.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.0
        )
    assert g.is_paused(Side.BUY, 100.0)
    assert 29.0 <= g.remaining_seconds(Side.BUY, 100.0) <= 30.01
    # ASK side unaffected.
    assert not g.is_paused(Side.SELL, 100.0)


def test_does_not_arm_when_median_above_threshold() -> None:
    g = _gate(threshold_bps=-5.0, min_fills=3)
    # 3 fills with median -2 (above threshold).
    for mo in (-1.0, -2.0, -3.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.0
        )
    assert not g.is_paused(Side.BUY, 100.0)


def test_pause_expires_after_duration() -> None:
    g = _gate(threshold_bps=-5.0, min_fills=3, pause_seconds=30.0)
    for mo in (-8.0, -10.0, -12.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.0
        )
    assert g.is_paused(Side.BUY, 100.0)
    assert g.is_paused(Side.BUY, 129.9)  # just before expiry
    assert not g.is_paused(Side.BUY, 130.1)  # past expiry


def test_per_side_independence() -> None:
    """A BUY-side adverse cluster doesn't pause the SELL side."""
    g = _gate(threshold_bps=-5.0, min_fills=3)
    for mo in (-8.0, -10.0, -12.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.0
        )
    assert g.is_paused(Side.BUY, 100.0)
    assert not g.is_paused(Side.SELL, 100.0)


def test_does_not_arm_below_min_fills() -> None:
    g = _gate(threshold_bps=-5.0, min_fills=5)
    # Only 3 adverse fills; needs 5.
    for mo in (-20.0, -20.0, -20.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.0
        )
    assert not g.is_paused(Side.BUY, 100.0)


def test_snapshot_dict_shape() -> None:
    g = _gate(threshold_bps=-5.0, min_fills=3)
    for mo in (-8.0, -10.0, -12.0):
        g.observe_resolved_5s_markout(
            _FakeFill(Side.BUY, mo, "at_touch"), 100.0
        )
    snap = g.snapshot_dict(110.0)
    assert snap["enabled"] is True
    assert snap["buy"]["paused"] is True
    assert snap["buy"]["fire_count"] >= 1
    assert snap["buy"]["recent_count"] == 3
    assert snap["buy"]["recent_median_bps"] == -10.0
    assert snap["sell"]["paused"] is False
