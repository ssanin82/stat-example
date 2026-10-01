"""Unit tests for the adaptive join-depth controller.

Pure controller behaviour: drive synthetic inputs (cross-rejection
counter, recent fills with markouts, fill-rate) and assert the
overlay converges as designed. No Bot / exchange / market wiring.

Design ref: ``plans/auto-tune.md``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from app.join_depth_controller import JoinDepthController
from tests.settings_helpers import UnitTestSettings


def _settings(**kw):
    """Settings helper enabling autotune by default; override via kw.
    Most tests want autotune ON since the contract under test is the
    controller's update logic, not the disable path."""
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "JOIN_DEPTH_AUTOTUNE_ENABLED": True,
        "JOIN_DEPTH_AUTOTUNE_UPDATE_SECONDS": 30.0,
        "JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA": 0.3,
        "JOIN_DEPTH_AUTOTUNE_OVERLAY_MIN_BPS": -2.0,
        "JOIN_DEPTH_AUTOTUNE_OVERLAY_MAX_BPS": 6.0,
        "JOIN_DEPTH_AUTOTUNE_ALPHA_REJECT": 0.5,
        "JOIN_DEPTH_AUTOTUNE_ALPHA_MARKOUT_1S": 0.3,
        "JOIN_DEPTH_AUTOTUNE_ALPHA_MARKOUT_5S": 0.2,
        "JOIN_DEPTH_AUTOTUNE_ALPHA_UNDERFILL": 0.1,
        "JOIN_DEPTH_AUTOTUNE_TARGET_FILLS_PER_MIN": 1.0,
        "JOIN_DEPTH_AUTOTUNE_SATURATION_WARN_SECONDS": 300.0,
    }
    base.update(kw)
    return UnitTestSettings.model_validate(base)


@dataclass
class _StubFill:
    markout_1s_bps: Optional[float] = None
    markout_5s_bps: Optional[float] = None


class _StubQuoteQuality:
    def __init__(self, count: int = 0):
        self.post_only_cross_rejection_count = count


def _stub_state(
    *,
    cross_reject_count: int = 0,
    fills: Optional[list[_StubFill]] = None,
    trades_per_min: float = 0.0,
):
    state = MagicMock()
    state.quote_quality = _StubQuoteQuality(cross_reject_count)
    state.recent_fills = deque(fills or [])
    state.trades_last_minute = MagicMock(return_value=trades_per_min)
    return state


# ---------------------------------------------------------------------------
# Disable path
# ---------------------------------------------------------------------------


def test_disabled_controller_returns_zero_overlay() -> None:
    s = _settings(JOIN_DEPTH_AUTOTUNE_ENABLED=False)
    c = JoinDepthController(s)
    state = _stub_state(cross_reject_count=10)
    c.tick(state, now_mono=100.0)
    # Even after a tick, disabled controller publishes 0.0.
    assert c.current_overlay_bps() == 0.0


def test_disabled_controller_does_not_update_internal_state() -> None:
    """Disabled controller never updates its bookkeeping, so flipping
    the flag back on later starts from a clean baseline."""
    s = _settings(JOIN_DEPTH_AUTOTUNE_ENABLED=False)
    c = JoinDepthController(s)
    state = _stub_state(cross_reject_count=10)
    c.tick(state, now_mono=100.0)
    assert c.snapshot().last_update_iso is None


# ---------------------------------------------------------------------------
# Update interval gating
# ---------------------------------------------------------------------------


def test_first_tick_runs_immediately() -> None:
    s = _settings()
    c = JoinDepthController(s)
    state = _stub_state(cross_reject_count=0)
    c.tick(state, now_mono=100.0)
    assert c.snapshot().last_update_iso is not None


def test_within_interval_skipped() -> None:
    """A tick within the update interval is a no-op."""
    s = _settings(JOIN_DEPTH_AUTOTUNE_UPDATE_SECONDS=30.0)
    c = JoinDepthController(s)
    state = _stub_state()
    c.tick(state, now_mono=100.0)  # first run, sets timer
    first_iso = c.snapshot().last_update_iso
    c.tick(state, now_mono=110.0)  # 10s later, < 30s
    assert c.snapshot().last_update_iso == first_iso


def test_after_interval_runs() -> None:
    s = _settings(JOIN_DEPTH_AUTOTUNE_UPDATE_SECONDS=30.0)
    c = JoinDepthController(s)
    state = _stub_state()
    c.tick(state, now_mono=100.0)
    first_iso = c.snapshot().last_update_iso
    c.tick(state, now_mono=131.0)  # 31s later
    assert c.snapshot().last_update_iso != first_iso


# ---------------------------------------------------------------------------
# Cross-rejection contribution
# ---------------------------------------------------------------------------


def test_cross_reject_pushes_overlay_positive() -> None:
    """Sustained cross-reject rate → overlay grows positive."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0,  # full step toward target
        JOIN_DEPTH_AUTOTUNE_ALPHA_REJECT=0.5,
    )
    c = JoinDepthController(s)
    # First tick establishes baseline: 0 rejects.
    c.tick(_stub_state(cross_reject_count=0), now_mono=100.0)
    # 31s later, 5 rejects → 5/31s × 60 = ~9.7 rejects/min →
    # contrib_reject ≈ 0.5 × 9.7 ≈ 4.84 bps.
    c.tick(_stub_state(cross_reject_count=5), now_mono=131.0)
    snap = c.snapshot()
    assert snap.contrib_reject_bps > 0
    assert c.current_overlay_bps() > 0


def test_zero_rejects_no_contribution() -> None:
    s = _settings(JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0)
    c = JoinDepthController(s)
    c.tick(_stub_state(cross_reject_count=0), now_mono=100.0)
    c.tick(_stub_state(cross_reject_count=0), now_mono=131.0)
    assert c.snapshot().contrib_reject_bps == 0.0


# ---------------------------------------------------------------------------
# Markout contribution
# ---------------------------------------------------------------------------


def test_adverse_1s_markouts_push_overlay_positive() -> None:
    """All fills show -3 bps 1s adverse → overlay rises."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0,
        JOIN_DEPTH_AUTOTUNE_ALPHA_MARKOUT_1S=0.3,
    )
    c = JoinDepthController(s)
    fills = [_StubFill(markout_1s_bps=-3.0) for _ in range(10)]
    c.tick(_stub_state(fills=fills), now_mono=100.0)
    # median = -3, contrib = 0.3 × 3 = 0.9 bps.
    snap = c.snapshot()
    assert snap.median_1s_markout_bps == pytest.approx(-3.0)
    assert snap.contrib_markout_1s_bps == pytest.approx(0.9)


def test_favorable_markouts_no_contribution() -> None:
    """Positive markouts (favourable) don't add to overlay — we only
    react to adverse selection, not free money."""
    s = _settings(JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0)
    c = JoinDepthController(s)
    fills = [_StubFill(markout_1s_bps=+3.0, markout_5s_bps=+5.0) for _ in range(10)]
    c.tick(_stub_state(fills=fills), now_mono=100.0)
    snap = c.snapshot()
    assert snap.contrib_markout_1s_bps == 0.0
    assert snap.contrib_markout_5s_bps == 0.0


def test_no_fills_no_markout_contribution() -> None:
    s = _settings(JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0)
    c = JoinDepthController(s)
    c.tick(_stub_state(fills=[]), now_mono=100.0)
    snap = c.snapshot()
    assert snap.contrib_markout_1s_bps == 0.0
    assert snap.contrib_markout_5s_bps == 0.0
    assert snap.median_1s_markout_bps is None


# ---------------------------------------------------------------------------
# Underfill contribution (negative — pull tighter)
# ---------------------------------------------------------------------------


def test_underfill_pulls_overlay_negative() -> None:
    """No fills + target=1/min → underfill = 1, contrib = -0.1 bps."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0,
        JOIN_DEPTH_AUTOTUNE_ALPHA_UNDERFILL=0.1,
        JOIN_DEPTH_AUTOTUNE_TARGET_FILLS_PER_MIN=1.0,
    )
    c = JoinDepthController(s)
    c.tick(_stub_state(trades_per_min=0.0), now_mono=100.0)
    snap = c.snapshot()
    assert snap.contrib_underfill_bps == pytest.approx(-0.1)


def test_overfill_no_negative_pull() -> None:
    """Fill rate above target → underfill term zero, no negative
    pull (we don't widen because we're filling fast)."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0,
        JOIN_DEPTH_AUTOTUNE_TARGET_FILLS_PER_MIN=1.0,
    )
    c = JoinDepthController(s)
    c.tick(_stub_state(trades_per_min=5.0), now_mono=100.0)
    assert c.snapshot().contrib_underfill_bps == 0.0


# ---------------------------------------------------------------------------
# Clamps
# ---------------------------------------------------------------------------


def test_overlay_clamped_at_max() -> None:
    """Massive cross-reject rate → clamped to overlay_max_bps."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0,
        JOIN_DEPTH_AUTOTUNE_OVERLAY_MAX_BPS=6.0,
        JOIN_DEPTH_AUTOTUNE_ALPHA_REJECT=10.0,  # huge → overshoot ceiling
    )
    c = JoinDepthController(s)
    c.tick(_stub_state(cross_reject_count=0), now_mono=100.0)
    c.tick(_stub_state(cross_reject_count=100), now_mono=131.0)
    assert c.current_overlay_bps() == pytest.approx(6.0)


def test_overlay_clamped_at_min() -> None:
    """Massive underfill at high alpha → clamped to overlay_min_bps."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0,
        JOIN_DEPTH_AUTOTUNE_OVERLAY_MIN_BPS=-2.0,
        JOIN_DEPTH_AUTOTUNE_ALPHA_UNDERFILL=10.0,  # huge → undershoot floor
        JOIN_DEPTH_AUTOTUNE_TARGET_FILLS_PER_MIN=10.0,
    )
    c = JoinDepthController(s)
    c.tick(_stub_state(trades_per_min=0.0), now_mono=100.0)
    assert c.current_overlay_bps() == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# EWMA smoothing
# ---------------------------------------------------------------------------


def test_ewma_smoothing_partial_step() -> None:
    """With ewma_alpha=0.3, one tick of full target should move
    overlay by ≈30 % of the way (zero underfill so we isolate the
    markout contribution)."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=0.3,
        JOIN_DEPTH_AUTOTUNE_ALPHA_MARKOUT_1S=1.0,
        JOIN_DEPTH_AUTOTUNE_TARGET_FILLS_PER_MIN=0.0,
    )
    c = JoinDepthController(s)
    fills = [_StubFill(markout_1s_bps=-5.0) for _ in range(10)]
    # First tick: target = 5, overlay = 0 + 0.3 × 5 = 1.5.
    c.tick(_stub_state(fills=fills), now_mono=100.0)
    assert c.current_overlay_bps() == pytest.approx(1.5)


def test_ewma_decays_when_target_drops() -> None:
    """If target drops to 0 (signals clear), overlay decays toward 0
    over multiple ticks."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=0.3,
        JOIN_DEPTH_AUTOTUNE_ALPHA_MARKOUT_1S=1.0,
        JOIN_DEPTH_AUTOTUNE_TARGET_FILLS_PER_MIN=0.0,
    )
    c = JoinDepthController(s)
    # Build up overlay to 1.5.
    fills_adv = [_StubFill(markout_1s_bps=-5.0) for _ in range(10)]
    c.tick(_stub_state(fills=fills_adv), now_mono=100.0)
    overlay_after_first = c.current_overlay_bps()
    # Now target = 0 (no fills, no underfill term). overlay = 0.7 × 1.5 = 1.05.
    c.tick(_stub_state(fills=[]), now_mono=131.0)
    assert c.current_overlay_bps() < overlay_after_first
    assert c.current_overlay_bps() == pytest.approx(0.7 * overlay_after_first)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_reset_clears_overlay() -> None:
    s = _settings(JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0, JOIN_DEPTH_AUTOTUNE_ALPHA_MARKOUT_1S=1.0)
    c = JoinDepthController(s)
    fills = [_StubFill(markout_1s_bps=-5.0) for _ in range(10)]
    c.tick(_stub_state(fills=fills), now_mono=100.0)
    assert c.current_overlay_bps() > 0
    c.reset()
    assert c.current_overlay_bps() == 0.0


def test_reset_clears_saturation_state() -> None:
    """If overlay was saturated, reset clears the saturation timer."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0,
        JOIN_DEPTH_AUTOTUNE_ALPHA_REJECT=10.0,
    )
    c = JoinDepthController(s)
    c.tick(_stub_state(cross_reject_count=0), now_mono=100.0)
    c.tick(_stub_state(cross_reject_count=100), now_mono=131.0)
    assert c.snapshot().saturated_seconds == 0  # first hit, just started
    c.reset()
    assert c.snapshot().overlay_bps == 0.0


# ---------------------------------------------------------------------------
# Saturation guard (warning emission tested separately via caplog)
# ---------------------------------------------------------------------------


def test_saturation_warning_after_threshold(caplog) -> None:
    """When overlay saturates at max for >= warn_seconds, log warning."""
    s = _settings(
        JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=1.0,
        JOIN_DEPTH_AUTOTUNE_ALPHA_REJECT=10.0,
        JOIN_DEPTH_AUTOTUNE_SATURATION_WARN_SECONDS=60.0,
    )
    c = JoinDepthController(s)
    state = _stub_state(cross_reject_count=0)
    c.tick(state, now_mono=100.0)
    state.quote_quality.post_only_cross_rejection_count = 100
    c.tick(state, now_mono=131.0)  # saturated, but only just
    assert c.current_overlay_bps() == pytest.approx(6.0)
    state.quote_quality.post_only_cross_rejection_count = 200
    with caplog.at_level("WARNING"):
        c.tick(state, now_mono=200.0)  # 69s after saturation start
    assert any("join_depth_autotune_saturated" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Snapshot integrity
# ---------------------------------------------------------------------------


def test_snapshot_reports_all_inputs() -> None:
    s = _settings(JOIN_DEPTH_AUTOTUNE_EWMA_ALPHA=0.3)
    c = JoinDepthController(s)
    fills = [_StubFill(markout_1s_bps=-2.0, markout_5s_bps=-3.0) for _ in range(5)]
    c.tick(
        _stub_state(cross_reject_count=2, fills=fills, trades_per_min=0.5),
        now_mono=100.0,
    )
    snap = c.snapshot()
    assert snap.median_1s_markout_bps == pytest.approx(-2.0)
    assert snap.median_5s_markout_bps == pytest.approx(-3.0)
    assert snap.trades_per_min == pytest.approx(0.5)
    assert snap.target_overlay_bps is not None
    assert snap.last_update_iso is not None
