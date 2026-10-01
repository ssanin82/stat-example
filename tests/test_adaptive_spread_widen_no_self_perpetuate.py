"""adaptive_spread_widen must not self-perpetuate.

Regression for ``tmp/snap_20260418_094415``. The armed-every-tick re-extension
meant that a single 3-SELL / 1-BUY fill sequence in minute 1 of the session
kept ``adaptive_spread_widen`` armed for the full 20 minutes — the widen
itself made subsequent orders invisible, no new fills came in, the same 4
fills kept arming the deadline forever.

Invariants (mirroring ``_maybe_arm_adverse_side_pause``):

1. Active deadline NOT extended by a repeat arming signal.
2. After deadline expires, re-arm only if at least one NEW fill has landed
   since the previous arming.

Logic lives inline in ``Bot.one_tick`` (app/bot.py). We test it by replicating
the decision block with a minimal shell — same approach the adverse_side_pause
tests use.
"""

from __future__ import annotations

import time

import pytest

from app.models import ToxicitySnapshot


class _ShellState:
    """Minimal surface of BotState needed for the arming block."""

    def __init__(self):
        self.adaptive_spread_widen_until_mono: float = 0.0
        self.adaptive_spread_widen_arm_n_fills: int = -1
        self.quote_quality_widen_latched: bool = False


class _ShellSettings:
    """The knobs consumed by the arming block."""

    def __init__(self, **overrides):
        self.adaptive_spread_adverse_overlay_half_spread_bps = overrides.get(
            "adaptive_spread_adverse_overlay_half_spread_bps", 4.0
        )
        self.toxicity_cooldown_seconds = overrides.get("toxicity_cooldown_seconds", 30.0)
        self.toxicity_markout_soft_bps = overrides.get("toxicity_markout_soft_bps", 3.0)
        self.toxicity_one_sided_fill_ratio = overrides.get(
            "toxicity_one_sided_fill_ratio", 0.75
        )


def _arm_block(
    state: _ShellState,
    settings: _ShellSettings,
    *,
    tox_snap: ToxicitySnapshot,
    n_fills_now: int,
    qq_sig: bool = False,
    now_mono: float | None = None,
) -> None:
    """Replicates the arming block in ``Bot.one_tick`` exactly."""
    from app.quoting import adverse_spread_widen_arm

    if now_mono is None:
        now_mono = time.monotonic()
    overlay_cfg = float(settings.adaptive_spread_adverse_overlay_half_spread_bps)
    if now_mono >= state.adaptive_spread_widen_until_mono:
        state.quote_quality_widen_latched = False
    widen_active = now_mono + 1e-9 < state.adaptive_spread_widen_until_mono
    prev_arm_n = int(state.adaptive_spread_widen_arm_n_fills)
    can_arm_fresh = (not widen_active) and (n_fills_now > prev_arm_n)
    want_arm_adverse = overlay_cfg > 0 and adverse_spread_widen_arm(settings, tox_snap)
    want_arm_qq = bool(qq_sig)
    if can_arm_fresh and (want_arm_adverse or want_arm_qq):
        state.adaptive_spread_widen_until_mono = (
            now_mono + float(settings.toxicity_cooldown_seconds)
        )
        state.adaptive_spread_widen_arm_n_fills = n_fills_now
        if want_arm_qq:
            state.quote_quality_widen_latched = True
    elif qq_sig and widen_active:
        state.quote_quality_widen_latched = True


def _tox_one_sided_snap(ratio: float = 0.75) -> ToxicitySnapshot:
    """Toxicity snapshot with the one-sided-fill-ratio trigger armed."""
    from dataclasses import dataclass

    return ToxicitySnapshot(
        score=0.2,
        one_sided_fill_ratio=ratio,
        avg_adverse_markout_bps=-1.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
        adverse_uses_delayed_markouts=True,
        delayed_markout_sample_count=4,
    )


def test_initial_arm_sets_deadline_and_fill_count() -> None:
    state = _ShellState()
    settings = _ShellSettings()
    now = 1000.0
    _arm_block(
        state,
        settings,
        tox_snap=_tox_one_sided_snap(),
        n_fills_now=4,
        now_mono=now,
    )
    assert state.adaptive_spread_widen_until_mono == pytest.approx(now + 30.0)
    assert state.adaptive_spread_widen_arm_n_fills == 4


def test_repeat_arm_while_active_does_not_extend_deadline() -> None:
    """The 094415 bug: signal re-armed every tick on stale data, extending the
    deadline forever. This is the primary invariant."""
    state = _ShellState()
    settings = _ShellSettings()
    now0 = 1000.0
    # First tick: arm.
    _arm_block(
        state, settings, tox_snap=_tox_one_sided_snap(), n_fills_now=4, now_mono=now0
    )
    first_deadline = state.adaptive_spread_widen_until_mono
    # 100 subsequent ticks with the SAME signal (same fill count, same toxicity).
    # The widen is still active; the deadline must not move.
    for i in range(1, 101):
        _arm_block(
            state,
            settings,
            tox_snap=_tox_one_sided_snap(),
            n_fills_now=4,
            now_mono=now0 + i * 0.5,
        )
    assert state.adaptive_spread_widen_until_mono == first_deadline


def test_expired_with_no_new_fills_does_not_rearm() -> None:
    """After the 30-s deadline passes: if no new fills landed (because we were
    invisible while armed), re-arming on stale data would re-create the
    lockout. Must refuse to re-arm."""
    state = _ShellState()
    settings = _ShellSettings()
    _arm_block(
        state, settings, tox_snap=_tox_one_sided_snap(), n_fills_now=4, now_mono=1000.0
    )
    # Fast-forward past the deadline. No new fills (n_fills_now still 4).
    _arm_block(
        state, settings, tox_snap=_tox_one_sided_snap(), n_fills_now=4, now_mono=1050.0
    )
    # Expired and not re-armed.
    assert state.adaptive_spread_widen_until_mono == pytest.approx(1030.0)


def test_expired_with_new_fill_rearms() -> None:
    """After expiry + at least one new fill, a still-adverse signal legitimately
    re-arms. This is the correct adaptive behaviour."""
    state = _ShellState()
    settings = _ShellSettings()
    _arm_block(
        state, settings, tox_snap=_tox_one_sided_snap(), n_fills_now=4, now_mono=1000.0
    )
    # Expire + new fill.
    _arm_block(
        state, settings, tox_snap=_tox_one_sided_snap(), n_fills_now=5, now_mono=1050.0
    )
    # Re-armed to now + 30.
    assert state.adaptive_spread_widen_until_mono == pytest.approx(1080.0)
    assert state.adaptive_spread_widen_arm_n_fills == 5


def test_qq_signal_does_not_extend_active_widen_either() -> None:
    """The QQ-signal branch must respect the same no-extend invariant — but
    it's allowed to keep the latch sticky for overlay computation."""
    state = _ShellState()
    settings = _ShellSettings()
    # Arm via adverse trigger first.
    _arm_block(
        state, settings, tox_snap=_tox_one_sided_snap(), n_fills_now=4, now_mono=1000.0
    )
    deadline = state.adaptive_spread_widen_until_mono
    # QQ signal fires while widen still active — must NOT extend deadline,
    # but MUST re-latch the QQ flag so the overlay picks it up next tick.
    state.quote_quality_widen_latched = False  # simulate prior clear
    _arm_block(
        state,
        settings,
        tox_snap=_tox_one_sided_snap(ratio=0.0),  # adverse inactive
        n_fills_now=4,
        qq_sig=True,
        now_mono=1005.0,
    )
    assert state.adaptive_spread_widen_until_mono == deadline  # not extended
    assert state.quote_quality_widen_latched is True


def test_regression_20260418_094415_scenario() -> None:
    """Replay the shape of ``tmp/snap_20260418_094415``: 4 fills arrived in
    the first minute, then the bot ran for 19 more minutes with no fills.
    The widen must NOT be armed the whole time — after 30 s it expires and,
    without new fills, stays inactive."""
    state = _ShellState()
    settings = _ShellSettings()
    now = 1000.0
    # Minute 1: 4 fills, one-sided ratio hits 0.75 → arm.
    _arm_block(
        state, settings, tox_snap=_tox_one_sided_snap(0.75), n_fills_now=4, now_mono=now
    )
    assert now + 1e-9 < state.adaptive_spread_widen_until_mono  # armed
    # Minutes 2-20: no fills, signal still True from stale window.
    # Simulate 20 minutes of ticks at 0.5 s cadence.
    for i in range(1, 2400):
        _arm_block(
            state,
            settings,
            tox_snap=_tox_one_sided_snap(0.75),
            n_fills_now=4,
            now_mono=now + i * 0.5,
        )
    # After 20 minutes (1200 s), the widen should have expired long ago
    # (deadline was ~1030) and not re-armed because n_fills stayed at 4.
    final_now = now + 2400 * 0.5
    assert state.adaptive_spread_widen_until_mono <= 1030.0
    assert final_now > state.adaptive_spread_widen_until_mono  # widen inactive


def test_gate_survives_beyond_deque_maxlen() -> None:
    """The real implementation uses ``session_fill_count`` (monotonic, never
    caps) rather than ``len(recent_fills)`` (capped at deque maxlen=200).
    This simulates a long session that has surpassed the deque maxlen and
    shows that the gate still fires legitimate re-arms on each new fill.
    """
    state = _ShellState()
    settings = _ShellSettings()
    # Simulate: 250 session fills so far (past maxlen=200).
    now = 1000.0
    _arm_block(
        state, settings, tox_snap=_tox_one_sided_snap(0.75), n_fills_now=250, now_mono=now
    )
    assert state.adaptive_spread_widen_arm_n_fills == 250
    first_deadline = state.adaptive_spread_widen_until_mono
    # Expire. At n=250 session (deque still reports 200 because of cap),
    # no new fills — must NOT re-arm.
    _arm_block(
        state,
        settings,
        tox_snap=_tox_one_sided_snap(0.75),
        n_fills_now=250,
        now_mono=now + 40.0,
    )
    assert state.adaptive_spread_widen_until_mono == first_deadline
    # Now a new fill lands: session counter grows to 251. Re-arm.
    _arm_block(
        state,
        settings,
        tox_snap=_tox_one_sided_snap(0.75),
        n_fills_now=251,
        now_mono=now + 45.0,
    )
    assert state.adaptive_spread_widen_until_mono > first_deadline
    assert state.adaptive_spread_widen_arm_n_fills == 251
