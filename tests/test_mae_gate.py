"""Tests for ``app.mae_gate`` — 30s post-fill MAE defensive gate.

The gate observes finalised ``mae_30s_bps`` values as they come off
the post-fill-excursion watcher, maintains a rolling N-fill average
of ``min(0, mae)``, and arms a HOLD_ALL cooldown when the average
crosses below ``-hard_threshold_bps``.

Parallel to the toxicity engine's hard trigger at a different
horizon. The unit tests cover state-machine behaviour only — the
producer wiring (watcher → observe) and consumer wiring (quote loop
→ is_active) are integration concerns.
"""

from __future__ import annotations

from app.mae_gate import (
    MaeGateState,
    is_active,
    observe,
    reset,
    seconds_remaining,
)


def _obs(state, mae, *, now_mono=0.0, fill_window=5, hard=4.0, cooldown=180.0):
    observe(
        state,
        now_mono=now_mono,
        mae_30s_bps=mae,
        fill_window=fill_window,
        hard_threshold_bps=hard,
        cooldown_seconds=cooldown,
    )


def test_starts_inactive() -> None:
    """Fresh state is inactive."""
    s = MaeGateState()
    assert not is_active(s, now_mono=0.0)
    assert seconds_remaining(s, now_mono=0.0) == 0.0
    assert s.fire_count == 0


def test_does_not_fire_until_window_full() -> None:
    """Warmup guard: the buffer must reach ``fill_window`` size
    before any fire is allowed, even if the first sample is brutal."""
    s = MaeGateState()
    _obs(s, mae=-50.0, fill_window=5, hard=4.0)
    assert s.fire_count == 0
    assert not is_active(s, now_mono=0.0)


def test_fires_when_avg_crosses_threshold() -> None:
    """Buffer fills, average crosses threshold → gate arms cooldown."""
    s = MaeGateState()
    # Five fills, all at -6 bp. Average = -6 ≤ -4 → fire.
    for _ in range(5):
        _obs(s, mae=-6.0, fill_window=5, hard=4.0, cooldown=180.0)
    assert s.fire_count == 1
    assert s.last_trigger_avg_bps == -6.0
    assert is_active(s, now_mono=0.0)
    assert is_active(s, now_mono=179.0)
    assert not is_active(s, now_mono=181.0)


def test_does_not_fire_below_threshold() -> None:
    """Window full but average not severe enough → no fire."""
    s = MaeGateState()
    for _ in range(5):
        _obs(s, mae=-3.0, fill_window=5, hard=4.0)
    assert s.fire_count == 0


def test_favourable_mae_clips_to_zero() -> None:
    """Favourable excursion (positive MAE — which shouldn't really
    happen for MAE but the watcher could pass nonzero in edge cases)
    clips to 0, doesn't subtract from the adverse count."""
    s = MaeGateState()
    _obs(s, mae=+5.0, fill_window=5, hard=4.0)
    _obs(s, mae=+5.0, fill_window=5, hard=4.0)
    _obs(s, mae=+5.0, fill_window=5, hard=4.0)
    _obs(s, mae=+5.0, fill_window=5, hard=4.0)
    _obs(s, mae=+5.0, fill_window=5, hard=4.0)
    # All clipped to 0; avg = 0 → no fire.
    assert s.fire_count == 0


def test_none_input_no_op() -> None:
    """``mae_30s_bps=None`` (degenerate watcher case) is a no-op:
    buffer unchanged, no fire."""
    s = MaeGateState()
    for _ in range(10):
        _obs(s, mae=None, fill_window=5, hard=4.0)
    assert len(s.samples) == 0
    assert s.fire_count == 0


def test_window_eviction_keeps_only_recent_samples() -> None:
    """Old samples slide out as new ones come in. After enough clean
    fills, the average rises back above the threshold and the gate
    stops firing — even with a very long cooldown that has already
    expired."""
    s = MaeGateState()
    # Five bad fills with a long cooldown so we don't re-fire on
    # incremental improvements.
    for _ in range(5):
        _obs(s, mae=-10.0, now_mono=0.0, fill_window=5, hard=4.0, cooldown=300.0)
    assert s.fire_count == 1
    # During cooldown, push five clean fills. They append + evict
    # but don't fire (cooldown active).
    for i in range(5):
        _obs(s, mae=0.0, now_mono=10.0 + i, fill_window=5, hard=4.0, cooldown=300.0)
    # Buffer is now [0,0,0,0,0]; cooldown still active so fire_count
    # unchanged.
    assert s.fire_count == 1
    assert list(s.samples) == [0.0, 0.0, 0.0, 0.0, 0.0]
    # Now jump past the cooldown and observe one more clean fill.
    # Average is 0; no fire.
    _obs(s, mae=0.0, now_mono=400.0, fill_window=5, hard=4.0, cooldown=300.0)
    assert s.fire_count == 1


def test_does_not_refire_during_cooldown() -> None:
    """While cooldown is active, even more bad fills don't bump
    fire_count — one cooldown per crossing."""
    s = MaeGateState()
    for _ in range(5):
        _obs(s, mae=-10.0, now_mono=0.0, fill_window=5, hard=4.0, cooldown=180.0)
    assert s.fire_count == 1
    for _ in range(5):
        _obs(s, mae=-20.0, now_mono=50.0, fill_window=5, hard=4.0, cooldown=180.0)
    assert s.fire_count == 1
    assert is_active(s, now_mono=50.0)


def test_can_refire_after_cooldown_expires_and_buffer_recrosses() -> None:
    """After cooldown expires, a fresh bad average can fire again."""
    s = MaeGateState()
    for _ in range(5):
        _obs(s, mae=-10.0, now_mono=0.0, fill_window=5, hard=4.0, cooldown=180.0)
    assert s.fire_count == 1
    # Cooldown expired. Buffer is still full of -10s, so next fill
    # also crosses → can fire again.
    _obs(s, mae=-10.0, now_mono=200.0, fill_window=5, hard=4.0, cooldown=180.0)
    assert s.fire_count == 2


def test_reset_clears_state() -> None:
    """``reset()`` clears buffer, cooldown, fire_count — used at
    session start so a prior session's tail doesn't pollute the new
    one."""
    s = MaeGateState()
    for _ in range(5):
        _obs(s, mae=-10.0, fill_window=5, hard=4.0)
    assert s.fire_count == 1
    assert is_active(s, now_mono=0.0)
    reset(s)
    assert s.fire_count == 0
    assert not is_active(s, now_mono=0.0)
    assert len(s.samples) == 0


def test_seconds_remaining_correct() -> None:
    """``seconds_remaining`` returns wall-equivalent time-to-resume."""
    s = MaeGateState()
    for _ in range(5):
        _obs(s, mae=-10.0, now_mono=100.0, fill_window=5, hard=4.0, cooldown=60.0)
    assert is_active(s, now_mono=130.0)
    assert seconds_remaining(s, now_mono=130.0) == 30.0
    assert seconds_remaining(s, now_mono=160.0) == 0.0
    assert seconds_remaining(s, now_mono=200.0) == 0.0


def test_realistic_260515_095056_regime_fires() -> None:
    """Sanity replay: the snapshot motivating this gate had 30s MAE
    median -6.8 bp BUY, -8.3 bp SELL. A 10-fill window of values in
    that band should fire a 5 bp threshold."""
    s = MaeGateState()
    realistic_mae = [-6.8, -8.3, -5.5, -7.2, -9.0, -6.0, -7.5, -8.0, -6.5, -7.8]
    for v in realistic_mae:
        _obs(s, mae=v, fill_window=10, hard=5.0, cooldown=180.0)
    assert s.fire_count == 1
    assert s.last_trigger_avg_bps < -5.0
