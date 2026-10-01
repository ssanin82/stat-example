"""Tests for ``app.post_swing_gate`` — the post-swing PnL cooldown.

Targets the snapshot 260510064549 incident (00:59-01:05 UTC peak →
give-back) where a +$0.50 PnL spike then drop-back happened over
~6 minutes. The gate should fire on the spike, hold cooldown
through the next leg, and reset cleanly afterwards.
"""

from __future__ import annotations

from app.post_swing_gate import (
    PostSwingState,
    is_active,
    observe,
    seconds_remaining,
)


def _kw(**overrides):
    base = {
        "window_seconds": 60.0,
        "pnl_delta_usd_threshold": 0.50,
        "cooldown_seconds": 120.0,
    }
    base.update(overrides)
    return base


def test_initial_state_inactive() -> None:
    s = PostSwingState()
    assert not is_active(s, now_mono=0.0)
    assert seconds_remaining(s, now_mono=0.0) == 0.0
    assert s.fire_count == 0
    assert s.last_trigger_reason is None


def test_warmup_does_not_fire_on_first_sample() -> None:
    """Single sample isn't a delta — gate shouldn't fire even if
    the value happens to be huge. The buffer needs to actually
    cover ``window_seconds`` before a trigger is meaningful."""
    s = PostSwingState()
    observe(s, now_mono=0.0, total_pnl_usd=10.0, **_kw())
    assert not is_active(s, now_mono=0.1)
    assert s.fire_count == 0


def test_warmup_does_not_fire_within_window() -> None:
    """Even multiple samples within the window shouldn't fire — we
    require a sample older than ``window_seconds`` to anchor the
    measurement. Without that guard, a fresh-start session with
    sparse samples could spuriously fire."""
    s = PostSwingState()
    observe(s, now_mono=0.0, total_pnl_usd=0.0, **_kw())
    observe(s, now_mono=10.0, total_pnl_usd=0.5, **_kw())
    observe(s, now_mono=30.0, total_pnl_usd=1.0, **_kw())  # +$1 in 30s
    assert not is_active(s, now_mono=30.0)


def test_fires_on_spike_after_window_warmup() -> None:
    """The realistic case: bot has been running, samples spread over
    >60s, a $0.50+ spike happens. Trigger fires."""
    s = PostSwingState()
    # Warmup: 70 seconds of flat PnL.
    for t in range(0, 70, 5):
        observe(s, now_mono=float(t), total_pnl_usd=0.0, **_kw())
    # Spike: PnL jumps 0.6 in the next sample.
    observe(s, now_mono=72.0, total_pnl_usd=0.6, **_kw())
    assert is_active(s, now_mono=72.0)
    assert s.last_trigger_reason == "spike"
    assert abs(s.last_trigger_delta_usd - 0.6) < 1e-6
    assert s.fire_count == 1


def test_fires_on_drop_after_window_warmup() -> None:
    """Symmetric: a $0.50+ drop fires the gate too. Targets the
    01:01-01:05 give-back leg where session PnL fell from +$1.23
    to +$0.02 in ~4 minutes."""
    s = PostSwingState()
    for t in range(0, 70, 5):
        observe(s, now_mono=float(t), total_pnl_usd=1.2, **_kw())
    observe(s, now_mono=72.0, total_pnl_usd=0.5, **_kw())
    assert is_active(s, now_mono=72.0)
    assert s.last_trigger_reason == "drop"
    assert s.last_trigger_delta_usd < 0


def test_cooldown_expires_and_state_reusable() -> None:
    """Once the cooldown deadline passes, ``is_active`` returns
    False and a fresh trigger can fire again."""
    s = PostSwingState()
    for t in range(0, 70, 5):
        observe(s, now_mono=float(t), total_pnl_usd=0.0, **_kw(cooldown_seconds=10.0))
    observe(s, now_mono=72.0, total_pnl_usd=0.6, **_kw(cooldown_seconds=10.0))
    assert is_active(s, now_mono=72.0)
    # Cooldown deadline = 72 + 10 = 82.
    assert is_active(s, now_mono=80.0)
    assert not is_active(s, now_mono=82.5)


def test_does_not_re_fire_during_cooldown() -> None:
    """A second swing while the cooldown is active should not
    re-arm the cooldown deadline. The point is "pause then resume",
    not "stay paused indefinitely on continued movement"."""
    s = PostSwingState()
    for t in range(0, 70, 5):
        observe(s, now_mono=float(t), total_pnl_usd=0.0, **_kw())
    observe(s, now_mono=72.0, total_pnl_usd=0.6, **_kw())  # First trigger
    deadline_before = s.cooldown_until_mono
    fire_count_before = s.fire_count
    # Another big swing while cooldown is active:
    observe(s, now_mono=80.0, total_pnl_usd=1.5, **_kw())
    assert s.cooldown_until_mono == deadline_before
    assert s.fire_count == fire_count_before


def test_below_threshold_does_not_fire() -> None:
    """A swing smaller than the configured threshold should not
    fire even with a fully-warmed buffer."""
    s = PostSwingState()
    for t in range(0, 70, 5):
        observe(s, now_mono=float(t), total_pnl_usd=0.0, **_kw())
    observe(s, now_mono=72.0, total_pnl_usd=0.30, **_kw())  # below 0.50
    assert not is_active(s, now_mono=72.0)
    assert s.fire_count == 0


def test_buffer_is_bounded() -> None:
    """The rolling buffer must evict samples older than
    ``window_seconds`` so memory stays bounded over a long session."""
    s = PostSwingState()
    # Simulate 1 hour of 1Hz samples (3600 of them).
    for t in range(0, 3600):
        observe(s, now_mono=float(t), total_pnl_usd=0.0, **_kw(window_seconds=60.0))
    # At 60s window with 1Hz sampling, buffer should hold ~60 samples.
    assert len(s.samples) <= 70  # some slack for boundary timing


# ---------------------------------------------------------------------------
# Phase 2K.4 — favorable-exit predicate
# ---------------------------------------------------------------------------


def _arm_swing(s: PostSwingState, *, cooldown_seconds: float = 120.0) -> float:
    """Drive the gate through one full arm cycle: warmup samples
    then a swing that fires. Returns the mono time when cooldown
    started. Dwell tests pass a larger cooldown so the ceiling
    doesn't pre-empt the favorable-exit path."""
    kw = _kw(cooldown_seconds=cooldown_seconds)
    for t in range(0, 70, 5):
        observe(s, now_mono=float(t), total_pnl_usd=0.0, **kw)
    observe(s, now_mono=72.0, total_pnl_usd=0.6, **kw)
    assert is_active(s, now_mono=72.0), "test setup: gate should have fired"
    return 72.0


def test_phase2k4_clears_early_when_delta_shrinks_for_dwell() -> None:
    """Active cooldown clears EARLY when the rolling PnL-delta drops
    below ``threshold * clear_band_mult`` and stays there for
    ``dwell_seconds``. With defaults (threshold 0.50, mult 0.5,
    dwell 15s), the clear band is 0.25. Uses a long cooldown
    (600 s) so the ceiling can't pre-empt the favorable-exit
    path during the test's manual timing."""
    s = PostSwingState()
    kw = _kw(cooldown_seconds=600.0)
    fire_t = _arm_swing(s, cooldown_seconds=600.0)
    fav_before = s.cleared_via_favorable_total
    # Stay at +0.6 — old 0.0 samples age out at window=60s.
    observe(s, now_mono=fire_t + 1, total_pnl_usd=0.6, **kw)
    observe(s, now_mono=fire_t + 5, total_pnl_usd=0.6, **kw)
    observe(s, now_mono=fire_t + 65, total_pnl_usd=0.6, **kw)
    # Now delta = 0. Dwell timer should have started; gate still active.
    assert is_active(s, now_mono=fire_t + 65)
    assert s.favorable_dwell_started_mono is not None
    # Wait out the 15s dwell.
    observe(s, now_mono=fire_t + 80, total_pnl_usd=0.6, **kw)
    # Dwell satisfied — favorable exit fires.
    assert not is_active(s, now_mono=fire_t + 80)
    assert s.cleared_via_favorable_total == fav_before + 1
    assert s.cleared_via_ceiling_total == 0
    assert s.favorable_dwell_started_mono is None


def test_phase2k4_re_swing_during_dwell_resets_timer() -> None:
    """If the PnL-delta re-widens past the clear band mid-dwell,
    the dwell timer resets. Long cooldown (600 s) prevents ceiling
    interference."""
    s = PostSwingState()
    kw = _kw(cooldown_seconds=600.0)
    fire_t = _arm_swing(s, cooldown_seconds=600.0)
    for off in (1, 5, 65):
        observe(s, now_mono=fire_t + off, total_pnl_usd=0.6, **kw)
    assert s.favorable_dwell_started_mono is not None
    dwell_started = s.favorable_dwell_started_mono
    # PnL re-spikes by 0.3 — delta past clear band (0.25).
    observe(s, now_mono=fire_t + 72, total_pnl_usd=0.9, **kw)
    assert s.favorable_dwell_started_mono is None
    # PnL settles at 0.9; old 0.6 samples age out by t=fire+135
    # (the t=72 sample is the latest 0.6, so window evicts it at
    # 72+60=132).
    for off in (75, 80, 135):
        observe(s, now_mono=fire_t + off, total_pnl_usd=0.9, **kw)
    assert s.favorable_dwell_started_mono is not None
    assert s.favorable_dwell_started_mono > dwell_started
    # Gate still active (dwell hasn't completed; long cooldown holds).
    assert is_active(s, now_mono=fire_t + 135)


def test_phase2k4_ceiling_fires_when_delta_stays_elevated() -> None:
    """When PnL keeps swinging beyond the clear band through the
    full 120 s cooldown, the gate clears via the MAX-cooldown
    ceiling. Attribution counter increments on the active→cleared
    edge regardless of any immediate re-arming after."""
    s = PostSwingState()
    fire_t = _arm_swing(s)  # 120 s default cooldown
    # Alternate 0.0 ↔ 0.7 every 5s — keeps delta at 0.7 (well above
    # the 0.25 clear band) the whole time.
    for off in range(5, 121, 5):
        pnl = 0.7 if (off // 5) % 2 else 0.0
        observe(s, now_mono=fire_t + off, total_pnl_usd=pnl, **_kw())
    # Just past 120 s deadline — single observation flushes the
    # active→cleared edge and attributes to ceiling. The gate may
    # immediately re-arm on the next swing (the wide window still
    # has alternating samples), but that's a fresh cooldown cycle;
    # we only check the counter for this one cycle.
    observe(s, now_mono=fire_t + 130, total_pnl_usd=0.0, **_kw())
    assert s.cleared_via_ceiling_total == 1
    assert s.cleared_via_favorable_total == 0


def test_phase2k4_disabled_mult_uses_pure_timer() -> None:
    """``clear_band_mult=0`` disables the favorable-exit predicate."""
    s = PostSwingState()
    kw = _kw()
    kw["clear_band_mult"] = 0.0
    for t in range(0, 70, 5):
        observe(s, now_mono=float(t), total_pnl_usd=0.0, **kw)
    observe(s, now_mono=72.0, total_pnl_usd=0.6, **kw)
    assert is_active(s, now_mono=72.0)
    for off in (5, 65, 90, 100):
        observe(s, now_mono=72.0 + off, total_pnl_usd=0.6, **kw)
    # Gate must still be active — mult=0 disabled early-exit.
    assert is_active(s, now_mono=72.0 + 100)
    assert s.cleared_via_favorable_total == 0
    assert s.favorable_dwell_started_mono is None
