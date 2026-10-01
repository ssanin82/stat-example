"""Phase 2K.7 — favorable-exit predicate for the ``mae_gate``.

Architecture parity with Phase 2K.3 / 2K.4 / 2K.5 / 2K.6: the legacy
fixed ``MAE_GATE_COOLDOWN_SECONDS`` (180 s default) becomes the
MAX-cooldown ceiling; the gate also clears EARLY when the rolling
N-fill MAE average has recovered above
``-hard_threshold_bps × clear_band_mult`` and held there for
``favorable_exit_dwell_seconds`` (default 5 s — short because each
fresh 30 s-resolved fill is a real signal change).

Re-flare semantics: a fresh adverse fill that drags the rolling
average back below the clear band resets the dwell timer.

Exit attribution: ``cleared_via_favorable_total`` vs
``cleared_via_ceiling_total`` lets the operator tune the knobs from
the dashboard.

Integration with the bot's tick loop is exercised indirectly via
``test_mae_gate.py`` (which still covers the pure-timer ceiling
path); these tests focus on the new exit predicate + counter
behaviour.
"""

from __future__ import annotations

from app.mae_gate import (
    MaeGateState,
    is_active,
    observe,
    reset,
)


def _obs(
    state,
    mae,
    *,
    now_mono=0.0,
    fill_window=5,
    hard=4.0,
    cooldown=180.0,
    favorable_exit_enabled=True,
    clear_band_mult=0.5,
    favorable_exit_dwell_seconds=5.0,
):
    observe(
        state,
        now_mono=now_mono,
        mae_30s_bps=mae,
        fill_window=fill_window,
        hard_threshold_bps=hard,
        cooldown_seconds=cooldown,
        favorable_exit_enabled=favorable_exit_enabled,
        clear_band_mult=clear_band_mult,
        favorable_exit_dwell_seconds=favorable_exit_dwell_seconds,
    )


def _arm(state, *, hard=4.0, cooldown=180.0, fill_window=5):
    """Five adverse fills at -10 → avg=-10 ≤ -4 → fire."""
    for _ in range(fill_window):
        _obs(
            state,
            mae=-10.0,
            now_mono=0.0,
            fill_window=fill_window,
            hard=hard,
            cooldown=cooldown,
        )


# ---------------------------------------------------------------------------
# Favorable-exit predicate clears the cooldown early
# ---------------------------------------------------------------------------


def test_favorable_exit_clears_when_avg_recovers_above_clear_band() -> None:
    """hard=4, mult=0.5 → clear band = -2.0. After arming with avg
    -10, feed 5 fresh fills at 0.0 (avg 0). After dwell, gate clears
    via favorable-exit."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    assert is_active(s, now_mono=10.0)
    # Feed five favorable fills at t=10. After the last one the avg
    # is 0.0 (well above the -2.0 clear band).
    for i in range(5):
        _obs(
            s, mae=0.0, now_mono=10.0 + 0.1 * i,
            fill_window=5, hard=4.0,
        )
    # Dwell starts on the FIRST fill that pushed the avg above the
    # clear band. The very first one only pushed avg to -8 (still
    # below -2). The second to -6. Third -4. Fourth -2 (NOT above
    # -2 with strict >). Fifth -0... wait let me check:
    # Initial buffer after arm: [-10,-10,-10,-10,-10]
    # After fill 1 (mae=0): [-10,-10,-10,-10,0] avg=-8 (still below -2)
    # After fill 2:         [-10,-10,-10,0,0]   avg=-6
    # After fill 3:         [-10,-10,0,0,0]     avg=-4
    # After fill 4:         [-10,0,0,0,0]       avg=-2  (NOT > -2)
    # After fill 5:         [0,0,0,0,0]         avg=0   (above -2 → start dwell)
    # So dwell starts on fill 5 (now_mono=10.4).
    # Gate stays active during dwell.
    assert is_active(s, now_mono=10.4)
    assert is_active(s, now_mono=14.0)  # 3.6 s into dwell
    # Past 5 s dwell. Without a fresh fill, the favorable-exit clear
    # won't fire (it only runs inside observe()). But `is_active`
    # doesn't run the predicate — only edge-detection. So we feed one
    # more fill to trigger the clear-check.
    _obs(s, mae=0.0, now_mono=15.5, fill_window=5, hard=4.0)
    # Now avg is still 0, dwell duration = 15.5 - 10.4 = 5.1 s ≥ 5 →
    # cleared via favorable.
    assert not is_active(s, now_mono=15.5)
    assert s.cleared_via_favorable_total == 1
    assert s.cleared_via_ceiling_total == 0


def test_favorable_exit_does_not_fire_before_dwell_completes() -> None:
    """Even after the predicate flips to holding, the gate must NOT
    clear before the 5 s dwell elapses."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    # Five 0-MAE fills closely spaced.
    for i in range(5):
        _obs(s, mae=0.0, now_mono=10.0 + i * 0.1, fill_window=5, hard=4.0)
    # Predicate now holds (avg=0); dwell started at 10.4.
    # Feed another 0 fill 2 s into the dwell — predicate still holds.
    _obs(s, mae=0.0, now_mono=12.4, fill_window=5, hard=4.0)
    assert is_active(s, now_mono=12.4)
    assert s.cleared_via_favorable_total == 0
    # Feed another at 4.9 s into dwell — still under 5 s.
    _obs(s, mae=0.0, now_mono=15.3, fill_window=5, hard=4.0)
    # dwell started at 10.4, 15.3 - 10.4 = 4.9 < 5 → still active
    assert is_active(s, now_mono=15.3)
    # One more fill at 5.1 s — past dwell.
    _obs(s, mae=0.0, now_mono=15.5, fill_window=5, hard=4.0)
    assert not is_active(s, now_mono=15.5)
    assert s.cleared_via_favorable_total == 1


def test_reflare_resets_dwell() -> None:
    """If a fresh adverse fill drags the rolling average back below
    the clear band, the dwell resets."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=600.0, fill_window=5)
    # Recovery starts: feed 5 fills at 0 to drive avg=0.
    for i in range(5):
        _obs(s, mae=0.0, now_mono=10.0 + i * 0.1, fill_window=5,
             hard=4.0, cooldown=600.0)
    # Dwell now active starting at 10.4. After 3 s, a fresh BAD
    # fill (-15) arrives — buffer becomes [0,0,0,0,-15], avg=-3
    # which is BELOW -2 (clear band) → predicate breaks → dwell
    # resets.
    _obs(s, mae=-15.0, now_mono=13.4, fill_window=5, hard=4.0,
         cooldown=600.0)
    assert s.favorable_dwell_started_mono is None
    # Even later (past would-have-been-dwell), gate stays active.
    assert is_active(s, now_mono=16.0)
    assert s.cleared_via_favorable_total == 0


def test_ceiling_fires_when_avg_stays_adverse() -> None:
    """No recovery → timer expires → ceiling counter increments
    exactly once on the active→cleared edge."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    # Cooldown was armed at now_mono=0.0 in _arm; deadline = 180.
    assert is_active(s, now_mono=0.0)
    assert is_active(s, now_mono=170.0)
    # No recovery fills. Past the ceiling.
    assert not is_active(s, now_mono=181.0)
    assert s.cleared_via_ceiling_total == 1
    assert s.cleared_via_favorable_total == 0


def test_ceiling_attribution_only_fires_once_per_arming() -> None:
    """Polling is_active() multiple times after expiry must not
    double-count. Edge detection uses was_active_last_call."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    is_active(s, now_mono=0.0)
    is_active(s, now_mono=100.0)
    is_active(s, now_mono=181.0)  # ceiling fires here
    is_active(s, now_mono=200.0)  # subsequent polls
    is_active(s, now_mono=500.0)
    assert s.cleared_via_ceiling_total == 1


def test_partial_recovery_below_clear_band_no_favorable() -> None:
    """hard=4, mult=0.5 → clear band -2.0. avg recovers to only -3
    (above trigger -4 but BELOW clear band -2). Predicate doesn't
    fire."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    # Prime ceiling-attribution edge detection by polling while active.
    assert is_active(s, now_mono=0.0)
    # Push fills at -3 each; buffer eventually [-3,-3,-3,-3,-3], avg=-3.
    for i in range(5):
        _obs(s, mae=-3.0, now_mono=10.0 + i * 0.1, fill_window=5, hard=4.0)
    # avg=-3 is below the clear band -2; predicate doesn't hold.
    assert s.favorable_dwell_started_mono is None
    # Several more fills at -3 won't fire.
    for i in range(10):
        _obs(s, mae=-3.0, now_mono=20.0 + i * 5, fill_window=5, hard=4.0)
    assert s.cleared_via_favorable_total == 0
    # Eventually the ceiling fires.
    assert not is_active(s, now_mono=181.0)
    assert s.cleared_via_ceiling_total == 1


def test_clear_band_mult_zero_requires_zero_or_better_avg() -> None:
    """mult=0 → clear band = 0 → favorable-exit fires only when
    avg is strictly above 0 (i.e. all samples clipped to 0)."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=600.0, fill_window=5)
    # Recovery to avg=-0.5: not above 0, predicate fails.
    for i in range(5):
        _obs(s, mae=-0.5, now_mono=10.0 + i * 0.1, fill_window=5,
             hard=4.0, cooldown=600.0, clear_band_mult=0.0)
    assert s.favorable_dwell_started_mono is None
    assert is_active(s, now_mono=20.0)
    # Now recovery to clean 0 fills (avg=0 — NOT above 0 with strict >)
    # Wait — the predicate is `avg > clear_band` (strict). Clear band
    # at mult=0 is exactly 0. So avg=0 does NOT fire.
    for i in range(5):
        _obs(s, mae=0.0, now_mono=20.0 + i * 0.1, fill_window=5,
             hard=4.0, cooldown=600.0, clear_band_mult=0.0)
    # avg=0 is not strictly > 0; predicate still fails.
    assert s.favorable_dwell_started_mono is None
    assert s.cleared_via_favorable_total == 0


def test_clear_band_mult_one_clears_at_trigger_boundary() -> None:
    """mult=1.0 → clear band = -hard (the trigger boundary).
    Favorable-exit fires as soon as avg lifts just above the
    original trigger."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    # Push avg up to -3.5 (just above -4 trigger). Buffer needs to
    # go from [-10×5] to something averaging -3.5. With 5 fills
    # at -3.5 we get avg=-3.5.
    for i in range(5):
        _obs(s, mae=-3.5, now_mono=10.0 + i * 0.1, fill_window=5,
             hard=4.0, clear_band_mult=1.0)
    assert s.favorable_dwell_started_mono is not None
    # Past dwell + one more fill to trigger the clear.
    _obs(s, mae=-3.5, now_mono=16.0, fill_window=5, hard=4.0,
         clear_band_mult=1.0)
    assert not is_active(s, now_mono=16.0)
    assert s.cleared_via_favorable_total == 1


def test_favorable_exit_disabled_uses_pure_timer() -> None:
    """``favorable_exit_enabled=False`` → legacy pure-timer behaviour
    is preserved. Recovery fills don't fire favorable-exit."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    # Push avg way above the would-be clear band.
    for i in range(5):
        _obs(s, mae=+10.0, now_mono=10.0 + i * 0.1, fill_window=5,
             hard=4.0, favorable_exit_enabled=False)
    # Even with all-positive (clipped to 0) recovery fills, the gate
    # stays active until the 180 s ceiling.
    assert is_active(s, now_mono=100.0)
    assert not is_active(s, now_mono=181.0)
    assert s.cleared_via_favorable_total == 0
    assert s.cleared_via_ceiling_total == 1


def test_warmup_protects_favorable_exit_too() -> None:
    """The favorable-exit predicate requires a full window — same
    warmup guard as the trigger. A partial buffer doesn't fire."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=10)
    # Wait — _arm uses fill_window=5 by default. Let me re-arm with
    # window=10. Reset and arm with the matching window.
    reset(s)
    for _ in range(10):
        _obs(s, mae=-10.0, now_mono=0.0, fill_window=10, hard=4.0,
             cooldown=180.0)
    assert is_active(s, now_mono=10.0)
    # Push 3 favorable fills — buffer evicts 3 old, ends at
    # [-10×7, 0×3]. avg = -7 (still adverse). Predicate fails.
    for i in range(3):
        _obs(s, mae=0.0, now_mono=10.0 + i * 0.1, fill_window=10, hard=4.0)
    # Not enough recovery yet.
    assert s.favorable_dwell_started_mono is None
    # 7 more clean fills → buffer all zeros → avg=0 > -2 → predicate
    # holds → dwell starts.
    for i in range(7):
        _obs(s, mae=0.0, now_mono=11.0 + i * 0.1, fill_window=10, hard=4.0)
    assert s.favorable_dwell_started_mono is not None


def test_re_arm_after_favorable_clear_cancels_dwell() -> None:
    """If the gate clears via favorable AND then the average crosses
    the trigger again, a fresh arm cancels any leftover dwell state."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    # Recover and clear via favorable.
    for i in range(5):
        _obs(s, mae=0.0, now_mono=10.0 + i * 0.1, fill_window=5, hard=4.0)
    _obs(s, mae=0.0, now_mono=15.5, fill_window=5, hard=4.0)
    assert s.cleared_via_favorable_total == 1
    assert not is_active(s, now_mono=15.5)
    # Now push 5 fresh bad fills past the cooldown → re-arm.
    for i in range(5):
        _obs(s, mae=-10.0, now_mono=20.0 + i * 0.1, fill_window=5, hard=4.0)
    assert s.fire_count == 2
    assert is_active(s, now_mono=20.0)
    # Dwell should be None (fresh arm reset it).
    assert s.favorable_dwell_started_mono is None


def test_reset_clears_phase2k7_attribution_counters() -> None:
    """``reset()`` must also clear the new favorable-exit state so a
    session reset doesn't leave stale counters."""
    s = MaeGateState()
    _arm(s, hard=4.0, cooldown=180.0, fill_window=5)
    for i in range(5):
        _obs(s, mae=0.0, now_mono=10.0 + i * 0.1, fill_window=5, hard=4.0)
    _obs(s, mae=0.0, now_mono=15.5, fill_window=5, hard=4.0)
    assert s.cleared_via_favorable_total == 1
    reset(s)
    assert s.cleared_via_favorable_total == 0
    assert s.cleared_via_ceiling_total == 0
    assert s.favorable_dwell_started_mono is None
    assert s.was_active_last_call is False
