"""Phase 2K.8 — favorable-exit predicate for the ``vol_spike`` latch.

Architecture parity with prior Phase 2K items:
``VOL_SPIKE_COOLDOWN_SECONDS`` becomes the MAX-cooldown ceiling; the
``evaluate_vol_spike_favorable_exit`` predicate clears the latch
EARLY when ``vol_ratio < VOL_SPIKE_THRESHOLD × clear_band_mult`` for
``favorable_exit_dwell_seconds``.

Pure-function tests; the bot.py wiring is exercised indirectly by
the existing ``test_vol_regime.py`` regression suite (still passing
unchanged).
"""

from __future__ import annotations

from app.vol_regime import (
    VolSpikeExitResult,
    evaluate_vol_spike_favorable_exit,
)
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides):
    base = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "grvt",
        "SYMBOL": "ETH_USDT_Perp",
        "VOL_SPIKE_THRESHOLD": 2.5,
        "VOL_SPIKE_COOLDOWN_SECONDS": 60.0,
        "VOL_SPIKE_FAVORABLE_EXIT_ENABLED": True,
        "VOL_SPIKE_CLEAR_BAND_MULT": 0.7,
        "VOL_SPIKE_FAVORABLE_EXIT_DWELL_SECONDS": 5.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _arm_latch(
    settings, *, at_mono: float = 0.0
) -> tuple[float, bool]:
    """Simulate arming the spike latch — return ``(until_mono,
    was_active_last_call=True)`` as if a previous tick had armed it."""
    cooldown = float(settings.vol_spike_cooldown_seconds)
    return (at_mono + cooldown, True)


# ---------------------------------------------------------------------------
# Favorable-exit: vol calms past band → latch clears
# ---------------------------------------------------------------------------


def test_favorable_exit_clears_when_vol_calms_past_band() -> None:
    """threshold=2.5, mult=0.7 → clear band 1.75. Drive vol_ratio
    below 1.75 for the dwell duration; latch clears early."""
    s = _settings()
    until, was_active = _arm_latch(s, at_mono=0.0)
    # t=10: vol_ratio=1.5 (below clear band 1.75) → start dwell.
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=1.5,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    assert r1.cleared_via == "none"
    assert r1.new_favorable_dwell_started_mono == 10.0
    assert r1.new_was_active_last_call is True
    # t=14: still calm, 4 s into dwell.
    r2 = evaluate_vol_spike_favorable_exit(
        s, now_mono=14.0, vol_ratio=1.4,
        vol_spike_until_mono=r1.new_until_mono,
        favorable_dwell_started_mono=r1.new_favorable_dwell_started_mono,
        was_active_last_call=r1.new_was_active_last_call,
    )
    assert r2.cleared_via == "none"
    assert r2.new_was_active_last_call is True  # still in spike
    # t=15.5: past 5 s dwell → clears via favorable.
    r3 = evaluate_vol_spike_favorable_exit(
        s, now_mono=15.5, vol_ratio=1.3,
        vol_spike_until_mono=r2.new_until_mono,
        favorable_dwell_started_mono=r2.new_favorable_dwell_started_mono,
        was_active_last_call=r2.new_was_active_last_call,
    )
    assert r3.cleared_via == "favorable"
    assert r3.new_until_mono == 0.0
    assert r3.new_favorable_dwell_started_mono is None
    assert r3.new_was_active_last_call is False


def test_favorable_exit_does_not_fire_before_dwell_completes() -> None:
    """Dwell must hold for the full configured duration."""
    s = _settings()
    until, was_active = _arm_latch(s, at_mono=0.0)
    r = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=1.0,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    for t in (11.0, 13.0, 14.9):
        r = evaluate_vol_spike_favorable_exit(
            s, now_mono=t, vol_ratio=1.0,
            vol_spike_until_mono=r.new_until_mono,
            favorable_dwell_started_mono=(
                r.new_favorable_dwell_started_mono
            ),
            was_active_last_call=r.new_was_active_last_call,
        )
        assert r.cleared_via == "none"
        assert r.new_was_active_last_call is True
    # Just past 5 s dwell.
    r_final = evaluate_vol_spike_favorable_exit(
        s, now_mono=15.1, vol_ratio=1.0,
        vol_spike_until_mono=r.new_until_mono,
        favorable_dwell_started_mono=r.new_favorable_dwell_started_mono,
        was_active_last_call=r.new_was_active_last_call,
    )
    assert r_final.cleared_via == "favorable"


def test_reflare_resets_dwell() -> None:
    """If vol_ratio re-crosses the clear band mid-dwell, the dwell
    timer resets and the latch stays active for the full ceiling."""
    s = _settings()
    until, was_active = _arm_latch(s, at_mono=0.0)
    # Start dwell at t=10.
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=1.0,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    assert r1.new_favorable_dwell_started_mono == 10.0
    # Re-flare at t=12: vol shoots back to 2.0 (above 1.75 clear
    # band but below 2.5 trigger — doesn't re-arm, just resets dwell).
    r2 = evaluate_vol_spike_favorable_exit(
        s, now_mono=12.0, vol_ratio=2.0,
        vol_spike_until_mono=r1.new_until_mono,
        favorable_dwell_started_mono=r1.new_favorable_dwell_started_mono,
        was_active_last_call=r1.new_was_active_last_call,
    )
    assert r2.new_favorable_dwell_started_mono is None
    assert r2.cleared_via == "none"
    # At t=15 (would have been past original dwell), still active.
    r3 = evaluate_vol_spike_favorable_exit(
        s, now_mono=15.0, vol_ratio=1.0,
        vol_spike_until_mono=r2.new_until_mono,
        favorable_dwell_started_mono=r2.new_favorable_dwell_started_mono,
        was_active_last_call=r2.new_was_active_last_call,
    )
    # New dwell started at 15 — predicate holds again.
    assert r3.new_favorable_dwell_started_mono == 15.0
    assert r3.cleared_via == "none"


def test_ceiling_fires_when_vol_stays_elevated() -> None:
    """No recovery → cooldown expires naturally → ceiling counter
    increments on the active→cleared edge."""
    s = _settings()
    until, was_active = _arm_latch(s, at_mono=0.0)
    # vol stays elevated throughout (above clear band 1.75).
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=30.0, vol_ratio=2.0,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    assert r1.cleared_via == "none"
    assert r1.new_was_active_last_call is True
    # Past the 60 s ceiling.
    r2 = evaluate_vol_spike_favorable_exit(
        s, now_mono=61.0, vol_ratio=2.0,
        vol_spike_until_mono=r1.new_until_mono,
        favorable_dwell_started_mono=r1.new_favorable_dwell_started_mono,
        was_active_last_call=r1.new_was_active_last_call,
    )
    assert r2.cleared_via == "ceiling"
    assert r2.new_was_active_last_call is False


def test_partial_recovery_above_clear_band_no_favorable() -> None:
    """vol drops below trigger 2.5 but stays above clear band 1.75:
    not enough recovery → predicate doesn't hold → ceiling fires."""
    s = _settings()
    until, was_active = _arm_latch(s, at_mono=0.0)
    # vol_ratio = 2.0 (below trigger, ABOVE clear band 1.75).
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=2.0,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    assert r1.new_favorable_dwell_started_mono is None
    # Ceiling at 60 s.
    r2 = evaluate_vol_spike_favorable_exit(
        s, now_mono=61.0, vol_ratio=2.0,
        vol_spike_until_mono=r1.new_until_mono,
        favorable_dwell_started_mono=None,
        was_active_last_call=True,
    )
    assert r2.cleared_via == "ceiling"


def test_disabled_falls_back_to_pure_timer() -> None:
    """``VOL_SPIKE_FAVORABLE_EXIT_ENABLED=False`` → legacy behaviour;
    favorable predicate never fires, ceiling attribution still works."""
    s = _settings(VOL_SPIKE_FAVORABLE_EXIT_ENABLED=False)
    until, was_active = _arm_latch(s, at_mono=0.0)
    # vol calms strongly but predicate is disabled.
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=1.0,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    assert r1.cleared_via == "none"
    assert r1.new_favorable_dwell_started_mono is None
    assert r1.new_was_active_last_call is True
    # Ceiling fires.
    r2 = evaluate_vol_spike_favorable_exit(
        s, now_mono=61.0, vol_ratio=1.0,
        vol_spike_until_mono=r1.new_until_mono,
        favorable_dwell_started_mono=None,
        was_active_last_call=True,
    )
    assert r2.cleared_via == "ceiling"


def test_mult_one_clears_at_trigger_boundary() -> None:
    """mult=1.0 → clear band equals trigger 2.5; predicate fires the
    moment vol drops below trigger."""
    s = _settings(VOL_SPIKE_CLEAR_BAND_MULT=1.0)
    until, was_active = _arm_latch(s, at_mono=0.0)
    # vol=2.4 — just below trigger 2.5 → predicate holds.
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=2.4,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    assert r1.new_favorable_dwell_started_mono == 10.0
    # Past dwell.
    r2 = evaluate_vol_spike_favorable_exit(
        s, now_mono=15.5, vol_ratio=2.4,
        vol_spike_until_mono=r1.new_until_mono,
        favorable_dwell_started_mono=r1.new_favorable_dwell_started_mono,
        was_active_last_call=r1.new_was_active_last_call,
    )
    assert r2.cleared_via == "favorable"


def test_mult_zero_requires_vol_below_one() -> None:
    """mult=0 → clear band = 0 → vol must drop below 0 (impossible)
    so predicate never fires. Pure-timer ceiling effectively."""
    s = _settings(VOL_SPIKE_CLEAR_BAND_MULT=0.0)
    until, was_active = _arm_latch(s, at_mono=0.0)
    # vol completely calm at 0.5 (but clipped to 1.0 in the function
    # for the trigger logic; the favorable-exit predicate sees the
    # raw value).
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=0.5,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    # max(0.0, 0.5) = 0.5; clear_band = 2.5 * 0 = 0.0; 0.5 < 0? No.
    # Predicate doesn't hold.
    assert r1.new_favorable_dwell_started_mono is None


def test_none_vol_ratio_treated_as_calm() -> None:
    """``vol_ratio=None`` defaults to vr=1.0; with threshold 2.5 and
    mult 0.7 → clear_band 1.75. 1.0 < 1.75 → predicate holds."""
    s = _settings()
    until, was_active = _arm_latch(s, at_mono=0.0)
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=None,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=was_active,
    )
    assert r1.new_favorable_dwell_started_mono == 10.0


def test_not_in_spike_no_dwell_no_attribution() -> None:
    """When the latch isn't active and never was, no transition is
    reported and the dwell stays None."""
    s = _settings()
    r = evaluate_vol_spike_favorable_exit(
        s, now_mono=10.0, vol_ratio=1.0,
        vol_spike_until_mono=0.0,
        favorable_dwell_started_mono=None,
        was_active_last_call=False,
    )
    assert r.cleared_via == "none"
    assert r.new_until_mono == 0.0
    assert r.new_favorable_dwell_started_mono is None
    assert r.new_was_active_last_call is False


def test_ceiling_only_fires_once_per_arming() -> None:
    """After the ceiling fires (was=True, now inactive), subsequent
    polls (was=False, now inactive) report ``"none"``."""
    s = _settings()
    until, _ = _arm_latch(s, at_mono=0.0)
    # First call past the ceiling — fires.
    r1 = evaluate_vol_spike_favorable_exit(
        s, now_mono=61.0, vol_ratio=2.0,
        vol_spike_until_mono=until,
        favorable_dwell_started_mono=None,
        was_active_last_call=True,
    )
    assert r1.cleared_via == "ceiling"
    # Next call — no longer active, was=False.
    r2 = evaluate_vol_spike_favorable_exit(
        s, now_mono=62.0, vol_ratio=2.0,
        vol_spike_until_mono=r1.new_until_mono,
        favorable_dwell_started_mono=None,
        was_active_last_call=r1.new_was_active_last_call,
    )
    assert r2.cleared_via == "none"


def test_result_is_frozen_dataclass() -> None:
    """``VolSpikeExitResult`` is frozen — caller can store it in a
    set, hash, or freely pass between threads."""
    r = VolSpikeExitResult(
        new_until_mono=0.0,
        new_favorable_dwell_started_mono=None,
        new_was_active_last_call=False,
        cleared_via="none",
    )
    try:
        r.new_until_mono = 5.0  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised, "VolSpikeExitResult must be immutable"
