"""Phase 4F (v1.4.170) — elevated-vol auto-pause.

When realised vol stays above the arm threshold for the arm dwell,
the bot's eligibility is forced to HOLD_ALL. Self-clears when vol
normalises (clear dwell) or at the MAX ceiling (safety net).

Pure-function tests for ``evaluate_vol_regime_auto_pause``. The
bot.py wiring (eligibility override + first-arm log + counter
bumps) is exercised indirectly by the broader regression sweep.
"""

from __future__ import annotations

from app.vol_regime_auto_pause import (
    VolRegimeAutoPauseDecision,
    evaluate_vol_regime_auto_pause,
)


def _kw(**overrides):
    base = dict(
        now_mono=100.0,
        vol_spike_ratio=1.0,
        currently_active=False,
        arm_dwell_started_mono=None,
        clear_dwell_started_mono=None,
        active_since_mono=0.0,
        arm_ratio=2.0,
        arm_dwell_seconds=60.0,
        clear_ratio=1.3,
        clear_dwell_seconds=120.0,
        max_pause_seconds=1800.0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Default disabled (arm_ratio=0.0)
# ---------------------------------------------------------------------------


def test_disabled_by_default_returns_passthrough() -> None:
    """``arm_ratio=0.0`` keeps the gate dormant regardless of how
    elevated the vol gets. Preserves pre-v1.4.170 behaviour."""
    r = evaluate_vol_regime_auto_pause(
        **_kw(arm_ratio=0.0, vol_spike_ratio=10.0)
    )
    assert r.new_active is False
    assert r.transition == "none"


def test_missing_signal_returns_passthrough() -> None:
    """Vol-ratio = None (toxicity engine warmup) leaves state
    unchanged. No spurious arming on missing data."""
    r = evaluate_vol_regime_auto_pause(**_kw(vol_spike_ratio=None))
    assert r.transition == "none"
    assert r.new_active is False


def test_nan_signal_returns_passthrough() -> None:
    r = evaluate_vol_regime_auto_pause(**_kw(vol_spike_ratio=float("nan")))
    assert r.transition == "none"
    r2 = evaluate_vol_regime_auto_pause(**_kw(vol_spike_ratio=float("inf")))
    assert r2.transition == "none"


# ---------------------------------------------------------------------------
# Arm path
# ---------------------------------------------------------------------------


def test_arm_requires_dwell_to_elapse() -> None:
    """Vol jumps to 2.5 (above arm=2.0). First tick records dwell
    start but doesn't arm. Next tick after 30 s still below dwell
    (60 s default). 60+ s later: arms."""
    # Tick 1: dwell starts.
    r1 = evaluate_vol_regime_auto_pause(
        **_kw(now_mono=100.0, vol_spike_ratio=2.5)
    )
    assert r1.new_active is False
    assert r1.new_arm_dwell_started_mono == 100.0
    assert r1.transition == "none"
    # Tick 2: 30 s in, still dwelling.
    r2 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=130.0,
            vol_spike_ratio=2.5,
            arm_dwell_started_mono=100.0,
        )
    )
    assert r2.new_active is False
    assert r2.new_arm_dwell_started_mono == 100.0
    # Tick 3: 60 s in, arms.
    r3 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=160.0,
            vol_spike_ratio=2.5,
            arm_dwell_started_mono=100.0,
        )
    )
    assert r3.new_active is True
    assert r3.transition == "armed"
    assert r3.new_active_since_mono == 160.0
    assert r3.new_arm_dwell_started_mono is None


def test_arm_resets_when_vol_drops_back_below_threshold() -> None:
    """Vol spikes briefly to 2.5, then drops back to 1.5 (below arm
    but above clear). Mid-dwell reset → fresh dwell starts on the
    next spike."""
    r1 = evaluate_vol_regime_auto_pause(
        **_kw(now_mono=100.0, vol_spike_ratio=2.5)
    )
    assert r1.new_arm_dwell_started_mono == 100.0
    # Vol drops below arm threshold mid-dwell.
    r2 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=120.0,
            vol_spike_ratio=1.5,
            arm_dwell_started_mono=100.0,
        )
    )
    assert r2.new_active is False
    assert r2.new_arm_dwell_started_mono is None
    # Vol spikes again — fresh dwell timer.
    r3 = evaluate_vol_regime_auto_pause(
        **_kw(now_mono=140.0, vol_spike_ratio=2.5)
    )
    assert r3.new_arm_dwell_started_mono == 140.0


def test_arm_does_not_fire_when_ratio_at_or_below_threshold() -> None:
    """Strict-greater-than check: ratio = arm_ratio exactly = not
    armed (the threshold is "elevated", not "at-or-above")."""
    r = evaluate_vol_regime_auto_pause(
        **_kw(now_mono=300.0, vol_spike_ratio=2.0, arm_ratio=2.0)
    )
    assert r.new_arm_dwell_started_mono is None


# ---------------------------------------------------------------------------
# Favorable-exit clear path
# ---------------------------------------------------------------------------


def test_favorable_exit_clears_after_clear_dwell() -> None:
    """While paused, vol drops below clear_ratio (1.3) → start clear
    dwell. After 120 s → clear via favorable."""
    # Tick 1: bot is paused, vol drops below clear threshold.
    r1 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=500.0,
            vol_spike_ratio=1.1,
            currently_active=True,
            active_since_mono=400.0,
        )
    )
    assert r1.new_active is True
    assert r1.new_clear_dwell_started_mono == 500.0
    assert r1.transition == "none"
    # Tick 2: 60 s in, still dwelling toward clear.
    r2 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=560.0,
            vol_spike_ratio=1.1,
            currently_active=True,
            active_since_mono=400.0,
            clear_dwell_started_mono=500.0,
        )
    )
    assert r2.new_active is True
    # Tick 3: 120+ s, clears.
    r3 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=620.5,
            vol_spike_ratio=1.1,
            currently_active=True,
            active_since_mono=400.0,
            clear_dwell_started_mono=500.0,
        )
    )
    assert r3.new_active is False
    assert r3.transition == "cleared_favorable"


def test_clear_dwell_resets_when_vol_rises_back_above_clear_threshold() -> None:
    """Mid-clear-dwell, vol pops back above clear_ratio → dwell
    resets, pause stays active."""
    r1 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=500.0,
            vol_spike_ratio=1.1,
            currently_active=True,
            active_since_mono=400.0,
        )
    )
    assert r1.new_clear_dwell_started_mono == 500.0
    # Vol re-elevates mid-dwell.
    r2 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=560.0,
            vol_spike_ratio=1.6,
            currently_active=True,
            active_since_mono=400.0,
            clear_dwell_started_mono=500.0,
        )
    )
    assert r2.new_active is True
    assert r2.new_clear_dwell_started_mono is None
    # Vol drops AGAIN → fresh dwell.
    r3 = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=600.0,
            vol_spike_ratio=1.1,
            currently_active=True,
            active_since_mono=400.0,
            clear_dwell_started_mono=None,
        )
    )
    assert r3.new_clear_dwell_started_mono == 600.0


# ---------------------------------------------------------------------------
# MAX-ceiling safety net
# ---------------------------------------------------------------------------


def test_max_ceiling_clears_pause_when_vol_stays_elevated() -> None:
    """Vol stays elevated past max_pause_seconds → ceiling fires,
    pause clears via ceiling (NOT favorable)."""
    # Pause active for 1800 s with vol stuck at 2.5.
    r = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=2200.5,
            vol_spike_ratio=2.5,
            currently_active=True,
            active_since_mono=400.0,
            max_pause_seconds=1800.0,
        )
    )
    assert r.new_active is False
    assert r.transition == "cleared_ceiling"


def test_ceiling_fires_even_during_clear_dwell() -> None:
    """If the clear dwell is in progress AND the ceiling expires at
    the same time, the ceiling wins (safety net is hard ceiling)."""
    r = evaluate_vol_regime_auto_pause(
        **_kw(
            now_mono=2300.0,
            vol_spike_ratio=1.1,
            currently_active=True,
            active_since_mono=400.0,
            clear_dwell_started_mono=2250.0,
            max_pause_seconds=1800.0,
        )
    )
    # clear-dwell hasn't elapsed (50 s < 120 s) but ceiling has fired
    # (1900 s > 1800 s).
    assert r.new_active is False
    assert r.transition == "cleared_ceiling"


# ---------------------------------------------------------------------------
# v1.4.166 snapshot replay
# ---------------------------------------------------------------------------


def test_v1_4_166_snapshot_replay_arms_within_60s() -> None:
    """Replay the snapshot v1.4.166-260521-000907 shape: vol_spike_
    ratio sustained around 2.15 (peak measured at 6.46/3 base ≈ 2.15).
    With ARM_RATIO=2.0 + ARM_DWELL=60 s, the gate arms exactly at the
    60-second mark."""
    # Simulate 5 ticks at vol_ratio = 2.15, 30 s apart.
    state_arm_dwell = None
    armed = False
    armed_at = None
    for i, t in enumerate([0.0, 30.0, 60.0, 90.0, 120.0]):
        r = evaluate_vol_regime_auto_pause(
            **_kw(
                now_mono=t,
                vol_spike_ratio=2.15,
                currently_active=armed,
                arm_dwell_started_mono=state_arm_dwell,
                active_since_mono=armed_at if armed_at else 0.0,
            )
        )
        state_arm_dwell = r.new_arm_dwell_started_mono
        if r.transition == "armed":
            armed = True
            armed_at = r.new_active_since_mono
    assert armed is True, "expected pause to ARM during 5-tick replay"
    # Should have armed at t=60 (the first tick where elapsed >= 60 s).
    assert armed_at == 60.0


# ---------------------------------------------------------------------------
# Result dataclass + naming
# ---------------------------------------------------------------------------


def test_decision_is_frozen() -> None:
    r = VolRegimeAutoPauseDecision(
        new_active=False,
        new_arm_dwell_started_mono=None,
        new_clear_dwell_started_mono=None,
        new_active_since_mono=0.0,
        transition="none",
    )
    try:
        r.new_active = True  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised


def test_helper_does_not_reference_any_specific_exchange() -> None:
    """Per v1.4.165 naming convention: pure-helper modules must not
    contain exchange-specific identifiers."""
    import app.vol_regime_auto_pause as mod
    import inspect

    src = inspect.getsource(mod)
    forbidden = ("binance", "okx", "grvt", "hyperliquid", "bluefin")
    for name in forbidden:
        assert name not in src.lower(), (
            f"Exchange-specific name {name!r} leaked into "
            "vol_regime_auto_pause.py — must stay exchange-agnostic."
        )


# ---------------------------------------------------------------------------
# Pure-function: idempotent
# ---------------------------------------------------------------------------


def test_helper_is_idempotent() -> None:
    """Two calls with the same inputs produce identical outputs.
    No hidden state."""
    kw = _kw(
        now_mono=160.0,
        vol_spike_ratio=2.5,
        arm_dwell_started_mono=100.0,
    )
    r1 = evaluate_vol_regime_auto_pause(**kw)
    r2 = evaluate_vol_regime_auto_pause(**kw)
    assert r1 == r2
