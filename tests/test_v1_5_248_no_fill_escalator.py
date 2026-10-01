"""v1.5.248 — no-fill aggression escalator tests."""

from __future__ import annotations

import pytest


def test_aggression_level_below_trigger_is_zero():
    from app.no_fill_escalator import compute_aggression_level
    assert compute_aggression_level(
        seconds_since_last_fill=30.0,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
    ) == 0.0


def test_aggression_level_at_trigger_is_zero():
    from app.no_fill_escalator import compute_aggression_level
    # At exactly the trigger, elapsed_past_trigger = 0 → level = 0.
    assert compute_aggression_level(
        seconds_since_last_fill=60.0,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
    ) == 0.0


def test_aggression_level_mid_ramp_is_linear():
    from app.no_fill_escalator import compute_aggression_level
    # Halfway through the ramp: 60s trigger + 90s = 150s → level 0.5.
    level = compute_aggression_level(
        seconds_since_last_fill=150.0,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
    )
    assert abs(level - 0.5) < 1e-9


def test_aggression_level_at_end_of_ramp_is_one():
    from app.no_fill_escalator import compute_aggression_level
    assert compute_aggression_level(
        seconds_since_last_fill=240.0,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
    ) == 1.0


def test_aggression_level_beyond_ramp_saturates_at_one():
    from app.no_fill_escalator import compute_aggression_level
    assert compute_aggression_level(
        seconds_since_last_fill=10000.0,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
    ) == 1.0


def test_aggression_level_handles_none_ssf():
    from app.no_fill_escalator import compute_aggression_level
    assert compute_aggression_level(
        seconds_since_last_fill=None,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
    ) == 0.0


def test_aggression_level_handles_zero_ramp_as_step():
    """ramp=0 → step function at trigger."""
    from app.no_fill_escalator import compute_aggression_level
    assert compute_aggression_level(
        seconds_since_last_fill=59.0, trigger_seconds=60.0, ramp_seconds=0.0,
    ) == 0.0
    assert compute_aggression_level(
        seconds_since_last_fill=61.0, trigger_seconds=60.0, ramp_seconds=0.0,
    ) == 1.0


def test_compute_output_disabled_is_noop():
    from app.no_fill_escalator import compute_escalator_output
    out = compute_escalator_output(
        seconds_since_last_fill=10000.0,
        enabled=False,  # ← disabled
        trigger_seconds=60.0,
        ramp_seconds=180.0,
        spread_compress_max_bps=10.0,
        microprice_attenuate=True,
        toxicity_attenuate=True,
        reservation_shift_mult_at_full=0.5,
    )
    assert out.aggression_level == 0.0
    assert out.spread_compression_bps == 0.0
    assert out.microprice_widen_multiplier == 1.0
    assert out.toxicity_widen_multiplier == 1.0
    assert out.reservation_shift_multiplier == 1.0
    assert out.active_dimensions == ()


def test_compute_output_below_trigger_is_noop_even_when_enabled():
    from app.no_fill_escalator import compute_escalator_output
    out = compute_escalator_output(
        seconds_since_last_fill=30.0,  # below trigger
        enabled=True,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
        spread_compress_max_bps=10.0,
        microprice_attenuate=True,
        toxicity_attenuate=True,
        reservation_shift_mult_at_full=0.5,
    )
    assert out.aggression_level == 0.0
    assert out.spread_compression_bps == 0.0
    assert out.active_dimensions == ()


def test_compute_output_full_aggression():
    """At level=1.0 with all dimensions enabled, every knob is at max."""
    from app.no_fill_escalator import compute_escalator_output
    out = compute_escalator_output(
        seconds_since_last_fill=10000.0,
        enabled=True,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
        spread_compress_max_bps=10.0,
        microprice_attenuate=True,
        toxicity_attenuate=True,
        reservation_shift_mult_at_full=0.5,
    )
    assert out.aggression_level == 1.0
    assert out.spread_compression_bps == 10.0
    assert out.microprice_widen_multiplier == 0.0  # fully suppressed
    assert out.toxicity_widen_multiplier == 0.0    # fully suppressed
    # reservation: 1.0 + 1.0 * (0.5 - 1.0) = 0.5
    assert abs(out.reservation_shift_multiplier - 0.5) < 1e-9
    assert len(out.active_dimensions) == 4


def test_compute_output_at_half_aggression():
    """At level=0.5, each knob is half-applied."""
    from app.no_fill_escalator import compute_escalator_output
    out = compute_escalator_output(
        seconds_since_last_fill=150.0,  # half-way: 90s of 180s ramp past 60s trigger
        enabled=True,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
        spread_compress_max_bps=10.0,
        microprice_attenuate=True,
        toxicity_attenuate=True,
        reservation_shift_mult_at_full=0.5,
    )
    assert abs(out.aggression_level - 0.5) < 1e-9
    assert abs(out.spread_compression_bps - 5.0) < 1e-9
    # microprice: 1.0 - 0.5 = 0.5
    assert abs(out.microprice_widen_multiplier - 0.5) < 1e-9
    # reservation: 1.0 + 0.5 * (0.5 - 1.0) = 0.75
    assert abs(out.reservation_shift_multiplier - 0.75) < 1e-9


def test_compute_output_selective_attenuation_flags():
    """Each attenuate flag controls its dimension independently."""
    from app.no_fill_escalator import compute_escalator_output
    out = compute_escalator_output(
        seconds_since_last_fill=10000.0,
        enabled=True,
        trigger_seconds=60.0,
        ramp_seconds=180.0,
        spread_compress_max_bps=10.0,
        microprice_attenuate=False,  # ← off
        toxicity_attenuate=True,
        reservation_shift_mult_at_full=1.0,  # ← no attenuation
    )
    assert out.microprice_widen_multiplier == 1.0  # not attenuated
    assert out.toxicity_widen_multiplier == 0.0    # attenuated
    assert out.reservation_shift_multiplier == 1.0  # no change


def test_settings_defaults():
    from app.config import Settings
    s = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
    )
    assert s.no_fill_escalator_enabled is False  # default off
    assert s.no_fill_escalator_trigger_seconds == 60.0
    assert s.no_fill_escalator_ramp_seconds == 180.0
    assert s.no_fill_escalator_spread_compress_max_bps == 10.0
    assert s.no_fill_escalator_microprice_attenuate is True
    assert s.no_fill_escalator_toxicity_attenuate is True
    assert s.no_fill_escalator_reservation_shift_mult_at_full == 0.5


def test_settings_can_override_via_env():
    from app.config import Settings
    s = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
        NO_FILL_ESCALATOR_ENABLED=True,
        NO_FILL_ESCALATOR_TRIGGER_SECONDS=45.0,
    )
    assert s.no_fill_escalator_enabled is True
    assert s.no_fill_escalator_trigger_seconds == 45.0
