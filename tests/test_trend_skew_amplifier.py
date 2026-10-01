"""Tests for ``app.trend_skew_amplifier`` — multiply the inventory
skew coefficient when position direction is aligned with short-term
drift.

Companion to ``test_momentum_gate``. Same trigger geometry, different
intervention: the gate blocks the adding side; the amplifier boosts
the reducing-side aggressiveness.

Targets snapshot 260515-095056 (2.3h, 86.6% long during -2.45% drop,
30s MAE -7 to -8 bp). The momentum gate alone stops adds; this layer
accelerates exits.
"""

from __future__ import annotations

from app.trend_skew_amplifier import compute_trend_skew_multiplier


def _kw(**overrides):
    base = {
        "drift_threshold_bps": 2.0,
        "inventory_pct_threshold": 0.40,
        "amplification_factor": 1.5,
        "enabled": True,
    }
    base.update(overrides)
    return base


def test_disabled_returns_unit_multiplier() -> None:
    """``enabled=False`` → multiplier=1.0 regardless of inputs."""
    mult, reason = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=10.0,
        **_kw(enabled=False),
    )
    assert mult == 1.0
    assert reason == ""


def test_factor_le_one_returns_unit_multiplier() -> None:
    """``amplification_factor <= 1.0`` is treated as disabled — the
    amplifier is strictly a boost, never a reduction."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(amplification_factor=1.0),
    )
    assert mult == 1.0
    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(amplification_factor=0.5),
    )
    assert mult == 1.0


def test_long_uptrend_aligned_returns_factor() -> None:
    """Long + positive drift = aligned. Multiplier = factor; reason
    string identifies long_uptrend."""
    mult, reason = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert mult == 1.5
    assert "long_uptrend" in reason
    assert "util=0.50" in reason
    assert "drift=+5.00bps" in reason
    assert "x1.50" in reason


def test_short_downtrend_aligned_returns_factor() -> None:
    """Symmetric: short + negative drift = aligned."""
    mult, reason = compute_trend_skew_multiplier(
        position_qty=-50.0,
        effective_abs_cap=100.0,
        drift_bps=-5.0,
        **_kw(),
    )
    assert mult == 1.5
    assert "short_downtrend" in reason


def test_long_downtrend_anti_aligned_no_amp() -> None:
    """Long + negative drift = ANTI-aligned. The amplifier is for
    aligned-with-drift cases (where the inventory exists because the
    market already rewarded the position direction); anti-aligned is
    already a fast-recovery scenario handled by normal skew. No
    amplification."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=-5.0,
        **_kw(),
    )
    assert mult == 1.0


def test_short_uptrend_anti_aligned_no_amp() -> None:
    """Symmetric anti-aligned case."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=-50.0,
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert mult == 1.0


def test_below_inventory_threshold_no_amp() -> None:
    """Aligned but small position → no amplification. The amplifier
    targets the wrong-sided-stuck-inventory case; small positions
    don't qualify."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=30.0,  # util=0.30 < 0.40 threshold
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert mult == 1.0


def test_below_drift_threshold_no_amp() -> None:
    """Aligned but trend too weak → no amplification. Routine drift
    doesn't qualify; needs a meaningful directional move."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=1.0,  # below 2.0 threshold
        **_kw(),
    )
    assert mult == 1.0


def test_missing_drift_no_amp() -> None:
    """``drift_bps=None`` (warmup, missing tick) → no amplification."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=None,
        **_kw(),
    )
    assert mult == 1.0


def test_non_finite_drift_no_amp() -> None:
    """NaN / inf drift → no amplification."""
    import math

    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=math.nan,
        **_kw(),
    )
    assert mult == 1.0
    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=math.inf,
        **_kw(),
    )
    assert mult == 1.0


def test_zero_cap_no_amp() -> None:
    """``effective_abs_cap=0`` (degenerate config) → no
    amplification (no divide-by-zero, no fire)."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=0.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert mult == 1.0


def test_flat_position_no_amp() -> None:
    """Zero position → no amplification (nothing to reduce)."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=0.0,
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert mult == 1.0


def test_at_thresholds_exact_fires() -> None:
    """Boundary: util == threshold and |drift| == threshold should
    fire (>=, not >)."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=40.0,  # util=0.40 == threshold
        effective_abs_cap=100.0,
        drift_bps=2.0,  # |drift|=2.0 == threshold
        **_kw(),
    )
    assert mult == 1.5


def test_just_below_drift_threshold_does_not_fire() -> None:
    """1.99 bps drift with 2.0 threshold → no amp."""
    mult, _ = compute_trend_skew_multiplier(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=1.99,
        **_kw(),
    )
    assert mult == 1.0


def test_amplifier_alignment_matches_momentum_gate() -> None:
    """Sanity: the amplifier fires under the same conditions the
    momentum gate fires. Co-arming is intentional — operator should
    not see one fire without the other."""
    from app.momentum_gate import evaluate_momentum_gate

    cases = [
        # (position_qty, drift_bps, should_fire)
        (50.0, 5.0, True),  # long_uptrend
        (-50.0, -5.0, True),  # short_downtrend
        (50.0, -5.0, False),  # anti-aligned long+down
        (-50.0, 5.0, False),  # anti-aligned short+up
        (30.0, 5.0, False),  # below util threshold
        (50.0, 1.0, False),  # below drift threshold
        (0.0, 5.0, False),  # flat
    ]
    for pos, drift, expected_fire in cases:
        amp_mult, _ = compute_trend_skew_multiplier(
            position_qty=pos,
            effective_abs_cap=100.0,
            drift_bps=drift,
            **_kw(),
        )
        gate_override, _ = evaluate_momentum_gate(
            position_qty=pos,
            effective_abs_cap=100.0,
            drift_bps=drift,
            drift_threshold_bps=2.0,
            inventory_pct_threshold=0.40,
            enabled=True,
        )
        amp_fired = amp_mult > 1.0
        gate_fired = gate_override is not None
        assert amp_fired == expected_fire, (
            f"amplifier mismatch for pos={pos} drift={drift}: "
            f"amp_fired={amp_fired} expected={expected_fire}"
        )
        assert gate_fired == expected_fire, (
            f"momentum_gate mismatch for pos={pos} drift={drift}: "
            f"gate_fired={gate_fired} expected={expected_fire}"
        )
