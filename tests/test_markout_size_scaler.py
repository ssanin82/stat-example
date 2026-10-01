"""Tests for ``app.markout_size_scaler`` — direct rolling-median
5s markout → size multiplier. Bypasses the toxicity composite
score so the dashboard's "markout" tier label and the bot's
size-shrink behavior align."""

from __future__ import annotations

import pytest

from app.markout_size_scaler import markout_size_mult


def test_clean_returns_full_size() -> None:
    assert markout_size_mult(0.0) == 1.0
    assert markout_size_mult(+5.0) == 1.0


def test_mild_adverse_returns_mild_mult() -> None:
    assert markout_size_mult(-0.5) == pytest.approx(0.85)


def test_moderate_adverse_returns_moderate_mult() -> None:
    assert markout_size_mult(-2.0) == pytest.approx(0.5)


def test_heavy_adverse_returns_heavy_mult() -> None:
    assert markout_size_mult(-5.0) == pytest.approx(0.25)


def test_threshold_boundaries_exact() -> None:
    """Threshold checks use ``>=`` so the boundary lands in the
    higher (less-adverse) tier."""
    # exactly 0 → clean
    assert markout_size_mult(0.0) == 1.0
    # exactly -1 → mild (mild_threshold is 0; moderate_threshold is -1)
    assert markout_size_mult(-1.0) == pytest.approx(0.85)
    # exactly -3 → moderate
    assert markout_size_mult(-3.0) == pytest.approx(0.5)


def test_none_returns_full() -> None:
    assert markout_size_mult(None) == 1.0


def test_nan_returns_full() -> None:
    assert markout_size_mult(float("nan")) == 1.0


def test_floor_clamp() -> None:
    """The heavy_mult result is clamped above floor."""
    # heavy_mult=0.10 below floor=0.20 → returns 0.20.
    out = markout_size_mult(-10.0, heavy_mult=0.10, floor=0.20)
    assert out == pytest.approx(0.20)


def test_custom_thresholds() -> None:
    """Operator can tune to a more conservative or more lenient
    profile via thresholds."""
    out = markout_size_mult(
        -2.0,
        mild_threshold_bps=-0.5,
        moderate_threshold_bps=-1.5,
        heavy_threshold_bps=-2.5,
    )
    # -2.0 falls between moderate (-1.5) and heavy (-2.5) → moderate tier.
    assert out == pytest.approx(0.5)


def test_invalid_input_returns_full() -> None:
    assert markout_size_mult("not-a-number") == 1.0
    assert markout_size_mult([]) == 1.0
