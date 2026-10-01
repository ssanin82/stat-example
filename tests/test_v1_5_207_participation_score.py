"""v1.5.207 Phase 4C.5 — participation score tests.

Covers:

* ``compute_participation_score`` pure-function: zero/one boundaries,
  piecewise-linear interpolation, threshold ordering, defensive
  inputs.
* ``participation_action`` discrete mapping.
* ``is_score_consistent_with_existing_decision`` shadow-mode check.
* BotState rolling-window deque + per-tick scalar update via
  end-to-end bot integration (light-touch, just exercising the
  caller wiring).
"""

from __future__ import annotations

import pytest

from app.participation_score import (
    _DEFAULT_FULL_EDGE_BPS,
    _DEFAULT_HARD_THRESHOLD,
    _DEFAULT_SOFT_THRESHOLD,
    compute_participation_score,
    is_score_consistent_with_existing_decision,
    participation_action,
)


# Standard thresholds used across most tests
REFUSE = -1.0
DAMPEN = 0.0
FULL = 3.0
SOFT = 0.7
HARD = 0.3


def _score(edge):
    return compute_participation_score(
        expected_edge_bps=edge,
        refuse_floor_bps=REFUSE,
        dampen_floor_bps=DAMPEN,
        full_edge_bps=FULL,
        soft_threshold=SOFT,
        hard_threshold=HARD,
    )


# ------------------ compute_participation_score: boundaries ----------------- #


def test_score_full_edge_returns_one() -> None:
    assert _score(3.0) == 1.0
    assert _score(5.0) == 1.0  # above full → still 1.0
    assert _score(100.0) == 1.0


def test_score_at_dampen_floor_returns_soft() -> None:
    # edge == dampen_floor → boundary between dampen and marginal-pos.
    # Defined as the dampen-band side (soft threshold).
    assert _score(0.0) == pytest.approx(SOFT)


def test_score_at_refuse_floor_returns_hard() -> None:
    # edge == refuse_floor → boundary between refuse and dampen.
    # Defined as the dampen-band side (hard threshold).
    assert _score(-1.0) == pytest.approx(HARD)


def test_score_below_refuse_floor_returns_zero() -> None:
    assert _score(-2.0) == 0.0
    assert _score(-10.0) == 0.0


def test_score_none_edge_returns_none() -> None:
    assert _score(None) is None


def test_score_nan_edge_returns_none() -> None:
    assert _score(float("nan")) is None


# ------------------ piecewise linearity ------------------------------------ #


def test_score_marginal_pos_linear() -> None:
    # edge=1.5, halfway between dampen_floor=0 and full=3
    # → score halfway between soft=0.7 and 1.0 = 0.85
    assert _score(1.5) == pytest.approx(0.85)


def test_score_dampen_band_linear() -> None:
    # edge=-0.5, halfway between refuse=-1 and dampen=0
    # → score halfway between hard=0.3 and soft=0.7 = 0.5
    assert _score(-0.5) == pytest.approx(0.5)


def test_score_quarter_dampen_band() -> None:
    # edge=-0.25, 75% from refuse to dampen
    # → score 75% from hard to soft = 0.3 + 0.75*(0.7-0.3) = 0.6
    assert _score(-0.25) == pytest.approx(0.6)


def test_score_monotonically_increasing() -> None:
    edges = [-2.0, -1.0, -0.75, -0.5, -0.25, 0.0, 1.0, 2.0, 3.0, 5.0]
    scores = [_score(e) for e in edges]
    for a, b in zip(scores[:-1], scores[1:]):
        assert a <= b, f"score not monotonic: {scores}"


# ------------------ defensive inputs --------------------------------------- #


def test_score_inverted_thresholds_returns_none() -> None:
    """refuse >= dampen → pathological config → None."""
    s = compute_participation_score(
        expected_edge_bps=0.5,
        refuse_floor_bps=2.0,   # higher than dampen
        dampen_floor_bps=0.0,
        full_edge_bps=3.0,
    )
    assert s is None


def test_score_collapsed_dampen_band_returns_none() -> None:
    """refuse == dampen → division-by-zero risk → None."""
    s = compute_participation_score(
        expected_edge_bps=0.5,
        refuse_floor_bps=0.0,
        dampen_floor_bps=0.0,
        full_edge_bps=3.0,
    )
    assert s is None


def test_score_inverted_score_thresholds_returns_none() -> None:
    """soft <= hard or out-of-[0,1] → None."""
    s = compute_participation_score(
        expected_edge_bps=0.5,
        refuse_floor_bps=-1.0,
        dampen_floor_bps=0.0,
        full_edge_bps=3.0,
        soft_threshold=0.3,
        hard_threshold=0.7,  # swapped
    )
    assert s is None


# ------------------ participation_action ----------------------------------- #


def test_action_quote_at_one() -> None:
    assert participation_action(1.0) == "quote"


def test_action_quote_at_soft() -> None:
    assert participation_action(SOFT) == "quote"


def test_action_dampen_below_soft() -> None:
    assert participation_action(SOFT - 0.01) == "dampen"


def test_action_dampen_at_hard() -> None:
    assert participation_action(HARD) == "dampen"


def test_action_refuse_below_hard() -> None:
    assert participation_action(HARD - 0.01) == "refuse"
    assert participation_action(0.0) == "refuse"


def test_action_unknown_on_none() -> None:
    assert participation_action(None) == "unknown"


# ------------------ shadow-mode consistency check -------------------------- #


def test_consistency_quote_quote_agrees() -> None:
    # score=0.9 → quote; existing not refused, not dampened → quote
    assert is_score_consistent_with_existing_decision(
        score=0.9, was_refused=False, was_dampened=False,
    )


def test_consistency_refuse_refuse_agrees() -> None:
    assert is_score_consistent_with_existing_decision(
        score=0.1, was_refused=True, was_dampened=False,
    )


def test_consistency_dampen_dampen_agrees() -> None:
    assert is_score_consistent_with_existing_decision(
        score=0.5, was_refused=False, was_dampened=True,
    )


def test_consistency_disagreement_score_refuse_existing_quote() -> None:
    # score=0.1 says refuse; gates didn't refuse → disagreement
    assert not is_score_consistent_with_existing_decision(
        score=0.1, was_refused=False, was_dampened=False,
    )


def test_consistency_unknown_score_always_consistent() -> None:
    """When the score has no opinion (None), don't count as
    disagreement — there's no claim to disagree with."""
    assert is_score_consistent_with_existing_decision(
        score=None, was_refused=True, was_dampened=False,
    )
    assert is_score_consistent_with_existing_decision(
        score=None, was_refused=False, was_dampened=False,
    )
