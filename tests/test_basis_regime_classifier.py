"""Unit tests for the online basis-regime classifier.

Production code: ``app/basis_regime.py::BasisRegimeClassifier``.

Design invariants pinned here:
- Before ``min_pair_samples`` pairs have accumulated, the classifier
  returns 0.0 (undecided), regardless of IC magnitude.
- IC sign < 0 (dev negatively correlated with future return) → −1.0.
- IC sign > 0 (dev positively correlated with future return) → +1.0.
- |IC| below ``ic_threshold`` → 0.0 (undecided).
- Malformed inputs (NaN / inf / non-positive mid) are silently dropped.
- The classifier only extracts one pair per observation (lagged obs
  at ``t - horizon`` paired with current mid).
"""

from __future__ import annotations

import math

import pytest

from app.basis_regime import BasisRegimeClassifier


def _mr_classifier() -> BasisRegimeClassifier:
    return BasisRegimeClassifier(
        horizon_seconds=1.0,
        window_samples=60,
        ic_threshold=0.15,
        min_pair_samples=10,
    )


# ---------- Cold-start behaviour ----------

def test_cold_start_returns_zero_sign() -> None:
    c = _mr_classifier()
    assert c.get_regime_sign() == 0.0
    assert c.last_ic is None
    assert c.pair_count == 0


def test_below_min_samples_returns_zero() -> None:
    """Feed a few observations — still below min_pair_samples → sign=0."""
    c = _mr_classifier()
    for i in range(5):
        c.observe(float(i), dev_bps=0.5 * i, mid=100.0 + 0.1 * i)
    # At horizon=1.0, each new obs creates one pair with obs from 1s ago.
    # After 5 obs at 1s cadence we have at most ~4 pairs, < 10 min.
    sign = c.get_regime_sign()
    assert sign == 0.0


# ---------- Mean-reversion regime (IC < 0) ----------

def test_mean_reversion_regime_detected() -> None:
    """Positive dev_bps paired with NEGATIVE realized returns → IC < 0.
    Expected regime sign: -1.0 (fade the deviation, legacy v1).
    """
    c = _mr_classifier()
    # Synthesize: at t, mid shifts opposite to dev. We set up pairs
    # where dev and return have opposite signs.
    # t0: mid=100, dev=+2
    # t1 (1s later): mid=99.98 (return = -2 bps) — mean reversion happened
    # So the pair recorded at t1 is (dev_at_t0=+2, return=-2).
    # Repeat with varying dev to build 30+ pairs, all with negative correlation.
    now_mono = 0.0
    for i in range(40):
        dev = 1.0 + (i % 5) * 0.5      # positive dev, varying magnitude
        # mid drops in proportion to dev (mean-reversion)
        mid = 100.0 - dev * 0.01        # 100 - dev * 0.01 → return ≈ -100*dev bps
        c.observe(now_mono, dev_bps=dev, mid=100.0)  # first obs anchors mid
        c.observe(now_mono + 1.5, dev_bps=0.0, mid=mid)  # return opposite to dev
        now_mono += 3.0
    sign = c.get_regime_sign()
    assert sign == -1.0
    assert c.last_ic is not None
    assert c.last_ic < -0.15


# ---------- Trend regime (IC > 0) ----------

def test_trend_regime_detected() -> None:
    """Positive dev paired with POSITIVE returns → IC > 0.
    Expected regime sign: +1.0.
    """
    c = _mr_classifier()
    now_mono = 0.0
    for i in range(40):
        dev = 1.0 + (i % 5) * 0.5
        mid_up = 100.0 + dev * 0.01     # mid rises in proportion to dev
        c.observe(now_mono, dev_bps=dev, mid=100.0)
        c.observe(now_mono + 1.5, dev_bps=0.0, mid=mid_up)
        now_mono += 3.0
    sign = c.get_regime_sign()
    assert sign == 1.0
    assert c.last_ic is not None
    assert c.last_ic > 0.15


# ---------- Undecided regime (|IC| < threshold) ----------

def test_undecided_when_ic_below_threshold() -> None:
    """Random dev/return pairs → IC near zero → sign=0."""
    c = _mr_classifier()
    now_mono = 0.0
    # Carefully construct so IC is small: alternate positive and
    # negative products so net correlation is ~0.
    for i in range(40):
        dev = 1.0 if i % 2 == 0 else -1.0
        # return does not track dev in any systematic way
        mid_change = 0.5 if i % 4 < 2 else -0.5
        mid_after = 100.0 + mid_change * 0.01
        c.observe(now_mono, dev_bps=dev, mid=100.0)
        c.observe(now_mono + 1.5, dev_bps=0.0, mid=mid_after)
        now_mono += 3.0
    sign = c.get_regime_sign()
    assert sign == 0.0
    if c.last_ic is not None:
        assert abs(c.last_ic) < 0.15


# ---------- Malformed input handling ----------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_dev_bps_is_dropped(bad) -> None:
    c = _mr_classifier()
    c.observe(0.0, dev_bps=bad, mid=100.0)
    assert len(c._obs_buffer) == 0  # nothing recorded


@pytest.mark.parametrize("bad_mid", [0.0, -10.0, float("nan"), float("inf")])
def test_bad_mid_is_dropped(bad_mid) -> None:
    c = _mr_classifier()
    c.observe(0.0, dev_bps=1.0, mid=bad_mid)
    assert len(c._obs_buffer) == 0


def test_sign_is_bounded_after_pathological_inputs() -> None:
    """Mixed good and bad observations shouldn't corrupt the classifier."""
    c = _mr_classifier()
    now = 0.0
    for i in range(30):
        c.observe(now, dev_bps=float("nan"), mid=100.0)  # dropped
        c.observe(now, dev_bps=1.0, mid=-1.0)             # dropped
        c.observe(now, dev_bps=1.0, mid=100.0)             # kept
        c.observe(now + 1.5, dev_bps=0.0, mid=99.98)       # pair produced (MR)
        now += 3.0
    sign = c.get_regime_sign()
    assert sign in (-1.0, 0.0, 1.0)


# ---------- Horizon arithmetic ----------

def test_no_pair_before_horizon_elapsed() -> None:
    """If we feed observations faster than the horizon, we only produce
    pairs once the horizon window has elapsed."""
    c = BasisRegimeClassifier(
        horizon_seconds=5.0,
        window_samples=60,
        ic_threshold=0.15,
        min_pair_samples=3,
    )
    # Feed 4 obs within the first 2 seconds — no pairs yet (5s horizon).
    c.observe(0.0, dev_bps=1.0, mid=100.0)
    c.observe(0.5, dev_bps=1.2, mid=100.01)
    c.observe(1.0, dev_bps=1.4, mid=100.02)
    c.observe(1.5, dev_bps=1.6, mid=100.03)
    # Check no pairs were produced yet.
    assert len(c._pairs_buffer) == 0
    # Cross the horizon — new obs at 6s should find the obs at 0.0 as lagged.
    c.observe(6.0, dev_bps=0.0, mid=100.5)
    assert len(c._pairs_buffer) == 1


# ---------- Snapshot diagnostic ----------

def test_snapshot_returns_expected_keys() -> None:
    c = _mr_classifier()
    snap = c.snapshot()
    assert "last_ic" in snap
    assert "last_regime_sign" in snap
    assert "pair_count" in snap
    assert "horizon_seconds" in snap
    assert "window_samples" in snap
    assert "ic_threshold" in snap
    assert "min_pair_samples" in snap


# ---------- Ranges ----------

def test_ic_is_bounded_minus1_to_1() -> None:
    """Even with pathologically large inputs, |IC| stays in [-1, 1]."""
    c = BasisRegimeClassifier(
        horizon_seconds=1.0,
        window_samples=50,
        ic_threshold=0.0,  # permissive to force a sign
        min_pair_samples=5,
    )
    now = 0.0
    for i in range(20):
        dev = 1e9 * (1.0 if i % 2 == 0 else -1.0)
        mid_after = 100.0 + dev * 1e-12
        c.observe(now, dev_bps=dev, mid=100.0)
        c.observe(now + 1.5, dev_bps=0.0, mid=mid_after)
        now += 3.0
    sign = c.get_regime_sign()
    assert sign in (-1.0, 0.0, 1.0)
    if c.last_ic is not None:
        assert -1.0 <= c.last_ic <= 1.0
