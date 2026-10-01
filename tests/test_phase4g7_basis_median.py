"""Phase 4G.7 (v1.5.18) -- basis-median rolling buffer + classifier wiring.

Until 4G.7, the forward-regime classifier's ``basis_stretch_cautious_ratio``
trigger was dormant because the bot didn't maintain a 30-minute median
of the binance basis. ``app/bot.py`` passed
``binance_basis_30min_median_bps=None`` and the basis-stretch indicator
returned ``None`` from ``_compute_basis_stretch_ratio``. The other three
CAUTIOUS triggers (vol slope, drift magnitude rising, OB imbalance
widening) carried the load.

This batch:

1. Adds ``forward_basis_bps_history`` to ``BotState`` with a longer
   retention window (default 30 min) than the other forward buffers
   (180 s) -- because basis-stretch compares to a MEDIAN, not a
   short-window derivative.
2. Extends ``append_forward_signal_history`` to accept ``basis_bps`` +
   ``basis_max_age_seconds`` and prune by the basis-specific cutoff.
3. Adds ``median_forward_basis_bps()`` to ``BotState`` -- computes the
   exact median of the buffer on each call.
4. Adds the ``REGIME_FORWARD_BASIS_MEDIAN_LOOKBACK_SECONDS`` setting
   (default 1800 s = 30 min).
5. Wires the bot's per-tick classifier call to read the median back as
   ``_fwd_basis_median_bps`` (replacing the hard-coded ``None``).

These tests cover the new code paths -- the classifier itself is
unchanged and exhaustively tested in
``test_regime_forward_signals.py``. Per CLAUDE.md, ONLY this test file
is run from the assistant; full-suite verification is the CI daemon's
job.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.regime_forward_signals import (
    ForwardRegime,
    ForwardSignalThresholds,
    classify_forward_regime,
)
from app.state import BotState


def _bs() -> BotState:
    """Fresh BotState with default Settings -- enough for the
    rolling-buffer helpers, which don't read settings directly."""
    return BotState(Settings())


# ---------------------------------------------------------------------------
# Median buffer mechanics
# ---------------------------------------------------------------------------


def test_median_empty_buffer_returns_none():
    """No history => no median. The classifier treats ``None`` as
    "no signal" -- basis_stretch stays dormant. This is the pre-warmup
    state right after bot start."""
    s = _bs()
    assert s.median_forward_basis_bps() is None


def test_median_single_sample_returns_that_sample():
    """Median of [x] = x. Sanity check for the edge case of one
    sample in the buffer."""
    s = _bs()
    s.append_forward_signal_history(
        now_mono=100.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,
        basis_bps=3.5,
        basis_max_age_seconds=1800.0,
    )
    assert s.median_forward_basis_bps() == pytest.approx(3.5)


def test_median_odd_count_returns_middle():
    """Median of [1, 2, 3, 4, 5] = 3. The sort handles the input
    being inserted in any order -- the buffer is time-ordered, not
    value-ordered."""
    s = _bs()
    # Append out-of-time-order is impossible in practice (now_mono is
    # monotonic), but VALUE order is arbitrary. Insert in scrambled
    # value order to verify the helper sorts internally.
    for t, basis in [(1.0, 5.0), (2.0, 1.0), (3.0, 3.0), (4.0, 2.0), (5.0, 4.0)]:
        s.append_forward_signal_history(
            now_mono=t,
            vol_bps=None,
            drift_30s_bps=None,
            ob_imbalance=None,
            max_age_seconds=180.0,
            basis_bps=basis,
            basis_max_age_seconds=1800.0,
        )
    assert s.median_forward_basis_bps() == pytest.approx(3.0)


def test_median_even_count_averages_two_middle():
    """Median of [1, 2, 3, 4] = 2.5. Even-count interpolation
    convention is the mean of the two middle elements."""
    s = _bs()
    for t, basis in [(1.0, 2.0), (2.0, 4.0), (3.0, 1.0), (4.0, 3.0)]:
        s.append_forward_signal_history(
            now_mono=t,
            vol_bps=None,
            drift_30s_bps=None,
            ob_imbalance=None,
            max_age_seconds=180.0,
            basis_bps=basis,
            basis_max_age_seconds=1800.0,
        )
    assert s.median_forward_basis_bps() == pytest.approx(2.5)


def test_basis_buffer_prunes_to_lookback_window():
    """Samples older than ``basis_max_age_seconds`` (default 1800 s)
    are pruned. The cutoff is now_mono - lookback -- a sample at
    t=100 stays alive until now_mono >= 100 + lookback.

    Pruning happens on append when the OLDEST entry is past the
    cutoff. This is the same pattern as the other three forward
    buffers (vol/drift/ob)."""
    s = _bs()
    # Three samples spaced 60 s apart, with a tight 90 s window so
    # the oldest WILL be pruned on the third append.
    s.append_forward_signal_history(
        now_mono=0.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,
        basis_bps=1.0,
        basis_max_age_seconds=90.0,
    )
    s.append_forward_signal_history(
        now_mono=60.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,
        basis_bps=2.0,
        basis_max_age_seconds=90.0,
    )
    s.append_forward_signal_history(
        now_mono=120.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,
        basis_bps=3.0,
        basis_max_age_seconds=90.0,
    )
    # cutoff = 120 - 90 = 30 -- the t=0 sample is pruned, t=60 + t=120
    # remain. Median of [2, 3] = 2.5.
    assert s.median_forward_basis_bps() == pytest.approx(2.5)
    assert len(s.forward_basis_bps_history) == 2


def test_basis_buffer_independent_from_other_forward_buffers():
    """The basis buffer has its own retention window (longer); a
    short ``max_age_seconds`` on the other buffers must NOT prune
    basis. Tested by appending a basis sample, then ticking with
    short max_age but no new basis -- basis should still be there."""
    s = _bs()
    # Initial basis at t=0.
    s.append_forward_signal_history(
        now_mono=0.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,  # short forward-buffer window
        basis_bps=5.0,
        basis_max_age_seconds=1800.0,  # long basis window (30 min)
    )
    # Tick at t=500 with short max_age=180 but no basis update.
    # The basis buffer's cutoff is 500 - 1800 = -1300 -- the t=0
    # sample is safely inside.
    s.append_forward_signal_history(
        now_mono=500.0,
        vol_bps=3.0,
        drift_30s_bps=2.0,
        ob_imbalance=0.1,
        max_age_seconds=180.0,
        basis_bps=None,
        basis_max_age_seconds=1800.0,
    )
    assert s.median_forward_basis_bps() == pytest.approx(5.0)
    assert len(s.forward_basis_bps_history) == 1


def test_basis_skipped_when_basis_bps_is_none():
    """``None`` basis means "no sample to record this tick" (early
    bot start, transient WS reconnect). Buffer unchanged."""
    s = _bs()
    s.append_forward_signal_history(
        now_mono=10.0,
        vol_bps=3.0,
        drift_30s_bps=2.0,
        ob_imbalance=0.1,
        max_age_seconds=180.0,
        basis_bps=None,
        basis_max_age_seconds=1800.0,
    )
    assert s.forward_basis_bps_history == []
    assert s.median_forward_basis_bps() is None


def test_basis_max_age_defaults_to_30min_when_not_passed():
    """If ``basis_max_age_seconds`` is omitted, the helper defaults
    to 1800 s. Verifies the default-window contract -- callers that
    haven't updated to the new parameter still get correct behaviour."""
    s = _bs()
    # No basis_max_age_seconds kwarg => default 1800 s.
    s.append_forward_signal_history(
        now_mono=0.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,
        basis_bps=1.0,
    )
    # Sample at t=1799 => still inside 1800 s window.
    s.append_forward_signal_history(
        now_mono=1799.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,
        basis_bps=3.0,
    )
    assert len(s.forward_basis_bps_history) == 2
    # Sample at t=1801 => cutoff = 1801 - 1800 = 1, t=0 pruned.
    s.append_forward_signal_history(
        now_mono=1801.0,
        vol_bps=None,
        drift_30s_bps=None,
        ob_imbalance=None,
        max_age_seconds=180.0,
        basis_bps=5.0,
    )
    assert len(s.forward_basis_bps_history) == 2
    # Buffer is now [(1799, 3.0), (1801, 5.0)] => median = 4.0.
    assert s.median_forward_basis_bps() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Integration -- median feeds the classifier
# ---------------------------------------------------------------------------


def test_classifier_basis_stretch_dormant_when_history_empty():
    """No basis history => classifier's basis_stretch indicator
    is None => trigger doesn't fire. Confirms the pre-4G.7 state
    (which was the bot's permanent state before this change)
    remains the steady state during warmup."""
    s = _bs()
    median = s.median_forward_basis_bps()
    assert median is None
    # Run classifier with the None median -- basis_stretch indicator
    # is silent regardless of current_binance_basis_bps.
    reading = classify_forward_regime(
        vol_bps_history=[],
        drift_30s_history=[],
        ob_imbalance_history=[],
        current_ob_imbalance=0.1,
        current_vol_bps=2.0,
        current_drift_30s_bps=2.0,
        current_binance_basis_bps=10.0,  # would be a HUGE stretch
        binance_basis_30min_median_bps=median,
        settings=ForwardSignalThresholds(),
        now_mono=100.0,
    )
    assert reading.basis_stretch_ratio_observed is None


def test_classifier_basis_stretch_fires_when_current_exceeds_2x_median():
    """The headline test: with a populated median buffer and
    current basis 2.5x the median, the classifier should return
    CAUTIOUS with a basis_stretch reason. This is the failure-mode
    Phase 4G.7 was added to catch."""
    s = _bs()
    # Build a 30-min history at median ~2 bps. Spread the samples so
    # the median is well-defined.
    for i in range(60):
        s.append_forward_signal_history(
            now_mono=float(i * 30),  # 30-s spacing
            vol_bps=None,
            drift_30s_bps=None,
            ob_imbalance=None,
            max_age_seconds=180.0,
            basis_bps=2.0,
            basis_max_age_seconds=1800.0,
        )
    median = s.median_forward_basis_bps()
    assert median == pytest.approx(2.0)

    # Classifier with current_basis = 6.0 (3x median, above the 2.0x
    # threshold) => CAUTIOUS fires on basis_stretch.
    reading = classify_forward_regime(
        vol_bps_history=[(0.0, 2.0), (60.0, 2.0)],
        drift_30s_history=[(0.0, 1.0), (60.0, 1.0)],
        ob_imbalance_history=[(0.0, 0.0), (60.0, 0.0)],
        current_ob_imbalance=0.0,
        current_vol_bps=2.0,
        current_drift_30s_bps=1.0,
        current_binance_basis_bps=6.0,
        binance_basis_30min_median_bps=median,
        settings=ForwardSignalThresholds(),
        now_mono=1800.0,
    )
    assert reading.classification == ForwardRegime.CAUTIOUS
    assert "basis_stretch" in reading.reason
    assert reading.basis_stretch_ratio_observed == pytest.approx(3.0)


def test_classifier_basis_stretch_dormant_when_median_below_floor():
    """When the median basis magnitude is below
    ``basis_stretch_floor_bps`` (default 1.0), the indicator
    returns None to avoid division-by-near-zero noise triggering
    CAUTIOUS spuriously. Tested at exactly the floor + a hair below."""
    s = _bs()
    # Build a buffer with median ~0.5 bps (below default floor 1.0).
    for i in range(20):
        s.append_forward_signal_history(
            now_mono=float(i * 30),
            vol_bps=None,
            drift_30s_bps=None,
            ob_imbalance=None,
            max_age_seconds=180.0,
            basis_bps=0.5,
            basis_max_age_seconds=1800.0,
        )
    median = s.median_forward_basis_bps()
    assert median == pytest.approx(0.5)

    # Current basis way above median in absolute terms, but median
    # is below the floor => classifier returns ratio=None.
    reading = classify_forward_regime(
        vol_bps_history=[(0.0, 2.0), (60.0, 2.0)],
        drift_30s_history=[(0.0, 1.0), (60.0, 1.0)],
        ob_imbalance_history=[(0.0, 0.0), (60.0, 0.0)],
        current_ob_imbalance=0.0,
        current_vol_bps=2.0,
        current_drift_30s_bps=1.0,
        current_binance_basis_bps=5.0,
        binance_basis_30min_median_bps=median,
        settings=ForwardSignalThresholds(),
        now_mono=1800.0,
    )
    assert reading.basis_stretch_ratio_observed is None
