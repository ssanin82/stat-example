"""Phase 2B (v1.5.26) -- canonical multi-horizon mid-drift dataclass.

The existing per-gate helpers (``inventory_drift_gate
.compute_short_window_drifts``, quote_eligibility's
``mid_return_500ms_bps`` / ``mid_return_long_window_bps``,
forward-classifier history) each walk the mid-samples deque
independently to compute drift over their respective windows.
Pre-2B those walks ran in separate code paths even though they
share the anchored-walk logic.

2B.1 ships ``MidDriftWindows`` (frozen dataclass with all 7
horizons) + ``compute_mid_drift_windows()`` (single function that
walks once). 2B.3 publishes the dataclass in ``snapshot_dict()``'s
``mid_drift_windows`` sub-block. **2B.2 (v1.5.181) ships the
per-consumer migration**: the anchor convention is now standardised
to the forward-walk, first-inside-window rule
(``ts >= cutoff``) matching what ``inventory_drift_gate`` has used
since Phase 1A. ``inventory_drift_gate._drift_bps_over_window``
now re-exports the helper from this module so there's a single
implementation. Bot-side consumers (shock_gate / regime_controller /
spread_composition paths) read drift values from
``state.mid_drift_windows`` instead of recomputing per-call.

Tests cover the dataclass, the compute helper at each window, and
the snapshot integration.

v1.5.181 -- updated expectations for the convention switch:
``test_short_deque_only_short_windows_populated`` and
``test_full_deque_each_window_uses_correct_anchor`` now reflect
forward-walk anchors. ``test_zero_anchor_returns_none`` reflects
the fact that forward-walk SKIPS invalid (mid<=0) samples instead
of breaking out at the first one.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.mid_drift_windows import (
    MidDriftWindows,
    compute_mid_drift_windows,
)
from app.state import BotState


# ---------------------------------------------------------------------------
# compute_mid_drift_windows
# ---------------------------------------------------------------------------


def test_empty_deque_returns_all_none():
    """No samples -> every window returns None (gate stays dormant)."""
    out = compute_mid_drift_windows(
        [], now_mono=1000.0, mid_now=100.0
    )
    assert isinstance(out, MidDriftWindows)
    assert out.drift_500ms_bps is None
    assert out.drift_5s_bps is None
    assert out.drift_10s_bps is None
    assert out.drift_30s_bps is None
    assert out.drift_60s_bps is None
    assert out.drift_5min_bps is None
    assert out.drift_15min_bps is None


def test_short_deque_only_short_windows_populated():
    """When the deque only spans 2 seconds, the 500ms window's
    forward-walk picks the current sample (t=1000) as anchor because
    no in-deque sample falls inside [999.5, 1000.0). Longer windows
    return None because no sample falls inside their cutoff range.

    v1.5.181 — under the forward-walk convention (``ts >= cutoff``)
    the 500ms anchor for integer-spaced samples on an integer ``now``
    lands on the sample at ``now`` itself, so drift collapses to 0.
    In production (2 Hz tick, ~500 ms sample spacing, fractional
    ``now``) the anchor is the second-most-recent sample so the
    500ms drift is a meaningful return — this test exercises the
    integer-tick degenerate case.
    """
    # Samples at t=998, 999, 1000 with mid going from 100 -> 101 -> 102.
    samples = [(998.0, 100.0), (999.0, 101.0), (1000.0, 102.0)]
    out = compute_mid_drift_windows(
        samples, now_mono=1000.0, mid_now=102.0
    )
    # 500ms window: cutoff=999.5; first sample with ts>=999.5 is
    # t=1000 (mid=102 == mid_now). drift = 0.
    assert out.drift_500ms_bps == pytest.approx(0.0)
    # 5s ago = 995 -- no sample inside [995, 1000]; the oldest
    # sample (998) IS inside this window, so anchor = (998, 100).
    # drift = (102 / 100 - 1) * 1e4 = 200 bps.
    assert out.drift_5s_bps == pytest.approx(200.0)
    # 30s window: cutoff=970, all samples qualify, oldest is t=998.
    # drift = same 200 bps.
    assert out.drift_30s_bps == pytest.approx(200.0)
    # 15min window: cutoff = 100, all samples qualify, oldest = t=998.
    assert out.drift_15min_bps == pytest.approx(200.0)


def test_full_deque_each_window_uses_correct_anchor():
    """Build a deque spanning 1000 s with mids = 100 + (t - 0). Each
    window's anchor under the forward-walk convention is the OLDEST
    sample at-or-after ``cutoff``.

    v1.5.181 — for integer ``now`` and integer-spaced samples, the
    cutoff at integer seconds (5s, 10s, 30s, 60s, 300s, 900s) lands
    EXACTLY on a sample, so forward-walk picks the same anchor as
    the previous backward-walk convention. The only divergence is
    at the fractional 500 ms cutoff: forward-walk picks t=1000
    (the current sample, drift=0) instead of t=999.
    """
    samples = [
        (float(t), 100.0 + float(t))
        for t in range(0, 1001, 1)  # t = 0..1000, mid = 100..1100
    ]
    out = compute_mid_drift_windows(
        samples, now_mono=1000.0, mid_now=1100.0
    )
    # 500ms cutoff = 999.5; forward-walk first ts>=999.5 is t=1000.
    # drift = (1100/1100 - 1)*1e4 = 0.
    assert out.drift_500ms_bps == pytest.approx(0.0)
    # 5s cutoff = 995, anchor (995, 1095). Same as pre-2B.2.
    assert out.drift_5s_bps == pytest.approx(
        (1100.0 / 1095.0 - 1.0) * 1e4
    )
    # 10s cutoff = 990, anchor (990, 1090).
    assert out.drift_10s_bps == pytest.approx(
        (1100.0 / 1090.0 - 1.0) * 1e4
    )
    # 30s cutoff = 970, anchor (970, 1070).
    assert out.drift_30s_bps == pytest.approx(
        (1100.0 / 1070.0 - 1.0) * 1e4
    )
    # 60s cutoff = 940, anchor (940, 1040).
    assert out.drift_60s_bps == pytest.approx(
        (1100.0 / 1040.0 - 1.0) * 1e4
    )
    # 5 min cutoff = 700, anchor (700, 800).
    assert out.drift_5min_bps == pytest.approx(
        (1100.0 / 800.0 - 1.0) * 1e4
    )
    # 15 min cutoff = 100, anchor (100, 200).
    assert out.drift_15min_bps == pytest.approx(
        (1100.0 / 200.0 - 1.0) * 1e4
    )


def test_negative_drift_when_mid_falls():
    """Mid drops over the window -> negative drift_bps."""
    samples = [(990.0, 100.0), (995.0, 98.0), (1000.0, 95.0)]
    out = compute_mid_drift_windows(
        samples, now_mono=1000.0, mid_now=95.0
    )
    # 10s ago = 990, anchor (990, 100); drift = (95/100 - 1) * 1e4 = -500 bps.
    assert out.drift_10s_bps == pytest.approx(-500.0)


def test_to_dict_shape():
    """``MidDriftWindows.to_dict()`` returns the dict shape the
    snapshot publishes."""
    samples = [(990.0, 100.0), (995.0, 100.5), (1000.0, 101.0)]
    out = compute_mid_drift_windows(
        samples, now_mono=1000.0, mid_now=101.0
    )
    d = out.to_dict()
    assert set(d.keys()) == {
        "drift_500ms_bps",
        "drift_5s_bps",
        "drift_10s_bps",
        "drift_30s_bps",
        "drift_60s_bps",
        "drift_5min_bps",
        "drift_15min_bps",
    }


def test_zero_mid_now_returns_all_none():
    """Defensive: invalid mid_now -> all None (don't divide by 0)."""
    samples = [(990.0, 100.0), (1000.0, 101.0)]
    out = compute_mid_drift_windows(
        samples, now_mono=1000.0, mid_now=0.0
    )
    assert out.drift_10s_bps is None


def test_zero_anchor_skipped_falls_through_to_next_sample():
    """Defensive: invalid anchor (mid<=0) gets SKIPPED; the walk
    continues to the next sample. v1.5.181: pre-2B.2 the backward-
    walk implementation broke out of the loop at the first ts<=cutoff
    even when its mid was zero, returning None. Post-2B.2 the
    forward-walk skips invalid samples and keeps walking — for this
    deque, the next sample is t=1000 (mid=101=mid_now), giving
    drift = 0.

    Catching division-by-zero is still guaranteed: the anchor
    variable is only set to a positive finite mid, and the final
    ``anchor <= 0`` guard remains in place.
    """
    samples = [(990.0, 0.0), (1000.0, 101.0)]
    out = compute_mid_drift_windows(
        samples, now_mono=1000.0, mid_now=101.0
    )
    # cutoff=990; t=990 has mid=0 (skipped); t=1000 has mid=101==mid_now.
    # drift = 0 (current-sample anchor).
    assert out.drift_10s_bps == pytest.approx(0.0)


def test_zero_mid_now_still_returns_none():
    """Defensive: invalid ``mid_now`` aborts the walk early."""
    samples = [(990.0, 100.0), (1000.0, 101.0)]
    out = compute_mid_drift_windows(
        samples, now_mono=1000.0, mid_now=0.0
    )
    assert out.drift_10s_bps is None


# ---------------------------------------------------------------------------
# BotState integration (2B.3)
# ---------------------------------------------------------------------------


def test_botstate_default_mid_drift_windows_is_none():
    """Fresh BotState has ``mid_drift_windows = None`` until the
    bot's per-tick block populates it."""
    s = Settings()
    st = BotState(s)
    assert st.mid_drift_windows is None


def test_snapshot_publishes_mid_drift_windows_as_none_when_unset():
    """``snapshot_dict()`` exposes the field as None during warmup."""
    s = Settings()
    st = BotState(s)
    snap = st.snapshot_dict()
    assert "mid_drift_windows" in snap
    assert snap["mid_drift_windows"] is None


def test_snapshot_publishes_mid_drift_windows_as_dict_when_set():
    """When ``state.mid_drift_windows`` is populated, the snapshot
    surfaces all 7 fields under the sub-block."""
    s = Settings()
    st = BotState(s)
    samples = [(float(t), 100.0 + t * 0.1) for t in range(0, 1000, 1)]
    st.mid_drift_windows = compute_mid_drift_windows(
        samples, now_mono=999.0, mid_now=200.0
    )
    snap = st.snapshot_dict()
    mdw = snap.get("mid_drift_windows") or {}
    assert isinstance(mdw, dict)
    assert "drift_500ms_bps" in mdw
    assert "drift_30s_bps" in mdw
    assert "drift_15min_bps" in mdw


# ---------------------------------------------------------------------------
# v1.5.35 regression — long windows must populate at production tick rates
# ---------------------------------------------------------------------------


def test_long_windows_populate_at_high_tick_rate():
    """Snapshot v1.5.28-260522-223401 (prod) showed drift_30s, drift_60s,
    drift_5min, drift_15min all returning ``None`` after 2 hours of
    session uptime. Root cause: the bot's quote loop is wake-event-
    driven (every BBO update wakes it), so the actual tick rate
    bursts to 15-20 Hz during shock-gate churn. The default deque
    maxlen of 512 only covered 25-30 seconds of history at that rate,
    making every drift window ≥30s return None.

    v1.5.35 bumped the defaults to retention=1000s and maxlen=20000
    so even at 20 Hz the deque covers the full 15-minute horizon.
    This regression test pumps samples at 20 Hz for 16 minutes and
    verifies all 7 windows return non-None.
    """
    s = Settings()
    st = BotState(s)

    # Confirm the defaults are big enough.
    assert st._mid_price_samples_long.maxlen >= 18000, (
        "deque maxlen must accommodate 15min × 20Hz = 18000 samples"
    )
    assert s.drift_long_window_seconds >= 900.0, (
        "retention must cover the 15-min window"
    )

    # Simulate 20 Hz tick rate for 16 minutes (= 19200 ticks, slightly
    # over the 15-min horizon for headroom).
    mono = 1000.0
    for i in range(19200):
        mono = 1000.0 + i * 0.05
        # Linearly rising mid so each window's drift is unambiguously
        # non-zero.
        mid = 1.95 + (i * 1e-5)
        st.record_mid_price_sample(mid, mono)

    samples = st.mid_price_samples_long_snapshot()
    span_s = samples[-1][0] - samples[0][0]
    assert span_s >= 900.0, (
        f"deque must span >= 15 min after the simulated load; got {span_s:.1f}s"
    )

    out = compute_mid_drift_windows(samples, now_mono=mono, mid_now=mid)
    # Every window must populate -- this is the bug the v1.5.35 fix
    # addresses.
    assert out.drift_500ms_bps is not None, "500ms window must populate"
    assert out.drift_5s_bps is not None
    assert out.drift_10s_bps is not None
    assert out.drift_30s_bps is not None, "30s window was None pre-fix"
    assert out.drift_60s_bps is not None, "60s window was None pre-fix"
    assert out.drift_5min_bps is not None, "5min window was None pre-fix"
    assert out.drift_15min_bps is not None, "15min window was None pre-fix"

    # Sanity: drifts should monotonically increase with horizon (since
    # we simulated a linear rise) -- ensures the anchor walk really
    # reaches back to the older samples and isn't truncated.
    assert out.drift_500ms_bps < out.drift_5s_bps
    assert out.drift_5s_bps < out.drift_10s_bps
    assert out.drift_10s_bps < out.drift_30s_bps
    assert out.drift_30s_bps < out.drift_60s_bps
    assert out.drift_60s_bps < out.drift_5min_bps
    assert out.drift_5min_bps < out.drift_15min_bps
