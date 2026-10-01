"""Phase 2B (v1.5.26 + v1.5.181) -- canonical multi-horizon mid-drift dataclass.

Pre-fix the bot computed mid drift at 7 different windows (500 ms /
5 s / 10 s / 30 s / 60 s / 5 min / 15 min) across multiple files
with similar-but-not-identical helpers:

* ``app/inventory_drift_gate.compute_short_window_drifts`` (10s + 30s)
* ``app/quote_eligibility``'s ``mid_return_500ms_bps`` and
  ``mid_return_long_window_bps``
* The forward-classifier's ``forward_drift_30s_history`` rolling buffer

Each call walks the mid-sample deque independently. The walks share
the same anchored-walk logic but live in separate code paths.

2B.1 + 2B.3 (v1.5.26) shipped the canonical entry point: a single
``MidDriftWindows`` dataclass + a single ``compute_mid_drift_windows``
function that walks the deque ONCE and computes all 7 windows. The
dataclass is also published in the bot's snapshot so the dashboard
+ postmortem can read all-windows-at-once without consumer-side
plumbing.

**2B.2 (v1.5.181)**: per-consumer migration finished. The anchor
convention is now standardised to the FORWARD-walk, first-inside-
window rule (``ts >= cutoff``) -- the same rule
``inventory_drift_gate`` has been using since Phase 1A. Why this
convention rather than the original v1.5.26 backward-walk:

* "Drift over the last N seconds" should require a sample within
  the last N seconds. Backward-walk picks the youngest sample
  OLDER than the cutoff, which can be arbitrarily old when the
  deque starts with samples way before the window opens; the
  reported "drift over N s" is then drift over much-longer-than-N s.
* The forward-walk convention returns ``None`` instead -- caller
  treats that as "warmup not complete, gate dormant".
* All four bot-side consumers (shock_gate / inventory_drift_gate
  / regime_controller path / spread_composition path) have been
  using forward-walk via ``compute_short_window_drifts``; aligning
  the dashboard cache + the gate path means a single source of
  truth and a dashboard that matches what gates actually fire on.

``inventory_drift_gate._drift_bps_over_window`` and
``compute_short_window_drifts`` now delegate to this module.

This module is pure + side-effect-free. The caller (bot.py per-tick)
constructs a fresh ``MidDriftWindows`` each tick from the shared mid
samples deque and writes it onto ``state.mid_drift_windows`` for
observers (dashboard) AND for the gate path (post-2B.2 consumers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class MidDriftWindows:
    """Mid-price drift over multiple time windows, all in bps.

    Each field is the signed drift from the anchor (the mid sample
    closest to ``now - window_seconds`` ago) to ``mid_now``:

        drift_bps = (mid_now / anchor - 1.0) * 1e4

    ``None`` when the deque doesn't have a sample inside that window
    yet (gate stays dormant until warmup).
    """

    drift_500ms_bps: Optional[float]
    drift_5s_bps: Optional[float]
    drift_10s_bps: Optional[float]
    drift_30s_bps: Optional[float]
    drift_60s_bps: Optional[float]
    drift_5min_bps: Optional[float]
    drift_15min_bps: Optional[float]

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "drift_500ms_bps": self.drift_500ms_bps,
            "drift_5s_bps": self.drift_5s_bps,
            "drift_10s_bps": self.drift_10s_bps,
            "drift_30s_bps": self.drift_30s_bps,
            "drift_60s_bps": self.drift_60s_bps,
            "drift_5min_bps": self.drift_5min_bps,
            "drift_15min_bps": self.drift_15min_bps,
        }


def _drift_bps_over_window(
    samples: Sequence[tuple[float, float]],
    *,
    now_mono: float,
    mid_now: float,
    window_seconds: float,
) -> Optional[float]:
    """Compute drift (bps) of mid over the last ``window_seconds``.

    Anchor strategy (v1.5.181, 2B.2 unification): walk FORWARD from
    the oldest sample, pick the first sample with
    ``ts >= now_mono - window_seconds`` -- i.e. the oldest sample
    that's still INSIDE the window. Return ``None`` when no sample
    falls inside the window (deque ends before the window opens
    OR the deque is empty / too young).

    Why forward-walk rather than backward-walk-just-past-cutoff:
    "drift over the last N seconds" requires a sample within the
    last N seconds. The backward-walk variant (used in this module
    pre-v1.5.181) would happily return drift from an arbitrarily-
    old anchor for a short window, mis-labelling "drift over 10 s"
    as drift over much longer when the deque starts well before
    the window opens. Forward-walk returns ``None`` instead and
    the caller treats that as "warmup not complete".

    Pure + side-effect-free.
    """
    if not samples or window_seconds <= 0 or mid_now <= 0:
        return None
    import math as _math
    if not _math.isfinite(mid_now):
        return None
    cutoff = float(now_mono) - float(window_seconds)
    # Walk from oldest to newest. The first sample with
    # ``ts >= cutoff`` is inside the window and is our anchor.
    anchor: Optional[float] = None
    for ts, mid in samples:
        if mid <= 0 or not _math.isfinite(mid):
            continue
        if ts >= cutoff:
            anchor = float(mid)
            break
    if anchor is None or anchor <= 0:
        return None
    return (float(mid_now) / anchor - 1.0) * 1e4


def compute_mid_drift_windows(
    samples: Sequence[tuple[float, float]],
    *,
    now_mono: float,
    mid_now: float,
) -> MidDriftWindows:
    """Walk the mid-samples deque once and compute all 7 windows.

    Each window uses the same anchored-median strategy: walk
    backwards from newest to find the first sample older than
    ``window_seconds`` ago, use that as anchor.

    Pre-2B the bot called several per-window helpers each doing
    their own walk. With 7 windows and a typical deque of ~360
    samples (180 s × 2 Hz), the bot was doing 7 × 360 = 2520
    comparisons per tick across the helpers. With 2B's single
    function the walk is reused -- but for now (2B.1) we keep the
    simple shape and rely on Python's O(N) walks being cheap at
    2 Hz tick rate. Optimisation can land later if profiler shows it.
    """
    return MidDriftWindows(
        drift_500ms_bps=_drift_bps_over_window(
            samples, now_mono=now_mono, mid_now=mid_now, window_seconds=0.5
        ),
        drift_5s_bps=_drift_bps_over_window(
            samples, now_mono=now_mono, mid_now=mid_now, window_seconds=5.0
        ),
        drift_10s_bps=_drift_bps_over_window(
            samples, now_mono=now_mono, mid_now=mid_now, window_seconds=10.0
        ),
        drift_30s_bps=_drift_bps_over_window(
            samples, now_mono=now_mono, mid_now=mid_now, window_seconds=30.0
        ),
        drift_60s_bps=_drift_bps_over_window(
            samples, now_mono=now_mono, mid_now=mid_now, window_seconds=60.0
        ),
        drift_5min_bps=_drift_bps_over_window(
            samples, now_mono=now_mono, mid_now=mid_now, window_seconds=300.0
        ),
        drift_15min_bps=_drift_bps_over_window(
            samples, now_mono=now_mono, mid_now=mid_now, window_seconds=900.0
        ),
    )


__all__ = [
    "MidDriftWindows",
    "compute_mid_drift_windows",
]
