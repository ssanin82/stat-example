"""v1.5.239 — EWMA-of-|log-return| volatility measure (trend-aware).

Companion to ``app/volatility.py``'s ``VolatilityEstimator``. The two
measures answer different questions:

* ``VolatilityEstimator.vol_bps``   = stdev of consecutive log-returns
                                       over a 32-mid-change window.
                                       Trend-blind — a steady walk of
                                       small same-direction returns
                                       produces near-zero stdev.
* ``VolAbsEwmaEstimator.value_bps`` = exponentially weighted mean of
                                       |log-return| with a configurable
                                       half-life. Trend-aware — the
                                       same steady walk produces an
                                       EWMA equal to the per-tick step
                                       size, NOT zero.

This module ships in v1.5.239 alongside the existing ``vol_bps``,
NOT as a replacement. Every downstream consumer of ``vol_bps`` has
calibrated thresholds against the stdev scale; switching units
would invalidate ~8 thresholds at once. The migration path is
single-consumer A/B: first wire is the regime classifier's
``vol_slope`` criterion, gated behind
``REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE`` (default ``False``).
If the A/B is positive, additional consumers swap one at a time.

The v1.5.231-260529-111332 snapshot is the motivating dataset:
drift ±22 bp on a trending TON day produced near-zero
``stdev(log_returns)`` because every per-tick log-return was the
same ~0.5 bp value, so the bot's ``vol_climbing_widen`` gate
never armed and the markout was −1.85 bp.

Computation
-----------

On each new mid (post-dedup — same mid as the previous push is
silently skipped, same as ``VolatilityEstimator``):

1. Compute ``r = log(mid_new / mid_old)``.
2. Take the absolute value scaled to bp: ``sample_bps = abs(r) * 1e4``.
3. Update the EWMA using continuous-time decay over wall-clock
   ``dt_seconds`` since the previous push.

The continuous-time decay matters: quiet periods (where mid doesn't
move for many seconds) cause the EWMA to decay toward zero, NOT to
freeze at its last value. This is the desired behaviour — if the
market is genuinely flat for a half-life, the measure should
report ~50% of its prior value.

Warm-up: the first push only seeds the cache; the second push
produces the first real EWMA value. ``value_bps()`` returns
``None`` until then.

Default-off (v1.5.239)
----------------------

``REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE=false`` by default — the
estimator runs unconditionally so the value is always available
on the snapshot (operators can compare it to ``vol_bps`` side-by-
side for calibration purposes), but no consumer reads it. Flipping
the flag in env starts the A/B with the regime classifier as the
first consumer.

Threading
---------

Same model as ``VolatilityEstimator``: updated by the quote-loop
thread (one call per tick via ``record_mid``), read by the same
thread. Snapshot read is GIL-atomic.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


def _ewma_step(
    prev: Optional[float],
    sample: float,
    dt_seconds: float,
    halflife_seconds: float,
) -> float:
    """Continuous-time EWMA update.

    Identical formula to ``app/ofi.py::_ewma_step`` (copied rather
    than imported to keep the modules independent). The decay
    constant for elapsed time ``dt`` against half-life ``H`` is
    ``alpha = 1 - 0.5 ** (dt / H)``. Cold-start (``prev is None``)
    returns the sample. ``halflife <= 0`` disables smoothing.
    ``dt <= 0`` returns ``prev`` unchanged (no double-counting on
    same-instant calls).
    """
    if prev is None:
        return float(sample)
    if halflife_seconds <= 0:
        return float(sample)
    if dt_seconds <= 0:
        return float(prev)
    alpha = 1.0 - 0.5 ** (dt_seconds / halflife_seconds)
    return float(prev) + alpha * (float(sample) - float(prev))


@dataclass(slots=True)
class VolAbsEwmaEstimator:
    """EWMA of |log-return| in bp, with continuous-time decay.

    Mirrors the structural shape of ``OFIAccumulator``: keeps a last-
    sample cache and one EWMA float, exposes a per-input
    ``record_mid`` mutation and a thread-safe ``snapshot_dict``.
    """

    halflife_seconds: float = 20.0

    # Cache of the previous mid + its push timestamp. ``None`` until
    # the first ``record_mid`` call.
    _last_mid: Optional[float] = None
    _last_ts_mono: Optional[float] = None

    # The EWMA value, in bp. ``None`` until the second ``record_mid``
    # call (the first call only seeds the cache; no return can be
    # computed without a prior mid).
    ewma_bps: Optional[float] = None

    # Telemetry counters.
    update_count: int = 0
    last_update_mono_seconds: Optional[float] = None

    def record_mid(
        self,
        mid: Optional[float],
        now_mono_seconds: Optional[float] = None,
    ) -> None:
        """Record a new mid; update the EWMA in place.

        Same dedup rule as ``VolatilityEstimator``: if ``mid`` equals
        the cached previous mid (or is None / non-positive), the call
        is silently skipped. This keeps the EWMA from being driven
        toward zero by repeated identical-mid samples on flat tape.

        ``now_mono_seconds`` defaults to ``time.monotonic()``; tests
        pass explicit values to drive deterministic decay.

        Defensive against non-finite / non-positive mids: skips the
        update without mutating cache state.
        """
        if mid is None:
            return
        try:
            m = float(mid)
        except (TypeError, ValueError):
            return
        if not math.isfinite(m) or m <= 0:
            return

        ts = (
            float(now_mono_seconds)
            if now_mono_seconds is not None
            else time.monotonic()
        )

        # Dedup: same mid as previous push → nothing to learn.
        if self._last_mid is not None and m == self._last_mid:
            return

        if self._last_mid is None:
            # First push — seed cache, no EWMA update.
            self._last_mid = m
            self._last_ts_mono = ts
            self.last_update_mono_seconds = ts
            return

        # Compute |log-return| in bp.
        try:
            ret = math.log(m / float(self._last_mid))
        except (ValueError, ZeroDivisionError):
            return
        sample_bps = abs(ret) * 10_000.0

        dt = ts - float(self._last_ts_mono or ts)
        self.ewma_bps = _ewma_step(
            self.ewma_bps, sample_bps, dt, self.halflife_seconds
        )

        self._last_mid = m
        self._last_ts_mono = ts
        self.last_update_mono_seconds = ts
        self.update_count += 1

    def value_bps(self) -> Optional[float]:
        """Current EWMA value in bp, or ``None`` during warm-up.

        Warm-up = before the second ``record_mid`` call. Returns the
        same value on repeated calls until the next ``record_mid``.
        """
        return self.ewma_bps

    def decay_to(self, now_mono_seconds: float) -> None:
        """Decay the EWMA toward zero based on wall-clock elapsed time.

        Optional utility for callers that want the EWMA to reflect
        "current quietness" even when no new mid has arrived for a
        while. NOT called from the hot tick loop — record_mid's per-
        update dt already does the right thing when mids are flowing.
        Use this when reading the value during a long quiet stretch
        (e.g. publishing to S3 every 5s with no recent mid updates).

        No-op when the estimator hasn't warmed up.
        """
        if self.ewma_bps is None or self._last_ts_mono is None:
            return
        dt = float(now_mono_seconds) - float(self._last_ts_mono)
        if dt <= 0:
            return
        # Sample = 0 (no new return), so EWMA decays toward 0.
        self.ewma_bps = _ewma_step(
            self.ewma_bps, 0.0, dt, self.halflife_seconds
        )
        self._last_ts_mono = float(now_mono_seconds)

    def snapshot_dict(self) -> dict:
        """Telemetry snapshot. Safe to call from any thread under GIL."""
        return {
            "ewma_bps": (
                float(self.ewma_bps) if self.ewma_bps is not None else None
            ),
            "halflife_seconds": float(self.halflife_seconds),
            "update_count": int(self.update_count),
            "last_update_mono_seconds": (
                float(self.last_update_mono_seconds)
                if self.last_update_mono_seconds is not None
                else None
            ),
        }
