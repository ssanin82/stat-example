"""Online regime classifier for cross-venue basis-deviation alpha (Priority #2 v2).

Background
----------
The v1 basis-deviation alpha (``app/quoting.py::compute_quote_decision``)
assumed *unconditional mean-reversion* — when the current
``grvt_mid − bybit_mid`` basis diverges from its long-run EWMA, the
reservation is shifted OPPOSITE the deviation direction. That works
when the dislocation is temporary order-flow noise, but it is actively
harmful during trending regimes where one venue is leading a persistent
move and the other is simply lagging.

Snap_20260420_081215 (96 min SOL, trending recovery 82→84) showed the
regression: markout mean blew out from −1.69 to −2.45 bps, with
SELL-side markout going from −2.00 to −2.90 bps. Both Deep Research
reports flagged this failure mode up front; we shipped v1 without
the regime override because it was cheaper, and paid the cost.

Approach
--------
Compute an online information coefficient (IC) between lagged
deviation and realized future mid-return:

    IC = Pearson_correlation(
        dev_bps_at_t,
        realized_mid_return_from_t_to_t+H
    )

over the last N observed pairs. The sign of IC selects the regime:

- IC < −threshold → mean-reversion dominant. Use ``sign = -1`` (fade
  the deviation, same as v1).
- IC > +threshold → trend-continuation dominant. Use ``sign = +1``
  (trail the deviation instead of fading it).
- |IC| < threshold → undecided. Use ``sign = 0`` (gate the alpha off
  entirely until the signal clarifies).

This is the simpler of the two regime approaches the research
suggested. Gemini's Hurst-exponent alternative is more theoretically
principled but expensive to compute online; we keep that as a v3
upgrade if the IC classifier proves unstable.

Usage
-----
Single-threaded assumption: the classifier lives on ``BotState`` and
is updated + read from the main quote loop. No internal locking.
Callers pass wall-monotonic timestamps (``_clock.monotonic()``) so the
horizon arithmetic is unaffected by wall-clock jumps.

Integration
-----------
``compute_quote_decision`` accepts a ``basis_deviation_regime_sign``
kwarg (default −1.0 preserves legacy v1 mean-reversion behaviour). The
classifier's ``get_regime_sign()`` returns −1 / 0 / +1 based on the
current IC; the main loop passes this into the decision call. When the
sign is 0, the basis-deviation shift is skipped.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from app import clock as _clock


@dataclass(slots=True)
class _Observation:
    ts_mono: float
    dev_bps: float
    mid: float


class BasisRegimeClassifier:
    """Online IC-based regime classifier for the cross-venue basis signal.

    Thread-safety: single-writer, single-reader. The bot main loop owns
    both ``observe`` and ``get_regime_sign`` calls; no external
    synchronisation is assumed.
    """

    def __init__(
        self,
        *,
        horizon_seconds: float = 2.0,
        window_samples: int = 240,
        ic_threshold: float = 0.15,
        min_pair_samples: int = 50,
    ) -> None:
        if horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be positive")
        if window_samples < 10:
            raise ValueError("window_samples must be >= 10")
        if ic_threshold < 0:
            raise ValueError("ic_threshold must be non-negative")
        if min_pair_samples < 2:
            raise ValueError("min_pair_samples must be >= 2")
        self._horizon_seconds = float(horizon_seconds)
        self._window_samples = int(window_samples)
        self._ic_threshold = float(ic_threshold)
        self._min_pair_samples = int(min_pair_samples)
        # Observation buffer — retains enough history to look back
        # ``horizon_seconds``. Size generous: assumes up to ~4 Hz
        # observation rate (public WS wakes + 0.5 s quote loop).
        obs_capacity = max(32, int(horizon_seconds * 8) + 16)
        self._obs_buffer: deque[_Observation] = deque(maxlen=obs_capacity)
        # (lagged_dev_bps, realized_return_bps) pairs for IC.
        self._pairs_buffer: deque[tuple[float, float]] = deque(
            maxlen=self._window_samples
        )
        # Diagnostics for telemetry / logging.
        self._last_ic: Optional[float] = None
        self._last_regime_sign: float = 0.0
        self._last_pair_count: int = 0

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def observe(self, now_mono: float, dev_bps: float, mid: float) -> None:
        """Record a new observation and, if a lagged pair is available,
        extract the ``(dev_at_t_minus_H, realized_return_from_t_minus_H_to_t)``
        pair and push it into the IC buffer.

        Silently drops the update when any input is non-finite or mid
        is non-positive — bad data must not poison the classifier.
        """
        if not (
            isinstance(now_mono, (int, float))
            and isinstance(dev_bps, (int, float))
            and isinstance(mid, (int, float))
            and math.isfinite(float(now_mono))
            and math.isfinite(float(dev_bps))
            and math.isfinite(float(mid))
            and float(mid) > 0
        ):
            return
        target = float(now_mono) - self._horizon_seconds
        lagged: Optional[_Observation] = None
        # Linear scan — obs_buffer is O(32) so this is cheap.
        # Find the newest obs with ts_mono <= target.
        for obs in self._obs_buffer:
            if obs.ts_mono <= target:
                if lagged is None or obs.ts_mono > lagged.ts_mono:
                    lagged = obs
            else:
                # Deque is insertion-ordered (which is time-ordered here)
                # so once ts crosses target we can stop.
                break
        if lagged is not None and lagged.mid > 0:
            realized_return_bps = (
                (float(mid) - lagged.mid) / lagged.mid * 10_000.0
            )
            if math.isfinite(realized_return_bps):
                self._pairs_buffer.append(
                    (lagged.dev_bps, realized_return_bps)
                )
        self._obs_buffer.append(
            _Observation(float(now_mono), float(dev_bps), float(mid))
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_regime_sign(self) -> float:
        """Return -1.0 / 0.0 / +1.0 based on current IC.

        -1.0 → mean-reversion dominant (legacy v1 sign)
        +1.0 → trend-continuation dominant (flip the v1 sign)
         0.0 → undecided / not enough samples (skip the alpha entirely)
        """
        n = len(self._pairs_buffer)
        self._last_pair_count = n
        if n < self._min_pair_samples:
            self._last_ic = None
            self._last_regime_sign = 0.0
            return 0.0
        mean_dev = 0.0
        mean_ret = 0.0
        for d, r in self._pairs_buffer:
            mean_dev += d
            mean_ret += r
        mean_dev /= n
        mean_ret /= n
        num = 0.0
        den_d_sq = 0.0
        den_r_sq = 0.0
        for d, r in self._pairs_buffer:
            dd = d - mean_dev
            rr = r - mean_ret
            num += dd * rr
            den_d_sq += dd * dd
            den_r_sq += rr * rr
        if den_d_sq <= 0.0 or den_r_sq <= 0.0:
            # Degenerate: all dev or all return constant. No signal.
            self._last_ic = None
            self._last_regime_sign = 0.0
            return 0.0
        ic = num / math.sqrt(den_d_sq * den_r_sq)
        # Guard against numerical drift pushing |IC| slightly above 1.
        if ic > 1.0:
            ic = 1.0
        elif ic < -1.0:
            ic = -1.0
        self._last_ic = ic
        if ic > self._ic_threshold:
            self._last_regime_sign = 1.0
        elif ic < -self._ic_threshold:
            self._last_regime_sign = -1.0
        else:
            self._last_regime_sign = 0.0
        return self._last_regime_sign

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def last_ic(self) -> Optional[float]:
        return self._last_ic

    @property
    def last_regime_sign(self) -> float:
        return self._last_regime_sign

    @property
    def pair_count(self) -> int:
        return self._last_pair_count

    @property
    def buffer_capacity(self) -> int:
        return self._window_samples

    def snapshot(self) -> dict[str, object]:
        """Non-blocking snapshot of the current state for telemetry."""
        return {
            "last_ic": self._last_ic,
            "last_regime_sign": self._last_regime_sign,
            "pair_count": self._last_pair_count,
            "horizon_seconds": self._horizon_seconds,
            "window_samples": self._window_samples,
            "ic_threshold": self._ic_threshold,
            "min_pair_samples": self._min_pair_samples,
        }

    # ------------------------------------------------------------------
    # Persistence (todo-010 / plans/telemetry.md Step 1, N12 wiring)
    # ------------------------------------------------------------------

    def seed_from_persisted(
        self,
        last_regime_sign: Optional[float],
        last_ic: Optional[float],
        pair_count: Optional[int],
    ) -> None:
        """Restore diagnostic fields from a previous shutdown snapshot.

        Only the *result* fields (`last_regime_sign`, `last_ic`,
        `_last_pair_count`) are seeded. The internal ``_obs_buffer`` and
        ``_pairs_buffer`` cannot be persisted (too large, stale fast)
        and remain empty until ``observe()`` calls re-populate them
        over the next ~5 min. That trade-off is intentional: until the
        buffers warm, the classifier's outputs reflect the persisted
        result rather than zeros — which closes the "5-10 min regime-
        blind after restart" gap described in
        ``plans/telemetry.md`` §2.

        Silently skips any field with a non-finite / None / wrong-typed
        value so a malformed persistent-runtime-state file can't
        corrupt the classifier on startup.
        """
        if (
            last_regime_sign is not None
            and isinstance(last_regime_sign, (int, float))
            and math.isfinite(float(last_regime_sign))
        ):
            v = float(last_regime_sign)
            # Coerce to canonical {-1, 0, +1}; defensive against drift.
            if v > 0.5:
                self._last_regime_sign = 1.0
            elif v < -0.5:
                self._last_regime_sign = -1.0
            else:
                self._last_regime_sign = 0.0
        if (
            last_ic is not None
            and isinstance(last_ic, (int, float))
            and math.isfinite(float(last_ic))
        ):
            self._last_ic = float(last_ic)
        if (
            pair_count is not None
            and isinstance(pair_count, int)
            and pair_count >= 0
        ):
            self._last_pair_count = int(pair_count)
