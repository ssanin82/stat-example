"""Phase 4C.3 (v1.5.41) — Side-specific historical edge as quote-decision input.

Out-of-order delivery of the Phase 4C "economics-first quoting" arc.
Phase 4C.1 + 4C.2 (v1.4.164) plumbed ``expected_net_edge_bps`` into a
per-side refusal gate. Phase 4C.3 mini (v1.4.161,
``realised_edge_side_suppress``) added a short-window per-side
trailing-mean detector with binary cooldown.

This module ships the FULL Phase 4C.3 piece per the defense plan:

* **4C.3.a** — maintain a rolling 1 h trailing per-side mean (and
  stdev) of ``markout_5s_bps + rebate_bps``.
* **4C.3.b** — feed the per-side trailing edge into the expected-edge
  computation as a MULTIPLICATIVE confidence factor:

      adjusted_edge = expected_edge × max(0.0, 1.0 + α × z_score)

  where ``α = SIDE_EDGE_HISTORY_ZSCORE_COEFF`` (default 0.5) and
  ``z_score = mean / stdev`` (distance from break-even in
  stdev-units of recent realised edge).
* **4C.3.c** — when the bot's recent realised edge on side X has
  degraded relative to its own variability, the refusal gate
  (4C.1+4C.2) triggers sooner because ``adjusted_edge`` is smaller
  than ``expected_edge``. When recent realised edge is strong, the
  multiplier exceeds 1 → expected_edge is amplified → gate is less
  likely to fire.

Distinct from the existing ``realised_edge_side_suppress`` gate:

* That gate is a BINARY cooldown — small window (20 fills), fires
  when mean drops below a hard threshold (-4 bps), suppresses the
  side for 60 s. Targets acute degradation.
* This one is a CONTINUOUS confidence factor — longer window (1 h),
  scales the expected-edge consumer's input proportionally. Targets
  slow drift in per-side economics.

Pure data structure. State lives on ``BotState`` (per-side rolling
deques). Bot calls ``history.note_fill(side, edge_bps, now_mono)``
from the 5 s markout resolution path, then queries
``history.confidence_multiplier(side, now_mono)`` once per tick when
evaluating the expected-edge gate.

Defense-plan reference: ``plans/20260520-defense-action-plan.md``
§4C.3.a / 4C.3.b / 4C.3.c.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

from app.enums import Side


@dataclass(frozen=True)
class SideEdgeStats:
    """Per-side trailing-window statistics. ``n`` is the count of
    fills in the window; ``mean`` and ``stdev`` are in bps. ``stdev``
    is the SAMPLE stdev (Bessel-corrected, divides by ``n-1``); it's
    ``0.0`` when ``n < 2``."""

    n: int
    mean_bps: float
    stdev_bps: float


def _sample_stdev(values: Sequence[float], mean: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    # Bessel correction: divide by (n-1). With small samples and
    # high variance this avoids systematically under-stating
    # variability.
    var = sum((v - mean) ** 2 for v in values) / float(n - 1)
    return math.sqrt(var)


class SideEdgeHistory:
    """Per-side rolling-window history of realised net edge bps.

    The window is **time-bounded** (default 1 h) — older samples are
    pruned by age at every append + query. There's also a hard
    ``max_samples`` cap to bound memory in pathological fill-storm
    conditions (deque maxlen). At 1 fill / second peak, 1 h = 3600
    samples; default cap is generous (8192) so the time bound is the
    binding constraint in normal operation.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 3600.0,
        max_samples: int = 8192,
        min_samples: int = 10,
        zscore_coeff: float = 0.5,
        mult_floor: float = 0.0,
        mult_ceil: float = 2.0,
        enabled: bool = False,
    ) -> None:
        """Construct the history.

        ``min_samples`` — below this count the z-score is treated as
        zero (multiplier = 1.0; no effect). Prevents pathologically
        low-sample stdev from producing a giant multiplier.

        ``zscore_coeff`` (α in the formula) — sets sensitivity of the
        multiplier to recent edge. 0 = disabled (multiplier always
        1.0). Higher = more reactive.

        ``mult_floor`` / ``mult_ceil`` — clamp the multiplier into a
        safe band. ``floor=0.0`` lets very-bad recent edge fully
        zero out expected_edge → forces refusal. ``ceil=2.0`` caps
        the amplification so a small stdev with strong positive edge
        doesn't overshoot.

        ``enabled=False`` — feature disabled (multiplier always 1.0
        regardless of stats). Default-off so the feature is safe to
        ship without operator opt-in.
        """
        self._window_seconds = float(window_seconds)
        self._min_samples = int(min_samples)
        self._zscore_coeff = float(zscore_coeff)
        self._mult_floor = float(mult_floor)
        self._mult_ceil = float(mult_ceil)
        self._enabled = bool(enabled)
        # Each entry: (ts_mono, edge_bps).
        self._buy: deque[tuple[float, float]] = deque(maxlen=int(max_samples))
        self._sell: deque[tuple[float, float]] = deque(maxlen=int(max_samples))

    # -----------------------------------------------------------------
    # config visibility (read-only)
    # -----------------------------------------------------------------

    def enabled(self) -> bool:
        return self._enabled

    def window_seconds(self) -> float:
        return self._window_seconds

    def min_samples(self) -> int:
        return self._min_samples

    def zscore_coeff(self) -> float:
        return self._zscore_coeff

    # -----------------------------------------------------------------
    # ingest
    # -----------------------------------------------------------------

    def note_fill(
        self,
        *,
        side: Side,
        markout_5s_bps: Optional[float],
        rebate_bps: float,
        now_mono: float,
    ) -> None:
        """Record a resolved-5 s fill. ``markout_5s_bps`` may be None
        for fills that never resolved (very rare — process exit
        before 5 s elapses); silently dropped.

        Convention matches the existing 4C.3-mini gate:
            edge_bps = markout_5s_bps + rebate_bps
        where ``rebate_bps`` is positive for maker rebates.
        """
        if markout_5s_bps is None:
            return
        edge_bps = float(markout_5s_bps) + float(rebate_bps)
        if not math.isfinite(edge_bps):
            return
        buf = self._buy if side == Side.BUY else self._sell
        buf.append((float(now_mono), edge_bps))
        self._prune(buf, float(now_mono))

    # -----------------------------------------------------------------
    # query
    # -----------------------------------------------------------------

    def stats(self, side: Side, now_mono: float) -> SideEdgeStats:
        """Trailing per-side stats. Mean & stdev are in bps; ``n`` is
        the count of in-window samples (after pruning)."""
        buf = self._buy if side == Side.BUY else self._sell
        self._prune(buf, float(now_mono))
        if not buf:
            return SideEdgeStats(n=0, mean_bps=0.0, stdev_bps=0.0)
        values = [e for _, e in buf]
        n = len(values)
        mean = sum(values) / float(n)
        stdev = _sample_stdev(values, mean)
        return SideEdgeStats(n=n, mean_bps=mean, stdev_bps=stdev)

    def zscore(self, side: Side, now_mono: float) -> Optional[float]:
        """Z-score of the trailing mean relative to break-even (0 bps).

        Returns:
          * z = mean / stdev when n >= min_samples AND stdev > 0
          * None otherwise (insufficient data / degenerate variance)

        Convention: z > 0 means recent realised edge is above
        break-even by z stdevs; z < 0 means below.
        """
        s = self.stats(side, now_mono)
        if s.n < self._min_samples:
            return None
        if s.stdev_bps <= 0.0:
            return None
        return s.mean_bps / s.stdev_bps

    def confidence_multiplier(
        self, side: Side, now_mono: float
    ) -> float:
        """Per-side multiplier to apply to ``expected_net_edge_bps``.

            multiplier = clamp(
                1.0 + zscore_coeff × zscore,
                mult_floor,
                mult_ceil,
            )

        Returns ``1.0`` (no-op) when:
          * feature disabled
          * z-score unavailable (insufficient samples or zero variance)

        Bounded by ``[mult_floor, mult_ceil]``. Default bounds are
        ``[0.0, 2.0]`` — bad edge can fully zero expected_edge (forces
        refusal); good edge can at most double it.
        """
        if not self._enabled:
            return 1.0
        z = self.zscore(side, now_mono)
        if z is None:
            return 1.0
        raw = 1.0 + self._zscore_coeff * z
        return max(self._mult_floor, min(self._mult_ceil, raw))

    # -----------------------------------------------------------------
    # internal
    # -----------------------------------------------------------------

    def _prune(
        self, buf: deque[tuple[float, float]], now_mono: float
    ) -> None:
        cutoff = now_mono - self._window_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    # -----------------------------------------------------------------
    # snapshot for telemetry / postmortem
    # -----------------------------------------------------------------

    def to_snapshot(self, now_mono: float) -> dict:
        """Return a JSON-serialisable snapshot of the per-side stats.
        Consumed by ``state.snapshot_dict`` / ``live_stats``."""
        out: dict = {
            "enabled": self._enabled,
            "window_seconds": self._window_seconds,
            "min_samples": self._min_samples,
            "zscore_coeff": self._zscore_coeff,
            "mult_floor": self._mult_floor,
            "mult_ceil": self._mult_ceil,
        }
        for side in (Side.BUY, Side.SELL):
            key = side.value.lower()
            s = self.stats(side, now_mono)
            z = self.zscore(side, now_mono)
            m = self.confidence_multiplier(side, now_mono)
            out[key] = {
                "n": s.n,
                "mean_bps": s.mean_bps,
                "stdev_bps": s.stdev_bps,
                "zscore": z,
                "multiplier": m,
            }
        return out


__all__ = ["SideEdgeHistory", "SideEdgeStats"]
