"""v1.5.209 — Phase 8D Order Flow Imbalance (OFI) directional alpha.

Cont, Kukanov & Stoikov (2014, "The Price Impact of Order Book Events")
showed that a single number per BBO update — the *order-flow imbalance*
— captures most of the short-horizon predictive information that the
classic level-1 OB-imbalance signal misses. OFI is a flow (event-rate)
measure where OB-imbalance is a stock (level) measure: OFI looks at
how the inside book CHANGED between two updates, OB-imbalance looks
at the inside book RIGHT NOW. The two are weakly correlated in calm
regimes and strongly disagree in regimes where one side of the book
is being repeatedly refilled (informed flow) — exactly the regime
where the OB-imbalance signal mistakes refilling-supply for stable
supply.

Definition (per-update OFI contribution)
----------------------------------------

Given two consecutive BBO snapshots ``(bid_px_0, bid_sz_0, ask_px_0,
ask_sz_0)`` → ``(bid_px_1, bid_sz_1, ask_px_1, ask_sz_1)``:

* Bid contribution:
  - bid_px_1 > bid_px_0 → +bid_sz_1       (someone joined at a better bid)
  - bid_px_1 == bid_px_0 → +(bid_sz_1 - bid_sz_0) (size change at touch)
  - bid_px_1 < bid_px_0 → -bid_sz_0       (touch retreated; old size lost)

* Ask contribution (sign-flipped — selling pressure is negative for
  reservation):
  - ask_px_1 < ask_px_0 → -ask_sz_1       (someone joined at a better ask)
  - ask_px_1 == ask_px_0 → -(ask_sz_1 - ask_sz_0)
  - ask_px_1 > ask_px_0 → +ask_sz_0       (ask retreated; old size lost)

Sum these two and you get a signed contribution: positive = buying
pressure, negative = selling pressure.

Two EWMAs (5 s, 30 s) of this signal smooth out per-update noise.
The 5 s EWMA is the short-horizon "is buying pressure ramping right
now" signal; the 30 s is the "is the regime tilted buy-side" anchor.
The integration layer in ``compute_quote_decision`` uses the 5 s
short-horizon EWMA as the reservation shift driver.

Default-dormant (v1.5.209)
--------------------------

``OFI_RESERVATION_ALPHA=0.0`` by default — the accumulator runs and
publishes EWMAs to ``live_stats`` (for offline calibration) but does
NOT shift reservation. Operator bumps the alpha to 0.05 / 0.1 after
auditing 24 h of attribution data via Phase 3F-style postmortem
breakdown. Same pattern as the existing 4 alphas
(``OB_IMBALANCE_ALPHA``, ``BASIS_DEVIATION_ALPHA``,
``FLOW_SCORE_RESERVATION_ALPHA``, ``TREND_DRIFT_RESERVATION_ALPHA``).

Threading
---------

Updated by the public-WS handler thread (one call per BBO update);
read by the quote-loop thread (once per ~500 ms tick). Updates
mutate three floats (last-snapshot cache + two EWMAs); reads are
torn-read-safe under the GIL. No external locking needed — same
pattern as ``state.market`` and ``state.flow_score``.

Pure-state accumulator + a per-input ``record_bbo`` mutation. All
EWMA math is exposed as pure helpers so they can be unit-tested
without the accumulator state.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional


def _bbo_contribution(
    *,
    bid_px_0: float,
    bid_sz_0: float,
    ask_px_0: float,
    ask_sz_0: float,
    bid_px_1: float,
    bid_sz_1: float,
    ask_px_1: float,
    ask_sz_1: float,
) -> float:
    """Per-update OFI contribution.

    Pure function. See module docstring for the formula. Returns a
    signed float in units of "shares of net buying pressure since the
    previous BBO snapshot".
    """
    # Bid side
    if bid_px_1 > bid_px_0:
        bid_c = float(bid_sz_1)
    elif bid_px_1 < bid_px_0:
        bid_c = -float(bid_sz_0)
    else:
        bid_c = float(bid_sz_1) - float(bid_sz_0)
    # Ask side (sign flipped — bigger ask = selling pressure → negative)
    if ask_px_1 < ask_px_0:
        ask_c = -float(ask_sz_1)
    elif ask_px_1 > ask_px_0:
        ask_c = float(ask_sz_0)
    else:
        ask_c = -(float(ask_sz_1) - float(ask_sz_0))
    return bid_c + ask_c


def _ewma_step(prev: Optional[float], sample: float, dt_seconds: float, halflife_seconds: float) -> float:
    """Continuous-time EWMA update with time-aware decay.

    ``alpha = 1 - 0.5 ** (dt / halflife)``. Returns ``sample`` when
    ``prev is None`` (cold start). Halflife <= 0 → returns latest
    sample (no smoothing).
    """
    if prev is None:
        return float(sample)
    if halflife_seconds <= 0:
        return float(sample)
    if dt_seconds <= 0:
        return float(prev)
    alpha = 1.0 - 0.5 ** (dt_seconds / halflife_seconds)
    return float(prev) + alpha * (float(sample) - float(prev))


def _normalise_to_unit(raw: float, scale: float) -> float:
    """Map a raw signed signal to ``[-1, 1]`` via ``tanh(raw / scale)``.

    Tanh is the standard normaliser for OFI in the literature because
    it bounds the output without clipping (linear in the small-signal
    regime, smoothly saturating in the tails). ``scale`` controls the
    "what counts as a strong signal" calibration — tuned by the
    operator from ``OFI_NORMALISATION_SCALE``.
    """
    if scale <= 0:
        return 0.0
    return math.tanh(float(raw) / float(scale))


@dataclass(slots=True)
class OFIAccumulator:
    """Per-symbol OFI accumulator.

    Maintains the last-BBO cache and two EWMAs (5 s short-horizon,
    30 s anchor). All times are monotonic seconds.
    """

    halflife_5s_seconds: float = 5.0
    halflife_30s_seconds: float = 30.0
    normalisation_scale: float = 100.0

    # Last-BBO cache. ``None`` until the first ``record_bbo`` call.
    _last_bid_px: Optional[float] = None
    _last_bid_sz: Optional[float] = None
    _last_ask_px: Optional[float] = None
    _last_ask_sz: Optional[float] = None
    _last_ts_mono: Optional[float] = None

    # EWMAs of the raw (un-normalised) contribution. ``None`` until
    # the first contribution is recorded.
    raw_ewma_5s: Optional[float] = None
    raw_ewma_30s: Optional[float] = None

    # Sample count + last update timestamp for telemetry / warmup.
    update_count: int = 0
    last_update_mono_seconds: Optional[float] = None

    def record_bbo(
        self,
        *,
        bid_px: float,
        bid_sz: float,
        ask_px: float,
        ask_sz: float,
        now_mono_seconds: Optional[float] = None,
    ) -> None:
        """Record a new BBO snapshot, update both EWMAs.

        First call seeds the last-BBO cache and does nothing else
        (no contribution can be computed without a prior snapshot).
        Subsequent calls compute the contribution and update both
        EWMAs in place.

        ``now_mono_seconds`` defaults to ``time.monotonic()`` — pass
        explicit values from tests to drive deterministic decay.

        Defensive against non-finite inputs: silently skips the
        update (the existing cache + EWMAs are preserved). This
        prevents one bad WS message from corrupting the EWMA state.
        """
        for v in (bid_px, bid_sz, ask_px, ask_sz):
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                return
        if float(bid_px) <= 0 or float(ask_px) <= 0:
            return
        # Allow zero size (a level can be drained); reject negatives.
        if float(bid_sz) < 0 or float(ask_sz) < 0:
            return

        ts = float(now_mono_seconds) if now_mono_seconds is not None else time.monotonic()

        if self._last_bid_px is None:
            # Seed only — no contribution yet.
            self._last_bid_px = float(bid_px)
            self._last_bid_sz = float(bid_sz)
            self._last_ask_px = float(ask_px)
            self._last_ask_sz = float(ask_sz)
            self._last_ts_mono = ts
            self.last_update_mono_seconds = ts
            return

        contrib = _bbo_contribution(
            bid_px_0=float(self._last_bid_px),
            bid_sz_0=float(self._last_bid_sz or 0.0),
            ask_px_0=float(self._last_ask_px or 0.0),
            ask_sz_0=float(self._last_ask_sz or 0.0),
            bid_px_1=float(bid_px),
            bid_sz_1=float(bid_sz),
            ask_px_1=float(ask_px),
            ask_sz_1=float(ask_sz),
        )

        dt = ts - float(self._last_ts_mono or ts)
        self.raw_ewma_5s = _ewma_step(self.raw_ewma_5s, contrib, dt, self.halflife_5s_seconds)
        self.raw_ewma_30s = _ewma_step(self.raw_ewma_30s, contrib, dt, self.halflife_30s_seconds)

        self._last_bid_px = float(bid_px)
        self._last_bid_sz = float(bid_sz)
        self._last_ask_px = float(ask_px)
        self._last_ask_sz = float(ask_sz)
        self._last_ts_mono = ts
        self.last_update_mono_seconds = ts
        self.update_count += 1

    def signal_5s_normalised(self) -> Optional[float]:
        """Short-horizon normalised signal in ``[-1, 1]``.

        Returns ``None`` if the accumulator has no samples yet
        (caller treats as "no signal"). Otherwise tanh-normalised.
        """
        if self.raw_ewma_5s is None:
            return None
        return _normalise_to_unit(self.raw_ewma_5s, self.normalisation_scale)

    def signal_30s_normalised(self) -> Optional[float]:
        """30 s anchor signal in ``[-1, 1]``. ``None`` during warmup."""
        if self.raw_ewma_30s is None:
            return None
        return _normalise_to_unit(self.raw_ewma_30s, self.normalisation_scale)

    def snapshot_dict(self) -> dict:
        """Telemetry snapshot. Safe to call from any thread."""
        return {
            "update_count": int(self.update_count),
            "raw_ewma_5s": float(self.raw_ewma_5s) if self.raw_ewma_5s is not None else None,
            "raw_ewma_30s": float(self.raw_ewma_30s) if self.raw_ewma_30s is not None else None,
            "signal_5s_normalised": self.signal_5s_normalised(),
            "signal_30s_normalised": self.signal_30s_normalised(),
            "halflife_5s_seconds": float(self.halflife_5s_seconds),
            "halflife_30s_seconds": float(self.halflife_30s_seconds),
            "normalisation_scale": float(self.normalisation_scale),
            "last_update_mono_seconds": (
                float(self.last_update_mono_seconds)
                if self.last_update_mono_seconds is not None else None
            ),
        }


def compute_ofi_reservation_shift_bps(
    *,
    ofi_signal_normalised: Optional[float],
    half_spread_bps: float,
    alpha: float,
    clip_bound: float = 0.95,
) -> float:
    """Pure helper: signal → bps shift for ``compute_quote_decision``.

    Mirrors the structure of the existing 4 alphas: ``shift = alpha *
    (half_spread / 2) * signal``. Returns ``0.0`` when the signal is
    ``None`` (warmup) or the alpha is zero (dormant).
    """
    if ofi_signal_normalised is None:
        return 0.0
    if alpha <= 0.0:
        return 0.0
    try:
        s = float(ofi_signal_normalised)
    except (TypeError, ValueError):
        return 0.0
    if s != s:  # NaN
        return 0.0
    # Defensive re-clip in case the accumulator hasn't tanh-saturated.
    if s > clip_bound:
        s = clip_bound
    elif s < -clip_bound:
        s = -clip_bound
    return float(alpha) * (float(half_spread_bps) / 2.0) * s
