from __future__ import annotations

import math
from collections import deque
from typing import Optional

from app.config import Settings


class VolatilityEstimator:
    """
    Rolling sample standard deviation of log-returns between consecutive mids.

    ``sigma`` is in log-return space per quote step (not annualized).
    ``vol_bps`` is ``sigma * 10_000`` as a rough same-scale input to the spread adder.

    1.2.24 (todo-010 fix): the deque now holds CHANGED mids only.
    Pre-fix the caller pushed `state.market.mid_price` once per
    quote cycle (every 500 ms), but on tight-tick venues (OKX SUI,
    1 bp tick) consecutive 500 ms reads frequently return the SAME
    mid — the venue's BBO didn't tick between snapshots. The deque
    then filled with 32 identical values → log-return variance = 0
    → vol_bps = 0. Three vol-conditioned defenses
    (vol_trend_gate, vol_regime_shrink_factor, VOL_SPIKE_*) were
    silently inert as a result.

    The fix is a one-line dedup at push time: skip the append when
    the new mid equals the deque's tail. Now the deque samples
    actual price changes regardless of how slowly the venue ticks
    relative to the quote loop. Trade-off: warm-up is slower in
    quiet markets (need 32 distinct mid values, not 32 cycles) —
    but in genuinely flat markets, vol IS effectively zero and
    reading 0 is the correct answer for that regime.
    """

    def __init__(self, settings: Settings) -> None:
        self._window = max(4, settings.vol_window_samples)
        self._mids: deque[float] = deque(maxlen=self._window + 1)
        self._last_sigma: Optional[float] = None
        # M8 Candidate A — recorder warm-start seed. When the tape
        # runtime feed supplies a 24 h vol estimate at startup
        # (``vol_bps_p95_24h``), we stash it here as a per-step sigma and
        # serve it from ``sigma_and_bps`` ONLY during the cold-start
        # window (before the live estimator has ``_window`` distinct
        # mids). Once warmed, the live computation takes over and the
        # seed is never consulted again — a warm-START, not a permanent
        # override. ``None`` until ``seed_from_recorder`` succeeds.
        self._seed_sigma: Optional[float] = None

    def push_mid(self, mid: Optional[float]) -> None:
        if mid is None or mid <= 0:
            return
        # 1.2.24: dedup. Skip the push when the venue mid hasn't
        # ticked since the last sample. Without this, the variance
        # calculation sees mostly zero log-returns from the
        # cycle-frequency oversampling and reports vol_bps=0 even
        # in active markets.
        if self._mids and self._mids[-1] == mid:
            return
        self._mids.append(mid)

    def seed_from_recorder(self, vol_bps: Optional[float]) -> bool:
        """Warm-start the estimator from the recorder's 24 h vol estimate
        (M8 Candidate A). ``vol_bps`` is a same-scale bps figure
        (``sigma * 10_000``); we store it as a per-step sigma.

        One-shot, called once at bot startup. Returns ``True`` if a usable
        seed was stored, ``False`` (no-op) when the field is absent /
        non-finite / non-positive — which is the current production state
        because ``vol_bps_p95_24h`` is still a dark (``NaN``) field on the
        wire. The bot's caller increments
        ``warmstart_vol_seeded_from_recorder_count`` only on ``True``."""
        if vol_bps is None or not math.isfinite(vol_bps) or vol_bps <= 0.0:
            return False
        self._seed_sigma = float(vol_bps) / 10_000.0
        return True

    @property
    def warmed_up(self) -> bool:
        return len(self._mids) >= self._window

    def sigma_and_bps(self) -> tuple[Optional[float], float]:
        if not self.warmed_up or len(self._mids) < 2:
            # Cold start: the live estimator hasn't seen ``_window``
            # distinct mids yet. If the recorder supplied a warm-start
            # seed, serve it so vol-conditioned logic isn't inert during
            # the warm-up window. Superseded by the live computation the
            # moment ``warmed_up`` flips True (this branch stops running).
            if self._seed_sigma is not None:
                self._last_sigma = self._seed_sigma
                return self._seed_sigma, self._seed_sigma * 10_000.0
            self._last_sigma = None
            return None, 0.0
        mids = list(self._mids)[-self._window :]
        rets: list[float] = []
        for i in range(1, len(mids)):
            a, b = mids[i - 1], mids[i]
            if a > 0 and b > 0:
                rets.append(math.log(b / a))
        if len(rets) < 2:
            return None, 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sigma = math.sqrt(max(var, 0.0))
        self._last_sigma = sigma
        # per-tick sigma -> rough bps equivalent on mid
        vol_bps = sigma * 10_000.0
        return sigma, vol_bps
