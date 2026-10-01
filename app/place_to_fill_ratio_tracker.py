"""Rolling place-to-fill ratio tracker.

Detects the failure mode where the bot is placing far more orders
than it fills — i.e., quotes are too far from the touch and end up
cancelled before they can interact with the book. A healthy MM
strategy on a liquid pair sits in the 5–30 % fills/places range; <1 %
means the bot is hedge-fund-flavoured churn and should be paused.

Real incident driving this: 2026-05-05 22:35 UTC, OKX SUI-USDT-SWAP.
3 048 places against 25 fills (~0.8 %) over 4 h, all while the bot's
own kill threshold (max_drawdown / max_session_loss) was nowhere near
firing because the strategy wasn't *losing* money in any one
direction — it was just spinning. Drawdown at kill was $0.28; the
real cost was the 4 h of compute and rate-limit budget burned for
nothing.

Window semantics (mirrors execution-error window):
  * ``window_seconds`` rolling deque of place + fill timestamps
    (``time.monotonic`` so it's wall-clock-jump immune)
  * Pruned on every read; O(N) per call but N is bounded by the
    quote loop cadence × window (~600 events at 0.5 Hz / 600 s).
  * Gate evaluation (``evaluate``) returns one of:
      ``ok``           — above the pause threshold OR not enough
                         places yet to judge.
      ``below_pause``  — fill-ratio below the pause threshold.
      ``below_kill``   — fill-ratio below the kill threshold; bot
                         should kill, not just pause.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from app import clock as _clock


@dataclass(frozen=True, slots=True)
class PlaceToFillSnapshot:
    """Read-only view returned by :meth:`PlaceToFillRatioTracker.snapshot`."""

    places: int
    fills: int
    ratio_pct: Optional[float]  # ``None`` when ``places == 0``
    window_seconds: float


class PlaceToFillRatioTracker:
    """Rolling place + fill counters in a sliding monotonic window.

    Two deques (places, fills) of ``_clock.monotonic()`` floats. Tail
    is pruned on read; ingest is O(1).
    """

    def __init__(self, *, window_seconds: float) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._lock = threading.Lock()
        self._window_s = float(window_seconds)
        self._places: deque[float] = deque()
        self._fills: deque[float] = deque()

    def note_place(self) -> None:
        """Record one place attempt. Best-effort; never raises."""
        ts = _clock.monotonic()
        with self._lock:
            self._places.append(ts)

    def note_fill(self) -> None:
        """Record one fill. Best-effort; never raises."""
        ts = _clock.monotonic()
        with self._lock:
            self._fills.append(ts)

    def _prune_unlocked(self, now: float) -> None:
        cutoff = now - self._window_s
        while self._places and self._places[0] < cutoff:
            self._places.popleft()
        while self._fills and self._fills[0] < cutoff:
            self._fills.popleft()

    def snapshot(self) -> PlaceToFillSnapshot:
        """Returns the current windowed (places, fills, ratio_pct).

        ``ratio_pct`` is ``None`` while ``places == 0`` (avoid the
        meaningless 0/0 division and avoid tripping the gate during
        the first ticks of a fresh session before any places land).
        """
        now = _clock.monotonic()
        with self._lock:
            self._prune_unlocked(now)
            p = len(self._places)
            f = len(self._fills)
            ratio: Optional[float] = (
                None if p == 0 else round((f / p) * 100.0, 3)
            )
            return PlaceToFillSnapshot(
                places=p,
                fills=f,
                ratio_pct=ratio,
                window_seconds=self._window_s,
            )

    def evaluate(
        self,
        *,
        min_places_before_gate: int,
        pause_pct: float,
        kill_pct: float,
    ) -> tuple[str, PlaceToFillSnapshot]:
        """Classify the current ratio.

        Returns ``(verdict, snapshot)`` where ``verdict`` is one of
        ``"ok"``, ``"below_pause"``, ``"below_kill"``. The snapshot
        is the same object :meth:`snapshot` would return — handed
        back so the caller can include the numbers in the kill /
        pause event payload without a second prune.

        ``min_places_before_gate`` is a hold-off: we don't pause /
        kill on a 0-of-5 sample, only after at least N places have
        landed in the window. Tune to roughly 1× quote-loop ×
        bid+ask = ``window_seconds × 2 × cadence_hz`` for the lower
        bound; default 100 ≈ 100 s of normal quoting at 0.5 Hz.
        """
        snap = self.snapshot()
        if snap.places < int(min_places_before_gate):
            return "ok", snap
        # ``ratio_pct`` is non-None whenever places > 0, which is
        # implied by ``min_places_before_gate >= 1`` above.
        ratio = snap.ratio_pct or 0.0
        if ratio < float(kill_pct):
            return "below_kill", snap
        if ratio < float(pause_pct):
            return "below_pause", snap
        return "ok", snap
