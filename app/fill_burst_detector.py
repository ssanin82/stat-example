"""Fill-burst detector (#3 burst defense, codex review followup
2026-05-12, from ``reports/codex-review-20260509.md`` Profitability
idea #3).

Fills clustering in time are a signature of adverse-selection
bursts: a single counterparty (or a swarm of correlated takers)
rapidly taking multiple sides of our quotes during a regime
change (news, microstructure event, vol spike). Existing gates
(``adaptive_widen``, ``vol_trend_gate``, ``post_swing_gate``) all
react to *price* signals; this one reacts to the *fill* signal
directly — and earlier, because we observe a fast fill burst
BEFORE the per-fill markouts have settled at 5 s.

Mechanic: maintain a sliding window of recent fill timestamps. When
the count within the window exceeds a threshold, arm a size-shrink
cooldown. During the cooldown the bot's per-side quote size is
multiplied by a configurable factor (typically 0.5) — smaller
exposure, same passive participation. Cooldown auto-expires.

Why size-shrink not full pause? The bot doesn't actually know yet
whether the burst is adverse (markouts unresolved). Shrinking is a
defensive bet without forfeiting all participation: if the burst is
benign flow we still earn rebate; if it's adverse we lose less per
fill.

Composes with the existing size_mult chain via ``min()`` semantics
in ``compute_quote_decision``; never grows size, only shrinks.

Disabled by default (``threshold == 0``).

Phase 2K.9 (v1.4.160) — favorable-exit predicate
------------------------------------------------

``FILL_BURST_COOLDOWN_SECONDS`` is the MAX-cooldown ceiling. On top
of it, the detector clears the size-shrink EARLY when the live
fill-in-window count has dropped below
``threshold × FILL_BURST_CLEAR_BAND_MULT`` (default mult=0.5: clear
band = half the trigger) and held there for
``FILL_BURST_FAVORABLE_EXIT_DWELL_SECONDS`` (default 5 s — short,
because each tick re-evaluates the live count).

The predicate is checked on every ``current_size_mult()`` /
``active()`` poll (typically the quote loop, 2 Hz), not just on
``note_fill``. This is important because the burst signal is "fills
have STOPPED arriving" — so a poll-time evaluation is correct;
waiting for the next fill to evaluate would miss the silent recovery.

Exit attribution: ``cleared_via_favorable_total`` vs
``cleared_via_ceiling_total`` lets the operator tune the knobs.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from app import clock as _clock


class FillBurstDetector:
    """Detects fast clusters of fills via a rolling time window.

    Use:
      detector = FillBurstDetector(threshold=3, window_seconds=30.0,
                                   size_mult=0.5, cooldown_seconds=60.0)
      detector.note_fill(now_mono=_clock.monotonic())   # on every fill
      m = detector.current_size_mult(now_mono=_clock.monotonic())
      # m is 1.0 normally; ``size_mult`` (e.g. 0.5) during a burst cooldown.

    Internals: a deque of recent fill timestamps (monotonic seconds).
    On each ``note_fill`` we append and evict expired (older than
    ``window_seconds``). If the live count >= threshold, arm the
    cooldown ``now + cooldown_seconds`` (extending if already armed
    by a fresher burst).
    """

    def __init__(
        self,
        *,
        threshold: int,
        window_seconds: float,
        size_mult: float,
        cooldown_seconds: float,
        favorable_exit_enabled: bool = True,
        clear_band_mult: float = 0.5,
        favorable_exit_dwell_seconds: float = 5.0,
    ) -> None:
        self._threshold = int(threshold)
        self._window_seconds = float(window_seconds)
        self._size_mult = float(size_mult)
        self._cooldown_seconds = float(cooldown_seconds)
        # Bounded buffer; threshold values in practice are 3-10,
        # so a deque maxlen of 256 is wasteful-but-safe.
        self._fill_ts_mono: deque[float] = deque(maxlen=256)
        self._cooldown_until_mono: float = 0.0
        self._fire_count: int = 0
        # Phase 2K.9 favorable-exit state.
        self._favorable_exit_enabled = bool(favorable_exit_enabled)
        self._clear_band_mult = float(clear_band_mult)
        self._favorable_exit_dwell_seconds = float(
            favorable_exit_dwell_seconds
        )
        self._favorable_dwell_started_mono: Optional[float] = None
        self._was_active_last_call: bool = False
        self._cleared_via_favorable_total: int = 0
        self._cleared_via_ceiling_total: int = 0

    def enabled(self) -> bool:
        return (
            self._threshold > 0
            and self._window_seconds > 0.0
            and self._size_mult < 1.0
            and self._cooldown_seconds > 0.0
        )

    def note_fill(self, now_mono: float) -> None:
        if not self.enabled():
            return
        # Evict expired timestamps before appending the new one.
        cutoff = now_mono - self._window_seconds
        while self._fill_ts_mono and self._fill_ts_mono[0] < cutoff:
            self._fill_ts_mono.popleft()
        self._fill_ts_mono.append(now_mono)
        if len(self._fill_ts_mono) >= self._threshold:
            # Arm or extend the cooldown. ``max`` here is defensive —
            # if the cooldown was set by an even-more-recent burst at
            # a longer horizon, don't shorten it.
            self._cooldown_until_mono = max(
                self._cooldown_until_mono,
                now_mono + self._cooldown_seconds,
            )
            self._fire_count += 1
            # A fresh arm cancels any in-flight favorable dwell.
            self._favorable_dwell_started_mono = None

    # ------------------------------------------------------------------
    # Phase 2K.9 favorable-exit + edge-detection internals
    # ------------------------------------------------------------------

    def _live_fill_count(self, now_mono: float) -> int:
        """Count of fills in the current window. Evicts expired
        entries as a side effect — same as ``recent_fill_count`` but
        kept private so the public API only has one entry point."""
        cutoff = now_mono - self._window_seconds
        while self._fill_ts_mono and self._fill_ts_mono[0] < cutoff:
            self._fill_ts_mono.popleft()
        return len(self._fill_ts_mono)

    def _evaluate_state_transition(self, now_mono: float) -> bool:
        """Internal: evaluate favorable-exit predicate, possibly clear
        the cooldown, and update attribution counters on the
        active→cleared edge. Returns the post-evaluation ``active``
        flag (caller uses this for ``current_size_mult`` / ``active``).

        Idempotent on repeated polls with the same ``now_mono``:
        the dwell timer is sticky, the counters only fire on the
        TRUE edge (was=True → active=False)."""
        if not self.enabled():
            return False
        was_active_by_timer = now_mono < self._cooldown_until_mono

        # Favorable-exit predicate evaluation (only meaningful while
        # the cooldown is active).
        if was_active_by_timer and self._favorable_exit_enabled:
            live = self._live_fill_count(now_mono)
            clear_band = self._threshold * self._clear_band_mult
            # threshold=3, mult=0.5 → clear_band=1.5 → predicate
            # holds when live < 1.5 (i.e. <= 1 fill in window).
            if live < clear_band:
                if self._favorable_dwell_started_mono is None:
                    self._favorable_dwell_started_mono = now_mono
                elif (
                    now_mono - self._favorable_dwell_started_mono
                    >= self._favorable_exit_dwell_seconds
                ):
                    # Dwell satisfied → clear early.
                    self._cooldown_until_mono = 0.0
                    self._cleared_via_favorable_total += 1
                    self._favorable_dwell_started_mono = None
                    self._was_active_last_call = False
                    return False
            else:
                # Re-flare — burst hasn't dissipated yet.
                self._favorable_dwell_started_mono = None

        active = now_mono < self._cooldown_until_mono
        # Ceiling attribution: was active last call, no longer active
        # now, AND no favorable clear this turn (we'd have returned
        # above). The ``cooldown_until_mono > 0`` check is implicit —
        # favorable-exit sets it to 0.0 and returns early.
        if self._was_active_last_call and not active:
            if self._cooldown_until_mono > 0.0:
                self._cleared_via_ceiling_total += 1
        self._was_active_last_call = active
        return active

    def current_size_mult(self, now_mono: float) -> float:
        """1.0 when no burst active; configured ``size_mult`` (<1.0)
        during the cooldown. Always in [size_mult, 1.0]. Polls the
        favorable-exit predicate (may clear the cooldown early)."""
        if not self.enabled():
            return 1.0
        active = self._evaluate_state_transition(now_mono)
        return self._size_mult if active else 1.0

    def active(self, now_mono: float) -> bool:
        if not self.enabled():
            return False
        return self._evaluate_state_transition(now_mono)

    def remaining_seconds(self, now_mono: float) -> float:
        if not self.enabled():
            return 0.0
        return max(0.0, self._cooldown_until_mono - now_mono)

    def recent_fill_count(self, now_mono: float) -> int:
        """Count of fills in the current window (evict-on-read)."""
        if not self.enabled():
            return 0
        return self._live_fill_count(now_mono)

    def snapshot_dict(self, now_mono: float) -> dict:
        """Diagnostic snapshot for live_stats / dashboard."""
        return {
            "enabled": self.enabled(),
            "threshold": self._threshold,
            "window_seconds": self._window_seconds,
            "size_mult": self._size_mult,
            "cooldown_seconds": self._cooldown_seconds,
            "active": self.active(now_mono),
            "seconds_remaining": self.remaining_seconds(now_mono),
            "fire_count": self._fire_count,
            "recent_fill_count": self.recent_fill_count(now_mono),
            # Phase 2K.9 favorable-exit attribution.
            "favorable_exit_enabled": self._favorable_exit_enabled,
            "clear_band_mult": self._clear_band_mult,
            "favorable_exit_dwell_seconds": (
                self._favorable_exit_dwell_seconds
            ),
            "favorable_dwell_active": (
                self._favorable_dwell_started_mono is not None
            ),
            "cleared_via_favorable_total": (
                self._cleared_via_favorable_total
            ),
            "cleared_via_ceiling_total": (
                self._cleared_via_ceiling_total
            ),
        }
