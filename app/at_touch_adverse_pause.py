"""At-touch adverse pause gate (#1 narrow, codex review followup
2026-05-12, from ``reports/codex-review-20260509.md`` Profitability
idea #1).

Closed-loop suppression on the *aggressiveness* axis of the fill
distribution. The full proposal — read the 2D bucket grid
(``quote_age × quote_aggressiveness × side``) and suppress
placements landing in negative-edge cells — is multi-week (sample-
size and calibration) and is deferred until post-colo.

This narrow version targets a specific failure mode that's observable
from a few sessions of data: **when recent ``at_touch`` fills on a
given side have median 5 s markout worse than a threshold, the bot
is being adversely selected when it joins the touch** — informed flow
is consistently hitting our at-touch quotes on that side. Hold off
the whole side for a cooldown to let the regime change.

Why suppress the whole side (not just at_touch placements)? Three
reasons:

1. At placement time we have to PREDICT the resulting aggressiveness
   (compare computed bid/ask px against current best_bid/ask). That
   logic exists for FILL classification but not for placement. Adding
   it as a closed loop is more surface; the simple version takes a
   coarser action.
2. ``behind_touch`` quotes on a bad side often eventually get walked
   through anyway (the toxic flow moves price past them). Sacrificing
   ~1 fill class for ~30 s is acceptable trade.
3. Existing whole-side ``adverse_side_pause`` provides the same
   suppression machinery; we're adding a second, more-specific
   trigger that fires on a different signal (at_touch markouts
   specifically, not aggregate side markouts).

Stateful: tracks per-side rolling at_touch markouts, arms a cooldown
timer when the trigger fires, exposes ``is_paused(side)`` and
``remaining_seconds(side)`` for the dashboard.

Disabled by default (``threshold_bps == 0.0``) so existing deploys
are unaffected.

Phase 2K.6 (v1.4.156) — favorable-exit predicate:

The cooldown is no longer purely time-based. While the timer is the
MAX-cooldown ceiling (preserved at ``AT_TOUCH_ADVERSE_PAUSE_SECONDS``),
the gate also clears EARLY when the per-side median markout recovers
past ``threshold_bps × clear_band_mult`` and holds there for
``favorable_exit_dwell_seconds``. The dwell + hysteresis filter prevents
quantization-induced flicker (each new at-touch fill snaps the median
by a discrete amount).

Exit attribution: per-side ``cleared_via_favorable_total`` and
``cleared_via_ceiling_total`` counters track which path cleared each
pause, so the operator can tune ``clear_band_mult`` from the dashboard
(>50 % via_favorable = predicate is doing meaningful work).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from app.enums import Side
from app.models import Fill


class AtTouchAdversePause:
    """Per-side at_touch adverse-fill detector + pause arming.

    The trigger condition is ``median(recent_at_touch_markouts_5s_bps) <
    threshold_bps`` over at least ``min_fills`` samples. Median is
    used (not mean) so a single −40 bp outlier doesn't drag the
    aggregate past threshold and arm the gate spuriously.

    On arm: ``_paused_until_mono[side]`` is set ``now + pause_seconds``.
    Further fills extend the timer when they re-trigger; otherwise it
    expires naturally OR — when ``favorable_exit_enabled`` — clears
    early when the median recovers past the clear band for the
    configured dwell.
    """

    def __init__(
        self,
        *,
        threshold_bps: float,
        pause_seconds: float,
        min_fills: int,
        window_size: int = 20,
        favorable_exit_enabled: bool = True,
        clear_band_mult: float = 0.5,
        favorable_exit_dwell_seconds: float = 5.0,
    ) -> None:
        self._threshold_bps = float(threshold_bps)
        self._pause_seconds = float(pause_seconds)
        self._min_fills = int(min_fills)
        self._buy_markouts: deque[float] = deque(maxlen=window_size)
        self._sell_markouts: deque[float] = deque(maxlen=window_size)
        self._buy_paused_until_mono: float = 0.0
        self._sell_paused_until_mono: float = 0.0
        self._buy_fire_count: int = 0
        self._sell_fire_count: int = 0
        # Phase 2K.6 favorable-exit state.
        self._favorable_exit_enabled = bool(favorable_exit_enabled)
        self._clear_band_mult = float(clear_band_mult)
        self._favorable_exit_dwell_seconds = float(favorable_exit_dwell_seconds)
        self._buy_favorable_dwell_started_mono: Optional[float] = None
        self._sell_favorable_dwell_started_mono: Optional[float] = None
        self._buy_was_paused_last: bool = False
        self._sell_was_paused_last: bool = False
        self._buy_cleared_via_favorable_total: int = 0
        self._sell_cleared_via_favorable_total: int = 0
        self._buy_cleared_via_ceiling_total: int = 0
        self._sell_cleared_via_ceiling_total: int = 0
        # v1.5.157 — position-aware favorable-exit attribution.
        # Same pattern as realised_edge_side_suppress: the markout-
        # based exit (above) requires NEW fills to lift the per-side
        # median past the clear band, but during a pause there ARE
        # no new fills on the paused side → predicate never re-
        # evaluates to favorable. The position-aware exit clears the
        # pause when that side is the REDUCING side for current
        # inventory ("reducing side must always be available").
        self._buy_cleared_via_position_favorable_total: int = 0
        self._sell_cleared_via_position_favorable_total: int = 0

    def enabled(self) -> bool:
        return (
            self._threshold_bps != 0.0
            and self._pause_seconds > 0.0
            and self._min_fills > 0
        )

    def observe_resolved_5s_markout(self, f: Fill, now_mono: float) -> None:
        """Called from the markout resolution path once a fill's 5 s
        markout is finalized. Ignores fills that aren't at_touch and
        fills with a missing markout. Side effect: may arm the pause
        for the fill's side; may also clear the pause early when the
        favorable-exit predicate flips True with the new sample.
        """
        if not self.enabled():
            return
        if getattr(f, "quote_aggressiveness", None) != "at_touch":
            return
        markout = getattr(f, "markout_5s_bps", None)
        if markout is None:
            return
        try:
            mv = float(markout)
        except (TypeError, ValueError):
            return
        if f.side == Side.BUY:
            self._buy_markouts.append(mv)
            if self._check_trigger(self._buy_markouts):
                self._buy_paused_until_mono = now_mono + self._pause_seconds
                self._buy_fire_count += 1
                # Re-arm cancels any in-flight favorable dwell — the
                # signal is freshly bad again.
                self._buy_favorable_dwell_started_mono = None
        elif f.side == Side.SELL:
            self._sell_markouts.append(mv)
            if self._check_trigger(self._sell_markouts):
                self._sell_paused_until_mono = now_mono + self._pause_seconds
                self._sell_fire_count += 1
                self._sell_favorable_dwell_started_mono = None
        # After updating the deque, re-evaluate favorable-exit so a
        # new sample that pushes the median above the clear band can
        # contribute to the dwell (or trip the dwell if it'd been
        # accumulating off the previous polls).
        self._maybe_clear_via_favorable_exit(f.side, now_mono)

    def _check_trigger(self, markouts: deque[float]) -> bool:
        if len(markouts) < self._min_fills:
            return False
        # Use the most-recent ``min_fills`` samples for the median.
        # Older samples in the deque are kept for observability but
        # don't dilute the trigger sensitivity.
        recent = sorted(list(markouts)[-self._min_fills:])
        n = len(recent)
        if n % 2 == 1:
            median = recent[n // 2]
        else:
            median = (recent[n // 2 - 1] + recent[n // 2]) / 2.0
        return median < self._threshold_bps

    # ------------------------------------------------------------------
    # Phase 2K.6 favorable-exit predicate
    # ------------------------------------------------------------------

    def _current_median(self, side: Side) -> Optional[float]:
        """Return the median of the most-recent ``min_fills`` samples,
        or ``None`` when there aren't enough samples yet."""
        d = (
            self._buy_markouts if side == Side.BUY
            else self._sell_markouts if side == Side.SELL
            else None
        )
        if d is None or len(d) < self._min_fills:
            return None
        recent = sorted(list(d)[-self._min_fills:])
        n = len(recent)
        if n % 2 == 1:
            return recent[n // 2]
        return (recent[n // 2 - 1] + recent[n // 2]) / 2.0

    def _favorable_predicate_holds(self, side: Side) -> bool:
        """True iff the per-side median has recovered past the clear
        band (``threshold_bps × clear_band_mult``). With
        ``threshold_bps`` negative and ``clear_band_mult`` in [0, 1],
        the clear band sits between zero and the trigger value:
        smaller mult → stronger recovery required.
        """
        if not self._favorable_exit_enabled:
            return False
        median = self._current_median(side)
        if median is None:
            return False
        clear_band = self._threshold_bps * self._clear_band_mult
        return median > clear_band

    def _maybe_clear_via_favorable_exit(
        self, side: Side, now_mono: float
    ) -> bool:
        """Evaluate the favorable-exit predicate for ``side`` and
        possibly clear the pause early. Returns True iff this call
        cleared the pause via favorable-exit (counter bumped). Idempotent
        when the gate is not paused.
        """
        if not self._favorable_exit_enabled:
            return False
        paused_until = (
            self._buy_paused_until_mono if side == Side.BUY
            else self._sell_paused_until_mono if side == Side.SELL
            else 0.0
        )
        if now_mono >= paused_until:
            # Not currently paused-by-timer → nothing to short-circuit.
            return False

        if self._favorable_predicate_holds(side):
            # Start or continue the dwell.
            if side == Side.BUY:
                if self._buy_favorable_dwell_started_mono is None:
                    self._buy_favorable_dwell_started_mono = now_mono
                dwell_started = self._buy_favorable_dwell_started_mono
            else:  # SELL
                if self._sell_favorable_dwell_started_mono is None:
                    self._sell_favorable_dwell_started_mono = now_mono
                dwell_started = self._sell_favorable_dwell_started_mono
            if now_mono - dwell_started >= self._favorable_exit_dwell_seconds:
                # Dwell satisfied → clear pause + bump favorable counter.
                if side == Side.BUY:
                    self._buy_paused_until_mono = 0.0
                    self._buy_cleared_via_favorable_total += 1
                    self._buy_favorable_dwell_started_mono = None
                    self._buy_was_paused_last = False
                else:  # SELL
                    self._sell_paused_until_mono = 0.0
                    self._sell_cleared_via_favorable_total += 1
                    self._sell_favorable_dwell_started_mono = None
                    self._sell_was_paused_last = False
                return True
        else:
            # Predicate not holding — reset dwell (re-flare hysteresis).
            if side == Side.BUY:
                self._buy_favorable_dwell_started_mono = None
            else:
                self._sell_favorable_dwell_started_mono = None
        return False

    def try_clear_via_position_favorable(
        self,
        *,
        side: Side,
        now_mono: float,
        position_qty: float,
        inventory_threshold: float,
    ) -> bool:
        """v1.5.157 — position-aware favorable-exit predicate.

        Clears the pause on ``side`` if that side is the REDUCING side
        for current inventory (the markout-based exit can't fire while
        paused because there are no new fills to lift the median):

        * BUY pause clears when ``position_qty <= -inventory_threshold``
          (bot is SHORT; BUY reduces SHORT)
        * SELL pause clears when ``position_qty >= +inventory_threshold``
          (bot is LONG; SELL reduces LONG)

        Mirrors the realised_edge_side_suppress v1.5.155 fix. The
        ADDING side stays paused (the original at-touch-adverse
        concern is about further adverse fills on the bleeding side,
        which is exactly what the reducing-side carve-out preserves).

        Per CLAUDE.md Rule 0c. Idempotent: returns False without
        side effects when the side is not currently paused OR the
        position predicate doesn't hold.

        Returns True if this call cleared the pause.
        """
        if not self.enabled():
            return False
        if side not in (Side.BUY, Side.SELL):
            return False
        until = (
            self._buy_paused_until_mono if side == Side.BUY
            else self._sell_paused_until_mono
        )
        if now_mono >= until:
            return False
        threshold = float(inventory_threshold)
        if side == Side.BUY:
            if position_qty > -threshold:
                return False
            self._buy_paused_until_mono = 0.0
            self._buy_cleared_via_position_favorable_total += 1
            self._buy_favorable_dwell_started_mono = None
            self._buy_was_paused_last = False
        else:
            if position_qty < threshold:
                return False
            self._sell_paused_until_mono = 0.0
            self._sell_cleared_via_position_favorable_total += 1
            self._sell_favorable_dwell_started_mono = None
            self._sell_was_paused_last = False
        return True

    def try_clear_via_idle(
        self,
        *,
        now_mono: float,
        last_fill_mono: Optional[float],
        idle_clear_seconds: float,
    ) -> bool:
        """v1.5.197 — idle-decay exit predicate (both sides).

        Clears any active pause when no fill has arrived for
        ``idle_clear_seconds``. Rationale: the pause is driven by
        recent fill markouts; if those fills are stale (no new data
        for N minutes because some upstream defense is suppressing
        flow), the pause should age out too.

        Mirrors ``mae_gate.evaluate_idle_clear``. Eliminates the
        defensive-deadlock pattern where the pause itself blocks
        the very signal (new fill markouts) that would clear it.

        Returns True if this call cleared at least one side.
        """
        if not self.enabled():
            return False
        if last_fill_mono is None:
            return False
        if idle_clear_seconds <= 0.0:
            return False
        if (now_mono - float(last_fill_mono)) < idle_clear_seconds:
            return False
        cleared_any = False
        if now_mono < self._buy_paused_until_mono:
            self._buy_paused_until_mono = 0.0
            self._buy_cleared_via_position_favorable_total += 1
            self._buy_favorable_dwell_started_mono = None
            self._buy_was_paused_last = False
            cleared_any = True
        if now_mono < self._sell_paused_until_mono:
            self._sell_paused_until_mono = 0.0
            self._sell_cleared_via_position_favorable_total += 1
            self._sell_favorable_dwell_started_mono = None
            self._sell_was_paused_last = False
            cleared_any = True
        return cleared_any

    def is_paused(self, side: Side, now_mono: float) -> bool:
        """Return True iff ``side`` is currently paused. Side effect:
        evaluates the favorable-exit predicate (may clear the pause
        early) and tracks the active→cleared edge for ceiling
        attribution. Polled from the bot's hot path each tick."""
        if not self.enabled():
            return False
        if side not in (Side.BUY, Side.SELL):
            return False

        # Compute the pre-update was-paused state so we can attribute
        # natural-expiry edges to the ceiling counter below.
        paused_until = (
            self._buy_paused_until_mono if side == Side.BUY
            else self._sell_paused_until_mono
        )
        was_paused_by_timer = now_mono < paused_until

        # Maybe short-circuit via favorable-exit. Idempotent when not
        # paused.
        cleared_via_favorable_this_call = (
            self._maybe_clear_via_favorable_exit(side, now_mono)
        )

        # Re-read after possible favorable clear.
        paused_until = (
            self._buy_paused_until_mono if side == Side.BUY
            else self._sell_paused_until_mono
        )
        currently_paused = now_mono < paused_until

        # Edge attribution: was paused last call, no longer paused now,
        # AND not cleared via favorable in this turn → ceiling fired.
        was_paused_last = (
            self._buy_was_paused_last if side == Side.BUY
            else self._sell_was_paused_last
        )
        if (
            was_paused_last
            and not currently_paused
            and not cleared_via_favorable_this_call
        ):
            if side == Side.BUY:
                self._buy_cleared_via_ceiling_total += 1
            else:
                self._sell_cleared_via_ceiling_total += 1

        # Update was_paused_last for next call.
        if side == Side.BUY:
            self._buy_was_paused_last = currently_paused
        else:
            self._sell_was_paused_last = currently_paused

        _ = was_paused_by_timer  # kept for clarity; not used downstream
        return currently_paused

    def remaining_seconds(self, side: Side, now_mono: float) -> float:
        if not self.enabled():
            return 0.0
        if side == Side.BUY:
            return max(0.0, self._buy_paused_until_mono - now_mono)
        if side == Side.SELL:
            return max(0.0, self._sell_paused_until_mono - now_mono)
        return 0.0

    def snapshot_dict(self, now_mono: float) -> dict:
        """Diagnostic snapshot for live_stats / dashboard. Includes
        recent-fill counts, current medians, remaining cooldown
        seconds per side, and Phase 2K.6 exit-attribution counters."""
        def _median_or_none(d: deque[float]) -> Optional[float]:
            if len(d) < self._min_fills:
                return None
            recent = sorted(list(d)[-self._min_fills:])
            n = len(recent)
            if n % 2 == 1:
                return recent[n // 2]
            return (recent[n // 2 - 1] + recent[n // 2]) / 2.0

        return {
            "enabled": self.enabled(),
            "threshold_bps": self._threshold_bps,
            "min_fills": self._min_fills,
            "pause_seconds": self._pause_seconds,
            # Phase 2K.6 settings surfaced for dashboard calibration.
            "favorable_exit_enabled": self._favorable_exit_enabled,
            "clear_band_mult": self._clear_band_mult,
            "favorable_exit_dwell_seconds": self._favorable_exit_dwell_seconds,
            "buy": {
                "recent_count": len(self._buy_markouts),
                "recent_median_bps": _median_or_none(self._buy_markouts),
                "paused": self.is_paused(Side.BUY, now_mono),
                "seconds_remaining": self.remaining_seconds(Side.BUY, now_mono),
                "fire_count": self._buy_fire_count,
                "cleared_via_favorable_total": (
                    self._buy_cleared_via_favorable_total
                ),
                "cleared_via_ceiling_total": (
                    self._buy_cleared_via_ceiling_total
                ),
                "cleared_via_position_favorable_total": (
                    self._buy_cleared_via_position_favorable_total
                ),
                "favorable_dwell_active": (
                    self._buy_favorable_dwell_started_mono is not None
                ),
            },
            "sell": {
                "recent_count": len(self._sell_markouts),
                "recent_median_bps": _median_or_none(self._sell_markouts),
                "paused": self.is_paused(Side.SELL, now_mono),
                "seconds_remaining": self.remaining_seconds(Side.SELL, now_mono),
                "fire_count": self._sell_fire_count,
                "cleared_via_favorable_total": (
                    self._sell_cleared_via_favorable_total
                ),
                "cleared_via_ceiling_total": (
                    self._sell_cleared_via_ceiling_total
                ),
                "cleared_via_position_favorable_total": (
                    self._sell_cleared_via_position_favorable_total
                ),
                "favorable_dwell_active": (
                    self._sell_favorable_dwell_started_mono is not None
                ),
            },
        }
