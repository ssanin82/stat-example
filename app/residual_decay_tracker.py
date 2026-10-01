"""Phase 2D (v1.5.26) -- residual-decay trigger for adaptive_widen.

The bot already widens the spread reactively when toxicity / one-sided
fill ratio / quote-quality / slow-trend / markout-adverse signals
fire (see ``app/bot.py``'s adaptive_widen arming block). Each of those
triggers reads a *single signal* in isolation. The residual-decay
trigger reads a *composite*: ``closed_pnl_bps + rebate_bps -
markout_5s_bps`` per fill, rolling over the last N fills.

Why a composite signal? Each component alone misses cases:

* Pure markout-adverse: catches "we got picked off" but blind to "we
  closed at a profit anyway so the fill was net-positive".
* Pure rebate: silent when rebates fully offset markout losses.
* Pure closed-PnL: misses the long-tail markout drag.

The residual is the bot's NET economic signal after rebates and
closed-PnL: if it's persistently negative, the bot is bleeding even
though the immediate signals look normal. Plan 2D's intent: arm the
adaptive_widen gate (which the bot already has for the simpler
triggers) when residual_bps drops below ``threshold_bps`` (a negative
number; default -0.5) and STAYS there for ``dwell_seconds`` (60s)
of accumulated fills. The cooldown clears via the existing
adaptive_widen ceiling (``TOXICITY_COOLDOWN_SECONDS``); 2K.5's
favorable-exit predicate for ``residual_decay`` reason is
intentionally ceiling-only (no early clearing).

This module is the rolling-window tracker. The arming integration
lives in ``app/bot.py``'s adaptive_widen block (one new ``want_arm_*``
branch). The favorable-exit predicate already returns False for
unknown reasons in ``app/quoting.py::adaptive_spread_widen_signal_cleared``
-- "residual_decay" inherits that behaviour without code change.

Disabled by default (``threshold_bps == 0.0``).
"""

from __future__ import annotations

from collections import deque
from typing import Optional


class ResidualDecayTracker:
    """Rolling N-fill window of net residual edge per fill.

    Trigger: the rolling mean of the buffer drops below
    ``threshold_bps`` (a negative number) AND has been below
    continuously for at least ``dwell_seconds`` of resolved-5s fills.
    The dwell is measured in WALL TIME between the first
    below-threshold sample and the most recent below-threshold
    sample; a single recovery sample resets the dwell timer.

    The tracker is observation-only. The arming integration lives
    in ``app/bot.py``'s adaptive_widen block, which polls
    ``is_armed(now_mono)`` once per tick.
    """

    def __init__(
        self,
        *,
        threshold_bps: float,
        dwell_seconds: float,
        window_fills: int,
        min_fills: int,
    ) -> None:
        self._threshold_bps = float(threshold_bps)
        self._dwell_seconds = float(dwell_seconds)
        self._min_fills = int(min_fills)
        # Buffer holds the per-fill residual_bps values. Capped at
        # ``window_fills``; older samples drop off when the window
        # fills. The rolling mean is computed fresh on each call
        # (window is small, the cost is negligible).
        self._buffer: deque[float] = deque(maxlen=int(window_fills))
        # Monotonic timestamp when the rolling mean first dropped
        # below threshold. ``None`` when the mean is currently above
        # threshold (or the buffer is empty / pre-min_fills).
        self._below_since_mono: Optional[float] = None
        # Session-cumulative diagnostic counters.
        self._fire_count: int = 0
        self._last_trigger_mean_bps: float = 0.0

    @property
    def threshold_bps(self) -> float:
        return self._threshold_bps

    @property
    def dwell_seconds(self) -> float:
        return self._dwell_seconds

    @property
    def fire_count(self) -> int:
        return self._fire_count

    @property
    def last_trigger_mean_bps(self) -> float:
        return self._last_trigger_mean_bps

    def enabled(self) -> bool:
        """Disabled when threshold is 0 or positive (the trigger only
        makes sense for a negative threshold), or when the window is
        empty, or when min_fills is non-positive."""
        return (
            self._threshold_bps < 0.0
            and self._buffer.maxlen is not None
            and self._buffer.maxlen > 0
            and self._min_fills > 0
            and self._dwell_seconds >= 0.0
        )

    def current_mean_bps(self) -> Optional[float]:
        """Rolling mean across the buffer. ``None`` when the buffer
        has fewer than ``min_fills`` samples (not enough signal)."""
        if len(self._buffer) < self._min_fills:
            return None
        return sum(self._buffer) / len(self._buffer)

    def note_fill(
        self,
        *,
        residual_bps: Optional[float],
        now_mono: float,
    ) -> None:
        """Append a fill's residual_bps to the rolling window and
        update the dwell timer.

        ``residual_bps`` = ``closed_pnl_bps + rebate_bps -
        markout_5s_bps`` -- caller computes from the fill's
        ``closed_pnl`` / ``fee`` / ``markout_5s_bps`` / ``notional``
        fields. Caller passes ``None`` when any input is missing
        (e.g. ``markout_5s_bps`` not yet resolved); the tracker
        silently skips -- the buffer length only grows on fills with
        a complete residual.

        The dwell-timer update:
        * If new mean is below threshold AND we weren't tracking yet
          -> start the dwell clock at ``now_mono``.
        * If new mean rises above threshold -> reset dwell clock.
        * If new mean stays below threshold -> leave dwell-start
          unchanged so ``is_armed()`` can measure (now - start).
        """
        if not self.enabled():
            return
        if residual_bps is None:
            return
        try:
            v = float(residual_bps)
        except (TypeError, ValueError):
            return
        self._buffer.append(v)

        mean = self.current_mean_bps()
        if mean is None:
            # Still below ``min_fills`` -- can't form a verdict.
            self._below_since_mono = None
            return

        if mean < self._threshold_bps:
            if self._below_since_mono is None:
                # First crossing into the below-threshold zone.
                self._below_since_mono = float(now_mono)
        else:
            # Recovery -- reset the dwell clock.
            if self._below_since_mono is not None:
                self._below_since_mono = None

    def is_armed(self, now_mono: float) -> bool:
        """Returns True when the rolling mean has been below
        ``threshold_bps`` continuously for at least ``dwell_seconds``.

        Side effect: increments ``fire_count`` + stamps
        ``last_trigger_mean_bps`` on the FIRST tick the predicate
        evaluates True. Callers (the adaptive_widen arming block in
        bot.py) typically poll once per tick; the first-tick
        increment matches Phase 2K's "arm once per episode" semantics.

        Idempotent within an episode: stays True until the rolling
        mean recovers above threshold (which resets the dwell timer
        via ``note_fill``). Once recovered, the next decline starts
        a fresh dwell.
        """
        if not self.enabled():
            return False
        mean = self.current_mean_bps()
        if mean is None:
            return False
        if mean >= self._threshold_bps:
            return False
        if self._below_since_mono is None:
            return False
        dwell = float(now_mono) - self._below_since_mono
        return dwell >= self._dwell_seconds

    def consume_arm(self, now_mono: float) -> None:
        """Bump fire_count + stamp diagnostic mean. Called by the
        adaptive_widen arming block AFTER it decides to use the
        residual_decay reason. Splits the predicate poll (``is_armed``,
        idempotent / side-effect-free) from the side-effect of
        arming. Without this split a polled-but-not-acted-on
        predicate would inflate fire_count."""
        self._fire_count += 1
        mean = self.current_mean_bps()
        if mean is not None:
            self._last_trigger_mean_bps = mean

    def snapshot_dict(self) -> dict[str, object]:
        """Operator-visible state. Mirrors the shape used by other
        Phase 2K gates so the dashboard / postmortem readers can
        treat it uniformly."""
        return {
            "enabled": self.enabled(),
            "threshold_bps": self._threshold_bps,
            "dwell_seconds": self._dwell_seconds,
            "min_fills": self._min_fills,
            "window_fills": self._buffer.maxlen,
            "current_mean_bps": self.current_mean_bps(),
            "buffer_size": len(self._buffer),
            "below_threshold_dwell_seconds_active": (
                None if self._below_since_mono is None else "tracking"
            ),
            "fire_count": self._fire_count,
            "last_trigger_mean_bps": self._last_trigger_mean_bps,
        }


def compute_residual_bps(
    *,
    closed_pnl: Optional[float],
    fee: float,
    markout_5s_bps: Optional[float],
    notional: float,
) -> Optional[float]:
    """Compute the residual edge in bps for a single fill.

    Formula: ``closed_pnl_bps + rebate_bps - markout_5s_bps``.

    * ``closed_pnl_bps = (closed_pnl / notional) * 1e4``  -- realised
      PnL from this fill closing some position. Zero on pure opening
      fills, non-zero on reducing/closing fills. Read from
      ``Fill.closed_pnl`` (populated by ingest from venue events).
    * ``rebate_bps = (-fee / notional) * 1e4``  -- maker rebate
      received (fee < 0 means rebate; convention matches
      ``realised_edge_side_suppress``).
    * ``markout_5s_bps`` -- adverse-selection signal. Positive means
      mid moved AGAINST our fill direction (adverse). The formula
      subtracts it so the residual is "net gain after adverse
      selection".

    Returns ``None`` when markout_5s_bps isn't yet resolved or when
    notional is non-positive (defensive). closed_pnl is treated as
    0.0 when missing (the common opening-fill case).
    """
    if markout_5s_bps is None:
        return None
    if notional is None or notional <= 0:
        return None
    try:
        n = float(notional)
        mk = float(markout_5s_bps)
        fe = float(fee or 0.0)
        cp = float(closed_pnl or 0.0)
    except (TypeError, ValueError):
        return None
    closed_pnl_bps = (cp / n) * 1e4
    rebate_bps = (-fe / n) * 1e4
    return closed_pnl_bps + rebate_bps - mk
