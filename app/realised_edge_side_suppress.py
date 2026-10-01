"""Phase 4C.3 mini (v1.4.161) — Per-side realised-edge side suppression.

Shipped out-of-order from the main Phase 4C arc (the maturity-doc's
"single biggest gap": today the bot survives via Phase 1 defences but
doesn't actively *stand aside* when expected-edge is negative). This
narrow piece targets one specific failure mode that v1.4.157's
inventory-skew tune did NOT address — see snapshot
``v1.4.157-260520-213540-prod.okx.ton.usdt.perp``:

* 21:13:54 Dubai — bot's resting SELL @ $2.042 picked off as TON
  began a +88 bp rally; ``markout_5s_bps = -31.83`` for that fill.
* Bot had already accumulated 3 SELL fills in the preceding ~3 min
  whose 5 s markouts were +2.5 / +2.5 / -2.5 → trailing-mean OK, BUT
  the latest one alone dragged the 4-fill mean to **−7.35 bps**.
* shock_gate / inventory_drift_gate / momentum_gate all share a
  ``util ≥ 0.60-0.80`` arming gate — by the time util crosses, the
  damage is done. The realised-edge signal has no util prerequisite:
  it fires the moment the per-side trailing markout degrades.

Mechanism (mirrors ``app.at_touch_adverse_pause`` for consistency):
maintain a per-side rolling N-fill buffer of
``markout_5s_bps + fee_rebate_bps`` (the NET realised edge). When the
trailing mean drops below ``threshold_bps`` (a negative number), arm
a cooldown that suppresses placements on that side. Phase 2K-style
favorable-exit predicate clears the cooldown early when the trailing
mean recovers past ``threshold × clear_band_mult``.

Why side-specific (not aggregate): the v1.4.157 snapshot's overall
markout was -1.78 bps mean — adverse but unremarkable. The SELL side
alone was -2.79 mean / -2.43 median while BUY was -0.74 / 0.00. An
aggregate gate masks the side-specific degradation. A per-side gate
catches it.

Why all fills (not just at_touch like ``at_touch_adverse_pause``):
this failure mode involves ``behind_touch`` SELL fills picked off
during a rally — the at-touch gate's filter would skip them.

Disabled by default (``threshold_bps == 0.0``).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

from app.enums import Side


class RealisedEdgeSideSuppressGate:
    """Per-side trailing realised-edge detector + suppression cooldown.

    Trigger: rolling N-fill mean of ``markout_5s_bps + rebate_bps``
    on that side drops below ``threshold_bps`` (which is itself
    negative). Mean (not median) because we WANT outliers to weigh
    heavily — a single -31 bps fill is exactly the signal we want
    the gate to react to.

    Favorable-exit (Phase 2K pattern): cooldown clears early when the
    same trailing mean rises above ``threshold_bps × clear_band_mult``
    (smaller mult → stronger recovery required) for
    ``favorable_exit_dwell_seconds``.
    """

    def __init__(
        self,
        *,
        threshold_bps: float,
        cooldown_seconds: float,
        min_fills: int,
        window_size: int = 20,
        favorable_exit_enabled: bool = True,
        clear_band_mult: float = 0.5,
        favorable_exit_dwell_seconds: float = 10.0,
    ) -> None:
        self._threshold_bps = float(threshold_bps)
        self._cooldown_seconds = float(cooldown_seconds)
        self._min_fills = int(min_fills)
        self._buy_edges: deque[float] = deque(maxlen=window_size)
        self._sell_edges: deque[float] = deque(maxlen=window_size)
        self._buy_suppressed_until_mono: float = 0.0
        self._sell_suppressed_until_mono: float = 0.0
        self._buy_fire_count: int = 0
        self._sell_fire_count: int = 0
        # Favorable-exit state (mirrors Phase 2K.6).
        self._favorable_exit_enabled = bool(favorable_exit_enabled)
        self._clear_band_mult = float(clear_band_mult)
        self._favorable_exit_dwell_seconds = float(
            favorable_exit_dwell_seconds
        )
        self._buy_favorable_dwell_started_mono: Optional[float] = None
        self._sell_favorable_dwell_started_mono: Optional[float] = None
        self._buy_was_active_last: bool = False
        self._sell_was_active_last: bool = False
        self._buy_cleared_via_favorable_total: int = 0
        self._sell_cleared_via_favorable_total: int = 0
        self._buy_cleared_via_ceiling_total: int = 0
        self._sell_cleared_via_ceiling_total: int = 0
        # v1.5.155 — position-aware favorable exit attribution.
        # Per CLAUDE.md Rule 0c: timer-based gates need a signal-
        # driven exit that considers current bot position. The
        # existing markout-based exit relies on NEW fills lifting the
        # trailing mean — but during suppression there ARE no fills
        # on the suppressed side, so the predicate never re-evaluates
        # to favorable. v1.5.154-260526-074029 snapshot showed 30
        # cooldowns / 0 favorable / 29 ceiling. The position-aware
        # exit clears the suppression when the SUPPRESSED side is
        # the REDUCING side for current inventory (bot needs that
        # side to unwind).
        self._buy_cleared_via_position_favorable_total: int = 0
        self._sell_cleared_via_position_favorable_total: int = 0
        # Diagnostic — the trailing mean at the moment of the most
        # recent trigger (for postmortem + dashboard).
        self._buy_last_trigger_mean_bps: float = 0.0
        self._sell_last_trigger_mean_bps: float = 0.0

    def enabled(self) -> bool:
        return (
            self._threshold_bps < 0.0
            and self._cooldown_seconds > 0.0
            and self._min_fills > 0
        )

    def note_fill(
        self,
        *,
        side: Side,
        markout_5s_bps: Optional[float],
        rebate_bps: float,
        now_mono: float,
    ) -> None:
        """Record a finalised fill's net realised edge
        (``markout_5s_bps + rebate_bps``) for ``side``.

        ``rebate_bps``: positive number = bot earned a rebate;
        negative = bot paid a fee. Caller computes this from the
        fill's fee/notional fields.

        Side effect: may arm the suppression cooldown when the
        trailing mean drops past ``threshold_bps``; may also clear an
        active cooldown via favorable-exit when the new sample lifts
        the mean above the clear band.
        """
        if not self.enabled():
            return
        if markout_5s_bps is None:
            return
        try:
            mk = float(markout_5s_bps)
            rb = float(rebate_bps)
        except (TypeError, ValueError):
            return
        edge = mk + rb
        if side == Side.BUY:
            self._buy_edges.append(edge)
            if self._check_trigger(self._buy_edges):
                self._buy_suppressed_until_mono = (
                    now_mono + self._cooldown_seconds
                )
                self._buy_fire_count += 1
                self._buy_last_trigger_mean_bps = self._current_mean(
                    Side.BUY
                ) or 0.0
                self._buy_favorable_dwell_started_mono = None
        elif side == Side.SELL:
            self._sell_edges.append(edge)
            if self._check_trigger(self._sell_edges):
                self._sell_suppressed_until_mono = (
                    now_mono + self._cooldown_seconds
                )
                self._sell_fire_count += 1
                self._sell_last_trigger_mean_bps = self._current_mean(
                    Side.SELL
                ) or 0.0
                self._sell_favorable_dwell_started_mono = None
        # Re-evaluate favorable-exit since a new sample changed the
        # trailing mean.
        self._maybe_clear_via_favorable_exit(side, now_mono)

    def _check_trigger(self, edges: deque[float]) -> bool:
        if len(edges) < self._min_fills:
            return False
        recent = list(edges)[-self._min_fills:]
        mean = sum(recent) / len(recent)
        return mean < self._threshold_bps

    def _current_mean(self, side: Side) -> Optional[float]:
        d = (
            self._buy_edges if side == Side.BUY
            else self._sell_edges if side == Side.SELL
            else None
        )
        if d is None or len(d) < self._min_fills:
            return None
        recent = list(d)[-self._min_fills:]
        return sum(recent) / len(recent)

    def _favorable_predicate_holds(self, side: Side) -> bool:
        if not self._favorable_exit_enabled:
            return False
        m = self._current_mean(side)
        if m is None:
            return False
        # threshold_bps is negative; clear_band = threshold * mult is
        # closer to zero (between threshold and 0). mult=0.5 means the
        # trailing mean must recover halfway back from the trigger.
        clear_band = self._threshold_bps * self._clear_band_mult
        return m > clear_band

    def _maybe_clear_via_favorable_exit(
        self, side: Side, now_mono: float
    ) -> bool:
        if not self._favorable_exit_enabled:
            return False
        until = (
            self._buy_suppressed_until_mono if side == Side.BUY
            else self._sell_suppressed_until_mono if side == Side.SELL
            else 0.0
        )
        if now_mono >= until:
            return False
        if self._favorable_predicate_holds(side):
            if side == Side.BUY:
                if self._buy_favorable_dwell_started_mono is None:
                    self._buy_favorable_dwell_started_mono = now_mono
                started = self._buy_favorable_dwell_started_mono
            else:
                if self._sell_favorable_dwell_started_mono is None:
                    self._sell_favorable_dwell_started_mono = now_mono
                started = self._sell_favorable_dwell_started_mono
            if (
                now_mono - started
                >= self._favorable_exit_dwell_seconds
            ):
                if side == Side.BUY:
                    self._buy_suppressed_until_mono = 0.0
                    self._buy_cleared_via_favorable_total += 1
                    self._buy_favorable_dwell_started_mono = None
                    self._buy_was_active_last = False
                else:
                    self._sell_suppressed_until_mono = 0.0
                    self._sell_cleared_via_favorable_total += 1
                    self._sell_favorable_dwell_started_mono = None
                    self._sell_was_active_last = False
                return True
        else:
            # Predicate not holding — reset dwell (re-flare).
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
        """v1.5.155 — position-aware favorable-exit predicate.

        Clears the suppression on ``side`` if that side is the
        REDUCING side for the current inventory (bot is meaningfully
        positioned in the OPPOSITE direction, so the suppressed side
        is the only way to unwind):

        * BUY suppression clears when ``position_qty <=
          -inventory_threshold`` (bot is SHORT; BUY reduces SHORT)
        * SELL suppression clears when ``position_qty >=
          +inventory_threshold`` (bot is LONG; SELL reduces LONG)

        The reducing side must always be available — pausing it
        traps the bot in adverse inventory. The adding side stays
        suppressed (the original adverse-realised-edge concern is
        about adding into a bleeding regime, which is exactly what
        the reducing-side-only carve-out preserves).

        Operator instruction CLAUDE.md Rule 0c. Idempotent: returns
        False without side-effects when ``side`` is not currently
        suppressed OR the position predicate doesn't hold.

        Returns True if this call cleared the suppression.
        """
        if not self.enabled():
            return False
        if side not in (Side.BUY, Side.SELL):
            return False
        until = (
            self._buy_suppressed_until_mono if side == Side.BUY
            else self._sell_suppressed_until_mono
        )
        if now_mono >= until:
            return False
        threshold = float(inventory_threshold)
        if side == Side.BUY:
            # Clear BUY suppression only when bot is meaningfully SHORT.
            if position_qty > -threshold:
                return False
            self._buy_suppressed_until_mono = 0.0
            self._buy_cleared_via_position_favorable_total += 1
            self._buy_favorable_dwell_started_mono = None
            self._buy_was_active_last = False
        else:
            # Clear SELL suppression only when bot is meaningfully LONG.
            if position_qty < threshold:
                return False
            self._sell_suppressed_until_mono = 0.0
            self._sell_cleared_via_position_favorable_total += 1
            self._sell_favorable_dwell_started_mono = None
            self._sell_was_active_last = False
        return True

    def try_clear_via_idle(
        self,
        *,
        now_mono: float,
        last_fill_mono: Optional[float],
        idle_clear_seconds: float,
    ) -> bool:
        """v1.5.197 — idle-decay exit predicate (both sides).

        Mirrors ``at_touch_adverse_pause.try_clear_via_idle`` and
        ``mae_gate.evaluate_idle_clear``. Eliminates the defensive-
        deadlock pattern where suppression of a side blocks the new
        fills whose markouts would naturally clear the suppression.

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
        if now_mono < self._buy_suppressed_until_mono:
            self._buy_suppressed_until_mono = 0.0
            self._buy_cleared_via_position_favorable_total += 1
            self._buy_favorable_dwell_started_mono = None
            self._buy_was_active_last = False
            cleared_any = True
        if now_mono < self._sell_suppressed_until_mono:
            self._sell_suppressed_until_mono = 0.0
            self._sell_cleared_via_position_favorable_total += 1
            self._sell_favorable_dwell_started_mono = None
            self._sell_was_active_last = False
            cleared_any = True
        return cleared_any

    def is_suppressed(self, side: Side, now_mono: float) -> bool:
        """Return True iff ``side`` is currently suppressed. Polled
        from the bot's hot path each tick. Evaluates favorable-exit
        (may clear early) + tracks the active→cleared edge for
        ceiling attribution."""
        if not self.enabled():
            return False
        if side not in (Side.BUY, Side.SELL):
            return False
        cleared_now = self._maybe_clear_via_favorable_exit(
            side, now_mono
        )
        until = (
            self._buy_suppressed_until_mono if side == Side.BUY
            else self._sell_suppressed_until_mono
        )
        currently = now_mono < until
        was_last = (
            self._buy_was_active_last if side == Side.BUY
            else self._sell_was_active_last
        )
        if was_last and not currently and not cleared_now:
            if side == Side.BUY:
                self._buy_cleared_via_ceiling_total += 1
            else:
                self._sell_cleared_via_ceiling_total += 1
        if side == Side.BUY:
            self._buy_was_active_last = currently
        else:
            self._sell_was_active_last = currently
        return currently

    def remaining_seconds(self, side: Side, now_mono: float) -> float:
        if not self.enabled():
            return 0.0
        if side == Side.BUY:
            return max(0.0, self._buy_suppressed_until_mono - now_mono)
        if side == Side.SELL:
            return max(0.0, self._sell_suppressed_until_mono - now_mono)
        return 0.0

    def snapshot_dict(self, now_mono: float) -> dict:
        return {
            "enabled": self.enabled(),
            "threshold_bps": self._threshold_bps,
            "min_fills": self._min_fills,
            "cooldown_seconds": self._cooldown_seconds,
            "favorable_exit_enabled": self._favorable_exit_enabled,
            "clear_band_mult": self._clear_band_mult,
            "favorable_exit_dwell_seconds": (
                self._favorable_exit_dwell_seconds
            ),
            "buy": {
                "recent_count": len(self._buy_edges),
                "recent_mean_bps": self._current_mean(Side.BUY),
                "suppressed": self.is_suppressed(Side.BUY, now_mono),
                "seconds_remaining": self.remaining_seconds(
                    Side.BUY, now_mono
                ),
                "fire_count": self._buy_fire_count,
                "last_trigger_mean_bps": self._buy_last_trigger_mean_bps,
                "favorable_dwell_active": (
                    self._buy_favorable_dwell_started_mono is not None
                ),
                "cleared_via_favorable_total": (
                    self._buy_cleared_via_favorable_total
                ),
                "cleared_via_ceiling_total": (
                    self._buy_cleared_via_ceiling_total
                ),
                "cleared_via_position_favorable_total": (
                    self._buy_cleared_via_position_favorable_total
                ),
            },
            "sell": {
                "recent_count": len(self._sell_edges),
                "recent_mean_bps": self._current_mean(Side.SELL),
                "suppressed": self.is_suppressed(Side.SELL, now_mono),
                "seconds_remaining": self.remaining_seconds(
                    Side.SELL, now_mono
                ),
                "fire_count": self._sell_fire_count,
                "last_trigger_mean_bps": self._sell_last_trigger_mean_bps,
                "favorable_dwell_active": (
                    self._sell_favorable_dwell_started_mono is not None
                ),
                "cleared_via_favorable_total": (
                    self._sell_cleared_via_favorable_total
                ),
                "cleared_via_ceiling_total": (
                    self._sell_cleared_via_ceiling_total
                ),
                "cleared_via_position_favorable_total": (
                    self._sell_cleared_via_position_favorable_total
                ),
            },
        }
