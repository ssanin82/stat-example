"""Adaptive join-depth controller.

Closes the loop between observable "queue moved away mid-place" and
"queue moved through us right after fill" signals and the bot's
quoting depth. Produces a single scalar overlay (bps) added to
``base_half_spread_bps`` in ``compute_quote_decision``.

Off by default — opt-in per profile via ``JOIN_DEPTH_AUTOTUNE_ENABLED``.

Design + rollout: ``plans/auto-tune.md``.

Inputs (sampled every ``JOIN_DEPTH_AUTOTUNE_UPDATE_SECONDS``):
    1. Cross-rejection rate (rejects/min) — derived from the
       ``QuoteQualityTelemetry`` session counter delta.
    2. Median 1s adverse markout (bps, signed; only the negative
       part contributes).
    3. Median 5s adverse markout (bps, signed; only the negative
       part contributes).
    4. Fill rate (trades/min) — when below target, pulls overlay
       NEGATIVE so the bot quotes tighter to compete.

Output:
    ``current_overlay_bps`` — float in
    ``[overlay_min_bps, overlay_max_bps]``.

Dynamics:
    target = α_reject·rejects_per_min
           + α_markout_1s·max(0, -median_1s_markout_bps)
           + α_markout_5s·max(0, -median_5s_markout_bps)
           - α_underfill·max(0, target_fills_per_min - trades_per_min)

    overlay ← ewma_alpha·target + (1-ewma_alpha)·overlay
    overlay ← clamp(overlay, min_bps, max_bps)

The overlay composes with the existing ``MIN_HALF_SPREAD_BPS`` /
``MAX_HALF_SPREAD_BPS`` clamps in ``compute_quote_decision``, so a
bad controller cannot push the bot outside the configured envelope.

Episode reset:
    ``reset()`` is called on kill / soft-flatten / manual pause so
    we don't carry stale overlay across regime changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from app.config import Settings

from app import clock as _clock

if TYPE_CHECKING:
    from app.state import BotState

logger = logging.getLogger(__name__)


def _median(vals: list[float]) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return (s[m - 1] + s[m]) / 2.0


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@dataclass
class JoinDepthSnapshot:
    """Last-tick decomposition for telemetry / dashboard.

    All fields are snapshot at the most recent ``tick`` call. None
    means the controller has never run (e.g. autotune disabled, or
    bot just started)."""

    enabled: bool
    overlay_bps: float
    target_overlay_bps: Optional[float]
    rejects_per_min: Optional[float]
    median_1s_markout_bps: Optional[float]
    median_5s_markout_bps: Optional[float]
    trades_per_min: Optional[float]
    contrib_reject_bps: float
    contrib_markout_1s_bps: float
    contrib_markout_5s_bps: float
    contrib_underfill_bps: float
    saturated_seconds: float
    last_update_iso: Optional[str]


class JoinDepthController:
    """See module docstring."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._overlay_bps: float = 0.0
        self._last_update_mono: Optional[float] = None
        self._last_update_iso: Optional[str] = None
        self._last_reject_count: int = 0
        self._saturation_started_at_mono: Optional[float] = None
        self._saturation_warned: bool = False
        self._last_snapshot: JoinDepthSnapshot = JoinDepthSnapshot(
            enabled=bool(settings.join_depth_autotune_enabled),
            overlay_bps=0.0,
            target_overlay_bps=None,
            rejects_per_min=None,
            median_1s_markout_bps=None,
            median_5s_markout_bps=None,
            trades_per_min=None,
            contrib_reject_bps=0.0,
            contrib_markout_1s_bps=0.0,
            contrib_markout_5s_bps=0.0,
            contrib_underfill_bps=0.0,
            saturated_seconds=0.0,
            last_update_iso=None,
        )

    # ------------------------------------------------------------------
    # Public read API — quote_engine + dashboard
    # ------------------------------------------------------------------

    def current_overlay_bps(self) -> float:
        """Bps to add to ``base_half_spread_bps``. Always 0 when
        autotune is disabled."""
        if not self._settings.join_depth_autotune_enabled:
            return 0.0
        return self._overlay_bps

    def snapshot(self) -> JoinDepthSnapshot:
        return self._last_snapshot

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Drop overlay back to 0. Called on kill / soft-flatten /
        manual pause so we don't carry stale tuning across regime
        changes."""
        self._overlay_bps = 0.0
        self._saturation_started_at_mono = None
        self._saturation_warned = False
        self._last_snapshot = self._snapshot_with(
            target_overlay_bps=None,
            rejects_per_min=None,
            median_1s_markout_bps=None,
            median_5s_markout_bps=None,
            trades_per_min=None,
            contribs=(0.0, 0.0, 0.0, 0.0),
        )

    # ------------------------------------------------------------------
    # Tick — sample inputs, recompute overlay
    # ------------------------------------------------------------------

    def tick(self, state: "BotState", now_mono: Optional[float] = None) -> None:
        """Update the overlay if the configured interval has elapsed.

        Caller invokes every quote-loop tick; this method is a
        cheap no-op until the update window has passed. Reading
        state is lockless — all the fields we touch are scalar
        counters or deque snapshots that can race harmlessly."""
        if not self._settings.join_depth_autotune_enabled:
            return
        if now_mono is None:
            now_mono = _clock.monotonic()
        update_interval = float(
            self._settings.join_depth_autotune_update_seconds
        )
        if (
            self._last_update_mono is not None
            and now_mono - self._last_update_mono < update_interval
        ):
            return

        # Cross-rejection rate. The session counter is monotonic; a
        # delta over the interval gives rate. First call uses the
        # configured update interval as a denominator (no prior
        # sample), so the first observation is accurate but slow.
        try:
            cur_reject_count = int(
                state.quote_quality.post_only_cross_rejection_count
            )
        except Exception:
            cur_reject_count = self._last_reject_count
        if self._last_update_mono is None:
            elapsed_s = update_interval
        else:
            elapsed_s = max(now_mono - self._last_update_mono, 0.001)
        rejects_per_min = (
            (cur_reject_count - self._last_reject_count) / elapsed_s * 60.0
            if cur_reject_count >= self._last_reject_count
            else 0.0
        )

        # Markouts. Use the last 50 fills (matches the
        # ``delayed_markout_summary`` window used for dashboard
        # display) — large enough to be statistically meaningful,
        # small enough to react within minutes.
        try:
            fills = list(state.recent_fills)[:50]
        except Exception:
            fills = []
        m1s_vals = [
            float(f.markout_1s_bps)
            for f in fills
            if f.markout_1s_bps is not None
        ]
        m5s_vals = [
            float(f.markout_5s_bps)
            for f in fills
            if f.markout_5s_bps is not None
        ]
        median_1s = _median(m1s_vals)
        median_5s = _median(m5s_vals)

        # Fill rate.
        try:
            trades_per_min = float(state.trades_last_minute())
        except Exception:
            trades_per_min = 0.0

        # Per-input contributions.
        alpha_reject = float(self._settings.join_depth_autotune_alpha_reject)
        alpha_m1 = float(self._settings.join_depth_autotune_alpha_markout_1s)
        alpha_m5 = float(self._settings.join_depth_autotune_alpha_markout_5s)
        alpha_uf = float(self._settings.join_depth_autotune_alpha_underfill)
        target_fills = float(
            self._settings.join_depth_autotune_target_fills_per_min
        )

        contrib_reject = alpha_reject * max(0.0, rejects_per_min)
        contrib_m1 = (
            alpha_m1 * max(0.0, -median_1s) if median_1s is not None else 0.0
        )
        contrib_m5 = (
            alpha_m5 * max(0.0, -median_5s) if median_5s is not None else 0.0
        )
        contrib_underfill = -alpha_uf * max(0.0, target_fills - trades_per_min)

        target_overlay = (
            contrib_reject + contrib_m1 + contrib_m5 + contrib_underfill
        )

        # EWMA blend toward the new target.
        ewma_alpha = float(self._settings.join_depth_autotune_ewma_alpha)
        new_overlay = (
            ewma_alpha * target_overlay
            + (1.0 - ewma_alpha) * self._overlay_bps
        )
        new_overlay = _clamp(
            new_overlay,
            float(self._settings.join_depth_autotune_overlay_min_bps),
            float(self._settings.join_depth_autotune_overlay_max_bps),
        )
        self._overlay_bps = new_overlay

        # Saturation guard.
        max_bps = float(self._settings.join_depth_autotune_overlay_max_bps)
        is_saturated = self._overlay_bps + 1e-9 >= max_bps
        if is_saturated:
            if self._saturation_started_at_mono is None:
                self._saturation_started_at_mono = now_mono
            elif (
                not self._saturation_warned
                and now_mono - self._saturation_started_at_mono
                >= float(
                    self._settings.join_depth_autotune_saturation_warn_seconds
                )
            ):
                self._saturation_warned = True
                logger.warning(
                    "join_depth_autotune_saturated overlay_bps=%.3f "
                    "for_seconds=%.0f rejects_per_min=%.2f median_1s_markout_bps=%s "
                    "median_5s_markout_bps=%s trades_per_min=%.2f -- regime may have "
                    "shifted, consider re-tuning static config",
                    self._overlay_bps,
                    now_mono - self._saturation_started_at_mono,
                    rejects_per_min,
                    median_1s,
                    median_5s,
                    trades_per_min,
                )
        else:
            self._saturation_started_at_mono = None
            self._saturation_warned = False

        # Persist for next tick + telemetry.
        self._last_update_mono = now_mono
        self._last_update_iso = _utc_now_iso()
        self._last_reject_count = cur_reject_count
        self._last_snapshot = self._snapshot_with(
            target_overlay_bps=target_overlay,
            rejects_per_min=rejects_per_min,
            median_1s_markout_bps=median_1s,
            median_5s_markout_bps=median_5s,
            trades_per_min=trades_per_min,
            contribs=(
                contrib_reject,
                contrib_m1,
                contrib_m5,
                contrib_underfill,
            ),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _snapshot_with(
        self,
        *,
        target_overlay_bps: Optional[float],
        rejects_per_min: Optional[float],
        median_1s_markout_bps: Optional[float],
        median_5s_markout_bps: Optional[float],
        trades_per_min: Optional[float],
        contribs: tuple[float, float, float, float],
    ) -> JoinDepthSnapshot:
        if (
            self._saturation_started_at_mono is not None
            and self._last_update_mono is not None
        ):
            sat_s = max(
                0.0, self._last_update_mono - self._saturation_started_at_mono
            )
        else:
            sat_s = 0.0
        return JoinDepthSnapshot(
            enabled=bool(self._settings.join_depth_autotune_enabled),
            overlay_bps=self._overlay_bps,
            target_overlay_bps=target_overlay_bps,
            rejects_per_min=rejects_per_min,
            median_1s_markout_bps=median_1s_markout_bps,
            median_5s_markout_bps=median_5s_markout_bps,
            trades_per_min=trades_per_min,
            contrib_reject_bps=contribs[0],
            contrib_markout_1s_bps=contribs[1],
            contrib_markout_5s_bps=contribs[2],
            contrib_underfill_bps=contribs[3],
            saturated_seconds=sat_s,
            last_update_iso=self._last_update_iso,
        )


def _utc_now_iso() -> str:
    """Local helper to avoid importing utc_now from another module
    (keeps the controller free of circular import concerns)."""
    from datetime import datetime, timezone

    return _clock.now_utc().isoformat()
