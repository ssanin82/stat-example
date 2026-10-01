"""
Delayed markouts: resolve 1s / 3s / 5s / 15s / 30s / 60s / 120s horizons
using mid observed on later bot ticks.

Horizons are measured from the exchange fill time; each value is the
first mid sampled at or after that delay (discrete loop timing, not
sub-tick interpolation).

v1.4.98 — added the 15s / 30s / 60s / 120s horizons. The 1s / 3s / 5s
endpoints (HFT-immediate-pickoff range) remain canonical for control
surfaces (sizing scaler, quote aging, session-drawdown gate, at-touch
adverse pause). The longer horizons are DIAGNOSTIC ONLY: they let the
postmortem / dashboard decompose "residual" (PnL outside the 5s window)
into time-resolved slices aligned with the bot's own gates
(POSITION_DRAWDOWN_GATE_DURATION_SECONDS=30, POST_SWING_WINDOW_SECONDS=60).

The pending-markout-jobs deque maxlen on ``BotState`` is sized for the
longest horizon × peak fill rate; see ``app/state.py`` for the
calculation and the silent-drop counter that signals overflow.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from app.enums import Side
from app.models import Fill

from app import clock as _clock

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.state import BotState
    from app.storage import Storage


def _aware(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts


def delayed_markout_bps(side: Side, fill_price: float, future_mid: float) -> float:
    """Signed bps vs fill price; negative = adverse (price moved against the fill)."""
    if (
        not isinstance(fill_price, (int, float))
        or not isinstance(future_mid, (int, float))
        or not math.isfinite(float(fill_price))
        or not math.isfinite(float(future_mid))
    ):
        return 0.0
    fill_price = float(fill_price)
    future_mid = float(future_mid)
    if fill_price <= 0 or future_mid <= 0:
        return 0.0
    if side == Side.BUY:
        out = (future_mid - fill_price) / fill_price * 10_000.0
    else:
        out = (fill_price - future_mid) / fill_price * 10_000.0
    return out if math.isfinite(out) else 0.0


@dataclass
class PendingMarkoutJob:
    fill_id: str
    side: Side
    fill_price: float
    t_fill: datetime
    # BUG-012: hold a reference to the source Fill so resolution does NOT
    # depend on the fill still being inside ``state.recent_fills`` (deque
    # capped at 1000; bursty regimes can still age fills out before all
    # horizons resolve, silently dropping toxicity-bearing markouts).
    # The reference is the same dataclass instance that's also in
    # ``recent_fills`` — mutating ``markout_*_bps`` here updates the
    # shared object.
    fill: Optional[Fill] = None


def register_fill_for_delayed_markouts(state: "BotState", f: Fill) -> None:
    with state._lock:
        # v1.4.36 (Codex #4): detect deque overflow before append.
        # ``state.pending_markout_jobs`` has ``maxlen=400``; appending
        # to a full deque silently evicts the oldest — pre-v1.4.36 the
        # eviction was invisible and could corrupt regime analytics
        # during bursty fill windows (>400 fills inside the 5 s
        # horizon). Bump the counter when we're about to overflow so
        # the operator can see it in ``state_current`` and on the
        # heartbeat-derived dashboards.
        maxlen = state.pending_markout_jobs.maxlen or 400
        if len(state.pending_markout_jobs) >= maxlen:
            state.markout_jobs_dropped_overflow_count += 1
        state.pending_markout_jobs.append(
            PendingMarkoutJob(
                fill_id=f.fill_id,
                side=f.side,
                fill_price=f.price,
                t_fill=f.ts_fill,
                fill=f,
            )
        )


def _find_fill(fills: deque[Fill], fill_id: str) -> Optional[Fill]:
    for x in fills:
        if x.fill_id == fill_id:
            return x
    return None


def process_pending_markouts(
    state: "BotState",
    storage: Optional["Storage"],
    now: datetime,
    mid: Optional[float],
    *,
    max_jobs: int = 0,
) -> None:
    """Resolve any due horizons; persists to SQLite when storage is set."""
    if (
        storage is None
        or mid is None
        or not isinstance(mid, (int, float))
        or not math.isfinite(float(mid))
        or float(mid) <= 0
    ):
        return

    # Fast path: avoid lock/scanning on common no-pending-jobs ticks.
    if not state.pending_markout_jobs:
        return

    now_a = _aware(now)
    keep: list[PendingMarkoutJob] = []
    # v1.4.98 — update tuples now carry 7 horizon slots:
    #   (fill_id, m1, m3, m5, m15, m30, m60, m120)
    # legacy 4-tuple shape preserved by the storage-layer fallback in
    # update_fill_markouts_many when only 1s/3s/5s are present.
    updates: list[
        tuple[
            str,
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
            Optional[float],
        ]
    ] = []
    cap = max_jobs if max_jobs > 0 else 10**9

    with state._lock:
        if not state.pending_markout_jobs:
            return
        maxlen = state.pending_markout_jobs.maxlen or 8000
        for idx, job in enumerate(list(state.pending_markout_jobs)):
            if idx >= cap:
                keep.append(job)
                continue
            # BUG-012: prefer the job-held fill reference (decoupled from
            # ``recent_fills`` rotation). Fall back to the deque scan for
            # legacy jobs that pre-date the reference field.
            f = job.fill or _find_fill(state.recent_fills, job.fill_id)
            if f is None:
                state.markout_jobs_orphaned_count = (
                    getattr(state, "markout_jobs_orphaned_count", 0) + 1
                )
                continue
            age = (now_a - _aware(job.t_fill)).total_seconds()
            u1 = u3 = u5 = u15 = u30 = u60 = u120 = None
            if age >= 1.0 and f.markout_1s_bps is None:
                f.markout_1s_bps = delayed_markout_bps(f.side, f.price, mid)
                u1 = f.markout_1s_bps
            if age >= 3.0 and f.markout_3s_bps is None:
                f.markout_3s_bps = delayed_markout_bps(f.side, f.price, mid)
                u3 = f.markout_3s_bps
            if age >= 5.0 and f.markout_5s_bps is None:
                f.markout_5s_bps = delayed_markout_bps(f.side, f.price, mid)
                u5 = f.markout_5s_bps
                # Tiered session-drawdown gate (1.2.1): when the gate
                # is in RESUME_TESTING, every freshly-finalized 5s
                # markout feeds the sample buffer. The gate's
                # observe_test_resume_fill decides "clear back" vs
                # "escalate to next tier" once the buffer fills.
                # No-op when not in RESUME_TESTING.
                try:
                    from app import session_drawdown_gate as _sdg
                    sdg = state.session_drawdown
                    if sdg.tier == _sdg.DrawdownTier.RESUME_TESTING:
                        thresholds = _sdg.TierThresholds(
                            tier1_widen_usd=float(state._settings.session_drawdown_tier1_widen_usd),
                            tier2_pause_short_usd=float(state._settings.session_drawdown_tier2_pause_short_usd),
                            tier3_pause_long_usd=float(state._settings.session_drawdown_tier3_pause_long_usd),
                            tier4_kill_usd=float(state._settings.session_drawdown_tier4_kill_usd),
                            pause_short_seconds=float(state._settings.session_drawdown_pause_short_seconds),
                            pause_long_seconds=float(state._settings.session_drawdown_pause_long_seconds),
                            test_resume_sample_fills=int(state._settings.session_drawdown_test_resume_sample_fills),
                            test_resume_max_adverse_bps=float(state._settings.session_drawdown_test_resume_max_adverse_bps),
                        )
                        _sdg.observe_test_resume_fill(
                            sdg,
                            fill_markout_5s_bps=float(f.markout_5s_bps),
                            now_mono=_clock.monotonic(),
                            now_iso=_clock.now_utc().isoformat(),
                            session_pnl_usd=float(state.pnl.total_pnl_usd or 0.0),
                            thresholds=thresholds,
                        )
                except Exception:
                    # Gate failures must never break markout
                    # finalisation — the markout / storage update
                    # path is more critical than the gate.
                    pass
                # 2026-05-12 codex-#1 narrow: observe the resolved 5s
                # markout for the at_touch_adverse_pause gate. No-op
                # when the gate is disabled or the fill isn't at_touch.
                try:
                    gate = getattr(state, "at_touch_adverse_pause", None)
                    if gate is not None:
                        gate.observe_resolved_5s_markout(f, _clock.monotonic())
                except Exception:
                    pass
                # v1.4.161 Phase 4C.3 mini — observe the resolved 5 s
                # net edge (markout + rebate) for the per-side
                # realised-edge suppression gate. All fills count
                # (not just at_touch). Rebate computed from fill's
                # fee + notional fields.
                try:
                    gate = getattr(
                        state, "realised_edge_side_suppress", None
                    )
                    if gate is not None:
                        # Convention: ``f.fee`` is in account ccy
                        # (USD). Maker rebate has negative ``fee``;
                        # rebate_bps is positive.
                        notional = float(getattr(f, "notional", 0.0))
                        fee = float(getattr(f, "fee", 0.0))
                        rebate_bps = (
                            (-fee / notional) * 1e4
                            if notional > 0
                            else 0.0
                        )
                        gate.note_fill(
                            side=f.side,
                            markout_5s_bps=f.markout_5s_bps,
                            rebate_bps=rebate_bps,
                            now_mono=_clock.monotonic(),
                        )
                        # v1.5.41 Phase 4C.3 — long-window per-side
                        # trailing-edge history. Same input shape as
                        # the short-window gate above; different
                        # consumer (expected_edge multiplier vs
                        # binary cooldown). Feeding both from the
                        # SAME computed rebate_bps keeps the two
                        # gates' inputs identical.
                        seh = getattr(state, "side_edge_history", None)
                        if seh is not None:
                            seh.note_fill(
                                side=f.side,
                                markout_5s_bps=f.markout_5s_bps,
                                rebate_bps=rebate_bps,
                                now_mono=_clock.monotonic(),
                            )
                except Exception:
                    pass
                # v1.5.26 Phase 2D -- residual-decay tracker.
                # Observes resolved-5s fills' net residual edge
                # (closed_pnl + rebate - markout_5s_dollars,
                # normalised to bps of notional). The bot's
                # adaptive_widen arming block polls
                # ``tracker.is_armed(now_mono)`` once per tick and
                # fires the gate with ``reason=residual_decay`` when
                # the rolling mean stays below threshold for the
                # configured dwell.
                try:
                    rdt = getattr(state, "residual_decay_tracker", None)
                    if rdt is not None:
                        from app.residual_decay_tracker import (
                            compute_residual_bps,
                        )
                        residual_bps = compute_residual_bps(
                            closed_pnl=getattr(f, "closed_pnl", None),
                            fee=float(getattr(f, "fee", 0.0)),
                            markout_5s_bps=f.markout_5s_bps,
                            notional=float(getattr(f, "notional", 0.0)),
                        )
                        rdt.note_fill(
                            residual_bps=residual_bps,
                            now_mono=_clock.monotonic(),
                        )
                except Exception:
                    pass
            # v1.4.98 — extended diagnostic horizons.
            # None of these participate in gate decisions; they exist
            # so the postmortem can decompose "residual" PnL across
            # the bot's actual holding period.
            if age >= 15.0 and f.markout_15s_bps is None:
                f.markout_15s_bps = delayed_markout_bps(f.side, f.price, mid)
                u15 = f.markout_15s_bps
            if age >= 30.0 and f.markout_30s_bps is None:
                f.markout_30s_bps = delayed_markout_bps(f.side, f.price, mid)
                u30 = f.markout_30s_bps
            if age >= 60.0 and f.markout_60s_bps is None:
                f.markout_60s_bps = delayed_markout_bps(f.side, f.price, mid)
                u60 = f.markout_60s_bps
            if age >= 120.0 and f.markout_120s_bps is None:
                f.markout_120s_bps = delayed_markout_bps(f.side, f.price, mid)
                u120 = f.markout_120s_bps
            if (
                u1 is not None
                or u3 is not None
                or u5 is not None
                or u15 is not None
                or u30 is not None
                or u60 is not None
                or u120 is not None
            ):
                updates.append((f.fill_id, u1, u3, u5, u15, u30, u60, u120))
            # Keep the job alive until EVERY horizon resolves. The
            # 120s tail dominates retention; budget the deque maxlen
            # in ``app/state.py`` accordingly.
            if (
                f.markout_1s_bps is None
                or f.markout_3s_bps is None
                or f.markout_5s_bps is None
                or f.markout_15s_bps is None
                or f.markout_30s_bps is None
                or f.markout_60s_bps is None
                or f.markout_120s_bps is None
            ):
                keep.append(job)

        state.pending_markout_jobs = deque(keep, maxlen=maxlen)

    # Keep storage I/O outside state lock.
    #
    # v1.4.37 (Codex #6): batch the markout finalisations into one
    # storage call → one SQLite transaction. Pre-v1.4.37 ran one
    # commit per fill which under bursty fill windows caused write
    # amplification + lock contention against fill ingestion + the
    # live-stats / heartbeat readers. The new
    # ``update_fill_markouts_many`` runs all N UPDATEs inside a
    # single ``self.connection()`` context (single transaction)
    # under one ``self._lock`` acquisition. Falls back to the legacy
    # per-fill method when the batch helper isn't present (defensive
    # — covers test doubles like ``_TrackingStorage`` that only
    # define ``update_fill_markouts``).
    if updates:
        batch = getattr(storage, "update_fill_markouts_many", None)
        if callable(batch):
            try:
                batch(updates)
            except Exception:
                # Fall back to the per-fill path on any batch failure
                # so a single mis-shaped row doesn't drop every other
                # finalisation in the same tick.
                logger.exception(
                    "update_fill_markouts_many_failed_falling_back_to_singles"
                )
                for fill_id, u1, u3, u5, u15, u30, u60, u120 in updates:
                    storage.update_fill_markouts(
                        fill_id, u1, u3, u5, u15, u30, u60, u120
                    )
        else:
            for fill_id, u1, u3, u5, u15, u30, u60, u120 in updates:
                storage.update_fill_markouts(
                    fill_id, u1, u3, u5, u15, u30, u60, u120
                )
