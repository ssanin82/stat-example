"""v1.4.82 wedge-elimination-cleanup Phase 3B — TelemetryStore.

Owns the bot's observability state:

* Cumulative counters (places / cancels / fills / rejections / reaper)
* Ring buffers (recent events, orchestrate decisions, quote refresh
  skip history)
* Per-side decision counters

The current implementation provides a read facade over the
``executor_state`` snapshot fields that already live on
``BotState``. Phase 3D will move ownership of these fields
into the store proper; until then this surface is the recommended
read path for new code (postmortem, dashboard, integration tests).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.state import BotState


class TelemetryStore:
    """Thin facade over ``BotState.executor_state_snapshot`` and the
    cumulative counters maintained on ``OrderManager``. Read-only.

    Phase 3B intent: surface a stable, focused API for telemetry
    consumers so they don't need to read the executor_state dict
    directly. Phase 4 / Phase 6 / Phase 7 lean on this facade.
    """

    def __init__(self, state: "BotState") -> None:
        self._state = state

    @property
    def executor_state(self) -> dict[str, Any]:
        """The full executor-state snapshot dict. Same object as
        ``state.executor_state_snapshot``."""
        return getattr(self._state, "executor_state_snapshot", {}) or {}

    # Phase 1+2+3 health counters — most operator-facing.

    def ws_unmatched_total(self) -> int:
        return int(self.executor_state.get("ws_event_unmatched_to_local_wo_total", 0))

    def hydration_merged_total(self) -> int:
        return int(self.executor_state.get("hydration_merged_existing_total", 0))

    def reaper_total(self) -> int:
        es = self.executor_state
        return (
            int(es.get("reaper_cancel_pending_reaped_total", 0))
            + int(es.get("reaper_desync_removed_total", 0))
            + int(es.get("reaper_sent_rejected_total", 0))
        )

    def phase2a_invariant_violations(self) -> int:
        return int(self.executor_state.get("gate_phase2a_invariant_violation_total", 0))

    # v1.4.92 Phase 4A cutover — ``qbr_unconsumed_total()`` REMOVED
    # along with the Phase 2D runtime audit. The typed ``BuildCommand``
    # sum-type makes the v1.4.66 regression class structurally
    # impossible; no runtime counter is needed.

    def risk_exec_state(self) -> str:
        return str(self.executor_state.get("risk_exec_state", "UNKNOWN"))

    def side_unresolved_enter_count(self) -> int:
        return int(self.executor_state.get("side_unresolved_enter_count", 0))

    def execution_idle_s(self) -> float | None:
        v = self.executor_state.get("execution_idle_s")
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def is_healthy(self) -> bool:
        """Aggregate health check.

        Returns False when any of the wedge-class counters are
        non-zero, the risk state is CANCELLING/SUPPRESSED, or
        Phase 2A invariant violations have fired.
        """
        return (
            self.ws_unmatched_total() == 0
            and self.reaper_total() == 0
            and self.phase2a_invariant_violations() == 0
            and self.risk_exec_state() == "NORMAL"
        )
