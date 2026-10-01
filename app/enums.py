from __future__ import annotations

from enum import Enum


class BotStatus(str, Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    RECOVERING_MARKET_DATA = "RECOVERING_MARKET_DATA"
    PAUSED = "PAUSED"
    # Aggressive market-IOC flatten (kill path / operator /flatten).
    FLATTENING = "FLATTENING"
    # Patient post-only-only flatten driven by the position-drawdown
    # gate. Distinct from FLATTENING because the bot stays alive and
    # resumes RUNNING once the position closes; no taker IOC is used.
    SOFT_FLATTENING = "SOFT_FLATTENING"
    # Service is shutting down (process exiting). Was previously
    # overloaded onto PAUSED, which made the dashboard's "Bot PAUSED"
    # toast fire on every deploy. Distinct enum value lets the
    # dashboard map it to "stopping" (amber, no toast). Set briefly
    # in main.py during graceful shutdown before the heartbeat
    # publisher's last write.
    SHUTTING_DOWN = "SHUTTING_DOWN"
    KILLED = "KILLED"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Lifecycle states of a ``WorkingOrder``.

    v1.4.68 wedge-elimination-cleanup Phase 0 — formal contract.

    Status graph (legal transitions):

        NEW_LOCAL ──> SENT ──> ACKED ─┬─> PARTIAL ──> FILLED
                                      │      ↑           (terminal)
                                      │      └─> ACKED (on cancel of remaining)
                                      │
                                      ├─> AMEND_PENDING ──> ACKED | PARTIAL
                                      │
                                      ├─> CANCEL_PENDING ──> CANCELED (terminal)
                                      │
                                      └─> REJECTED (terminal)

        Any non-terminal status ──> DESYNC (terminal-equivalent; given-up)

    Terminal partitions (used by reconciler + risk state machine):

        TERMINAL          = {CANCELED, FILLED, REJECTED, DESYNC}
        IN_FLIGHT         = {SENT, CANCEL_PENDING, AMEND_PENDING, NEW_LOCAL}
        LIVE_ON_VENUE     = {ACKED, PARTIAL}

    Mutation rule (Phase 0.6):

        Every ``WorkingOrder.status`` mutation MUST go through
        ``BotState.order_store.transition(wo, new_status, reason)``
        once Phase 3A lands. Direct ``wo.status = X`` assignment is
        a violation of the contract and will fail the index-consistency
        invariant test.

    Per-state semantics:

    * ``NEW_LOCAL`` — the bot has constructed the WO locally but has
      not yet sent the place request to the venue. Transient (sub-ms).

    * ``SENT`` — the bot has dispatched the place HTTP request. The
      venue has NOT yet confirmed (no place-response received, no WS
      ``live`` event processed). Time bound: must transition out
      within ``SENT_ORDER_UNRESOLVED_TIMEOUT_SECONDS`` or the side
      flips to ``side_unresolved`` and the bot pauses on that side.

      v1.4.67 lesson: a stale SENT WO is BLIND to the engine's hard-
      age cap. The cap only fires against ``ACKED`` / ``PARTIAL``.
      So if SENT lingers, the order ages on the wire un-capped. This
      is the v1.4.67-260518-195033 wedge mechanism. Phase 1A fixes
      the WS→WO race that lets SENT linger; Phase 1C.4 adds a wall-
      lifetime cap that fires even on SENT/PARTIAL.

    * ``ACKED`` — the place is confirmed: HTTP place-response is OK
      AND/OR WS ``live`` has arrived AND the WO has an
      ``order_id_exchange`` set. This is the "live on venue, no in-
      flight amend/cancel" steady state.

    * ``PARTIAL`` — same as ACKED but a partial fill has reduced the
      remaining size. Hard-age cap and reconciler treat the same as
      ACKED.

    * ``AMEND_PENDING`` — the bot has dispatched an amend request.
      See per-exit table in the original 1.4.15 docstring above.
      v1.4.70 (Phase 1C.5): a hard-age cancel preempts an in-flight
      amend by abandoning the amend and dispatching the cancel
      immediately.

    * ``CANCEL_PENDING`` — the bot has dispatched a cancel. Time
      bound: must reach a terminal state within
      ``CANCEL_PENDING_UNRESOLVED_TIMEOUT_SECONDS`` or the reaper
      (Phase 2B) force-transitions to ``CANCELED``.

      v1.4.66 lesson: stale CANCEL_PENDING orders blocked the risk
      state machine for 52 s (snapshot v1.4.66-260518-192744). The
      cancel-pending watchdog only inspected inside-rung slots, so
      orphan-slot and hydrated CANCEL_PENDING WOs lived forever.
      v1.4.67 partially fixed by making
      ``_local_has_cancellable_wos()`` skip stale CANCEL_PENDING;
      Phase 2B finishes the job by actually reaping them.

    * ``CANCELED`` (terminal) — confirmed cancelled via WS or REST.

    * ``FILLED`` (terminal) — fully filled via WS.

    * ``REJECTED`` (terminal) — venue rejected the place outright
      (e.g., 51604 post-only-cross at place time, transport rate
      limit, central pre-send risk check failure).

    * ``DESYNC`` (terminal-equivalent) — the bot detected an
      exchange↔local mismatch, issued recovery cancels for the
      orphan(s), and is no longer tracking this order. DESYNC is
      treated as TERMINAL everywhere downstream:

        - reconciler ``_TERMINAL`` set (v1.4.67)
        - ``_local_has_cancellable_wos()`` skip list (v1.4.67)
        - ``cancel_all_orders_for_symbol`` skip list (v1.4.67)

      Phase 2B's reaper removes DESYNC entries from the store
      entirely after ``DESYNC_REAP_TIMEOUT_SECONDS`` (default 30 s).
    """

    NEW_LOCAL = "NEW_LOCAL"
    SENT = "SENT"
    ACKED = "ACKED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    AMEND_PENDING = "AMEND_PENDING"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    DESYNC = "DESYNC"


class ActiveSides(str, Enum):
    BOTH = "BOTH"
    BID_ONLY = "BID_ONLY"
    ASK_ONLY = "ASK_ONLY"
    NONE = "NONE"


class EventSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class RiskAction(str, Enum):
    ALLOW = "ALLOW"
    BID_ONLY = "BID_ONLY"
    ASK_ONLY = "ASK_ONLY"
    NO_QUOTE = "NO_QUOTE"
    CANCEL_ALL = "CANCEL_ALL"
    FLATTEN = "FLATTEN"
    # Soft-flatten request — patient post-only exit via the
    # SF worker (no taker, unless phase-3 fallback fires). Used by
    # the toxicity-trigger path so it can request aggressive
    # post-only exit without resorting to the taker market_close
    # path that ``FLATTEN`` invokes.
    SOFT_FLATTEN = "SOFT_FLATTEN"
    KILL = "KILL"


class RiskExecState(str, Enum):
    """v1.4.56 wedge-elimination Phase 2 — risk-action execution state machine.

    The bot's risk decision (``RiskAction``) is recomputed every tick.
    Pre-v1.4.56 each tick that landed on a {KILL, CANCEL_ALL, FLATTEN}
    risk action re-invoked ``cancel_resting_for_risk`` → re-iterated
    cancellation. Phase 1 made each individual cancel idempotent
    (CANCEL_PENDING tombstone), but the per-tick LOOP still ran
    ~100×/sec on a BBO-driven cycle — wasted CPU + log noise.

    This state machine transforms the level-triggered re-invocation
    into edge-triggered transitions:

      NORMAL ─risk_decides_{KILL,CANCEL_ALL,FLATTEN}─> CANCELLING
                                                          │
                                                          │ all WOs reach
                                                          │ terminal status
                                                          ▼
        ┌─────risk_returns_to_quotable─── SUPPRESSED
        ▼
      NORMAL

      (any state) ─risk_decides_{KILL,CANCEL_ALL,FLATTEN}─> CANCELLING
                                                       (re-arms; safe)

    ``CANCELLING``: cancel-all was dispatched on entry; ticks during
    this state return True without firing any cancels.
    ``SUPPRESSED``: cancels confirmed via the dispatcher's CANCEL_PENDING
    → terminal transition; ticks return True (still no quoting) but
    no work happens. The bot is at idle, waiting for risk to ALLOW.
    ``NORMAL``: cancel_resting_for_risk is a no-op; the normal
    ``_orchestrate`` path drives quoting.

    The machine is fault-tolerant: any tick that lands on a cancelling
    risk action re-enters CANCELLING and re-fires cancel-all — that's
    fine because Phase 1's cancel-all is idempotent against in-flight /
    terminal WOs.
    """

    NORMAL = "NORMAL"
    CANCELLING = "CANCELLING"
    SUPPRESSED = "SUPPRESSED"


class DesyncPhase(str, Enum):
    """Explicit order-book / local vs exchange reconciliation state."""

    OK = "OK"
    DETECTED = "DETECTED"
    RECONCILING = "RECONCILING"
    RECOVERED = "RECOVERED"
    UNRECOVERABLE = "UNRECOVERABLE"


class FlattenResult(str, Enum):
    """Outcome of a flatten attempt (for logging and control flow)."""

    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    SKIPPED_ALREADY_FLAT = "SKIPPED_ALREADY_FLAT"


class QuoteEligibility(str, Enum):
    """
    Pre-inventory cap on which sides may quote (applied after hard safety checks).

    Narrower than :class:`ActiveSides` naming: BUY/SELL only refer to which passive quotes are allowed.
    """

    QUOTE_BOTH = "QUOTE_BOTH"
    QUOTE_BUY_ONLY = "QUOTE_BUY_ONLY"
    QUOTE_SELL_ONLY = "QUOTE_SELL_ONLY"
    HOLD_ALL = "HOLD_ALL"


class TouchPlacementMode(str, Enum):
    """
    How aggressively touch-distance / band logic may re-anchor passive quotes.

    normal_two_sided_mm: wide passive quotes are execution-safe only (tick, post-only, floor);
    quote_max_distance_to_touch_ticks is not a hard placement constraint.
    Other modes keep the legacy near-touch band + symmetric rescue behavior.
    """

    NORMAL_TWO_SIDED_MM = "normal_two_sided_mm"
    ONE_SIDED_INVENTORY_REDUCTION = "one_sided_inventory_reduction"
    DEGRADED_FALLBACK = "degraded_fallback"
    EMERGENCY = "emergency"
