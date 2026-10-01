from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from app.enums import ActiveSides, EventSeverity, OrderStatus, RiskAction, Side
from app.inbound_timing import InboundPublicTiming
from app.utils.time import utc_now

from app import clock as _clock


@dataclass
class BestBidAsk:
    symbol: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid_price: Optional[float]
    spread_bps: Optional[float]
    ts_exchange_ms: Optional[int] = None
    ts_local: datetime = field(default_factory=utc_now)
    inbound_public_timing: Optional[InboundPublicTiming] = None
    # Top-of-book depth for microprice-based reservation
    # (``microprice = (bid_size*best_ask + ask_size*best_bid) / (bid_size+ask_size)``).
    # Optional because older public-WS payloads / tests may omit them; the
    # reservation path falls back to midprice when either is missing.
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None


@dataclass(frozen=True, slots=True)
class TradePrint:
    """A single public trade print from the venue's trade-stream feed.

    Used by the Priority #3 flow-direction score (``app/flow_score.py``).
    Field names mirror GRVT's ``ws_trade_feed_data_v1`` payload shape so
    the parser is a thin projection — ``is_taker_buyer``, ``event_time``,
    ``trade_id``, ``price``, ``size``. Extra fields (mark_price,
    index_price, etc.) are discarded by the parser; they're not needed
    for flow-direction analysis and we want the bounded-memory deque
    on BotState to stay small.

    ``ts_exchange_ms`` is derived from the venue's ``event_time``
    (typically ns precision on GRVT — divided by 1e6 on parse).
    ``ts_local_ms`` is the local receipt wall time (``_clock.time_seconds()*1000``)
    so the gap between the two is useful for cross-stream latency
    diagnostics.

    ``aggressor_side`` is ``Side.BUY`` when the taker was a buyer (the
    print lifted an ask), ``Side.SELL`` when the taker was a seller (the
    print hit a bid). This is the directional signal that feeds into
    trade-flow-imbalance (TFI).
    """

    ts_exchange_ms: int
    ts_local_ms: int
    price: float
    size: float
    aggressor_side: "Side"
    trade_id: str


@dataclass
class AccountSnapshot:
    equity_usd: Optional[float]
    cash_usd: Optional[float]
    withdrawable_usd: Optional[float]
    ts_local: datetime = field(default_factory=utc_now)


@dataclass
class PositionSnapshot:
    symbol: str
    position_qty: float
    avg_entry_price: Optional[float]
    mark_price: Optional[float]
    position_notional: float
    unrealized_pnl_usd: float
    ts_local: datetime = field(default_factory=utc_now)


@dataclass
class WorkingOrder:
    order_id_local: str
    order_id_exchange: Optional[int]
    client_order_id: Optional[str]
    symbol: str
    side: Side
    price: float
    size: float
    post_only: bool
    status: OrderStatus
    # Reduce-only flag. Venue enforces: the order can only close
    # position, never grow it past zero. Used by the soft-flatten
    # worker so a runaway can never invert position. Default False
    # for backward compat with normal quoting.
    reduce_only: bool = False
    # ``ts_created`` — local wall-clock when the WO object was constructed.
    # Set in NEW_LOCAL state, before the place HTTP is dispatched. For
    # hydrated WOs (status set directly to ACKED by ``_hydrate_working_from_exchange``),
    # ``ts_created`` equals hydration time, NOT the venue's create time.
    ts_created: datetime = field(default_factory=utc_now)
    # ``ts_sent`` — local wall-clock when the place HTTP request was
    # dispatched. ``None`` for hydrated orders. Used as the anchor for
    # the wall-lifetime cap (Phase 1C.4 of wedge-elimination-cleanup):
    # the bot guarantees no order lives on the wire past
    # ``BEHIND_TOUCH_MAX_AGE_SECONDS / AT_TOUCH_MAX_AGE_SECONDS``
    # measured from ``ts_sent``, regardless of when ACK arrives.
    ts_sent: Optional[datetime] = None
    # ``ts_ack`` — local wall-clock when the WO transitioned to ACKED.
    # SEMANTICS (Phase 0.3 contract): set at the moment the HTTP place
    # response confirms acceptance (place_outcome=accepted with an
    # exchange OID), NOT when WS ``live`` arrives. Reason: the place
    # response is the earliest reliable confirmation; the WS ``live``
    # event can arrive before the response (sub-ms race) or be lost.
    # For hydrated orders, ``ts_ack`` is set to the venue's ``uTime`` /
    # ``cTime`` so age math is consistent with the order's true on-wire
    # age (caveat: this is the venue's clock, not the bot's).
    #
    # The v1.4.67 wedge mechanism (snapshot v1.4.67-260518-195033) was
    # that ``ts_ack`` lagged the WS ``live`` event by 20-33 s because
    # of a WS→WO matcher race. Phase 1A buffers WS events by cloid so
    # the place-response handler can apply pre-arrival events
    # immediately on commit, eliminating the gap.
    ts_ack: Optional[datetime] = None
    # ``ts_closed`` — local wall-clock when the WO reached a terminal
    # status (CANCELED / FILLED / REJECTED / DESYNC). Set ONCE; if a
    # subsequent transition tries to mutate it, the order_store
    # transition method (Phase 3A) rejects.
    ts_closed: Optional[datetime] = None
    # v1.4.34 (Codex #1 fix): True for orders adopted from the
    # exchange's open-orders snapshot at startup / reconcile (i.e.
    # ``_hydrate_working_from_exchange``). Distinct from orders the
    # bot placed itself in this process lifetime.
    #
    # WHY THIS MATTERS:
    # 1. Fast-cancel post-only-cross detector (``execution.py``):
    #    the heuristic "ts_ack < 100 ms ago AND no cancel was
    #    requested by us → infer venue silently cancelled a
    #    crossing post-only place" mis-fires on hydrated orders,
    #    because their ``ts_ack`` is set from the VENUE's
    #    creation/update timestamp (or, in degraded paths, from the
    #    bot's clock at hydrate time). A long-resting order that
    #    happens to get cancelled by an external trigger
    #    (Binance-cross-venue, operator action via OKX UI, exchange
    #    sweep) right after restart would be mis-classified as a
    #    fresh cross reject and would arm the
    #    ``post_only_cross_cooldown``, suppressing future quotes
    #    for the cooldown window without cause.
    # 2. Aging callers (``app.quote_aging``) anchor ``order_age_s``
    #    to ``ts_ack``. For a hydrated order we use the venue's
    #    real ``uTime`` / ``cTime`` (HLOpenOrderRaw.timestamp) as
    #    ``ts_ack`` so age calculations remain accurate — but if the
    #    venue didn't supply a timestamp we fall back to wall-clock
    #    ``now`` which over-estimates freshness; the flag lets
    #    callers exempt hydrated orders from aggressive aging
    #    actions until the venue's first ack lands on a real local
    #    place.
    hydrated_from_exchange: bool = False
    # 2026-05-13 regime-observability Phase 1: cancel-race diagnostic
    # source. Set the moment the execution path issues a cancel for this
    # order. Used at fill ingestion to derive
    # ``cancel_requested_before_fill`` (was a cancel already in-flight
    # when this fill landed?) and ``ms_cancel_request_to_fill`` (how
    # close was the race?). None for orders whose cancel was never
    # requested (e.g. orders that filled in their natural lifetime).
    ts_cancel_requested: Optional[datetime] = None
    cancel_reason: Optional[str] = None
    replace_group_id: Optional[str] = None
    quote_cycle_id: Optional[str] = None
    # Database id of the SF episode this order belongs to, when the
    # order was placed during a SOFT_FLATTENING window. None for all
    # normal-quoting orders. Set on construction from
    # ``state.soft_flatten_event_id`` and persisted by ``order_row``;
    # downstream fills inherit it via SQL lookup. Plan ref:
    # plans/20260507-sf-frontend.md Phase 2.
    soft_flatten_event_id: Optional[int] = None
    # v1.5.33 — TP attribution mirror of SF. Set on construction from
    # ``state.tp_event_id`` when the bot is in TP mode at place-time;
    # propagated to the fill via the same storage lookup. None on all
    # non-TP orders. SF and TP are mutually exclusive at the executor
    # level so at most one of the two is non-None on a single row.
    tp_event_id: Optional[int] = None
    # todo-005 / todo-006: target half-spread (bps) the QuoteEngine
    # asked for at this order's placement, and a coarse aggressiveness
    # tag (``at_touch`` / ``inside`` / ``aged_tightened``). Both are
    # populated by ``OrderManager._stage_place_order_local`` and
    # propagated onto the resulting Fill via storage lookup at
    # ingestion time. Supports the offline "spread quality" report
    # (target − captured) and the live FillBucketsCard 2D slicing
    # (quote-age × aggressiveness).
    target_half_spread_bps: Optional[float] = None
    quote_aggressiveness: Optional[str] = None
    # v1.5.190 Phase 8A Option C — per-order Avellaneda-Stoikov
    # attribution. Snapshot of ``state.last_quote_breakdown.base_half_
    # spread_bps`` at this order's placement time. When AS is enabled
    # the breakdown's base half-spread is the AS-computed value
    # (``compute_as_half_spread_bps``); when AS is disabled or the
    # path didn't fire (no k-intensity cache yet, etc.) this is the
    # legacy vol-adaptive base. NULL on legacy / SF / TP / manual paths
    # where no breakdown is available. STAMP ONLY — never read back
    # into quote construction. Propagated to the Fill via the same
    # ``_DECISION_STATE_COLS`` lookup the other ``*_at_decision``
    # fields use, so the postmortem can decompose per-fill captured
    # spread into "AS-base contribution" vs. "skew / overlay / inventory
    # widen contribution" without re-deriving from the time-series.
    as_base_half_spread_bps_at_decision: Optional[float] = None
    # Transport worker: monotonic per-side intent generation for stale completion drops.
    transport_intent_seq: int = 0
    cancel_transport_seq: int = 0
    # 2026-05-13 regime-observability Phase 1: decision-state stamping.
    # These fields are populated at WorkingOrder creation time from the
    # QuoteDecision / state snapshot the order was emitted from. They
    # flow into the orders table and are propagated onto each resulting
    # Fill via a single indexed storage lookup at fill-ingestion time
    # (same pattern as ``target_half_spread_bps`` /
    # ``quote_aggressiveness`` above).
    #
    # Purely observational — none of these fields are read back into the
    # quote-construction path. They exist so the operator can answer
    # "what regime was the bot in when this fill happened?" from
    # snapshot data without joining against the quote-cycle time-series.
    # See ``plans/regime-observability.md`` Phase 1 for the full rationale.
    toxicity_score_at_decision: Optional[float] = None
    vol_estimate_at_decision: Optional[float] = None
    active_sides_at_decision: Optional[str] = None
    decision_reason_at_decision: Optional[str] = None
    binance_basis_ewma_at_decision: Optional[float] = None
    adaptive_widen_active_at_decision: Optional[bool] = None
    post_fill_cooldown_active_bid_at_decision: Optional[bool] = None
    post_fill_cooldown_active_ask_at_decision: Optional[bool] = None
    at_touch_adverse_pause_bid_at_decision: Optional[bool] = None
    at_touch_adverse_pause_ask_at_decision: Optional[bool] = None
    quote_distance_to_touch_ticks_at_placement: Optional[float] = None
    # 2026-05-13 regime-observability Phase 4a: placeholder formula
    # for expected net edge. STAMP ONLY — not a quote-construction
    # input. See plans/regime-observability.md Phase 4a scope guard.
    expected_net_edge_bps_at_decision: Optional[float] = None
    # 2026-05-14 todo-027 Tier 2: instrumentation for the three
    # "fire-counted-only" gates so the dashboard's
    # gate-effectiveness table can populate every column for them.
    # vol_trend / post_swing are bools (gate active at place-time);
    # session_drawdown_tier is a string label (CLEAR / WIDEN /
    # PAUSE_SHORT / PAUSE_LONG / RESUME_TESTING / KILLED) because
    # the ladder has more than a binary "active" state. All three
    # are STAMP-ONLY — none are read back into quote construction.
    vol_trend_active_at_decision: Optional[bool] = None
    post_swing_active_at_decision: Optional[bool] = None
    session_drawdown_tier_at_decision: Optional[str] = None
    # v1.4.175 Phase 3F — reservation-alpha shift contributions in bps
    # AT THE DECISION that produced this order. Read from
    # ``state.last_quote_breakdown`` in ``place_passive_order_manual_only``
    # and stamped here so the postmortem can correlate per-alpha shift
    # direction with per-fill realised markout. STAMP-ONLY: not read
    # back into any quote-construction path. NULL on legacy / SF /
    # manual-test paths where no breakdown is available.
    ob_imbalance_shift_bps_at_decision: Optional[float] = None
    trend_drift_shift_bps_at_decision: Optional[float] = None
    flow_score_shift_bps_at_decision: Optional[float] = None
    basis_deviation_shift_bps_at_decision: Optional[float] = None
    # v1.5.204 Phase 4A — microprice-gate widening contribution at
    # decision time. Per-side because the gate is asymmetric: only
    # the thin side gets widened (the other side is 0). Read from
    # ``state.last_quote_breakdown.microprice_{bid,ask}_bps`` in
    # ``place_passive_order_manual_only``. NULL on legacy / SF /
    # manual-test paths. STAMP ONLY — not read back into quote
    # construction. Consumer: the v1.5.204 postmortem section
    # ``microprice_widen_attribution`` + the snapshot acceptance check
    # ``check_v1_5_204_microprice_widen_markout_within_noise`` (the
    # Phase 4A.3 iteration acceptance gate).
    microprice_bid_widen_bps_at_decision: Optional[float] = None
    microprice_ask_widen_bps_at_decision: Optional[float] = None
    # v1.5.306 audit §5 P0 #2 — Active-Quoting-Controller state at
    # decision time. Stamped from ``state.active_quoting_controller``
    # (the PI aggression output [0,1] + the markout safety-floor flag)
    # at place-time. NULL on legacy / SF / manual-test paths + whenever
    # AQC is disabled (controller is None). STAMP ONLY — the
    # controller's quoting effect already flows through the half-spread
    # floor / inventory gates / skew; these record what its state was so
    # the postmortem can ask "did fills placed while aggression was high
    # / the safety floor engaged show better/worse markout?". Propagated
    # to each Fill via the ``_DECISION_STATE_COLS`` lookup.
    aqc_aggression_level_at_decision: Optional[float] = None
    aqc_safety_floor_engaged_at_decision: Optional[bool] = None

    # 1.3.82 connectivity diagnostics — full attribution of every
    # gone_on_exchange / phantom-place / late-cancel event back to
    # the specific code path + venue response that produced it.
    # Stamped at place-response + cancel-dispatch + cancel-response
    # time. None when not applicable (legacy / non-MM path).
    #
    # cancel_trigger_reason — the code path that issued the cancel:
    #   reprice_replace / hard_age_cap / side_suppressed /
    #   soft_flatten / binance_cross_venue / kill_flow / reconcile /
    #   startup_cancel_all / risk_flatten / other.
    cancel_trigger_reason: Optional[str] = None
    # When the place HTTP response was interpreted (NULL when never
    # received — phantom-place signature). Distinct from ts_ack:
    # ts_ack is only set on a successful accepted-with-oid response,
    # whereas ts_place_response is set on ANY parseable response
    # (including rejections and unconfirmed paths).
    ts_place_response: Optional[datetime] = None
    # Place-response classification:
    #   accepted / exchange_rejected / transport_rejected /
    #   unconfirmed (NULL when never received).
    place_response_outcome: Optional[str] = None
    # Cancel-response classification:
    #   success / benign_missing / transport / error (NULL when
    #   never sent or never received).
    cancel_response_outcome: Optional[str] = None
    # 1.3.83: detail strings from the venue (the OKX sCode + sMsg or
    # equivalent). Populated alongside the *_outcome category. Used
    # by the Connectivity tab to group rejections by specific reason
    # (e.g. ``post_only_would_cross`` is benign, ``insufficient_margin``
    # is not; ``okx_row_51400`` is order-already-gone, ``okx_row_50001``
    # is rate-limited; etc.). NULL on accepted/success rows where there
    # is no reject detail to record.
    place_response_detail: Optional[str] = None
    cancel_response_detail: Optional[str] = None
    # 1.3.86: cancel-before-ack defer flag. Set by ``cancel_order`` /
    # ``_enqueue_cancel_quote_path`` when invoked on a SENT order
    # whose ts_ack is None (the place HTTP response hasn't landed
    # yet). Flushed by the place-response handler the moment the
    # order transitions to ACKED — at that point the deferred cancel
    # fires for real, with the original trigger_reason intact.
    # Cleared on REJECTED transitions (nothing left to cancel).
    # In-memory only — not persisted to SQLite (bot restart relies on
    # cancel-all-on-startup to clean up any straggler).
    cancel_pending_after_ack: bool = False
    # 1.4.0 cancel-prio Phase 0.5 — cancel-latency decomposition.
    # ``ts_cancel_requested`` is the bot's DECISION timestamp (t1).
    # The three fields below are the missing legs that let the
    # operator (and the postmortem tool) decompose cancel latency
    # into (decision→send), (send→ack-via-HTTP), (decision→close-
    # via-WS). Without them the dashboard's "cancel→confirm"
    # histogram bundles six legs into one bar.
    #
    # All three are optional. They're stamped by
    # ``OrderManager._cancel_http_transport`` (sent/acked) and the
    # WS CANCELED handler (venue uTime). NULL on legacy rows + on
    # cancel-failed-because-order-filled samples (which are dropped
    # from the latency aggregator anyway via ``cancel_response_outcome``
    # filtering).
    #
    # ``ts_cancel_sent`` — bot wall clock immediately before the
    # HTTP cancel request leaves the process. Used as t2.
    ts_cancel_sent: Optional[datetime] = None
    # ``ts_cancel_acked`` — bot wall clock immediately after
    # ``_interpret_cancel_response`` returns success on the HTTP
    # response. Used as t3 (HTTP-ack variant). NULL on transport
    # errors (no real ack came back).
    ts_cancel_acked: Optional[datetime] = None
    # ``venue_cancel_utime_ms`` — OKX server-side timestamp from the
    # WS ``orders`` channel CANCELED event (``uTime`` field, ms
    # epoch). Sanity-check anchor: any drift between this and
    # ``ts_cancel_acked`` reveals clock skew + WS path length.
    # NULL on non-OKX venues or when the WS event omits uTime.
    venue_cancel_utime_ms: Optional[int] = None
    # 1.3.130 multi-rung Phase 2: which rung this order represents in
    # the ladder. ``0`` = inside rung (closest to mid); higher values
    # = outer rungs at wider prices. At N=1 (single-rung mode, the
    # default for backward compat) this stays 0 on every order.
    # Persisted as a column on the orders table so the postmortem can
    # attribute per-rung markout / fill-rate. Identical in spirit to
    # ``LadderRung.level_idx`` in app.ladder.
    level_idx: int = 0

    # 1.4.15 amend-prio Phase 1: amend lifecycle fields.
    #
    # ``amend_intent_seq`` parallels ``transport_intent_seq`` (places)
    # and ``cancel_transport_seq`` (cancels) — monotonic per-WO counter
    # bumped when an amend is enqueued. The dispatcher checks it to
    # drop stale intents whose target price/size no longer matches the
    # latest reprice decision (analogous to the place-intent
    # staleness filter in ``_execute_place_intent``).
    #
    # ``amend_target_px`` / ``amend_target_sz`` capture the new
    # price/size an in-flight amend is trying to apply. On success
    # they're commited into ``price`` / ``size`` and cleared; on any
    # other terminal outcome they're cleared without mutating the
    # underlying fields.
    #
    # ``ts_amend_sent`` / ``ts_amend_response`` mirror the
    # ``ts_cancel_sent`` / ``ts_cancel_acked`` pattern for postmortem
    # RTT analysis.
    #
    # ``amend_response_outcome`` is one of
    # ``accepted`` / ``below_filled`` / ``order_gone`` /
    # ``exchange_rejected`` / ``transport_rejected`` / ``unconfirmed``
    # — matches the kinds produced by
    # ``interpret_okx_amend_batch_response``.
    #
    # Storage: persisted via migration v24. Legacy rows have NULLs
    # and downstream code treats them as "no amend ever attempted."
    amend_intent_seq: int = 0
    amend_target_px: Optional[float] = None
    amend_target_sz: Optional[float] = None
    ts_amend_sent: Optional[datetime] = None
    ts_amend_response: Optional[datetime] = None
    amend_response_outcome: Optional[str] = None
    amend_response_detail: Optional[str] = None
    # v1.4.169 Phase 2I — per-order amend rate guard. Each entry is a
    # ``_clock.monotonic()`` from the moment an amend intent was
    # SUCCESSFULLY dispatched (placed onto the outbound queue). The
    # rate-defence guard at the top of ``_enqueue_amend_quote_path``
    # prunes entries older than 1 s on every check and rejects new
    # dispatches when:
    #   * the most-recent entry is less than
    #     ``AMEND_TICK_FLICKER_MIN_MS`` ago, OR
    #   * the live count exceeds ``AMEND_PER_ORDER_MAX_PER_SEC``.
    # Source: snapshot v1.4.102-260520-120555 at 12:31:07 — a single
    # SELL order received 244 amend dispatches in ~1 s as the bot's
    # computed price ping-ponged between two adjacent ticks
    # (1.941 ↔ 1.942) every 5-7 ms quote cycle. OKX's per-order amend
    # rate-limit (50011) fired. The per-account aggregate counter
    # showed plenty of headroom (12/2s vs cap) so the breach was
    # invisible to existing pacing. This buffer + the guard are the
    # structural fix.
    amend_recent_dispatch_mono: list[float] = field(default_factory=list)

    def resting_age_seconds(self, now: datetime) -> Optional[float]:
        """Seconds since the order became live (ACKED); None if not resting."""
        if self.ts_ack is None:
            return None
        if self.status not in (OrderStatus.ACKED, OrderStatus.PARTIAL):
            return None
        return max(0.0, (now - self.ts_ack).total_seconds())


@dataclass
class Fill:
    fill_id: str
    order_id_exchange: Optional[int]
    client_order_id: Optional[str]
    ts_fill: datetime
    symbol: str
    side: Side
    price: float
    size: float
    notional: float
    fee: float
    liquidity_flag: str
    # Same L2 HTTP snapshot as bid/ask below (ingestion-time book), not a delayed markout.
    mid_at_fill: Optional[float]
    best_bid_at_fill: Optional[float] = None
    best_ask_at_fill: Optional[float] = None
    # "full" = bid+ask present; "mid_only" = only mid usable; "unknown" = legacy / missing
    book_snapshot_quality: str = "unknown"
    # How the fill-time book row was aligned to ts_fill (honest, local-clock semantics).
    book_reference_quality: str = "missing_reference"
    markout_1s_bps: Optional[float] = None
    markout_3s_bps: Optional[float] = None
    markout_5s_bps: Optional[float] = None
    # v1.4.98 — extended markout horizons aligned to the bot's own
    # time-constants. The 1s/3s/5s endpoints above measure HFT-immediate
    # pickoff; these capture the inflection points the bot's gates
    # actually operate on:
    #   * 15s  — post-first-cancel; bot has cycled its quote 30× by now
    #            (QUOTE_LOOP_SECONDS=0.5, BEHIND_TOUCH_MAX_AGE=1.5,
    #             AT_TOUCH_MAX_AGE=2.5) so adverse selection from the
    #            ORIGINAL fill's price level is fully expressed.
    #   * 30s  — soft-flatten / adverse-side-pause window
    #            (POSITION_DRAWDOWN_GATE_DURATION_SECONDS=30,
    #             ADVERSE_SIDE_PAUSE_SECONDS=30, AT_TOUCH_ADVERSE_PAUSE_
    #             SECONDS=30). Captures whether the position-skew
    #             machinery's exit window engaged.
    #   * 60s  — post-cooldown clearance (POST_SWING_WINDOW_SECONDS=60,
    #             WATCHDOG_QUOTE_ACTIVITY_WINDOW_SECONDS=120 half).
    #             Captures whether the regime cleared.
    #   * 120s — long-tail held-position settlement. Useful for
    #             decomposing "residual" (PnL outside the 5s window)
    #             into "what eventually got marked through the 120s
    #             window" vs "still-open exposure beyond 120s".
    #
    # All four are diagnostic-only — no gate reads them. The 5s endpoint
    # remains canonical for sizing/aging control surfaces.
    markout_15s_bps: Optional[float] = None
    markout_30s_bps: Optional[float] = None
    markout_60s_bps: Optional[float] = None
    markout_120s_bps: Optional[float] = None
    # v1.4.100 ladder-observability F1 — which rung the parent
    # working-order represented. ``0`` = inside rung (closest to
    # mid), ``1+`` = outer rungs at wider prices. At N=1 (single-
    # rung mode) this stays 0 on every fill. Populated at ingest
    # time from the parent WorkingOrder's ``level_idx`` field
    # (which has existed since v1.3.130). Legacy fills (pre-
    # v1.4.100) carry ``0`` from the schema default.
    #
    # Enables ``tools/postmortem/sections/ladder_attribution.py``
    # to compute per-rung markout breakdowns. Without this, the
    # postmortem can only see aggregate fill economics, not
    # rung-decomposed economics — i.e. cannot answer "is rung-1
    # profitable?" which gates the N=2 → N=3 bump decision.
    level_idx: int = 0
    # Diagnostics at fill time (quote eligibility guard); optional for backward compatibility.
    quote_eligibility_state: Optional[str] = None
    quote_eligibility_reason: Optional[str] = None
    book_age_seconds_at_fill: Optional[float] = None
    mid_return_100ms_bps_at_fill: Optional[float] = None
    mid_return_250ms_bps_at_fill: Optional[float] = None
    mid_return_500ms_bps_at_fill: Optional[float] = None
    fill_during_quote_cooldown: bool = False
    # Inbound latency attribution (session diagnostics; optional).
    effective_book_age_at_last_decision_ms: Optional[float] = None
    effective_book_age_at_fill_ms: Optional[float] = None
    private_ws_receive_to_state_apply_ms: Optional[float] = None
    quote_cycle_to_first_transport_send_ms: Optional[float] = None
    quote_cycle_to_first_ack_ms: Optional[float] = None
    ack_to_private_ws_lifecycle_ms: Optional[float] = None
    # Exchange-reported realised (closed) PnL component for this fill.
    # Populated from venue private-WS / REST event payloads (Bluefin's
    # ``realizedPnlE9`` field, etc.). Zero for pure opening fills,
    # non-zero on reducing/closing fills. Surfaced to Telegram fill
    # broadcasts and any observer that wants per-fill realised PnL
    # without re-deriving it from position state.
    closed_pnl: Optional[float] = None
    # todo-005 / todo-006: target half-spread the parent order asked
    # for, and aggressiveness category at place-time (one of
    # ``at_touch`` / ``inside`` / ``aged_tightened``). Both copied
    # from the parent ``WorkingOrder`` row at fill-ingestion time
    # (single indexed lookup against ``orders.order_id_exchange``).
    # NULL on legacy fills + on fills whose parent order isn't in
    # the local DB. Used by the Bot Stats fill-bucket card and the
    # offline metrics report's "spread quality" section.
    target_half_spread_bps: Optional[float] = None
    quote_aggressiveness: Optional[str] = None
    # v1.5.190 Phase 8A Option C — per-fill Avellaneda-Stoikov attribution.
    # Snapshot of the parent order's ``base_half_spread_bps`` at decision
    # time. See ``WorkingOrder.as_base_half_spread_bps_at_decision`` for
    # the source-of-truth comment. Propagated at fill-ingest time via the
    # ``order_metadata_for_fill_ingest`` lookup (same channel as
    # ``target_half_spread_bps`` / ``toxicity_score_at_decision`` / etc.).
    # NULL on legacy fills (pre-v1.5.190) and on fills whose parent order
    # isn't in the local DB or had no breakdown recorded.
    as_base_half_spread_bps_at_decision: Optional[float] = None
    # todo-006: how long the parent order rested between venue ACK and
    # being hit, in ms. Computed at fill-ingest time by
    # ``FillBucketAggregator.note_fill`` (which returns the age it just
    # bucketed) and stamped here so the offline postmortem can slice
    # fills by quote-age without the in-memory bucket aggregator.
    # NULL for non-session-scoped REST catchup fills (the ack cache
    # only covers the current process's session) and for fills whose
    # parent order's ack was evicted or never recorded.
    quote_age_at_fill_ms: Optional[float] = None
    # Database id of the SF episode this fill belongs to, copied from
    # the parent order's ``soft_flatten_event_id`` at fill-ingest time
    # (see ``app.fill_ingestion.ingest_hl_fill_raw``). None when the
    # parent order was placed outside an SF window, when the parent
    # order isn't in the local DB (shouldn't happen but is harmless),
    # or when the fill's ``order_id_exchange`` is unknown.
    soft_flatten_event_id: Optional[int] = None
    # v1.5.33 — TP attribution mirror of SF on the fill side. Stamped
    # at fill-ingest time via ``app.fill_ingestion.resolve_tp_event_id_
    # for_fill`` (parent-order metadata hit → active-TP fallback →
    # grace cache). NULL on non-TP fills.
    tp_event_id: Optional[int] = None
    # v1.4.173 (Phase 4D.4) — SF phase-ladder phase in effect when the
    # fill arrived. Integer in {0..4} matching ``soft_flatten.PHASE_*``:
    #   0 = post-only at near touch (passive, rebate)
    #   1 = post-only at far touch + 1 tick (passive, rebate)
    #   2 = IOC cross 1 tick (taker, cross @ opposite touch)
    #   3 = IOC cross 2 ticks (taker, eats one extra level worst case)
    #   4 = market_close (taker, terminal)
    # Stamped at ingestion from ``state.sf_phase_ladder_phase`` when SF
    # is active. NULL on non-SF fills and on legacy / ladder-disabled
    # episodes. Used by ``_exit_soft_flatten`` to compute the per-
    # episode ``fills_by_phase_json`` + ``taker_spread_bps_paid``
    # columns on ``soft_flatten_events``.
    sf_force_phase: Optional[int] = None
    # Analysis-day instrumentation (2026-05-10).
    # N1 — Basis-regime sign (-1 / 0 / +1) in effect at fill time.
    # Snapshot of ``state.basis_regime.last_regime_sign`` taken when
    # the fill is recorded. Lets postmortem reports decompose PnL by
    # regime axis without joining against the quote-cycle time-series.
    basis_regime_sign: Optional[int] = None
    # N3 — Top-of-book sizes at fill time, copied from the
    # ``state.market`` ``BestBidAsk`` snapshot. Pairs with
    # ``best_bid_at_fill`` / ``best_ask_at_fill`` (which carry the
    # prices) so the postmortem queue-imbalance / depth-conditional
    # markout analysis can run against a complete top-of-book row
    # per fill. NULL when the venue's BBO didn't carry sizes (e.g.
    # legacy / partial book).
    bid_size_top_at_fill: Optional[float] = None
    ask_size_top_at_fill: Optional[float] = None
    # 2026-05-13 regime-observability Phase 1.
    #
    # Trivial derivations computed at fill ingestion from data already
    # in scope (best_bid_at_fill / best_ask_at_fill, bid_size_top /
    # ask_size_top, state.position):
    #
    #   - spread_bps_at_fill = (ask - bid) / mid * 10000
    #   - microprice_at_fill = (bid_sz*ask + ask_sz*bid) / (bid_sz+ask_sz)
    #   - imbalance_top_at_fill = (bid_sz - ask_sz) / (bid_sz + ask_sz)
    #   - inventory_qty_before_fill = state.position.position_qty
    #                                 - (signed_qty_of_this_fill)
    #   - inventory_utilization_before_fill = |inv_before| / max_abs_pos
    spread_bps_at_fill: Optional[float] = None
    microprice_at_fill: Optional[float] = None
    imbalance_top_at_fill: Optional[float] = None
    inventory_qty_before_fill: Optional[float] = None
    inventory_utilization_before_fill: Optional[float] = None
    # Decision-state propagation. Same channel as
    # ``target_half_spread_bps`` / ``quote_aggressiveness`` above: copied
    # from the parent ``WorkingOrder`` row at fill-ingestion time via
    # ``Storage.order_decision_state_for_order``. NULL on legacy fills
    # and on fills whose parent order isn't in the local DB. Purely
    # observational — never read back into the quote-construction path.
    toxicity_score_at_decision: Optional[float] = None
    vol_estimate_at_decision: Optional[float] = None
    active_sides_at_decision: Optional[str] = None
    decision_reason_at_decision: Optional[str] = None
    binance_basis_ewma_at_decision: Optional[float] = None
    adaptive_widen_active_at_decision: Optional[bool] = None
    post_fill_cooldown_active_bid_at_decision: Optional[bool] = None
    post_fill_cooldown_active_ask_at_decision: Optional[bool] = None
    at_touch_adverse_pause_bid_at_decision: Optional[bool] = None
    at_touch_adverse_pause_ask_at_decision: Optional[bool] = None
    quote_distance_to_touch_ticks_at_placement: Optional[float] = None
    # Cancel-race diagnostics. ``cancel_requested_before_fill`` is True
    # when the bot had already issued a cancel for the parent order
    # before this fill arrived (cancel raced the fill). The ms delta
    # gives the magnitude of the race — small positive values mean we
    # *just barely* lost. Both NULL when no cancel was requested at all
    # (the parent order filled while still in normal-resting state).
    cancel_requested_before_fill: Optional[bool] = None
    ms_cancel_request_to_fill: Optional[float] = None
    # 2026-05-13 regime-observability Phase 4a: stamped placeholder
    # expected net edge. Propagated from parent order. STAMP ONLY —
    # not a quote-construction input. See
    # plans/regime-observability.md Phase 4a scope guard.
    expected_net_edge_bps_at_decision: Optional[float] = None
    # 2026-05-14 todo-027 Tier 2: instrumentation for the three
    # "fire-counted-only" gates. Propagated from parent order via the
    # same ``order_decision_state_for_order`` lookup the other
    # ``*_at_decision`` fields use. See WorkingOrder for the
    # source-of-truth comment.
    vol_trend_active_at_decision: Optional[bool] = None
    post_swing_active_at_decision: Optional[bool] = None
    session_drawdown_tier_at_decision: Optional[str] = None
    # v1.4.175 Phase 3F — reservation-alpha shifts at decision time,
    # propagated from the parent order via
    # ``order_metadata_for_fill_ingest``. See WorkingOrder for the
    # source-of-truth comment + the postmortem
    # ``reservation_alpha_attribution`` section for the consumer.
    ob_imbalance_shift_bps_at_decision: Optional[float] = None
    trend_drift_shift_bps_at_decision: Optional[float] = None
    flow_score_shift_bps_at_decision: Optional[float] = None
    basis_deviation_shift_bps_at_decision: Optional[float] = None
    # v1.5.204 Phase 4A — microprice-gate widening contribution
    # at decision time, propagated from the parent order via
    # ``order_metadata_for_fill_ingest``. See WorkingOrder for the
    # source-of-truth comment + the postmortem section
    # ``microprice_widen_attribution`` for the consumer.
    microprice_bid_widen_bps_at_decision: Optional[float] = None
    microprice_ask_widen_bps_at_decision: Optional[float] = None
    # v1.5.306 audit §5 P0 #2 — Active-Quoting-Controller state at
    # decision time, propagated from the parent order via
    # ``order_metadata_for_fill_ingest``. See WorkingOrder for the
    # source-of-truth comment + the postmortem consumer.
    aqc_aggression_level_at_decision: Optional[float] = None
    aqc_safety_floor_engaged_at_decision: Optional[bool] = None
    # 2026-05-13 regime-observability Phase 4c: maximum adverse /
    # maximum favorable excursion in the N-second window post-fill.
    # Written LATE (5s and 30s after the fill) by
    # ``PostFillExcursionWatcher`` via an UPDATE — so NULL on the
    # fill row right after ingestion, populated by the deadline tick.
    # Sign convention: from the fill side's perspective.
    #   BUY:  MAE = (min_mid - mid_at_fill) / mid_at_fill * 10000
    #         MFE = (max_mid - mid_at_fill) / mid_at_fill * 10000
    #   SELL: MAE = (mid_at_fill - max_mid) / mid_at_fill * 10000
    #         MFE = (mid_at_fill - min_mid) / mid_at_fill * 10000
    # MAE is always ≤ 0 (or 0 if never adverse); MFE ≥ 0.
    mae_5s_bps: Optional[float] = None
    mfe_5s_bps: Optional[float] = None
    mae_30s_bps: Optional[float] = None
    mfe_30s_bps: Optional[float] = None
    # 2026-05-13 regime-observability Phase 4c follow-up: time-to-flat.
    # Seconds elapsed from this fill's timestamp to the next moment
    # ``state.position.position_qty`` crossed (or returned to) zero.
    # Written LATE by ``PostFillTimeToFlatWatcher`` via UPDATE when
    # the flat-crossing is observed. NULL when:
    #   - The watcher is disabled (``OBSERVABILITY_TIME_TO_FLAT_ENABLED=false``)
    #   - Position never returned to flat within the max-wait cap
    #     (default 300s); the row stays NULL rather than carrying a
    #     "we hit the cap" sentinel — operator sees NULL = unflat,
    #     populated = duration in seconds.
    #   - The session ended with non-zero position (the watcher
    #     thread dies; the row's NULL captures "still holding".)
    # Purely observational. Combined with markout / MAE / MFE this
    # answers "how long was my inventory exposed after this fill?"
    # — the residual-loss attribution question.
    time_to_flat_seconds: Optional[float] = None


@dataclass
class QuoteDecision:
    ts: datetime
    symbol: str
    mid_price: float
    vol_estimate: float
    inventory: float
    reservation_price: float
    target_spread_bps: float
    target_bid: float
    target_ask: float
    quoted_bid: float
    quoted_ask: float
    quoted_bid_sz: float
    quoted_ask_sz: float
    active_sides: ActiveSides
    toxicity_score: float
    decision_reason: str
    quote_cycle_id: str
    # Extra economic half-spread floor (bps) from adaptive widen cooldown; see bot + quoting.
    spread_floor_overlay_half_spread_bps: float = 0.0
    # v1.5.281 AQC Phase 2: live controller aggression_level in [0,1]
    # stamped from ``state.active_quoting_controller`` at decision time
    # (bot.py), or None when AQC is disabled. Threaded into
    # ``compute_effective_min_half_spread_bps`` to tighten the economic
    # min-half-spread base floor (tighten-only), but ONLY applied when
    # ``AQC_WIRE_MIN_HALF_SPREAD`` is on — so None / wire-off both leave
    # the spread pipeline byte-identical to Phase 1. See
    # app/active_quoting_controller.py + plans/aqc-execute.md Phase 2.
    aqc_aggression_level: Optional[float] = None
    # Pre-inventory eligibility cap (QUOTE_*) applied after compute; see quote_eligibility + bot.
    quote_eligibility: Optional[str] = None
    quote_eligibility_reason: Optional[str] = None
    # Market-data / inbound timing at decision (complements quote_eligibility telemetry).
    source_book_ts_exchange_ms: Optional[int] = None
    source_book_ts_local_iso: Optional[str] = None
    effective_book_age_at_decision_ms: Optional[float] = None
    book_apply_to_decision_ms: Optional[float] = None
    public_ws_queue_wait_ms_latest: Optional[float] = None
    public_ws_receive_to_apply_ms_latest: Optional[float] = None
    decision_market_data_regime: Optional[str] = None
    # Microprice actually used as the reservation reference (None if the
    # feature is disabled, top-of-book depth was unavailable, or the
    # formula evaluated to a non-finite value — in all three cases the
    # reservation falls back to ``mid_price``). Persisted for offline
    # analysis so we can compare the reservation's behaviour before and
    # after the microprice refactor.
    microprice: Optional[float] = None
    # Binance Level 1 cross-venue reference, snapshotted at decision time.
    # Both are None when ``BINANCE_WS_ENABLED=false``, the feed hasn't yet
    # delivered its first message, or the basis EWMA hasn't been seeded
    # (requires one sample with GRVT mid present too). Enables per-cycle
    # shadow analysis: "what fair value was Binance implying when this
    # quote was built, and how big was the GRVT-vs-Binance basis?"
    # Paired with the ``binance_cross_venue_cancel`` event (which only
    # fires on trigger moments), these fields give every quote cycle
    # cross-venue context — decisive for understanding when our quotes
    # drift past Binance's lead without the cancel-on-move threshold
    # being breached.
    binance_mid: Optional[float] = None
    binance_basis_ewma: Optional[float] = None
    # Analysis-day instrumentation (2026-05-10) N2: per-quote-cycle
    # snapshot of the basis-regime classifier's state. Sampling these
    # at every quote cycle lets postmortem reports retrospectively
    # tune the IC threshold (today's 0.15 is a guess) and decompose
    # PnL by regime sign across an arbitrary time window.
    #
    # ``basis_regime_sign``: -1 / 0 / +1 (0 = below IC threshold,
    #     classifier is dormant — no regime signal active).
    # ``basis_ic``: rolling Spearman / Pearson IC of basis vs forward
    #     return on the last N paired samples. None when fewer than
    #     ``min_pair_samples`` paired observations exist yet.
    # ``basis_pair_count``: number of paired samples backing the IC
    #     at decision time; sanity-check for the IC's reliability.
    basis_regime_sign: Optional[int] = None
    basis_ic: Optional[float] = None
    basis_pair_count: Optional[int] = None
    # 1.2.34: per-cycle decomposition for the dashboard's Spread
    # tab. Populated by ``compute_quote_decision`` with every
    # input that contributed to this cycle's quote (mid →
    # reservation shifts, half-spread stack, size mult chain).
    # Use ``Any`` to avoid the import cycle with
    # ``app.quote_breakdown``; the actual type is
    # ``QuoteBreakdownSnapshot``. Backward-compat: default None
    # so callers built pre-1.2.34 don't need to populate it.
    breakdown: Any = None


@dataclass
class PnlSnapshot:
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    total_pnl_usd: float
    fees_usd: float
    equity_usd: Optional[float]
    drawdown_usd: float
    session_peak_equity_usd: float
    ts: datetime = field(default_factory=utc_now)


@dataclass
class BotEvent:
    ts: datetime
    severity: EventSeverity
    event_type: str
    message: str
    payload: Optional[dict[str, Any]] = None


@dataclass
class RiskDecision:
    action: RiskAction
    reasons: list[str]
    bid_size_mult: float = 1.0
    ask_size_mult: float = 1.0
    spread_add_bps: float = 0.0


@dataclass
class ToxicitySnapshot:
    score: float
    one_sided_fill_ratio: float
    avg_adverse_markout_bps: float
    vol_spike_ratio: float
    hard_trigger: bool
    soft_trigger: bool
    toxic_side: Optional[Side] = None
    ts: datetime = field(default_factory=utc_now)
    # Adverse leg uses resolved delayed markouts only (not mid_at_fill vs current mid).
    delayed_markout_sample_count: int = 0
    adverse_uses_delayed_markouts: bool = False
    # Per-side average markout (bps, negative = adverse) across the recent-fill window.
    # ``None`` when that side has fewer than ``adverse_side_pause_min_fills`` resolved
    # markouts. Consumed by ``OrderManager._maybe_arm_adverse_side_pause`` for localized
    # side withdrawal (see ``adverse_side_pause_*`` settings).
    buy_side_avg_markout_bps: Optional[float] = None
    sell_side_avg_markout_bps: Optional[float] = None
    buy_side_fill_count: int = 0
    sell_side_fill_count: int = 0
