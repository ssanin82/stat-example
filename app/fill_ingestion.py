"""Shared fill ingestion for REST recovery and private websocket (single downstream path)."""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Literal, Optional

from app.enums import Side
from app.exchange.hyperliquid_types import HLFillRaw
from app.exchange.private_events import PrivateFillEvent
from app.inbound_timing import InboundPrivateTiming, private_timing_derived_ms
from app.markout import register_fill_for_delayed_markouts
from app.models import Fill
from app.pnl import PnlTracker
from app.state import BotState
from app.storage import Storage

from app import clock as _clock

logger = logging.getLogger(__name__)

FillSource = Literal["rest", "private_ws", "rest_catchup"]


def resolve_sf_event_id_for_fill(
    *,
    parent_meta_sf_event_id: Optional[int],
    state: BotState,
    storage: Storage,
    oid: str,
    now_mono: float,
) -> Optional[int]:
    """Resolve the SF event id to stamp on a fill (or None).

    The labeling pipeline has been bug-prone (broke on v1.4.157 and
    v1.4.189 for the taker paths), so the resolution chain is pulled
    out of the hot-path inline conditional pyramid into this named
    helper. Unit-tested in ``tests/test_sf_event_id_labeling.py``.

    Priority chain — first hit wins:

    1. **Parent-order lookup hit** (the happy path). When the bot's
       ``OrderManager`` places an SF order it stamps the row's
       ``soft_flatten_event_id`` from ``state.soft_flatten_event_id``
       at insert time. Any later fill on that oid finds it via
       ``storage.order_metadata_for_fill_ingest``. Covers 100 % of
       maker / post-only SF fills.

    2. **Active-SF fallback** (v1.4.163). When the SF taker paths
       (``client.market_close`` / ``client.place_ioc_reduce_only``)
       fire, they bypass ``OrderManager`` — no row gets written with
       a real exchange oid. The corresponding fills arrive later with
       no parent order. If ``state.soft_flatten_event_id`` is STILL
       set at ingest time (i.e. the WS fill beat the SF exit), stamp
       from state.

    3. **Recent-event grace fallback** (v1.4.192). Closes the
       structural race between ``_exit_soft_flatten`` (which clears
       ``state.soft_flatten_event_id`` synchronously inside the same
       SF tick that fired ``client.market_close``) and the late-
       arriving WS fill (which lands tens-to-hundreds of ms later).
       ``_exit_soft_flatten`` copies the just-cleared event id into
       ``state.sf_recent_event_id`` with a TTL
       (``sf_recent_event_id_valid_until_mono``); this fallback reads
       it. Default TTL is 30 s — well beyond any observed WS fill
       latency, well below typical SF-re-entry cadence.

    The ``parent_order_exists`` guard on fallbacks 2 + 3 distinguishes
    "no parent" (the taker case we want to fix) from "parent exists
    but legitimately untagged" (e.g. a pre-SF order whose cancel
    raced a fill — those must keep ``sf=None``).

    Side effects on the grace-fallback hit:

    - Bumps ``state.sf_recent_event_tag_total`` (operator counter:
      non-zero proves the grace path is doing real work).
    - Emits a single INFO log line.

    Returns the resolved sf_event_id, or None if unresolved.
    """
    # Fallback 0: parent lookup hit. Caller already did the indexed
    # SELECT; this helper consumes the result.
    if parent_meta_sf_event_id is not None:
        return int(parent_meta_sf_event_id)

    # Fallback 1: active-SF (v1.4.163).
    try:
        active_sf = getattr(state, "soft_flatten_event_id", None)
        if active_sf is not None and not storage.parent_order_exists(oid):
            sf_id = int(active_sf)
            logger.info(
                "sf_taker_fallback_fill_tagged "
                "oid=%s sf_event_id=%s — parent absent, "
                "stamping from state.soft_flatten_event_id",
                oid,
                sf_id,
            )
            return sf_id
    except Exception:
        logger.exception(
            "sf_taker_fallback_tag_check_failed oid=%s", oid
        )

    # Fallback 2: recent-event grace cache (v1.4.192).
    try:
        recent_sf = getattr(state, "sf_recent_event_id", None)
        if recent_sf is None:
            return None
        valid_until = float(
            getattr(state, "sf_recent_event_id_valid_until_mono", 0.0)
            or 0.0
        )
        if now_mono >= valid_until:
            return None
        if storage.parent_order_exists(oid):
            return None
        sf_id = int(recent_sf)
        try:
            state.sf_recent_event_tag_total = (
                int(getattr(state, "sf_recent_event_tag_total", 0)) + 1
            )
        except Exception:
            pass
        logger.info(
            "sf_recent_event_fallback_fill_tagged "
            "oid=%s sf_event_id=%s — parent absent, active SF "
            "cleared, stamping from grace cache (valid for %.1f more s)",
            oid,
            sf_id,
            valid_until - now_mono,
        )
        return sf_id
    except Exception:
        logger.exception(
            "sf_recent_event_fallback_tag_check_failed oid=%s", oid
        )
    return None


def resolve_tp_event_id_for_fill(
    *,
    parent_meta_tp_event_id: Optional[int],
    state: BotState,
    storage: Storage,
    oid: str,
    now_mono: float,
) -> Optional[int]:
    """Resolve the TP event id to stamp on a fill (or None).

    Mirror of ``resolve_sf_event_id_for_fill``. TP only places
    post-only orders (never bypasses ``OrderManager``), so the
    "active-TP / no parent" fallback is theoretical — a TP order
    always writes an orders row before its fill arrives. We still
    implement the same three-priority chain for safety, parity, and
    in case a future TP variant adds a non-OrderManager path:

    1. Parent-order lookup hit (canonical path).
    2. ``state.tp_event_id`` set AND parent absent — defensive
       fallback. Logged so the operator can see it triggered.
    3. Grace cache ``state.tp_recent_event_id`` set AND parent
       absent AND TTL valid — closes the race between TP exit
       clearing state and a late WS fill landing.

    Returns the resolved tp_event_id, or None.
    """
    if parent_meta_tp_event_id is not None:
        return int(parent_meta_tp_event_id)
    try:
        active_tp = getattr(state, "tp_event_id", None)
        if active_tp is not None and not storage.parent_order_exists(oid):
            tp_id = int(active_tp)
            logger.info(
                "tp_fallback_fill_tagged oid=%s tp_event_id=%s -- "
                "parent absent, stamping from state.tp_event_id",
                oid,
                tp_id,
            )
            return tp_id
    except Exception:
        logger.exception(
            "tp_active_fallback_tag_check_failed oid=%s", oid
        )
    try:
        recent_tp = getattr(state, "tp_recent_event_id", None)
        if recent_tp is None:
            return None
        valid_until = float(
            getattr(state, "tp_recent_event_id_valid_until_mono", 0.0)
            or 0.0
        )
        if now_mono >= valid_until:
            return None
        if storage.parent_order_exists(oid):
            return None
        tp_id = int(recent_tp)
        try:
            state.tp_recent_event_tag_total = (
                int(getattr(state, "tp_recent_event_tag_total", 0)) + 1
            )
        except Exception:
            pass
        logger.info(
            "tp_recent_event_fallback_fill_tagged oid=%s "
            "tp_event_id=%s -- parent absent, active TP cleared, "
            "stamping from grace cache (valid for %.1f more s)",
            oid,
            tp_id,
            valid_until - now_mono,
        )
        return tp_id
    except Exception:
        logger.exception(
            "tp_recent_event_fallback_tag_check_failed oid=%s", oid
        )
    return None


def _fill_ts_is_session_scoped(ts_fill: datetime, session_started_at_utc: datetime) -> bool:
    """True when the exchange-reported fill time is on or after this process's session start."""
    a = ts_fill if ts_fill.tzinfo else ts_fill.replace(tzinfo=timezone.utc)
    b = (
        session_started_at_utc
        if session_started_at_utc.tzinfo
        else session_started_at_utc.replace(tzinfo=timezone.utc)
    )
    return a >= b


def fill_row(f: Fill) -> dict:
    # Helper: serialize Optional[bool] → SQLite INTEGER (NULL / 0 / 1).
    def _b(v: Optional[bool]) -> Optional[int]:
        return None if v is None else (1 if v else 0)

    return {
        "fill_id": f.fill_id,
        "order_id_exchange": str(f.order_id_exchange) if f.order_id_exchange else None,
        "client_order_id": f.client_order_id,
        "ts_fill": f.ts_fill.isoformat(),
        "symbol": f.symbol,
        "side": f.side.value,
        "price": f.price,
        "size": f.size,
        "notional": f.notional,
        "fee": f.fee,
        "liquidity_flag": f.liquidity_flag,
        "mid_at_fill": f.mid_at_fill,
        "best_bid_at_fill": f.best_bid_at_fill,
        "best_ask_at_fill": f.best_ask_at_fill,
        "book_snapshot_quality": f.book_snapshot_quality,
        "book_reference_quality": f.book_reference_quality,
        "markout_1s_bps": f.markout_1s_bps,
        "markout_3s_bps": f.markout_3s_bps,
        "markout_5s_bps": f.markout_5s_bps,
        # v1.4.98 — extended diagnostic horizons. NULL on initial
        # insert; the markout resolution loop fills these via the
        # update path as each horizon's age threshold passes.
        "markout_15s_bps": f.markout_15s_bps,
        "markout_30s_bps": f.markout_30s_bps,
        "markout_60s_bps": f.markout_60s_bps,
        "markout_120s_bps": f.markout_120s_bps,
        # v1.4.100 ladder-observability F1 — copies from the parent
        # WO at fill-ingest time. Default 0 for fills whose parent
        # WO isn't found (defensive — e.g. REST-catchup fills whose
        # parent oid isn't in the local store anymore).
        "level_idx": int(getattr(f, "level_idx", 0) or 0),
        "quote_eligibility_state": f.quote_eligibility_state,
        "quote_eligibility_reason": f.quote_eligibility_reason,
        "book_age_seconds_at_fill": f.book_age_seconds_at_fill,
        "mid_return_100ms_bps_at_fill": f.mid_return_100ms_bps_at_fill,
        "mid_return_250ms_bps_at_fill": f.mid_return_250ms_bps_at_fill,
        "mid_return_500ms_bps_at_fill": f.mid_return_500ms_bps_at_fill,
        "fill_during_quote_cooldown": 1 if f.fill_during_quote_cooldown else 0,
        "effective_book_age_at_last_decision_ms": f.effective_book_age_at_last_decision_ms,
        "effective_book_age_at_fill_ms": f.effective_book_age_at_fill_ms,
        "private_ws_receive_to_state_apply_ms": f.private_ws_receive_to_state_apply_ms,
        "quote_cycle_to_first_transport_send_ms": f.quote_cycle_to_first_transport_send_ms,
        "quote_cycle_to_first_ack_ms": f.quote_cycle_to_first_ack_ms,
        "ack_to_private_ws_lifecycle_ms": f.ack_to_private_ws_lifecycle_ms,
        "soft_flatten_event_id": f.soft_flatten_event_id,
        # v1.5.33 — TP attribution mirror of SF (storage v42 column).
        "tp_event_id": f.tp_event_id,
        # v1.4.173 — Phase 4D.4 phase-ladder phase at fill time (NULL
        # when the ladder is disabled or no SF was active).
        "sf_force_phase": f.sf_force_phase,
        "target_half_spread_bps": f.target_half_spread_bps,
        "quote_aggressiveness": f.quote_aggressiveness,
        "quote_age_at_fill_ms": f.quote_age_at_fill_ms,
        "closed_pnl": f.closed_pnl,
        "basis_regime_sign": f.basis_regime_sign,
        "bid_size_top_at_fill": f.bid_size_top_at_fill,
        "ask_size_top_at_fill": f.ask_size_top_at_fill,
        # 2026-05-13 regime-observability Phase 1.
        "spread_bps_at_fill": f.spread_bps_at_fill,
        "microprice_at_fill": f.microprice_at_fill,
        "imbalance_top_at_fill": f.imbalance_top_at_fill,
        "inventory_qty_before_fill": f.inventory_qty_before_fill,
        "inventory_utilization_before_fill": f.inventory_utilization_before_fill,
        "toxicity_score_at_decision": f.toxicity_score_at_decision,
        "vol_estimate_at_decision": f.vol_estimate_at_decision,
        "active_sides_at_decision": f.active_sides_at_decision,
        "decision_reason_at_decision": f.decision_reason_at_decision,
        "binance_basis_ewma_at_decision": f.binance_basis_ewma_at_decision,
        "adaptive_widen_active_at_decision": _b(
            f.adaptive_widen_active_at_decision
        ),
        "post_fill_cooldown_active_bid_at_decision": _b(
            f.post_fill_cooldown_active_bid_at_decision
        ),
        "post_fill_cooldown_active_ask_at_decision": _b(
            f.post_fill_cooldown_active_ask_at_decision
        ),
        "at_touch_adverse_pause_bid_at_decision": _b(
            f.at_touch_adverse_pause_bid_at_decision
        ),
        "at_touch_adverse_pause_ask_at_decision": _b(
            f.at_touch_adverse_pause_ask_at_decision
        ),
        "quote_distance_to_touch_ticks_at_placement": (
            f.quote_distance_to_touch_ticks_at_placement
        ),
        "cancel_requested_before_fill": _b(f.cancel_requested_before_fill),
        "ms_cancel_request_to_fill": f.ms_cancel_request_to_fill,
        "expected_net_edge_bps_at_decision": (
            f.expected_net_edge_bps_at_decision
        ),
        # 2026-05-14 todo-027 Tier 2 propagation.
        "vol_trend_active_at_decision": _b(
            f.vol_trend_active_at_decision
        ),
        "post_swing_active_at_decision": _b(
            f.post_swing_active_at_decision
        ),
        "session_drawdown_tier_at_decision": (
            f.session_drawdown_tier_at_decision
        ),
        # v1.4.175 Phase 3F — reservation-alpha shifts.
        "ob_imbalance_shift_bps_at_decision": (
            f.ob_imbalance_shift_bps_at_decision
        ),
        "trend_drift_shift_bps_at_decision": (
            f.trend_drift_shift_bps_at_decision
        ),
        "flow_score_shift_bps_at_decision": (
            f.flow_score_shift_bps_at_decision
        ),
        "basis_deviation_shift_bps_at_decision": (
            f.basis_deviation_shift_bps_at_decision
        ),
        # v1.5.190 Phase 8A Option C — per-fill AS attribution. NULL on
        # legacy fills and on fills whose parent order had no breakdown
        # recorded (SF / TP / manual / hydrated).
        "as_base_half_spread_bps_at_decision": (
            f.as_base_half_spread_bps_at_decision
        ),
        # v1.5.204 Phase 4A — per-fill microprice widen attribution.
        # Same nullability semantics as above.
        "microprice_bid_widen_bps_at_decision": (
            f.microprice_bid_widen_bps_at_decision
        ),
        "microprice_ask_widen_bps_at_decision": (
            f.microprice_ask_widen_bps_at_decision
        ),
        # v1.5.306 audit §5 P0 #2 — per-fill AQC attribution. Float
        # passthrough for the aggression level; SQLite-bool encode for
        # the floor flag. Same nullability semantics as above.
        "aqc_aggression_level_at_decision": (
            f.aqc_aggression_level_at_decision
        ),
        "aqc_safety_floor_engaged_at_decision": _b(
            f.aqc_safety_floor_engaged_at_decision
        ),
        # 2026-05-13 Phase 4c: written NULL initially; populated by
        # PostFillExcursionWatcher via UPDATE 5s / 30s after the fill.
        "mae_5s_bps": f.mae_5s_bps,
        "mfe_5s_bps": f.mfe_5s_bps,
        "mae_30s_bps": f.mae_30s_bps,
        "mfe_30s_bps": f.mfe_30s_bps,
        # 2026-05-13 Phase 4c follow-up: NULL initially; populated by
        # PostFillTimeToFlatWatcher when position_qty crosses 0.
        "time_to_flat_seconds": f.time_to_flat_seconds,
    }


def _side_from_hl(s: str) -> Side:
    s = (s or "").upper()
    if s in ("B", "BUY", "LONG"):
        return Side.BUY
    return Side.SELL


def private_fill_event_to_hl_raw(ev: PrivateFillEvent) -> HLFillRaw:
    raw = dict(ev.raw)
    raw.setdefault("crossed", ev.crossed)
    return HLFillRaw(
        fill_id=ev.fill_id,
        oid=ev.oid,
        coin=ev.coin,
        side=_side_from_hl(ev.side),
        px=ev.px,
        sz=ev.sz,
        fee=ev.fee,
        time_ms=ev.time_ms,
        closed_pnl=ev.closed_pnl,
        raw=raw,
    )


def ingest_hl_fill_raw(
    *,
    state: BotState,
    storage: Storage | None,
    pnl: PnlTracker | None,
    symbol: str,
    fr: HLFillRaw,
    source: FillSource,
    private_inbound_timing: InboundPrivateTiming | None = None,
    shadow_update_position: bool = True,
    pre_inventory_qty_override: Optional[float] = None,
) -> bool:
    """
    Build Fill, dedupe, record_fill, markouts, PnL, storage.
    Returns True if this fill was new (not a duplicate fill_id).

    ``shadow_update_position`` (default True) controls whether
    ``state.record_fill`` applies the fill's signed delta to
    ``state.position.position_qty``. REST catch-up paths must pass
    ``False`` because the venue snapshot they just installed via
    ``apply_account_position_only`` already reflects every fill —
    a fresh shadow-add would double-count. See state.record_fill
    docstring for the bug story (Codex review 2026-05-08).

    ``pre_inventory_qty_override`` (Codex bug review 2026-05-13 MED #4):
    when supplied, use this value as the ``inventory_qty_before_fill``
    stamp instead of ``state.position.position_qty``. REST catch-up
    callers must compute this themselves because by the time they
    call ``ingest_hl_fill_raw``, ``apply_account_position_only`` has
    already overwritten ``state.position.position_qty`` with the
    POST-everything venue snapshot — reading it here would record
    post-fill inventory as "before_fill" and bias every per-regime
    inventory-bucket analysis on catch-up fills. The WS path leaves
    this ``None`` (default) and the pre-fill state is read inline
    because shadow_update_position=True only mutates position
    AFTER the read on line below.
    """
    if fr.coin != symbol:
        return False
    ts_fill = datetime.fromtimestamp(fr.time_ms / 1000.0, tz=timezone.utc)
    session_scoped = _fill_ts_is_session_scoped(ts_fill, state.session_started_at_utc)
    (
        mid_snap,
        bid_snap,
        ask_snap,
        bid_sz_snap,
        ask_sz_snap,
        ref_q,
        sq,
    ) = state.resolve_fill_book_reference(ts_fill)
    with state._lock:
        snap = dict(state.quote_eligibility_snapshot_dict)
        book_age_s = state.book_age_seconds
        eff_dec = state.last_effective_book_age_at_decision_ms
        q_to_tr = state.outbound_quote_cycle_to_first_transport_send_ms
        q_to_ack = state.outbound_quote_cycle_to_first_ack_ms
        ack_life = state.outbound_ack_to_private_ws_lifecycle_ms
        # N1: snapshot the basis-regime classifier's current sign at
        # fill time. ``state.basis_regime`` always exists; the sign
        # is -1 / 0 / +1 (0 means "below IC threshold — regime
        # signal not active"). Captured under the same lock as the
        # quote-eligibility snapshot to avoid torn reads.
        try:
            basis_regime_sign_at_fill = int(
                getattr(state.basis_regime, "last_regime_sign", 0) or 0
            )
        except Exception:
            basis_regime_sign_at_fill = None
    # 2026-05-12 codex-#4: stamp ``state_apply_mono`` at the start of
    # ingestion. ``InboundPrivateTiming`` is frozen with
    # ``state_apply_mono=0.0`` by default, and no WS handler writes
    # this field — so ``_ms(ws_recv_mono, state_apply_mono=0)``
    # returned None on every fill and the column was 0/N populated.
    # Replacing the frozen instance with one carrying a real
    # ``state_apply_mono`` (now-ish) makes ``_ms`` produce a real
    # value: roughly "WS receive → ingestion handler running",
    # which is a reasonable proxy for receive→apply latency.
    priv_rtsa = None
    if private_inbound_timing is not None:
        import time as _ts_apply_time
        from dataclasses import replace as _replace_timing
        try:
            private_inbound_timing = _replace_timing(
                private_inbound_timing,
                state_apply_mono=_ts_apply_clock.monotonic(),
            )
        except Exception:
            # Frozen dataclass replace should never fail; defensive
            # fallback keeps ingestion working even if it did.
            pass
        dm = private_timing_derived_ms(private_inbound_timing)
        v = dm.get("private_ws_receive_to_state_apply_ms")
        if isinstance(v, (int, float)):
            priv_rtsa = float(v)
    eff_fill_ms = None
    if book_age_s is not None:
        eff_fill_ms = float(book_age_s) * 1000.0
    # v1.4.37 (Codex #5): single combined parent-order metadata lookup.
    # Pre-v1.4.37 the hot path ran THREE serialised indexed lookups
    # against the same ``orders.order_id_exchange`` row, each acquiring
    # ``storage._lock`` independently. Bursty fill windows showed
    # ~60-100 µs of avoidable per-fill latency + lock contention against
    # the heartbeat / live-stats readers. ``order_metadata_for_fill_ingest``
    # collapses them into one SELECT under one lock acquisition.
    # Returns a flat dict with ``soft_flatten_event_id``,
    # ``target_half_spread_bps``, ``quote_aggressiveness``, and every
    # decision-state column. All values are None when the parent order
    # isn't in the local DB — same fallback as the legacy three-method
    # path. Plan refs: plans/20260507-sf-frontend.md (SF),
    # todo-005/006 (quote quality), regime-observability Phase 1
    # (decision-state).
    sf_event_id: Optional[int] = None
    # v1.5.33 — TP event id stamped at fill ingest. Mirror of SF
    # resolution chain (parent-meta hit → active fallback → grace
    # cache). NULL on non-TP fills.
    tp_event_id: Optional[int] = None
    # v1.4.173 (Phase 4D.4) — SF phase at fill time. Stamped from
    # ``state.sf_phase_ladder_phase`` when SF is active (matches the
    # ``soft_flatten_event_id`` stamping pattern). NULL otherwise.
    sf_force_phase_at_fill: Optional[int] = None
    target_half_spread_bps: Optional[float] = None
    quote_aggressiveness: Optional[str] = None
    # v1.4.100 ladder-observability F1 — parent-order rung index.
    # Default 0 ("single-rung-equivalent") when no parent order is
    # found, matching the column default and the pre-v1.4.100
    # historical bucketing.
    parent_level_idx: int = 0
    decision_state: dict = {}
    if storage is not None and fr.oid is not None:
        try:
            meta = storage.order_metadata_for_fill_ingest(str(fr.oid))
            # v1.4.192 — SF event-id resolution chain extracted into
            # ``resolve_sf_event_id_for_fill`` (testable + observable).
            # Three priority levels: parent-order hit → active-SF
            # fallback (v1.4.163) → recent-event grace fallback
            # (v1.4.192). See helper docstring for the bug history.
            sf_event_id = resolve_sf_event_id_for_fill(
                parent_meta_sf_event_id=meta.get("soft_flatten_event_id"),
                state=state,
                storage=storage,
                oid=str(fr.oid),
                now_mono=_clock.monotonic(),
            )
            # v1.5.33 — TP attribution mirror.
            tp_event_id = resolve_tp_event_id_for_fill(
                parent_meta_tp_event_id=meta.get("tp_event_id"),
                state=state,
                storage=storage,
                oid=str(fr.oid),
                now_mono=_clock.monotonic(),
            )
            # v1.4.173 (Phase 4D.4) — SF phase-ladder phase at fill
            # time. Stamp from ``state.sf_phase_ladder_phase`` whenever
            # the fill is attributed to an SF episode (whether by
            # parent-order lookup OR by the taker-fallback path above).
            # Phase is a per-episode integer that monotonically
            # advances; reading "now" gives the phase the fill landed
            # in, modulo private-WS latency. Pre-v1.4.172 episodes ran
            # the legacy 2-phase post-only worker which never wrote
            # ``sf_phase_ladder_phase``, so the field stays 0 (the
            # default) and we leave the column NULL to mark
            # "ladder not active". Active-ladder episodes always have
            # at least one tick of dispatcher run before the first fill
            # so the field is meaningfully populated.
            if sf_event_id is not None:
                try:
                    phase_raw = getattr(
                        state, "sf_phase_ladder_phase", None
                    )
                    ladder_enabled = bool(
                        getattr(
                            getattr(state, "settings", None),
                            "sf_phase_ladder_enabled",
                            False,
                        )
                    )
                    if (
                        ladder_enabled
                        and phase_raw is not None
                    ):
                        sf_force_phase_at_fill = int(phase_raw)
                except Exception:
                    logger.exception(
                        "sf_force_phase_stamp_failed oid=%s", fr.oid
                    )
            target_half_spread_bps = meta.get("target_half_spread_bps")
            quote_aggressiveness = meta.get("quote_aggressiveness")
            # v1.4.100 F1 — read parent rung index; defensive cast.
            _lvl_raw = meta.get("level_idx")
            if _lvl_raw is not None:
                try:
                    parent_level_idx = int(_lvl_raw)
                except (TypeError, ValueError):
                    parent_level_idx = 0
            # Decision state is the rest of the dict, minus the four
            # non-decision keys (now including level_idx). Mirror the
            # legacy contract: callers downstream pull individual keys
            # via ``decision_state.get(...)``.
            decision_state = {
                k: v
                for k, v in meta.items()
                if k
                not in (
                    "soft_flatten_event_id",
                    "tp_event_id",
                    "target_half_spread_bps",
                    "quote_aggressiveness",
                    "level_idx",
                )
            }
        except Exception:
            logger.exception(
                "order_metadata_for_fill_ingest_failed oid=%s", fr.oid
            )
    # SQLite bool storage convention: NULL / 0 / 1. Normalize back to
    # Optional[bool] for the Fill dataclass.
    def _opt_bool(v):
        if v is None:
            return None
        try:
            return bool(int(v))
        except (TypeError, ValueError):
            return None

    def _opt_float(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    # Cancel-race diagnostics. ``ts_cancel_requested`` is an ISO
    # timestamp on the orders row (set by the cancel-issue path).
    # If present and earlier than ts_fill, the cancel was already in
    # flight when the fill arrived.
    _ts_cancel_req_iso = decision_state.get("ts_cancel_requested")
    cancel_requested_before_fill: Optional[bool] = None
    ms_cancel_request_to_fill: Optional[float] = None
    if _ts_cancel_req_iso:
        try:
            ts_cancel_req = datetime.fromisoformat(_ts_cancel_req_iso)
            if ts_cancel_req.tzinfo is None:
                ts_cancel_req = ts_cancel_req.replace(tzinfo=timezone.utc)
            cancel_requested_before_fill = ts_cancel_req < ts_fill
            ms_cancel_request_to_fill = max(
                0.0, (ts_fill - ts_cancel_req).total_seconds() * 1000.0
            )
        except (ValueError, TypeError):
            cancel_requested_before_fill = None
            ms_cancel_request_to_fill = None

    # Trivial derivations at fill ingestion. Data already in scope:
    # ``mid_snap``, ``bid_snap``, ``ask_snap``, ``bid_sz_snap``,
    # ``ask_sz_snap``, ``state.position.position_qty``,
    # ``settings.max_abs_position`` (via state's settings link).
    spread_bps_at_fill: Optional[float] = None
    microprice_at_fill: Optional[float] = None
    imbalance_top_at_fill: Optional[float] = None
    if (
        mid_snap is not None
        and bid_snap is not None
        and ask_snap is not None
        and float(mid_snap) > 0
    ):
        try:
            spread_bps_at_fill = (
                (float(ask_snap) - float(bid_snap)) / float(mid_snap) * 10_000.0
            )
        except (TypeError, ValueError, ZeroDivisionError):
            spread_bps_at_fill = None
    if (
        bid_sz_snap is not None
        and ask_sz_snap is not None
        and bid_snap is not None
        and ask_snap is not None
    ):
        try:
            bsz = float(bid_sz_snap)
            asz = float(ask_sz_snap)
            denom = bsz + asz
            if denom > 0:
                microprice_at_fill = (
                    bsz * float(ask_snap) + asz * float(bid_snap)
                ) / denom
                imbalance_top_at_fill = (bsz - asz) / denom
        except (TypeError, ValueError, ZeroDivisionError):
            microprice_at_fill = None
            imbalance_top_at_fill = None

    # Inventory-before-fill: there are TWO ways this field is sourced.
    #
    # WS path (shadow_update_position=True, override is None):
    #   ``state.position.position_qty`` reflects PRE-fill state because
    #   the shadow-add inside ``record_fill`` happens AFTER this read.
    #
    # REST catch-up path (shadow_update_position=False, override
    # supplied by caller — Codex bug review 2026-05-13 MED #4):
    #   ``state.position.position_qty`` has ALREADY been overwritten by
    #   ``apply_account_position_only`` with the post-everything venue
    #   snapshot. Reading it here would record post-fill inventory as
    #   "before_fill" and silently bias every per-regime inventory-
    #   bucket analysis on catch-up fills (which is exactly the path
    #   meant to repair missed private-WS fills, so the bias hits the
    #   case the analysis is supposed to investigate). The caller
    #   computes the genuine pre-fill qty by walking the catch-up
    #   batch backwards (subtracting each signed delta) and supplies it
    #   via ``pre_inventory_qty_override``.
    inventory_qty_before_fill: Optional[float] = None
    inventory_utilization_before_fill: Optional[float] = None
    try:
        if (
            pre_inventory_qty_override is not None
            and math.isfinite(pre_inventory_qty_override)
        ):
            inventory_qty_before_fill = float(pre_inventory_qty_override)
        else:
            inventory_qty_before_fill = float(state.position.position_qty)
        max_abs = float(
            getattr(state.settings, "max_abs_position", 0.0) or 0.0
        )
        if max_abs > 0:
            inventory_utilization_before_fill = (
                abs(inventory_qty_before_fill) / max_abs
            )
    except Exception:
        # Defensive: position may be None on first fills before state
        # is fully initialized. Leave both fields None.
        pass

    f = Fill(
        fill_id=fr.fill_id,
        order_id_exchange=fr.oid,
        client_order_id=None,
        ts_fill=ts_fill,
        symbol=symbol,
        side=fr.side,
        price=fr.px,
        size=fr.sz,
        notional=abs(fr.px * fr.sz),
        fee=fr.fee,
        liquidity_flag="crossed" if fr.raw.get("crossed") else "resting",
        mid_at_fill=mid_snap,
        best_bid_at_fill=bid_snap,
        best_ask_at_fill=ask_snap,
        book_snapshot_quality=sq,
        book_reference_quality=ref_q,
        quote_eligibility_state=snap.get("quote_eligibility_state"),
        quote_eligibility_reason=(
            (snap.get("quote_eligibility_reason") or "")[:2000]
            if snap.get("quote_eligibility_reason")
            else None
        ),
        book_age_seconds_at_fill=book_age_s,
        mid_return_100ms_bps_at_fill=snap.get("mid_return_100ms_bps"),
        mid_return_250ms_bps_at_fill=snap.get("mid_return_250ms_bps"),
        mid_return_500ms_bps_at_fill=snap.get("mid_return_500ms_bps"),
        fill_during_quote_cooldown=bool(snap.get("in_cooldown")),
        effective_book_age_at_last_decision_ms=eff_dec,
        effective_book_age_at_fill_ms=eff_fill_ms,
        private_ws_receive_to_state_apply_ms=priv_rtsa,
        quote_cycle_to_first_transport_send_ms=q_to_tr,
        quote_cycle_to_first_ack_ms=q_to_ack,
        ack_to_private_ws_lifecycle_ms=ack_life,
        closed_pnl=fr.closed_pnl,
        soft_flatten_event_id=sf_event_id,
        tp_event_id=tp_event_id,
        target_half_spread_bps=target_half_spread_bps,
        quote_aggressiveness=quote_aggressiveness,
        # v1.4.100 ladder-observability F1 — stamp rung index from
        # the parent order row.
        level_idx=parent_level_idx,
        basis_regime_sign=basis_regime_sign_at_fill,
        bid_size_top_at_fill=bid_sz_snap,
        ask_size_top_at_fill=ask_sz_snap,
        # 2026-05-13 regime-observability Phase 1.
        spread_bps_at_fill=spread_bps_at_fill,
        microprice_at_fill=microprice_at_fill,
        imbalance_top_at_fill=imbalance_top_at_fill,
        inventory_qty_before_fill=inventory_qty_before_fill,
        inventory_utilization_before_fill=inventory_utilization_before_fill,
        toxicity_score_at_decision=_opt_float(
            decision_state.get("toxicity_score_at_decision")
        ),
        vol_estimate_at_decision=_opt_float(
            decision_state.get("vol_estimate_at_decision")
        ),
        active_sides_at_decision=decision_state.get(
            "active_sides_at_decision"
        ),
        decision_reason_at_decision=decision_state.get(
            "decision_reason_at_decision"
        ),
        binance_basis_ewma_at_decision=_opt_float(
            decision_state.get("binance_basis_ewma_at_decision")
        ),
        adaptive_widen_active_at_decision=_opt_bool(
            decision_state.get("adaptive_widen_active_at_decision")
        ),
        post_fill_cooldown_active_bid_at_decision=_opt_bool(
            decision_state.get("post_fill_cooldown_active_bid_at_decision")
        ),
        post_fill_cooldown_active_ask_at_decision=_opt_bool(
            decision_state.get("post_fill_cooldown_active_ask_at_decision")
        ),
        at_touch_adverse_pause_bid_at_decision=_opt_bool(
            decision_state.get("at_touch_adverse_pause_bid_at_decision")
        ),
        at_touch_adverse_pause_ask_at_decision=_opt_bool(
            decision_state.get("at_touch_adverse_pause_ask_at_decision")
        ),
        quote_distance_to_touch_ticks_at_placement=_opt_float(
            decision_state.get("quote_distance_to_touch_ticks_at_placement")
        ),
        cancel_requested_before_fill=cancel_requested_before_fill,
        ms_cancel_request_to_fill=ms_cancel_request_to_fill,
        # 2026-05-13 Phase 4a propagation.
        expected_net_edge_bps_at_decision=_opt_float(
            decision_state.get("expected_net_edge_bps_at_decision")
        ),
        # 2026-05-14 todo-027 Tier 2 propagation.
        vol_trend_active_at_decision=_opt_bool(
            decision_state.get("vol_trend_active_at_decision")
        ),
        post_swing_active_at_decision=_opt_bool(
            decision_state.get("post_swing_active_at_decision")
        ),
        session_drawdown_tier_at_decision=decision_state.get(
            "session_drawdown_tier_at_decision"
        ),
        # v1.4.175 Phase 3F — reservation-alpha shifts at decision
        # time, propagated from the parent order row. Consumed by
        # ``tools/postmortem/sections/reservation_alpha_attribution.py``.
        ob_imbalance_shift_bps_at_decision=_opt_float(
            decision_state.get("ob_imbalance_shift_bps_at_decision")
        ),
        trend_drift_shift_bps_at_decision=_opt_float(
            decision_state.get("trend_drift_shift_bps_at_decision")
        ),
        flow_score_shift_bps_at_decision=_opt_float(
            decision_state.get("flow_score_shift_bps_at_decision")
        ),
        basis_deviation_shift_bps_at_decision=_opt_float(
            decision_state.get("basis_deviation_shift_bps_at_decision")
        ),
        # v1.5.190 Phase 8A Option C — per-fill AS-base half-spread.
        # Propagated from parent-order row via decision_state. NULL on
        # legacy fills and on fills whose parent had no breakdown.
        as_base_half_spread_bps_at_decision=_opt_float(
            decision_state.get("as_base_half_spread_bps_at_decision")
        ),
        # v1.5.204 Phase 4A — per-fill microprice widen attribution.
        # Propagated from the parent-order row via decision_state.
        microprice_bid_widen_bps_at_decision=_opt_float(
            decision_state.get("microprice_bid_widen_bps_at_decision")
        ),
        microprice_ask_widen_bps_at_decision=_opt_float(
            decision_state.get("microprice_ask_widen_bps_at_decision")
        ),
        # v1.5.306 audit §5 P0 #2 — per-fill AQC attribution. Propagated
        # from the parent-order row via decision_state. NULL on legacy
        # fills + fills whose parent was placed while AQC was disabled.
        aqc_aggression_level_at_decision=_opt_float(
            decision_state.get("aqc_aggression_level_at_decision")
        ),
        aqc_safety_floor_engaged_at_decision=_opt_bool(
            decision_state.get("aqc_safety_floor_engaged_at_decision")
        ),
    )
    if not state.record_fill(
        f,
        session_scoped=session_scoped,
        shadow_update_position=shadow_update_position,
    ):
        logger.debug(
            "fill_ingest_duplicate_skipped fill_id=%s source=%s",
            fr.fill_id,
            source,
        )
        return False
    # 2026-05-13 regime-observability Phase 4c: hand the fill to the
    # post-fill excursion watcher (if enabled). Safe-by-default: no-op
    # if the watcher wasn't installed (feature flag off) or if
    # mid_at_fill is missing. The track_fill call is a single dict
    # write under the watcher's own lock — does not block the strategy
    # thread.
    _watcher = getattr(state, "post_fill_excursion_watcher", None)
    if _watcher is not None and f.mid_at_fill is not None:
        try:
            _watcher.track_fill(
                f.fill_id, f.side.value, float(f.mid_at_fill)
            )
        except Exception:
            logger.exception(
                "post_fill_excursion_watcher_track_failed fill_id=%s",
                f.fill_id,
            )
    # 2026-05-13 Phase 4c follow-up: time-to-flat watcher.
    # Same safe-by-default pattern. Captures position_qty_before_fill
    # for diagnostic context (not used in the flat-crossing check;
    # the watcher triggers on absolute zero, not on undo-the-delta).
    _ttf_watcher = getattr(state, "post_fill_time_to_flat_watcher", None)
    if _ttf_watcher is not None:
        try:
            _ttf_watcher.track_fill(
                f.fill_id, inventory_qty_before_fill
            )
        except Exception:
            logger.exception(
                "post_fill_time_to_flat_watcher_track_failed fill_id=%s",
                f.fill_id,
            )
    logger.info(
        "fill_ingest applied fill_id=%s oid=%s source=%s symbol=%s session_scoped=%s",
        fr.fill_id,
        fr.oid,
        source,
        symbol,
        session_scoped,
    )
    if session_scoped:
        register_fill_for_delayed_markouts(state, f)
    if pnl and session_scoped:
        pnl.on_fill(f, closed_pnl_component=fr.closed_pnl)
    if session_scoped:
        state.bump_operator_metrics_on_session_fill(f, fr.closed_pnl)
    # todo-006 fill-age bucketing: record this fill into the per-quote-age
    # bucket aggregator so the live_stats payload (and the frontend Bot
    # Stats panel) can show the toxic-vs-passive distribution. Cheap:
    # one dict lookup + one append. The aggregator holds a *reference*
    # to the Fill so delayed-markout values (1s/3s/5s) populated in
    # place by app/markout.py are picked up automatically when the
    # bucket aggregate is read. See app/fill_bucket_metrics.py.
    if session_scoped:
        try:
            # ``note_fill`` returns the quote-age (ms) it just bucketed;
            # stamp it on the Fill so the DB row carries the same value
            # the in-memory aggregator used. None for unknown-age fills
            # (ack evicted / never recorded). Non-session-scoped REST
            # catchup fills don't reach this branch and stay None — the
            # ack cache only covers the current process's session.
            f.quote_age_at_fill_ms = state.fill_buckets.note_fill(f)
        except Exception:
            logger.exception("fill_buckets_note_fill_failed")
    if storage:
        storage.insert_fill_row(fill_row(f))
    return True
