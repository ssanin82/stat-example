from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
import logging
from dataclasses import dataclass, replace
import math
import queue
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from app.clock import Clock
    from app.startup_cleanup import CleanBookResult

from app.config import Settings
from app.enums import (
    ActiveSides,
    BotStatus,
    DesyncPhase,
    EventSeverity,
    OrderStatus,
    RiskAction,
    RiskExecState,
    Side,
)
from app.exchange.hyperliquid_precision import (
    HL_PERP_LIMIT_PRICE_PIPELINE_ID,
    HL_PERP_MAX_SIG_FIGS,
    normalize_order_pair,
    rejection_is_below_min_notional_usd,
    wire_format_preview_limit_px,
)
from app.exchange.symbol_spec import SymbolSpec
from app.exchange.base import OpenOrderRaw, PerpExchangeAdapter
# Back-compat re-exports: external scripts and existing tests still import
# these names from ``app.execution``. The implementations live next to the
# Hyperliquid adapter (``app.exchange.hyperliquid_responses``) so the bot
# core no longer owns venue-specific wire-format logic. New code should
# call the corresponding ``PerpExchangeAdapter`` methods instead.
from app.exchange.hyperliquid_responses import (
    interpret_hl_cancel_response,
    interpret_hl_order_status_response,
    interpret_hl_place_order_response,
    make_deterministic_cloid_hex,
)
# Keep legacy response interpreter symbols exported from this module for
# external scripts/tests, even though runtime paths now delegate via adapter.
_LEGACY_HL_RESPONSE_REEXPORTS = (
    interpret_hl_cancel_response,
    interpret_hl_order_status_response,
    interpret_hl_place_order_response,
)
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
    PrivateWsConnectionEvent,
    PrivateWsConnectionKind,
)
from app.fill_ingestion import ingest_hl_fill_raw, private_fill_event_to_hl_raw
from app.inbound_timing import private_timing_derived_ms
from app.models import BestBidAsk, QuoteDecision, ToxicitySnapshot, WorkingOrder
from app.outbound_dispatch import CancelTransportIntent, OutboundDispatchCoordinator, PlaceTransportIntent
from app.outbound_trace import OutboundActionTrace
from app.quote_engine import FinalQuoteOrder, QuoteBuildContext, QuoteEngine
from app.reconciler import (
    AmendAction,
    CancelAction,
    DesiredLadderState,
    DesiredOrderState,
    NoOpAction,
    PlaceAction,
    apply_reducing_side_bypass,
    apply_side_suppression,
    reconcile,
)
from app.pnl import PnlTracker
from app.storage import Storage
from app.utils.logfmt import log_extra
from app.utils.math import clip
from app.utils.time import seconds_since, utc_now, utc_now_iso

logger = logging.getLogger(__name__)

_MAX_PRIVATE_EVENTS_PER_TICK = 2000
# Fast-start: cap processing of deferred private snapshot fills after RUNNING.
_FAST_START_DEFERRED_SNAPSHOT_FILLS_PER_TICK = 200
# v1.4.68 Phase 1A: per-cloid WS-event buffer bounds.
# Per-cloid max: typical race produces 1-2 events before the place
# response binds the OID; 16 gives ample headroom for genuinely-slow
# place responses while bounding memory.
_PENDING_WS_BUFFER_PER_CLOID_MAX = 16
# Total buffer cap across all cloids — protects against pathological
# floods of unmatchable events (would indicate a venue or cloid format
# regression).
_PENDING_WS_BUFFER_TOTAL_MAX = 256
# Sweep stale entries older than this. The place response and WS
# ``live`` arrive within ms of each other on a healthy connection; 5 s
# is generous and catches genuinely-orphaned events.
_PENDING_WS_BUFFER_TTL_SECONDS = 5.0
# Rate-limit the sweeper to once per second so we don't iterate the
# buffer on every WS event.
_PENDING_WS_BUFFER_SWEEP_INTERVAL_SECONDS = 1.0
# v1.4.75 Phase 2B: reaper rate-limit. The reaper iterates
# ``state.all_working_orders()`` (a small list, ≤8 entries typically)
# but the per-WO transition + persist is non-trivial. Bound to once
# per second so the hot path stays cheap even if many WOs need
# reaping.
_REAPER_MIN_INTERVAL_SECONDS = 1.0
# v1.4.78 hot-path: ring-buffer sample interval for noisy no-op
# decisions. With 1-in-20, a snapshot like v1.4.77 (32k noop:acked
# decisions in 10 min) generates ~1600 ring-buffer entries instead
# of 32000 — same statistical signal, ~20× fewer dict allocations.
# Counter increments are NOT sampled (cumulative accuracy preserved).
_NOOP_TRACE_SAMPLE_INTERVAL = 20
# Cancel-storm protection: suppress a repeated orphan cancel for the same (symbol, oid)
# within this window after a successful or benign-missing dispatch. The exchange may still
# be processing the first cancel when the next reconcile cycle runs; the dedup cache
# prevents spamming HTTP cancel calls during that settlement window. Failed dispatches are
# intentionally NOT cached so operators retain retry semantics.
_ORPHAN_CANCEL_DEDUP_TTL_SECONDS = 10.0
# Hard cap for the dedup cache to keep memory bounded even under pathological spew.
_ORPHAN_CANCEL_DEDUP_MAX_ENTRIES = 512

# Max length of exchange error text embedded in quarantine strike keys (exact match per intent).
_REJECTION_QUARANTINE_REASON_KEY_LEN = 500

# Cancel-pending retry escalation: number of oid-based cancel retries we will
# issue before switching to cancel-by-cloid (with GRVT's ``time_to_live_ms``).
# GRVT has been observed to return ``{"result": {"ack": true}}`` on oid
# cancels yet leave the order alive on the book (``open_orders`` continues
# to return it). The cloid-cancel path goes through a different handler and
# gives the matching engine a different resolution signal.
#
# Lowered from 2 to 1 based on evidence from logs.1776583271526.json
# (2026-04-19 07:03–07:07): two sides went stuck in CANCEL_PENDING for
# ~2.5 minutes. First two retry attempts used oid-based cancel and
# received 200 OK but no terminal WS event. The THIRD attempt escalated
# to cloid-based cancel; GRVT delivered the CLIENT_CANCEL terminal WS
# event within ~30 ms. Escalating at attempt 2 (not 3) cuts stuck time
# by ~50 s per episode and reduces the side-unresolved latch window
# proportionally.
#
# Why not 0 (escalate on every retry)? One oid-based retry first keeps
# the default path simple when GRVT is just briefly delayed on an
# otherwise valid oid cancel. If that first retry succeeds, we keep the
# simpler bookkeeping. If not, we fall through to cloid + TTL.
_CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS = 1

# 1.3.117: per-profile escape hatch for the cloid escalation above. On
# OKX colo (alibaba-hk Stage 3 verification, 2026-05-16), the WS cancel
# frame for cloid-based cancellation requires ``instIdCode`` (numeric)
# instead of ``instId`` (string) — see ``plans/_DONE/sbe.md`` §2. Until
# the WS layer fetches + uses ``instIdCode``, the cloid-escalation path
# emits OKX sCode 50014 ``Parameter instIdCode can not be empty`` and
# the retry loop keeps firing until ``execution_errors`` self-kills
# the bot. Setting this to False on OKX profiles keeps the cancel
# retry on the ordId path indefinitely, which works on colo WS
# because OKX matches by ``ordId`` directly without needing the
# instId-to-instIdCode lookup. Safe on OKX because Phase 1a's sync
# ordId binding guarantees the ordId is bound by the time the cancel
# retry fires (already verified Snapshot B + heartbeat
# ``cancel_deferred_until_ack_total`` near zero).
# Setting type: ``cancel_pending_cloid_escalation_enabled``.

# Back-compat for tests importing private intent names.
_PlaceTransportIntent = PlaceTransportIntent
_CancelTransportIntent = CancelTransportIntent


def is_post_only_immediate_match_rejection(reason: str) -> bool:
    """
    True when the exchange rejected a post-only order because it would cross / match immediately.

    Treated as a benign quote-timing issue (not transport failure, not execution_errors budget).
    """
    r = (reason or "").strip().lower()
    if not r:
        return False
    if "post only" not in r and "post-only" not in r:
        return False
    if "immediately matched" in r:
        return True
    if "would cross" in r or "would have crossed" in r:
        return True
    return False


def exchange_validation_failure_class(reason: str) -> Optional[str]:
    """Map Hyperliquid validation error text to a stable quarantine bucket, or None."""
    if not reason:
        return None
    r = reason.lower()
    if "divisible" in r and "tick" in r:
        return "hl_price_tick_divisibility"
    if ("significant" in r and "fig" in r) or "sig fig" in r:
        return "hl_price_sigfigs"
    return None


def _normalize_rejection_reason_key(reason: str) -> str:
    return (reason or "").strip()[:_REJECTION_QUARANTINE_REASON_KEY_LEN]


def _is_benign_place_outcome(outcome: str, reason: Optional[str]) -> bool:
    """Mirror of the frontend's ``_isBenignPlaceDetail`` heuristic
    (1.4.6) — classify rejected place responses as operationally
    benign (post-only would cross, order-already-gone variants) vs
    attention-worthy (auth, rate-limit, insufficient margin, etc.).
    Used by the sticky-rejection summary recorder so the dashboard
    can colour-code rows without re-classifying client-side.
    """
    if outcome == "accepted":
        return True
    r = (reason or "").lower()
    if not r:
        return False
    # post-only would cross is the canonical benign case — quote-timing
    # issue, recovers on next cycle.
    if "post_only_would_cross" in r or "would cross" in r or "would have crossed" in r:
        return True
    if "immediately matched" in r:
        return True
    # OKX-style row codes for "order already gone" — benign on cancel
    # path; on place path these shouldn't fire but treat as benign.
    for code in ("51400", "51401", "51402", "51503"):
        if f"okx_row_{code}" in r:
            return True
    return False


def _is_benign_cancel_outcome(outcome: str, reason: Optional[str]) -> bool:
    """Mirror of the frontend's ``_isBenignCancelDetail`` (1.4.6).
    ``benign_missing`` (51402: order filled before cancel) and
    ``unexpected_gone`` (51400/51401/51503: order gone for non-fill
    reason) are both benign as operational outcomes — the local WO
    reconciles cleanly. The attention-worthy class is everything
    else (transport, exchange refusal for invalid input)."""
    if outcome in ("success", "benign_missing"):
        return True
    if outcome == "unexpected_gone":
        return True
    return False
_ORDER_WS_TS_EPS_MS = 1


def cloids_match(a: str, b: str) -> bool:
    aa = a.strip().lower()
    bb = b.strip().lower()
    if not aa.startswith("0x"):
        aa = "0x" + aa
    if not bb.startswith("0x"):
        bb = "0x" + bb
    return aa == bb


def resting_clip_size_for_position_headroom(wo: Optional[WorkingOrder]) -> float:
    """
    Extra exposure beyond the single managed slot per side.

    ACKED/PARTIAL rows are excluded: the execution layer never stacks a second passive
    on the same side without clearing the slot, so counting them would double-penalize
    cancel/replace sizing. SENT and CANCEL_PENDING still count — the exchange may hold
    size we have not reconciled or fully cleared yet.
    """
    if wo is None:
        return 0.0
    if wo.status not in (OrderStatus.SENT, OrderStatus.CANCEL_PENDING):
        return 0.0
    return max(0.0, float(wo.size))


def snapshot_hl_place_response(resp: Any) -> dict[str, Any]:
    """Shallow, log-safe summary (trade data only; omit any signature-like blobs if present)."""
    out: dict[str, Any] = {}
    if not isinstance(resp, dict):
        out["response_shape"] = "non_dict"
        return out
    out["top_level_status"] = resp.get("status")
    r = resp.get("response")
    if not isinstance(r, dict):
        out["response_type"] = None
        out["first_status_kind"] = "n/a"
        return out
    out["response_type"] = r.get("type")
    d = r.get("data")
    if not isinstance(d, dict):
        out["first_status_kind"] = "no_data"
        return out
    if isinstance(d.get("error"), str) and d["error"]:
        out["batch_error"] = str(d["error"])[:300]
    st = d.get("statuses")
    if not isinstance(st, list) or not st:
        out["first_status_kind"] = "no_statuses"
        return out
    s0 = st[0]
    if isinstance(s0, str):
        out["first_status_kind"] = "string_token"
        out["first_status_value"] = s0
        return out
    if not isinstance(s0, dict):
        out["first_status_kind"] = type(s0).__name__
        return out
    if "error" in s0:
        out["first_status_kind"] = "error"
        out["parsed_substatus"] = str(s0.get("error"))[:400]
    elif "resting" in s0:
        out["first_status_kind"] = "resting"
        rest = s0.get("resting")
        if isinstance(rest, dict):
            out["parsed_substatus"] = f"oid={rest.get('oid')}"
        else:
            out["parsed_substatus"] = str(rest)[:120]
    elif "filled" in s0:
        out["first_status_kind"] = "filled"
        f = s0.get("filled")
        if isinstance(f, dict):
            out["parsed_substatus"] = (
                f"oid={f.get('oid')} totalSz={f.get('totalSz')} avgPx={f.get('avgPx')}"
            )
        else:
            out["parsed_substatus"] = str(f)[:120]
    else:
        out["first_status_kind"] = "unknown"
        out["status_keys"] = ",".join(sorted(s0.keys()))[:200]
    return out


def summarize_hl_place_response_json(resp: Any, max_len: int = 1800) -> str:
    if not isinstance(resp, dict):
        return type(resp).__name__
    try:
        text = json.dumps(resp, default=str, separators=(",", ":"))
    except Exception:
        text = repr(resp)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text

# Conservative cushion so we do not lean on exchange min size / rounding.
_POSITION_EPS = 1e-10


def transition(wo: WorkingOrder, status: OrderStatus, reason: Optional[str] = None) -> None:
    # v1.5.143 (Phase 1c migration of this free function) — was
    # using ``datetime.now(timezone.utc)`` directly, bypassing the
    # bot's clock abstraction. In live trading the result was
    # identical; in REPLAY mode it meant ts_sent / ts_ack /
    # ts_closed on every WorkingOrder got wall-clock timestamps
    # (the real moment the bot's code ran) instead of the
    # simulated tick time. The bot_db extract preserved those
    # wall-clock values, which then broke the backtesting viewer's
    # order-events overlay (events plotted 2 days away from the
    # rest of the chart). See the 2026-05-25 incident in
    # ``plans/_DONE/`` for the full diagnosis.
    #
    # Phase 1c shipped the ``app.clock`` module-level proxy +
    # ``app.utils.time.utc_now()`` wrapper a while ago; this call
    # site was the last hold-out using stdlib directly. Now: the
    # wrapper routes through the module clock, which the replay
    # driver swaps to a ReplayClock at startup.
    from app.utils.time import utc_now
    wo.status = status
    if reason:
        wo.cancel_reason = reason
    now = utc_now()
    if status == OrderStatus.SENT:
        wo.ts_sent = now
    if status == OrderStatus.ACKED:
        wo.ts_ack = now
    if status in (OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED):
        wo.ts_closed = now


def _utc_aware(dt: datetime) -> datetime:
    """Promote a naïve datetime to UTC-aware so it can be compared
    against current UTC time without TypeError.

    Historical paths in this codebase stored ts_ack / ts_sent as both
    naïve and aware values depending on the venue adapter. New writes
    are always aware; older paths produce naïve. Treat naïve as UTC
    (the bot's convention everywhere) so the orphan-fill guard works
    against both.
    """
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def order_row(wo: WorkingOrder) -> dict:
    return {
        "order_id_local": wo.order_id_local,
        "order_id_exchange": str(wo.order_id_exchange) if wo.order_id_exchange else None,
        "client_order_id": wo.client_order_id,
        # v1.4.100 ladder-observability F1 — persist the rung index
        # so fill-ingestion can copy it onto the fill row at ingest
        # time, enabling per-rung postmortem decomposition.
        "level_idx": int(getattr(wo, "level_idx", 0) or 0),
        "ts_created": wo.ts_created.isoformat(),
        "ts_sent": wo.ts_sent.isoformat() if wo.ts_sent else None,
        "ts_ack": wo.ts_ack.isoformat() if wo.ts_ack else None,
        "ts_closed": wo.ts_closed.isoformat() if wo.ts_closed else None,
        "ts_cancel_requested": (
            wo.ts_cancel_requested.isoformat()
            if wo.ts_cancel_requested
            else None
        ),
        "symbol": wo.symbol,
        "side": wo.side.value,
        "price": wo.price,
        "size": wo.size,
        "post_only": 1 if wo.post_only else 0,
        "status": wo.status.value,
        "cancel_reason": wo.cancel_reason,
        "replace_group_id": wo.replace_group_id,
        "quote_cycle_id": wo.quote_cycle_id,
        "soft_flatten_event_id": wo.soft_flatten_event_id,
        "tp_event_id": wo.tp_event_id,
        "target_half_spread_bps": wo.target_half_spread_bps,
        "quote_aggressiveness": wo.quote_aggressiveness,
        # 2026-05-13 regime-observability Phase 1: decision-state
        # stamping. Bool fields serialize to 1/0 INTEGER for SQLite;
        # NULL when unset (legacy / non-MM placement path).
        "toxicity_score_at_decision": wo.toxicity_score_at_decision,
        "vol_estimate_at_decision": wo.vol_estimate_at_decision,
        "active_sides_at_decision": wo.active_sides_at_decision,
        "decision_reason_at_decision": wo.decision_reason_at_decision,
        "binance_basis_ewma_at_decision": wo.binance_basis_ewma_at_decision,
        "adaptive_widen_active_at_decision": (
            None
            if wo.adaptive_widen_active_at_decision is None
            else (1 if wo.adaptive_widen_active_at_decision else 0)
        ),
        "post_fill_cooldown_active_bid_at_decision": (
            None
            if wo.post_fill_cooldown_active_bid_at_decision is None
            else (1 if wo.post_fill_cooldown_active_bid_at_decision else 0)
        ),
        "post_fill_cooldown_active_ask_at_decision": (
            None
            if wo.post_fill_cooldown_active_ask_at_decision is None
            else (1 if wo.post_fill_cooldown_active_ask_at_decision else 0)
        ),
        "at_touch_adverse_pause_bid_at_decision": (
            None
            if wo.at_touch_adverse_pause_bid_at_decision is None
            else (1 if wo.at_touch_adverse_pause_bid_at_decision else 0)
        ),
        "at_touch_adverse_pause_ask_at_decision": (
            None
            if wo.at_touch_adverse_pause_ask_at_decision is None
            else (1 if wo.at_touch_adverse_pause_ask_at_decision else 0)
        ),
        "quote_distance_to_touch_ticks_at_placement": (
            wo.quote_distance_to_touch_ticks_at_placement
        ),
        # 2026-05-13 Phase 4a.
        "expected_net_edge_bps_at_decision": (
            wo.expected_net_edge_bps_at_decision
        ),
        # 2026-05-14 todo-027 Tier 2.
        "vol_trend_active_at_decision": (
            None
            if wo.vol_trend_active_at_decision is None
            else (1 if wo.vol_trend_active_at_decision else 0)
        ),
        "post_swing_active_at_decision": (
            None
            if wo.post_swing_active_at_decision is None
            else (1 if wo.post_swing_active_at_decision else 0)
        ),
        "session_drawdown_tier_at_decision": (
            wo.session_drawdown_tier_at_decision
        ),
        # v1.4.175 Phase 3F — reservation-alpha shift contributions
        # at decision time. NULL on legacy / SF / manual paths where
        # the parent breakdown wasn't available.
        "ob_imbalance_shift_bps_at_decision": (
            wo.ob_imbalance_shift_bps_at_decision
        ),
        "trend_drift_shift_bps_at_decision": (
            wo.trend_drift_shift_bps_at_decision
        ),
        "flow_score_shift_bps_at_decision": (
            wo.flow_score_shift_bps_at_decision
        ),
        "basis_deviation_shift_bps_at_decision": (
            wo.basis_deviation_shift_bps_at_decision
        ),
        # v1.5.190 Phase 8A Option C — per-order AS attribution. NULL
        # on legacy / SF / manual / hydrated rows where no breakdown
        # was available at place-time.
        "as_base_half_spread_bps_at_decision": (
            wo.as_base_half_spread_bps_at_decision
        ),
        # v1.5.204 Phase 4A — per-order microprice widen attribution.
        # Same nullability semantics as the row above.
        "microprice_bid_widen_bps_at_decision": (
            wo.microprice_bid_widen_bps_at_decision
        ),
        "microprice_ask_widen_bps_at_decision": (
            wo.microprice_ask_widen_bps_at_decision
        ),
        # v1.5.306 audit §5 P0 #2 — per-order AQC attribution. NULL on
        # legacy / SF / manual / hydrated rows + whenever AQC is disabled
        # (controller None at place-time). Float passthrough for the
        # aggression level; SQLite-bool encode for the floor flag.
        "aqc_aggression_level_at_decision": (
            wo.aqc_aggression_level_at_decision
        ),
        "aqc_safety_floor_engaged_at_decision": (
            None
            if wo.aqc_safety_floor_engaged_at_decision is None
            else (1 if wo.aqc_safety_floor_engaged_at_decision else 0)
        ),
        # 1.3.82 connectivity diagnostics.
        "cancel_trigger_reason": wo.cancel_trigger_reason,
        "ts_place_response": (
            wo.ts_place_response.isoformat() if wo.ts_place_response else None
        ),
        "place_response_outcome": wo.place_response_outcome,
        "cancel_response_outcome": wo.cancel_response_outcome,
        # 1.3.83 venue-side detail strings (truncated to 400 chars
        # because OKX msgs can be verbose).
        "place_response_detail": (
            (wo.place_response_detail or "")[:400]
            if wo.place_response_detail else None
        ),
        "cancel_response_detail": (
            (wo.cancel_response_detail or "")[:400]
            if wo.cancel_response_detail else None
        ),
        # 1.4.0 cancel-prio Phase 0.5 — cancel-latency decomposition.
        # t2 (HTTP-send wall clock) and t3 (HTTP-ack wall clock) +
        # OKX server-side cancel timestamp from the WS event.
        "ts_cancel_sent": (
            wo.ts_cancel_sent.isoformat() if wo.ts_cancel_sent else None
        ),
        "ts_cancel_acked": (
            wo.ts_cancel_acked.isoformat() if wo.ts_cancel_acked else None
        ),
        "venue_cancel_utime_ms": wo.venue_cancel_utime_ms,
        # 1.4.15 amend-prio Phase 1: persist amend lifecycle fields.
        "amend_intent_seq": int(getattr(wo, "amend_intent_seq", 0) or 0),
        "amend_target_px": getattr(wo, "amend_target_px", None),
        "amend_target_sz": getattr(wo, "amend_target_sz", None),
        "ts_amend_sent": (
            wo.ts_amend_sent.isoformat()
            if getattr(wo, "ts_amend_sent", None)
            else None
        ),
        "ts_amend_response": (
            wo.ts_amend_response.isoformat()
            if getattr(wo, "ts_amend_response", None)
            else None
        ),
        "amend_response_outcome": getattr(
            wo, "amend_response_outcome", None
        ),
        "amend_response_detail": (
            (wo.amend_response_detail or "")[:400]
            if getattr(wo, "amend_response_detail", None)
            else None
        ),
    }


@dataclass(frozen=True, slots=True)
class SlotDiff:
    """v1.4.88 wedge-elimination-cleanup Phase 5B — per-slot diff struct.

    Pre-Phase-5B the three materiality helpers
    (``_should_replace_working_order``, ``_action_materiality_allows_replace``,
    ``_should_emit_fresh_place_intent``) each independently recomputed
    px/sz deltas from the same ``(current, desired)`` pair. Codex F11
    flagged this as redundant work on the 0.5 s quote loop.

    ``SlotDiff`` is the unified per-tick diff struct: build ONCE per
    slot per tick via ``SlotDiff.from_pair`` and pass to the helpers
    that need it. The arithmetic cost is microscopic (~3 mul / 2 div
    per call), so the win is code clarity, not CPU. But: having a
    single point of truth for "how far is the current order from the
    desired one" eliminates the class of bugs where the three helpers
    silently drift out of agreement (e.g., one uses ticks, another
    uses bps; one uses ``cur.size``, another uses ``desired.size``).
    """

    # |cur.price - desired.price| / tick
    px_ticks: float
    # |cur.price - desired.price| / mid * 10_000 — the legacy bps metric
    # used by ``_should_replace_working_order``.
    px_bps_vs_mid: float
    # |cur.size - desired.size| / cur.size  (legacy denominator)
    sz_rel: float

    @staticmethod
    def from_pair(
        cur_price: float,
        cur_size: float,
        desired_price: float,
        desired_size: float,
        *,
        tick: float,
        mid_price: float,
    ) -> "SlotDiff":
        """Construct a SlotDiff from raw float inputs. Matches the
        legacy helpers' arithmetic bit-for-bit:

        * ``tick`` floor of 1e-15 (matches ``_action_materiality_allows_replace``).
        * ``mid_price`` floor of 1e-12 (matches ``_should_replace_working_order``).
        * ``cur_size`` denominator floor of 1e-15 (matches
          ``_action_materiality_allows_replace``) — note the legacy
          ``_should_replace_working_order`` used 1e-12 instead. The
          difference between 1e-15 and 1e-12 only matters when
          ``cur_size`` is sub-femto, which is impossible in production
          (min_size > 0). Using the 1e-15 floor is safe.
        """
        px_diff = abs(float(cur_price) - float(desired_price))
        sz_diff = abs(float(cur_size) - float(desired_size))
        return SlotDiff(
            px_ticks=px_diff / max(float(tick), 1e-15),
            px_bps_vs_mid=px_diff / max(float(mid_price), 1e-12) * 10_000.0,
            sz_rel=sz_diff / max(float(cur_size), 1e-15),
        )


class OrderManager:
    """Exchange execution: quote maintenance, transport, and lifecycle updates.

    Ownership: ``QuoteEngine`` owns desired prices/sizes; ``maybe_refresh_quotes`` owns the
    working-vs-desired diff; the transport worker owns in-flight HTTP submit/cancel; private
    order/fill events own post-ACK lifecycle; reconcile is a safety repair path, not the
    primary placement loop.
    """

    def __init__(
        self,
        settings: Settings,
        client: PerpExchangeAdapter,
        storage: Storage,
        state: "BotState",
        private_event_queue: Optional[queue.Queue] = None,
        request_kill_fn: Optional[Callable[[str, Optional[dict]], None]] = None,
        clock: Optional["Clock"] = None,
    ) -> None:
        from app.state import BotState

        self._settings = settings
        self._client = client
        self._storage = storage
        self._state: BotState = state
        # Phase 1b (v1.4.230) — Clock abstraction. Default to
        # ``SystemClock`` so production behaviour is unchanged.
        # ``Bot.__init__`` passes its own ``self._clock`` through to
        # us so the bot + OrderManager share a single clock instance
        # (essential — they read each other's monotonic timestamps
        # via shared state).
        if clock is None:
            from app.clock import SystemClock
            clock = SystemClock()
        self._clock: "Clock" = clock
        self._private_q: Optional[queue.Queue] = private_event_queue
        # 2026-05-14 BUG-024: callback wired by Bot to its own ``kill``
        # method so OrderManager can trigger an immediate, full-fidelity
        # shutdown (cancel-all → flatten → Telegram → mark KILLED) on
        # CRITICAL connectivity failures (place response unconfirmed,
        # silent SENT → reconcile gone_on_exchange). Optional only so
        # legacy test contexts that don't wire a Bot can still
        # construct an OrderManager.
        self._request_kill_fn: Optional[
            Callable[[str, Optional[dict]], None]
        ] = request_kill_fn
        # ``drain_private_events`` may be called concurrently from two
        # threads: the quote loop (one_tick) for low-latency processing
        # during normal operation, AND a dedicated drain thread (started
        # by ``Bot.run_forever``) that keeps draining even when the bot
        # is killed — so post-kill exchange events still land in the DB.
        # The mutex serialises those two callers so in-memory state
        # mutations inside ``_dispatch_private_event`` (wo.status, pnl,
        # etc.) can't race.
        self._drain_mutex: threading.Lock = threading.Lock()
        self._private_ws_healthy: bool = False
        # First-connect vs reconnect tracker. On a true reconnect (we
        # previously saw DISCONNECTED), the catchup REST burst is
        # warranted -- we may have missed fills during the gap. On the
        # FIRST connect of a fresh process, there's nothing to catch
        # up on (we have no prior state). Distinguishing the two
        # prevents the startup REST flood that 429s OKX's per-endpoint
        # rate limit (10/2s on /account/positions and /balance, even
        # on MM-tier sub-accounts where the aggregate budget is much
        # higher). See bug-018 / 2026-05-05 OKX bring-up incident.
        self._private_ws_seen_disconnect: bool = False
        # Fast-start: defer private WS isSnapshot userFills for analytics after startup readiness.
        # This avoids startup-state interference from historical replay.
        self._deferred_private_snapshot_fills: deque[PrivateFillEvent] = deque(
            maxlen=max(1024, int(self._settings.private_ws_queue_max))
        )
        self._bot_tick_counter: int = 0
        self._last_order_ws_applied: dict[int, tuple[int, str]] = {}
        # v1.4.68 wedge-elimination-cleanup Phase 1A: per-cloid WS event
        # buffer to close the WS→WO matcher race.
        #
        # Symptom (snapshot v1.4.67-260518-195033, OID 3577472333660495873):
        # the WS ``live`` event arrived 250 µs after the HTTP place
        # response, but the local WO transitioned SENT→ACKED 33 seconds
        # later. ``ws_event_count=45`` on the trace — events were
        # arriving but not actioning the WO.
        #
        # Root cause: ``_handle_private_order_update`` matches against
        # ``state.all_working_orders()`` by OID first, then cloid. When
        # a WS event arrives between (a) the WO being staged with its
        # cloid and (b) the place response binding the OID, the cloid
        # path SHOULD match — but if the event arrives BEFORE the WO
        # is in state, or processing order causes a missed match, the
        # event is silently dropped (see the ``return`` at the
        # ``private_order_update_no_local_working`` branch).
        #
        # Fix: when the matcher fails AND the event has a cloid,
        # buffer the event keyed by cloid (bounded). When a place
        # response commits the OID + transitions to ACKED, the place
        # handler drains the buffer for that cloid and applies the
        # buffered events in order, picking the highest-state.
        #
        # Bound: ``_PENDING_WS_BUFFER_PER_CLOID_MAX`` entries per cloid
        # (oldest evicted with WARN). Total buffer capped at
        # ``_PENDING_WS_BUFFER_TOTAL_MAX`` entries. A sweeper drops
        # entries older than ``_PENDING_WS_BUFFER_TTL_SECONDS``.
        self._pending_ws_events_by_cloid: dict[
            str, list[tuple[float, PrivateOrderUpdateEvent]]
        ] = {}
        # Cumulative counter exposed in executor_state_snapshot. If this
        # stays high after Phase 1A ships, the matcher bug isn't fully
        # closed and we need to investigate further.
        self._ws_event_unmatched_to_local_wo_total: int = 0
        # Cumulative counter for buffer drains. Each time a place
        # response commits the OID and applies buffered events, this
        # increments by the number of events applied.
        self._ws_event_buffered_replay_applied_total: int = 0
        # Buffer eviction / drop counters (separate from successful
        # replay so the operator can distinguish "we caught a race"
        # from "the buffer overflowed").
        self._ws_event_buffer_dropped_oldest_total: int = 0
        self._ws_event_buffer_swept_stale_total: int = 0
        self._ws_event_buffer_dropped_full_total: int = 0
        # Last sweep monotonic timestamp (rate-limited).
        self._ws_event_buffer_last_sweep_mono: float = 0.0
        # v1.4.69 Phase 1B: hydration dedup counters.
        # ``_hydration_merged_existing_total`` — incremented when
        # hydration found a non-terminal local WO with matching
        # (oid, cloid) and merged into it. Non-zero = the
        # duplicate-WO bug was averted in flight.
        # ``_hydration_skipped_terminal_match_total`` — incremented
        # when hydration found a TERMINAL match and skipped. Benign;
        # typically REST returning a freshly-cancelled order.
        self._hydration_merged_existing_total: int = 0
        self._hydration_skipped_terminal_match_total: int = 0
        # v1.4.74 Phase 2A: gate-applied counters. Each tick where a
        # gate (side_unresolved / adverse_side_pause /
        # post_only_cross_cooldown) suppressed at least one slot via
        # the desired-state fold increments the matching counter.
        # Used to measure how often each gate is actively shaping
        # quote dispatch. Surfaced in ``executor_state_snapshot``.
        self._gate_side_unresolved_applied_total: int = 0
        self._gate_adverse_side_pause_applied_total: int = 0
        self._gate_post_only_cross_cooldown_applied_total: int = 0
        # Phase 2A invariant violation: counts dispatched PlaceAction
        # /AmendAction on a side that is unresolved. Should always
        # be zero in production; non-zero indicates a future code
        # path bypassed the desired-state fold.
        self._gate_phase2a_invariant_violation_total: int = 0
        # v1.4.75 Phase 2B: stale-ghost reaper.
        # ``_reaper_last_call_mono`` — last time _reap_stale_ghosts ran.
        # Rate-limited to ``_REAPER_MIN_INTERVAL_SECONDS``.
        self._reaper_last_call_mono: float = 0.0
        # Cumulative reap counts per category. A non-zero value here
        # means the LEGITIMATE recovery paths failed; the reaper is
        # the last-resort safety net. Sustained non-zero counts in
        # prod indicate an upstream bug worth investigating.
        self._reaper_cancel_pending_reaped_total: int = 0
        self._reaper_desync_removed_total: int = 0
        self._reaper_sent_rejected_total: int = 0
        # v1.4.92 Phase 4A cutover — Phase 2D audit fields REMOVED.
        # The runtime audit (``_tick_consumed_qbr_fields``,
        # ``_quote_build_result_unconsumed_field_total``) was a
        # stopgap to catch the v1.4.66 regression class
        # ("new control field added to QuoteBuildResult, no consumer
        # reads it"). The typed ``BuildCommand`` sum-type makes this
        # regression class structurally impossible — new fields must
        # be added to a specific variant, and exhaustive ``match``
        # statements force consumers to handle them.
        self._rest_fill_catchup_until_tick: int = 0
        self._force_open_orders_reconcile: bool = False
        # Max statusTimestamp seen for terminal WS order updates (filled/canceled/rejected).
        self._ws_order_terminal_ts: dict[int, int] = {}
        # 1.2.68: drift-detection for the permissive order-status mapper.
        # Tracks the set of distinct raw status strings we've seen on the
        # private-WS feed. The first time a new value appears, we log it
        # at INFO level (event_type ``novel_order_status``) so the operator
        # can grep journalctl and decide whether a permissive
        # classification of a new venue state is correct. See
        # ``BUGS/bug-021.md`` for the failure-mode catalogue.
        self._seen_order_statuses: set[str] = set()
        self._quote_exec_telemetry: dict[str, Any] = {
            "exec_raw_bid_px": None,
            "exec_raw_bid_sz": None,
            "exec_raw_ask_px": None,
            "exec_raw_ask_sz": None,
            "exec_norm_bid_px": None,
            "exec_norm_bid_sz": None,
            "exec_norm_ask_px": None,
            "exec_norm_ask_sz": None,
            # Legacy aliases — same as meta grid (do not confuse with “only tick-step” placement).
            "exec_price_tick": None,
            "exec_size_step": None,
            "exec_meta_decimal_grid_price_tick": None,
            "exec_meta_decimal_size_step": None,
            "exec_hl_max_sig_figs_nonint": None,
            "exec_price_normalize_pipeline": None,
            "exec_wire_bid_limit_p": None,
            "exec_wire_ask_limit_p": None,
        }
        self._execution_readiness_logged: bool = False
        # Deterministic exchange rejection quarantine (placement only; does not touch desync).
        # Strikes/until keyed by symbol|side|normalized px|sz|exact rejection text (normalized).
        self._vk_strikes: dict[str, int] = {}
        self._vk_until: dict[str, float] = {}
        # Last deterministic rejection reason per placement intent (for pre-submit suppression).
        self._vk_last_reason: dict[str, str] = {}
        self._sent_ambiguous_polls: dict[str, int] = {}
        self._post_only_cross_cooldown_until: dict[Side, float] = {
            Side.BUY: 0.0,
            Side.SELL: 0.0,
        }
        # Phase 2K.11 (v1.4.160) — favorable-exit state for the
        # post-only-cross cooldown. ``rejected_price`` captures the
        # price the venue rejected so we can detect "touch moved
        # ≥ 1 tick away" and clear the cooldown early.
        self._post_only_cross_cooldown_rejected_price: dict[
            Side, Optional[float]
        ] = {Side.BUY: None, Side.SELL: None}
        self._post_only_cross_cooldown_was_active_last: dict[
            Side, bool
        ] = {Side.BUY: False, Side.SELL: False}
        self._post_only_cross_cooldown_cleared_via_favorable_total: int = 0
        self._post_only_cross_cooldown_cleared_via_ceiling_total: int = 0
        # Adverse-side pause: per-side deadline (monotonic seconds). Armed by
        # ``_maybe_arm_adverse_side_pause`` when that side's recent fills average
        # worse than ``-adverse_side_pause_soft_bps``. Cleared naturally when
        # the bot's clock passes the deadline; the side becomes tradable
        # again on the next quote cycle.
        self._adverse_side_pause_until: dict[Side, float] = {
            Side.BUY: 0.0,
            Side.SELL: 0.0,
        }
        # Per-side fill count at the moment of arming. Used to prevent stale
        # re-arming: after the pause expires, we only re-arm if there's been at
        # least one NEW fill on that side since the prior arming. Without this,
        # a single adverse fill would re-arm the pause forever — the pause itself
        # prevents new placements, so no new data ever arrives to refresh the
        # markout average (observed in ``tmp/snap_20260418_091244``: 499 skips
        # over 3+ minutes after the last BUY fill, re-armed every tick on the
        # same -11.77 bps outlier).
        self._adverse_side_pause_arm_n_fills: dict[Side, int] = {
            Side.BUY: -1,
            Side.SELL: -1,
        }
        # Count of placements suppressed by ``adverse_side_pause`` — observability.
        self._adverse_side_pause_skip_count: dict[Side, int] = {
            Side.BUY: 0,
            Side.SELL: 0,
        }
        # v1.4.152 Phase 2K.2 — favorable-exit support. Records the
        # arm-time signed threshold per side so the exit predicate
        # can compare the current per-side markout against a stable
        # bar (avoids the operator changing config mid-session and
        # accidentally relaxing an active pause). NaN = no arm-time
        # threshold recorded (back-compat for state existing before
        # the favorable-exit predicate was added).
        self._adverse_side_pause_arm_threshold: dict[Side, float] = {
            Side.BUY: float("nan"),
            Side.SELL: float("nan"),
        }
        # Exit-attribution counters. Increment when an ACTIVE pause
        # CLEARS — either via the favorable-exit predicate (markout
        # recovered past the configured clear band) or via the MAX
        # cooldown ceiling firing as a safety net. Ratio of the two
        # tells the operator whether the favorable-exit predicate is
        # doing meaningful work or whether the ceiling is the binding
        # constraint. Surfaced via telemetry / live_stats.
        self.adverse_side_pause_cleared_via_favorable_total: int = 0
        self.adverse_side_pause_cleared_via_ceiling_total: int = 0
        # Set when a mandatory quote reprice cancel is sent; cleared after replace place or skip.
        self._quote_reprice_replace_pending: dict[Side, bool] = {
            Side.BUY: False,
            Side.SELL: False,
        }
        self._min_notional_passive_block: dict[Side, bool] = {
            Side.BUY: False,
            Side.SELL: False,
        }
        self._sub_min_notional_flatten_next_mono: float = 0.0
        self._dust_ignored_log_next_mono: float = 0.0
        self._decision_to_first_place_ms_window: deque[float] = deque(
            maxlen=max(1, int(self._settings.execution_latency_window_samples))
        )
        self._open_orders_reconcile_in_flight: bool = False
        self._open_orders_reconcile_next_allowed_mono: float = 0.0
        self._next_interval_reconcile_request_mono: float = (
            self._clock.monotonic() + float(self._settings.open_orders_reconcile_request_interval_seconds)
        )
        self._reconcile_interval_requests_total: int = 0
        self._reconcile_requests_total: int = 0
        self._reconcile_runs_total: int = 0
        self._reconcile_skip_cooldown_total: int = 0
        self._reconcile_skip_inflight_total: int = 0
        self._open_orders_rest_calls_total: int = 0
        self._open_orders_rate_limited_total: int = 0
        # v1.4.43 rate-limit Tier 2 open-orders cache REMOVED in
        # v1.4.55 (wedge-elimination Phase 1). The cache was built for
        # the old ``cancel_all_orders_for_symbol`` path that iterated
        # the exchange snapshot. Phase 1 rewrote cancel-all to use
        # local WO state + dispatcher; the cache had no consumers
        # left. The reconcile path is the canonical source of truth
        # for exchange-vs-local divergence and runs uncached.
        self._dup_open_orders_diag_latched: bool = False
        self._desync_reconcile_next_allowed_mono: float = 0.0
        self._side_unresolved_active: dict[Side, bool] = {Side.BUY: False, Side.SELL: False}
        self._side_unresolved_reason: dict[Side, Optional[str]] = {
            Side.BUY: None,
            Side.SELL: None,
        }
        self._side_unresolved_since_mono: dict[Side, float] = {Side.BUY: 0.0, Side.SELL: 0.0}
        self._cancel_pending_since_mono: dict[Side, Optional[float]] = {
            Side.BUY: None,
            Side.SELL: None,
        }
        # Per-side last-retry monotonic timestamp for cancel-pending-timeout
        # retries. Keyed on side rather than ``order_id_local`` so the watchdog
        # firing for the OTHER side (``cur=None``, cleanup branch) can't wipe
        # the active retry throttle and cause a cancel-HTTP storm (observed in
        # ``tmp/snap_20260417_180429`` — 132 retries in 33 s on a single stuck
        # SELL). Reset to 0 when the side leaves CANCEL_PENDING via the normal
        # cleanup path.
        self._cancel_pending_retry_last_mono: dict[Side, float] = {
            Side.BUY: 0.0,
            Side.SELL: 0.0,
        }
        self._cancel_pending_retry_dispatch_count: int = 0
        # Per-side attempt counter for the current CANCEL_PENDING episode.
        # Incremented each time the watchdog dispatches a cancel retry.
        # Reset when the side leaves CANCEL_PENDING. After
        # ``_CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS`` oid-cancels have
        # been acked-but-ineffective, the retry switches to
        # cancel-by-cloid (+ GRVT's ``time_to_live_ms``) — observed on GRVT
        # to resolve orders that oid-cancel acks-without-cancelling.
        self._cancel_pending_retry_attempts: dict[Side, int] = {
            Side.BUY: 0,
            Side.SELL: 0,
        }
        self._duplicate_open_detect_count: int = 0
        self._side_unresolved_enter_count: int = 0
        self._cancel_pending_timeout_count: int = 0
        self._suppress_place_due_unresolved_count: int = 0
        # Per-side transition flag: True once we have emitted a
        # ``same_side_place_suppressed`` log for the current unresolved
        # episode. Reset back to False in ``_clear_side_unresolved``.
        # Without this, every quote cycle re-emits the suppression log
        # while the side is stuck — observed in post-Binance logs:
        # 10,390 lines in 4 minutes (42.7% of log bytes) for a single
        # stuck episode. The counter above still increments on every
        # suppression so telemetry is unchanged; only the log line is
        # now one-per-episode.
        self._suppress_place_unresolved_logged: dict[Side, bool] = {
            Side.BUY: False,
            Side.SELL: False,
        }
        # Per-side (oid, dispatched_at_mono) dedup for
        # ``cancel_pending_timeout_recovery`` log. The existing retry
        # debounce in ``_maybe_handle_cancel_pending_timeout`` rate-
        # limits the cancel RE-DISPATCH; without this separate tracker
        # the LOG line still fires on every tick the timeout check
        # runs. Observed: 1,020 identical log lines for a single stuck
        # order in a 4-minute run. Reset when the order clears (oid
        # changes or side becomes resolved).
        self._cancel_pending_timeout_logged_oid: dict[Side, Optional[int]] = {
            Side.BUY: None,
            Side.SELL: None,
        }
        self._desync_recovery_attempts: int = 0
        # Cancel-storm protection: TTL dedup keyed on (symbol, oid). See
        # ``_ORPHAN_CANCEL_DEDUP_TTL_SECONDS``. Populated only on successful / benign_missing
        # dispatches; failures bypass the cache so retries still reach the exchange.
        self._recent_orphan_cancels: dict[tuple[str, int], float] = {}
        self._orphan_cancel_dedup_skips: int = 0
        # Per-side latch: when True the failsafe timeout in ``_is_side_unresolved`` MUST NOT
        # auto-release this side. Set when we transitioned into unresolved without a confirmed
        # exchange cancel (orphan cancel dispatch failed). Cleared by any subsequent
        # ``_clear_side_unresolved`` (which is only called from paths that have positive
        # exchange confirmation — reconcile_*, ws_terminal_*, working_slot_released_*).
        self._side_unresolved_requires_confirm: dict[Side, bool] = {
            Side.BUY: False,
            Side.SELL: False,
        }
        self._side_unresolved_confirm_block_count: int = 0
        # Persistent-no-quote diagnostic counter (2026-05-08).
        # Tracks consecutive ticks where ``build_quotes`` returned
        # ``mode="no_quote"``, so we can surface silent wedges (engine
        # producing no orders for any reason — sub-spec dust residual,
        # latched side_unresolved, etc.) BEFORE the deadlock watchdog
        # fires at 600 s. Reset on any tick that produces orders.
        # Threshold and re-log cadence are tuneable via settings.
        self._engine_no_quote_streak_ticks: int = 0
        self._engine_no_quote_diag_last_log_mono: float = 0.0
        # v1.5.159 BUG-031 fix — record when the engine RECOVERS from
        # an extended no_quote run. The wedge detector consults this
        # timestamp to add a short grace period: during/right after
        # an engine no_quote stretch, the executor's
        # ``last_outbound_attempt_ts_mono`` accumulates idle seconds
        # (no place attempts happen because no_quote produces no
        # orders to place). When the engine recovers to two_sided,
        # the wedge detector would see idle > 60 s + eligibility =
        # QUOTE_BOTH and false-positive fire while the executor
        # is still catching up. The grace period eliminates this
        # false-positive class.
        #
        # ``0.0`` is the sentinel "never recovered" — the bot's
        # initial state. Only set to a real timestamp when streak
        # transitions from a positive count back to 0 (i.e. real
        # recovery).
        self._engine_no_quote_recovery_ts_mono: float = 0.0
        # v1.4.40 BUG-025: re-log cadence guard for the executor
        # silent-wedge detector. Mirrors the engine_no_quote variant
        # above. Reset to 0.0 means "next firing is the first
        # emission" — `first_hit` branch in the detector.
        self._silent_wedge_diag_last_log_mono: float = 0.0
        # v1.4.50 BUG-025 follow-on: per-tick executor decision trace.
        # Ring buffer per side records the last N ``_orchestrate``
        # decisions; counters tally (side, reason) pairs since session
        # start. Both fields are surfaced in
        # ``_build_executor_state_snapshot`` and inlined into the
        # ``executor_silent_wedge_detected`` event so the postmortem
        # can see EXACTLY which silent-return branch fired during the
        # wedge. Always recorded — zero-cost regardless of the log
        # flag. The optional INFO log per decision is gated by
        # ``settings.executor_decision_trace_enabled``.
        _odt_size = max(
            20, int(getattr(settings, "executor_decision_trace_buffer_size", 200))
        )
        self._orchestrate_decision_history: dict[Side, deque[dict[str, Any]]] = {
            Side.BUY: deque(maxlen=_odt_size),
            Side.SELL: deque(maxlen=_odt_size),
        }
        self._orchestrate_decision_counts: dict[str, int] = {}
        # Cache the most-recent "why _stage_place_order_local returned
        # None" per side so the decision trace can surface the actual
        # reason (write_access, risk_check, side_unresolved, etc.)
        # instead of just "stage_returned_none".
        self._last_stage_place_skip_reason: dict[str, Optional[str]] = {
            "BUY": None,
            "SELL": None,
        }
        # v1.4.53 wedge-investigation follow-on: separate ring buffer
        # for ``maybe_refresh_quotes`` early-returns (the path that runs
        # BEFORE ``_orchestrate``). The 2026-05-18 v1.4.52-160650 snapshot
        # showed the executor wedged for 2 minutes with NO
        # ``_orchestrate`` decisions recorded — proving the wedge wasn't
        # in ``_orchestrate`` itself but upstream, in one of the
        # uninstrumented early-return paths of ``maybe_refresh_quotes``.
        # This second buffer surfaces which early-return is firing.
        self._quote_refresh_skip_history: deque[dict[str, Any]] = deque(
            maxlen=_odt_size
        )
        self._quote_refresh_skip_counts: dict[str, int] = {}
        # v1.4.54 cooldown removed in v1.4.55 — superseded by Phase 1
        # cancel-all rewrite (state-aware idempotency).
        # v1.4.56 wedge-elimination Phase 2: risk-action execution
        # state machine. ``NORMAL`` (no suppression), ``CANCELLING``
        # (cancel-all in flight, ticks no-op), ``SUPPRESSED`` (cancels
        # confirmed, waiting for risk to clear). See
        # ``app/enums.py:RiskExecState`` for the transition diagram.
        self._risk_exec_state: RiskExecState = RiskExecState.NORMAL
        self._risk_exec_state_entered_mono: float = 0.0
        self._risk_exec_state_last_risk_action: Optional[RiskAction] = None
        # Cumulative transition counters surfaced in executor_state
        # snapshot for postmortem visibility.
        self._risk_exec_state_transition_counts: dict[str, int] = {}
        self._quote_engine = QuoteEngine(settings, client.symbol_spec)
        # Last bot-tick latency slices (ms); reset in ``reset_tick_latency_metrics``.
        self._last_tick_reconcile_rest_ms: float = 0.0
        self._last_tick_order_submit_rtt_ms: float = 0.0
        self._last_tick_quote_engine_build_ms: float = 0.0
        # Rolling place-to-ack RTT tracker (microsecond precision, p50/p95/p99).
        # Read by /latency Telegram command and /health API. Bounded
        # 1024-sample deque; lock-free from the bot's main thread.
        from app.order_rtt_tracker import OrderRttTracker

        self._order_rtt_tracker = OrderRttTracker(max_samples=1024)
        # Expose the rolling RTT summary to the heartbeat publisher so
        # the operator dashboard's metrics panel doesn't need a second
        # data path. ``BotState`` is already a shared object, so a
        # bound-method assignment is the lightest hand-off.
        try:
            self._state.order_rtt_summary_provider = self.order_rtt_summary
            # 1.4.0 cancel-prio Phase 0.5: also expose the cancel RTT
            # summary. Same tracker, filtered by ``op="cancel"``. The
            # heartbeat publisher reads this for the new
            # ``cancel_submit_rtt_ms`` block.
            self._state.cancel_rtt_summary_provider = self.cancel_rtt_summary
            # v1.4.58 todo-037: unified TX → ack summary across all
            # three op kinds (place + amend + cancel). Dashboard
            # LATENCY panel reads this single combined metric instead
            # of two per-op rows. Per-op trackers remain for postmortem.
            self._state.tx_rtt_summary_provider = self.tx_rtt_summary
        except Exception:
            # Strictly best-effort wiring; never block startup over a
            # dashboard-only field.
            logger.exception("order_rtt_summary_provider_wire_failed")
        # Phase 3+: outbound dispatcher (coalesced batch; WS-primary exchange actions in client).
        self._place_intent_seq: dict[Side, int] = {Side.BUY: 0, Side.SELL: 0}
        self._cancel_intent_seq: dict[Side, int] = {Side.BUY: 0, Side.SELL: 0}
        self._outbound_traces: dict[str, OutboundActionTrace] = {}
        self._ws_action_send_count = 0
        self._http_action_send_count = 0
        self._dropped_stale_intent_count = 0
        self._avg_batch_size_window: deque[float] = deque(maxlen=50)
        self._last_outbound_replace_mono: dict[Side, float] = {Side.BUY: 0.0, Side.SELL: 0.0}
        self._last_emitted_fp: dict[Side, Optional[tuple[float, float]]] = {
            Side.BUY: None,
            Side.SELL: None,
        }
        # v1.4.180 — Phase 2I fall-through bug fix. When
        # ``_enqueue_amend_quote_path`` returns False, callers need to
        # know WHY: suppressed by the Phase 2I tick-flicker / rate-
        # throttle guard (in which case the AmendAction dispatcher
        # MUST NOT fall through to cancel-replace — that defeats the
        # guard's intent) vs. genuine enqueue failure (cancel-replace
        # is the correct recovery). Set inside ``_enqueue_amend_quote_path``
        # on each invocation: ``None`` on success or generic failure,
        # ``"tick_flicker"`` / ``"rate_throttle"`` on Phase 2I suppression.
        # The AmendAction dispatcher reads this attribute immediately
        # after the call.
        self._last_amend_enqueue_suppress_reason: Optional[str] = None
        # Quote-cycle suppression observation state. Events are emitted only on
        # transitions (a reason newly appearing after being absent), so an
        # always-on suppressor produces one event, not one per cycle.
        self._last_suppression_reasons: frozenset[str] = frozenset()
        self._last_decision_to_submit_dispatch_ms: float = 0.0
        self._last_submit_queue_wait_ms: float = 0.0
        self._last_decision_to_first_submit_dispatch_ms: float = 0.0
        self._last_ack_resolution_ms: Optional[float] = None
        self._maybe_refresh_t0_perf: float = 0.0
        self._first_enqueue_perf: Optional[float] = None
        self._outbound = OutboundDispatchCoordinator(
            settings,
            execute_place=self._execute_place_intent,
            execute_cancel=self._execute_cancel_intent,
            # 1.4.0 cancel-prio Phase 1b: opportunistic batch-cancel.
            # The coordinator calls this when 2+ cancels for distinct
            # sides queue together; the implementation builds a single
            # batch HTTP payload and dispatches per-row outcomes to
            # ``_on_cancel_response_for_intent``. Gated on the
            # ``batch_cancels_enabled`` setting (default ON).
            execute_cancel_batch=self._execute_cancel_batch_intents,
            # 1.4.4: opportunistic batch-PLACE. Parallel surface to
            # execute_cancel_batch for the place lane — when 2+
            # places for distinct (side, level_idx) queue together
            # (or 1+ in always-batch mode), the coordinator calls
            # this with the whole list. Adapter-dependent: when the
            # underlying client does not expose
            # ``batch_place_post_only_limit`` the callback raises
            # AttributeError on first use and the coordinator falls
            # back to per-intent. Currently wired only on OKX.
            execute_place_batch=self._execute_place_batch_intents,
            # amend-prio Phase 3 (v1.4.16): amend executor callbacks.
            # The place lane carries amend intents with kind="amend"
            # and the coordinator routes them here. ``execute_amend``
            # delegates to the batch path (mirroring the v1.4.6
            # "always-batch" convention so single amends route through
            # the same /trade/amend-batch-orders rate-limit pool). The
            # batch executor handles per-row outcome dispatch (accept,
            # below_filled, order_gone, post_only_cross, etc.) per
            # ``plans/amend-prio.md`` Phase 3.
            execute_amend=self._execute_amend_intent,
            execute_amend_batch=self._execute_amend_batch_intents,
            on_stats=self._on_outbound_batch_stats,
        )
        self._outbound.start()

    @property
    def private_ws_healthy(self) -> bool:
        return self._private_ws_healthy

    @property
    def last_tick_reconcile_rest_ms(self) -> float:
        return float(self._last_tick_reconcile_rest_ms)

    @property
    def last_tick_order_submit_rtt_ms(self) -> float:
        return float(self._last_tick_order_submit_rtt_ms)

    @property
    def last_tick_quote_engine_build_ms(self) -> float:
        return float(self._last_tick_quote_engine_build_ms)

    def order_rtt_summary(self) -> dict[str, Any]:
        """Rolling order-send-to-ack RTT distribution. Read by the
        ``/latency`` Telegram command, ``/health`` API, and the
        dashboard's Latency panel.

        1.4.0 cancel-prio Phase 0.5: cancel samples now share the
        same tracker but are exposed via :meth:`cancel_rtt_summary`.

        v1.4.27 — combined ``"place"`` + ``"amend"`` samples. Before
        the amend-prio rollout (v1.4.16+) only places hit the venue's
        place pool, so this summary was place-only. With amend
        active, the dashboard's "Order → ack" panel should reflect
        BOTH place and amend RTT (both are the bot's outbound write
        path to the venue). Pre-v1.4.16 traffic still has all samples
        tagged ``op="place"`` so old data is included automatically.
        """
        return self._order_rtt_tracker.summary(
            op_filter=("place", "amend")
        )

    def cancel_rtt_summary(self) -> dict[str, Any]:
        """Rolling cancel-send-to-ack RTT distribution (Phase 0.5).

        Sibling of :meth:`order_rtt_summary` for cancels. Same shape
        (min/median/p95/p99/max/mean ms + sample counts), filtered
        to ``op="cancel"``. Read by the heartbeat publisher; surfaces
        on ``live_stats.latency.cancel_submit_rtt_ms``.

        Samples are ingested in ``_cancel_http_transport`` only when
        the cancel response interpreted as success — so the
        distribution is a clean wire RTT, not contaminated by
        order-already-gone races or transport errors.
        """
        return self._order_rtt_tracker.summary(op_filter="cancel")

    def tx_rtt_summary(self) -> dict[str, Any]:
        """v1.4.58 todo-037: unified outbound-transport RTT distribution
        across all three op kinds (``place + amend + cancel``).

        Operator review of the dashboard's LATENCY panel
        (snapshot v1.4.53-260518) showed:
          * Place RTT and cancel RTT have statistically indistinguishable
            distributions on OKX (same WS channel, same payload size).
          * Cancel sample counts are tiny (n=19 in the observed snapshot)
            and produce noisy p95/p99.
          * Amend RTT — the DOMINANT outbound action class (~14k/session
            vs ~1k places vs ~1k cancels) — wasn't shown at all.

        This method pools all three op kinds for a single
        "Tx send → ack" headline metric. The per-op trackers remain
        available via :meth:`order_rtt_summary` (place + amend) and
        :meth:`cancel_rtt_summary` (cancel only) for postmortem use
        cases that need per-op breakdown.

        Same return shape as the per-op summaries.
        """
        return self._order_rtt_tracker.summary(
            op_filter=("place", "amend", "cancel")
        )

    def reset_tick_latency_metrics(self) -> None:
        self._last_tick_reconcile_rest_ms = 0.0
        self._last_tick_order_submit_rtt_ms = 0.0
        self._last_tick_quote_engine_build_ms = 0.0
        self._last_decision_to_submit_dispatch_ms = 0.0
        self._last_submit_queue_wait_ms = 0.0
        self._last_decision_to_first_submit_dispatch_ms = 0.0
        self._first_enqueue_perf = None

    def _active_unresolved_sides(self) -> list[Side]:
        out: list[Side] = []
        for side in (Side.BUY, Side.SELL):
            if self._is_side_unresolved(side):
                out.append(side)
        return out

    def _is_side_unresolved(self, side: Side) -> bool:
        active = bool(self._side_unresolved_active.get(side, False))
        if not active:
            return False
        now_m = self._clock.monotonic()
        since = float(self._side_unresolved_since_mono.get(side, 0.0))
        timeout_s = float(self._settings.side_unresolved_suppression_timeout_seconds)
        with self._state._lock:
            # v1.4.194: migrated off the deprecated ``working_bid`` /
            # ``working_ask`` shims (the warning was firing on every
            # ``_is_side_unresolved`` call). ``get_working_order`` is
            # the unlocked per-rung accessor — safe inside the existing
            # ``state._lock`` context. Inside-rung is ``level_idx=0``.
            cur = self._state.get_working_order(side, 0)
        # Failsafe: if local slot is already empty, don't keep suppressing forever —
        # UNLESS the side latch requires a confirmed exchange observation to clear.
        # The latch is set when we became unresolved without a confirmed orphan cancel
        # (see ``_reconcile_side`` exchange_mismatch branch). Auto-releasing in that case
        # would let a new same-side placement race a still-live phantom on the exchange.
        if cur is None and since > 0 and (now_m - since) >= timeout_s:
            if self._side_unresolved_requires_confirm.get(side, False):
                self._side_unresolved_confirm_block_count += 1
                # Log once per timeout window to aid operator visibility without spamming.
                log_extra(
                    logger,
                    logging.WARNING,
                    "side_unresolved_auto_release_blocked",
                    {
                        "side": side.value,
                        "reason": self._side_unresolved_reason.get(side),
                        "elapsed_s": round(now_m - since, 3),
                        "block_count": self._side_unresolved_confirm_block_count,
                    },
                )
                # Extend the "since" reference so we don't re-log every call — only re-log
                # when another full timeout window elapses.
                self._side_unresolved_since_mono[side] = now_m
                return True
            self._clear_side_unresolved(side, reason="timeout_release_empty_slot")
            return False
        return True

    def _set_side_unresolved(
        self,
        side: Side,
        *,
        reason: str,
        payload: Optional[dict[str, Any]] = None,
        requires_confirm: bool = False,
    ) -> None:
        was = bool(self._side_unresolved_active.get(side, False))
        now_m = self._clock.monotonic()
        self._side_unresolved_active[side] = True
        self._side_unresolved_reason[side] = reason
        if not was:
            self._side_unresolved_since_mono[side] = now_m
            self._side_unresolved_enter_count += 1
        # ``requires_confirm`` is monotonic-up for the duration of this unresolved episode:
        # once set True we do not downgrade to False on re-entry, because a pending phantom
        # from the earlier entry is still the dominant safety concern. It clears only on
        # ``_clear_side_unresolved``, which is called from paths with positive exchange
        # confirmation.
        if requires_confirm:
            self._side_unresolved_requires_confirm[side] = True
        ev_payload = {
            "side": side.value,
            "reason": reason,
            "already_active": was,
            "requires_confirm": bool(self._side_unresolved_requires_confirm.get(side, False)),
            "active_sides": [s.value for s in self._active_unresolved_sides()],
        }
        if payload:
            ev_payload.update(payload)
        log_extra(
            logger,
            logging.DEBUG if was else logging.WARNING,
            "side_unresolved_entered",
            ev_payload,
        )

    def _bump_gone_on_exchange_counters(self, wo: WorkingOrder) -> None:
        """v1.4.96 — three-tier classification of a reconcile-confirmed-gone
        terminal transition.

        Called from each ``transition(wo, OrderStatus.CANCELED,
        "gone_on_exchange")`` site in ``_reconcile_side``. Callers MUST
        already hold ``self._state._lock``. Side-effect-only; never
        raises.

        Classification (first-match wins, mutually exclusive):

        **TIER 1 — RED (counted under ``gone_on_exchange_total``):**

        - ``phantom_no_ack`` (BUG-024) — ``ts_ack is None``. Place
          sent, no parseable HTTP ack.
        - ``acked_no_cancel`` (BUG-023) — ``ts_ack`` set,
          ``ts_cancel_requested is None``. Order acked-live, no
          cancel ever dispatched. Silent leak risk.
        - ``cancel_no_http_confirm`` (v1.4.96 new) — cancel was sent
          (``ts_cancel_requested`` set) but no HTTP-success response.
          Bot doesn't know if cancel landed.
        - ``other`` — unclassified.

        **TIER 2 — WARN (counted under ``http_acked_no_ws_total``,
        NOT ``gone_on_exchange_total``):**

        - ``http_acked_no_ws`` — cancel HTTP confirmed success
          (``ts_cancel_acked`` set with ``cancel_response_outcome``
          indicating success) but ``venue_cancel_utime_ms is None``
          (WS terminal didn't arrive). The venue confirmed the
          cancel landed via HTTP; only the WS publish leg failed.
          Tolerable when rare. Has its own lateness histogram +
          ring.

        The Tier-3 WS-arrived-late counter is NOT bumped here — it
        belongs to the unmatched-WS-event handler in
        ``_handle_private_order_update_v2``. See
        ``_record_terminated_oid_for_late_arrival_detection``.

        Every classified terminal (regardless of tier) is stamped
        into ``terminated_oids_recent`` so the unmatched-WS handler
        can detect late WS arrivals against it.
        """
        try:
            state = self._state
            ts_ack = getattr(wo, "ts_ack", None)
            ts_cancel_req = getattr(wo, "ts_cancel_requested", None)
            ts_cancel_ack = getattr(wo, "ts_cancel_acked", None)
            cancel_outcome = (
                getattr(wo, "cancel_response_outcome", None) or ""
            )
            ts_closed = getattr(wo, "ts_closed", None)
            venue_cancel_utime_ms = getattr(wo, "venue_cancel_utime_ms", None)

            # HTTP-cancel-success heuristic. The cancel response was
            # parsed as an explicit success classification by the
            # transport layer (``cancel_response_outcome``). Anything
            # else (transport_rejected, timeout, no_response,
            # exchange_rejected, benign_missing, ...) is NOT proof
            # the cancel landed.
            cancel_http_confirmed = (
                ts_cancel_ack is not None
                and str(cancel_outcome).lower() in ("success", "ok", "accepted")
            )

            # ------------------------------------------------------------------
            # Classify.
            # ------------------------------------------------------------------
            tier: str  # "red" | "warn"
            if ts_ack is None:
                signature = "phantom_no_ack"
                tier = "red"
                state.gone_on_exchange_phantom_no_ack_total += 1
                state.gone_on_exchange_total += 1
            elif ts_cancel_req is None:
                signature = "acked_no_cancel"
                tier = "red"
                state.gone_on_exchange_acked_no_cancel_total += 1
                state.gone_on_exchange_total += 1
            elif cancel_http_confirmed and venue_cancel_utime_ms is None:
                signature = "http_acked_no_ws"
                tier = "warn"
                state.http_acked_no_ws_total += 1
            elif not cancel_http_confirmed:
                signature = "cancel_no_http_confirm"
                tier = "red"
                state.gone_on_exchange_cancel_no_http_confirm_total += 1
                state.gone_on_exchange_total += 1
            else:
                signature = "other"
                tier = "red"
                state.gone_on_exchange_other_total += 1
                state.gone_on_exchange_total += 1

            # ------------------------------------------------------------------
            # Lateness histogram (TIER 2 only).
            # ------------------------------------------------------------------
            lateness_ms: Optional[float] = None
            if ts_cancel_ack is not None:
                close_t = ts_closed or self._clock.now_utc()
                lateness_ms = max(
                    0.0, (close_t - ts_cancel_ack).total_seconds() * 1000.0
                )
                if signature == "http_acked_no_ws":
                    cur_min = state.http_acked_no_ws_lateness_ms_min
                    cur_max = state.http_acked_no_ws_lateness_ms_max
                    state.http_acked_no_ws_lateness_ms_min = (
                        lateness_ms
                        if cur_min is None
                        else min(cur_min, lateness_ms)
                    )
                    state.http_acked_no_ws_lateness_ms_max = (
                        lateness_ms
                        if cur_max is None
                        else max(cur_max, lateness_ms)
                    )
                    state.http_acked_no_ws_lateness_samples.append(lateness_ms)

            # ------------------------------------------------------------------
            # Forensic ring entry (per-tier ring).
            # ------------------------------------------------------------------
            side_val = wo.side.value if hasattr(wo.side, "value") else str(wo.side)
            entry: dict[str, Any] = {
                "ts_terminal": (
                    ts_closed.isoformat()
                    if ts_closed
                    else self._clock.now_utc().isoformat()
                ),
                "order_id_local": str(wo.order_id_local or ""),
                "order_id_exchange": str(wo.order_id_exchange or ""),
                "side": side_val,
                "px": float(wo.price) if wo.price is not None else None,
                "sz": float(wo.size) if wo.size is not None else None,
                "ts_ack": ts_ack.isoformat() if ts_ack else None,
                "ts_cancel_requested": (
                    ts_cancel_req.isoformat() if ts_cancel_req else None
                ),
                "ts_cancel_acked": (
                    ts_cancel_ack.isoformat() if ts_cancel_ack else None
                ),
                "cancel_response_outcome": str(cancel_outcome) or None,
                "venue_cancel_utime_ms": venue_cancel_utime_ms,
                "signature": signature,
                "ws_terminal_lateness_ms": lateness_ms,
                "tier": tier,
            }
            if tier == "red":
                state.gone_on_exchange_recent.append(entry)
            else:
                state.http_acked_no_ws_recent.append(entry)

            # ------------------------------------------------------------------
            # Stamp the oid for late-WS-arrival detection. Both tiers
            # qualify — if the WS event eventually arrives, the
            # unmatched-WS handler will bump ws_arrived_late_total.
            # ------------------------------------------------------------------
            self._record_terminated_oid_for_late_arrival_detection(
                wo, signature=signature, tier=tier
            )
        except Exception:
            # Never let the diagnostic bump break the trading loop.
            logger.exception("bump_gone_on_exchange_counters_failed")

    def _record_terminated_oid_for_late_arrival_detection(
        self, wo: WorkingOrder, *, signature: str, tier: str
    ) -> None:
        """v1.4.96 — remember a recently-terminated oid so the
        unmatched-WS-event handler can detect a late WS terminal
        arrival and bump ``ws_arrived_late_total`` (tier 3, WARN).

        Bounded FIFO map keyed by ``order_id_exchange``. Caller holds
        ``self._state._lock``.
        """
        oid_ex = str(getattr(wo, "order_id_exchange", "") or "")
        if not oid_ex:
            return
        state = self._state
        # FIFO eviction.
        while (
            len(state.terminated_oids_recent)
            >= state.terminated_oids_recent_max
        ):
            try:
                state.terminated_oids_recent.popitem(last=False)
            except KeyError:
                break
        state.terminated_oids_recent[oid_ex] = {
            "signature": signature,
            "tier": tier,
            "ts_terminal": self._clock.now_utc().isoformat(),
            "order_id_local": str(getattr(wo, "order_id_local", "") or ""),
            "side": (
                wo.side.value if hasattr(wo.side, "value") else str(wo.side)
            ),
            "px": float(wo.price) if wo.price is not None else None,
            "sz": float(wo.size) if wo.size is not None else None,
        }

    def _check_ws_arrived_late_for_terminated_oid(
        self, oid_ex: str, ws_state: Optional[str]
    ) -> bool:
        """v1.4.96 — return True if ``oid_ex`` was recently terminated
        locally, indicating the inbound WS event is a "late arrival"
        for an oid the bot has already cleaned up.

        When the match is found:
        - bump ``ws_arrived_late_total``
        - append a ring entry to ``ws_arrived_late_recent``
        - REMOVE the oid from ``terminated_oids_recent`` so a single
          oid is not double-counted

        Caller holds ``self._state._lock``.
        """
        state = self._state
        rec = state.terminated_oids_recent.pop(oid_ex, None)
        if rec is None:
            return False
        state.ws_arrived_late_total += 1
        state.ws_arrived_late_recent.append(
            {
                "ts_late_ws": self._clock.now_utc().isoformat(),
                "order_id_exchange": oid_ex,
                "order_id_local": rec.get("order_id_local"),
                "side": rec.get("side"),
                "px": rec.get("px"),
                "sz": rec.get("sz"),
                "ws_state_received": ws_state,
                "original_signature": rec.get("signature"),
                "original_tier": rec.get("tier"),
                "ts_original_terminal": rec.get("ts_terminal"),
            }
        )
        return True

    def _clear_side_unresolved(self, side: Side, *, reason: str) -> None:
        if not self._side_unresolved_active.get(side, False):
            return
        prev_reason = self._side_unresolved_reason.get(side)
        prev_requires_confirm = bool(self._side_unresolved_requires_confirm.get(side, False))
        self._side_unresolved_active[side] = False
        self._side_unresolved_reason[side] = None
        self._side_unresolved_since_mono[side] = 0.0
        self._cancel_pending_since_mono[side] = None
        self._side_unresolved_requires_confirm[side] = False
        # Reset the per-episode suppress-log transition flag so the
        # NEXT unresolved episode will log its first suppression once
        # more (state transitions deserve one log line; subsequent
        # same-state evaluations are silent counter bumps).
        self._suppress_place_unresolved_logged[side] = False
        # Reset the cancel-pending-timeout log dedup too — a cleared
        # episode means the next stuck cancel (if any) is a new event
        # worth logging.
        self._cancel_pending_timeout_logged_oid[side] = None
        log_extra(
            logger,
            logging.INFO,
            "side_unresolved_cleared",
            {
                "side": side.value,
                "reason": reason,
                "previous_reason": prev_reason,
                "previous_requires_confirm": prev_requires_confirm,
                "active_sides": [s.value for s in self._active_unresolved_sides()],
            },
        )

    def _mark_cancel_pending(self, side: Side) -> None:
        if self._cancel_pending_since_mono.get(side) is None:
            self._cancel_pending_since_mono[side] = self._clock.monotonic()
        self._set_side_unresolved(
            side,
            reason="cancel_pending_wait",
            payload={"cancel_pending_since_mono": self._cancel_pending_since_mono[side]},
        )

    def _reap_wo_after_cancel_unexpected_gone(
        self,
        wo: WorkingOrder,
        *,
        source: str,
        detail: str,
    ) -> None:
        """BUG-025 Phase 3 (v1.4.41): reap a local WorkingOrder when
        OKX returns ``unexpected_gone`` (sCode 51400 / 51401 / 51503)
        on a cancel request. OKX is the authoritative truth: if it
        says "this order has been filled, canceled or doesn't exist,"
        the local record is provably stale and MUST be removed.

        Without this reap, the stale WO sits in
        ``self._state._working_orders[side][level_idx]`` in
        ``CANCEL_PENDING`` status, the next ``_orchestrate`` cycle
        finds it, re-issues a cancel, gets another 51400, re-stamps
        ``cancel_pending_wait`` via ``_request_cancel_http`` ->
        ``_mark_cancel_pending`` — and the bot wedges in a per-cycle
        cancel ping-pong (BUG-025 wedge captured 2026-05-18 in
        ``snapshots/v1.4.40-260518-093848``: 28k+ side_unresolved
        enters in 16 min, two zombie OIDs, 105 cancel attempts apiece,
        deadlock-watchdog kill at idle=600 s).

        Mirrors the WS-CANCELED terminal handler at
        ``_handle_private_order_update`` (lines ~3377-3382). Idempotent
        — safe to call when the slot is already cleared / the WO is
        already CANCELED / side_unresolved is already inactive.

        ``source`` is "batch" or "single" for log attribution.
        ``detail`` is the raw OKX response detail string (e.g.
        ``"okx_row_51400:Order cancellation failed ..."``); the sCode
        digits are extracted for the transition reason so the orders
        row carries the underlying venue code.
        """
        # Extract the sCode digits for the cancel_reason audit field.
        detail_code = "unknown"
        if detail:
            import re

            m = re.search(r"(\d{5})", detail)
            if m:
                detail_code = m.group(1)
        with self._state._lock:
            transition(
                wo,
                OrderStatus.CANCELED,
                f"http:unexpected_gone_{detail_code}",
            )
            try:
                self.persist(wo)
            except Exception:
                logger.exception(
                    "reap_unexpected_gone_persist_failed source=%s oid=%s",
                    source,
                    wo.order_id_exchange,
                )
            level_idx = int(getattr(wo, "level_idx", 0) or 0)
            self._state.set_working_order(wo.side, level_idx, None)
            self._clear_side_unresolved(
                wo.side, reason="http_terminal_unexpected_gone"
            )
            oid = wo.order_id_exchange
            if oid is not None:
                try:
                    self._last_order_ws_applied.pop(int(oid), None)
                except (TypeError, ValueError):
                    pass
                # Guard against late WS lifecycle events for this oid
                # re-introducing the order. Use current wall-clock ms;
                # any subsequent WS payload with an older ``status_
                # timestamp_ms`` will be skipped at the ``_handle_
                # private_order_update`` stale-after-terminal check.
                try:
                    self._record_ws_order_terminal(
                        int(oid), int(self._clock.time() * 1000)
                    )
                except (TypeError, ValueError):
                    pass

    def has_order_state_uncertainty(self) -> bool:
        """Back-compat: True when either desync OR any side is unresolved.

        Prefer :meth:`global_order_state_uncertainty` + :meth:`uncertain_sides`
        for callers that want to distinguish global from per-side — the
        quote-eligibility gate was treating a single stuck ``CANCEL_PENDING``
        as a total bot halt (HOLD_ALL), which meant one stuck side shut down
        the *other* side's rebate capture for as long as the stall lasted.
        """
        if self._state.order_desync:
            return True
        return any(self._is_side_unresolved(s) for s in (Side.BUY, Side.SELL))

    def global_order_state_uncertainty(self) -> bool:
        """True only for account-level desync (not per-side unresolved).

        This preserves the "HOLD_ALL on desync" safety gate while letting a
        single stuck side suppress only that side in the eligibility layer.
        """
        return bool(self._state.order_desync)

    def uncertain_sides(self) -> set[Side]:
        """Set of sides currently in an unresolved local state.

        Used by the eligibility gate so that a BUY stuck in CANCEL_PENDING
        yields ``QUOTE_SELL_ONLY`` instead of ``HOLD_ALL`` — SELL keeps
        earning rebates while BUY's cancel retry path runs to resolution.
        """
        out: set[Side] = set()
        for s in (Side.BUY, Side.SELL):
            if self._is_side_unresolved(s):
                out.add(s)
        return out

    def shutdown_transport(self) -> None:
        """Stop the outbound dispatcher (best-effort; used on bot shutdown and tests)."""
        self._outbound.stop()

    def wait_transport_idle(self, timeout_s: float = 2.0) -> None:
        """Wait until queued outbound work is finished (for tests)."""
        self._outbound.wait_until_idle(timeout_s)

    def _on_outbound_batch_stats(self, payload: dict[str, float | int]) -> None:
        bs = float(payload.get("last_batch_size") or 0.0)
        if bs > 0:
            self._avg_batch_size_window.append(bs)
        with self._state._lock:
            self._state.outbound_action_last_batch_size = bs
            self._state.outbound_action_queue_depth = int(payload.get("queue_depth_after") or 0)
        self._sync_outbound_state_flags()

    def _sync_outbound_state_flags(self) -> None:
        snap = self._outbound.snapshot_stats()
        with self._state._lock:
            self._state.outbound_action_queue_depth = int(snap["action_dispatch_queue_depth"])
            self._state.outbound_action_queue_hwm = max(
                self._state.outbound_action_queue_hwm,
                int(snap["action_dispatch_queue_high_watermark"]),
            )
            self._state.outbound_coalesced_intent_count = int(snap["coalesced_intent_count"])
            if self._avg_batch_size_window:
                self._state.outbound_avg_batch_size = float(
                    sum(self._avg_batch_size_window) / len(self._avg_batch_size_window)
                )
            self._state.outbound_dropped_stale_intent_count = self._dropped_stale_intent_count
            # v1.4.40 BUG-025: cache the executor-state snapshot for
            # publication in ``state_current.json``. Built every time
            # outbound stats refresh — covers the cycle-tick path AND
            # every cancel/place/amend batch finalisation.
            self._state.executor_state_snapshot = (
                self._build_executor_state_snapshot(outbound_snap=snap)
            )

    def _build_executor_state_snapshot(
        self,
        *,
        outbound_snap: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """v1.4.40 BUG-025: read-only snapshot of every executor /
        dispatcher in-memory field that could cause a silent wedge.

        Surfaced via ``BotState.executor_state_snapshot`` →
        ``state_current.json:executor_state``. Zero hot-path cost —
        plain attribute reads inside the existing
        ``_sync_outbound_state_flags`` lock window.

        See ``issues/bug-025-executor-silent-wedge.md`` for the
        bug class this catches. The dict's shape is intentionally
        flat (snake_case keys, primitive values) so the dashboard
        can render it row-by-row and the postmortem can diff
        across snapshots without nested-walk gymnastics.
        """
        now_m = self._clock.monotonic()
        snap = outbound_snap if outbound_snap is not None else self._outbound.snapshot_stats()
        last_place_mono = float(
            getattr(self._state, "last_place_attempt_ts_mono", 0.0) or 0.0
        )
        execution_idle_s = (
            round(now_m - last_place_mono, 3)
            if last_place_mono > 0
            else None
        )
        return {
            # ---- engine-no-quote streak ----
            # Counter that bumps whenever the engine returns
            # ``mode='no_quote'``. Resets to 0 when a valid quote is
            # observed. A non-zero value while the engine is
            # currently producing valid quotes (visible in
            # quotes_recent) is the smoking gun for the v1.4.38 wedge
            # class.
            "engine_no_quote_streak_ticks": int(
                self._engine_no_quote_streak_ticks
            ),
            "engine_no_quote_diag_last_log_age_s": (
                round(now_m - self._engine_no_quote_diag_last_log_mono, 3)
                if self._engine_no_quote_diag_last_log_mono > 0
                else None
            ),
            # ---- side_unresolved (full coverage) ----
            "side_unresolved": {
                side.value.lower(): {
                    "active": bool(
                        self._side_unresolved_active.get(side, False)
                    ),
                    "reason": self._side_unresolved_reason.get(side),
                    "requires_confirm": bool(
                        self._side_unresolved_requires_confirm.get(side, False)
                    ),
                    "since_s": (
                        round(now_m - self._side_unresolved_since_mono.get(side, 0.0), 3)
                        if self._side_unresolved_since_mono.get(side, 0.0) > 0
                        else None
                    ),
                }
                for side in (Side.BUY, Side.SELL)
            },
            "side_unresolved_enter_count": int(
                self._side_unresolved_enter_count
            ),
            "side_unresolved_confirm_block_count": int(
                self._side_unresolved_confirm_block_count
            ),
            # ---- reprice-replace pending flags ----
            # True when a cancel-and-replace cycle is in progress on
            # that side. Stuck True with no in-flight cancel/place is
            # a wedge fingerprint.
            "reprice_replace_pending": {
                "bid": bool(
                    self._quote_reprice_replace_pending.get(Side.BUY, False)
                ),
                "ask": bool(
                    self._quote_reprice_replace_pending.get(Side.SELL, False)
                ),
            },
            # ---- cancel-pending-since (per side) ----
            # Stamped when the bot issues a cancel; cleared on
            # terminal status (canceled / filled / explicit clear in
            # ``_clear_side_unresolved``). Stuck non-None means the
            # cancel never reached terminal state in local view.
            "cancel_pending_since_s": {
                side.value.lower(): (
                    round(now_m - self._cancel_pending_since_mono[side], 3)
                    if self._cancel_pending_since_mono.get(side) is not None
                    else None
                )
                for side in (Side.BUY, Side.SELL)
            },
            # ---- post-only-cross cooldown (per side) ----
            # Armed by the v1.4.26 fast-cancel detector + by explicit
            # 51604 sCode responses. The cooldown timestamp can
            # extend on the auto-release-blocked path in
            # ``_is_side_unresolved``; long-running cooldowns indicate
            # the extension logic is in a loop.
            "post_only_cross_cooldown_remaining_s": {
                side.value.lower(): round(
                    max(
                        0.0,
                        self._post_only_cross_cooldown_until.get(side, 0.0)
                        - now_m,
                    ),
                    3,
                )
                for side in (Side.BUY, Side.SELL)
            },
            # ---- last-outbound-replace timestamps (per side) ----
            # Used to throttle reprice cancel→place dispatch under
            # high churn. A stale value paired with no in-flight
            # work suggests the throttle path took a non-terminating
            # branch.
            "last_outbound_replace_age_s": {
                side.value.lower(): (
                    round(now_m - self._last_outbound_replace_mono.get(side, 0.0), 3)
                    if self._last_outbound_replace_mono.get(side, 0.0) > 0
                    else None
                )
                for side in (Side.BUY, Side.SELL)
            },
            # ---- outbound dispatcher latches ----
            # The (side, level_idx) sets that gate re-dispatch on
            # the same rung. An entry without matching in-flight
            # work is the dispatcher-side wedge fingerprint.
            "outbound_inflight_count": int(
                snap.get("inflight_count", 0)
            ),
            "outbound_active_place_sides": list(
                snap.get("active_place_sides", []) or []
            ),
            "outbound_active_cancel_sides": list(
                snap.get("active_cancel_sides", []) or []
            ),
            "outbound_active_amend_sides": list(
                snap.get("active_amend_sides", []) or []
            ),
            "outbound_queue_depth": int(
                snap.get("action_dispatch_queue_depth", 0)
            ),
            # ---- orphan-cancel dedup cache ----
            # Large cache size with sustained orphan_cancel_dedup_
            # skips suggests reconcile is finding the same phantom
            # repeatedly and the bot keeps deduping the cancel
            # attempt instead of confirming the orphan is gone.
            "recent_orphan_cancels_size": len(self._recent_orphan_cancels),
            "orphan_cancel_dedup_skips": int(
                self._orphan_cancel_dedup_skips
            ),
            # ---- derived: idle since last place ----
            "execution_idle_s": execution_idle_s,
            "snapshot_age_s": 0.0,
            # ---- v1.4.50 BUG-025 follow-on: working_order status per side ----
            # The single most useful field for diagnosing the wedge
            # class: what status is each side's WorkingOrder actually
            # in right now? A WO stuck in NEW_LOCAL or DESYNC, with no
            # in-flight transport activity, is a smoking gun. The
            # ``orchestrate_decision_history`` block below pairs with
            # this to show the FULL sequence of skip-branches that
            # held during the wedge.
            "working_order_status": {
                side.value.lower(): self._working_order_status_summary(side)
                for side in (Side.BUY, Side.SELL)
            },
            # ---- v1.4.50 BUG-025 follow-on: decision-trace counters ----
            # ``orchestrate_decision_counts`` is the cumulative
            # (side, branch, action) histogram since session start.
            # Useful for "how often did materiality_below_threshold
            # fire?" without scrubbing logs.
            "orchestrate_decision_counts": dict(self._orchestrate_decision_counts),
            # ``orchestrate_decision_history`` is the per-side ring
            # buffer of the most recent N decisions. The silent-wedge
            # detector inlines this directly into the event payload so
            # the postmortem sees EXACTLY which branch held during the
            # wedge — no more guessing which silent-return path is the
            # culprit. Truncated to the last 30 per side for the
            # snapshot (full buffer available on the executor object;
            # the snapshot is bandwidth-conscious).
            "orchestrate_decision_history": {
                side.value.lower(): list(
                    self._orchestrate_decision_history[side]
                )[-30:]
                for side in (Side.BUY, Side.SELL)
            },
            # ---- v1.4.50 BUG-025 follow-on: stage-place skip reasons ----
            # Cached per-side reason from the most-recent
            # ``_stage_place_order_local`` call that returned None.
            # Cleared on the next successful stage. When a wedge fires
            # and these are populated, you instantly see which one of
            # the seven silent-return paths in stage_place fired.
            "last_stage_place_skip_reason": dict(self._last_stage_place_skip_reason),
            # ---- v1.4.53 wedge-investigation: maybe_refresh_quotes skips ----
            # The 2026-05-18 v1.4.52-160650 wedge proved the bug isn't
            # in ``_orchestrate`` — it's in one of the early-return
            # paths UPSTREAM (no_write_access, cancel_resting_for_risk,
            # no_quote_hold_resting, residual_flatten_requested,
            # ladder_dispatch_no_levels). The ``orchestrate_decision_history``
            # is empty during a wedge because ``_orchestrate`` never
            # runs. The buffer below shows which gate held instead.
            "quote_refresh_skip_counts": dict(self._quote_refresh_skip_counts),
            "quote_refresh_skip_history": list(
                self._quote_refresh_skip_history
            )[-30:],
            # ---- v1.4.60 wedge-elimination Phase 5: reducing-side bypass status ----
            # Surface per-side: is this side the REDUCER (given current
            # position) and which cooldown bypass flags are enabled.
            # The dashboard + postmortem use this to verify the
            # invariant "reducing side is never fully suppressed by
            # defensive cooldowns" is operating as designed. The
            # v1.4.59-260518-173229 wedge would have been instantly
            # diagnosable from this surface.
            "reducing_side_bypass": {
                "position_qty": float(getattr(self._state.position, "position_qty", 0.0) or 0.0),
                "buy_is_reducer": bool(self._is_reducing_side(Side.BUY)),
                "sell_is_reducer": bool(self._is_reducing_side(Side.SELL)),
                "adverse_side_pause_bypass_enabled": bool(
                    getattr(
                        self._settings,
                        "adverse_side_pause_bypass_reducing_side",
                        True,
                    )
                ),
                "post_only_cross_cooldown_bypass_enabled": bool(
                    getattr(
                        self._settings,
                        "post_only_cross_cooldown_bypass_reducing_side",
                        True,
                    )
                ),
            },
            # ---- v1.4.56 wedge-elimination Phase 2: risk-exec state machine ----
            # Current state of the risk-action execution state machine
            # (NORMAL / CANCELLING / SUPPRESSED). Edge-triggers cancel-
            # all so a sustained CANCEL_ALL risk action doesn't burn
            # rate-limit budget on per-tick re-iteration. See
            # ``app/enums.py:RiskExecState``.
            "risk_exec_state": self._risk_exec_state.value,
            "risk_exec_state_age_s": (
                round(now_m - self._risk_exec_state_entered_mono, 3)
                if self._risk_exec_state_entered_mono > 0
                else None
            ),
            "risk_exec_state_last_risk_action": (
                self._risk_exec_state_last_risk_action.value
                if self._risk_exec_state_last_risk_action is not None
                else None
            ),
            "risk_exec_state_transition_counts": dict(
                self._risk_exec_state_transition_counts
            ),
            # v1.4.93 wedge-elimination-cleanup Phase 6D.3 —
            # ``wedge_episode_count_session`` is the operator-facing
            # one-liner answer to "did the bot wedge at all this
            # session?". Counts transitions FROM NORMAL into the
            # non-quotable cohort (CANCELLING / SUPPRESSED). Each
            # episode is one entry into a non-trivial risk state;
            # re-arming transitions (CANCELLING → CANCELLING) don't
            # count. Derived from the existing transition-count dict,
            # no new state. Exposed in Telegram /status via this key.
            "wedge_episode_count_session": sum(
                c for key, c in self._risk_exec_state_transition_counts.items()
                if key.startswith("NORMAL->") and (
                    key.split("->", 1)[1] in ("CANCELLING", "SUPPRESSED")
                )
            ),
            # ---- v1.4.68 Phase 1A: WS→WO matcher race diagnostics ----
            # The v1.4.67-260518-195033 wedge mechanism: WS ``live``
            # event arrives before the place-response handler commits
            # the OID; matcher fails; event is dropped; WO stays in
            # SENT for tens of seconds. Phase 1A buffers the events
            # by cloid and drains on place-response commit.
            #
            # Operator-facing rule (Phase 7C acceptance):
            #   ``ws_event_unmatched_to_local_wo_total`` should be ~0
            #   on a healthy day. If non-zero, the matcher missed AT
            #   LEAST ONCE — the buffer rescued state but the bug
            #   class is still present in the matcher.
            #   ``ws_event_buffered_replay_applied_total`` is the
            #   number of rescued events. If it climbs in lockstep
            #   with ``unmatched_total``, the buffer is doing its
            #   job. If ``unmatched`` climbs faster than ``applied``,
            #   events are timing out in the buffer — diagnose.
            "ws_event_unmatched_to_local_wo_total": int(
                self._ws_event_unmatched_to_local_wo_total
            ),
            "ws_event_buffered_replay_applied_total": int(
                self._ws_event_buffered_replay_applied_total
            ),
            "ws_event_buffer_dropped_oldest_total": int(
                self._ws_event_buffer_dropped_oldest_total
            ),
            "ws_event_buffer_swept_stale_total": int(
                self._ws_event_buffer_swept_stale_total
            ),
            "ws_event_buffer_dropped_full_total": int(
                self._ws_event_buffer_dropped_full_total
            ),
            "ws_event_buffer_current_size": sum(
                len(v) for v in self._pending_ws_events_by_cloid.values()
            ),
            # ---- v1.4.69 Phase 1B: hydration dedup ----
            # ``hydration_merged_existing_total`` — number of times
            # ``_hydrate_working_from_exchange`` found a non-terminal
            # local WO with matching (oid, cloid) and MERGED into it
            # instead of creating a duplicate row. Non-zero = the
            # duplicate-WO bug was averted in flight (good, but the
            # underlying race that caused the dedup to be needed is
            # still happening).
            # ``hydration_skipped_terminal_match_total`` — found a
            # TERMINAL match and skipped. Benign; REST is showing a
            # freshly-cancelled order.
            "hydration_merged_existing_total": int(
                self._hydration_merged_existing_total
            ),
            "hydration_skipped_terminal_match_total": int(
                self._hydration_skipped_terminal_match_total
            ),
            # ---- v1.4.74 Phase 2A: gates → desired-state layer ----
            # Counts of ticks where each gate suppressed at least one
            # slot via the desired-state fold. Operator can compare
            # against ``BUY:side_unresolved:return_silent`` /
            # ``BUY:reconciler:place_stage_returned_none`` from
            # ``orchestrate_decision_counts`` — the silent-veto
            # counts should drop to near-zero now that the gates are
            # visible upstream.
            #
            # ``gate_phase2a_invariant_violation_total`` should stay
            # at 0 in production. Non-zero indicates a future code
            # path emitted a PlaceAction/AmendAction for an
            # unresolved side without going through the fold —
            # would have been silently veto'd pre-Phase-2A; now it's
            # a loud invariant-violation log + counter.
            "gate_side_unresolved_applied_total": int(
                self._gate_side_unresolved_applied_total
            ),
            "gate_adverse_side_pause_applied_total": int(
                self._gate_adverse_side_pause_applied_total
            ),
            "gate_post_only_cross_cooldown_applied_total": int(
                self._gate_post_only_cross_cooldown_applied_total
            ),
            # Phase 2K.11 (v1.4.160) — post-only-cross favorable-exit
            # attribution. Operator reads
            # ``cleared_via_favorable / (favorable + ceiling)`` to
            # see how often the "touch moved 1 tick away" predicate
            # cleared the cooldown before the 2.5 s timer ran out.
            "post_only_cross_cooldown_cleared_via_favorable_total": int(
                self._post_only_cross_cooldown_cleared_via_favorable_total
            ),
            "post_only_cross_cooldown_cleared_via_ceiling_total": int(
                self._post_only_cross_cooldown_cleared_via_ceiling_total
            ),
            # v1.4.169 Phase 2I — per-order amend rate-defence
            # counters. Operator sees these in ``executor_state``;
            # a non-zero rate is the signal that the bot's reprice
            # tick was generating an amend faster than the venue
            # could ack the previous one, OR that the per-order 1-s
            # cap saved the bot from a runaway loop (v1.4.102
            # incident shape).
            "amend_tick_flicker_suppressed_total": int(
                getattr(
                    self._state,
                    "amend_tick_flicker_suppressed_total",
                    0,
                )
                or 0
            ),
            "amend_rate_throttle_suppressed_total": int(
                getattr(
                    self._state,
                    "amend_rate_throttle_suppressed_total",
                    0,
                )
                or 0
            ),
            "gate_phase2a_invariant_violation_total": int(
                self._gate_phase2a_invariant_violation_total
            ),
            # ---- v1.4.75 Phase 2B: stale-ghost reaper ----
            # Cumulative counts of WOs the reaper has force-terminalled.
            # In production these should be RARE (single digits per day).
            # A sustained non-zero rate indicates an upstream legitimate
            # recovery path is failing and the reaper is papering over
            # it. Use these as a leading indicator: spike = investigate
            # the upstream path.
            "reaper_cancel_pending_reaped_total": int(
                self._reaper_cancel_pending_reaped_total
            ),
            "reaper_desync_removed_total": int(
                self._reaper_desync_removed_total
            ),
            "reaper_sent_rejected_total": int(
                self._reaper_sent_rejected_total
            ),
            # v1.4.92 Phase 4A cutover — ``quote_build_result_unconsumed_field_total``
            # REMOVED. The runtime audit it tracked is gone; the typed
            # ``BuildCommand`` sum-type prevents the regression class
            # structurally. Dashboards / postmortem readers that
            # consumed this key must drop it.
        }

    def _working_order_status_summary(self, side: Side) -> dict[str, Any]:
        """v1.4.50 BUG-025: compact per-side WO status block for the
        executor_state snapshot. Reads the inside-rung slot (level_idx=0)
        — that's the one the wedge detector cares about. Outer rungs are
        ladder-only and the wedge class is on the inside rung.
        """
        with self._state._lock:
            wo = self._state.get_working_order(side, 0)
        if wo is None:
            return {
                "status": "none",
                "level_idx": 0,
                "exchange_oid": None,
                "px": None,
                "sz": None,
                "age_s": None,
            }
        now_m = self._clock.monotonic()
        ts_created = float(getattr(wo, "ts_created_mono", 0.0) or 0.0)
        return {
            "status": wo.status.value,
            "level_idx": int(getattr(wo, "level_idx", 0) or 0),
            "exchange_oid": str(wo.order_id_exchange) if wo.order_id_exchange else None,
            "px": float(wo.price),
            "sz": float(wo.size),
            "age_s": round(now_m - ts_created, 3) if ts_created > 0 else None,
        }

    def _bump_transport_counters(self) -> None:
        mode = str(getattr(self._client, "last_exchange_transport_mode", "http"))
        if mode == "ws":
            self._ws_action_send_count += 1
        else:
            self._http_action_send_count += 1
        with self._state._lock:
            self._state.outbound_ws_action_send_count = self._ws_action_send_count
            self._state.outbound_http_action_send_count = self._http_action_send_count
            self._state.outbound_transport_mode_current = mode

    def _publish_outbound_trace_metrics(self, tr: Optional[OutboundActionTrace]) -> None:
        if tr is None:
            return
        m = tr.to_metrics_dict(now_perf=time.perf_counter())
        with self._state._lock:
            pit = m.get("place_intent_to_ack_ms")
            if pit is not None:
                self._state.outbound_place_intent_to_ack_ms = float(pit)
            bw = m.get("batch_wait_ms")
            if bw is not None:
                self._state.outbound_action_batch_wait_ms = float(bw)
            sg = m.get("signing_ms")
            if sg is not None:
                self._state.outbound_signing_latency_ms = float(sg)

    def _working_order_by_local_id(self, local_id: str) -> Optional[WorkingOrder]:
        # 1.3.130 multi-rung Phase 2: scan ALL rungs across both sides
        # rather than just the inside rungs. The transport intent path
        # (place / cancel workers) needs to resolve outer-rung WOs
        # whose order_id_local is on the intent. At N=1 every rung
        # is level_idx=0 — same coverage as the pre-Phase-2 path.
        with self._state._lock:
            for wo in self._state.all_working_orders():
                if wo.order_id_local == local_id:
                    return wo
        return None

    def _find_local_wo_by_oid_or_cloid(
        self,
        oid: Optional[int],
        cloid: Optional[str],
    ) -> Optional[WorkingOrder]:
        """Find a local WorkingOrder matching ``oid`` (preferred) or
        ``cloid``. Used by Phase 1B hydration dedup.

        v1.4.80 Phase 3A: delegates to ``OrderStore.find_by_oid_or_cloid``
        which maintains O(1) indexes. The pre-Phase-3A O(n) scan
        fallback survives inside the store for entries inserted by
        legacy direct-dict callers (eventually removed in Phase 3D).
        """
        store = getattr(self._state, "order_store", None)
        if store is not None:
            return store.find_by_oid_or_cloid(oid, cloid)
        # Defensive fallback — should never fire in production
        # (BotState constructs the store unconditionally).
        if oid is None and not cloid:
            return None
        with self._state._lock:
            all_wos = self._state.all_working_orders()
            if oid is not None:
                for wo in all_wos:
                    if wo.order_id_exchange == oid:
                        return wo
            if cloid:
                for wo in all_wos:
                    if (
                        not wo.order_id_exchange
                        and wo.client_order_id
                        and cloids_match(wo.client_order_id, cloid)
                    ):
                        return wo
        return None

    def _execute_place_intent(self, intent: PlaceTransportIntent) -> None:
        wo = self._working_order_by_local_id(intent.wo_order_id_local)
        if wo is None:
            return
        if wo.transport_intent_seq != intent.intent_seq:
            self._dropped_stale_intent_count += 1
            logger.debug(
                "place_intent_stale_skip local_id=%s intent_seq=%s wo_seq=%s",
                intent.wo_order_id_local,
                intent.intent_seq,
                wo.transport_intent_seq,
            )
            return
        if wo.status != OrderStatus.SENT:
            return
        qw = max(0.0, (self._clock.monotonic() - intent.enqueued_mono) * 1000.0)
        self._last_submit_queue_wait_ms = max(self._last_submit_queue_wait_ms, qw)
        tr = self._outbound_traces.get(wo.order_id_local)
        if tr is not None:
            tr.batch_selected_perf = time.perf_counter()
        self._complete_place_http(
            wo, quote_cycle_id=intent.quote_cycle_id, intent_seq=intent.intent_seq
        )

    def _execute_cancel_intent(self, intent: CancelTransportIntent) -> None:
        wo = self._working_order_by_local_id(intent.wo_order_id_local)
        if wo is None:
            return
        if wo.cancel_transport_seq != intent.intent_seq:
            self._dropped_stale_intent_count += 1
            logger.debug(
                "cancel_intent_stale_skip local_id=%s intent_seq=%s wo_seq=%s",
                intent.wo_order_id_local,
                intent.intent_seq,
                wo.cancel_transport_seq,
            )
            return
        if wo.status != OrderStatus.CANCEL_PENDING:
            return
        qw = max(0.0, (self._clock.monotonic() - intent.enqueued_mono) * 1000.0)
        self._last_submit_queue_wait_ms = max(self._last_submit_queue_wait_ms, qw)
        tr = self._outbound_traces.get(wo.order_id_local)
        if tr is not None:
            tr.batch_selected_perf = time.perf_counter()
        self._cancel_http_transport(wo)
        if tr is not None:
            tr.cancel_sent_perf = time.perf_counter()

    def _execute_cancel_batch_intents(
        self, intents: list[CancelTransportIntent]
    ) -> None:
        """Execute multiple cancels in a single OKX batch HTTP call
        (1.4.0 cancel-prio Phase 1b).

        Invoked by ``OutboundDispatchCoordinator`` when 2+ cancels for
        DISTINCT sides queue together — the typical full-reprice case
        (cancel BUY + cancel SELL simultaneously). Saves one full RTT
        per such cycle vs the per-intent path.

        Falls back to per-intent ``_cancel_http_transport`` calls when:
          * fewer than 2 valid intents survive the staleness filter
          * the batch HTTP itself raises (transport error)

        Per-row outcome handling mirrors ``_cancel_http_transport``
        verbatim (Phase 0.5 timestamps stamped, RTT tracker fed on
        success, ``cancel_race_lost_to_fill_total`` bumped on
        benign-missing rows, ``execution_errors`` bumped on errors).
        """
        # Resolve intents → WOs with the same staleness filter as
        # _execute_cancel_intent. We collect (intent, wo) pairs so the
        # response-side mapping can re-find each WO.
        valid_pairs: list[tuple[CancelTransportIntent, WorkingOrder]] = []
        for intent in intents:
            wo = self._working_order_by_local_id(intent.wo_order_id_local)
            if wo is None:
                continue
            if wo.cancel_transport_seq != intent.intent_seq:
                self._dropped_stale_intent_count += 1
                continue
            if wo.status != OrderStatus.CANCEL_PENDING:
                continue
            valid_pairs.append((intent, wo))

        # v1.4.34 cancel-pool routing completion: lowered both
        # batch-size thresholds (``valid_pairs``, ``valid_for_send``)
        # from ``< 2`` to ``< 1`` so 1-element batches dispatch
        # through ``cancel_batch_orders`` (CANCEL_BATCH pool, 300/2 s)
        # instead of falling back to single-cancel HTTP (CANCEL_SINGLE,
        # 60/2 s).
        #
        # Backstory: v1.4.33's dispatcher fix lowered ``len(cancels)``
        # to ``>= 1`` at the dispatch site, but this executor method
        # had its own internal ``< 2`` guard that re-fell-back to the
        # per-intent (single-endpoint) path on any 1-element batch.
        # Snapshot ``v1.4.33-260517-234512`` confirmed the issue:
        # ``cancel_single`` total = 19,071, ``cancel_batch`` total =
        # 68 — almost identical numbers to the pre-v1.4.33 state. The
        # dispatcher was correctly handing 1-element batches off, but
        # this function was bouncing them back to the single endpoint.
        # Fixing both thresholds here completes the v1.4.33 fix.
        if len(valid_pairs) < 1:
            # Nothing to cancel (all intents stale / WO already gone).
            return

        # Build the batch request body. OKX V5 accepts mixed
        # ordId / clOrdId per row. Prefer cloid (more reliable —
        # ordId may not be bound yet on the rare not-acked path,
        # though the defer-until-ack guard should prevent that here).
        symbol = self._settings.symbol
        refs: list[dict[str, str]] = []
        valid_for_send: list[tuple[CancelTransportIntent, WorkingOrder]] = []
        for intent, wo in valid_pairs:
            if wo.client_order_id:
                refs.append({"clOrdId": str(wo.client_order_id)})
            elif wo.order_id_exchange:
                refs.append({"ordId": str(wo.order_id_exchange)})
            else:
                # No identifier — shouldn't happen on the dispatcher
                # path; defer-until-ack should have parked it. Skip
                # defensively.
                logger.warning(
                    "cancel_batch_missing_identifier side=%s local_id=%s",
                    wo.side.value,
                    wo.order_id_local,
                )
                continue
            valid_for_send.append((intent, wo))

        if len(valid_for_send) < 1:
            # All valid pairs lacked usable identifiers (defensive —
            # the dispatcher's defer-until-ack guard should prevent
            # this, but it's not impossible for a hydrated WO with a
            # very stale state).
            return

        # 1.4.0 cancel-prio Phase 0.5: stamp ts_cancel_sent on every
        # WO in the batch just before the HTTP leaves. Same wall-clock
        # value for all rows (single send moment).
        sent_ts = self._clock.now_utc()
        for _intent, wo in valid_for_send:
            wo.ts_cancel_sent = sent_ts

        try:
            resp = self._client.cancel_batch_orders(symbol, refs)
        except Exception:
            logger.exception("cancel_batch_http_failed_falling_back_to_singles")
            self._state.bump_execution_errors("cancel_batch_http_transport_exception")
            # Fall back to per-cancel HTTPs. Each will re-stamp
            # ts_cancel_sent (slight time skew vs the batch attempt
            # — acceptable, the singles path is the source of truth
            # when batch fails).
            for _intent, wo in valid_for_send:
                self._cancel_http_transport(wo)
            return

        # Parse per-row outcomes. The interpreter returns the rows in
        # OKX's submission order; we also build a clOrdId → outcome
        # map for safety in case OKX ever reorders.
        from app.exchange.okx_responses import (
            interpret_okx_cancel_batch_response,
        )
        _top_kind, _top_detail, rows = interpret_okx_cancel_batch_response(
            resp
        )

        ack_ts = self._clock.now_utc()
        rows_by_cloid: dict[str, tuple[str, str]] = {}
        for r_kind, r_detail, _r_ord_id, r_cl_ord_id in rows:
            if r_cl_ord_id:
                rows_by_cloid[r_cl_ord_id] = (r_kind, r_detail)

        success_count = 0
        for idx, (_intent, wo) in enumerate(valid_for_send):
            # Match the row to this WO. Prefer cloid match; fall back
            # to positional index (OKX preserves submission order in
            # the response).
            kind: str
            detail: str
            cl_id = wo.client_order_id
            if cl_id and cl_id in rows_by_cloid:
                kind, detail = rows_by_cloid[cl_id]
            elif idx < len(rows):
                kind, detail = rows[idx][0], rows[idx][1]
            else:
                kind, detail = "error", "missing_row_for_wo"

            # Same updates as the single-cancel HTTP path
            # (_cancel_http_transport):
            wo.cancel_response_outcome = (kind or None)
            if detail:
                wo.cancel_response_detail = detail[:400]
            # 1.4.6 sticky cancel rejection recorder.
            if kind and kind != "success":
                try:
                    self._state.record_cancel_rejection(
                        outcome=kind,
                        detail=detail or "",
                        is_benign=_is_benign_cancel_outcome(kind, detail),
                        ts_iso=self._clock.now_utc().isoformat(),
                    )
                except Exception:
                    logger.exception(
                        "record_cancel_rejection_failed_batch"
                    )
            try:
                self._state.record_cancel_outcome(
                    outcome=kind or "unknown",
                    latency_ms=None,
                )
            except Exception:
                logger.exception(
                    "record_cancel_outcome_failed_batch"
                )

            if kind == "success":
                # Phase 0.5: ts_cancel_acked + RTT tracker only on
                # success.
                wo.ts_cancel_acked = ack_ts
                try:
                    rtt_ms = (
                        (ack_ts - wo.ts_cancel_sent).total_seconds()
                        * 1000.0
                    )
                    if rtt_ms >= 0 and wo.ts_cancel_sent is not None:
                        self._order_rtt_tracker.ingest(
                            rtt_ms=rtt_ms,
                            op="cancel",
                            outcome="accepted",
                        )
                except Exception:
                    logger.exception("cancel_rtt_ingest_failed_batch")
                success_count += 1
            elif kind == "benign_missing":
                # Same counter / log as the single path.
                self._state.cancel_race_lost_to_fill_total += 1
                logger.info(
                    "cancel_benign_missing side=%s exchange_oid=%s cloid=%s detail=%s (batch)",
                    wo.side.value,
                    wo.order_id_exchange,
                    (wo.client_order_id[:18] + "...") if wo.client_order_id else None,
                    detail[:300] if detail else "",
                )
            elif kind == "unexpected_gone":
                # v1.4.38 (2026-05-18) classifier-completeness fix:
                # mirror the single-cancel HTTP path's handling. The
                # ``unexpected_gone`` taxonomy (v1.3.120) means the
                # order IS gone but NOT via fill (OKX sCode 51400 /
                # 51401 / 51503). Bot transitions the WO locally,
                # logs at WARNING, bumps the dedicated
                # ``cancel_unexpected_gone_total`` counter, and DOES
                # NOT bump ``execution_errors``. Pre-v1.4.38 the
                # batch path fell into the ``else`` below and bumped
                # ``cancel_http_exchange_reject`` — which under the
                # v1.4.33 + v1.4.35 routing change (every cancel now
                # flows through this path) caused the v1.4.37
                # production self-kill on
                # `snapshots/v1.4.37-260518-075317-prod.okx.ton.usdt.perp/`.
                # Strict-mode honoured (operator opt-in to alert).
                #
                # v1.4.41 BUG-025 Phase 3 fix: REAP the local WO. The
                # pre-fix behaviour bumped the counter and logged, but
                # left the WO in ``working_orders[side][level_idx]`` in
                # CANCEL_PENDING status. The next orchestrate cycle
                # found it, re-cancelled, got 51400 again, re-stamped
                # side_unresolved cancel_pending_wait — infinite ping-
                # pong, deadlock-watchdog kill at idle=600 s. Captured
                # 2026-05-18 in v1.4.40-260518-093848 with 28k+
                # side_unresolved enters in 16 min on two zombie OIDs.
                self._state.cancel_unexpected_gone_total += 1
                strict_mode = bool(
                    getattr(
                        self._settings,
                        "cancel_unexpected_gone_strict_mode",
                        False,
                    )
                )
                logger.warning(
                    "cancel_unexpected_gone side=%s exchange_oid=%s "
                    "cloid=%s strict=%s detail=%s (batch)",
                    wo.side.value,
                    wo.order_id_exchange,
                    (wo.client_order_id[:18] + "...") if wo.client_order_id else None,
                    strict_mode,
                    detail[:400] if detail else "",
                )
                if strict_mode:
                    self._state.bump_execution_errors(
                        "cancel_http_unexpected_gone_strict"
                    )
                self._reap_wo_after_cancel_unexpected_gone(
                    wo, source="batch", detail=detail or ""
                )
            else:
                # error / transport — count + log + bump execution
                # errors. Reserved for genuine errors (e.g. 51008
                # insufficient margin) — NOT ``unexpected_gone``,
                # which is handled above.
                logger.warning(
                    "cancel_exchange_rejected side=%s kind=%s detail=%s (batch)",
                    wo.side.value,
                    kind,
                    detail[:400] if detail else "",
                )
                self._state.bump_execution_errors(
                    "cancel_http_exchange_reject"
                )

            # Phase 0b: persist AFTER the wire moment. The batch path
            # already sent the cancel; persisting now just records the
            # outcome.
            try:
                self.persist(wo)
            except Exception:
                logger.exception("cancel_batch_persist_failed")

        if success_count > 0:
            self._bump_transport_counters()

    def _execute_place_batch_intents(
        self, intents: list[PlaceTransportIntent]
    ) -> None:
        """Execute multiple places in a single OKX batch HTTP/WS call
        (1.4.4 — the rate-limit fix work-unit).

        Invoked by ``OutboundDispatchCoordinator`` when 2+ places for
        distinct (side, level_idx) keys queue together — the typical
        case at N≥2 ladder during a full reprice. The wins:
          1. RTT compression: N places → 1 batch RTT.
          2. Rate-limit pool: /trade/batch-orders has a 300/2s budget
             on OKX standard, separate from /trade/order's 60/2s. The
             v1.4.2 row-level "Rate limit reached" rejections came
             from the latter pool's per-instrument flow limit.

        Falls back to per-intent ``_complete_place_http`` calls when:
          * fewer than 2 valid intents survive the staleness filter
            (single-order path is simpler than a 1-element batch)
          * the batch wire call itself raises a transport error

        Per-row outcome handling mirrors the single-place path's
        post-response branches — accept (bind ordId, ACK), exchange
        reject (REJECTED + cooldown), transport reject (leave SENT,
        watchdog retries), unconfirmed (REJECTED + kill flow when
        ``strict_place_unconfirmed_kill``). Each row is independent;
        a partial batch (some rows succeed, others fail) is fully
        supported because OKX's deterministic clOrdId echo lets us
        correlate every row to its parent WO unambiguously.
        """
        # Staleness filter — same pattern as _execute_cancel_batch_intents
        valid_pairs: list[tuple[PlaceTransportIntent, WorkingOrder]] = []
        for intent in intents:
            wo = self._working_order_by_local_id(intent.wo_order_id_local)
            if wo is None:
                continue
            if wo.transport_intent_seq != intent.intent_seq:
                self._dropped_stale_intent_count += 1
                continue
            if wo.status != OrderStatus.SENT:
                continue
            valid_pairs.append((intent, wo))

        # 1.4.6: when ``BATCH_PLACES_ALWAYS=true`` is set, route every
        # batch — including size-1 — through the batch endpoint. This
        # keeps the place hot path on /trade/batch-orders (300/2s
        # pool) instead of mixing in /trade/order (60/2s pool) for
        # size-1 flushes. The dashboard's "ACTIONS" counter was 1:1
        # with "NEW" pre-1.4.6 because this fallback was firing for
        # most flushes — defeating the always-batch flag.
        always_batch = bool(
            getattr(self._settings, "batch_places_always", False)
        )
        min_to_batch = 1 if always_batch else 2
        if len(valid_pairs) < min_to_batch:
            # Degraded — fall back to per-intent path. Each call
            # handles its own state updates and timing.
            for intent, _wo in valid_pairs:
                self._execute_place_intent(intent)
            return
        if len(valid_pairs) == 0:
            return

        # Pre-wire setup (per-WO trace + place-to-fill ratio + watchdog).
        # Done as a loop so each WO's lifecycle metadata is correct;
        # the actual wire call below is single-shot.
        t_pre = self._clock.monotonic()
        with self._state._lock:
            self._state.last_place_attempt_ts_mono = t_pre
            # v1.4.42: broader "any outbound attempt" tracker used by
            # the watchdog + silent-wedge detector. See state.py docstring.
            self._state.last_outbound_attempt_ts_mono = t_pre
            self._state.session_place_attempt_count += len(valid_pairs)
        for _intent, wo in valid_pairs:
            self._state.place_to_fill_ratio_tracker.note_place()
            try:
                self._state.order_trace.begin_order(
                    client_order_id=wo.client_order_id or "",
                    side=wo.side.value,
                    price=float(wo.price),
                    size_base=float(wo.size),
                )
            except Exception:
                logger.exception("order_trace_begin_failed_batch")

        # Build the batch payload.
        symbol = self._settings.symbol
        orders: list[dict[str, Any]] = []
        for _intent, wo in valid_pairs:
            orders.append(
                {
                    "is_buy": wo.side == Side.BUY,
                    "sz": float(wo.size),
                    "limit_px": float(wo.price),
                    "client_order_id": wo.client_order_id,
                    "reduce_only": bool(getattr(wo, "reduce_only", False)),
                }
            )

        # Per-WO trace stamps. We capture send-time once for the whole
        # batch; ack-time is captured after the call returns.
        traces: list[Any] = []
        for _intent, wo in valid_pairs:
            tr = self._outbound_traces.get(wo.order_id_local)
            traces.append(tr)
        t_net = time.perf_counter()
        for tr in traces:
            if tr is not None:
                tr.sign_start_perf = t_net
                tr.transport_send_perf = t_net

        try:
            resp = self._client.batch_place_post_only_limit(symbol, orders)
        except AttributeError:
            # Adapter doesn't expose batch_place — fall back to singles.
            # This is the "Binance / older adapters" path; will not
            # fire on OKX where the method is defined.
            logger.info(
                "batch_place_unsupported_falling_back_to_singles"
            )
            for intent, _wo in valid_pairs:
                self._execute_place_intent(intent)
            return
        except Exception:
            logger.exception("batch_place_http_failed_falling_back_to_singles")
            self._state.bump_execution_errors(
                "batch_place_http_transport_exception"
            )
            for intent, _wo in valid_pairs:
                self._execute_place_intent(intent)
            return

        ack_t = time.perf_counter()
        rtt_ms = (ack_t - t_net) * 1000.0
        self._last_tick_order_submit_rtt_ms = max(
            float(self._last_tick_order_submit_rtt_ms), float(rtt_ms)
        )
        for tr in traces:
            if tr is not None:
                tr.sign_end_perf = ack_t
                tr.transport_write_done_perf = ack_t
                tr.transport_mode = str(
                    getattr(self._client, "last_exchange_transport_mode", "http")
                )
        self._bump_transport_counters()

        # Parse the response. The 1.4.4 batch interpreter correctly
        # classifies row-level "Rate limit reached" as
        # transport_rejected (the fix for the v1.4.2 misclassify).
        from app.exchange.okx_responses import (
            interpret_okx_place_batch_response,
        )
        top_kind, top_detail, rows = interpret_okx_place_batch_response(resp)
        # When the envelope itself is transport (auth / 50011), the
        # batch as a whole fails; the WOs stay in SENT and the
        # watchdog retries them. Bump execution errors so the bot
        # notices sustained envelope failures.
        if top_kind == "transport":
            logger.warning(
                "batch_place_transport_envelope detail=%s — leaving %d WOs in SENT",
                top_detail[:300],
                len(valid_pairs),
            )
            self._state.bump_execution_errors(
                "batch_place_envelope_transport"
            )
            return

        # Build a cloid → row map for safe correlation. OKX preserves
        # submission order in the response but we don't rely on that —
        # the deterministic clOrdId echo is unambiguous even on
        # reordered or partial responses.
        rows_by_cloid: dict[
            str, tuple[Optional[int], str, str]
        ] = {}
        for r_oid, r_kind, r_detail, r_cl_ord_id in rows:
            if r_cl_ord_id:
                rows_by_cloid[r_cl_ord_id] = (r_oid, r_kind, r_detail)

        row_rate_limit_hits = 0
        success_count = 0
        for idx, (intent, wo) in enumerate(valid_pairs):
            tr = traces[idx]
            cl_id = wo.client_order_id
            if cl_id and cl_id in rows_by_cloid:
                ex_oid, outcome, reason = rows_by_cloid[cl_id]
            elif idx < len(rows):
                ex_oid, outcome, reason, _ = rows[idx]
            else:
                ex_oid, outcome, reason = None, "unconfirmed", "missing_row_for_wo"

            wo.ts_place_response = self._clock.now_utc()
            wo.place_response_outcome = (outcome or None)
            if reason:
                wo.place_response_detail = reason[:400]
            try:
                self._state.order_trace.record_place_response(
                    client_order_id=wo.client_order_id or "",
                    outcome=outcome,
                    detail=reason,
                    order_id_exchange=ex_oid,
                )
            except Exception:
                logger.exception("order_trace_place_response_failed_batch")
            # 1.4.6 sticky rejection recorder — never wiped (parallel
            # to the single-place path's recorder). See
            # ``BotState.record_place_rejection``.
            if outcome and outcome != "accepted":
                try:
                    self._state.record_place_rejection(
                        outcome=outcome,
                        detail=reason or "",
                        is_benign=_is_benign_place_outcome(outcome, reason),
                        ts_iso=self._clock.now_utc().isoformat(),
                    )
                except Exception:
                    logger.exception(
                        "record_place_rejection_failed_batch"
                    )
            try:
                self._state.record_place_outcome(
                    outcome=outcome or "unknown",
                    # Batch RTT applies uniformly to every WO in the
                    # batch — same wire moment. accepted rows only,
                    # to keep the latency distribution clean.
                    latency_ms=rtt_ms if outcome == "accepted" else None,
                )
            except Exception:
                logger.exception(
                    "record_place_outcome_failed_batch"
                )

            if ex_oid == 0:
                ex_oid = None
                outcome = "unconfirmed"
                reason = reason or "place_ack_without_oid"

            if outcome == "accepted" and ex_oid is not None:
                wo.order_id_exchange = ex_oid
                self._clear_sent_ambiguous_polls(wo)
                if tr is not None:
                    tr.exchange_ack_perf = ack_t
                    tr.local_order_open_perf = ack_t
                transition(wo, OrderStatus.ACKED)
                # v1.4.68 Phase 1A: drain any WS events that arrived
                # before this place response committed the OID. The
                # buffer is keyed by cloid; events were stashed in
                # ``_handle_private_order_update``'s no-WO branch.
                # Draining HERE (immediately after the SENT→ACKED
                # transition) closes the WS→WO race that caused 33-s
                # ts_ack lag in snapshot v1.4.67-260518-195033.
                try:
                    self._drain_pending_ws_events_for_cloid(wo.client_order_id)
                except Exception:
                    logger.exception(
                        "drain_pending_ws_events_failed_batch_cloid=%s",
                        (wo.client_order_id or "")[:24],
                    )
                try:
                    self._order_rtt_tracker.ingest(
                        rtt_ms=rtt_ms,
                        op="place",
                        outcome="accepted",
                    )
                except Exception:
                    logger.exception("order_rtt_ingest_failed_batch")
                try:
                    if wo.ts_ack is not None:
                        self._state.fill_buckets.note_ack(
                            order_id_exchange=ex_oid,
                            ts_ack=wo.ts_ack,
                        )
                except Exception:
                    logger.exception("fill_buckets_note_ack_failed_batch")
                self._state.note_first_place_latency()
                # Flush any deferred cancel — same logic as the single
                # path. The order is ACKED now so a cancel-by-cloid
                # against ex_oid is safe.
                if wo.cancel_pending_after_ack:
                    wo.cancel_pending_after_ack = False
                    self.persist(wo)
                    logger.info(
                        "cancel_flushing_post_ack_batch side=%s oid=%s cloid=%s",
                        wo.side.value,
                        wo.order_id_exchange,
                        (wo.client_order_id[:18] + "...")
                        if wo.client_order_id
                        else None,
                    )
                    self._enqueue_cancel_quote_path(
                        wo, trigger_reason=wo.cancel_trigger_reason
                    )
                else:
                    self.persist(wo)
                success_count += 1
                continue

            if outcome == "transport_rejected":
                # v1.4.64 wedge-elimination follow-up: the venue
                # NEVER accepted this place. The order does not
                # exist anywhere — local state must reflect that.
                #
                # Pre-v1.4.64 the WO was left in SENT with the
                # comment "the cancel-pending watchdog (or the next
                # quote cycle's diff) will retry." That comment was
                # a LIE: there is no cancel-pending watchdog for
                # SENT status, and the reconciler treats SENT as
                # in_flight → NoOp forever. The only recovery was
                # the (formerly 120 s, then 15 s, now 1 s)
                # ``sent_order_unresolved_timeout`` — which produced
                # a 136-second stuck-SENT in snapshot
                # v1.4.61-260518-181059 (a rate-limited SELL place
                # whose WO stayed in SENT until the timeout fired).
                #
                # Correct behaviour: transition to REJECTED + release
                # slot. This matches the ``exchange_rejected`` branch
                # below AND the single-place HTTP path at line ~6352
                # (which already correctly handles both
                # ``exchange_rejected`` and ``transport_rejected``
                # identically). The asymmetry between batch and
                # single paths was the bug.
                #
                # After REJECTED + slot release, the next quote
                # tick's reconciler sees ``cur=terminal`` → emits
                # PlaceAction → fresh place. The dispatcher's
                # ``okx_row_rate_limit_total`` counter still
                # increments so rate-limit-driven retries are
                # observable in the dashboard.
                if reason and "rate_limit" in reason.lower():
                    row_rate_limit_hits += 1
                logger.info(
                    "batch_place_row_transport_rejected side=%s cloid=%s reason=%s",
                    wo.side.value,
                    (wo.client_order_id[:18] + "...")
                    if wo.client_order_id
                    else None,
                    (reason or "")[:300],
                )
                transition(
                    wo,
                    OrderStatus.REJECTED,
                    (reason or "transport_rejected")[:2000],
                )
                if wo.cancel_pending_after_ack:
                    wo.cancel_pending_after_ack = False
                self.persist(wo)
                self._release_working_slot_if_matches(wo)
                continue

            if outcome == "exchange_rejected":
                logger.warning(
                    "place_order_rejected_batch side=%s symbol=%s price=%s "
                    "size=%s outcome=%s reason=%s",
                    wo.side.value,
                    wo.symbol,
                    f"{wo.price:.12g}",
                    f"{wo.size:.12g}",
                    outcome,
                    (reason or "")[:400],
                )
                if is_post_only_immediate_match_rejection(reason or ""):
                    self._arm_post_only_cross_cooldown(
                        wo.side, rejected_price=wo.price
                    )
                    try:
                        self._state.quote_quality.note_post_only_cross_rejection()
                    except Exception:
                        logger.exception(
                            "note_post_only_cross_rejection_failed_batch"
                        )
                transition(wo, OrderStatus.REJECTED, (reason or outcome)[:2000])
                if wo.cancel_pending_after_ack:
                    wo.cancel_pending_after_ack = False
                self.persist(wo)
                self._release_working_slot_if_matches(wo)
                try:
                    self._state.order_trace.record_terminal(
                        client_order_id=wo.client_order_id or "",
                        status=OrderStatus.REJECTED.value,
                        reason=(reason or outcome)[:400],
                        source="place_response",
                    )
                except Exception:
                    logger.exception("order_trace_terminal_failed_batch")
                continue

            # outcome == "unconfirmed" (ex_oid None + not classified
            # as a known rejection). Same critical-kill flow as the
            # single path — guards against silent phantom-place
            # incidents (BUG-024). Row-level kill is per-WO because
            # different rows could have different outcomes.
            if self._settings.strict_place_unconfirmed_kill:
                raw_repr = self._format_raw_place_response_for_log(resp)
                payload: dict[str, Any] = {
                    "side": wo.side.value,
                    "symbol": wo.symbol,
                    "price": f"{wo.price:.12g}",
                    "size": f"{wo.size:.12g}",
                    "client_order_id": wo.client_order_id,
                    "order_id_local": wo.order_id_local,
                    "outcome": outcome,
                    "reason": (reason or "")[:1200],
                    "raw_response_truncated": raw_repr,
                    "batch_size": len(valid_pairs),
                }
                logger.critical(
                    "place_response_unconfirmed_critical_batch "
                    "side=%s symbol=%s price=%s size=%s cloid=%s reason=%s",
                    wo.side.value,
                    wo.symbol,
                    f"{wo.price:.12g}",
                    f"{wo.size:.12g}",
                    wo.client_order_id,
                    (reason or "")[:400],
                )
                self._state.place_unconfirmed_critical_total += 1
                try:
                    self._storage.insert_bot_event(
                        self._clock.now_utc().isoformat(),
                        EventSeverity.CRITICAL.value,
                        "place_response_unconfirmed",
                        f"unconfirmed batch-place row — venue gave neither "
                        f"clean accept nor reject for side={wo.side.value} "
                        f"price={wo.price} size={wo.size} cloid="
                        f"{wo.client_order_id}; order may be live on venue.",
                        payload,
                    )
                except Exception:
                    logger.exception(
                        "place_unconfirmed_bot_event_failed_batch"
                    )
                transition(
                    wo,
                    OrderStatus.REJECTED,
                    f"unconfirmed_critical:{(reason or outcome)[:200]}",
                )
                if self._request_kill_fn is not None:
                    try:
                        self._request_kill_fn(
                            "place_response_unconfirmed", payload
                        )
                    except Exception:
                        logger.exception(
                            "request_kill_fn_raised_on_unconfirmed_batch"
                        )
                else:
                    with self._state._lock:
                        self._state.killed = True
                        self._state.kill_reason = (
                            "place_response_unconfirmed"
                        )
                self.persist(wo)
                if wo.status == OrderStatus.REJECTED:
                    self._release_working_slot_if_matches(wo)
                continue

            # Non-strict mode: leave in SENT, let reconcile clean up.
            self.persist(wo)

        if row_rate_limit_hits > 0:
            try:
                self._client.bump_row_rate_limit_counter(row_rate_limit_hits)
            except AttributeError:
                pass  # adapter doesn't expose the counter

    # ------------------------------------------------------------------
    # amend-prio Phase 3 (v1.4.16) — amend executor + decision logic
    # ------------------------------------------------------------------

    def _count_amend_pending_unlocked(self) -> int:
        """Count working orders currently in AMEND_PENDING. Caller
        must hold ``self._state._lock``. Used by the high-watermark
        tracking (Phase 4 amend rollout counter).

        Reads ``_working_orders`` directly rather than via
        ``all_working_orders()``, which would re-acquire the lock and
        deadlock on the reentrant guard."""
        n = 0
        wos = getattr(self._state, "_working_orders", None)
        if wos is None:
            return 0
        for _side, rungs in wos.items():
            for _idx, wo in rungs.items():
                if wo.status == OrderStatus.AMEND_PENDING:
                    n += 1
        return n

    def _execute_amend_intent(self, intent: PlaceTransportIntent) -> None:
        """Single-amend executor. Delegates to the batch path with a
        1-element list, mirroring the v1.4.6 always-batch convention
        for places. Keeps every amend on the
        ``/trade/amend-batch-orders`` rate-limit pool (separate from
        the place pools), which is the whole point of amend-on-reprice.
        """
        self._execute_amend_batch_intents([intent])

    def _execute_amend_batch_intents(
        self, intents: list[PlaceTransportIntent]
    ) -> None:
        """Execute one or more amends via the OKX
        ``/trade/amend-batch-orders`` endpoint.

        Each ``intent`` carries the local WO id; the WO holds the
        target px/sz under ``amend_target_px`` / ``amend_target_sz``.
        On success the target commits into ``wo.price`` / ``wo.size``
        and the WO transitions AMEND_PENDING → ACKED (or PARTIAL if
        it was PARTIAL before — but Phase 3 only enters from ACKED so
        this branch is reserved for Phase 6 / partial-fill extension).

        Per-row outcomes (see ``plans/amend-prio.md`` Phase 3):

        * ``accepted``           → AMEND_PENDING → ACKED; commit target
                                   into price/size; clear target_*
        * ``below_filled``       → AMEND_PENDING → CANCEL_PENDING; emit
                                   a follow-up cancel (cancel-then-place
                                   fallback fires next cycle)
        * ``order_gone``         → AMEND_PENDING → CANCELED; release
                                   the working slot
        * ``exchange_rejected``  → AMEND_PENDING → ACKED (original
                                   survives); if post_only_would_cross,
                                   arm the side's cooldown
        * ``transport_rejected`` → AMEND_PENDING → ACKED (original
                                   survives); the next reprice cycle
                                   will re-attempt if still warranted
        * ``unconfirmed``        → same strict-place-unconfirmed-kill
                                   flow as ``_execute_place_batch_intents``
        """
        # Staleness filter — only act on WOs whose amend_intent_seq
        # still matches the intent. The submit-side bumped this each
        # time _enqueue_amend_quote_path fired; an older intent that
        # landed after a newer one was queued must drop.
        valid_pairs: list[tuple[PlaceTransportIntent, WorkingOrder]] = []
        for intent in intents:
            wo = self._working_order_by_local_id(intent.wo_order_id_local)
            if wo is None:
                continue
            if int(getattr(wo, "amend_intent_seq", 0) or 0) != int(
                intent.intent_seq
            ):
                self._dropped_stale_intent_count += 1
                continue
            if wo.status != OrderStatus.AMEND_PENDING:
                continue
            if not wo.order_id_exchange:
                # Safety: can't amend an order with no ordId. The
                # state machine should not produce this (AMEND_PENDING
                # requires ACKED, which requires an ordId), but guard
                # defensively so a bug elsewhere doesn't manifest as
                # a venue 400.
                logger.error(
                    "amend_intent_missing_ord_id wo=%s — reverting to ACKED",
                    wo.order_id_local,
                )
                self._revert_amend_to_acked(
                    wo, outcome="exchange_rejected", detail="missing_ord_id"
                )
                continue
            valid_pairs.append((intent, wo))

        if not valid_pairs:
            return

        # Build the amend payload.
        symbol = self._settings.symbol
        amends: list[dict[str, Any]] = []
        for _intent, wo in valid_pairs:
            row: dict[str, Any] = {"ord_id": str(wo.order_id_exchange)}
            if wo.amend_target_px is not None:
                row["new_px"] = float(wo.amend_target_px)
            if wo.amend_target_sz is not None:
                row["new_sz"] = float(wo.amend_target_sz)
            # Deterministic reqId for natural dedup at the venue.
            # OKX V5 ``reqId`` accepts ``[a-zA-Z0-9]{1,32}`` — STRICTLY
            # alphanumeric, no separators. Two earlier attempts failed:
            #   v1.4.17 used ``{ordId}-{seq}`` (dash) → sCode 51000
            #   v1.4.18 used ``a{ordId}_{seq}`` (underscore) → sCode 51000
            # Both surfaced the same "Parameter reqId  error" (note the
            # double space — OKX's templated error with the rendered
            # value omitted, characteristic of a parameter-validation
            # reject). The deploy snapshot at v1.4.18 confirmed
            # underscores are also rejected.
            #
            # Final format: ``a{ordId}s{seq}`` — letter ``s`` as the
            # alphanumeric separator. Total length: 1 + ≤19 + 1 + ≤10
            # = 31-char ceiling, fits the 32-char OKX cap. The leading
            # ``a`` keeps the value non-numeric (so OKX never confuses
            # it with an integer ordId); the ``s`` between ordId and
            # seq preserves a deterministic split for the correlation
            # key without using a separator character that OKX rejects.
            row["req_id"] = (
                f"a{wo.order_id_exchange}s"
                f"{int(getattr(wo, 'amend_intent_seq', 0) or 0)}"
            )
            amends.append(row)

        t_net = time.perf_counter()
        try:
            resp = self._client.amend_batch_orders(symbol, amends)
        except AttributeError:
            # Adapter doesn't expose amend_batch_orders. The bot is
            # configured with amend on a venue that doesn't support
            # it — fall back to cancel-then-place for every intent.
            logger.warning(
                "amend_batch_unsupported_adapter falling_back_to_cancel %d wos",
                len(valid_pairs),
            )
            with self._state._lock:
                self._state.amend_exchange_rejected_other_total += len(valid_pairs)
            for _intent, wo in valid_pairs:
                self._revert_amend_to_acked(
                    wo,
                    outcome="exchange_rejected",
                    detail="adapter_no_amend_support",
                )
            return
        except Exception:
            logger.exception(
                "amend_batch_http_failed reverting %d wos to ACKED",
                len(valid_pairs),
            )
            self._state.bump_execution_errors(
                "amend_batch_http_transport_exception"
            )
            with self._state._lock:
                self._state.amend_transport_rejected_total += len(valid_pairs)
            for _intent, wo in valid_pairs:
                self._revert_amend_to_acked(
                    wo,
                    outcome="transport_rejected",
                    detail="amend_batch_http_transport_exception",
                )
            return

        ack_t = time.perf_counter()
        rtt_ms = (ack_t - t_net) * 1000.0
        self._last_tick_order_submit_rtt_ms = max(
            float(self._last_tick_order_submit_rtt_ms), float(rtt_ms)
        )
        self._bump_transport_counters()

        from app.exchange.okx_responses import (
            interpret_okx_amend_batch_response,
        )
        top_kind, top_detail, rows = interpret_okx_amend_batch_response(resp)
        if top_kind == "transport":
            logger.warning(
                "amend_batch_transport_envelope detail=%s — reverting %d WOs",
                top_detail[:300],
                len(valid_pairs),
            )
            self._state.bump_execution_errors(
                "amend_batch_envelope_transport"
            )
            with self._state._lock:
                self._state.amend_transport_rejected_total += len(valid_pairs)
            for _intent, wo in valid_pairs:
                self._revert_amend_to_acked(
                    wo,
                    outcome="transport_rejected",
                    detail=top_detail[:400],
                )
            return

        # Correlate rows to WOs by reqId (deterministic ordId-seq).
        rows_by_corr: dict[
            str, tuple[Optional[int], str, str]
        ] = {}
        for r_oid, r_kind, r_detail, r_corr in rows:
            if r_corr:
                rows_by_corr[r_corr] = (r_oid, r_kind, r_detail)

        row_rate_limit_hits = 0
        for idx, (intent, wo) in enumerate(valid_pairs):
            # Correlation key must mirror the sent reqId format (see
            # ``row["req_id"] = ...`` above): ``a{ordId}s{seq}``.
            corr = (
                f"a{wo.order_id_exchange}s"
                f"{int(getattr(wo, 'amend_intent_seq', 0) or 0)}"
            )
            if corr in rows_by_corr:
                ex_oid, outcome, reason = rows_by_corr[corr]
            elif idx < len(rows):
                ex_oid, outcome, reason, _ = rows[idx]
            else:
                ex_oid, outcome, reason = None, "unconfirmed", "missing_row_for_amend"

            wo.ts_amend_response = self._clock.now_utc()
            wo.amend_response_outcome = outcome or None
            if reason:
                wo.amend_response_detail = reason[:400]

            if outcome == "accepted":
                # Commit target → price/size; preserve ordId; clear target_*.
                # BUG-040: an amend that CHANGES THE PRICE re-bases the
                # fill-age clock — see the note_ack re-stamp below. Capture
                # the px-change flag BEFORE we clear amend_target_px.
                _amend_px_changed = wo.amend_target_px is not None
                if wo.amend_target_px is not None:
                    wo.price = float(wo.amend_target_px)
                if wo.amend_target_sz is not None:
                    wo.size = float(wo.amend_target_sz)
                wo.amend_target_px = None
                wo.amend_target_sz = None
                transition(wo, OrderStatus.ACKED)
                # BUG-040: re-base the fill-age clock to THIS reprice.
                # OKX amend-in-place preserves order_id_exchange, so the
                # fill-bucket LRU (keyed by ex_oid) would otherwise keep
                # the ORIGINAL placement ts forever — making
                # ``quote_age_at_fill_ms`` measure age-since-first-placement
                # across every reprice (this session: ~12 amends per fill),
                # not age-at-current-price. ``transition`` above already
                # refreshed ``wo.ts_ack`` to now; mirror the place-path
                # note_ack so the fill side measures from the last price
                # change. Gate on px change only: a size-only amend leaves
                # the quote's price exposure clock running (correct toxicity
                # semantics). ``note_ack`` is idempotent-on-oid and
                # overwrites the cached ts (app/fill_bucket_metrics.py).
                if _amend_px_changed:
                    try:
                        if wo.ts_ack is not None and wo.order_id_exchange:
                            self._state.fill_buckets.note_ack(
                                order_id_exchange=wo.order_id_exchange,
                                ts_ack=wo.ts_ack,
                            )
                    except Exception:
                        logger.exception(
                            "fill_buckets_note_ack_failed_amend"
                        )
                try:
                    self._order_rtt_tracker.ingest(
                        rtt_ms=rtt_ms, op="amend", outcome="accepted",
                    )
                except Exception:
                    logger.exception("amend_rtt_ingest_failed")
                with self._state._lock:
                    self._state.amend_success_total += 1
                self.persist(wo)
                continue

            if outcome == "below_filled":
                # new_sz < already-filled — fall back to cancel-then-
                # place. Clear target_*, transition to CANCEL_PENDING,
                # emit a fresh cancel. The next reprice cycle's
                # _orchestrate will issue the place after the cancel
                # lands (existing legacy path).
                wo.amend_target_px = None
                wo.amend_target_sz = None
                with self._state._lock:
                    self._state.amend_below_filled_total += 1
                # Bump the cancel intent seq + persist BEFORE submit so
                # the executor sees an authoritative seq.
                self._enqueue_cancel_quote_path(
                    wo, trigger_reason="amend_below_filled"
                )
                continue

            if outcome == "order_gone":
                wo.amend_target_px = None
                wo.amend_target_sz = None
                transition(wo, OrderStatus.CANCELED, (reason or outcome)[:400])
                with self._state._lock:
                    self._state.amend_order_gone_total += 1
                self.persist(wo)
                self._release_working_slot_if_matches(wo)
                continue

            if outcome == "exchange_rejected":
                # Original order survives. Revert to ACKED, clear
                # target_*. If post_only_would_cross, arm the cooldown
                # for the side (same as place-side).
                #
                # v1.4.18: WARN-log every exchange_rejected row. The
                # v1.4.17 first-deploy missed a 100%-reject reqId-format
                # bug for nearly 100 seconds because the silent counter
                # bump was the only signal. From here forward every
                # rejected row produces a visible log line.
                if is_post_only_immediate_match_rejection(reason or ""):
                    # For AMEND rejections, the price the venue
                    # rejected is the amend target, not the live wo
                    # price. Capture it BEFORE ``_revert_amend_to_acked``
                    # zeroes out ``amend_target_px`` below.
                    _amend_rejected_px = getattr(
                        wo, "amend_target_px", None
                    )
                    if _amend_rejected_px is None:
                        _amend_rejected_px = wo.price
                    self._arm_post_only_cross_cooldown(
                        wo.side, rejected_price=_amend_rejected_px
                    )
                    try:
                        self._state.quote_quality.note_post_only_cross_rejection()
                    except Exception:
                        logger.exception(
                            "note_post_only_cross_rejection_failed_amend"
                        )
                    with self._state._lock:
                        self._state.amend_post_only_cross_total += 1
                else:
                    with self._state._lock:
                        self._state.amend_exchange_rejected_other_total += 1
                logger.warning(
                    "amend_row_exchange_rejected side=%s ord_id=%s "
                    "amend_intent_seq=%s reason=%s",
                    wo.side.value,
                    wo.order_id_exchange,
                    int(getattr(wo, "amend_intent_seq", 0) or 0),
                    (reason or "")[:300],
                )
                self._revert_amend_to_acked(
                    wo, outcome="exchange_rejected", detail=reason or outcome
                )
                continue

            if outcome == "transport_rejected":
                if reason and "rate_limit" in reason.lower():
                    row_rate_limit_hits += 1
                with self._state._lock:
                    self._state.amend_transport_rejected_total += 1
                logger.info(
                    "amend_row_transport_rejected side=%s ord_id=%s "
                    "amend_intent_seq=%s reason=%s",
                    wo.side.value,
                    wo.order_id_exchange,
                    int(getattr(wo, "amend_intent_seq", 0) or 0),
                    (reason or "")[:300],
                )
                self._revert_amend_to_acked(
                    wo, outcome="transport_rejected", detail=reason or outcome
                )
                continue

            # outcome == "unconfirmed" — strict-place-unconfirmed-kill
            # flow. The amend may have landed at the venue but we
            # don't know. Same gravity as an unconfirmed place: if the
            # operator opted into strict kill, kill the bot; else
            # leave AMEND_PENDING for the watchdog (Phase 4).
            #
            # v1.5.206 — tombstone-aware swallow. When the bot's order
            # store has a fresh tombstone for ``wo.order_id_exchange``
            # (i.e. the order was already terminated locally for some
            # OTHER reason — cancel, fill, hydration purge — within
            # the last ``_TOMBSTONE_MAX_AGE_SECONDS``), the missing
            # amend response is benign: the order is gone, the amend
            # is moot. Log + treat as ``order_gone`` instead of kill.
            # This closes the race that killed v1.5.204-260528-072247.
            try:
                tomb = None
                if (
                    reason == "missing_row_for_amend"
                    and wo.order_id_exchange
                    and hasattr(self._state, "order_store")
                ):
                    tomb = self._state.order_store.was_recently_removed(
                        oid=int(wo.order_id_exchange),
                        cloid=wo.client_order_id,
                    )
            except Exception:
                tomb = None
            if tomb is not None:
                logger.info(
                    "amend_response_unconfirmed_swallowed_tombstone "
                    "side=%s ord_id=%s reason=missing_row_for_amend "
                    "tombstone_reason=%s tombstone_age_s=%.2f — "
                    "order gone locally; amend is moot, NOT killing",
                    wo.side.value,
                    wo.order_id_exchange,
                    tomb.removed_reason,
                    (self._clock.monotonic() - tomb.removed_at_mono),
                )
                wo.amend_response_outcome = "order_gone"
                wo.amend_response_detail = (
                    f"amend_raced_with_removal:{tomb.removed_reason}"[:400]
                )
                wo.amend_target_px = None
                wo.amend_target_sz = None
                with self._state._lock:
                    # Mirror the order_gone branch's counter so the
                    # operator sees this in /status without a special
                    # field.
                    self._state.amend_order_gone_total += 1
                    # New v1.5.206 counter — distinct from clean
                    # order_gone so we can spot the race in postmortem.
                    self._state.amend_unconfirmed_swallowed_by_tombstone_total = (
                        int(
                            getattr(
                                self._state,
                                "amend_unconfirmed_swallowed_by_tombstone_total",
                                0,
                            )
                            or 0
                        )
                        + 1
                    )
                # Don't change status to CANCELED here — the tombstone
                # source did that already if applicable. Just persist
                # the amend-response observation fields.
                self.persist(wo)
                continue
            # v1.5.216 — UNCONDITIONAL swallow for missing_row_for_amend.
            #
            # The v1.5.206 tombstone-aware swallow above only catches the
            # narrow case where the bot LOCALLY removed the WO between
            # amend-send and amend-response. The 2026-05-28 v1.5.215
            # incident (kill_timestamp 13:50:18 on ord_id
            # 3606222898225680384, SELL) hit a DIFFERENT race: the WO was
            # alive in AMEND_PENDING state when the batch response came
            # back, but our response parser couldn't correlate any row to
            # this amend — either OKX dropped a row from the batch, or
            # the dispatcher's batch payload diverged from the response
            # row count. The reason ``missing_row_for_amend`` is
            # constructed CLIENT-side at the missing-correlation fallback
            # site above (search ``ex_oid, outcome, reason = None,
            # "unconfirmed", "missing_row_for_amend"``).
            #
            # CRITICAL semantic: ``missing_row_for_amend`` does NOT mean
            # the order is wedged. It means the bot can't tell from the
            # batch response whether the amend was applied. That is an
            # AMBIGUITY, not a wedge. Killing is the wrong response — it
            # discards a live working order, halts trading, and requires
            # operator restart. The right response is to force a
            # reconcile and let the venue's authoritative state resolve
            # the ambiguity.
            #
            # Per operator (verbatim, 2026-05-28 after second kill):
            # "the bug from earlier was not fixed!!". The v1.5.206 fix
            # was scoped too narrowly. v1.5.216 makes the swallow
            # unconditional for this reason code.
            if reason == "missing_row_for_amend":
                logger.warning(
                    "amend_response_missing_row_swallowed "
                    "side=%s ord_id=%s — batch response had no row for "
                    "this amend; treating as ambiguous (NOT killing); "
                    "forcing reconcile to resolve venue state",
                    wo.side.value,
                    wo.order_id_exchange,
                )
                # Don't claim accepted/rejected — caller can't know.
                # Leave the WO in AMEND_PENDING; the reconcile will
                # re-snap the local state to whatever the venue says
                # (alive at old px/sz / cancelled / filled).
                wo.amend_response_outcome = "ambiguous_missing_row"
                wo.amend_response_detail = (
                    "batch response missing correlated row; "
                    "reconciling against venue"
                )[:400]
                wo.amend_target_px = None
                wo.amend_target_sz = None
                with self._state._lock:
                    # New v1.5.216 counter — distinct from the v1.5.206
                    # tombstone-swallow counter so postmortem can tell
                    # the two races apart.
                    self._state.amend_missing_row_swallowed_total = (
                        int(
                            getattr(
                                self._state,
                                "amend_missing_row_swallowed_total",
                                0,
                            )
                            or 0
                        )
                        + 1
                    )
                    # Trigger a reconcile on the next tick — the venue
                    # knows the truth.
                    self._state.force_reconcile_requested = True
                self.persist(wo)
                continue
            if self._settings.strict_place_unconfirmed_kill:
                payload: dict[str, Any] = {
                    "side": wo.side.value,
                    "symbol": wo.symbol,
                    "order_id_exchange": wo.order_id_exchange,
                    "amend_target_px": wo.amend_target_px,
                    "amend_target_sz": wo.amend_target_sz,
                    "outcome": outcome,
                    "reason": (reason or "")[:1200],
                }
                logger.critical(
                    "amend_response_unconfirmed_critical side=%s symbol=%s "
                    "ord_id=%s reason=%s",
                    wo.side.value,
                    wo.symbol,
                    wo.order_id_exchange,
                    (reason or "")[:400],
                )
                self._state.place_unconfirmed_critical_total += 1
                try:
                    self._storage.insert_bot_event(
                        self._clock.now_utc().isoformat(),
                        EventSeverity.CRITICAL.value,
                        "amend_response_unconfirmed",
                        f"unconfirmed amend row — venue gave neither clean "
                        f"accept nor reject for side={wo.side.value} "
                        f"ord_id={wo.order_id_exchange}; order state may have "
                        f"changed on venue.",
                        payload,
                    )
                except Exception:
                    logger.exception(
                        "amend_unconfirmed_bot_event_failed"
                    )
                if self._request_kill_fn is not None:
                    try:
                        self._request_kill_fn(
                            "amend_response_unconfirmed", payload
                        )
                    except Exception:
                        logger.exception(
                            "request_kill_fn_raised_on_amend_unconfirmed"
                        )
                else:
                    with self._state._lock:
                        self._state.killed = True
                        self._state.kill_reason = (
                            "amend_response_unconfirmed"
                        )
            # Non-strict: leave AMEND_PENDING; the Phase 4 watchdog
            # reverts after a timeout.
            self.persist(wo)

        if row_rate_limit_hits > 0:
            try:
                self._client.bump_row_rate_limit_counter(row_rate_limit_hits)
            except AttributeError:
                pass

    def _revert_amend_to_acked(
        self,
        wo: WorkingOrder,
        *,
        outcome: str,
        detail: Optional[str] = None,
    ) -> None:
        """Helper: roll the WO back from AMEND_PENDING to ACKED with
        the original (pre-amend) px/sz intact. Stamps the response
        fields so diagnostics can see WHY the amend didn't take.

        Used by every non-accept, non-gone, non-below_filled outcome
        (exchange_rejected, transport_rejected, watchdog timeout in
        Phase 4). The amend_target_* fields are cleared because the
        amend is finalised (succeeded would've committed them; failed
        means they're no longer relevant).
        """
        wo.amend_target_px = None
        wo.amend_target_sz = None
        wo.amend_response_outcome = outcome
        if detail:
            wo.amend_response_detail = detail[:400]
        wo.ts_amend_response = self._clock.now_utc()
        transition(wo, OrderStatus.ACKED)
        self.persist(wo)

    def _amend_viable(
        self,
        cur: WorkingOrder,
        desired: FinalQuoteOrder,
    ) -> bool:
        """Predicate: is amend-on-reprice viable for this transition?

        amend-prio Phase 3 (v1.4.16). See ``plans/amend-prio.md``
        §"Phase 3 — Decision logic". Returns True only when:

          * The operator-flipped knob ``OKX_AMEND_ON_REPRICE_ENABLED``
            is True
          * The WO has a venue ordId bound (no amend without it)
          * The WO is ACKED (not PARTIAL — Phase 3 stays conservative;
            PARTIAL → amend extension is Phase 6 work)
          * The post-only-cross cooldown is NOT armed for the side
            (an amend that would cross would be rejected too —
            preempt the round-trip)
          * The adapter exposes amend_batch_orders (defensive — bot
            on a venue without amend support falls through to
            cancel-then-place)

        Side / size / price changes are always structurally valid
        amend payloads on OKX (newPx + newSz); no further filter
        needed at this layer.
        """
        if not bool(
            getattr(self._settings, "okx_amend_on_reprice_enabled", False)
        ):
            return False
        if not cur.order_id_exchange:
            return False
        if cur.status != OrderStatus.ACKED:
            # PARTIAL excluded: bot doesn't track filled qty on the
            # WO, so we can't pre-check the OKX below_filled (51016)
            # constraint here. Cancel-then-place is the safer path.
            return False
        if self._post_only_cross_cooldown_active(cur.side):
            return False
        # Adapter capability check. ``getattr(..., None)`` avoids
        # raising on adapters that lack the method.
        if not callable(
            getattr(self._client, "amend_batch_orders", None)
        ):
            return False
        return True

    def _enqueue_amend_quote_path(
        self,
        cur: WorkingOrder,
        desired: FinalQuoteOrder,
        *,
        trigger_reason: Optional[str] = None,
    ) -> bool:
        """Stamp the WO with amend target px/sz, transition to
        AMEND_PENDING, bump intent_seq, and submit a kind="amend"
        PlaceTransportIntent.

        Returns True when the intent was successfully submitted.
        Returns False on guards (no write access, missing ordId — but
        ``_amend_viable`` should have caught these upstream).

        The persisted WO row is written AFTER the dispatcher enqueue
        (Phase 0b "persist after wire" convention shared with the
        place/cancel paths).
        """
        if not self._client.has_write_access():
            return False
        if not cur.order_id_exchange:
            logger.error(
                "amend_enqueue_missing_ord_id wo=%s", cur.order_id_local
            )
            return False
        # v1.4.169 Phase 2I — per-order amend rate-defence guard.
        # STRUCTURAL guard against the snapshot v1.4.102-260520-120555
        # pattern (244 amends/sec on a single order ping-ponging
        # between adjacent ticks). Two complementary checks:
        #   1. Tick-flicker: reject when the previous dispatch on
        #      this same order was less than AMEND_TICK_FLICKER_MIN_MS
        #      ago. Stops the bot from generating an amend faster
        #      than the venue can ack the previous one.
        #   2. Per-order 1-s cap: reject when the live count of
        #      dispatches inside the last 1 s already meets/exceeds
        #      AMEND_PER_ORDER_MAX_PER_SEC.
        # Both fire BEFORE any state mutation so a suppression leaves
        # the WO untouched. The dispatcher receives nothing; the
        # caller sees False and moves on.
        now_mono = self._clock.monotonic()
        recent: list[float] = getattr(
            cur, "amend_recent_dispatch_mono", []
        )
        # Prune entries older than 1 s in-place. The ring is small
        # (a runaway loop would push 200+ in a second but the steady
        # state is a handful), so a linear scan is fine.
        recent[:] = [t for t in recent if now_mono - t < 1.0]
        flicker_min_s = float(
            getattr(
                self._settings,
                "amend_tick_flicker_min_ms",
                250.0,
            )
        ) / 1000.0
        rate_cap = int(
            getattr(
                self._settings,
                "amend_per_order_max_per_sec",
                8,
            )
        )
        # v1.4.180: reset the suppress-reason flag at top of every
        # invocation so a previous tick's value can't leak through.
        # The AmendAction dispatcher reads this attribute IMMEDIATELY
        # after the call; the value is meaningful only in that window.
        self._last_amend_enqueue_suppress_reason = None
        suppress_reason: Optional[str] = None
        if recent and (now_mono - recent[-1]) < flicker_min_s - 1e-12:
            suppress_reason = "tick_flicker"
        elif len(recent) >= rate_cap:
            suppress_reason = "rate_throttle"
        if suppress_reason is not None:
            # Bump the matching counter + log once per side per
            # session (subsequent suppressions are silent; the counter
            # tells the operator the rate).
            try:
                if suppress_reason == "tick_flicker":
                    self._state.amend_tick_flicker_suppressed_total += 1
                else:
                    self._state.amend_rate_throttle_suppressed_total += 1
                first_logged = getattr(
                    self._state,
                    "amend_rate_defence_first_arm_logged",
                    {},
                )
                if not first_logged.get(cur.side, False):
                    logger.warning(
                        "amend_rate_defence_armed reason=%s side=%s "
                        "wo=%s recent_in_1s=%d flicker_min_ms=%.0f "
                        "rate_cap=%d — first arm this session for "
                        "this side; subsequent suppressions silent; "
                        "watch the per-side cumulative counter",
                        suppress_reason,
                        cur.side.value,
                        cur.order_id_local,
                        len(recent),
                        flicker_min_s * 1000.0,
                        rate_cap,
                    )
                    first_logged[cur.side] = True
                    self._state.amend_rate_defence_first_arm_logged = (
                        first_logged
                    )
            except Exception:
                logger.exception(
                    "amend_rate_defence_counter_bump_failed"
                )
            # Postmortem breadcrumb — same channel the other
            # gate suppressions use.
            try:
                self._record_orchestrate_decision(
                    cur.side,
                    f"amend_suppress:{suppress_reason}",
                    cycle_id=str(cur.quote_cycle_id or ""),
                    extra={
                        "wo_order_id_local": cur.order_id_local,
                        "wo_order_id_exchange": cur.order_id_exchange,
                        "recent_dispatches_in_1s": len(recent),
                        "flicker_min_ms": flicker_min_s * 1000.0,
                        "rate_cap_per_sec": rate_cap,
                        "trigger_reason": trigger_reason or "",
                    },
                )
            except Exception:
                # Breadcrumb failure must not affect the guard's
                # decision — it's already made above.
                pass
            # v1.4.180: stash the suppress reason on the instance so
            # the AmendAction dispatcher can distinguish Phase 2I
            # suppression (NOT to fall through to cancel-replace,
            # because that defeats the rate-defence intent) from
            # genuine enqueue failures (fall through to cancel-
            # replace is correct).
            self._last_amend_enqueue_suppress_reason = suppress_reason
            return False
        # Guard passed; record this dispatch in the ring for the
        # next call. Persist the updated list back onto the WO via
        # ``amend_recent_dispatch_mono`` (the field is a list, mutated
        # in place above by the prune, then appended here).
        recent.append(now_mono)
        cur.amend_recent_dispatch_mono = recent
        if self._first_enqueue_perf is None:
            self._first_enqueue_perf = time.perf_counter()
        cur.amend_target_px = float(desired.price)
        cur.amend_target_sz = float(desired.size)
        cur.ts_amend_sent = self._clock.now_utc()
        cur.amend_intent_seq = int(getattr(cur, "amend_intent_seq", 0) or 0) + 1
        if trigger_reason and not cur.amend_response_detail:
            # Capture the trigger as a breadcrumb until the response
            # lands (response_detail will overwrite on completion).
            cur.amend_response_detail = f"trigger:{trigger_reason}"
        transition(cur, OrderStatus.AMEND_PENDING)
        # amend-prio Phase 4 (v1.4.17): bump the emitted-intent counter
        # + recompute the high-watermark of concurrent AMEND_PENDING.
        # Both feed the operator-facing Telegram /status line.
        with self._state._lock:
            self._state.amend_intents_emitted_total += 1
            self._state.amend_pending_high_watermark = max(
                self._state.amend_pending_high_watermark,
                self._count_amend_pending_unlocked(),
            )
        side = cur.side
        tr = self._outbound_traces.get(cur.order_id_local)
        if tr is None:
            tr = OutboundActionTrace(
                order_id_local=cur.order_id_local,
                side_value=cur.side.value,
                intent_created_perf=time.perf_counter(),
                dispatcher_enqueue_perf=time.perf_counter(),
            )
            self._outbound_traces[cur.order_id_local] = tr
        ic = (
            float(self._maybe_refresh_t0_perf)
            if self._maybe_refresh_t0_perf
            else time.perf_counter()
        )
        self._outbound.submit_place(
            PlaceTransportIntent(
                wo_order_id_local=cur.order_id_local,
                side=side,
                intent_seq=int(cur.amend_intent_seq),
                quote_cycle_id="amend",
                enqueued_mono=self._clock.monotonic(),
                intent_created_perf=ic,
                level_idx=int(getattr(cur, "level_idx", 0) or 0),
                kind="amend",
            )
        )
        self.persist(cur)
        self._sync_outbound_state_flags()
        return True

    def _post_only_cross_cooldown_active(self, side: Side) -> bool:
        """True iff the post-only-cross cooldown for ``side`` is still
        active. Phase 2K.11 (v1.4.160): also evaluates the
        favorable-exit predicate ("touch moved ≥ 1 tick away from the
        rejected price") and clears the cooldown early when it holds.
        Tracks the active→cleared edge for ceiling attribution.

        Defensive init: some test files use shell-mock ``OrderManager``
        instances that bypass ``__init__``; the Phase 2K.11 attrs are
        created on first access via ``hasattr`` checks below. Cheap;
        no-op in production once the attrs exist.
        """
        if float(self._settings.post_only_cross_cooldown_seconds) <= 0:
            return False
        if not hasattr(self, "_post_only_cross_cooldown_rejected_price"):
            self._post_only_cross_cooldown_rejected_price = {
                Side.BUY: None,
                Side.SELL: None,
            }
        if not hasattr(self, "_post_only_cross_cooldown_was_active_last"):
            self._post_only_cross_cooldown_was_active_last = {
                Side.BUY: False,
                Side.SELL: False,
            }
        if not hasattr(
            self, "_post_only_cross_cooldown_cleared_via_favorable_total"
        ):
            self._post_only_cross_cooldown_cleared_via_favorable_total = 0
        if not hasattr(
            self, "_post_only_cross_cooldown_cleared_via_ceiling_total"
        ):
            self._post_only_cross_cooldown_cleared_via_ceiling_total = 0
        until = self._post_only_cross_cooldown_until.get(side, 0.0)
        now = self._clock.monotonic()
        active_by_timer = now < until

        # Phase 2K.11 favorable-exit predicate. Only evaluate while
        # the timer is still ticking — there's nothing to short-
        # circuit otherwise.
        if active_by_timer and bool(
            getattr(
                self._settings,
                "post_only_cross_cooldown_favorable_exit_enabled",
                True,
            )
        ):
            try:
                from app.post_only_cross_cooldown import (
                    touch_moved_away_from_rejected_price,
                )

                rejected_price = (
                    self._post_only_cross_cooldown_rejected_price.get(
                        side
                    )
                )
                bb, ba = self._current_best_bid_ask()
                tick = self._current_tick_size()
                tick_mult = float(
                    getattr(
                        self._settings,
                        "post_only_cross_cooldown_clear_tick_multiplier",
                        1.0,
                    )
                )
                cleared = touch_moved_away_from_rejected_price(
                    side=side,
                    rejected_price=rejected_price,
                    best_bid=bb,
                    best_ask=ba,
                    tick_size=tick,
                    tick_multiplier=tick_mult,
                )
                if cleared:
                    self._post_only_cross_cooldown_until[side] = 0.0
                    self._post_only_cross_cooldown_rejected_price[
                        side
                    ] = None
                    self._post_only_cross_cooldown_cleared_via_favorable_total += 1
                    self._post_only_cross_cooldown_was_active_last[
                        side
                    ] = False
                    return False
            except Exception:
                # Favorable-exit failure must NEVER let the cooldown
                # exit early or stuck — fall through to the timer.
                logger.exception(
                    "post_only_cross_cooldown_favorable_exit_failed "
                    "side=%s",
                    getattr(side, "value", side),
                )

        # Re-read the timer state after possible favorable clearing.
        active = now < self._post_only_cross_cooldown_until.get(
            side, 0.0
        )

        # Ceiling attribution: active→cleared edge with the timer
        # still in the past (favorable-exit zeroes it out, so a
        # non-zero ``until`` here means the timer actually expired).
        was = self._post_only_cross_cooldown_was_active_last.get(
            side, False
        )
        if was and not active:
            if (
                self._post_only_cross_cooldown_until.get(side, 0.0)
                > 0.0
            ):
                self._post_only_cross_cooldown_cleared_via_ceiling_total += 1
        self._post_only_cross_cooldown_was_active_last[side] = active
        return active

    def _current_best_bid_ask(self) -> tuple[
        Optional[float], Optional[float]
    ]:
        """Return ``(best_bid, best_ask)`` from the market store.
        Helper used by the Phase 2K.11 favorable-exit predicate;
        isolated so the favorable-exit path can be tested + audited
        independently of the market_store internals.

        ``MarketStore.best_bid`` / ``.best_ask`` are methods (not
        properties); call them and unwrap to plain floats."""
        try:
            ms = getattr(self._state, "market_store", None)
            if ms is None:
                return (None, None)
            bb = ms.best_bid() if callable(
                getattr(ms, "best_bid", None)
            ) else None
            ba = ms.best_ask() if callable(
                getattr(ms, "best_ask", None)
            ) else None
            try:
                bb = float(bb) if bb is not None else None
            except (TypeError, ValueError):
                bb = None
            try:
                ba = float(ba) if ba is not None else None
            except (TypeError, ValueError):
                ba = None
            return (bb, ba)
        except Exception:
            return (None, None)

    def _current_tick_size(self) -> Optional[float]:
        """Return the symbol's price tick. Reads from the venue
        adapter's ``symbol_spec.price_tick`` (the source-of-truth in
        the bot). Returns ``None`` if the spec isn't available or
        the tick is non-positive — the favorable-exit predicate will
        decline to fire in that case, deferring to the timer."""
        try:
            spec = getattr(self._client, "symbol_spec", None)
            if spec is None:
                return None
            t = getattr(spec, "price_tick", None)
            if t is None:
                return None
            t = float(t)
            return t if t > 0 else None
        except Exception:
            return None

    def _arm_post_only_cross_cooldown(
        self, side: Side, rejected_price: Optional[float] = None
    ) -> None:
        """Arm the post-only-cross cooldown for ``side`` for
        ``POST_ONLY_CROSS_COOLDOWN_SECONDS``. ``rejected_price`` is
        captured for the Phase 2K.11 favorable-exit predicate; pass
        ``None`` for legacy call paths that don't have the price
        handy — the predicate will skip and the timer continues to
        govern the worst-case."""
        d = float(self._settings.post_only_cross_cooldown_seconds)
        if d <= 0:
            return
        self._post_only_cross_cooldown_until[side] = self._clock.monotonic() + d
        # Defensive init (same reason as ``_post_only_cross_cooldown_active``).
        if not hasattr(self, "_post_only_cross_cooldown_rejected_price"):
            self._post_only_cross_cooldown_rejected_price = {
                Side.BUY: None,
                Side.SELL: None,
            }
        if rejected_price is not None:
            try:
                self._post_only_cross_cooldown_rejected_price[side] = (
                    float(rejected_price)
                )
            except (TypeError, ValueError):
                self._post_only_cross_cooldown_rejected_price[side] = None

    def _adverse_side_pause_active(self, side: Side) -> bool:
        """True when placements on ``side`` are suppressed due to adverse markouts."""
        if float(self._settings.adverse_side_pause_seconds) <= 0:
            return False
        return self._clock.monotonic() < self._adverse_side_pause_until.get(side, 0.0)

    def _is_reducing_side(self, side: Side) -> bool:
        """True when placing on ``side`` would reduce current inventory.

        BUG-010: the adverse-side pause must NOT block the inventory-reducing
        side, otherwise a mild localised toxicity signal can strand the bot
        with directional inventory by suppressing the very side needed to
        close it. Long → SELL reduces; short → BUY reduces; flat → no
        reducing side.
        """
        try:
            qty = float(self._state.position.position_qty)
        except (AttributeError, TypeError, ValueError):
            return False
        if abs(qty) <= _POSITION_EPS:
            return False
        return (qty > 0.0 and side == Side.SELL) or (qty < 0.0 and side == Side.BUY)

    def _maybe_arm_adverse_side_pause(self, toxicity: ToxicitySnapshot) -> None:
        """Arm per-side suppression when recent markouts on that side are adverse.

        Called once per tick from the bot loop. Uses the per-side average markout
        already computed by :class:`ToxicityEngine.snapshot`. Invariants:

        1. **No re-extend while active.** If the pause is already running, do
           nothing — let it expire naturally. Extending the deadline every tick
           creates a self-perpetuating lockout: the pause prevents new fills on
           that side, no new data refreshes the markout avg, the same old
           adverse samples keep re-triggering the arm, forever. See
           ``tmp/snap_20260418_091244``: a single -11.77 bps BUY fill armed the
           pause and the same fill kept re-arming it for 3+ minutes until the
           operator stopped the bot.
        2. **No stale re-arm.** After the pause expires, we only re-arm if
           there's been at least one NEW fill on that side since the previous
           arming. Without new data there is no new adverse signal — just the
           same old fill still in the window.

        When both invariants pass: fresh arming. Log once per arming event.

        Thresholds:
        - ``adverse_side_pause_soft_bps`` (default 2.0): if ``avg_markout <=
          -threshold``, consider arming.
        - ``adverse_side_pause_min_fills`` (default 2): ignore the signal until
          this many resolved markouts for that side are available.
        - ``adverse_side_pause_seconds`` (default 15.0): cooldown duration.
        """
        pause_s = float(self._settings.adverse_side_pause_seconds)
        if pause_s <= 0:
            return
        threshold = -float(self._settings.adverse_side_pause_soft_bps)
        min_fills = int(self._settings.adverse_side_pause_min_fills)
        # v1.4.152 Phase 2K.2 — favorable-exit clear band. Pause
        # clears early when avg_markout recovers past
        # ``-soft_bps × (1 - mult)`` (e.g. soft=2.0, mult=0.5 →
        # clear when avg > -1.0). mult=0 disables early-exit.
        clear_mult = float(
            getattr(
                self._settings,
                "adverse_side_pause_clear_threshold_mult",
                0.5,
            )
            or 0.0
        )
        clear_threshold = threshold * (1.0 - clear_mult)
        now_m = self._clock.monotonic()
        # Defensive init for shells/tests that bypass __init__. The
        # real Executor sets these in __init__; tests that subclass
        # with a lighter constructor may not. Same back-compat
        # pattern other Phase 2K refactors use.
        if not hasattr(self, "_adverse_side_pause_arm_threshold"):
            self._adverse_side_pause_arm_threshold = {
                Side.BUY: float("nan"),
                Side.SELL: float("nan"),
            }
        if not hasattr(self, "adverse_side_pause_cleared_via_favorable_total"):
            self.adverse_side_pause_cleared_via_favorable_total = 0
        if not hasattr(self, "adverse_side_pause_cleared_via_ceiling_total"):
            self.adverse_side_pause_cleared_via_ceiling_total = 0
        # Use ``session_fill_count_by_side`` (monotonic session counter) as the
        # "new data" gate. ``toxicity.buy_side_fill_count`` / ``sell_side_fill_count``
        # are window-bounded (by the toxicity engine's 20-fill slice and the
        # underlying ``recent_fills`` deque at maxlen=200), so they'd eventually
        # stop growing and break this invariant on long sessions.
        session_n_by_side = dict(self._state.session_fill_count_by_side)
        for side, (avg, window_n) in (
            (Side.BUY, (toxicity.buy_side_avg_markout_bps, toxicity.buy_side_fill_count)),
            (Side.SELL, (toxicity.sell_side_avg_markout_bps, toxicity.sell_side_fill_count)),
        ):
            prev_until = self._adverse_side_pause_until.get(side, 0.0)
            currently_active = prev_until > now_m + 1e-9

            # v1.4.152 Phase 2K.2 — favorable-exit check. Runs on
            # every tick for active pauses. If markout has recovered
            # past the clear band, drop the deadline NOW (attribute
            # to "favorable"). The pause's two original invariants
            # (no re-extend, no stale re-arm) are unaffected — they
            # still gate ARMING, while this gates CLEARING.
            if currently_active:
                if (
                    clear_mult > 0.0
                    and avg is not None
                    and window_n >= min_fills
                    and avg > clear_threshold + 1e-12
                ):
                    # Favorable exit — markout recovered halfway-ish.
                    self._adverse_side_pause_until[side] = 0.0
                    self._adverse_side_pause_arm_threshold[side] = float("nan")
                    self.adverse_side_pause_cleared_via_favorable_total += 1
                    log_extra(
                        logger,
                        logging.INFO,
                        "adverse_side_pause_cleared_favorable",
                        {
                            "side": side.value,
                            "avg_markout_bps": round(avg, 4),
                            "clear_threshold_bps": round(clear_threshold, 4),
                            "ceiling_remaining_s": round(
                                max(0.0, prev_until - now_m), 3
                            ),
                        },
                    )
                # If still active and not cleared, skip the rest of
                # this iteration (no re-extend per Invariant 1).
                continue

            # Was active last tick OR earlier, deadline has now passed.
            # If we had an arm-threshold recorded, that means the
            # ceiling fired — attribute the clearing.
            prev_arm_threshold = self._adverse_side_pause_arm_threshold.get(
                side, float("nan")
            )
            if prev_until > 0.0 and not math.isnan(prev_arm_threshold):
                # Ceiling fired (deadline lapsed, never cleared early).
                self.adverse_side_pause_cleared_via_ceiling_total += 1
                self._adverse_side_pause_arm_threshold[side] = float("nan")
                # Drop the stale ``_adverse_side_pause_until`` to 0.0
                # so we don't re-attribute on subsequent ticks.
                self._adverse_side_pause_until[side] = 0.0

            # Arming side (unchanged from pre-2K.2):
            # ``window_n`` still gates the MIN-SAMPLE rule — need enough recent
            # fills on that side to trust the markout average. This is window-
            # bounded on purpose (we don't want to treat 3-hour-old data as
            # "recent markout").
            if avg is None or window_n < min_fills:
                continue
            if avg > threshold + 1e-12:
                continue
            # Invariant 2: require a new session-fill on that side since the
            # previous arming. Uses the monotonic counter so the gate doesn't
            # break when the deque hits maxlen. First arming: ``prev_arm_n=-1``,
            # so any non-negative session count passes.
            session_n = int(session_n_by_side.get(side, 0))
            prev_arm_n = int(self._adverse_side_pause_arm_n_fills.get(side, -1))
            if session_n <= prev_arm_n:
                continue
            self._adverse_side_pause_until[side] = now_m + pause_s
            self._adverse_side_pause_arm_n_fills[side] = session_n
            self._adverse_side_pause_arm_threshold[side] = threshold
            log_extra(
                logger,
                logging.INFO,
                "adverse_side_pause_armed",
                {
                    "side": side.value,
                    "avg_markout_bps": round(avg, 4),
                    "threshold_bps": threshold,
                    "clear_threshold_bps": round(clear_threshold, 4),
                    "pause_seconds": pause_s,
                    "window_n_fills": window_n,
                    "session_n_fills": session_n,
                    "prev_arm_session_n": prev_arm_n,
                },
            )

    def _clear_sent_ambiguous_polls(self, wo: WorkingOrder) -> None:
        self._sent_ambiguous_polls.pop(wo.order_id_local, None)

    def _note_sent_order_cloid_ambiguous(self, wo: WorkingOrder, reason: str, detail: str) -> None:
        if wo.status != OrderStatus.SENT:
            return
        k = wo.order_id_local
        n = self._sent_ambiguous_polls.get(k, 0) + 1
        self._sent_ambiguous_polls[k] = n
        logger.warning(
            "sent_order_cloid_ambiguous order_local=%s poll_n=%s reason=%s detail=%s",
            k,
            n,
            reason,
            (detail or "")[:200],
        )
        self._maybe_finalize_stuck_sent_order(wo)

    def _maybe_finalize_stuck_sent_order(self, wo: WorkingOrder) -> None:
        """Bounded recovery: SENT cannot block a side forever on bad/ambiguous orderStatus."""
        if wo.status != OrderStatus.SENT:
            return
        n = self._sent_ambiguous_polls.get(wo.order_id_local, 0)
        age_s = 0.0
        if wo.ts_sent:
            age_s = max(0.0, (self._clock.now_utc() - wo.ts_sent).total_seconds())
        timeout_s = float(self._settings.sent_order_unresolved_timeout_seconds)
        max_polls = int(self._settings.sent_order_unresolved_max_ambiguous_polls)
        over_time = age_s >= timeout_s
        over_polls = max_polls > 0 and n >= max_polls
        if not over_time and not over_polls:
            return
        self._clear_sent_ambiguous_polls(wo)
        reason = f"sent_unresolved_age_s={age_s:.0f}_polls={n}"
        transition(wo, OrderStatus.REJECTED, reason[:2000])
        self.persist(wo)
        self._release_working_slot_if_matches(wo)
        self._storage.insert_bot_event(
            self._clock.now_utc().isoformat(),
            EventSeverity.WARNING.value,
            "sent_order_unresolved_timeout",
            f"SENT order unresolved after bounded wait; releasing slot ({reason})",
            {
                "symbol": wo.symbol,
                "side": wo.side.value,
                "order_id_local": wo.order_id_local,
                "client_order_id": wo.client_order_id,
                "age_seconds": age_s,
                "ambiguous_polls": n,
                "timeout_seconds": timeout_s,
                "max_ambiguous_polls": max_polls,
            },
        )
        logger.warning(
            "sent_order_unresolved_released slot side=%s local=%s %s",
            wo.side.value,
            wo.order_id_local,
            reason,
        )

    def _rejection_quarantine_base_key(
        self, symbol: str, side: Side, price: float, size: float
    ) -> str:
        return f"{symbol}|{side.value}|{price!r}|{size!r}"

    def _rejection_quarantine_strike_key(self, base_key: str, reason_norm: str) -> str:
        return f"{base_key}|{reason_norm}"

    def _prune_expired_validation_quarantines(self) -> None:
        now = self._clock.monotonic()
        for k, u in list(self._vk_until.items()):
            if u <= now:
                del self._vk_until[k]
                self._vk_strikes.pop(k, None)

    def _rejection_quarantine_active(
        self, symbol: str, side: Side, price: float, size: float
    ) -> tuple[bool, Optional[str], Optional[str], Optional[str], float]:
        """
        If the last deterministic reject for this intent is under quarantine, block resubmit.

        Returns (active, strike_key, reason_norm, validation_class, until_monotonic).
        """
        now = self._clock.monotonic()
        base = self._rejection_quarantine_base_key(symbol, side, price, size)
        last = self._vk_last_reason.get(base)
        if not last:
            return False, None, None, None, 0.0
        sk = self._rejection_quarantine_strike_key(base, last)
        until = self._vk_until.get(sk, 0.0)
        if until <= now:
            return False, None, None, None, 0.0
        vclass = exchange_validation_failure_class(last)
        return True, sk, last, vclass, until

    def _record_order_rejection_quarantine_strike(
        self,
        symbol: str,
        side: Side,
        price: float,
        size: float,
        vclass: str,
        raw_reason: str,
        quote_cycle_id: str,
    ) -> None:
        reason_norm = _normalize_rejection_reason_key(raw_reason)
        if not reason_norm:
            return
        base = self._rejection_quarantine_base_key(symbol, side, price, size)
        self._vk_last_reason[base] = reason_norm
        sk = self._rejection_quarantine_strike_key(base, reason_norm)
        n = self._vk_strikes.get(sk, 0) + 1
        self._vk_strikes[sk] = n
        need = self._settings.exchange_validation_quarantine_min_repeats
        if n < need:
            return
        dur = self._settings.exchange_validation_quarantine_seconds
        until = self._clock.monotonic() + dur
        self._vk_until[sk] = until
        if n == need:
            logger.warning(
                "order_rejection_quarantine_started symbol=%s side=%s class=%s "
                "strikes=%s quarantine_s=%s reason=%s quote_cycle_id=%s key=%s",
                symbol,
                side.value,
                vclass,
                n,
                dur,
                reason_norm[:500],
                quote_cycle_id,
                sk,
            )
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.WARNING.value,
                "order_rejection_quarantine_started",
                (
                    f"order rejection quarantine started symbol={symbol} side={side.value} "
                    f"class={vclass} strikes={n} quarantine_s={dur} quote_cycle_id={quote_cycle_id}"
                ),
                {
                    "symbol": symbol,
                    "side": side.value,
                    "quote_cycle_id": quote_cycle_id,
                    "validation_class": vclass,
                    "strikes": n,
                    "quarantine_seconds": dur,
                    "reason": (raw_reason or "")[:800],
                    "key": sk,
                    "normalized_price": price,
                    "normalized_size": size,
                },
            )

    def _clear_validation_quarantine_for_side(self, symbol: str, side: Side) -> None:
        pfx = f"{symbol}|{side.value}|"
        for k in list(set(self._vk_strikes) | set(self._vk_until)):
            if k.startswith(pfx):
                self._vk_strikes.pop(k, None)
                self._vk_until.pop(k, None)
        for b in list(self._vk_last_reason):
            if b.startswith(pfx):
                del self._vk_last_reason[b]

    def on_bot_tick_start(self) -> None:
        self._bot_tick_counter += 1
        self._state.reset_inbound_tick_counters()

    def should_ingest_fills_via_rest(self) -> bool:
        if self._settings.fast_start_skip_historical_fill_replay:
            with self._state._lock:
                if self._state.bot_status == BotStatus.STARTING:
                    self._state.startup_historical_fill_replay_skipped = True
                    self._state.startup_rest_fill_replay_skipped_count += 1
                    return False
        if not self._settings.private_ws_enabled or self._private_q is None:
            return True
        with self._state._lock:
            if self._state.private_ws_recovery_pending:
                return True
        if not self._private_ws_healthy:
            return True
        if self._bot_tick_counter <= self._rest_fill_catchup_until_tick:
            return True
        n = self._settings.rest_fill_reconcile_interval_ticks
        if n > 0 and self._bot_tick_counter % n == 0:
            return True
        return False

    def _apply_private_ws_overflow_recovery_if_needed(self) -> None:
        with self._state._lock:
            if not self._state.private_ws_recovery_pending:
                return
            self._state.private_ws_recovery_pending = False
            drops = self._state.private_ws_queue_drops
        self._force_open_orders_reconcile = True
        catchup = self._settings.private_ws_recovery_fill_catchup_ticks
        self._rest_fill_catchup_until_tick = max(
            self._rest_fill_catchup_until_tick,
            self._bot_tick_counter + catchup,
        )
        logger.warning(
            "private_ws_overflow_recovery scheduled drops_total=%s open_orders_reconcile_forced=true "
            "rest_fill_catchup_until_tick=%s",
            drops,
            self._rest_fill_catchup_until_tick,
        )
        self._storage.insert_bot_event(
            self._clock.now_utc().isoformat(),
            EventSeverity.WARNING.value,
            "private_ws_queue_overflow_recovery",
            (
                f"private WS queue dropped events (drops_total={drops}); "
                f"scheduled REST open-order reconcile + fill catchup until tick "
                f"{self._rest_fill_catchup_until_tick}"
            ),
            {
                "drops_total": drops,
                "catchup_until_tick": self._rest_fill_catchup_until_tick,
                "catchup_ticks": catchup,
            },
        )

    def drain_private_events(self, pnl: PnlTracker | None) -> None:
        # Serialise the two callers (quote loop + background drain
        # thread) so in-memory state mutations inside
        # ``_dispatch_private_event`` can't race. The DB itself is
        # already thread-safe (WAL + Storage._lock), but wo.status /
        # pnl counters live in Python memory and need this guard.
        with self._drain_mutex:
            self._drain_private_events_locked(pnl)

    def _drain_private_events_locked(self, pnl: PnlTracker | None) -> None:
        if not self._private_q:
            self._maybe_process_deferred_snapshot_fills(pnl)
            return
        self._apply_private_ws_overflow_recovery_if_needed()
        n = 0
        t_drain0 = time.perf_counter()
        try:
            peak_depth = self._private_q.qsize()
        except Exception:
            peak_depth = 0
        while n < _MAX_PRIVATE_EVENTS_PER_TICK:
            try:
                ev = self._private_q.get_nowait()
            except queue.Empty:
                break
            n += 1
            deq_mono = time.perf_counter()
            ev = self._enrich_private_event_at_dequeue(ev, deq_mono)
            self._dispatch_private_event(ev, pnl)
            try:
                peak_depth = max(peak_depth, self._private_q.qsize())
            except Exception:
                pass
        drain_ms = (time.perf_counter() - t_drain0) * 1000.0
        try:
            depth_after = self._private_q.qsize()
        except Exception:
            depth_after = 0
        with self._state._lock:
            self._state.private_ws_drain_time_ms_last_tick = float(drain_ms)
            self._state.private_ws_queue_high_watermark = max(
                int(self._state.private_ws_queue_high_watermark), int(peak_depth)
            )
            self._state.private_ws_max_events_drained_per_tick = max(
                int(self._state.private_ws_max_events_drained_per_tick), int(n)
            )
            self._state.private_ws_events_drained_last_tick = int(n)
            self._state.private_ws_queue_depth_after_drain = int(depth_after)
        if n >= _MAX_PRIVATE_EVENTS_PER_TICK:
            try:
                backlog = not self._private_q.empty()
            except Exception:
                backlog = False
            if backlog:
                with self._state._lock:
                    self._state.private_ws_recovery_pending = True
                    self._state.private_ws_queue_backlog_after_drain_count += 1
                self._apply_private_ws_overflow_recovery_if_needed()
                logger.warning(
                    "private_ws_event_backlog_after_tick_cap processed=%s scheduling_recovery",
                    n,
                )
        self._maybe_process_deferred_snapshot_fills(pnl)

    def _maybe_process_deferred_snapshot_fills(self, pnl: PnlTracker | None) -> None:
        """Process deferred private WS snapshot fills after startup readiness.

        These are for analytics/persistence only and are limited to truly historical fills
        (fill ts < session_started_at_utc) to avoid polluting session-scoped PnL/toxicity.
        """
        if (
            not self._settings.fast_start_skip_historical_fill_replay
            or not self._deferred_private_snapshot_fills
        ):
            return
        if self._state.bot_status == BotStatus.STARTING:
            return

        with self._state._lock:
            session_start_ms = int(self._state.session_started_at_utc.timestamp() * 1000.0)

        processed = 0
        while (
            self._deferred_private_snapshot_fills
            and processed < _FAST_START_DEFERRED_SNAPSHOT_FILLS_PER_TICK
        ):
            ev = self._deferred_private_snapshot_fills[0]
            # Drop fills that might be within this session (avoid impacting session-scoped totals).
            if ev.time_ms >= session_start_ms:
                self._deferred_private_snapshot_fills.popleft()
                continue

            self._deferred_private_snapshot_fills.popleft()
            try:
                fr = private_fill_event_to_hl_raw(ev)
                ingest_hl_fill_raw(
                    state=self._state,
                    storage=self._storage,
                    pnl=pnl,
                    symbol=self._settings.symbol,
                    fr=fr,
                    source="private_ws_snapshot_deferred",
                    private_inbound_timing=ev.inbound_timing,
                )
                self._finalize_private_inbound_timing(ev)
            except Exception:
                logger.exception(
                    "fast_start_snapshot_deferred_ingest_failed fill_id=%s",
                    getattr(ev, "fill_id", None),
                )
            processed += 1

    def _enrich_private_event_at_dequeue(self, ev: Any, dequeue_mono: float) -> Any:
        if isinstance(ev, (PrivateFillEvent, PrivateOrderUpdateEvent)) and ev.inbound_timing is not None:
            it = ev.inbound_timing
            merged = replace(
                it,
                dequeue_mono=dequeue_mono,
                handler_start_mono=dequeue_mono,
            )
            return replace(ev, inbound_timing=merged)
        return ev

    def _dispatch_private_event(self, ev: Any, pnl: PnlTracker | None) -> None:
        if isinstance(ev, PrivateWsConnectionEvent):
            self._handle_private_ws_connection(ev)
        elif isinstance(ev, PrivateFillEvent):
            self._handle_private_fill(ev, pnl)
        elif isinstance(ev, PrivateOrderUpdateEvent):
            self._handle_private_order_update(ev)
        else:
            logger.warning("private_event_unknown_type %s", type(ev).__name__)

    def _handle_private_ws_connection(self, ev: PrivateWsConnectionEvent) -> None:
        sym = self._settings.symbol
        if ev.kind == PrivateWsConnectionKind.CONNECTED:
            self._private_ws_healthy = True
            self._force_open_orders_reconcile = True
            # Only schedule the REST fill-catchup burst if this is a
            # true RECONNECT (we previously saw DISCONNECTED). On a
            # first-time CONNECT of a fresh process there are no
            # missed fills to catch up on, and the catchup burst
            # combined with normal startup reconciles blows past
            # OKX's per-endpoint rate limit (10/2s on /positions and
            # /balance). See bug-018 / 2026-05-05 OKX bring-up.
            is_reconnect = self._private_ws_seen_disconnect
            if is_reconnect:
                catchup = self._settings.private_ws_recovery_fill_catchup_ticks
                self._rest_fill_catchup_until_tick = (
                    self._bot_tick_counter + catchup
                )
                logger.info(
                    "private_ws_recovered scheduling_rest_catchup_ticks=%s until_tick=%s",
                    catchup,
                    self._rest_fill_catchup_until_tick,
                )
            else:
                # First-time connect: log but skip catchup. The normal
                # startup-reconcile path covers the initial state read.
                logger.info(
                    "private_ws_first_connect catchup_skipped (no prior disconnect; nothing to catch up)"
                )
            with self._state._lock:
                self._state.private_ws_connected = True
                self._state.private_ws_healthy = True
        elif ev.kind == PrivateWsConnectionKind.DISCONNECTED:
            self._private_ws_healthy = False
            # Mark that we've seen at least one DISCONNECTED so the
            # next CONNECTED is treated as a real reconnect (eligible
            # for catchup).
            self._private_ws_seen_disconnect = True
            logger.warning("private_ws_disconnected_exec detail=%s", ev.detail[:400])
            with self._state._lock:
                self._state.private_ws_connected = False
                self._state.private_ws_healthy = False
        elif ev.kind == PrivateWsConnectionKind.ERROR:
            logger.warning("private_ws_error_event detail=%s", ev.detail[:400])
        elif ev.kind == PrivateWsConnectionKind.RECONNECT_SCHEDULED:
            logger.info(
                "private_ws_reconnect_scheduled backoff_s=%s detail=%s",
                ev.backoff_seconds,
                ev.detail[:200],
            )

    def _handle_private_fill(self, ev: PrivateFillEvent, pnl: PnlTracker | None) -> None:
        if ev.coin != self._settings.symbol:
            return
        if (
            self._settings.fast_start_skip_historical_fill_replay
            and ev.is_snapshot
            and self._state.bot_status == BotStatus.STARTING
        ):
            with self._state._lock:
                self._state.startup_historical_fill_replay_skipped = True
                self._state.startup_private_snapshot_fills_skipped_count += 1
            self._deferred_private_snapshot_fills.append(ev)
            return
        fr = private_fill_event_to_hl_raw(ev)
        logger.info(
            "private_fill_dispatch fill_id=%s oid=%s isSnapshot=%s sz=%s",
            ev.fill_id,
            ev.oid,
            ev.is_snapshot,
            ev.sz,
        )
        ingest_hl_fill_raw(
            state=self._state,
            storage=self._storage,
            pnl=pnl,
            symbol=self._settings.symbol,
            fr=fr,
            source="private_ws",
            private_inbound_timing=ev.inbound_timing,
        )
        self._finalize_private_inbound_timing(ev)

    def _finalize_private_inbound_timing(
        self,
        ev: PrivateFillEvent | PrivateOrderUpdateEvent,
    ) -> None:
        if ev.inbound_timing is None:
            return
        done = time.perf_counter()
        merged = replace(
            ev.inbound_timing,
            handler_end_mono=done,
            state_apply_mono=done,
        )
        dm = private_timing_derived_ms(merged)
        self._state.note_private_inbound_metrics(merged, dm)

    def _map_hl_ws_order_status(
        self, status: str, remaining_sz: float, orig_sz: float
    ) -> tuple[Optional[OrderStatus], bool]:
        """Returns (target_status, is_terminal).

        Permissive classification (1.2.68): only KNOWN-TERMINAL statuses
        (filled, rejected, ``*canceled``) map to terminal local states.
        Anything else — including names we've never seen before — is
        classified as resting (``ACKED`` / ``PARTIAL`` / ``FILLED`` based
        on remaining-vs-orig sizes).

        This unblocks unknown intermediate states the venue may add
        without an adapter patch. Risk: a venue adding a new TERMINAL
        state (e.g. ``expired``, ``mmp_paused``) would briefly be
        mis-classified as resting until the periodic REST reconcile
        catches up (10 s).

        Mitigation: ``_seen_order_statuses`` records each distinct
        raw status string. The first time a new value appears, we log
        it at INFO with ``event=novel_order_status``. Grep journalctl
        periodically for that event; if a NEW string shows up that
        SHOULD be terminal, add it to the explicit terminal whitelist
        above. See ``BUGS/bug-021.md`` for the failure-mode catalogue.
        """
        s = (status or "").lower()

        # ----- Drift detection ------------------------------------------
        if s and s not in self._seen_order_statuses:
            self._seen_order_statuses.add(s)
            logger.info(
                "novel_order_status status=%r will_be_classified=%s",
                s,
                ("terminal" if s in (
                    "filled", "rejected", "canceled", "cancelled"
                ) or s.endswith("canceled") or s.endswith("cancelled")
                 else "resting (permissive default)"),
            )

        # ----- Explicit terminal classifications ------------------------
        if s == "filled":
            return OrderStatus.FILLED, True
        if s == "rejected":
            return OrderStatus.REJECTED, True
        if s in ("canceled", "cancelled") or s.endswith("canceled") or s.endswith("cancelled"):
            return OrderStatus.CANCELED, True

        # ----- Permissive default: treat as resting ---------------------
        # Covers documented OKX states (``live``, ``partially_filled``)
        # AND legacy/observed variants (``open``, ``triggered``, ``new``)
        # AND any future state we haven't seen yet — operator gets a
        # ``novel_order_status`` log line to investigate.
        if orig_sz > 1e-12 and remaining_sz <= 1e-12:
            return OrderStatus.FILLED, True
        if remaining_sz + 1e-10 < orig_sz:
            return OrderStatus.PARTIAL, False
        return OrderStatus.ACKED, False

    def _record_ws_order_terminal(self, oid: int, status_timestamp_ms: int) -> None:
        prev = self._ws_order_terminal_ts.get(oid, -1)
        self._ws_order_terminal_ts[oid] = max(prev, status_timestamp_ms)

    # ---- v1.4.68 Phase 1A: WS event buffer ----

    def _buffer_unmatched_ws_event(self, ev: PrivateOrderUpdateEvent) -> None:
        """Buffer a WS order-update event whose cloid has no matching
        local WorkingOrder yet. Drained by ``_drain_pending_ws_events_for_cloid``
        when a place response commits the OID.

        Bounded by ``_PENDING_WS_BUFFER_PER_CLOID_MAX`` per cloid and
        ``_PENDING_WS_BUFFER_TOTAL_MAX`` overall. Oldest evicted first.

        Must be called WITHOUT holding ``state._lock`` — the buffer has
        its own implicit lock via the GIL on dict mutations; the
        place-response handler will drain with the state lock held.
        """
        cloid = ev.cloid
        if not cloid:
            return
        now_mono = self._clock.monotonic()
        # Total cap — drop oldest entry across all cloids if exceeded.
        total = sum(len(v) for v in self._pending_ws_events_by_cloid.values())
        if total >= _PENDING_WS_BUFFER_TOTAL_MAX:
            # Find the oldest entry and drop it.
            oldest_cloid = None
            oldest_t = float("inf")
            for c, lst in self._pending_ws_events_by_cloid.items():
                if lst and lst[0][0] < oldest_t:
                    oldest_t = lst[0][0]
                    oldest_cloid = c
            if oldest_cloid is not None:
                self._pending_ws_events_by_cloid[oldest_cloid].pop(0)
                if not self._pending_ws_events_by_cloid[oldest_cloid]:
                    del self._pending_ws_events_by_cloid[oldest_cloid]
                self._ws_event_buffer_dropped_full_total += 1
                logger.warning(
                    "ws_event_buffer_full_dropped_oldest cloid=%s total_was=%d",
                    oldest_cloid[:24],
                    total,
                )
        lst = self._pending_ws_events_by_cloid.setdefault(cloid, [])
        # Per-cloid cap — drop oldest in this cloid's list.
        while len(lst) >= _PENDING_WS_BUFFER_PER_CLOID_MAX:
            lst.pop(0)
            self._ws_event_buffer_dropped_oldest_total += 1
            logger.warning(
                "ws_event_buffer_per_cloid_full_dropped_oldest cloid=%s",
                cloid[:24],
            )
        lst.append((now_mono, ev))
        logger.info(
            "ws_event_buffered_for_cloid cloid=%s oid=%s status=%s buffer_size=%d",
            cloid[:24],
            ev.oid,
            ev.raw_status,
            len(lst),
        )

    def _drain_pending_ws_events_for_cloid(
        self, cloid: Optional[str], pnl: PnlTracker | None = None
    ) -> int:
        """Drain buffered WS events for ``cloid`` and apply them via the
        normal dispatch path. Called from the place-response handler
        immediately after committing the WO's OID + transitioning to
        ACKED. Returns the number of events drained.

        v1.4.68 Phase 1A.

        Must be called WITHOUT holding ``state._lock``; the recursive
        dispatch into ``_handle_private_order_update`` will take it.
        """
        if not cloid:
            return 0
        lst = self._pending_ws_events_by_cloid.pop(cloid, None)
        if not lst:
            return 0
        # Sort by status_timestamp_ms so we apply them in venue order.
        lst.sort(key=lambda pair: pair[1].status_timestamp_ms)
        applied = 0
        for _ts_mono, ev in lst:
            try:
                self._handle_private_order_update(ev)
                applied += 1
            except Exception:
                logger.exception(
                    "ws_event_buffered_replay_failed cloid=%s oid=%s status=%s",
                    cloid[:24],
                    ev.oid,
                    ev.raw_status,
                )
        if applied:
            self._ws_event_buffered_replay_applied_total += applied
            logger.info(
                "ws_event_buffered_replay_drained cloid=%s applied=%d",
                cloid[:24],
                applied,
            )
        return applied

    def _sweep_pending_ws_event_buffer(self) -> None:
        """Drop buffered events older than the TTL. Rate-limited to once
        per ``_PENDING_WS_BUFFER_SWEEP_INTERVAL_SECONDS``.

        v1.4.68 Phase 1A. A buffered event that lingers past the TTL
        indicates the venue acked an order via WS but the bot never
        registered the matching cloid locally — typically because the
        place HTTP timed out or was never sent. Such events are
        unrecoverable and dropped with INFO logging.
        """
        now_mono = self._clock.monotonic()
        if (
            now_mono - self._ws_event_buffer_last_sweep_mono
            < _PENDING_WS_BUFFER_SWEEP_INTERVAL_SECONDS
        ):
            return
        self._ws_event_buffer_last_sweep_mono = now_mono
        ttl = _PENDING_WS_BUFFER_TTL_SECONDS
        for cloid in list(self._pending_ws_events_by_cloid.keys()):
            lst = self._pending_ws_events_by_cloid[cloid]
            kept = [(t, ev) for (t, ev) in lst if now_mono - t < ttl]
            dropped = len(lst) - len(kept)
            if dropped:
                self._ws_event_buffer_swept_stale_total += dropped
                logger.info(
                    "ws_event_buffer_swept_stale cloid=%s dropped=%d "
                    "remaining=%d (TTL=%.1fs)",
                    cloid[:24],
                    dropped,
                    len(kept),
                    ttl,
                )
            if kept:
                self._pending_ws_events_by_cloid[cloid] = kept
            else:
                del self._pending_ws_events_by_cloid[cloid]

    def _handle_private_order_update(self, ev: PrivateOrderUpdateEvent) -> None:
        if ev.coin != self._settings.symbol:
            return
        term_ts = self._ws_order_terminal_ts.get(ev.oid)
        if term_ts is not None and ev.status_timestamp_ms + _ORDER_WS_TS_EPS_MS < term_ts:
            logger.info(
                "private_order_update_stale_after_terminal_skipped oid=%s ts=%s terminal_ts=%s status=%s",
                ev.oid,
                ev.status_timestamp_ms,
                term_ts,
                ev.raw_status,
            )
            return
        prev = self._last_order_ws_applied.get(ev.oid)
        if prev is not None:
            prev_ts, prev_st = prev
            if ev.status_timestamp_ms + _ORDER_WS_TS_EPS_MS < prev_ts:
                logger.info(
                    "private_order_update_stale_skipped oid=%s ts=%s prev_ts=%s status=%s",
                    ev.oid,
                    ev.status_timestamp_ms,
                    prev_ts,
                    ev.raw_status,
                )
                return
            if (
                ev.status_timestamp_ms == prev_ts
                and ev.raw_status == prev_st
            ):
                logger.info(
                    "private_order_update_duplicate_skipped oid=%s ts=%s status=%s",
                    ev.oid,
                    ev.status_timestamp_ms,
                    ev.raw_status,
                )
                return

        # Match by exchange oid first. If that misses, fall back to cloid:
        # GRVT's synchronous ``create_order`` response does not always carry
        # the real oid ("Filled by GRVT Backend") — so our local WO stays with
        # ``order_id_exchange=None`` until a subsequent reconcile_bind_by_cloid
        # lands. WS order updates arrive earlier than that bind, and without
        # this fallback they were silently discarded. Binding the oid here
        # (late-bind) means the order-state machine finally tracks the order.
        with self._state._lock:
            # 1.3.130 multi-rung Phase 2: iterate ALL rungs across both
            # sides. At N=1 this is exactly two WOs (inside bid + inside
            # ask) — same effective coverage as the previous
            # working_bid/working_ask check. At N>1 outer rungs are
            # also scanned so a fill or cancel on a (BUY, 2) outer
            # bid is correctly attributed.
            wo = None
            all_wos = self._state.all_working_orders()
            for candidate in all_wos:
                if candidate.order_id_exchange == ev.oid:
                    wo = candidate
                    break
            if wo is None and ev.cloid:
                for candidate in all_wos:
                    if (
                        not candidate.order_id_exchange
                        and candidate.client_order_id
                        and cloids_match(candidate.client_order_id, ev.cloid)
                    ):
                        candidate.order_id_exchange = ev.oid
                        wo = candidate
                        logger.info(
                            "private_order_update_bound_by_cloid side=%s oid=%s cloid=%s level_idx=%d",
                            candidate.side.value,
                            ev.oid,
                            candidate.client_order_id,
                            int(getattr(candidate, "level_idx", 0) or 0),
                        )
                        break
        if wo is None:
            # v1.4.68 wedge-elimination-cleanup Phase 1A: instead of
            # silently dropping the event, buffer it by cloid. The
            # place-response handler will drain the buffer when it
            # commits the OID to the matching WO.
            #
            # Without the buffer, a WS ``live`` event that arrives in
            # the queue BEFORE the place-response handler completes
            # (sub-millisecond race observed in snapshot
            # v1.4.67-260518-195033) is lost, leaving the WO stuck in
            # SENT until the (much later) reconcile-by-cloid path
            # rescues it 20-33 s later.
            self._ws_event_unmatched_to_local_wo_total += 1
            # v1.4.96 TIER 3 (WARN) — late-WS-arrival detection. If
            # the unmatched oid was previously terminated by reconcile
            # (tier-1 RED or tier-2 WARN), the WS terminal is just
            # late. Record it in the ws_arrived_late ring so the
            # operator sees the symmetry: "X events declared gone
            # without WS, Y events received late WS afterwards."
            # The match also removes the oid from
            # ``terminated_oids_recent`` so a single oid can't be
            # counted as both tier-2 (http_acked_no_ws) AND tier-3
            # (ws_arrived_late) — tier-2 already counted it once at
            # reconcile time. Tier-3 is the "vindication" record.
            if ev.oid:
                self._check_ws_arrived_late_for_terminated_oid(
                    str(ev.oid), getattr(ev, "raw_status", None)
                )
            if ev.cloid:
                self._buffer_unmatched_ws_event(ev)
            else:
                logger.debug(
                    "private_order_update_no_local_working oid=%s status=%s "
                    "(no cloid; cannot buffer)",
                    ev.oid,
                    ev.raw_status,
                )
            # Always run the sweeper after a miss (rate-limited).
            self._sweep_pending_ws_event_buffer()
            return

        mapped, terminal = self._map_hl_ws_order_status(
            ev.status, ev.remaining_sz, ev.orig_sz
        )
        if mapped is None:
            logger.warning(
                "private_order_update_unknown_status oid=%s status=%r applying_size_only",
                ev.oid,
                ev.raw_status,
            )
            with self._state._lock:
                # 1.3.130 multi-rung Phase 2: same per-rung scan as the
                # binding block above — match by oid across all rungs.
                for w in self._state.all_working_orders():
                    if w.order_id_exchange == ev.oid:
                        if ev.remaining_sz + 1e-12 < w.size:
                            transition(w, OrderStatus.PARTIAL)
                            w.size = ev.remaining_sz
                            self.persist(w)
                        break
            self._last_order_ws_applied[ev.oid] = (ev.status_timestamp_ms, ev.raw_status)
            self._finalize_private_inbound_timing(ev)
            return

        logger.info(
            "private_order_update_applied oid=%s hl_status=%s -> local=%s remaining=%s orig=%s",
            ev.oid,
            ev.raw_status,
            mapped.value,
            ev.remaining_sz,
            ev.orig_sz,
        )

        trw = self._outbound_traces.get(wo.order_id_local)
        if trw is not None and trw.first_private_ws_lifecycle_perf is None:
            trw.first_private_ws_lifecycle_perf = time.perf_counter()
            m = trw.to_metrics_dict(now_perf=time.perf_counter())
            al = m.get("ack_to_first_lifecycle_ms")
            if al is not None:
                with self._state._lock:
                    self._state.outbound_ack_to_private_ws_lifecycle_ms = float(al)

        with self._state._lock:
            # 1.3.130 multi-rung Phase 2: per-rung scan + per-rung clear.
            # ``target`` is the matching WO across all rungs both sides;
            # terminal transitions clear the specific (side, level_idx)
            # slot rather than always rung 0.
            target = None
            for candidate in self._state.all_working_orders():
                if candidate.order_id_exchange == ev.oid:
                    target = candidate
                    break
            if target is None:
                return
            target_level_idx = int(getattr(target, "level_idx", 0) or 0)
            if mapped == OrderStatus.PARTIAL:
                if target.status != OrderStatus.CANCEL_PENDING:
                    transition(target, OrderStatus.PARTIAL)
                target.size = ev.remaining_sz
                self.persist(target)
            elif mapped == OrderStatus.ACKED:
                if target.status == OrderStatus.CANCEL_PENDING:
                    pass
                elif target.status == OrderStatus.SENT:
                    transition(target, OrderStatus.ACKED)
                target.size = ev.remaining_sz
                self.persist(target)
            elif mapped == OrderStatus.FILLED:
                transition(target, OrderStatus.FILLED, "ws_order_status")
                self.persist(target)
                self._state.set_working_order(target.side, target_level_idx, None)
                self._clear_side_unresolved(target.side, reason="ws_terminal_filled")
                self._last_order_ws_applied.pop(ev.oid, None)
                self._record_ws_order_terminal(ev.oid, ev.status_timestamp_ms)
            elif mapped == OrderStatus.CANCELED:
                # 1.4.0 cancel-prio Phase 0.5: capture OKX's server-
                # side cancel timestamp (``uTime`` ms epoch, surfaced
                # via PrivateOrderUpdateEvent.status_timestamp_ms) on
                # the WO before persist. Sanity-check anchor: drift vs
                # ``ts_cancel_acked`` (HTTP receive wall clock) reveals
                # clock skew + WS path length. NULL when the venue
                # event omits a timestamp (non-OKX adapters; legacy
                # rows). One-line copy of a value already in scope.
                if ev.status_timestamp_ms is not None:
                    try:
                        target.venue_cancel_utime_ms = int(
                            ev.status_timestamp_ms
                        )
                    except (TypeError, ValueError):
                        pass
                # v1.4.26 crossing-place hot-loop fix (Fix C):
                # Detect the "silent post-only-would-cross at match-
                # time" pattern. OKX accepts the place, then within
                # ~ms the matching engine sees the order would cross
                # the spread and silently cancels it (no sCode 51604
                # in the place response; the cancel arrives via the
                # user-data-WS). Without explicit detection the bot
                # treats it as a normal cancel and immediately places
                # a fresh order at the same crossing price → 100/sec
                # place-cancel-place-cancel loop (observed in snapshot
                # v1.4.25-260517-202432: 1681 short-lived orders, all
                # canceled within <5ms of ack).
                #
                # Heuristic: ts_ack just set (<100ms) AND no cancel
                # was requested by the bot → almost certainly a venue
                # silent cancel. Arm the post-only-cross cooldown for
                # the side; ``_post_only_cross_cooldown_active``
                # already suppresses new places on that side for the
                # cooldown duration. Cuts the loop to ~1 attempt per
                # cooldown_seconds instead of 1/tick.
                # NOTE: do NOT use ``or 100.0`` as a default — the
                # operator-set value of 0.0 disables the detector and
                # ``0.0 or X`` would always coerce to X (0.0 is falsy).
                raw_threshold = getattr(
                    self._settings,
                    "fast_cancel_post_only_cross_threshold_ms",
                    100.0,
                )
                fast_cancel_threshold_ms = (
                    float(raw_threshold)
                    if raw_threshold is not None
                    else 100.0
                )
                # v1.4.34 (Codex #1 fix): skip the fast-cancel
                # heuristic for orders adopted from the venue's
                # open-orders snapshot at startup. Such orders have a
                # synthetic-or-venue-derived ``ts_ack`` that does NOT
                # represent a fresh local place — a cancel arriving
                # within the fast-cancel window after process start
                # is far more likely to be an external trigger
                # (Binance-cross-venue, operator action on the venue
                # UI, exchange sweep) than a post-only cross reject.
                # The previous behaviour falsely armed the
                # ``post_only_cross_cooldown`` on every restart-with-
                # open-orders, suppressing legitimate quoting for the
                # cooldown window.
                if (
                    target.ts_ack is not None
                    and target.ts_cancel_requested is None
                    and fast_cancel_threshold_ms > 0
                    and not getattr(target, "hydrated_from_exchange", False)
                ):
                    age_ms = (
                        self._clock.now_utc() - _utc_aware(target.ts_ack)
                    ).total_seconds() * 1000.0
                    if 0 <= age_ms <= fast_cancel_threshold_ms:
                        # Inferred post-only-cross at match time.
                        # Arm the cooldown + bump the counter so the
                        # operator can see the rate of this pattern.
                        self._arm_post_only_cross_cooldown(
                            target.side,
                            rejected_price=getattr(target, "price", None),
                        )
                        try:
                            self._state.quote_quality.note_post_only_cross_rejection()
                        except Exception:
                            logger.exception(
                                "note_post_only_cross_rejection_failed_ws"
                            )
                        logger.info(
                            "inferred_post_only_cross_at_match_time "
                            "side=%s ord_id=%s age_since_ack_ms=%.1f "
                            "threshold_ms=%.1f price=%s size=%s — "
                            "armed post_only_cross_cooldown",
                            target.side.value,
                            target.order_id_exchange,
                            age_ms,
                            fast_cancel_threshold_ms,
                            target.price,
                            target.size,
                        )
                transition(target, OrderStatus.CANCELED, f"ws:{ev.raw_status}")
                self.persist(target)
                self._state.set_working_order(target.side, target_level_idx, None)
                self._clear_side_unresolved(target.side, reason="ws_terminal_canceled")
                self._last_order_ws_applied.pop(ev.oid, None)
                self._record_ws_order_terminal(ev.oid, ev.status_timestamp_ms)
            elif mapped == OrderStatus.REJECTED:
                transition(target, OrderStatus.REJECTED, "ws_rejected")
                self.persist(target)
                self._state.set_working_order(target.side, target_level_idx, None)
                self._clear_side_unresolved(target.side, reason="ws_terminal_rejected")
                self._last_order_ws_applied.pop(ev.oid, None)
                self._record_ws_order_terminal(ev.oid, ev.status_timestamp_ms)

        self._finalize_private_inbound_timing(ev)
        if not terminal:
            self._last_order_ws_applied[ev.oid] = (ev.status_timestamp_ms, ev.raw_status)

    def _rest_reconcile_reason(self) -> Optional[str]:
        if self._force_open_orders_reconcile:
            return "ws_recovery_or_forced"
        if not self._settings.private_ws_enabled or self._private_q is None:
            return "private_ws_disabled"
        if not self._private_ws_healthy:
            return "private_ws_unhealthy"
        if self._bot_tick_counter <= 2:
            return "bootstrap_ticks"
        with self._state._lock:
            cancel_pending = any(
                wo and wo.status == OrderStatus.CANCEL_PENDING
                for wo in (
                    self._state.get_working_order(Side.BUY, 0),
                    self._state.get_working_order(Side.SELL, 0),
                )
            )
        if cancel_pending:
            w = self._settings.cancel_pending_rest_watchdog_ticks
            if w <= 1 or self._bot_tick_counter % w == 0:
                return "cancel_pending_watchdog"
        if self.has_order_state_uncertainty():
            return "side_unresolved_or_desync"
        now_m = self._clock.monotonic()
        if now_m + 1e-9 >= self._next_interval_reconcile_request_mono:
            iv_s = max(0.1, float(self._settings.open_orders_reconcile_request_interval_seconds))
            self._next_interval_reconcile_request_mono = now_m + iv_s
            self._reconcile_interval_requests_total += 1
            return "interval_time"
        return None

    @staticmethod
    def _is_rate_limited_exception(exc: BaseException) -> bool:
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        resp = getattr(exc, "response", None)
        if resp is not None and getattr(resp, "status_code", None) == 429:
            return True
        txt = str(exc).lower()
        return "429" in txt or "too many requests" in txt or "rate limit" in txt

    def _should_reconcile_open_orders(self, *, force: bool, emergency: bool) -> tuple[bool, str]:
        """
        Open-orders reconcile is a rare safety sync, never a hot-path control-loop action.
        Most requests obey one global cooldown. During active side convergence/desync, force/emergency
        requests may use a tighter bounded retry lane to avoid long blind windows.
        """
        if self._open_orders_reconcile_in_flight:
            return False, "in_flight"
        now_m = self._clock.monotonic()
        next_allowed = self._open_orders_reconcile_next_allowed_mono
        uncertain = self.has_order_state_uncertainty()
        if uncertain and (force or emergency):
            next_allowed = min(next_allowed, self._desync_reconcile_next_allowed_mono)
        if now_m + 1e-9 < next_allowed:
            return False, "cooldown"
        if uncertain and (force or emergency):
            return True, "allowed_desync_lane"
        return True, "allowed_force_or_emergency" if (force or emergency) else "allowed"

    def _run_open_orders_reconcile(self, *, reason: str, emergency: bool = False) -> None:
        if self._open_orders_reconcile_in_flight:
            return
        self._open_orders_reconcile_in_flight = True
        self._reconcile_runs_total += 1
        t0 = time.perf_counter()
        logger.info("reconcile_started reason=%s emergency=%s", reason, emergency)
        outcome = "error"
        try:
            uncertain_before = self.has_order_state_uncertainty()
            if uncertain_before:
                self._desync_recovery_attempts += 1
                log_extra(
                    logger,
                    logging.INFO,
                    "reconcile_desync_recovery_attempt",
                    {
                        "reason": reason,
                        "emergency": emergency,
                        "active_unresolved_sides": [s.value for s in self._active_unresolved_sides()],
                        "order_desync": bool(self._state.order_desync),
                    },
                )
            outcome = self._sync_open_orders_impl()
            now_m = self._clock.monotonic()
            if outcome == "rate_limited":
                self._open_orders_reconcile_next_allowed_mono = max(
                    self._open_orders_reconcile_next_allowed_mono,
                    now_m + float(self._settings.open_orders_reconcile_429_backoff_seconds),
                )
            else:
                self._open_orders_reconcile_next_allowed_mono = max(
                    self._open_orders_reconcile_next_allowed_mono,
                    now_m + float(self._settings.open_orders_reconcile_cooldown_seconds),
                )
            if uncertain_before or self.has_order_state_uncertainty():
                if outcome == "rate_limited":
                    self._desync_reconcile_next_allowed_mono = now_m + float(
                        self._settings.desync_reconcile_429_backoff_seconds
                    )
                else:
                    self._desync_reconcile_next_allowed_mono = now_m + float(
                        self._settings.desync_reconcile_interval_seconds
                    )
        finally:
            self._open_orders_reconcile_in_flight = False
            dur_ms = (time.perf_counter() - t0) * 1000.0
            self._last_tick_reconcile_rest_ms = dur_ms
            logger.info(
                "reconcile_completed reason=%s emergency=%s duration_ms=%.2f outcome=%s "
                "next_allowed_in_s=%.2f desync_next_allowed_in_s=%.2f "
                "open_orders_rest_calls_total=%s open_orders_rate_limited_total=%s",
                reason,
                emergency,
                dur_ms,
                outcome,
                max(0.0, self._open_orders_reconcile_next_allowed_mono - self._clock.monotonic()),
                max(0.0, self._desync_reconcile_next_allowed_mono - self._clock.monotonic()),
                self._open_orders_rest_calls_total,
                self._open_orders_rate_limited_total,
            )

    def request_open_orders_reconcile(
        self,
        *,
        reason: str,
        force: bool = False,
        emergency: bool = False,
    ) -> None:
        # force/emergency are metadata only; both obey the same global cooldown gate.
        self._reconcile_requests_total += 1
        # Log the request at DEBUG — the **accepted** path re-logs as
        # ``reconcile_started`` with the same reason (and the same
        # emergency flag), so info-level ``reconcile_requested`` logs
        # are pure duplication of accepted reconciles AND pure noise
        # for rejected ones (cooldown / in_flight).
        #
        # Measured on a 4-minute flooded run: 9,773 INFO-level
        # ``reconcile_requested`` lines (33.2% of log bytes) but only
        # 37 actual reconciles started — i.e. 99.6% of those lines
        # were redundant. Counter still increments so
        # ``reconcile_requests_total`` in telemetry is unchanged.
        logger.debug(
            "reconcile_requested reason=%s force=%s emergency=%s",
            reason,
            force,
            emergency,
        )
        ok, cause = self._should_reconcile_open_orders(force=force, emergency=emergency)
        if not ok:
            if cause == "cooldown":
                self._reconcile_skip_cooldown_total += 1
            elif cause == "in_flight":
                self._reconcile_skip_inflight_total += 1
            # Skipped path is DEBUG for cooldown/in_flight (expected),
            # INFO otherwise (other causes are rare and worth seeing).
            lvl = logging.DEBUG if cause in ("cooldown", "in_flight") else logging.INFO
            logger.log(lvl, "reconcile_skipped reason=%s cause=%s", reason, cause)
            return
        self._run_open_orders_reconcile(reason=reason, emergency=emergency)

    def maybe_sync_open_orders(self) -> None:
        reason = self._rest_reconcile_reason()
        if reason is None:
            logger.debug(
                "rest_open_orders_reconcile_not_due private_ws_healthy=%s tick=%s next_interval_in_s=%.2f",
                self._private_ws_healthy,
                self._bot_tick_counter,
                max(0.0, self._next_interval_reconcile_request_mono - self._clock.monotonic()),
            )
            return
        was_forced = self._force_open_orders_reconcile
        self._force_open_orders_reconcile = False
        side_unresolved = reason == "side_unresolved_or_desync"
        self.request_open_orders_reconcile(
            reason=reason,
            force=bool(was_forced or side_unresolved),
            emergency=bool(side_unresolved),
        )

    def sync_open_orders(self, *, force: bool = False, emergency: bool = False) -> None:
        if force:
            self.request_open_orders_reconcile(reason="force", force=True, emergency=emergency)
            return
        self.request_open_orders_reconcile(reason="manual_or_periodic", force=False, emergency=emergency)

    def get_reconcile_runtime_counters(self) -> dict[str, Any]:
        return {
            "reconcile_requests_total": int(self._reconcile_requests_total),
            "reconcile_interval_requests_total": int(self._reconcile_interval_requests_total),
            "reconcile_runs_total": int(self._reconcile_runs_total),
            "reconcile_skips_cooldown_total": int(self._reconcile_skip_cooldown_total),
            "reconcile_skips_in_flight_total": int(self._reconcile_skip_inflight_total),
            "open_orders_rest_calls_total": int(self._open_orders_rest_calls_total),
            "open_orders_rate_limited_total": int(self._open_orders_rate_limited_total),
            "duplicate_open_detect_count": int(self._duplicate_open_detect_count),
            "side_unresolved_enter_count": int(self._side_unresolved_enter_count),
            "cancel_pending_timeout_count": int(self._cancel_pending_timeout_count),
            "suppress_place_due_unresolved_count": int(
                self._suppress_place_due_unresolved_count
            ),
            "desync_recovery_attempts": int(self._desync_recovery_attempts),
            "active_unresolved_sides": [s.value for s in self._active_unresolved_sides()],
            "active_unresolved_reasons": {
                s.value: self._side_unresolved_reason.get(s)
                for s in (Side.BUY, Side.SELL)
                if self._is_side_unresolved(s)
            },
        }

    def get_quote_exec_telemetry(self) -> dict[str, Any]:
        # Filter engine/legacy diagnostic keys (names say "finalize"/exec_*; not a second quote pass).
        internal_only = {
            "final_submitted_bid_px",
            "final_submitted_ask_px",
            "pre_finalize_norm_bid_px",
            "pre_finalize_norm_ask_px",
            "post_only_adjustment_applied_bid",
            "post_only_adjustment_applied_ask",
            "economic_floor_requested_bid",
            "economic_floor_requested_ask",
            "economic_floor_blocked_by_post_only_bid",
            "economic_floor_blocked_by_post_only_ask",
            "normal_mm_cycle_valid",
            "normal_mm_downgrade_reason",
            "normal_mm_bid_contract_ok",
            "normal_mm_ask_contract_ok",
            "downgraded_cycle_fallback_attempted",
            "downgraded_cycle_fallback_placed",
            "downgraded_cycle_skipped_reason",
            "downgraded_side_target",
            "downgraded_side_executable",
            "bid_executable",
            "ask_executable",
            "bid_executable_notional_usd",
            "ask_executable_notional_usd",
            "bid_min_notional_block",
            "ask_min_notional_block",
            "quote_contract_build_ms",
            "executable_size_check_ms",
            "placement_mode_eval_ms",
            "finalize_ms",
            "order_submit_prep_ms",
        }
        return {
            k: v
            for k, v in self._quote_exec_telemetry.items()
            if not k.startswith("exec_finalize_")
            and not k.startswith("exec_normal_mm_")
            and not k.startswith("quote_engine_")
            and k not in internal_only
        }

    def _diagnostic_model_quote_prices_sizes(
        self,
        decision: QuoteDecision,
        bid_mult: float,
        ask_mult: float,
        spread_add_bps: float,
    ) -> tuple[float, float, float, float]:
        """Raw bid/ask after multipliers and inventory clip — **diagnostics only**.

        Not used for runtime order construction. Live desired orders come only from
        ``QuoteEngine.build_quotes``. This mirrors a simplified model path for
        ``build_quote_execution_telemetry`` / dry-run rows (e.g. quote_decisions preview).
        """
        bid_px = decision.quoted_bid * (1.0 - spread_add_bps / 10_000.0)
        ask_px = decision.quoted_ask * (1.0 + spread_add_bps / 10_000.0)
        bid_sz = decision.quoted_bid_sz * bid_mult
        ask_sz = decision.quoted_ask_sz * ask_mult
        bid_sz = clip(bid_sz, 0.0, self._settings.max_order_notional_usd / max(bid_px, 1e-12))
        ask_sz = clip(ask_sz, 0.0, self._settings.max_order_notional_usd / max(ask_px, 1e-12))
        with self._state._lock:
            pos_qty = self._state.position.position_qty
            # v1.4.194: migrated off the deprecated property shims.
            wb = self._state.get_working_order(Side.BUY, 0)
            wa = self._state.get_working_order(Side.SELL, 0)
        bid_sz, ask_sz = self._clip_entry_sizes(
            pos_qty,
            bid_sz,
            ask_sz,
            resting_bid_sz=resting_clip_size_for_position_headroom(wb),
            resting_ask_sz=resting_clip_size_for_position_headroom(wa),
            bid_price=bid_px,
            ask_price=ask_px,
        )
        return bid_px, bid_sz, ask_px, ask_sz

    def _diagnostic_quote_execution_telemetry_from_prices_sizes(
        self,
        bid_px: float,
        bid_sz: float,
        ask_px: float,
        ask_sz: float,
    ) -> dict[str, Any]:
        """Normalized preview from *model* prices/sizes — not live ``QuoteEngine`` output."""
        spec = self._client.symbol_spec
        tick = float(spec.price_tick)
        step = float(spec.size_step)
        out: dict[str, Any] = {
            "exec_raw_bid_px": bid_px if bid_px > 0 and bid_sz > 0 else None,
            "exec_raw_bid_sz": bid_sz if bid_px > 0 and bid_sz > 0 else None,
            "exec_raw_ask_px": ask_px if ask_px > 0 and ask_sz > 0 else None,
            "exec_raw_ask_sz": ask_sz if ask_px > 0 and ask_sz > 0 else None,
            "exec_norm_bid_px": None,
            "exec_norm_bid_sz": None,
            "exec_norm_ask_px": None,
            "exec_norm_ask_sz": None,
            "exec_price_tick": tick,
            "exec_size_step": step,
            "exec_meta_decimal_grid_price_tick": tick,
            "exec_meta_decimal_size_step": step,
            "exec_hl_max_sig_figs_nonint": int(HL_PERP_MAX_SIG_FIGS),
            "exec_price_normalize_pipeline": HL_PERP_LIMIT_PRICE_PIPELINE_ID,
            "exec_wire_bid_limit_p": None,
            "exec_wire_ask_limit_p": None,
        }
        mq = float(self._settings.min_quote_notional_usd)
        if bid_px > 0 and bid_sz > 0:
            bp, _rej = normalize_order_pair(spec, bid_px, bid_sz)
            if bp:
                ntn = bp[0] * bp[1]
                if ntn + 1e-9 < mq:
                    logger.info(
                        "quote_side_suppressed_below_min_quote_notional side=BUY raw_px=%s raw_sz=%s "
                        "normalized_notional_usd=%s min_quote_notional_usd=%s",
                        bid_px,
                        bid_sz,
                        ntn,
                        mq,
                    )
                else:
                    out["exec_norm_bid_px"] = bp[0]
                    out["exec_norm_bid_sz"] = bp[1]
                    out["exec_wire_bid_limit_p"] = wire_format_preview_limit_px(bp[0])
        if ask_px > 0 and ask_sz > 0:
            ap, _rej = normalize_order_pair(spec, ask_px, ask_sz)
            if ap:
                ntn = ap[0] * ap[1]
                if ntn + 1e-9 < mq:
                    logger.info(
                        "quote_side_suppressed_below_min_quote_notional side=SELL raw_px=%s raw_sz=%s "
                        "normalized_notional_usd=%s min_quote_notional_usd=%s",
                        ask_px,
                        ask_sz,
                        ntn,
                        mq,
                    )
                else:
                    out["exec_norm_ask_px"] = ap[0]
                    out["exec_norm_ask_sz"] = ap[1]
                    out["exec_wire_ask_limit_p"] = wire_format_preview_limit_px(ap[0])
        return out

    def build_quote_execution_telemetry(
        self,
        decision: QuoteDecision,
        bid_mult: float,
        ask_mult: float,
        spread_add_bps: float,
    ) -> dict[str, Any]:
        """Dry-run / DB preview: ``exec_*`` fields from a simplified model path.

        **Not** the same as ``QuoteEngine.build_quotes`` output. For live cycle truth,
        use telemetry keyed from the engine (``quote_engine_*``, ``final_submitted_*``)
        after ``maybe_refresh_quotes``. Runtime submit uses only ``QuoteEngine`` →
        ``_submit_passive_order_verbatim``.
        """
        bid_px, bid_sz, ask_px, ask_sz = self._diagnostic_model_quote_prices_sizes(
            decision, bid_mult, ask_mult, spread_add_bps
        )
        return self._diagnostic_quote_execution_telemetry_from_prices_sizes(
            bid_px, bid_sz, ask_px, ask_sz
        )

    def _compute_executable_quote_size(
        self,
        *,
        price: float,
        desired_size: float,
        min_quote_notional_usd: float,
    ) -> tuple[float, bool, Optional[float], bool]:
        """
        Return (size_for_pipeline, executable, executable_notional_usd, min_notional_block).
        Keeps checks deterministic and local to hot-path contract build.
        """
        if price <= 0 or desired_size <= 0:
            return 0.0, False, None, False
        sp = self._client.symbol_spec
        min_sz = max(float(sp.min_size), float(sp.size_step))
        size_req = max(float(desired_size), min_sz)
        normed, rej = normalize_order_pair(sp, float(price), float(size_req))
        if normed is None:
            return 0.0, False, None, bool(rejection_is_below_min_notional_usd(rej))
        npx, nsz = normed
        ntn = float(npx) * float(nsz)
        if ntn + 1e-9 < float(min_quote_notional_usd):
            return 0.0, False, ntn, True
        if ntn + 1e-9 < float(sp.min_notional_usd):
            return 0.0, False, ntn, True
        return float(nsz), True, ntn, False

    def _maybe_emit_latency_guardrails(self, *, quote_cycle_id: str) -> None:
        with self._state._lock:
            ms = self._state.last_latency_decision_to_first_place_ms
        if ms is None:
            return
        cur = float(ms)
        self._decision_to_first_place_ms_window.append(cur)
        warn_ms = float(self._settings.execution_latency_warn_ms)
        degrade_ms = float(self._settings.execution_latency_degrade_ms)
        min_breaches = max(1, int(self._settings.execution_latency_degrade_min_breaches))
        high_count = sum(1 for x in self._decision_to_first_place_ms_window if x > degrade_ms + 1e-9)
        if cur > warn_ms + 1e-9:
            payload = {
                "event": "execution_latency_warning",
                "quote_cycle_id": quote_cycle_id,
                "latency_decision_to_first_place_ms": cur,
                "warn_threshold_ms": warn_ms,
                "degrade_threshold_ms": degrade_ms,
                "window_samples": len(self._decision_to_first_place_ms_window),
                "window_high_latency_count": high_count,
            }
            log_extra(logger, logging.WARNING, "execution_latency_warning", payload)
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.WARNING.value,
                "execution_latency_warning",
                "execution_latency_warning",
                payload,
            )
        if high_count >= min_breaches:
            payload = {
                "event": "execution_latency_sustained_high",
                "quote_cycle_id": quote_cycle_id,
                "latency_decision_to_first_place_ms": cur,
                "degrade_threshold_ms": degrade_ms,
                "window_samples": len(self._decision_to_first_place_ms_window),
                "window_high_latency_count": high_count,
                "min_breaches_for_event": min_breaches,
            }
            log_extra(logger, logging.WARNING, "execution_latency_sustained_high", payload)
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.WARNING.value,
                "execution_latency_sustained_high",
                "execution_latency_sustained_high",
                payload,
            )

    def persist(self, wo: WorkingOrder) -> None:
        with self._state._lock:
            self._state.note_passive_order_lifetime_if_new(wo)
        self._storage.insert_order_row(order_row(wo))

    def _release_working_slot_if_matches(self, wo: WorkingOrder) -> None:
        # 1.3.130 multi-rung Phase 2: release the SPECIFIC per-rung slot.
        # ``wo.level_idx`` is 0 for the inside rung (so the legacy
        # single-rung path is bit-identical); outer rungs release their
        # own slot.
        level_idx = int(getattr(wo, "level_idx", 0) or 0)
        with self._state._lock:
            cur = self._state.get_working_order(wo.side, level_idx)
            if cur is wo:
                self._state.set_working_order(wo.side, level_idx, None)
        self._clear_side_unresolved(wo.side, reason=f"working_slot_released_{wo.status.value}")

    def _try_resolve_sent_order_by_cloid(self, wo: WorkingOrder) -> None:
        """When openOrders is empty for this side, bind or clear a SENT row using orderStatus+cloid."""
        if wo.status != OrderStatus.SENT or not wo.client_order_id:
            return
        addr = self._settings.hl_account_address
        try:
            resp = self._client.query_order_status_by_cloid(addr, wo.client_order_id)
        except Exception:
            logger.exception(
                "query_order_status_by_cloid failed cloid=%s",
                wo.client_order_id[:24],
            )
            self._note_sent_order_cloid_ambiguous(wo, "orderStatus_exception", "")
            return
        oid, outcome, detail = self._interpret_order_status_response(resp)
        if outcome == "open" and oid is not None:
            self._clear_sent_ambiguous_polls(wo)
            wo.order_id_exchange = oid
            transition(wo, OrderStatus.ACKED)
            self.persist(wo)
            logger.info(
                "sent_order_bound_via_orderStatus cloid=%s oid=%s",
                wo.client_order_id[:24],
                oid,
            )
            return
        if outcome == "not_found":
            self._clear_sent_ambiguous_polls(wo)
            transition(wo, OrderStatus.REJECTED, "orderStatus:unknownOid")
            self.persist(wo)
            self._release_working_slot_if_matches(wo)
            return
        if outcome == "filled":
            self._clear_sent_ambiguous_polls(wo)
            transition(wo, OrderStatus.FILLED, "orderStatus:filled")
            self.persist(wo)
            self._release_working_slot_if_matches(wo)
            return
        if outcome == "canceled":
            self._clear_sent_ambiguous_polls(wo)
            transition(wo, OrderStatus.CANCELED, f"orderStatus:{detail}")
            self.persist(wo)
            self._release_working_slot_if_matches(wo)
            return
        if outcome == "rejected":
            self._clear_sent_ambiguous_polls(wo)
            transition(wo, OrderStatus.REJECTED, f"orderStatus:{detail}")
            self.persist(wo)
            self._release_working_slot_if_matches(wo)
            return
        if outcome in ("invalid", "transport", "unknown_proc"):
            self._note_sent_order_cloid_ambiguous(wo, f"orderStatus_{outcome}", detail)
            return

    def _sent_order_matches_remote_open(
        self, wo: WorkingOrder, remote: OpenOrderRaw, addr: str
    ) -> Optional[bool]:
        """
        True if this SENT working order is the same as the single open order on this side.
        False = definitive mismatch. None = transient / cannot confirm (no DESYNC escalation).
        """
        if not wo.client_order_id:
            return False
        if remote.cloid and cloids_match(remote.cloid, wo.client_order_id):
            self._clear_sent_ambiguous_polls(wo)
            return True
        try:
            resp = self._client.query_order_status_by_cloid(addr, wo.client_order_id)
        except Exception:
            logger.exception("query_order_status_by_cloid failed during openOrders reconcile")
            self._note_sent_order_cloid_ambiguous(wo, "openOrders_reconcile_exception", "")
            return None
        oid, outcome, detail = self._interpret_order_status_response(resp)
        if outcome == "open" and oid is not None and oid == remote.oid:
            self._clear_sent_ambiguous_polls(wo)
            return True
        if outcome == "not_found":
            return False
        if outcome == "open" and oid is not None and oid != remote.oid:
            return False
        if outcome in ("filled", "canceled", "rejected"):
            return False
        if outcome in ("invalid", "transport", "unknown_proc"):
            self._note_sent_order_cloid_ambiguous(wo, f"openOrders_reconcile_{outcome}", detail)
            return None
        return None

    @staticmethod
    def _format_raw_place_response_for_log(resp: Any) -> str:
        """Render a place-response object as a safe, truncated string
        for CRITICAL log + bot_event payloads. Bounded length so a
        single line can't blow out journald / telegram buffers; we
        only need enough context to diagnose what shape the venue
        actually returned for an unconfirmed outcome.

        2026-05-14 BUG-024.
        """
        try:
            import json as _json

            s = _json.dumps(resp, default=str)
        except Exception:
            s = repr(resp)
        if len(s) > 2000:
            s = s[:2000] + "...<truncated>"
        return s

    def _interpret_place_response(self, resp: Any) -> tuple[Optional[int], str, str]:
        parser = getattr(self._client, "interpret_place_response", None)
        if callable(parser):
            parsed = parser(resp)
            if isinstance(parsed, tuple) and len(parsed) == 3:
                return parsed
        return interpret_hl_place_order_response(resp)

    def _interpret_cancel_response(self, resp: Any) -> tuple[str, str]:
        parser = getattr(self._client, "interpret_cancel_response", None)
        if callable(parser):
            parsed = parser(resp)
            if isinstance(parsed, tuple) and len(parsed) == 2:
                return parsed
        return interpret_hl_cancel_response(resp)

    def _interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        parser = getattr(self._client, "interpret_order_status_response", None)
        if callable(parser):
            parsed = parser(resp)
            if isinstance(parsed, tuple) and len(parsed) == 3:
                return parsed
        return interpret_hl_order_status_response(resp)

    def parse_place_response(self, resp: Any) -> Optional[int]:
        oid, _, _ = self._interpret_place_response(resp)
        return oid

    def cancel_all_orders_for_symbol(self) -> None:
        """Cancel all working orders for the currently-traded symbol.

        v1.4.55 wedge-elimination Phase 1: STRUCTURAL REWRITE.

        Pre-v1.4.55 this method iterated the (1-s TTL-cached) exchange
        open-orders snapshot and fired per-order HTTP cancels in a tight
        loop. When the snapshot was stale (cancel-all called >1×/s
        during a desync window) the loop kept hitting the same already-
        gone OIDs with 51400 ``unexpected_gone`` responses with NO state
        mutation — the bot had no memory of what it tried 7 ms ago.
        Snapshot v1.4.53-260518-162051 caught 319 of these self-DoS
        firings in a 3-s desync window, all hammering the same 3 ghost
        OIDs.

        Root cause: the bot kept three views of order state (local WO
        dict, TTL-cached exchange snapshot, true exchange state) and
        cancel-all bypassed the local view that has lifecycle status.

        Phase 1 fix: cancel-all now reads from ``state.all_working_orders()``
        and routes through ``_enqueue_cancel_quote_path`` — the same
        idempotent dispatcher path that single-cancels use. CANCEL_PENDING
        IS the tombstone: once a cancel is dispatched for a WO, the
        dispatcher dedupes future cancels for the same OID until the
        cancel reaches terminal status (or the cancel-pending watchdog
        clears it).

        Behaviour:
          * Skip WOs in terminal status (CANCELED / FILLED / REJECTED) —
            already done.
          * Skip WOs in CANCEL_PENDING — already in-flight.
          * Skip WOs in NEW_LOCAL / SENT — placement not yet
            confirmed; cancelling pre-ack is unsafe. The dispatcher
            will pick them up once they reach ACKED.
          * Cancel WOs in ACKED / PARTIAL / AMEND_PENDING via the
            normal dispatcher path.

        Orphan detection (exchange OID exists with no matching local
        WO) is handled by ``_sync_open_orders_impl`` →
        ``_cancel_orphan_remote_order`` on the reconcile path. That
        path is unchanged.

        The cancel-all method is symbol-scoped by construction: it
        reads from ``state.all_working_orders()`` which is per-bot-instance
        and only ever contains the configured symbol's WOs.
        """
        if not self._client.has_write_access():
            return
        # Snapshot the WO list under lock so a concurrent WS-update
        # thread can't mutate it mid-iteration. Cheap — max 4 WOs
        # (2 sides × 2 rungs at typical config).
        with self._state._lock:
            wos_snapshot = list(self._state.all_working_orders())
        for wo in wos_snapshot:
            status = wo.status
            if status in (
                OrderStatus.CANCELED,
                OrderStatus.FILLED,
                OrderStatus.REJECTED,
                # v1.4.67: DESYNC means the exchange_mismatch path
                # already issued cancels for the orphan(s) and gave up
                # on local tracking. Re-issuing another cancel just
                # burns the venue rate limiter on a ghost OID. Treat
                # as terminal for cancel-all purposes.
                OrderStatus.DESYNC,
            ):
                # Already terminal — nothing to cancel.
                continue
            if status == OrderStatus.CANCEL_PENDING:
                # Already in-flight via the dispatcher. The
                # cancel-pending watchdog handles retry/timeout.
                continue
            if status in (OrderStatus.NEW_LOCAL, OrderStatus.SENT):
                # Pre-ack placement. We don't have a confirmed
                # exchange OID yet (or the WS ack hasn't landed).
                # Cancelling a SENT order before its ack lands can
                # race with the place response — the dispatcher will
                # cancel it once it reaches ACKED naturally.
                continue
            # Status is one of: ACKED, PARTIAL, AMEND_PENDING.
            # All three are cancellable; route through the normal
            # quote-path enqueue. ``_enqueue_cancel_quote_path``
            # transitions the WO to CANCEL_PENDING, which makes it
            # invisible to the next iteration of this loop (the
            # CANCEL_PENDING skip above) — idempotent by construction.
            self._enqueue_cancel_quote_path(
                wo, trigger_reason="cancel_all_for_symbol"
            )

    # Back-compat alias. Prefer the explicit name; this exists so any
    # out-of-tree caller still works.
    cancel_all_orders = cancel_all_orders_for_symbol

    def cancel_all_orders_for_symbol_bulk_or_fallback(self) -> str:
        """Prefer the venue's single-request bulk cancel; fall back on error.

        Returns a short outcome tag that callers can log / persist:

          * ``"bulk_ok"`` — venue has a bulk-cancel endpoint (GRVT) and the
            request succeeded.
          * ``"bulk_error_fallback_ok"`` — bulk call raised; we ran the
            per-order loop and it finished (per-order errors still counted
            individually via ``bump_execution_errors``).
          * ``"fallback_ok"`` — venue has no bulk endpoint (Hyperliquid);
            ran per-order loop.
          * ``"no_write_access"`` — adapter is read-only; nothing attempted.

        The bulk path collapses N+1 REST round trips into 1 and is
        evaluated atomically on the matching engine — any order that
        indexes *during* the request is still caught. This is
        particularly important under any supervisor's SIGTERM →
        SIGKILL shutdown, where a per-order loop could exceed the grace
        window and leave orders resting on the book.
        """
        if not self._client.has_write_access():
            return "no_write_access"
        bulk = getattr(self._client, "cancel_all_orders_bulk_for_symbol", None)
        if callable(bulk):
            try:
                bulk(self._settings.symbol)
                return "bulk_ok"
            except Exception:
                logger.exception(
                    "cancel_all_bulk_failed falling back to per-order loop"
                )
                self._state.bump_execution_errors("cancel_all_bulk_exception")
                self.cancel_all_orders_for_symbol()
                return "bulk_error_fallback_ok"
        # Venue has no bulk endpoint (e.g. Hyperliquid) — per-order loop.
        self.cancel_all_orders_for_symbol()
        return "fallback_ok"

    def reset_startup_grace_counters(self) -> None:
        """v1.4.203 — clear the counters that are allowed to tick
        once during the startup window (before private WS subscribes
        and the book has been drained). Called by
        ``app.startup_cleanup.run_startup_cleanup`` after the REST
        cancel-all + poll-until-empty completes, just before the
        private WS subscribes.

        Counters reset:

        * ``_ws_event_unmatched_to_local_wo_total`` — increments
          when a WS order-update arrives for an oid the executor
          doesn't know. At session start with no local orders any
          residual terminal from a pre-existing cancel would tick
          this — resetting AFTER the drain gives the wedge-
          acceptance gate a clean strict-zero contract for mid-
          session monitoring.

        Other startup-noise counters can be added here as the
        wedge-acceptance check set grows. Counters that genuinely
        reflect MID-SESSION state (e.g. ``wedge_episode_count_session``,
        ``gate_phase2a_invariant_violation_total``) are intentionally
        NOT reset — they should be zero from initialisation and any
        non-zero value at this point already indicates a regression.
        """
        self._ws_event_unmatched_to_local_wo_total = 0

    def verify_book_clean_on_venue(
        self,
        *,
        timeout_s: float = 15.0,
        poll_interval_s: float = 0.5,
    ) -> "CleanBookResult":
        """Poll the venue's open-orders REST endpoint until no orders
        remain for the configured symbol (or ``timeout_s`` elapses).

        Thin executor-side wrapper over
        ``app.startup_cleanup.wait_for_clean_book`` so the in-place
        drain path (``Bot.drain``) and the startup-cleanup path share
        ONE poll-until-empty implementation. Both confirm the same
        invariant — the venue REST shows zero resting orders for our
        symbol — and both rely on that helper reading the venue-neutral
        ``coin`` field (see ``wait_for_clean_book``'s note on the
        historical ``.symbol`` no-op bug).

        Unlike the startup helper's 30-s default, the drain default is a
        tighter 15-s window: at shutdown we've already issued cancel-all
        and waited for the dispatcher to drain, so the book should be
        empty within a couple of REST polls; a long timeout here just
        delays the operator's stop.

        Returns the ``CleanBookResult`` so the caller can branch on
        ``success`` / ``final_count`` (e.g. emit INFO when clean,
        WARNING when orders persist past the timeout).
        """
        from app.startup_cleanup import wait_for_clean_book

        return wait_for_clean_book(
            client=self._client,
            settings=self._settings,
            address=self._settings.hl_account_address,
            timeout_seconds=timeout_s,
            poll_interval_seconds=poll_interval_s,
        )

    def _duplicate_open_orders_diag_payload(
        self,
        *,
        sym: str,
        sym_orders: list[OpenOrderRaw],
        dup_buy: bool,
        dup_sell: bool,
    ) -> dict[str, Any]:
        with self._state._lock:
            # v1.4.194: migrated off the deprecated property shims.
            wb = self._state.get_working_order(Side.BUY, 0)
            wa = self._state.get_working_order(Side.SELL, 0)

        def _local(wo: Optional[WorkingOrder]) -> Optional[dict[str, Any]]:
            if wo is None:
                return None
            return {
                "order_id_local": wo.order_id_local,
                "order_id_exchange": wo.order_id_exchange,
                "client_order_id": wo.client_order_id,
                "status": wo.status.value,
                "quote_cycle_id": wo.quote_cycle_id,
                "transport_intent_seq": wo.transport_intent_seq,
                "cancel_transport_seq": wo.cancel_transport_seq,
            }

        return {
            "symbol": sym,
            "dup_buy": bool(dup_buy),
            "dup_sell": bool(dup_sell),
            "local_working_bid": _local(wb),
            "local_working_ask": _local(wa),
            "exchange_open_orders": [
                {
                    "oid": int(o.oid),
                    "side": o.side.value,
                    "price": float(o.limit_px),
                    "size": float(o.sz),
                    "cloid": o.cloid,
                }
                for o in sym_orders
            ],
            "in_flight_transport": {
                "bid": bool(wb and wb.status in (OrderStatus.SENT, OrderStatus.CANCEL_PENDING)),
                "ask": bool(wa and wa.status in (OrderStatus.SENT, OrderStatus.CANCEL_PENDING)),
            },
            "active_unresolved_sides": [s.value for s in self._active_unresolved_sides()],
            "unresolved_reasons": {
                s.value: self._side_unresolved_reason.get(s)
                for s in (Side.BUY, Side.SELL)
                if self._is_side_unresolved(s)
            },
        }

    def _sync_open_orders_impl(self) -> str:
        addr = self._settings.hl_account_address
        self._open_orders_rest_calls_total += 1
        # 2026-05-14 (orphan-fill race fix): capture the wall-clock
        # instant we DISPATCH the REST request. Any local working
        # order whose ``ts_ack`` (or ``ts_sent`` for the SENT path)
        # post-dates this snapshot CANNOT possibly be visible in the
        # response — OKX hadn't seen our ack yet when the snapshot
        # was prepared. ``_reconcile_side`` uses this to suppress
        # ``gone_on_exchange`` for freshly-acked orders, which is
        # the bug behind BUG-XXX (3-orders-open / orphan-fill).
        rest_request_dispatched_at = self._clock.now_utc()
        try:
            raw = self._client.fetch_open_orders_raw(addr)
        except Exception as e:
            if self._is_rate_limited_exception(e):
                self._open_orders_rate_limited_total += 1
                logger.warning("sync_open_orders rate_limited")
                self._state.bump_execution_errors("sync_open_orders_rate_limited")
                return "rate_limited"
            logger.exception("sync_open_orders failed")
            self._state.bump_execution_errors("sync_open_orders_exception")
            return "error"
        # v1.4.55 wedge-elimination Phase 1: removed the
        # ``_cached_open_orders_raw`` hydration here. The cache was
        # built (v1.4.43) for the OLD ``cancel_all_orders_for_symbol``
        # path that iterated the exchange snapshot. Phase 1 rewrote
        # cancel-all to use local WO state + dispatcher; the snapshot
        # cache has no consumers.
        sym = self._settings.symbol
        sym_orders = [o for o in raw if o.coin == sym]
        buy_remote = [o for o in sym_orders if o.side == Side.BUY]
        sell_remote = [o for o in sym_orders if o.side == Side.SELL]

        # v1.4.71 wedge-elimination-cleanup Phase 1D — slot-aware
        # duplicate detection.
        #
        # Pre-Phase-1D: dup_buy = len(buy_remote) > 1. With
        # ``LADDER_NUM_LEVELS_PER_SIDE > 1``, every legitimate outer
        # rung was misclassified as a duplicate, triggering
        # ``desync_detected`` + ``set_side_unresolved`` + extra-cancel
        # storms (Codex Finding 1, Bug 1). Snapshot
        # v1.4.66-260518-192744 caught 8 orphan_remote_cancel_dispatched
        # events from this in a 75-second window;
        # v1.4.70-260518-213819 still shows 9 desync_detected events
        # in 12 minutes with the SAME OIDs merging every 3s for 47s.
        #
        # Phase 1D rule:
        #   * Map each remote to a local slot via OID then cloid.
        #   * Slot conflict (2+ remotes mapped to same slot) → real
        #     duplicate; cancel extras.
        #   * Unconfigured slot (level_idx >= configured) → orphan;
        #     cancel.
        #   * Unbound remote (no local match) AND open configured
        #     slot → assign best-fit; not a duplicate.
        #   * Unbound remote AND no open slot → orphan; cancel.
        #
        # When ``LADDER_NUM_LEVELS_PER_SIDE == 1`` the path reduces
        # to the pre-Phase-1D side-keyed behaviour.
        configured_rungs = max(
            1, int(getattr(self._settings, "ladder_num_levels_per_side", 1) or 1)
        )

        def _classify_per_slot(
            remotes: list,
            side: Side,
            configured: int,
        ) -> tuple[
            dict[int, "OpenOrderRaw"],
            list[tuple["OpenOrderRaw", str]],
            bool,
        ]:
            """Classify remote orders for one side.

            Returns ``(keepers, orphans_with_reason, is_duplicate)``.

            * ``keepers`` — dict keyed by ``level_idx``, exactly one
              remote per slot.
            * ``orphans_with_reason`` — list of ``(OpenOrderRaw, reason)``
              tuples. ``reason`` is either ``"duplicate_same_side_extra"``
              (legacy reason for slot collisions or unbound overflow —
              the pre-Phase-1D "too many same-side orders" case) or
              ``"orphan_unmatched_or_beyond_configured_rung"`` (a
              remote bound to a slot beyond the configured ladder,
              typically left over from a config drop).
            * ``is_duplicate`` — True when this side has EITHER a true
              slot collision OR more unbound remotes than open slots.
              Either case is what the pre-Phase-1D code flagged as
              ``dup_buy=True``/``dup_sell=True``; preserving the
              dispatch into ``_set_side_unresolved`` and the desync
              state-machine.

            Newest-timestamp wins in both slot collisions and unbound
            assignment — matches the pre-Phase-1D ``_pick_keeper``
            heuristic (``max(o.timestamp)``).
            """
            if not remotes:
                return {}, [], False
            # Local lookup by OID then cloid (per side; slot index
            # is per-side).
            local_by_oid: dict[int, int] = {}
            local_by_cloid: dict[str, int] = {}
            with self._state._lock:
                for idx, wo in self._state.iter_working_orders(side):
                    if wo is None:
                        continue
                    if wo.order_id_exchange:
                        local_by_oid[int(wo.order_id_exchange)] = int(idx)
                    if wo.client_order_id:
                        local_by_cloid[wo.client_order_id] = int(idx)
            by_slot: dict[int, list] = {}
            unbound: list = []
            for r in remotes:
                slot_idx: Optional[int] = None
                if r.oid is not None and int(r.oid) in local_by_oid:
                    slot_idx = local_by_oid[int(r.oid)]
                elif r.cloid and r.cloid in local_by_cloid:
                    slot_idx = local_by_cloid[r.cloid]
                if slot_idx is not None:
                    by_slot.setdefault(slot_idx, []).append(r)
                else:
                    unbound.append(r)

            keepers: dict[int, "OpenOrderRaw"] = {}
            orphans: list[tuple["OpenOrderRaw", str]] = []
            slot_conflict = False
            # Stable newest-first sort key (ts desc, then oid desc).
            def _newest_first(seq):
                return sorted(
                    seq,
                    key=lambda o: (int(o.timestamp or 0), int(o.oid or 0)),
                    reverse=True,
                )

            for idx, cands in by_slot.items():
                if idx >= configured:
                    # Beyond-rung: orphan the lot. Distinct reason
                    # from "too many same-side"; this can happen when
                    # the operator reduces ``LADDER_NUM_LEVELS_PER_SIDE``
                    # mid-session.
                    for c in cands:
                        orphans.append((c, "orphan_unmatched_or_beyond_configured_rung"))
                    continue
                if len(cands) == 1:
                    keepers[idx] = cands[0]
                else:
                    # 2+ remotes mapped to the same configured slot —
                    # the legacy v1.4.66 duplicate case (slot-keyed).
                    slot_conflict = True
                    cands_sorted = _newest_first(cands)
                    keepers[idx] = cands_sorted[0]
                    for c in cands_sorted[1:]:
                        orphans.append((c, "duplicate_same_side_extra"))

            # Unbound assignment: NEWEST first (legacy ``_pick_keeper``
            # heuristic — "last placed wins"). The user-visible
            # invariant is preserved: when 2+ remotes have no cloid
            # match, the newest one is the keeper.
            open_slots = sorted(i for i in range(configured) if i not in keepers)
            unbound_sorted = _newest_first(unbound)
            unbound_overflow = False
            for r in unbound_sorted:
                if open_slots:
                    keepers[open_slots.pop(0)] = r
                else:
                    # No open slot for this unbound remote — legacy
                    # "duplicate" case (2+ same-side without local
                    # tracking). Reason preserved for the existing
                    # postmortem / metric vocabulary.
                    unbound_overflow = True
                    orphans.append((r, "duplicate_same_side_extra"))

            is_duplicate = bool(slot_conflict or unbound_overflow)
            return keepers, orphans, is_duplicate

        buy_keepers, buy_orphans, dup_buy = _classify_per_slot(
            buy_remote, Side.BUY, configured_rungs
        )
        sell_keepers, sell_orphans, dup_sell = _classify_per_slot(
            sell_remote, Side.SELL, configured_rungs
        )

        # --- Residual-order audit ------------------------------------------------
        # Belt-and-suspenders visibility for the cancel-confirmation gating
        # (see BLUEFIN_CANCEL_CONFIRM_*). Compare our local working-order
        # view against what the exchange actually holds and emit a one-line
        # summary each reconcile cycle. Operators can spot accumulating
        # drift from /state/current + /events/recent without having to
        # cross-reference DB and logs manually.
        #
        # The existing duplicate-detection + _cancel_orphan_remote_order
        # pipeline below ACTS on discrepancies. This block just LOGS them.
        with self._state._lock:
            # v1.4.194: migrated off the deprecated property shims.
            local_bid = self._state.get_working_order(Side.BUY, 0)
            local_ask = self._state.get_working_order(Side.SELL, 0)
        local_bid_live = local_bid is not None and local_bid.status in (
            OrderStatus.ACKED,
            OrderStatus.SENT,
        )
        local_ask_live = local_ask is not None and local_ask.status in (
            OrderStatus.ACKED,
            OrderStatus.SENT,
        )
        local_buy_count = 1 if local_bid_live else 0
        local_sell_count = 1 if local_ask_live else 0
        # Buy/sell deltas: positive = more on server than locally known
        # (orphan residuals); negative = fewer on server than locally
        # known (a local working order that the server lost track of —
        # rare, usually means fill or cancel we missed via WS).
        buy_delta = len(buy_remote) - local_buy_count
        sell_delta = len(sell_remote) - local_sell_count
        # Only log when there's real news (non-zero delta OR duplicates).
        # Zero-delta zero-dup is the boring happy path and we don't want
        # to spam the reconcile log every cooldown tick.
        if buy_delta != 0 or sell_delta != 0 or dup_buy or dup_sell:
            log_extra(
                logger,
                logging.INFO,
                "residual_order_audit",
                {
                    "event": "residual_order_audit",
                    "symbol": sym,
                    "server_buy": len(buy_remote),
                    "server_sell": len(sell_remote),
                    "local_buy": local_buy_count,
                    "local_sell": local_sell_count,
                    "buy_delta": buy_delta,
                    "sell_delta": sell_delta,
                    "dup_buy": dup_buy,
                    "dup_sell": dup_sell,
                },
            )
        # ------------------------------------------------------------------------

        # v1.4.71 Phase 1D — inside-rung keeper for legacy
        # ``_reconcile_side`` path. Outer-rung keepers are addressed
        # via the existing hydration merge path (Phase 1B) since
        # ``_reconcile_side`` is inside-rung-only by design.
        keeper_buy = buy_keepers.get(0)
        keeper_sell = sell_keepers.get(0)
        by_side: dict[Side, OpenOrderRaw] = {}
        if keeper_buy is not None:
            by_side[Side.BUY] = keeper_buy
        if keeper_sell is not None:
            by_side[Side.SELL] = keeper_sell

        # Cancel orphans (per-Phase-1D classification). Each orphan
        # carries its own reason from the classifier:
        #   * ``duplicate_same_side_extra`` — legacy reason for slot
        #     collisions or unbound overflow (preserves the
        #     pre-Phase-1D postmortem vocabulary and dispatch
        #     behaviour).
        #   * ``orphan_unmatched_or_beyond_configured_rung`` — a
        #     remote bound to a slot beyond the configured ladder.
        all_extra_cancels_ok: dict[Side, bool] = {Side.BUY: True, Side.SELL: True}
        for side, side_orphans in (
            (Side.BUY, buy_orphans),
            (Side.SELL, sell_orphans),
        ):
            for extra, reason in side_orphans:
                ok = self._cancel_orphan_remote_order(
                    symbol=sym,
                    oid=int(extra.oid),
                    cloid=extra.cloid,
                    reason=reason,
                    context={
                        "side": side.value,
                        "configured_rungs": int(configured_rungs),
                        "side_remote_count": len(
                            buy_remote if side == Side.BUY else sell_remote
                        ),
                        "keeper_oid_inside": (
                            int(keeper_buy.oid)
                            if side == Side.BUY and keeper_buy is not None
                            else (
                                int(keeper_sell.oid)
                                if side == Side.SELL and keeper_sell is not None
                                else None
                            )
                        ),
                    },
                )
                if not ok:
                    all_extra_cancels_ok[side] = False

        d_buy = dup_buy
        d_sell = dup_sell
        if dup_buy or dup_sell:
            self._duplicate_open_detect_count += 1
            if dup_buy:
                self._set_side_unresolved(
                    Side.BUY,
                    reason="duplicate_open_orders_detected",
                    payload={"symbol": sym},
                    requires_confirm=(not all_extra_cancels_ok[Side.BUY]),
                )
            if dup_sell:
                self._set_side_unresolved(
                    Side.SELL,
                    reason="duplicate_open_orders_detected",
                    payload={"symbol": sym},
                    requires_confirm=(not all_extra_cancels_ok[Side.SELL]),
                )
            if not self._dup_open_orders_diag_latched:
                self._dup_open_orders_diag_latched = True
                payload = self._duplicate_open_orders_diag_payload(
                    sym=sym,
                    sym_orders=sym_orders,
                    dup_buy=dup_buy,
                    dup_sell=dup_sell,
                )
                payload["fresh_place_suppressed"] = {
                    "buy": bool(dup_buy),
                    "sell": bool(dup_sell),
                }
                log_extra(
                    logger,
                    logging.WARNING,
                    "multiple_open_orders_same_side",
                    {"event": "multiple_open_orders_same_side", **payload},
                )
            else:
                logger.debug(
                    "multiple_open_orders_same_side symbol=%s dup_buy=%s dup_sell=%s (latched)",
                    sym,
                    dup_buy,
                    dup_sell,
                )
        else:
            self._dup_open_orders_diag_latched = False
        d_buy = d_buy or self._reconcile_side(
            Side.BUY,
            by_side.get(Side.BUY),
            rest_request_dispatched_at=rest_request_dispatched_at,
        )
        d_sell = d_sell or self._reconcile_side(
            Side.SELL,
            by_side.get(Side.SELL),
            rest_request_dispatched_at=rest_request_dispatched_at,
        )
        new_desync = d_buy or d_sell
        kill_n = self._settings.desync_unrecoverable_after_ticks
        q_ticks = self._settings.desync_quarantine_ticks
        events: list[tuple[str, str, dict]] = []

        with self._state._lock:
            st = self._state
            was_desync = st.order_desync
            st.order_desync = new_desync
            # Per-side desync detail (Item 4 of the Market tab surface).
            # ``d_buy`` and ``d_sell`` are computed separately above
            # but historically OR'd into the single ``order_desync``
            # boolean; surface both for the dashboard so the operator
            # can tell which side is stuck without reading event logs.
            st.order_desync_buy = bool(d_buy)
            st.order_desync_sell = bool(d_sell)

            if new_desync:
                if not was_desync:
                    st.desync_consecutive_ticks = 1
                    st.desync_phase = DesyncPhase.DETECTED
                    events.append(
                        (
                            "desync_detected",
                            f"order desync detected symbol={sym}",
                            {"symbol": sym, "dup_buy": dup_buy, "dup_sell": dup_sell},
                        )
                    )
                else:
                    st.desync_consecutive_ticks += 1
                    if st.desync_phase == DesyncPhase.DETECTED:
                        st.desync_phase = DesyncPhase.RECONCILING
                        events.append(
                            (
                                "desync_reconciling",
                                f"order desync reconciling symbol={sym}",
                                {"symbol": sym, "dup_buy": dup_buy, "dup_sell": dup_sell},
                            )
                        )
                    elif st.desync_phase not in (
                        DesyncPhase.UNRECOVERABLE,
                        DesyncPhase.RECONCILING,
                    ):
                        st.desync_phase = DesyncPhase.RECONCILING
                if (
                    st.desync_consecutive_ticks >= kill_n
                    and st.desync_phase != DesyncPhase.UNRECOVERABLE
                ):
                    st.desync_phase = DesyncPhase.UNRECOVERABLE
                    events.append(
                        (
                            "desync_unrecoverable",
                            f"order desync unrecoverable symbol={sym} after {kill_n} ticks",
                            {"symbol": sym, "ticks": st.desync_consecutive_ticks},
                        )
                    )
            else:
                st.desync_consecutive_ticks = 0
                if was_desync:
                    st.desync_phase = DesyncPhase.RECOVERED
                    st.desync_quarantine_remaining = max(1, q_ticks)
                    events.append(
                        (
                            "desync_recovered",
                            f"order desync cleared symbol={sym}",
                            {"symbol": sym},
                        )
                    )
                elif st.desync_quarantine_remaining > 0:
                    st.desync_quarantine_remaining -= 1
                    if st.desync_quarantine_remaining == 0:
                        st.desync_phase = DesyncPhase.OK

        for ev_type, msg, payload in events:
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.WARNING.value
                if ev_type
                in ("desync_detected", "desync_reconciling", "desync_unrecoverable")
                else EventSeverity.INFO.value,
                ev_type,
                msg,
                payload,
            )
        return "ok"

    def _hydrate_working_from_exchange(self, side: Side, remote: OpenOrderRaw) -> bool:
        """Adopt a single resting order from the exchange when local has no working order (restart / MM continuity).

        Returns True when the hydration was performed, False when it
        was skipped because the bot already saw this (oid, cloid)
        terminate recently — i.e. the REST snapshot is just lagging
        behind WS, not a genuine orphan.
        """
        # 1.3.85 hydration race guard. If the bot's order_trace shows
        # a recent terminal (FILLED / CANCELED / REJECTED, either from
        # local state machine or from a WS event), this REST snapshot
        # is stale — the order is dead, REST hasn't caught up.
        # Skipping the hydration avoids the WS-vs-REST race signature
        # (hydrate-as-ACKED → next-reconcile-sees-gone → declare
        # gone_on_exchange) that produced the 6 hydration-ghost
        # events in snapshot 260515-095056.
        try:
            window_s = float(
                getattr(
                    self._settings,
                    "reconcile_hydration_recent_terminal_window_seconds",
                    60.0,
                )
            )
        except (TypeError, ValueError):
            window_s = 60.0
        if window_s > 0 and self._state.order_trace.was_recently_terminal(
            order_id_exchange=remote.oid,
            client_order_id=remote.cloid,
            max_age_seconds=window_s,
        ):
            self._state.hydration_skipped_recently_terminal_total += 1
            logger.info(
                "reconcile_skip_hydration_recently_terminal side=%s oid=%s cloid=%s window_s=%.0f",
                side.value,
                remote.oid,
                (remote.cloid[:18] + "...") if remote.cloid else None,
                window_s,
            )
            return False

        # v1.4.69 wedge-elimination-cleanup Phase 1B: dedup by
        # ``(oid, cloid)`` against active local WOs.
        #
        # Snapshot v1.4.66-260518-192744 (and v1.4.67-260518-195033)
        # caught the same OID appearing as TWO WorkingOrder records
        # in ``orders_lifecycle-since.json``: one CANCEL_PENDING
        # never reaped, one hydrated CANCELED. That state is the
        # foundation of the duplicate-WO wedge mechanism.
        #
        # The dedup logic:
        #
        #   1. Scan ``state.all_working_orders()`` for any non-
        #      terminal WO matching ``remote.oid`` (preferred) or
        #      ``remote.cloid``. If found → MERGE: update the
        #      existing WO's status from hydrated state. Don't
        #      create a duplicate. Return True (treated as a
        #      successful hydration for the caller's purposes).
        #
        #   2. If a terminal WO matches → log INFO and SKIP. The
        #      order is conceptually done locally; resurrecting it
        #      would re-introduce the bug.
        #
        # In Phase 3A this becomes ``state.order_store.find_by_oid_or_cloid()``.
        # For now we do the O(n) scan inline; n is small (≤ 4 typically).
        existing = self._find_local_wo_by_oid_or_cloid(
            remote.oid, remote.cloid
        )
        if existing is not None:
            if existing.status in (
                OrderStatus.CANCELED,
                OrderStatus.FILLED,
                OrderStatus.REJECTED,
                OrderStatus.DESYNC,
            ):
                self._hydration_skipped_terminal_match_total += 1
                logger.info(
                    "hydration_skipped_terminal_match side=%s oid=%s cloid=%s "
                    "existing_status=%s — bot already done with this order",
                    side.value,
                    remote.oid,
                    (remote.cloid[:18] + "...") if remote.cloid else None,
                    existing.status.value,
                )
                return False
            # Non-terminal match: merge. Adopt the hydrated state.
            # We trust the venue's view of the order over local
            # stale guess.
            with self._state._lock:
                # Bind OID if it was missing.
                if not existing.order_id_exchange and remote.oid:
                    existing.order_id_exchange = remote.oid
                # If we were stuck in SENT and the venue says it's
                # live, transition to ACKED.
                if existing.status == OrderStatus.SENT:
                    transition(existing, OrderStatus.ACKED)
                # Adopt the venue's price/size view.
                if remote.limit_px and float(remote.limit_px) > 0:
                    existing.price = float(remote.limit_px)
                if remote.sz and float(remote.sz) > 0:
                    existing.size = float(remote.sz)
                # Mark as hydrated so downstream heuristics (fast-cancel
                # post-only detector, aging exemptions) treat it as
                # such.
                existing.hydrated_from_exchange = True
                self.persist(existing)
            self._hydration_merged_existing_total += 1
            logger.info(
                "hydration_merged_existing side=%s oid=%s cloid=%s "
                "existing_local_id=%s existing_status_before=%s",
                side.value,
                remote.oid,
                (remote.cloid[:18] + "...") if remote.cloid else None,
                existing.order_id_local,
                # Re-read after transition; for diagnostics record
                # the post-merge status.
                existing.status.value,
            )
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.INFO.value,
                "order_hydrated_merged",
                f"merged {side.value} oid={remote.oid} into existing local WO",
                {
                    "symbol": self._settings.symbol,
                    "side": side.value,
                    "exchange_oid": remote.oid,
                    "existing_local_id": existing.order_id_local,
                    "post_merge_status": existing.status.value,
                },
            )
            return True

        sym = self._settings.symbol
        now = self._clock.now_utc()
        # v1.4.34 (Codex #1 fix): anchor ``ts_ack`` to the venue's
        # real creation / last-update timestamp when available.
        # ``HLOpenOrderRaw.timestamp`` is the ms-epoch from the
        # adapter (OKX: ``uTime`` or ``cTime``; HL: native ack time).
        # The previous behaviour set ``ts_ack=now``, which made age-
        # based aging restart the clock on every process restart and
        # caused the fast-cancel heuristic to false-fire when a long-
        # resting order got cancelled within ~100 ms of hydrate.
        # Fall back to ``now`` only when the adapter didn't supply a
        # usable timestamp (``timestamp <= 0``); the
        # ``hydrated_from_exchange`` flag below ensures downstream
        # callers still exempt the order from heuristics that assume
        # a real local ack.
        venue_ts_ms = int(getattr(remote, "timestamp", 0) or 0)
        if venue_ts_ms > 0:
            try:
                venue_ack = datetime.fromtimestamp(
                    venue_ts_ms / 1000.0, tz=timezone.utc
                )
            except (TypeError, ValueError, OSError, OverflowError):
                venue_ack = now
        else:
            venue_ack = now
        wo = WorkingOrder(
            order_id_local=str(uuid.uuid4()),
            order_id_exchange=remote.oid,
            client_order_id=remote.cloid,
            symbol=sym,
            side=side,
            price=float(remote.limit_px),
            size=float(remote.sz),
            post_only=True,
            status=OrderStatus.ACKED,
            ts_sent=venue_ack,
            ts_ack=venue_ack,
            quote_cycle_id=None,
            hydrated_from_exchange=True,
        )
        self.persist(wo)
        # v1.4.194: ``state.set_working_order(side, 0, wo)`` replaces
        # the deprecated ``working_bid`` / ``working_ask`` property
        # setters. Inside-rung is ``level_idx=0``; behaviour identical.
        self._state.set_working_order(side, 0, wo)
        logger.info(
            "resting_order_hydrated_from_exchange side=%s oid=%s px=%s sz=%s "
            "venue_ts_ms=%d ts_ack_age_s=%.1f",
            side.value,
            remote.oid,
            remote.limit_px,
            remote.sz,
            venue_ts_ms,
            (now - venue_ack).total_seconds(),
        )
        self._storage.insert_bot_event(
            self._clock.now_utc().isoformat(),
            EventSeverity.INFO.value,
            "order_hydrated_from_exchange",
            f"hydrated {side.value} oid={remote.oid}",
            {
                "symbol": sym,
                "side": side.value,
                "exchange_oid": remote.oid,
                "price": remote.limit_px,
                "size": remote.sz,
            },
        )
        return True

    def _cancel_orphan_remote_order(
        self,
        *,
        symbol: str,
        oid: Optional[int],
        cloid: Optional[str],
        reason: str,
        context: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Dispatch an HTTP cancel for a remote exchange order that has NO owning local slot.

        Used by reconcile/desync paths: clearing a local working slot without canceling the
        remote leaves a phantom order alive on the exchange. This helper issues the cancel
        synchronously (same transport as ``cancel_all_orders``), persists a telemetry event,
        and returns True on success / benign_missing. Failures are logged and emit an
        ``orphan_cancel_failed`` event so operators can see unresolved phantoms.

        The outbound dispatcher is intentionally NOT used: the orphan has no local
        ``WorkingOrder`` in the side slot, and the dispatcher's callbacks rely on
        ``_working_order_by_local_id`` for lookup.
        """
        # Treat ``oid == 0`` the same as "no oid" — GRVT order ids are never zero,
        # so an incoming 0 is a placeholder (or leftover from a bogus ack) and
        # sending it as ``{"order_id": "0"}`` makes GRVT reject the cancel.
        if not oid:
            oid = None
        if oid is None and not cloid:
            logger.warning(
                "orphan_cancel_no_identifier side_reason=%s symbol=%s",
                reason,
                symbol,
            )
            return False
        if not self._client.has_write_access():
            # Read-only clients (simulation / dry-run): record, do not attempt exchange call.
            return False
        # Cancel-storm protection: if we successfully dispatched a cancel for this (symbol, oid)
        # within the last ``_ORPHAN_CANCEL_DEDUP_TTL_SECONDS``, skip the HTTP call. The exchange
        # may still be processing the first cancel; re-sending would add load without changing
        # the outcome. Only ``oid`` is used as the cache key (cloid-only cancels, rare on this
        # path, bypass the cache and hit the exchange each time).
        now_mono = self._clock.monotonic()
        dedup_key: Optional[tuple[str, int]] = (
            (symbol, int(oid)) if oid is not None else None
        )
        if dedup_key is not None:
            last_mono = self._recent_orphan_cancels.get(dedup_key)
            if last_mono is not None and (now_mono - last_mono) < _ORPHAN_CANCEL_DEDUP_TTL_SECONDS:
                self._orphan_cancel_dedup_skips += 1
                log_extra(
                    logger,
                    logging.DEBUG,
                    "orphan_cancel_deduped",
                    {
                        "reason": reason,
                        "symbol": symbol,
                        "remote_oid": int(oid) if oid is not None else None,
                        "since_last_dispatch_s": round(now_mono - last_mono, 3),
                    },
                )
                return True
        try:
            if oid is not None:
                resp = self._client.cancel_order(symbol, int(oid))
            else:
                resp = self._client.cancel_order_by_cloid(symbol, str(cloid))
        except Exception:
            logger.exception(
                "orphan_cancel_transport_failed reason=%s symbol=%s oid=%s cloid=%s",
                reason,
                symbol,
                oid,
                (cloid[:18] + "...") if isinstance(cloid, str) and cloid else cloid,
            )
            self._state.bump_execution_errors("orphan_cancel_transport_exception")
            payload: dict[str, Any] = {
                "reason": reason,
                "symbol": symbol,
                "remote_oid": int(oid) if oid is not None else None,
                "had_cloid": bool(cloid),
                "outcome": "transport_exception",
            }
            if context:
                payload.update(context)
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.ERROR.value,
                "orphan_cancel_failed",
                f"phantom cancel transport error reason={reason} symbol={symbol}",
                payload,
            )
            return False
        kind, detail = self._interpret_cancel_response(resp)
        # v1.4.38 (2026-05-18) classifier-completeness fix: include
        # ``unexpected_gone`` in the OK set on the ORPHAN-CANCEL path.
        # Reasoning: for orphan cancels, "the order is gone" IS the
        # desired outcome — we wanted it gone, and it's gone. The
        # v1.3.120 reclassification moved 51400 / 51401 / 51503 from
        # ``benign_missing`` to ``unexpected_gone`` to make the
        # ANOMALY visible on the normal-cancel path (operator should
        # investigate why a normal cancel produced "order doesn't
        # exist"). But the orphan-cancel path's whole purpose is
        # reconciling against an order that probably IS gone; treating
        # ``unexpected_gone`` as a confirmation rather than an error is
        # the correct semantics. Pre-v1.4.38 this path bumped
        # ``orphan_cancel_exchange_reject`` on every 51400 row — which
        # under v1.4.34's hydrated_from_exchange flag (removed the
        # accidental fast-cancel cooldown rate-limiter on the orphan
        # loop) drove the v1.4.37 production self-kill at 56 errors in
        # 5 minutes vs threshold 20. The ``unexpected_gone`` counter
        # is still incremented on ``BotState`` for visibility; the
        # operator can still spot anomalies via that counter without
        # the bot self-killing.
        ok = kind in ("success", "benign_missing", "unexpected_gone")
        payload = {
            "reason": reason,
            "symbol": symbol,
            "remote_oid": int(oid) if oid is not None else None,
            "had_cloid": bool(cloid),
            "outcome": kind,
            "detail": (detail[:300] if detail else ""),
        }
        if context:
            payload.update(context)
        if kind == "unexpected_gone":
            # Bump the same counter the single-cancel HTTP path uses
            # so the operator can see how often orphan cancels are
            # racing the venue's silent cleanup. Distinct from
            # ``cancel_race_lost_to_fill_total`` (51402 = filled),
            # which counts the "we lost the cancel/fill race" case.
            self._state.cancel_unexpected_gone_total += 1
        if ok:
            self._bump_transport_counters()
            # Record in dedup cache so repeated reconciles within the settlement window
            # don't respam the exchange. Bounded-size to stay memory-safe under unexpected
            # spew (e.g., a misbehaving reconcile path). On overflow drop the oldest entry.
            if dedup_key is not None:
                if len(self._recent_orphan_cancels) >= _ORPHAN_CANCEL_DEDUP_MAX_ENTRIES:
                    # Evict the single oldest entry — O(n) but n is bounded by the cap.
                    try:
                        oldest = min(
                            self._recent_orphan_cancels.items(), key=lambda kv: kv[1]
                        )[0]
                        self._recent_orphan_cancels.pop(oldest, None)
                    except ValueError:
                        pass
                self._recent_orphan_cancels[dedup_key] = now_mono
            log_extra(
                logger,
                logging.WARNING if kind == "success" else logging.INFO,
                "orphan_remote_cancel_dispatched",
                payload,
            )
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.WARNING.value,
                "orphan_remote_cancel_dispatched",
                f"phantom cancel dispatched reason={reason} symbol={symbol} oid={oid}",
                payload,
            )
            return True
        logger.warning(
            "orphan_cancel_exchange_reject reason=%s oid=%s kind=%s detail=%s",
            reason,
            oid,
            kind,
            (detail[:300] if detail else ""),
        )
        self._state.bump_execution_errors("orphan_cancel_exchange_reject")
        self._storage.insert_bot_event(
            self._clock.now_utc().isoformat(),
            EventSeverity.ERROR.value,
            "orphan_cancel_failed",
            f"phantom cancel rejected reason={reason} symbol={symbol}",
            payload,
        )
        return False

    def _reconcile_side(
        self,
        side: Side,
        remote: Optional[OpenOrderRaw],
        *,
        rest_request_dispatched_at: Optional[datetime] = None,
    ) -> bool:
        """Returns True if local state disagrees with the exchange for this side.

        Invariant enforced by this path: when a divergence is detected and the local slot
        is cleared, the previously-observed remote exchange order MUST have a cancel
        dispatched. Otherwise we leave a phantom order alive and every subsequent
        placement on that side will race a duplicate.

        ``rest_request_dispatched_at`` is the UTC wall-clock instant at which the open-
        orders REST request was sent. Any local order whose ``ts_ack`` /
        ``ts_sent`` post-dates this instant cannot possibly appear in the REST
        response and must NOT be marked ``gone_on_exchange``. This is the 2026-05-14
        orphan-fill fix.
        """
        # v1.4.194: migrated off the deprecated property shims.
        wo = self._state.get_working_order(side, 0)
        addr = self._settings.hl_account_address
        sym = self._settings.symbol

        if remote is None:
            if wo and wo.status == OrderStatus.SENT and wo.client_order_id:
                self._try_resolve_sent_order_by_cloid(wo)
                return False
            if wo and wo.status in (
                OrderStatus.ACKED,
                OrderStatus.PARTIAL,
                OrderStatus.CANCEL_PENDING,
                # amend-prio Phase 4 wedge fix (v1.4.24): AMEND_PENDING
                # MUST appear in this list. Otherwise an amend that
                # returned ``unconfirmed`` (non-strict mode) or whose
                # response was lost entirely leaves the WO stuck in
                # AMEND_PENDING with no path to terminal. ``_orchestrate``
                # early-returns on AMEND_PENDING → the slot is never
                # released → the bot wedges silently.
                #
                # Observed on the v1.4.23 deploy (snapshot
                # v1.4.23-260517-194714): after a BUY fill at t=33s
                # of the session, the bot's SELL side stuck in
                # AMEND_PENDING and no new venue ops fired for the
                # remaining 5 minutes. ``open_orders`` count at venue
                # = 0; reconcile saw the empty REST snapshot but
                # silently passed (this fall-through). Bot wedged.
                #
                # The ``rest_request_dispatched_at`` race guard below
                # uses ``ts_ack`` as the anchor; AMEND_PENDING WOs are
                # by definition already ACKED so ``ts_ack`` is set.
                # An in-flight amend's ``ts_amend_sent`` may post-date
                # the REST request — the guard correctly skips those.
                OrderStatus.AMEND_PENDING,
            ):
                # 2026-05-14 orphan-fill race fix: the REST snapshot
                # was prepared at a wall-clock instant that we can
                # bound below by ``rest_request_dispatched_at``. Any
                # local order whose ack post-dates that bound CANNOT
                # have been visible to OKX when the response was built
                # — declaring it ``gone_on_exchange`` here would orphan
                # the order (we mark it canceled locally without
                # sending a cancel, then it can silently fill on the
                # venue). Skip the decision; the next reconcile cycle
                # will see the order in REST and behave normally.
                #
                # v1.4.24 extension: for AMEND_PENDING WOs, ALSO skip
                # when an amend was issued AFTER the REST snapshot was
                # taken — the amend response is still in flight and
                # might transition the WO back to ACKED imminently.
                # Treating it as gone here would race the response.
                if (
                    rest_request_dispatched_at is not None
                    and wo.status == OrderStatus.AMEND_PENDING
                    and wo.ts_amend_sent is not None
                    and _utc_aware(wo.ts_amend_sent) >= rest_request_dispatched_at
                ):
                    self._state.reconcile_skip_snapshot_stale_total += 1
                    logger.warning(
                        "reconcile_skip_snapshot_stale_amend_pending "
                        "side=%s oid=%s ts_amend_sent=%s "
                        "rest_dispatched_at=%s",
                        side.value,
                        wo.order_id_exchange,
                        wo.ts_amend_sent.isoformat(),
                        rest_request_dispatched_at.isoformat(),
                    )
                    return False
                if (
                    rest_request_dispatched_at is not None
                    and wo.ts_ack is not None
                    and _utc_aware(wo.ts_ack) >= rest_request_dispatched_at
                ):
                    self._state.reconcile_skip_snapshot_stale_total += 1
                    logger.warning(
                        "reconcile_skip_snapshot_stale side=%s oid=%s "
                        "ts_ack=%s rest_dispatched_at=%s status=%s",
                        side.value,
                        wo.order_id_exchange,
                        wo.ts_ack.isoformat(),
                        rest_request_dispatched_at.isoformat(),
                        wo.status.value,
                    )
                    return False
                # 1.3.82 extension: also skip the gone_on_exchange
                # decision when the order entered CANCEL_PENDING via
                # the cloid path (ts_ack is still None but a cancel
                # is in flight) and the cancel was issued AFTER the
                # REST snapshot was taken. The REST snapshot cannot
                # know about an order whose cancel is still travelling.
                # Without this guard, the place-then-immediate-cancel
                # race produces a gone_on_exchange terminal even when
                # the cancel was a legitimate operator-decision
                # cancel. Targets the 50 phantom-place events from
                # snapshot 260515-095056.
                if (
                    rest_request_dispatched_at is not None
                    and wo.ts_ack is None
                    and wo.ts_cancel_requested is not None
                    and _utc_aware(wo.ts_cancel_requested) >= rest_request_dispatched_at
                ):
                    self._state.reconcile_skip_snapshot_stale_total += 1
                    logger.warning(
                        "reconcile_skip_snapshot_stale_cancel_pending "
                        "side=%s oid=%s cloid=%s "
                        "ts_cancel_requested=%s rest_dispatched_at=%s status=%s",
                        side.value,
                        wo.order_id_exchange,
                        (wo.client_order_id[:18] + "...") if wo.client_order_id else None,
                        wo.ts_cancel_requested.isoformat(),
                        rest_request_dispatched_at.isoformat(),
                        wo.status.value,
                    )
                    return False
                # Race fix (1.1.44): the WS thread may have already
                # received a terminal event for this oid that is sitting
                # in the inbound queue, waiting for the next tick's
                # drain to process it. Reconcile fires late in the same
                # tick AFTER drain, but BEFORE the next tick — so events
                # arriving mid-tick are visible in the trace (stamped at
                # parse time) but not yet applied to the working slot.
                # If we see a terminal in the trace, defer the
                # gone_on_exchange decision to next tick. Phase 2 trace
                # 260508095019 showed 49/200 races, all with a single
                # canceled WS event arriving ~10ms before reconcile.
                if self._state.order_trace.has_pending_ws_terminal(
                    wo.order_id_exchange,
                    ack_ts=wo.ts_ack,
                ):
                    logger.info(
                        "reconcile_skip_gone_pending_ws side=%s oid=%s status=%s",
                        side.value,
                        wo.order_id_exchange,
                        wo.status.value,
                    )
                    return False
                transition(wo, OrderStatus.CANCELED, "gone_on_exchange")
                self.persist(wo)
                # v1.4.95: session-cumulative counter + signature
                # breakdown + bounded ring of recent terminals. See
                # ``_bump_gone_on_exchange_counters`` docstring for the
                # contract (zero is the bar; non-zero is a postmortem
                # fatal). Must run AFTER transition() so wo.ts_closed
                # is stamped, enabling lateness computation.
                self._bump_gone_on_exchange_counters(wo)
                # 1.3.130 multi-rung Phase 2: clear the specific per-rung slot.
                self._state.set_working_order(
                    side, int(getattr(wo, "level_idx", 0) or 0), None
                )
                self._clear_side_unresolved(side, reason="reconcile_confirmed_gone")
                # Phase 2 trace: capture WHY each gone_on_exchange fired.
                # If the trace shows ws_events=[] for this entry, the WS
                # genuinely never delivered an update → connectivity bug.
                # If ws_events has rows, the local state machine ignored
                # them → state-machine race. Distinguishes the two.
                try:
                    self._state.order_trace.record_terminal(
                        order_id_exchange=wo.order_id_exchange,
                        client_order_id=wo.client_order_id or "",
                        status=OrderStatus.CANCELED.value,
                        reason="gone_on_exchange",
                        source="reconcile",
                    )
                except Exception:
                    logger.exception("order_trace_terminal_failed")
                return False
            if wo and wo.status == OrderStatus.SENT:
                # 2026-05-14 orphan-fill race fix (SENT branch).
                # Mirror of the ACKED-branch guard above using
                # ``ts_sent`` (the order may not have an ack yet).
                # If we dispatched the place AFTER the REST snapshot,
                # the snapshot cannot contain it — skip and let the
                # next reconcile cycle handle it cleanly.
                if (
                    rest_request_dispatched_at is not None
                    and wo.ts_sent is not None
                    and _utc_aware(wo.ts_sent) >= rest_request_dispatched_at
                ):
                    self._state.reconcile_skip_snapshot_stale_total += 1
                    logger.warning(
                        "reconcile_skip_snapshot_stale_sent side=%s oid=%s "
                        "ts_sent=%s rest_dispatched_at=%s",
                        side.value,
                        wo.order_id_exchange,
                        wo.ts_sent.isoformat(),
                        rest_request_dispatched_at.isoformat(),
                    )
                    return False
                # Same race-fix guard as the ACKED/PARTIAL/CANCEL_PENDING
                # branch above. SENT can race too: the WS may carry a
                # rejection that the queue hasn't drained yet.
                if self._state.order_trace.has_pending_ws_terminal(
                    wo.order_id_exchange,
                    ack_ts=wo.ts_ack,
                ):
                    logger.info(
                        "reconcile_skip_sent_gone_pending_ws side=%s oid=%s",
                        side.value,
                        wo.order_id_exchange,
                    )
                    return False
                transition(wo, OrderStatus.CANCELED, "gone_on_exchange")
                self.persist(wo)
                # v1.4.95: session-cumulative counter + signature ring.
                # SENT-branch terminals are typically BUG-024 phantoms
                # (ts_ack None) but the classifier handles all four
                # signatures uniformly.
                self._bump_gone_on_exchange_counters(wo)
                # 1.3.130 multi-rung Phase 2: clear specific per-rung slot.
                self._state.set_working_order(
                    side, int(getattr(wo, "level_idx", 0) or 0), None
                )
                self._clear_side_unresolved(side, reason="reconcile_sent_gone")
                try:
                    self._state.order_trace.record_terminal(
                        order_id_exchange=wo.order_id_exchange,
                        client_order_id=wo.client_order_id or "",
                        status=OrderStatus.CANCELED.value,
                        reason="gone_on_exchange",
                        source="reconcile",
                    )
                except Exception:
                    logger.exception("order_trace_terminal_failed")
                return False
            self._clear_side_unresolved(side, reason="reconcile_side_empty")
            return False

        if wo is not None and wo.order_id_exchange == remote.oid:
            # amend-prio Phase 4 (v1.4.17): when the WO is AMEND_PENDING,
            # the venue's `remote.sz` may already reflect the new amend
            # target (a size REDUCTION via amend looks identical to a
            # PARTIAL fill to the legacy `remote.sz < wo.size` check). To
            # avoid a spurious AMEND_PENDING → PARTIAL transition that
            # would corrupt the amend lifecycle, just skip all reconcile
            # mutations for this WO. The amend response handler will
            # commit the target (or revert) within ~10 ms of the wire
            # ack; reconcile picks up the new state on the next cycle.
            if wo.status == OrderStatus.AMEND_PENDING:
                return False
            if remote.sz + 1e-12 < wo.size:
                transition(wo, OrderStatus.PARTIAL)
                wo.size = remote.sz
                self.persist(wo)
            if wo.status in (OrderStatus.ACKED, OrderStatus.PARTIAL):
                self._clear_side_unresolved(side, reason="reconcile_oid_match")
            return False

        if wo is not None and wo.status == OrderStatus.SENT and wo.client_order_id:
            match = self._sent_order_matches_remote_open(wo, remote, addr)
            if match is True:
                wo.order_id_exchange = remote.oid
                if remote.sz + 1e-12 < wo.size:
                    transition(wo, OrderStatus.PARTIAL)
                else:
                    transition(wo, OrderStatus.ACKED)
                wo.size = remote.sz
                self.persist(wo)
                self._clear_side_unresolved(side, reason="reconcile_sent_matched_by_cloid")
                return False
            if match is None:
                logger.info(
                    "reconcile_sent_deferred side=%s remote_oid=%s (cloid_orderStatus_uncertain)",
                    side.value,
                    remote.oid,
                )
                return False

        if wo is not None:
            # Recovery before DESYNC: if the remote order carries the same client_order_id
            # as our local working slot, this is the SAME logical order — we are just
            # missing the exchange oid locally (e.g. GRVT's create_order response arrived
            # without ``order_id`` populated, or we cleared it in response to a placeholder
            # 0). Bind the exchange oid into the local wo instead of tearing it down.
            # This avoids a spurious DESYNC + orphan-cancel pair where the "local ghost"
            # cancel would hit ``{"order_id": "0"}`` and fail.
            local_cloid = wo.client_order_id
            if (
                local_cloid
                and remote.cloid
                and cloids_match(remote.cloid, local_cloid)
            ):
                logger.info(
                    "reconcile_bind_by_cloid side=%s local_order_id=%s remote_oid=%s "
                    "prior_local_oid=%s status=%s",
                    side.value,
                    wo.order_id_local,
                    remote.oid,
                    wo.order_id_exchange,
                    wo.status.value,
                )
                wo.order_id_exchange = remote.oid
                # Don't change status here: SENT/ACKED/CANCEL_PENDING all remain valid —
                # this binding just adopts the oid that the exchange picked for this cloid.
                if remote.sz + 1e-12 < wo.size and wo.status in (
                    OrderStatus.ACKED,
                    OrderStatus.CANCEL_PENDING,
                ):
                    transition(wo, OrderStatus.PARTIAL)
                    wo.size = remote.sz
                self.persist(wo)
                self._clear_side_unresolved(side, reason="reconcile_bind_by_cloid")
                return False
            # EXCHANGE MISMATCH: the remote has an order whose oid does not match our local
            # working slot, and we could not bind it via cloid. The REMOTE order is a phantom
            # relative to our local view — if we only clear local state, it will live on the
            # exchange indefinitely and every subsequent same-side placement will race a
            # duplicate (the bug observed in snap_20260416_154159 oid=384103722786).
            #
            # Actively remediate: issue a cancel for the observed remote.oid BEFORE nulling
            # local slot. Also cancel wo.order_id_exchange if set and different (it may or
            # may not still exist on the exchange; a benign_missing is acceptable).
            local_oid = wo.order_id_exchange
            # Treat local_oid == 0 as "no local oid" — see _cancel_orphan_remote_order.
            # Without this, the local_ghost branch below sent ``{"order_id": "0"}`` and
            # GRVT replied "Either order ID or client order ID must be supplied".
            if not local_oid:
                local_oid = None
            mismatch_ctx = {
                "local_order_id": wo.order_id_local,
                "local_oid": int(local_oid) if local_oid is not None else None,
                "local_status": wo.status.value,
                "remote_oid": int(remote.oid),
                "side": side.value,
            }
            cancel_ok_remote = self._cancel_orphan_remote_order(
                symbol=sym,
                oid=int(remote.oid),
                cloid=remote.cloid,
                reason="exchange_mismatch_remote",
                context=mismatch_ctx,
            )
            if local_oid is not None and int(local_oid) != int(remote.oid):
                self._cancel_orphan_remote_order(
                    symbol=sym,
                    oid=int(local_oid),
                    cloid=local_cloid,
                    reason="exchange_mismatch_local_ghost",
                    context=mismatch_ctx,
                )
            transition(wo, OrderStatus.DESYNC, "exchange_mismatch")
            self.persist(wo)
            # 1.3.130 multi-rung Phase 2: clear specific per-rung slot.
            self._state.set_working_order(
                side, int(getattr(wo, "level_idx", 0) or 0), None
            )
            payload = dict(mismatch_ctx)
            payload["remote_cancel_dispatched"] = bool(cancel_ok_remote)
            # If the orphan cancel did NOT succeed (neither success nor benign_missing), the
            # remote phantom may still be live on the exchange. Latch the side so the failsafe
            # timeout in ``_is_side_unresolved`` cannot silently auto-release quoting over the
            # still-live phantom. Only a subsequent confirmed reconcile path clears the latch.
            self._set_side_unresolved(
                side,
                reason="exchange_mismatch",
                payload=payload,
                requires_confirm=(not bool(cancel_ok_remote)),
            )
            return True
        self._hydrate_working_from_exchange(side, remote)
        self._set_side_unresolved(
            side,
            reason="exchange_only_order_hydrated",
            payload={"remote_oid": int(remote.oid)},
        )
        return False

    def _central_pre_send_risk_check(
        self,
        side: Side,
        *,
        price: float,
        size: float,
        quote_cycle_id: str,
        reduce_only: bool = False,
        replacing_slot: Optional[tuple[Side, int]] = None,
    ) -> bool:
        """Last-resort pre-send gate. Returns ``True`` if the order is
        safe to stage, ``False`` if it must be refused (refusal logs
        CRITICAL with full context).

        Path-agnostic enforcement. Two layers of check:

        1. ``size * price <= max_order_notional_usd × max_order_notional_hard_multiplier``
           — single-order USD hard cap (default 2× hard multiplier).

        2. Worst-case post-fill position check: if this order fills
           AND any same-side resting order fills, would the resulting
           position exceed ``max_abs_position`` or
           ``max_position_notional_usd``? Refuse if so.

           **Reduce-only orders skip the position check.** Reduce-only
           is the venue's own guarantee that the order can ONLY close
           position; the post-fill worst case is "position closer to
           zero", which can't violate either cap regardless of what
           the bot's local view of position currently says. (This is
           important for soft-flatten resume after restart, where
           the bot's local position state may briefly disagree with
           the venue's truth.)

        Updated 2026-05-06 (Codex MED #1) to make checks 2 and 3
        path-agnostic: previously they relied on QuoteEngine's
        ``_clip_entry_sizes`` to be the only sizer. Now any path
        (manual place, soft-flatten, future paths) gets the same
        treatment.

        **v1.5.150 BUG-027 fix.** ``replacing_slot`` excludes the
        named (side, level_idx) slot's existing resting WO from the
        worst-case sum, because that slot is about to be cancelled
        by the same orchestrator action that's staging this place
        (the cancel-replace flow). Without this exclusion the same-
        slot replace's existing rung double-counted against the cap:
        ``resting (the one we're about to cancel) + new (same size)``
        breaches ``max_abs_position`` even though the resting will
        be cancelled before the new could fill. Observed pattern in
        snapshot ``v1.5.148-260525-201857``: 6,043 refusals (5,330
        at ``pos_qty=0``) producing 3 silent-wedge episodes; the
        same wedge signature against the ``v1.5.55-260523-190954``
        session was filed as BUG-027 with ``Phase 2A folds`` as the
        hypothesis. Today's log forensics rule out the folds and
        confirm this gate as the actual mechanism.

        Reduce-only orders bypass the position check entirely, so
        ``replacing_slot`` is a no-op when ``reduce_only=True``
        (the kwarg is still accepted for caller-symmetry).
        """
        if size is None or price is None:
            logger.critical(
                "central_pre_send_risk_REFUSED reason=null_price_or_size "
                "side=%s price=%s size=%s cycle=%s",
                side.value,
                price,
                size,
                quote_cycle_id,
            )
            return False
        try:
            sz = float(size)
            px = float(price)
        except (TypeError, ValueError):
            logger.critical(
                "central_pre_send_risk_REFUSED reason=non_numeric_price_or_size "
                "side=%s price=%s size=%s cycle=%s",
                side.value,
                price,
                size,
                quote_cycle_id,
            )
            return False
        if sz <= 0 or px <= 0:
            # Caller-zero is fine (means "we decided not to place"),
            # let downstream handle. But negative is a bug.
            if sz < 0 or px < 0:
                logger.critical(
                    "central_pre_send_risk_REFUSED reason=negative_price_or_size "
                    "side=%s price=%s size=%s cycle=%s",
                    side.value,
                    price,
                    size,
                    quote_cycle_id,
                )
                return False
            return True  # zero size; caller's downstream will skip

        notional_usd = sz * px

        # 2. Hard order-notional cap (max_order_notional × hard_mult)
        max_order = float(self._settings.max_order_notional_usd or 0.0)
        hard_mult = float(
            getattr(
                self._settings,
                "max_order_notional_hard_multiplier",
                2.0,
            )
            or 2.0
        )
        hard_cap = max_order * hard_mult
        if max_order > 0 and notional_usd > hard_cap + 1e-9:
            logger.critical(
                "central_pre_send_risk_REFUSED reason=order_notional_over_hard_cap "
                "side=%s price=%s size=%s notional_usd=%.4f hard_cap=%.4f "
                "(max_order_notional_usd=%.4f × %.2fx) cycle=%s",
                side.value,
                px,
                sz,
                notional_usd,
                hard_cap,
                max_order,
                hard_mult,
                quote_cycle_id,
            )
            return False

        # ----------------------------------------------------------
        # POST-FILL POSITION CAP CHECK (skipped on reduce_only)
        # ----------------------------------------------------------
        # Reduce-only is the venue-enforced "can only close" guarantee
        # -- exempt from the position cap check because the venue
        # itself bounds the post-fill position. For non-reduce-only
        # orders, the worst case is that this order AND any same-side
        # resting order both fill in full -- if that would push
        # |position| past the configured caps, refuse.
        if reduce_only:
            return True

        max_abs_pos = float(self._settings.max_abs_position or 0.0)
        max_pos_usd = float(self._settings.max_position_notional_usd or 0.0)
        if max_abs_pos <= 0 and max_pos_usd <= 0:
            return True  # no position caps configured

        # v1.4.76 wedge-elimination-cleanup Phase 2C — multi-rung
        # pre-send risk check.
        #
        # Pre-Phase-2C this read only the inside-rung slot
        # (``working_bid`` / ``working_ask``). With
        # ``LADDER_NUM_LEVELS_PER_SIDE >= 2``, outer rungs were
        # invisible to the worst-case exposure check, so two
        # legitimate same-side rungs could each pass the cap check
        # individually while their COMBINED fill would breach it
        # (Codex Bug 6). Phase 2C sums the size across every
        # in-flight or live same-side WO across all rungs.
        #
        # Statuses summed (the set of "this order could still fill"):
        #   NEW_LOCAL, SENT — placed but ack still in flight; venue may already have it
        #   ACKED, PARTIAL  — live on the book
        #   AMEND_PENDING   — live on the book, amend in flight (the
        #                     ORIGINAL order is still resting at venue
        #                     until the amend response commits)
        #
        # Statuses NOT summed (no longer fillable from the bot's
        # perspective):
        #   CANCEL_PENDING — bot has requested cancel; if it fills
        #                    that's a cancel/fill race outside the
        #                    risk check's authority
        #   CANCELED, FILLED, REJECTED, DESYNC — terminal/equivalent.
        with self._state._lock:
            cur_qty = float(self._state.position.position_qty)
            resting_same_sz = 0.0
            for _idx, wo in self._state.iter_working_orders(side):
                if wo is None:
                    continue
                # v1.5.150 BUG-027 fix — exclude the about-to-be-
                # cancelled slot. The reconciler's cancel-replace flow
                # stages this place AND issues the cancel for the same
                # slot in the same orchestrator action; counting the
                # resting WO against worst-case double-counts the
                # exposure that will (atomically, from the cap's
                # perspective) be removed.
                if (
                    replacing_slot is not None
                    and replacing_slot[0] == side
                    and int(replacing_slot[1]) == int(_idx)
                ):
                    continue
                if wo.status in (
                    OrderStatus.NEW_LOCAL,
                    OrderStatus.SENT,
                    OrderStatus.ACKED,
                    OrderStatus.PARTIAL,
                    OrderStatus.AMEND_PENDING,
                ):
                    resting_same_sz += float(wo.size or 0.0)

        # Worst case: this order + any resting same-side both fill.
        delta = sz + resting_same_sz
        worst_qty_after = (
            cur_qty + delta if side == Side.BUY else cur_qty - delta
        )

        if max_abs_pos > 0 and abs(worst_qty_after) > max_abs_pos + _POSITION_EPS:
            logger.critical(
                "central_pre_send_risk_REFUSED reason=worst_post_fill_qty_over_max_abs_position "
                "side=%s pos_qty=%.6f resting_same_sz=%.6f new_size=%.6f "
                "worst_qty_after=%.6f max_abs_position=%.4f cycle=%s",
                side.value,
                cur_qty,
                resting_same_sz,
                sz,
                worst_qty_after,
                max_abs_pos,
                quote_cycle_id,
            )
            return False

        if max_pos_usd > 0:
            worst_notional_after = abs(worst_qty_after) * px
            if worst_notional_after > max_pos_usd + 1e-9:
                logger.critical(
                    "central_pre_send_risk_REFUSED reason=worst_post_fill_notional_over_cap "
                    "side=%s pos_qty=%.6f resting_same_sz=%.6f new_size=%.6f "
                    "worst_qty_after=%.6f price=%.6f worst_notional_usd=%.4f "
                    "max_position_notional_usd=%.4f cycle=%s",
                    side.value,
                    cur_qty,
                    resting_same_sz,
                    sz,
                    worst_qty_after,
                    px,
                    worst_notional_after,
                    max_pos_usd,
                    quote_cycle_id,
                )
                return False

        return True

    def _stage_place_order_local(
        self,
        side: Side,
        *,
        price: float,
        size: float,
        quote_cycle_id: str,
        reduce_only: bool = False,
        target_half_spread_bps: Optional[float] = None,
        aging_tighten_applied: bool = False,
        decision: Optional[QuoteDecision] = None,
        level_idx: int = 0,
        replacing_slot: Optional[tuple[Side, int]] = None,
    ) -> Optional[WorkingOrder]:
        """Create working order, persist SENT, assign per-side intent sequence; **no HTTP**.

        This is the funnel point for **every** order that ever gets
        sent -- QuoteEngine output, soft-flatten worker, operator
        manual-place, kill-path flatten, retries. The centralized
        pre-send risk gate below enforces the configured caps
        (max_order_notional, max_abs_position, max_position_notional_usd)
        regardless of which path produced the order. Failures log
        CRITICAL and refuse to stage; nothing leaves the bot.

        Added 2026-05-06 after the soft-flatten incident where the
        worker bypassed every cap by calling ``_submit_passive_order_verbatim``
        directly with ``size = abs(pos_qty)`` and put $2700-notional
        orders on the book against a $20 configured cap.
        """
        if not self._client.has_write_access():
            self._last_stage_place_skip_reason[side.value] = "no_write_access"
            return None
        # ----------------------------------------------------------
        # CENTRALIZED PRE-SEND RISK GATE
        # ----------------------------------------------------------
        # Same checks every code path -- no exceptions, no bypass.
        # ``reduce_only`` is threaded so the gate can skip the
        # post-fill position checks (venue enforces non-grow on
        # reduce-only orders).
        if not self._central_pre_send_risk_check(
            side,
            price=price,
            size=size,
            quote_cycle_id=quote_cycle_id,
            reduce_only=reduce_only,
            replacing_slot=replacing_slot,
        ):
            self._last_stage_place_skip_reason[side.value] = "central_pre_send_risk_check_failed"
            return None
        if self._is_side_unresolved(side):
            self._suppress_place_due_unresolved_count += 1
            # Log once per unresolved episode (on the transition into
            # the suppressing state). Subsequent suppressions in the
            # same episode are silent counter bumps.
            if not self._suppress_place_unresolved_logged.get(side, False):
                self._suppress_place_unresolved_logged[side] = True
                log_extra(
                    logger,
                    logging.INFO,
                    "same_side_place_suppressed",
                    {
                        "side": side.value,
                        "reason": "side_unresolved",
                        "unresolved_reason": self._side_unresolved_reason.get(side),
                    },
                )
            self._last_stage_place_skip_reason[side.value] = (
                f"side_unresolved:{self._side_unresolved_reason.get(side) or 'unknown'}"
            )
            return None
        if self._adverse_side_pause_active(side):
            # BUG-010: reducing-side bypass — never block the inventory-reducing
            # side on adverse-side pause, otherwise a localised toxicity signal
            # can strand the bot with directional inventory. Adding-side
            # behaviour (full pause) is preserved.
            if (
                bool(self._settings.adverse_side_pause_bypass_reducing_side)
                and self._is_reducing_side(side)
            ):
                remaining_s = max(
                    0.0, self._adverse_side_pause_until.get(side, 0.0) - self._clock.monotonic()
                )
                log_extra(
                    logger,
                    logging.INFO,
                    "adverse_side_pause_reducing_bypass",
                    {
                        "side": side.value,
                        "remaining_seconds": round(remaining_s, 3),
                        "position_qty": float(self._state.position.position_qty),
                    },
                )
                # fall through — let the placement proceed.
            else:
                self._adverse_side_pause_skip_count[side] += 1
                remaining_s = max(
                    0.0, self._adverse_side_pause_until.get(side, 0.0) - self._clock.monotonic()
                )
                log_extra(
                    logger,
                    logging.INFO,
                    "adverse_side_pause_skip",
                    {
                        "side": side.value,
                        "remaining_seconds": round(remaining_s, 3),
                        "count_total": self._adverse_side_pause_skip_count[side],
                    },
                )
                self._last_stage_place_skip_reason[side.value] = (
                    f"adverse_side_pause:remaining_s={round(remaining_s, 3)}"
                )
                return None
        if self._post_only_cross_cooldown_active(side):
            # v1.4.60 wedge-elimination Phase 5: reducing-side bypass.
            # Mirror the ``adverse_side_pause_bypass_reducing_side``
            # behaviour (BUG-010) — when the suppressed side is the
            # REDUCING side AND inventory is non-zero, bypass the
            # cooldown so the bot can flatten. The venue's 51604
            # post-only-cross cost is small compared to "bot can't
            # reduce a +1 long during a downward move."
            #
            # The v1.4.59-260518-173229 wedge proved this bypass is
            # needed: 909 SELL stage_returned_none events from this
            # cooldown blocked the reducer for 90+ s while position
            # +1 long needed to be flattened. The bot has 6 venue-
            # side cooldown arms per session; bypassing the reducer
            # turns that into "venue rejects occasionally but the
            # bot keeps trying to flatten" instead of "bot wedges
            # for ~5 s per arm".
            #
            # The reducing-side bypass is the central invariant of
            # the Phase 5 reconciler architecture. Conceptually it's
            # the same fold ``apply_reducing_side_bypass`` defined
            # in ``app/reconciler.py`` — encoded here at stage time
            # so all gates (current AND any future ones added to
            # _stage_place_order_local) inherit the same protection
            # without an N×N "did I remember the bypass?" review.
            position_qty = float(self._state.position.position_qty)
            bypass_enabled = bool(
                getattr(
                    self._settings,
                    "post_only_cross_cooldown_bypass_reducing_side",
                    True,
                )
            )
            if (
                bypass_enabled
                and position_qty != 0.0
                and self._is_reducing_side(side)
            ):
                remaining_s = max(
                    0.0, self._post_only_cross_cooldown_until.get(side, 0.0) - self._clock.monotonic()
                )
                log_extra(
                    logger,
                    logging.INFO,
                    "post_only_cross_cooldown_reducing_bypass",
                    {
                        "side": side.value,
                        "remaining_seconds": round(remaining_s, 3),
                        "position_qty": position_qty,
                    },
                )
                # fall through — let the reducer placement proceed.
            else:
                remaining_s = max(
                    0.0, self._post_only_cross_cooldown_until.get(side, 0.0) - self._clock.monotonic()
                )
                log_extra(
                    logger,
                    logging.INFO,
                    "post_only_cross_cooldown_skip",
                    {
                        "side": side.value,
                        "remaining_seconds": round(remaining_s, 3),
                    },
                )
                self._last_stage_place_skip_reason[side.value] = (
                    f"post_only_cross_cooldown:remaining_s={round(remaining_s, 3)}"
                )
                return None
        with self._state._lock:
            # 1.3.130 multi-rung Phase 2: check the SPECIFIC rung slot
            # rather than always the inside rung. With N>1 the inside
            # bid can be ACKED while the outer bid slot (level_idx=1)
            # is empty — that outer slot should still be placeable.
            ex = self._state.get_working_order(side, int(level_idx))
        if ex and ex.status in (
            OrderStatus.CANCEL_PENDING,
            OrderStatus.SENT,
            OrderStatus.ACKED,
            OrderStatus.PARTIAL,
        ):
            # v1.4.50: this is the race-window guard. ``_orchestrate``
            # already gated on cur.status, but a private-WS handler can
            # transition the WO between the cur read at line 8227 and
            # the ex read here. When this fires it's a benign race; the
            # next quote cycle re-evaluates. Record so we can see if
            # the race is fast or pathological.
            self._last_stage_place_skip_reason[side.value] = (
                f"existing_wo_active_race:status={ex.status.value}"
            )
            logger.info(
                "stage_place_skipped_existing_wo side=%s level_idx=%d "
                "existing_status=%s existing_oid_ex=%s",
                side.value,
                int(level_idx),
                ex.status.value,
                str(ex.order_id_exchange) if ex.order_id_exchange else None,
            )
            return None
        fp, fs = float(price), float(size)
        if not math.isfinite(fp) or not math.isfinite(fs) or fp <= 0 or fs <= 0:
            self._last_stage_place_skip_reason[side.value] = (
                f"invalid_px_or_sz:px={fp},sz={fs}"
            )
            logger.warning(
                "stage_place_skipped_invalid_input side=%s level_idx=%d px=%s sz=%s",
                side.value,
                int(level_idx),
                fp,
                fs,
            )
            return None
        self._place_intent_seq[side] += 1
        seq = self._place_intent_seq[side]
        oid_local = str(uuid.uuid4())
        # Delegate to the adapter so each venue produces its own cloid format
        # (Hyperliquid: 0x + 32 hex chars; GRVT: uint64 decimal string). A single
        # shared format would silently fail validation on one of the venues —
        # GRVT previously rejected HL-shaped cloids as non-uint64, producing
        # 400 Bad Request on every ``create_order`` submit.
        cloid = self._client.make_client_order_id(
            self._settings.symbol, side, quote_cycle_id, fp, fs
        )
        # Stamp the SF episode id on every order placed during an
        # active SF episode. ``state.soft_flatten_event_id`` is None
        # outside SF, so non-SF orders are correctly tagged NULL.
        # Both placement paths reach this constructor:
        #   - normal MM (``maybe_refresh_quotes``) — paused during SF,
        #     so its quote_cycle_id orders get NULL SF FK.
        #   - SF worker (``_run_soft_flatten_tick`` →
        #     ``place_passive_order_manual_only``) — fires only when
        #     state.soft_flatten_event_id is set, so SF orders get the
        #     active episode id. Plan ref:
        #     plans/20260507-sf-frontend.md Phase 2.
        sf_event_id = self._state.soft_flatten_event_id
        # v1.5.33 — TP attribution. Mirror of SF: non-None only during
        # an active TP episode. Stamped on the WorkingOrder so the
        # storage row + the fill row both pick it up via the parent-
        # order metadata lookup. SF and TP are mutually exclusive at
        # the executor level (TP defers when SF is active), so at most
        # one of the two FKs is non-None on any given order.
        tp_event_id = getattr(self._state, "tp_event_id", None)
        # todo-005 / todo-006 quality tag. Compares the chosen price
        # against the current best bid/ask snapshot and the aging-
        # tighten signal:
        #   - ``aged_tightened`` if the QuoteEngine's quote-aging path
        #     pushed this price tighter than its model target;
        #   - ``inside`` if we're posting strictly tighter than the
        #     current touch (improving the venue's best price);
        #   - ``at_touch`` if we're at the existing best price (most
        #     passive — joining the queue at the best level);
        #   - ``behind_touch`` if we're posting at a worse price than
        #     the current best (joining the queue at a deeper level);
        #   - ``unknown`` when market data is missing (no BBO snapshot
        #     yet, or the tick comparison can't be made).
        # The classification is best-effort and lock-free: we only
        # snapshot ``state.market`` once for the comparison. Note:
        # ``inside`` is structurally impossible for this bot because
        # ``QuoteEngine._build_side`` clamps bid ≤ best_bid (and ask
        # ≥ best_ask) for post-only safety. The category exists for
        # completeness in case that policy is ever relaxed.
        with self._state._lock:
            mkt_snap = self._state.market
        bb = getattr(mkt_snap, "best_bid", None) if mkt_snap is not None else None
        ba = getattr(mkt_snap, "best_ask", None) if mkt_snap is not None else None
        spec_for_cmp = getattr(self._client, "symbol_spec", None)
        tick_for_cmp = (
            float(spec_for_cmp.price_tick)
            if spec_for_cmp is not None
            and getattr(spec_for_cmp, "price_tick", None) is not None
            else 0.0
        )
        eps = max(tick_for_cmp * 0.5, 1e-12)
        if aging_tighten_applied:
            quote_aggressiveness = "aged_tightened"
        elif side == Side.BUY and bb is not None and math.isfinite(float(bb)):
            if fp > float(bb) + eps:
                quote_aggressiveness = "inside"
            elif abs(fp - float(bb)) <= eps:
                quote_aggressiveness = "at_touch"
            else:
                quote_aggressiveness = "behind_touch"
        elif side == Side.SELL and ba is not None and math.isfinite(float(ba)):
            if fp < float(ba) - eps:
                quote_aggressiveness = "inside"
            elif abs(fp - float(ba)) <= eps:
                quote_aggressiveness = "at_touch"
            else:
                quote_aggressiveness = "behind_touch"
        else:
            quote_aggressiveness = "unknown"

        # 2026-05-13 regime-observability Phase 1: decision-state
        # stamping. Read all fields lock-free (state is already
        # under the lock above where needed) at WorkingOrder
        # construction. Decision-derived fields come from the
        # ``decision`` parameter when threaded (normal MM path);
        # state-derived fields read directly from ``self._state``
        # to stay venue-agnostic. NULL on legacy / SF / manual
        # paths where the decision isn't threaded.
        _toxicity = (
            float(decision.toxicity_score) if decision is not None else None
        )
        _vol_est = (
            float(decision.vol_estimate) if decision is not None else None
        )
        _active_sides = (
            str(decision.active_sides.value)
            if decision is not None
            and getattr(decision, "active_sides", None) is not None
            and hasattr(decision.active_sides, "value")
            else None
        )
        _decision_reason = (
            str(decision.decision_reason)
            if decision is not None and decision.decision_reason
            else None
        )
        _binance_basis_ewma = getattr(
            self._state, "binance_basis_ewma", None
        )
        if _binance_basis_ewma is not None:
            try:
                _binance_basis_ewma = float(_binance_basis_ewma)
            except (TypeError, ValueError):
                _binance_basis_ewma = None
        # Adaptive widen + post-fill cooldown + at-touch-adverse-pause:
        # read from the existing per-side state caches that the bot
        # maintains for the gates themselves. These reads do NOT
        # acquire the strategy lock; they're best-effort snapshots.
        #
        # 2026-05-13 Codex bug review MED #5: ``adaptive_widen_active``
        # is a property on ``BotState`` (added in the same review)
        # that derives the flag from ``adaptive_spread_widen_until_mono``.
        # Pre-fix, this getattr fell back to ``False`` because no such
        # attribute existed — so every order / fill silently stamped
        # ``adaptive_widen_active_at_decision = False`` even mid-widen,
        # breaking regime slicing by that field.
        _adaptive_widen = None
        try:
            _adaptive_widen = bool(
                getattr(self._state, "adaptive_widen_active", False)
            )
        except Exception:
            _adaptive_widen = None
        try:
            _post_fill_bid = side == Side.BUY and bool(
                getattr(self, "_post_fill_cooldown_active", lambda s: False)(Side.BUY)
            )
        except Exception:
            _post_fill_bid = None
        try:
            _post_fill_ask = side == Side.SELL and bool(
                getattr(self, "_post_fill_cooldown_active", lambda s: False)(Side.SELL)
            )
        except Exception:
            _post_fill_ask = None
        try:
            _at_touch_pause_bid = bool(
                getattr(self, "_at_touch_adverse_pause", None)
                and self._at_touch_adverse_pause.is_paused(Side.BUY)  # type: ignore[attr-defined]
            )
        except Exception:
            _at_touch_pause_bid = None
        try:
            _at_touch_pause_ask = bool(
                getattr(self, "_at_touch_adverse_pause", None)
                and self._at_touch_adverse_pause.is_paused(Side.SELL)  # type: ignore[attr-defined]
            )
        except Exception:
            _at_touch_pause_ask = None
        # Quote-distance-to-touch in ticks at the moment of placement.
        # Uses the same bb/ba snapshot + tick already computed for the
        # aggressiveness classifier above.
        _qdtt_ticks: Optional[float] = None
        if tick_for_cmp > 0:
            if side == Side.BUY and bb is not None and math.isfinite(float(bb)):
                _qdtt_ticks = max(0.0, (float(bb) - fp) / tick_for_cmp)
            elif side == Side.SELL and ba is not None and math.isfinite(float(ba)):
                _qdtt_ticks = max(0.0, (fp - float(ba)) / tick_for_cmp)

        # 2026-05-13 Phase 4a: expected_net_edge_bps at decision.
        # Placeholder formula (STAMP ONLY — bot's quote-construction
        # path does NOT read this). Operator can calibrate the priors
        # per profile via OBSERVABILITY_MAKER_REBATE_BPS and
        # OBSERVABILITY_TYPICAL_ADVERSE_MARKOUT_BPS. See
        # plans/regime-observability.md Phase 4a for the formula
        # caveats.
        _expected_edge: Optional[float] = None
        if target_half_spread_bps is not None and math.isfinite(
            float(target_half_spread_bps)
        ):
            try:
                rebate_bps = float(
                    getattr(
                        self._settings,
                        "observability_maker_rebate_bps",
                        1.0,
                    )
                )
                adverse_bps = float(
                    getattr(
                        self._settings,
                        "observability_typical_adverse_markout_bps",
                        2.0,
                    )
                )
                _expected_edge = (
                    float(target_half_spread_bps) + rebate_bps - adverse_bps
                )
            except (TypeError, ValueError):
                _expected_edge = None

        # 2026-05-14 todo-027 Tier 2: capture the three previously
        # fire-counted-only gates at place-time so they appear on the
        # order (and propagate to the resulting fill). Same defensive
        # pattern as the other _at_decision captures above — gate
        # objects may not be wired in every test/profile context, and
        # any failure here must NOT block order placement (these are
        # observation fields only).
        _vol_trend_active: Optional[bool] = None
        try:
            vt = getattr(self._state, "vol_trend_gate", None)
            if vt is not None:
                vt_until = float(
                    getattr(vt, "cooldown_until_mono", 0.0) or 0.0
                )
                _vol_trend_active = self._clock.monotonic() < vt_until
        except Exception:
            _vol_trend_active = None

        _post_swing_active: Optional[bool] = None
        try:
            ps = getattr(self._state, "post_swing", None)
            if ps is not None:
                ps_until = float(
                    getattr(ps, "cooldown_until_mono", 0.0) or 0.0
                )
                _post_swing_active = self._clock.monotonic() < ps_until
        except Exception:
            _post_swing_active = None

        _session_drawdown_tier: Optional[str] = None
        try:
            sd = getattr(self._state, "session_drawdown", None)
            if sd is not None:
                tier = getattr(sd, "tier", None)
                if tier is not None:
                    _session_drawdown_tier = (
                        getattr(tier, "value", None) or str(tier)
                    )
        except Exception:
            _session_drawdown_tier = None

        # v1.4.175 Phase 3F — reservation-alpha shift contributions
        # read from the just-computed ``state.last_quote_breakdown``.
        # The breakdown is populated by ``compute_quote_decision`` at
        # the end of each quote cycle (before this place runs), so the
        # values are fresh when the WO is constructed. NULL when no
        # breakdown is available (legacy / SF / unit-test paths).
        _ob_imbalance_shift: Optional[float] = None
        _trend_drift_shift: Optional[float] = None
        _flow_score_shift: Optional[float] = None
        _basis_deviation_shift: Optional[float] = None
        # v1.5.190 Phase 8A Option C — per-order AS attribution. Same
        # ``state.last_quote_breakdown`` source as the Phase 3F shifts
        # above; this is the AS-computed base half-spread when AS is
        # enabled, or the legacy vol-adaptive base when AS is disabled.
        # Defensive — observation field only, not load-bearing.
        _as_base_half_spread: Optional[float] = None
        # v1.5.204 Phase 4A — per-order microprice gate widening.
        # SpreadComposition.microprice_{bid,ask}_bps holds the bps
        # the microprice gate contributed to each side's half-spread
        # at decision time; >0 on the thin side when the gate fired,
        # 0 otherwise. Both NULL on paths where no breakdown is
        # available. Used by the Phase 4A.3 acceptance check.
        _microprice_bid_widen: Optional[float] = None
        _microprice_ask_widen: Optional[float] = None
        try:
            bk = getattr(self._state, "last_quote_breakdown", None)
            if bk is not None:
                _ob_imbalance_shift = float(
                    getattr(bk, "ob_imbalance_shift_bps", 0.0) or 0.0
                )
                _trend_drift_shift = float(
                    getattr(bk, "trend_drift_shift_bps", 0.0) or 0.0
                )
                _flow_score_shift = float(
                    getattr(bk, "flow_score_shift_bps", 0.0) or 0.0
                )
                _basis_deviation_shift = float(
                    getattr(bk, "basis_deviation_shift_bps", 0.0) or 0.0
                )
                _bh = getattr(bk, "base_half_spread_bps", None)
                if _bh is not None:
                    _bh_f = float(_bh)
                    if math.isfinite(_bh_f):
                        _as_base_half_spread = _bh_f
                # v1.5.229 BUG FIX: pre-v1.5.229 this code read
                # ``bk.spread_composition.microprice_{bid,ask}_bps``,
                # but ``spread_composition`` was never a field on
                # ``QuoteBreakdownSnapshot`` — so this lookup
                # silently returned None on every tick and every
                # fill's microprice attribution column was NULL.
                # The Phase 4A acceptance check
                # (``check_v1_5_204_microprice_widen_markout_within_noise``)
                # has been returning N/A on every snapshot since
                # v1.5.204 shipped. Operator caught the gap
                # 2026-05-28 on Phase 5 snapshot v1.5.225.
                #
                # Fix: read the per-side widening from the two new
                # top-level breakdown fields populated by
                # compute_quote_decision from its ``composition``
                # kwarg.
                try:
                    _mp_bid = float(getattr(bk, "microprice_bid_widen_bps", 0.0) or 0.0)
                    if math.isfinite(_mp_bid):
                        _microprice_bid_widen = _mp_bid
                    _mp_ask = float(getattr(bk, "microprice_ask_widen_bps", 0.0) or 0.0)
                    if math.isfinite(_mp_ask):
                        _microprice_ask_widen = _mp_ask
                except (TypeError, ValueError):
                    _microprice_bid_widen = None
                    _microprice_ask_widen = None
        except Exception:
            # Defensive — breakdown is observability, not load-bearing.
            _ob_imbalance_shift = None
            _trend_drift_shift = None
            _flow_score_shift = None
            _basis_deviation_shift = None
            _as_base_half_spread = None
            _microprice_bid_widen = None
            _microprice_ask_widen = None

        # v1.5.306 audit §5 P0 #2 — per-order AQC attribution. Source is
        # ``state.active_quoting_controller`` (NOT last_quote_breakdown):
        # the PI aggression output + the markout safety-floor flag at the
        # moment this order is placed. Controller is None when AQC is
        # disabled → both stay None. Observability-only, so wrapped in a
        # blanket try/except that leaves both None on any failure (must
        # never break order placement).
        _aqc_aggression: Optional[float] = None
        _aqc_safety_floor: Optional[bool] = None
        try:
            _aqc = getattr(self._state, "active_quoting_controller", None)
            if _aqc is not None:
                _agg = getattr(_aqc, "aggression_level", None)
                if _agg is not None:
                    _agg_f = float(_agg)
                    if math.isfinite(_agg_f):
                        _aqc_aggression = _agg_f
                _aqc_safety_floor = bool(
                    getattr(_aqc, "safety_floor_engaged", False)
                )
        except Exception:
            _aqc_aggression = None
            _aqc_safety_floor = None

        wo = WorkingOrder(
            order_id_local=oid_local,
            order_id_exchange=None,
            client_order_id=cloid,
            symbol=self._settings.symbol,
            side=side,
            price=fp,
            size=fs,
            post_only=True,
            status=OrderStatus.NEW_LOCAL,
            reduce_only=bool(reduce_only),
            quote_cycle_id=quote_cycle_id,
            transport_intent_seq=seq,
            soft_flatten_event_id=sf_event_id,
            tp_event_id=tp_event_id,
            target_half_spread_bps=target_half_spread_bps,
            quote_aggressiveness=quote_aggressiveness,
            toxicity_score_at_decision=_toxicity,
            vol_estimate_at_decision=_vol_est,
            active_sides_at_decision=_active_sides,
            decision_reason_at_decision=_decision_reason,
            binance_basis_ewma_at_decision=_binance_basis_ewma,
            adaptive_widen_active_at_decision=_adaptive_widen,
            post_fill_cooldown_active_bid_at_decision=_post_fill_bid,
            post_fill_cooldown_active_ask_at_decision=_post_fill_ask,
            at_touch_adverse_pause_bid_at_decision=_at_touch_pause_bid,
            at_touch_adverse_pause_ask_at_decision=_at_touch_pause_ask,
            quote_distance_to_touch_ticks_at_placement=_qdtt_ticks,
            expected_net_edge_bps_at_decision=_expected_edge,
            # 2026-05-14 todo-027 Tier 2 stamps.
            vol_trend_active_at_decision=_vol_trend_active,
            post_swing_active_at_decision=_post_swing_active,
            session_drawdown_tier_at_decision=_session_drawdown_tier,
            # v1.4.175 Phase 3F — reservation-alpha shifts at decision.
            ob_imbalance_shift_bps_at_decision=_ob_imbalance_shift,
            trend_drift_shift_bps_at_decision=_trend_drift_shift,
            flow_score_shift_bps_at_decision=_flow_score_shift,
            basis_deviation_shift_bps_at_decision=_basis_deviation_shift,
            # v1.5.190 Phase 8A Option C — per-order AS attribution.
            as_base_half_spread_bps_at_decision=_as_base_half_spread,
            # v1.5.204 Phase 4A — per-order microprice widen attribution.
            microprice_bid_widen_bps_at_decision=_microprice_bid_widen,
            microprice_ask_widen_bps_at_decision=_microprice_ask_widen,
            # v1.5.306 audit §5 P0 #2 — per-order AQC attribution.
            aqc_aggression_level_at_decision=_aqc_aggression,
            aqc_safety_floor_engaged_at_decision=_aqc_safety_floor,
            # 1.3.130 multi-rung Phase 2: which rung this order
            # represents (0 = inside, default).
            level_idx=int(level_idx),
        )
        self.persist(wo)
        transition(wo, OrderStatus.SENT)
        self.persist(wo)
        with self._state._lock:
            # 1.3.130 multi-rung Phase 2: write to the per-rung slot.
            # At level_idx=0 (default — single-rung mode) this hits the
            # same slot the working_bid / working_ask property shims
            # back, so all 60+ legacy read sites see the same WO they
            # did pre-Phase-2.
            self._state.set_working_order(side, int(level_idx), wo)
        # v1.4.50: clear any stale skip-reason from a prior failed
        # stage attempt. Successful staging always supersedes a prior
        # skip reason for the same side.
        self._last_stage_place_skip_reason[side.value] = None
        return wo

    def _enqueue_place_transport(self, wo: WorkingOrder, quote_cycle_id: str) -> bool:
        """Queue outbound place intent; strategy thread returns immediately (non-blocking)."""
        if self._first_enqueue_perf is None:
            self._first_enqueue_perf = time.perf_counter()
        enq = time.perf_counter()
        ic = float(self._maybe_refresh_t0_perf) if self._maybe_refresh_t0_perf else enq
        tr = OutboundActionTrace(
            order_id_local=wo.order_id_local,
            side_value=wo.side.value,
            intent_created_perf=ic,
            dispatcher_enqueue_perf=enq,
            quote_cycle_id=quote_cycle_id,
        )
        self._outbound_traces[wo.order_id_local] = tr
        self._outbound.submit_place(
            PlaceTransportIntent(
                wo.order_id_local,
                wo.side,
                wo.transport_intent_seq,
                quote_cycle_id,
                self._clock.monotonic(),
                intent_created_perf=ic,
                # 1.3.130 multi-rung Phase 2: rung identifier for
                # per-(side, level_idx) dispatcher coalescing.
                level_idx=int(getattr(wo, "level_idx", 0) or 0),
            )
        )
        self._sync_outbound_state_flags()
        return True

    def _complete_place_http(
        self,
        wo: WorkingOrder,
        *,
        quote_cycle_id: str,
        intent_seq: int,
    ) -> None:
        """HTTP submit + response handling; **transport worker or sync manual path**."""
        with self._state._lock:
            # 1.3.130 multi-rung Phase 2: look up the per-rung slot. At
            # level_idx=0 this returns the inside rung (same as the
            # legacy ``working_bid`` / ``working_ask`` lookup).
            slot = self._state.get_working_order(wo.side, int(wo.level_idx))
        if slot is None or slot.order_id_local != wo.order_id_local:
            logger.debug(
                "place_http_stale_skip slot_mismatch local_id=%s",
                wo.order_id_local,
            )
            return
        if slot.transport_intent_seq != intent_seq:
            logger.info(
                "place_http_stale_skip intent_seq local_id=%s expected=%s got=%s",
                wo.order_id_local,
                intent_seq,
                slot.transport_intent_seq,
            )
            return
        if wo.status != OrderStatus.SENT:
            return
        tr = self._outbound_traces.get(wo.order_id_local)
        try:
            if tr is not None:
                tr.sign_start_perf = time.perf_counter()
            t_net = time.perf_counter()
            # Watchdog signal: mark that we DID dispatch an order attempt.
            # Set here (before the call) so even if place_post_only_limit
            # raises, the watchdog sees "execution was active" — the
            # class we're protecting against is "execution silently
            # refuses to attempt" (latched pending-cancel gate, cascading
            # rejection avoidance, etc.), which never reaches this point.
            with self._state._lock:
                self._state.last_place_attempt_ts_mono = t_net
                # v1.4.42: any-outbound-attempt tracker (see state.py).
                self._state.last_outbound_attempt_ts_mono = t_net
                self._state.session_place_attempt_count += 1
            # Rolling place-to-fill ratio: feed the place side here
            # so the gate sees actual order-send activity, not just
            # intent-creation. Tracker has its own lock.
            self._state.place_to_fill_ratio_tracker.note_place()
            # Phase 2 trace: record the order at the moment we send it
            # to the venue. Lookups by cloid (we don't have ex_oid yet).
            try:
                self._state.order_trace.begin_order(
                    client_order_id=wo.client_order_id or "",
                    side=wo.side.value,
                    price=float(wo.price),
                    size_base=float(wo.size),
                )
            except Exception:
                logger.exception("order_trace_begin_failed")
            # ``reduce_only`` is sourced from the WorkingOrder created
            # by ``_stage_place_order_local``; soft-flatten path sets it
            # True so the venue refuses any post-zero growth.
            resp = self._client.place_post_only_limit(
                self._settings.symbol,
                wo.side == Side.BUY,
                wo.size,
                wo.price,
                client_order_id=wo.client_order_id,
                reduce_only=bool(getattr(wo, "reduce_only", False)),
            )
            t_done = time.perf_counter()
            if tr is not None:
                tr.sign_end_perf = t_done
                tr.transport_send_perf = t_net
                tr.transport_write_done_perf = t_done
                lm = float(getattr(self._client, "last_exchange_signing_ms", 0.0) or 0.0)
                if lm > 0:
                    tr.sign_start_perf = max(tr.sign_start_perf, t_done - lm / 1000.0)
                tr.transport_mode = str(getattr(self._client, "last_exchange_transport_mode", "http"))
            rtt_ms = (t_done - t_net) * 1000.0
            self._last_tick_order_submit_rtt_ms = max(
                float(self._last_tick_order_submit_rtt_ms), float(rtt_ms)
            )
            # Feed the rolling RTT tracker (place-to-ack distribution).
            # Outcome is "accepted" if the response interpreter classified
            # it as accepted; downstream code paths set this flag elsewhere
            # but at this layer we only know the transport completed
            # cleanly — finer outcome attribution can be added later if
            # the operator wants per-outcome percentiles.
            try:
                self._order_rtt_tracker.ingest(
                    rtt_ms=rtt_ms,
                    op="place",
                    outcome="accepted",
                )
            except Exception:
                logger.exception("order_rtt_tracker_ingest_failed")
            self._bump_transport_counters()
        except Exception:
            logger.exception("submit_passive_order_verbatim failed")
            transition(wo, OrderStatus.REJECTED, "exception")
            self.persist(wo)
            self._release_working_slot_if_matches(wo)
            self._state.bump_execution_errors("place_order_exception")
            try:
                self._state.order_trace.record_terminal(
                    client_order_id=wo.client_order_id or "",
                    status=OrderStatus.REJECTED.value,
                    reason="exception",
                    source="place_response",
                )
            except Exception:
                logger.exception("order_trace_terminal_failed")
            return

        ex_oid, outcome, reason = self._interpret_place_response(resp)
        # 1.3.82: stamp the place-response classification directly on the
        # WorkingOrder (not just the in-memory order_trace). The trace
        # is a bounded ring buffer; the WO row is persistent. When
        # ``ts_place_response is None`` on a closed order, the response
        # was never received — that's the phantom-place signature.
        ts_iso_now = self._clock.now_utc().isoformat()
        wo.ts_place_response = self._clock.now_utc()
        wo.place_response_outcome = (outcome or None)
        # 1.3.83: also stamp the venue-side detail. For accepted rows
        # the interpreter returns reason="" — leave detail NULL there
        # so the Connectivity tab can group only on rejection reasons.
        if reason:
            wo.place_response_detail = reason[:400]
        try:
            self._state.order_trace.record_place_response(
                client_order_id=wo.client_order_id or "",
                outcome=outcome,
                detail=reason,
                order_id_exchange=ex_oid,
            )
        except Exception:
            logger.exception("order_trace_place_response_failed")
        # 1.4.6: STICKY rejection summary — never wiped, survives past
        # the 5000-row orders_lifecycle window. See
        # ``BotState.record_place_rejection`` for rationale.
        if outcome and outcome != "accepted":
            try:
                self._state.record_place_rejection(
                    outcome=outcome,
                    detail=reason or "",
                    is_benign=_is_benign_place_outcome(outcome, reason),
                    ts_iso=ts_iso_now,
                )
            except Exception:
                logger.exception("record_place_rejection_failed")
        # 1.4.6: per-outcome cumulative counter + latency moments —
        # what the dashboard reads for total-placements / ack-rate
        # so those numbers stay correct after the 5000-row window
        # evicts older orders.
        try:
            self._state.record_place_outcome(
                outcome=outcome or "unknown",
                latency_ms=rtt_ms if outcome == "accepted" else None,
            )
        except Exception:
            logger.exception("record_place_outcome_failed")
        # Defensive: no real exchange uses 0 as an order id. GRVT in particular
        # may return ``order_id: null`` / ``"0"`` on synchronous accept while the
        # matching engine is still assigning the real 128-bit id. Treat 0 as
        # "not yet assigned" so we don't ACK the local wo with a bogus oid that
        # would later confuse the cancel path.
        if ex_oid == 0:
            ex_oid = None
            outcome = "unconfirmed"
            reason = (reason or "place_ack_without_oid")
        if ex_oid is None:
            if outcome in ("exchange_rejected", "transport_rejected"):
                # Surface the venue's rejection reason at WARNING so operators
                # can see why a post-only place failed without grepping the raw
                # HTTP layer or the SQLite order row. The venue-specific adapter
                # has already logged a lower-level ``<venue>_create_order_rejected``
                # line with the full wire context.
                logger.warning(
                    "place_order_rejected side=%s symbol=%s price=%s size=%s "
                    "outcome=%s reason=%s",
                    wo.side.value,
                    wo.symbol,
                    f"{wo.price:.12g}",
                    f"{wo.size:.12g}",
                    outcome,
                    (reason or "")[:400],
                )
                # Post-only cross-reject: arm the per-side cooldown so we don't
                # immediately re-place at BBO (would cross again if the book
                # hasn't moved). Benign — recovers automatically when cooldown
                # lapses. Enables the ``POST_ONLY_TOUCH_BUFFER_TICKS=1``
                # (BBO-join) strategy to be safe: the worst case is one cross
                # and then a bounded-duration skip, not a reject-spam loop.
                if is_post_only_immediate_match_rejection(reason or ""):
                    self._arm_post_only_cross_cooldown(wo.side)
                    try:
                        self._state.quote_quality.note_post_only_cross_rejection()
                    except Exception:
                        logger.exception("note_post_only_cross_rejection_failed")
                    log_extra(
                        logger,
                        logging.INFO,
                        "post_only_cross_rejection_armed_cooldown",
                        {
                            "side": wo.side.value,
                            "cooldown_seconds": float(
                                self._settings.post_only_cross_cooldown_seconds
                            ),
                            "reason": (reason or "")[:200],
                        },
                    )
                transition(wo, OrderStatus.REJECTED, (reason or outcome)[:2000])
                # 1.3.86: order was rejected at the venue → it never
                # existed there → there is nothing to cancel. Clear
                # any deferred-cancel intent so the post-ack flush
                # path (which won't run, but defensive) can't re-fire.
                if wo.cancel_pending_after_ack:
                    wo.cancel_pending_after_ack = False
                    logger.info(
                        "cancel_discarded_place_rejected side=%s cloid=%s reason=%s",
                        wo.side.value,
                        (wo.client_order_id[:18] + "...")
                        if wo.client_order_id
                        else None,
                        (reason or outcome)[:200],
                    )
            elif self._settings.strict_place_unconfirmed_kill:
                # 2026-05-14 BUG-024 — CRITICAL CONNECTIVITY FAILURE.
                #
                # ex_oid is None AND outcome is NEITHER
                # exchange_rejected NOR transport_rejected — i.e.
                # ``unconfirmed``. This is the silent-failure path:
                # the bot sent a place, the venue acknowledged
                # neither acceptance nor rejection in a way the
                # parser could classify. The order may be alive on
                # the venue (will silently fill into our position),
                # or it may have been swallowed entirely; we have no
                # way to tell from this response. Per the SUI
                # session evidence (294-805 phantom rows per day
                # across 2026-05-12/13/14, every one with
                # ``ts_ack=NULL`` and ``cancel_reason=gone_on_exchange``
                # left as bot-side cleanup of a bot that had no idea
                # what happened to its own order), this is a
                # systematic state-machine failure, not a transient
                # OKX glitch.
                #
                # The previous behaviour was to silently leave the
                # order in SENT and let reconcile clean it up later.
                # That is incorrect by operator decree: missing
                # ack/reject is NOT benign — it's a critical
                # connectivity failure. Force the bot down, cancel
                # any in-flight orders, flatten (if safe), push to
                # Telegram, mark the dashboard KILLED so the
                # operator sees it within seconds.
                #
                # Gated by ``STRICT_PLACE_UNCONFIRMED_KILL`` (default
                # True). Set False ONLY on venues with a proven
                # reconcile-by-cloid recovery path — HL's
                # ``waitingForFill``, GRVT's ``order_id: 0`` PENDING
                # states are legitimately handled by reconcile and
                # should NOT trigger this kill. OKX has no such
                # recovery path: every observed unconfirmed there
                # turned into a phantom row.
                raw_repr = self._format_raw_place_response_for_log(resp)
                payload: dict[str, Any] = {
                    "side": wo.side.value,
                    "symbol": wo.symbol,
                    "price": f"{wo.price:.12g}",
                    "size": f"{wo.size:.12g}",
                    "client_order_id": wo.client_order_id,
                    "order_id_local": wo.order_id_local,
                    "outcome": outcome,
                    "reason": (reason or "")[:1200],
                    "raw_response_truncated": raw_repr,
                }
                logger.critical(
                    "place_response_unconfirmed_critical "
                    "side=%s symbol=%s price=%s size=%s cloid=%s "
                    "outcome=%s reason=%s raw=%s",
                    wo.side.value,
                    wo.symbol,
                    f"{wo.price:.12g}",
                    f"{wo.size:.12g}",
                    wo.client_order_id,
                    outcome,
                    (reason or "")[:400],
                    raw_repr,
                )
                self._state.place_unconfirmed_critical_total += 1
                try:
                    self._storage.insert_bot_event(
                        self._clock.now_utc().isoformat(),
                        EventSeverity.CRITICAL.value,
                        "place_response_unconfirmed",
                        f"unconfirmed place response — venue gave neither "
                        f"clean accept nor reject for side={wo.side.value} "
                        f"price={wo.price} size={wo.size} cloid="
                        f"{wo.client_order_id}; order may be live on venue.",
                        payload,
                    )
                except Exception:
                    logger.exception("place_unconfirmed_bot_event_failed")
                # Transition the order to REJECTED so it doesn't sit
                # in SENT forever. The KILL flow below will cancel-all
                # which catches anything live on the venue under this
                # cloid. Reason carries the diagnostic.
                transition(
                    wo,
                    OrderStatus.REJECTED,
                    f"unconfirmed_critical:{(reason or outcome)[:200]}",
                )
                # Trigger the full Bot.kill flow: cancel-all, flatten
                # (if safe), CRITICAL Telegram push, dashboard kill
                # chip, auto-restart eligibility. If the callback
                # isn't wired (test contexts), at minimum mark state
                # killed so the trading loop halts on next tick.
                if self._request_kill_fn is not None:
                    try:
                        self._request_kill_fn(
                            "place_response_unconfirmed", payload
                        )
                    except Exception:
                        logger.exception(
                            "request_kill_fn_raised_on_unconfirmed"
                        )
                else:
                    with self._state._lock:
                        self._state.killed = True
                        self._state.kill_reason = (
                            "place_response_unconfirmed"
                        )
            self.persist(wo)
            if wo.status == OrderStatus.REJECTED:
                self._release_working_slot_if_matches(wo)
                try:
                    self._state.order_trace.record_terminal(
                        client_order_id=wo.client_order_id or "",
                        status=OrderStatus.REJECTED.value,
                        reason=(reason or outcome)[:400],
                        source="place_response",
                    )
                except Exception:
                    logger.exception("order_trace_terminal_failed")
            return

        wo.order_id_exchange = ex_oid
        self._clear_sent_ambiguous_polls(wo)
        ack_t = time.perf_counter()
        if tr is not None:
            tr.exchange_ack_perf = ack_t
            tr.local_order_open_perf = ack_t
        transition(wo, OrderStatus.ACKED)
        self.persist(wo)
        # v1.4.68 Phase 1A: drain pre-arrived WS events (single-place
        # path). Mirrors the batch path's drain in
        # ``_execute_place_batch_intents``.
        try:
            self._drain_pending_ws_events_for_cloid(wo.client_order_id)
        except Exception:
            logger.exception(
                "drain_pending_ws_events_failed_single_cloid=%s",
                (wo.client_order_id or "")[:24],
            )
        # 1.3.86: flush any cancel that was deferred while the order
        # was in flight. The defer guard ensured we never sent a
        # cancel-by-cloid that could race the place at OKX; now that
        # the order is ACKED, the cancel can fire safely against a
        # real exchange_oid. The trigger_reason was stamped at the
        # original cancel call, so we re-enter the dispatch path
        # without losing attribution.
        if wo.cancel_pending_after_ack:
            wo.cancel_pending_after_ack = False
            self.persist(wo)
            logger.info(
                "cancel_flushing_post_ack side=%s oid=%s cloid=%s trigger=%s",
                wo.side.value,
                wo.order_id_exchange,
                (wo.client_order_id[:18] + "...")
                if wo.client_order_id
                else None,
                wo.cancel_trigger_reason or "(unknown)",
            )
            self._enqueue_cancel_quote_path(
                wo, trigger_reason=wo.cancel_trigger_reason
            )
        # todo-006 fill-age bucketing: stamp the ack ts into the
        # aggregator's small LRU keyed by ex_oid. The fill side reads
        # this back to compute quote_age_ms when a fill lands. See
        # app/fill_bucket_metrics.py.
        try:
            if wo.ts_ack is not None:
                self._state.fill_buckets.note_ack(
                    order_id_exchange=ex_oid,
                    ts_ack=wo.ts_ack,
                )
        except Exception:
            logger.exception("fill_buckets_note_ack_failed")
        self._state.note_first_place_latency()
        self._maybe_emit_latency_guardrails(quote_cycle_id=quote_cycle_id)
        self._publish_outbound_trace_metrics(tr)
        if tr is not None:
            with self._state._lock:
                qd = self._state.quote_decision_perf_counter
            if qd is not None and tr.transport_send_perf > 0:
                with self._state._lock:
                    self._state.outbound_quote_cycle_to_first_transport_send_ms = float(
                        max(0.0, (tr.transport_send_perf - qd) * 1000.0)
                    )
                    self._state.outbound_quote_cycle_to_first_ack_ms = float(
                        max(0.0, (ack_t - qd) * 1000.0)
                    )

    def _submit_passive_order_verbatim(
        self,
        side: Side,
        *,
        price: float,
        size: float,
        quote_cycle_id: str,
        reduce_only: bool = False,
    ) -> Optional[WorkingOrder]:
        """Synchronous submit (manual / tests). Quote loop uses ``_stage_place_order_local`` + queue."""
        wo = self._stage_place_order_local(
            side,
            price=price,
            size=size,
            quote_cycle_id=quote_cycle_id,
            reduce_only=reduce_only,
        )
        if wo is None:
            return None
        self._complete_place_http(
            wo, quote_cycle_id=quote_cycle_id, intent_seq=wo.transport_intent_seq
        )
        return wo

    def place_passive_order_manual_only(
        self,
        side: Side,
        price: float,
        size: float,
        quote_cycle_id: str,
        *,
        reduce_only: bool = False,
    ) -> Optional[WorkingOrder]:
        """Manual, tests, and diagnostics only — not the runtime quote-refresh path.

        Applies ``normalize_order_pair`` and min-notional / min-quote gates, then calls
        ``_submit_passive_order_verbatim``. Active quoting must use ``QuoteEngine.build_quotes``
        and submit its output verbatim; do not route engine orders through this method.
        """
        if not self._client.has_write_access():
            return None
        with self._state._lock:
            # v1.4.194: migrated off the deprecated property shims.
            ex = self._state.get_working_order(side, 0)
        if ex and ex.status in (
            OrderStatus.CANCEL_PENDING,
            OrderStatus.SENT,
            OrderStatus.ACKED,
            OrderStatus.PARTIAL,
        ):
            logger.warning(
                "place_passive_order_manual_only refused side=%s existing_status=%s exchange_oid=%s",
                side.value,
                ex.status.value,
                ex.order_id_exchange,
            )
            return None
        if size <= 0 or price <= 0:
            return None
        # ----------------------------------------------------------
        # HARD MAXIMUM SIZE CHECK — defense in depth
        # ----------------------------------------------------------
        # Caller is expected to have clipped the size already; this
        # is a last-resort gate against caller bugs that bypass the
        # QuoteEngine clipping (e.g. the 2026-05-06 soft-flatten
        # incident where the worker passed |position_qty| directly
        # without any clip and produced 2672-SUI orders against a
        # $20 max-position-notional-usd cap).
        max_order_notional = float(
            self._settings.max_order_notional_usd or 0.0
        )
        hard_mult = float(
            getattr(
                self._settings,
                "max_order_notional_hard_multiplier",
                2.0,
            )
            or 2.0
        )
        hard_cap = max_order_notional * hard_mult
        notional_usd = float(size) * float(price)
        if max_order_notional > 0 and notional_usd > hard_cap + 1e-9:
            logger.critical(
                "place_passive_order_manual_only_REFUSED_OVERSIZED "
                "side=%s price=%s size=%s notional_usd=%.4f "
                "hard_cap=%.4f (max_order_notional=%.4f × %.2fx) "
                "quote_cycle_id=%s",
                side.value,
                price,
                size,
                notional_usd,
                hard_cap,
                max_order_notional,
                hard_mult,
                quote_cycle_id,
            )
            return None
        max_abs_pos = float(self._settings.max_abs_position or 0.0)
        if max_abs_pos > 0 and float(size) > max_abs_pos + 1e-9:
            logger.critical(
                "place_passive_order_manual_only_REFUSED_OVER_MAX_ABS_POSITION "
                "side=%s size=%s max_abs_position=%.4f quote_cycle_id=%s",
                side.value,
                size,
                max_abs_pos,
                quote_cycle_id,
            )
            return None
        if self._min_notional_passive_block.get(side, False):
            logger.debug(
                "place_passive_order_manual_only_suppressed side=%s reason=min_notional_passive_block_active",
                side.value,
            )
            return None
        sp = self._client.symbol_spec
        raw_px, raw_sz = price, size
        normed, rej = normalize_order_pair(sp, raw_px, raw_sz)
        if normed is None:
            if rejection_is_below_min_notional_usd(rej):
                with self._state._lock:
                    pn = float(self._state.position.position_notional)
                    pq = float(self._state.position.position_qty)
                mn_usd = float(sp.min_notional_usd)
                dust_th = float(self._settings.dust_position_notional_usd)
                is_dust = (
                    abs(pq) > _POSITION_EPS
                    and pn > _POSITION_EPS
                    and pn + 1e-12 < dust_th
                )
                if not is_dust:
                    self._min_notional_passive_block[side] = True
                logger.info(
                    "order_suppressed_below_min_notional_usd side=%s raw_price=%s raw_size=%s "
                    "rejection_reason=%s",
                    side.value,
                    raw_px,
                    raw_sz,
                    rej,
                )
            return None
        px, sz = normed
        mq = float(self._settings.min_quote_notional_usd)
        ntn = float(px) * float(sz)
        if ntn + 1e-9 < mq:
            self._min_notional_passive_block[side] = True
            logger.info(
                "quote_side_suppressed_below_min_quote_notional side=%s normalized_px=%s normalized_sz=%s "
                "normalized_notional_usd=%s min_quote_notional_usd=%s quote_cycle_id=%s",
                side.value,
                px,
                sz,
                ntn,
                mq,
                quote_cycle_id,
            )
            return None
        result = self._submit_passive_order_verbatim(
            side,
            price=float(px),
            size=float(sz),
            quote_cycle_id=quote_cycle_id,
            reduce_only=reduce_only,
        )
        # Auto-clear the min-notional latch on successful placement
        # for this side. The latch is set when a prior placement
        # rejected for being below the venue / local min-notional;
        # a subsequent successful placement at a viable size proves
        # the latching condition has cleared (price moved, settings
        # changed, residual grew, etc.). Without this clear path,
        # one bad SF placement attempt could suppress the side for
        # the rest of the process — the original Codex review
        # 2026-05-08 finding (HIGH-2 in plans/codex-review-20260507.md).
        if result is not None:
            self._min_notional_passive_block[side] = False
        return result

    def clear_min_notional_passive_block(
        self, side: Optional[Side] = None
    ) -> None:
        """Public clear path for the per-side ``_min_notional_passive_block``
        latch. ``side=None`` clears both sides; otherwise only the
        specified side.

        Callers:
          * ``Bot._enter_soft_flatten`` / ``Bot._exit_soft_flatten``
            — a fresh SF episode shouldn't be blocked by a stale
            latch from the previous one.
          * Any path that has positive evidence the latching
            condition has cleared (config change, position growth,
            etc.).

        Idempotent — clearing an already-clear side is a no-op.
        """
        if side is None:
            self._min_notional_passive_block[Side.BUY] = False
            self._min_notional_passive_block[Side.SELL] = False
        else:
            self._min_notional_passive_block[side] = False

    # --------------------------------------------------------------
    # v1.4.56 wedge-elimination Phase 2: risk-action state machine
    # --------------------------------------------------------------
    #
    # The state machine wraps ``cancel_resting_for_risk`` so the
    # cancel-all iteration fires on TRANSITIONS into the cancelling
    # cohort (KILL / CANCEL_ALL / FLATTEN), not on every TICK that
    # happens to land on one of those actions. See
    # ``app/enums.py:RiskExecState`` for the state-transition diagram.

    _RISK_CANCELLING_ACTIONS = (
        RiskAction.KILL,
        RiskAction.CANCEL_ALL,
        RiskAction.FLATTEN,
    )

    def _transition_risk_exec_state(
        self,
        new_state: RiskExecState,
        *,
        reason: str,
        risk_action: Optional[RiskAction],
    ) -> None:
        """Move the risk-exec state machine to ``new_state``. Logs the
        transition (with previous state + reason) and bumps the
        transition counter for the postmortem.
        """
        if new_state == self._risk_exec_state:
            return
        prev = self._risk_exec_state
        self._risk_exec_state = new_state
        self._risk_exec_state_entered_mono = self._clock.monotonic()
        key = f"{prev.value}->{new_state.value}"
        self._risk_exec_state_transition_counts[key] = (
            self._risk_exec_state_transition_counts.get(key, 0) + 1
        )
        log_extra(
            logger,
            logging.INFO,
            "risk_exec_state_transition",
            {
                "prev": prev.value,
                "next": new_state.value,
                "reason": reason,
                "risk_action": (
                    risk_action.value if risk_action is not None else None
                ),
            },
        )

    def _local_has_cancellable_wos(self) -> bool:
        """True iff at least one WO in local state is not yet terminal
        and not yet in CANCEL_PENDING — i.e., the dispatcher still has
        work to do for the cancel-all that just fired.

        v1.4.67 wedge fix — snapshot ``v1.4.66-260518-192744`` caught a
        52-s silent wedge where the risk state machine sat in
        SUPPRESSED forever because a CANCEL_PENDING from a previous
        session (oid 3577421873029259264) and a hydrated
        CANCEL_PENDING (oid 3577425853826408448) never reached
        terminal status. Risk action was ALLOW for 2041 of 2491
        suppressed ticks — the bot was fine, the state machine was
        stuck.

        Two categories of "stuck non-terminal WOs" must NOT block
        SUPPRESSED → NORMAL:

          * ``DESYNC`` — the bot has already given up on this order
            (issued the cancels it could). No further dispatcher work
            is pending; the WO is just a tombstone in state.
          * ``CANCEL_PENDING`` older than
            ``cancel_pending_unresolved_timeout_seconds`` — the cancel
            ack never came. The cancel-pending watchdog only tracks
            inside-rung slots (``state.working_bid`` / ``working_ask``);
            orphan-slot or hydrated CANCEL_PENDING WOs slip through
            and block the state machine. Skip them once they've aged
            past the configured timeout.

        Without this, the bot has a permanent SUPPRESSED wedge whenever
        a venue cancel ack is lost or a hydrate finds a non-trivial
        leftover.
        """
        try:
            cancel_pending_timeout_s = float(
                getattr(self._settings, "cancel_pending_unresolved_timeout_seconds", 0.0) or 0.0
            )
        except Exception:
            cancel_pending_timeout_s = 0.0
        with self._state._lock:
            for wo in self._state.all_working_orders():
                st = wo.status
                # Hard-terminal: nothing more to do.
                if st in (
                    OrderStatus.CANCELED,
                    OrderStatus.FILLED,
                    OrderStatus.REJECTED,
                    OrderStatus.DESYNC,  # v1.4.67: given-up state, treat as terminal here
                ):
                    continue
                # Stale CANCEL_PENDING: the cancel ack never came.
                # If the WO has been in CANCEL_PENDING longer than the
                # configured timeout, the state machine should NOT keep
                # waiting on it — the dispatcher has no more work to do
                # for this WO from a risk-cancel-all perspective.
                #
                # ``ts_cancel_requested`` is set the moment the cancel
                # is dispatched (see ``WorkingOrder`` docs). For
                # hydrated orphans where the cancel happened externally
                # to this process, the field may be None — in that case
                # fall back to ``ts_created`` (when the WO appeared in
                # local state).
                if st == OrderStatus.CANCEL_PENDING and cancel_pending_timeout_s > 0.0:
                    cp_anchor = getattr(wo, "ts_cancel_requested", None) or getattr(wo, "ts_created", None)
                    if cp_anchor is not None:
                        try:
                            age_s = max(0.0, (self._clock.now_utc() - cp_anchor).total_seconds())
                        except Exception:
                            age_s = 0.0
                        if age_s > cancel_pending_timeout_s:
                            # Stale beyond the watchdog horizon — treat
                            # as terminal for state-machine purposes.
                            continue
                # Any other non-terminal status: the dispatcher still
                # has work to do.
                return True
        return False

    def cancel_resting_for_risk(self, risk_action: RiskAction) -> bool:
        """Risk-driven cancel-all entrypoint, edge-triggered state machine.

        Returns ``True`` when quoting should stop for this tick. The
        return contract is unchanged from v1.4.55, but the IMPLEMENTATION
        is now a state machine:

          * NORMAL + risk in {KILL,CANCEL_ALL,FLATTEN}
              → transition NORMAL → CANCELLING
              → fire ``cancel_all_orders_for_symbol`` ONCE
              → return True
          * NORMAL + risk in other actions
              → no transition
              → return False
          * CANCELLING + risk in {KILL,CANCEL_ALL,FLATTEN}
              → no transition (already cancelling)
              → return True (quoting still suppressed; no new cancel
                fires this tick — the dispatcher has the work)
          * CANCELLING + risk returns to allowable
              → transition CANCELLING → NORMAL (immediate when no WOs
                are pending) or CANCELLING → SUPPRESSED → NORMAL when
                the dispatcher confirms terminal status
              → return False (and quoting can resume next tick)
          * CANCELLING + local WOs all reached terminal status
              → transition CANCELLING → SUPPRESSED
              → return True (still suppressed by risk)
          * SUPPRESSED + risk in {KILL,CANCEL_ALL,FLATTEN}
              → no transition, return True
          * SUPPRESSED + risk returns to allowable
              → transition SUPPRESSED → NORMAL
              → return False

        Re-entering CANCELLING from a non-NORMAL state (e.g.,
        SUPPRESSED → CANCELLING because a new exchange order appeared
        between ticks) re-fires cancel-all once, which is safe by
        Phase 1's idempotency.

        Snapshot v1.4.53-260518-162051 caught 319 cancel-all firings
        in a 3-s desync window. With this state machine that becomes
        1 firing per NORMAL→CANCELLING transition — typically 1 per
        desync episode. ~300× reduction in venue rate-limit budget.
        """
        self._risk_exec_state_last_risk_action = risk_action
        is_cancelling_action = risk_action in self._RISK_CANCELLING_ACTIONS
        cur = self._risk_exec_state

        if cur == RiskExecState.NORMAL:
            if is_cancelling_action:
                self._transition_risk_exec_state(
                    RiskExecState.CANCELLING,
                    reason="risk_action_entered_cancelling_cohort",
                    risk_action=risk_action,
                )
                self.cancel_all_orders_for_symbol()
                # Note: after cancel_all returns, if local state has
                # ZERO cancellable WOs (e.g. we were already empty),
                # transition straight to SUPPRESSED so the next tick
                # doesn't loop. Phase 1's cancel-all leaves no work
                # pending in that case.
                if not self._local_has_cancellable_wos():
                    self._transition_risk_exec_state(
                        RiskExecState.SUPPRESSED,
                        reason="cancels_completed_immediate",
                        risk_action=risk_action,
                    )
                return True
            return False

        if cur == RiskExecState.CANCELLING:
            if is_cancelling_action:
                # Stay in CANCELLING — the dispatcher's CANCEL_PENDING
                # tombstone makes any re-issued cancel a no-op, but we
                # don't even attempt it. Just observe: have all WOs
                # reached terminal status yet? If so, transition to
                # SUPPRESSED.
                if not self._local_has_cancellable_wos():
                    self._transition_risk_exec_state(
                        RiskExecState.SUPPRESSED,
                        reason="cancels_completed",
                        risk_action=risk_action,
                    )
                return True
            # Risk returned to non-cancelling — leave the cancelling cohort.
            # If there are still WOs pending (mid-cancel), go to SUPPRESSED
            # briefly; once they clear, the next tick goes to NORMAL.
            if self._local_has_cancellable_wos():
                self._transition_risk_exec_state(
                    RiskExecState.SUPPRESSED,
                    reason="risk_cleared_mid_cancel",
                    risk_action=risk_action,
                )
                # Quoting can NOT resume yet — we still have orders
                # in flight on the wire. Caller treats this as suppress.
                return True
            self._transition_risk_exec_state(
                RiskExecState.NORMAL,
                reason="risk_cleared_no_pending",
                risk_action=risk_action,
            )
            return False

        # SUPPRESSED state
        if is_cancelling_action:
            # Re-arm if local state shows orders again (e.g. an orphan
            # got hydrated). Phase 1's cancel-all is idempotent.
            if self._local_has_cancellable_wos():
                self._transition_risk_exec_state(
                    RiskExecState.CANCELLING,
                    reason="re_armed_for_new_orders",
                    risk_action=risk_action,
                )
                self.cancel_all_orders_for_symbol()
            return True
        # Risk cleared. If WOs are gone, go NORMAL. Otherwise stay
        # SUPPRESSED until they clear (next tick will retry).
        if self._local_has_cancellable_wos():
            return True
        self._transition_risk_exec_state(
            RiskExecState.NORMAL,
            reason="risk_cleared_and_no_pending",
            risk_action=risk_action,
        )
        return False

    def _clip_entry_sizes(
        self,
        position_qty: float,
        bid_sz: float,
        ask_sz: float,
        *,
        resting_bid_sz: float = 0.0,
        resting_ask_sz: float = 0.0,
        bid_price: float = 0.0,
        ask_price: float = 0.0,
    ) -> tuple[float, float]:
        """Cap sizes so a full immediate fill cannot push position past
        either of the configured caps:

        * ``MAX_ABS_POSITION`` (base units; e.g. 25 SUI)
        * ``MAX_POSITION_NOTIONAL_USD`` (USD; e.g. $20)

        Both caps applied; the tighter wins. The USD cap requires
        prices and is skipped if either price is 0 or the cap is 0.
        Mirrors ``QuoteEngine._clip_entry_sizes`` -- both sides need
        identical math for the QuoteEngine output to round-trip
        through the OrderManager unchanged.
        """
        mx = self._settings.max_abs_position
        if mx <= 0:
            return 0.0, 0.0
        max_buy = max(0.0, mx - position_qty - resting_bid_sz - _POSITION_EPS)
        max_sell = max(0.0, position_qty + mx - resting_ask_sz - _POSITION_EPS)
        max_pos_usd = float(self._settings.max_position_notional_usd or 0.0)
        if max_pos_usd > 0:
            if bid_price > 0:
                allowed_long_base = max_pos_usd / bid_price
                max_buy_usd = max(
                    0.0,
                    allowed_long_base - position_qty - resting_bid_sz - _POSITION_EPS,
                )
                max_buy = min(max_buy, max_buy_usd)
            if ask_price > 0:
                allowed_short_base = max_pos_usd / ask_price
                max_sell_usd = max(
                    0.0,
                    position_qty + allowed_short_base - resting_ask_sz - _POSITION_EPS,
                )
                max_sell = min(max_sell, max_sell_usd)
        return min(bid_sz, max_buy), min(ask_sz, max_sell)

    def _cancel_http_transport(
        self,
        wo: WorkingOrder,
        *,
        force_cloid: bool = False,
    ) -> bool:
        """Synchronous cancel HTTP (reconcile, cancel_all, transport worker).

        ``force_cloid=True`` bypasses the oid-preferred path and uses
        cancel-by-cloid regardless of whether an exchange oid is bound. This
        is used by the cancel-pending retry escalation: GRVT has been
        observed to return ``{"result": {"ack": true}}`` on cancel-by-oid
        calls yet leave the order alive on the book (confirmed via repeated
        ``open_orders`` fetches — see ``code_reports/cancel_retry_escalation.md``).
        The cloid-cancel path goes through a different GRVT handler and
        carries ``time_to_live_ms=5000``, which gives the matching engine a
        different resolution window for the same logical order.
        """
        have_oid = bool(wo.order_id_exchange)
        use_cloid = force_cloid or not have_oid
        # 1.4.0 cancel-prio Phase 0.5: stamp t2 immediately before the
        # HTTP cancel leaves the process. Used together with
        # ``ts_cancel_requested`` (t1) and ``ts_cancel_acked`` (t3) to
        # decompose cancel latency into (decision→send) and (send→ack)
        # legs. Missing ~0.2 ms of OKX signing time (which happens
        # inside the client call) — acceptable for diagnostic
        # accuracy. NULL on legacy rows + on cancels where this code
        # path didn't execute (e.g. cancel-all admin flow).
        wo.ts_cancel_sent = self._clock.now_utc()
        try:
            if use_cloid:
                if not wo.client_order_id:
                    logger.warning(
                        "cancel_order missing cloid side=%s force_cloid=%s",
                        wo.side.value,
                        force_cloid,
                    )
                    return False
                resp = self._client.cancel_order_by_cloid(wo.symbol, wo.client_order_id)
            else:
                resp = self._client.cancel_order(wo.symbol, int(wo.order_id_exchange))
        except Exception:
            logger.exception("cancel failed")
            self._state.bump_execution_errors("cancel_http_transport_exception")
            return False
        kind, detail = self._interpret_cancel_response(resp)
        # 1.4.0 cancel-prio Phase 0.5: stamp t3 (HTTP-ack wall clock)
        # only when the response interprets as success. Transport
        # errors + benign_missing have no real ack we'd want in the
        # latency aggregator — they're either order-already-gone races
        # or network failures, both noise in the cancel-RTT signal.
        # The aggregator naturally filters by ``ts_cancel_acked IS NOT
        # NULL`` so this single condition does both stamping + filter.
        if kind == "success":
            wo.ts_cancel_acked = self._clock.now_utc()
            # 1.4.0 cancel-prio Phase 0.5: feed the RTT tracker with
            # ``op="cancel"``. The tracker already supported this
            # discriminator (since v1.x) but nobody was ingesting
            # cancel samples — only places. The tracker's filtered
            # summary surfaces on heartbeat/live_stats as
            # ``cancel_submit_rtt_ms`` (sibling of the existing
            # ``order_submit_rtt_ms`` block).
            try:
                rtt_ms = (
                    (wo.ts_cancel_acked - wo.ts_cancel_sent).total_seconds()
                    * 1000.0
                )
                if rtt_ms >= 0 and wo.ts_cancel_sent is not None:
                    self._order_rtt_tracker.ingest(
                        rtt_ms=rtt_ms, op="cancel", outcome="accepted"
                    )
            except Exception:
                # Defensive: never let a metrics path break a real
                # cancel. A failed ingest is a missed sample, nothing
                # worse.
                logger.exception("cancel_rtt_ingest_failed")
            # 1.3.121: trust the venue sCode 0 success as terminal
            # proof. When enabled, immediately transition the WO to
            # CANCELED locally instead of waiting for the inbound
            # user-data WS event (which can lag 90+ seconds on OKX
            # colo). The user-data WS handler's oid-match path is
            # idempotent — when it eventually arrives, ``working_bid``
            # / ``working_ask`` is already None, so the handler
            # silently returns. See settings docstring for the
            # tiny-race-window caveat.
            if bool(
                getattr(
                    self._settings,
                    "cancel_trust_trade_ws_success_terminal",
                    False,
                )
            ):
                transition(wo, OrderStatus.CANCELED, "trade_ws_sCode_0")
                self.persist(wo)
                # 1.3.130 multi-rung Phase 2: clear the specific per-rung
                # slot. wo.level_idx identifies which rung — the inside
                # rung (0) for the legacy N=1 path, outer rungs for N>1.
                wo_level_idx = int(getattr(wo, "level_idx", 0) or 0)
                with self._state._lock:
                    cur = self._state.get_working_order(
                        wo.side, wo_level_idx
                    )
                    if cur is not None and cur.order_id_local == wo.order_id_local:
                        self._state.set_working_order(
                            wo.side, wo_level_idx, None
                        )
                self._clear_side_unresolved(
                    wo.side, reason="trade_ws_sCode_0_terminal"
                )
        # 1.3.82: stamp the cancel-response classification on the WO so
        # the dashboard's Connectivity tab + postmortem can attribute
        # every closed order to the exact HTTP outcome of its cancel
        # attempt. ``kind`` is one of: success / benign_missing /
        # transport / error. Persisted on the orders row.
        wo.cancel_response_outcome = (kind or None)
        # 1.3.83: stamp venue-side detail too. For ``success`` the
        # interpreter returns detail="" so leave the field NULL; for
        # benign_missing / error / transport the detail carries the
        # OKX sCode + sMsg which the dashboard groups on.
        if detail:
            wo.cancel_response_detail = detail[:400]
        # 1.4.6 sticky cancel-rejection recorder.
        if kind and kind != "success":
            try:
                self._state.record_cancel_rejection(
                    outcome=kind,
                    detail=detail or "",
                    is_benign=_is_benign_cancel_outcome(kind, detail),
                    ts_iso=self._clock.now_utc().isoformat(),
                )
            except Exception:
                logger.exception("record_cancel_rejection_failed")
        # 1.4.6: per-outcome cumulative counter for cancels.
        try:
            self._state.record_cancel_outcome(
                outcome=kind or "unknown",
                latency_ms=None,  # cancel RTT tracked separately via order_rtt_tracker
            )
        except Exception:
            logger.exception("record_cancel_outcome_failed")
        if kind == "benign_missing":
            # 1.3.120 narrowed: now ONLY 51402 (true cancel-race-lost-
            # to-fill — the order matched, position changed via the
            # user-data WS fill event, our cancel arrived too late).
            # Count the race, log at INFO. The 51400/51401/51503
            # codes that USED to live here are now ``unexpected_gone``.
            self._state.cancel_race_lost_to_fill_total += 1
            logger.info(
                "cancel_benign_missing side=%s exchange_oid=%s cloid=%s detail=%s",
                wo.side.value,
                wo.order_id_exchange,
                (wo.client_order_id[:18] + "...") if wo.client_order_id else None,
                detail[:300] if detail else "",
            )
        elif kind == "unexpected_gone":
            # 1.3.120: order is gone but NOT via fill (OKX sCode
            # 51400 / 51401 / 51503). Could be a stale-state retry
            # race (we already cancelled and OKX is acknowledging
            # again), venue admin / risk action, or a real state-
            # desync bug. Operator should investigate WHY this
            # counter climbs.
            #
            # The WO transitions out of CANCEL_PENDING (we return
            # True) because the order IS gone per OKX — but the log
            # at WARNING + dedicated counter make the anomaly
            # operator-visible. By default (CANCEL_UNEXPECTED_GONE_
            # STRICT_MODE=false) we do NOT bump execution_errors —
            # cleanup-cancel-all races would otherwise self-kill on
            # sustained occurrence (incident 2026-05-05). Operators
            # wanting maximum alerting flip the strict-mode flag.
            #
            # v1.4.41 BUG-025 Phase 3 fix: REAP the local WO. Same
            # rationale as the batch path — see the comment in
            # ``_execute_cancel_batch_intents`` for the captured-wedge
            # context.
            self._state.cancel_unexpected_gone_total += 1
            strict_mode = bool(
                getattr(
                    self._settings,
                    "cancel_unexpected_gone_strict_mode",
                    False,
                )
            )
            logger.warning(
                "cancel_unexpected_gone side=%s exchange_oid=%s "
                "cloid=%s strict=%s detail=%s",
                wo.side.value,
                wo.order_id_exchange,
                (wo.client_order_id[:18] + "...") if wo.client_order_id else None,
                strict_mode,
                detail[:400] if detail else "",
            )
            if strict_mode:
                self._state.bump_execution_errors(
                    "cancel_http_unexpected_gone_strict"
                )
            self._reap_wo_after_cancel_unexpected_gone(
                wo, source="single", detail=detail or ""
            )
        elif kind == "transport":
            # 1.3.119: transport classification is "venue acknowledged
            # but the response was ambiguous / misleading" — auth,
            # rate-limit, or the OKX-colo cancel-WS sCode 50014 quirk.
            # The cancel-pending watchdog will retry; the local WO
            # stays CANCEL_PENDING so we don't claim a terminal state
            # we can't prove. CRITICAL: do NOT bump execution_errors
            # here — transport is by definition transient and the
            # retry loop is the correct recovery. Bumping the
            # counter would self-kill the bot on a chatty venue.
            logger.warning(
                "cancel_transport_retry side=%s detail=%s",
                wo.side.value,
                detail[:400] if detail else "",
            )
            return False
        elif kind != "success":
            logger.warning(
                "cancel_exchange_rejected side=%s kind=%s detail=%s",
                wo.side.value,
                kind,
                detail[:400] if detail else "",
            )
            self._state.bump_execution_errors("cancel_http_exchange_reject")
            return False
        self._bump_transport_counters()
        logger.info(
            "cancel_request_sent side=%s exchange_oid=%s cloid=%s (awaiting reconciliation)",
            wo.side.value,
            wo.order_id_exchange,
            (wo.client_order_id[:18] + "...") if wo.client_order_id else None,
        )
        return True

    def _maybe_cancel_on_binance_move(self) -> None:
        """Cross-venue cancel-or-AMEND trigger using Binance's
        fair-value reference.

        v1.5.46 — side-aware anchor. Computes
        ``ref_bid = binance_best_bid + basis_ewma`` and
        ``ref_ask = binance_best_ask + basis_ewma``; the BUY rung is
        checked against ``ref_bid``, the SELL rung against ``ref_ask``.
        ``fair_value = binance_mid + basis_ewma`` is still computed and
        emitted in the bot_event payload for back-compat / dashboards,
        but the gate's threshold decision uses the side-matched anchor
        only.

        Until v1.5.46 both sides compared against ``fair_value``, which
        baked the bot's natural half-target-spread offset into
        ``delta_bps`` as a structural artefact (every SELL was ~half_spread
        ABOVE fair_value by construction; the trigger fired on the artefact,
        not on real cross-venue dislocation). See the comment block in the
        body for the diagnosis from snapshot v1.5.45-260523-104811.

        For each working order in a cancelable state, checks whether the
        order's price is more than the dynamic threshold away from its
        side's reference; if yes, attempts to AMEND the order toward the
        desired quote price (when amend is viable + we have a target
        price from the last quote breakdown), else enqueues a cancel via
        the hot-path transport.

        Silent no-op when:
          - ``BINANCE_WS_ENABLED`` is false
          - No Binance bid / ask / mid yet (stream not connected or first
            message not received)
          - Binance feed is stale beyond
            ``BINANCE_WS_FAIR_VALUE_MAX_AGE_SECONDS`` (don't trust it)
          - No working orders in cancelable status
          - Basis EWMA not yet populated (would bias the reference
            by the full raw GRVT-vs-Binance premium)

        v1.4.21 (rate-limit work): until v1.4.20 this path called
        ``_enqueue_cancel_quote_path`` directly, completely bypassing
        the amend-on-reprice decision tree in ``_orchestrate``. On
        TON-USDT-SWAP that meant **84% of all venue actions** went
        through cancel-then-place even though amend was wired —
        2349 binance-triggered cancels in 87s vs only 28 amends.
        See ``plans/amend-prio.md`` post-deploy diagnosis.

        Fires only as an ADDITIVE cancel trigger; the regular reprice
        logic below still runs for cancels due to GRVT-observed moves,
        inventory-skew shifts, normal-MM band updates, etc.
        """
        if not self._settings.binance_ws_enabled:
            return
        with self._state._lock:
            b_mid = self._state.binance_mid
            b_bid = self._state.binance_best_bid
            b_ask = self._state.binance_best_ask
            basis = self._state.binance_basis_ewma
            last_ts = self._state.binance_last_message_wall_ts
            # v1.4.194: migrated off the deprecated ``working_bid`` /
            # ``working_ask`` shims (the warning was firing in
            # test_binance_cancel_on_move_relative_threshold's CI run).
            # ``get_working_order`` is the unlocked per-rung accessor —
            # safe inside the existing ``state._lock`` context.
            wb = self._state.get_working_order(Side.BUY, 0)
            wa = self._state.get_working_order(Side.SELL, 0)
        if (
            b_mid is None
            or b_bid is None
            or b_ask is None
            or basis is None
            or last_ts is None
        ):
            return
        age_s = (self._clock.now_utc() - last_ts).total_seconds()
        max_age = float(self._settings.binance_ws_fair_value_max_age_seconds)
        if age_s > max_age:
            # Stale — fall back to GRVT-only behaviour (don't cancel on
            # a feed that might be wildly out of date).
            return
        fair_value = float(b_mid) + float(basis)
        if fair_value <= 0 or not math.isfinite(fair_value):
            return
        # v1.5.46 — per-side reference price. The bot's BUY rests at the
        # target-venue bid (~half_target_spread BELOW target_mid); the
        # SELL rests at the target-venue ask (~half_target_spread ABOVE
        # target_mid). Comparing both sides against ``fair_value``
        # (binance_mid + basis) historically baked the full half-target-
        # spread into ``delta_bps`` as a structural artefact — visible on
        # snapshot v1.5.45-260523-104811 where the SELL was 9–10 bp from
        # fair_value by construction (4 bp target_half_spread + skews),
        # firing the gate at the 9-bp threshold despite zero genuine
        # cross-venue dislocation. Cause: asymmetric comparison (ask vs
        # mid).
        #
        # Fix: compare side-aware against the matching reference touch.
        #   - BUY  → binance_best_bid + basis  (anchor: where Binance bid is)
        #   - SELL → binance_best_ask + basis  (anchor: where Binance ask is)
        # The half-target-spread cancels with half-reference-spread; only
        # genuine cross-venue dislocation (residual spread differential
        # + real basis drift) shows up in ``delta_bps``. Threshold
        # semantics shift slightly — the same numeric value is now a
        # tighter, more-meaningful trigger — but the dynamic floor
        # (``max(static_floor, half_spread + buffer)``) still absorbs
        # the residual differential.
        ref_bid = float(b_bid) + float(basis)
        ref_ask = float(b_ask) + float(basis)
        if (
            ref_bid <= 0
            or ref_ask <= 0
            or not math.isfinite(ref_bid)
            or not math.isfinite(ref_ask)
        ):
            return
        # v1.4.45 — relative cross-venue threshold (Option A from
        # plans/20260518-rate-optimization.md aftermath).
        #
        # The cross-venue panic threshold MUST be wider than the bot's
        # natural quote distance (target_half_spread_bps) or every order
        # is born past the threshold and gets instant-canceled. Until
        # v1.4.44 the threshold was a static config value: when the
        # half-spread organically grew past the static threshold (vol
        # regime up, toxicity bump active, etc), the bot self-defeated —
        # confirmed 2026-05-18 in v1.4.44-260518-140038 snapshot where
        # target_half_spread=13.7 bps with static threshold=10 bps
        # → instant cancel → 30+ min of zero trading.
        #
        # v1.4.45 design: threshold = max(static_floor, half_spread +
        # buffer). Two bounds:
        #   - ``static_floor`` (BINANCE_CANCEL_ON_MOVE_BPS): minimum
        #     distance we'll let an order sit, regardless of half_spread.
        #     Stops the bot from being TOO patient in calm regimes
        #     where Binance has wandered far.
        #   - ``half_spread + buffer`` (BINANCE_CANCEL_ON_MOVE_BUFFER_BPS
        #     above the engine's current target half-spread): scales
        #     automatically with vol/toxicity, so the bot never quotes
        #     a price the very same code will cancel on placement.
        # Max() because both bounds must be satisfied — the threshold
        # is the looser of "you must be at least this far" + "you must
        # be at least this much further than my own quote distance".
        #
        # Buffer disabled (=0) OR breakdown unavailable (pre-first-tick
        # / engine returned no quote this cycle) → falls back to the
        # pre-v1.4.45 static-only behaviour (full back-compat).
        static_floor_bps = float(self._settings.binance_cancel_on_move_bps)
        buffer_bps = float(
            getattr(self._settings, "binance_cancel_on_move_buffer_bps", 0.0)
        )
        threshold_bps = static_floor_bps
        if buffer_bps > 0.0:
            bd = self._state.last_quote_breakdown
            if bd is not None:
                hs = getattr(bd, "target_half_spread_bps", None)
                if (
                    hs is not None
                    and math.isfinite(float(hs))
                    and float(hs) > 0.0
                ):
                    dynamic_bps = float(hs) + buffer_bps
                    threshold_bps = max(static_floor_bps, dynamic_bps)

        def _maybe_trigger(wo: Optional[WorkingOrder]) -> None:
            if wo is None:
                return
            # Only cancel orders that are actually resting on the book;
            # in-flight states (SENT, CANCEL_PENDING) are handled by
            # other paths.
            if wo.status not in (OrderStatus.ACKED, OrderStatus.PARTIAL):
                return
            try:
                wo_px = float(wo.price)
            except (TypeError, ValueError):
                return
            if wo_px <= 0:
                return
            # v1.5.46 — side-aware anchor (see the long comment above
            # the ``ref_bid`` / ``ref_ask`` setup). BUY → bid anchor;
            # SELL → ask anchor.
            ref_px = ref_bid if wo.side == Side.BUY else ref_ask
            delta_bps = abs(wo_px - ref_px) / ref_px * 10_000.0
            if delta_bps <= threshold_bps:
                return
            # v1.4.27: target_px computation simplified after the
            # v1.4.21 → v1.4.23 saga and the v1.4.26 deploy diagnosis.
            #
            # Earlier approaches and their failure modes:
            #   v1.4.21: target_px = breakdown.quoted_bid_px (raw).
            #     BROKE because OKX mid lags Binance → the breakdown's
            #     quoted_bid_px was always far from fair_value → the
            #     in-threshold check rejected it → cancel-route taken
            #     in 402/403 cases.
            #   v1.4.23: target_px = fair_value − (bd.mid − bd.quoted_bid_px).
            #     The intent was to extract the "intended offset" and
            #     anchor to fair. BROKE because the bid/ask offset in
            #     the breakdown carries the FULL stack of adjustments
            #     (half_spread + inventory_skew + microprice_shift +
            #     ob_imbalance_shift + flow_score_shift). When inventory
            #     is large, the offset is 30+ bps. Applying that to
            #     fair_value produces a target_px that is itself FAR
            #     from fair → re-triggers the threshold next tick.
            #     Observed in v1.4.26 snapshot: 94 amends/sec, every
            #     amend a no-op (target_px == current wo_px).
            #
            # v1.4.27 fix: target_px = fair_value ± target_half_spread_bps
            # (in price units). The breakdown's ``target_half_spread_bps``
            # is the engine's MODEL HALF-SPREAD — already capped to
            # the normal_mm market-anchor or the max-cap. It does NOT
            # include the OKX-mid-lag baggage. Applying it to Binance
            # fair_value produces a target that is ALWAYS ~half-spread
            # bps from fair — well within the cross-venue threshold by
            # construction.
            #
            # Trade-off: this loses the bot's careful skew/microprice
            # tweaks for the cross-venue amend path. The regular quote
            # loop tick (every ~500 ms, or per Binance/OKX BBO wake)
            # still runs the full _orchestrate → amend path with
            # complete skew. The cross-venue amend is a FAST-PATH
            # response to "you're far from fair, get closer NOW"; the
            # next tick's full quote build refines the price further.
            #
            # Fall-through to cancel remains for: bd is None
            # (pre-first-tick), missing target_half_spread_bps field
            # (older bot build / engine no-quote result), non-positive
            # half-spread, or a non-positive target_px.
            bd = self._state.last_quote_breakdown
            target_px: Optional[float] = None
            if bd is not None:
                hs_bps = getattr(bd, "target_half_spread_bps", None)
                if (
                    hs_bps is not None
                    and float(hs_bps) > 0
                    and math.isfinite(float(hs_bps))
                ):
                    hs_bps_f = float(hs_bps)
                    offset_price = fair_value * (hs_bps_f / 10_000.0)
                    if wo.side == Side.BUY:
                        cand = fair_value - offset_price
                    else:
                        cand = fair_value + offset_price
                    if cand > 0:
                        target_px = cand
            # v1.4.27: no-op amend skip. After grid rounding (which the
            # downstream amend pipeline does), if target_px lands within
            # ONE PRICE TICK of the current wo.price, the amend would be
            # a wire-level no-op — same px after rounding. Skip the amend
            # entirely (fall back to cancel/no-op trigger) so the bot
            # doesn't burn one venue call per Binance tick on a price
            # change too small to land on the grid. Diagnosed in the
            # v1.4.26 snapshot where the bot fired 94 amends/sec with
            # ``amend_target_px == wo.order.price`` every time.
            if target_px is not None:
                try:
                    tick_sz = float(self._client.symbol_spec.price_tick or 0.0)
                except Exception:
                    tick_sz = 0.0
                if tick_sz > 0 and abs(target_px - wo_px) < tick_sz - 1e-12:
                    # Would round to the same grid cell. Skip.
                    target_px = None
            amend_route = False
            if target_px is not None:
                from app.quote_engine import FinalQuoteOrder
                desired = FinalQuoteOrder(
                    side=wo.side,
                    price=target_px,
                    size=float(wo.size),
                )
                if self._amend_viable(wo, desired):
                    amend_route = True

            # Trigger: enqueue a cancel OR amend via the hot path.
            # Log + persist a bot_event so the trigger can be queried
            # post-hoc.
            event_kind = (
                "binance_cross_venue_amend"
                if amend_route
                else "binance_cross_venue_cancel"
            )
            payload = {
                "side": wo.side.value,
                "order_price": wo_px,
                "binance_mid": float(b_mid),
                # v1.5.46: reference-venue BBO + per-side anchor that the
                # threshold compares against. ``fair_value`` and ``delta_bps``
                # are kept for explain_moment.py back-compat but the gate
                # now decides on ``reference_price`` / ``delta_bps`` (which
                # uses ref_bid for BUY, ref_ask for SELL).
                "binance_best_bid": float(b_bid),
                "binance_best_ask": float(b_ask),
                "basis_ewma": float(basis),
                "fair_value": fair_value,
                "reference_price": ref_px,
                "delta_bps": delta_bps,
                "threshold_bps": threshold_bps,
                "binance_feed_age_s": age_s,
                "order_id_local": wo.order_id_local,
                "order_id_exchange": wo.order_id_exchange,
                "route": "amend" if amend_route else "cancel",
                "amend_target_px": target_px,
            }
            log_extra(
                logger,
                logging.INFO,
                event_kind,
                payload,
            )
            try:
                self._storage.insert_bot_event(
                    self._clock.now_utc().isoformat(),
                    EventSeverity.INFO.value,
                    event_kind,
                    event_kind,
                    payload,
                )
            except Exception:
                # DB write failures must never block the trigger —
                # the event is best-effort telemetry, the trigger
                # is the primary action.
                logger.exception(
                    "binance_cross_venue_event_db_write_failed"
                )
            # 1.3.31 / 1.4.21: bump the session counter(s) exposed
            # via live_stats for the dashboard's cross-venue card.
            # Plain ``int`` write of a session-cumulative counter —
            # no lock needed; a torn read yields a count off by one
            # at worst (see BotState comment). Cancel and amend
            # counters are tracked separately so the operator can
            # see how often the amend path won (= action savings).
            try:
                if amend_route:
                    self._state.session_cross_venue_amend_count += 1
                else:
                    self._state.session_cross_venue_cancel_count += 1
            except Exception:
                logger.exception(
                    "binance_cross_venue_counter_bump_failed"
                )
            if amend_route:
                # ``_amend_viable`` already passed; build the desired
                # FinalQuoteOrder again (cheap; same params as above).
                from app.quote_engine import FinalQuoteOrder
                self._enqueue_amend_quote_path(
                    wo,
                    FinalQuoteOrder(
                        side=wo.side,
                        price=float(target_px),  # narrowed by amend_route
                        size=float(wo.size),
                    ),
                    trigger_reason="binance_cross_venue_amend",
                )
            else:
                self._enqueue_cancel_quote_path(
                    wo, trigger_reason="binance_cross_venue"
                )

        _maybe_trigger(wb)
        _maybe_trigger(wa)

    def _maybe_cancel_on_target_venue_fast_move(self) -> None:
        """v1.4.165 Phase 4E — target-venue fast-move cancel.

        Complementary to :meth:`_maybe_cancel_on_binance_move` (which
        uses the REFERENCE venue as a leading indicator). This trigger
        watches the TARGET venue's OWN 500 ms mid-return kinematic.
        Catches fast moves that originate on the trading venue itself
        (local taker batches, local-only microstructure events,
        liquidity vacuums) — moves the reference-venue trigger may
        not see in time because the reference didn't lead.

        Exchange-agnostic naming: ``target_venue_*`` parallels the
        existing ``reference_venue_*`` settings; no exchange-specific
        identifiers appear in this method or the supporting helper.
        The legacy ``BINANCE_CANCEL_ON_MOVE_BPS`` setting name is
        exchange-coupled (queued for compat-safe rename).

        Silent no-op when:
          - ``TARGET_VENUE_CANCEL_ON_MOVE_BPS <= 0`` (default = disabled)
          - ``state.last_mid_return_500ms_bps`` is None
            (kinematics not warmed up, or degenerate market)
          - The signal magnitude is below threshold
          - No resting orders on the side the helper indicates
        """
        threshold_bps = float(
            getattr(
                self._settings,
                "target_venue_cancel_on_move_bps",
                0.0,
            )
        )
        if threshold_bps <= 0.0:
            return
        from app.fast_move_cancel import detect_target_venue_fast_move

        # v1.5.157 — selectable drift-signal window. Default 0.5 →
        # legacy ``state.last_mid_return_500ms_bps``. Operator can
        # opt into longer windows via
        # ``TARGET_VENUE_FAST_MOVE_CANCEL_DRIFT_WINDOW_SECONDS`` to
        # filter the per-tick noise that drove the bug-030 12x
        # cancel asymmetry. NB: the asymmetry is partly structural
        # (one side always cancels more in trending regimes); this
        # is a magnitude lever, not a directional fix.
        _fmc_window_s = float(
            getattr(
                self._settings,
                "target_venue_fast_move_cancel_drift_window_seconds",
                0.5,
            )
        )
        mid_return_bps: Optional[float] = None
        if _fmc_window_s <= 0.6:
            mid_return_bps = getattr(
                self._state, "last_mid_return_500ms_bps", None
            )
        else:
            _mdw = getattr(self._state, "mid_drift_windows", None)
            if _mdw is not None:
                if _fmc_window_s <= 7.5:
                    mid_return_bps = _mdw.drift_5s_bps
                elif _fmc_window_s <= 20.0:
                    mid_return_bps = _mdw.drift_10s_bps
            # Fallback chain when chosen long window is None (warmup):
            # drift_5s → 500ms → None
            if mid_return_bps is None and _mdw is not None:
                mid_return_bps = _mdw.drift_5s_bps
            if mid_return_bps is None:
                mid_return_bps = getattr(
                    self._state, "last_mid_return_500ms_bps", None
                )
        side_to_cancel = detect_target_venue_fast_move(
            mid_return_bps=mid_return_bps,
            threshold_bps=threshold_bps,
        )
        if side_to_cancel is None:
            return

        with self._state._lock:
            # v1.4.194: migrated off the deprecated property shims.
            wo = self._state.get_working_order(side_to_cancel, 0)
        if wo is None:
            return
        if wo.status not in (OrderStatus.ACKED, OrderStatus.PARTIAL):
            # In-flight states (SENT, CANCEL_PENDING) are handled by
            # other paths; we only act on resting orders.
            return

        payload = {
            "side": side_to_cancel.value,
            "order_price": float(wo.price),
            # v1.5.157 — the field name is kept legacy for back-compat
            # with downstream log parsers; the VALUE is now whatever
            # window the operator selected via
            # TARGET_VENUE_FAST_MOVE_CANCEL_DRIFT_WINDOW_SECONDS.
            "mid_return_500ms_bps": float(mid_return_bps)
            if mid_return_bps is not None
            else None,
            "drift_window_seconds": _fmc_window_s,
            "threshold_bps": threshold_bps,
            "order_id_local": wo.order_id_local,
            "order_id_exchange": wo.order_id_exchange,
            "route": "cancel",
        }
        log_extra(
            logger,
            logging.INFO,
            "target_venue_fast_move_cancel",
            payload,
        )
        try:
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.INFO.value,
                "target_venue_fast_move_cancel",
                "target_venue_fast_move_cancel",
                payload,
            )
        except Exception:
            # DB write failures must never block the trigger.
            pass
        self._enqueue_cancel_quote_path(
            wo, trigger_reason="target_venue_fast_move"
        )
        # Counter increment — useful for the snapshot dashboard so the
        # operator can verify the gate is firing on real moves.
        try:
            if side_to_cancel == Side.BUY:
                self._state.target_venue_fast_move_cancel_bid_total += 1
            else:
                self._state.target_venue_fast_move_cancel_ask_total += 1
        except Exception:
            pass

    def cancel_order(
        self, wo: WorkingOrder, *, trigger_reason: Optional[str] = None
    ) -> bool:
        """Synchronous cancel (reconcile, risk flatten, tests). Hot quote path uses enqueue.

        ``trigger_reason`` (1.3.82) identifies which code path requested
        the cancel: reprice_replace / hard_age_cap / side_suppressed /
        soft_flatten / binance_cross_venue / kill_flow / reconcile /
        startup_cancel_all / risk_flatten / other. Stamped on the WO
        and persisted; the Connectivity tab uses this to attribute
        gone_on_exchange events to specific decisions.

        1.3.86 cancel-before-ack guard: when the WO is still in SENT
        with ts_ack=None (place HTTP response not yet received), the
        cancel is PARKED — its trigger_reason and ts_cancel_requested
        are stamped, but the HTTP cancel is NOT dispatched. The
        place-response handler flushes the deferred cancel as soon as
        the ACKED transition fires. Disable via
        CANCEL_DEFER_UNTIL_ACK_ENABLED=false.
        """
        if not self._client.has_write_access():
            return False
        if not wo.order_id_exchange and not wo.client_order_id:
            return False
        # 1.4.0 cancel-prio Phase 0b: ``persist(wo)`` is deferred to
        # AFTER the HTTP cancel so the disk I/O (SQLite ``_lock`` +
        # WAL fsync, median 0.5-3 ms, p99 ~30 ms) does not serialize
        # the cancel-on-the-wire moment behind the DB write. The
        # in-memory WO already carries the CANCEL_PENDING transition
        # via ``transition()``; persist is only for restart recovery,
        # which tolerates a millisecond-scale lag. ``needs_persist``
        # carries the "we transitioned" signal across the HTTP call.
        needs_persist = False
        if wo.status != OrderStatus.CANCEL_PENDING:
            # 2026-05-13 regime-observability Phase 1: stamp the
            # cancel-request timestamp on the WorkingOrder before the
            # status transition / persist. Fill ingestion reads this
            # column to derive ``cancel_requested_before_fill`` and
            # ``ms_cancel_request_to_fill``. Set only when transitioning
            # INTO CANCEL_PENDING (not on subsequent retries of the
            # cancel for the same already-pending order).
            if wo.ts_cancel_requested is None:
                wo.ts_cancel_requested = self._clock.now_utc()
            if trigger_reason and not wo.cancel_trigger_reason:
                wo.cancel_trigger_reason = trigger_reason
            self._maybe_bump_place_cancel_race(wo)
            if self._defer_cancel_if_unacked(wo, trigger_reason):
                return True
            transition(wo, OrderStatus.CANCEL_PENDING)
            needs_persist = True
        self._mark_cancel_pending(wo.side)
        result = self._cancel_http_transport(wo)
        if needs_persist:
            self.persist(wo)
        return result

    def _defer_cancel_if_unacked(
        self, wo: WorkingOrder, trigger_reason: Optional[str]
    ) -> bool:
        """1.3.86 cancel-before-ack guard. Returns True when the
        cancel was deferred (caller should NOT dispatch); False when
        the cancel should proceed normally.

        Defer condition (all must hold):
          * feature enabled (CANCEL_DEFER_UNTIL_ACK_ENABLED)
          * status == SENT
          * ts_ack is None
          * order_id_exchange is None — the venue has NOT given us
            any oid yet (even a 0). On GRVT the synchronous accept
            response sets order_id_exchange to a placeholder (0)
            BEFORE the matching engine assigns the real id; that
            state is reconcile-by-cloid territory and cancels there
            are legitimate. The race we fix is purely the "place
            HTTP truly hasn't returned" case.

        Idempotent — re-deferring an already-deferred WO is a no-op
        on the counter / log line.
        """
        if not bool(
            getattr(self._settings, "cancel_defer_until_ack_enabled", True)
        ):
            return False
        if wo.status != OrderStatus.SENT or wo.ts_ack is not None:
            return False
        if wo.order_id_exchange is not None:
            return False
        if not wo.cancel_pending_after_ack:
            wo.cancel_pending_after_ack = True
            self._state.cancel_deferred_until_ack_total += 1
            logger.info(
                "cancel_deferred_until_ack side=%s cloid=%s trigger=%s",
                wo.side.value,
                (wo.client_order_id[:18] + "...")
                if wo.client_order_id
                else None,
                trigger_reason or wo.cancel_trigger_reason or "(unknown)",
            )
            self.persist(wo)
        return True

    def _maybe_bump_place_cancel_race(self, wo: WorkingOrder) -> None:
        """1.3.82: detect and count "place-then-immediate-cancel" races.

        Fires when a cancel is enqueued for an order whose ``ts_ack``
        is still None AND the cancel comes within 50 ms of ``ts_sent``.
        This is the exact signature of the 50 phantom-place
        gone_on_exchange events seen in snapshot 260515-095056 — the
        bot's quote-cycle or outbound dispatcher racing its own place.
        Counter increments at most once per WO. Best-effort; failure
        never blocks the cancel.
        """
        try:
            if wo.ts_ack is not None:
                return
            if wo.ts_sent is None:
                return
            now = self._clock.now_utc()
            delta_ms = (now - wo.ts_sent).total_seconds() * 1000.0
            if 0 <= delta_ms <= 50.0:
                self._state.place_cancel_race_total += 1
        except Exception:
            logger.exception("place_cancel_race_counter_failed")

    def _enqueue_cancel_quote_path(
        self, wo: WorkingOrder, *, trigger_reason: Optional[str] = None
    ) -> bool:
        """Queue cancel intent; strategy thread returns immediately (non-blocking).

        ``trigger_reason`` (1.3.82) — see ``cancel_order`` for the
        taxonomy. Defaults to None to keep call-sites that haven't
        been updated working; the dashboard will surface
        unattributed cancels under ``trigger_reason="(unknown)"``.

        1.3.86: when the WO is unacked, defer the cancel until ack
        lands (see ``_defer_cancel_if_unacked``). Returns True in
        the deferred case so callers see "cancel registered" without
        a transport-side dispatch.
        """
        if not self._client.has_write_access():
            return False
        if not wo.order_id_exchange and not wo.client_order_id:
            return False
        # 1.4.0 cancel-prio Phase 0b: ``persist(wo)`` is deferred to
        # AFTER the dispatcher enqueue so the disk I/O does not block
        # the moment the cancel is queued for transport. See companion
        # comment in ``cancel_order``. On the hot quote path the
        # speedup is small (the dispatcher enqueue itself is sub-µs),
        # but the consistency of doing this in both cancel paths
        # keeps reasoning simpler.
        needs_persist = False
        if wo.status != OrderStatus.CANCEL_PENDING:
            # 2026-05-13 regime-observability Phase 1: stamp the
            # cancel-request timestamp on the WorkingOrder before the
            # status transition / persist. See companion comment in
            # ``cancel_order``.
            if wo.ts_cancel_requested is None:
                wo.ts_cancel_requested = self._clock.now_utc()
            if trigger_reason and not wo.cancel_trigger_reason:
                wo.cancel_trigger_reason = trigger_reason
            self._maybe_bump_place_cancel_race(wo)
            if self._defer_cancel_if_unacked(wo, trigger_reason):
                return True
            transition(wo, OrderStatus.CANCEL_PENDING)
            needs_persist = True
        self._mark_cancel_pending(wo.side)
        side = wo.side
        self._cancel_intent_seq[side] += 1
        wo.cancel_transport_seq = self._cancel_intent_seq[side]
        if self._first_enqueue_perf is None:
            self._first_enqueue_perf = time.perf_counter()
        tr = self._outbound_traces.get(wo.order_id_local)
        if tr is None:
            tr = OutboundActionTrace(
                order_id_local=wo.order_id_local,
                side_value=wo.side.value,
                intent_created_perf=time.perf_counter(),
                dispatcher_enqueue_perf=time.perf_counter(),
            )
            self._outbound_traces[wo.order_id_local] = tr
        tr.cancel_intent_perf = time.perf_counter()
        self._outbound.submit_cancel(
            CancelTransportIntent(
                wo.order_id_local,
                side,
                wo.cancel_transport_seq,
                self._clock.monotonic(),
                # 1.3.130 multi-rung Phase 2: rung identifier for
                # per-(side, level_idx) dispatcher coalescing.
                level_idx=int(getattr(wo, "level_idx", 0) or 0),
            )
        )
        # v1.4.28 dashboard counter: per-intent cancel count for the
        # operator-facing FILLS|ORDERS|CANCEL|VOLUME card. Parallel
        # to ``session_place_attempt_count``.
        with self._state._lock:
            self._state.session_cancel_attempt_count += 1
        if needs_persist:
            self.persist(wo)
        self._sync_outbound_state_flags()
        return True

    def _quote_book_fresh_for_reprice(self, mkt: Optional[BestBidAsk]) -> bool:
        if mkt is None or mkt.ts_local is None:
            return False
        sa = seconds_since(mkt.ts_local)
        if sa is None:
            return False
        return float(sa) < float(self._settings.stale_data_warn_seconds)

    def _quote_reprice_log_and_event(
        self,
        event: str,
        payload: dict[str, Any],
        *,
        severity: str = EventSeverity.INFO.value,
    ) -> None:
        log_extra(logger, logging.INFO, event, {"event": event, **payload})
        self._storage.insert_bot_event(self._clock.now_utc().isoformat(), severity, event, event, payload)
        if event == "quote_reprice_required":
            with self._state._lock:
                self._state.quote_quality.note_quote_reprice_required()

    def _collect_quote_cycle_suppression_reasons(
        self, decision: QuoteDecision
    ) -> frozenset[str]:
        """Aggregate active suppression reasons from already-computed decision / build fields.

        Returns a set of normalized reason tags (``source:tag``) covering:
        - ``quoting:*`` — inventory-skew and toxicity suppressors from ``quoting.py``
          (read from ``decision.decision_reason`` — a comma-separated list).
        - ``eligibility:*`` — freshness / drift / jump gates from ``quote_eligibility.py``
          (read from ``decision.quote_eligibility_reason``).
        - ``engine:inventory_exec_bias_*`` — silent suppressor from ``quote_engine.py``
          (read from ``self._quote_exec_telemetry`` flags set by
          ``_apply_inventory_execution_bias``).

        The central-observation design avoids threading a storage/logger into
        strategy modules. Every suppression signal is already surfaced via
        ``decision_reason`` / ``quote_eligibility_reason`` / engine telemetry —
        we just classify them here.
        """
        reasons: set[str] = set()

        dr = decision.decision_reason or ""
        quoting_parts = {
            "at_max_long", "at_max_short",
            "hard_skew_long", "hard_skew_short",
            "soft_skew_long", "soft_skew_short",
            "toxic_bid", "toxic_ask",
            # todo-011: per-side post-fill replace cooldown markers.
            # Stamped in ``compute_quote_decision`` when the cooldown
            # is suppressing the just-filled side; surfaced into
            # ``suppression_reason_counts_session`` so the operator
            # can verify the gate is firing.
            "post_fill_cooldown_bid", "post_fill_cooldown_ask",
            # 2026-05-12 codex-#1 narrow: at-touch adverse pause
            # markers. Same observability machinery as above.
            "at_touch_adverse_pause_bid", "at_touch_adverse_pause_ask",
        }
        for part in (p.strip() for p in dr.split(",")):
            if part in quoting_parts:
                reasons.add(f"quoting:{part}")

        er = decision.quote_eligibility_reason or ""
        if er.startswith("freshness_hold"):
            reasons.add("eligibility:freshness_hold")
        elif er.startswith("freshness_one_sided"):
            reasons.add("eligibility:freshness_one_sided")
        elif er.startswith("drift_direction_conflict"):
            reasons.add("eligibility:drift_hold")
        elif er.startswith("drift:"):
            reasons.add("eligibility:drift_one_sided")
        elif er.startswith("jump_250ms") or er.startswith("abs_mid_return_500ms"):
            reasons.add("eligibility:jump_hold")
        elif er.startswith("freshness_or_drift_hold") or er.startswith("freshness_drift_conflict"):
            reasons.add("eligibility:freshness_drift_hold")

        t = self._quote_exec_telemetry
        if t.get("quote_engine_inventory_bias_suppressed_bid"):
            reasons.add("engine:inventory_exec_bias_bid")
        if t.get("quote_engine_inventory_bias_suppressed_ask"):
            reasons.add("engine:inventory_exec_bias_ask")

        return frozenset(reasons)

    def _record_quote_cycle_telemetry(
        self,
        decision: QuoteDecision,
        *,
        want_bid: bool,
        want_ask: bool,
        can_bid: bool,
        can_ask: bool,
        tick: float,
    ) -> None:
        t = self._quote_exec_telemetry
        nb = t.get("exec_finalize_bid_px")
        na = t.get("exec_finalize_ask_px")
        if nb is None:
            nb = t.get("exec_norm_bid_px")
        if na is None:
            na = t.get("exec_norm_ask_px")
        mid = decision.mid_price
        two_sided_effective = bool(can_bid and can_ask)
        intended_two_sided = bool(want_bid and want_ask)
        spread_bps: Optional[float] = None
        one_tick = False
        if (
            isinstance(nb, (int, float))
            and isinstance(na, (int, float))
            and math.isfinite(float(nb))
            and math.isfinite(float(na))
            and isinstance(mid, (int, float))
            and float(mid) > 0
            and float(na) > float(nb)
        ):
            spread_bps = (float(na) - float(nb)) / float(mid) * 10_000.0
            if tick > 0:
                one_tick = (float(na) - float(nb)) <= tick * 1.00001
        reasons = self._collect_quote_cycle_suppression_reasons(decision)
        with self._state._lock:
            self._state.quote_quality.record_quote_cycle(
                quoted_spread_bps=spread_bps,
                one_tick_wide=one_tick if spread_bps is not None else False,
                two_sided_effective=two_sided_effective,
                intended_two_sided=intended_two_sided,
            )
            for reason in reasons:
                self._state.quote_quality.record_suppression(reason)
        # Emit one event per newly-appeared reason (transition edge). Suppressors
        # that persist across cycles do not re-emit; this is the rate-limit.
        new_reasons = reasons - self._last_suppression_reasons
        for reason in sorted(new_reasons):
            self._quote_reprice_log_and_event(
                "quote_side_suppressed",
                {
                    "reason": reason,
                    "decision_reason": decision.decision_reason,
                    "quote_eligibility_reason": decision.quote_eligibility_reason,
                    "active_sides": decision.active_sides.value,
                    "quote_cycle_id": decision.quote_cycle_id,
                },
            )
        self._last_suppression_reasons = reasons

    def _maybe_emit_engine_no_quote_diagnostic(
        self,
        *,
        telemetry: dict[str, Any],
        decision_quote_cycle_id: str,
        position_qty: float,
        position_notional: float,
    ) -> None:
        """Emit a diagnostic event if the engine has been producing
        ``mode="no_quote"`` for a persistent streak of ticks.

        Catches silent-wedge scenarios where:
          * The dust-residual gating in ``QuoteEngine.build_quotes`` is
            unreachable for the operator's config combination, leaving
            sub-spec residuals stuck (Bug A in snapshot 260507114312).
          * A side is latched as unresolved with ``requires_confirm``
            and the auto-release timeout cannot clear it.
          * Both build_quotes sides return None for some other reason.

        The event surfaces the most-actionable telemetry (engine
        reasons, inventory bias, side_unresolved status) so the
        operator (and future automated analysis) can identify the
        cause without re-deriving it from sparse logs.
        """
        mode = str(telemetry.get("quote_engine_mode") or "")
        if mode != "no_quote":
            # v1.5.159 BUG-031 fix — capture the recovery moment when
            # the streak transitions from a real count back to 0. The
            # wedge detector reads this to skip false-positive fires
            # in the grace window where the executor is catching up
            # to the freshly-recovered engine.
            if self._engine_no_quote_streak_ticks > 0:
                self._engine_no_quote_recovery_ts_mono = (
                    self._clock.monotonic()
                )
            self._engine_no_quote_streak_ticks = 0
            return
        # v1.4.90 wedge-elimination-cleanup FU-2 — suppress the warning
        # when desync recovery is in progress. During desync_recovered/
        # desync_detected transitions the engine correctly HOLDS quotes
        # ("side_not_requested" on both sides) for the recovery window
        # (~30 s). That's the right behavior, not a silent wedge. The
        # warning's purpose is to catch "we SHOULD be quoting but aren't"
        # — desync-active explicitly means "we shouldn't quote yet".
        #
        # Evidence: snapshot v1.4.85-260519-104553 fired this WARNING at
        # 06:26:44 during legitimate desync recovery from the FU-1
        # cancel-then-place drift event. False positive — the
        # reconciler did its job within seconds.
        #
        # Also reset the streak so when desync clears, a fresh count
        # starts from 0 (don't re-fire immediately based on accumulated
        # ticks during recovery).
        desync_phase = ""
        try:
            dp = getattr(self._state, "desync_phase", None)
            desync_phase = getattr(dp, "value", None) or str(dp or "")
        except Exception:
            desync_phase = ""
        if desync_phase and desync_phase.upper() != "OK":
            # v1.5.159 BUG-031 fix — same recovery capture as above.
            # Desync-recovery streak reset also constitutes a
            # "recovery" event from the wedge detector's perspective.
            if self._engine_no_quote_streak_ticks > 0:
                self._engine_no_quote_recovery_ts_mono = (
                    self._clock.monotonic()
                )
            self._engine_no_quote_streak_ticks = 0
            return
        self._engine_no_quote_streak_ticks += 1
        threshold = int(
            getattr(self._settings, "engine_no_quote_diag_streak_ticks", 20)
        )
        if threshold < 2:
            threshold = 2
        if self._engine_no_quote_streak_ticks < threshold:
            return
        now_m = self._clock.monotonic()
        relog_s = float(
            getattr(self._settings, "engine_no_quote_diag_relog_seconds", 30.0)
        )
        # First-time hit on threshold OR re-log cadence elapsed.
        last = float(self._engine_no_quote_diag_last_log_mono)
        first_hit = self._engine_no_quote_streak_ticks == threshold
        relog_due = (now_m - last) >= relog_s if last > 0.0 else True
        if not (first_hit or relog_due):
            return
        self._engine_no_quote_diag_last_log_mono = now_m
        # Compose payload from build telemetry + execution-side latches.
        payload: dict[str, Any] = {
            "quote_cycle_id": decision_quote_cycle_id,
            "streak_ticks": int(self._engine_no_quote_streak_ticks),
            "position_qty": float(position_qty),
            "position_notional_usd": float(position_notional),
            "first_emission": bool(first_hit),
        }
        # Include the engine telemetry that explains the no-quote state.
        for k in (
            "quote_engine_inventory_bias_active",
            "quote_engine_inventory_bias_suppressed_bid",
            "quote_engine_inventory_bias_suppressed_ask",
            "quote_engine_bid_reason",
            "quote_engine_ask_reason",
            "quote_engine_normal_mode_requested",
            "exec_raw_bid_sz",
            "exec_raw_ask_sz",
            "downgraded_cycle_skipped_reason",
        ):
            if k in telemetry:
                payload[k] = telemetry[k]
        # Side-unresolved status. Captures the latched-block scenario.
        for side in (Side.BUY, Side.SELL):
            payload[f"side_unresolved_active_{side.value.lower()}"] = bool(
                self._side_unresolved_active.get(side, False)
            )
            payload[f"side_unresolved_reason_{side.value.lower()}"] = (
                self._side_unresolved_reason.get(side)
            )
            payload[f"side_unresolved_requires_confirm_{side.value.lower()}"] = bool(
                self._side_unresolved_requires_confirm.get(side, False)
            )
        log_extra(
            logger,
            logging.WARNING,
            "engine_no_quote_persistent",
            payload,
        )
        try:
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.WARNING.value,
                "engine_no_quote_persistent",
                "engine returned no_quote for persistent streak — possible silent wedge",
                payload,
            )
        except Exception:
            logger.exception("engine_no_quote_persistent_event_persist_failed")

    # v1.4.78 hot-path optimization: high-frequency noop branches
    # that the postmortem rarely cares about. Per-tick rate-limited
    # to ~1-in-N (configurable; default 1-in-20) so the ring buffer
    # still gets representative samples but the dict-allocation hot
    # path doesn't pay 30k/min for tracing a benign state.
    #
    # Counter still increments on EVERY decision (cumulative trace
    # accuracy preserved). Only the ring-buffer entry + dict
    # construction is sampled.
    _NOOP_HIGH_FREQ_ACTIONS: frozenset[str] = frozenset({
        "noop:acked_no_reprice_needed",
        "noop:empty_match",
        "noop:terminal_match",
    })

    # v1.4.89 Phase 5C — incident auto-elevation window. When a
    # silent_wedge fires, ``incident_only`` mode behaves as ``always``
    # for this many seconds after the fire. 300s = 5 minutes; enough
    # to cover most wedge investigation windows.
    _TRACE_MODE_INCIDENT_WINDOW_S: float = 300.0

    def _trace_mode_incident_active(self) -> bool:
        """v1.4.89 Phase 5C — is an "incident" currently active for
        trace-mode auto-elevation purposes?

        An incident is active when ANY of:
          * the bot's risk_exec_state is NOT in the healthy set
            (NORMAL is healthy; SUPPRESSED / DEGRADED / etc are not),
          * a silent_wedge diagnostic fired in the last
            ``_TRACE_MODE_INCIDENT_WINDOW_S`` seconds.

        Used by ``_record_orchestrate_decision`` in ``incident_only``
        trace mode: high-freq noops are 100% skipped when no incident
        is active, but trace returns to full fidelity automatically
        during incidents.
        """
        # Silent-wedge recency check via the existing
        # ``_silent_wedge_diag_last_log_mono`` timestamp.
        last_wedge_mono = float(
            getattr(self, "_silent_wedge_diag_last_log_mono", 0.0) or 0.0
        )
        if last_wedge_mono > 0:
            elapsed = self._clock.monotonic() - last_wedge_mono
            if elapsed < self._TRACE_MODE_INCIDENT_WINDOW_S:
                return True
        # Risk-state check. The bot's risk_exec_state lives on
        # ``self._state.executor_state_snapshot["risk_exec_state"]``
        # via the TelemetryStore facade; reading defensively.
        try:
            risk = self._state.telemetry_store.risk_exec_state()
        except Exception:
            risk = "UNKNOWN"
        if str(risk).upper() not in ("NORMAL", "UNKNOWN"):
            return True
        return False

    def _record_orchestrate_decision(
        self,
        *,
        side: Side,
        level_idx: int,
        cur: Optional[WorkingOrder],
        desired: Optional[FinalQuoteOrder],
        decision_branch: str,
        action: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """v1.4.50 BUG-025 follow-on: log + record every _orchestrate decision.

        Every return path in the nested ``_orchestrate`` closure
        (inside ``maybe_refresh_quotes``) and the ``_stage_place_order_local``
        funnel calls this. Two outputs:

          1. **Ring buffer** (``self._orchestrate_decision_history[side]``):
             Last N entries per side, kept for postmortem visibility.
             The silent-wedge detector inlines this list so the operator
             sees EXACTLY which branch fired during the wedge — no more
             guessing which silent-return path is the culprit. Always
             recorded regardless of the log flag.

          2. **Counter dict** (``self._orchestrate_decision_counts``):
             Cumulative (side, decision_branch, action) counts since
             session start. Useful for "how many times did
             materiality_below_threshold fire?" without scrubbing logs.

          3. **Optional INFO log line** (gated by
             ``settings.executor_decision_trace_enabled``): one line per
             decision. At the 0.5 s quote loop that's ~4 lines/sec.
             Toggle off via ``EXECUTOR_DECISION_TRACE_ENABLED=false``
             once the root cause is identified.

        ``decision_branch`` names the entered branch (e.g.
        ``"acked_partial_reprice"``, ``"terminal_fresh_place"``,
        ``"unhandled_wo_status"``).

        ``action`` names what the bot DID (e.g. ``"return_silent"``,
        ``"cancel_for_side_suppressed"``, ``"amend_dispatched"``,
        ``"placed_fresh"``, ``"stage_returned_none"``).

        ``extra`` is a small dict of branch-specific details (e.g.
        ``{"hard_reasons": [...]}``, ``{"unresolved_reason": "..."}``).
        Keep keys/values primitive — this gets JSON-serialized.
        """
        # v1.4.78 hot-path optimization: counter ALWAYS bumps
        # (cumulative trace accuracy preserved). Dict construction +
        # ring-buffer append is rate-sampled for high-frequency noop
        # actions — they flood the ring buffer with no postmortem
        # value (32k+ events in 10 min snapshot v1.4.77-260518-234804;
        # ring buffer is bounded so noise crowds out genuinely useful
        # entries).
        #
        # The FIRST occurrence of any (side, branch, action) tuple is
        # always recorded (count == 1) so postmortem sees at least
        # one exemplar. Subsequent samples follow the interval. Net:
        # ~5% of high-frequency noops hit the ring buffer.
        counter_key = f"{side.value}:{decision_branch}:{action}"
        new_count = self._orchestrate_decision_counts.get(counter_key, 0) + 1
        self._orchestrate_decision_counts[counter_key] = new_count
        # v1.4.89 wedge-elimination-cleanup Phase 5C — trace-mode gate.
        # The cumulative counter ALWAYS increments above (postmortem
        # aggregate accuracy preserved). The mode gates only the ring-
        # buffer entry + dict construction + log line.
        #
        # Modes:
        #   * always         — full trace (v1.4.78 sampling still applies)
        #   * incident_only  — skip ALL high-freq noops when no incident
        #                      active (risk == NORMAL && no silent_wedge
        #                      in last 5 min). Cancels/places/amends/
        #                      errors / non-NORMAL risk always trace.
        #   * off            — skip ring buffer + log for all actions
        #                      (counter already incremented above).
        trace_mode = getattr(self._settings, "executor_trace_mode", "always")
        if trace_mode == "off":
            return
        if (
            trace_mode == "incident_only"
            and action in self._NOOP_HIGH_FREQ_ACTIONS
            and self._trace_mode_incident_active() is False
        ):
            return
        if (
            action in self._NOOP_HIGH_FREQ_ACTIONS
            and new_count != 1
            and (new_count % _NOOP_TRACE_SAMPLE_INTERVAL) != 0
        ):
            # Skip dict construction + ring-buffer append for this
            # sample. The counter increment above ensures cumulative
            # decision-count integrity for postmortem aggregates.
            return
        now_iso = self._clock.now_utc().isoformat()
        cur_status = cur.status.value if cur is not None else "none"
        cur_px = float(cur.price) if cur is not None else None
        cur_sz = float(cur.size) if cur is not None else None
        cur_oid_ex = (
            str(cur.order_id_exchange) if cur is not None and cur.order_id_exchange else None
        )
        cur_age_s: Optional[float] = None
        if cur is not None:
            cur_ts_created = getattr(cur, "ts_created_mono", 0.0) or 0.0
            if cur_ts_created > 0:
                cur_age_s = round(self._clock.monotonic() - float(cur_ts_created), 3)
        desired_present = desired is not None
        desired_px = float(desired.price) if desired is not None else None
        desired_sz = float(desired.size) if desired is not None else None
        entry: dict[str, Any] = {
            "ts": now_iso,
            "side": side.value,
            "level_idx": int(level_idx),
            "cur_status": cur_status,
            "cur_px": cur_px,
            "cur_sz": cur_sz,
            "cur_oid_ex": cur_oid_ex,
            "cur_age_s": cur_age_s,
            "desired_present": desired_present,
            "desired_px": desired_px,
            "desired_sz": desired_sz,
            "decision_branch": decision_branch,
            "action": action,
        }
        if extra:
            # Shallow-merge — branch extras are small (a few keys).
            for k, v in extra.items():
                entry[k] = v
        # Ring buffer — recorded for every NON-sampled decision.
        # (For high-freq noops, only 1 in N is added to the buffer;
        # see fast-path above.)
        try:
            self._orchestrate_decision_history[side].append(entry)
        except Exception:
            # Defensive — should be impossible, but never let
            # instrumentation kill the hot path.
            pass
        # Optional INFO log — gated.
        if bool(
            getattr(self._settings, "executor_decision_trace_enabled", False)
        ):
            log_extra(
                logger,
                logging.INFO,
                "orchestrate_decision",
                entry,
            )

    def _record_quote_refresh_skip(
        self,
        *,
        reason: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """v1.4.53 wedge-investigation: record a maybe_refresh_quotes
        early-return. The four early-return paths (no_write_access,
        cancel_resting_for_risk, no_quote_hold_resting, residual_flatten)
        all skip ``_orchestrate`` entirely — which means
        ``_record_orchestrate_decision`` won't fire, and the wedge
        detector sees ``_orchestrate`` looking idle without knowing
        why. This helper closes that observability gap.

        Same shape as ``_record_orchestrate_decision``:
          * Ring buffer (``_quote_refresh_skip_history``): last N entries.
          * Counter dict (``_quote_refresh_skip_counts``): cumulative
            per-reason tally.
          * Optional INFO log (gated by
            ``executor_decision_trace_enabled``).

        Surfaced in executor_state_snapshot under
        ``quote_refresh_skip_history`` and
        ``quote_refresh_skip_counts`` so the next wedge event payload
        carries the full picture.
        """
        entry: dict[str, Any] = {
            "ts": self._clock.now_utc().isoformat(),
            "reason": reason,
        }
        if extra:
            for k, v in extra.items():
                entry[k] = v
        try:
            self._quote_refresh_skip_history.append(entry)
        except Exception:
            pass
        self._quote_refresh_skip_counts[reason] = (
            self._quote_refresh_skip_counts.get(reason, 0) + 1
        )
        if bool(
            getattr(self._settings, "executor_decision_trace_enabled", False)
        ):
            log_extra(
                logger,
                logging.INFO,
                "quote_refresh_skip",
                entry,
            )

    # ------------------------------------------------------------------
    # v1.4.60 Phase 5: reconciler-driven dispatch
    # ------------------------------------------------------------------
    #
    # ``compute_desired_state(...)`` is the pure decision layer entry
    # point: it consumes engine output (which itself has already applied
    # engine-level gates: post_fill_cooldown, inventory_exec_bias,
    # microprice_gate, markout_adverse_cancel, etc.) and produces a
    # :class:`app.reconciler.DesiredLadderState`. The
    # ``apply_reducing_side_bypass`` fold runs LAST and is the
    # structural invariant that prevents the v1.4.59 wedge class.
    #
    # ``_dispatch_action(...)`` is the side-effect layer: it consumes
    # one :class:`app.reconciler.Action` and dispatches the matching
    # transport intent via the existing helpers
    # (``_enqueue_place_transport``, ``_enqueue_cancel_quote_path``,
    # ``_enqueue_amend_quote_path``). Same wire behaviour as the
    # pre-v1.4.60 ``_orchestrate`` closure; cleaner control flow.

    # v1.4.92 Phase 4A cutover — Phase 2D consumer audit REMOVED.
    # The runtime validator ``_mark_qbr_consumed`` /
    # ``_validate_qbr_consumption`` + counter
    # ``_quote_build_result_unconsumed_field_total`` are deleted. The
    # typed ``BuildCommand`` sum-type replaces the audit
    # structurally: new fields on a variant are explicit attributes
    # whose readers are forced at edit time by exhaustive ``match``
    # statements, not detected at runtime by "did anyone call
    # _mark_qbr_consumed for this?". See plan section "Phase 4A".

    def compute_desired_state(
        self,
        *,
        build,
        decision,
        ladder,
        position_qty: float,
        num_levels: int,
        client_spec,
    ) -> tuple[DesiredLadderState, DesiredLadderState, list[dict]]:
        """Pure(-ish) decision layer.

        Returns ``(post_gates_desired, engine_desired_snapshot, rejections)``:

        * ``post_gates_desired`` is the state after all folds, INCLUDING
          the reducing-side bypass. The reconciler diffs against this.
        * ``engine_desired_snapshot`` is the pre-bypass desired state —
          used by ``apply_reducing_side_bypass`` to restore the reducer
          when a gate had suppressed it. Surfaced so callers can see
          which slots the bypass restored vs left suppressed.
        * ``rejections`` is the per-rung ``normalize_order_pair`` reject
          list — surfaced for the existing ``ladder_dispatch_no_levels``
          diagnostic + the wedge-event payload.

        This is "pure-ish" because it reads ``self._settings`` and
        ``client_spec`` (constant inputs across a tick) but never
        mutates state. Same inputs → same outputs.
        """
        # v1.4.92 Phase 4A cutover — Phase 2D ``_mark_qbr_consumed``
        # calls removed. ``bid_order`` / ``ask_order`` are read in the
        # loop below via the BuildCommand variants' legacy-compat
        # properties (which return Optional[FinalQuoteOrder] for any
        # variant); no audit needed because the variants' fields are
        # all explicit.
        engine_desired: DesiredLadderState = {}
        rejections: list[dict] = []
        num_levels = max(1, int(num_levels))
        use_ladder = ladder is not None and num_levels > 1

        for side in (Side.BUY, Side.SELL):
            engine_order = build.bid_order if side == Side.BUY else build.ask_order
            inside_target_half = (
                engine_order.target_half_spread_bps
                if engine_order is not None
                else None
            )
            if use_ladder:
                rungs = ladder.bids if side == Side.BUY else ladder.asks
                rung_by_idx = {int(r.level_idx): r for r in rungs}
                for lvl in range(num_levels):
                    # v1.4.65 wedge fix — INSIDE RUNG self-heal.
                    #
                    # Pre-v1.4.65: every ladder rung (including level 0,
                    # the inside rung) was normalized via
                    # ``normalize_order_pair`` which does grid rounding
                    # but NOT min-notional self-heal. The engine's
                    # ``_build_side`` already produced a fully self-
                    # healed ``build.bid_order`` / ``build.ask_order``
                    # for the inside rung — but the multi-rung dispatch
                    # path threw that away and re-derived the inside
                    # rung from ``ladder.bids[0].sz`` (which is the
                    # PRE-self-heal float ``decision.quoted_bid_sz``).
                    #
                    # Snapshot v1.4.64-260518-185510 caught the wedge:
                    # position +3 long, SELL is reducer, engine produced
                    # ask at 1.947 with self-healed size, but ladder
                    # inside rung had sz=2.83 → rounded to 2.0 → notional
                    # $3.89 < min_notional $5.0 → REJECTED. Bot stuck
                    # for 90+ s with no orders.
                    #
                    # Fix: for the INSIDE rung (lvl=0), use
                    # ``engine_order`` (which IS already grid-aligned
                    # AND min-notional-satisfied). For OUTER rungs
                    # (lvl >= 1), keep the ladder-rung path — outer
                    # rungs are operator-tuned offsets and their sizes
                    # are intentional; if they fall below min_notional
                    # after rounding, skipping them is correct.
                    if lvl == 0 and engine_order is not None:
                        engine_desired[(side, 0)] = DesiredOrderState(
                            side=side,
                            level_idx=0,
                            should_exist=True,
                            price=float(engine_order.price),
                            size=float(engine_order.size),
                            target_half_spread_bps=engine_order.target_half_spread_bps,
                            aging_tighten_applied=bool(engine_order.aging_tighten_applied),
                            reason="engine_quote",
                        )
                        continue
                    if lvl == 0 and engine_order is None:
                        # Engine produced no inside-rung quote; treat as empty.
                        engine_desired[(side, 0)] = DesiredOrderState.empty(
                            side=side, level_idx=0, reason="engine_no_quote_inside"
                        )
                        continue
                    rung = rung_by_idx.get(lvl)
                    if rung is None:
                        engine_desired[(side, lvl)] = DesiredOrderState.empty(
                            side=side, level_idx=lvl, reason="engine_no_rung"
                        )
                        continue
                    normed, rej = normalize_order_pair(
                        client_spec, float(rung.px), float(rung.sz)
                    )
                    if normed is None:
                        rejections.append({
                            "side": side.value,
                            "level_idx": lvl,
                            "px": float(rung.px),
                            "sz": float(rung.sz),
                            "rej": str(rej) if rej else "unknown",
                        })
                        # v1.4.222 — Phase 4G.11 instrumentation. Bump
                        # the dispatcher-side normalize-rejection counter
                        # so the operator can see "WHY is rung-1 never
                        # placing?" without parsing logs. Two sub-reasons
                        # so the operator can act surgically.
                        try:
                            if rejection_is_below_min_notional_usd(rej):
                                self._state.ladder_rung_dropped_normalize_below_min_notional_total += 1
                            else:
                                self._state.ladder_rung_dropped_normalize_other_total += 1
                        except Exception:
                            # Counter bump must never break the trading loop.
                            pass
                        engine_desired[(side, lvl)] = DesiredOrderState.empty(
                            side=side,
                            level_idx=lvl,
                            reason=f"normalize_rejected:{rej}" if rej else "normalize_rejected",
                        )
                        continue
                    norm_px, norm_sz = normed
                    # v1.4.86 wedge-elimination-cleanup — post-normalize
                    # dedup. The build_ladder dedup catches most grid
                    # collisions but can't see the engine's _build_side
                    # post-only / min-half-spread clamps that shift the
                    # INSIDE rung's final price after the ladder was
                    # built. Without this check, an outer rung that
                    # rounds to the SAME final price as the engine's
                    # post-clamp inside rung lands on the book as a
                    # same-price duplicate (snapshot 260519 BUY 2.031×2
                    # SELL 2.034×2). Drop the outer rung in that case.
                    collision = False
                    if engine_order is not None:
                        inside_final_px = float(engine_order.price)
                        if abs(float(norm_px) - inside_final_px) < 1e-9:
                            collision = True
                    if not collision:
                        # Also check against any already-emitted outer
                        # rung on this side (lvl < current lvl). Three-
                        # rung ladder could have lvl 1 and lvl 2 collide
                        # at the same tick even if both differ from lvl 0.
                        for prior_lvl in range(1, lvl):
                            prior_st = engine_desired.get((side, prior_lvl))
                            if (
                                prior_st is not None
                                and prior_st.should_exist
                                and abs(float(norm_px) - float(prior_st.price)) < 1e-9
                            ):
                                collision = True
                                break
                    if collision:
                        rejections.append({
                            "side": side.value,
                            "level_idx": lvl,
                            "px": float(norm_px),
                            "sz": float(norm_sz),
                            "rej": "post_normalize_grid_collision",
                        })
                        engine_desired[(side, lvl)] = DesiredOrderState.empty(
                            side=side,
                            level_idx=lvl,
                            reason="post_normalize_grid_collision",
                        )
                        continue
                    engine_desired[(side, lvl)] = DesiredOrderState(
                        side=side,
                        level_idx=lvl,
                        should_exist=True,
                        price=float(norm_px),
                        size=float(norm_sz),
                        target_half_spread_bps=inside_target_half,
                        aging_tighten_applied=False,  # outer rungs never aging-tighten
                        reason="engine_quote",
                    )
            else:
                # Single-rung path: only level 0 exists.
                if engine_order is None:
                    engine_desired[(side, 0)] = DesiredOrderState.empty(
                        side=side, level_idx=0, reason="engine_no_quote"
                    )
                else:
                    engine_desired[(side, 0)] = DesiredOrderState(
                        side=side,
                        level_idx=0,
                        should_exist=True,
                        price=float(engine_order.price),
                        size=float(engine_order.size),
                        target_half_spread_bps=engine_order.target_half_spread_bps,
                        aging_tighten_applied=bool(engine_order.aging_tighten_applied),
                        reason="engine_quote",
                    )

        # v1.4.66 → v1.4.70 wedge fix — HARD AGE CAP enforcement.
        #
        # The QuoteEngine populates ``build.hard_cancel_by_slot`` (Phase
        # 1C) with per-slot reason tuples when a resting order has
        # exceeded ``BEHIND_TOUCH_MAX_AGE_SECONDS`` or
        # ``AT_TOUCH_MAX_AGE_SECONDS``. Pre-Phase-5 ``_orchestrate``
        # consumed similar signals via ``hard_cancel_bid_reasons`` /
        # ``hard_cancel_ask_reasons`` for the inside rung only.
        #
        # Phase 1C upgrade:
        #   * Multi-rung: empty EVERY (side, level_idx) slot with
        #     non-empty reasons, not just (side, 0). Closes Codex F6/B3
        #     where outer rungs aged forever.
        #   * Wall-lifetime: ``resting_age_seconds`` now uses ``ts_sent``
        #     so SENT orders (slow ACK) are also cap-eligible. Closes
        #     the v1.4.67/v1.4.69 paired slow-ACK wedge mechanism.
        #
        # When the cap fires, the reconciler emits CancelAction (cur
        # in ACKED/PARTIAL/SENT, desired=empty → cancel). The engine's
        # NEXT tick produces a fresh quote.
        #
        # Backward compat: if ``hard_cancel_by_slot`` is missing
        # (unlikely after v1.4.70 but defensive), fall back to the
        # inside-rung tuples.
        # v1.4.92 Phase 4A cutover — Phase 2D ``_mark_qbr_consumed``
        # calls removed. The hard_cancel_by_slot map is read via the
        # BuildCommand variants' legacy-compat property; defaults to
        # empty for NoQuote / ResidualFlatten variants.
        hard_by_slot: dict[tuple[Side, int], tuple[str, ...]] = dict(
            getattr(build, "hard_cancel_by_slot", {}) or {}
        )
        if not hard_by_slot:
            hard_bid = tuple(getattr(build, "hard_cancel_bid_reasons", ()) or ())
            hard_ask = tuple(getattr(build, "hard_cancel_ask_reasons", ()) or ())
            if hard_bid:
                hard_by_slot[(Side.BUY, 0)] = hard_bid
            if hard_ask:
                hard_by_slot[(Side.SELL, 0)] = hard_ask
        for (side_key, lvl), reasons in hard_by_slot.items():
            if not reasons:
                continue
            # Always override with the hard_age_cap reason — even if
            # the slot was already empty from a normalize_rejected
            # path. The cap reason is more semantically accurate for
            # postmortem ("we want this slot dead because the order
            # has aged out", not "we want this slot dead because the
            # engine couldn't quote there"). The reconciler emits
            # CancelAction when current is ACKED/PARTIAL regardless.
            engine_desired[(side_key, int(lvl))] = DesiredOrderState.empty(
                side=side_key,
                level_idx=int(lvl),
                reason=f"hard_age_cap:lvl={int(lvl)}:{','.join(reasons[:3])}",
            )
        # Gate folds (v1.4.74 wedge-elimination-cleanup Phase 2A).
        #
        # Pre-Phase-2A the dispatcher silently vetoed PlaceActions on
        # three conditions:
        #   1. ``_is_side_unresolved(side)`` → ``return_silent`` branch
        #      (538 such suppressions per snapshot v1.4.66-260518-192744
        #      per side; 1264 in v1.4.73-260518-222923).
        #   2. ``_stage_place_order_local`` returned None due to
        #      ``adverse_side_pause`` → ``place_stage_returned_none``
        #      branch (143 in v1.4.66).
        #   3. Same for ``post_only_cross_cooldown``.
        #
        # The silent veto was the structural property that allowed the
        # v1.4.66 hard-age-cap signal drop to go unnoticed — Codex F2:
        # "the dispatcher still has post-reconciler silent vetoes".
        # Phase 2A lifts these gates UP into the desired-state layer
        # as folds. The reconciler now sees the suppressed slot as
        # ``should_exist=False, reason="gate:<gate>:..."`` and emits a
        # CancelAction (for any live WO) or NoOpAction with the gate
        # reason in the decision-trace buffer.
        #
        # Layering wrt the reducing-side bypass:
        #
        #   * ``side_unresolved`` is a state-uncertainty signal ("we
        #     can't tell what's on the venue"), NOT a defensive
        #     cooldown. The reducer-bypass MUST NOT restore in this
        #     case — placing more orders while uncertain risks
        #     duplicates. Applied BEFORE the bypass-source snapshot
        #     so the bypass sees the slot as already suppressed and
        #     cannot restore it.
        #
        #   * ``adverse_side_pause`` and ``post_only_cross_cooldown``
        #     ARE defensive cooldowns. The reducer-bypass MUST
        #     override them for the reducer when inventory exists
        #     (BUG-010 / v1.4.60 invariant). Applied AFTER the
        #     bypass-source snapshot so the bypass can restore the
        #     untouched engine quote for the reducer.
        #
        # The defensive gates inside ``_stage_place_order_local`` stay
        # in place as belt-and-suspenders — they should now rarely
        # fire because the desired-state fold short-circuits the place
        # intent upstream.
        unresolved_sides: set[Side] = set()
        for side in (Side.BUY, Side.SELL):
            if self._is_side_unresolved(side):
                unresolved_sides.add(side)
                sub_reason = self._side_unresolved_reason.get(side) or "unknown"
                engine_desired = apply_side_suppression(
                    engine_desired,
                    side=side,
                    reason=f"side_unresolved:{sub_reason}",
                )
        if unresolved_sides:
            self._gate_side_unresolved_applied_total += 1

        # v1.4.78 hot-path optimization: only allocate the bypass-
        # source snapshot if cooldown gates MIGHT modify
        # ``engine_desired`` further AND inventory is non-zero (the
        # only state where the bypass actually restores). Saves a
        # per-tick ``dict(engine_desired)`` allocation on the happy
        # path (no cooldowns active, or position flat).
        adverse_applied = False
        post_only_applied = False
        # Peek-check whether any cooldown gate is active on a
        # non-unresolved side. Cheap: 2-4 calls to dict-lookup
        # helpers. Avoids the dict allocation when no fold will fire.
        will_apply_cooldown_fold = False
        if position_qty != 0.0:
            for side in (Side.BUY, Side.SELL):
                if side in unresolved_sides:
                    continue
                if (
                    self._adverse_side_pause_active(side)
                    or self._post_only_cross_cooldown_active(side)
                ):
                    will_apply_cooldown_fold = True
                    break

        if will_apply_cooldown_fold:
            # Snapshot the pre-cooldown state so the bypass has an
            # untouched source for reducer restoration.
            engine_desired_bypass_source: dict = dict(engine_desired)
        else:
            # No bypass will fire (flat position OR no cooldowns
            # active). Reuse the same reference; nothing downstream
            # mutates it after this point.
            engine_desired_bypass_source = engine_desired

        for side in (Side.BUY, Side.SELL):
            if side in unresolved_sides:
                continue  # Already suppressed; cooldowns moot.
            if self._adverse_side_pause_active(side):
                adverse_applied = True
                remaining_s = max(
                    0.0,
                    self._adverse_side_pause_until.get(side, 0.0) - self._clock.monotonic(),
                )
                engine_desired = apply_side_suppression(
                    engine_desired,
                    side=side,
                    reason=f"adverse_side_pause:remaining_s={round(remaining_s, 3)}",
                )
            if self._post_only_cross_cooldown_active(side):
                post_only_applied = True
                remaining_s = max(
                    0.0,
                    self._post_only_cross_cooldown_until.get(side, 0.0)
                    - self._clock.monotonic(),
                )
                engine_desired = apply_side_suppression(
                    engine_desired,
                    side=side,
                    reason=f"post_only_cross_cooldown:remaining_s={round(remaining_s, 3)}",
                )
        if adverse_applied:
            self._gate_adverse_side_pause_applied_total += 1
        if post_only_applied:
            self._gate_post_only_cross_cooldown_applied_total += 1

        # v1.4.78 hot-path optimization: skip the bypass call when
        # ``position_qty == 0`` (the bypass internally early-returns
        # in that case, but we save the function call + arg binding).
        if position_qty != 0.0:
            post_gates_desired = apply_reducing_side_bypass(
                engine_desired,
                position_qty=position_qty,
                engine_desired=engine_desired_bypass_source,
                enabled=True,
            )
        else:
            post_gates_desired = engine_desired
        return post_gates_desired, engine_desired_bypass_source, rejections

    def _snapshot_working_orders_for_reconciler(
        self, num_levels: int
    ) -> dict[tuple[Side, int], WorkingOrder]:
        """Read every relevant (side, level_idx) working order slot
        under a single state-lock acquire. Returns a flat dict for
        the reconciler.

        Includes BOTH the configured rungs AND any orphan rungs (idx
        beyond num_levels) so the reconciler can issue cancels for
        rungs left over from a config change (operator dropped
        ``LADDER_NUM_LEVELS_PER_SIDE`` from 2 → 1).
        """
        snapshot: dict[tuple[Side, int], WorkingOrder] = {}
        configured = max(1, int(num_levels))
        with self._state._lock:
            for side in (Side.BUY, Side.SELL):
                # Configured slots.
                for lvl in range(configured):
                    wo = self._state.get_working_order(side, lvl)
                    if wo is not None:
                        snapshot[(side, lvl)] = wo
                # Orphan slots beyond configured.
                for idx, wo in self._state.iter_working_orders(side):
                    if int(idx) >= configured and wo is not None:
                        snapshot[(side, int(idx))] = wo
        return snapshot

    def _dispatch_action(
        self,
        action,
        *,
        decision,
        mid_price: float,
        tick: float,
    ) -> None:
        """Execute one reconciler Action via the appropriate transport
        helper. Same wire behaviour as the pre-v1.4.60 ``_orchestrate``
        closure; cleaner control flow.

        v1.4.74 wedge-elimination-cleanup Phase 2A: the per-side
        ``side_unresolved`` gate has been LIFTED out of the dispatcher
        into ``compute_desired_state`` as a fold. The reconciler now
        sees an empty desired state for unresolved sides and emits
        ``CancelAction`` (for live WOs) or ``NoOpAction`` directly.
        The dispatcher becomes transport-only; place stage-time gates
        stay as belt-and-suspenders.

        Cancels CAN still flow on unresolved sides — they must, so the
        bot can clear orders during state-uncertainty. Pre-Phase-2A
        cancels were also silently suppressed alongside places, which
        contributed to the wedge mechanism (orders pile up on
        unresolved sides indefinitely).
        """
        side = action.side
        level_idx = int(action.level_idx)

        # Per-side housekeeping: cancel-pending timeout. Runs against
        # the slot's CURRENT wo regardless of what the action says.
        with self._state._lock:
            cur = self._state.get_working_order(side, level_idx)
        self._maybe_handle_cancel_pending_timeout(side, cur)

        # v1.4.74 Phase 2A: PlaceAction-only defense-in-depth check.
        # The Phase 2A fold already emptied desired-state for
        # unresolved sides, so the reconciler should not be emitting
        # ``PlaceAction`` here. If it does (a future caller bypassed
        # the fold), refuse and trace — this is a contract violation
        # we want visible, not silent.
        if (
            self._is_side_unresolved(side)
            and isinstance(action, (PlaceAction, AmendAction))
        ):
            if not self._suppress_place_unresolved_logged.get(side, False):
                self._suppress_place_unresolved_logged[side] = True
                log_extra(
                    logger,
                    logging.WARNING,
                    "phase2a_unresolved_place_dispatch_refused",
                    {
                        "side": side.value,
                        "reason": "side_unresolved",
                        "unresolved_reason": self._side_unresolved_reason.get(side),
                        "action_type": type(action).__name__,
                        "current_status": cur.status.value if cur else None,
                        "comment": (
                            "Phase 2A invariant violated: PlaceAction/AmendAction "
                            "reached dispatcher with side_unresolved active. "
                            "Should have been emptied in compute_desired_state."
                        ),
                    },
                )
            self._suppress_place_due_unresolved_count += 1
            self._gate_phase2a_invariant_violation_total += 1
            self._record_orchestrate_decision(
                side=side,
                level_idx=level_idx,
                cur=cur,
                desired=None,
                decision_branch="phase2a_invariant_violation",
                action="place_dispatch_refused_unresolved",
                extra={"unresolved_reason": self._side_unresolved_reason.get(side)},
            )
            return

        # NoOp: just record and return.
        if isinstance(action, NoOpAction):
            self._record_orchestrate_decision(
                side=side,
                level_idx=level_idx,
                cur=cur,
                desired=None,
                decision_branch="reconciler",
                action=f"noop:{action.reason}",
            )
            return

        # CancelAction
        if isinstance(action, CancelAction):
            if cur is None:
                # Race: WO went terminal between snapshot and dispatch.
                self._record_orchestrate_decision(
                    side=side,
                    level_idx=level_idx,
                    cur=None,
                    desired=None,
                    decision_branch="reconciler",
                    action=f"cancel_race_cur_none",
                    extra={"trigger_reason": action.trigger_reason},
                )
                return
            cancel_enqueued = self._enqueue_cancel_quote_path(
                cur, trigger_reason=action.trigger_reason
            )
            if cancel_enqueued and action.trigger_reason == "reprice_replace":
                self._last_outbound_replace_mono[side] = self._clock.monotonic()
                self._quote_reprice_replace_pending[side] = True
            self._record_orchestrate_decision(
                side=side,
                level_idx=level_idx,
                cur=cur,
                desired=None,
                decision_branch="reconciler",
                action=f"cancel_{'dispatched' if cancel_enqueued else 'enqueue_failed'}",
                extra={"trigger_reason": action.trigger_reason},
            )
            return

        # AmendAction
        if isinstance(action, AmendAction):
            if cur is None:
                # Race: WO disappeared. Convert to fresh place.
                self._record_orchestrate_decision(
                    side=side,
                    level_idx=level_idx,
                    cur=None,
                    desired=None,
                    decision_branch="reconciler",
                    action="amend_race_cur_none_falling_to_place",
                )
                # Fall through to PlaceAction handling using the
                # AmendAction's desired state.
                place = PlaceAction(
                    side=side, level_idx=level_idx, desired=action.desired
                )
                self._dispatch_action(place, decision=decision, mid_price=mid_price, tick=tick)
                return
            # Re-check amend viability at dispatch time (cur status
            # may have shifted since the reconciler ran).
            desired_fqo = self._desired_to_final_quote_order(action.desired)
            if self._amend_viable(cur, desired_fqo):
                if self._enqueue_amend_quote_path(
                    cur, desired_fqo, trigger_reason="reprice_amend"
                ):
                    amend_ts = self._clock.monotonic()
                    self._last_outbound_replace_mono[side] = amend_ts
                    self._state.last_outbound_attempt_ts_mono = amend_ts
                    self._record_orchestrate_decision(
                        side=side,
                        level_idx=level_idx,
                        cur=cur,
                        desired=desired_fqo,
                        decision_branch="reconciler",
                        action="amend_dispatched",
                    )
                    return
                # v1.4.180: distinguish Phase 2I suppression from a
                # genuine enqueue failure. If suppressed
                # (``tick_flicker`` / ``rate_throttle``) the rate-
                # defence guard fired — falling through to cancel-
                # replace defeats the guard's intent (cancel-replace
                # loses queue position just like rapid amends, and
                # both spike the order-event rate the guard is meant
                # to dampen). Skip the action this tick; the existing
                # order keeps resting. The BEHIND_TOUCH_MAX_AGE_SECONDS
                # safety net still caps how long a stale order lives.
                # Snapshot v1.4.176-260521-110704 surfaced this: 737
                # amend_enqueue_failed_falling_through events paired
                # with 737 amend_to_cancel_dispatched events in a 35
                # min session, ~3 cancel-replaces/sec, 1 fill total.
                suppress_reason = (
                    self._last_amend_enqueue_suppress_reason
                )
                if suppress_reason in ("tick_flicker", "rate_throttle"):
                    self._record_orchestrate_decision(
                        side=side,
                        level_idx=level_idx,
                        cur=cur,
                        desired=desired_fqo,
                        decision_branch="reconciler",
                        action=(
                            "amend_phase2i_suppressed_skipping"
                            f":{suppress_reason}"
                        ),
                    )
                    return
                # Amend viable but enqueue failed for a non-Phase-2I
                # reason — fall through to cancel-replace as before.
                self._record_orchestrate_decision(
                    side=side,
                    level_idx=level_idx,
                    cur=cur,
                    desired=desired_fqo,
                    decision_branch="reconciler",
                    action="amend_enqueue_failed_falling_through",
                )
            # Fallback: cancel-for-replace.
            cancel_enqueued = self._enqueue_cancel_quote_path(
                cur, trigger_reason="reprice_replace"
            )
            if cancel_enqueued:
                self._last_outbound_replace_mono[side] = self._clock.monotonic()
                self._quote_reprice_replace_pending[side] = True
            self._record_orchestrate_decision(
                side=side,
                level_idx=level_idx,
                cur=cur,
                desired=desired_fqo,
                decision_branch="reconciler",
                action=f"amend_to_cancel_{'dispatched' if cancel_enqueued else 'enqueue_failed'}",
            )
            return

        # PlaceAction
        if isinstance(action, PlaceAction):
            # Bluefin async-cancel gate (same as pre-v1.4.60 _orchestrate).
            has_pending_cancel = getattr(self._client, "has_pending_cancel", None)
            if callable(has_pending_cancel) and has_pending_cancel(
                self._settings.symbol, side
            ):
                log_extra(
                    logger,
                    logging.INFO,
                    "quote_skipped",
                    {
                        "reason": "bluefin_cancel_pending",
                        "side": side.value,
                        "symbol": self._settings.symbol,
                    },
                )
                self._record_orchestrate_decision(
                    side=side,
                    level_idx=level_idx,
                    cur=cur,
                    desired=None,
                    decision_branch="reconciler",
                    action="place_skipped_bluefin_cancel_pending",
                )
                return
            desired_fqo = self._desired_to_final_quote_order(action.desired)
            if not self._should_emit_fresh_place_intent(side, desired_fqo, tick):
                self._record_orchestrate_decision(
                    side=side,
                    level_idx=level_idx,
                    cur=cur,
                    desired=desired_fqo,
                    decision_branch="reconciler",
                    action="place_skipped_should_not_emit_fresh",
                )
                return
            wo = self._stage_place_order_local(
                side,
                price=float(desired_fqo.price),
                size=float(desired_fqo.size),
                quote_cycle_id=decision.quote_cycle_id,
                target_half_spread_bps=desired_fqo.target_half_spread_bps,
                aging_tighten_applied=bool(desired_fqo.aging_tighten_applied),
                decision=decision,
                level_idx=level_idx,
            )
            if wo is None:
                # Stage gate blocked (adverse_side_pause, post_only_cross_cooldown,
                # invalid size, etc.). The skip reason was cached.
                skip_reason = self._last_stage_place_skip_reason.get(side.value)
                self._record_orchestrate_decision(
                    side=side,
                    level_idx=level_idx,
                    cur=cur,
                    desired=desired_fqo,
                    decision_branch="reconciler",
                    action="place_stage_returned_none",
                    extra={"stage_skip_reason": skip_reason},
                )
                return
            if self._enqueue_place_transport(wo, decision.quote_cycle_id):
                self._last_emitted_fp[side] = (
                    float(desired_fqo.price),
                    float(desired_fqo.size),
                )
                self._last_outbound_replace_mono[side] = self._clock.monotonic()
                self._quote_reprice_replace_pending[side] = False
                self._record_orchestrate_decision(
                    side=side,
                    level_idx=level_idx,
                    cur=cur,
                    desired=desired_fqo,
                    decision_branch="reconciler",
                    action="place_dispatched",
                    extra={
                        "wo_local_id": wo.order_id_local,
                        "wo_status": wo.status.value,
                    },
                )
                return
            # Never-happens path; enqueue currently always returns True.
            self._record_orchestrate_decision(
                side=side,
                level_idx=level_idx,
                cur=cur,
                desired=desired_fqo,
                decision_branch="reconciler",
                action="place_enqueue_transport_failed",
                extra={
                    "wo_local_id": wo.order_id_local,
                    "wo_status": wo.status.value,
                },
            )
            return

    def _desired_to_final_quote_order(
        self, desired: DesiredOrderState
    ) -> FinalQuoteOrder:
        """Adapter: convert a DesiredOrderState into the existing
        FinalQuoteOrder shape expected by the transport helpers
        (``_stage_place_order_local``, ``_enqueue_amend_quote_path``,
        ``_amend_viable``, ``_should_replace_working_order``).
        """
        return FinalQuoteOrder(
            side=desired.side,
            price=float(desired.price),
            size=float(desired.size),
            target_half_spread_bps=desired.target_half_spread_bps,
            aging_tighten_applied=bool(desired.aging_tighten_applied),
        )

    def _reconcile_materiality_check(
        self,
        cur: WorkingOrder,
        desired: DesiredOrderState,
        *,
        replace_threshold_bps: float,
        mid_price: float,
        tick: float,
    ) -> bool:
        """Materiality predicate for the reconciler. Returns True iff
        the price/size delta is large enough to warrant a reprice.

        v1.4.88 Phase 5B: build the ``SlotDiff`` ONCE here and route
        both helpers through it. Pre-Phase-5B each helper recomputed
        px_diff and sz_diff from the same (cur, desired) pair —
        Codex F11 flagged the redundancy. The behavioural contract is
        unchanged; this is purely a "single source of truth for diff
        arithmetic" cleanup.
        """
        desired_fqo = self._desired_to_final_quote_order(desired)
        diff = SlotDiff.from_pair(
            cur_price=float(cur.price),
            cur_size=float(cur.size),
            desired_price=float(desired_fqo.price),
            desired_size=float(desired_fqo.size),
            tick=float(tick),
            mid_price=float(mid_price),
        )
        if not self._replace_threshold_check_from_diff(
            diff, replace_threshold_bps=replace_threshold_bps
        ):
            return False
        if not self._action_materiality_from_diff(diff, side=cur.side):
            return False
        return True

    @staticmethod
    def _replace_threshold_check_from_diff(
        diff: SlotDiff,
        *,
        replace_threshold_bps: float,
    ) -> bool:
        """v1.4.88 Phase 5B — diff-driven variant of
        ``_should_replace_working_order``. Pure, no self-reads, takes
        the pre-computed ``SlotDiff``. Identical semantics to the
        legacy helper:

            return not (px_bps_vs_mid < replace_threshold_bps
                        and sz_rel <= 0.15)
        """
        return not (
            diff.px_bps_vs_mid < float(replace_threshold_bps)
            and diff.sz_rel <= 0.15
        )

    def _action_materiality_from_diff(
        self,
        diff: SlotDiff,
        *,
        side: Side,
    ) -> bool:
        """v1.4.88 Phase 5B — diff-driven variant of
        ``_action_materiality_allows_replace``. Identical semantics
        to the legacy helper: epsilon-tick + epsilon-resize gates
        plus the per-side min-interval rate limit.
        """
        eps = float(self._settings.action_reprice_epsilon_ticks)
        rs = float(self._settings.action_resize_epsilon_ratio)
        min_iv = float(self._settings.action_min_replace_interval_ms) / 1000.0
        if eps <= 0 and rs <= 0 and min_iv <= 0:
            return True
        material = False
        if eps > 0 and diff.px_ticks + 1e-9 >= eps:
            material = True
        if rs > 0 and diff.sz_rel + 1e-9 >= rs:
            material = True
        if eps <= 0 and rs <= 0:
            material = True
        elif not material:
            return False
        if min_iv > 0:
            last = float(self._last_outbound_replace_mono.get(side, 0.0))
            if last > 0 and (self._clock.monotonic() - last) + 1e-9 < min_iv:
                return False
        return True

    def _reconcile_amend_viable_check(
        self,
        cur: WorkingOrder,
        desired: DesiredOrderState,
    ) -> bool:
        """Amend-viability predicate for the reconciler."""
        return self._amend_viable(
            cur, self._desired_to_final_quote_order(desired)
        )

    def _maybe_emit_silent_wedge_diagnostic(
        self,
        *,
        telemetry: dict[str, Any],
        decision_quote_cycle_id: str,
        eligibility: str,
        active_sides: str,
        position_qty: float,
        position_notional: float,
    ) -> None:
        """v1.4.40 BUG-025: catch the executor silent-wedge class.

        Distinct from ``_maybe_emit_engine_no_quote_diagnostic``
        which catches engine-side stalls (engine returns
        ``mode='no_quote'`` for many ticks). This method catches the
        INVERSE — the engine IS producing valid quotes, eligibility
        and gates are clean, but the executor never places.

        The conjoint detection condition:

        - ``mode != 'no_quote'`` (engine is healthy)
        - ``eligibility == 'QUOTE_BOTH'`` (no eligibility gate)
        - ``active_sides == 'BOTH'`` (decision layer agrees)
        - No ``side_unresolved`` is set
        - ``execution_idle_seconds > threshold`` (default 60 s)

        With ALL of these true, the bot is silently wedged by
        definition. Fires ``executor_silent_wedge_detected`` at
        ERROR level with the full executor-state block. Re-fires
        every N seconds while sustained (same cadence as
        ``engine_no_quote_persistent``).
        """
        if not bool(
            getattr(self._settings, "silent_wedge_detect_enabled", True)
        ):
            return
        # The engine must be healthy this cycle.
        if str(telemetry.get("quote_engine_mode") or "") == "no_quote":
            return
        # v1.4.206 BUG-025 false-positive fix. The engine returns
        # ``one_sided`` when one side is intentionally suppressed
        # (e.g. ``inventory_bias_suppressed_adding_side`` when the bot
        # is at high util) — the OTHER side may have a perfectly
        # healthy resting order maintaining the quote. Treating that
        # as a wedge produced ~20 false-positive fires per session
        # observed in ``snapshots/v1.4.180-260521-141619``.
        # ``no_quote`` already exempted above; ``one_sided`` joins it.
        if str(telemetry.get("quote_engine_mode") or "") == "one_sided":
            return
        if eligibility != "QUOTE_BOTH":
            return
        if active_sides != "BOTH":
            return
        # Any side_unresolved means the bot has explicit
        # justification for not placing — not a silent wedge.
        if any(
            self._side_unresolved_active.get(side, False)
            for side in (Side.BUY, Side.SELL)
        ):
            return
        # v1.4.206 BUG-025 second false-positive fix. If the bot has
        # at least one ACKED working order on EITHER side, the bot
        # is NOT silently wedged — it's maintaining a stable quote
        # and ``last_outbound_attempt_ts_mono`` correctly hasn't
        # advanced because no amend/place was needed. The original
        # detector assumption ("no place/amend in N seconds ⇒ wedge")
        # is wrong for the steady-state-resting case. The genuine
        # wedge class is "executor produces nothing AND no working
        # orders are protecting the quote" — that's what this check
        # narrows down to. ``WorkingOrder.status`` is ACKED for live
        # quotes; PARTIAL also counts (partially filled, still on
        # book); SENT / NEW_LOCAL / CANCEL_PENDING / AMEND_PENDING
        # are in-flight states that DO advance the outbound counter,
        # so they're not the steady-state-resting class we want to
        # exempt here.
        try:
            store = getattr(self._state, "order_store", None)
            if store is not None:
                for side in (Side.BUY, Side.SELL):
                    wo = store.get(side, 0)
                    if wo is not None and wo.status in (
                        OrderStatus.ACKED,
                        OrderStatus.PARTIAL,
                    ):
                        return
        except Exception:
            # Defensive — never let the detector's own bug suppress
            # a legitimate wedge alert. If the order_store check
            # itself throws, fall through to the original detection.
            logger.exception("silent_wedge_resting_order_check_failed")
        # v1.5.159 BUG-031 fix — engine-no_quote-recovery grace
        # period. When the engine has just recovered from an
        # extended no_quote stretch (could be 60 s to many minutes,
        # depending on what gates were firing), the executor's
        # ``last_outbound_attempt_ts_mono`` has not advanced
        # because no_quote produces no orders to place. The wedge
        # detector would then see ``idle_s = now - last_attempt`` >
        # 60 s and incorrectly fire. Without this guard the v1.5.157
        # snapshot showed 14 wedges in 1.31 h (vs the genuine wedge
        # rate which should be near zero post-v1.5.150).
        #
        # ``_engine_no_quote_recovery_ts_mono`` is set when the
        # streak transitions from > 0 back to 0 (see
        # ``_maybe_emit_engine_no_quote_diagnostic``). Within the
        # grace window after recovery, skip the wedge check — give
        # the executor time to issue a fresh place / amend.
        recovery_ts = float(
            getattr(self, "_engine_no_quote_recovery_ts_mono", 0.0) or 0.0
        )
        grace_s = float(
            getattr(
                self._settings,
                "silent_wedge_no_quote_recovery_grace_seconds",
                10.0,
            )
        )
        if recovery_ts > 0.0:
            now_m_for_grace = self._clock.monotonic()
            if (now_m_for_grace - recovery_ts) < grace_s:
                return
        # Idle threshold check. v1.4.42 BUG-025-adjacent: switch from
        # the place-only ``last_place_attempt_ts_mono`` to the broader
        # ``last_outbound_attempt_ts_mono`` (places + amends, NOT
        # cancels). Same rationale as the watchdog change — see
        # watchdog.py and state.py for the full motivation.
        last_attempt_mono = float(
            getattr(self._state, "last_outbound_attempt_ts_mono", 0.0) or 0.0
        )
        if last_attempt_mono <= 0.0:
            # Fallback to legacy field for back-compat (tests).
            last_attempt_mono = float(
                getattr(self._state, "last_place_attempt_ts_mono", 0.0) or 0.0
            )
        if last_attempt_mono <= 0.0:
            # Bot has never placed in this session — pre-warm-up,
            # not a wedge.
            return
        now_m = self._clock.monotonic()
        idle_s = now_m - last_attempt_mono
        threshold_s = float(
            getattr(
                self._settings,
                "silent_wedge_detect_threshold_seconds",
                60.0,
            )
        )
        if idle_s < threshold_s:
            return
        # Re-log cadence guard (same pattern as
        # ``_maybe_emit_engine_no_quote_diagnostic``).
        relog_s = float(
            getattr(
                self._settings,
                "silent_wedge_detect_relog_seconds",
                30.0,
            )
        )
        last_log = float(self._silent_wedge_diag_last_log_mono)
        first_hit = last_log <= 0.0
        relog_due = first_hit or (now_m - last_log) >= relog_s
        if not relog_due:
            return
        self._silent_wedge_diag_last_log_mono = now_m
        # Compose payload — full executor-state snapshot so the
        # operator (and postmortem) can identify the stuck latch
        # without rerunning anything. Bug-025 reference doc has
        # the latch-class taxonomy.
        executor_state = self._build_executor_state_snapshot()
        payload: dict[str, Any] = {
            "quote_cycle_id": decision_quote_cycle_id,
            "execution_idle_s": round(idle_s, 3),
            "threshold_s": threshold_s,
            "position_qty": float(position_qty),
            "position_notional_usd": float(position_notional),
            "eligibility": eligibility,
            "active_sides": active_sides,
            "first_emission": bool(first_hit),
            "executor_state": executor_state,
        }
        # Include the engine telemetry for context (what was the
        # engine trying to do this cycle?).
        for k in (
            "quote_engine_mode",
            "quote_engine_normal_mode_requested",
            "quote_engine_bid_reason",
            "quote_engine_ask_reason",
            "quote_engine_bid_final_px",
            "quote_engine_ask_final_px",
            "quote_engine_bid_final_sz",
            "quote_engine_ask_final_sz",
        ):
            if k in telemetry:
                payload[k] = telemetry[k]
        log_extra(
            logger,
            logging.ERROR,
            "executor_silent_wedge_detected",
            payload,
        )
        try:
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.ERROR.value,
                "executor_silent_wedge_detected",
                (
                    "executor stalled while engine producing valid quotes — "
                    f"idle={round(idle_s, 1)}s, eligibility={eligibility}, "
                    "no gates active, no side_unresolved. See BUG-025."
                ),
                payload,
            )
        except Exception:
            logger.exception(
                "executor_silent_wedge_detected_event_persist_failed"
            )
        # v1.4.57 wedge-elimination Phase 3b: liveness invariant.
        # Silent-wedge detection is no longer purely a LOG event — it
        # now triggers a forced reconcile via the existing
        # ``request_open_orders_reconcile`` machinery. The reconcile
        # is the canonical recovery mechanism for "local state has
        # diverged from exchange state"; firing it once per
        # silent_wedge emission closes the loop.
        #
        # Why this works: the wedges in v1.4.52 / v1.4.53 had local
        # state showing "WO is ACKED at X" while exchange had no
        # such order (or vice versa). A reconcile re-fetches the
        # ground truth from the exchange, hydrates / clears local WOs
        # accordingly, and the next quote tick has consistent state
        # to work from.
        #
        # Gated by ``silent_wedge_force_reconcile_enabled`` (default
        # True) so an operator who finds this over-aggressive can
        # disable via env without code change.
        if bool(
            getattr(
                self._settings,
                "silent_wedge_force_reconcile_enabled",
                True,
            )
        ):
            try:
                self.request_open_orders_reconcile(
                    reason="silent_wedge_detected",
                    emergency=True,
                )
            except Exception:
                logger.exception(
                    "silent_wedge_force_reconcile_request_failed"
                )

    def _run_sub_min_notional_residual_flatten(self, quote_cycle_id: str) -> None:
        """Cancel open orders then exchange market_close for position too small to quote passively."""
        nowm = self._clock.monotonic()
        if nowm < self._sub_min_notional_flatten_next_mono:
            return
        self._sub_min_notional_flatten_next_mono = nowm + 8.0
        sym = self._settings.symbol
        with self._state._lock:
            pn = float(self._state.position.position_notional)
            pq = float(self._state.position.position_qty)
        mn = float(self._client.symbol_spec.min_notional_usd)
        force_th = float(self._settings.force_flatten_notional_usd)
        logger.warning(
            "forced_residual_flatten_start symbol=%s quote_cycle_id=%s "
            "position_qty=%s position_notional_usd=%s exchange_min_notional_usd=%s "
            "force_flatten_notional_usd=%s action=cancel_open_orders_then_market_close",
            sym,
            quote_cycle_id,
            pq,
            pn,
            mn,
            force_th,
        )
        self._storage.insert_bot_event(
            self._clock.now_utc().isoformat(),
            EventSeverity.WARNING.value,
            "forced_residual_flatten",
            "forced_residual_flatten",
            {
                "symbol": sym,
                "quote_cycle_id": quote_cycle_id,
                "position_qty": pq,
                "position_notional_usd": pn,
                "min_notional_usd": mn,
                "force_flatten_notional_usd": force_th,
            },
        )
        try:
            self.cancel_all_orders_for_symbol()
        except Exception:
            logger.exception("residual_sub_min_notional_flatten cancel_all_orders_for_symbol failed")
        try:
            resp = self._client.market_close(sym)
            st = resp.get("status") if isinstance(resp, dict) else None
            logger.info(
                "residual_sub_min_notional_flatten_market_close symbol=%s quote_cycle_id=%s "
                "response_status=%s summary=%s",
                sym,
                quote_cycle_id,
                st,
                snapshot_hl_place_response(resp) if isinstance(resp, dict) else type(resp).__name__,
            )
        except Exception:
            logger.exception(
                "residual_sub_min_notional_flatten_market_close_failed symbol=%s quote_cycle_id=%s",
                sym,
                quote_cycle_id,
            )

    def _action_materiality_allows_replace(
        self,
        cur: WorkingOrder,
        desired: FinalQuoteOrder,
        *,
        tick: float,
        side: Side,
    ) -> bool:
        eps = float(self._settings.action_reprice_epsilon_ticks)
        rs = float(self._settings.action_resize_epsilon_ratio)
        min_iv = float(self._settings.action_min_replace_interval_ms) / 1000.0
        if eps <= 0 and rs <= 0 and min_iv <= 0:
            return True
        px_ticks = abs(float(cur.price) - float(desired.price)) / max(tick, 1e-15)
        sz_rel = abs(float(cur.size) - float(desired.size)) / max(float(cur.size), 1e-15)
        material = False
        if eps > 0 and px_ticks + 1e-9 >= eps:
            material = True
        if rs > 0 and sz_rel + 1e-9 >= rs:
            material = True
        if eps <= 0 and rs <= 0:
            material = True
        elif not material:
            return False
        if min_iv > 0:
            last = float(self._last_outbound_replace_mono.get(side, 0.0))
            if last > 0 and (self._clock.monotonic() - last) + 1e-9 < min_iv:
                return False
        return True

    def _should_emit_fresh_place_intent(self, side: Side, desired: FinalQuoteOrder, tick: float) -> bool:
        eps = float(self._settings.action_reprice_epsilon_ticks)
        rs = float(self._settings.action_resize_epsilon_ratio)
        min_iv = float(self._settings.action_min_replace_interval_ms) / 1000.0
        if eps <= 0 and rs <= 0 and min_iv <= 0:
            return True
        last = self._last_emitted_fp.get(side)
        if last is None:
            return True
        lp, ls = last
        px_ticks = abs(float(desired.price) - lp) / max(tick, 1e-15)
        sz_rel = abs(float(desired.size) - ls) / max(ls, 1e-15)
        dup = True
        if eps > 0 and px_ticks + 1e-9 >= eps:
            dup = False
        if rs > 0 and sz_rel + 1e-9 >= rs:
            dup = False
        if not dup:
            return True
        if min_iv <= 0:
            return False
        lm = float(self._last_outbound_replace_mono.get(side, 0.0))
        if lm <= 0:
            return False
        return (self._clock.monotonic() - lm) + 1e-9 >= min_iv

    @staticmethod
    def _should_replace_working_order(
        current_order: WorkingOrder,
        desired_order: FinalQuoteOrder,
        *,
        replace_threshold_bps: float,
        mid_price: float,
    ) -> bool:
        """
        Order-maintenance decision helper: whether a resting order should be replaced.

        This compares current working-order price/size vs desired order and returns True
        when execution should cancel/replace. It is not quote construction logic.
        """
        order_diff_bps = (
            abs(float(current_order.price) - float(desired_order.price))
            / max(float(mid_price), 1e-12)
            * 10_000.0
        )
        order_size_rel_diff = abs(float(current_order.size) - float(desired_order.size)) / max(
            float(current_order.size), 1e-12
        )
        return not (order_diff_bps < float(replace_threshold_bps) and order_size_rel_diff <= 0.15)

    def poke_cancel_pending_recovery(self) -> None:
        """Public entry point that runs the cancel-pending timeout/retry
        machinery for both sides. Designed for callers that bypass the
        normal ``_orchestrate`` flow but still need stuck CANCEL_PENDING
        orders to recover — specifically the soft-flatten worker.

        Without this, an order that entered CANCEL_PENDING just before
        SF entry (e.g. the regular MM cycle's reprice cancel was in
        flight when toxicity-hard fired) is never resolved during SF —
        ``place_passive_order_manual_only`` refuses to send the SF
        flatten order while the same side has a CANCEL_PENDING WO,
        producing a silent deadlock until the deadlock watchdog fires
        at 600 s. Reproduced 2026-05-08, snapshot 260507112118.

        Idempotent and cheap (constant work per side) — safe to call
        every SF tick.
        """
        with self._state._lock:
            # v1.4.194: migrated off the deprecated property shims.
            wo_bid = self._state.get_working_order(Side.BUY, 0)
            wo_ask = self._state.get_working_order(Side.SELL, 0)
        self._maybe_handle_cancel_pending_timeout(Side.BUY, wo_bid)
        self._maybe_handle_cancel_pending_timeout(Side.SELL, wo_ask)

    def _reap_stale_ghosts(self) -> int:
        """v1.4.75 wedge-elimination-cleanup Phase 2B — full-state
        stale-ghost reaper.

        Walks ``state.all_working_orders()`` and force-terminals
        entries that have aged past safety thresholds without
        reaching terminal state via normal paths. Returns the
        number of WOs reaped this call.

        Three categories handled, each with its own threshold:

          1. ``CANCEL_PENDING`` aged past
             ``cancel_pending_unresolved_timeout_seconds``:
             transition to ``CANCELED`` with reason
             ``"reaper:stale_cancel_pending"`` and clear the slot
             via ``set_working_order(side, lvl, None)``.

             Why this matters: pre-Phase-2B,
             ``_maybe_handle_cancel_pending_timeout`` only inspected
             ``working_bid`` / ``working_ask`` (inside rung). Orphan-
             slot, outer-rung, and hydrated CANCEL_PENDING WOs were
             never reaped and could pin the RiskExecState in
             SUPPRESSED indefinitely. Snapshot v1.4.66-260518-192744
             caught the resulting 52 s wedge with two such ghosts
             outstanding.

          2. ``DESYNC`` aged past ``desync_reap_timeout_seconds``:
             remove from state entirely. DESYNC is a "given up"
             state (cancels already issued via exchange_mismatch
             path); after the reap timeout the tombstone serves no
             purpose and may interfere with hydration dedup.

          3. ``SENT`` aged past
             ``sent_order_unresolved_timeout_seconds * sent_reaper_safety_multiplier``:
             transition to ``REJECTED`` with reason
             ``"reaper:stale_sent_no_response"``.

             Why this matters: the legitimate SENT-resolution path
             (``_clear_sent_ambiguous_polls`` + REST poll +
             side_unresolved latch) handles ordinary slow ACKs.
             This reaper is the LAST-RESORT safety net for SENT WOs
             that slipped that path entirely (e.g., a race where
             the place response is lost AND the WS event is
             dropped AND the REST reconcile misses it).

        Called from ``maybe_refresh_quotes`` at the top of each
        tick, BEFORE the risk-action check. Runs even during
        CANCELLING / SUPPRESSED so the bot can recover from a
        wedged state. Self rate-limited (skip if last reap < 1 s
        ago) to keep per-tick cost bounded.

        Disabled when ``settings.reaper_enabled`` is False.
        """
        if not bool(getattr(self._settings, "reaper_enabled", True)):
            return 0

        now_mono = self._clock.monotonic()
        # Rate-limit: 1 reap call per second max. Cheap latch.
        if (
            now_mono - getattr(self, "_reaper_last_call_mono", 0.0)
            < _REAPER_MIN_INTERVAL_SECONDS
        ):
            return 0
        self._reaper_last_call_mono = now_mono

        cancel_pending_timeout_s = float(
            getattr(self._settings, "cancel_pending_unresolved_timeout_seconds", 0.0)
            or 0.0
        )
        desync_timeout_s = float(
            getattr(self._settings, "desync_reap_timeout_seconds", 30.0) or 30.0
        )
        sent_timeout_s = float(
            getattr(self._settings, "sent_order_unresolved_timeout_seconds", 0.0)
            or 0.0
        )
        sent_safety_mult = float(
            getattr(self._settings, "sent_reaper_safety_multiplier", 10.0) or 10.0
        )
        sent_reap_threshold_s = sent_timeout_s * sent_safety_mult

        now_utc = self._clock.now_utc()
        reaped = 0

        # Snapshot the WOs to reap UNDER lock; mutate AFTER releasing
        # so the persist-then-set path can take the lock cleanly.
        #
        # Slot index comes from ``iter_working_orders`` (the slot key
        # in the state dict), NOT from ``wo.level_idx``. The dataclass
        # field is not always populated at construction time and can
        # diverge from the actual slot the WO occupies. The slot key
        # is the authoritative answer to "which slot should I clear?"
        to_reap: list[tuple[Side, int, WorkingOrder, str]] = []
        with self._state._lock:
            wos_by_slot: list[tuple[Side, int, WorkingOrder]] = []
            for side_iter in (Side.BUY, Side.SELL):
                for idx, wo in self._state.iter_working_orders(side_iter):
                    if wo is not None:
                        wos_by_slot.append((side_iter, int(idx), wo))
            for side, lvl, wo in wos_by_slot:
                st = wo.status
                if st == OrderStatus.CANCEL_PENDING and cancel_pending_timeout_s > 0:
                    anchor = (
                        wo.ts_cancel_requested
                        or wo.ts_cancel_sent
                        or wo.ts_created
                    )
                    if anchor is None:
                        continue
                    try:
                        age_s = max(0.0, (now_utc - anchor).total_seconds())
                    except Exception:
                        continue
                    if age_s > cancel_pending_timeout_s:
                        to_reap.append((
                            side, lvl, wo,
                            f"stale_cancel_pending:age_s={round(age_s, 1)}",
                        ))
                elif st == OrderStatus.DESYNC:
                    anchor = wo.ts_closed or wo.ts_ack or wo.ts_created
                    if anchor is None:
                        continue
                    try:
                        age_s = max(0.0, (now_utc - anchor).total_seconds())
                    except Exception:
                        continue
                    if age_s > desync_timeout_s:
                        to_reap.append((
                            side, lvl, wo,
                            f"stale_desync:age_s={round(age_s, 1)}",
                        ))
                elif st == OrderStatus.SENT and sent_reap_threshold_s > 0:
                    anchor = wo.ts_sent or wo.ts_created
                    if anchor is None:
                        continue
                    try:
                        age_s = max(0.0, (now_utc - anchor).total_seconds())
                    except Exception:
                        continue
                    if age_s > sent_reap_threshold_s:
                        to_reap.append((
                            side, lvl, wo,
                            f"stale_sent_no_response:age_s={round(age_s, 1)}",
                        ))

        if not to_reap:
            return 0

        # Apply transitions OUTSIDE the iterator lock to avoid lock
        # ordering issues with persist+set_working_order.
        for side, lvl, wo, detail in to_reap:
            try:
                if wo.status == OrderStatus.CANCEL_PENDING:
                    transition(wo, OrderStatus.CANCELED, f"reaper:{detail}")
                    self.persist(wo)
                    with self._state._lock:
                        cur = self._state.get_working_order(side, lvl)
                        if cur is wo:
                            self._state.set_working_order(side, lvl, None)
                    self._reaper_cancel_pending_reaped_total += 1
                elif wo.status == OrderStatus.DESYNC:
                    # Remove from state entirely (no transition; DESYNC
                    # is the terminal-equivalent).
                    with self._state._lock:
                        cur = self._state.get_working_order(side, lvl)
                        if cur is wo:
                            self._state.set_working_order(side, lvl, None)
                    self._reaper_desync_removed_total += 1
                elif wo.status == OrderStatus.SENT:
                    transition(wo, OrderStatus.REJECTED, f"reaper:{detail}")
                    self.persist(wo)
                    with self._state._lock:
                        cur = self._state.get_working_order(side, lvl)
                        if cur is wo:
                            self._state.set_working_order(side, lvl, None)
                    self._reaper_sent_rejected_total += 1
                log_extra(
                    logger,
                    logging.WARNING,
                    "phase2b_reaper_action",
                    {
                        "side": side.value,
                        "level_idx": lvl,
                        "previous_status": wo.cancel_reason or "",
                        "current_status": wo.status.value,
                        "order_id_local": wo.order_id_local,
                        "order_id_exchange": wo.order_id_exchange,
                        "client_order_id": (
                            (wo.client_order_id[:24] + "...")
                            if wo.client_order_id and len(wo.client_order_id) > 24
                            else wo.client_order_id
                        ),
                        "detail": detail,
                    },
                )
                reaped += 1
            except Exception:
                logger.exception(
                    "reaper_action_failed side=%s lvl=%d local_id=%s",
                    side.value,
                    lvl,
                    wo.order_id_local,
                )
        return reaped

    def _maybe_handle_cancel_pending_timeout(self, side: Side, cur: Optional[WorkingOrder]) -> None:
        if cur is None or cur.status != OrderStatus.CANCEL_PENDING:
            self._cancel_pending_since_mono[side] = None
            # Reset only THIS side's retry bookkeeping. Previous implementation
            # wiped the whole dict whenever ``cur=None``, which in a
            # two-side orchestration (cleanup runs for the empty side on the
            # same tick the other side is actively retrying) reset the
            # other side's rate-limit to 0 and produced a cancel storm.
            self._cancel_pending_retry_last_mono[side] = 0.0
            self._cancel_pending_retry_attempts[side] = 0
            if self._side_unresolved_reason.get(side) == "cancel_pending_wait":
                self._clear_side_unresolved(side, reason="cancel_pending_not_active")
            return
        started = self._cancel_pending_since_mono.get(side)
        if started is None:
            started = self._clock.monotonic()
            self._cancel_pending_since_mono[side] = started
        age_s = max(0.0, self._clock.monotonic() - float(started))
        timeout_s = float(self._settings.cancel_pending_unresolved_timeout_seconds)
        if age_s < timeout_s:
            return
        self._cancel_pending_timeout_count += 1
        self._set_side_unresolved(
            side,
            reason="cancel_pending_timeout",
            payload={
                "age_s": age_s,
                "timeout_s": timeout_s,
                "order_id_local": cur.order_id_local,
                "order_id_exchange": cur.order_id_exchange,
                "client_order_id": cur.client_order_id,
            },
        )
        # Per-(side, oid) dedup for the recovery WARNING. Once logged
        # for a specific stuck order we suppress re-logs until the oid
        # changes (new cancel episode) or the side clears. The
        # ``_set_side_unresolved`` call above already transitions-only
        # its own log. Observed: 3,167 duplicate WARNING lines in the
        # 4-minute flooded run for just 2 stuck orders; this guard
        # reduces that to 2 first-time logs + a handful for any retries
        # that produce a new oid.
        last_logged_oid = self._cancel_pending_timeout_logged_oid.get(side)
        if last_logged_oid != cur.order_id_exchange:
            self._cancel_pending_timeout_logged_oid[side] = cur.order_id_exchange
            log_extra(
                logger,
                logging.WARNING,
                "cancel_pending_timeout_recovery",
                {
                    "side": side.value,
                    "age_s": age_s,
                    "timeout_s": timeout_s,
                    "order_id_local": cur.order_id_local,
                    "order_id_exchange": cur.order_id_exchange,
                },
            )
        self.request_open_orders_reconcile(
            reason=f"cancel_pending_timeout_{side.value.lower()}",
            force=True,
            emergency=True,
        )
        # Re-issue the HTTP cancel itself. The reconcile above only *observes*
        # exchange state; if GRVT returned a success ack on the original cancel
        # but the order never actually left the book (or the terminal WS event
        # was dropped), the observe-only path leaves the local WO stuck in
        # CANCEL_PENDING forever, latches the side as unresolved, and the
        # whole side stops quoting. Retrying is idempotent (cancel-by-oid or
        # cancel-by-cloid+TTL both handle a no-op gracefully) and closes the
        # loop either by actually cancelling or by producing a "not found"
        # that the next reconcile turns into a terminal CANCELED locally.
        retry_interval_s = max(2.0, timeout_s / 3.0)
        now_mono = self._clock.monotonic()
        last_retry = self._cancel_pending_retry_last_mono.get(side, 0.0)
        if (now_mono - last_retry) + 1e-9 >= retry_interval_s:
            self._cancel_pending_retry_last_mono[side] = now_mono
            self._cancel_pending_retry_dispatch_count += 1
            # Count attempts for THIS CANCEL_PENDING episode. After
            # ``_CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS`` oid-cancels
            # have been acked-but-ineffective, escalate to cancel-by-cloid
            # (with GRVT's ``time_to_live_ms``). Observed on GRVT: the oid
            # path can return ``ack=true`` without the matching engine
            # actually cancelling; the cloid path goes through a different
            # handler and has been effective in similar scenarios.
            self._cancel_pending_retry_attempts[side] += 1
            attempts_this_episode = self._cancel_pending_retry_attempts[side]
            # 1.3.117: per-profile escape hatch — disable cloid
            # escalation on OKX colo (see config.py docstring + the
            # ``plans/_DONE/sbe.md`` §2 note on the ``instIdCode``
            # vs ``instId`` colo-WS contract).
            escalation_enabled = bool(
                getattr(
                    self._settings,
                    "cancel_pending_cloid_escalation_enabled",
                    True,
                )
            )
            use_cloid = (
                escalation_enabled
                and attempts_this_episode > _CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS
                and bool(cur.client_order_id)
            )
            log_extra(
                logger,
                logging.WARNING,
                "cancel_pending_retry_dispatched",
                {
                    "side": side.value,
                    "order_id_local": cur.order_id_local,
                    "order_id_exchange": cur.order_id_exchange,
                    "age_s": age_s,
                    "retry_interval_s": retry_interval_s,
                    "retry_count_total": self._cancel_pending_retry_dispatch_count,
                    "attempts_this_episode": attempts_this_episode,
                    "escalated_to_cloid": use_cloid,
                },
            )
            # Synchronous — cheap HTTP call, rate-limited above so the retry
            # loop can't saturate the GRVT cancel endpoint.
            try:
                self._cancel_http_transport(cur, force_cloid=use_cloid)
            except Exception:
                logger.exception(
                    "cancel_pending_retry_dispatch_failed side=%s order_id_local=%s",
                    side.value,
                    cur.order_id_local,
                )

    def maybe_refresh_quotes(
        self,
        decision: QuoteDecision,
        risk_action: RiskAction,
        bid_mult: float,
        ask_mult: float,
        spread_add_bps: float,
        *,
        cancel_on_no_quote: bool = False,
    ) -> None:
        """Reconcile working orders with the latest desired quotes from ``QuoteEngine``.

        Execution only: snapshot state → ``build_quotes`` → optional residual flatten action
        → diff/cancel/replace → stage new rests + enqueue transport (non-blocking). It does **not**
        normalize, re-round, or apply inventory policy to prices/sizes (that is all in
        ``QuoteEngine``). Placement order is fixed (BUY then SELL) and does not encode strategy.
        """
        # ORDER MAINTENANCE LAYER
        # This function does NOT construct or modify quotes.
        # It only compares current working orders vs desired orders from QuoteEngine
        # and decides whether to keep, cancel, replace, or place orders.
        self._last_tick_order_submit_rtt_ms = 0.0
        self._first_enqueue_perf = None
        self._last_decision_to_submit_dispatch_ms = 0.0
        self._last_submit_queue_wait_ms = 0.0
        self._last_decision_to_first_submit_dispatch_ms = 0.0
        t0 = time.perf_counter()
        self._maybe_refresh_t0_perf = t0
        # v1.4.75 Phase 2B: run the stale-ghost reaper BEFORE the
        # risk-action check. The reaper must operate regardless of
        # risk state — its whole purpose is to clean residual ghosts
        # so the SUPPRESSED state machine can transition out. Rate-
        # limited internally; very cheap on the happy path
        # (state.all_working_orders() is small).
        try:
            self._reap_stale_ghosts()
        except Exception:
            logger.exception("reap_stale_ghosts_failed_continuing")
        # v1.4.92 Phase 4A cutover — Phase 2D per-tick reset removed
        # along with the rest of the audit infrastructure.
        if not self._client.has_write_access():
            # v1.4.53: instrument early-return so the wedge postmortem
            # sees which gate held. ``_orchestrate`` is skipped → would
            # otherwise leave no breadcrumb.
            self._record_quote_refresh_skip(
                reason="no_write_access",
                extra={"quote_cycle_id": decision.quote_cycle_id},
            )
            return
        if self.cancel_resting_for_risk(risk_action):
            self._record_quote_refresh_skip(
                reason="cancel_resting_for_risk",
                extra={
                    "risk_action": risk_action.value if hasattr(risk_action, "value") else str(risk_action),
                    "quote_cycle_id": decision.quote_cycle_id,
                },
            )
            return
        # Binance Level 1: cross-venue cancel trigger. Fires BEFORE the
        # standard reprice logic so Binance-driven cancels go out without
        # waiting on GRVT's own BBO to catch up. No-op when Binance is
        # disabled, disconnected, or stale.
        self._maybe_cancel_on_binance_move()
        # v1.4.165 Phase 4E — target-venue fast-move cancel. Mirrors
        # the reference-venue cancel above but watches the LOCAL
        # venue's own 500 ms kinematic — catches fast moves that
        # didn't originate on the reference venue. No-op when the
        # threshold is unset (default).
        self._maybe_cancel_on_target_venue_fast_move()
        if risk_action == RiskAction.NO_QUOTE and not cancel_on_no_quote:
            self._quote_exec_telemetry = {
                "exec_mode": "no_quote_hold_resting",
                "decision_to_submit_dispatch_ms": 0.0,
                "submit_queue_wait_ms": 0.0,
                "submit_transport_rtt_ms": 0.0,
                "decision_to_first_submit_dispatch_ms": 0.0,
                "ack_resolution_ms": self._last_ack_resolution_ms,
            }
            # v1.4.53: this is the suspected wedge-class trigger. When
            # the bot's risk decision returns NO_QUOTE and
            # ``cancel_on_no_quote`` is False, ``_orchestrate`` is
            # skipped — existing WOs sit untouched and no fresh places
            # happen. If risk_action stays NO_QUOTE across many ticks
            # while engine + eligibility are healthy, that's the wedge.
            self._record_quote_refresh_skip(
                reason="no_quote_hold_resting",
                extra={
                    "risk_action": risk_action.value if hasattr(risk_action, "value") else str(risk_action),
                    "cancel_on_no_quote": bool(cancel_on_no_quote),
                    "quote_cycle_id": decision.quote_cycle_id,
                },
            )
            return

        # v1.4.87 wedge-elimination-cleanup Phase 5A — single-tick state
        # snapshot. Pre-Phase-5A this block opened ``state._lock``
        # explicitly to copy out 5 separate fields (pos, market, wb, wa,
        # iter_working_orders). The ``state.tick_snapshot()`` factory
        # shipped in Phase 3C does the same thing in ONE lock acquisition
        # and returns an immutable composite view — so callees can read
        # everything without re-touching the lock or risking torn reads
        # across WS-handler interleavings.
        #
        # The lock-count regression test
        # (``tests/test_phase5a_one_snapshot_per_tick.py``) instruments
        # ``state._lock`` and asserts the hot path takes it exactly once
        # per tick. Codex F9 ("7 acquisitions on the happy path") was
        # the original motivator.
        #
        # Note: ``OrderStore`` already filters terminal WOs out of its
        # active-set iteration, so the ``OrderStatus in {CANCELED,
        # FILLED, REJECTED, DESYNC}`` check that lived inside this
        # block is gone — ``tick.orders.by_slot`` is already filtered
        # to live WOs by construction.
        _tick = self._state.tick_snapshot()
        pn = float(_tick.position.notional_abs_usd)
        pos_qty = float(_tick.position.qty)
        mkt = _tick.market_raw  # raw BestBidAsk reference, captured under lock
        wb = _tick.orders.slot(Side.BUY, 0)
        wa = _tick.orders.slot(Side.SELL, 0)
        slot_map: dict[tuple[Side, int], WorkingOrder] = {
            (s, lvl): wo for s, lvl, wo in _tick.orders.by_slot
            if wo.status not in (
                OrderStatus.CANCELED,
                OrderStatus.FILLED,
                OrderStatus.REJECTED,
                OrderStatus.DESYNC,
            )
        }
        # Phase 4G.4 (v1.4.210) — regime-conditional effective inventory
        # budget. Read ``inventory_budget_mult`` from the per-tick regime
        # knobs and pass the shrunken cap through QuoteBuildContext so
        # ``_clip_entry_sizes`` + ``_inventory_exec_bias_active`` see it.
        # When the multiplier is 1.0 (NORMAL/CALM defaults) we pass
        # ``None`` so the engine takes its pre-4G backward-compat branch
        # and reads ``settings.max_abs_position`` directly.
        #
        # NOT consumed by the SF flatten clips or the hard risk-kill
        # check — those keep the raw ``settings.max_abs_position`` so the
        # bot can ALWAYS flatten regardless of regime knobs (rule lives
        # inside the QuoteBuildContext docstring and the place-time
        # consumers).
        _inv_budget_mult = float(
            getattr(self._state.regime_knobs, "inventory_budget_mult", 1.0) or 1.0
        )
        if _inv_budget_mult != 1.0 and _inv_budget_mult > 0.0:
            _effective_max_abs_position: Optional[float] = float(
                self._settings.max_abs_position
            ) * _inv_budget_mult
        else:
            _effective_max_abs_position = None
        build = self._quote_engine.build_quotes(
            QuoteBuildContext(
                decision=decision,
                market=mkt,
                risk_action=risk_action,
                bid_mult=bid_mult,
                ask_mult=ask_mult,
                spread_add_bps=spread_add_bps,
                position_qty=pos_qty,
                position_notional=pn,
                resting_bid=wb,
                resting_ask=wa,
                reprice_replace_pending_bid=self._quote_reprice_replace_pending.get(Side.BUY, False),
                reprice_replace_pending_ask=self._quote_reprice_replace_pending.get(Side.SELL, False),
                working_orders_by_slot=slot_map,
                effective_max_abs_position=_effective_max_abs_position,
            )
        )
        # Residual flatten: execution action from engine decision (not a second quoting pass).
        # v1.4.92 Phase 4A: ``build`` is now a ``BuildCommand`` variant.
        # The ``residual_flatten_requested`` property is True ONLY on
        # the ``ResidualFlatten`` variant; other variants return False
        # via the legacy-compat property.
        if build.residual_flatten_requested:
            self._record_quote_refresh_skip(
                reason="residual_flatten_requested",
                extra={
                    "quote_cycle_id": decision.quote_cycle_id,
                    "position_qty": float(pos_qty),
                    "position_notional_usd": float(pn),
                },
            )
            self._run_sub_min_notional_residual_flatten(decision.quote_cycle_id)
            return
        self._quote_exec_telemetry = dict(build.telemetry)
        # Persistent no-quote diagnostic (2026-05-08).
        # Detects silent wedges where the engine produces no orders for
        # many consecutive ticks but no other event surfaces the cause.
        # Fires a WARNING event after the configured streak threshold;
        # re-emits at a slower cadence while still stuck. Reset on any
        # tick that produces at least one order.
        self._maybe_emit_engine_no_quote_diagnostic(
            telemetry=build.telemetry,
            decision_quote_cycle_id=decision.quote_cycle_id,
            position_qty=pos_qty,
            position_notional=pn,
        )
        # v1.4.40 BUG-025: silent-wedge detector. Catches the
        # inverse failure mode of ``engine_no_quote_persistent`` —
        # engine is healthy, eligibility is clean, no gates fire,
        # but execution_idle climbs without bound. Fires only when
        # all five conjoint conditions hold (see method docstring),
        # so no false-positive risk.
        try:
            self._maybe_emit_silent_wedge_diagnostic(
                telemetry=build.telemetry,
                decision_quote_cycle_id=decision.quote_cycle_id,
                eligibility=str(decision.quote_eligibility or ""),
                active_sides=str(
                    decision.active_sides.value
                    if hasattr(decision.active_sides, "value")
                    else decision.active_sides
                    or ""
                ),
                position_qty=pos_qty,
                position_notional=pn,
            )
        except Exception:
            # Detector failures must never affect quoting — log and
            # continue.
            logger.exception("silent_wedge_diag_failed")
        # Volatility-adaptive reprice threshold (opt-in via
        # ``REPRICE_THRESHOLD_VOL_MULTIPLIER > 0``). In quiet markets the
        # effective threshold shrinks toward ``reprice_threshold_bps_min`` so
        # tiny moves trigger a re-quote (tighten queue position); in volatile
        # markets it stays at the configured ceiling to avoid cancel/replace
        # churn. Default multiplier 0.0 preserves the fixed-threshold behaviour.
        vol_mult = float(self._settings.reprice_threshold_vol_multiplier)
        if vol_mult > 0.0:
            vol_now = float(getattr(decision, "vol_estimate", 0.0) or 0.0)
            floor_bps = float(self._settings.reprice_threshold_bps_min)
            ceil_bps = float(self._settings.reprice_threshold_bps)
            raw = vol_mult * vol_now
            replace_threshold_bps = min(ceil_bps, max(floor_bps, raw))
        else:
            replace_threshold_bps = float(self._settings.reprice_threshold_bps)
        mid = float(decision.mid_price) if decision.mid_price > 0 else 1e-12
        tick = float(self._client.symbol_spec.price_tick) if self._client.symbol_spec.price_tick > 0 else 1e-12

        # v1.4.60 wedge-elimination Phase 5: reconciler-driven dispatch.
        #
        # Pre-v1.4.60 the dispatch flow was a nested ``_orchestrate``
        # closure called per-rung from two parallel paths (multi-rung
        # ladder + single-rung scalar). Each call did "decide AND
        # dispatch" in one function with ~13 branches. v1.4.60 splits
        # this into three pure layers:
        #
        #   1. ``compute_desired_state(...)`` — pure function over the
        #      engine output + ladder + position → DesiredLadderState.
        #      Applies the reducing-side-bypass invariant as the last
        #      fold.
        #   2. ``reconcile(desired, current)`` — pure diff → Action list
        #      (PlaceAction / AmendAction / CancelAction / NoOpAction).
        #   3. ``_dispatch_action(action, ...)`` — side-effect layer.
        #      Routes via the existing transport helpers; same wire
        #      behaviour as the pre-v1.4.60 closure.
        #
        # See ``app/reconciler.py`` for the pure-module spec + tests.
        ladder = getattr(self._state, "last_ladder_decision", None)
        # Phase 4G.8 (v1.4.219) — honour the regime knob's ladder cap.
        # bot.py applies the cap when calling build_ladder; mirror it
        # here so compute_desired_state iterates the same range and
        # the reconciler emits cancels for orphan rungs left over
        # from a prior NORMAL tick. Single source of truth =
        # min(config, regime knob). Floored at 1 for SHOCK back-
        # compat UNLESS Phase 4G.10's ``SHOCK_LADDER_ALLOW_FULL_DARK``
        # flag is set, in which case cap=0 is honoured and the
        # reconciler will cancel any inside rungs left from a prior
        # mode. See ``app/bot.py``'s mirror.
        _cfg_levels = int(
            getattr(self._settings, "ladder_num_levels_per_side", 1)
        )
        _regime_levels_cap = getattr(
            self._state.regime_knobs, "ladder_levels_max", None
        )
        _allow_full_dark_exec = bool(
            getattr(
                self._settings,
                "shock_ladder_allow_full_dark",
                False,
            )
        )
        if _regime_levels_cap is not None:
            _knob_capped = min(_cfg_levels, int(_regime_levels_cap))
            if _allow_full_dark_exec:
                num_levels_per_side = max(0, _knob_capped)
            else:
                num_levels_per_side = max(1, _knob_capped)
        else:
            num_levels_per_side = max(1, _cfg_levels)
        # _snapshot_working_orders_for_reconciler is called with the
        # capped value but still scans for orphan rungs beyond the
        # cap (its own loop at line 10462) so cleanup of post-
        # transition rungs (NORMAL→CAUTIOUS with rung-1 alive)
        # works correctly.
        # Reconciler-driven path. Replaces both the multi-rung ladder
        # dispatch and the single-rung scalar path (they collapse into
        # one flow: desired-state per slot → reconcile → dispatch).
        client_spec = self._client.symbol_spec
        post_gates_desired, engine_desired, ladder_rung_rejections = (
            self.compute_desired_state(
                build=build,
                decision=decision,
                ladder=ladder,
                position_qty=pos_qty,
                num_levels=num_levels_per_side,
                client_spec=client_spec,
            )
        )
        current_snapshot = self._snapshot_working_orders_for_reconciler(
            num_levels_per_side
        )
        actions = reconcile(
            desired=post_gates_desired,
            current=current_snapshot,
            materiality_check=lambda c, d: self._reconcile_materiality_check(
                c, d,
                replace_threshold_bps=replace_threshold_bps,
                mid_price=mid,
                tick=tick,
            ),
            amend_viable_check=self._reconcile_amend_viable_check,
        )
        for action in actions:
            self._dispatch_action(
                action,
                decision=decision,
                mid_price=mid,
                tick=tick,
            )
        # ladder_dispatch_no_levels diagnostic (kept for postmortem):
        # surfaces "engine produced no rung targets AND no working
        # orders on this side" — the v1.4.57 always-iterate invariant
        # means this is rarely interesting but the rejection-reason
        # detail still helps when normalize_order_pair rejects all
        # rungs (eg sub-min-notional after rounding).
        if ladder_rung_rejections:
            self._record_quote_refresh_skip(
                reason="ladder_rung_normalize_rejected",
                extra={
                    "ladder_rung_rejections": ladder_rung_rejections,
                    "quote_cycle_id": decision.quote_cycle_id,
                },
            )
        if self._first_enqueue_perf is not None:
            self._last_decision_to_submit_dispatch_ms = (
                self._first_enqueue_perf - self._maybe_refresh_t0_perf
            ) * 1000.0
            with self._state._lock:
                qd = self._state.quote_decision_perf_counter
            if qd is not None:
                self._last_decision_to_first_submit_dispatch_ms = max(
                    0.0, (self._first_enqueue_perf - qd) * 1000.0
                )
        self._quote_exec_telemetry["decision_to_submit_dispatch_ms"] = float(
            self._last_decision_to_submit_dispatch_ms
        )
        self._quote_exec_telemetry["submit_queue_wait_ms"] = float(self._last_submit_queue_wait_ms)
        self._quote_exec_telemetry["submit_transport_rtt_ms"] = float(
            self._last_tick_order_submit_rtt_ms
        )
        self._quote_exec_telemetry["decision_to_first_submit_dispatch_ms"] = float(
            self._last_decision_to_first_submit_dispatch_ms
        )
        self._quote_exec_telemetry["ack_resolution_ms"] = self._last_ack_resolution_ms
        self._last_tick_quote_engine_build_ms = float(
            self._quote_exec_telemetry.get("quote_contract_build_ms") or 0.0
        )
        self._quote_exec_telemetry["placement_mode_eval_ms"] = max(
            0.0, (time.perf_counter() - t0) * 1000.0
        )
        self._quote_exec_telemetry["finalize_ms"] = 0.0
        self._quote_exec_telemetry["order_submit_prep_ms"] = 0.0
        self._record_quote_cycle_telemetry(
            decision,
            want_bid=build.bid_order is not None or decision.active_sides in (ActiveSides.BOTH, ActiveSides.BID_ONLY),
            want_ask=build.ask_order is not None or decision.active_sides in (ActiveSides.BOTH, ActiveSides.ASK_ONLY),
            can_bid=build.bid_order is not None,
            can_ask=build.ask_order is not None,
            tick=tick,
        )
        # v1.4.92 Phase 4A cutover — Phase 2D end-of-tick validation
        # removed. The typed ``BuildCommand`` sum-type prevents the
        # v1.4.66 regression class structurally (new fields on a
        # variant are explicit dataclass attributes; missing variant
        # cases in match consumers are type errors at edit time).
