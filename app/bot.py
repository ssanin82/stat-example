from __future__ import annotations

import json
import logging
import math
import queue
import statistics
import threading
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from app.clock import Clock

from app.config import Settings
from app.persistent_runtime_io import try_save_persistent_runtime_state
from app.enums import (
    ActiveSides,
    BotStatus,
    EventSeverity,
    FlattenResult,
    OrderStatus,
    QuoteEligibility,
    RiskAction,
    Side,
)
from app.execution import OrderManager
from app.exchange.base import PerpExchangeAdapter
from app.exchange.factory import venue_account_address
from app.inventory_consistency import (
    InventoryBreach,
    check_inventory_consistency,
)
from app.markout import process_pending_markouts
from app.market_data import refresh_account_only
from app.pnl import PnlTracker
from app import position_drawdown_gate, soft_flatten, take_profit
from app.quote_eligibility import (
    QuoteEligibilityResult,
    apply_recovery_cooldown,
    compute_quote_eligibility,
    effective_staleness_ms_from_exchange_ts,
    maybe_arm_recovery_cooldown,
)
from app.quoting import (
    adverse_spread_widen_arm,
    apply_quote_eligibility_to_decision,
    build_spread_composition,
    compute_quote_decision,
)
from app.risk import evaluate_risk
from app.state import BotState
from app.storage import Storage
from app.toxicity import ToxicityEngine
from app.utils.logfmt import log_extra
from app.utils.time import seconds_since
from app.models import BestBidAsk, QuoteDecision, RiskDecision
from app.volatility import VolatilityEstimator

logger = logging.getLogger(__name__)

def _effective_book_age_ms_at_decision(m: BestBidAsk | None, now_utc: datetime) -> Optional[float]:
    if m is None:
        return None
    if m.ts_exchange_ms is not None:
        ev_dt = datetime.fromtimestamp(m.ts_exchange_ms / 1000.0, tz=timezone.utc)
        return max(0.0, (now_utc - ev_dt).total_seconds() * 1000.0)
    if m.ts_local:
        return max(0.0, (now_utc - m.ts_local).total_seconds() * 1000.0)
    return None


def _decision_market_data_regime(state: BotState) -> str:
    with state._lock:
        st = state.bot_status
        stale_warn = state.stale_book_warning_active
        stall = state.market_data_stall_latched
    if st == BotStatus.RECOVERING_MARKET_DATA:
        return "recovery"
    if stale_warn or stall:
        return "degraded"
    return "ok"


# Sampling cadence for ``quote_eligibility_tick_sample`` — emits the full
# eligibility diagnostic payload every N ticks (~5s at QUOTE_LOOP_SECONDS=0.5)
# so a quietly-clamped state isn't invisible in the log.
_QUOTE_ELIGIBILITY_TICK_SAMPLE_EVERY_N = 10


_QUOTE_EXEC_COLS = (
    "exec_raw_bid_px",
    "exec_raw_bid_sz",
    "exec_raw_ask_px",
    "exec_raw_ask_sz",
    "exec_norm_bid_px",
    "exec_norm_bid_sz",
    "exec_norm_ask_px",
    "exec_norm_ask_sz",
    "exec_price_tick",
    "exec_size_step",
    "exec_meta_decimal_grid_price_tick",
    "exec_meta_decimal_size_step",
    "exec_hl_max_sig_figs_nonint",
    "exec_price_normalize_pipeline",
    "exec_wire_bid_limit_p",
    "exec_wire_ask_limit_p",
)


def quote_decision_row(
    d: QuoteDecision,
    quote_cycle_id: str,
    *,
    ladder: Any = None,
) -> dict:
    """Serialise a QuoteDecision into the storage row format.

    1.2.14: optional ``ladder`` parameter accepts a
    ``app.ladder.LadderDecision``; when provided, the resulting
    row carries the per-rung ladder + diagnostics in the
    ``ladder_*`` columns. When None (default — current
    single-level callers), those columns are NULL. This keeps
    existing call sites unchanged.
    """
    row = {
        "ts": d.ts.isoformat(),
        "symbol": d.symbol,
        "mid_price": d.mid_price,
        "vol_estimate": d.vol_estimate,
        "inventory": d.inventory,
        "reservation_price": d.reservation_price,
        "target_spread_bps": d.target_spread_bps,
        "target_bid": d.target_bid,
        "target_ask": d.target_ask,
        "quoted_bid": d.quoted_bid,
        "quoted_ask": d.quoted_ask,
        "quoted_bid_sz": d.quoted_bid_sz,
        "quoted_ask_sz": d.quoted_ask_sz,
        "active_sides": d.active_sides.value,
        "toxicity_score": d.toxicity_score,
        "spread_floor_overlay_half_spread_bps": d.spread_floor_overlay_half_spread_bps,
        "decision_reason": d.decision_reason,
        "quote_cycle_id": quote_cycle_id,
        "quote_eligibility": d.quote_eligibility,
        "quote_eligibility_reason": d.quote_eligibility_reason,
        "source_book_ts_exchange_ms": d.source_book_ts_exchange_ms,
        "source_book_ts_local_iso": d.source_book_ts_local_iso,
        "effective_book_age_at_decision_ms": d.effective_book_age_at_decision_ms,
        "book_apply_to_decision_ms": d.book_apply_to_decision_ms,
        "public_ws_queue_wait_ms_latest": d.public_ws_queue_wait_ms_latest,
        "public_ws_receive_to_apply_ms_latest": d.public_ws_receive_to_apply_ms_latest,
        "decision_market_data_regime": d.decision_market_data_regime,
        # Microprice reference value (None when depth unavailable or
        # the formula evaluated to non-finite). Persisted even when the
        # reservation fell back to midprice — enables shadow-mode
        # comparison of "what would the reservation have been if the
        # flag were flipped?"
        "microprice": d.microprice,
        # Binance L1 cross-venue reference at decision time (None when
        # BINANCE_WS_ENABLED=false, first message not yet received, or
        # basis EWMA not seeded). Paired with the
        # ``binance_cross_venue_cancel`` bot_event, these support
        # per-cycle shadow analysis — "at the moment we built this
        # quote, what was Binance's mid and how stale was our view
        # relative to it?"
        "binance_mid": d.binance_mid,
        "binance_basis_ewma": d.binance_basis_ewma,
        # N2 analysis-day instrumentation: basis-regime classifier
        # snapshot at decision time. Persisted alongside the rest of
        # the quote-cycle context so postmortem reports can build a
        # regime-sign time-series and re-tune the IC threshold.
        "basis_regime_sign": d.basis_regime_sign,
        "basis_ic": d.basis_ic,
        "basis_pair_count": d.basis_pair_count,
    }
    # 1.2.14: ladder fields. Default to NULL when no ladder was
    # built (legacy callers). When the bot's quote-cycle path
    # passes the LadderDecision, fields are JSON-encoded for the
    # array columns and integers for the count diagnostics.
    if ladder is not None:
        import json as _json
        ladder_dict = ladder.to_dict()
        row["ladder_bids"] = _json.dumps(ladder_dict["bids"], separators=(",", ":"))
        row["ladder_asks"] = _json.dumps(ladder_dict["asks"], separators=(",", ":"))
        row["ladder_requested_levels"] = int(ladder_dict["requested_levels"])
        row["ladder_effective_levels_buy"] = int(ladder_dict["effective_levels_buy"])
        row["ladder_effective_levels_sell"] = int(ladder_dict["effective_levels_sell"])
        row["ladder_gate_caps"] = (
            _json.dumps(ladder_dict["gate_caps"], separators=(",", ":"))
            if ladder_dict["gate_caps"]
            else None
        )
    else:
        row["ladder_bids"] = None
        row["ladder_asks"] = None
        row["ladder_requested_levels"] = None
        row["ladder_effective_levels_buy"] = None
        row["ladder_effective_levels_sell"] = None
        row["ladder_gate_caps"] = None
    for k in _QUOTE_EXEC_COLS:
        row[k] = None
    return row


def quote_decision_row_with_exec_telemetry(
    decision: QuoteDecision,
    quote_cycle_id: str,
    order_manager: OrderManager,
    risk: RiskDecision,
    *,
    ladder: Any = None,
) -> dict:
    """Quote row with raw/normalized execution preview and symbol spec grid (same as live placement)."""
    row = quote_decision_row(decision, quote_cycle_id, ladder=ladder)
    row.update(
        order_manager.build_quote_execution_telemetry(
            decision,
            risk.bid_size_mult,
            risk.ask_size_mult,
            risk.spread_add_bps,
        )
    )
    return row


# Kill reasons that imply the bot is BLIND — i.e. the public market feed
# is unreliable or absent, so we cannot price an exit order. On these
# kills we cancel outstanding orders (safe: no pricing required) but do
# NOT market-flatten, because a market order would cross into a book we
# cannot see. The operator regains control with the position intact and
# can ``/control/flatten`` once market data has recovered.
#
# Motivation: ``tmp/snap_20260418_170340`` — a flash-crash caused the
# public WS to fall behind; stale-data recovery exhausted its 15 s
# budget, the bot self-killed with reason ``stale_data_kill_escalated``,
# and the default ``flatten_on_kill=True`` policy immediately issued a
# 37-AXS ``market_close`` at the crash low. The flatten was executed as
# a TAKER fill (flatten path uses ``client.market_close``, not the
# post-only ``place_post_only_limit`` quoting path) — paying a $0.025
# taker fee on $41 notional and realising the trough as the fill price.
# The kill itself was correct; the follow-on flatten was self-harm.
_BLIND_KILL_REASONS: frozenset[str] = frozenset(
    {
        "stale_data_kill",
        "stale_data_kill_threshold",
        "stale_data_kill_escalated",
        "public_ws_disconnected",
        "public_ws_message_gap",
        "public_ws_stale_kill",
    }
)


# BUG-013: kill reasons that are environmental / transient — the bot
# should exit with the same non-zero code the deadlock watchdog uses so
# the process supervisor (systemd ``Restart=on-failure`` / k8s
# ``restartPolicy=OnFailure`` / container manager) brings the process
# back automatically.
# Without this, ``bot.kill()`` previously left the process alive in
# KILLED state forever (snap_20260427_040436: bot killed itself on a
# 9-min Bluefin WS stall and sat dead-but-alive for 2.5 h until the
# operator noticed).
#
# Reasons NOT in this set (manual /kill, drawdown, session loss, desync,
# execution errors) leave the bot KILLED-alive for human review — those
# are real safety stops where auto-restart would mask a problem.
_AUTO_RESTART_KILL_REASONS = frozenset({
    "stale_data_kill",
    "stale_data_kill_escalated",
    "public_ws_stale_kill",
    "public_ws_disconnected",
    "public_ws_message_gap",
    "no_market_data",
    # Account-data freshness gate (TODO-002): same family — auto-recover.
    "account_data_stale",
    # Reconcile stalled because the venue endpoint was unhealthy; once a
    # fresh process re-establishes connections it usually clears.
    "reconcile_stall",
})


def _kill_reason_should_auto_restart(reason: str) -> bool:
    """BUG-013: True iff EVERY comma-separated token in ``reason`` is in
    ``_AUTO_RESTART_KILL_REASONS``.

    Conservative-by-default: a kill that bundles a recoverable token
    (e.g. ``stale_data_kill``) WITH a non-recoverable one (e.g. a
    drawdown breach) does NOT auto-restart — the operator should look
    at the non-recoverable cause first.
    """
    tokens = [t.strip() for t in reason.split(",") if t.strip()]
    if not tokens:
        return False
    return all(t in _AUTO_RESTART_KILL_REASONS for t in tokens)


def _kill_reason_is_blind(reason: str) -> bool:
    """True if any comma-separated token in ``reason`` indicates that
    public market data is unreliable.

    The ``Bot.kill(reason, ...)`` contract accepts either a single
    reason string or a comma-joined list (see ``bot.py:1229`` and
    ``bot.py:1528`` where risk-module reasons are joined before being
    passed in). We conservatively treat the kill as blind if **any**
    token matches — if we can't see the market, no other reason on the
    list justifies a market-flatten.
    """
    if not reason:
        return False
    tokens = {t.strip() for t in reason.split(",") if t.strip()}
    return bool(tokens & _BLIND_KILL_REASONS)


class Bot:
    def __init__(
        self,
        settings: Settings,
        state: BotState,
        client: PerpExchangeAdapter,
        storage: Storage,
        private_event_queue: Optional[queue.Queue] = None,
        private_stream: Any = None,
        public_stream: Any = None,
        notifier: Any = None,
        exit_fn: Optional[Any] = None,
        clock: Optional["Clock"] = None,
    ) -> None:
        self._settings = settings
        self._state = state
        self._client = client
        self._storage = storage
        self._private_stream = private_stream
        self._public_stream = public_stream
        # Phase 1a (v1.4.229) — Clock abstraction. Default to
        # ``SystemClock`` so production behaviour is unchanged
        # (behaviourally identical to direct stdlib monotonic calls).
        # Phase 1b (v1.4.230) migrated the bot.py + execution.py
        # hot-path call sites to ``self._clock.<method>``.
        # Phase 1c (v1.4.231) covered free functions + helper
        # classes via the module-level clock proxy in ``app.clock``;
        # we install our own clock as the module-active one here so
        # the bot's method callers and free-function callers share
        # the same clock instance. Backtesting replay swaps in a
        # ``ReplayClock`` before Bot.__init__ runs.
        if clock is None:
            from app.clock import SystemClock
            clock = SystemClock()
        self._clock: "Clock" = clock
        # Install as the module-level active clock for free-function
        # callers (utils/time.py helpers, gate evaluators, etc.).
        # When tests construct multiple Bot instances sequentially,
        # the last-constructed one's clock becomes the active one —
        # acceptable because the bot is a singleton in production.
        from app.clock import set_module_clock
        set_module_clock(self._clock)
        # Telegram notifier reference. Optional; ``None`` disables
        # outbound forwarding from ``_log_event``. The notifier itself
        # is fire-and-forget (enqueue, return immediately) so this hook
        # cannot stall the trading hot path.
        self._notifier = notifier
        # BUG-013: process-exit hook used by ``kill()`` when the kill
        # reason is in ``_AUTO_RESTART_KILL_REASONS``. Mirrors the
        # watchdog's pattern (app/watchdog.py:102): production uses
        # ``os._exit`` to skip stdlib atexit handlers that could
        # re-enter paused threads; tests inject a recorder.
        import os as _os

        self._exit_fn = exit_fn or (lambda code: _os._exit(code))
        self._stop = threading.Event()
        self._vol = VolatilityEstimator(settings)
        # M7.4 / M8 — open the tape runtime feed (shared-memory reader).
        # Returns None unless REGIME_USE_RUNTIME_RECORDER_FEED=true (the
        # default is OFF), or when the segment can't be opened (recorder
        # not running / wrong OS / perms). The bot NEVER depends on the
        # recorder: when this is None every consumer falls back to its
        # in-process signals. Read once per tick in the quote cycle.
        from app.runtime_recorder_feed import open_runtime_recorder_feed

        self._runtime_feed = open_runtime_recorder_feed(
            settings, logger_=logger
        )
        # M8 Candidate A — warm-start the volatility estimator from the
        # recorder's 24 h vol estimate, one-shot at startup. No-ops today
        # because ``vol_bps_p95_24h`` is still a dark (NaN) wire field; it
        # lights up when the recorder begins publishing it, with zero bot
        # redeploy. The counter stays 0 until a real seed is applied.
        if self._runtime_feed is not None:
            try:
                _seed_stale_ns = int(
                    float(settings.regime_runtime_feed_stale_threshold_s)
                    * 1e9
                )
                _seed_snap = self._runtime_feed.read_fresh(_seed_stale_ns)
                if _seed_snap is not None and self._vol.seed_from_recorder(
                    _seed_snap.vol_bps_p95_24h
                ):
                    self._state.warmstart_vol_seeded_from_recorder_count += 1
                    logger.info(
                        "runtime_recorder_feed: warm-started vol estimator "
                        "from recorder vol_bps_p95_24h=%s",
                        _seed_snap.vol_bps_p95_24h,
                    )
            except Exception:
                # Warm-start is best-effort; never block startup on it.
                logger.exception("runtime_feed_vol_seed_failed")
        # v1.5.239 — replace the placeholder VolAbsEwmaEstimator on
        # state (built with default half-life in BotState.__init__)
        # with one driven by the Settings half-life knob. Running
        # unconditionally; consumer wiring (regime classifier
        # vol_slope) gated by REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE.
        from app.vol_abs_ewma import VolAbsEwmaEstimator as _VAE
        state.vol_abs_ewma = _VAE(
            halflife_seconds=float(settings.vol_abs_ewma_halflife_seconds)
        )
        # v1.5.277 / AQC Phase 1 — replace the placeholder controller
        # on state (built with default AQCSettings in BotState.__init__)
        # with one driven by the env knobs. Phase 1 is observe-only;
        # the AQC computes aggression_level but no consumer reads it
        # for trading decisions yet. The diagnostic value is
        # published via snapshot_dict so the operator can see what
        # the controller WOULD have done.
        from app.active_quoting_controller import (
            ActiveQuotingController as _AQC,
            AQCSettings as _AQCS,
        )
        state.active_quoting_controller = _AQC(
            settings=_AQCS(
                enabled=bool(settings.aqc_enabled),
                target_net_edge_per_min_usd=float(
                    settings.aqc_target_net_edge_per_min_usd
                ),
                markout_floor_bps=float(settings.aqc_markout_floor_bps),
                # v1.5.290 — outlier-robust floor: median + min-fill gate.
                markout_floor_min_fills=int(
                    settings.aqc_markout_floor_min_fills
                ),
                pi_kp=float(settings.aqc_pi_kp),
                pi_ki=float(settings.aqc_pi_ki),
                # Phase 2 (v1.5.281) — output wiring (default off).
                wire_min_half_spread=bool(
                    settings.aqc_wire_min_half_spread
                ),
                min_half_spread_floor_at_full_aggression_bps=float(
                    settings.aqc_min_half_spread_floor_at_full_aggression_bps
                ),
                # Phase 3 (v1.5.282) — inventory exec-bias wiring (off).
                wire_inventory=bool(settings.aqc_wire_inventory),
                inventory_util_floor_at_full_aggression_pct=float(
                    settings.aqc_inventory_util_floor_at_full_aggression_pct
                ),
                # Phase 4 (v1.5.283) — inventory skew-coeff wiring (off).
                wire_skew=bool(settings.aqc_wire_skew),
                skew_coeff_at_full_aggression_bps=float(
                    settings.aqc_skew_coeff_at_full_aggression_bps
                ),
            )
        )
        # 1.2.26 (todo-010 Phase 2): subscribe the vol estimator to
        # mid-change events from the public-WS BBO handler. Lets
        # the estimator sample at the venue's natural cadence
        # (~7-8 ticks/sec on OKX SUI per 1.2.25 measurements)
        # instead of the quote-cycle rate (2/sec). Combined with
        # the v1.2.24 dedup, we now capture every real mid tick.
        # The per-cycle ``self._vol.push_mid(mid)`` call in the
        # quote loop stays as defense in depth — dedup handles
        # any redundant pushes silently.
        state.add_mid_change_listener(self._vol.push_mid)
        self._tox = ToxicityEngine(settings)
        self._pnl = PnlTracker()
        self._exec = OrderManager(
            settings,
            client,
            storage,
            state,
            private_event_queue=private_event_queue,
            # 2026-05-14 BUG-024: OrderManager calls this on
            # CRITICAL connectivity failures (e.g. place response
            # "unconfirmed" — OKX neither acked nor rejected). The
            # callback runs full ``Bot.kill()`` flow: cancel-all,
            # flatten (if safe), CRITICAL bot_event, Telegram push,
            # dashboard kill chip.
            request_kill_fn=self.kill,
            # Phase 1b (v1.4.230) — share Bot's clock so the
            # OrderManager + Bot read the same monotonic time. They
            # exchange monotonic timestamps via shared BotState; a
            # split-clock setup would silently corrupt those
            # invariants under replay.
            clock=self._clock,
        )
        self._flatten_lock = threading.Lock()
        # Independent drain thread — keeps writing fills / order updates
        # to ``trading.db`` even when the bot is KILLED. The quote loop
        # stops on kill, but the private-WS thread continues receiving
        # events (e.g. a resting order matching post-kill before our
        # cancel lands), and those events must still be persisted so
        # the local DB matches exchange truth. See ``_drain_loop``.
        self._drain_thread: Optional[threading.Thread] = None
        self._drain_loop_interval_s: float = 0.05  # 50 ms poll
        self._last_snapshot_monotonic = 0.0
        # Throttled observation logs (no per-tick spam).
        self._obs_heartbeat_last_monotonic: float | None = None
        self._logged_first_market_snapshot = False
        self._logged_first_quote_persisted = False
        # Stale-market WARNING is per-tick by nature (the risk gate
        # re-evaluates every tick). A single sustained book gap — e.g.
        # the genuine 2–5 s BBO gaps OKX's bbo-tbt feed shows in quiet
        # periods, which also surface 1:1 in replay — would otherwise
        # emit one identical WARNING per 500 ms tick. Log once on
        # episode ENTRY and suppress repeats until the book goes fresh
        # again. Never gates quoting (risk.action is unchanged); this
        # only de-spams the log, per the "avoid per-event WARNING logs,
        # keep the hot path lean" convention.
        self._stale_data_warn_episode_active = False
        self._last_persistent_save_monotonic: float = 0.0
        self._account_refresh_suppressed_order_uncertainty_count: int = 0

    def _stamp_executable_half_spread_on_breakdown(
        self, decision: QuoteDecision
    ) -> None:
        """2026-05-12 codex-#3: stamp the actual executable half-spread
        onto ``state.last_quote_breakdown`` after the quote engine has
        run. Reads the post-cap placed prices from ``self._exec``'s
        telemetry (``exec_finalize_{bid,ask}_px``); when both sides
        were placed and we can read both prices, computes the half
        as ``(ask - bid) / 2 / mid * 10000``.

        Returns silently when either side is missing or mid is not
        finite — the field stays ``None`` in those cases, which the
        Spread-tab UI renders as "—".
        """
        bk = getattr(self._state, "last_quote_breakdown", None)
        if bk is None:
            return
        tel = self._exec.get_quote_exec_telemetry() or {}
        nb = tel.get("exec_finalize_bid_px") or tel.get("exec_norm_bid_px")
        na = tel.get("exec_finalize_ask_px") or tel.get("exec_norm_ask_px")
        try:
            mid = float(decision.mid_price)
        except (TypeError, ValueError):
            return
        if (
            nb is None
            or na is None
            or not isinstance(nb, (int, float))
            or not isinstance(na, (int, float))
            or not math.isfinite(float(nb))
            or not math.isfinite(float(na))
            or not math.isfinite(mid)
            or mid <= 0.0
            or float(na) <= float(nb)
        ):
            return
        executable_half_bps = (
            (float(na) - float(nb)) / 2.0 / mid * 10_000.0
        )
        from dataclasses import replace as _replace
        try:
            new_bk = _replace(
                bk,
                executable_half_spread_bps=round(executable_half_bps, 3),
            )
            self._state.last_quote_breakdown = new_bk
        except Exception:
            logger.exception("executable_half_spread_replace_failed")

    def _hold_all_should_cancel(self, eligibility_reason: str) -> bool:
        """2026-05-12 codex-#1: decide whether a HOLD_ALL cycle should
        cancel resting orders or leave them alive.

        Defaults to True (cancel). Returns False only when every
        recognized reason in the eligibility-reason string is in the
        ``HOLD_ALL_KEEP_RESTING_REASONS`` allow-list — typically just
        ``recovery_cooldown``, which is a sub-second pause where the
        existing order is fresh and re-placing it after the pause
        would just churn the same order.

        Reason strings look like ``ok|fresh=freshness_ok|drift=drift_ok|recovery_cooldown``
        — pipe-separated tags. Tags starting with ``ok`` or ending in
        ``_ok`` are filtered out as healthy markers; the remaining
        tags must all be in the allow-list.
        """
        if not self._settings.cancel_resting_on_hold_all:
            return False
        if not eligibility_reason:
            # No reason string at all — fall back to "cancel" so we
            # don't accidentally leave orders resting on a malformed
            # state.
            return True
        # Allow-list from config (comma-separated). Trim + ignore empties.
        allow_raw = self._settings.hold_all_keep_resting_reasons or ""
        allow = {p.strip() for p in allow_raw.split(",") if p.strip()}
        # Parse the eligibility-reason tag list. Filter out healthy
        # markers so they don't disqualify the allow-list match.
        parts = [p.strip() for p in eligibility_reason.split("|") if p.strip()]
        meaningful = [
            p for p in parts
            if not p.startswith("ok") and not p.endswith("_ok")
        ]
        if not meaningful:
            # All-OK reason → no actual HOLD_ALL trigger → don't cancel
            # (defensive — shouldn't happen if HOLD_ALL is the state,
            # but be safe).
            return False
        # Strip ``key=value`` form down to the key for matching.
        def _tag(part: str) -> str:
            # ``freshness=freshness_ok`` → ``freshness``; bare tag → unchanged.
            return part.split("=", 1)[0].strip() if "=" in part else part

        meaningful_tags = {_tag(p) for p in meaningful}
        # If every meaningful tag is in the allow-list → keep resting.
        # Otherwise cancel.
        return not meaningful_tags.issubset(allow)

    def _compute_post_fill_cooldown_remaining_ms(self, side: Side) -> float:
        """todo-011: ms remaining on the post-fill replace cooldown
        for ``side``. Returns 0.0 when the feature is disabled
        (``POST_FILL_REPLACE_COOLDOWN_MS<=0``), when no fill has been
        observed on this side this session, or when the cooldown
        has already elapsed. Otherwise returns the remaining ms
        (always > 0 when active).

        Computed from ``state.last_fill_monotonic_ms_{buy,sell}``
        (updated in ``state.record_fill``) and the current monotonic
        clock. Robust to wall-clock jumps because monotonic time is
        monotonic by definition. Cheap: 2 attribute reads + 2 floats.
        """
        cooldown_ms = float(self._settings.post_fill_replace_cooldown_ms)
        if cooldown_ms <= 0.0:
            return 0.0
        if side == Side.BUY:
            last_ms = float(self._state.last_fill_monotonic_ms_buy)
        elif side == Side.SELL:
            last_ms = float(self._state.last_fill_monotonic_ms_sell)
        else:
            return 0.0
        if last_ms <= 0.0:
            return 0.0
        now_ms = self._clock.monotonic() * 1000.0
        remaining = cooldown_ms - (now_ms - last_ms)
        return remaining if remaining > 0.0 else 0.0

    def _compute_queue_position_ratio(self, side: Side) -> Optional[float]:
        """v1.5.215 Phase 8B — per-side queue-position ratio for the
        queue-aware sizing + inside-post features.

        Returns ``None`` (no signal, no shrink) when:
          * The bot has no resting WorkingOrder on this side, OR
          * The L1 size on this side is missing / non-positive.

        Otherwise returns a value in ``[0, 1]`` where ``0.0`` = bot's
        own resting size IS the entire queue (nothing ahead) and
        ``1.0`` = bot's own resting size is a tiny fraction of the
        queue (everything ahead). The current L1-only estimate is
        intentionally conservative — assumes the bot is LAST in the
        queue (worst case). See ``app/queue_model.py``.

        Cheap: 3 attribute reads + one division.
        """
        try:
            from app.queue_model import estimate_queue_position_ratio
        except Exception:
            return None
        m = self._state.market
        if m is None:
            return None
        if side == Side.BUY:
            inside_total = getattr(m, "bid_size", None)
        elif side == Side.SELL:
            inside_total = getattr(m, "ask_size", None)
        else:
            return None
        if inside_total is None:
            return None
        # Own resting size on this side. Sum sizes across slots on
        # the requested side from ``state.all_working_orders()``.
        own = 0.0
        try:
            for wo in self._state.all_working_orders():
                if wo.side != side:
                    continue
                sz = getattr(wo, "size_remaining", None)
                if sz is None:
                    sz = getattr(wo, "size", None)
                if sz is not None:
                    own += float(sz)
        except Exception:
            return None
        return estimate_queue_position_ratio(
            own_resting_size=own,
            inside_total_size=float(inside_total),
        )

    def _maybe_save_persistent_runtime_state(self) -> None:
        interval = self._settings.persistent_runtime_save_interval_seconds
        if interval <= 0:
            return
        now_m = self._clock.monotonic()
        if (now_m - self._last_persistent_save_monotonic) < interval:
            return
        self._last_persistent_save_monotonic = now_m
        try_save_persistent_runtime_state(self._settings, self._state)

    def _save_persistent_runtime_state_now(self) -> None:
        """Force an immediate persistent-state save, bypassing the
        throttle. Used at soft-flatten entry/exit transitions where
        the regular tail save would be skipped (SF paths return
        before reaching ``_maybe_save_persistent_runtime_state`` in
        ``one_tick``). Updates the throttle timestamp so a subsequent
        tail save in the same window is suppressed.

        Codex review 2026-05-07 / 1.1.37 fix (HIGH-1).
        """
        if self._settings.persistent_runtime_save_interval_seconds <= 0:
            return
        self._last_persistent_save_monotonic = self._clock.monotonic()
        try_save_persistent_runtime_state(self._settings, self._state)

    def _flatten_drain_private_ws(self) -> None:
        """
        Process private WS backlog during blocking flatten so fills/order updates are not
        stranded while the bot thread is inside flatten; force REST open-order reconcile so
        local working state stays aligned after catchup.
        """
        self._exec.drain_private_events(self._pnl)
        if self._settings.trading_enabled:
            with self._state._lock:
                pos_abs = abs(float(self._state.position.position_qty))
                open_cnt = self._state.open_order_count()
            if pos_abs > 1e-12 or open_cnt > 0:
                self._exec.request_open_orders_reconcile(
                    reason="flatten_drain_private_ws",
                    force=True,
                    emergency=False,
                )

    def _maybe_promote_starting_to_running(self, healthy: bool) -> None:
        """
        After the first healthy exchange snapshot and open-order reconcile, leave STARTING so
        risk/execution match the pre-restart policy (inherit inventory; no flatten on nonzero size).
        """
        with self._state._lock:
            if self._state.bot_status != BotStatus.STARTING:
                return
            if not healthy or self._state.manual_pause or self._state.killed:
                return
            if self._state.order_desync:
                return
            self._state.bot_status = BotStatus.RUNNING
            if self._state.fast_start_mode_enabled:
                self._state.startup_ready_without_fill_replay = (
                    self._state.startup_historical_fill_replay_skipped
                )
        self._log_event(
            EventSeverity.INFO,
            "startup_reconciled",
            "exchange snapshot reconciled — RUNNING (inherit inventory; cancel-all already applied if configured)",
            {
                "symbol": self._settings.symbol,
                "fast_start_mode_enabled": self._state.fast_start_mode_enabled,
                "startup_historical_fill_replay_skipped": self._state.startup_historical_fill_replay_skipped,
                "startup_ready_without_fill_replay": self._state.startup_ready_without_fill_replay,
                "startup_rest_fill_replay_skipped_count": self._state.startup_rest_fill_replay_skipped_count,
                "startup_private_snapshot_fills_skipped_count": self._state.startup_private_snapshot_fills_skipped_count,
            },
        )

    def _exchange_snapshot_healthy(self, addr: str) -> bool:
        """True when book is usable and (if an account is configured)
        the account snapshot was loaded with valid contents.

        BUG-006: previously this checked only ``acct is not None``,
        but a fail-open ``fetch_account_snapshot`` could return an
        all-None ``AccountSnapshot`` object that passed the
        ``is not None`` check while having no real data. Now we
        require at least one of ``equity_usd`` / ``withdrawable_usd``
        to be a finite positive number — which is what's needed for
        the drawdown cap and capital-adjusted reporting downstream.
        Without that, the bot is trading blind to capital.
        """
        with self._state._lock:
            m = self._state.market
            acct = self._state.account
        if m is None:
            return False
        mid_ok = (
            m.mid_price is not None
            and math.isfinite(float(m.mid_price))
            and float(m.mid_price) > 0
        )
        bba_ok = (
            m.best_bid is not None
            and m.best_ask is not None
            and math.isfinite(float(m.best_bid))
            and math.isfinite(float(m.best_ask))
            and float(m.best_bid) > 0
            and float(m.best_ask) > 0
        )
        book_ok = mid_ok or bba_ok
        if not (addr or "").strip():
            return book_ok
        if acct is None:
            return False

        def _ok_money(v: Any) -> bool:
            try:
                f = float(v) if v is not None else None
            except (TypeError, ValueError):
                return False
            return f is not None and math.isfinite(f) and f > 0.0

        acct_ok = _ok_money(getattr(acct, "equity_usd", None)) or _ok_money(
            getattr(acct, "withdrawable_usd", None)
        )
        return book_ok and acct_ok

    def _public_ws_live_path(self) -> bool:
        return bool(
            self._settings.public_ws_enabled and self._public_stream is not None
        )

    def _public_ws_risk_kwargs(self) -> dict[str, Any]:
        if not self._public_ws_live_path():
            return {
                "public_ws_live_path": False,
                "public_ws_connected": True,
                "public_ws_seconds_since_message": None,
                "public_ws_seen_first_bbo": False,
            }
        with self._state._lock:
            last = self._state.public_ws_last_message_wall_ts
            conn = self._state.public_ws_connected
            seen = self._state.public_ws_seen_first_bbo
        return {
            "public_ws_live_path": True,
            "public_ws_connected": conn,
            "public_ws_seconds_since_message": seconds_since(last) if last else None,
            "public_ws_seen_first_bbo": seen,
        }

    def _book_snapshot_has_valid_touch(self, m: BestBidAsk | None) -> bool:
        if m is None:
            return False
        if m.mid_price is None or not math.isfinite(float(m.mid_price)) or float(m.mid_price) <= 0:
            return False
        if (
            m.best_bid is None
            or m.best_ask is None
            or not math.isfinite(float(m.best_bid))
            or not math.isfinite(float(m.best_ask))
            or float(m.best_bid) <= 0
            or float(m.best_ask) <= 0
        ):
            return False
        return True

    def _market_book_fresh(self, m: BestBidAsk | None) -> bool:
        """Is the book fresh enough to trade on / exit recovery with?

        Freshness is determined by **message age**, not by the current
        ``public_ws_connected`` flag. Rationale: the WS is often in a
        mid-reconnect window (close → reopen takes hundreds of ms on
        GRVT), during which ``conn=False`` but the last BBO we received
        is seconds-fresh. The old implementation treated transient
        disconnects as "book stale" and prevented recovery from ever
        completing while the WS was flapping (seen in snap_20260419_063856
        — 13 public_ws reconnects in 13 min, bot stuck in
        RECOVERING_MARKET_DATA the whole time despite 0.6 s-old books).

        Before the first BBO ever arrives (``last_w is None``), we
        return False — there's no book to check age against, so we
        wait for one regardless of connection flag. Once we've seen a
        message, message age is the authoritative signal; the
        connection flag becomes a secondary hint used only for entry
        logic (reconnect bursts) and telemetry.

        If the WS is truly broken long-term, ``last_w`` stops advancing
        and the age check naturally fails — no freshness regression.
        """
        if not self._book_snapshot_has_valid_touch(m):
            return False
        assert m is not None
        if self._public_ws_live_path():
            with self._state._lock:
                last_w = self._state.public_ws_last_message_wall_ts
            w_age = seconds_since(last_w) if last_w else None
            if w_age is None:
                return False
            return float(w_age) < float(self._settings.public_ws_stale_warn_seconds)
        age = seconds_since(m.ts_local)
        if age is None:
            return False
        return float(age) < float(self._settings.stale_data_warn_seconds)

    def _market_recovery_burst(self) -> None:
        addr = venue_account_address(self._settings)
        ingest = self._exec.should_ingest_fills_via_rest()
        max_a = int(self._settings.market_data_recovery_max_attempts)
        backoff = float(self._settings.market_data_recovery_backoff_seconds)
        for attempt in range(max_a):
            if attempt > 0 and backoff > 0:
                time.sleep(backoff)
            if self._public_stream is not None:
                self._public_stream.request_reconnect()
            refresh_account_only(
                self._client,
                self._state,
                addr,
                self._storage,
                self._pnl,
                ingest_fills_via_rest=ingest,
            )
            with self._state._lock:
                self._state.market_data_recovery_refresh_attempts_episode += 1
            self._exec.drain_private_events(self._pnl)
            att_payload = {
                "event": "market_data_recovery_attempt",
                "symbol": self._settings.symbol,
                "attempt_index": attempt + 1,
                "max_attempts": max_a,
            }
            log_extra(logger, logging.INFO, "market_data_recovery_attempt", att_payload)
            self._log_event(
                EventSeverity.INFO,
                "market_data_recovery_attempt",
                "market data recovery refresh attempt",
                {
                    "symbol": self._settings.symbol,
                    "attempt_index": attempt + 1,
                    "max_attempts": max_a,
                },
            )

    def _complete_market_recovery(self) -> None:
        with self._state._lock:
            self._state.bot_status = BotStatus.RUNNING
            self._state.market_data_recovery_started_monotonic = None
            self._state.market_data_recovery_logged_stale_detected = False
            self._state.market_data_recovery_refresh_attempts_episode = 0
            # Reset exponential-backoff state so the next outage starts
            # with a fresh initial-delay ramp (not resumed from the
            # previous outage's accumulated backoff).
            self._state.market_data_recovery_next_burst_mono = None
            self._state.market_data_recovery_current_burst_backoff_s = 0.0
        payload = {"symbol": self._settings.symbol}
        log_extra(
            logger,
            logging.INFO,
            "market_data_recovery_succeeded",
            {"event": "market_data_recovery_succeeded", **payload},
        )
        self._log_event(
            EventSeverity.INFO,
            "market_data_recovery_succeeded",
            "market data recovery succeeded",
            payload,
        )

    def _market_data_recovery_supervisor(self) -> bool:
        """
        When the public BBO stream is stale or disconnected (or REST book age in test mode),
        reconnect/resubscribe with bounded attempts before a hard kill.
        Returns True if the tick should stop (hard kill after failed recovery).
        """
        if not self._settings.market_data_recovery_enabled:
            return False
        if not self._settings.trading_enabled or self._state.is_killed():
            return False

        with self._state._lock:
            status = self._state.bot_status
            m = self._state.market
            started = self._state.market_data_recovery_started_monotonic

        stale_age_log: Optional[float] = None
        if self._public_ws_live_path():
            kill_age = float(self._settings.public_ws_stale_kill_seconds)
            with self._state._lock:
                conn = self._state.public_ws_connected
                last_w = self._state.public_ws_last_message_wall_ts
                seen = self._state.public_ws_seen_first_bbo
            w_age = seconds_since(last_w) if last_w else None
            if w_age is not None:
                stale_age_log = float(w_age)
            mid_ok = (
                m is not None
                and m.mid_price is not None
                and math.isfinite(float(m.mid_price))
                and float(m.mid_price) > 0
            )
            # Stale criterion:
            #   • BEFORE first BBO (bootstrap): ``not conn`` is the only
            #     signal available — no message history to measure
            #     freshness against. If the WS is disconnected during
            #     startup, we defer entering RUNNING.
            #   • AFTER first BBO: message age is authoritative. A brief
            #     reconnect gap (``not conn`` for a few hundred ms)
            #     must NOT latch the bot into RECOVERING_MARKET_DATA
            #     if the last BBO is still seconds-fresh — that was
            #     the bug that kept the bot stuck on GRVT's flaky
            #     public WS, with 0.6 s-old books but ``conn=False``
            #     at the moment of each check. Book-age checks in
            #     ``_market_book_fresh`` use the same contract.
            if not seen:
                ws_stale = not conn
            else:
                ws_stale = w_age is None or float(w_age) >= kill_age
            stale_past_kill = mid_ok and ws_stale
        else:
            kill_age = float(self._settings.stale_data_kill_seconds)
            age = seconds_since(m.ts_local) if m and m.ts_local else None
            if age is not None:
                stale_age_log = float(age)
            stale_past_kill = (
                m is not None
                and m.mid_price is not None
                and math.isfinite(float(m.mid_price))
                and float(m.mid_price) > 0
                and age is not None
                and float(age) >= kill_age
            )

        def log_stale_detected_once() -> None:
            with self._state._lock:
                if self._state.market_data_recovery_logged_stale_detected:
                    return
                self._state.market_data_recovery_logged_stale_detected = True
            pl = {
                "event": "stale_data_detected",
                "symbol": self._settings.symbol,
                "seconds_since_book_ts": stale_age_log,
                "stale_data_warn_seconds": self._settings.public_ws_stale_warn_seconds
                if self._public_ws_live_path()
                else self._settings.stale_data_warn_seconds,
                "stale_data_kill_seconds": kill_age,
            }
            log_extra(logger, logging.WARNING, "stale_data_detected", pl)
            self._log_event(
                EventSeverity.WARNING,
                "stale_data_detected",
                "stale book past kill threshold — market data recovery",
                pl,
            )

        if status == BotStatus.RECOVERING_MARKET_DATA:
            now_mono = self._clock.monotonic()
            if started is not None:
                elapsed = now_mono - started
                if elapsed > float(self._settings.market_data_recovery_max_duration_seconds):
                    fail_pl = {
                        "event": "market_data_recovery_failed",
                        "symbol": self._settings.symbol,
                        "reason": "max_duration_exceeded",
                        "max_duration_seconds": self._settings.market_data_recovery_max_duration_seconds,
                        "elapsed_seconds": elapsed,
                    }
                    log_extra(logger, logging.ERROR, "market_data_recovery_failed", fail_pl)
                    self._log_event(
                        EventSeverity.ERROR,
                        "market_data_recovery_failed",
                        "market data recovery window exhausted",
                        fail_pl,
                    )
                    esc_pl = {
                        "event": "stale_data_kill_escalated",
                        "symbol": self._settings.symbol,
                    }
                    log_extra(logger, logging.CRITICAL, "stale_data_kill_escalated", esc_pl)
                    self._log_event(
                        EventSeverity.CRITICAL,
                        "stale_data_kill_escalated",
                        "stale market data — kill after failed recovery",
                        esc_pl,
                    )
                    self.kill("stale_data_kill_escalated")
                    return True

            # Non-disruptive recovery exit. Run BEFORE the backoff
            # gate and BEFORE running another burst: if the market
            # data is already fresh, the original stale event has
            # resolved itself (often the case — a transient WS
            # hiccup that the public-WS reconnect logic already
            # cleared) and we should just transition back to RUNNING.
            #
            # Without this check, the bot can stay stuck in
            # RECOVERING_MARKET_DATA for many minutes even after the
            # data has been continuously fresh, because:
            #   1. ``_market_recovery_burst`` calls
            #      ``request_reconnect()`` on the public WS;
            #   2. that briefly tears down the active WS during the
            #      reconnect handshake;
            #   3. the post-burst freshness check therefore sees a
            #      momentarily-stale ``last_message_wall_ts`` and
            #      schedules another backoff;
            #   4. exponential backoff stretches the gap to minutes;
            #   5. between bursts the WS is healthy + book is fresh
            #      but the recovery branch never re-checks — it just
            #      waits for the next burst window.
            # Observed in snapshot_prod.okx.hype.usdt.perp_260509054855
            # on 2026-05-09: real 8.4-s stale at 05:42:59, book
            # recovered by 05:43:02 (regular ``market_data_refresh_
            # success`` events firing throughout), but bot remained
            # in RECOVERING_MARKET_DATA for 6+ minutes emitting no
            # quotes despite being healthy.
            with self._state._lock:
                m_pre = self._state.market
            if self._market_book_fresh(m_pre):
                self._complete_market_recovery()
                return False

            # Exponential-backoff gate between successive bursts. Each
            # failed burst sets ``next_burst_mono`` to ``now + current_backoff``;
            # ticks that fire before then are skipped so we don't hammer
            # the venue during an outage. First burst on a new stale
            # event runs immediately (``next_burst_mono`` is None).
            with self._state._lock:
                next_burst_mono = self._state.market_data_recovery_next_burst_mono
            if next_burst_mono is not None and now_mono < next_burst_mono:
                return False

            self._market_recovery_burst()
            with self._state._lock:
                m2 = self._state.market
            if self._market_book_fresh(m2):
                self._complete_market_recovery()
                return False

            # Burst failed — schedule the next one with exponential backoff.
            initial_backoff = float(
                self._settings.market_data_recovery_burst_backoff_initial_seconds
            )
            max_backoff = float(
                self._settings.market_data_recovery_burst_backoff_max_seconds
            )
            multiplier = float(
                self._settings.market_data_recovery_burst_backoff_multiplier
            )
            with self._state._lock:
                cur = float(self._state.market_data_recovery_current_burst_backoff_s)
                if cur <= 0.0:
                    new_backoff = initial_backoff
                else:
                    new_backoff = min(cur * multiplier, max_backoff)
                self._state.market_data_recovery_current_burst_backoff_s = new_backoff
                self._state.market_data_recovery_next_burst_mono = (
                    now_mono + new_backoff
                )

            fail_pl = {
                "event": "market_data_recovery_failed",
                "symbol": self._settings.symbol,
                "reason": "burst_did_not_yield_fresh_book",
                "seconds_since_book_ts": seconds_since(m2.ts_local) if m2 and m2.ts_local else None,
                "next_burst_in_seconds": new_backoff,
                "elapsed_seconds": (now_mono - started) if started is not None else None,
            }
            log_extra(logger, logging.WARNING, "market_data_recovery_failed", fail_pl)
            self._log_event(
                EventSeverity.WARNING,
                "market_data_recovery_failed",
                "market data recovery burst did not yield fresh book",
                fail_pl,
            )
            return False

        if status in (BotStatus.RUNNING, BotStatus.STARTING) and stale_past_kill:
            log_stale_detected_once()
            with self._state._lock:
                self._state.bot_status = BotStatus.RECOVERING_MARKET_DATA
                self._state.market_data_recovery_started_monotonic = self._clock.monotonic()
                self._state.market_data_recovery_refresh_attempts_episode = 0
                # Fresh outage — reset backoff state. First burst runs
                # immediately (no gating delay); subsequent failed bursts
                # ramp up via the ``RECOVERING_MARKET_DATA`` branch above.
                self._state.market_data_recovery_next_burst_mono = None
                self._state.market_data_recovery_current_burst_backoff_s = 0.0
            start_pl = {"event": "market_data_recovery_started", "symbol": self._settings.symbol}
            log_extra(logger, logging.WARNING, "market_data_recovery_started", start_pl)
            self._log_event(
                EventSeverity.WARNING,
                "market_data_recovery_started",
                "market data recovery started",
                start_pl,
            )
            self._market_recovery_burst()
            with self._state._lock:
                m2 = self._state.market
            if self._market_book_fresh(m2):
                self._complete_market_recovery()
            else:
                # Initial burst failed — schedule the next burst with the
                # same exponential-backoff logic the main RECOVERING branch
                # uses. Without this, the next tick re-bursts immediately
                # (no gating) and the "every tick = burst" behaviour we
                # just fixed would leak back in through this entry path.
                initial_backoff = float(
                    self._settings.market_data_recovery_burst_backoff_initial_seconds
                )
                now_mono_entry = self._clock.monotonic()
                with self._state._lock:
                    self._state.market_data_recovery_current_burst_backoff_s = initial_backoff
                    self._state.market_data_recovery_next_burst_mono = (
                        now_mono_entry + initial_backoff
                    )
                fail_pl = {
                    "event": "market_data_recovery_failed",
                    "symbol": self._settings.symbol,
                    "reason": "initial_burst_did_not_yield_fresh_book",
                    "seconds_since_book_ts": seconds_since(m2.ts_local) if m2 and m2.ts_local else None,
                    "next_burst_in_seconds": initial_backoff,
                }
                log_extra(logger, logging.WARNING, "market_data_recovery_failed", fail_pl)
                self._log_event(
                    EventSeverity.WARNING,
                    "market_data_recovery_failed",
                    "initial market data recovery burst did not yield fresh book",
                    fail_pl,
                )

        return False

    def _drain_loop(self) -> None:
        """Continuously drain the private-WS event queue into Storage,
        independent of kill state.

        History and motivation:
          ``tmp/snap_20260419_035212`` + ``tmp/xinfo_20260419_044643`` —
          after the bot self-killed on stale data at 02:04:59 UTC, the
          GRVT exchange processed two more fills on the bot's account:
          a post-kill maker match at 02:05:28 (cancel lost the race
          against the match) and the operator's manual taker-close at
          04:02:50. Both events were delivered to the private WS queue
          (the WS thread keeps running post-kill) but NEVER written to
          ``trading.db`` because the only consumer —
          ``drain_private_events`` inside ``Bot.one_tick()`` — stops
          running once ``is_killed()`` returns True.

        This thread consumes the queue on a tight 50 ms poll and keeps
        the DB in sync with exchange reality regardless of bot status.
        It exits only on ``Bot.stop()`` (``self._stop.set()``). The
        quote loop still calls ``drain_private_events`` directly for
        low-latency handling during normal operation; the
        ``_drain_mutex`` in ``OrderManager`` serialises the two
        callers safely.
        """
        while not self._stop.is_set():
            try:
                self._exec.drain_private_events(self._pnl)
            except Exception:
                # Transient errors (rare parser bugs, state exceptions)
                # must NOT kill the drain thread — that would reopen
                # the very bug we're fixing. Log and keep looping.
                logger.exception("fill_drain_thread tick failed")
                self._state.bump_execution_errors("fill_drain_thread_exception")
            # ``Event.wait(timeout=...)`` returns True immediately when
            # ``self._stop`` is set, so shutdown is responsive. Otherwise
            # it sleeps the poll interval.
            if self._stop.wait(timeout=self._drain_loop_interval_s):
                break

    def _start_drain_thread(self) -> None:
        """Start the independent drain thread. Idempotent — safe to call
        multiple times; only spawns one thread per Bot instance."""
        t = self._drain_thread
        if t is not None and t.is_alive():
            return
        t = threading.Thread(
            target=self._drain_loop,
            name="mm-fill-drain",
            daemon=True,
        )
        self._drain_thread = t
        t.start()

    def stop(self) -> None:
        # v1.5.5 -- HALT TRADING FIRST. Operator safety contract:
        # after ``ops stop`` is initiated (or SIGTERM lands), the
        # FIRST observable action MUST be the book going clean. The
        # original ordering set ``_stop``, tore down the dispatcher,
        # then let ``main.py`` run cancel-all afterward -- leaving a
        # ~5-60s window where the bot's last tick continued to place
        # orders before ``main.py``'s cancel-all swept them. The
        # rebuild here:
        #
        #   1. Set ``_stop`` first so the next ``one_tick`` no-ops
        #      via its top-of-tick gate (see ``one_tick``). Any tick
        #      mid-flight finishes its current iteration but the
        #      next iteration is a no-op.
        #   2. Fire a SYNCHRONOUS bulk cancel-all directly via the
        #      adapter (NOT via the dispatcher) so the cancel hits
        #      the venue BEFORE we tear down the dispatcher. Bulk
        #      cancel is one request, atomic on the matching engine
        #      -- under 500ms on OKX.
        #   3. THEN tear down the dispatcher. Any place intents the
        #      bot's last tick queued before step 1 may have been
        #      dispatched already and the responses will land via
        #      private WS; that's fine -- step 2 cleared the book
        #      so post-stop fills are a closed set.
        #
        # ``main.py``'s cancel-all (currently after ``stop_bot()``)
        # stays as defence-in-depth -- catches any order that
        # indexed at the matching engine AFTER step 2's cancel
        # completed but BEFORE the dispatcher was torn down.
        self._stop.set()
        try:
            if (
                self._settings.cancel_all_on_shutdown
                and self._settings.trading_enabled
                and self._client.has_write_access()
            ):
                outcome = self._exec.cancel_all_orders_for_symbol_bulk_or_fallback()
                logger.warning(
                    "bot_stop_precancel_outcome=%s symbol=%s",
                    outcome, self._settings.symbol,
                )
        except Exception:
            logger.exception("bot_stop_precancel_failed")
        self._exec.shutdown_transport()
        self._state.wake_quote_loop()
        # Wait briefly for the drain thread to finish (if it was started
        # by run_forever). 2 s is plenty: the loop wakes on the Event
        # immediately and has a bounded per-tick drain budget.
        t = self._drain_thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=2.0)
        # M7.4 — unmap the tape runtime feed (best-effort; idempotent).
        # No-op when the feed was never opened (knob OFF).
        try:
            if self._runtime_feed is not None:
                self._runtime_feed.close()
        except Exception:
            logger.exception("runtime_feed_close_failed")
        # Phase F1 (v1.5.42) — write bot_shutdown.json into the recorder
        # session dir if forensics are active. Idempotent: paired
        # stop()+kill() calls both invoke this; the first write wins.
        # Best-effort: any failure logs but never blocks shutdown.
        self._maybe_write_shutdown_manifest(kill_reason=None)

    def _maybe_write_shutdown_manifest(
        self, *, kill_reason: Optional[str]
    ) -> None:
        """Phase F1 (v1.5.42) — best-effort write of bot_shutdown.json
        into the recorder's session dir. Idempotent (the helper short-
        circuits on second call by file-existence check). Called from
        both ``stop()`` and ``kill()`` — whichever lands first wins.

        Resolves the recorder session dir lazily from the pointer file;
        skip cleanly when recorder is disabled / pointer absent.
        """
        try:
            import os as _os
            from app.session_manifest import (
                read_recorder_pointer,
                write_bot_shutdown,
            )
            from app import __version__ as _bot_version

            _profile_path = _os.environ.get("APP_ENV_FILE", "") or ""
            _profile_name = (
                _os.path.splitext(_os.path.basename(_profile_path))[0]
                if _profile_path
                else ""
            ) or "unknown"
            _rec_session_dir = read_recorder_pointer(_profile_name)
            if _rec_session_dir is None:
                return
            write_bot_shutdown(
                session_dir=_rec_session_dir,
                state=self._state,
                bot_version=_bot_version,
                kill_reason=kill_reason,
            )
        except Exception:
            logger.exception("bot_shutdown_manifest_write_failed")

    def _dump_crash_snapshot(self, reason: str) -> Optional[str]:
        """v1.5.152 — write a self-contained diagnostic bundle to
        persistent disk BEFORE the kill path's optional ``os._exit(42)``
        fires. The bundle preserves the operator-pullable snapshot data
        (the same JSON files ``stats_snapshot.py`` would produce) even
        when the bot's HTTP API stops responding seconds later.

        Output directory shape (matches the existing ``snapshots/`` /
        ``snapshots_crash/`` convention so operator tooling works
        unchanged):

            <CRASH_SNAPSHOT_DIR>/v<ver>-<YYYYMMDD-HHMMSS>-<reason>/
                meta.json              — version, profile, kill_reason
                state_current.json     — BotState.snapshot_dict() (full
                                          per-tick state + every
                                          behavioural-gate publisher
                                          block)
                events_recent.json     — last N bot_events rows
                                          (configurable via
                                          ``CRASH_SNAPSHOT_EVENT_LIMIT``)

        Fail-closed: returns the absolute path of the created
        directory on success, ``None`` on any failure. Failures are
        logged but never raised — the dump is an OBSERVABILITY aid,
        not part of the kill's safety contract; a dump failure must
        not block the cancel/flatten/exit cleanup that follows.

        The configured directory is created on first use (operator
        prepares the parent on initial setup via
        ``mkdir -p /var/lib/dtc-mm-as``). If the configured path
        can't be created (no permission, no disk), the helper falls
        back to ``<repo>/snapshots_crash/`` so a kill always lands
        a bundle SOMEWHERE the operator can find via SSH.

        Empty ``CRASH_SNAPSHOT_DIR=""`` disables the feature
        entirely (returns None without writing).
        """
        try:
            import json as _json
            import os as _os
            import re as _re
            from datetime import datetime as _dt, timezone as _tz
            from pathlib import Path as _Path

            from app import __version__ as _bot_version

            target_root_cfg = str(
                getattr(self._settings, "crash_snapshot_dir", "") or ""
            ).strip()
            if not target_root_cfg:
                logger.info("crash_snapshot_dump_disabled_by_config")
                return None

            # Slugify the kill reason for use as a directory name.
            # Reasons can be multi-token comma-separated; replace any
            # filesystem-hostile chars with underscore + truncate at
            # 60 chars so directory listings stay readable.
            slug = _re.sub(r"[^A-Za-z0-9_.-]+", "_", reason or "unknown")
            slug = slug.strip("_") or "unknown"
            slug = slug[:60]
            ts_iso = self._clock.now_utc().strftime("%Y%m%d-%H%M%S")
            leaf = f"v{_bot_version}-{ts_iso}-{slug}"

            target_dir = _Path(target_root_cfg) / leaf
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                # Fall back to a repo-relative path so we ALWAYS write
                # something. The repo's mm-as checkout is on persistent
                # disk by design (the bot's mm.db lives in the same
                # tree), so this fallback is safe.
                fallback_root = _Path(_os.getcwd()) / "snapshots_crash"
                logger.warning(
                    "crash_snapshot_primary_path_failed primary=%s err=%s "
                    "falling_back_to=%s",
                    target_root_cfg,
                    e,
                    str(fallback_root),
                )
                target_dir = fallback_root / leaf
                target_dir.mkdir(parents=True, exist_ok=True)

            # --- meta.json ----------------------------------------------
            try:
                _profile_path = _os.environ.get("APP_ENV_FILE", "") or ""
                _profile_name = (
                    _os.path.splitext(_os.path.basename(_profile_path))[0]
                    if _profile_path
                    else ""
                ) or "unknown"
                meta = {
                    "bot_version": _bot_version,
                    "bot_profile": _profile_name,
                    "kill_reason": reason,
                    "captured_at_utc": self._clock.now_utc().isoformat(),
                    "pid": _os.getpid(),
                    "hostname": _os.uname().nodename
                    if hasattr(_os, "uname")
                    else _os.environ.get("HOSTNAME", "unknown"),
                    "symbol": str(getattr(self._settings, "symbol", "")),
                }
                (target_dir / "meta.json").write_text(
                    _json.dumps(meta, indent=2)
                )
            except Exception:
                logger.exception("crash_snapshot_meta_write_failed")

            # --- state_current.json -------------------------------------
            try:
                state_dict = self._state.snapshot_dict()
                (target_dir / "state_current.json").write_text(
                    _json.dumps(state_dict, indent=2, default=str)
                )
            except Exception:
                logger.exception("crash_snapshot_state_write_failed")

            # --- events_recent.json -------------------------------------
            try:
                event_limit = int(
                    getattr(
                        self._settings, "crash_snapshot_event_limit", 500
                    )
                )
                events = self._storage.recent_bot_events(limit=event_limit)
                (target_dir / "events_recent.json").write_text(
                    _json.dumps(events, indent=2, default=str)
                )
            except Exception:
                logger.exception("crash_snapshot_events_write_failed")

            logger.warning(
                "crash_snapshot_dumped reason=%s path=%s",
                reason,
                str(target_dir),
            )
            return str(target_dir)
        except Exception:
            # Catch-all so a snapshot bug never blocks the kill path.
            logger.exception("crash_snapshot_dump_top_level_failed")
            return None

    def _log_event(
        self,
        severity: EventSeverity,
        event_type: str,
        message: str,
        payload: dict | None = None,
    ) -> None:
        self._storage.insert_bot_event(
            self._clock.now_utc().isoformat(),
            severity.value,
            event_type,
            message,
            payload,
        )
        log_fn = {
            EventSeverity.INFO: logger.info,
            EventSeverity.WARNING: logger.warning,
            EventSeverity.ERROR: logger.error,
            EventSeverity.CRITICAL: logger.critical,
        }.get(severity, logger.info)
        log_fn("%s: %s", event_type, message)
        # Forward CRITICAL events (kill, hard failures) to the Telegram
        # ops channel. We deliberately limit to CRITICAL — WARNING /
        # ERROR fire often enough that broadcasting them all would make
        # the channel useless. The notifier's notify_ops enqueues and
        # returns immediately so this never blocks the trading loop.
        if (
            self._notifier is not None
            and severity == EventSeverity.CRITICAL
        ):
            try:
                self._notifier.notify_ops(
                    severity.name, event_type, message, payload
                )
            except Exception:  # noqa: BLE001
                logger.exception("notifier_forward_failed event=%s", event_type)

    def _maybe_alert_shock_telegram(
        self,
        *,
        mode: Any,
        prior_mode: Any,
        transition_reason: Optional[str],
        util: float,
        vol_ratio: Optional[float],
        now_mono: float,
        mode_since_mono: float,
    ) -> None:
        """Phase 2G (v1.4.204) — emit Telegram alerts for SHOCK-mode
        transitions and stuck-in-SHOCK persistence.

        Called once per regime-controller tick (every quote loop
        iteration). Three exit paths:

        * Alerts globally disabled or no Telegram notifier configured
          → no-op.
        * Not currently in SHOCK → reset the per-episode "already
          alerted" flags so the NEXT SHOCK entry gets a fresh alert,
          then return.
        * In SHOCK: maybe fire the entry alert (if just-entered AND
          not yet alerted), maybe fire the persistence alert (if
          dwell > threshold AND not yet alerted).

        Both alerts are session-counted on BotState so the operator
        can verify in snapshots that the alerts are firing.

        Hot-path-safe: ``notify_ops`` is fire-and-forget (enqueues
        and returns immediately); the only blocking work here is a
        few float comparisons + the counter bump.
        """
        # Local imports — avoid pulling regime types into bot.py's top
        # which would create an import cycle.
        from app.regime_controller import Mode as RegimeMode

        if not bool(
            getattr(
                self._settings,
                "regime_shock_telegram_alerts_enabled",
                True,
            )
        ):
            return
        if self._notifier is None:
            return

        currently_in_shock = mode is RegimeMode.SHOCK
        if not currently_in_shock:
            # Reset flags so the next SHOCK entry alerts fresh.
            if self._state.shock_entry_telegram_alerted_for_episode:
                self._state.shock_entry_telegram_alerted_for_episode = False
            if self._state.shock_persistence_telegram_alerted_for_episode:
                self._state.shock_persistence_telegram_alerted_for_episode = (
                    False
                )
            return

        is_entry = (
            transition_reason is not None
            and prior_mode is not None
            and prior_mode is not RegimeMode.SHOCK
        )

        # ENTRY alert — one-shot per episode.
        if (
            is_entry
            and not self._state.shock_entry_telegram_alerted_for_episode
        ):
            self._state.shock_entry_telegram_alerted_for_episode = True
            self._state.shock_telegram_entry_alerts_sent_total += 1
            payload = {
                "from": (
                    prior_mode.value
                    if hasattr(prior_mode, "value")
                    else str(prior_mode)
                ),
                "to": "SHOCK",
                "reason": transition_reason,
                "util": util,
                "vol_ratio": vol_ratio,
            }
            try:
                self._notifier.notify_ops(
                    "WARNING",
                    "regime_shock_entered",
                    (
                        f"regime SHOCK entered — util={util:.2f} "
                        f"reason={transition_reason} "
                        f"(SELL_ONLY if long / BUY_ONLY if short; "
                        f"locked until shock_gate clears)"
                    ),
                    payload,
                )
            except Exception:  # noqa: BLE001
                logger.exception("shock_entry_telegram_send_failed")

        # PERSISTENCE alert — one-shot per episode, fires when dwell
        # exceeds the threshold. Skip if mode_since_mono is unset
        # (fresh-process boundary case).
        threshold_s = float(
            getattr(
                self._settings,
                "regime_shock_telegram_persistence_seconds",
                300.0,
            )
            or 0.0
        )
        if (
            threshold_s > 0
            and mode_since_mono > 0
            and not self._state.shock_persistence_telegram_alerted_for_episode
        ):
            dwell_s = max(0.0, now_mono - mode_since_mono)
            if dwell_s >= threshold_s:
                self._state.shock_persistence_telegram_alerted_for_episode = True
                self._state.shock_telegram_persistence_alerts_sent_total += 1
                payload = {
                    "dwell_seconds": dwell_s,
                    "threshold_seconds": threshold_s,
                    "util": util,
                    "vol_ratio": vol_ratio,
                }
                try:
                    self._notifier.notify_ops(
                        "WARNING",
                        "regime_shock_persistent",
                        (
                            f"regime SHOCK persistent — dwell={dwell_s:.0f}s "
                            f"(> {threshold_s:.0f}s threshold); util={util:.2f}. "
                            f"shock_gate may be stuck; check connectivity + "
                            f"drift signal."
                        ),
                        payload,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("shock_persistence_telegram_send_failed")

    def _maybe_send_daily_regime_summary(self, *, now_mono: float) -> None:
        """v1.5.26 Phase 2G.3 -- emit a daily time-in-mode summary to
        Telegram once per 24h (configurable).

        Called once per regime-controller tick. Three exit paths:

        * Daily summary globally disabled or no Telegram notifier
          configured -> no-op.
        * Time since last summary < interval -> no-op (cheap subtract).
        * Interval elapsed -> compose payload from regime_controller's
          session-cumulative time-in-mode counters + transition count,
          fire-and-forget notify_ops, update last_sent + counter.

        Pre-v1.5.26 this lived as a "DEFERRED -- needs cron" item.
        Implementation here uses the bot's existing tick cadence as
        the scheduler. No external cron required.

        Hot-path-safe: notify_ops is fire-and-forget; the only work
        here is a few attribute reads + a comparison.
        """
        if not bool(
            getattr(self._settings, "regime_daily_summary_enabled", True)
        ):
            return
        if self._notifier is None:
            return

        interval_s = float(
            getattr(
                self._settings,
                "regime_daily_summary_interval_seconds",
                86400.0,
            )
            or 0.0
        )
        if interval_s <= 0:
            return

        last_sent = float(
            getattr(self._state, "regime_daily_summary_last_sent_mono", 0.0)
            or 0.0
        )
        elapsed = now_mono - last_sent
        if elapsed < interval_s:
            return

        # Compose payload. Read directly from the regime_controller
        # state -- those counters are session-cumulative, which means
        # the FIRST summary (last_sent=0, fires ~24h after process
        # start) covers the whole session-since-start, and subsequent
        # summaries technically cover session-since-start too (the
        # counters don't reset on summary send). That's fine for the
        # operator's use case ("did the bot spend much time in
        # DEFENSIVE/SHOCK today?") -- counters monotonic-increasing
        # gives them total time-in-mode visibility per Telegram ping.
        try:
            rc = self._state.regime_controller
            mode_now = getattr(getattr(rc, "mode", None), "value", "?")
            t_normal = float(getattr(rc, "time_in_normal_seconds", 0.0) or 0.0)
            t_def = float(getattr(rc, "time_in_defensive_seconds", 0.0) or 0.0)
            t_shock = float(getattr(rc, "time_in_shock_seconds", 0.0) or 0.0)
            t_calm = float(getattr(rc, "time_in_calm_seconds", 0.0) or 0.0)
            t_caut = float(getattr(rc, "time_in_cautious_seconds", 0.0) or 0.0)
            n_transitions = int(getattr(rc, "transition_count", 0) or 0)
        except Exception:  # noqa: BLE001
            logger.exception("daily_summary_state_read_failed")
            return

        def _fmt_dwell(seconds: float) -> str:
            if seconds < 60.0:
                return f"{seconds:.0f}s"
            if seconds < 3600.0:
                return f"{seconds / 60.0:.1f}m"
            return f"{seconds / 3600.0:.1f}h"

        payload = {
            "current_mode": mode_now,
            "time_in_normal_s": round(t_normal, 1),
            "time_in_defensive_s": round(t_def, 1),
            "time_in_shock_s": round(t_shock, 1),
            "time_in_calm_s": round(t_calm, 1),
            "time_in_cautious_s": round(t_caut, 1),
            "transitions_total": n_transitions,
            "interval_s": interval_s,
            "since_last_summary_s": elapsed,
        }

        # Compose a short human-readable line; non-zero counters only
        # so the Telegram message stays compact when the bot has been
        # in a quiet regime.
        parts: list[str] = []
        for label, secs in (
            ("CALM", t_calm),
            ("NORMAL", t_normal),
            ("CAUTIOUS", t_caut),
            ("DEFENSIVE", t_def),
            ("SHOCK", t_shock),
        ):
            if secs > 0:
                parts.append(f"{label} {_fmt_dwell(secs)}")
        body_parts = " · ".join(parts) if parts else "no mode samples yet"
        message = (
            f"regime daily summary -- current: {mode_now} · "
            f"{body_parts} · transitions: {n_transitions}"
        )

        try:
            self._notifier.notify_ops(
                "INFO",
                "regime_daily_summary",
                message,
                payload,
            )
            self._state.regime_daily_summary_last_sent_mono = now_mono
            self._state.regime_daily_summary_sent_total += 1
        except Exception:  # noqa: BLE001
            logger.exception("regime_daily_summary_send_failed")

    def _aqc_quote_decision_fields(self) -> dict[str, Any]:
        """v1.5.306 audit §5 P0 #2 — per-tick AQC trace fields for the
        ``quote_decisions`` row.

        Returns the five Active-Quoting-Controller observability columns
        (PI aggression output + integrator term + markout safety-floor
        flag + the two observation inputs) as a dict ready to
        ``.update()`` onto a quote-decision row. Read from
        ``state.active_quoting_controller`` — the SAME state the quote
        engine consulted for THIS tick, since the controller isn't
        advanced until ``_update_active_quoting_controller()`` runs later
        in the tick. All five are None when AQC is disabled (controller
        is None) or on any read failure: this is observability-only and
        must NEVER affect the hot path, so the whole body is wrapped in a
        blanket try/except.

        Unlike the per-fill stamp (which persists only when an order is
        placed), this row is written EVERY tick — including no-order
        ticks — so the replay report + postmortem can see the
        controller's trajectory even across stretches where nothing got
        quoted.
        """
        empty: dict[str, Any] = {
            "aqc_aggression_level": None,
            "aqc_integrator": None,
            "aqc_safety_floor_engaged": None,
            "aqc_observed_net_edge_per_min_usd": None,
            "aqc_observed_markout_5s_mean_bps": None,
        }
        try:
            aqc = getattr(self._state, "active_quoting_controller", None)
            if aqc is None:
                return dict(empty)

            def _f(v: Any) -> Optional[float]:
                if v is None:
                    return None
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    return None
                return fv if math.isfinite(fv) else None

            return {
                "aqc_aggression_level": _f(
                    getattr(aqc, "aggression_level", None)
                ),
                "aqc_integrator": _f(getattr(aqc, "integrator", None)),
                "aqc_safety_floor_engaged": (
                    1 if bool(getattr(aqc, "safety_floor_engaged", False)) else 0
                ),
                "aqc_observed_net_edge_per_min_usd": _f(
                    getattr(aqc, "last_observed_net_edge_per_min_usd", None)
                ),
                "aqc_observed_markout_5s_mean_bps": _f(
                    getattr(aqc, "last_observed_markout_5s_mean_bps", None)
                ),
            }
        except Exception:
            logger.exception("aqc_quote_decision_fields_failed")
            return dict(empty)

    def _update_active_quoting_controller(self) -> None:
        """v1.5.277 / AQC — compute observed net edge per minute +
        rolling markout 5s statistics and feed them to the controller.

        The controller advances its PI state and exposes diagnostics
        in ``state.snapshot_dict()``. From Phase 2+ its
        ``aggression_level`` modulates effective multipliers; Phase 1
        is observe-only.

        Computation:
        * Window: last ``AQC_WINDOW_SECONDS`` (default 300 s).
        * net_edge_usd = sum(fee_usd_negated_to_rebate) + sum(markout_5s_dollar_impact)
        * net_edge_per_min_usd = net_edge_usd / (window_seconds / 60)
        * markout_5s_median_bps = MEDIAN of fills' ``markout_5s_bps``
          (the floor-driving statistic, v1.5.290)
        * markout_5s_mean_bps = mean (diagnostic only)
        * markout_sample_count = number of fills with markout

        v1.5.290 — the min-fill gate moved INTO the controller: this
        feeder passes the median, mean and sample count for ANY number
        of fills (including 0 → all None), and the controller decides
        whether ≥``markout_floor_min_fills`` distinct fills are present
        before letting the safety floor engage. Using the MEDIAN (not
        the mean) means a single flash fill can no longer pin aggression
        for the whole window (the v1.5.289 incident). Per the v1.5.279
        decouple, a sparse/absent markout window just means "no safety
        brake this tick", NOT "freeze the controller" — the PI still
        advances on net-edge (computable from a single fill, or 0.0 for
        an empty window) so it stays alive at TON's low fill rate.
        """
        aqc = getattr(self._state, "active_quoting_controller", None)
        if aqc is None:
            return
        now_mono = self._clock.monotonic()
        window_seconds = float(
            getattr(self._settings, "aqc_window_seconds", 300.0)
        )
        if window_seconds <= 0:
            return
        cutoff_wall = self._clock.now_utc().timestamp() - window_seconds
        rebate_usd = 0.0
        markout_dollar = 0.0
        markout_bps_samples: list[float] = []
        with self._state._lock:
            for f in self._state.recent_fills:
                ts_fill = getattr(f, "ts_fill", None)
                if ts_fill is None:
                    continue
                try:
                    ts_seconds = ts_fill.timestamp()
                except Exception:
                    continue
                if ts_seconds < cutoff_wall:
                    continue
                # Fee is signed: positive = paid, negative = rebate.
                # Convention follows ``Fill.fee``. Net edge counts
                # rebate as POSITIVE, so we negate.
                try:
                    rebate_usd += -float(getattr(f, "fee", 0.0) or 0.0)
                except (TypeError, ValueError):
                    pass
                # Markout dollar impact = markout_5s_bps * notional / 1e4.
                # Sign convention: negative markout = adverse selection
                # = negative dollar impact for the MM.
                mk_bps = getattr(f, "markout_5s_bps", None)
                if mk_bps is None:
                    continue
                try:
                    mk_bps_f = float(mk_bps)
                    notional_f = float(getattr(f, "notional", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                markout_dollar += mk_bps_f * notional_f / 1e4
                markout_bps_samples.append(mk_bps_f)

        from app.active_quoting_controller import (
            compute_rolling_net_edge_per_min,
        )
        net_edge_per_min = compute_rolling_net_edge_per_min(
            recent_fills_window_seconds=window_seconds,
            rebate_usd_in_window=rebate_usd,
            markout_dollar_impact_usd_in_window=markout_dollar,
        )
        # v1.5.290 — compute median (floor-driving), mean (diagnostic)
        # and sample count for ANY number of fills. The min-fill gate now
        # lives IN the controller: it requires ≥``markout_floor_min_fills``
        # distinct fills AND a finite median before the safety floor can
        # engage. We pass None for all three only on a truly empty window.
        # Below the gate the controller skips the brake (v1.5.279
        # decouple) but still advances its PI on net-edge.
        markout_5s_median_bps: Optional[float] = None
        markout_5s_mean_bps: Optional[float] = None
        markout_sample_count: Optional[int] = None
        if markout_bps_samples:
            markout_5s_median_bps = float(
                statistics.median(markout_bps_samples)
            )
            markout_5s_mean_bps = sum(markout_bps_samples) / len(
                markout_bps_samples
            )
            markout_sample_count = len(markout_bps_samples)
        aqc.update(
            observed_net_edge_per_min_usd=net_edge_per_min,
            observed_markout_5s_median_bps=markout_5s_median_bps,
            markout_sample_count=markout_sample_count,
            observed_markout_5s_mean_bps=markout_5s_mean_bps,
            now_mono_seconds=now_mono,
        )

    def _persist_snapshots(self, force: bool = False) -> None:
        interval = self._settings.snapshot_interval_seconds
        now_m = self._clock.monotonic()
        if (
            not force
            and interval > 0
            and (now_m - self._last_snapshot_monotonic) < interval
        ):
            return
        self._last_snapshot_monotonic = now_m
        with self._state._lock:
            pos = self._state.position
            pnl = self._state.pnl
            wd = (
                self._state.account.withdrawable_usd
                if self._state.account
                else None
            )
        self._storage.insert_position_snapshot(
            {
                "ts": self._clock.now_utc().isoformat(),
                "symbol": pos.symbol,
                "position_qty": pos.position_qty,
                "avg_entry_price": pos.avg_entry_price,
                "mark_price": pos.mark_price,
                "position_notional": pos.position_notional,
                "unrealized_pnl_usd": pos.unrealized_pnl_usd,
            }
        )
        # 1.2.9: also capture mid_price + vol_bps + session-cum traded
        # notional so the dashboard's Session PnL chart can render
        # synced sub-bands (mid trace, vol trace, per-interval traded
        # volume bars) under the PnL line. Per-interval volume is
        # derived client-side by diffing adjacent samples'
        # ``session_traded_notional_usd`` — one less aggregator on the
        # bot side. NULL when not yet seen (warmup) — schema permits.
        try:
            with self._state._lock:
                m = self._state.market
                vol = self._state.vol_bps
                traded_cum = float(
                    getattr(self._state, "session_traded_notional_usd", 0.0) or 0.0
                )
                # 1.3.31 (todo-030): cross-venue basis EWMA sampled
                # under the same lock as the rest of the equity-snapshot
                # row so the value lines up with ``mid_price`` /
                # ``position_qty``. NULL when Binance is disabled or
                # the EWMA hasn't seeded yet — same semantics as the
                # v14 ``quote_decisions.binance_basis_ewma`` column.
                basis_ewma_for_snap = getattr(
                    self._state, "binance_basis_ewma", None
                )
            mid_for_snap: Optional[float] = (
                getattr(m, "mid_price", None) if m is not None else None
            )
            vol_for_snap: Optional[float] = (
                float(vol) if vol is not None else None
            )
            basis_ewma_for_snap_typed: Optional[float] = (
                float(basis_ewma_for_snap)
                if basis_ewma_for_snap is not None
                else None
            )
        except Exception:
            mid_for_snap = None
            vol_for_snap = None
            traded_cum = 0.0
            basis_ewma_for_snap_typed = None
        self._storage.insert_equity_snapshot(
            {
                "ts": self._clock.now_utc().isoformat(),
                "equity_usd": pnl.equity_usd if pnl.equity_usd is not None else 0.0,
                "cash_usd": wd if wd is not None else 0.0,
                "realized_pnl_usd": pnl.realized_pnl_usd,
                "unrealized_pnl_usd": pnl.unrealized_pnl_usd,
                "fees_usd": pnl.fees_usd,
                "drawdown_usd": pnl.drawdown_usd,
                "mid_price": mid_for_snap,
                "vol_bps": vol_for_snap,
                "session_traded_notional_usd": traded_cum,
                # Phase 8A Option B (v1.5.189) — per-sample AS
                # attribution. Time-series of (a) the half-spread
                # the AS formula produced for the last quote cycle,
                # (b) the cached k-intensity that fed the formula.
                # Both NULL when AS is disabled OR when the cache
                # / breakdown hasn't been populated yet. Forwarded
                # by the equity_history publisher into S3 so the
                # dashboard's Session-PnL sub-band stack can chart
                # AS-base over time alongside vol_bps.
                "base_half_spread_bps": (
                    float(
                        getattr(
                            self._state.last_quote_breakdown,
                            "base_half_spread_bps",
                            None,
                        )
                    )
                    if getattr(self._state, "last_quote_breakdown", None) is not None
                    and getattr(
                        self._state.last_quote_breakdown,
                        "base_half_spread_bps",
                        None,
                    ) is not None
                    else None
                ),
                "as_k_intensity_per_min": (
                    float(self._state.as_k_intensity_per_min)
                    if getattr(self._state, "as_k_intensity_per_min", None)
                    is not None
                    else None
                ),
                # 1.2.15: signed inventory at snapshot time. Drives
                # the dashboard's Inventory sub-band (the "is the
                # bot mean-reverting?" diagnostic). Read from the
                # already-locked ``pos`` snapshot above so we don't
                # take the lock twice.
                "position_qty": pos.position_qty,
                # 1.2.25: BBO + mid-change session-cumulative
                # counters (todo-010 Phase 2 decision data). NOT
                # locked because they're monotonic-increasing
                # plain ints; a torn read at worst gives a slightly
                # stale value, never a corrupt one.
                "bbo_event_count_session": int(
                    getattr(self._state, "bbo_event_count_session", 0) or 0
                ),
                "mid_change_count_session": int(
                    getattr(self._state, "mid_change_count_session", 0) or 0
                ),
                # 1.3.31 (todo-030): basis EWMA for the basis sub-band
                # on the dashboard's Session PnL chart. Computed above
                # under the same state lock as the other sub-band feeds.
                "binance_basis_ewma": basis_ewma_for_snap_typed,
            }
        )

    def _build_kill_payload(self, reasons: list[str]) -> dict | None:
        """Build a structured payload for a kill event.

        Currently attaches an ``execution_errors`` breakdown whenever that
        reason is present, so operators can see which call sites contributed
        to the windowed count that tripped the kill threshold.
        """
        if "execution_errors" not in reasons:
            return None
        snap = self._state.execution_errors_window_snapshot(
            self._settings.execution_errors_window_seconds
        )
        return {
            "reasons": list(reasons),
            "execution_errors": {
                **snap,
                "threshold": int(self._settings.max_execution_errors),
            },
        }

    def drain(
        self,
        *,
        max_cancel_passes: int = 2,
        idle_timeout_s: float = 2.0,
        clean_book_timeout_s: float = 15.0,
    ) -> dict[str, Any]:
        """De-risk the bot IN PLACE without stopping the process.

        This is the first phase of the operator's *finalized* stop
        (``ops.ps1 <profile> stop`` WITHOUT ``-Urgent``): quiesce the
        bot so a full HTTP diagnostic snapshot can be captured from the
        STILL-RUNNING process, then the caller stops the service. The
        HTTP snapshot's live half (``state_current.json``,
        ``session_summary.json``, the venue REST cross-checks) only
        exists while the process is up, so we must drain in place rather
        than snapshot a corpse.

        Sequence (cancel-only — the inventory position is intentionally
        LEFT INTACT and carried to the next session; this is NOT a
        flatten and NOT a kill):

          1. ``set_manual_pause(True)`` — stops new quote cycles
             (``manual_pause`` flips ``bot_status_allows_quoting`` to
             False and the risk gate emits NO_QUOTE). The HTTP API and
             all reconcile / WS machinery stay UP.
          2. ``wait_transport_idle`` — let any in-flight outbound batch
             finish so we cancel against a settled local book.
          3. Up to ``max_cancel_passes`` rounds of
             ``cancel_all_orders_for_symbol_bulk_or_fallback()`` →
             ``wait_transport_idle`` → ``verify_book_clean_on_venue``.
             On OKX cancel-all is ASYNC (no venue bulk endpoint; the
             fallback ENQUEUES per-order cancels on the dispatcher), so
             we must let the dispatcher fire (idle) and then confirm via
             venue REST that the book actually drained. Break early once
             the venue shows zero resting orders.

        Idempotent and resumable: because we only pause + cancel, a
        subsequent ``/control/resume`` (or a fresh process) brings the
        bot straight back. Best-effort throughout — a cancel/verify
        hiccup is logged and the next pass retries; we never raise into
        the operator's stop path.

        Returns a summary dict for the ops script to inspect / log::

            {
              "clean": bool,                # venue REST showed 0 orders
              "remaining_open_orders": int, # last venue count (-1 = unread)
              "cancel_passes": int,         # cancel rounds actually run
              "cancel_outcomes": list[str], # per-pass bulk-or-fallback tag
              "paused": bool,               # manual_pause was set
            }
        """
        outcomes: list[str] = []
        self._log_event(
            EventSeverity.INFO,
            "control_drain_begin",
            "drain requested: pausing quotes + cancelling resting orders "
            "(position left intact)",
            {
                "max_cancel_passes": max_cancel_passes,
                "idle_timeout_s": idle_timeout_s,
                "clean_book_timeout_s": clean_book_timeout_s,
            },
        )

        # Phase 1: pause new quoting. The risk gate reads manual_pause
        # directly, so this takes effect on the very next quote tick.
        paused = False
        try:
            self._state.set_manual_pause(True)
            paused = True
        except Exception:
            logger.exception("drain_pause_failed")

        # Phase 2: let any in-flight outbound batch settle before we
        # start cancelling, so cancel-all sees a stable local WO set.
        try:
            self._exec.wait_transport_idle(idle_timeout_s)
        except Exception:
            logger.exception("drain_initial_wait_idle_failed")

        # Phase 3: cancel -> idle -> verify, up to max_cancel_passes.
        clean = False
        remaining = -1
        passes = 0
        for _ in range(max(1, int(max_cancel_passes))):
            passes += 1
            try:
                outcome = (
                    self._exec.cancel_all_orders_for_symbol_bulk_or_fallback()
                )
            except Exception:
                logger.exception("drain_cancel_all_failed pass=%d", passes)
                outcome = "cancel_exception"
            outcomes.append(outcome)

            # OKX cancel-all enqueues on the dispatcher; wait for it to
            # actually fire the HTTP cancels before we read the venue.
            try:
                self._exec.wait_transport_idle(idle_timeout_s)
            except Exception:
                logger.exception("drain_wait_idle_failed pass=%d", passes)

            try:
                res = self._exec.verify_book_clean_on_venue(
                    timeout_s=clean_book_timeout_s,
                )
                clean = bool(res.success)
                remaining = int(res.final_count)
            except Exception:
                logger.exception("drain_verify_failed pass=%d", passes)
                clean = False

            if clean:
                break

        # Final event — INFO when clean, WARNING when orders persist so
        # the operator's stop log shows a loud, actionable line.
        payload = {
            "clean": clean,
            "remaining_open_orders": remaining,
            "cancel_passes": passes,
            "cancel_outcomes": outcomes,
            "paused": paused,
        }
        if clean:
            self._log_event(
                EventSeverity.INFO,
                "control_drain_complete",
                f"drain complete: book clean after {passes} cancel pass(es)",
                payload,
            )
        else:
            self._log_event(
                EventSeverity.WARNING,
                "control_drain_incomplete",
                (
                    f"drain INCOMPLETE: {remaining} order(s) still resting "
                    f"after {passes} cancel pass(es) — proceeding with stop "
                    f"anyway; the SIGTERM shutdown cancel-all is the backstop"
                ),
                payload,
            )

        return payload

    def kill(self, reason: str, payload: dict | None = None) -> None:
        """Terminate the bot. Always cancel outstanding orders; flatten
        only when we can trust market data to price the exit; auto-restart
        the process when the kill reason is environmental (BUG-013).

        Order of operations:
          1. Mark state KILLED (stops new quote cycles).
          2. Log the CRITICAL ``kill`` event.
          3. ``cancel_all_orders_for_symbol`` — safe regardless of
             market-data state: cancellation by order-id needs no price.
          4. If ``FLATTEN_ON_KILL`` is true AND the kill reason does not
             indicate blindness, call ``flatten``. Otherwise log
             ``blind_kill_skip_flatten``.
          5. BUG-013: if the kill reason is in
             ``_AUTO_RESTART_KILL_REASONS`` (stale_data_kill_escalated,
             public_ws_stale_kill, etc.), schedule a delayed
             ``os._exit(42)`` so the container manager restarts the
             process. Reasons NOT in that set (manual kill, drawdown,
             session loss, desync) leave the bot in KILLED-alive state
             for human review.

             The grace delay is set by
             ``KILL_AUTO_RESTART_GRACE_SECONDS`` (default 5.0) — long
             enough to flush the CRITICAL kill event to Telegram, short
             enough that the operator's pager isn't waiting forever.
             Set to 0 to disable the auto-restart entirely (legacy
             behaviour: KILLED process stays alive indefinitely).

             Reproducer: ``tmp/snap_20260427_040436`` — bot self-killed
             at 01:23:50 UTC on a Bluefin public-WS 9-min stall, then
             sat dead-but-alive for 2.5 h until the operator noticed.
             With this fix, the same scenario exits with code 42 ~5 s
             later and the process supervisor restarts immediately.
        """
        with self._state._lock:
            self._state.killed = True
            self._state.kill_reason = reason
            self._state.kill_timestamp = self._clock.now_utc()
            self._state.bot_status = BotStatus.KILLED
        self._log_event(EventSeverity.CRITICAL, "kill", reason, payload)
        # v1.5.275 (BUG-039 fix): force a final per-period snapshot of
        # position + equity right after marking KILLED, BEFORE the
        # cancel / flatten / restart cleanup runs. The periodic writer
        # (`_persist_snapshots`) operates on an interval; if the kill
        # fires partway through that interval, the last persisted row
        # is from before the fill that triggered the kill, and the
        # dashboard's inventory / unrealized-PnL sub-bands display
        # stale values at the chart's right edge. Reproduced on
        # snapshot v1.5.273-260530-075807: inventory chart showed +2
        # while state_current.json correctly showed 0 (the SF#25
        # taker fill at 07:35:54 cleared the position before
        # sf_fatigue_tier4 killed the bot a moment later, but the
        # chart's last position_snapshots row predated the clearing
        # fill). Forcing the flush here captures the post-fill /
        # pre-cleanup state — same value state_current.json holds —
        # so chart and state agree.
        try:
            self._persist_snapshots(force=True)
        except Exception:
            logger.exception("kill_final_snapshot_flush_failed")
        # Phase F1 (v1.5.42) — write bot_shutdown.json with the kill
        # reason BEFORE the cancel/flatten ops so the manifest captures
        # state at the moment of kill rather than after the cleanup
        # actions. Idempotent vs. stop()'s call.
        self._maybe_write_shutdown_manifest(kill_reason=reason)
        # v1.5.152 — crash-snapshot dump to persistent disk.
        # Captures BotState.snapshot_dict() + recent bot_events as
        # JSON under <CRASH_SNAPSHOT_DIR>/v<ver>-<ts>-<reason>/. This
        # preserves the full diagnostic state even when:
        #   * the bot is configured with KILL_AUTO_RESTART_GRACE_SECONDS=0
        #     (process stays alive but HTTP API is in KILLED state),
        #   * OR auto-restart fires and the HTTP API dies before the
        #     operator can run ``ops.ps1 snapshot``,
        #   * OR the recorder cleanup script wipes the prior session
        #     dir on next bot start (see start_recorder_on_colo.sh —
        #     separate fix queued).
        # Best-effort. Failure logs but doesn't raise — kill cleanup
        # below must still run.
        self._dump_crash_snapshot(reason)
        # Always safe: cancel by order-id does not require a live book.
        self._exec.cancel_all_orders_for_symbol()
        # Drop join-depth overlay so a fresh process / restart starts
        # from a clean baseline rather than carrying tuning learned
        # in the regime that produced the kill.
        try:
            self._state.join_depth_controller.reset()
        except Exception:
            logger.exception("join_depth_controller_reset_failed_on_kill")

        try:
            if self._settings.flatten_on_kill:
                if _kill_reason_is_blind(reason):
                    self._log_event(
                        EventSeverity.WARNING,
                        "blind_kill_skip_flatten",
                        "flatten-on-kill skipped: kill reason indicates "
                        "public market data is unreliable; a market-flatten "
                        "would cross an un-priceable book. Position left "
                        "intact for operator /control/flatten.",
                        {
                            "kill_reason": reason,
                            "flatten_on_kill": True,
                        },
                    )
                else:
                    self.flatten(blocking=True)
        finally:
            # BUG-013: schedule auto-restart even if flatten raised — the
            # recovery path on the next process boot will re-seed from
            # the venue.
            self._maybe_schedule_auto_restart_after_kill(reason)

    def _maybe_schedule_auto_restart_after_kill(self, reason: str) -> None:
        """BUG-013: if ``reason`` is environmental (in
        ``_AUTO_RESTART_KILL_REASONS``), spawn a daemon timer that calls
        ``self._exit_fn(42)`` after a configurable grace period.

        Why a timer rather than ``os._exit`` inline:
          * Lets the CRITICAL ``kill`` event propagate to Telegram before
            the process dies (the notifier is fire-and-forget, enqueued
            on a background thread).
          * Lets ``cancel_all_orders_for_symbol`` complete its REST round
            trip (already running on this thread).
          * Lets the persistent-runtime-state writer finish (so the
            replacement process can session-resume cleanly).

        Why exit code 42:
          * Same code the deadlock watchdog uses (see ``app/watchdog.py``).
          * systemd ``Restart=on-failure`` (current colo deploy) and
            every PaaS container manager bounce the process on any
            non-zero exit, so no new infrastructure config is required.
        """
        if not _kill_reason_should_auto_restart(reason):
            return
        grace_s = float(
            getattr(self._settings, "kill_auto_restart_grace_seconds", 5.0)
        )
        if grace_s <= 0:
            # Operator-disabled (legacy behaviour: stay KILLED-alive).
            self._log_event(
                EventSeverity.INFO,
                "kill_auto_restart_disabled",
                "kill_auto_restart disabled by KILL_AUTO_RESTART_GRACE_SECONDS=0; "
                "process will remain in KILLED state",
                {"reason": reason},
            )
            return
        self._log_event(
            EventSeverity.WARNING,
            "kill_auto_restart_scheduled",
            f"process exit scheduled in {grace_s:.1f}s for container "
            f"manager auto-restart (reason={reason})",
            {"reason": reason, "grace_seconds": grace_s, "exit_code": 42},
        )

        def _do_exit() -> None:
            try:
                logger.critical(
                    "kill_auto_restart_firing reason=%s exit_code=42",
                    reason,
                )
                self._exit_fn(42)
            except Exception:  # noqa: BLE001
                logger.exception("kill_auto_restart_exit_failed")

        # daemon=True so test processes that construct a Bot then exit
        # don't get a zombie ``os._exit(42)`` firing post-teardown. In
        # production the Timer always fires within its grace window
        # because the bot's other threads (drain, WS, watchdog) keep the
        # process alive past the kill() return — so the daemon flag
        # doesn't shorten production semantics.
        t = threading.Timer(grace_s, _do_exit)
        t.daemon = True
        t.start()

    def flatten(self, blocking: bool = False) -> FlattenResult | None:
        acquired = self._flatten_lock.acquire(blocking=blocking)
        if not acquired:
            return None
        t0 = self._clock.time()
        attempts = 0
        last_err: str | None = None
        result = FlattenResult.FAILED
        ending_abs = 0.0

        with self._state._lock:
            self._state.bot_status = BotStatus.FLATTENING
            self._state.flatten_mode = True
            start_abs = abs(self._state.position.position_qty)

        self._log_event(
            EventSeverity.WARNING,
            "flatten_started",
            "flatten started",
            {
                "starting_abs_qty": start_abs,
            },
        )
        self._exec.cancel_all_orders_for_symbol()
        self._flatten_drain_private_ws()
        deadline = self._clock.time() + self._settings.flatten_timeout_seconds

        try:
            if not self._client.has_write_access():
                elapsed = self._clock.time() - t0
                self._log_event(
                    EventSeverity.ERROR,
                    "flatten_failed",
                    "flatten failed: no exchange write access",
                    {
                        "flatten_result": FlattenResult.FAILED.value,
                        "starting_abs_qty": start_abs,
                        "ending_abs_qty": start_abs,
                        "attempts": 0,
                        "elapsed_seconds": elapsed,
                        "last_error": "no_write_access",
                    },
                )
                result = FlattenResult.FAILED
                ending_abs = start_abs
            else:
                refresh_account_only(
                    self._client,
                    self._state,
                    venue_account_address(self._settings),
                    self._storage,
                    self._pnl,
                    ingest_fills_via_rest=True,
                )
                self._flatten_drain_private_ws()
                with self._state._lock:
                    ending_abs = abs(self._state.position.position_qty)

                if ending_abs < 1e-8:
                    result = FlattenResult.SKIPPED_ALREADY_FLAT
                    self._log_event(
                        EventSeverity.INFO,
                        "flatten_completed",
                        "flatten completed (already flat)",
                        {
                            "flatten_result": result.value,
                            "starting_abs_qty": start_abs,
                            "ending_abs_qty": ending_abs,
                            "attempts": 0,
                            "elapsed_seconds": self._clock.time() - t0,
                            "last_error": None,
                        },
                    )
                else:
                    last_abs: float | None = None
                    stagnant = 0
                    # BUG-014: latch so the CRITICAL no-fills alert fires
                    # at most once per flatten call (the stagnation count
                    # keeps climbing on repeated misses, but the operator
                    # only needs to be told once that this venue isn't
                    # actually closing).
                    no_fills_alerted = False
                    base_delay = max(
                        0.05,
                        float(self._settings.flatten_attempt_delay_seconds),
                    )
                    while self._clock.time() < deadline:
                        refresh_account_only(
                            self._client,
                            self._state,
                            venue_account_address(self._settings),
                            self._storage,
                            self._pnl,
                            ingest_fills_via_rest=True,
                        )
                        self._flatten_drain_private_ws()
                        with self._state._lock:
                            ending_abs = abs(self._state.position.position_qty)
                        if last_abs is not None and abs(ending_abs - last_abs) < 1e-10:
                            stagnant += 1
                        else:
                            stagnant = 0
                        last_abs = ending_abs

                        # BUG-014: if 3 consecutive market_close attempts
                        # closed zero qty, surface a CRITICAL event NOW
                        # (rather than waiting for the timeout). With the
                        # Layer-1 slippage fix this should never fire in
                        # practice; if it does, the venue has a deeper
                        # issue and the operator needs to know early to
                        # use a different exit path (UI / manual /flatten
                        # retry / direct REST call). Latched so the same
                        # flatten doesn't spam the channel.
                        if (
                            attempts >= 3
                            and stagnant >= 3
                            and not no_fills_alerted
                            and ending_abs > 1e-8
                        ):
                            no_fills_alerted = True
                            self._log_event(
                                EventSeverity.CRITICAL,
                                "flatten_market_close_no_fills",
                                f"market_close has closed zero qty after "
                                f"{attempts} attempts (abs_qty={ending_abs}). "
                                "Venue may be rejecting the IOC market order; "
                                "investigate before the flatten timeout.",
                                {
                                    "abs_position_qty": ending_abs,
                                    "attempts": attempts,
                                    "stagnant_consecutive": stagnant,
                                },
                            )

                        if ending_abs < 1e-8:
                            result = FlattenResult.COMPLETED
                            self._log_event(
                                EventSeverity.INFO,
                                "flatten_completed",
                                "flatten completed",
                                {
                                    "flatten_result": result.value,
                                    "starting_abs_qty": start_abs,
                                    "ending_abs_qty": ending_abs,
                                    "attempts": attempts,
                                    "elapsed_seconds": self._clock.time() - t0,
                                    "last_error": last_err,
                                },
                            )
                            break

                        if stagnant >= 3 and attempts > 0:
                            extra = min(30.0, base_delay * (2 ** min(stagnant - 2, 4)))
                            logger.info(
                                "flatten_skip_market_close_stagnant abs=%s stagnant=%s sleep_s=%.2f",
                                ending_abs,
                                stagnant,
                                extra,
                            )
                            time.sleep(extra)
                            continue

                        try:
                            resp = self._client.market_close(self._settings.symbol)
                            attempts += 1
                            st = None
                            if isinstance(resp, dict):
                                st = resp.get("status")
                            logger.debug(
                                "flatten_market_close_attempt n=%s status=%s detail=%s",
                                attempts,
                                st,
                                str(resp)[:240] if resp is not None else None,
                            )
                        except Exception as e:
                            last_err = str(e)[:500]
                            attempts += 1
                            logger.exception(
                                "flatten_market_close_attempt n=%s failed",
                                attempts,
                            )
                        time.sleep(base_delay)
                    else:
                        self._flatten_drain_private_ws()
                        refresh_account_only(
                            self._client,
                            self._state,
                            venue_account_address(self._settings),
                            self._storage,
                            self._pnl,
                            ingest_fills_via_rest=True,
                        )
                        self._flatten_drain_private_ws()
                        with self._state._lock:
                            ending_abs = abs(self._state.position.position_qty)
                        result = FlattenResult.TIMED_OUT
                        elapsed = self._clock.time() - t0
                        msg = f"flatten timed out with open inventory abs_qty={ending_abs}"
                        log_extra(
                            logger,
                            logging.CRITICAL,
                            msg,
                            {
                                "event": "flatten_timed_out",
                                "abs_position_qty": ending_abs,
                                "attempts": attempts,
                                "elapsed_seconds": elapsed,
                                "last_error": last_err,
                            },
                        )
                        self._log_event(
                            EventSeverity.CRITICAL,
                            "flatten_timed_out",
                            msg,
                            {
                                "flatten_result": result.value,
                                "starting_abs_qty": start_abs,
                                "ending_abs_qty": ending_abs,
                                "attempts": attempts,
                                "elapsed_seconds": elapsed,
                                "last_error": last_err,
                            },
                        )
        except Exception as e:
            last_err = str(e)[:500]
            logger.exception("flatten failed")
            with self._state._lock:
                ending_abs = abs(self._state.position.position_qty)
            result = FlattenResult.FAILED
            self._log_event(
                EventSeverity.ERROR,
                "flatten_failed",
                "flatten failed: exception",
                {
                    "flatten_result": result.value,
                    "starting_abs_qty": start_abs,
                    "ending_abs_qty": ending_abs,
                    "attempts": attempts,
                    "elapsed_seconds": self._clock.time() - t0,
                    "last_error": last_err,
                },
            )
        finally:
            with self._state._lock:
                self._state.flatten_mode = False
                residual = ending_abs >= 1e-8
                if result in (FlattenResult.TIMED_OUT, FlattenResult.FAILED):
                    self._state.flatten_incomplete = residual
                    self._state.flatten_residual_abs_qty = ending_abs if residual else 0.0
                else:
                    self._state.flatten_incomplete = False
                    self._state.flatten_residual_abs_qty = 0.0
                if self._state.bot_status == BotStatus.FLATTENING:
                    if self._state.killed:
                        self._state.bot_status = BotStatus.KILLED
                    elif result in (
                        FlattenResult.COMPLETED,
                        FlattenResult.SKIPPED_ALREADY_FLAT,
                    ):
                        self._state.bot_status = BotStatus.RUNNING
                        self._state.clear_pause()
                    else:
                        self._state.bot_status = BotStatus.PAUSED
                        # FlattenResult.FAILED or .TIMED_OUT -- the
                        # bot is parked with non-zero inventory and
                        # needs operator attention.
                        self._state.mark_paused(
                            f"post_flatten_{(result.value if result else 'unknown').lower()}"
                        )
            self._flatten_lock.release()

        return result

    # ------------------------------------------------------------------
    # Soft-flatten worker
    # ------------------------------------------------------------------
    # Patient post-only-only exit triggered by the position-drawdown
    # gate. Entered from one_tick (top-level branch) when the gate
    # fires; loops on each tick maintaining a single reduce-side
    # post-only at best price; exits and resumes RUNNING when the
    # position closes. No taker IOC anywhere on this path -- the
    # absolute drawdown / session-loss gates remain the hard floor.
    # ------------------------------------------------------------------

    def _enter_soft_flatten(
        self,
        ev: "position_drawdown_gate.PositionDrawdownEvaluation | None",
        *,
        force_phase: Optional[int] = None,
        taker_fallback_ticks: Optional[int] = None,
        trigger_reason: str = "position_drawdown_gate",
        log_message_override: Optional[str] = None,
        log_payload_override: Optional[dict[str, Any]] = None,
    ) -> None:
        """Enter soft-flatten mode.

        Args:
            ev: Position-drawdown evaluation (the original trigger). May be
                ``None`` for non-drawdown triggers (e.g. toxicity); in that
                case ``log_message_override`` and ``log_payload_override``
                must be supplied for diagnostic continuity.
            force_phase: When set to 2, skips the patient phase-1 (near-touch)
                window and starts at phase-2 pricing (far-touch ∓ 1 tick) on
                the very first tick. Used by the toxicity-trigger path which
                wants speed without taker fees. Drawdown-gate path leaves
                this ``None`` (default phased behaviour).
            taker_fallback_ticks: Per-entry override of the SF taker-fallback
                threshold. ``None`` means "use the global setting"
                (``SOFT_FLATTEN_TAKER_FALLBACK_TICKS``); ``0`` disables the
                fallback for this entry. Cannot exceed the global setting.
            trigger_reason: Short string identifying what triggered this
                entry (used in the event log payload).
            log_message_override / log_payload_override: For non-drawdown
                triggers, replaces the default position-drawdown-gate
                message + payload.
        """
        # v1.5.198 — re-entry cooldown. After SF exits, refuse to
        # re-enter for ``soft_flatten_reentry_cooldown_seconds`` even
        # if the trigger fires again. Eliminates the tox-hard SF
        # loop pattern (4708 SF starts in 3 min) observed in
        # snapshot v1.5.195-260527-161959 where each SF episode
        # completed in ~25ms and immediately re-fired on the next
        # tick. The cooldown is checked here BEFORE any other entry
        # work — defensive against the loop being re-invoked.
        # Drawdown-gate triggers (the originally-intended SF path)
        # have a 30-second re-arming window of their own (the gate's
        # breach_duration_required), so the cooldown is effectively
        # only seen by tox-hard / explicit-trigger paths.
        try:
            reentry_cooldown_s = float(
                getattr(
                    self._settings,
                    "soft_flatten_reentry_cooldown_seconds",
                    30.0,
                )
                or 0.0
            )
        except (TypeError, ValueError):
            reentry_cooldown_s = 30.0
        if reentry_cooldown_s > 0.0:
            last_exit = self._state.soft_flatten_last_exited_at_mono
            if last_exit is not None:
                elapsed_since_exit = (
                    self._clock.monotonic() - float(last_exit)
                )
                if elapsed_since_exit < reentry_cooldown_s:
                    # Defer this entry — log a single WARNING for
                    # operator visibility and return without mutating
                    # any SF state.
                    self._log_event(
                        EventSeverity.WARNING,
                        "soft_flatten_reentry_blocked",
                        (
                            f"soft-flatten re-entry deferred "
                            f"({elapsed_since_exit:.1f}s since last "
                            f"exit < {reentry_cooldown_s:.1f}s cooldown); "
                            f"trigger={trigger_reason}"
                        ),
                        {
                            "trigger_reason": trigger_reason,
                            "seconds_since_last_exit": round(
                                elapsed_since_exit, 3
                            ),
                            "reentry_cooldown_seconds": (
                                reentry_cooldown_s
                            ),
                        },
                    )
                    return

        # v1.5.291 — sub-min-notional residual guard (Fix A). A position
        # whose notional is below the max(venue, local) min-notional floor
        # CANNOT be flattened by a single closing order — the SF worker
        # detects this and exits with reason="residual_below_min_notional"
        # (see ``_run_soft_flatten_tick``), but only AFTER the episode has
        # started AND been counted toward the sf_fatigue ladder via
        # ``note_sf_event`` below. On a persistently sub-min-notional
        # position with a sticky trigger (toxicity-hard whose markout
        # window can't age out because there are no new fills, or a
        # drawdown gate that keeps re-arming), each re-entry produces a
        # NO-OP SF episode that the fatigue ladder counts as real churn —
        # escalating CLEAR->WIDEN->PAUSE->KILL on un-actionable retries.
        # This is the exact mechanism behind the
        # v1.5.290-260530-223859 tier-4 kill (12 no-op episodes ~30s
        # apart, killed over a $0.006 loss) and the identical pre-deploy
        # kill at 03:35. Guard HERE, before note_sf_event: a
        # sub-min-notional residual is correctly worked off by ordinary
        # passive quoting (opposite-side fills), not a taker flatten, so
        # suppress the episode entirely and do NOT count it. Re-evaluated
        # on every entry attempt, so it self-clears the instant the
        # position grows back above the floor. Uses the SAME
        # max(venue, local) floor + ``|qty| * mid`` test as the worker's
        # exit check, so a position that would immediately exit
        # "residual_below_min_notional" never starts an episode at all.
        try:
            _venue_min_ntn = float(
                getattr(self._client.symbol_spec, "min_notional_usd", 0.0)
                or 0.0
            )
            _local_min_ntn = float(
                getattr(self._settings, "min_quote_notional_usd", 0.0) or 0.0
            )
            _min_ntn_floor = max(_venue_min_ntn, _local_min_ntn)
            if _min_ntn_floor > 0.0:
                _pos_qty_now = float(self._state.position.position_qty)
                _mkt0 = self._state.market
                _bb0 = getattr(_mkt0, "best_bid", None) if _mkt0 else None
                _ba0 = getattr(_mkt0, "best_ask", None) if _mkt0 else None
                _mid0: Optional[float] = None
                if (
                    _bb0 is not None
                    and _ba0 is not None
                    and float(_bb0) > 0
                    and float(_ba0) > 0
                ):
                    _mid0 = (float(_bb0) + float(_ba0)) / 2.0
                if (
                    _mid0 is not None
                    and abs(_pos_qty_now) * _mid0 + 1e-9 < _min_ntn_floor
                ):
                    self._log_event(
                        EventSeverity.WARNING,
                        "soft_flatten_suppressed_sub_min_notional",
                        (
                            f"soft-flatten entry suppressed: residual "
                            f"{abs(_pos_qty_now):.4f} @ mid {_mid0:.6f} = "
                            f"${abs(_pos_qty_now) * _mid0:.3f} < min-notional "
                            f"${_min_ntn_floor:.3f}; cannot taker-flatten, "
                            f"leaving to passive quoting; "
                            f"trigger={trigger_reason}"
                        ),
                        {
                            "trigger_reason": trigger_reason,
                            "residual_qty": round(abs(_pos_qty_now), 6),
                            "residual_notional_usd": round(
                                abs(_pos_qty_now) * _mid0, 4
                            ),
                            "min_notional_usd": round(_min_ntn_floor, 4),
                        },
                    )
                    return
        except Exception:
            # Never fail the SF entry path on a guard bookkeeping error —
            # fall through to normal SF entry (the worker's own
            # residual_below_min_notional exit remains the backstop).
            logger.exception("soft_flatten_sub_min_notional_guard_failed")

        # v1.5.202 — SF fatigue ladder. Count this entry (re-entry
        # cooldown above has already gated us, so we know this is a
        # REAL episode and not a re-fire). The gate's evaluator runs
        # in the eligibility engine and decides whether the bot's
        # quoting should WIDEN, PAUSE, or KILL based on the cumulative
        # SF rate. See ``app/sf_fatigue_gate.py``.
        try:
            from app import sf_fatigue_gate as _sf_fatigue

            _sf_fatigue.note_sf_event(
                self._state.sf_fatigue,
                now_mono=self._clock.monotonic(),
            )
        except Exception:
            # Never fail the SF entry path on a fatigue-gate bookkeeping
            # error — better to let the bot continue defending than to
            # crash on an instrumentation bug.
            pass

        global_fallback = int(
            getattr(self._settings, "soft_flatten_taker_fallback_ticks", 0)
            or 0
        )
        if taker_fallback_ticks is None:
            effective_fallback: Optional[int] = (
                global_fallback if global_fallback > 0 else None
            )
        else:
            # Per-entry value cannot exceed the global ceiling. The
            # operator's env var is the upper bound; per-trigger
            # callers can request tighter (smaller) thresholds, not
            # looser ones.
            if global_fallback <= 0:
                effective_fallback = None
            else:
                effective_fallback = max(
                    0, min(int(taker_fallback_ticks), global_fallback)
                )
                if effective_fallback == 0:
                    effective_fallback = None
        # v1.5.33 — if TP is currently active when SF arms (toxicity
        # hard-trigger or drawdown-gate fires while we're parked in
        # TP), SF wins per design. Cleanly disarm TP first so its
        # resting maker order is cancelled and its event row is
        # finalized before SF's state-mutation block runs.
        if bool(getattr(self._state, "tp_active", False)):
            self._exit_take_profit(reason="sf_takes_over")
        # Snapshot mid for adverse-drift tracking. Read under the
        # state lock alongside the active flag to keep entry_mid +
        # active in agreement.
        with self._state._lock:
            self._state.soft_flatten_active = True
            now_mono_entry = self._clock.monotonic()
            self._state.soft_flatten_started_at_mono = now_mono_entry
            # v1.5.2 Phase 4D.3 — capture pre-SF position so the
            # post-SF cooldown gate (run at SF exit) knows which
            # direction to suppress re-accumulation toward.
            try:
                self._state.soft_flatten_pre_position_qty = float(
                    self._state.position.position_qty
                )
            except (TypeError, ValueError, AttributeError):
                self._state.soft_flatten_pre_position_qty = None
            self._state.bot_status = BotStatus.SOFT_FLATTENING
            self._state.soft_flatten_force_phase = force_phase
            self._state.soft_flatten_taker_fallback_ticks = effective_fallback
            entry_mid: Optional[float] = None
            mkt = self._state.market
            if mkt is not None:
                bb = getattr(mkt, "best_bid", None)
                ba = getattr(mkt, "best_ask", None)
                if (
                    bb is not None
                    and ba is not None
                    and float(bb) > 0
                    and float(ba) > 0
                ):
                    entry_mid = (float(bb) + float(ba)) / 2.0
            self._state.soft_flatten_entry_mid = entry_mid
            # Phase 4D wiring (v1.4.172): seed the phase-ladder state
            # alongside the legacy entry_mid. When the ladder is
            # disabled (default) these fields are ignored but kept in
            # sync so a mid-episode enable doesn't see stale data.
            # Start phase = 0 unless ``force_phase >= 2`` (the
            # toxicity-trigger entry path requests aggressive starting
            # pricing — translate that to PHASE_1 of the new ladder so
            # we still post-only at the far touch + 1 tick before any
            # taker step. Phase 2/3/4 are reserved for adaptive
            # escalation, never as an initial entry).
            initial_phase = 0
            if force_phase is not None and int(force_phase) >= 2:
                initial_phase = soft_flatten.PHASE_1_POST_ONLY_FAR_PLUS_TICK
            self._state.sf_phase_ladder_phase = initial_phase
            self._state.sf_phase_started_mono = now_mono_entry
            self._state.sf_consecutive_rejects_in_phase = 0
            self._state.sf_entry_mid_for_phase_ladder = entry_mid
            # Phase 4D.5 (v1.4.190) — reset action-throttle state on
            # SF enter so the first action of the episode is always
            # allowed (no carry-over from a previous SF or the
            # main quote loop).
            self._state.sf_last_action_mono = None
            self._state.sf_throttle_first_arm_logged = False
            entry_position_qty = float(self._state.position.position_qty)
        # Persist a row for this SF episode so dashboards can attribute
        # orders + fills back to the panic window. Best-effort: if the
        # write fails, SF still runs; the episode just won't be
        # DB-attributed (state.soft_flatten_event_id stays None).
        # Plan ref: plans/20260507-sf-frontend.md Phase 2.
        try:
            sf_event_id = self._storage.insert_soft_flatten_event(
                {
                    "ts_start": self._clock.now_utc().isoformat(),
                    "trigger_reason": trigger_reason,
                    "initial_force_phase": (
                        int(force_phase) if force_phase is not None else None
                    ),
                    "taker_fallback_ticks": (
                        int(effective_fallback)
                        if effective_fallback is not None
                        else None
                    ),
                    "entry_position_qty": entry_position_qty,
                    "entry_mid_price": entry_mid,
                }
            )
            with self._state._lock:
                self._state.soft_flatten_event_id = sf_event_id
        except Exception:
            logger.exception("soft_flatten_event_insert_failed")
        self._exec.cancel_all_orders_for_symbol()
        # Reset join-depth overlay: soft-flatten is a regime change
        # and the overlay's signal inputs (cross-rejects, fill rate)
        # become meaningless during the patient post-only-only exit.
        try:
            self._state.join_depth_controller.reset()
        except Exception:
            logger.exception(
                "join_depth_controller_reset_failed_on_soft_flatten"
            )
        # Clear the min-notional passive-block latches on SF entry.
        # Without this, a stale latch from a prior episode could
        # silently suppress the close-side placement for this fresh
        # SF run — the operator-visible symptom is "SF active but
        # no orders placed." Codex review 2026-05-08 / 1.1.36 fix.
        try:
            self._exec.clear_min_notional_passive_block()
        except Exception:
            logger.exception(
                "clear_min_notional_passive_block_failed_on_soft_flatten_enter"
            )
        if log_message_override is not None:
            message = log_message_override
            payload: dict[str, Any] = dict(log_payload_override or {})
        elif ev is not None:
            message = (
                f"position-drawdown gate triggered: "
                f"adverse={ev.adverse_bps:.1f}bps "
                f"(threshold {ev.threshold_bps:.1f}) for "
                f"{ev.breach_seconds:.0f}s -- entering post-only flatten"
            )
            payload = {
                "adverse_bps": round(ev.adverse_bps, 3),
                "threshold_bps": ev.threshold_bps,
                "breach_seconds": round(ev.breach_seconds, 3),
                "duration_required_seconds": ev.duration_required_seconds,
                "position_notional_usd": ev.position_notional_usd,
                "unrealized_pnl_usd": ev.unrealized_pnl_usd,
            }
        else:
            message = "soft-flatten started"
            payload = {}
        # Decorate the event payload with the trigger metadata so
        # operators / Codex / future Claude sessions can tell at a
        # glance which path triggered which behaviour.
        payload["trigger_reason"] = trigger_reason
        if force_phase is not None:
            payload["force_phase"] = int(force_phase)
        if effective_fallback is not None:
            payload["taker_fallback_ticks"] = int(effective_fallback)
        if entry_mid is not None:
            payload["entry_mid"] = round(float(entry_mid), 8)
        self._log_event(
            EventSeverity.WARNING,
            "soft_flatten_started",
            message,
            payload,
        )
        # Persist the SF-active intent immediately. Without this,
        # a crash during SOFT_FLATTENING comes back with the adverse
        # inventory but the flatten *intent* lost — the bot resumes
        # normal quoting on adverse position until the drawdown gate
        # re-fires (another 30+ s of breach). Codex review 2026-05-07
        # / 1.1.37 fix (HIGH-1). The regular throttled save would
        # skip this transition because SF paths return before the
        # one_tick tail save.
        self._save_persistent_runtime_state_now()

    def _exit_soft_flatten(self, *, reason: str = "position_closed") -> None:
        with self._state._lock:
            was_active = self._state.soft_flatten_active
            elapsed = (
                self._clock.monotonic() - self._state.soft_flatten_started_at_mono
                if self._state.soft_flatten_started_at_mono
                else 0.0
            )
            # Capture the SF event id BEFORE clearing it so we can
            # update its end-row outside the lock (storage write
            # mustn't run under state._lock — both have their own
            # locking and nesting risks deadlock).
            sf_event_id = self._state.soft_flatten_event_id
            self._state.soft_flatten_active = False
            self._state.soft_flatten_started_at_mono = None
            # v1.5.198 — stamp exit time for re-entry cooldown.
            # Prevents the toxicity-hard SF loop pattern observed in
            # v1.5.195-260527-161959 (4708 SF starts in 3 min, each
            # cycle ~25ms). The re-entry cooldown gates SF entry so
            # consecutive triggers must space out by at least
            # ``soft_flatten_reentry_cooldown_seconds``.
            self._state.soft_flatten_last_exited_at_mono = float(
                self._clock.monotonic()
            )
            self._state.position_drawdown_breach_started_at_mono = None
            # v1.5.2 Phase 4D.3 — arm the post-SF cooldown. If we
            # remember the pre-SF position direction, suppress the
            # side that would re-accumulate toward it. Cleared by
            # time; restart the bot to clear early.
            pre_qty = self._state.soft_flatten_pre_position_qty
            self._state.soft_flatten_pre_position_qty = None
            try:
                cooldown_enabled = bool(
                    getattr(self._settings, "post_sf_cooldown_enabled", True)
                )
                cooldown_s = float(
                    getattr(self._settings, "post_sf_cooldown_seconds", 60.0)
                    or 0.0
                )
            except (TypeError, ValueError):
                cooldown_enabled = True
                cooldown_s = 60.0
            if cooldown_enabled and cooldown_s > 0.0 and pre_qty is not None:
                if pre_qty > 0:
                    # Pre-SF LONG → suppress BUY (don't re-add long).
                    self._state.post_sf_cooldown_lean_side = Side.BUY
                elif pre_qty < 0:
                    # Pre-SF SHORT → suppress SELL.
                    self._state.post_sf_cooldown_lean_side = Side.SELL
                else:
                    # Pre-SF flat (rare, SF would've been a no-op).
                    self._state.post_sf_cooldown_lean_side = None
                if self._state.post_sf_cooldown_lean_side is not None:
                    self._state.post_sf_cooldown_until_mono = (
                        self._clock.monotonic() + cooldown_s
                    )
            # Clear per-entry SF parameters so the next entry starts
            # from a clean slate.
            self._state.soft_flatten_force_phase = None
            self._state.soft_flatten_taker_fallback_ticks = None
            self._state.soft_flatten_entry_mid = None
            # v1.4.192 — populate the SF event-id grace cache BEFORE
            # clearing ``soft_flatten_event_id``. Late-arriving WS
            # fills from the taker paths (``client.market_close`` /
            # ``place_ioc``) can still be tagged correctly within the
            # grace window. See ``BotState.sf_recent_event_id`` for
            # the bug history (v1.4.157 / v1.4.189 untagged-taker
            # incidents). The check uses ``> 0`` for the grace window;
            # setting ``SF_RECENT_EVENT_ID_GRACE_SECONDS=0`` disables.
            if sf_event_id is not None:
                grace_s = float(
                    getattr(
                        self._settings,
                        "sf_recent_event_id_grace_seconds",
                        30.0,
                    )
                    or 0.0
                )
                if grace_s > 0.0:
                    self._state.sf_recent_event_id = int(sf_event_id)
                    self._state.sf_recent_event_id_valid_until_mono = (
                        self._clock.monotonic() + grace_s
                    )
            self._state.soft_flatten_event_id = None
            # Phase 4D wiring (v1.4.172): reset phase-ladder state.
            # No-op when the ladder is disabled (default); keeps the
            # fields in their default state for the next episode.
            self._state.sf_phase_ladder_phase = 0
            self._state.sf_phase_started_mono = 0.0
            self._state.sf_consecutive_rejects_in_phase = 0
            self._state.sf_entry_mid_for_phase_ladder = None
            if self._state.bot_status == BotStatus.SOFT_FLATTENING:
                self._state.bot_status = BotStatus.RUNNING
        if was_active:
            # Best-effort cancel of any still-resting flatten order
            # (private-WS lifecycle should already have updated it,
            # but a paranoid sweep keeps the resume-quoting path
            # clean).
            self._exec.cancel_all_orders_for_symbol()
            # Clear the min-notional passive-block latches on SF
            # exit so the next regular-MM placement attempt isn't
            # silently suppressed by a stale latch from this SF
            # episode. Codex review 2026-05-08 / 1.1.36 fix.
            try:
                self._exec.clear_min_notional_passive_block()
            except Exception:
                logger.exception(
                    "clear_min_notional_passive_block_failed_on_soft_flatten_exit"
                )
            self._log_event(
                EventSeverity.INFO,
                "soft_flatten_completed",
                (
                    f"soft flatten completed via post-only ({reason}); "
                    f"resuming RUNNING after {elapsed:.1f}s"
                ),
                {"elapsed_seconds": round(elapsed, 3), "reason": reason},
            )
            # Close out the SF episode row (best-effort; if the start
            # insert failed earlier, sf_event_id will be None and we
            # skip). Plan ref: plans/20260507-sf-frontend.md Phase 2.
            #
            # v1.4.173 (Phase 4D.4): also roll up the per-phase fill
            # counts + average taker spread paid, stamped on the same
            # row in one combined UPDATE. The rollup query reads the
            # fills table by ``soft_flatten_event_id`` so it runs after
            # private-WS has had a chance to land the SF episode's
            # final fills (drain happened at the top of the tick that
            # decided to exit; the close-side fill — IOC or market — is
            # already in the table). Stats stay NULL on episodes
            # captured before this migration / before the ladder was
            # active, which the dashboard interprets as "ladder not
            # active" and renders the legacy SF marker tooltip.
            if sf_event_id is not None:
                try:
                    end_payload: dict[str, Any] = {
                        "ts_end": self._clock.now_utc().isoformat(),
                        "exit_reason": reason,
                    }
                    try:
                        rollup = self._storage.compute_sf_episode_phase_totals(
                            int(sf_event_id)
                        )
                        if rollup:
                            counts = rollup.get("fills_by_phase")
                            if counts is not None:
                                end_payload["fills_by_phase_json"] = (
                                    json.dumps(counts, separators=(",", ":"))
                                )
                            taker_bps = rollup.get("taker_spread_bps_paid")
                            if taker_bps is not None:
                                end_payload["taker_spread_bps_paid"] = (
                                    float(taker_bps)
                                )
                    except Exception:
                        # Rollup failure shouldn't block the episode close.
                        logger.exception(
                            "soft_flatten_phase_rollup_failed sf_event_id=%s",
                            sf_event_id,
                        )
                    self._storage.update_soft_flatten_event_end(
                        sf_event_id,
                        end_payload,
                    )
                except Exception:
                    logger.exception(
                        "soft_flatten_event_end_update_failed sf_event_id=%s",
                        sf_event_id,
                    )
            # Persist SF-inactive immediately so a restart between
            # exit-now and the next throttled tail save doesn't come
            # back believing SF is still active. Codex review
            # 2026-05-07 / 1.1.37 fix (HIGH-1).
            self._save_persistent_runtime_state_now()

    # ------------------------------------------------------------------
    # v1.5.33 — take-profit (TP) opportunistic harvest mode.
    # See ``app/take_profit.py`` for the pure helpers. TP is structured
    # as an OVERLAY on the bot's quoting loop:
    #   * arming check runs at the top of ``one_tick`` (next to the
    #     SF / drawdown-gate arm),
    #   * when active, the TP executor takes over — normal quoting
    #     suspends, a single aggressive post-only close is maintained,
    #   * disarm conditions (uPnL retraced, dwell timeout, position-
    #     flat, SF takeover) restore normal quoting + arm a cooldown.
    # ------------------------------------------------------------------

    def _enter_take_profit(
        self,
        *,
        upnl_bps: float,
        close_side: "Side",
        target_price: float,
        entry_mid: Optional[float],
    ) -> None:
        """Transition into TP mode. Snapshots entry state, persists
        the tp_events row, cancels the existing ladder, places a
        single aggressive post-only close order.
        """
        now_mono = self._clock.monotonic()
        with self._state._lock:
            self._state.tp_active = True
            self._state.tp_armed_at_mono = now_mono
            self._state.tp_close_side = close_side
            self._state.tp_target_price = float(target_price)
            self._state.tp_entry_upnl_bps = float(upnl_bps)
            try:
                self._state.tp_entry_position_qty = float(
                    self._state.position.position_qty
                )
            except (TypeError, ValueError, AttributeError):
                self._state.tp_entry_position_qty = None
            self._state.tp_armed_total = (
                int(self._state.tp_armed_total) + 1
            )
            self._state.tp_sum_upnl_bps_at_arm += float(upnl_bps)
            entry_qty = self._state.tp_entry_position_qty
        try:
            tp_event_id = self._storage.insert_tp_event(
                {
                    "ts_start": self._clock.now_utc().isoformat(),
                    "trigger_upnl_bps": float(upnl_bps),
                    "trigger_threshold_bps": float(
                        self._settings.upnl_harvest_trigger_bps
                    ),
                    "disarm_margin_bps": float(
                        self._settings.upnl_harvest_disarm_margin_bps
                    ),
                    "entry_position_qty": entry_qty,
                    "entry_mid_price": (
                        float(entry_mid) if entry_mid is not None else None
                    ),
                    "entry_target_price": float(target_price),
                    "close_side": close_side.value,
                }
            )
            with self._state._lock:
                self._state.tp_event_id = tp_event_id
        except Exception:
            logger.exception("tp_event_insert_failed")
        self._log_event(
            EventSeverity.INFO,
            "tp_armed",
            (
                f"take-profit armed: uPnL={upnl_bps:.1f}bps "
                f"(trigger {self._settings.upnl_harvest_trigger_bps:.1f}) "
                f"-- placing aggressive post-only "
                f"{close_side.value} @ {target_price}"
            ),
            {
                "upnl_bps": round(float(upnl_bps), 3),
                "trigger_bps": float(
                    self._settings.upnl_harvest_trigger_bps
                ),
                "close_side": close_side.value,
                "target_price": float(target_price),
                "entry_mid_price": (
                    float(entry_mid) if entry_mid is not None else None
                ),
                "entry_position_qty": entry_qty,
            },
        )
        # Cancel the existing ladder. TP owns the book until disarm.
        try:
            self._exec.cancel_all_orders_for_symbol()
        except Exception:
            logger.exception("tp_cancel_all_failed_on_enter")
        # Place the TP order. Reduce-only — venue enforces "close only"
        # so a runaway buy can't accidentally open a new opposite
        # position if the WS position state is stale.
        size = abs(float(entry_qty or 0.0))
        if size <= 0:
            # Should be impossible (arming gate checks for inventory)
            # but guard against race with a concurrent flatten.
            logger.warning("tp_enter_zero_size_skipped")
            return
        cycle = f"take_profit_{int(now_mono * 1000)}"
        try:
            self._exec.place_passive_order_manual_only(
                close_side,
                float(target_price),
                size,
                cycle,
                reduce_only=True,
            )
        except Exception:
            logger.exception(
                "tp_place_failed side=%s price=%s size=%s",
                close_side.value,
                target_price,
                size,
            )

    def _exit_take_profit(self, *, reason: str) -> None:
        """Transition out of TP mode. Cancels any resting TP order,
        clears state flags, arms the re-arm cooldown, finalizes the
        tp_events row, and emits a telemetry log line.
        """
        now_mono = self._clock.monotonic()
        with self._state._lock:
            if not self._state.tp_active:
                return
            armed_at = self._state.tp_armed_at_mono or now_mono
            dwell_s = max(0.0, now_mono - armed_at)
            tp_event_id = self._state.tp_event_id
            entry_upnl = self._state.tp_entry_upnl_bps
            # Bump per-reason counters.
            if reason == "position_flat":
                self._state.tp_filled_total = (
                    int(self._state.tp_filled_total) + 1
                )
                # Approximate realised uPnL_bps by entry-side value.
                # A more accurate "uPnL at fill" would require reading
                # the just-filled price; entry is good enough for the
                # rolling average + cheaper to compute.
                if entry_upnl is not None:
                    self._state.tp_sum_upnl_bps_at_fill += float(entry_upnl)
            elif reason == "sf_takes_over":
                self._state.tp_exited_sf_takeover_total = (
                    int(self._state.tp_exited_sf_takeover_total) + 1
                )
            else:
                # "upnl_retraced" or "max_dwell" — opportunity passed.
                self._state.tp_exited_unfilled_total = (
                    int(self._state.tp_exited_unfilled_total) + 1
                )
            # Arm cooldown so we don't ping-pong near the threshold.
            cooldown_s = float(
                getattr(
                    self._settings,
                    "upnl_harvest_arm_cooldown_seconds",
                    10.0,
                )
                or 0.0
            )
            self._state.tp_arm_cooldown_until_mono = now_mono + cooldown_s
            # Move tp_event_id into grace cache. TTL mirrors SF's 30s
            # default for late-WS fills.
            if tp_event_id is not None:
                self._state.tp_recent_event_id = tp_event_id
                self._state.tp_recent_event_id_valid_until_mono = (
                    now_mono + 30.0
                )
            # Clear active flags.
            self._state.tp_active = False
            self._state.tp_armed_at_mono = None
            self._state.tp_event_id = None
            self._state.tp_target_price = None
            self._state.tp_close_side = None
            self._state.tp_entry_upnl_bps = None
            self._state.tp_entry_position_qty = None
        # Cancel any resting TP order. Safe-cancel: the order is
        # reduce-only post-only; an organic fill before our cancel
        # is just a clean exit.
        try:
            self._exec.cancel_all_orders_for_symbol()
        except Exception:
            logger.exception("tp_cancel_all_failed_on_exit")
        # Close out the tp_events row.
        if tp_event_id is not None:
            try:
                self._storage.update_tp_event_end(
                    int(tp_event_id),
                    {
                        "ts_end": self._clock.now_utc().isoformat(),
                        "exit_reason": reason,
                        "exit_upnl_bps": (
                            float(entry_upnl)
                            if entry_upnl is not None
                            else None
                        ),
                    },
                )
            except Exception:
                logger.exception(
                    "tp_event_end_update_failed tp_event_id=%s",
                    tp_event_id,
                )
        self._log_event(
            EventSeverity.INFO,
            "tp_disarmed",
            f"take-profit disarmed ({reason}) after {dwell_s:.1f}s",
            {
                "reason": reason,
                "dwell_seconds": round(dwell_s, 3),
                "entry_upnl_bps": (
                    round(float(entry_upnl), 3)
                    if entry_upnl is not None
                    else None
                ),
                "cooldown_seconds": cooldown_s,
            },
        )

    def _maybe_arm_take_profit(self) -> bool:
        """Called from the main quote loop. Evaluates the arming gate;
        on hit, transitions into TP mode and returns True (caller
        should ``return`` and let the next tick run the TP executor).
        """
        if not bool(
            getattr(self._settings, "upnl_harvest_enabled", False)
        ):
            return False
        # Build inputs.
        with self._state._lock:
            position_qty = float(self._state.position.position_qty)
            position_notional = float(
                self._state.position.position_notional or 0.0
            )
            unrealized = self._state.pnl.unrealized_pnl_usd
            sf_active = bool(self._state.soft_flatten_active)
            tp_active = bool(self._state.tp_active)
            cooldown_until = float(
                self._state.tp_arm_cooldown_until_mono or 0.0
            )
            mkt = self._state.market
        if tp_active or sf_active:
            return False
        upnl_bps = take_profit.compute_upnl_bps(
            unrealized_pnl_usd=unrealized,
            position_notional_usd=position_notional,
        )
        # Min-notional floor: same as SF's below-min-notional skip.
        venue_min_ntn = float(
            getattr(self._client.symbol_spec, "min_notional_usd", 0.0)
            or 0.0
        )
        local_min_ntn = float(
            getattr(self._settings, "min_quote_notional_usd", 0.0) or 0.0
        )
        min_ntn = max(venue_min_ntn, local_min_ntn)
        bot_status_allows_quoting = (
            not self._state.killed
            and not self._state.manual_pause
            and not self._state.flatten_mode
            and self._state.bot_status == BotStatus.RUNNING
        )
        decision = take_profit.should_arm_tp(
            enabled=True,
            trigger_bps=float(
                self._settings.upnl_harvest_trigger_bps
            ),
            upnl_bps=upnl_bps,
            position_qty=position_qty,
            position_notional_usd=position_notional,
            min_notional_usd=min_ntn,
            sf_active=sf_active,
            tp_active=tp_active,
            now_mono=self._clock.monotonic(),
            arm_cooldown_until_mono=cooldown_until,
            bot_status_allows_quoting=bot_status_allows_quoting,
        )
        if not decision.should_arm:
            return False
        # Compute target price + close side.
        if (
            mkt is None
            or mkt.best_bid is None
            or mkt.best_ask is None
            or float(mkt.best_bid) <= 0
            or float(mkt.best_ask) <= 0
            or float(mkt.best_bid) >= float(mkt.best_ask)
        ):
            # No usable book; skip. Next tick re-evaluates.
            return False
        tick_size = float(
            getattr(self._client.symbol_spec, "price_tick", 0.0) or 0.0
        )
        close_side, target_price = take_profit.compute_tp_target_price(
            pos_qty=position_qty,
            best_bid=float(mkt.best_bid),
            best_ask=float(mkt.best_ask),
            tick_size=tick_size,
        )
        entry_mid = (float(mkt.best_bid) + float(mkt.best_ask)) / 2.0
        # decision.upnl_bps is guaranteed non-None here (arm path).
        self._enter_take_profit(
            upnl_bps=float(decision.upnl_bps or 0.0),
            close_side=close_side,
            target_price=target_price,
            entry_mid=entry_mid,
        )
        return True

    def _run_take_profit_tick(self) -> None:
        """Per-tick maintenance of TP mode. Mirrors
        ``_run_soft_flatten_tick`` for the shared bits (drain private
        events, account refresh) and adds the disarm check + amend-if-
        moved logic.
        """
        try:
            self._exec.drain_private_events(self._pnl)
        except Exception:
            logger.exception("tp drain_private_events failed")
        try:
            self._exec.poke_cancel_pending_recovery()
        except Exception:
            logger.exception("tp poke_cancel_pending_recovery failed")
        addr = venue_account_address(self._settings)
        do_acc_refresh, _ = self._should_refresh_account_rest(addr)
        if do_acc_refresh:
            try:
                refresh_account_only(
                    self._client,
                    self._state,
                    addr,
                    self._storage,
                    self._pnl,
                    ingest_fills_via_rest=False,
                )
            except Exception:
                logger.exception("tp account refresh failed")

        # Recompute uPnL + check disarm.
        with self._state._lock:
            position_qty = float(self._state.position.position_qty)
            position_notional = float(
                self._state.position.position_notional or 0.0
            )
            unrealized = self._state.pnl.unrealized_pnl_usd
            armed_at = self._state.tp_armed_at_mono or 0.0
            mkt = self._state.market
            target_price_existing = self._state.tp_target_price
            close_side_existing = self._state.tp_close_side
        upnl_bps = take_profit.compute_upnl_bps(
            unrealized_pnl_usd=unrealized,
            position_notional_usd=position_notional,
        )
        now_mono = self._clock.monotonic()
        position_flat = abs(position_qty) < 1e-8
        disarm = take_profit.should_disarm_tp(
            upnl_bps=upnl_bps,
            trigger_bps=float(
                self._settings.upnl_harvest_trigger_bps
            ),
            disarm_margin_bps=float(
                self._settings.upnl_harvest_disarm_margin_bps
            ),
            armed_at_mono=armed_at,
            now_mono=now_mono,
            max_dwell_seconds=float(
                self._settings.upnl_harvest_max_dwell_seconds
            ),
            sf_armed_this_tick=False,  # SF takeover handled below
            position_flat=position_flat,
        )
        if disarm.should_disarm:
            self._exit_take_profit(reason=disarm.reason)
            return
        # Not disarming. Amend-if-moved: if the touch has shifted,
        # recompute the target price and re-place. Skipping the amend
        # is fine — we just sit on the old price; but ditching stale
        # quotes is cheap and helps fill probability if the touch
        # tightened in our favour.
        if (
            mkt is None
            or mkt.best_bid is None
            or mkt.best_ask is None
            or float(mkt.best_bid) <= 0
            or float(mkt.best_ask) <= 0
            or float(mkt.best_bid) >= float(mkt.best_ask)
        ):
            # Bad book — skip placement this tick, hold position. Don't
            # cancel: any resting TP order is reduce-only and fine to
            # leave parked momentarily.
            return
        tick_size = float(
            getattr(self._client.symbol_spec, "price_tick", 0.0) or 0.0
        )
        close_side, new_target = take_profit.compute_tp_target_price(
            pos_qty=position_qty,
            best_bid=float(mkt.best_bid),
            best_ask=float(mkt.best_ask),
            tick_size=tick_size,
        )
        # If close-side flipped (position flipped sign while in TP),
        # exit and let the next tick re-arm cleanly. Very rare; defensive.
        if (
            close_side_existing is not None
            and close_side != close_side_existing
        ):
            self._exit_take_profit(reason="side_flip")
            return
        # Only re-place when the target moved by at least one tick;
        # avoids spam on micro-fluctuations.
        moved_ticks = (
            abs(float(new_target) - float(target_price_existing or 0.0))
            / tick_size
            if tick_size > 0
            else 0.0
        )
        if moved_ticks < 0.5:
            return
        # Re-place: cancel + post at the new target. Reduce-only.
        try:
            self._exec.cancel_all_orders_for_symbol()
        except Exception:
            logger.exception("tp_cancel_failed_on_amend")
        size = abs(position_qty)
        if size <= 0:
            return
        cycle = f"take_profit_{int(now_mono * 1000)}"
        try:
            self._exec.place_passive_order_manual_only(
                close_side,
                float(new_target),
                size,
                cycle,
                reduce_only=True,
            )
            with self._state._lock:
                self._state.tp_target_price = float(new_target)
                self._state.tp_close_side = close_side
        except Exception:
            logger.exception(
                "tp_replace_failed side=%s price=%s size=%s",
                close_side.value,
                new_target,
                size,
            )

    def _risk_kill_check_during_soft_flatten(self) -> bool:
        """Abbreviated risk eval that runs before the soft-flatten
        worker. Catches kill conditions (drawdown, session loss,
        exec errors, public-WS disconnect, stale data) that would
        otherwise be skipped because soft-flatten short-circuits the
        normal one_tick risk path.

        Returns ``True`` if the caller should skip running the SF
        worker this tick:

          * ``KILL``     — transition out of SF and kill the bot
          * ``FLATTEN``  — transition out of SF and aggressive flatten
          * ``CANCEL_ALL`` — cancel any resting SF orders, hold SF
                            mode, skip placement until risk clears
                            (Codex 2026-05-09 HIGH-2)
          * ``NO_QUOTE`` — skip placement, hold SF mode (resting
                          orders left in place; reduce-only is still
                          a useful exit if they fill organically)
                          (Codex 2026-05-09 HIGH-2)

        Returns ``False`` for ALLOW / per-side actions
        (BID_ONLY / ASK_ONLY) — the SF worker handles per-side
        considerations internally, so it should run.

        Implementation: drains pending fills + refreshes PnL state
        (so drawdown / session-loss are evaluated against the freshest
        position + price), then runs ``evaluate_risk`` with the same
        kwargs the main ``one_tick`` would use.
        """
        # Drain so we see the freshest fills (the soft-flatten worker
        # would do this anyway; doing it here lets risk eval see
        # accurate position + PnL before worker placement decisions).
        try:
            self._exec.drain_private_events(self._pnl)
        except Exception:
            logger.exception("soft_flatten_risk_check_drain_failed")
        # Refresh account REST if it's been long enough; same
        # throttling as the main path. Without this, drawdown/session
        # loss are evaluated against stale equity.
        addr = venue_account_address(self._settings)
        do_refresh, _ = self._should_refresh_account_rest(addr)
        if do_refresh:
            ingest_rest = self._exec.should_ingest_fills_via_rest()
            try:
                refresh_account_only(
                    self._client,
                    self._state,
                    addr,
                    self._storage,
                    self._pnl,
                    ingest_fills_via_rest=ingest_rest,
                )
            except Exception:
                logger.exception("soft_flatten_risk_check_account_refresh_failed")

        # Build the same risk-eval inputs the main path uses.
        with self._state._lock:
            pos = self._state.position
            tox_snap = self._state.toxicity
            reconcile_ap = self._state.reconcile_auto_pause
            equity = (
                self._state.account.equity_usd
                if self._state.account
                else None
            )
        pnl_snap = self._pnl.build_snapshot(pos, equity)
        with self._state._lock:
            self._state.pnl = pnl_snap

        # Account-data-stale measurement (same as main path).
        # 2026-05-16 Codex #1 fix: read the success-only clock so the
        # gate measures "time since last good data", not "time since
        # last retry attempt".
        acct_last_m = self._state.account_rest_last_success_monotonic
        acct_age_s = (
            None if acct_last_m is None else (self._clock.monotonic() - acct_last_m)
        )

        risk = evaluate_risk(
            self._settings,
            bot_status=self._state.bot_status,
            manual_pause=self._state.manual_pause,
            killed=self._state.killed,
            # ``flatten_mode`` is the OLD aggressive flatten flag; we
            # are in soft_flatten_active which is distinct. Pass the
            # underlying flag faithfully (False during soft-flatten).
            flatten_mode=self._state.flatten_mode,
            market=self._state.market,
            position_qty=pos.position_qty,
            position_notional=pos.position_notional,
            open_order_count=self._state.open_order_count(),
            pnl=pnl_snap,
            toxicity=tox_snap,
            execution_errors=self._state.execution_errors_window_snapshot(
                self._settings.execution_errors_window_seconds
            )["windowed"],
            desync=self._state.order_desync,
            desync_phase=self._state.desync_phase,
            desync_quarantine_remaining=self._state.desync_quarantine_remaining,
            trades_last_minute=self._state.trades_last_minute(),
            reconcile_auto_pause=reconcile_ap,
            account_seconds_since_refresh=acct_age_s,
            **self._public_ws_risk_kwargs(),
        )
        if risk.action == RiskAction.KILL:
            # Kill takes priority over soft-flatten. The kill path
            # cancels all orders and (by default) market-flattens,
            # so we don't need to also exit soft_flatten cleanly --
            # ``self.kill`` sets bot_status = KILLED and the next
            # tick's ``is_killed()`` check at the top of one_tick
            # short-circuits before soft-flatten dispatch. We just
            # clear the soft_flatten_active flag here so heartbeat
            # / dashboards stop reporting that mode.
            with self._state._lock:
                self._state.soft_flatten_active = False
                self._state.soft_flatten_started_at_mono = None
            self.kill(
                ",".join(risk.reasons),
                self._build_kill_payload(risk.reasons),
            )
            return True
        if risk.action == RiskAction.FLATTEN:
            # Hard market-flatten escalation overrides patient
            # post-only flatten. Set bot_status to FLATTENING and
            # call the existing ``flatten`` (taker IOC).
            with self._state._lock:
                self._state.soft_flatten_active = False
                self._state.soft_flatten_started_at_mono = None
            self.flatten(blocking=True)
            return True
        # Codex 2026-05-09 HIGH-2: pre-fix this method only gated on
        # KILL / FLATTEN; NO_QUOTE and CANCEL_ALL fell through and the
        # SF worker happily continued placing quotes against a stale
        # book / desync state / max-orders limit. Treat the same
        # severe states the main path treats them.
        if risk.action == RiskAction.CANCEL_ALL:
            # Cancel any resting SF orders so they don't sit on the
            # book during a degraded state. Hold SF mode so we resume
            # placement when conditions clear.
            self._cancel_resting_soft_flatten_orders(reasons=tuple(risk.reasons))
            return True
        if risk.action == RiskAction.NO_QUOTE:
            # Skip placement. Don't cancel resting orders — they're
            # reduce-only and a counterparty hit is still a clean
            # exit. The next tick re-evaluates and either resumes or
            # escalates.
            return True
        return False

    def _cancel_resting_soft_flatten_orders(
        self, *, reasons: tuple[str, ...] = ()
    ) -> None:
        """Cancel any resting SF reduce-only orders. Used by the
        SF risk-check path when a CANCEL_ALL severity fires. Resting
        orders go through the same cancel-quote-path the regular
        execution layer uses; failures bump exec-errors but never
        raise.
        """
        with self._state._lock:
            # v1.4.194: migrated off the deprecated property shims.
            wo_bid = self._state.get_working_order(Side.BUY, 0)
            wo_ask = self._state.get_working_order(Side.SELL, 0)
        for wo in (wo_bid, wo_ask):
            if wo is None:
                continue
            if wo.status not in (
                OrderStatus.ACKED,
                OrderStatus.PARTIAL,
                OrderStatus.SENT,
            ):
                continue
            try:
                self._exec._enqueue_cancel_quote_path(
                    wo, trigger_reason="soft_flatten"
                )
            except Exception:
                logger.exception(
                    "soft_flatten_cancel_all_failed reasons=%s side=%s",
                    ",".join(reasons) or "?",
                    wo.side.value if hasattr(wo.side, "value") else str(wo.side),
                )

    def _run_soft_flatten_tick(self) -> None:
        """Per-tick maintenance of the soft-flatten state.

        Order of operations mirrors ``one_tick`` for the parts that
        soft-flatten cares about (private-WS drain so we see fills,
        account refresh so position is fresh) but skips quoting,
        toxicity, risk eval, etc. The result is ~one cancel-and-place
        round trip per tick when the price moves; no work otherwise.
        """
        addr = venue_account_address(self._settings)
        # Advance the execution tick counter for SF ticks too. Without
        # this, the periodic-modulo gate inside
        # ``should_ingest_fills_via_rest`` is driven by a counter that
        # stops advancing during SF, so the periodic REST fill-reconcile
        # path never fires until SF exits — even on long SF episodes
        # where private WS may have silently missed a fill. Codex review
        # 2026-05-07 / 1.1.37 fix (MEDIUM). Calling it here also resets
        # per-tick public-WS counters so SF-tick observability matches
        # normal-tick behavior.
        self._exec.on_bot_tick_start()
        ingest_rest_fills = self._exec.should_ingest_fills_via_rest()
        try:
            self._exec.drain_private_events(self._pnl)
        except Exception:
            logger.exception("soft_flatten drain_private_events failed")
        # Run the cancel-pending timeout/retry machinery during SF too.
        # Without this, an order that entered CANCEL_PENDING just before
        # SF entry (e.g. the regular MM cycle's reprice cancel was in
        # flight when toxicity-hard fired) is never resolved here —
        # ``place_passive_order_manual_only`` refuses to send the SF
        # flatten order while the same side has a CANCEL_PENDING WO,
        # producing a silent deadlock until the deadlock watchdog fires
        # at 600 s. Reproduced 2026-05-08, snapshot 260507112118; fix
        # in plans/<this commit>. Idempotent.
        try:
            self._exec.poke_cancel_pending_recovery()
        except Exception:
            logger.exception("soft_flatten poke_cancel_pending_recovery failed")
        do_acc_refresh, _ = self._should_refresh_account_rest(addr)
        if do_acc_refresh:
            try:
                refresh_account_only(
                    self._client,
                    self._state,
                    addr,
                    self._storage,
                    self._pnl,
                    ingest_fills_via_rest=ingest_rest_fills,
                )
            except Exception:
                logger.exception("soft_flatten account refresh failed")

        with self._state._lock:
            pos_qty = float(self._state.position.position_qty)
            # v1.4.194: migrated off the deprecated property shims.
            wo_bid = self._state.get_working_order(Side.BUY, 0)
            wo_ask = self._state.get_working_order(Side.SELL, 0)
            market = self._state.market

        # Dust check: lot-aware. Anything below half a lot is noise.
        # SymbolSpec exposes ``size_step`` and ``min_size`` (NOT
        # ``lot_size`` — that was a Codex-flagged typo, 2026-05-09).
        # ``size_step`` is the venue grid; ``min_size`` is the smaller
        # of (size_step, venue-min) when a venue advertises a smaller
        # min than its grid increment. Keep the same fallback chain
        # so non-conforming adapters still produce a sane lot.
        sp = self._client.symbol_spec
        lot = float(
            getattr(sp, "size_step", 0.0)
            or getattr(sp, "min_size", 0.0)
            or 0.0
        )
        dust_qty = max(lot * 0.5, 1e-9)
        if abs(pos_qty) < dust_qty:
            self._exit_soft_flatten()
            return
        # Notional dust: residual is too small to place a single
        # closing order that meets BOTH floors —
        #   * the venue's spec ``min_notional_usd`` (hard reject by venue),
        #   * the bot's local ``MIN_QUOTE_NOTIONAL_USD`` (soft reject in
        #     ``place_passive_order_manual_only``).
        # Take the MAX of the two; SF can't usefully place an order
        # below either. Pre-1.1.36 fix: only the venue floor was
        # checked, so on configurations where the local min-quote was
        # higher than the venue spec (TON has $5.5 vs $5), SF could
        # enter on a residual that's above venue min but below local
        # min, fail the placement, latch
        # ``_min_notional_passive_block``, and wedge silently. Codex
        # review 2026-05-08 (HIGH-2 in plans/codex-review-20260507.md).
        # Without this, the worker quietly stalls forever (the
        # pre-send risk check rejects the order, then sets a per-side
        # passive_block that short-circuits every subsequent attempt;
        # see execution.py:place_passive_order_manual_only). Treat as
        # job done and resume RUNNING -- normal quoting will work the
        # residual off via opposite-side fills, and the position-
        # drawdown gate has its own below-min-notional skip that
        # prevents re-entry.
        venue_min_ntn = float(getattr(sp, "min_notional_usd", 0.0) or 0.0)
        local_min_ntn = float(
            getattr(self._settings, "min_quote_notional_usd", 0.0) or 0.0
        )
        min_ntn = max(venue_min_ntn, local_min_ntn)
        # Codex review 2026-05-09 MED-5: a public-WS reconnect can
        # leave ``state.market`` populated but with one side
        # (``best_bid`` or ``best_ask``) still ``None`` while the
        # other has refreshed. The original code dereferenced both
        # sides as ``float(market.best_bid) + float(market.best_ask)``
        # which raised ``TypeError`` on the ``float(None)``, throwing
        # before the missing-book guard below could cancel quotes
        # and wait. Now: only compute the mid when BOTH sides are
        # populated; partial-book (or missing market) falls through
        # to the generic missing/invalid-book path which cancels
        # resting flatten orders and waits.
        if (
            min_ntn > 0
            and market is not None
            and market.best_bid is not None
            and market.best_ask is not None
        ):
            mid = (float(market.best_bid) + float(market.best_ask)) / 2.0
            if mid > 0 and abs(pos_qty) * mid + 1e-9 < min_ntn:
                self._exit_soft_flatten(reason="residual_below_min_notional")
                return

        # Defensive: locked / crossed books (best_bid >= best_ask)
        # produce broken phase-2 pricing -- ``best_bid + 1tick`` could
        # equal or exceed ``best_ask``, making any post-only SELL a
        # guaranteed REJECTED. On a degraded venue feed it's safer to
        # cancel resting flatten orders and wait for the book to come
        # back rather than churn rejects. Treated identically to
        # missing-book below.
        book_locked_or_crossed = (
            market is not None
            and market.best_bid is not None
            and market.best_ask is not None
            and float(market.best_bid) >= float(market.best_ask)
        )
        if (
            market is None
            or market.best_bid is None
            or market.best_ask is None
            or float(market.best_bid) <= 0
            or float(market.best_ask) <= 0
            or book_locked_or_crossed
        ):
            # Missing / invalid book. Pre-fix this just `return`'d,
            # leaving any resting flatten order parked at a stale
            # price (post-only at a now-irrelevant level) for the
            # operator to clean up manually. Cancel any outstanding
            # flatten order so we don't risk filling at a price we
            # never agreed to. Next tick the worker re-enters and
            # re-places when book is fresh.
            same_side_active_statuses = (
                OrderStatus.NEW_LOCAL,
                OrderStatus.SENT,
                OrderStatus.ACKED,
                OrderStatus.PARTIAL,
            )
            if (
                wo_bid is not None and wo_bid.status in same_side_active_statuses
            ) or (
                wo_ask is not None and wo_ask.status in same_side_active_statuses
            ):
                logger.warning(
                    "soft_flatten_no_book_cancel_resting "
                    "best_bid=%s best_ask=%s",
                    market.best_bid if market else None,
                    market.best_ask if market else None,
                )
                self._exec.cancel_all_orders_for_symbol()
            return

        # Phase 4D dispatch (v1.4.172). When the operator enables
        # ``SOFT_FLATTEN_PHASE_LADDER_ENABLED``, route the per-tick
        # decision through ``evaluate_sf_phase_ladder`` so post-only /
        # IOC / market_close are unified under one state machine.
        # Otherwise fall through to the legacy 2-phase post-only +
        # adverse-drift-taker-fallback path (preserved verbatim
        # below). The branch is config-only, no behaviour change for
        # profiles that don't opt in.
        if bool(getattr(self._settings, "sf_phase_ladder_enabled", False)):
            sp_tick = float(getattr(sp, "price_tick", 0.0) or 0.0)
            self._run_sf_phase_ladder_dispatch(
                pos_qty=pos_qty,
                market=market,
                wo_bid=wo_bid,
                wo_ask=wo_ask,
                tick_size=sp_tick,
            )
            return

        # Staged pricing. Phase 1 (default 5s) sits at the near-
        # touch. Phase 2 advances to one tick INTO the spread from
        # the far touch -- the most aggressive post-only price
        # possible; on a 1-tick spread it collapses to phase 1.
        # Pricing logic lives in ``soft_flatten`` so it can be unit-
        # tested without spinning up a Bot.
        #
        # Phase override (``soft_flatten_force_phase``): when set to
        # 2 (by the toxicity-trigger entry path), ``in_phase_2`` is
        # forced True regardless of elapsed time. Lets the operator
        # configure aggressive starting pricing for fast-moving
        # toxicity events without changing the patient default for
        # the position-drawdown gate.
        #
        # Phase 3 (taker fallback): if adverse mid drift exceeds
        # ``soft_flatten_taker_fallback_ticks`` since SF entry, the
        # worker exits SF via a single taker market_close. Accepts
        # taker fees as cheaper than continued post-only chasing
        # against an escaping price. Default disabled (= legacy
        # post-only-forever).
        # SymbolSpec exposes ``price_tick``, NOT ``tick_size``
        # (Codex-flagged typo, 2026-05-09). Pre-fix this returned 0
        # silently, gating phase-2 / reprice-tolerance / taker-fallback
        # off entirely on every venue with a real SymbolSpec — only
        # the MagicMock-based tests passed. Keep the variable name
        # local-only so callers reading this scope stay legible.
        tick_size = float(getattr(sp, "price_tick", 0.0) or 0.0)
        phase_1_s = float(self._settings.soft_flatten_phase_1_seconds)
        with self._state._lock:
            started = self._state.soft_flatten_started_at_mono
            force_phase = self._state.soft_flatten_force_phase
            fallback_ticks = self._state.soft_flatten_taker_fallback_ticks
            entry_mid = self._state.soft_flatten_entry_mid
        elapsed = (
            self._clock.monotonic() - started if started is not None else 0.0
        )
        in_phase_2 = elapsed >= phase_1_s
        if force_phase is not None and int(force_phase) >= 2:
            in_phase_2 = True

        # Phase 3 — adverse-drift escape via taker market_close.
        # Compute drift in the direction that hurts our position:
        #   long position (pos_qty > 0): adverse = mid moving DOWN
        #   short position (pos_qty < 0): adverse = mid moving UP
        # Fires only when a non-zero ticks threshold is set, the
        # tick size is positive, and we have a valid entry mid +
        # current mid to compare.
        if (
            fallback_ticks is not None
            and int(fallback_ticks) > 0
            and tick_size > 0.0
            and entry_mid is not None
            and entry_mid > 0
            and market is not None
            and market.best_bid is not None
            and market.best_ask is not None
        ):
            current_mid = (
                float(market.best_bid) + float(market.best_ask)
            ) / 2.0
            if pos_qty > 0:
                adverse_drift = float(entry_mid) - current_mid
            elif pos_qty < 0:
                adverse_drift = current_mid - float(entry_mid)
            else:
                adverse_drift = 0.0
            adverse_drift_ticks = (
                adverse_drift / tick_size if tick_size > 0 else 0.0
            )
            if adverse_drift_ticks >= float(int(fallback_ticks)):
                # Take the L. Cancel any resting SF orders, fire one
                # market reduce-only, exit SF.
                logger.warning(
                    "soft_flatten_taker_fallback_fired pos_qty=%s "
                    "entry_mid=%s current_mid=%s adverse_drift_ticks=%.2f "
                    "threshold_ticks=%s",
                    pos_qty,
                    entry_mid,
                    current_mid,
                    adverse_drift_ticks,
                    fallback_ticks,
                )
                try:
                    self._exec.cancel_all_orders_for_symbol()
                except Exception:
                    logger.exception(
                        "soft_flatten_taker_fallback_cancel_failed"
                    )
                # v1.4.173 (Phase 4D.4): synthetic ORDERS row for the
                # legacy adverse-drift market_close path too — same
                # rationale as the phase-ladder dispatcher's terminal
                # branch. ORDERS history would otherwise miss this
                # placement entirely (the fill gets SF-tagged at
                # ingestion via the v1.4.163 fallback, but no order row
                # exists for the dashboard's per-episode list).
                # ``phase=4`` is the terminal-market label; the legacy
                # path never set ``sf_phase_ladder_phase`` so we tag it
                # here for dashboard consistency.
                try:
                    self._persist_synthetic_sf_order_row(
                        phase=4,
                        side=(
                            Side.SELL if pos_qty > 0 else Side.BUY
                        ),
                        price=None,
                        size=abs(pos_qty),
                        post_only=False,
                        order_kind="market_close_legacy_drift",
                    )
                except Exception:
                    # Defensive; the helper already logs internally.
                    pass
                try:
                    self._client.market_close(self._settings.symbol)
                except Exception:
                    logger.exception(
                        "soft_flatten_taker_fallback_market_close_failed"
                    )
                self._exit_soft_flatten(
                    reason="taker_fallback_after_adverse_drift"
                )
                return

        close_side, target_price = soft_flatten.compute_target_price(
            pos_qty=pos_qty,
            best_bid=float(market.best_bid),
            best_ask=float(market.best_ask),
            tick_size=tick_size,
            in_phase_2=in_phase_2,
        )
        if close_side == Side.SELL:
            existing = wo_ask
            unwanted = wo_bid
        else:
            existing = wo_bid
            unwanted = wo_ask

        active_statuses = (
            OrderStatus.NEW_LOCAL,
            OrderStatus.SENT,
            OrderStatus.ACKED,
            OrderStatus.PARTIAL,
        )

        # Stray order on the wrong side -- a leftover from normal
        # quoting. Cancel before placing the flatten side; otherwise
        # both sides could fill simultaneously and unwind the close.
        if unwanted is not None and unwanted.status in active_statuses:
            self._exec.cancel_all_orders_for_symbol()
            return

        # ----------------------------------------------------------
        # SIZING — three caps + outstanding-order accounting
        # ----------------------------------------------------------
        # Without these clips, the worker quotes |position| as one
        # giant post-only at the touch (2026-05-06 incident: fills of
        # 2672 SUI / $2700 against $20 cap). Caps applied:
        #   1. ``MAX_ORDER_NOTIONAL_USD`` — single-order USD cap
        #   2. ``MAX_ABS_POSITION`` — base-units position cap
        #   3. ``MAX_POSITION_NOTIONAL_USD`` — USD position cap
        #
        # PLUS: outstanding live flatten order size is SUBTRACTED.
        # If an existing same-side order is already resting / sent,
        # its size already counts toward the close. Without subtracting,
        # the worker would pile a second flatten on top of the first
        # and over-flatten into the opposite direction.
        target_size = abs(pos_qty)
        max_order_notional = float(
            self._settings.max_order_notional_usd or 0.0
        )
        if max_order_notional > 0 and target_price > 0:
            target_size = min(target_size, max_order_notional / target_price)
        max_abs_pos = float(self._settings.max_abs_position or 0.0)
        if max_abs_pos > 0:
            target_size = min(target_size, max_abs_pos)
        max_pos_notional = float(
            self._settings.max_position_notional_usd or 0.0
        )
        if max_pos_notional > 0 and target_price > 0:
            target_size = min(target_size, max_pos_notional / target_price)

        # ``desired_total_size`` = how much SHOULD be on the book in
        # total to flatten the current position (capped). Outstanding
        # exposure on the close-side counts toward this; the new
        # order tops it up. This restructure (vs the previous
        # subtract-then-decide) ensures we always evaluate the
        # existing order's PRICE -- a stale-price existing order
        # that "covers" the position must still be re-priced.
        desired_total_size = target_size
        outstanding_close_sz = 0.0
        if existing is not None and existing.status in active_statuses:
            outstanding_close_sz = float(existing.size or 0.0)

        reprice_threshold = tick_size * float(
            self._settings.soft_flatten_reprice_ticks
        )

        # Phase 4D.5 (v1.4.190) — action-rate throttle. Compute the
        # gap-elapsed-since-last-action predicate ONCE per tick; reuse
        # for both the cancel-replace branch and the place branch.
        # See ``app/config.py`` ``sf_action_min_gap_ms`` for rationale.
        now_mono = self._clock.monotonic()
        sf_throttle_active = self._sf_action_throttled(now_mono)

        if existing is not None and existing.status in active_statuses:
            existing_px = float(existing.price)
            price_acceptable = (
                abs(existing_px - target_price) <= reprice_threshold + 1e-12
            )
            size_sufficient = (
                outstanding_close_sz + 1e-9 >= desired_total_size
            )
            if price_acceptable and size_sufficient:
                # Existing covers what we need at the right price.
                return
            # Either price drifted past tolerance OR the existing
            # order is too small (e.g. position grew while it rested,
            # so caps allow more now). Cancel and re-place; next tick
            # will issue a fresh order at the right size + price.
            if sf_throttle_active:
                self._note_sf_throttle_suppression("cancel_reprice")
                return
            self._exec.cancel_all_orders_for_symbol()
            self._state.sf_last_action_mono = self._clock.monotonic()
            return

        # No outstanding flatten order -- size the new one.
        new_size = max(0.0, desired_total_size - outstanding_close_sz)
        if new_size <= 0:
            return  # nothing to add this tick

        if sf_throttle_active:
            self._note_sf_throttle_suppression("place")
            return

        # Place with REDUCE-ONLY. Venue-side guarantee against the
        # snowball: OKX / Binance / et al. enforce that the order can
        # only *close* position, never grow it past zero. If position
        # is already flat, the venue rejects the order
        # (REDUCE_ONLY_REJECT) rather than letting a runaway grow
        # opposite-direction exposure.
        cycle = f"soft_flatten_{int(self._clock.monotonic() * 1000)}"
        try:
            self._exec.place_passive_order_manual_only(
                close_side,
                target_price,
                new_size,
                cycle,
                reduce_only=True,
            )
            self._state.sf_last_action_mono = self._clock.monotonic()
        except Exception:
            logger.exception(
                "soft_flatten_place_failed side=%s price=%s size=%s",
                close_side.value,
                target_price,
                new_size,
            )

    def _sf_action_throttled(self, now_mono: float) -> bool:
        """Phase 4D.5 (v1.4.190) — return True if the SF tick should
        suppress its cancel-replace / place action this iteration
        because the previous action was less than
        ``SF_ACTION_MIN_GAP_MS`` ago. Always False when the gap is 0.

        Pure predicate: does NOT mutate state. The caller is
        responsible for bumping the suppression counter via
        ``_note_sf_throttle_suppression`` if it returns True.
        """
        gap_ms = float(
            getattr(self._settings, "sf_action_min_gap_ms", 250.0) or 0.0
        )
        if gap_ms <= 0.0:
            return False
        last_mono = getattr(self._state, "sf_last_action_mono", None)
        if last_mono is None:
            return False
        return (now_mono - float(last_mono)) * 1000.0 < gap_ms - 1e-9

    def _note_sf_throttle_suppression(self, action_kind: str) -> None:
        """Phase 4D.5 (v1.4.190) — record an SF action that was
        suppressed by the throttle. Bumps the session counter +
        emits a first-arm WARNING (subsequent suppressions are
        silent; the counter tells the operator the rate)."""
        try:
            self._state.sf_throttle_suppressed_total += 1
            if not getattr(self._state, "sf_throttle_first_arm_logged", False):
                gap_ms = float(
                    getattr(self._settings, "sf_action_min_gap_ms", 250.0)
                    or 0.0
                )
                logger.warning(
                    "sf_action_throttle_armed action_kind=%s "
                    "gap_ms=%.0f sf_event_id=%s — first arm for this "
                    "SF episode; subsequent suppressions silent; "
                    "watch sf_throttle_suppressed_total",
                    action_kind,
                    gap_ms,
                    self._state.soft_flatten_event_id,
                )
                self._state.sf_throttle_first_arm_logged = True
        except Exception:
            logger.exception("sf_action_throttle_counter_bump_failed")

    def _persist_synthetic_sf_order_row(
        self,
        *,
        phase: int,
        side: Side,
        price: Optional[float],
        size: float,
        post_only: bool,
        order_kind: str,
    ) -> None:
        """v1.4.173 (Phase 4D.4) — write a stub ``orders`` row for an
        SF taker placement that bypasses ``OrderManager`` (the IOC fire-
        and-forget path + the legacy / phase-4 ``market_close`` path).

        Without this stub the ORDERS history in the dashboard
        underrepresents what SF actually did: phase-2/3 IOC firings +
        phase-4 market closes produce fills that get SF-tagged at
        ingestion (v1.4.163) but no order row, so the operator sees a
        fill "out of nowhere" rather than an order → fill chain.

        Best-effort. The row's primary key is a synthetic local id
        (``sf_p<phase>_<event>_<ms>``) so it can't collide with a real
        managed order. The exchange order id stays NULL — these
        placements are fire-and-forget and we never bind the response,
        so we have no real exchange oid to record. The
        ``soft_flatten_event_id`` column is the join key the dashboard
        uses to render "orders for this SF episode" panels.
        """
        try:
            sf_event_id = getattr(
                self._state, "soft_flatten_event_id", None
            )
            ts_ms = int(self._clock.monotonic() * 1000.0)
            now_iso = self._clock.now_utc().isoformat()
            row: dict[str, Any] = {
                "order_id_local": (
                    f"sf_p{int(phase)}_"
                    f"{int(sf_event_id) if sf_event_id is not None else 0}_"
                    f"{ts_ms}"
                ),
                "order_id_exchange": None,
                "client_order_id": None,
                "ts_created": now_iso,
                "ts_sent": now_iso,
                "ts_ack": None,
                "ts_closed": None,
                "symbol": self._settings.symbol,
                "side": side.value,
                "price": (
                    float(price) if price is not None and price > 0 else None
                ),
                "size": float(size),
                "post_only": 1 if post_only else 0,
                # Synthetic rows are fire-and-forget; we don't track a
                # full lifecycle. Marking SENT keeps the dashboard from
                # filtering them out (the "active" filter excludes
                # FILLED / CANCELED / REJECTED but includes SENT) and
                # the cancel_reason cell carries the placement kind for
                # operator forensics.
                "status": OrderStatus.SENT.value,
                "cancel_reason": f"sf_synthetic_{order_kind}",
                "replace_group_id": None,
                "quote_cycle_id": f"soft_flatten_p{int(phase)}",
                "level_idx": 0,
                "soft_flatten_event_id": (
                    int(sf_event_id) if sf_event_id is not None else None
                ),
            }
            self._storage.insert_order_row(row)
        except Exception:
            logger.exception(
                "sf_synthetic_order_row_failed phase=%s kind=%s",
                phase,
                order_kind,
            )

    def _run_sf_phase_ladder_dispatch(
        self,
        *,
        pos_qty: float,
        market: Any,
        wo_bid: Any,
        wo_ask: Any,
        tick_size: float,
    ) -> None:
        """Phase 4D wiring (v1.4.172).

        Per-tick dispatcher invoked from ``_run_soft_flatten_tick`` when
        ``SOFT_FLATTEN_PHASE_LADDER_ENABLED`` is on. Consults the pure
        helper ``evaluate_sf_phase_ladder`` (logic shipped in v1.4.164)
        and routes the resulting ``order_type`` to the matching transport:

        * ``post_only`` → ``place_passive_order_manual_only`` (existing
          managed path; same cancel-and-replace + sizing logic as the
          legacy 2-phase post-only worker).
        * ``ioc``       → ``client.place_ioc_reduce_only`` (NEW direct
          path; fire-and-forget, fills stamped as SF at ingestion via
          ``state.soft_flatten_event_id``).
        * ``market``    → ``client.market_close`` + ``_exit_soft_flatten``
          (terminal; equivalent to the legacy taker fallback).

        Targets the v1.4.157-260520-213540 failure (1,443 post-only
        place-cancels in 14 s → market_close at the worst spread). The
        drift-based + time-based escalation rules in the pure helper
        jump straight to IOC once the touch has run past the SF entry
        mid by ≥ ``SOFT_FLATTEN_FAST_ESCALATE_TICKS`` (default 3), so
        the bot crosses 1-2 ticks of spread for a high-probability fill
        instead of timing out and crossing whatever spread the market
        has by then.

        The reject-counter escalation rule (rule 2 in
        ``evaluate_sf_phase_ladder``) is on the helper's surface but
        currently fed ``0`` from this wiring — drift + time-based
        escalation alone cover the v1.4.157 scenario, and the
        reject-tracking plumbing is a tracked follow-up (see plan entry
        v1.4.172). Operator can lean on the drift threshold or the
        per-phase dwell budgets to tune escalation aggressiveness in
        the meantime.
        """
        now_mono = self._clock.monotonic()
        with self._state._lock:
            current_phase = int(self._state.sf_phase_ladder_phase)
            current_started = float(self._state.sf_phase_started_mono or 0.0)
            consecutive_rejects = int(
                self._state.sf_consecutive_rejects_in_phase
            )
            entry_mid_for_ladder = self._state.sf_entry_mid_for_phase_ladder
            # Defensive: if _enter_soft_flatten ran before the ladder
            # was enabled (mid-session config flip), entry_mid_for_ladder
            # may be None even though SF is active. Backfill from the
            # legacy field so the drift-escalation rule has a baseline.
            if entry_mid_for_ladder is None:
                entry_mid_for_ladder = self._state.soft_flatten_entry_mid
                self._state.sf_entry_mid_for_phase_ladder = (
                    entry_mid_for_ladder
                )
            # Defensive: if sf_phase_started_mono wasn't seeded (mid-
            # session enable), seed from now so the first dwell budget
            # starts ticking here.
            if current_started <= 0.0:
                current_started = now_mono
                self._state.sf_phase_started_mono = now_mono

        s = self._settings
        phase_durations = (
            float(getattr(s, "sf_phase_0_duration_seconds", 3.0)),
            float(getattr(s, "sf_phase_1_duration_seconds", 4.0)),
            float(getattr(s, "sf_phase_2_duration_seconds", 4.0)),
            float(getattr(s, "sf_phase_3_duration_seconds", 2.0)),
        )
        fast_escalate_ticks = float(
            getattr(s, "sf_fast_escalate_ticks", 3.0)
        )
        rejects_to_escalate = int(
            getattr(s, "sf_consecutive_rejects_to_escalate", 10)
        )

        # v1.5.198 — episode-level hard timeout. Anchored to the
        # initial SF entry (``soft_flatten_started_at_mono``), NOT
        # reset on phase transitions or re-entries. If the SF episode
        # has been active longer than ``soft_flatten_episode_max_duration_seconds``,
        # the ladder force-escalates to phase 4 (market_close) so the
        # episode terminates regardless of per-phase state. Addresses
        # the v1.5.195-260527-165258 30-min pause where SF hung in
        # phase 2 with a post-only that never filled.
        episode_started = float(
            getattr(self._state, "soft_flatten_started_at_mono", 0.0)
            or 0.0
        )
        episode_max_duration = float(
            getattr(
                self._settings,
                "soft_flatten_episode_max_duration_seconds",
                60.0,
            )
            or 0.0
        )

        decision = soft_flatten.evaluate_sf_phase_ladder(
            pos_qty=pos_qty,
            best_bid=float(market.best_bid),
            best_ask=float(market.best_ask),
            tick_size=tick_size,
            now_mono=now_mono,
            current_phase=current_phase,
            current_phase_started_mono=current_started,
            consecutive_rejects_in_phase=consecutive_rejects,
            entry_mid_for_phase_ladder=entry_mid_for_ladder,
            phase_durations_s=phase_durations,
            fast_escalate_ticks=fast_escalate_ticks,
            consecutive_rejects_to_escalate=rejects_to_escalate,
            # v1.5.198 — episode-level hard timeout.
            episode_started_mono=(
                episode_started if episode_started > 0.0 else None
            ),
            episode_max_duration_s=episode_max_duration,
        )

        # Persist transition + reset per-phase counters when the phase
        # changes. Single-line WARNING on each transition so the
        # operator can see escalation history in the bot log.
        if decision.new_phase != current_phase:
            logger.warning(
                "soft_flatten_phase_escalated pos_qty=%s prev_phase=%s "
                "new_phase=%s order_type=%s target_price=%s reason=%s",
                pos_qty,
                current_phase,
                decision.new_phase,
                decision.order_type,
                decision.target_price,
                decision.escalate_reason,
            )
            with self._state._lock:
                self._state.sf_phase_ladder_phase = decision.new_phase
                self._state.sf_phase_started_mono = (
                    decision.new_phase_started_mono
                )
                # Reset per-phase reject counter on phase transition.
                # Phase-2/3 IOC rejects don't accumulate the same way
                # post-only ones do, so cross-phase resets are the
                # right policy.
                self._state.sf_consecutive_rejects_in_phase = 0

        # ---- Dispatch by order_type ----------------------------------------

        # Phase 4D.5 (v1.4.191) — action-rate throttle for the phase-
        # ladder dispatcher. WITHOUT this, the dispatcher was firing
        # IOC placements (and the terminal market_close) on EVERY
        # quote-loop wake. Observed in snapshot
        # ``v1.4.189-260521-143033-prod.okx.ton.usdt.perp``: SF#11172
        # ran for 13.2 s and produced **1,249 orders** — 787 post-only
        # cancel/replaces, 461 IOC firings (none filled), and 1
        # terminal market_close (the operator-flagged taker fill at
        # $1.995). The 461 IOC bombardment is the same WS-wake pathology
        # the legacy SF tick has, just amplified because IOC is fire-
        # and-forget so there's no "wait for cancel ack" backpressure.
        #
        # Terminal market_close is NOT throttled — when the helper says
        # phase 4, we close. That's the safety net by design. IOC + the
        # post-only branches further down ARE throttled (they're the
        # bombardment sources).
        if decision.order_type != "market":
            if self._sf_action_throttled(self._clock.monotonic()):
                self._note_sf_throttle_suppression(
                    f"phase_ladder:{decision.order_type}"
                )
                return

        # Terminal: market_close + exit SF. Same semantics as the
        # legacy adverse-drift taker fallback.
        if decision.order_type == "market":
            try:
                self._exec.cancel_all_orders_for_symbol()
            except Exception:
                logger.exception(
                    "soft_flatten_phase4_cancel_failed"
                )
            # v1.4.173 (Phase 4D.4): stub orders row so the ORDERS
            # history reflects the market_close placement. Size is the
            # absolute remaining position (the venue computes the exact
            # close size internally); price stays NULL (market). Best-
            # effort write — failure doesn't block the close.
            self._persist_synthetic_sf_order_row(
                phase=decision.new_phase,
                side=decision.close_side,
                price=None,
                size=abs(pos_qty),
                post_only=False,
                order_kind="market_close",
            )
            try:
                self._client.market_close(self._settings.symbol)
                # Phase 4D.5 (v1.4.191) — record terminal market_close
                # for the throttle (so if SF re-enters quickly the gap
                # still applies to the first action of the next episode
                # ... in practice _enter_soft_flatten clears
                # sf_last_action_mono so this is moot, but the symmetry
                # with the IOC + post_only stamps is intentional).
                self._state.sf_last_action_mono = self._clock.monotonic()
            except Exception:
                logger.exception(
                    "soft_flatten_phase4_market_close_failed"
                )
            self._exit_soft_flatten(reason="phase_ladder_terminal_market")
            return

        # Both post_only and ioc need the close-side WO + sizing pass.
        # Pull the same sizing logic as the legacy post-only path so
        # all three caps (MAX_ORDER_NOTIONAL_USD, MAX_ABS_POSITION,
        # MAX_POSITION_NOTIONAL_USD) apply uniformly across phases.
        if decision.target_price is None or decision.target_price <= 0:
            # Defensive: should only happen for phase 4 (market), which
            # is handled above. If we got here with no price, skip.
            return

        if decision.close_side == Side.SELL:
            existing = wo_ask
            unwanted = wo_bid
        else:
            existing = wo_bid
            unwanted = wo_ask

        active_statuses = (
            OrderStatus.NEW_LOCAL,
            OrderStatus.SENT,
            OrderStatus.ACKED,
            OrderStatus.PARTIAL,
        )

        # Stray opposite-side order from prior normal quoting — cancel
        # before placing the flatten leg. Same hazard rationale as the
        # legacy path: simultaneous fills on both sides would unwind
        # the close.
        if unwanted is not None and unwanted.status in active_statuses:
            self._exec.cancel_all_orders_for_symbol()
            return

        target_size = abs(pos_qty)
        max_order_notional = float(
            self._settings.max_order_notional_usd or 0.0
        )
        if max_order_notional > 0 and decision.target_price > 0:
            target_size = min(
                target_size, max_order_notional / decision.target_price
            )
        max_abs_pos = float(self._settings.max_abs_position or 0.0)
        if max_abs_pos > 0:
            target_size = min(target_size, max_abs_pos)
        max_pos_notional = float(
            self._settings.max_position_notional_usd or 0.0
        )
        if max_pos_notional > 0 and decision.target_price > 0:
            target_size = min(
                target_size, max_pos_notional / decision.target_price
            )

        outstanding_close_sz = 0.0
        if existing is not None and existing.status in active_statuses:
            outstanding_close_sz = float(existing.size or 0.0)

        # ---- post_only branch: managed cancel-and-replace ------------
        if decision.order_type == "post_only":
            reprice_threshold = tick_size * float(
                self._settings.soft_flatten_reprice_ticks
            )
            if existing is not None and existing.status in active_statuses:
                existing_px = float(existing.price)
                price_acceptable = (
                    abs(existing_px - decision.target_price)
                    <= reprice_threshold + 1e-12
                )
                size_sufficient = (
                    outstanding_close_sz + 1e-9 >= target_size
                )
                if price_acceptable and size_sufficient:
                    return  # existing covers what we need
                # Price drift or undersized → cancel; next tick re-places.
                self._exec.cancel_all_orders_for_symbol()
                # Phase 4D.5 (v1.4.191) — stamp the cancel-replace step.
                self._state.sf_last_action_mono = self._clock.monotonic()
                return
            new_size = max(0.0, target_size - outstanding_close_sz)
            if new_size <= 0:
                return
            cycle = (
                f"soft_flatten_p{decision.new_phase}_"
                f"{int(self._clock.monotonic() * 1000)}"
            )
            try:
                self._exec.place_passive_order_manual_only(
                    decision.close_side,
                    decision.target_price,
                    new_size,
                    cycle,
                    reduce_only=True,
                )
                # Phase 4D.5 (v1.4.191) — stamp the post-only place.
                self._state.sf_last_action_mono = self._clock.monotonic()
            except Exception:
                logger.exception(
                    "soft_flatten_phase_place_post_only_failed "
                    "phase=%s side=%s price=%s size=%s",
                    decision.new_phase,
                    decision.close_side.value,
                    decision.target_price,
                    new_size,
                )
            return

        # ---- ioc branch: cross-tick reduce-only IOC ------------------
        if decision.order_type == "ioc":
            # IOC takes liquidity. Any resting post-only from the
            # previous phase needs to clear before we cross — both
            # because the venue may reject overlapping reduce-only
            # intents AND because a left-over post-only could fill
            # alongside the IOC and over-flatten across the zero line.
            # (Reduce-only protects us, but cancelling first keeps the
            # close trajectory deterministic.)
            any_active = (
                (wo_bid is not None and wo_bid.status in active_statuses)
                or (wo_ask is not None and wo_ask.status in active_statuses)
            )
            if any_active:
                try:
                    self._exec.cancel_all_orders_for_symbol()
                except Exception:
                    logger.exception(
                        "soft_flatten_phase_ioc_pre_cancel_failed"
                    )
                # Wait one tick for the cancel to propagate before
                # firing the IOC. The phase dwell budget keeps ticking;
                # if cancels take so long that the dwell expires, the
                # ladder escalates further on its own.
                return

            place_ioc = getattr(self._client, "place_ioc_reduce_only", None)
            if not callable(place_ioc):
                # Adapter doesn't expose an IOC path. The phase ladder
                # is an OKX-shaped feature today; non-OKX adapters
                # should either implement the method (Hyperliquid
                # already has it) or leave the ladder disabled per
                # profile. Falling back to market_close is the safe
                # exit (avoids stalling in phase 2/3 forever).
                logger.warning(
                    "soft_flatten_phase_ioc_unsupported_falling_back_market "
                    "client=%s phase=%s",
                    type(self._client).__name__,
                    decision.new_phase,
                )
                # v1.4.173: synthetic orders row for the fallback
                # market_close (same rationale as the terminal market
                # branch above).
                self._persist_synthetic_sf_order_row(
                    phase=decision.new_phase,
                    side=decision.close_side,
                    price=None,
                    size=abs(pos_qty),
                    post_only=False,
                    order_kind="market_close_ioc_fallback",
                )
                try:
                    self._client.market_close(self._settings.symbol)
                except Exception:
                    logger.exception(
                        "soft_flatten_phase_ioc_fallback_market_close_failed"
                    )
                self._exit_soft_flatten(
                    reason="phase_ladder_ioc_unsupported_fallback_market"
                )
                return

            new_size = max(0.0, target_size - outstanding_close_sz)
            if new_size <= 0:
                return  # already covered by an existing intent
            # v1.5.146 Phase 4D.3 — slice cap. When
            # ``SOFT_FLATTEN_SLICE_NOTIONAL_USD > 0`` cap THIS IOC's
            # size below the configured notional. The remaining
            # quantity stays on the position; the next quote-loop
            # tick re-runs the dispatcher, which:
            #   * Reads a fresh touch (market may have moved
            #     favourably during this slice).
            #   * Re-evaluates phase escalation (the dwell timer
            #     keeps ticking — slicing doesn't reset it).
            #   * Fires the next IOC slice at the new target price.
            # That's the cross-venue spread saving the plan
            # describes: while phase-3 trades the first slice,
            # Binance may pull back and the next slice can land at
            # +0 ticks instead of +1.
            #
            # Counter ``sf_slice_dispatched_total`` only bumps when
            # the cap ACTUALLY trims new_size — operator can see at
            # postmortem time whether slicing was effective vs. the
            # full-flatten path that ran before this knob existed.
            _slice_notional = float(
                getattr(
                    self._settings,
                    "soft_flatten_slice_notional_usd",
                    0.0,
                )
            )
            if (
                _slice_notional > 0.0
                and decision.target_price > 0
            ):
                _slice_size = _slice_notional / float(
                    decision.target_price
                )
                if _slice_size < new_size:
                    new_size = _slice_size
                    try:
                        self._state.sf_slice_dispatched_total += 1
                    except Exception:  # noqa: BLE001 -- defensive
                        pass
            # v1.4.173 (Phase 4D.4): stub orders row BEFORE the IOC
            # fires so the ORDERS history shows the placement even if
            # the IOC fully fills + the fill ingestion runs before the
            # row write. (Race-tolerant: the fill-ingest path doesn't
            # look up by ``order_id_exchange`` for synthetic rows; it
            # uses the SF-tagging fallback path from v1.4.163.)
            self._persist_synthetic_sf_order_row(
                phase=decision.new_phase,
                side=decision.close_side,
                price=decision.target_price,
                size=new_size,
                post_only=False,
                order_kind="ioc",
            )
            try:
                place_ioc(
                    self._settings.symbol,
                    decision.close_side == Side.BUY,
                    new_size,
                    decision.target_price,
                )
                # Phase 4D.5 (v1.4.191) — record IOC fire as a throttle
                # action so the next dispatcher tick respects the gap.
                self._state.sf_last_action_mono = self._clock.monotonic()
            except Exception:
                logger.exception(
                    "soft_flatten_phase_ioc_place_failed "
                    "phase=%s side=%s price=%s size=%s",
                    decision.new_phase,
                    decision.close_side.value,
                    decision.target_price,
                    new_size,
                )
            return

        # Unknown order_type — defensive log only (the dataclass +
        # helper restrict this to {"post_only", "ioc", "market"}, but a
        # future helper edit could add a fourth and forget to wire it
        # here; the WARNING surfaces that gap).
        logger.warning(
            "soft_flatten_phase_unknown_order_type %s phase=%s",
            decision.order_type,
            decision.new_phase,
        )

    def _maybe_periodic_bot_heartbeat(self) -> None:
        """At most once per 15s: INFO summary while the bot loop runs."""
        now_m = self._clock.monotonic()
        if self._obs_heartbeat_last_monotonic is None:
            self._obs_heartbeat_last_monotonic = now_m
            should_emit = True
        elif now_m - self._obs_heartbeat_last_monotonic >= 15.0:
            self._obs_heartbeat_last_monotonic = now_m
            should_emit = True
        else:
            should_emit = False
        if not should_emit:
            return
        flags = self._state.status_flags_dict()
        with self._state._lock:
            last_mid = self._state.market.mid_price if self._state.market else None
            pos_q = self._state.position.position_qty
            market_success_ts = self._state.market_data_last_success_wall_ts
            market_failed_streak = self._state.market_data_failed_refresh_streak
            market_unchanged_streak = self._state.market_data_unchanged_snapshot_streak
        exec_obs = self._exec.get_reconcile_runtime_counters()
        rest_obs_fn = getattr(self._client, "rest_runtime_counters", None)
        rest_obs: dict[str, Any] = {}
        if callable(rest_obs_fn):
            try:
                candidate = rest_obs_fn()
                if isinstance(candidate, dict):
                    rest_obs = candidate
            except Exception:
                logger.debug("rest_runtime_counters unavailable", exc_info=True)
        # v1.4.20 rate-limit-observability Phase 2: stash the per-pool
        # rate snapshot on BotState so ``/state/current`` (and the
        # Connectivity dashboard panel that consumes it) shows live
        # per-pool pressure. ``okx_rate_window_per_pool`` is published
        # by the OKX adapter; for non-OKX adapters it's absent and the
        # field stays empty.
        per_pool = rest_obs.get("okx_rate_window_per_pool")
        if isinstance(per_pool, dict):
            self._state.okx_rate_window_per_pool = per_pool
        log_extra(
            logger,
            logging.INFO,
            "bot heartbeat",
            {
                "event": "bot_heartbeat",
                "bot_status": flags["bot_status"],
                "symbol": self._settings.symbol,
                "trading_enabled": self._settings.trading_enabled,
                "market_data_available": flags["market_data_available"],
                "account_data_available": flags["account_data_available"],
                "last_mid_price": last_mid,
                "position_qty": pos_q,
                "desync_phase": flags["desync_phase"],
                "seconds_since_last_market_data_refresh": (
                    seconds_since(market_success_ts) if market_success_ts else None
                ),
                "market_data_failed_refresh_streak": market_failed_streak,
                "market_data_unchanged_snapshot_streak": market_unchanged_streak,
                "account_refresh_suppressed_order_uncertainty_count": (
                    self._account_refresh_suppressed_order_uncertainty_count
                ),
                "order_state_uncertainty_active": self._exec.has_order_state_uncertainty(),
                **exec_obs,
                **rest_obs,
            },
        )

    def _should_refresh_account_rest(self, addr: str) -> tuple[bool, str]:
        """Full REST account/position refresh — skipped on hot ticks when WS + snapshot are healthy.

        STARTING note: previously this function unconditionally returned
        True when ``bot_status == STARTING``, bypassing the throttle.
        That was the proximate cause of bug-018: after the first 429,
        the throttle still bypassed every tick, producing a 50-Hz retry
        loop against /account/positions until ``MAX_EXECUTION_ERRORS``
        kicked in. Removed 2026-05-05. The ``last_m is None`` path
        below still lets the FIRST refresh through (no throttle history
        yet); subsequent attempts respect the configured min interval.
        """
        if not (addr or "").strip():
            return True, "missing_addr"
        order_uncertain = self._exec.has_order_state_uncertainty()
        with self._state._lock:
            snapshot_unhealthy = self._state.exchange_snapshot_unhealthy_streak > 0
            private_recovery = self._state.private_ws_recovery_pending
            last_m = self._state.account_rest_last_monotonic
        min_iv = float(self._settings.account_rest_min_interval_seconds)
        if order_uncertain:
            min_iv = max(
                min_iv,
                float(self._settings.order_state_uncertainty_account_rest_interval_seconds),
            )
        if (
            self._exec.should_ingest_fills_via_rest()
            or snapshot_unhealthy
            or private_recovery
            or not self._exec.private_ws_healthy
        ):
            min_iv = max(
                min_iv,
                float(self._settings.unhealthy_account_rest_min_interval_seconds),
            )
        if last_m is None:
            return True, "first_refresh"
        elapsed = max(0.0, self._clock.monotonic() - last_m)
        if elapsed >= min_iv:
            return True, "min_interval_elapsed"
        if order_uncertain:
            self._account_refresh_suppressed_order_uncertainty_count += 1
            log_extra(
                logger,
                logging.DEBUG,
                "account_refresh_suppressed",
                {
                    "event": "account_refresh_suppressed",
                    "reason": "order_state_uncertainty",
                    "elapsed_s": elapsed,
                    "required_interval_s": min_iv,
                    "count": self._account_refresh_suppressed_order_uncertainty_count,
                },
            )
            return False, "suppressed_order_uncertainty"
        return False, "min_interval_not_elapsed"

    def _collect_ladder_gate_caps(self) -> dict[str, Optional[int]]:
        """Per-gate ``max_levels_per_side`` caps for the current cycle.

        1.4.13 Phase 3 cleanup: the regime-response gates
        (vol_trend, post_swing, basis_regime) NO LONGER publish
        ladder caps from this function. Post-cutover those gates
        respond via SpreadComposition widening — outer ladder rungs
        get pushed to MAX_HALF_SPREAD_BPS wide already, so an
        additional N-cap is redundant. The cap was a binary
        artifact of the pre-cutover suppression model; widening's
        continuous response replaces it.

        What REMAINS in this function:
        * session_drawdown ladder — safety, stays binary:
          PAUSE_* → cap=1, WIDEN/RESUME → cap=2
        * markout_size_scaler heavy tier — separate mechanism
          (size-driven, not regime-eligibility-driven). Stays.

        Returns an empty dict when no caps fire OR when
        ``ladder_gates_limit_levels`` is False (build_ladder also
        checks the flag — defensive null at both ends).
        """
        if not bool(self._settings.ladder_gates_limit_levels):
            return {}
        now_mono = self._clock.monotonic()
        caps: dict[str, Optional[int]] = {}

        # session_drawdown ladder — safety; stays binary.
        try:
            sd = self._state.session_drawdown
            tier = getattr(sd, "tier", None)
            tier_str = (
                getattr(tier, "value", None)
                or (str(tier) if tier is not None else "CLEAR")
            )
            if "PAUSE" in tier_str:
                caps["session_drawdown"] = 1
            elif "WIDEN" in tier_str or "RESUME" in tier_str:
                caps["session_drawdown"] = 2
        except Exception:
            pass

        # markout_size_scaler — separate from the regime-gate
        # widening chain (size-driven, not eligibility-driven).
        # Stays as a cap.
        try:
            median_5s = self._state.toxicity_recent_median_markout_5s_bps()
            heavy_thr = float(
                getattr(self._settings, "markout_size_scaler_heavy_threshold_bps", -3.0)
            )
            if median_5s is not None and float(median_5s) <= heavy_thr:
                caps["markout_size_scaler"] = 2
        except Exception:
            pass

        # Suppress unused-variable warning for the now-unreferenced
        # ``now_mono`` (kept in scope in case a future cap reads it).
        del now_mono

        return caps

    def _apply_structural_bias_throttle_gate(
        self,
        eff_q: QuoteEligibilityResult,
    ) -> QuoteEligibilityResult:
        """v1.4.228 — Phase 4G.13. Structural-bias auto-throttle.

        Reads session-cumulative ``inventory_exec_bias`` suppression
        counts and forces one-sided REDUCING eligibility when the
        ratio crosses the configured threshold. Breaks the
        "accumulate → bleed → SF → re-accumulate" cycle by preventing
        the bot from re-leaning into the same direction.

        Composes via ``more_restrictive`` with the existing
        eligibility chain — never widens, only narrows. When the gate
        is OFF / below min-samples / below threshold, returns eff_q
        unchanged.

        Direction logic mirrors the v1.4.220 Bias card:

          * BID-suppressed > ASK-suppressed → LONG bias → force
            ``QUOTE_SELL_ONLY`` (bot can only sell to reduce)
          * ASK-suppressed > BID-suppressed → SHORT bias → force
            ``QUOTE_BUY_ONLY`` (bot can only buy to reduce)
          * Equal → balanced, no action

        Per-tick fire counter on ``BotState`` reflects cumulative
        time the bot was throttled this session.
        """
        from dataclasses import replace as _replace

        from app.quote_eligibility import more_restrictive

        # Default: clear the latched flag for this tick (set to True
        # below when the gate engages). Lets the snapshot reflect
        # last-tick truth instead of stale state.
        self._state.structural_bias_throttle_active_last_tick = False
        self._state.structural_bias_throttle_direction = None

        if not getattr(
            self._settings, "structural_bias_auto_throttle_enabled", False
        ):
            return eff_q

        # Pull suppression counts. Defensive against missing
        # quote_quality attribute (very-early-tick path).
        try:
            counts = self._state.quote_quality._suppression_counts
        except (AttributeError, KeyError):
            return eff_q

        # v1.5.197 — idle-decay session counters. If no fill has
        # arrived for IDLE_DECAY_SECONDS, halve the
        # inventory_exec_bias counters. Prevents the stale-counter
        # deadlock: the throttle reads cumulative session counts
        # which only grow (never decay). After one storm builds an
        # asymmetric ratio (e.g. 20 bid-suppressions vs 0 ask-
        # suppressions in an early-session uptrend), the throttle
        # stays armed forever — including across regime changes —
        # because no mechanism reduces the bid count. Halving on
        # idle gives the counter a half-life equal to
        # IDLE_DECAY_SECONDS, so historical storms eventually wash
        # out instead of locking the bot indefinitely.
        try:
            _idle_decay_s = float(
                getattr(
                    self._settings,
                    "inventory_exec_bias_idle_decay_seconds",
                    300.0,
                )
                or 0.0
            )
            _now_mono_decay = self._clock.monotonic()
            _last_fill_mono = self._state.last_fill_at_mono
            _last_decay_mono = float(
                getattr(
                    self._state,
                    "_structural_bias_last_decay_mono",
                    0.0,
                )
                or 0.0
            )
            if (
                _idle_decay_s > 0.0
                and _last_fill_mono is not None
                and (_now_mono_decay - _last_fill_mono) >= _idle_decay_s
                and (_now_mono_decay - _last_decay_mono) >= _idle_decay_s
            ):
                # Halve both counters and mark the decay timestamp so
                # we don't decay again until another IDLE_DECAY_SECONDS
                # has elapsed (i.e., one decay per idle window).
                cur_bid = int(counts.get("engine:inventory_exec_bias_bid", 0) or 0)
                cur_ask = int(counts.get("engine:inventory_exec_bias_ask", 0) or 0)
                counts["engine:inventory_exec_bias_bid"] = cur_bid // 2
                counts["engine:inventory_exec_bias_ask"] = cur_ask // 2
                self._state._structural_bias_last_decay_mono = _now_mono_decay
        except Exception:
            # Decay is best-effort; never break the trading path.
            pass

        bid = int(counts.get("engine:inventory_exec_bias_bid", 0) or 0)
        ask = int(counts.get("engine:inventory_exec_bias_ask", 0) or 0)
        total = bid + ask
        min_samples = int(
            self._settings.structural_bias_auto_throttle_min_samples
        )
        if total < min_samples:
            return eff_q

        threshold = float(
            self._settings.structural_bias_auto_throttle_ratio_threshold
        )
        # Ratio = max / max(1, min). The max(1, ...) guards against
        # divide-by-zero when one side is at 0.
        ratio = max(bid, ask) / max(1, min(bid, ask))
        if ratio < threshold:
            return eff_q

        # v1.5.151 BUG-029 fix — position-aware gate. The session-
        # cumulative ``bid > ask`` count means "engine SUPPRESSED bids
        # more often than asks" — which builds during ANY period
        # where adding-to-long was being correctly refused. In a
        # rising trend, the bot's defences correctly suppress BID
        # whenever it has any long inventory (don't add at higher
        # price). Those suppressions accumulate even when the bot's
        # CURRENT position is 0 or has flipped negative. The bias
        # ratio is historical; firing the throttle anyway forces
        # ``QUOTE_SELL_ONLY`` against a flat/short bot — which is
        # the exact mechanism the v1.5.150-260525-213511 snapshot
        # showed (4 SF events all on -3 short, all triggered by the
        # throttle's wrong-direction force during an uptrend).
        #
        # The throttle is INTENDED as a reducer-side enforcement:
        # "bot is long-trapped, force SELL to unwind." Reading the
        # current position direction makes the intent explicit.
        # Dormant when position is FLAT (no inventory to reduce) or
        # in the OPPOSITE direction of the bias (the lean has
        # already reversed; the cumulative ratio is stale).
        position_qty = float(
            getattr(self._state.position, "position_qty", 0.0) or 0.0
        )
        _POSITION_DEADZONE = 1e-9  # treat tiny residuals as flat
        if bid > ask:
            # LONG bias detected. Fire only if bot is actually long.
            if position_qty <= _POSITION_DEADZONE:
                return eff_q
            cap = QuoteEligibility.QUOTE_SELL_ONLY
            direction = "LONG"
        elif ask > bid:
            # SHORT bias detected. Fire only if bot is actually short.
            if position_qty >= -_POSITION_DEADZONE:
                return eff_q
            cap = QuoteEligibility.QUOTE_BUY_ONLY
            direction = "SHORT"
        else:
            # Equal counts but above threshold — shouldn't happen
            # since ratio = 1.0 here. Defensive no-op.
            return eff_q

        # Compose with existing eligibility — intersect, never widen.
        new_eligibility = more_restrictive(eff_q.eligibility, cap)
        # v1.5.155 publisher clarity: distinguish "evaluated and the
        # ratio-threshold + position guards both passed" (fire_count,
        # legacy semantic, was misleadingly large in the
        # v1.5.154-260526-074029 snapshot) from "actually narrowed
        # eligibility this tick" (changed_eligibility_count — the
        # number that actually matters for "did this gate suppress a
        # quote?"). When eff_q.eligibility was already at-or-below
        # ``cap`` (e.g. some upstream gate already returned HOLD_ALL),
        # more_restrictive returns the input unchanged → throttle did
        # not effectively suppress anything.
        self._state.structural_bias_throttle_fire_count_total += 1
        if new_eligibility != eff_q.eligibility:
            self._state.structural_bias_throttle_changed_eligibility_total += 1
        self._state.structural_bias_throttle_active_last_tick = True
        self._state.structural_bias_throttle_direction = direction
        self._state.structural_bias_throttle_last_ratio = ratio
        self._state.structural_bias_throttle_last_bid_count = bid
        self._state.structural_bias_throttle_last_ask_count = ask

        # Reason string mirrors the v1.4.220 Bias card's format so
        # the operator can correlate "Bias card showed LONG 5.2×"
        # with "structural_bias_throttle:LONG_5.2x" in the eligibility
        # reason.
        new_reason = f"{eff_q.reason}|structural_bias_throttle:{direction}_{ratio:.1f}x"
        return _replace(
            eff_q,
            eligibility=new_eligibility,
            reason=new_reason,
        )

    def _compute_vol_adaptive_cap(self) -> Optional[float]:
        """v1.5.158 Option C — vol-adaptive MAX_ABS_POSITION
        override.

        When ``VOL_ADAPTIVE_POSITION_CAP_ENABLED=true`` AND current
        ``vol_bps >= threshold``, return
        ``MAX_ABS_POSITION * reduction_factor`` (floored at 1.0 — a
        cap < 1 would mean "can't trade at all", which would be a
        Rule 0c violation: gate must stay adaptive, not effectively
        block trading).

        Returns ``None`` when disabled / vol below threshold (caller
        falls back to ``settings.max_abs_position``).

        Pure read; safe to call every tick.
        """
        if not getattr(
            self._settings, "vol_adaptive_position_cap_enabled", False
        ):
            return None
        vol_bps = float(getattr(self._state, "vol_bps", 0.0) or 0.0)
        thr = float(getattr(
            self._settings,
            "vol_adaptive_position_cap_threshold_bps_per_s",
            10.0,
        ))
        if vol_bps < thr:
            return None
        factor = float(getattr(
            self._settings,
            "vol_adaptive_position_cap_reduction_factor",
            0.5,
        ))
        full_cap = float(self._settings.max_abs_position)
        reduced = max(1.0, full_cap * factor)
        return reduced

    def _compute_vol_climbing_ratio(self) -> Optional[float]:
        """v1.5.158 Option A — short-MA vs long-MA ratio over the
        vol_bps_history deque.

        Walks the deque once and computes both window means by
        timestamp distance from ``now_mono``. Returns ``None`` when
        either window has fewer than ``min_samples`` points
        (warmup), or when long_MA is degenerate (zero / non-finite).

        Pure read; safe to call every tick. Cost: O(n) over the
        deque each call; at maxlen=7200 and 2 Hz cadence, ~7200
        comparisons / 500 ms = trivially fast.
        """
        if not getattr(self._settings, "vol_climbing_widen_enabled", False):
            return None
        deq = getattr(self._state, "vol_bps_history", None)
        if not deq or len(deq) == 0:
            return None
        short_s = float(getattr(
            self._settings, "vol_climbing_widen_short_window_seconds", 300.0
        ))
        long_s = float(getattr(
            self._settings, "vol_climbing_widen_long_window_seconds", 1800.0
        ))
        min_samples = int(getattr(
            self._settings, "vol_climbing_widen_min_samples", 20
        ))
        if short_s <= 0.0 or long_s <= 0.0 or short_s >= long_s:
            return None
        now_mono = self._clock.monotonic()
        short_sum = 0.0
        short_n = 0
        long_sum = 0.0
        long_n = 0
        # Single pass — newest entries are the most relevant for
        # both windows; deque is appended newest-last so iterate
        # backwards.
        for ts, v in reversed(deq):
            age = now_mono - float(ts)
            if age <= long_s:
                long_sum += float(v)
                long_n += 1
                if age <= short_s:
                    short_sum += float(v)
                    short_n += 1
            else:
                break  # all older samples are beyond long window
        if short_n < min_samples or long_n < min_samples:
            return None
        long_ma = long_sum / long_n
        if long_ma <= 0.0 or not math.isfinite(long_ma):
            return None
        short_ma = short_sum / short_n
        return short_ma / long_ma

    def _compute_seconds_since_last_fill(self) -> Optional[float]:
        """v1.5.156 Option B — seconds since the most recent fill
        on either side, or ``None`` if no fill has happened yet
        this session.

        Returns ``None`` instead of a huge number for the no-fill-
        yet case so the downstream no-fill-compression mechanism
        doesn't fire before a baseline fill rate has been
        established (otherwise a brand-new bot would immediately
        compress its quotes before learning what the natural touch
        looks like).

        Reads ``state.last_fill_monotonic_ms_{buy,sell}`` which the
        fill-ingestion path bumps on each finalised private fill.
        Both are 0.0 at session start; treats that as "no fill yet".
        """
        last_buy_ms = float(
            getattr(self._state, "last_fill_monotonic_ms_buy", 0.0) or 0.0
        )
        last_sell_ms = float(
            getattr(self._state, "last_fill_monotonic_ms_sell", 0.0) or 0.0
        )
        last_ms = max(last_buy_ms, last_sell_ms)
        if last_ms <= 0.0:
            return None
        now_mono = self._clock.monotonic()
        elapsed_s = now_mono - (last_ms / 1000.0)
        if elapsed_s < 0.0:
            # Clock skew or pre-startup race — treat as "no fill yet".
            return None
        return elapsed_s

    def _select_trend_drift_signal(self, eff_q) -> Optional[float]:
        """v1.5.155 — select the drift signal that feeds
        ``trend_drift_reservation_alpha`` in ``compute_quote_decision``.

        Pre-v1.5.155 the constructive trend skew read
        ``eff_q.mid_return_250ms_bps`` exclusively. In any sustained
        trend (e.g. 10 bps/30 s) the 250 ms drift is ~0.08 bps, so
        ``alpha * drift`` ≈ 0.024 bps midpoint shift — effectively
        zero. The bot quoted symmetrically around mid in trends and
        got adversely selected on both sides. See
        ``snapshots/v1.5.154-260526-074029`` for the canonical
        failure mode.

        Window selector
        ---------------
        ``TREND_DRIFT_SIGNAL_WINDOW_SECONDS`` maps to:

        * ``<= 0.3`` → ``eff_q.mid_return_250ms_bps`` (legacy)
        * ``<= 0.6`` → ``eff_q.mid_return_500ms_bps`` (falls back to 250 ms)
        * ``<= 7.5`` → ``state.mid_drift_windows.drift_5s_bps``
        * ``<= 20.0`` → ``state.mid_drift_windows.drift_10s_bps``
        * ``<= 45.0`` → ``state.mid_drift_windows.drift_30s_bps``
        * ``> 45.0`` → ``state.mid_drift_windows.drift_60s_bps``

        Fallback chain when the chosen window is ``None`` (deque
        warmup): chosen → 5 s → 500 ms → 250 ms → ``None``. Keeps
        the constructive skew partially-alive during the first ~10 s
        of bot lifetime instead of going dark.

        Pure read — no side effects. Safe to call every tick.
        """
        window_s = float(
            getattr(self._settings, "trend_drift_signal_window_seconds", 0.25)
        )

        # Fast eligibility-snapshot signals (always available once
        # quote eligibility has been computed for this tick).
        fast_500ms = getattr(eff_q, "mid_return_500ms_bps", None)
        fast_250ms = getattr(eff_q, "mid_return_250ms_bps", None)

        # Legacy paths.
        if window_s <= 0.3:
            return fast_250ms
        if window_s <= 0.6:
            return fast_500ms if fast_500ms is not None else fast_250ms

        # Long-window path: read from state.mid_drift_windows.
        mdw = getattr(self._state, "mid_drift_windows", None)
        chosen: Optional[float] = None
        if mdw is not None:
            if window_s <= 7.5:
                chosen = mdw.drift_5s_bps
            elif window_s <= 20.0:
                chosen = mdw.drift_10s_bps
            elif window_s <= 45.0:
                chosen = mdw.drift_30s_bps
            else:
                chosen = mdw.drift_60s_bps

        if chosen is not None:
            return float(chosen)

        # Fallback chain: chosen window not warmed up yet.
        if mdw is not None and mdw.drift_5s_bps is not None:
            return float(mdw.drift_5s_bps)
        if fast_500ms is not None:
            return float(fast_500ms)
        if fast_250ms is not None:
            return float(fast_250ms)
        return None

    def _apply_post_sf_cooldown_gate(
        self,
        eff_q: QuoteEligibilityResult,
    ) -> QuoteEligibilityResult:
        """v1.5.2 Phase 4D.3 — post-SF cooldown gate.

        For ``POST_SF_COOLDOWN_SECONDS`` after a soft-flatten
        completes, **pause quoting entirely** (HOLD_ALL).

        v1.5.154 design rewrite (driving snapshot
        ``v1.5.150-260525-220850``):

        The original v1.5.2 design forced ``QUOTE_X_ONLY`` in the
        side OPPOSITE to the pre-SF position direction (e.g. pre-SF
        LONG → suppress BUY → force SELL_ONLY). The intent was
        "don't re-add to the bad direction". In a sustained trending
        regime this turned into "force the bot to quote the side
        that's about to be adversely selected by the trend": after
        a long-closing SF in an uptrend, the 5-min SELL_ONLY window
        guaranteed the bot kept getting picked off on SELL until
        the cooldown cleared OR another SF re-armed it in the
        opposite direction. The v1.5.150-260525-220850 snapshot
        showed 8 -3 SHORT transitions vs 2 +3 LONG over 23 min in
        an uptrend session, with PnL bleeding on each forced-side
        cycle.

        The v1.5.154 fix: cooldown forces ``HOLD_ALL`` (no quoting
        either side) instead of one-sided. The "don't re-add"
        intent is preserved (BUY is suppressed during the window if
        pre-SF was LONG), but the SELL side is ALSO suppressed,
        removing the forced-direction adverse-selection mechanism.
        Bot pauses, market does what it does, bot resumes
        two-sided quoting when the timer expires.

        Trade-off vs the original side-suppression: the bot misses
        any natural REDUCING-side fills that would have favourably
        unwound a still-non-zero post-SF residual position. In
        practice SF completes with position=0 (that's its job), so
        there's no residual to unwind during the window. When SF
        ends with a residual (e.g. partial close due to timeout),
        the bot waits for the cooldown then resumes normal quoting
        — the residual gets closed naturally by the BUY/SELL gate
        the next time the engine produces a reducing-side intent.

        State fields ``post_sf_cooldown_lean_side`` and
        ``post_sf_cooldown_until_mono`` are still populated by the
        arming code (``_exit_soft_flatten``) for dashboard /
        postmortem visibility ("the cooldown was armed after a
        LONG-closing SF"), but only the deadline drives the
        HOLD_ALL decision now.

        Composes via ``more_restrictive`` — never widens, only
        narrows. Self-clearing on a time deadline; no manual reset
        path (restart the bot to clear early).
        """
        from dataclasses import replace as _replace

        from app.quote_eligibility import more_restrictive

        # Default: clear the latched flag for this tick.
        self._state.post_sf_cooldown_active_last_tick = False

        if not getattr(self._settings, "post_sf_cooldown_enabled", True):
            return eff_q

        deadline = float(self._state.post_sf_cooldown_until_mono or 0.0)
        if deadline <= 0.0:
            return eff_q

        now_mono = self._clock.monotonic()
        if now_mono >= deadline:
            # Cooldown expired — clear the state so subsequent ticks
            # short-circuit at the deadline check.
            self._state.post_sf_cooldown_until_mono = 0.0
            self._state.post_sf_cooldown_lean_side = None
            return eff_q

        # v1.5.154 — HOLD_ALL pause regardless of pre-SF direction.
        # The lean_side field is preserved for dashboard / postmortem
        # ("which direction did we just SF from?") but does NOT drive
        # the eligibility decision anymore.
        lean_side = self._state.post_sf_cooldown_lean_side
        if lean_side == Side.BUY:
            direction = "LONG"
        elif lean_side == Side.SELL:
            direction = "SHORT"
        else:
            # Edge: deadline set but lean_side None (e.g. pre-SF=0).
            # Per the arming logic in ``_exit_soft_flatten``, this
            # only happens if a future code path arms the timer
            # without setting the direction. Honour the HOLD_ALL.
            direction = "UNKNOWN"

        new_eligibility = more_restrictive(
            eff_q.eligibility, QuoteEligibility.HOLD_ALL
        )
        self._state.post_sf_cooldown_fire_count_total += 1
        self._state.post_sf_cooldown_active_last_tick = True

        remaining_s = max(0.0, deadline - now_mono)
        new_reason = (
            f"{eff_q.reason}|post_sf_cooldown:HOLD_ALL"
            f"_prev={direction}_{remaining_s:.0f}s"
        )
        return _replace(
            eff_q,
            eligibility=new_eligibility,
            reason=new_reason,
        )

    def _apply_session_drawdown_gate(
        self,
        eff_q: QuoteEligibilityResult,
        *,
        now_mono: float,
        now_iso: str,
    ) -> tuple[QuoteEligibilityResult, float]:
        """Tiered session-PnL drawdown ladder. Returns
        ``(eff_q, extra_spread_floor_overlay_bps)`` — the second
        value is added to the bot's existing ``spread_floor_overlay``
        chain so tier-1 widen stacks with toxicity / QQ overlays.

        On tier-4 (KILLED), this method calls ``self.kill(...)``
        and returns immediately; the caller should check the bot
        status afterwards. On PAUSE_*, eff_q is forced to
        HOLD_ALL with an appended reason. On WIDEN /
        RESUME_TESTING, the spread floor adds the configured
        widen overlay.
        """
        from dataclasses import replace as _replace

        from app import session_drawdown_gate as sdg

        if not getattr(self._settings, "session_drawdown_enabled", True):
            return eff_q, 0.0

        thresholds = sdg.TierThresholds(
            tier1_widen_usd=float(self._settings.session_drawdown_tier1_widen_usd),
            tier2_pause_short_usd=float(self._settings.session_drawdown_tier2_pause_short_usd),
            tier3_pause_long_usd=float(self._settings.session_drawdown_tier3_pause_long_usd),
            tier4_kill_usd=float(self._settings.session_drawdown_tier4_kill_usd),
            pause_short_seconds=float(self._settings.session_drawdown_pause_short_seconds),
            pause_long_seconds=float(self._settings.session_drawdown_pause_long_seconds),
            test_resume_sample_fills=int(self._settings.session_drawdown_test_resume_sample_fills),
            test_resume_max_adverse_bps=float(self._settings.session_drawdown_test_resume_max_adverse_bps),
            favorable_exit_enabled=bool(
                self._settings.session_drawdown_favorable_exit_enabled
            ),
            favorable_exit_clear_band_ratio=float(
                self._settings.session_drawdown_favorable_exit_clear_band_ratio
            ),
            favorable_exit_dwell_seconds=float(
                self._settings.session_drawdown_favorable_exit_dwell_seconds
            ),
        )

        try:
            session_pnl = float(self._state.pnl.total_pnl_usd or 0.0)
        except (TypeError, ValueError, AttributeError):
            session_pnl = 0.0

        st = self._state.session_drawdown
        sd_fires_before = int(getattr(st, "fire_count", 0) or 0)
        sd_tier_before = getattr(st, "tier", None)

        # Pause-expiry → enter RESUME_TESTING (must check before observe_pnl
        # so a pnl tick doesn't immediately re-trip on the same number).
        sdg.observe_pause_expiry(
            st, now_mono=now_mono, now_iso=now_iso, thresholds=thresholds
        )
        # Then evaluate ladder against fresh pnl.
        sdg.observe_pnl(
            st,
            now_mono=now_mono,
            now_iso=now_iso,
            session_pnl_usd=session_pnl,
            thresholds=thresholds,
        )
        # Emit a bot_event when the tier transitions so the
        # events_since timeline carries every escalation /
        # de-escalation. Compare fire_count + tier before vs after
        # the two observe() calls.
        sd_fires_after = int(getattr(st, "fire_count", 0) or 0)
        sd_tier_after = getattr(st, "tier", None)
        if sd_fires_after > sd_fires_before and sd_tier_after != sd_tier_before:
            tier_str = (
                getattr(sd_tier_after, "value", None)
                or (str(sd_tier_after) if sd_tier_after is not None else "?")
            )
            prev_str = (
                getattr(sd_tier_before, "value", None)
                or (str(sd_tier_before) if sd_tier_before is not None else "?")
            )
            self._log_event(
                EventSeverity.WARNING,
                "session_drawdown_transition",
                (
                    f"session-drawdown ladder {prev_str} → {tier_str} at "
                    f"pnl=${st.last_trigger_pnl_usd:+.3f}"
                ),
                {
                    "prev_tier": prev_str,
                    "new_tier": tier_str,
                    "session_pnl_usd": st.last_trigger_pnl_usd,
                    "fire_count": int(st.fire_count),
                },
            )

        # Tier-4 kill: terminal. Mirrors MAX_DRAWDOWN_USD semantics.
        if sdg.is_killed(st):
            try:
                self.kill(
                    "session_drawdown_tier4_kill",
                    self._build_kill_payload(["session_drawdown_tier4_kill"]),
                )
            except Exception:
                _state_logger = logging.getLogger("session_drawdown")
                _state_logger.exception("session_drawdown_kill_failed")
            return eff_q, 0.0

        # PAUSE_* → force HOLD_ALL with reason.
        if sdg.is_quote_paused(st):
            remaining = max(0.0, st.cooldown_until_mono - now_mono)
            eff_q = _replace(
                eff_q,
                eligibility=QuoteEligibility.HOLD_ALL,
                reason=(
                    f"{eff_q.reason}|session_drawdown:{st.tier.value}"
                    f"|pnl=${st.last_trigger_pnl_usd:+.3f}"
                    f"|remaining={remaining:.0f}s"
                ),
            )
            return eff_q, 0.0

        # WIDEN / RESUME_TESTING → contribute to spread floor.
        if sdg.is_widen_active(st):
            cfg_overlay = float(
                getattr(
                    self._settings,
                    "adaptive_spread_adverse_overlay_half_spread_bps",
                    0.0,
                )
                or 0.0
            )
            return eff_q, max(0.0, cfg_overlay)

        return eff_q, 0.0

    def _apply_sf_fatigue_gate(
        self,
        eff_q: QuoteEligibilityResult,
        *,
        now_mono: float,
    ) -> tuple[QuoteEligibilityResult, float]:
        """v1.5.202 — SF-event-count fatigue ladder. Orthogonal to
        ``_apply_session_drawdown_gate`` (which keys on PnL): this
        keys on the COUNT of SF episode entries in a rolling window.

        Returns ``(eff_q, widen_multiplier)`` — the second value is
        the multiplicative half-spread widening to apply on tier-1
        WIDEN (1.0 = no-op; >1.0 = widen). The caller is expected to
        compose this multiplicatively with the existing spread floor.

        On tier-4 (KILLED), calls ``self.kill(...)``. On PAUSE_*,
        forces HOLD_ALL with an explicit reason. Both entry AND exit
        are driven by the rolling window count (events age out
        automatically) plus the pause-budget cooldown — adaptive per
        CLAUDE.md Rule 0c.
        """
        from dataclasses import replace as _replace

        from app import sf_fatigue_gate as sfg

        if not getattr(self._settings, "sf_fatigue_gate_enabled", True):
            return eff_q, 1.0

        st = self._state.sf_fatigue
        prev_tier = st.current_tier
        prev_fires = int(getattr(st, "fire_count", 0) or 0)

        window_seconds = float(
            getattr(self._settings, "sf_fatigue_window_seconds", 1800.0) or 0.0
        )
        tier1 = int(getattr(self._settings, "sf_fatigue_tier1_widen_count", 3) or 0)
        tier2 = int(
            getattr(self._settings, "sf_fatigue_tier2_pause_short_count", 5) or 0
        )
        tier3 = int(
            getattr(self._settings, "sf_fatigue_tier3_pause_long_count", 8) or 0
        )
        tier4 = int(
            getattr(self._settings, "sf_fatigue_tier4_kill_count", 12) or 0
        )
        pause_short = float(
            getattr(self._settings, "sf_fatigue_pause_short_seconds", 600.0) or 0.0
        )
        pause_long = float(
            getattr(self._settings, "sf_fatigue_pause_long_seconds", 1800.0) or 0.0
        )
        widen_mult = float(
            getattr(self._settings, "sf_fatigue_widen_multiplier", 1.3) or 1.0
        )

        new_tier = sfg.evaluate_tier(
            st,
            now_mono=now_mono,
            window_seconds=window_seconds,
            tier1_widen_count=tier1,
            tier2_pause_short_count=tier2,
            tier3_pause_long_count=tier3,
            tier4_kill_count=tier4,
            pause_short_seconds=pause_short,
            pause_long_seconds=pause_long,
        )

        # Emit a transition event when fire_count bumped (rising
        # edge into a stricter tier). Mirrors session_drawdown's
        # transition-event pattern.
        new_fires = int(getattr(st, "fire_count", 0) or 0)
        if new_fires > prev_fires and new_tier != prev_tier:
            self._log_event(
                EventSeverity.WARNING,
                "sf_fatigue_transition",
                (
                    f"sf-fatigue ladder {prev_tier} → {new_tier} at "
                    f"events_in_window={sfg.event_count_in_window(st)}"
                ),
                {
                    "prev_tier": prev_tier,
                    "new_tier": new_tier,
                    "events_in_window": sfg.event_count_in_window(st),
                    "fire_count": new_fires,
                    "window_seconds": window_seconds,
                },
            )

        # Tier-4 → kill. Terminal; mirrors MAX_DRAWDOWN_USD / session
        # tier-4 kill semantics.
        if new_tier == sfg.TIER_KILLED:
            try:
                self.kill(
                    "sf_fatigue_tier4_kill",
                    self._build_kill_payload(["sf_fatigue_tier4_kill"]),
                )
            except Exception:
                logging.getLogger("sf_fatigue").exception(
                    "sf_fatigue_kill_failed"
                )
            return eff_q, 1.0

        # PAUSE_* → HOLD_ALL with reason.
        if new_tier in (sfg.TIER_PAUSE_SHORT, sfg.TIER_PAUSE_LONG):
            remaining = sfg.seconds_remaining_on_pause(
                st,
                now_mono=now_mono,
                pause_short_seconds=pause_short,
                pause_long_seconds=pause_long,
            )
            eff_q = _replace(
                eff_q,
                eligibility=QuoteEligibility.HOLD_ALL,
                reason=(
                    f"{eff_q.reason}|sf_fatigue:{new_tier}"
                    f"|events={sfg.event_count_in_window(st)}"
                    f"|remaining={remaining:.0f}s"
                ),
            )
            return eff_q, 1.0

        # WIDEN → return the configured multiplier; caller composes.
        if new_tier == sfg.TIER_WIDEN:
            return eff_q, max(1.0, widen_mult)

        # CLEAR — no-op.
        return eff_q, 1.0

    def _apply_regime_gates(
        self,
        eff_q: QuoteEligibilityResult,
        *,
        now_mono: float,
        raw_q: QuoteEligibilityResult,
    ) -> QuoteEligibilityResult:
        """Three new defensive regime gates (1.2.2):

        * vol_trend (2a): vol × drift conjunction with persistence
          → HOLD_ALL during cooldown.
        * basis_regime IC (2b): when |IC| < threshold, the
          classifier sees a signal-absent regime → HOLD_ALL.
        * microprice / OB imbalance (1a): when book is visibly
          asymmetric, suppress the thin-side quoting (one-sided cap).

        All three narrow eligibility; none widen. Reasons are
        appended so the operator can see *which* gate fired.
        """
        from dataclasses import replace as _replace

        from app import basis_regime_gate, microprice_gate, vol_trend_gate
        from app.quote_eligibility import more_restrictive

        # ---- 2a — vol × trend conjunction --------------------------------
        if self._settings.vol_trend_gate_enabled:
            vol_ratio: Optional[float] = None
            try:
                tox = self._state.toxicity
                vol_ratio = float(getattr(tox, "vol_spike_ratio", 1.0) or 1.0)
            except Exception:
                vol_ratio = 1.0
            drift_bps: Optional[float] = (
                raw_q.mid_return_500ms_bps
                if raw_q.mid_return_500ms_bps is not None
                else raw_q.mid_return_250ms_bps
            )
            vt_state = self._state.vol_trend_gate
            vt_fires_before = int(getattr(vt_state, "fire_count", 0) or 0)
            vol_trend_gate.observe(
                vt_state,
                now_mono=now_mono,
                vol_ratio=vol_ratio,
                drift_bps=drift_bps,
                vol_multiplier=float(self._settings.vol_trend_gate_vol_multiplier),
                drift_threshold_bps=float(
                    self._settings.vol_trend_gate_drift_threshold_bps
                ),
                persistence_seconds=float(
                    self._settings.vol_trend_gate_persistence_seconds
                ),
                cooldown_seconds=float(
                    self._settings.vol_trend_gate_cooldown_seconds
                ),
                # v1.4.153 Phase 2K.3 — favorable-exit predicate. Pure
                # back-compat: if these settings don't exist on the
                # current Settings shape (older config snapshots
                # without the v1.4.153 fields), default to the same
                # values the gate uses internally.
                clear_band_mult=float(
                    getattr(
                        self._settings,
                        "vol_trend_gate_clear_band_mult",
                        0.7,
                    )
                ),
                favorable_exit_dwell_seconds=float(
                    getattr(
                        self._settings,
                        "vol_trend_gate_favorable_exit_dwell_seconds",
                        10.0,
                    )
                ),
            )
            if int(getattr(vt_state, "fire_count", 0) or 0) > vt_fires_before:
                # Gate just fired — emit event with the trigger context
                # so the events_since timeline carries gate firings
                # alongside the existing soft_flatten / desync entries.
                self._log_event(
                    EventSeverity.WARNING,
                    "vol_trend_gate_fired",
                    (
                        f"vol×trend gate fired — "
                        f"vol_ratio={vt_state.last_trigger_vol_ratio:.2f}, "
                        f"drift={vt_state.last_trigger_drift_bps:+.2f} bps; "
                        f"HOLD_ALL for "
                        f"{self._settings.vol_trend_gate_cooldown_seconds:.0f}s"
                    ),
                    {
                        "vol_ratio": vt_state.last_trigger_vol_ratio,
                        "drift_bps": vt_state.last_trigger_drift_bps,
                        "cooldown_seconds": float(
                            self._settings.vol_trend_gate_cooldown_seconds
                        ),
                        "fire_count": int(vt_state.fire_count),
                    },
                )
            vt_firing = vol_trend_gate.is_active(
                self._state.vol_trend_gate, now_mono
            )
            # 1.4.7 Phase 0 gate-attribution telemetry.
            self._state.record_gate_firing(
                "vol_trend_gate", firing_now=vt_firing, now_mono=now_mono
            )
            # 1.4.12 cutover: vol_trend no longer clamps eligibility.
            # Its widening contribution (from vol_trend_gate.widening_bps,
            # composed in build_spread_composition) drives the spread
            # response instead. We still annotate eff_q.reason so the
            # operator's diagnostics show the gate fired.
            if vt_firing:
                vt = self._state.vol_trend_gate
                remaining = vol_trend_gate.seconds_remaining(vt, now_mono)
                eff_q = _replace(
                    eff_q,
                    reason=(
                        f"{eff_q.reason}|vol_trend_gate"
                        f"|vol_ratio={vt.last_trigger_vol_ratio:.2f}"
                        f"|drift={vt.last_trigger_drift_bps:+.2f}bps"
                        f"|remaining={remaining:.0f}s"
                    ),
                )

        # ---- 2b — basis_regime IC gate -----------------------------------
        # Two response modes (``BASIS_REGIME_GATE_MODE``):
        #   * ``hold_all``   — eligibility clamp (legacy). Sits the
        #                      bot out entirely on signal-absent.
        #   * ``size_shrink``— keep quoting at reduced size; the
        #                      shrink composes into the size_mult
        #                      chain inside ``compute_quote_decision``
        #                      via ``self._state.basis_regime_size_mult``.
        # The default flipped to ``size_shrink`` in 1.2.8 after the
        # 2026-05-10 wedge made the cost of "sit out on weak signal"
        # visible. Reset state each tick — the gate is stateless.
        self._state.basis_regime_size_mult = 1.0
        self._state.basis_regime_size_mult_reason = None
        if self._settings.basis_regime_gate_enabled:
            try:
                last_ic = self._state.basis_regime.last_ic
                pair_count = int(self._state.basis_regime.pair_count)
            except Exception:
                last_ic = None
                pair_count = 0
            min_pair_samples = int(
                self._settings.basis_deviation_regime_min_pair_samples
            )
            mode = str(
                getattr(self._settings, "basis_regime_gate_mode", "hold_all")
            ).lower()
            if mode == "size_shrink":
                mult, br_reason = basis_regime_gate.compute_size_mult(
                    last_ic=last_ic,
                    pair_count=pair_count,
                    ic_min_quote_threshold=float(
                        self._settings.basis_regime_gate_ic_min_quote
                    ),
                    min_pair_samples=min_pair_samples,
                    size_mult_signal_absent=float(
                        self._settings.basis_regime_gate_size_mult_signal_absent
                    ),
                    enabled=True,
                )
                br_firing = br_reason is not None
                if br_firing:
                    self._state.basis_regime_size_mult = mult
                    self._state.basis_regime_size_mult_reason = br_reason
                    # Annotate eligibility reason so the operator can
                    # see the gate fired on the live dashboard, even
                    # though eligibility itself is not clamped.
                    eff_q = _replace(eff_q, reason=f"{eff_q.reason}|{br_reason}")
            else:
                br_reason = basis_regime_gate.evaluate_gate(
                    last_ic=last_ic,
                    pair_count=pair_count,
                    ic_min_quote_threshold=float(
                        self._settings.basis_regime_gate_ic_min_quote
                    ),
                    min_pair_samples=min_pair_samples,
                    enabled=True,
                )
                br_firing = br_reason is not None
                # 1.4.12 cutover: basis_regime no longer clamps
                # eligibility in hold_all mode either. Widening
                # contribution (basis_regime_gate.widening_bps) drives
                # the response.
                if br_firing:
                    eff_q = _replace(
                        eff_q,
                        reason=f"{eff_q.reason}|{br_reason}",
                    )
            # 1.4.7 Phase 0 telemetry.
            self._state.record_gate_firing(
                "basis_regime_gate",
                firing_now=br_firing,
                now_mono=now_mono,
            )

        # ---- 1a — microprice / OB-imbalance gate -------------------------
        if self._settings.microprice_gate_enabled:
            override, override_reason = microprice_gate.evaluate_gate(
                ob_imbalance_ewma=self._state.ob_imbalance_ewma,
                threshold=float(
                    self._settings.microprice_gate_ob_imbalance_threshold
                ),
                enabled=True,
            )
            mp_firing = override is not None
            # 1.4.7 Phase 0 telemetry.
            self._state.record_gate_firing(
                "microprice_gate",
                firing_now=mp_firing,
                now_mono=now_mono,
            )
            # 1.4.12 cutover: microprice no longer clamps eligibility.
            # The asymmetric widening contribution
            # (microprice_gate.widening_bps) widens the suppressed side
            # instead of cutting it off entirely.
            if mp_firing:
                eff_q = _replace(
                    eff_q,
                    reason=f"{eff_q.reason}|{override_reason}",
                )

        return eff_q

    def _apply_post_swing_and_momentum_gates(
        self,
        eff_q: QuoteEligibilityResult,
        *,
        now_mono: float,
        mid_now: Optional[float],
        raw_q: QuoteEligibilityResult,
    ) -> QuoteEligibilityResult:
        """Stack two analysis-day behavioural caps onto the existing
        effective eligibility:

        * #2 post-swing PnL cooldown: HOLD_ALL while the cooldown is
          active. Driven by sampling ``state.pnl.total_pnl_usd`` into
          a rolling buffer; trigger fires on |delta| over the window.
        * #3 momentum gate: refuses to add to the side that's
          *aligned* with recent mid-drift when |inventory| is above
          a utilisation threshold. Aligned long+uptrend → SELL_ONLY;
          aligned short+downtrend → BUY_ONLY.

        Both gates narrow eligibility — neither widens. The reason
        string is appended so the operator (and the eligibility log)
        can see *why* the cap tightened.
        """
        from dataclasses import replace as _replace

        from app import post_swing_gate
        from app.momentum_gate import evaluate_momentum_gate
        from app.quote_eligibility import more_restrictive

        # --- #0 v1.4.170 Phase 4F — elevated-vol auto-pause -------------
        # MACRO defense. Fires FIRST (priority highest): if the regime
        # itself is bad (vol_spike_ratio elevated for the arm dwell),
        # force HOLD_ALL — the bot sits out entirely until vol
        # normalises (clear dwell) or the MAX ceiling (default 30 min)
        # fires. Subsumes all per-event gates below — when active, none
        # of their widen / shrink / suppress responses matter because
        # the bot isn't quoting at all. Default DISABLED via
        # ``VOL_AUTO_PAUSE_ARM_RATIO=0.0``. See
        # ``app/vol_regime_auto_pause.py`` for the pure-function
        # evaluator that runs in the toxicity-refresh path each tick.
        if bool(
            getattr(self._state, "vol_auto_pause_active", False)
        ):
            elapsed_since_arm = max(
                0.0,
                now_mono
                - float(
                    getattr(
                        self._state,
                        "vol_auto_pause_active_since_mono",
                        0.0,
                    )
                    or 0.0
                ),
            )
            eff_q = _replace(
                eff_q,
                eligibility=QuoteEligibility.HOLD_ALL,
                reason=(
                    f"{eff_q.reason}|vol_regime_auto_paused"
                    f"|elapsed={elapsed_since_arm:.0f}s"
                ),
            )

        # --- #2 post-swing PnL cooldown ---------------------------------
        if self._settings.post_swing_enabled:
            try:
                total_pnl = float(self._state.pnl.total_pnl_usd or 0.0)
            except (TypeError, ValueError, AttributeError):
                total_pnl = 0.0
            ps_state = self._state.post_swing
            ps_fires_before = int(getattr(ps_state, "fire_count", 0) or 0)
            post_swing_gate.observe(
                ps_state,
                now_mono=now_mono,
                total_pnl_usd=total_pnl,
                window_seconds=float(self._settings.post_swing_window_seconds),
                pnl_delta_usd_threshold=float(self._settings.post_swing_pnl_delta_usd),
                cooldown_seconds=float(self._settings.post_swing_cooldown_seconds),
                # v1.4.154 Phase 2K.4 — favorable-exit predicate.
                # Defensive getattr in case the bot is running
                # against a Settings shape from before the v1.4.154
                # fields existed (defaults match the function's own).
                clear_band_mult=float(
                    getattr(
                        self._settings,
                        "post_swing_clear_band_mult",
                        0.5,
                    )
                ),
                favorable_exit_dwell_seconds=float(
                    getattr(
                        self._settings,
                        "post_swing_favorable_exit_dwell_seconds",
                        15.0,
                    )
                ),
            )
            if int(getattr(ps_state, "fire_count", 0) or 0) > ps_fires_before:
                self._log_event(
                    EventSeverity.WARNING,
                    "post_swing_gate_fired",
                    (
                        f"post-swing PnL cooldown fired ({ps_state.last_trigger_reason}) — "
                        f"Δ=${ps_state.last_trigger_delta_usd:+.3f}; HOLD_ALL for "
                        f"{self._settings.post_swing_cooldown_seconds:.0f}s"
                    ),
                    {
                        "trigger_reason": ps_state.last_trigger_reason,
                        "delta_usd": ps_state.last_trigger_delta_usd,
                        "cooldown_seconds": float(
                            self._settings.post_swing_cooldown_seconds
                        ),
                        "fire_count": int(ps_state.fire_count),
                    },
                )
            ps_firing = post_swing_gate.is_active(
                self._state.post_swing, now_mono
            )
            # 1.4.7 Phase 0 telemetry.
            self._state.record_gate_firing(
                "post_swing_gate", firing_now=ps_firing, now_mono=now_mono
            )
            # 1.4.12 cutover: post_swing no longer clamps eligibility.
            # post_swing_gate.widening_bps (in build_spread_composition)
            # carries the response.
            if ps_firing:
                ps = self._state.post_swing
                remaining = post_swing_gate.seconds_remaining(ps, now_mono)
                eff_q = _replace(
                    eff_q,
                    reason=(
                        f"{eff_q.reason}|post_swing:{ps.last_trigger_reason or 'unknown'}"
                        f"|delta=${ps.last_trigger_delta_usd:+.3f}"
                        f"|remaining={remaining:.1f}s"
                    ),
                )

        # --- #2b 30s-MAE gate (1.3.82) ----------------------------------
        # Parallel to the toxicity-engine hard trigger, observes the
        # post-fill 30s window where directional-trend bleed shows up.
        # State is fed by the post-fill-excursion watcher daemon;
        # the only thing this path does is check is_active(). When
        # the gate is disabled (default off) the state buffer is
        # empty and is_active() is False — zero hot-path cost.
        if self._settings.mae_gate_enabled:
            from app import mae_gate

            if mae_gate.is_active(self._state.mae_gate, now_mono):
                # v1.5.155 — position-aware favorable-exit attempt
                # BEFORE the HOLD_ALL is applied. If the bot has
                # significant inventory and current drift is
                # favorable for that inventory (drift in same
                # direction as position), clear the cooldown so the
                # bot can take advantage of the favorable move to
                # unwind. Operator instruction CLAUDE.md Rule 0c.
                pos_qty_for_mae = float(
                    getattr(self._state.position, "position_qty", 0.0) or 0.0
                )
                # Use the same long-window drift signal the
                # constructive trend skew uses (v1.5.155 selector).
                drift_for_mae = self._select_trend_drift_signal(eff_q)
                cleared = mae_gate.evaluate_position_favorable_exit(
                    self._state.mae_gate,
                    now_mono=now_mono,
                    position_qty=pos_qty_for_mae,
                    drift_bps=drift_for_mae,
                    inventory_threshold=float(
                        getattr(
                            self._settings,
                            "mae_gate_position_favorable_inventory_threshold",
                            1.0,
                        )
                    ),
                    drift_threshold_bps=float(
                        getattr(
                            self._settings,
                            "mae_gate_position_favorable_drift_threshold_bps",
                            5.0,
                        )
                    ),
                )
                # v1.5.197 — also try the idle-decay clearance path. If
                # the gate has been active and no fills have arrived for
                # IDLE_CLEAR_SECONDS, clear it because its triggering
                # signal (recent fill MAE) is stale. Eliminates the
                # deadlock where the gate is active, suppresses fills,
                # and never re-evaluates its own predicate.
                if not cleared:
                    cleared = mae_gate.evaluate_idle_clear(
                        self._state.mae_gate,
                        now_mono=now_mono,
                        last_fill_mono=self._state.last_fill_at_mono,
                        idle_clear_seconds=float(
                            getattr(
                                self._settings,
                                "mae_gate_idle_clear_seconds",
                                300.0,
                            )
                        ),
                    )
                if not cleared:
                    mg = self._state.mae_gate
                    remaining = mae_gate.seconds_remaining(mg, now_mono)
                    eff_q = _replace(
                        eff_q,
                        eligibility=QuoteEligibility.HOLD_ALL,
                        reason=(
                            f"{eff_q.reason}|mae_gate"
                            f"|avg30s={mg.last_trigger_avg_bps:+.2f}bps"
                            f"|remaining={remaining:.1f}s"
                        ),
                    )

        # --- #3 inventory-aligned-with-momentum gate -------------------
        if self._settings.momentum_gate_enabled:
            pos_qty = float(self._state.position.position_qty)
            max_abs = float(self._settings.max_abs_position)
            max_notional = float(getattr(self._settings, "max_position_notional_usd", 0.0) or 0.0)
            if max_abs > 0 and mid_now and mid_now > 0:
                # Effective cap mirrors the postmortem inventory-regime
                # axis: min(qty cap, notional cap / mid).
                eff_cap = max_abs
                if max_notional > 0:
                    eff_cap = min(eff_cap, max_notional / float(mid_now))
            else:
                eff_cap = max_abs

            # Reuse the per-tick mid-return signal already computed
            # by ``compute_quote_eligibility``. 500ms covers the
            # short-horizon drift the gate cares about; falls back
            # through 250ms / 100ms when the longer window hasn't
            # accumulated samples yet (warmup / first ticks).
            drift_bps = (
                raw_q.mid_return_500ms_bps
                or raw_q.mid_return_250ms_bps
                or raw_q.mid_return_100ms_bps
            )
            override, override_reason = evaluate_momentum_gate(
                position_qty=pos_qty,
                effective_abs_cap=eff_cap,
                drift_bps=drift_bps,
                drift_threshold_bps=float(
                    self._settings.momentum_gate_drift_threshold_bps
                ),
                inventory_pct_threshold=float(
                    self._settings.momentum_gate_inventory_pct
                ),
                enabled=True,
            )
            mom_firing = override is not None
            # 1.4.7 Phase 0 telemetry.
            self._state.record_gate_firing(
                "momentum_gate", firing_now=mom_firing, now_mono=now_mono
            )
            # 1.4.12 cutover: momentum no longer clamps eligibility.
            # The asymmetric widening contribution
            # (momentum_gate.widening_bps) widens the adding-into-trend
            # side instead of suppressing it.
            if mom_firing:
                eff_q = _replace(
                    eff_q,
                    reason=f"{eff_q.reason}|{override_reason}",
                )

        # ---- v1.5.181 Phase 2B.2 — canonical drift cache ----------------
        # Compute the multi-horizon mid-drift dataclass ONCE per tick
        # before any consumer reads it. All gates (shock_gate /
        # inventory_drift_gate / regime_controller path / spread-
        # composition) consume from ``state.mid_drift_windows`` rather
        # than recomputing. Always-on so observers (dashboard / postmortem)
        # see the cache even when shock_gate is disabled.
        try:
            _mkt = self._state.market
            _mid_now_cache = (
                float(_mkt.mid_price)
                if _mkt is not None and _mkt.mid_price is not None
                else None
            )
        except Exception:
            _mid_now_cache = None
        if _mid_now_cache is not None and _mid_now_cache > 0:
            try:
                from app.mid_drift_windows import (
                    compute_mid_drift_windows as _mdw_compute,
                )
                self._state.mid_drift_windows = _mdw_compute(
                    self._state.mid_price_samples_long_snapshot(),
                    now_mono=now_mono,
                    mid_now=_mid_now_cache,
                )
            except Exception:  # noqa: BLE001 -- never break the tick
                self._state.mid_drift_windows = None
        else:
            self._state.mid_drift_windows = None

        # ---- Phase 8A (v1.5.185) — Avellaneda-Stoikov k-intensity cache.
        # Refreshed every ``AS_K_INTENSITY_REFRESH_SECONDS`` (default
        # 60 s) — NOT per-tick — so the O(N) scan over ``recent_fills``
        # (cap 1000) stays off the hot path. Cached value lives on
        # state.as_k_intensity_per_min; ``compute_quote_decision``
        # reads it when AS_ENABLED. Defensive: never break the tick
        # if anything in the helper raises.
        if bool(getattr(self._settings, "avellaneda_stoikov_enabled", False)):
            try:
                refresh_interval = float(
                    self._settings.as_k_intensity_refresh_seconds
                )
                last_refresh = float(
                    self._state.as_k_intensity_last_refresh_mono
                )
                if (
                    self._state.as_k_intensity_per_min is None
                    or now_mono - last_refresh >= refresh_interval
                ):
                    from app.avellaneda_stoikov import (
                        estimate_k_intensity_per_min as _est_k,
                    )
                    self._state.as_k_intensity_per_min = _est_k(
                        self._state.recent_fills,
                        now=self._clock.now_utc(),
                        window_seconds=float(
                            self._settings.as_k_intensity_window_seconds
                        ),
                    )
                    self._state.as_k_intensity_last_refresh_mono = now_mono
            except Exception:  # noqa: BLE001 — never break the tick
                # Leave cached value as-is; AS path falls back to floor.
                pass
        else:
            # Clear the cache when AS is disabled so a re-enable
            # forces a fresh recompute (avoids stale-cache surprises
            # if the operator flips the flag mid-session).
            self._state.as_k_intensity_per_min = None
            self._state.as_k_intensity_last_refresh_mono = 0.0

        # ---- v1.4.107 Phase 1B — shock_gate (acute spike, binary) -------
        # Unlike the gates above, shock_gate is intentionally NOT
        # gate-to-widening: at SHOCK magnitudes the bot's freshness risk
        # dominates spread economics, and wider quotes cannot fix a
        # stale-price-vs-shock-mid mismatch. Apply as a binary
        # eligibility clamp via `more_restrictive`.
        if self._settings.shock_gate_enabled:
            from app import shock_gate as _shock_gate_mod

            sg_state = self._state.shock_gate
            sg_fires_before = int(getattr(sg_state, "fire_count", 0) or 0)
            # v1.5.181 Phase 2B.2 — read drift from the canonical
            # cache populated above instead of recomputing per-gate.
            _mid_now = _mid_now_cache
            _sg_d10: Optional[float] = None
            _sg_d30: Optional[float] = None
            _mdw = self._state.mid_drift_windows
            if _mdw is not None:
                _sg_d10 = _mdw.drift_10s_bps
                _sg_d30 = _mdw.drift_30s_bps
            try:
                _pos_qty = float(self._state.position.position_qty)
            except Exception:
                _pos_qty = 0.0
            sg_override, sg_reason = _shock_gate_mod.observe(
                sg_state,
                now_mono=now_mono,
                position_qty=_pos_qty,
                effective_abs_cap=float(self._settings.max_abs_position),
                drift_bps_10s=_sg_d10,
                drift_bps_30s=_sg_d30,
                enabled=True,
                shock_inventory_pct_threshold=float(
                    self._settings.shock_inventory_pct_threshold
                ),
                shock_threshold_bps_10s=float(
                    self._settings.shock_threshold_bps_10s
                ),
                shock_threshold_bps_30s=float(
                    self._settings.shock_threshold_bps_30s
                ),
                clear_util_threshold=float(
                    self._settings.shock_clear_util_threshold
                ),
                max_cooldown_seconds=float(
                    self._settings.shock_max_cooldown_seconds
                ),
            )
            if int(getattr(sg_state, "fire_count", 0) or 0) > sg_fires_before:
                # First-arm WARNING log per plan 1B.4. NO Telegram
                # alert at this phase (lands in Phase 2G).
                self._log_event(
                    EventSeverity.WARNING,
                    "shock_gate_fired",
                    (
                        f"shock_gate fired — "
                        f"util={sg_state.last_trigger_util:.3f}, "
                        f"drift_{sg_state.last_trigger_window_label}="
                        f"{sg_state.last_trigger_drift_bps:+.2f} bps; "
                        f"side={sg_state.locked_side}; lock until clear"
                    ),
                    {
                        "util": float(sg_state.last_trigger_util),
                        "drift_bps": float(sg_state.last_trigger_drift_bps),
                        "window": str(sg_state.last_trigger_window_label),
                        "locked_side": str(sg_state.locked_side),
                        "fire_count": int(sg_state.fire_count),
                    },
                )
            self._state.record_gate_firing(
                "shock_gate",
                firing_now=bool(sg_state.locked),
                now_mono=now_mono,
            )
            if sg_override is not None:
                eff_q = _replace(
                    eff_q,
                    eligibility=more_restrictive(
                        eff_q.eligibility, sg_override
                    ),
                    reason=f"{eff_q.reason}|{sg_reason}",
                )

        # ---- v1.4.112 Phase 1C — regime_controller FSM --------------------
        # Aggregates Phase 1A (inventory_drift_gate), Phase 1B (shock_gate),
        # the existing slow_trend_gate, vol-ratio, and util signals into a
        # single NORMAL / DEFENSIVE / SHOCK mode label. Publishes per-tick
        # knob overlays consumed by ``compute_quote_decision``.
        if self._settings.regime_controller_enabled:
            from app import regime_controller as _regime_mod
            # v1.5.181 Phase 2B.2 — drift values read from
            # ``state.mid_drift_windows`` cache; only the gate
            # evaluator import is still needed here.
            from app.inventory_drift_gate import (
                evaluate_inventory_drift_gate as _eval_id_rc,
            )
            from app.slow_trend_gate import (
                evaluate_slow_trend_gate as _eval_st_rc,
            )

            # Util — same definition shock_gate uses.
            try:
                _rc_pos_qty = float(self._state.position.position_qty)
            except Exception:
                _rc_pos_qty = 0.0
            _rc_cap = float(self._settings.max_abs_position) or 1.0
            _rc_util = (
                abs(_rc_pos_qty) / _rc_cap if _rc_cap > 0 else 0.0
            )

            # Vol ratio — from toxicity engine. ``vol_spike_ratio`` is
            # current_vol_bps / baseline; threshold is ratio-style
            # (≥2.0 = vol doubled) per plan.
            _rc_vol_ratio: Optional[float] = None
            try:
                _rc_vol_ratio = float(
                    getattr(self._state.toxicity, "vol_spike_ratio", 1.0)
                    or 1.0
                )
            except Exception:
                _rc_vol_ratio = None

            # slow_trend_active: re-evaluate the gate cheaply. Stateless
            # function reading the same mid-history deque already
            # snapshotted upstream.
            _rc_slow_trend_active = False
            try:
                if (
                    self._settings.slow_trend_gate_enabled
                    and self._state.market is not None
                    and self._state.market.mid_price is not None
                    and self._state.market.mid_price > 0
                ):
                    _st_over, _, _ = _eval_st_rc(
                        samples_long=self._state.mid_price_samples_long_snapshot(),
                        now_mono=now_mono,
                        mid_now=float(self._state.market.mid_price),
                        enabled=True,
                        window_seconds=float(
                            self._settings.slow_trend_window_seconds
                        ),
                        threshold_bps=float(
                            self._settings.slow_trend_threshold_bps
                        ),
                        min_samples=int(
                            self._settings.slow_trend_min_samples
                        ),
                        anchor_fraction=float(
                            self._settings.slow_trend_anchor_fraction
                        ),
                    )
                    _rc_slow_trend_active = _st_over is not None
            except Exception:
                _rc_slow_trend_active = False

            # inventory_drift_active: same — re-evaluate using the same
            # 10s/30s drifts the spread-composition path uses.
            # v1.5.181 Phase 2B.2 — read from the canonical
            # ``state.mid_drift_windows`` cache populated at the top
            # of the tick instead of recomputing.
            _rc_inv_drift_active = False
            try:
                if self._settings.inventory_drift_gate_enabled:
                    _mkt = self._state.market
                    _mid_now_rc = (
                        float(_mkt.mid_price)
                        if _mkt is not None
                        and _mkt.mid_price is not None
                        else None
                    )
                    if _mid_now_rc is not None and _mid_now_rc > 0:
                        _mdw_rc = self._state.mid_drift_windows
                        _d10_rc = _mdw_rc.drift_10s_bps if _mdw_rc else None
                        _d30_rc = _mdw_rc.drift_30s_bps if _mdw_rc else None
                        _id_over_rc, _, _ = _eval_id_rc(
                            position_qty=_rc_pos_qty,
                            effective_abs_cap=_rc_cap,
                            drift_bps_10s=_d10_rc,
                            drift_bps_30s=_d30_rc,
                            inventory_pct_threshold=float(
                                self._settings.inventory_drift_inventory_pct_threshold
                            ),
                            drift_threshold_bps_10s=float(
                                self._settings.inventory_drift_threshold_bps_10s
                            ),
                            drift_threshold_bps_30s=float(
                                self._settings.inventory_drift_threshold_bps_30s
                            ),
                            enabled=True,
                        )
                        _rc_inv_drift_active = _id_over_rc is not None
            except Exception:
                _rc_inv_drift_active = False

            # Phase 4G.5 (v1.4.211) — forward-classifier wire-up.
            # When ``regime_forward_enabled`` is True, append current
            # readings to the history buffers and classify the forward
            # regime. The result is passed to ``evaluate_mode`` as
            # ``forward_regime=`` and drives the CALM / CAUTIOUS entry
            # paths. When disabled (the default), we skip the classifier
            # entirely and pass ``forward_regime=None`` — pre-4G
            # behavioural identity is preserved.
            _rc_forward_regime = None
            _rc_forward_reading = None
            if self._settings.regime_forward_enabled:
                from app.regime_forward_signals import (
                    ForwardSignalThresholds as _FwdThresh,
                    classify_forward_regime as _classify_forward,
                )

                # Current readings — defensive against missing data on
                # early ticks. ``vol_bps`` lives on BotState; the 30 s
                # drift was computed above for the inventory-drift gate.
                _fwd_vol_bps: Optional[float] = None
                try:
                    _fwd_vol_bps = float(getattr(self._state, "vol_bps", None))
                except Exception:
                    _fwd_vol_bps = None
                # 30 s drift — re-use the one computed above when the
                # inventory_drift gate path ran; otherwise compute
                # independently so the forward classifier doesn't depend
                # on ``inventory_drift_gate_enabled`` being True.
                _fwd_drift_30s: Optional[float] = None
                try:
                    _fwd_drift_30s = float(_d30_rc)  # type: ignore[has-type]  # noqa: F821
                except Exception:
                    _fwd_drift_30s = None
                if _fwd_drift_30s is None:
                    # v1.5.181 Phase 2B.2 — read from canonical cache.
                    try:
                        _mdw_f = self._state.mid_drift_windows
                        _d30_f = _mdw_f.drift_30s_bps if _mdw_f else None
                        if _d30_f is not None:
                            _fwd_drift_30s = float(_d30_f)
                    except Exception:
                        _fwd_drift_30s = None
                _fwd_ob_imb: Optional[float] = None
                try:
                    _fwd_ob_imb = (
                        float(self._state.ob_imbalance_ewma)
                        if getattr(self._state, "ob_imbalance_ewma", None) is not None
                        else None
                    )
                except Exception:
                    _fwd_ob_imb = None
                _fwd_basis_bps: Optional[float] = None
                try:
                    _fwd_basis_bps = (
                        float(self._state.binance_basis_ewma) * 10_000.0
                        if getattr(self._state, "binance_basis_ewma", None) is not None
                        else None
                    )
                except Exception:
                    _fwd_basis_bps = None
                # v1.5.18 Phase 4G.7 -- the 4th CAUTIOUS leading
                # indicator. Append current basis to the rolling
                # buffer (separate, longer retention window) and read
                # the median back for the classifier. Until 4G.7 this
                # was ``None`` and the basis_stretch trigger was
                # dormant.
                _fwd_basis_median_bps: Optional[float] = (
                    self._state.median_forward_basis_bps()
                )

                # Append + truncate history buffers.
                # v1.5.239 — also append the trend-aware EWMA value.
                # Read live from the estimator (not from the snapshot
                # variable above) so we capture any update since the
                # tick started. ``None`` during the EWMA's 2-mid
                # warm-up; append_forward_signal_history silently
                # skips None.
                _fwd_vol_abs_ewma_bps: Optional[float]
                try:
                    _fwd_vol_abs_ewma_bps = (
                        self._state.vol_abs_ewma.value_bps()
                    )
                except Exception:
                    _fwd_vol_abs_ewma_bps = None
                self._state.append_forward_signal_history(
                    now_mono=now_mono,
                    vol_bps=_fwd_vol_bps,
                    drift_30s_bps=_fwd_drift_30s,
                    ob_imbalance=_fwd_ob_imb,
                    max_age_seconds=float(
                        self._settings.regime_forward_history_buffer_seconds
                    ),
                    basis_bps=_fwd_basis_bps,
                    basis_max_age_seconds=float(
                        getattr(
                            self._settings,
                            "regime_forward_basis_median_lookback_seconds",
                            1800.0,
                        )
                    ),
                    vol_abs_ewma_bps=_fwd_vol_abs_ewma_bps,
                )

                _fwd_thresholds = _FwdThresh(
                    # v1.5.191 band hysteresis — enter/exit pairs
                    # per criterion. See app/regime_forward_signals.py
                    # ForwardSignalThresholds docstring.
                    cautious_enter_vol_slope_bps_per_min=float(
                        self._settings.regime_forward_cautious_enter_vol_slope_bps_per_min
                    ),
                    cautious_exit_vol_slope_bps_per_min=float(
                        self._settings.regime_forward_cautious_exit_vol_slope_bps_per_min
                    ),
                    vol_slope_lookback_seconds=float(
                        self._settings.regime_forward_vol_slope_lookback_seconds
                    ),
                    cautious_enter_drift_magnitude_rising_ratio=float(
                        self._settings.regime_forward_cautious_enter_drift_magnitude_rising_ratio
                    ),
                    cautious_exit_drift_magnitude_rising_ratio=float(
                        self._settings.regime_forward_cautious_exit_drift_magnitude_rising_ratio
                    ),
                    drift_magnitude_lookback_seconds=float(
                        self._settings.regime_forward_drift_magnitude_lookback_seconds
                    ),
                    drift_magnitude_floor_bps=float(
                        self._settings.regime_forward_drift_magnitude_floor_bps
                    ),
                    cautious_enter_ob_imbalance_widening_delta=float(
                        self._settings.regime_forward_cautious_enter_ob_imbalance_widening_delta
                    ),
                    cautious_exit_ob_imbalance_widening_delta=float(
                        self._settings.regime_forward_cautious_exit_ob_imbalance_widening_delta
                    ),
                    ob_imbalance_lookback_seconds=float(
                        self._settings.regime_forward_ob_imbalance_lookback_seconds
                    ),
                    cautious_enter_basis_stretch_ratio=float(
                        self._settings.regime_forward_cautious_enter_basis_stretch_ratio
                    ),
                    cautious_exit_basis_stretch_ratio=float(
                        self._settings.regime_forward_cautious_exit_basis_stretch_ratio
                    ),
                    basis_stretch_floor_bps=float(
                        self._settings.regime_forward_basis_stretch_floor_bps
                    ),
                    calm_enter_max_vol_bps=float(
                        self._settings.regime_forward_calm_enter_max_vol_bps
                    ),
                    calm_exit_max_vol_bps=float(
                        self._settings.regime_forward_calm_exit_max_vol_bps
                    ),
                    calm_enter_max_drift_magnitude_bps=float(
                        self._settings.regime_forward_calm_enter_max_drift_magnitude_bps
                    ),
                    calm_exit_max_drift_magnitude_bps=float(
                        self._settings.regime_forward_calm_exit_max_drift_magnitude_bps
                    ),
                    calm_enter_max_ob_imbalance_magnitude=float(
                        self._settings.regime_forward_calm_enter_max_ob_imbalance_magnitude
                    ),
                    calm_exit_max_ob_imbalance_magnitude=float(
                        self._settings.regime_forward_calm_exit_max_ob_imbalance_magnitude
                    ),
                    calm_min_history_seconds=float(
                        self._settings.regime_forward_calm_min_history_seconds
                    ),
                )
                # v1.5.191 — pass current FSM mode into the classifier
                # so the band-hysteresis logic picks the right
                # enter/exit threshold per criterion. Translate the
                # FSM ``Mode`` (which has DEFENSIVE / SHOCK too) into
                # the classifier's ``ForwardRegime`` (CALM / NORMAL /
                # CAUTIOUS only) — non-classifier modes are treated
                # as NORMAL for threshold-selection purposes.
                from app.regime_forward_signals import ForwardRegime as _FR
                _cur_fsm = self._state.regime_controller.mode
                if _cur_fsm is _regime_mod.Mode.CALM:
                    _cur_fwd = _FR.CALM
                elif _cur_fsm is _regime_mod.Mode.CAUTIOUS:
                    _cur_fwd = _FR.CAUTIOUS
                else:
                    _cur_fwd = _FR.NORMAL
                # v1.5.239 — when REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE
                # is on, the classifier's vol_slope criterion reads
                # from the trend-aware EWMA history instead of the
                # stdev-based vol_bps history. Both buffers are
                # populated each tick (see append_forward_signal_history
                # above), so flipping the flag at runtime is safe —
                # the alternative buffer is already warm.
                _vol_history_for_slope = (
                    self._state.forward_vol_abs_ewma_bps_history
                    if getattr(
                        self._settings,
                        "regime_forward_use_vol_abs_ewma_for_slope",
                        False,
                    )
                    else self._state.forward_vol_bps_history
                )
                _rc_forward_reading = _classify_forward(
                    vol_bps_history=_vol_history_for_slope,
                    drift_30s_history=self._state.forward_drift_30s_history,
                    ob_imbalance_history=self._state.forward_ob_imbalance_history,
                    current_ob_imbalance=_fwd_ob_imb,
                    current_vol_bps=_fwd_vol_bps,
                    current_drift_30s_bps=_fwd_drift_30s,
                    current_binance_basis_bps=_fwd_basis_bps,
                    binance_basis_30min_median_bps=_fwd_basis_median_bps,
                    settings=_fwd_thresholds,
                    now_mono=now_mono,
                    current_mode=_cur_fwd,
                )
                _rc_forward_regime = _rc_forward_reading.classification
                # Cache the full reading on state so snapshot_dict +
                # postmortem can surface diagnostics.
                self._state.last_forward_signal_reading = _rc_forward_reading

            _rc_state = self._state.regime_controller
            _prior_mode = _rc_state.mode
            _rc_mode, _rc_transition_reason = _regime_mod.evaluate_mode(
                _rc_state,
                now_mono=now_mono,
                util=float(_rc_util),
                vol_ratio=_rc_vol_ratio,
                slow_trend_active=bool(_rc_slow_trend_active),
                inventory_drift_active=bool(_rc_inv_drift_active),
                shock_gate_locked=bool(
                    getattr(self._state.shock_gate, "locked", False)
                ),
                enabled=True,
                entry_dwell_seconds=float(
                    self._settings.regime_entry_dwell_seconds
                ),
                exit_dwell_seconds=float(
                    self._settings.regime_exit_dwell_seconds
                ),
                util_entry_threshold=float(
                    self._settings.regime_util_entry_threshold
                ),
                util_exit_threshold=float(
                    self._settings.regime_util_exit_threshold
                ),
                vol_ratio_entry_threshold=float(
                    self._settings.regime_vol_ratio_entry_threshold
                ),
                # Phase 4G.5 — forward-classifier hook + dwell timings.
                # v1.5.231 — also pass the classifier's reason text so
                # the FSM transition log records WHICH criterion fired
                # (vol_slope / drift_rising / ob_widening / basis_stretch
                # or the calm-failure tag). Lets the operator count
                # criterion firings from the postmortem.
                forward_regime=_rc_forward_regime,
                forward_reason=(
                    getattr(_rc_forward_reading, "reason", None)
                    if _rc_forward_reading is not None
                    else None
                ),
                cautious_entry_dwell_seconds=float(
                    self._settings.regime_forward_cautious_entry_dwell_seconds
                ),
                cautious_exit_dwell_seconds=float(
                    self._settings.regime_forward_cautious_exit_dwell_seconds
                ),
                calm_entry_dwell_seconds=float(
                    self._settings.regime_forward_calm_entry_dwell_seconds
                ),
                calm_exit_dwell_seconds=float(
                    self._settings.regime_forward_calm_exit_dwell_seconds
                ),
            )
            # Accumulate session-cumulative time-in-mode.
            _regime_mod.accumulate_time_in_mode(_rc_state, now_mono=now_mono)
            # Publish the knob overlay for this tick. Consumers
            # (``compute_quote_decision``) read ``state.regime_knobs``.
            self._state.regime_knobs = _regime_mod.compute_knobs_for_mode(
                _rc_mode,
                ladder_levels_max_config=int(
                    getattr(self._settings, "ladder_num_levels_per_side", 1)
                ),
            )
            # First-arm WARNING log on actual mode change. v1.4.204
            # Phase 2G: ADDITIONALLY forward SHOCK transitions to the
            # Telegram ops channel via ``_maybe_alert_shock_telegram``.
            # Only SHOCK qualifies for real-time alerts (memory note
            # ``feedback_postmortem_preferred_over_runtime_alerts`` —
            # routine NORMAL ↔ DEFENSIVE transitions stay
            # postmortem-only).
            if (
                _rc_transition_reason is not None
                and _prior_mode is not _rc_mode
            ):
                self._log_event(
                    EventSeverity.WARNING,
                    "regime_mode_transition",
                    (
                        f"regime mode {_prior_mode.value} → "
                        f"{_rc_mode.value}; {_rc_transition_reason}"
                    ),
                    {
                        "from": _prior_mode.value,
                        "to": _rc_mode.value,
                        "reason": _rc_transition_reason,
                        "util": float(_rc_util),
                        "vol_ratio": (
                            float(_rc_vol_ratio)
                            if _rc_vol_ratio is not None
                            else None
                        ),
                    },
                )

            # Phase 2G (v1.4.204) — SHOCK Telegram alerts. Runs on
            # EVERY regime tick (not just transitions) so the
            # persistence-> 5min alert can fire mid-episode. Helper
            # short-circuits to no-op when not in SHOCK / when
            # alerts disabled / when notifier absent / when already
            # alerted for this episode.
            try:
                self._maybe_alert_shock_telegram(
                    mode=_rc_mode,
                    prior_mode=_prior_mode,
                    transition_reason=_rc_transition_reason,
                    util=float(_rc_util),
                    vol_ratio=(
                        float(_rc_vol_ratio)
                        if _rc_vol_ratio is not None
                        else None
                    ),
                    now_mono=now_mono,
                    mode_since_mono=float(
                        getattr(_rc_state, "mode_since_mono", 0.0) or 0.0
                    ),
                )
            except Exception:  # noqa: BLE001
                logger.exception("shock_telegram_alert_check_failed")

            # v1.5.26 Phase 2G.3 -- daily time-in-mode summary on the
            # same hook point. Short-circuits inside the helper when:
            # disabled, no notifier, interval not elapsed. Hot-path safe.
            try:
                self._maybe_send_daily_regime_summary(now_mono=now_mono)
            except Exception:  # noqa: BLE001
                logger.exception("regime_daily_summary_check_failed")

            # ---- v1.4.113 Phase 1D — post-reduction cooldown ------------
            # Fires ONLY when:
            #   1. The cooldown feature is enabled (seconds > 0)
            #   2. A prior fill armed the cooldown (timestamp + side set)
            #   3. Time since arming < cooldown_seconds                  [MAX ceiling]
            #   4. regime_controller.mode is DEFENSIVE or SHOCK
            #      (NORMAL keeps round-trip rebate capture intact)
            #   5. (v1.4.144 Phase 2K.1) The favorable-exit predicate
            #      is NOT satisfied — i.e. position util is still at or
            #      above ``post_reduction_cooldown_clear_util_pct``. As
            #      soon as util drops below this threshold, the cooldown
            #      clears early because the gate's original "don't
            #      fast-flip into the just-exited side" intent is met
            #      by the natural unwind.
            # Composes with the regime FSM via more_restrictive() — does
            # not relax existing eligibility, only narrows.
            prc_seconds = float(
                getattr(self._settings, "post_reduction_cooldown_seconds", 0.0)
                or 0.0
            )
            if (
                prc_seconds > 0.0
                and _rc_mode is not _regime_mod.Mode.NORMAL
                and self._state.last_inventory_reduction_at_mono is not None
                and self._state.last_inventory_reduction_suppressed_side
                is not None
            ):
                elapsed = (
                    now_mono
                    - float(self._state.last_inventory_reduction_at_mono)
                )
                # MAX-ceiling check: time-since-arm vs configured seconds.
                ceiling_satisfied = 0.0 <= elapsed < prc_seconds
                # Favorable-exit predicate: current absolute position
                # utilisation relative to MAX_ABS_POSITION. NaN-safe
                # via the same try/except pattern other gates use.
                clear_util_pct = float(
                    getattr(
                        self._settings,
                        "post_reduction_cooldown_clear_util_pct",
                        0.30,
                    )
                    or 0.0
                )
                util_above_threshold = True
                try:
                    abs_cap = float(self._settings.max_abs_position)
                    if abs_cap > 1e-12:
                        cur_util = abs(
                            float(self._state.position.position_qty)
                        ) / abs_cap
                        util_above_threshold = cur_util >= clear_util_pct
                except (TypeError, ValueError, AttributeError):
                    # Defensive: if we can't read util, fall back to
                    # the pure-timer behaviour (util considered "high").
                    util_above_threshold = True
                if ceiling_satisfied and util_above_threshold:
                    suppress_side = (
                        self._state.last_inventory_reduction_suppressed_side
                    )
                    # First-arm WARNING log per cooldown window. The
                    # ``was_active_last_tick`` flag re-arms when the
                    # cooldown expires AND a fresh fill re-arms it.
                    if not self._state.post_reduction_cooldown_was_active_last_tick:
                        # v1.5.191 BUG-032 — bump fire_count HERE at the
                        # actual engagement edge (not in state.py on
                        # every reducing fill). This makes
                        # ``fire_count`` semantically match
                        # ``cleared_via_*``: both count
                        # active-cooldown-window engagements, which is
                        # what the AC + dashboard interpret the counter
                        # as. See issues/bug-032.md.
                        self._state.post_reduction_cooldown_fire_count = (
                            int(self._state.post_reduction_cooldown_fire_count) + 1
                        )
                        self._log_event(
                            EventSeverity.WARNING,
                            "post_reduction_cooldown_armed",
                            (
                                f"post-reduction cooldown armed for "
                                f"{prc_seconds:.0f} s; suppressed_side="
                                f"{suppress_side.name}; mode={_rc_mode.value}"
                            ),
                            {
                                "cooldown_seconds": float(prc_seconds),
                                "suppressed_side": suppress_side.name,
                                "mode": _rc_mode.value,
                                "remaining_seconds": float(
                                    prc_seconds - elapsed
                                ),
                                "clear_util_pct": float(clear_util_pct),
                            },
                        )
                    self._state.post_reduction_cooldown_was_active_last_tick = True
                    eff_q = _replace(
                        eff_q,
                        eligibility=more_restrictive(
                            eff_q.eligibility, suppress_side
                        ),
                        reason=(
                            f"{eff_q.reason}|post_reduction_cooldown:"
                            f"side={suppress_side.name},rem="
                            f"{(prc_seconds - elapsed):.1f}s"
                        ),
                    )
                else:
                    # Cooldown cleared this tick — attribute the exit
                    # to either the favorable-exit predicate (util
                    # dropped below threshold while ceiling still in
                    # window) or the MAX-cooldown ceiling firing as a
                    # safety net. Counter increments only on the
                    # arming → clearing edge; further "still cleared"
                    # ticks do not double-count.
                    if self._state.post_reduction_cooldown_was_active_last_tick:
                        if ceiling_satisfied and not util_above_threshold:
                            self._state.post_reduction_cooldown_cleared_via_favorable_total += 1
                        else:
                            self._state.post_reduction_cooldown_cleared_via_ceiling_total += 1
                    # Also clear the arming timestamps so a stale ref
                    # doesn't persist — the next reduction-fill will
                    # re-arm via ``_note_inventory_reduction_for_cooldown``.
                    self._state.last_inventory_reduction_at_mono = None
                    self._state.last_inventory_reduction_suppressed_side = None
                    self._state.post_reduction_cooldown_was_active_last_tick = False
            else:
                # Either feature off, mode NORMAL, or no fill ever
                # armed it. Keep flag false.
                self._state.post_reduction_cooldown_was_active_last_tick = False

        return eff_q

    def _build_quote_eligibility_diag(
        self,
        raw_q: QuoteEligibilityResult,
        eff_q: QuoteEligibilityResult,
    ) -> dict[str, Any]:
        """Per-tick eligibility diagnostic payload.

        Shared by ``quote_eligibility_transition`` and
        ``quote_eligibility_tick_sample`` so a single live log line attributes
        a clamp to the specific rule that produced it (gap, staleness, drift,
        jump, order-state) instead of just reporting the effective cap.
        """
        rf = self._state.quote_elig_recovery_floor
        return {
            "raw_eligibility": raw_q.eligibility.value,
            "raw_reason": raw_q.reason[:400],
            "counter_tags": list(eff_q.counter_tags),
            "in_cooldown": bool(eff_q.in_cooldown),
            "recovery_floor": rf.value if rf is not None else None,
            "recovery_remaining_ms": self._state.quote_elig_recovery_remaining_ms,
            "seconds_since_public_book_update": eff_q.seconds_since_last_public_book_update,
            "effective_staleness_ms": eff_q.effective_staleness_ms,
            "gap_p95_ms": eff_q.market_data_gap_p95_ms,
            "gap_median_ms": eff_q.market_data_gap_median_ms,
            "mid_return_100ms_bps": raw_q.mid_return_100ms_bps,
            "mid_return_250ms_bps": raw_q.mid_return_250ms_bps,
            "mid_return_500ms_bps": raw_q.mid_return_500ms_bps,
            "jump_100ms_bps": raw_q.jump_100ms_bps,
            "jump_250ms_bps": raw_q.jump_250ms_bps,
            "jump_500ms_bps": raw_q.jump_500ms_bps,
            "order_state_uncertainty": self._exec.has_order_state_uncertainty(),
        }

    # todo-019 Part A — default allow-list for the inverted
    # NO_QUOTE cancel policy. Used when the bot is called as an
    # instance and the operator hasn't overridden via env, AND
    # when tests / external callers invoke the staticmethod
    # without an instance (backwards-compat with bug-009-era
    # tests).
    _NO_QUOTE_DEFAULT_ALLOW_LIST: frozenset[str] = frozenset(
        {"recovery_cooldown", "trade_rate_limit"}
    )

    @staticmethod
    def _should_cancel_resting_on_no_quote(
        reasons: list[str],
        allow_list: frozenset[str] | set[str] | None = None,
    ) -> bool:
        """todo-019 Part A — invert NO_QUOTE cancel policy.

        Pre-1.2.79 this was a severe-data-only whitelist (BUG-009
        era): only ``stale_data_warn`` / ``killed`` etc. would
        cancel; every other NO_QUOTE reason (toxicity gate, vol
        regime, drawdown gate, etc.) left existing orders resting
        while the engine refused to add new ones. Codex's review
        #1 flagged that as a dominant source of stale-quote toxic
        fills — the bot decides not to quote but old quotes
        continue advertising liquidity at increasingly stale
        prices.

        v1.2.79 inverts the policy: every NO_QUOTE cycle cancels
        resting orders unless EVERY reason is in the allow-list.
        Empty / missing reasons → cancel defensively.

        ``allow_list`` defaults to ``_NO_QUOTE_DEFAULT_ALLOW_LIST``
        (``{"recovery_cooldown", "trade_rate_limit"}``). The bot
        wrapper ``_should_cancel_resting_on_no_quote_bound`` below
        reads the operator-configurable allow-list from settings
        and forwards here.

        Allow-list rationale:
        - ``recovery_cooldown`` — sub-second post-fill pause; the
          existing order is fresh and re-placing would just churn.
        - ``trade_rate_limit`` — about to re-emit anyway when the
          window opens; cancelling adds nothing.

        Risk: cancel churn on quiet markets when a brief NO_QUOTE
        reason fires repeatedly. Walk-back recipe is to expand
        ``NO_QUOTE_KEEP_RESTING_REASONS`` env, not revert code.

        Stop-loss criterion: if
        ``avg_passive_order_lifetime_seconds`` drops below ~0.5 s
        post-deploy, the inversion is over-cancelling and the
        allow-list needs another entry.

        Pre-1.2.79 behavior change: ``account_data_stale`` /
        ``stale_data_warn`` etc. were on the cancel-promote list.
        Under the new allow-list semantics they're not in the
        allow-list, so they continue to trigger cancel — the
        change is forward-compatible for all the bug-009-era
        severe reasons.
        """
        if not reasons:
            return True  # malformed → cancel defensively
        allow = (
            allow_list
            if allow_list is not None
            else Bot._NO_QUOTE_DEFAULT_ALLOW_LIST
        )
        # If every reason is in the allow-list → keep resting.
        # Otherwise cancel.
        return not all(r in allow for r in reasons)

    def _should_cancel_resting_on_no_quote_bound(
        self, reasons: list[str]
    ) -> bool:
        """Instance wrapper that reads the operator-configurable
        allow-list from settings before delegating to the
        staticmethod."""
        allow_raw = getattr(
            self._settings, "no_quote_keep_resting_reasons", ""
        ) or ""
        allow = frozenset(
            {p.strip() for p in allow_raw.split(",") if p.strip()}
        )
        if not allow:
            allow = self._NO_QUOTE_DEFAULT_ALLOW_LIST
        return self._should_cancel_resting_on_no_quote(reasons, allow)

    def _handle_inventory_consistency_breach(self, breach: InventoryBreach) -> None:
        """TODO-001: arm manual_pause + cancel resting + alert on breach.

        Idempotent across consecutive breaches (the watchdog only fires once
        per check interval, but a persistent divergence would re-fire each
        cycle). The first breach arms manual_pause; subsequent ones log/alert
        again so the operator gets a follow-up signal that the divergence
        hasn't resolved.
        """
        payload = dict(breach.to_payload())
        payload["breach_count"] = self._state.inventory_consistency_breach_count
        msg = (
            f"inventory_consistency_breach venue_qty={breach.venue_qty:+.4f} "
            f"expected={breach.expected_qty:+.4f} drift={breach.drift:+.4f} "
            f"tol={breach.tolerance:.4f} symbol={breach.symbol}"
        )
        try:
            self._storage.insert_bot_event(
                self._clock.now_utc().isoformat(),
                EventSeverity.CRITICAL.value,
                "inventory_consistency_breach",
                msg,
                payload,
            )
        except Exception:
            logger.exception("inventory_consistency_event_write_failed")
        # First breach: arm manual_pause and cancel resting orders so the
        # bot stops trading until the operator confirms the venue state.
        with self._state._lock:
            already_paused = self._state.manual_pause
            self._state.manual_pause = True
        if not already_paused:
            try:
                self._exec.cancel_resting_for_risk(RiskAction.NO_QUOTE)
            except Exception:
                logger.exception("inventory_consistency_cancel_failed")
        # Telegram CRITICAL — best-effort; failure to deliver should not
        # crash the bot loop.
        notifier = self._notifier
        if notifier is not None and getattr(notifier, "enabled", False):
            try:
                notifier.notify_ops(
                    "CRITICAL",
                    "inventory_consistency_breach",
                    msg,
                    payload,
                )
            except Exception:
                logger.exception("inventory_consistency_telegram_failed")

    def _apply_tick_latency(self, lat: dict[str, Any]) -> None:
        exec_obs = self._exec.get_reconcile_runtime_counters()
        exec_obs["account_refresh_suppressed_order_uncertainty_count"] = int(
            self._account_refresh_suppressed_order_uncertainty_count
        )
        with self._state._lock:
            self._state.latency_hot_path_local_compute_ms = lat.get(
                "latency_hot_path_local_compute_ms"
            )
            self._state.latency_tick_preamble_ms = lat.get("latency_tick_preamble_ms")
            self._state.latency_quote_engine_build_ms = lat.get("latency_quote_engine_build_ms")
            self._state.latency_order_maintenance_local_ms = lat.get(
                "latency_order_maintenance_local_ms"
            )
            self._state.latency_account_refresh_rest_ms = lat.get("latency_account_refresh_rest_ms")
            self._state.latency_reconcile_rest_ms = lat.get("latency_reconcile_rest_ms")
            self._state.latency_order_submit_rtt_ms = lat.get("latency_order_submit_rtt_ms")
            self._state.latency_private_queue_wait_ms = lat.get("latency_private_queue_wait_ms")
            self._state.latency_public_ws_queue_wait_ms = lat.get(
                "latency_public_ws_queue_wait_ms"
            )
            self._state.exec_runtime_counters = dict(exec_obs)

    def one_tick(self) -> None:
        lat: dict[str, Any] = {}
        t_tick_start = time.perf_counter()
        self._exec.reset_tick_latency_metrics()
        # v1.5.5 hard-stop gate. Operator safety contract: once
        # ``bot.stop()`` has been signalled (SIGTERM / ops stop /
        # ``ops deploy``), the bot MUST place no new orders. Pre-fix,
        # ``one_tick`` ran to completion when stop arrived mid-tick,
        # which let the last tick enqueue fresh places + amends into
        # the outbound dispatcher in the ~5-60s window before the
        # systemd shutdown completed. The pre-cancel in ``bot.stop()``
        # would clear the book; the dispatcher's drain would then
        # ship the bot's just-enqueued NEW orders into that empty
        # book. Net: orders kept appearing during shutdown.
        #
        # The gate here returns BEFORE any state mutation or
        # outbound enqueue, so a tick that lands while stop is in
        # flight is a true no-op. Quote loop's outer ``_stop`` check
        # catches the same condition between ticks; this catches the
        # MID-TICK race.
        if self._stop.is_set():
            return
        try:
            if self._state.is_killed():
                return

            # Soft-flatten dedicated path. Skips normal quoting and
            # runs a worker that maintains a single post-only reduce-
            # side order at best until the position closes; then exits
            # back to normal quoting on the next tick. See
            # ``_run_soft_flatten_tick`` for the worker logic.
            #
            # CRITICAL: an abbreviated risk eval runs FIRST so kill
            # gates (drawdown, session_loss, exec_errors,
            # public_ws_disconnect, stale-data) still fire while the
            # bot is in soft-flatten mode. Pre-2026-05-06 those gates
            # were entirely bypassed during soft-flatten, which let
            # the drawdown grow well past the configured cap because
            # nothing checked it for the duration of the flatten.
            if self._state.soft_flatten_active:
                if self._risk_kill_check_during_soft_flatten():
                    return
                self._run_soft_flatten_tick()
                return

            # v1.5.33 — take-profit (TP) overlay. Runs AFTER SF check
            # (SF wins on conflict) but BEFORE normal quoting (TP is a
            # quoting takeover). Each tick maintains the post-only
            # close + checks disarm. See ``_run_take_profit_tick`` and
            # ``app/take_profit.py``.
            if self._state.tp_active:
                self._run_take_profit_tick()
                return

            # Adaptive join-depth controller (no-op when
            # ``JOIN_DEPTH_AUTOTUNE_ENABLED`` is off, which is the
            # default). Cheap on every tick — internally gated by an
            # update-interval timer, so it only does real work every
            # ``JOIN_DEPTH_AUTOTUNE_UPDATE_SECONDS`` (default 30s).
            try:
                self._state.join_depth_controller.tick(self._state)
            except Exception:
                logger.exception("join_depth_controller_tick_failed")

            self._state.maybe_rotate_operator_day_to_wall_clock()
            addr = venue_account_address(self._settings)

            self._exec.on_bot_tick_start()
            ingest_rest_fills = self._exec.should_ingest_fills_via_rest()
            self._exec.drain_private_events(self._pnl)
            do_acc_refresh, _acc_reason = self._should_refresh_account_rest(addr)
            if do_acc_refresh:
                t_acc = time.perf_counter()
                refresh_account_only(
                    self._client,
                    self._state,
                    addr,
                    self._storage,
                    self._pnl,
                    ingest_fills_via_rest=ingest_rest_fills,
                )
                lat["latency_account_refresh_rest_ms"] = (time.perf_counter() - t_acc) * 1000.0
            else:
                lat["latency_account_refresh_rest_ms"] = 0.0

            healthy = self._exchange_snapshot_healthy(addr)
            with self._state._lock:
                if healthy:
                    self._state.exchange_snapshot_unhealthy_streak = 0
                    if (
                        self._state.reconcile_auto_pause
                        and not self._state.manual_pause
                        and not self._state.killed
                        and self._state.bot_status == BotStatus.PAUSED
                    ):
                        self._state.reconcile_auto_pause = False
                        self._state.bot_status = BotStatus.RUNNING
                        self._state.clear_pause()
                        self._log_event(
                            EventSeverity.INFO,
                            "reconcile_recovered",
                            "exchange snapshot healthy — auto-resumed from reconcile stall",
                            None,
                        )
                else:
                    # Bootstrap suppression: until the first public BBO
                    # arrives, ``state.market`` is legitimately None and
                    # the snapshot is "unhealthy" by definition. The
                    # streak counter is a "we WERE healthy and now we're
                    # not" detector, not a startup delay measure --
                    # incrementing during STARTING falsely accuses
                    # bootstrap of being a stall.
                    #
                    # Triggered live on 2026-05-05 with the OKX bbo-tbt
                    # subscription: bbo-tbt is event-driven (sends on
                    # next BBO change) rather than 10Hz-throttled
                    # (books5), so a quiet ~500ms market window at
                    # subscribe time let private_ws_recovery's 5
                    # catch-up REST polls run before the first BBO
                    # landed -- streak hit 5 and PAUSE(reconcile_stall)
                    # latched. Books5's heartbeat had hidden this race.
                    if not self._state.public_ws_seen_first_bbo:
                        # Skip the increment + pause check entirely.
                        # As soon as the first BBO arrives, healthy=True
                        # on the next reconcile and streak resets to 0.
                        pass
                    else:
                        self._state.exchange_snapshot_unhealthy_streak += 1
                        stall_n = self._settings.exchange_reconcile_stall_ticks
                        if (
                            self._settings.trading_enabled
                            and (addr or "").strip()
                            and self._state.bot_status
                            in (BotStatus.RUNNING, BotStatus.STARTING)
                            and not self._state.manual_pause
                            and not self._state.killed
                            and self._state.exchange_snapshot_unhealthy_streak >= stall_n
                        ):
                            self._state.reconcile_auto_pause = True
                            self._state.bot_status = BotStatus.PAUSED
                            self._state.mark_paused("reconcile_stall")
                            self._log_event(
                                EventSeverity.WARNING,
                                "reconcile_stall_pause",
                                f"exchange snapshot unhealthy for {stall_n}+ ticks — PAUSED(reconcile_stall)",
                                {
                                    "streak": self._state.exchange_snapshot_unhealthy_streak,
                                    "stall_ticks": stall_n,
                                },
                            )

            with self._state._lock:
                m = self._state.market
                mid = m.mid_price if m else None
                pos = self._state.position
                last_w = self._state.public_ws_last_message_wall_ts
                if self._public_ws_live_path():
                    age = seconds_since(last_w) if last_w else None
                    self._state.book_age_seconds = age
                    w = float(self._settings.public_ws_stale_warn_seconds)
                    k = float(self._settings.public_ws_stale_kill_seconds)
                    self._state.stale_book_warning_active = (
                        age is not None and w <= age < k
                    )
                elif m and m.ts_local:
                    age = seconds_since(m.ts_local)
                    self._state.book_age_seconds = age
                    w = float(self._settings.stale_data_warn_seconds)
                    k = float(self._settings.stale_data_kill_seconds)
                    self._state.stale_book_warning_active = (
                        age is not None and w <= age < k
                    )
                else:
                    self._state.book_age_seconds = None
                    self._state.stale_book_warning_active = False

            if not self._logged_first_market_snapshot and m is not None:
                self._logged_first_market_snapshot = True
                log_extra(
                    logger,
                    logging.INFO,
                    "first market snapshot received after startup",
                    {
                        "event": "first_market_snapshot",
                        "symbol": self._settings.symbol,
                        "mid_price": mid,
                    },
                )

            if self._market_data_recovery_supervisor():
                return

            # Position-aware drawdown gate. Fires when unrealized PnL
            # on the current position has been adverse (≥ threshold
            # bps relative to position notional) for ≥ duration
            # seconds. Action: enter SOFT_FLATTENING (post-only-only,
            # no kill, no taker). Catches the small-position-slow-
            # drift regime that the absolute drawdown gate ($10) is
            # too loose to cover. Disabled while flatten_mode is
            # already true (an aggressive flatten is in progress).
            # Below-min-notional skip: if the residual position is
            # too small for a single closing order to clear the
            # bot's min_notional_usd floor, soft-flatten can't make
            # progress (pre-send risk check rejects + sets a per-side
            # passive_block). Don't even enter; a 191bps adverse on
            # $4 is 8 cents of unrealized loss, well below the hard-
            # kill line ($10), and normal quoting will work it off.
            # v1.5.291 — use the SAME max(venue, local) floor as the SF
            # worker's residual_below_min_notional exit (see
            # ``_run_soft_flatten_tick``) and the universal guard in
            # ``_enter_soft_flatten``. Pre-v1.5.291 this used the
            # VENUE min only, so a position above venue-min but below
            # the (larger) local min_quote_notional_usd passed this
            # entry gate, started an SF episode, then immediately
            # exited residual_below_min_notional — the no-op loop the
            # universal guard now also catches. This is defense-in-
            # depth: the universal guard is the primary chokepoint, but
            # keeping the two floors consistent avoids a misleading
            # "gate fired then bailed" episode in the events log.
            sp_min_ntn = max(
                float(
                    getattr(self._client.symbol_spec, "min_notional_usd", 0.0)
                    or 0.0
                ),
                float(getattr(self._settings, "min_quote_notional_usd", 0.0) or 0.0),
            )
            pos_ntn = float(self._state.position.position_notional or 0.0)
            below_min_ntn = sp_min_ntn > 0 and pos_ntn + 1e-9 < sp_min_ntn
            if (
                self._settings.position_drawdown_gate_enabled
                and not self._state.flatten_mode
                and not self._state.killed
                and not self._state.manual_pause
                and not below_min_ntn
            ):
                ev = position_drawdown_gate.evaluate(
                    self._settings, self._state
                )
                if ev.triggered:
                    self._enter_soft_flatten(ev)
                    return

            # v1.5.33 — take-profit (TP) arming check. Evaluated after
            # the position-drawdown gate (so SF wins when both would
            # fire — adverse uPnL trumps favorable). Mutually exclusive
            # with TP-already-active: ``_maybe_arm_take_profit`` checks
            # ``state.tp_active`` and returns False if so. On a true
            # return, the bot has just transitioned into TP mode; we
            # return so the executor runs on the next tick.
            if self._maybe_arm_take_profit():
                return

            if mid:
                self._vol.push_mid(mid)
                # v1.5.239 — feed the trend-aware EWMA-of-|return|
                # estimator from the same mid stream. Same dedup
                # (no-op on identical-mid push). Runs unconditionally;
                # the regime classifier's vol_slope criterion reads
                # this value only when REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE
                # is true.
                self._state.vol_abs_ewma.record_mid(
                    mid, self._clock.monotonic()
                )
            # v1.5.232 — sigma_and_bps() returns (None, 0.0) during
            # warm-up (< VOL_WINDOW_SAMPLES distinct mids). Capture
            # the warm-up state explicitly so we can publish a
            # genuine None to consumers that distinguish "warming up"
            # from "actually flat". Pre-v1.5.232 we wrote 0.0 to
            # state.vol_bps in both cases, which the dashboard and
            # the snapshot pipeline could not tell apart.
            _vol_sigma, _vol_bps_or_zero = self._vol.sigma_and_bps()
            vol_bps_or_none: Optional[float] = (
                _vol_bps_or_zero if _vol_sigma is not None else None
            )
            # v1.5.239 — mirror the EWMA value onto state for snapshot
            # consumers + the regime classifier wiring below. ``None``
            # during the estimator's 2-mid warm-up.
            vol_abs_ewma_bps_or_none: Optional[float] = (
                self._state.vol_abs_ewma.value_bps()
            )
            # Toxicity engine: keep the pre-v1.5.232 float contract.
            # `set_baseline_vol` already guards on `vol_bps > 0` (no
            # baseline seeded during warm-up), and `snapshot` falls
            # back to `max(current_vol_bps, 1e-6)`. Passing 0.0 here
            # is BEHAVIOUR-IDENTICAL to pre-v1.5.232 warm-up.
            self._tox.set_baseline_vol(_vol_bps_or_zero)

            with self._state._lock:
                fills = list(self._state.recent_fills)

            # v1.5.197 — pass ``now`` so the toxicity engine can
            # time-decay the recent_fills window. Eliminates the
            # defensive-deadlock pattern where the toxicity score
            # stayed elevated indefinitely after defenses suppressed
            # the fill flow that would refresh the rolling buffer.
            tox_snap = self._tox.snapshot(
                mid, _vol_bps_or_zero, fills, now=self._clock.now_utc(),
            )
            with self._state._lock:
                self._state.toxicity = tox_snap
                # v1.5.232 — store None during warm-up so the
                # frontend can show "insufficient data" instead of a
                # misleading 0.0 line. Consumers downstream are
                # already None-safe:
                #  * live_stats: _round_or_none() handles None
                #  * exposure_bar_emitter: _safe_float(getattr(...))
                #  * equity snapshot: explicit `is not None` guard
                #  * snapshot_dict: JSON null is correct
                self._state.vol_bps = vol_bps_or_none
                # v1.5.239 — also mirror the trend-aware EWMA value
                # onto state. None during the estimator's 2-mid
                # warm-up; same Optional[float] contract as vol_bps.
                self._state.vol_abs_ewma_bps = vol_abs_ewma_bps_or_none
                # v1.5.158 Option A — feed the vol-climbing gate's
                # rolling history. (mono_ts, vol_bps) tuple per tick.
                # v1.5.232: skip the append during warm-up — vol_climbing_widen
                # already has its own min_samples gate, so missing warm-up
                # samples just delay arming by ~10s on a slow market. This
                # keeps the deque's `float(v)` cast safe.
                if vol_bps_or_none is not None:
                    self._state.vol_bps_history.append(
                        (self._clock.monotonic(), float(vol_bps_or_none))
                    )
                self._state.operator_rolling_toxicity_markout_bps = (
                    tox_snap.avg_adverse_markout_bps
                )
                self._state.operator_rolling_one_sided_fill_ratio = (
                    tox_snap.one_sided_fill_ratio
                )
                # Vol-regime adjustment (BUGS/todo-009.md). Pure
                # function on the freshly-updated toxicity snapshot;
                # caller (here) writes the new spike-window deadline
                # back to state so the persistence carries across
                # ticks. Default OFF when ``VOL_SHRINK_COEFF=0`` —
                # the call is identity in that case.
                from app.vol_regime import (
                    compute_vol_regime_adjustment,
                    evaluate_vol_spike_favorable_exit,
                )

                _vra_now_mono = self._clock.monotonic()
                vra, new_until = compute_vol_regime_adjustment(
                    self._settings,
                    vol_ratio=getattr(tox_snap, "vol_spike_ratio", None),
                    now_mono=_vra_now_mono,
                    vol_spike_until_mono=self._state.vol_spike_until_mono,
                )
                # Phase 2K.8 — favorable-exit predicate. Clears the
                # spike latch early when vol_ratio has calmed past the
                # clear band for the configured dwell. Defensive
                # ``getattr`` for BotState shapes from before v1.4.160.
                _vse = evaluate_vol_spike_favorable_exit(
                    self._settings,
                    now_mono=_vra_now_mono,
                    vol_ratio=getattr(tox_snap, "vol_spike_ratio", None),
                    vol_spike_until_mono=new_until,
                    favorable_dwell_started_mono=getattr(
                        self._state,
                        "vol_spike_favorable_dwell_started_mono",
                        None,
                    ),
                    was_active_last_call=bool(
                        getattr(
                            self._state,
                            "vol_spike_was_active_last_call",
                            False,
                        )
                    ),
                )
                self._state.vol_spike_until_mono = _vse.new_until_mono
                self._state.vol_spike_favorable_dwell_started_mono = (
                    _vse.new_favorable_dwell_started_mono
                )
                self._state.vol_spike_was_active_last_call = (
                    _vse.new_was_active_last_call
                )
                if _vse.cleared_via == "favorable":
                    self._state.vol_spike_cleared_via_favorable_total += 1
                elif _vse.cleared_via == "ceiling":
                    self._state.vol_spike_cleared_via_ceiling_total += 1
                # If favorable-exit cleared the latch, re-derive the
                # adjustment so the same tick reflects the change
                # (in_spike_window, half_spread_bump_bps, tier_name).
                if _vse.cleared_via == "favorable":
                    vra, _ = compute_vol_regime_adjustment(
                        self._settings,
                        vol_ratio=getattr(
                            tox_snap, "vol_spike_ratio", None
                        ),
                        now_mono=_vra_now_mono,
                        vol_spike_until_mono=0.0,
                    )
                self._state.vol_regime_adjustment = vra
            # v1.4.170 Phase 4F — elevated-vol auto-pause. Macro
            # defense: when ``vol_spike_ratio`` stays above the arm
            # threshold for the arm dwell, force eligibility to
            # HOLD_ALL until vol normalises (clear dwell) or the MAX
            # ceiling fires. Reads + updates BotState. Defensive
            # ``getattr`` for Settings shapes from before v1.4.170.
            try:
                from app.vol_regime_auto_pause import (
                    evaluate_vol_regime_auto_pause,
                )

                _vrap_now_mono = self._clock.monotonic()
                _vrap_result = evaluate_vol_regime_auto_pause(
                    now_mono=_vrap_now_mono,
                    vol_spike_ratio=getattr(
                        tox_snap, "vol_spike_ratio", None
                    ),
                    currently_active=bool(
                        getattr(
                            self._state,
                            "vol_auto_pause_active",
                            False,
                        )
                    ),
                    arm_dwell_started_mono=getattr(
                        self._state,
                        "vol_auto_pause_arm_dwell_started_mono",
                        None,
                    ),
                    clear_dwell_started_mono=getattr(
                        self._state,
                        "vol_auto_pause_clear_dwell_started_mono",
                        None,
                    ),
                    active_since_mono=float(
                        getattr(
                            self._state,
                            "vol_auto_pause_active_since_mono",
                            0.0,
                        )
                        or 0.0
                    ),
                    arm_ratio=float(
                        getattr(
                            self._settings,
                            "vol_auto_pause_arm_ratio",
                            0.0,
                        )
                    ),
                    arm_dwell_seconds=float(
                        getattr(
                            self._settings,
                            "vol_auto_pause_arm_dwell_seconds",
                            60.0,
                        )
                    ),
                    clear_ratio=float(
                        getattr(
                            self._settings,
                            "vol_auto_pause_clear_ratio",
                            1.3,
                        )
                    ),
                    clear_dwell_seconds=float(
                        getattr(
                            self._settings,
                            "vol_auto_pause_clear_dwell_seconds",
                            120.0,
                        )
                    ),
                    max_pause_seconds=float(
                        getattr(
                            self._settings,
                            "vol_auto_pause_max_seconds",
                            1800.0,
                        )
                    ),
                )
                self._state.vol_auto_pause_active = (
                    _vrap_result.new_active
                )
                self._state.vol_auto_pause_arm_dwell_started_mono = (
                    _vrap_result.new_arm_dwell_started_mono
                )
                self._state.vol_auto_pause_clear_dwell_started_mono = (
                    _vrap_result.new_clear_dwell_started_mono
                )
                self._state.vol_auto_pause_active_since_mono = (
                    _vrap_result.new_active_since_mono
                )
                if _vrap_result.transition == "armed":
                    self._state.vol_auto_pause_armed_total += 1
                    if not getattr(
                        self._state,
                        "vol_auto_pause_first_arm_logged",
                        False,
                    ):
                        self._log_event(
                            EventSeverity.WARNING,
                            "vol_regime_auto_pause_armed",
                            (
                                "vol_regime_auto_pause ARMED — "
                                f"vol_spike_ratio="
                                f"{getattr(tox_snap, 'vol_spike_ratio', None)}; "
                                "bot will sit out (HOLD_ALL) until ratio "
                                "drops below clear threshold for clear dwell, "
                                "or MAX pause ceiling fires. First arm this "
                                "session; subsequent re-arms are silent."
                            ),
                            {
                                "vol_spike_ratio": getattr(
                                    tox_snap, "vol_spike_ratio", None
                                ),
                                "arm_ratio": float(
                                    self._settings.vol_auto_pause_arm_ratio
                                ),
                                "arm_dwell_seconds": float(
                                    self._settings.vol_auto_pause_arm_dwell_seconds
                                ),
                            },
                        )
                        self._state.vol_auto_pause_first_arm_logged = True
                elif _vrap_result.transition == "cleared_favorable":
                    self._state.vol_auto_pause_cleared_via_favorable_total += 1
                    self._log_event(
                        EventSeverity.INFO,
                        "vol_regime_auto_pause_cleared",
                        "vol_regime_auto_pause CLEARED via favorable-exit "
                        "(vol returned to baseline + dwell)",
                        {
                            "vol_spike_ratio": getattr(
                                tox_snap, "vol_spike_ratio", None
                            ),
                            "clear_ratio": float(
                                self._settings.vol_auto_pause_clear_ratio
                            ),
                            "via": "favorable",
                        },
                    )
                elif _vrap_result.transition == "cleared_ceiling":
                    self._state.vol_auto_pause_cleared_via_ceiling_total += 1
                    self._log_event(
                        EventSeverity.WARNING,
                        "vol_regime_auto_pause_cleared",
                        "vol_regime_auto_pause CLEARED via MAX-ceiling "
                        "(vol stayed elevated past safety timeout)",
                        {
                            "vol_spike_ratio": getattr(
                                tox_snap, "vol_spike_ratio", None
                            ),
                            "max_pause_seconds": float(
                                self._settings.vol_auto_pause_max_seconds
                            ),
                            "via": "ceiling",
                        },
                    )
            except Exception:
                # Auto-pause evaluation failures must NEVER block the
                # trading loop — log + continue.
                logger.exception(
                    "vol_regime_auto_pause_eval_failed"
                )
            # Arm the per-side adverse pause from the fresh toxicity snapshot.
            # This is the localized feedback loop on top of the global hard/soft
            # triggers — it fires on milder single-side adverse patterns the
            # global path misses (observed in ``snap_20260417_183547``).
            try:
                self._exec._maybe_arm_adverse_side_pause(tox_snap)
            except Exception:
                logger.exception("maybe_arm_adverse_side_pause_failed")

            if mid is None:
                t_hot = time.perf_counter()
                # Preamble is everything before the hot-path compute timer.
                lat["latency_tick_preamble_ms"] = (t_hot - t_tick_start) * 1000.0
                eq_nm = None
                with self._state._lock:
                    acct_nm = self._state.account
                    if acct_nm and acct_nm.equity_usd is not None:
                        eq_nm = acct_nm.equity_usd
                pnl_snap_nm = self._pnl.build_snapshot(pos, eq_nm)
                with self._state._lock:
                    self._state.pnl = pnl_snap_nm
                    reconcile_ap = self._state.reconcile_auto_pause
                risk_nm = evaluate_risk(
                    self._settings,
                    bot_status=self._state.bot_status,
                    manual_pause=self._state.manual_pause,
                    killed=self._state.killed,
                    flatten_mode=self._state.flatten_mode,
                    market=m,
                    position_qty=pos.position_qty,
                    position_notional=pos.position_notional,
                    open_order_count=self._state.open_order_count(),
                    pnl=pnl_snap_nm,
                    toxicity=tox_snap,
                    execution_errors=self._state.execution_errors_window_snapshot(
                        self._settings.execution_errors_window_seconds
                    )["windowed"],
                    desync=self._state.order_desync,
                    desync_phase=self._state.desync_phase,
                    desync_quarantine_remaining=self._state.desync_quarantine_remaining,
                    trades_last_minute=self._state.trades_last_minute(),
                    reconcile_auto_pause=reconcile_ap,
                    **self._public_ws_risk_kwargs(),
                )
                lat["latency_hot_path_local_compute_ms"] = (
                    time.perf_counter() - t_hot
                ) * 1000.0
                if risk_nm.action == RiskAction.KILL:
                    self.kill(
                        ",".join(risk_nm.reasons),
                        self._build_kill_payload(risk_nm.reasons),
                    )
                    return
                if risk_nm.action == RiskAction.FLATTEN:
                    self.flatten(blocking=True)
                    return
                if risk_nm.action == RiskAction.SOFT_FLATTEN:
                    # Aggressive post-only exit via the soft-flatten
                    # worker — used by the toxicity-trigger path so
                    # we don't pay taker fees on every adverse-flow
                    # detection. Skip phase 1 (start at phase 2 = far
                    # touch − 1 tick) and apply the configured taker
                    # fallback (default operator-set ceiling) so a
                    # truly escaping price still gets capped exit
                    # cost. See ``risk.py:210`` for the trigger.
                    self._enter_soft_flatten(
                        None,
                        force_phase=2,
                        taker_fallback_ticks=int(
                            getattr(
                                self._settings,
                                "soft_flatten_taker_fallback_ticks",
                                0,
                            )
                            or 0
                        )
                        or None,
                        trigger_reason="toxicity_hard_trigger",
                        log_message_override=(
                            "toxicity hard-trigger with inventory — "
                            "entering aggressive post-only soft-flatten "
                            "(phase 2 from t=0, taker fallback if drift "
                            "exceeds threshold)"
                        ),
                        log_payload_override={
                            "reasons": list(risk_nm.reasons),
                        },
                    )
                    return
                if self._settings.trading_enabled and self._state.bot_status in (
                    BotStatus.RUNNING,
                    BotStatus.STARTING,
                    BotStatus.RECOVERING_MARKET_DATA,
                ):
                    if (
                        risk_nm.action != RiskAction.NO_QUOTE
                        or self._should_cancel_resting_on_no_quote_bound(risk_nm.reasons)
                    ):
                        self._exec.cancel_resting_for_risk(risk_nm.action)
                    self._exec.maybe_sync_open_orders()
                lat["latency_reconcile_rest_ms"] = self._exec.last_tick_reconcile_rest_ms
                self._maybe_promote_starting_to_running(healthy)
                self._persist_snapshots()
                self._maybe_save_persistent_runtime_state()
                return

            t_hot = time.perf_counter()
            # Preamble is everything before the hot-path compute timer.
            lat["latency_tick_preamble_ms"] = (t_hot - t_tick_start) * 1000.0
            now_mono = self._clock.monotonic()
            overlay_cfg = float(self._settings.adaptive_spread_adverse_overlay_half_spread_bps)
            qq_overlay_bps = float(self._settings.quote_quality_overlay_half_spread_bps)
            with self._state._lock:
                if now_mono >= self._state.adaptive_spread_widen_until_mono:
                    self._state.quote_quality_widen_latched = False
                fills_qq = list(self._state.recent_fills)
                qq_sig = (
                    qq_overlay_bps > 0
                    and self._state.quote_quality.spread_widen_signal(
                        fills_for_markout=fills_qq,
                        markout_window=int(self._settings.runtime_toxicity_fill_window),
                        min_quote_cycles=int(self._settings.quote_quality_widen_min_cycles),
                    )
                )
                # Invariants for arming (mirrors ``_maybe_arm_adverse_side_pause``):
                #   1. No re-extend while active — let the deadline run down.
                #   2. No stale re-arm — require at least one new fill since the
                #      previous arming. Without this a single adverse signal
                #      pins the widen forever: the widen itself prevents new
                #      fills, so the signal never clears. See
                #      ``tmp/snap_20260418_094415``: 4 fills in minute 1 armed
                #      the widen on ``one_sided_fill_ratio=0.75``, 17 minutes of
                #      invisible quoting followed.
                # Use ``session_fill_count`` (monotonic session counter), NOT
                # ``len(recent_fills)`` — the latter caps at the deque maxlen
                # (200) and would break this gate on long sessions.
                widen_active = now_mono + 1e-9 < self._state.adaptive_spread_widen_until_mono
                n_fills_now = int(self._state.session_fill_count)
                prev_arm_n = int(self._state.adaptive_spread_widen_arm_n_fills)
                can_arm_fresh = (not widen_active) and (n_fills_now > prev_arm_n)
                want_arm_adverse = overlay_cfg > 0 and adverse_spread_widen_arm(self._settings, tox_snap)
                want_arm_qq = bool(qq_sig)
                # v1.5.26 Phase 2D -- residual-decay trigger. The
                # tracker observes resolved-5s fills via the
                # markout-resolution hook; the predicate stays True
                # as long as the rolling mean is below threshold AND
                # the below-threshold dwell exceeds the configured
                # window. Idempotent within an episode -- the
                # ``adaptive_spread_widen_arm_n_fills`` re-arm guard
                # below ensures we only fire on FRESH fill activity.
                want_arm_residual_decay = False
                if overlay_cfg > 0:
                    rdt = getattr(self._state, "residual_decay_tracker", None)
                    if rdt is not None and rdt.enabled():
                        try:
                            want_arm_residual_decay = rdt.is_armed(now_mono)
                        except Exception:  # noqa: BLE001
                            want_arm_residual_decay = False
                # v1.4.102 — slow-trend overlay arming. When the
                # slow_trend gate is firing (sustained directional
                # drift), arm the adverse overlay so the half-spread
                # gets an EXTRA widening kick on TOP of the slow_trend
                # gate's own contribution. Defence-in-depth: the gate
                # widens one side; this overlay widens BOTH sides so
                # the bot is uniformly less aggressive across a
                # trending regime.
                want_arm_slow_trend = False
                if (
                    overlay_cfg > 0
                    and bool(self._settings.slow_trend_gate_enabled)
                    and self._state.market is not None
                    and self._state.market.mid_price is not None
                    and self._state.market.mid_price > 0
                ):
                    try:
                        from app.slow_trend_gate import (
                            evaluate_slow_trend_gate,
                        )
                        _st_override, _, _ = evaluate_slow_trend_gate(
                            samples_long=self._state.mid_price_samples_long_snapshot(),
                            now_mono=now_mono,
                            mid_now=float(self._state.market.mid_price),
                            enabled=True,
                            window_seconds=float(
                                self._settings.slow_trend_window_seconds
                            ),
                            threshold_bps=float(
                                self._settings.slow_trend_threshold_bps
                            ),
                            min_samples=int(
                                self._settings.slow_trend_min_samples
                            ),
                            anchor_fraction=float(
                                self._settings.slow_trend_anchor_fraction
                            ),
                        )
                        want_arm_slow_trend = _st_override is not None
                    except Exception:
                        # Don't let a slow-trend lookup crash the
                        # adaptive_widen flow.
                        want_arm_slow_trend = False
                # ---- v1.5.157 — position-aware favorable exit ----
                # Per CLAUDE.md Rule 0c: timer-based gates must have a
                # signal-driven exit. The existing Phase 2K.5 markout-
                # based exit only fires when the ARM-TIME reason signal
                # clears (markout_adverse / quote_quality / slow_trend).
                # During an adverse-trend regime those signals stay
                # adverse for the full window → cleared via ceiling 88
                # times vs 10 favorable in the v1.5.154-260526-074029
                # snapshot. This second clearance path fires when the
                # bot has significant adverse inventory AND drift is
                # moving favorably for that inventory (good time to
                # unwind). Same predicate as mae_gate's v1.5.155 fix.
                if (
                    widen_active
                    and bool(
                        getattr(
                            self._settings,
                            "adaptive_spread_widen_position_favorable_exit_enabled",
                            True,
                        )
                    )
                ):
                    _aw_pos_qty = float(
                        getattr(self._state.position, "position_qty", 0.0)
                        or 0.0
                    )
                    _aw_inv_thresh = float(
                        getattr(
                            self._settings,
                            "adaptive_spread_widen_position_favorable_inventory_threshold",
                            1.0,
                        )
                    )
                    _aw_drift_thresh = float(
                        getattr(
                            self._settings,
                            "adaptive_spread_widen_position_favorable_drift_threshold_bps",
                            5.0,
                        )
                    )
                    # v1.5.191 BUG-033 — flat-position fast-clear. The
                    # original (v1.5.157) predicate required ``|pos|
                    # >= 1 AND sign(pos) × drift >= 5 bps`` — perfectly
                    # reasonable when the gate arms post-fill (mae_gate
                    # pattern), but adaptive_widen also arms on
                    # ``quote_quality`` / ``slow_trend`` signals that
                    # are orthogonal to inventory. Snapshot v1.5.187-
                    # 260527-130257 was 73 % flat-position; the
                    # ``|pos| >= 1`` precondition failed on most ticks
                    # during widen windows → 0/10 position-favorable
                    # clears, 9/10 ceiling exits (CLAUDE.md Rule 0c
                    # violation: gate became a pure timer).
                    #
                    # When the bot is currently flat, there's no
                    # inventory to defend, so the widen is pure fill-
                    # rate cost. Clear immediately. The drift-aligned
                    # branch below still handles the non-flat case.
                    if abs(_aw_pos_qty) < 1e-9:
                        self._state.adaptive_spread_widen_until_mono = 0.0
                        self._state.adaptive_spread_widen_favorable_dwell_started_mono = None
                        self._state.quote_quality_widen_latched = False
                        self._state.adaptive_spread_widen_cleared_via_position_favorable_total += 1
                        self._state.adaptive_spread_widen_was_active_last_tick = False
                        widen_active = False
                    elif abs(_aw_pos_qty) >= _aw_inv_thresh:
                        try:
                            _aw_drift = self._select_trend_drift_signal(eff_q)
                        except Exception:
                            _aw_drift = None
                        if (
                            _aw_drift is not None
                            and math.isfinite(float(_aw_drift))
                        ):
                            _aw_sign_pos = (
                                math.copysign(1.0, _aw_pos_qty)
                                if _aw_pos_qty != 0.0 else 0.0
                            )
                            if _aw_sign_pos * float(_aw_drift) >= _aw_drift_thresh:
                                self._state.adaptive_spread_widen_until_mono = 0.0
                                self._state.adaptive_spread_widen_favorable_dwell_started_mono = None
                                self._state.quote_quality_widen_latched = False
                                self._state.adaptive_spread_widen_cleared_via_position_favorable_total += 1
                                self._state.adaptive_spread_widen_was_active_last_tick = False
                                widen_active = False
                # ---- v1.4.155 Phase 2K.5 — favorable-exit predicate ----
                # When the cooldown is active and the ARM-TIME signal
                # has cleared (per ``adaptive_spread_widen_signal_cleared``
                # for the recorded ``reason``) continuously for the
                # configured dwell, drop the deadline early. Mirrors
                # 2K.3 / 2K.4's hysteresis+dwell architecture.
                if (
                    widen_active
                    and bool(
                        getattr(
                            self._settings,
                            "adaptive_spread_widen_favorable_exit_enabled",
                            True,
                        )
                    )
                ):
                    fav_dwell_s = float(
                        getattr(
                            self._settings,
                            "adaptive_spread_widen_favorable_exit_dwell_seconds",
                            10.0,
                        )
                    )
                    if fav_dwell_s > 0.0:
                        from app.quoting import (
                            adaptive_spread_widen_signal_cleared as _aw_sig_cleared,
                        )
                        reason = self._state.adaptive_spread_widen_reason
                        if reason is not None:
                            sig_cleared = _aw_sig_cleared(
                                self._settings,
                                reason,
                                tox_snap,
                                quote_quality_signal=bool(qq_sig),
                                slow_trend_signal=bool(want_arm_slow_trend),
                            )
                            dwell_started = (
                                self._state.adaptive_spread_widen_favorable_dwell_started_mono
                            )
                            if sig_cleared:
                                if dwell_started is None:
                                    self._state.adaptive_spread_widen_favorable_dwell_started_mono = now_mono
                                elif now_mono - dwell_started >= fav_dwell_s:
                                    # Favorable exit fires.
                                    self._state.adaptive_spread_widen_until_mono = 0.0
                                    self._state.adaptive_spread_widen_favorable_dwell_started_mono = None
                                    self._state.quote_quality_widen_latched = False
                                    self._state.adaptive_spread_widen_cleared_via_favorable_total += 1
                                    self._state.adaptive_spread_widen_was_active_last_tick = False
                                    widen_active = False
                            else:
                                # Re-flare — reset the dwell timer.
                                self._state.adaptive_spread_widen_favorable_dwell_started_mono = None
                # Ceiling attribution: if the cooldown was active LAST
                # tick but the deadline has now lapsed without
                # favorable-exit firing, attribute the clearing to
                # the MAX-cooldown ceiling. Edge-triggered — single
                # increment per cooldown cycle.
                if (
                    self._state.adaptive_spread_widen_was_active_last_tick
                    and not widen_active
                    and self._state.adaptive_spread_widen_until_mono > 0.0
                ):
                    self._state.adaptive_spread_widen_cleared_via_ceiling_total += 1
                    self._state.adaptive_spread_widen_until_mono = 0.0
                    self._state.adaptive_spread_widen_favorable_dwell_started_mono = None
                    self._state.adaptive_spread_widen_was_active_last_tick = False
                # v1.4.155 — after favorable-exit / ceiling clearing,
                # ``widen_active`` may have flipped to False mid-tick.
                # Re-evaluate ``can_arm_fresh`` so re-arming on a
                # DIFFERENT signal can happen this tick rather than
                # waiting another full quote cycle.
                can_arm_fresh = (not widen_active) and (
                    n_fills_now > prev_arm_n
                )
                if can_arm_fresh and (
                    want_arm_adverse
                    or want_arm_qq
                    or want_arm_slow_trend
                    or want_arm_residual_decay
                ):
                    aw_was_active = (
                        now_mono < self._state.adaptive_spread_widen_until_mono
                    )
                    self._state.adaptive_spread_widen_until_mono = (
                        now_mono + float(self._settings.toxicity_cooldown_seconds)
                    )
                    self._state.adaptive_spread_widen_arm_n_fills = n_fills_now
                    # v1.4.155 — fresh arm cycle: reset dwell + edge
                    # detector so the next clearing event attributes
                    # to the right exit path.
                    self._state.adaptive_spread_widen_favorable_dwell_started_mono = None
                    self._state.adaptive_spread_widen_was_active_last_tick = True
                    # Tag the trigger reason so the dashboard can show
                    # *why* the spread widened (not just that it did).
                    # Priority: hard > soft > markout > one-sided > QQ.
                    # Mirrors the predicate order in ``adverse_spread_widen_arm``.
                    if want_arm_adverse and tox_snap.hard_trigger:
                        self._state.adaptive_spread_widen_reason = "toxicity_hard"
                    elif want_arm_adverse and tox_snap.soft_trigger:
                        self._state.adaptive_spread_widen_reason = "toxicity_soft"
                    elif want_arm_adverse and (
                        tox_snap.adverse_uses_delayed_markouts
                        and tox_snap.avg_adverse_markout_bps + 1e-12
                        <= -float(self._settings.toxicity_markout_soft_bps)
                    ):
                        self._state.adaptive_spread_widen_reason = "markout_adverse"
                    elif want_arm_adverse and (
                        tox_snap.one_sided_fill_ratio + 1e-12
                        >= float(self._settings.toxicity_one_sided_fill_ratio)
                    ):
                        self._state.adaptive_spread_widen_reason = "one_sided_ratio"
                    elif want_arm_qq:
                        self._state.adaptive_spread_widen_reason = "quote_quality"
                    elif want_arm_slow_trend:
                        # v1.4.102 — overlay armed by slow_trend gate
                        # signal (sustained directional drift over the
                        # multi-minute window). The gate itself
                        # contributes asymmetric widening via the
                        # SpreadComposition path; THIS overlay layers
                        # symmetric widening on top so the entire
                        # quote backs off during a trending regime.
                        self._state.adaptive_spread_widen_reason = "slow_trend"
                    elif want_arm_residual_decay:
                        # v1.5.26 Phase 2D -- composite-signal arm:
                        # closed_pnl + rebate - markout_5s adverse,
                        # rolling mean below threshold for dwell.
                        # Distinct from markout_adverse / one_sided
                        # / quote_quality because those each read a
                        # single signal; residual_decay catches cases
                        # where each individual signal looks fine
                        # but the bot is bleeding in aggregate.
                        # Clearing path: ceiling-only via 2K.5's
                        # adaptive_widen_signal_cleared (which
                        # returns False for unknown reasons,
                        # including "residual_decay").
                        self._state.adaptive_spread_widen_reason = "residual_decay"
                        try:
                            self._state.residual_decay_tracker.consume_arm(
                                now_mono
                            )
                        except Exception:  # noqa: BLE001
                            pass
                    if want_arm_qq:
                        self._state.quote_quality_widen_latched = True
                    if not aw_was_active:
                        self._log_event(
                            EventSeverity.WARNING,
                            "adaptive_widen_armed",
                            (
                                f"adaptive spread-widen armed "
                                f"(reason={self._state.adaptive_spread_widen_reason}) "
                                f"for {self._settings.toxicity_cooldown_seconds:.0f}s"
                            ),
                            {
                                "reason": self._state.adaptive_spread_widen_reason,
                                "cooldown_seconds": float(
                                    self._settings.toxicity_cooldown_seconds
                                ),
                                "toxicity_score": float(
                                    getattr(tox_snap, "score", 0.0) or 0.0
                                ),
                            },
                        )
                elif qq_sig and widen_active:
                    # Already active: don't extend, but keep the QQ latch sticky
                    # for overlay selection on subsequent ticks.
                    self._state.quote_quality_widen_latched = True
                spread_floor_overlay = 0.0
                if now_mono < self._state.adaptive_spread_widen_until_mono:
                    if overlay_cfg > 0:
                        spread_floor_overlay = max(spread_floor_overlay, overlay_cfg)
                    if qq_overlay_bps > 0 and self._state.quote_quality_widen_latched:
                        spread_floor_overlay = max(spread_floor_overlay, qq_overlay_bps)

            self._state.record_mid_price_sample(mid, now_mono)
            gap_med, gap_p95, gap_last, gap_n = (
                self._state.market_data_gap_tracker.recent_gap_stats_for_gate()
            )
            eff_stal = effective_staleness_ms_from_exchange_ts(
                m.ts_exchange_ms if m else None
            )
            sec_bbo = self._state.seconds_since_last_public_bbo(now_mono)
            with self._state._lock:
                last_eff = self._state.quote_eligibility_last_effective
            raw_q = compute_quote_eligibility(
                self._settings,
                # Global gate: desync only. Per-side stalls go through ``uncertain_sides``
                # so a stuck CANCEL_PENDING on one side doesn't silence the other.
                order_state_uncertainty=self._exec.global_order_state_uncertainty(),
                uncertain_sides=frozenset(self._exec.uncertain_sides()),
                mid_now=float(mid),
                now_mono=now_mono,
                mid_samples=self._state.mid_price_samples_snapshot(),
                seconds_since_public_bbo=sec_bbo,
                gap_median_ms=gap_med,
                gap_p95_ms=gap_p95,
                effective_staleness_ms=eff_stal,
                # vol_bps drives the vol-scaled drift/jump thresholds so the filter
                # adapts to the symbol's observed volatility. When all per-horizon
                # ``*_VOL_MULTIPLIER`` settings are 0 (default), this is ignored and
                # the absolute bps thresholds apply — preserves pre-fix behaviour.
                # v1.5.232 — pass the None-during-warm-up flavour: the gate's
                # vol-scaled thresholds correctly fall back to absolute-bps mode
                # when vol_bps is None (estimator not warmed up).
                vol_bps=(
                    float(vol_bps_or_none)
                    if vol_bps_or_none is not None
                    else None
                ),
                # Cold-start / post-outlier safety nets on the p95 hold:
                # - ``gap_last_ms`` enables the "feed is alive NOW" override fallback
                #   when the ring-buffer median is polluted but recent samples are fine.
                # - ``gap_sample_count`` lets the gate skip p95-hold while the ring
                #   hasn't warmed up (small-sample outliers are unreliable).
                gap_last_ms=gap_last,
                gap_sample_count=gap_n,
                # Multi-minute mid history for the long-window drift
                # gate (BUGS/bug-002.md). Independent from the
                # sub-second ``mid_samples`` deque above; populated
                # by the same recording path with longer retention.
                mid_samples_long=self._state.mid_price_samples_long_snapshot(),
                # 2026-05-12 codex-#5: current position so the one-
                # sided freshness fallback prefers the inventory-
                # reducing side instead of a static config preference.
                position_qty=float(pos.position_qty)
                if pos is not None and pos.position_qty is not None
                else None,
            )
            # v1.4.165 Phase 4E — publish the 500 ms mid-return
            # kinematic to BotState so ``OrderManager`` can read it
            # from the target-venue fast-move cancel trigger path.
            # Exchange-agnostic; the trigger logic lives in
            # ``app/fast_move_cancel.py``.
            try:
                self._state.last_mid_return_500ms_bps = (
                    float(raw_q.mid_return_500ms_bps)
                    if raw_q.mid_return_500ms_bps is not None
                    else None
                )
            except (TypeError, ValueError):
                self._state.last_mid_return_500ms_bps = None
            prev_ru = float(self._state.quote_elig_recovery_until_mono)
            prev_rf = self._state.quote_elig_recovery_floor
            ru, rf = maybe_arm_recovery_cooldown(
                self._settings,
                last_effective=last_eff,
                new_raw=raw_q.eligibility,
                now_mono=now_mono,
                recovery_until_mono=self._state.quote_elig_recovery_until_mono,
                recovery_floor=self._state.quote_elig_recovery_floor,
            )
            self._state.quote_elig_recovery_until_mono = ru
            self._state.quote_elig_recovery_floor = rf
            self._state.quote_elig_recovery_remaining_ms = (
                max(0.0, (ru - now_mono) * 1000.0) if ru > 0.0 else 0.0
            )
            newly_armed = (ru > prev_ru + 1e-12) or (rf != prev_rf)
            if newly_armed:
                log_extra(
                    logger,
                    logging.INFO,
                    "quote_eligibility_recovery_armed",
                    {
                        "event": "quote_eligibility_recovery_armed",
                        "raw": raw_q.eligibility.value,
                        "raw_reason": raw_q.reason[:400],
                        # counter_tags on the raw result identify which specific rule
                        # (gap, staleness, drift, jump, order-state) produced this dip
                        # — the transition log re-emits them for the effective value
                        # but the arming snapshot captures the raw trigger directly.
                        "counter_tags": list(raw_q.counter_tags),
                        "last_effective": last_eff.value,
                        "recovery_floor": rf.value if rf is not None else None,
                        "recovery_remaining_ms": self._state.quote_elig_recovery_remaining_ms,
                    },
                )
            eff_q = apply_recovery_cooldown(
                raw_q,
                now_mono=now_mono,
                recovery_until_mono=self._state.quote_elig_recovery_until_mono,
                recovery_floor=self._state.quote_elig_recovery_floor,
            )
            # 1.4.12 cutover: regime-only eligibility clamps are
            # demoted to widening contributions (composition carries
            # them). Detect "the only thing clamping us is freshness
            # one-sided and/or recovery cooldown" and promote
            # eligibility back to QUOTE_BOTH so the bot keeps both
            # sides quoting (at wider spreads from the composition).
            #
            # raw_q.reason carries the freshness signature; eff_q.
            # in_cooldown carries the recovery state. Both stay set
            # so the widening_bps helpers in quote_eligibility.py
            # can still read them — only the eligibility itself is
            # unclamped.
            #
            # Safety / system clamps (stale_data_warn, stale_data_kill,
            # drift_*, jump_*, order_state_uncertainty,
            # side_uncertainty, order_desync) are NOT in the strip
            # list — they keep their HOLD_ALL / one-sided semantics.
            if eff_q.eligibility != QuoteEligibility.QUOTE_BOTH:
                _reason_lower = (eff_q.reason or "").lower()
                _safety_sigs = (
                    "stale_data_warn", "stale_data_kill",
                    "drift_100ms", "drift_250ms", "drift_500ms",
                    "jump_100ms", "jump_250ms", "jump_500ms",
                    "order_state_uncertainty",
                    "per_side_uncertainty",
                    "order_desync", "desync_",
                )
                _safety_clamp = any(
                    sig in _reason_lower for sig in _safety_sigs
                )
                if not _safety_clamp:
                    # Only regime causes (freshness_one_sided +/-
                    # recovery_cooldown). Demote to BOTH; widening
                    # carries the spread response.
                    eff_q = replace(
                        eff_q,
                        eligibility=QuoteEligibility.QUOTE_BOTH,
                    )
            # If the clamp has expired, clear recovery state so the next tick's arming logic
            # compares against the true effective cap (not a stale floor).
            if (
                not eff_q.in_cooldown
                and self._state.quote_elig_recovery_until_mono > 0.0
                and now_mono + 1e-12 >= self._state.quote_elig_recovery_until_mono
            ):
                self._state.quote_elig_recovery_until_mono = 0.0
                self._state.quote_elig_recovery_floor = None
                self._state.quote_elig_recovery_remaining_ms = 0.0
            # 1.4.7 Phase 0 telemetry: freshness one-sided trigger and
            # recovery_cooldown clamp are the two most-impactful
            # "gates" by share of session time, so they need their
            # own fire-stats entries alongside the explicit gate
            # modules above. The signal is encoded in the reason
            # string (raw_q for the freshness trigger; eff_q for the
            # cooldown clamp since it's applied post-arming).
            self._state.record_gate_firing(
                "freshness_one_sided",
                firing_now="freshness_one_sided" in (raw_q.reason or ""),
                now_mono=now_mono,
            )
            self._state.record_gate_firing(
                "recovery_cooldown",
                firing_now=bool(eff_q.in_cooldown),
                now_mono=now_mono,
            )

            # === Analysis-day 2026-05-10 behavioural gates =================
            # Three additional eligibility caps stacked on top of ``eff_q``:
            #   - tiered session-PnL drawdown ladder (1.2.1) — KILL on tier-4,
            #     HOLD_ALL on PAUSE_*, spread floor on WIDEN / RESUME_TESTING
            #   - post-swing PnL cooldown (#2)
            #   - inventory-aligned-with-momentum gate (#3)
            # All narrow the existing eligibility; none widen it.
            #
            # Order matters: session-drawdown runs first because tier-4 kills
            # the bot (no point computing momentum gate on a dead bot) and
            # tier-1 widen feeds an extra overlay to the spread floor below.
            eff_q, sd_overlay = self._apply_session_drawdown_gate(
                eff_q,
                now_mono=now_mono,
                now_iso=self._clock.now_utc().isoformat(),
            )
            if sd_overlay > 0.0:
                spread_floor_overlay = max(spread_floor_overlay, sd_overlay)
            # If the session-drawdown gate killed the bot, status is now
            # KILLED and the rest of this tick is wasted work — bail.
            if self._state.bot_status == BotStatus.KILLED:
                return

            # v1.5.202 — SF-event-count fatigue ladder. Orthogonal
            # to session_drawdown above (PnL-based) — this keys on
            # SF episode count in a rolling window so storm-cluster
            # patterns trigger a brake even when no single SF hit
            # the PnL drawdown tier. Composes multiplicatively with
            # the spread floor on tier-1 (WIDEN); forces HOLD_ALL on
            # PAUSE_*; kills on tier-4.
            eff_q, sf_widen_mult = self._apply_sf_fatigue_gate(
                eff_q, now_mono=now_mono,
            )
            if sf_widen_mult > 1.0 + 1e-12 and spread_floor_overlay > 0.0:
                spread_floor_overlay = spread_floor_overlay * sf_widen_mult
            elif sf_widen_mult > 1.0 + 1e-12:
                # When there's no existing overlay to multiply, apply
                # the WIDEN multiplier on top of the
                # adverse-overlay knob (same knob session_drawdown
                # uses) so WIDEN has a teeth even with a clean floor.
                base_overlay = float(
                    getattr(
                        self._settings,
                        "adaptive_spread_adverse_overlay_half_spread_bps",
                        0.0,
                    )
                    or 0.0
                )
                spread_floor_overlay = max(
                    spread_floor_overlay,
                    base_overlay * sf_widen_mult,
                )
            if self._state.bot_status == BotStatus.KILLED:
                return

            eff_q = self._apply_post_swing_and_momentum_gates(
                eff_q, now_mono=now_mono, mid_now=mid, raw_q=raw_q
            )

            # 1.2.2 regime gates: vol × trend conjunction (2a),
            # basis_regime IC absent (2b), microprice OB-imbalance
            # adverse-selection (1a). Stacked on top of the post-
            # swing / momentum gates above; same intersect-only
            # semantics — they only narrow, never widen.
            eff_q = self._apply_regime_gates(
                eff_q, now_mono=now_mono, raw_q=raw_q
            )

            # v1.4.228 — Phase 4G.13 structural-bias auto-throttle.
            # Backstop for the forward classifier: if the session's
            # inventory_exec_bias suppression ratio crossed the
            # threshold (≥ 5× default), force REDUCING-only on the
            # lean direction. Intersects with everything above.
            # Composes naturally with shock_gate's reducing-side
            # lock — both narrow eligibility in the same direction
            # under structural stress.
            eff_q = self._apply_structural_bias_throttle_gate(eff_q)
            # v1.5.2 Phase 4D.3 — post-SF cooldown gate. After SF
            # completion, suppress the side that would re-add to the
            # pre-SF direction for ``POST_SF_COOLDOWN_SECONDS``.
            # Composes with everything above; never widens.
            eff_q = self._apply_post_sf_cooldown_gate(eff_q)

            self._state.bump_quote_eligibility_counters(
                eff_q.eligibility, eff_q.counter_tags
            )
            self._state.set_quote_eligibility_snapshot_dict(asdict(eff_q.to_snapshot()))
            # 2026-05-13 regime-observability bugfix: push gate flags
            # into state for the exposure-bar emitter to sample.
            # Best-effort — each lookup catches exceptions
            # independently so a transient attribute miss doesn't
            # block the tick. Lock-free reads; the bot writes, the
            # emitter reads.
            #
            # 2026-05-13 bugfix-2 (after 4h verification snapshot):
            # corrected method/attribute paths. The previous attempt
            # called ``self._exec._post_fill_cooldown_active(side)``
            # (doesn't exist) and ``self._exec._at_touch_adverse_pause``
            # (lives on state, not exec). All four gate flags were
            # silently None in v1.3.2's exposure_bars.
            try:
                gate_flags: dict[str, Any] = {}
                try:
                    gate_flags["post_fill_cooldown_bid"] = (
                        self._compute_post_fill_cooldown_remaining_ms(
                            Side.BUY
                        ) > 0.0
                    )
                except Exception:
                    gate_flags["post_fill_cooldown_bid"] = None
                try:
                    gate_flags["post_fill_cooldown_ask"] = (
                        self._compute_post_fill_cooldown_remaining_ms(
                            Side.SELL
                        ) > 0.0
                    )
                except Exception:
                    gate_flags["post_fill_cooldown_ask"] = None
                try:
                    atap = getattr(
                        self._state, "at_touch_adverse_pause", None
                    )
                    gate_flags["at_touch_adverse_pause_bid"] = (
                        bool(atap.is_paused(Side.BUY))
                        if atap is not None
                        else None
                    )
                    gate_flags["at_touch_adverse_pause_ask"] = (
                        bool(atap.is_paused(Side.SELL))
                        if atap is not None
                        else None
                    )
                except Exception:
                    gate_flags["at_touch_adverse_pause_bid"] = None
                    gate_flags["at_touch_adverse_pause_ask"] = None
                self._state.set_observability_gate_flags(gate_flags)
                # v1.4.116 Phase 1E.3.f — feed the firing-rate tracker
                # with edge transitions (False → True) on the gate
                # flags. The tracker records one fire per edge so the
                # Detectors card section 5 can show "post_fill_cooldown
                # fired 12 ×/min" instead of just ON/OFF.
                try:
                    from app.feature_firing_rate import (
                        detect_edge_transitions,
                    )
                    self._state._prior_observability_gate_flags = (
                        detect_edge_transitions(
                            tracker=self._state.feature_firing_rates,
                            now_mono=now_mono,
                            prior_flags=self._state._prior_observability_gate_flags,
                            current_flags=gate_flags,
                        )
                    )
                except Exception:
                    # Tracker is observability-only; never let it
                    # interrupt the tick path.
                    pass
            except Exception:
                # Belt-and-suspenders: if the entire setup fails for
                # some reason, don't break the tick.
                pass
            with self._state._lock:
                prev_log = self._state.quote_elig_last_logged
                cur_e = eff_q.eligibility.value
                if prev_log != cur_e:
                    self._state.quote_elig_last_logged = cur_e
                    do_elig_log = True
                else:
                    do_elig_log = False
            if do_elig_log:
                elig_payload = self._build_quote_eligibility_diag(raw_q, eff_q)
                elig_payload.update(
                    {
                        "event": "quote_eligibility_transition",
                        "from": prev_log,
                        "to": cur_e,
                        "reason": eff_q.reason[:1500],
                    }
                )
                log_extra(
                    logger,
                    logging.INFO,
                    "quote_eligibility_transition",
                    elig_payload,
                )
            # Every N ticks emit a sample of the eligibility signals even if the
            # effective cap didn't move — a clamp that silently stays HOLD_ALL for
            # many ticks is invisible otherwise. Gated on the exec tick counter so
            # STARTING ticks are included.
            tick_no = int(getattr(self._exec, "_bot_tick_counter", 0))
            if tick_no > 0 and tick_no % _QUOTE_ELIGIBILITY_TICK_SAMPLE_EVERY_N == 0:
                sample_payload = self._build_quote_eligibility_diag(raw_q, eff_q)
                sample_payload.update(
                    {
                        "event": "quote_eligibility_tick_sample",
                        "effective": eff_q.eligibility.value,
                        "tick": tick_no,
                    }
                )
                log_extra(
                    logger,
                    logging.INFO,
                    "quote_eligibility_tick_sample",
                    sample_payload,
                )

            # Snap cross-venue reference fair value for adaptive lever #7.
            # ``binance_mid + basis_ewma`` is the Bybit/Binance-anchored
            # view of what GRVT should trade at under the current basis
            # regime. Blend weight controlled by
            # ``REFERENCE_VENUE_FAIR_BLEND_ALPHA``; compute_quote_decision
            # no-ops the blend when this value is None.
            ref_fair_price: Optional[float] = None
            # Also pass through the instantaneous basis (grvt_mid − bybit_mid)
            # and its smoothed EWMA for the Priority #2 basis-deviation
            # alpha term. Caller does the arithmetic; compute_quote_decision
            # no-ops when either input is missing.
            basis_now_for_decision: Optional[float] = None
            basis_ewma_for_decision: Optional[float] = None
            with self._state._lock:
                bm = self._state.binance_mid
                be = self._state.binance_basis_ewma
            if (
                isinstance(bm, (int, float))
                and isinstance(be, (int, float))
                and bm > 0
            ):
                ref_fair_price = float(bm) + float(be)
                # Basis inputs for the deviation alpha term:
                #   basis_now  = grvt_mid − bybit_mid (price units)
                #   basis_ewma = state.binance_basis_ewma (price units)
                # We already have grvt mid as ``mid`` at this point in the
                # loop. Skip the pair when the EWMA hasn't seeded or the
                # reference mid is stale / missing (fallbacks to zero shift).
                if (
                    isinstance(mid, (int, float))
                    and math.isfinite(float(mid))
                    and float(mid) > 0
                ):
                    basis_now_for_decision = float(mid) - float(bm)
                    basis_ewma_for_decision = float(be)

            # Priority #2 v2 — update the regime classifier each cycle
            # and pull the current sign. Classifier gates the
            # basis-deviation shift when IC is weak (|IC| < threshold)
            # or pair count is below the warmup floor. Default sign is
            # -1.0 (legacy mean-reversion) when the classifier has no
            # verdict yet — but with ``BASIS_DEVIATION_ALPHA=0`` in the
            # current SOL profile the shift is off regardless.
            basis_regime_sign_for_decision: float = -1.0
            if (
                basis_now_for_decision is not None
                and basis_ewma_for_decision is not None
                and isinstance(mid, (int, float))
                and math.isfinite(float(mid))
                and float(mid) > 0
            ):
                dev_px = basis_now_for_decision - basis_ewma_for_decision
                dev_bps_for_regime = dev_px / float(mid) * 10_000.0
                self._state.basis_regime.observe(
                    self._clock.monotonic(), dev_bps_for_regime, float(mid)
                )
                basis_regime_sign_for_decision = (
                    self._state.basis_regime.get_regime_sign()
                )
            # Short-term drift for adaptive lever #1 (trend_drift_
            # reservation_alpha). v1.5.155 — selectable signal window
            # via ``TREND_DRIFT_SIGNAL_WINDOW_SECONDS``. Default 0.25
            # = legacy 250 ms behaviour; prod profiles can opt into
            # longer windows (5/10/30/60 s) read from
            # ``state.mid_drift_windows`` to give the constructive
            # skew enough signal to actually lean in sustained trends.
            # See ``app/config.py::trend_drift_signal_window_seconds``
            # for the mapping. ``None`` when even the warmup fallback
            # hasn't accumulated samples yet.
            drift_bps = self._select_trend_drift_signal(eff_q)

            # Priority #1 adaptive lever — order-book imbalance.
            # Compute instantaneous L1 imbalance from current state.market
            # depth, EWMA-smooth in state, pass the smoothed value to
            # compute_quote_decision. No-op when the feature is disabled
            # (ob_imbalance_alpha=0) or depth is below the configured
            # floor. See docs/priorities.md #1 for the design.
            ob_imbalance_for_decision: Optional[float] = None
            if float(self._settings.ob_imbalance_alpha) > 0.0 and m is not None:
                b_sz = m.bid_size
                a_sz = m.ask_size
                b_px = m.best_bid
                a_px = m.best_ask
                if (
                    isinstance(b_sz, (int, float))
                    and isinstance(a_sz, (int, float))
                    and isinstance(b_px, (int, float))
                    and isinstance(a_px, (int, float))
                    and b_sz >= 0
                    and a_sz >= 0
                    and b_px > 0
                    and a_px > 0
                ):
                    total_sz = float(b_sz) + float(a_sz)
                    # Depth floor check (USD): skip the update when the
                    # top-of-book is too thin to be informative.
                    depth_usd = total_sz * (float(b_px) + float(a_px)) / 2.0
                    if (
                        total_sz > 0
                        and depth_usd >= float(self._settings.ob_imbalance_depth_floor_usd)
                    ):
                        inst_imb = (float(b_sz) - float(a_sz)) / total_sz
                        s = float(self._settings.ob_imbalance_smoothing_alpha)
                        with self._state._lock:
                            prev = self._state.ob_imbalance_ewma
                            if prev is None:
                                self._state.ob_imbalance_ewma = inst_imb
                            else:
                                self._state.ob_imbalance_ewma = (
                                    (1.0 - s) * float(prev) + s * inst_imb
                                )
                            ob_imbalance_for_decision = self._state.ob_imbalance_ewma
                    else:
                        # Hold last-seen EWMA across sparse ticks (don't reset).
                        with self._state._lock:
                            ob_imbalance_for_decision = self._state.ob_imbalance_ewma

            # Flow-score signals (Priority #3 v2 reservation alpha).
            # Pulled once outside the call so the snapshot is consistent
            # for the decision; ``FlowScoreAccumulator.snapshot()`` is
            # O(window) but bounded by ``FLOW_SCORE_RECENT_TRADES_MAXLEN``.
            flow_snap = None
            if float(self._settings.flow_score_reservation_alpha) > 0.0:
                try:
                    flow_snap = self._state.flow_score.snapshot()
                except Exception:
                    flow_snap = None

            # 1.4.9 gate-to-widening: build the SpreadComposition for
            # this tick. Passed as a per-side FLOOR into
            # ``compute_quote_decision`` so when a gate fires its
            # widening contribution caps the bid/ask half-spread at
            # MAX. The legacy eligibility-clamp logic stays in
            # place at v1.4.9 (no behaviour change) — the cutover release removes
            # those clamps and lets the composition's widening BE
            # the regime response. See plans/gate-to-widening.md.
            try:
                # Derive ActiveSides from the effective eligibility
                # via the existing intersect helper (always-BOTH base).
                from app.quoting import (
                    _intersect_active_sides_for_eligibility,
                )
                _active_for_composition = (
                    _intersect_active_sides_for_eligibility(
                        ActiveSides.BOTH, eff_q.eligibility
                    )
                )
                # v1.4.102 — slow_trend_gate needs the multi-minute
                # mid-history deque + current mid for its anchored-
                # median computation. Same data source the existing
                # ``_long_drift_eligibility`` uses (already snapshotted
                # above as ``mid_samples_long_snap``).
                _slow_trend_mid_now = (
                    float(self._state.market.mid_price)
                    if self._state.market is not None
                    and self._state.market.mid_price is not None
                    else None
                )
                # v1.4.106 Phase 1A — inventory_drift_gate inputs.
                # Compute 10s / 30s drifts once from the same mid-
                # history deque slow_trend uses, then pass into
                # build_spread_composition. Gate stays dormant when
                # mid is missing (cold start) or the deque is empty.
                # v1.5.181 Phase 2B.2 — read drift values from the
                # canonical ``state.mid_drift_windows`` cache populated
                # at tick start, not via per-call recompute.
                _id_drift_10s: float | None = None
                _id_drift_30s: float | None = None
                if (
                    bool(self._settings.inventory_drift_gate_enabled)
                    and _slow_trend_mid_now is not None
                    and _slow_trend_mid_now > 0
                ):
                    try:
                        _mdw_sc = self._state.mid_drift_windows
                        _id_drift_10s = (
                            _mdw_sc.drift_10s_bps if _mdw_sc else None
                        )
                        _id_drift_30s = (
                            _mdw_sc.drift_30s_bps if _mdw_sc else None
                        )
                    except Exception:
                        # Don't let a drift-lookup exception crash the
                        # composition build. Gate stays dormant.
                        _id_drift_10s = None
                        _id_drift_30s = None
                # M8 Candidate B — read the tape runtime feed once this
                # tick and derive the live microprice-deviation z-score
                # for the recorder-fed widen contributor. The feed is
                # ``None`` unless REGIME_USE_RUNTIME_RECORDER_FEED is on
                # (default OFF), so this whole block no-ops in production
                # today. Mirror the reader's monotonic health counters
                # onto BotState for the snapshot surface. Never raises on
                # the hot path: a defensive try/except yields ``None`` so
                # the bot falls back to its in-process signals.
                _mpz_runtime: float | None = None
                if self._runtime_feed is not None:
                    try:
                        _rf_stale_ns = int(
                            float(
                                self._settings.regime_runtime_feed_stale_threshold_s
                            )
                            * 1e9
                        )
                        _rf_snap = self._runtime_feed.read_fresh(_rf_stale_ns)
                        self._state.runtime_feed_read_count = (
                            self._runtime_feed.frames_read_total
                        )
                        self._state.runtime_feed_stale_count = (
                            self._runtime_feed.stale_count
                        )
                        self._state.runtime_feed_version_mismatch_count = (
                            self._runtime_feed.version_mismatch_count
                        )
                        self._state.runtime_feed_collision_count = (
                            self._runtime_feed.collision_count
                        )
                        if (
                            _rf_snap is not None
                            and _rf_snap.coverage_valid
                            and _rf_snap.microprice_dev_z_24h is not None
                        ):
                            _mpz_runtime = float(
                                _rf_snap.microprice_dev_z_24h
                            )
                    except Exception:
                        _mpz_runtime = None
                spread_composition = build_spread_composition(
                    settings=self._settings,
                    toxicity=tox_snap,
                    active_sides=_active_for_composition,
                    spread_floor_overlay_half_spread_bps=spread_floor_overlay,
                    vol_trend_state=self._state.vol_trend_gate,
                    post_swing_state=self._state.post_swing,
                    ob_imbalance_ewma=self._state.ob_imbalance_ewma,
                    microprice_dev_z_runtime=_mpz_runtime,
                    basis_regime_last_ic=getattr(
                        self._state.basis_regime, "last_ic", None
                    ),
                    basis_regime_pair_count=int(
                        getattr(self._state.basis_regime, "pair_count", 0)
                        or 0
                    ),
                    position_qty=pos.position_qty,
                    effective_abs_cap=float(
                        self._settings.max_abs_position
                    ),
                    drift_bps=(
                        raw_q.mid_return_500ms_bps
                        if raw_q.mid_return_500ms_bps is not None
                        else raw_q.mid_return_250ms_bps
                    ),
                    raw_eligibility=raw_q,
                    effective_eligibility=eff_q,
                    now_mono=now_mono,
                    mid_samples_long=self._state.mid_price_samples_long_snapshot(),
                    mid_now=_slow_trend_mid_now,
                    drift_bps_10s=_id_drift_10s,
                    drift_bps_30s=_id_drift_30s,
                )
            except Exception:
                logger.exception("build_spread_composition_failed")
                spread_composition = None
            # M8.4 Candidate B — count ticks where the recorder-fed
            # microprice-z widen actually contributed (nonzero on either
            # side). Stays 0 when the feed is off / dark / below
            # threshold. Gate G7 acceptance reads this off the snapshot.
            if spread_composition is not None and (
                spread_composition.microprice_runtime_bid_bps
                + spread_composition.microprice_runtime_ask_bps
            ) > 0.0:
                self._state.microprice_widen_z_runtime_feed_fires += 1
            # v1.4.164 Phase 4C.1+4C.2 — per-side expected-edge
            # suppression. Read each side's target half-spread from
            # the just-built composition, run the pure helper, and
            # write the resulting per-side ``refused`` flag back to
            # ``BotState`` so ``compute_quote_decision`` below can
            # consume it. Default ``MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE
            # = 0.0`` keeps the gate dormant; operator opts in by
            # setting a small negative threshold.
            try:
                from app.expected_edge import (
                    evaluate_per_side_expected_edge_suppression,
                )

                _max_bps = float(self._settings.max_half_spread_bps)
                _ths_bid: Optional[float] = None
                _ths_ask: Optional[float] = None
                if spread_composition is not None:
                    try:
                        _ths_bid = float(
                            spread_composition.effective_half_spread_bid_bps(
                                _max_bps
                            )
                        )
                    except Exception:
                        _ths_bid = None
                    try:
                        _ths_ask = float(
                            spread_composition.effective_half_spread_ask_bps(
                                _max_bps
                            )
                        )
                    except Exception:
                        _ths_ask = None
                _min_edge = float(
                    getattr(
                        self._settings,
                        "min_expected_net_edge_bps_per_side",
                        0.0,
                    )
                )
                _hyst_ticks = int(
                    getattr(
                        self._settings,
                        "expected_edge_hysteresis_ticks",
                        3,
                    )
                )
                _recovery_margin = float(
                    getattr(
                        self._settings,
                        "expected_edge_recovery_margin_bps",
                        0.2,
                    )
                )
                _rebate = float(
                    getattr(
                        self._settings,
                        "observability_maker_rebate_bps",
                        1.0,
                    )
                )
                _adverse = float(
                    getattr(
                        self._settings,
                        "observability_typical_adverse_markout_bps",
                        2.0,
                    )
                )
                # v1.5.41 Phase 4C.3 — per-side confidence multiplier
                # from trailing realised-edge history. Returns 1.0 when
                # feature disabled / insufficient samples / zero stdev,
                # so the call is safe to make unconditionally and the
                # call site stays symmetric for BID / ASK.
                _now_mono_ee = self._clock.monotonic()
                _seh = getattr(self._state, "side_edge_history", None)
                _bid_conf_mult = 1.0
                _ask_conf_mult = 1.0
                if _seh is not None:
                    try:
                        _bid_conf_mult = float(
                            _seh.confidence_multiplier(
                                Side.BUY, _now_mono_ee
                            )
                        )
                        _ask_conf_mult = float(
                            _seh.confidence_multiplier(
                                Side.SELL, _now_mono_ee
                            )
                        )
                    except Exception:  # noqa: BLE001 -- defensive
                        _bid_conf_mult = 1.0
                        _ask_conf_mult = 1.0

                # v1.5.205 Phase 4C.4 — stale-resting-quote penalty.
                # Sample current per-side quote age (seconds since ack)
                # into the rolling window, then compute the P50 and the
                # per-side penalty for the refuse + dampen evaluators.
                # Both penalty values default to 0.0 when:
                #   * the coefficient is 0 (feature disabled)
                #   * no working order on that side (fresh place)
                #   * rolling window hasn't accumulated 20+ samples
                #   * current age <= P50 (only upper tail is taxed)
                from app.expected_edge import compute_stale_risk_penalty_bps

                _stale_coeff = float(
                    getattr(
                        self._settings,
                        "stale_risk_penalty_bps_per_sec",
                        0.0,
                    )
                )
                _bid_stale_penalty_bps = 0.0
                _ask_stale_penalty_bps = 0.0
                if _stale_coeff > 0.0:
                    # Per-side ages: now_mono - WO.ts_acked. ts_acked
                    # is wall-clock (datetime), so use the bot's wall
                    # clock for the diff to stay consistent.
                    _now_wall = self._clock.now_utc()
                    _bid_age_sec: Optional[float] = None
                    _ask_age_sec: Optional[float] = None
                    try:
                        _wo_bid = self._state.get_working_order(Side.BUY, 0)
                        if _wo_bid is not None and _wo_bid.ts_acked is not None:
                            _bid_age_sec = (
                                _now_wall - _wo_bid.ts_acked
                            ).total_seconds()
                    except Exception:
                        _bid_age_sec = None
                    try:
                        _wo_ask = self._state.get_working_order(Side.SELL, 0)
                        if _wo_ask is not None and _wo_ask.ts_acked is not None:
                            _ask_age_sec = (
                                _now_wall - _wo_ask.ts_acked
                            ).total_seconds()
                    except Exception:
                        _ask_age_sec = None
                    # Push the LARGER age into the rolling window each
                    # tick. Using max(bid, ask) keeps a single-stream
                    # P50 that represents "how long has any side been
                    # resting" — both sides typically refresh together
                    # so the two ages are highly correlated.
                    _sample_age = None
                    if _bid_age_sec is not None and _ask_age_sec is not None:
                        _sample_age = max(_bid_age_sec, _ask_age_sec)
                    elif _bid_age_sec is not None:
                        _sample_age = _bid_age_sec
                    elif _ask_age_sec is not None:
                        _sample_age = _ask_age_sec
                    if _sample_age is not None:
                        self._state.record_quote_age_decision_sample_seconds(
                            _sample_age
                        )
                    _p50 = self._state.quote_age_decision_p50_seconds()
                    _bid_stale_penalty_bps = compute_stale_risk_penalty_bps(
                        current_quote_age_seconds=_bid_age_sec,
                        p50_quote_age_seconds=_p50,
                        coeff_bps_per_sec=_stale_coeff,
                    )
                    _ask_stale_penalty_bps = compute_stale_risk_penalty_bps(
                        current_quote_age_seconds=_ask_age_sec,
                        p50_quote_age_seconds=_p50,
                        coeff_bps_per_sec=_stale_coeff,
                    )

                # BID side
                _prev_refused_bid = bool(
                    getattr(self._state, "expected_edge_refused_bid", False)
                )
                _bid_eval = evaluate_per_side_expected_edge_suppression(
                    target_half_spread_bps=_ths_bid,
                    currently_refused=_prev_refused_bid,
                    recovery_ticks=int(
                        getattr(
                            self._state,
                            "expected_edge_recovery_ticks_bid",
                            0,
                        )
                    ),
                    min_expected_edge_bps=_min_edge,
                    hysteresis_ticks=_hyst_ticks,
                    recovery_margin_bps=_recovery_margin,
                    maker_rebate_bps=_rebate,
                    typical_adverse_markout_bps=_adverse,
                    confidence_multiplier=_bid_conf_mult,
                    stale_risk_penalty_bps=_bid_stale_penalty_bps,
                )
                self._state.expected_edge_refused_bid = _bid_eval.refused
                self._state.expected_edge_recovery_ticks_bid = (
                    _bid_eval.new_recovery_ticks
                )
                self._state.expected_edge_last_bid_bps = (
                    _bid_eval.expected_edge_bps
                )
                if _bid_eval.transition == "armed":
                    self._state.expected_edge_armed_bid_total += 1
                elif _bid_eval.transition == "cleared":
                    self._state.expected_edge_cleared_bid_total += 1
                # ASK side
                _prev_refused_ask = bool(
                    getattr(self._state, "expected_edge_refused_ask", False)
                )
                _ask_eval = evaluate_per_side_expected_edge_suppression(
                    target_half_spread_bps=_ths_ask,
                    currently_refused=_prev_refused_ask,
                    recovery_ticks=int(
                        getattr(
                            self._state,
                            "expected_edge_recovery_ticks_ask",
                            0,
                        )
                    ),
                    min_expected_edge_bps=_min_edge,
                    hysteresis_ticks=_hyst_ticks,
                    recovery_margin_bps=_recovery_margin,
                    maker_rebate_bps=_rebate,
                    typical_adverse_markout_bps=_adverse,
                    confidence_multiplier=_ask_conf_mult,
                    stale_risk_penalty_bps=_ask_stale_penalty_bps,
                )
                self._state.expected_edge_refused_ask = _ask_eval.refused
                self._state.expected_edge_recovery_ticks_ask = (
                    _ask_eval.new_recovery_ticks
                )
                self._state.expected_edge_last_ask_bps = (
                    _ask_eval.expected_edge_bps
                )
                if _ask_eval.transition == "armed":
                    self._state.expected_edge_armed_ask_total += 1
                elif _ask_eval.transition == "cleared":
                    self._state.expected_edge_cleared_ask_total += 1
            except Exception:
                logger.exception(
                    "expected_edge_suppression_eval_failed"
                )
            # v1.5.146 Phase 4C.2.a — dampen band. Runs AFTER the
            # refuse evaluator so it can pass ``already_refused`` to
            # short-circuit when refusal is firing. The widening (in
            # bps) is folded into the existing SpreadComposition via
            # ``dataclasses.replace`` + ``with_caps``; that single
            # update preserves the composition's audit-trail contract
            # (the new ``negative_expectancy_dampen_{bid,ask}_bps``
            # fields are visible in ``quote_decisions`` rows).
            #
            # The block is gated on ``spread_composition is not None``
            # because there's no way to dampen without a composition
            # to mutate; the refuse evaluator above accepts None
            # (passes through as dormant) but dampening must have a
            # composition to add to. Default config (DAMPEN_WIDEN_BPS=0)
            # short-circuits inside the helper.
            try:
                if spread_composition is not None:
                    from dataclasses import replace as _replace

                    from app.expected_edge import (
                        evaluate_per_side_dampen_band,
                    )

                    _dampen_max = float(
                        getattr(
                            self._settings,
                            "expected_edge_dampen_max_bps",
                            0.0,
                        )
                    )
                    _dampen_widen = float(
                        getattr(
                            self._settings,
                            "expected_edge_dampen_widen_bps",
                            0.0,
                        )
                    )
                    _bid_dampen = evaluate_per_side_dampen_band(
                        target_half_spread_bps=_ths_bid,
                        refuse_threshold_bps=_min_edge,
                        dampen_max_bps=_dampen_max,
                        dampen_widen_bps=_dampen_widen,
                        maker_rebate_bps=_rebate,
                        typical_adverse_markout_bps=_adverse,
                        confidence_multiplier=_bid_conf_mult,
                        already_refused=bool(_bid_eval.refused),
                        stale_risk_penalty_bps=_bid_stale_penalty_bps,
                    )
                    _ask_dampen = evaluate_per_side_dampen_band(
                        target_half_spread_bps=_ths_ask,
                        refuse_threshold_bps=_min_edge,
                        dampen_max_bps=_dampen_max,
                        dampen_widen_bps=_dampen_widen,
                        maker_rebate_bps=_rebate,
                        typical_adverse_markout_bps=_adverse,
                        confidence_multiplier=_ask_conf_mult,
                        already_refused=bool(_ask_eval.refused),
                        stale_risk_penalty_bps=_ask_stale_penalty_bps,
                    )
                    if _bid_dampen.armed or _ask_dampen.armed:
                        spread_composition = _replace(
                            spread_composition,
                            negative_expectancy_dampen_bid_bps=float(
                                _bid_dampen.widen_bps
                            ),
                            negative_expectancy_dampen_ask_bps=float(
                                _ask_dampen.widen_bps
                            ),
                        ).with_caps(max_bps=_max_bps)
                    # Update counters + last-bps telemetry. Counters
                    # bump every tick the band fires (matches the
                    # other gate counter cadence on BotState).
                    if _bid_dampen.armed:
                        self._state.negative_expectancy_dampen_bid_total += 1
                    if _ask_dampen.armed:
                        self._state.negative_expectancy_dampen_ask_total += 1
                    self._state.negative_expectancy_dampen_last_bid_bps = (
                        float(_bid_dampen.widen_bps)
                    )
                    self._state.negative_expectancy_dampen_last_ask_bps = (
                        float(_ask_dampen.widen_bps)
                    )
            except Exception:
                logger.exception(
                    "expected_edge_dampen_band_eval_failed"
                )
            # v1.5.207 Phase 4C.5 — participation score per side.
            # Derived from each side's expected_net_edge (post-stale-
            # risk-penalty, post-confidence-multiplier). Continuous
            # [0, 1] summary of the existing 3-state refuse/dampen/
            # quote decision. Observability-only in v1.5.207; the
            # gates above still drive behavior. A future release can
            # flip a feature flag to make the score itself the
            # driver. Disagreement counters bump when the score's
            # implied action differs from what the existing gates
            # decided — operator audits the calibration via these.
            try:
                from app.participation_score import (
                    compute_participation_score,
                    is_score_consistent_with_existing_decision,
                )

                _full_edge = float(
                    getattr(
                        self._settings,
                        "participation_score_full_edge_bps",
                        3.0,
                    )
                )
                _soft = float(
                    getattr(
                        self._settings,
                        "participation_score_soft_threshold",
                        0.7,
                    )
                )
                _hard = float(
                    getattr(
                        self._settings,
                        "participation_score_hard_threshold",
                        0.3,
                    )
                )
                _dampen_max_p = float(
                    getattr(
                        self._settings,
                        "expected_edge_dampen_max_bps",
                        0.0,
                    )
                )
                _refuse_floor_p = float(
                    getattr(
                        self._settings,
                        "min_expected_net_edge_bps_per_side",
                        -1.0,
                    )
                )
                # Score the BID side.
                _bid_edge = self._state.expected_edge_last_bid_bps
                _bid_score = compute_participation_score(
                    expected_edge_bps=_bid_edge,
                    refuse_floor_bps=_refuse_floor_p,
                    dampen_floor_bps=_dampen_max_p,
                    full_edge_bps=_full_edge,
                    soft_threshold=_soft,
                    hard_threshold=_hard,
                )
                self._state.participation_score_bid = _bid_score
                if _bid_score is not None:
                    self._state.participation_score_bid_recent.append(
                        float(_bid_score)
                    )
                    _bid_consistent = (
                        is_score_consistent_with_existing_decision(
                            _bid_score,
                            was_refused=bool(
                                self._state.expected_edge_refused_bid
                            ),
                            was_dampened=(
                                self._state.negative_expectancy_dampen_last_bid_bps
                                > 1e-9
                            ),
                            soft_threshold=_soft,
                            hard_threshold=_hard,
                        )
                    )
                    if not _bid_consistent:
                        self._state.participation_score_disagreement_bid_total += 1
                # Score the ASK side.
                _ask_edge = self._state.expected_edge_last_ask_bps
                _ask_score = compute_participation_score(
                    expected_edge_bps=_ask_edge,
                    refuse_floor_bps=_refuse_floor_p,
                    dampen_floor_bps=_dampen_max_p,
                    full_edge_bps=_full_edge,
                    soft_threshold=_soft,
                    hard_threshold=_hard,
                )
                self._state.participation_score_ask = _ask_score
                if _ask_score is not None:
                    self._state.participation_score_ask_recent.append(
                        float(_ask_score)
                    )
                    _ask_consistent = (
                        is_score_consistent_with_existing_decision(
                            _ask_score,
                            was_refused=bool(
                                self._state.expected_edge_refused_ask
                            ),
                            was_dampened=(
                                self._state.negative_expectancy_dampen_last_ask_bps
                                > 1e-9
                            ),
                            soft_threshold=_soft,
                            hard_threshold=_hard,
                        )
                    )
                    if not _ask_consistent:
                        self._state.participation_score_disagreement_ask_total += 1
            except Exception:
                logger.exception("participation_score_eval_failed")
            # Store on state for the heartbeat publisher to emit.
            try:
                self._state.last_spread_composition = spread_composition
            except Exception:
                pass

            # v1.5.155 — position-aware favorable-exit attempt for
            # realised_edge_side_suppress, BEFORE ``is_suppressed``
            # is read by the decision build below. Clears a per-side
            # suppression when that side is the only way to reduce
            # current adverse inventory. Per CLAUDE.md Rule 0c.
            try:
                _res = getattr(
                    self._state, "realised_edge_side_suppress", None
                )
                if _res is not None and _res.enabled():
                    _pos_qty = float(
                        getattr(self._state.position, "position_qty", 0.0)
                        or 0.0
                    )
                    _inv_thresh = float(
                        getattr(
                            self._settings,
                            "realised_edge_suppress_position_favorable_inventory_threshold",
                            1.0,
                        )
                    )
                    _now_mono = self._clock.monotonic()
                    _res.try_clear_via_position_favorable(
                        side=Side.BUY,
                        now_mono=_now_mono,
                        position_qty=_pos_qty,
                        inventory_threshold=_inv_thresh,
                    )
                    _res.try_clear_via_position_favorable(
                        side=Side.SELL,
                        now_mono=_now_mono,
                        position_qty=_pos_qty,
                        inventory_threshold=_inv_thresh,
                    )
                    # v1.5.197 — idle-decay clearance.
                    _res.try_clear_via_idle(
                        now_mono=_now_mono,
                        last_fill_mono=self._state.last_fill_at_mono,
                        idle_clear_seconds=float(
                            getattr(
                                self._settings,
                                "realised_edge_suppress_idle_clear_seconds",
                                300.0,
                            )
                        ),
                    )
            except Exception:
                logger.exception(
                    "realised_edge_side_suppress_position_favorable_failed"
                )

            # v1.5.157 — position-aware favorable-exit attempt for
            # at_touch_adverse_pause, before is_paused is read by the
            # decision build below. Same "reducing-side must always
            # be available" principle as realised_edge_side_suppress.
            try:
                _atp = getattr(
                    self._state, "at_touch_adverse_pause", None
                )
                if _atp is not None and _atp.enabled():
                    _pos_qty_atp = float(
                        getattr(self._state.position, "position_qty", 0.0)
                        or 0.0
                    )
                    _inv_thresh_atp = float(
                        getattr(
                            self._settings,
                            "at_touch_adverse_pause_position_favorable_inventory_threshold",
                            1.0,
                        )
                    )
                    _now_mono_atp = self._clock.monotonic()
                    _atp.try_clear_via_position_favorable(
                        side=Side.BUY,
                        now_mono=_now_mono_atp,
                        position_qty=_pos_qty_atp,
                        inventory_threshold=_inv_thresh_atp,
                    )
                    _atp.try_clear_via_position_favorable(
                        side=Side.SELL,
                        now_mono=_now_mono_atp,
                        position_qty=_pos_qty_atp,
                        inventory_threshold=_inv_thresh_atp,
                    )
                    # v1.5.197 — idle-decay clearance.
                    _atp.try_clear_via_idle(
                        now_mono=_now_mono_atp,
                        last_fill_mono=self._state.last_fill_at_mono,
                        idle_clear_seconds=float(
                            getattr(
                                self._settings,
                                "at_touch_adverse_pause_idle_clear_seconds",
                                300.0,
                            )
                        ),
                    )
            except Exception:
                logger.exception(
                    "at_touch_adverse_pause_position_favorable_failed"
                )

            # v1.5.281/283 AQC — read the controller's live
            # aggression_level ONCE here, before the decision is built,
            # so both the constructive skew lerp inside
            # ``compute_quote_decision`` (Phase 4, AQC_WIRE_SKEW) AND the
            # decision stamp below use the SAME value. Stamped whenever
            # AQC is enabled; None otherwise → quote pipeline byte-
            # identical to pre-AQC. The controller is updated later this
            # tick (after the stamp), so this reads the prior tick's
            # level — a benign 1-tick lag for a 300 s-window PI.
            aqc_aggr_stamp: Optional[float] = None
            _aqc_ctrl = getattr(
                self._state, "active_quoting_controller", None
            )
            if _aqc_ctrl is not None and bool(
                getattr(_aqc_ctrl.settings, "enabled", False)
            ):
                _aqc_lvl = getattr(_aqc_ctrl, "aggression_level", None)
                if isinstance(_aqc_lvl, (int, float)) and math.isfinite(
                    float(_aqc_lvl)
                ):
                    aqc_aggr_stamp = float(_aqc_lvl)
            # Depth fields come from the same ``BestBidAsk`` we used for
            # ``mid``. Missing depth (older WS payloads, test fixtures,
            # early ticks before full book state) falls back to mid
            # inside ``compute_quote_decision``.
            decision = compute_quote_decision(
                self._settings,
                mid,
                pos.position_qty,
                # v1.5.232 — compute_quote_decision requires float, not
                # Optional[float]. Pass the 0.0-during-warm-up flavour
                # (`_vol_bps_or_zero`) to preserve pre-v1.5.232
                # behaviour for the quote engine (warm-up → vol=0 →
                # no vol widening contribution, same as before).
                _vol_bps_or_zero,
                tox_snap,
                spread_floor_overlay_half_spread_bps=spread_floor_overlay,
                composition=spread_composition,
                best_bid=m.best_bid if m is not None else None,
                best_ask=m.best_ask if m is not None else None,
                bid_size=m.bid_size if m is not None else None,
                ask_size=m.ask_size if m is not None else None,
                reference_fair_price=ref_fair_price,
                short_term_drift_bps=drift_bps,
                # v1.5.283 AQC Phase 4 — constructive inventory-skew
                # lerp input (None / wire-off → base coeff, byte-
                # identical). Same value stamped onto the decision below.
                aqc_aggression_level=aqc_aggr_stamp,
                ob_imbalance_smoothed=ob_imbalance_for_decision,
                cross_venue_basis_now=basis_now_for_decision,
                cross_venue_basis_ewma=basis_ewma_for_decision,
                basis_deviation_regime_sign=basis_regime_sign_for_decision,
                join_depth_overlay_bps=(
                    self._state.join_depth_controller.current_overlay_bps()
                ),
                flow_score_tfi_signed=(
                    flow_snap.tfi_signed_normalised if flow_snap is not None else None
                ),
                flow_score_streak_buy=(
                    flow_snap.streak_buy_count if flow_snap is not None else 0
                ),
                flow_score_streak_sell=(
                    flow_snap.streak_sell_count if flow_snap is not None else 0
                ),
                flow_score_streak_window_prints=int(
                    self._settings.flow_score_streak_window_prints
                ),
                vol_regime_shrink_factor=float(
                    self._state.vol_regime_adjustment.shrink_factor
                ),
                vol_regime_half_spread_bump_bps=float(
                    self._state.vol_regime_adjustment.half_spread_bump_bps
                ),
                # 1.2.3: rolling-median 5s markout from the toxicity
                # engine's recent-fills window. Drives the markout-tier
                # size scaler (separate from toxicity-score scaler).
                # ``None`` when fewer than min markout samples have
                # accumulated (warmup); the scaler is silent in that
                # case.
                recent_markout_5s_median_bps=(
                    self._state.toxicity_recent_median_markout_5s_bps()
                ),
                # Phase 8A (v1.5.185) — current AS k-intensity cache.
                # ``None`` until the first refresh (cold start) OR
                # when AS is disabled. ``compute_quote_decision``
                # only consumes this when
                # ``AVELLANEDA_STOIKOV_ENABLED=true``.
                as_k_intensity_per_min=getattr(
                    self._state, "as_k_intensity_per_min", None
                ),
                # 1.2.8: basis-regime size shrink (when mode=size_shrink
                # and IC is in the signal-absent band). 1.0 when the
                # gate isn't firing; < 1.0 when it is. Composes with
                # the existing toxicity / vol / markout shrinks via
                # ``min`` semantics in ``compute_quote_decision``.
                basis_regime_size_mult=float(
                    getattr(self._state, "basis_regime_size_mult", 1.0)
                ),
                basis_regime_size_mult_reason=(
                    getattr(self._state, "basis_regime_size_mult_reason", None)
                ),
                # todo-011: post-fill replace cooldown remaining ms,
                # per side. Zero when feature disabled
                # (``POST_FILL_REPLACE_COOLDOWN_MS=0``) or when no fill
                # has occurred yet on that side this session, or when
                # the cooldown has already elapsed. Positive value =
                # suppress placements on that side this cycle. See
                # ``BUGS/todo-011.md``.
                post_fill_cooldown_bid_remaining_ms=(
                    self._compute_post_fill_cooldown_remaining_ms(Side.BUY)
                ),
                post_fill_cooldown_ask_remaining_ms=(
                    self._compute_post_fill_cooldown_remaining_ms(Side.SELL)
                ),
                # 2026-05-12 codex-#1 narrow: at-touch adverse pause.
                # ``is_paused`` is False when the gate is disabled or
                # when no trigger fired recently. When True, the
                # whole side is suppressed for this cycle.
                at_touch_adverse_pause_bid=(
                    self._state.at_touch_adverse_pause.is_paused(
                        Side.BUY, self._clock.monotonic()
                    )
                ),
                at_touch_adverse_pause_ask=(
                    self._state.at_touch_adverse_pause.is_paused(
                        Side.SELL, self._clock.monotonic()
                    )
                ),
                # v1.4.161 Phase 4C.3 mini — per-side realised-edge
                # suppression. Defensive ``getattr`` for BotState
                # shapes from before v1.4.161.
                realised_edge_suppress_bid=bool(
                    getattr(
                        self._state,
                        "realised_edge_side_suppress",
                        None,
                    )
                    and self._state.realised_edge_side_suppress.is_suppressed(
                        Side.BUY, self._clock.monotonic()
                    )
                ),
                realised_edge_suppress_ask=bool(
                    getattr(
                        self._state,
                        "realised_edge_side_suppress",
                        None,
                    )
                    and self._state.realised_edge_side_suppress.is_suppressed(
                        Side.SELL, self._clock.monotonic()
                    )
                ),
                # v1.4.164 Phase 4C.1+4C.2 — per-side expected-edge
                # refusal. The flag is maintained by
                # ``_evaluate_expected_edge_suppression`` (called
                # earlier in the tick from the SpreadComposition's
                # per-side half-spread). Read the state cache here.
                expected_edge_refused_bid=bool(
                    getattr(
                        self._state,
                        "expected_edge_refused_bid",
                        False,
                    )
                ),
                expected_edge_refused_ask=bool(
                    getattr(
                        self._state,
                        "expected_edge_refused_ask",
                        False,
                    )
                ),
                # 2026-05-12 codex-#3: fill-burst size shrink. 1.0
                # outside burst cooldown; configured shrink factor
                # during cooldown. Composes via ``min`` semantics.
                fill_burst_size_mult=(
                    self._state.fill_burst_detector.current_size_mult(
                        self._clock.monotonic()
                    )
                ),
                # v1.4.112 Phase 1C — regime_controller knob overlays.
                # Both default to 1.0 (no-op) when the FSM is disabled
                # or in NORMAL mode. DEFENSIVE widens half-spread 1.5×
                # and halves size; SHOCK widens 2× and keeps full
                # size (shock_gate's binary clamp handles adding-side
                # suppression).
                regime_base_half_spread_mult=float(
                    self._state.regime_knobs.base_half_spread_mult
                ),
                regime_quote_notional_mult=float(
                    self._state.regime_knobs.quote_notional_mult
                ),
                # v1.5.156 Option B — no-fill spread compression. Pass
                # the seconds-since-last-fill so the function can apply
                # the gradual half-spread compression when no fills
                # have happened for a while. ``None`` at session
                # start (before any fill has been recorded) so the
                # compression doesn't fire before a baseline is
                # established — only kicks in after the first fill
                # establishes the timer, then on the next no-fill
                # gap that exceeds the configured trigger.
                seconds_since_last_fill=(
                    self._compute_seconds_since_last_fill()
                ),
                # v1.5.248 — no-fill aggression escalator. When
                # NO_FILL_ESCALATOR_ENABLED is true, computes a
                # per-tick aggression level and per-dimension
                # multipliers; otherwise returns the no-op output.
                # Replaces the simple NO_FILL_COMPRESS path
                # (still respected when escalator is disabled).
                no_fill_escalator_output=(
                    (lambda _ssf: (
                        __import__(
                            "app.no_fill_escalator", fromlist=["compute_escalator_output"]
                        ).compute_escalator_output(
                            seconds_since_last_fill=_ssf,
                            enabled=bool(getattr(
                                self._settings,
                                "no_fill_escalator_enabled",
                                False,
                            )),
                            trigger_seconds=float(getattr(
                                self._settings,
                                "no_fill_escalator_trigger_seconds",
                                60.0,
                            )),
                            ramp_seconds=float(getattr(
                                self._settings,
                                "no_fill_escalator_ramp_seconds",
                                180.0,
                            )),
                            spread_compress_max_bps=float(getattr(
                                self._settings,
                                "no_fill_escalator_spread_compress_max_bps",
                                10.0,
                            )),
                            microprice_attenuate=bool(getattr(
                                self._settings,
                                "no_fill_escalator_microprice_attenuate",
                                True,
                            )),
                            toxicity_attenuate=bool(getattr(
                                self._settings,
                                "no_fill_escalator_toxicity_attenuate",
                                True,
                            )),
                            reservation_shift_mult_at_full=float(getattr(
                                self._settings,
                                "no_fill_escalator_reservation_shift_mult_at_full",
                                0.5,
                            )),
                        )
                    ))(self._compute_seconds_since_last_fill())
                ),
                # v1.5.158 Option A — vol-climbing ratio for the
                # anticipatory-widening gate. None when feature
                # disabled or buffer not warm.
                vol_climbing_ratio=self._compute_vol_climbing_ratio(),
                # v1.5.158 Option B — current UTC hour/minute for
                # the funding-settle widening overlay. Cheap;
                # always set so the gate's own enabled flag is
                # the on/off switch.
                utc_hour_minute=(
                    lambda _u: (_u.hour, _u.minute)
                )(self._clock.now_utc()),
                # v1.5.158 Option C — vol-adaptive position-cap
                # override. When enabled AND vol exceeds the
                # configured threshold, pass a reduced cap so
                # inventory_skew + at-max logic + soft/hard skew
                # bands all use the smaller value. Bot keeps
                # quoting both sides — only the cap shrinks.
                # ``None`` = no override (legacy).
                max_abs_position_override=(
                    self._compute_vol_adaptive_cap()
                ),
                # v1.5.215 Phase 8D OFI alpha — 5th reservation-alpha.
                # Pure read of the public-WS-fed accumulator on
                # ``state.ofi``. Returns ``None`` during the first
                # few hundred ms after start (no BBO contribution yet)
                # OR when the accumulator hasn't been fed (e.g. a
                # venue whose public WS hasn't been wired to call
                # ``state.ofi.record_bbo``). ``compute_quote_decision``
                # treats ``None`` as "no signal" → zero shift applied.
                ofi_signal_5s_normalised=(
                    self._state.ofi.signal_5s_normalised()
                    if getattr(self._state, "ofi", None) is not None
                    else None
                ),
                # v1.5.215 Phase 8B queue-aware sizing + inside-post.
                # Per-side queue-position ratios computed from the
                # current L1 sizes vs the bot's own resting WO size
                # on that side. Both default ``None`` when the bot
                # has no resting WO on that side OR L1 total is
                # unavailable. ``compute_quote_decision`` treats
                # ``None`` as "no signal" → no shrink + no inside-post.
                queue_position_ratio_bid=self._compute_queue_position_ratio(Side.BUY),
                queue_position_ratio_ask=self._compute_queue_position_ratio(Side.SELL),
                # Phase 1b (v1.5.90) — clock-routing for the two
                # utc_now() sites inside compute_quote_decision. Lets
                # the replay driver produce deterministic timestamps.
                clock=self._clock,
            )
            decision = apply_quote_eligibility_to_decision(
                decision,
                eff_q.eligibility,
                eligibility_reason=eff_q.reason,
            )
            # Phase 8A Option B (v1.5.189) — increment the AS fire
            # counter when the flag is on. Counts ticks where the
            # AS path was the chosen branch in
            # ``compute_quote_decision``. Surfaces in the snapshot
            # for the AC verification check.
            if bool(getattr(self._settings, "avellaneda_stoikov_enabled", False)):
                self._state.as_path_fire_count += 1
            now_u = self._clock.now_utc()
            eff_age = _effective_book_age_ms_at_decision(m, now_u)
            with self._state._lock:
                pub = dict(self._state.public_ws_last_inbound_derived_ms)
            # Reuse the cross-venue reference snap taken before the
            # decision call for telemetry (no need to re-lock — these
            # values drift slowly vs the quote cycle).
            binance_mid_snap = bm
            binance_basis_snap = be
            regime = _decision_market_data_regime(self._state)
            qw = pub.get("public_ws_queue_wait_ms")
            qra = pub.get("public_ws_receive_to_apply_ms")
            # N2 analysis-day instrumentation: snapshot the basis-regime
            # classifier state at decision time. ``observe()`` was called
            # earlier in this cycle (lines ~3055), so these properties
            # reflect the IC computed off the freshest paired-sample
            # buffer. ``last_ic`` is None when fewer than
            # ``min_pair_samples`` pairs are buffered yet (warmup).
            try:
                basis_ic_snap = self._state.basis_regime.last_ic
                basis_pair_count_snap = int(self._state.basis_regime.pair_count)
                basis_regime_sign_snap = int(
                    self._state.basis_regime.last_regime_sign or 0
                )
            except Exception:
                basis_ic_snap = None
                basis_pair_count_snap = None
                basis_regime_sign_snap = None
            # v1.5.281/283 AQC — stamp the controller's live
            # aggression_level (read once above, before the decision was
            # built) onto the decision so the quote engine can tighten
            # the econ min-half-spread floor (Phase 2) / engage exec-bias
            # earlier (Phase 3); the skew lerp (Phase 4) already consumed
            # the same value inside ``compute_quote_decision``. The field
            # is also useful telemetry in observe-only mode. None →
            # quote pipeline byte-identical.
            decision = replace(
                decision,
                aqc_aggression_level=aqc_aggr_stamp,
                source_book_ts_exchange_ms=m.ts_exchange_ms if m else None,
                source_book_ts_local_iso=m.ts_local.isoformat() if m and m.ts_local else None,
                effective_book_age_at_decision_ms=eff_age,
                public_ws_queue_wait_ms_latest=float(qw) if isinstance(qw, (int, float)) else None,
                public_ws_receive_to_apply_ms_latest=(
                    float(qra) if isinstance(qra, (int, float)) else None
                ),
                decision_market_data_regime=regime,
                binance_mid=binance_mid_snap,
                binance_basis_ewma=binance_basis_snap,
                basis_regime_sign=basis_regime_sign_snap,
                basis_ic=basis_ic_snap,
                basis_pair_count=basis_pair_count_snap,
            )
            self._state.set_quote_decision_markers(
                m.ts_local if m else None,
                effective_book_age_at_decision_ms=eff_age,
                book_ts_exchange_ms=m.ts_exchange_ms if m else None,
                book_apply_to_decision_ms=None,
                public_ws_queue_wait_ms_latest=decision.public_ws_queue_wait_ms_latest,
                public_ws_receive_to_apply_ms_latest=decision.public_ws_receive_to_apply_ms_latest,
                decision_market_data_regime=regime,
            )
            with self._state._lock:
                b2d = self._state.last_book_apply_to_decision_ms
                self._state.last_quote_cycle_id = decision.quote_cycle_id
                self._state.last_active_sides = decision.active_sides.value
                # Watchdog signal: note when the quote engine wants to
                # quote at least one side. A persistent gap between
                # this timestamp and ``last_place_attempt_ts_mono`` is
                # the deadlock signature — engine says "quote", but
                # execution isn't dispatching.
                if decision.active_sides.value != "NONE":
                    self._state.last_quote_engine_non_hold_ts_mono = self._clock.monotonic()
                self._state.note_quote_eligibility_resume(last_eff, eff_q.eligibility)
                self._state.quote_eligibility_last_effective = eff_q.eligibility
            decision = replace(decision, book_apply_to_decision_ms=b2d)

            eq = None
            with self._state._lock:
                acct = self._state.account
                if acct and acct.equity_usd is not None:
                    eq = acct.equity_usd

            pnl_snap = self._pnl.build_snapshot(pos, eq)
            with self._state._lock:
                self._state.pnl = pnl_snap
                reconcile_ap = self._state.reconcile_auto_pause

            # TODO-002: account-data-stale gate. Compute "seconds since last
            # successful account refresh" from the monotonic clock anchor; pass
            # to evaluate_risk so the gate can fire on Bluefin REST hangs.
            # 2026-05-16 Codex #1 fix: read the success-only clock. The
            # pre-fix code read the shared attempt/success clock, which
            # repeated REST failures kept advancing — defeating the gate.
            acct_last_m = self._state.account_rest_last_success_monotonic
            acct_age_s = (
                None if acct_last_m is None else (self._clock.monotonic() - acct_last_m)
            )
            risk = evaluate_risk(
                self._settings,
                bot_status=self._state.bot_status,
                manual_pause=self._state.manual_pause,
                killed=self._state.killed,
                flatten_mode=self._state.flatten_mode,
                market=m,
                position_qty=pos.position_qty,
                position_notional=pos.position_notional,
                open_order_count=self._state.open_order_count(),
                pnl=pnl_snap,
                toxicity=tox_snap,
                execution_errors=self._state.execution_errors_window_snapshot(
                    self._settings.execution_errors_window_seconds
                )["windowed"],
                desync=self._state.order_desync,
                desync_phase=self._state.desync_phase,
                desync_quarantine_remaining=self._state.desync_quarantine_remaining,
                trades_last_minute=self._state.trades_last_minute(),
                reconcile_auto_pause=reconcile_ap,
                account_seconds_since_refresh=acct_age_s,
                **self._public_ws_risk_kwargs(),
            )
            lat["latency_hot_path_local_compute_ms"] = (time.perf_counter() - t_hot) * 1000.0

            # TODO-001: inventory consistency watchdog (rate-limited by config).
            # Independent of normal trading flow — runs every quote tick but only
            # fires the actual comparison on its own cadence. On breach, suspend
            # quoting, cancel resting, and Telegram CRITICAL.
            try:
                breach = check_inventory_consistency(self._state, self._settings)
            except Exception:
                logger.exception("inventory_consistency_check_failed")
                breach = None
            if breach is not None:
                self._handle_inventory_consistency_breach(breach)

            if risk.action == RiskAction.KILL:
                self.kill(
                    ",".join(risk.reasons),
                    self._build_kill_payload(risk.reasons),
                )
                return
            if risk.action == RiskAction.FLATTEN:
                self.flatten(blocking=True)
                return
            if risk.action == RiskAction.SOFT_FLATTEN:
                # See identical handler ~500 lines up. Toxicity-trigger
                # path: enter soft-flatten with phase-2-immediate +
                # operator-configured taker fallback. Replaces the
                # legacy ``RiskAction.FLATTEN`` taker market_close.
                self._enter_soft_flatten(
                    None,
                    force_phase=2,
                    taker_fallback_ticks=int(
                        getattr(
                            self._settings,
                            "soft_flatten_taker_fallback_ticks",
                            0,
                        )
                        or 0
                    )
                    or None,
                    trigger_reason="toxicity_hard_trigger",
                    log_message_override=(
                        "toxicity hard-trigger with inventory — "
                        "entering aggressive post-only soft-flatten "
                        "(phase 2 from t=0, taker fallback if drift "
                        "exceeds threshold)"
                    ),
                    log_payload_override={
                        "reasons": list(risk.reasons),
                    },
                )
                return

            _stale_warn_now = (
                "stale_data_warn" in risk.reasons
                or "public_ws_stale_warn" in risk.reasons
            )
            if _stale_warn_now and m is not None:
                # Throttle (v1.5.301): the risk gate re-flags staleness
                # EVERY tick, so a single sustained book gap — e.g. the
                # genuine 2–5 s bbo-tbt gaps OKX shows in quiet periods,
                # which also replay 1:1 in the backtester — would emit one
                # identical WARNING per 500 ms tick. Log ONCE on episode
                # entry; suppress the per-tick repeats until the book goes
                # fresh again. Quoting is unaffected: risk.action already
                # disabled this tick's quote regardless of whether we log.
                if not self._stale_data_warn_episode_active:
                    self._stale_data_warn_episode_active = True
                    with self._state._lock:
                        last_market_success = self._state.market_data_last_success_wall_ts
                        market_failed_streak = self._state.market_data_failed_refresh_streak
                        market_unchanged_streak = self._state.market_data_unchanged_snapshot_streak
                        recovery_state = self._state.bot_status.value
                    if self._public_ws_live_path():
                        with self._state._lock:
                            lw = self._state.public_ws_last_message_wall_ts
                            pub_conn = self._state.public_ws_connected
                        stale_s = seconds_since(lw) if lw else None
                        warn_s = self._settings.public_ws_stale_warn_seconds
                        kill_s = self._settings.public_ws_stale_kill_seconds
                    else:
                        stale_s = seconds_since(m.ts_local)
                        with self._state._lock:
                            pub_conn = self._state.public_ws_connected
                        warn_s = self._settings.stale_data_warn_seconds
                        kill_s = self._settings.stale_data_kill_seconds
                    log_extra(
                        logger,
                        logging.WARNING,
                        "stale market data — quoting disabled",
                        {
                            "event": "stale_data_warn",
                            "seconds_since_book_ts": stale_s,
                            "warn_threshold_s": warn_s,
                            "kill_threshold_s": kill_s,
                            "seconds_since_public_ws_message": (
                                self._public_ws_risk_kwargs().get("public_ws_seconds_since_message")
                                if self._public_ws_live_path()
                                else None
                            ),
                            "seconds_since_last_market_data_refresh": (
                                seconds_since(last_market_success) if last_market_success else None
                            ),
                            "public_ws_connected": pub_conn,
                            "market_data_recovery_state": recovery_state,
                            "market_data_failed_refresh_streak": market_failed_streak,
                            "market_data_unchanged_snapshot_streak": market_unchanged_streak,
                        },
                    )
            elif not _stale_warn_now:
                # Book fresh again (or staleness escalated past warn into
                # the kill path, which logs separately) — re-arm so the
                # next distinct stale episode logs its own entry line.
                self._stale_data_warn_episode_active = False

            if "trade_rate_limit" in risk.reasons:
                log_extra(
                    logger,
                    logging.WARNING,
                    "trade activity limit — quoting disabled for this tick",
                    {
                        "event": "trade_rate_limit",
                        "trades_last_minute": self._state.trades_last_minute(),
                        "max_trades_per_minute": self._settings.max_trades_per_minute,
                    },
                )

            quote_persisted = False
            deferred_allow_row: dict | None = None
            defer_exec_telemetry = (
                risk.action == RiskAction.ALLOW
                and self._settings.trading_enabled
                and self._state.bot_status == BotStatus.RUNNING
            )
            # 1.2.14: build the ladder for this cycle. At
            # ``LADDER_NUM_LEVELS_PER_SIDE=1`` (default) this returns a
            # single-rung ladder whose px / sz exactly match the
            # decision's scalar quote — no behavioural drift. At N > 1
            # the multi-rung ladder is computed and persisted to the
            # ``quote_decisions`` row for snapshot calibration; only the
            # inside rung is actually placed in Phase 1 (Phase 2 wires
            # the multi-rung execution path).
            from app.ladder import LadderConfig, build_ladder
            # Phase 4G.8 (v1.4.219) — apply the regime-controller's
            # ``ladder_levels_max`` knob to the ladder builder's
            # num_levels_per_side. Pre-v1.4.219 this knob was set on
            # CAUTIOUS / DEFENSIVE / SHOCK regime rows (cap=1 / 1 / 0
            # respectively) but never consulted by the build path —
            # the bot still built 2-rung ladders in those modes,
            # invalidating the "CAUTIOUS = 1 rung" pitch documented in
            # the regime narrative. ``state.regime_knobs.ladder_levels_max``
            # is None on NORMAL / CALM (full ladder per config) and
            # a positive int on the defensive modes.
            #
            # Floor at 1 by default: SHOCK's documented cap=0 would
            # zero the entire ladder and rely exclusively on the SF
            # path to flatten — a behavioural change historically
            # gated behind the v1.5.146 ``SHOCK_LADDER_ALLOW_FULL_DARK``
            # flag (Phase 4G.10). When that flag is True the cap is
            # honoured as-is and SHOCK builds zero rungs.
            _cfg_levels = int(self._settings.ladder_num_levels_per_side)
            _regime_levels_cap = getattr(
                self._state.regime_knobs, "ladder_levels_max", None
            )
            _allow_full_dark = bool(
                getattr(
                    self._settings,
                    "shock_ladder_allow_full_dark",
                    False,
                )
            )
            if _regime_levels_cap is not None:
                _knob_capped = min(_cfg_levels, int(_regime_levels_cap))
                if _allow_full_dark:
                    # Honour cap=0 (SHOCK fully dark on passive
                    # quoting). Still ``max(0, ...)`` for negative-
                    # cap defensiveness; build_ladder accepts 0.
                    _effective_levels = max(0, _knob_capped)
                else:
                    _effective_levels = max(1, _knob_capped)
            else:
                _effective_levels = _cfg_levels
            # v1.5.26 Phase 2C.3 -- regime-mode rung-drop attribution.
            # Bumps the counter by (cfg - effective) when the regime
            # cap pulls effective_levels below the configured ladder.
            # Floors via max() above means the floor-at-1 doesn't bump
            # for SHOCK's cap=0 (we still build one reducing-side
            # rung; the cap=0-vs-cap=1 design call is the 4G.10
            # follow-up). Per-cycle: counted once per tick, by the
            # number of rungs the cap shaved off the configured depth.
            if (
                _regime_levels_cap is not None
                and _effective_levels < _cfg_levels
            ):
                try:
                    self._state.ladder_rung_dropped_regime_mode_total += (
                        _cfg_levels - _effective_levels
                    )
                except Exception:  # noqa: BLE001
                    pass
            ladder_cfg = LadderConfig(
                num_levels_per_side=int(_effective_levels),
                offset_step=float(self._settings.ladder_offset_step),
                size_decay=float(self._settings.ladder_size_decay),
                inside_full_size=bool(self._settings.ladder_inside_full_size),
                gates_limit_levels=bool(self._settings.ladder_gates_limit_levels),
                batch_orders_enabled=bool(self._settings.ladder_batch_orders_enabled),
                # v1.4.169 Phase 2J — tick-floor knob. Defensive
                # ``getattr`` for Settings shapes from before v1.4.169.
                tick_floor_steps=int(
                    getattr(
                        self._settings, "ladder_tick_floor_steps", 1
                    )
                ),
                # v1.4.99 — inventory-aware rung pruning. DORMANT
                # default (the env knob default is False). See
                # ``app/ladder.py`` LadderConfig docstring and
                # ``plans/ladder-observability.md`` for the
                # calibration-driven flip criteria.
                inventory_aware_pruning_enabled=bool(
                    self._settings.ladder_inventory_aware_pruning_enabled
                ),
                inventory_aware_pruning_threshold_pct=float(
                    self._settings.ladder_inventory_aware_pruning_threshold_pct
                ),
            )
            # v1.4.79 ladder grid-collision dedup: pass the tick size
            # so ``build_ladder`` can predict downstream rounding and
            # skip outer rungs that would collapse to the inside
            # rung's grid value. Snapshot v1.4.78-260519-081151 caught
            # this with two same-side same-price orders on the book.
            _client_spec = getattr(self._exec, "_client", None)
            _spec = getattr(_client_spec, "symbol_spec", None) if _client_spec else None
            _tick_size = float(getattr(_spec, "price_tick", 0.0) or 0.0) if _spec else 0.0
            # v1.4.99 — inventory-aware pruning inputs. Read position
            # qty defensively (state.position may be None at startup).
            # max_abs_position from settings is always present.
            _pos = getattr(self._state, "position", None)
            _position_qty = float(_pos.position_qty) if (
                _pos is not None and _pos.position_qty is not None
            ) else None
            _max_abs_position = float(
                getattr(self._settings, "max_abs_position", 0.0) or 0.0
            ) or None
            # v1.4.100 ladder-observability F2 — drop-attribution
            # callback. Dispatches each rung-drop reason to the
            # matching ``BotState.ladder_rung_dropped_*_total``
            # counter. Defensive: unrecognised reasons land in the
            # ``other`` bucket so a future drop site without a
            # named category is still visible.
            _state_ref = self._state

            def _bump_rung_drop(reason: str) -> None:
                try:
                    if reason == "grid_collision":
                        _state_ref.ladder_rung_dropped_grid_collision_total += 1
                    elif reason == "min_notional":
                        _state_ref.ladder_rung_dropped_min_notional_total += 1
                    elif reason == "inventory_buffer":
                        _state_ref.ladder_rung_dropped_inventory_buffer_total += 1
                    elif reason == "inventory_aware_pruning":
                        _state_ref.ladder_rung_dropped_inventory_aware_pruning_total += 1
                    elif reason == "position_cap":
                        _state_ref.ladder_rung_dropped_position_cap_total += 1
                    elif reason == "in_flight":
                        _state_ref.ladder_rung_dropped_in_flight_total += 1
                    else:
                        _state_ref.ladder_rung_dropped_other_total += 1
                except Exception:
                    # Counter bump must never break the trading loop.
                    pass

            # v1.4.169 Phase 2J — tick-floor adjustment callback.
            # Fires when a rung's bps-derived price was shifted outward
            # to satisfy the tick-floor (rung was about to underflow
            # the 1-tick gap from inside). Per-side counters.
            def _bump_rung_floor(side_str: str) -> None:
                try:
                    if side_str == "bid":
                        _state_ref.ladder_rung_tick_floor_adjusted_bid_total += 1
                    elif side_str == "ask":
                        _state_ref.ladder_rung_tick_floor_adjusted_ask_total += 1
                except Exception:
                    pass

            cycle_ladder = build_ladder(
                decision=decision,
                cfg=ladder_cfg,
                # half_spread for the cycle = target_spread_bps / 2; this
                # already includes any adaptive widening / vol overlays
                # that compute_quote_decision baked into target_spread_bps.
                half_spread_bps=float(decision.target_spread_bps) / 2.0,
                gate_caps=self._collect_ladder_gate_caps(),
                tick_size=_tick_size if _tick_size > 0 else None,
                position_qty=_position_qty,
                max_abs_position=_max_abs_position,
                # v1.4.100 F2: routes rung-drop reasons to per-category
                # counters on BotState. See _bump_rung_drop above.
                on_rung_dropped=_bump_rung_drop,
                # v1.4.169 Phase 2J: per-side counter for the new
                # tick-floor adjustment.
                on_rung_floor_adjusted=_bump_rung_floor,
            )
            # Stash on state so observers (snapshot, telegram /status,
            # postmortem) can read the per-cycle ladder without
            # re-computing it.
            try:
                self._state.last_ladder_decision = cycle_ladder
            except Exception:
                # state attribute may not exist on older state objects;
                # the snapshot path already tolerates absence.
                pass

            # 1.2.34: finalize the spread-tab breakdown that
            # ``compute_quote_decision`` stamped onto ``decision.breakdown``.
            # Fill in the latched-state context the quoting function
            # couldn't see (adaptive_widen reason + remaining seconds,
            # final eligibility, ladder rungs) and stash on state so
            # ``snapshot_dict`` serialises it into ``state_current.json``.
            try:
                bk = getattr(decision, "breakdown", None)
                if bk is not None:
                    # v1.5.314: read through the bot's clock (SystemClock in
                    # prod, ReplayClock in backtest) instead of a raw
                    # ``time.monotonic()``. The writer
                    # ``adaptive_spread_widen_until_mono`` (see bot.py ~8085) is
                    # already stamped from ``self._clock.monotonic()``, so the
                    # remaining-seconds telemetry must read from the same clock
                    # to be consistent. Under SystemClock this is byte-identical
                    # to the old code; under ReplayClock it stops leaking
                    # wall-time into the backtest's adaptive-widen countdown.
                    now_mono = self._clock.monotonic()
                    aw_until = float(
                        getattr(self._state, "adaptive_spread_widen_until_mono", 0.0) or 0.0
                    )
                    aw_remaining = max(0.0, aw_until - now_mono)
                    aw_reason = getattr(
                        self._state, "adaptive_spread_widen_reason", None
                    )
                    # Build ladder rung dicts from cycle_ladder.
                    bids_list = (
                        [r.to_dict() for r in cycle_ladder.bids]
                        if cycle_ladder is not None
                        else []
                    )
                    asks_list = (
                        [r.to_dict() for r in cycle_ladder.asks]
                        if cycle_ladder is not None
                        else []
                    )
                    # QuoteBreakdownSnapshot is a frozen dataclass; use
                    # dataclasses.replace to update specific fields.
                    from dataclasses import replace as _replace
                    bk = _replace(
                        bk,
                        adaptive_widen_active=(aw_remaining > 0.0),
                        adaptive_widen_reason=aw_reason,
                        adaptive_widen_seconds_remaining=round(aw_remaining, 2),
                        quote_eligibility=str(decision.quote_eligibility or ""),
                        quote_eligibility_reason=str(
                            decision.quote_eligibility_reason or ""
                        ),
                        ladder_bids=bids_list,
                        ladder_asks=asks_list,
                        ladder_requested_levels=int(
                            cycle_ladder.requested_levels
                            if cycle_ladder is not None
                            else 1
                        ),
                        ladder_effective_levels_buy=int(
                            cycle_ladder.effective_levels_buy
                            if cycle_ladder is not None
                            else 0
                        ),
                        ladder_effective_levels_sell=int(
                            cycle_ladder.effective_levels_sell
                            if cycle_ladder is not None
                            else 0
                        ),
                    )
                    self._state.last_quote_breakdown = bk
                    # v1.5.217 — write-back of v1.5.207+ per-tick
                    # observability fields. The breakdown carries the
                    # values; both publishers (snapshot_dict + live_stats
                    # _ofi_block + _queue_aware_block) read from
                    # ``state.ofi_last_shift_bps`` /
                    # ``state.queue_position_ratio_{bid,ask}`` etc.
                    # Without this write-back the snapshot would always
                    # show stale defaults (0.0 / None / 1.0). Found
                    # 2026-05-28 while preparing Phase 3 of the A/B
                    # sequence — Phase 2 acceptance silently passed
                    # because ofi_last_shift_bps stayed at 0.0 even
                    # though the OFI alpha was actually contributing.
                    self._state.ofi_last_shift_bps = float(
                        getattr(bk, "ofi_shift_bps", 0.0) or 0.0
                    )
                    self._state.queue_position_ratio_bid = getattr(
                        bk, "queue_position_ratio_bid", None
                    )
                    self._state.queue_position_ratio_ask = getattr(
                        bk, "queue_position_ratio_ask", None
                    )
                    self._state.queue_size_mult_bid = float(
                        getattr(bk, "queue_size_mult_bid", 1.0) or 1.0
                    )
                    self._state.queue_size_mult_ask = float(
                        getattr(bk, "queue_size_mult_ask", 1.0) or 1.0
                    )
                    # Cumulative armed-counter increments. Bump when
                    # the per-side multiplier is meaningfully < 1.0
                    # (sizing armed) OR the inside-post step is > 0
                    # (inside-post armed). These counters are the
                    # acceptance-check signal that the feature is
                    # actually biting, not just dormant.
                    _qsmb = float(getattr(bk, "queue_size_mult_bid", 1.0) or 1.0)
                    _qsma = float(getattr(bk, "queue_size_mult_ask", 1.0) or 1.0)
                    _qibp = float(
                        getattr(bk, "queue_inside_post_step_bid_bps", 0.0) or 0.0
                    )
                    _qiap = float(
                        getattr(bk, "queue_inside_post_step_ask_bps", 0.0) or 0.0
                    )
                    if _qsmb < 1.0 - 1e-9:
                        self._state.queue_size_mult_armed_bid_total = int(
                            getattr(self._state, "queue_size_mult_armed_bid_total", 0) or 0
                        ) + 1
                    if _qsma < 1.0 - 1e-9:
                        self._state.queue_size_mult_armed_ask_total = int(
                            getattr(self._state, "queue_size_mult_armed_ask_total", 0) or 0
                        ) + 1
                    if _qibp > 1e-9:
                        self._state.queue_inside_post_active_bid = True
                        self._state.queue_inside_post_armed_bid_total = int(
                            getattr(self._state, "queue_inside_post_armed_bid_total", 0) or 0
                        ) + 1
                    else:
                        self._state.queue_inside_post_active_bid = False
                    if _qiap > 1e-9:
                        self._state.queue_inside_post_active_ask = True
                        self._state.queue_inside_post_armed_ask_total = int(
                            getattr(self._state, "queue_inside_post_armed_ask_total", 0) or 0
                        ) + 1
                    else:
                        self._state.queue_inside_post_active_ask = False
            except Exception:
                # Defensive: breakdown population is observability-only;
                # any failure here must not affect quoting.
                logger.exception("spread_tab_breakdown_finalize_failed")

            # v1.5.306 audit §5 P0 #2 — capture the AQC controller state
            # ONCE here so whichever quote_decisions row this tick writes
            # reflects a consistent start-of-tick snapshot: the state that
            # actually influenced THIS tick's decision (the controller is
            # not advanced until _update_active_quoting_controller() runs
            # below). Observability-only + best-effort (all-None on any
            # failure) so it can never affect the hot path.
            aqc_fields = self._aqc_quote_decision_fields()

            if risk.action == RiskAction.ALLOW:
                if defer_exec_telemetry:
                    deferred_allow_row = quote_decision_row(
                        decision, decision.quote_cycle_id, ladder=cycle_ladder
                    )
                    deferred_allow_row.update(aqc_fields)
                else:
                    _allow_row = quote_decision_row_with_exec_telemetry(
                        decision,
                        decision.quote_cycle_id,
                        self._exec,
                        risk,
                        ladder=cycle_ladder,
                    )
                    _allow_row.update(aqc_fields)
                    self._storage.insert_quote_decision(_allow_row)
                    quote_persisted = True
            elif (
                not self._settings.trading_enabled
                and risk.action == RiskAction.NO_QUOTE
                and "trading_disabled" in risk.reasons
            ):
                # Observability only: computed quote + risk gate; no orders (see trading_enabled check below).
                _obs_row = quote_decision_row_with_exec_telemetry(
                    decision,
                    decision.quote_cycle_id,
                    self._exec,
                    risk,
                    ladder=cycle_ladder,
                )
                _obs_row.update(aqc_fields)
                self._storage.insert_quote_decision(_obs_row)
                quote_persisted = True

            if self._settings.trading_enabled and self._state.bot_status == BotStatus.RUNNING:
                cancel_no_quote = (
                    risk.action == RiskAction.NO_QUOTE
                    and self._should_cancel_resting_on_no_quote_bound(risk.reasons)
                )
                # 2026-05-12 codex-#1: also cancel-resting when quote
                # eligibility says HOLD_ALL. Pre-fix the bot stopped
                # PLACING new quotes on HOLD_ALL but left existing
                # ones in the book — 8-13 % of fills in recent OKX
                # snapshots arrived during HOLD_ALL state. Each carried
                # the venue's structural -2.17 bp markout, so the
                # leakage was material. Allow-list keeps a small set
                # of brief / transient HOLD_ALL reasons (default just
                # ``recovery_cooldown``) where cancel would just churn
                # an order that's about to be re-placed.
                if (
                    not cancel_no_quote
                    and self._settings.cancel_resting_on_hold_all
                    and str(decision.quote_eligibility or "") == "HOLD_ALL"
                ):
                    cancel_no_quote = self._hold_all_should_cancel(
                        decision.quote_eligibility_reason or ""
                    )
                # TODO-004: accumulate "blind but still resting" exposure.
                # Sample once per tick before maybe_refresh_quotes runs, so a
                # NO_QUOTE that doesn't cancel resting (benign reasons:
                # trade_rate_limit etc.) still counts as exposure for the
                # tiny pre-cancel window. After BUG-009 the only NO_QUOTE
                # paths that leave orders resting are intentionally benign.
                self._state.note_blind_resting_sample(
                    blind=(
                        risk.action == RiskAction.NO_QUOTE
                        and self._state.has_resting_passive_order()
                    ),
                    now_mono=self._clock.monotonic(),
                )
                # v1.5.277 / AQC Phase 1 — feed the controller per
                # tick. Compute rolling net edge per minute + rolling
                # markout 5s mean from the recent fills. Observe-only:
                # the controller advances its state and publishes
                # diagnostics, but no downstream code reads its
                # aggression_level for trading decisions in Phase 1.
                # Best-effort: any exception is swallowed so the AQC
                # can never affect the hot path.
                try:
                    self._update_active_quoting_controller()
                except Exception:
                    logger.exception("aqc_update_failed")
                self._exec.maybe_refresh_quotes(
                    decision,
                    risk.action,
                    risk.bid_size_mult,
                    risk.ask_size_mult,
                    risk.spread_add_bps,
                    cancel_on_no_quote=cancel_no_quote,
                )
                # 2026-05-12 codex-#3: stamp the post-engine
                # executable half-spread back into the breakdown.
                # The engine's normal-MM market cap may have
                # tightened the model spread; this captures the
                # actual placed value so dashboard analytics + fill
                # records reflect reality, not the pre-cap target.
                try:
                    self._stamp_executable_half_spread_on_breakdown(decision)
                except Exception:
                    logger.exception(
                        "spread_tab_executable_half_spread_stamp_failed"
                    )
            elif self._settings.trading_enabled and self._state.bot_status == BotStatus.STARTING:
                if (
                    risk.action != RiskAction.NO_QUOTE
                    or self._should_cancel_resting_on_no_quote_bound(risk.reasons)
                ):
                    self._exec.cancel_resting_for_risk(risk.action)
            if self._settings.trading_enabled and self._state.bot_status in (
                BotStatus.RUNNING,
                BotStatus.STARTING,
                BotStatus.RECOVERING_MARKET_DATA,
            ):
                self._exec.maybe_sync_open_orders()
            self._maybe_promote_starting_to_running(healthy)
            tel = self._exec.get_quote_exec_telemetry()
            lat["latency_order_maintenance_local_ms"] = tel.get("placement_mode_eval_ms")
            lat["latency_quote_engine_build_ms"] = self._exec.last_tick_quote_engine_build_ms
            lat["latency_order_submit_rtt_ms"] = self._exec.last_tick_order_submit_rtt_ms
            lat["latency_reconcile_rest_ms"] = self._exec.last_tick_reconcile_rest_ms

            if deferred_allow_row is not None:
                deferred_allow_row.update(self._exec.get_quote_exec_telemetry())
                self._storage.insert_quote_decision(deferred_allow_row)
                quote_persisted = True

            if quote_persisted and not self._logged_first_quote_persisted:
                self._logged_first_quote_persisted = True
                log_extra(
                    logger,
                    logging.INFO,
                    "first quote decision persisted after startup",
                    {
                        "event": "first_quote_decision_persisted",
                        "symbol": self._settings.symbol,
                        "quote_cycle_id": decision.quote_cycle_id,
                    },
                )

            self._persist_snapshots()
            self._maybe_save_persistent_runtime_state()
        finally:
            try:
                with self._state._lock:
                    m_end = self._state.market
                    mid_end = m_end.mid_price if m_end else None
                process_pending_markouts(
                    self._state, self._storage, self._clock.now_utc(), mid_end, max_jobs=48
                )
            except Exception:
                logger.exception("process_pending_markouts_failed")
            self._state.note_public_bbo_burst_max_for_tick()
            self._apply_tick_latency(lat)
            self._maybe_periodic_bot_heartbeat()

    def run_forever(self) -> None:
        self._log_event(
            EventSeverity.INFO,
            "bot_start",
            "bot loop started (STARTING until first healthy reconcile; inherit exchange inventory; no startup flatten)",
        )
        # Start the independent fill-drain thread BEFORE the quote loop.
        # Runs until ``Bot.stop()``; specifically survives ``Bot.kill()``
        # so exchange events that land after kill still hit ``trading.db``.
        self._start_drain_thread()
        interval = float(self._settings.quote_loop_seconds)
        self._state.wake_quote_loop()
        while not self._stop.is_set():
            if self._state.quote_wake_event.wait(timeout=interval):
                self._state.quote_wake_event.clear()
            if self._stop.is_set():
                break
            try:
                self.one_tick()
                with self._state._lock:
                    self._state.loop_counter += 1
                    self._state.mark_heartbeat()
            except Exception:
                logger.exception("bot tick failed")
                self._state.bump_execution_errors("bot_tick_exception")
        self._log_event(EventSeverity.INFO, "bot_stop", "bot loop stopped")


def start_bot_thread(bot: Bot) -> tuple[threading.Thread, Callable[[], None]]:
    t = threading.Thread(target=bot.run_forever, name="mm-bot", daemon=True)
    t.start()
    return t, bot.stop
