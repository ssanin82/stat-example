"""Periodic S3 publisher for live trading internals.

Distinct from ``heartbeat.py``: heartbeat at 30 s carries slow-moving
operator-status fields (kill_reason, leverage, version). This module
publishes the *fast-moving* trading internals (working bid/ask,
microprice, basis, inventory skew, recent markouts) at a 5 s default
cadence so the dashboard's Bot Stats tab can render near-real-time
without polling the bot's HTTP API directly.

S3 path: ``s3://<logs_bucket>/live_stats/<profile>.json``.

Trading impact: zero. Same daemon-thread pattern as heartbeat; the
state snapshot under lock is microseconds, the JSON build is
sub-millisecond, the boto3 PUT is on a separate connection pool. The
trading loop never waits on this.

Failure semantics: best-effort. Boto3 ImportError, IAM denial, S3
network glitches all log a warning (with backoff) and the publisher
keeps trying; the bot continues trading regardless.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from app import __version__
from app.config import Settings
from app.enums import Side
from app.state import BotState

from app import clock as _clock

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return _clock.now_utc().isoformat()


def _round_or_none(v: Any, ndigits: int = 6) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return round(f, ndigits)


class LiveStatsPublisher:
    """Daemon-thread S3 writer for live trading internals."""

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        bucket: str,
        profile_name: str,
        storage: Any = None,
    ) -> None:
        self._settings = settings
        self._state = state
        self._bucket = bucket.strip()
        self._profile_name = (profile_name or "unknown").strip() or "unknown"
        self._key = f"live_stats/{self._profile_name}.json"
        self._interval_s = float(settings.live_stats_interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._consecutive_errors = 0
        self._client: Any = None
        # Optional Storage handle. When provided, the publisher
        # includes the soft-flatten attribution block in the payload
        # (recent SF episodes + tag maps from orders/fills tables).
        # Plan ref: plans/20260507-sf-frontend.md Phase 2.
        self._storage = storage

    @classmethod
    def maybe_create(
        cls,
        settings: Settings,
        state: BotState,
        profile_name: str,
        storage: Any = None,
    ) -> Optional["LiveStatsPublisher"]:
        if not bool(settings.live_stats_enabled):
            logger.info("live_stats_disabled reason=LIVE_STATS_ENABLED=false")
            return None
        bucket = (settings.logs_bucket or "").strip()
        if not bucket:
            logger.info(
                "live_stats_disabled reason=no_logs_bucket "
                "set_LOGS_BUCKET_to_enable"
            )
            return None
        return cls(settings, state, bucket, profile_name, storage=storage)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import boto3  # type: ignore[import-untyped]

            self._client = boto3.client("s3")
        except Exception as e:
            logger.warning(
                "live_stats_disabled reason=boto3_init_failed err=%s",
                str(e)[:200],
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="live_stats", daemon=True
        )
        self._thread.start()
        logger.info(
            "live_stats_started bucket=%s key=%s interval_s=%.1f",
            self._bucket,
            self._key,
            self._interval_s,
        )

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        # Publish once immediately so the dashboard's first poll lands
        # a fresh file rather than a 5 s wait.
        self._safe_publish_once()
        while not self._stop.wait(self._interval_s):
            self._safe_publish_once()

    def _safe_publish_once(self) -> None:
        try:
            self._publish_once()
            self._consecutive_errors = 0
        except Exception as e:
            self._consecutive_errors += 1
            level = (
                logging.WARNING
                if self._consecutive_errors <= 3
                else logging.DEBUG
            )
            logger.log(
                level,
                "live_stats_publish_failed errs=%d err=%s",
                self._consecutive_errors,
                str(e)[:200],
            )

    def _publish_once(self) -> None:
        if self._client is None:
            return
        body = self._build_payload()
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key,
            Body=json.dumps(body, indent=2).encode("utf-8"),
            ContentType="application/json",
            CacheControl="no-store, max-age=0",
        )

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    def _build_payload(self) -> dict[str, Any]:
        # All state reads under one lock to avoid mid-mutation tearing
        # (e.g. seeing a working_bid that's been replaced but a
        # market that's already moved past it).
        with self._state._lock:
            sym = self._state.symbol
            # v1.4.194: migrated off the deprecated property shims.
            wo_bid = self._state.get_working_order(Side.BUY, 0)
            wo_ask = self._state.get_working_order(Side.SELL, 0)
            market = self._state.market
            position = self._state.position
            binance_bid = self._state.binance_best_bid
            binance_ask = self._state.binance_best_ask
            binance_mid = self._state.binance_mid
            binance_bid_size = self._state.binance_bid_size
            binance_ask_size = self._state.binance_ask_size
            basis_ewma = self._state.binance_basis_ewma
            recent_fills = list(self._state.recent_fills)
            ob_imbalance = self._state.ob_imbalance_ewma
            vol_bps = self._state.vol_bps
            tox_snap = self._state.toxicity
            jd_snap = self._state.join_depth_controller.snapshot()
            # Capture the full PnlSnapshot under the lock so the
            # session_pnl block below sees one consistent view.
            # PnlTracker is the single venue-independent source of
            # truth for session PnL — frontend should display
            # ``session_pnl.total_usd`` directly rather than
            # re-deriving from FillHistory.
            pnl_snapshot = self._state.pnl
            # v1.4.95 — connectivity counters snapshot. Read INSIDE the
            # lock so the four gone_on_exchange counters + the recent-
            # events ring are taken in one consistent view. The dashboard
            # exposes these in a dedicated tile; the postmortem fatal
            # gate ``check_gone_on_exchange_zero`` reads from the same
            # state (via ``state_current.json``). Five pre-existing
            # counters (``cancel_unexpected_gone_total``,
            # ``cancel_race_lost_to_fill_total``,
            # ``reconcile_skip_snapshot_stale_total``,
            # ``place_cancel_race_total``,
            # ``hydration_skipped_recently_terminal_total``) are also
            # surfaced — none were previously published to live_stats
            # despite being maintained by the executor. Operator's
            # "lying statistics" complaint applies to all of them.
            connectivity_counters = {
                # ----- TIER 1 (FATAL): gone_on_exchange -----
                # Bot has no HTTP confirmation that its place/cancel
                # reached the venue. Sub-categorised by which
                # confirmation is missing. Any non-zero value is a
                # postmortem fatal acceptance failure.
                "gone_on_exchange_total": int(
                    self._state.gone_on_exchange_total
                ),
                "gone_on_exchange_phantom_no_ack_total": int(
                    self._state.gone_on_exchange_phantom_no_ack_total
                ),
                "gone_on_exchange_acked_no_cancel_total": int(
                    self._state.gone_on_exchange_acked_no_cancel_total
                ),
                "gone_on_exchange_cancel_no_http_confirm_total": int(
                    self._state.gone_on_exchange_cancel_no_http_confirm_total
                ),
                "gone_on_exchange_other_total": int(
                    self._state.gone_on_exchange_other_total
                ),
                "gone_on_exchange_recent": list(
                    self._state.gone_on_exchange_recent
                ),
                # ----- TIER 2 (WARN): http_acked_no_ws -----
                # Cancel HTTP succeeded, venue confirmed the cancel
                # landed, but the WS `canceled` terminal event never
                # arrived in time. Tolerable when rare. The 6 events
                # in snapshot v1.4.92-260519-161411 (previously
                # misclassified as gone_on_exchange) belong here.
                "http_acked_no_ws_total": int(
                    self._state.http_acked_no_ws_total
                ),
                "http_acked_no_ws_recent": list(
                    self._state.http_acked_no_ws_recent
                ),
                "http_acked_no_ws_lateness_ms_min": (
                    float(self._state.http_acked_no_ws_lateness_ms_min)
                    if self._state.http_acked_no_ws_lateness_ms_min
                    is not None
                    else None
                ),
                "http_acked_no_ws_lateness_ms_max": (
                    float(self._state.http_acked_no_ws_lateness_ms_max)
                    if self._state.http_acked_no_ws_lateness_ms_max
                    is not None
                    else None
                ),
                "http_acked_no_ws_lateness_p50_ms": (
                    self._state._ws_lateness_percentile_unlocked(0.50)
                ),
                "http_acked_no_ws_lateness_p95_ms": (
                    self._state._ws_lateness_percentile_unlocked(0.95)
                ),
                # ----- TIER 3 (WARN): ws_arrived_late -----
                # WS terminal event eventually arrived for an oid the
                # bot had already cleaned up via reconcile. The WS
                # DID deliver — just past the reconcile cycle.
                # Tolerable when rare.
                "ws_arrived_late_total": int(
                    self._state.ws_arrived_late_total
                ),
                "ws_arrived_late_recent": list(
                    self._state.ws_arrived_late_recent
                ),
                # v1.4.100 ladder-observability F2 — drop-attribution
                # counters surface in live_stats for the dashboard
                # D3 chip. See plans/ladder-observability.md F2 for
                # taxonomy.
                "ladder_rung_drops": {
                    "grid_collision_total": int(
                        self._state.ladder_rung_dropped_grid_collision_total
                    ),
                    "min_notional_total": int(
                        self._state.ladder_rung_dropped_min_notional_total
                    ),
                    "inventory_buffer_total": int(
                        self._state.ladder_rung_dropped_inventory_buffer_total
                    ),
                    "inventory_aware_pruning_total": int(
                        self._state.ladder_rung_dropped_inventory_aware_pruning_total
                    ),
                    "position_cap_total": int(
                        self._state.ladder_rung_dropped_position_cap_total
                    ),
                    "in_flight_total": int(
                        self._state.ladder_rung_dropped_in_flight_total
                    ),
                    "other_total": int(
                        self._state.ladder_rung_dropped_other_total
                    ),
                },
                # Pre-existing counters surfaced for the first time so
                # the dashboard never has to re-derive them from rolling-
                # window aggregations.
                "cancel_unexpected_gone_total": int(
                    self._state.cancel_unexpected_gone_total
                ),
                "cancel_race_lost_to_fill_total": int(
                    self._state.cancel_race_lost_to_fill_total
                ),
                "reconcile_skip_snapshot_stale_total": int(
                    self._state.reconcile_skip_snapshot_stale_total
                ),
                "place_cancel_race_total": int(
                    self._state.place_cancel_race_total
                ),
                "hydration_skipped_recently_terminal_total": int(
                    self._state.hydration_skipped_recently_terminal_total
                ),
                "private_ws_reconnect_count": int(
                    self._state.private_ws_reconnect_count
                ),
                "private_ws_last_connect_ts": (
                    self._state.private_ws_last_connect_wall_ts.isoformat()
                    if self._state.private_ws_last_connect_wall_ts is not None
                    else None
                ),
                # BUG-034 surfacing (v1.5.258). Counts the times REST
                # /positions disagreed with the bot's shadow position
                # by >0.5 lot — each is a WS-missed fill that the
                # REST path silently absorbed pre-fix. The recovery
                # counter tracks how many of those triggered the
                # automated REST /fills catch-up (post-fix, every
                # divergence should). If divergence keeps rising but
                # recovery doesn't, the bot lost the trigger logic.
                # If both rise but session_fill_count tracks
                # |position_changes| accurately, the catch-up is
                # working end-to-end.
                "shadow_position_divergence_count": int(
                    self._state.shadow_position_divergence_count
                ),
                "shadow_position_divergence_recovery_count": int(
                    self._state.shadow_position_divergence_recovery_count
                ),
                "shadow_position_last_divergence_qty": (
                    float(
                        self._state.shadow_position_last_divergence_qty
                    )
                    if self._state.shadow_position_last_divergence_qty
                    is not None
                    else None
                ),
            }

        # PnL attribution over the rolling fill window
        # (``state.recent_fills`` is capped at 1000). Realized is
        # computed inside ``_build_attribution_block`` from the
        # same window, so the decomposition equation
        # ``realized = rebate + markout + residual`` is internally
        # self-consistent regardless of session length. Session-
        # cumulative realized is exposed separately via the
        # ``session_pnl`` block below — operator should look there
        # for the headline number.
        attribution = _build_attribution_block(recent_fills)

        # v1.4.101 — per-horizon markout + attribution decomposition.
        # Sibling of ``attribution`` (which stays canonical 5s).
        # Computed from the SAME ``recent_fills`` window so all
        # horizons share the same denominator. The dashboard's
        # "Markout horizon ladder" tile reads this directly; it tells
        # the operator whether markouts trend favourably as the
        # holding period extends. Builds defensively — empty dict
        # on any failure / no fills.
        multi_horizon_attribution = _build_multi_horizon_attribution_block_from_fills(
            recent_fills
        )

        # Canonical session PnL block. Mirrors PnlTracker.build_snapshot:
        # ``total = realized + unrealized − fees_usd`` (fees signed:
        # positive = paid, negative = rebate). Same number across
        # venues; no FillHistory recomputation in the frontend.
        if pnl_snapshot is not None:
            session_pnl = {
                "realized_usd": _round_or_none(
                    pnl_snapshot.realized_pnl_usd, 6
                ),
                "unrealized_usd": _round_or_none(
                    pnl_snapshot.unrealized_pnl_usd, 6
                ),
                "fees_usd": _round_or_none(pnl_snapshot.fees_usd, 6),
                "total_usd": _round_or_none(pnl_snapshot.total_pnl_usd, 6),
                "drawdown_usd": _round_or_none(
                    pnl_snapshot.drawdown_usd, 6
                ),
                "session_peak_equity_usd": _round_or_none(
                    pnl_snapshot.session_peak_equity_usd, 6
                ),
            }
        else:
            session_pnl = {
                "realized_usd": None,
                "unrealized_usd": None,
                "fees_usd": None,
                "total_usd": None,
                "drawdown_usd": None,
                "session_peak_equity_usd": None,
            }

        # Public-WS gap stats. Surfaced in the dashboard's Latency panel
        # so the operator can see how often the venue's market data
        # ticks (min / median / p95 / max). On HYPE the natural cadence
        # is much slower than majors — visibility helps the operator
        # judge whether the bot is reacting fast enough or being held
        # back by the venue's update cadence. Reads from the existing
        # MarketDataGapTracker; no new state required.
        try:
            gap_stats = self._state.market_data_gap_tracker.to_api_dict(
                session_id=self._state.session_id, symbol=sym
            )
        except Exception:
            gap_stats = None
            logger.exception("live_stats_gap_stats_block_failed")

        # Reference-venue (Binance) gap stats. Same panel as OKX gaps
        # but separately rendered so the operator can compare the two
        # cadences side-by-side. OKX uses books5 (100ms throttle, L2);
        # Binance uses bookTicker (L1, event-driven, sub-10ms typical
        # median). The asymmetry matters: the bot's fair-value blend
        # reacts to whichever side moves first, but quote placement
        # is bottlenecked by the slower OKX cadence.
        gap_stats_reference: Optional[dict[str, Any]] = None
        try:
            tracker = self._state.binance_public_ws_timing
            if tracker is not None:
                summary = tracker.summary()
                exch_gaps = summary.get("exchange_gap_ms") or {}
                # Only emit when the tracker has at least one valid
                # gap; otherwise keep None so the frontend hides the
                # row instead of showing meaningless zeros.
                if (
                    isinstance(exch_gaps, dict)
                    and exch_gaps.get("median") is not None
                ):
                    valid_gaps = int(
                        summary.get("samples_with_valid_exchange_gap") or 0
                    )
                    total_samples = int(summary.get("total_samples") or 0)
                    gap_stats_reference = {
                        "source_type": "binance_public_ws",
                        "update_count": total_samples,
                        "gap_count": valid_gaps,
                        "min_gap_ms": exch_gaps.get("min"),
                        "max_gap_ms": exch_gaps.get("max"),
                        # PublicWsTimingTracker.summary() doesn't compute
                        # arithmetic mean (only quantiles). Keep None to
                        # match the shape; frontend renders "—".
                        "mean_gap_ms": None,
                        "median_gap_ms": exch_gaps.get("median"),
                        "p95_gap_ms": exch_gaps.get("p95"),
                        # Same: no last-gap field on the timing
                        # tracker. Could derive from latest sample but
                        # not worth the plumbing for this view.
                        "last_gap_ms": None,
                        "first_update_ts": None,
                        "last_update_ts": None,
                    }
        except Exception:
            gap_stats_reference = None
            logger.exception("live_stats_gap_stats_reference_block_failed")

        return {
            "schema_version": 1,
            "profile": self._profile_name,
            "symbol": sym,
            "version": __version__,
            "captured_at_utc": _now_iso(),
            "interval_seconds": self._interval_s,
            "working_bid": _working_order_dict(wo_bid),
            "working_ask": _working_order_dict(wo_ask),
            "trading_venue": _trading_venue_block(market),
            "reference_venue": _reference_venue_block(
                self._settings,
                binance_bid,
                binance_ask,
                binance_mid,
                binance_bid_size,
                binance_ask_size,
            ),
            "basis": _basis_block(market, binance_mid, basis_ewma),
            "inventory": _inventory_block(self._settings, position),
            "skew": _skew_block(self._settings, position),
            "markouts": _markouts_block(recent_fills),
            "book_signals": {
                "ob_imbalance_ewma": _round_or_none(ob_imbalance, 4),
                "vol_bps": _round_or_none(vol_bps, 3),
                "toxicity_score": _round_or_none(
                    getattr(tox_snap, "score", None), 4
                ),
                "toxicity_avg_adverse_markout_bps": _round_or_none(
                    getattr(tox_snap, "avg_adverse_markout_bps", None), 3
                ),
            },
            # Item 1: adaptive spread-widen cooldown surface for the
            # Market tab. ``active`` is True while the overlay is
            # gating the quote engine. ``seconds_remaining`` is wall-
            # clock-equivalent (derived from monotonic deadline so
            # it's robust across system clock jumps). ``reason``
            # categorises *why* the most recent overlay was armed.
            "adaptive_widen": _adaptive_widen_block(self._state),
            # Item 2: quote-quality rollup (avg quoted spread, post-only
            # rejects, suppression reasons, etc.). The dedicated
            # ``/telemetry/quote-quality`` endpoint also serves this
            # but the Market tab needs it on the same poll cadence as
            # everything else. Use the locked accessor on BotState
            # which threads in the realised-PnL + recent-fills + window
            # kwargs ``QuoteQualityRollup.to_dict`` requires.
            "quote_quality": self._state.quote_quality_dict(),
            # Item 3: server-side vol-regime tier classification with
            # the multiplicative shrink factor + half-spread bump
            # already in effect. Frontend renders the tier as a
            # coloured pill.
            "vol_regime": _vol_regime_block(self._state),
            # Item 4: per-side execution-desync detail. Internally
            # ``execution.py`` computes ``d_buy`` and ``d_sell``
            # separately; this surfaces both alongside the OR'd
            # ``order_desync`` and ``desync_phase`` already in
            # ``/state/current``.
            "desync": _desync_block(self._state),
            # 1.2.2: basis-regime classifier snapshot (sign + IC +
            # pair_count) so the dashboard's Market tab can render
            # the regime state without polling /state/current
            # (which is loopback-only). Frontend reads this for the
            # "basis regime" row of the Engine State card.
            "basis_regime": _basis_regime_block(self._state),
            # 1.2.2: flow_score snapshot (TFI signed normalised +
            # streak counts + buy/sell toxic scores). Same rationale
            # as basis_regime — surfacing for the Engine State card.
            "flow_score": _flow_score_block(self._state),
            # 1.2.2: vol × trend conjunction gate state. None of
            # the active-gate fields previously surfaced; this gives
            # the Market tab the cooldown timer + last-trigger
            # diagnostic.
            "vol_trend_gate": _vol_trend_gate_block(self._state),
            # v1.4.115 Phase 1E.1.b — regime_controller mode label.
            # Top-level block so the dashboard's Alerts chip + Detectors
            # card can read it directly without digging into
            # behavioural_gates. Renders unconditionally (per always-
            # render contract); pre-v1.4.112 bots emit ``None``.
            "regime_mode": _regime_mode_block(self._state),
            # v1.4.115 Phase 1E.1 — Phase 1B shock_gate state, top-level
            # for the Detectors card. Same shape as the behavioural_gates
            # block but surfaced at top level for parity with the other
            # regime-stack blocks above.
            "shock_gate": _shock_gate_block(self._state),
            # v1.4.115 Phase 1E.1 — Phase 1D post-reduction cooldown
            # state. Top-level for the Detectors card.
            "post_reduction_cooldown": _post_reduction_cooldown_block(
                self._state
            ),
            # v1.4.116 Phase 1E.3.f — per-quoting-feature firing rate
            # over a rolling 60 s window. Drives the Detectors card's
            # section 5 ("Quoting features"). Each entry is a fire
            # count, not an ON/OFF state — the operator wants
            # frequency for the features that produce per-tick
            # suppression rather than sustained cooldowns.
            "feature_firing_rates": _feature_firing_rates_block(
                self._state
            ),
            # 1.2.1: tiered session-PnL drawdown ladder state.
            "session_drawdown": _session_drawdown_block(self._state),
            # v1.5.202: SF-event-count fatigue ladder. Orthogonal
            # to session_drawdown (count- vs PnL-based). Surfaced as
            # a top-level block so the dashboard can render an "SF
            # tier" chip next to the "DD tier" chip without digging
            # into the full behavioural_gates dict.
            "sf_fatigue": _sf_fatigue_block(self._settings, self._state),
            # v1.5.207 Phase 4C.5 — participation score per side
            # ([0,1] continuous; 1=quote, 0=refused). Surfaces both
            # the current tick and the rolling-window mean so the
            # dashboard can render a per-side chip + sparkline.
            "participation_score": _participation_score_block(
                self._settings, self._state,
            ),
            # v1.5.209 Phase 8D — Order Flow Imbalance accumulator
            # state. Published unconditionally (the accumulator runs
            # even when alpha=0.0 dormant) so the operator can audit
            # calibration before flipping OFI_RESERVATION_ALPHA from
            # 0.0 to a non-zero value.
            "ofi": _ofi_block(self._settings, self._state),
            # v1.5.209 Phase 8B — Queue-position-aware sizing +
            # inside-post telemetry. Per-side ratios + multipliers +
            # cumulative-active counters.
            "queue_aware": _queue_aware_block(self._settings, self._state),
            # 1.1.131: post-swing PnL cooldown gate state. Surfaced
            # for the Market tab "Gates" card so the operator sees
            # the cooldown active flag + last-trigger reason
            # (spike/drop) at a glance.
            "post_swing_gate": _post_swing_gate_block(self._state),
            # 1.2.41: quote-eligibility recovery-cooldown countdown.
            # When eligibility was recently more restrictive (e.g.
            # HOLD_ALL) and is now improving, a clamp keeps it at the
            # most-restrictive level for ``QUOTE_HOLD_COOLDOWN_MS``
            # before letting it fully recover. Surfaced here so the
            # dashboard's TradingModeChip can show "recovery_cooldown
            # 0.7s remaining" instead of just the static reason
            # string. Distinct from ``adaptive_widen`` which is a
            # spread-floor overlay, not an eligibility clamp.
            "quote_eligibility_recovery": _quote_elig_recovery_block(
                self._state
            ),
            # v1.5.199 — full behavioural_gates dict, mirroring the
            # state_current.json structure. Pre-v1.5.199 the live_stats
            # payload only exposed a small subset of gates as top-level
            # blocks (shock_gate, post_reduction_cooldown, vol_trend_gate,
            # post_swing_gate, session_drawdown, quote_eligibility_recovery,
            # adaptive_widen, basis_regime, regime_mode). 11 gates that
            # have been added since v1.4.116 (mae_gate, structural_bias_
            # throttle, at_touch_adverse_pause, realised_edge_side_suppress,
            # vol_regime_auto_pause, target_venue_fast_move_cancel,
            # expected_edge_side_refuse, negative_expectancy_dampen,
            # sf_slice, fill_burst_detector, vol_spike) were invisible
            # in the Detectors card despite firing during real
            # production failure modes (v1.5.193 structural_bias_throttle
            # lock-up, v1.5.195 mae_gate / SF stuck states).
            #
            # Publishing the full block once here lets the Detectors
            # card render any gate without per-gate publisher wiring.
            # ``_behavioural_gates_snapshot`` is the same builder
            # ``state_current.json`` uses, so the shape is identical.
            "behavioural_gates": _behavioural_gates_snapshot_for_live_stats(
                self._state
            ),
            "session_pnl": session_pnl,
            "pnl_attribution": attribution,
            # v1.4.101 — per-horizon markout + attribution decomposition.
            # Mirrors the ``markouts_multi_horizon`` block in
            # ``session_summary.json`` so the dashboard's "Markout
            # horizon ladder" tile can read either source with the
            # same parser. Empty dict on legacy bots / empty windows.
            "markouts_multi_horizon": multi_horizon_attribution,
            # v1.4.95: session-cumulative connectivity counters +
            # bounded ring of recent gone_on_exchange events. The
            # dashboard's Connectivity tab renders a dedicated tile
            # from this block. Contract: gone_on_exchange_total
            # MUST stay at zero on a clean session; the postmortem
            # `wedge_acceptance.check_gone_on_exchange_zero` makes
            # any non-zero value a fatal acceptance failure.
            "connectivity_counters": connectivity_counters,
            "join_depth_autotune": {
                "enabled": bool(jd_snap.enabled),
                "overlay_bps": _round_or_none(jd_snap.overlay_bps, 3),
                "target_overlay_bps": _round_or_none(
                    jd_snap.target_overlay_bps, 3
                ),
                "rejects_per_min": _round_or_none(jd_snap.rejects_per_min, 2),
                "median_1s_markout_bps": _round_or_none(
                    jd_snap.median_1s_markout_bps, 3
                ),
                "median_5s_markout_bps": _round_or_none(
                    jd_snap.median_5s_markout_bps, 3
                ),
                "trades_per_min": _round_or_none(jd_snap.trades_per_min, 2),
                "contrib_reject_bps": _round_or_none(
                    jd_snap.contrib_reject_bps, 3
                ),
                "contrib_markout_1s_bps": _round_or_none(
                    jd_snap.contrib_markout_1s_bps, 3
                ),
                "contrib_markout_5s_bps": _round_or_none(
                    jd_snap.contrib_markout_5s_bps, 3
                ),
                "contrib_underfill_bps": _round_or_none(
                    jd_snap.contrib_underfill_bps, 3
                ),
                "saturated_seconds": _round_or_none(
                    jd_snap.saturated_seconds, 1
                ),
                "last_update_iso": jd_snap.last_update_iso,
            },
            # v1.4.36 (Codex #3 + #7): scope the attribution block to
            # the current session. SF episodes persist in the DB across
            # sessions by design (historical record for postmortem +
            # equity-history overlays), but the live publication MUST
            # show only current-session episodes — the frontend labels
            # this block "events this session" and a cross-session leak
            # corrupts the count + the operator's mental model.
            "soft_flatten_attribution": _soft_flatten_attribution_block(
                self._storage,
                session_started_at_iso=(
                    self._state.session_started_at_utc.isoformat()
                    if getattr(self._state, "session_started_at_utc", None)
                    is not None
                    else None
                ),
            ),
            # v1.5.33 — take-profit attribution. Same shape as the SF
            # block; frontend reads ``tp_attribution.events`` for the
            # PnL sub-band markers and the ``order_tags`` / ``fill_tags``
            # for the per-row TP#N badges.
            "tp_attribution": _tp_attribution_block(
                self._storage,
                session_started_at_iso=(
                    self._state.session_started_at_utc.isoformat()
                    if getattr(self._state, "session_started_at_utc", None)
                    is not None
                    else None
                ),
            ),
            # TP telemetry counters surfaced for the dashboard.
            "tp_telemetry": {
                "enabled": bool(
                    getattr(
                        self._settings, "upnl_harvest_enabled", False
                    )
                ),
                "trigger_bps": float(
                    getattr(
                        self._settings,
                        "upnl_harvest_trigger_bps",
                        0.0,
                    )
                ),
                "disarm_margin_bps": float(
                    getattr(
                        self._settings,
                        "upnl_harvest_disarm_margin_bps",
                        0.0,
                    )
                ),
                "active": bool(
                    getattr(self._state, "tp_active", False)
                ),
                "armed_total": int(
                    getattr(self._state, "tp_armed_total", 0)
                ),
                "filled_total": int(
                    getattr(self._state, "tp_filled_total", 0)
                ),
                "exited_unfilled_total": int(
                    getattr(
                        self._state, "tp_exited_unfilled_total", 0
                    )
                ),
                "exited_sf_takeover_total": int(
                    getattr(
                        self._state,
                        "tp_exited_sf_takeover_total",
                        0,
                    )
                ),
            },
            # todo-006: quote-age fill bucketing. Aggregates the last 100
            # fills into 5 buckets by how long the order rested before
            # being hit. Fast (<100ms) fills typically toxic; slow
            # (>2s) fills typically passive. Frontend Bot Stats
            # renders this as a bar chart.
            "fill_buckets": self._state.fill_buckets.to_dict(),
            # v1.4.100 ladder-observability D2: per-rung sliding-window
            # fill counts (1-min + 5-min windows). Tiny live chip in
            # the Bot Stats card: "rung-0 = 8/min, rung-1 = 1/min".
            # Pre-v1.4.100 fills lack ``level_idx`` so they roll into
            # the L0 bucket — same as single-rung behaviour. Cheap:
            # iterates ``recent_fills`` once.
            "ladder_per_rung_fills": _ladder_per_rung_fills_block(
                recent_fills, now_utc=_clock.now_utc()
            ),
            # v1.4.100 ladder-observability D1: snapshot of currently-
            # active working orders grouped by ``level_idx``. Drives
            # the working-ladder dashboard view that replaces the
            # legacy scalar bid/ask display.
            "working_ladder": _working_ladder_block(
                self._state, now_utc=_clock.now_utc()
            ),
            # Public-WS market-data gap stats (min/median/p95/max in ms).
            # Surfaced in the Latency panel.
            "gap_stats": gap_stats,
            # Reference-venue (Binance) gap stats — second row in the
            # Latency panel. Lets the operator see the cadence
            # asymmetry (OKX 100ms-throttled L2 vs Binance L1
            # event-driven).
            "gap_stats_reference": gap_stats_reference,
            # 1.2.34: per-cycle quote breakdown for the dashboard's
            # Spread tab. Captures every contribution to the
            # current cycle's quote (reservation shifts, half-
            # spread stack, size mult chain). ``None`` until the
            # first quote cycle has run. Stamped by the bot at
            # the end of each cycle onto ``state.last_quote_breakdown``.
            #
            # v1.4.176: the dashboard's Bot-Stats "Spread composition"
            # card reads ``p.quote_breakdown.spread_composition`` —
            # but that field never existed in ``QuoteBreakdownSnapshot``;
            # it's a separate per-tick artifact computed by
            # ``build_spread_composition`` and stashed on
            # ``state.last_spread_composition``. Pre-v1.4.176 the card
            # was effectively dead from v1.4.13 onward (visible as
            # "not in payload — bot on pre-v1.4.13 build"). Wire it
            # in here by merging the composition dict into the
            # quote_breakdown sub-object so the existing frontend
            # path resolves without changes.
            "quote_breakdown": (
                {
                    **self._state.last_quote_breakdown.to_dict(),
                    "spread_composition": (
                        self._state.last_spread_composition.to_dict()
                        if getattr(
                            self._state, "last_spread_composition", None
                        )
                        is not None
                        and hasattr(
                            self._state.last_spread_composition, "to_dict"
                        )
                        else None
                    ),
                }
                if getattr(self._state, "last_quote_breakdown", None) is not None
                else None
            ),
            # 1.3.31 (todo-029 § 6c): session-cumulative cross-venue
            # cancel count. Bumped in ``execution.py``'s
            # ``binance_cross_venue_cancel`` branch every time a
            # working order is cancelled because Binance's mid moved
            # the basis-shifted fair value past the configured tick
            # threshold relative to the resting order. Surfaced for
            # the Market tab cross-venue card so the operator can
            # see "this session: N cancels driven by Binance leading
            # the touch by more than the threshold" at a glance.
            "session_cross_venue_cancel_count": int(
                getattr(
                    self._state, "session_cross_venue_cancel_count", 0
                )
                or 0
            ),
            # v1.4.21: companion counter for cross-venue AMENDs (vs
            # cancels). Surfaced so the dashboard's Connectivity panel
            # can show "amend won vs cancel won" for each Binance
            # cross-venue trigger — when ratio is amend-dominant the
            # rate-limit pools see ~50% less consumption per reprice.
            "session_cross_venue_amend_count": int(
                getattr(
                    self._state, "session_cross_venue_amend_count", 0
                )
                or 0
            ),
            # v1.4.20 rate-limit-observability Phase 1+2 + v1.4.22
            # per-second stats extension: per-endpoint pool rate
            # snapshot. Keyed by pool name (place_batch / cancel_batch
            # / amend_batch / place_single / cancel_single /
            # amend_single / reads / aggregate). Each value carries
            # current_2s, peak_2s_60s, total, cap, pct_of_cap, plus
            # rate_per_sec_min / _max / _median / _p95 / _samples.
            # The bot's heartbeat handler updates
            # ``state.okx_rate_window_per_pool`` from the OKX adapter's
            # ``rest_runtime_counters()`` each tick; this publish makes
            # it available to the dashboard Connectivity panel.
            "okx_rate_window_per_pool": dict(
                getattr(self._state, "okx_rate_window_per_pool", {}) or {}
            ),
            # 1.3.59: cancel-race-lost-to-fill counter. Incremented on
            # every venue cancel response classified as
            # ``benign_missing`` — meaning the bot's cancel arrived
            # AFTER the fill / cancel had already settled. Each event
            # is a cancel-race-exposure window. Surfaced so the
            # dashboard's Execution-quality card can show a live
            # session count.
            "cancel_race_lost_to_fill_total": int(
                getattr(
                    self._state, "cancel_race_lost_to_fill_total", 0
                )
                or 0
            ),
        }


def _working_order_dict(wo: Any) -> Optional[dict[str, Any]]:
    if wo is None:
        return None
    status = getattr(wo, "status", None)
    status_str = getattr(status, "value", str(status) if status else None)
    # Filter out closed-status orders so the panel doesn't render a
    # stale price for an already-cancelled order.
    if status_str in (
        None,
        "FILLED",
        "CANCELED",
        "REJECTED",
        "DESYNC",
    ):
        return None
    return {
        "price": _round_or_none(getattr(wo, "price", None), 8),
        "size": _round_or_none(getattr(wo, "size", None), 6),
        "status": status_str,
        "exchange_oid": (
            str(getattr(wo, "order_id_exchange", None) or "")
            if getattr(wo, "order_id_exchange", None) is not None
            else None
        ),
    }


def _trading_venue_block(market: Any) -> dict[str, Any]:
    if market is None:
        return {
            "best_bid": None,
            "best_ask": None,
            "mid": None,
            "microprice": None,
            "spread_bps": None,
            "bid_size": None,
            "ask_size": None,
        }
    best_bid = getattr(market, "best_bid", None)
    best_ask = getattr(market, "best_ask", None)
    bid_size = getattr(market, "bid_size", None)
    ask_size = getattr(market, "ask_size", None)
    mid = getattr(market, "mid_price", None)
    spread_bps = getattr(market, "spread_bps", None)
    return {
        "best_bid": _round_or_none(best_bid, 8),
        "best_ask": _round_or_none(best_ask, 8),
        "mid": _round_or_none(mid, 8),
        "microprice": _round_or_none(
            _compute_microprice(best_bid, best_ask, bid_size, ask_size), 8
        ),
        "spread_bps": _round_or_none(spread_bps, 3),
        "bid_size": _round_or_none(bid_size, 4),
        "ask_size": _round_or_none(ask_size, 4),
    }


def _reference_venue_block(
    settings: Settings,
    bid: Optional[float],
    ask: Optional[float],
    mid: Optional[float],
    bid_size: Optional[float],
    ask_size: Optional[float],
) -> dict[str, Any]:
    return {
        "name": settings.reference_exchange or "off",
        "symbol": getattr(settings, "binance_symbol", None) or settings.symbol,
        "best_bid": _round_or_none(bid, 8),
        "best_ask": _round_or_none(ask, 8),
        "mid": _round_or_none(mid, 8),
        "microprice": _round_or_none(
            _compute_microprice(bid, ask, bid_size, ask_size), 8
        ),
        "bid_size": _round_or_none(bid_size, 4),
        "ask_size": _round_or_none(ask_size, 4),
    }


def _adaptive_widen_block(state: Any) -> dict[str, Any]:
    """Render the adaptive spread-widen cooldown for the Market tab.

    Returns ``active`` (whether the overlay is currently gating the
    engine), ``seconds_remaining`` (wall-equivalent countdown derived
    from the monotonic deadline), ``reason`` (categorical trigger),
    and ``quote_quality_latched`` (true when the QQ side of the
    overlay is sticky). Safe against torn reads — only single-field
    accesses, no compound state.
    """
    until_mono = float(getattr(state, "adaptive_spread_widen_until_mono", 0.0) or 0.0)
    now_mono = _clock.monotonic()
    remaining = max(0.0, until_mono - now_mono)
    return {
        "active": remaining > 0.0,
        "seconds_remaining": round(remaining, 2) if remaining > 0.0 else 0.0,
        "reason": getattr(state, "adaptive_spread_widen_reason", None),
        "quote_quality_latched": bool(
            getattr(state, "quote_quality_widen_latched", False)
        ),
        # v1.4.155 Phase 2K.5 — exit-attribution counters. Operator
        # calibration: ratio favorable / (favorable + ceiling) over
        # the session indicates whether the per-reason favorable-exit
        # predicates are doing meaningful work or whether the
        # MAX-cooldown ceiling is the binding constraint.
        "cleared_via_favorable_total": int(
            getattr(
                state,
                "adaptive_spread_widen_cleared_via_favorable_total",
                0,
            )
            or 0
        ),
        "cleared_via_ceiling_total": int(
            getattr(
                state,
                "adaptive_spread_widen_cleared_via_ceiling_total",
                0,
            )
            or 0
        ),
    }


def _quote_elig_recovery_block(state: Any) -> dict[str, Any]:
    """Render the quote-eligibility recovery-cooldown countdown.

    Eligibility recovery is the clamp that keeps eligibility at the
    most-restrictive recent level (typically HOLD_ALL) for a brief
    cooldown after the raw signal has improved — gives the venue book
    a moment to settle before we resume full quoting. The clamp
    deadline is stored as ``quote_elig_recovery_until_mono`` and the
    "floor" (the eligibility level we're clamped to) as
    ``quote_elig_recovery_floor``. We surface remaining wall seconds
    + the floor so the dashboard can show "recovery_cooldown 0.7s".

    Returns ``active=False`` when no recovery is armed; the dashboard
    chip filters those out and only renders the countdown when the
    clamp is actually doing something.
    """
    until_mono = float(
        getattr(state, "quote_elig_recovery_until_mono", 0.0) or 0.0
    )
    now_mono = _clock.monotonic()
    remaining = max(0.0, until_mono - now_mono)
    floor = getattr(state, "quote_elig_recovery_floor", None)
    floor_str = floor.value if floor is not None else None
    return {
        "active": remaining > 0.0,
        "seconds_remaining": round(remaining, 2) if remaining > 0.0 else 0.0,
        "floor": floor_str,
    }


def _vol_regime_block(state: Any) -> dict[str, Any]:
    """Render the vol-regime tier the engine is currently using to
    scale quote sizes / spreads. ``tier`` is a categorical label
    (``calm`` / ``normal`` / ``elevated`` / ``spike`` / ``off``);
    ``shrink_factor`` and ``half_spread_bump_bps`` are the actual
    multipliers in effect this tick.
    """
    adj = getattr(state, "vol_regime_adjustment", None)
    if adj is None:
        return {
            "tier": None,
            "shrink_factor": None,
            "half_spread_bump_bps": None,
            "in_spike_window": False,
        }
    tier = getattr(adj, "tier_name", None)
    return {
        "tier": tier,
        "shrink_factor": _round_or_none(getattr(adj, "shrink_factor", None), 3),
        "half_spread_bump_bps": _round_or_none(
            getattr(adj, "half_spread_bump_bps", None), 3
        ),
        "in_spike_window": bool(getattr(adj, "in_spike_window", False)),
    }


def _desync_block(state: Any) -> dict[str, Any]:
    """Per-side execution-desync detail. ``buy`` and ``sell`` are the
    raw per-side flags computed in ``execution.py`` (currently OR'd
    into the legacy ``order_desync`` boolean for back-compat). ``phase``
    is the current desync FSM phase (``OK`` / ``QUARANTINE`` / etc.).
    """
    phase = getattr(state, "desync_phase", None)
    phase_str = (
        getattr(phase, "value", None) or (str(phase) if phase is not None else None)
    )
    return {
        "any": bool(getattr(state, "order_desync", False)),
        "buy": bool(getattr(state, "order_desync_buy", False)),
        "sell": bool(getattr(state, "order_desync_sell", False)),
        "phase": phase_str,
    }


def _basis_regime_block(state: Any) -> dict[str, Any]:
    """Snapshot of the basis-regime classifier — sign + IC + pair
    count + warmup status. Surfaced to the Market tab so the
    operator sees the regime decision the bot is making in
    real-time."""
    br = getattr(state, "basis_regime", None)
    if br is None:
        return {
            "last_regime_sign": None,
            "last_ic": None,
            "pair_count": 0,
            "warmed_up": False,
        }
    try:
        snap = br.snapshot()
    except Exception:
        return {
            "last_regime_sign": None,
            "last_ic": None,
            "pair_count": 0,
            "warmed_up": False,
        }
    return {
        "last_regime_sign": snap.get("last_regime_sign"),
        "last_ic": _round_or_none(snap.get("last_ic"), 4),
        "pair_count": int(snap.get("pair_count", 0) or 0),
        "warmed_up": bool(
            int(snap.get("pair_count", 0) or 0)
            >= int(snap.get("min_pair_samples", 0) or 0)
        ),
    }


def _flow_score_block(state: Any) -> dict[str, Any]:
    """Snapshot of the flow-score signals (TFI + streaks + toxic
    scores). Frontend Market tab renders a composite tier from these."""
    fs = getattr(state, "flow_score", None)
    if fs is None:
        return {
            "tfi_signed_normalised": None,
            "streak_buy_count": 0,
            "streak_sell_count": 0,
            "buy_toxic_score": None,
            "sell_toxic_score": None,
        }
    try:
        snap = fs.snapshot()
    except Exception:
        return {
            "tfi_signed_normalised": None,
            "streak_buy_count": 0,
            "streak_sell_count": 0,
            "buy_toxic_score": None,
            "sell_toxic_score": None,
        }
    return {
        "tfi_signed_normalised": _round_or_none(
            getattr(snap, "tfi_signed_normalised", None), 4
        ),
        "streak_buy_count": int(getattr(snap, "streak_buy_count", 0) or 0),
        "streak_sell_count": int(getattr(snap, "streak_sell_count", 0) or 0),
        "buy_toxic_score": _round_or_none(
            getattr(snap, "buy_toxic_score", None), 3
        ),
        "sell_toxic_score": _round_or_none(
            getattr(snap, "sell_toxic_score", None), 3
        ),
    }


def _regime_mode_block(state: Any) -> Any:
    """v1.4.115 Phase 1E.1.b — top-level regime_controller mode
    block for the Alerts chip + Detectors card.

    Returns the same payload as
    ``regime_controller.snapshot_dict(state.regime_controller, now)``
    plus a derived ISO timestamp for ``mode_since`` so the frontend
    can render ``Mode: DEFENSIVE · since 12:31:07`` without doing
    its own monotonic-to-wall conversion.

    Returns ``None`` when the regime_controller isn't initialised
    on this BotState (legacy / replay paths). The frontend's
    back-compat path renders ``"—"`` in that case.
    """
    rc = getattr(state, "regime_controller", None)
    if rc is None:
        return None
    try:
        from app import regime_controller as _regime_mod
        now_mono = _clock.monotonic()
        # Phase 4G.5 (v1.4.211) — pass the latest forward signal
        # reading so the snapshot's ``forward_signal`` block carries
        # the current classification + diagnostics. ``None`` when the
        # forward layer is disabled (default).
        _fwd_reading = getattr(state, "last_forward_signal_reading", None)
        snap = _regime_mod.snapshot_dict(
            rc, now_mono=now_mono, forward_reading=_fwd_reading
        )
        # Derive a wall-clock ISO for ``mode_since`` from the
        # monotonic deadline. The chip wants this for the operator
        # ("DEFENSIVE since 12:31:07 UTC"), and computing it in the
        # publisher avoids the frontend needing access to the bot's
        # monotonic origin.
        seconds_in_mode = float(snap.get("seconds_in_mode", 0.0) or 0.0)
        from datetime import datetime, timezone, timedelta
        mode_since_iso = (
            _clock.now_utc() - timedelta(seconds=seconds_in_mode)
        ).isoformat()
        snap["mode_since_iso"] = mode_since_iso
        return snap
    except Exception:
        return None


def _shock_gate_block(state: Any) -> Any:
    """v1.4.115 Phase 1E.1 — top-level shock_gate block for the
    Detectors card's Safety-gates section. Mirrors the existing
    behavioural_gates.shock_gate payload."""
    sg = getattr(state, "shock_gate", None)
    if sg is None:
        return None
    try:
        from app import shock_gate as _shock_mod
        return _shock_mod.snapshot_dict(sg, now_mono=_clock.monotonic())
    except Exception:
        return None


def _feature_firing_rates_block(state: Any) -> Any:
    """v1.4.116 Phase 1E.3.f — per-quoting-feature 60 s firing rates.

    Reads from ``state.feature_firing_rates`` (FeatureFiringRateTracker).
    Returns a dict ``{feature_name: count_in_60s, ..., "window_seconds":
    60.0}`` so the frontend renders "post_fill_cooldown_bid: 12 /min"
    style chips. ``window_seconds`` is included so the renderer can
    label the time period — keeps the contract stable if we later
    publish 1 min + 5 min in parallel.
    """
    try:
        tracker = getattr(state, "feature_firing_rates", None)
        if tracker is None:
            return None
        window = 60.0
        rates = tracker.all_rates(window, _clock.monotonic())
        return {
            "window_seconds": window,
            "rates": rates,
        }
    except Exception:
        return None


def _post_reduction_cooldown_block(state: Any) -> Any:
    """v1.4.115 Phase 1E.1 — top-level post-reduction cooldown block
    for the Detectors card. Mirrors the existing
    behavioural_gates.post_reduction_cooldown payload."""
    try:
        settings = getattr(state, "settings", None)
        cd_seconds = (
            float(
                getattr(settings, "post_reduction_cooldown_seconds", 0.0)
                or 0.0
            )
            if settings is not None
            else 0.0
        )
        armed_at = getattr(state, "last_inventory_reduction_at_mono", None)
        if armed_at is None or cd_seconds <= 0.0:
            remaining = 0.0
            active = False
        else:
            elapsed = _clock.monotonic() - float(armed_at)
            remaining = max(0.0, cd_seconds - elapsed)
            active = remaining > 0.0
        sup_side = getattr(
            state, "last_inventory_reduction_suppressed_side", None
        )
        return {
            "enabled": cd_seconds > 0.0,
            "cooldown_seconds": cd_seconds,
            "clear_util_pct": float(
                getattr(settings, "post_reduction_cooldown_clear_util_pct", 0.30)
                if settings is not None
                else 0.30
            ),
            "active": bool(active),
            "seconds_remaining": round(remaining, 2),
            "suppressed_side": (
                sup_side.name if sup_side is not None else None
            ),
            "fire_count": int(
                getattr(state, "post_reduction_cooldown_fire_count", 0)
                or 0
            ),
            # v1.4.144 Phase 2K.1 — exit attribution.
            "cleared_via_favorable_total": int(
                getattr(
                    state,
                    "post_reduction_cooldown_cleared_via_favorable_total",
                    0,
                )
                or 0
            ),
            "cleared_via_ceiling_total": int(
                getattr(
                    state,
                    "post_reduction_cooldown_cleared_via_ceiling_total",
                    0,
                )
                or 0
            ),
        }
    except Exception:
        return None


def _vol_trend_gate_block(state: Any) -> dict[str, Any]:
    """1.2.2 gate state — surfaced for the Market tab cooldown
    indicator. Returns ``active`` + ``seconds_remaining`` derived
    from the monotonic deadline."""
    vt = getattr(state, "vol_trend_gate", None)
    if vt is None:
        return {
            "active": False,
            "seconds_remaining": 0.0,
            "fire_count": 0,
            "last_trigger_vol_ratio": None,
            "last_trigger_drift_bps": None,
        }
    until_mono = float(getattr(vt, "cooldown_until_mono", 0.0) or 0.0)
    now_mono = _clock.monotonic()
    remaining = max(0.0, until_mono - now_mono)
    return {
        "active": remaining > 0.0,
        "seconds_remaining": round(remaining, 2) if remaining > 0.0 else 0.0,
        "fire_count": int(getattr(vt, "fire_count", 0) or 0),
        "last_trigger_vol_ratio": _round_or_none(
            getattr(vt, "last_trigger_vol_ratio", None), 3
        ),
        "last_trigger_drift_bps": _round_or_none(
            getattr(vt, "last_trigger_drift_bps", None), 3
        ),
        # v1.4.153 Phase 2K.3 — exit attribution counters. Operator
        # calibrates ``VOL_TREND_GATE_CLEAR_BAND_MULT`` and
        # ``VOL_TREND_GATE_FAVORABLE_EXIT_DWELL_SECONDS`` from the
        # favorable/ceiling ratio over the session.
        "cleared_via_favorable_total": int(
            getattr(vt, "cleared_via_favorable_total", 0) or 0
        ),
        "cleared_via_ceiling_total": int(
            getattr(vt, "cleared_via_ceiling_total", 0) or 0
        ),
    }


def _post_swing_gate_block(state: Any) -> dict[str, Any]:
    """1.1.131 gate state — cooldown flag + last-trigger reason
    (spike/drop) + fire count. Surfaced for the Market tab Gates
    card."""
    ps = getattr(state, "post_swing", None)
    if ps is None:
        return {
            "active": False,
            "seconds_remaining": 0.0,
            "last_trigger_reason": None,
            "last_trigger_delta_usd": 0.0,
            "fire_count": 0,
        }
    until_mono = float(getattr(ps, "cooldown_until_mono", 0.0) or 0.0)
    now_mono = _clock.monotonic()
    remaining = max(0.0, until_mono - now_mono)
    return {
        "active": remaining > 0.0,
        "seconds_remaining": round(remaining, 2) if remaining > 0.0 else 0.0,
        "last_trigger_reason": getattr(ps, "last_trigger_reason", None),
        "last_trigger_delta_usd": _round_or_none(
            getattr(ps, "last_trigger_delta_usd", 0.0), 4
        ),
        "fire_count": int(getattr(ps, "fire_count", 0) or 0),
        # v1.4.154 Phase 2K.4 — exit-attribution counters. Operator
        # calibration: ratio favorable / (favorable + ceiling) over
        # the session indicates whether the early-clear predicate is
        # doing real work or whether the MAX-cooldown ceiling is the
        # binding constraint.
        "cleared_via_favorable_total": int(
            getattr(ps, "cleared_via_favorable_total", 0) or 0
        ),
        "cleared_via_ceiling_total": int(
            getattr(ps, "cleared_via_ceiling_total", 0) or 0
        ),
    }


def _behavioural_gates_snapshot_for_live_stats(state: Any) -> dict[str, Any]:
    """v1.5.199 — expose the full behavioural_gates dict in live_stats.

    Reuses ``state._behavioural_gates_snapshot`` (the canonical builder
    used by ``state_current.json``). Returns the same shape so the
    Detectors card can render any gate without per-gate publisher
    plumbing. Returns ``{}`` on legacy bots / failure (defensive —
    must not break the live_stats publication path).
    """
    try:
        from app.state import _behavioural_gates_snapshot
        return _behavioural_gates_snapshot(state)
    except Exception:  # noqa: BLE001
        # Defensive: live_stats publication MUST not raise. A missing
        # gate block is preferable to a broken publish loop.
        return {}


def _session_drawdown_block(state: Any) -> dict[str, Any]:
    """1.2.1 tiered session-PnL drawdown ladder state. Returns
    the current tier, cooldown remaining (when paused), trigger
    pnl, and fire count."""
    sd = getattr(state, "session_drawdown", None)
    if sd is None:
        return {
            "tier": "CLEAR",
            "cooldown_seconds_remaining": 0.0,
            "test_resume_fills_remaining": 0,
            "last_trigger_pnl_usd": 0.0,
            "fire_count": 0,
        }
    until_mono = float(getattr(sd, "cooldown_until_mono", 0.0) or 0.0)
    now_mono = _clock.monotonic()
    remaining = max(0.0, until_mono - now_mono)
    tier = getattr(sd, "tier", None)
    tier_str = (
        getattr(tier, "value", None) or (str(tier) if tier is not None else "CLEAR")
    )
    return {
        "tier": tier_str,
        "cooldown_seconds_remaining": round(remaining, 2) if remaining > 0.0 else 0.0,
        "test_resume_fills_remaining": int(
            getattr(sd, "test_resume_fills_remaining", 0) or 0
        ),
        "last_trigger_pnl_usd": _round_or_none(
            getattr(sd, "last_trigger_pnl_usd", 0.0), 4
        ),
        "fire_count": int(getattr(sd, "fire_count", 0) or 0),
    }


def _sf_fatigue_block(settings: Any, state: Any) -> dict[str, Any]:
    """v1.5.202 — SF-event-count fatigue ladder. Mirrors
    ``session_drawdown`` shape so the dashboard chip can be wired
    once and rendered identically for both ladders. Reads the gate
    state from ``state.sf_fatigue`` and the thresholds from
    ``settings``; returns the JSON-serialisable snapshot dict via
    ``sf_fatigue_gate.snapshot_dict``."""
    try:
        from app.sf_fatigue_gate import snapshot_dict
    except Exception:
        return {
            "enabled": False,
            "tier": "CLEAR",
            "events_in_window": 0,
            "fire_count": 0,
            "cooldown_seconds_remaining": 0.0,
        }
    sf = getattr(state, "sf_fatigue", None)
    if sf is None:
        return {
            "enabled": False,
            "tier": "CLEAR",
            "events_in_window": 0,
            "fire_count": 0,
            "cooldown_seconds_remaining": 0.0,
        }
    return snapshot_dict(
        sf,
        now_mono=_clock.monotonic(),
        window_seconds=float(
            getattr(settings, "sf_fatigue_window_seconds", 1800.0) or 0.0
        ),
        tier1_widen_count=int(
            getattr(settings, "sf_fatigue_tier1_widen_count", 3) or 0
        ),
        tier2_pause_short_count=int(
            getattr(settings, "sf_fatigue_tier2_pause_short_count", 5) or 0
        ),
        tier3_pause_long_count=int(
            getattr(settings, "sf_fatigue_tier3_pause_long_count", 8) or 0
        ),
        tier4_kill_count=int(
            getattr(settings, "sf_fatigue_tier4_kill_count", 12) or 0
        ),
        pause_short_seconds=float(
            getattr(settings, "sf_fatigue_pause_short_seconds", 600.0) or 0.0
        ),
        pause_long_seconds=float(
            getattr(settings, "sf_fatigue_pause_long_seconds", 1800.0) or 0.0
        ),
        enabled=bool(getattr(settings, "sf_fatigue_gate_enabled", True)),
    )


def _participation_score_block(settings: Any, state: Any) -> dict[str, Any]:
    """v1.5.207 Phase 4C.5 — participation score block.

    Per-side current + rolling-window mean + disagreement counters
    + the discrete action label the score implies. Mirrors the
    other top-level gate blocks (``sf_fatigue``, ``session_drawdown``).
    """
    try:
        from app.participation_score import participation_action
    except Exception:
        participation_action = None  # type: ignore[assignment]

    def _safe_mean(samples: Any) -> Optional[float]:
        try:
            xs = list(samples)
        except Exception:
            return None
        if not xs:
            return None
        return sum(xs) / float(len(xs))

    soft = float(
        getattr(settings, "participation_score_soft_threshold", 0.7) or 0.7
    )
    hard = float(
        getattr(settings, "participation_score_hard_threshold", 0.3) or 0.3
    )
    bid = getattr(state, "participation_score_bid", None)
    ask = getattr(state, "participation_score_ask", None)
    bid_mean = _safe_mean(
        getattr(state, "participation_score_bid_recent", []) or []
    )
    ask_mean = _safe_mean(
        getattr(state, "participation_score_ask_recent", []) or []
    )
    bid_action = (
        participation_action(bid, soft_threshold=soft, hard_threshold=hard)
        if participation_action is not None and bid is not None
        else "unknown"
    )
    ask_action = (
        participation_action(ask, soft_threshold=soft, hard_threshold=hard)
        if participation_action is not None and ask is not None
        else "unknown"
    )
    return {
        "bid": _round_or_none(bid, 3),
        "ask": _round_or_none(ask, 3),
        "bid_mean": _round_or_none(bid_mean, 3),
        "ask_mean": _round_or_none(ask_mean, 3),
        "bid_action": bid_action,
        "ask_action": ask_action,
        "soft_threshold": soft,
        "hard_threshold": hard,
        "full_edge_bps": float(
            getattr(settings, "participation_score_full_edge_bps", 3.0) or 3.0
        ),
        "disagreement_bid_total": int(
            getattr(state, "participation_score_disagreement_bid_total", 0)
            or 0
        ),
        "disagreement_ask_total": int(
            getattr(state, "participation_score_disagreement_ask_total", 0)
            or 0
        ),
    }


def _ofi_block(settings: Any, state: Any) -> dict[str, Any]:
    """v1.5.209 Phase 8D — OFI accumulator state.

    Top-level block. Publishes both EWMAs (5s short-horizon + 30s
    anchor), the normalised signals, the alpha value (so the
    operator can see if the lever is dormant), and the last-tick
    shift applied to reservation. Accumulator runs unconditionally
    so this block is populated whenever the bot has seen ≥ 2 BBO
    updates.
    """
    ofi = getattr(state, "ofi", None)
    if ofi is None:
        return {
            "enabled": bool(getattr(settings, "ofi_enabled", True)),
            "alpha": float(getattr(settings, "ofi_reservation_alpha", 0.0) or 0.0),
            "available": False,
        }
    snap = ofi.snapshot_dict() if hasattr(ofi, "snapshot_dict") else {}
    return {
        "enabled": bool(getattr(settings, "ofi_enabled", True)),
        "alpha": float(getattr(settings, "ofi_reservation_alpha", 0.0) or 0.0),
        "available": True,
        "update_count": int(snap.get("update_count", 0) or 0),
        "raw_ewma_5s": _round_or_none(snap.get("raw_ewma_5s"), 4),
        "raw_ewma_30s": _round_or_none(snap.get("raw_ewma_30s"), 4),
        "signal_5s_normalised": _round_or_none(snap.get("signal_5s_normalised"), 4),
        "signal_30s_normalised": _round_or_none(snap.get("signal_30s_normalised"), 4),
        "halflife_5s_seconds": float(snap.get("halflife_5s_seconds", 5.0) or 5.0),
        "halflife_30s_seconds": float(snap.get("halflife_30s_seconds", 30.0) or 30.0),
        "normalisation_scale": float(snap.get("normalisation_scale", 100.0) or 100.0),
        "last_shift_bps": float(getattr(state, "ofi_last_shift_bps", 0.0) or 0.0),
    }


def _queue_aware_block(settings: Any, state: Any) -> dict[str, Any]:
    """v1.5.209 Phase 8B — queue-aware sizing + inside-post block."""
    return {
        "sizing_enabled": bool(
            getattr(settings, "queue_aware_sizing_enabled", False)
        ),
        "inside_post_enabled": bool(
            getattr(settings, "queue_aware_inside_post_enabled", False)
        ),
        "size_floor": float(
            getattr(settings, "queue_aware_size_floor", 0.3) or 0.3
        ),
        "size_decay": float(
            getattr(settings, "queue_aware_size_decay", 0.7) or 0.7
        ),
        "inside_post_threshold": float(
            getattr(settings, "queue_aware_inside_post_threshold", 0.7) or 0.7
        ),
        "inside_post_step_bps": float(
            getattr(settings, "queue_aware_inside_post_step_bps", 1.0) or 1.0
        ),
        "ratio_bid": _round_or_none(
            getattr(state, "queue_position_ratio_bid", None), 3
        ),
        "ratio_ask": _round_or_none(
            getattr(state, "queue_position_ratio_ask", None), 3
        ),
        "size_mult_bid": float(getattr(state, "queue_size_mult_bid", 1.0) or 1.0),
        "size_mult_ask": float(getattr(state, "queue_size_mult_ask", 1.0) or 1.0),
        "inside_post_active_bid": bool(
            getattr(state, "queue_inside_post_active_bid", False)
        ),
        "inside_post_active_ask": bool(
            getattr(state, "queue_inside_post_active_ask", False)
        ),
        "size_mult_armed_bid_total": int(
            getattr(state, "queue_size_mult_armed_bid_total", 0) or 0
        ),
        "size_mult_armed_ask_total": int(
            getattr(state, "queue_size_mult_armed_ask_total", 0) or 0
        ),
        "inside_post_armed_bid_total": int(
            getattr(state, "queue_inside_post_armed_bid_total", 0) or 0
        ),
        "inside_post_armed_ask_total": int(
            getattr(state, "queue_inside_post_armed_ask_total", 0) or 0
        ),
        "arrival_rate_bid_per_sec": _round_or_none(
            getattr(
                getattr(state, "queue_arrival_rate", None),
                "bid_arrival_rate_per_sec",
                None,
            ),
            3,
        ),
        "arrival_rate_ask_per_sec": _round_or_none(
            getattr(
                getattr(state, "queue_arrival_rate", None),
                "ask_arrival_rate_per_sec",
                None,
            ),
            3,
        ),
    }


def _basis_block(
    market: Any,
    binance_mid: Optional[float],
    basis_ewma: Optional[float],
) -> dict[str, Any]:
    okx_mid = getattr(market, "mid_price", None) if market is not None else None
    fair = None
    if binance_mid is not None and basis_ewma is not None:
        try:
            fair = float(binance_mid) + float(basis_ewma)
        except (TypeError, ValueError):
            fair = None
    delta_bps = None
    if okx_mid is not None and binance_mid is not None and float(binance_mid) > 0:
        try:
            delta_bps = (
                (float(okx_mid) - float(binance_mid)) / float(binance_mid)
            ) * 10_000.0
        except (TypeError, ValueError, ZeroDivisionError):
            delta_bps = None
    return {
        "ewma": _round_or_none(basis_ewma, 8),
        "fair_value": _round_or_none(fair, 8),
        "okx_minus_binance_bps": _round_or_none(delta_bps, 3),
    }


def _inventory_block(settings: Settings, position: Any) -> dict[str, Any]:
    qty = getattr(position, "position_qty", 0.0) if position else 0.0
    notional = getattr(position, "position_notional", 0.0) if position else 0.0
    cap = float(settings.max_position_notional_usd or 0.0)
    util = (
        (abs(float(notional)) / cap) * 100.0
        if cap > 0 and notional is not None
        else None
    )
    return {
        "qty": _round_or_none(qty, 6),
        "notional_usd": _round_or_none(notional, 4),
        "max_notional_usd": _round_or_none(cap, 4),
        "utilization_pct": _round_or_none(util, 2),
    }


def _skew_block(settings: Settings, position: Any) -> dict[str, Any]:
    """Reproduces the live skew the QuoteEngine applies on each tick.

    Mirrors the math in ``app/quoting.py``:
        norm_inv = clip(qty / max_abs_position, -1, 1)
        signed_adj = sign(norm_inv) * |norm_inv|^skew_exponent
        skew_bps = INVENTORY_SKEW_COEFF_BPS * signed_adj

    Surfacing this in the panel lets the operator see at a glance
    whether the skew is meaningfully pulling the reservation off-mid
    -- distinguishes "near-flat: skew tiny" from "near-cap: skew is
    why the BID/ASK is so wide".
    """
    qty = float(getattr(position, "position_qty", 0.0) or 0.0)
    max_abs = float(settings.max_abs_position or 0.0)
    if max_abs <= 0:
        return {"bps": None, "norm_inventory": None}
    norm = max(-1.0, min(1.0, qty / max_abs))
    exp = float(settings.inventory_skew_exponent)
    if exp != 1.0 and norm != 0.0:
        signed_adj = math.copysign(abs(norm) ** exp, norm)
    else:
        signed_adj = norm
    skew_bps = float(settings.inventory_skew_coeff_bps) * signed_adj
    return {
        "bps": _round_or_none(skew_bps, 3),
        "norm_inventory": _round_or_none(norm, 4),
    }


def _working_ladder_block(state: BotState, *, now_utc) -> dict[str, Any]:
    """v1.4.100 ladder-observability D1: snapshot of all working
    orders grouped by ``level_idx``. One row per (side, level_idx)
    slot in the bot's OrderStore. Lets the dashboard render the per-
    rung working-ladder table instead of the legacy scalar bid/ask
    chip.

    Returns:

    ```json
    {
      "rungs": [
        {"side": "BUY", "level_idx": 0, "status": "ACKED",
         "price": 2.030, "size": 3.0, "age_seconds": 1.2,
         "oid": "35795...", "cloid": "abc..."},
        ...
      ],
      "max_level_idx_buy": 1,
      "max_level_idx_sell": 1
    }
    ```

    The Order Store snapshot is taken under the BotState lock by
    ``state.all_working_orders()``. Pre-v1.3.130 builds carry
    ``level_idx=0`` on all rows (single-rung mode).
    """
    rungs: list[dict[str, Any]] = []
    max_buy = 0
    max_sell = 0
    try:
        wos = state.all_working_orders()
    except Exception:
        return {"rungs": [], "max_level_idx_buy": 0, "max_level_idx_sell": 0}
    for wo in wos:
        try:
            side_val = getattr(getattr(wo, "side", None), "value", None) or str(
                getattr(wo, "side", "")
            )
            status_val = getattr(getattr(wo, "status", None), "value", None) or str(
                getattr(wo, "status", "")
            )
            lvl = int(getattr(wo, "level_idx", 0) or 0)
            if side_val.upper().startswith("B"):
                if lvl > max_buy:
                    max_buy = lvl
            else:
                if lvl > max_sell:
                    max_sell = lvl
            ts_created = getattr(wo, "ts_created", None)
            age_s: Optional[float] = None
            if ts_created is not None:
                try:
                    age_s = max(0.0, (now_utc - ts_created).total_seconds())
                except Exception:
                    age_s = None
            rungs.append(
                {
                    "side": side_val,
                    "level_idx": lvl,
                    "status": status_val,
                    "price": _round_or_none(getattr(wo, "price", None), 6),
                    "size": _round_or_none(getattr(wo, "size", None), 6),
                    "age_seconds": _round_or_none(age_s, 2),
                    "oid": str(getattr(wo, "order_id_exchange", "") or ""),
                    "cloid": str(getattr(wo, "client_order_id", "") or ""),
                }
            )
        except Exception:
            continue
    # Sort: BUY rungs ascending by level_idx then SELL ascending.
    rungs.sort(key=lambda r: (0 if r["side"].upper().startswith("B") else 1, r["level_idx"]))
    return {
        "rungs": rungs,
        "max_level_idx_buy": max_buy,
        "max_level_idx_sell": max_sell,
    }


def _ladder_per_rung_fills_block(
    recent_fills: list[Any], *, now_utc
) -> dict[str, Any]:
    """v1.4.100 ladder-observability D2: per-rung sliding-window fill
    counts. For each rung (level_idx), report fill count in the last
    1-min and 5-min windows. Operator's "rung-0 = 8/min, rung-1 =
    1/min" at-a-glance chip lives off this.

    Pre-v1.4.100 fills lack ``level_idx`` so every fill rolls into the
    L0 bucket — same as single-rung behaviour. Returns:

    ```json
    {
      "by_rung": {
        "0": {"count_1m": 8, "count_5m": 35},
        "1": {"count_1m": 1, "count_5m": 6}
      },
      "max_level_seen": 1
    }
    ```
    """
    from datetime import timedelta

    out: dict[str, Any] = {"by_rung": {}, "max_level_seen": 0}
    if not recent_fills:
        return out
    cutoff_1m = now_utc - timedelta(seconds=60)
    cutoff_5m = now_utc - timedelta(seconds=300)
    max_lvl = 0
    counts: dict[int, dict[str, int]] = {}
    for f in recent_fills:
        ts = getattr(f, "ts_fill", None)
        if ts is None:
            continue
        try:
            lvl = int(getattr(f, "level_idx", 0) or 0)
        except (TypeError, ValueError):
            lvl = 0
        if lvl < 0:
            lvl = 0
        if lvl > max_lvl:
            max_lvl = lvl
        if ts < cutoff_5m:
            continue
        bucket = counts.setdefault(lvl, {"count_1m": 0, "count_5m": 0})
        bucket["count_5m"] += 1
        if ts >= cutoff_1m:
            bucket["count_1m"] += 1
    out["by_rung"] = {str(k): v for k, v in sorted(counts.items())}
    out["max_level_seen"] = max_lvl
    return out


def _markouts_block(recent_fills: list[Any]) -> dict[str, Any]:
    """Median + adverse-fraction over the rolling window of recent
    fills. Sample size capped to the most-recent 50 to keep the
    aggregate stable across single fills."""
    sample = recent_fills[:50]
    n = len(sample)
    out: dict[str, Any] = {
        "n_recent_fills": n,
        "median_1s_bps": None,
        "median_5s_bps": None,
        "adverse_pct_1s": None,
        "adverse_pct_5s": None,
    }
    if n == 0:
        return out
    m1 = [
        float(getattr(f, "markout_1s_bps", None))
        for f in sample
        if getattr(f, "markout_1s_bps", None) is not None
    ]
    m5 = [
        float(getattr(f, "markout_5s_bps", None))
        for f in sample
        if getattr(f, "markout_5s_bps", None) is not None
    ]
    if m1:
        out["median_1s_bps"] = round(statistics.median(m1), 3)
        out["adverse_pct_1s"] = round(
            sum(1 for x in m1 if x < 0.0) / len(m1) * 100.0, 1
        )
    if m5:
        out["median_5s_bps"] = round(statistics.median(m5), 3)
        out["adverse_pct_5s"] = round(
            sum(1 for x in m5 if x < 0.0) / len(m5) * 100.0, 1
        )
    return out


# v1.4.101 — horizons surfaced in the multi-horizon attribution block.
# Subset of ``app.pnl_attribution._SUPPORTED_HORIZONS`` chosen for the
# Bot Stats "Markout horizon ladder" tile. Includes the canonical 5s
# horizon so the tile can highlight it. The 3s horizon is dropped here
# (it's between 1s and 5s with no operationally distinct narrative);
# kept available in the postmortem path which uses the full set.
_LIVE_STATS_MULTI_HORIZONS_S: tuple[int, ...] = (1, 5, 15, 30, 60, 120)


def _build_multi_horizon_attribution_block_from_fills(
    recent_fills: list[Any],
) -> dict[str, Any]:
    """Convenience wrapper around
    :func:`_build_multi_horizon_attribution_block` that accepts raw
    ``Fill`` dataclass instances. Mirrors the Fill→dict conversion
    in :func:`_build_attribution_block` so the two builders consume
    the same window without doubling the conversion code.
    """
    realized_pnl_window_usd: float = 0.0
    for f in recent_fills:
        cp = getattr(f, "closed_pnl", None)
        if cp is None:
            continue
        try:
            realized_pnl_window_usd += float(cp)
        except (TypeError, ValueError):
            continue
    fill_dicts: list[dict[str, Any]] = []
    for f in recent_fills:
        try:
            fill_dicts.append(
                {
                    "side": str(getattr(f, "side", "") or ""),
                    "price": float(getattr(f, "price", 0.0) or 0.0),
                    "size": float(getattr(f, "size", 0.0) or 0.0),
                    "notional": float(getattr(f, "notional", 0.0) or 0.0),
                    "fee": float(getattr(f, "fee", 0.0) or 0.0),
                    "liquidity_flag": getattr(f, "liquidity_flag", None),
                    "markout_1s_bps": getattr(f, "markout_1s_bps", None),
                    "markout_3s_bps": getattr(f, "markout_3s_bps", None),
                    "markout_5s_bps": getattr(f, "markout_5s_bps", None),
                    "markout_15s_bps": getattr(f, "markout_15s_bps", None),
                    "markout_30s_bps": getattr(f, "markout_30s_bps", None),
                    "markout_60s_bps": getattr(f, "markout_60s_bps", None),
                    "markout_120s_bps": getattr(f, "markout_120s_bps", None),
                    "closed_pnl": getattr(f, "closed_pnl", None),
                }
            )
        except Exception:
            continue
    return _build_multi_horizon_attribution_block(
        fill_dicts, realized_pnl_window_usd
    )


def _build_multi_horizon_attribution_block(
    fill_dicts: list[dict[str, Any]],
    realized_pnl_window_usd: float,
) -> dict[str, Any]:
    """v1.4.101: per-horizon markout + PnL-attribution decomposition.

    Returns a dict keyed by horizon-string (``"1s"`` / ``"5s"`` / …)
    with each value carrying:

    * ``markout`` block — mean/median bps, adverse/favorable counts,
      win rate, fill count at that horizon (some fills may lack a
      far-horizon markout if they fired within the horizon window).
    * ``pnl_attribution_usd`` block — rebate income, markout dollar
      impact, residual at that horizon.
    * ``is_canonical`` — True only for the 5s horizon (matches the
      canonical ``pnl_attribution`` block in the payload).

    Errors at any individual horizon are absorbed silently — the
    horizon's slot is omitted rather than killing the whole block.
    The 5s canonical slot survives independently because the caller
    publishes ``pnl_attribution`` from a separate code path.

    Mirrors the shape in ``session_summary.py``'s
    ``markouts_multi_horizon`` block so the dashboard can read either
    source with the same parser. See `plans/_DONE/ladder-observability
    .md` and the v1.4.98 markout-horizon work for the broader context.
    """
    out: dict[str, Any] = {}
    if not fill_dicts:
        return out
    try:
        from app.pnl_attribution import compute_attribution
    except Exception:
        return out
    for h in _LIVE_STATS_MULTI_HORIZONS_S:
        try:
            att_h = compute_attribution(
                fills=fill_dicts,
                horizon_s=h,
                realized_pnl_total_usd=realized_pnl_window_usd,
            )
        except Exception:
            # Insufficient samples at this horizon (e.g. session
            # younger than 120s yet) — skip the slot rather than
            # crashing the publish.
            continue
        markout = att_h.get("markout") or {}
        pnl_attr = att_h.get("pnl_attribution_usd") or {}
        out[f"{h}s"] = {
            "markout": {
                "mean_bps": _round_or_none(markout.get("mean_bps"), 3),
                "median_bps": _round_or_none(markout.get("median_bps"), 3),
                "adverse_count": int(markout.get("adverse_count") or 0),
                "favorable_count": int(markout.get("favorable_count") or 0),
                "win_rate": _round_or_none(markout.get("win_rate"), 3),
                # ``sample_count`` is the number of fills that actually
                # carried a markout value at this horizon (a subset of
                # the window when the horizon hasn't fully resolved
                # yet — e.g. fills younger than 120s have no 120s
                # markout). Equals adverse_count + favorable_count.
                "sample_count": int(markout.get("sample_count") or 0),
            },
            "pnl_attribution_usd": {
                "fee_income": _round_or_none(pnl_attr.get("fee_income"), 6),
                "markout_dollar_impact": _round_or_none(
                    pnl_attr.get("markout_dollar_impact"), 6
                ),
                "residual": _round_or_none(pnl_attr.get("residual"), 6),
                # Realized is horizon-independent (per-fill closed_pnl
                # sum), but the residual = realized - rebate - markout
                # equation makes more sense with realized inline.
                "realized_pnl_total": _round_or_none(
                    pnl_attr.get("realized_pnl_total"), 4
                ),
            },
            "is_canonical": h == 5,
        }
    return out


def _build_attribution_block(
    recent_fills: list[Any],
) -> dict[str, Any]:
    """Wraps ``app.pnl_attribution.compute_attribution`` for the
    live_stats payload. Decomposes realized PnL into rebate income,
    markout dollar impact, and residual at the 5s horizon over the
    rolling ``recent_fills`` window (capped at 1000; widened from
    200 on 2026-05-10 operator request — wider window means the
    Bot Stats panel reflects a longer recent history before fills
    start aging out).

    Realized is computed from the SAME window as rebate/markout
    (sum of per-fill ``closed_pnl`` from venue private-WS), so the
    decomposition is internally self-consistent: the equation
    ``realized = rebate + markout + residual`` holds at any session
    length.

    Pre-fix (2026-05-09 operator-flagged), realized was passed in
    from PnlTracker as a session-cumulative number while
    rebate/markout were window-scoped. Once a session exceeded the
    deque maxlen, realized covered all fills but rebate/markout only
    the most recent N, making the residual a meaningless mix-of-
    scopes number.

    Returned dict matches the schema the dashboard's Bot Stats
    panel expects (see ``LiveStatsPayload.pnl_attribution`` in
    ``frontend/lib/live_stats.ts``).

    On any error or empty input, returns a benign empty block so
    the dashboard's card renders "no samples yet" rather than
    breaking the whole payload.
    """
    # Window-scope realized: sum per-fill ``closed_pnl`` (set by the
    # venue's per-fill realized PnL field; ``None`` on opening fills).
    realized_pnl_window_usd: float = 0.0
    for f in recent_fills:
        cp = getattr(f, "closed_pnl", None)
        if cp is None:
            continue
        try:
            realized_pnl_window_usd += float(cp)
        except (TypeError, ValueError):
            continue

    empty: dict[str, Any] = {
        "horizon_s": 5,
        "fill_count": 0,
        "realized_pnl_usd": _round_or_none(realized_pnl_window_usd, 4),
        "rebate_income_usd": None,
        "markout_dollar_impact_usd": None,
        "residual_usd": None,
        "rebate_bps_of_notional": None,
        "mean_markout_bps": None,
        "median_markout_bps": None,
        "adverse_count": 0,
        "favorable_count": 0,
        "win_rate": None,
        "maker_count": 0,
        "taker_count": 0,
    }
    if not recent_fills:
        return empty
    # Convert Fill dataclass instances to the dict shape compute_
    # attribution expects. Keep only the fields it reads.
    fill_dicts: list[dict[str, Any]] = []
    for f in recent_fills:
        try:
            fill_dicts.append(
                {
                    "side": str(getattr(f, "side", "") or ""),
                    "price": float(getattr(f, "price", 0.0) or 0.0),
                    "size": float(getattr(f, "size", 0.0) or 0.0),
                    "notional": float(getattr(f, "notional", 0.0) or 0.0),
                    "fee": float(getattr(f, "fee", 0.0) or 0.0),
                    "liquidity_flag": getattr(f, "liquidity_flag", None),
                    "markout_1s_bps": getattr(f, "markout_1s_bps", None),
                    "markout_3s_bps": getattr(f, "markout_3s_bps", None),
                    "markout_5s_bps": getattr(f, "markout_5s_bps", None),
                    # v1.4.98 — extended diagnostic horizons. Read
                    # defensively for cross-version compatibility
                    # (older Fill instances in the recent_fills deque
                    # may lack these fields after an in-place restart).
                    "markout_15s_bps": getattr(f, "markout_15s_bps", None),
                    "markout_30s_bps": getattr(f, "markout_30s_bps", None),
                    "markout_60s_bps": getattr(f, "markout_60s_bps", None),
                    "markout_120s_bps": getattr(f, "markout_120s_bps", None),
                }
            )
        except Exception:
            # Skip malformed fills rather than blowing up the whole
            # publish — the live panel can tolerate gaps.
            continue
    if not fill_dicts:
        return empty
    try:
        from app.pnl_attribution import compute_attribution

        att = compute_attribution(
            fills=fill_dicts,
            horizon_s=5,
            realized_pnl_total_usd=realized_pnl_window_usd,
        )
    except Exception as e:
        logger.warning("live_stats_attribution_failed err=%s", str(e)[:200])
        return empty

    fills_block = att.get("fills") or {}
    fees_block = att.get("fees") or {}
    markout_block = att.get("markout") or {}
    pnl_block = att.get("pnl_attribution_usd") or {}
    return {
        "horizon_s": int(att.get("horizon_s") or 5),
        "fill_count": int(fills_block.get("total") or 0),
        "realized_pnl_usd": _round_or_none(
            pnl_block.get("realized_pnl_total"), 4
        ),
        "rebate_income_usd": _round_or_none(pnl_block.get("fee_income"), 6),
        "markout_dollar_impact_usd": _round_or_none(
            pnl_block.get("markout_dollar_impact"), 6
        ),
        "residual_usd": _round_or_none(pnl_block.get("residual"), 6),
        "rebate_bps_of_notional": _round_or_none(
            fees_block.get("rebate_bps_of_notional"), 3
        ),
        "mean_markout_bps": _round_or_none(markout_block.get("mean_bps"), 3),
        "median_markout_bps": _round_or_none(
            markout_block.get("median_bps"), 3
        ),
        "adverse_count": int(markout_block.get("adverse_count") or 0),
        "favorable_count": int(markout_block.get("favorable_count") or 0),
        "win_rate": _round_or_none(markout_block.get("win_rate"), 3),
        "maker_count": int(fills_block.get("maker_count") or 0),
        "taker_count": int(fills_block.get("taker_count") or 0),
    }


def _compute_microprice(
    best_bid: Optional[float],
    best_ask: Optional[float],
    bid_size: Optional[float],
    ask_size: Optional[float],
) -> Optional[float]:
    """Imbalance-weighted mid: heavy bid stack pulls toward the ask
    (because takers are likely to keep buying through), and vice versa.
    Returns None when depth is missing -- caller falls back to mid."""
    try:
        if (
            best_bid is None
            or best_ask is None
            or bid_size is None
            or ask_size is None
        ):
            return None
        bb = float(best_bid)
        ba = float(best_ask)
        bs = float(bid_size)
        as_ = float(ask_size)
        denom = bs + as_
        if denom <= 0 or not (math.isfinite(bb) and math.isfinite(ba)):
            return None
        return (bs * ba + as_ * bb) / denom
    except (TypeError, ValueError):
        return None


def _tp_attribution_block(
    storage: Any,
    *,
    session_started_at_iso: Optional[str] = None,
) -> dict[str, Any]:
    """Build the TP attribution block published to S3 every ~5 s.

    Mirror of ``_soft_flatten_attribution_block`` but reading the
    ``tp_events`` table (storage v42) and the ``tp_event_id`` FK on
    orders + fills. Empty / silent on any storage failure.
    Plan ref: v1.5.33 take-profit feature.
    """
    out: dict[str, Any] = {
        "events": [],
        "order_tags": {},
        "fill_tags": {},
    }
    if storage is None:
        return out
    try:
        events = storage.recent_tp_events(
            limit=50, since_ts=session_started_at_iso
        )
    except Exception:
        logger.exception("live_stats_tp_events_fetch_failed")
        return out
    out["events"] = [
        {
            "id": int(e["id"]),
            "ts_start": e.get("ts_start"),
            "ts_end": e.get("ts_end"),
            "trigger_upnl_bps": e.get("trigger_upnl_bps"),
            "trigger_threshold_bps": e.get("trigger_threshold_bps"),
            "disarm_margin_bps": e.get("disarm_margin_bps"),
            "entry_position_qty": e.get("entry_position_qty"),
            "entry_mid_price": e.get("entry_mid_price"),
            "entry_target_price": e.get("entry_target_price"),
            "close_side": e.get("close_side"),
            "exit_reason": e.get("exit_reason"),
            "exit_upnl_bps": e.get("exit_upnl_bps"),
            "attributed_orders_count": int(
                e.get("attributed_orders_count") or 0
            ),
            "attributed_fills_count": int(
                e.get("attributed_fills_count") or 0
            ),
            "attributed_fills_notional_usd": float(
                e.get("attributed_fills_notional_usd") or 0.0
            ),
        }
        for e in events
    ]
    try:
        recent_orders = storage.recent_orders(limit=200)
        for o in recent_orders:
            tpid = o.get("tp_event_id")
            oxid = o.get("order_id_exchange")
            if tpid is not None and oxid:
                out["order_tags"][str(oxid)] = int(tpid)
    except Exception:
        logger.exception("live_stats_tp_order_tags_fetch_failed")
    try:
        recent_fills = storage.recent_fills(limit=200)
        for f in recent_fills:
            tpid = f.get("tp_event_id")
            fid = f.get("fill_id")
            if tpid is not None and fid:
                out["fill_tags"][str(fid)] = int(tpid)
    except Exception:
        logger.exception("live_stats_tp_fill_tags_fetch_failed")
    return out


def _soft_flatten_attribution_block(
    storage: Any,
    *,
    session_started_at_iso: Optional[str] = None,
) -> dict[str, Any]:
    """Build the SF attribution block published to S3 every ~5 s.

    Plan ref: plans/20260507-sf-frontend.md Phase 2.

    Three sub-blocks:
    - ``events``: recent SF episodes (lifecycle row + LEFT-JOIN
      attribution counts). Up to 50 rows.
    - ``order_tags``: ``{order_id_exchange: soft_flatten_event_id}``
      for orders in the recent window that carry an FK. Frontend
      uses this to map OKX REST order rows to their SF episode.
    - ``fill_tags``: same shape for fills.

    Empty / silent on any storage failure — the rest of the live
    stats payload still publishes. Pre-Phase-2 SQLite files (no
    ``soft_flatten_events`` table) return empty blocks; the schema
    migration on bot startup adds the table so steady state is
    populated.

    v1.4.36 (Codex #3 + #7): when ``session_started_at_iso`` is
    supplied, scope the events list AND the order/fill tag maps to
    the current session. Without the filter the block included
    historical episodes from prior sessions even though the
    frontend labelled it "events this session" — see the Codex
    review at ``plans/20260517-codex-review.md`` for the operator-
    confusion symptom.

    DB events themselves are NOT scoped — they persist as the
    historical record. Postmortem + equity-history callers still
    invoke ``recent_soft_flatten_events`` without ``since_ts`` and
    see the full history.
    """
    out: dict[str, Any] = {
        "events": [],
        "order_tags": {},
        "fill_tags": {},
    }
    if storage is None:
        return out
    try:
        events = storage.recent_soft_flatten_events(
            limit=50, since_ts=session_started_at_iso
        )
    except Exception:
        logger.exception("live_stats_sf_events_fetch_failed")
        return out
    out["events"] = [
        {
            "id": int(e["id"]),
            "ts_start": e.get("ts_start"),
            "ts_end": e.get("ts_end"),
            "trigger_reason": e.get("trigger_reason"),
            "initial_force_phase": e.get("initial_force_phase"),
            "taker_fallback_ticks": e.get("taker_fallback_ticks"),
            "entry_position_qty": e.get("entry_position_qty"),
            "entry_mid_price": e.get("entry_mid_price"),
            "exit_phase_reached": e.get("exit_phase_reached"),
            "exit_reason": e.get("exit_reason"),
            "attributed_orders_count": int(
                e.get("attributed_orders_count") or 0
            ),
            "attributed_fills_count": int(
                e.get("attributed_fills_count") or 0
            ),
            "attributed_fills_notional_usd": float(
                e.get("attributed_fills_notional_usd") or 0.0
            ),
        }
        for e in events
    ]
    # Tag maps. Bound the lookup to the rows the dashboard actually
    # renders (last ~200) — covers the 100-row visible window with
    # generous slack and keeps the payload compact even on long
    # sessions where most rows have NULL FKs.
    try:
        recent_orders = storage.recent_orders(limit=200)
        for o in recent_orders:
            sfid = o.get("soft_flatten_event_id")
            oxid = o.get("order_id_exchange")
            if sfid is not None and oxid:
                out["order_tags"][str(oxid)] = int(sfid)
    except Exception:
        logger.exception("live_stats_sf_order_tags_fetch_failed")
    try:
        recent_fills = storage.recent_fills(limit=200)
        for f in recent_fills:
            sfid = f.get("soft_flatten_event_id")
            fid = f.get("fill_id")
            if sfid is not None and fid:
                out["fill_tags"][str(fid)] = int(sfid)
    except Exception:
        logger.exception("live_stats_sf_fill_tags_fetch_failed")
    return out
