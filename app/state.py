from __future__ import annotations

import logging
import math
import threading
import time
import uuid
import warnings
from collections import deque
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Optional

# Phase 1c (v1.4.231): all ``time.monotonic`` / ``time.time`` /
# ``datetime.now(timezone.utc)`` calls in this module route through
# the module-level clock proxy in ``app.clock`` so backtest replay
# can swap in a ``ReplayClock``. Under production the proxy
# delegates to the stdlib — bit-identical behaviour.
from app import clock as _clock

_state_logger = logging.getLogger(__name__)

# v1.4.91 wedge-elimination-cleanup Phase 3A.6 — once-per-process gate
# for the legacy ``working_bid`` / ``working_ask`` deprecation warnings.
# These accessors are called from 60+ legacy read sites in production
# code; emitting a DeprecationWarning per access would flood test
# output and prod logs. The gate ensures each property's warning is
# emitted at most ONCE per process — enough to surface the marker to
# devs and CI without runaway noise. Module-level (not class-level)
# so pytest test isolation doesn't reset it.
_WORKING_BID_DEPRECATION_EMITTED: bool = False
_WORKING_ASK_DEPRECATION_EMITTED: bool = False


def _behavioural_gates_snapshot(state: "BotState") -> dict[str, Any]:
    """Module-level helper: render every behavioural gate's current
    state into a single dict for inclusion in
    ``BotState.snapshot_dict()``. Mirrors the live_stats blocks but
    lands in ``state_current.json`` (HTTP /state/current → snapshot
    bundle), so postmortem analysis can reconstruct gate activity
    from a static snapshot alone.

    Each gate's block answers: is it currently active, how long
    until it clears, what fired it last, how many times has it
    fired this session. Stateless gates (microprice_gate,
    momentum_gate) are NOT in here — they're instantaneous checks,
    no state to capture. basis_regime_gate is stateless on the gate
    side but the bot publishes its current size-shrink output here
    (1.2.8) so postmortem can see what the gate decided each tick.
    """
    out: dict[str, Any] = {}

    # adaptive_widen — already had ``adaptive_spread_widen_until_mono``
    # exposed but in raw monotonic form. Compose a friendlier block.
    try:
        until = float(getattr(state, "adaptive_spread_widen_until_mono", 0.0) or 0.0)
        remaining = max(0.0, until - _clock.monotonic())
        out["adaptive_widen"] = {
            "active": remaining > 0.0,
            "seconds_remaining": round(remaining, 2),
            "reason": getattr(state, "adaptive_spread_widen_reason", None),
            "quote_quality_latched": bool(
                getattr(state, "quote_quality_widen_latched", False)
            ),
            # v1.4.155 Phase 2K.5 — favorable-exit attribution.
            "favorable_dwell_started_mono": getattr(
                state,
                "adaptive_spread_widen_favorable_dwell_started_mono",
                None,
            ),
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
            "cleared_via_position_favorable_total": int(
                getattr(
                    state,
                    "adaptive_spread_widen_cleared_via_position_favorable_total",
                    0,
                )
                or 0
            ),
        }
    except Exception:
        out["adaptive_widen"] = None

    # post_swing PnL cooldown.
    try:
        ps = state.post_swing
        until = float(getattr(ps, "cooldown_until_mono", 0.0) or 0.0)
        remaining = max(0.0, until - _clock.monotonic())
        out["post_swing"] = {
            "active": remaining > 0.0,
            "seconds_remaining": round(remaining, 2),
            "last_trigger_reason": getattr(ps, "last_trigger_reason", None),
            "last_trigger_delta_usd": float(getattr(ps, "last_trigger_delta_usd", 0.0)),
            "fire_count": int(getattr(ps, "fire_count", 0) or 0),
            # v1.4.154 Phase 2K.4 — favorable-exit attribution.
            "favorable_dwell_started_mono": getattr(
                ps, "favorable_dwell_started_mono", None
            ),
            "cleared_via_favorable_total": int(
                getattr(ps, "cleared_via_favorable_total", 0) or 0
            ),
            "cleared_via_ceiling_total": int(
                getattr(ps, "cleared_via_ceiling_total", 0) or 0
            ),
        }
    except Exception:
        out["post_swing"] = None

    # vol × trend conjunction.
    try:
        vt = state.vol_trend_gate
        until = float(getattr(vt, "cooldown_until_mono", 0.0) or 0.0)
        remaining = max(0.0, until - _clock.monotonic())
        # v1.4.153 Phase 2K.3 — favorable-exit attribution surfaces.
        # Operator reads the ratio
        #   favorable / (favorable + ceiling)
        # to calibrate ``VOL_TREND_GATE_CLEAR_BAND_MULT`` and
        # ``VOL_TREND_GATE_FAVORABLE_EXIT_DWELL_SECONDS``.
        out["vol_trend_gate"] = {
            "active": remaining > 0.0,
            "seconds_remaining": round(remaining, 2),
            "last_trigger_vol_ratio": float(
                getattr(vt, "last_trigger_vol_ratio", 0.0)
            ),
            "last_trigger_drift_bps": float(
                getattr(vt, "last_trigger_drift_bps", 0.0)
            ),
            "armed_at_mono": getattr(vt, "armed_at_mono", None),
            "fire_count": int(getattr(vt, "fire_count", 0) or 0),
            "favorable_dwell_started_mono": getattr(
                vt, "favorable_dwell_started_mono", None
            ),
            "cleared_via_favorable_total": int(
                getattr(vt, "cleared_via_favorable_total", 0) or 0
            ),
            "cleared_via_ceiling_total": int(
                getattr(vt, "cleared_via_ceiling_total", 0) or 0
            ),
        }
    except Exception:
        out["vol_trend_gate"] = None

    # v1.4.107 Phase 1B — shock_gate. Binary acute-spike defence;
    # the locked/cooldown machinery is owned by ``app/shock_gate.py``
    # (`ShockGateState` mutated on every quote tick by
    # ``shock_gate.observe()``).
    try:
        from app import shock_gate as _shock_gate_mod

        sg = state.shock_gate
        out["shock_gate"] = _shock_gate_mod.snapshot_dict(
            sg, now_mono=_clock.monotonic()
        )
    except Exception:
        out["shock_gate"] = None

    # v1.4.112 Phase 1C — regime_controller FSM mode label + dwell +
    # transition log + session-cumulative time-in-mode counters.
    # Surfaced in ``state_current.json`` (via this snapshot) AND
    # heartbeat (via the same path) AND live_stats (1E reads it
    # from here for the top-level ``regime_mode`` block).
    try:
        from app import regime_controller as _regime_controller_mod

        rc = state.regime_controller
        # Phase 4G.5 (v1.4.211) — pass the latest forward signal
        # reading so the snapshot's ``forward_signal`` block surfaces
        # the current classification + indicator diagnostics. ``None``
        # when the forward layer is disabled (default) — the block
        # still renders with placeholder fields.
        _fwd_reading = getattr(state, "last_forward_signal_reading", None)
        out["regime_mode"] = _regime_controller_mod.snapshot_dict(
            rc, now_mono=_clock.monotonic(), forward_reading=_fwd_reading
        )
    except Exception:
        out["regime_mode"] = None

    # v1.4.113 Phase 1D — post-reduction re-entry cooldown. Always
    # renders (per memory note ``feedback_bot_stats_panels_always_render``).
    # Fields are dormant-safe when no fill has armed the cooldown yet.
    try:

        prc_seconds = float(
            getattr(state.settings, "post_reduction_cooldown_seconds", 0.0)
            or 0.0
        )
        armed_at = state.last_inventory_reduction_at_mono
        if armed_at is None or prc_seconds <= 0.0:
            remaining = 0.0
            active = False
        else:
            elapsed = _clock.monotonic() - float(armed_at)
            remaining = max(0.0, prc_seconds - elapsed)
            active = remaining > 0.0
        sup_side = state.last_inventory_reduction_suppressed_side
        # v1.4.144 Phase 2K.1 — exit-attribution counters surface so
        # the operator can see whether the favorable-exit predicate
        # (util-based early clearing) is doing meaningful work, or
        # whether the MAX cooldown ceiling is still the binding
        # constraint. Ratio of favorable / (favorable + ceiling)
        # over a session is the operator's calibration signal for
        # the clear_util_pct knob.
        out["post_reduction_cooldown"] = {
            "enabled": prc_seconds > 0.0,
            "cooldown_seconds": prc_seconds,
            "clear_util_pct": float(
                getattr(
                    state.settings,
                    "post_reduction_cooldown_clear_util_pct",
                    0.30,
                )
                or 0.0
            ),
            "active": bool(active),
            "seconds_remaining": round(remaining, 2),
            "suppressed_side": (
                sup_side.name if sup_side is not None else None
            ),
            "fire_count": int(state.post_reduction_cooldown_fire_count),
            "cleared_via_favorable_total": int(
                state.post_reduction_cooldown_cleared_via_favorable_total
            ),
            "cleared_via_ceiling_total": int(
                state.post_reduction_cooldown_cleared_via_ceiling_total
            ),
        }
    except Exception:
        out["post_reduction_cooldown"] = None

    # v1.4.228 — Phase 4G.13 structural-bias auto-throttle.
    # Surfaces the gate's current armed state + session-cumulative
    # fire-tick count. Operator reads this alongside the v1.4.220
    # Bias card to verify: when Bias card shows RED (ratio ≥ 5×),
    # this block should show ``active=true`` and a matching
    # direction. Mismatch = either the gate is disabled, or the
    # min-samples floor hasn't been reached yet.
    try:
        out["structural_bias_throttle"] = {
            "enabled": bool(
                getattr(state.settings, "structural_bias_auto_throttle_enabled", False)
            ),
            "ratio_threshold": float(
                getattr(state.settings, "structural_bias_auto_throttle_ratio_threshold", 5.0)
            ),
            "min_samples": int(
                getattr(state.settings, "structural_bias_auto_throttle_min_samples", 20)
            ),
            "active": bool(
                getattr(state, "structural_bias_throttle_active_last_tick", False)
            ),
            "direction": getattr(
                state, "structural_bias_throttle_direction", None
            ),
            # v1.5.155: ``fire_count`` is "evaluated and the position
            # guard passed", which can include ticks where some
            # upstream gate (e.g. shock_gate HOLD_ALL) had already
            # narrowed eligibility past this throttle's cap — so the
            # throttle's contribution was a no-op. The
            # ``changed_eligibility_count`` is the actionable number:
            # ticks where this gate's cap was the binding constraint.
            "fire_count": int(
                getattr(state, "structural_bias_throttle_fire_count_total", 0) or 0
            ),
            "changed_eligibility_count": int(
                getattr(state, "structural_bias_throttle_changed_eligibility_total", 0) or 0
            ),
            "last_ratio": float(
                getattr(state, "structural_bias_throttle_last_ratio", 0.0) or 0.0
            ),
            "last_bid_count": int(
                getattr(state, "structural_bias_throttle_last_bid_count", 0) or 0
            ),
            "last_ask_count": int(
                getattr(state, "structural_bias_throttle_last_ask_count", 0) or 0
            ),
        }
    except Exception:
        out["structural_bias_throttle"] = None

    # session_drawdown ladder.
    try:
        sd = state.session_drawdown
        until = float(getattr(sd, "cooldown_until_mono", 0.0) or 0.0)
        remaining = max(0.0, until - _clock.monotonic())
        tier = getattr(sd, "tier", None)
        tier_str = (
            getattr(tier, "value", None)
            or (str(tier) if tier is not None else "CLEAR")
        )
        out["session_drawdown"] = {
            "tier": tier_str,
            "cooldown_seconds_remaining": round(remaining, 2),
            "test_resume_fills_remaining": int(
                getattr(sd, "test_resume_fills_remaining", 0) or 0
            ),
            "last_trigger_pnl_usd": float(
                getattr(sd, "last_trigger_pnl_usd", 0.0) or 0.0
            ),
            "last_transition_iso": getattr(sd, "last_transition_iso", None),
            "fire_count": int(getattr(sd, "fire_count", 0) or 0),
        }
    except Exception:
        out["session_drawdown"] = None

    # v1.5.202 — SF-event-count fatigue ladder. Mirrors
    # session_drawdown shape so the dashboard's chip can render the
    # two side-by-side. The settings handle here is brittle (we'd
    # need to pass it through); fall back to the in-state cached
    # values stamped each tick by the eligibility engine. We snapshot
    # via the gate's own helper.
    try:
        from app.sf_fatigue_gate import snapshot_dict as _sf_snapshot

        sf_state = state.sf_fatigue
        # The settings object lives on the Bot, not BotState. We don't
        # have direct access here, so we read whatever the eligibility
        # engine stamped onto sf_state (current_tier, fire_count) and
        # publish a minimal block. The live_stats helper has access to
        # settings + computes the full shape there.
        until_armed = sf_state.pause_armed_at_mono
        remaining_default = 0.0
        if until_armed is not None and sf_state.current_tier in (
            "PAUSE_SHORT",
            "PAUSE_LONG",
        ):
            # Conservative — without settings we don't know the exact
            # budget; surface 0.0 here and rely on live_stats for the
            # accurate value.
            remaining_default = 0.0
        out["sf_fatigue"] = {
            "tier": sf_state.current_tier,
            "events_in_window": len(sf_state.event_timestamps_mono),
            "fire_count": int(getattr(sf_state, "fire_count", 0) or 0),
            "cooldown_seconds_remaining": remaining_default,
        }
    except Exception:
        out["sf_fatigue"] = None

    # basis_regime gate output (1.2.8). The gate itself is stateless
    # (re-evaluated each tick from the BasisRegimeClassifier's IC),
    # but we publish the current size-shrink decision here so
    # postmortem analysis can answer "was the gate firing at this
    # snapshot, and what mult did it apply?" without joining
    # against quote_decisions rows.
    try:
        mult = float(getattr(state, "basis_regime_size_mult", 1.0) or 1.0)
        reason = getattr(state, "basis_regime_size_mult_reason", None)
        out["basis_regime"] = {
            "active": mult < 1.0 - 1e-12,
            "size_mult": round(mult, 4),
            "reason": reason,
        }
    except Exception:
        out["basis_regime"] = None

    # v1.4.170 Phase 4F — elevated-vol auto-pause. Macro defense:
    # when ``vol_spike_ratio`` stays above the arm threshold for the
    # arm dwell, the bot's eligibility is forced to HOLD_ALL until
    # vol normalises (clear dwell) or the MAX ceiling fires.
    # Operator reads ``armed_total`` + ``cleared_via_favorable_total``
    # / ``cleared_via_ceiling_total`` to verify the gate is firing
    # on real regime shifts (not spuriously) and to tune the arm
    # ratio + dwell.
    try:
        now_mono = _clock.monotonic()
        active = bool(
            getattr(state, "vol_auto_pause_active", False)
        )
        active_since = float(
            getattr(state, "vol_auto_pause_active_since_mono", 0.0)
            or 0.0
        )
        elapsed_active = (
            max(0.0, now_mono - active_since) if active else 0.0
        )
        arm_ratio = float(
            getattr(
                state.settings,
                "vol_auto_pause_arm_ratio",
                0.0,
            )
            or 0.0
        )
        out["vol_regime_auto_pause"] = {
            "enabled": arm_ratio > 0.0,
            "arm_ratio": arm_ratio,
            "arm_dwell_seconds": float(
                getattr(
                    state.settings,
                    "vol_auto_pause_arm_dwell_seconds",
                    60.0,
                )
                or 60.0
            ),
            "clear_ratio": float(
                getattr(
                    state.settings,
                    "vol_auto_pause_clear_ratio",
                    1.3,
                )
                or 1.3
            ),
            "clear_dwell_seconds": float(
                getattr(
                    state.settings,
                    "vol_auto_pause_clear_dwell_seconds",
                    120.0,
                )
                or 120.0
            ),
            "max_pause_seconds": float(
                getattr(
                    state.settings,
                    "vol_auto_pause_max_seconds",
                    1800.0,
                )
                or 1800.0
            ),
            # Most-recent vol_spike_ratio from the toxicity engine
            # (so the operator sees what the gate is reading without
            # cross-referencing the toxicity block).
            "current_vol_spike_ratio": getattr(
                getattr(state, "toxicity", None),
                "vol_spike_ratio",
                None,
            ),
            "active": active,
            "active_since_mono": active_since,
            "active_elapsed_seconds": round(elapsed_active, 1),
            "arm_dwell_in_progress": getattr(
                state, "vol_auto_pause_arm_dwell_started_mono", None
            )
            is not None,
            "clear_dwell_in_progress": getattr(
                state, "vol_auto_pause_clear_dwell_started_mono", None
            )
            is not None,
            "armed_total": int(
                getattr(state, "vol_auto_pause_armed_total", 0) or 0
            ),
            "cleared_via_favorable_total": int(
                getattr(
                    state,
                    "vol_auto_pause_cleared_via_favorable_total",
                    0,
                )
                or 0
            ),
            "cleared_via_ceiling_total": int(
                getattr(
                    state,
                    "vol_auto_pause_cleared_via_ceiling_total",
                    0,
                )
                or 0
            ),
        }
    except Exception:
        out["vol_regime_auto_pause"] = None

    # v1.4.165 Phase 4E — target-venue fast-move cancel observability.
    # Surfaces the latest 500 ms mid-return kinematic, the threshold,
    # and the per-side cumulative cancel counters. Operator reads the
    # bid/ask cancel totals over a session to verify the gate is
    # firing on real moves (and tune the threshold).
    try:
        out["target_venue_fast_move_cancel"] = {
            "enabled": float(
                getattr(
                    state.settings,
                    "target_venue_cancel_on_move_bps",
                    0.0,
                )
                or 0.0
            )
            > 0.0,
            "threshold_bps": float(
                getattr(
                    state.settings,
                    "target_venue_cancel_on_move_bps",
                    0.0,
                )
                or 0.0
            ),
            "last_mid_return_500ms_bps": getattr(
                state, "last_mid_return_500ms_bps", None
            ),
            "cancel_bid_total": int(
                getattr(
                    state,
                    "target_venue_fast_move_cancel_bid_total",
                    0,
                )
                or 0
            ),
            "cancel_ask_total": int(
                getattr(
                    state,
                    "target_venue_fast_move_cancel_ask_total",
                    0,
                )
                or 0
            ),
        }
    except Exception:
        out["target_venue_fast_move_cancel"] = None

    # v1.4.164 Phase 4C.1+4C.2 — expected-edge per-side refusal.
    # Surfaces per-side current expected_edge_bps + refused flag +
    # session-cumulative arm/clear counters. Operator reads
    # ``armed_total / cleared_total`` per side to verify the gate is
    # firing as expected and to calibrate
    # ``MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE``.
    try:
        out["expected_edge_side_refuse"] = {
            "enabled": float(
                getattr(
                    state.settings,
                    "min_expected_net_edge_bps_per_side",
                    0.0,
                )
                or 0.0
            )
            < 0.0,
            "min_expected_edge_bps": float(
                getattr(
                    state.settings,
                    "min_expected_net_edge_bps_per_side",
                    0.0,
                )
                or 0.0
            ),
            "hysteresis_ticks": int(
                getattr(
                    state.settings,
                    "expected_edge_hysteresis_ticks",
                    3,
                )
                or 3
            ),
            "recovery_margin_bps": float(
                getattr(
                    state.settings,
                    "expected_edge_recovery_margin_bps",
                    0.2,
                )
                or 0.2
            ),
            "bid": {
                "refused": bool(
                    getattr(state, "expected_edge_refused_bid", False)
                ),
                "current_expected_edge_bps": getattr(
                    state, "expected_edge_last_bid_bps", None
                ),
                "recovery_ticks": int(
                    getattr(
                        state, "expected_edge_recovery_ticks_bid", 0
                    )
                    or 0
                ),
                "armed_total": int(
                    getattr(state, "expected_edge_armed_bid_total", 0)
                    or 0
                ),
                "cleared_total": int(
                    getattr(
                        state, "expected_edge_cleared_bid_total", 0
                    )
                    or 0
                ),
            },
            "ask": {
                "refused": bool(
                    getattr(state, "expected_edge_refused_ask", False)
                ),
                "current_expected_edge_bps": getattr(
                    state, "expected_edge_last_ask_bps", None
                ),
                "recovery_ticks": int(
                    getattr(
                        state, "expected_edge_recovery_ticks_ask", 0
                    )
                    or 0
                ),
                "armed_total": int(
                    getattr(state, "expected_edge_armed_ask_total", 0)
                    or 0
                ),
                "cleared_total": int(
                    getattr(
                        state, "expected_edge_cleared_ask_total", 0
                    )
                    or 0
                ),
            },
        }
    except Exception:
        out["expected_edge_side_refuse"] = None

    # v1.5.149 Phase 4C.2.a — dampen band publisher. Pairs with the
    # refuse-band block above; surfaces per-side fire counters + the
    # last-tick widening bps so the operator can see (a) whether
    # the band is active config-wise, (b) how often it fires, and
    # (c) what bps it's currently adding. The widening is also
    # visible in ``quote_decisions.spread_composition`` via the
    # ``negative_expectancy_dampen_{bid,ask}_bps`` fields, but those
    # are per-tick; this block summarises session-cumulative state.
    try:
        out["negative_expectancy_dampen"] = {
            "enabled": float(
                getattr(
                    state.settings,
                    "expected_edge_dampen_widen_bps",
                    0.0,
                )
                or 0.0
            )
            > 0.0,
            "widen_bps_config": float(
                getattr(
                    state.settings,
                    "expected_edge_dampen_widen_bps",
                    0.0,
                )
                or 0.0
            ),
            "max_bps_config": float(
                getattr(
                    state.settings,
                    "expected_edge_dampen_max_bps",
                    0.0,
                )
                or 0.0
            ),
            "bid": {
                "fired_total": int(
                    getattr(
                        state,
                        "negative_expectancy_dampen_bid_total",
                        0,
                    )
                    or 0
                ),
                "last_widen_bps": float(
                    getattr(
                        state,
                        "negative_expectancy_dampen_last_bid_bps",
                        0.0,
                    )
                    or 0.0
                ),
            },
            "ask": {
                "fired_total": int(
                    getattr(
                        state,
                        "negative_expectancy_dampen_ask_total",
                        0,
                    )
                    or 0
                ),
                "last_widen_bps": float(
                    getattr(
                        state,
                        "negative_expectancy_dampen_last_ask_bps",
                        0.0,
                    )
                    or 0.0
                ),
            },
        }
    except Exception:
        out["negative_expectancy_dampen"] = None

    # v1.5.149 Phase 4D.3 + 4G.10 — SF slice + SHOCK full-dark
    # publisher. Both are profile knobs whose effect is only
    # observable from session counters. SF slice bumps each
    # phase-2/3 IOC where the slice cap actually trimmed the size;
    # SHOCK full-dark is a binary config toggle, surfaced for
    # dashboard / postmortem so the reader can correlate the
    # ladder rung_dropped_regime_mode_total counter with the
    # toggle state.
    try:
        out["sf_slice"] = {
            "enabled": float(
                getattr(
                    state.settings,
                    "soft_flatten_slice_notional_usd",
                    0.0,
                )
                or 0.0
            )
            > 0.0,
            "slice_notional_usd_config": float(
                getattr(
                    state.settings,
                    "soft_flatten_slice_notional_usd",
                    0.0,
                )
                or 0.0
            ),
            "dispatched_total": int(
                getattr(state, "sf_slice_dispatched_total", 0) or 0
            ),
        }
    except Exception:
        out["sf_slice"] = None

    try:
        out["shock_ladder_full_dark"] = {
            "enabled": bool(
                getattr(
                    state.settings,
                    "shock_ladder_allow_full_dark",
                    False,
                )
            ),
        }
    except Exception:
        out["shock_ladder_full_dark"] = None

    # v1.4.160 Phase 2K.9 — fill_burst_detector favorable-exit
    # attribution. Surfaces the size-shrink cooldown state + new
    # attribution counters. Operator reads
    # ``cleared_via_favorable / (cleared_via_favorable + ceiling)`` to
    # calibrate ``FILL_BURST_CLEAR_BAND_MULT`` and
    # ``FILL_BURST_FAVORABLE_EXIT_DWELL_SECONDS``.
    try:
        fbd = getattr(state, "fill_burst_detector", None)
        if fbd is not None:
            out["fill_burst_detector"] = fbd.snapshot_dict(
                _clock.monotonic()
            )
        else:
            out["fill_burst_detector"] = None
    except Exception:
        out["fill_burst_detector"] = None

    # v1.4.156 Phase 2K.6 — at_touch_adverse_pause favorable-exit
    # attribution. The gate is per-side (BUY / SELL), so the block
    # surfaces both pause states + both attribution counters. Operator
    # reads ``cleared_via_favorable / (cleared_via_favorable + ceiling)``
    # PER SIDE to calibrate ``AT_TOUCH_ADVERSE_PAUSE_CLEAR_BAND_MULT``
    # and ``AT_TOUCH_ADVERSE_PAUSE_FAVORABLE_EXIT_DWELL_SECONDS``.
    try:
        atap = getattr(state, "at_touch_adverse_pause", None)
        if atap is not None:
            out["at_touch_adverse_pause"] = atap.snapshot_dict(
                _clock.monotonic()
            )
        else:
            out["at_touch_adverse_pause"] = None
    except Exception:
        out["at_touch_adverse_pause"] = None

    # v1.4.161 Phase 4C.3 mini — per-side realised-edge side suppression.
    # Out-of-order delivery of one Phase 4C item; targets the
    # v1.4.157-260520-213540 failure mode (resting SELL picked off into
    # a rally before any util-gated defence could fire). Mirrors the
    # ``at_touch_adverse_pause`` block shape so the dashboard can use
    # the same rendering code.
    try:
        ress = getattr(state, "realised_edge_side_suppress", None)
        if ress is not None:
            out["realised_edge_side_suppress"] = ress.snapshot_dict(
                _clock.monotonic()
            )
        else:
            out["realised_edge_side_suppress"] = None
    except Exception:
        out["realised_edge_side_suppress"] = None

    # v1.4.160 Phase 2K.8 — vol_spike favorable-exit attribution.
    # The vol_spike latch (``state.vol_spike_until_mono``) extends
    # ``VOL_SPIKE_COOLDOWN_SECONDS`` past each threshold crossing; the
    # Phase 2K.8 predicate clears it early when vol calms past the
    # band for the dwell duration. Surfaces the latch state + tier
    # label + attribution counters for dashboard calibration.
    try:
        until = float(
            getattr(state, "vol_spike_until_mono", 0.0) or 0.0
        )
        remaining = max(0.0, until - _clock.monotonic())
        vra = getattr(state, "vol_regime_adjustment", None)
        out["vol_spike"] = {
            "active": remaining > 0.0,
            "seconds_remaining": round(remaining, 2),
            "tier_name": getattr(vra, "tier_name", "off"),
            "shrink_factor": float(
                getattr(vra, "shrink_factor", 1.0) or 1.0
            ),
            "in_spike_window": bool(
                getattr(vra, "in_spike_window", False)
            ),
            "half_spread_bump_bps": float(
                getattr(vra, "half_spread_bump_bps", 0.0) or 0.0
            ),
            "favorable_dwell_started_mono": getattr(
                state,
                "vol_spike_favorable_dwell_started_mono",
                None,
            ),
            "cleared_via_favorable_total": int(
                getattr(
                    state,
                    "vol_spike_cleared_via_favorable_total",
                    0,
                )
                or 0
            ),
            "cleared_via_ceiling_total": int(
                getattr(
                    state,
                    "vol_spike_cleared_via_ceiling_total",
                    0,
                )
                or 0
            ),
        }
    except Exception:
        out["vol_spike"] = None

    # v1.4.158 Phase 2K.7 — mae_gate favorable-exit attribution.
    # Surfaces the 30s-horizon defensive gate's cooldown state + new
    # attribution counters. Operator reads
    # ``cleared_via_favorable / (cleared_via_favorable + ceiling)`` to
    # calibrate ``MAE_GATE_CLEAR_BAND_MULT`` and
    # ``MAE_GATE_FAVORABLE_EXIT_DWELL_SECONDS``.
    try:
        mg = getattr(state, "mae_gate", None)
        if mg is not None:
            now_mono = _clock.monotonic()
            remaining = max(0.0, float(mg.cooldown_until_mono) - now_mono)
            avg_bps: float | None = None
            try:
                if len(mg.samples) > 0:
                    avg_bps = sum(mg.samples) / len(mg.samples)
            except Exception:
                avg_bps = None
            out["mae_gate"] = {
                "active": remaining > 0.0,
                "seconds_remaining": round(remaining, 2),
                "last_trigger_avg_bps": float(
                    getattr(mg, "last_trigger_avg_bps", 0.0) or 0.0
                ),
                "current_avg_bps": (
                    None if avg_bps is None else float(avg_bps)
                ),
                "sample_count": int(len(getattr(mg, "samples", []) or [])),
                "fire_count": int(getattr(mg, "fire_count", 0) or 0),
                "favorable_dwell_started_mono": getattr(
                    mg, "favorable_dwell_started_mono", None
                ),
                "cleared_via_favorable_total": int(
                    getattr(mg, "cleared_via_favorable_total", 0) or 0
                ),
                "cleared_via_ceiling_total": int(
                    getattr(mg, "cleared_via_ceiling_total", 0) or 0
                ),
                "cleared_via_position_favorable_total": int(
                    getattr(mg, "cleared_via_position_favorable_total", 0) or 0
                ),
            }
        else:
            out["mae_gate"] = None
    except Exception:
        out["mae_gate"] = None

    # 1.2.14: multi-level ladder. The most-recent LadderDecision
    # built by the bot's quote cycle, serialised to a JSON-safe
    # dict. At ``LADDER_NUM_LEVELS_PER_SIDE=1`` (default) this is
    # a single rung per side that mirrors the inside scalar quote.
    # At N > 1 the multi-rung ladder is shown — useful in shadow
    # mode where execution still single-rung but the operator
    # wants to see "what would the ladder look like".
    try:
        last_ladder = getattr(state, "last_ladder_decision", None)
        out["ladder"] = (
            last_ladder.to_dict() if last_ladder is not None else None
        )
    except Exception:
        out["ladder"] = None

    return out

from app.book_history import TopOfBookRow, match_top_of_book, snapshot_fullness
from app.config import Settings
from app.fill_bucket_metrics import FillBucketAggregator
from app.order_trace import OrderTraceBuffer
from app.market_data_gap_stats import MarketDataGapTracker, market_snapshot_eligible_for_gap_stats
from app.market_data_timing import PublicWsTimingTracker
from app.enums import BotStatus, DesyncPhase, OrderStatus, QuoteEligibility, Side
from app.persistent_runtime_state import PersistentRuntimeState
from app.markout import PendingMarkoutJob
from app.inbound_timing import InboundPrivateTiming, InboundPublicTiming, public_timing_derived_ms
from app.vol_regime import VolRegimeAdjustment
from app.models import (
    AccountSnapshot,
    BestBidAsk,
    Fill,
    PnlSnapshot,
    PositionSnapshot,
    ToxicitySnapshot,
    WorkingOrder,
)
from app.quote_quality_telemetry import QuoteQualityRollup, build_net_edge_summary
from app.runtime_toxicity_aggregate import build_runtime_toxicity_summary
from app.storage import Storage
from app.utils.time import utc_now


_EXECUTION_ERRORS_WINDOW_MAXLEN = 1024


class BotState:
    """Thread-safe in-memory state for API + bot loop.

    v1.4.82 wedge-elimination-cleanup Phase 3 ARCHITECTURE
    ======================================================

    Historically a 3,400+-line god-class owning every piece of mutable
    state in the bot. Phase 3 dissolves it into focused stores:

      * ``order_store``     — working orders + (oid, cloid) indexes
      * ``position_store``  — position state + reducing-side / headroom
      * ``market_store``    — best-bid/ask + spread / freshness
      * ``telemetry_store`` — executor-state aggregates + health

    New callers SHOULD reach state through the stores. The legacy
    attribute access (``state._working_orders``, ``state.working_bid``,
    ``state.position``, ``state.market``) is preserved through the
    Phase 3-5 incremental migration but the stores are the canonical
    surface for any new code.

    Single-tick consumers (decision layer, dispatcher, reconciler)
    should build ``state.tick_snapshot()`` ONCE at tick start and
    read all four views from the frozen ``TickSnapshot``. This is
    the Phase 5A pattern — Phase 3C ships the snapshot infrastructure
    and Phase 5A is the cutover that wires the hot path.

    Lock policy
    -----------
    Today ``state._lock`` is the global serialization point for every
    state mutation across the bot. The 200+ ``with state._lock:``
    sites in the WS handlers, the executor, and the dispatcher are
    correct AS-IS and are NOT migrated by Phase 3D. The per-store
    lock split is a perf optimization sequenced to Phase 5A, where
    the snapshot cutover makes it observably valuable (today the
    global lock is held briefly enough that splitting it has no
    measurable effect on the hot path).
    """

    @property
    def settings(self) -> Settings:
        """Read-only public accessor for the bot's Settings object.

        Added 2026-05-13 (regime-observability bugfix) — the v1.3.0
        ``fill_ingestion`` and ``exposure_bar_emitter`` modules
        attempted ``state.settings.max_abs_position`` lookups that
        silently returned 0 because ``settings`` wasn't a public
        attribute. Exposed as a property (read-only) so external
        consumers don't need to reach into the private
        ``_settings`` slot.
        """
        return self._settings

    @property
    def adaptive_widen_active(self) -> bool:
        """Whether the adaptive-widen overlay is currently armed.

        Derived from the deadline ``adaptive_spread_widen_until_mono``
        — there is no dedicated boolean state field. Returns True iff
        the current monotonic clock is less than the deadline.

        Added 2026-05-13 (Codex bug review MED #5). Pre-fix,
        ``execution.py::_stage_place_order_local`` and the fill-side
        decision-state stamping path did
        ``getattr(state, "adaptive_widen_active", False)`` on an
        attribute that did not exist — defaulting to False — so
        ``adaptive_widen_active_at_decision`` was permanently False
        on every order / fill row. Exposure bars + live_stats
        already derived the flag correctly from the deadline (in
        ``app/exposure_bar_emitter.py`` and ``app/live_stats.py``
        respectively); this property exists so all three consumers
        share one source of truth and the same derivation can't drift.
        """
        try:
            return _clock.monotonic() < float(
                self.adaptive_spread_widen_until_mono or 0.0
            )
        except (TypeError, ValueError):
            return False

    def __init__(self, settings: Settings) -> None:
        self._lock = threading.RLock()
        self._settings = settings
        self.bot_status = BotStatus.STARTING
        self.symbol = settings.symbol
        self.market: Optional[BestBidAsk] = None
        self.account: Optional[AccountSnapshot] = None
        self.position: PositionSnapshot = PositionSnapshot(
            symbol=settings.symbol,
            position_qty=0.0,
            avg_entry_price=None,
            mark_price=None,
            position_notional=0.0,
            unrealized_pnl_usd=0.0,
        )
        # 1.3.130 multi-rung Phase 2 — per-rung working set, keyed by
        # (side, level_idx). Replaces the scalar working_bid / working_ask
        # storage. The ``working_bid`` / ``working_ask`` @property shims
        # below preserve the original scalar interface for the 60+ read
        # sites that only care about the inside rung (level_idx=0).
        #
        # Invariants:
        #   * Outer keys are exactly {Side.BUY, Side.SELL}.
        #   * Inner keys are non-negative ints (0 = inside rung).
        #   * Setting ``working_bid = None`` deletes the (BUY, 0) entry
        #     entirely; the dict shape doesn't accumulate None sentinels.
        #   * level_idx on the stored WorkingOrder must match the dict
        #     key (callers are responsible for this; not defensively
        #     enforced to keep the hot path lean).
        self._working_orders: dict[Side, dict[int, WorkingOrder]] = {
            Side.BUY: {},
            Side.SELL: {},
        }
        # v1.4.80 wedge-elimination-cleanup Phase 3A — OrderStore
        # facade. Provides O(1) (oid, cloid) lookup and a single
        # transition point. Shares the same underlying
        # ``_working_orders`` dict with this class during the
        # incremental migration; Phase 3D moves ownership entirely
        # into the store and removes the dict from ``BotState``.
        #
        # Import here (not at module top) to avoid a circular
        # import: ``app.stores.order_store`` typing-imports
        # ``BotState`` for the constructor signature.
        from app.stores.order_store import OrderStore
        from app.stores.position_store import PositionStore
        from app.stores.market_store import MarketStore
        from app.stores.telemetry_store import TelemetryStore
        self.order_store: OrderStore = OrderStore(self)
        self.position_store: PositionStore = PositionStore(self)
        self.market_store: MarketStore = MarketStore(self)
        self.telemetry_store: TelemetryStore = TelemetryStore(self)
        self.recent_fills: deque[Fill] = deque(maxlen=1000)
        # v1.5.197 — monotonic timestamp of the most-recent fill ingested.
        # Used by defensive gates to detect "idle" periods (long stretches
        # with no fills) and self-clear stale state. Pre-v1.5.197 there
        # was no idle-aware path — gates relied on new fills arriving to
        # refresh their predicates, but if the gate itself suppressed
        # fills, the predicate stayed stale indefinitely (deadlock observed
        # in v1.5.195 snapshot post-mortem).
        #
        # ``None`` until the first fill arrives. Set on every fill in
        # ``record_fill``. Monotonic time (not wall) so it's robust to
        # clock skew. Consumers compute ``now_mono - last_fill_at_mono``
        # and compare to their idle-clear threshold.
        self.last_fill_at_mono: Optional[float] = None
        # Monotonic session-scoped counters of unique fills observed. Grows
        # without bound over a session; unlike ``len(recent_fills)`` it does
        # NOT cap at the deque maxlen. Consumed by self-perpetuation guards
        # (adaptive_spread_widen, adverse_side_pause) that need a strictly-
        # monotonic "has new data arrived?" signal — ``len(recent_fills)``
        # would stop growing after 1000 fills, breaking the guards on long
        # sessions (a multi-day session at 2 fills/min hits maxlen).
        self.session_fill_count: int = 0
        self.session_fill_count_by_side: dict[Any, int] = {}
        # Cumulative session-scoped traded notional (USD). Grows by
        # ``abs(fill.notional)`` on every fill — never decremented.
        # Surfaced via the heartbeat for the dashboard's session
        # activity card. Independent of recent_fills (which is capped
        # at 1000) so it stays accurate over long sessions.
        self.session_traded_notional_usd: float = 0.0
        self.vol_sigma: Optional[float] = None
        # v1.5.232 — None during warm-up (< VOL_WINDOW_SAMPLES distinct
        # mids in app/volatility.py), float once the estimator has
        # produced a real sigma. Pre-v1.5.232 this was 0.0 in both
        # cases, which the dashboard could not tell apart from
        # genuinely-flat market. All consumers are None-safe (see
        # bot.py's `vol_bps_or_none` block for the audit trail).
        self.vol_bps: Optional[float] = None
        # v1.5.239 — EWMA-of-|log-return| trend-aware vol measure
        # (app/vol_abs_ewma.py). Constructed with default half-life
        # 20s; bot.py replaces with a Settings-driven instance during
        # startup. Runs unconditionally — consumer wiring (regime
        # classifier vol_slope) is gated behind
        # REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE.
        from app.vol_abs_ewma import VolAbsEwmaEstimator as _VAE
        self.vol_abs_ewma: _VAE = _VAE()
        # Current EWMA value in bp, mirrored each tick from the
        # estimator. Same Optional[float] contract as ``vol_bps`` —
        # ``None`` during warm-up (before the second mid push).
        self.vol_abs_ewma_bps: Optional[float] = None

        # v1.5.277 / AQC Phase 1 — Active Quoting Controller.
        # Constructed with default AQCSettings; bot.py replaces the
        # settings during startup with values driven by Settings
        # (AQC_ENABLED, AQC_TARGET_NET_EDGE_PER_MIN_USD, etc.).
        # Phase 1 is observe-only: nothing consumes
        # ``aggression_level`` for trading decisions yet — the value
        # is published in ``snapshot_dict()`` so the operator can
        # compare against actual behavior on the same snapshots.
        from app.active_quoting_controller import (
            ActiveQuotingController as _AQC,
        )
        self.active_quoting_controller: _AQC = _AQC()
        # Phase 8A (v1.5.185) — Avellaneda-Stoikov k-intensity
        # cache. Refreshed every ``AS_K_INTENSITY_REFRESH_SECONDS``
        # (default 60 s) by the bot's main loop, NOT per-tick, so
        # the O(N) scan over ``recent_fills`` (cap 1000) stays off
        # the hot path. Consumed by ``compute_quote_decision`` when
        # ``AVELLANEDA_STOIKOV_ENABLED=true``.
        # ``None`` until the first refresh OR when AS is disabled.
        self.as_k_intensity_per_min: Optional[float] = None
        self.as_k_intensity_last_refresh_mono: float = 0.0
        # Phase 8A Option B (v1.5.189) — session-cumulative counter
        # of ticks where the AS path produced the base half-spread
        # (i.e. AVELLANEDA_STOIKOV_ENABLED was true at decision
        # time). Surfaced in the snapshot for AC verification:
        # operator-facing "did AS actually fire" sanity check that
        # doesn't rely on log-scraping.
        self.as_path_fire_count: int = 0
        # v1.5.158 Option A — rolling history of (mono_ts, vol_bps)
        # samples for the vol-climbing-anticipatory-widening gate.
        # Appended once per tick in ``app/bot.py`` right after
        # vol_bps is updated. Sized for ~30 minutes at the typical
        # 2 Hz quote cadence with headroom (3600 samples = ~30 min
        # at 2 Hz; bumped to 7200 to be safe under bursty cadences).
        # The gate computes a short-window MA and a long-window MA
        # over this deque and fires when the ratio (short/long)
        # exceeds the configured threshold — vol is climbing
        # measurably faster than its longer-horizon baseline.
        from collections import deque as _deque
        self.vol_bps_history: _deque[tuple[float, float]] = _deque(
            maxlen=7200
        )
        # Monotonic deadline: while the monotonic clock is less than this, quote spread floor uses adaptive overlay.
        self.adaptive_spread_widen_until_mono: float = 0.0
        # Latched when quote_quality signals widen; cleared when adaptive_spread_widen_until_mono expires.
        self.quote_quality_widen_latched: bool = False
        # Reason the most recent adaptive-widen overlay was armed. One of
        # ``"toxicity_hard"``, ``"toxicity_soft"``, ``"markout_adverse"``,
        # ``"one_sided_ratio"``, ``"quote_quality"``, or None when no
        # overlay has been armed in the current session. Surfaced via
        # the live_stats payload for the dashboard's Market tab so the
        # operator can see why spreads widened, not just that they did.
        self.adaptive_spread_widen_reason: Optional[str] = None
        # Recent-fills count at the moment of the last adaptive-widen arming.
        # Used to prevent self-perpetuating re-arming: once the deadline passes,
        # we require at least one new fill before arming again. Without this, a
        # single stale adverse signal (e.g. 3:1 fill ratio from the first flurry)
        # keeps arming the widen forever because the widen itself prevents new
        # fills — observed in ``tmp/snap_20260418_094415``: 4 fills in minute 1,
        # widen armed on one_sided_fill_ratio=0.75, 17 minutes of invisible
        # quoting followed because the same 4 fills stayed in the window.
        self.adaptive_spread_widen_arm_n_fills: int = -1
        # v1.4.155 Phase 2K.5 — favorable-exit support for the
        # adaptive_spread_widen cooldown. Each of the six trigger
        # reasons (toxicity_hard / toxicity_soft / markout_adverse /
        # one_sided_ratio / quote_quality / slow_trend) has its own
        # "arming signal cleared" predicate; when the predicate
        # holds for ``favorable_exit_dwell_seconds``, the widen
        # clears EARLY rather than waiting the full
        # ``toxicity_cooldown_seconds`` deadline. Cooldown stays as
        # MAX-ceiling safety net.
        self.adaptive_spread_widen_favorable_dwell_started_mono: Optional[
            float
        ] = None
        self.adaptive_spread_widen_cleared_via_favorable_total: int = 0
        self.adaptive_spread_widen_cleared_via_ceiling_total: int = 0
        # v1.5.157 — position-aware favorable-exit attribution.
        # Mirrors mae_gate / realised_edge_side_suppress pattern. The
        # markout-based exit (above) requires the ARM-TIME signal
        # (markout_adverse / quote_quality / slow_trend) to clear,
        # which often doesn't happen during a sustained adverse
        # regime → cleared via ceiling 88x vs 10 favorable in the
        # v1.5.154-260526-074029 snapshot. The position-aware exit
        # adds a second path: clear when bot has significant
        # inventory AND drift is moving favorably for that
        # inventory (good time to unwind, the widening is now
        # actively obstructing the unwind).
        self.adaptive_spread_widen_cleared_via_position_favorable_total: int = 0
        # Set to True while the cooldown is active; flipped to False
        # on the call that clears it. Edge detector for attribution
        # — without it we can't distinguish "deadline lapsed naturally"
        # from "deadline already cleared, signal still elevated".
        self.adaptive_spread_widen_was_active_last_tick: bool = False
        self.toxicity: ToxicitySnapshot = ToxicitySnapshot(
            score=0.0,
            one_sided_fill_ratio=0.0,
            avg_adverse_markout_bps=0.0,
            vol_spike_ratio=1.0,
            hard_trigger=False,
            soft_trigger=False,
            delayed_markout_sample_count=0,
            adverse_uses_delayed_markouts=False,
        )
        self.pnl: PnlSnapshot = PnlSnapshot(
            realized_pnl_usd=0.0,
            unrealized_pnl_usd=0.0,
            total_pnl_usd=0.0,
            fees_usd=0.0,
            equity_usd=None,
            drawdown_usd=0.0,
            session_peak_equity_usd=0.0,
        )
        # Venue-side leverage / margin / position-mode snapshot.
        # Populated once at startup (best-effort REST fetch); the
        # heartbeat publishes them so the dashboard's position panel
        # can show "Leverage 10× / Cross / net_mode" without a live
        # OKX call. None when the venue doesn't expose them or the
        # fetch failed (don't block startup over a display field).
        self.venue_leverage: Optional[str] = None
        self.venue_margin_mode: Optional[str] = None  # "cross" / "isolated"
        self.venue_position_mode: Optional[str] = None  # "net_mode" / "long_short_mode"
        self.kill_reason: Optional[str] = None
        self.kill_timestamp: Optional[datetime] = None
        # Pause-reason + timestamp pair, analogous to kill_reason.
        # Set at every site that transitions ``bot_status`` to PAUSED
        # (manual_pause via Telegram, reconcile_stall, post-flatten
        # failure, etc.) so the dashboard can show *why* the bot is
        # paused rather than just *that* it's paused. Cleared when
        # bot_status leaves PAUSED.
        self.pause_reason: Optional[str] = None
        self.pause_timestamp: Optional[datetime] = None
        self.manual_pause = False
        self.flatten_mode = False
        # Patient post-only-only flatten triggered by the position-
        # drawdown gate. Distinct from ``flatten_mode`` (taker IOC).
        # The bot's quote loop suspends normal placement while this
        # is True; a dedicated worker maintains a single post-only
        # reduce-side order at the best price until the position
        # closes, after which this flag clears and quoting resumes.
        self.soft_flatten_active: bool = False
        self.soft_flatten_started_at_mono: Optional[float] = None
        # v1.5.198 — monotonic time of the LAST SF exit. Used by
        # ``_enter_soft_flatten`` to enforce a re-entry cooldown:
        # when a tox-hard / position-drawdown trigger would re-enter
        # SF within ``soft_flatten_reentry_cooldown_seconds`` of the
        # previous exit, the entry is deferred. Eliminates the
        # tox-hard SF loop pattern (4708 starts in 3 min) observed
        # in v1.5.195-260527-161959. None until the first SF exit.
        self.soft_flatten_last_exited_at_mono: Optional[float] = None
        # v1.5.2 Phase 4D.3 — post-SF cooldown gate. Captured at SF
        # entry; consumed at SF exit to populate the cooldown window
        # below. ``None`` outside SF.
        self.soft_flatten_pre_position_qty: Optional[float] = None
        # v1.5.2 Phase 4D.3 — post-SF cooldown deadline + which side
        # is suppressed. ``post_sf_cooldown_until_mono = 0.0`` means
        # the cooldown is inactive. ``lean_side`` is the side that
        # would RE-ADD to the pre-SF direction (e.g. ``BUY`` when
        # pre-SF was LONG); it's the side the gate FORCES into
        # one-sided-reducing eligibility. Cleared by time (no manual
        # reset path; restart the bot to clear early).
        self.post_sf_cooldown_until_mono: float = 0.0
        self.post_sf_cooldown_lean_side: Optional[Any] = None  # Side
        # Latched display flags (mirror of structural_bias_throttle_*
        # so the dashboard / snapshot can show "this gate fired N
        # times this session"). Reset to False at top of each tick;
        # set to True when the gate engages.
        self.post_sf_cooldown_active_last_tick: bool = False
        self.post_sf_cooldown_fire_count_total: int = 0
        # Per-entry soft-flatten parameters. None means "use default
        # phased behaviour" (phase 1 near-touch → phase 2 far-touch
        # after SOFT_FLATTEN_PHASE_1_SECONDS). When the toxicity
        # path triggers SF, it sets ``force_phase=2`` to skip phase 1
        # entirely and start aggressive at the very first tick.
        self.soft_flatten_force_phase: Optional[int] = None
        # Taker-fallback threshold for the current SF entry (ticks of
        # adverse mid drift since entry; None or 0 = disabled). When
        # set and exceeded, the SF worker calls ``client.market_close``
        # once and exits SF — the "phase 3 escape" for cases where
        # phase 2 isn't filling because the price keeps escaping.
        # The effective cap is also bounded by
        # SOFT_FLATTEN_TAKER_FALLBACK_TICKS in settings (operator's
        # global ceiling); per-entry value cannot exceed it.
        self.soft_flatten_taker_fallback_ticks: Optional[int] = None
        # Mid price at the moment SF entered. Used to compute
        # adverse drift (in ticks) for the phase-3 fallback check.
        # None when the entry tick had no valid mid (extremely rare).
        self.soft_flatten_entry_mid: Optional[float] = None
        # Database id of the current SF episode row in the
        # ``soft_flatten_events`` table. Set on ``_enter_soft_flatten``,
        # used by the SF order placement path to stamp every
        # SF-originated order with this id, looked up by fill ingest
        # to copy the id onto the fill row, and cleared on
        # ``_exit_soft_flatten``. None outside SF episodes. See
        # plans/20260507-sf-frontend.md Phase 2.
        self.soft_flatten_event_id: Optional[int] = None
        # v1.5.33 — take-profit (TP) opportunistic harvest mode.
        # Mirror of the SF state surface: ``tp_active`` flips on entry,
        # ``tp_event_id`` is the storage row id stamped on TP-originated
        # orders + fills, ``tp_armed_at_mono`` drives the dwell timer,
        # ``tp_recent_event_id*`` is a grace cache for late-arriving WS
        # fills (mirrors the SF v1.4.192 fix). ``tp_arm_cooldown_until_mono``
        # blocks immediate re-arm after disarm. See app/take_profit.py.
        self.tp_active: bool = False
        self.tp_armed_at_mono: Optional[float] = None
        self.tp_event_id: Optional[int] = None
        self.tp_target_price: Optional[float] = None
        self.tp_close_side: Optional[Any] = None  # Side
        self.tp_arm_cooldown_until_mono: float = 0.0
        self.tp_entry_upnl_bps: Optional[float] = None
        self.tp_entry_position_qty: Optional[float] = None
        # Grace cache for late WS fills after exit (mirror of
        # state.sf_recent_event_id_*).
        self.tp_recent_event_id: Optional[int] = None
        self.tp_recent_event_id_valid_until_mono: float = 0.0
        self.tp_recent_event_tag_total: int = 0
        # Session-cumulative TP telemetry. Surface in
        # ``snapshot_dict`` + Telegram ``/status`` so the operator can
        # see fire rate and uPnL realised without grepping logs.
        self.tp_armed_total: int = 0
        self.tp_filled_total: int = 0
        self.tp_exited_unfilled_total: int = 0  # timeout / retrace
        self.tp_exited_sf_takeover_total: int = 0
        # Running sums for averages. Computed on demand:
        #   avg_at_arm_bps  = sum_upnl_bps_at_arm  / max(armed_total, 1)
        #   avg_at_fill_bps = sum_upnl_bps_at_fill / max(filled_total, 1)
        self.tp_sum_upnl_bps_at_arm: float = 0.0
        self.tp_sum_upnl_bps_at_fill: float = 0.0
        # Vol-spike persistence window (``app/vol_regime.py`` /
        # BUGS/todo-009.md). When the realised-vol ratio crosses
        # ``VOL_SPIKE_THRESHOLD``, this is set to (now_mono +
        # VOL_SPIKE_COOLDOWN_SECONDS) so the bot stays defensive for
        # a configured window after the spike. ``0.0`` means no spike
        # window currently active. Read every tick by the quote
        # engine + execution sizing path.
        self.vol_spike_until_mono: float = 0.0
        # Phase 2K.8 (v1.4.160) — favorable-exit state for the
        # vol_spike latch. ``evaluate_vol_spike_favorable_exit`` reads
        # and round-trips these each tick.
        self.vol_spike_favorable_dwell_started_mono: Optional[float] = None
        self.vol_spike_was_active_last_call: bool = False
        self.vol_spike_cleared_via_favorable_total: int = 0
        self.vol_spike_cleared_via_ceiling_total: int = 0
        # Latest vol-regime adjustment (shrink_factor +
        # in_spike_window + half_spread_bump_bps). Computed once per
        # tick by Bot.one_tick from ``state.toxicity.vol_spike_ratio``
        # and stored here so quoting + execution can read it without
        # re-doing the computation. Identity-default when the feature
        # is off (``VOL_SHRINK_COEFF=0``). See BUGS/todo-009.md.
        self.vol_regime_adjustment: VolRegimeAdjustment = VolRegimeAdjustment(
            shrink_factor=1.0,
            in_spike_window=False,
            half_spread_bump_bps=0.0,
        )
        # Monotonic timestamp when the position-drawdown threshold
        # first became breached in the current breach episode. Reset
        # to None whenever the position is no-longer adverse beyond
        # threshold. The gate fires when ``now - started_at >=
        # duration_seconds``.
        self.position_drawdown_breach_started_at_mono: Optional[float] = None
        # Counters for the fill-derived shadow position. Incremented
        # in ``record_fill`` whenever a session-scoped fill applies a
        # delta to ``position.position_qty``; ``divergence_count``
        # tracks how often a REST refresh later disagreed materially
        # with the shadow value (diagnostic only -- REST is always
        # the authoritative truth).
        self.shadow_position_apply_count: int = 0
        self.shadow_position_divergence_count: int = 0
        self.shadow_position_last_divergence_qty: Optional[float] = None
        # BUG-034 fix (v1.5.258): counts the number of times a
        # shadow/REST divergence triggered the existing REST-fill
        # catch-up path. When a divergence fires, we set
        # ``private_ws_recovery_pending = True`` so the next tick's
        # ``should_ingest_fills_via_rest()`` returns True, the bot
        # fetches ``/fills`` from REST, and the missed fill is
        # dedup-aware ingested via ``ingest_hl_fill_raw`` (so it
        # lands in fills.jsonl, recent_fills, session counters,
        # markout, etc.). This counter lets the operator see the
        # mechanism firing — if it's incrementing but
        # ``shadow_position_divergence_count`` keeps rising at the
        # same rate, the REST /fills endpoint isn't returning the
        # missing fill within the catch-up window.
        self.shadow_position_divergence_recovery_count: int = 0
        self.killed = False
        self.last_heartbeat: Optional[datetime] = None
        self.last_market_ts: Optional[datetime] = None
        self.loop_counter = 0
        self.execution_errors = 0
        # Rolling window of (monotonic_ts, source) tuples. Used by the risk gate
        # together with ``settings.execution_errors_window_seconds`` to distinguish
        # a sustained error storm (kill-worthy) from slow background noise that
        # would otherwise trip the cumulative counter over hours of uptime.
        # Capped to _EXECUTION_ERRORS_WINDOW_MAXLEN entries as a memory guard.
        self._execution_error_events: deque[tuple[float, str]] = deque(
            maxlen=_EXECUTION_ERRORS_WINDOW_MAXLEN
        )
        self.order_desync = False
        # Per-side desync detail. ``order_desync`` (above) is the OR of
        # the two — kept for back-compat with anything that already
        # consumes the legacy boolean — but the dashboard's Market tab
        # surfaces both sides separately so the operator can tell
        # whether a stuck CANCEL_PENDING is on bid only, ask only, or
        # both. Set together with ``order_desync`` in ``execution.py``.
        self.order_desync_buy: bool = False
        self.order_desync_sell: bool = False
        self.desync_phase = DesyncPhase.OK
        self.desync_consecutive_ticks = 0
        self.desync_quarantine_remaining = 0
        self._book_snapshots: deque[TopOfBookRow] = deque(maxlen=400)
        self.book_age_seconds: Optional[float] = None
        self.stale_book_warning_active: bool = False
        # Market-data recovery (public WS reconnect + fresh BBO; not order lifecycle).
        self.market_data_recovery_started_monotonic: Optional[float] = None
        self.market_data_recovery_logged_stale_detected: bool = False
        self.market_data_recovery_refresh_attempts_episode: int = 0
        # Exponential-backoff between recovery bursts. ``next_burst_mono`` is
        # the monotonic-clock moment at which the next burst is allowed;
        # None means "run the next one immediately" (first burst on stale
        # detection, or post-success reset). ``current_burst_backoff_s``
        # is the wait actually applied after the most recent failed burst
        # (doubles on each subsequent failure, capped by config).
        self.market_data_recovery_next_burst_mono: Optional[float] = None
        self.market_data_recovery_current_burst_backoff_s: float = 0.0

        # Cross-venue reference state — historically Binance-only and
        # named ``binance_*`` throughout this codebase. Today these
        # fields hold whatever reference venue the operator selected
        # via ``REFERENCE_EXCHANGE`` (Binance default; Bybit
        # supported; HTX and others planned). The naming is a legacy
        # artefact — see ``BUGS/todo-014-generic-reference-fields.md``
        # for the planned rename to ``reference_*`` with property
        # aliases. For now: writes from any reference-venue WS handler
        # land in these fields, and ``reference_venue_name`` below
        # records which venue is actually feeding them.
        #
        # 2026-05-12 codex-#6: identifies the source feed. ``"binance"``
        # / ``"bybit"`` / future entries. ``None`` when no reference
        # WS has started, or when ``REFERENCE_EXCHANGE`` is empty.
        self.reference_venue_name: Optional[str] = None
        self.binance_best_bid: Optional[float] = None
        self.binance_best_ask: Optional[float] = None
        self.binance_mid: Optional[float] = None
        self.binance_bid_size: Optional[float] = None
        self.binance_ask_size: Optional[float] = None
        self.binance_last_message_wall_ts: Optional[datetime] = None
        self.binance_last_connect_ts: Optional[datetime] = None
        self.binance_ws_last_connect_ts: Optional[datetime] = None
        self.binance_ws_connected: bool = False
        self.binance_ws_reconnect_count: int = 0
        self.binance_basis_ewma: Optional[float] = None
        # Smoothed L1 order-book imbalance. Updated by the bot loop once
        # per quote cycle from state.market bid/ask sizes. Used by
        # ``compute_quote_decision`` as an additive alpha term on the
        # reservation when ``OB_IMBALANCE_ALPHA > 0``. ``None`` until the
        # first valid update with enough depth to trust.
        self.ob_imbalance_ewma: Optional[float] = None
        # Priority #3 v1 — flow-direction / toxicity score from public
        # trade prints. ``recent_trades`` is the raw deque of ``TradePrint``
        # records from ``v1.trade``; ``flow_score`` is the accumulator
        # over those trades that computes TFI + streak scores per side.
        # See ``app/flow_score.py`` for the score derivation. Both are
        # bounded — maxlen honours ``FLOW_SCORE_RECENT_TRADES_MAXLEN`` so
        # memory stays flat.
        from app.flow_score import FlowScoreAccumulator
        from app.models import TradePrint
        self.recent_trades: deque[TradePrint] = deque(
            maxlen=int(settings.flow_score_recent_trades_maxlen)
        )
        self.flow_score: FlowScoreAccumulator = FlowScoreAccumulator(
            tfi_window_seconds=float(settings.flow_score_tfi_window_seconds),
            streak_window_prints=int(settings.flow_score_streak_window_prints),
            recent_trades_maxlen=int(settings.flow_score_recent_trades_maxlen),
        )
        # Priority #2 v2 — online IC-based regime classifier for the
        # cross-venue basis-deviation alpha. Maintains a rolling buffer
        # of (dev_bps, realized_return_bps) pairs and returns
        # ``-1.0 / 0.0 / +1.0`` based on Pearson IC. See
        # ``app/basis_regime.py`` for the design rationale.
        from app.basis_regime import BasisRegimeClassifier
        self.basis_regime: BasisRegimeClassifier = BasisRegimeClassifier(
            horizon_seconds=float(settings.basis_deviation_regime_horizon_seconds),
            window_samples=int(settings.basis_deviation_regime_window_samples),
            ic_threshold=float(settings.basis_deviation_regime_ic_threshold),
            min_pair_samples=int(settings.basis_deviation_regime_min_pair_samples),
        )
        # v1.5.209 Phase 8D — Order Flow Imbalance (OFI) accumulator.
        # Updated by the public-WS handler on every BBO update; read
        # by the quote loop once per tick. Two EWMAs (5s short-horizon
        # feeds the reservation shift; 30s anchor is published to
        # live_stats). See ``app/ofi.py``.
        from app.ofi import OFIAccumulator
        self.ofi: OFIAccumulator = OFIAccumulator(
            halflife_5s_seconds=float(settings.ofi_halflife_5s_seconds),
            halflife_30s_seconds=float(settings.ofi_halflife_30s_seconds),
            normalisation_scale=float(settings.ofi_normalisation_scale),
        )
        # Last-tick OFI shift telemetry — published in live_stats for
        # offline calibration. Single scalar (the same shift applied
        # to bid and ask reservations; OFI is symmetric in that sense).
        self.ofi_last_shift_bps: float = 0.0
        # v1.5.209 Phase 8B — Queue-position arrival-rate EWMA.
        # Updated by the public-WS trade-print handler; read by the
        # quote loop's per-tick caller to compute expected_wait.
        # See ``app/queue_model.py``.
        from app.queue_model import ArrivalRateEwma
        self.queue_arrival_rate: ArrivalRateEwma = ArrivalRateEwma(
            halflife_seconds=float(settings.queue_arrival_rate_halflife_seconds),
        )
        # Last-tick queue-aware sizing telemetry (per side). Surfaces
        # in live_stats so the operator can see when the multiplier
        # is biting. Both default 1.0 when the feature is disabled.
        self.queue_size_mult_bid: float = 1.0
        self.queue_size_mult_ask: float = 1.0
        self.queue_position_ratio_bid: Optional[float] = None
        self.queue_position_ratio_ask: Optional[float] = None
        self.queue_inside_post_active_bid: bool = False
        self.queue_inside_post_active_ask: bool = False
        # Cumulative counters — bumps every tick the feature is active.
        self.queue_size_mult_armed_bid_total: int = 0
        self.queue_size_mult_armed_ask_total: int = 0
        self.queue_inside_post_armed_bid_total: int = 0
        self.queue_inside_post_armed_ask_total: int = 0
        # Post-swing PnL cooldown gate state (analysis-day 2026-05-10).
        # See ``app/post_swing_gate.py`` for the trigger algorithm and
        # the snapshot 260510064549 incident that motivated it.
        from app.post_swing_gate import PostSwingState
        self.post_swing: PostSwingState = PostSwingState()
        # 30s-MAE gate state (1.3.82, snapshot 260515-095056). Parallel
        # to the toxicity-engine hard trigger but at the 30s post-fill
        # horizon, where bleed from directional-trend markets becomes
        # visible. Producer is the post-fill-excursion watcher daemon;
        # consumer is the bot's quote-eligibility path.
        from app.mae_gate import MaeGateState
        self.mae_gate: MaeGateState = MaeGateState()
        # Tiered session-PnL drawdown ladder (1.2.1, snapshot
        # 260510081612 follow-up). State machine catches chronic
        # bleed that single-event gates miss; see
        # ``app/session_drawdown_gate.py`` for the algorithm.
        from app.session_drawdown_gate import SessionDrawdownState
        self.session_drawdown: SessionDrawdownState = SessionDrawdownState()
        # v1.5.202 — SF-event-count fatigue ladder. Orthogonal to
        # session_drawdown: counts SF episode entries in a rolling
        # window, so storm-cluster SF patterns trigger a brake even
        # when no single episode hit the PnL drawdown tier. See
        # ``app/sf_fatigue_gate.py``.
        from app.sf_fatigue_gate import SfFatigueGateState
        self.sf_fatigue: SfFatigueGateState = SfFatigueGateState()
        # v1.5.205 Phase 4C.4 — rolling window of decision-time quote
        # ages (seconds). Sampled per-tick from
        # ``WorkingOrder.ts_acked`` of each side's resting quote (the
        # larger of bid/ask age). Used by the stale-risk penalty
        # evaluator to compute a rolling P50 against which the
        # current age is compared. 200 samples at ~2 ticks/sec = 100s
        # of history — small enough to track regime shifts (a faster-
        # filling regime brings the median down; slower brings it up)
        # while smoothing out the per-tick chatter of the place/
        # amend/cancel cycle.
        self.quote_age_decision_samples_seconds: deque[float] = deque(
            maxlen=200,
        )
        # Vol × trend conjunction gate state (1.2.2). Pre-emptive
        # gate against directional-burst regimes; see
        # ``app/vol_trend_gate.py``. basis_regime_gate and
        # microprice_gate are stateless and don't need a state field.
        from app.vol_trend_gate import VolTrendState
        self.vol_trend_gate: VolTrendState = VolTrendState()
        # v1.4.107 Phase 1B — shock_gate state. Binary acute-spike
        # defence; lock persists until util AND drift normalise (or
        # the max-cooldown safety ceiling). See ``app/shock_gate.py``.
        from app.shock_gate import ShockGateState
        self.shock_gate: ShockGateState = ShockGateState()
        # v1.4.112 Phase 1C — regime_controller FSM state. NORMAL /
        # DEFENSIVE / SHOCK mode label + knob overlays. See
        # ``app/regime_controller.py``.
        from app.regime_controller import (
            RegimeControllerState,
            RegimeKnobs,
        )
        self.regime_controller: RegimeControllerState = (
            RegimeControllerState()
        )
        # Cached knobs from the most recent tick. Consumers
        # (``compute_quote_decision``) read this; mutator is the
        # bot tick after ``evaluate_mode``.
        self.regime_knobs: RegimeKnobs = RegimeKnobs()
        # Phase 4G.5 (v1.4.211) — forward-classifier history buffers.
        # The classifier needs timestamped histories of vol_bps,
        # drift_30s_bps, and ob_imbalance to compute its leading
        # indicators (vol slope, drift magnitude rising ratio, OB
        # imbalance widening). The bot tick path appends current
        # values + prunes entries older than
        # ``settings.regime_forward_history_buffer_seconds`` (default
        # 180 s — covers the longest classifier lookback of 60 s plus
        # the 60 s CALM history-span requirement, with margin).
        #
        # Each tuple is ``(ts_mono, value)``. Stored as plain lists
        # (not deques) because the truncation step rebuilds the list
        # by mono cutoff — index access isn't on the hot path.
        #
        # Latest forward classification + diagnostics, populated by
        # the per-tick wire-up in ``app/bot.py``. ``None`` when the
        # forward layer is disabled (the default) — snapshot_dict
        # surfaces a stable placeholder block in that case.
        self.forward_vol_bps_history: list[tuple[float, float]] = []
        # v1.5.239 — parallel history for the trend-aware EWMA vol
        # measure (app/vol_abs_ewma.py). Appended in lock-step with
        # forward_vol_bps_history above. The classifier's vol_slope
        # criterion reads THIS deque instead of the stdev one when
        # REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE=true. Both
        # deques are maintained so the operator can compare side-by-
        # side from the snapshot regardless of which flag state is
        # active.
        self.forward_vol_abs_ewma_bps_history: list[tuple[float, float]] = []
        self.forward_drift_30s_history: list[tuple[float, float]] = []
        self.forward_ob_imbalance_history: list[tuple[float, float]] = []
        # v1.5.18 Phase 4G.7 -- rolling 30-min basis history for the
        # classifier's basis_stretch CAUTIOUS trigger. Holds
        # ``(now_mono, basis_bps)`` pairs; pruned to the
        # ``regime_forward_basis_median_lookback_seconds`` window each
        # tick. The classifier's
        # ``binance_basis_30min_median_bps`` parameter is computed
        # from this buffer via ``median_forward_basis_bps``. Until 4G.7
        # the 4th leading-indicator trigger was dormant
        # (``_fwd_basis_median_bps=None`` in bot.py) -- the other
        # three CAUTIOUS triggers carried the load.
        self.forward_basis_bps_history: list[tuple[float, float]] = []
        # Latest ``ForwardSignalReading`` (or None when disabled).
        # Typed as Any to avoid import-cycle at state-module load time;
        # the dataclass lives in ``app/regime_forward_signals.py``.
        self.last_forward_signal_reading: Optional[Any] = None
        # v1.4.113 Phase 1D — post-reduction re-entry cooldown state.
        # Stamped on every fill that reduces |position_qty|; consumed
        # by `_apply_regime_gates` to suppress the side that would
        # re-add to the post-fill direction. Active ONLY under
        # regime_controller modes != NORMAL — NORMAL keeps round-trip
        # rebate capture intact.
        # ``last_inventory_reduction_suppressed_side`` is one of
        # ``QuoteEligibility.QUOTE_SELL_ONLY`` / ``QUOTE_BUY_ONLY``
        # (the eligibility cap to apply during the cooldown window),
        # or None when no cooldown is armed.
        self.last_inventory_reduction_at_mono: Optional[float] = None
        from app.enums import QuoteEligibility as _QuoteEligibility
        self.last_inventory_reduction_suppressed_side: Optional[
            _QuoteEligibility
        ] = None
        # Tracks whether the cooldown was active on the prior tick so
        # the WARNING log fires exactly once per cooldown arming, not
        # every tick of the cooldown window.
        self.post_reduction_cooldown_was_active_last_tick: bool = False
        # Session-cumulative arming counter — surfaced via
        # ``_behavioural_gates_snapshot`` for postmortem.
        self.post_reduction_cooldown_fire_count: int = 0
        # v1.4.144 Phase 2K.1 — exit-attribution counters. Increment
        # when the cooldown CLEARS — either via the favorable-exit
        # predicate (position util dropped below the configured
        # threshold) or via the MAX-cooldown ceiling firing as a
        # safety net. Ratio of the two tells the operator whether
        # the favorable-exit predicate is doing meaningful work or
        # whether the ceiling is still binding. Surfaced via
        # ``behavioural_gates.post_reduction_cooldown`` + live_stats.
        self.post_reduction_cooldown_cleared_via_favorable_total: int = 0
        self.post_reduction_cooldown_cleared_via_ceiling_total: int = 0
        # v1.4.116 Phase 1E.3.f — per-quoting-feature firing-rate
        # tracker. Records edge transitions (False → True) on
        # ``observability_gate_flags`` and exposes 60 s rolling
        # counts for the Detectors card's section 5. Read by
        # ``live_stats._feature_firing_rates_block()``; mutated by
        # the bot tick after the flags are refreshed.
        from app.feature_firing_rate import FeatureFiringRateTracker
        self.feature_firing_rates: FeatureFiringRateTracker = (
            FeatureFiringRateTracker()
        )
        # Stashed snapshot of ``observability_gate_flags`` from the
        # prior tick — used by ``detect_edge_transitions`` to find
        # False → True flips. Empty dict on first tick.
        self._prior_observability_gate_flags: dict[str, Optional[bool]] = {}
        # 2026-05-12 codex-#1 narrow + codex-#3. Two new gates, both
        # disabled-by-default via their config knobs (threshold==0).
        # at_touch_adverse_pause arms a per-side cooldown when recent
        # at_touch fills cluster adverse on that side; fill_burst_
        # detector applies a size-shrink when fills cluster in time.
        # Both consume timestamps from ``time.monotonic`` so they're
        # robust to wall-clock jumps.
        from app.at_touch_adverse_pause import AtTouchAdversePause
        self.at_touch_adverse_pause = AtTouchAdversePause(
            threshold_bps=float(settings.at_touch_adverse_pause_threshold_bps),
            pause_seconds=float(settings.at_touch_adverse_pause_seconds),
            min_fills=int(settings.at_touch_adverse_pause_min_fills),
            # Phase 2K.6 favorable-exit knobs. Defensive ``getattr`` for
            # Settings shapes from before v1.4.156.
            favorable_exit_enabled=bool(
                getattr(
                    settings,
                    "at_touch_adverse_pause_favorable_exit_enabled",
                    True,
                )
            ),
            clear_band_mult=float(
                getattr(
                    settings,
                    "at_touch_adverse_pause_clear_band_mult",
                    0.5,
                )
            ),
            favorable_exit_dwell_seconds=float(
                getattr(
                    settings,
                    "at_touch_adverse_pause_favorable_exit_dwell_seconds",
                    5.0,
                )
            ),
        )
        # Phase 4C.3 mini (v1.4.161) — per-side realised-edge side
        # suppression. Out-of-order delivery of one Phase 4C piece;
        # see ``app/realised_edge_side_suppress.py`` docstring.
        from app.realised_edge_side_suppress import (
            RealisedEdgeSideSuppressGate,
        )
        self.realised_edge_side_suppress = RealisedEdgeSideSuppressGate(
            threshold_bps=float(
                getattr(
                    settings,
                    "realised_edge_suppress_threshold_bps",
                    0.0,
                )
            ),
            cooldown_seconds=float(
                getattr(
                    settings,
                    "realised_edge_suppress_cooldown_seconds",
                    60.0,
                )
            ),
            min_fills=int(
                getattr(
                    settings,
                    "realised_edge_suppress_min_fills",
                    4,
                )
            ),
            favorable_exit_enabled=bool(
                getattr(
                    settings,
                    "realised_edge_suppress_favorable_exit_enabled",
                    True,
                )
            ),
            clear_band_mult=float(
                getattr(
                    settings,
                    "realised_edge_suppress_clear_band_mult",
                    0.5,
                )
            ),
            favorable_exit_dwell_seconds=float(
                getattr(
                    settings,
                    "realised_edge_suppress_favorable_exit_dwell_seconds",
                    10.0,
                )
            ),
        )
        # Phase 4C.1/4C.2 (v1.4.164) — per-side expected-edge side
        # suppression state. Bot updates these once per tick via the
        # pure helper in ``app/expected_edge.py``.
        self.expected_edge_refused_bid: bool = False
        self.expected_edge_refused_ask: bool = False
        self.expected_edge_recovery_ticks_bid: int = 0
        self.expected_edge_recovery_ticks_ask: int = 0
        self.expected_edge_last_bid_bps: Optional[float] = None
        self.expected_edge_last_ask_bps: Optional[float] = None
        # Session-cumulative counters for the operator-tuning ratio.
        self.expected_edge_armed_bid_total: int = 0
        self.expected_edge_armed_ask_total: int = 0
        self.expected_edge_cleared_bid_total: int = 0
        self.expected_edge_cleared_ask_total: int = 0
        # v1.5.146 Phase 4C.2.a — dampen band counters + last-tick
        # widening bps for the per-side audit trail. The counter
        # bumps every tick the band fires; the bps value reflects
        # the most recent fire (zero when not armed).
        self.negative_expectancy_dampen_bid_total: int = 0
        self.negative_expectancy_dampen_ask_total: int = 0
        self.negative_expectancy_dampen_last_bid_bps: float = 0.0
        self.negative_expectancy_dampen_last_ask_bps: float = 0.0
        # v1.5.207 Phase 4C.5 — participation score per side, a single
        # continuous summary of the bot's willingness to quote that
        # side ([0,1]: 1=full quote, 0=fully refused). Computed each
        # tick from the side's expected_net_edge via the pure helper
        # in ``app/participation_score.py``. The rolling deques feed
        # the dashboard "mean participation score" tile + the
        # acceptance check that correlates score with realised
        # markout (Phase 4C.5.d).
        self.participation_score_bid: Optional[float] = None
        self.participation_score_ask: Optional[float] = None
        # 200 samples at 2 ticks/sec ≈ 100s of history. Same cadence
        # as the v1.5.205 quote_age_decision_samples deque.
        self.participation_score_bid_recent: deque[float] = deque(maxlen=200)
        self.participation_score_ask_recent: deque[float] = deque(maxlen=200)
        # Per-tick mismatch counter: bumps when the score's action
        # disagrees with the existing refuse/dampen gates. v1.5.207
        # ships as observability-only; this counter helps the operator
        # see whether the score is calibrated correctly before any
        # behavior flip.
        self.participation_score_disagreement_bid_total: int = 0
        self.participation_score_disagreement_ask_total: int = 0
        # Phase 4D (v1.4.164) — SF phase ladder state. Carries across
        # ticks inside a single SF episode; reset on
        # ``_enter_soft_flatten`` / ``_exit_soft_flatten``.
        self.sf_phase_ladder_phase: int = 0
        self.sf_phase_started_mono: float = 0.0
        self.sf_consecutive_rejects_in_phase: int = 0
        self.sf_entry_mid_for_phase_ladder: Optional[float] = None
        # v1.5.146 Phase 4D.3 — slice-cap session counter. Bumps each
        # phase-2/3 IOC dispatch where the slice cap actually trimmed
        # the IOC's size below the would-be all-at-once value. Lets
        # the operator see at a glance whether slicing was active in
        # any SF episode this session.
        self.sf_slice_dispatched_total: int = 0
        # Phase 4D.5 (v1.4.190) — SF action-rate throttle state.
        # ``sf_last_action_mono`` is the monotonic timestamp of the
        # most recent throttled SF action (cancel-replace or place).
        # ``sf_throttle_suppressed_total`` is a session counter for
        # the operator-visible suppression rate (surfaced in the
        # executor-state snapshot). ``sf_throttle_first_arm_logged``
        # gates the WARNING that fires once on first suppression to
        # avoid log spam — subsequent suppressions are silent
        # (counter still bumps).
        self.sf_last_action_mono: Optional[float] = None
        self.sf_throttle_suppressed_total: int = 0
        self.sf_throttle_first_arm_logged: bool = False
        # v1.4.192 — SF event id grace cache. The v1.4.163 fill-tag
        # fallback was structurally racy for the taker paths
        # (``client.market_close`` / ``place_ioc``): ``_exit_soft_flatten``
        # clears ``soft_flatten_event_id`` synchronously right after
        # the taker fires, but the corresponding fill arrives via
        # private WS tens-to-hundreds of ms LATER. At fill-ingest
        # time the "is SF active?" check sees None and the fill
        # ends up untagged.
        #
        # Observed twice: v1.4.157 (2026-05-20 17:14, 3 untagged
        # taker fills) and v1.4.189 (2026-05-21 10:29, 1 untagged
        # taker fill). All maker-only SF episodes tagged correctly.
        #
        # ``sf_recent_event_id`` is set by ``_exit_soft_flatten`` to
        # the event id that just exited; ``sf_recent_event_id_valid_until_mono``
        # is now+grace. The fill-ingest fallback consults this AFTER
        # the active-SF check fails. Default grace is 30 s
        # (``SF_RECENT_EVENT_ID_GRACE_SECONDS``) — well beyond any
        # observed private-WS fill latency, well below any reasonable
        # SF-re-entry cadence.
        self.sf_recent_event_id: Optional[int] = None
        self.sf_recent_event_id_valid_until_mono: float = 0.0
        # Session counter for operator-facing observability: how many
        # fills got tagged via the recent-event grace path (vs the
        # active-SF path). A non-zero count proves the grace fallback
        # is doing real work; the WARNING also fires on first use.
        self.sf_recent_event_tag_total: int = 0
        # Phase 2G (v1.4.204) — SHOCK-mode Telegram alert bookkeeping.
        # The mode FSM publishes regime_mode_transition events
        # unconditionally; this batch additionally forwards SHOCK-
        # related ones to the Telegram ops channel. Memory note
        # ``feedback_postmortem_preferred_over_runtime_alerts``: only
        # the rare + operator-actionable transitions get a real-time
        # alert. SHOCK qualifies (it locks one side, persists until
        # shock_gate clears, and may indicate the v1.4.189-style
        # phase-ladder catastrophe).
        #
        # Both flags are CLEARED on SHOCK exit so the next entry
        # gets a fresh alert. Set on first fire so re-evaluations
        # mid-episode don't re-spam.
        self.shock_entry_telegram_alerted_for_episode: bool = False
        self.shock_persistence_telegram_alerted_for_episode: bool = False
        # Session counters — operator-visible in the snapshot's
        # ``state_current.json`` so they can verify the alerts are
        # firing in production. Bumped once per fire (not once per
        # tick during a SHOCK episode).
        self.shock_telegram_entry_alerts_sent_total: int = 0
        self.shock_telegram_persistence_alerts_sent_total: int = 0
        # v1.5.26 Phase 2G.3 -- daily time-in-mode Telegram summary.
        # Stores the mono timestamp of the LAST daily summary so the
        # 24h cadence check is a simple subtract. ``0.0`` means
        # "never sent in this process" -- the first summary fires
        # ~24h after process start (gated by the configurable
        # interval), not immediately. Counter tracks per-session
        # firing count so the operator can verify in snapshots.
        self.regime_daily_summary_last_sent_mono: float = 0.0
        self.regime_daily_summary_sent_total: int = 0
        # v1.5.26 Phase 2D -- residual-decay trigger for adaptive_widen.
        # The tracker observes resolved-5s fills via
        # ``app/markout.py`` (same hook point as the at_touch /
        # realised_edge gates). When the rolling mean of
        # ``(closed_pnl + rebate - markout_5s) / notional`` drops
        # below threshold_bps for dwell_seconds, the adaptive_widen
        # arming block in bot.py sees ``is_armed() == True`` and
        # fires the gate with ``reason=residual_decay``.
        from app.residual_decay_tracker import ResidualDecayTracker as _RDT
        self.residual_decay_tracker = _RDT(
            threshold_bps=float(settings.residual_decay_widen_threshold_bps),
            dwell_seconds=float(settings.residual_decay_widen_dwell_seconds),
            window_fills=int(settings.residual_decay_widen_window_fills),
            min_fills=int(settings.residual_decay_widen_min_fills),
        )
        # Phase 4C.3 (v1.5.41) — side-specific historical edge as
        # quote-decision confidence factor. Long-window (default 1 h)
        # per-side trailing mean+stdev of ``markout_5s_bps + rebate_bps``.
        # The z-score of that mean feeds back into the expected-edge
        # consumer as a multiplicative factor (``expected_edge ×
        # max(0, 1 + α × z)``), which lets the refusal gate
        # (4C.1+4C.2) trigger sooner when recent realised edge has
        # degraded. Fed from ``app/markout.py``'s 5 s resolution path
        # alongside the existing ``realised_edge_side_suppress`` gate
        # (which is the short-window binary cooldown for acute
        # degradation; this is the long-window continuous one for
        # slow drift). See ``app/side_edge_history.py``.
        from app.side_edge_history import SideEdgeHistory as _SEH
        self.side_edge_history = _SEH(
            window_seconds=float(
                getattr(settings, "side_edge_history_window_seconds", 3600.0)
            ),
            max_samples=int(
                getattr(settings, "side_edge_history_max_samples", 8192)
            ),
            min_samples=int(
                getattr(settings, "side_edge_history_min_samples", 10)
            ),
            zscore_coeff=float(
                getattr(settings, "side_edge_history_zscore_coeff", 0.5)
            ),
            mult_floor=float(
                getattr(settings, "side_edge_history_mult_floor", 0.0)
            ),
            mult_ceil=float(
                getattr(settings, "side_edge_history_mult_ceil", 2.0)
            ),
            enabled=bool(
                getattr(settings, "side_edge_history_enabled", False)
            ),
        )
        # Phase 4E (v1.4.165) — target-venue fast-move cancel counters.
        # Incremented whenever ``detect_target_venue_fast_move``
        # returns a side AND the cancel was dispatched. Operator
        # reads these in the next snapshot to verify the gate is
        # firing on real moves, not spuriously.
        self.target_venue_fast_move_cancel_bid_total: int = 0
        self.target_venue_fast_move_cancel_ask_total: int = 0
        # Phase 4E bridge — populated by bot.py at the top of each
        # quote-tick from the just-computed ``raw_q.mid_return_500ms_bps``.
        # ``OrderManager`` reads this from the trigger path. ``None``
        # until the first tick with valid kinematics.
        self.last_mid_return_500ms_bps: Optional[float] = None
        # Phase 2I (v1.4.169) — per-order amend rate-defence counters.
        # ``tick_flicker`` increments when an amend dispatch was
        # rejected because the previous dispatch on the SAME order
        # was less than ``AMEND_TICK_FLICKER_MIN_MS`` ago. ``rate_throttle``
        # increments when the per-order 1-second cap
        # ``AMEND_PER_ORDER_MAX_PER_SEC`` was exceeded. Surfaced via
        # ``executor_state_snapshot`` so the operator sees the rate.
        # Also a per-side first-arm tracker so the WARNING log fires
        # ONCE per side per session, not on every suppression.
        self.amend_tick_flicker_suppressed_total: int = 0
        self.amend_rate_throttle_suppressed_total: int = 0
        self.amend_rate_defence_first_arm_logged: dict[Side, bool] = {
            Side.BUY: False,
            Side.SELL: False,
        }
        # Phase 4F (v1.4.170) — elevated-vol auto-pause state. When
        # ``vol_auto_pause_active`` is True the bot's eligibility is
        # forced to HOLD_ALL with reason ``vol_regime_auto_paused``.
        # See ``app/vol_regime_auto_pause.py`` for the pure-function
        # evaluator that the bot's tick calls each cycle. Default
        # OFF via ``VOL_AUTO_PAUSE_ARM_RATIO=0.0``.
        self.vol_auto_pause_active: bool = False
        self.vol_auto_pause_arm_dwell_started_mono: Optional[float] = None
        self.vol_auto_pause_clear_dwell_started_mono: Optional[float] = None
        self.vol_auto_pause_active_since_mono: float = 0.0
        self.vol_auto_pause_armed_total: int = 0
        self.vol_auto_pause_cleared_via_favorable_total: int = 0
        self.vol_auto_pause_cleared_via_ceiling_total: int = 0
        # First-arm WARNING / first-clear INFO log de-dupe (once per
        # session per state-change, not per tick).
        self.vol_auto_pause_first_arm_logged: bool = False
        from app.fill_burst_detector import FillBurstDetector
        self.fill_burst_detector = FillBurstDetector(
            threshold=int(settings.fill_burst_threshold),
            window_seconds=float(settings.fill_burst_window_seconds),
            size_mult=float(settings.fill_burst_size_mult),
            cooldown_seconds=float(settings.fill_burst_cooldown_seconds),
            # Phase 2K.9 favorable-exit knobs. Defensive ``getattr``
            # for Settings shapes from before v1.4.160.
            favorable_exit_enabled=bool(
                getattr(
                    settings,
                    "fill_burst_favorable_exit_enabled",
                    True,
                )
            ),
            clear_band_mult=float(
                getattr(
                    settings,
                    "fill_burst_clear_band_mult",
                    0.5,
                )
            ),
            favorable_exit_dwell_seconds=float(
                getattr(
                    settings,
                    "fill_burst_favorable_exit_dwell_seconds",
                    5.0,
                )
            ),
        )
        # 1.2.8: basis_regime gate's per-tick size-shrink output
        # (when ``BASIS_REGIME_GATE_MODE=size_shrink``). Set to 1.0
        # when the gate isn't firing; drops to
        # ``BASIS_REGIME_GATE_SIZE_MULT_SIGNAL_ABSENT`` (default 0.5)
        # when |IC| < threshold. Read by the quote-decision build
        # path in ``bot.py``; included in the behavioural-gates
        # snapshot for postmortem visibility.
        self.basis_regime_size_mult: float = 1.0
        self.basis_regime_size_mult_reason: Optional[str] = None
        # 1.2.14: per-cycle multi-level ladder decision. Built by
        # ``build_ladder`` after every quote cycle and stamped here
        # so observers (snapshot, telegram /status, postmortem) can
        # see "what would the ladder have done" without recomputing
        # it. ``None`` until the first cycle has run. At
        # ``LADDER_NUM_LEVELS_PER_SIDE=1`` the ladder is a single
        # rung that exactly matches the inside scalar quote.
        self.last_ladder_decision: Any = None
        # 1.2.34: per-cycle quote breakdown for the dashboard's
        # Spread tab. Captures every input that contributed to
        # this cycle's quote (mid → reservation shifts, half-spread
        # stack, size mult chain). Built by ``compute_quote_decision``
        # and stamped here by the bot's quote loop. ``None`` until
        # the first cycle. Serialised into ``state_current.json`` by
        # ``snapshot_dict``.
        self.last_quote_breakdown: Any = None
        # Market book update instrumentation (public WS BBO; not order lifecycle).
        self.market_data_last_success_wall_ts: Optional[datetime] = None
        self.market_data_failed_refresh_streak: int = 0
        self.market_data_unchanged_snapshot_streak: int = 0
        self.market_data_last_refresh_latency_ms: Optional[float] = None
        self._market_data_snapshot_fingerprint: Optional[tuple[Any, ...]] = None
        self.market_data_stall_latched: bool = False
        self.market_data_transport_reset_count: int = 0
        self._market_data_last_success_log_monotonic: float = 0.0
        self.flatten_incomplete: bool = False
        self.flatten_residual_abs_qty: float = 0.0
        self.last_quote_cycle_id: Optional[str] = None
        self.last_active_sides: Optional[str] = None
        # Watchdog signals (see app/watchdog.py). Updated by the quote
        # loop and by the execution layer respectively. The watchdog
        # treats a large gap between these two timestamps (quote engine
        # saying "quote both sides" but execution not dispatching) as a
        # deadlock and forces the process to exit so the container
        # manager brings it back with clean state.
        # ``last_quote_engine_non_hold_ts_mono`` — set when the quote
        #   engine emits a decision with active_sides != NONE (i.e.
        #   engine thinks we SHOULD be placing).
        # ``last_place_attempt_ts_mono`` — set right before calling
        #   ``client.place_post_only_limit()``, regardless of outcome.
        #   Covers the case where execution is trying AND the server
        #   rejects; only leaves ``last_place_attempt_ts_mono`` stale
        #   when the execution layer is NOT calling into the client at
        #   all (which is the stuck-forever class we need to detect).
        #
        # ``last_outbound_attempt_ts_mono`` — v1.4.42 BUG-025-adjacent:
        #   set right before EITHER a place OR an amend dispatch. The
        #   watchdog + silent-wedge detector both use this (not the
        #   place-only field above) because the v1.4.16 amend-on-reprice
        #   path can keep a quote alive via continuous amends without
        #   ever dispatching a fresh place — which made the place-only
        #   counter go stale and produced false-positive watchdog kills
        #   every ~10 minutes of amend-heavy quoting (observed 2026-05-18,
        #   snapshot v1.4.41-260518-103610 line 232: idle=605.8s while
        #   amends were firing every 670 ms). Cancels are deliberately
        #   excluded — a cancels-only loop is the BUG-025 wedge pattern
        #   and SHOULD still trip the watchdog.
        self.last_quote_engine_non_hold_ts_mono: float = 0.0
        self.last_place_attempt_ts_mono: float = 0.0
        self.last_outbound_attempt_ts_mono: float = 0.0
        # Counter incremented every time the execution layer calls
        # ``client.place_post_only_limit`` (success OR failure — same
        # semantics as ``last_place_attempt_ts_mono``). Surfaces on
        # the Telegram ``/status`` reply as the liveliness heartbeat
        # so the operator can answer "is the bot actively placing?"
        # without having to read the bot's logs. Distinct from the
        # ``outbound_*_action_send_count`` family which also counts
        # cancels — this counter is placements only.
        self.session_place_attempt_count: int = 0
        # v1.4.28: session-cumulative cancel-intent counter, parallel
        # to ``session_place_attempt_count``. Incremented once per
        # successful enqueue in ``_enqueue_cancel_quote_path`` (the
        # hot quote-path) and in the synchronous ``cancel_order`` /
        # ``_cancel_orphan_remote_order`` / ``cancel_all_orders_for_symbol``
        # callers. The dashboard needs a per-INTENT cancel count so
        # the operator can distinguish "lots of reprice cycles (high
        # cancel count)" from "few but big batches (low cancel-call
        # count, large per-batch payload)" — derivation from
        # ``session_action_count - session_place_attempt_count -
        # amend_intents_emitted_total`` doesn't work because
        # session_action_count is per-CALL (one per batch) while
        # the others are per-INTENT.
        self.session_cancel_attempt_count: int = 0
        # 1.4.6: STICKY rejection summaries. These accumulate over the
        # WHOLE session and NEVER decay — bug fix for the dashboard
        # surfacing only the most-recent 5000 orders_lifecycle rows
        # (the operator observation 2026-05-17: rejection counts
        # would appear, then disappear once the ring filled with
        # newer accepted rows).
        #
        # The per-detail map is bounded by the cardinality of distinct
        # venue rejection messages — small in practice (~10-50 unique
        # strings even on a long session) because OKX sCode messages
        # bucket naturally.
        #
        # ``count`` is monotonic; ``first_seen_ts`` / ``last_seen_ts``
        # are UTC ISO strings; ``is_benign`` is the same classification
        # the existing connectivity surface uses (post_only_would_cross
        # etc. are benign).
        self.session_place_reject_summary: dict[
            str, dict[str, object]
        ] = {}
        self.session_cancel_reject_summary: dict[
            str, dict[str, object]
        ] = {}
        self.session_place_reject_count_total: int = 0
        self.session_cancel_reject_count_total: int = 0
        # 1.4.6: cumulative per-outcome counters + latency moments.
        # Mirror surface to the sticky reject summary but covers
        # ACCEPTED outcomes too, so the dashboard's connectivity
        # header ("N placements · X% ack rate") can read full-session
        # totals instead of the 5000-row recent window. NEVER decays.
        # Latency stats are running moments (count/min/max/sum) —
        # p50/p95 still computed from the recent-row window where
        # "current performance" is the question, not "lifetime."
        #
        # Map shape: outcome -> {count, min_ms, max_ms, sum_ms}.
        self.session_place_outcome_counts: dict[
            str, dict[str, float]
        ] = {}
        self.session_cancel_outcome_counts: dict[
            str, dict[str, float]
        ] = {}
        # 1.4.7 gate-widening Phase 0: per-gate fire telemetry.
        # ``record_gate_firing(name, firing_now=..., now_mono=...)``
        # is called once per quote-cycle tick PER gate; it tracks
        # rising / falling edges of the gate's active state and
        # accumulates total fire-seconds.
        #
        # Format per gate:
        #   {
        #     "fire_count": int,            # rising-edge count
        #     "fire_seconds_total": float,  # cumulative active time
        #     "active_now": bool,           # last-known state
        #     "active_since_mono": float | None,  # mono ts of rising edge
        #   }
        #
        # Used by the gate-widening baseline (plans/gate-to-widening.md
        # Phase 0) to quantify per-gate dark-time cost before any
        # behaviour change.
        self.session_gate_fire_stats: dict[str, dict[str, Any]] = {}
        # 1.4.9 gate-to-widening Phase 1: the latest SpreadComposition
        # built per quote-cycle tick. The heartbeat publisher reads
        # this to surface per-contributor bps to the operator
        # dashboard. ``None`` until the first tick completes. The
        # type is :class:`app.quoting.SpreadComposition` but it's
        # stored as ``Any`` here to avoid a circular import (state
        # imports nothing from quoting).
        self.last_spread_composition: Any = None
        self._seen_fill_ids: set[str] = set()
        # Fill observers: callables invoked AFTER ``record_fill`` releases
        # the state lock for a session-scoped (i.e. new, post-startup) fill.
        # Used by the Telegram notifier to broadcast fills without coupling
        # state.py to the notifier module. Observers are best-effort: any
        # exception is swallowed and logged; a slow observer cannot stall
        # the trading loop because the registration site (Telegram) does
        # async fire-and-forget enqueue, not network IO.
        self._fill_observers: list[Callable[["Fill"], None]] = []
        self.session_id = str(uuid.uuid4())
        self.market_data_gap_tracker = MarketDataGapTracker(
            int(settings.market_data_gap_ring_buffer_samples)
        )
        self.public_ws_timing: Optional[PublicWsTimingTracker] = (
            PublicWsTimingTracker(
                symbol=settings.symbol,
                max_samples=int(settings.market_data_timing_max_samples),
            )
            if bool(settings.market_data_timing_window_enabled)
            else None
        )
        # Parallel timing tracker for the Binance cross-venue reference
        # stream. Instantiated under the same ``market_data_timing_window_enabled``
        # flag as the GRVT tracker; fed by ``BinancePublicStream`` on every
        # received message (see ``app/exchange/binance_public_ws.py``).
        # ``symbol`` carries the Binance-side instrument name so the summary
        # distinguishes "ETHUSDT on Binance" from "ETH on GRVT" when both
        # are dumped in the same snapshot folder.
        #
        # Why a second instance rather than a dict keyed by venue: the
        # state object follows a flat-field pattern, and adding a parallel
        # tracker is a strictly additive change with no risk to the
        # existing GRVT path. A future third venue (e.g. OKX) follows the
        # same pattern — a new field next to this one.
        self.binance_public_ws_timing: Optional[PublicWsTimingTracker] = (
            PublicWsTimingTracker(
                symbol=settings.binance_symbol,
                max_samples=int(settings.market_data_timing_max_samples),
            )
            if bool(settings.market_data_timing_window_enabled)
            else None
        )
        # Lazy provider for the rolling place-to-ack RTT summary. The
        # tracker lives on the executor (built later in ``main.py``);
        # the heartbeat publisher reads this provider, falls back to
        # ``None`` when the executor isn't up yet (first ~second of
        # startup) or when the bot was built without an executor (a
        # handful of unit tests). Signature: () -> dict-or-None.
        self.order_rtt_summary_provider: Optional[Callable[[], dict[str, Any]]] = None
        # 1.4.0 cancel-prio Phase 0.5: sibling provider for the
        # cancel-RTT block on the heartbeat. Same shape as
        # ``order_rtt_summary_provider``; wired from OrderManager init.
        # NULL until OrderManager exposes it (legacy / unit-test paths
        # without a full executor).
        self.cancel_rtt_summary_provider: Optional[Callable[[], dict[str, Any]]] = None
        # v1.4.58 todo-037: unified TX → ack summary across all three
        # op kinds (place + amend + cancel). Single dashboard row
        # instead of two per-op rows; ~16k samples/session instead of
        # ~1k+19. Wired from OrderManager init.
        self.tx_rtt_summary_provider: Optional[Callable[[], dict[str, Any]]] = None

        # Place-to-fill ratio tracker (rolling window). Was originally
        # wired to a kill/pause gate that was removed 2026-05-06 after
        # producing a false-positive kill on a calm-market 10-minute
        # no-fill stretch. Tracker is preserved for observability —
        # the heartbeat publishes its (places, fills, ratio) snapshot
        # so the dashboard / operator can spot churn without it
        # driving any automated decisions.
        from app.place_to_fill_ratio_tracker import PlaceToFillRatioTracker

        self.place_to_fill_ratio_tracker = PlaceToFillRatioTracker(
            window_seconds=600.0,
        )
        # Adaptive join-depth controller (off by default; opt-in via
        # ``JOIN_DEPTH_AUTOTUNE_ENABLED``). Owned by state so it
        # survives across ticks; ticked from ``Bot.one_tick``; read
        # by ``compute_quote_decision`` for the half-spread overlay.
        # See ``app/join_depth_controller.py`` + ``plans/auto-tune.md``.
        from app.join_depth_controller import JoinDepthController

        self.join_depth_controller = JoinDepthController(settings)
        # Local receipt time of last public BBO apply (monotonic seconds at ingest).
        self.public_last_bbo_receipt_monotonic: Optional[float] = None
        self._mid_price_samples: deque[tuple[float, float]] = deque(
            maxlen=max(32, int(settings.quote_mid_history_max_samples))
        )
        # Sparse multi-minute mid history for the long-window drift
        # gate (BUGS/bug-002.md). Same shape as ``_mid_price_samples``
        # but with much longer retention (default 5 min worth of
        # samples). Fed from the same ingest site; pruned by age
        # (``drift_long_window_seconds``) at append time. Empty on
        # cold start; the gate stays inactive until the window has
        # accumulated ``drift_long_window_min_samples`` entries.
        self._mid_price_samples_long: deque[tuple[float, float]] = deque(
            maxlen=max(
                64, int(getattr(settings, "drift_long_window_max_samples", 512))
            )
        )
        self.quote_eligibility_snapshot_dict: dict[str, Any] = {}
        # 2026-05-13 regime-observability bugfix: exposure-bar emitter
        # needs the per-side gate flags (post_fill_cooldown,
        # at_touch_adverse_pause) that live on OrderManager, not on
        # state. Bot pushes them here once per tick so the emitter
        # can sample them lock-free. Schema:
        #   {"post_fill_cooldown_bid": bool | None,
        #    "post_fill_cooldown_ask": bool | None,
        #    "at_touch_adverse_pause_bid": bool | None,
        #    "at_touch_adverse_pause_ask": bool | None}
        # Empty dict default → all values resolve to None in the
        # emitter (NULL columns in exposure_bars). Populated by the
        # bot's tick once the OrderManager+gate references are wired.
        self.observability_gate_flags: dict[str, Any] = {}
        self.quote_eligibility_last_effective: QuoteEligibility = QuoteEligibility.QUOTE_BOTH
        self.quote_elig_recovery_until_mono: float = 0.0
        self.quote_elig_recovery_floor: Optional[QuoteEligibility] = None
        # Observability only (updated by bot thread): remaining recovery clamp time.
        self.quote_elig_recovery_remaining_ms: float = 0.0
        self.quote_elig_last_logged: Optional[str] = None
        self.quote_elig_hold_all_count: int = 0
        self.quote_elig_buy_only_count: int = 0
        self.quote_elig_sell_only_count: int = 0
        self.quote_elig_hold_due_stale_count: int = 0
        self.quote_elig_hold_due_jump_count: int = 0
        self.quote_elig_one_sided_due_drift_count: int = 0
        self.quote_elig_one_sided_due_freshness_count: int = 0
        self.quote_elig_resume_count: int = 0
        # todo-011: per-side last-fill monotonic timestamps (ms). Used by the
        # post-fill replace cooldown to dodge the 0-100 ms predator window —
        # ``quoting.compute_quote_decision`` reads these via the caller's
        # computed ``post_fill_cooldown_{bid,ask}_remaining_ms`` kwargs and
        # suppresses new placements on the side that was just filled until
        # ``POST_FILL_REPLACE_COOLDOWN_MS`` elapses. ``0.0`` means "no fill
        # observed yet on this side" (or pre-1.2.49 bot). Updated in
        # ``record_fill`` for session-scoped fills only — replay paths
        # don't arm the cooldown.
        self.last_fill_monotonic_ms_buy: float = 0.0
        self.last_fill_monotonic_ms_sell: float = 0.0
        # Fills with exchange ts strictly before this instant are replay/history for this process;
        # they are deduped and may be persisted but do not drive session PnL / toxicity / trade-rate.
        self.session_started_at_utc: datetime = utc_now()
        # Fast-start mode (optional): during BotStatus.STARTING, skip historical fill replay
        # (REST recent fills + private WS isSnapshot userFills) so startup readiness depends only
        # on current account/position + current open-orders reconciliation.
        self.fast_start_mode_enabled: bool = bool(
            settings.fast_start_skip_historical_fill_replay
        )
        self.startup_historical_fill_replay_skipped: bool = False
        self.startup_rest_fill_replay_skipped_count: int = 0
        self.startup_private_snapshot_fills_skipped_count: int = 0
        self.startup_ready_without_fill_replay: bool = False
        # v1.4.98 — maxlen sized for the LONGEST diagnostic horizon
        # (120s) at peak fill rate. Pre-v1.4.98 sizing of 400 was a
        # 5-second × 80 fills/s budget. Going to 120s would need 9600
        # at the same rate, but observed fill rates on OKX-TON cap at
        # ~1.2/min ≈ 0.02 fills/s — so even a 120-second horizon
        # would only retain ~2.4 jobs per second of bursting. Budget
        # 8000 for a 10× safety margin against any short-window burst
        # (the snapshot 260518-090000 had 80+ fills in a 60s sliver).
        # Worst-case memory: ~7000 × ~120 bytes/job ≈ 1 MB. Trivial.
        #
        # The overflow counter (below) makes any future overrun visible.
        self.pending_markout_jobs: deque[PendingMarkoutJob] = deque(maxlen=8000)
        # v1.4.36 (Codex #4): overflow counter for ``pending_markout_jobs``.
        # ``deque.append`` on a full deque silently evicts the oldest
        # entry — pre-v1.4.36 those evictions were invisible, so a
        # bursty fill window would silently drop 1s/3s/5s/15s/30s/60s/
        # 120s markout completions for the oldest fills, corrupting
        # regime analytics + any downstream logic that depends on
        # resolved markouts.
        #
        # Codex's suggested fix proposed an unbounded keyed structure
        # to eliminate the drop entirely; that's a larger refactor and
        # deserves its own batch. For v1.4.36 we (a) make the loss
        # OBSERVABLE by counting overflow evictions at the callsite
        # (``register_fill_for_delayed_markouts``), and (b) keep the
        # bounded deque so memory remains predictable. The counter is
        # session-cumulative and surfaces in ``state_current`` so the
        # operator can see it accumulating during bursty windows; a
        # non-zero value tells the operator their fill rate occasionally
        # exceeds the 80/s sustained budget that 400 fills / 5 s
        # horizon implies, and they should consider raising ``maxlen``
        # or moving to the unbounded structure.
        self.markout_jobs_dropped_overflow_count: int = 0
        self.quote_quality = QuoteQualityRollup(
            window_samples=int(settings.quote_quality_window_samples),
        )
        self._qq_lifetime_recorded_local_ids: set[str] = set()
        # Trade-activity limiter: wall-clock timestamps of new fills (last 60s window).
        self._trade_activity_ts: deque[float] = deque(maxlen=720)
        self.last_latency_fill_to_process_ms: Optional[float] = None
        # Exchange-clock/mixed-clock age (exchange ts -> local now); not a pure local latency metric.
        self.last_exchange_ts_to_decision_ms: Optional[float] = None
        self.last_latency_decision_to_first_place_ms: Optional[float] = None
        self.quote_decision_perf_counter: Optional[float] = None
        # Event-driven quote loop: set by public BBO / private fill+order threads.
        self.quote_wake_event = threading.Event()
        # Two distinct clocks tracking REST account/position refreshes.
        # The 2026-05-16 Codex #1 fix split this into two anchors
        # because the previous single field defeated the stale-account
        # gate on REST-flaky days:
        #
        # * ``account_rest_last_monotonic`` — last ATTEMPT (success or
        #   failure). Used by the throttle-spacing logic in
        #   ``Bot._needs_account_only_refresh`` so failed attempts
        #   still consume budget and don't tight-loop into 429s.
        # * ``account_rest_last_success_monotonic`` — last successful
        #   refresh. Used by the ``account_data_stale`` risk gate so
        #   "seconds since last good account data" measures actual
        #   freshness, not retry cadence. Without this split, a
        #   sustained REST-failure regime kept advancing the attempt
        #   clock every retry interval and the stale gate never fired
        #   while position/account truth could be hours out of date.
        self.account_rest_last_monotonic: Optional[float] = None
        self.account_rest_last_success_monotonic: Optional[float] = None
        # Last-tick decomposed latency (ms); updated at end of ``Bot.one_tick``.
        self.latency_hot_path_local_compute_ms: Optional[float] = None
        self.latency_quote_engine_build_ms: Optional[float] = None
        self.latency_order_maintenance_local_ms: Optional[float] = None
        self.latency_account_refresh_rest_ms: Optional[float] = None
        self.latency_reconcile_rest_ms: Optional[float] = None
        self.latency_order_submit_rtt_ms: Optional[float] = None
        # one_tick() preamble cost before the hot-path timer starts.
        self.latency_tick_preamble_ms: Optional[float] = None
        self.latency_private_queue_wait_ms: Optional[float] = None
        self.latency_public_ws_queue_wait_ms: Optional[float] = None
        self.exec_runtime_counters: dict[str, Any] = {}
        # TODO-004: cumulative wall-clock seconds where risk returned NO_QUOTE
        # AND at least one passive order was still resting on the book. Should
        # be ~0 in healthy operation; non-zero quantifies BUG-009-class
        # exposure. ``_blind_resting_last_check_mono`` is the monotonic-clock
        # anchor used to compute per-tick deltas (None until the first sample).
        self.blind_resting_seconds_total: float = 0.0
        self.blind_resting_tick_count: int = 0
        self._blind_resting_last_check_mono: Optional[float] = None
        # TODO-001: inventory consistency watchdog state.
        # ``inventory_baseline_qty`` is the position seeded at session start;
        # ``session_signed_qty_total`` accumulates signed sizes from session
        # fills (BUY positive, SELL negative). The expected current position is
        # ``baseline + session_signed_qty_total``; comparing to
        # ``state.position.position_qty`` (the venue truth) catches divergence
        # caused by bugs in the address-plumbing or refresh paths (BUG-005
        # was the canonical example).
        self.inventory_baseline_qty: float = 0.0
        self.inventory_baseline_set: bool = False
        self.session_signed_qty_total: float = 0.0
        self.inventory_consistency_last_check_mono: Optional[float] = None
        self.inventory_consistency_breach_count: int = 0
        self.inventory_consistency_last_breach: Optional[dict[str, Any]] = None
        # Hardening (post-snap_20260426_103520 false positive): consecutive
        # divergent reads before we declare a real breach. Resets to 0 on
        # the first non-breach sample OR on a rebaseline-via-/resume. The
        # threshold lives in
        # ``settings.inventory_consistency_consecutive_breaches_required``.
        self.inventory_consistency_consecutive_drift_count: int = 0
        # Private websocket (order/fill stream) — connectivity flags updated from bot thread
        # (OrderManager); timestamps / transport metrics also updated from the private WS thread when set.
        self.private_ws_connected: bool = False
        self.private_ws_healthy: bool = False
        # WS writer thread: bounded queue full → drops counted; bot thread clears pending via OrderManager.drain.
        self.private_ws_queue_drops: int = 0
        self.private_ws_recovery_pending: bool = False
        self.private_ws_last_connect_wall_ts: Optional[datetime] = None
        self.private_ws_last_ping_sent_wall_ts: Optional[datetime] = None
        self.private_ws_last_pong_wall_ts: Optional[datetime] = None
        self.private_ws_last_message_wall_ts: Optional[datetime] = None
        self.private_ws_disconnect_reason_last: Optional[str] = None
        self.private_ws_disconnect_close_code_last: Optional[int] = None
        self.private_ws_disconnect_histogram: dict[str, int] = {}
        self.private_ws_reconnect_count: int = 0
        self.private_ws_reconnect_reason_counts: dict[str, int] = {}
        self.private_ws_queue_high_watermark: int = 0
        self.private_ws_queue_backlog_after_drain_count: int = 0
        self.private_ws_max_events_drained_per_tick: int = 0
        self.private_ws_events_drained_last_tick: int = 0
        self.private_ws_queue_depth_after_drain: int = 0
        self.private_ws_drain_time_ms_last_tick: float = 0.0
        self.private_ws_last_inbound_derived_ms: dict[str, Any] = {}
        # OKX-specific WS diagnostics (populated by app/exchange/okx_ws.py
        # only; stays at defaults on Bluefin / GRVT / Hyperliquid). Added
        # 2026-05-08 to close the visibility gap that hid whether silent
        # OKX sockets were causing the gone_on_exchange spiral.
        self.okx_ws_orders_msgs_session: int = 0
        self.okx_ws_orders_msgs_by_state: dict[str, int] = {}
        self.okx_ws_pong_gap_seconds_max: float = 0.0
        # Tape runtime-feed (shared-memory) health + consumer counters
        # (M7.6 / M8.4). All stay 0 when REGIME_USE_RUNTIME_RECORDER_FEED
        # is false (the default) — the feed is never opened, so nothing
        # increments. The quote loop MIRRORS the reader's monotonic
        # counters into the *_count fields each tick it reads (assignment,
        # not increment); the consumer *_fires / *_seeded counters are
        # bumped by the bot directly. Surfaced in snapshot_dict() ->
        # state_current.json for gate G6 (feed live) and G7 (consumer
        # firing). See app/runtime_recorder_feed.py.
        self.runtime_feed_read_count: int = 0
        self.runtime_feed_stale_count: int = 0
        self.runtime_feed_version_mismatch_count: int = 0
        self.runtime_feed_collision_count: int = 0
        # Candidate A (warm-start vol seed) — one-shot at startup, so this
        # is 0 or 1 in steady state (>1 only across an in-process re-seed).
        self.warmstart_vol_seeded_from_recorder_count: int = 0
        # Candidate B (microprice-dev-z widen) — per-tick fires count.
        self.microprice_widen_z_runtime_feed_fires: int = 0
        # Per-order lifecycle trace ring buffer. Phase 2 of the
        # gone_on_exchange diagnostic (2026-05-08). Phase 1 confirmed
        # OKX socket health; Phase 2 records each order's full history
        # (place dispatch -> place response -> WS events -> terminal)
        # so we can tell *which* of the three branches caused each
        # gone_on_exchange event. See app/order_trace.py.
        self.order_trace = OrderTraceBuffer()
        # Quote-age fill bucketing (todo-006). Aggregates fills into
        # 5 quote-age buckets (<100ms / 100-500ms / 500ms-2s / 2-10s /
        # 10s+) so the operator can see at a glance whether the bot's
        # edge lives in slow passive fills or is being eaten by fast
        # toxic fills. Surfaced live in the Bot Stats panel via
        # live_stats. See app/fill_bucket_metrics.py.
        self.fill_buckets = FillBucketAggregator()
        self.public_ws_events_applied_last_tick: int = 0
        self.public_ws_last_inbound_derived_ms: dict[str, Any] = {}
        self.public_ws_max_burst_per_tick: int = 0
        # 1.2.25: session-cumulative BBO event counters (todo-010
        # Phase 2 decision data). ``bbo_event_count_session``
        # counts every public-WS top-of-book apply;
        # ``mid_change_count_session`` counts only those that
        # actually changed the mid price. Together with
        # session_duration they yield BBO/sec and mid-change/sec
        # rates — needed to decide whether Phase 2 (push from BBO
        # event handler instead of quote-cycle tick) buys
        # additional resolution beyond the v1.2.24 dedup fix.
        self.bbo_event_count_session: int = 0
        self.mid_change_count_session: int = 0
        # v1.5.244 — rolling rate of mid changes (5-min window).
        # Source: each mid change appends its monotonic timestamp to
        # `mid_change_recent_ts`; on read, prune entries older than
        # 300s and report `len(deque) / 5.0` as the per-minute rate.
        # Surfaces the bot's binding constraint on fill rate (each
        # session's fills are ~10% of mid changes; below ~6/min the
        # session is unlikely to clear the 25-fills / 30-min calibration
        # validity floor). Read by snapshot_dict, live_stats, and the
        # market-activity acceptance check.
        from collections import deque as _deque
        self.mid_change_recent_ts: _deque[float] = _deque(maxlen=10000)
        # 1.3.31 (todo-029 § 6c): session-cumulative count of
        # cross-venue cancels triggered by ``binance_cross_venue_cancel``
        # in execution.py. Read by live_stats so the dashboard's
        # Market-tab cross-venue card (todo-029) can show
        # "this session: N cancels driven by Binance leading the
        # quote-touch by more than the configured threshold".
        # Incremented on the quote thread under no specific lock —
        # plain ``int`` write of a session-cumulative counter, a
        # torn read at worst yields a stale-by-one count.
        self.session_cross_venue_cancel_count: int = 0
        # v1.4.21 (rate-limit work): companion counter for cross-venue
        # AMENDs (vs cancels). When the Binance cross-venue trigger
        # fires and amend is viable + a target price is available from
        # the last quote breakdown, ``_maybe_cancel_on_binance_move``
        # routes through ``_enqueue_amend_quote_path`` instead of
        # ``_enqueue_cancel_quote_path`` — saving one venue call per
        # reprice. This counter tracks how often the amend path won.
        # The cancel + amend counts together are the total cross-venue
        # trigger volume.
        self.session_cross_venue_amend_count: int = 0
        # 2026-05-14 orphan-fill race fix. Increments every time the
        # reconcile path declined to mark a working order as
        # ``gone_on_exchange`` because the open-orders REST request
        # was dispatched BEFORE the order's ts_ack / ts_sent. The
        # snapshot couldn't have contained the order; declaring it
        # gone would have orphaned it on the venue (silent fill risk).
        # Surfaced in /state/current for the operator-visible health
        # signal and used in tests to assert the guard fired.
        self.reconcile_skip_snapshot_stale_total: int = 0
        # 2026-05-14 BUG-024. CRITICAL counter — increments every time
        # the place-response handler classified the venue's reply as
        # ``unconfirmed`` (neither a clean accept nor a clean reject).
        # Each occurrence triggers ``Bot.kill("place_response_unconfirmed")``
        # via the OrderManager → Bot callback, so in practice this
        # counter caps at 1 per session; the bot halts and the
        # operator must investigate before restart.
        #
        # Per operator decree (2026-05-14): missing ack/reject is NOT
        # benign — it's a critical connectivity failure. Even one
        # occurrence per month warrants stopping the bot, flattening,
        # and surfacing the failure to Telegram + dashboard. The bot
        # must NEVER silently absorb venue responses it can't parse.
        self.place_unconfirmed_critical_total: int = 0
        # 1.3.59 todo-execution-quality: cancel-race counter. Incremented
        # every time the bot's cancel request received a benign-missing
        # response from the venue — meaning the cancel arrived AFTER the
        # fill/cancel had already settled. Each increment is a window in
        # which the venue could have (and may have) filled the order
        # against an aged quote before the cancel landed. Zero hot-path
        # cost: bumped only in the existing ``benign_missing`` branch
        # that was already classifying the response.
        self.cancel_race_lost_to_fill_total: int = 0
        # 1.3.120: count cancel responses classifying as
        # ``unexpected_gone`` — the order is gone but NOT via fill.
        # OKX sCodes 51400 / 51401 / 51503 land here. Could be
        # stale-state retry races (we already cancelled, OKX returns
        # "already canceled"), venue admin / risk actions, or real
        # state-desync bugs (we think order exists, OKX says no). The
        # operator should investigate WHY this counter climbs — it's
        # NOT a normal cancel-race-lost-to-fill (that's
        # ``cancel_race_lost_to_fill_total`` above which counts
        # 51402 only). See ``app/exchange/okx_responses.py``
        # ``_UNEXPECTED_GONE_CODES`` for the response-code rationale.
        self.cancel_unexpected_gone_total: int = 0
        # 1.3.82 connectivity diagnostics: count "place-then-immediate-
        # cancel" races. Bumped when a cancel is enqueued for a WO whose
        # ts_ack is still None AND the cancel comes within 50ms of
        # ts_sent — the exact signature of the 50 phantom-place
        # gone_on_exchange events observed in snapshot 260515-095056.
        # Each event indicates the bot's quote-cycle or outbound
        # dispatcher racing its own place. Surfaced on the
        # Connectivity tab.
        self.place_cancel_race_total: int = 0
        # v1.4.96 — three-tier connectivity classification (2026-05-19).
        # Re-shaped from the v1.4.95 single-bucket design after operator
        # feedback: WS-stream latency spikes are categorically different
        # from "bot doesn't know if its order is on the venue" and
        # should not share a counter. The three tiers are independent;
        # each has its own severity in the postmortem gate:
        #
        # TIER 1 — gone_on_exchange_total (FATAL)
        #
        #   The bot has NEITHER WS terminal NOR HTTP confirmation that
        #   its place/cancel reached the venue. Sub-classified by the
        #   missing-confirmation signature. Any non-zero value is a
        #   postmortem fatal finding because the venue may be in an
        #   unknown state relative to the bot's local view:
        #
        #     * phantom_no_ack (BUG-024) — place sent, no HTTP ack
        #     * acked_no_cancel (BUG-023) — order acked-live, no
        #       cancel ever dispatched (silent leak risk)
        #     * cancel_no_http_confirm — cancel sent (ts_cancel_sent
        #       set) but no HTTP success response. Bot doesn't know
        #       whether the cancel landed.
        #     * other — unclassified
        #
        # TIER 2 — http_acked_no_ws_total (WARN, tolerated if rare)
        #
        #   Cancel HTTP returned success — the venue confirmed the
        #   cancel landed. The WS `canceled` terminal event simply
        #   hadn't arrived by the time reconcile checked. Benign on
        #   the venue side (no orphan-fill risk) but each occurrence
        #   indicates a WS publish leg that was slow or never
        #   delivered. The 6 events in snapshot v1.4.92-260519-161411
        #   were ALL this signature — previously misclassified as
        #   gone_on_exchange.
        #
        # TIER 3 — ws_arrived_late_total (WARN, tolerated if rare)
        #
        #   The WS terminal event for a previously-terminated oid
        #   eventually arrived after the local state had already
        #   cleaned up the slot. Distinct from tier 2: the WS DID
        #   deliver eventually, just past the reconcile cycle. Bumped
        #   in ``_handle_private_order_update_v2`` when an unmatched
        #   WS event resolves to an oid in
        #   ``_terminated_oids_recent``.
        #
        # CONTRACT — A healthy session has:
        #   * gone_on_exchange_total == 0 (FATAL gate)
        #   * http_acked_no_ws_total < ~50 (WARN gate)
        #   * ws_arrived_late_total < ~50 (WARN gate)
        self.gone_on_exchange_total: int = 0
        # BUG-024 signature: ts_ack None — place sent but no parseable
        # HTTP ack arrived; reconcile cleaned up locally without the
        # bot knowing whether the order landed on the venue.
        self.gone_on_exchange_phantom_no_ack_total: int = 0
        # BUG-023 signature: ts_ack set AND ts_cancel_requested None.
        # Order was live on the venue, bot declared it gone without
        # ever dispatching a real cancel. Highest-risk subcategory
        # because the venue may still believe the order is open.
        self.gone_on_exchange_acked_no_cancel_total: int = 0
        # v1.4.96 third RED subcategory: cancel was sent
        # (ts_cancel_requested set) but no HTTP-success response.
        # Bot doesn't know whether the cancel reached the venue. The
        # venue may still have the order open. (Distinct from the
        # tier-2 http_acked_no_ws case where the HTTP DID come back
        # success.)
        self.gone_on_exchange_cancel_no_http_confirm_total: int = 0
        # Unclassified gone_on_exchange events that match no specific
        # signature. Should stay at zero; non-zero means we discovered
        # a new failure mode worth named-classifying.
        self.gone_on_exchange_other_total: int = 0
        # Bounded ring of the last N=100 RED gone_on_exchange terminals.
        # Each entry is a JSON-serialisable dict with the minimum
        # forensic context: timestamps, oid, side, px/sz, signature,
        # ws_terminal_lateness_ms (when applicable). Fixed ~20 KB worst
        # case; survives the entire session.
        self.gone_on_exchange_recent: deque[dict[str, Any]] = deque(maxlen=100)
        # v1.4.96 TIER 2 (WARN) — cancel HTTP succeeded, WS terminal
        # didn't arrive in time. Tolerable if rare. Has its own
        # bounded ring + lateness histogram (ts_closed -
        # ts_cancel_acked).
        self.http_acked_no_ws_total: int = 0
        self.http_acked_no_ws_recent: deque[dict[str, Any]] = deque(maxlen=100)
        self.http_acked_no_ws_lateness_ms_min: Optional[float] = None
        self.http_acked_no_ws_lateness_ms_max: Optional[float] = None
        self.http_acked_no_ws_lateness_samples: deque[float] = deque(maxlen=100)
        # v1.4.96 TIER 3 (WARN) — WS terminal arrived AFTER local state
        # had already cleaned up the oid. Bumped from the unmatched-WS
        # event handler in ``execution.py`` when ``ev.oid`` matches an
        # entry in ``_terminated_oids_recent``.
        self.ws_arrived_late_total: int = 0
        self.ws_arrived_late_recent: deque[dict[str, Any]] = deque(maxlen=100)
        # Small bounded set/dict of recently-terminated oids so the
        # unmatched-WS-event handler can detect "WS arrived late"
        # cases. Keyed by ``order_id_exchange`` (str); value carries
        # the terminal context (which tier it was in, when it was
        # declared terminal). Iteration is bounded; lookup is O(1).
        # Bounded at 200 entries (FIFO via OrderedDict.popitem) — far
        # more than the steady-state needs since both tier-1 RED and
        # tier-2 WARN events are expected to be rare in healthy
        # sessions.
        from collections import OrderedDict as _OD
        self.terminated_oids_recent: _OD[str, dict[str, Any]] = _OD()
        # Maximum size of terminated_oids_recent. Keep public so
        # tests can shrink to exercise eviction.
        self.terminated_oids_recent_max: int = 200
        # v1.4.100 ladder-observability F2 — drop-attribution counters.
        # Track WHY an outer ladder rung was dropped before placement,
        # so the operator can distinguish "soft inventory-buffer push
        # collapsed outer rung to inside grid value" from "grid-collision
        # accident" from "position cap refused the rung" etc. Without
        # these, the v1.4.92 production-observed "1-vs-2 rung asymmetry"
        # could only be diagnosed by code-reading the build path.
        #
        # Each rung drop fires exactly one bump (mutually exclusive
        # at each drop site). Sum across categories == total rungs
        # dropped this session. The `other` category should stay
        # at zero; non-zero means we discovered a new drop path worth
        # named-classifying.
        #
        # See plans/ladder-observability.md F2 for the full taxonomy.
        self.ladder_rung_dropped_grid_collision_total: int = 0
        self.ladder_rung_dropped_min_notional_total: int = 0
        self.ladder_rung_dropped_inventory_buffer_total: int = 0
        self.ladder_rung_dropped_inventory_aware_pruning_total: int = 0
        self.ladder_rung_dropped_position_cap_total: int = 0
        self.ladder_rung_dropped_in_flight_total: int = 0
        # v1.5.26 Phase 2C.3 -- regime-mode ladder-pruning attribution.
        # Pre-fix the regime_knobs.ladder_levels_max cap reduced
        # effective_levels in bot.py BEFORE LadderConfig was built,
        # so the rungs were "never computed" rather than "dropped"
        # and no on_rung_dropped callback fired. This counter is
        # bumped explicitly at the call site whenever the regime
        # cap pulls effective_levels below cfg, by (cfg - effective)
        # so it counts per-side-equivalent rungs cut from the
        # configured ladder shape.
        self.ladder_rung_dropped_regime_mode_total: int = 0
        # v1.5.26 Phase 2B -- canonical multi-horizon mid-drift cache.
        # The bot's regime evaluation block populates this each tick
        # via ``compute_mid_drift_windows(mid_samples, ...)``. Observers
        # (snapshot, postmortem, future gate migrations) read all 7
        # windows from a single source. ``None`` until the first tick
        # computes it; observers default to "warmup" in that case.
        from app.mid_drift_windows import MidDriftWindows as _MDW
        self.mid_drift_windows: Optional[_MDW] = None
        self.ladder_rung_dropped_other_total: int = 0
        # v1.4.222 — Phase 4G.11 instrumentation. Counts outer-rung
        # drops at the DISPATCHER's ``normalize_order_pair`` step (in
        # ``compute_desired_state``), separately from the build_ladder
        # counters above (which only catch drops INSIDE build_ladder).
        # Driving incident: v1.4.219 snapshot showed 0 rung-1 orders
        # across 601 total despite 2-rung config, and the existing
        # counters were all 0 — the drop site was downstream of
        # build_ladder. Two distinct sub-reasons:
        #   * ``normalize_below_min_notional_total`` — rung sized down
        #     by venue ``size_step`` rounding fell below
        #     ``spec.min_notional_usd``. Common on integer-step venues
        #     (TON-USDT-SWAP size_step=1.0) when QUOTE_NOTIONAL is low.
        #   * ``normalize_other_total`` — any other normalize_order_pair
        #     rejection (price-grid issues, NaN inputs, etc.).
        # When ``normalize_below_min_notional_total`` is non-zero and
        # ``MIN_QUOTE_NOTIONAL_USD`` is already at the venue floor,
        # operator should raise ``QUOTE_NOTIONAL_USD`` (more base
        # contracts at the inside → larger geometric-decay result for
        # outer rungs) — see v1.4.222 plan entry for the TON math.
        self.ladder_rung_dropped_normalize_below_min_notional_total: int = 0
        self.ladder_rung_dropped_normalize_other_total: int = 0
        # v1.4.228 — Phase 4G.13. Structural-bias auto-throttle
        # session-cumulative fire counter. Bumps every tick the
        # gate forces reducing-only eligibility because the
        # ``inventory_exec_bias`` suppression ratio crossed
        # ``STRUCTURAL_BIAS_AUTO_THROTTLE_RATIO_THRESHOLD``. Per-
        # tick (NOT per-engagement) so the counter reflects the
        # cumulative time the bot was throttled this session.
        # Surfaced in ``behavioural_gates.structural_bias_throttle``.
        self.structural_bias_throttle_fire_count_total: int = 0
        # v1.5.155 — engagement-vs-effective distinction. The
        # v1.5.154-260526-074029 snapshot showed fire_count = 508,550
        # — alarming until inspection revealed the cumulative bias
        # ratio was 1.05x (370/388) at snapshot time, well below the
        # 5x threshold. The high fire_count likely reflects historical
        # tick-by-tick repeated firings when the gate WAS above
        # threshold + position-aligned but where some upstream gate
        # (e.g. shock_gate's HOLD_ALL) had already produced a more-
        # restrictive eligibility. In that case more_restrictive
        # returns the upstream value unchanged → this gate didn't
        # actually narrow anything but still counted as a fire.
        # This counter bumps ONLY when the gate's contribution
        # actually narrowed eligibility (more_restrictive returned a
        # value different from eff_q.eligibility) — the actionable
        # number for "did the throttle suppress a quote this tick?".
        self.structural_bias_throttle_changed_eligibility_total: int = 0
        # Last-tick ratio + per-side suppression counts. Published in
        # the snapshot so operators can verify "is the gate close to
        # firing?" without having to inspect quote_quality counts
        # separately. Updated on every engagement (when the position +
        # ratio guards pass); stale between engagements.
        self.structural_bias_throttle_last_ratio: float = 0.0
        self.structural_bias_throttle_last_bid_count: int = 0
        self.structural_bias_throttle_last_ask_count: int = 0
        # v1.5.197 — monotonic timestamp of the last
        # ``inventory_exec_bias`` counter decay. Used by
        # ``Bot._apply_structural_bias_throttle_gate`` to ensure the
        # decay fires at most once per IDLE_DECAY_SECONDS window
        # (avoid hammering the counters into zero on every tick of a
        # long idle stretch). Initialized to 0 so the first idle
        # decay fires as soon as the idle window is reached.
        self._structural_bias_last_decay_mono: float = 0.0
        # Latched flag — True iff the gate fired in the most recent
        # tick. Used by snapshot + Telegram alerts. False between
        # engagements.
        self.structural_bias_throttle_active_last_tick: bool = False
        # Direction of the lean (when active): "LONG", "SHORT", or
        # None when inactive. Surfaced in the snapshot for operator
        # visibility.
        self.structural_bias_throttle_direction: Optional[str] = None
        # v1.4.169 Phase 2J — tick-floor adjustment counter. Bumped
        # once per OUTER rung whose bps-derived price was shifted
        # outward by the tick-floor (= rung was about to land within
        # 1 tick of the inside-rung grid before this fix). A non-zero
        # session value means either ``half_spread_bps`` is unusually
        # tight for the symbol's tick size, or ``offset_step`` is too
        # small, or ``num_levels_per_side`` is too high. Per-side
        # split for diagnostic clarity.
        self.ladder_rung_tick_floor_adjusted_bid_total: int = 0
        self.ladder_rung_tick_floor_adjusted_ask_total: int = 0
        # 1.3.85: count hydration attempts skipped because the bot
        # already recorded a recent terminal for that (oid, cloid).
        # Bumped from ``_hydrate_working_from_exchange`` whenever the
        # ``was_recently_terminal`` guard fires. Each increment is a
        # WS-vs-REST race we successfully avoided turning into a
        # ``gone_on_exchange`` event. Surfaced on the Connectivity
        # tab and in postmortem connectivity-health.
        self.hydration_skipped_recently_terminal_total: int = 0
        # amend-prio Phase 4 (v1.4.17): per-outcome counters for the
        # amend rollout. Each amend response increments exactly one of
        # these (mutually exclusive). The operator-facing Telegram
        # ``/status`` line + postmortem ``amend_attribution`` section
        # consume these. Surfaced via ``execution.snapshot_stats()``
        # alongside place / cancel counters.
        #   * ``amend_intents_emitted_total``  — every call to
        #     ``_enqueue_amend_quote_path``; the denominator for
        #     amend-success rate
        #   * ``amend_success_total``           — sCode 0 (accepted)
        #   * ``amend_below_filled_total``      — sCode 51016
        #   * ``amend_order_gone_total``        — sCode 51400/51401/51503
        #   * ``amend_post_only_cross_total``   — sCode 51604
        #   * ``amend_exchange_rejected_other`` — other sCode rejects
        #   * ``amend_transport_rejected_total``— row-rate-limit + envelope
        #     transport failures
        #   * ``amend_pending_high_watermark``  — max concurrent
        #     AMEND_PENDING WOs observed this session
        self.amend_intents_emitted_total: int = 0
        self.amend_success_total: int = 0
        self.amend_below_filled_total: int = 0
        self.amend_order_gone_total: int = 0
        self.amend_post_only_cross_total: int = 0
        self.amend_exchange_rejected_other_total: int = 0
        self.amend_transport_rejected_total: int = 0
        self.amend_pending_high_watermark: int = 0
        # v1.5.206 — counts amend-response unconfirmed swallows where the
        # bot had LOCALLY removed the WO between amend-send and amend-
        # response (the original tombstone race).
        self.amend_unconfirmed_swallowed_by_tombstone_total: int = 0
        # v1.5.216 — counts the broader ``missing_row_for_amend`` swallow
        # where the bot's batch-response parser cannot correlate a row
        # to the amend it sent. v1.5.215-260528-180032 incident:
        # ord_id=3606222898225680384 SELL, killed after 4m46s. Reading
        # this counter in /status tells the operator how often the
        # ambiguity arises and whether the post-swallow reconcile
        # produces a clean recovery each time.
        self.amend_missing_row_swallowed_total: int = 0
        # v1.5.216 — set by the missing_row swallow path; consumed by
        # the next reconcile tick which then clears it. Forces a
        # reconcile to resolve the ambiguity (alive at old px/sz vs
        # cancelled vs filled).
        self.force_reconcile_requested: bool = False
        # rate-limit-observability Phase 2 (v1.4.20): per-pool snapshot
        # of the OKX rate-limit window. Populated by the bot's
        # heartbeat handler from
        # ``client.rest_runtime_counters()["okx_rate_window_per_pool"]``
        # each tick. Surfaced via ``snapshot_dict()`` so
        # ``/state/current`` (and the Connectivity dashboard panel)
        # shows live per-pool pressure: current 2s rate, peak 60s,
        # cap, pct_of_cap. Empty dict when the adapter doesn't
        # expose per-pool data (non-OKX venues) or before the first
        # heartbeat tick.
        self.okx_rate_window_per_pool: dict[str, Any] = {}
        # 1.3.86: count cancels deferred until the place ack lands.
        # The bot's quote-cycle / cross-venue cancel / soft-flatten
        # paths sometimes try to cancel an order before its place
        # response has returned. Pre-1.3.86 we dispatched a
        # cancel-by-cloid that could beat the place at the venue,
        # producing phantom-orphan orders. Now we PARK those cancels
        # and flush them right after the ACKED transition fires.
        # Each increment is a successfully-deferred race; the cancel
        # later fires safely against an acked order.
        self.cancel_deferred_until_ack_total: int = 0
        # 1.2.26 (todo-010 Phase 2): callbacks fired when the mid
        # changes, from the public-WS BBO handler. Subscribers see
        # every real mid tick (~7-8/sec on OKX SUI) instead of only
        # the cycle-rate sampling (2/sec). Used by the bot to push
        # to ``VolatilityEstimator`` directly from the WS thread.
        # Registration is single-threaded at bot startup; firing
        # happens on the WS thread. Callbacks must be cheap and
        # fail-safe (any raise is logged + swallowed; we never let
        # one bad listener take down the WS handler).
        self._on_mid_change_callbacks: list = []
        # Last quote decision: book freshness vs decision instant (for fills + telemetry).
        self.last_effective_book_age_at_decision_ms: Optional[float] = None
        self.last_decision_book_ts_exchange_ms: Optional[int] = None
        self.last_decision_book_apply_mono: Optional[float] = None
        self.last_book_apply_to_decision_ms: Optional[float] = None
        self.last_decision_public_ws_queue_wait_ms: Optional[float] = None
        self.last_decision_public_ws_receive_to_apply_ms: Optional[float] = None
        self.last_decision_market_data_regime: Optional[str] = None
        # Public websocket (BBO) — updated from the public WS thread and the bot thread (reconnect).
        self.public_ws_connected: bool = False
        self.public_ws_last_message_wall_ts: Optional[datetime] = None
        self.public_ws_reconnect_count: int = 0
        self.public_ws_seen_first_bbo: bool = False
        self.live_market_data_source: str = "none"
        # Last quote refresh aging/distance diagnostics (API /state/current, status.log via diagnostics.py).
        self.quote_refresh_aging: dict[str, Any] = {}
        # Startup / live exchange snapshot health (bot thread).
        self.exchange_snapshot_unhealthy_streak: int = 0
        self.reconcile_auto_pause: bool = False
        # Operator metrics (UTC calendar day); persisted when PERSISTENT_RUNTIME_STATE_PATH is set.
        self.operator_day_anchor_utc: date = utc_now().date()
        self.daily_realized_pnl: float = 0.0
        self.daily_trade_count: int = 0
        self.daily_traded_notional: float = 0.0
        self.operator_last_fill_ts: Optional[datetime] = None
        self.recent_buy_fill_count: int = 0
        self.recent_sell_fill_count: int = 0
        self.operator_rolling_toxicity_markout_bps: Optional[float] = None
        self.operator_rolling_one_sided_fill_ratio: Optional[float] = None
        # v1.4.40 BUG-025: cache of OrderManager / OutboundDispatchCoordinator
        # internal state machine fields. Populated by
        # ``OrderManager._sync_executor_state_snapshot`` on every
        # quote cycle + on every outbound batch event. Surfaced in
        # ``snapshot_dict`` as the top-level ``executor_state`` block
        # so a snapshot at wedge time exposes which specific latch
        # is stuck. See ``issues/bug-025-executor-silent-wedge.md`` for
        # the blocker that motivated this — bot enters a state with
        # valid engine output, clean eligibility, no gates, but no
        # place_attempt fires, and only the 600 s deadlock watchdog
        # catches it. Empty until the first cycle populates it.
        self.executor_state_snapshot: dict[str, Any] = {}
        # Outbound action dispatcher + transport path (OrderManager → HyperliquidClient).
        self.outbound_action_queue_depth: int = 0
        self.outbound_action_queue_hwm: int = 0
        self.outbound_avg_batch_size: float = 0.0
        self.outbound_action_last_batch_size: float = 0.0
        self.outbound_ws_action_send_count: int = 0
        self.outbound_http_action_send_count: int = 0
        self.outbound_coalesced_intent_count: int = 0
        self.outbound_dropped_stale_intent_count: int = 0
        self.outbound_transport_mode_current: str = "http"
        self.outbound_place_intent_to_ack_ms: Optional[float] = None
        self.outbound_cancel_intent_to_closed_ms: Optional[float] = None
        self.outbound_ack_to_private_ws_lifecycle_ms: Optional[float] = None
        self.outbound_signing_latency_ms: Optional[float] = None
        self.outbound_action_batch_wait_ms: Optional[float] = None
        self.outbound_quote_cycle_to_first_transport_send_ms: Optional[float] = None
        self.outbound_quote_cycle_to_first_ack_ms: Optional[float] = None

    def _operator_fill_utc_date(self, ts_fill: datetime) -> date:
        u = ts_fill if ts_fill.tzinfo else ts_fill.replace(tzinfo=timezone.utc)
        return u.astimezone(timezone.utc).date()

    def _reset_operator_day_scoped_unlocked(self, *, anchor: date) -> None:
        self.operator_day_anchor_utc = anchor
        self.daily_realized_pnl = 0.0
        self.daily_trade_count = 0
        self.daily_traded_notional = 0.0
        self.recent_buy_fill_count = 0
        self.recent_sell_fill_count = 0
        self.operator_last_fill_ts = None
        self.operator_rolling_toxicity_markout_bps = None
        self.operator_rolling_one_sided_fill_ratio = None

    def maybe_rotate_operator_day_to_wall_clock(self) -> None:
        """At UTC midnight, zero day-scoped operator metrics without waiting for a fill."""
        today = utc_now().date()
        with self._lock:
            if today != self.operator_day_anchor_utc:
                self._reset_operator_day_scoped_unlocked(anchor=today)

    def apply_persistent_runtime_state(self, loaded: PersistentRuntimeState) -> None:
        """
        Restore operator metrics from disk. If ``loaded.day_anchor_utc`` is not today's UTC date,
        day-scoped counters are reset and the anchor is set to today (prior-day file is not merged).

        Soft-flatten resume: ``soft_flatten_active`` is restored from
        the saved file regardless of day rotation. The pre-crash bot
        was in the middle of patient post-only flatten; we resume in
        the same mode so the position close-out continues without
        waiting for the drawdown gate to re-fire.
        """
        today = utc_now().date()
        with self._lock:
            if loaded.day_anchor_utc == today:
                self.operator_day_anchor_utc = loaded.day_anchor_utc
                self.daily_realized_pnl = loaded.daily_realized_pnl
                self.daily_trade_count = loaded.daily_trade_count
                self.daily_traded_notional = loaded.daily_traded_notional
                self.operator_last_fill_ts = loaded.last_fill_ts
                self.recent_buy_fill_count = loaded.recent_buy_fill_count
                self.recent_sell_fill_count = loaded.recent_sell_fill_count
                self.operator_rolling_toxicity_markout_bps = loaded.rolling_toxicity_markout_bps
                self.operator_rolling_one_sided_fill_ratio = (
                    loaded.rolling_one_sided_fill_ratio
                )
            else:
                self._reset_operator_day_scoped_unlocked(anchor=today)
            # Soft-flatten: restore unconditionally (not day-rotated).
            # The flatten mode is about LIVE position state, not
            # daily metrics; a midnight rotation shouldn't drop it.
            if loaded.soft_flatten_active:
                self.soft_flatten_active = True
                if loaded.soft_flatten_started_at_iso:
                    try:
                        from datetime import datetime as _dt
                        # The on-disk timestamp was a wall-clock ISO
                        # of when soft-flatten originally started; we
                        # don't have monotonic continuity across
                        # restarts, so set the worker's started_at
                        # to "now" for phase-timer purposes. Pre-
                        # restart phase progress is lost (acceptable
                        # — phase-1/2 spans seconds, restart takes
                        # longer). The wall-clock start is kept for
                        # diagnostics via ``soft_flatten_started_at_utc``.
                        _dt.fromisoformat(
                            loaded.soft_flatten_started_at_iso.replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass
                self.soft_flatten_started_at_mono = _clock.monotonic()
                # ``bot_status`` is set in main.py after this method;
                # main.py also re-checks ``soft_flatten_active`` and
                # transitions to SOFT_FLATTENING accordingly.
            # 2026-05-12 N12 wiring (telemetry plan Step 1): seed the
            # basis-regime classifier from the persisted result fields.
            # Closes the "5-10 min regime-blind after restart" gap —
            # the classifier's diagnostic state (`last_regime_sign`,
            # `last_ic`, `pair_count`) resumes immediately. The full
            # pairs-buffer is NOT persisted; it re-warms over ~5 min.
            try:
                self.basis_regime.seed_from_persisted(
                    last_regime_sign=loaded.basis_regime_last_sign,
                    last_ic=loaded.basis_regime_last_ic,
                    pair_count=loaded.basis_regime_pair_count,
                )
            except Exception:
                logger.exception("basis_regime_seed_from_persisted_failed")

    def build_persistent_runtime_state(self) -> PersistentRuntimeState:
        with self._lock:
            sf_started_iso: Optional[str] = None
            if self.soft_flatten_active and self.soft_flatten_started_at_mono is not None:
                # Best-effort wall-clock snapshot for diagnostics. The
                # restart path doesn't restore monotonic continuity;
                # this just lets logs show "started ~30s before
                # crash, resumed after 5s of restart".
                sf_started_iso = utc_now().isoformat()
            # 2026-05-12 N12 wiring (telemetry plan Step 1): also
            # persist the basis-regime classifier's diagnostic fields
            # so a restart resumes with the previous result instead
            # of zero-warmup for 5-10 min. The pairs buffer itself
            # is not persisted; only the result.
            bs_last_sign: Optional[float] = None
            bs_last_ic: Optional[float] = None
            bs_pair_count: Optional[int] = None
            try:
                bs_last_sign = float(self.basis_regime.last_regime_sign)
                _ic = self.basis_regime.last_ic
                bs_last_ic = float(_ic) if _ic is not None else None
                bs_pair_count = int(self.basis_regime.pair_count)
            except Exception:
                # Defensive — never block the save on a classifier
                # access failure. Fields stay None and load-side
                # ``seed_from_persisted`` will skip seeding.
                pass
            return PersistentRuntimeState(
                day_anchor_utc=self.operator_day_anchor_utc,
                daily_realized_pnl=self.daily_realized_pnl,
                daily_trade_count=self.daily_trade_count,
                daily_traded_notional=self.daily_traded_notional,
                last_fill_ts=self.operator_last_fill_ts,
                recent_buy_fill_count=self.recent_buy_fill_count,
                recent_sell_fill_count=self.recent_sell_fill_count,
                rolling_toxicity_markout_bps=self.operator_rolling_toxicity_markout_bps,
                rolling_one_sided_fill_ratio=self.operator_rolling_one_sided_fill_ratio,
                soft_flatten_active=self.soft_flatten_active,
                soft_flatten_started_at_iso=sf_started_iso,
                basis_regime_last_sign=bs_last_sign,
                basis_regime_last_ic=bs_last_ic,
                basis_regime_pair_count=bs_pair_count,
            )

    def apply_session_resume(
        self,
        *,
        session_id: str,
        session_started_at_utc: datetime,
        session_fill_count: int,
        realized_pnl_usd: float,
        fees_usd: float,
        peak_equity_usd: Optional[float],
    ) -> None:
        """Restore cumulative session METRICS from storage on restart.

        SAFE to restore (cumulative; doesn't change behavior):
          - ``session_id``, ``session_started_at_utc`` — keeps PnL window
            continuous so /stats and the session-summary endpoint see
            the prior session's totals
          - ``session_fill_count`` — fill count visible in /status
          - ``pnl.realized_pnl_usd``, ``pnl.fees_usd``,
            ``pnl.session_peak_equity_usd`` — cumulative PnL fields

        BEHAVIORAL state is intentionally NOT touched by this method:
          - private-WS connection / pending-cancel / hash-side caches
          - cancel-confirmation gates, cooldown timers
          - toxicity rolling window, market-data recovery state
          - quote-eligibility gates, adaptive widen state
          - position cache (always reconcile from exchange via
            ``operator_metrics_reconcile`` after this is called)

        Restoring behavioral state would defeat the deadlock watchdog
        (re-introduce the latched gates the restart was meant to clear).
        ``test_session_resume.py`` has a regression guard.
        """
        with self._lock:
            self.session_id = session_id
            self.session_started_at_utc = session_started_at_utc
            self.session_fill_count = int(session_fill_count)
            existing = self.pnl
            peak = (
                float(peak_equity_usd)
                if peak_equity_usd is not None
                else float(existing.session_peak_equity_usd)
            )
            self.pnl = existing.__class__(
                realized_pnl_usd=float(realized_pnl_usd),
                unrealized_pnl_usd=existing.unrealized_pnl_usd,
                total_pnl_usd=float(realized_pnl_usd) + existing.unrealized_pnl_usd,
                fees_usd=float(fees_usd),
                equity_usd=existing.equity_usd,
                drawdown_usd=existing.drawdown_usd,
                session_peak_equity_usd=peak,
                ts=existing.ts,
            )

    def apply_operator_exchange_reconciliation(
        self,
        *,
        trade_count: int,
        traded_notional: float,
        realized_pnl: float,
        update_realized_pnl: bool,
        last_fill_ts: Optional[datetime],
        buy_count: int,
        sell_count: int,
    ) -> None:
        """
        Replace day-scoped trade/notional (and optionally realized PnL) from a startup
        ``user_fills`` aggregation. Does not clear toxicity rollups from the JSON load.
        """
        with self._lock:
            today = utc_now().date()
            if today != self.operator_day_anchor_utc:
                self._reset_operator_day_scoped_unlocked(anchor=today)
            self.daily_trade_count = int(trade_count)
            self.daily_traded_notional = float(traded_notional)
            self.recent_buy_fill_count = int(buy_count)
            self.recent_sell_fill_count = int(sell_count)
            self.operator_last_fill_ts = last_fill_ts
            if update_realized_pnl:
                self.daily_realized_pnl = float(realized_pnl)

    def bump_operator_metrics_on_session_fill(self, f: Fill, closed_pnl: float) -> None:
        """Increment UTC day operator stats for a session-scoped fill (caller ensures scope)."""
        with self._lock:
            d = self._operator_fill_utc_date(f.ts_fill)
            if d != self.operator_day_anchor_utc:
                self._reset_operator_day_scoped_unlocked(anchor=d)
            self.daily_realized_pnl += float(closed_pnl)
            self.daily_trade_count += 1
            self.daily_traded_notional += float(f.notional)
            self.operator_last_fill_ts = f.ts_fill
            if f.side == Side.BUY:
                self.recent_buy_fill_count += 1
                self.session_signed_qty_total += float(f.size)
            else:
                self.recent_sell_fill_count += 1
                self.session_signed_qty_total -= float(f.size)

    def note_private_ws_queue_drop(self, _label: str) -> None:
        """Called from the private WS thread when the bounded queue is full (event not enqueued)."""
        with self._lock:
            self.private_ws_queue_drops += 1
            self.private_ws_recovery_pending = True

    def note_private_inbound_metrics(
        self,
        _timing: InboundPrivateTiming,
        derived_ms: dict[str, Any],
    ) -> None:
        """Roll up last private WS user event timing (fill/order); keys from private_timing_derived_ms."""
        with self._lock:
            self.private_ws_last_inbound_derived_ms = dict(derived_ms)
            q = derived_ms.get("private_ws_queue_wait_ms")
            if isinstance(q, (int, float)):
                self.latency_private_queue_wait_ms = float(q)

    def note_public_inbound_apply(self, timing: InboundPublicTiming, derived_ms: dict[str, Any]) -> None:
        """After BBO applied to state; updates last public inbound timing snapshot."""
        with self._lock:
            self.public_ws_last_inbound_derived_ms = dict(derived_ms)
            w = derived_ms.get("public_ws_queue_wait_ms")
            if isinstance(w, (int, float)):
                self.latency_public_ws_queue_wait_ms = float(w)

    def add_mid_change_listener(self, callback) -> None:
        """1.2.26 (todo-010 Phase 2): register a callback that fires
        on every actual mid change, called from the public-WS BBO
        handler. Callable signature: ``(new_mid: float) -> None``.

        Used by ``Bot.__init__`` to wire ``VolatilityEstimator.push_mid``
        directly into the WS event path so the estimator samples at
        the venue's natural cadence (~7-8 mid changes/sec on OKX SUI)
        rather than the bot's quote-cycle rate (2/sec). Combined with
        the v1.2.24 dedup, this captures sub-cycle vol moves the
        cycle-rate path was missing.

        Registration is expected to be single-threaded at startup;
        no lock is taken. Firing happens on the WS thread; the fire
        path takes a snapshot of the list to tolerate any rare
        concurrent registration without crashing.
        """
        self._on_mid_change_callbacks.append(callback)

    def reset_inbound_tick_counters(self) -> None:
        """Bot tick start: per-tick public BBO apply count (incremented from WS thread)."""
        with self._lock:
            self.public_ws_events_applied_last_tick = 0

    def note_public_bbo_burst_max_for_tick(self) -> None:
        """End of bot tick: high-water burst of BBO applies seen this tick."""
        with self._lock:
            self.public_ws_max_burst_per_tick = max(
                int(self.public_ws_max_burst_per_tick),
                int(self.public_ws_events_applied_last_tick),
            )

    def mark_heartbeat(self) -> None:
        with self._lock:
            self.last_heartbeat = utc_now()

    def apply_market_snapshot(
        self,
        market,
        position,
        account,
        *,
        market_data_source: Optional[str] = None,
        storage: Optional[Storage] = None,
    ) -> None:
        # v1.5.315: gap-tracker timestamp from the bot's clock (SystemClock
        # in prod, ReplayClock in backtest) instead of raw perf_counter. The
        # gap tracker only deltas consecutive values against each other, so
        # prod is a same-family swap (deltas preserved) and backtest gaps now
        # reflect RECORDED cadence, not replay wall-speed. All three
        # gap-tracker feeders move to the clock together.
        mono_now = _clock.monotonic()
        wall_now = utc_now()
        with self._lock:
            self.market = market
            self.last_market_ts = market.ts_local if market else None
            if market_data_source:
                self.live_market_data_source = market_data_source
            # Shadow-position divergence detection (see
            # ``apply_account_position_only`` for the equivalent path
            # and rationale). REST is authoritative; this just counts
            # how off the shadow was.
            old_qty = float(self.position.position_qty)
            new_qty = float(position.position_qty)
            if abs(new_qty - old_qty) > 0.5:
                self.shadow_position_divergence_count += 1
                self.shadow_position_last_divergence_qty = new_qty - old_qty
                _state_logger.warning(
                    "shadow_position_divergence_via_market_snapshot "
                    "rest_qty=%.6f shadow_qty=%.6f delta=%.6f",
                    new_qty,
                    old_qty,
                    new_qty - old_qty,
                )
                # BUG-034 fix (v1.5.258): see apply_account_position_only
                # for the rationale. Trigger the REST fill catch-up so
                # the missed fill is ingested into the fills table /
                # session counters / markout, instead of being silently
                # absorbed by this position overwrite.
                self.private_ws_recovery_pending = True
                self.shadow_position_divergence_recovery_count += 1
            self.position = position
            self.account = account
            if market and (
                (market.mid_price is not None and market.mid_price > 0)
                or (
                    market.best_bid
                    and market.best_ask
                    and market.best_bid > 0
                    and market.best_ask > 0
                )
            ):
                self._book_snapshots.append(
                    TopOfBookRow(
                        ts=market.ts_local,
                        best_bid=market.best_bid,
                        best_ask=market.best_ask,
                        mid=market.mid_price,
                        bid_size=market.bid_size,
                        ask_size=market.ask_size,
                    )
                )
            recovery_state = self._market_data_metrics_unlocked()["market_data_recovery_state"]
            eligible = market_snapshot_eligible_for_gap_stats(market)
            sym = self.symbol
            sid = self.session_id
            settings_ref = self._settings
        if eligible:
            self.market_data_gap_tracker.note_successful_update(
                perf_now=mono_now,
                wall_now=wall_now,
                source=(market_data_source or "unknown"),
                symbol=sym,
                recovery_state=recovery_state,
                settings=settings_ref,
                storage=storage,
                session_id=sid,
            )

    def apply_market_book_only(
        self,
        market: BestBidAsk,
        *,
        market_data_source: Optional[str] = None,
        storage: Optional[Storage] = None,
    ) -> None:
        """Apply top-of-book from public WS only (does not touch position/account)."""
        # v1.5.315: ``perf_now`` (perf_counter) stays for the inbound-apply
        # timing diagnostic below — it pairs with ``ws_recv_mono`` (also
        # perf_counter, stamped in the live WS handler) and is skipped in
        # backtest (the replay BestBidAsk carries no inbound_public_timing).
        # ``mono_now`` (clock) drives the freshness book-age clock, the
        # book-apply→decision marker, and the gap tracker — all of which have
        # clock-based readers, so backtest sees recorded time, not wall-speed.
        perf_now = time.perf_counter()
        mono_now = _clock.monotonic()
        wall_now = utc_now()
        m = market
        if m.inbound_public_timing is not None:
            it = replace(m.inbound_public_timing, apply_mono=perf_now)
            m = replace(m, inbound_public_timing=it)
            derived = public_timing_derived_ms(it)
            self.note_public_inbound_apply(it, derived)
            # Public WS timing window (in-memory only): capture exchange cadence, one-way age, and local apply delay.
            if self.public_ws_timing is not None:
                recv_wall_ms = (
                    int(it.ws_recv_wall.timestamp() * 1000.0)
                    if it.ws_recv_wall is not None
                    else None
                )
                recv_mono_ns = (
                    int(float(it.ws_recv_mono) * 1_000_000_000.0)
                    if it.ws_recv_mono is not None
                    else None
                )
                self.public_ws_timing.ingest(
                    local_receive_wall_ms=recv_wall_ms,
                    local_receive_mono_ns=recv_mono_ns,
                    exchange_ts_ms=m.ts_exchange_ms,
                    local_apply_wall_ms=int(wall_now.timestamp() * 1000.0),
                    local_apply_mono_ns=int(float(perf_now) * 1_000_000_000.0),
                    seq=None,
                )
        with self._lock:
            # 1.2.25: bump BBO + mid-change session counters before
            # the assignment so we can compare prev → new mid.
            # 1.2.26: also capture (mid_changed, fire_value) so we
            # can fire ``_on_mid_change_callbacks`` AFTER releasing
            # the lock — listeners (e.g. VolatilityEstimator) must
            # never block the WS thread under the state lock.
            prev_mid = (
                self.market.mid_price if self.market is not None else None
            )
            new_mid = m.mid_price if m is not None else None
            self.bbo_event_count_session += 1
            mid_changed = prev_mid != new_mid
            if mid_changed:
                self.mid_change_count_session += 1
                # v1.5.244 — also feed the 5-min rolling rate.
                # Wall-clock timestamp (UTC seconds) — the rate
                # accessor below prunes by wall time. Robust to bot
                # restarts (timestamps from old sessions are pruned
                # within 5 min of restart).
                try:
                    import time as _time_mod
                    self.mid_change_recent_ts.append(_time_mod.time())
                except Exception:
                    pass
            self.market = m
            self.last_market_ts = m.ts_local if m else None
            src = market_data_source or "unknown"
            self.live_market_data_source = src
            if m and m.mid_price and m.mid_price > 0:
                self.public_ws_seen_first_bbo = True
            if m and (
                (m.mid_price is not None and m.mid_price > 0)
                or (
                    m.best_bid
                    and m.best_ask
                    and m.best_bid > 0
                    and m.best_ask > 0
                )
            ):
                self.public_last_bbo_receipt_monotonic = mono_now
                self.last_decision_book_apply_mono = mono_now
                self.public_ws_events_applied_last_tick = (
                    int(self.public_ws_events_applied_last_tick) + 1
                )
                self._book_snapshots.append(
                    TopOfBookRow(
                        ts=m.ts_local,
                        best_bid=m.best_bid,
                        best_ask=m.best_ask,
                        mid=m.mid_price,
                        bid_size=m.bid_size,
                        ask_size=m.ask_size,
                    )
                )
            # Codex 2026-05-09 HIGH-4: live re-mark of unrealized PnL
            # from the new public mid. Pre-fix, ``unrealized_pnl_usd``
            # only updated on REST account refresh (~8 s) or on a fill
            # (via ``record_fill``'s shadow re-mark). Between those,
            # the drawdown / session-loss / position-drawdown gates
            # ran on a stale unrealized — so a position could move
            # materially against the bot before the gates noticed.
            #
            # Re-mark equation:  (mid - avg_entry) * qty
            #
            # Conditions:
            #  - position non-zero (else unrealized is trivially 0)
            #  - avg_entry_price set (REST owns it; shadow-mark from
            #    flat or sign-flip leaves it None until REST repairs)
            #  - mid available + positive (sanity check on the BBO)
            #
            # We touch ONLY ``unrealized_pnl_usd``. ``mark_price`` is
            # the venue-side mark used elsewhere (e.g. liquidation
            # math) and stays REST-authoritative. Mid is a reasonable
            # proxy for live mark on perps for risk-gating purposes.
            if (
                m is not None
                and m.mid_price is not None
                and float(m.mid_price) > 0.0
                and abs(float(self.position.position_qty)) > 1e-12
                and self.position.avg_entry_price is not None
                and float(self.position.avg_entry_price) > 0.0
            ):
                live_unrealized = (
                    float(m.mid_price) - float(self.position.avg_entry_price)
                ) * float(self.position.position_qty)
                self.position = replace(
                    self.position,
                    unrealized_pnl_usd=live_unrealized,
                )
            recovery_state = self._market_data_metrics_unlocked()["market_data_recovery_state"]
            eligible = market_snapshot_eligible_for_gap_stats(m)
            sym = self.symbol
            sid = self.session_id
            settings_ref = self._settings
        if eligible:
            self.market_data_gap_tracker.note_successful_update(
                perf_now=mono_now,
                wall_now=wall_now,
                source=src,
                symbol=sym,
                recovery_state=recovery_state,
                settings=settings_ref,
                storage=storage,
                session_id=sid,
            )
        # 1.2.26 (todo-010 Phase 2): fire mid-change listeners
        # OUTSIDE the lock. Bot's VolatilityEstimator subscribes
        # via ``add_mid_change_listener`` so vol sampling tracks
        # actual price ticks at the venue's natural cadence
        # instead of the quote-cycle rate. Listener errors are
        # logged + swallowed; we never let a misbehaving
        # subscriber take down the public-WS handler. ``new_mid``
        # is the post-update value captured under the lock above.
        if mid_changed and new_mid is not None and new_mid > 0:
            # Snapshot the list so concurrent registration (rare,
            # generally only at bot startup) can't trip us.
            for cb in list(self._on_mid_change_callbacks):
                try:
                    cb(new_mid)
                except Exception:
                    _state_logger.exception(
                        "mid_change_listener_failed callback=%s", cb
                    )

    def note_book_heartbeat(
        self,
        market: BestBidAsk,
        *,
        market_data_source: str = "public_ws",
        storage: Optional[Storage] = None,
    ) -> None:
        """Liveness heartbeat from a secondary book channel (BUG-028).

        The live OKX feed subscribes to a SINGLE *primary* book channel —
        ``bbo-tbt`` (event-driven L1, fires only when the touch changes).
        In genuinely quiet markets the touch can sit unchanged for several
        seconds, so ``bbo-tbt`` goes silent and the freshness gate's
        book-age clock (``public_last_bbo_receipt_monotonic``) grows past
        ``QUOTE_HOLD_MAX_BOOK_AGE_MS`` → ``freshness_drift_hold`` fires and
        suppresses quoting even though the market is perfectly healthy
        (the false positive documented in ``issues/bug-028``).

        To keep a liveness signal flowing we ALSO subscribe to ``books5``
        on the same WS connection (100 ms throttled top-5 snapshot, which
        pushes on ANY top-5 change — far more frequently than top-1-only
        ``bbo-tbt``). Each ``books5`` frame is routed here as a heartbeat.

        This method is deliberately NARROWER than
        :meth:`apply_market_book_only`:

        * It ALWAYS refreshes the book-age clock and feeds the
          market-data gap tracker — *that* is the fix: a received book
          frame (even one that does not move the touch) proves the feed
          is alive, so the freshness gate's age + p95 inputs stay healthy.
        * It drives the touch (``self.market``) ONLY when the incoming
          ``books5`` frame is at least as fresh (by ``ts_exchange_ms``) as
          the currently-held touch. This is the frozen-touch safety net:
          if ``bbo-tbt`` dies but ``books5`` keeps flowing, the touch must
          not freeze while the gate believes the feed is healthy. The
          ts-monotonic guard stops a lagging ``books5`` snapshot from
          clobbering a fresher ``bbo-tbt`` touch in active markets.
        * It does NOT route through ``main._on_public_bbo`` and therefore
          does NOT feed the fingerprint stall-reconnect detector
          (``market_refresh_note_success``), the OFI accumulator, or
          ``wake_quote_loop``. ``books5``'s fingerprint / size signal
          would pollute those; the quote loop's own 2 Hz cadence picks up
          a heartbeat-driven touch within one tick (a quiet market is, by
          definition, not latency-critical).
        * It does NOT set the WS-layer ``_first_book_data_event`` (that is
          gated to the PRIMARY channel in ``okx_public_ws`` so the bug-019
          silent-subscribe fallback timer still fires if ``bbo-tbt`` never
          streams — a ``books5`` heartbeat must not mask that).

        Stall detection is preserved (Rule 0c — adaptive, not permanent):
        a genuine feed stall silences BOTH ``bbo-tbt`` AND ``books5`` (and
        ``trades``), so the gap ring starves and the age clock grows → the
        gate fires for real. Entry/exit stay driven by live inter-event
        cadence, never a fixed timer.
        """
        # v1.5.315: see apply_market_book_only — ``perf_now`` stays for the
        # prod-only inbound-apply timing; ``mono_now`` (clock) drives the
        # freshness age clock + gap tracker for backtest fidelity.
        perf_now = time.perf_counter()
        mono_now = _clock.monotonic()
        wall_now = utc_now()
        m = market
        # An invalid frame proves nothing about feed liveness — bail
        # before touching the age clock or the gap ring (mirrors the
        # eligibility gate in ``apply_market_book_only``).
        if not market_snapshot_eligible_for_gap_stats(m):
            return
        new_mid_for_cb: Optional[float] = None
        with self._lock:
            held = self.market
            held_ts = held.ts_exchange_ms if held is not None else None
            incoming_ts = m.ts_exchange_ms
            # Adopt when we have no touch yet, when either side lacks an
            # exchange ts (can't compare → bias toward the fresher data),
            # or when the incoming frame is at least as fresh as the held
            # touch. In active markets ``books5`` usually lags the just-
            # applied ``bbo-tbt`` touch → no adoption → touch untouched.
            adopt = (
                held is None
                or held_ts is None
                or incoming_ts is None
                or int(incoming_ts) >= int(held_ts)
            )
            if adopt:
                prev_mid = held.mid_price if held is not None else None
                touch_changed = (
                    held is None
                    or held.best_bid != m.best_bid
                    or held.best_ask != m.best_ask
                )
                mid_changed = prev_mid != m.mid_price
                # Keep the stored touch's apply timing sane WITHOUT feeding
                # the bbo-tbt timing tracker (a books5 frame must not skew
                # the primary channel's latency diagnostic).
                if m.inbound_public_timing is not None:
                    m = replace(
                        m,
                        inbound_public_timing=replace(
                            m.inbound_public_timing, apply_mono=perf_now
                        ),
                    )
                self.market = m
                self.last_market_ts = m.ts_local
                # Honest: we have observed a valid BBO. Setting this is
                # safe w.r.t. bug-019 — the fallback timer keys off the
                # WS-layer ``_first_book_data_event``, which this path
                # never sets.
                self.public_ws_seen_first_bbo = True
                if touch_changed:
                    # Counts as a top-of-book apply only when the touch
                    # actually moved — never on a same-price re-confirm,
                    # so ``_book_snapshots`` (maxlen=400, feeds fill-time
                    # book matching) is not flushed by 100 ms heartbeats.
                    self.bbo_event_count_session += 1
                    self._book_snapshots.append(
                        TopOfBookRow(
                            ts=m.ts_local,
                            best_bid=m.best_bid,
                            best_ask=m.best_ask,
                            mid=m.mid_price,
                            bid_size=m.bid_size,
                            ask_size=m.ask_size,
                        )
                    )
                    # Live re-mark of unrealized PnL from the new mid, so
                    # the risk gates see a books5-driven move when bbo-tbt
                    # is silent (mirrors apply_market_book_only HIGH-4).
                    if (
                        m.mid_price is not None
                        and float(m.mid_price) > 0.0
                        and abs(float(self.position.position_qty)) > 1e-12
                        and self.position.avg_entry_price is not None
                        and float(self.position.avg_entry_price) > 0.0
                    ):
                        live_unrealized = (
                            float(m.mid_price)
                            - float(self.position.avg_entry_price)
                        ) * float(self.position.position_qty)
                        self.position = replace(
                            self.position,
                            unrealized_pnl_usd=live_unrealized,
                        )
                    if mid_changed:
                        self.mid_change_count_session += 1
                        # Feed the 5-min mid-change rate so it stays
                        # honest if books5 is carrying touches during a
                        # partial bbo-tbt outage.
                        try:
                            self.mid_change_recent_ts.append(time.time())
                        except Exception:
                            pass
                        if m.mid_price is not None and m.mid_price > 0:
                            new_mid_for_cb = m.mid_price
            # THE FIX (age component): a received book frame — even one
            # that does NOT move the touch — proves the public feed is
            # alive. Refresh the age clock unconditionally so the freshness
            # gate's age input does not grow during quiet (touch-stable)
            # periods. ``_freshness_eligibility`` treats the age path as
            # non-overridable, so this refresh is load-bearing.
            self.public_last_bbo_receipt_monotonic = mono_now
            recovery_state = self._market_data_metrics_unlocked()[
                "market_data_recovery_state"
            ]
            sym = self.symbol
            sid = self.session_id
            settings_ref = self._settings
        # THE FIX (p95 component): feed the shared gap ring so
        # ``recent_gap_stats_for_gate`` sees a healthy inter-arrival
        # cadence from the combined bbo-tbt + books5 stream. ``source`` is
        # "public_ws" (same connection / venue as the primary) so the
        # gap-stats ``source_type`` contract stays stable. ``storage`` is
        # None on the live path (WS handler has no storage handle) — the
        # gate reads the in-memory ring, not the DB, so this is fine.
        self.market_data_gap_tracker.note_successful_update(
            perf_now=mono_now,
            wall_now=wall_now,
            source=market_data_source,
            symbol=sym,
            recovery_state=recovery_state,
            settings=settings_ref,
            storage=storage,
            session_id=sid,
        )
        if new_mid_for_cb is not None and new_mid_for_cb > 0:
            for cb in list(self._on_mid_change_callbacks):
                try:
                    cb(new_mid_for_cb)
                except Exception:
                    _state_logger.exception(
                        "mid_change_listener_failed callback=%s", cb
                    )

    def record_mid_price_sample(self, mid: float, mono: float) -> None:
        """Append mid with local monotonic time for short-horizon drift/jump guards."""
        with self._lock:
            if not (
                isinstance(mid, (int, float))
                and math.isfinite(float(mid))
                and float(mid) > 0
            ):
                return
            self._mid_price_samples.append((mono, float(mid)))
            max_age_s = float(self._settings.quote_mid_history_max_age_ms) / 1000.0
            while self._mid_price_samples and (mono - self._mid_price_samples[0][0]) > max_age_s:
                self._mid_price_samples.popleft()
            # Long-window companion deque for the multi-minute drift
            # gate. Same input, different retention (``drift_long_window_seconds``).
            self._mid_price_samples_long.append((mono, float(mid)))
            long_max_age_s = float(
                getattr(self._settings, "drift_long_window_seconds", 300.0)
            )
            while (
                self._mid_price_samples_long
                and (mono - self._mid_price_samples_long[0][0]) > long_max_age_s
            ):
                self._mid_price_samples_long.popleft()

    def mid_price_samples_snapshot(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._mid_price_samples)

    def mid_price_samples_long_snapshot(self) -> list[tuple[float, float]]:
        """Snapshot of the multi-minute mid-price ring buffer used by
        the long-window drift gate (BUGS/bug-002.md)."""
        with self._lock:
            return list(self._mid_price_samples_long)

    def seconds_since_last_public_bbo(self, now_mono: float) -> Optional[float]:
        """Seconds since last public BBO apply (monotonic), for receipt-time freshness."""
        with self._lock:
            t = self.public_last_bbo_receipt_monotonic
        if t is None:
            return None
        return max(0.0, float(now_mono - t))

    def bump_quote_eligibility_counters(
        self, elig: QuoteEligibility, tags: tuple[str, ...]
    ) -> None:
        with self._lock:
            if elig == QuoteEligibility.HOLD_ALL:
                self.quote_elig_hold_all_count += 1
            if elig == QuoteEligibility.QUOTE_BUY_ONLY:
                self.quote_elig_buy_only_count += 1
            if elig == QuoteEligibility.QUOTE_SELL_ONLY:
                self.quote_elig_sell_only_count += 1
            for tg in tags:
                if tg == "hold_due_stale":
                    self.quote_elig_hold_due_stale_count += 1
                elif tg == "hold_due_jump":
                    self.quote_elig_hold_due_jump_count += 1
                elif tg == "one_sided_due_drift":
                    self.quote_elig_one_sided_due_drift_count += 1
                elif tg == "one_sided_due_freshness":
                    self.quote_elig_one_sided_due_freshness_count += 1

    def note_quote_eligibility_resume(self, last_e: QuoteEligibility, new_e: QuoteEligibility) -> None:
        """Count transitions back to quoting both sides after a restrictive episode."""
        with self._lock:
            if last_e != QuoteEligibility.QUOTE_BOTH and new_e == QuoteEligibility.QUOTE_BOTH:
                self.quote_elig_resume_count += 1

    def set_quote_eligibility_snapshot_dict(self, d: dict[str, Any]) -> None:
        with self._lock:
            self.quote_eligibility_snapshot_dict = dict(d)

    def set_observability_gate_flags(self, d: dict[str, Any]) -> None:
        """Bot calls this once per tick to push gate-flag state for
        the exposure-bar emitter. Same lock-free read pattern as
        ``quote_eligibility_snapshot_dict``. See
        ``ExposureBarEmitter._capture_bar`` for the consumer.

        2026-05-13 regime-observability bugfix.
        """
        with self._lock:
            self.observability_gate_flags = dict(d)

    def apply_account_position_only(
        self,
        position: PositionSnapshot,
        account: Optional[AccountSnapshot],
    ) -> None:
        """REST account/position refresh without changing the live order book.

        REST is the authoritative truth; this overwrites the shadow
        position kept up-to-date by ``record_fill``. We compare the
        two to count divergences (any drift > 0.5 lot is recorded as
        a divergence event so the operator can see how well the
        shadow is tracking).
        """
        with self._lock:
            old_qty = float(self.position.position_qty)
            new_qty = float(position.position_qty)
            divergence = abs(new_qty - old_qty)
            # 0.5 lot is the rough sensitivity threshold -- below this,
            # diff is just rounding/timing noise (e.g. fill landed
            # between REST start and end). Above, something is off:
            # missed fill event, double-applied delta, or a venue-side
            # adjustment we don't see (funding, liquidation, etc.).
            if divergence > 0.5:
                self.shadow_position_divergence_count += 1
                self.shadow_position_last_divergence_qty = new_qty - old_qty
                _state_logger.warning(
                    "shadow_position_divergence rest_qty=%.6f shadow_qty=%.6f delta=%.6f total_divergences=%d",
                    new_qty,
                    old_qty,
                    new_qty - old_qty,
                    self.shadow_position_divergence_count,
                )
                # BUG-034 fix (v1.5.258): a divergence means the WS
                # missed at least one fill. The shadow was correct
                # for everything it saw, so the gap == the fill(s)
                # we never received via WS. Trigger the same REST
                # fill catch-up path the queue-overflow recovery
                # uses: on the next tick, ``should_ingest_fills_
                # via_rest()`` will return True, ``refresh_account_
                # only`` will call ``fetch_recent_fills_raw``, and
                # the missed fill will be dedup-aware ingested via
                # ``ingest_hl_fill_raw(shadow_update_position=
                # False)``. Without this, the fill never lands in
                # ``fills.jsonl`` / ``recent_fills`` / session
                # counters / markout sampling — silently distorting
                # every fill-derived metric.
                self.private_ws_recovery_pending = True
                self.shadow_position_divergence_recovery_count += 1
            self.position = position
            self.account = account

    def resolve_fill_book_reference(
        self, ts_fill: datetime
    ) -> tuple[
        Optional[float],  # mid
        Optional[float],  # best_bid (price)
        Optional[float],  # best_ask (price)
        Optional[float],  # bid_size (top-of-book quantity)
        Optional[float],  # ask_size (top-of-book quantity)
        str,              # book_reference_quality
        str,              # book_snapshot_quality
    ]:
        """Return top-of-book context (prices + sizes) at fill time.

        Sizes are optional because some venue adapters publish only
        prices on partial / book-recovery events. ``None`` is
        forward-compatible — the postmortem queue-imbalance analysis
        treats missing sizes as "skip this fill from the size-axis
        breakdown" rather than failing.
        """
        with self._lock:
            rows = list(self._book_snapshots)
            skew_ms = self._settings.book_reference_max_skew_ms
        max_skew = timedelta(milliseconds=skew_ms)
        row, ref_q = match_top_of_book(rows, ts_fill, max_skew)
        if row is None:
            return None, None, None, None, None, ref_q, "unknown"
        b, a, m = row.best_bid, row.best_ask, row.mid
        return m, b, a, row.bid_size, row.ask_size, ref_q, snapshot_fullness(b, a, m)

    def attach_fill_observer(self, cb: Callable[["Fill"], None]) -> None:
        """Register a callable invoked after each session-scoped fill.

        Observers fire OUTSIDE the state lock so they can do their own
        synchronisation safely (the Telegram notifier enqueues to a
        bounded queue — non-blocking — so the trading loop never waits
        on it). Exceptions from observers are swallowed and logged.

        Idempotency: it's the caller's responsibility to attach exactly
        once; this method does not deduplicate.
        """
        self._fill_observers.append(cb)

    def record_fill(
        self,
        f: Fill,
        *,
        session_scoped: bool = True,
        shadow_update_position: bool = True,
    ) -> bool:
        """
        Register a new fill_id (idempotent). When ``session_scoped`` is False, the fill is only
        deduped — not added to ``recent_fills``, trade-activity, or fill-process latency (replay path).

        FILL-DERIVED SHADOW POSITION
        ============================
        Session-scoped fills update ``self.position.position_qty`` in
        real time (delta = +qty for BUY, -qty for SELL). The previous
        behaviour (only REST refresh updates position) left the
        position state stale by up to ``ACCOUNT_REST_MIN_INTERVAL_SECONDS``
        between refreshes, which on 2026-05-06 SUI runaway let the
        soft-flatten worker read pos=-21 across 7+ ticks of repeated
        BUY 21 placements while the venue had already filled them.

        REST refresh remains AUTHORITATIVE (in ``apply_market_snapshot``
        / ``apply_account_position_only``); the shadow update closes
        the staleness gap between refreshes. Divergence between shadow
        and REST is recorded in ``shadow_position_divergence_count``
        for diagnostics.

        ``shadow_update_position`` decouples "this is a session fill
        I should track for metrics" from "this is a delta I haven't
        yet applied to position." REST catch-up paths
        (``refresh_account_only`` after ``apply_account_position_only``)
        must pass ``shadow_update_position=False`` because the position
        snapshot they just installed already reflects every fill — a
        fresh shadow-add would double-count. Bug discovered 2026-05-08
        in the Codex review (snapshot 260507140841 era).
        """
        fire_observers = False
        with self._lock:
            if f.fill_id in self._seen_fill_ids:
                return False
            self._seen_fill_ids.add(f.fill_id)
            if session_scoped:
                self.recent_fills.appendleft(f)
                self.session_fill_count += 1
                prev = int(self.session_fill_count_by_side.get(f.side, 0))
                self.session_fill_count_by_side[f.side] = prev + 1
                # todo-011: arm the post-fill replace cooldown timer for
                # the side that just got filled. ``time.monotonic`` × 1000
                # gives ms resolution and is robust to wall-clock jumps.
                # Read by the quote-cycle when computing
                # ``post_fill_cooldown_{bid,ask}_remaining_ms``; see
                # ``app/quoting.compute_quote_decision``.
                _now_mono_sec = _clock.monotonic()
                _now_mono_ms = _now_mono_sec * 1000.0
                # v1.5.197 — stamp last-fill monotonic time for idle-clear
                # predicates on defensive gates. Side-agnostic; the
                # per-side counters above are separate (used for the
                # post-fill cooldown which IS side-aware).
                self.last_fill_at_mono = _now_mono_sec
                if f.side == Side.BUY:
                    self.last_fill_monotonic_ms_buy = _now_mono_ms
                elif f.side == Side.SELL:
                    self.last_fill_monotonic_ms_sell = _now_mono_ms
                # 2026-05-12 codex-#3: note the fill for the burst
                # detector. Reads on every cycle to compute the size
                # shrink. When the feature is disabled
                # (FILL_BURST_THRESHOLD=0) ``note_fill`` returns early.
                try:
                    self.fill_burst_detector.note_fill(_now_mono_sec)
                except Exception:
                    logger.exception("fill_burst_detector_note_fill_failed")
                # Cumulative traded notional — single addition per
                # fill, no rollover or windowing. abs() because side
                # sign should not affect "how much have we traded".
                try:
                    self.session_traded_notional_usd += abs(
                        float(getattr(f, "notional", 0.0) or 0.0)
                    )
                except (TypeError, ValueError):
                    pass
                self.quote_quality.note_fill_spread_capture_usd(f)
                now = _clock.time_seconds()
                self._trade_activity_ts.append(now)
                cutoff = now - 60.0
                while self._trade_activity_ts and self._trade_activity_ts[0] < cutoff:
                    self._trade_activity_ts.popleft()
                # Place-to-fill ratio tracker. Tiny non-locking call
                # (PlaceToFillRatioTracker has its own lock); never
                # raises, so doesn't need a try/except around it.
                self.place_to_fill_ratio_tracker.note_fill()
                try:
                    lag_ms = (utc_now() - f.ts_fill).total_seconds() * 1000.0
                    self.last_latency_fill_to_process_ms = max(0.0, lag_ms)
                except (TypeError, ValueError, OverflowError):
                    pass
                # SHADOW-POSITION update. Apply the signed delta of
                # this fill to position_qty so risk/quoting/flatten
                # logic sees a near-real-time view between REST
                # refreshes. Notional is recomputed from the new qty
                # using the latest known price reference (mark > entry
                # > fill price). REST refresh still overwrites this
                # snapshot when it lands -- it's the truth, this is
                # just the closing-the-gap layer.
                #
                # Gated separately on ``shadow_update_position`` so
                # REST catch-up paths can call ``record_fill`` for
                # session-metrics bookkeeping WITHOUT double-applying
                # the position delta on top of an already-fresh
                # ``apply_account_position_only`` snapshot.
                if shadow_update_position:
                    try:
                        delta = float(f.size) if f.side == Side.BUY else -float(f.size)
                        prev_qty = float(self.position.position_qty)
                        new_qty = prev_qty + delta
                        prev_avg = self.position.avg_entry_price

                        # SHADOW AVG_ENTRY_PRICE seeding (Codex MED-7
                        # follow-up, 1.1.134). Pre-fix, when the bot
                        # transitioned from flat → position, the shadow
                        # update preserved ``avg_entry_price=None`` until
                        # REST refresh. That left the position-drawdown
                        # gate and session-loss logic with a non-zero
                        # qty but no entry reference for the first few
                        # seconds — exactly the window where a fast
                        # adverse move can do the most damage.
                        #
                        # Cases:
                        #   1. Was flat, now non-flat → seed avg_entry =
                        #      f.price (the only price we have).
                        #   2. Sign flipped (prev * new < 0) → the close
                        #      leg is fully realised; the open leg's
                        #      entry IS this fill's price.
                        #   3. Adding to existing position (same sign) →
                        #      keep the prior avg_entry. REST will
                        #      reconcile on the next refresh; an
                        #      incremental average is server-side
                        #      bookkeeping (closed_pnl etc.) that the
                        #      bot doesn't try to mirror perfectly.
                        new_avg_entry = prev_avg
                        if abs(new_qty) < 1e-12:
                            # Closed flat — clear stale entry too.
                            new_avg_entry = None
                        elif abs(prev_qty) < 1e-12:
                            # Case 1: flat → opened.
                            new_avg_entry = float(f.price)
                        elif prev_qty * new_qty < 0:
                            # Case 2: sign flip — re-seed from this fill.
                            new_avg_entry = float(f.price)

                        ref_price = (
                            float(self.position.mark_price)
                            if self.position.mark_price is not None
                            and self.position.mark_price > 0
                            else (
                                float(new_avg_entry)
                                if new_avg_entry is not None and new_avg_entry > 0
                                else float(f.price)
                            )
                        )
                        new_notional = abs(new_qty) * ref_price if ref_price > 0 else 0.0
                        # SHADOW UNREALIZED PNL recompute. Without this,
                        # ``record_fill`` updates ``position_qty`` /
                        # ``position_notional`` but leaves
                        # ``unrealized_pnl_usd`` stale, so drawdown +
                        # session-loss gates run against the unrealized
                        # number from the pre-fill snapshot until the
                        # next REST refresh — masking risk by up to the
                        # account-refresh interval under fast fill
                        # bursts. Codex review 2026-05-07 / 1.1.37 fix
                        # (HIGH-3). Now uses the freshly-seeded
                        # ``new_avg_entry`` (Codex MED-7) so the very
                        # first fill from flat carries a non-stale
                        # unrealized value.
                        new_unrealized = self.position.unrealized_pnl_usd
                        if abs(new_qty) < 1e-12:
                            new_unrealized = 0.0
                        elif (
                            self.position.mark_price is not None
                            and self.position.mark_price > 0
                            and new_avg_entry is not None
                            and new_avg_entry > 0
                        ):
                            new_unrealized = (
                                float(self.position.mark_price)
                                - float(new_avg_entry)
                            ) * new_qty
                        # Replace via dataclass field assignment (PositionSnapshot is a frozen-ish dataclass; assignments work since it isn't frozen=True).
                        self.position = replace(
                            self.position,
                            position_qty=new_qty,
                            position_notional=new_notional,
                            avg_entry_price=new_avg_entry,
                            unrealized_pnl_usd=new_unrealized,
                        )
                        self.shadow_position_apply_count += 1
                        # v1.4.113 Phase 1D — post-reduction cooldown
                        # trigger. Fires inside the shadow-update block
                        # because that's where ``prev_qty`` / ``new_qty``
                        # are computed; identifies whether THIS fill
                        # reduced |position| and arms the eligibility
                        # clamp consumed by ``_apply_regime_gates``.
                        # Best-effort: any exception is swallowed in the
                        # outer try/except so the fill record path is
                        # never interrupted.
                        self._note_inventory_reduction_for_cooldown(
                            now_mono=_now_mono_sec,
                            fill_side=f.side,
                            prev_qty=prev_qty,
                            new_qty=new_qty,
                        )
                    except Exception:
                        # Never let a shadow-update bug interrupt the
                        # fill record path. Worst case: position state
                        # stays as-is (legacy behaviour) until the next
                        # REST refresh.
                        _state_logger.exception("shadow_position_update_failed")
                fire_observers = True

        if fire_observers and self._fill_observers:
            # Snapshot the observer list under no lock — appends are at
            # startup only, so a list copy here is safe and avoids
            # holding state._lock across user callbacks.
            for cb in list(self._fill_observers):
                try:
                    cb(f)
                except Exception:  # noqa: BLE001
                    _state_logger.exception("fill_observer_failed")
        return True

    def _note_inventory_reduction_for_cooldown(
        self,
        *,
        now_mono: float,
        fill_side: "Side",
        prev_qty: float,
        new_qty: float,
    ) -> None:
        """v1.4.113 Phase 1D — arm the post-reduction cooldown.

        Called from inside ``record_fill``'s shadow-update branch on
        every session-scoped fill. Determines whether THIS fill
        reduced ``|position_qty|`` and, if so, stamps the cooldown
        timestamp + the eligibility side to suppress during the
        cooldown window.

        Suppressed side semantics:

        * ``new_qty > 0`` (post-fill LONG) → suppress BUY (adding to
          LONG) → ``QuoteEligibility.QUOTE_SELL_ONLY``.
        * ``new_qty < 0`` (post-fill SHORT) → suppress SELL (adding
          to SHORT) → ``QuoteEligibility.QUOTE_BUY_ONLY``.
        * ``new_qty == 0`` (closed to flat) → suppress the side
          OPPOSITE the fill direction. Prevents the 06:50:05 fast-
          flip pattern: SELL-to-flat doesn't get an immediate BUY-
          back-to-long, and vice versa.

        Caller holds ``self._lock``. No I/O. Best-effort: callers
        wrap in try/except so a bug here can't interrupt the fill
        path.

        Feature is OFF when ``POST_REDUCTION_COOLDOWN_SECONDS <= 0``
        (default). The eligibility clamp in ``_apply_regime_gates``
        also gates on ``regime_controller.mode != NORMAL`` — NORMAL
        mode keeps round-trip rebate capture intact regardless of
        whether the timestamp is fresh.
        """
        cd_seconds = float(
            getattr(self._settings, "post_reduction_cooldown_seconds", 0.0)
            or 0.0
        )
        if cd_seconds <= 0.0:
            return
        if not (
            math.isfinite(prev_qty)
            and math.isfinite(new_qty)
        ):
            return
        # Only arm on actual reductions. 1e-9 tolerance guards against
        # floating-point noise; real fills move by ≥ size_step which
        # is orders of magnitude larger.
        if abs(new_qty) >= abs(prev_qty) - 1e-9:
            return
        # Local import to avoid module-load circular: state ← enums
        # is fine, but we delay the Side comparison import to keep
        # the class definition cleaner.
        from app.enums import QuoteEligibility
        self.last_inventory_reduction_at_mono = float(now_mono)
        if new_qty > 1e-9:
            self.last_inventory_reduction_suppressed_side = (
                QuoteEligibility.QUOTE_SELL_ONLY
            )
        elif new_qty < -1e-9:
            self.last_inventory_reduction_suppressed_side = (
                QuoteEligibility.QUOTE_BUY_ONLY
            )
        else:
            # Flat. Suppress side opposite the fill direction —
            # prevents fast-flip re-entry on the side we just exited.
            self.last_inventory_reduction_suppressed_side = (
                QuoteEligibility.QUOTE_SELL_ONLY
                if fill_side == Side.SELL
                else QuoteEligibility.QUOTE_BUY_ONLY
            )
        # v1.5.191 BUG-032 — ``post_reduction_cooldown_fire_count`` is
        # NO LONGER bumped here. Pre-v1.5.191 the counter incremented
        # on every reducing fill regardless of regime — i.e., it
        # counted "potential arming events", not actual gate engagements.
        # The ``cleared_via_*`` counters only fire inside the regime-
        # gated cooldown block in app/bot.py (DEFENSIVE/SHOCK only), so
        # NORMAL/CAUTIOUS-only sessions showed ``fire_count: N,
        # cleared_via_X: 0`` for any non-trivial N — silently breaking
        # the AC's position-favorable-exit metric for this gate.
        # The increment moved to the cooldown-engagement edge in
        # ``Bot._apply_eligibility_engine`` so ``fire_count`` now means
        # actual engagements that match the ``cleared_via_*`` semantics.
        # See issues/bug-032.md for the full diagnosis.

    def trades_last_minute(self) -> int:
        with self._lock:
            return self._trade_count_last_60s_unlocked()

    def set_quote_decision_markers(
        self,
        market_ts_local: Optional[datetime],
        *,
        effective_book_age_at_decision_ms: Optional[float] = None,
        book_ts_exchange_ms: Optional[int] = None,
        book_apply_to_decision_ms: Optional[float] = None,
        public_ws_queue_wait_ms_latest: Optional[float] = None,
        public_ws_receive_to_apply_ms_latest: Optional[float] = None,
        decision_market_data_regime: Optional[str] = None,
    ) -> None:
        """
        Call after quote decision; sets perf counter for decision→first-place latency.

        `last_book_apply_to_decision_ms` is the preferred local latency metric
        (pure monotonic delta). `last_exchange_ts_to_decision_ms` is kept only
        as exchange-age/mixed-clock context.
        """
        # v1.5.315: ``now_perf`` (perf_counter) stays for
        # ``quote_decision_perf_counter`` — it pairs with the outbound
        # transport_send_perf / ack_t / _first_enqueue_perf stamps
        # (perf_counter) read in execution.py. ``now_mono`` (clock) drives the
        # book-apply→decision delta, pairing with the now clock-based
        # ``last_decision_book_apply_mono`` writer.
        now_perf = time.perf_counter()
        now_mono = _clock.monotonic()
        with self._lock:
            self.quote_decision_perf_counter = now_perf
            tap = self.last_decision_book_apply_mono
            if tap is not None:
                self.last_book_apply_to_decision_ms = max(0.0, (now_mono - tap) * 1000.0)
            else:
                self.last_book_apply_to_decision_ms = book_apply_to_decision_ms
            self.last_effective_book_age_at_decision_ms = effective_book_age_at_decision_ms
            self.last_decision_book_ts_exchange_ms = book_ts_exchange_ms
            self.last_decision_public_ws_queue_wait_ms = public_ws_queue_wait_ms_latest
            self.last_decision_public_ws_receive_to_apply_ms = public_ws_receive_to_apply_ms_latest
            self.last_decision_market_data_regime = decision_market_data_regime
            if market_ts_local is not None:
                self.last_exchange_ts_to_decision_ms = max(
                    0.0,
                    (utc_now() - market_ts_local).total_seconds() * 1000.0,
                )
            else:
                self.last_exchange_ts_to_decision_ms = None

    def note_first_place_latency(self) -> None:
        """First successful ACK per tick clears the decision perf marker."""
        with self._lock:
            t0 = self.quote_decision_perf_counter
            if t0 is not None:
                self.last_latency_decision_to_first_place_ms = (
                    time.perf_counter() - t0
                ) * 1000.0
                self.quote_decision_perf_counter = None

    def _trade_count_last_60s_unlocked(self) -> int:
        now = _clock.time_seconds()
        cutoff = now - 60.0
        while self._trade_activity_ts and self._trade_activity_ts[0] < cutoff:
            self._trade_activity_ts.popleft()
        return len(self._trade_activity_ts)

    def _private_ws_seconds_since_last_message_unlocked(self) -> Optional[float]:
        ts = self.private_ws_last_message_wall_ts
        if ts is None:
            return None
        return max(0.0, (utc_now() - ts).total_seconds())

    def _ws_lateness_percentile_unlocked(
        self, q: float
    ) -> Optional[float]:
        """v1.4.96: percentile of the http_acked_no_ws lateness
        reservoir. ``q`` in [0,1]. Returns None when empty.

        Pre-v1.4.96 this read from gone_on_exchange_ws_lateness_samples;
        the field was renamed when the ws_terminal_late signature
        moved out of the gone_on_exchange tree into its own tier-2
        warning category (``http_acked_no_ws``).
        """
        samples = list(self.http_acked_no_ws_lateness_samples)
        if not samples:
            return None
        samples.sort()
        n = len(samples)
        if n == 1:
            return float(samples[0])
        idx = max(0, min(n - 1, int(round(q * (n - 1)))))
        return float(samples[idx])

    def _binance_ws_status_dict_unlocked(self) -> dict[str, Any]:
        """Binance cross-venue reference fields for /state/current, /status, /health.

        Assumes caller already holds ``self._lock`` (called from
        ``snapshot_dict`` and ``status_flags_dict`` which both acquire it).

        All fields are always present so the shape is stable regardless of
        whether ``BINANCE_WS_ENABLED`` is true or the first message has
        arrived yet — downstream analysis (``stats_snapshot.py``
        state_current.json, ``explain_moment.py``) can rely on the keys.
        """
        last_msg_ts = self.binance_last_message_wall_ts
        seconds_since = (
            max(0.0, (utc_now() - last_msg_ts).total_seconds())
            if last_msg_ts is not None
            else None
        )
        last_conn_ts = self.binance_ws_last_connect_ts
        return {
            "binance_ws_connected": self.binance_ws_connected,
            "binance_ws_reconnect_count": self.binance_ws_reconnect_count,
            "binance_ws_last_connect_ts": (
                last_conn_ts.isoformat() if last_conn_ts is not None else None
            ),
            "binance_ws_last_message_ts": (
                last_msg_ts.isoformat() if last_msg_ts is not None else None
            ),
            "binance_ws_seconds_since_last_message": seconds_since,
            "binance_best_bid": self.binance_best_bid,
            "binance_best_ask": self.binance_best_ask,
            "binance_bid_size": self.binance_bid_size,
            "binance_ask_size": self.binance_ask_size,
            "binance_mid": self.binance_mid,
            "binance_basis_ewma": self.binance_basis_ewma,
        }

    def market_refresh_note_failure(self, latency_ms: Optional[float]) -> int:
        """Account REST refresh failed before snapshot apply (book comes from public WS). Returns streak."""
        with self._lock:
            self.market_data_failed_refresh_streak += 1
            if latency_ms is not None:
                self.market_data_last_refresh_latency_ms = float(latency_ms)
            return self.market_data_failed_refresh_streak

    def note_account_only_refresh_success(self, latency_ms: float) -> None:
        """Successful account/position REST refresh (independent of public BBO).

        Stamps BOTH the attempt clock (so the throttle treats this as
        the most recent attempt) AND the success clock (so the
        stale-account risk gate sees this as the most recent good
        data point). The two clocks were split in the 2026-05-16
        Codex #1 fix; see the field docstrings in ``__init__``.
        """
        with self._lock:
            self.market_data_failed_refresh_streak = 0
            self.market_data_last_refresh_latency_ms = float(latency_ms)
            now = _clock.monotonic()
            self.account_rest_last_monotonic = now
            self.account_rest_last_success_monotonic = now

    def note_account_only_refresh_attempt(self) -> None:
        """Mark "we just attempted a REST refresh" -- success or failure.

        Used to engage the ``account_rest_min_interval_seconds`` throttle
        on the FAILURE path too. Without this, an HTTP 429 (or any
        exception) leaves ``account_rest_last_monotonic`` un-advanced,
        and the bot's next tick runs another refresh immediately --
        producing the tight-loop retry that 429'd us repeatedly during
        the 2026-05-05 OKX bring-up (bug-018 follow-up).

        Does NOT stamp ``account_rest_last_success_monotonic`` — the
        stale-account risk gate measures real success, not attempts.
        This is the 2026-05-16 Codex #1 fix: previously this method
        wrote to a single ``last_monotonic`` field that both throttle
        and stale-gate read, so repeated failures kept the gate
        asleep while account truth was hours stale. The fields are
        now split (see ``__init__`` docstrings) so the throttle
        still engages on failure but the gate is anchored on success.

        Callers should invoke ``_attempt`` on failure paths and
        ``_success`` on the success path.
        """
        with self._lock:
            self.account_rest_last_monotonic = _clock.monotonic()

    def wake_quote_loop(self) -> None:
        """Wake the bot thread for an immediate ``one_tick`` (market or private event)."""
        self.quote_wake_event.set()

    def market_refresh_note_success(self, bb: BestBidAsk, latency_ms: float) -> dict[str, Any]:
        """
        After a successful live book apply (``apply_market_book_only`` or full snapshot).
        Updates freshness metrics; returns hints for logging / optional WS reconnect on stall.
        """
        with self._lock:
            fp = (bb.best_bid, bb.best_ask, bb.mid_price, bb.ts_exchange_ms)
            if self._market_data_snapshot_fingerprint == fp:
                self.market_data_unchanged_snapshot_streak += 1
            else:
                self.market_data_unchanged_snapshot_streak = 0
                self.market_data_stall_latched = False
            self._market_data_snapshot_fingerprint = fp
            self.market_data_failed_refresh_streak = 0
            self.market_data_last_refresh_latency_ms = float(latency_ms)
            self.market_data_last_success_wall_ts = utc_now()
            thr = int(self._settings.market_data_stall_unchanged_threshold)
            stall_just_crossed = (
                self.market_data_unchanged_snapshot_streak >= thr
                and not self.market_data_stall_latched
            )
            if stall_just_crossed:
                self.market_data_stall_latched = True
            now_m = _clock.monotonic()
            interval = float(self._settings.market_data_success_log_interval_seconds)
            emit_success_info = False
            if interval > 0 and (now_m - self._market_data_last_success_log_monotonic) >= interval:
                self._market_data_last_success_log_monotonic = now_m
                emit_success_info = True
            attempt_transport_reset = (
                stall_just_crossed and self._settings.market_data_reset_transport_on_stall
            )
            return {
                "stall_just_crossed": stall_just_crossed,
                "emit_success_info": emit_success_info,
                "unchanged_streak": self.market_data_unchanged_snapshot_streak,
                "attempt_transport_reset": attempt_transport_reset,
            }

    def market_refresh_note_transport_reset_done(self) -> None:
        with self._lock:
            self.market_data_transport_reset_count += 1

    def _market_data_metrics_unlocked(self) -> dict[str, Any]:
        rs = (
            "RECOVERING"
            if self.bot_status == BotStatus.RECOVERING_MARKET_DATA
            else "OK"
        )
        return {
            "market_data_last_success_wall_ts": self.market_data_last_success_wall_ts.isoformat()
            if self.market_data_last_success_wall_ts
            else None,
            "market_data_refresh_latency_ms": self.market_data_last_refresh_latency_ms,
            "market_data_failed_refresh_streak": self.market_data_failed_refresh_streak,
            "market_data_unchanged_snapshot_streak": self.market_data_unchanged_snapshot_streak,
            "market_data_recovery_state": rs,
            "market_data_recovery_refresh_attempts_episode": self.market_data_recovery_refresh_attempts_episode,
            "market_data_transport_reset_count": self.market_data_transport_reset_count,
        }

    def runtime_toxicity_summary_dict(self) -> dict[str, Any]:
        with self._lock:
            fills = list(self.recent_fills)
            w = int(self._settings.runtime_toxicity_fill_window)
            sid = self.session_id
        return build_runtime_toxicity_summary(fills, window=w, session_id=sid)

    def toxicity_recent_median_markout_5s_bps(self) -> Optional[float]:
        """Median 5s markout over the recent-fills window — same
        signal that drives the dashboard's "markout" tier label
        (clean / mild / moderate / heavy adverse). Returns None
        when fewer than 3 fills with finalised 5s markout exist
        (warmup; insufficient sample for a meaningful median).
        Used by the 1.2.3 markout-tier size scaler in
        ``compute_quote_decision`` to shrink orders directly on
        markout magnitude — bypasses the toxicity composite score.
        """
        with self._lock:
            samples: list[float] = []
            for f in self.recent_fills:
                mk5 = getattr(f, "markout_5s_bps", None)
                if mk5 is None:
                    continue
                try:
                    v = float(mk5)
                except (TypeError, ValueError):
                    continue
                samples.append(v)
        if len(samples) < 3:
            return None
        # Simple median via sort (small N, no need for a heap).
        samples.sort()
        n = len(samples)
        if n % 2 == 1:
            return samples[n // 2]
        return (samples[n // 2 - 1] + samples[n // 2]) / 2.0

    def quote_quality_dict(self) -> dict[str, Any]:
        with self._lock:
            return self.quote_quality.to_dict(
                realized_pnl_usd=self.pnl.realized_pnl_usd,
                fills_for_markout=list(self.recent_fills),
                markout_window=int(self._settings.runtime_toxicity_fill_window),
            )

    def note_passive_order_lifetime_if_new(self, wo: WorkingOrder) -> None:
        """Record ACK→close lifetime once per local order id (post-only FILLED/CANCELED)."""
        if not wo.post_only:
            return
        if wo.status not in (OrderStatus.FILLED, OrderStatus.CANCELED):
            return
        if wo.ts_ack is None or wo.ts_closed is None:
            return
        oid = wo.order_id_local
        if oid in self._qq_lifetime_recorded_local_ids:
            return
        self._qq_lifetime_recorded_local_ids.add(oid)
        sec = (wo.ts_closed - wo.ts_ack).total_seconds()
        self.quote_quality.note_order_lifetime_seconds(sec)

    def mid_change_rate_per_min_5m(self) -> Optional[float]:
        """v1.5.244 — rolling rate of mid changes over the last 5 min.

        Returns mid changes per minute as a float, or ``None`` when
        the session has been running for less than 60 s (the rate is
        not yet meaningful at very small samples). Prunes the
        underlying deque to wall-clock entries newer than 300 s
        before computing.

        Wall-clock based (uses ``time.time()``) so it's robust to
        bot restarts — old timestamps fall out within 5 min.
        Cheap: O(N) where N is the deque size capped at 10K. Called
        once per snapshot, not per-tick.
        """
        import time as _t
        try:
            now = _t.time()
        except Exception:
            return None
        cutoff = now - 300.0
        deq = self.mid_change_recent_ts
        # Prune from the LEFT (oldest first). collections.deque
        # popleft is O(1).
        while deq and deq[0] < cutoff:
            deq.popleft()
        # Require at least 60 s of session before reporting a rate
        # (otherwise early-session samples are noisy / misleading).
        try:
            sess_start = self.session_started_at_utc
        except AttributeError:
            sess_start = None
        if sess_start is not None:
            try:
                session_age_s = (
                    now - sess_start.timestamp()
                    if hasattr(sess_start, "timestamp")
                    else None
                )
            except Exception:
                session_age_s = None
            if session_age_s is not None and session_age_s < 60.0:
                return None
        # Rate = count / 5 minutes. Window scales naturally when the
        # session is shorter than 5 min (only counts entries that
        # fit, but session_age guard above ensures sufficient runtime).
        window_min = min(5.0, max(1.0, (now - cutoff) / 60.0))
        return float(len(deq)) / window_min

    def snapshot_dict(self) -> dict:
        with self._lock:
            m = self.market
            rt = build_runtime_toxicity_summary(
                list(self.recent_fills),
                window=int(self._settings.runtime_toxicity_fill_window),
                session_id=self.session_id,
            )
            return {
                "bot_status": self.bot_status.value,
                "symbol": self.symbol,
                "best_bid": m.best_bid if m else None,
                "best_ask": m.best_ask if m else None,
                "mid_price": m.mid_price if m else None,
                "spread_bps": m.spread_bps if m else None,
                "short_vol_bps": self.vol_bps,
                # v1.5.239 — trend-aware EWMA-of-|return| vol measure,
                # in bp. Same Optional[float] semantics as short_vol_bps
                # (None during the estimator's 2-mid warm-up). Always
                # populated regardless of REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE
                # so operators can compare both measures side-by-side
                # for calibration without flipping the consumer flag.
                "short_vol_abs_ewma_bps": self.vol_abs_ewma_bps,
                # Phase 8A Option B (v1.5.189) — AS attribution
                # surfaces. ``as_k_intensity_per_min`` is the
                # cached fill-rate driving the AS formula; None
                # when AS disabled or pre-first-refresh.
                # ``as_path_fire_count`` is the session-cumulative
                # count of ticks where AS produced the base
                # half-spread; non-monotonic only on restart.
                "as_k_intensity_per_min": getattr(
                    self, "as_k_intensity_per_min", None
                ),
                "as_path_fire_count": int(
                    getattr(self, "as_path_fire_count", 0) or 0
                ),
                "position_qty": self.position.position_qty,
                "position_notional": self.position.position_notional,
                "avg_entry_price": self.position.avg_entry_price,
                "realized_pnl_usd": self.pnl.realized_pnl_usd,
                "unrealized_pnl_usd": self.pnl.unrealized_pnl_usd,
                "total_pnl_usd": self.pnl.total_pnl_usd,
                "equity_usd": self.pnl.equity_usd,
                "drawdown_usd": self.pnl.drawdown_usd,
                "active_sides": self.last_active_sides,
                "toxicity_score": rt["toxicity_score"],
                "order_desync": self.order_desync,
                "desync_phase": self.desync_phase.value,
                "desync_quarantine_remaining": self.desync_quarantine_remaining,
                "book_age_seconds": self.book_age_seconds,
                "stale_book_warning_active": self.stale_book_warning_active,
                "flatten_incomplete": self.flatten_incomplete,
                "flatten_residual_abs_qty": self.flatten_residual_abs_qty,
                "last_update_ts": self.last_market_ts.isoformat()
                if self.last_market_ts
                else None,
                "market_data_available": m is not None
                and m.mid_price is not None
                and m.mid_price > 0,
                "account_data_available": self.account is not None,
                "withdrawable_usd": self.account.withdrawable_usd
                if self.account
                else None,
                "toxicity_delayed_markout_samples": self.toxicity.delayed_markout_sample_count,
                "toxicity_adverse_from_delayed": self.toxicity.adverse_uses_delayed_markouts,
                "trades_last_minute": self._trade_count_last_60s_unlocked(),
                "execution_errors_snapshot": self._execution_errors_snapshot_unlocked(
                    float(self._settings.execution_errors_window_seconds)
                ),
                "latency_fill_to_process_ms": self.last_latency_fill_to_process_ms,
                "exchange_ts_to_decision_ms": self.last_exchange_ts_to_decision_ms,
                "latency_decision_to_first_place_ms": self.last_latency_decision_to_first_place_ms,
                "latency_tick_preamble_ms": self.latency_tick_preamble_ms,
                "latency_hot_path_local_compute_ms": self.latency_hot_path_local_compute_ms,
                "latency_quote_engine_build_ms": self.latency_quote_engine_build_ms,
                "latency_order_maintenance_local_ms": self.latency_order_maintenance_local_ms,
                "latency_account_refresh_rest_ms": self.latency_account_refresh_rest_ms,
                "latency_reconcile_rest_ms": self.latency_reconcile_rest_ms,
                "latency_order_submit_rtt_ms": self.latency_order_submit_rtt_ms,
                "latency_private_queue_wait_ms": self.latency_private_queue_wait_ms,
                "latency_public_ws_queue_wait_ms": self.latency_public_ws_queue_wait_ms,
                "private_ws_connected": self.private_ws_connected,
                "private_ws_healthy": self.private_ws_healthy,
                "private_ws_queue_drops": self.private_ws_queue_drops,
                "private_ws_recovery_pending": self.private_ws_recovery_pending,
                "private_ws_last_connect_ts": self.private_ws_last_connect_wall_ts.isoformat()
                if self.private_ws_last_connect_wall_ts
                else None,
                "private_ws_last_ping_sent_ts": self.private_ws_last_ping_sent_wall_ts.isoformat()
                if self.private_ws_last_ping_sent_wall_ts
                else None,
                "private_ws_last_pong_ts": self.private_ws_last_pong_wall_ts.isoformat()
                if self.private_ws_last_pong_wall_ts
                else None,
                "private_ws_last_message_ts": self.private_ws_last_message_wall_ts.isoformat()
                if self.private_ws_last_message_wall_ts
                else None,
                "private_ws_seconds_since_last_message": self._private_ws_seconds_since_last_message_unlocked(),
                "private_ws_disconnect_reason_last": self.private_ws_disconnect_reason_last,
                "private_ws_disconnect_close_code_last": self.private_ws_disconnect_close_code_last,
                "private_ws_disconnect_histogram": dict(self.private_ws_disconnect_histogram),
                "private_ws_reconnect_count": self.private_ws_reconnect_count,
                "private_ws_reconnect_reason_counts": dict(self.private_ws_reconnect_reason_counts),
                "private_ws_queue_high_watermark": self.private_ws_queue_high_watermark,
                "private_ws_queue_overflow_count": self.private_ws_queue_drops,
                "private_ws_queue_backlog_after_drain_count": self.private_ws_queue_backlog_after_drain_count,
                "private_ws_max_events_drained_per_tick": self.private_ws_max_events_drained_per_tick,
                "private_ws_events_drained_last_tick": self.private_ws_events_drained_last_tick,
                "private_ws_queue_depth_after_drain": self.private_ws_queue_depth_after_drain,
                "private_ws_drain_time_ms_last_tick": self.private_ws_drain_time_ms_last_tick,
                "private_ws_last_inbound_derived_ms": dict(self.private_ws_last_inbound_derived_ms),
                # OKX-only diagnostics (zeros / empty dict on other venues).
                "okx_ws_orders_msgs_session": self.okx_ws_orders_msgs_session,
                "okx_ws_orders_msgs_by_state": dict(self.okx_ws_orders_msgs_by_state),
                "okx_ws_pong_gap_seconds_max": self.okx_ws_pong_gap_seconds_max,
                # Tape runtime-feed (shmem IPC) health + consumer-fire
                # counters (M7.6 / M8.4). All zero when
                # REGIME_USE_RUNTIME_RECORDER_FEED=false (the default) —
                # byte-identical surface to pre-M7 snapshots in that case.
                # The quote loop MIRRORS the reader's monotonic counters
                # by assignment each tick; the seeded / fires counters are
                # bumped directly by the consumers.
                "runtime_feed_read_count": self.runtime_feed_read_count,
                "runtime_feed_stale_count": self.runtime_feed_stale_count,
                "runtime_feed_version_mismatch_count": self.runtime_feed_version_mismatch_count,
                "runtime_feed_collision_count": self.runtime_feed_collision_count,
                "warmstart_vol_seeded_from_recorder_count": self.warmstart_vol_seeded_from_recorder_count,
                "microprice_widen_z_runtime_feed_fires": self.microprice_widen_z_runtime_feed_fires,
                "public_ws_events_applied_last_tick": self.public_ws_events_applied_last_tick,
                "public_ws_max_burst_per_tick": self.public_ws_max_burst_per_tick,
                "public_ws_last_inbound_derived_ms": dict(self.public_ws_last_inbound_derived_ms),
                "public_ws_connected": self.public_ws_connected,
                "public_ws_last_message_wall_ts": self.public_ws_last_message_wall_ts.isoformat()
                if self.public_ws_last_message_wall_ts
                else None,
                "public_ws_reconnect_count": self.public_ws_reconnect_count,
                "public_ws_seen_first_bbo": self.public_ws_seen_first_bbo,
                # 1.2.25: session-cumulative BBO + mid-change counts
                # for todo-010 Phase 2 decision data. Divide by
                # session duration to get rates.
                "bbo_event_count_session": self.bbo_event_count_session,
                "mid_change_count_session": self.mid_change_count_session,
                # v1.5.244 — rolling 5-min rate. The binding constraint
                # on fill rate (fills ≈ 0.10 × mid_change_rate). Below
                # ~6/min the bot likely fails the 25-fills / 30-min
                # calibration validity gate.
                "mid_change_rate_per_min_5m": self.mid_change_rate_per_min_5m(),
                "live_market_data_source": self.live_market_data_source,
                "quote_refresh_aging": dict(self.quote_refresh_aging),
                "exchange_snapshot_unhealthy_streak": self.exchange_snapshot_unhealthy_streak,
                "reconcile_auto_pause": self.reconcile_auto_pause,
                "recovering_market_data": self.bot_status == BotStatus.RECOVERING_MARKET_DATA,
                **self._market_data_metrics_unlocked(),
                **self._binance_ws_status_dict_unlocked(),
                "session_id": self.session_id,
                "session_started_at_utc": self.session_started_at_utc.isoformat(),
                # 1.3.59: session-cumulative guard counters surfaced
                # for the postmortem connectivity-health section + the
                # dashboard's Execution-quality card. Each increment
                # is a connectivity/race-loss event the operator needs
                # visibility into. Zero-cost on the hot path: plain
                # ``int += 1`` in the existing fall-through branches.
                "reconcile_skip_snapshot_stale_total": int(
                    self.reconcile_skip_snapshot_stale_total
                ),
                "place_unconfirmed_critical_total": int(
                    self.place_unconfirmed_critical_total
                ),
                "cancel_race_lost_to_fill_total": int(
                    self.cancel_race_lost_to_fill_total
                ),
                "cancel_unexpected_gone_total": int(
                    self.cancel_unexpected_gone_total
                ),
                "place_cancel_race_total": int(
                    self.place_cancel_race_total
                ),
                # v1.5.267 BUG-034 surfacing fix. Was originally added
                # to live_stats.py (the S3 publisher) in v1.5.262 — but
                # the snapshot acceptance check reads state_current.json
                # which is generated by THIS dict (snapshot_dict), not
                # by live_stats. Move the counters here so the
                # acceptance gate ``check_v1_5_262_bug034_divergence_
                # recovery_triggered`` can see them and verify the
                # invariant that every divergence triggers a REST fill
                # catch-up (count_recovery == count_divergence).
                "shadow_position_divergence_count": int(
                    self.shadow_position_divergence_count
                ),
                "shadow_position_divergence_recovery_count": int(
                    self.shadow_position_divergence_recovery_count
                ),
                "shadow_position_last_divergence_qty": (
                    float(self.shadow_position_last_divergence_qty)
                    if self.shadow_position_last_divergence_qty is not None
                    else None
                ),
                # v1.5.277 / AQC Phase 1 — Active Quoting Controller
                # diagnostics. Phase 1 is observe-only; the snapshot
                # value documents what the AQC WOULD have done if its
                # output were consumed. Phase 2+ wires the
                # ``aggression_level`` into effective multipliers on
                # the existing defensive gates.
                "active_quoting_controller": (
                    self.active_quoting_controller.snapshot_dict()
                ),
                # v1.4.96 — three-tier connectivity counters. See
                # ``BotState.__init__`` for tier semantics + contracts.
                # ----- TIER 1 (FATAL): gone_on_exchange -----
                "gone_on_exchange_total": int(self.gone_on_exchange_total),
                "gone_on_exchange_phantom_no_ack_total": int(
                    self.gone_on_exchange_phantom_no_ack_total
                ),
                "gone_on_exchange_acked_no_cancel_total": int(
                    self.gone_on_exchange_acked_no_cancel_total
                ),
                "gone_on_exchange_cancel_no_http_confirm_total": int(
                    self.gone_on_exchange_cancel_no_http_confirm_total
                ),
                "gone_on_exchange_other_total": int(
                    self.gone_on_exchange_other_total
                ),
                "gone_on_exchange_recent": list(self.gone_on_exchange_recent),
                # ----- TIER 2 (WARN): http_acked_no_ws -----
                "http_acked_no_ws_total": int(self.http_acked_no_ws_total),
                "http_acked_no_ws_recent": list(self.http_acked_no_ws_recent),
                "http_acked_no_ws_lateness_ms_min": (
                    float(self.http_acked_no_ws_lateness_ms_min)
                    if self.http_acked_no_ws_lateness_ms_min is not None
                    else None
                ),
                "http_acked_no_ws_lateness_ms_max": (
                    float(self.http_acked_no_ws_lateness_ms_max)
                    if self.http_acked_no_ws_lateness_ms_max is not None
                    else None
                ),
                "http_acked_no_ws_lateness_p50_ms": (
                    self._ws_lateness_percentile_unlocked(0.50)
                ),
                "http_acked_no_ws_lateness_p95_ms": (
                    self._ws_lateness_percentile_unlocked(0.95)
                ),
                # ----- TIER 3 (WARN): ws_arrived_late -----
                "ws_arrived_late_total": int(self.ws_arrived_late_total),
                "ws_arrived_late_recent": list(self.ws_arrived_late_recent),
                # v1.4.100 ladder-observability F2 — drop-attribution counters.
                # v1.5.26 Phase 2B -- canonical multi-horizon
                # mid-drift dataclass published as a sub-block. All 7
                # windows from one walk; observers (postmortem,
                # dashboard, future gate consumers) read here. None
                # entries during the warmup window before each
                # deque has a sample inside that horizon.
                "mid_drift_windows": (
                    self.mid_drift_windows.to_dict()
                    if self.mid_drift_windows is not None
                    else None
                ),
                # v1.5.41 Phase 4C.3 — per-side trailing realised-edge
                # history. Publishes mean / stdev / z-score / and the
                # multiplier currently applied to expected_edge per
                # side. Snapshot is None when the feature is disabled
                # or the deque hasn't accumulated min_samples yet
                # (multiplier surfaces as 1.0 in that case).
                "side_edge_history": (
                    self.side_edge_history.to_snapshot(
                        _clock.monotonic()
                    )
                    if getattr(self, "side_edge_history", None) is not None
                    else None
                ),
                "ladder_rung_drops": {
                    "grid_collision_total": int(
                        self.ladder_rung_dropped_grid_collision_total
                    ),
                    "min_notional_total": int(
                        self.ladder_rung_dropped_min_notional_total
                    ),
                    "inventory_buffer_total": int(
                        self.ladder_rung_dropped_inventory_buffer_total
                    ),
                    "inventory_aware_pruning_total": int(
                        self.ladder_rung_dropped_inventory_aware_pruning_total
                    ),
                    "position_cap_total": int(
                        self.ladder_rung_dropped_position_cap_total
                    ),
                    "in_flight_total": int(
                        self.ladder_rung_dropped_in_flight_total
                    ),
                    # v1.5.26 Phase 2C.3 -- regime-mode ladder cap.
                    "regime_mode_total": int(
                        self.ladder_rung_dropped_regime_mode_total
                    ),
                    "other_total": int(
                        self.ladder_rung_dropped_other_total
                    ),
                    # v1.4.222 (Phase 4G.11) — dispatcher-side
                    # ``normalize_order_pair`` drop counters. Separate
                    # from the build_ladder counters above because the
                    # drop site is downstream of build_ladder. See the
                    # field docstring at construction time for the
                    # full taxonomy.
                    "normalize_below_min_notional_total": int(
                        self.ladder_rung_dropped_normalize_below_min_notional_total
                    ),
                    "normalize_other_total": int(
                        self.ladder_rung_dropped_normalize_other_total
                    ),
                    "sum_total": (
                        int(self.ladder_rung_dropped_grid_collision_total)
                        + int(self.ladder_rung_dropped_min_notional_total)
                        + int(self.ladder_rung_dropped_inventory_buffer_total)
                        + int(self.ladder_rung_dropped_inventory_aware_pruning_total)
                        + int(self.ladder_rung_dropped_position_cap_total)
                        + int(self.ladder_rung_dropped_in_flight_total)
                        + int(self.ladder_rung_dropped_regime_mode_total)
                        + int(self.ladder_rung_dropped_other_total)
                        + int(self.ladder_rung_dropped_normalize_below_min_notional_total)
                        + int(self.ladder_rung_dropped_normalize_other_total)
                    ),
                },
                "hydration_skipped_recently_terminal_total": int(
                    self.hydration_skipped_recently_terminal_total
                ),
                "cancel_deferred_until_ack_total": int(
                    self.cancel_deferred_until_ack_total
                ),
                # v1.4.36 (Codex #4): silent-drop counter for
                # pending-markout-jobs deque overflow. Bumps when
                # ``register_fill_for_delayed_markouts`` appends while
                # the deque is already at ``maxlen``; a non-zero value
                # means at least one 1s/3s/5s markout completion was
                # silently lost. See ``register_fill_for_delayed_
                # markouts`` for the bump site.
                "markout_jobs_dropped_overflow_count": int(
                    getattr(self, "markout_jobs_dropped_overflow_count", 0)
                ),
                # v1.4.40 BUG-025: executor internal state machine
                # snapshot. Populated by
                # ``OrderManager._sync_executor_state_snapshot`` on
                # every quote cycle + outbound batch event. Surfaces
                # which latch (if any) is wedged when the bot enters
                # the "engine emits valid quotes, no place fires"
                # silent-wedge state. Empty dict before the first
                # cycle populates it.
                "executor_state": dict(self.executor_state_snapshot)
                if getattr(self, "executor_state_snapshot", None)
                else {},
                "session_cross_venue_cancel_count": int(
                    self.session_cross_venue_cancel_count
                ),
                "session_cross_venue_amend_count": int(
                    self.session_cross_venue_amend_count
                ),
                # amend-prio Phase 4 (v1.4.17): amend rollout counters.
                # See the ``amend_*_total`` field definitions above for
                # the per-counter semantics. All zero until the operator
                # flips ``OKX_AMEND_ON_REPRICE_ENABLED=true``.
                "amend_intents_emitted_total": int(
                    self.amend_intents_emitted_total
                ),
                "amend_success_total": int(self.amend_success_total),
                "amend_below_filled_total": int(
                    self.amend_below_filled_total
                ),
                "amend_order_gone_total": int(self.amend_order_gone_total),
                "amend_post_only_cross_total": int(
                    self.amend_post_only_cross_total
                ),
                "amend_exchange_rejected_other_total": int(
                    self.amend_exchange_rejected_other_total
                ),
                "amend_transport_rejected_total": int(
                    self.amend_transport_rejected_total
                ),
                "amend_pending_high_watermark": int(
                    self.amend_pending_high_watermark
                ),
                "amend_to_cancel_fallback_total": int(
                    self.amend_below_filled_total
                    + self.amend_exchange_rejected_other_total
                ),
                # v1.5.206 / v1.5.216 — amend race resilience counters
                "amend_unconfirmed_swallowed_by_tombstone_total": int(
                    getattr(
                        self, "amend_unconfirmed_swallowed_by_tombstone_total", 0
                    ) or 0
                ),
                "amend_missing_row_swallowed_total": int(
                    getattr(self, "amend_missing_row_swallowed_total", 0) or 0
                ),
                # v1.5.207 Phase 4C.5 — participation score per side.
                # Re-uses the same data the live_stats publisher emits;
                # surfaced here so acceptance scripts reading
                # state_current.json can see it. Defensive getattr
                # guards against pre-v1.5.207 hot-reloads.
                "participation_score": {
                    "bid": getattr(self, "participation_score_bid", None),
                    "ask": getattr(self, "participation_score_ask", None),
                    "disagreement_bid_total": int(
                        getattr(self, "participation_score_disagreement_bid_total", 0) or 0
                    ),
                    "disagreement_ask_total": int(
                        getattr(self, "participation_score_disagreement_ask_total", 0) or 0
                    ),
                },
                # v1.5.209 Phase 8D — OFI accumulator snapshot. The
                # accumulator runs unconditionally; this block tells the
                # operator whether it's actually been fed BBO updates
                # and what signal it's producing.
                "ofi": (
                    self.ofi.snapshot_dict()
                    if getattr(self, "ofi", None) is not None
                    else None
                ),
                "ofi_last_shift_bps": float(
                    getattr(self, "ofi_last_shift_bps", 0.0) or 0.0
                ),
                # v1.5.209 Phase 8B — queue-aware sizing + inside-post
                # telemetry. Per-side ratios + multipliers + cumulative
                # armed counters.
                "queue_aware": {
                    "ratio_bid": getattr(self, "queue_position_ratio_bid", None),
                    "ratio_ask": getattr(self, "queue_position_ratio_ask", None),
                    "size_mult_bid": float(
                        getattr(self, "queue_size_mult_bid", 1.0) or 1.0
                    ),
                    "size_mult_ask": float(
                        getattr(self, "queue_size_mult_ask", 1.0) or 1.0
                    ),
                    "inside_post_active_bid": bool(
                        getattr(self, "queue_inside_post_active_bid", False)
                    ),
                    "inside_post_active_ask": bool(
                        getattr(self, "queue_inside_post_active_ask", False)
                    ),
                    "size_mult_armed_bid_total": int(
                        getattr(self, "queue_size_mult_armed_bid_total", 0) or 0
                    ),
                    "size_mult_armed_ask_total": int(
                        getattr(self, "queue_size_mult_armed_ask_total", 0) or 0
                    ),
                    "inside_post_armed_bid_total": int(
                        getattr(self, "queue_inside_post_armed_bid_total", 0) or 0
                    ),
                    "inside_post_armed_ask_total": int(
                        getattr(self, "queue_inside_post_armed_ask_total", 0) or 0
                    ),
                },
                # rate-limit-observability Phase 2 (v1.4.20): per-pool
                # rate-limit window snapshot. Populated by the bot's
                # heartbeat handler each tick from the OKX adapter's
                # ``rest_runtime_counters()``. Keys: ``aggregate``,
                # ``place_batch``, ``cancel_batch``, ``amend_batch``,
                # ``place_single``, ``cancel_single``, ``reads``,
                # ``other`` — only pools that have seen traffic
                # appear. Each entry: ``current_2s``, ``peak_2s_60s``,
                # ``total``, ``cap``, ``pct_of_cap``.
                "okx_rate_window_per_pool": dict(
                    self.okx_rate_window_per_pool
                ),
                "quote_quality": self.quote_quality.to_dict(
                    realized_pnl_usd=self.pnl.realized_pnl_usd,
                    fills_for_markout=list(self.recent_fills),
                    markout_window=int(self._settings.runtime_toxicity_fill_window),
                ),
                # Priority #2 v2 — basis-deviation regime classifier state.
                # Exposed so the operator can verify the classifier is
                # producing sensible IC values / regime decisions across
                # different market conditions before enabling the
                # basis-deviation alpha. See ``app/basis_regime.py``.
                "basis_regime": self.basis_regime.snapshot(),
                # Priority #3 v1 — flow-direction / toxicity score. In
                # v1 this is observability-only: ``FLOW_SCORE_PAUSE_THRESHOLD``
                # defaults to 1.1 (unreachable) so the integration layer
                # never fires. Watch ``buy_toxic_score`` /
                # ``sell_toxic_score`` across sessions to calibrate the
                # threshold before enabling pre-fill side suppression.
                "flow_score": self.flow_score.snapshot_dict(),
                # 1.2.3: behavioural-gate state surfaced for snapshot
                # capture (Codex MED-7 follow-up + tiered drawdown +
                # vol×trend gate + post-swing). Each block carries
                # ``active`` / ``fire_count`` / last-trigger summary
                # so postmortem can reconstruct gate activity from the
                # snapshot alone — matches what's in ``live_stats``
                # but here it lands in the static ``state_current.json``
                # the snapshot bundle pulls from /state/current.
                "behavioural_gates": _behavioural_gates_snapshot(self),
                # 1.2.34: per-cycle spread decomposition for the
                # dashboard's Spread tab. ``None`` until the first
                # quote cycle has run. The dashboard polls this and
                # renders every contribution to the current quote
                # (reservation shifts, half-spread stack, size mult
                # chain) in one structured view.
                "last_quote_breakdown": (
                    self.last_quote_breakdown.to_dict()
                    if getattr(self, "last_quote_breakdown", None) is not None
                    else None
                ),
            }

    # ------------------------------------------------------------------
    # 1.3.130 multi-rung Phase 2 — per-rung working-order accessors
    # ------------------------------------------------------------------

    @property
    def working_bid(self) -> Optional[WorkingOrder]:
        """Backward-compat shim: returns the INSIDE bid rung (level_idx=0).
        At N=1 (single-rung mode, default) this is the only rung; the
        60+ existing read sites that consult ``working_bid`` continue
        to work unchanged. At N>1 outer bid rungs are visible via
        ``get_working_order(Side.BUY, level_idx)`` or
        ``iter_working_orders(Side.BUY)``.

        v1.4.82 wedge-elimination-cleanup Phase 3D: NEW callers should
        prefer ``state.order_store.get(Side.BUY, 0)`` (or
        ``state.tick_snapshot().orders.slot(Side.BUY, 0)`` when reading
        inside a tick). The legacy shim is preserved for the existing
        read sites; Phase 5A migrated the hot path to the snapshot
        pattern and the remaining ~60 sites are migrated at flag-day.

        v1.4.91 Phase 3A.6: emits a ``DeprecationWarning`` ONCE per
        process. Test runs with ``-W error::DeprecationWarning`` will
        flag new uses; prod with the default silent-DW filter is
        unaffected."""
        global _WORKING_BID_DEPRECATION_EMITTED
        if not _WORKING_BID_DEPRECATION_EMITTED:
            _WORKING_BID_DEPRECATION_EMITTED = True
            warnings.warn(
                "BotState.working_bid is deprecated since v1.4.91 "
                "(Phase 3A.6). Prefer state.order_store.get(Side.BUY, 0) "
                "or state.tick_snapshot().orders.slot(Side.BUY, 0). "
                "This warning fires once per process.",
                DeprecationWarning,
                stacklevel=2,
            )
        wos = getattr(self, "_working_orders", None)
        if wos is None:
            return None
        return wos.get(Side.BUY, {}).get(0)

    @working_bid.setter
    def working_bid(self, wo: Optional[WorkingOrder]) -> None:
        """Backward-compat shim: assigns the INSIDE bid rung. Setting
        to ``None`` removes the (BUY, 0) entry — the dict doesn't
        accumulate None sentinels."""
        wos = getattr(self, "_working_orders", None)
        if wos is None:
            wos = {Side.BUY: {}, Side.SELL: {}}
            self._working_orders = wos
        if wo is None:
            wos[Side.BUY].pop(0, None)
        else:
            wos[Side.BUY][0] = wo

    @property
    def working_ask(self) -> Optional[WorkingOrder]:
        """Backward-compat shim: returns the INSIDE ask rung (level_idx=0).
        See ``working_bid`` for the multi-rung accessor pattern and the
        v1.4.82 Phase 3D preferred-call-path note (``state.order_store``
        or ``state.tick_snapshot()``).

        v1.4.91 Phase 3A.6: emits a ``DeprecationWarning`` ONCE per
        process. See ``working_bid`` for rationale."""
        global _WORKING_ASK_DEPRECATION_EMITTED
        if not _WORKING_ASK_DEPRECATION_EMITTED:
            _WORKING_ASK_DEPRECATION_EMITTED = True
            warnings.warn(
                "BotState.working_ask is deprecated since v1.4.91 "
                "(Phase 3A.6). Prefer state.order_store.get(Side.SELL, 0) "
                "or state.tick_snapshot().orders.slot(Side.SELL, 0). "
                "This warning fires once per process.",
                DeprecationWarning,
                stacklevel=2,
            )
        wos = getattr(self, "_working_orders", None)
        if wos is None:
            return None
        return wos.get(Side.SELL, {}).get(0)

    @working_ask.setter
    def working_ask(self, wo: Optional[WorkingOrder]) -> None:
        wos = getattr(self, "_working_orders", None)
        if wos is None:
            wos = {Side.BUY: {}, Side.SELL: {}}
            self._working_orders = wos
        if wo is None:
            wos[Side.SELL].pop(0, None)
        else:
            wos[Side.SELL][0] = wo

    def get_working_order(
        self, side: Side, level_idx: int
    ) -> Optional[WorkingOrder]:
        """Per-rung accessor: returns the rung at ``(side, level_idx)``
        or None when absent. ``level_idx=0`` is the inside rung; outer
        rungs only exist when ``LADDER_NUM_LEVELS_PER_SIDE > 1``."""
        wos = getattr(self, "_working_orders", None)
        if wos is None:
            return None
        return wos.get(side, {}).get(int(level_idx))

    def set_working_order(
        self, side: Side, level_idx: int, wo: Optional[WorkingOrder]
    ) -> None:
        """Per-rung setter. Use this from the ladder orchestrator when
        placing an outer rung. ``wo=None`` removes the entry. The
        ``working_bid`` / ``working_ask`` property setters are the
        inside-rung-specific specialisations (level_idx=0).

        v1.4.80 Phase 3A: also updates the ``OrderStore`` indexes
        when present. This setter is the LEGACY API — new callers
        should prefer ``self.order_store.set(side, lvl, wo)`` which
        does the index maintenance up front. Both paths produce
        identical final state.
        """
        wos = getattr(self, "_working_orders", None)
        if wos is None:
            wos = {Side.BUY: {}, Side.SELL: {}}
            self._working_orders = wos
        bucket = wos.setdefault(side, {})
        # v1.4.80 Phase 3A: keep the OrderStore indexes in sync.
        # ``order_store`` may not exist yet during __init__ (when
        # the dict is constructed before the store); the getattr
        # guard handles that bootstrap window.
        store = getattr(self, "order_store", None)
        existing = bucket.get(int(level_idx))
        if wo is None:
            bucket.pop(int(level_idx), None)
            if store is not None and existing is not None:
                store._index_remove(existing, reason="slot_cleared")
        else:
            bucket[int(level_idx)] = wo
            if store is not None:
                if existing is not None and existing is not wo:
                    store._index_remove(existing, reason="slot_replaced")
                # Only index non-terminal WOs (matches OrderStore.set).
                from app.stores.order_store import _TERMINAL_STATUSES
                if wo.status not in _TERMINAL_STATUSES:
                    store._index_add(wo)

    def iter_working_orders(
        self, side: Side
    ) -> list[tuple[int, WorkingOrder]]:
        """Per-rung iteration helper. Returns ``(level_idx, WorkingOrder)``
        tuples sorted by ``level_idx`` ascending (inside first). Empty
        list when the side has no working orders. Snapshot — safe to
        iterate without holding the lock as long as the caller doesn't
        mutate via this list."""
        wos = getattr(self, "_working_orders", None)
        if wos is None:
            return []
        return sorted(wos.get(side, {}).items())

    def all_working_orders(self) -> list[WorkingOrder]:
        """Flat list of every working order across both sides, all
        rungs. Convenience for telemetry / blind-resting checks that
        need to see every order regardless of side or rung. Order:
        BUY rungs (inside first) then SELL rungs (inside first)."""
        out: list[WorkingOrder] = []
        for side in (Side.BUY, Side.SELL):
            for _, wo in self.iter_working_orders(side):
                out.append(wo)
        return out

    def append_forward_signal_history(
        self,
        *,
        now_mono: float,
        vol_bps: Optional[float],
        drift_30s_bps: Optional[float],
        ob_imbalance: Optional[float],
        max_age_seconds: float,
        basis_bps: Optional[float] = None,
        basis_max_age_seconds: Optional[float] = None,
        vol_abs_ewma_bps: Optional[float] = None,
    ) -> None:
        """Phase 4G.5 (v1.4.211) -- append current readings to the
        forward-classifier history buffers and prune entries older
        than ``max_age_seconds`` (typically
        ``settings.regime_forward_history_buffer_seconds``, default
        180 s).

        Phase 4G.7 (v1.5.18) extension: ``basis_bps`` is appended to
        a separate buffer with its own longer retention window
        (``basis_max_age_seconds``, default 1800 s = 30 min) so the
        classifier's ``basis_stretch_cautious_ratio`` trigger has the
        median-baseline it needs. The other three buffers retain
        ``max_age_seconds`` (~180 s) -- they're consumed by
        derivative indicators (vol slope, drift ratio, OB delta) for
        which a shorter window is correct.

        Defensive: ``None`` values are silently skipped per-channel
        (early ticks before WS warm-up may have any subset missing).
        The classifier itself handles short / missing histories with
        ``None`` returns from its indicator helpers.

        Called once per quote-loop tick from the bot's regime
        evaluation block. Cheap (O(N) where N = ~180 s × 2 Hz ≈ 360
        entries for the short buffers; ~3600 for the basis buffer at
        30 min) -- still well under any hot-path concern.
        """
        cutoff = float(now_mono) - float(max_age_seconds)
        if vol_bps is not None:
            self.forward_vol_bps_history.append((float(now_mono), float(vol_bps)))
            if self.forward_vol_bps_history and self.forward_vol_bps_history[0][0] < cutoff:
                self.forward_vol_bps_history = [
                    (t, v) for (t, v) in self.forward_vol_bps_history if t >= cutoff
                ]
        # v1.5.239 — parallel append for the trend-aware EWMA vol
        # measure. Same retention window as the stdev-based buffer
        # so the two are directly comparable.
        if vol_abs_ewma_bps is not None:
            self.forward_vol_abs_ewma_bps_history.append(
                (float(now_mono), float(vol_abs_ewma_bps))
            )
            if (
                self.forward_vol_abs_ewma_bps_history
                and self.forward_vol_abs_ewma_bps_history[0][0] < cutoff
            ):
                self.forward_vol_abs_ewma_bps_history = [
                    (t, v)
                    for (t, v) in self.forward_vol_abs_ewma_bps_history
                    if t >= cutoff
                ]
        if drift_30s_bps is not None:
            self.forward_drift_30s_history.append(
                (float(now_mono), float(drift_30s_bps))
            )
            if self.forward_drift_30s_history and self.forward_drift_30s_history[0][0] < cutoff:
                self.forward_drift_30s_history = [
                    (t, v) for (t, v) in self.forward_drift_30s_history if t >= cutoff
                ]
        if ob_imbalance is not None:
            self.forward_ob_imbalance_history.append(
                (float(now_mono), float(ob_imbalance))
            )
            if self.forward_ob_imbalance_history and self.forward_ob_imbalance_history[0][0] < cutoff:
                self.forward_ob_imbalance_history = [
                    (t, v) for (t, v) in self.forward_ob_imbalance_history if t >= cutoff
                ]
        # Phase 4G.7 -- basis buffer (independent retention window).
        if basis_bps is not None:
            basis_window = float(
                basis_max_age_seconds
                if basis_max_age_seconds is not None
                else 1800.0
            )
            basis_cutoff = float(now_mono) - basis_window
            self.forward_basis_bps_history.append(
                (float(now_mono), float(basis_bps))
            )
            if (
                self.forward_basis_bps_history
                and self.forward_basis_bps_history[0][0] < basis_cutoff
            ):
                self.forward_basis_bps_history = [
                    (t, v) for (t, v) in self.forward_basis_bps_history
                    if t >= basis_cutoff
                ]

    def median_forward_basis_bps(self) -> Optional[float]:
        """Phase 4G.7 -- median of the rolling basis-bps buffer.

        Returns ``None`` when the buffer is empty (no basis samples
        observed yet, typically the first ~30 s after bot start). The
        classifier treats ``None`` as "no signal" -- the basis_stretch
        CAUTIOUS trigger stays dormant until enough history exists.

        Implementation note: we deliberately compute the median fresh
        on each call (O(N log N) sort) rather than maintaining a
        running-median data structure. At the buffer's expected size
        (~3600 entries at 30 min × 2 Hz) this is sub-millisecond and
        only runs once per quote-loop tick. Trading correctness
        (simple, audit-able) for negligible CPU.
        """
        if not self.forward_basis_bps_history:
            return None
        values = [v for (_, v) in self.forward_basis_bps_history]
        values.sort()
        n = len(values)
        mid = n // 2
        if n % 2 == 0:
            return 0.5 * (values[mid - 1] + values[mid])
        return values[mid]

    def tick_snapshot(self):
        """v1.4.82 Phase 3C — build an immutable point-in-time view
        of the state for the current tick. Returns a ``TickSnapshot``
        with ``ImmutableOrderView`` + ``ImmutablePositionView`` +
        ``ImmutableMarketView`` + captured_at + bot_status.

        Acquires ``state._lock`` ONCE; downstream callers read from
        the immutable view without re-entering the lock. Phase 5A
        will cut ``maybe_refresh_quotes`` over to consume this.
        """
        from app.stores.tick_snapshot import build_tick_snapshot
        return build_tick_snapshot(self)

    # v1.5.205 Phase 4C.4 helpers --------------------------------------- #

    def record_quote_age_decision_sample_seconds(self, age_seconds: float) -> None:
        """v1.5.205 — push one sample of decision-time quote age into
        the rolling window the stale-risk penalty reads. Caller passes
        the LARGER of the bid/ask resting-quote age (or whichever
        side it's evaluating). Silently drops non-finite / negative
        samples — defensive against clock edge-cases.
        """
        try:
            f = float(age_seconds)
        except (TypeError, ValueError):
            return
        if f != f or f < 0.0 or f == float("inf"):
            return
        with self._lock:
            self.quote_age_decision_samples_seconds.append(f)

    def quote_age_decision_p50_seconds(self) -> Optional[float]:
        """v1.5.205 — return the P50 of the rolling decision-time
        quote-age window, in seconds. Returns ``None`` until the
        window has at least 20 samples (the warmup floor) — the
        stale-risk penalty stays dormant until then.
        """
        with self._lock:
            samples = list(self.quote_age_decision_samples_seconds)
        if len(samples) < 20:
            return None
        sorted_s = sorted(samples)
        mid = len(sorted_s) // 2
        if len(sorted_s) % 2 == 1:
            return sorted_s[mid]
        return (sorted_s[mid - 1] + sorted_s[mid]) / 2.0

    def open_order_count(self) -> int:
        # v1.5.136 (Codex bug #10) — DESYNC added to the terminal-
        # exclusion set so this counter agrees with the rest of the
        # codebase about which orders are "live".
        #
        # Pre-fix the counter omitted DESYNC even though DESYNC is
        # documented as ``terminal-equivalent`` in OrderStatus's
        # docstring (``TERMINAL = {CANCELED, FILLED, REJECTED,
        # DESYNC}``) AND is already excluded by three other code
        # paths (reconciler ``_TERMINAL`` set,
        # ``_local_has_cancellable_wos`` skip list,
        # ``cancel_all_orders_for_symbol`` skip list).
        #
        # Production impact of the pre-fix bug: every quote tick
        # (~2 Hz) ``evaluate_risk`` (app/bot.py:6280 and :7609)
        # compared this count against ``settings.max_open_orders``.
        # If a DESYNC tombstone tipped the ladder over the cap, the
        # risk decision returned ``CANCEL_ALL`` and the bot cancelled
        # its own live orders. Phase 2B's reaper retires DESYNC
        # tombstones after ``DESYNC_REAP_TIMEOUT_SECONDS`` (default
        # 30 s), so the failure mode was a 30-second cancel-storm
        # window after every legitimate DESYNC event.
        with self._lock:
            n = 0
            for o in self.all_working_orders():
                if o.status not in (
                    OrderStatus.CANCELED,
                    OrderStatus.FILLED,
                    OrderStatus.REJECTED,
                    OrderStatus.DESYNC,
                ):
                    n += 1
            return n

    def set_inventory_baseline_at_session_start(self, qty: float) -> None:
        """TODO-001: capture the inherited position at session start. Subsequent
        session fills accumulate into ``session_signed_qty_total``; the
        watchdog compares ``baseline + session_signed_qty_total`` to
        ``state.position.position_qty`` (the venue truth) to detect
        divergence. Called once per session by the startup seed path; idempotent
        once set so a re-call (e.g. mid-tick refresh) cannot reset it.
        """
        with self._lock:
            if self.inventory_baseline_set:
                return
            self.inventory_baseline_qty = float(qty)
            self.session_signed_qty_total = 0.0
            self.inventory_baseline_set = True

    def has_resting_passive_order(self) -> bool:
        """TODO-004: True iff at least one passive order is currently on the book.

        ``ACKED`` / ``PARTIAL`` are confirmed-on-book; ``CANCEL_PENDING`` orders
        are still on the book until the cancel-confirm arrives, so they count
        too — that's the window the blind-resting metric is meant to catch.
        ``SENT`` is excluded (in flight, not yet confirmed on book).
        """
        with self._lock:
            for o in self.all_working_orders():
                if o.status in (
                    OrderStatus.ACKED,
                    OrderStatus.PARTIAL,
                    OrderStatus.CANCEL_PENDING,
                ):
                    return True
            return False

    def note_blind_resting_sample(self, *, blind: bool, now_mono: float) -> None:
        """TODO-004: accumulate ``blind_resting_seconds_total`` across ticks.

        Should be called once per tick from the bot loop with ``blind=True``
        when (risk == NO_QUOTE AND has_resting_passive_order()) and
        ``blind=False`` otherwise. Uses the monotonic clock so the metric is
        immune to wall-clock jumps.
        """
        with self._lock:
            prev = self._blind_resting_last_check_mono
            self._blind_resting_last_check_mono = now_mono
            if prev is None:
                return
            dt = now_mono - prev
            if dt <= 0.0 or dt > 60.0:
                # Bound: skip giant jumps (sleep / process pause) — those
                # are not actually "blind-with-orders" exposure.
                return
            if blind:
                self.blind_resting_seconds_total += dt
                self.blind_resting_tick_count += 1

    def is_killed(self) -> bool:
        with self._lock:
            return self.killed

    def mark_paused(self, reason: str) -> None:
        """Transition the bot into PAUSED with a reason tag.

        Caller is expected to already hold ``self._lock`` if
        atomicity vs. other state mutations matters; this helper only
        protects the pause_reason/pause_timestamp fields. The
        ``bot_status`` write is the caller's responsibility because
        in some paths (e.g. reconcile_stall) the call is made under
        an already-held lock.
        """
        from app.utils.time import utc_now

        self.pause_reason = str(reason or "unspecified")
        self.pause_timestamp = utc_now()

    def clear_pause(self) -> None:
        """Drop the pause_reason / pause_timestamp pair. Called when
        the bot transitions OUT of PAUSED (typically to RUNNING)."""
        self.pause_reason = None
        self.pause_timestamp = None

    def record_place_rejection(
        self,
        *,
        outcome: str,
        detail: str,
        is_benign: bool,
        ts_iso: str,
    ) -> None:
        """Record one non-accepted place response into the sticky
        cumulative summary (1.4.6).

        Unlike the orders_lifecycle row buffer (capped at 5000 most-
        recent rows), this summary NEVER decays. Bucketed by the full
        ``detail`` string so distinct venue rejection reasons surface
        independently; cardinality bounded by the venue's distinct
        sMsg set (small in practice).

        Operator visibility on this counter is the primary fix for
        2026-05-17 — exchange errors must never disappear from the
        dashboard just because newer accepted orders pushed them out
        of the most-recent window.
        """
        if not detail:
            detail = f"({outcome} / no detail)"
        bucket_key = str(detail)
        with self._lock:
            entry = self.session_place_reject_summary.get(bucket_key)
            if entry is None:
                entry = {
                    "count": 0,
                    "outcome": str(outcome),
                    "is_benign": bool(is_benign),
                    "first_seen_ts": ts_iso,
                    "last_seen_ts": ts_iso,
                }
                self.session_place_reject_summary[bucket_key] = entry
            entry["count"] = int(entry["count"]) + 1  # type: ignore[arg-type]
            entry["last_seen_ts"] = ts_iso
            self.session_place_reject_count_total += 1

    def record_cancel_rejection(
        self,
        *,
        outcome: str,
        detail: str,
        is_benign: bool,
        ts_iso: str,
    ) -> None:
        """Sibling of :meth:`record_place_rejection` for cancels.

        Cancels have their own benign / unexpected-gone / error
        taxonomy (see :mod:`app.exchange.okx_responses`); callers
        classify each non-success cancel response and pass the
        bucketed detail string here.
        """
        if not detail:
            detail = f"({outcome} / no detail)"
        bucket_key = str(detail)
        with self._lock:
            entry = self.session_cancel_reject_summary.get(bucket_key)
            if entry is None:
                entry = {
                    "count": 0,
                    "outcome": str(outcome),
                    "is_benign": bool(is_benign),
                    "first_seen_ts": ts_iso,
                    "last_seen_ts": ts_iso,
                }
                self.session_cancel_reject_summary[bucket_key] = entry
            entry["count"] = int(entry["count"]) + 1  # type: ignore[arg-type]
            entry["last_seen_ts"] = ts_iso
            self.session_cancel_reject_count_total += 1

    def record_place_outcome(
        self,
        *,
        outcome: str,
        latency_ms: Optional[float],
    ) -> None:
        """Cumulative per-outcome place counter + latency moments
        (1.4.6). Bumps the count for ``outcome``; updates min/max/sum
        if ``latency_ms`` is a finite non-negative number.

        Counterpart for the dashboard connectivity header so total
        placements / ack rate stay accurate after the 5000-row
        recent-window evicts older orders. ``record_place_rejection``
        still fires separately for the per-detail breakdown — this
        method is the per-outcome aggregate.
        """
        out = str(outcome or "unknown")
        with self._lock:
            entry = self.session_place_outcome_counts.get(out)
            if entry is None:
                entry = {
                    "count": 0.0,
                    "min_ms": float("inf"),
                    "max_ms": float("-inf"),
                    "sum_ms": 0.0,
                    "samples_with_latency": 0.0,
                }
                self.session_place_outcome_counts[out] = entry
            entry["count"] += 1.0
            if latency_ms is not None:
                try:
                    lm = float(latency_ms)
                except (TypeError, ValueError):
                    lm = float("nan")
                if math.isfinite(lm) and lm >= 0.0:
                    entry["sum_ms"] += lm
                    entry["samples_with_latency"] += 1.0
                    if lm < entry["min_ms"]:
                        entry["min_ms"] = lm
                    if lm > entry["max_ms"]:
                        entry["max_ms"] = lm

    def record_cancel_outcome(
        self,
        *,
        outcome: str,
        latency_ms: Optional[float],
    ) -> None:
        """Sibling of :meth:`record_place_outcome` for cancels."""
        out = str(outcome or "unknown")
        with self._lock:
            entry = self.session_cancel_outcome_counts.get(out)
            if entry is None:
                entry = {
                    "count": 0.0,
                    "min_ms": float("inf"),
                    "max_ms": float("-inf"),
                    "sum_ms": 0.0,
                    "samples_with_latency": 0.0,
                }
                self.session_cancel_outcome_counts[out] = entry
            entry["count"] += 1.0
            if latency_ms is not None:
                try:
                    lm = float(latency_ms)
                except (TypeError, ValueError):
                    lm = float("nan")
                if math.isfinite(lm) and lm >= 0.0:
                    entry["sum_ms"] += lm
                    entry["samples_with_latency"] += 1.0
                    if lm < entry["min_ms"]:
                        entry["min_ms"] = lm
                    if lm > entry["max_ms"]:
                        entry["max_ms"] = lm

    def record_gate_firing(
        self,
        gate_name: str,
        *,
        firing_now: bool,
        now_mono: float,
    ) -> None:
        """Per-tick gate-firing telemetry recorder (1.4.7 Phase 0).

        Call EVERY tick for EVERY instrumented gate. The function
        detects rising/falling edges and accumulates fire-seconds.

        Semantics:
          * Rising edge (was inactive, now firing): bump fire_count,
            stamp active_since_mono.
          * Falling edge (was firing, now inactive): accumulate
            (now_mono - active_since_mono) into fire_seconds_total,
            clear active_since_mono.
          * Steady state (no change): no-op. The snapshot method
            includes the in-flight active duration so the operator
            sees real-time active time without waiting for the
            falling edge.

        The shape is designed so that the dashboard can render
        per-gate fire rate (fire_count / session_duration) AND
        per-gate "dark time" share (fire_seconds_total /
        session_duration). The latter is the headline metric for
        the Phase 0 baseline report.
        """
        name = str(gate_name or "unknown")
        with self._lock:
            entry = self.session_gate_fire_stats.get(name)
            if entry is None:
                entry = {
                    "fire_count": 0,
                    "fire_seconds_total": 0.0,
                    "active_now": False,
                    "active_since_mono": None,
                }
                self.session_gate_fire_stats[name] = entry
            was_active = bool(entry["active_now"])
            if firing_now and not was_active:
                # Rising edge.
                entry["fire_count"] = int(entry["fire_count"]) + 1
                entry["active_since_mono"] = float(now_mono)
                entry["active_now"] = True
            elif (not firing_now) and was_active:
                # Falling edge.
                started = entry.get("active_since_mono")
                if isinstance(started, (int, float)):
                    delta = max(0.0, float(now_mono) - float(started))
                    entry["fire_seconds_total"] = (
                        float(entry["fire_seconds_total"]) + delta
                    )
                entry["active_since_mono"] = None
                entry["active_now"] = False
            # Steady state: no update.

    def gate_attribution_snapshot(
        self, *, now_mono: float
    ) -> dict[str, dict[str, Any]]:
        """JSON-safe per-gate fire summary for dashboard publisher
        (1.4.7 Phase 0).

        For gates currently firing, includes the in-flight active
        duration in ``fire_seconds_total`` so the operator sees
        real-time accrual.
        """
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            for name, entry in self.session_gate_fire_stats.items():
                fire_seconds = float(entry.get("fire_seconds_total", 0.0))
                active_since = entry.get("active_since_mono")
                if entry.get("active_now") and isinstance(
                    active_since, (int, float)
                ):
                    fire_seconds += max(
                        0.0, float(now_mono) - float(active_since)
                    )
                out[name] = {
                    "fire_count": int(entry.get("fire_count", 0)),
                    "fire_seconds_total": fire_seconds,
                    "active_now": bool(entry.get("active_now", False)),
                }
        return out

    def outcome_aggregates_snapshot(self) -> dict[str, Any]:
        """Return JSON-safe per-outcome aggregates for the dashboard
        publisher. Empties out the special inf/-inf sentinels so the
        wire payload stays clean.
        """
        def _clean(
            src: dict[str, dict[str, float]],
        ) -> dict[str, dict[str, Any]]:
            out: dict[str, dict[str, Any]] = {}
            for k, v in src.items():
                count = int(v.get("count", 0.0))
                samples = int(v.get("samples_with_latency", 0.0))
                row: dict[str, Any] = {"count": count}
                if samples > 0:
                    row["min_ms"] = float(v["min_ms"])
                    row["max_ms"] = float(v["max_ms"])
                    row["mean_ms"] = float(v["sum_ms"]) / samples
                    row["samples_with_latency"] = samples
                else:
                    row["min_ms"] = None
                    row["max_ms"] = None
                    row["mean_ms"] = None
                    row["samples_with_latency"] = 0
                out[k] = row
            return out

        with self._lock:
            return {
                "place": _clean(self.session_place_outcome_counts),
                "cancel": _clean(self.session_cancel_outcome_counts),
            }

    def reject_summary_snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe copy of both sticky reject summaries
        for the dashboard publisher.

        Format:
          {
            "place": {"<detail>": {count, outcome, is_benign,
                                   first_seen_ts, last_seen_ts}, ...},
            "cancel": {...},
            "place_total": int,
            "cancel_total": int,
          }
        """
        with self._lock:
            place = {
                k: dict(v) for k, v in self.session_place_reject_summary.items()
            }
            cancel = {
                k: dict(v) for k, v in self.session_cancel_reject_summary.items()
            }
            return {
                "place": place,
                "cancel": cancel,
                "place_total": int(self.session_place_reject_count_total),
                "cancel_total": int(self.session_cancel_reject_count_total),
            }

    def bump_execution_errors(self, source: str = "unspecified") -> None:
        """Record one execution-side error.

        ``source`` is a short stable tag (e.g. ``"cancel_http_exchange_reject"``)
        that identifies which call path incremented the counter. It is used
        both for kill-time diagnostics (payload breakdown) and for windowed
        evaluation in :meth:`execution_errors_window_snapshot`.
        """
        now_mono = _clock.monotonic()
        tag = str(source) if source else "unspecified"
        with self._lock:
            self.execution_errors += 1
            self._execution_error_events.append((now_mono, tag))

    def _execution_errors_snapshot_unlocked(
        self, window_seconds: float
    ) -> dict[str, Any]:
        """Same as :meth:`execution_errors_window_snapshot` but assumes the
        caller already holds ``self._lock``. Used from ``snapshot_dict`` to
        avoid re-entering the RLock."""
        now_mono = _clock.monotonic()
        ws = max(0.0, float(window_seconds))
        total = int(self.execution_errors)
        if ws <= 0.0:
            sources: dict[str, int] = {}
            for _, src in self._execution_error_events:
                sources[src] = sources.get(src, 0) + 1
            return {
                "total": total,
                "windowed": total,
                "window_seconds": 0.0,
                "sources": sources,
            }
        cutoff = now_mono - ws
        sources = {}
        windowed = 0
        for ts, src in self._execution_error_events:
            if ts >= cutoff:
                windowed += 1
                sources[src] = sources.get(src, 0) + 1
        return {
            "total": total,
            "windowed": windowed,
            "window_seconds": ws,
            "sources": sources,
        }

    def execution_errors_window_snapshot(
        self, window_seconds: float
    ) -> dict[str, Any]:
        """Return a lifetime + windowed view of execution-error bumps.

        When ``window_seconds <= 0`` the windowed count equals the lifetime
        count (opt-out of rolling-window semantics; legacy cumulative
        behaviour). Otherwise only bumps within the last ``window_seconds``
        contribute to ``windowed`` and to the ``sources`` breakdown.
        """
        with self._lock:
            return self._execution_errors_snapshot_unlocked(window_seconds)

    def kill_state_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "killed": self.killed,
                "kill_reason": self.kill_reason,
                "kill_timestamp": self.kill_timestamp.isoformat()
                if self.kill_timestamp
                else None,
            }

    def position_dict(self) -> dict[str, Any]:
        """Thread-safe copy for API (avoids races with the bot loop)."""
        with self._lock:
            p = self.position
            return {
                "symbol": p.symbol,
                "position_qty": p.position_qty,
                "avg_entry_price": p.avg_entry_price,
                "mark_price": p.mark_price,
                "position_notional": p.position_notional,
                "unrealized_pnl_usd": p.unrealized_pnl_usd,
            }

    def pnl_dict(self) -> dict[str, Any]:
        with self._lock:
            p = self.pnl
            return {
                "realized_pnl_usd": p.realized_pnl_usd,
                "unrealized_pnl_usd": p.unrealized_pnl_usd,
                "total_pnl_usd": p.total_pnl_usd,
                "fees_usd": p.fees_usd,
                "equity_usd": p.equity_usd,
                "drawdown_usd": p.drawdown_usd,
                "session_peak_equity_usd": p.session_peak_equity_usd,
                "ts": p.ts.isoformat(),
                "equity_available": p.equity_usd is not None,
                "withdrawable_usd": self.account.withdrawable_usd
                if self.account
                else None,
                "session_started_at_utc": self.session_started_at_utc.isoformat(),
                "realized_source_note": (
                    "Session-scoped: sum of Hyperliquid closedPnl per fill with ts_fill >= "
                    "session_started_at_utc only; fees_usd likewise. Older replay fills are deduped "
                    "but excluded from these session totals."
                ),
            }

    def status_flags_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "bot_status": self.bot_status.value,
                "killed": self.killed,
                "flatten_mode": self.flatten_mode,
                "manual_pause": self.manual_pause,
                # Multi-minute mid drift (BUGS/bug-002.md). None until
                # the long-window deque has accumulated enough samples
                # for the gate to evaluate. Surfaced here so the
                # Telegram /status reply can show "is the bot in a
                # trending regime right now" at a glance.
                "mid_return_long_window_bps": (
                    self.quote_eligibility_snapshot_dict.get(
                        "mid_return_long_window_bps"
                    )
                ),
                "last_heartbeat": self.last_heartbeat.isoformat()
                if self.last_heartbeat
                else None,
                "last_market_ts": self.last_market_ts.isoformat()
                if self.last_market_ts
                else None,
                "book_age_seconds": self.book_age_seconds,
                "stale_book_warning_active": self.stale_book_warning_active,
                "order_desync": self.order_desync,
                "desync_phase": self.desync_phase.value,
                # Fast-start mode: decouple startup readiness from historical fill replay.
                "fast_start_mode_enabled": self.fast_start_mode_enabled,
                "startup_historical_fill_replay_skipped": self.startup_historical_fill_replay_skipped,
                "startup_ready_without_fill_replay": self.startup_ready_without_fill_replay,
                "startup_rest_fill_replay_skipped_count": self.startup_rest_fill_replay_skipped_count,
                "startup_private_snapshot_fills_skipped_count": self.startup_private_snapshot_fills_skipped_count,
                "desync_quarantine_remaining": self.desync_quarantine_remaining,
                "flatten_incomplete": self.flatten_incomplete,
                "flatten_residual_abs_qty": self.flatten_residual_abs_qty,
                "market_data_available": self.market is not None
                and self.market.mid_price is not None
                and self.market.mid_price > 0,
                "account_data_available": self.account is not None,
                "trades_last_minute": self._trade_count_last_60s_unlocked(),
                "blind_resting_seconds_total": round(self.blind_resting_seconds_total, 3),
                "blind_resting_tick_count": self.blind_resting_tick_count,
                # TODO-001: inventory consistency watchdog visibility.
                "inventory_baseline_qty": round(self.inventory_baseline_qty, 6),
                "session_signed_qty_total": round(self.session_signed_qty_total, 6),
                "inventory_consistency_breach_count": self.inventory_consistency_breach_count,
                "inventory_consistency_last_breach": self.inventory_consistency_last_breach,
                # TODO-003: rolling net-edge-after-fees over last N fills.
                "net_edge_summary": build_net_edge_summary(
                    list(self.recent_fills),
                    window=int(self._settings.net_edge_window_fills),
                ),
                # amend-prio Phase 4 (v1.4.17): per-outcome amend
                # counters for the operator-facing /status line.
                # Telegram surfaces a single-line summary when
                # ``amend_intents_emitted_total > 0`` and hides the
                # row entirely when the knob is off (= no traffic).
                "amend_intents_emitted_total": int(
                    self.amend_intents_emitted_total
                ),
                "amend_success_total": int(self.amend_success_total),
                "amend_below_filled_total": int(
                    self.amend_below_filled_total
                ),
                "amend_order_gone_total": int(self.amend_order_gone_total),
                "amend_post_only_cross_total": int(
                    self.amend_post_only_cross_total
                ),
                "amend_exchange_rejected_other_total": int(
                    self.amend_exchange_rejected_other_total
                ),
                "amend_transport_rejected_total": int(
                    self.amend_transport_rejected_total
                ),
                "amend_pending_high_watermark": int(
                    self.amend_pending_high_watermark
                ),
                # v1.4.93 wedge-elimination-cleanup Phase 6D.3 —
                # operator-facing "did the bot wedge at all this session"
                # counter. Derived from the risk-exec-state-machine's
                # transition history in executor_state_snapshot; reads
                # safely as 0 before the first transition. Surfaced on
                # Telegram /status so the operator can see at a glance
                # whether any non-trivial risk-state entry occurred.
                "wedge_episode_count_session": int(
                    (self.executor_state_snapshot or {}).get(
                        "wedge_episode_count_session", 0
                    ) or 0
                ),
                # Market-regime block: enough state for an operator (or LLM)
                # to read a /status message and reason about the regime
                # without pulling klines. The values mirror what already
                # lives in ``snapshot_dict`` but are surfaced here so
                # ``/status`` doesn't need a second call.
                "short_vol_bps": self.vol_bps,
                # v1.5.239 — trend-aware EWMA-of-|return| vol measure,
                # in bp. Same Optional[float] semantics as short_vol_bps
                # (None during the estimator's 2-mid warm-up). Always
                # populated regardless of REGIME_FORWARD_USE_VOL_ABS_EWMA_FOR_SLOPE
                # so operators can compare both measures side-by-side
                # for calibration without flipping the consumer flag.
                "short_vol_abs_ewma_bps": self.vol_abs_ewma_bps,
                "venue_spread_bps": (
                    self.market.spread_bps if self.market is not None else None
                ),
                "basis_to_binance_bps": (
                    None
                    if self.binance_basis_ewma is None
                    else float(self.binance_basis_ewma) * 10_000.0
                ),
                "active_sides": self.last_active_sides,
                "book_age_ms": (
                    None
                    if self.book_age_seconds is None
                    else float(self.book_age_seconds) * 1000.0
                ),
                "position_notional_usd": float(self.position.position_notional),
                "max_position_notional_usd": float(
                    self._settings.max_position_notional_usd
                ),
                "position_notional_pct_of_cap": (
                    None
                    if float(self._settings.max_position_notional_usd) <= 0
                    else round(
                        100.0
                        * float(self.position.position_notional)
                        / float(self._settings.max_position_notional_usd),
                        1,
                    )
                ),
                "toxicity_score": build_runtime_toxicity_summary(
                    list(self.recent_fills),
                    window=int(self._settings.runtime_toxicity_fill_window),
                    session_id=self.session_id,
                ).get("toxicity_score"),
                "latency_fill_to_process_ms": self.last_latency_fill_to_process_ms,
                "exchange_ts_to_decision_ms": self.last_exchange_ts_to_decision_ms,
                "latency_decision_to_first_place_ms": self.last_latency_decision_to_first_place_ms,
                "latency_tick_preamble_ms": self.latency_tick_preamble_ms,
                "latency_hot_path_local_compute_ms": self.latency_hot_path_local_compute_ms,
                "latency_quote_engine_build_ms": self.latency_quote_engine_build_ms,
                "latency_order_maintenance_local_ms": self.latency_order_maintenance_local_ms,
                "latency_account_refresh_rest_ms": self.latency_account_refresh_rest_ms,
                "latency_reconcile_rest_ms": self.latency_reconcile_rest_ms,
                "latency_order_submit_rtt_ms": self.latency_order_submit_rtt_ms,
                "latency_private_queue_wait_ms": self.latency_private_queue_wait_ms,
                "latency_public_ws_queue_wait_ms": self.latency_public_ws_queue_wait_ms,
                "private_ws_connected": self.private_ws_connected,
                "private_ws_healthy": self.private_ws_healthy,
                "private_ws_queue_drops": self.private_ws_queue_drops,
                "private_ws_recovery_pending": self.private_ws_recovery_pending,
                "private_ws_last_connect_ts": self.private_ws_last_connect_wall_ts.isoformat()
                if self.private_ws_last_connect_wall_ts
                else None,
                "private_ws_last_ping_sent_ts": self.private_ws_last_ping_sent_wall_ts.isoformat()
                if self.private_ws_last_ping_sent_wall_ts
                else None,
                "private_ws_last_pong_ts": self.private_ws_last_pong_wall_ts.isoformat()
                if self.private_ws_last_pong_wall_ts
                else None,
                "private_ws_last_message_ts": self.private_ws_last_message_wall_ts.isoformat()
                if self.private_ws_last_message_wall_ts
                else None,
                "private_ws_seconds_since_last_message": self._private_ws_seconds_since_last_message_unlocked(),
                "private_ws_disconnect_reason_last": self.private_ws_disconnect_reason_last,
                "private_ws_disconnect_close_code_last": self.private_ws_disconnect_close_code_last,
                "private_ws_disconnect_histogram": dict(self.private_ws_disconnect_histogram),
                "private_ws_reconnect_count": self.private_ws_reconnect_count,
                "private_ws_reconnect_reason_counts": dict(self.private_ws_reconnect_reason_counts),
                "private_ws_queue_high_watermark": self.private_ws_queue_high_watermark,
                "private_ws_queue_overflow_count": self.private_ws_queue_drops,
                "private_ws_queue_backlog_after_drain_count": self.private_ws_queue_backlog_after_drain_count,
                "private_ws_max_events_drained_per_tick": self.private_ws_max_events_drained_per_tick,
                "private_ws_events_drained_last_tick": self.private_ws_events_drained_last_tick,
                "private_ws_queue_depth_after_drain": self.private_ws_queue_depth_after_drain,
                "private_ws_drain_time_ms_last_tick": self.private_ws_drain_time_ms_last_tick,
                "private_ws_last_inbound_derived_ms": dict(self.private_ws_last_inbound_derived_ms),
                # OKX-only diagnostics (zeros / empty dict on other venues).
                "okx_ws_orders_msgs_session": self.okx_ws_orders_msgs_session,
                "okx_ws_orders_msgs_by_state": dict(self.okx_ws_orders_msgs_by_state),
                "okx_ws_pong_gap_seconds_max": self.okx_ws_pong_gap_seconds_max,
                # Tape runtime-feed (shmem IPC) health + consumer-fire
                # counters (M7.6 / M8.4). All zero when
                # REGIME_USE_RUNTIME_RECORDER_FEED=false (the default) —
                # byte-identical surface to pre-M7 snapshots in that case.
                # The quote loop MIRRORS the reader's monotonic counters
                # by assignment each tick; the seeded / fires counters are
                # bumped directly by the consumers.
                "runtime_feed_read_count": self.runtime_feed_read_count,
                "runtime_feed_stale_count": self.runtime_feed_stale_count,
                "runtime_feed_version_mismatch_count": self.runtime_feed_version_mismatch_count,
                "runtime_feed_collision_count": self.runtime_feed_collision_count,
                "warmstart_vol_seeded_from_recorder_count": self.warmstart_vol_seeded_from_recorder_count,
                "microprice_widen_z_runtime_feed_fires": self.microprice_widen_z_runtime_feed_fires,
                "public_ws_events_applied_last_tick": self.public_ws_events_applied_last_tick,
                "public_ws_max_burst_per_tick": self.public_ws_max_burst_per_tick,
                "public_ws_last_inbound_derived_ms": dict(self.public_ws_last_inbound_derived_ms),
                "public_ws_connected": self.public_ws_connected,
                "public_ws_last_message_wall_ts": self.public_ws_last_message_wall_ts.isoformat()
                if self.public_ws_last_message_wall_ts
                else None,
                "public_ws_reconnect_count": self.public_ws_reconnect_count,
                "public_ws_seen_first_bbo": self.public_ws_seen_first_bbo,
                # 1.2.25: session-cumulative BBO + mid-change counts
                # for todo-010 Phase 2 decision data. Divide by
                # session duration to get rates.
                "bbo_event_count_session": self.bbo_event_count_session,
                "mid_change_count_session": self.mid_change_count_session,
                # v1.5.244 — rolling 5-min rate. The binding constraint
                # on fill rate (fills ≈ 0.10 × mid_change_rate). Below
                # ~6/min the bot likely fails the 25-fills / 30-min
                # calibration validity gate.
                "mid_change_rate_per_min_5m": self.mid_change_rate_per_min_5m(),
                "live_market_data_source": self.live_market_data_source,
                "exchange_snapshot_unhealthy_streak": self.exchange_snapshot_unhealthy_streak,
                "reconcile_auto_pause": self.reconcile_auto_pause,
                "recovering_market_data": self.bot_status == BotStatus.RECOVERING_MARKET_DATA,
                "quote_eligibility_guard": dict(self.quote_eligibility_snapshot_dict),
                "quote_eligibility_recovery_floor": (
                    self.quote_elig_recovery_floor.value
                    if self.quote_elig_recovery_floor is not None
                    else None
                ),
                "quote_eligibility_recovery_remaining_ms": float(
                    self.quote_elig_recovery_remaining_ms
                ),
                "quote_elig_hold_all_count": self.quote_elig_hold_all_count,
                "quote_elig_buy_only_count": self.quote_elig_buy_only_count,
                "quote_elig_sell_only_count": self.quote_elig_sell_only_count,
                "quote_elig_hold_due_stale_count": self.quote_elig_hold_due_stale_count,
                "quote_elig_hold_due_jump_count": self.quote_elig_hold_due_jump_count,
                "quote_elig_one_sided_due_drift_count": self.quote_elig_one_sided_due_drift_count,
                "quote_elig_one_sided_due_freshness_count": self.quote_elig_one_sided_due_freshness_count,
                "quote_elig_resume_count": self.quote_elig_resume_count,
                "action_dispatch_queue_depth": self.outbound_action_queue_depth,
                "action_dispatch_queue_high_watermark": self.outbound_action_queue_hwm,
                "avg_batch_size": self.outbound_avg_batch_size,
                "ws_action_send_count": self.outbound_ws_action_send_count,
                "http_action_send_count": self.outbound_http_action_send_count,
                "coalesced_intent_count": self.outbound_coalesced_intent_count,
                "dropped_stale_intent_count": self.outbound_dropped_stale_intent_count,
                "place_intent_to_ack_ms": self.outbound_place_intent_to_ack_ms,
                "cancel_intent_to_closed_ms": self.outbound_cancel_intent_to_closed_ms,
                "ack_to_private_ws_lifecycle_ms": self.outbound_ack_to_private_ws_lifecycle_ms,
                "signing_latency_ms": self.outbound_signing_latency_ms,
                "action_batch_wait_ms": self.outbound_action_batch_wait_ms,
                "transport_mode_current": self.outbound_transport_mode_current,
                "quote_cycle_to_first_transport_send_ms": self.outbound_quote_cycle_to_first_transport_send_ms,
                "quote_cycle_to_first_ack_ms": self.outbound_quote_cycle_to_first_ack_ms,
                **dict(self.exec_runtime_counters),
                **self._market_data_metrics_unlocked(),
                **self._binance_ws_status_dict_unlocked(),
            }

    def set_manual_pause(self, paused: bool) -> bool:
        """Set ``manual_pause``. Returns ``True`` iff this call transitioned
        from True → False — the "operator just resumed after a pause"
        edge.

        On that transition we rebaseline the TODO-001 inventory consistency
        watchdog: ``inventory_baseline_qty`` is reset to the current venue
        truth (``state.position.position_qty``) and
        ``session_signed_qty_total`` is zeroed. This lets the operator's
        natural intervention flow — ``/pause`` → close manually on the
        venue UI → ``/resume`` — work without the watchdog firing on the
        legitimate divergence between expected and venue inventory.

        Returns ``True`` only on True→False so callers can log the
        rebaseline as a distinct audit event.
        """
        with self._lock:
            prev = self.manual_pause
            self.manual_pause = bool(paused)
            transitioned_to_unpaused = bool(prev) and not bool(paused)
            if transitioned_to_unpaused:
                try:
                    qty = float(
                        getattr(self.position, "position_qty", 0.0) or 0.0
                    )
                except (TypeError, ValueError):
                    qty = 0.0
                self.inventory_baseline_qty = qty
                self.inventory_baseline_set = True
                self.session_signed_qty_total = 0.0
                # Reset the watchdog rate-limit anchor so the next check
                # evaluates fresh against the new baseline (rather than
                # waiting up to a full interval to confirm consistency).
                self.inventory_consistency_last_check_mono = None
                self.inventory_consistency_last_breach = None
                # Also reset the consecutive-drift counter — operator
                # has just re-synced expected to venue, so any prior drift
                # streak is irrelevant going forward.
                self.inventory_consistency_consecutive_drift_count = 0
            # NOTE: ``bot_status`` is NOT flipped here. The risk gate
            # reads ``manual_pause`` directly (see ``risk.py``) and
            # blocks quoting via NO_QUOTE without a status enum
            # transition. The original ``if paused: bot_status =
            # PAUSED`` branch lives further down but is unreachable
            # (return above) -- leaving as-is to avoid scope-creep.
            # We do tag the pause reason for heartbeat visibility:
            if paused:
                self.mark_paused("manual_pause")
            else:
                # Only clear if the prior reason was manual_pause; we
                # don't want operator /resume to wipe a reconcile-
                # stall or post-flatten reason that's still active.
                if self.pause_reason == "manual_pause":
                    self.clear_pause()
            return transitioned_to_unpaused
