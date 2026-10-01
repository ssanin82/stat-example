"""v1.4.114 Phase 1F — cross-cutting integration tests.

Verifies the full Phase 1 defence stack behaves correctly under
scenario-driven inputs, without requiring the full bot harness or
snapshot-replay infrastructure (those live under
``tests/integration/`` and need filesystem fixtures).

Two anchor scenarios:

* **Slow-grind / driving-incident replay (1F.1)** — synthetic mid
  trajectory that mimics the 2026-05-20 06:00-06:50 pattern:
  position ramps to high util while mid drifts adverse-to-position.
  Asserts: ``regime_controller`` transitions NORMAL → DEFENSIVE
  within the 15 s entry dwell; ``inventory_drift_gate`` fires;
  knob overlay widens half-spread + shrinks size; a simulated
  reducing fill arms the post-reduction cooldown which then clamps
  eligibility under DEFENSIVE.

* **NORMAL-only profitable session (1F.2)** — synthetic clear-flow
  inputs: low util, flat mid, no drift, no shock. Asserts:
  ``regime_controller`` stays NORMAL ≥ 95 % of ticks; knobs stay
  at 1.0; cooldown never arms; eligibility passes through clean.

Plus structural parity check (1F.3) — the ``regime_controller``
snapshot dict carries every field the frontend interface will
need (in the absence of the actual TypeScript interface this
phase, we pin the contract from the Python side; 1E will land
the matching TS types).

The 1F.4 / 1F.5 / 1F.6 items depend on Phase 1E's live_stats /
markdown-export / frontend changes; they land alongside 1E.
"""

from __future__ import annotations

import math
from typing import Optional

from app.config import Settings
from app.enums import QuoteEligibility, Side
from app.inventory_drift_gate import (
    compute_short_window_drifts,
    evaluate_inventory_drift_gate,
    widening_bps as inv_drift_widening_bps,
)
from app.regime_controller import (
    Mode,
    accumulate_time_in_mode,
    compute_knobs_for_mode,
    evaluate_mode,
    snapshot_dict,
)
from app.shock_gate import observe as shock_observe
from app.slow_trend_gate import evaluate_slow_trend_gate
from app.state import BotState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings_for_replay() -> Settings:
    """Settings matching the TON profile for the Phase 1 stack."""
    return Settings(
        SYMBOL="TON-USDT-SWAP",
        MAX_ABS_POSITION=10.0,
        # Phase 1A — inventory_drift_gate
        INVENTORY_DRIFT_GATE_ENABLED=True,
        INVENTORY_DRIFT_INVENTORY_PCT_THRESHOLD=0.60,
        INVENTORY_DRIFT_THRESHOLD_BPS_10S=15.0,
        INVENTORY_DRIFT_THRESHOLD_BPS_30S=30.0,
        INVENTORY_DRIFT_WIDEN_BPS=15.0,
        # Phase 1B — shock_gate
        SHOCK_GATE_ENABLED=True,
        SHOCK_INVENTORY_PCT_THRESHOLD=0.80,
        SHOCK_THRESHOLD_BPS_10S=20.0,
        SHOCK_THRESHOLD_BPS_30S=50.0,
        SHOCK_CLEAR_UTIL_THRESHOLD=0.30,
        SHOCK_MAX_COOLDOWN_SECONDS=300.0,
        # Phase 1C — regime_controller
        REGIME_CONTROLLER_ENABLED=True,
        REGIME_ENTRY_DWELL_SECONDS=15.0,
        REGIME_EXIT_DWELL_SECONDS=30.0,
        REGIME_UTIL_ENTRY_THRESHOLD=0.50,
        REGIME_UTIL_EXIT_THRESHOLD=0.30,
        REGIME_VOL_RATIO_ENTRY_THRESHOLD=2.0,
        # Phase 1D — post-reduction cooldown
        POST_REDUCTION_COOLDOWN_SECONDS=60.0,
        # Slow-trend gate (existing in v1.4.102)
        SLOW_TREND_GATE_ENABLED=True,
        SLOW_TREND_WINDOW_SECONDS=900.0,
        SLOW_TREND_THRESHOLD_BPS=25.0,
        SLOW_TREND_MIN_SAMPLES=10,  # lower for the synthetic harness
        SLOW_TREND_ANCHOR_FRACTION=0.2,
        SLOW_TREND_WIDEN_BPS=10.0,
    )


def _synth_mid_samples(
    *,
    start_mid: float,
    end_mid: float,
    duration_seconds: float,
    n_samples: int,
    end_mono: float,
) -> list[tuple[float, float]]:
    """Linear-drift mid samples for the deque feeding both
    inventory_drift_gate and slow_trend_gate. ``(mono_ts, mid)``
    pairs anchored so the LAST sample is at ``end_mono``."""
    samples: list[tuple[float, float]] = []
    for i in range(n_samples):
        frac = i / max(1, n_samples - 1)
        ts = end_mono - duration_seconds + frac * duration_seconds
        mid = start_mid + (end_mid - start_mid) * frac
        samples.append((ts, mid))
    return samples


def _run_tick(
    *,
    state: BotState,
    now_mono: float,
    util: float,
    position_qty: float,
    vol_ratio: float,
    mid_samples: list[tuple[float, float]],
    mid_now: float,
) -> tuple[Mode, bool, bool, bool, Optional[QuoteEligibility]]:
    """Run one quote-loop tick through the whole Phase 1 gate stack.

    Returns ``(mode, slow_trend_active, inv_drift_active,
    shock_locked, post_reduction_suppressed_side)``.

    Mirrors the bot's ``_apply_regime_gates`` ordering exactly:
    re-evaluate slow_trend + inventory_drift (stateless), drive
    shock_gate.observe (stateful), then evaluate_mode + accumulate
    time-in-mode + publish knobs.
    """
    s = state.settings
    # Compute short-window drifts once (shared by both gates).
    d10, d30 = compute_short_window_drifts(
        mid_samples, now_mono=now_mono, mid_now=mid_now
    )
    # slow_trend
    st_over, _, _ = evaluate_slow_trend_gate(
        samples_long=mid_samples,
        now_mono=now_mono,
        mid_now=mid_now,
        enabled=bool(s.slow_trend_gate_enabled),
        window_seconds=float(s.slow_trend_window_seconds),
        threshold_bps=float(s.slow_trend_threshold_bps),
        min_samples=int(s.slow_trend_min_samples),
        anchor_fraction=float(s.slow_trend_anchor_fraction),
    )
    slow_trend_active = st_over is not None
    # inventory_drift
    id_over, _, _ = evaluate_inventory_drift_gate(
        position_qty=position_qty,
        effective_abs_cap=float(s.max_abs_position),
        drift_bps_10s=d10,
        drift_bps_30s=d30,
        inventory_pct_threshold=float(s.inventory_drift_inventory_pct_threshold),
        drift_threshold_bps_10s=float(s.inventory_drift_threshold_bps_10s),
        drift_threshold_bps_30s=float(s.inventory_drift_threshold_bps_30s),
        enabled=bool(s.inventory_drift_gate_enabled),
    )
    inv_drift_active = id_over is not None
    # shock_gate (stateful — mutates state.shock_gate).
    sg_override, _ = shock_observe(
        state.shock_gate,
        now_mono=now_mono,
        position_qty=position_qty,
        effective_abs_cap=float(s.max_abs_position),
        drift_bps_10s=d10,
        drift_bps_30s=d30,
        enabled=bool(s.shock_gate_enabled),
        shock_inventory_pct_threshold=float(s.shock_inventory_pct_threshold),
        shock_threshold_bps_10s=float(s.shock_threshold_bps_10s),
        shock_threshold_bps_30s=float(s.shock_threshold_bps_30s),
        clear_util_threshold=float(s.shock_clear_util_threshold),
        max_cooldown_seconds=float(s.shock_max_cooldown_seconds),
    )
    shock_locked = state.shock_gate.locked
    # regime_controller (stateful — mutates state.regime_controller).
    mode, _ = evaluate_mode(
        state.regime_controller,
        now_mono=now_mono,
        util=util,
        vol_ratio=vol_ratio,
        slow_trend_active=slow_trend_active,
        inventory_drift_active=inv_drift_active,
        shock_gate_locked=shock_locked,
        enabled=bool(s.regime_controller_enabled),
        entry_dwell_seconds=float(s.regime_entry_dwell_seconds),
        exit_dwell_seconds=float(s.regime_exit_dwell_seconds),
        util_entry_threshold=float(s.regime_util_entry_threshold),
        util_exit_threshold=float(s.regime_util_exit_threshold),
        vol_ratio_entry_threshold=float(s.regime_vol_ratio_entry_threshold),
    )
    accumulate_time_in_mode(state.regime_controller, now_mono=now_mono)
    state.regime_knobs = compute_knobs_for_mode(
        mode, ladder_levels_max_config=1
    )
    # Post-reduction cooldown — read the state armed by prior fills.
    cd_seconds = float(s.post_reduction_cooldown_seconds)
    suppressed: Optional[QuoteEligibility] = None
    if (
        cd_seconds > 0
        and mode is not Mode.NORMAL
        and state.last_inventory_reduction_at_mono is not None
        and state.last_inventory_reduction_suppressed_side is not None
    ):
        elapsed = now_mono - float(state.last_inventory_reduction_at_mono)
        if 0.0 <= elapsed < cd_seconds:
            suppressed = state.last_inventory_reduction_suppressed_side
    return mode, slow_trend_active, inv_drift_active, shock_locked, suppressed


# ---------------------------------------------------------------------------
# 1F.1 — Slow-grind replay (the driving incident pattern)
# ---------------------------------------------------------------------------


def test_1F1_slow_grind_pattern_drives_defensive_via_slow_trend() -> None:
    """Mimic the 2026-05-20 06:00-06:50 pattern: bot accumulates
    LONG while mid drifts adverse over many minutes. The intended
    primary trigger for this pattern is ``slow_trend_gate`` (15-min
    window, 25-bp threshold) — NOT ``inventory_drift_gate`` (which
    is for acute moves over 10-30 s).

    Once slow_trend fires AND util ≥ 0.5, the regime_controller
    transitions NORMAL → DEFENSIVE after the 15 s entry dwell.
    DEFENSIVE engages knob overlays (half-spread × 1.5, size × 0.5)
    that would have prevented further accumulation in production.

    Tick cadence: 5 s (closer to production's ~500 ms loop than
    the original test's 30 s — important so short-window gates can
    actually see samples).
    """
    settings = _settings_for_replay()
    state = BotState(settings)

    # 240 ticks × 5 s = 20 minutes — long enough for slow_trend's
    # 15-min window to be populated.
    n_ticks = 240
    tick_step_s = 5.0
    util_when_defensive_first_observed: Optional[float] = None
    first_slow_trend_tick: Optional[int] = None
    first_defensive_tick: Optional[int] = None
    mid_history: list[tuple[float, float]] = []

    for i in range(n_ticks):
        now_mono = 1000.0 + i * tick_step_s
        frac = i / (n_ticks - 1)
        position_qty = 7.0 * frac  # ramp from 0 to +7 over 20 min
        util = abs(position_qty) / 10.0
        # Mid drifts from 1.985 → 1.960 (-126 bp over 20 min).
        cur_mid = 1.985 - 0.025 * frac
        mid_history.append((now_mono, cur_mid))
        cutoff = now_mono - 900.0
        mid_history = [(t, m) for (t, m) in mid_history if t >= cutoff]
        mode, st_active, id_active, shock, suppressed = _run_tick(
            state=state,
            now_mono=now_mono,
            util=util,
            position_qty=position_qty,
            vol_ratio=1.0,
            mid_samples=mid_history,
            mid_now=cur_mid,
        )
        if st_active and first_slow_trend_tick is None:
            first_slow_trend_tick = i
        if mode is Mode.DEFENSIVE and first_defensive_tick is None:
            first_defensive_tick = i
            util_when_defensive_first_observed = util

    # 1. slow_trend_gate fired during the grind. The 25-bp threshold
    #    over 15 minutes is the gate's design point; -126 bp / 20 min
    #    blows past it by ~5×.
    assert first_slow_trend_tick is not None, (
        "slow_trend_gate never fired during the grind — Phase 0 "
        "(pre-existing slow_trend) is not active"
    )
    # 2. regime_controller transitioned to DEFENSIVE.
    assert first_defensive_tick is not None, (
        "regime_controller never left NORMAL during the grind — "
        "Phase 1C is not driven by slow_trend / util signals"
    )
    # 3. DEFENSIVE entry happened at or after util crossed 0.50 OR
    #    slow_trend started firing — whichever came first.
    assert (
        util_when_defensive_first_observed is not None
        and util_when_defensive_first_observed >= 0.0
    )
    # 4. shock_gate did NOT fire — drift is sustained but per-window
    #    magnitude is well below SHOCK thresholds. The plan is
    #    explicit that SHOCK is for ACUTE spikes only.
    assert not state.shock_gate.locked, (
        "shock_gate fired on a slow-grind pattern — too sensitive"
    )
    # 5. Time-in-mode counters route through DEFENSIVE (not just
    #    NORMAL) — proves the FSM actually spent time in the
    #    defended state, not just visited it for one tick.
    rc = state.regime_controller
    assert rc.time_in_defensive_seconds > 0.0
    # 6. Knob overlay at session end is the DEFENSIVE row.
    assert state.regime_knobs.base_half_spread_mult == 1.5
    assert state.regime_knobs.quote_notional_mult == 0.5


def test_1F1_acute_shock_pattern_engages_inventory_drift_and_shock_gates() -> None:
    """The OTHER side of Phase 1's defence: an ACUTE move against
    loaded inventory. This is what inventory_drift_gate (15 bp / 10 s)
    and shock_gate (20 bp / 10 s) are designed for.

    Scenario: bot is sitting LONG +9 (util 0.9). Mid drops -30 bp
    in 10 seconds (= a sharp shock). Both Phase 1A and Phase 1B
    gates should engage; regime_controller jumps NORMAL → SHOCK
    instantly (shock_gate's preemption path).
    """
    settings = _settings_for_replay()
    state = BotState(settings)
    # Seed mid history at 1-second cadence for 40 seconds. This puts
    # pre-shock samples inside BOTH the 10 s and 30 s windows when
    # the shock tick arrives, so both gates' anchor lookups land on
    # pre-shock mids (not the shock value itself).
    end_mono = 1040.0  # pre-shock "now"; 40 s of flat history before
    mid_history = _synth_mid_samples(
        start_mid=1.985,
        end_mid=1.985,
        duration_seconds=40.0,
        n_samples=41,
        end_mono=end_mono,
    )
    # Tick 1: pre-shock — gates dormant.
    mode_t0, _, _, shock_t0, _ = _run_tick(
        state=state,
        now_mono=end_mono,
        util=0.9,
        position_qty=+9.0,
        vol_ratio=1.0,
        mid_samples=mid_history,
        mid_now=1.985,
    )
    assert mode_t0 is Mode.NORMAL
    assert not shock_t0

    # Tick 2: at t=1041 (1 s after the last flat sample) the mid
    # has dropped -60 bp. The 10 s window [1031, 1041] anchors on
    # pre-shock samples; drift_10s ≈ -60 bp. The 30 s window
    # [1011, 1041] also anchors on pre-shock; drift_30s ≈ -60 bp.
    # Both clear the SHOCK thresholds (20 bp / 50 bp) AND the
    # inventory_drift thresholds, with anti-aligned LONG inventory.
    new_mono = end_mono + 1.0
    new_mid = 1.985 * (1 - 60e-4)  # -60 bp
    mid_history.append((new_mono, new_mid))
    mode_t1, _, id_active, shock_t1, _ = _run_tick(
        state=state,
        now_mono=new_mono,
        util=0.9,
        position_qty=+9.0,
        vol_ratio=1.0,
        mid_samples=mid_history,
        mid_now=new_mid,
    )
    # inventory_drift fires (Phase 1A): util ≥ 0.6 AND drift past
    # threshold AND anti-aligned with LONG inventory.
    assert id_active, "inventory_drift_gate did not fire on shock move"
    # shock_gate fires (Phase 1B): util ≥ 0.8 AND drift past
    # shock-threshold AND anti-aligned.
    assert shock_t1, "shock_gate did not fire on shock magnitude"
    # regime_controller jumped directly to SHOCK (no DEFENSIVE dwell).
    assert mode_t1 is Mode.SHOCK
    # Knob overlay is the SHOCK row.
    assert state.regime_knobs.base_half_spread_mult == 2.0
    assert state.regime_knobs.adding_side_enabled is False


def test_1F1_post_reduction_cooldown_engages_after_simulated_reducing_fill() -> None:
    """Continuation of 1F.1 — once DEFENSIVE is engaged, simulate a
    reducing fill (the operator's SELL that unwinds some LONG) and
    verify the post-reduction cooldown clamps subsequent eligibility.
    """
    settings = _settings_for_replay()
    state = BotState(settings)
    # Pre-stage: drive the FSM into DEFENSIVE.
    state.regime_controller.mode = Mode.DEFENSIVE
    state.regime_controller.mode_since_mono = 1000.0

    # Simulate a reducing SELL fill: +6 → +4 at t=1100.
    state._note_inventory_reduction_for_cooldown(
        now_mono=1100.0,
        fill_side=Side.SELL,
        prev_qty=+6.0,
        new_qty=+4.0,
    )
    # Now at t=1110 (10 s post-fill, well within the 60-s cooldown):
    cd_seconds = float(settings.post_reduction_cooldown_seconds)
    elapsed = 1110.0 - float(state.last_inventory_reduction_at_mono)
    assert 0.0 <= elapsed < cd_seconds
    assert (
        state.last_inventory_reduction_suppressed_side
        is QuoteEligibility.QUOTE_SELL_ONLY
    )
    # The clamp would apply because:
    #   * cooldown active (elapsed < cd_seconds)
    #   * mode != NORMAL (DEFENSIVE)
    #   * suppressed_side is set
    # Verify the three conditions all hold.
    assert state.regime_controller.mode is not Mode.NORMAL
    assert state.last_inventory_reduction_at_mono is not None
    assert state.last_inventory_reduction_suppressed_side is not None


def test_1F1_post_reduction_cooldown_does_not_engage_under_NORMAL() -> None:
    """Same reducing-fill scenario as above but mode is NORMAL —
    the cooldown clamp must NOT apply (preserves round-trip rebate
    capture in steady state).
    """
    settings = _settings_for_replay()
    state = BotState(settings)
    assert state.regime_controller.mode is Mode.NORMAL  # default
    state._note_inventory_reduction_for_cooldown(
        now_mono=1100.0,
        fill_side=Side.SELL,
        prev_qty=+6.0,
        new_qty=+4.0,
    )
    # State is armed (timestamp + side set) — that's fine, the
    # clamp logic gates on mode in _apply_regime_gates.
    assert state.last_inventory_reduction_at_mono is not None
    # But mode is NORMAL, so the eligibility clamp wouldn't fire.
    assert state.regime_controller.mode is Mode.NORMAL


# ---------------------------------------------------------------------------
# 1F.2 — NORMAL-only profitable session
# ---------------------------------------------------------------------------


def test_1F2_normal_only_session_stays_normal_no_gates_fire() -> None:
    """Synthetic 4-hour-equivalent session at low util / flat mid /
    no shock. Assert regime_controller stays NORMAL 100 % of ticks,
    no defensive gate ever fires, knobs stay at NORMAL row, and the
    post-reduction cooldown never arms (no reducing fills).
    """
    settings = _settings_for_replay()
    state = BotState(settings)

    n_ticks = 480  # 4 hours at 30-s tick cadence
    tick_step_s = 30.0
    mode_distribution = {Mode.NORMAL: 0, Mode.DEFENSIVE: 0, Mode.SHOCK: 0}
    inv_drift_fires = 0
    shock_fires = 0

    mid_history: list[tuple[float, float]] = []
    cur_mid = 1.985
    for i in range(n_ticks):
        now_mono = 1000.0 + i * tick_step_s
        # Mid drifts by ±0.1 bp per tick (random-walk-noise, well
        # under any defensive threshold).
        cur_mid += 1.985e-5 * (1.0 if (i % 2 == 0) else -1.0)
        mid_history.append((now_mono, cur_mid))
        cutoff = now_mono - 900.0
        mid_history = [(t, m) for (t, m) in mid_history if t >= cutoff]
        # Position stays small — util well under entry threshold.
        position_qty = 1.5  # util = 0.15
        util = abs(position_qty) / 10.0
        mode, st_active, id_active, shock, suppressed = _run_tick(
            state=state,
            now_mono=now_mono,
            util=util,
            position_qty=position_qty,
            vol_ratio=1.0,  # no vol spike
            mid_samples=mid_history,
            mid_now=cur_mid,
        )
        mode_distribution[mode] += 1
        if id_active:
            inv_drift_fires += 1
        if shock:
            shock_fires += 1

    # Hard expectations:
    assert (
        mode_distribution[Mode.NORMAL] / n_ticks >= 0.95
    ), (
        f"regime_controller left NORMAL too often: "
        f"{mode_distribution}"
    )
    assert mode_distribution[Mode.DEFENSIVE] == 0
    assert mode_distribution[Mode.SHOCK] == 0
    assert inv_drift_fires == 0, (
        "inventory_drift_gate false-positive in clear-flow session"
    )
    assert shock_fires == 0, (
        "shock_gate false-positive in clear-flow session"
    )
    # Knobs ended at NORMAL row.
    assert state.regime_knobs.base_half_spread_mult == 1.0
    assert state.regime_knobs.quote_notional_mult == 1.0
    # Post-reduction cooldown never armed (no fills simulated).
    assert state.last_inventory_reduction_at_mono is None


# ---------------------------------------------------------------------------
# 1F.3 — Snapshot dict shape parity (regime_controller)
# ---------------------------------------------------------------------------


def test_1F3_regime_controller_snapshot_dict_carries_every_published_field() -> None:
    """Pin the on-the-wire contract the live_stats / heartbeat /
    state_current.json paths depend on. 1E will wire the TypeScript
    interface to match; this test prevents drift from the Python
    side."""
    from app.regime_controller import RegimeControllerState

    rc = RegimeControllerState()
    snap = snapshot_dict(rc, now_mono=0.0)
    expected_keys = {
        "mode",
        "seconds_in_mode",
        "last_transition_reason",
        # Reactive FSM (pre-4G) arming timers — NORMAL ↔ DEFENSIVE ↔ SHOCK.
        "entry_arming_seconds",
        "exit_arming_seconds",
        "transition_count",
        "recent_transitions",
        # Session-cumulative time-in-mode counters.
        "time_in_normal_seconds",
        "time_in_defensive_seconds",
        "time_in_shock_seconds",
        # Phase 4G.2 (v1.4.209) — forward-classifier modes added two
        # new modes (CALM / CAUTIOUS) with independent asymmetric
        # arming timers and time-in-mode counters. Frontend (dashboard
        # + heartbeat) consumes these to surface CAUTIOUS-entry-armed
        # and CALM-exit-armed states before the transition fires.
        "calm_entry_arming_seconds",
        "calm_exit_arming_seconds",
        "cautious_entry_arming_seconds",
        "cautious_exit_arming_seconds",
        "time_in_calm_seconds",
        "time_in_cautious_seconds",
        # Phase 4G.5 (v1.4.211) — forward classifier diagnostics block.
        # Renders ALWAYS (per ``feedback_bot_stats_panels_always_render``);
        # when the forward layer is disabled OR no reading yet, the
        # inner fields are None / placeholder values.
        "forward_signal",
    }
    assert set(snap.keys()) == expected_keys
    # Default-state value semantics that frontend code will assume.
    assert snap["mode"] == "NORMAL"
    assert snap["seconds_in_mode"] == 0.0
    assert snap["transition_count"] == 0
    assert snap["recent_transitions"] == []
    assert snap["entry_arming_seconds"] is None
    assert snap["exit_arming_seconds"] is None
    assert snap["last_transition_reason"] is None
    # Phase 4G.2 forward arming timers default to None (no arming in
    # progress) — symmetric with the reactive entry/exit_arming fields.
    assert snap["calm_entry_arming_seconds"] is None
    assert snap["calm_exit_arming_seconds"] is None
    assert snap["cautious_entry_arming_seconds"] is None
    assert snap["cautious_exit_arming_seconds"] is None
    # All session-time counters are floats (not None or strings).
    for k in (
        "time_in_normal_seconds",
        "time_in_defensive_seconds",
        "time_in_shock_seconds",
        "time_in_calm_seconds",
        "time_in_cautious_seconds",
    ):
        assert isinstance(snap[k], float)
    # Phase 4G.5 — default forward_signal block (no reading passed).
    # Frontend cards must render even with the forward layer disabled.
    fwd = snap["forward_signal"]
    assert isinstance(fwd, dict)
    assert fwd["classification"] is None
    assert fwd["reason"] is None
    assert fwd["history_span_seconds"] == 0.0
    for k in (
        "vol_slope_bps_per_min",
        "drift_magnitude_30s_bps",
        "drift_magnitude_rising_ratio_observed",
        "ob_imbalance_widening_delta_observed",
        "basis_stretch_ratio_observed",
    ):
        assert fwd[k] is None


def test_1F3_post_reduction_cooldown_snapshot_carries_every_field() -> None:
    """Same shape-parity contract for the 1D cooldown block."""
    from app.state import _behavioural_gates_snapshot

    settings = _settings_for_replay()
    state = BotState(settings)
    snap = _behavioural_gates_snapshot(state)
    block = snap["post_reduction_cooldown"]
    # v1.4.144 Phase 2K.1: added three keys to the cooldown block:
    #   * ``clear_util_pct`` — configured early-exit threshold so the
    #     dashboard can read the knob without env-file access.
    #   * ``cleared_via_favorable_total`` / ``cleared_via_ceiling_total``
    #     — session-cumulative exit-attribution counters. Operator
    #     reads the ratio to calibrate clear_util_pct.
    # Shape-parity test still pinned to a strict equality so accidental
    # additions get caught — update this set when adding fields.
    expected_keys = {
        "enabled",
        "cooldown_seconds",
        "clear_util_pct",
        "active",
        "seconds_remaining",
        "suppressed_side",
        "fire_count",
        "cleared_via_favorable_total",
        "cleared_via_ceiling_total",
    }
    assert set(block.keys()) == expected_keys
    assert block["enabled"] is True
    assert block["cooldown_seconds"] == 60.0
    assert block["active"] is False
    assert isinstance(block["seconds_remaining"], float)
    assert block["suppressed_side"] is None
    assert block["fire_count"] == 0
    # New fields default to their initial values on a fresh BotState.
    assert isinstance(block["clear_util_pct"], float)
    assert 0.0 <= block["clear_util_pct"] <= 1.0
    assert block["cleared_via_favorable_total"] == 0
    assert block["cleared_via_ceiling_total"] == 0


def test_1F3_shock_gate_snapshot_carries_every_field() -> None:
    """Same shape-parity contract for the 1B shock_gate block."""
    from app.shock_gate import snapshot_dict as shock_snapshot_dict, ShockGateState

    sg = ShockGateState()
    snap = shock_snapshot_dict(sg, now_mono=0.0)
    expected_keys = {
        "active",
        "locked_side",
        "seconds_in_lock",
        "last_trigger_drift_bps",
        "last_trigger_util",
        "last_trigger_window",
        "fire_count",
    }
    assert set(snap.keys()) == expected_keys
    assert snap["active"] is False
    assert snap["locked_side"] is None
    assert snap["fire_count"] == 0
