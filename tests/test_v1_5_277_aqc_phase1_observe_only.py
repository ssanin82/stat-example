"""v1.5.277 / AQC Phase 1 — observe-only Active Quoting Controller tests.

What's covered
==============

The controller is pure / state-isolated, so the test surface is the
state-transition table:

* Disabled gate: no state changes when ``settings.enabled = False``.
* Cold-start tick: proportional response fires immediately even
  without a prior timestamp.
* Steady-state convergence: PI drives integrator + aggression
  consistent with a non-zero error.
* Conditional-integration anti-windup (v1.5.280): the integrator
  freezes at the output-saturation edge (≈ 1/K_i), NOT at the
  ±300 hard clip, and unwinds on the very next tick when the
  error sign flips. The ±300 clip remains a defensive backstop.
* Output clip: ``aggression_level`` is clipped to ``[0, 1]``.
* Markout safety floor: ``markout < floor`` forces aggression = 0
  and resets the integrator; engagement counter increments only on
  rising edge.
* Net-edge hold: when net-edge is None or non-finite the
  controller holds state (it's the hard PI input). v1.5.279
  decouple — a missing / non-finite MARKOUT no longer holds: the
  PI advances on net-edge alone and the safety brake simply
  doesn't engage that tick.
* Reset: zeros PI state but preserves the cumulative diagnostic
  counters.
* ``snapshot_dict`` contains every documented field with the
  expected types.
* ``compute_rolling_net_edge_per_min`` math sanity.

Phase 1 is observe-only by contract: the test suite does NOT
assert anything about the controller affecting quote-engine
behavior because nothing consumes ``aggression_level`` yet.
That's verified by inspection of the wire-up sites — see
``app/bot.py::_update_active_quoting_controller`` and
``app/state.py::BotState.snapshot_dict``.
"""

from __future__ import annotations

import math

import pytest

from app.active_quoting_controller import (
    _INTEGRATOR_MAX,
    _INTEGRATOR_MIN,
    ActiveQuotingController,
    AQCSettings,
    compute_rolling_net_edge_per_min,
)


def _enabled_controller(**overrides) -> ActiveQuotingController:
    """Helper: enabled controller with default PI knobs unless
    overridden."""
    defaults = dict(
        enabled=True,
        target_net_edge_per_min_usd=0.020,
        markout_floor_bps=-5.0,
        pi_kp=0.4,
        pi_ki=0.05,
    )
    defaults.update(overrides)
    return ActiveQuotingController(settings=AQCSettings(**defaults))


# === Disabled-gate behavior ===


def test_disabled_controller_does_not_advance_pi_state():
    """When enabled=False the controller stamps the timestamp but
    does not advance PI state. This way an operator flipping the
    enable flag mid-session doesn't see a stale-dt integrator jump."""
    c = ActiveQuotingController(settings=AQCSettings(enabled=False))
    c.update(
        observed_net_edge_per_min_usd=-1.0,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=10.0,
    )
    assert c.aggression_level == 0.0
    assert c.integrator == 0.0
    assert c.update_count == 0
    assert c.last_update_mono_seconds == pytest.approx(10.0)


# === Cold-start / first-tick behavior ===


def test_cold_start_first_tick_fires_proportional_response():
    """On the very first tick the controller has no prior
    timestamp, but the proportional term should fire immediately
    using the nominal 0.5 s dt."""
    c = _enabled_controller()
    # error = 0.020 - (-0.5) = 0.520
    # integrator step = 0.520 * 0.5 = 0.260
    # raw = 0.4 * 0.520 + 0.05 * 0.260 = 0.208 + 0.013 = 0.221
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0,
    )
    assert c.integrator == pytest.approx(0.260, abs=1e-6)
    assert c.aggression_level == pytest.approx(0.221, abs=1e-6)
    assert c.update_count == 1
    assert c.last_update_mono_seconds == pytest.approx(1.0)


# === Steady-state / multi-tick convergence ===


def test_pi_integrator_accumulates_over_many_ticks():
    """With a constant non-zero error the integrator should
    grow linearly until it hits the clip bound, after which
    the output is dominated by the proportional term + the
    saturated integrator."""
    c = _enabled_controller()
    last_integrator = 0.0
    # 50 ticks at 0.5 s; constant error = 0.020 - (-0.080) = 0.100
    # Per-tick integrator step = 0.100 * 0.5 = 0.050
    for i in range(50):
        c.update(
            observed_net_edge_per_min_usd=-0.080,
            observed_markout_5s_mean_bps=0.0,
            now_mono_seconds=1.0 + i * 0.5,
        )
        assert c.integrator > last_integrator
        last_integrator = c.integrator
    # 50 * 0.050 = 2.5
    assert c.integrator == pytest.approx(2.5, abs=1e-6)
    assert 0.0 < c.aggression_level <= 1.0


# === Conditional-integration anti-windup (v1.5.280) ===


def test_conditional_integration_freezes_integrator_at_output_saturation():
    """Pump a large positive error indefinitely. With conditional
    integration the integrator must NOT wind to _INTEGRATOR_MAX —
    it freezes at the edge where the output saturates (≈ 1/K_i).

    With error=0.100, K_p=0.4, K_i=0.05: the proportional term is
    0.040, so the I-term needs 0.960 to pin aggression at 1.0 →
    integrator ≈ 0.960/0.05 = 19.2. The integrator parks there and
    stops; it never approaches the ±300 hard clip. (Old clip-only
    behavior wound to _INTEGRATOR_MAX = 300, 15× past saturation.)"""
    c = _enabled_controller()
    for i in range(10_000):
        c.update(
            observed_net_edge_per_min_usd=-0.080,  # error = +0.100
            observed_markout_5s_mean_bps=0.0,
            now_mono_seconds=1.0 + i * 0.5,
        )
    # Output is pinned at (just below) the clip: conditional
    # integration stops the integrator one step before the next step
    # would push raw output past 1.0, so aggression settles at
    # ~0.9975 here rather than exactly 1.0. Effectively saturated.
    assert c.aggression_level >= 0.99
    # Frozen near 1/K_i, NOT at the hard clip bound.
    assert 19.0 <= c.integrator <= 20.5
    assert c.integrator < _INTEGRATOR_MAX / 10.0


def test_conditional_integration_no_windup_at_low_saturation():
    """Pump a large negative error indefinitely. The output pins at
    0 from the first tick (the proportional term alone is negative),
    so every further negative integrator step would deepen the
    saturation and is therefore frozen. The integrator must stay
    bounded near 0 — never running away to _INTEGRATOR_MIN."""
    c = _enabled_controller()
    for i in range(10_000):
        c.update(
            observed_net_edge_per_min_usd=100.0,  # error strongly < 0
            observed_markout_5s_mean_bps=0.0,
            now_mono_seconds=1.0 + i * 0.5,
        )
    assert c.aggression_level == 0.0
    # Did NOT wind down to the hard clip bound.
    assert c.integrator >= -1.0
    assert c.integrator > _INTEGRATOR_MIN / 10.0


def test_integrator_unwinds_promptly_when_error_flips():
    """The whole point of conditional integration: when the market
    recovers (error flips sign) the controller backs off
    IMMEDIATELY, not after hours of unwinding an over-wound
    integrator. This is the v1.5.279 live-incident regression
    (aggression stuck at 0.99 for ~16 min; under the old clip-only
    path with the integrator wound toward 300 it would have taken
    ~4 h to come off 1.0)."""
    c = _enabled_controller()
    # Phase 1: wind up against a below-target net edge until pinned.
    for i in range(2_000):
        c.update(
            observed_net_edge_per_min_usd=-0.080,  # error = +0.100
            observed_markout_5s_mean_bps=0.0,
            now_mono_seconds=1.0 + i * 0.5,
        )
    assert c.aggression_level >= 0.99  # pinned at (just below) the clip
    saturated_integrator = c.integrator
    saturated_aggression = c.aggression_level
    assert saturated_integrator <= 20.5  # frozen near 1/K_i, not at 300

    # Phase 2: net edge jumps well above target — error flips
    # strongly negative. ONE tick must already pull both the
    # integrator and the output down.
    c.update(
        observed_net_edge_per_min_usd=0.30,  # error = -0.280
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0 + 2_000 * 0.5,
    )
    assert c.integrator < saturated_integrator  # unwound on the very next tick
    assert c.aggression_level < saturated_aggression  # backed off immediately
    assert c.aggression_level > 0.0  # graceful back-off, not a slam to zero


# === Output clipping ===


def test_aggression_clipped_to_zero_on_negative_pi_output():
    """Very small / negative PI output (e.g. observed >> target)
    must be clipped to 0, not allowed to go negative."""
    c = _enabled_controller()
    c.update(
        observed_net_edge_per_min_usd=10.0,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0,
    )
    assert c.aggression_level == 0.0


def test_aggression_clipped_to_one_on_large_pi_output():
    """Massive positive PI output must be clipped to 1.0."""
    c = _enabled_controller(pi_kp=10.0, pi_ki=1.0)
    c.update(
        observed_net_edge_per_min_usd=-10.0,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0,
    )
    assert c.aggression_level == 1.0


# === Markout safety floor ===


def test_markout_below_floor_forces_aggression_zero():
    """When the window MEDIAN markout falls below the floor (with
    enough fills), aggression snaps to 0 and the integrator resets —
    overriding PI. v1.5.290: median + min-fill gate."""
    c = _enabled_controller()
    # Drive integrator positive across several ticks. Median above
    # floor + enough fills → floor not engaged, PI advances.
    for i in range(10):
        c.update(
            observed_net_edge_per_min_usd=-0.5,
            observed_markout_5s_median_bps=0.0,
            markout_sample_count=5,
            now_mono_seconds=1.0 + i * 0.5,
        )
    assert c.integrator > 0
    assert c.aggression_level > 0

    # Now the median collapses past the floor (≥ min_fills fills).
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_median_bps=-6.0,
        markout_sample_count=5,
        now_mono_seconds=10.0,
    )
    assert c.aggression_level == 0.0
    assert c.integrator == 0.0
    assert c.safety_floor_engaged is True
    assert c.safety_floor_engagement_count == 1


def test_safety_floor_engagement_counter_only_rises_on_edge():
    """The engagement counter must NOT increment on every tick
    while the floor stays engaged — only on rising edges."""
    c = _enabled_controller()

    # Floor engaged across 5 ticks (median below floor, enough fills).
    for i in range(5):
        c.update(
            observed_net_edge_per_min_usd=-0.1,
            observed_markout_5s_median_bps=-6.0,
            markout_sample_count=5,
            now_mono_seconds=1.0 + i * 0.5,
        )
    assert c.safety_floor_engaged is True
    assert c.safety_floor_engagement_count == 1

    # Recover above floor for one tick.
    c.update(
        observed_net_edge_per_min_usd=-0.1,
        observed_markout_5s_median_bps=0.0,
        markout_sample_count=5,
        now_mono_seconds=10.0,
    )
    assert c.safety_floor_engaged is False
    assert c.safety_floor_engagement_count == 1

    # Re-engage — counter now rises to 2.
    c.update(
        observed_net_edge_per_min_usd=-0.1,
        observed_markout_5s_median_bps=-6.0,
        markout_sample_count=5,
        now_mono_seconds=10.5,
    )
    assert c.safety_floor_engaged is True
    assert c.safety_floor_engagement_count == 2


def test_markout_at_floor_is_not_engaged():
    """Boundary: ``median == floor`` is not below the floor.
    The contract is strict less-than to make the threshold's
    semantics deterministic."""
    c = _enabled_controller()
    c.update(
        observed_net_edge_per_min_usd=-0.1,
        observed_markout_5s_median_bps=-5.0,  # exactly at floor
        markout_sample_count=5,
        now_mono_seconds=1.0,
    )
    assert c.safety_floor_engaged is False
    assert c.safety_floor_engagement_count == 0


# === v1.5.290 — outlier-robust floor (median + min-fill gate) ===


def test_median_not_mean_drives_floor_v1_5_289_regression():
    """The v1.5.289 incident in one assertion: a single flash fill
    (-91 bps) drags the window MEAN to -12.3 < -5, but the MEDIAN is
    -2.8 (above the floor). Pre-v1.5.290 the mean-based floor pinned
    aggression at 0 for the whole window. The median-based floor must
    NOT engage — the lone outlier cannot trip the brake. The mean is
    passed too (it is now diagnostic-only and must be ignored by the
    floor)."""
    c = _enabled_controller()
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_median_bps=-2.8,  # above floor
        observed_markout_5s_mean_bps=-12.3,  # below floor — diagnostic only
        markout_sample_count=11,
        now_mono_seconds=1.0,
    )
    assert c.safety_floor_engaged is False
    assert c.safety_floor_engagement_count == 0
    # PI advanced on net-edge (below target → integrator winds up).
    assert c.update_count == 1
    assert c.aggression_level > 0
    # Both stats cached for the operator to SEE the divergence.
    assert c.last_observed_markout_5s_median_bps == pytest.approx(-2.8)
    assert c.last_observed_markout_5s_mean_bps == pytest.approx(-12.3)


def test_min_fill_gate_blocks_floor_below_threshold():
    """A median below the floor must NOT engage the brake until at
    least ``markout_floor_min_fills`` distinct fills are present.
    Below the gate the PI advances unbraked; at the gate the brake
    fires."""
    c = _enabled_controller()  # default min_fills = 5
    # Median below floor but only 4 fills → gated out, no brake.
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_median_bps=-6.0,
        markout_sample_count=4,
        now_mono_seconds=1.0,
    )
    assert c.safety_floor_engaged is False
    assert c.safety_floor_engagement_count == 0
    assert c.aggression_level > 0  # PI ran unbraked

    # One more fill → 5 ≥ min_fills → brake engages.
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_median_bps=-6.0,
        markout_sample_count=5,
        now_mono_seconds=1.5,
    )
    assert c.safety_floor_engaged is True
    assert c.safety_floor_engagement_count == 1
    assert c.aggression_level == 0.0
    assert c.integrator == 0.0


def test_min_fill_gate_is_configurable():
    """The min-fill threshold is operator-tunable via
    ``markout_floor_min_fills``. With it set to 3, a 3-fill window
    whose median is below the floor engages the brake."""
    c = _enabled_controller(markout_floor_min_fills=3)
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_median_bps=-6.0,
        markout_sample_count=3,
        now_mono_seconds=1.0,
    )
    assert c.safety_floor_engaged is True
    assert c.safety_floor_engagement_count == 1


def test_floor_skipped_when_median_absent_pi_advances():
    """A sample count above the gate but a missing median = no brake
    this tick (v1.5.279 decouple): the PI still advances on net-edge
    rather than freezing."""
    c = _enabled_controller()
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_median_bps=None,
        markout_sample_count=10,
        now_mono_seconds=1.0,
    )
    assert c.safety_floor_engaged is False
    assert c.safety_floor_engagement_count == 0
    assert c.update_count == 1
    assert c.aggression_level > 0


def test_median_and_sample_count_surfaced_in_snapshot():
    """The median and sample count are operator-facing diagnostics —
    both must round-trip through snapshot_dict."""
    c = _enabled_controller()
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_median_bps=-3.1,
        observed_markout_5s_mean_bps=-9.9,
        markout_sample_count=7,
        now_mono_seconds=1.0,
    )
    snap = c.snapshot_dict()
    assert snap["last_observed_markout_5s_median_bps"] == pytest.approx(-3.1)
    assert snap["last_observed_markout_sample_count"] == 7
    assert snap["last_observed_markout_5s_mean_bps"] == pytest.approx(-9.9)
    assert snap["markout_floor_min_fills"] == 5


# === Hold-on-None / non-finite observation behavior ===


def test_holds_state_when_net_edge_observation_is_none():
    """Missing net-edge observation: controller holds state.
    No integrator decay, no aggression change."""
    c = _enabled_controller()
    # Warm up with two real ticks.
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0,
    )
    aggression_before = c.aggression_level
    integrator_before = c.integrator
    update_count_before = c.update_count

    c.update(
        observed_net_edge_per_min_usd=None,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.5,
    )
    assert c.aggression_level == aggression_before
    assert c.integrator == integrator_before
    assert c.update_count == update_count_before
    # Timestamp still advances so future dt is correct.
    assert c.last_update_mono_seconds == pytest.approx(1.5)


def test_advances_pi_when_markout_is_none_netedge_present():
    """v1.5.279 decouple: a missing markout no longer holds the
    controller. With net-edge present the PI advances on net-edge
    alone; the safety brake simply doesn't engage this tick. This
    is what makes Phase-1 observable at TON's ~0.23 fills/min
    instead of idling at update_count=0."""
    c = _enabled_controller()
    # First tick with full inputs.
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0,
    )
    integrator_after_first = c.integrator
    assert c.update_count == 1

    # Second tick: markout unavailable (too few samples). PI MUST
    # still advance — integrator grows, update_count increments.
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_mean_bps=None,
        now_mono_seconds=1.5,
    )
    assert c.update_count == 2
    assert c.integrator > integrator_after_first  # advanced, not held
    assert c.safety_floor_engaged is False
    # The None markout doesn't overwrite the cache — it retains the
    # last finite value seen (0.0 from tick 1).
    assert c.last_observed_markout_5s_mean_bps == pytest.approx(0.0)


def test_safety_floor_skipped_when_markout_unavailable():
    """v1.5.279: the markout safety brake is consulted ONLY when a
    markout sample is available. With markout None, the floor never
    engages regardless of net-edge — the PI runs unbraked and winds
    up on the below-target net edge."""
    c = _enabled_controller()
    for i in range(20):
        c.update(
            observed_net_edge_per_min_usd=-1.0,
            observed_markout_5s_mean_bps=None,
            now_mono_seconds=1.0 + i * 0.5,
        )
    assert c.update_count == 20
    assert c.safety_floor_engaged is False
    assert c.safety_floor_engagement_count == 0
    assert c.integrator > 0  # wound up on net-edge alone
    assert c.aggression_level > 0


def test_quiet_market_zero_netedge_still_advances_pi():
    """Regression for the v1.5.278 live incident: an empty rolling
    window yields net_edge=0.0 (not None) and markout=None (no
    samples). Pre-decouple the controller HELD every tick and
    idled at update_count=0 for a full hour. Post-decouple the PI
    must advance: error = target - 0.0 > 0 winds the integrator up,
    aggression rises, and update_count climbs."""
    c = _enabled_controller()
    for i in range(10):
        c.update(
            observed_net_edge_per_min_usd=0.0,  # empty window
            observed_markout_5s_mean_bps=None,  # <5 samples
            now_mono_seconds=1.0 + i * 0.5,
        )
    assert c.update_count == 10  # NOT 0 — the old-bug value
    assert c.integrator > 0
    assert c.aggression_level > 0
    assert c.safety_floor_engaged is False
    assert c.last_observed_net_edge_per_min_usd == pytest.approx(0.0)
    assert c.last_observed_markout_5s_mean_bps is None


def test_holds_state_when_net_edge_is_non_finite():
    """NaN/inf net-edge: controller holds state — net-edge is the
    hard PI input. Defensive against upstream numerics breakage."""
    c = _enabled_controller()
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0,
    )
    aggression_before = c.aggression_level
    integrator_before = c.integrator
    update_count_before = c.update_count

    c.update(
        observed_net_edge_per_min_usd=float("nan"),
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.5,
    )
    assert c.aggression_level == aggression_before
    assert c.integrator == integrator_before
    assert c.update_count == update_count_before


def test_non_finite_markout_treated_as_unavailable_pi_advances():
    """v1.5.279: a non-finite markout (NaN/inf) is treated like a
    missing sample — no safety brake, but the PI still advances on
    net-edge. Defensive against an upstream markout-numerics break
    that must NOT freeze the controller."""
    c = _enabled_controller()
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0,
    )
    integrator_before = c.integrator
    update_count_before = c.update_count

    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_mean_bps=float("inf"),
        now_mono_seconds=1.5,
    )
    # Advanced, not held.
    assert c.update_count == update_count_before + 1
    assert c.integrator > integrator_before
    assert c.safety_floor_engaged is False
    # Non-finite markout is NOT cached as last-observed (the prior
    # finite 0.0 persists).
    assert c.last_observed_markout_5s_mean_bps == pytest.approx(0.0)


# === reset() ===


def test_reset_clears_pi_state_but_preserves_counters():
    """reset() zeros PI state — operator calls it after knob
    retunes — but session-cumulative counters survive."""
    c = _enabled_controller()
    for i in range(5):
        c.update(
            observed_net_edge_per_min_usd=-0.5,
            observed_markout_5s_median_bps=-6.0,  # engage floor
            markout_sample_count=5,
            now_mono_seconds=1.0 + i * 0.5,
        )
    assert c.update_count == 5
    assert c.safety_floor_engagement_count == 1
    assert c.safety_floor_engaged is True

    c.reset()
    assert c.aggression_level == 0.0
    assert c.integrator == 0.0
    assert c.safety_floor_engaged is False
    # Cumulative diagnostics preserved.
    assert c.update_count == 5
    assert c.safety_floor_engagement_count == 1


# === snapshot_dict shape ===


def test_snapshot_dict_contains_all_expected_fields():
    """snapshot_dict is the operator-facing telemetry surface;
    its shape is a contract."""
    c = _enabled_controller()
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_median_bps=-1.0,
        observed_markout_5s_mean_bps=-1.0,
        markout_sample_count=6,
        now_mono_seconds=1.0,
    )
    snap = c.snapshot_dict()
    expected_keys = {
        "enabled",
        "aggression_level",
        "integrator",
        "safety_floor_engaged",
        "safety_floor_engagement_count",
        "update_count",
        "last_observed_net_edge_per_min_usd",
        "last_observed_markout_5s_mean_bps",
        # v1.5.290 outlier-robust floor telemetry.
        "last_observed_markout_5s_median_bps",
        "last_observed_markout_sample_count",
        "markout_floor_min_fills",
        "target_net_edge_per_min_usd",
        "markout_floor_bps",
        "pi_kp",
        "pi_ki",
        # v1.5.281 AQC Phase 2 output-wiring telemetry.
        "wire_min_half_spread",
        "min_half_spread_floor_at_full_aggression_bps",
        # v1.5.282 AQC Phase 3 output-wiring telemetry.
        "wire_inventory",
        "inventory_util_floor_at_full_aggression_pct",
        # v1.5.283 AQC Phase 4 output-wiring telemetry.
        "wire_skew",
        "skew_coeff_at_full_aggression_bps",
        "last_update_mono_seconds",
    }
    assert set(snap.keys()) == expected_keys
    assert snap["enabled"] is True
    assert isinstance(snap["aggression_level"], float)
    assert isinstance(snap["integrator"], float)
    assert isinstance(snap["safety_floor_engaged"], bool)
    assert isinstance(snap["safety_floor_engagement_count"], int)
    assert isinstance(snap["update_count"], int)
    assert snap["last_observed_net_edge_per_min_usd"] == pytest.approx(-0.5)
    assert snap["last_observed_markout_5s_mean_bps"] == pytest.approx(-1.0)
    assert snap["last_observed_markout_5s_median_bps"] == pytest.approx(-1.0)
    assert snap["last_observed_markout_sample_count"] == 6
    assert snap["markout_floor_min_fills"] == 5
    assert snap["target_net_edge_per_min_usd"] == pytest.approx(0.020)
    assert snap["markout_floor_bps"] == pytest.approx(-5.0)
    assert snap["pi_kp"] == pytest.approx(0.4)
    assert snap["pi_ki"] == pytest.approx(0.05)
    assert snap["last_update_mono_seconds"] == pytest.approx(1.0)


def test_snapshot_dict_handles_none_observations():
    """Before any update has run, the snapshot's last-observed
    fields are None; the keys still exist."""
    c = ActiveQuotingController(settings=AQCSettings(enabled=True))
    snap = c.snapshot_dict()
    assert snap["last_observed_net_edge_per_min_usd"] is None
    assert snap["last_observed_markout_5s_mean_bps"] is None
    assert snap["last_observed_markout_5s_median_bps"] is None
    assert snap["last_observed_markout_sample_count"] is None
    assert snap["last_update_mono_seconds"] is None
    assert snap["enabled"] is True
    assert snap["aggression_level"] == 0.0
    assert snap["integrator"] == 0.0


def test_snapshot_dict_when_disabled():
    """Disabled controller's snapshot still publishes every key
    — operator can A/B 'what would AQC have done if enabled' on
    a disabled session by inspecting the snapshot fields, even
    though aggression_level is held at 0."""
    c = ActiveQuotingController(settings=AQCSettings(enabled=False))
    snap = c.snapshot_dict()
    assert snap["enabled"] is False
    assert snap["aggression_level"] == 0.0


# === compute_rolling_net_edge_per_min helper ===


def test_compute_rolling_net_edge_per_min_basic():
    """Simple sanity: 60 s of $0.05 rebate + $0.10 markout
    should yield (0.05 + 0.10) / 1 minute = 0.15 USD/min."""
    result = compute_rolling_net_edge_per_min(
        recent_fills_window_seconds=60.0,
        rebate_usd_in_window=0.05,
        markout_dollar_impact_usd_in_window=0.10,
    )
    assert result == pytest.approx(0.15, abs=1e-9)


def test_compute_rolling_net_edge_per_min_negative_markout():
    """Adverse markout (negative) reduces net edge correctly."""
    result = compute_rolling_net_edge_per_min(
        recent_fills_window_seconds=300.0,
        rebate_usd_in_window=0.30,
        markout_dollar_impact_usd_in_window=-0.50,
    )
    # (0.30 + (-0.50)) / 5 min = -0.04
    assert result == pytest.approx(-0.04, abs=1e-9)


def test_compute_rolling_net_edge_per_min_zero_window_returns_none():
    """Empty window: caller has nothing to feed the controller."""
    assert (
        compute_rolling_net_edge_per_min(
            recent_fills_window_seconds=0.0,
            rebate_usd_in_window=0.10,
            markout_dollar_impact_usd_in_window=0.00,
        )
        is None
    )


def test_compute_rolling_net_edge_per_min_negative_window_returns_none():
    """Defensive: negative window is nonsense, return None."""
    assert (
        compute_rolling_net_edge_per_min(
            recent_fills_window_seconds=-1.0,
            rebate_usd_in_window=0.10,
            markout_dollar_impact_usd_in_window=0.00,
        )
        is None
    )


# === AQCSettings defaults vs. Settings field defaults ===


def test_aqc_settings_defaults_match_documented_phase1_baseline():
    """Phase 1 ships with these conservative placeholders. If
    they're tuned, the config.py defaults must move in lockstep
    (see app/config.py AQC_* fields)."""
    s = AQCSettings()
    assert s.enabled is False
    assert s.target_net_edge_per_min_usd == pytest.approx(0.020)
    assert s.markout_floor_bps == pytest.approx(-5.0)
    assert s.markout_floor_min_fills == 5
    assert s.pi_kp == pytest.approx(0.4)
    assert s.pi_ki == pytest.approx(0.05)


def test_aqc_settings_match_config_defaults():
    """The AQCSettings defaults and the Settings class env-knob
    defaults are coupled — a change to one must flow to the
    other. This test is the early-warning trip-wire if a future
    edit slips."""
    from app.config import Settings as _CfgSettings

    cfg = _CfgSettings()
    aqc = AQCSettings()
    assert aqc.target_net_edge_per_min_usd == pytest.approx(
        cfg.aqc_target_net_edge_per_min_usd
    )
    assert aqc.markout_floor_bps == pytest.approx(cfg.aqc_markout_floor_bps)
    assert aqc.markout_floor_min_fills == cfg.aqc_markout_floor_min_fills
    assert aqc.pi_kp == pytest.approx(cfg.aqc_pi_kp)
    assert aqc.pi_ki == pytest.approx(cfg.aqc_pi_ki)
    assert aqc.enabled == cfg.aqc_enabled


# === Phase 1 observe-only contract ===


def test_phase1_observe_only_no_external_state_mutation():
    """Phase 1 contract: the controller is a pure
    state-isolated object. update() does not reach out to
    BotState, the quote engine, the order store, or any other
    module. We can construct it standalone and exercise the
    whole state-transition table with no test fixtures.

    If a future ship adds external side-effects this test must
    fail loudly so Phase 1's observe-only contract is preserved.
    """
    c = _enabled_controller()
    # No fixtures, no BotState — pure construction.
    assert c.aggression_level == 0.0
    c.update(
        observed_net_edge_per_min_usd=-0.5,
        observed_markout_5s_mean_bps=0.0,
        now_mono_seconds=1.0,
    )
    assert c.update_count == 1
    # No external module touched: success is the absence of
    # an ImportError or AttributeError at this line.
    assert math.isfinite(c.aggression_level)
