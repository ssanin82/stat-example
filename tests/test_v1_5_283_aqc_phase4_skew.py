"""v1.5.283 / AQC Phase 4 — aggression -> inventory skew coefficient.

Phase 4 wires the controller's ``aggression_level`` (0..1) into the
inventory skew coefficient via
``compute_effective_inventory_skew_coeff_bps``. The wire is
INCREASE-ONLY (it can only RAISE the coefficient → reservation shifts
further per unit of inventory → the reducing-side quote pulls closer to
touch → faster, constructive inventory flattening) and guarded by
``AQC_WIRE_SKEW`` (default false), so the skew is byte-identical to
pre-Phase-4 unless the operator flips the flag.

Direction note (the MIRROR of Phase 2/3): Phase 2/3 are TIGHTEN-ONLY
(``min(base, lerped)``). Phase 4 is INCREASE-ONLY (``max(base,
lerped)``) because "more aggressive" here means a HIGHER skew
coefficient — the constructive, churn-risk-DOWN direction. This is the
Rule-0c-clean lever: it never holds risk longer than configured.

What's covered
==============

* Default OFF: with ``AQC_WIRE_SKEW=false`` the ``aqc_aggression_level``
  kwarg is ignored for every level (0, 0.5, 1.0, None, out-of-range) —
  pre-Phase-4 byte-identical.
* None-safe / NaN-safe: even with the wire ON, a None/NaN level leaves
  the coefficient at base (the decision wasn't stamped → no-op).
* Lerp endpoints + midpoint: aggression 0 -> base, 1 -> target,
  0.5 -> midpoint.
* Increase-only: a mis-configured target BELOW base can never LOWER the
  coefficient, and no aggression in [0,1] ever drops below base.
* Clamp: aggression outside [0,1] is clamped before lerp.
* Engine integration: ``compute_quote_decision`` reads the
  ``aqc_aggression_level`` kwarg — at a fixed long position the
  breakdown's ``inventory_skew_bps`` magnitude GROWS from wire-off /
  aggr-0 to wire-on / aggr-1 (base 10 -> target 20 doubles it). The
  teeth proof.
* Safety envelope survives: even an absurd AQC skew target can't push
  the reservation past ``MAX_RESERVATION_SHIFT_BPS_FROM_MID``.
* Plumbing: AQCSettings carries the two new fields and publishes them
  in ``snapshot_dict``; ``Settings`` parses both env aliases.
"""

from __future__ import annotations

import pytest

from app.active_quoting_controller import ActiveQuotingController, AQCSettings
from app.config import Settings
from app.quoting import (
    compute_effective_inventory_skew_coeff_bps,
    compute_quote_decision,
)
from app.toxicity import ToxicitySnapshot


# Prod-like TON base inventory skew coefficient: 10 bps.
_BASE_ENV = dict(
    SYMBOL="TON",
    INVENTORY_SKEW_COEFF_BPS="10",
)

_BASE = 10.0  # INVENTORY_SKEW_COEFF_BPS above.


def _settings(**overrides) -> Settings:
    env = dict(_BASE_ENV)
    env.update(overrides)
    return Settings(**env)


def _skew(settings: Settings, level) -> float:
    return compute_effective_inventory_skew_coeff_bps(
        settings, aqc_aggression_level=level
    )


# --------------------------------------------------------------------
# Default OFF — pre-Phase-4 byte-identical.
# --------------------------------------------------------------------

@pytest.mark.parametrize("level", [0.0, 0.5, 1.0, None, 2.0, -1.0])
def test_wire_off_ignores_aggression(level):
    s = _settings()  # AQC_WIRE_SKEW defaults false
    assert s.aqc_wire_skew is False
    assert _skew(s, level) == pytest.approx(_BASE)


def test_wire_off_matches_no_kwarg():
    s = _settings()
    base = compute_effective_inventory_skew_coeff_bps(s)
    assert _skew(s, 1.0) == pytest.approx(base)
    assert base == pytest.approx(_BASE)


# --------------------------------------------------------------------
# Wire ON — lerp behavior.
# --------------------------------------------------------------------

def test_wire_on_none_is_noop():
    s = _settings(AQC_WIRE_SKEW="true")
    assert _skew(s, None) == pytest.approx(_BASE)


def test_wire_on_nan_is_noop():
    s = _settings(AQC_WIRE_SKEW="true")
    assert _skew(s, float("nan")) == pytest.approx(_BASE)


def test_wire_on_aggr_zero_is_base():
    s = _settings(AQC_WIRE_SKEW="true")
    assert _skew(s, 0.0) == pytest.approx(_BASE)


def test_wire_on_aggr_full_is_target():
    s = _settings(
        AQC_WIRE_SKEW="true",
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS="20.0",
    )
    assert _skew(s, 1.0) == pytest.approx(20.0)


def test_wire_on_aggr_half_is_midpoint():
    s = _settings(
        AQC_WIRE_SKEW="true",
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS="20.0",
    )
    # 10 + 0.5*(20 - 10) = 15.0
    assert _skew(s, 0.5) == pytest.approx(15.0)


def test_monotonic_non_decreasing_in_aggression():
    s = _settings(
        AQC_WIRE_SKEW="true",
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS="20.0",
    )
    prev = _skew(s, 0.0)
    for lv in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        cur = _skew(s, lv)
        assert cur >= prev - 1e-12
        prev = cur


# --------------------------------------------------------------------
# Clamp + increase-only guard.
# --------------------------------------------------------------------

def test_clamp_above_one_is_target():
    s = _settings(
        AQC_WIRE_SKEW="true",
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS="20.0",
    )
    assert _skew(s, 5.0) == pytest.approx(20.0)


def test_clamp_below_zero_is_base():
    s = _settings(
        AQC_WIRE_SKEW="true",
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS="20.0",
    )
    assert _skew(s, -3.0) == pytest.approx(_BASE)


def test_target_below_base_is_noop():
    # A mis-set endpoint BELOW base must never LOWER the coefficient.
    s = _settings(
        AQC_WIRE_SKEW="true",
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS="5.0",
    )
    assert _skew(s, 1.0) == pytest.approx(_BASE)
    assert _skew(s, 0.5) == pytest.approx(_BASE)


def test_never_below_base():
    s = _settings(
        AQC_WIRE_SKEW="true",
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS="20.0",
    )
    for lv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        assert _skew(s, lv) >= _BASE - 1e-12


# --------------------------------------------------------------------
# Engine integration — the teeth proof via compute_quote_decision.
# --------------------------------------------------------------------

def _tox():
    return ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.5,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=0.0,
        hard_trigger=False,
        soft_trigger=False,
    )


def _qd_settings(**overrides) -> Settings:
    # Trend-skew amplifier off so the inventory-skew comparison is clean
    # (the amplifier multiplies on top; with drift=0 it is a no-op anyway,
    # but disable it explicitly for determinism).
    base = dict(
        VENUE="binance",
        SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0,
        MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
        TOXICITY_ENABLED=False,
        TREND_SKEW_AMPLIFIER_ENABLED=False,
        INVENTORY_SKEW_COEFF_BPS=10.0,
    )
    base.update(overrides)
    return Settings(**base)


def _qd(settings: Settings, aqc_aggression_level=None):
    # Long position (norm_inv = 2.5 / 5.0 = 0.5) → inventory skew shifts
    # reservation DOWN (negative breakdown.inventory_skew_bps).
    return compute_quote_decision(
        settings=settings,
        mid=100.0,
        position_qty=2.5,
        vol_bps=5.0,
        toxicity=_tox(),
        short_term_drift_bps=0.0,
        aqc_aggression_level=aqc_aggression_level,
    )


def test_engine_wire_off_aggression_ignored():
    # Wire off: aggression is ignored, reservation byte-identical.
    off_none = _qd(_qd_settings(), aqc_aggression_level=None)
    off_full = _qd(_qd_settings(), aqc_aggression_level=1.0)
    assert off_full.breakdown.inventory_skew_bps == pytest.approx(
        off_none.breakdown.inventory_skew_bps
    )
    assert off_full.reservation_price == pytest.approx(
        off_none.reservation_price
    )


def test_engine_long_skew_is_negative_delta():
    # Sanity: a long position skews the reservation BELOW mid.
    d = _qd(_qd_settings(), aqc_aggression_level=None)
    assert d.breakdown.inventory_skew_bps < 0.0


def test_engine_teeth_skew_magnitude_grows_with_aggression():
    # base 10 -> target 20 at aggr 1 doubles the inventory skew shift.
    off = _qd(_qd_settings(), aqc_aggression_level=0.0)
    on = _qd(
        _qd_settings(
            AQC_WIRE_SKEW=True,
            AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS=20.0,
        ),
        aqc_aggression_level=1.0,
    )
    off_skew = off.breakdown.inventory_skew_bps
    on_skew = on.breakdown.inventory_skew_bps
    assert abs(on_skew) > abs(off_skew)
    assert abs(on_skew) == pytest.approx(abs(off_skew) * 2.0, rel=1e-6)


def test_engine_teeth_aggr_half_is_midpoint():
    # aggr 0.5 -> coefficient 15 -> 1.5x the base skew magnitude.
    off = _qd(_qd_settings(), aqc_aggression_level=0.0)
    half = _qd(
        _qd_settings(
            AQC_WIRE_SKEW=True,
            AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS=20.0,
        ),
        aqc_aggression_level=0.5,
    )
    assert abs(half.breakdown.inventory_skew_bps) == pytest.approx(
        abs(off.breakdown.inventory_skew_bps) * 1.5, rel=1e-6
    )


def test_engine_reservation_clamp_bounds_aggressive_skew():
    # Safety envelope: even an absurd AQC skew target can't push the
    # reservation past MAX_RESERVATION_SHIFT_BPS_FROM_MID.
    s = _qd_settings(
        AQC_WIRE_SKEW=True,
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS=200.0,
        MAX_RESERVATION_SHIFT_BPS_FROM_MID=3.0,
    )
    d = _qd(s, aqc_aggression_level=1.0)
    assert d.breakdown.reservation_clamp_active is True
    # Long position → reservation below mid → negative delta, magnitude 3.
    assert abs(d.breakdown.reservation_delta_from_mid_bps) == pytest.approx(
        3.0, abs=0.01
    )
    assert d.breakdown.reservation_delta_from_mid_bps < 0.0


# --------------------------------------------------------------------
# Plumbing.
# --------------------------------------------------------------------

def test_aqc_settings_has_phase4_fields():
    s = AQCSettings()
    assert s.wire_skew is False
    assert s.skew_coeff_at_full_aggression_bps == pytest.approx(40.0)


def test_snapshot_dict_publishes_phase4_keys():
    c = ActiveQuotingController(
        settings=AQCSettings(
            enabled=True,
            wire_skew=True,
            skew_coeff_at_full_aggression_bps=22.0,
        )
    )
    snap = c.snapshot_dict()
    assert snap["wire_skew"] is True
    assert snap["skew_coeff_at_full_aggression_bps"] == pytest.approx(22.0)


def test_settings_parses_env_aliases():
    s = _settings(
        AQC_WIRE_SKEW="true",
        AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS="18.0",
    )
    assert s.aqc_wire_skew is True
    assert s.aqc_skew_coeff_at_full_aggression_bps == pytest.approx(18.0)


def test_settings_defaults_off():
    s = _settings()
    assert s.aqc_wire_skew is False
    assert s.aqc_skew_coeff_at_full_aggression_bps == pytest.approx(40.0)
