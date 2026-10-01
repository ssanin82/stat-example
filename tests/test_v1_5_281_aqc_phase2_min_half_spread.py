"""v1.5.281 / AQC Phase 2 — aggression -> min-half-spread wiring tests.

Phase 2 wires the controller's ``aggression_level`` (0..1) into the
economic min-half-spread floor via
``compute_effective_min_half_spread_bps``. The wire is TIGHTEN-ONLY
and guarded by ``AQC_WIRE_MIN_HALF_SPREAD`` (default false), so the
spread pipeline is byte-identical to Phase 1 unless the operator flips
the flag.

What's covered
==============

* Default OFF: with ``AQC_WIRE_MIN_HALF_SPREAD=false`` the
  ``aqc_aggression_level`` kwarg is ignored for every level (0, 0.5,
  1.0, None) — Phase-1 byte-identical.
* None-safe: even with the wire ON, ``aqc_aggression_level=None``
  leaves the floor at base (the decision wasn't stamped → no-op).
* Lerp endpoints + midpoint: aggression 0 -> base, 1 -> target,
  0.5 -> midpoint.
* Tighten-only: a mis-configured target ABOVE base can never WIDEN
  the floor, and no aggression in [0,1] ever exceeds base.
* Clamp: aggression outside [0,1] is clamped before lerp.
* Sub-tick guard: the post-AQC tick floor still clamps tightening up
  to >= 1 tick (AQC can't push the floor sub-tick).
* Additive composition: tox_bump + overlay still add ON TOP of the
  tightened base, so adaptive-widening defenses are never undone.
* Teeth on a wide book: feeding the AQC-tightened ``eff_h`` into
  ``compute_normal_mm_market_capped_half_spread_bps`` with a wide
  market lowers the quoted half-spread (the binding-path proof).
* Plumbing: AQCSettings carries the two new fields and publishes them
  in ``snapshot_dict``; ``QuoteDecision`` has the ``aqc_aggression_level``
  field (default None); ``Settings`` parses both env aliases.
"""

from __future__ import annotations

import math

import pytest

from app.active_quoting_controller import ActiveQuotingController, AQCSettings
from app.config import Settings
from app.models import ActiveSides, QuoteDecision
from app.quote_aging import compute_normal_mm_market_capped_half_spread_bps
from app.quoting import compute_effective_min_half_spread_bps


# Prod-like TON floors: neutral 3.5 / inventory 2.0, rails 0.5 / 4.0.
_BASE_ENV = dict(
    SYMBOL="TON",
    MIN_HALF_SPREAD_BPS="0.5",
    MAX_HALF_SPREAD_BPS="4.0",
    ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS="3.5",
    ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS="2.0",
)


def _settings(**overrides) -> Settings:
    env = dict(_BASE_ENV)
    env.update(overrides)
    return Settings(**env)


def _eff(settings: Settings, level, *, sides=ActiveSides.BOTH, tox=0.0,
         overlay=0.0, tick=None, fv=None) -> float:
    return compute_effective_min_half_spread_bps(
        settings,
        sides,
        tox,
        spread_floor_overlay_half_spread_bps=overlay,
        price_tick=tick,
        fair_value=fv,
        aqc_aggression_level=level,
    )


# --------------------------------------------------------------------
# Default OFF — Phase 1 byte-identical.
# --------------------------------------------------------------------

@pytest.mark.parametrize("level", [0.0, 0.5, 1.0, None, 2.0, -1.0])
def test_wire_off_ignores_aggression(level):
    s = _settings()  # AQC_WIRE_MIN_HALF_SPREAD defaults false
    assert s.aqc_wire_min_half_spread is False
    # Neutral two-sided floor is the econ neutral 3.5.
    assert _eff(s, level) == pytest.approx(3.5)


def test_wire_off_matches_no_kwarg():
    s = _settings()
    base = compute_effective_min_half_spread_bps(s, ActiveSides.BOTH, 0.0)
    assert _eff(s, 1.0) == pytest.approx(base)


# --------------------------------------------------------------------
# Wire ON — lerp behavior.
# --------------------------------------------------------------------

def test_wire_on_none_is_noop():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true")
    assert _eff(s, None) == pytest.approx(3.5)


def test_wire_on_aggr_zero_is_base():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true")
    assert _eff(s, 0.0) == pytest.approx(3.5)


def test_wire_on_aggr_full_is_target():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0")
    assert _eff(s, 1.0) == pytest.approx(2.0)


def test_wire_on_aggr_half_is_midpoint():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0")
    # 3.5 + 0.5*(2.0-3.5) = 2.75
    assert _eff(s, 0.5) == pytest.approx(2.75)


def test_wire_on_monotonic_in_aggression():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0")
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    vals = [_eff(s, lv) for lv in levels]
    # Strictly non-increasing as aggression rises (tighten-only).
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
    assert vals[0] == pytest.approx(3.5)
    assert vals[-1] == pytest.approx(2.0)


def test_inventory_floor_modulated_when_one_sided():
    # One-sided -> inventory econ floor 2.0; target below it (1.0).
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="1.0")
    assert _eff(s, 0.0, sides=ActiveSides.BID_ONLY) == pytest.approx(2.0)
    assert _eff(s, 1.0, sides=ActiveSides.BID_ONLY) == pytest.approx(1.0)


# --------------------------------------------------------------------
# Tighten-only safety.
# --------------------------------------------------------------------

def test_target_above_base_is_noop():
    # Mis-config: target 9.0 > neutral base 3.5. Must NEVER widen.
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="9.0")
    for lv in (0.0, 0.5, 1.0):
        assert _eff(s, lv) == pytest.approx(3.5)


def test_never_exceeds_base_for_any_level():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0")
    for i in range(0, 101):
        lv = i / 100.0
        assert _eff(s, lv) <= 3.5 + 1e-9


def test_aggression_clamped_above_one():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0")
    # >1 clamps to 1 -> target.
    assert _eff(s, 5.0) == pytest.approx(2.0)


def test_aggression_clamped_below_zero():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0")
    # <0 clamps to 0 -> base.
    assert _eff(s, -3.0) == pytest.approx(3.5)


def test_nan_aggression_is_noop():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0")
    assert _eff(s, float("nan")) == pytest.approx(3.5)


# --------------------------------------------------------------------
# Composition: sub-tick guard + additive defenses.
# --------------------------------------------------------------------

def test_sub_tick_guard_clamps_aggression_tightening():
    # Target 0.1 bps half would be sub-tick. With a 1-tick floor the
    # AQC tightening can't push below the tick floor.
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="0.1",
                  ONE_SIDED_EXTRA_TICK_NEUTRAL="0.0")
    # tick=0.01 at fv=2.30 -> tick_bps_half = (0.01/2.30)*1e4/2 ~= 21.7
    # That's larger than base, so the floor is the tick floor and AQC
    # can't tighten below it. Use a small tick so tick_floor < base
    # to isolate the guard: tick=0.0001 at fv=2.30 -> ~0.217 bps half.
    fv, tick = 2.30, 0.0001
    tick_bps_half = (tick / fv) * 10_000.0 / 2.0
    out = _eff(s, 1.0, tick=tick, fv=fv)
    # AQC asked for 0.1 but the tick floor (~0.217) wins via max().
    assert out == pytest.approx(tick_bps_half, rel=1e-6)
    assert out >= 0.1


def test_overlay_adds_on_top_of_tightened_base():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0")
    # Full aggression -> base 2.0; overlay 1.0 adds on top -> 3.0.
    out = _eff(s, 1.0, overlay=1.0)
    assert out == pytest.approx(3.0)
    # Still below the 4.0 max rail so no clamp.


def test_toxicity_bump_adds_on_top_of_tightened_base():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0",
                  ECONOMIC_TOXICITY_SCORE_HALF_SPREAD_BPS="4.0")
    # base 2.0 + tox 0.25*4.0=1.0 -> 3.0.
    out = _eff(s, 1.0, tox=0.25)
    assert out == pytest.approx(3.0)


def test_max_rail_still_caps_after_tightening():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0",
                  MAX_HALF_SPREAD_BPS="2.5")
    # base 2.0 + overlay 2.0 = 4.0 -> capped to 2.5.
    out = _eff(s, 1.0, overlay=2.0)
    assert out == pytest.approx(2.5)


# --------------------------------------------------------------------
# Teeth on a wide book — the binding-path proof.
# --------------------------------------------------------------------

def test_tightening_lowers_quoted_half_spread_on_wide_market():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="2.0",
                  NORMAL_MM_USE_MARKET_SPREAD_ANCHOR="true",
                  NORMAL_MM_MAX_COMPETITIVE_HALF_SPREAD_BPS="0.0",
                  NORMAL_MM_MIN_COMPETITIVE_HALF_SPREAD_BPS="0.5",
                  NORMAL_MM_MAX_DISTANCE_TO_TOUCH_TICKS="8.0")
    # Fine tick so the 1-tick sub-tick floor (~1.09 bps half here) sits
    # BELOW the AQC target (2.0) and doesn't mask it.
    mid, tick = 2.30, 0.0005
    # Wide market: ~10 bps half so the econ floor binds, not the anchor.
    best_bid = mid * (1 - 10.0 / 10_000.0)
    best_ask = mid * (1 + 10.0 / 10_000.0)

    eff_off = _eff(s, 0.0, tick=tick, fv=mid)   # base 3.5
    eff_on = _eff(s, 1.0, tick=tick, fv=mid)    # target 2.0
    assert eff_off == pytest.approx(3.5)
    assert eff_on == pytest.approx(2.0)
    assert eff_off > eff_on

    capped_off, _ = compute_normal_mm_market_capped_half_spread_bps(
        s, model_half_spread_bps=eff_off, best_bid=best_bid,
        best_ask=best_ask, mid=mid, tick=tick)
    capped_on, _ = compute_normal_mm_market_capped_half_spread_bps(
        s, model_half_spread_bps=eff_on, best_bid=best_bid,
        best_ask=best_ask, mid=mid, tick=tick)
    # eff_h binds on the wide book (anchor ~10 > eff_h), so tightening
    # the floor tightens the quoted half-spread 1:1.
    assert capped_off == pytest.approx(eff_off)
    assert capped_on == pytest.approx(eff_on)
    assert capped_on < capped_off


# --------------------------------------------------------------------
# Plumbing.
# --------------------------------------------------------------------

def test_aqc_settings_has_phase2_fields():
    s = AQCSettings()
    assert s.wire_min_half_spread is False
    assert s.min_half_spread_floor_at_full_aggression_bps == pytest.approx(2.0)


def test_snapshot_dict_publishes_phase2_fields():
    ctrl = ActiveQuotingController(
        settings=AQCSettings(
            enabled=True,
            wire_min_half_spread=True,
            min_half_spread_floor_at_full_aggression_bps=2.0,
        )
    )
    snap = ctrl.snapshot_dict()
    assert snap["wire_min_half_spread"] is True
    assert snap["min_half_spread_floor_at_full_aggression_bps"] == pytest.approx(2.0)


def test_quote_decision_has_aggression_field_default_none():
    fields = QuoteDecision.__dataclass_fields__
    assert "aqc_aggression_level" in fields
    assert fields["aqc_aggression_level"].default is None


def test_settings_parses_env_aliases():
    s = _settings(AQC_WIRE_MIN_HALF_SPREAD="true",
                  AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS="1.25")
    assert s.aqc_wire_min_half_spread is True
    assert s.aqc_min_half_spread_floor_at_full_aggression_bps == pytest.approx(1.25)
