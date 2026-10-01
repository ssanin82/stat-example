"""v1.5.282 / AQC Phase 3 — aggression -> inventory exec-bias floor.

Phase 3 wires the controller's ``aggression_level`` (0..1) into the
inventory exec-bias util floor via
``compute_effective_inventory_util_floor_pct``. The wire is
TIGHTEN-ONLY (it can only LOWER the floor → exec-bias engages earlier →
faster inventory churn) and guarded by ``AQC_WIRE_INVENTORY`` (default
false), so the exec-bias gate is byte-identical to pre-Phase-3 unless
the operator flips the flag.

What's covered
==============

* Default OFF: with ``AQC_WIRE_INVENTORY=false`` the
  ``aqc_aggression_level`` kwarg is ignored for every level (0, 0.5,
  1.0, None, out-of-range) — pre-Phase-3 byte-identical.
* None-safe / NaN-safe: even with the wire ON, a None/NaN level leaves
  the floor at base (the decision wasn't stamped → no-op).
* Lerp endpoints + midpoint: aggression 0 -> base, 1 -> target,
  0.5 -> midpoint.
* Tighten-only: a mis-configured target ABOVE base can never RAISE the
  floor, and no aggression in [0,1] ever exceeds base.
* Clamp: aggression outside [0,1] is clamped before lerp.
* Engine integration: ``QuoteEngine._inventory_exec_bias_active`` reads
  the stamped ``ctx.decision.aqc_aggression_level`` — at a util between
  the base floor and the aggressive floor the gate flips from inactive
  (wire off / aggr 0) to active (wire on, aggr 1). The teeth proof.
* Ratio clamp survives: the engagement util can never drop below
  ``INVENTORY_EXEC_BIAS_RATIO`` even at a sub-ratio AQC target.
* Plumbing: AQCSettings carries the two new fields and publishes them
  in ``snapshot_dict``; ``Settings`` parses both env aliases.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.active_quoting_controller import ActiveQuotingController, AQCSettings
from app.config import Settings
from app.quote_engine import QuoteEngine
from app.quoting import compute_effective_inventory_util_floor_pct


# Prod-like TON inventory floors: base exec-bias floor 0.45, ratio 0.02.
_BASE_ENV = dict(
    SYMBOL="TON",
    INVENTORY_EXEC_BIAS_MIN_UTIL_PCT="0.45",
    INVENTORY_EXEC_BIAS_RATIO="0.02",
)

_BASE = 0.45  # INVENTORY_EXEC_BIAS_MIN_UTIL_PCT above.


def _settings(**overrides) -> Settings:
    env = dict(_BASE_ENV)
    env.update(overrides)
    return Settings(**env)


def _floor(settings: Settings, level) -> float:
    return compute_effective_inventory_util_floor_pct(
        settings, aqc_aggression_level=level
    )


# --------------------------------------------------------------------
# Default OFF — pre-Phase-3 byte-identical.
# --------------------------------------------------------------------

@pytest.mark.parametrize("level", [0.0, 0.5, 1.0, None, 2.0, -1.0])
def test_wire_off_ignores_aggression(level):
    s = _settings()  # AQC_WIRE_INVENTORY defaults false
    assert s.aqc_wire_inventory is False
    assert _floor(s, level) == pytest.approx(_BASE)


def test_wire_off_matches_no_kwarg():
    s = _settings()
    base = compute_effective_inventory_util_floor_pct(s)
    assert _floor(s, 1.0) == pytest.approx(base)
    assert base == pytest.approx(_BASE)


# --------------------------------------------------------------------
# Wire ON — lerp behavior.
# --------------------------------------------------------------------

def test_wire_on_none_is_noop():
    s = _settings(AQC_WIRE_INVENTORY="true")
    assert _floor(s, None) == pytest.approx(_BASE)


def test_wire_on_nan_is_noop():
    s = _settings(AQC_WIRE_INVENTORY="true")
    assert _floor(s, float("nan")) == pytest.approx(_BASE)


def test_wire_on_aggr_zero_is_base():
    s = _settings(AQC_WIRE_INVENTORY="true")
    assert _floor(s, 0.0) == pytest.approx(_BASE)


def test_wire_on_aggr_full_is_target():
    s = _settings(
        AQC_WIRE_INVENTORY="true",
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT="0.10",
    )
    assert _floor(s, 1.0) == pytest.approx(0.10)


def test_wire_on_aggr_half_is_midpoint():
    s = _settings(
        AQC_WIRE_INVENTORY="true",
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT="0.10",
    )
    # 0.45 + 0.5*(0.10 - 0.45) = 0.275
    assert _floor(s, 0.5) == pytest.approx(0.275)


def test_monotonic_non_increasing_in_aggression():
    s = _settings(
        AQC_WIRE_INVENTORY="true",
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT="0.10",
    )
    prev = _floor(s, 0.0)
    for lv in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        cur = _floor(s, lv)
        assert cur <= prev + 1e-12
        prev = cur


# --------------------------------------------------------------------
# Clamp + tighten-only guard.
# --------------------------------------------------------------------

def test_clamp_above_one_is_target():
    s = _settings(
        AQC_WIRE_INVENTORY="true",
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT="0.10",
    )
    assert _floor(s, 5.0) == pytest.approx(0.10)


def test_clamp_below_zero_is_base():
    s = _settings(
        AQC_WIRE_INVENTORY="true",
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT="0.10",
    )
    assert _floor(s, -3.0) == pytest.approx(_BASE)


def test_target_above_base_is_noop():
    # A mis-set endpoint ABOVE base must never RAISE the floor.
    s = _settings(
        AQC_WIRE_INVENTORY="true",
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT="0.60",
    )
    assert _floor(s, 1.0) == pytest.approx(_BASE)
    assert _floor(s, 0.5) == pytest.approx(_BASE)


def test_never_exceeds_base():
    s = _settings(
        AQC_WIRE_INVENTORY="true",
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT="0.10",
    )
    for lv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        assert _floor(s, lv) <= _BASE + 1e-12


# --------------------------------------------------------------------
# Engine integration — the teeth proof via _inventory_exec_bias_active.
# --------------------------------------------------------------------

def _engine(**overrides) -> tuple[QuoteEngine, Path]:
    from tests.exchange_client_mocks import mock_mm_client
    from tests.settings_helpers import UnitTestSettings

    path = (
        Path(tempfile.gettempdir())
        / f"mm_aqc3_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 10.0,
        "MAX_POSITION_NOTIONAL_USD": 100_000.0,
        "INVENTORY_EXEC_BIAS_RATIO": 0.02,
        "INVENTORY_EXEC_BIAS_MIN_UTIL_PCT": 0.45,
    }
    base.update(overrides)
    s = UnitTestSettings.model_validate(base)
    return QuoteEngine(s, mock_mm_client().symbol_spec), path


def test_engine_wire_off_inactive_at_mid_util():
    # util = 2.0/10 = 0.20; base floor 0.45 -> inactive regardless of aggr.
    eng, path = _engine()
    try:
        assert eng._inventory_exec_bias_active(2.0) is False
        assert (
            eng._inventory_exec_bias_active(2.0, aqc_aggression_level=1.0)
            is False
        )
    finally:
        path.unlink(missing_ok=True)


def test_engine_wire_on_full_aggression_activates_at_mid_util():
    # Wire ON, aggression 1.0 lerps floor 0.45 -> 0.10; util 0.20 >= 0.10
    # so exec-bias now engages where it would NOT at base.
    eng, path = _engine(
        AQC_WIRE_INVENTORY=True,
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT=0.10,
    )
    try:
        assert (
            eng._inventory_exec_bias_active(2.0, aqc_aggression_level=1.0)
            is True
        )
        # aggression 0 leaves floor at base -> still inactive at 0.20.
        assert (
            eng._inventory_exec_bias_active(2.0, aqc_aggression_level=0.0)
            is False
        )
        # None level (decision unstamped) -> no-op -> inactive.
        assert eng._inventory_exec_bias_active(2.0) is False
    finally:
        path.unlink(missing_ok=True)


def test_engine_ratio_clamp_holds_below_aqc_target():
    # AQC target BELOW the ratio: floor lerps to 0.0 but max(ratio, floor)
    # keeps engagement util at the 0.02 ratio. util 0.01 < 0.02 inactive;
    # util 0.03 >= 0.02 active.
    eng, path = _engine(
        AQC_WIRE_INVENTORY=True,
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT=0.0,
    )
    try:
        assert (
            eng._inventory_exec_bias_active(0.1, aqc_aggression_level=1.0)
            is False
        )  # util 0.01
        assert (
            eng._inventory_exec_bias_active(0.3, aqc_aggression_level=1.0)
            is True
        )  # util 0.03
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------
# Plumbing.
# --------------------------------------------------------------------

def test_aqc_settings_has_phase3_fields():
    s = AQCSettings()
    assert s.wire_inventory is False
    assert s.inventory_util_floor_at_full_aggression_pct == pytest.approx(0.10)


def test_snapshot_dict_publishes_phase3_keys():
    c = ActiveQuotingController(
        settings=AQCSettings(
            enabled=True,
            wire_inventory=True,
            inventory_util_floor_at_full_aggression_pct=0.12,
        )
    )
    snap = c.snapshot_dict()
    assert snap["wire_inventory"] is True
    assert snap["inventory_util_floor_at_full_aggression_pct"] == pytest.approx(
        0.12
    )


def test_settings_parses_env_aliases():
    s = _settings(
        AQC_WIRE_INVENTORY="true",
        AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT="0.15",
    )
    assert s.aqc_wire_inventory is True
    assert s.aqc_inventory_util_floor_at_full_aggression_pct == pytest.approx(
        0.15
    )


def test_settings_defaults_off():
    s = _settings()
    assert s.aqc_wire_inventory is False
    assert s.aqc_inventory_util_floor_at_full_aggression_pct == pytest.approx(
        0.10
    )
