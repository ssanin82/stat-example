"""Asymmetric (nonlinear) inventory skew — ``INVENTORY_SKEW_EXPONENT``.

The exponent shapes the ramp between 0 and the coefficient (max shift
at full utilisation). At ``exponent=1.0`` the behaviour is the legacy
linear shift. At ``exponent>1.0`` the ramp is gentle near zero and
steep near the cap — ordinary inventory oscillation doesn't push the
reducing-side quote off the book.

Production code: ``app/quoting.py::compute_quote_decision``.
"""

from __future__ import annotations

import pytest

from app.models import ToxicitySnapshot
from app.quoting import compute_quote_decision
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "MAX_ABS_POSITION": 1.0,
        "INVENTORY_SKEW_COEFF_BPS": 10.0,
        "INVENTORY_SOFT_LIMIT_PCT": 0.72,
        "INVENTORY_HARD_LIMIT_PCT": 0.85,
        "BASE_HALF_SPREAD_BPS": 1.0,
        "MIN_HALF_SPREAD_BPS": 0.01,
        "MAX_HALF_SPREAD_BPS": 50.0,
        "VOL_MULTIPLIER": 0.0,
        "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 0.01,
        "ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS": 0.01,
        "TOXICITY_SCORE_HALF_SPREAD_BPS": 0.0,
        "MICROPRICE_RESERVATION_ENABLED": False,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _tox() -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.5,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=0.0,
        hard_trigger=False,
        soft_trigger=False,
    )


def _reservation(position_qty: float, *, exp: float, coeff: float = 10.0) -> float:
    settings = _settings(INVENTORY_SKEW_COEFF_BPS=coeff, INVENTORY_SKEW_EXPONENT=exp)
    d = compute_quote_decision(
        settings=settings,
        mid=2000.0,
        position_qty=position_qty,
        vol_bps=0.0,
        toxicity=_tox(),
    )
    return d.reservation_price


# ---------- exp=1.0: linear (legacy) ----------

def test_linear_legacy_matches_pre_exponent_formula() -> None:
    """``exp=1.0`` should be bit-identical to the linear skew."""
    # At 50% util, 10 bps coefficient → 5 bps shift from mid.
    r = _reservation(0.5, exp=1.0)
    expected = 2000.0 - 10.0 * 0.5 * 2000.0 / 10_000.0  # −$1.00
    assert r == pytest.approx(expected, abs=1e-9)


def test_linear_symmetric_around_zero() -> None:
    r_long = _reservation(0.3, exp=1.0)
    r_short = _reservation(-0.3, exp=1.0)
    assert (2000.0 - r_long) == pytest.approx(r_short - 2000.0, abs=1e-9)


# ---------- exp=3.0: cubic (the production setting) ----------

def test_cubic_half_util_is_quarter_of_linear() -> None:
    """|0.5|**3 / |0.5| = 0.25 → shift is one quarter the linear shift at 50% util."""
    r_linear = _reservation(0.5, exp=1.0)
    r_cubic = _reservation(0.5, exp=3.0)
    linear_shift = 2000.0 - r_linear   # +1.00 (long → reservation below mid)
    cubic_shift = 2000.0 - r_cubic
    assert cubic_shift == pytest.approx(linear_shift / 4.0, abs=1e-9)


def test_cubic_matches_linear_at_full_cap() -> None:
    """At |norm_inv|=1 the exponent collapses to the same value (1)."""
    assert _reservation(1.0, exp=3.0) == pytest.approx(_reservation(1.0, exp=1.0), abs=1e-9)
    assert _reservation(-1.0, exp=3.0) == pytest.approx(_reservation(-1.0, exp=1.0), abs=1e-9)


def test_cubic_monotonic_across_util() -> None:
    """Shift must be non-decreasing in magnitude as util grows from 0 → 1."""
    mids = [2000.0 - _reservation(u, exp=3.0) for u in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)]
    for a, b in zip(mids, mids[1:]):
        assert b >= a - 1e-12


def test_cubic_gentler_than_linear_below_cap() -> None:
    """For any 0 < |u| < 1 the cubic shift must be strictly less than linear."""
    for u in (0.1, 0.25, 0.5, 0.72, 0.85):
        lin = 2000.0 - _reservation(u, exp=1.0)
        cub = 2000.0 - _reservation(u, exp=3.0)
        assert cub < lin


def test_cubic_symmetric_around_zero() -> None:
    r_long = _reservation(0.4, exp=3.0)
    r_short = _reservation(-0.4, exp=3.0)
    assert (2000.0 - r_long) == pytest.approx(r_short - 2000.0, abs=1e-9)


# ---------- zero / edge cases ----------

def test_zero_position_no_shift_any_exp() -> None:
    for exp in (1.0, 2.0, 3.0, 5.0):
        assert _reservation(0.0, exp=exp) == pytest.approx(2000.0, abs=1e-12)


def test_over_cap_position_clipped_to_unit() -> None:
    """``norm_inv`` is clipped to ±1 before shaping — positions > cap
    don't blow past the coefficient."""
    r = _reservation(5.0, exp=3.0)   # wildly over cap
    expected_at_max = 2000.0 - 10.0 * 1.0 * 2000.0 / 10_000.0  # $2.00 shift
    assert r == pytest.approx(expected_at_max, abs=1e-9)
