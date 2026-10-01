"""Order-book imbalance alpha term on the reservation (Priority #1).

Production code: ``app/quoting.py::compute_quote_decision``,
``ob_imbalance_smoothed`` keyword argument.

Design invariants pinned here:
- At ``OB_IMBALANCE_ALPHA = 0`` (default) the term is a pure no-op, even
  when the caller supplies a valid imbalance value.
- Positive imbalance (bid-heavy book) shifts reservation up.
- Negative imbalance shifts reservation down.
- The shift is capped at ``alpha × (half_spread / 2) × clip_bound`` (in bps)
  regardless of raw input magnitude.
- Missing, non-finite, or None input is silently ignored (no crash).
- Composes additively with inventory skew and trend-drift biases.
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
        "INVENTORY_SKEW_COEFF_BPS": 0.0,
        "BASE_HALF_SPREAD_BPS": 2.0,         # exact 2 bps half-spread
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


# ---------- Feature-off / no-input behaviour ----------

def test_alpha_zero_is_legacy_even_with_input() -> None:
    """With ALPHA=0 the term is a strict no-op, regardless of input."""
    s = _settings(OB_IMBALANCE_ALPHA=0.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=0.5,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_no_input_is_noop() -> None:
    s = _settings(OB_IMBALANCE_ALPHA=0.3)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        # ob_imbalance_smoothed omitted → None → no-op
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_input_is_noop(bad) -> None:
    s = _settings(OB_IMBALANCE_ALPHA=0.3)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=bad,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Positive / negative imbalance ----------

def test_positive_imbalance_shifts_reservation_up() -> None:
    """Bid-heavy book → shift up.
    ALPHA=0.3, half_spread=2 bps, I=+0.5 (within clip of 0.85):
    shift_bps = 0.3 * 1 * 0.5 = 0.15 bps of ref (100)
    reservation = 100 + 0.15 * 100/10_000 = 100.0015
    """
    s = _settings(OB_IMBALANCE_ALPHA=0.3, OB_IMBALANCE_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=0.5,
    )
    assert d.reservation_price == pytest.approx(100.0015, abs=1e-9)


def test_negative_imbalance_shifts_reservation_down() -> None:
    s = _settings(OB_IMBALANCE_ALPHA=0.3, OB_IMBALANCE_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=-0.5,
    )
    assert d.reservation_price == pytest.approx(99.9985, abs=1e-9)


def test_symmetric_around_zero() -> None:
    s = _settings(OB_IMBALANCE_ALPHA=0.3)
    up = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=0.4,
    )
    down = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=-0.4,
    )
    assert (up.reservation_price - 100.0) == pytest.approx(100.0 - down.reservation_price, abs=1e-9)


def test_zero_imbalance_no_shift() -> None:
    s = _settings(OB_IMBALANCE_ALPHA=0.3)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=0.0,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Clipping ----------

def test_clip_bounds_extreme_positive() -> None:
    """I=+0.95 gets clipped to +0.85, shift = 0.3 * 1 * 0.85 = 0.255 bps."""
    s = _settings(OB_IMBALANCE_ALPHA=0.3, OB_IMBALANCE_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=0.95,
    )
    # Reservation shift capped at 0.255 bps of ref.
    assert d.reservation_price == pytest.approx(100.00255, abs=1e-9)


def test_clip_bounds_extreme_negative() -> None:
    s = _settings(OB_IMBALANCE_ALPHA=0.3, OB_IMBALANCE_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=-1.5,
    )
    # Clipped to -0.85, shift = 0.3 * 1 * -0.85 = -0.255 bps.
    assert d.reservation_price == pytest.approx(99.99745, abs=1e-9)


# ---------- Max shift bound ----------

def test_max_shift_is_alpha_times_half_half_spread() -> None:
    """Largest possible shift with ALPHA=0.3, CLIP=0.85, half_spread=2 bps:
    shift_bps = 0.3 * 1 * 0.85 = 0.255 bps. Below half_spread (2 bps),
    so bid stays below and ask stays above the mid in normal conditions.
    """
    s = _settings(OB_IMBALANCE_ALPHA=0.3, OB_IMBALANCE_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=1.0,  # raw, gets clipped
    )
    shift_bps = (d.reservation_price - 100.0) / 100.0 * 10_000
    assert shift_bps == pytest.approx(0.255, abs=1e-6)
    # Reservation must remain below half-spread distance from mid,
    # otherwise the ask would cross mid.
    assert abs(shift_bps) < s.base_half_spread_bps


# ---------- Composition ----------

def test_composes_with_inventory_skew() -> None:
    """Inventory skew and OB imbalance compose additively.
    Linear skew, util=0.5, coeff=10: skew shifts ref by -5 bps (long inventory).
    OB I=+0.4, alpha=0.3, half_spread=2: OB shifts ref by 0.3*1*0.4 = +0.12 bps.
    Net: -5 + 0.12 = -4.88 bps from mid 100 → 99.9512.
    """
    s = _settings(
        INVENTORY_SKEW_COEFF_BPS=10.0,
        INVENTORY_SKEW_EXPONENT=1.0,
        OB_IMBALANCE_ALPHA=0.3,
    )
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.5, vol_bps=0.0,
        toxicity=_tox(), ob_imbalance_smoothed=0.4,
    )
    assert d.reservation_price == pytest.approx(99.9512, abs=1e-9)


def test_composes_with_drift_and_basis() -> None:
    """Three alpha terms compose additively on top of skew-free base."""
    s = _settings(
        REFERENCE_VENUE_FAIR_BLEND_ALPHA=0.5,
        TREND_DRIFT_RESERVATION_ALPHA=0.5,
        OB_IMBALANCE_ALPHA=0.3,
    )
    # Blend: ref = 0.5*100 + 0.5*110 = 105.
    # Drift: +0.5 * 10 * 105 / 10000 = 0.0525.
    # OB: 0.3 * 1 * 0.4 = 0.12 bps of 105 = 0.00126 (shift_px).
    # Expected: 105 + 0.0525 + 0.00126 = 105.05376.
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        reference_fair_price=110.0,
        short_term_drift_bps=10.0,
        ob_imbalance_smoothed=0.4,
    )
    assert d.reservation_price == pytest.approx(105.05376, abs=1e-6)


# ---------- Feature-flag default + legacy preservation ----------

def test_defaults_preserve_legacy_behaviour() -> None:
    """With no new config overrides, identical input gives identical output
    regardless of whether caller supplies the new ob_imbalance kwarg."""
    s = _settings()  # ALPHA defaults to 0.0
    d1 = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
    )
    d2 = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        ob_imbalance_smoothed=0.9,
    )
    assert d1.reservation_price == pytest.approx(d2.reservation_price, abs=1e-9)
