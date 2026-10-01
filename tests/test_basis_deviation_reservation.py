"""Cross-venue basis-deviation alpha term (Priority #2).

Production code: ``app/quoting.py::compute_quote_decision``,
``cross_venue_basis_now`` and ``cross_venue_basis_ewma`` kwargs.

Design invariants pinned here:
- At ``BASIS_DEVIATION_ALPHA = 0`` (default) the term is a no-op.
- Positive deviation (GRVT rich vs Bybit-anchored fair) shifts
  reservation DOWN (mean-reversion assumption).
- Negative deviation shifts reservation UP.
- Shift magnitude is capped at ``alpha × (half_spread / 2)`` regardless
  of raw deviation magnitude.
- Missing or non-finite inputs are silently ignored.
- Composes additively with inventory skew and OB imbalance.
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
        "BASE_HALF_SPREAD_BPS": 2.0,       # 2 bps half-spread
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
    s = _settings(BASIS_DEVIATION_ALPHA=0.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.01,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_missing_basis_now_is_noop() -> None:
    s = _settings(BASIS_DEVIATION_ALPHA=0.3)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_ewma=0.01,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_missing_basis_ewma_is_noop() -> None:
    s = _settings(BASIS_DEVIATION_ALPHA=0.3)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_inputs_are_noop(bad) -> None:
    s = _settings(BASIS_DEVIATION_ALPHA=0.3)
    d1 = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=bad, cross_venue_basis_ewma=0.0,
    )
    d2 = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.0, cross_venue_basis_ewma=bad,
    )
    assert d1.reservation_price == pytest.approx(100.0, abs=1e-9)
    assert d2.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Mean-reversion sign ----------

def test_positive_deviation_shifts_reservation_down() -> None:
    """basis_now − basis_ewma = +0.01 at ref 100 → dev_bps = +1.0.
    With CLIP_BPS=2.0, dev_norm = 0.5. Shift = -0.3 * 1 * 0.5 = -0.15 bps.
    reservation = 100 + (-0.15) * 100 / 10000 = 99.9985.
    """
    s = _settings(BASIS_DEVIATION_ALPHA=0.3, BASIS_DEVIATION_CLIP_BPS=2.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.04,
    )
    assert d.reservation_price == pytest.approx(99.9985, abs=1e-9)


def test_negative_deviation_shifts_reservation_up() -> None:
    s = _settings(BASIS_DEVIATION_ALPHA=0.3, BASIS_DEVIATION_CLIP_BPS=2.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.04, cross_venue_basis_ewma=0.05,
    )
    # dev_bps = -1.0, dev_norm = -0.5, shift = +0.15 bps
    assert d.reservation_price == pytest.approx(100.0015, abs=1e-9)


def test_symmetric_around_zero_deviation() -> None:
    s = _settings(BASIS_DEVIATION_ALPHA=0.3)
    up = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.04, cross_venue_basis_ewma=0.06,
    )
    down = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.06, cross_venue_basis_ewma=0.04,
    )
    assert (up.reservation_price - 100.0) == pytest.approx(100.0 - down.reservation_price, abs=1e-9)


def test_zero_deviation_no_shift() -> None:
    s = _settings(BASIS_DEVIATION_ALPHA=0.3)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.05,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Clipping / saturation ----------

def test_huge_positive_deviation_saturates_at_max_shift() -> None:
    """Raw deviation way beyond CLIP_BPS → dev_norm = 1.0.
    shift = -0.3 * 1 * 1 = -0.3 bps (max possible).
    """
    s = _settings(BASIS_DEVIATION_ALPHA=0.3, BASIS_DEVIATION_CLIP_BPS=2.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=10.0, cross_venue_basis_ewma=0.0,  # dev = +10 px = +1000 bps
    )
    # Max shift = -0.3 bps, reservation = 100 * (1 - 0.3/10000) = 99.997
    assert d.reservation_price == pytest.approx(99.997, abs=1e-9)


def test_huge_negative_deviation_saturates() -> None:
    s = _settings(BASIS_DEVIATION_ALPHA=0.3, BASIS_DEVIATION_CLIP_BPS=2.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.0, cross_venue_basis_ewma=10.0,
    )
    assert d.reservation_price == pytest.approx(100.003, abs=1e-9)


# ---------- Composition with other terms ----------

def test_composes_with_ob_imbalance() -> None:
    """Both alpha terms active, both at 0.3 on a 2 bps half-spread.
    OB I=+0.4 → shift = +0.3 * 1 * 0.4 = +0.12 bps.
    Basis dev_norm = -0.5 → shift = -0.3 * 1 * -0.5 = +0.15 bps.
    Total: +0.27 bps → reservation = 100.0027.
    """
    s = _settings(
        OB_IMBALANCE_ALPHA=0.3,
        BASIS_DEVIATION_ALPHA=0.3,
        BASIS_DEVIATION_CLIP_BPS=2.0,
    )
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        ob_imbalance_smoothed=0.4,
        cross_venue_basis_now=0.04, cross_venue_basis_ewma=0.05,  # dev_bps = -1 → norm = -0.5
    )
    assert d.reservation_price == pytest.approx(100.0027, abs=1e-9)


def test_composes_with_inventory_skew() -> None:
    s = _settings(
        INVENTORY_SKEW_COEFF_BPS=10.0,
        INVENTORY_SKEW_EXPONENT=1.0,
        BASIS_DEVIATION_ALPHA=0.3,
        BASIS_DEVIATION_CLIP_BPS=2.0,
    )
    # Long 50% util → linear skew shifts ref by -5 bps (long inventory → reservation down)
    # Basis dev_bps = +1.0 → norm = +0.5 → shift = -0.3 * 1 * 0.5 = -0.15 bps
    # Total: -5.15 bps of 100 → reservation = 99.9485
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.5, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.04,
    )
    assert d.reservation_price == pytest.approx(99.9485, abs=1e-9)


def test_defaults_preserve_legacy_behaviour() -> None:
    """With no alphas enabled, providing basis inputs has no effect."""
    s = _settings()
    d1 = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
    )
    d2 = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.5, cross_venue_basis_ewma=0.1,
    )
    assert d1.reservation_price == pytest.approx(d2.reservation_price, abs=1e-9)
    assert d1.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Regime-sign parameter (v2) ----------

def test_regime_sign_default_is_mean_reversion() -> None:
    """Without passing ``basis_deviation_regime_sign``, default is -1.0
    (legacy mean-reversion). Positive dev → shift DOWN."""
    s = _settings(BASIS_DEVIATION_ALPHA=0.3, BASIS_DEVIATION_CLIP_BPS=2.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.04,
    )
    # dev_bps = +1 → dev_norm = +0.5 → shift with default sign -1:
    #   shift = -1 × 0.3 × 1 × 0.5 = -0.15 bps → reservation 99.9985
    assert d.reservation_price == pytest.approx(99.9985, abs=1e-9)


def test_regime_sign_positive_flips_direction() -> None:
    """With ``regime_sign = +1.0`` (trend), positive dev shifts reservation UP."""
    s = _settings(BASIS_DEVIATION_ALPHA=0.3, BASIS_DEVIATION_CLIP_BPS=2.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.04,
        basis_deviation_regime_sign=1.0,
    )
    # shift = +1 × 0.3 × 1 × 0.5 = +0.15 bps → reservation 100.0015
    assert d.reservation_price == pytest.approx(100.0015, abs=1e-9)


def test_regime_sign_zero_skips_alpha() -> None:
    """``regime_sign = 0.0`` (undecided) gates the alpha off entirely."""
    s = _settings(BASIS_DEVIATION_ALPHA=0.3, BASIS_DEVIATION_CLIP_BPS=2.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.04,
        basis_deviation_regime_sign=0.0,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_regime_sign_negative_matches_legacy() -> None:
    """Explicit ``regime_sign = -1.0`` matches the default-path result."""
    s = _settings(BASIS_DEVIATION_ALPHA=0.3, BASIS_DEVIATION_CLIP_BPS=2.0)
    d_default = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.04,
    )
    d_explicit = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        cross_venue_basis_now=0.05, cross_venue_basis_ewma=0.04,
        basis_deviation_regime_sign=-1.0,
    )
    assert d_default.reservation_price == pytest.approx(d_explicit.reservation_price, abs=1e-12)
