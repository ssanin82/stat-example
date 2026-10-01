"""Adaptive reservation biases — Tier 1 levers #7 and #1.

Production code: ``app/quoting.py::compute_quote_decision``.

- Lever #7 — reference-venue fair-value blend.
  ``REFERENCE_VENUE_FAIR_BLEND_ALPHA`` weights the reservation between
  the local (GRVT) mid/microprice and a caller-supplied cross-venue
  fair price (Bybit mid + smoothed basis). At alpha=0 the legacy
  reservation is preserved; at alpha=1 the reservation is fully
  anchored on the reference.

- Lever #1 — short-term drift bias.
  ``TREND_DRIFT_RESERVATION_ALPHA`` shifts the reservation by
  ``alpha * drift_bps * ref / 10_000``. At alpha=0 legacy; positive
  drift (market rising) pushes reservation up.

Both biases compose additively with the existing inventory skew.
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
        "INVENTORY_SKEW_COEFF_BPS": 0.0,      # isolate bias effects
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


# ---------- Lever #7 — reference-venue fair-value blend ----------

def test_blend_alpha_zero_is_legacy() -> None:
    """alpha=0 ⇒ reservation depends only on GRVT mid, reference ignored."""
    s = _settings(REFERENCE_VENUE_FAIR_BLEND_ALPHA=0.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), reference_fair_price=110.0,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_blend_alpha_one_fully_anchors_on_reference() -> None:
    """alpha=1 ⇒ reservation = reference_fair_price (ignoring GRVT mid)."""
    s = _settings(REFERENCE_VENUE_FAIR_BLEND_ALPHA=1.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), reference_fair_price=110.0,
    )
    assert d.reservation_price == pytest.approx(110.0, abs=1e-9)


def test_blend_alpha_half_midway() -> None:
    """alpha=0.5 ⇒ halfway between GRVT mid and reference."""
    s = _settings(REFERENCE_VENUE_FAIR_BLEND_ALPHA=0.5)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), reference_fair_price=110.0,
    )
    assert d.reservation_price == pytest.approx(105.0, abs=1e-9)


def test_blend_no_reference_supplied_is_noop() -> None:
    """Missing reference_fair_price → behave as if alpha=0 (no crash, legacy)."""
    s = _settings(REFERENCE_VENUE_FAIR_BLEND_ALPHA=0.5)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("bad_ref", [float("nan"), float("inf"), -50.0, 0.0])
def test_blend_rejects_non_finite_or_non_positive_reference(bad_ref) -> None:
    """Silently no-op on malformed reference price (no crash, legacy behaviour)."""
    s = _settings(REFERENCE_VENUE_FAIR_BLEND_ALPHA=0.5)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), reference_fair_price=bad_ref,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Lever #1 — short-term drift bias ----------

def test_drift_alpha_zero_is_legacy() -> None:
    s = _settings(TREND_DRIFT_RESERVATION_ALPHA=0.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), short_term_drift_bps=5.0,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_positive_drift_shifts_reservation_up() -> None:
    """Drift +10 bps with alpha=0.5 → reservation += 0.5·10·100/10000 = +0.05."""
    s = _settings(TREND_DRIFT_RESERVATION_ALPHA=0.5)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), short_term_drift_bps=10.0,
    )
    assert d.reservation_price == pytest.approx(100.05, abs=1e-9)


def test_negative_drift_shifts_reservation_down() -> None:
    s = _settings(TREND_DRIFT_RESERVATION_ALPHA=0.5)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), short_term_drift_bps=-10.0,
    )
    assert d.reservation_price == pytest.approx(99.95, abs=1e-9)


def test_drift_none_is_noop() -> None:
    s = _settings(TREND_DRIFT_RESERVATION_ALPHA=0.5)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), short_term_drift_bps=None,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("bad_drift", [float("nan"), float("inf"), float("-inf")])
def test_drift_rejects_non_finite(bad_drift) -> None:
    s = _settings(TREND_DRIFT_RESERVATION_ALPHA=0.5)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(), short_term_drift_bps=bad_drift,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Composition ----------

def test_blend_and_drift_compose_additively() -> None:
    """Both biases active → reservation = blend(mid, ref) + drift_shift."""
    s = _settings(
        REFERENCE_VENUE_FAIR_BLEND_ALPHA=0.5,
        TREND_DRIFT_RESERVATION_ALPHA=0.5,
    )
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        reference_fair_price=110.0,
        short_term_drift_bps=10.0,
    )
    # Blend: 0.5*100 + 0.5*110 = 105.
    # Drift: +0.5*10*105/10000 = +0.0525.
    # (drift applied to blended ref_price, not raw mid.)
    assert d.reservation_price == pytest.approx(105.0525, abs=1e-9)


def test_bias_composes_with_inventory_skew() -> None:
    """Inventory skew must still apply on top of the adaptive biases."""
    s = _settings(
        INVENTORY_SKEW_COEFF_BPS=10.0,
        INVENTORY_SKEW_EXPONENT=1.0,    # linear for simplicity
        REFERENCE_VENUE_FAIR_BLEND_ALPHA=0.5,
        TREND_DRIFT_RESERVATION_ALPHA=0.0,
    )
    # Long 50 % util with blend alpha 0.5 and ref 110:
    #   ref_price = 0.5*100 + 0.5*110 = 105
    #   skew      = 10 * 0.5 = 5 bps of ref = 0.0525
    #   reservation = 105 - 0.0525 = 104.9475
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.5, vol_bps=0.0,
        toxicity=_tox(), reference_fair_price=110.0,
    )
    assert d.reservation_price == pytest.approx(104.9475, abs=1e-9)


def test_defaults_preserve_legacy_behaviour() -> None:
    """With new config at defaults (0.0), compute_quote_decision is unchanged."""
    s = _settings()  # no overrides → both alphas default to 0.0
    d1 = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
    )
    # Even when caller supplies the new inputs, they're ignored at alpha=0.
    d2 = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        reference_fair_price=150.0,
        short_term_drift_bps=42.0,
    )
    assert d1.reservation_price == pytest.approx(d2.reservation_price, abs=1e-9)
    assert d1.reservation_price == pytest.approx(100.0, abs=1e-9)
