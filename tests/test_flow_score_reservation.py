"""Flow-score alpha term on the reservation (Priority #3 v2).

Production code: ``app/quoting.py::compute_quote_decision``,
``flow_score_tfi_signed`` / ``flow_score_streak_buy`` /
``flow_score_streak_sell`` / ``flow_score_streak_window_prints``
keyword arguments.

Design invariants pinned here:
- At ``FLOW_SCORE_RESERVATION_ALPHA = 0`` (default) the term is a strict
  no-op, even when the caller supplies valid signals.
- Positive composite (buy-pressure / buy-streak) shifts reservation up;
  negative shifts down.
- The shift is bounded at ``alpha × (half_spread / 2) × clip_bound``
  regardless of raw input magnitude (single-feature saturation OR streak
  saturation OR both at once).
- Missing, non-finite, or None TFI is silently ignored (no crash).
- Composes additively with OB-imbalance and trend-drift on the same
  reservation.
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
        "BASE_HALF_SPREAD_BPS": 2.0,
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
    """With ALPHA=0 the term is a strict no-op, even given a saturated
    buy-pressure signal."""
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.0)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=0.9,
        flow_score_streak_buy=10,
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=10,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_no_input_is_noop() -> None:
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        # all flow-score args omitted
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_tfi_is_noop(bad) -> None:
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=bad,
        flow_score_streak_buy=5,
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=10,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Positive / negative composite ----------

def test_buy_pressure_shifts_reservation_up() -> None:
    """Buy-heavy TFI + buy streak → shift up.
    ALPHA=0.4, half_spread=2, TFI=0.5 (clip 0.85), streak_buy=5/10:
      tfi_clipped = 0.5
      streak_signed = 5/10 = 0.5
      composite = 0.5*0.5 + 0.5*0.5 = 0.5
      shift_bps = 0.4 * 1 * 0.5 = 0.2 bps
      reservation = 100 + 0.2 * 100/10_000 = 100.002
    """
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4, FLOW_SCORE_RESERVATION_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=0.5,
        flow_score_streak_buy=5,
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=10,
    )
    assert d.reservation_price == pytest.approx(100.002, abs=1e-9)


def test_sell_pressure_shifts_reservation_down() -> None:
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4, FLOW_SCORE_RESERVATION_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=-0.5,
        flow_score_streak_buy=0,
        flow_score_streak_sell=5,
        flow_score_streak_window_prints=10,
    )
    # symmetric to the up-shift case
    assert d.reservation_price == pytest.approx(99.998, abs=1e-9)


def test_zero_signal_is_noop() -> None:
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=0.0,
        flow_score_streak_buy=0,
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=10,
    )
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


# ---------- Bound / clip behaviour ----------

def test_tfi_above_clip_is_clipped() -> None:
    """TFI=1.0 (max) clipped to 0.85; streak=0 →
    composite = 0.5*0.85 + 0 = 0.425
    shift_bps = 0.4 * 1 * 0.425 = 0.17
    """
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4, FLOW_SCORE_RESERVATION_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=1.0,  # raw → clipped to 0.85
        flow_score_streak_buy=0,
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=10,
    )
    assert d.reservation_price == pytest.approx(100.0017, abs=1e-9)


def test_full_streak_capped_at_window() -> None:
    """Buy streak count > window is internally clamped to ±1.
    streak_buy=20, window=10 → streak_signed = 20/10 = 2.0 → clamped to 1.
    With TFI=0: composite = 0 + 0.5*1 = 0.5 → after clip(0.85): 0.5.
    shift_bps = 0.4 * 1 * 0.5 = 0.2.
    """
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4, FLOW_SCORE_RESERVATION_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=0.0,
        flow_score_streak_buy=20,  # impossibly large; clamped
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=10,
    )
    assert d.reservation_price == pytest.approx(100.002, abs=1e-9)


def test_full_saturation_caps_shift_at_clip() -> None:
    """Both signals fully saturated buy-side. composite raw =
    0.5*0.85 + 0.5*1.0 = 0.925. Re-clipped to 0.85 by the defensive
    second clamp. shift_bps = 0.4 * 1 * 0.85 = 0.34.
    """
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4, FLOW_SCORE_RESERVATION_CLIP=0.85)
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=2.0,  # clipped to 0.85
        flow_score_streak_buy=15,   # clipped to 1.0
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=10,
    )
    assert d.reservation_price == pytest.approx(100.0034, abs=1e-9)


# ---------- Composition with other alphas ----------

def test_composes_with_ob_imbalance() -> None:
    """OB-imbalance shift up + flow-score shift up = sum of both shifts.
    OB: alpha=0.3, half=2, I=0.4 → shift_ob = 0.3*1*0.4 = 0.12 bps.
    Flow: alpha=0.4, half=2, TFI=0.5, streak=0 → shift_flow = 0.4*1*(0.5*0.5) = 0.10.
    Total = 0.22 bps → reservation = 100.0022.
    """
    s = _settings(
        OB_IMBALANCE_ALPHA=0.3, OB_IMBALANCE_CLIP=0.85,
        FLOW_SCORE_RESERVATION_ALPHA=0.4, FLOW_SCORE_RESERVATION_CLIP=0.85,
    )
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        ob_imbalance_smoothed=0.4,
        flow_score_tfi_signed=0.5,
        flow_score_streak_buy=0,
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=10,
    )
    assert d.reservation_price == pytest.approx(100.0022, abs=1e-9)


def test_opposing_signals_partially_cancel() -> None:
    """OB bid-heavy (+) but flow-score sell-aggressed (−). Should net to
    something between the two endpoints (not crash, not double up)."""
    s = _settings(
        OB_IMBALANCE_ALPHA=0.3, OB_IMBALANCE_CLIP=0.85,
        FLOW_SCORE_RESERVATION_ALPHA=0.4, FLOW_SCORE_RESERVATION_CLIP=0.85,
    )
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        ob_imbalance_smoothed=0.5,         # → +0.15 bps
        flow_score_tfi_signed=-0.5,
        flow_score_streak_buy=0,
        flow_score_streak_sell=5,          # → -0.10 bps
        flow_score_streak_window_prints=10,
    )
    # Net shift = 0.15 - 0.5*0.4*1*(-0.5) wait: composite = 0.5*-0.5 + 0.5*-0.5 = -0.5
    # Flow shift = 0.4 * 1 * -0.5 = -0.20 bps
    # Net = +0.15 + (-0.20) = -0.05 bps → 100 - 0.0005 = 99.9995
    assert d.reservation_price == pytest.approx(99.9995, abs=1e-9)


def test_streak_window_zero_is_safe() -> None:
    """Defensive: streak_window_prints<2 should clamp to 2 internally
    (keeps the divisor non-zero) without crashing."""
    s = _settings(FLOW_SCORE_RESERVATION_ALPHA=0.4)
    # Should not raise, and the streak component normalises against 2.
    d = compute_quote_decision(
        settings=s, mid=100.0, position_qty=0.0, vol_bps=0.0,
        toxicity=_tox(),
        flow_score_tfi_signed=0.0,
        flow_score_streak_buy=1,
        flow_score_streak_sell=0,
        flow_score_streak_window_prints=0,
    )
    assert d.reservation_price > 100.0  # buy streak shifts up
