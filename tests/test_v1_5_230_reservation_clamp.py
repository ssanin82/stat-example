"""v1.5.230 — reservation-shift clamp tests.

Closes the cap×shift interaction bug discovered in the v1.5.229
Phase 5-prime snapshot: trending regime → trend_drift_alpha pushed
reservation +8.1 bp above mid → MAX_HALF=4 cap left bid above
best_ask → bid rejected by post-only → only 4 fills in 75 min.

The fix: ``MAX_RESERVATION_SHIFT_BPS_FROM_MID`` clamps
``abs(reservation − mid)`` to a configured maximum AFTER all alpha
shifts are summed, BEFORE bid/ask prices are computed. Default 0.0
preserves legacy unclamped behaviour.
"""

from __future__ import annotations

import pytest

from app.quoting import compute_quote_decision
from app.toxicity import ToxicitySnapshot


def _tox():
    return ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.5,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=0.0,
        hard_trigger=False,
        soft_trigger=False,
    )


def _settings(**overrides):
    from app.config import Settings
    return Settings(
        VENUE="binance",
        SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0,
        MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
        TOXICITY_ENABLED=False,
        **overrides,
    )


def test_clamp_disabled_by_default_legacy_behaviour():
    """Default value 0.0 → no clamp, reservation can shift freely."""
    s = _settings(
        TREND_DRIFT_RESERVATION_ALPHA=1.0,
    )
    d = compute_quote_decision(
        settings=s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=_tox(),
        short_term_drift_bps=10.0,  # big trend → shifts reservation up
    )
    bk = d.breakdown
    assert bk.reservation_clamp_active is False
    assert abs(bk.reservation_delta_from_mid_bps) > 5.0, \
        "without clamp, big drift should produce big shift"


def test_clamp_active_when_alphas_exceed_threshold():
    """With MAX_RESERVATION_SHIFT_BPS_FROM_MID=3.0 and a big trend drift,
    the reservation should be clamped to mid ± 3 bp."""
    s = _settings(
        TREND_DRIFT_RESERVATION_ALPHA=1.0,
        MAX_RESERVATION_SHIFT_BPS_FROM_MID=3.0,
    )
    d = compute_quote_decision(
        settings=s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=_tox(),
        short_term_drift_bps=10.0,  # would shift reservation by ~10 bp
    )
    bk = d.breakdown
    assert bk.reservation_clamp_active is True
    # Clamped to +3 bp (sign matches positive drift)
    assert bk.reservation_delta_from_mid_bps == pytest.approx(3.0, abs=0.01)


def test_clamp_inactive_when_shifts_small():
    """Clamp doesn't fire when alphas keep reservation within bound."""
    s = _settings(
        TREND_DRIFT_RESERVATION_ALPHA=1.0,
        MAX_RESERVATION_SHIFT_BPS_FROM_MID=5.0,
    )
    d = compute_quote_decision(
        settings=s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=_tox(),
        short_term_drift_bps=1.0,  # tiny drift, well below clamp
    )
    bk = d.breakdown
    assert bk.reservation_clamp_active is False
    # Shift should match the drift contribution (1 bp × alpha 1.0)
    assert abs(bk.reservation_delta_from_mid_bps) < 5.0


def test_clamp_symmetric_negative_drift():
    """Clamp works equally for negative shifts (down-trending regime)."""
    s = _settings(
        TREND_DRIFT_RESERVATION_ALPHA=1.0,
        MAX_RESERVATION_SHIFT_BPS_FROM_MID=3.0,
    )
    d = compute_quote_decision(
        settings=s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=_tox(),
        short_term_drift_bps=-10.0,
    )
    bk = d.breakdown
    assert bk.reservation_clamp_active is True
    assert bk.reservation_delta_from_mid_bps == pytest.approx(-3.0, abs=0.01)


def test_clamp_prevents_cross_quote_scenario():
    """Regression test mimicking the v1.5.229 Phase 5-prime snapshot:
    mid=1.7645, big positive drift, MAX_HALF_SPREAD_BPS=4.
    Without clamp: bid lands above best_ask (cross).
    With clamp at 3 bp: bid stays at-or-below mid."""
    # Without clamp: confirm the broken behaviour
    s_no_clamp = _settings(
        TREND_DRIFT_RESERVATION_ALPHA=1.0,
        MAX_HALF_SPREAD_BPS=4.0,
        MIN_HALF_SPREAD_BPS=1.0,
    )
    d_no = compute_quote_decision(
        settings=s_no_clamp,
        mid=1.7645,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=_tox(),
        short_term_drift_bps=10.0,
    )
    # Bid is shifted high by the big drift + tight cap
    bid_no = d_no.quoted_bid
    # Bid would be above mid (cross-quote risk vs venue's inside ask)
    assert bid_no > 1.7645, \
        f"without clamp, bid {bid_no} should be above mid (cross-risk)"

    # With clamp at 3 bp: shift bounded, bid stays at-or-below mid
    s_clamp = _settings(
        TREND_DRIFT_RESERVATION_ALPHA=1.0,
        MAX_HALF_SPREAD_BPS=4.0,
        MIN_HALF_SPREAD_BPS=1.0,
        MAX_RESERVATION_SHIFT_BPS_FROM_MID=3.0,
    )
    d_clamp = compute_quote_decision(
        settings=s_clamp,
        mid=1.7645,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=_tox(),
        short_term_drift_bps=10.0,
    )
    bid_clamp = d_clamp.quoted_bid
    # With clamp 3bp + max_half 4bp, bid = mid*(1+3/10k)*(1-4/10k) ≈ mid * 0.9999
    # which is slightly BELOW mid (safe)
    assert bid_clamp < 1.7645, \
        f"with clamp, bid {bid_clamp} should be at or below mid (no cross)"


def test_clamp_telemetry_in_breakdown():
    """The clamp-active flag must be exposed in the breakdown so the
    dashboard / acceptance script can count clamp firings."""
    s = _settings(
        TREND_DRIFT_RESERVATION_ALPHA=1.0,
        MAX_RESERVATION_SHIFT_BPS_FROM_MID=3.0,
    )
    d = compute_quote_decision(
        settings=s,
        mid=100.0,
        position_qty=0.0,
        vol_bps=5.0,
        toxicity=_tox(),
        short_term_drift_bps=20.0,
    )
    assert hasattr(d.breakdown, "reservation_clamp_active")
    assert d.breakdown.reservation_clamp_active is True
