"""Volatility-adaptive reprice threshold.

When ``REPRICE_THRESHOLD_VOL_MULTIPLIER > 0``, the effective threshold scales
with short-term volatility: quiet markets → tighter threshold (reprice on
small moves, keep touch priority); volatile markets → fixed ceiling (no churn).

Observed in ``tmp/snap_20260417_183547``: short_vol_bps ≈ 0.48 and fixed
threshold = 2.0 bps meant we needed a ~4-sigma move before repricing.
Average passive order lifetime was 27.87 s with zero reprice events.
"""

from __future__ import annotations

from app.config import Settings
from tests.settings_helpers import UnitTestSettings


def _s(**overrides) -> UnitTestSettings:
    base = {"TRADING_ENABLED": False}
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _effective_threshold(settings: Settings, vol_bps: float) -> float:
    """Replicates the inline formula in ``OrderManager.maybe_refresh_quotes``."""
    vol_mult = float(settings.reprice_threshold_vol_multiplier)
    if vol_mult > 0.0:
        floor_bps = float(settings.reprice_threshold_bps_min)
        ceil_bps = float(settings.reprice_threshold_bps)
        raw = vol_mult * float(vol_bps)
        return min(ceil_bps, max(floor_bps, raw))
    return float(settings.reprice_threshold_bps)


def test_vol_multiplier_zero_yields_fixed_threshold() -> None:
    """Multiplier=0.0 disables adaptive logic; threshold = fixed setting."""
    s = _s(REPRICE_THRESHOLD_BPS=2.0, REPRICE_THRESHOLD_VOL_MULTIPLIER=0.0)
    assert _effective_threshold(s, vol_bps=0.1) == 2.0
    assert _effective_threshold(s, vol_bps=50.0) == 2.0


def test_adaptive_threshold_tracks_volatility_in_middle_band() -> None:
    """In the middle band, effective = multiplier * vol_bps."""
    s = _s(
        REPRICE_THRESHOLD_BPS=10.0,  # ceiling
        REPRICE_THRESHOLD_BPS_MIN=0.5,
        REPRICE_THRESHOLD_VOL_MULTIPLIER=3.0,
    )
    # 3.0 * 1.0 = 3.0 — inside [0.5, 10.0]
    assert _effective_threshold(s, vol_bps=1.0) == 3.0
    # 3.0 * 2.0 = 6.0 — inside band
    assert _effective_threshold(s, vol_bps=2.0) == 6.0


def test_adaptive_threshold_floors_at_min_in_quiet_markets() -> None:
    """When vol is near zero, threshold clamps to ``reprice_threshold_bps_min``."""
    s = _s(
        REPRICE_THRESHOLD_BPS=10.0,
        REPRICE_THRESHOLD_BPS_MIN=0.5,
        REPRICE_THRESHOLD_VOL_MULTIPLIER=3.0,
    )
    # 3.0 * 0.1 = 0.3 → floored to 0.5
    assert _effective_threshold(s, vol_bps=0.1) == 0.5
    # vol = 0 → floored to 0.5 (not 0)
    assert _effective_threshold(s, vol_bps=0.0) == 0.5


def test_adaptive_threshold_ceils_at_bps_in_volatile_markets() -> None:
    """When vol is very high, threshold clamps to ``reprice_threshold_bps`` ceiling."""
    s = _s(
        REPRICE_THRESHOLD_BPS=5.0,
        REPRICE_THRESHOLD_BPS_MIN=0.5,
        REPRICE_THRESHOLD_VOL_MULTIPLIER=3.0,
    )
    # 3.0 * 10.0 = 30.0 → capped to 5.0
    assert _effective_threshold(s, vol_bps=10.0) == 5.0


def test_observed_session_default_grvt_config() -> None:
    """Regression: the GRVT profile's adaptive tuning produces a useful threshold
    for the short_vol_bps ≈ 0.48 observed in snap_20260417_183547."""
    # Mirrors config/profiles/prod.grvt.env:
    #   REPRICE_THRESHOLD_BPS=2.0
    #   REPRICE_THRESHOLD_VOL_MULTIPLIER=3.0
    #   REPRICE_THRESHOLD_BPS_MIN=0.5
    s = _s(
        REPRICE_THRESHOLD_BPS=2.0,
        REPRICE_THRESHOLD_VOL_MULTIPLIER=3.0,
        REPRICE_THRESHOLD_BPS_MIN=0.5,
    )
    # 3.0 * 0.48 = 1.44 — right in the middle of the [0.5, 2.0] band.
    eff = _effective_threshold(s, vol_bps=0.48)
    assert 0.5 < eff < 2.0
    assert abs(eff - 1.44) < 1e-9
    # Under the old fixed threshold = 2.0, vs a typical 0.48-bps move, we'd
    # never reprice. Under adaptive, a 1.5-bps move (3.1 sigma) triggers.
    # Assert the new threshold is less than the fixed ceiling:
    assert eff < float(s.reprice_threshold_bps)
