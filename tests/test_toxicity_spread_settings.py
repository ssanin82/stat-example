"""Toxicity-based spread widening is settings-driven, not hardcoded.

C1: ``toxicity.score * 12.0`` (hardcoded in ``compute_quote_decision``) is now
``toxicity.score * settings.toxicity_score_half_spread_bps``. Default 12 for
HL back-compat; GRVT profile drops it to match the 1-tick book.

C2: ``toxicity_one_sided_fill_ratio`` is gated by a minimum sample count
(``toxicity_one_sided_min_fills``, default 4). With fewer fills the ratio
stays 0.0 — preventing noise from a 3:1 early-session split from arming
widen.
"""

from __future__ import annotations

import pytest

from app.enums import Side
from app.models import Fill, ToxicitySnapshot
from app.quoting import compute_quote_decision
from app.toxicity import ToxicityEngine
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    """Minimal settings with the economic floor disabled so we can isolate the
    toxicity coefficient under test. These tests are checking the
    compute_quote_decision math; the economic floor has its own tests."""
    base = {
        "TRADING_ENABLED": False,
        "SYMBOL": "ETH_USDT_Perp",
        "BASE_HALF_SPREAD_BPS": 1.0,
        "MIN_HALF_SPREAD_BPS": 0.05,
        "MAX_HALF_SPREAD_BPS": 40.0,
        "VOL_MULTIPLIER": 0.0,  # isolate the toxicity component
        "MAX_ABS_POSITION": 1.0,
        # Economic floor disabled so only the model math drives half-spread.
        "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 0.05,
        "ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS": 0.05,
        "ECONOMIC_TOXICITY_SCORE_HALF_SPREAD_BPS": 0.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _tox(score: float = 0.25, soft: bool = False) -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=score,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=soft,
    )


# ------------------------- C1: toxicity coefficient ----------------------------


def test_c1_hl_default_coefficient_is_12() -> None:
    """HL-profile default: the coefficient remains 12 for back-compat."""
    s = _settings()  # defaults
    assert float(s.toxicity_score_half_spread_bps) == pytest.approx(12.0)


def test_c1_decision_uses_toxicity_score_half_spread_bps_setting() -> None:
    """At default 12, score 0.25 adds 3 bps (same as pre-fix hardcoded)."""
    s = _settings(TOXICITY_SCORE_HALF_SPREAD_BPS=12.0)
    d = compute_quote_decision(s, mid=2375.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(0.25))
    expected_half = 1.0 + 0.25 * 12.0  # base + tox
    assert d.target_spread_bps == pytest.approx(2.0 * expected_half, abs=1e-6)


def test_c1_grvt_low_coefficient_keeps_half_spread_tight() -> None:
    """GRVT profile lowers to 2.0: same toxicity score now only adds 0.5 bps."""
    s = _settings(TOXICITY_SCORE_HALF_SPREAD_BPS=2.0)
    d = compute_quote_decision(s, mid=2375.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(0.25))
    expected_half = 1.0 + 0.25 * 2.0
    assert d.target_spread_bps == pytest.approx(2.0 * expected_half, abs=1e-6)


def test_c1_soft_trigger_bump_is_configurable() -> None:
    """The hardcoded ``+= 4.0`` on soft trigger is now
    ``TOXICITY_SOFT_TRIGGER_HALF_SPREAD_BUMP_BPS``."""
    s = _settings(
        TOXICITY_SCORE_HALF_SPREAD_BPS=0.0,  # isolate the soft bump
        TOXICITY_SOFT_TRIGGER_HALF_SPREAD_BUMP_BPS=1.5,
    )
    d = compute_quote_decision(s, mid=2375.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(0.3, soft=True))
    # base 1.0 + soft_bump 1.5 = 2.5 bps half → 5.0 bps spread
    assert d.target_spread_bps == pytest.approx(5.0, abs=1e-6)


# ------------------------- C2: one-sided min fills -----------------------------


def _fill(side: Side) -> Fill:
    from datetime import datetime, timezone

    return Fill(
        fill_id=f"f{side.value}",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="ETH_USDT_Perp",
        side=side,
        price=2375.0,
        size=0.01,
        notional=23.75,
        fee=-0.0001,
        liquidity_flag="resting",
        mid_at_fill=2375.0,
    )


def test_c2_one_sided_ratio_stays_zero_below_min_fills() -> None:
    """Below TOXICITY_ONE_SIDED_MIN_FILLS, the ratio stays 0.0 regardless of
    the actual split — the 3:1 early-session noise cannot arm widen."""
    s = _settings(TOXICITY_ONE_SIDED_MIN_FILLS=10)
    eng = ToxicityEngine(s)
    # 4 fills: 3 SELL, 1 BUY — the snap_20260418_094415 shape.
    fills = [_fill(Side.SELL), _fill(Side.SELL), _fill(Side.SELL), _fill(Side.BUY)]
    snap = eng.snapshot(mid=2375.0, current_vol_bps=0.5, fills=fills)
    assert snap.one_sided_fill_ratio == 0.0


def test_c2_one_sided_ratio_computed_once_min_met() -> None:
    """At the minimum, the ratio computes normally."""
    s = _settings(TOXICITY_ONE_SIDED_MIN_FILLS=4)
    eng = ToxicityEngine(s)
    fills = [_fill(Side.SELL), _fill(Side.SELL), _fill(Side.SELL), _fill(Side.BUY)]
    snap = eng.snapshot(mid=2375.0, current_vol_bps=0.5, fills=fills)
    assert snap.one_sided_fill_ratio == pytest.approx(0.75)


def test_c2_default_min_fills_preserves_legacy_behaviour() -> None:
    """Default 4 matches the prior hardcoded threshold."""
    s = _settings()
    assert int(s.toxicity_one_sided_min_fills) == 4


def test_c2_high_min_fills_suppresses_early_session_noise() -> None:
    """GRVT profile uses 10 → a 4-fill 3:1 session yields ratio=0 (no widen)."""
    s = _settings(TOXICITY_ONE_SIDED_MIN_FILLS=10)
    eng = ToxicityEngine(s)
    fills = [_fill(Side.SELL), _fill(Side.SELL), _fill(Side.SELL), _fill(Side.BUY)]
    snap = eng.snapshot(mid=2375.0, current_vol_bps=0.5, fills=fills)
    from app.quoting import adverse_spread_widen_arm

    # adverse_spread_widen_arm checks one_sided_fill_ratio ≥ setting.
    # With ratio=0 it won't fire via the one-sided path.
    armed = adverse_spread_widen_arm(s, snap)
    # Without overlay enabled, armed is False regardless. Turn it on:
    s2 = _settings(
        TOXICITY_ONE_SIDED_MIN_FILLS=10,
        ADAPTIVE_SPREAD_ADVERSE_OVERLAY_HALF_SPREAD_BPS=4.0,
    )
    armed2 = adverse_spread_widen_arm(s2, snap)
    assert armed2 is False
