"""M8 consumer tests — the two tape-runtime-feed consumers.

Candidate B (microprice-deviation z-score widen) and Candidate A
(volatility warm-start seed) are the first two consumers of the recorder
runtime feed. Both are gated by ``REGIME_USE_RUNTIME_RECORDER_FEED`` and
default OFF, so the bot is byte-identical to pre-M8 in production today.
These tests pin the consumer contracts WITHOUT spinning up the bot loop:

* **Candidate B** is exercised through ``build_spread_composition`` with
  the same call shape ``bot.py`` uses (mirroring
  ``test_build_spread_composition_smoke.py``):
    - knob OFF + a live z ⇒ zero runtime contribution (the byte-identical
      default — this is the regression guard for the whole feature).
    - knob ON + ``z >= +threshold`` ⇒ widens ASK only.
    - knob ON + ``z <= -threshold`` ⇒ widens BID only.
    - knob ON + ``|z| < threshold`` ⇒ silent.
    - knob ON + ``z is None`` (feed dark/stale, resolved by caller) ⇒
      silent.
    - the new ``microprice_runtime_*`` keys are in ``to_dict`` and JSON-
      safe.
* **Candidate A** is exercised on ``VolatilityEstimator`` directly:
    - a real seed is served during the cold-start window.
    - a dark (``None`` / ``NaN``) field is a no-op (today's production
      state).
    - once the live estimator warms, the seed is no longer consulted
      (warm-START, not permanent override): a seeded and an unseeded
      estimator fed the same warmed series agree.
"""

from __future__ import annotations

import json
import math

from app.enums import ActiveSides, QuoteEligibility
from app.models import ToxicitySnapshot
from app.post_swing_gate import PostSwingState
from app.quote_eligibility import QuoteEligibilityResult
from app.quoting import build_spread_composition
from app.volatility import VolatilityEstimator
from app.vol_trend_gate import VolTrendState
from tests.settings_helpers import UnitTestSettings


# --------------------------------------------------------------------------
# Shared builders (mirror test_build_spread_composition_smoke.py)
# --------------------------------------------------------------------------


def _settings(*, feed_on: bool) -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "MAX_ABS_POSITION": 10.0,
            "MAX_POSITION_NOTIONAL_USD": 100.0,
            "MAX_HALF_SPREAD_BPS": 30.0,
            "REGIME_USE_RUNTIME_RECORDER_FEED": feed_on,
            "MICROPRICE_Z_WIDEN_THRESHOLD": 1.0,
            "MICROPRICE_Z_WIDEN_BPS": 4.0,
        }
    )


def _clean_eligibility() -> QuoteEligibilityResult:
    return QuoteEligibilityResult(
        eligibility=QuoteEligibility.QUOTE_BOTH,
        reason="ok|fresh=freshness_ok|drift=drift_ok",
        seconds_since_last_public_book_update=0.0,
        effective_staleness_ms=0.0,
        market_data_gap_p95_ms=0.0,
        market_data_gap_median_ms=0.0,
        mid_return_100ms_bps=None,
        mid_return_250ms_bps=None,
        mid_return_500ms_bps=None,
        jump_100ms_bps=None,
        jump_250ms_bps=None,
        jump_500ms_bps=None,
        in_cooldown=False,
    )


def _tox() -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
    )


def _build(*, feed_on: bool, z, position_qty: float = 0.0):
    s = _settings(feed_on=feed_on)
    return build_spread_composition(
        settings=s,
        toxicity=_tox(),
        active_sides=ActiveSides.BOTH,
        spread_floor_overlay_half_spread_bps=0.0,
        vol_trend_state=VolTrendState(),
        post_swing_state=PostSwingState(),
        ob_imbalance_ewma=0.0,  # keep the in-process microprice gate silent
        basis_regime_last_ic=0.5,
        basis_regime_pair_count=100,
        position_qty=position_qty,
        effective_abs_cap=10.0,
        drift_bps=None,
        raw_eligibility=_clean_eligibility(),
        effective_eligibility=_clean_eligibility(),
        now_mono=100.0,
        microprice_dev_z_runtime=z,
    )


# --------------------------------------------------------------------------
# Candidate B — microprice-z widen through the builder
# --------------------------------------------------------------------------


def test_candidate_b_knob_off_is_byte_identical() -> None:
    # The whole-feature regression guard: even a strong live z must not
    # move the composition when the feed knob is OFF (the default).
    comp = _build(feed_on=False, z=5.0)
    assert comp.microprice_runtime_bid_bps == 0.0
    assert comp.microprice_runtime_ask_bps == 0.0
    # And the only contributor is still the econ floor.
    assert comp.total_bid_bps_uncapped() == comp.econ_floor_bps
    assert comp.total_ask_bps_uncapped() == comp.econ_floor_bps


def test_candidate_b_ask_thin_widens_ask_only() -> None:
    comp = _build(feed_on=True, z=2.5)  # +z ⇒ bid-heavy ⇒ ask thin
    assert comp.microprice_runtime_bid_bps == 0.0
    assert comp.microprice_runtime_ask_bps == 4.0
    # Adds on top of econ floor on the ask side only.
    assert comp.total_ask_bps_uncapped() == comp.econ_floor_bps + 4.0
    assert comp.total_bid_bps_uncapped() == comp.econ_floor_bps


def test_candidate_b_bid_thin_widens_bid_only() -> None:
    comp = _build(feed_on=True, z=-2.5)  # -z ⇒ ask-heavy ⇒ bid thin
    assert comp.microprice_runtime_bid_bps == 4.0
    assert comp.microprice_runtime_ask_bps == 0.0


def test_candidate_b_below_threshold_silent() -> None:
    comp = _build(feed_on=True, z=0.4)  # |z| < 1.0
    assert comp.microprice_runtime_bid_bps == 0.0
    assert comp.microprice_runtime_ask_bps == 0.0


def test_candidate_b_none_z_silent() -> None:
    # Caller passes None when the feed is off / dark / stale.
    comp = _build(feed_on=True, z=None)
    assert comp.microprice_runtime_bid_bps == 0.0
    assert comp.microprice_runtime_ask_bps == 0.0


def test_candidate_b_inventory_direction_suppression() -> None:
    # Bot LONG + ask thin (z positive): ask is the inventory-reducing
    # side → BUG-026 structural suppression (default sentinel) zeroes the
    # widen, identical to the OB-imbalance sibling.
    comp = _build(feed_on=True, z=2.5, position_qty=5.0)
    assert comp.microprice_runtime_ask_bps == 0.0
    assert comp.microprice_runtime_bid_bps == 0.0


def test_candidate_b_to_dict_has_keys_and_is_json_safe() -> None:
    comp = _build(feed_on=True, z=2.5)
    d = comp.to_dict()
    assert "microprice_runtime_bid_bps" in d
    assert "microprice_runtime_ask_bps" in d
    assert d["microprice_runtime_ask_bps"] == 4.0
    json.dumps(d)  # must be serialisable for the heartbeat payload


# --------------------------------------------------------------------------
# Candidate A — volatility warm-start seed
# --------------------------------------------------------------------------


def _vol() -> VolatilityEstimator:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "MAX_ABS_POSITION": 10.0,
            "MAX_POSITION_NOTIONAL_USD": 100.0,
        }
    )
    return VolatilityEstimator(s)


def test_candidate_a_seed_served_during_cold_start() -> None:
    v = _vol()
    assert v.sigma_and_bps() == (None, 0.0)  # cold, no seed
    assert v.seed_from_recorder(20.0) is True
    sigma, bps = v.sigma_and_bps()
    assert sigma == 0.002  # 20 bps / 10_000
    assert bps == 20.0


def test_candidate_a_dark_field_is_noop() -> None:
    v = _vol()
    assert v.seed_from_recorder(None) is False  # NaN field today
    assert v.seed_from_recorder(float("nan")) is False
    assert v.seed_from_recorder(-1.0) is False
    assert v.seed_from_recorder(0.0) is False
    assert v.sigma_and_bps() == (None, 0.0)


def test_candidate_a_seed_superseded_once_warmed() -> None:
    # The seed is a warm-START, not a permanent override: once the live
    # estimator has enough distinct mids, the seed is never consulted.
    # Property: a seeded and an unseeded estimator fed the IDENTICAL
    # warmed series produce the SAME sigma.
    seeded = _vol()
    unseeded = _vol()
    seeded.seed_from_recorder(999.0)  # absurd seed; must be ignored once warm
    mids = [100.0 + i * 0.5 for i in range(seeded._window + 5)]
    for m in mids:
        seeded.push_mid(m)
        unseeded.push_mid(m)
    assert seeded.warmed_up is True
    assert unseeded.warmed_up is True
    s_sigma, s_bps = seeded.sigma_and_bps()
    u_sigma, u_bps = unseeded.sigma_and_bps()
    assert s_sigma is not None and u_sigma is not None
    assert math.isclose(s_sigma, u_sigma, rel_tol=1e-12)
    assert math.isclose(s_bps, u_bps, rel_tol=1e-12)
