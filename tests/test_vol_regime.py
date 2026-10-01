"""Vol-spike runtime adapter (BUGS/todo-009.md).

Pure-function tests for ``compute_vol_regime_adjustment`` plus
behavior tests through ``compute_quote_decision`` that exercise the
shrink_factor and half_spread_bump effects on real quote outputs.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.config import Settings
from app.enums import ActiveSides, Side
from app.models import ToxicitySnapshot
from app.quoting import compute_quote_decision
from app.vol_regime import (
    VolRegimeAdjustment,
    compute_vol_regime_adjustment,
)
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> Settings:
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": "sqlite:///:memory:",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _tox(score: float = 0.0, vol_ratio: float = 1.0) -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=score,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=vol_ratio,
        hard_trigger=False,
        soft_trigger=False,
    )


# ---------------------------------------------------------------------------
# Pure function: feature off
# ---------------------------------------------------------------------------


def test_feature_off_returns_identity_adjustment() -> None:
    """``VOL_SHRINK_COEFF=0`` (default) must short-circuit to a
    no-op adjustment. Critical for backward compatibility — existing
    profiles without the feature env vars must behave exactly as
    before."""
    s = _settings()
    assert s.vol_shrink_coeff == 0.0  # default
    adj, new_until = compute_vol_regime_adjustment(
        s, vol_ratio=10.0, now_mono=1000.0, vol_spike_until_mono=0.0
    )
    assert adj.shrink_factor == 1.0
    assert adj.in_spike_window is False
    assert adj.half_spread_bump_bps == 0.0
    assert new_until == 0.0


def test_feature_off_does_not_open_spike_window_even_on_huge_ratio() -> None:
    """Spike window only opens when the SIZING feature is also on —
    they live and die together. Otherwise an operator who set
    threshold=2.0 but forgot to set shrink_coeff would get a
    spread-only effect with no telemetry of why."""
    s = _settings(VOL_SPIKE_THRESHOLD=1.5, VOL_SPIKE_COOLDOWN_SECONDS=60.0)
    adj, new_until = compute_vol_regime_adjustment(
        s, vol_ratio=10.0, now_mono=1000.0, vol_spike_until_mono=0.0
    )
    assert adj.in_spike_window is False
    assert new_until == 0.0


# ---------------------------------------------------------------------------
# Pure function: shrink factor
# ---------------------------------------------------------------------------


def test_calm_regime_returns_full_size() -> None:
    """``vol_ratio=1.0`` (current vol == baseline) → no shrink."""
    s = _settings(VOL_SHRINK_COEFF=0.25)
    adj, _ = compute_vol_regime_adjustment(
        s, vol_ratio=1.0, now_mono=1000.0, vol_spike_until_mono=0.0
    )
    assert adj.shrink_factor == pytest.approx(1.0)


def test_subbaseline_vol_clamped_to_1_no_growth() -> None:
    """``vol_ratio < 1.0`` (calmer than baseline) MUST NOT grow caps.
    The feature only ever shrinks; calm just means full size."""
    s = _settings(VOL_SHRINK_COEFF=0.5)
    adj, _ = compute_vol_regime_adjustment(
        s, vol_ratio=0.5, now_mono=1000.0, vol_spike_until_mono=0.0
    )
    assert adj.shrink_factor == pytest.approx(1.0)


def test_shrink_factor_linear_in_excess_ratio() -> None:
    """Formula: ``1 - coeff * (ratio - 1)``, clipped to [floor, 1.0]."""
    s = _settings(VOL_SHRINK_COEFF=0.25, VOL_SHRINK_FLOOR=0.1)
    # ratio=3 → 1 - 0.25*(3-1) = 0.5
    adj, _ = compute_vol_regime_adjustment(
        s, vol_ratio=3.0, now_mono=1000.0, vol_spike_until_mono=0.0
    )
    assert adj.shrink_factor == pytest.approx(0.5)


def test_shrink_factor_clipped_to_floor() -> None:
    """Runaway vol must not collapse caps below the operator-set
    floor — leaves the venue self-heal min-notional path room to
    work."""
    s = _settings(VOL_SHRINK_COEFF=1.0, VOL_SHRINK_FLOOR=0.3)
    # ratio=10 → 1 - 1.0*(10-1) = -8 → clipped to 0.3
    adj, _ = compute_vol_regime_adjustment(
        s, vol_ratio=10.0, now_mono=1000.0, vol_spike_until_mono=0.0
    )
    assert adj.shrink_factor == pytest.approx(0.3)


def test_none_or_invalid_vol_ratio_treated_as_calm() -> None:
    """Defensive: missing toxicity snapshot or NaN values must not
    crash the adapter. Calm-default keeps the bot in normal sizing."""
    s = _settings(VOL_SHRINK_COEFF=0.5)
    for v in [None, float("nan"), -1.0]:
        adj, _ = compute_vol_regime_adjustment(
            s, vol_ratio=v, now_mono=1000.0, vol_spike_until_mono=0.0
        )
        assert adj.shrink_factor == pytest.approx(1.0), f"vr={v} should be calm"


# ---------------------------------------------------------------------------
# Pure function: spike window persistence
# ---------------------------------------------------------------------------


def test_spike_window_opens_at_threshold_crossing() -> None:
    """Once ``vol_ratio >= threshold``, ``in_spike_window`` is True
    and ``new_until`` advances by ``cooldown_seconds``."""
    s = _settings(
        VOL_SHRINK_COEFF=0.25,
        VOL_SPIKE_THRESHOLD=2.5,
        VOL_SPIKE_COOLDOWN_SECONDS=60.0,
    )
    adj, new_until = compute_vol_regime_adjustment(
        s, vol_ratio=3.0, now_mono=1000.0, vol_spike_until_mono=0.0
    )
    assert adj.in_spike_window is True
    assert new_until == pytest.approx(1060.0)


def test_spike_window_persists_after_vol_calms() -> None:
    """The whole point of the persistence cooldown: even when vol
    drops back to baseline, ``in_spike_window`` stays True until
    the deadline expires. This is what captures the post-spike
    bounce-and-retest pattern."""
    s = _settings(
        VOL_SHRINK_COEFF=0.25,
        VOL_SPIKE_THRESHOLD=2.5,
        VOL_SPIKE_COOLDOWN_SECONDS=60.0,
        VOL_SPIKE_HALF_SPREAD_BUMP_BPS=1.0,
    )
    # vol_ratio back to 1 (calm) but window not yet expired
    adj, new_until = compute_vol_regime_adjustment(
        s, vol_ratio=1.0, now_mono=1030.0, vol_spike_until_mono=1060.0
    )
    assert adj.in_spike_window is True
    assert adj.half_spread_bump_bps == pytest.approx(1.0)
    # Calm vol → shrink relaxes back to 1.0; only the spread
    # defense persists during the cooldown.
    assert adj.shrink_factor == pytest.approx(1.0)
    # ``new_until`` not advanced (vol_ratio < threshold).
    assert new_until == pytest.approx(1060.0)


def test_spike_window_expires_after_cooldown() -> None:
    """``now_mono >= until`` → ``in_spike_window`` False, bump zero."""
    s = _settings(
        VOL_SHRINK_COEFF=0.25,
        VOL_SPIKE_THRESHOLD=2.5,
        VOL_SPIKE_COOLDOWN_SECONDS=60.0,
        VOL_SPIKE_HALF_SPREAD_BUMP_BPS=1.0,
    )
    adj, new_until = compute_vol_regime_adjustment(
        s, vol_ratio=1.0, now_mono=1100.0, vol_spike_until_mono=1060.0
    )
    assert adj.in_spike_window is False
    assert adj.half_spread_bump_bps == 0.0
    # ``new_until`` still in the past — caller will keep writing it
    # until something clamps it; that's fine.
    assert new_until == pytest.approx(1060.0)


def test_threshold_crossing_extends_existing_window() -> None:
    """Re-arming during an active window should push the deadline
    out, not reset it. ``max(existing, now+cooldown)`` semantics."""
    s = _settings(
        VOL_SHRINK_COEFF=0.25,
        VOL_SPIKE_THRESHOLD=2.5,
        VOL_SPIKE_COOLDOWN_SECONDS=60.0,
    )
    # Existing window expires at 1060; vol re-spikes at 1030 → new
    # deadline = max(1060, 1030+60) = 1090.
    adj, new_until = compute_vol_regime_adjustment(
        s, vol_ratio=3.0, now_mono=1030.0, vol_spike_until_mono=1060.0
    )
    assert adj.in_spike_window is True
    assert new_until == pytest.approx(1090.0)


def test_half_spread_bump_zero_outside_window_even_when_set() -> None:
    s = _settings(
        VOL_SHRINK_COEFF=0.25,
        VOL_SPIKE_THRESHOLD=2.5,
        VOL_SPIKE_COOLDOWN_SECONDS=60.0,
        VOL_SPIKE_HALF_SPREAD_BUMP_BPS=2.5,
    )
    # No prior window, vol calm → no bump.
    adj, _ = compute_vol_regime_adjustment(
        s, vol_ratio=1.5, now_mono=1000.0, vol_spike_until_mono=0.0
    )
    assert adj.in_spike_window is False
    assert adj.half_spread_bump_bps == 0.0


# ---------------------------------------------------------------------------
# Behavior through compute_quote_decision: shrink_factor cuts size
# ---------------------------------------------------------------------------


def test_quote_decision_size_cut_by_shrink_factor() -> None:
    """End-to-end: a non-1.0 ``vol_regime_shrink_factor`` produces
    smaller ``quoted_*_sz`` than the same call with default 1.0.
    Composes with toxicity sizing (here zero toxicity)."""
    s = _settings(
        QUOTE_NOTIONAL_USD=100.0,
        MIN_QUOTE_NOTIONAL_USD=10.0,
    )
    full = compute_quote_decision(
        s, mid=1.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox()
    )
    half = compute_quote_decision(
        s,
        mid=1.0,
        position_qty=0.0,
        vol_bps=0.0,
        toxicity=_tox(),
        vol_regime_shrink_factor=0.5,
    )
    assert half.quoted_bid_sz == pytest.approx(full.quoted_bid_sz * 0.5)
    assert half.quoted_ask_sz == pytest.approx(full.quoted_ask_sz * 0.5)
    assert "vol_shrink" in half.decision_reason
    assert "vol_shrink" not in full.decision_reason


def test_quote_decision_shrink_floored_by_min_notional() -> None:
    """The shrink factor is re-clipped to the venue self-heal floor
    (``min_quote_notional / quote_notional``) so the venue's
    min-notional gate doesn't suppress the side."""
    s = _settings(
        QUOTE_NOTIONAL_USD=100.0,
        MIN_QUOTE_NOTIONAL_USD=50.0,  # floor at 0.5
    )
    decision = compute_quote_decision(
        s,
        mid=1.0,
        position_qty=0.0,
        vol_bps=0.0,
        toxicity=_tox(),
        vol_regime_shrink_factor=0.1,  # would push way below floor
    )
    # bid_sz * mid >= 50 (the min_quote_notional)
    assert decision.quoted_bid_sz * 1.0 >= 50.0 - 1e-9


def test_quote_decision_shrink_compose_with_toxicity_sizing() -> None:
    """Toxicity sizing and vol-shrink stack multiplicatively."""
    s = _settings(
        QUOTE_NOTIONAL_USD=100.0,
        MIN_QUOTE_NOTIONAL_USD=5.0,  # very low floor so we observe full effect
        TOXICITY_SIZE_REDUCTION_COEFF=1.0,
    )
    # tox.score=0.5 → tox sizing 0.5; vol_shrink 0.5 → combined 0.25
    tox_only = compute_quote_decision(
        s,
        mid=1.0,
        position_qty=0.0,
        vol_bps=0.0,
        toxicity=_tox(score=0.5),
    )
    both = compute_quote_decision(
        s,
        mid=1.0,
        position_qty=0.0,
        vol_bps=0.0,
        toxicity=_tox(score=0.5),
        vol_regime_shrink_factor=0.5,
    )
    # Composing vol_shrink=0.5 on top of toxicity-reduced size halves
    # it again (modulo a re-clip to the venue self-heal floor, which
    # is 0.05 here so doesn't bind).
    assert both.quoted_bid_sz == pytest.approx(tox_only.quoted_bid_sz * 0.5)


# ---------------------------------------------------------------------------
# Behavior through compute_quote_decision: half_spread bump
# ---------------------------------------------------------------------------


def test_quote_decision_half_spread_lifts_under_bump() -> None:
    """Spread bump while in spike window should push the eff_min
    floor, which is reflected in the gross spread."""
    s = _settings(
        BASE_HALF_SPREAD_BPS=1.0,
        MIN_HALF_SPREAD_BPS=0.5,
        MAX_HALF_SPREAD_BPS=30.0,
        ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS=1.0,
        ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS=1.0,
    )
    no_bump = compute_quote_decision(
        s, mid=1.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox()
    )
    with_bump = compute_quote_decision(
        s,
        mid=1.0,
        position_qty=0.0,
        vol_bps=0.0,
        toxicity=_tox(),
        vol_regime_half_spread_bump_bps=2.0,
    )
    # The bump should produce a wider gross spread.
    no_bump_spread_bps = (
        (no_bump.quoted_ask - no_bump.quoted_bid) / no_bump.mid_price * 10_000.0
    )
    with_bump_spread_bps = (
        (with_bump.quoted_ask - with_bump.quoted_bid)
        / with_bump.mid_price
        * 10_000.0
    )
    assert with_bump_spread_bps > no_bump_spread_bps + 1e-9
    # Diagnostic reason recorded.
    assert "vol_spike_window" in with_bump.decision_reason


# ---------------------------------------------------------------------------
# Wiring sentinel: Bot.one_tick computes + persists adjustment
# ---------------------------------------------------------------------------


def test_bot_one_tick_calls_vol_regime_compute_and_persists() -> None:
    """Source-substring guard: ``Bot.one_tick`` must call
    ``compute_vol_regime_adjustment`` and write both
    ``vol_spike_until_mono`` and ``vol_regime_adjustment`` back to
    state. Catches a future refactor that silently drops the wiring
    (the feature is a no-op when ``VOL_SHRINK_COEFF=0``, so a
    silent regression is otherwise undetectable from PnL alone)."""
    import inspect

    from app.bot import Bot

    src = inspect.getsource(Bot.one_tick)
    assert "compute_vol_regime_adjustment" in src
    assert "vol_spike_until_mono" in src
    assert "vol_regime_adjustment" in src


def test_bot_passes_adjustment_into_compute_quote_decision() -> None:
    """The adjustment is read by quoting.py only via the
    ``vol_regime_shrink_factor`` + ``vol_regime_half_spread_bump_bps``
    kwargs on ``compute_quote_decision``. If the bot stops passing
    them, the feature dies silently."""
    import inspect

    from app.bot import Bot

    src = inspect.getsource(Bot.one_tick)
    assert "vol_regime_shrink_factor" in src
    assert "vol_regime_half_spread_bump_bps" in src
