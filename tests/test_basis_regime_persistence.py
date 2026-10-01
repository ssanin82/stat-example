"""Tests for N12 wiring — basis-regime persist + restore.

Closes ``plans/telemetry.md`` §2 (Step 1). The schema for the
``PersistentRuntimeState.basis_regime_*`` fields was already in
place; this verifies the end-to-end wiring:

  1. ``BotState.build_persistent_runtime_state()`` reads from
     ``state.basis_regime`` and populates the three fields.
  2. ``BotState.apply_persistent_runtime_state(loaded)`` seeds the
     classifier from the persisted fields.
  3. Round-trip preserves values.
  4. The classifier's diagnostic outputs (last_regime_sign /
     last_ic / pair_count) reflect the seeded values immediately
     after restart.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.basis_regime import BasisRegimeClassifier
from app.persistent_runtime_state import PersistentRuntimeState
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings() -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )


# ---------------------------------------------------------------------------
# BasisRegimeClassifier.seed_from_persisted
# ---------------------------------------------------------------------------


def test_seed_from_persisted_restores_sign_ic_and_pair_count() -> None:
    cls = BasisRegimeClassifier(
        horizon_seconds=2.0,
        window_samples=240,
        ic_threshold=0.15,
        min_pair_samples=50,
    )
    cls.seed_from_persisted(
        last_regime_sign=-1.0,
        last_ic=-0.42,
        pair_count=237,
    )
    assert cls.last_regime_sign == -1.0
    assert cls.last_ic == -0.42
    assert cls.pair_count == 237


def test_seed_from_persisted_canonicalizes_drifty_sign() -> None:
    """Defensive: even if persisted sign is e.g. 0.7 instead of {-1, 0, +1},
    seeding coerces it back to canonical."""
    cls = BasisRegimeClassifier(
        horizon_seconds=2.0,
        window_samples=240,
        ic_threshold=0.15,
        min_pair_samples=50,
    )
    cls.seed_from_persisted(last_regime_sign=0.7, last_ic=0.2, pair_count=10)
    assert cls.last_regime_sign == 1.0  # canonicalized

    cls.seed_from_persisted(last_regime_sign=-0.7, last_ic=-0.2, pair_count=10)
    assert cls.last_regime_sign == -1.0  # canonicalized

    cls.seed_from_persisted(last_regime_sign=0.3, last_ic=0.0, pair_count=10)
    assert cls.last_regime_sign == 0.0  # zero band


def test_seed_from_persisted_ignores_invalid_fields() -> None:
    """None / NaN / wrong-type fields don't poison the classifier."""
    import math

    cls = BasisRegimeClassifier(
        horizon_seconds=2.0,
        window_samples=240,
        ic_threshold=0.15,
        min_pair_samples=50,
    )
    cls.seed_from_persisted(last_regime_sign=1.0, last_ic=0.5, pair_count=100)
    # Now pass garbage — values should NOT change.
    cls.seed_from_persisted(
        last_regime_sign=float("nan"),
        last_ic=float("inf"),
        pair_count=-5,
    )
    assert cls.last_regime_sign == 1.0
    assert cls.last_ic == 0.5
    assert cls.pair_count == 100

    cls.seed_from_persisted(last_regime_sign=None, last_ic=None, pair_count=None)
    # Nones leave fields unchanged.
    assert cls.last_regime_sign == 1.0
    assert cls.last_ic == 0.5
    assert cls.pair_count == 100


# ---------------------------------------------------------------------------
# BotState round-trip
# ---------------------------------------------------------------------------


def test_build_persistent_runtime_state_captures_basis_regime() -> None:
    s = _settings()
    state = BotState(s)
    # Manually seed the classifier to a known state.
    state.basis_regime.seed_from_persisted(
        last_regime_sign=-1.0,
        last_ic=-0.31,
        pair_count=215,
    )
    prs = state.build_persistent_runtime_state()
    assert prs.basis_regime_last_sign == -1.0
    assert prs.basis_regime_last_ic == -0.31
    assert prs.basis_regime_pair_count == 215


def test_apply_persistent_runtime_state_seeds_basis_regime() -> None:
    """Loading a persisted state restores the classifier's diagnostic
    fields on startup. Schedule today so the day-anchor branch runs
    (otherwise day-scoped counters get reset and we'd need to verify
    a different branch)."""
    s = _settings()
    state = BotState(s)
    today = state.operator_day_anchor_utc  # today's UTC date
    loaded = PersistentRuntimeState(
        day_anchor_utc=today,
        daily_realized_pnl=0.0,
        daily_trade_count=0,
        daily_traded_notional=0.0,
        last_fill_ts=None,
        recent_buy_fill_count=0,
        recent_sell_fill_count=0,
        rolling_toxicity_markout_bps=None,
        rolling_one_sided_fill_ratio=None,
        soft_flatten_active=False,
        soft_flatten_started_at_iso=None,
        basis_regime_last_sign=1.0,
        basis_regime_last_ic=0.27,
        basis_regime_pair_count=180,
    )
    state.apply_persistent_runtime_state(loaded)
    assert state.basis_regime.last_regime_sign == 1.0
    assert state.basis_regime.last_ic == 0.27
    assert state.basis_regime.pair_count == 180


def test_basis_regime_persistence_round_trip() -> None:
    """Build → load → build cycle preserves all three fields."""
    s = _settings()
    state1 = BotState(s)
    state1.basis_regime.seed_from_persisted(
        last_regime_sign=-1.0,
        last_ic=-0.18,
        pair_count=240,
    )
    prs1 = state1.build_persistent_runtime_state()

    # Simulate restart: fresh state, apply the persisted bundle.
    state2 = BotState(s)
    state2.apply_persistent_runtime_state(prs1)

    # Re-build and assert the same fields come back.
    prs2 = state2.build_persistent_runtime_state()
    assert prs2.basis_regime_last_sign == prs1.basis_regime_last_sign
    assert prs2.basis_regime_last_ic == prs1.basis_regime_last_ic
    assert prs2.basis_regime_pair_count == prs1.basis_regime_pair_count


def test_apply_persistent_runtime_state_without_basis_regime_fields() -> None:
    """Backward compat: a persisted state from before N12 wiring
    has ``basis_regime_*`` fields as None. Loading must not blow up
    and must leave the classifier in its initial state."""
    s = _settings()
    state = BotState(s)
    initial_sign = state.basis_regime.last_regime_sign
    initial_ic = state.basis_regime.last_ic
    initial_pairs = state.basis_regime.pair_count

    today = state.operator_day_anchor_utc
    loaded = PersistentRuntimeState(
        day_anchor_utc=today,
        daily_realized_pnl=0.0,
        daily_trade_count=0,
        daily_traded_notional=0.0,
        last_fill_ts=None,
        recent_buy_fill_count=0,
        recent_sell_fill_count=0,
        rolling_toxicity_markout_bps=None,
        rolling_one_sided_fill_ratio=None,
        soft_flatten_active=False,
        soft_flatten_started_at_iso=None,
        # No basis_regime_* — pre-N12-wiring persisted state.
    )
    state.apply_persistent_runtime_state(loaded)
    # No exception; classifier still at initial state.
    assert state.basis_regime.last_regime_sign == initial_sign
    assert state.basis_regime.last_ic == initial_ic
    assert state.basis_regime.pair_count == initial_pairs
