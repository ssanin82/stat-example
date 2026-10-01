"""v1.5.232 — vol_bps publishes None during warm-up tests.

Pre-v1.5.232 the bot wrote `state.vol_bps = 0.0` during the
VolatilityEstimator's warm-up window (< 32 distinct mids).
Downstream (dashboard, snapshot pipeline) could not tell apart
"warming up" from "actually flat market", which is why the
dashboard's Vol sub-band showed a 10-min flat zero line at session
start and the `Vol · 0 – X bp/s` label was anchored to those
warm-up zeros.

This file verifies:

* The estimator's existing contract is unchanged
  (`sigma_and_bps()` still returns `(None, 0.0)` during warm-up
  and `(sigma, sigma*10000)` post-warm-up).
* `state.vol_bps` is now typed `Optional[float]` and initialised
  to `None`.
* All downstream consumers handle `None` without crashing.
"""

from __future__ import annotations

import pytest


def test_estimator_unchanged_during_warmup():
    """v1.5.232 doesn't touch the estimator — sigma_and_bps()
    still returns (None, 0.0) during warm-up and (sigma, vol_bps)
    post-warm-up. The fix is at the bot.py call-site, not in the
    estimator."""
    from app.volatility import VolatilityEstimator
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
        VOL_WINDOW_SAMPLES=4,  # minimum window for fast test
    )
    est = VolatilityEstimator(s)
    # No mids pushed: returns (None, 0.0)
    sigma, vol_bps = est.sigma_and_bps()
    assert sigma is None
    assert vol_bps == 0.0
    assert not est.warmed_up

    # Push 3 distinct mids: still warming up (need 4).
    for m in (100.0, 100.1, 100.05):
        est.push_mid(m)
    sigma, vol_bps = est.sigma_and_bps()
    assert sigma is None
    assert vol_bps == 0.0
    assert not est.warmed_up

    # Push the 4th distinct mid: warmed up, returns real values.
    est.push_mid(100.2)
    sigma, vol_bps = est.sigma_and_bps()
    assert sigma is not None
    assert vol_bps > 0.0
    assert est.warmed_up


def test_state_vol_bps_initialised_to_none():
    """v1.5.232 — initial state.vol_bps is None, not 0.0. Snapshot
    pipeline and dashboard rely on this to distinguish warm-up from
    flat market."""
    from app.state import BotState
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
    )
    state = BotState(s)
    assert state.vol_bps is None, (
        f"state.vol_bps should be None at init (v1.5.232), got {state.vol_bps!r}"
    )


def test_state_snapshot_dict_with_none_vol_bps():
    """snapshot_dict must serialise None vol_bps as JSON null,
    not crash."""
    import json
    from app.state import BotState
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
    )
    state = BotState(s)
    assert state.vol_bps is None
    snap = state.snapshot_dict()
    assert snap["short_vol_bps"] is None
    # Verify the whole dict is JSON-serialisable (no surprise non-None
    # field that doesn't tolerate the new type contract).
    json.dumps(snap, default=str)  # default=str for any datetime, etc.


def test_state_vol_bps_assignment_accepts_both_none_and_float():
    """v1.5.232 — the field accepts both None (warm-up) and float
    (post-warm-up). No runtime type-check should reject either."""
    from app.state import BotState
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
    )
    state = BotState(s)
    # Both transitions must work without raising.
    state.vol_bps = 5.5
    assert state.vol_bps == 5.5
    state.vol_bps = None
    assert state.vol_bps is None
    state.vol_bps = 0.0  # genuine flat market post-warm-up
    assert state.vol_bps == 0.0


def test_toxicity_set_baseline_safe_with_zero_fallback():
    """v1.5.232 — bot.py converts None → 0.0 before calling
    set_baseline_vol. The toxicity engine's existing guard
    (vol_bps > 0) means 0.0 doesn't seed the baseline, which is
    the desired warm-up behaviour. This test asserts the engine's
    contract is preserved (not strictly a v1.5.232 change, but
    documents the invariant the bot.py edit relies on)."""
    from app.toxicity import ToxicityEngine
    from app.config import Settings
    s = Settings(
        VENUE="binance", SYMBOL="BTCUSDT",
        QUOTE_NOTIONAL_USD=100.0, MIN_QUOTE_NOTIONAL_USD=10.0,
        MAX_ABS_POSITION=5.0,
    )
    eng = ToxicityEngine(s)
    eng.set_baseline_vol(0.0)
    assert eng._baseline_vol_bps is None, (
        "0.0 must not seed the baseline (warm-up behaviour)"
    )
    eng.set_baseline_vol(5.5)
    assert eng._baseline_vol_bps == 5.5, (
        "first positive value seeds the baseline"
    )
    eng.set_baseline_vol(7.0)
    assert eng._baseline_vol_bps == 5.5, (
        "subsequent positive values don't re-seed"
    )
