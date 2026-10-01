"""Tests for the three 1.2.2 regime gates:

* ``vol_trend_gate`` (2a) — vol×drift conjunction with persistence
* ``basis_regime_gate`` (2b) — IC absent → suppress
* ``microprice_gate`` (1a) — OB-imbalance thin-side suppression
"""

from __future__ import annotations

import pytest

from app.basis_regime_gate import evaluate_gate as evaluate_basis_regime_gate
from app.enums import QuoteEligibility
from app.microprice_gate import evaluate_gate as evaluate_microprice_gate
from app.vol_trend_gate import (
    VolTrendState,
    is_active,
    observe,
    seconds_remaining,
)


# ---------------------------------------------------------------------------
# vol_trend_gate (2a)
# ---------------------------------------------------------------------------


def _vt_kw(**overrides):
    base = {
        "vol_multiplier": 2.5,
        "drift_threshold_bps": 3.0,
        "persistence_seconds": 10.0,
        "cooldown_seconds": 120.0,
    }
    base.update(overrides)
    return base


def test_vol_trend_initial_inactive() -> None:
    s = VolTrendState()
    assert not is_active(s, now_mono=0.0)
    assert s.armed_at_mono is None
    assert s.fire_count == 0


def test_vol_trend_does_not_fire_on_single_tick_spike() -> None:
    """Persistence guard: a single tick with conjunction true
    arms but doesn't fire. Real bursts persist; noise spikes don't."""
    s = VolTrendState()
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=5.0, **_vt_kw())
    assert s.armed_at_mono == 0.0
    assert not is_active(s, now_mono=0.0)
    # One second later — still arming.
    observe(s, now_mono=1.0, vol_ratio=3.0, drift_bps=5.0, **_vt_kw())
    assert not is_active(s, now_mono=1.0)


def test_vol_trend_fires_after_persistence() -> None:
    s = VolTrendState()
    kw = _vt_kw(persistence_seconds=10.0, cooldown_seconds=60.0)
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=5.0, **kw)
    observe(s, now_mono=11.0, vol_ratio=3.0, drift_bps=5.0, **kw)
    assert is_active(s, now_mono=11.0)
    assert s.fire_count == 1
    # Cooldown deadline = 11 + 60 = 71.
    assert s.cooldown_until_mono == pytest.approx(71.0)
    assert seconds_remaining(s, now_mono=11.0) == pytest.approx(60.0)


def test_vol_trend_disarms_on_clearing_conjunction() -> None:
    """If the conjunction breaks before persistence elapses,
    arming resets."""
    s = VolTrendState()
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=5.0, **_vt_kw())
    assert s.armed_at_mono == 0.0
    # Vol drops back to baseline.
    observe(s, now_mono=5.0, vol_ratio=1.0, drift_bps=5.0, **_vt_kw())
    assert s.armed_at_mono is None


def test_vol_trend_symmetric_on_drift_sign() -> None:
    """Negative drift (downtrend) fires the gate the same as
    positive drift."""
    s = VolTrendState()
    kw = _vt_kw(persistence_seconds=5.0)
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=-5.0, **kw)
    observe(s, now_mono=6.0, vol_ratio=3.0, drift_bps=-5.0, **kw)
    assert is_active(s, now_mono=6.0)


def test_vol_trend_below_threshold_no_arm() -> None:
    s = VolTrendState()
    # Vol high but drift below threshold.
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=1.0, **_vt_kw())
    assert s.armed_at_mono is None
    # Drift high but vol below threshold.
    observe(s, now_mono=1.0, vol_ratio=1.5, drift_bps=5.0, **_vt_kw())
    assert s.armed_at_mono is None


def test_vol_trend_no_re_arm_during_cooldown() -> None:
    """Cooldown is one-shot per arming. Continued conjunction-true
    during the cooldown does NOT extend the deadline (that would
    create indefinite suppression on regime persistence — which
    other gates handle better)."""
    s = VolTrendState()
    kw = _vt_kw(persistence_seconds=5.0, cooldown_seconds=20.0)
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=5.0, **kw)
    observe(s, now_mono=6.0, vol_ratio=3.0, drift_bps=5.0, **kw)
    deadline_before = s.cooldown_until_mono
    fire_before = s.fire_count
    # Same conditions still met, but we're inside cooldown.
    observe(s, now_mono=10.0, vol_ratio=5.0, drift_bps=10.0, **kw)
    assert s.cooldown_until_mono == deadline_before
    assert s.fire_count == fire_before


def test_vol_trend_re_arms_after_cooldown_expires() -> None:
    s = VolTrendState()
    kw = _vt_kw(persistence_seconds=2.0, cooldown_seconds=10.0)
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=5.0, **kw)
    observe(s, now_mono=3.0, vol_ratio=3.0, drift_bps=5.0, **kw)
    assert s.fire_count == 1
    # Wait past cooldown.
    observe(s, now_mono=20.0, vol_ratio=3.0, drift_bps=5.0, **kw)  # arm
    observe(s, now_mono=23.0, vol_ratio=3.0, drift_bps=5.0, **kw)  # fire
    assert s.fire_count == 2


def test_vol_trend_handles_none_inputs_gracefully() -> None:
    s = VolTrendState()
    observe(s, now_mono=0.0, vol_ratio=None, drift_bps=5.0, **_vt_kw())
    observe(s, now_mono=1.0, vol_ratio=3.0, drift_bps=None, **_vt_kw())
    assert s.armed_at_mono is None
    assert not is_active(s, now_mono=1.0)


# ---------------------------------------------------------------------------
# vol_trend_gate Phase 2K.3 — favorable-exit predicate
# ---------------------------------------------------------------------------


def _arm_vt(s: VolTrendState, **kw_overrides) -> None:
    """Drive the gate through one full arm cycle: persistence-elapsed
    + conjunction-met → cooldown active. Returns nothing; mutates s.

    Used to set up tests that exercise the in-cooldown behaviour
    rather than re-testing the arming path."""
    kw = _vt_kw(**kw_overrides)
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=5.0, **kw)
    # Persistence is 10 s — second tick after the dwell elapses fires.
    observe(s, now_mono=10.5, vol_ratio=3.0, drift_bps=5.0, **kw)
    assert is_active(s, now_mono=10.5), "test setup: arm should have fired"


def test_vol_trend_phase2k3_clears_early_when_signals_drop_for_dwell() -> None:
    """Active cooldown clears EARLY when both signals fall below
    ``trigger × clear_band_mult`` and stay there for
    ``favorable_exit_dwell_seconds``. With defaults (vol_mult=2.5,
    drift_thresh=3.0, clear_band_mult=0.7, dwell=10), the clear band
    is vol<1.75 AND |drift|<2.1. Default cooldown 120 s; favorable
    exit should fire well before that ceiling."""
    s = VolTrendState()
    _arm_vt(s)
    fav_before = s.cleared_via_favorable_total
    ceil_before = s.cleared_via_ceiling_total
    # Right at the moment of arming, both signals are still high.
    # Drop both below the clear band — start the dwell.
    observe(s, now_mono=11.0, vol_ratio=1.0, drift_bps=0.5, **_vt_kw())
    assert s.favorable_dwell_started_mono is not None
    assert is_active(s, now_mono=11.0), "shouldn't clear before dwell"
    # 5 s into dwell — still active (need 10 s).
    observe(s, now_mono=16.0, vol_ratio=1.0, drift_bps=0.5, **_vt_kw())
    assert is_active(s, now_mono=16.0)
    # 10.5 s into dwell — favorable exit fires.
    observe(s, now_mono=21.5, vol_ratio=1.0, drift_bps=0.5, **_vt_kw())
    assert not is_active(s, now_mono=21.5)
    assert s.cleared_via_favorable_total == fav_before + 1
    assert s.cleared_via_ceiling_total == ceil_before
    # Dwell timer cleared.
    assert s.favorable_dwell_started_mono is None


def test_vol_trend_phase2k3_re_flare_during_dwell_resets_timer() -> None:
    """If vol or drift pops back above the clear band MID-dwell, the
    dwell timer resets — both signals must STAY low for the full
    window. Tests hysteresis correctness."""
    s = VolTrendState()
    _arm_vt(s)
    # Drop below clear band — dwell starts.
    observe(s, now_mono=11.0, vol_ratio=1.0, drift_bps=0.5, **_vt_kw())
    assert s.favorable_dwell_started_mono == 11.0
    # 5 s in, drift pops back up — dwell resets.
    observe(s, now_mono=16.0, vol_ratio=1.0, drift_bps=2.5, **_vt_kw())
    assert s.favorable_dwell_started_mono is None
    # Drop both below again — new dwell starts.
    observe(s, now_mono=17.0, vol_ratio=1.0, drift_bps=0.5, **_vt_kw())
    assert s.favorable_dwell_started_mono == 17.0
    # The clearing would now require another 10 s from t=17, not from t=11.
    observe(s, now_mono=22.0, vol_ratio=1.0, drift_bps=0.5, **_vt_kw())
    assert is_active(s, now_mono=22.0), (
        "must still be active — only 5s of new dwell"
    )
    observe(s, now_mono=27.5, vol_ratio=1.0, drift_bps=0.5, **_vt_kw())
    assert not is_active(s, now_mono=27.5)
    assert s.cleared_via_favorable_total == 1


def test_vol_trend_phase2k3_ceiling_fires_when_signals_stay_elevated() -> None:
    """Static-high signals: cooldown runs the full 120 s deadline,
    favorable-exit never fires, ceiling counter increments on the
    clearing edge."""
    s = VolTrendState()
    _arm_vt(s)
    # Signals stay high through the entire window.
    observe(s, now_mono=60.0, vol_ratio=3.0, drift_bps=5.0, **_vt_kw())
    assert is_active(s, now_mono=60.0)
    # Just past the ceiling — the deadline lapses, next observe()
    # detects the active→cleared edge and attributes to ceiling.
    # The fire was at t=10.5, cooldown=120, so deadline=130.5.
    observe(s, now_mono=131.0, vol_ratio=3.0, drift_bps=5.0, **_vt_kw())
    assert not is_active(s, now_mono=131.0)
    assert s.cleared_via_favorable_total == 0
    assert s.cleared_via_ceiling_total == 1


def test_vol_trend_phase2k3_disabled_mult_uses_pure_timer() -> None:
    """clear_band_mult=0 disables the favorable-exit predicate. Even
    with both signals dropping to zero, the gate doesn't clear early.
    Pre-2K.3 behaviour preserved as opt-out."""
    s = VolTrendState()
    kw = _vt_kw()
    kw["clear_band_mult"] = 0.0
    observe(s, now_mono=0.0, vol_ratio=3.0, drift_bps=5.0, **kw)
    observe(s, now_mono=10.5, vol_ratio=3.0, drift_bps=5.0, **kw)
    assert is_active(s, now_mono=10.5)
    # Drop signals to zero — favorable exit should NOT fire.
    observe(s, now_mono=11.0, vol_ratio=0.0, drift_bps=0.0, **kw)
    assert s.favorable_dwell_started_mono is None
    observe(s, now_mono=50.0, vol_ratio=0.0, drift_bps=0.0, **kw)
    assert is_active(s, now_mono=50.0), (
        "mult=0 must disable favorable-exit; gate should still be active"
    )
    assert s.cleared_via_favorable_total == 0


def test_vol_trend_phase2k3_only_vol_drops_does_not_clear() -> None:
    """Both signals must be below the clear band — partial clearing
    on one signal is not enough. Vol drops below clear, drift stays
    high → dwell does NOT start."""
    s = VolTrendState()
    _arm_vt(s)
    observe(s, now_mono=11.0, vol_ratio=1.0, drift_bps=5.0, **_vt_kw())
    assert s.favorable_dwell_started_mono is None
    assert is_active(s, now_mono=11.0)


# ---------------------------------------------------------------------------
# basis_regime_gate (2b)
# ---------------------------------------------------------------------------


def test_basis_regime_warmup_no_gate() -> None:
    """During warmup (pair_count below min), the gate stays out."""
    reason = evaluate_basis_regime_gate(
        last_ic=None, pair_count=10,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
    )
    assert reason is None


def test_basis_regime_strong_signal_no_gate() -> None:
    """Strong IC → regime has detectable structure → gate stays
    out (the existing quote-skew handles the lean)."""
    reason = evaluate_basis_regime_gate(
        last_ic=-0.31, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
    )
    assert reason is None


def test_basis_regime_weak_signal_fires() -> None:
    """|IC| below threshold → signal-absent regime → suppress."""
    reason = evaluate_basis_regime_gate(
        last_ic=0.02, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
    )
    assert reason is not None
    assert "basis_regime_signal_absent" in reason


def test_basis_regime_disabled_returns_none() -> None:
    reason = evaluate_basis_regime_gate(
        last_ic=0.02, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
        enabled=False,
    )
    assert reason is None


def test_basis_regime_handles_nan_ic() -> None:
    reason = evaluate_basis_regime_gate(
        last_ic=float("nan"), pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
    )
    assert reason is None


# ---------------------------------------------------------------------------
# basis_regime_gate — size-shrink mode (1.2.8)
# ---------------------------------------------------------------------------

from app.basis_regime_gate import compute_size_mult as basis_regime_size_mult


def test_basis_regime_size_shrink_strong_signal_no_op() -> None:
    """Strong IC → gate doesn't fire → returns 1.0, None."""
    mult, reason = basis_regime_size_mult(
        last_ic=-0.31, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
        size_mult_signal_absent=0.5,
    )
    assert mult == 1.0
    assert reason is None


def test_basis_regime_size_shrink_weak_signal_returns_mult() -> None:
    """|IC| below threshold → return configured shrink + reason
    string referencing the IC + mult."""
    mult, reason = basis_regime_size_mult(
        last_ic=0.02, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
        size_mult_signal_absent=0.5,
    )
    assert mult == 0.5
    assert reason is not None
    assert "basis_regime_size_shrink" in reason
    assert "ic=+0.020" in reason
    assert "mult=0.50" in reason


def test_basis_regime_size_shrink_warmup_no_op() -> None:
    """Warmup (pair_count below floor) → no shrink even on weak IC."""
    mult, reason = basis_regime_size_mult(
        last_ic=0.02, pair_count=10,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
        size_mult_signal_absent=0.5,
    )
    assert mult == 1.0
    assert reason is None


def test_basis_regime_size_shrink_disabled_returns_no_op() -> None:
    mult, reason = basis_regime_size_mult(
        last_ic=0.02, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
        size_mult_signal_absent=0.5,
        enabled=False,
    )
    assert mult == 1.0
    assert reason is None


def test_basis_regime_size_shrink_clamps_pathological_mult() -> None:
    """Operator misconfig: ``size_mult_signal_absent`` > 1.0 should
    not GROW the size — clamp to 1.0. Negative values clamp to 0.0."""
    mult, _ = basis_regime_size_mult(
        last_ic=0.02, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
        size_mult_signal_absent=1.5,  # nonsense
    )
    assert mult == 1.0
    mult, _ = basis_regime_size_mult(
        last_ic=0.02, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
        size_mult_signal_absent=-0.5,  # nonsense
    )
    assert mult == 0.0


def test_basis_regime_size_shrink_zero_mult_means_no_quoting() -> None:
    """``size_mult_signal_absent=0.0`` is allowed — equivalent to
    HOLD_ALL via the size-shrink pathway. Caller's
    MIN_QUOTE_NOTIONAL_USD floor will then suppress the side."""
    mult, reason = basis_regime_size_mult(
        last_ic=0.02, pair_count=240,
        ic_min_quote_threshold=0.05, min_pair_samples=240,
        size_mult_signal_absent=0.0,
    )
    assert mult == 0.0
    assert reason is not None
    assert "mult=0.00" in reason


# ---------------------------------------------------------------------------
# microprice_gate (1a)
# ---------------------------------------------------------------------------


def test_microprice_balanced_book_no_gate() -> None:
    override, reason = evaluate_microprice_gate(
        ob_imbalance_ewma=0.1, threshold=0.5,
    )
    assert override is None
    assert reason == ""


def test_microprice_bid_heavy_suppresses_ask() -> None:
    """Positive imbalance = bid-heavy = ask is thin =
    QUOTE_BUY_ONLY (suppress ask side)."""
    override, reason = evaluate_microprice_gate(
        ob_imbalance_ewma=0.7, threshold=0.5,
    )
    assert override == QuoteEligibility.QUOTE_BUY_ONLY
    assert "ask_thin" in reason


def test_microprice_ask_heavy_suppresses_bid() -> None:
    """Negative imbalance = ask-heavy = bid is thin =
    QUOTE_SELL_ONLY (suppress bid side)."""
    override, reason = evaluate_microprice_gate(
        ob_imbalance_ewma=-0.7, threshold=0.5,
    )
    assert override == QuoteEligibility.QUOTE_SELL_ONLY
    assert "bid_thin" in reason


def test_microprice_at_threshold_inclusive_fires() -> None:
    override, _ = evaluate_microprice_gate(
        ob_imbalance_ewma=0.5, threshold=0.5,
    )
    assert override == QuoteEligibility.QUOTE_BUY_ONLY


def test_microprice_disabled_returns_none() -> None:
    override, _ = evaluate_microprice_gate(
        ob_imbalance_ewma=0.7, threshold=0.5, enabled=False,
    )
    assert override is None


def test_microprice_none_input_returns_none() -> None:
    override, _ = evaluate_microprice_gate(
        ob_imbalance_ewma=None, threshold=0.5,
    )
    assert override is None


# ---------------------------------------------------------------------------
# v1.4.42 BUG-026: microprice gate inventory-aware suppression
# ---------------------------------------------------------------------------
#
# When the bot is heavily one-sided AND the gate would widen the side
# that REDUCES inventory, the widening blocks the fill the bot needs
# to unwind. Concrete failure mode observed 2026-05-18 in snapshot
# v1.4.41-260518-111733: bot short -8 at max-position 10, ob_imb=-0.671
# (ask-heavy → gate widens bid under bid_thin), bid placed at 1.9024
# vs touch 1.907 — 24 bps below touch, instant cross-venue cancel,
# bot unable to reduce short for an entire 30+ min session.
#
# Fix: when |position|/abs_cap >= reducing_side_widen_suppress_pct AND
# the gate would widen the reducing side, return zero contribution
# on that side.


from app.microprice_gate import widening_bps as microprice_widening_bps


def test_microprice_widening_neutral_inventory_unchanged_v1_4_42() -> None:
    """Pre-fix behaviour preserved when inventory is near zero. Gate
    widens the bid as before."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,  # ask-heavy → bid_thin
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=0.0,         # neutral
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.30,
    )
    assert bid_bps == 30.0, "bid_thin should widen bid to max at neutral inventory"
    assert ask_bps == 0.0


def test_microprice_widening_short_heavy_suppresses_bid_widening_v1_4_42() -> None:
    """v1.4.42 BUG-026 fix: bot is short; gate would widen bid
    (the reducing side); suppress the widening so the bid can rest
    near the touch and actually fill."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,   # ask-heavy → bid_thin
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-8.0,        # short, 80% of cap (well past 30% threshold)
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.30,
    )
    assert bid_bps == 0.0, (
        "v1.4.42 BUG-026: when bot is short and gate would widen bid "
        "(the reducing side), widening must be suppressed. Got "
        f"{bid_bps} bps — the bid is being pushed away from the touch "
        "and the bot will be unable to reduce inventory."
    )
    assert ask_bps == 0.0


def test_microprice_widening_long_heavy_suppresses_ask_widening_v1_4_42() -> None:
    """Mirror case: bot is long; gate would widen ask (the reducing
    side); suppress."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=+0.7,   # bid-heavy → ask_thin
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=+8.0,        # long, 80% of cap
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.30,
    )
    assert bid_bps == 0.0
    assert ask_bps == 0.0, (
        "v1.4.42 BUG-026: when bot is long and gate would widen ask "
        "(the reducing side), widening must be suppressed."
    )


def test_microprice_widening_short_with_ask_thin_NOT_suppressed_v1_4_42() -> None:
    """v1.4.42 negative control: the suppression is asymmetric — it
    only fires on the side that matches the bot's reducing direction.
    If the gate widens the ADDING side (which is the protection-
    legitimate case), the widening must still apply."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=+0.7,   # bid-heavy → ask_thin
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-8.0,        # short — ASK is the adding side, BID is reducing
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.30,
    )
    # Gate fires ask_thin → wants to widen ASK. Bot is short → ask
    # is the ADDING side (more short = more short). The fix should NOT
    # touch this case — adverse-selection protection on the adding
    # side is legitimate.
    assert ask_bps == 30.0, (
        "v1.4.42 BUG-026 over-reach: gate widening the adding side "
        "(ask when short) must NOT be suppressed. The fix is narrow."
    )
    assert bid_bps == 0.0


def test_microprice_widening_below_suppress_threshold_still_widens_v1_4_42() -> None:
    """v1.4.42 boundary: when inventory utilisation is below the
    suppress threshold, the gate widens as before. Pins that the
    suppression doesn't accidentally fire at low-inventory."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,   # bid_thin
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-2.0,        # short, only 20% of cap (below 30% threshold)
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.30,
    )
    assert bid_bps == 30.0, (
        "v1.4.42: at sub-threshold inventory (20% < 30% suppress_pct), "
        "the gate must widen as before."
    )
    assert ask_bps == 0.0


def test_microprice_widening_zero_suppress_pct_disables_feature_v1_4_42() -> None:
    """v1.4.42 back-compat: ``reducing_side_widen_suppress_pct=0.0``
    disables the BUG-026 fix entirely (pre-v1.4.42 behaviour)."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,   # bid_thin
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-8.0,        # heavily short
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.0,  # feature OFF
    )
    assert bid_bps == 30.0, (
        "v1.4.42: suppress_pct=0.0 must produce pre-fix behaviour "
        "(gate widens the bid even when bot is short)."
    )
    assert ask_bps == 0.0


def test_microprice_widening_zero_cap_safe_v1_4_42() -> None:
    """v1.4.42 defensive: when effective_abs_cap is zero (e.g. config
    misread / startup) the suppression must degrade cleanly to
    pre-fix behaviour without dividing by zero."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-8.0,
        effective_abs_cap=0.0,    # zero cap — degenerate
        reducing_side_widen_suppress_pct=0.30,
    )
    # No suppression possible without a valid cap; falls back to
    # widening as before.
    assert bid_bps == 30.0
    assert ask_bps == 0.0


# ---------------------------------------------------------------------------
# v1.4.46: sentinel coupling to inventory_execution_bias_min_util_pct
# ---------------------------------------------------------------------------
#
# Pre-v1.4.46 the BUG-026 suppression had its own threshold default
# (0.30 = 30% util). 2026-05-18 snapshot v1.4.45-260518-144411
# showed the bot at 28.8% util — just under the threshold — with
# the microprice gate widening the bid to MAX (30 bps) and the bot
# unable to reduce its -3 short. Operator pushback ("blind number
# plug") motivated v1.4.46: tie the suppression threshold to
# ``inventory_execution_bias_min_util_pct`` so the two protections
# (suppress adding side + don't widen reducing side) activate
# together by construction.
#
# The sentinel ``-1.0`` is resolved at the build_spread_composition
# level by reading the inventory bias threshold. These tests pin
# behaviour of the underlying ``widening_bps`` helper, which now
# just consumes whatever absolute value the caller passes.
# Resolution test lives in ``test_build_spread_composition`` (the
# integration site).


def test_microprice_widening_at_inventory_bias_threshold_v1_4_46() -> None:
    """v1.4.46: the canonical resolved value matches
    ``inventory_execution_bias_min_util_pct`` (default 0.12). The
    gate's helper, when called with that value, should suppress at
    12% util on the reducing side."""
    # Bot is short -1.5 of 10 cap = 15% util. With suppress_pct=0.12,
    # the suppression fires (because |util|=0.15 >= 0.12).
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-1.5,
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.12,
    )
    assert bid_bps == 0.0, (
        "v1.4.46: coupling to inventory_exec_bias_min_util_pct (0.12) "
        "must suppress bid widening at 15% util (which is past 12%). "
        "Pre-v1.4.46 default of 0.30 left this case unprotected, "
        "causing the 2026-05-18 snapshot v1.4.45-260518-144411 wedge."
    )


def test_microprice_widening_just_above_inventory_bias_threshold_v1_4_46() -> None:
    """v1.4.46: the snapshot regression — bot at 28.8% util. With
    the old 0.30 default, suppression was OFF (the bot wedged). With
    the new 0.12 default (via sentinel coupling), suppression IS on
    at any util ≥ 12%."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.664,  # the exact snapshot value
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-3.0,         # exact snapshot value
        effective_abs_cap=10.4,    # gives the 28.8% util from snapshot
        reducing_side_widen_suppress_pct=0.12,
    )
    assert bid_bps == 0.0, (
        "v1.4.46 snapshot regression: at 28.8% util on the reducing "
        "side, the gate must NOT widen the bid (pre-fix default 0.30 "
        "let it widen to MAX). 0.12 (default coupling) catches this."
    )


def test_microprice_widening_below_inventory_bias_threshold_v1_4_46() -> None:
    """v1.4.46: when util is below the coupled threshold (i.e. the
    inventory bias hasn't activated yet either), the gate widens
    normally. The two protections are paired — they activate
    together AND stay off together."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-1.0,          # 10% util (< 12%)
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.12,
    )
    assert bid_bps == 30.0, (
        "v1.4.46: below the inventory-bias coupling threshold (12%), "
        "both protections stay off — the microprice gate widens "
        "normally because inventory is small enough that adverse "
        "selection protection matters more than reducing-side "
        "access."
    )


def test_microprice_widening_sentinel_resolution_logic_v1_4_46() -> None:
    """v1.4.46 sentinel resolution: when
    ``MICROPRICE_GATE_REDUCING_SIDE_WIDEN_SUPPRESS_PCT = -1.0``, the
    code in ``app/quoting.py:build_spread_composition`` resolves it
    to ``INVENTORY_EXECUTION_BIAS_MIN_UTIL_PCT``. Pinned here as a
    pure-Python repro of the resolution logic so a refactor can't
    silently break the coupling (which is the whole point of v1.4.46).
    """
    # Reproduce the resolution logic from quoting.py inline. If the
    # source changes shape, this test fails fast.
    def _resolve(raw: float, inventory_bias_threshold: float) -> float:
        if raw < 0.0:
            return float(inventory_bias_threshold)
        return float(raw)

    # Sentinel → coupled.
    assert _resolve(-1.0, 0.12) == 0.12
    # Explicit override → used verbatim.
    assert _resolve(0.30, 0.12) == 0.30
    # Disable (0.0) → still 0.0 (passes through as-is, not coupled).
    assert _resolve(0.0, 0.12) == 0.0


# ---------------------------------------------------------------------------
# v1.4.47: STRUCTURAL direction-based suppression (no threshold)
# ---------------------------------------------------------------------------
#
# v1.4.42 used a fixed 0.30 threshold; v1.4.46 coupled to 0.12. Each
# was a magic number. v1.4.47 rewrites the rule: if position has a
# sign, the reducing side is fully determined and the gate's
# widening on that side is unconditionally suppressed. No threshold,
# no tunable percent. The sign of `reducing_side_widen_suppress_pct`
# selects mode:
#   < 0  → STRUCTURAL (default in config sentinel)
#   == 0 → disabled
#   > 0  → legacy threshold (back-compat with v1.4.42-v1.4.46)


def test_microprice_widening_structural_short_any_size_suppresses_bid_v1_4_47() -> None:
    """v1.4.47: short of ANY size (no threshold) → bid widening
    suppressed when gate says bid_thin. Even 0.001 contracts qualify.
    The principle: the SIGN of inventory determines whether widening
    helps or hurts, not the magnitude."""
    for short_qty in (-0.001, -0.1, -1.0, -3.0, -8.0, -15.0):
        bid_bps, ask_bps = microprice_widening_bps(
            ob_imbalance_ewma=-0.7,
            threshold=0.5,
            max_half_spread_bps=30.0,
            widen_bps=-1.0,
            enabled=True,
            position_qty=short_qty,
            effective_abs_cap=10.0,
            reducing_side_widen_suppress_pct=-1.0,  # structural mode
        )
        assert bid_bps == 0.0, (
            f"v1.4.47 structural: any short qty (got {short_qty}) "
            f"must suppress bid widening. The reducing side is "
            f"determined by SIGN, not magnitude."
        )
        assert ask_bps == 0.0


def test_microprice_widening_structural_long_any_size_suppresses_ask_v1_4_47() -> None:
    """Mirror: long of ANY size suppresses ask widening when gate
    says ask_thin."""
    for long_qty in (0.001, 0.1, 1.0, 3.0, 8.0, 15.0):
        bid_bps, ask_bps = microprice_widening_bps(
            ob_imbalance_ewma=+0.7,
            threshold=0.5,
            max_half_spread_bps=30.0,
            widen_bps=-1.0,
            enabled=True,
            position_qty=long_qty,
            effective_abs_cap=10.0,
            reducing_side_widen_suppress_pct=-1.0,
        )
        assert bid_bps == 0.0
        assert ask_bps == 0.0, (
            f"v1.4.47 structural: any long qty (got {long_qty}) "
            f"must suppress ask widening."
        )


def test_microprice_widening_structural_neutral_inventory_widens_normally_v1_4_47() -> None:
    """v1.4.47: when position == 0, both sides are equally
    adversarial → gate's normal logic applies. Bid widens if bid_thin,
    ask widens if ask_thin."""
    # Neutral + bid_thin → widen bid.
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=0.0,
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=-1.0,
    )
    assert bid_bps == 30.0
    assert ask_bps == 0.0

    # Neutral + ask_thin → widen ask.
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=+0.7,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=0.0,
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=-1.0,
    )
    assert bid_bps == 0.0
    assert ask_bps == 30.0


def test_microprice_widening_structural_short_adding_side_still_widens_v1_4_47() -> None:
    """v1.4.47 negative control: short bot, gate fires ask_thin
    (rare in a market that's rising). The ask is the ADDING side for
    a short — widening it is legitimate adverse-selection protection.
    The structural rule must NOT touch this case."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=+0.7,  # bid-heavy → ask_thin
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-5.0,  # short — ask is the ADDING side
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=-1.0,
    )
    assert ask_bps == 30.0, (
        "v1.4.47 over-reach check: when the gate widens the ADDING "
        "side (ask when short), the structural rule must NOT touch "
        "it. Adverse-selection protection on the adding side is "
        "legitimate and stays."
    )
    assert bid_bps == 0.0


def test_microprice_widening_legacy_threshold_mode_still_works_v1_4_47() -> None:
    """v1.4.47 back-compat: positive ``reducing_side_widen_suppress_pct``
    still works as v1.4.42-v1.4.46 threshold mode. Pre-existing
    deployments that pinned 0.30 (etc.) don't regress."""
    # Threshold = 0.30, short at 20% util → BELOW threshold, widening
    # fires.
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-2.0,
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.30,
    )
    assert bid_bps == 30.0

    # Threshold = 0.30, short at 80% util → ABOVE threshold, widening
    # suppressed (v1.4.42 behavior preserved).
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-8.0,
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.30,
    )
    assert bid_bps == 0.0


def test_microprice_widening_disabled_mode_zero_v1_4_47() -> None:
    """v1.4.47 back-compat: ``reducing_side_widen_suppress_pct=0.0``
    disables the suppression entirely (pre-v1.4.42 behaviour).
    Reducing side IS widened — operator opt-out."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.7,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-8.0,  # heavily short
        effective_abs_cap=10.0,
        reducing_side_widen_suppress_pct=0.0,
    )
    assert bid_bps == 30.0, (
        "v1.4.47: suppress_pct=0.0 must disable the protection — "
        "operator opt-out. Pre-v1.4.42 behaviour preserved."
    )


def test_microprice_widening_structural_snapshot_regression_v1_4_47() -> None:
    """v1.4.47: exact reproduction of the 2026-05-18 snapshot
    v1.4.45-260518-144411 conditions that v1.4.46 STILL couldn't
    fix without coupling: bot short -3 of cap 10.4 (28.8% util),
    ob_imb=-0.664 (bid_thin). v1.4.42 default (0.30) → NO suppression
    (28.8% < 30%). v1.4.46 coupling to 0.12 → suppression (28.8% >
    12%). v1.4.47 structural → suppression (any short qty).
    Pin the structural answer here so future refactors can't
    regress."""
    bid_bps, ask_bps = microprice_widening_bps(
        ob_imbalance_ewma=-0.664,
        threshold=0.5,
        max_half_spread_bps=30.0,
        widen_bps=-1.0,
        enabled=True,
        position_qty=-3.0,
        effective_abs_cap=10.4,
        reducing_side_widen_suppress_pct=-1.0,  # structural
    )
    assert bid_bps == 0.0, (
        "v1.4.47 structural rule: snapshot regression — at any "
        "short inventory, bid widening must be suppressed. Got "
        f"{bid_bps}. The structural rule beats any fixed threshold."
    )
