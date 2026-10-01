"""v1.4.107 Phase 1B — shock_gate unit tests.

Pin the binary acute-spike defence's contract:

1. Disabled → no-op.
2. Util below threshold → idle (gate dormant for low-inventory bot).
3. Util high, drift aligned with position → idle.
4. Util high, drift anti-aligned but below threshold → idle.
5. LONG + 10s shock down → lock SELL_ONLY, fire_count bumps.
6. SHORT + 30s shock up → lock BUY_ONLY, fire_count bumps.
7. Lock persists across subsequent ticks while util / drift stay
   elevated.
8. Soft clear: util drops below clear_util AND drift normalises →
   lock clears, override goes None.
9. Soft clear requires BOTH conditions — drift clears but util still
   high → still locked.
10. Hard clear: max-cooldown timestamp reached → lock auto-clears
    even if soft conditions never trigger (safety net).
11. Direction-freeze: position flips during lock → original side
    stays suppressed (no fast-flip).
12. Snapshot dict shape: rendered fields present whether dormant or
    locked.
13. Replay acceptance: 260520 06:50:05 (LONG, sudden -100 bp drift)
    → gate fires SELL_ONLY.

The cooldown semantics specifically protect against the 260520
06:50 fast-flip pattern: the bot unwound LONG and immediately
opened SHORT into the bounce. Direction-freeze + sticky cooldown
prevents that.
"""

from __future__ import annotations

import math

from app.enums import QuoteEligibility
from app.shock_gate import ShockGateState, observe, seconds_in_lock, snapshot_dict


# ---------------------------------------------------------------------------
# Disabled / dormant cases
# ---------------------------------------------------------------------------


def test_disabled_returns_no_op() -> None:
    state = ShockGateState()
    override, reason = observe(
        state,
        now_mono=1000.0,
        position_qty=10.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-50.0,
        drift_bps_30s=-100.0,
        enabled=False,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is None
    assert reason == "shock_gate_disabled"
    assert state.locked is False
    assert state.fire_count == 0


def test_util_below_threshold_idle() -> None:
    """Bot at 50 % util — gate idle even if drift is huge."""
    state = ShockGateState()
    override, reason = observe(
        state,
        now_mono=1000.0,
        position_qty=5.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-50.0,
        drift_bps_30s=-100.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is None
    assert reason.startswith("shock_gate_idle:util_below")
    assert state.fire_count == 0


def test_aligned_drift_idle() -> None:
    """LONG + drift UP — bot on the right side. Gate idle."""
    state = ShockGateState()
    override, reason = observe(
        state,
        now_mono=1000.0,
        position_qty=9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=+50.0,
        drift_bps_30s=+100.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is None
    assert reason.startswith("shock_gate_idle:no_shock")
    assert state.fire_count == 0


def test_drift_below_threshold_idle() -> None:
    """Util high, drift below shock thresholds on both windows."""
    state = ShockGateState()
    override, reason = observe(
        state,
        now_mono=1000.0,
        position_qty=9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-15.0,  # below 20.0 → 10s no-shock
        drift_bps_30s=-30.0,  # below 50.0 → 30s no-shock
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is None
    assert reason.startswith("shock_gate_idle:no_shock")
    assert state.fire_count == 0


# ---------------------------------------------------------------------------
# Fire conditions
# ---------------------------------------------------------------------------


def test_long_anti_aligned_10s_shock_fires_sell_only() -> None:
    state = ShockGateState()
    override, reason = observe(
        state,
        now_mono=1000.0,
        position_qty=9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-25.0,
        drift_bps_30s=-10.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert "shock_gate_fire" in reason
    assert "down" in reason
    assert "10s" in reason
    assert state.locked is True
    assert state.locked_side is QuoteEligibility.QUOTE_SELL_ONLY
    assert state.last_trigger_drift_bps == -25.0
    assert state.last_trigger_util == 0.9
    assert state.fire_count == 1


def test_short_anti_aligned_30s_shock_fires_buy_only() -> None:
    state = ShockGateState()
    override, reason = observe(
        state,
        now_mono=1000.0,
        position_qty=-9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=+10.0,
        drift_bps_30s=+60.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is QuoteEligibility.QUOTE_BUY_ONLY
    assert "up" in reason
    assert "30s" in reason
    assert state.last_trigger_drift_bps == +60.0
    assert state.last_trigger_window_label == "30s"
    assert state.fire_count == 1


def test_both_windows_fire_picks_larger_drift() -> None:
    """When both windows trip, the larger-|drift| one is logged."""
    state = ShockGateState()
    override, _ = observe(
        state,
        now_mono=1000.0,
        position_qty=9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-25.0,
        drift_bps_30s=-80.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert state.last_trigger_drift_bps == -80.0
    assert state.last_trigger_window_label == "30s"


# ---------------------------------------------------------------------------
# Lock persistence + cooldown machinery
# ---------------------------------------------------------------------------


def _fire(state: ShockGateState, now_mono: float = 1000.0) -> None:
    """Helper — fire the gate at a known time."""
    observe(
        state,
        now_mono=now_mono,
        position_qty=9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-25.0,
        drift_bps_30s=-60.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )


def test_lock_persists_across_ticks() -> None:
    state = ShockGateState()
    _fire(state, now_mono=1000.0)
    assert state.locked is True

    override, reason = observe(
        state,
        now_mono=1010.0,
        position_qty=9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-25.0,
        drift_bps_30s=-60.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert "shock_gate_locked" in reason
    assert state.fire_count == 1  # no second fire — same lock


def test_soft_clear_drops_lock() -> None:
    """Util below 0.30 AND |drift_30s| < threshold/3 → unlock."""
    state = ShockGateState()
    _fire(state, now_mono=1000.0)

    override, reason = observe(
        state,
        now_mono=1200.0,
        position_qty=2.0,  # util 0.20 → below 0.30
        effective_abs_cap=10.0,
        drift_bps_10s=-5.0,
        drift_bps_30s=-10.0,  # |drift_30s| 10 < 50/3 ≈ 16.67
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is None
    assert "shock_gate_cleared" in reason
    assert state.locked is False
    assert state.locked_side is None


def test_soft_clear_requires_both_conditions() -> None:
    """Drift clears but util still high → still locked."""
    state = ShockGateState()
    _fire(state, now_mono=1000.0)

    override, _ = observe(
        state,
        now_mono=1100.0,
        position_qty=8.5,  # util 0.85 → above 0.30
        effective_abs_cap=10.0,
        drift_bps_10s=-5.0,
        drift_bps_30s=-10.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert state.locked is True


def test_hard_clear_at_max_cooldown_ceiling() -> None:
    """Max-cooldown timestamp reached → lock auto-clears even if soft
    conditions still imply lock."""
    state = ShockGateState()
    _fire(state, now_mono=1000.0)
    # Max cooldown is 300 s. At t=1301 we're past the ceiling.

    override, reason = observe(
        state,
        now_mono=1301.0,
        position_qty=9.0,  # still maxed long
        effective_abs_cap=10.0,
        drift_bps_10s=-25.0,
        drift_bps_30s=-60.0,  # still in shock magnitudes
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is None
    assert "shock_gate_cleared:max_cooldown" in reason
    assert state.locked is False


def test_direction_freeze_position_flips_during_lock() -> None:
    """Position flips LONG → SHORT during cooldown. The SELL_ONLY
    lock from the original fire must stay in place — the gate must
    NOT re-fire as BUY_ONLY against the new SHORT inventory while
    the original lock is in force."""
    state = ShockGateState()
    _fire(state, now_mono=1000.0)
    assert state.locked_side is QuoteEligibility.QUOTE_SELL_ONLY

    # Position flips to SHORT, util high. Drift bounces back UP.
    override, reason = observe(
        state,
        now_mono=1050.0,
        position_qty=-8.5,  # flipped to SHORT
        effective_abs_cap=10.0,
        drift_bps_10s=+30.0,
        drift_bps_30s=+60.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    # The original SELL_ONLY lock holds — we don't re-fire as
    # BUY_ONLY against the new (SHORT + UP-drift) signature.
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert state.locked is True
    assert state.locked_side is QuoteEligibility.QUOTE_SELL_ONLY
    assert state.fire_count == 1


# ---------------------------------------------------------------------------
# Snapshot dict shape
# ---------------------------------------------------------------------------


def test_snapshot_dict_dormant() -> None:
    state = ShockGateState()
    snap = snapshot_dict(state, now_mono=0.0)
    assert snap["active"] is False
    assert snap["locked_side"] is None
    assert snap["seconds_in_lock"] is None
    assert snap["fire_count"] == 0
    # Diagnostic fields default-render (zero / None) even when dormant.
    assert "last_trigger_drift_bps" in snap
    assert "last_trigger_util" in snap


def test_snapshot_dict_locked() -> None:
    state = ShockGateState()
    _fire(state, now_mono=1000.0)
    snap = snapshot_dict(state, now_mono=1015.5)
    assert snap["active"] is True
    assert snap["locked_side"] == "QUOTE_SELL_ONLY"
    assert snap["seconds_in_lock"] == 15.5
    assert snap["last_trigger_drift_bps"] == -60.0
    assert snap["last_trigger_window"] == "30s"
    assert snap["fire_count"] == 1


def test_seconds_in_lock_helper() -> None:
    state = ShockGateState()
    assert seconds_in_lock(state, now_mono=1000.0) is None
    _fire(state, now_mono=1000.0)
    v = seconds_in_lock(state, now_mono=1042.0)
    assert v is not None
    assert math.isclose(v, 42.0)


# ---------------------------------------------------------------------------
# Replay acceptance
# ---------------------------------------------------------------------------


def test_replay_06_50_05_scenario_fires() -> None:
    """At 06:50:05 in the 260520-074215 snapshot the mid snapped
    -100 bp in ~30 s while bot was maxed LONG (position +9 / cap
    +10 → util 0.9). The shock_gate MUST fire SELL_ONLY before the
    bot tries to short into the bounce."""
    state = ShockGateState()
    override, reason = observe(
        state,
        now_mono=1000.0,
        position_qty=+9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-40.0,
        drift_bps_30s=-90.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    assert override is QuoteEligibility.QUOTE_SELL_ONLY
    assert state.locked is True
    assert "shock_gate_fire" in reason
