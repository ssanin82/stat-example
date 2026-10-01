"""v1.5.155 — position-aware favorable-exit for the ``mae_gate``.

Per CLAUDE.md Rule 0c (operator instruction 2026-05-26): every
timer-based gate must have a signal-driven conditional exit that
considers current bot position. The pre-existing Phase 2K.7 markout-
based favorable exit only fires when recent fill bleeding stops —
but during a cooldown there are no new fills (gate paused both
sides), so the markout predicate never triggers. v1.5.154-260526-074029
snapshot showed 78 cooldowns / 0 favorable / 78 ceiling clearance
ratio — the markout exit is effectively dead in sustained adverse
regimes.

The v1.5.155 position-aware exit fires when:
* ``|position_qty| >= inventory_threshold`` (meaningful inventory)
* ``sign(position_qty) * drift_bps >= drift_threshold_bps`` (drift
  in same direction as position = inventory gaining = good time to
  unwind)

These tests exercise the new ``evaluate_position_favorable_exit``
entry point in isolation. Integration with the bot tick loop is
covered indirectly by the existing ``test_mae_gate.py``.

Per CLAUDE.md: only this test file is run from the assistant; full-
suite verification is the CI daemon's job.
"""

from __future__ import annotations

from app.mae_gate import (
    MaeGateState,
    evaluate_position_favorable_exit,
    observe,
)


def _arm_cooldown(state: MaeGateState, *, now_mono: float = 0.0) -> None:
    """Drive the gate into cooldown via 5 adverse fills."""
    for _ in range(5):
        observe(
            state,
            now_mono=now_mono,
            mae_30s_bps=-10.0,
            fill_window=5,
            hard_threshold_bps=4.0,
            cooldown_seconds=180.0,
        )
    assert state.cooldown_until_mono > 0.0  # sanity


# ---------------------------------------------------------------------------
# Predicate firing conditions
# ---------------------------------------------------------------------------


def test_short_with_down_drift_clears_cooldown():
    """Bot is SHORT and drift is DOWN → inventory gaining → favorable
    moment to unwind → cooldown clears."""
    state = MaeGateState()
    _arm_cooldown(state)
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,  # still well within cooldown
        position_qty=-3.0,
        drift_bps=-10.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is True
    assert state.cooldown_until_mono == 0.0
    assert state.cleared_via_position_favorable_total == 1


def test_long_with_up_drift_clears_cooldown():
    """Bot is LONG and drift is UP → inventory gaining → favorable
    moment to unwind."""
    state = MaeGateState()
    _arm_cooldown(state)
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=+3.0,
        drift_bps=+8.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is True
    assert state.cooldown_until_mono == 0.0
    assert state.cleared_via_position_favorable_total == 1


# ---------------------------------------------------------------------------
# Predicate non-firing conditions
# ---------------------------------------------------------------------------


def test_short_with_up_drift_does_not_clear():
    """Bot is SHORT and drift is UP → inventory LOSING value (uptrend
    against SHORT position) → NOT favorable → cooldown stays."""
    state = MaeGateState()
    _arm_cooldown(state)
    before = state.cooldown_until_mono
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=-3.0,
        drift_bps=+10.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is False
    assert state.cooldown_until_mono == before
    assert state.cleared_via_position_favorable_total == 0


def test_long_with_down_drift_does_not_clear():
    """Bot is LONG and drift is DOWN → inventory LOSING value
    (downtrend against LONG position) → NOT favorable."""
    state = MaeGateState()
    _arm_cooldown(state)
    before = state.cooldown_until_mono
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=+3.0,
        drift_bps=-10.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is False
    assert state.cooldown_until_mono == before


def test_flat_position_does_not_clear():
    """No inventory → no position-aware exit (even with strong drift)."""
    state = MaeGateState()
    _arm_cooldown(state)
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=0.0,
        drift_bps=-15.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is False


def test_subthreshold_inventory_does_not_clear():
    """Inventory below threshold (e.g. 0.5 contract residual) — too
    small to warrant clearing the gate."""
    state = MaeGateState()
    _arm_cooldown(state)
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=-0.5,
        drift_bps=-10.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is False


def test_subthreshold_drift_does_not_clear():
    """Drift in the favorable direction but too small to count as
    favorable (e.g. ±2 bps of noise)."""
    state = MaeGateState()
    _arm_cooldown(state)
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=-3.0,
        drift_bps=-2.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is False


def test_no_drift_data_does_not_clear():
    """drift_bps=None (warmup) → no-op, no error."""
    state = MaeGateState()
    _arm_cooldown(state)
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=-3.0,
        drift_bps=None,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is False


# ---------------------------------------------------------------------------
# Idempotency / no-op when not in cooldown
# ---------------------------------------------------------------------------


def test_not_in_cooldown_is_noop():
    """When the gate is not currently in cooldown, the call is a
    no-op — no counter increment, no state mutation."""
    state = MaeGateState()
    # No arm; cooldown_until_mono stays at 0.0.
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=-3.0,
        drift_bps=-10.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is False
    assert state.cleared_via_position_favorable_total == 0


def test_called_after_cooldown_expired_is_noop():
    """Once the cooldown deadline has passed (gate already cleared
    via ceiling), the call doesn't increment the favorable counter."""
    state = MaeGateState()
    _arm_cooldown(state)
    # Jump past the cooldown deadline.
    later = state.cooldown_until_mono + 1.0
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=later,
        position_qty=-3.0,
        drift_bps=-10.0,
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is False


# ---------------------------------------------------------------------------
# v1.5.154-260526-074029 reproducer
# ---------------------------------------------------------------------------


def test_v1_5_154_overnight_snapshot_reproducer():
    """Snapshot moment: bot at -3 SHORT in a sustained downtrend
    (drift_10s = -10.5 bps), mae_gate fired 78 times overnight with
    0 favorable exits — every clear via timer ceiling. Under v1.5.155
    the position-aware predicate would have fired immediately:
    SHORT + down-drift = inventory gaining = clear the pause so the
    bot can sell into the trend (= further accumulate SHORT into the
    favorable move) AND buy back at the local oscillation lows
    (= unwind profitably).

    NB: in practice the cooldown clears mean the bot resumes BOTH
    sides — and the v1.5.155 trend skew (also shipped this release)
    will make the bot lean correctly into the down-trend rather than
    quoting symmetrically. The two changes compound."""
    state = MaeGateState()
    _arm_cooldown(state)
    cleared = evaluate_position_favorable_exit(
        state,
        now_mono=1.0,
        position_qty=-3.0,  # SHORT 3 contracts (mid-night inventory)
        drift_bps=-10.5,  # drift_10s_bps from the actual snapshot
        inventory_threshold=1.0,
        drift_threshold_bps=5.0,
    )
    assert cleared is True
    assert state.cleared_via_position_favorable_total == 1
