"""v1.4.7 gate-widening Phase 0 — BotState.record_gate_firing
state-machine tests.

The recorder is the foundation for the dark-time baseline: it tracks
rising/falling edges of each gate's active state and accumulates
total fire-seconds. Every gate module in `bot.py` calls it once per
quote-cycle tick, so correctness here is load-bearing for the
entire Phase 0 telemetry pipeline.

Properties verified:

* fire_count bumps once per rising edge, never twice for sustained
  firing.
* fire_seconds_total accumulates correctly across falling edges.
* In-flight active duration is included in the snapshot so the
  operator sees real-time accrual.
* Per-gate independence: recording one gate doesn't perturb another.
* Idle gate (never seen) is absent from the snapshot — not silently
  zero-filled.
"""

from __future__ import annotations

from tests.settings_helpers import UnitTestSettings
from app.state import BotState


def _state() -> BotState:
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "MAX_ABS_POSITION": 1.0,
            "MAX_POSITION_NOTIONAL_USD": 100.0,
        }
    )
    return BotState(s)


def test_initial_state_is_empty() -> None:
    bs = _state()
    snap = bs.gate_attribution_snapshot(now_mono=100.0)
    assert snap == {}


def test_single_rising_edge_bumps_fire_count() -> None:
    bs = _state()
    bs.record_gate_firing("vol_trend_gate", firing_now=True, now_mono=100.0)
    snap = bs.gate_attribution_snapshot(now_mono=100.0)
    assert snap["vol_trend_gate"]["fire_count"] == 1
    assert snap["vol_trend_gate"]["active_now"] is True


def test_sustained_firing_does_not_double_count() -> None:
    """Multiple consecutive ``firing_now=True`` calls represent the
    SAME fire episode. fire_count is a rising-edge count, not a tick
    count."""
    bs = _state()
    for t in (100.0, 100.5, 101.0, 101.5):
        bs.record_gate_firing("vol_trend_gate", firing_now=True, now_mono=t)
    snap = bs.gate_attribution_snapshot(now_mono=101.5)
    assert snap["vol_trend_gate"]["fire_count"] == 1


def test_falling_edge_accumulates_fire_seconds() -> None:
    bs = _state()
    bs.record_gate_firing("vol_trend_gate", firing_now=True, now_mono=100.0)
    bs.record_gate_firing("vol_trend_gate", firing_now=False, now_mono=103.5)
    snap = bs.gate_attribution_snapshot(now_mono=103.5)
    assert snap["vol_trend_gate"]["fire_count"] == 1
    assert snap["vol_trend_gate"]["fire_seconds_total"] == 3.5
    assert snap["vol_trend_gate"]["active_now"] is False


def test_inflight_duration_included_in_snapshot() -> None:
    """When the gate is currently firing, the snapshot must include
    the partial duration since the rising edge. Otherwise the
    operator dashboard would show stale numbers during active fire
    episodes (which is exactly when they look)."""
    bs = _state()
    bs.record_gate_firing("vol_trend_gate", firing_now=True, now_mono=100.0)
    snap = bs.gate_attribution_snapshot(now_mono=102.0)
    assert snap["vol_trend_gate"]["fire_seconds_total"] == 2.0


def test_multiple_fire_episodes_accumulate() -> None:
    bs = _state()
    bs.record_gate_firing("vol_trend_gate", firing_now=True, now_mono=100.0)
    bs.record_gate_firing("vol_trend_gate", firing_now=False, now_mono=101.0)
    bs.record_gate_firing("vol_trend_gate", firing_now=True, now_mono=110.0)
    bs.record_gate_firing("vol_trend_gate", firing_now=False, now_mono=112.5)
    snap = bs.gate_attribution_snapshot(now_mono=112.5)
    assert snap["vol_trend_gate"]["fire_count"] == 2
    assert snap["vol_trend_gate"]["fire_seconds_total"] == 3.5


def test_per_gate_independence() -> None:
    bs = _state()
    bs.record_gate_firing("vol_trend_gate", firing_now=True, now_mono=100.0)
    bs.record_gate_firing("momentum_gate", firing_now=False, now_mono=100.0)
    bs.record_gate_firing("microprice_gate", firing_now=True, now_mono=100.5)
    bs.record_gate_firing("vol_trend_gate", firing_now=False, now_mono=101.0)
    snap = bs.gate_attribution_snapshot(now_mono=101.0)
    assert snap["vol_trend_gate"]["fire_count"] == 1
    assert snap["vol_trend_gate"]["fire_seconds_total"] == 1.0
    assert snap["vol_trend_gate"]["active_now"] is False
    # microprice is still firing; in-flight duration counts.
    assert snap["microprice_gate"]["fire_count"] == 1
    assert snap["microprice_gate"]["active_now"] is True
    assert snap["microprice_gate"]["fire_seconds_total"] == 0.5
    # momentum was never firing — not a rising edge, no entry except
    # the seed (active_now=False, count=0).
    assert "momentum_gate" in snap  # seed exists from first record() call
    assert snap["momentum_gate"]["fire_count"] == 0
    assert snap["momentum_gate"]["active_now"] is False


def test_starts_inactive_then_falls_is_noop() -> None:
    """A falling edge from never-fired state is a no-op — no negative
    durations, no fire_count bump."""
    bs = _state()
    bs.record_gate_firing("vol_trend_gate", firing_now=False, now_mono=100.0)
    bs.record_gate_firing("vol_trend_gate", firing_now=False, now_mono=101.0)
    snap = bs.gate_attribution_snapshot(now_mono=101.0)
    assert snap["vol_trend_gate"]["fire_count"] == 0
    assert snap["vol_trend_gate"]["fire_seconds_total"] == 0.0
