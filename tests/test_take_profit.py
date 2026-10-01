"""Unit tests for app/take_profit.py — pure helpers for TP mode.

v1.5.33. Tests cover:
  * upnl_bps sign + dust handling
  * should_arm_tp priority of suppressions (9 short-circuit branches)
  * should_disarm_tp four trigger conditions + hysteresis sanity
  * compute_tp_target_price BUY / SELL / 1-tick-spread collapse
"""

from __future__ import annotations

import pytest

from app.enums import Side
from app.take_profit import (
    compute_tp_target_price,
    compute_upnl_bps,
    should_arm_tp,
    should_disarm_tp,
)


# ---------------------------------------------------------------------------
# compute_upnl_bps
# ---------------------------------------------------------------------------


def test_upnl_bps_favorable_long():
    # $1 profit on $1000 notional = 10 bps favorable.
    assert compute_upnl_bps(
        unrealized_pnl_usd=1.0, position_notional_usd=1000.0
    ) == pytest.approx(10.0)


def test_upnl_bps_adverse_returns_negative():
    # $1 loss on $1000 notional = -10 bps (sign of TP is favorable +).
    assert compute_upnl_bps(
        unrealized_pnl_usd=-1.0, position_notional_usd=1000.0
    ) == pytest.approx(-10.0)


def test_upnl_bps_short_position_uses_abs_notional():
    # Notional is the absolute value; sign of position doesn't matter.
    assert compute_upnl_bps(
        unrealized_pnl_usd=2.0, position_notional_usd=-1000.0
    ) == pytest.approx(20.0)


def test_upnl_bps_none_inputs():
    assert compute_upnl_bps(
        unrealized_pnl_usd=None, position_notional_usd=1000.0
    ) is None
    assert compute_upnl_bps(
        unrealized_pnl_usd=1.0, position_notional_usd=None
    ) is None


def test_upnl_bps_dust_notional():
    # Below 1e-9 — return None (would divide by zero).
    assert compute_upnl_bps(
        unrealized_pnl_usd=1.0, position_notional_usd=1e-12
    ) is None


# ---------------------------------------------------------------------------
# should_arm_tp — suppression priority. Each test isolates ONE suppression
# while leaving the others non-blocking.
# ---------------------------------------------------------------------------


def _default_kwargs():
    return dict(
        enabled=True,
        trigger_bps=50.0,
        upnl_bps=60.0,
        position_qty=10.0,
        position_notional_usd=100.0,
        min_notional_usd=10.0,
        sf_active=False,
        tp_active=False,
        now_mono=1000.0,
        arm_cooldown_until_mono=0.0,
        bot_status_allows_quoting=True,
    )


def test_arm_clean_path():
    d = should_arm_tp(**_default_kwargs())
    assert d.should_arm is True
    assert d.suppressed_reason == ""


def test_arm_suppressed_disabled():
    d = should_arm_tp(**{**_default_kwargs(), "enabled": False})
    assert d.should_arm is False
    assert d.suppressed_reason == "disabled"


def test_arm_suppressed_already_active():
    d = should_arm_tp(**{**_default_kwargs(), "tp_active": True})
    assert d.should_arm is False
    assert d.suppressed_reason == "already_active"


def test_arm_suppressed_sf_active():
    d = should_arm_tp(**{**_default_kwargs(), "sf_active": True})
    assert d.should_arm is False
    assert d.suppressed_reason == "sf_active"


def test_arm_suppressed_bot_status_blocks():
    d = should_arm_tp(
        **{**_default_kwargs(), "bot_status_allows_quoting": False}
    )
    assert d.should_arm is False
    assert d.suppressed_reason == "bot_status_blocks"


def test_arm_suppressed_flat():
    d = should_arm_tp(**{**_default_kwargs(), "position_qty": 0.0})
    assert d.should_arm is False
    assert d.suppressed_reason == "flat"


def test_arm_suppressed_below_min_notional():
    d = should_arm_tp(
        **{
            **_default_kwargs(),
            "position_notional_usd": 5.0,
            "min_notional_usd": 10.0,
        }
    )
    assert d.should_arm is False
    assert d.suppressed_reason == "below_min_notional"


def test_arm_suppressed_arm_cooldown():
    d = should_arm_tp(
        **{
            **_default_kwargs(),
            "now_mono": 100.0,
            "arm_cooldown_until_mono": 200.0,
        }
    )
    assert d.should_arm is False
    assert d.suppressed_reason == "arm_cooldown"


def test_arm_suppressed_upnl_unavailable():
    d = should_arm_tp(**{**_default_kwargs(), "upnl_bps": None})
    assert d.should_arm is False
    assert d.suppressed_reason == "upnl_unavailable"


def test_arm_suppressed_below_trigger():
    d = should_arm_tp(**{**_default_kwargs(), "upnl_bps": 49.9})
    assert d.should_arm is False
    assert d.suppressed_reason == "below_trigger"


def test_arm_at_exactly_trigger():
    # Boundary: == trigger should arm (>= semantic).
    d = should_arm_tp(**{**_default_kwargs(), "upnl_bps": 50.0})
    assert d.should_arm is True


# ---------------------------------------------------------------------------
# should_disarm_tp — four trigger conditions
# ---------------------------------------------------------------------------


def _disarm_kwargs():
    return dict(
        upnl_bps=55.0,
        trigger_bps=50.0,
        disarm_margin_bps=5.0,
        armed_at_mono=1000.0,
        now_mono=1003.0,
        max_dwell_seconds=30.0,
        sf_armed_this_tick=False,
        position_flat=False,
    )


def test_disarm_stay_armed_above_exit_threshold():
    d = should_disarm_tp(**_disarm_kwargs())
    assert d.should_disarm is False


def test_disarm_sf_takeover_wins():
    d = should_disarm_tp(
        **{**_disarm_kwargs(), "sf_armed_this_tick": True}
    )
    assert d.should_disarm is True
    assert d.reason == "sf_takes_over"


def test_disarm_position_flat():
    d = should_disarm_tp(**{**_disarm_kwargs(), "position_flat": True})
    assert d.should_disarm is True
    assert d.reason == "position_flat"


def test_disarm_hysteresis_at_exit_threshold():
    # trigger=50, margin=5 → exit at 45. uPnL = 45 should disarm
    # (<= exit_threshold).
    d = should_disarm_tp(**{**_disarm_kwargs(), "upnl_bps": 45.0})
    assert d.should_disarm is True
    assert d.reason == "upnl_retraced"


def test_disarm_hysteresis_just_above_exit_threshold():
    # uPnL = 45.1 → stays armed.
    d = should_disarm_tp(**{**_disarm_kwargs(), "upnl_bps": 45.1})
    assert d.should_disarm is False


def test_disarm_max_dwell():
    # dwell = now - armed_at = 30s. max=30 → disarm.
    d = should_disarm_tp(
        **{**_disarm_kwargs(), "now_mono": 1030.0}
    )
    assert d.should_disarm is True
    assert d.reason == "max_dwell"


def test_disarm_upnl_none_stays_armed():
    # Book momentarily unavailable; don't disarm on missing data.
    d = should_disarm_tp(**{**_disarm_kwargs(), "upnl_bps": None})
    assert d.should_disarm is False


def test_disarm_priority_sf_over_flat():
    # Both conditions true; SF wins by check order.
    d = should_disarm_tp(
        **{
            **_disarm_kwargs(),
            "sf_armed_this_tick": True,
            "position_flat": True,
        }
    )
    assert d.reason == "sf_takes_over"


# ---------------------------------------------------------------------------
# compute_tp_target_price
# ---------------------------------------------------------------------------


def test_target_price_long_normal_spread():
    # Long position → SELL close. tick=0.001, bid=100.000, ask=100.005.
    # SELL at bid + 1 tick = 100.001 (inside the spread, post-only OK).
    side, px = compute_tp_target_price(
        pos_qty=10.0,
        best_bid=100.000,
        best_ask=100.005,
        tick_size=0.001,
    )
    assert side == Side.SELL
    assert px == pytest.approx(100.001)


def test_target_price_short_normal_spread():
    # Short position → BUY close. BUY at ask - 1 tick = 100.004.
    side, px = compute_tp_target_price(
        pos_qty=-10.0,
        best_bid=100.000,
        best_ask=100.005,
        tick_size=0.001,
    )
    assert side == Side.BUY
    assert px == pytest.approx(100.004)


def test_target_price_long_one_tick_spread_collapses_to_touch():
    # 1-tick spread: bid=100.000, ask=100.001. SELL at bid+1tick=100.001
    # which equals ask. The min() cap keeps it at the touch (post-only
    # rests at ask, no crossing).
    side, px = compute_tp_target_price(
        pos_qty=10.0,
        best_bid=100.000,
        best_ask=100.001,
        tick_size=0.001,
    )
    assert side == Side.SELL
    assert px == pytest.approx(100.001)


def test_target_price_short_one_tick_spread_collapses_to_touch():
    side, px = compute_tp_target_price(
        pos_qty=-10.0,
        best_bid=100.000,
        best_ask=100.001,
        tick_size=0.001,
    )
    assert side == Side.BUY
    assert px == pytest.approx(100.000)


def test_target_price_zero_tick_falls_back_to_far_touch():
    # Degenerate spec — should fall back to far touch (no tick offset).
    side, px = compute_tp_target_price(
        pos_qty=10.0,
        best_bid=100.000,
        best_ask=100.010,
        tick_size=0.0,
    )
    assert side == Side.SELL
    assert px == pytest.approx(100.010)
