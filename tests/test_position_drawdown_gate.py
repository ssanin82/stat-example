"""Position-aware drawdown gate tests.

The gate fires when the unrealized PnL on the current position has
been adverse (>= threshold bps relative to |position_notional|) for
>= duration seconds. Action is to enter SOFT_FLATTENING (post-only-
only exit) -- not KILL. Designed to bridge the gap between inventory-
skew (which prevents accumulation but not exit) and absolute drawdown
($10) which is too loose for small positions ($24 = ~40% adverse
move before triggering).
"""

from __future__ import annotations

from app.models import PositionSnapshot
from app.position_drawdown_gate import evaluate
from app.state import BotState

from tests.settings_helpers import UnitTestSettings


def _settings(**kw: object) -> UnitTestSettings:
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "0x" + "11" * 32,
        "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
        "POSITION_DRAWDOWN_GATE_ENABLED": True,
        "POSITION_DRAWDOWN_GATE_THRESHOLD_BPS": 50.0,
        "POSITION_DRAWDOWN_GATE_DURATION_SECONDS": 30.0,
    }
    for k, v in kw.items():
        data[k] = v
    return UnitTestSettings.model_validate(data)


def _set_position(
    state: BotState, qty: float, notional: float, unrealized: float
) -> None:
    """Patch the position fields used by the gate."""
    state.position = PositionSnapshot(
        symbol=state.symbol,
        position_qty=qty,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=notional,
        unrealized_pnl_usd=unrealized,
    )


def test_gate_does_not_trigger_when_flat() -> None:
    settings = _settings()
    state = BotState(settings)
    _set_position(state, qty=0.0, notional=0.0, unrealized=0.0)
    ev = evaluate(settings, state, now_mono=100.0)
    assert ev.triggered is False
    assert state.position_drawdown_breach_started_at_mono is None


def test_gate_does_not_trigger_when_unrealized_favorable() -> None:
    """Position making money -- adverse_bps is negative, gate quiet."""
    settings = _settings()
    state = BotState(settings)
    # Long $20 notional, +$0.10 unrealized = +50 bps in our favour.
    _set_position(state, qty=20.0, notional=20.0, unrealized=0.10)
    ev = evaluate(settings, state, now_mono=100.0)
    assert ev.triggered is False
    assert ev.adverse_bps < 0  # negative = favorable
    assert state.position_drawdown_breach_started_at_mono is None


def test_gate_does_not_trigger_below_threshold() -> None:
    settings = _settings()
    state = BotState(settings)
    # Long $20 notional, -$0.05 unrealized = -25 bps adverse < 50 bps.
    _set_position(state, qty=20.0, notional=20.0, unrealized=-0.05)
    ev = evaluate(settings, state, now_mono=100.0)
    assert ev.triggered is False
    assert ev.adverse_bps == 25.0
    assert state.position_drawdown_breach_started_at_mono is None


def test_gate_starts_timer_when_threshold_breached_first_time() -> None:
    """Threshold breached but duration not yet elapsed -- timer
    starts but gate stays quiet."""
    settings = _settings()
    state = BotState(settings)
    _set_position(state, qty=20.0, notional=20.0, unrealized=-0.20)  # -100 bps
    ev = evaluate(settings, state, now_mono=100.0)
    assert ev.triggered is False
    assert ev.adverse_bps == 100.0
    assert state.position_drawdown_breach_started_at_mono == 100.0


def test_gate_does_not_trigger_within_duration() -> None:
    """29s of breach < 30s required."""
    settings = _settings()
    state = BotState(settings)
    _set_position(state, qty=20.0, notional=20.0, unrealized=-0.20)
    evaluate(settings, state, now_mono=100.0)  # start timer
    ev = evaluate(settings, state, now_mono=129.0)  # 29s later
    assert ev.triggered is False
    assert ev.breach_seconds == 29.0


def test_gate_triggers_after_sustained_breach() -> None:
    """30s breach hits the duration threshold."""
    settings = _settings()
    state = BotState(settings)
    _set_position(state, qty=20.0, notional=20.0, unrealized=-0.20)
    evaluate(settings, state, now_mono=100.0)
    ev = evaluate(settings, state, now_mono=130.0)
    assert ev.triggered is True
    assert ev.adverse_bps == 100.0
    assert ev.breach_seconds == 30.0


def test_gate_resets_timer_when_position_recovers() -> None:
    """If unrealized PnL recovers above threshold, timer resets."""
    settings = _settings()
    state = BotState(settings)
    _set_position(state, qty=20.0, notional=20.0, unrealized=-0.20)
    evaluate(settings, state, now_mono=100.0)
    assert state.position_drawdown_breach_started_at_mono == 100.0
    # Position recovers (mark price bounces back).
    _set_position(state, qty=20.0, notional=20.0, unrealized=-0.05)  # -25 bps
    ev = evaluate(settings, state, now_mono=120.0)
    assert ev.triggered is False
    assert state.position_drawdown_breach_started_at_mono is None


def test_gate_works_for_short_positions() -> None:
    """Gate is symmetric -- a short with adverse drift triggers
    just like a long."""
    settings = _settings()
    state = BotState(settings)
    # Short 20 SUI: position_qty=-20, position_notional=20 (abs).
    # Price went UP, so unrealized PnL is negative.
    _set_position(state, qty=-20.0, notional=20.0, unrealized=-0.20)
    evaluate(settings, state, now_mono=100.0)
    ev = evaluate(settings, state, now_mono=130.0)
    assert ev.triggered is True
    assert ev.adverse_bps == 100.0


def test_gate_handles_zero_notional_safely() -> None:
    """Defensive: zero notional must not divide by zero."""
    settings = _settings()
    state = BotState(settings)
    _set_position(state, qty=0.0, notional=0.0, unrealized=-1.0)
    ev = evaluate(settings, state, now_mono=100.0)
    assert ev.triggered is False
    assert ev.adverse_bps == 0.0


def test_gate_threshold_can_be_tightened_via_setting() -> None:
    """Aggressive operator: 25 bps threshold trips on smaller drift."""
    settings = _settings(POSITION_DRAWDOWN_GATE_THRESHOLD_BPS=25.0)
    state = BotState(settings)
    _set_position(state, qty=20.0, notional=20.0, unrealized=-0.06)  # -30 bps
    evaluate(settings, state, now_mono=100.0)
    ev = evaluate(settings, state, now_mono=130.0)
    assert ev.triggered is True


def test_gate_duration_zero_fires_immediately() -> None:
    """Duration of 0 means trip on the first breached evaluation."""
    settings = _settings(POSITION_DRAWDOWN_GATE_DURATION_SECONDS=0.0)
    state = BotState(settings)
    _set_position(state, qty=20.0, notional=20.0, unrealized=-0.20)
    ev = evaluate(settings, state, now_mono=100.0)
    assert ev.triggered is True
