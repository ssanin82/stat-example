"""Position-aware drawdown gate.

Closes a stuck losing position via post-only orders before it has
time to grow large enough to trip the absolute drawdown / session-
loss gates. Bridges the gap between "inventory-skew prevents
accumulation" and "$10 absolute drawdown trips a hard kill" — for
small positions ($20-$30 notional on a $1000 account), $10 drawdown
is a 30-50 % adverse move that effectively never fires; this gate
fires on a 50-bp adverse-on-position move sustained for 30 s.

Mechanism: not a kill. It transitions the bot into ``SOFT_FLATTENING``
mode, which suspends normal quoting and runs a patient post-only-only
exit (see ``bot.py::_run_soft_flatten_tick``). Once flat, the bot
resumes RUNNING. Operator never has to intervene.

Why post-only and not taker:
    A 50-bp threshold is well below any catastrophic-loss line. The
    bot has time. Paying 2 bps of taker fee + a few bps of slippage
    on every soft-flatten would frequently consume more PnL than the
    drift itself; staying patient and earning a maker rebate on the
    exit is strictly better. The hard floor (drawdown $10 / session
    loss $10) still fires KILL with the existing taker-flatten if
    things actually go bad.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.config import Settings
from app.state import BotState

from app import clock as _clock


@dataclass(frozen=True, slots=True)
class PositionDrawdownEvaluation:
    """Snapshot of the gate's view of the world at one tick.

    ``triggered=True`` means the bot should enter SOFT_FLATTENING.
    ``adverse_bps`` is the current adverse-on-position level (positive
    number = adverse; 0 or negative = no breach). ``breach_seconds``
    is how long we've been continuously over the threshold.
    """

    triggered: bool
    adverse_bps: float
    threshold_bps: float
    breach_seconds: float
    duration_required_seconds: float
    position_notional_usd: float
    unrealized_pnl_usd: float


def _adverse_pnl_bps(
    unrealized_pnl_usd: Optional[float],
    position_notional_usd: Optional[float],
) -> Optional[float]:
    """Convert unrealized PnL into adverse-on-position bps. Positive
    return value = adverse (we're losing), negative = favorable.
    """
    if unrealized_pnl_usd is None or position_notional_usd is None:
        return None
    notional = abs(float(position_notional_usd))
    if notional <= 1e-9:
        return None
    return -(float(unrealized_pnl_usd) / notional) * 10_000.0


def evaluate(
    settings: Settings,
    state: BotState,
    *,
    now_mono: Optional[float] = None,
) -> PositionDrawdownEvaluation:
    """Pure-ish evaluator (mutates one timer field on ``state``).

    Reads the current position + unrealized PnL, classifies whether
    the adverse threshold is breached, and tracks the breach-start
    monotonic timestamp on ``state.position_drawdown_breach_started_at_mono``.
    Returns whether the gate should fire (``triggered=True``).

    Caller is responsible for actually transitioning the bot to
    SOFT_FLATTENING -- this function does NOT mutate ``soft_flatten_active``.
    """
    if now_mono is None:
        now_mono = _clock.monotonic()
    threshold_bps = float(settings.position_drawdown_gate_threshold_bps)
    duration_s = float(settings.position_drawdown_gate_duration_seconds)
    pos = state.position
    unrealized = pos.unrealized_pnl_usd
    notional = pos.position_notional
    adverse_bps = _adverse_pnl_bps(unrealized, notional)
    if adverse_bps is None or adverse_bps < threshold_bps:
        # Not breached -- clear the timer.
        state.position_drawdown_breach_started_at_mono = None
        return PositionDrawdownEvaluation(
            triggered=False,
            adverse_bps=adverse_bps if adverse_bps is not None else 0.0,
            threshold_bps=threshold_bps,
            breach_seconds=0.0,
            duration_required_seconds=duration_s,
            position_notional_usd=float(notional or 0.0),
            unrealized_pnl_usd=float(unrealized or 0.0),
        )
    # Breached. Start (or continue) the timer.
    if state.position_drawdown_breach_started_at_mono is None:
        state.position_drawdown_breach_started_at_mono = now_mono
    breach_seconds = now_mono - state.position_drawdown_breach_started_at_mono
    triggered = breach_seconds >= duration_s
    return PositionDrawdownEvaluation(
        triggered=triggered,
        adverse_bps=adverse_bps,
        threshold_bps=threshold_bps,
        breach_seconds=breach_seconds,
        duration_required_seconds=duration_s,
        position_notional_usd=float(notional or 0.0),
        unrealized_pnl_usd=float(unrealized or 0.0),
    )
