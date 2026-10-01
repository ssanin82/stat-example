"""Take-profit (TP) mode pure helpers.

v1.5.33 — opportunistic harvest of in-the-money positions via
aggressive post-only maker exit.

Operator-observed motivation (2026-05-22): the bot's existing far-
side ladder rung often watches a favorable price spike all the way
through its quote and back to flat — we requote ahead of the move
each tick, and when the trend reverses we never captured the bulk
of the favorable swing. TP closes that loop:

  * When ``uPnL / |position_notional| * 10_000`` crosses
    ``UPNL_HARVEST_TRIGGER_BPS``, we ARM TP mode.
  * Normal ladder quoting is suspended; we cancel the ladder and
    place a single aggressive post-only close order at the most
    aggressive non-crossing maker price (one tick inside the far
    touch from the close direction).
  * If price retraces and uPnL drops below
    ``(trigger - UPNL_HARVEST_DISARM_MARGIN_BPS)``, OR the dwell
    exceeds ``UPNL_HARVEST_MAX_DWELL_SECONDS`` without a fill, we
    DISARM and resume normal quoting. A cooldown of
    ``UPNL_HARVEST_ARM_COOLDOWN_SECONDS`` prevents immediate re-arm
    so chop near the threshold doesn't ping-pong the executor.

Distinction vs SF (soft-flatten):

  * SF is a SAFETY exit — adverse uPnL or toxicity hard-trigger.
    Escalates to taker (IOC then market_close) so the position
    ALWAYS closes.
  * TP is an OPPORTUNITY exit — favorable uPnL. NEVER escalates
    past post-only; if price escapes, we just disarm and resume
    quoting. No urgency.

If SF is active when TP would arm, TP yields (SF wins). If TP is
already active when SF arms, the caller cancels the TP order and
transitions into SOFT_FLATTENING.

This module holds the LOGIC. The executor wiring lives in
``bot.py::_run_take_profit_tick`` and the arming-check site is
in the main quote loop. Pure functions here let tests cover
sign / threshold / hysteresis corner cases without booting a Bot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.enums import Side


def compute_upnl_bps(
    *,
    unrealized_pnl_usd: Optional[float],
    position_notional_usd: Optional[float],
) -> Optional[float]:
    """Convert unrealized PnL into favorable-on-position bps. Positive
    return value = favorable (we're winning), negative = adverse.

    Mirror image of ``position_drawdown_gate._adverse_pnl_bps`` but
    with the sign flipped — TP arms on the favorable tail, the
    drawdown gate arms on the adverse tail. Returns None when either
    input is missing or notional is dust.
    """
    if unrealized_pnl_usd is None or position_notional_usd is None:
        return None
    notional = abs(float(position_notional_usd))
    if notional <= 1e-9:
        return None
    return (float(unrealized_pnl_usd) / notional) * 10_000.0


@dataclass(frozen=True)
class TPArmDecision:
    """Output of the per-tick arming check.

    ``should_arm=True`` means the caller should transition the bot
    into TP mode. ``upnl_bps`` and ``trigger_bps`` are included for
    the entry-log payload + the storage row.

    ``suppressed_reason`` is set when the gate WOULD have armed
    based on uPnL but was suppressed by a higher-priority condition
    (cooldown active, SF active, below min notional, feature
    disabled). Mostly diagnostic — lets the operator see in the
    snapshot whether the feature is gated, vs simply not
    triggering.
    """

    should_arm: bool
    upnl_bps: Optional[float]
    trigger_bps: float
    suppressed_reason: str = ""


def should_arm_tp(
    *,
    enabled: bool,
    trigger_bps: float,
    upnl_bps: Optional[float],
    position_qty: float,
    position_notional_usd: float,
    min_notional_usd: float,
    sf_active: bool,
    tp_active: bool,
    now_mono: float,
    arm_cooldown_until_mono: float,
    bot_status_allows_quoting: bool,
) -> TPArmDecision:
    """Decide whether to arm TP on this tick. Pure function.

    Priority of suppressions (first hit short-circuits to
    ``should_arm=False``):

    1. ``enabled=False`` -> feature off.
    2. ``tp_active=True`` -> already in TP, nothing to do.
    3. ``sf_active=True`` -> SF wins, suppress.
    4. ``bot_status_allows_quoting=False`` -> bot is killed /
       paused / flatten_mode; don't arm a new opportunistic mode.
    5. ``abs(position_qty) < 1e-8`` -> no inventory.
    6. ``position_notional_usd < min_notional_usd`` -> below the
       venue + local floor; can't form a valid close order.
    7. ``now_mono < arm_cooldown_until_mono`` -> hysteresis
       cooldown after a recent disarm.
    8. ``upnl_bps is None`` -> can't compute uPnL (no notional, no
       mark, etc.). Same as ``below threshold``.
    9. ``upnl_bps < trigger_bps`` -> threshold not breached.

    Only when all nine are clear does ``should_arm=True``.
    """
    if not enabled:
        return TPArmDecision(False, upnl_bps, trigger_bps, "disabled")
    if tp_active:
        return TPArmDecision(False, upnl_bps, trigger_bps, "already_active")
    if sf_active:
        return TPArmDecision(False, upnl_bps, trigger_bps, "sf_active")
    if not bot_status_allows_quoting:
        return TPArmDecision(
            False, upnl_bps, trigger_bps, "bot_status_blocks"
        )
    if abs(float(position_qty)) < 1e-8:
        return TPArmDecision(False, upnl_bps, trigger_bps, "flat")
    if (
        float(min_notional_usd) > 0
        and float(position_notional_usd) + 1e-9 < float(min_notional_usd)
    ):
        return TPArmDecision(
            False, upnl_bps, trigger_bps, "below_min_notional"
        )
    if float(now_mono) < float(arm_cooldown_until_mono):
        return TPArmDecision(
            False, upnl_bps, trigger_bps, "arm_cooldown"
        )
    if upnl_bps is None:
        return TPArmDecision(
            False, upnl_bps, trigger_bps, "upnl_unavailable"
        )
    if float(upnl_bps) < float(trigger_bps):
        return TPArmDecision(
            False, upnl_bps, trigger_bps, "below_trigger"
        )
    return TPArmDecision(True, upnl_bps, trigger_bps, "")


@dataclass(frozen=True)
class TPDisarmDecision:
    """Output of the per-tick disarm check while TP is active."""

    should_disarm: bool
    reason: str  # "" when staying armed; otherwise a short tag
    upnl_bps: Optional[float]
    dwell_seconds: float


def should_disarm_tp(
    *,
    upnl_bps: Optional[float],
    trigger_bps: float,
    disarm_margin_bps: float,
    armed_at_mono: float,
    now_mono: float,
    max_dwell_seconds: float,
    sf_armed_this_tick: bool,
    position_flat: bool,
) -> TPDisarmDecision:
    """Decide whether to disarm TP on this tick. Pure function.

    Disarm triggers (any one fires):

    1. ``sf_armed_this_tick=True`` — SF takes precedence. Caller is
       responsible for transitioning into SF after disarm.
    2. ``position_flat=True`` — already flat (TP order filled, or
       independent fill closed the position). Disarm cleanly.
    3. ``upnl_bps`` falls to ``trigger_bps - disarm_margin_bps`` or
       below — hysteresis exit. Price retraced, harvest opportunity
       gone.
    4. Dwell ``now_mono - armed_at_mono >= max_dwell_seconds``.
       Timeout. The post-only never filled and price isn't coming
       back to us; give up and resume quoting.

    ``upnl_bps is None`` does NOT disarm — book may be momentarily
    unavailable; stay armed and let the executor see if it can
    place against a fresh quote next tick.
    """
    dwell = max(0.0, float(now_mono) - float(armed_at_mono))
    if sf_armed_this_tick:
        return TPDisarmDecision(True, "sf_takes_over", upnl_bps, dwell)
    if position_flat:
        return TPDisarmDecision(True, "position_flat", upnl_bps, dwell)
    exit_threshold_bps = float(trigger_bps) - float(disarm_margin_bps)
    if upnl_bps is not None and float(upnl_bps) <= exit_threshold_bps:
        return TPDisarmDecision(True, "upnl_retraced", upnl_bps, dwell)
    if dwell >= float(max_dwell_seconds):
        return TPDisarmDecision(True, "max_dwell", upnl_bps, dwell)
    return TPDisarmDecision(False, "", upnl_bps, dwell)


def compute_tp_target_price(
    *,
    pos_qty: float,
    best_bid: float,
    best_ask: float,
    tick_size: float,
) -> tuple[Side, float]:
    """Returns ``(close_side, target_price)`` for the TP executor.

    TP places the most aggressive post-only price that won't cross.
    For a long position closing (SELL): one tick INTO the spread
    from the bid (``best_bid + tick_size``), capped at the ask so
    on a 1-tick spread we stay at the touch rather than crossing.
    Symmetric for a short.

    Mirror of ``soft_flatten.compute_target_price`` phase-2 logic,
    except TP never has a "phase 1" patient stage — TP IS the
    aggressive variant by design. If the operator wants patient
    behaviour they leave the feature disabled and trust the ladder.

    ``tick_size <= 0`` (degenerate symbol spec) falls back to the
    far touch with no tick offset — equivalent to phase-0 SF
    pricing. Should not happen in production.
    """
    if pos_qty > 0:
        side = Side.SELL
        if tick_size <= 0:
            return side, float(best_ask)
        # SELL: post one tick above bid, but never above ask (would
        # cross). On a 1-tick spread (bid+1 == ask) this collapses
        # to AT the ask, which is post-only-OK (we sit at the touch).
        return side, min(float(best_ask), float(best_bid) + float(tick_size))
    side = Side.BUY
    if tick_size <= 0:
        return side, float(best_bid)
    # BUY: post one tick below ask, but never below bid.
    return side, max(float(best_bid), float(best_ask) - float(tick_size))
