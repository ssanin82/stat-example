from __future__ import annotations

from typing import Optional

from app.config import Settings
from app.enums import BotStatus, DesyncPhase, RiskAction
from app.models import BestBidAsk, PnlSnapshot, RiskDecision, ToxicitySnapshot
from app.utils.time import seconds_since


def evaluate_risk(
    settings: Settings,
    *,
    bot_status: BotStatus,
    manual_pause: bool,
    killed: bool,
    flatten_mode: bool,
    market: BestBidAsk | None,
    position_qty: float,
    position_notional: float,
    open_order_count: int,
    pnl: PnlSnapshot,
    toxicity: ToxicitySnapshot,
    execution_errors: int,
    desync: bool,
    desync_phase: DesyncPhase = DesyncPhase.OK,
    desync_quarantine_remaining: int = 0,
    trades_last_minute: int = 0,
    reconcile_auto_pause: bool = False,
    public_ws_live_path: bool = False,
    public_ws_connected: bool = True,
    public_ws_seconds_since_message: Optional[float] = None,
    public_ws_seen_first_bbo: bool = False,
    account_seconds_since_refresh: Optional[float] = None,
) -> RiskDecision:
    """
    Single-symbol pre-trade checks. Order matters: capital / kill triggers before
    requiring a fresh book; quoting-specific rules need valid mid.
    """
    reasons: list[str] = []

    if killed or bot_status == BotStatus.KILLED:
        return RiskDecision(action=RiskAction.KILL, reasons=["killed"])

    if not settings.trading_enabled:
        return RiskDecision(action=RiskAction.NO_QUOTE, reasons=["trading_disabled"])

    if manual_pause:
        return RiskDecision(action=RiskAction.NO_QUOTE, reasons=["manual_pause"])

    if bot_status == BotStatus.PAUSED:
        sub = "reconcile_stall" if reconcile_auto_pause else "paused"
        return RiskDecision(action=RiskAction.NO_QUOTE, reasons=[sub])

    if bot_status == BotStatus.RECOVERING_MARKET_DATA:
        # Supervisor clears this status to RUNNING after a fresh book; never quote while recovering.
        return RiskDecision(
            action=RiskAction.NO_QUOTE,
            reasons=["recovering_market_data"],
        )

    if flatten_mode or bot_status == BotStatus.FLATTENING:
        return RiskDecision(action=RiskAction.FLATTEN, reasons=["flatten_mode"])

    if desync_phase == DesyncPhase.UNRECOVERABLE:
        return RiskDecision(action=RiskAction.KILL, reasons=["desync_unrecoverable"])

    if desync_quarantine_remaining > 0 or desync_phase == DesyncPhase.RECOVERED:
        return RiskDecision(action=RiskAction.NO_QUOTE, reasons=["desync_quarantine"])

    if desync:
        return RiskDecision(action=RiskAction.CANCEL_ALL, reasons=["order_desync"])

    if execution_errors >= settings.max_execution_errors:
        return RiskDecision(action=RiskAction.KILL, reasons=["execution_errors"])

    if pnl.total_pnl_usd <= -settings.max_session_loss_usd:
        return RiskDecision(action=RiskAction.KILL, reasons=["max_session_loss"])

    if pnl.drawdown_usd >= settings.max_drawdown_usd:
        return RiskDecision(action=RiskAction.KILL, reasons=["max_drawdown"])

    if market is None or market.mid_price is None or market.mid_price <= 0:
        return RiskDecision(action=RiskAction.NO_QUOTE, reasons=["no_market_data"])

    if public_ws_live_path:
        if not public_ws_connected:
            if settings.market_data_recovery_enabled:
                return RiskDecision(
                    action=RiskAction.NO_QUOTE,
                    reasons=["stale_market_recovery", "public_ws_disconnected"],
                )
            return RiskDecision(
                action=RiskAction.KILL, reasons=["public_ws_disconnected"]
            )
        w_age = public_ws_seconds_since_message
        if w_age is None:
            if not public_ws_seen_first_bbo:
                return RiskDecision(
                    action=RiskAction.NO_QUOTE,
                    reasons=["public_ws_awaiting_first_bbo"],
                )
            if settings.market_data_recovery_enabled:
                return RiskDecision(
                    action=RiskAction.NO_QUOTE,
                    reasons=["stale_market_recovery", "public_ws_message_gap"],
                )
            return RiskDecision(
                action=RiskAction.KILL, reasons=["public_ws_message_gap"]
            )
        pk = float(settings.public_ws_stale_kill_seconds)
        pw = float(settings.public_ws_stale_warn_seconds)
        if w_age >= pk:
            if settings.market_data_recovery_enabled:
                return RiskDecision(
                    action=RiskAction.NO_QUOTE,
                    reasons=["stale_market_recovery", "public_ws_stale_kill"],
                )
            return RiskDecision(
                action=RiskAction.KILL, reasons=["public_ws_stale_kill"]
            )
        if w_age >= pw:
            reasons.append("public_ws_stale_warn")
    else:
        now_stale = seconds_since(market.ts_local)
        if now_stale is not None:
            if now_stale >= settings.stale_data_kill_seconds:
                if settings.market_data_recovery_enabled:
                    return RiskDecision(
                        action=RiskAction.NO_QUOTE,
                        reasons=["stale_market_recovery", "stale_data_kill_threshold"],
                    )
                return RiskDecision(action=RiskAction.KILL, reasons=["stale_data_kill"])
            if now_stale >= settings.stale_data_warn_seconds:
                reasons.append("stale_data_warn")

    if "stale_data_warn" in reasons or "public_ws_stale_warn" in reasons:
        return RiskDecision(action=RiskAction.NO_QUOTE, reasons=reasons)

    # TODO-002: account-data-stale gate. A healthy book + stale account is a
    # dangerous combination on Bluefin (BUG-006-style scenario where REST
    # refresh silently stops). Independent of public-WS health — this gate
    # catches "REST refresh just stopped happening" cases that wouldn't trip
    # the market-data freshness gates above. Returns NO_QUOTE with reason
    # ``account_data_stale`` so Bot._should_cancel_resting_on_no_quote
    # promotes it to cancel-resting (BUG-009 family).
    if (
        account_seconds_since_refresh is not None
        and float(settings.account_data_stale_kill_seconds) > 0
        and account_seconds_since_refresh
        >= float(settings.account_data_stale_kill_seconds)
    ):
        return RiskDecision(
            action=RiskAction.NO_QUOTE,
            reasons=["account_data_stale"],
        )

    if (
        settings.max_trades_per_minute > 0
        and trades_last_minute >= settings.max_trades_per_minute
    ):
        return RiskDecision(
            action=RiskAction.NO_QUOTE,
            reasons=["trade_rate_limit"],
        )

    if abs(position_qty) > settings.max_abs_position * 1.0001:
        return RiskDecision(
            action=RiskAction.ASK_ONLY if position_qty > 0 else RiskAction.BID_ONLY,
            reasons=["max_abs_position"],
        )

    if position_notional > settings.max_position_notional_usd * 1.0001:
        eps = 1e-10
        if position_qty > eps:
            return RiskDecision(
                action=RiskAction.ASK_ONLY,
                reasons=["max_position_notional"],
            )
        if position_qty < -eps:
            return RiskDecision(
                action=RiskAction.BID_ONLY,
                reasons=["max_position_notional"],
            )
        return RiskDecision(
            action=RiskAction.NO_QUOTE,
            reasons=["max_position_notional_ambiguous"],
        )

    if open_order_count > settings.max_open_orders:
        return RiskDecision(action=RiskAction.CANCEL_ALL, reasons=["max_open_orders"])

    if toxicity.hard_trigger and toxicity.toxic_side:
        reasons.append("toxicity_hard")
        # Only request a flatten when there is actually inventory to
        # close. Without this guard, a sustained toxicity hard-trigger
        # on a flat position made the bot hit the FLATTEN branch on
        # every quote tick: each call cancelled all open orders (which
        # the Bluefin cancel-by-hash workaround amplifies into orphan
        # cancellation events), the OrderManager's local view desynced,
        # and the watchdog fired every 20-30 minutes from the
        # cancel-and-reconcile thrash. Side-suppression (the path
        # below) is the correct response when there is no inventory:
        # same risk-control intent (don't trade against toxic flow),
        # without the busy-loop side effects.
        # Observed 2026-04-25 in logs.1777120993719.json: 3915
        # flatten_started events in 13 minutes on a flat position;
        # two watchdog firings 24 minutes apart from the same root
        # cause.
        #
        # 2026-05-07: replaced the legacy taker-flatten with
        # ``RiskAction.SOFT_FLATTEN`` so the toxicity path uses the
        # post-only soft-flatten worker (with phase-2-immediate +
        # taker-fallback) instead of an immediate market_close. See
        # ``plans/20260507-sf-frontend.md`` Phase 6 for the design
        # rationale. The legacy ``flatten_on_kill`` flag now only
        # gates kill-event flattening, not toxicity-triggered ones.
        if abs(position_qty) >= 1e-8:
            return RiskDecision(
                action=RiskAction.SOFT_FLATTEN,
                reasons=reasons + ["toxicity_soft_flatten"],
            )
        if toxicity.toxic_side.value == "BUY":
            return RiskDecision(
                action=RiskAction.ASK_ONLY,
                reasons=reasons,
                bid_size_mult=0.0,
            )
        return RiskDecision(
            action=RiskAction.BID_ONLY,
            reasons=reasons,
            ask_size_mult=0.0,
        )

    spread_add = 0.0
    bid_mult = 1.0
    ask_mult = 1.0
    if toxicity.soft_trigger:
        spread_add = 3.0
        bid_mult = ask_mult = 0.85
        reasons.append("toxicity_soft")

    return RiskDecision(
        action=RiskAction.ALLOW,
        reasons=reasons or ["ok"],
        bid_size_mult=bid_mult,
        ask_size_mult=ask_mult,
        spread_add_bps=spread_add,
    )
