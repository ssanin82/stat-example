"""
Optional startup: rebuild UTC-day operator counters from Hyperliquid ``user_fills``.

Does not touch order state or the cancel-all / open-order reconcile path. When the returned
fill window looks incomplete vs the loaded JSON (or vs itself under truncation risk), the
in-memory operator metrics are left as after :func:`try_load_persistent_runtime_state` and a
warning is logged.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

from app.config import Settings
from app.enums import Side
from app.exchange.base import FillRaw, PerpExchangeAdapter
from app.exchange.factory import venue_account_address
from app.state import BotState
from app.utils.time import utc_now

logger = logging.getLogger(__name__)


def _utc_date_from_ms(time_ms: int) -> date:
    return datetime.fromtimestamp(time_ms / 1000.0, tz=timezone.utc).date()


def try_reconcile_operator_metrics_from_exchange_fills(
    settings: Settings,
    state: BotState,
    client: PerpExchangeAdapter,
) -> None:
    if not settings.persistent_runtime_reconcile_from_fills:
        return
    # Use the venue-aware address resolver (BUG-008). Previously
    # hardcoded ``hl_account_address`` which is empty on Bluefin /
    # GRVT, so operator-metric repair silently skipped on those venues
    # and we never noticed the daily counters weren't being rebuilt
    # on restart.
    addr = (venue_account_address(settings) or "").strip()
    if not addr:
        logger.info(
            "operator_metrics_reconcile skipped reason=no_account_address",
        )
        return

    symbol = settings.symbol
    try:
        symbol_fills: list[FillRaw] = client.fetch_recent_fills_raw(addr, symbol)
    except Exception:
        logger.warning(
            "operator_metrics_reconcile skipped: user_fills fetch failed; keeping loaded operator metrics",
            exc_info=True,
        )
        return

    today = utc_now().date()
    cap = settings.persistent_runtime_reconcile_fill_window_cap
    n = len(symbol_fills)

    oldest_d: Optional[date] = None
    if n:
        oldest_ms = min(f.time_ms for f in symbol_fills)
        oldest_d = _utc_date_from_ms(oldest_ms)

    fills_today = [f for f in symbol_fills if _utc_date_from_ms(f.time_ms) == today]
    recon_count = len(fills_today)
    recon_notional = sum(abs(f.px * f.sz) for f in fills_today)
    recon_pnl = sum(f.closed_pnl for f in fills_today)
    buy_c = sum(1 for f in fills_today if f.side == Side.BUY)
    sell_c = sum(1 for f in fills_today if f.side == Side.SELL)

    last_ts: Optional[datetime] = None
    if fills_today:
        last_ms = max(f.time_ms for f in fills_today)
        last_ts = datetime.fromtimestamp(last_ms / 1000.0, tz=timezone.utc)

    with state._lock:
        persisted_count = state.daily_trade_count
        persisted_anchor = state.operator_day_anchor_utc

    incomplete = False
    reasons: list[str] = []

    if persisted_anchor == today and recon_count < persisted_count:
        incomplete = True
        reasons.append(
            f"exchange_today_fill_count={recon_count} < persisted_daily_trade_count={persisted_count}"
        )

    if persisted_anchor == today and recon_count == 0 and persisted_count > 0:
        incomplete = True
        reasons.append("no_exchange_fills_today_but_persisted_daily_trade_count_gt_0")

    if n >= cap and oldest_d is not None and oldest_d >= today:
        incomplete = True
        reasons.append(
            f"fill_window_at_or_above_cap={cap}_and_no_fill_before_today_utc (oldest_seen={oldest_d.isoformat()})"
        )

    if incomplete:
        logger.warning(
            "operator_metrics_reconcile incomplete (%s); keeping metrics from persistent file / prior load",
            "; ".join(reasons),
        )
        return

    covers_before_today = oldest_d is not None and oldest_d < today
    pnl_trusted = covers_before_today or n < cap

    state.apply_operator_exchange_reconciliation(
        trade_count=recon_count,
        traded_notional=recon_notional,
        realized_pnl=recon_pnl,
        update_realized_pnl=pnl_trusted,
        last_fill_ts=last_ts,
        buy_count=buy_c,
        sell_count=sell_c,
    )
    logger.info(
        "operator_metrics_reconcile applied from user_fills "
        "trade_count=%s notional=%s realized_pnl_updated=%s",
        recon_count,
        recon_notional,
        pnl_trusted,
    )
