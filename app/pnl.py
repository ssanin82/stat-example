from __future__ import annotations

from app.models import Fill, PnlSnapshot, PositionSnapshot
from app.utils.time import utc_now


class PnlTracker:
    """
    Session PnL from fills + unrealized from the latest position snapshot.

    - realized_pnl_usd: sum of Hyperliquid per-fill ``closedPnl`` (as returned by the
      API) for **current-process session fills only** (``ts_fill >= session_started_at_utc``).
      Replay fills from REST / WS snapshot before that boundary are ingested for dedupe /
      persistence but must not call :meth:`on_fill`.
    - fees_usd: running SIGNED sum of fees per fill. Bot convention is
      positive = cost (paid to platform), negative = income (rebate
      received). A maker-rebate-heavy session ends with NEGATIVE
      fees_usd; the ``- fees_usd`` math below then credits the rebate
      to net PnL automatically. Pre-2026-05-05 this used ``abs(fee)``
      and silently dropped rebate signal -- see HLFillRaw docstring +
      BUG-018.
    - unrealized_pnl_usd: from the latest position snapshot (exchange mark).
    - total_pnl_usd: realized_pnl_usd + unrealized_pnl_usd - fees_usd. **Net of fees.**
      Both Hyperliquid `closedPnl` and Bluefin `realizedPnlE9` are gross of
      fees (fees come on a separate field), so this subtraction is the
      single, non-double-counting source of net session PnL — which is what
      `MAX_SESSION_LOSS_USD` and `MAX_DRAWDOWN_USD` are meant to bound.
    - equity_usd: account USD basis from ``fetch_account_snapshot`` (perps margin/cross
      ``accountValue``, or spot USDC total under unified accounts). Used for drawdown vs
      session peak. If equity is unknown, drawdown stays 0.
    """

    def __init__(self) -> None:
        self._realized = 0.0
        self._fees = 0.0
        self._session_peak_equity: float | None = None

    def on_fill(self, f: Fill, closed_pnl_component: float = 0.0) -> None:
        # SIGNED accumulation -- ``f.fee`` is positive for paid fees,
        # negative for rebates received (per HLFillRaw convention).
        # The ``- self._fees`` subtraction in build_snapshot then
        # produces correct net PnL whether the session was rebate-
        # positive or fee-negative.
        self._fees += f.fee
        self._realized += closed_pnl_component

    def build_snapshot(
        self,
        position: PositionSnapshot,
        equity: float | None,
    ) -> PnlSnapshot:
        unreal = position.unrealized_pnl_usd
        total = self._realized + unreal - self._fees
        peak = self._session_peak_equity
        if equity is not None:
            if peak is None:
                peak = equity
            else:
                peak = max(peak, equity)
            self._session_peak_equity = peak
        drawdown = 0.0
        if peak is not None and equity is not None:
            drawdown = max(0.0, peak - equity)
        return PnlSnapshot(
            realized_pnl_usd=self._realized,
            unrealized_pnl_usd=unreal,
            total_pnl_usd=total,
            fees_usd=self._fees,
            equity_usd=equity,
            drawdown_usd=drawdown,
            session_peak_equity_usd=peak or 0.0,
            ts=utc_now(),
        )
