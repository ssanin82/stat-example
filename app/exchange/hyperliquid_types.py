from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.enums import Side


@dataclass
class HLFillRaw:
    """Normalized fill row from Hyperliquid user_fills.

    Fee sign convention (bot-internal, applied uniformly across venues
    by their respective adapters):

    * **Positive ``fee`` = cost to us** (we paid the platform a fee).
    * **Negative ``fee`` = income to us** (we received a maker rebate).
    * **Zero ``fee`` = no charge** (rare; promo windows, dust fills).

    Per-venue raw values follow each venue's native convention; the
    adapter normalizes to the bot convention above. Specifically:

    * **Hyperliquid**: native is positive=paid (no rebates today).
      Pass-through.
    * **Binance**: ``commission`` field is always positive (paid).
      Pass-through.
    * **OKX**: native is the OPPOSITE — positive=rebate-received,
      negative=fee-paid (per OKX V5 ``fillFee`` docs). The OKX adapter
      negates so the bot sees the canonical convention.
    * **Bluefin**: native is bot convention; pass-through.
    * **GRVT**: native is bot convention; pass-through.

    Downstream PnL math: ``total_pnl = realized + unrealized - fees``
    where ``fees`` is the SIGNED sum across the session. A maker-rebate-
    heavy session has negative ``fees``, so the subtraction credits the
    rebate to net PnL automatically.

    History note (BUG-018 / 2026-05-05): all adapters previously did
    ``abs(fee)`` here, which silently flipped OKX rebates into
    apparent fees-paid and undercounted net PnL by 2× the rebate.
    Switched to signed convention 2026-05-05; abs() removed in OKX
    adapters; other venues retain their effective behavior since
    they don't currently see rebates in production.
    """

    fill_id: str
    oid: Optional[int]
    coin: str
    side: Side
    px: float
    sz: float
    fee: float
    time_ms: int
    closed_pnl: float
    raw: dict[str, Any]


@dataclass
class HLOpenOrderRaw:
    oid: int
    coin: str
    side: Side
    limit_px: float
    sz: float
    timestamp: int
    cloid: Optional[str] = None


# Venue-neutral aliases — see ``app.exchange.base`` for the single source of
# truth. These exist so legacy imports from this module keep working.
OpenOrderRaw = HLOpenOrderRaw
FillRaw = HLFillRaw
