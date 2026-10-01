"""
Per-symbol trading constraints for Hyperliquid perps.

``price_tick`` is the **decimal grid** step ``10 ** -(6 - szDecimals)`` (same basis as
hyperliquid-python-sdk ``_slippage_price`` rounding). Hyperliquid **also** enforces a
maximum of **5 significant figures** for non-integer limit prices; see official
`tick-and-lot-size <https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size>`_.
So ``price_tick=0.01`` for ETH (``szDecimals=4``) does **not** mean every ``0.01`` grid
price is valid (e.g. ``2244.95`` has six significant figures and is rejected).

Use :mod:`app.exchange.hyperliquid_precision` for full limit-price normalization before submit.

Hyperliquid enforces a **$10 minimum order notional** on perps at the matching engine;
``meta()`` may omit ``minNotionalUsd`` or expose a lower value. We always apply at least
that **$10 floor**; if ``minNotionalUsd`` in meta is higher, we use the larger value.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

logger = logging.getLogger(__name__)

# Hyperliquid perp minimum order value (exchange-enforced; not always reflected in meta).
_PERP_MIN_NOTIONAL_USD_EXCHANGE_FLOOR = 10.0

Source = Literal[
    "hyperliquid_meta",
    "grvt_meta",
    "bluefin_meta",
    "binance_meta",
    "fallback",
]


@dataclass(frozen=True, slots=True)
class SymbolSpec:
    price_tick: float
    size_step: float
    min_size: float
    min_notional_usd: float
    sz_decimals: int
    source: Source


# Last-resort values (legacy ETH-like); only used when meta fetch fails.
FALLBACK_SYMBOL_SPEC = SymbolSpec(
    price_tick=0.01,
    size_step=0.0001,
    min_size=0.001,
    min_notional_usd=_PERP_MIN_NOTIONAL_USD_EXCHANGE_FLOOR,
    sz_decimals=4,
    source="fallback",
)


def symbol_spec_from_hyperliquid_meta(
    meta: dict[str, Any],
    symbol: str,
) -> SymbolSpec:
    """
    Build a SymbolSpec from ``Info.meta()`` (perp dex ``""``) response.
    Raises ValueError if the symbol is missing or fields are invalid.
    """
    universe = meta.get("universe")
    if not isinstance(universe, list):
        raise ValueError("meta.universe missing or not a list")
    row = next((u for u in universe if isinstance(u, dict) and u.get("name") == symbol), None)
    if row is None:
        raise ValueError(f"symbol {symbol!r} not found in Hyperliquid perp universe")

    try:
        sz_decimals = int(row["szDecimals"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"invalid szDecimals for {symbol!r}") from e
    if sz_decimals < 0 or sz_decimals > 12:
        raise ValueError(f"szDecimals out of range for {symbol!r}: {sz_decimals}")

    size_step = 10.0 ** (-sz_decimals)
    price_decimals = max(0, 6 - sz_decimals)
    price_tick = 10.0 ** (-price_decimals)

    min_size = size_step
    raw_min = row.get("minOrderSz")
    if raw_min is not None:
        try:
            min_size = max(float(raw_min), size_step)
        except (TypeError, ValueError):
            pass

    min_notional = _PERP_MIN_NOTIONAL_USD_EXCHANGE_FLOOR
    raw_ntl = row.get("minNotionalUsd")
    if raw_ntl is not None:
        try:
            parsed = float(raw_ntl)
            min_notional = max(_PERP_MIN_NOTIONAL_USD_EXCHANGE_FLOOR, parsed)
        except (TypeError, ValueError):
            pass
    else:
        logger.debug(
            "symbol_spec minNotionalUsd missing in meta for %s; using exchange floor %s",
            symbol,
            _PERP_MIN_NOTIONAL_USD_EXCHANGE_FLOOR,
        )

    return SymbolSpec(
        price_tick=price_tick,
        size_step=size_step,
        min_size=min_size,
        min_notional_usd=min_notional,
        sz_decimals=sz_decimals,
        source="hyperliquid_meta",
    )
