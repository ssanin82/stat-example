"""
Venue-selecting factory for exchange adapter + WS streams.

This is the single place in the codebase that branches on ``settings.exchange``.
``app.main`` and test harnesses call these helpers; everything else stays
venue-agnostic behind :class:`app.exchange.base.PerpExchangeAdapter`.

Adding a new venue means:

1. implement the adapter (``PerpExchangeAdapter``) and the two WS streams;
2. add an ``elif venue == "<name>": ...`` branch in each builder below.

No other file should need to learn the new venue name.
"""

from __future__ import annotations

import logging
import queue
from typing import Any, Callable, Optional

from app.config import Settings
from app.exchange.base import PerpExchangeAdapter
from app.models import BestBidAsk
from app.state import BotState

logger = logging.getLogger(__name__)


def _normalise_venue(settings: Settings) -> str:
    """Return the canonical venue key
    (``"hyperliquid"`` | ``"grvt"`` | ``"bluefin"`` | ``"binance"``).

    ``Settings.exchange_allowed`` already normalises on load, but callers may
    build ``Settings`` via ``model_construct`` (tests), so we normalise here too.
    """
    v = (settings.exchange or "").strip().lower()
    if v in ("", "hl"):
        return "hyperliquid"
    return v


def build_adapter(settings: Settings) -> PerpExchangeAdapter:
    """Construct the venue-specific adapter implementing ``PerpExchangeAdapter``."""
    venue = _normalise_venue(settings)
    if venue == "hyperliquid":
        # Imported locally so test harnesses that only exercise GRVT paths don't
        # need to eagerly load the HL SDK.
        from app.exchange.hyperliquid_client import HyperliquidClient

        return HyperliquidClient(settings)
    if venue == "grvt":
        from app.exchange.grvt_client import GrvtClient

        return GrvtClient(
            config={
                "symbol": settings.symbol,
                "api_key": settings.grvt_api_key,
                "api_secret": settings.grvt_api_secret,
                "api_secret_file": settings.grvt_api_secret_file,
                "sub_account_id": settings.grvt_sub_account_id,
                "account_address": settings.grvt_account_address,
                "env": settings.grvt_env,
                "edge_url": settings.grvt_edge_url,
                "trade_url": settings.grvt_trade_url,
                "market_data_url": settings.grvt_market_data_url,
                "public_ws_url": settings.grvt_public_ws_url,
                "private_ws_url": settings.grvt_private_ws_url,
            }
        )
    if venue == "bluefin":
        from app.exchange.bluefin_client import BluefinClient

        return BluefinClient(settings)
    if venue == "binance":
        from app.exchange.binance_client import BinanceClient

        return BinanceClient(settings)
    if venue == "okx":
        from app.exchange.okx_client import OkxClient

        return OkxClient(settings)
    raise ValueError(
        f"build_adapter: unsupported EXCHANGE={settings.exchange!r} "
        "(expected 'hyperliquid' | 'grvt' | 'bluefin' | 'binance' | 'okx')"
    )


def build_public_stream(
    settings: Settings,
    state: BotState,
    symbol: str,
    on_bbo: Callable[[BestBidAsk], None],
) -> Any:
    """Construct the public-WS stream for the selected venue.

    Returns an object with ``start()``, ``stop()``, and ``request_reconnect()``.
    No common Protocol is defined for these because the bot core interacts with
    the stream only through ``main.py`` (composition root) — adding a Protocol
    here would be abstraction-for-abstraction's-sake.
    """
    venue = _normalise_venue(settings)
    if venue == "hyperliquid":
        from app.exchange.hyperliquid_public_ws import HyperliquidPublicStream

        return HyperliquidPublicStream(settings, state, symbol, on_bbo)
    if venue == "grvt":
        from app.exchange.grvt_public_ws import GrvtPublicStream

        return GrvtPublicStream(settings, state, symbol, on_bbo)
    if venue == "bluefin":
        from app.exchange.bluefin_public_ws import BluefinPublicStream

        return BluefinPublicStream(settings, state, symbol, on_bbo)
    if venue == "binance":
        # Trading-venue public stream — distinct from
        # binance_public_ws.py (which is for cross-venue REFERENCE).
        # See binance_trading_public_ws.py header for the rationale.
        from app.exchange.binance_trading_public_ws import (
            BinanceTradingPublicStream,
        )

        return BinanceTradingPublicStream(settings, state, symbol, on_bbo)
    if venue == "okx":
        from app.exchange.okx_public_ws import OkxPublicStream

        return OkxPublicStream(settings, state, symbol, on_bbo)
    raise ValueError(
        f"build_public_stream: unsupported EXCHANGE={settings.exchange!r}"
    )


def build_private_stream(
    settings: Settings,
    state: BotState,
    user_address: str,
    out_queue: queue.Queue,
    on_queue_drop: Optional[Callable[[int], None]] = None,
    adapter: Optional[PerpExchangeAdapter] = None,
) -> Any:
    """Construct the private-WS stream for the selected venue.

    Returns an object with ``start()`` and ``stop()``.

    When ``adapter`` is supplied and exposes ``on_cancel_confirmed``
    (currently Bluefin only — see ``app/exchange/base.py`` docstring on
    cancel-confirmation gating), the venue stream is wired to forward
    ``OrderCancellationUpdate`` events into the adapter so its pending
    cancel state clears and the replacement-placement gate opens.
    """
    venue = _normalise_venue(settings)
    if venue == "hyperliquid":
        from app.exchange.hyperliquid_ws import HyperliquidPrivateStream

        return HyperliquidPrivateStream(
            settings,
            user_address,
            out_queue,
            on_queue_drop=on_queue_drop,
            state=state,
        )
    if venue == "grvt":
        from app.exchange.grvt_ws import GrvtPrivateStream

        return GrvtPrivateStream(
            settings,
            user_address,
            out_queue,
            on_queue_drop=on_queue_drop,
            state=state,
        )
    if venue == "bluefin":
        from app.exchange.bluefin_ws import BluefinPrivateStream

        cancel_cb = getattr(adapter, "on_cancel_confirmed", None) if adapter is not None else None
        return BluefinPrivateStream(
            settings,
            user_address,
            out_queue,
            on_queue_drop=on_queue_drop,
            state=state,
            on_cancel_confirmed=cancel_cb,
        )
    if venue == "binance":
        from app.exchange.binance_ws import BinancePrivateStream

        # The Binance private WS needs the listenKey lifecycle helpers
        # off the adapter. Pull them via getattr to keep this factory
        # tolerant of test harnesses that mock the adapter.
        if adapter is None:
            raise ValueError(
                "build_private_stream: EXCHANGE=binance requires the "
                "adapter argument (used for listenKey spawn / keepalive)"
            )
        spawn = getattr(adapter, "spawn_listen_key")
        keepalive = getattr(adapter, "keepalive_listen_key")
        return BinancePrivateStream(
            settings,
            out_queue,
            listen_key_provider=spawn,
            listen_key_keepalive=keepalive,
            on_queue_drop=on_queue_drop,
            state=state,
        )
    if venue == "okx":
        from app.exchange.okx_ws import OkxPrivateStream

        stream = OkxPrivateStream(
            settings,
            out_queue,
            on_queue_drop=on_queue_drop,
            state=state,
        )
        # Inject the contract-value conversion factor so the WS can
        # translate OKX's contract-quantity wire format to base units.
        # Adapter must be the OkxClient instance the bot already built;
        # we read .contract_value off it via getattr to stay tolerant of
        # test mocks.
        if adapter is not None:
            ctval = getattr(adapter, "contract_value", None)
            if isinstance(ctval, (int, float)) and ctval > 0:
                stream.set_contract_value(float(ctval))
        return stream
    raise ValueError(
        f"build_private_stream: unsupported EXCHANGE={settings.exchange!r}"
    )


def venue_account_address(settings: Settings) -> str:
    """Return the per-venue account address used for private-WS subscriptions.

    Kept in the factory so ``main.py`` doesn't branch on venue.

    Binance has no on-chain account address; we derive a stable
    identifier from the API key prefix so log lines that key off
    "address" still produce something usable. The identifier is NOT
    the credential — it's just for log correlation.
    """
    venue = _normalise_venue(settings)
    if venue == "hyperliquid":
        return (settings.hl_account_address or "").strip()
    if venue == "grvt":
        return (settings.grvt_sub_account_id or settings.grvt_account_address or "").strip()
    if venue == "bluefin":
        return (settings.bluefin_account_address or "").strip()
    if venue == "binance":
        # Binance: derive a stable, non-secret identifier from the API
        # key prefix. The full key is the credential and stays in env;
        # we emit only the first 8 chars + "...".
        key = (settings.binance_api_key or "").strip()
        if not key:
            return ""
        return f"{key[:8]}..." if len(key) > 8 else key
    if venue == "okx":
        # OKX: same shape as Binance -- API key prefix as a stable
        # non-secret display identifier. Full key stays in env.
        key = (settings.okx_api_key or "").strip()
        if not key:
            return ""
        return f"{key[:8]}..." if len(key) > 8 else key
    return ""


__all__ = [
    "build_adapter",
    "build_public_stream",
    "build_private_stream",
    "venue_account_address",
]
