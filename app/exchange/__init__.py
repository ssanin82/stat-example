"""
Exchange adapters.

Do not import concrete clients (``HyperliquidClient``, ``GrvtClient``) from this package
root — that would eagerly load venue SDKs. Import them directly from their modules at the
app entrypoint (``app.exchange.hyperliquid_client``, ``app.exchange.grvt_client``).

For type hints in bot / execution / market_data, use :class:`PerpExchangeAdapter` from
:mod:`app.exchange.base`. ``HyperliquidMMClient`` is a legacy alias kept for back-compat.
"""

from app.exchange.base import FillRaw, OpenOrderRaw, PerpExchangeAdapter
from app.exchange.mm_client_protocol import HyperliquidMMClient

__all__ = [
    "PerpExchangeAdapter",
    "OpenOrderRaw",
    "FillRaw",
    "HyperliquidMMClient",
]
