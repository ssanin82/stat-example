"""
Back-compat shim.

The exchange adapter Protocol has been renamed and moved to
:mod:`app.exchange.base` as :class:`~app.exchange.base.PerpExchangeAdapter`.
This module re-exports the Protocol under its previous name
``HyperliquidMMClient`` so any external code / scripts that still import
from here keep working. New code should import from ``app.exchange.base``.
"""

from __future__ import annotations

from app.exchange.base import PerpExchangeAdapter

# Legacy alias. Semantically identical to PerpExchangeAdapter.
HyperliquidMMClient = PerpExchangeAdapter

__all__ = ["HyperliquidMMClient", "PerpExchangeAdapter"]
