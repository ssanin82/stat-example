"""Tests for Codex MED #2: non-OKX adapters must REFUSE
``reduce_only=True`` rather than silently sending a non-reduce-only
order. Soft-flatten depends on the venue-side reduce-only guarantee
to prevent snowball; silently downgrading is a critical safety risk.

OKX adapter is the one currently wired for reduce-only and is
covered separately in ``test_okx_adapter_hard_cap.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.exchange = "test"
    s.symbol = "TEST"
    s.max_order_notional_usd = 100.0
    s.max_order_notional_hard_multiplier = 2.0
    return s


# ---------------------------------------------------------------------------
# Binance
# ---------------------------------------------------------------------------


def test_binance_reduce_only_true_raises() -> None:
    """Binance USDM supports reduceOnly natively, but the adapter
    doesn't propagate it yet. Refusal forces wiring before use."""
    from app.exchange.binance_client import BinanceClient

    client = BinanceClient.__new__(BinanceClient)
    client.has_write_access = lambda: True
    with pytest.raises(NotImplementedError, match="reduce_only"):
        client.place_post_only_limit(
            symbol="TEST",
            is_buy=True,
            sz=1.0,
            limit_px=100.0,
            client_order_id="cloid",
            reduce_only=True,
        )


# ---------------------------------------------------------------------------
# GRVT
# ---------------------------------------------------------------------------


def test_grvt_reduce_only_true_raises() -> None:
    from app.exchange.grvt_client import GrvtClient

    client = GrvtClient.__new__(GrvtClient)
    client.has_write_access = lambda: True
    with pytest.raises(NotImplementedError, match="reduce_only"):
        client.place_post_only_limit(
            symbol="TEST",
            is_buy=True,
            sz=1.0,
            limit_px=100.0,
            client_order_id="cloid",
            reduce_only=True,
        )


# ---------------------------------------------------------------------------
# Bluefin
# ---------------------------------------------------------------------------


def test_bluefin_reduce_only_true_raises() -> None:
    from app.exchange.bluefin_client import BluefinClient

    client = BluefinClient.__new__(BluefinClient)
    client.has_write_access = lambda: True
    with pytest.raises(NotImplementedError, match="reduce_only"):
        client.place_post_only_limit(
            symbol="TEST",
            is_buy=True,
            sz=1.0,
            limit_px=100.0,
            client_order_id="cloid",
            reduce_only=True,
        )


# ---------------------------------------------------------------------------
# Hyperliquid
# ---------------------------------------------------------------------------


def test_hyperliquid_reduce_only_true_raises() -> None:
    from app.exchange.hyperliquid_client import HyperliquidClient

    client = HyperliquidClient.__new__(HyperliquidClient)
    client._exchange = MagicMock()  # truthy
    with pytest.raises(NotImplementedError, match="reduce_only"):
        client.place_post_only_limit(
            symbol="TEST",
            is_buy=True,
            sz=1.0,
            limit_px=100.0,
            client_order_id="cloid",
            reduce_only=True,
        )
