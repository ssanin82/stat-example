"""Tests for the venue-selecting factory in ``app.exchange.factory``.

These tests guarantee:

* ``build_adapter`` / ``build_public_stream`` / ``build_private_stream`` dispatch
  on ``settings.exchange`` without hardcoding venue anywhere else.
* the GRVT selection path returns GRVT classes, not HL ones.
* ``require_trading_credentials_when_enabled`` branches per venue.
* unknown venues are rejected loudly by ``Settings`` validation.

We do not connect real sockets in unit tests; construction + parser hooks only.
"""

from __future__ import annotations

import queue
from unittest.mock import patch

import pytest

from app.config import Settings, require_trading_credentials_when_enabled
from app.exchange.factory import (
    build_adapter,
    build_private_stream,
    build_public_stream,
    venue_account_address,
)
from app.exchange.grvt_client import GrvtClient
from app.exchange.grvt_public_ws import GrvtPublicStream
from app.exchange.grvt_ws import GrvtPrivateStream
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> Settings:
    """Build test Settings without touching any dotenv file.

    All GRVT credential fields must be explicitly blanked here so that a
    real ``.env`` / exported shell value (e.g. ``GRVT_SUB_ACCOUNT_ID``
    set for live trading) doesn't leak into tests that assert on the
    absence-of-credentials path — ``venue_account_address`` and
    ``require_trading_credentials_when_enabled`` both read those fields
    directly from ``Settings``.
    """
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        # Force deterministic GRVT credential defaults inside tests (no env leakage).
        "GRVT_API_KEY": "",
        "GRVT_API_SECRET": "",
        "GRVT_API_SECRET_FILE": "",
        "GRVT_ACCOUNT_ADDRESS": "",
        "GRVT_SUB_ACCOUNT_ID": "",
        "SYMBOL": "ETH",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _state(settings: Settings) -> BotState:
    return BotState(settings)


# ---------------------------------------------------------------------------
# Settings field validation
# ---------------------------------------------------------------------------


def test_exchange_default_is_hyperliquid() -> None:
    s = _settings()
    assert s.exchange == "hyperliquid"


def test_exchange_accepts_grvt() -> None:
    s = _settings(EXCHANGE="grvt")
    assert s.exchange == "grvt"


def test_exchange_accepts_hl_alias() -> None:
    s = _settings(EXCHANGE="hl")
    assert s.exchange == "hyperliquid"


def test_exchange_is_case_insensitive() -> None:
    s = _settings(EXCHANGE="GRVT")
    assert s.exchange == "grvt"


def test_exchange_rejects_unknown_venue() -> None:
    """As of v1.0.21+ ``binance`` is a supported trading venue (see
    plans/20260420-binance-move/plan.md). Use a different unknown
    string to exercise the rejection path.
    """
    with pytest.raises(ValueError):
        _settings(EXCHANGE="kraken")


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_build_adapter_defaults_to_hyperliquid_client() -> None:
    """``EXCHANGE`` unset → HyperliquidClient. Avoids SDK network calls via patching."""
    s = _settings()
    # Patch the SDK entry points so constructing the HL client does not hit the network.
    with patch("app.exchange.hyperliquid_client.Info") as _info:
        _info.return_value.meta.return_value = {"universe": []}
        client = build_adapter(s)
    from app.exchange.hyperliquid_client import HyperliquidClient

    assert isinstance(client, HyperliquidClient)


def test_build_adapter_for_grvt_returns_scaffold() -> None:
    s = _settings(EXCHANGE="grvt")
    with patch("app.exchange.grvt_client.GrvtClient._load_symbol_spec_strict") as _spec:
        from app.exchange.symbol_spec import SymbolSpec

        _spec.return_value = SymbolSpec(
            price_tick=0.1,
            size_step=0.001,
            min_size=0.001,
            min_notional_usd=10.0,
            sz_decimals=3,
            source="fallback",
        )
        client = build_adapter(s)
    assert isinstance(client, GrvtClient)


def test_build_public_stream_selects_venue() -> None:
    s_grvt = _settings(EXCHANGE="grvt")
    st = _state(s_grvt)
    stream = build_public_stream(s_grvt, st, "ETH", on_bbo=lambda _bb: None)
    assert isinstance(stream, GrvtPublicStream)

    s_hl = _settings(EXCHANGE="hyperliquid")
    from app.exchange.hyperliquid_public_ws import HyperliquidPublicStream

    stream_hl = build_public_stream(s_hl, _state(s_hl), "ETH", on_bbo=lambda _bb: None)
    assert isinstance(stream_hl, HyperliquidPublicStream)


def test_build_private_stream_selects_venue() -> None:
    s_grvt = _settings(EXCHANGE="grvt")
    st = _state(s_grvt)
    q: queue.Queue = queue.Queue(maxsize=16)
    stream = build_private_stream(s_grvt, st, "0xabc", q, on_queue_drop=None)
    assert isinstance(stream, GrvtPrivateStream)

    s_hl = _settings(EXCHANGE="hyperliquid")
    from app.exchange.hyperliquid_ws import HyperliquidPrivateStream

    stream_hl = build_private_stream(
        s_hl, _state(s_hl), "0xabc", q, on_queue_drop=None
    )
    assert isinstance(stream_hl, HyperliquidPrivateStream)


def test_venue_account_address_picks_per_venue_field() -> None:
    s_hl = _settings(HL_ACCOUNT_ADDRESS="0xhl")
    assert venue_account_address(s_hl) == "0xhl"

    s_grvt = _settings(EXCHANGE="grvt", GRVT_ACCOUNT_ADDRESS="0xgrvt")
    assert venue_account_address(s_grvt) == "0xgrvt"


# ---------------------------------------------------------------------------
# GRVT stream parser hooks / no-op safety
# ---------------------------------------------------------------------------


def test_grvt_public_stream_feed_message_for_tests() -> None:
    s = _settings(EXCHANGE="grvt")
    st = _state(s)
    applied: list[float] = []
    stream = GrvtPublicStream(
        s,
        st,
        "ETH_USDT_Perp",
        on_bbo=lambda bb: applied.append(float(bb.mid_price or 0.0)),
    )
    # Mini-ticker snapshot feed (``v1.mini.s@200``): every message carries
    # all BBO fields — both sides are populated on every tick.
    stream.feed_message_for_tests(
        '{"stream":"v1.mini.s","feed":{"instrument":"ETH_USDT_Perp",'
        '"event_time":"1700000000000000000",'
        '"best_bid_price":"100","best_bid_size":"1",'
        '"best_ask_price":"101","best_ask_size":"1"}}'
    )
    assert applied == [100.5]
    assert st.public_ws_last_message_wall_ts is not None
    # Second snapshot updates the bid and re-emits against the fresh ask.
    stream.feed_message_for_tests(
        '{"stream":"v1.mini.s","feed":{"instrument":"ETH_USDT_Perp",'
        '"event_time":"1700000000200000000",'
        '"best_bid_price":"100.5","best_ask_price":"101"}}'
    )
    assert applied == [100.5, 100.75]
    # Messages on other streams (book snapshots, mini delta) must be ignored
    # so a stale subscription doesn't leak through the BBO callback.
    stream.feed_message_for_tests(
        '{"stream":"v1.book.s","feed":{"instrument":"ETH_USDT_Perp",'
        '"event_time":"1700000000400000000",'
        '"bids":[{"price":"99","size":"1"}],"asks":[{"price":"100","size":"1"}]}}'
    )
    stream.feed_message_for_tests(
        '{"stream":"v1.mini.d","feed":{"instrument":"ETH_USDT_Perp",'
        '"event_time":"1700000000400000000",'
        '"best_bid_price":"99","best_ask_price":"100"}}'
    )
    assert applied == [100.5, 100.75]


def test_grvt_public_stream_start_is_safe_without_ws_dependency() -> None:
    s = _settings(EXCHANGE="grvt", PUBLIC_WS_ENABLED=False)
    stream = GrvtPublicStream(s, _state(s), "ETH", on_bbo=lambda _bb: None)
    stream.start()
    stream.stop()


def test_grvt_public_stream_stop_is_safe_noop() -> None:
    s = _settings(EXCHANGE="grvt")
    stream = GrvtPublicStream(s, _state(s), "ETH", on_bbo=lambda _bb: None)
    # Must not raise even if start() was never called.
    stream.stop()
    stream.request_reconnect()


def test_grvt_private_stream_feed_message_for_tests() -> None:
    s = _settings(EXCHANGE="grvt")
    q: queue.Queue = queue.Queue(maxsize=16)
    stream = GrvtPrivateStream(s, "0xabc", q)
    stream.feed_message_for_tests(
        '{"stream":"v1.fill","feed":{"event_time":"1700000000000000000","trade_id":"tid1","order_id":"0x10","instrument":"ETH_USDT_Perp","is_buyer":true,"size":"0.01","price":"3000","fee":"-0.001","realized_pnl":"0"}}'
    )
    ev = q.get_nowait()
    assert ev.fill_id.startswith("tid1_")


def test_grvt_private_stream_stop_is_safe_noop() -> None:
    s = _settings(EXCHANGE="grvt")
    q: queue.Queue = queue.Queue(maxsize=16)
    stream = GrvtPrivateStream(s, "0xabc", q)
    stream.stop()


# ---------------------------------------------------------------------------
# Credentials validation per venue
# ---------------------------------------------------------------------------


def test_credentials_check_hl_requires_hl_fields() -> None:
    s = _settings(TRADING_ENABLED=True, EXCHANGE="hyperliquid")
    with pytest.raises(ValueError, match="HL_SECRET_KEY"):
        require_trading_credentials_when_enabled(s)

    s2 = _settings(
        TRADING_ENABLED=True,
        EXCHANGE="hyperliquid",
        HL_SECRET_KEY="k",
        HL_ACCOUNT_ADDRESS="",
    )
    with pytest.raises(ValueError, match="HL_ACCOUNT_ADDRESS"):
        require_trading_credentials_when_enabled(s2)

    s_ok = _settings(
        TRADING_ENABLED=True,
        EXCHANGE="hyperliquid",
        HL_SECRET_KEY="k",
        HL_ACCOUNT_ADDRESS="0xhl",
    )
    require_trading_credentials_when_enabled(s_ok)  # no raise


def test_credentials_check_grvt_requires_grvt_fields() -> None:
    s = _settings(TRADING_ENABLED=True, EXCHANGE="grvt")
    with pytest.raises(ValueError, match="GRVT_API_KEY"):
        require_trading_credentials_when_enabled(s)

    s_secret = _settings(
        TRADING_ENABLED=True,
        EXCHANGE="grvt",
        GRVT_API_KEY="k",
    )
    with pytest.raises(ValueError, match="GRVT_API_SECRET"):
        require_trading_credentials_when_enabled(s_secret)

    s_addr = _settings(
        TRADING_ENABLED=True,
        EXCHANGE="grvt",
        GRVT_API_KEY="k",
        GRVT_API_SECRET="s",
    )
    with pytest.raises(ValueError, match="GRVT_SUB_ACCOUNT_ID"):
        require_trading_credentials_when_enabled(s_addr)

    s_ok = _settings(
        TRADING_ENABLED=True,
        EXCHANGE="grvt",
        GRVT_API_KEY="k",
        GRVT_API_SECRET="s",
        GRVT_SUB_ACCOUNT_ID="123456",
    )
    require_trading_credentials_when_enabled(s_ok)  # no raise


def test_credentials_check_hl_ignores_grvt_fields() -> None:
    """Having GRVT creds set while EXCHANGE=hyperliquid must still demand HL creds."""
    s = _settings(
        TRADING_ENABLED=True,
        EXCHANGE="hyperliquid",
        GRVT_API_KEY="k",
        GRVT_API_SECRET="s",
        GRVT_ACCOUNT_ADDRESS="0xgrvt",
    )
    with pytest.raises(ValueError, match="HL_SECRET_KEY"):
        require_trading_credentials_when_enabled(s)


def test_sanitized_dict_masks_grvt_secret() -> None:
    s = _settings(EXCHANGE="grvt", GRVT_API_SECRET="top-secret")
    d = s.sanitized_dict()
    assert d.get("GRVT_API_SECRET") == "***"


def test_sanitized_dict_masks_telegram_bot_token() -> None:
    s = _settings(TELEGRAM_BOT_TOKEN="123456:secret-token")
    d = s.sanitized_dict()
    assert d.get("TELEGRAM_BOT_TOKEN") == "***"


def test_sanitized_dict_masks_binance_api_key_and_secret() -> None:
    # Both halves of Binance auth are sensitive: knowing only the API
    # key is harmless on its own, but pairing it with a leaked secret
    # makes signed requests possible. Redact both.
    s = _settings(
        EXCHANGE="binance",
        BINANCE_API_KEY="pk-abcdef",
        BINANCE_API_SECRET="sk-ghijkl",
    )
    d = s.sanitized_dict()
    assert d.get("BINANCE_API_KEY") == "***"
    assert d.get("BINANCE_API_SECRET") == "***"


def test_sanitized_dict_helper_excludes_file_paths() -> None:
    # *_FILE keys point at where the secret lives on disk, not the
    # secret itself -- safe to surface in /config so operators can
    # diagnose "is the bot reading the right file?". We test the
    # name-pattern helper directly because building a Settings with
    # HL_SECRET_KEY_FILE=<path> triggers a validator that wants to
    # actually open the file -- not what we want to assert here.
    from app.config import _is_sensitive_key_name

    assert _is_sensitive_key_name("HL_SECRET_KEY") is True
    assert _is_sensitive_key_name("BINANCE_API_KEY") is True
    assert _is_sensitive_key_name("BINANCE_API_SECRET") is True
    assert _is_sensitive_key_name("TELEGRAM_BOT_TOKEN") is True
    assert _is_sensitive_key_name("BLUEFIN_PRIVATE_KEY") is True
    # *_FILE excluded
    assert _is_sensitive_key_name("HL_SECRET_KEY_FILE") is False
    assert _is_sensitive_key_name("GRVT_API_SECRET_FILE") is False
    # Operational config not redacted
    assert _is_sensitive_key_name("HL_BASE_URL") is False
    assert _is_sensitive_key_name("EXCHANGE") is False
    assert _is_sensitive_key_name("PORT") is False


def test_sanitized_dict_passes_through_non_secret_fields() -> None:
    # URLs, hostnames, ports, exchange name, etc. are operational
    # config and must not be redacted -- /config is a debugging surface.
    s = _settings(EXCHANGE="binance", PORT=8000)
    d = s.sanitized_dict()
    assert d.get("EXCHANGE") == "binance"
    assert d.get("PORT") == 8000
    assert d.get("HL_BASE_URL", "").startswith("https://")
