"""Tests for BUG-003 — inherited-position seeding at startup.

Two layers of coverage:

1. **Parser-level**: ``BluefinClient.fetch_position`` correctly maps
   the various ``/api/v1/account`` response shapes into a
   ``PositionSnapshot``. This is where BUG-003's root cause likely
   lives — the parser zeroing a non-zero venue position because of
   field-name / sign / unit mismatch.

2. **State integration**: ``BotState.apply_account_position_only`` +
   the live ``state.position`` round-trip preserve the parsed values.
   Catches any future change that re-introduces the silent zeroing
   between parser and state.

The main.py lifespan integration that wires
``client.fetch_position`` into startup is exercised end-to-end at
deploy time via the operator's smoke checks; these tests cover the
two layers below it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from app.exchange.bluefin_client import BluefinClient
from app.models import PositionSnapshot
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


_DUMMY_HEX64 = "0x" + "01" * 32
_DUMMY_PRIVKEY = "0x" + ("00" * 31) + "01"


def _settings(**overrides: Any) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "bluefin",
        "BLUEFIN_PRIVATE_KEY": _DUMMY_PRIVKEY,
        "BLUEFIN_ACCOUNT_ADDRESS": _DUMMY_HEX64,
        "BLUEFIN_NETWORK": "SUI_PROD",
        "BLUEFIN_ONE_CT_ENABLED": True,
        "BLUEFIN_REST_URL": "https://api.sui-prod.bluefin.io",
        "BLUEFIN_AUTH_URL": "https://auth.api.sui-prod.bluefin.io",
        "BLUEFIN_TRADE_URL": "https://trade.api.sui-prod.bluefin.io",
        "BLUEFIN_PUBLIC_WS_URL": "wss://stream.api.sui-prod.bluefin.io",
        "BLUEFIN_PRIVATE_WS_URL": "wss://stream.api.sui-prod.bluefin.io",
        "SYMBOL": "SUI-PERP",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _patch_exchange_info_response() -> dict[str, Any]:
    """The constructor calls /v1/exchange/info; return a minimal stub
    that satisfies the strict bootstrap reference-data check (mirrors
    the helper in test_bluefin_adapter_smoke.py)."""
    return {
        "markets": [
            {
                "symbol": "SUI-PERP",
                "marketAddress": _DUMMY_HEX64,
                "tickSizeE9": "100000",
                "stepSizeE9": "100000000",
                "minOrderQuantityE9": "100000000",
                "minTradePriceE9": "100000",
                "minTradeQuantityE9": "100000000",
            }
        ],
        "contractsConfig": {"idsId": _DUMMY_HEX64},
        "tradingGasFeeE9": "0",
        "serverTimeAtMillis": 1700000000000,
        "timezone": "UTC",
    }


def _build_client(positions_response: dict[str, Any]) -> BluefinClient:
    """Build a BluefinClient with ``_request`` mocked to return the
    given account response when fetch_position calls /api/v1/account."""
    s = _settings()
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _patch_exchange_info_response()
        client = BluefinClient(s)
    # Now replace the bound method to return our positions response on
    # the next call.
    def _fake_request(label, method, url, path, **kwargs):
        if path == "/api/v1/account":
            return positions_response
        return {}
    client._request = _fake_request  # type: ignore[method-assign]
    return client


# --- happy paths ---


def test_fetch_position_long_returns_signed_qty() -> None:
    """A non-zero LONG position must seed with positive qty.

    This is the BUG-003 regression case: with `sizeE9=336000000000`
    and `side=LONG`, the parser must return position_qty=336.0 (not 0).
    """
    client = _build_client({
        "positions": [
            {
                "symbol": "SUI-PERP",
                "sizeE9": "336000000000",  # 336 SUI
                "side": "LONG",
                "avgEntryPriceE9": "942500000",  # $0.9425
                "markPriceE9": "932300000",  # $0.9323
                "unrealizedPnlE9": "-3427200",  # -$0.0034272
            }
        ]
    })
    pos = client.fetch_position(_DUMMY_HEX64, "SUI-PERP")
    assert pos.symbol == "SUI-PERP"
    assert pos.position_qty == 336.0
    assert pos.avg_entry_price == 0.9425
    assert pos.mark_price == 0.9323
    # notional = 336 × 0.9323 = 313.2528
    assert pos.position_notional == pytest.approx(313.2528, abs=1e-3)


def test_fetch_position_short_returns_negative_qty() -> None:
    client = _build_client({
        "positions": [
            {
                "symbol": "SUI-PERP",
                "sizeE9": "100000000000",  # |size| = 100
                "side": "SHORT",
                "avgEntryPriceE9": "942500000",
                "markPriceE9": "932300000",
                "unrealizedPnlE9": "1020000000",
            }
        ]
    })
    pos = client.fetch_position(_DUMMY_HEX64, "SUI-PERP")
    assert pos.position_qty == -100.0
    assert pos.position_notional == pytest.approx(93.23, abs=1e-3)


def test_fetch_position_no_positions_list_returns_zero() -> None:
    """Account response with no positions key — bot starts flat
    (correct behaviour, no false alarm)."""
    client = _build_client({"positions": []})
    pos = client.fetch_position(_DUMMY_HEX64, "SUI-PERP")
    assert pos.position_qty == 0.0
    assert pos.avg_entry_price is None
    assert pos.mark_price is None
    assert pos.position_notional == 0.0


def test_fetch_position_other_symbol_does_not_match() -> None:
    """The account has a position in a DIFFERENT symbol; we should
    return zero for the configured symbol."""
    client = _build_client({
        "positions": [
            {
                "symbol": "ETH-PERP",
                "sizeE9": "1000000000000",
                "side": "LONG",
                "avgEntryPriceE9": "3000000000000",
                "markPriceE9": "3000000000000",
                "unrealizedPnlE9": "0",
            }
        ]
    })
    pos = client.fetch_position(_DUMMY_HEX64, "SUI-PERP")
    assert pos.position_qty == 0.0


def test_fetch_position_multiple_symbols_picks_correct_one() -> None:
    """Account with positions in multiple symbols — must pick the
    configured symbol's row, not the first."""
    client = _build_client({
        "positions": [
            {
                "symbol": "ETH-PERP",
                "sizeE9": "1000000000000",
                "side": "LONG",
                "avgEntryPriceE9": "3000000000000",
                "markPriceE9": "3000000000000",
                "unrealizedPnlE9": "0",
            },
            {
                "symbol": "SUI-PERP",
                "sizeE9": "200000000000",
                "side": "LONG",
                "avgEntryPriceE9": "942500000",
                "markPriceE9": "932300000",
                "unrealizedPnlE9": "0",
            },
        ]
    })
    pos = client.fetch_position(_DUMMY_HEX64, "SUI-PERP")
    assert pos.position_qty == 200.0


def test_fetch_position_error_response_raises() -> None:
    """Server-side error (response includes ``code``) → raise rather
    than fabricate a flat snapshot. BUG-006: previously fail-open
    behaviour silently zeroed position state on auth/server errors,
    making safety caps think the bot was flat when it wasn't.
    """
    client = _build_client({"code": "ERR_AUTH"})
    with pytest.raises(RuntimeError, match="ERR_AUTH"):
        client.fetch_position(_DUMMY_HEX64, "SUI-PERP")


def test_fetch_account_snapshot_error_response_raises() -> None:
    """Same fail-closed contract for account snapshot."""
    client = _build_client({"code": "ERR_AUTH"})
    with pytest.raises(RuntimeError, match="ERR_AUTH"):
        client.fetch_account_snapshot(_DUMMY_HEX64)


def test_fetch_position_zero_size_returns_zero() -> None:
    """Closed-out position — sizeE9=0 — returns clean zero snapshot."""
    client = _build_client({
        "positions": [
            {
                "symbol": "SUI-PERP",
                "sizeE9": "0",
                "side": "LONG",
                "avgEntryPriceE9": "0",
                "markPriceE9": "0",
                "unrealizedPnlE9": "0",
            }
        ]
    })
    pos = client.fetch_position(_DUMMY_HEX64, "SUI-PERP")
    assert pos.position_qty == 0.0


def test_fetch_position_case_insensitive_symbol_match() -> None:
    """Bluefin's symbol field may come back lower / upper case;
    parser should match either."""
    client = _build_client({
        "positions": [
            {
                "symbol": "sui-perp",  # lowercase from venue
                "sizeE9": "100000000000",
                "side": "LONG",
                "avgEntryPriceE9": "942500000",
                "markPriceE9": "932300000",
                "unrealizedPnlE9": "0",
            }
        ]
    })
    pos = client.fetch_position(_DUMMY_HEX64, "SUI-PERP")
    assert pos.position_qty == 100.0


# --- state round-trip ---


def test_apply_account_position_only_round_trips_long() -> None:
    """End-to-end: parser produces a PositionSnapshot, state applies
    it, the live state.position reflects the venue qty. This is the
    contract main.py's startup-seed step relies on."""
    s = _settings()
    state = BotState(s)
    snap = PositionSnapshot(
        symbol="SUI-PERP",
        position_qty=336.0,
        avg_entry_price=0.9425,
        mark_price=0.9323,
        position_notional=313.2528,
        unrealized_pnl_usd=-3.4272,
    )
    state.apply_account_position_only(snap, None)
    assert state.position.symbol == "SUI-PERP"
    assert state.position.position_qty == 336.0
    assert state.position.avg_entry_price == 0.9425
    assert state.position.mark_price == 0.9323
    assert state.position.position_notional == pytest.approx(313.2528)


def test_apply_account_position_only_round_trips_short() -> None:
    s = _settings()
    state = BotState(s)
    snap = PositionSnapshot(
        symbol="SUI-PERP",
        position_qty=-100.0,
        avg_entry_price=0.9425,
        mark_price=0.9323,
        position_notional=93.23,
        unrealized_pnl_usd=1.02,
    )
    state.apply_account_position_only(snap, None)
    assert state.position.position_qty == -100.0


def test_apply_account_position_only_round_trips_zero() -> None:
    """Flat position from venue → state position is flat."""
    s = _settings()
    state = BotState(s)
    snap = PositionSnapshot(
        symbol="SUI-PERP",
        position_qty=0.0,
        avg_entry_price=None,
        mark_price=None,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    state.apply_account_position_only(snap, None)
    assert state.position.position_qty == 0.0
    assert state.position.avg_entry_price is None


# --- regression guards for the BUG-003 fix path ---


def test_refresh_account_only_preserves_state_on_rest_error() -> None:
    """BUG-006 regression: when ``client.fetch_position`` raises
    (e.g. auth error), ``refresh_account_only`` must NOT overwrite
    state with a flat snapshot. Previously the client returned
    a synthetic flat ``PositionSnapshot``; now it raises, and the
    caller's existing exception handler preserves last-known state.
    """
    from app.market_data import refresh_account_only
    from app.models import PositionSnapshot

    s = _settings()
    state = BotState(s)
    # Seed state with a known non-zero position from a prior
    # successful refresh.
    seeded = PositionSnapshot(
        symbol="SUI-PERP",
        position_qty=200.0,
        avg_entry_price=0.94,
        mark_price=0.93,
        position_notional=186.0,
        unrealized_pnl_usd=-2.0,
    )
    state.apply_account_position_only(seeded, None)
    assert state.position.position_qty == 200.0

    # Now build a client whose fetch_position raises.
    class _BoomClient(BluefinClient):
        pass
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _patch_exchange_info_response()
        client = _BoomClient(s)

    def _boom(label, method, url, path, **kwargs):
        if path == "/api/v1/account":
            raise RuntimeError("simulated network failure")
        return _patch_exchange_info_response()
    client._request = _boom  # type: ignore[method-assign]

    # Run refresh; the inner exception handler should swallow + log,
    # state must NOT change.
    refresh_account_only(
        client, state, "0xdeadbeef" + "00" * 30, None, None,
        ingest_fills_via_rest=False,
    )
    assert state.position.position_qty == 200.0, (
        "REST failure must NOT zero state.position; previously "
        "synthetic-flat handling silently wiped inventory truth."
    )


def test_seed_path_unblocks_max_abs_position_cap() -> None:
    """Regression: with venue position +336 SUI seeded into state,
    a downstream call to risk.evaluate_risk that includes
    position_qty=336 must return ASK_ONLY (cap fires) on a config
    with max_abs_position=100.

    This is the BUG-003 acceptance test from the perspective of the
    safety mechanism that was neutered: once the seed step is in
    place, the cap path that was already correct works as intended.
    """
    from app.enums import BotStatus, RiskAction
    from app.models import BestBidAsk, PnlSnapshot, ToxicitySnapshot
    from app.risk import evaluate_risk

    s = _settings(MAX_ABS_POSITION=100.0, TRADING_ENABLED=True)
    pnl = PnlSnapshot(
        realized_pnl_usd=0.0,
        unrealized_pnl_usd=0.0,
        total_pnl_usd=0.0,
        fees_usd=0.0,
        equity_usd=400.0,
        drawdown_usd=0.0,
        session_peak_equity_usd=400.0,
    )
    tox = ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.5,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=1.0,
        hard_trigger=False,
        soft_trigger=False,
        toxic_side=None,
        delayed_markout_sample_count=0,
        adverse_uses_delayed_markouts=False,
    )
    market = BestBidAsk(
        symbol="SUI-PERP",
        best_bid=0.9322,
        best_ask=0.9324,
        mid_price=0.9323,
        spread_bps=2.146,
        ts_local=None,
    )

    decision = evaluate_risk(
        s,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=market,
        position_qty=336.0,  # the inherited position
        position_notional=313.2528,
        open_order_count=0,
        pnl=pnl,
        toxicity=tox,
        execution_errors=0,
        desync=False,
    )
    # Cap fires: with +336 long and 100 cap, ASK_ONLY is returned.
    assert decision.action == RiskAction.ASK_ONLY, (
        f"max_abs_position cap must fire when position_qty exceeds "
        f"the cap; got {decision.action} reasons={decision.reasons}"
    )
    assert "max_abs_position" in decision.reasons


