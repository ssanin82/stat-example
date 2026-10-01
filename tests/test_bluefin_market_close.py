"""BUG-014: ``BluefinClient.market_close`` must use a permissive
slippage cap on the IOC limit price; clipping to ``mark_price`` left
the order non-marketable in fast-moving regimes (snap_20260427_055140:
3 attempts × 125 s with zero fills as the market dropped 22 bps during
a long-close, leaving the operator with a stuck +65 SUI inventory).

Fix: ``MARKET_CLOSE_SLIPPAGE_BPS`` (default 100 = 1 %) shifts the limit
AGAINST the close direction so any realistic depth at retail size
fills through. ``reduce_only=True`` and IOC keep the safety semantics
intact (order can't grow inventory or rest at a bad price).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from app.exchange.bluefin_client import BluefinClient
from app.models import PositionSnapshot
from tests.settings_helpers import UnitTestSettings


_DUMMY_HEX64 = "0x" + "01" * 32
_DUMMY_PRIVKEY = "0x" + ("00" * 31) + "01"


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": True,
        "EXCHANGE": "bluefin",
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "GRVT_API_KEY": "",
        "GRVT_API_SECRET": "",
        "GRVT_ACCOUNT_ADDRESS": "",
        "GRVT_SUB_ACCOUNT_ID": "",
        "BLUEFIN_PRIVATE_KEY": _DUMMY_PRIVKEY,
        "BLUEFIN_ACCOUNT_ADDRESS": _DUMMY_HEX64,
        "BLUEFIN_NETWORK": "SUI_PROD",
        "BLUEFIN_ONE_CT_ENABLED": False,
        "SYMBOL": "SUI-PERP",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _exchange_info() -> Any:
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


def _make_client(settings) -> BluefinClient:
    """Construct a BluefinClient with the bootstrap REST round trip mocked."""
    with patch("app.exchange.bluefin_client.BluefinClient._request") as m:
        m.return_value = _exchange_info()
        client = BluefinClient(settings)
    return client


def _capture_place_order_call(client: BluefinClient) -> dict[str, Any]:
    """Replace ``_place_order`` with a recorder so we can inspect the
    args ``market_close`` passes through.
    """
    captured: dict[str, Any] = {}

    def fake_place_order(**kwargs):
        captured.update(kwargs)
        return {"data": {"status": "FILLED", "filled_qty": kwargs["sz"]}}

    client._place_order = fake_place_order  # type: ignore[assignment]
    return captured


def _set_position(client: BluefinClient, qty: float, mark: float) -> None:
    """Stub fetch_position to return a synthetic snapshot."""

    def fake_fetch(addr: str, sym: str) -> PositionSnapshot:
        return PositionSnapshot(
            symbol=sym,
            position_qty=qty,
            avg_entry_price=mark,
            mark_price=mark,
            position_notional=abs(qty * mark),
            unrealized_pnl_usd=0.0,
        )

    client.fetch_position = fake_fetch  # type: ignore[assignment]


# --- close-long → SELL with limit BELOW mark -------------------------------


def test_close_long_uses_limit_below_mark_by_slippage_bps() -> None:
    """A long position closes via SELL. The SELL limit must be BELOW
    mark by ``MARKET_CLOSE_SLIPPAGE_BPS`` so any bid in the slippage
    window crosses.

    Pre-fix: limit = mark exactly → SELL non-marketable in any
    downtrend.
    Post-fix at default 100 bps: limit = mark × (1 − 0.01).
    """
    s = _settings(MARKET_CLOSE_SLIPPAGE_BPS=100.0)
    client = _make_client(s)
    captured = _capture_place_order_call(client)
    _set_position(client, qty=49.0, mark=0.9316)

    client.market_close("SUI-PERP")
    assert captured["is_buy"] is False  # closing long ⇒ SELL
    assert captured["sz"] == 49.0
    expected = 0.9316 * (1.0 - 100.0 / 10_000.0)
    assert abs(captured["limit_px"] - expected) < 1e-9, (
        f"SELL limit {captured['limit_px']} not mark*(1-slip); expected {expected}"
    )
    # Reduce-only + IOC stay set so safety semantics hold.
    assert captured["reduce_only"] is True
    assert captured["ioc"] is True


def test_close_short_uses_limit_above_mark_by_slippage_bps() -> None:
    """A short position closes via BUY. The BUY limit must be ABOVE
    mark by the slippage so any ask in the slippage window crosses.
    """
    s = _settings(MARKET_CLOSE_SLIPPAGE_BPS=100.0)
    client = _make_client(s)
    captured = _capture_place_order_call(client)
    _set_position(client, qty=-30.0, mark=0.9400)

    client.market_close("SUI-PERP")
    assert captured["is_buy"] is True  # closing short ⇒ BUY
    assert captured["sz"] == 30.0
    expected = 0.9400 * (1.0 + 100.0 / 10_000.0)
    assert abs(captured["limit_px"] - expected) < 1e-9


def test_slippage_setting_respected() -> None:
    """Operator can tune the slippage. 50 bps default for a deep-book
    pair (BTC); 200 bps for a venue with wider spreads."""
    s = _settings(MARKET_CLOSE_SLIPPAGE_BPS=200.0)
    client = _make_client(s)
    captured = _capture_place_order_call(client)
    _set_position(client, qty=10.0, mark=1000.0)

    client.market_close("SUI-PERP")
    expected = 1000.0 * (1.0 - 200.0 / 10_000.0)
    assert abs(captured["limit_px"] - expected) < 1e-9


def test_market_close_noop_when_already_flat() -> None:
    """No order should be placed if position is already 0."""
    s = _settings()
    client = _make_client(s)
    captured = _capture_place_order_call(client)
    _set_position(client, qty=0.0, mark=0.9316)

    resp = client.market_close("SUI-PERP")
    assert "noop_already_flat" in str(resp)
    assert captured == {}, "no _place_order call expected on flat position"


def test_market_close_with_explicit_size_uses_that_size() -> None:
    """Operator can pass an explicit size to partially close."""
    s = _settings()
    client = _make_client(s)
    captured = _capture_place_order_call(client)
    _set_position(client, qty=49.0, mark=0.9316)

    client.market_close("SUI-PERP", sz=10.0)
    assert captured["sz"] == 10.0


def test_market_close_handles_zero_or_missing_mark() -> None:
    """Defensive: if mark is 0/None (e.g. fresh-launch market), don't
    crash — fall back to limit_px=0 which the venue clamps to its own
    bounds.
    """
    s = _settings()
    client = _make_client(s)
    captured = _capture_place_order_call(client)
    _set_position(client, qty=49.0, mark=0.0)

    client.market_close("SUI-PERP")
    assert captured["limit_px"] == 0.0


def test_pre_fix_repro_would_be_non_marketable() -> None:
    """snap_20260427_055140 reproducer: long 49, mark 0.9316, market in a
    fast downtrend. With the fix, the SELL limit lands at ~0.9223 which
    crosses any bid above that price (almost always true at ≤22 bps slip
    that the snapshot saw). Pre-fix the limit was 0.9316 — exactly the
    mid — and any bid below 0.9316 left the IOC silently non-marketable.
    """
    s = _settings(MARKET_CLOSE_SLIPPAGE_BPS=100.0)
    client = _make_client(s)
    captured = _capture_place_order_call(client)
    _set_position(client, qty=49.0, mark=0.9316)

    client.market_close("SUI-PERP")
    # The fix: SELL limit is ~93 bps below mark (1% of 0.9316 ≈ 0.0093).
    assert captured["limit_px"] < 0.9316
    # And specifically: limit is at or below the worst bid that the
    # snap_20260427 trough hit (mark went 0.9338 → 0.9316 — a 22 bp drop),
    # so even worse downtrends would still cross.
    assert captured["limit_px"] <= 0.9223 + 1e-9


# ---------------------------------------------------------------------------
# BUG-015: tests on what reaches the WIRE, not what the caller passed
# down. The v1.0.19 fix asserted on _place_order's args, missing the fact
# that _place_order zeroes priceE9 for MARKET orders. Layer the tests
# below on _request so future regressions in the price-construction
# pipeline get caught at the correct boundary.
# ---------------------------------------------------------------------------


def _capture_request_body(client) -> dict:
    """Intercept the HTTP body sent by ``_request`` so tests can assert
    on what the venue actually receives — including the signed
    ``priceE9`` field that BUG-015's misdiagnosed Layer-1 fix overlooked.
    """
    captured: dict = {"calls": []}

    def fake_request(op, method, base, path, *, json_body=None, auth=False, params=None):
        captured["calls"].append({
            "op": op,
            "method": method,
            "path": path,
            "body": json_body,
            "auth": auth,
            "params": params,
        })
        # Return a plausible 2xx response with an order hash so the
        # caller's downstream parsing doesn't fail.
        return {"orderHash": "0x" + "ab" * 32, "_http_status": 200}

    client._request = fake_request  # type: ignore[method-assign]
    return captured


def _set_position_via_real_path(client, qty: float, mark: float) -> None:
    """Make ``fetch_position`` return the synthetic snapshot via the
    same code path market_close uses (so we catch fetch_position itself
    if it ever changes).
    """
    def fake_fetch(addr: str, sym: str):
        from app.models import PositionSnapshot
        return PositionSnapshot(
            symbol=sym,
            position_qty=qty,
            avg_entry_price=mark,
            mark_price=mark,
            position_notional=abs(qty * mark),
            unrealized_pnl_usd=0.0,
        )

    client.fetch_position = fake_fetch  # type: ignore[assignment]


def test_market_close_default_uses_limit_order_type_not_market() -> None:
    """BUG-015 headline: the BUG-014 fix's slippage cap was a no-op
    because ``_place_order`` zeroes ``priceE9`` for MARKET orders. The
    v1.0.20 fix routes through the LIMIT path so the limit actually
    lands in the signed payload.

    Assertion is on the wire body, not on the call into ``_place_order``.
    """
    s = _settings(MARKET_CLOSE_SLIPPAGE_BPS=100.0)
    client = _make_client(s)
    # We need real _place_order so the request body is constructed; only
    # mock the HTTP layer.
    captured = _capture_request_body(client)
    _set_position_via_real_path(client, qty=57.0, mark=0.9287)

    client.market_close("SUI-PERP")

    place_calls = [c for c in captured["calls"] if c["op"] in ("create_order", "market_close")]
    assert len(place_calls) == 1, (
        f"expected one place call; got {[c['op'] for c in captured['calls']]}"
    )
    body = place_calls[0]["body"]
    # The fix: type is LIMIT, not MARKET.
    assert body["type"] == "LIMIT", (
        f"market_close must route through LIMIT to honour priceE9; got type={body['type']}"
    )
    # And: priceE9 is the slippage-adjusted value, NOT '0'.
    expected_limit = 0.9287 * (1.0 - 100.0 / 10_000.0)
    expected_price_e9 = int(round(expected_limit * 1_000_000_000))
    actual_price_e9 = int(body["signedFields"]["priceE9"])
    # Tolerate ±1 LSB in the 1e9 fixed-point conversion (float-precision
    # noise on multiplications like 0.9287 * 0.99 doesn't always round to
    # the nearest integer cleanly).
    assert abs(actual_price_e9 - expected_price_e9) <= 1, (
        f"priceE9={actual_price_e9} but expected ≈{expected_price_e9} "
        "(mark * 0.99 in 1e9 base) — the BUG-015 regression is back: "
        "the slippage cap is being silently zeroed."
    )
    # Anchor: must be a 9-digit number (~mark * 1e9), NOT '0' (which is
    # the BUG-015 failure shape).
    assert actual_price_e9 > 100_000_000, (
        "priceE9 is suspiciously small — looks like the MARKET-zeroing "
        "branch may be back"
    )
    # And: IOC + reduce-only, so semantics are preserved.
    assert body["timeInForce"] == "IOC"
    assert body["reduceOnly"] is True
    assert body["postOnly"] is False
    # And: the call is logged as create_order (LIMIT path) rather than
    # market_close (MARKET path).
    assert place_calls[0]["op"] == "create_order"


def test_market_close_legacy_mode_uses_market_order_type() -> None:
    """Operator escape hatch: ``BLUEFIN_MARKET_CLOSE_USE_LIMIT_IOC=False``
    reverts to the broken-on-SUI-PERP MARKET path. Useful if Bluefin
    fixes the upstream MARKET behaviour or if the LIMIT path develops
    its own quirk on a different venue/symbol combination.
    """
    s = _settings(BLUEFIN_MARKET_CLOSE_USE_LIMIT_IOC=False)
    client = _make_client(s)
    captured = _capture_request_body(client)
    _set_position_via_real_path(client, qty=57.0, mark=0.9287)

    client.market_close("SUI-PERP")

    place_calls = [c for c in captured["calls"] if c["op"] in ("create_order", "market_close")]
    assert len(place_calls) == 1
    body = place_calls[0]["body"]
    assert body["type"] == "MARKET"
    # In legacy MARKET mode, priceE9 is forced to "0" by _place_order.
    # Document this as the EXPECTED behaviour of the legacy path.
    assert body["signedFields"]["priceE9"] == "0"
    assert place_calls[0]["op"] == "market_close"


def test_market_close_limit_ioc_buy_path_for_short_close() -> None:
    """Symmetric: closing a short via BUY puts the limit ABOVE mark."""
    s = _settings(MARKET_CLOSE_SLIPPAGE_BPS=100.0)
    client = _make_client(s)
    captured = _capture_request_body(client)
    _set_position_via_real_path(client, qty=-30.0, mark=0.9400)

    client.market_close("SUI-PERP")

    body = [c for c in captured["calls"] if c["op"] == "create_order"][0]["body"]
    assert body["type"] == "LIMIT"
    assert body["signedFields"]["side"] == "LONG"  # BUY closes short
    expected_limit = 0.9400 * (1.0 + 100.0 / 10_000.0)
    expected_price_e9 = int(round(expected_limit * 1_000_000_000))
    actual_price_e9 = int(body["signedFields"]["priceE9"])
    assert abs(actual_price_e9 - expected_price_e9) <= 1
    assert actual_price_e9 > 100_000_000


def test_market_close_zero_mark_falls_back_to_market_order() -> None:
    """Defensive: if mark is 0 (fresh-launch / data missing), fall back
    to MARKET so the venue's own clamps decide. With the LIMIT-IOC
    default we'd otherwise produce limit=0 which Bluefin would treat
    however it wants — safer to be explicit and use MARKET here.
    """
    s = _settings(MARKET_CLOSE_SLIPPAGE_BPS=100.0)
    client = _make_client(s)
    captured = _capture_request_body(client)
    _set_position_via_real_path(client, qty=10.0, mark=0.0)

    client.market_close("SUI-PERP")

    body = [c for c in captured["calls"] if c["op"] == "market_close"][0]["body"]
    assert body["type"] == "MARKET"
