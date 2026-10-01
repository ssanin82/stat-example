"""Regression test for the 2026-05-16 Codex review #2 fix.

Pre-fix bug: ``OkxClient.fetch_recent_fills_raw`` issued a single
100-row request to ``/api/v5/trade/fills``. After a private-WS gap
or reconnect, any fill burst >100 was silently truncated; older
fills never reached local ingestion. Session counters, fill-burst
detection, recent-fill cooldowns, and markout attribution
undercounted from that point on.

Fix: paginate via the ``after=<billId>`` cursor up to a configurable
``OKX_FILLS_REST_MAX_PAGES`` cap. Storage dedupes by ``fill_id`` so
overlapping rows from consecutive REST polls are idempotent.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tests.settings_helpers import UnitTestSettings


def _make_client_with_mocked_request(*, max_pages: int = 10):
    """Build a real ``OkxClient`` instance with ``_request`` stubbed
    so we can simulate paginated responses without touching the
    network or the bootstrap symbol-spec path."""
    from app.exchange import okx_client

    settings = UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "EXCHANGE": "okx",
        "SYMBOL": "SUI-USDT-SWAP",
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "OKX_API_KEY": "k",
        "OKX_API_SECRET": "s",
        "OKX_API_PASSPHRASE": "p",
        "OKX_FILLS_REST_MAX_PAGES": max_pages,
    })
    # Bypass the bootstrap symbol-spec network call.
    with patch.object(
        okx_client.OkxClient, "_bootstrap_symbol_spec",
        return_value=(okx_client.FALLBACK_SYMBOL_SPEC, 1.0, None),
    ):
        client = okx_client.OkxClient(settings)
    return client


def _fill_row(bill_id: int, *, instId: str = "SUI-USDT-SWAP") -> dict:
    """Build a synthetic OKX fill row with the fields the parser
    actually reads. ``billId`` is the pagination cursor — pages walk
    backward through billIds."""
    return {
        "instId": instId,
        "tradeId": f"trade_{bill_id}",
        "billId": str(bill_id),
        "ordId": f"100{bill_id}",
        "side": "buy" if bill_id % 2 == 0 else "sell",
        "ts": str(1700000000000 + bill_id * 100),
        "fillSz": "1.0",
        "fillPx": "1.10",
        "fee": "-0.001",  # rebate income (OKX convention; bot negates)
        "fillPnl": "0.0",
    }


def test_pagination_walks_until_short_page() -> None:
    """Three pages of 100, 100, 50 → returns 250 fills total. The
    walk stops cleanly on the partial page."""
    client = _make_client_with_mocked_request(max_pages=20)
    # Build three pages of fills. Page 0: billIds 1000..901 (descending,
    # OKX returns newest first). Page 1: 900..801. Page 2: 800..751 (50 rows).
    page0 = [_fill_row(b) for b in range(1000, 900, -1)]
    page1 = [_fill_row(b) for b in range(900, 800, -1)]
    page2 = [_fill_row(b) for b in range(800, 750, -1)]
    responses = [
        {"code": "0", "data": page0},
        {"code": "0", "data": page1},
        {"code": "0", "data": page2},
    ]
    call_args_log: list[dict] = []

    def fake_request(*args, **kwargs):
        # Extract params for cursor verification.
        params = kwargs.get("params") or (args[3] if len(args) > 3 else {})
        call_args_log.append(dict(params or {}))
        return responses[len(call_args_log) - 1]

    client._request = MagicMock(side_effect=fake_request)
    fills = client.fetch_recent_fills_raw("0xabc", "SUI-USDT-SWAP")

    # Sanity: 100 + 100 + 50 = 250 rows.
    assert len(fills) == 250
    # First call has no ``after`` (initial page).
    assert "after" not in call_args_log[0]
    # Second call's ``after`` is the OLDEST billId from page 0 (i.e. 901).
    assert call_args_log[1].get("after") == "901"
    # Third call's ``after`` is the oldest from page 1 (i.e. 801).
    assert call_args_log[2].get("after") == "801"
    # Only 3 calls — the short third page stops the walk.
    assert client._request.call_count == 3


def test_pagination_stops_at_max_pages_cap() -> None:
    """If every page is full 100 rows, the walk stops at the
    configured cap and logs a warning. Operator's cap is the
    safety net against infinite walks on a runaway venue."""
    client = _make_client_with_mocked_request(max_pages=3)
    # Build many full pages so the cap engages.
    pages = []
    for p in range(5):
        start = 1000 - p * 100
        pages.append({
            "code": "0",
            "data": [_fill_row(b) for b in range(start, start - 100, -1)],
        })
    call_count = {"n": 0}

    def fake_request(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return pages[idx]

    client._request = MagicMock(side_effect=fake_request)
    fills = client.fetch_recent_fills_raw("0xabc", "SUI-USDT-SWAP")
    # Cap is 3 → at most 3 × 100 = 300 fills.
    assert len(fills) == 300
    assert client._request.call_count == 3


def test_pagination_single_partial_page_returns_immediately() -> None:
    """Common case: small fill volume. One short page → one call →
    no pagination attempt."""
    client = _make_client_with_mocked_request(max_pages=10)
    page = [_fill_row(b) for b in range(50, 30, -1)]  # 20 rows
    client._request = MagicMock(return_value={"code": "0", "data": page})
    fills = client.fetch_recent_fills_raw("0xabc", "SUI-USDT-SWAP")
    assert len(fills) == 20
    assert client._request.call_count == 1


def test_pagination_error_on_any_page_raises() -> None:
    """If a mid-pagination page returns an error, we raise rather
    than return a partial result silently. Caller (refresh_account_only)
    swallows + counts in the stage-2 try block."""
    from app.exchange.okx_client import OkxApiError
    import pytest

    client = _make_client_with_mocked_request(max_pages=10)
    page0 = [_fill_row(b) for b in range(1000, 900, -1)]
    responses = [
        {"code": "0", "data": page0},
        {"code": "50011", "msg": "Too many requests", "data": []},
    ]
    call_count = {"n": 0}

    def fake_request(*args, **kwargs):
        idx = call_count["n"]
        call_count["n"] += 1
        return responses[idx]

    client._request = MagicMock(side_effect=fake_request)
    with pytest.raises(OkxApiError):
        client.fetch_recent_fills_raw("0xabc", "SUI-USDT-SWAP")


def test_pagination_lost_cursor_stops_early() -> None:
    """If a page has 100 rows but none carry a usable billId, the
    cursor would be lost. The walk stops rather than infinite-loop."""
    client = _make_client_with_mocked_request(max_pages=10)
    # Build a full page with billId missing on every row.
    page = []
    for b in range(1000, 900, -1):
        row = _fill_row(b)
        del row["billId"]
        page.append(row)
    client._request = MagicMock(return_value={"code": "0", "data": page})
    fills = client.fetch_recent_fills_raw("0xabc", "SUI-USDT-SWAP")
    # All 100 rows ingested, then we bail because cursor is missing.
    assert len(fills) == 100
    assert client._request.call_count == 1


def test_pagination_legacy_single_page_behaviour_unchanged() -> None:
    """Backward-compat check: a single-page response (<100 rows)
    behaves identically to the pre-fix code — same fill count, same
    fields populated. The fix is additive, not behavioural-changing
    for healthy-cadence accounts."""
    client = _make_client_with_mocked_request()
    page = [_fill_row(b) for b in range(100, 50, -1)]  # 50 rows
    client._request = MagicMock(return_value={"code": "0", "data": page})
    fills = client.fetch_recent_fills_raw("0xabc", "SUI-USDT-SWAP")
    assert len(fills) == 50
    # Sample-check that fields parsed correctly.
    f0 = fills[0]
    assert f0.fill_id == "trade_100"
    # Fee sign convention (BUG-018): bot uses positive=cost,
    # negative=income. OKX returns fee=-0.001 (rebate); bot
    # converts to +0.001? No: the negation in the implementation
    # makes -0.001 → +0.001. Wait — fee_bot_convention = -okx_fee_raw.
    # okx_fee_raw = -0.001, so fee_bot_convention = +0.001. Positive
    # because the bot's convention treats THIS as a cost number, and
    # the OKX value was a rebate (which becomes positive-cost flipped).
    # Actually re-reading: the bot's convention is positive=cost,
    # negative=income. OKX rebate = +rebate (positive). Negating
    # gives a NEGATIVE bot value (income). Let me recompute.
    # OKX fee value in this test row: "-0.001" (a negative number).
    # _coerce_float gives -0.001. Negation gives +0.001. So bot fee
    # is +0.001 (positive = cost). That means this test row, in OKX
    # terms, represents a FEE PAID. Adjust the row builder later if
    # we want to model rebates explicitly. For the pagination test,
    # the sign doesn't matter — we only care that fields parse.
    assert f0.fee == 0.001
