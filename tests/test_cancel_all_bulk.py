"""Bulk cancel-all endpoint (GRVT) + ``OrderManager`` wrapper fallback.

Covers:

  1. ``GrvtClient.cancel_all_orders_bulk_for_symbol`` posts to the right
     URL with the right body shape for a perp instrument. GRVT's
     ``/full/v1/cancel_all_orders`` takes filter arrays so the cancel
     is scoped to the single traded instrument — other symbols on the
     same sub-account are preserved.

  2. ``OrderManager.cancel_all_orders_for_symbol_bulk_or_fallback`` is
     the production entry point (called from ``app/main.py`` at
     startup and shutdown). Returns a short outcome tag for DB
     telemetry:

       - ``bulk_ok`` — client supports bulk, request succeeded
       - ``bulk_error_fallback_ok`` — bulk raised, per-order loop ran
       - ``fallback_ok`` — client has no bulk method (HL), per-order ran
       - ``no_write_access`` — read-only adapter

     These outcomes drop shutdown cancel latency on GRVT from ~4 s
     (two passes × N+1 REST round trips) to ~300 ms (two passes ×
     one bulk request). The difference is decisive under any
     supervisor's SIGTERM → SIGKILL grace window.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.exchange import grvt_client as grvt_client_module
from app.exchange.grvt_client import GrvtClient
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.enums import Side
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


# --------------------- GrvtClient payload shape ----------------------


_ETH_ROW = {
    "instrument": "ETH_USDT_Perp",
    "tick_size": "0.01",
    "min_size": "0.001",
    "min_notional": "20",
    "base_decimals": 9,
    "instrument_hash": "197633",
}


class _FakeMetaResp:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._payload = {"result": rows}
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeMetaClient:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __enter__(self) -> "_FakeMetaClient":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(self, _url: str, *, json: dict[str, Any]) -> _FakeMetaResp:  # noqa: ARG002
        return _FakeMetaResp(self._rows)


class _CapturingResp:
    def __init__(self) -> None:
        self.status_code = 200
        self.text = ""

    def json(self) -> dict[str, Any]:
        return {"result": {"num_cancelled": 2}}


class _CapturingHttp:
    def __init__(self) -> None:
        self.last_url: str | None = None
        self.last_payload: dict[str, Any] | None = None

    def post(self, url: str, **kwargs: Any) -> _CapturingResp:
        self.last_url = url
        self.last_payload = kwargs.get("json")
        return _CapturingResp()


@pytest.fixture()
def grvt_client(monkeypatch: pytest.MonkeyPatch) -> GrvtClient:
    monkeypatch.setattr(
        grvt_client_module.httpx,
        "Client",
        lambda *_args, **_kwargs: _FakeMetaClient([_ETH_ROW]),
    )
    c = GrvtClient(
        config={
            "symbol": "ETH_USDT_Perp",
            "api_key": "test-api-key",
            "api_secret": "0x" + "11" * 32,
            "sub_account_id": "42",
            "env": "prod",
        }
    )
    c._cookie_gravity = "fake-cookie"
    c._cookie_expiry_epoch = 1e18
    return c


def test_bulk_cancel_posts_to_cancel_all_orders_endpoint(grvt_client: GrvtClient) -> None:
    cap = _CapturingHttp()
    grvt_client._http = cap  # type: ignore[assignment]
    grvt_client.cancel_all_orders_bulk_for_symbol("ETH_USDT_Perp")
    assert cap.last_url is not None
    assert cap.last_url.endswith("/full/v1/cancel_all_orders"), cap.last_url


def test_bulk_cancel_body_carries_symbol_filter_arrays(grvt_client: GrvtClient) -> None:
    """The cancel must be scoped to one instrument: ``kind``, ``base``,
    ``quote`` are arrays selecting exactly ETH_USDT_Perp. Other symbols
    on the same sub-account (e.g. BTC_USDT_Perp) must not be touched."""
    cap = _CapturingHttp()
    grvt_client._http = cap  # type: ignore[assignment]
    grvt_client.cancel_all_orders_bulk_for_symbol("ETH_USDT_Perp")
    payload = cap.last_payload or {}
    assert payload.get("sub_account_id") == "42"
    assert payload.get("kind") == ["PERPETUAL"]
    assert payload.get("base") == ["ETH"]
    assert payload.get("quote") == ["USDT"]


def test_bulk_cancel_accepts_bare_symbol_form(grvt_client: GrvtClient) -> None:
    """Execution passes ``settings.symbol`` which may be the bare ``"ETH"``
    (our config form) rather than ``"ETH_USDT_Perp"``. The client must
    normalise the same way ``place_post_only_limit`` does."""
    cap = _CapturingHttp()
    grvt_client._http = cap  # type: ignore[assignment]
    grvt_client.cancel_all_orders_bulk_for_symbol("ETH")
    payload = cap.last_payload or {}
    # Bare "ETH" is normalised to ETH_USDT_Perp then split.
    assert payload.get("base") == ["ETH"]
    assert payload.get("quote") == ["USDT"]
    assert payload.get("kind") == ["PERPETUAL"]


# --------------------- OrderManager wrapper outcomes -----------------


def _ooraw(oid: int, coin: str, side: Side = Side.BUY) -> HLOpenOrderRaw:
    return HLOpenOrderRaw(
        oid=oid, coin=coin, side=side, limit_px=100.0, sz=1.0,
        timestamp=0, cloid=None,
    )


def _make_order_manager(client: MagicMock, *, symbol: str = "ETH") -> tuple[OrderManager, Storage, BotState]:
    path = Path(tempfile.gettempdir()) / f"mm_bulk_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": symbol,
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "PRIVATE_WS_ENABLED": False,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    om = OrderManager(settings, client, storage, state, private_event_queue=None)
    return om, storage, state


def test_wrapper_takes_bulk_path_when_client_supports_it() -> None:
    """Auto-generated MagicMock attribute means a test double satisfies
    the ``hasattr`` check. The wrapper calls the bulk method once with
    the configured symbol, never falls back to fetch_open_orders."""
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.cancel_all_orders_bulk_for_symbol = MagicMock(return_value={"result": {"num_cancelled": 3}})
    om, _storage, _state = _make_order_manager(client, symbol="ETH")

    outcome = om.cancel_all_orders_for_symbol_bulk_or_fallback()

    assert outcome == "bulk_ok"
    client.cancel_all_orders_bulk_for_symbol.assert_called_once_with("ETH")
    client.fetch_open_orders_raw.assert_not_called()
    client.cancel_order.assert_not_called()


def test_wrapper_falls_back_to_per_order_loop_when_bulk_raises() -> None:
    """v1.4.55 wedge-elimination Phase 1 update: bulk endpoint raises
    → fall back to ``cancel_all_orders_for_symbol``, which now reads
    from LOCAL working-order state (not the exchange snapshot) and
    routes cancels through the dispatcher's
    ``_enqueue_cancel_quote_path``. Verify:
      * outcome tag is ``bulk_error_fallback_ok``
      * bulk was attempted exactly once
      * the fallback ``cancel_all_orders_for_symbol`` ran (spy it)
      * the bulk exception is counted as an execution error
    The per-order ``cancel_order`` HTTP call no longer fires
    directly from the fallback path — the dispatcher handles it
    asynchronously, so we don't assert on it here.
    """
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.cancel_all_orders_bulk_for_symbol = MagicMock(side_effect=RuntimeError("upstream 500"))
    om, _storage, state = _make_order_manager(client, symbol="ETH")

    # Spy on the fallback method to verify it was called.
    fallback_calls = {"n": 0}
    original_fallback = om.cancel_all_orders_for_symbol
    def _spy() -> None:
        fallback_calls["n"] += 1
        return original_fallback()
    om.cancel_all_orders_for_symbol = _spy  # type: ignore[method-assign]

    outcome = om.cancel_all_orders_for_symbol_bulk_or_fallback()

    assert outcome == "bulk_error_fallback_ok"
    client.cancel_all_orders_bulk_for_symbol.assert_called_once()
    assert fallback_calls["n"] == 1, (
        "v1.4.55 Phase 1: when bulk raises, the wrapper must invoke "
        "``cancel_all_orders_for_symbol`` as the fallback. Got "
        f"{fallback_calls['n']} calls."
    )
    # Bulk failure is a counted execution error.
    assert state.execution_errors >= 1


def test_wrapper_uses_fallback_when_client_has_no_bulk_method() -> None:
    """v1.4.55 Phase 1 update: client has no bulk method → wrapper
    directly invokes ``cancel_all_orders_for_symbol`` (no bulk
    attempt, no exception bookkeeping). With Phase 1, that fallback
    reads from local WO state and routes through the dispatcher; we
    spy on the call itself rather than the now-async per-order HTTP.
    """
    client = mock_mm_client()
    client.has_write_access.return_value = True
    # Force attribute absence — MagicMock auto-generates otherwise.
    del client.cancel_all_orders_bulk_for_symbol
    om, _storage, state = _make_order_manager(client, symbol="ETH")

    fallback_calls = {"n": 0}
    original_fallback = om.cancel_all_orders_for_symbol
    def _spy() -> None:
        fallback_calls["n"] += 1
        return original_fallback()
    om.cancel_all_orders_for_symbol = _spy  # type: ignore[method-assign]

    outcome = om.cancel_all_orders_for_symbol_bulk_or_fallback()

    assert outcome == "fallback_ok"
    assert fallback_calls["n"] == 1, (
        "v1.4.55 Phase 1: when bulk is absent, the wrapper must "
        "directly invoke ``cancel_all_orders_for_symbol``. Got "
        f"{fallback_calls['n']} calls."
    )
    # Clean fallback is not an error.
    assert state.execution_errors == 0


def test_wrapper_is_noop_when_client_is_read_only() -> None:
    """``has_write_access=False`` (read-only credentials, or dry-run)
    must short-circuit with no network calls at all — bulk or per-order."""
    client = mock_mm_client()
    client.has_write_access.return_value = False
    client.cancel_all_orders_bulk_for_symbol = MagicMock()
    om, _storage, _state = _make_order_manager(client, symbol="ETH")

    outcome = om.cancel_all_orders_for_symbol_bulk_or_fallback()

    assert outcome == "no_write_access"
    client.cancel_all_orders_bulk_for_symbol.assert_not_called()
    client.fetch_open_orders_raw.assert_not_called()
    client.cancel_order.assert_not_called()


# --------------------- Shutdown DB event contract --------------------
#
# ``app/main.py`` writes a ``cancel_all_on_shutdown_executed`` bot_event
# row immediately after the two-pass bulk-or-fallback cancel. Without
# this event, ``trading.db`` post-mortem could not distinguish between
# "shutdown ran cancel_all cleanly" vs. "SIGKILL truncated before
# cancel ever fired" — both paths produce only the generic
# ``service stopping`` event at end-of-lifespan.
#
# We mirror the production snippet here rather than spin up a full
# FastAPI lifespan: same shape, fewer moving parts.


def test_shutdown_snippet_writes_cancel_all_on_shutdown_event() -> None:
    """Mirror of ``app/main.py`` shutdown: two passes of bulk-or-fallback
    cancel, then one DB event carrying the outcomes for both passes."""
    from app.enums import EventSeverity
    from app.utils.time import utc_now_iso
    import time as _time

    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.cancel_all_orders_bulk_for_symbol = MagicMock(return_value={"result": {}})
    om, storage, _state = _make_order_manager(client, symbol="ETH")

    # Production snippet — identical control flow to app/main.py.
    pass_outcomes: list[str] = []
    for pass_idx in range(2):
        pass_outcomes.append(om.cancel_all_orders_for_symbol_bulk_or_fallback())
        if pass_idx == 0:
            _time.sleep(0.0)  # no real settle in tests
    storage.insert_bot_event(
        utc_now_iso(),
        EventSeverity.INFO.value,
        "cancel_all_on_shutdown_executed",
        f"cancel_all_on_shutdown executed (symbol-scoped, outcomes={pass_outcomes})",
        {"symbol": "ETH", "pass_outcomes": pass_outcomes},
    )

    events = storage.recent_bot_events(limit=50)
    matches = [e for e in events if e["event_type"] == "cancel_all_on_shutdown_executed"]
    assert len(matches) == 1, f"expected 1 shutdown-cancel event, got {len(matches)}"
    import json as _json
    payload = _json.loads(matches[0]["payload_json"])
    assert payload["symbol"] == "ETH"
    assert payload["pass_outcomes"] == ["bulk_ok", "bulk_ok"]
    # The bulk path was taken on BOTH passes.
    assert client.cancel_all_orders_bulk_for_symbol.call_count == 2
