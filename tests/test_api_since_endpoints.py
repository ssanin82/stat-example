"""Integration tests for the Phase-1 observability ``/since`` endpoints.

Covers the 5 new endpoints:

* ``GET /orders/since``
* ``GET /events/since``
* ``GET /quotes/since``
* ``GET /inventory/since``
* ``GET /equity/since``

Plus the ``until`` query param added to ``/fills/since`` for bounded
windows.

Each endpoint:

* Defaults ``since`` to the current session start.
* Accepts optional ``until`` (exclusive upper bound).
* Returns rows ordered by timestamp ascending.
* Clamps ``limit`` to a safe maximum.
* Filters by ``symbol`` where the table has one.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _client(session_start_iso: str = "2026-04-18T12:00:00+00:00") -> tuple[TestClient, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_since_api_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    # Pin session_started_at_utc so the default ``since`` is predictable
    # in tests — the endpoints fall back to this when ``since`` is omitted.
    state.session_started_at_utc = session_start_iso
    app = FastAPI()
    app.include_router(router)
    app.state.settings = settings
    app.state.bot_state = state
    app.state.storage = storage
    return TestClient(app), storage


def _ts(hours: int = 0, minutes: int = 0, seconds: int = 0) -> str:
    return f"2026-04-18T{hours:02d}:{minutes:02d}:{seconds:02d}+00:00"


# --------------------- /orders/since ---------------------------------


def _insert_order(storage: Storage, **fields) -> None:
    row = {
        "order_id_local": uuid.uuid4().hex[:16],
        "order_id_exchange": None,
        "client_order_id": None,
        "ts_created": fields["ts_created"],
        "ts_sent": None,
        "ts_ack": None,
        "ts_closed": None,
        "symbol": fields.get("symbol", "ETH_USDT_Perp"),
        "side": fields.get("side", "BUY"),
        "price": 1.0,
        "size": 1.0,
        "post_only": 1,
        "status": "ACKED",
        "cancel_reason": None,
        "replace_group_id": None,
        "quote_cycle_id": None,
    }
    storage.insert_order_row(row)


def test_orders_since_default_uses_session_start() -> None:
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    _insert_order(storage, ts_created=_ts(11, 0, 0))   # pre-session (excluded)
    _insert_order(storage, ts_created=_ts(12, 30, 0))  # in-session
    _insert_order(storage, ts_created=_ts(13, 0, 0))   # in-session

    r = client.get("/orders/since")
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 2
    assert [row["ts_created"] for row in rows] == [_ts(12, 30, 0), _ts(13, 0, 0)]


def test_orders_since_explicit_window() -> None:
    client, storage = _client()
    for t in (_ts(12, 0, 0), _ts(13, 0, 0), _ts(14, 0, 0), _ts(15, 0, 0)):
        _insert_order(storage, ts_created=t)
    r = client.get(
        "/orders/since",
        params={"since": _ts(12, 30, 0), "until": _ts(14, 30, 0)},
    )
    assert r.status_code == 200
    tss = [row["ts_created"] for row in r.json()]
    assert tss == [_ts(13, 0, 0), _ts(14, 0, 0)]


def test_orders_since_symbol_filter() -> None:
    client, storage = _client()
    _insert_order(storage, ts_created=_ts(12, 30, 0), symbol="ETH_USDT_Perp")
    _insert_order(storage, ts_created=_ts(12, 30, 5), symbol="AXS_USDT_Perp")
    _insert_order(storage, ts_created=_ts(12, 30, 10), symbol="ETH_USDT_Perp")

    r = client.get("/orders/since", params={"symbol": "ETH_USDT_Perp"})
    assert r.status_code == 200
    rows = r.json()
    assert all(row["symbol"] == "ETH_USDT_Perp" for row in rows)
    assert len(rows) == 2


def test_orders_since_limit_clamped_to_max() -> None:
    client, _storage = _client()
    r = client.get("/orders/since", params={"limit": 999999})
    assert r.status_code == 200
    # Handler clamps limit to [1, 50000] via _clamp_limit; empty result
    # is fine — the assertion is that no error is raised.


# --------------------- /events/since ---------------------------------


def test_events_since_default_uses_session_start() -> None:
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    storage.insert_bot_event(_ts(11, 59, 0), "INFO", "pre", "", None)
    storage.insert_bot_event(_ts(12, 0, 30), "INFO", "in1", "", None)
    storage.insert_bot_event(_ts(13, 0, 0), "WARNING", "in2", "", None)

    r = client.get("/events/since")
    assert r.status_code == 200
    rows = r.json()
    # Pre-session event excluded.
    assert [row["event_type"] for row in rows] == ["in1", "in2"]


def test_events_since_until_exclusive() -> None:
    client, storage = _client()
    for (stamp, name) in (
        (_ts(12, 0, 0), "a"),
        (_ts(12, 30, 0), "b"),
        (_ts(13, 0, 0), "c"),
    ):
        storage.insert_bot_event(stamp, "INFO", name, "", None)
    r = client.get("/events/since", params={"since": _ts(12, 0, 0), "until": _ts(13, 0, 0)})
    assert r.status_code == 200
    events = [row["event_type"] for row in r.json()]
    assert events == ["a", "b"]  # "c" (at 13:00:00) excluded.


# --------------------- /quotes/since ---------------------------------


def _insert_quote(storage: Storage, *, ts: str, symbol: str = "ETH_USDT_Perp") -> None:
    with storage._lock:
        with storage.connection() as conn:
            conn.execute(
                "INSERT INTO quote_decisions (ts, symbol, mid_price, quote_cycle_id) "
                "VALUES (?, ?, ?, ?)",
                (ts, symbol, 100.0, uuid.uuid4().hex),
            )


def test_quotes_since_default_and_ordering() -> None:
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    _insert_quote(storage, ts=_ts(11, 59, 0))  # pre-session
    _insert_quote(storage, ts=_ts(12, 0, 2))
    _insert_quote(storage, ts=_ts(12, 0, 0))
    _insert_quote(storage, ts=_ts(12, 0, 1))

    r = client.get("/quotes/since")
    assert r.status_code == 200
    tss = [row["ts"] for row in r.json()]
    assert tss == [_ts(12, 0, 0), _ts(12, 0, 1), _ts(12, 0, 2)]


def test_quotes_since_symbol_filter() -> None:
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    _insert_quote(storage, ts=_ts(12, 0, 1), symbol="ETH_USDT_Perp")
    _insert_quote(storage, ts=_ts(12, 0, 2), symbol="AXS_USDT_Perp")
    _insert_quote(storage, ts=_ts(12, 0, 3), symbol="ETH_USDT_Perp")
    r = client.get("/quotes/since", params={"symbol": "AXS_USDT_Perp"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "AXS_USDT_Perp"


# --------------------- /inventory/since ------------------------------


def _insert_position(storage: Storage, *, ts: str, qty: float, symbol: str = "ETH_USDT_Perp") -> None:
    with storage._lock:
        with storage.connection() as conn:
            conn.execute(
                "INSERT INTO position_snapshots (ts, symbol, position_qty, avg_entry_price, "
                "mark_price, position_notional, unrealized_pnl_usd) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, symbol, qty, None, 100.0, qty * 100.0, 0.0),
            )


def test_inventory_since_default_and_ordering() -> None:
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    _insert_position(storage, ts=_ts(12, 0, 10), qty=5.0)
    _insert_position(storage, ts=_ts(12, 0, 5), qty=3.0)
    _insert_position(storage, ts=_ts(12, 0, 15), qty=8.0)

    r = client.get("/inventory/since")
    assert r.status_code == 200
    rows = r.json()
    assert [row["position_qty"] for row in rows] == [3.0, 5.0, 8.0]


def test_inventory_since_symbol_filter() -> None:
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    _insert_position(storage, ts=_ts(12, 0, 1), qty=5.0, symbol="ETH_USDT_Perp")
    _insert_position(storage, ts=_ts(12, 0, 2), qty=50.0, symbol="AXS_USDT_Perp")
    r = client.get("/inventory/since", params={"symbol": "AXS_USDT_Perp"})
    assert r.status_code == 200
    rows = r.json()
    assert len(rows) == 1
    assert rows[0]["position_qty"] == 50.0


# --------------------- /equity/since ---------------------------------


def _insert_equity(storage: Storage, *, ts: str, equity: float) -> None:
    with storage._lock:
        with storage.connection() as conn:
            conn.execute(
                "INSERT INTO equity_snapshots (ts, equity_usd, cash_usd, realized_pnl_usd, "
                "unrealized_pnl_usd, fees_usd, drawdown_usd) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, equity, equity, 0.0, 0.0, 0.0, 0.0),
            )


def test_equity_since_default_and_ordering() -> None:
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    _insert_equity(storage, ts=_ts(12, 0, 0), equity=100.0)
    _insert_equity(storage, ts=_ts(12, 0, 30), equity=101.0)
    _insert_equity(storage, ts=_ts(12, 1, 0), equity=102.0)

    r = client.get("/equity/since")
    assert r.status_code == 200
    rows = r.json()
    assert [row["equity_usd"] for row in rows] == [100.0, 101.0, 102.0]


def test_equity_since_until_exclusive() -> None:
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    _insert_equity(storage, ts=_ts(12, 0, 0), equity=100.0)
    _insert_equity(storage, ts=_ts(12, 0, 30), equity=101.0)
    _insert_equity(storage, ts=_ts(12, 1, 0), equity=102.0)
    r = client.get("/equity/since", params={"until": _ts(12, 1, 0)})
    assert r.status_code == 200
    equities = [row["equity_usd"] for row in r.json()]
    assert equities == [100.0, 101.0]


# --------------------- /fills/since until param ---------------------


def test_fills_since_until_param() -> None:
    """The pre-existing ``/fills/since`` endpoint now accepts ``until``.

    Backward-compat: the param is optional — callers that don't pass
    it behave exactly as before (entire ``[since, now]`` window).
    """
    client, storage = _client(session_start_iso=_ts(12, 0, 0))
    stamps = [_ts(12, 0, 0), _ts(12, 0, 30), _ts(12, 1, 0)]
    for stamp in stamps:
        storage.insert_fill_row(
            {
                "fill_id": uuid.uuid4().hex,
                "order_id_exchange": None,
                "client_order_id": None,
                "ts_fill": stamp,
                "symbol": "ETH_USDT_Perp",
                "side": "BUY",
                "price": 1.0,
                "size": 1.0,
                "notional": 1.0,
                "fee": 0.0,
                "liquidity_flag": "resting",
                "mid_at_fill": 1.0,
                "best_bid_at_fill": 0.999,
                "best_ask_at_fill": 1.001,
                "book_snapshot_quality": "full",
                "markout_1s_bps": 0.0,
                "markout_3s_bps": 0.0,
                "markout_5s_bps": 0.0,
                "book_reference_quality": "exact_or_prior",
            }
        )

    r_all = client.get("/fills/since")
    assert r_all.status_code == 200
    assert len(r_all.json()) == 3

    r_window = client.get("/fills/since", params={"until": _ts(12, 1, 0)})
    assert r_window.status_code == 200
    rows = r_window.json()
    # 12:01:00 excluded
    assert len(rows) == 2
    assert [row["ts_fill"] for row in rows] == [_ts(12, 0, 0), _ts(12, 0, 30)]
