"""v1.4.93 wedge-elimination-cleanup Phase 6B —
stateful mock OKX client for full-flow integration tests.

Implements the ``PerpExchangeAdapter`` Protocol with simulated state
+ configurable timing / failure injection. Used by
``tests/test_phase6b_full_flow_scenarios.py`` to exercise the bot's
real `maybe_refresh_quotes` against deterministic venue behavior.

**What's stateful:**
* Tracks placed orders by ``ordId`` and ``clOrdId`` with their
  lifecycle state (live / canceled / filled).
* Records every place / cancel call so tests can assert on call
  sequences.
* Supports cancel-by-oid AND cancel-by-cloid lookup paths.

**What's configurable:**
* `place_response_outcome` — accepted / exchange_rejected /
  transport_rejected per call (override via context manager or
  per-call key).
* `cancel_response_code` — success / 51410 (gone) / 51604
  (post-only cross) / 51603 (transport) per cancel call.
* `place_latency_ms` — simulated wait inside `place_post_only_limit`
  before returning. Real wall-clock sleep so timing-sensitive
  scenarios behave naturally.

**WS event injection:**
* `emit_ws_live(ordId)` / `emit_ws_canceled(ordId)` / `emit_ws_filled(ordId, fill_qty, fill_px)` —
  push events to the bot's `private_event_queue` with the same shape
  the OKX adapter parses.
* `schedule_ws_event(ordId, kind, after_ms)` — for time-sensitive
  scenarios where the WS event must arrive after a specific delay
  relative to the place response.

This file is the FOUNDATION of Phase 6B. The 9 scenarios in
``test_phase6b_full_flow_scenarios.py`` each configure the client's
state + timing to drive a specific failure mode.

NOTE: This client speaks the OKX wire shape (sCode/sMsg, ordId/clOrdId,
batch responses). The bot's OKX adapter (`app/exchange/okx_client.py`)
parses these via `interpret_*_response`. We expose interpret_* methods
that match.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from unittest.mock import MagicMock

from app.enums import Side
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC, SymbolSpec
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot


# ---------------------------------------------------------------------------
# Mock-side order state
# ---------------------------------------------------------------------------


class MockOrderState(str, Enum):
    LIVE = "live"
    CANCELED = "canceled"
    FILLED = "filled"
    PARTIAL = "partial_fill"


@dataclass
class MockOrder:
    """Order as it exists on the mock 'venue'. Distinct from the
    bot's WorkingOrder — this is what the venue knows."""

    ord_id: int
    cl_ord_id: str
    side: Side
    price: float
    size: float
    placed_at: datetime
    state: MockOrderState = MockOrderState.LIVE
    canceled_at: Optional[datetime] = None
    filled_qty: float = 0.0
    filled_px: Optional[float] = None
    filled_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Configurable failure injection
# ---------------------------------------------------------------------------


@dataclass
class MockClientConfig:
    """Per-scenario tuning. Mutate before / between calls."""

    # Place-response behavior.
    place_response_outcome: str = "accepted"  # accepted | exchange_rejected | transport_rejected
    place_reject_scode: str = "50011"  # used when outcome != "accepted"
    place_reject_smsg: str = "Rate limit reached"
    place_latency_ms: float = 0.0  # simulated wait before returning

    # Cancel-response behavior. None = success.
    # Common OKX cancel reject codes:
    #   51410 — order does not exist (benign — already gone)
    #   51400 — order canceled / filled / does not exist (similar)
    #   51604 — post-only crossing (different code class, but seen)
    #   51603 — transport issue
    cancel_response_scode: Optional[str] = None
    cancel_response_smsg: str = ""
    # When True, the second consecutive cancel returns a DIFFERENT code.
    # Useful for scenarios like "first cancel 51410, retry succeeds".
    cancel_response_oneshot: bool = False
    cancel_response_oneshot_used: bool = False


# ---------------------------------------------------------------------------
# WS event injector
# ---------------------------------------------------------------------------


def _build_ws_order_event(
    mock_order: MockOrder,
    *,
    new_state: MockOrderState,
    symbol: str = "TON-USDT-SWAP",
) -> dict[str, Any]:
    """Construct an OKX-shaped order WS event for the given state
    transition. Matches the parser in ``app/exchange/okx_public_ws.py``
    (or the private-order parser — the bot's OrderManager consumes
    events from `private_event_queue` shaped like this).
    """
    state_str = {
        MockOrderState.LIVE: "live",
        MockOrderState.CANCELED: "canceled",
        MockOrderState.FILLED: "filled",
        MockOrderState.PARTIAL: "partially_filled",
    }[new_state]
    return {
        "ordId": str(mock_order.ord_id),
        "clOrdId": mock_order.cl_ord_id,
        "instId": symbol,
        "side": mock_order.side.value.lower(),
        "px": str(mock_order.price),
        "sz": str(mock_order.size),
        "accFillSz": str(mock_order.filled_qty),
        "state": state_str,
        "uTime": str(int(time.time() * 1000)),
        "cTime": str(int(mock_order.placed_at.timestamp() * 1000)),
        "ordType": "post_only",
    }


# ---------------------------------------------------------------------------
# The mock client itself
# ---------------------------------------------------------------------------


class MockOkxClient:
    """Stateful PerpExchangeAdapter-compatible mock.

    Construct with a ``threading.Lock``-protected order book. Tests
    drive scenarios by:

    1. Configuring ``self.config`` to choose response behaviors.
    2. Calling `place_post_only_limit` / `cancel_order` through the
       bot's normal code path.
    3. Inspecting `self.order_book` to assert state.
    4. Optionally calling `emit_ws_*` to inject WS events into the
       provided `private_event_queue`.
    """

    def __init__(
        self,
        symbol_spec: Optional[SymbolSpec] = None,
        symbol: str = "TON-USDT-SWAP",
    ) -> None:
        self._symbol_spec = symbol_spec or FALLBACK_SYMBOL_SPEC
        self.symbol_spec_fetched_ok = True
        self.symbol = symbol

        self.config = MockClientConfig()

        # Order book: ordId → MockOrder. Lock-protected because the
        # bot may concurrently place / cancel from multiple threads.
        self._lock = threading.Lock()
        self.order_book: dict[int, MockOrder] = {}
        self._next_ord_id = 1_000_000_000_000_000_000  # 19-digit OKX-shaped

        # Call history — every method invocation appended.
        self.calls: list[tuple[str, dict[str, Any]]] = []

        # The bot's private_event_queue. Set by harness via
        # `attach_event_queue` so emit_ws_* can push events.
        self._event_queue: Any = None

        # Cloid → ordId for fast cancel-by-cloid lookup.
        self._cloid_to_ord_id: dict[str, int] = {}

    # ----- harness wiring ----------------------------------------------------

    def attach_event_queue(self, q: Any) -> None:
        """Attach the bot's private_event_queue so emit_ws_* can push."""
        self._event_queue = q

    # ----- PerpExchangeAdapter Protocol surface ------------------------------

    @property
    def symbol_spec(self) -> SymbolSpec:
        return self._symbol_spec

    def has_write_access(self) -> bool:
        return True

    def fetch_best_bid_ask(self, symbol: str) -> BestBidAsk:
        self.calls.append(("fetch_best_bid_ask", {"symbol": symbol}))
        return BestBidAsk(
            symbol=symbol,
            best_bid=2.000,
            best_ask=2.002,
            mid_price=2.001,
            spread_bps=10.0,
            ts_local=datetime.now(timezone.utc),
        )

    def fetch_position(self, address: str, symbol: str) -> PositionSnapshot:
        return PositionSnapshot(
            symbol=symbol,
            position_qty=0.0,
            avg_entry_price=None,
            mark_price=None,
            position_notional=0.0,
            unrealized_pnl_usd=0.0,
        )

    def fetch_account_snapshot(self, address: str) -> AccountSnapshot:
        # Minimal: just a sentinel with non-zero equity. Tests rarely
        # care about the specific shape.
        return AccountSnapshot(  # type: ignore[call-arg]
            address=address,
            account_value_usd=1000.0,
            ts_local=datetime.now(timezone.utc),
        )

    def fetch_open_orders_raw(self, address: str) -> list[Any]:
        # Returns currently-live orders in OKX shape.
        with self._lock:
            return [
                {
                    "ordId": str(o.ord_id),
                    "clOrdId": o.cl_ord_id,
                    "side": o.side.value.lower(),
                    "px": str(o.price),
                    "sz": str(o.size),
                    "instId": self.symbol,
                    "state": "live",
                    "ordType": "post_only",
                }
                for o in self.order_book.values()
                if o.state == MockOrderState.LIVE
            ]

    def fetch_recent_fills_raw(self, address: str, symbol: str) -> list[Any]:
        return []

    def place_post_only_limit(
        self,
        symbol: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        *,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]:
        """Stateful place. Records the call. If
        ``config.place_response_outcome == "accepted"``, creates an
        order on the mock book. Otherwise returns the configured
        reject response.

        Returns an OKX-shaped response.
        """
        cl_id = client_order_id or f"mockcl{uuid.uuid4().hex[:24]}"
        self.calls.append((
            "place_post_only_limit",
            {
                "symbol": symbol, "is_buy": is_buy, "sz": sz,
                "limit_px": limit_px, "client_order_id": cl_id,
                "reduce_only": reduce_only,
            },
        ))

        if self.config.place_latency_ms > 0:
            time.sleep(self.config.place_latency_ms / 1000.0)

        if self.config.place_response_outcome == "exchange_rejected":
            # OKX semantics: top code "0" + row sCode != "0" = the
            # venue rejected at the row level (e.g., post-only crossing,
            # min-notional, etc). The interpreter classifies this as
            # ``exchange_rejected``.
            return {
                "code": "0", "msg": "",
                "data": [{
                    "sCode": self.config.place_reject_scode,
                    "sMsg": self.config.place_reject_smsg,
                    "clOrdId": cl_id,
                    "ordId": "",
                }],
            }
        if self.config.place_response_outcome == "transport_rejected":
            # OKX semantics: top code != "0" = transport / rate-limit /
            # auth failure. The interpreter classifies as
            # ``transport_rejected``.
            return {
                "code": "1",
                "msg": self.config.place_reject_smsg,
                "data": [{
                    "sCode": self.config.place_reject_scode,
                    "sMsg": self.config.place_reject_smsg,
                    "clOrdId": cl_id,
                    "ordId": "",
                }],
            }

        # Accepted — assign an oid and register on the book.
        with self._lock:
            self._next_ord_id += 1
            oid = self._next_ord_id
            order = MockOrder(
                ord_id=oid,
                cl_ord_id=cl_id,
                side=Side.BUY if is_buy else Side.SELL,
                price=float(limit_px),
                size=float(sz),
                placed_at=datetime.now(timezone.utc),
                state=MockOrderState.LIVE,
            )
            self.order_book[oid] = order
            self._cloid_to_ord_id[cl_id] = oid

        return {
            "code": "0", "msg": "",
            "data": [{
                "sCode": "0", "sMsg": "",
                "ordId": str(oid),
                "clOrdId": cl_id,
                "tag": "",
            }],
        }

    def batch_place_post_only_limit(
        self, symbol: str, orders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """OKX batch-place — multiple orders in one HTTP call. Each
        order in ``orders`` is a dict with side/sz/px/clOrdId fields
        matching the OKX batch endpoint shape.

        Returns top {"code": ..., "data": [row, row, ...]} with one
        row per requested order. Per-order config is taken from
        ``self.config`` (so all orders in a batch share the same
        reject behavior — fine for the scenarios this mock covers).
        """
        self.calls.append((
            "batch_place_post_only_limit",
            {"symbol": symbol, "order_count": len(orders)},
        ))
        if self.config.place_latency_ms > 0:
            time.sleep(self.config.place_latency_ms / 1000.0)

        rows: list[dict[str, Any]] = []
        for o in orders:
            cl_id = str(o.get("clOrdId") or "") or f"mockcl{uuid.uuid4().hex[:24]}"
            is_buy = str(o.get("side", "")).lower() == "buy"
            try:
                sz = float(o.get("sz") or 0.0)
                px = float(o.get("px") or 0.0)
            except (TypeError, ValueError):
                sz, px = 0.0, 0.0

            if self.config.place_response_outcome == "exchange_rejected":
                rows.append({
                    "sCode": self.config.place_reject_scode,
                    "sMsg": self.config.place_reject_smsg,
                    "clOrdId": cl_id,
                    "ordId": "",
                })
                continue
            if self.config.place_response_outcome == "transport_rejected":
                # Top-level error path — handled below by changing top code.
                rows.append({
                    "sCode": self.config.place_reject_scode,
                    "sMsg": self.config.place_reject_smsg,
                    "clOrdId": cl_id,
                    "ordId": "",
                })
                continue
            # Accepted — register on the book.
            with self._lock:
                self._next_ord_id += 1
                oid = self._next_ord_id
                order = MockOrder(
                    ord_id=oid,
                    cl_ord_id=cl_id,
                    side=Side.BUY if is_buy else Side.SELL,
                    price=px,
                    size=sz,
                    placed_at=datetime.now(timezone.utc),
                    state=MockOrderState.LIVE,
                )
                self.order_book[oid] = order
                self._cloid_to_ord_id[cl_id] = oid
            rows.append({
                "sCode": "0", "sMsg": "",
                "ordId": str(oid),
                "clOrdId": cl_id,
                "tag": "",
            })

        top_code = (
            "1" if self.config.place_response_outcome == "transport_rejected" else "0"
        )
        return {"code": top_code, "msg": "", "data": rows}

    def cancel_order(self, symbol: str, oid: int) -> dict[str, Any]:
        return self._cancel_impl(symbol, oid=int(oid), cl_ord_id=None)

    def cancel_order_by_cloid(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any]:
        return self._cancel_impl(symbol, oid=None, cl_ord_id=client_order_id)

    def cancel_batch_orders(
        self, symbol: str, refs: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Batch cancel — used by the dispatcher's batched path."""
        self.calls.append(("cancel_batch_orders", {"symbol": symbol, "refs": refs}))
        rows: list[dict[str, Any]] = []
        for r in refs:
            oid_str = r.get("ordId")
            cloid = r.get("clOrdId")
            oid = int(oid_str) if oid_str else None
            single = self._cancel_impl(symbol, oid=oid, cl_ord_id=cloid)
            row_data = single.get("data") or [{}]
            row = row_data[0] if row_data else {}
            rows.append({
                "sCode": row.get("sCode", "0"),
                "sMsg": row.get("sMsg", ""),
                "ordId": str(oid) if oid else (row.get("ordId") or ""),
                "clOrdId": cloid or row.get("clOrdId", ""),
            })
        # Top code = 0 if any row succeeded, 1 otherwise.
        any_ok = any(r["sCode"] == "0" for r in rows)
        return {"code": "0" if any_ok else "1", "msg": "", "data": rows}

    def _cancel_impl(
        self,
        symbol: str,
        *,
        oid: Optional[int],
        cl_ord_id: Optional[str],
    ) -> dict[str, Any]:
        """Shared cancel logic — resolves the target order, applies the
        configured response code, and updates state on success."""
        self.calls.append((
            "cancel_order",
            {"symbol": symbol, "oid": oid, "cl_ord_id": cl_ord_id},
        ))
        # Choose response code: oneshot-then-success or sticky.
        scode = self.config.cancel_response_scode
        smsg = self.config.cancel_response_smsg
        if (
            self.config.cancel_response_oneshot
            and not self.config.cancel_response_oneshot_used
            and scode is not None
        ):
            self.config.cancel_response_oneshot_used = True
            # Use the configured code this once, then reset to None.
        elif self.config.cancel_response_oneshot:
            scode = None
            smsg = ""

        # Locate the order.
        with self._lock:
            order: Optional[MockOrder] = None
            if oid is not None:
                order = self.order_book.get(int(oid))
            elif cl_ord_id is not None:
                resolved_oid = self._cloid_to_ord_id.get(cl_ord_id)
                if resolved_oid is not None:
                    order = self.order_book.get(resolved_oid)

            if scode is not None and scode != "0":
                # Configured reject.
                return {
                    "code": "1", "msg": "",
                    "data": [{
                        "sCode": scode,
                        "sMsg": smsg,
                        "ordId": str(order.ord_id) if order else (str(oid) if oid else ""),
                        "clOrdId": cl_ord_id or (order.cl_ord_id if order else ""),
                    }],
                }

            # Success path — transition the order to CANCELED.
            if order is not None and order.state == MockOrderState.LIVE:
                order.state = MockOrderState.CANCELED
                order.canceled_at = datetime.now(timezone.utc)

            return {
                "code": "0", "msg": "",
                "data": [{
                    "sCode": "0", "sMsg": "",
                    "ordId": str(order.ord_id) if order else "",
                    "clOrdId": cl_ord_id or (order.cl_ord_id if order else ""),
                }],
            }

    def query_order_status_by_cloid(
        self, address: str, client_order_id: str
    ) -> dict[str, Any]:
        with self._lock:
            oid = self._cloid_to_ord_id.get(client_order_id)
            order = self.order_book.get(oid) if oid else None
        if order is None:
            return {"code": "0", "msg": "", "data": []}
        return {
            "code": "0", "msg": "",
            "data": [{
                "ordId": str(order.ord_id),
                "clOrdId": order.cl_ord_id,
                "state": order.state.value,
                "px": str(order.price),
                "sz": str(order.size),
                "side": order.side.value.lower(),
            }],
        }

    def market_close(self, symbol: str, sz: Optional[float] = None) -> dict[str, Any]:
        self.calls.append(("market_close", {"symbol": symbol, "sz": sz}))
        return {"code": "0", "msg": "", "data": [{"sCode": "0", "sMsg": ""}]}

    def rest_runtime_counters(self) -> dict[str, int]:
        return {"http_call_count": len(self.calls)}

    # ----- wire-format interpretation ---------------------------------------

    def interpret_place_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        """Mirrors OKX adapter's interpret_place_response."""
        if not isinstance(resp, dict):
            return None, "transport_rejected", "non_dict_response"
        top_code = str(resp.get("code", ""))
        data = resp.get("data") or []
        if top_code != "0":
            # Top-level error = transport / rate-limit.
            return None, "transport_rejected", str(resp.get("msg", ""))
        if not data:
            return None, "unconfirmed", "empty_data"
        row = data[0]
        scode = str(row.get("sCode", ""))
        if scode == "0":
            try:
                return int(row.get("ordId") or 0), "accepted", ""
            except (ValueError, TypeError):
                return None, "exchange_rejected", "bad_ord_id"
        return None, "exchange_rejected", row.get("sMsg", "") or scode

    def interpret_cancel_response(self, resp: Any) -> tuple[str, str]:
        """Mirrors OKX adapter's interpret_cancel_response."""
        if not isinstance(resp, dict):
            return "transport", "non_dict_response"
        data = resp.get("data") or []
        if not data:
            return "transport", "empty_data"
        row = data[0]
        scode = str(row.get("sCode", ""))
        if scode == "0":
            return "success", ""
        if scode in ("51400", "51410"):
            return "benign_missing", row.get("sMsg", "") or scode
        return "error", row.get("sMsg", "") or scode

    def interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        if not isinstance(resp, dict):
            return None, "transport", "non_dict_response"
        data = resp.get("data") or []
        if not data:
            return None, "not_found", "empty_data"
        row = data[0]
        try:
            oid = int(row.get("ordId") or 0)
        except (ValueError, TypeError):
            oid = None
        state = str(row.get("state", "")).lower()
        if state == "live":
            return oid, "open", ""
        if state == "canceled":
            return oid, "canceled", ""
        if state == "filled":
            return oid, "filled", ""
        return oid, "unknown_proc", state

    def make_client_order_id(
        self,
        symbol: str,
        side: Side,
        quote_cycle_id: str,
        price: float,
        size: float,
    ) -> str:
        """Deterministic cloid in OKX shape (≤32 alphanumeric chars)."""
        # Format: <symbol_short><side><cycle_id_short><price_int>
        sym_short = "".join(c for c in symbol if c.isalnum())[:6]
        side_c = "B" if side == Side.BUY else "S"
        cycle_short = "".join(c for c in quote_cycle_id if c.isalnum())[:8]
        price_str = f"{int(price * 1_000_000):x}"
        return f"{sym_short}{side_c}{cycle_short}{price_str}"[:32]

    # ----- cancel-pending capability (Bluefin) — return False ---------------

    def has_pending_cancel(self, symbol: str, side: Side) -> bool:
        return False

    # ----- WS event injection -----------------------------------------------

    def emit_ws_event(
        self,
        ord_id: int,
        new_state: MockOrderState,
    ) -> None:
        """Push a ``PrivateOrderUpdateEvent`` for ``ord_id`` to the
        attached event queue. Tests call this to simulate the venue's
        WS feed firing live/canceled/filled events.

        The bot's ``_dispatch_private_event`` checks ``isinstance`` of
        a few specific event classes (PrivateOrderUpdateEvent /
        PrivateFillEvent / PrivateWsConnectionEvent). Mock pushes the
        same dataclass shapes so the bot's handlers see them naturally.
        """
        from app.exchange.private_events import PrivateOrderUpdateEvent
        with self._lock:
            order = self.order_book.get(int(ord_id))
        if order is None:
            raise ValueError(f"no mock order with ord_id={ord_id}")
        if self._event_queue is None:
            raise RuntimeError(
                "MockOkxClient.emit_ws_event called without attached event_queue. "
                "Call `client.attach_event_queue(om._private_q)` first."
            )
        # Map mock state → raw status string the bot will normalize.
        raw_status = {
            MockOrderState.LIVE: "live",
            MockOrderState.CANCELED: "canceled",
            MockOrderState.FILLED: "filled",
            MockOrderState.PARTIAL: "partially_filled",
        }[new_state]
        # Side: bot accepts "B"/"A" (HL shape) OR full strings.
        side_str = "B" if order.side == Side.BUY else "A"
        remaining = max(0.0, order.size - order.filled_qty)
        evt = PrivateOrderUpdateEvent(
            oid=int(order.ord_id),
            coin=self.symbol,
            status=raw_status,
            status_timestamp_ms=int(time.time() * 1000),
            side=side_str,
            limit_px=float(order.price),
            remaining_sz=remaining,
            orig_sz=float(order.size),
            raw_status=raw_status,
            cloid=order.cl_ord_id,
        )
        self._event_queue.put(evt)

    def emit_ws_live(self, ord_id: int) -> None:
        self.emit_ws_event(ord_id, MockOrderState.LIVE)

    def emit_ws_canceled(self, ord_id: int) -> None:
        self.emit_ws_event(ord_id, MockOrderState.CANCELED)

    def emit_ws_filled(
        self,
        ord_id: int,
        fill_qty: Optional[float] = None,
        fill_px: Optional[float] = None,
    ) -> None:
        with self._lock:
            order = self.order_book.get(int(ord_id))
            if order is None:
                raise ValueError(f"no mock order with ord_id={ord_id}")
            order.state = MockOrderState.FILLED
            order.filled_qty = float(fill_qty) if fill_qty is not None else order.size
            order.filled_px = float(fill_px) if fill_px is not None else order.price
            order.filled_at = datetime.now(timezone.utc)
        self.emit_ws_event(ord_id, MockOrderState.FILLED)

    # ----- Test-assertion helpers -------------------------------------------

    def count_calls(self, method_name: str) -> int:
        return sum(1 for n, _ in self.calls if n == method_name)

    def live_orders(self) -> list[MockOrder]:
        with self._lock:
            return [o for o in self.order_book.values() if o.state == MockOrderState.LIVE]

    def canceled_orders(self) -> list[MockOrder]:
        with self._lock:
            return [o for o in self.order_book.values() if o.state == MockOrderState.CANCELED]

    def reset(self) -> None:
        """Clear all state — useful between test scenarios."""
        with self._lock:
            self.order_book.clear()
            self._cloid_to_ord_id.clear()
            self.calls.clear()
        self.config = MockClientConfig()
