"""
Generic perpetual-exchange adapter boundary.

This module is the single seam between the bot core (``app.bot``,
``app.execution``, ``app.market_data``, ``app.operator_metrics_reconcile``)
and any specific venue adapter (Hyperliquid today, GRVT / Aster / etc.
tomorrow). The bot core imports ``PerpExchangeAdapter``, ``OpenOrderRaw``
and ``FillRaw`` from here — it does not import any venue-specific name.

Design rules enforced by this module:

- The adapter Protocol is the *only* place where the list of operations
  the bot requires of an exchange is written down.
- Raw data types carry venue-neutral field names (``oid``, ``coin``,
  ``side``, ``limit_px``, ``sz``, ``cloid``, ``time_ms``, …). They
  intentionally mirror Hyperliquid's shape because Hyperliquid was the
  first adapter, but the names no longer mention Hyperliquid. New
  adapters are expected to normalise into these same shapes at their
  own edges.
- Response-wire interpretation (``interpret_place_response`` etc.) and
  client-order-id generation (``make_client_order_id``) are adapter
  methods, not free functions the bot core imports. Each adapter owns
  its own wire-format knowledge.

See ``code_reports/connectivity.md`` for the full design note.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

from app.enums import Side
from app.exchange.symbol_spec import SymbolSpec
from app.exchange.hyperliquid_types import HLFillRaw, HLOpenOrderRaw
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot


# ---------------------------------------------------------------------------
# Raw data types (venue-neutral names)
# ---------------------------------------------------------------------------
# Today these are aliases of the existing Hyperliquid dataclasses because
# their shape is already venue-neutral. If a future venue needs different
# fields (e.g. decimal fees, auth-specific ids) we split them.
OpenOrderRaw = HLOpenOrderRaw
FillRaw = HLFillRaw


# ---------------------------------------------------------------------------
# Adapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PerpExchangeAdapter(Protocol):
    """Structural type every venue adapter must satisfy.

    The bot core is typed against this Protocol. Concrete adapters
    (Hyperliquid, GRVT, …) satisfy it by structural typing — no explicit
    inheritance required.

    ``@runtime_checkable`` allows cheap ``isinstance`` smoke tests in
    bootstrapping / tests; it does **not** validate method signatures at
    runtime — static typing carries that weight.
    """

    # --- venue metadata ----------------------------------------------------
    @property
    def symbol_spec(self) -> SymbolSpec: ...

    symbol_spec_fetched_ok: bool

    def has_write_access(self) -> bool: ...

    # --- market / account reads -------------------------------------------
    def fetch_best_bid_ask(self, symbol: str) -> BestBidAsk: ...

    def fetch_position(self, address: str, symbol: str) -> PositionSnapshot: ...

    def fetch_account_snapshot(self, address: str) -> AccountSnapshot: ...

    def fetch_open_orders_raw(self, address: str) -> list[OpenOrderRaw]: ...

    def fetch_recent_fills_raw(self, address: str, symbol: str) -> list[FillRaw]: ...

    # --- order lifecycle ---------------------------------------------------
    def place_post_only_limit(
        self,
        symbol: str,
        is_buy: bool,
        sz: float,
        limit_px: float,
        *,
        client_order_id: Optional[str] = None,
        reduce_only: bool = False,
    ) -> dict[str, Any]: ...

    def cancel_order(self, symbol: str, oid: int) -> dict[str, Any]: ...

    def cancel_order_by_cloid(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any]: ...

    # --- cancel-confirmation gating (optional capability) -----------------
    #
    # Some venues (Bluefin Pro) ack cancel requests synchronously (HTTP 202)
    # but defer the actual matching-engine effect to an asynchronous WS
    # ``OrderCancellationUpdate`` event. Until that event arrives, the
    # order may still match incoming flow — and if the bot reacts to the
    # 202 by placing a same-side replacement immediately, both orders can
    # fill back-to-back (observed 2026-04-23: 5× BUY fills in 2s on
    # SUI-PERP, 50 SUI accumulated while the bot thought it had one open).
    #
    # Adapters for such venues implement an additional method::
    #
    #     def has_pending_cancel(self, symbol: str, side: Side) -> bool: ...
    #
    # returning True whenever an in-flight cancel on the given (symbol,
    # side) has not yet been confirmed. The execution layer's
    # replacement-placement path consults this via
    # ``getattr(client, "has_pending_cancel", None)`` (see
    # ``app/execution.py::_orchestrate``) and skips one tick if a cancel
    # is still pending. On timeout the entry clears, so a stuck WS never
    # blocks quoting beyond ``BLUEFIN_CANCEL_CONFIRM_TIMEOUT_SECONDS``.
    #
    # This method is intentionally NOT on the Protocol surface — it is an
    # additive capability. Synchronous-cancel venues (Hyperliquid, GRVT)
    # omit it, and the execution layer treats its absence as "no pending".

    def query_order_status_by_cloid(
        self, address: str, client_order_id: str
    ) -> dict[str, Any]: ...

    def market_close(self, symbol: str, sz: Optional[float] = None) -> dict[str, Any]: ...

    def rest_runtime_counters(self) -> dict[str, int]: ...

    # --- wire-format interpretation (venue-specific, adapter-owned) -------
    #
    # Each adapter parses its own wire responses. The bot core only sees
    # the normalised tuples below. Return-tuple contracts:
    #
    #   interpret_place_response(resp) -> (exchange_oid, outcome, reason)
    #     outcome ∈ {"accepted", "exchange_rejected", "transport_rejected",
    #                "unconfirmed"}
    #     exchange_oid is set iff outcome == "accepted".
    #     reason is a short free-form string.
    #
    #   interpret_cancel_response(resp) -> (kind, detail)
    #     kind ∈ {"success", "benign_missing", "error", "transport"}
    #
    #   interpret_order_status_response(resp) -> (oid_if_known, outcome, detail)
    #     outcome ∈ {"open", "filled", "canceled", "rejected",
    #                "not_found", "invalid", "transport", "unknown_proc"}

    def interpret_place_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]: ...

    def interpret_cancel_response(self, resp: Any) -> tuple[str, str]: ...

    def interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]: ...

    def make_client_order_id(
        self,
        symbol: str,
        side: Side,
        quote_cycle_id: str,
        price: float,
        size: float,
    ) -> str:
        """Deterministic client-order-id for a normalised quote intent.

        The adapter picks a format compatible with its venue (e.g.
        Hyperliquid: 16-byte ``0x`` + 32 hex). The bot core only relies
        on stability for the same inputs and opacity otherwise.
        """
        ...


__all__ = [
    "FillRaw",
    "OpenOrderRaw",
    "PerpExchangeAdapter",
]
