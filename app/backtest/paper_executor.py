"""Paper executor — Phase 2 of the backtesting build (v1.4.232).

A synthetic exchange adapter that:

* Implements the ``PerpExchangeAdapter`` Protocol surface the bot
  relies on (place / cancel / amend / fetch / interpret).
* Receives recorded market events from the replay driver via
  ``process_book_event`` / ``process_trade_event``.
* Synthesizes order acks + fills back into the bot's private event
  queue using a back-of-queue queue-position model.
* Runs single-threaded, deterministic, no network sockets.

Fill model (§2.2 of execution-plan):

When a post-only BUY is placed at price ``P``:

1. Snapshot ``queue_volume_ahead`` from the current top-of-book — if
   ``P > best_ask`` reject as post-only cross; if ``P > best_bid``
   the new order is the top (``queue_ahead = 0``); if ``P == best_bid``
   it goes behind the existing bid (``queue_ahead = bid_size``); if
   ``P < best_bid`` the order is below touch — queue_ahead = bid_size
   as a pessimistic floor (real depth unknown).
2. Order becomes "fillable" only after ``sim_place_latency_s`` has
   elapsed (simulates round-trip + exchange ack delay).
3. On every incoming SELL trade at price ``Tp <= P``: decrement
   ``queue_volume_ahead`` by the trade size. Once it reaches zero,
   the order starts filling against the remainder of the trade.

Symmetric for SELLs.

Edge cases handled:

* **Post-only cross-reject** — placing a BUY at price > current ask
  (or SELL at price < current bid) returns the
  ``"post_only_cross"`` error code.
* **Self-cross** — placing a BUY at ``P_b`` when we already rest a
  SELL at ``P_s <= P_b`` rejects the BUY with the same code (the
  matching engine would self-trade). The bot's
  ``is_post_only_immediate_match_rejection`` recognises it and arms
  the cooldown.
* **Cancel race** — a cancel takes ``sim_cancel_latency_s`` to
  apply. Trades arriving inside that window can still fill the
  order; the fill wins.
* **Amend resets queue position** — under back-of-queue policy, an
  amend is equivalent to ``cancel old + place new`` so the new
  order starts at the back of the queue at the new price.
* **Partial fills** — if an incoming trade is larger than our
  remaining size, fill what we have and leave the rest of the
  trade unmodeled (it would have hit the next resting order, but
  that order isn't ours).
* **Self-trade prevention** — never match a BUY against a SELL we
  own. Naturally enforced by (a) rejecting self-crossing
  placements, and (b) only synthesising fills against opposite-side
  resting orders when an actual trade event lands at that price.

Out of scope for Phase 2 (deferred to later phases):

* Latency variance / jitter (constant latencies for now)
* Depth beyond top-of-book (queue ahead estimated only from bid/ask
  size — sub-touch resting orders see ``bid_size`` as a pessimistic
  proxy)
* Funding payments
* IOC / FOK / non-post-only orders (bot quotes post-only only)
* Reduce-only intricacies (accepted as a flag; not separately
  modelled because the fill model is symmetric anyway)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Optional

from app.clock import Clock
from app.enums import Side
from app.exchange.base import FillRaw, OpenOrderRaw, PerpExchangeAdapter
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
)
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC, SymbolSpec
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PaperExecutorConfig:
    """Paper-executor knobs. Defaults match the §2.1 spec.

    ``fee_maker_bps`` is signed — negative = rebate (income), positive =
    fee paid. OKX MM-tier maker rebate is -0.005% = -0.5 bps. ``fee_
    taker_bps`` is unused today (post-only only) but kept for parity
    with the design doc + future non-post-only modes.
    """

    sim_place_latency_s: float = 0.020
    sim_cancel_latency_s: float = 0.015
    fee_maker_bps: float = -0.5
    fee_taker_bps: float = 1.0
    queue_policy: str = "back_of_queue"
    # v1.5.316 — backtest-only fill-rate lever. Scales the displayed
    # top-of-book size that ``_estimate_queue_ahead`` reports as the
    # "queue ahead of us" floor. The back-of-queue model is
    # deliberately pessimistic: a resting order only starts filling
    # after incoming opposite-side trades have chewed through the FULL
    # displayed size sitting in front of it (we have no L2 capture, so
    # we assume worst-case last-in-line). That produces a chronic,
    # MEASURED under-fill — surfaced as ``fill_ratio`` /
    # ``queue_block_ratio`` in ``fill_attribution_snapshot()``.
    #
    #   * 1.0 (default) — full displayed size ahead of us. BYTE-
    #     IDENTICAL to every replay before this knob existed (``x *
    #     1.0`` is an IEEE-754 identity), so the determinism contract
    #     is preserved. This is the conservative baseline.
    #   * <1.0          — assume we sit closer to the front of the
    #     queue (0.5 = behind only half the shown size). Orders fill
    #     SOONER → higher fill rate.
    #   * 0.0           — front-of-queue: fill as soon as ANY
    #     addressable trade prints at/through our price.
    #
    # This is intentionally NOT a direct "fill-rate %" output knob —
    # that would be circular and would BREAK the adverse-selection
    # correlation (toxicity/markout would look artificially benign).
    # Fills stay gated on REAL recorded trades crossing our price; only
    # our assumed queue position moves. Best used as a SENSITIVITY BAND
    # (e.g. 1.0 / 0.5 / 0.25) to check whether a Tier-B verdict's SIGN
    # is stable, then calibrated against the live fill rate and locked.
    # Deterministic (no RNG). Backtest-only: the live bot never reads
    # this field.
    queue_ahead_fraction: float = 1.0
    # Phase 4c (v1.4.237) — starting equity for the synthetic
    # account snapshot. The bot's ``_exchange_snapshot_healthy``
    # gate requires ``equity_usd > 0`` to pass; with the default 0
    # the bot pauses after 5 ticks on ``reconcile_stall``. Set this
    # to a positive number to let the strategy run. The exact value
    # affects drawdown-cap math + capital-adjusted position sizing
    # (Phase 4D.1 capital-aware throttle), so pick something
    # representative of the production account being modelled.
    starting_equity_usd: float = 1000.0

    def __post_init__(self) -> None:
        if self.sim_place_latency_s < 0:
            raise ValueError("sim_place_latency_s must be non-negative")
        if self.sim_cancel_latency_s < 0:
            raise ValueError("sim_cancel_latency_s must be non-negative")
        if self.queue_policy != "back_of_queue":
            raise ValueError(
                f"unknown queue_policy {self.queue_policy!r} (only "
                f"'back_of_queue' supported in Phase 2)"
            )
        if self.queue_ahead_fraction < 0:
            raise ValueError("queue_ahead_fraction must be non-negative")
        if self.starting_equity_usd < 0:
            raise ValueError("starting_equity_usd must be non-negative")


# ---------------------------------------------------------------------------
# Internal resting-order record
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _RestingOrder:
    """One simulated order resting on the paper book.

    ``queue_ahead`` is decremented as opposite-side trades arrive at
    or through the order's price. Once it reaches zero the order
    starts filling.

    ``active_at_mono`` is when the order becomes eligible to fill
    (``placed_at_mono + sim_place_latency_s``). Trades arriving
    before that are ignored for this order (the place hasn't been
    "ack'd" by the venue yet).

    ``cancel_pending_until_mono``: when ``cancel_order`` is called,
    this is set to ``clock.monotonic() + sim_cancel_latency_s``. The
    order is still fillable until then (cancel race) and is then
    removed.
    """

    oid: int
    cloid: Optional[str]
    side: Side
    price: float
    orig_size: float
    remaining_size: float
    queue_ahead: float
    placed_at_mono: float
    active_at_mono: float
    cancel_pending_until_mono: Optional[float] = None
    reduce_only: bool = False


# ---------------------------------------------------------------------------
# PaperExecutor
# ---------------------------------------------------------------------------


class PaperExecutor:
    """Synthetic ``PerpExchangeAdapter`` for replay.

    Single-threaded. All time reads go through the injected
    :class:`~app.clock.Clock` (always a ``ReplayClock`` under replay,
    a ``SystemClock`` is fine for unit tests). All emitted events go
    into the injected ``private_event_sink`` queue.

    The replay driver calls ``process_book_event`` / ``process_trade
    _event`` between bot ticks; the executor synthesises fills and
    drops them into ``private_event_sink`` for the bot's normal
    drain path to consume on the next tick.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        private_event_sink: Optional["Queue[Any]"] = None,
        config: Optional[PaperExecutorConfig] = None,
        symbol_spec: Optional[SymbolSpec] = None,
        symbol: str = "TON-USDT-SWAP",
    ) -> None:
        self._clock = clock
        self._sink: "Queue[Any]" = private_event_sink if private_event_sink is not None else Queue()
        self._config = config if config is not None else PaperExecutorConfig()
        self._symbol_spec = symbol_spec if symbol_spec is not None else FALLBACK_SYMBOL_SPEC
        self._symbol = symbol

        # Resting orders keyed by oid (the executor's exchange-side id).
        self._orders: dict[int, _RestingOrder] = {}
        # Cloid → oid index so cancel-by-cloid / amend works without scanning.
        self._cloid_to_oid: dict[str, int] = {}
        # Deterministic oid generator (counts up from 1).
        self._next_oid: int = 1

        # Current book state (last seen via process_book_event).
        self._best_bid: Optional[float] = None
        self._best_ask: Optional[float] = None
        self._bid_size: Optional[float] = None
        self._ask_size: Optional[float] = None
        self._last_trade_price: Optional[float] = None  # used as mark

        # Position + PnL bookkeeping.
        self._position_qty: float = 0.0
        self._avg_entry_price: float = 0.0
        self._realized_pnl: float = 0.0
        self._fees_total: float = 0.0
        # Monotonic fill-id sequence for synthetic fills.
        self._next_fill_seq: int = 1

        # Cumulative ack/fill counters used by tests + the Phase 4 report.
        self.acks_emitted: int = 0
        self.fills_emitted: int = 0
        self.rejects_emitted: int = 0
        self.cancels_emitted: int = 0
        # Phase 4 (v1.4.235) — cumulative notional traded (px × sz).
        # Used by the replay report's volume_usd metric.
        self.total_fill_volume_usd: float = 0.0

        # BUG-G fill-attribution diagnostics (v1.5.303). Let the replay
        # report distinguish the THREE causes of a low fill count:
        #   * quiet market   → trade_base_size_addressable ≈ 0
        #                       (few/no trades crossed our resting price)
        #   * back-of-queue  → addressable >> 0 but queue_ahead_absorbed
        #     pessimism         ≈ addressable and filled_base ≈ 0
        #   * not quoting    → addressable ≈ 0 AND time_with_orders_pct
        #                       low (read from the report summary)
        # All sizes are in BASE units (same units as queue_ahead /
        # bid_size / ask_size) so they're directly comparable.
        self.trade_prints_seen: int = 0
        self.trade_base_size_seen: float = 0.0
        # Trade volume whose price crossed ≥1 active resting order of
        # ours (i.e. WOULD have filled us if we were front-of-queue).
        self.trade_base_size_addressable: float = 0.0
        # Of the addressable volume, how much the back-of-queue policy
        # ate ahead of us before reaching our size.
        self.queue_ahead_absorbed_base: float = 0.0
        # Of the addressable volume, how much actually filled us.
        self.filled_base_from_trades: float = 0.0
        # Count of crossing trades that produced ZERO fill because the
        # queue ahead of us absorbed the whole print.
        self.fills_blocked_by_queue_only: int = 0

        # Required by ``PerpExchangeAdapter`` Protocol.
        self.symbol_spec_fetched_ok: bool = True

    # ------------------------------------------------------------------
    # PerpExchangeAdapter — venue metadata
    # ------------------------------------------------------------------

    @property
    def symbol_spec(self) -> SymbolSpec:
        return self._symbol_spec

    def has_write_access(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # PerpExchangeAdapter — market / account reads
    # ------------------------------------------------------------------

    def fetch_best_bid_ask(self, symbol: str) -> BestBidAsk:
        bb = self._best_bid
        ba = self._best_ask
        mid = None
        spread_bps = None
        if bb is not None and ba is not None and bb > 0 and ba > 0:
            mid = (bb + ba) / 2.0
            spread_bps = (ba - bb) / mid * 1e4 if mid else None
        return BestBidAsk(
            symbol=symbol,
            best_bid=bb,
            best_ask=ba,
            mid_price=mid,
            spread_bps=spread_bps,
            ts_exchange_ms=int(self._clock.time() * 1000),
            ts_local=self._clock.now_utc(),
            bid_size=self._bid_size,
            ask_size=self._ask_size,
        )

    def fetch_position(self, address: str, symbol: str) -> PositionSnapshot:
        mark = self._mark_price()
        notional = abs(self._position_qty) * (mark if mark is not None else 0.0)
        # Unrealised PnL = (mark - avg_entry) * position_qty
        unrealised = 0.0
        if mark is not None and self._position_qty != 0.0:
            unrealised = (mark - self._avg_entry_price) * self._position_qty
        return PositionSnapshot(
            symbol=symbol,
            position_qty=self._position_qty,
            avg_entry_price=self._avg_entry_price if self._position_qty != 0.0 else None,
            mark_price=mark,
            position_notional=notional,
            unrealized_pnl_usd=unrealised,
            ts_local=self._clock.now_utc(),
        )

    def fetch_account_snapshot(self, address: str) -> AccountSnapshot:
        # Equity = starting_equity + realised_pnl − fees. The bot's
        # health gate requires positive equity; a backtest report
        # cares about *changes* relative to starting equity. Drawdown
        # math (computed in the report) measures peak-to-trough
        # excursion from this equity curve.
        equity = (
            self._config.starting_equity_usd
            + self._realized_pnl
            - self._fees_total
        )
        return AccountSnapshot(
            equity_usd=equity,
            cash_usd=equity,
            withdrawable_usd=equity,
            ts_local=self._clock.now_utc(),
        )

    def fetch_open_orders_raw(self, address: str) -> list[OpenOrderRaw]:
        # Cancel-pending orders are still "open" until the cancel
        # latency elapses — reconciler should see them as live.
        out: list[OpenOrderRaw] = []
        for o in self._orders.values():
            out.append(
                OpenOrderRaw(
                    oid=o.oid,
                    coin=self._symbol,
                    side=o.side,
                    limit_px=o.price,
                    sz=o.remaining_size,
                    timestamp=int(o.placed_at_mono * 1000),
                    cloid=o.cloid,
                )
            )
        return out

    def fetch_recent_fills_raw(self, address: str, symbol: str) -> list[FillRaw]:
        # The fill stream is exposed via the private event queue. For
        # callers that prefer REST-style polling we'd need to keep a
        # ring buffer of recent fills — out of scope for Phase 2
        # (replay driver consumes via the queue path).
        return []

    # ------------------------------------------------------------------
    # PerpExchangeAdapter — order lifecycle
    # ------------------------------------------------------------------

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
        """Post-only limit place. Returns a dict the bot parses via
        ``interpret_place_response``.

        Cross-reject and self-cross are reported as
        ``{"status": "rejected", "code": "post_only_cross", ...}`` so
        the bot's existing ``is_post_only_immediate_match_rejection``
        detector trips and arms the cooldown.
        """
        side = Side.BUY if is_buy else Side.SELL

        # Post-only cross-reject vs current touch.
        if self._would_cross_book(side, limit_px):
            self.rejects_emitted += 1
            return {
                "status": "rejected",
                "code": "post_only_cross",
                "msg": "Post-only order would cross the book",
                "cloid": client_order_id,
            }

        # Self-cross check vs our own resting orders.
        if self._would_self_cross(side, limit_px):
            self.rejects_emitted += 1
            return {
                "status": "rejected",
                "code": "post_only_cross",
                "msg": "post-only would cross own order",
                "cloid": client_order_id,
            }

        oid = self._next_oid
        self._next_oid += 1
        now_mono = self._clock.monotonic()
        queue_ahead = self._estimate_queue_ahead(side, limit_px)
        order = _RestingOrder(
            oid=oid,
            cloid=client_order_id,
            side=side,
            price=limit_px,
            orig_size=sz,
            remaining_size=sz,
            queue_ahead=queue_ahead,
            placed_at_mono=now_mono,
            active_at_mono=now_mono + self._config.sim_place_latency_s,
            reduce_only=reduce_only,
        )
        self._orders[oid] = order
        if client_order_id is not None:
            self._cloid_to_oid[client_order_id] = oid

        # Emit synthetic order-update for the ack (status="open").
        self._emit_order_update(order, status="open")
        self.acks_emitted += 1

        return {
            "status": "ok",
            "oid": oid,
            "cloid": client_order_id,
        }

    def cancel_order(self, symbol: str, oid: int) -> dict[str, Any]:
        order = self._orders.get(oid)
        if order is None:
            return {
                "status": "benign_missing",
                "msg": f"order {oid} not found (already filled/cancelled)",
            }
        if order.cancel_pending_until_mono is not None:
            # Already cancelling — return success again (idempotent).
            return {"status": "ok", "oid": oid}

        now_mono = self._clock.monotonic()
        order.cancel_pending_until_mono = now_mono + self._config.sim_cancel_latency_s
        return {"status": "ok", "oid": oid}

    def cancel_order_by_cloid(
        self, symbol: str, client_order_id: str
    ) -> dict[str, Any]:
        oid = self._cloid_to_oid.get(client_order_id)
        if oid is None:
            return {
                "status": "benign_missing",
                "msg": f"cloid {client_order_id} not found",
            }
        return self.cancel_order(symbol, oid)

    # ------------------------------------------------------------------
    # Batch endpoints (OKX-shaped responses)
    # ------------------------------------------------------------------

    def cancel_batch_orders(
        self, symbol: str, refs: list[dict[str, str]]
    ) -> dict[str, Any]:
        """OKX-shaped batch cancel.

        Mirrors ``/api/v5/trade/cancel-batch-orders``. Each ref is
        ``{"clOrdId": "..."}`` or ``{"ordId": "..."}``. Returns:
            {"code": "0"|"1", "msg": "...",
             "data": [{"sCode": "0"|"51402"|..., "ordId": "...",
                       "clOrdId": "...", "sMsg": "..."}, ...]}

        Top code is "0" when every row succeeded, "1" when any row
        failed (OKX convention). The bot's ``interpret_okx_cancel_batch_response``
        reads per-row sCode for ground truth.
        """
        rows: list[dict[str, Any]] = []
        any_error = False
        for ref in refs:
            cloid = ref.get("clOrdId")
            oid_str = ref.get("ordId")
            target_oid: Optional[int] = None
            if cloid:
                target_oid = self._cloid_to_oid.get(cloid)
            elif oid_str:
                try:
                    target_oid = int(oid_str)
                except (TypeError, ValueError):
                    target_oid = None
            if target_oid is None or target_oid not in self._orders:
                # Map to OKX's "order not found" benign code so the
                # bot's interpreter classifies as ``benign_missing``.
                rows.append({
                    "sCode": "51402",
                    "sMsg": "order not found",
                    "ordId": oid_str,
                    "clOrdId": cloid,
                })
                continue
            inner = self.cancel_order(symbol, target_oid)
            if inner.get("status") == "ok":
                rows.append({
                    "sCode": "0",
                    "sMsg": "",
                    "ordId": str(target_oid),
                    "clOrdId": cloid,
                })
            else:
                any_error = True
                rows.append({
                    "sCode": "50000",
                    "sMsg": str(inner.get("msg", "")),
                    "ordId": str(target_oid),
                    "clOrdId": cloid,
                })
        return {
            "code": "1" if any_error else "0",
            "msg": "",
            "data": rows,
        }

    def batch_place_post_only_limit(
        self, symbol: str, orders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """OKX-shaped batch place.

        Each order in ``orders`` carries the same fields as
        ``place_post_only_limit``'s kwargs: ``side`` ("BUY"/"SELL"),
        ``sz``, ``limit_px``, ``client_order_id``, ``reduce_only``.
        """
        rows: list[dict[str, Any]] = []
        any_error = False
        for order in orders:
            try:
                is_buy = str(order.get("side", "")).upper() == "BUY"
                sz = float(order.get("sz", 0.0))
                limit_px = float(order.get("limit_px", 0.0))
                cloid = order.get("client_order_id")
                reduce_only = bool(order.get("reduce_only", False))
            except (TypeError, ValueError) as e:
                any_error = True
                rows.append({
                    "sCode": "50001",
                    "sMsg": f"bad row: {e}",
                    "ordId": "",
                    "clOrdId": order.get("client_order_id"),
                })
                continue
            inner = self.place_post_only_limit(
                symbol, is_buy=is_buy, sz=sz, limit_px=limit_px,
                client_order_id=cloid, reduce_only=reduce_only,
            )
            if inner.get("status") == "ok":
                rows.append({
                    "sCode": "0",
                    "sMsg": "",
                    "ordId": str(inner.get("oid", "")),
                    "clOrdId": cloid,
                })
            else:
                any_error = True
                # Map post-only cross → OKX code 51008 (post-only
                # would have crossed). Other rejects → generic 50001.
                code = "51008" if inner.get("code") == "post_only_cross" else "50001"
                rows.append({
                    "sCode": code,
                    "sMsg": str(inner.get("msg", "")),
                    "ordId": "",
                    "clOrdId": cloid,
                })
        return {
            "code": "1" if any_error else "0",
            "msg": "",
            "data": rows,
        }

    def amend_batch_orders(
        self, symbol: str, amends: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """OKX-shaped batch amend.

        Each entry: ``{"ordId": "...", "clOrdId": "...", "newSz": ...,
        "newPx": ...}``. Forwards each to ``amend_order``.
        """
        rows: list[dict[str, Any]] = []
        any_error = False
        for amend in amends:
            oid_str = amend.get("ordId")
            cloid = amend.get("clOrdId")
            target_oid: Optional[int] = None
            if cloid:
                target_oid = self._cloid_to_oid.get(cloid)
            elif oid_str:
                try:
                    target_oid = int(oid_str)
                except (TypeError, ValueError):
                    target_oid = None
            if target_oid is None or target_oid not in self._orders:
                rows.append({
                    "sCode": "51402",
                    "sMsg": "order not found",
                    "ordId": oid_str,
                    "clOrdId": cloid,
                })
                continue
            new_px = amend.get("newPx")
            new_sz = amend.get("newSz")
            try:
                px = float(new_px) if new_px is not None else None
                sz = float(new_sz) if new_sz is not None else None
            except (TypeError, ValueError):
                any_error = True
                rows.append({
                    "sCode": "50001",
                    "sMsg": "bad row: non-numeric newPx/newSz",
                    "ordId": str(target_oid),
                    "clOrdId": cloid,
                })
                continue
            inner = self.amend_order(symbol, target_oid, new_price=px, new_size=sz)
            if inner.get("status") == "ok":
                rows.append({
                    "sCode": "0",
                    "sMsg": "",
                    "ordId": str(target_oid),
                    "clOrdId": cloid,
                })
            else:
                any_error = True
                code = "51008" if inner.get("code") == "post_only_cross" else "50001"
                rows.append({
                    "sCode": code,
                    "sMsg": str(inner.get("msg", "")),
                    "ordId": str(target_oid),
                    "clOrdId": cloid,
                })
        return {
            "code": "1" if any_error else "0",
            "msg": "",
            "data": rows,
        }

    def bump_row_rate_limit_counter(self, n: int) -> None:
        """No-op on the paper executor — no real rate limits to track.
        The production OKX client uses this hook to drive the
        ``rest_runtime_counters``; we just ignore it."""
        return None

    def amend_order(
        self,
        symbol: str,
        oid: int,
        *,
        new_price: Optional[float] = None,
        new_size: Optional[float] = None,
    ) -> dict[str, Any]:
        """Amend = cancel old + place new at new price/size.

        Under back-of-queue policy the amend resets queue position —
        the new order goes to the back of the queue at the new price.
        Real OKX behavior matches this (PMP amend → new queue position).
        """
        order = self._orders.get(oid)
        if order is None:
            return {
                "status": "benign_missing",
                "msg": f"order {oid} not found",
            }
        if order.cancel_pending_until_mono is not None:
            return {
                "status": "rejected",
                "code": "cancel_in_flight",
                "msg": "cannot amend while cancel pending",
            }
        # Apply new params.
        price = new_price if new_price is not None else order.price
        size = new_size if new_size is not None else order.remaining_size
        # Self-cross / cross-book check on the new price.
        if self._would_cross_book(order.side, price, exclude_oid=oid):
            self.rejects_emitted += 1
            return {
                "status": "rejected",
                "code": "post_only_cross",
                "msg": "Post-only amend would cross the book",
            }
        if self._would_self_cross(order.side, price, exclude_oid=oid):
            self.rejects_emitted += 1
            return {
                "status": "rejected",
                "code": "post_only_cross",
                "msg": "post-only amend would cross own order",
            }
        now_mono = self._clock.monotonic()
        order.price = price
        order.orig_size = size
        order.remaining_size = size
        order.queue_ahead = self._estimate_queue_ahead(order.side, price)
        order.placed_at_mono = now_mono
        order.active_at_mono = now_mono + self._config.sim_place_latency_s
        # Emit refreshed open event.
        self._emit_order_update(order, status="open")
        return {"status": "ok", "oid": oid}

    def query_order_status_by_cloid(
        self, address: str, client_order_id: str
    ) -> dict[str, Any]:
        oid = self._cloid_to_oid.get(client_order_id)
        if oid is None:
            return {"status": "not_found"}
        order = self._orders.get(oid)
        if order is None:
            # In the cloid index but not in the live book → terminal.
            return {"status": "filled_or_cancelled", "oid": oid}
        return {
            "status": "open",
            "oid": oid,
            "remaining_sz": order.remaining_size,
            "limit_px": order.price,
        }

    def market_close(
        self, symbol: str, sz: Optional[float] = None
    ) -> dict[str, Any]:
        """Market-close at the current touch. Generates a taker fill
        against the position. ``sz`` defaults to the full position."""
        qty = sz if sz is not None else abs(self._position_qty)
        if qty <= 0.0 or self._position_qty == 0.0:
            return {"status": "no_position"}
        # Reduce the long → sell at bid; reduce the short → buy at ask.
        if self._position_qty > 0:
            side = Side.SELL
            price = self._best_bid if self._best_bid is not None else self._mark_price()
        else:
            side = Side.BUY
            price = self._best_ask if self._best_ask is not None else self._mark_price()
        if price is None or price <= 0:
            return {"status": "no_book", "msg": "no touch to market-close against"}
        fill_qty = min(qty, abs(self._position_qty))
        self._apply_fill(
            side=side,
            price=price,
            size=fill_qty,
            is_taker=True,
            oid=None,
            cloid=None,
        )
        return {"status": "ok", "filled_sz": fill_qty, "price": price}

    def rest_runtime_counters(self) -> dict[str, int]:
        return {
            "place_total": self.acks_emitted + self.rejects_emitted,
            "place_ok": self.acks_emitted,
            "place_rejected": self.rejects_emitted,
            "cancel_total": self.cancels_emitted,
            "fills_total": self.fills_emitted,
        }

    # ------------------------------------------------------------------
    # PerpExchangeAdapter — wire-format interpretation
    # ------------------------------------------------------------------

    def interpret_place_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        if not isinstance(resp, dict):
            return None, "transport_rejected", f"non-dict response: {resp!r}"
        status = resp.get("status")
        if status == "ok":
            oid = resp.get("oid")
            if not isinstance(oid, int):
                return None, "unconfirmed", "missing oid in ok response"
            return oid, "accepted", "ok"
        if status == "rejected":
            code = resp.get("code", "unknown")
            msg = resp.get("msg", "")
            return None, "exchange_rejected", f"{code}: {msg}"
        return None, "transport_rejected", f"unknown status {status!r}"

    def interpret_cancel_response(self, resp: Any) -> tuple[str, str]:
        if not isinstance(resp, dict):
            return "transport", f"non-dict response: {resp!r}"
        status = resp.get("status")
        if status == "ok":
            return "success", "ok"
        if status == "benign_missing":
            return "benign_missing", resp.get("msg", "")
        return "error", f"unknown status {status!r}"

    def interpret_order_status_response(
        self, resp: Any
    ) -> tuple[Optional[int], str, str]:
        if not isinstance(resp, dict):
            return None, "transport", f"non-dict response: {resp!r}"
        status = resp.get("status")
        oid = resp.get("oid") if isinstance(resp.get("oid"), int) else None
        if status == "open":
            return oid, "open", "ok"
        if status == "filled_or_cancelled":
            return oid, "filled", "terminal"
        if status == "not_found":
            return None, "not_found", "missing"
        return oid, "unknown_proc", f"unknown status {status!r}"

    def make_client_order_id(
        self,
        symbol: str,
        side: Side,
        quote_cycle_id: str,
        price: float,
        size: float,
    ) -> str:
        """Deterministic cloid for replay. Same inputs → same id."""
        # Hash-free deterministic encoding so two replays produce
        # byte-identical reports.
        side_char = "B" if side == Side.BUY else "S"
        # Round price/size to the tick/step grid so float noise can't
        # change the cloid across runs.
        tick = self._symbol_spec.price_tick or 1.0
        step = self._symbol_spec.size_step or 1.0
        px_int = int(round(price / tick))
        sz_int = int(round(size / step))
        return f"paper-{symbol}-{side_char}-{quote_cycle_id}-{px_int}-{sz_int}"

    # ------------------------------------------------------------------
    # Replay-driver-only surface
    # ------------------------------------------------------------------

    def process_book_event(
        self,
        bid: float,
        ask: float,
        bid_size: float,
        ask_size: float,
    ) -> None:
        """Update the executor's view of top-of-book.

        Must be called before any ``process_trade_event`` at the same
        timestamp so the fill matcher sees the correct book.
        """
        if bid is not None:
            self._best_bid = float(bid)
        if ask is not None:
            self._best_ask = float(ask)
        if bid_size is not None:
            self._bid_size = float(bid_size)
        if ask_size is not None:
            self._ask_size = float(ask_size)
        # Cancel-expiry sweep — orders whose cancel window has lapsed
        # are removed and a "canceled" event emitted.
        self._sweep_pending_cancels()

    def process_trade_event(
        self,
        price: float,
        size: float,
        side: str,
    ) -> None:
        """One trade tape print at ``price`` for ``size``.

        ``side`` is the AGGRESSOR side — "SELL" means a seller hit
        the bid (resting BUY orders may fill). "BUY" means a buyer
        lifted the ask (resting SELL orders may fill).
        """
        if size <= 0 or price <= 0:
            return
        self._last_trade_price = float(price)
        # BUG-G: total market activity seen by the matcher.
        self.trade_prints_seen += 1
        self.trade_base_size_seen += float(size)
        aggressor = side.upper() if isinstance(side, str) else str(side).upper()
        if aggressor in ("SELL", "S", "A"):
            self._match_sell_aggressor(price=float(price), size=float(size))
        elif aggressor in ("BUY", "B"):
            self._match_buy_aggressor(price=float(price), size=float(size))
        # Sweep cancels after each event (cancel-race semantics).
        self._sweep_pending_cancels()

    def current_position_qty(self) -> float:
        return self._position_qty

    def current_realized_pnl(self) -> float:
        return self._realized_pnl

    def current_fees(self) -> float:
        return self._fees_total

    def current_avg_entry(self) -> float:
        return self._avg_entry_price

    def update_from_rest_snapshot(self, _snapshot: Any) -> None:
        """Phase 3 hook — replay driver delivers periodic REST polls
        to ensure paper state stays in sync with what the bot's
        reconciler expects to see. No-op in Phase 2; the replay
        driver populates this when it's needed."""
        return None

    def fill_attribution_snapshot(self) -> dict[str, float]:
        """BUG-G (v1.5.303): expose why the fill count is what it is.

        The audit of v1.5.294 found 7 fills over 24,256 ticks with NO way
        to tell whether the cause was (a) a quiet market, (b) back-of-queue
        pessimism, or (c) the bot not quoting. This snapshot makes the
        replay report self-diagnosing. All *_base figures are in BASE units
        and directly comparable to each other.

        ``queue_block_ratio`` = queue_ahead_absorbed_base / addressable:
        of all the trade volume that crossed our resting price, the fraction
        the back-of-queue model ate before reaching us. Near 1.0 with
        ``filled_base_from_trades`` ≈ 0 ⇒ low fills are a queue-position
        artifact (cause b). ``trade_base_size_addressable`` ≈ 0 ⇒ either a
        quiet market (cause a) or not quoting (cause c) — disambiguate
        against ``time_with_orders_pct`` in the report summary.
        """
        addressable = self.trade_base_size_addressable
        queue_block_ratio = (
            self.queue_ahead_absorbed_base / addressable
            if addressable > 0.0
            else 0.0
        )
        fill_ratio = (
            self.filled_base_from_trades / addressable
            if addressable > 0.0
            else 0.0
        )
        return {
            "trade_prints_seen": int(self.trade_prints_seen),
            "trade_base_size_seen": self.trade_base_size_seen,
            "trade_base_size_addressable": addressable,
            "queue_ahead_absorbed_base": self.queue_ahead_absorbed_base,
            "filled_base_from_trades": self.filled_base_from_trades,
            "fills_blocked_by_queue_only": int(self.fills_blocked_by_queue_only),
            "queue_block_ratio": queue_block_ratio,
            "fill_ratio": fill_ratio,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_price(self) -> Optional[float]:
        """Use mid as mark; fall back to last trade if no book."""
        if self._best_bid is not None and self._best_ask is not None:
            return (self._best_bid + self._best_ask) / 2.0
        return self._last_trade_price

    def _would_cross_book(
        self, side: Side, price: float, *, exclude_oid: Optional[int] = None
    ) -> bool:
        """Would a post-only at this price cross the visible touch?"""
        if side == Side.BUY:
            if self._best_ask is not None and price >= self._best_ask:
                return True
        else:
            if self._best_bid is not None and price <= self._best_bid:
                return True
        return False

    def _would_self_cross(
        self, side: Side, price: float, *, exclude_oid: Optional[int] = None
    ) -> bool:
        """Does this placement cross our own opposite-side rests?"""
        for o in self._orders.values():
            if exclude_oid is not None and o.oid == exclude_oid:
                continue
            if o.side == side:
                continue
            # opposite side
            if side == Side.BUY and price >= o.price:
                return True
            if side == Side.SELL and price <= o.price:
                return True
        return False

    def _estimate_queue_ahead(self, side: Side, price: float) -> float:
        """Pessimistic queue-volume-ahead estimate from top-of-book.

        * BUY at price > best_bid (new top of book): 0 (we're alone).
        * BUY at price == best_bid: bid_size (we sit behind it).
        * BUY at price < best_bid: bid_size (pessimistic floor — real
          depth at that level unknown without full L2 capture).
        * Symmetric for SELL vs best_ask / ask_size.

        The non-zero result is scaled by
        ``config.queue_ahead_fraction`` (default 1.0 = full displayed
        size = the pessimistic baseline). Lower values model a more
        front-of-queue position and raise the fill rate; see
        ``PaperExecutorConfig.queue_ahead_fraction``. At the default
        ``x * 1.0`` is an IEEE-754 identity, so this is byte-identical
        to the pre-knob behaviour.
        """
        frac = self._config.queue_ahead_fraction
        if side == Side.BUY:
            if self._best_bid is None or self._bid_size is None:
                return 0.0
            if price > self._best_bid:
                return 0.0
            return float(self._bid_size) * frac
        else:
            if self._best_ask is None or self._ask_size is None:
                return 0.0
            if price < self._best_ask:
                return 0.0
            return float(self._ask_size) * frac

    def _match_sell_aggressor(self, *, price: float, size: float) -> None:
        """A SELL trade prints at ``price``. Eats resting BUYs with
        limit ≥ price, in price-time priority."""
        remaining = size
        # Sort eligible BUYs: best price first (highest), then oldest
        # placement first.
        candidates = [
            o for o in self._orders.values()
            if o.side == Side.BUY and o.price >= price
        ]
        candidates.sort(key=lambda o: (-o.price, o.placed_at_mono))
        # BUG-G fill attribution: a trade is "addressable" if it crosses at
        # least one of OUR active resting orders. If addressable volume goes
        # by but produces no fill, the cause is back-of-queue pessimism
        # (queue_ahead absorbed it), not a quiet market or not-quoting.
        has_active = any(self._is_active(o) for o in candidates)
        if has_active:
            self.trade_base_size_addressable += float(size)
            filled_before = self.filled_base_from_trades
        for order in candidates:
            if remaining <= 0:
                break
            if not self._is_active(order):
                continue
            remaining = self._consume_into_order(order, remaining, fill_price=price)
        if has_active and self.filled_base_from_trades == filled_before:
            self.fills_blocked_by_queue_only += 1
        self._reap_filled()

    def _match_buy_aggressor(self, *, price: float, size: float) -> None:
        """A BUY trade prints at ``price``. Eats resting SELLs with
        limit ≤ price."""
        remaining = size
        candidates = [
            o for o in self._orders.values()
            if o.side == Side.SELL and o.price <= price
        ]
        candidates.sort(key=lambda o: (o.price, o.placed_at_mono))
        # BUG-G fill attribution: see _match_sell_aggressor.
        has_active = any(self._is_active(o) for o in candidates)
        if has_active:
            self.trade_base_size_addressable += float(size)
            filled_before = self.filled_base_from_trades
        for order in candidates:
            if remaining <= 0:
                break
            if not self._is_active(order):
                continue
            remaining = self._consume_into_order(order, remaining, fill_price=price)
        if has_active and self.filled_base_from_trades == filled_before:
            self.fills_blocked_by_queue_only += 1
        self._reap_filled()

    def _is_active(self, order: _RestingOrder) -> bool:
        """Order is eligible to match if its place-latency has elapsed.
        Orders pending cancel remain active until cancel-latency lapses
        (cancel race)."""
        return self._clock.monotonic() >= order.active_at_mono

    def _consume_into_order(
        self,
        order: _RestingOrder,
        trade_remaining: float,
        *,
        fill_price: float,
    ) -> float:
        """Apply trade volume against ``order``'s queue + size.

        Returns the trade volume left after eating through this order.
        """
        # First, the trade chews through the queue ahead of us.
        if order.queue_ahead > 0:
            eaten = min(order.queue_ahead, trade_remaining)
            order.queue_ahead -= eaten
            trade_remaining -= eaten
            # BUG-G fill attribution: track base volume the queue-ahead model
            # absorbed before it could reach us. If this dominates the
            # addressable total while filled_base stays ~0, low fills are a
            # queue-position artifact, not a quiet market or not-quoting.
            self.queue_ahead_absorbed_base += eaten
            if trade_remaining <= 0:
                return 0.0
        # Now fill the order itself.
        fill_sz = min(order.remaining_size, trade_remaining)
        if fill_sz <= 0:
            return trade_remaining
        # BUG-G fill attribution: base volume that actually reached our order.
        self.filled_base_from_trades += fill_sz
        # Fill is at the order's limit price (we sat on the book — the
        # trade tape may have printed worse, but we got our limit).
        # This is the standard post-only-maker convention.
        self._apply_fill(
            side=order.side,
            price=order.price,
            size=fill_sz,
            is_taker=False,
            oid=order.oid,
            cloid=order.cloid,
        )
        order.remaining_size -= fill_sz
        # Emit the follow-up order-update AFTER the decrement so its
        # status + remaining reflect the TRUE post-fill state. This is
        # load-bearing: the bot's PrivateFillEvent path
        # (``ingest_hl_fill_raw``) only updates position/PnL — it does
        # NOT retire the WorkingOrder. The WO is terminalized solely on
        # the order-update path via ``_map_hl_ws_order_status``. So a
        # full fill MUST surface as ``status="filled"`` remaining 0,
        # else the WO lingers, the next REST reconcile sees the order
        # missing on the (already-reaped) exchange, and the bot enters
        # the exchange_mismatch → side_unresolved → orphan-cancel
        # cascade. A partial emits ``status="open"`` with the correct
        # remaining (maps to PARTIAL — WO stays). The ``<= 0`` test is
        # the SAME one ``_reap_filled`` uses, so an emitted "filled"
        # always coincides with the exchange dropping the order.
        status = "filled" if order.remaining_size <= 0 else "open"
        self._emit_order_update(order, status=status)
        return trade_remaining - fill_sz

    def _apply_fill(
        self,
        *,
        side: Side,
        price: float,
        size: float,
        is_taker: bool,
        oid: Optional[int],
        cloid: Optional[str],
    ) -> None:
        """Update position + realised PnL + fees, emit fill event."""
        # Position delta: +size for BUY, -size for SELL.
        delta = size if side == Side.BUY else -size
        prev_qty = self._position_qty
        new_qty = prev_qty + delta
        realized_from_fill = 0.0

        if prev_qty == 0.0:
            # Opening fresh — avg = fill price.
            self._avg_entry_price = price
        elif prev_qty * delta > 0.0:
            # Same direction → weighted-average entry.
            total_notional = abs(prev_qty) * self._avg_entry_price + size * price
            self._avg_entry_price = total_notional / (abs(prev_qty) + size)
        else:
            # Reducing or flipping.
            closing_sz = min(abs(prev_qty), size)
            if prev_qty > 0:
                # Closing long via SELL.
                realized_from_fill = (price - self._avg_entry_price) * closing_sz
            else:
                # Closing short via BUY.
                realized_from_fill = (self._avg_entry_price - price) * closing_sz
            if abs(delta) > abs(prev_qty):
                # Flip — residual opens fresh in the new direction.
                self._avg_entry_price = price
            # else: avg_entry stays (still some position left at old entry).

        self._position_qty = new_qty
        if self._position_qty == 0.0:
            self._avg_entry_price = 0.0

        self._realized_pnl += realized_from_fill
        bps = self._config.fee_taker_bps if is_taker else self._config.fee_maker_bps
        fee = size * price * bps / 1e4
        self._fees_total += fee
        self.total_fill_volume_usd += size * price

        # Emit synthetic PrivateFillEvent into the sink.
        fill_id = f"paper-fill-{self._next_fill_seq}"
        self._next_fill_seq += 1
        time_ms = int(self._clock.time() * 1000)
        self._sink.put(
            PrivateFillEvent(
                fill_id=fill_id,
                oid=oid,
                coin=self._symbol,
                px=price,
                sz=size,
                side="B" if side == Side.BUY else "A",
                time_ms=time_ms,
                fee=fee,
                closed_pnl=realized_from_fill,
                crossed=is_taker,
                is_snapshot=False,
                raw={},
            )
        )
        self.fills_emitted += 1
        # NOTE: the follow-up order-update is emitted by the CALLER
        # (``_consume_into_order``) AFTER it decrements
        # ``order.remaining_size`` — emitting here would use the stale
        # pre-decrement size and never produce a terminal "filled".
        # The taker path (``market_close``, oid=None) intentionally
        # emits no order-update: there is no resting WO to terminalize.

    def _emit_order_update(self, order: _RestingOrder, *, status: str) -> None:
        time_ms = int(self._clock.time() * 1000)
        self._sink.put(
            PrivateOrderUpdateEvent(
                oid=order.oid,
                coin=self._symbol,
                status=status,
                status_timestamp_ms=time_ms,
                side="B" if order.side == Side.BUY else "A",
                limit_px=order.price,
                remaining_sz=order.remaining_size,
                orig_sz=order.orig_size,
                raw_status=status,
                cloid=order.cloid,
            )
        )

    def _reap_filled(self) -> None:
        """Remove orders that have ``remaining_size <= 0``."""
        terminal_oids = [
            oid for oid, o in self._orders.items() if o.remaining_size <= 0
        ]
        for oid in terminal_oids:
            o = self._orders.pop(oid)
            if o.cloid is not None:
                self._cloid_to_oid.pop(o.cloid, None)

    def _sweep_pending_cancels(self) -> None:
        """Remove orders whose cancel-pending window has lapsed."""
        now_mono = self._clock.monotonic()
        terminal_oids: list[int] = []
        for oid, o in self._orders.items():
            if o.cancel_pending_until_mono is None:
                continue
            if now_mono >= o.cancel_pending_until_mono:
                terminal_oids.append(oid)
        for oid in terminal_oids:
            o = self._orders.pop(oid)
            if o.cloid is not None:
                self._cloid_to_oid.pop(o.cloid, None)
            # Emit canceled event.
            self._emit_order_update(o, status="canceled")
            self.cancels_emitted += 1


# Note: Protocol conformance is verified at test time via
# ``isinstance(executor, PerpExchangeAdapter)`` — see
# ``tests/test_paper_executor.py::test_satisfies_perpexchangeadapter_protocol``.
# That covers the structural-typing surface without an import-time
# side effect.
