"""Replay-time stream adapters — Phase 3 (v1.4.234).

The production bot consumes market data through two streams:

* ``OkxPublicStream`` (and venue equivalents) — pushes BBO updates
  to an ``on_bbo`` callback registered at construction. Also
  optionally pushes trade prints when subscribed to the trades
  channel.
* Private WS streams enqueue ``PrivateOrderUpdateEvent`` /
  ``PrivateFillEvent`` into a shared ``queue.Queue`` that the bot
  drains via ``_exec.drain_private_events()`` at the start of each
  tick.

For replay we substitute:

* :class:`ReplayPublicStream` — parses recorded OKX bbo-tbt /
  books5 / trades and Binance bookTicker frames, calls the
  registered ``on_bbo`` / ``on_trade`` callbacks synchronously.
* :class:`ReplayPrivateStream` — parses recorded OKX private order
  / fill messages and pushes normalised ``Private*Event``s into
  the same private-event queue the paper executor uses. (In Phase
  3 the paper executor is the SOURCE of synthetic fills; if the
  bot is later wired in, the *recorded* private events get pushed
  too, simulating multi-account drift checks.)

Both streams expose the production lifecycle surface
(``start()`` / ``stop()`` / ``request_reconnect()``) as no-ops so
they're drop-in for code that calls those.

Determinism: parsing is purely a function of the recorded payload.
Two replays of the same fixture call the same callbacks with the
same arguments in the same order.
"""

from __future__ import annotations

import logging
from queue import Queue
from typing import Any, Callable, Optional

from app.backtest.event_stream import RecordedEvent
from app.clock import Clock
from app.exchange.private_events import (
    PrivateFillEvent,
    PrivateOrderUpdateEvent,
)
from app.models import BestBidAsk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_okx_book_data_row(
    row: dict[str, Any],
    *,
    symbol: str,
) -> Optional[BestBidAsk]:
    """Parse one OKX ``bbo-tbt`` / ``books5`` data entry.

    Both channels share the same row shape:

    .. code-block:: json

        {
          "asks": [["2.048", "5411", "0", "28"], ...],
          "bids": [["2.047", "7534", "0", "36"], ...],
          "ts": "1779353641504",
          "seqId": 10989504109
        }

    Returns ``None`` if asks or bids are missing.
    """
    asks = row.get("asks") or []
    bids = row.get("bids") or []
    if not asks or not bids:
        return None
    try:
        best_ask = float(asks[0][0])
        ask_size = float(asks[0][1])
        best_bid = float(bids[0][0])
        bid_size = float(bids[0][1])
    except (IndexError, TypeError, ValueError) as e:
        logger.warning("malformed okx book row %r: %s", row, e)
        return None
    if best_bid <= 0 or best_ask <= 0:
        return None
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 1e4 if mid else None
    ts_ms = row.get("ts")
    ts_exchange_ms: Optional[int]
    if isinstance(ts_ms, str) and ts_ms.isdigit():
        ts_exchange_ms = int(ts_ms)
    elif isinstance(ts_ms, int):
        ts_exchange_ms = ts_ms
    else:
        ts_exchange_ms = None
    return BestBidAsk(
        symbol=symbol,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid,
        spread_bps=spread_bps,
        ts_exchange_ms=ts_exchange_ms,
        bid_size=bid_size,
        ask_size=ask_size,
    )


def _parse_binance_book_ticker(
    msg: dict[str, Any],
    *,
    symbol: str,
) -> Optional[BestBidAsk]:
    """Parse a Binance ``bookTicker`` payload.

    .. code-block:: json

        {"e":"bookTicker","u":...,"s":"TONUSDT",
         "b":"2.0482","B":"335.7","a":"2.0483","A":"555.5",
         "T":1779353641206,"E":1779353641206}
    """
    best_bid = _coerce_float(msg.get("b"))
    bid_size = _coerce_float(msg.get("B"))
    best_ask = _coerce_float(msg.get("a"))
    ask_size = _coerce_float(msg.get("A"))
    if best_bid is None or best_ask is None or best_bid <= 0 or best_ask <= 0:
        return None
    mid = (best_bid + best_ask) / 2.0
    spread_bps = (best_ask - best_bid) / mid * 1e4 if mid else None
    t_ms = msg.get("T")
    ts_exchange_ms = int(t_ms) if isinstance(t_ms, int) else None
    return BestBidAsk(
        symbol=symbol,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid,
        spread_bps=spread_bps,
        ts_exchange_ms=ts_exchange_ms,
        bid_size=bid_size,
        ask_size=ask_size,
    )


def _parse_okx_trade_row(row: dict[str, Any]) -> Optional[tuple[float, float, str]]:
    """Parse one OKX ``trades`` data entry → ``(price, size, side)``."""
    price = _coerce_float(row.get("px"))
    size = _coerce_float(row.get("sz"))
    side = row.get("side")
    if price is None or size is None or not isinstance(side, str):
        return None
    if price <= 0 or size <= 0:
        return None
    return price, size, side.upper()


# ---------------------------------------------------------------------------
# ReplayPublicStream
# ---------------------------------------------------------------------------


class ReplayPublicStream:
    """Replay-time stand-in for ``OkxPublicStream`` / equivalents.

    Construction surface mirrors what production exchange-public
    streams accept:

    * ``on_bbo`` — called synchronously for every parsed BBO update.
    * ``on_trade`` — optional; called for every parsed trade print.

    Lifecycle methods (``start``/``stop``/``request_reconnect``) are
    no-ops — replay is driver-pulled, not thread-pushed.

    Use ``deliver(event)`` from the replay driver to feed one
    recorded event in. ``deliver_all(stream)`` is a convenience that
    iterates an event source end-to-end. Both update the counters
    below for the final report.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        on_bbo: Optional[Callable[[BestBidAsk], None]] = None,
        on_trade: Optional[Callable[[float, float, str, str], None]] = None,
        okx_symbol: str = "TON-USDT-SWAP",
        binance_symbol: str = "TONUSDT",
    ) -> None:
        self._clock = clock
        self._on_bbo = on_bbo
        self._on_trade = on_trade
        self._okx_symbol = okx_symbol
        self._binance_symbol = binance_symbol

        # Counters (for the Phase 4 report)
        self.events_seen = 0
        self.bbo_dispatched = 0
        self.trades_dispatched = 0
        self.events_skipped = 0
        # BUG-E: per-reason breakdown of events_skipped so the operator
        # can tell benign channel/ack filtering from real event masking.
        self.skip_reasons: dict[str, int] = {}
        # Track the latest parsed BBO per venue (lets the driver
        # forward the current touch to the paper executor without
        # holding state itself).
        self.last_okx_bba: Optional[BestBidAsk] = None
        self.last_binance_bba: Optional[BestBidAsk] = None

    # --- production-surface no-ops ---------------------------------------

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def request_reconnect(self) -> None:
        return None

    def _skip(self, reason: str) -> None:
        """Count one skipped event under a categorized reason (BUG-E).

        ``events_skipped`` stays the grand total; ``skip_reasons``
        breaks it down so the operator can tell benign filtering
        (subscribe acks, off-symbol channels) from a real
        event-masking bug at a glance.
        """
        self.events_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    # --- replay-driver surface -------------------------------------------

    def deliver(self, event: RecordedEvent) -> None:
        """Parse + dispatch one recorded event."""
        self.events_seen += 1
        if event.source == "okx_public":
            self._deliver_okx_public(event)
        elif event.source == "binance_public":
            self._deliver_binance_public(event)
        else:
            self._skip(f"unknown_source:{event.source}")

    def _deliver_okx_public(self, event: RecordedEvent) -> None:
        msg = event.msg
        if not isinstance(msg, dict):
            self._skip("okx_non_dict_msg")
            return
        arg = msg.get("arg") or {}
        channel = arg.get("channel") if isinstance(arg, dict) else None
        # Subscribe acks have no "data" key — benign.
        if "data" not in msg:
            self._skip("okx_subscribe_ack")
            return
        data = msg.get("data")
        if not isinstance(data, list):
            self._skip("okx_non_list_data")
            return
        if channel in ("bbo-tbt", "books5", "books-l2-tbt", "books50-l2-tbt"):
            for row in data:
                if not isinstance(row, dict):
                    continue
                bba = _parse_okx_book_data_row(row, symbol=self._okx_symbol)
                if bba is None:
                    continue
                self.last_okx_bba = bba
                if self._on_bbo is not None:
                    self._on_bbo(bba)
                self.bbo_dispatched += 1
        elif channel == "trades":
            for row in data:
                if not isinstance(row, dict):
                    continue
                parsed = _parse_okx_trade_row(row)
                if parsed is None:
                    continue
                price, size, side = parsed
                if self._on_trade is not None:
                    self._on_trade(price, size, side, "okx")
                self.trades_dispatched += 1
        else:
            self._skip(f"okx_unknown_channel:{channel}")

    def _deliver_binance_public(self, event: RecordedEvent) -> None:
        msg = event.msg
        if not isinstance(msg, dict):
            self._skip("binance_non_dict_msg")
            return
        # Binance bookTicker carries 'e':'bookTicker'.
        if msg.get("e") == "bookTicker":
            bba = _parse_binance_book_ticker(msg, symbol=self._binance_symbol)
            if bba is None:
                self._skip("binance_unparseable")
                return
            self.last_binance_bba = bba
            if self._on_bbo is not None:
                self._on_bbo(bba)
            self.bbo_dispatched += 1
        else:
            self._skip(f"binance_unknown_event:{msg.get('e')}")


# ---------------------------------------------------------------------------
# ReplayPrivateStream
# ---------------------------------------------------------------------------


class ReplayPrivateStream:
    """Replay-time stand-in for OKX private WS.

    Pushes recorded order-update / fill events into ``event_sink``
    (the bot's private event queue). The bot drains the queue at
    each tick.

    For the paper-only mode (no real Bot wired in), the sink is
    typically the same queue used by :class:`~app.backtest.paper_executor.PaperExecutor`
    so the driver can inspect both synthetic and recorded events
    together.

    ``forward_to_sink`` (default ``True``) gates whether parsed events
    are actually pushed onto ``event_sink``. In ``--with-bot`` mode the
    real :class:`~app.bot.Bot` drains the SAME queue, and its
    ``PaperExecutor`` is the authoritative source of private events for
    the bot's OWN simulated orders. Replaying the ORIGINAL live
    session's recorded order-updates/fills into that queue makes the
    bot ingest foreign-account state — bumping its shadow position
    against paper-exec's true (0-based, bot-only) position. That fires
    ``shadow_position_divergence`` → ``private_ws_overflow_recovery`` →
    reconcile churn / orphan-cancel / side-unresolved (the v1.5.294
    audit's "warning storm"). ``build_bot_runner`` therefore sets
    ``forward_to_sink = False`` so the recorded private stream is
    still parsed + counted (for the driver report) but never
    contaminates the bot's queue. Paper-only mode keeps it ``True``.

    OKX private message shapes vary by channel; we accept the
    following normalised forms (which the recorder emits raw):

    * orders channel: each ``data[]`` entry has ``ordId``, ``state``,
      ``side``, ``px`` / ``sz`` / ``accFillSz``, ``cTime``, ``clOrdId``.
    * fills channel: each ``data[]`` entry has ``billId``, ``ordId``,
      ``fillPx``, ``fillSz``, ``side``, ``ts``, ``fee``, ``pnl``.

    Unknown channels are counted as skips and not propagated.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        event_sink: "Queue[Any]",
        okx_symbol: str = "TON-USDT-SWAP",
        forward_to_sink: bool = True,
    ) -> None:
        self._clock = clock
        self._sink = event_sink
        self._symbol = okx_symbol
        # See class docstring: False in --with-bot mode so recorded
        # private events don't contaminate the bot's shadow position.
        self.forward_to_sink = forward_to_sink
        self.events_seen = 0
        self.order_updates_dispatched = 0
        self.fills_dispatched = 0
        # When forward_to_sink is False these count the events that
        # WOULD have been delivered, for the driver's report.
        self.order_updates_suppressed = 0
        self.fills_suppressed = 0
        self.events_skipped = 0
        # BUG-E: per-reason breakdown. The recorded private feed carries
        # account / positions / balance channels we don't replay; this
        # lets the operator confirm a large events_skipped is benign
        # off-channel filtering (e.g. unknown_channel:account), not a
        # bug dropping real order/fill events.
        self.skip_reasons: dict[str, int] = {}

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def request_reconnect(self) -> None:
        return None

    def _skip(self, reason: str) -> None:
        """Count one skipped event under a categorized reason (BUG-E)."""
        self.events_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def deliver(self, event: RecordedEvent) -> None:
        self.events_seen += 1
        if event.source != "okx_private":
            self._skip(f"wrong_source:{event.source}")
            return
        msg = event.msg
        if not isinstance(msg, dict):
            self._skip("non_dict_msg")
            return
        if "data" not in msg:
            self._skip("subscribe_ack")
            return
        arg = msg.get("arg") or {}
        channel = arg.get("channel") if isinstance(arg, dict) else None
        data = msg.get("data")
        if not isinstance(data, list):
            self._skip("non_list_data")
            return
        if channel == "orders":
            for row in data:
                if not isinstance(row, dict):
                    continue
                ev = self._parse_okx_order_row(row)
                if ev is None:
                    continue
                if self.forward_to_sink:
                    self._sink.put(ev)
                    self.order_updates_dispatched += 1
                else:
                    self.order_updates_suppressed += 1
        elif channel == "fills":
            for row in data:
                if not isinstance(row, dict):
                    continue
                ev = self._parse_okx_fill_row(row)
                if ev is None:
                    continue
                if self.forward_to_sink:
                    self._sink.put(ev)
                    self.fills_dispatched += 1
                else:
                    self.fills_suppressed += 1
        else:
            self._skip(f"unknown_channel:{channel}")

    def _parse_okx_order_row(
        self, row: dict[str, Any]
    ) -> Optional[PrivateOrderUpdateEvent]:
        oid_raw = row.get("ordId")
        try:
            oid = int(oid_raw)
        except (TypeError, ValueError):
            return None
        state = row.get("state", "")
        side_raw = row.get("side", "")
        side = "B" if side_raw == "buy" else ("A" if side_raw == "sell" else side_raw)
        px = _coerce_float(row.get("px"))
        sz = _coerce_float(row.get("sz"))
        acc = _coerce_float(row.get("accFillSz")) or 0.0
        if px is None or sz is None:
            return None
        remaining = max(0.0, sz - acc)
        try:
            ts_ms = int(row.get("uTime") or row.get("cTime") or 0)
        except (TypeError, ValueError):
            ts_ms = 0
        return PrivateOrderUpdateEvent(
            oid=oid,
            coin=self._symbol,
            status=state,
            status_timestamp_ms=ts_ms,
            side=side,
            limit_px=px,
            remaining_sz=remaining,
            orig_sz=sz,
            raw_status=state,
            cloid=row.get("clOrdId") or None,
        )

    def _parse_okx_fill_row(
        self, row: dict[str, Any]
    ) -> Optional[PrivateFillEvent]:
        oid_raw = row.get("ordId")
        try:
            oid: Optional[int] = int(oid_raw) if oid_raw is not None else None
        except (TypeError, ValueError):
            oid = None
        fill_id = row.get("billId") or row.get("tradeId") or ""
        if not isinstance(fill_id, str):
            fill_id = str(fill_id)
        fill_px = _coerce_float(row.get("fillPx"))
        fill_sz = _coerce_float(row.get("fillSz"))
        if fill_px is None or fill_sz is None:
            return None
        side_raw = row.get("side", "")
        side = "B" if side_raw == "buy" else ("A" if side_raw == "sell" else side_raw)
        try:
            ts_ms = int(row.get("ts") or 0)
        except (TypeError, ValueError):
            ts_ms = 0
        # OKX fee convention: positive = rebate received, negative =
        # fee paid. The OKX adapter negates so the bot sees the
        # bot-canonical convention (positive = cost). Replay stays in
        # the canonical convention because that's what consumers expect.
        fee_raw = _coerce_float(row.get("fee")) or 0.0
        fee_canonical = -fee_raw  # mirror OKX adapter normalisation
        closed_pnl = _coerce_float(row.get("pnl")) or 0.0
        return PrivateFillEvent(
            fill_id=fill_id,
            oid=oid,
            coin=self._symbol,
            px=fill_px,
            sz=fill_sz,
            side=side,
            time_ms=ts_ms,
            fee=fee_canonical,
            closed_pnl=closed_pnl,
            crossed=row.get("execType") == "T",  # taker
            is_snapshot=False,
            raw=row,
        )


__all__ = ["ReplayPublicStream", "ReplayPrivateStream"]
