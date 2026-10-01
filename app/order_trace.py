"""Per-order lifecycle trace ring buffer.

Plan reference: ``BUGS/`` Phase 2 of the OKX gone_on_exchange diagnostic.

Records every order's full lifecycle so the operator can answer, per
event, the question: *"why did this order go gone_on_exchange?"* The
three branches that look identical without per-order tracking:

  1. **Post-only cross at submit** — venue rejected at place time. The
     local state machine SHOULD transition to REJECTED on the place
     response. If we instead see a `gone_on_exchange` later, it means
     the rejection wasn't caught and reconcile cleaned up. Trace
     reveals: ``place_outcome="exchange_rejected"`` but no terminal
     transition recorded → state-machine bug.

  2. **WS event arrived but state didn't apply it before reconcile
     fired** — a race. Trace reveals: ``ws_events`` contains
     ``{state: "canceled"}`` whose ts predates the reconcile call,
     yet ``terminal_reason="gone_on_exchange"``.

  3. **WS event genuinely missing** — connectivity / queue drop /
     order-channel filter mismatch. Trace reveals: ``ws_events`` is
     empty, ``place_outcome="accepted"``, no terminal transition.

Phase 1 (in v1.1.42 ``okx_ws.py``) confirmed the OKX WS is healthy at
the socket level — 1500+ msgs/33min, 5 ms pong RTT. So branch 3 is
unlikely; branches 1 and 2 are the candidates. Phase 2 distinguishes
them.

Bounded memory: the buffer is a ring of fixed size (default 200
entries). Memory cost: ~50 KB max. Old entries are evicted FIFO; we
never need history older than ~5 minutes for live diagnostics, and
the snapshot pull captures whatever is in the buffer at the moment
the operator runs ``ops.ps1 ... snapshot``.

Threading: every public method takes the buffer's internal lock.
Callers (the bot tick thread, the WS thread, the reconcile path) can
record concurrently without coordination.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app import clock as _clock


# Ring-buffer cap. 5000 entries covers ~1.5-2 hours at typical
# 50 places/min cadence — enough that a multi-hour session's
# gone_on_exchange events stay in the trace through publisher
# pickup. Old entries are evicted FIFO. ~5 KB per entry × 5000 ≈
# 25 MB worst case; trivial against the rest of BotState.
#
# Bumped 2026-05-15 from 200 after the 260515-095056 snapshot
# rolled out all 56 gone_on_exchange events before they could be
# analysed. At 200 entries the buffer captured only the most-recent
# happy-path orders.
_DEFAULT_MAX_ENTRIES = 5000


def _utc_now_iso() -> str:
    return _clock.now_utc().isoformat()


@dataclass
class OrderTraceWsEvent:
    """One row received on the OKX private orders channel for this oid.

    Captures (ts, state) so the trace can answer: "did we receive a
    cancel event from the exchange before reconcile fired?" The
    timeline is what matters; we don't need the full row payload.
    """

    ts: str
    state: str  # OKX state value verbatim: "live", "partially_filled", "filled", "canceled", "mmp_canceled"


@dataclass
class OrderTraceEntry:
    """Full lifecycle of one order, from place dispatch through terminal.

    Field order roughly chronological so the JSON dump is readable
    without reordering. ``Optional`` fields are None until the
    relevant lifecycle event lands.
    """

    # Identifiers (always set at begin_order)
    client_order_id: str
    side: str  # "BUY" / "SELL"
    price: float
    size_base: float

    # Set when the bot begins constructing the place request.
    ts_request_sent: str = ""

    # Set when the place-response interpretation completes.
    ts_place_response: Optional[str] = None
    place_outcome: Optional[str] = None  # "accepted" / "exchange_rejected" / "transport_rejected" / "unconfirmed"
    place_outcome_detail: Optional[str] = None  # post_only_would_cross details, sCode, etc.
    order_id_exchange: Optional[str] = None

    # Each WS row received on the orders channel for this oid.
    # Append-only: the timeline is the data.
    ws_events: list[OrderTraceWsEvent] = field(default_factory=list)

    # Terminal: set when the local state machine transitions the order
    # to a terminal status (FILLED / CANCELED / REJECTED / DESYNC).
    # ``terminal_source`` is the layer that drove the transition:
    #   "place_response" — venue rejected at submit
    #   "ws"             — WS event drove the local terminal
    #   "reconcile"      — REST reconcile / lookup drove it (THE branch
    #                      we're investigating: gone_on_exchange path)
    ts_terminal_local: Optional[str] = None
    terminal_status: Optional[str] = None
    terminal_reason: Optional[str] = None
    terminal_source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "order_id_exchange": self.order_id_exchange,
            "side": self.side,
            "price": self.price,
            "size_base": self.size_base,
            "ts_request_sent": self.ts_request_sent,
            "ts_place_response": self.ts_place_response,
            "place_outcome": self.place_outcome,
            "place_outcome_detail": self.place_outcome_detail,
            "ws_events": [
                {"ts": e.ts, "state": e.state} for e in self.ws_events
            ],
            "ws_event_count": len(self.ws_events),
            "ts_terminal_local": self.ts_terminal_local,
            "terminal_status": self.terminal_status,
            "terminal_reason": self.terminal_reason,
            "terminal_source": self.terminal_source,
        }


class OrderTraceBuffer:
    """Thread-safe bounded ring of OrderTraceEntry, indexed by both
    client-order-id (known at place time) and order-id-exchange (known
    after place response).

    The two-index design matters: WS events arrive carrying ONLY the
    exchange oid, but reconcile and place-response paths know the
    cloid. Either path can find the same entry.

    On eviction (deque maxlen reached), the oldest entry is also
    purged from both indexes so the dicts stay bounded. The eviction
    is O(1) for the deque + O(1) per index removal.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._lock = threading.Lock()
        self._entries: deque[OrderTraceEntry] = deque(maxlen=max_entries)
        self._by_cloid: dict[str, OrderTraceEntry] = {}
        self._by_oid: dict[str, OrderTraceEntry] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def begin_order(
        self,
        *,
        client_order_id: str,
        side: str,
        price: float,
        size_base: float,
    ) -> None:
        """Record a new order at the moment the bot dispatches it to
        the venue. Idempotent on cloid: re-calling with the same cloid
        is a no-op (avoids duplicate entries when the place path retries
        after a transient transport blip).
        """
        if not client_order_id:
            return
        ts = _utc_now_iso()
        with self._lock:
            if client_order_id in self._by_cloid:
                return
            # Evict the oldest entry's index keys before deque eviction
            # happens implicitly on append.
            if len(self._entries) >= self._entries.maxlen:
                oldest = self._entries[0]
                self._by_cloid.pop(oldest.client_order_id, None)
                if oldest.order_id_exchange is not None:
                    self._by_oid.pop(oldest.order_id_exchange, None)
            entry = OrderTraceEntry(
                client_order_id=client_order_id,
                side=side,
                price=price,
                size_base=size_base,
                ts_request_sent=ts,
            )
            self._entries.append(entry)
            self._by_cloid[client_order_id] = entry

    def record_place_response(
        self,
        *,
        client_order_id: str,
        outcome: str,
        detail: str,
        order_id_exchange: Optional[int],
    ) -> None:
        """Stamp the place-response result onto the existing entry. If
        no entry exists for this cloid (begin_order missed), do nothing
        — better silent-drop than a partial entry that confuses the
        diagnosis.
        """
        if not client_order_id:
            return
        ts = _utc_now_iso()
        with self._lock:
            entry = self._by_cloid.get(client_order_id)
            if entry is None:
                return
            entry.ts_place_response = ts
            entry.place_outcome = outcome
            entry.place_outcome_detail = (detail or "")[:400]
            if order_id_exchange is not None:
                ex_str = str(order_id_exchange)
                entry.order_id_exchange = ex_str
                self._by_oid[ex_str] = entry

    def record_ws_event(
        self,
        *,
        order_id_exchange: Optional[int],
        state: str,
    ) -> None:
        """Append a WS orders-channel event to the matching entry's
        timeline. Lookup is by order_id_exchange (the cloid is on the
        WS row too, but oid is the canonical join key for OKX
        orders-channel rows). If no entry matches (e.g. an order placed
        before the buffer started, or already evicted), we silently
        drop — the entry would be incomplete anyway.
        """
        if order_id_exchange is None or order_id_exchange == 0:
            return
        ts = _utc_now_iso()
        ex_str = str(order_id_exchange)
        with self._lock:
            entry = self._by_oid.get(ex_str)
            if entry is None:
                return
            entry.ws_events.append(OrderTraceWsEvent(ts=ts, state=state))

    def record_terminal(
        self,
        *,
        order_id_exchange: Optional[int] = None,
        client_order_id: Optional[str] = None,
        status: str,
        reason: str,
        source: str,
    ) -> None:
        """Mark the entry as terminated. ``source`` is the layer that
        drove the transition: ``"place_response"``, ``"ws"``, or
        ``"reconcile"``. The reconcile branch is the one we want to
        catch — paired with an empty ws_events list, that's the bug
        signature.

        Re-calling on an already-terminal entry is a no-op (preserves
        the FIRST terminal transition; later cleanup paths shouldn't
        overwrite the original cause).
        """
        with self._lock:
            entry: Optional[OrderTraceEntry] = None
            if order_id_exchange is not None and order_id_exchange != 0:
                entry = self._by_oid.get(str(order_id_exchange))
            if entry is None and client_order_id:
                entry = self._by_cloid.get(client_order_id)
            if entry is None:
                return
            if entry.terminal_status is not None:
                return
            entry.ts_terminal_local = _utc_now_iso()
            entry.terminal_status = status
            entry.terminal_reason = (reason or "")[:400]
            entry.terminal_source = source

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict[str, Any]]:
        """Snapshot the current buffer as a list of dicts, oldest first.
        Safe to call from any thread; takes the lock once and copies
        out under it. Used by the HTTP /orders/lifecycle-trace endpoint
        and indirectly by ops.ps1 ... snapshot.
        """
        with self._lock:
            return [e.to_dict() for e in self._entries]

    def was_recently_terminal(
        self,
        *,
        order_id_exchange: Optional[int] = None,
        client_order_id: Optional[str] = None,
        max_age_seconds: float = 60.0,
    ) -> bool:
        """1.3.85 hydration race guard.

        Returns True when this (oid, cloid) had a local terminal
        transition OR a terminal-state WS event within the last
        ``max_age_seconds``. Distinct from ``has_pending_ws_terminal``:
        that one tests for a not-yet-drained event still in flight;
        THIS one tests for a confirmed terminal that the bot has
        already processed.

        Used by ``_hydrate_working_from_exchange`` to skip hydrating
        an exchange-only order that the bot KNOWS already terminated
        locally — the REST snapshot is just stale. Distinguishes
        WS-vs-REST race (benign) from a genuine orphan (bot really
        didn't know about this order).

        Either ``order_id_exchange`` or ``client_order_id`` must be
        provided. Returns False on unknown identifiers or if the
        matching entry has no terminal record / no terminal WS event
        within the window.
        """
        if not order_id_exchange and not client_order_id:
            return False
        with self._lock:
            entry: Optional[OrderTraceEntry] = None
            if order_id_exchange is not None and order_id_exchange != 0:
                entry = self._by_oid.get(str(order_id_exchange))
            if entry is None and client_order_id:
                entry = self._by_cloid.get(client_order_id)
            if entry is None:
                return False
            now = _clock.now_utc()
            # Path A: bot already recorded a local terminal transition
            # for this entry. If it was recent, skip hydration.
            if entry.ts_terminal_local:
                try:
                    t_ts = datetime.fromisoformat(entry.ts_terminal_local)
                    if (now - t_ts).total_seconds() <= max_age_seconds:
                        return True
                except ValueError:
                    pass
            # Path B: the most recent WS event was a terminal state.
            # Catches the case where WS announced the terminal but the
            # local state-machine hasn't recorded ts_terminal_local yet
            # (it does, but defensive parity with has_pending_ws_terminal).
            if entry.ws_events:
                last = entry.ws_events[-1]
                if last.state in ("canceled", "filled", "mmp_canceled"):
                    try:
                        e_ts = datetime.fromisoformat(last.ts)
                        if (now - e_ts).total_seconds() <= max_age_seconds:
                            return True
                    except ValueError:
                        pass
            return False

    def has_pending_ws_terminal(
        self,
        order_id_exchange: Optional[int],
        *,
        ack_ts: Optional[datetime] = None,
        max_age_s: float = 5.0,
    ) -> bool:
        """True if the most recent WS event for this oid is a terminal
        state (``canceled`` / ``filled`` / ``mmp_canceled``) AND that
        event is plausibly still un-drained.

        Used by ``_reconcile_side`` to suppress a ``gone_on_exchange``
        decision when the WS thread already received a terminal event
        for the order — the event is sitting in the inbound queue
        waiting for the next tick's drain, and reconcile would
        otherwise race.

        Two staleness guards (1.1.45 fix for the watchdog deadlock seen
        in snapshot 260508105856):

        * ``ack_ts`` — when set, a WS event whose timestamp predates
          the working order's ACK is ignored. Catches the *re-hydrate*
          scenario: an oid the bot lost track of and re-discovered via
          ``order_hydrated_from_exchange`` carries a fresh ack from
          the hydrate, but the trace may still hold a "canceled" event
          from the order's PRIOR lifecycle — that stale event must not
          gate the new lifecycle. Without this guard, reconcile defers
          forever and the watchdog (600 s execution-idle) eventually
          kills the bot.

        * ``max_age_s`` — even with no ack_ts hint, a WS terminal event
          older than this window has almost certainly already been
          processed by drain (drain runs every 0.5 s tick). If reconcile
          is firing on this oid and the event is N seconds old, it
          isn't a race. Default 5 s = 10 ticks of drain headroom.

        For the original race fix (terminal event arrived ~10 ms before
        reconcile fires), both guards pass: event is newer than ack
        AND within 5 s. So the original protection still holds.

        Returns False for unknown oids, missing entries, empty
        ws_events lists, non-terminal last states, or stale events.
        """
        if order_id_exchange is None or order_id_exchange == 0:
            return False
        with self._lock:
            entry = self._by_oid.get(str(order_id_exchange))
            if entry is None or not entry.ws_events:
                return False
            last = entry.ws_events[-1]
            if last.state not in ("canceled", "filled", "mmp_canceled"):
                return False
            try:
                event_ts = datetime.fromisoformat(last.ts)
            except ValueError:
                return False
            now = _clock.now_utc()
            # Stale-by-age: drain has had plenty of time to process.
            if (now - event_ts).total_seconds() > max_age_s:
                return False
            # Stale-by-lifecycle: event predates the current ack.
            if ack_ts is not None:
                ack_aware = (
                    ack_ts
                    if ack_ts.tzinfo is not None
                    else ack_ts.replace(tzinfo=timezone.utc)
                )
                if event_ts < ack_aware:
                    return False
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
