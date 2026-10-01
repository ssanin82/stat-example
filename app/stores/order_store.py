"""v1.4.80 wedge-elimination-cleanup Phase 3A — OrderStore.

Single owner of working-order state. Replaces direct
``BotState._working_orders`` access. Maintains ``(oid, cloid)``
indexes for O(1) lookup (vs the O(n) scan used pre-Phase-3A).

Design notes
============

**Storage is delegated, not duplicated.** ``OrderStore`` does NOT
own the underlying ``_working_orders`` dict — it shares a reference
with ``BotState``. This is the additive-first migration strategy
described in ``plans/wedge-elimination-cleanup.md`` Phase 3:

* Phase 3A (this file): build the new API on top of the existing
  dict. Callers can use either path. Indexes are maintained
  alongside the dict via the ``set`` / ``delete`` API. Direct
  dict mutations bypass the indexes but still work — index
  lookups can miss those entries, callers fall back to scan.
* Phase 3D: rewrite the ownership so the dict lives INSIDE
  ``OrderStore`` and ``BotState`` exposes a property. Direct
  access is removed.

This separation lets us land the new API without coordinating a
flag-day migration of all 81 ``BotState`` callers.

**Indexes are best-effort.** Index lookups (``find_by_oid``,
``find_by_cloid``) are O(1) when the entry was inserted via the
new API, and degrade to a full scan when the entry was inserted
via direct dict mutation. The scan-fallback is correct but slow;
in production hot paths we should reach the indexed path because
all Phase 1B+ code uses ``set`` / ``transition``.

**Locking.** The store reuses ``state._lock`` rather than holding
its own. Phase 3B splits per-store locks once the other stores
land; for Phase 3A we keep semantics identical to the god-class
to minimize regression surface.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from app.enums import OrderStatus, Side
from app.models import WorkingOrder

if TYPE_CHECKING:
    from app.state import BotState

logger = logging.getLogger(__name__)


# v1.5.206 — Tombstone cache. When a WO is removed from the live
# index (terminal status transition, hydration purge, etc.), we
# stash the IDs + reason for a short window so the amend-response
# handler can recognise "this amend's response is missing because
# the order is already gone locally" → no kill, no panic.
#
# Bounded by maxlen on the deque (200 entries). For a bot doing
# <10 orders/sec this covers >20 seconds of history — well past any
# realistic amend-RTT (~10s max). For higher cadences we'd want
# time-based eviction; the snapshot-driving cadence currently is
# 2 quote cycles/sec ≈ ~4 orders/sec.
_TOMBSTONE_MAX = 200
# Cap on tombstone age (seconds) the amend-response handler accepts.
# Older than this: the cache hit is treated as not-found because the
# bot is in a different epoch and an "unconfirmed amend" really is
# anomalous. Tuned to comfortably exceed amend RTT P99 + venue
# rate-limit retry windows.
_TOMBSTONE_MAX_AGE_SECONDS = 30.0


@dataclass(frozen=True)
class _TombstoneEntry:
    """One entry in the recently-removed-orders cache.

    Read by the amend-response handler when it gets an unconfirmed
    response for a WO that the venue's batch reply didn't echo. If
    the WO's ord_id matches a tombstone less than
    ``_TOMBSTONE_MAX_AGE_SECONDS`` old, the missing response is
    treated as benign (the order is gone locally, the amend is moot).
    """

    order_id_exchange: Optional[int]
    client_order_id: Optional[str]
    removed_at_mono: float
    removed_reason: str   # e.g. "transitioned_to_CANCELED"


# Terminal statuses for the order-state contract. Mirrors
# ``app/reconciler.py:_TERMINAL`` — kept synced to that set.
_TERMINAL_STATUSES = frozenset(
    {
        OrderStatus.CANCELED,
        OrderStatus.FILLED,
        OrderStatus.REJECTED,
        OrderStatus.DESYNC,
    }
)


class OrderStore:
    """Working-order storage facade.

    Phase 3A — additive thin wrapper over ``BotState._working_orders``.
    Provides the ``OrderStore`` API contract documented in
    ``plans/wedge-elimination-cleanup.md`` Phase 3. Direct
    ``state._working_orders`` access remains valid until Phase 3D.

    Indexes (Phase 1B.3 promised these as O(1) but the v1.4.69
    implementation was O(n) scan — Phase 3A fulfills the original
    promise):

    * ``_wo_by_oid``: ``int (oid) → WorkingOrder``
    * ``_wo_by_cloid``: ``str (cloid) → WorkingOrder``

    Both indexes are maintained on ``set``, ``delete``,
    ``transition``, and ``rebuild_indexes`` (called once on
    construction to pick up any WOs already in state).
    """

    def __init__(self, state: "BotState") -> None:
        # Hold a reference to the BotState so we can read/write the
        # underlying _working_orders dict. Cyclic ref is intentional
        # and bounded by process lifetime.
        self._state = state
        # Indexes. Populated by ``rebuild_indexes`` immediately
        # below; updated on every ``set`` / ``delete`` / ``transition``.
        self._wo_by_oid: dict[int, WorkingOrder] = {}
        self._wo_by_cloid: dict[str, WorkingOrder] = {}
        # v1.5.206 — recently-removed-WO tombstone cache. Every
        # ``_index_remove`` appends an entry. Used by
        # ``was_recently_removed`` to recognise amend responses that
        # missed because the order was already gone locally — closes
        # the race that killed v1.5.204-260528-072247 with
        # ``amend_response_unconfirmed (missing_row_for_amend)``.
        self._recently_removed: deque[_TombstoneEntry] = deque(
            maxlen=_TOMBSTONE_MAX,
        )
        self.rebuild_indexes()

    # ------------------------------------------------------------------
    # Index maintenance
    # ------------------------------------------------------------------

    def rebuild_indexes(self) -> None:
        """Wipe and re-populate ``_wo_by_oid`` / ``_wo_by_cloid`` from
        the underlying dict. Called once at construction; can be
        called by callers that have bypassed the store API to
        re-sync. Idempotent.

        Skips terminal WOs (they should not be looked up — the
        Phase 1B.2 contract: hydration / find paths treat terminals
        as conceptually done).
        """
        self._wo_by_oid.clear()
        self._wo_by_cloid.clear()
        for wo in self._all_working_orders_unlocked():
            if wo is None:
                continue
            if wo.status in _TERMINAL_STATUSES:
                continue
            self._index_add(wo)

    def _index_add(self, wo: WorkingOrder) -> None:
        """Add ``wo`` to both indexes. Idempotent — re-adding the
        same WO is a no-op."""
        if wo.order_id_exchange:
            self._wo_by_oid[int(wo.order_id_exchange)] = wo
        if wo.client_order_id:
            self._wo_by_cloid[wo.client_order_id] = wo

    def _index_remove(
        self, wo: WorkingOrder, *, reason: str = "unspecified"
    ) -> None:
        """Remove ``wo`` from both indexes + push a tombstone.
        Idempotent.

        ``reason`` is for diagnostic logging only; common values:
        ``transitioned_to_CANCELED`` / ``transitioned_to_FILLED`` /
        ``transitioned_to_REJECTED`` / ``slot_clear`` /
        ``hydration_purge``. v1.5.206 — used by the amend-response
        handler's tombstone-lookup path.
        """
        if wo.order_id_exchange:
            existing = self._wo_by_oid.get(int(wo.order_id_exchange))
            if existing is wo:
                self._wo_by_oid.pop(int(wo.order_id_exchange), None)
        if wo.client_order_id:
            existing = self._wo_by_cloid.get(wo.client_order_id)
            if existing is wo:
                self._wo_by_cloid.pop(wo.client_order_id, None)
        # v1.5.206 — tombstone push. Cheap (O(1) deque append).
        try:
            self._recently_removed.append(
                _TombstoneEntry(
                    order_id_exchange=(
                        int(wo.order_id_exchange)
                        if wo.order_id_exchange
                        else None
                    ),
                    client_order_id=wo.client_order_id or None,
                    removed_at_mono=time.monotonic(),
                    removed_reason=str(reason)[:80],
                )
            )
        except (TypeError, ValueError):
            # Defensive — never let tombstone bookkeeping crash the
            # index-remove path. The downstream resilience handler is
            # a quality-of-life feature, not load-bearing.
            pass

    # ------------------------------------------------------------------
    # v1.5.206 — Tombstone read API for amend-response resilience.
    # ------------------------------------------------------------------

    def was_recently_removed(
        self,
        oid: Optional[int] = None,
        cloid: Optional[str] = None,
        *,
        now_mono: Optional[float] = None,
    ) -> Optional[_TombstoneEntry]:
        """Return the most-recent tombstone matching ``oid`` or ``cloid``
        whose age is under ``_TOMBSTONE_MAX_AGE_SECONDS``. ``None`` if
        no match (caller treats as "this is a real anomaly, escalate").

        OID match takes precedence (more specific); cloid is the
        fallback for the pre-ack window where OID isn't bound yet.

        Hot-path safety. Linear scan over a bounded deque (max 200);
        worst case ~200 dict ops, completes in microseconds. Called
        ONLY on the amend-response unconfirmed branch (rare path),
        not every tick.
        """
        if oid is None and not cloid:
            return None
        if now_mono is None:
            now_mono = time.monotonic()
        cutoff = now_mono - _TOMBSTONE_MAX_AGE_SECONDS
        match: Optional[_TombstoneEntry] = None
        # Iterate newest-first via reversed() so we pick the most
        # recent removal if there are duplicates (e.g. an OID was
        # tombstoned twice via different code paths).
        for entry in reversed(self._recently_removed):
            if entry.removed_at_mono < cutoff:
                # All older entries beyond this point — deque is
                # insertion-ordered, so we can stop.
                break
            if oid is not None and entry.order_id_exchange == int(oid):
                match = entry
                break
            if cloid and entry.client_order_id == cloid:
                match = entry
                break
        return match

    def _all_working_orders_unlocked(self) -> list[WorkingOrder]:
        """Iterate ``state._working_orders`` without acquiring the
        lock. Caller MUST hold ``state._lock``. Helper used by
        rebuild_indexes and other inside-lock paths.
        """
        wos = getattr(self._state, "_working_orders", None)
        if wos is None:
            return []
        out: list[WorkingOrder] = []
        for side_bucket in wos.values():
            for wo in side_bucket.values():
                if wo is not None:
                    out.append(wo)
        return out

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def get(self, side: Side, level_idx: int) -> Optional[WorkingOrder]:
        """Return the WO at ``(side, level_idx)`` slot, or ``None``."""
        with self._state._lock:
            return self._state.get_working_order(side, int(level_idx))

    def iter_side(self, side: Side) -> list[tuple[int, WorkingOrder]]:
        """``(level_idx, WorkingOrder)`` tuples for ``side``, sorted
        by ``level_idx`` ascending. Wraps the existing
        ``state.iter_working_orders``.
        """
        with self._state._lock:
            return self._state.iter_working_orders(side)

    def iter_all(self) -> list[WorkingOrder]:
        """All working orders across both sides + all rungs."""
        with self._state._lock:
            return self._state.all_working_orders()

    def find_by_oid(self, oid: Optional[int]) -> Optional[WorkingOrder]:
        """O(1) lookup by ``order_id_exchange``. Returns ``None``
        when ``oid`` is None/0 or not present in the index.

        Phase 1B.2 contract: terminal WOs (CANCELED / FILLED /
        REJECTED / DESYNC) are NOT lookup-targets. Both the index
        and the scan-fallback respect this.
        """
        if not oid:
            return None
        with self._state._lock:
            wo = self._wo_by_oid.get(int(oid))
            if wo is not None:
                return wo
            # Index miss — fall back to a scan in case a direct
            # dict mutation bypassed the index. Phase 3D removes
            # this fallback when all callers use the store API.
            # Filter out terminal WOs: they don't belong in the
            # index and shouldn't be returned to lookup callers.
            for cand in self._all_working_orders_unlocked():
                if cand.order_id_exchange == oid:
                    if cand.status in _TERMINAL_STATUSES:
                        return None
                    # Lazy-repair the index.
                    self._index_add(cand)
                    return cand
        return None

    def find_by_cloid(self, cloid: Optional[str]) -> Optional[WorkingOrder]:
        """O(1) lookup by ``client_order_id``. Returns ``None``
        when ``cloid`` is empty or not present in the index.

        Phase 1B.2 contract: terminal WOs are not lookup-targets.
        """
        if not cloid:
            return None
        with self._state._lock:
            wo = self._wo_by_cloid.get(cloid)
            if wo is not None:
                return wo
            for cand in self._all_working_orders_unlocked():
                if cand.client_order_id == cloid:
                    if cand.status in _TERMINAL_STATUSES:
                        return None
                    self._index_add(cand)
                    return cand
        return None

    def find_by_oid_or_cloid(
        self,
        oid: Optional[int],
        cloid: Optional[str],
    ) -> Optional[WorkingOrder]:
        """Combined lookup with OID priority. Used by hydration
        dedup (Phase 1B) and exchange-reconcile slot classification
        (Phase 1D).

        Legacy semantic (from the pre-Phase-3A
        ``OrderManager._find_local_wo_by_oid_or_cloid`` helper):

        1. OID match first (most specific).
        2. Cloid match ONLY against WOs that have no OID yet.
           This is the place-before-ack race window: a WO with cloid
           but no OID is the local representation of an order whose
           HTTP response hasn't bound the OID yet. Matching against
           OID-bound WOs would let a cloid collision resurface a
           stale WO.

        Step 1 is O(1) via the OID index. Step 2 is intentionally a
        targeted scan (filtered to no-OID WOs); since cloid-only WOs
        are rare (the race window is sub-millisecond), the scan is
        cheap in practice.
        """
        if oid is None and not cloid:
            return None
        if oid is not None:
            wo = self.find_by_oid(oid)
            if wo is not None:
                return wo
        if cloid:
            with self._state._lock:
                for cand in self._all_working_orders_unlocked():
                    if (
                        not cand.order_id_exchange
                        and cand.client_order_id == cloid
                    ):
                        return cand
        return None

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def set(
        self,
        side: Side,
        level_idx: int,
        wo: Optional[WorkingOrder],
    ) -> None:
        """Install or remove ``wo`` at the ``(side, level_idx)`` slot.

        ``wo=None`` deletes the slot.

        Maintains the OID + cloid indexes. If a different WO already
        occupies the slot, that WO is removed from the indexes first
        (so re-indexed lookups don't return a stale reference).
        """
        with self._state._lock:
            existing = self._state.get_working_order(side, int(level_idx))
            if existing is not None and existing is not wo:
                self._index_remove(existing, reason="slot_replaced")
            self._state.set_working_order(side, int(level_idx), wo)
            if wo is not None and wo.status not in _TERMINAL_STATUSES:
                self._index_add(wo)
            elif wo is None:
                # set(side, lvl, None) deletes — existing was removed above.
                pass
            else:
                # wo provided but terminal → don't index it.
                pass

    def delete(self, side: Side, level_idx: int) -> None:
        """Remove the WO at the slot. Equivalent to ``set(side, lvl, None)``."""
        self.set(side, int(level_idx), None)

    def transition(
        self,
        wo: WorkingOrder,
        new_status: OrderStatus,
        reason: Optional[str] = None,
    ) -> None:
        """Phase 0 contract: the SINGLE point of ``OrderStatus``
        mutation. Updates the WO in place, maintains the indexes,
        and (when the new status is terminal) removes the WO from
        the indexes so future lookups don't return a tombstone.

        Note: this is the NEW API. The legacy module-level
        ``app.execution.transition(wo, status, reason)`` function
        still exists and many call sites use it directly. Phase 3D
        migrates them. For now, callers who want index-coherent
        transitions should use this method.
        """
        # Use the same setter logic as the legacy transition helper.
        from app.utils.time import utc_now

        with self._state._lock:
            previous_status = wo.status
            wo.status = new_status
            if reason:
                wo.cancel_reason = reason
            now = utc_now()
            if new_status == OrderStatus.SENT:
                wo.ts_sent = now
            if new_status == OrderStatus.ACKED:
                wo.ts_ack = now
            if new_status in (
                OrderStatus.CANCELED,
                OrderStatus.FILLED,
                OrderStatus.REJECTED,
            ):
                wo.ts_closed = now
            # Update indexes. If we became terminal, drop from indexes.
            if new_status in _TERMINAL_STATUSES:
                if previous_status not in _TERMINAL_STATUSES:
                    self._index_remove(
                        wo, reason=f"transitioned_to_{new_status.value}"
                    )
            else:
                # Non-terminal: ensure indexes are populated. This
                # also handles the case where OID/cloid were bound
                # after construction.
                self._index_add(wo)

    # ------------------------------------------------------------------
    # Snapshot API (Phase 3C will return ImmutableOrderView)
    # ------------------------------------------------------------------

    def snapshot_by_slot(
        self,
        *,
        include_orphan_idx_gte: Optional[int] = None,
    ) -> dict[tuple[Side, int], WorkingOrder]:
        """Flat ``(side, level_idx) → WorkingOrder`` dict.

        Used by ``OrderManager._snapshot_working_orders_for_reconciler``
        and the Phase 1C ``working_orders_by_slot`` engine context
        field. Single state-lock acquire.

        ``include_orphan_idx_gte=N``: also include slot entries
        whose ``level_idx >= N`` (orphan rungs beyond the
        configured ladder). Set to the configured rung count to
        preserve the legacy semantic.
        """
        out: dict[tuple[Side, int], WorkingOrder] = {}
        with self._state._lock:
            for side in (Side.BUY, Side.SELL):
                for idx, wo in self._state.iter_working_orders(side):
                    if wo is None:
                        continue
                    out[(side, int(idx))] = wo
        return out

    def index_size(self) -> tuple[int, int]:
        """Returns ``(len(_wo_by_oid), len(_wo_by_cloid))`` for
        observability. Useful in tests to verify index growth /
        shrinkage."""
        return (len(self._wo_by_oid), len(self._wo_by_cloid))
