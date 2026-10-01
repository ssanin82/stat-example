"""
Outbound trading action dispatcher: micro-batch + per-side coalescing + cancel-first ordering.

Single transport thread owns enqueue → batch → execute callbacks (HTTP/WS inside client).
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

from app.config import Settings
from app.enums import Side

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlaceTransportIntent:
    """Worker-only: place / amend for an already-staged ``WorkingOrder``.

    ``kind="place"`` (default) — for ``status=SENT`` WOs going out
    via ``batch_place_post_only_limit`` / single ``place_post_only_limit``.

    ``kind="amend"`` (amend-prio Phase 2) — for ``status=AMEND_PENDING``
    WOs going out via ``amend_batch_orders``. The amend updates an
    EXISTING venue order (preserves ordId / queue position) rather
    than placing a new one. The dispatcher routes by ``kind`` to the
    appropriate executor callback. See ``plans/amend-prio.md`` Phase 2.
    """

    wo_order_id_local: str
    side: Side
    intent_seq: int
    quote_cycle_id: str
    enqueued_mono: float
    intent_created_perf: float = 0.0
    # 1.3.130 multi-rung Phase 2: which rung this place targets
    # (0 = inside, default). The dispatcher coalesces by (side,
    # level_idx) so two intents for different rungs on the same side
    # both reach the venue. At N=1 every intent has level_idx=0 and
    # the coalescing reduces to the pre-Phase-2 per-side semantics.
    level_idx: int = 0
    # amend-prio Phase 2 (v1.4.16): discriminator for place vs amend.
    # Legal values: ``"place"`` (default — fresh order via
    # ``batch_place``) and ``"amend"`` (in-place modify via
    # ``amend_batch_orders``). The coalesce key is
    # ``(side, level_idx, kind)`` so a place and an amend on the same
    # slot do not collapse — they target different orders (fresh vs
    # existing). In practice the bot only ever emits one ``kind`` at a
    # time per slot because the WO state machine gates which path is
    # eligible (SENT → place; ACKED → amend); the extended coalesce
    # key is a defence-in-depth invariant.
    kind: str = "place"


@dataclass(frozen=True, slots=True)
class CancelTransportIntent:
    """Worker-only: cancel for ``CANCEL_PENDING`` order."""

    wo_order_id_local: str
    side: Side
    intent_seq: int
    enqueued_mono: float
    # 1.3.130 multi-rung Phase 2: rung identifier — same role as on
    # PlaceTransportIntent. Per-(side, level_idx) coalescing.
    level_idx: int = 0


class OutboundDispatchCoordinator:
    """
    Two lanes: cancel (urgent) then post-only place. Coalesce per-side to newest intent.

    Same-side exclusivity (strategy-invariant: at most one live order per side per symbol)
    is enforced at three levels:

    1. ``OrderManager._stage_place_order_local`` refuses to stage a second same-side
       local working order while the existing slot is non-terminal.
    2. ``submit_place`` here coalesces same-side intents so only the newest one ever
       enters the lane.
    3. ``_flush_one_batch`` additionally checks ``_active_place_sides`` and will defer
       (re-enqueue) a place whose side is *already in-flight* this batch. This matters
       only if two independent place intents for the same side arrive within one flush
       due to external re-entrant submission; the lane coalesce would already have
       collapsed them for the normal path.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        execute_place: Callable[[PlaceTransportIntent], None],
        execute_cancel: Callable[[CancelTransportIntent], None],
        execute_cancel_batch: Optional[
            Callable[[list[CancelTransportIntent]], None]
        ] = None,
        execute_place_batch: Optional[
            Callable[[list[PlaceTransportIntent]], None]
        ] = None,
        # amend-prio Phase 2 (v1.4.16): amend executor callbacks.
        # The place lane carries amend intents (discriminator
        # ``kind="amend"``); the coordinator routes them to these
        # callbacks instead of ``execute_place`` / ``execute_place_batch``.
        # Both default None — when no amend producer is wired
        # (Phase 1 lifecycle layer only), the amend lane stays empty
        # and the dispatcher behaves bit-identical to pre-Phase-2.
        execute_amend: Optional[
            Callable[[PlaceTransportIntent], None]
        ] = None,
        execute_amend_batch: Optional[
            Callable[[list[PlaceTransportIntent]], None]
        ] = None,
        on_stats: Optional[Callable[[dict[str, float | int]], None]] = None,
    ) -> None:
        self._settings = settings
        self._execute_place = execute_place
        self._execute_cancel = execute_cancel
        # 1.4.0 cancel-prio Phase 1b: opportunistic batch-cancel.
        # When 2+ cancels for different sides queue in the same flush,
        # call this callback with the whole list instead of iterating
        # ``execute_cancel`` per intent. Saves ~1 HTTP RTT on full-
        # reprice cycles where BUY + SELL cancel simultaneously.
        # Optional — when None or batch_cancels_enabled=False, the
        # legacy per-intent path is used (back-compat default for
        # adapters that don't expose a batch-cancel endpoint).
        self._execute_cancel_batch = execute_cancel_batch
        # 1.4.4: opportunistic batch-PLACE. Mirror of execute_cancel_batch
        # for place intents. When 2+ places for different (side,
        # level_idx) keys queue in the same flush AND
        # ``batch_places_enabled=True``, dispatched as ONE
        # /api/v5/trade/batch-orders call instead of N single-order
        # /api/v5/trade/order calls. Two wins:
        #   1. Saves N-1 RTTs (cuts reprice latency, more importantly
        #      reduces the time window where one side has placed
        #      and the other hasn't).
        #   2. Taps the SEPARATE rate-limit pool for batch-orders
        #      (300/2s on OKX standard, vs 60/2s for the single-order
        #      endpoint) — directly addresses the v1.4.2 row-level
        #      rate-limit pressure that triggered this work-unit.
        # Optional, like execute_cancel_batch — when None or
        # ``batch_places_enabled=False`` the per-intent path is used.
        self._execute_place_batch = execute_place_batch
        # amend-prio Phase 2 (v1.4.16): amend lane callbacks. The
        # place-lane partition by ``intent.kind`` routes ``"amend"``
        # intents to these executors instead of place ones. Both are
        # optional — when None, intents with ``kind="amend"`` raise on
        # dispatch (a programming error; the WO state machine should
        # prevent them from being submitted unless ``execute_amend*``
        # is wired).
        self._execute_amend = execute_amend
        self._execute_amend_batch = execute_amend_batch
        self._on_stats = on_stats
        self._stop = threading.Event()
        self._cond = threading.Condition()
        self._cancel_lane: deque[CancelTransportIntent] = deque()
        self._place_lane: deque[PlaceTransportIntent] = deque()
        self._coalesced_place_count = 0
        self._coalesced_cancel_count = 0
        self._queue_hwm = 0
        # Legacy single worker — runs ``_loop`` / ``_flush_one_batch``
        # when parallel mode is OFF (the v1.3.108-and-earlier code path).
        self._thread: Optional[threading.Thread] = None
        # 1.3.109 Phase 3: parallel worker threads. ``_cancel_thread``
        # drains the cancel lane; ``_place_thread`` drains the place
        # lane. Both wake on the shared ``self._cond``. Allocated in
        # ``start()`` when ``outbound_cancel_worker_enabled`` is True.
        self._cancel_thread: Optional[threading.Thread] = None
        self._place_thread: Optional[threading.Thread] = None
        # Captured at ``start()`` time so a single coordinator instance
        # can't be half-in/half-out of parallel mode if the setting
        # changes mid-run.
        self._parallel_workers: bool = False
        self._last_batch_size = 0
        self._inflight = 0
        # 1.3.130 multi-rung Phase 2: same-side exclusivity tracking is
        # now keyed on (side, level_idx). At N=1 every intent has
        # level_idx=0 and the keys reduce to (side, 0) — preserves the
        # pre-Phase-2 "at most one place + one cancel per side" semantics.
        # At N>1 the inside (level=0) and outer (level=1) intents on the
        # same side are independent — both dispatched concurrently.
        self._active_place_sides: set[tuple[Side, int]] = set()
        self._active_cancel_sides: set[tuple[Side, int]] = set()
        # amend-prio Phase 2 (v1.4.16): per-(side, level_idx) tracking
        # of in-flight amend operations. The cross-lane defer matrix
        # (described in ``plans/amend-prio.md`` Phase 2 table) uses
        # this set to gate cancels: when a cancel arrives and an amend
        # for the same WO is in flight, the cancel defers (the amend
        # response will update wo.price/size — the cancel then fires
        # against the now-current order). Amends that arrive while a
        # same-slot cancel is in flight are DISCARDED (the WO is
        # about to disappear; the amend would have no target).
        self._active_amend_sides: set[tuple[Side, int]] = set()
        self._same_side_inflight_deferrals = 0
        # 1.3.109 Phase 3: count of CROSS-LANE deferrals — a place
        # for side S was held back because a cancel for side S was in
        # flight at submit time. Replaces the implicit cancel-before-
        # place ordering the single-worker mode achieved via sequential
        # execution. Surfaced via ``snapshot_stats`` for operator
        # visibility into how often the guard fires (should be roughly
        # one per full-reprice cycle under typical traffic).
        self._cross_lane_deferrals = 0
        # 1.4.0 cancel-prio Phase 1b: count of batched-cancel dispatches
        # surfaced via ``snapshot_stats``. Each entry counts the
        # WHOLE batch (so a 2-cancel batch increments by 1, not 2).
        self._batch_cancel_dispatch_count = 0
        # 1.4.4: parallel surface for batched-place dispatches.
        self._batch_place_dispatch_count = 0
        # amend-prio Phase 2 (v1.4.16): amend dispatch counters.
        # ``batch_amend_dispatch_count`` increments once per batch
        # call (matching the place / cancel pattern). The cross-lane
        # counters track defer / discard rates so the operator can
        # verify the matrix fires under real traffic.
        self._batch_amend_dispatch_count = 0
        self._amend_discarded_cancel_inflight_total = 0
        self._cancel_deferred_amend_inflight_total = 0

    def start(self) -> None:
        # 1.3.109 Phase 3: branch on ``outbound_cancel_worker_enabled``.
        # Default is True (parallel mode); set OUTBOUND_CANCEL_WORKER_
        # ENABLED=false to revert to the legacy single-worker path. The
        # flag is captured here (rather than read on each iteration) so
        # the worker topology is stable for the coordinator's lifetime.
        self._parallel_workers = bool(
            getattr(self._settings, "outbound_cancel_worker_enabled", True)
        )
        if self._parallel_workers:
            if self._cancel_thread is not None or self._place_thread is not None:
                return
            self._cancel_thread = threading.Thread(
                target=self._cancel_loop,
                name="mm-outbound-dispatch-cancel",
                daemon=True,
            )
            self._place_thread = threading.Thread(
                target=self._place_loop,
                name="mm-outbound-dispatch-place",
                daemon=True,
            )
            self._cancel_thread.start()
            self._place_thread.start()
        else:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._loop, name="mm-outbound-dispatch", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        # Join whichever worker(s) are running. Multiple ``stop`` calls
        # are safe; ``join`` on a completed thread is a no-op.
        for t in (self._thread, self._cancel_thread, self._place_thread):
            if t is not None:
                t.join(timeout=5.0)

    def submit_place(self, intent: PlaceTransportIntent) -> None:
        with self._cond:
            # 1.3.130 multi-rung Phase 2: coalesce per (side, level_idx)
            # rather than just per side. At N=1 every intent has
            # level_idx=0 → reduces to the pre-Phase-2 per-side coalescing
            # behaviour. At N=2 the inside (level=0) and outer (level=1)
            # intents on the same side are independent — both reach the
            # venue. The "at most one live order per side" invariant
            # mentioned in the class docstring relaxes to "at most one
            # live order per (side, rung)" under multi-rung.
            #
            # amend-prio Phase 2 (v1.4.16): coalesce key extended to
            # ``(side, level_idx, kind)``. A place and an amend on the
            # same (side, level_idx) refer to DIFFERENT orders — a
            # place creates a new ordId, an amend updates an existing
            # one — so they must not collapse. In normal flow only one
            # kind is ever pending per slot (the WO state machine
            # ensures it); the extended key is defence-in-depth.
            kind = str(getattr(intent, "kind", "place") or "place")
            key = (
                intent.side,
                int(getattr(intent, "level_idx", 0) or 0),
                kind,
            )
            same_key = [
                p for p in self._place_lane
                if (
                    p.side,
                    int(getattr(p, "level_idx", 0) or 0),
                    str(getattr(p, "kind", "place") or "place"),
                )
                == key
            ]
            if same_key:
                self._coalesced_place_count += len(same_key)
            self._place_lane = deque(
                p for p in self._place_lane
                if (
                    p.side,
                    int(getattr(p, "level_idx", 0) or 0),
                    str(getattr(p, "kind", "place") or "place"),
                )
                != key
            )
            self._place_lane.append(intent)
            self._bump_hwm()
        self._notify()

    def submit_cancel(self, intent: CancelTransportIntent) -> None:
        with self._cond:
            # 1.3.130 multi-rung Phase 2: coalesce per (side, level_idx).
            # Same reasoning as submit_place.
            key = (intent.side, int(getattr(intent, "level_idx", 0) or 0))
            same_rung = [
                c for c in self._cancel_lane
                if (c.side, int(getattr(c, "level_idx", 0) or 0)) == key
            ]
            if same_rung:
                self._coalesced_cancel_count += len(same_rung)
            self._cancel_lane = deque(
                c for c in self._cancel_lane
                if (c.side, int(getattr(c, "level_idx", 0) or 0)) != key
            )
            self._cancel_lane.append(intent)
            self._bump_hwm()
        self._notify()

    def _bump_hwm(self) -> None:
        d = len(self._cancel_lane) + len(self._place_lane)
        if d > self._queue_hwm:
            self._queue_hwm = d

    def _notify(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def queue_depth(self) -> int:
        return len(self._cancel_lane) + len(self._place_lane)

    def snapshot_stats(self) -> dict[str, float | int]:
        return {
            "action_dispatch_queue_depth": self.queue_depth(),
            "action_dispatch_queue_high_watermark": int(self._queue_hwm),
            "coalesced_intent_count": int(self._coalesced_place_count + self._coalesced_cancel_count),
            "avg_batch_size": float(self._last_batch_size),
            "same_side_inflight_deferrals": int(self._same_side_inflight_deferrals),
            # v1.4.40 BUG-025: surface the latch sets that gate
            # new dispatches. A non-empty set without active
            # in-flight work is the silent-wedge fingerprint —
            # downstream cleanup missed a release.
            "inflight_count": int(self._inflight),
            "active_place_sides_count": len(self._active_place_sides),
            "active_cancel_sides_count": len(self._active_cancel_sides),
            "active_amend_sides_count": len(self._active_amend_sides),
            "active_place_sides": sorted(
                f"{s.value}@{idx}" for s, idx in self._active_place_sides
            ),
            "active_cancel_sides": sorted(
                f"{s.value}@{idx}" for s, idx in self._active_cancel_sides
            ),
            "active_amend_sides": sorted(
                f"{s.value}@{idx}" for s, idx in self._active_amend_sides
            ),
            # 1.4.0 cancel-prio Phase 1b: count of batch-cancel
            # dispatches (each entry = one HTTP call carrying 2+
            # cancels). Useful for the operator to verify the
            # batching actually fires under real traffic.
            "batch_cancel_dispatch_count": int(self._batch_cancel_dispatch_count),
            # 1.4.4: parallel surface — number of batch-PLACE
            # dispatches. The dominant win on MM reprice cycles
            # (cuts N place RTTs to 1 batch RTT).
            "batch_place_dispatch_count": int(self._batch_place_dispatch_count),
            # 1.3.109 Phase 3: count of cross-lane deferrals (place
            # held back while a same-side cancel was in flight). Zero
            # in legacy single-worker mode (no parallel races). In
            # parallel mode, increments roughly once per full-reprice
            # cycle — the operator can use this to verify the cross-
            # lane guard fires (vs. silently letting same-side races
            # through, which would violate strategy invariants).
            "cross_lane_deferrals": int(self._cross_lane_deferrals),
            # amend-prio Phase 2 (v1.4.16): amend dispatch + cross-lane
            # counters. ``batch_amend_dispatch_count`` mirrors
            # batch_place_dispatch_count. The two cross-lane counters
            # surface how often the matrix's defer/discard rules
            # fire — non-zero values indicate genuine cancel/amend
            # races (expected at moderate frequency under reprice
            # bursts; sustained high values suggest a state-machine
            # bug elsewhere).
            "batch_amend_dispatch_count": int(self._batch_amend_dispatch_count),
            "amend_discarded_cancel_inflight_total": int(
                self._amend_discarded_cancel_inflight_total
            ),
            "cancel_deferred_amend_inflight_total": int(
                self._cancel_deferred_amend_inflight_total
            ),
        }

    def wait_until_idle(self, timeout_s: float) -> None:
        import time as time_mod

        deadline = time_mod.monotonic() + timeout_s
        while time_mod.monotonic() < deadline:
            if self.queue_depth() == 0 and self._inflight == 0:
                return
            time_mod.sleep(0.01)

    def _loop(self) -> None:
        while not self._stop.is_set():
            interval_ms = float(self._settings.action_batch_interval_ms)
            with self._cond:
                if interval_ms <= 0:
                    # Event-driven mode: no timeout batching delay, wait only for submit notify.
                    while (
                        not self._stop.is_set()
                        and len(self._cancel_lane) == 0
                        and len(self._place_lane) == 0
                    ):
                        self._cond.wait()
                else:
                    self._cond.wait(timeout=max(0.001, interval_ms / 1000.0))
            if self._stop.is_set():
                break
            self._flush_one_batch()
            # While a place/cancel is in flight, another submit may notify; that wake is lost if we
            # were not in wait(). Drain until empty so multi-side submits always complete.
            while self.queue_depth() > 0 and not self._stop.is_set():
                self._flush_one_batch()

    def _flush_one_batch(self) -> None:
        with self._cond:
            cancels = list(self._cancel_lane)
            places_all = list(self._place_lane)
            self._cancel_lane.clear()
            self._place_lane.clear()
        # amend-prio Phase 2 (v1.4.16): partition the place lane into
        # place vs amend by ``intent.kind``. Each subset is dispatched
        # to its own executor (batch_place / batch_amend). The single-
        # worker flush processes them serially: cancels first (cancel-
        # prio), then places, then amends. A flush can fire BOTH a
        # batch-place AND a batch-amend call — they hit different OKX
        # endpoints with separate rate-limit pools so parallelism is
        # fine. See ``plans/amend-prio.md`` Phase 2.
        places = [
            p for p in places_all
            if str(getattr(p, "kind", "place") or "place") == "place"
        ]
        amends = [
            p for p in places_all
            if str(getattr(p, "kind", "place") or "place") == "amend"
        ]
        max_n = max(1, int(self._settings.action_max_batch_size))
        budget = max_n
        done_c = 0
        done_p = 0
        done_a = 0
        leftover_c: list[CancelTransportIntent] = []
        leftover_p: list[PlaceTransportIntent] = []
        leftover_a: list[PlaceTransportIntent] = []
        # 1.3.130 multi-rung Phase 2: same-side exclusivity keys now
        # carry the rung index. At N=1 these collapse to (side, 0).
        batch_cancel_sides: set[tuple[Side, int]] = set()
        batch_place_sides: set[tuple[Side, int]] = set()
        # amend-prio Phase 2 (v1.4.16): amend-side exclusivity keys
        # (mirrors the place pattern). Tracked separately from the
        # place set so a place + an amend on different slots inside
        # the same flush don't trample each other.
        batch_amend_sides: set[tuple[Side, int]] = set()
        # 1.4.0 cancel-prio Phase 1b: opportunistic batch-cancel. When
        # the batch callback is wired AND we have 1+ cancels for
        # distinct sides AND none are blocked by an active in-flight
        # cancel on the same side, dispatch them as ONE batch call
        # via ``_execute_cancel_batch``. The original 2-cancel case
        # (BUY + SELL on a full reprice) saves a full HTTP RTT.
        #
        # v1.4.33 (2026-05-17) rate-limit-pool fix: lowered threshold
        # 2 → 1. Snapshot ``v1.4.32-260517-231426`` showed
        # ``cancels (single)`` at 412 % of the 60/2 s cap (peak 247/2 s
        # over 3439 total cancels), because almost every cancel was a
        # solo Binance-cross-venue trigger on one side — never two in
        # the flush window together — so the old ``>= 2`` threshold
        # always fell through to the per-cancel loop and pummeled the
        # 60/2 s single-cancel endpoint. Routing 1-element cancels
        # through ``cancel-batch-orders`` (300/2 s pool, 5× the
        # budget) eliminates the cap pressure without changing any
        # other behavior. RTT is essentially identical — OKX accepts a
        # 1-row body on the batch endpoint, and we already pay the
        # same TLS/signing cost regardless of route.
        if (
            self._execute_cancel_batch is not None
            and bool(getattr(self._settings, "batch_cancels_enabled", True))
            and len(cancels) >= 1
            and budget >= len(cancels)
        ):
            batched_set: set[tuple[Side, int]] = set()
            batched: list[CancelTransportIntent] = []
            still_single: list[CancelTransportIntent] = []
            for c in cancels:
                # Same invariants as the single-cancel loop: skip same-
                # rung conflicts and in-flight rung flags.
                ckey = (c.side, int(getattr(c, "level_idx", 0) or 0))
                if ckey in batched_set or ckey in self._active_cancel_sides:
                    still_single.append(c)
                    continue
                # amend-prio Phase 2 (v1.4.16) cross-lane defer: a
                # cancel arriving while an amend for the same WO is
                # in flight DEFERS. Rationale: the amend's response
                # will update wo.price/size; the cancel must fire
                # against the up-to-date order (otherwise the bot
                # might cancel against the pre-amend price/qty and
                # the venue rejects with 51503).
                if ckey in self._active_amend_sides:
                    self._cancel_deferred_amend_inflight_total += 1
                    still_single.append(c)
                    continue
                batched_set.add(ckey)
                batched.append(c)
            if len(batched) >= 1:
                self._active_cancel_sides |= batched_set
                self._inflight += len(batched)
                try:
                    self._execute_cancel_batch(batched)
                    self._batch_cancel_dispatch_count += 1
                except Exception:
                    logger.exception("outbound_execute_cancel_batch_failed")
                finally:
                    self._inflight -= len(batched)
                    self._active_cancel_sides -= batched_set
                budget -= len(batched)
                done_c += len(batched)
                # Continue with any cancels that didn't fit (same-side
                # conflicts) via the per-cancel loop below.
                cancels = still_single
            # If zero batchable cancels survived the filter (all blocked
            # by same-side / amend-inflight invariants), the per-cancel
            # loop below handles them — single-endpoint here is a
            # legitimate fallback, not the hot path.
        for c in cancels:
            if budget <= 0:
                leftover_c.append(c)
                continue
            # 1.3.130 multi-rung Phase 2: same-rung cancel invariant.
            # Two cancels for the same (side, level_idx) collapse;
            # two cancels for different rungs on the same side both
            # dispatch.
            ckey = (c.side, int(getattr(c, "level_idx", 0) or 0))
            if ckey in batch_cancel_sides or ckey in self._active_cancel_sides:
                self._same_side_inflight_deferrals += 1
                leftover_c.append(c)
                continue
            # amend-prio Phase 2 (v1.4.16) cross-lane defer: see batch
            # block above. DEFER cancel until the amend completes; the
            # leftover queue + the place worker's notify_all (in
            # parallel mode) re-runs us when the amend lands.
            if ckey in self._active_amend_sides:
                self._cancel_deferred_amend_inflight_total += 1
                leftover_c.append(c)
                continue
            batch_cancel_sides.add(ckey)
            self._active_cancel_sides.add(ckey)
            self._inflight += 1
            try:
                self._execute_cancel(c)
            except Exception:
                logger.exception("outbound_execute_cancel_failed")
            finally:
                self._inflight -= 1
                self._active_cancel_sides.discard(ckey)
            budget -= 1
            done_c += 1
        # 1.4.4: opportunistic batch-PLACE. Same shape as the batch-
        # cancel block above. When the callback is wired AND we have
        # 2+ places that don't conflict on (side, level_idx) AND
        # budget allows the whole batch, dispatch them as ONE
        # /api/v5/trade/batch-orders call. The wins are described
        # on ``self._execute_place_batch`` (init).
        #
        # Why "2+": OKX accepts a 1-element batch payload, but the
        # batch-orders endpoint has its own rate-limit budget (300/2s
        # standard) vs the single-order endpoint (60/2s). At N=1 we
        # don't want to route every single place through the batch
        # pool just to spend 1 slot — keeps the budgets predictable.
        # The "always batch" knob (``batch_places_always``) flips
        # this — see below.
        always_batch = bool(
            getattr(self._settings, "batch_places_always", False)
        )
        min_for_batch = 1 if always_batch else 2
        if (
            self._execute_place_batch is not None
            and bool(getattr(self._settings, "batch_places_enabled", True))
            and len(places) >= min_for_batch
            and budget >= len(places)
        ):
            batched_set: set[tuple[Side, int]] = set()
            batched_p: list[PlaceTransportIntent] = []
            still_single_p: list[PlaceTransportIntent] = []
            for p in places:
                pkey = (p.side, int(getattr(p, "level_idx", 0) or 0))
                if pkey in batched_set or pkey in self._active_place_sides:
                    still_single_p.append(p)
                    continue
                batched_set.add(pkey)
                batched_p.append(p)
            if len(batched_p) >= min_for_batch:
                self._active_place_sides |= batched_set
                self._inflight += len(batched_p)
                try:
                    self._execute_place_batch(batched_p)
                    self._batch_place_dispatch_count += 1
                except Exception:
                    logger.exception("outbound_execute_place_batch_failed")
                finally:
                    self._inflight -= len(batched_p)
                    self._active_place_sides -= batched_set
                budget -= len(batched_p)
                done_p += len(batched_p)
                # Continue with any places that didn't fit
                # (same-(side, level_idx) conflicts) via the per-place
                # loop below.
                places = still_single_p
        for p in places:
            if budget <= 0:
                leftover_p.append(p)
                continue
            # 1.3.130 multi-rung Phase 2: same-rung place invariant.
            # Two places for the same (side, level_idx) collapse;
            # two places for different rungs on the same side both
            # dispatch.
            pkey = (p.side, int(getattr(p, "level_idx", 0) or 0))
            if pkey in batch_place_sides or pkey in self._active_place_sides:
                self._same_side_inflight_deferrals += 1
                leftover_p.append(p)
                continue
            batch_place_sides.add(pkey)
            self._active_place_sides.add(pkey)
            self._inflight += 1
            try:
                self._execute_place(p)
            except Exception:
                logger.exception("outbound_execute_place_failed")
            finally:
                self._inflight -= 1
                self._active_place_sides.discard(pkey)
            budget -= 1
            done_p += 1
        # amend-prio Phase 2 (v1.4.16): amend lane dispatch. The same
        # opportunistic-batch shape as places — if 2+ amends queue
        # together (or 1+ in always-batch mode) call
        # ``execute_amend_batch`` with the whole list; else fall back
        # to per-intent ``execute_amend``. Cross-lane discard rule:
        # an amend whose target slot has a cancel in flight is
        # discarded (the WO is about to disappear; no point amending
        # it). Counter ``amend_discarded_cancel_inflight_total``
        # surfaces this for the operator.
        always_batch_a = bool(
            getattr(self._settings, "batch_places_always", False)
        )
        min_for_batch_a = 1 if always_batch_a else 2
        # Pre-filter: drop amends whose slot has a cancel in flight.
        runnable_amends: list[PlaceTransportIntent] = []
        for a in amends:
            akey = (a.side, int(getattr(a, "level_idx", 0) or 0))
            if (
                akey in self._active_cancel_sides
                or akey in batch_cancel_sides
            ):
                self._amend_discarded_cancel_inflight_total += 1
                continue
            runnable_amends.append(a)
        amends = runnable_amends
        if (
            self._execute_amend_batch is not None
            and bool(getattr(self._settings, "batch_places_enabled", True))
            and len(amends) >= min_for_batch_a
            and budget >= len(amends)
        ):
            batched_set_a: set[tuple[Side, int]] = set()
            batched_a: list[PlaceTransportIntent] = []
            still_single_a: list[PlaceTransportIntent] = []
            for a in amends:
                akey = (a.side, int(getattr(a, "level_idx", 0) or 0))
                if akey in batched_set_a or akey in self._active_amend_sides:
                    still_single_a.append(a)
                    continue
                batched_set_a.add(akey)
                batched_a.append(a)
            if len(batched_a) >= min_for_batch_a:
                self._active_amend_sides |= batched_set_a
                self._inflight += len(batched_a)
                try:
                    self._execute_amend_batch(batched_a)
                    self._batch_amend_dispatch_count += 1
                except Exception:
                    logger.exception("outbound_execute_amend_batch_failed")
                finally:
                    self._inflight -= len(batched_a)
                    self._active_amend_sides -= batched_set_a
                budget -= len(batched_a)
                done_a += len(batched_a)
                amends = still_single_a
        for a in amends:
            if budget <= 0:
                leftover_a.append(a)
                continue
            akey = (a.side, int(getattr(a, "level_idx", 0) or 0))
            if akey in batch_amend_sides or akey in self._active_amend_sides:
                self._same_side_inflight_deferrals += 1
                leftover_a.append(a)
                continue
            batch_amend_sides.add(akey)
            self._active_amend_sides.add(akey)
            self._inflight += 1
            try:
                if self._execute_amend is not None:
                    self._execute_amend(a)
                else:
                    # Defensive — amend intent submitted but no
                    # executor wired. Drop with a log; the WO will
                    # remain AMEND_PENDING and the orchestrate
                    # watchdog (Phase 4) will revert it.
                    logger.error(
                        "outbound_amend_no_executor_wired wo=%s side=%s",
                        a.wo_order_id_local, a.side.value,
                    )
            except Exception:
                logger.exception("outbound_execute_amend_failed")
            finally:
                self._inflight -= 1
                self._active_amend_sides.discard(akey)
            budget -= 1
            done_a += 1
        if leftover_c or leftover_p or leftover_a:
            with self._cond:
                for c in leftover_c:
                    self._cancel_lane.append(c)
                for p in leftover_p:
                    self._place_lane.append(p)
                for a in leftover_a:
                    self._place_lane.append(a)
                self._bump_hwm()
        self._last_batch_size = done_c + done_p + done_a
        if self._on_stats is not None:
            try:
                self._on_stats(
                    {
                        "last_batch_size": self._last_batch_size,
                        "queue_depth_after": self.queue_depth(),
                    }
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 1.3.109 Phase 3: parallel-worker mode (cancel + place threads).
    # ------------------------------------------------------------------
    #
    # Two threads drain the lanes concurrently. The shared ``self._cond``
    # protects ALL mutations to ``_cancel_lane`` / ``_place_lane`` /
    # ``_active_*_sides`` / ``_inflight`` / counters — the legacy
    # ``_flush_one_batch`` ran on a single thread and didn't need
    # locking around the active-side sets; these new methods do.
    #
    # Same-side ordering preservation: the place worker refuses to
    # dispatch a place for side S while a cancel for side S is in flight
    # (``_active_cancel_sides``). The cancel worker notifies on
    # completion, which wakes the place worker to retry the deferred
    # candidates. This restores the implicit "cancel-before-place"
    # ordering that the single-worker mode achieved via sequential
    # execution within ``_flush_one_batch``. The cancel worker does
    # NOT need the symmetric guard — a cancel targets an OLD ordId
    # (already bound via Phase 1a's sync place-response binding), while
    # a place creates a NEW ordId; the two never collide at the venue.

    def _flush_cancel_batch(self) -> int:
        """Drain the cancel lane once (parallel-worker path). Returns
        the count of cancels actually dispatched (excludes deferrals).
        All ``_active_cancel_sides`` / ``_inflight`` mutations are
        lock-protected because the place worker reads these concurrently
        for its cross-lane defer check."""
        with self._cond:
            cancels = list(self._cancel_lane)
            self._cancel_lane.clear()
        if not cancels:
            return 0

        max_n = max(1, int(self._settings.action_max_batch_size))
        budget = max_n
        done_c = 0
        leftover_c: list[CancelTransportIntent] = []

        # Opportunistic batch-cancel (Phase 1b). Same invariants as the
        # single-worker path; just lock around the active-side mutation.
        # v1.4.33: threshold lowered 2 → 1 to route 1-element cancels
        # through the CANCEL_BATCH pool (300/2 s) instead of
        # CANCEL_SINGLE (60/2 s). See the sequential path's comment
        # for the snapshot evidence (cancels-single at 412 % of cap).
        if (
            self._execute_cancel_batch is not None
            and bool(getattr(self._settings, "batch_cancels_enabled", True))
            and len(cancels) >= 1
            and budget >= len(cancels)
        ):
            # 1.3.130 multi-rung Phase 2: per-rung exclusivity keys.
            batched_set: set[tuple[Side, int]] = set()
            batched: list[CancelTransportIntent] = []
            still_single: list[CancelTransportIntent] = []
            with self._cond:
                for c in cancels:
                    ckey = (c.side, int(getattr(c, "level_idx", 0) or 0))
                    if (
                        ckey in batched_set
                        or ckey in self._active_cancel_sides
                    ):
                        still_single.append(c)
                        continue
                    # amend-prio Phase 2 (v1.4.16) cross-lane defer:
                    # DEFER cancel when amend in-flight on same slot.
                    # The amend's response will update wo.price/size;
                    # the cancel must fire against the new state.
                    if ckey in self._active_amend_sides:
                        self._cancel_deferred_amend_inflight_total += 1
                        still_single.append(c)
                        continue
                    batched_set.add(ckey)
                    batched.append(c)
                if len(batched) >= 1:
                    self._active_cancel_sides |= batched_set
                    self._inflight += len(batched)
            if len(batched) >= 1:
                try:
                    self._execute_cancel_batch(batched)
                    with self._cond:
                        self._batch_cancel_dispatch_count += 1
                except Exception:
                    logger.exception("outbound_execute_cancel_batch_failed")
                finally:
                    with self._cond:
                        self._inflight -= len(batched)
                        self._active_cancel_sides -= batched_set
                        # Wake the place worker — a cross-lane defer
                        # for one of these sides may now be unblocked.
                        self._cond.notify_all()
                budget -= len(batched)
                done_c += len(batched)
                cancels = still_single

        # Per-cancel loop. 1.3.130 multi-rung Phase 2: same-rung defer
        # (within this batch + against global ``_active_cancel_sides``).
        batch_cancel_sides: set[tuple[Side, int]] = set()
        for c in cancels:
            if budget <= 0:
                leftover_c.append(c)
                continue
            ckey = (c.side, int(getattr(c, "level_idx", 0) or 0))
            with self._cond:
                if (
                    ckey in batch_cancel_sides
                    or ckey in self._active_cancel_sides
                ):
                    self._same_side_inflight_deferrals += 1
                    leftover_c.append(c)
                    continue
                # amend-prio Phase 2 (v1.4.16) cross-lane defer.
                if ckey in self._active_amend_sides:
                    self._cancel_deferred_amend_inflight_total += 1
                    leftover_c.append(c)
                    continue
                batch_cancel_sides.add(ckey)
                self._active_cancel_sides.add(ckey)
                self._inflight += 1
            try:
                self._execute_cancel(c)
            except Exception:
                logger.exception("outbound_execute_cancel_failed")
            finally:
                with self._cond:
                    self._inflight -= 1
                    self._active_cancel_sides.discard(ckey)
                    self._cond.notify_all()
            budget -= 1
            done_c += 1

        if leftover_c:
            with self._cond:
                for c in leftover_c:
                    self._cancel_lane.append(c)
                self._bump_hwm()

        self._last_batch_size = done_c
        if self._on_stats is not None:
            try:
                self._on_stats(
                    {
                        "last_batch_size": done_c,
                        "queue_depth_after": self.queue_depth(),
                    }
                )
            except Exception:
                pass
        return done_c

    def _flush_place_batch(self) -> int:
        """Drain the place lane once (parallel-worker path). Returns
        the count of places + amends actually dispatched. The cross-
        lane defer guard holds back a place for side S when a cancel
        for side S is in flight; deferred intents are re-queued and
        the place worker's outer loop falls back to ``wait()`` until
        the cancel worker's completion ``notify_all`` wakes it.

        amend-prio Phase 2 (v1.4.16): the place lane carries BOTH
        place and amend intents (discriminator ``intent.kind``). This
        method partitions them by kind and dispatches each subset to
        its own executor. The cross-lane rules expand: amends whose
        slot has a cancel in flight are DISCARDED (the WO is about to
        disappear); cancels whose slot has an amend in flight are
        DEFERRED in ``_flush_cancel_batch``.
        """
        with self._cond:
            places_all = list(self._place_lane)
            self._place_lane.clear()
        if not places_all:
            return 0

        # amend-prio Phase 2 (v1.4.16): partition by kind.
        places = [
            p for p in places_all
            if str(getattr(p, "kind", "place") or "place") == "place"
        ]
        amends = [
            p for p in places_all
            if str(getattr(p, "kind", "place") or "place") == "amend"
        ]

        max_n = max(1, int(self._settings.action_max_batch_size))
        budget = max_n
        done_p = 0
        done_a = 0
        leftover_p: list[PlaceTransportIntent] = []
        leftover_a: list[PlaceTransportIntent] = []

        # 1.4.4: opportunistic batch-PLACE. Same shape as the cancel-
        # worker's batch-cancel block (lines ~530-575). When the
        # callback is wired and the flush has 2+ places (or 1+ in
        # always-batch mode), dispatch them as ONE batch call.
        always_batch = bool(
            getattr(self._settings, "batch_places_always", False)
        )
        min_for_batch = 1 if always_batch else 2
        if (
            self._execute_place_batch is not None
            and bool(getattr(self._settings, "batch_places_enabled", True))
            and len(places) >= min_for_batch
            and budget >= len(places)
        ):
            pre_batched_set: set[tuple[Side, int]] = set()
            pre_batched: list[PlaceTransportIntent] = []
            still_single_p: list[PlaceTransportIntent] = []
            with self._cond:
                for p in places:
                    pkey = (p.side, int(getattr(p, "level_idx", 0) or 0))
                    # Skip same-rung dupes within this batch + against
                    # global active set. ALSO honour the cross-lane
                    # defer rule: a place against a (side, level_idx)
                    # whose cancel is in flight stays single (and may
                    # be re-deferred below) so the cancel-then-place
                    # ordering is preserved.
                    if (
                        pkey in pre_batched_set
                        or pkey in self._active_place_sides
                        or pkey in self._active_cancel_sides
                    ):
                        still_single_p.append(p)
                        continue
                    pre_batched_set.add(pkey)
                    pre_batched.append(p)
                if len(pre_batched) >= min_for_batch:
                    self._active_place_sides |= pre_batched_set
                    self._inflight += len(pre_batched)
            if len(pre_batched) >= min_for_batch:
                try:
                    self._execute_place_batch(pre_batched)
                    with self._cond:
                        self._batch_place_dispatch_count += 1
                except Exception:
                    logger.exception("outbound_execute_place_batch_failed")
                finally:
                    with self._cond:
                        self._inflight -= len(pre_batched)
                        self._active_place_sides -= pre_batched_set
                        self._cond.notify_all()
                budget -= len(pre_batched)
                done_p += len(pre_batched)
                places = still_single_p

        # 1.3.130 multi-rung Phase 2: same-rung exclusivity keys.
        batch_place_sides: set[tuple[Side, int]] = set()
        for p in places:
            if budget <= 0:
                leftover_p.append(p)
                continue
            pkey = (p.side, int(getattr(p, "level_idx", 0) or 0))
            with self._cond:
                # 1.3.130: same-rung place exclusivity (within this
                # batch + global). Two places for distinct rungs on the
                # same side both dispatch concurrently.
                if (
                    pkey in batch_place_sides
                    or pkey in self._active_place_sides
                ):
                    self._same_side_inflight_deferrals += 1
                    leftover_p.append(p)
                    continue
                # Cross-lane: hold back a place while a same-rung
                # cancel is in flight. Maintains the "cancel before
                # place" ordering for the SAME rung; other rungs on
                # the same side are unaffected.
                if pkey in self._active_cancel_sides:
                    self._cross_lane_deferrals += 1
                    leftover_p.append(p)
                    continue
                batch_place_sides.add(pkey)
                self._active_place_sides.add(pkey)
                self._inflight += 1
            try:
                self._execute_place(p)
            except Exception:
                logger.exception("outbound_execute_place_failed")
            finally:
                with self._cond:
                    self._inflight -= 1
                    self._active_place_sides.discard(pkey)
                    self._cond.notify_all()
            budget -= 1
            done_p += 1

        # amend-prio Phase 2 (v1.4.16): amend dispatch block. Mirrors
        # the place block above but uses ``execute_amend_batch`` /
        # ``execute_amend`` and the ``_active_amend_sides`` set. The
        # cross-lane DISCARD (vs DEFER) rule for amends arises from
        # the WO lifecycle: a cancel that's in flight will turn the
        # WO terminal — the amend would have no target. Counter
        # ``amend_discarded_cancel_inflight_total`` surfaces the rate.
        runnable_amends: list[PlaceTransportIntent] = []
        with self._cond:
            for a in amends:
                akey = (a.side, int(getattr(a, "level_idx", 0) or 0))
                if akey in self._active_cancel_sides:
                    self._amend_discarded_cancel_inflight_total += 1
                    continue
                runnable_amends.append(a)
        amends = runnable_amends
        if (
            self._execute_amend_batch is not None
            and bool(getattr(self._settings, "batch_places_enabled", True))
            and len(amends) >= min_for_batch
            and budget >= len(amends)
        ):
            pre_batched_set_a: set[tuple[Side, int]] = set()
            pre_batched_a: list[PlaceTransportIntent] = []
            still_single_a: list[PlaceTransportIntent] = []
            with self._cond:
                for a in amends:
                    akey = (a.side, int(getattr(a, "level_idx", 0) or 0))
                    if (
                        akey in pre_batched_set_a
                        or akey in self._active_amend_sides
                    ):
                        still_single_a.append(a)
                        continue
                    # Re-check cross-lane (could have changed since
                    # the pre-filter under separate lock acquisitions).
                    if akey in self._active_cancel_sides:
                        self._amend_discarded_cancel_inflight_total += 1
                        continue
                    pre_batched_set_a.add(akey)
                    pre_batched_a.append(a)
                if len(pre_batched_a) >= min_for_batch:
                    self._active_amend_sides |= pre_batched_set_a
                    self._inflight += len(pre_batched_a)
            if len(pre_batched_a) >= min_for_batch:
                try:
                    self._execute_amend_batch(pre_batched_a)
                    with self._cond:
                        self._batch_amend_dispatch_count += 1
                except Exception:
                    logger.exception("outbound_execute_amend_batch_failed")
                finally:
                    with self._cond:
                        self._inflight -= len(pre_batched_a)
                        self._active_amend_sides -= pre_batched_set_a
                        self._cond.notify_all()
                budget -= len(pre_batched_a)
                done_a += len(pre_batched_a)
                amends = still_single_a

        batch_amend_sides: set[tuple[Side, int]] = set()
        for a in amends:
            if budget <= 0:
                leftover_a.append(a)
                continue
            akey = (a.side, int(getattr(a, "level_idx", 0) or 0))
            with self._cond:
                if (
                    akey in batch_amend_sides
                    or akey in self._active_amend_sides
                ):
                    self._same_side_inflight_deferrals += 1
                    leftover_a.append(a)
                    continue
                if akey in self._active_cancel_sides:
                    # Cross-lane DISCARD (vs the place block's DEFER):
                    # the WO is about to disappear. Don't re-queue.
                    self._amend_discarded_cancel_inflight_total += 1
                    continue
                batch_amend_sides.add(akey)
                self._active_amend_sides.add(akey)
                self._inflight += 1
            try:
                if self._execute_amend is not None:
                    self._execute_amend(a)
                else:
                    logger.error(
                        "outbound_amend_no_executor_wired wo=%s side=%s",
                        a.wo_order_id_local, a.side.value,
                    )
            except Exception:
                logger.exception("outbound_execute_amend_failed")
            finally:
                with self._cond:
                    self._inflight -= 1
                    self._active_amend_sides.discard(akey)
                    self._cond.notify_all()
            budget -= 1
            done_a += 1

        if leftover_p or leftover_a:
            with self._cond:
                for p in leftover_p:
                    self._place_lane.append(p)
                for a in leftover_a:
                    self._place_lane.append(a)
                self._bump_hwm()

        self._last_batch_size = done_p + done_a
        if self._on_stats is not None:
            try:
                self._on_stats(
                    {
                        "last_batch_size": done_p + done_a,
                        "queue_depth_after": self.queue_depth(),
                    }
                )
            except Exception:
                pass
        # amend-prio Phase 2 (v1.4.16): return places + amends so the
        # outer drain loop continues when amends made progress (the
        # caller treats positive return as "fired something this pass").
        return done_p + done_a

    def _cancel_loop(self) -> None:
        """Parallel-worker cancel-lane drain loop. Wakes on
        ``submit_cancel`` notifications (or the batch interval) and
        drains until the lane is empty before going back to wait. The
        outer ``while-progress`` drain mirrors the legacy loop's
        "drain until empty" semantics — a notification arriving mid-
        HTTP must not be lost."""
        while not self._stop.is_set():
            interval_ms = float(self._settings.action_batch_interval_ms)
            with self._cond:
                if interval_ms <= 0:
                    while not self._stop.is_set() and not self._cancel_lane:
                        self._cond.wait()
                else:
                    if not self._cancel_lane:
                        self._cond.wait(
                            timeout=max(0.001, interval_ms / 1000.0)
                        )
            if self._stop.is_set():
                break
            self._flush_cancel_batch()
            while self._cancel_lane and not self._stop.is_set():
                self._flush_cancel_batch()

    def _place_loop(self) -> None:
        """Parallel-worker place-lane drain loop. The wait predicate
        is richer than the cancel worker's: we wait while either the
        lane is empty OR all places in the lane are blocked by a
        same-side cancel in flight (cross-lane defer would re-queue
        them all). The cancel worker's completion ``notify_all`` will
        wake this loop when a blocking cancel finishes.

        The ``progress > 0`` guard on the outer drain prevents busy-
        looping when ``_flush_place_batch`` would re-defer every
        candidate; we fall back to ``wait()`` in that case."""
        while not self._stop.is_set():
            interval_ms = float(self._settings.action_batch_interval_ms)
            with self._cond:
                if interval_ms <= 0:
                    while (
                        not self._stop.is_set()
                        and not self._place_has_runnable_locked()
                    ):
                        self._cond.wait()
                else:
                    if not self._place_has_runnable_locked():
                        self._cond.wait(
                            timeout=max(0.001, interval_ms / 1000.0)
                        )
            if self._stop.is_set():
                break
            progress = self._flush_place_batch()
            while (
                progress > 0
                and self._place_lane
                and not self._stop.is_set()
            ):
                progress = self._flush_place_batch()

    def _place_has_runnable_locked(self) -> bool:
        """Returns True if at least one place / amend in the lane
        would NOT be cross-lane-deferred. Must be called while holding
        ``self._cond``'s lock — reads ``_place_lane``,
        ``_active_cancel_sides`` and ``_active_amend_sides``.

        1.3.130 multi-rung Phase 2: cross-lane check is per-(side,
        level_idx). A place for (BUY, 1) is runnable even when a cancel
        for (BUY, 0) is in flight — the rungs are independent orders.

        amend-prio Phase 2 (v1.4.16): an amend intent is runnable as
        long as no cancel for its slot is in flight (amends are DISCARDED
        on cross-lane conflict — they remove themselves from the lane
        rather than being re-queued, so a discardable amend should NOT
        keep the worker blocked in wait()). We treat it as runnable
        here so the worker wakes, then the flush method does the
        discard.
        """
        if not self._place_lane:
            return False
        for p in self._place_lane:
            pkey = (p.side, int(getattr(p, "level_idx", 0) or 0))
            kind = str(getattr(p, "kind", "place") or "place")
            if kind == "amend":
                # Always wake — flush will either dispatch or discard.
                return True
            if pkey not in self._active_cancel_sides:
                return True
        return False
