"""Single-symbol perpetual market-making service.

Multi-venue support (Binance / Bluefin / GRVT / Hyperliquid / OKX).

Concurrency model (v1.4.93 wedge-elimination-cleanup Phase 7B.1):
================================================================

The bot is a single-symbol multi-rung passive market maker. State
mutation is serialized through one of two locks:

1. ``BotState._lock`` — the global state lock. Held briefly during:
   * WS event handlers mutating position / market / WO records
   * The hot path's per-tick snapshot construction (one acquire per
     tick via ``state.tick_snapshot()`` — Phase 5A pattern)
   * `_record_orchestrate_decision` ring-buffer appends

2. ``OrderStore._lock`` — the order store's internal lock (Phase 3A).
   Currently a reference to ``BotState._lock`` (additive migration);
   per-store locks are a Phase 5+ perf concern that hasn't been
   shipped because the global lock is held briefly enough that
   splitting it has no measurable hot-path effect.

Order identity: ``(symbol, side, level_idx)``. At most one live order
per slot. Slots are independent. ``LADDER_NUM_LEVELS_PER_SIDE``
(default 1) sets the slot count. Every layer that touches orders is
slot-aware — see ``docs/order-state-contract.md``.

Thread model:
* **Main quote loop** (single thread) drives ``maybe_refresh_quotes``
  at ~2 Hz (every 500 ms). It's the only writer for the hot path.
* **Private WS handler** (dedicated thread) enqueues
  ``PrivateOrderUpdateEvent`` / ``PrivateFillEvent`` into a shared
  ``queue.Queue``. The quote loop drains it via
  ``drain_private_events()`` at tick start.
* **Outbound dispatcher** (worker thread, `app/outbound_dispatch.py`)
  processes place / cancel / amend intents enqueued by
  ``maybe_refresh_quotes``. Fires HTTP calls; results come back via
  the private WS event queue.
* **Public WS handler** (per-venue threads) updates ``state.market``
  atomically. Reads from the quote loop are torn-read-safe under the
  Python GIL.

Wedge-prevention contracts (these MUST hold; see
``docs/wedge-prevention-runbook.md`` for the full rationale):

* Every order has at most ONE matching WorkingOrder in local state.
  Enforced by ``OrderStore._wo_by_oid`` / ``_wo_by_cloid`` indexes
  (Phase 1B / 3A).
* No WorkingOrder lives past ``BEHIND_TOUCH_MAX_AGE_SECONDS × 2``.
  Enforced by the wall-lifetime cap (Phase 1C) + the Phase 2B reaper
  as a safety net.
* The dispatcher never silently drops a non-NoOp action. Every
  Place/Cancel/Amend either produces a transport call or an
  explicit suppression trace via ``_record_orchestrate_decision``.
  Property-tested by `test_phase2a5_dispatcher_no_silent_drops`.
* The bot's local order state and the venue's order state agree
  after every tick of steady-state operation. Verified by the
  reconciler (`app/reconciler.py`) and asserted in Phase 6A's
  multi-tick replay tests.

Snapshot format and replay framework:
* Operator snapshots are dropped under `snapshots/` by
  ``scripts/fetch_bot_snapshot.ps1`` and `stats_snapshot.py`.
* The replay framework lives in
  ``tests/integration/snapshot_replay.py`` and exercises the bot's
  state-loading paths against captured production data. See
  ``docs/snapshot-replay-howto.md``.
* For an automated periodic acceptance check, run
  ``python scripts/verify_snapshot.py <snapshot_dir>``. Exit code 0
  means the snapshot passes the Phase 7C acceptance criteria.

Type system: the quote engine returns a typed sum-type
``BuildCommand = QuoteBoth | QuoteOneSided | NoQuote | ResidualFlatten``
(Phase 4A). New fields on a variant are dataclass attributes; missing
match cases on the consumer side are caught at edit time, not runtime.
See ``docs/build-command-types.md``.

Stores (Phase 3A-3D):
* ``state.order_store`` — `OrderStore`, sole owner of working-order
  state with O(1) (oid, cloid) indexes
* ``state.position_store`` — facade over position state
* ``state.market_store`` — facade over best-bid/ask
* ``state.telemetry_store`` — facade over executor-state aggregates
* ``state.tick_snapshot()`` — frozen per-tick view for the hot path

See ``docs/store-architecture.md`` for mutation rules + lock policy.
"""

__version__ = "1.5.317"
