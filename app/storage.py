from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

from app.config import Settings

from app import clock as _clock


class Storage:
    """SQLite persistence with a small convenience API."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._path = settings.effective_sqlite_path()
        self._lock = threading.Lock()
        # Keep a persistent connection in normal runtime. In pytest, keep legacy
        # per-call lifecycle to avoid Windows temp-file unlink contention in tests
        # that aggressively unlink temporary DB files without explicit storage.close().
        self._persistent_connection = os.environ.get("PYTEST_CURRENT_TEST") is None
        self._conn: sqlite3.Connection | None = None
        if self._persistent_connection:
            self._conn = self._connect()
            self._conn.execute("PRAGMA foreign_keys=ON")

    def _connect(self) -> sqlite3.Connection:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # v1.5.266 BUG-035 mitigation. ``synchronous`` and
        # ``wal_autocheckpoint`` are CONNECTION-level pragmas — they
        # don't persist to other connections that open the same file.
        # Apply on every connection so the safety holds for both the
        # persistent runtime connection AND the per-call connections
        # the test path uses (PYTEST_CURRENT_TEST → per-call). The
        # FULL synchronous + small autocheckpoint reduce the abrupt-
        # restart corruption window observed on snapshot v1.5.251-
        # 260529-160310 / v1.5.257-260529-173313. Errors are swallowed
        # so a fresh DB whose journal_mode hasn't been set to WAL yet
        # (the journal_mode pragma in init_schema is what flips that)
        # doesn't fail this connect — wal_autocheckpoint is a no-op
        # outside WAL mode and is fine to ignore.
        try:
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA wal_autocheckpoint=200")
        except Exception:
            pass
        return conn

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        if self._persistent_connection:
            conn = self._conn
            if conn is None:
                raise RuntimeError("Storage connection is closed")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return

        conn = self._connect()
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def close(self) -> None:
        with self._lock:
            conn = self._conn
            self._conn = None
            if conn is not None:
                # v1.5.266 BUG-035 mitigation: explicit WAL checkpoint
                # before close so the WAL file is fully merged into the
                # main DB and truncated. Without this, an abrupt restart
                # immediately after close (e.g. systemctl restart races)
                # can leave the WAL index in a state that the next
                # process opens, fails to replay correctly, and writes
                # over — producing the "database disk image is
                # malformed" corruption observed on snapshot
                # v1.5.251-260529-160310 and v1.5.257-260529-173313.
                # TRUNCATE mode (vs PASSIVE / FULL / RESTART) is the
                # strongest variant — it blocks for all readers to
                # complete then resizes WAL to zero. At process exit
                # there are no remaining readers so the block is
                # essentially free.
                # Best-effort: any failure here must NOT prevent
                # conn.close() from running, otherwise the file handle
                # leaks and Windows test cleanup fails.
                try:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception:
                    pass
                conn.close()

    def __del__(self) -> None:
        try:
            conn = getattr(self, "_conn", None)
            if conn is not None:
                conn.close()
                self._conn = None
        except Exception:
            pass

    SCHEMA_VERSION: int = 46

    def init_schema(self) -> None:
        with self._lock:
            with self.connection() as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA foreign_keys=ON")
                # v1.5.266 BUG-035 mitigation. Default WAL mode uses
                # synchronous=NORMAL which is safe against power loss
                # but can corrupt under specific abrupt-restart races
                # (SIGTERM during WAL checkpoint + SIGKILL escalation
                # before checkpoint finishes, observed today during
                # rapid systemctl restart for the v1.5.249-251 deploy
                # cluster). FULL synchronous adds an fsync after every
                # commit and after WAL frames — closes the race at the
                # cost of ~5 % write throughput, which is negligible
                # at this bot's write rate (~150 writes/min).
                conn.execute("PRAGMA synchronous=FULL")
                # Auto-checkpoint every 200 frames (legacy default is
                # 1000). Keeps WAL file smaller so a restart never sees
                # a multi-MB un-checkpointed tail that has to be
                # replayed. The trade-off is more frequent (smaller)
                # checkpoints in the writer thread; at our write rate
                # that's ~one checkpoint per 90 seconds vs the legacy
                # ~once per 8 minutes.
                conn.execute("PRAGMA wal_autocheckpoint=200")
                cur = conn.execute("PRAGMA user_version")
                row = cur.fetchone()
                current_v = int(row[0]) if row else 0
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id_local TEXT PRIMARY KEY,
                        order_id_exchange TEXT,
                        client_order_id TEXT,
                        ts_created TEXT,
                        ts_sent TEXT,
                        ts_ack TEXT,
                        ts_closed TEXT,
                        symbol TEXT,
                        side TEXT,
                        price REAL,
                        size REAL,
                        post_only INTEGER,
                        status TEXT,
                        cancel_reason TEXT,
                        replace_group_id TEXT,
                        quote_cycle_id TEXT,
                        level_idx INTEGER DEFAULT 0
                    );
                    CREATE TABLE IF NOT EXISTS fills (
                        fill_id TEXT PRIMARY KEY,
                        order_id_exchange TEXT,
                        client_order_id TEXT,
                        ts_fill TEXT,
                        symbol TEXT,
                        side TEXT,
                        price REAL,
                        size REAL,
                        notional REAL,
                        fee REAL,
                        liquidity_flag TEXT,
                        mid_at_fill REAL,
                        best_bid_at_fill REAL,
                        best_ask_at_fill REAL,
                        book_snapshot_quality TEXT,
                        markout_1s_bps REAL,
                        markout_3s_bps REAL,
                        markout_5s_bps REAL,
                        markout_15s_bps REAL,
                        markout_30s_bps REAL,
                        markout_60s_bps REAL,
                        markout_120s_bps REAL,
                        level_idx INTEGER DEFAULT 0,
                        book_reference_quality TEXT,
                        closed_pnl REAL
                    );
                    CREATE TABLE IF NOT EXISTS position_snapshots (
                        ts TEXT,
                        symbol TEXT,
                        position_qty REAL,
                        avg_entry_price REAL,
                        mark_price REAL,
                        position_notional REAL,
                        unrealized_pnl_usd REAL
                    );
                    CREATE TABLE IF NOT EXISTS equity_snapshots (
                        ts TEXT,
                        equity_usd REAL,
                        cash_usd REAL,
                        realized_pnl_usd REAL,
                        unrealized_pnl_usd REAL,
                        fees_usd REAL,
                        drawdown_usd REAL
                    );
                    CREATE TABLE IF NOT EXISTS quote_decisions (
                        ts TEXT,
                        symbol TEXT,
                        mid_price REAL,
                        vol_estimate REAL,
                        inventory REAL,
                        reservation_price REAL,
                        target_spread_bps REAL,
                        target_bid REAL,
                        target_ask REAL,
                        quoted_bid REAL,
                        quoted_ask REAL,
                        quoted_bid_sz REAL,
                        quoted_ask_sz REAL,
                        active_sides TEXT,
                        toxicity_score REAL,
                        decision_reason TEXT,
                        quote_cycle_id TEXT,
                        spread_floor_overlay_half_spread_bps REAL DEFAULT 0
                    );
                    CREATE TABLE IF NOT EXISTS bot_events (
                        ts TEXT,
                        severity TEXT,
                        event_type TEXT,
                        message TEXT,
                        payload_json TEXT
                    );
                    CREATE INDEX IF NOT EXISTS idx_fills_ts ON fills(ts_fill);
                    CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
                    CREATE INDEX IF NOT EXISTS idx_quotes_ts ON quote_decisions(ts);
                    CREATE INDEX IF NOT EXISTS idx_events_ts ON bot_events(ts);
                    """
                )
                self._migrate(conn, current_v)

    def _migrate(self, conn: sqlite3.Connection, from_v: int) -> None:
        """Apply incremental schema upgrades; bump SCHEMA_VERSION when adding ALTER steps."""
        if from_v < 1:
            pass  # v1: initial tables (CREATE IF NOT EXISTS above)
        if from_v < 2:
            for stmt in (
                "ALTER TABLE fills ADD COLUMN best_bid_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN best_ask_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN book_snapshot_quality TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 3:
            try:
                conn.execute(
                    "ALTER TABLE fills ADD COLUMN book_reference_quality TEXT"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 4:
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN exec_norm_bid_px REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_norm_bid_sz REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_norm_ask_px REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_norm_ask_sz REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 5:
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN exec_raw_bid_px REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_raw_bid_sz REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_raw_ask_px REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_raw_ask_sz REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_price_tick REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_size_step REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 6:
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN exec_meta_decimal_grid_price_tick REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_meta_decimal_size_step REAL",
                "ALTER TABLE quote_decisions ADD COLUMN exec_hl_max_sig_figs_nonint INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN exec_price_normalize_pipeline TEXT",
                "ALTER TABLE quote_decisions ADD COLUMN exec_wire_bid_limit_p TEXT",
                "ALTER TABLE quote_decisions ADD COLUMN exec_wire_ask_limit_p TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 7:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_data_gap_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    ts_utc TEXT NOT NULL,
                    gap_ms REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mdgap_session_id ON market_data_gap_samples(session_id)"
            )
        if from_v < 8:
            try:
                conn.execute(
                    "ALTER TABLE quote_decisions ADD COLUMN spread_floor_overlay_half_spread_bps REAL DEFAULT 0"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 9:
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN decision_to_submit_dispatch_ms REAL",
                "ALTER TABLE quote_decisions ADD COLUMN submit_queue_wait_ms REAL",
                "ALTER TABLE quote_decisions ADD COLUMN submit_transport_rtt_ms REAL",
                "ALTER TABLE quote_decisions ADD COLUMN decision_to_first_submit_dispatch_ms REAL",
                "ALTER TABLE quote_decisions ADD COLUMN ack_resolution_ms REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 10:
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN quote_eligibility TEXT",
                "ALTER TABLE quote_decisions ADD COLUMN quote_eligibility_reason TEXT",
                "ALTER TABLE fills ADD COLUMN quote_eligibility_state TEXT",
                "ALTER TABLE fills ADD COLUMN quote_eligibility_reason TEXT",
                "ALTER TABLE fills ADD COLUMN book_age_seconds_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN mid_return_100ms_bps_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN mid_return_250ms_bps_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN mid_return_500ms_bps_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN fill_during_quote_cooldown INTEGER",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 11:
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN source_book_ts_exchange_ms INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN source_book_ts_local_iso TEXT",
                "ALTER TABLE quote_decisions ADD COLUMN effective_book_age_at_decision_ms REAL",
                "ALTER TABLE quote_decisions ADD COLUMN book_apply_to_decision_ms REAL",
                "ALTER TABLE quote_decisions ADD COLUMN public_ws_queue_wait_ms_latest REAL",
                "ALTER TABLE quote_decisions ADD COLUMN public_ws_receive_to_apply_ms_latest REAL",
                "ALTER TABLE quote_decisions ADD COLUMN decision_market_data_regime TEXT",
                "ALTER TABLE fills ADD COLUMN effective_book_age_at_last_decision_ms REAL",
                "ALTER TABLE fills ADD COLUMN effective_book_age_at_fill_ms REAL",
                "ALTER TABLE fills ADD COLUMN private_ws_receive_to_state_apply_ms REAL",
                "ALTER TABLE fills ADD COLUMN quote_cycle_to_first_transport_send_ms REAL",
                "ALTER TABLE fills ADD COLUMN quote_cycle_to_first_ack_ms REAL",
                "ALTER TABLE fills ADD COLUMN ack_to_private_ws_lifecycle_ms REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 12:
            # Indexes for the new ``/*/since`` time-range endpoints.
            # ``orders`` previously only had ``idx_orders_symbol`` — large
            # time-range scans on ``ts_created`` were O(n). ``position_snapshots``
            # and ``equity_snapshots`` had no indexes at all.
            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_orders_ts_created ON orders(ts_created)",
                "CREATE INDEX IF NOT EXISTS idx_positions_ts ON position_snapshots(ts)",
                "CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_snapshots(ts)",
            ):
                conn.execute(stmt)
        if from_v < 13:
            # Microprice-based reservation: persist the microprice value
            # used (or that WOULD be used when the feature is disabled,
            # enabling shadow-mode measurement). NULL when depth was
            # unavailable on the cycle's book snapshot.
            try:
                conn.execute(
                    "ALTER TABLE quote_decisions ADD COLUMN microprice REAL"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 14:
            # Binance Level 1 cross-venue reference: record the fair-value
            # inputs used at decision time so post-hoc shadow analysis can
            # answer, per cycle: "what was Binance mid and the
            # GRVT-vs-Binance basis when we built this quote?". Without
            # these columns, the only cross-venue state in the DB is the
            # ``binance_cross_venue_cancel`` event — which only fires on
            # trigger moments, not on every cycle. Both columns are NULL
            # when ``BINANCE_WS_ENABLED=false`` or the feed hasn't warmed
            # up yet (first message not received, or basis EWMA not seeded).
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN binance_mid REAL",
                "ALTER TABLE quote_decisions ADD COLUMN binance_basis_ewma REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 15:
            # Soft-flatten episode attribution (plans/20260507-sf-frontend.md
            # Phase 2). One row per SF episode; FK column on orders + fills
            # so the dashboard can colour rows that belong to a "panic
            # window." NULL on every pre-existing order / fill (additive
            # migration; no backfill).
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS soft_flatten_events (
                    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_start               TEXT NOT NULL,
                    ts_end                 TEXT,
                    trigger_reason         TEXT,
                    initial_force_phase    INTEGER,
                    taker_fallback_ticks   INTEGER,
                    entry_position_qty     REAL,
                    entry_mid_price        REAL,
                    exit_phase_reached     INTEGER,
                    exit_reason            TEXT,
                    notes                  TEXT
                )
                """
            )
            for stmt in (
                "ALTER TABLE orders ADD COLUMN soft_flatten_event_id INTEGER",
                "ALTER TABLE fills  ADD COLUMN soft_flatten_event_id INTEGER",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_sf_events_ts_start "
                "ON soft_flatten_events(ts_start)",
                "CREATE INDEX IF NOT EXISTS idx_orders_sf_event_id "
                "ON orders(soft_flatten_event_id)",
                "CREATE INDEX IF NOT EXISTS idx_fills_sf_event_id "
                "ON fills(soft_flatten_event_id)",
            ):
                conn.execute(stmt)
        if from_v < 16:
            # todo-005 / todo-006: target half-spread + aggressiveness
            # tag stamped at place-time on the order, propagated to the
            # fill at ingestion. Drives the live FillBucketsCard's
            # aggressiveness slice and the offline "spread quality"
            # report. Additive migration: NULL on every pre-existing
            # row, no backfill.
            for stmt in (
                "ALTER TABLE orders ADD COLUMN target_half_spread_bps REAL",
                "ALTER TABLE orders ADD COLUMN quote_aggressiveness TEXT",
                "ALTER TABLE fills  ADD COLUMN target_half_spread_bps REAL",
                "ALTER TABLE fills  ADD COLUMN quote_aggressiveness TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 17:
            # plans/20260507-calibrate.md Tier 2 #1 + #5: behind-touch
            # buffer on adding side at high inventory + markout-based
            # quote aging. New telemetry columns flow into every
            # ``quote_decisions`` row (markout_*) or appear when the
            # buffer fires (inventory_high_*). Additive migration;
            # NULL on every pre-existing row.
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN markout_adverse_bps_bid REAL",
                "ALTER TABLE quote_decisions ADD COLUMN markout_adverse_bps_ask REAL",
                "ALTER TABLE quote_decisions ADD COLUMN markout_adverse_breach_elapsed_s_bid REAL",
                "ALTER TABLE quote_decisions ADD COLUMN markout_adverse_breach_elapsed_s_ask REAL",
                "ALTER TABLE quote_decisions ADD COLUMN markout_adverse_cancel_bid INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN markout_adverse_cancel_ask INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN inventory_high_adding_side_bid_buffered INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN inventory_high_adding_side_ask_buffered INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN inventory_high_adding_side_bid_buffer_ticks REAL",
                "ALTER TABLE quote_decisions ADD COLUMN inventory_high_adding_side_ask_buffer_ticks REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 18:
            # Codex review 2026-05-09 (MED-7) follow-up: persist per-fill
            # ``closed_pnl`` so postmortem reports can compute win-rate /
            # profit-factor / per-fill PnL distributions. The value is
            # already on ``Fill.closed_pnl`` (every venue adapter populates
            # it) and consumed by ``PnlTracker.on_fill``; this column
            # extends storage so the value also flows into snapshots.
            try:
                conn.execute("ALTER TABLE fills ADD COLUMN closed_pnl REAL")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 19:
            # Analysis-day instrumentation (2026-05-10): persist
            # basis-regime classifier output and top-of-book sizes
            # per fill so postmortem reports can decompose PnL
            # along basis-regime + queue-imbalance axes. All columns
            # are additive; pre-19 rows remain valid with NULLs.
            #
            # On fills:
            #   * basis_regime_sign       — N1: -1 / 0 / +1 from
            #                               BasisRegimeClassifier at fill time
            #   * bid_size_top_at_fill    — N3: best-bid size at fill snapshot
            #   * ask_size_top_at_fill    — N3: best-ask size at fill snapshot
            #
            # On quote_decisions (per-quote-cycle time-series):
            #   * basis_regime_sign       — N2: regime sign in effect this cycle
            #   * basis_ic                — N2: rolling information coefficient
            #   * basis_pair_count        — N2: number of paired samples
            #                               feeding the IC at decision time
            for stmt in (
                "ALTER TABLE fills ADD COLUMN basis_regime_sign INTEGER",
                "ALTER TABLE fills ADD COLUMN bid_size_top_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN ask_size_top_at_fill REAL",
                "ALTER TABLE quote_decisions ADD COLUMN basis_regime_sign INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN basis_ic REAL",
                "ALTER TABLE quote_decisions ADD COLUMN basis_pair_count INTEGER",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 20:
            # Codex review 2026-05-09 (MED-6): the per-fill order
            # attribution lookups (``soft_flatten_event_id_for_order``,
            # ``order_quote_quality_for_order``) filter ``orders``
            # by ``order_id_exchange`` but no index covers that
            # column. Per-fill lookups degraded to table scans as
            # the orders table grew (was ``idx_orders_symbol`` +
            # ``idx_orders_ts_created`` only, neither of which the
            # query planner can use for an exact ``WHERE
            # order_id_exchange = ?`` predicate). Adding the index
            # turns the lookup into an O(log n) seek; on a
            # multi-day session with 50k+ orders the speedup is
            # measurable.
            try:
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_orders_order_id_exchange "
                    "ON orders(order_id_exchange)"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 21:
            # todo-006 closure: persist the quote-age that
            # ``FillBucketAggregator`` already computes at fill-ingest
            # time. Lets the offline postmortem slice fills by quote-age
            # without rebuilding the in-memory ack cache. Additive
            # migration; NULL on every pre-existing row.
            try:
                conn.execute(
                    "ALTER TABLE fills ADD COLUMN quote_age_at_fill_ms REAL"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 22:
            # 1.2.9: Add mid_price + vol_bps + session_traded_notional_usd
            # to equity_snapshots so the dashboard's Session PnL chart
            # can render synced sub-bands (mid trace, volatility trace,
            # traded-volume bars) under the main PnL line. Operator
            # wants to see "what was the market doing during this
            # swing" without joining against inventory_history /
            # quote_decisions feeds. Traded volume is captured as a
            # session-cumulative value so the dashboard diffs adjacent
            # samples to get per-interval volume — that means we don't
            # need a sliding-window aggregator on the bot side.
            # Additive migration. NULL on every pre-existing row.
            for stmt in (
                "ALTER TABLE equity_snapshots ADD COLUMN mid_price REAL",
                "ALTER TABLE equity_snapshots ADD COLUMN vol_bps REAL",
                "ALTER TABLE equity_snapshots ADD COLUMN session_traded_notional_usd REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 23:
            # 1.2.14: Multi-level ladder Phase 1 (shadow). Persist
            # the per-cycle LadderDecision into quote_decisions so
            # the operator can analyze what a ladder would have
            # done in snapshot data without taking outer-rung risk
            # (only the inside rung is actually placed in Phase 1;
            # Phase 2 wires multi-rung execution).
            #
            # Two TEXT (JSON) columns for the rung arrays + small
            # int diagnostics. NULL when the bot is on a pre-1.2.14
            # build. ``ladder_bids`` / ``ladder_asks`` are the
            # rung lists (length 0..5 each). Effective level counts
            # let the operator filter by "ladder fully suppressed"
            # vs "partial suppression" without parsing the JSON.
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN ladder_bids TEXT",
                "ALTER TABLE quote_decisions ADD COLUMN ladder_asks TEXT",
                "ALTER TABLE quote_decisions ADD COLUMN ladder_requested_levels INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN ladder_effective_levels_buy INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN ladder_effective_levels_sell INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN ladder_gate_caps TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 24:
            # 1.2.15: Inventory sub-band on the Session PnL chart.
            # Add ``position_qty`` to equity_snapshots so the
            # dashboard can render an inventory trace synced with
            # the PnL line — operator wants to see "what was the
            # inventory doing during this swing?" and roughly
            # estimate mean-reversion half-life. Tiny extra cost;
            # additive migration. NULL on pre-1.2.15 rows.
            try:
                conn.execute(
                    "ALTER TABLE equity_snapshots ADD COLUMN position_qty REAL"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 25:
            # 1.2.25: BBO + mid-change session-cumulative counters
            # added to equity_snapshots so per-10s rates can be
            # derived offline (todo-010 Phase 2 decision data).
            # Diff adjacent samples → per-snapshot-interval BBO
            # event count and mid-change count. Lets us answer "is
            # there sub-cycle BBO traffic that Phase 1 dedup
            # misses?" from snapshot data alone, without live
            # logging.
            for stmt in (
                "ALTER TABLE equity_snapshots "
                "ADD COLUMN bbo_event_count_session INTEGER",
                "ALTER TABLE equity_snapshots "
                "ADD COLUMN mid_change_count_session INTEGER",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 26:
            # 2026-05-13 regime-observability Phase 1: fill-side
            # enrichment + decision-state propagation. See
            # ``plans/regime-observability.md`` Phase 1 for rationale.
            #
            # Additive migration: 11 new columns on ``orders`` (one per
            # decision-state field on WorkingOrder), 18 new columns on
            # ``fills`` (derivations + decision-state propagation +
            # cancel-race diagnostics). All NULL on pre-1.2.85 rows.
            # Purely observational — none of these columns are read by
            # the bot's quote-construction path.
            for stmt in (
                # orders table — decision-state stamping at placement.
                "ALTER TABLE orders ADD COLUMN toxicity_score_at_decision REAL",
                "ALTER TABLE orders ADD COLUMN vol_estimate_at_decision REAL",
                "ALTER TABLE orders ADD COLUMN active_sides_at_decision TEXT",
                "ALTER TABLE orders ADD COLUMN decision_reason_at_decision TEXT",
                "ALTER TABLE orders ADD COLUMN binance_basis_ewma_at_decision REAL",
                "ALTER TABLE orders ADD COLUMN adaptive_widen_active_at_decision INTEGER",
                "ALTER TABLE orders ADD COLUMN post_fill_cooldown_active_bid_at_decision INTEGER",
                "ALTER TABLE orders ADD COLUMN post_fill_cooldown_active_ask_at_decision INTEGER",
                "ALTER TABLE orders ADD COLUMN at_touch_adverse_pause_bid_at_decision INTEGER",
                "ALTER TABLE orders ADD COLUMN at_touch_adverse_pause_ask_at_decision INTEGER",
                "ALTER TABLE orders ADD COLUMN quote_distance_to_touch_ticks_at_placement REAL",
                # orders table — cancel-request timestamp. Set by the
                # execution path the moment a cancel is issued for the
                # order. Used at fill ingestion to derive
                # ``cancel_requested_before_fill`` + ``ms_cancel_request_to_fill``.
                "ALTER TABLE orders ADD COLUMN ts_cancel_requested TEXT",
                # fills table — trivial derivations.
                "ALTER TABLE fills ADD COLUMN spread_bps_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN microprice_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN imbalance_top_at_fill REAL",
                "ALTER TABLE fills ADD COLUMN inventory_qty_before_fill REAL",
                "ALTER TABLE fills ADD COLUMN inventory_utilization_before_fill REAL",
                # fills table — decision-state propagation (mirrors orders).
                "ALTER TABLE fills ADD COLUMN toxicity_score_at_decision REAL",
                "ALTER TABLE fills ADD COLUMN vol_estimate_at_decision REAL",
                "ALTER TABLE fills ADD COLUMN active_sides_at_decision TEXT",
                "ALTER TABLE fills ADD COLUMN decision_reason_at_decision TEXT",
                "ALTER TABLE fills ADD COLUMN binance_basis_ewma_at_decision REAL",
                "ALTER TABLE fills ADD COLUMN adaptive_widen_active_at_decision INTEGER",
                "ALTER TABLE fills ADD COLUMN post_fill_cooldown_active_bid_at_decision INTEGER",
                "ALTER TABLE fills ADD COLUMN post_fill_cooldown_active_ask_at_decision INTEGER",
                "ALTER TABLE fills ADD COLUMN at_touch_adverse_pause_bid_at_decision INTEGER",
                "ALTER TABLE fills ADD COLUMN at_touch_adverse_pause_ask_at_decision INTEGER",
                "ALTER TABLE fills ADD COLUMN quote_distance_to_touch_ticks_at_placement REAL",
                # fills table — cancel-race diagnostics.
                "ALTER TABLE fills ADD COLUMN cancel_requested_before_fill INTEGER",
                "ALTER TABLE fills ADD COLUMN ms_cancel_request_to_fill REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 27:
            # 2026-05-13 regime-observability Phase 2: exposure bars.
            # New table capturing periodic (default 5s) snapshots of
            # market + strategy state regardless of whether a fill
            # happened. Without this "exposure denominator", per-regime
            # analytics measure outcomes but not how long the bot was
            # exposed to a regime — making fills-per-regime ratios
            # uninterpretable. See ``plans/regime-observability.md``
            # Phase 2 for the rationale.
            #
            # Cadence: configurable via ``OBSERVABILITY_EXPOSURE_BAR_INTERVAL_SECONDS``
            # (default 5.0). At 5s, a 15h session writes ~10.8K rows
            # (~10MB raw, ~1MB compressed in tarballs).
            #
            # Schema: keep flat (no nested types) for SQLite + Parquet
            # portability. Bools serialize as INTEGER NULL/0/1.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS exposure_bars (
                    session_id TEXT NOT NULL,
                    ts_bar TEXT NOT NULL,
                    symbol TEXT,
                    -- Market state
                    mid REAL,
                    spread_bps REAL,
                    microprice REAL,
                    imbalance_top REAL,
                    bid_size_top REAL,
                    ask_size_top REAL,
                    -- Strategy state
                    inventory_qty REAL,
                    inventory_utilization REAL,
                    active_sides TEXT,
                    quote_eligibility TEXT,
                    toxicity_score REAL,
                    vol_estimate REAL,
                    -- Basis (cross-venue)
                    binance_basis_ewma REAL,
                    basis_regime_sign INTEGER,
                    -- Liquidity provision (this bar's working orders)
                    bid_live INTEGER,
                    ask_live INTEGER,
                    bid_distance_ticks REAL,
                    ask_distance_ticks REAL,
                    quoted_spread_bps REAL,
                    -- Gates
                    adaptive_widen_active INTEGER,
                    hold_all_active INTEGER,
                    recovery_cooldown_active INTEGER,
                    post_fill_cooldown_active_bid INTEGER,
                    post_fill_cooldown_active_ask INTEGER,
                    at_touch_adverse_pause_bid INTEGER,
                    at_touch_adverse_pause_ask INTEGER
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exposure_bars_ts "
                "ON exposure_bars(ts_bar)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_exposure_bars_session "
                "ON exposure_bars(session_id)"
            )
        if from_v < 28:
            # 2026-05-13 regime-observability Phase 4a: stamp
            # ``expected_net_edge_bps_at_decision`` on both orders
            # and fills. STAMP ONLY — not a quote-construction input.
            # See plans/regime-observability.md Phase 4a scope guard
            # for the explicit non-use rule.
            for stmt in (
                "ALTER TABLE orders ADD COLUMN expected_net_edge_bps_at_decision REAL",
                "ALTER TABLE fills ADD COLUMN expected_net_edge_bps_at_decision REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 29:
            # 2026-05-13 regime-observability Phase 4c: MAE/MFE
            # excursion columns on fills. Written LATE by
            # ``PostFillExcursionWatcher`` via UPDATE at the 5s and
            # 30s deadlines after each fill. NULL on the fill row
            # immediately after ingestion; populated by the watcher
            # within 30s if the feature is enabled.
            for stmt in (
                "ALTER TABLE fills ADD COLUMN mae_5s_bps REAL",
                "ALTER TABLE fills ADD COLUMN mfe_5s_bps REAL",
                "ALTER TABLE fills ADD COLUMN mae_30s_bps REAL",
                "ALTER TABLE fills ADD COLUMN mfe_30s_bps REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 30:
            # 2026-05-13 regime-observability Phase 4c follow-up:
            # time-to-flat column. Written LATE by
            # ``PostFillTimeToFlatWatcher`` when position_qty
            # crosses zero post-fill. NULL when the watcher is
            # disabled, the position didn't return to flat within
            # the configured cap, or the session ended unflat.
            try:
                conn.execute(
                    "ALTER TABLE fills ADD COLUMN time_to_flat_seconds REAL"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 31:
            # 2026-05-14 (todo-030): add the cross-venue basis EWMA
            # to ``equity_snapshots`` so the dashboard's Session-PnL
            # chart can render a basis sub-band synced with the PnL
            # line. Without this column the dashboard would have to
            # join against ``exposure_bars`` (5s cadence, different
            # row-density) or ``quote_decisions`` (per-cycle,
            # extremely dense, expensive to scan) for a value the
            # equity-snapshot writer can stamp at trivial cost (one
            # float read off ``BotState`` it already locks for the
            # rest of the row).
            #
            # Bot.py:_persist_equity_snapshot is the writer; the new
            # column is populated alongside ``mid_price`` / ``vol_bps``
            # / ``position_qty`` (all v22-v24 sub-band feeds). NULL on
            # every pre-existing row and on rows written when
            # ``BINANCE_WS_ENABLED=false`` or the EWMA hasn't seeded
            # yet — same NULL semantics as
            # ``quote_decisions.binance_basis_ewma`` (v14 addition).
            try:
                conn.execute(
                    "ALTER TABLE equity_snapshots ADD COLUMN binance_basis_ewma REAL"
                )
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        if from_v < 32:
            # 2026-05-14 (todo-027 Tier 2): full instrumentation for
            # three "fire-counted-only" gates so the dashboard's
            # gate-effectiveness table can populate every column for
            # them instead of greying them out as
            # "(needs backend — todo-027 Tier 2)".
            #
            # The three gates today expose only a session-cumulative
            # ``fire_count`` via ``live_stats``; we don't know how
            # much TIME they spent active, or WHICH fills landed
            # while they were active. Adding bar-level + per-fill
            # snapshots closes the gap without touching the gates'
            # decision logic — these columns are STAMP-ONLY (the
            # bot's quote-construction path doesn't read them back).
            #
            # On exposure_bars: three new flags written by the
            # ExposureBarEmitter every 5 s alongside the existing
            # ``adaptive_widen_active`` / ``hold_all_active`` etc.
            #   * vol_trend_active            — derived from
            #     ``state.vol_trend_gate.cooldown_until_mono`` vs
            #     ``_clock.monotonic()`` (same pattern as
            #     ``adaptive_widen_active`` already in this table).
            #   * post_swing_active           — same pattern off
            #     ``state.post_swing.cooldown_until_mono``.
            #   * session_drawdown_tier       — string tier label off
            #     ``state.session_drawdown.tier`` (CLEAR / WIDEN /
            #     PAUSE_SHORT / PAUSE_LONG / RESUME_TESTING / KILLED).
            #     TEXT (not INTEGER) because the tier carries more
            #     than a binary "active/inactive" — operator wants to
            #     differentiate "spread widened" from "long pause".
            #
            # On orders + fills: mirror the existing
            # ``adaptive_widen_active_at_decision`` pattern — three
            # ``*_at_decision`` columns stamped at place-time on the
            # parent order, propagated to the resulting Fill via the
            # existing ``order_decision_state_for_order`` lookup.
            for stmt in (
                "ALTER TABLE exposure_bars ADD COLUMN vol_trend_active INTEGER",
                "ALTER TABLE exposure_bars ADD COLUMN post_swing_active INTEGER",
                "ALTER TABLE exposure_bars ADD COLUMN session_drawdown_tier TEXT",
                "ALTER TABLE orders ADD COLUMN vol_trend_active_at_decision INTEGER",
                "ALTER TABLE orders ADD COLUMN post_swing_active_at_decision INTEGER",
                "ALTER TABLE orders ADD COLUMN session_drawdown_tier_at_decision TEXT",
                "ALTER TABLE fills ADD COLUMN vol_trend_active_at_decision INTEGER",
                "ALTER TABLE fills ADD COLUMN post_swing_active_at_decision INTEGER",
                "ALTER TABLE fills ADD COLUMN session_drawdown_tier_at_decision TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 33:
            # 2026-05-15 (1.3.82): connectivity-diagnostic instrumentation.
            # Four new fields stamped on every order so the dashboard's
            # Connectivity tab can attribute every gone_on_exchange /
            # phantom-place / late-cancel event to the specific code
            # path + venue response that produced it.
            #
            #   cancel_trigger_reason   — local-side: which decision
            #     enqueued the cancel (reprice_replace / hard_age_cap /
            #     side_suppressed / soft_flatten / etc.). NULL when
            #     no cancel was ever requested.
            #   ts_place_response       — when the place HTTP response
            #     was interpreted. NULL when never received (the
            #     phantom-place signature).
            #   place_response_outcome  — venue-side: accepted /
            #     exchange_rejected / transport_rejected / unconfirmed.
            #   cancel_response_outcome — venue-side: success /
            #     benign_missing / transport / error.
            #
            # All four are STAMP-ONLY — the quote-construction path
            # never reads them back. They exist so reconcile +
            # postmortem + dashboard can answer "what happened to
            # this order?" without grepping live logs.
            for stmt in (
                "ALTER TABLE orders ADD COLUMN cancel_trigger_reason TEXT",
                "ALTER TABLE orders ADD COLUMN ts_place_response TEXT",
                "ALTER TABLE orders ADD COLUMN place_response_outcome TEXT",
                "ALTER TABLE orders ADD COLUMN cancel_response_outcome TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 34:
            # 2026-05-15 (1.3.83): venue-side detail strings alongside
            # the v33 outcome categories. Lets the Connectivity tab
            # surface "what specifically did OKX reject this for"
            # (e.g. post_only_would_cross vs insufficient_margin vs
            # rate_limit) instead of only the high-level bucket.
            for stmt in (
                "ALTER TABLE orders ADD COLUMN place_response_detail TEXT",
                "ALTER TABLE orders ADD COLUMN cancel_response_detail TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 35:
            # 2026-05-16 (1.4.0 cancel-prio Phase 0.5): cancel-latency
            # decomposition. The existing ``cancel_to_close_ms`` derived
            # column (= ``ts_closed - ts_cancel_requested``) bundles 6
            # legs into one number. These three new columns let us
            # split the cancel path into ``decision → send`` and
            # ``send → ack-via-HTTP`` legs, plus a sanity-check anchor
            # against OKX's server-side cancel timestamp from the WS
            # ``orders`` channel ``uTime`` field.
            #
            # All three are stamp-only; the bot's logic never reads
            # them back into the quote-construction path.
            for stmt in (
                "ALTER TABLE orders ADD COLUMN ts_cancel_sent TEXT",
                "ALTER TABLE orders ADD COLUMN ts_cancel_acked TEXT",
                "ALTER TABLE orders ADD COLUMN venue_cancel_utime_ms INTEGER",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 36:
            # 2026-05-17 (1.4.15 amend-prio Phase 1): amend lifecycle
            # columns on the orders table. The bot now supports
            # AMEND_PENDING as a WO status; these columns persist the
            # amend's target price/size, response classification, and
            # send/response timestamps so the postmortem can attribute
            # per-fill amend behaviour (queue-position preservation,
            # below-filled fallback rate, etc.).
            #
            # Legacy rows have NULLs across all six columns and are
            # treated as "amend never attempted" by downstream code.
            for stmt in (
                "ALTER TABLE orders ADD COLUMN amend_intent_seq INTEGER DEFAULT 0",
                "ALTER TABLE orders ADD COLUMN amend_target_px REAL",
                "ALTER TABLE orders ADD COLUMN amend_target_sz REAL",
                "ALTER TABLE orders ADD COLUMN ts_amend_sent TEXT",
                "ALTER TABLE orders ADD COLUMN ts_amend_response TEXT",
                "ALTER TABLE orders ADD COLUMN amend_response_outcome TEXT",
                "ALTER TABLE orders ADD COLUMN amend_response_detail TEXT",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 37:
            # v1.4.98 — extended diagnostic markout horizons aligned to
            # the bot's gate time-constants. The 1s/3s/5s columns above
            # are unchanged; these capture the inflection points the
            # bot's gates actually operate on (15s post-first-cancel,
            # 30s soft-flatten window, 60s post-swing window, 120s
            # long-tail held-position settlement).
            #
            # All four are diagnostic-only — no gate reads them. Legacy
            # rows have NULL across all four columns. See
            # ``app/markout.py`` for the resolution loop and
            # ``app/models.py:Fill`` for the field semantics.
            for stmt in (
                "ALTER TABLE fills ADD COLUMN markout_15s_bps REAL",
                "ALTER TABLE fills ADD COLUMN markout_30s_bps REAL",
                "ALTER TABLE fills ADD COLUMN markout_60s_bps REAL",
                "ALTER TABLE fills ADD COLUMN markout_120s_bps REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 38:
            # v1.4.100 ladder-observability F1 — `level_idx` on fills
            # AND on orders. The corresponding `WorkingOrder.level_idx`
            # field has existed since v1.3.130 (carrying the per-rung
            # index 0 = inside / 1+ = outer in memory) but was never
            # persisted to the `orders` table; this migration makes
            # it queryable on both tables so:
            #
            #   * The postmortem can decompose per-rung markouts, fill
            #     rates, and net edge (the headline calibration artifact
            #     for the N=2 → N=3 decision — plans/ladder-observability.md O1).
            #   * The fill-ingestion path can copy `level_idx` from the
            #     parent order row (joined via `order_id_exchange`) to
            #     the fill row, so historical fills are correctly
            #     bucketed by rung.
            #
            # Legacy rows (pre-v1.4.100) default to 0, which buckets
            # them as "single-rung-equivalent". Correct for the long
            # pre-multi-rung era; for the v1.4.0–v1.4.99 multi-rung
            # era it understates rung-1 economics in historical data.
            # Going forward (v1.4.100+) every order persist + every
            # fill ingest stamps the column from the live in-memory
            # value.
            for stmt in (
                "ALTER TABLE fills ADD COLUMN level_idx INTEGER DEFAULT 0",
                "ALTER TABLE orders ADD COLUMN level_idx INTEGER DEFAULT 0",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 40:
            # v1.4.173 (Phase 4D.4) — SF phase-ladder observability.
            # Three additive columns so the operator can decompose every
            # SF episode by which phase of the adaptive-aggressiveness
            # ladder (v1.4.172 wiring) each fill landed in:
            #
            #   * ``fills.sf_force_phase`` — integer in {0..4}, copied
            #     at fill-ingest time from ``state.sf_phase_ladder_phase``
            #     when SF is active. NULL on non-SF fills and on legacy
            #     pre-ladder fills. Used by the per-episode rollup below
            #     + by the postmortem to slice realised markouts /
            #     spreads-paid by ladder phase.
            #
            #   * ``soft_flatten_events.fills_by_phase_json`` — JSON
            #     dict like ``{"0":3,"1":2,"2":1,"3":0,"4":0}`` written
            #     by ``_exit_soft_flatten`` from a SELECT over the
            #     episode's fills. Stored as JSON because the schema is
            #     stable-but-extensible (a future Phase 4D iteration
            #     might add phase 5 or per-side sub-buckets without
            #     another migration). NULL on episodes that exited
            #     before this version landed.
            #
            #   * ``soft_flatten_events.taker_spread_bps_paid`` — single
            #     REAL: notional-weighted average of per-fill cross-
            #     spread cost (in bps vs the SF entry mid) across all
            #     non-passive fills (phases 2/3/4) in this episode. The
            #     calibration knob this lets the operator tune is
            #     "average bps paid per SF crossing"; goal per the
            #     v1.4.157 postmortem is to bring it BELOW the legacy
            #     market_close worst-case (≈ 30 bp on the failing day)
            #     toward the cross-1-tick floor (≈ 5 bp on TON).
            #
            # All three default to NULL on legacy rows; the publisher
            # + dashboard treat NULL as "ladder not active during this
            # episode" and skip the phase-mix tooltip.
            for stmt in (
                "ALTER TABLE fills ADD COLUMN sf_force_phase INTEGER",
                "ALTER TABLE soft_flatten_events "
                "ADD COLUMN fills_by_phase_json TEXT",
                "ALTER TABLE soft_flatten_events "
                "ADD COLUMN taker_spread_bps_paid REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 41:
            # v1.4.175 Phase 3F — reservation-alpha attribution.
            # Four shift contributions in bps captured AT DECISION TIME
            # from ``state.last_quote_breakdown``, persisted on the
            # parent ``orders`` row and copied to ``fills`` at ingest
            # time (mirror of the storage v23 ``_at_decision`` pattern).
            #
            # Reservation-price decomposition: at each quote tick,
            # ``compute_quote_decision`` sets the reservation price as
            # ``ref_price_anchor + sum(shift_bps)``. The four shifts:
            #
            #   * ``ob_imbalance_shift_bps_at_decision`` — order-book
            #     imbalance lever (``OB_IMBALANCE_ALPHA``).
            #   * ``trend_drift_shift_bps_at_decision`` — short-window
            #     trend lever (``TREND_DRIFT_RESERVATION_ALPHA``).
            #   * ``flow_score_shift_bps_at_decision`` — taker-flow
            #     pressure lever (``FLOW_SCORE_RESERVATION_ALPHA``).
            #   * ``basis_deviation_shift_bps_at_decision`` — cross-
            #     venue basis lever (``BASIS_DEVIATION_ALPHA``,
            #     currently dormant @ 0.0 in TON prod).
            #
            # Each is the *signed* contribution applied during the
            # decision that produced this order. Positive = shift moved
            # reservation UP (favors selling); negative = shift moved
            # reservation DOWN (favors buying). The postmortem
            # ``reservation_alpha_attribution`` section reads these +
            # the per-fill realised markout to compute per-alpha
            # directional accuracy + a counterfactual (what would the
            # markout have been without this alpha's contribution).
            #
            # Verdict thresholds — see plan 3F.3:
            #   * helpful: directional accuracy > 0.55 AND
            #     |cumulative bps contributed| > 0.5 bp
            #   * noise:   directional accuracy in [0.45, 0.55]
            #   * harmful: directional accuracy < 0.45
            #
            # All four columns default NULL on legacy / pre-migration
            # rows. The postmortem section omits an alpha when n < 100
            # fills with non-null shifts (insufficient power for a
            # verdict).
            for tbl in ("orders", "fills"):
                for col in (
                    "ob_imbalance_shift_bps_at_decision",
                    "trend_drift_shift_bps_at_decision",
                    "flow_score_shift_bps_at_decision",
                    "basis_deviation_shift_bps_at_decision",
                ):
                    try:
                        conn.execute(
                            f"ALTER TABLE {tbl} ADD COLUMN {col} REAL"
                        )
                    except sqlite3.OperationalError as e:
                        if "duplicate column" not in str(e).lower():
                            raise
        if from_v < 42:
            # v1.5.33 — take-profit (TP) opportunistic harvest mode.
            # Mirror of the v15 ``soft_flatten_events`` migration: one
            # row per TP episode + FK columns on orders & fills so the
            # dashboard can colour TP-attributed rows and the
            # postmortem can attribute taker_spread / dwell / outcome
            # back to each episode. Additive migration; NULL on every
            # pre-existing row.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tp_events (
                    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_start               TEXT NOT NULL,
                    ts_end                 TEXT,
                    trigger_upnl_bps       REAL,
                    trigger_threshold_bps  REAL,
                    disarm_margin_bps      REAL,
                    entry_position_qty     REAL,
                    entry_mid_price        REAL,
                    entry_target_price     REAL,
                    close_side             TEXT,
                    exit_reason            TEXT,
                    exit_upnl_bps          REAL,
                    fills_count            INTEGER,
                    fills_notional_usd     REAL,
                    notes                  TEXT
                )
                """
            )
            for stmt in (
                "ALTER TABLE orders ADD COLUMN tp_event_id INTEGER",
                "ALTER TABLE fills  ADD COLUMN tp_event_id INTEGER",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_tp_events_ts_start "
                "ON tp_events(ts_start)",
                "CREATE INDEX IF NOT EXISTS idx_orders_tp_event_id "
                "ON orders(tp_event_id)",
                "CREATE INDEX IF NOT EXISTS idx_fills_tp_event_id "
                "ON fills(tp_event_id)",
            ):
                conn.execute(stmt)
        if from_v < 43:
            # v1.5.189 (Phase 8A Option B instrumentation) — per-sample
            # AS attribution columns on ``equity_snapshots``. Lets the
            # dashboard chart (and the postmortem) reconstruct the
            # AS-computed base half-spread + k-intensity time-series
            # alongside vol_bps. Both columns are NULL on pre-v1.5.189
            # rows and on rows written with AS disabled.
            for stmt in (
                "ALTER TABLE equity_snapshots ADD COLUMN "
                "base_half_spread_bps REAL",
                "ALTER TABLE equity_snapshots ADD COLUMN "
                "as_k_intensity_per_min REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 44:
            # v1.5.190 (Phase 8A Option C — per-fill AS attribution).
            # Adds the AS-base half-spread snapshot column to both
            # ``orders`` (stamped at place-time from
            # ``state.last_quote_breakdown.base_half_spread_bps``) and
            # ``fills`` (copied at ingest time via
            # ``order_metadata_for_fill_ingest``). Lets the postmortem
            # decompose per-fill captured spread into "AS-base
            # contribution" vs. "skew / overlay / inventory widen
            # contribution" without re-deriving from the time-series.
            # NULL on legacy rows + on paths where no breakdown was
            # recorded (SF / TP / manual).
            for stmt in (
                "ALTER TABLE orders ADD COLUMN "
                "as_base_half_spread_bps_at_decision REAL",
                "ALTER TABLE fills  ADD COLUMN "
                "as_base_half_spread_bps_at_decision REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 45:
            # v1.5.204 (Phase 4A — microprice widen per-fill attribution).
            # Stamps the per-side microprice gate widening (bps) at
            # decision time onto each order; copied to every resulting
            # fill via the same lookup as the other Phase 3F shifts.
            # Lets the Phase 4A.3 acceptance check + the
            # ``microprice_widen_attribution`` postmortem section answer
            # "did fills landing during a microprice-gate-firing tick
            # show better/worse markout than fills outside?" — i.e. is
            # the configured ``MICROPRICE_WIDEN_BPS`` value paying for
            # its protection. Per-side because the gate is asymmetric:
            # only the thin side gets widened.
            for stmt in (
                "ALTER TABLE orders ADD COLUMN "
                "microprice_bid_widen_bps_at_decision REAL",
                "ALTER TABLE orders ADD COLUMN "
                "microprice_ask_widen_bps_at_decision REAL",
                "ALTER TABLE fills  ADD COLUMN "
                "microprice_bid_widen_bps_at_decision REAL",
                "ALTER TABLE fills  ADD COLUMN "
                "microprice_ask_widen_bps_at_decision REAL",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        if from_v < 46:
            # v1.5.306 (audit §5 P0 #2 — Active-Quoting-Controller
            # attribution). TWO independent stamping mechanisms, both
            # observability-only (the AQC's effect on quoting already
            # flows through min_half_spread / inventory gates / skew —
            # these columns just RECORD what the controller's state was,
            # they are never read back into quote construction).
            #
            # (1) PER-TICK trace into ``quote_decisions`` (audit §4.1).
            #     Written every tick (including no-order ticks) from
            #     ``state.active_quoting_controller``. Lets the replay
            #     report + postmortem answer "what was the controller
            #     doing on ticks where nothing got placed" — the
            #     per-fill stamp below can't, because it only persists
            #     when an order is actually placed.
            #       * aqc_aggression_level            — PI output [0,1]
            #       * aqc_integrator                  — PI integrator term
            #       * aqc_safety_floor_engaged        — 0/1 markout brake
            #       * aqc_observed_net_edge_per_min_usd — primary signal
            #       * aqc_observed_markout_5s_mean_bps  — safety-brake input
            #
            # (2) PER-FILL attribution via ``_DECISION_STATE_COLS``
            #     (audit §4.2). Stamped on the WorkingOrder at place-time
            #     in execution.py; copied to each resulting fill via
            #     ``order_metadata_for_fill_ingest``. Lets the postmortem
            #     ask "did fills placed while aggression was high / while
            #     the safety floor was engaged show better/worse markout?"
            #     Only the two control-relevant scalars are propagated to
            #     fills (aggression + floor flag) — the raw observation
            #     inputs live on the per-tick trace, not per fill.
            for stmt in (
                "ALTER TABLE quote_decisions ADD COLUMN "
                "aqc_aggression_level REAL",
                "ALTER TABLE quote_decisions ADD COLUMN "
                "aqc_integrator REAL",
                "ALTER TABLE quote_decisions ADD COLUMN "
                "aqc_safety_floor_engaged INTEGER",
                "ALTER TABLE quote_decisions ADD COLUMN "
                "aqc_observed_net_edge_per_min_usd REAL",
                "ALTER TABLE quote_decisions ADD COLUMN "
                "aqc_observed_markout_5s_mean_bps REAL",
                "ALTER TABLE orders ADD COLUMN "
                "aqc_aggression_level_at_decision REAL",
                "ALTER TABLE orders ADD COLUMN "
                "aqc_safety_floor_engaged_at_decision INTEGER",
                "ALTER TABLE fills  ADD COLUMN "
                "aqc_aggression_level_at_decision REAL",
                "ALTER TABLE fills  ADD COLUMN "
                "aqc_safety_floor_engaged_at_decision INTEGER",
            ):
                try:
                    conn.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column" not in str(e).lower():
                        raise
        conn.execute(f"PRAGMA user_version = {int(self.SCHEMA_VERSION)}")

    def insert_order_row(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    f"INSERT OR REPLACE INTO orders ({cols}) VALUES ({placeholders})",
                    tuple(row.values()),
                )

    def insert_fill_row(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    f"INSERT OR REPLACE INTO fills ({cols}) VALUES ({placeholders})",
                    tuple(row.values()),
                )

    def update_fill_excursion(
        self,
        fill_id: str,
        *,
        mae_5s_bps: Optional[float] = None,
        mfe_5s_bps: Optional[float] = None,
        mae_30s_bps: Optional[float] = None,
        mfe_30s_bps: Optional[float] = None,
    ) -> None:
        """Late UPDATE of MAE/MFE columns on a fill row. Called by
        ``PostFillExcursionWatcher`` at the 5s and 30s deadlines.

        Each column uses ``COALESCE(?, col)`` semantics so callers can
        update only the 5s pair or only the 30s pair without clobbering
        the other.

        2026-05-13 regime-observability Phase 4c.
        """
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    """
                    UPDATE fills SET
                      mae_5s_bps  = COALESCE(?, mae_5s_bps),
                      mfe_5s_bps  = COALESCE(?, mfe_5s_bps),
                      mae_30s_bps = COALESCE(?, mae_30s_bps),
                      mfe_30s_bps = COALESCE(?, mfe_30s_bps)
                    WHERE fill_id = ?
                    """,
                    (mae_5s_bps, mfe_5s_bps, mae_30s_bps, mfe_30s_bps, fill_id),
                )

    def update_fill_time_to_flat(
        self,
        fill_id: str,
        time_to_flat_seconds: float,
    ) -> None:
        """Late UPDATE of ``time_to_flat_seconds`` on a fill row. Called
        by ``PostFillTimeToFlatWatcher`` when ``state.position.position_qty``
        crosses zero post-fill.

        Idempotent: COALESCE with existing value means re-writing
        the same fill (rare; watcher dedupes) preserves the first
        observation rather than over-writing with a later crossing.

        2026-05-13 regime-observability Phase 4c follow-up.
        """
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    """
                    UPDATE fills SET
                      time_to_flat_seconds = COALESCE(time_to_flat_seconds, ?)
                    WHERE fill_id = ?
                    """,
                    (float(time_to_flat_seconds), fill_id),
                )

    def update_fill_markouts(
        self,
        fill_id: str,
        m1: Optional[float],
        m3: Optional[float],
        m5: Optional[float],
        m15: Optional[float] = None,
        m30: Optional[float] = None,
        m60: Optional[float] = None,
        m120: Optional[float] = None,
    ) -> None:
        """Update per-fill markout columns.

        v1.4.98 — added m15 / m30 / m60 / m120 (default None). Pre-
        v1.4.98 callers pass only the first three positional args; the
        new horizon args take their None defaults and the COALESCE
        preserves whatever was there.
        """
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    """
                    UPDATE fills SET markout_1s_bps = COALESCE(?, markout_1s_bps),
                    markout_3s_bps = COALESCE(?, markout_3s_bps),
                    markout_5s_bps = COALESCE(?, markout_5s_bps),
                    markout_15s_bps = COALESCE(?, markout_15s_bps),
                    markout_30s_bps = COALESCE(?, markout_30s_bps),
                    markout_60s_bps = COALESCE(?, markout_60s_bps),
                    markout_120s_bps = COALESCE(?, markout_120s_bps)
                    WHERE fill_id = ?
                    """,
                    (m1, m3, m5, m15, m30, m60, m120, fill_id),
                )

    def update_fill_markouts_many(
        self,
        rows: list[
            tuple[
                str,
                Optional[float],
                Optional[float],
                Optional[float],
                # v1.4.98 — these four are added as positional 5..8.
                # Older callers passing 4-tuples are still supported
                # via the normalisation step below.
                Optional[float],
                Optional[float],
                Optional[float],
                Optional[float],
            ]
        ],
    ) -> None:
        """v1.4.37 (Codex #6) — batch finalise markouts in ONE
        transaction. v1.4.98 — extended to 7 horizons.

        Pre-v1.4.37 ``markout.process_pending_markouts`` looped one
        UPDATE per fill; each opened its own transaction + commit.
        Under bursty fill windows with multiple horizons resolving
        in the same tick, that produced N synchronous commits per
        tick → write amplification + lock contention with fill
        ingestion and live-stats / heartbeat readers.

        This batched method runs N ``UPDATE`` statements inside ONE
        ``self.connection()`` context (= one transaction), under a
        single ``self._lock`` acquisition. The lock is held for the
        duration of the batch but the read-side waits are amortised
        across N updates instead of N separate acquisitions.

        ``rows`` is an iterable of tuples — pre-v1.4.98 callers may
        still pass 4-tuples ``(fill_id, m1, m3, m5)``; the
        normalisation step below pads them to the 8-tuple shape with
        Nones in the extended-horizon slots. New callers should pass
        the full 8-tuple.

        ``None`` values in any markout slot preserve the existing
        column value (COALESCE semantics). Empty / null input is a
        no-op.
        """
        if not rows:
            return
        # Normalise to 8-tuple shape for the legacy 4-tuple callers.
        normalized: list[
            tuple[
                Optional[float], Optional[float], Optional[float],
                Optional[float], Optional[float], Optional[float],
                Optional[float], str,
            ]
        ] = []
        for r in rows:
            if len(r) == 4:
                fill_id, m1, m3, m5 = r
                m15 = m30 = m60 = m120 = None
            elif len(r) == 8:
                fill_id, m1, m3, m5, m15, m30, m60, m120 = r
            else:
                # Defensive: skip unrecognised shapes rather than
                # corrupt the batch.
                continue
            normalized.append((m1, m3, m5, m15, m30, m60, m120, fill_id))
        if not normalized:
            return
        with self._lock:
            with self.connection() as conn:
                conn.executemany(
                    """
                    UPDATE fills SET markout_1s_bps = COALESCE(?, markout_1s_bps),
                    markout_3s_bps = COALESCE(?, markout_3s_bps),
                    markout_5s_bps = COALESCE(?, markout_5s_bps),
                    markout_15s_bps = COALESCE(?, markout_15s_bps),
                    markout_30s_bps = COALESCE(?, markout_30s_bps),
                    markout_60s_bps = COALESCE(?, markout_60s_bps),
                    markout_120s_bps = COALESCE(?, markout_120s_bps)
                    WHERE fill_id = ?
                    """,
                    normalized,
                )

    def insert_position_snapshot(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    f"INSERT INTO position_snapshots ({cols}) VALUES ({placeholders})",
                    tuple(row.values()),
                )

    def insert_equity_snapshot(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    f"INSERT INTO equity_snapshots ({cols}) VALUES ({placeholders})",
                    tuple(row.values()),
                )

    def insert_exposure_bar(self, row: dict[str, Any]) -> None:
        """Append one exposure-bar row.

        2026-05-13 regime-observability Phase 2.

        Single dict-based write — same pattern as ``insert_fill_row`` /
        ``insert_order_row``. Called from
        ``ExposureBarEmitter._loop`` on its own thread; the storage
        lock is acquired briefly for the commit, then released. The
        strategy thread does NOT call this — it has no awareness of
        the emitter. Hot-path-invisible by construction.
        """
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    f"INSERT INTO exposure_bars ({cols}) VALUES ({placeholders})",
                    tuple(row.values()),
                )

    def exposure_bars_since(
        self,
        since_iso: str,
        *,
        until_iso: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100000,
        most_recent: bool = False,
    ) -> list[dict[str, Any]]:
        """Return exposure-bar rows in ``[since_iso, until_iso)`` window,
        ordered by ``ts_bar`` ascending. Used by the
        ``/exposure_since`` REST endpoint + the snapshot pipeline.

        ``limit`` defaults high (100K) because a 15h session at 5s
        cadence is ~10.8K rows — well below. Clamping prevents
        accidental gigantic responses if the cadence is set absurdly
        low.

        ``most_recent=True`` (1.3.99): fetch with ``ORDER BY ts_bar
        DESC LIMIT ?`` and reverse client-side, so the caller still
        gets oldest-first rows but they represent the *most recent*
        ``limit`` bars. Used by the dashboard publisher whose chart
        tracks the live tail — with the default ASC + LIMIT path,
        long sessions clip at the OLDEST ``limit`` bars and the
        Gate-activity strip / Eligibility-state band freeze at the
        first ~2h45m of the session. Mirrors the same fix on
        :meth:`equity_history_since` (see docstring there).

        2026-05-13 regime-observability Phase 2.
        """
        sql = "SELECT * FROM exposure_bars WHERE ts_bar >= ?"
        params: list[Any] = [since_iso]
        if until_iso:
            sql += " AND ts_bar < ?"
            params.append(until_iso)
        if symbol:
            sql += " AND (symbol = ? OR symbol IS NULL)"
            params.append(symbol)
        if most_recent:
            sql += " ORDER BY ts_bar DESC LIMIT ?"
        else:
            sql += " ORDER BY ts_bar ASC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            with self.connection() as conn:
                rows = self._rows(conn, sql, tuple(params))
        if most_recent:
            rows.reverse()
        return rows

    def insert_quote_decision(self, row: dict[str, Any]) -> None:
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    f"INSERT INTO quote_decisions ({cols}) VALUES ({placeholders})",
                    tuple(row.values()),
                )

    def insert_bot_event(
        self,
        ts: str,
        severity: str,
        event_type: str,
        message: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        payload_json = json.dumps(payload) if payload else None
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO bot_events (ts, severity, event_type, message, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ts, severity, event_type, message, payload_json),
                )

    def _rows(self, conn: sqlite3.Connection, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def fills_since(
        self,
        since_ts: str,
        limit: int = 10000,
        *,
        until_ts: Optional[str] = None,
        most_recent: bool = False,
    ) -> list[dict[str, Any]]:
        """Fills with ``ts_fill >= since_ts``, oldest-first, up to ``limit`` rows.

        Optional keyword-only ``until_ts`` caps the range (exclusive-upper)
        for bounded window queries. Kept positional-compat with the
        existing two-arg call sites in ``api.py`` (``/pnl/attribution``,
        ``/session/summary``).

        Use this instead of :meth:`recent_fills` when you need session-level
        history (analytics / attribution). Main tables are NOT session-scoped
        on disk (see module docstring — the DB accumulates across restarts
        unless explicitly wiped), so session-scoping is achieved by passing
        ``since_ts = session_started_at_utc``.

        ``most_recent=True`` (1.3.99): fetch with ``ORDER BY ts_fill DESC
        LIMIT ?`` and reverse client-side, so the caller still gets
        oldest-first rows but they represent the *most recent*
        ``limit`` fills. Used by the dashboard publisher's
        ``fills_since`` artifact whose chart tracks the live tail —
        with the default ASC + LIMIT path, long sessions clip at the
        OLDEST ``limit`` fills and the Regime / Fill-Drilldown / Bot
        Stats tabs all freeze on those early fills. Mirrors the same
        fix on :meth:`equity_history_since`,
        :meth:`exposure_bars_since`, :meth:`orders_lifecycle_since`.
        """
        sql = "SELECT * FROM fills WHERE ts_fill >= ?"
        params: list[Any] = [since_ts]
        if until_ts is not None:
            sql += " AND ts_fill < ?"
            params.append(until_ts)
        if most_recent:
            sql += " ORDER BY ts_fill DESC LIMIT ?"
        else:
            sql += " ORDER BY ts_fill ASC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            with self.connection() as conn:
                rows = self._rows(conn, sql, tuple(params))
        if most_recent:
            rows.reverse()
        return rows

    def equity_history_since(
        self,
        since_ts: str,
        limit: int = 10000,
        *,
        until_ts: Optional[str] = None,
        most_recent: bool = False,
    ) -> list[dict[str, Any]]:
        """Equity snapshots with ``ts >= since_ts``, oldest-first.

        When ``most_recent=True``, the SQL fetches with ``ORDER BY ts
        DESC LIMIT ?`` and we reverse client-side, so the returned
        rows are still oldest-first to the caller but represent the
        *most recent* ``limit`` samples within the window. Used by
        the equity-history publisher so its rolling-window chart
        keeps tracking the live tail when sessions exceed the cap.
        With the default ``False`` behaviour (oldest-first ASC + LIMIT)
        the chart would freeze at the oldest 6h40min once a session
        crossed that mark.
        """
        sql = "SELECT * FROM equity_snapshots WHERE ts >= ?"
        params: list[Any] = [since_ts]
        if until_ts is not None:
            sql += " AND ts < ?"
            params.append(until_ts)
        if most_recent:
            sql += " ORDER BY ts DESC LIMIT ?"
        else:
            sql += " ORDER BY ts ASC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            with self.connection() as conn:
                rows = self._rows(conn, sql, tuple(params))
        if most_recent:
            rows.reverse()
        return rows

    # v1.5.275 (BUG-038 fix): the set of "structural" event types
    # whose retention is privileged over the generic LIMIT cap below.
    # On a long session, high-rate trash events (e.g.
    # ``target_venue_fast_move_cancel`` at 6 167 rows over an 8 h
    # snapshot) saturate the 10 000-row LIMIT and silently evict the
    # rare-and-important structural events that postmortems rely on.
    #
    # Each prefix here matches event_type strings via ``LIKE prefix%``.
    # Rows matching any of these prefixes are returned IN FULL (no
    # cap) by ``bot_events_since`` in addition to the up-to-LIMIT
    # most-recent rows of everything else, then merged + sorted by
    # timestamp before return.
    #
    # The list is conservative: each entry corresponds to a category
    # of events that the operator/dashboard cannot reconstruct from
    # `fills_since.json` or `state_current.json` alone. Adding a new
    # structural type means adding the prefix here AND, if it's a
    # new logger output, making sure the existing event-type column
    # in ``bot_events`` contains the expected string.
    _BOT_EVENTS_STRUCTURAL_PREFIXES: tuple[str, ...] = (
        "bot_start",
        "bot_shutdown",
        "soft_flatten_",            # soft_flatten_started/_completed
        "sf_fatigue_",              # sf_fatigue_tier{1,2,3,4}_armed/_cleared
        "tp_event_",                # take-profit episodes
        "regime_mode_transition",
        "shock_gate_",              # shock_gate_armed/_cleared
        "desync_",                  # desync_detected/_recovered
        "kill",                     # any *_kill event (e.g. sf_fatigue_tier4_kill)
        "gone_on_exchange_",        # tier-1 connectivity events
        "ws_arrived_late",
        "http_acked_no_ws",
        "market_data_recovery_",    # NOT _success which is high-rate, only _started/_failed/_succeeded if distinct types exist
        "stale_data_detected",
        "engine_no_quote_persistent",  # not_quoting watchdog
        "executor_silent_wedge",    # wedge detection
        "shadow_position_divergence",  # BUG-034 trigger events
        "private_ws_queue_overflow", # BUG-035 recovery
    )

    def bot_events_since(
        self,
        since_ts: str,
        limit: int = 10000,
        *,
        until_ts: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Bot events with ``ts >= since_ts``, oldest-first.

        v1.5.275 (BUG-038 fix): returns the union of
        (a) **ALL** rows whose ``event_type`` matches one of the
            ``_BOT_EVENTS_STRUCTURAL_PREFIXES`` (no per-type cap), and
        (b) the **most recent** ``limit`` rows of everything else,

        merged + re-sorted by ts ASC before return. Structural events
        are not subject to the cap so a long session's SF episodes,
        regime transitions, kills, etc. always survive even when a
        high-rate trash event type would otherwise saturate the LIMIT.

        Total returned rows can therefore exceed ``limit`` by the
        count of structural events in the window (typically 10-100;
        worst-case bounded by the structural event types' aggregate
        rate, which is by definition low).
        """
        # Build a single SQL query that does both selections in one
        # round-trip. Structural events use a separate sub-select
        # with no cap; non-structural events use the LIMIT clause
        # against the most-recent rows.
        struct_pattern_clauses = " OR ".join(
            ["event_type LIKE ?"] * len(self._BOT_EVENTS_STRUCTURAL_PREFIXES)
        )
        struct_patterns = [
            f"{prefix}%" for prefix in self._BOT_EVENTS_STRUCTURAL_PREFIXES
        ]

        # Selection 1: structural events, no cap.
        struct_sql = f"SELECT * FROM bot_events WHERE ts >= ? AND ({struct_pattern_clauses})"
        struct_params: list[Any] = [since_ts] + struct_patterns
        if until_ts is not None:
            struct_sql += " AND ts < ?"
            struct_params.append(until_ts)

        # Selection 2: non-structural events, most-recent N.
        rest_sql = f"SELECT * FROM bot_events WHERE ts >= ? AND NOT ({struct_pattern_clauses})"
        rest_params: list[Any] = [since_ts] + struct_patterns
        if until_ts is not None:
            rest_sql += " AND ts < ?"
            rest_params.append(until_ts)
        rest_sql += " ORDER BY ts DESC LIMIT ?"
        rest_params.append(int(limit))

        with self._lock:
            with self.connection() as conn:
                struct_rows = self._rows(conn, struct_sql, tuple(struct_params))
                rest_rows = self._rows(conn, rest_sql, tuple(rest_params))

        # Merge + sort. Use a stable sort keyed on ts (string ISO
        # sorts lexicographically because the format is fixed-width).
        # Dedup by rowid if both queries somehow surfaced the same
        # row (shouldn't happen given the disjoint WHERE clauses, but
        # be defensive — the partition is by event_type prefix so a
        # type bug could in theory cross the boundary).
        seen_keys: set[tuple] = set()
        merged: list[dict[str, Any]] = []
        for row in struct_rows + rest_rows:
            key = (row.get("ts"), row.get("event_type"), row.get("message"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(row)
        merged.sort(key=lambda r: (r.get("ts") or "", r.get("event_type") or ""))
        return merged

    def orders_since(
        self,
        since_ts: str,
        limit: int = 10000,
        *,
        until_ts: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Orders with ``ts_created >= since_ts``, oldest-first.

        Optional ``until_ts`` caps the range (exclusive-upper); optional
        ``symbol`` filters to a single trading symbol (matches the
        ``orders.symbol`` column). ``ts_created`` is the anchor time
        because ``ts_ack``/``ts_closed`` may be null or delayed.
        """
        sql = "SELECT * FROM orders WHERE ts_created >= ?"
        params: list[Any] = [since_ts]
        if until_ts is not None:
            sql += " AND ts_created < ?"
            params.append(until_ts)
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY ts_created ASC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            with self.connection() as conn:
                return self._rows(conn, sql, tuple(params))

    def quote_decisions_since(
        self,
        since_ts: str,
        limit: int = 10000,
        *,
        until_ts: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Quote-decision cycles with ``ts >= since_ts``, oldest-first.

        This table is the highest-volume one (1–5 rows/sec). For
        multi-hour windows pick a tight range and tune ``limit`` — the
        default 10000 rows is ~30 min of quoting at 5 Hz.
        """
        sql = "SELECT * FROM quote_decisions WHERE ts >= ?"
        params: list[Any] = [since_ts]
        if until_ts is not None:
            sql += " AND ts < ?"
            params.append(until_ts)
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY ts ASC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            with self.connection() as conn:
                return self._rows(conn, sql, tuple(params))

    def position_snapshots_since(
        self,
        since_ts: str,
        limit: int = 10000,
        *,
        until_ts: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Position snapshots with ``ts >= since_ts``, oldest-first."""
        sql = "SELECT * FROM position_snapshots WHERE ts >= ?"
        params: list[Any] = [since_ts]
        if until_ts is not None:
            sql += " AND ts < ?"
            params.append(until_ts)
        if symbol is not None:
            sql += " AND symbol = ?"
            params.append(symbol)
        sql += " ORDER BY ts ASC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            with self.connection() as conn:
                return self._rows(conn, sql, tuple(params))

    def recent_orders(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            with self.connection() as conn:
                return self._rows(
                    conn,
                    "SELECT * FROM orders ORDER BY ts_created DESC LIMIT ?",
                    (limit,),
                )

    def recent_fills(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            with self.connection() as conn:
                return self._rows(
                    conn,
                    "SELECT * FROM fills ORDER BY ts_fill DESC LIMIT ?",
                    (limit,),
                )

    def recent_quote_decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            with self.connection() as conn:
                return self._rows(
                    conn,
                    "SELECT * FROM quote_decisions ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )

    def recent_bot_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            with self.connection() as conn:
                return self._rows(
                    conn,
                    "SELECT * FROM bot_events ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )

    def insert_soft_flatten_event(self, row: dict[str, Any]) -> int:
        """Insert a new SF episode row, return its autoincrement id.

        Caller passes the start-time fields (``ts_start``,
        ``trigger_reason``, ``initial_force_phase``,
        ``taker_fallback_ticks``, ``entry_position_qty``,
        ``entry_mid_price``); the end-time fields (``ts_end``,
        ``exit_phase_reached``, ``exit_reason``) get filled in later
        via :meth:`update_soft_flatten_event_end`. Plan reference:
        plans/20260507-sf-frontend.md Phase 2.
        """
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        with self._lock:
            with self.connection() as conn:
                cur = conn.execute(
                    f"INSERT INTO soft_flatten_events ({cols}) "
                    f"VALUES ({placeholders})",
                    tuple(row.values()),
                )
                return int(cur.lastrowid)

    def update_soft_flatten_event_end(
        self, event_id: int, end_row: dict[str, Any]
    ) -> None:
        """Update the SF episode row with end-time fields.

        Idempotent overwrite — calling twice with the same payload
        leaves the row unchanged. Caller is expected to pass at
        minimum ``ts_end``; ``exit_phase_reached`` / ``exit_reason``
        are recommended.
        """
        if not end_row:
            return
        set_clause = ", ".join(f"{k} = ?" for k in end_row.keys())
        params = tuple(end_row.values()) + (int(event_id),)
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    f"UPDATE soft_flatten_events SET {set_clause} "
                    f"WHERE id = ?",
                    params,
                )

    def compute_sf_episode_phase_totals(
        self, event_id: int
    ) -> dict[str, Any]:
        """v1.4.173 (Phase 4D.4) — roll up the fills belonging to an SF
        episode into the per-phase counts + the notional-weighted
        average cross-spread bps paid for the non-passive phases.

        Reads from the ``fills`` table where
        ``soft_flatten_event_id = event_id``. Buckets by
        ``sf_force_phase`` (0..4); NULL phases are bucketed as "0"
        for back-compat with episodes that arose with the legacy
        2-phase post-only worker (since pre-ladder fills sat in the
        same near-touch slot as ladder phase 0). The cross-spread
        weighting reads ``soft_flatten_events.entry_mid_price`` as
        the reference: per-fill cost in bps = (fill_price vs
        entry_mid) signed by the close direction, weighted by
        ``ABS(notional)``.

        Returns a dict with two keys:
          * ``fills_by_phase`` — dict[str, int], e.g.
            ``{"0": 5, "1": 0, "2": 1, "3": 0, "4": 0}``. Always
            contains entries for all five phases (0 if unused).
          * ``taker_spread_bps_paid`` — float | None, average bps
            across all fills with ``sf_force_phase >= 2`` (the
            non-passive phases). None when no taker fills occurred.

        Operator interpretation: bot patient (mostly p0/p1) +
        taker_spread_bps_paid small → ladder is doing its job;
        mostly p4 (market) + spread high → drift / dwell budgets
        too generous, tighten or lower fast-escalate threshold.

        Best-effort: any SQL error returns an empty dict (caller
        treats it as "stats unavailable" and writes NULL to the row,
        which the dashboard renders as the legacy SF marker without
        the phase-mix tooltip).
        """
        out: dict[str, Any] = {}
        try:
            with self._lock:
                with self.connection() as conn:
                    # Per-phase count, treating NULL as phase 0
                    # (legacy 2-phase post-only era; ladder disabled).
                    rows = conn.execute(
                        """
                        SELECT
                          COALESCE(sf_force_phase, 0) AS phase,
                          COUNT(*) AS n
                        FROM fills
                        WHERE soft_flatten_event_id = ?
                        GROUP BY COALESCE(sf_force_phase, 0)
                        """,
                        (int(event_id),),
                    ).fetchall()
                    counts = {str(p): 0 for p in range(5)}
                    for r in rows:
                        ph = int(r[0])
                        if 0 <= ph <= 4:
                            counts[str(ph)] = int(r[1])
                    out["fills_by_phase"] = counts

                    # Notional-weighted average cross-spread cost for
                    # taker phases (2/3/4). Reference price = the
                    # episode's entry_mid_price; per-fill cost = the
                    # adverse direction price delta in bps.
                    #
                    #   long close (SELL): cost_bps = (entry_mid -
                    #       fill_price) / entry_mid * 10000
                    #   short close (BUY): cost_bps = (fill_price -
                    #       entry_mid) / entry_mid * 10000
                    #
                    # We don't know the close side per-fill cheaply
                    # (the bot only closes one direction per episode),
                    # so use ABS of the delta — for a single-side close
                    # this is equivalent. Robust against book noise
                    # because the entry mid is fixed per episode.
                    taker_row = conn.execute(
                        """
                        SELECT
                          e.entry_mid_price AS entry_mid,
                          SUM(
                            ABS(f.notional) *
                            ABS(f.price - e.entry_mid_price) /
                            e.entry_mid_price * 10000.0
                          ) AS bps_notional_product,
                          SUM(ABS(f.notional)) AS total_notional
                        FROM fills f
                        JOIN soft_flatten_events e
                          ON f.soft_flatten_event_id = e.id
                        WHERE f.soft_flatten_event_id = ?
                          AND COALESCE(f.sf_force_phase, 0) >= 2
                          AND e.entry_mid_price IS NOT NULL
                          AND e.entry_mid_price > 0
                        """,
                        (int(event_id),),
                    ).fetchone()
                    if (
                        taker_row is not None
                        and taker_row[2] is not None
                        and float(taker_row[2]) > 0
                    ):
                        out["taker_spread_bps_paid"] = (
                            float(taker_row[1]) / float(taker_row[2])
                        )
                    else:
                        out["taker_spread_bps_paid"] = None
        except Exception:
            logger.exception(
                "compute_sf_episode_phase_totals_failed event_id=%s",
                event_id,
            )
        return out

    def recent_soft_flatten_events(
        self,
        limit: int = 50,
        *,
        since_ts: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Return recent SF episodes with attribution counts joined
        in (orders + fills + sum-of-fill-notional).

        Counts are computed at read time via LEFT JOIN against
        ``orders`` / ``fills`` on ``soft_flatten_event_id``. At
        $1k-account fill volumes this is cheap; if it ever becomes
        a hot path, denormalise the counts onto the events row.

        v1.4.36 (Codex #3 + #7): added optional ``since_ts`` filter.
        SF episodes persist in the DB across sessions by design (they
        are the historical record used by postmortem, equity-history
        overlays, and cross-session analytics). For the live-stats
        publication, however, the operator wants ONLY the current
        session's events — pre-v1.4.36 the publisher labelled the
        block as "events this session" but pulled all-time history,
        causing the count to inherit episodes from prior sessions.
        The caller passes ``session_started_at_utc`` to filter to the
        current session; existing callers (postmortem, equity-history,
        cross-session reports) pass nothing and continue to see the
        full history.
        """
        since_clause = ""
        params: tuple = (limit,)
        if since_ts is not None:
            since_clause = "WHERE e.ts_start >= ?"
            params = (since_ts, limit)
        with self._lock:
            with self.connection() as conn:
                return self._rows(
                    conn,
                    f"""
                    SELECT
                      e.*,
                      (SELECT COUNT(*) FROM orders o
                         WHERE o.soft_flatten_event_id = e.id)
                        AS attributed_orders_count,
                      (SELECT COUNT(*) FROM fills f
                         WHERE f.soft_flatten_event_id = e.id)
                        AS attributed_fills_count,
                      (SELECT COALESCE(SUM(ABS(f.notional)), 0.0)
                         FROM fills f
                         WHERE f.soft_flatten_event_id = e.id)
                        AS attributed_fills_notional_usd
                    FROM soft_flatten_events e
                    {since_clause}
                    ORDER BY e.ts_start DESC
                    LIMIT ?
                    """,
                    params,
                )

    # ------------------------------------------------------------------
    # v1.5.33 — take-profit (TP) attribution helpers.
    # Mirror of the SF storage surface (``insert_soft_flatten_event``,
    # ``update_soft_flatten_event_end``, ``recent_soft_flatten_events``).
    # ------------------------------------------------------------------

    def insert_tp_event(self, row: dict[str, Any]) -> int:
        """Insert a new TP episode row, return its autoincrement id.

        Caller passes start-time fields (``ts_start``, threshold/upnl,
        entry side, entry price, entry position qty); end-time fields
        are filled in later via :meth:`update_tp_event_end`. Schema:
        ``tp_events`` (storage v42).
        """
        cols = ", ".join(row.keys())
        placeholders = ", ".join("?" * len(row))
        with self._lock:
            with self.connection() as conn:
                cur = conn.execute(
                    f"INSERT INTO tp_events ({cols}) "
                    f"VALUES ({placeholders})",
                    tuple(row.values()),
                )
                return int(cur.lastrowid)

    def update_tp_event_end(
        self, event_id: int, end_row: dict[str, Any]
    ) -> None:
        """Update the TP episode row with end-time fields. Idempotent
        overwrite. Caller passes ``ts_end`` at minimum.
        """
        if not end_row:
            return
        set_clause = ", ".join(f"{k} = ?" for k in end_row.keys())
        params = tuple(end_row.values()) + (int(event_id),)
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    f"UPDATE tp_events SET {set_clause} "
                    f"WHERE id = ?",
                    params,
                )

    def recent_tp_events(
        self,
        limit: int = 50,
        *,
        since_ts: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Recent TP episodes with attribution counts joined in.

        Mirrors ``recent_soft_flatten_events``: LEFT JOIN against
        orders + fills on ``tp_event_id``. ``since_ts`` filters by
        ``ts_start`` for the live-stats publication; postmortem
        callers omit it to see the full history.
        """
        since_clause = ""
        params: tuple = (limit,)
        if since_ts is not None:
            since_clause = "WHERE e.ts_start >= ?"
            params = (since_ts, limit)
        with self._lock:
            with self.connection() as conn:
                return self._rows(
                    conn,
                    f"""
                    SELECT
                      e.*,
                      (SELECT COUNT(*) FROM orders o
                         WHERE o.tp_event_id = e.id)
                        AS attributed_orders_count,
                      (SELECT COUNT(*) FROM fills f
                         WHERE f.tp_event_id = e.id)
                        AS attributed_fills_count,
                      (SELECT COALESCE(SUM(ABS(f.notional)), 0.0)
                         FROM fills f
                         WHERE f.tp_event_id = e.id)
                        AS attributed_fills_notional_usd
                    FROM tp_events e
                    {since_clause}
                    ORDER BY e.ts_start DESC
                    LIMIT ?
                    """,
                    params,
                )

    def soft_flatten_event_id_for_order(
        self, order_id_exchange: str
    ) -> Optional[int]:
        """Return the ``soft_flatten_event_id`` stamped on an order
        when it was inserted, or ``None`` if the order is not in the
        DB or was placed outside an SF episode.

        Used by fill ingestion to attribute fills to the same SF
        episode as their parent order — single indexed lookup per
        fill. The orders table is keyed on ``order_id_local``;
        this query uses ``idx_orders_order_id_exchange`` (added in
        storage v20, Codex MED-6 follow-up) and picks the most
        recent row if the venue ever recycles the id (OKX/HL don't
        in practice; defensive).
        """
        with self._lock:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT soft_flatten_event_id FROM orders "
                    "WHERE order_id_exchange = ? "
                    "ORDER BY ts_created DESC LIMIT 1",
                    (str(order_id_exchange),),
                ).fetchone()
                if row is None:
                    return None
                v = row[0]
                return None if v is None else int(v)

    def order_quote_quality_for_order(
        self, order_id_exchange: str
    ) -> tuple[Optional[float], Optional[str]]:
        """Return ``(target_half_spread_bps, quote_aggressiveness)`` for
        an order, or ``(None, None)`` if the order is not in the DB or
        has no quality tag (legacy order or non-MM placement path).

        Single indexed lookup per fill; mirrors
        ``soft_flatten_event_id_for_order``. Used by fill ingestion to
        copy the place-time tags onto the resulting Fill.
        """
        with self._lock:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT target_half_spread_bps, quote_aggressiveness "
                    "FROM orders WHERE order_id_exchange = ? "
                    "ORDER BY ts_created DESC LIMIT 1",
                    (str(order_id_exchange),),
                ).fetchone()
                if row is None:
                    return (None, None)
                t = row[0]
                a = row[1]
                t_f = None if t is None else float(t)
                a_s = None if a is None else str(a)
                return (t_f, a_s)

    # Column names for the decision-state propagation lookup. Defined
    # once at module level so the SELECT in the helper below stays in
    # sync with the Fill model + the migration above. Includes
    # ``ts_cancel_requested`` — added in v26 alongside the rest of
    # Phase 1 (it's specifically the cancel-race diagnostic source).
    _DECISION_STATE_COLS: tuple[str, ...] = (
        "toxicity_score_at_decision",
        "vol_estimate_at_decision",
        "active_sides_at_decision",
        "decision_reason_at_decision",
        "binance_basis_ewma_at_decision",
        "adaptive_widen_active_at_decision",
        "post_fill_cooldown_active_bid_at_decision",
        "post_fill_cooldown_active_ask_at_decision",
        "at_touch_adverse_pause_bid_at_decision",
        "at_touch_adverse_pause_ask_at_decision",
        "quote_distance_to_touch_ticks_at_placement",
        "ts_cancel_requested",
        # 2026-05-13 Phase 4a propagation field.
        "expected_net_edge_bps_at_decision",
        # 2026-05-14 todo-027 Tier 2 propagation fields. Stamped on
        # WorkingOrder at place-time in execution.py; pulled from
        # the orders row via this lookup at fill-ingest time;
        # written onto the Fill row.
        "vol_trend_active_at_decision",
        "post_swing_active_at_decision",
        "session_drawdown_tier_at_decision",
        # v1.4.175 Phase 3F — reservation-alpha shift contributions
        # at decision time. Persisted on the parent order row; copied
        # to fills via the same ``order_metadata_for_fill_ingest``
        # path; consumed by ``tools/postmortem/sections/
        # reservation_alpha_attribution.py``.
        "ob_imbalance_shift_bps_at_decision",
        "trend_drift_shift_bps_at_decision",
        "flow_score_shift_bps_at_decision",
        "basis_deviation_shift_bps_at_decision",
        # v1.5.190 Phase 8A Option C — per-fill AS attribution. Stamped
        # on the WorkingOrder at place-time in execution.py from
        # ``state.last_quote_breakdown.base_half_spread_bps`` (which is
        # the AS-computed value when AS is enabled, else the legacy
        # vol-adaptive base); persisted on the orders row; copied onto
        # each resulting fill via this lookup. STAMP-ONLY — not read
        # back into quote construction.
        "as_base_half_spread_bps_at_decision",
        # v1.5.204 Phase 4A — microprice gate widening (bps) at
        # decision time. Per-side because the gate is asymmetric.
        # Source: ``state.last_quote_breakdown.microprice_{bid,ask}_bps``.
        # Used by the Phase 4A.3 acceptance check + the
        # ``microprice_widen_attribution`` postmortem section.
        "microprice_bid_widen_bps_at_decision",
        "microprice_ask_widen_bps_at_decision",
        # v1.5.306 audit §5 P0 #2 — Active-Quoting-Controller per-fill
        # attribution. Stamped on the WorkingOrder at place-time in
        # execution.py from ``state.active_quoting_controller`` (the PI
        # aggression output + the markout safety-floor flag, captured at
        # the moment the order was placed); persisted on the orders row;
        # copied onto each resulting fill via this lookup. STAMP-ONLY —
        # the controller's effect on quoting already flows through the
        # half-spread floor / inventory gates / skew; these record state.
        "aqc_aggression_level_at_decision",
        "aqc_safety_floor_engaged_at_decision",
    )

    def orders_lifecycle_since(
        self,
        since_iso: str,
        *,
        until_iso: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 50000,
        most_recent: bool = False,
    ) -> list[dict[str, Any]]:
        """Order lifecycle rows with derived fields suitable for the
        ``orders_lifecycle_since.json`` snapshot artifact.

        Derived fields (computed at SELECT time, NOT stored):

        - ``lifetime_ms`` — ``ts_closed - ts_created`` in milliseconds,
          NULL when the order is still open.
        - ``cancel_to_close_ms`` — ``ts_closed - ts_cancel_requested`` in
          milliseconds, captures the venue's cancel-confirm latency.
          NULL when cancel was not requested.
        - ``placement_to_ack_ms`` — ``ts_ack - ts_sent`` in milliseconds,
          captures venue REST submit RTT. NULL when not acked.
        - ``cancel_decision_to_send_ms`` — ``ts_cancel_sent -
          ts_cancel_requested`` in ms (1.4.0 Phase 0.5). Pure bot-side
          latency: decision → wire. NULL on legacy rows + on cancels
          where the HTTP path wasn't reached (e.g. cancel-failed-
          because-filled races).
        - ``cancel_send_to_ack_ms`` — ``ts_cancel_acked - ts_cancel_sent``
          in ms (1.4.0 Phase 0.5). Pure transport latency: wire → HTTP
          ack. Companion to ``placement_to_ack_ms`` for places.
          ``ts_cancel_acked`` is only stamped when the cancel response
          interpreted as success, so this naturally filters out
          order-already-gone races.

        Pure post-processing of existing fields — no schema change.

        ``most_recent=True`` (1.3.99): fetch with ``ORDER BY ts_created
        DESC LIMIT ?`` and reverse client-side, so the caller still
        gets oldest-first rows but they represent the *most recent*
        ``limit`` orders. Used by the dashboard publisher whose chart
        tracks the live tail — with the default ASC + LIMIT path,
        long sessions clip at the OLDEST ``limit`` orders and the new
        Places/min / Cancels/min sub-bands cluster on the left edge
        of the chart even though the bot is still actively quoting.
        Mirrors the same fix on :meth:`equity_history_since` and
        :meth:`exposure_bars_since`.

        2026-05-13 regime-observability Phase 4b.
        """
        sql = (
            "SELECT *, "
            "  CAST((julianday(ts_closed) - julianday(ts_created)) * 86400000 AS INTEGER) "
            "    AS lifetime_ms, "
            "  CAST((julianday(ts_closed) - julianday(ts_cancel_requested)) * 86400000 AS INTEGER) "
            "    AS cancel_to_close_ms, "
            "  CAST((julianday(ts_ack) - julianday(ts_sent)) * 86400000 AS INTEGER) "
            "    AS placement_to_ack_ms, "
            # 1.4.0 cancel-prio Phase 0.5 — cancel-latency decomposition.
            "  CAST((julianday(ts_cancel_sent) - julianday(ts_cancel_requested)) * 86400000 AS INTEGER) "
            "    AS cancel_decision_to_send_ms, "
            "  CAST((julianday(ts_cancel_acked) - julianday(ts_cancel_sent)) * 86400000 AS INTEGER) "
            "    AS cancel_send_to_ack_ms "
            "FROM orders WHERE ts_created >= ?"
        )
        params: list[Any] = [since_iso]
        if until_iso:
            sql += " AND ts_created < ?"
            params.append(until_iso)
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        if most_recent:
            sql += " ORDER BY ts_created DESC LIMIT ?"
        else:
            sql += " ORDER BY ts_created ASC LIMIT ?"
        params.append(int(limit))
        with self._lock:
            with self.connection() as conn:
                rows = self._rows(conn, sql, tuple(params))
        if most_recent:
            rows.reverse()
        return rows

    def order_decision_state_for_order(
        self, order_id_exchange: str
    ) -> dict[str, Any]:
        """Return a dict of decision-state fields stamped on the parent
        order at placement time, plus ``ts_cancel_requested`` for the
        cancel-race diagnostic.

        Keys mirror ``_DECISION_STATE_COLS``. Missing / NULL values are
        returned as ``None``. If the order isn't in the local DB at all
        (legacy / pre-1.2.85 row, or REST-catch-up fill whose parent
        wasn't recorded), every value is ``None`` — fill ingestion
        treats this as "no decision-state context available" and leaves
        the corresponding Fill columns NULL.

        Single indexed lookup per fill; analogous to
        ``order_quote_quality_for_order`` but returning a richer
        payload. Used by ``fill_ingestion.ingest_hl_fill_raw``.

        2026-05-13 regime-observability Phase 1.

        v1.4.37 NOTE: kept for back-compat with non-fill-ingestion
        callers (tests + ad-hoc inspection). Fill ingestion now uses
        the combined :meth:`order_metadata_for_fill_ingest` (Codex
        #5) which fetches all three of soft_flatten / quote_quality /
        decision_state in one SELECT.
        """
        cols = ", ".join(self._DECISION_STATE_COLS)
        empty = {k: None for k in self._DECISION_STATE_COLS}
        with self._lock:
            with self.connection() as conn:
                try:
                    row = conn.execute(
                        f"SELECT {cols} FROM orders "
                        "WHERE order_id_exchange = ? "
                        "ORDER BY ts_created DESC LIMIT 1",
                        (str(order_id_exchange),),
                    ).fetchone()
                except sqlite3.OperationalError:
                    # Pre-v26 schema. Should not happen post-migration
                    # but defends against direct manual DB edits.
                    return empty
                if row is None:
                    return empty
                out: dict[str, Any] = {}
                for key, val in zip(self._DECISION_STATE_COLS, row):
                    out[key] = val
                return out

    def parent_order_exists(self, order_id_exchange: str) -> bool:
        """v1.4.163 — return True iff a row in ``orders`` carries this
        ``order_id_exchange``.

        Used by ``fill_ingestion`` to disambiguate "parent absent
        entirely" (returns False) from "parent present but all
        decision-state fields are NULL" (returns True) — the two
        cases are otherwise indistinguishable through
        ``order_metadata_for_fill_ingest`` (which returns an all-None
        dict for both).

        The distinction matters for SF tagging: the **taker-fallback
        path** at ``app/bot.py:2369`` calls ``client.market_close()``
        directly, which bypasses ``ExecutionEngine.place_order`` and
        never inserts an orders row — so any resulting fills end up
        with no parent in the local DB. Fill ingestion then needs a
        fallback ("stamp with active SF id when no parent exists AND
        SF is active") that depends on telling absent-vs-untagged
        apart.

        Cheap — single indexed lookup. Called only on the rare cold
        path (~3 fills per SF episode, none in normal operation).
        """
        with self._lock:
            with self.connection() as conn:
                row = conn.execute(
                    "SELECT 1 FROM orders "
                    "WHERE order_id_exchange = ? LIMIT 1",
                    (str(order_id_exchange),),
                ).fetchone()
                return row is not None

    def order_metadata_for_fill_ingest(
        self, order_id_exchange: str
    ) -> dict[str, Any]:
        """v1.4.37 (Codex #5) — single-query combined fetch for fill
        ingestion.

        Pre-v1.4.37 ``fill_ingestion.ingest_hl_fill_raw`` made THREE
        separate indexed lookups against the same ``orders`` row:

        - :meth:`soft_flatten_event_id_for_order`
        - :meth:`order_quote_quality_for_order`
        - :meth:`order_decision_state_for_order`

        Each acquired ``self._lock`` and opened a fresh connection.
        On bursty fill windows the three serialised round-trips
        showed up as 60-100 µs of avoidable latency per fill, plus
        lock contention against the heartbeat / live-stats readers.

        This combined method does all three in ONE SELECT under one
        lock acquisition + connection. Returns a flat dict with keys:

        - ``soft_flatten_event_id`` (Optional[int])
        - ``target_half_spread_bps`` (Optional[float])
        - ``quote_aggressiveness`` (Optional[str])
        - every key in ``_DECISION_STATE_COLS`` (None when NULL)

        Missing rows return a dict with all values None — the caller
        treats this as "no parent order recorded locally" exactly as
        the three legacy methods did. The legacy methods are
        preserved for non-ingestion callers (tests / ad-hoc).
        """
        decision_cols = list(self._DECISION_STATE_COLS)
        all_cols = [
            "soft_flatten_event_id",
            # v1.5.33 — TP attribution mirror of SF. Stamped on the
            # parent order row at place-time when TP is active;
            # propagated to fill via this single-SELECT lookup.
            "tp_event_id",
            "target_half_spread_bps",
            "quote_aggressiveness",
            # v1.4.100 ladder-observability F1 — pull the parent
            # order's rung index so fill ingestion can stamp it on
            # the fill row.
            "level_idx",
        ] + decision_cols
        empty: dict[str, Any] = {k: None for k in all_cols}
        cols_sql = ", ".join(all_cols)
        with self._lock:
            with self.connection() as conn:
                try:
                    row = conn.execute(
                        f"SELECT {cols_sql} FROM orders "
                        "WHERE order_id_exchange = ? "
                        "ORDER BY ts_created DESC LIMIT 1",
                        (str(order_id_exchange),),
                    ).fetchone()
                except sqlite3.OperationalError:
                    # Pre-v26 schema (one of the decision-state cols
                    # missing). Defensive — bot startup runs the
                    # migration. Same behaviour as the legacy methods.
                    return empty
                if row is None:
                    return empty
                out: dict[str, Any] = {}
                for key, val in zip(all_cols, row):
                    out[key] = val
                # Type-normalise the two columns the legacy
                # ``order_quote_quality_for_order`` did explicit casts
                # on, to keep the contract identical for the caller.
                if out["target_half_spread_bps"] is not None:
                    try:
                        out["target_half_spread_bps"] = float(
                            out["target_half_spread_bps"]
                        )
                    except (TypeError, ValueError):
                        out["target_half_spread_bps"] = None
                if out["quote_aggressiveness"] is not None:
                    out["quote_aggressiveness"] = str(
                        out["quote_aggressiveness"]
                    )
                if out["soft_flatten_event_id"] is not None:
                    try:
                        out["soft_flatten_event_id"] = int(
                            out["soft_flatten_event_id"]
                        )
                    except (TypeError, ValueError):
                        out["soft_flatten_event_id"] = None
                if out.get("tp_event_id") is not None:
                    try:
                        out["tp_event_id"] = int(out["tp_event_id"])
                    except (TypeError, ValueError):
                        out["tp_event_id"] = None
                return out

    def equity_history(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            with self.connection() as conn:
                return self._rows(
                    conn,
                    "SELECT * FROM equity_snapshots ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )

    def insert_market_data_gap_sample(
        self,
        *,
        session_id: str,
        ts_utc: str,
        gap_ms: float,
        symbol: str,
        source: str,
        max_rows_per_session: int,
    ) -> None:
        """Optional bounded persistence for gap samples (current session only by pruning)."""
        with self._lock:
            with self.connection() as conn:
                conn.execute(
                    """
                    INSERT INTO market_data_gap_samples (session_id, ts_utc, gap_ms, symbol, source)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, ts_utc, gap_ms, symbol, source),
                )
                row = conn.execute(
                    "SELECT COUNT(*) FROM market_data_gap_samples WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                n = int(row[0]) if row and row[0] is not None else 0
                if n > max_rows_per_session:
                    to_drop = n - max_rows_per_session
                    conn.execute(
                        """
                        DELETE FROM market_data_gap_samples WHERE rowid IN (
                            SELECT rowid FROM market_data_gap_samples
                            WHERE session_id = ?
                            ORDER BY rowid ASC
                            LIMIT ?
                        )
                        """,
                        (session_id, to_drop),
                    )

    def position_history(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self._lock:
            with self.connection() as conn:
                return self._rows(
                    conn,
                    "SELECT * FROM position_snapshots ORDER BY ts DESC LIMIT ?",
                    (limit,),
                )

    def export_table_csv(self, table: str, out_path: Path) -> int:
        import csv

        allowed = {
            "orders",
            "fills",
            "position_snapshots",
            "equity_snapshots",
            "quote_decisions",
            "bot_events",
        }
        if table not in allowed:
            raise ValueError(f"Unknown table {table}")
        with self._lock:
            with self.connection() as conn:
                cur = conn.execute(f"SELECT * FROM {table}")
                rows = cur.fetchall()
                if not rows:
                    out_path.write_text("", encoding="utf-8")
                    return 0
                headers = [d[0] for d in cur.description]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with out_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(headers)
                    for r in rows:
                        w.writerow(list(r))
                return len(rows)
