"""Phase F3 (v1.5.43) — tests for ``scripts/colo_bot_db_extract.py``.

The extract script is the colo-side worker invoked by
``backtesting/ops/colo.py pull``. It opens the bot's SQLite DB
read-only, runs scoped SELECT queries against each forensic table,
and writes gzipped JSONL.

Test strategy: build a synthetic sqlite DB with the bot's actual
schema columns + fixture rows spanning a known time window. Run
the extract; verify per-table file presence, line counts, JSON
shape, and the window scoping (only rows in [since, until] land).

The DB-touching path is the same code path the production bot
uses, so this test doubles as a regression check on
``Storage.init_schema`` round-tripping with the extract reader.
"""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACT_SCRIPT = REPO_ROOT / "scripts" / "colo_bot_db_extract.py"


def _build_test_db(tmp_path: Path) -> Path:
    """Build a synthetic bot DB with the same table schemas the
    extract script reads. Mirrors the production schema enough for
    the extract logic to round-trip; doesn't have to be byte-
    identical (the extract uses SELECT * which adapts to columns
    present)."""
    import sqlite3

    db_path = tmp_path / "test_mm.db"
    conn = sqlite3.connect(db_path)
    # Minimal schemas matching production columns used by extract.
    conn.executescript(
        """
        CREATE TABLE orders (
            order_id_local TEXT PRIMARY KEY,
            order_id_exchange TEXT,
            ts_created TEXT,
            side TEXT,
            price REAL,
            size REAL,
            status TEXT,
            soft_flatten_event_id INTEGER,
            tp_event_id INTEGER
        );
        CREATE TABLE fills (
            fill_id TEXT PRIMARY KEY,
            ts_fill TEXT,
            side TEXT,
            price REAL,
            size REAL,
            notional REAL,
            fee REAL,
            soft_flatten_event_id INTEGER,
            tp_event_id INTEGER,
            markout_5s_bps REAL
        );
        CREATE TABLE quote_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            quote_cycle_id TEXT,
            eligibility TEXT
        );
        CREATE TABLE bot_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            severity TEXT,
            event_type TEXT,
            message TEXT,
            payload_json TEXT
        );
        CREATE TABLE position_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            position_qty REAL,
            position_notional REAL
        );
        CREATE TABLE equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            equity_usd REAL,
            realized_pnl_usd REAL,
            unrealized_pnl_usd REAL
        );
        CREATE TABLE soft_flatten_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start TEXT,
            ts_end TEXT,
            trigger_reason TEXT,
            exit_reason TEXT
        );
        CREATE TABLE tp_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start TEXT,
            ts_end TEXT,
            trigger_upnl_bps REAL,
            exit_reason TEXT
        );
        CREATE TABLE exposure_bars (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_bar TEXT,
            ts_bucket_end TEXT,
            position_qty REAL
        );
        """
    )

    # Insert rows across 3 windows:
    #   T0 = 2026-05-23T11:00:00 — BEFORE session start
    #   T1 = 2026-05-23T12:30:00 — IN window
    #   T2 = 2026-05-23T13:30:00 — IN window
    #   T3 = 2026-05-23T16:00:00 — AFTER session end
    # Extract for window [12:00, 14:00] should pull T1 + T2 only.
    rows_in = [("2026-05-23T12:30:00+00:00",),
               ("2026-05-23T13:30:00+00:00",)]
    rows_out = [("2026-05-23T11:00:00+00:00",),
                ("2026-05-23T16:00:00+00:00",)]

    # orders
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO orders(order_id_local, order_id_exchange, ts_created, "
            "side, price, size, status, soft_flatten_event_id, tp_event_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"oid-{i}", f"{1000000+i}", ts, "BUY", 1.95, 3.0,
             "ACKED", None, None),
        )
    # fills
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO fills(fill_id, ts_fill, side, price, size, notional, "
            "fee, soft_flatten_event_id, tp_event_id, markout_5s_bps) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"fid-{i}", ts, "BUY", 1.95, 3.0, 5.85, -0.001,
             None, None, -2.5),
        )
    # quote_decisions
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO quote_decisions(ts, quote_cycle_id, eligibility) "
            "VALUES (?, ?, ?)",
            (ts, f"qc-{i}", "QUOTE_BOTH"),
        )
    # bot_events
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO bot_events(ts, severity, event_type, message) "
            "VALUES (?, ?, ?, ?)",
            (ts, "INFO", "test_event", f"msg-{i}"),
        )
    # position_snapshots
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO position_snapshots(ts, position_qty, position_notional) "
            "VALUES (?, ?, ?)",
            (ts, 3.0, 5.85),
        )
    # equity_snapshots
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO equity_snapshots(ts, equity_usd, realized_pnl_usd, "
            "unrealized_pnl_usd) VALUES (?, ?, ?, ?)",
            (ts, 1000.0, 0.05, -0.02),
        )
    # soft_flatten_events (uses ts_start)
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO soft_flatten_events(ts_start, ts_end, trigger_reason, "
            "exit_reason) VALUES (?, ?, ?, ?)",
            (ts, ts, "position_drawdown_gate", "position_closed"),
        )
    # tp_events (uses ts_start)
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO tp_events(ts_start, ts_end, trigger_upnl_bps, "
            "exit_reason) VALUES (?, ?, ?, ?)",
            (ts, ts, 120.0, "position_flat"),
        )
    # exposure_bars (uses ts_bar — must match app/storage.py schema, NOT
    # the long-dead ts_bucket_start name the extractor used to query)
    for i, (ts,) in enumerate(rows_in + rows_out):
        conn.execute(
            "INSERT INTO exposure_bars(ts_bar, ts_bucket_end, "
            "position_qty) VALUES (?, ?, ?)",
            (ts, ts, 3.0),
        )
    conn.commit()
    conn.close()
    return db_path


def _run_extract(
    db: Path,
    out_dir: Path,
    since: str,
    until: str,
    *,
    skip_decisions: bool = False,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--db", str(db),
        "--out-dir", str(out_dir),
        "--since", since,
        "--until", until,
    ]
    if skip_decisions:
        cmd.append("--skip-decisions")
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _read_gzip_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as gz:
        return [json.loads(line) for line in gz if line.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_extract_produces_all_nine_tables(tmp_path):
    db = _build_test_db(tmp_path)
    out_dir = tmp_path / "bot_db"
    r = _run_extract(
        db, out_dir,
        since="2026-05-23T12:00:00+00:00",
        until="2026-05-23T14:00:00+00:00",
    )
    assert r.returncode == 0, f"extract failed: {r.stderr}"
    expected_files = {
        "orders.jsonl.gz",
        "fills.jsonl.gz",
        "quote_decisions.jsonl.gz",
        "bot_events.jsonl.gz",
        "position_snapshots.jsonl.gz",
        "equity_snapshots.jsonl.gz",
        "soft_flatten_events.jsonl.gz",
        "tp_events.jsonl.gz",
        "exposure_bars.jsonl.gz",
    }
    landed = {p.name for p in out_dir.glob("*.jsonl.gz")}
    assert landed == expected_files, f"missing: {expected_files - landed}, extra: {landed - expected_files}"


def test_extract_scopes_to_session_window(tmp_path):
    """Rows outside [since, until] are excluded; rows inside are
    included. 4 rows total (2 in, 2 out); extract should return 2."""
    db = _build_test_db(tmp_path)
    out_dir = tmp_path / "bot_db"
    _run_extract(
        db, out_dir,
        since="2026-05-23T12:00:00+00:00",
        until="2026-05-23T14:00:00+00:00",
    )
    for table_file in (
        "orders.jsonl.gz",
        "fills.jsonl.gz",
        "quote_decisions.jsonl.gz",
        "bot_events.jsonl.gz",
        "position_snapshots.jsonl.gz",
        "equity_snapshots.jsonl.gz",
        "soft_flatten_events.jsonl.gz",
        "tp_events.jsonl.gz",
        "exposure_bars.jsonl.gz",
    ):
        rows = _read_gzip_jsonl(out_dir / table_file)
        assert len(rows) == 2, (
            f"{table_file}: expected 2 in-window rows, got {len(rows)}"
        )


def test_extract_skip_decisions_omits_quote_decisions(tmp_path):
    db = _build_test_db(tmp_path)
    out_dir = tmp_path / "bot_db"
    _run_extract(
        db, out_dir,
        since="2026-05-23T12:00:00+00:00",
        until="2026-05-23T14:00:00+00:00",
        skip_decisions=True,
    )
    assert not (out_dir / "quote_decisions.jsonl.gz").exists()
    # Other tables still landed.
    assert (out_dir / "orders.jsonl.gz").exists()
    assert (out_dir / "fills.jsonl.gz").exists()


def test_extract_rows_are_valid_json_with_expected_fields(tmp_path):
    db = _build_test_db(tmp_path)
    out_dir = tmp_path / "bot_db"
    _run_extract(
        db, out_dir,
        since="2026-05-23T12:00:00+00:00",
        until="2026-05-23T14:00:00+00:00",
    )
    fills = _read_gzip_jsonl(out_dir / "fills.jsonl.gz")
    assert len(fills) == 2
    assert all("fill_id" in f for f in fills)
    assert all("ts_fill" in f for f in fills)
    assert all("markout_5s_bps" in f for f in fills)
    assert all("tp_event_id" in f for f in fills)
    # Rows sorted ascending by ts_fill.
    assert fills[0]["ts_fill"] <= fills[1]["ts_fill"]


def test_extract_handles_missing_table(tmp_path):
    """Pre-v42 DB without tp_events → script logs WARN + emits empty
    file (or no file) for the missing table; OTHER tables still
    extract normally."""
    import sqlite3
    db_path = tmp_path / "pre_v42.db"
    conn = sqlite3.connect(db_path)
    # Only create the orders table — others missing.
    conn.execute(
        "CREATE TABLE orders (order_id_local TEXT, ts_created TEXT, "
        "side TEXT, price REAL)"
    )
    conn.execute(
        "INSERT INTO orders VALUES (?, ?, ?, ?)",
        ("oid-1", "2026-05-23T12:30:00+00:00", "BUY", 1.95),
    )
    conn.commit()
    conn.close()

    out_dir = tmp_path / "bot_db"
    r = _run_extract(
        db_path, out_dir,
        since="2026-05-23T12:00:00+00:00",
        until="2026-05-23T14:00:00+00:00",
    )
    assert r.returncode == 0, f"extract failed: {r.stderr}"
    # orders.jsonl.gz exists with 1 row.
    assert (out_dir / "orders.jsonl.gz").exists()
    rows = _read_gzip_jsonl(out_dir / "orders.jsonl.gz")
    assert len(rows) == 1
    # Missing tables produce no file (or empty) — at minimum, the
    # extract didn't crash.
    fills_path = out_dir / "fills.jsonl.gz"
    # Either absent or empty — both acceptable.
    if fills_path.exists():
        assert _read_gzip_jsonl(fills_path) == []


def test_extract_fails_cleanly_on_missing_db(tmp_path):
    r = _run_extract(
        tmp_path / "nope.db",
        tmp_path / "bot_db",
        since="2026-05-23T12:00:00+00:00",
        until="2026-05-23T14:00:00+00:00",
    )
    assert r.returncode != 0
    assert "db not found" in r.stderr.lower()


def test_extract_immutable_mode_safe_against_concurrent_writes(tmp_path):
    """Sanity check: with the DB opened immutable=1, the extract
    completes even if another process is also connected to the DB.
    Doesn't prove no-corruption under heavy load — just that the
    URI parsing works and basic locking is permissive."""
    import sqlite3

    db = _build_test_db(tmp_path)
    # Open another connection in normal (writable) mode.
    other = sqlite3.connect(db)
    other.execute("PRAGMA journal_mode=WAL")
    try:
        out_dir = tmp_path / "bot_db"
        r = _run_extract(
            db, out_dir,
            since="2026-05-23T12:00:00+00:00",
            until="2026-05-23T14:00:00+00:00",
        )
        assert r.returncode == 0, f"extract failed: {r.stderr}"
    finally:
        other.close()


def test_table_specs_cols_match_real_storage_schema(tmp_path):
    """Regression guard for the v1.5.300 exposure_bars extract bug.

    Each ``TABLE_SPECS`` entry pairs a forensic table with the
    timestamp column the extractor scopes on
    (``WHERE <ts_col> BETWEEN ?``). If that name drifts from the
    column ``app/storage.py`` actually creates, the scoped SELECT
    raises ``OperationalError: no such column``, ``extract_table``
    swallows it, and the table silently exports ZERO rows — in every
    backtest variant AND every live ``snapshot_on_colo.sh --with-db``
    pull.

    This bit ``exposure_bars``: the bot writes ``ts_bar`` but the
    extractor queried ``ts_bucket_start`` (a column that never
    existed). The OTHER tests in this file missed it because they
    hand-build the fixture DB with the SAME column names the
    extractor queries — validating the extractor against itself, not
    against the production schema. This test builds the schema the
    bot ACTUALLY creates (``Storage.init_schema``) and asserts every
    ts-column is real, so the next drift fails loudly here instead of
    silently zeroing a table in production."""
    import importlib.util
    import sqlite3

    from app.storage import Storage
    from tests.settings_helpers import UnitTestSettings

    db_path = tmp_path / "real_schema.db"
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH",
            "DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
            "PRIVATE_WS_ENABLED": False,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    storage.close()

    # Import TABLE_SPECS from the extract script (guarded main → import
    # is side-effect-free; argparse only runs under __main__).
    spec = importlib.util.spec_from_file_location(
        "colo_bot_db_extract", EXTRACT_SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    conn = sqlite3.connect(db_path)
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        problems = []
        for table_name, ts_column in mod.TABLE_SPECS:
            if table_name not in tables:
                problems.append(f"{table_name}: table absent from schema")
                continue
            cols = {
                r[1] for r in conn.execute(f"PRAGMA table_info({table_name})")
            }
            if ts_column not in cols:
                problems.append(
                    f"{table_name}: ts-column {ts_column!r} not in {sorted(cols)}"
                )
        assert not problems, (
            "TABLE_SPECS drifted from the real app/storage.py schema; "
            "scoped SELECT would silently export 0 rows:\n  "
            + "\n  ".join(problems)
        )
    finally:
        conn.close()
