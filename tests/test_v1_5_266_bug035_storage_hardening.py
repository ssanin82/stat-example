"""v1.5.266 / BUG-035 — storage corruption mitigation.

Snapshot `v1.5.251-260529-160310-prod.okx.ton.usdt.perp` and
`v1.5.257-260529-173313-prod.okx.ton.usdt.perp` both reported the
bot's SQLite DB as corrupt (`PRAGMA quick_check` raised
``SQLITE_CORRUPT``). The corruption was introduced between
v1.5.248 (last clean) and v1.5.251 — coinciding with a cluster of
rapid `systemctl restart dtc-bot` deploys (v1.5.249, .250, .251).

Storage was configured with `PRAGMA journal_mode=WAL` but no
explicit `PRAGMA synchronous`, defaulting to NORMAL in WAL mode.
NORMAL survives power loss but can leave the WAL in a corruptible
state if a SIGTERM lands during checkpoint and is escalated to
SIGKILL before the checkpoint finishes.

These tests verify the three v1.5.266 mitigations:
  1. `PRAGMA synchronous=FULL` is applied at init_schema()
  2. `PRAGMA wal_autocheckpoint=200` is applied at init_schema()
  3. `PRAGMA wal_checkpoint(TRUNCATE)` runs in close() so the WAL
     file is fully merged before process exit.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest


def _make_storage(tmp_path: Path):
    """Create a Storage instance pointed at an isolated temp DB file."""
    # Storage's pytest-detection (PYTEST_CURRENT_TEST env) makes it use
    # per-call connection lifecycle so the temp file unlinks cleanly.
    # For the close() test we want the persistent-connection path so
    # we can verify the TRUNCATE actually runs.
    from app.config import Settings
    from app.storage import Storage

    db_path = tmp_path / "test.db"
    settings = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
        SQLITE_PATH=str(db_path),
    )
    storage = Storage(settings)
    storage.init_schema()
    return storage, db_path


def test_init_schema_applies_synchronous_full(tmp_path):
    """PRAGMA synchronous is set to FULL (2) after init_schema."""
    storage, db_path = _make_storage(tmp_path)
    try:
        with storage.connection() as conn:
            cur = conn.execute("PRAGMA synchronous")
            (sync,) = cur.fetchone()
        # SQLite returns synchronous as integer: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA.
        assert sync == 2, f"expected synchronous=FULL (2), got {sync}"
    finally:
        storage.close()


def test_init_schema_applies_wal_autocheckpoint_200(tmp_path):
    """wal_autocheckpoint=200 (vs default 1000) keeps WAL small."""
    storage, db_path = _make_storage(tmp_path)
    try:
        with storage.connection() as conn:
            cur = conn.execute("PRAGMA wal_autocheckpoint")
            (n,) = cur.fetchone()
        assert n == 200, f"expected wal_autocheckpoint=200, got {n}"
    finally:
        storage.close()


def test_init_schema_uses_wal_mode(tmp_path):
    """Regression guard — WAL mode itself must stay on."""
    storage, db_path = _make_storage(tmp_path)
    try:
        with storage.connection() as conn:
            cur = conn.execute("PRAGMA journal_mode")
            (mode,) = cur.fetchone()
        assert str(mode).lower() == "wal", f"expected wal mode, got {mode}"
    finally:
        storage.close()


def test_close_invokes_wal_checkpoint_truncate(tmp_path, monkeypatch):
    """close() must explicitly invoke PRAGMA wal_checkpoint(TRUNCATE)
    so that any un-merged WAL frames are flushed and the file
    is truncated to zero before the underlying connection closes.

    This is the operational guarantee — at process exit there should
    be no un-merged WAL data that the next bot process has to replay
    (replay during a contested race is the v1.5.266 corruption mode).

    We verify by intercepting the executed SQL on the persistent
    connection rather than inspecting -wal file size (which is
    flaky across platforms and depends on SQLite internals — e.g.
    Windows file-handle semantics and synchronous=FULL can cause
    the -wal to be absent or pre-truncated even before close()).
    """
    # Force the persistent-connection path.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    storage, db_path = _make_storage(tmp_path)

    # Wrap the persistent connection's execute() to log SQL calls
    # without breaking the underlying functionality.
    executed_sql: list[str] = []
    real_conn = storage._conn
    assert real_conn is not None
    real_execute = real_conn.execute

    class _TracingConn:
        """Forwards everything to real_conn but logs execute() SQL."""
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            executed_sql.append(sql)
            return real_execute(sql, *args, **kwargs)

        def close(self):
            return self._inner.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    storage._conn = _TracingConn(real_conn)  # type: ignore[assignment]

    storage.close()

    # The close() contract is: try wal_checkpoint(TRUNCATE), then
    # close the connection.
    matched = [s for s in executed_sql
               if "wal_checkpoint" in s.lower() and "truncate" in s.lower()]
    assert matched, (
        f"close() must invoke PRAGMA wal_checkpoint(TRUNCATE); "
        f"executed SQLs were: {executed_sql}"
    )
    # And the persistent connection slot is cleared.
    assert storage._conn is None


def test_close_is_safe_when_checkpoint_fails(tmp_path):
    """If the TRUNCATE checkpoint fails (e.g. concurrent reader, or
    the connection is already in a bad state), close() must STILL
    close the underlying connection — never leak a file handle. The
    bug-035 fix uses try/except for this reason.

    Strategy: replace the persistent connection with a stub whose
    execute() raises but whose close() succeeds. The Storage.close()
    contract is: try the checkpoint, swallow the exception, then
    call conn.close(). The stub records that close() was called.
    """
    prev = os.environ.pop("PYTEST_CURRENT_TEST", None)
    try:
        storage, db_path = _make_storage(tmp_path)

        class _StubConn:
            def __init__(self):
                self.execute_calls = []
                self.close_called = False

            def execute(self, sql, *args, **kwargs):
                self.execute_calls.append(sql)
                if "wal_checkpoint" in sql.lower():
                    raise sqlite3.OperationalError(
                        "simulated checkpoint failure",
                    )
                return None

            def close(self):
                self.close_called = True

        stub = _StubConn()
        # Close the real connection first to avoid leaking the file
        # handle, then swap in the stub for the close() contract test.
        if storage._conn is not None:
            storage._conn.close()
        storage._conn = stub  # type: ignore[assignment]

        # close() should swallow the checkpoint failure and proceed.
        storage.close()  # must not raise

        # The stub's checkpoint call was attempted (and raised), and
        # close() was still called on the underlying connection.
        assert any("wal_checkpoint" in s.lower() for s in stub.execute_calls), (
            "expected close() to attempt PRAGMA wal_checkpoint(TRUNCATE)"
        )
        assert stub.close_called, (
            "close() must call the underlying connection's close() even when "
            "the preceding checkpoint raises (otherwise the file handle leaks)"
        )
        # State after close: persistent connection cleared.
        assert storage._conn is None
    finally:
        if prev is not None:
            os.environ["PYTEST_CURRENT_TEST"] = prev
