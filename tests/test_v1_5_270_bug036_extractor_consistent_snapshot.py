"""v1.5.270 / BUG-036 — extractor uses VACUUM INTO for a consistent
point-in-time snapshot instead of reading the live DB with immutable=1.

Symptom: snapshot v1.5.269-260529-202009 reported
``[FAIL] bot_db SQLite integrity — corrupt: PRAGMA raised: database
disk image is malformed`` despite every per-table dump succeeding
(NO ``.error`` files). That pattern is impossible for genuine
corruption — bulk reads would have hit the same bad pages.

Root cause: ``scripts/colo_bot_db_extract.py::open_db_readonly`` opened
the live DB with ``file:{path}?mode=ro&immutable=1``. The
``immutable=1`` URI flag tells SQLite the file CANNOT change and is
appropriate only for read-only media. Per the SQLite docs:

  "Setting the immutable property on a database file that does in
   fact change can result in incorrect query results and/or
   SQLITE_CORRUPT errors."

The bot writes to the DB continuously; the extractor's ``immutable=1``
view raced against the WAL checkpoint and reported phantom corruption.

Fix: ``vacuum_into_consistent_copy`` takes a real point-in-time
copy via ``VACUUM INTO`` before extraction. The bot keeps writing
throughout; the copy is a clean compacted DB the extractor can
read at leisure. The temp file is cleaned up after extraction.

These tests verify:
1. The helper successfully copies an in-progress WAL-mode DB
2. ``open_db_readonly`` opens the consistent copy with ``mode=ro``
   (no ``immutable=1``) so future bugs of this shape are visible
3. ``PRAGMA quick_check`` against the copy returns ``ok``, proving
   the consistent-copy mechanism doesn't reproduce the false
   positive
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest


def _make_busy_wal_db(db_path: Path, rows_pre: int = 500) -> None:
    """Create a WAL-mode DB with some content and an open writer
    handle, simulating a live bot mid-session."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fills ("
        "fill_id TEXT PRIMARY KEY, ts TEXT, sz REAL, px REAL)"
    )
    for i in range(rows_pre):
        conn.execute(
            "INSERT INTO fills VALUES (?, ?, ?, ?)",
            (f"f_{i:06d}", f"2026-05-29T20:{i//60:02d}:{i%60:02d}Z",
             3.0, 1.73 + i * 0.0001),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# vacuum_into_consistent_copy
# ---------------------------------------------------------------------------

def test_vacuum_into_creates_clean_consistent_copy(tmp_path):
    """The copy is a valid SQLite DB with the same content."""
    from scripts.colo_bot_db_extract import vacuum_into_consistent_copy

    src = tmp_path / "mm.db"
    dst = tmp_path / "copy.db"
    _make_busy_wal_db(src, rows_pre=200)

    vacuum_into_consistent_copy(src, dst)

    assert dst.is_file()
    conn = sqlite3.connect(str(dst))
    try:
        n = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        assert n == 200
        # PRAGMA quick_check on the copy must return ok.
        rows = conn.execute("PRAGMA quick_check").fetchall()
        assert len(rows) == 1
        assert (rows[0][0] or "").strip().lower() == "ok"
    finally:
        conn.close()


def test_vacuum_into_works_while_writer_is_active(tmp_path):
    """The whole point: VACUUM INTO succeeds even while another
    process is writing to the source DB. Reproduces the live-bot
    scenario the immutable=1 path was failing on."""
    from scripts.colo_bot_db_extract import vacuum_into_consistent_copy

    src = tmp_path / "mm.db"
    dst = tmp_path / "copy.db"
    _make_busy_wal_db(src, rows_pre=100)

    stop = threading.Event()
    writer_errors: list[Exception] = []

    def writer():
        """Background writer simulating the live bot."""
        try:
            c = sqlite3.connect(str(src))
            c.execute("PRAGMA journal_mode=WAL")
            i = 1000
            while not stop.is_set():
                try:
                    c.execute(
                        "INSERT INTO fills VALUES (?, ?, ?, ?)",
                        (f"w_{i:06d}", f"2026-05-29T21:00:{i%60:02d}Z",
                         3.0, 1.73),
                    )
                    c.commit()
                    i += 1
                    time.sleep(0.005)  # ~200 writes/sec
                except sqlite3.Error:
                    # Brief lock-contention errors are acceptable;
                    # the writer just keeps trying.
                    pass
            c.close()
        except Exception as e:
            writer_errors.append(e)

    t = threading.Thread(target=writer)
    t.start()
    try:
        # Let the writer do its thing for a moment so VACUUM INTO
        # has to actually contend.
        time.sleep(0.1)
        vacuum_into_consistent_copy(src, dst)
    finally:
        stop.set()
        t.join(timeout=2.0)

    assert not writer_errors, (
        f"writer crashed during VACUUM INTO: {writer_errors}"
    )
    # The copy should be a clean, complete DB.
    conn = sqlite3.connect(str(dst))
    try:
        n = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        assert n >= 100  # at least the pre-existing rows
        rows = conn.execute("PRAGMA quick_check").fetchall()
        assert (rows[0][0] or "").strip().lower() == "ok", (
            f"VACUUM INTO produced a corrupt copy: {rows}"
        )
    finally:
        conn.close()


def test_vacuum_into_overwrites_existing_destination(tmp_path):
    """If a stale temp file exists from a prior aborted run, the
    helper unlinks it before VACUUM INTO."""
    from scripts.colo_bot_db_extract import vacuum_into_consistent_copy

    src = tmp_path / "mm.db"
    dst = tmp_path / "copy.db"
    _make_busy_wal_db(src, rows_pre=50)

    # Pre-existing junk at destination — would normally make
    # VACUUM INTO fail with "output file already exists".
    dst.write_bytes(b"junk-from-prior-run")
    assert dst.is_file()

    vacuum_into_consistent_copy(src, dst)

    # New DB is in place and readable.
    conn = sqlite3.connect(str(dst))
    try:
        n = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        assert n == 50
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# open_db_readonly
# ---------------------------------------------------------------------------

def test_open_db_readonly_uses_mode_ro_not_immutable(tmp_path, monkeypatch):
    """Regression guard: the read-only opener must NOT pass
    ``immutable=1`` to SQLite (the v1.5.270 bug's root cause).
    Verified by intercepting ``sqlite3.connect`` and inspecting
    the URI string the opener actually constructs."""
    import scripts.colo_bot_db_extract as mod

    src = tmp_path / "any.db"
    _make_busy_wal_db(src, rows_pre=5)

    captured_uris: list[str] = []
    real_connect = sqlite3.connect

    def fake_connect(target, *args, **kwargs):
        if kwargs.get("uri"):
            captured_uris.append(target)
        return real_connect(target, *args, **kwargs)

    monkeypatch.setattr(mod.sqlite3, "connect", fake_connect)

    conn = mod.open_db_readonly(src)
    conn.close()

    assert captured_uris, (
        "open_db_readonly should connect via a URI string (uri=True)"
    )
    uri = captured_uris[0]
    assert "immutable=1" not in uri, (
        f"open_db_readonly URI must NOT contain immutable=1 — that "
        f"was the v1.5.270 BUG-036 root cause (SQLite docs explicitly "
        f"warn it can produce SQLITE_CORRUPT on files that change). "
        f"Got URI: {uri!r}"
    )
    assert "mode=ro" in uri, (
        f"open_db_readonly URI should specify mode=ro; got {uri!r}"
    )


def test_open_db_readonly_can_read_consistent_copy(tmp_path):
    """End-to-end: vacuum a busy WAL DB, then open the copy
    read-only and run a quick_check. Should return ok cleanly."""
    from scripts.colo_bot_db_extract import (
        vacuum_into_consistent_copy, open_db_readonly, check_integrity,
    )

    src = tmp_path / "mm.db"
    dst = tmp_path / "copy.db"
    _make_busy_wal_db(src, rows_pre=300)
    vacuum_into_consistent_copy(src, dst)

    conn = open_db_readonly(dst)
    try:
        ok, detail = check_integrity(conn)
        assert ok, (
            f"check_integrity on a fresh VACUUM INTO copy must return ok; "
            f"got {detail!r}"
        )
        n = conn.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        assert n == 300
    finally:
        conn.close()
