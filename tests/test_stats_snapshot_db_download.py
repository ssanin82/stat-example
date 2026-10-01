"""DB-download robustness for scripts/stats_snapshot.py.

Motivation: tmp/snap_20260419_080117/_snapshot_meta.json captured a
real failure mode —

  RemoteProtocolError('peer closed connection without sending complete
  message body (received 119537664 bytes, expected 119549952)')

Railway's HTTP proxy dropped the stream 12 KB short of a 119 MB DB
(99.99% complete). The partial file was on disk but SQLite refused
to open it (``database disk image is malformed``), and the snapshot
was effectively useless.

The fix in ``fetch_db_with_retries`` adds three defences:

  1. Retry on ``RemoteProtocolError`` / ``ReadError`` etc. — each
     attempt is a fresh HTTP request so proxy idle timers reset.
  2. Download to ``trading.db.partial`` and only rename to
     ``trading.db`` after integrity validation — no window where a
     truncated file masquerades as a good snapshot.
  3. ``PRAGMA integrity_check`` on each completed download so
     silent corruption (truncation inside a page) is detected
     before rename.

Invariants pinned here:

  A. One-shot success — normal path works on a clean server.
  B. Transient failure then success — retry catches Railway's
     end-of-stream drop.
  C. All attempts fail — returns outcome=="exhausted"/"stream_error"
     and LEAVES the partial file for ``.recover`` salvage. The target
     name (``trading.db``) must NOT exist on failure.
  D. HTTP 404 / permission failure — not retryable-successfully but
     still captured with ``response_body`` + ``note`` for the
     operator.
  E. Integrity-check failure on a corrupt body triggers retry (not
     silent success) — the contract is "good file or no file".
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Callable

import httpx
import pytest

# Import the helper — scripts/ is added to sys.path by stats_snapshot
# at import time, but we need the module itself to be importable here.
_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import stats_snapshot  # noqa: E402


def _scratch_dir() -> Path:
    d = Path(tempfile.gettempdir()) / f"mm_stats_snap_{os.getpid()}_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_valid_sqlite_bytes() -> bytes:
    """Produce a real in-memory SQLite DB and return its bytes.

    Using a real file (not a hand-crafted header) means the integrity
    check actually validates — we're testing the pipeline, not a
    stubbed validator.
    """
    tmp = Path(tempfile.gettempdir()) / f"mm_src_{uuid.uuid4().hex}.db"
    try:
        conn = sqlite3.connect(str(tmp))
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany(
            "INSERT INTO t (v) VALUES (?)",
            [(f"row-{i}",) for i in range(100)],
        )
        conn.commit()
        conn.close()
        return tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)


def _client_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
) -> httpx.Client:
    """httpx.Client backed by MockTransport — no network, deterministic."""
    return httpx.Client(transport=httpx.MockTransport(handler), timeout=10.0)


# --------------------- Case A: one-shot success ----------------------


def test_download_success_on_first_attempt_renames_atomically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _make_valid_sqlite_bytes()
    call_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        assert req.url.path == "/db/download", req.url
        return httpx.Response(200, content=body)

    # Patch httpx.Client where ``fetch_db_with_retries`` imports it.
    monkeypatch.setattr(
        stats_snapshot, "httpx",
        type(
            "m",
            (),
            {
                "Client": lambda *a, **kw: _client_with_handler(handler),
                "RequestError": httpx.RequestError,
            },
        ),
    )

    out_dir = _scratch_dir()
    target = out_dir / "trading.db"
    meta = stats_snapshot.fetch_db_with_retries(
        "http://test.local", target, attempts=3, backoff_seconds=0.0
    )
    assert meta["outcome"] == "ok"
    assert meta["http_status"] == 200
    assert meta["bytes"] == len(body)
    # Atomic rename landed the final file.
    assert target.exists(), "trading.db must exist on success"
    assert not target.with_suffix(".db.partial").exists(), (
        "partial temp file must be gone after successful rename"
    )
    assert call_count["n"] == 1, "must not retry on first success"
    # And it's really a SQLite DB.
    ok, detail = stats_snapshot._sqlite_integrity_ok(target)
    assert ok, detail


# --------------------- Case B: transient then success ---------------


def test_download_retries_after_transient_failure_and_renames_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _make_valid_sqlite_bytes()
    call_count = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] < 3:
            # First two attempts blow up with a network-layer error —
            # simulates Railway dropping the connection.
            raise httpx.ReadError("simulated peer closed")
        return httpx.Response(200, content=body)

    monkeypatch.setattr(
        stats_snapshot, "httpx",
        type(
            "m",
            (),
            {
                "Client": lambda *a, **kw: _client_with_handler(handler),
                "RequestError": httpx.RequestError,
            },
        ),
    )

    out_dir = _scratch_dir()
    target = out_dir / "trading.db"
    meta = stats_snapshot.fetch_db_with_retries(
        "http://test.local", target, attempts=5, backoff_seconds=0.0
    )
    assert meta["outcome"] == "ok"
    assert meta["bytes"] == len(body)
    assert target.exists()
    assert call_count["n"] == 3, "must retry until success"
    # Attempts log records every try including the failures.
    attempts = meta["attempts_log"]
    assert isinstance(attempts, list) and len(attempts) == 3
    assert attempts[0]["outcome"] == "stream_error"
    assert attempts[1]["outcome"] == "stream_error"
    assert attempts[2]["outcome"] == "ok"


# --------------------- Case C: all retries exhausted ----------------


def test_all_retries_exhausted_keeps_partial_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every attempt fails, ``trading.db`` must NOT exist (no
    fake good file) but ``trading.db.partial`` is preserved so the
    operator can attempt ``sqlite3 .recover`` on what DID arrive."""
    truncated = _make_valid_sqlite_bytes()[:-32]  # chop the final page trailer

    def handler(req: httpx.Request) -> httpx.Response:
        # Stream returns HTTP 200 + truncated body → integrity check
        # fails → retry exhausts.
        return httpx.Response(200, content=truncated)

    monkeypatch.setattr(
        stats_snapshot, "httpx",
        type(
            "m",
            (),
            {
                "Client": lambda *a, **kw: _client_with_handler(handler),
                "RequestError": httpx.RequestError,
            },
        ),
    )

    out_dir = _scratch_dir()
    target = out_dir / "trading.db"
    meta = stats_snapshot.fetch_db_with_retries(
        "http://test.local", target, attempts=3, backoff_seconds=0.0
    )
    assert meta["outcome"] == "integrity_failed", meta
    assert not target.exists(), (
        "failure path must not create a trading.db — the JSON snapshot "
        "should clearly reflect 'no DB'"
    )
    partial = target.with_suffix(target.suffix + ".partial")
    assert partial.exists(), "partial file must remain for .recover salvage"
    # Each attempt's integrity check was recorded.
    attempts = meta["attempts_log"]
    assert len(attempts) == 3
    for a in attempts:
        assert a["outcome"] == "integrity_failed"
        assert "integrity_check" in a


# --------------------- Case D: HTTP 404 ----------------------------


def test_http_404_is_captured_and_not_obscured_by_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``DB_DOWNLOAD_ENABLED=false`` on the server, the endpoint
    404s. Retrying won't help — but we retry anyway (cheap) and keep
    the diagnostic body + remediation note in the metadata so the
    operator can fix the config."""
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b'{"detail":"Not Found"}')

    monkeypatch.setattr(
        stats_snapshot, "httpx",
        type(
            "m",
            (),
            {
                "Client": lambda *a, **kw: _client_with_handler(handler),
                "RequestError": httpx.RequestError,
            },
        ),
    )

    out_dir = _scratch_dir()
    target = out_dir / "trading.db"
    meta = stats_snapshot.fetch_db_with_retries(
        "http://test.local", target, attempts=2, backoff_seconds=0.0
    )
    assert meta["outcome"] == "http_error"
    assert meta["http_status"] == 404
    assert not target.exists()
    # The remediation hint is in each attempt's info.
    attempts = meta["attempts_log"]
    assert any(
        "DB_DOWNLOAD_ENABLED" in str(a.get("note", ""))
        for a in attempts
    ), "404 attempts must carry the operator remediation note"


# --------------------- Case E: integrity check logic ---------------


def test_sqlite_integrity_ok_on_real_db() -> None:
    p = Path(tempfile.gettempdir()) / f"mm_ok_{uuid.uuid4().hex}.db"
    try:
        p.write_bytes(_make_valid_sqlite_bytes())
        ok, detail = stats_snapshot._sqlite_integrity_ok(p)
        assert ok, f"valid DB must pass, got detail={detail!r}"
        assert detail == "ok"
    finally:
        p.unlink(missing_ok=True)


def test_sqlite_integrity_fails_on_end_truncation() -> None:
    """The 12-KB end-truncation we saw in production must be detected."""
    p = Path(tempfile.gettempdir()) / f"mm_trunc_{uuid.uuid4().hex}.db"
    try:
        # Chop a full page (4096 bytes) off the tail — guaranteed to
        # invalidate a structural check.
        p.write_bytes(_make_valid_sqlite_bytes()[:-4096])
        ok, detail = stats_snapshot._sqlite_integrity_ok(p)
        assert not ok, (
            f"truncated DB must fail integrity, got detail={detail!r}"
        )
    finally:
        p.unlink(missing_ok=True)


def test_sqlite_integrity_fails_on_non_sqlite_garbage() -> None:
    """A non-SQLite file (HTML 500 response body etc.) must fail —
    the integrity check is the last defence against saving a
    gateway-error page as ``trading.db``."""
    p = Path(tempfile.gettempdir()) / f"mm_garbage_{uuid.uuid4().hex}.db"
    try:
        p.write_bytes(b"<html><body>502 Bad Gateway</body></html>" * 100)
        ok, detail = stats_snapshot._sqlite_integrity_ok(p)
        assert not ok, f"garbage file must fail, got detail={detail!r}"
    finally:
        p.unlink(missing_ok=True)
