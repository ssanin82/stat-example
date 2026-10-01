"""``GET /db/download`` — stream the SQLite file for offline analysis.

Gated on ``DB_DOWNLOAD_ENABLED`` (default ``True``). Intentionally split
from the ``CONTROL_ENDPOINTS_ENABLED`` guard used for the mutating
POST ``/control/*`` routes — the DB download is read-only and doesn't
need the same "operator action" guard.

Tests cover:

* **Default behavior**: with no env vars set, `/db/download` returns 200
  (so ``stats_snapshot.py`` works out of the box without operator
  action).
* **Explicit disable**: ``DB_DOWNLOAD_ENABLED=false`` returns 404.
* **Independent from control endpoints**: ``CONTROL_ENDPOINTS_ENABLED``
  does not affect ``/db/download`` either way.
* 200 response body is a valid SQLite file (verified by re-opening it).
* Attachment filename matches the DB basename.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import router
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _client(
    *,
    db_download_enabled: bool | None = None,
    control_enabled: bool | None = None,
) -> tuple[TestClient, Storage, Path]:
    """Build a test client with optional overrides for the two flags.

    When a flag is ``None`` the config default is used — important for
    the "works out of the box" default-True assertions on
    ``DB_DOWNLOAD_ENABLED``.
    """
    path = Path(tempfile.gettempdir()) / f"mm_dl_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base: dict[str, object] = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    }
    if db_download_enabled is not None:
        base["DB_DOWNLOAD_ENABLED"] = db_download_enabled
    if control_enabled is not None:
        base["CONTROL_ENDPOINTS_ENABLED"] = control_enabled
    settings = UnitTestSettings.model_validate(base)
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    app = FastAPI()
    app.include_router(router)
    app.state.settings = settings
    app.state.bot_state = state
    app.state.storage = storage
    return TestClient(app), storage, path


def test_db_download_enabled_by_default() -> None:
    """No env vars set → /db/download returns 200. This is the "works
    out of the box" default so ``scripts/stats_snapshot.py`` just runs
    without any operator action."""
    client, storage, path = _client()
    try:
        r = client.get("/db/download")
        assert r.status_code == 200, (
            f"/db/download must be enabled by default; got {r.status_code}: {r.text[:200]}"
        )
        # And it actually returns a valid SQLite file.
        assert r.content.startswith(b"SQLite format 3\x00")
    finally:
        storage.close()
        path.unlink(missing_ok=True)


def test_db_download_404_when_explicitly_disabled() -> None:
    """Operators who want to gate trading-history downloads can set
    ``DB_DOWNLOAD_ENABLED=false`` to require an explicit opt-in.
    Behaviour in that case is 404 with a clear detail message."""
    client, storage, path = _client(db_download_enabled=False)
    try:
        r = client.get("/db/download")
        assert r.status_code == 404
        assert "db download disabled" in r.text
    finally:
        storage.close()
        path.unlink(missing_ok=True)


def test_db_download_independent_of_control_endpoints_flag() -> None:
    """``CONTROL_ENDPOINTS_ENABLED`` gates the mutating POST routes only;
    flipping it does NOT affect ``/db/download``. Explicit test to pin
    the separation — prevents future refactors from accidentally
    re-coupling them."""
    # Control POSTs explicitly OFF, DB download at its default True.
    client, storage, path = _client(control_enabled=False)
    try:
        r_dl = client.get("/db/download")
        assert r_dl.status_code == 200, (
            "/db/download must not depend on CONTROL_ENDPOINTS_ENABLED"
        )
        r_kill = client.post("/control/kill")
        assert r_kill.status_code == 404, (
            "/control/kill must still be gated when CONTROL_ENDPOINTS_ENABLED=False"
        )
    finally:
        storage.close()
        path.unlink(missing_ok=True)

    # Inverse: control POSTs ON, but DB download explicitly OFF — the
    # flags are fully independent.
    client, storage, path = _client(control_enabled=True, db_download_enabled=False)
    try:
        r_dl = client.get("/db/download")
        assert r_dl.status_code == 404
        r_pause = client.post("/control/pause")
        assert r_pause.status_code == 200, (
            "/control/pause should work when CONTROL_ENDPOINTS_ENABLED=True"
        )
    finally:
        storage.close()
        path.unlink(missing_ok=True)


def test_db_download_returns_valid_sqlite_with_written_rows() -> None:
    """200 response body is a valid SQLite file containing the rows we
    just wrote — confirms the WAL checkpoint runs and the download is
    self-contained."""
    client, storage, path = _client()
    try:
        storage.insert_bot_event(
            "2026-04-18T17:00:00+00:00",
            "INFO",
            "canary_event",
            "canary for download test",
            None,
        )
        r = client.get("/db/download")
        assert r.status_code == 200
        body = r.content
        assert body.startswith(b"SQLite format 3\x00")
        download_path = Path(tempfile.gettempdir()) / f"mm_dl_verify_{uuid.uuid4().hex}.db"
        download_path.write_bytes(body)
        try:
            conn = sqlite3.connect(str(download_path))
            rows = list(
                conn.execute(
                    "SELECT event_type, message FROM bot_events WHERE event_type = ?",
                    ("canary_event",),
                )
            )
            conn.close()
        finally:
            download_path.unlink(missing_ok=True)
        assert len(rows) == 1
        assert rows[0][1] == "canary for download test"
    finally:
        storage.close()
        path.unlink(missing_ok=True)


def test_db_download_filename_matches_db_basename() -> None:
    client, storage, path = _client()
    try:
        r = client.get("/db/download")
        assert r.status_code == 200
        cd = r.headers.get("content-disposition", "")
        assert path.name in cd, f"Content-Disposition={cd!r} missing {path.name!r}"
    finally:
        storage.close()
        path.unlink(missing_ok=True)
