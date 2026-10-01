from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings_for_db(path: Path) -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )


def test_storage_reuses_single_connection_until_close(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    path = Path(tempfile.gettempdir()) / f"mm_storage_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = Storage(_settings_for_db(path))
    s.init_schema()
    conn0 = s._conn
    s.insert_bot_event("2026-01-01T00:00:00+00:00", "INFO", "test", "m", None)
    s.insert_bot_event("2026-01-01T00:00:01+00:00", "INFO", "test", "m", None)
    assert s._conn is conn0
    s.close()
    with pytest.raises(RuntimeError):
        s.recent_bot_events(1)
    path.unlink(missing_ok=True)


def test_storage_close_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    path = Path(tempfile.gettempdir()) / f"mm_storage_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = Storage(_settings_for_db(path))
    s.init_schema()
    s.close()
    s.close()  # no-op / idempotent
    with pytest.raises(RuntimeError):
        s.insert_bot_event("2026-01-01T00:00:02+00:00", "INFO", "test", "m", None)
    path.unlink(missing_ok=True)
