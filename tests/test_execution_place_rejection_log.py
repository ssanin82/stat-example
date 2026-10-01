"""A rejected place must leave a searchable log line at the execution layer.

Before this, a GRVT 400 turned into a silent ``OrderStatus.REJECTED`` with the
reason buried in the SQLite orders row. Operators had no way to see why quotes
kept disappearing without attaching a debugger or querying the DB. The
adapter now emits a venue-specific ``*_create_order_rejected`` line, and
execution emits a cross-venue ``place_order_rejected`` summary.
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.enums import Side
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[OrderManager, Any, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_exec_reject_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    # Simulate a GRVT/HL 400: adapter parsers turn this into exchange_rejected.
    client.place_post_only_limit.return_value = {
        "code": 3,
        "message": "invalid client_order_id: expected uint64",
    }
    client.interpret_place_response.return_value = (
        None,
        "exchange_rejected",
        "invalid client_order_id: expected uint64",
    )
    om = OrderManager(settings, client, storage, state)
    return om, client, path


def test_place_rejection_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    om, _client, path = _setup()
    try:
        with caplog.at_level(logging.WARNING, logger="app.execution"):
            om.place_passive_order_manual_only(Side.BUY, 3000.0, 0.01, "q1")
        matches = [
            r
            for r in caplog.records
            if r.getMessage().startswith("place_order_rejected")
        ]
        assert matches, "expected a place_order_rejected warning after a 400"
        msg = matches[0].getMessage()
        assert "side=BUY" in msg
        assert "invalid client_order_id" in msg
        assert "outcome=exchange_rejected" in msg
    finally:
        path.unlink(missing_ok=True)
