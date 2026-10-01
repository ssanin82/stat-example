"""Tests for ``app/dashboard_state_publisher.py``.

Covers the publisher prereq from plans/20260514-execution.md:

1. v31 schema migration: ``equity_snapshots.binance_basis_ewma`` exists.
2. ``maybe_create`` returns None when disabled or unconfigured.
3. ``maybe_create`` returns a working instance when configured.
4. ``_safe_publish_once`` writes three S3 PUTs (one per table).
5. ``_safe_publish_once`` isolates failures per table — one bad PUT
   doesn't suppress the other two.
6. ``session_cross_venue_cancel_count`` initialises to 0 on BotState.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.dashboard_state_publisher import DashboardStatePublisher
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _db() -> tuple[Storage, Path]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_dash_pub_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {"DATABASE_URL": f"sqlite:///{path.as_posix()}"}
    )
    storage = Storage(settings)
    storage.init_schema()
    return storage, path


def _cleanup(storage: Storage, path: Path) -> None:
    storage.close()
    path.unlink(missing_ok=True)


def _state() -> BotState:
    """Minimal BotState for publisher tests. Real BotState constructor
    only requires settings; everything else is initialised to defaults."""
    settings = UnitTestSettings.model_validate({})
    return BotState(settings)


# ----------------------------------------------------------------------
# 1. v31 migration adds the basis-EWMA column to equity_snapshots
# ----------------------------------------------------------------------


def test_migration_v31_adds_binance_basis_ewma_to_equity_snapshots() -> None:
    """The v31 ALTER adds the new column. Inserting a row that
    references the column must succeed."""
    storage, path = _db()
    try:
        # The migration ran in init_schema(). Verify the column is
        # readable by inserting + selecting it.
        storage.insert_equity_snapshot(
            {
                "ts": "2026-05-14T12:00:00+00:00",
                "equity_usd": 100.0,
                "cash_usd": 100.0,
                "realized_pnl_usd": 0.0,
                "unrealized_pnl_usd": 0.0,
                "fees_usd": 0.0,
                "drawdown_usd": 0.0,
                "binance_basis_ewma": 0.0123,
            }
        )
        rows = storage.equity_history_since("2026-05-14T00:00:00+00:00")
        assert len(rows) == 1
        assert rows[0]["binance_basis_ewma"] == pytest.approx(0.0123)
    finally:
        _cleanup(storage, path)


def test_migration_v31_allows_null_basis_ewma_for_backward_compat() -> None:
    """Pre-1.3.31 rows had no basis EWMA. NULL on the column must
    round-trip cleanly."""
    storage, path = _db()
    try:
        storage.insert_equity_snapshot(
            {
                "ts": "2026-05-14T12:00:00+00:00",
                "equity_usd": 100.0,
                "cash_usd": 100.0,
                "realized_pnl_usd": 0.0,
                "unrealized_pnl_usd": 0.0,
                "fees_usd": 0.0,
                "drawdown_usd": 0.0,
                # No binance_basis_ewma — should default to NULL.
            }
        )
        rows = storage.equity_history_since("2026-05-14T00:00:00+00:00")
        assert len(rows) == 1
        assert rows[0]["binance_basis_ewma"] is None
    finally:
        _cleanup(storage, path)


# ----------------------------------------------------------------------
# 2. maybe_create gating
# ----------------------------------------------------------------------


def test_maybe_create_returns_none_when_disabled() -> None:
    settings = UnitTestSettings.model_validate(
        {
            "OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED": "false",
            "LIVE_STATS_ENABLED": "true",
            "LOGS_BUCKET": "test-bucket",
        }
    )
    state = _state()
    result = DashboardStatePublisher.maybe_create(
        settings, state, "prof", storage=MagicMock()
    )
    assert result is None


def test_maybe_create_returns_none_when_live_stats_disabled() -> None:
    """Sibling gate: if the global S3 publish kill-switch is off, this
    publisher also stays off."""
    settings = UnitTestSettings.model_validate(
        {
            "OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED": "true",
            "LIVE_STATS_ENABLED": "false",
            "LOGS_BUCKET": "test-bucket",
        }
    )
    state = _state()
    result = DashboardStatePublisher.maybe_create(
        settings, state, "prof", storage=MagicMock()
    )
    assert result is None


def test_maybe_create_returns_none_when_no_bucket() -> None:
    settings = UnitTestSettings.model_validate(
        {
            "OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED": "true",
            "LIVE_STATS_ENABLED": "true",
            "LOGS_BUCKET": "",
        }
    )
    state = _state()
    result = DashboardStatePublisher.maybe_create(
        settings, state, "prof", storage=MagicMock()
    )
    assert result is None


def test_maybe_create_returns_none_when_no_storage() -> None:
    settings = UnitTestSettings.model_validate(
        {
            "OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED": "true",
            "LIVE_STATS_ENABLED": "true",
            "LOGS_BUCKET": "test-bucket",
        }
    )
    state = _state()
    result = DashboardStatePublisher.maybe_create(
        settings, state, "prof", storage=None
    )
    assert result is None


def test_maybe_create_returns_instance_when_configured() -> None:
    settings = UnitTestSettings.model_validate(
        {
            "OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED": "true",
            "LIVE_STATS_ENABLED": "true",
            "LOGS_BUCKET": "test-bucket",
        }
    )
    state = _state()
    storage = MagicMock()
    result = DashboardStatePublisher.maybe_create(
        settings, state, "prof.okx.sui.usdt.perp", storage=storage
    )
    assert result is not None
    assert result._bucket == "test-bucket"
    assert result._profile_name == "prof.okx.sui.usdt.perp"


# ----------------------------------------------------------------------
# 3. _safe_publish_once writes three S3 PUTs
# ----------------------------------------------------------------------


def test_safe_publish_once_writes_three_s3_puts() -> None:
    """One PUT per table per tick. Each PUT carries an
    independently-built payload."""
    storage, path = _db()
    try:
        settings = UnitTestSettings.model_validate(
            {
                "OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED": "true",
                "LIVE_STATS_ENABLED": "true",
                "LOGS_BUCKET": "test-bucket",
            }
        )
        state = _state()
        publisher = DashboardStatePublisher.maybe_create(
            settings, state, "prof.test", storage=storage
        )
        assert publisher is not None
        publisher._client = MagicMock()

        publisher._safe_publish_once()

        # Three PUTs: exposure / fills / orders_lifecycle.
        assert publisher._client.put_object.call_count == 3
        keys = [
            c.kwargs["Key"] for c in publisher._client.put_object.call_args_list
        ]
        assert "dashboard/exposure_since_prof.test.json" in keys
        assert "dashboard/fills_since_prof.test.json" in keys
        assert (
            "dashboard/orders_lifecycle_since_prof.test.json" in keys
        )
        # All zero errors after a clean publish.
        assert publisher._consecutive_errors == {
            "exposure": 0,
            "fills": 0,
            "orders_lifecycle": 0,
        }
    finally:
        _cleanup(storage, path)


def test_safe_publish_once_isolates_per_table_failures() -> None:
    """One bad PUT (e.g. transient S3 error) must NOT suppress the
    other two healthy PUTs. Verifies per-table error counters."""
    storage, path = _db()
    try:
        settings = UnitTestSettings.model_validate(
            {
                "OBSERVABILITY_DASHBOARD_PUBLISH_ENABLED": "true",
                "LIVE_STATS_ENABLED": "true",
                "LOGS_BUCKET": "test-bucket",
            }
        )
        state = _state()
        publisher = DashboardStatePublisher.maybe_create(
            settings, state, "prof.test", storage=storage
        )
        assert publisher is not None

        mock_client = MagicMock()

        def _put_object_side_effect(*, Bucket, Key, Body, **kw):
            # Fail only for the fills key; succeed for the others.
            if "fills_since" in Key:
                raise RuntimeError("transient s3 error")
            return None

        mock_client.put_object.side_effect = _put_object_side_effect
        publisher._client = mock_client

        publisher._safe_publish_once()

        # Three calls attempted (one per table) regardless of failure.
        assert mock_client.put_object.call_count == 3
        # Fills counter bumped to 1; other two stayed at 0.
        assert publisher._consecutive_errors["fills"] == 1
        assert publisher._consecutive_errors["exposure"] == 0
        assert publisher._consecutive_errors["orders_lifecycle"] == 0
    finally:
        _cleanup(storage, path)


# ----------------------------------------------------------------------
# 4. session_cross_venue_cancel_count counter
# ----------------------------------------------------------------------


def test_bot_state_initialises_cross_venue_cancel_count_to_zero() -> None:
    """The new counter must exist as an attribute with default 0 so
    live_stats can always read it without AttributeError. Same
    discipline as bbo_event_count_session."""
    state = _state()
    assert hasattr(state, "session_cross_venue_cancel_count")
    assert state.session_cross_venue_cancel_count == 0
