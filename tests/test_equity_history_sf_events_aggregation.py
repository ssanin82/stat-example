"""Regression test for ``equity_history_publisher`` SF events
aggregation (v1.4.162).

Bug history: pre-v1.4.162 the publisher grouped ``fills`` rows on
``soft_flatten_event_id`` to build the ``sf_events`` list rendered as
yellow vertical markers on the dashboard's Session PnL chart. That
worked for SF episodes closed via the passive post-only path (where
the ``WorkingOrder`` constructor stamps the SF tag and fill ingestion
propagates it), but FAILED for episodes closed via the taker-fallback
path in ``bot.py`` line 2369-2374: ``self._client.market_close(...)``
bypasses the normal place-order flow entirely, so no row is inserted
into the local ``orders`` table and the resulting taker fills end up
with ``soft_flatten_event_id=None`` at ingest time.

Reproduction in production: snapshot
``v1.4.157-260520-213540-prod.okx.ton.usdt.perp`` — SF#11167 closed
via the taker fallback, 3 fills at 17:15:00.757, none tagged, no
marker rendered.

Fix: the publisher now queries the ``soft_flatten_events`` table
directly (which IS always populated regardless of close path). Every
SF episode renders a marker.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.equity_history_publisher import EquityHistoryPublisher
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _make_settings() -> UnitTestSettings:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_sfeh_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "LOGS_BUCKET": "fake-bucket",
        }
    )


def _make_publisher(settings: UnitTestSettings) -> tuple[
    EquityHistoryPublisher, BotState, Storage
]:
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    # Force a fresh session-start so any seeded SF events are
    # in-window.
    state.session_started_at_utc = datetime(
        2026, 5, 20, 16, 50, 0, tzinfo=timezone.utc
    )
    pub = EquityHistoryPublisher(
        settings=settings,
        state=state,
        bucket="fake-bucket",
        profile_name="test",
        storage=storage,
    )
    return pub, state, storage


def _insert_sf_event(
    storage: Storage,
    *,
    ts_start: str,
    ts_end: str | None = None,
    entry_position_qty: float = 0.0,
    exit_reason: str = "completed",
) -> int:
    """Insert an SF episode row directly. Mirrors what
    ``soft_flatten_tick.enter_soft_flatten`` does at runtime."""
    return storage.insert_soft_flatten_event(
        {
            "ts_start": ts_start,
            "trigger_reason": "position_drawdown_test",
            "initial_force_phase": 0,
            "taker_fallback_ticks": 5,
            "entry_position_qty": entry_position_qty,
            "entry_mid_price": 2.04,
            "notes": "test fixture",
        }
    )


def test_sf_marker_renders_for_taker_fallback_episode() -> None:
    """The bug reproduction: an SF episode with NO tagged fills (the
    taker fallback case) must still produce a marker entry in
    ``sf_events``."""
    s = _make_settings()
    pub, state, storage = _make_publisher(s)
    # Insert an SF event with no associated fills (simulates
    # taker-fallback close).
    event_id = _insert_sf_event(
        storage,
        ts_start="2026-05-20T17:14:46.000+00:00",
        ts_end="2026-05-20T17:15:00.757+00:00",
        entry_position_qty=-9.0,  # bot was short → flatten = BUY
        exit_reason="taker_fallback_after_adverse_drift",
    )
    storage.update_soft_flatten_event_end(
        event_id,
        {
            "ts_end": "2026-05-20T17:15:00.757+00:00",
            "exit_phase_reached": 1,
            "exit_reason": "taker_fallback_after_adverse_drift",
        },
    )

    payload = pub._build_payload()
    sf_events = payload["sf_events"]

    assert len(sf_events) == 1, (
        f"Expected 1 SF marker for the taker-fallback episode; "
        f"got {len(sf_events)}: {sf_events}"
    )
    m = sf_events[0]
    assert m["event_id"] == event_id
    assert m["ts_first_fill"] == "2026-05-20T17:14:46.000+00:00"
    assert m["ts_last_fill"] == "2026-05-20T17:15:00.757+00:00"
    # Bot was short -9 → close side = BUY.
    assert m["side"] == "BUY"
    # No tagged fills exist (taker-fallback) → counts are zero, but
    # the marker still renders.
    assert m["fill_count"] == 0
    assert m["notional_usd"] == 0.0


def test_sf_marker_includes_attributed_fill_counts_when_tagged() -> None:
    """When SF episodes DO have tagged fills (the passive close path),
    the marker carries the correct count + notional sum."""
    s = _make_settings()
    pub, state, storage = _make_publisher(s)
    event_id = _insert_sf_event(
        storage,
        ts_start="2026-05-20T17:00:00.000+00:00",
        ts_end="2026-05-20T17:00:30.000+00:00",
        entry_position_qty=+6.0,  # bot was long → flatten = SELL
        exit_reason="completed_post_only",
    )
    # Stamp two tagged fills against this SF event.
    from app.enums import Side
    from app.models import Fill
    from app.fill_ingestion import fill_row

    for i, px in enumerate([2.041, 2.040]):
        f = Fill(
            fill_id=f"f{event_id}-{i}",
            order_id_exchange=f"ord-{event_id}-{i}",
            client_order_id=f"cl-{event_id}-{i}",
            ts_fill=datetime(
                2026, 5, 20, 17, 0, 10 + i, tzinfo=timezone.utc
            ),
            symbol="TON-USDT-SWAP",
            side=Side.SELL,
            price=px,
            size=3.0,
            notional=3.0 * px,
            fee=-0.001,
            liquidity_flag="resting",
            mid_at_fill=px,
            soft_flatten_event_id=event_id,
        )
        storage.insert_fill_row(fill_row(f))

    payload = pub._build_payload()
    sf_events = payload["sf_events"]
    assert len(sf_events) == 1
    m = sf_events[0]
    assert m["event_id"] == event_id
    assert m["side"] == "SELL"
    assert m["fill_count"] == 2
    # 3.0 × 2.041 + 3.0 × 2.040 = 12.243; rounded to 4 dp.
    assert abs(m["notional_usd"] - 12.243) < 1e-6


def test_sf_markers_sorted_chronologically() -> None:
    """Multiple SF episodes in a session must be sorted by ts_start
    (which we surface as ``ts_first_fill`` for the frontend)."""
    s = _make_settings()
    pub, state, storage = _make_publisher(s)
    later_id = _insert_sf_event(
        storage,
        ts_start="2026-05-20T17:30:00.000+00:00",
        entry_position_qty=-3.0,
    )
    earlier_id = _insert_sf_event(
        storage,
        ts_start="2026-05-20T17:00:00.000+00:00",
        entry_position_qty=+3.0,
    )
    payload = pub._build_payload()
    sf_events = payload["sf_events"]
    assert len(sf_events) == 2
    assert sf_events[0]["event_id"] == earlier_id
    assert sf_events[1]["event_id"] == later_id


def test_sf_markers_respect_session_start_filter() -> None:
    """SF events that started BEFORE the current session must not
    appear in this session's chart."""
    s = _make_settings()
    pub, state, storage = _make_publisher(s)
    # Session starts at 16:50. Insert one SF event from a prior
    # session (before session start) and one from the current.
    prior_id = _insert_sf_event(
        storage,
        ts_start="2026-05-19T10:00:00.000+00:00",  # well before
        entry_position_qty=-3.0,
    )
    current_id = _insert_sf_event(
        storage,
        ts_start="2026-05-20T17:00:00.000+00:00",
        entry_position_qty=+3.0,
    )
    payload = pub._build_payload()
    sf_events = payload["sf_events"]
    event_ids = {e["event_id"] for e in sf_events}
    assert current_id in event_ids
    assert prior_id not in event_ids, (
        "Pre-session SF event leaked into current-session markers"
    )


def test_sf_markers_empty_when_no_episodes() -> None:
    s = _make_settings()
    pub, state, storage = _make_publisher(s)
    payload = pub._build_payload()
    assert payload["sf_events"] == []


def test_sf_event_with_zero_entry_qty_has_no_side() -> None:
    """An SF event armed at qty=0 (shouldn't really happen but is
    survivable) renders with ``side=None``."""
    s = _make_settings()
    pub, state, storage = _make_publisher(s)
    _insert_sf_event(
        storage,
        ts_start="2026-05-20T17:00:00.000+00:00",
        entry_position_qty=0.0,
    )
    payload = pub._build_payload()
    assert len(payload["sf_events"]) == 1
    assert payload["sf_events"][0]["side"] is None
