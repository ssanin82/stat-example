"""v1.5.275 / BUG-038 — bot_events_since preserves structural events.

Pre-fix (lines 1692-1709 of app/storage.py): the SQL was
``SELECT * FROM bot_events WHERE ts >= ? ORDER BY ts ASC LIMIT 10000``.
On an 8-hour adverse session, a single high-rate event type
(``target_venue_fast_move_cancel`` at ~6 167 rows in the
v1.5.273-260530-075807 snapshot) saturated the 10 000-row LIMIT
and evicted 21 of 25 SF episode events from the JSON export. The
operator could not reconstruct the SF timeline from the snapshot
alone — they had to drop to ``sqlite3 mm.db`` directly.

Fix: ``bot_events_since`` now returns the union of (a) all
structural events (no cap) + (b) the most-recent ``limit`` rows of
everything else. Structural prefixes defined in
``Storage._BOT_EVENTS_STRUCTURAL_PREFIXES``. The total returned
row count can exceed ``limit`` by the count of structural events
in the window — typically small (10-100), bounded by the structural
types' aggregate firing rate.
"""

from __future__ import annotations

import os
import tempfile
import pytest


def _make_storage(tmp_path):
    """Return a Storage instance with a fresh isolated SQLite DB.

    Both ``DATABASE_URL`` and ``SQLITE_PATH`` are passed because
    ``Settings.effective_sqlite_path()`` checks ``database_url``
    FIRST and falls back to ``sqlite_path`` only when the URL
    isn't a sqlite scheme. Passing both keeps the override
    working under any future internal precedence change.
    """
    import uuid

    from app.config import Settings
    from app.storage import Storage

    db_path = tmp_path / f"test-{uuid.uuid4().hex}.db"
    settings = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
        SQLITE_PATH=str(db_path),
        DATABASE_URL=f"sqlite:///{db_path}",
    )
    storage = Storage(settings)
    storage.init_schema()
    return storage


def _insert_event(storage, ts: str, event_type: str, severity: str = "INFO",
                  message: str = ""):
    """Direct insert into bot_events table for test fixtures (single-row)."""
    with storage.connection() as conn:
        conn.execute(
            "INSERT INTO bot_events (ts, severity, event_type, message, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, severity, event_type, message, "{}"),
        )


def _insert_events_bulk(storage, rows):
    """Bulk-insert events in a single transaction. ``rows`` is an
    iterable of (ts, event_type, severity, message) tuples. Speeds
    the 15K-row eviction test from ~30 s to < 1 s."""
    with storage.connection() as conn:
        conn.executemany(
            "INSERT INTO bot_events (ts, severity, event_type, message, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            [(ts, sev, et, msg, "{}") for ts, et, sev, msg in rows],
        )


def test_structural_events_survive_high_rate_eviction(tmp_path):
    """The smoking gun: pre-fix, a single rare SF event would be
    evicted by 15,000 trash events; post-fix, the SF event survives."""
    storage = _make_storage(tmp_path)

    # 15,000 high-rate trash events (older — would be evicted first
    # by the old ORDER BY ts ASC LIMIT, returned first by the
    # post-fix `most recent` semantic for non-structural).
    trash_rows = []
    for i in range(15000):
        sec = i % 60
        minute = (i // 60) % 60
        hour = (i // 3600) % 24
        ts = f"2026-05-30T{hour:02d}:{minute:02d}:{sec:02d}.{i%1000:03d}+00:00"
        trash_rows.append((ts, "target_venue_fast_move_cancel", "INFO", ""))
    _insert_events_bulk(storage, trash_rows)

    # 5 structural SF events scattered throughout the day.
    sf_events = [
        ("2026-05-30T01:00:00.000+00:00", "soft_flatten_started"),
        ("2026-05-30T01:00:10.000+00:00", "soft_flatten_completed"),
        ("2026-05-30T02:00:00.000+00:00", "soft_flatten_started"),
        ("2026-05-30T05:00:00.000+00:00", "soft_flatten_started"),
        ("2026-05-30T07:00:00.000+00:00", "sf_fatigue_tier4_kill"),
    ]
    for ts, et in sf_events:
        _insert_event(storage, ts, et)

    # Call bot_events_since with the default 10,000 cap.
    rows = storage.bot_events_since("2026-05-30T00:00:00.000+00:00", limit=10000)

    # ALL 5 SF events must be present.
    sf_types_returned = [
        r["event_type"] for r in rows
        if r["event_type"].startswith("soft_flatten_")
        or r["event_type"].startswith("sf_fatigue_")
    ]
    assert len(sf_types_returned) == 5, (
        f"BUG-038 regression: only {len(sf_types_returned)} of 5 SF events "
        f"survived eviction. Got: {sf_types_returned}"
    )
    # Specifically, the sf_fatigue_tier4_kill (last event) must be
    # present even though 15K trash events are between it and the
    # first events.
    assert any(
        r["event_type"] == "sf_fatigue_tier4_kill" for r in rows
    ), "the terminal kill event must survive eviction"


def test_non_structural_events_limited_to_most_recent(tmp_path):
    """Non-structural events get the LIMIT cap, and the cap returns
    the MOST RECENT rows (not the oldest, which was the pre-fix
    behavior that evicted recent SF events)."""
    storage = _make_storage(tmp_path)

    # 5,000 trash events spanning 5 hours (1 per second-ish).
    rows = []
    for i in range(5000):
        ts = f"2026-05-30T{(i // 3600):02d}:{((i // 60) % 60):02d}:{(i % 60):02d}.000+00:00"
        rows.append((ts, "target_venue_fast_move_cancel", "INFO", f"event#{i}"))
    _insert_events_bulk(storage, rows)

    # Call with limit=100. Should return the 100 MOST RECENT trash events.
    rows = storage.bot_events_since(
        "2026-05-30T00:00:00.000+00:00", limit=100,
    )
    trash = [r for r in rows
             if r["event_type"] == "target_venue_fast_move_cancel"]
    assert len(trash) == 100, f"expected 100 trash events, got {len(trash)}"

    # The kept events must be the LATEST 100, not the earliest.
    # Extract the event index from message (we wrote event#N).
    indices = sorted(int(r["message"].split("#")[1]) for r in trash)
    assert indices[0] >= 4900, (
        f"expected kept events to be the most recent (indices >= 4900); "
        f"got earliest index {indices[0]} — looks like the pre-fix "
        f"ORDER BY ts ASC was still being used"
    )


def test_total_rows_returned_can_exceed_limit_by_structural_count(tmp_path):
    """The semantic is: limit applies to non-structural events; the
    structural events are returned in full regardless. Confirm that
    the returned row count = (structural in window) + min(limit, non-structural)."""
    storage = _make_storage(tmp_path)

    # 500 non-structural events.
    rows = [
        (f"2026-05-30T01:{(i // 60):02d}:{(i % 60):02d}.000+00:00",
         "quote_side_suppressed", "INFO", "")
        for i in range(500)
    ]
    # 30 structural events.
    rows += [
        (f"2026-05-30T02:{(i // 60):02d}:{(i % 60):02d}.000+00:00",
         "regime_mode_transition", "INFO", "")
        for i in range(30)
    ]
    _insert_events_bulk(storage, rows)

    rows = storage.bot_events_since("2026-05-30T00:00:00.000+00:00", limit=200)
    # Should have 200 non-structural (capped) + 30 structural (uncapped) = 230.
    assert len(rows) == 230, (
        f"expected 230 rows (200 capped trash + 30 uncapped structural); "
        f"got {len(rows)}"
    )


def test_empty_db_returns_empty_list(tmp_path):
    """Edge case: empty bot_events table."""
    storage = _make_storage(tmp_path)
    rows = storage.bot_events_since("2026-05-30T00:00:00.000+00:00", limit=10000)
    assert rows == []


def test_until_ts_caps_window(tmp_path):
    """The until_ts parameter still works on both partitions."""
    storage = _make_storage(tmp_path)

    _insert_event(storage, "2026-05-30T01:00:00.000+00:00",
                  "soft_flatten_started")  # in window
    _insert_event(storage, "2026-05-30T03:00:00.000+00:00",
                  "soft_flatten_started")  # outside window
    _insert_event(storage, "2026-05-30T01:30:00.000+00:00",
                  "target_venue_fast_move_cancel")  # in window
    _insert_event(storage, "2026-05-30T04:00:00.000+00:00",
                  "target_venue_fast_move_cancel")  # outside window

    rows = storage.bot_events_since(
        "2026-05-30T00:00:00.000+00:00",
        until_ts="2026-05-30T02:00:00.000+00:00",
    )
    timestamps = sorted(r["ts"] for r in rows)
    assert all(ts < "2026-05-30T02:00:00.000+00:00" for ts in timestamps)
    assert len(rows) == 2


def test_returned_rows_sorted_by_timestamp(tmp_path):
    """Merge of two sub-queries must produce ts-ascending output."""
    storage = _make_storage(tmp_path)

    # Interleave structural and non-structural events.
    events = [
        ("2026-05-30T01:00:00.000+00:00", "target_venue_fast_move_cancel"),
        ("2026-05-30T01:00:05.000+00:00", "soft_flatten_started"),
        ("2026-05-30T01:00:10.000+00:00", "quote_side_suppressed"),
        ("2026-05-30T01:00:15.000+00:00", "regime_mode_transition"),
        ("2026-05-30T01:00:20.000+00:00", "target_venue_fast_move_cancel"),
        ("2026-05-30T01:00:25.000+00:00", "sf_fatigue_tier1_armed"),
    ]
    for ts, et in events:
        _insert_event(storage, ts, et)

    rows = storage.bot_events_since("2026-05-30T00:00:00.000+00:00")
    timestamps = [r["ts"] for r in rows]
    assert timestamps == sorted(timestamps), (
        f"output must be ts-sorted; got {timestamps}"
    )


def test_structural_prefix_list_contains_known_critical_types():
    """Lock in the structural-prefix set so the next person who adds
    a new event type knows where to register it."""
    from app.storage import Storage

    expected_prefixes = {
        "soft_flatten_", "sf_fatigue_", "regime_mode_transition",
        "shock_gate_", "desync_", "kill", "shadow_position_divergence",
    }
    actual_prefixes = set(Storage._BOT_EVENTS_STRUCTURAL_PREFIXES)
    missing = expected_prefixes - actual_prefixes
    assert not missing, (
        f"v1.5.275 fix regression: structural prefixes missing: {missing}"
    )
