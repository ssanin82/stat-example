"""Regression tests for the time-to-flat post-fill watcher.

Originally Phase 4c was scoped to ship MAE/MFE only; ``time_to_flat_seconds``
was deferred during the rush. Implemented after the operator's challenge
"why can't this be implemented right now?" — the answer was no good
reason, and this test file verifies it works.

Covers:
- Schema v30: ``time_to_flat_seconds`` column on fills.
- ``maybe_create`` factory: respects the feature flag + storage gate.
- ``track_fill`` records the fill.
- Detection: when ``state.position.position_qty`` returns to zero,
  the watcher writes the elapsed seconds via UPDATE.
- Timeout: when position never flattens within the max-wait cap,
  the row stays NULL and the record is dropped (no UPDATE called).
- Lifecycle: start() → stop() → thread joinable.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.models import PositionSnapshot
from app.post_fill_time_to_flat_watcher import PostFillTimeToFlatWatcher
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _make_settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "SYMBOL": "TON-USDT-SWAP",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_storage() -> Storage:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_ttf_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    s = Storage(settings)
    s.init_schema()
    return s


def _set_position_qty(state: BotState, qty: float) -> None:
    state.position = PositionSnapshot(
        symbol="X",
        position_qty=qty,
        avg_entry_price=None,
        mark_price=None,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )


# --------------------------------------------------------------------- #
# Schema                                                                #
# --------------------------------------------------------------------- #


def test_schema_v30_time_to_flat_column_exists() -> None:
    s = _make_storage()
    with s.connection() as conn:
        cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(fills)"
            ).fetchall()
        }
    assert "time_to_flat_seconds" in cols


def test_storage_update_fill_time_to_flat_roundtrip() -> None:
    from app.enums import Side
    from app.fill_ingestion import fill_row
    from app.models import Fill

    s = _make_storage()
    f = Fill(
        fill_id="ttf-rt",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        symbol="X",
        side=Side.BUY,
        price=100.0,
        size=1.0,
        notional=100.0,
        fee=-0.01,
        liquidity_flag="resting",
        mid_at_fill=100.0,
    )
    s.insert_fill_row(fill_row(f))
    s.update_fill_time_to_flat("ttf-rt", 12.3)
    with s.connection() as conn:
        row = conn.execute(
            "SELECT time_to_flat_seconds FROM fills WHERE fill_id='ttf-rt'"
        ).fetchone()
    assert row[0] == 12.3


def test_storage_update_fill_time_to_flat_coalesce_preserves_first() -> None:
    """COALESCE semantics: first write wins. Re-writing the same
    fill_id later (rare, but defensive) doesn't clobber the original
    measurement."""
    from app.enums import Side
    from app.fill_ingestion import fill_row
    from app.models import Fill

    s = _make_storage()
    f = Fill(
        fill_id="ttf-coal",
        order_id_exchange=1, client_order_id=None,
        ts_fill=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        symbol="X", side=Side.BUY, price=100.0, size=1.0,
        notional=100.0, fee=-0.01, liquidity_flag="resting",
        mid_at_fill=100.0,
    )
    s.insert_fill_row(fill_row(f))
    s.update_fill_time_to_flat("ttf-coal", 5.5)
    s.update_fill_time_to_flat("ttf-coal", 99.0)  # second write — ignored
    with s.connection() as conn:
        row = conn.execute(
            "SELECT time_to_flat_seconds FROM fills WHERE fill_id='ttf-coal'"
        ).fetchone()
    assert row[0] == 5.5


# --------------------------------------------------------------------- #
# Watcher factory + lifecycle                                           #
# --------------------------------------------------------------------- #


def test_watcher_disabled_by_default() -> None:
    """Default-off — like MAE/MFE."""
    settings = _make_settings()
    state = BotState(settings)
    storage = _make_storage()
    w = PostFillTimeToFlatWatcher.maybe_create(settings, state, storage)
    assert w is None


def test_watcher_enabled_with_flag() -> None:
    settings = _make_settings(OBSERVABILITY_TIME_TO_FLAT_ENABLED=True)
    state = BotState(settings)
    storage = _make_storage()
    w = PostFillTimeToFlatWatcher.maybe_create(settings, state, storage)
    assert w is not None


def test_watcher_disabled_when_storage_absent() -> None:
    settings = _make_settings(OBSERVABILITY_TIME_TO_FLAT_ENABLED=True)
    state = BotState(settings)
    w = PostFillTimeToFlatWatcher.maybe_create(settings, state, storage=None)
    assert w is None


def test_watcher_track_fill_records_state() -> None:
    settings = _make_settings(OBSERVABILITY_TIME_TO_FLAT_ENABLED=True)
    state = BotState(settings)
    storage = _make_storage()
    w = PostFillTimeToFlatWatcher(settings, state, storage)
    w.track_fill("f1", position_qty_before_fill=2.0)
    assert "f1" in w._records
    rec = w._records["f1"]
    assert rec["pos_before_fill"] == 2.0


def test_watcher_start_stop_cleanly() -> None:
    settings = _make_settings(
        OBSERVABILITY_TIME_TO_FLAT_ENABLED=True,
        OBSERVABILITY_TIME_TO_FLAT_POLL_INTERVAL_SECONDS=0.05,
    )
    state = BotState(settings)
    storage = _make_storage()
    w = PostFillTimeToFlatWatcher(settings, state, storage)
    w.start()
    time.sleep(0.05)
    assert w._thread is not None and w._thread.is_alive()
    w.stop()
    w._thread.join(timeout=1.0)
    assert not w._thread.is_alive()


# --------------------------------------------------------------------- #
# Detection: position returns to flat → UPDATE fires                    #
# --------------------------------------------------------------------- #


def test_watcher_fires_on_flat_crossing() -> None:
    """End-to-end: fill happens at position=+3, eventually position
    returns to 0, watcher writes elapsed time."""
    from app.enums import Side
    from app.fill_ingestion import fill_row
    from app.models import Fill

    settings = _make_settings(
        OBSERVABILITY_TIME_TO_FLAT_ENABLED=True,
        OBSERVABILITY_TIME_TO_FLAT_POLL_INTERVAL_SECONDS=0.05,
        OBSERVABILITY_TIME_TO_FLAT_MAX_WAIT_SECONDS=10.0,
    )
    state = BotState(settings)
    storage = _make_storage()

    # Pre-insert a fill so the UPDATE has a target.
    f = Fill(
        fill_id="ttf-flat",
        order_id_exchange=1, client_order_id=None,
        ts_fill=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        symbol="X", side=Side.BUY, price=100.0, size=1.0,
        notional=100.0, fee=-0.01, liquidity_flag="resting",
        mid_at_fill=100.0,
    )
    storage.insert_fill_row(fill_row(f))

    # Start with non-zero position.
    _set_position_qty(state, 3.0)

    w = PostFillTimeToFlatWatcher(settings, state, storage)
    w.start()
    w.track_fill("ttf-flat", position_qty_before_fill=0.0)

    # Wait briefly while position is non-zero.
    time.sleep(0.2)
    # Now bring position to flat.
    _set_position_qty(state, 0.0)
    # Give the watcher a couple of poll cycles to detect.
    time.sleep(0.2)
    w.stop()
    w._thread.join(timeout=1.0)

    with storage.connection() as conn:
        row = conn.execute(
            "SELECT time_to_flat_seconds FROM fills WHERE fill_id='ttf-flat'"
        ).fetchone()
    assert row[0] is not None, "time_to_flat_seconds should have been written"
    # Should be in the 0.2-0.5s range (elapsed from track_fill to flat).
    assert 0.1 <= row[0] <= 1.0, f"expected ~0.2-0.5s, got {row[0]}"


def test_watcher_fires_immediately_if_already_flat() -> None:
    """If position is already flat when the watcher polls, the first
    crossing detection fires immediately. Elapsed time will be very
    small (one poll interval) but non-NULL."""
    from app.enums import Side
    from app.fill_ingestion import fill_row
    from app.models import Fill

    settings = _make_settings(
        OBSERVABILITY_TIME_TO_FLAT_ENABLED=True,
        OBSERVABILITY_TIME_TO_FLAT_POLL_INTERVAL_SECONDS=0.05,
    )
    state = BotState(settings)
    storage = _make_storage()
    f = Fill(
        fill_id="ttf-imm",
        order_id_exchange=1, client_order_id=None,
        ts_fill=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        symbol="X", side=Side.BUY, price=100.0, size=1.0,
        notional=100.0, fee=-0.01, liquidity_flag="resting",
        mid_at_fill=100.0,
    )
    storage.insert_fill_row(fill_row(f))
    _set_position_qty(state, 0.0)  # already flat

    w = PostFillTimeToFlatWatcher(settings, state, storage)
    w.start()
    w.track_fill("ttf-imm", position_qty_before_fill=2.0)
    time.sleep(0.15)
    w.stop()
    w._thread.join(timeout=1.0)

    with storage.connection() as conn:
        row = conn.execute(
            "SELECT time_to_flat_seconds FROM fills WHERE fill_id='ttf-imm'"
        ).fetchone()
    assert row[0] is not None
    assert row[0] < 0.3  # immediate detection


def test_watcher_times_out_when_position_stays_nonzero() -> None:
    """When the cap expires without flat-crossing, the watcher drops
    the record without writing. Column stays NULL — signaling
    'didn't flatten within the window'."""
    from app.enums import Side
    from app.fill_ingestion import fill_row
    from app.models import Fill

    settings = _make_settings(
        OBSERVABILITY_TIME_TO_FLAT_ENABLED=True,
        OBSERVABILITY_TIME_TO_FLAT_POLL_INTERVAL_SECONDS=0.02,
        OBSERVABILITY_TIME_TO_FLAT_MAX_WAIT_SECONDS=0.2,  # tight cap for test
    )
    state = BotState(settings)
    storage = _make_storage()
    f = Fill(
        fill_id="ttf-timeout",
        order_id_exchange=1, client_order_id=None,
        ts_fill=datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc),
        symbol="X", side=Side.BUY, price=100.0, size=1.0,
        notional=100.0, fee=-0.01, liquidity_flag="resting",
        mid_at_fill=100.0,
    )
    storage.insert_fill_row(fill_row(f))
    _set_position_qty(state, 5.0)  # stays non-zero throughout

    w = PostFillTimeToFlatWatcher(settings, state, storage)
    w.start()
    w.track_fill("ttf-timeout", position_qty_before_fill=2.0)
    time.sleep(0.4)  # exceeds cap
    w.stop()
    w._thread.join(timeout=1.0)

    with storage.connection() as conn:
        row = conn.execute(
            "SELECT time_to_flat_seconds FROM fills WHERE fill_id='ttf-timeout'"
        ).fetchone()
    assert row[0] is None
    # Record should have been dropped from the watcher's in-flight set.
    assert "ttf-timeout" not in w._records
