"""Regression tests for the regime-observability Phase 2 exposure bars.

Covers:
- Schema migration v27: ``exposure_bars`` table exists with the
  expected columns + indexes.
- ``Storage.insert_exposure_bar`` round-trip through SQLite.
- ``Storage.exposure_bars_since`` filtering by ts_bar / symbol /
  limit.
- ``ExposureBarEmitter.maybe_create``: disabled by config flag,
  disabled when storage absent, enabled when both present.
- ``ExposureBarEmitter._capture_bar`` produces a row dict with the
  expected keys.
- Lifecycle: start() → stop() → thread is joinable.

Hot-path safety is verified indirectly: the emitter only reads
state attributes (no lock acquisition on ``state._lock``), and the
captured row dict's keys are all schema-defined.

See ``plans/regime-observability.md`` Phase 2.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from app.exposure_bar_emitter import ExposureBarEmitter
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _make_storage() -> Storage:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_p2_{os.getpid()}_{uuid.uuid4().hex}.db"
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


def _make_settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "SYMBOL": "TON-USDT-SWAP",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


# --------------------------------------------------------------------- #
# Schema v27.                                                           #
# --------------------------------------------------------------------- #


def test_schema_v27_exposure_bars_table_exists() -> None:
    s = _make_storage()
    with s.connection() as conn:
        tbls = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "exposure_bars" in tbls


def test_schema_v27_exposure_bars_columns_exist() -> None:
    s = _make_storage()
    with s.connection() as conn:
        cols = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(exposure_bars)"
            ).fetchall()
        }
    expected = {
        "session_id",
        "ts_bar",
        "symbol",
        "mid",
        "spread_bps",
        "microprice",
        "imbalance_top",
        "bid_size_top",
        "ask_size_top",
        "inventory_qty",
        "inventory_utilization",
        "active_sides",
        "quote_eligibility",
        "toxicity_score",
        "vol_estimate",
        "binance_basis_ewma",
        "basis_regime_sign",
        "bid_live",
        "ask_live",
        "bid_distance_ticks",
        "ask_distance_ticks",
        "quoted_spread_bps",
        "adaptive_widen_active",
        "hold_all_active",
        "recovery_cooldown_active",
        "post_fill_cooldown_active_bid",
        "post_fill_cooldown_active_ask",
        "at_touch_adverse_pause_bid",
        "at_touch_adverse_pause_ask",
    }
    missing = expected - cols
    assert not missing, f"Missing exposure_bars columns: {missing}"


def test_schema_v27_exposure_bars_indexes_exist() -> None:
    s = _make_storage()
    with s.connection() as conn:
        idxs = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
    assert "idx_exposure_bars_ts" in idxs
    assert "idx_exposure_bars_session" in idxs


# --------------------------------------------------------------------- #
# Storage round-trip.                                                   #
# --------------------------------------------------------------------- #


def _make_bar(**overrides) -> dict:
    base = dict(
        session_id="sess-1",
        ts_bar="2026-05-13T12:00:00+00:00",
        symbol="TON-USDT-SWAP",
        mid=2.30,
        spread_bps=4.3,
        microprice=2.301,
        imbalance_top=0.1,
        bid_size_top=100.0,
        ask_size_top=80.0,
        inventory_qty=3.0,
        inventory_utilization=0.375,
        active_sides="BOTH",
        quote_eligibility="QUOTE_BOTH",
        toxicity_score=0.2,
        vol_estimate=12.5,
        binance_basis_ewma=0.0004,
        basis_regime_sign=-1,
        bid_live=1,
        ask_live=1,
        bid_distance_ticks=1.0,
        ask_distance_ticks=1.0,
        quoted_spread_bps=8.0,
        adaptive_widen_active=0,
        hold_all_active=0,
        recovery_cooldown_active=0,
        post_fill_cooldown_active_bid=None,
        post_fill_cooldown_active_ask=None,
        at_touch_adverse_pause_bid=None,
        at_touch_adverse_pause_ask=None,
    )
    base.update(overrides)
    return base


def test_insert_exposure_bar_roundtrip() -> None:
    s = _make_storage()
    row = _make_bar()
    s.insert_exposure_bar(row)
    out = s.exposure_bars_since("2026-05-13T00:00:00+00:00")
    assert len(out) == 1
    assert out[0]["mid"] == 2.30
    assert out[0]["inventory_qty"] == 3.0
    assert out[0]["active_sides"] == "BOTH"
    assert out[0]["basis_regime_sign"] == -1
    assert out[0]["bid_live"] == 1
    assert out[0]["adaptive_widen_active"] == 0


def test_exposure_bars_since_filters_by_window() -> None:
    s = _make_storage()
    s.insert_exposure_bar(_make_bar(ts_bar="2026-05-13T11:00:00+00:00"))
    s.insert_exposure_bar(_make_bar(ts_bar="2026-05-13T12:00:00+00:00"))
    s.insert_exposure_bar(_make_bar(ts_bar="2026-05-13T13:00:00+00:00"))
    # since-only.
    out = s.exposure_bars_since("2026-05-13T12:00:00+00:00")
    assert len(out) == 2
    # since + until.
    out = s.exposure_bars_since(
        "2026-05-13T11:30:00+00:00",
        until_iso="2026-05-13T12:30:00+00:00",
    )
    assert len(out) == 1
    assert out[0]["ts_bar"].startswith("2026-05-13T12:00")


def test_exposure_bars_since_filters_by_symbol() -> None:
    s = _make_storage()
    s.insert_exposure_bar(_make_bar(symbol="TON-USDT-SWAP"))
    s.insert_exposure_bar(_make_bar(symbol="SOL-USDT-SWAP"))
    s.insert_exposure_bar(_make_bar(symbol=None))  # symbol-agnostic row
    out = s.exposure_bars_since(
        "2026-05-13T00:00:00+00:00", symbol="TON-USDT-SWAP"
    )
    # Symbol-matching row + symbol-NULL row both pass the filter.
    assert len(out) == 2


def test_exposure_bars_since_respects_limit() -> None:
    s = _make_storage()
    # Insert 5 rows with increasing ts.
    for i in range(5):
        s.insert_exposure_bar(
            _make_bar(ts_bar=f"2026-05-13T12:0{i}:00+00:00")
        )
    out = s.exposure_bars_since(
        "2026-05-13T00:00:00+00:00", limit=3
    )
    assert len(out) == 3
    # Oldest-first ordering.
    assert out[0]["ts_bar"] < out[1]["ts_bar"] < out[2]["ts_bar"]


# --------------------------------------------------------------------- #
# Emitter factory + lifecycle.                                          #
# --------------------------------------------------------------------- #


def test_emitter_disabled_when_flag_off() -> None:
    settings = _make_settings(OBSERVABILITY_EXPOSURE_BARS_ENABLED=False)
    state = BotState(settings)
    storage = _make_storage()
    em = ExposureBarEmitter.maybe_create(settings, state, storage)
    assert em is None


def test_emitter_disabled_when_storage_absent() -> None:
    settings = _make_settings()
    state = BotState(settings)
    em = ExposureBarEmitter.maybe_create(settings, state, storage=None)
    assert em is None


def test_emitter_enabled_when_flag_and_storage_present() -> None:
    settings = _make_settings()
    state = BotState(settings)
    storage = _make_storage()
    em = ExposureBarEmitter.maybe_create(settings, state, storage)
    assert em is not None


def test_emitter_capture_bar_produces_row_dict() -> None:
    settings = _make_settings()
    state = BotState(settings)
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    row = em._capture_bar()
    # First call may return None if state.market isn't populated yet.
    # That's by design — the emitter tolerates pre-warmup. Inject a
    # market snapshot and try again.
    if row is None:
        from app.models import BestBidAsk

        state.market = BestBidAsk(
            symbol="TON-USDT-SWAP",
            best_bid=2.299,
            best_ask=2.301,
            mid_price=2.300,
            spread_bps=8.7,
            bid_size=100.0,
            ask_size=80.0,
        )
        row = em._capture_bar()
    assert row is not None
    # All schema columns are present as keys.
    expected_keys = {
        "session_id",
        "ts_bar",
        "symbol",
        "mid",
        "spread_bps",
        "microprice",
        "imbalance_top",
        "bid_size_top",
        "ask_size_top",
        "inventory_qty",
        "inventory_utilization",
        "active_sides",
        "quote_eligibility",
        "toxicity_score",
        "vol_estimate",
        "binance_basis_ewma",
        "basis_regime_sign",
        "bid_live",
        "ask_live",
        "bid_distance_ticks",
        "ask_distance_ticks",
        "quoted_spread_bps",
        "adaptive_widen_active",
        "hold_all_active",
        "recovery_cooldown_active",
        "post_fill_cooldown_active_bid",
        "post_fill_cooldown_active_ask",
        "at_touch_adverse_pause_bid",
        "at_touch_adverse_pause_ask",
    }
    missing = expected_keys - set(row.keys())
    assert not missing, f"Captured row missing keys: {missing}"
    # Market fields populated from the injected market.
    assert row["mid"] == 2.300
    assert abs(row["spread_bps"] - (0.002 / 2.300 * 10_000.0)) < 1e-6


def test_emitter_start_stop_cleanly() -> None:
    """The emitter is a daemon thread; ``stop()`` sets the event and
    the loop exits on the next tick. Verify the thread becomes
    non-alive shortly after stop()."""
    settings = _make_settings(
        OBSERVABILITY_EXPOSURE_BAR_INTERVAL_SECONDS=0.1
    )
    state = BotState(settings)
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    em.start()
    # Give it a moment to start.
    time.sleep(0.05)
    assert em._thread is not None and em._thread.is_alive()
    em.stop()
    # The stop event causes ``_loop`` to exit on the next ``wait``
    # cycle (interval 0.1s). Give it up to 1s to join.
    em._thread.join(timeout=1.0)
    assert not em._thread.is_alive()


def test_emitter_writes_bar_to_storage_via_loop() -> None:
    """End-to-end: start emitter with very short interval, verify
    at least one row lands in the exposure_bars table."""
    settings = _make_settings(
        OBSERVABILITY_EXPOSURE_BAR_INTERVAL_SECONDS=0.05
    )
    state = BotState(settings)
    # Inject a market snapshot so _capture_bar produces a non-None row.
    from app.models import BestBidAsk

    state.market = BestBidAsk(
        symbol="TON-USDT-SWAP",
        best_bid=2.299,
        best_ask=2.301,
        mid_price=2.300,
        spread_bps=8.7,
        bid_size=100.0,
        ask_size=80.0,
    )
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    em.start()
    # Wait for at least 2-3 emit ticks.
    time.sleep(0.25)
    em.stop()
    em._thread.join(timeout=1.0) if em._thread else None
    # Query the table.
    rows = storage.exposure_bars_since("2026-05-13T00:00:00+00:00")
    assert len(rows) >= 1
    assert rows[0]["mid"] == 2.300
