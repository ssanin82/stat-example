"""PersistentRuntimeState JSON persistence: round-trip, corrupt file, version mismatch."""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

from app.persistent_runtime_state import (
    PERSISTENT_RUNTIME_STATE_VERSION,
    PersistentRuntimeState,
    load_persistent_runtime_state,
    save_persistent_runtime_state,
)


def _sample_state() -> PersistentRuntimeState:
    return PersistentRuntimeState(
        day_anchor_utc=date(2026, 4, 11),
        daily_realized_pnl=-12.5,
        daily_trade_count=42,
        daily_traded_notional=15_000.25,
        last_fill_ts=datetime(2026, 4, 11, 14, 30, 0, tzinfo=timezone.utc),
        recent_buy_fill_count=20,
        recent_sell_fill_count=22,
        rolling_toxicity_markout_bps=-1.25,
        rolling_one_sided_fill_ratio=0.62,
    )


def test_round_trip_save_load() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "operator_metrics.json"
        original = _sample_state()
        save_persistent_runtime_state(path, original)
        loaded = load_persistent_runtime_state(path)
        assert loaded == original
        text = path.read_text(encoding="utf-8")
        outer = json.loads(text)
        assert outer["version"] == PERSISTENT_RUNTIME_STATE_VERSION
        assert "data" in outer


def test_round_trip_null_optionals() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "m.json"
        s = PersistentRuntimeState(
            day_anchor_utc=date(2026, 1, 2),
            daily_realized_pnl=0.0,
            daily_trade_count=0,
            daily_traded_notional=0.0,
            last_fill_ts=None,
            recent_buy_fill_count=0,
            recent_sell_fill_count=0,
            rolling_toxicity_markout_bps=None,
            rolling_one_sided_fill_ratio=None,
        )
        save_persistent_runtime_state(path, s)
        loaded = load_persistent_runtime_state(path)
        assert loaded == s


def test_missing_file_returns_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "nope.json"
        assert load_persistent_runtime_state(path) is None


def test_corrupt_json_returns_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "bad.json"
        path.write_text("not json {{{", encoding="utf-8")
        assert load_persistent_runtime_state(path) is None


def test_wrong_version_returns_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "old.json"
        path.write_text(
            json.dumps(
                {
                    "version": PERSISTENT_RUNTIME_STATE_VERSION + 99,
                    "data": {"day_anchor_utc": "2026-04-11"},
                }
            ),
            encoding="utf-8",
        )
        assert load_persistent_runtime_state(path) is None


def test_missing_or_invalid_data_returns_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        p1 = Path(td) / "a.json"
        p1.write_text(json.dumps({"version": PERSISTENT_RUNTIME_STATE_VERSION}), encoding="utf-8")
        assert load_persistent_runtime_state(p1) is None

        p2 = Path(td) / "b.json"
        p2.write_text(
            json.dumps({"version": PERSISTENT_RUNTIME_STATE_VERSION, "data": "not_a_dict"}),
            encoding="utf-8",
        )
        assert load_persistent_runtime_state(p2) is None


def test_invalid_payload_fields_returns_none() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "inv.json"
        path.write_text(
            json.dumps(
                {
                    "version": PERSISTENT_RUNTIME_STATE_VERSION,
                    "data": {
                        "day_anchor_utc": "not-a-date",
                        "daily_realized_pnl": 0.0,
                        "daily_trade_count": 0,
                        "daily_traded_notional": 0.0,
                        "recent_buy_fill_count": 0,
                        "recent_sell_fill_count": 0,
                    },
                }
            ),
            encoding="utf-8",
        )
        assert load_persistent_runtime_state(path) is None


def test_overwrite_existing_file() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "metrics.json"
        save_persistent_runtime_state(path, _sample_state())
        s2 = PersistentRuntimeState(
            day_anchor_utc=date(2026, 4, 12),
            daily_realized_pnl=1.0,
            daily_trade_count=1,
            daily_traded_notional=1.0,
            recent_buy_fill_count=1,
            recent_sell_fill_count=0,
        )
        save_persistent_runtime_state(path, s2)
        loaded = load_persistent_runtime_state(path)
        assert loaded == s2
