"""Tests for the 1.3.85 multi-snapshot postmortem merge.

The operator wanted to span a session that was disrupted by a redeploy
across two snapshots — fills from the cut-off session + fills from the
fresh session — without losing context. ``load_and_merge_snapshots``
concatenates the time-series and uses the newest snapshot's dict fields.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import pytest

from tools.postmortem.loaders.snapshot_loader import (
    load_and_merge_snapshots,
    load_snapshot,
)


def _write_snapshot(
    root: Path,
    name: str,
    *,
    bot_version: str,
    captured_at_utc: str,
    fills: list[dict],
    events: list[dict],
    orders: list[dict] | None = None,
    session_started_at_utc: str = "2026-05-15T00:00:00+00:00",
    config: dict | None = None,
) -> Path:
    snap_dir = root / name
    stats = snap_dir / "stats"
    stats.mkdir(parents=True, exist_ok=True)
    (snap_dir / "meta.json").write_text(
        json.dumps(
            {
                "bot_version": bot_version,
                "captured_at_utc": captured_at_utc,
                "bot_profile": "test",
            }
        )
    )
    (stats / "fills_since.json").write_text(json.dumps(fills))
    (stats / "events_since.json").write_text(json.dumps(events))
    (stats / "orders_since.json").write_text(json.dumps(orders or []))
    (stats / "state_current.json").write_text(
        json.dumps({"bot_version": bot_version})
    )
    (stats / "pnl_current.json").write_text(
        json.dumps({"realized_pnl_usd": -0.1, "ts": captured_at_utc})
    )
    (stats / "pnl_attribution.json").write_text(json.dumps({}))
    (stats / "session_summary.json").write_text(
        json.dumps(
            {
                "session_started_at_utc": session_started_at_utc,
                "now_utc": captured_at_utc,
            }
        )
    )
    (stats / "config.json").write_text(
        json.dumps(config if config is not None else {"version": bot_version})
    )
    # Empty time series for the rest:
    for empty_name in (
        "inventory_since.json",
        "equity_since.json",
        "quotes_since.json",
    ):
        (stats / empty_name).write_text(json.dumps([]))
    return snap_dir


def test_load_and_merge_two_snapshots_concatenates_fills(tmp_path: Path) -> None:
    """Two snapshots with 2 fills each → merged result has 4 fills,
    sorted by ts_fill."""
    snap1 = _write_snapshot(
        tmp_path,
        "snap_a_260515-100000",
        bot_version="1.3.83",
        captured_at_utc="2026-05-15T10:00:00+00:00",
        fills=[
            {"ts_fill": "2026-05-15T09:00:00+00:00", "side": "BUY", "price": 1.0, "size": 1.0},
            {"ts_fill": "2026-05-15T09:30:00+00:00", "side": "SELL", "price": 1.1, "size": 1.0},
        ],
        events=[
            {"ts": "2026-05-15T09:00:00+00:00", "event_type": "bot_start", "severity": "INFO", "message": "old session", "payload_json": "{}"},
        ],
    )
    snap2 = _write_snapshot(
        tmp_path,
        "snap_b_260515-110000",
        bot_version="1.3.84",
        captured_at_utc="2026-05-15T11:00:00+00:00",
        fills=[
            {"ts_fill": "2026-05-15T10:30:00+00:00", "side": "BUY", "price": 1.05, "size": 1.0},
            {"ts_fill": "2026-05-15T10:45:00+00:00", "side": "SELL", "price": 1.06, "size": 1.0},
        ],
        events=[
            {"ts": "2026-05-15T10:00:00+00:00", "event_type": "bot_start", "severity": "INFO", "message": "new session", "payload_json": "{}"},
        ],
    )

    merged = load_and_merge_snapshots([snap1, snap2])
    assert len(merged.fills) == 4
    # Sorted chronologically.
    ts_list = merged.fills["ts_fill"].tolist()
    assert all(
        str(ts_list[i]) <= str(ts_list[i + 1]) for i in range(len(ts_list) - 1)
    )
    # Newest meta is the merged meta.
    assert merged.meta["bot_version"] == "1.3.84"
    # merged_from index labels both inputs.
    assert "merged_from" in merged.meta
    assert len(merged.meta["merged_from"]) == 2
    versions = sorted(m["bot_version"] for m in merged.meta["merged_from"])
    assert versions == ["1.3.83", "1.3.84"]


def test_load_and_merge_preserves_newest_dict_fields(tmp_path: Path) -> None:
    """Dict fields take the newest snapshot's value, regardless of
    insertion order."""
    snap_old = _write_snapshot(
        tmp_path,
        "snap_old_260515-100000",
        bot_version="1.3.83",
        captured_at_utc="2026-05-15T10:00:00+00:00",
        fills=[],
        events=[],
    )
    snap_new = _write_snapshot(
        tmp_path,
        "snap_new_260515-110000",
        bot_version="1.3.84",
        captured_at_utc="2026-05-15T11:00:00+00:00",
        fills=[],
        events=[],
    )
    # Pass in reverse chronological order to test the sort.
    merged = load_and_merge_snapshots([snap_new, snap_old])
    assert merged.meta["bot_version"] == "1.3.84"
    assert merged.config["version"] == "1.3.84"


def test_load_and_merge_single_snapshot(tmp_path: Path) -> None:
    """N=1 should still work (no-op merge)."""
    snap = _write_snapshot(
        tmp_path,
        "snap_only_260515-100000",
        bot_version="1.3.83",
        captured_at_utc="2026-05-15T10:00:00+00:00",
        fills=[{"ts_fill": "2026-05-15T09:00:00+00:00", "side": "BUY", "price": 1.0, "size": 1.0}],
        events=[],
    )
    merged = load_and_merge_snapshots([snap])
    assert len(merged.fills) == 1
    assert merged.meta["bot_version"] == "1.3.83"
    assert len(merged.meta["merged_from"]) == 1


def test_load_and_merge_rejects_empty_input() -> None:
    """Empty input list is operator error — raise."""
    with pytest.raises(ValueError):
        load_and_merge_snapshots([])


def test_find_n_newest_snapshots(tmp_path: Path) -> None:
    """The helper picks the N most-recent snapshots in chronological order."""
    from tools.postmortem.__main__ import _find_n_newest_snapshots

    # Write three snapshots with distinct timestamps.
    _write_snapshot(
        tmp_path,
        "snap_a_260514-100000",
        bot_version="v1",
        captured_at_utc="2026-05-14T10:00:00+00:00",
        fills=[], events=[],
    )
    _write_snapshot(
        tmp_path,
        "snap_b_260515-100000",
        bot_version="v2",
        captured_at_utc="2026-05-15T10:00:00+00:00",
        fills=[], events=[],
    )
    _write_snapshot(
        tmp_path,
        "snap_c_260515-180000",
        bot_version="v3",
        captured_at_utc="2026-05-15T18:00:00+00:00",
        fills=[], events=[],
    )
    found = _find_n_newest_snapshots(2, snapshots_dir=tmp_path)
    assert len(found) == 2
    # Oldest-first: snap_b then snap_c.
    assert found[0].name == "snap_b_260515-100000"
    assert found[1].name == "snap_c_260515-180000"

    # N larger than available → return what's there.
    found_all = _find_n_newest_snapshots(10, snapshots_dir=tmp_path)
    assert len(found_all) == 3

    # N=0 → empty list.
    assert _find_n_newest_snapshots(0, snapshots_dir=tmp_path) == []


def test_merge_detects_quoting_config_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """When two snapshots have different values for a quoting-relevant
    config knob, the merger annotates meta and prints a stdout warning."""
    snap1 = _write_snapshot(
        tmp_path,
        "snap_x_260515-100000",
        bot_version="1.3.83",
        captured_at_utc="2026-05-15T10:00:00+00:00",
        fills=[],
        events=[],
        config={
            "BASE_HALF_SPREAD_BPS": 2.0,
            "INVENTORY_SKEW_COEFF_BPS": 12,
            "MOMENTUM_GATE_ENABLED": True,
        },
    )
    snap2 = _write_snapshot(
        tmp_path,
        "snap_y_260515-110000",
        bot_version="1.3.84",
        captured_at_utc="2026-05-15T11:00:00+00:00",
        fills=[],
        events=[],
        config={
            # BASE_HALF_SPREAD_BPS bumped, INVENTORY_SKEW_COEFF_BPS lowered.
            "BASE_HALF_SPREAD_BPS": 2.5,
            "INVENTORY_SKEW_COEFF_BPS": 8,
            "MOMENTUM_GATE_ENABLED": True,  # unchanged
        },
    )
    merged = load_and_merge_snapshots([snap1, snap2])
    assert merged.meta["quoting_config_drift_detected"] is True
    assert "BASE_HALF_SPREAD_BPS" in merged.meta["quoting_config_drift_keys"]
    assert (
        "INVENTORY_SKEW_COEFF_BPS" in merged.meta["quoting_config_drift_keys"]
    )
    # MOMENTUM_GATE_ENABLED unchanged → not in drift keys.
    assert (
        "MOMENTUM_GATE_ENABLED" not in merged.meta["quoting_config_drift_keys"]
    )

    # The older snapshot's merged_from entry carries the diff.
    older_entry = next(
        e for e in merged.meta["merged_from"]
        if e["bot_version"] == "1.3.83"
    )
    assert "BASE_HALF_SPREAD_BPS" in older_entry["quoting_config_diff_vs_newest"]
    a, b = older_entry["quoting_config_diff_vs_newest"]["BASE_HALF_SPREAD_BPS"]
    assert a == 2.5 and b == 2.0  # (newest, older)

    # Stdout carries the WARNING banner.
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "BASE_HALF_SPREAD_BPS" in out


def test_merge_passes_when_quoting_config_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """Identical configs across snapshots → drift_detected False, no
    warning printed."""
    same_config = {
        "BASE_HALF_SPREAD_BPS": 2.5,
        "INVENTORY_SKEW_COEFF_BPS": 8,
    }
    snap1 = _write_snapshot(
        tmp_path,
        "snap_p_260515-100000",
        bot_version="1.3.84",
        captured_at_utc="2026-05-15T10:00:00+00:00",
        fills=[],
        events=[],
        config=same_config,
    )
    snap2 = _write_snapshot(
        tmp_path,
        "snap_q_260515-110000",
        bot_version="1.3.84",
        captured_at_utc="2026-05-15T11:00:00+00:00",
        fills=[],
        events=[],
        config=same_config,
    )
    merged = load_and_merge_snapshots([snap1, snap2])
    assert merged.meta["quoting_config_drift_detected"] is False
    assert merged.meta["quoting_config_drift_keys"] == []
    out = capsys.readouterr().out
    assert "WARNING" not in out
    assert "ok" in out  # "merged config check ok:" success line


def test_drift_normalises_string_vs_typed_values(tmp_path: Path) -> None:
    """Env-loader sometimes round-trips numerics as strings.
    ``"2.5"`` should equal ``2.5`` for drift purposes — we don't want
    false positives on serialization artifacts."""
    snap1 = _write_snapshot(
        tmp_path,
        "snap_s_260515-100000",
        bot_version="1.3.84",
        captured_at_utc="2026-05-15T10:00:00+00:00",
        fills=[],
        events=[],
        config={"BASE_HALF_SPREAD_BPS": "2.5", "MOMENTUM_GATE_ENABLED": "true"},
    )
    snap2 = _write_snapshot(
        tmp_path,
        "snap_t_260515-110000",
        bot_version="1.3.84",
        captured_at_utc="2026-05-15T11:00:00+00:00",
        fills=[],
        events=[],
        config={"BASE_HALF_SPREAD_BPS": 2.5, "MOMENTUM_GATE_ENABLED": True},
    )
    merged = load_and_merge_snapshots([snap1, snap2])
    assert merged.meta["quoting_config_drift_detected"] is False
