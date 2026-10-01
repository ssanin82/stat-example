"""v1.5.268 — fixes to scripts/snapshot_acceptance.py.

Three pre-existing acceptance-script bugs caught by the
v1.5.266-260529-190358 snapshot:

* (b) ``check_v1_5_205_stale_quote_p99_shifted_left`` used hardcoded
      2.4 s / 3.5 s thresholds anchored to v1.5.200's
      AT_TOUCH_MAX_AGE_SECONDS=2.5. After v1.5.264 raised the cap to
      5.0 s for fill-rate calibration, the check started false-
      FAILing on every snapshot. Rewritten to dynamically anchor on
      AT_TOUCH_MAX_AGE_SECONDS from config.json.

* (c1) ``check_v1_5_231_regime_transition_rate_sane`` accessed
       ``snap.events_since`` but ``SnapshotData`` didn't declare the
       field. Result: AttributeError raised on every snapshot.
       Added ``events_since: Optional[list[dict[str, Any]]]`` to the
       dataclass + loader.

* (c2) ``check_v1_5_231_cautious_quote_notional_mult_safe`` does
       ``from app.regime_controller import _KNOBS_CAUTIOUS`` but the
       script's sys.path didn't include the repo root, so the
       import always failed with "No module named 'app'". Added an
       explicit ``sys.path.insert`` near the top of the script.

These tests exercise the structural fixes — that the script CAN run
each check without raising, and that the dynamic-threshold check
returns the right verdict for known input shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))


def _write_snapshot(tmp_path: Path, *,
                    bot_version: str = "1.5.268",
                    config: dict | None = None,
                    fills: list[dict] | None = None,
                    events: list[dict] | None = None,
                    session_summary: dict | None = None,
                    state_current: dict | None = None) -> Path:
    """Create a minimal snapshot dir with the bits we want to test."""
    snap_dir = tmp_path / f"v{bot_version}-260529-200000-prod.test"
    stats = snap_dir / "stats"
    stats.mkdir(parents=True)
    (snap_dir / "meta.json").write_text(json.dumps({
        "bot_version": bot_version,
        "captured_at_utc": "2026-05-29T20:00:00Z",
    }))
    if config is not None:
        (stats / "config.json").write_text(json.dumps(config))
    if fills is not None:
        (stats / "fills_since.json").write_text(json.dumps(fills))
    if events is not None:
        (stats / "events_since.json").write_text(json.dumps(events))
    if session_summary is not None:
        (stats / "session_summary.json").write_text(json.dumps(session_summary))
    if state_current is not None:
        (stats / "state_current.json").write_text(json.dumps(state_current))
    return snap_dir


# ---------------------------------------------------------------------------
# (c1) SnapshotData carries events_since
# ---------------------------------------------------------------------------

def test_snapshot_data_loads_events_since(tmp_path):
    """events_since.json is loaded into SnapshotData.events_since."""
    from scripts.snapshot_acceptance import _load_snapshot

    events = [
        {"event_type": "regime_mode_transition", "ts": "2026-05-29T20:00:01Z"},
        {"event_type": "fill", "ts": "2026-05-29T20:00:02Z"},
    ]
    snap_dir = _write_snapshot(tmp_path, events=events)
    snap = _load_snapshot(snap_dir)
    assert snap.events_since == events


def test_snapshot_data_events_since_falls_back_to_events_recent(tmp_path):
    """If events_since.json missing, fall back to events_recent.json."""
    from scripts.snapshot_acceptance import _load_snapshot

    snap_dir = _write_snapshot(tmp_path)  # no events file
    recent = [{"event_type": "shock_gate_armed"}]
    (snap_dir / "stats" / "events_recent.json").write_text(json.dumps(recent))
    snap = _load_snapshot(snap_dir)
    assert snap.events_since == recent


def test_snapshot_data_events_since_is_none_when_both_missing(tmp_path):
    """When both events files are absent, events_since == None (N/A path)."""
    from scripts.snapshot_acceptance import _load_snapshot

    snap_dir = _write_snapshot(tmp_path)
    snap = _load_snapshot(snap_dir)
    assert snap.events_since is None


# ---------------------------------------------------------------------------
# (c1) regime-transition-rate check actually runs
# ---------------------------------------------------------------------------

def test_regime_transition_rate_check_runs_without_attributeerror(tmp_path):
    """The check should NEVER raise AttributeError on snap.events_since."""
    from scripts.snapshot_acceptance import (
        _load_snapshot, check_v1_5_231_regime_transition_rate_sane,
    )

    # Healthy session: 2 transitions in 30 min = 0.067/min → PASS.
    events = [
        {"event_type": "regime_mode_transition", "ts": f"2026-05-29T20:{i:02d}:00Z"}
        for i in range(2)
    ]
    snap_dir = _write_snapshot(
        tmp_path,
        events=events,
        session_summary={"duration_seconds": 1800.0},
    )
    snap = _load_snapshot(snap_dir)
    result = check_v1_5_231_regime_transition_rate_sane(snap)
    assert result.status == "PASS"
    assert result.measured["transitions"] == 2
    assert abs(result.measured["rate_per_min"] - 0.067) < 0.01


def test_regime_transition_rate_check_flags_flapping(tmp_path):
    """High transition rate → FAIL with actionable message."""
    from scripts.snapshot_acceptance import (
        _load_snapshot, check_v1_5_231_regime_transition_rate_sane,
    )

    # 100 transitions in 30 min = 3.33/min → FAIL (> 2.0).
    events = [
        {"event_type": "regime_mode_transition"} for _ in range(100)
    ]
    snap_dir = _write_snapshot(
        tmp_path,
        events=events,
        session_summary={"duration_seconds": 1800.0},
    )
    snap = _load_snapshot(snap_dir)
    result = check_v1_5_231_regime_transition_rate_sane(snap)
    assert result.status == "FAIL"
    assert "flapping" in result.detail


# ---------------------------------------------------------------------------
# (c2) cautious-mult check can import _KNOBS_CAUTIOUS
# ---------------------------------------------------------------------------

def test_cautious_mult_check_can_import_knobs(tmp_path):
    """The script's sys.path now includes the repo root so the
    `from app.regime_controller import _KNOBS_CAUTIOUS` works."""
    from scripts.snapshot_acceptance import (
        _load_snapshot, check_v1_5_231_cautious_quote_notional_mult_safe,
    )

    snap_dir = _write_snapshot(
        tmp_path,
        config={
            "QUOTE_NOTIONAL_USD": 7.0,
            "MIN_QUOTE_NOTIONAL_USD": 5.0,
        },
    )
    snap = _load_snapshot(snap_dir)
    result = check_v1_5_231_cautious_quote_notional_mult_safe(snap)
    # Whatever the verdict, it must NOT be the import-error
    # INSUFFICIENT_DATA from before — that one had the substring
    # "could not import _KNOBS_CAUTIOUS".
    assert "could not import" not in result.detail


# ---------------------------------------------------------------------------
# (b) stale-quote check uses dynamic AT_TOUCH_MAX_AGE_SECONDS threshold
# ---------------------------------------------------------------------------

def _fills_with_age(p50_ms: float, p99_ms: float, n: int = 50) -> list[dict]:
    """Construct n fake fills whose quote_age_at_fill_ms distribution
    has the requested p50 and p99 (approximately)."""
    # Simple shape: 99% of fills clustered at p50, 1% at p99.
    fills = []
    n_high = max(1, n // 100)
    n_low = n - n_high
    for _ in range(n_low):
        fills.append({"quote_age_at_fill_ms": p50_ms})
    for _ in range(n_high):
        fills.append({"quote_age_at_fill_ms": p99_ms})
    return fills


def test_stale_quote_check_pass_with_low_p99_below_cap_70pct(tmp_path):
    """P99 < cap × 0.7 → PASS (penalty creating headroom)."""
    from scripts.snapshot_acceptance import (
        _load_snapshot, check_v1_5_205_stale_quote_p99_shifted_left,
    )

    snap_dir = _write_snapshot(
        tmp_path,
        config={"AT_TOUCH_MAX_AGE_SECONDS": 5.0},
        fills=_fills_with_age(p50_ms=500.0, p99_ms=2000.0, n=100),
    )
    snap = _load_snapshot(snap_dir)
    result = check_v1_5_205_stale_quote_p99_shifted_left(snap)
    assert result.status == "PASS", result.detail


def test_stale_quote_check_warn_at_cap(tmp_path):
    """P99 near the cap (within 0.7-1.05 of cap) → WARN."""
    from scripts.snapshot_acceptance import (
        _load_snapshot, check_v1_5_205_stale_quote_p99_shifted_left,
    )

    snap_dir = _write_snapshot(
        tmp_path,
        config={"AT_TOUCH_MAX_AGE_SECONDS": 5.0},
        fills=_fills_with_age(p50_ms=2500.0, p99_ms=4970.0, n=100),
    )
    snap = _load_snapshot(snap_dir)
    result = check_v1_5_205_stale_quote_p99_shifted_left(snap)
    assert result.status == "WARN", result.detail


def test_stale_quote_check_fail_beyond_cap_5pct(tmp_path):
    """P99 > cap × 1.05 → FAIL (orders outliving the cap is a real bug)."""
    from scripts.snapshot_acceptance import (
        _load_snapshot, check_v1_5_205_stale_quote_p99_shifted_left,
    )

    snap_dir = _write_snapshot(
        tmp_path,
        config={"AT_TOUCH_MAX_AGE_SECONDS": 5.0},
        fills=_fills_with_age(p50_ms=2500.0, p99_ms=8000.0, n=100),
    )
    snap = _load_snapshot(snap_dir)
    result = check_v1_5_205_stale_quote_p99_shifted_left(snap)
    assert result.status == "FAIL", result.detail


def test_stale_quote_check_insufficient_data_when_cap_missing(tmp_path):
    """No AT_TOUCH_MAX_AGE_SECONDS in config → INSUFFICIENT_DATA."""
    from scripts.snapshot_acceptance import (
        _load_snapshot, check_v1_5_205_stale_quote_p99_shifted_left,
    )

    snap_dir = _write_snapshot(
        tmp_path,
        config={},
        fills=_fills_with_age(p50_ms=500.0, p99_ms=2000.0, n=100),
    )
    snap = _load_snapshot(snap_dir)
    result = check_v1_5_205_stale_quote_p99_shifted_left(snap)
    assert result.status == "INSUFFICIENT_DATA"
    assert "AT_TOUCH_MAX_AGE_SECONDS" in result.detail


def test_stale_quote_check_adapts_to_old_cap(tmp_path):
    """Same fills that PASS under cap=5.0 also score correctly under
    cap=2.5 (= old baseline). Dynamic anchoring works both ways."""
    from scripts.snapshot_acceptance import (
        _load_snapshot, check_v1_5_205_stale_quote_p99_shifted_left,
    )

    # P99=2.0s under cap=5.0 → ratio 0.4 → PASS
    # Same P99=2.0s under cap=2.5 → ratio 0.8 → WARN
    fills = _fills_with_age(p50_ms=500.0, p99_ms=2000.0, n=100)
    snap_dir_new = _write_snapshot(
        tmp_path / "new",
        config={"AT_TOUCH_MAX_AGE_SECONDS": 5.0},
        fills=fills,
    )
    snap_dir_old = _write_snapshot(
        tmp_path / "old",
        config={"AT_TOUCH_MAX_AGE_SECONDS": 2.5},
        fills=fills,
    )
    snap_new = _load_snapshot(snap_dir_new)
    snap_old = _load_snapshot(snap_dir_old)
    result_new = check_v1_5_205_stale_quote_p99_shifted_left(snap_new)
    result_old = check_v1_5_205_stale_quote_p99_shifted_left(snap_old)
    assert result_new.status == "PASS"
    assert result_old.status == "WARN"
