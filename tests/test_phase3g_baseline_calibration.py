"""Phase 3G (v1.4.185) — per-version baseline calibration.

Three layers:

1. **Metric extraction** — `extract_session_metrics` correctly
   computes the headline numbers from the snapshot dicts, returns
   None when data is missing.

2. **File I/O + FIFO** — `load_baseline` round-trips with
   `write_baseline`; `upsert_baseline_entry` deduplicates by
   session_id and caps to `max_sessions`.

3. **Delta math** — `compute_deltas` returns the right mean / stdev /
   z-score per metric, and `flagged` fires only when there's enough
   baseline AND |z| ≥ threshold.

Plus a renderer smoke test (markdown + HTML emit without error in
all the documented edge cases: empty baseline, partial baseline,
full baseline with flagged regression).
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pytest

from tools.postmortem.baseline import (
    BASELINE_SCHEMA_VERSION,
    BaselineConfig,
    BaselineEntry,
    BaselineFile,
    MetricDelta,
    SessionMetrics,
    asdict_metrics,
    compute_deltas,
    extract_session_metrics,
    load_baseline,
    upsert_baseline_entry,
    write_baseline,
)
from tools.postmortem.sections.baseline_deltas import (
    render_html_section,
    render_markdown_section,
)


def _tmp_snapshots() -> Path:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_phase3g_{os.getpid()}_{uuid.uuid4().hex}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


class _FillsList(list):
    """Minimal stand-in for ``snap.fills`` — list of dicts that
    advertises ``.empty`` like a pandas DataFrame would."""

    @property
    def empty(self) -> bool:
        return len(self) == 0


def test_extract_metrics_fill_rate_per_hour() -> None:
    """Duration 3600 s + 12 fills → 12.0 fills/h."""
    m = extract_session_metrics(
        fills=_FillsList([]),
        session_summary={
            "duration_seconds": 3600.0,
            "key_counters": {"session_fill_count": 12},
        },
        state_current={},
        pnl_current={},
    )
    assert m.fill_rate_per_hour == pytest.approx(12.0)


def test_extract_metrics_mean_markout_needs_5_fills() -> None:
    """< 5 fills with markout_5s_bps → None (insufficient power)."""
    fills = _FillsList(
        [{"markout_5s_bps": -2.0} for _ in range(4)]
    )
    m = extract_session_metrics(
        fills=fills,
        session_summary={"duration_seconds": 600.0},
        state_current={},
        pnl_current={},
    )
    assert m.mean_markout_5s_bps is None
    # 5+ fills → returns mean.
    fills = _FillsList(
        [{"markout_5s_bps": -2.0} for _ in range(10)]
    )
    m = extract_session_metrics(
        fills=fills,
        session_summary={"duration_seconds": 600.0},
        state_current={},
        pnl_current={},
    )
    assert m.mean_markout_5s_bps == pytest.approx(-2.0)


def test_extract_metrics_adverse_fill_pct() -> None:
    """Fraction of fills with mae_30s_bps < -5.0."""
    fills = _FillsList(
        [{"mae_30s_bps": -3.0}] * 5  # not adverse
        + [{"mae_30s_bps": -8.0}] * 5  # adverse
    )
    m = extract_session_metrics(
        fills=fills,
        session_summary={"duration_seconds": 600.0},
        state_current={},
        pnl_current={},
    )
    assert m.adverse_fill_pct == pytest.approx(0.5)


def test_extract_metrics_edge_bps_per_min() -> None:
    """Net PnL / starting equity / minutes * 10_000."""
    m = extract_session_metrics(
        fills=_FillsList([]),
        session_summary={"duration_seconds": 600.0},  # 10 min
        state_current={},
        pnl_current={
            "realized_pnl_usd": 5.0,
            "unrealized_pnl_usd": 0.0,
            "fees_usd": -1.0,  # fees stored as negative
            "equity_usd": 1000.0,
        },
    )
    # net = 5.0 + 0 + (-1.0) = 4.0; start_eq = 1000 - 4 = 996
    # bps = 4 / 996 * 10000 = 40.16; per minute = 40.16/10 = 4.016
    assert m.edge_bps_per_min == pytest.approx(4.016, rel=1e-2)


def test_extract_metrics_time_in_modes() -> None:
    m = extract_session_metrics(
        fills=_FillsList([]),
        session_summary={"duration_seconds": 600.0},
        state_current={
            "behavioural_gates": {
                "regime_mode": {
                    "time_in_defensive_seconds": 120.0,
                    "time_in_shock_seconds": 30.0,
                }
            }
        },
        pnl_current={},
    )
    assert m.time_in_defensive_seconds == pytest.approx(120.0)
    assert m.time_in_shock_seconds == pytest.approx(30.0)


def test_extract_metrics_all_none_on_empty_inputs() -> None:
    m = extract_session_metrics(
        fills=_FillsList([]),
        session_summary={},
        state_current={},
        pnl_current={},
    )
    assert m.fill_rate_per_hour is None
    assert m.mean_markout_5s_bps is None
    assert m.adverse_fill_pct is None
    assert m.edge_bps_per_min is None


# ---------------------------------------------------------------------------
# File I/O + FIFO
# ---------------------------------------------------------------------------


def _entry(sid: str, mins_ago: int, fill_rate: float) -> BaselineEntry:
    from datetime import datetime, timedelta, timezone

    started = datetime.now(timezone.utc) - timedelta(minutes=mins_ago)
    return BaselineEntry(
        ts_captured_utc=started.isoformat(),
        ts_session_started_utc=started.isoformat(),
        duration_seconds=600.0,
        bot_version="1.4.184",
        session_id=sid,
        metrics={
            "fill_rate_per_hour": fill_rate,
            "mean_markout_5s_bps": -1.0,
            "adverse_fill_pct": 0.3,
            "edge_bps_per_min": 0.1,
            "time_in_defensive_seconds": 60.0,
            "time_in_shock_seconds": 10.0,
        },
    )


def test_write_then_load_round_trip() -> None:
    root = _tmp_snapshots()
    bf = BaselineFile(profile="test.profile", sessions=[_entry("s1", 60, 25.0)])
    write_baseline(bf, snapshots_root=root)
    loaded = load_baseline(snapshots_root=root, profile="test.profile")
    assert loaded is not None
    assert loaded.profile == "test.profile"
    assert loaded.schema_version == BASELINE_SCHEMA_VERSION
    assert len(loaded.sessions) == 1
    assert loaded.sessions[0].session_id == "s1"
    assert loaded.sessions[0].metrics["fill_rate_per_hour"] == 25.0


def test_load_missing_file_returns_none() -> None:
    root = _tmp_snapshots()
    out = load_baseline(snapshots_root=root, profile="never.existed")
    assert out is None


def test_load_profile_mismatch_returns_none() -> None:
    """Defensive: file with wrong profile name doesn't contaminate."""
    root = _tmp_snapshots()
    bf = BaselineFile(profile="profile.a", sessions=[_entry("s1", 60, 25.0)])
    write_baseline(bf, snapshots_root=root)
    # Read with a different profile name.
    out = load_baseline(snapshots_root=root, profile="profile.b")
    assert out is None


def test_upsert_replaces_same_session_id_in_place() -> None:
    """Re-running the postmortem on the same snapshot must not
    double-count. The entry with matching session_id is REPLACED."""
    bf = BaselineFile(
        profile="test.profile",
        sessions=[
            _entry("s1", 60, 25.0),
            _entry("s2", 50, 30.0),
        ],
    )
    cfg = BaselineConfig(max_sessions=10)
    # Replace s2 with new metrics.
    new = _entry("s2", 50, 99.0)
    out = upsert_baseline_entry(bf, new, config=cfg)
    assert len(out.sessions) == 2
    sids = [e.session_id for e in out.sessions]
    assert sids == ["s1", "s2"]
    # The metrics for s2 should be the NEW ones.
    s2 = next(e for e in out.sessions if e.session_id == "s2")
    assert s2.metrics["fill_rate_per_hour"] == 99.0


def test_upsert_caps_to_max_sessions_fifo() -> None:
    """Adding past the cap drops the OLDEST."""
    bf = BaselineFile(profile="test.profile", sessions=[])
    cfg = BaselineConfig(max_sessions=3)
    # Add 5 sessions in chronological order.
    for i, mins_ago in enumerate([100, 80, 60, 40, 20]):
        bf = upsert_baseline_entry(
            bf,
            _entry(f"s{i}", mins_ago, 10.0 + i),
            config=cfg,
        )
    # Should keep only the 3 most recent (s2, s3, s4).
    assert len(bf.sessions) == 3
    sids = [e.session_id for e in bf.sessions]
    assert sids == ["s2", "s3", "s4"]


# ---------------------------------------------------------------------------
# Delta math
# ---------------------------------------------------------------------------


def _baseline_with(fill_rates: list[float]) -> BaselineFile:
    sessions = [
        _entry(f"s{i}", 100 - i * 10, fr)
        for i, fr in enumerate(fill_rates)
    ]
    return BaselineFile(profile="test.profile", sessions=sessions)


def test_compute_deltas_returns_one_row_per_metric() -> None:
    current = SessionMetrics(
        fill_rate_per_hour=30.0,
        mean_markout_5s_bps=-1.0,
        adverse_fill_pct=0.3,
        edge_bps_per_min=0.1,
        time_in_defensive_seconds=60.0,
        time_in_shock_seconds=10.0,
    )
    bf = _baseline_with([25.0, 27.0, 30.0, 28.0, 26.0])
    cfg = BaselineConfig()
    deltas = compute_deltas(current_metrics=current, baseline=bf, config=cfg)
    names = [d.name for d in deltas]
    assert names == [
        "fill_rate_per_hour",
        "mean_markout_5s_bps",
        "adverse_fill_pct",
        "edge_bps_per_min",
        "time_in_defensive_seconds",
        "time_in_shock_seconds",
    ]


def test_compute_deltas_z_score_correct_when_baseline_sufficient() -> None:
    """5 prior sessions with fill_rate {20, 22, 24, 26, 28} (mean=24,
    stdev≈3.16). Current = 30 → delta=+6, z=+1.9."""
    current = SessionMetrics(
        fill_rate_per_hour=30.0,
        mean_markout_5s_bps=None,
        adverse_fill_pct=None,
        edge_bps_per_min=None,
        time_in_defensive_seconds=None,
        time_in_shock_seconds=None,
    )
    bf = _baseline_with([20.0, 22.0, 24.0, 26.0, 28.0])
    cfg = BaselineConfig(sigma_flag_threshold=1.0)
    deltas = compute_deltas(current_metrics=current, baseline=bf, config=cfg)
    fr = next(d for d in deltas if d.name == "fill_rate_per_hour")
    assert fr.baseline_mean == pytest.approx(24.0)
    assert fr.baseline_stdev == pytest.approx(3.162, rel=1e-2)
    assert fr.delta == pytest.approx(6.0)
    assert fr.z_score == pytest.approx(1.897, rel=1e-2)
    # |z| ≈ 1.9 > threshold 1.0 → flagged.
    assert fr.flagged is True
    # Higher fill rate = better → favourable.
    assert fr.higher_is_better is True


def test_compute_deltas_below_min_baseline_does_not_flag() -> None:
    """With only 2 prior sessions (< min=3), the z-score and flag
    must not fire — the stdev isn't trustworthy enough to decide."""
    current = SessionMetrics(
        fill_rate_per_hour=100.0,  # huge delta
        mean_markout_5s_bps=None,
        adverse_fill_pct=None,
        edge_bps_per_min=None,
        time_in_defensive_seconds=None,
        time_in_shock_seconds=None,
    )
    bf = _baseline_with([20.0, 22.0])
    cfg = BaselineConfig(
        sigma_flag_threshold=1.0, min_baseline_sessions_for_stats=3
    )
    deltas = compute_deltas(current_metrics=current, baseline=bf, config=cfg)
    fr = next(d for d in deltas if d.name == "fill_rate_per_hour")
    assert fr.flagged is False
    # Mean is still computed (we have ≥1 prior) — just no z.
    assert fr.baseline_mean == pytest.approx(21.0)
    assert fr.z_score is None


def test_compute_deltas_handles_none_current() -> None:
    """Current metric is None (e.g., < 5 fills for markout). Delta
    + z-score are None; flag is False (can't compare what isn't
    there)."""
    current = SessionMetrics(
        fill_rate_per_hour=None,
        mean_markout_5s_bps=None,
        adverse_fill_pct=None,
        edge_bps_per_min=None,
        time_in_defensive_seconds=None,
        time_in_shock_seconds=None,
    )
    bf = _baseline_with([20.0, 22.0, 24.0, 26.0])
    cfg = BaselineConfig()
    deltas = compute_deltas(current_metrics=current, baseline=bf, config=cfg)
    fr = next(d for d in deltas if d.name == "fill_rate_per_hour")
    assert fr.current is None
    assert fr.delta is None
    assert fr.z_score is None
    assert fr.flagged is False


def test_compute_deltas_empty_baseline_returns_only_current() -> None:
    """First-ever session: no baseline file yet. Every row has a
    ``current`` value but no mean/stdev/delta/z."""
    current = SessionMetrics(
        fill_rate_per_hour=30.0,
        mean_markout_5s_bps=None,
        adverse_fill_pct=None,
        edge_bps_per_min=None,
        time_in_defensive_seconds=None,
        time_in_shock_seconds=None,
    )
    cfg = BaselineConfig()
    deltas = compute_deltas(
        current_metrics=current, baseline=None, config=cfg
    )
    for d in deltas:
        assert d.baseline_mean is None
        assert d.baseline_stdev is None
        assert d.n_baseline == 0
        assert d.flagged is False


def test_v1_4_176_regression_would_have_flagged() -> None:
    """The v1.4.176 disaster (16/h → 1.7/h) had a clear baseline.
    Confirm Phase 3G would have flagged it as regressed."""
    current = SessionMetrics(
        fill_rate_per_hour=1.7,
        mean_markout_5s_bps=None,
        adverse_fill_pct=None,
        edge_bps_per_min=None,
        time_in_defensive_seconds=None,
        time_in_shock_seconds=None,
    )
    # 5 prior sessions at the v1.4.92 baseline range.
    bf = _baseline_with([45.0, 42.0, 48.0, 44.0, 46.0])
    cfg = BaselineConfig(sigma_flag_threshold=1.0)
    deltas = compute_deltas(
        current_metrics=current, baseline=bf, config=cfg
    )
    fr = next(d for d in deltas if d.name == "fill_rate_per_hour")
    assert fr.flagged is True
    # delta = 1.7 - 45 = -43.3, very negative → regressed (BAD on a
    # higher-is-better metric).
    assert fr.delta < -40
    assert fr.higher_is_better is True
    # Verify the renderer's verdict logic agrees this is "regressed".
    md = render_markdown_section(
        current_metrics=current,
        deltas=deltas,
        config=cfg,
        bot_version="1.4.176",
        snapshot_name="snap-test",
    )
    assert "regressed" in md
    assert "regression flagged" in md


# ---------------------------------------------------------------------------
# Renderer smoke
# ---------------------------------------------------------------------------


def test_render_markdown_empty_baseline_shows_seed_note() -> None:
    """First-ever session: render an explanatory note, not a table."""
    current = SessionMetrics(
        fill_rate_per_hour=30.0,
        mean_markout_5s_bps=None,
        adverse_fill_pct=None,
        edge_bps_per_min=None,
        time_in_defensive_seconds=None,
        time_in_shock_seconds=None,
    )
    cfg = BaselineConfig()
    deltas = compute_deltas(
        current_metrics=current, baseline=None, config=cfg
    )
    md = render_markdown_section(
        current_metrics=current,
        deltas=deltas,
        config=cfg,
        bot_version="1.4.185",
        snapshot_name="snap-test",
    )
    assert "First recorded session" in md
    assert "rolling baseline" in md


def test_render_html_renders_when_baseline_present() -> None:
    current = SessionMetrics(
        fill_rate_per_hour=30.0,
        mean_markout_5s_bps=None,
        adverse_fill_pct=None,
        edge_bps_per_min=None,
        time_in_defensive_seconds=None,
        time_in_shock_seconds=None,
    )
    bf = _baseline_with([28.0, 32.0, 31.0, 29.0])
    cfg = BaselineConfig()
    deltas = compute_deltas(
        current_metrics=current, baseline=bf, config=cfg
    )
    html = render_html_section(
        current_metrics=current,
        deltas=deltas,
        config=cfg,
        bot_version="1.4.185",
        snapshot_name="snap-test",
    )
    assert "<section class='baseline-deltas'>" in html
    assert "Fill rate" in html
