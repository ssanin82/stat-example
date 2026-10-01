"""Tests for ``app.backtest.event_stream`` — Phase 3 (v1.4.234)."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from app.backtest import RecordedEvent, SortedEventStream


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------


def _write_jsonl_gz(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d) + "\n")


def _synthetic_fixture(
    tmp_path: Path,
    *,
    okx_events: list[tuple[int, dict]],
    binance_events: list[tuple[int, dict]] | None = None,
    gaps: int = 0,
) -> Path:
    """Build a fixture with the given per-source events."""
    fixture = tmp_path / "synth"
    fixture.mkdir(parents=True, exist_ok=True)

    files = []
    okx_path = fixture / "okx_public.jsonl.gz"
    _write_jsonl_gz(
        okx_path,
        [{"t_recv_ns": t, "source": "okx_public", "msg": m} for t, m in okx_events],
    )
    files.append({
        "path": "okx_public.jsonl.gz",
        "source": "okx_public",
        "lines": len(okx_events),
        "first_t_recv_ns": okx_events[0][0] if okx_events else 0,
        "last_t_recv_ns": okx_events[-1][0] if okx_events else 0,
        "gaps": gaps,
    })

    if binance_events:
        b_path = fixture / "binance_public.jsonl.gz"
        _write_jsonl_gz(
            b_path,
            [{"t_recv_ns": t, "source": "binance_public", "msg": m}
             for t, m in binance_events],
        )
        files.append({
            "path": "binance_public.jsonl.gz",
            "source": "binance_public",
            "lines": len(binance_events),
            "first_t_recv_ns": binance_events[0][0],
            "last_t_recv_ns": binance_events[-1][0],
            "gaps": 0,
        })

    manifest = {
        "schema_version": 1,
        "session_id": "synth-test",
        "recorder_version": "0.1.0",
        "is_finalized": True,
        "files": files,
    }
    (fixture / "manifest.json").write_text(json.dumps(manifest))
    return fixture


# ---------------------------------------------------------------------------
# Manifest + structural tests
# ---------------------------------------------------------------------------


def test_missing_manifest_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no-manifest"
    bad.mkdir()
    with pytest.raises(FileNotFoundError):
        SortedEventStream(bad)


def test_missing_referenced_file_raises(tmp_path: Path) -> None:
    fixture = tmp_path / "broken"
    fixture.mkdir()
    (fixture / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "session_id": "x",
        "files": [{
            "path": "missing.jsonl.gz",
            "source": "okx_public",
            "first_t_recv_ns": 0,
            "last_t_recv_ns": 1,
            "lines": 0,
            "gaps": 0,
        }],
    }))
    with pytest.raises(FileNotFoundError):
        SortedEventStream(fixture)


def test_empty_sources_filter_raises(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(tmp_path, okx_events=[(1, {})])
    with pytest.raises(ValueError):
        SortedEventStream(fixture, sources={"nonexistent_source"})


def test_first_last_t_ns_from_manifest(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(
        tmp_path,
        okx_events=[(100, {"a": 1}), (200, {"a": 2}), (300, {"a": 3})],
    )
    s = SortedEventStream(fixture)
    assert s.first_t_ns == 100
    assert s.last_t_ns == 300


def test_total_lines_and_gaps_from_manifest(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(
        tmp_path,
        okx_events=[(100, {})],
        gaps=5,
    )
    # Gaps reported → must opt in via allow_gaps.
    with pytest.raises(ValueError):
        SortedEventStream(fixture)
    s = SortedEventStream(fixture, allow_gaps=True)
    assert s.total_gaps == 5
    assert s.total_lines == 1


def test_session_id_and_recorder_version_exposed(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(tmp_path, okx_events=[(100, {})])
    s = SortedEventStream(fixture)
    assert s.session_id == "synth-test"
    assert s.recorder_version == "0.1.0"
    assert s.schema_version == 1


# ---------------------------------------------------------------------------
# K-way merge ordering
# ---------------------------------------------------------------------------


def test_single_source_iteration(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(
        tmp_path,
        okx_events=[(100, {"i": 0}), (200, {"i": 1}), (300, {"i": 2})],
    )
    events = list(SortedEventStream(fixture))
    assert len(events) == 3
    assert [e.t_recv_ns for e in events] == [100, 200, 300]
    assert all(isinstance(e, RecordedEvent) for e in events)


def test_two_sources_interleaved_in_time_order(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(
        tmp_path,
        okx_events=[(100, {"v": "a"}), (300, {"v": "c"}), (500, {"v": "e"})],
        binance_events=[(200, {"v": "b"}), (400, {"v": "d"})],
    )
    events = list(SortedEventStream(fixture))
    assert [e.t_recv_ns for e in events] == [100, 200, 300, 400, 500]
    # Source labels preserved.
    assert events[0].source == "okx_public"
    assert events[1].source == "binance_public"


def test_tie_breaker_is_deterministic(tmp_path: Path) -> None:
    """Two events with same t_recv_ns from different sources — order
    must be stable across re-iterations."""
    fixture = _synthetic_fixture(
        tmp_path,
        okx_events=[(100, {"src": "okx"})],
        binance_events=[(100, {"src": "binance"})],
    )
    s = SortedEventStream(fixture)
    a = [(e.t_recv_ns, e.source) for e in s]
    b = [(e.t_recv_ns, e.source) for e in s]
    assert a == b  # Same order on the second iteration.


def test_iteration_is_repeatable(tmp_path: Path) -> None:
    """Re-iterating yields the full sequence again (no exhaustion)."""
    fixture = _synthetic_fixture(
        tmp_path,
        okx_events=[(t, {"i": i}) for i, t in enumerate(range(100, 1000, 100))],
    )
    s = SortedEventStream(fixture)
    a = list(s)
    b = list(s)
    assert len(a) == len(b) == 9


# ---------------------------------------------------------------------------
# Malformed lines are skipped, not raised
# ---------------------------------------------------------------------------


def test_malformed_json_line_skipped(tmp_path: Path) -> None:
    fixture = tmp_path / "malformed"
    fixture.mkdir()
    okx_path = fixture / "okx_public.jsonl.gz"
    with gzip.open(okx_path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"t_recv_ns": 100, "source": "okx_public", "msg": {}}) + "\n")
        f.write("not-json\n")
        f.write(json.dumps({"t_recv_ns": 300, "source": "okx_public", "msg": {}}) + "\n")
    (fixture / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "files": [{
            "path": "okx_public.jsonl.gz",
            "source": "okx_public",
            "lines": 3,
            "first_t_recv_ns": 100,
            "last_t_recv_ns": 300,
            "gaps": 0,
        }],
    }))
    events = list(SortedEventStream(fixture))
    # Malformed line dropped; the two valid ones survive.
    assert [e.t_recv_ns for e in events] == [100, 300]


def test_missing_required_fields_skipped(tmp_path: Path) -> None:
    fixture = tmp_path / "missing"
    fixture.mkdir()
    okx_path = fixture / "okx_public.jsonl.gz"
    with gzip.open(okx_path, "wt", encoding="utf-8") as f:
        # No t_recv_ns.
        f.write(json.dumps({"source": "okx_public", "msg": {}}) + "\n")
        # No source.
        f.write(json.dumps({"t_recv_ns": 100, "msg": {}}) + "\n")
        # Valid.
        f.write(json.dumps({"t_recv_ns": 200, "source": "okx_public", "msg": {}}) + "\n")
    (fixture / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "files": [{
            "path": "okx_public.jsonl.gz",
            "source": "okx_public",
            "lines": 3,
            "first_t_recv_ns": 200,
            "last_t_recv_ns": 200,
            "gaps": 0,
        }],
    }))
    events = list(SortedEventStream(fixture))
    assert [e.t_recv_ns for e in events] == [200]


# ---------------------------------------------------------------------------
# Source filter
# ---------------------------------------------------------------------------


def test_sources_filter_only_iterates_requested(tmp_path: Path) -> None:
    fixture = _synthetic_fixture(
        tmp_path,
        okx_events=[(100, {})],
        binance_events=[(150, {})],
    )
    s = SortedEventStream(fixture, sources={"binance_public"})
    events = list(s)
    assert len(events) == 1
    assert events[0].source == "binance_public"


# ---------------------------------------------------------------------------
# Real fixture (laptop-smoke-1)
# ---------------------------------------------------------------------------


_REAL_FIXTURE = Path(__file__).resolve().parent.parent / "backtesting/data/sessions/laptop-smoke-1"


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="laptop-smoke-1 fixture not present",
)
def test_real_fixture_loads_and_counts_match_manifest() -> None:
    s = SortedEventStream(_REAL_FIXTURE)
    expected_total = sum(int(e.get("lines", 0)) for e in s.file_metadata())
    count = sum(1 for _ in s)
    # All lines should round-trip (no malformed entries in production fixtures).
    assert count == expected_total


@pytest.mark.skipif(
    not _REAL_FIXTURE.exists(),
    reason="laptop-smoke-1 fixture not present",
)
def test_real_fixture_events_monotonically_non_decreasing() -> None:
    s = SortedEventStream(_REAL_FIXTURE)
    last = -1
    for e in s:
        assert e.t_recv_ns >= last
        last = e.t_recv_ns
