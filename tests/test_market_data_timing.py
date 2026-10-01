from __future__ import annotations

from app.market_data_timing import PublicWsTimingTracker


def test_timing_tracker_ingest_missing_exchange_ts_and_gaps() -> None:
    tr = PublicWsTimingTracker(symbol="ETH", max_samples=10)
    tr.ingest(
        local_receive_wall_ms=1000,
        local_receive_mono_ns=1_000_000_000,
        exchange_ts_ms=None,
        local_apply_wall_ms=1001,
        local_apply_mono_ns=1_001_000_000,
    )
    tr.ingest(
        local_receive_wall_ms=1100,
        local_receive_mono_ns=2_000_000_000,
        exchange_ts_ms=900,
        local_apply_wall_ms=1101,
        local_apply_mono_ns=2_002_000_000,
    )
    s = tr.summary()
    assert s["current_buffer_size"] == 2
    assert s["missing_exchange_ts_count"] == 1
    assert s["samples_with_exchange_ts"] == 1

    rec = tr.recent_samples(limit=10)
    # newest-first
    assert rec[0].local_receive_wall_time_ms == 1100
    assert rec[0].local_receive_gap_ms == 100
    assert rec[0].exchange_to_local_receive_ms == 200.0
    assert rec[0].receive_to_apply_ms is not None


def test_timing_tracker_anomaly_counters_negative_exchange_gap_and_one_way() -> None:
    tr = PublicWsTimingTracker(symbol="ETH", max_samples=10)
    tr.ingest(
        local_receive_wall_ms=2000,
        local_receive_mono_ns=1,
        exchange_ts_ms=2500,  # one-way negative
        local_apply_wall_ms=2000,
        local_apply_mono_ns=2,
    )
    tr.ingest(
        local_receive_wall_ms=2100,
        local_receive_mono_ns=10,
        exchange_ts_ms=2400,  # exchange gap negative vs prior 2500
        local_apply_wall_ms=2100,
        local_apply_mono_ns=20,
    )
    s = tr.summary()
    assert s["negative_one_way_delay_count"] == 2
    assert s["invalid_exchange_gap_count"] == 1


def test_timing_tracker_rolling_eviction() -> None:
    tr = PublicWsTimingTracker(symbol="ETH", max_samples=3)
    for i in range(10):
        tr.ingest(
            local_receive_wall_ms=1000 + i,
            local_receive_mono_ns=100 + i,
            exchange_ts_ms=1000 + i,
            local_apply_wall_ms=1000 + i,
            local_apply_mono_ns=110 + i,
        )
    s = tr.summary()
    assert s["current_buffer_size"] == 3
    rec = tr.recent_samples(limit=10)
    assert rec[0].local_receive_wall_time_ms == 1009
    assert rec[-1].local_receive_wall_time_ms == 1007


def test_missing_receive_wall_keeps_one_way_and_receive_gap_missing() -> None:
    tr = PublicWsTimingTracker(symbol="ETH", max_samples=10)
    tr.ingest(
        local_receive_wall_ms=None,
        local_receive_mono_ns=1_000,
        exchange_ts_ms=1000,
        local_apply_wall_ms=2000,
        local_apply_mono_ns=2_000,
    )
    s = tr.summary()
    assert s["missing_local_receive_wall_count"] == 1
    rec = tr.recent_samples(limit=1)[0]
    assert rec.exchange_to_local_receive_ms is None
    assert rec.local_receive_gap_ms is None


def test_missing_receive_mono_does_not_use_zero_prefers_wall_fallback() -> None:
    tr = PublicWsTimingTracker(symbol="ETH", max_samples=10)
    tr.ingest(
        local_receive_wall_ms=1000,
        local_receive_mono_ns=None,
        exchange_ts_ms=900,
        local_apply_wall_ms=1010,
        local_apply_mono_ns=5_000,
    )
    rec = tr.recent_samples(limit=1)[0]
    # should fall back to wall-clock (both wall times present)
    assert rec.receive_to_apply_ms == 10.0
    s = tr.summary()
    assert s["missing_local_receive_mono_count"] == 1


def test_boundary_reset_prevents_gap_calculation_across_segments() -> None:
    tr = PublicWsTimingTracker(symbol="ETH", max_samples=10)
    tr.ingest(
        local_receive_wall_ms=1000,
        local_receive_mono_ns=100,
        exchange_ts_ms=1000,
        local_apply_wall_ms=1000,
        local_apply_mono_ns=110,
    )
    tr.note_stream_reset("test")
    tr.ingest(
        local_receive_wall_ms=1100,
        local_receive_mono_ns=200,
        exchange_ts_ms=1100,
        local_apply_wall_ms=1100,
        local_apply_mono_ns=210,
    )
    rec = tr.recent_samples(limit=2)
    # newest sample should not compute gaps vs prior segment
    assert rec[0].exchange_gap_ms is None
    assert rec[0].local_receive_gap_ms is None
    s = tr.summary()
    assert s["boundary_event_count"] == 1


def test_receive_to_apply_source_counts_and_boundary_metadata() -> None:
    tr = PublicWsTimingTracker(symbol="ETH", max_samples=10)
    tr.note_stream_reset("test_reason")
    # monotonic-backed
    tr.ingest(
        local_receive_wall_ms=1000,
        local_receive_mono_ns=1_000_000,
        exchange_ts_ms=900,
        local_apply_wall_ms=1002,
        local_apply_mono_ns=3_000_000,
    )
    # wall-fallback-backed
    tr.ingest(
        local_receive_wall_ms=1100,
        local_receive_mono_ns=None,
        exchange_ts_ms=1000,
        local_apply_wall_ms=1107,
        local_apply_mono_ns=10_000_000,
    )
    s = tr.summary()
    assert s["samples_with_valid_receive_to_apply"] == 2
    assert s["samples_with_valid_receive_to_apply_monotonic"] == 1
    assert s["samples_with_valid_receive_to_apply_wall_fallback"] == 1
    assert s["last_boundary_reason"] == "test_reason"
    assert s["last_boundary_wall_time_ms"] is not None
    assert s["last_boundary_wall_time_iso"] is not None
    assert "boundary_count_semantics" in s

