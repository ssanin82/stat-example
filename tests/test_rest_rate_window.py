"""v1.4.4 — ``RestRateWindow`` sliding-2s gauge for the dashboard.

Tests the bucket arithmetic + peak-history decay. Critical for the
gauge's correctness because:

  * Operator reads the gauge as "REST/2s pressure right now" — if
    it under-counts due to lost increments under thread contention,
    the operator gets a false sense of headroom.
  * The peak metric carries the operator's "did we get close in the
    last minute" question — if the decay logic strands old peaks,
    every gauge reading shows stale alarm.

Time control: we patch ``time.monotonic`` to drive bucket boundaries
deterministically (the production code uses 100ms buckets so a real-
time test would be flaky on slow CI).

v1.4.20 rate-limit-observability Phase 1: extended with per-pool
tracking tests covering ``RateLimitPool`` enum, ``pool_for_endpoint``
mapping, and ``snapshot_per_pool`` surface.
"""

from __future__ import annotations

from unittest.mock import patch

from app.exchange.okx_client import (
    POOL_CAPS_2S,
    RateLimitPool,
    RestRateWindow,
    pool_for_endpoint,
)


def test_initial_state_is_zero() -> None:
    w = RestRateWindow()
    snap = w.snapshot()
    assert snap["current_2s_rate"] == 0
    assert snap["peak_2s_in_last_60s"] == 0
    assert snap["total_recorded"] == 0


def test_records_accumulate_within_one_bucket() -> None:
    w = RestRateWindow()
    with patch("app.exchange.okx_client.time.monotonic", return_value=1.0):
        for _ in range(5):
            w.record()
        snap = w.snapshot()
    assert snap["current_2s_rate"] == 5
    assert snap["total_recorded"] == 5


def test_window_sums_across_buckets() -> None:
    """Spread records across 10 distinct 100ms buckets; rolling-2s
    should still see all 10 because 10 × 100ms = 1s < 2s window."""
    w = RestRateWindow()
    base = 100.0
    for i in range(10):
        # Each call lands in a fresh 100ms bucket (0.1s apart).
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + i * 0.1,
        ):
            w.record()
    # Snapshot at 0.9s after last record — still within the 2s window.
    with patch(
        "app.exchange.okx_client.time.monotonic",
        return_value=base + 9 * 0.1 + 0.05,
    ):
        snap = w.snapshot()
    assert snap["current_2s_rate"] == 10
    assert snap["total_recorded"] == 10


def test_old_records_decay_after_two_seconds() -> None:
    """A record placed >2s before the snapshot should have decayed
    out of the current_2s_rate. The peak history still remembers it
    until 60s elapse."""
    w = RestRateWindow()
    base = 100.0
    # 3 records at t=0
    with patch("app.exchange.okx_client.time.monotonic", return_value=base):
        for _ in range(3):
            w.record()
    # Snapshot at t=2.5s — well past the 2s window.
    with patch(
        "app.exchange.okx_client.time.monotonic", return_value=base + 2.5
    ):
        snap = w.snapshot()
    assert snap["current_2s_rate"] == 0
    # Peak still remembers (the 3 fired within one bucket; rolling
    # sum at the moment of the second-bucket roll captured them).
    assert snap["peak_2s_in_last_60s"] >= 3
    assert snap["total_recorded"] == 3


def test_peak_decays_after_sixty_seconds() -> None:
    """Peaks older than 60s should fall off so the gauge doesn't show
    stale alarm. After a long idle period, peak should drop to 0
    (matching the current rate)."""
    w = RestRateWindow()
    base = 100.0
    with patch("app.exchange.okx_client.time.monotonic", return_value=base):
        for _ in range(50):
            w.record()
    # Snapshot at +30s — peak should still see the burst.
    with patch(
        "app.exchange.okx_client.time.monotonic", return_value=base + 30.0
    ):
        snap = w.snapshot()
    assert snap["peak_2s_in_last_60s"] >= 50
    assert snap["current_2s_rate"] == 0
    # Snapshot at +90s — well past the 60s peak window.
    with patch(
        "app.exchange.okx_client.time.monotonic", return_value=base + 90.0
    ):
        snap = w.snapshot()
    assert snap["current_2s_rate"] == 0
    # Peak should have fully decayed.
    assert snap["peak_2s_in_last_60s"] == 0


def test_total_recorded_is_monotonic() -> None:
    """``total_recorded`` is a cumulative cross-check counter — sums
    every record() call, never decays. Lets the operator notice
    bucket arithmetic bugs (e.g. lost increments under threading)."""
    w = RestRateWindow()
    base = 100.0
    for i in range(7):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + i * 0.5,
        ):
            w.record()
    with patch(
        "app.exchange.okx_client.time.monotonic", return_value=base + 7 * 0.5
    ):
        snap = w.snapshot()
    assert snap["total_recorded"] == 7


# ----------------------------------------------------------------------
# v1.4.20 — per-pool tracking (rate-limit-observability Phase 1)
# ----------------------------------------------------------------------


def test_pool_for_endpoint_maps_paths_correctly() -> None:
    """The mapper must distinguish each OKX trade endpoint. The
    substring-trap is ``/trade/order`` being a prefix-fragment of
    other paths — we check the more-specific paths first."""
    assert pool_for_endpoint(path="/api/v5/trade/order") == RateLimitPool.PLACE_SINGLE
    assert pool_for_endpoint(path="/api/v5/trade/batch-orders") == RateLimitPool.PLACE_BATCH
    assert pool_for_endpoint(path="/api/v5/trade/cancel-order") == RateLimitPool.CANCEL_SINGLE
    assert pool_for_endpoint(path="/api/v5/trade/cancel-batch-orders") == RateLimitPool.CANCEL_BATCH
    assert pool_for_endpoint(path="/api/v5/trade/amend-order") == RateLimitPool.AMEND_SINGLE
    assert pool_for_endpoint(path="/api/v5/trade/amend-batch-orders") == RateLimitPool.AMEND_BATCH
    # Reads
    assert pool_for_endpoint(path="/api/v5/trade/orders-pending") == RateLimitPool.READS
    assert pool_for_endpoint(path="/api/v5/trade/fills") == RateLimitPool.READS
    assert pool_for_endpoint(path="/api/v5/account/balance") == RateLimitPool.READS
    # Unclassified
    assert pool_for_endpoint(path="/api/v5/public/instruments") == RateLimitPool.OTHER
    assert pool_for_endpoint() == RateLimitPool.OTHER


def test_pool_for_endpoint_maps_ws_ops_correctly() -> None:
    """WS op variants — what the action-WS coordinator sends."""
    assert pool_for_endpoint(op="order") == RateLimitPool.PLACE_SINGLE
    assert pool_for_endpoint(op="batch-orders") == RateLimitPool.PLACE_BATCH
    assert pool_for_endpoint(op="cancel-order") == RateLimitPool.CANCEL_SINGLE
    assert pool_for_endpoint(op="batch-cancel-orders") == RateLimitPool.CANCEL_BATCH
    assert pool_for_endpoint(op="amend-order") == RateLimitPool.AMEND_SINGLE
    assert pool_for_endpoint(op="batch-amend-orders") == RateLimitPool.AMEND_BATCH


def test_pool_caps_match_okx_documented_values() -> None:
    """Sanity-pin the caps to the OKX-published values for the the partner
    MM tier. If OKX changes these we want to know — bumping the
    bot's adaptive throttle without bumping the cap is a real bug
    risk."""
    assert POOL_CAPS_2S[RateLimitPool.PLACE_SINGLE] == 60
    assert POOL_CAPS_2S[RateLimitPool.PLACE_BATCH] == 300
    assert POOL_CAPS_2S[RateLimitPool.CANCEL_SINGLE] == 60
    assert POOL_CAPS_2S[RateLimitPool.CANCEL_BATCH] == 300
    assert POOL_CAPS_2S[RateLimitPool.AMEND_BATCH] == 300
    assert POOL_CAPS_2S[RateLimitPool.READS] == 20
    # OTHER has no cap — absent from the dict.
    assert RateLimitPool.OTHER not in POOL_CAPS_2S


def test_per_pool_record_segregates_pools() -> None:
    """A record on PLACE_BATCH must NOT increment CANCEL_BATCH's
    counters and vice versa. Aggregate sees both."""
    w = RestRateWindow()
    with patch("app.exchange.okx_client.time.monotonic", return_value=1.0):
        for _ in range(7):
            w.record(pool=RateLimitPool.PLACE_BATCH)
        for _ in range(3):
            w.record(pool=RateLimitPool.CANCEL_BATCH)
        snap = w.snapshot_per_pool()
    assert snap["place_batch"]["current_2s"] == 7
    assert snap["place_batch"]["total"] == 7
    assert snap["cancel_batch"]["current_2s"] == 3
    assert snap["cancel_batch"]["total"] == 3
    # Aggregate is the sum.
    assert snap["aggregate"]["current_2s"] == 10
    assert snap["aggregate"]["total"] == 10
    # Pools not touched are absent.
    assert "amend_batch" not in snap


def test_per_pool_pct_of_cap_computed() -> None:
    """The pct_of_cap field lets the dashboard colour-code each
    pool's pressure. 100 records into PLACE_BATCH (cap 300) →
    33%."""
    w = RestRateWindow()
    with patch("app.exchange.okx_client.time.monotonic", return_value=1.0):
        for _ in range(100):
            w.record(pool=RateLimitPool.PLACE_BATCH)
        snap = w.snapshot_per_pool()
    pb = snap["place_batch"]
    assert pb["current_2s"] == 100
    assert pb["cap"] == 300
    # peak is at least current, so pct = peak/cap ≥ 100/300
    assert pb["pct_of_cap"] is not None
    assert abs(pb["pct_of_cap"] - 100 / 300) < 0.01


def test_per_pool_default_is_other_pool() -> None:
    """Calls without an explicit pool land in OTHER. ``cap`` is None
    so dashboard renders OTHER as informational only."""
    w = RestRateWindow()
    with patch("app.exchange.okx_client.time.monotonic", return_value=1.0):
        w.record()  # no pool kwarg
        snap = w.snapshot_per_pool()
    assert "other" in snap
    assert snap["other"]["current_2s"] == 1
    assert snap["other"]["cap"] is None
    assert snap["other"]["pct_of_cap"] is None


def test_per_pool_aggregate_window_unchanged_by_pool_split() -> None:
    """Back-compat: ``snapshot()`` (legacy aggregate) returns the
    same numbers as before regardless of which pool each record
    used. Existing dashboard consumers reading
    ``okx_rate_window_current_2s`` keep working."""
    w = RestRateWindow()
    with patch("app.exchange.okx_client.time.monotonic", return_value=1.0):
        w.record(pool=RateLimitPool.PLACE_BATCH)
        w.record(pool=RateLimitPool.CANCEL_BATCH)
        w.record(pool=RateLimitPool.AMEND_BATCH)
        legacy = w.snapshot()
    assert legacy["current_2s_rate"] == 3
    assert legacy["total_recorded"] == 3


def test_per_sec_stats_absent_with_fewer_than_two_samples() -> None:
    """v1.4.22 — per-second min/max/median are None until 2+
    finalized 1-second samples exist. A single sample's median is
    just itself; we want distribution shape."""
    w = RestRateWindow()
    with patch("app.exchange.okx_client.time.monotonic", return_value=1.0):
        # 5 records all in the first second — current second is not
        # yet finalized.
        for _ in range(5):
            w.record(pool=RateLimitPool.PLACE_BATCH)
        snap = w.snapshot_per_pool()
    pb = snap["place_batch"]
    assert pb["rate_per_sec_samples"] == 0
    assert pb["rate_per_sec_min"] is None
    assert pb["rate_per_sec_median"] is None
    assert pb["rate_per_sec_p95"] is None


def test_per_sec_stats_compute_after_three_seconds() -> None:
    """v1.4.22 — record at distinct second boundaries; the prior
    seconds finalize as they're crossed. After 3 seconds of records
    (10/sec, 20/sec, 30/sec), the stats reflect the distribution
    of the FIRST TWO seconds (the third is still accumulating)."""
    w = RestRateWindow()
    base = 100.0
    # Second 1: 10 records.
    for i in range(10):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + 0.0 + i * 0.05,
        ):
            w.record(pool=RateLimitPool.PLACE_BATCH)
    # Second 2: 20 records.
    for i in range(20):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + 1.0 + i * 0.04,
        ):
            w.record(pool=RateLimitPool.PLACE_BATCH)
    # Second 3: 30 records (still in progress when we snapshot).
    for i in range(30):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + 2.0 + i * 0.02,
        ):
            w.record(pool=RateLimitPool.PLACE_BATCH)
    with patch(
        "app.exchange.okx_client.time.monotonic",
        return_value=base + 2.5,
    ):
        snap = w.snapshot_per_pool()
    pb = snap["place_batch"]
    # Two finalized samples: 10 and 20. Third (30) is mid-flight.
    assert pb["rate_per_sec_samples"] == 2
    assert pb["rate_per_sec_min"] == 10
    assert pb["rate_per_sec_max"] == 20
    assert pb["rate_per_sec_median"] == 15.0  # mean of [10, 20]
    # p95 = linear interp of sorted [10, 20] at rank 0.95: 10 + (20-10)*0.95 = 19.5
    assert abs(pb["rate_per_sec_p95"] - 19.5) < 0.01


def test_per_sec_stats_capped_at_60_samples() -> None:
    """v1.4.22 — deque is bounded at 60 (1-minute window). After
    >60 finalized seconds, only the most recent 60 are retained."""
    w = RestRateWindow()
    base = 100.0
    # 90 distinct seconds, 1 record each — but only 60 should be
    # retained in the history deque.
    for sec in range(90):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + sec + 0.1,
        ):
            w.record(pool=RateLimitPool.PLACE_BATCH)
    # Snapshot mid-second 91.
    with patch(
        "app.exchange.okx_client.time.monotonic",
        return_value=base + 90.5,
    ):
        snap = w.snapshot_per_pool()
    pb = snap["place_batch"]
    # 60 finalized samples (seconds 30-89; the rest evicted; sec 90
    # is mid-flight, not yet finalized).
    assert pb["rate_per_sec_samples"] == 60
    assert pb["rate_per_sec_min"] == 1
    assert pb["rate_per_sec_max"] == 1
    assert pb["rate_per_sec_median"] == 1.0


def test_per_sec_idle_gap_pads_zeros() -> None:
    """v1.4.22 — when there's an idle gap between recordings, the
    intervening seconds should be padded with 0 samples so the
    distribution correctly reflects "the bot did nothing during
    that time" (rate_per_sec_min should drop to 0)."""
    w = RestRateWindow()
    base = 100.0
    # Burst of 5 records in second 1 (sec=100000).
    for i in range(5):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + 0.0 + i * 0.05,
        ):
            w.record(pool=RateLimitPool.PLACE_BATCH)
    # 10-second idle gap — no records.
    # One record in second 11 (sec=111000).
    with patch(
        "app.exchange.okx_client.time.monotonic",
        return_value=base + 11.0,
    ):
        w.record(pool=RateLimitPool.PLACE_BATCH)
    # Snapshot at base+12.5 (sec=112000); snapshot-side advance
    # finalizes second 11's count=1.
    with patch(
        "app.exchange.okx_client.time.monotonic",
        return_value=base + 12.5,
    ):
        snap = w.snapshot_per_pool()
    pb = snap["place_batch"]
    # Finalized samples timeline:
    #   sec 0 (base+0.0..0.2)  → 5 records
    #   secs 1-10              → 0 each (10 zeros padded on the
    #                            second-11 record's finalize step)
    #   sec 11 (base+11.0)     → 1 record (finalized by snapshot
    #                            advance at sec 12)
    # Total: 12 samples. Sec 12 is the snapshot's current second
    # and stays accumulating (count=0 — no records there).
    assert pb["rate_per_sec_samples"] == 12
    assert pb["rate_per_sec_min"] == 0
    assert pb["rate_per_sec_max"] == 5
    # 12 samples sorted = [0×10, 1, 5]; median = mean(samples[5], samples[6]) = 0
    assert pb["rate_per_sec_median"] == 0.0


def test_per_sec_stats_per_pool_isolated() -> None:
    """v1.4.22 — separate pools have their own 1Hz histories.
    A burst on PLACE_BATCH must not bleed into CANCEL_BATCH stats."""
    w = RestRateWindow()
    base = 100.0
    # Second 1: 30 records on PLACE_BATCH, 5 on CANCEL_BATCH.
    for i in range(30):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + i * 0.01,
        ):
            w.record(pool=RateLimitPool.PLACE_BATCH)
    for i in range(5):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + 0.5 + i * 0.05,
        ):
            w.record(pool=RateLimitPool.CANCEL_BATCH)
    # Second 2: 10 records on PLACE_BATCH only.
    for i in range(10):
        with patch(
            "app.exchange.okx_client.time.monotonic",
            return_value=base + 1.0 + i * 0.05,
        ):
            w.record(pool=RateLimitPool.PLACE_BATCH)
    # Snapshot mid second 3.
    with patch(
        "app.exchange.okx_client.time.monotonic",
        return_value=base + 2.5,
    ):
        snap = w.snapshot_per_pool()
    # PLACE_BATCH: 2 finalized samples [30, 10].
    pb = snap["place_batch"]
    assert pb["rate_per_sec_samples"] == 2
    assert pb["rate_per_sec_min"] == 10
    assert pb["rate_per_sec_max"] == 30
    # CANCEL_BATCH: 2 finalized samples [5, 0]. (Second 2 had no
    # cancels → pad zero so the deque stays time-aligned.)
    cb = snap["cancel_batch"]
    assert cb["rate_per_sec_samples"] == 2
    assert cb["rate_per_sec_min"] == 0
    assert cb["rate_per_sec_max"] == 5


def test_per_pool_decay_after_two_seconds() -> None:
    """Per-pool current_2s decays after 2s same as aggregate."""
    w = RestRateWindow()
    base = 100.0
    with patch("app.exchange.okx_client.time.monotonic", return_value=base):
        for _ in range(5):
            w.record(pool=RateLimitPool.AMEND_BATCH)
    with patch(
        "app.exchange.okx_client.time.monotonic", return_value=base + 2.5
    ):
        snap = w.snapshot_per_pool()
    assert snap["amend_batch"]["current_2s"] == 0
    assert snap["amend_batch"]["total"] == 5
    # Peak history still remembers (within 60s window).
    assert snap["amend_batch"]["peak_2s_60s"] >= 5


def test_long_gap_then_burst() -> None:
    """A long idle period (>2s, so the window stales) followed by a
    fresh burst should report the burst correctly — exercises the
    "diff_buckets >= _NUM_BUCKETS" branch where the whole ring is
    cleared in one go."""
    w = RestRateWindow()
    base = 100.0
    # Initial small activity.
    with patch("app.exchange.okx_client.time.monotonic", return_value=base):
        w.record()
    # Long idle — 10s gap.
    with patch(
        "app.exchange.okx_client.time.monotonic", return_value=base + 10.0
    ):
        # Single fresh record — should be the only thing in the window.
        w.record()
        snap = w.snapshot()
    assert snap["current_2s_rate"] == 1
    assert snap["total_recorded"] == 2
