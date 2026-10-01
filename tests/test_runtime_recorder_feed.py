"""Unit tests for the bot-side tape runtime-feed reader (M7.4 / M7.8).

The reader (`app/runtime_recorder_feed.py`) maps the recorder's 256-byte
``RuntimeSignalFrame`` shared-memory segment read-only and pulls typed,
version-checked, staleness-aware values out of it under a seqlock. These
tests exercise every branch WITHOUT a real ``/dev/shm`` segment — the
reader's ``_buffer`` injection hook lets us feed a synthetic frame, so the
whole suite runs cross-platform (the bot is developed on Windows, deployed
on Linux; the POSIX ``os.open`` / ``mmap`` path is never hit here).

Coverage:

* **Happy path** — a well-formed v2 frame decodes to the right typed
  fields; ``NaN`` f32 sentinels map to ``None``; counters advance.
* **None-path 1 — collision (odd seq)** — writer caught mid-update on
  every retry → ``read`` returns ``None``, ``collision_count`` bumps.
* **None-path 2 — collision (torn read)** — seq changes between the two
  atomic loads on every retry (simulated by a buffer that bumps its seq
  whenever the body is sliced) → ``None`` + ``collision_count``; and a
  *single* tear recovers on retry (proves the seqlock loop works).
* **None-path 3 — version mismatch** — ``version != EXPECTED`` →
  ``None`` + ``version_mismatch_count``.
* **None-path 4 — staleness** — ``read_fresh`` drops a frame older than
  the budget → ``None`` + ``stale_count`` (while plain ``read`` still
  returns the stale snapshot).
* **None-path 5 — missing segment** — the default-OFF factory returns
  ``None`` for the knob-off case and for a knob-on-but-no-segment case.
* **Layout guard** — the ctypes mirror is exactly 256 B with the
  load-bearing offsets fixed.
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import tempfile
import types

import pytest

from app import runtime_recorder_feed as rrf
from app.runtime_recorder_feed import (
    EXPECTED_FRAME_VERSION,
    RuntimeFeedSnapshot,
    RuntimeRecorderFeed,
    SHMEM_SIZE,
    _nan_to_none,
    open_runtime_recorder_feed,
)

# --------------------------------------------------------------------------
# Synthetic-frame builder — packs a byte-exact RuntimeSignalFrame so we can
# inject it via RuntimeRecorderFeed(_buffer=...). Keep the format string in
# lockstep with _RawFrame._fields_ in the module under test.
# --------------------------------------------------------------------------

# <  little-endian, packed (no implicit padding — matches _pack_ = 1)
# Q seq | I version | I _pad0 | Q ts_mono | Q ts_wall
# 7×f warm-start group A floats | I coverage_valid
# 5×f runtime group B floats | I sweep_in_progress | f sweep_magnitude_bps
# 164s reserved tail
_FRAME_FMT = "<QIIQQfffffffIfffffIf164s"

assert struct.calcsize(_FRAME_FMT) == SHMEM_SIZE, (
    f"test frame format is {struct.calcsize(_FRAME_FMT)} B, expected "
    f"{SHMEM_SIZE} — the test packer drifted from the wire layout"
)

# f32 fields, in packing order, that default to the NaN "not available" wire
# sentinel. coverage_valid / sweep_in_progress are u32 and handled separately.
_FLOAT_FIELDS = (
    "vol_bps_z_24h",
    "vol_bps_p95_24h",
    "microprice_dev_z_24h",
    "microprice_dev_z_7d",
    "basis_ic_1h_mean",
    "basis_ic_7d_mean",
    "coverage_hours",
    "l2_queue_pos_buy",
    "l2_queue_pos_sell",
    "l2_near_touch_depth_ratio",
    "ccf_lead_ms",
    "ccf_strength",
    "sweep_magnitude_bps",
)


def make_frame(
    *,
    seq: int = 2,
    version: int = EXPECTED_FRAME_VERSION,
    ts_mono: int | None = None,
    ts_wall: int = 0,
    coverage_valid: int = 0,
    sweep_in_progress: int = 0,
    **floats: float,
) -> bytearray:
    """Build a 256-byte frame. All f32 signal fields default to NaN; pass
    any of :data:`_FLOAT_FIELDS` by name to light them up. ``ts_mono``
    defaults to *now* (fresh) on the module's own monotonic clock."""
    bad = set(floats) - set(_FLOAT_FIELDS)
    assert not bad, f"unknown float field(s): {sorted(bad)}"
    vals = {k: float("nan") for k in _FLOAT_FIELDS}
    vals.update(floats)
    if ts_mono is None:
        ts_mono = rrf._monotonic_ns()
    packed = struct.pack(
        _FRAME_FMT,
        seq,
        version,
        0,  # _pad0
        ts_mono,
        ts_wall,
        vals["vol_bps_z_24h"],
        vals["vol_bps_p95_24h"],
        vals["microprice_dev_z_24h"],
        vals["microprice_dev_z_7d"],
        vals["basis_ic_1h_mean"],
        vals["basis_ic_7d_mean"],
        vals["coverage_hours"],
        coverage_valid,
        vals["l2_queue_pos_buy"],
        vals["l2_queue_pos_sell"],
        vals["l2_near_touch_depth_ratio"],
        vals["ccf_lead_ms"],
        vals["ccf_strength"],
        sweep_in_progress,
        vals["sweep_magnitude_bps"],
        b"\x00" * (SHMEM_SIZE - 92),
    )
    assert len(packed) == SHMEM_SIZE
    return bytearray(packed)


class _TornBuffer(bytearray):
    """A ``bytearray`` that bumps its seqlock counter (u64 @0) every time
    its body is sliced (``mm[8:256]``) — simulating the recorder publishing
    a new frame *during* the reader's copy.

    Mechanics: ``struct.unpack_from`` reads the underlying memory through the
    buffer protocol (C level, bypassing ``__getitem__``), so the bump done
    here is visible to the reader's *second* ``seq`` load and the read tears.
    ``+2`` keeps the seq even, so the reader never short-circuits on the
    odd-seq check and always reaches the body copy. Set :attr:`bump_limit`
    to stop tearing after N slices (to test seqlock recovery)."""

    bump_limit: int | None = None  # None ⇒ tear on every slice

    def __getitem__(self, key):  # type: ignore[override]
        result = super().__getitem__(key)
        if isinstance(key, slice) and key.start == 8:
            bumps = getattr(self, "_bumps", 0)
            if self.bump_limit is None or bumps < self.bump_limit:
                cur = struct.unpack_from("<Q", self, 0)[0]
                struct.pack_into("<Q", self, 0, cur + 2)
                self._bumps = bumps + 1
        return result


# --------------------------------------------------------------------------
# Layout guard
# --------------------------------------------------------------------------


def test_raw_frame_layout_is_256_with_fixed_offsets() -> None:
    assert ctypes.sizeof(rrf._RawFrame) == SHMEM_SIZE
    assert rrf._RawFrame.seq.offset == 0
    assert rrf._RawFrame.version.offset == 8
    assert rrf._RawFrame.ts_recorder_mono_ns.offset == 16
    assert rrf._RawFrame.vol_bps_z_24h.offset == 32
    assert rrf._RawFrame.microprice_dev_z_24h.offset == 40
    assert rrf._RawFrame.coverage_valid.offset == 60
    assert rrf._RawFrame.sweep_in_progress.offset == 84


def test_nan_to_none_maps_sentinel() -> None:
    assert _nan_to_none(float("nan")) is None
    assert _nan_to_none(1.5) == pytest.approx(1.5)
    assert _nan_to_none(0.0) == 0.0  # real zero passes through (not None)


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_read_happy_path_decodes_fields_and_counts() -> None:
    frame = make_frame(
        seq=4,
        microprice_dev_z_24h=2.5,
        basis_ic_1h_mean=0.42,
        coverage_hours=26.0,
        coverage_valid=1,
    )
    feed = RuntimeRecorderFeed(_buffer=frame)
    snap = feed.read()

    assert snap is not None
    assert snap.seq == 4
    assert snap.version == EXPECTED_FRAME_VERSION
    # Lit fields decode through.
    assert snap.microprice_dev_z_24h == pytest.approx(2.5)
    assert snap.basis_ic_1h_mean == pytest.approx(0.42)
    assert snap.coverage_hours == pytest.approx(26.0)
    assert snap.coverage_valid is True
    # Dark fields are NaN on the wire → None for callers.
    assert snap.vol_bps_p95_24h is None
    assert snap.microprice_dev_z_7d is None
    assert snap.ccf_strength is None
    # Counters.
    assert feed.frames_read_total == 1
    assert feed.collision_count == 0
    assert feed.version_mismatch_count == 0
    assert feed.stale_count == 0


def test_read_counters_advance_monotonically() -> None:
    feed = RuntimeRecorderFeed(_buffer=make_frame(coverage_valid=1))
    for i in range(1, 4):
        assert feed.read() is not None
        assert feed.frames_read_total == i


@pytest.mark.parametrize(
    "bits, want_buy, want_sell",
    [(0b00, False, False), (0b01, True, False), (0b10, False, True), (0b11, True, True)],
)
def test_sweep_bit_decode(bits: int, want_buy: bool, want_sell: bool) -> None:
    snap = RuntimeRecorderFeed(_buffer=make_frame(sweep_in_progress=bits)).read()
    assert snap is not None
    assert snap.sweep_in_progress == bits
    assert snap.sweep_buy is want_buy
    assert snap.sweep_sell is want_sell


def test_coverage_valid_false_when_zero() -> None:
    snap = RuntimeRecorderFeed(_buffer=make_frame(coverage_valid=0)).read()
    assert snap is not None
    assert snap.coverage_valid is False


# --------------------------------------------------------------------------
# None-path 1 — collision via odd seq (writer mid-update on every retry)
# --------------------------------------------------------------------------


def test_read_returns_none_and_counts_collision_on_odd_seq() -> None:
    feed = RuntimeRecorderFeed(_buffer=make_frame(seq=1))  # odd ⇒ never stable
    assert feed.read() is None
    assert feed.collision_count == 1
    assert feed.frames_read_total == 0
    assert feed.version_mismatch_count == 0


# --------------------------------------------------------------------------
# None-path 2 — collision via torn read, plus single-tear recovery
# --------------------------------------------------------------------------


def test_read_returns_none_and_counts_collision_on_persistent_tear() -> None:
    feed = RuntimeRecorderFeed(_buffer=_TornBuffer(make_frame(seq=2)))
    assert feed.read() is None
    assert feed.collision_count == 1
    assert feed.frames_read_total == 0


def test_read_recovers_after_single_tear() -> None:
    buf = _TornBuffer(make_frame(seq=2, coverage_valid=1))
    buf.bump_limit = 1  # tear once, then settle
    feed = RuntimeRecorderFeed(_buffer=buf)
    snap = feed.read()
    assert snap is not None
    assert snap.seq == 4  # advanced by the one tear, then stable
    assert feed.frames_read_total == 1
    assert feed.collision_count == 0


# --------------------------------------------------------------------------
# None-path 3 — version mismatch
# --------------------------------------------------------------------------


def test_read_returns_none_and_counts_version_mismatch() -> None:
    feed = RuntimeRecorderFeed(_buffer=make_frame(version=EXPECTED_FRAME_VERSION + 7))
    assert feed.read() is None
    assert feed.version_mismatch_count == 1
    assert feed.frames_read_total == 0
    assert feed.collision_count == 0


def test_reader_honours_custom_expected_version() -> None:
    # A reader pinned to a future version refuses today's frames cleanly.
    feed = RuntimeRecorderFeed(
        _buffer=make_frame(version=EXPECTED_FRAME_VERSION),
        expected_version=EXPECTED_FRAME_VERSION + 1,
    )
    assert feed.read() is None
    assert feed.version_mismatch_count == 1


# --------------------------------------------------------------------------
# None-path 4 — staleness
# --------------------------------------------------------------------------


def test_read_fresh_drops_stale_frame_but_read_keeps_it() -> None:
    base = rrf._monotonic_ns()
    stale = make_frame(ts_mono=base - 10_000_000_000, coverage_valid=1)  # 10 s old
    feed = RuntimeRecorderFeed(_buffer=stale)

    # Plain read() does NOT gate on staleness — returns the snapshot.
    snap = feed.read()
    assert snap is not None
    assert snap.is_stale() is True
    assert snap.staleness_ns >= 10_000_000_000
    assert feed.stale_count == 0  # read() never bumps stale_count

    # read_fresh() drops it and bumps the counter.
    assert feed.read_fresh() is None
    assert feed.stale_count == 1


def test_read_fresh_passes_fresh_frame() -> None:
    feed = RuntimeRecorderFeed(_buffer=make_frame(coverage_valid=1))
    snap = feed.read_fresh()
    assert snap is not None
    assert snap.is_stale() is False
    assert feed.stale_count == 0


def test_read_fresh_honours_custom_budget() -> None:
    base = rrf._monotonic_ns()
    feed = RuntimeRecorderFeed(_buffer=make_frame(ts_mono=base - 1_000_000_000))  # 1 s
    # Generous budget keeps it.
    assert feed.read_fresh(max_staleness_ns=2_000_000_000) is not None
    assert feed.stale_count == 0
    # Tight budget drops it.
    assert feed.read_fresh(max_staleness_ns=500_000_000) is None
    assert feed.stale_count == 1


def test_snapshot_is_stale_threshold() -> None:
    snap = RuntimeFeedSnapshot(
        seq=2,
        version=EXPECTED_FRAME_VERSION,
        ts_recorder_mono_ns=0,
        ts_recorder_wall_ns=0,
        staleness_ns=1_000_000_000,
        vol_bps_z_24h=None,
        vol_bps_p95_24h=None,
        microprice_dev_z_24h=None,
        microprice_dev_z_7d=None,
        basis_ic_1h_mean=None,
        basis_ic_7d_mean=None,
        coverage_hours=None,
        coverage_valid=False,
        l2_queue_pos_buy=None,
        l2_queue_pos_sell=None,
        l2_near_touch_depth_ratio=None,
        ccf_lead_ms=None,
        ccf_strength=None,
        sweep_in_progress=0,
        sweep_magnitude_bps=None,
    )
    assert snap.is_stale(max_staleness_ns=500_000_000) is True
    assert snap.is_stale(max_staleness_ns=2_000_000_000) is False


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def test_read_after_close_returns_none() -> None:
    feed = RuntimeRecorderFeed(_buffer=make_frame())
    assert feed.read() is not None
    feed.close()
    assert feed.read() is None
    feed.close()  # idempotent — no raise


def test_context_manager_closes() -> None:
    with RuntimeRecorderFeed(_buffer=make_frame()) as feed:
        assert feed.read() is not None
    assert feed.read() is None


# --------------------------------------------------------------------------
# None-path 5 — factory: default-OFF and missing-segment
# --------------------------------------------------------------------------


def test_factory_returns_none_when_knob_off() -> None:
    settings = types.SimpleNamespace(regime_use_runtime_recorder_feed=False)
    assert open_runtime_recorder_feed(settings) is None


def test_factory_returns_none_when_knob_absent() -> None:
    # A settings object lacking the attribute defaults to OFF (getattr).
    assert open_runtime_recorder_feed(types.SimpleNamespace()) is None


def test_factory_returns_none_when_segment_missing() -> None:
    settings = types.SimpleNamespace(regime_use_runtime_recorder_feed=True)
    missing = os.path.join(
        tempfile.gettempdir(), "dtc-tape-runtime-does-not-exist-xyz"
    )
    assert not os.path.exists(missing)
    # os.open raises FileNotFoundError before any POSIX-only mmap call, so
    # this is safe to run on Windows too.
    assert open_runtime_recorder_feed(settings, path=missing) is None
