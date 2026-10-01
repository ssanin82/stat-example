"""Bot-side reader for the tape runtime-feed shared-memory frame (M7.4).

The recorder (`tape`, Rust — repo ``dtc-tools-tape``) is the single writer.
It publishes one ``RuntimeSignalFrame`` (256 bytes, ``#[repr(C, align(64))]``)
into a POSIX shared-memory segment (``/dev/shm/dtc-tape-runtime-v1`` on Linux)
under a seqlock, at ~1 Hz. This module maps that SAME segment read-only and
pulls typed, version-checked, staleness-aware values out of it with no syscall
on the hot read path (just an ``mmap`` slice + two atomic ``seq`` loads).

This is the production bot integration. The executable wire-contract /
reference implementation lives in the recorder repo
(``clients/python/runtime_recorder_feed.py``); the two MUST stay byte-for-byte
in lockstep with ``shared/src/shmem_layout.rs`` (the source of truth) and
``docs/architecture.md`` §9.4 / §10.

Bot integration rules
----------------------
* **Default OFF.** The feed only opens when
  ``REGIME_USE_RUNTIME_RECORDER_FEED=true``. Use
  :func:`open_runtime_recorder_feed` — it returns ``None`` (the bot trades
  on its in-process signals) when the knob is off OR the segment can't be
  opened. The bot NEVER depends on the recorder being up.
* **Never raises on the hot path.** :meth:`RuntimeRecorderFeed.read` /
  :meth:`read_fresh` return ``None`` on collision, version mismatch, or
  staleness; the caller falls back to in-process signals and surfaces the
  failure via the feed's counters.
* **Sentinels.** Every ``f32`` signal field uses ``NaN`` as its "not
  available yet" marker; we map ``NaN -> None`` so callers never branch on
  ``math.isnan``.
* **Version gate.** A segment ``version`` != :data:`EXPECTED_FRAME_VERSION`
  is refused (``None`` + ``version_mismatch_count``). This is the bot side
  of the bot-first deploy rule: the bot ships its new-version reader BEFORE
  the recorder starts publishing the new version, and tolerates the
  mismatch cleanly.
* **Staleness.** The recorder stamps ``CLOCK_MONOTONIC`` (system-wide on
  Linux), so we compare directly against our own monotonic clock with no
  cross-clock skew. :meth:`read_fresh` returns ``None`` past a staleness
  budget and bumps ``stale_count``.

Portability
-----------
The bot is developed on Windows (``C:\\Work\\DTC\\dtc-mm-as``) and deployed
on Linux colo. This module IMPORTS cleanly everywhere — the only
platform-specific calls (``os.open`` on ``/dev/shm``, ``mmap.MAP_SHARED``,
``CLOCK_MONOTONIC``) happen at *open / read* time, which on a dev box never
runs because the knob is OFF and there is no segment. :func:`_monotonic_ns`
falls back to ``time.monotonic_ns`` where ``CLOCK_MONOTONIC`` is absent so
unit tests (which inject a synthetic buffer) run on any OS.

Pure standard library: ``os``, ``mmap``, ``struct``, ``ctypes``, ``math``,
``time``, ``logging``, ``dataclasses``.
"""

from __future__ import annotations

import ctypes
import logging
import math
import mmap
import os
import struct
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Wire constants — keep in lockstep with shared/src/shmem_layout.rs.
# --------------------------------------------------------------------------

#: POSIX shared-memory object name passed to ``shm_open`` by the recorder.
SHMEM_NAME = "/dtc-tape-runtime-v1"

#: Where the kernel exposes that object on Linux (a normal file we mmap).
SHMEM_PATH = "/dev/shm/dtc-tape-runtime-v1"

#: Segment size in bytes — one 4-cache-line block. Must equal the frame.
SHMEM_SIZE = 256

#: Frame schema version this reader understands. Must match the Rust
#: ``FRAME_VERSION``. Bumping the Rust side without bumping here makes every
#: read a clean version-mismatch ``None`` (intended, bot-first deploy: ship
#: this bump first, then the recorder).
EXPECTED_FRAME_VERSION = 2

#: Bounded seqlock retries before we give up and report a collision.
SEQLOCK_MAX_RETRIES = 5

#: Default staleness budget for :meth:`RuntimeRecorderFeed.read_fresh`.
#: The recorder publishes ~1 Hz (warm-start cadence), so 2 s = a couple of
#: missed publishes. The bot trades on in-process signals beyond this.
DEFAULT_MAX_STALENESS_NS = 2_000_000_000


if hasattr(time, "clock_gettime_ns") and hasattr(time, "CLOCK_MONOTONIC"):

    def _monotonic_ns() -> int:
        """System-wide ``CLOCK_MONOTONIC`` (Linux/POSIX) — directly
        comparable to the recorder's ``ts_recorder_mono_ns`` stamp."""
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

else:  # pragma: no cover - Windows dev/test only; never reads a real segment

    def _monotonic_ns() -> int:
        """Portable fallback for platforms without ``CLOCK_MONOTONIC``
        (Windows). Only exercised by unit tests against a synthetic buffer
        — a real segment never exists off-Linux, so the loss of cross-
        process comparability is irrelevant here."""
        return time.monotonic_ns()


class _RawFrame(ctypes.Structure):
    """Byte-exact mirror of the Rust ``RuntimeSignalFrame``.

    Field order, types, and (implicit, natural) alignment match the
    ``#[repr(C)]`` struct, so each field lands at the same offset. The
    trailing reserved array pads the whole thing to ``SHMEM_SIZE``; we
    assert ``sizeof`` below to catch any drift at import time.
    """

    _pack_ = 1  # no implicit padding; the layout is already self-aligned
    _fields_ = [
        # ---- header (32 B) ----
        ("seq", ctypes.c_uint64),                   # @0
        ("version", ctypes.c_uint32),               # @8
        ("_pad0", ctypes.c_uint32),                 # @12
        ("ts_recorder_mono_ns", ctypes.c_uint64),   # @16
        ("ts_recorder_wall_ns", ctypes.c_uint64),   # @24
        # ---- warm-start group A (32 B) ----
        ("vol_bps_z_24h", ctypes.c_float),          # @32
        ("vol_bps_p95_24h", ctypes.c_float),        # @36
        ("microprice_dev_z_24h", ctypes.c_float),   # @40
        ("microprice_dev_z_7d", ctypes.c_float),    # @44
        ("basis_ic_1h_mean", ctypes.c_float),       # @48
        ("basis_ic_7d_mean", ctypes.c_float),       # @52
        ("coverage_hours", ctypes.c_float),         # @56
        ("coverage_valid", ctypes.c_uint32),        # @60
        # ---- runtime group B (28 B) ----
        ("l2_queue_pos_buy", ctypes.c_float),           # @64
        ("l2_queue_pos_sell", ctypes.c_float),          # @68
        ("l2_near_touch_depth_ratio", ctypes.c_float),  # @72
        ("ccf_lead_ms", ctypes.c_float),                # @76
        ("ccf_strength", ctypes.c_float),               # @80
        ("sweep_in_progress", ctypes.c_uint32),         # @84
        ("sweep_magnitude_bps", ctypes.c_float),        # @88
        # ---- reserved tail (164 B) ----
        ("_reserved", ctypes.c_uint8 * (SHMEM_SIZE - 92)),  # @92
    ]


_FRAME_STRUCT_SIZE = ctypes.sizeof(_RawFrame)
assert _FRAME_STRUCT_SIZE == SHMEM_SIZE, (
    f"_RawFrame is {_FRAME_STRUCT_SIZE} B, expected {SHMEM_SIZE} — the "
    "ctypes mirror has drifted from shared/src/shmem_layout.rs"
)
# `seq` MUST be first (offset 0): the writer overlays an atomic u64 on the
# segment base, and our seqlock reads u64 at offset 0.
assert _RawFrame.seq.offset == 0
assert _RawFrame.version.offset == 8
assert _RawFrame.coverage_valid.offset == 60
assert _RawFrame.sweep_in_progress.offset == 84


def _nan_to_none(value: float) -> Optional[float]:
    """Map the wire ``NaN`` sentinel to ``None``; pass real values through."""
    return None if math.isnan(value) else float(value)


@dataclass(frozen=True)
class RuntimeFeedSnapshot:
    """A typed, NaN-cleaned view of one consistent frame read.

    ``f32`` signal fields are ``Optional[float]`` — ``None`` means the
    recorder hasn't warmed that field (or hasn't built its source aggregator
    yet). Integer/header fields are always present.

    ``staleness_ns`` is captured at read time (``now_mono -
    ts_recorder_mono_ns``); :meth:`is_stale` thresholds it.
    """

    # ---- header ----
    seq: int
    version: int
    ts_recorder_mono_ns: int
    ts_recorder_wall_ns: int
    staleness_ns: int

    # ---- warm-start group A ----
    vol_bps_z_24h: Optional[float]
    vol_bps_p95_24h: Optional[float]
    microprice_dev_z_24h: Optional[float]
    microprice_dev_z_7d: Optional[float]
    basis_ic_1h_mean: Optional[float]
    basis_ic_7d_mean: Optional[float]
    coverage_hours: Optional[float]
    coverage_valid: bool

    # ---- runtime group B ----
    l2_queue_pos_buy: Optional[float]
    l2_queue_pos_sell: Optional[float]
    l2_near_touch_depth_ratio: Optional[float]
    ccf_lead_ms: Optional[float]
    ccf_strength: Optional[float]
    sweep_in_progress: int
    sweep_magnitude_bps: Optional[float]

    @property
    def sweep_buy(self) -> bool:
        """Buy-side sweep in progress (bit 0 of ``sweep_in_progress``)."""
        return bool(self.sweep_in_progress & 0b01)

    @property
    def sweep_sell(self) -> bool:
        """Sell-side sweep in progress (bit 1 of ``sweep_in_progress``)."""
        return bool(self.sweep_in_progress & 0b10)

    def is_stale(self, max_staleness_ns: int = DEFAULT_MAX_STALENESS_NS) -> bool:
        """True if the frame is older than ``max_staleness_ns``."""
        return self.staleness_ns > max_staleness_ns

    @classmethod
    def _from_raw(
        cls, raw: _RawFrame, staleness_ns: int
    ) -> "RuntimeFeedSnapshot":
        return cls(
            seq=raw.seq,
            version=raw.version,
            ts_recorder_mono_ns=raw.ts_recorder_mono_ns,
            ts_recorder_wall_ns=raw.ts_recorder_wall_ns,
            staleness_ns=staleness_ns,
            vol_bps_z_24h=_nan_to_none(raw.vol_bps_z_24h),
            vol_bps_p95_24h=_nan_to_none(raw.vol_bps_p95_24h),
            microprice_dev_z_24h=_nan_to_none(raw.microprice_dev_z_24h),
            microprice_dev_z_7d=_nan_to_none(raw.microprice_dev_z_7d),
            basis_ic_1h_mean=_nan_to_none(raw.basis_ic_1h_mean),
            basis_ic_7d_mean=_nan_to_none(raw.basis_ic_7d_mean),
            coverage_hours=_nan_to_none(raw.coverage_hours),
            coverage_valid=bool(raw.coverage_valid),
            l2_queue_pos_buy=_nan_to_none(raw.l2_queue_pos_buy),
            l2_queue_pos_sell=_nan_to_none(raw.l2_queue_pos_sell),
            l2_near_touch_depth_ratio=_nan_to_none(
                raw.l2_near_touch_depth_ratio
            ),
            ccf_lead_ms=_nan_to_none(raw.ccf_lead_ms),
            ccf_strength=_nan_to_none(raw.ccf_strength),
            sweep_in_progress=raw.sweep_in_progress,
            sweep_magnitude_bps=_nan_to_none(raw.sweep_magnitude_bps),
        )


class RuntimeRecorderFeed:
    """Read-only seqlock reader over the recorder's shmem segment.

    Open once at bot startup (via :func:`open_runtime_recorder_feed`), then
    call :meth:`read` / :meth:`read_fresh` from the quoting loop. Both are
    allocation-light and lock-free; a read is an ``mmap`` body copy plus two
    8-byte ``seq`` loads.

    The object is NOT thread-safe (single reader, matching the recorder's
    single-writer model). On any failure the methods return ``None`` and bump
    the relevant counter so the bot can surface feed health without raising
    on the hot path.

    Counters (session-cumulative, monotonic):
      * ``frames_read_total``       — consistent frames returned by ``read``
      * ``collision_count``         — seqlock gave up after retries
      * ``version_mismatch_count``  — frame version != expected
      * ``stale_count``             — ``read_fresh`` dropped a frame for age
    """

    def __init__(
        self,
        path: str = SHMEM_PATH,
        expected_version: int = EXPECTED_FRAME_VERSION,
        *,
        _buffer: Any = None,
    ) -> None:
        self._path = path
        self._expected_version = int(expected_version)

        # Observability counters (mirror the recorder's feed counters).
        self.frames_read_total = 0
        self.collision_count = 0
        self.version_mismatch_count = 0
        self.stale_count = 0

        if _buffer is not None:
            # Injection path (unit tests): a pre-populated bytes-like buffer
            # supporting ``struct.unpack_from`` + slicing. Skips the POSIX
            # open so tests run on any OS without a real segment.
            self._fd = None
            self._mm: Any = _buffer
            return

        self._fd = os.open(path, os.O_RDONLY)
        try:
            self._mm = mmap.mmap(
                self._fd, SHMEM_SIZE, mmap.MAP_SHARED, mmap.PROT_READ
            )
        except Exception:
            os.close(self._fd)
            self._fd = None
            raise

    # -- context manager -------------------------------------------------

    def __enter__(self) -> "RuntimeRecorderFeed":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Unmap and close the segment. Idempotent."""
        mm = getattr(self, "_mm", None)
        if mm is not None and isinstance(mm, mmap.mmap):
            mm.close()
        self._mm = None
        fd = getattr(self, "_fd", None)
        if fd is not None:
            os.close(fd)
            self._fd = None

    # -- reads -----------------------------------------------------------

    def read(self) -> Optional[RuntimeFeedSnapshot]:
        """Return the latest consistent frame, or ``None``.

        ``None`` means one of: the writer collided on every retry
        (``collision_count``), or the segment's version doesn't match
        (``version_mismatch_count``). Staleness is reported on the snapshot
        but does NOT cause ``None`` here — use :meth:`read_fresh` for that.
        """
        mm = self._mm
        if mm is None:
            return None
        for _ in range(SEQLOCK_MAX_RETRIES):
            seq1 = struct.unpack_from("<Q", mm, 0)[0]
            if seq1 & 1:
                continue  # writer mid-update — spin
            # Copy the body [8:SHMEM_SIZE]; the seq bytes are owned by the two
            # atomic loads and reconstructed from the validated seq1.
            body = mm[8:SHMEM_SIZE]
            seq2 = struct.unpack_from("<Q", mm, 0)[0]
            if seq1 != seq2:
                continue  # torn — writer published during the copy
            # Capture staleness as close to the consistent read as we can.
            now_mono = _monotonic_ns()
            raw = _RawFrame.from_buffer_copy(struct.pack("<Q", seq1) + bytes(body))
            if raw.version != self._expected_version:
                self.version_mismatch_count += 1
                return None
            staleness = now_mono - raw.ts_recorder_mono_ns
            self.frames_read_total += 1
            return RuntimeFeedSnapshot._from_raw(raw, staleness)
        self.collision_count += 1
        return None

    def read_fresh(
        self, max_staleness_ns: int = DEFAULT_MAX_STALENESS_NS
    ) -> Optional[RuntimeFeedSnapshot]:
        """Like :meth:`read` but also returns ``None`` (and bumps
        ``stale_count``) if the frame is staler than ``max_staleness_ns``
        (recorder wedged / restarting)."""
        snap = self.read()
        if snap is None:
            return None
        if snap.is_stale(max_staleness_ns):
            self.stale_count += 1
            return None
        return snap


def open_runtime_recorder_feed(
    settings: Any,
    *,
    logger_: Optional[logging.Logger] = None,
    path: str = SHMEM_PATH,
) -> Optional[RuntimeRecorderFeed]:
    """Bot-side factory: honour the default-OFF knob and never raise.

    Returns ``None`` (bot uses in-process signals) when:
      * ``REGIME_USE_RUNTIME_RECORDER_FEED`` is false (the default), or
      * the segment can't be opened (recorder not running, wrong OS, perms).

    Returns an open :class:`RuntimeRecorderFeed` otherwise. The caller owns
    the object lifetime and should :meth:`RuntimeRecorderFeed.close` it on
    shutdown (or use it as a context manager).
    """
    log = logger_ or logger
    if not bool(getattr(settings, "regime_use_runtime_recorder_feed", False)):
        return None
    try:
        feed = RuntimeRecorderFeed(path=path)
    except FileNotFoundError:
        log.warning(
            "runtime_recorder_feed: no segment at %s — recorder not "
            "running; using in-process signals",
            path,
        )
        return None
    except OSError as exc:
        log.warning(
            "runtime_recorder_feed: open failed (%s) — using in-process "
            "signals",
            exc,
        )
        return None
    log.info(
        "runtime_recorder_feed: attached to %s (expected_version=%d)",
        path,
        EXPECTED_FRAME_VERSION,
    )
    return feed


def _format_opt(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:+.4f}"


def _main() -> int:
    """Print one snapshot of the default segment (operator smoke check)::

        python -m app.runtime_recorder_feed
    """
    import argparse

    parser = argparse.ArgumentParser(description="tape runtime-feed reader")
    parser.add_argument("--path", default=SHMEM_PATH, help="segment path")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"no segment at {args.path} — is the recorder running?")
        return 1

    with RuntimeRecorderFeed(args.path) as feed:
        snap = feed.read()
        if snap is None:
            print(
                "read returned None "
                f"(collisions={feed.collision_count}, "
                f"version_mismatch={feed.version_mismatch_count})"
            )
            return 1

        age_ms = snap.staleness_ns / 1e6
        print(f"segment           : {args.path}")
        print(f"seq               : {snap.seq} (even/stable)")
        print(
            f"version           : {snap.version} "
            f"(expected {EXPECTED_FRAME_VERSION})"
        )
        print(
            f"staleness         : {age_ms:.1f} ms"
            f"{'  [STALE]' if snap.is_stale() else ''}"
        )
        print(f"ts_recorder_mono  : {snap.ts_recorder_mono_ns}")
        print(f"ts_recorder_wall  : {snap.ts_recorder_wall_ns}")
        print("-- warm-start (A) --")
        print(f"  coverage_valid       : {snap.coverage_valid}")
        print(f"  coverage_hours       : {_format_opt(snap.coverage_hours)}")
        print(f"  vol_bps_z_24h        : {_format_opt(snap.vol_bps_z_24h)}")
        print(f"  vol_bps_p95_24h      : {_format_opt(snap.vol_bps_p95_24h)}")
        print(
            f"  microprice_dev_z_24h : {_format_opt(snap.microprice_dev_z_24h)}"
        )
        print(
            f"  microprice_dev_z_7d  : {_format_opt(snap.microprice_dev_z_7d)}"
        )
        print(f"  basis_ic_1h_mean     : {_format_opt(snap.basis_ic_1h_mean)}")
        print(f"  basis_ic_7d_mean     : {_format_opt(snap.basis_ic_7d_mean)}")
        print("-- runtime (B) --")
        print(f"  l2_queue_pos_buy        : {_format_opt(snap.l2_queue_pos_buy)}")
        print(
            f"  l2_queue_pos_sell       : {_format_opt(snap.l2_queue_pos_sell)}"
        )
        print(
            "  l2_near_touch_depth_ratio: "
            f"{_format_opt(snap.l2_near_touch_depth_ratio)}"
        )
        print(f"  ccf_lead_ms             : {_format_opt(snap.ccf_lead_ms)}")
        print(f"  ccf_strength            : {_format_opt(snap.ccf_strength)}")
        print(
            f"  sweep_in_progress       : {snap.sweep_in_progress} "
            f"(buy={snap.sweep_buy}, sell={snap.sweep_sell})"
        )
        print(
            f"  sweep_magnitude_bps     : {_format_opt(snap.sweep_magnitude_bps)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
