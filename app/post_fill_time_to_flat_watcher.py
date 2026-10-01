"""Post-fill time-to-flat watcher.

2026-05-13 regime-observability Phase 4c follow-up. Originally
deferred during the rushed v1.3.0 rollout; implemented after the
operator's clear-eyed challenge "why can't this be implemented
right now?" — the answer is "no good reason."

Tracks, per fill, the number of seconds from the fill's timestamp
until ``state.position.position_qty`` first crosses zero. Combined
with the per-fill markout / MAE / MFE data, this isolates the
"how long was inventory exposed" dimension that the markout-5s
column hides:

* markout 5s = "what did mid do in the next 5s"
* MAE / MFE  = "what was the extreme price during 30s"
* time_to_flat = "how long was I actually exposed"

Different questions, different answers. All three together let the
operator decompose post-fill loss into "selection magnitude" (MAE)
vs "directional drift" (markout) vs "holding duration" (time_to_flat).

Hot-path safety
---------------

Same daemon-thread + lock-free-read + late-UPDATE pattern as
``PostFillExcursionWatcher``. The strategy thread never waits on
this; the bot writes ``state.position``, the watcher reads it.

Scope guard
-----------

This is **observability only**. The ``time_to_flat_seconds`` value
is stamped on the fill row but NEVER fed back into the quote-
construction path.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from app.config import Settings
from app.state import BotState

from app import clock as _clock

logger = logging.getLogger(__name__)


class PostFillTimeToFlatWatcher:
    """Daemon-thread time-to-flat tracker for fills.

    Lifecycle (mirror of ``PostFillExcursionWatcher``):
        watcher = PostFillTimeToFlatWatcher.maybe_create(...)
        if watcher: watcher.start()
        # On each fill ingestion:
        if watcher: watcher.track_fill(fill_id, position_qty_before_fill)
        # On bot shutdown:
        if watcher: watcher.stop()
    """

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        storage: Any,
    ) -> None:
        self._settings = settings
        self._state = state
        self._storage = storage
        self._poll_interval_s = float(
            getattr(
                settings,
                "observability_time_to_flat_poll_interval_seconds",
                1.0,
            )
        )
        self._max_wait_s = float(
            getattr(
                settings,
                "observability_time_to_flat_max_wait_seconds",
                300.0,
            )
        )
        self._records: dict[str, dict[str, Any]] = {}
        self._records_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error_count = 0
        self._error_log_throttle_until_mono: float = 0.0

    @classmethod
    def maybe_create(
        cls,
        settings: Settings,
        state: BotState,
        storage: Any,
    ) -> Optional["PostFillTimeToFlatWatcher"]:
        if not bool(
            getattr(settings, "observability_time_to_flat_enabled", False)
        ):
            logger.info(
                "post_fill_time_to_flat_watcher_disabled "
                "reason=OBSERVABILITY_TIME_TO_FLAT_ENABLED=false"
            )
            return None
        if storage is None:
            logger.info(
                "post_fill_time_to_flat_watcher_disabled reason=no_storage"
            )
            return None
        return cls(settings, state, storage)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="post_fill_time_to_flat_watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "post_fill_time_to_flat_watcher_started "
            "poll_s=%.2f max_wait_s=%.0f",
            self._poll_interval_s,
            self._max_wait_s,
        )

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Public API: tracked at fill-ingestion time.                        #
    # ------------------------------------------------------------------ #

    def track_fill(
        self,
        fill_id: str,
        position_qty_before_fill: Optional[float],
    ) -> None:
        """Begin tracking time-to-flat for the given fill.

        ``position_qty_before_fill`` is informational — the watcher
        actually triggers on ``state.position.position_qty`` crossing
        zero (absolute flat). It captures the pre-fill value purely
        so the diagnostic log can show "started at X, flattened at
        Y after Z seconds" if useful.

        Idempotent: re-tracking the same fill_id resets the clock.
        Lock acquired briefly (single dict write).
        """
        if not fill_id:
            return
        with self._records_lock:
            self._records[fill_id] = {
                "ts_mono": _clock.monotonic(),
                "pos_before_fill": (
                    float(position_qty_before_fill)
                    if position_qty_before_fill is not None
                    else None
                ),
            }

    # ------------------------------------------------------------------ #
    # Internal loop.                                                      #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        while not self._stop.wait(self._poll_interval_s):
            try:
                self._tick_once()
            except Exception as e:
                self._error_count += 1
                now_mono = _clock.monotonic()
                if (
                    self._error_count <= 3
                    or now_mono >= self._error_log_throttle_until_mono
                ):
                    logger.warning(
                        "post_fill_time_to_flat_watcher_tick_failed "
                        "consecutive=%d err=%s",
                        self._error_count,
                        str(e)[:200],
                    )
                    self._error_log_throttle_until_mono = now_mono + 60.0

    def _tick_once(self) -> None:
        # Read current position lock-free. ``state.position`` is a
        # PositionSnapshot dataclass; ``.position_qty`` is a single
        # float — atomic to read.
        pos = self._state.position
        if pos is None:
            return
        try:
            qty_now = float(pos.position_qty)
        except (TypeError, ValueError):
            return

        now_mono = _clock.monotonic()
        flat_eps = 1e-9  # absolute zero crossing; venues quote in lots
        to_write: list[tuple[str, float]] = []  # (fill_id, seconds)
        to_drop: list[str] = []  # fill_ids to remove (timed out)

        with self._records_lock:
            for fid, rec in self._records.items():
                elapsed = now_mono - rec["ts_mono"]
                if abs(qty_now) <= flat_eps:
                    # Position is now flat — record the elapsed time.
                    to_write.append((fid, elapsed))
                    to_drop.append(fid)
                elif elapsed >= self._max_wait_s:
                    # Timed out — give up; column stays NULL.
                    to_drop.append(fid)
            for fid in to_drop:
                del self._records[fid]

        # Write outside the records lock (storage has its own lock).
        for fid, seconds in to_write:
            try:
                self._storage.update_fill_time_to_flat(fid, seconds)
            except Exception:
                logger.exception(
                    "update_fill_time_to_flat_failed fill=%s", fid
                )
