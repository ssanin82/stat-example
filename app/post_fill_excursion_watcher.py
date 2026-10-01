"""Post-fill excursion watcher: MAE / MFE / time-to-flat tracker.

2026-05-13 regime-observability Phase 4c.

Tracks the price path for ``window`` seconds after each fill and
records:

* **MAE** (Maximum Adverse Excursion) — worst price the bot would have
  seen unwinding the fill at any moment inside the window. Sign
  convention: ≤ 0 ("most adverse from fill price").
* **MFE** (Maximum Favorable Excursion) — best price the bot would
  have seen unwinding. Sign convention: ≥ 0.

Two windows: 5s and 30s. The 5s window catches immediate adverse-
selection magnitude; the 30s window captures the slower-mean-reversion
path that single-point markouts miss.

Hot-path safety
---------------

By construction, this watcher is hot-path-invisible:

* Daemon thread polling at ``OBSERVABILITY_MAE_MFE_POLL_INTERVAL_SECONDS``
  (default 500 ms).
* Reads ``state.market.mid_price`` lock-free (Python's GIL makes
  single-attribute float reads atomic enough for this purpose).
* Writes to storage via ``Storage.update_fill_excursion`` outside the
  records lock.
* Per-fill record state is ~100 bytes; in-flight cap is ~6 records
  on a 1-fill/minute symbol (30s watch × 1 fill/min = 0.5 typical,
  6 in burst). Memory bounded.
* The strategy thread never waits on this. Disable via
  ``OBSERVABILITY_MAE_MFE_ENABLED=false`` for the single-knob revert.

Scope guard
-----------

This is **observability only**. MAE/MFE values are stamped on the
fill row but NEVER fed back into the quote-construction path. See
``plans/regime-observability.md`` Phase 4c.
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


# Windows over which to record excursions, in seconds. Ordered low →
# high; the loop captures snapshots at each crossing.
_WATCH_WINDOWS_SECONDS = (5.0, 30.0)
_MAX_WATCH_SECONDS = max(_WATCH_WINDOWS_SECONDS)


class PostFillExcursionWatcher:
    """Daemon-thread MAE/MFE tracker for fills.

    Lifecycle:
        watcher = PostFillExcursionWatcher.maybe_create(...)
        if watcher: watcher.start()
        # On each fill ingestion:
        if watcher: watcher.track_fill(fill_id, side, mid_at_fill)
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
                "observability_mae_mfe_poll_interval_seconds",
                0.5,
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
    ) -> Optional["PostFillExcursionWatcher"]:
        if not bool(getattr(settings, "observability_mae_mfe_enabled", False)):
            logger.info(
                "post_fill_excursion_watcher_disabled "
                "reason=OBSERVABILITY_MAE_MFE_ENABLED=false"
            )
            return None
        if storage is None:
            logger.info(
                "post_fill_excursion_watcher_disabled reason=no_storage"
            )
            return None
        return cls(settings, state, storage)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="post_fill_excursion_watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "post_fill_excursion_watcher_started poll_s=%.2f windows=%s",
            self._poll_interval_s,
            _WATCH_WINDOWS_SECONDS,
        )

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Public API: tracked at fill-ingestion time.                        #
    # ------------------------------------------------------------------ #

    def track_fill(
        self,
        fill_id: str,
        side: str,
        mid_at_fill: float,
    ) -> None:
        """Begin tracking MAE/MFE for the given fill. Called from
        ``fill_ingestion.ingest_hl_fill_raw`` after the Fill object
        is constructed.

        Idempotent: re-tracking the same fill_id resets the running
        min/max. Lock acquired briefly (single dict write).

        ``side`` is "BUY" or "SELL" (matches ``Fill.side.value``).
        """
        if not fill_id:
            return
        try:
            mid_f = float(mid_at_fill)
        except (TypeError, ValueError):
            return
        if mid_f <= 0 or mid_f != mid_f:  # negative or NaN
            return
        with self._records_lock:
            self._records[fill_id] = {
                "ts_mono": _clock.monotonic(),
                "side": str(side or "").upper(),
                "mid_at_fill": mid_f,
                "min_mid": mid_f,
                "max_mid": mid_f,
                "written_5s": False,
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
                        "post_fill_excursion_watcher_tick_failed "
                        "consecutive=%d err=%s",
                        self._error_count,
                        str(e)[:200],
                    )
                    self._error_log_throttle_until_mono = now_mono + 60.0

    def _tick_once(self) -> None:
        # Read mid lock-free.
        mkt = self._state.market
        if mkt is None:
            return
        mid_attr = getattr(mkt, "mid_price", None)
        if mid_attr is None:
            return
        try:
            mid_now = float(mid_attr)
        except (TypeError, ValueError):
            return
        if mid_now <= 0 or mid_now != mid_now:
            return

        now_mono = _clock.monotonic()
        to_write: list[tuple[str, dict[str, Any], bool]] = []
        # (fill_id, record_snapshot, is_final) — gathered under the lock,
        # written outside.

        short_window_s = _WATCH_WINDOWS_SECONDS[0]
        long_window_s = _MAX_WATCH_SECONDS
        with self._records_lock:
            to_drop: list[str] = []
            for fid, rec in self._records.items():
                # Update running min/max.
                if mid_now < rec["min_mid"]:
                    rec["min_mid"] = mid_now
                if mid_now > rec["max_mid"]:
                    rec["max_mid"] = mid_now
                elapsed = now_mono - rec["ts_mono"]
                # Short-window deadline (first crossing).
                if not rec["written_5s"] and elapsed >= short_window_s:
                    rec["written_5s"] = True
                    to_write.append((fid, dict(rec), False))
                # Long-window deadline (final; drop after).
                if elapsed >= long_window_s:
                    to_write.append((fid, dict(rec), True))
                    to_drop.append(fid)
            for fid in to_drop:
                del self._records[fid]

        # Write outside the records lock. Storage has its own lock.
        for fid, rec, is_final in to_write:
            try:
                mae_bps, mfe_bps = self._compute_mae_mfe_bps(rec)
                if is_final:
                    self._storage.update_fill_excursion(
                        fid,
                        mae_30s_bps=mae_bps,
                        mfe_30s_bps=mfe_bps,
                    )
                    # 1.3.82: also notify the 30s-MAE gate. The gate
                    # is a thread-safe state machine that the quote
                    # loop polls. When disabled in settings this is
                    # still cheap — observe() with the wrong window
                    # would arm nothing because the gate's own
                    # enabled flag is checked in the quote loop's
                    # is_active() call path.
                    if bool(
                        getattr(self._settings, "mae_gate_enabled", False)
                    ):
                        try:
                            from app import mae_gate

                            mae_gate.observe(
                                self._state.mae_gate,
                                now_mono=now_mono,
                                mae_30s_bps=mae_bps,
                                fill_window=int(
                                    self._settings.mae_gate_fill_window
                                ),
                                hard_threshold_bps=float(
                                    self._settings.mae_gate_hard_threshold_bps
                                ),
                                cooldown_seconds=float(
                                    self._settings.mae_gate_cooldown_seconds
                                ),
                                # Phase 2K.7 favorable-exit knobs.
                                # Defensive ``getattr`` for Settings
                                # shapes from before v1.4.158.
                                favorable_exit_enabled=bool(
                                    getattr(
                                        self._settings,
                                        "mae_gate_favorable_exit_enabled",
                                        True,
                                    )
                                ),
                                clear_band_mult=float(
                                    getattr(
                                        self._settings,
                                        "mae_gate_clear_band_mult",
                                        0.5,
                                    )
                                ),
                                favorable_exit_dwell_seconds=float(
                                    getattr(
                                        self._settings,
                                        "mae_gate_favorable_exit_dwell_seconds",
                                        5.0,
                                    )
                                ),
                            )
                        except Exception:
                            # Observation failure must never derail
                            # the watcher's primary job (storage
                            # stamping). Log + continue.
                            logger.exception(
                                "mae_gate_observe_failed fill=%s", fid
                            )
                else:
                    self._storage.update_fill_excursion(
                        fid,
                        mae_5s_bps=mae_bps,
                        mfe_5s_bps=mfe_bps,
                    )
            except Exception:
                logger.exception("update_fill_excursion_failed fill=%s", fid)

    @staticmethod
    def _compute_mae_mfe_bps(
        rec: dict[str, Any],
    ) -> tuple[Optional[float], Optional[float]]:
        """Convert running min/max into MAE/MFE bps for the fill's side.

        BUY side: bot is long after the fill. Adverse = price down.
          MAE = (min_mid - mid_at_fill) / mid_at_fill * 10000  (≤ 0)
          MFE = (max_mid - mid_at_fill) / mid_at_fill * 10000  (≥ 0)

        SELL side: bot is short after the fill. Adverse = price up.
          MAE = (mid_at_fill - max_mid) / mid_at_fill * 10000  (≤ 0)
          MFE = (mid_at_fill - min_mid) / mid_at_fill * 10000  (≥ 0)
        """
        m_at = rec["mid_at_fill"]
        if m_at <= 0:
            return None, None
        side = rec["side"]
        if side == "BUY":
            mae = (rec["min_mid"] - m_at) / m_at * 10_000.0
            mfe = (rec["max_mid"] - m_at) / m_at * 10_000.0
        elif side == "SELL":
            mae = (m_at - rec["max_mid"]) / m_at * 10_000.0
            mfe = (m_at - rec["min_mid"]) / m_at * 10_000.0
        else:
            return None, None
        return round(mae, 4), round(mfe, 4)
