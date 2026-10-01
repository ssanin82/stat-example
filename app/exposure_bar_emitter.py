"""Periodic exposure-bar emitter for regime-observability Phase 2.

Writes one row per ``OBSERVABILITY_EXPOSURE_BAR_INTERVAL_SECONDS``
(default 5s) capturing the bot's market + strategy state regardless
of whether a fill happened in that window. Provides the "exposure
denominator" that turns per-regime fill outcomes into per-regime
edge / risk-adjusted metrics.

Why this exists
---------------

Without exposure bars, per-regime analytics measure outcomes but not
how long the bot was exposed to a regime. "Bucket X had 12 fills with
+1.2 bp markout" is uninterpretable without knowing whether the bot
spent 30 minutes or 8 hours in that bucket. Exposure bars provide the
minutes-in-regime denominator that makes such ratios meaningful.

See ``plans/regime-observability.md`` Phase 2 for the design rationale,
field choices, and hot-path-safety rules.

Hot-path impact
---------------

By construction, this emitter is hot-path-invisible:

* Runs on its own daemon thread (``threading.Thread(daemon=True)``).
* Reads ``state.market`` / ``state.position`` / ``state.toxicity_snapshot``
  / ``state.binance_basis_ewma`` etc. **lock-free**. These are simple
  attribute reads against either atomic-reference fields (``state.market``
  is a BestBidAsk dataclass swap) or single-float fields. Python's GIL
  guarantees no torn reads for these.
* Writes via ``Storage.insert_exposure_bar``, which acquires the
  storage lock for the commit only. The strategy thread does not
  share this lock — it has its own ``state._lock``.
* On any per-cycle exception, logs at most one WARNING per 60s and
  continues. Never blocks the strategy thread.

Disable via ``OBSERVABILITY_EXPOSURE_BARS_ENABLED=false`` if any
unforeseen impact appears. Single-knob walk-back, no code revert.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional

from app.config import Settings
from app.state import BotState

from app import clock as _clock

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return _clock.now_utc().isoformat()


def _to_int_bool(v: Optional[bool]) -> Optional[int]:
    """SQLite INTEGER bool convention: NULL / 0 / 1."""
    if v is None:
        return None
    return 1 if v else 0


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # NaN / inf → None (SQLite can store them but they break downstream).
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


class ExposureBarEmitter:
    """Daemon-thread emitter for ``exposure_bars`` table rows.

    Lifecycle: construct → start() → (runs in background) → stop().
    Same pattern as ``EquityHistoryPublisher`` / ``LiveStatsPublisher``.

    The class is intentionally small. State sampling is in
    ``_capture_bar``; everything else is plumbing.
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
        self._interval_s = float(
            getattr(
                settings,
                "observability_exposure_bar_interval_seconds",
                5.0,
            )
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Error throttling: log every error initially, then suppress
        # duplicates beyond 3 consecutive failures to avoid log
        # spam if the storage layer is misbehaving. The emitter
        # keeps trying regardless — best-effort.
        self._consecutive_errors = 0
        self._error_log_throttle_until_mono: float = 0.0

    @classmethod
    def maybe_create(
        cls,
        settings: Settings,
        state: BotState,
        storage: Any,
    ) -> Optional["ExposureBarEmitter"]:
        """Factory with graceful disablement.

        Returns ``None`` (and logs the reason) if:
        - ``OBSERVABILITY_EXPOSURE_BARS_ENABLED`` is false
        - ``storage`` is None (e.g. tests, or storage init failed)
        """
        if not bool(
            getattr(settings, "observability_exposure_bars_enabled", True)
        ):
            logger.info(
                "exposure_bars_disabled reason=OBSERVABILITY_EXPOSURE_BARS_ENABLED=false"
            )
            return None
        if storage is None:
            logger.info("exposure_bars_disabled reason=no_storage")
            return None
        return cls(settings, state, storage)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="exposure_bars", daemon=True
        )
        self._thread.start()
        logger.info(
            "exposure_bars_started interval_s=%.2f",
            self._interval_s,
        )

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Internal: loop + capture                                            #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        """Periodic capture loop. ``threading.Event.wait`` provides
        responsive shutdown (no need to wait the full interval)."""
        while not self._stop.wait(self._interval_s):
            try:
                row = self._capture_bar()
                if row is not None:
                    self._storage.insert_exposure_bar(row)
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                now_mono = _clock.monotonic()
                # Log every error for the first 3, then throttle to
                # one per 60s. The emitter keeps trying regardless.
                if (
                    self._consecutive_errors <= 3
                    or now_mono >= self._error_log_throttle_until_mono
                ):
                    logger.warning(
                        "exposure_bar_emit_failed consecutive=%d err=%s",
                        self._consecutive_errors,
                        str(e)[:200],
                    )
                    self._error_log_throttle_until_mono = now_mono + 60.0

    def _capture_bar(self) -> Optional[dict[str, Any]]:
        """Snapshot the bot's current state into one exposure-bar row.

        All reads are lock-free; the snapshot is a copy of a few
        floats / enums. Python's GIL guarantees no torn reads for
        single-attribute access against atomic types.

        Returns ``None`` when there's no meaningful state to record
        yet (e.g. first few ticks after startup before market data
        arrives).

        2026-05-13 bugfix pass: corrected attribute paths after the
        v1.3.0 deployment revealed that ~7 fields were always null.
        Wrong paths fixed:
          - state.toxicity (not state.toxicity_snapshot)
          - state.vol_bps + state.vol_sigma (not state.last_quote_decision.vol_estimate)
          - state.last_active_sides (not state.last_quote_decision.active_sides)
          - state.adaptive_spread_widen_until_mono (deadline) → derived bool
          - state.quote_elig_recovery_remaining_ms → derived bool
        Newly computed:
          - bid_distance_ticks / ask_distance_ticks (never computed before)
        Still deferred (TODO): post_fill_cooldown_active_*,
          at_touch_adverse_pause_* — these live on OrderManager, not
          state. Requires wiring through bot lifecycle to populate.
        """
        state = self._state
        # Market — single reference swap to a frozen-ish BestBidAsk dataclass.
        mkt = state.market
        if mkt is None:
            return None
        mid = _safe_float(getattr(mkt, "mid_price", None))
        bb = _safe_float(getattr(mkt, "best_bid", None))
        ba = _safe_float(getattr(mkt, "best_ask", None))
        bsz = _safe_float(getattr(mkt, "bid_size", None))
        asz = _safe_float(getattr(mkt, "ask_size", None))

        spread_bps: Optional[float] = None
        microprice: Optional[float] = None
        imbalance_top: Optional[float] = None
        if mid is not None and bb is not None and ba is not None and mid > 0:
            spread_bps = (ba - bb) / mid * 10_000.0
        if (
            bsz is not None
            and asz is not None
            and bb is not None
            and ba is not None
        ):
            denom = bsz + asz
            if denom > 0:
                microprice = (bsz * ba + asz * bb) / denom
                imbalance_top = (bsz - asz) / denom

        # Position / inventory.
        inventory_qty: Optional[float] = None
        inventory_utilization: Optional[float] = None
        try:
            inventory_qty = float(state.position.position_qty)
            max_abs = float(
                getattr(self._settings, "max_abs_position", 0.0) or 0.0
            )
            if max_abs > 0:
                inventory_utilization = abs(inventory_qty) / max_abs
        except Exception:
            pass

        # Strategy state — corrected attribute paths.
        active_sides: Optional[str] = None
        quote_eligibility: Optional[str] = None
        toxicity_score: Optional[float] = None
        vol_estimate: Optional[float] = None
        try:
            elig_snap = state.quote_eligibility_snapshot_dict or {}
            quote_eligibility = elig_snap.get("quote_eligibility_state")
        except Exception:
            pass
        # FIX: ``state.toxicity`` (not ``state.toxicity_snapshot``).
        try:
            tox = getattr(state, "toxicity", None)
            if tox is not None:
                toxicity_score = _safe_float(getattr(tox, "score", None))
        except Exception:
            pass
        # FIX: vol estimate lives directly on state as vol_bps (bps
        # scale, more useful than vol_sigma's raw sigma).
        try:
            vol_estimate = _safe_float(getattr(state, "vol_bps", None))
        except Exception:
            pass
        # FIX: active_sides → state.last_active_sides (string snapshot
        # of the most recent quote-decision's active_sides value).
        try:
            las = getattr(state, "last_active_sides", None)
            if las is not None:
                active_sides = str(las)
        except Exception:
            pass

        # Basis.
        binance_basis_ewma = _safe_float(
            getattr(state, "binance_basis_ewma", None)
        )
        basis_regime_sign: Optional[int] = None
        try:
            br = getattr(state, "basis_regime", None)
            if br is not None:
                _sign = getattr(br, "last_regime_sign", None)
                if _sign is not None:
                    basis_regime_sign = int(_sign)
        except Exception:
            pass

        # Liquidity provision — what's currently on the book?
        bid_live: Optional[int] = None
        ask_live: Optional[int] = None
        bid_distance_ticks: Optional[float] = None
        ask_distance_ticks: Optional[float] = None
        quoted_spread_bps: Optional[float] = None
        try:
            # v1.4.194: migrated off the deprecated property shims.
            from app.enums import OrderStatus, Side  # local import to avoid cycle
            wb = state.get_working_order(Side.BUY, 0)
            wa = state.get_working_order(Side.SELL, 0)

            bid_alive = wb is not None and wb.status in (
                OrderStatus.ACKED,
                OrderStatus.PARTIAL,
            )
            ask_alive = wa is not None and wa.status in (
                OrderStatus.ACKED,
                OrderStatus.PARTIAL,
            )
            bid_live = 1 if bid_alive else 0
            ask_live = 1 if ask_alive else 0
            # FIX: compute bid/ask_distance_ticks. Uses the venue's
            # ``price_tick`` from symbol_spec if available; otherwise
            # falls back to "1 tick = 1 quoted unit" which produces
            # approximate distances but never breaks. Distance is the
            # working order's price vs the touch, in ticks (≥ 0).
            tick = self._resolve_price_tick(state)
            if (
                bid_alive
                and wb is not None
                and bb is not None
                and tick > 0
            ):
                bid_distance_ticks = max(0.0, (bb - float(wb.price)) / tick)
            if (
                ask_alive
                and wa is not None
                and ba is not None
                and tick > 0
            ):
                ask_distance_ticks = max(0.0, (float(wa.price) - ba) / tick)
            # Quoted spread (only meaningful when both sides alive).
            if (
                bid_alive
                and ask_alive
                and mid is not None
                and mid > 0
                and wb is not None
                and wa is not None
            ):
                quoted_spread_bps = (
                    (float(wa.price) - float(wb.price)) / mid * 10_000.0
                )
        except Exception:
            pass

        # Gates. Defensive — each lookup independent.
        adaptive_widen_active: Optional[int] = None
        # FIX: state.adaptive_spread_widen_until_mono is a monotonic
        # deadline. Active iff now_mono < deadline.
        try:
            deadline = float(
                getattr(state, "adaptive_spread_widen_until_mono", 0.0) or 0.0
            )
            now_mono = _clock.monotonic()
            adaptive_widen_active = 1 if now_mono < deadline else 0
        except Exception:
            pass

        hold_all_active: Optional[int] = None
        try:
            elig = quote_eligibility or ""
            if elig == "HOLD_ALL":
                hold_all_active = 1
            elif elig in (
                "QUOTE_BOTH",
                # OKX-style enum values (QuoteEligibility.QUOTE_BUY_ONLY etc.)
                "QUOTE_BUY_ONLY",
                "QUOTE_SELL_ONLY",
                # Legacy/HL-style enum values (kept for back-compat).
                "QUOTE_BID_ONLY",
                "QUOTE_ASK_ONLY",
                # Bare side variants used by ActiveSides.
                "BID_ONLY",
                "ASK_ONLY",
            ):
                hold_all_active = 0
        except Exception:
            pass

        # FIX: recovery_cooldown_active derived from
        # state.quote_elig_recovery_remaining_ms (>0 means active).
        recovery_cooldown_active: Optional[int] = None
        try:
            rem_ms = getattr(
                state, "quote_elig_recovery_remaining_ms", None
            )
            if rem_ms is not None:
                recovery_cooldown_active = 1 if float(rem_ms) > 0.0 else 0
        except Exception:
            pass

        # Gate flags that live on OrderManager (not state). Captured
        # via a small dict that the bot can update lock-free; see
        # ``set_observability_gate_flags`` on BotState. If the bot
        # never updates this dict (older deploys), the values stay
        # None.
        gate_flags = getattr(state, "observability_gate_flags", None) or {}
        post_fill_cooldown_active_bid = _to_int_bool(
            gate_flags.get("post_fill_cooldown_bid")
        )
        post_fill_cooldown_active_ask = _to_int_bool(
            gate_flags.get("post_fill_cooldown_ask")
        )
        at_touch_adverse_pause_bid = _to_int_bool(
            gate_flags.get("at_touch_adverse_pause_bid")
        )
        at_touch_adverse_pause_ask = _to_int_bool(
            gate_flags.get("at_touch_adverse_pause_ask")
        )

        # 2026-05-14 todo-027 Tier 2: per-bar active flags for the
        # three previously fire-counted-only gates. Same monotonic-
        # deadline-vs-now pattern as ``adaptive_widen_active`` above.
        # Defensive on every attribute access — these gates are
        # populated by separate subsystems and may not be wired in
        # every test/profile context.
        vol_trend_active: Optional[int] = None
        try:
            vt = getattr(state, "vol_trend_gate", None)
            if vt is not None:
                vt_until = float(
                    getattr(vt, "cooldown_until_mono", 0.0) or 0.0
                )
                vol_trend_active = (
                    1 if _clock.monotonic() < vt_until else 0
                )
        except Exception:
            pass

        post_swing_active: Optional[int] = None
        try:
            ps = getattr(state, "post_swing", None)
            if ps is not None:
                ps_until = float(
                    getattr(ps, "cooldown_until_mono", 0.0) or 0.0
                )
                post_swing_active = (
                    1 if _clock.monotonic() < ps_until else 0
                )
        except Exception:
            pass

        # Tier label rather than 0/1: the drawdown ladder has 6
        # states (CLEAR / WIDEN / PAUSE_SHORT / PAUSE_LONG /
        # RESUME_TESTING / KILLED) and the operator wants to
        # distinguish "spread widened" from "long pause" in the
        # dashboard's gate-effectiveness table. Storing the enum
        # value string keeps the breakdown faithful at zero extra
        # cost vs storing a binary.
        session_drawdown_tier: Optional[str] = None
        try:
            sd = getattr(state, "session_drawdown", None)
            if sd is not None:
                tier = getattr(sd, "tier", None)
                if tier is not None:
                    session_drawdown_tier = (
                        getattr(tier, "value", None) or str(tier)
                    )
        except Exception:
            pass

        return {
            "session_id": state.session_id or "unknown",
            "ts_bar": _now_iso(),
            "symbol": getattr(self._settings, "symbol", None),
            "mid": mid,
            "spread_bps": spread_bps,
            "microprice": microprice,
            "imbalance_top": imbalance_top,
            "bid_size_top": bsz,
            "ask_size_top": asz,
            "inventory_qty": inventory_qty,
            "inventory_utilization": inventory_utilization,
            "active_sides": active_sides,
            "quote_eligibility": quote_eligibility,
            "toxicity_score": toxicity_score,
            "vol_estimate": vol_estimate,
            "binance_basis_ewma": binance_basis_ewma,
            "basis_regime_sign": basis_regime_sign,
            "bid_live": bid_live,
            "ask_live": ask_live,
            "bid_distance_ticks": bid_distance_ticks,
            "ask_distance_ticks": ask_distance_ticks,
            "quoted_spread_bps": quoted_spread_bps,
            "adaptive_widen_active": adaptive_widen_active,
            "hold_all_active": hold_all_active,
            "recovery_cooldown_active": recovery_cooldown_active,
            "post_fill_cooldown_active_bid": post_fill_cooldown_active_bid,
            "post_fill_cooldown_active_ask": post_fill_cooldown_active_ask,
            "at_touch_adverse_pause_bid": at_touch_adverse_pause_bid,
            "at_touch_adverse_pause_ask": at_touch_adverse_pause_ask,
            # 2026-05-14 todo-027 Tier 2.
            "vol_trend_active": vol_trend_active,
            "post_swing_active": post_swing_active,
            "session_drawdown_tier": session_drawdown_tier,
        }

    @staticmethod
    def _resolve_price_tick(state: BotState) -> float:
        """Best-effort lookup of the venue's ``price_tick`` for the
        active symbol. Reads ``state.symbol_spec.price_tick`` if the
        bot has set it; otherwise returns 0.0 (caller treats as
        "no tick info" and skips the tick-based distance calculation).
        """
        try:
            spec = getattr(state, "symbol_spec", None)
            if spec is not None:
                t = getattr(spec, "price_tick", None)
                if t is not None and float(t) > 0:
                    return float(t)
        except Exception:
            pass
        return 0.0


def _maybe_bool(obj: Any, attr: str) -> Optional[bool]:
    """Safe attribute access for bool fields. Returns None on any
    error (missing attribute, type mismatch). Used in the emitter
    to keep state sampling resilient to evolving state schemas."""
    try:
        v = getattr(obj, attr, None)
        if v is None:
            return None
        return bool(v)
    except Exception:
        return None
