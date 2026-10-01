"""Real-Bot replay integration — Phase 4b (v1.4.236).

Phase 3 + 4a shipped paper-only replay (event flow + paper executor
synthesizing fills + summary metrics). Phase 4b wires the actual
``app.bot.Bot`` against the replay harness so the recorded market
data drives the strategy code that runs in production.

What's wired:

* **Bot** — constructed with the substituted dependencies:
  - ``client`` = :class:`~app.backtest.paper_executor.PaperExecutor`
  - ``private_stream`` = :class:`~app.backtest.streams.ReplayPrivateStream`
  - ``public_stream`` = :class:`~app.backtest.streams.ReplayPublicStream`
  - ``clock`` = :class:`~app.clock.ReplayClock`
  - ``storage`` = fresh in-memory SQLite (each replay starts clean)
  - ``state`` = fresh :class:`~app.state.BotState`

* **State mutations** mirroring production's ``main.py``
  ``_on_public_bbo`` callback:
  - OKX BBO updates ``state.market`` via
    ``state.apply_market_book_only`` + ``state.market_refresh_note_success``.
  - Binance BBO updates ``state.binance_best_bid/best_ask/mid``
    + folds into the basis EWMA.

* **Tick driving** — :meth:`BotRunner.one_tick` calls the bot's
  ``one_tick()``; the replay driver invokes this from its ``on_tick``
  callback at the configured cadence.

* **Gate-firing extraction** — at finalize, :meth:`extract_gates_fired`
  scrapes BotState's gate counters into the report's ``gates_fired``.

State seeding (audit P2 #14) — corrected understanding:

* **The warmup window DOES pre-roll market-data state.** During the
  ``warmup_seconds`` window the driver still DISPATCHES every event
  (``okx_public`` / ``binance_public`` deliveries) — it only gates the
  bot's ``one_tick()`` (see ``driver.py`` ``_maybe_fire_ticks``). So the
  ``on_bbo`` / ``on_trade`` callbacks installed by
  :func:`_install_replay_callbacks` fire throughout warmup, which means:

  - ``state.binance_basis_ewma`` folds in on every Binance BBO;
  - ``state.flow_score`` (and ``state.recent_trades``) accrues on every
    OKX trade;
  - the bot's ``VolatilityEstimator`` accrues mids — it's registered as
    a mid-change listener (``state.add_mid_change_listener``), fired by
    ``apply_market_book_only`` on every OKX BBO.

  These estimators have short time constants (flow TFI window ~1 s;
  basis EWMA α=0.05 ≈ 20 samples; vol needs ``VOL_WINDOW_SAMPLES``=32
  distinct mids), so the :data:`DEFAULT_REPLAY_WARMUP_SECONDS` default
  warms all three with comfortable margin. The earlier docstring
  claimed warmup "does NOT pre-roll any history" — that was wrong; only
  the bot's quoting tick is gated, not event dispatch.

* **The remaining gap is the SCORED-window cost + fill history.** The
  warmup window is carved OUT of the scored span, so warming this way
  costs scored time. A scenario may instead carry an
  ``initial_state.json`` (``ScenarioManifest.initial_state_snapshot``);
  :func:`seed_state_from_snapshot` applies it at construction so the
  scored window starts already-warm with NO carve-out. Fill history is
  deliberately NOT seeded — see that function's docstring for why.

* **No simulated REST polls.** The bot's reconciler / account-
  refresh logic that polls the exchange isn't fed any data; gates
  depending on REST snapshots may behave differently than they
  would in production.

* **No mark price feed.** The paper executor uses mid as the mark.
  Production OKX mark is a separate WS stream not currently in the
  recorder. Drawdown-on-mark gates may fire differently.

* **TRADING_ENABLED=False by default** so the bot won't actually
  place real orders during a replay misconfiguration. Override via
  config for full strategy execution.

For testing this module: see ``tests/test_replay_bot_runner.py``.
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, Optional

from app.backtest.driver import ReplayConfig, ReplayHarness
from app.backtest.paper_executor import PaperExecutor
from app.backtest.streams import ReplayPrivateStream, ReplayPublicStream
from app.clock import Clock, ReplayClock
from app.models import BestBidAsk

if TYPE_CHECKING:
    from app.bot import Bot
    from app.config import Settings
    from app.state import BotState
    from app.storage import Storage

logger = logging.getLogger(__name__)


# Default warmup for replay (audit P2 #14). The market-data state the bot
# quotes on — Binance basis EWMA, flow_score, the VolatilityEstimator —
# warms purely from event callbacks during the warmup window (events still
# DISPATCH during warmup; only the bot's one_tick is gated — see driver.py
# ``_maybe_fire_ticks`` and the module docstring above). These estimators
# have short time constants (flow TFI window ~1 s; basis EWMA α=0.05 ≈ 20
# samples; vol needs VOL_WINDOW_SAMPLES=32 distinct mids), so ~2 min warms
# all three with comfortable margin even in sparse-tick (low-vol) regimes.
#
# The warmup window is carved OUT of the SCORED span. Override per-run via
# ``--warmup-seconds`` (e.g. drop it for very short scenarios where 2 min
# would consume the whole slice), or supply an ``initial_state.json``
# snapshot (:func:`seed_state_from_snapshot`) to start the scored window
# warm with no carve-out at all. Was 0.0 (cold-start every replay) through
# v1.5.309 — the first ~minute of every replay was throwaway (audit §1.2).
DEFAULT_REPLAY_WARMUP_SECONDS: float = 120.0


# ---------------------------------------------------------------------------
# Settings construction
# ---------------------------------------------------------------------------


def make_replay_settings(
    *,
    symbol: str = "TON-USDT-SWAP",
    profile_env_path: Optional[Path] = None,
    overrides: Optional[dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> "Settings":
    """Build a ``Settings`` instance suitable for replay.

    Uses ``UnitTestSettings`` (no dotenv loading; env_prefix isolated)
    so the operator's live environment variables don't leak into the
    replay. The ``DATABASE_URL`` defaults to a unique SQLite path in
    the temp dir — every replay starts with a clean schema.

    ``profile_env_path`` — when supplied, the profile env file is
    parsed and every ``KEY=VALUE`` is folded into the base settings
    BEFORE the ``overrides`` dict. This is critical for replay
    fidelity: the bot's quote engine refuses to quote when
    ``MAX_ABS_POSITION`` × mark price < venue min notional. The
    production profile tunes these for TON's $12 min; without the
    profile values, the bot produces ``below_min_notional`` no-quote
    every tick. Pass ``Path("config/profiles/prod.okx.ton.usdt.perp.env")``
    to replay against the live tunes.

    ``TRADING_ENABLED`` defaults to ``True``; the paper executor
    handles order routing regardless of this flag, but the bot's
    higher-level gates check it.

    Override any setting via the ``overrides`` dict — keys are the
    UPPERCASE env-style names (e.g. ``MAX_ABS_POSITION``).
    """
    from tests.settings_helpers import UnitTestSettings

    if db_path is None:
        db_path = (
            Path(tempfile.gettempdir())
            / f"replay_{os.getpid()}_{uuid.uuid4().hex}.db"
        )
    base: dict[str, Any] = {
        "TRADING_ENABLED": True,  # Let the bot actually run quoting logic.
        "HL_SECRET_KEY": "replay-placeholder",
        "HL_ACCOUNT_ADDRESS": "0xreplay",
        "DATABASE_URL": f"sqlite:///{db_path.as_posix()}",
        "SYMBOL": symbol,
        # Disable daemons / publishers we don't want to fire during
        # replay. Note: not every "disable" knob exists as a Settings
        # field; pydantic ignores unknown env keys (extra="ignore"),
        # so listing them is safe even when they're inert. The list
        # below documents intent.
        "BINANCE_WS_ENABLED": False,
        "PUBLIC_WS_ENABLED": False,
        "ACTION_WS_ENABLED": False,
        # Outbound stays in-process; PaperExecutor handles orders.
        "ACTION_BATCH_INTERVAL_MS": 0.0,
        # Recording lifecycle is irrelevant in replay (we ARE the
        # replay; the recorder doesn't run). Stay off so the
        # /status indicator doesn't show a stale state.
        "RECORDING_ENABLED": False,
    }

    # Fold in the production profile env if supplied. This makes the
    # replay see the same MAX_ABS_POSITION, QUOTE_NOTIONAL_USD,
    # MIN_QUOTE_NOTIONAL_USD, etc. that the live bot uses — without
    # them, the quote engine produces below_min_notional no-quote
    # every tick because the test default MAX_ABS_POSITION=0.05 ×
    # mark $2 = $0.10 << venue min $12.
    if profile_env_path is not None:
        profile_dict = _parse_profile_env(profile_env_path)
        base.update(profile_dict)
        # Profile env's HL_SECRET_KEY would be the real key; we don't
        # need it for paper replay. Overwrite with the placeholder
        # so a replay session never tries to use real venue auth.
        base["HL_SECRET_KEY"] = "replay-placeholder"
        base["HL_ACCOUNT_ADDRESS"] = "0xreplay"
        # Keep our temp DB; profile may carry a host-specific path.
        base["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
        # Replay stays non-recording regardless of profile setting.
        base["RECORDING_ENABLED"] = False

    if overrides:
        base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _parse_profile_env(path: Path) -> dict[str, str]:
    """Parse a ``config/profiles/*.env`` file into a dict.

    Tolerant of comments (``#``), blank lines, and inline trailing
    comments. Values are returned as strings — pydantic handles the
    coercion to bool/int/float per ``Settings``' field definitions.

    Strips surrounding quotes if present (single or double).
    """
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline ``# comment`` AFTER the value (if any). We
        # don't try to be clever about ``#`` inside quoted values;
        # the project's env files don't have those.
        hash_pos = line.find(" #")
        if hash_pos > 0:
            line = line[:hash_pos].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        if key:
            out[key] = val
    return out


# ---------------------------------------------------------------------------
# Operational counters folded into ``extract_gates_fired`` source (1).
# Each entry is ``(report_key, BotState attr, is_bool_flag)``.
# ``is_bool_flag`` maps a bool state flag → {0, 1} (event-count
# semantics for the flat ``gates_fired`` dict). These are bespoke
# ``*_total`` / count attrs with no common registry, so they stay
# enumerated here (unlike the gates, which auto-discover via
# ``gate_attribution_snapshot`` — see BUG-A / v1.5.302).
# ---------------------------------------------------------------------------
#
# v1.5.303 BUG-C audit finding: the original list mapped THREE keys to
# attrs that never existed on BotState — ``place_attempts_total``,
# ``place_rejects_total``, ``cancel_attempts_total`` — and THREE more to
# attrs that are still unwired (``soft_flatten_invocations_total``,
# ``kill_count``, ``silent_wedge_diag_emissions_total``). Under the old
# ``getattr(s, attr, 0)`` these six reported a constant 0 in EVERY replay
# (the exact "metric silently vanishes" failure BUG-C names). Fixed:
#   * place/cancel attempts + place rejects → real ``session_*`` counters
#     (live; see execution.py / state.py increments). Added the parallel
#     ``cancel_rejects`` (``session_cancel_reject_count_total``).
#   * the three unwired keys are removed — no BotState counter exists for
#     soft-flatten invocations, kill invocations (only the ``killed``
#     bool, already mapped), or silent-wedge emissions (only a relog
#     timestamp on the executor). Wiring real session counters for those
#     is a separate instrumentation task, deliberately NOT done here to
#     keep the hot path lean (postmortem-preferred-over-runtime-alerts).
_OPS_COUNTERS: tuple[tuple[str, str, bool], ...] = (
    ("place_attempts", "session_place_attempt_count", False),
    ("place_rejects", "session_place_reject_count_total", False),
    ("cancel_attempts", "session_cancel_attempt_count", False),
    ("cancel_rejects", "session_cancel_reject_count_total", False),
    ("amend_attempts", "amend_intents_emitted_total", False),
    ("amend_flicker_suppressed", "amend_tick_flicker_suppressed_total", False),
    ("amend_rate_throttled", "amend_rate_throttle_suppressed_total", False),
    # Risk / pause state — bool flags folded to {0,1} event counts.
    ("manual_pause_active", "manual_pause", True),
    ("killed", "killed", True),
)

# One-shot WARN dedupe (BUG-C). A renamed/removed counter should log
# ONCE per process, not once per replay/finalize call, and never at
# all on healthy code where every attr exists.
_WARNED_MISSING_OPS_COUNTERS: set[str] = set()


def _warn_missing_ops_counters(missing: list[str]) -> None:
    """Log a one-shot WARN for ops counters absent from BotState (BUG-C).

    Before this, ``extract_gates_fired`` read each counter via
    ``getattr(state, attr, 0)`` — which returns ``0`` whether the attr
    is genuinely zero OR doesn't exist. So renaming a counter on
    BotState made its metric silently report as ``0`` (or vanish),
    with no signal that the wiring had drifted. Now a genuinely-missing
    attr is detected via ``hasattr`` and named here exactly once.

    ``missing`` entries are ``report_key<-state_attr`` strings.
    """
    fresh = [m for m in missing if m not in _WARNED_MISSING_OPS_COUNTERS]
    if not fresh:
        return
    _WARNED_MISSING_OPS_COUNTERS.update(fresh)
    logger.warning(
        "extract_gates_fired: %d operational counter(s) missing from "
        "BotState — metric(s) ABSENT from the report (not silently 0). "
        "A counter was likely renamed/removed; update _OPS_COUNTERS. "
        "Missing: %s",
        len(fresh),
        ", ".join(sorted(fresh)),
    )


# ---------------------------------------------------------------------------
# BotRunner
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BotRunner:
    """Container for one replay's real-Bot wiring.

    Constructed by :func:`build_bot_runner`. Exposes the bot + state
    + storage for inspection from the on-tick callback and the
    final report builder.
    """

    settings: "Settings"
    state: "BotState"
    storage: "Storage"
    bot: "Bot"
    harness: ReplayHarness

    # --------------------------------------------------------------
    # Tick driving
    # --------------------------------------------------------------

    def one_tick(self) -> None:
        """Run one bot tick + drain outbound synchronously.

        Production runs the outbound dispatcher as a daemon thread
        that processes place/cancel/amend intents asynchronously.
        For deterministic replay we instead drain the lanes
        synchronously after every tick — same code path, just on
        the main thread. The PaperExecutor's emitted fills land in
        the shared private-event queue, which the *next* tick's
        ``drain_private_events`` consumes.

        Exceptions are logged but not raised — a replay should
        complete even if a single tick blows up, so the report still
        surfaces what happened up to that point.
        """
        try:
            self.bot.one_tick()
        except Exception:
            logger.exception("bot_one_tick_failed")
        # Drain outbound synchronously after the tick. The dispatcher
        # threads aren't started in replay; we call its private flush
        # method to mirror the worker loop. Loop until empty so multi-
        # side submits all complete in this tick window.
        try:
            outbound = getattr(self.bot._exec, "_outbound", None)
            if outbound is not None:
                # Cap iterations as a safety net — a misbehaving
                # callback that re-enqueues could otherwise spin.
                for _ in range(100):
                    if outbound.queue_depth() == 0:
                        break
                    outbound._flush_one_batch()
        except Exception:
            logger.exception("outbound_drain_failed")

    # --------------------------------------------------------------
    # Observability for the report
    # --------------------------------------------------------------

    def is_quoting(self) -> bool:
        """True if the bot currently has resting orders. Phase 4b's
        proxy for ``time_in_quote_pct``."""
        try:
            return len(self.harness.paper._orders) > 0
        except Exception:
            return False

    def extract_gates_fired(self) -> dict[str, int]:
        """Snapshot the bot's gate-firing + ops counters into one flat
        ``{name: int}`` dict for the report.

        Three sources, folded together. All values are non-negative
        ints and monotonic non-decreasing across a replay (cumulative /
        rising-edge counters only — transient booleans like a gate's
        ``active_now`` are deliberately excluded so the report's
        "counts never go down" contract holds). Counts are cumulative
        across the replay (each fresh BotRunner starts at zero).

        1. **Operational counters** (place / cancel / amend attempts,
           kills, soft-flatten, …). These live as bespoke ``*_total``
           attributes on BotState with no common registry, so they
           stay enumerated (``_OPS_COUNTERS``). BUG-C fix (v1.5.303):
           a genuinely-MISSING attr is no longer silently reported as
           ``0`` — it's detected via ``hasattr``, omitted from the
           report, and named in a one-shot WARN so a renamed counter
           can't vanish unnoticed.

        2. **Per-gate fire counts — auto-discovered (BUG-A fix,
           v1.5.302).** EVERY instrumented gate calls
           ``state.record_gate_firing`` once per tick, so
           ``gate_attribution_snapshot`` already knows them all
           (shock_gate, sf_fatigue_gate, microprice_gate,
           at_touch_adverse_pause, vol_trend_gate, mae_gate,
           inventory_skew, …). We fold each gate's ``fire_count`` in
           under the gate's own name. Before this, the extractor
           hard-coded ~11 names and silently dropped every gate it
           didn't enumerate — so newly-added gates never surfaced in
           replay reports and "adding a counter to a gate" meant
           editing this method.

        3. **AQC controller counters.** ``update_count`` /
           ``safety_floor_engagement_count`` off
           ``state.active_quoting_controller.snapshot_dict()`` (the
           live ``state_current.json`` source), namespaced ``aqc_*``
           since ``update_count`` is too generic to stand alone. These
           are not edge-detected gate firings, so they don't appear in
           ``gate_attribution_snapshot``.
        """
        s = self.state
        out: dict[str, int] = {}

        # (1) Operational counters — bespoke attrs, no common registry,
        # so they stay enumerated. ``hasattr`` distinguishes "attr
        # present and 0" from "attr MISSING" (BUG-C): a missing attr is
        # collected + named in a one-shot WARN, never silently 0.
        missing: list[str] = []
        for report_key, attr, is_bool_flag in _OPS_COUNTERS:
            if not hasattr(s, attr):
                missing.append(f"{report_key}<-{attr}")
                continue
            try:
                raw = getattr(s, attr)
                if is_bool_flag:
                    out[report_key] = 1 if bool(raw) else 0
                elif isinstance(raw, (int, float)):
                    out[report_key] = int(raw)
            except Exception:
                pass
        if missing:
            _warn_missing_ops_counters(missing)

        # (2) Auto-discovered per-gate fire counts. ``now_mono`` only
        # drives in-flight ``fire_seconds_total`` accrual, which we
        # discard here (we read the clock-independent ``fire_count``),
        # so any monotonic value is fine.
        try:
            gate_snap = s.gate_attribution_snapshot(now_mono=time.monotonic())
            for gate_name, entry in gate_snap.items():
                fc = entry.get("fire_count")
                if isinstance(fc, (int, float)):
                    # Don't let a gate name shadow a hard-listed ops
                    # counter (no overlap today; future-proofing).
                    out.setdefault(str(gate_name), int(fc))
        except Exception:
            logger.exception("extract_gates_fired_gate_attribution_failed")

        # (3) AQC controller counters (namespaced).
        try:
            aqc = getattr(s, "active_quoting_controller", None)
            if aqc is not None:
                snap = aqc.snapshot_dict()
                for key in ("update_count", "safety_floor_engagement_count"):
                    v = snap.get(key)
                    if isinstance(v, (int, float)):
                        out[f"aqc_{key}"] = int(v)
        except Exception:
            logger.exception("extract_gates_fired_aqc_failed")

        return out


# ---------------------------------------------------------------------------
# State-mutation callbacks (mirror production main.py wiring)
# ---------------------------------------------------------------------------


def _install_replay_callbacks(
    *,
    runner: BotRunner,
    okx_symbol: str,
    binance_symbol: str,
    basis_alpha: float = 0.05,
) -> None:
    """Register on_bbo / on_trade callbacks on the harness's public
    stream that mirror what production ``main.py`` installs.

    For OKX BBO: ``state.apply_market_book_only`` +
    ``state.market_refresh_note_success`` + wake-quote-loop signal.
    For Binance BBO: update ``state.binance_*`` fields + fold into
    ``state.binance_basis_ewma``.

    For trades: forward to paper executor (already done by the
    Phase 3 callback chain; this preserves that).
    """
    state = runner.state
    storage = runner.storage
    paper = runner.harness.paper

    def _on_bbo(bba: BestBidAsk) -> None:
        symbol = bba.symbol or ""
        if symbol == okx_symbol:
            # Mirror production OKX path. The real OKX public WS
            # handler (``app/exchange/okx_public_ws.py:521-533``)
            # sets ``public_ws_last_message_wall_ts`` +
            # ``public_ws_connected`` + ``public_ws_seen_first_bbo``
            # BEFORE calling ``on_bbo``. The bot's stale-data /
            # market-data-recovery gates read these directly, so
            # without them the bot trips "stale book past kill
            # threshold" within seconds of replay start.
            try:
                now_utc = runner.harness.clock.now_utc()
                with state._lock:
                    state.public_ws_last_message_wall_ts = now_utc
                    state.public_ws_connected = True
                    state.public_ws_seen_first_bbo = True
                state.apply_market_book_only(
                    bba, market_data_source="public_ws", storage=storage
                )
                state.market_refresh_note_success(bba, 0.0)
            except Exception:
                logger.exception("replay_on_bbo_okx_failed")
            # Also feed the paper executor's book tracking.
            paper.process_book_event(
                bid=bba.best_bid if bba.best_bid is not None else 0.0,
                ask=bba.best_ask if bba.best_ask is not None else 0.0,
                bid_size=bba.bid_size if bba.bid_size is not None else 0.0,
                ask_size=bba.ask_size if bba.ask_size is not None else 0.0,
            )
        elif symbol == binance_symbol:
            # Mirror production Binance path — update reference-venue
            # state + basis EWMA. Production
            # ``binance_public_ws.py`` sets connection flags and
            # last-message wall-ts here too; the bot's cross-venue
            # cancel-on-move gate reads them.
            try:
                now_utc = runner.harness.clock.now_utc()
                with state._lock:
                    state.binance_best_bid = bba.best_bid
                    state.binance_best_ask = bba.best_ask
                    state.binance_mid = bba.mid_price
                    state.binance_bid_size = bba.bid_size
                    state.binance_ask_size = bba.ask_size
                    state.binance_last_message_wall_ts = now_utc
                    state.binance_ws_connected = True
                    # Fold into basis EWMA against current OKX mid
                    # (mirrors binance_public_ws.py:316-329).
                    okx_mid = None
                    if (
                        state.market is not None
                        and state.market.mid_price is not None
                    ):
                        try:
                            okx_mid = float(state.market.mid_price)
                        except (TypeError, ValueError):
                            okx_mid = None
                    if (
                        okx_mid is not None
                        and bba.mid_price is not None
                        and bba.mid_price > 0
                    ):
                        basis_raw = okx_mid - bba.mid_price
                        cur = state.binance_basis_ewma
                        if cur is None:
                            state.binance_basis_ewma = basis_raw
                        else:
                            state.binance_basis_ewma = (
                                basis_alpha * basis_raw
                                + (1.0 - basis_alpha) * cur
                            )
            except Exception:
                logger.exception("replay_on_bbo_binance_failed")

    def _on_trade(price: float, size: float, side: str, venue: str) -> None:
        if venue != "okx":
            return
        # Forward to the paper executor's matching engine (used to
        # synthesise fills against our resting orders).
        paper.process_trade_event(price=price, size=size, side=side)
        # v1.5.2 state-seeding: also mirror the production OKX WS
        # handler (``okx_public_ws.py:612-620``) — append to
        # ``state.recent_trades`` + feed ``state.flow_score``. These
        # power the bot's flow-toxicity gate and adverse-side pause
        # logic. Without this, the bot's flow_score stays empty even
        # when trades fire in the recording, and gates that need
        # flow information silently misbehave in replay.
        try:
            from app.enums import Side
            from app.models import TradePrint
            now_utc = runner.harness.clock.now_utc()
            ts_ms = int(now_utc.timestamp() * 1000)
            aggressor = Side.BUY if side.upper() == "BUY" else Side.SELL
            tp = TradePrint(
                ts_exchange_ms=ts_ms,
                ts_local_ms=ts_ms,
                price=price,
                size=size,
                aggressor_side=aggressor,
                trade_id="",
            )
            with state._lock:
                try:
                    state.recent_trades.append(tp)
                except Exception:
                    pass
                try:
                    state.flow_score.record_trade(tp)
                except Exception:
                    pass
        except Exception:
            logger.exception("replay_on_trade_state_update_failed")

    # Late-bind on the public stream — overrides whatever the driver
    # installed (typically a no-op forwarding into paper executor;
    # we replace it with the production-faithful version).
    runner.harness.public_stream._on_bbo = _on_bbo
    runner.harness.public_stream._on_trade = _on_trade


# ---------------------------------------------------------------------------
# Pre-T₀ state seeding (audit P2 #14)
# ---------------------------------------------------------------------------


def seed_state_from_snapshot(
    state: "BotState",
    snapshot: Optional[dict[str, Any]],
    *,
    vol_estimator: Any = None,
    logger_: logging.Logger = logger,
) -> dict[str, int]:
    """Warm-start a fresh replay ``BotState`` from a pre-T₀ snapshot.

    A *scenario* (``app.backtest.scenario_storage``) may carry an
    ``initial_state.json`` — a snapshot of the SOURCE recording's bot
    state captured at the cut boundary (T₀). Seeding the replay bot from
    it lets the scored window ``[T₀, T_end]`` start with already-warm
    market-data estimators, so NONE of the scored window is spent warming
    up. Contrast the ``warmup_seconds`` window, which warms the same
    estimators (events dispatch during warmup — see the module docstring)
    but is carved OUT of the scored span.

    **Contract — ``initial_state.json`` schema.** All keys optional; the
    cutter writes whatever it has, this consumer applies what it
    recognises and ignores the rest. Field names mirror
    ``state_current.json`` so the cutter can copy the source snapshot
    near-verbatim::

        {
          "binance_basis_ewma": <float>,  # OKX-minus-Binance mid EWMA
          "vol_bps": <float>,             # per-step sigma × 1e4; seeds the
                                          #   vol estimator's cold-start
                                          #   window (served until the live
                                          #   estimator warms, then dropped)
          "vol_sigma": <float>            # optional; per-step sigma scalar
        }

    **Deliberately NOT seeded:**

    * ``flow_score`` — the accumulator's TFI window is ~1 s
      (``FLOW_SCORE_TFI_WINDOW_SECONDS``), so it self-warms within the
      first second of replayed trades. Seeding buys nothing.
    * ``recent_fills`` / fill-rate history — the source recording's fills
      belong to the LIVE bot's order flow, which ``--with-bot`` mode
      deliberately gates OUT of the replay bot's event queue
      (``forward_to_sink=False`` in :func:`build_bot_runner`) to avoid
      shadow-position divergence. Injecting them as the replay bot's
      "own" recent fills would misrepresent its fill-rate signal (e.g.
      the no-fill spread-compression path keys off the replay bot's own
      paper fills). The replay bot builds that history once it quotes.

    Best-effort + construction-time only: every field is applied under its
    own ``try/except``, so a malformed snapshot degrades to a partial /
    empty seed and NEVER raises into bot construction. Zero hot-path
    impact (called once, before the first tick). Returns a ``{field: 1}``
    dict of what was applied (for the caller's log line).

    ``vol_estimator`` — the bot's live ``VolatilityEstimator``
    (``bot._vol``). ``vol_bps`` is applied via its ``seed_from_recorder``
    so the seeded sigma serves the cold-start window and is superseded the
    moment the live estimator warms — a warm-START, not an override.
    """
    applied: dict[str, int] = {}
    if not isinstance(snapshot, dict) or not snapshot:
        return applied

    # Binance basis EWMA — directly-settable scalar. The first Binance BBO
    # in the replay folds the raw basis into this with α=0.05, so seeding
    # blends (doesn't reset) — exactly the warm-start we want.
    try:
        v = snapshot.get("binance_basis_ewma")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            state.binance_basis_ewma = float(v)
            applied["binance_basis_ewma"] = 1
    except Exception:
        logger_.exception("seed_state_from_snapshot: basis EWMA seed failed")

    # Volatility — seed the estimator's cold-start sigma (preferred: it
    # survives until the live estimator warms, then this branch stops
    # firing) AND publish vol_bps onto state so the very first tick's
    # vol-conditioned reads see a warm value instead of 0.0.
    try:
        vb = snapshot.get("vol_bps")
        if (
            isinstance(vb, (int, float))
            and math.isfinite(float(vb))
            and float(vb) > 0.0
        ):
            if vol_estimator is not None and hasattr(
                vol_estimator, "seed_from_recorder"
            ):
                if vol_estimator.seed_from_recorder(float(vb)):
                    applied["vol_estimator_seed"] = 1
            state.vol_bps = float(vb)
            applied["vol_bps"] = 1
    except Exception:
        logger_.exception("seed_state_from_snapshot: vol_bps seed failed")

    try:
        vs = snapshot.get("vol_sigma")
        if (
            isinstance(vs, (int, float))
            and math.isfinite(float(vs))
            and float(vs) >= 0.0
        ):
            state.vol_sigma = float(vs)
            applied["vol_sigma"] = 1
    except Exception:
        logger_.exception("seed_state_from_snapshot: vol_sigma seed failed")

    return applied


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_bot_runner(
    *,
    settings: "Settings",
    harness: ReplayHarness,
    okx_symbol: str = "TON-USDT-SWAP",
    binance_symbol: str = "TONUSDT",
    initial_state: Optional[dict[str, Any]] = None,
) -> BotRunner:
    """Wire the real Bot against an existing replay harness.

    The harness's ``private_event_queue`` is the queue the bot's
    executor will drain; both the paper executor's synthetic fills
    and the recorded private events flow through it.

    The harness's clock instance is also used by the bot — so all
    monotonic / wall-clock reads inside the bot follow the replay
    timeline.

    v1.5.2: the PaperExecutor's symbol_spec is replaced with one
    matching the operator's actual venue (OKX TON-USDT-SWAP today).
    The fallback Hyperliquid spec carries ``min_notional_usd=10``
    which clashes with the bot's profile-tuned
    ``MAX_ORDER_NOTIONAL_USD=8`` — quote engine emits
    ``min_notional_exceeds_max_order_notional`` every tick.

    ``initial_state`` (audit P2 #14) — an optional ``initial_state.json``
    snapshot (see :func:`seed_state_from_snapshot`). When present, the
    fresh ``BotState`` is warm-started from it right after construction so
    the scored window starts with already-warm estimators. ``None`` (the
    default) leaves the state cold — the warmup window then warms it
    instead (see :data:`DEFAULT_REPLAY_WARMUP_SECONDS`).
    """
    from app.bot import Bot
    from app.clock import set_module_clock
    from app.exchange.symbol_spec import SymbolSpec
    from app.state import BotState
    from app.storage import Storage

    # Replace the PaperExecutor's fallback SymbolSpec with one that
    # matches the trading venue + symbol. Hard-coded for OKX
    # TON-USDT-SWAP today; can be extended via okx_symbol dispatch
    # when more venues are added.
    if okx_symbol == "TON-USDT-SWAP":
        # OKX values from `okx_alibaba_sg.py` profile + 2026-05-22
        # operator notes:
        # * contract_value = 1 TON (size_step=1, integer contracts)
        # * price_tick = 0.0001 USDT
        # * min_size = 1 contract
        # * min_notional ≈ 1 × current TON price (~$2 mid 2026Q2)
        harness.paper._symbol_spec = SymbolSpec(
            price_tick=0.0001,
            size_step=1.0,
            min_size=1.0,
            min_notional_usd=2.0,  # 1 contract × ~$2 TON
            sz_decimals=0,
            source="fallback",
        )
        harness.paper._symbol = okx_symbol

    # --with-bot contract: the real Bot drains harness.private_event_queue,
    # and harness.paper (PaperExecutor) is the AUTHORITATIVE source of
    # private events for the bot's own simulated orders. Stop the recorded
    # private stream from also pushing the ORIGINAL live session's
    # order-updates/fills into that same queue — otherwise the bot ingests
    # foreign-account state, its shadow position diverges from paper-exec's
    # true (bot-only) position, and we get the shadow_position_divergence →
    # private_ws_overflow_recovery → reconcile/orphan/side-unresolved storm
    # diagnosed in the v1.5.294 audit. The stream still parses + counts the
    # recorded events (surfaced as *_suppressed in the driver report); it
    # just doesn't contaminate the bot's queue.
    harness.private_stream.forward_to_sink = False

    # Install the replay clock as the module-level clock BEFORE
    # constructing BotState — ``state.session_started_at_utc``
    # captures the module clock's ``now_utc()`` at __init__ time,
    # and we want that anchored to the replay's start instant, not
    # the operator's wall clock. Otherwise synthetic fills from
    # PaperExecutor would land "before the session" (replay clock
    # = 2023 vs system clock = now) and ``_fill_ts_is_session_scoped``
    # would treat them as pre-session, skipping PnL accumulation.
    set_module_clock(harness.clock)
    state = BotState(settings)
    storage = Storage(settings)
    storage.init_schema()

    bot = Bot(
        settings,
        state,
        harness.paper,
        storage,
        private_event_queue=harness.private_event_queue,
        private_stream=harness.private_stream,
        public_stream=harness.public_stream,
        clock=harness.clock,
        # exit_fn no-op: replay must not call os._exit on kill — we
        # want to observe the kill event in the report.
        exit_fn=lambda _code: None,
        notifier=None,
    )

    # Warm-start the fresh state from a scenario's initial_state.json, if
    # supplied (audit P2 #14). Inert when ``initial_state`` is None — the
    # warmup window warms the same estimators instead. Best-effort: the
    # seed helper swallows per-field errors and never raises into
    # construction.
    if initial_state is not None:
        try:
            seeded = seed_state_from_snapshot(
                state,
                initial_state,
                vol_estimator=getattr(bot, "_vol", None),
            )
            if seeded:
                logger.info(
                    "build_bot_runner: warm-started replay state from "
                    "initial_state.json snapshot: %s",
                    ", ".join(sorted(seeded)),
                )
        except Exception:
            logger.exception("build_bot_runner_state_seed_failed")

    runner = BotRunner(
        settings=settings,
        state=state,
        storage=storage,
        bot=bot,
        harness=harness,
    )

    _install_replay_callbacks(
        runner=runner,
        okx_symbol=okx_symbol,
        binance_symbol=binance_symbol,
    )

    return runner


__all__ = [
    "DEFAULT_REPLAY_WARMUP_SECONDS",
    "BotRunner",
    "build_bot_runner",
    "make_replay_settings",
    "seed_state_from_snapshot",
]
