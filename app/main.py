from __future__ import annotations

import logging
import os
import queue
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from app import __version__
from app.api import router as api_router
from app.bot import Bot, start_bot_thread
from app.config import Settings, require_trading_credentials_when_enabled
from app.env_bootstrap import ensure_env_bootstrapped, env_file_startup_log_line
from app.operator_metrics_reconcile import try_reconcile_operator_metrics_from_exchange_fills
from app.persistent_runtime_io import (
    try_load_persistent_runtime_state,
    try_save_persistent_runtime_state,
)
from app.enums import BotStatus, EventSeverity
from app.exchange.factory import (
    build_adapter,
    build_private_stream,
    build_public_stream,
    venue_account_address,
)
from app.market_data import instrument_book_update_success
from app.models import BestBidAsk
from app.session_resume import (
    try_resume_session_from_storage,
    write_bot_start_event,
)
from app.state import BotState
from app.startup_report import log_runtime_config, log_startup_readiness
from app.storage import Storage
from app.telegram_commands import TelegramCommandPoller
from app.telegram_notifier import TelegramNotifier
from app.utils.logging import setup_logging
from app.utils.time import utc_now_iso

from app import clock as _clock

logger = logging.getLogger(__name__)

# Repository root (parent of ``app/``): optional ``favicon.ico`` for browsers.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FAVICON_PATH = _REPO_ROOT / "favicon.ico"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_env_bootstrapped(argv=None)
    settings = Settings()
    require_trading_credentials_when_enabled(settings)
    setup_logging(settings.log_level)
    logger.info("%s", env_file_startup_log_line())
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    try_load_persistent_runtime_state(settings, state)

    # Soft-flatten resume: if the persistent state says we were in
    # the middle of a patient post-only flatten when the process
    # crashed/restarted, transition immediately to SOFT_FLATTENING
    # so the bot resumes the close-out instead of normal quoting on
    # adverse inventory. ``apply_persistent_runtime_state`` already
    # set ``soft_flatten_active=True``; just sync the bot_status.
    if state.soft_flatten_active:
        with state._lock:
            state.bot_status = BotStatus.SOFT_FLATTENING
        logger.warning(
            "soft_flatten_resumed_from_persistent_state -- "
            "patient post-only flatten was in progress at last "
            "shutdown; resuming. Position will be re-checked from "
            "venue after first REST refresh."
        )

    # Telegram subsystem — disabled when no token. Notifier first
    # (publishes both startup and resume status), then command poller
    # (started after the bot is up so its handlers have something to act on).
    telegram_notifier = TelegramNotifier(settings)
    telegram_notifier.start()

    # S3 heartbeat publisher -- writes a tiny JSON every ~30s so the
    # operator dashboard can show bot session uptime distinct from
    # EC2 host uptime. Strictly best-effort; ``maybe_create`` returns
    # None when LOGS_BUCKET is unset or boto3 is missing, in which
    # case the bot still trades fine just without the dashboard chip.
    from app.heartbeat import HeartbeatPublisher
    import os as _os
    _profile_path = _os.environ.get("APP_ENV_FILE", "") or ""
    _profile_name = (
        _os.path.splitext(_os.path.basename(_profile_path))[0]
        if _profile_path
        else ""
    ) or "unknown"
    heartbeat = HeartbeatPublisher.maybe_create(
        settings, state, _profile_name
    )
    if heartbeat is not None:
        heartbeat.start()

    # Live trading internals -- richer / faster cadence than heartbeat.
    # Powers the dashboard's Bot Stats tab. Same daemon-thread pattern
    # as heartbeat; trading impact is zero. Best-effort: failures
    # (no LOGS_BUCKET, boto3 missing, IAM denial) silently disable
    # without affecting trading.
    from app.live_stats import LiveStatsPublisher

    live_stats = LiveStatsPublisher.maybe_create(
        settings, state, _profile_name, storage=storage
    )
    if live_stats is not None:
        live_stats.start()

    # Equity-history publisher (1.1.48): periodic S3 dump of the
    # session's equity samples so the dashboard's Bot Stats panel
    # can render the PnL path + HWM + drawdown trough. Cadence 60s,
    # payload typically 10-50 KB. Best-effort like the other S3
    # publishers — disable conditions are logged but never fatal.
    from app.equity_history_publisher import EquityHistoryPublisher

    equity_history_publisher = EquityHistoryPublisher.maybe_create(
        settings, state, _profile_name, storage=storage
    )
    if equity_history_publisher is not None:
        equity_history_publisher.start()

    # Dashboard-state publisher (1.3.31, todo-028 / todo-032 prereq):
    # periodic S3 dump of the three session-scoped tables the
    # dashboard's Inventory / Gates / Execution-quality / Regimes
    # panels read from — ``exposure_bars``, ``fills``,
    # ``orders_lifecycle``. Cadence 30s, payload typically a few MB
    # combined on a busy session. Same disable-on-misbehaviour
    # contract as the rest of the S3 publishers.
    from app.dashboard_state_publisher import DashboardStatePublisher

    dashboard_state_publisher = DashboardStatePublisher.maybe_create(
        settings, state, _profile_name, storage=storage
    )
    if dashboard_state_publisher is not None:
        dashboard_state_publisher.start()

    # 2026-05-13 regime-observability Phase 2: exposure-bar emitter.
    # Periodic (default 5s) snapshot of market + strategy state, written
    # to the new ``exposure_bars`` table. Provides the exposure
    # denominator that turns per-regime fill outcomes into meaningful
    # ratios. Daemon thread, lock-free reads — hot-path-invisible.
    # Disable via ``OBSERVABILITY_EXPOSURE_BARS_ENABLED=false`` if
    # any unforeseen impact appears.
    from app.exposure_bar_emitter import ExposureBarEmitter

    exposure_bar_emitter = ExposureBarEmitter.maybe_create(
        settings, state, storage
    )
    if exposure_bar_emitter is not None:
        exposure_bar_emitter.start()
        app.state.exposure_bar_emitter = exposure_bar_emitter

    # 2026-05-13 regime-observability Phase 4c: post-fill excursion
    # watcher. Tracks MAE / MFE for each fill over 5s and 30s windows.
    # Default-disabled (``OBSERVABILITY_MAE_MFE_ENABLED=false``) — opt
    # in per profile after the baseline is stable.
    from app.post_fill_excursion_watcher import PostFillExcursionWatcher

    post_fill_excursion_watcher = PostFillExcursionWatcher.maybe_create(
        settings, state, storage
    )
    if post_fill_excursion_watcher is not None:
        post_fill_excursion_watcher.start()
        # Attach to ``state`` so ``fill_ingestion.ingest_hl_fill_raw``
        # can call ``watcher.track_fill(...)`` when a new fill lands.
        state.post_fill_excursion_watcher = post_fill_excursion_watcher
        app.state.post_fill_excursion_watcher = post_fill_excursion_watcher

    # 2026-05-13 Phase 4c follow-up: time-to-flat watcher. Tracks
    # how long after each fill the bot's position returns to zero —
    # the inventory-exposure dimension that MAE/MFE doesn't capture.
    # Default-disabled like MAE/MFE; flip
    # ``OBSERVABILITY_TIME_TO_FLAT_ENABLED=true`` per profile to opt in.
    from app.post_fill_time_to_flat_watcher import PostFillTimeToFlatWatcher

    post_fill_time_to_flat_watcher = PostFillTimeToFlatWatcher.maybe_create(
        settings, state, storage
    )
    if post_fill_time_to_flat_watcher is not None:
        post_fill_time_to_flat_watcher.start()
        state.post_fill_time_to_flat_watcher = post_fill_time_to_flat_watcher
        app.state.post_fill_time_to_flat_watcher = post_fill_time_to_flat_watcher

    # Config publisher (one-shot): dumps raw APP_ENV_FILE (with
    # secret values redacted) plus resolved Settings to
    # ``s3://<logs_bucket>/config/<profile>.json`` exactly once at
    # startup. Powers the Bot Stats panel's CONFIG tab.
    from app.config_publisher import ConfigPublisher

    config_publisher = ConfigPublisher.maybe_create(
        settings, state, _profile_name, env_file_path=_profile_path
    )
    if config_publisher is not None:
        config_publisher.start()

    # Best-effort cumulative-metric resume from storage. Reads the
    # most-recent ``bot_start`` event and re-aggregates fills since
    # then; falls through cleanly to fresh start on any failure or if
    # the prior session is older than the continuity window.
    # IMPORTANT: only METRIC state is restored — no pending-cancel
    # dicts, gates, toxicity windows, or position cache. See
    # ``app/session_resume.py`` and the regression test in
    # ``tests/test_session_resume.py``.
    resume_result = try_resume_session_from_storage(settings, state, storage)
    logger.info(
        "session_resume status=%s reason=%s",
        resume_result.status,
        resume_result.reason,
    )
    # Write the new bot_start event AFTER resume — its ts/payload becomes
    # the anchor for the next startup's resume attempt.
    write_bot_start_event(
        storage,
        session_id=state.session_id,
        session_started_at_utc=state.session_started_at_utc,
        version=__version__,
        extra={"resume_status": resume_result.status},
    )
    # Telegram heads-up on resume outcome.
    if telegram_notifier.enabled:
        sev = "INFO" if resume_result.status in {"resumed", "fresh"} else "WARNING"
        telegram_notifier.notify_ops(
            sev,
            "session_resume",
            resume_result.telegram_summary(),
            {
                "status": resume_result.status,
                "session_id": state.session_id,
                "session_started_at_utc": state.session_started_at_utc.isoformat(),
            },
        )

    client = build_adapter(settings)
    logger.info("exchange_adapter_selected venue=%s", settings.exchange)
    if settings.trading_enabled and not client.symbol_spec_fetched_ok:
        raise RuntimeError(
            f"{settings.exchange} symbol metadata could not be loaded (see logs). "
            "Refusing to start with TRADING_ENABLED=true; fix network/SYMBOL or exchange availability."
        )
    # 2026-05-13 regime-observability bugfix: attach symbol_spec to
    # state so ``ExposureBarEmitter._capture_bar`` can compute
    # ``bid_distance_ticks`` / ``ask_distance_ticks`` lock-free. The
    # emitter doesn't have a reference to ``client``; this push lets
    # it read ``state.symbol_spec.price_tick`` directly.
    try:
        state.symbol_spec = client.symbol_spec  # type: ignore[attr-defined]
    except Exception:
        # Defensive: missing/null symbol_spec is non-fatal; the
        # emitter falls back to None for the tick distances.
        pass
    log_startup_readiness(settings, client)
    log_runtime_config(settings)

    # Best-effort fetch of venue-side account / per-symbol settings
    # for the dashboard position panel: leverage, margin mode,
    # position mode. Strictly informational; failures (non-OKX
    # adapters, missing methods, transient REST errors) silently
    # leave the fields as None so the dashboard renders "—". Never
    # blocks trading.
    try:
        if hasattr(client, "fetch_account_config"):
            cfg = client.fetch_account_config()  # type: ignore[attr-defined]
            pos_mode = (cfg or {}).get("posMode")
            if pos_mode:
                state.venue_position_mode = str(pos_mode)
    except Exception:
        logger.exception("startup_fetch_account_config_failed")
    try:
        if hasattr(client, "fetch_leverage_info"):
            # Cross is the bot's standard mode; if the operator put
            # the symbol on isolated, the cross response will have an
            # empty data array and we just swallow it. Future: query
            # both, prefer the one with non-empty rows.
            lv = client.fetch_leverage_info(  # type: ignore[attr-defined]
                settings.symbol, "cross"
            )
            if lv:
                state.venue_leverage = str(lv.get("lever") or "") or None
                state.venue_margin_mode = (
                    str(lv.get("mgnMode") or "") or None
                )
    except Exception:
        logger.exception("startup_fetch_leverage_info_failed")

    if settings.fast_start_skip_historical_fill_replay:
        logger.info(
            "operator_metrics_reconcile skipped reason=fast_start_skip_historical_fill_replay"
        )
    else:
        try_reconcile_operator_metrics_from_exchange_fills(settings, state, client)

    # v1.4.203 — startup ordering refactor. Pre-v1.4.203 the streams
    # were ``.start()``-ed BEFORE the cancel-all REST call ran. The
    # private WS thus received cancel-terminals for pre-existing
    # orphan orders from the previous process; those events arrived
    # in the queue with no local working orders to match against,
    # ticking ``ws_event_unmatched_to_local_wo_total`` and failing the
    # wedge-acceptance gate's strict-zero check on routine restarts.
    #
    # Post-v1.4.203 order:
    #   1. BUILD streams but do NOT .start() them.
    #   2. BUILD Bot.
    #   3. ``run_startup_cleanup`` — REST cancel-all + REST poll-
    #      until-book-empty + reset startup-grace counters.
    #   4. NOW .start() the streams (private + public + binance/bybit).
    #   5. Position seed, telegram poller, FastAPI, bot trading loop.
    #
    # The result: at the moment the private WS subscribes, the
    # symbol's book is empty (REST-confirmed). No orphan-cancel
    # terminals can land. Validation counters genuinely start at zero.
    #
    # Operator directive (2026-05-21): "initial all cancellations
    # must fully complete before the trade starts and any counting
    # begins. Startup cleanup must be reliable and complete, ONLY
    # AFTER that bot starts, validation counters start."

    private_q: queue.Queue | None = None
    private_stream = None
    addr = venue_account_address(settings)
    if addr and settings.private_ws_enabled:
        private_q = queue.Queue(maxsize=settings.private_ws_queue_max)
        private_stream = build_private_stream(
            settings,
            state,
            addr,
            private_q,
            on_queue_drop=state.note_private_ws_queue_drop,
            adapter=client,
        )
        # v1.4.203: do NOT start yet — see ordering rationale above.

    public_stream = None
    stream_cell: list[Any] = [None]

    def _on_public_bbo(bb: BestBidAsk) -> None:
        state.apply_market_book_only(
            bb, market_data_source="public_ws", storage=storage
        )
        out = state.market_refresh_note_success(bb, 0.0)
        ps = stream_cell[0]
        instrument_book_update_success(
            state,
            storage,
            sym=settings.symbol,
            bb=bb,
            latency_ms=0.0,
            out=out,
            on_stall_reconnect=(ps.request_reconnect if ps is not None else None),
        )
        # v1.5.215 — feed the Phase 8D OFI accumulator (Cont-Kukanov-
        # Stoikov 2014 order-flow imbalance). One call per BBO update;
        # accumulator handles the seed-then-diff state machine + the
        # two EWMAs internally. Safe to call unconditionally even when
        # OFI_RESERVATION_ALPHA=0.0 — the alpha gate is checked at the
        # consumer (compute_quote_decision), not here.
        try:
            if (
                bb.best_bid is not None and bb.best_ask is not None
                and bb.bid_size is not None and bb.ask_size is not None
            ):
                state.ofi.record_bbo(
                    bid_px=float(bb.best_bid),
                    bid_sz=float(bb.bid_size),
                    ask_px=float(bb.best_ask),
                    ask_sz=float(bb.ask_size),
                )
        except Exception:
            # OFI is observability-class; never let it break the hot path.
            logger.exception("ofi_record_bbo_failed")
        state.wake_quote_loop()

    if settings.public_ws_enabled:
        public_stream = build_public_stream(
            settings,
            state,
            settings.symbol,
            _on_public_bbo,
        )
        stream_cell[0] = public_stream
        # v1.4.203: do NOT start yet.

    # Cross-venue reference (Level 1). ``REFERENCE_EXCHANGE`` selects
    # which public-WS feed to subscribe to:
    #   binance (default, back-compat) — ``BinancePublicStream``
    #   bybit                          — ``BybitPublicStream`` (Singapore
    #                                     latency ~40 ms better than
    #                                     Binance Tokyo; see config.py)
    #   off                            — disable entirely
    # Both impls write to the same ``state.binance_*`` fields so the
    # cross-venue-cancel path in ``execution.py`` is venue-agnostic.
    # The legacy ``BINANCE_WS_ENABLED=false`` also disables everything.
    binance_stream: Any = None
    if settings.binance_ws_enabled and settings.reference_exchange != "off":
        if settings.reference_exchange == "bybit":
            from app.exchange.bybit_public_ws import BybitPublicStream
            binance_stream = BybitPublicStream(
                settings,
                state,
                on_bbo_callback=state.wake_quote_loop,
            )
        else:
            from app.exchange.binance_public_ws import BinancePublicStream
            binance_stream = BinancePublicStream(
                settings,
                state,
                on_bbo_callback=state.wake_quote_loop,
            )
        # v1.4.203: do NOT start yet (the reference-venue feed is the
        # least sensitive to the orphan-terminal race, but we hold it
        # to the same ordering for consistency).

    bot = Bot(
        settings,
        state,
        client,
        storage,
        private_event_queue=private_q,
        private_stream=private_stream,
        public_stream=public_stream,
        notifier=telegram_notifier,
    )

    # ----------------------------------------------------------------
    # STEP 3 — REST-only startup cleanup. Runs BEFORE any WS subscribe.
    # ----------------------------------------------------------------
    if settings.cancel_all_on_startup and settings.trading_enabled and client.has_write_access() and addr:
        from app.startup_cleanup import run_startup_cleanup

        run_startup_cleanup(
            bot=bot,
            client=client,
            settings=settings,
            address=addr,
            storage=storage,
        )

    # ----------------------------------------------------------------
    # STEP 4 — NOW start the WS subscriptions. The book is clean
    # (REST-confirmed empty for our symbol); no orphan-cancel
    # terminals can land in the private queue.
    # ----------------------------------------------------------------
    if private_stream is not None:
        private_stream.start()
    if public_stream is not None:
        public_stream.start()
    if binance_stream is not None:
        binance_stream.start()

    logger.info(
        "startup_policy inherit_exchange_position=true cancel_all_on_startup=%s "
        "(no automatic flatten on nonzero inventory; bot stays STARTING until first healthy reconcile)",
        settings.cancel_all_on_startup,
    )

    # --- BUG-003 fix: explicit position seed at startup ---
    # The bot was previously relying on the first quote-tick's
    # ``refresh_account_only`` to seed ``state.position`` from venue
    # truth. For an *inherited* position with no recent activity, that
    # path raced (or the parser returned 0 silently) and the bot ran
    # blind: state.position.position_qty = 0 while the venue held real
    # inventory. Both ``MAX_ABS_POSITION`` and ``MAX_POSITION_NOTIONAL_USD``
    # caps read this stale 0 and never fired.
    #
    # Observed 2026-04-25 in tmp/snap_20260425_193320: state position 0
    # while Bluefin UI showed +336 SUI long; bot quoted both sides
    # despite caps that should have been forcing ASK_ONLY.
    #
    # Fix: explicit ``client.fetch_position`` call here, log raw + applied
    # values, alert operator via Telegram + storage event when inherited
    # non-zero. Failure modes (REST error, parser zeroing) all logged
    # explicitly so the operator can diagnose without snapshot diving.
    addr = venue_account_address(settings)
    if addr and client.has_write_access():
        try:
            seeded_pos = client.fetch_position(addr, settings.symbol)
        except Exception:
            logger.exception(
                "position_seed_fetch_failed; bot may operate with "
                "stale position state until first refresh"
            )
            seeded_pos = None
        if seeded_pos is not None:
            try:
                state.apply_account_position_only(seeded_pos, None)
            except Exception:
                logger.exception("position_seed_apply_failed")
            try:
                qty_state = float(
                    getattr(state.position, "position_qty", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                qty_state = 0.0
            # TODO-001: anchor the inventory consistency watchdog at the
            # seeded position so subsequent session fills can be reconciled
            # against the venue truth.
            state.set_inventory_baseline_at_session_start(qty_state)
            qty_venue = float(getattr(seeded_pos, "position_qty", 0.0) or 0.0)
            mark = getattr(seeded_pos, "mark_price", None)
            notional = float(getattr(seeded_pos, "position_notional", 0.0) or 0.0)
            logger.info(
                "position_seed_applied venue_qty=%.6f state_qty=%.6f "
                "mark=%s notional=%.4f symbol=%s",
                qty_venue,
                qty_state,
                mark,
                notional,
                settings.symbol,
            )
            try:
                storage.insert_bot_event(
                    utc_now_iso(),
                    EventSeverity.INFO.value,
                    "position_seed_applied",
                    f"position seeded at startup: venue_qty={qty_venue:.4f} "
                    f"state_qty={qty_state:.4f} symbol={settings.symbol}",
                    {
                        "venue_qty": qty_venue,
                        "state_qty": qty_state,
                        "mark": mark,
                        "notional": notional,
                        "symbol": settings.symbol,
                    },
                )
            except Exception:
                logger.exception("position_seed_event_write_failed")
            # Inherited non-zero position: surface to the operator so
            # they can decide whether it's intentional. The risk caps
            # are now active (state matches venue), so the bot is safe;
            # the alert is informational, not actionable in itself.
            if abs(qty_state) >= 1e-8:
                msg = (
                    f"inherited non-zero position at startup: "
                    f"{qty_state:+.4f} {settings.symbol} "
                    f"(notional ${notional:.2f}). risk caps now active."
                )
                try:
                    storage.insert_bot_event(
                        utc_now_iso(),
                        EventSeverity.WARNING.value,
                        "position_seed_inherited",
                        msg,
                        {
                            "position_qty": qty_state,
                            "symbol": settings.symbol,
                            "notional": notional,
                        },
                    )
                except Exception:
                    logger.exception("position_seed_inherited_event_failed")
                if telegram_notifier.enabled:
                    try:
                        telegram_notifier.notify_ops(
                            "WARNING",
                            "position_seed_inherited",
                            msg,
                            {
                                "position_qty": qty_state,
                                "symbol": settings.symbol,
                                "notional": notional,
                            },
                        )
                    except Exception:
                        logger.exception(
                            "position_seed_inherited_telegram_failed"
                        )

    # Phase F1 (v1.5.42) — write bot session-start forensic manifest
    # into the recorder's session dir if the recorder is up. Best-effort:
    # every step is wrapped so a forensic-artifact write failure never
    # aborts bot startup. Skipped entirely when the recorder is disabled
    # (no pointer file → ``read_recorder_pointer`` returns None).
    # See ``backtesting/docs/execution-plan.md`` §Phase F for the
    # full spec (originally ``plans/forensics.md``, merged + archived
    # to plans/_DONE/ on 2026-05-23).
    try:
        from app import __version__ as _bot_version
        from app.session_manifest import (
            read_recorder_pointer,
            write_bot_manifest,
        )

        _rec_session_dir = read_recorder_pointer(_profile_name)
        if _rec_session_dir is not None:
            write_bot_manifest(
                session_dir=_rec_session_dir,
                state=state,
                settings=settings,
                env_file_path=(
                    Path(_profile_path) if _profile_path else None
                ),
                bot_version=_bot_version,
                bot_profile=_profile_name,
            )
        else:
            logger.info(
                "session_manifest_skipped: recorder pointer absent "
                "(profile=%s) — non-forensic-grade session",
                _profile_name,
            )
    except Exception:
        logger.exception("session_manifest_write_failed")

    # Wire fill observer for Telegram trades-channel notifications. The
    # notifier's notify_trade_fill is non-blocking (enqueue only), so it
    # is safe to call from the fill ingestion path.
    if telegram_notifier.enabled:
        def _on_fill(f: Any) -> None:
            try:
                telegram_notifier.notify_trade_fill(
                    {
                        "symbol": getattr(f, "symbol", ""),
                        "side": getattr(getattr(f, "side", None), "value", str(getattr(f, "side", ""))),
                        "price": getattr(f, "price", None),
                        "size": getattr(f, "size", None),
                        "notional": getattr(f, "notional", None),
                        "fee": getattr(f, "fee", None),
                        "closed_pnl": getattr(f, "closed_pnl", None),
                        "markout_5s_bps": None,  # populated later by markout job
                    }
                )
            except Exception:
                logger.exception("telegram_fill_observer_failed")
        state.attach_fill_observer(_on_fill)

    _, stop_bot = start_bot_thread(bot)

    # Deadlock watchdog — starts after the bot thread so the state it
    # observes has been initialised. Daemon thread; no explicit stop on
    # shutdown (exits with the process). See app/watchdog.py for the
    # deadlock pattern this guards against. The pre-exit hook gives the
    # operator a 1-shot Telegram alert before the process dies, so the
    # ops channel sees the watchdog firing even if the next bot session
    # never starts (auth failure, OOM, etc.).
    from app.watchdog import Watchdog

    def _on_watchdog_pre_exit() -> None:
        if not telegram_notifier.enabled:
            return
        # Snapshot the diagnostic state at the moment the watchdog
        # fires. Telegram is the only place this lands in the
        # operator's eyeline; the same data is in the deploy host's
        # logs but the operator shouldn't have to dig there for every
        # event.
        # The fields below mirror what's in the
        # ``watchdog_deadlock_detected`` log line plus contextual
        # state (bot_status, ws health, pending-cancel count) that
        # together identify the most-likely stuck pattern at a glance.
        now_mono = _clock.monotonic()
        quote_ts = float(
            getattr(state, "last_quote_engine_non_hold_ts_mono", 0.0) or 0.0
        )
        place_ts = float(
            getattr(state, "last_place_attempt_ts_mono", 0.0) or 0.0
        )
        quote_idle_s = (
            round(now_mono - quote_ts, 1) if quote_ts > 0 else None
        )
        place_idle_s = (
            round(now_mono - place_ts, 1) if place_ts > 0 else None
        )
        # Pending-cancel snapshot — Bluefin-specific, but other
        # adapters (or none) just return an empty list, so the
        # call is always safe. A stuck pending-cancel is the most
        # common cause of an execution_idle deadlock on Bluefin.
        pending_cancel_count: Optional[int] = None
        try:
            snap = client.pending_cancel_snapshot()  # type: ignore[attr-defined]
            if isinstance(snap, list):
                pending_cancel_count = len(snap)
        except Exception:  # noqa: BLE001
            pending_cancel_count = None
        # Status flags via the same helper the API + Telegram /status
        # commands use.
        try:
            flags = state.status_flags_dict()
            bot_status = flags.get("bot_status")
            killed = bool(flags.get("killed", False))
            manual_pause = bool(flags.get("manual_pause", False))
            flatten_mode = bool(flags.get("flatten_mode", False))
        except Exception:  # noqa: BLE001
            bot_status = None
            killed = manual_pause = flatten_mode = False
        payload = {
            "exit_code": 42,
            "session_id": state.session_id,
            "quote_engine_idle_s": quote_idle_s,
            "execution_idle_s": place_idle_s,
            "threshold_no_place_s": float(
                getattr(settings, "watchdog_no_place_attempt_seconds", 600.0)
            ),
            "threshold_quote_window_s": float(
                getattr(
                    settings, "watchdog_quote_activity_window_seconds", 120.0
                )
            ),
            "bot_status": bot_status,
            "killed": killed,
            "manual_pause": manual_pause,
            "flatten_mode": flatten_mode,
            "session_fill_count": int(
                getattr(state, "session_fill_count", 0)
            ),
            "session_place_attempt_count": int(
                getattr(state, "session_place_attempt_count", 0)
            ),
            "pending_cancel_count": pending_cancel_count,
            "private_ws_connected": bool(
                getattr(state, "private_ws_connected", False)
            ),
            "private_ws_healthy": bool(
                getattr(state, "private_ws_healthy", False)
            ),
            "private_ws_reconnect_count": int(
                getattr(state, "private_ws_reconnect_count", 0)
            ),
        }
        # One-line headline that humans can read in 2 seconds. The
        # full payload (above) lands as JSON in the same Telegram
        # message body — no need to dig into the deploy host's logs
        # for the basics.
        msg = (
            f"deadlock watchdog fired (exit 42)\n"
            f"quote_engine_idle={quote_idle_s}s "
            f"(threshold <={payload['threshold_quote_window_s']:.0f}s = ENGINE-ALIVE)\n"
            f"execution_idle={place_idle_s}s "
            f"(threshold >={payload['threshold_no_place_s']:.0f}s = STUCK)\n"
            f"bot_status={bot_status} pending_cancels={pending_cancel_count} "
            f"ws_healthy={payload['private_ws_healthy']}\n"
            f"fills={payload['session_fill_count']} "
            f"places={payload['session_place_attempt_count']}"
        )
        telegram_notifier.notify_ops_blocking(
            "CRITICAL",
            "watchdog_fired",
            msg,
            payload,
        )

    watchdog = Watchdog(settings, state, on_pre_exit=_on_watchdog_pre_exit)
    watchdog.start()

    # Telegram inbound command poller. Started AFTER the bot+watchdog
    # are up so /status / /pause / /flatten have something live to act on.
    telegram_commands = TelegramCommandPoller(
        settings,
        bot=bot,
        state=state,
        storage=storage,
        notifier=telegram_notifier,
        # Pass the venue client so the poller's background refresher can
        # populate the /status volume cache (7d / 30d). Optional: poller
        # tolerates a missing client or one that doesn't expose
        # ``fetch_account_volume_usd`` (only Bluefin does today).
        client=client,
    )
    telegram_commands.start()

    app.state.settings = settings
    app.state.bot_state = state
    app.state.storage = storage
    app.state.client = client
    app.state.bot = bot
    app.state.watchdog = watchdog
    app.state.telegram_notifier = telegram_notifier
    app.state.telegram_commands = telegram_commands

    logger.info(
        "application_started venue=%s symbol=%s trading=%s",
        settings.exchange,
        settings.symbol,
        settings.trading_enabled,
    )

    if telegram_notifier.enabled:
        telegram_notifier.notify_ops(
            "INFO",
            "bot_start",
            f"bot started venue={settings.exchange} symbol={settings.symbol} "
            f"trading={settings.trading_enabled} version={__version__}",
            {
                "session_id": state.session_id,
                "session_started_at_utc": state.session_started_at_utc.isoformat(),
                "resume_status": resume_result.status,
            },
        )

    yield

    # --- graceful shutdown ---
    # Order matters here. We want orders off the book BEFORE we tear down
    # streams and storage, and we want to avoid a race against our own
    # still-in-flight placements. Two concrete hazards on GRVT:
    #
    #   Race A: ``stop_bot()`` only signals the bot thread — a
    #           ``place_post_only_limit`` REST call already in flight can
    #           complete after cancel_all returns, leaving an order on the
    #           book. Mitigation: wait for the outbound dispatcher to drain
    #           before the first cancel sweep.
    #
    #   Race B: GRVT's gateway returns 200 OK before the matching engine
    #           has indexed the order. ``/open_orders`` will miss it until
    #           a few hundred ms later, so a single cancel sweep can
    #           under-report. Mitigation: run the cancel sweep TWICE with a
    #           short delay — the second pass catches anything that
    #           materialised after the first ``open_orders`` fetch.
    #
    # The symbol-scoping is enforced inside ``fetch_open_orders_raw`` on
    # each adapter: GRVT filters by ``self._symbol`` before returning, so
    # ``cancel_all_orders`` only touches orders for the currently-traded
    # symbol — other symbols / manual orders on the account are left alone.

    stop_bot()
    try:
        bot._exec.wait_transport_idle(timeout_s=2.0)
    except Exception:
        logger.exception("shutdown wait_transport_idle failed")

    if settings.cancel_all_on_shutdown and settings.trading_enabled and client.has_write_access():
        pass_outcomes: list[str] = []
        for pass_idx in range(2):
            outcome = "exception"
            try:
                # Symbol-scoped — only orders for ``settings.symbol`` are
                # cancelled. Other bot instances / manual orders on the
                # same account are preserved. Uses the venue's single-request
                # bulk cancel when available (GRVT); this drops a two-pass
                # shutdown from ~4 s (N+1 REST round trips × 2) to
                # ~200-300 ms total — a huge reliability win under any
                # supervisor's SIGTERM → SIGKILL grace window (PaaS hosts
                # typically give 10-30 s; systemd defaults to 90 s).
                outcome = bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback()
            except Exception:
                logger.exception("shutdown cancel_all failed pass=%d", pass_idx)
            pass_outcomes.append(outcome)
            # Short settle between passes so any order still in the
            # gateway→matching-engine indexing window shows up in the
            # second pass.
            if pass_idx == 0:
                time.sleep(0.25)
        # Persist an explicit shutdown-cancel event so post-mortem
        # analysis (``explain_moment.py`` / DBeaver) can tell whether
        # the shutdown path actually executed the cancel before the
        # process died. Without this, the only clue in ``trading.db``
        # was the generic "service stopping" INFO at end-of-lifespan —
        # indistinguishable from a SIGKILL truncating mid-sequence.
        try:
            storage.insert_bot_event(
                utc_now_iso(),
                EventSeverity.INFO.value,
                "cancel_all_on_shutdown_executed",
                f"cancel_all_on_shutdown executed (symbol-scoped, outcomes={pass_outcomes})",
                {"symbol": settings.symbol, "pass_outcomes": pass_outcomes},
            )
        except Exception:
            logger.exception("shutdown cancel_all event_write failed")

    try_save_persistent_runtime_state(settings, state)
    if private_stream is not None:
        private_stream.stop()
    if public_stream is not None:
        public_stream.stop()
    if binance_stream is not None:
        binance_stream.stop()
    with state._lock:
        # Distinct from PAUSED so the dashboard doesn't fire a
        # "Bot PAUSED" toast on every deploy. Mapped to "stopping"
        # (amber, no toast) on the frontend. See ``BotStatus``
        # enum for rationale.
        state.bot_status = BotStatus.SHUTTING_DOWN
    storage.insert_bot_event(
        utc_now_iso(),
        EventSeverity.INFO.value,
        "shutdown",
        "service stopping",
        None,
    )
    # Telegram: emit shutdown notification synchronously (so it lands
    # before the worker thread is joined), then stop the poller + worker.
    if telegram_notifier.enabled:
        try:
            telegram_notifier.notify_ops_blocking(
                "INFO",
                "bot_shutdown",
                "bot shutting down (clean lifespan exit)",
                {"session_id": state.session_id},
            )
        except Exception:
            logger.exception("telegram_shutdown_notify_failed")
    try:
        telegram_commands.stop()
    except Exception:
        logger.exception("telegram_commands_stop_failed")
    try:
        telegram_notifier.stop()
    except Exception:
        logger.exception("telegram_notifier_stop_failed")
    storage.close()


def create_app() -> FastAPI:
    # ``version=`` is surfaced by FastAPI as ``info.version`` in
    # ``/openapi.json`` and in the Swagger UI at ``/docs``. This lets any
    # snapshot the operator pulls later pin the exact running build.
    app = FastAPI(title="Hyperliquid MM", version=__version__, lifespan=lifespan)
    app.include_router(api_router)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> FileResponse:
        if not _FAVICON_PATH.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(_FAVICON_PATH, media_type="image/x-icon")

    return app


app = create_app()


def main() -> None:
    import sys

    import uvicorn

    ensure_env_bootstrapped(argv=sys.argv)

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app.main:app", host=host, port=port, log_level="info", access_log=True)


if __name__ == "__main__":
    main()
