from __future__ import annotations

import logging
import time
from typing import Any, Optional

from app.enums import EventSeverity, Side
from app.exchange.base import PerpExchangeAdapter
from app.fill_ingestion import ingest_hl_fill_raw
from app.models import BestBidAsk, PositionSnapshot
from app.pnl import PnlTracker
from app.state import BotState
from app.storage import Storage
from app.utils.logfmt import log_extra
from app.utils.time import utc_now, utc_now_iso

logger = logging.getLogger(__name__)


def _empty_position(symbol: str) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        position_qty=0.0,
        avg_entry_price=None,
        mark_price=None,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
        ts_local=utc_now(),
    )


def _instrument_refresh_failure(
    state: BotState,
    storage: Storage | None,
    *,
    stage: str,
    sym: str,
    latency_ms: Optional[float],
    exc: BaseException | None = None,
) -> None:
    streak = state.market_refresh_note_failure(latency_ms)
    payload = {
        "event": "market_data_refresh_failure",
        "symbol": sym,
        "stage": stage,
        "latency_ms": latency_ms,
        "failed_refresh_streak": streak,
        "error": str(exc)[:500] if exc else None,
    }
    log_extra(logger, logging.WARNING, "market_data_refresh_failure", payload)
    if storage:
        storage.insert_bot_event(
            utc_now_iso(),
            EventSeverity.WARNING.value,
            "market_data_refresh_failure",
            f"market refresh failed stage={stage}",
            {k: v for k, v in payload.items() if k != "event"},
        )


def instrument_book_update_success(
    state: BotState,
    storage: Storage | None,
    *,
    sym: str,
    bb: BestBidAsk,
    latency_ms: float,
    out: dict[str, Any],
    on_stall_reconnect: Any = None,
) -> None:
    dbg = {
        "event": "market_data_refresh_success",
        "symbol": sym,
        "latency_ms": round(latency_ms, 3),
        "unchanged_snapshot_streak": out["unchanged_streak"],
    }
    log_extra(logger, logging.DEBUG, "market_data_refresh_success", dbg)
    if out.get("emit_success_info"):
        log_extra(logger, logging.INFO, "market_data_refresh_success", dbg)
        if storage:
            storage.insert_bot_event(
                utc_now_iso(),
                EventSeverity.INFO.value,
                "market_data_refresh_success",
                "market data refresh ok (throttled)",
                {k: v for k, v in dbg.items() if k != "event"},
            )
    if out.get("stall_just_crossed"):
        thr = int(state._settings.market_data_stall_unchanged_threshold)
        pl = {
            "event": "market_data_snapshot_stalled",
            "symbol": sym,
            "unchanged_snapshot_streak": out["unchanged_streak"],
            "stall_unchanged_threshold": thr,
        }
        log_extra(logger, logging.WARNING, "market_data_snapshot_stalled", pl)
        if storage:
            storage.insert_bot_event(
                utc_now_iso(),
                EventSeverity.WARNING.value,
                "market_data_snapshot_stalled",
                "book fingerprint unchanged across many OK BBO updates",
                {k: v for k, v in pl.items() if k != "event"},
            )
    if out.get("attempt_transport_reset"):
        reconnect_fn = on_stall_reconnect
        callable_rc = callable(reconnect_fn)
        restarted = False
        if callable_rc:
            try:
                reconnect_fn()
                restarted = True
                state.market_refresh_note_transport_reset_done()
            except Exception:
                logger.exception("public_ws_stall_reconnect_failed")
        rp = {
            "event": "market_data_task_restarted",
            "symbol": sym,
            "public_ws_reconnect_requested": callable_rc,
            "public_ws_reconnect_succeeded": restarted,
        }
        log_extra(logger, logging.WARNING, "market_data_task_restarted", rp)
        if storage:
            storage.insert_bot_event(
                utc_now_iso(),
                EventSeverity.WARNING.value,
                "market_data_task_restarted",
                "public websocket reconnect after BBO fingerprint stall",
                {k: v for k, v in rp.items() if k != "event"},
            )


def refresh_account_only(
    client: PerpExchangeAdapter,
    state: BotState,
    address: str,
    storage: Storage | None = None,
    pnl: PnlTracker | None = None,
    *,
    ingest_fills_via_rest: bool = True,
) -> None:
    """Position, account, optional REST fills. Live BBO must come from public websocket."""
    sym = state.symbol
    addr = (address or "").strip()
    t0 = time.perf_counter()

    if not addr:
        pos = _empty_position(sym)
        acct = None
        state.apply_account_position_only(pos, acct)
        state.note_account_only_refresh_success(0.0)
        return

    # 2026-05-13 Codex bug review HIGH #2: split into a two-stage
    # refresh.
    #
    # Stage 1 (REQUIRED): position + account. These are the
    # authoritative inventory-truth path. Treat any failure here
    # as a real refresh failure — return early, engage the throttle,
    # log the exception. Pre-fix, an exception thrown by
    # ``fetch_recent_fills_raw`` (stage 2) would discard fresh
    # position/account state via the shared try block, leaving the
    # bot to trip ``account_data_stale`` on a transient
    # fills-endpoint hiccup. Splitting the try keeps inventory-truth
    # alive when only the optional observability/catch-up path fails.
    try:
        pos = client.fetch_position(addr, sym)
        acct = client.fetch_account_snapshot(addr)
    except Exception as e:
        logger.exception("account refresh failed")
        lat = (time.perf_counter() - t0) * 1000.0
        _instrument_refresh_failure(
            state, storage, stage="account_fetch", sym=sym, latency_ms=lat, exc=e
        )
        # Engage the rate-limit throttle even on failure. Without this,
        # an HTTP 429 (or any other exception) would leave the throttle
        # window un-advanced, and the bot's next tick would re-fire
        # the refresh immediately -- producing the tight-loop 429
        # cascade that killed the OKX bot during 2026-05-05 bring-up.
        # See bug-018 + state.note_account_only_refresh_attempt docstring.
        state.note_account_only_refresh_attempt()
        state.bump_execution_errors("account_refresh_exception")
        return

    state.apply_account_position_only(pos, acct)
    lat = (time.perf_counter() - t0) * 1000.0
    state.note_account_only_refresh_success(lat)

    # Stage 2 (OPTIONAL): recent-fills catch-up. Failure here is
    # logged + counted but doesn't trip ``account_data_stale``.
    # Distinct error counter from stage 1 so the operator can spot
    # "fills-endpoint sad, account fine" patterns in telemetry.
    if not ingest_fills_via_rest:
        logger.debug(
            "rest_fill_fetch_skipped reason=private_ws_primary_path symbol=%s",
            sym,
        )
        return

    try:
        fills_raw = client.fetch_recent_fills_raw(addr, sym)
    except Exception as e:
        # Bug 3 fix (Codex review 2026-05-13 MED #3) — on the new
        # ``OkxApiError`` path the adapter raises with a typed
        # exception that carries OKX's ``code``/``msg``. Log it as a
        # distinct degraded-state signal so the operator can
        # distinguish "fills catch-up failed" from "zero fills
        # returned" (which previously rendered identically).
        logger.exception("rest_fill_fetch_failed (catch-up path; account ok)")
        _instrument_refresh_failure(
            state,
            storage,
            stage="fills_fetch",
            sym=sym,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            exc=e,
        )
        state.bump_execution_errors("rest_fill_fetch_exception")
        return

    logger.debug(
        "rest_fill_reconcile invoked reason=tick_ingest fills_returned=%s",
        len(fills_raw),
    )
    # Pre-fix this loop sliced ``fills_raw[:50]`` even though OKX
    # fetches up to 100 fills per call (and other venues may return
    # more). Half of any private-WS-missed burst was silently
    # dropped — fees / realized PnL / toxicity / session counters
    # stayed permanently incomplete. Codex 2026-05-09 HIGH-3 fix:
    # ingest the entire returned page. The fetch-side limit (set by
    # the adapter) is the only cap that should apply.
    #
    # NOTE: a stronger fix would paginate by tradeId/timestamp until
    # the last seen fill, so a >100-fill burst is also captured. That
    # needs a per-symbol catch-up cursor in storage; flagged for a
    # follow-up. The simple version below at least stops dropping
    # half of the data we already fetched.
    #
    # 2026-05-13 Codex bug review MED #4 fix: pre-compute each
    # catch-up fill's TRUE pre-fill inventory by walking the batch
    # in reverse. After ``apply_account_position_only`` ran above,
    # ``state.position.position_qty`` reflects the POST-everything
    # venue snapshot. To recover the genuine pre-fill qty for fill
    # ``i``, we start from post-all and subtract each fill's signed
    # delta from i+1..N-1. We pass the result into
    # ``ingest_hl_fill_raw`` via ``pre_inventory_qty_override`` so
    # the persisted ``inventory_qty_before_fill`` is correct for
    # regime attribution on catch-up fills (which is exactly the
    # path meant to repair missed private-WS fills — so the bias
    # used to hit the case the analysis was supposed to investigate).
    try:
        post_all_qty = float(state.position.position_qty)
    except (TypeError, ValueError):
        post_all_qty = None
    pre_fill_qty_overrides: list[Optional[float]] = [None] * len(fills_raw)
    if post_all_qty is not None and fills_raw:
        rolling = post_all_qty
        for i in range(len(fills_raw) - 1, -1, -1):
            fr = fills_raw[i]
            # Compute signed delta. ``HLFillRaw.sz`` is unsigned;
            # apply sign from ``side``. BUY adds to position, SELL
            # subtracts. Match the convention used by
            # ``BotState.record_fill`` so the reverse-walk inverts
            # cleanly. ``getattr`` is used for both ``sz`` and
            # ``side`` so mock fills delivered as plain dicts in tests
            # don't crash this loop — they fall through to the
            # ``None`` override (no pre-fill data), which preserves
            # the legacy (pre-fix) behaviour for those test fixtures
            # rather than mis-stamping post-fill qty.
            try:
                sz = float(getattr(fr, "sz", 0.0) or 0.0)
            except (TypeError, ValueError):
                sz = 0.0
            side_val = getattr(fr, "side", None)
            if side_val is None:
                # Can't compute signed delta — leave override as None
                # for this fill so ingest falls back to the legacy
                # ``state.position.position_qty`` read.
                pre_fill_qty_overrides[i] = None
                # Don't advance rolling — we don't know the delta, so
                # subsequent fills upstream of this one can't be
                # computed either. Bail out of the reverse walk.
                break
            signed_delta = sz if side_val == Side.BUY else -sz
            pre_fill_qty_overrides[i] = rolling - signed_delta
            rolling = rolling - signed_delta
    for idx, fr in enumerate(fills_raw):
        # ``shadow_update_position=False``: the
        # ``apply_account_position_only(pos, acct)`` call above already
        # installed the venue's authoritative position, which by
        # definition reflects every fill on the account. Letting
        # ``record_fill`` shadow-add the delta on top would
        # double-count the fill into ``position_qty``. Codex review
        # 2026-05-08 found this — pre-fix, a private-WS-missed fill
        # caught up by REST landed twice in local state until the next
        # refresh repaired it. ``record_fill`` still does session-
        # metrics bookkeeping (fill counts, recent_fills, traded
        # notional, latency) — only the shadow-position write is
        # suppressed.
        ingest_hl_fill_raw(
            state=state,
            storage=storage,
            pnl=pnl,
            symbol=sym,
            fr=fr,
            source="rest",
            shadow_update_position=False,
            pre_inventory_qty_override=pre_fill_qty_overrides[idx],
        )
