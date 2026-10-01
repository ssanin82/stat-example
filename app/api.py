from __future__ import annotations

"""
HTTP surface: GET routes are always on; POST /control/* returns 404 unless
CONTROL_ENDPOINTS_ENABLED=true (fail-closed).
"""

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.enums import EventSeverity
from app.exchange.factory import venue_account_address
from app.market_data_timing import PublicWsTimingTracker
from app.utils.time import utc_now

router = APIRouter()

_MARKET_DATA_STATUS_KEYS = (
    "market_data_last_success_wall_ts",
    "market_data_refresh_latency_ms",
    "market_data_failed_refresh_streak",
    "market_data_unchanged_snapshot_streak",
    "market_data_recovery_state",
    "market_data_recovery_refresh_attempts_episode",
    "market_data_transport_reset_count",
    "live_market_data_source",
    "public_ws_connected",
    "public_ws_last_message_wall_ts",
    "public_ws_reconnect_count",
    "public_ws_seen_first_bbo",
)

_PRIVATE_WS_STATUS_KEYS = (
    "private_ws_connected",
    "private_ws_healthy",
    "private_ws_last_connect_ts",
    "private_ws_last_ping_sent_ts",
    "private_ws_last_pong_ts",
    "private_ws_last_message_ts",
    "private_ws_seconds_since_last_message",
    "private_ws_disconnect_reason_last",
    "private_ws_reconnect_count",
    "private_ws_queue_high_watermark",
    "private_ws_queue_overflow_count",
)

# Binance cross-venue reference feed (optional; no-op when BINANCE_WS_ENABLED=false).
# Forwarded into /status and /health so the snapshot captures whether the
# Binance stream is alive, how fresh its last message is, and the current
# GRVT-vs-Binance basis EWMA used by the cancel-on-move trigger.
_BINANCE_WS_STATUS_KEYS = (
    "binance_ws_connected",
    "binance_ws_reconnect_count",
    "binance_ws_last_connect_ts",
    "binance_ws_last_message_ts",
    "binance_ws_seconds_since_last_message",
    "binance_best_bid",
    "binance_best_ask",
    "binance_bid_size",
    "binance_ask_size",
    "binance_mid",
    "binance_basis_ewma",
)


def _clamp_limit(limit: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, limit))


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Process liveness: ``status`` is ``ok`` whenever this handler runs (not a full bot-health probe)."""
    st = request.app.state.bot_state
    flags = st.status_flags_dict()
    out: dict[str, Any] = {
        "status": "ok",
        "bot_status": flags["bot_status"],
        "symbol": st.symbol,
        "timestamp": utc_now().isoformat(),
        "trading_enabled": request.app.state.settings.trading_enabled,
        "market_data_available": flags["market_data_available"],
        "account_data_available": flags["account_data_available"],
        "order_desync": flags["order_desync"],
        "desync_phase": flags["desync_phase"],
        "desync_quarantine_remaining": flags["desync_quarantine_remaining"],
        "flatten_incomplete": flags["flatten_incomplete"],
        "flatten_residual_abs_qty": flags["flatten_residual_abs_qty"],
        "trades_last_minute": flags["trades_last_minute"],
        "latency_fill_to_process_ms": flags["latency_fill_to_process_ms"],
        "exchange_ts_to_decision_ms": flags["exchange_ts_to_decision_ms"],
        "latency_decision_to_first_place_ms": flags["latency_decision_to_first_place_ms"],
        "latency_tick_preamble_ms": flags["latency_tick_preamble_ms"],
        "last_market_data_ts": flags.get("last_market_ts"),
    }
    for k in _MARKET_DATA_STATUS_KEYS + _PRIVATE_WS_STATUS_KEYS + _BINANCE_WS_STATUS_KEYS:
        if k in flags:
            out[k] = flags[k]
    return out


@router.get("/status")
def status(request: Request) -> dict[str, Any]:
    st = request.app.state.bot_state
    s = request.app.state.settings
    flags = st.status_flags_dict()
    out: dict[str, Any] = {
        "bot_status": flags["bot_status"],
        "symbol": st.symbol,
        "last_heartbeat": flags["last_heartbeat"],
        "last_market_data_ts": flags["last_market_ts"],
        "quote_loop_seconds": s.quote_loop_seconds,
        "snapshot_interval_seconds": s.snapshot_interval_seconds,
        "killed": flags["killed"],
        "flatten_mode": flags["flatten_mode"],
        "manual_pause": flags["manual_pause"],
        "market_data_available": flags["market_data_available"],
        "account_data_available": flags["account_data_available"],
        "order_desync": flags["order_desync"],
        "desync_phase": flags["desync_phase"],
        "desync_quarantine_remaining": flags["desync_quarantine_remaining"],
        "flatten_incomplete": flags["flatten_incomplete"],
        "flatten_residual_abs_qty": flags["flatten_residual_abs_qty"],
        "control_endpoints_enabled": s.control_endpoints_enabled,
        "trades_last_minute": flags["trades_last_minute"],
        "latency_fill_to_process_ms": flags["latency_fill_to_process_ms"],
        "exchange_ts_to_decision_ms": flags["exchange_ts_to_decision_ms"],
        "latency_decision_to_first_place_ms": flags["latency_decision_to_first_place_ms"],
        "latency_tick_preamble_ms": flags["latency_tick_preamble_ms"],
    }
    for k in _MARKET_DATA_STATUS_KEYS + _PRIVATE_WS_STATUS_KEYS + _BINANCE_WS_STATUS_KEYS:
        if k in flags:
            out[k] = flags[k]
    # v1.5.2: recording state (icon + size indicator in the dashboard).
    # Safe-by-construction — never raises.
    try:
        from app.recording_status import recording_status
        out["recording"] = recording_status(s)
    except Exception:
        # Defensive: a status endpoint must never 500.
        out["recording"] = {
            "enabled": bool(getattr(s, "recording_enabled", False)),
            "active": False,
            "session_name": None,
            "session_path": None,
            "bytes": 0,
            "files": 0,
            "profile_resolved": "",
        }
    return out


@router.get("/config")
def config_snapshot(request: Request) -> dict[str, Any]:
    return request.app.state.settings.sanitized_dict()


@router.get("/market-data/gap-stats")
def market_data_gap_stats(request: Request) -> dict[str, Any]:
    """Update-to-update gaps between successful market snapshot applies (in-memory; see quantiles_note)."""
    st = request.app.state.bot_state
    return st.market_data_gap_tracker.to_api_dict(session_id=st.session_id, symbol=st.symbol)


@router.get("/market-data/timing-summary")
def market_data_timing_summary(request: Request) -> dict[str, Any]:
    """
    Rolling timing summary for the public websocket market-data feed.

    Semantics:
    - exchange_gap_ms: spacing between successive exchange-provided timestamps (exchange cadence).
    - local_receive_gap_ms: spacing between successive local receive instants (arrival cadence).
    - exchange_to_local_receive_ms: one-way age at receipt (exchange_ts_ms -> local receipt wall clock).
    - receive_to_apply_ms: local delay from receive callback to state apply (our process).
    """
    st = request.app.state.bot_state
    tr = getattr(st, "public_ws_timing", None)
    if tr is None:
        return {
            "symbol": st.symbol,
            "source_type": "public_ws",
            "enabled": False,
            "notes": "MARKET_DATA_TIMING_WINDOW_ENABLED=false",
        }
    out = tr.summary()
    out["enabled"] = True
    out["session_id"] = st.session_id
    out["notes"] = {
        "exchange_gap_ms": "distance between successive exchange timestamps (ms); indicates exchange update cadence",
        "local_receive_gap_ms": "distance between successive local receipt wall times (ms); indicates arrival cadence",
        "exchange_to_local_receive_ms": "local_receive_wall_ms - exchange_ts_ms (ms); one-way age at receipt (negative => clock anomaly)",
        "receive_to_apply_ms": (
            "local processing delay from receipt to state apply (ms); "
            "computed from monotonic timestamps when available, else wall-clock when both wall times exist"
        ),
        # NOTE: boundary_event_count semantics live in summary["boundary_count_semantics"]
        # (single source of truth). Do not duplicate that explanation here.
    }
    return out


@router.get("/market-data/timing-recent")
def market_data_timing_recent(request: Request, limit: int = 200) -> dict[str, Any]:
    """Most recent timing samples (newest-first)."""
    st = request.app.state.bot_state
    s = request.app.state.settings
    tr = getattr(st, "public_ws_timing", None)
    if tr is None:
        return {
            "symbol": st.symbol,
            "source_type": "public_ws",
            "enabled": False,
            "samples": [],
        }
    cap = int(getattr(s, "market_data_timing_raw_endpoint_max_limit", 1000))
    lim = _clamp_limit(int(limit), 1, cap)
    summ = tr.summary()
    raw = tr.recent_samples(limit=lim)
    return {
        "symbol": st.symbol,
        "source_type": "public_ws",
        "enabled": True,
        "newest_first": True,
        "limit": lim,
        "max_limit": cap,
        "configured_window_size": summ.get("configured_window_size"),
        "current_buffer_size": summ.get("current_buffer_size"),
        "samples": [PublicWsTimingTracker.serialize_sample(x) for x in raw],
    }


@router.get("/market-data/binance-timing-summary")
def market_data_binance_timing_summary(request: Request) -> dict[str, Any]:
    """Rolling timing summary for the Binance cross-venue reference feed.

    Same metric set as ``/market-data/timing-summary`` (exchange-gap,
    receive-gap, one-way delay, receive-to-apply) but against the
    Binance-side timestamps (Futures ``E`` field) rather than GRVT's.
    Used to measure Binance-Tokyo → bot-host transport latency and
    decide whether a lower-latency reference venue (e.g. OKX Singapore)
    would be worth the integration cost.

    Returns ``enabled: false`` when:
      * ``BINANCE_WS_ENABLED=false`` (no stream running), OR
      * ``MARKET_DATA_TIMING_WINDOW_ENABLED=false`` (no tracker).
    """
    st = request.app.state.bot_state
    s = request.app.state.settings
    tr = getattr(st, "binance_public_ws_timing", None)
    if tr is None or not bool(getattr(s, "binance_ws_enabled", False)):
        return {
            "symbol": getattr(s, "binance_symbol", None),
            "source_type": "binance_public_ws",
            "enabled": False,
            "notes": (
                "BINANCE_WS_ENABLED=false"
                if not bool(getattr(s, "binance_ws_enabled", False))
                else "MARKET_DATA_TIMING_WINDOW_ENABLED=false"
            ),
        }
    out = tr.summary()
    out["enabled"] = True
    out["session_id"] = st.session_id
    # Override the tracker's default ``source_type`` so consumers of
    # both timing endpoints can tell the series apart in a snapshot
    # folder without looking at the filename.
    out["source_type"] = "binance_public_ws"
    out["notes"] = {
        "exchange_gap_ms": "distance between successive Binance event times (ms); exchange update cadence",
        "local_receive_gap_ms": "distance between successive local receipt wall times (ms); arrival cadence",
        "exchange_to_local_receive_ms": (
            "local_receive_wall_ms - binance_event_time_ms (ms); "
            "one-way transport latency Binance Tokyo → our host"
        ),
        "receive_to_apply_ms": (
            "local processing delay from receipt to state apply (ms); "
            "monotonic when available, else wall-clock"
        ),
    }
    return out


@router.get("/market-data/binance-timing-recent")
def market_data_binance_timing_recent(
    request: Request, limit: int = 200
) -> dict[str, Any]:
    """Most recent Binance timing samples (newest-first).

    Mirrors ``/market-data/timing-recent`` but against the Binance
    tracker. ``limit`` is clamped by
    ``MARKET_DATA_TIMING_RAW_ENDPOINT_MAX_LIMIT``, same as the GRVT
    endpoint.
    """
    st = request.app.state.bot_state
    s = request.app.state.settings
    tr = getattr(st, "binance_public_ws_timing", None)
    if tr is None or not bool(getattr(s, "binance_ws_enabled", False)):
        return {
            "symbol": getattr(s, "binance_symbol", None),
            "source_type": "binance_public_ws",
            "enabled": False,
            "samples": [],
        }
    cap = int(getattr(s, "market_data_timing_raw_endpoint_max_limit", 1000))
    lim = _clamp_limit(int(limit), 1, cap)
    summ = tr.summary()
    raw = tr.recent_samples(limit=lim)
    return {
        "symbol": getattr(s, "binance_symbol", None),
        "source_type": "binance_public_ws",
        "enabled": True,
        "newest_first": True,
        "limit": lim,
        "max_limit": cap,
        "configured_window_size": summ.get("configured_window_size"),
        "current_buffer_size": summ.get("current_buffer_size"),
        "samples": [PublicWsTimingTracker.serialize_sample(x) for x in raw],
    }


@router.get("/state/current")
def state_current(request: Request) -> dict[str, Any]:
    return request.app.state.bot_state.snapshot_dict()


@router.get("/toxicity/current")
def toxicity_current(request: Request) -> dict[str, Any]:
    """Session-scoped rolling toxicity from in-memory recent fills (observational; not strategy)."""
    return request.app.state.bot_state.runtime_toxicity_summary_dict()


@router.get("/telemetry/quote-quality")
def telemetry_quote_quality(request: Request) -> dict[str, Any]:
    """Quote-cycle spread, two-sided mix, execution counters, spread-capture and delayed markout (observational)."""
    return request.app.state.bot_state.quote_quality_dict()


@router.get("/state/position")
def state_position(request: Request) -> dict[str, Any]:
    return request.app.state.bot_state.position_dict()


@router.get("/orders/recent")
def orders_recent(request: Request, limit: int = 100) -> list[dict[str, Any]]:
    return request.app.state.storage.recent_orders(_clamp_limit(limit, 1, 500))


@router.get("/orders/lifecycle-trace")
def orders_lifecycle_trace(request: Request) -> list[dict[str, Any]]:
    """Per-order lifecycle ring buffer (Phase 2 of the gone_on_exchange
    diagnostic). Each entry records: place dispatch -> place response
    -> WS events -> terminal transition. Used to distinguish a silent
    WS (ws_events empty) from a state-machine race (ws_events present
    but reconcile fired anyway). See ``app/order_trace.py``.
    """
    return request.app.state.bot_state.order_trace.to_list()


@router.get("/fills/recent")
def fills_recent(request: Request, limit: int = 100) -> list[dict[str, Any]]:
    return request.app.state.storage.recent_fills(_clamp_limit(limit, 1, 500))


@router.get("/quotes/recent")
def quotes_recent(request: Request, limit: int = 100) -> list[dict[str, Any]]:
    return request.app.state.storage.recent_quote_decisions(_clamp_limit(limit, 1, 500))


@router.get("/pnl/current")
def pnl_current(request: Request) -> dict[str, Any]:
    return request.app.state.bot_state.pnl_dict()


@router.get("/pnl/history")
def pnl_history(request: Request, limit: int = 1000) -> list[dict[str, Any]]:
    return request.app.state.storage.equity_history(_clamp_limit(limit, 1, 5000))


@router.get("/inventory/history")
def inventory_history(request: Request, limit: int = 1000) -> list[dict[str, Any]]:
    return request.app.state.storage.position_history(_clamp_limit(limit, 1, 5000))


@router.get("/events/recent")
def events_recent(request: Request, limit: int = 100) -> list[dict[str, Any]]:
    return request.app.state.storage.recent_bot_events(_clamp_limit(limit, 1, 500))


def _session_started_iso(request: Request) -> str:
    """Current bot session start timestamp in ISO-8601 (used to default
    ``since`` params on session-scoped endpoints).

    The main tables (``fills``, ``equity_snapshots``, ``bot_events``) are NOT
    wiped between runs, so every session-level query must filter by this
    timestamp to avoid picking up junk from prior sessions.
    """
    st = request.app.state.bot_state
    started = getattr(st, "session_started_at_utc", None)
    if started is None:
        return utc_now().isoformat()
    try:
        return started.isoformat()
    except AttributeError:
        return str(started)


@router.get("/fills/since")
def fills_since(
    request: Request,
    ts: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Fills with ``ts_fill >= ts``, oldest-first.

    Unlike ``/fills/recent`` (last N newest-first, capped at 500) this
    returns a session-scoped window suitable for analytics. Default ``ts``
    is the current session's start. Optional ``until`` caps the range
    (exclusive-upper) for bounded-window pulls.
    """
    since = ts or _session_started_iso(request)
    n = _clamp_limit(limit, 1, 10000)
    return request.app.state.storage.fills_since(since, n, until_ts=until)


# ---------------------------------------------------------------------
# Bulk history endpoints for multi-hour / multi-day analysis. Each
# endpoint returns rows oldest-first over ``[since, until)``, defaulting
# ``since`` to the current session start. Paired with the ``*_since``
# helpers on ``Storage`` — see ``scripts/explain_moment.py`` for the
# correlation CLI that consumes them.
#
# These complement the existing ``/<table>/recent`` handlers (last N
# newest-first, capped at 500-1000 rows). ``/<table>/since`` is the
# right endpoint for explaining "what happened at 17:00:43?" across a
# multi-hour window; ``/<table>/recent`` is for live dashboards.
# ---------------------------------------------------------------------


@router.get("/orders/since")
def orders_since(
    request: Request,
    since: Optional[str] = None,
    until: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Orders with ``ts_created >= since``, oldest-first.

    Defaults ``since`` to the current session start. Optional
    ``symbol`` filters to a single trading symbol (handy when the same
    bot has been run against multiple symbols in sequence — the
    ``orders`` table is not auto-pruned between runs).
    """
    since_iso = since or _session_started_iso(request)
    n = _clamp_limit(limit, 1, 50000)
    return request.app.state.storage.orders_since(
        since_iso, n, until_ts=until, symbol=symbol
    )


@router.get("/events/since")
def events_since(
    request: Request,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Bot events with ``ts >= since``, oldest-first.

    Events are the structured, operational signals (kills, reconnects,
    flattens, market-data recoveries, etc.) — the things you want to
    correlate with fills and price movement after a long run. For raw
    Python log lines use the deploy host's stdout/stderr stream
    (systemd journal on the current colo box; container log on PaaS).
    """
    since_iso = since or _session_started_iso(request)
    n = _clamp_limit(limit, 1, 50000)
    return request.app.state.storage.bot_events_since(since_iso, n, until_ts=until)


@router.get("/quotes/since")
def quotes_since(
    request: Request,
    since: Optional[str] = None,
    until: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Quote-decision cycles with ``ts >= since``, oldest-first.

    This is the highest-volume table (1-5 rows/sec). For multi-hour
    windows pick a tight time range; the default limit of 10000 rows
    is about 30 min of quoting at 5 Hz. Hard cap is 50000 — above that,
    pull from the DB directly.
    """
    since_iso = since or _session_started_iso(request)
    n = _clamp_limit(limit, 1, 50000)
    return request.app.state.storage.quote_decisions_since(
        since_iso, n, until_ts=until, symbol=symbol
    )


@router.get("/orders/lifecycle-since")
def orders_lifecycle_since(
    request: Request,
    since: Optional[str] = None,
    until: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50000,
) -> list[dict[str, Any]]:
    """Order lifecycle view with derived fields.

    2026-05-13 regime-observability Phase 4b.

    Same rows as ``/orders/since`` but with three SELECT-time derived
    columns added: ``lifetime_ms``, ``cancel_to_close_ms``,
    ``placement_to_ack_ms``. Lets the snapshot consumer slice by
    "orders that filled vs got cancelled" + "venue cancel-confirm
    latency" + "venue submit RTT" without joining the timestamps
    client-side.

    The existing ``/orders/since`` endpoint stays unchanged — this
    is a parallel view for the lifecycle analytics consumer.
    """
    since_iso = since or _session_started_iso(request)
    n = _clamp_limit(limit, 1, 100000)
    return request.app.state.storage.orders_lifecycle_since(
        since_iso, until_iso=until, symbol=symbol, limit=n
    )


@router.get("/exposure/since")
def exposure_since(
    request: Request,
    since: Optional[str] = None,
    until: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50000,
) -> list[dict[str, Any]]:
    """Exposure-bar rows with ``ts_bar >= since``, oldest-first.

    2026-05-13 regime-observability Phase 2.

    Each row is a periodic snapshot (default 5 s cadence) of the bot's
    market + strategy state — independent of whether a fill happened.
    Provides the exposure denominator for per-regime PnL slicing.

    At 5 s cadence, a 15 h session is ~10.8 K rows (~1 MB JSON).
    Default limit of 50 K covers >2 days; raise the cap only for
    extreme deep-dive sessions.
    """
    since_iso = since or _session_started_iso(request)
    n = _clamp_limit(limit, 1, 200000)
    return request.app.state.storage.exposure_bars_since(
        since_iso, until_iso=until, symbol=symbol, limit=n
    )


@router.get("/inventory/since")
def inventory_since(
    request: Request,
    since: Optional[str] = None,
    until: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Position snapshots with ``ts >= since``, oldest-first.

    Use this to reconstruct position trajectory across a long session —
    when did we build inventory, when did we unwind, when did we hit
    the cap.
    """
    since_iso = since or _session_started_iso(request)
    n = _clamp_limit(limit, 1, 50000)
    return request.app.state.storage.position_snapshots_since(
        since_iso, n, until_ts=until, symbol=symbol
    )


@router.get("/equity/since")
def equity_since(
    request: Request,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 10000,
) -> list[dict[str, Any]]:
    """Equity snapshots with ``ts >= since``, oldest-first.

    Paired with ``/fills/since`` + ``/pnl/attribution`` to reconstruct
    the realised / unrealised / drawdown trajectory. ``equity_snapshots``
    previously had no ``/recent`` endpoint either — this is the primary
    way to pull the equity time-series.
    """
    since_iso = since or _session_started_iso(request)
    n = _clamp_limit(limit, 1, 50000)
    return request.app.state.storage.equity_history_since(
        since_iso, n, until_ts=until
    )


@router.get("/pnl/attribution")
def pnl_attribution(
    request: Request,
    since: Optional[str] = None,
    horizon_s: int = 5,
) -> dict[str, Any]:
    """P&L decomposition into fee income / markout dollar impact / residual.

    This is the answer to "why am I losing money?" for a maker running near
    flat — it tells you whether the loss is adverse selection (markout),
    fee structure (fees > rebates), or something else (residual: inventory
    drift past the markout horizon).

    Default ``since`` = current session start. ``horizon_s`` must be 1, 3, or 5.
    """
    if horizon_s not in (1, 3, 5):
        raise HTTPException(
            status_code=400,
            detail=f"horizon_s must be 1, 3, or 5 (got {horizon_s})",
        )
    since_iso = since or _session_started_iso(request)
    until_iso = utc_now().isoformat()
    fills = request.app.state.storage.fills_since(since_iso, 100000)
    st = request.app.state.bot_state
    realized = float(getattr(st.pnl, "realized_pnl_usd", 0.0) or 0.0)
    from app.pnl_attribution import compute_attribution

    return compute_attribution(
        fills=fills,
        horizon_s=horizon_s,
        window_since_iso=since_iso,
        window_until_iso=until_iso,
        realized_pnl_total_usd=realized,
    )


@router.get("/session/summary")
def session_summary(
    request: Request,
    horizon_s: int = 5,
) -> dict[str, Any]:
    """One-shot rollup of the current session: throughput, equity, attribution,
    key counters. Intended to answer "how is the session going?" in one GET.
    """
    if horizon_s not in (1, 3, 5):
        raise HTTPException(
            status_code=400,
            detail=f"horizon_s must be 1, 3, or 5 (got {horizon_s})",
        )
    st = request.app.state.bot_state
    since_iso = _session_started_iso(request)
    now_iso = utc_now().isoformat()
    fills = request.app.state.storage.fills_since(since_iso, 100000)
    equity_hist = request.app.state.storage.equity_history_since(since_iso, 100000)

    # Key counters: surface operational signals that matter for a session
    # health read. All optional — the presence of the field reflects the
    # current state fields, nothing fabricated.
    key_counters: dict[str, Any] = {}
    for attr, label in (
        ("private_ws_reconnect_count", "private_ws_reconnect_count"),
        ("public_ws_reconnect_count", "public_ws_reconnect_count"),
        ("private_ws_queue_drops", "private_ws_queue_drops"),
        ("private_ws_queue_overflow_count", "private_ws_queue_overflow_count"),
        ("session_fill_count", "session_fill_count"),
    ):
        if hasattr(st, attr):
            key_counters[label] = getattr(st, attr)
    # Per-side session fill counts (the monotonic counter added in the
    # self-perpetuation audit).
    by_side = getattr(st, "session_fill_count_by_side", None)
    if by_side:
        key_counters["session_fill_count_by_side"] = {
            str(k.value if hasattr(k, "value") else k): int(v) for k, v in by_side.items()
        }

    # Reasoning for NOT surfacing exec-layer counters (adverse_side_pause_skip_count,
    # cloid-escalation retry totals, etc.) here: they live on OrderManager which
    # the API layer does not hold a direct reference to. They show up already
    # in ``/telemetry/quote-quality`` and ``/events/recent``; keeping this
    # endpoint reading from ``BotState`` only avoids a layering change.

    bot_status = None
    status_attr = getattr(st, "bot_status", None)
    if status_attr is not None:
        bot_status = status_attr.value if hasattr(status_attr, "value") else str(status_attr)

    realized = float(getattr(st.pnl, "realized_pnl_usd", 0.0) or 0.0)

    from app.session_summary import build_session_summary

    settings_obj = getattr(request.app.state, "settings", None)
    experiment_id = (
        (getattr(settings_obj, "experiment_id", "") or "").strip() or None
        if settings_obj is not None
        else None
    )

    return build_session_summary(
        session_id=str(getattr(st, "session_id", "")),
        session_started_at_utc=since_iso,
        now_utc=now_iso,
        fills=fills,
        equity_history=equity_hist,
        realized_pnl_total_usd=realized,
        markout_horizon_s=horizon_s,
        bot_status=bot_status,
        key_counters=key_counters,
        experiment_id=experiment_id,
    )


@router.get("/trading-state")
def trading_state(request: Request) -> dict[str, Any]:
    """One-shot operator-oriented trading state: **live** open orders + position + session PnL.

    Intended for ad-hoc inspection when the exchange UI shows stale data
    (observed on Bluefin during active quoting with sub-second order
    lifetimes). Three distinct data sources in one payload:

    - ``open_orders``: fetched live from the exchange REST at request time
      via ``client.fetch_open_orders_raw`` — NOT from local cache or DB.
      Filtered to the bot's configured ``symbol``. ``count`` reflects what
      the matching engine currently holds; ``error`` is non-null if the
      REST call failed (network, auth, rate-limit).
    - ``position``: from ``BotState.position_dict()``. Updated in real time
      via private-WS fill events; effectively live.
    - ``pnl_since_session_start``: from ``BotState.pnl_dict()``. Scoped to
      the current bot session (``session_started_at_utc``), NOT to the
      account's lifetime history.

    Do not poll at high frequency — each call costs one REST round-trip
    to the exchange and a private-WS snapshot. Intended for
    ``scripts/trading_state.py`` and interactive curl, not dashboards.
    """
    st = request.app.state.bot_state
    client = request.app.state.client
    settings = request.app.state.settings
    target_symbol = st.symbol

    live_orders: list[dict[str, Any]] = []
    live_orders_error: Optional[str] = None
    try:
        addr = venue_account_address(settings)
        if not addr:
            live_orders_error = "no account address configured for this venue"
        else:
            raw = client.fetch_open_orders_raw(addr)
            now_ms = int(utc_now().timestamp() * 1000)
            for o in raw:
                # Same defensive symbol filter as execution.cancel_all:
                # the adapter should have scoped to our symbol already,
                # but guard against a misbehaving adapter leaking orders
                # from other symbols if the account has them.
                coin = getattr(o, "coin", None)
                if target_symbol and coin != target_symbol:
                    continue
                age_s: Optional[float] = None
                # OpenOrderRaw exposes the creation timestamp as ``timestamp``
                # (ms epoch). ``time_ms`` is the name used on FillRaw — not
                # here. Prior revisions of this endpoint used the wrong name
                # and every order rendered with ``age=—``.
                t_ms = getattr(o, "timestamp", None) or getattr(o, "time_ms", None)
                if t_ms:
                    try:
                        age_s = max(0.0, (now_ms - int(t_ms)) / 1000.0)
                    except (TypeError, ValueError):
                        age_s = None
                side = getattr(o, "side", None)
                side_str = side.value if hasattr(side, "value") else str(side)
                live_orders.append(
                    {
                        "oid": getattr(o, "oid", None),
                        "cloid": getattr(o, "cloid", None),
                        "symbol": coin,
                        "side": side_str,
                        "price": getattr(o, "limit_px", None),
                        "size": getattr(o, "sz", None),
                        "time_ms": t_ms,
                        "age_seconds": age_s,
                    }
                )
    except Exception as e:  # noqa: BLE001
        # Broad catch is deliberate: this endpoint must always return a
        # payload the operator can read. If the live fetch fails we still
        # want them to see position/PnL.
        live_orders_error = f"{type(e).__name__}: {str(e)[:200]}"

    position = st.position_dict()
    pnl = st.pnl_dict()

    session_started = getattr(st, "session_started_at_utc", None)
    try:
        session_started_iso = session_started.isoformat() if session_started else None
    except AttributeError:
        session_started_iso = str(session_started) if session_started else None

    bot_status_val = getattr(st, "bot_status", None)
    bot_status_str: Optional[str] = None
    if bot_status_val is not None:
        bot_status_str = (
            bot_status_val.value if hasattr(bot_status_val, "value") else str(bot_status_val)
        )

    return {
        "requested_at_utc": utc_now().isoformat(),
        "symbol": target_symbol,
        "bot_status": bot_status_str,
        "session_started_at_utc": session_started_iso,
        "position": position,
        "pnl_since_session_start": pnl,
        "open_orders": {
            "source": "exchange_rest_live",
            "count": len(live_orders),
            "orders": live_orders,
            "error": live_orders_error,
        },
    }


@router.get("/kill-state")
def kill_state(request: Request) -> dict[str, Any]:
    st = request.app.state.bot_state
    flags = st.status_flags_dict()
    out = st.kill_state_dict()
    s = request.app.state.settings
    out["flatten_on_kill"] = s.flatten_on_kill
    out["flatten_incomplete"] = flags["flatten_incomplete"]
    out["flatten_residual_abs_qty"] = flags["flatten_residual_abs_qty"]
    out["order_desync"] = flags["order_desync"]
    out["desync_phase"] = flags["desync_phase"]
    out["desync_quarantine_remaining"] = flags["desync_quarantine_remaining"]
    return out


def _require_control(request: Request) -> None:
    if not request.app.state.settings.control_endpoints_enabled:
        raise HTTPException(status_code=404, detail="control endpoints disabled")


@router.post("/control/pause")
def control_pause(request: Request) -> dict[str, str]:
    _require_control(request)
    st = request.app.state.bot_state
    st.set_manual_pause(True)
    request.app.state.storage.insert_bot_event(
        utc_now().isoformat(),
        EventSeverity.INFO.value,
        "control_pause",
        "manual pause",
        None,
    )
    return {"status": "paused"}


@router.post("/control/resume")
def control_resume(request: Request) -> dict[str, str]:
    _require_control(request)
    st = request.app.state.bot_state
    if st.status_flags_dict()["killed"]:
        raise HTTPException(status_code=400, detail="bot killed; restart service")
    st.set_manual_pause(False)
    request.app.state.storage.insert_bot_event(
        utc_now().isoformat(),
        EventSeverity.INFO.value,
        "control_resume",
        "manual resume",
        None,
    )
    return {"status": "running"}


@router.post("/control/flatten")
def control_flatten(request: Request) -> dict[str, str | None]:
    _require_control(request)
    r = request.app.state.bot.flatten(blocking=True)
    return {"flatten_result": r.value if r else None}


@router.post("/control/kill")
def control_kill(request: Request, reason: Optional[str] = None) -> dict[str, str]:
    _require_control(request)
    request.app.state.bot.kill(reason or "manual_kill")
    return {"status": "killed"}


@router.post("/control/drain")
def control_drain(request: Request) -> dict[str, Any]:
    """De-risk the bot IN PLACE (pause quoting + cancel resting orders)
    WITHOUT stopping the process — the first phase of the operator's
    finalized stop, run while the bot is still up so the subsequent
    HTTP snapshot captures live state (``state_current.json`` +
    venue REST cross-checks only exist while the process runs).

    DELIBERATELY UNGATED — does NOT call ``_require_control``, unlike
    the other ``/control/*`` routes. Rationale: ``/control/pause``,
    ``/flatten`` and ``/kill`` can move positions or take the bot down,
    so they fail-closed behind ``CONTROL_ENDPOINTS_ENABLED``.
    ``/control/drain`` is pure DE-RISK: it only pauses new quoting and
    cancels our OWN resting orders (cancel-by-id needs no price and
    moves no funds); the inventory position is LEFT INTACT. The worst
    an unauthorized caller could do is stop the bot quoting and clear
    its book — strictly safe (the SIGTERM shutdown does exactly that).
    Keeping it ungated lets the colo stop script invoke it over the
    loopback API without the operator flipping
    ``CONTROL_ENDPOINTS_ENABLED`` (which would also expose flatten/kill).

    ``Bot.drain`` self-logs ``control_drain_begin`` +
    ``control_drain_complete`` / ``control_drain_incomplete`` bot
    events, so this thin route adds no extra event row.
    """
    bot = request.app.state.bot
    result = bot.drain()
    return {"status": "drained", **result}


@router.get("/db/download")
def db_download(request: Request) -> FileResponse:
    """Stream the current SQLite file for offline analysis.

    Gated on ``DB_DOWNLOAD_ENABLED`` (default ``True``). Intentionally
    separate from the ``CONTROL_ENDPOINTS_ENABLED`` guard used for the
    POST ``/control/*`` routes — downloading the DB is a read-only
    operation and doesn't need the same "operator action" protection
    that the mutating routes do.

    Before streaming, the WAL is consolidated into the main file via
    ``PRAGMA wal_checkpoint(TRUNCATE)`` so the downloaded file is
    self-contained and doesn't require the ``-wal`` / ``-shm`` sidecars.

    Typical workflow: pull the file to a laptop, open it in DBeaver /
    sqlite3, run ad-hoc SQL over multi-day history. Operators who want
    trading-history downloads to require an explicit opt-in can set
    ``DB_DOWNLOAD_ENABLED=false``; the JSON ``/*/since`` endpoints
    still return the same data (capped at 10k rows per table).
    """
    if not request.app.state.settings.db_download_enabled:
        raise HTTPException(status_code=404, detail="db download disabled")
    storage = request.app.state.storage
    db_path = Path(storage._path)
    if not db_path.is_file():
        raise HTTPException(status_code=404, detail=f"DB file not found at {db_path}")
    # Checkpoint the WAL so the main file is a complete snapshot. TRUNCATE
    # mode blocks briefly on contention with the writer; acceptable for
    # an on-demand operator download (the bot's quote loop is fine with
    # a short write pause).
    with storage._lock:
        with storage.connection() as conn:
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                # Best-effort: if checkpoint fails for any reason (e.g.
                # concurrent writer holding the exclusive lock), serve
                # the file as-is. The downloaded file may miss the last
                # few un-checkpointed rows until the next checkpoint.
                pass
    return FileResponse(
        path=db_path,
        media_type="application/octet-stream",
        filename=db_path.name,
    )
