"""Per-session operational + economic summary.

Computes a single rollup suitable for a one-shot ``GET /session/summary``
response. Does NOT read from state — takes already-fetched rows. Stays
stateless and testable in isolation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from app.pnl_attribution import compute_attribution


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        # Accept both "Z" and "+00:00" suffixes.
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def build_session_summary(
    *,
    session_id: str,
    session_started_at_utc: str,
    now_utc: str,
    fills: list[dict[str, Any]],
    equity_history: list[dict[str, Any]],
    realized_pnl_total_usd: Optional[float],
    markout_horizon_s: int = 5,
    bot_status: Optional[str] = None,
    key_counters: Optional[dict[str, Any]] = None,
    experiment_id: Optional[str] = None,
) -> dict[str, Any]:
    """Session-scoped rollup.

    Args:
        session_id: UUID of the current bot session.
        session_started_at_utc: ISO-8601 timestamp (string) of session start.
        now_utc: ISO-8601 timestamp of the report instant.
        fills: Fills from the session window, oldest-first preferred.
        equity_history: Equity snapshots from the session window.
        realized_pnl_total_usd: Current realized PnL (negative for loss).
        markout_horizon_s: Horizon for the attribution block (1, 3, or 5).
        bot_status: Current bot status ("RUNNING", "PAUSED", etc.).
        key_counters: Free-form dict of session counters the caller wants to
            surface (ws reconnect counts, adverse-side-pause skips, etc.).
        experiment_id: Free-form operator tag for cross-session A/B testing
            (N7 analysis-day instrumentation, 2026-05-10). Set via the
            ``EXPERIMENT_ID`` env var at session start; commonly a config
            hash or a human label like "cap-4x-2026-05-09". When unset,
            ``None`` and the field is omitted from the response.
    """
    start_dt = _parse_iso(session_started_at_utc)
    now_dt = _parse_iso(now_utc)
    duration_s: Optional[float] = None
    if start_dt is not None and now_dt is not None:
        duration_s = max(0.0, (now_dt - start_dt).total_seconds())

    fills_count = len(fills)
    fill_rate_per_min: Optional[float] = None
    if duration_s and duration_s > 0:
        fill_rate_per_min = fills_count / (duration_s / 60.0)

    # Max drawdown from equity history. drawdown_usd field is already tracked
    # by the bot; we just want the peak.
    max_drawdown_usd = 0.0
    for e in equity_history:
        try:
            d = float(e.get("drawdown_usd") or 0.0)
        except (TypeError, ValueError):
            d = 0.0
        if d > max_drawdown_usd:
            max_drawdown_usd = d

    # Equity trajectory endpoints.
    equity_start_usd: Optional[float] = None
    equity_end_usd: Optional[float] = None
    fees_cumulative_usd: Optional[float] = None
    if equity_history:
        first = equity_history[0]
        last = equity_history[-1]
        try:
            equity_start_usd = float(first.get("equity_usd") or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            equity_end_usd = float(last.get("equity_usd") or 0.0)
        except (TypeError, ValueError):
            pass
        try:
            fees_cumulative_usd = float(last.get("fees_usd") or 0.0)
        except (TypeError, ValueError):
            pass

    attribution = compute_attribution(
        fills=fills,
        horizon_s=markout_horizon_s,
        window_since_iso=session_started_at_utc,
        window_until_iso=now_utc,
        realized_pnl_total_usd=realized_pnl_total_usd,
    )

    # v1.4.98 — multi-horizon attribution. The canonical ``attribution``
    # block stays at the configured horizon (default 5s) so all existing
    # consumers and the postmortem fatal/warn gates are unchanged. The
    # additional ``markouts_multi_horizon`` block exposes the SAME
    # rebate/markout-dollar-impact/residual decomposition at each of
    # the supported diagnostic horizons. ``residual`` at the 120s
    # horizon should approach the unrealised PnL on still-open
    # inventory — anything beyond that is genuinely uncaptured.
    #
    # Builds defensively: if any horizon raises (insufficient samples,
    # field missing on legacy fills), the slot is omitted. The
    # canonical 5s block survives independently of the multi-horizon
    # extra.
    multi_horizon_attributions: dict[str, Any] = {}
    for h in (1, 3, 5, 15, 30, 60, 120):
        if h == markout_horizon_s:
            # Already computed above; surface a reference so consumers
            # don't have to special-case "which key holds the canonical
            # horizon".
            multi_horizon_attributions[f"{h}s"] = {
                "markout": attribution.get("markout"),
                "pnl_attribution_usd": attribution.get("pnl_attribution_usd"),
                "is_canonical": True,
            }
            continue
        try:
            att_h = compute_attribution(
                fills=fills,
                horizon_s=h,
                window_since_iso=session_started_at_utc,
                window_until_iso=now_utc,
                realized_pnl_total_usd=realized_pnl_total_usd,
            )
            multi_horizon_attributions[f"{h}s"] = {
                "markout": att_h.get("markout"),
                "pnl_attribution_usd": att_h.get("pnl_attribution_usd"),
                "is_canonical": False,
            }
        except Exception:
            # Insufficient samples / missing column / etc. — skip the
            # horizon rather than fail the whole summary build.
            continue

    return {
        "session_id": session_id,
        "session_started_at_utc": session_started_at_utc,
        "now_utc": now_utc,
        "duration_seconds": None if duration_s is None else round(duration_s, 2),
        "bot_status": bot_status,
        "experiment_id": experiment_id,
        "equity": {
            "start_usd": None if equity_start_usd is None else round(equity_start_usd, 6),
            "end_usd": None if equity_end_usd is None else round(equity_end_usd, 6),
            "change_usd": (
                None
                if (equity_start_usd is None or equity_end_usd is None)
                else round(equity_end_usd - equity_start_usd, 6)
            ),
            "max_drawdown_usd": round(max_drawdown_usd, 6),
            "fees_cumulative_usd": (
                None if fees_cumulative_usd is None else round(fees_cumulative_usd, 6)
            ),
        },
        "throughput": {
            "fills_count": fills_count,
            "fill_rate_per_min": (
                None if fill_rate_per_min is None else round(fill_rate_per_min, 3)
            ),
            "total_notional_usd": attribution["fills"]["total_notional_usd"],
        },
        "attribution": attribution,
        # v1.4.98: per-horizon markout decomposition. Diagnostic only;
        # the canonical ``attribution`` block above is what gates /
        # dashboards anchor on.
        "markouts_multi_horizon": multi_horizon_attributions,
        "key_counters": dict(key_counters) if key_counters else {},
    }
