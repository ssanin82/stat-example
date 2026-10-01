"""Best-effort session resume on restart.

Reads the most recent ``bot_start`` event from the SQLite ``bot_events``
table; if the prior session is within ``BOT_SESSION_CONTINUITY_MAX_HOURS``
of now, restores cumulative METRIC state (session_id, session_started_at_utc,
session_fill_count, realized_pnl_usd, fees_usd, peak_equity_usd) by
re-aggregating from ``fills`` and ``equity_snapshots``.

Hard rule
---------
Only METRIC state is restored. Behavioral state (pending-cancel dicts,
cancel-confirmation gates, cooldown timers, toxicity window samples,
position cache) is NEVER touched. Restoring those would defeat the
deadlock watchdog — re-introducing the very latched state the restart
just cleared. There is a regression test that asserts none of the
behavioral fields change after :func:`try_resume_session_from_storage`.

Failure modes
-------------
The function is intentionally non-raising: any storage failure, missing
prior session, expired session, or malformed payload returns a status
string and leaves state unchanged. The bot startup path must not be
blocked by a resume failure — fall through to fresh start cleanly.

Volume-wipe behavior
--------------------
When the operator wipes the persistent volume, both the SQLite DB and
any sidecar files disappear together. The next startup finds no prior
``bot_start`` event and reports ``no_prior_session`` (not ``error``).
That is the intended behavior: empty volume = clean fresh start.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from app import clock as _clock

logger = logging.getLogger(__name__)


@dataclass
class SessionResumeResult:
    """Outcome of a resume attempt.

    ``status`` values:
        - ``"resumed"`` — prior session within window; metrics restored
        - ``"fresh"`` — feature disabled (default); no action
        - ``"expired"`` — prior session older than continuity window
        - ``"no_prior_session"`` — no ``bot_start`` event in storage yet
        - ``"error"`` — storage read or parse failed
    """

    status: str
    reason: str = ""
    session_id: Optional[str] = None
    session_started_at_utc: Optional[datetime] = None
    fill_count: int = 0
    realized_pnl_usd: float = 0.0
    fees_usd: float = 0.0
    peak_equity_usd: Optional[float] = None
    age_hours: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def telegram_summary(self) -> str:
        """One-line summary suitable for an ops Telegram message."""
        if self.status == "resumed":
            sid = (self.session_id or "?")[:8]
            return (
                f"resumed session {sid} (age {self.age_hours:.1f}h): "
                f"{self.fill_count} fills, realized=${self.realized_pnl_usd:.4f}, "
                f"fees=${self.fees_usd:.4f}"
            )
        if self.status == "expired":
            return f"prior session expired ({self.reason}); starting fresh"
        if self.status == "no_prior_session":
            return "no prior session in storage; starting fresh"
        if self.status == "fresh":
            return "session resume disabled; starting fresh"
        if self.status == "error":
            return f"resume failed: {self.reason}; starting fresh"
        return f"resume status={self.status}: {self.reason}"


def try_resume_session_from_storage(
    settings: Any, state: Any, storage: Any
) -> SessionResumeResult:
    """Best-effort restore of cumulative session metrics. Never raises.

    Returns a :class:`SessionResumeResult`. The caller is expected to
    emit a Telegram ops notification with :meth:`telegram_summary`.
    """
    if not bool(getattr(settings, "bot_resume_session_on_restart", False)):
        return SessionResumeResult(status="fresh", reason="resume disabled")

    max_hours = float(
        getattr(settings, "bot_session_continuity_max_hours", 24.0)
    )

    # 1. Find most recent prior bot_start event.
    try:
        events = storage.recent_bot_events(limit=200)
    except Exception as exc:  # noqa: BLE001
        return SessionResumeResult(
            status="error", reason=f"recent_bot_events: {exc}"
        )

    prior: Optional[dict[str, Any]] = None
    for e in events:
        if (e.get("event_type") or "") == "bot_start":
            prior = e
            break

    if prior is None:
        return SessionResumeResult(
            status="no_prior_session", reason="no bot_start event in storage"
        )

    # 2. Extract session_id + start_ts from payload, falling back to event ts.
    payload: dict[str, Any] = {}
    try:
        raw = prior.get("payload_json")
        if isinstance(raw, str) and raw:
            payload = json.loads(raw) or {}
    except Exception:  # noqa: BLE001
        payload = {}

    prev_session_id = payload.get("session_id")
    prev_started_iso = (
        payload.get("session_started_at_utc")
        or prior.get("ts")
        or ""
    )
    if not isinstance(prev_started_iso, str) or not prev_started_iso:
        return SessionResumeResult(
            status="error", reason="prior session timestamp missing"
        )
    if not prev_session_id or not isinstance(prev_session_id, str):
        # No session_id in payload — treat as expired/unknown rather than
        # mint a synthetic id. Resume is only meaningful when the prior
        # session can be uniquely identified.
        return SessionResumeResult(
            status="error", reason="prior session_id missing"
        )

    try:
        prev_started_dt = _parse_iso_utc(prev_started_iso)
    except ValueError as exc:
        return SessionResumeResult(
            status="error", reason=f"invalid timestamp: {exc}"
        )

    # 3. Continuity window check.
    now_dt = _clock.now_utc()
    age_seconds = max(0.0, (now_dt - prev_started_dt).total_seconds())
    age_hours = age_seconds / 3600.0
    if age_hours > max_hours:
        return SessionResumeResult(
            status="expired",
            reason=f"age={age_hours:.1f}h > max={max_hours:.1f}h",
            age_hours=age_hours,
        )

    # 4. Aggregate cumulative metrics from fills + equity_snapshots.
    since_iso = prev_started_dt.isoformat()
    try:
        fills = storage.fills_since(since_iso, limit=10000) or []
    except Exception as exc:  # noqa: BLE001
        return SessionResumeResult(
            status="error", reason=f"fills_since: {exc}"
        )
    fill_count = len(fills)
    fees_total = 0.0
    for f in fills:
        v = f.get("fee")
        if v is None:
            continue
        try:
            fees_total += float(v)
        except (TypeError, ValueError):
            continue

    try:
        equity_rows = storage.equity_history_since(since_iso, limit=10000) or []
    except Exception:  # noqa: BLE001
        equity_rows = []
    realized_total = 0.0
    peak_equity: Optional[float] = None
    if equity_rows:
        # Use the latest snapshot as the cumulative realized PnL.
        last = equity_rows[-1]
        v = last.get("realized_pnl_usd")
        if v is not None:
            try:
                realized_total = float(v)
            except (TypeError, ValueError):
                pass
        for r in equity_rows:
            v = r.get("equity_usd")
            if v is None:
                continue
            try:
                e = float(v)
            except (TypeError, ValueError):
                continue
            if peak_equity is None or e > peak_equity:
                peak_equity = e

    # 5. Mutate state — METRICS ONLY.
    try:
        state.apply_session_resume(
            session_id=prev_session_id,
            session_started_at_utc=prev_started_dt,
            session_fill_count=fill_count,
            realized_pnl_usd=realized_total,
            fees_usd=fees_total,
            peak_equity_usd=peak_equity,
        )
    except Exception as exc:  # noqa: BLE001
        return SessionResumeResult(
            status="error", reason=f"state.apply_session_resume: {exc}"
        )

    return SessionResumeResult(
        status="resumed",
        reason="ok",
        session_id=prev_session_id,
        session_started_at_utc=prev_started_dt,
        fill_count=fill_count,
        realized_pnl_usd=realized_total,
        fees_usd=fees_total,
        peak_equity_usd=peak_equity,
        age_hours=age_hours,
    )


def write_bot_start_event(
    storage: Any,
    *,
    session_id: str,
    session_started_at_utc: datetime,
    version: str,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Write a ``bot_start`` event into ``bot_events``.

    The next startup uses this row as the resume anchor: its ``ts``
    column and its ``payload_json`` carry the session_id +
    session_started_at_utc that future :func:`try_resume_session_from_storage`
    calls will read.
    """
    payload: dict[str, Any] = {
        "session_id": session_id,
        "session_started_at_utc": session_started_at_utc.isoformat(),
        "version": version,
    }
    if extra:
        payload.update(extra)
    try:
        storage.insert_bot_event(
            session_started_at_utc.isoformat(),
            "INFO",
            "bot_start",
            f"bot started session={session_id[:8]} version={version}",
            payload,
        )
    except Exception:  # noqa: BLE001
        # Best-effort: never block startup on event write failure.
        logger.exception("write_bot_start_event_failed")


def _parse_iso_utc(s: str) -> datetime:
    """Parse an ISO-8601 timestamp; default to UTC if naive. Accepts a
    trailing 'Z'."""
    candidate = s.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    dt = datetime.fromisoformat(candidate)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
