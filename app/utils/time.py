"""Time-helper utilities.

v1.4.231 (Phase 1c): these helpers now route through the
module-level clock proxy in ``app.clock`` so all 55+ call sites
across ``app/`` that use ``utc_now()`` / ``utc_now_iso()`` /
``seconds_since(...)`` automatically get replay-clock behaviour
under backtesting WITHOUT touching any of those call sites.

Under production (``set_module_clock`` either uncalled OR called
with ``SystemClock``) the behaviour is bit-identical to the
pre-v1.4.231 implementation that called ``datetime.now(timezone.utc)``
directly.

Under backtest replay (``set_module_clock(ReplayClock(...))``
installed before bot code reads time), every helper here returns
the replay-time equivalent.
"""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> datetime:
    """Tz-aware UTC ``datetime`` at the current clock instant.

    Routes through the module-level clock proxy so replay-mode
    callers see captured-data time, not OS wall-clock time.
    """
    # Local import to break the potential circular: many ``app/``
    # modules import ``app.utils.time`` at module load; if THIS
    # function were a top-level ``from app.clock import ...`` we'd
    # risk an import cycle. Local import is cheap at runtime
    # (Python caches the import).
    from app.clock import now_utc as _now_utc
    return _now_utc()


def utc_now_iso() -> str:
    """ISO 8601 string at the current clock instant. Wraps
    ``utc_now().isoformat()``."""
    return utc_now().isoformat()


def seconds_since(ts: datetime | None) -> float | None:
    """Seconds elapsed between ``ts`` and the current clock instant.

    Returns ``None`` when ``ts`` is None (defensive — many caller
    paths have nullable timestamp fields).

    Naïve datetimes are promoted to UTC-aware before the subtraction
    so the result is always meaningful regardless of how the caller
    stored the timestamp.
    """
    if ts is None:
        return None
    now = utc_now()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts).total_seconds()
