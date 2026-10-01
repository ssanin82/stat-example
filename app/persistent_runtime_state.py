"""
Restart-stable operator metrics persistence (JSON file).

Only the fields below are stored — no execution slots, WS, reconcile scratch, or SENT state.

**UTC calendar day:** ``day_anchor_utc`` is the UTC date for ``daily_*`` and ``recent_*`` fill
counts. On startup, if the file's anchor is **before** today's UTC date, the loader **does not**
carry prior-day totals forward: it sets the anchor to today and zeros day-scoped fields (and
clears optional toxicity summaries). The bot also calls ``maybe_rotate_operator_day_to_wall_clock``
once per tick so counters roll forward at UTC midnight without requiring a fill.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Optional

# Bump when the on-disk envelope or field semantics change incompatibly.
# v1 -> v2: added soft_flatten_active + soft_flatten_started_at_iso so a
# crash mid-flatten resumes in the same mode rather than letting the
# bot quote normally on adverse inventory until the drawdown gate
# re-fires.
# v2 -> v3: added basis_regime_state (last sign + IC + pair count) so
# the basis-regime classifier doesn't reset its 240-pair buffer on
# every restart (the first ~5-10 min after restart used to be
# regime-blind). N12 analysis-day instrumentation, 2026-05-10.
PERSISTENT_RUNTIME_STATE_VERSION = 3


@dataclass(frozen=True)
class PersistentRuntimeState:
    """Explicit schema; on-disk version must match SCHEMA_VERSION."""

    SCHEMA_VERSION: ClassVar[int] = PERSISTENT_RUNTIME_STATE_VERSION

    day_anchor_utc: date
    daily_realized_pnl: float
    daily_trade_count: int
    daily_traded_notional: float
    last_fill_ts: Optional[datetime] = None
    recent_buy_fill_count: int = 0
    recent_sell_fill_count: int = 0
    rolling_toxicity_markout_bps: Optional[float] = None
    rolling_one_sided_fill_ratio: Optional[float] = None
    # Soft-flatten resume bridge. When the bot crashes / restarts
    # mid-flatten, the position seeds back from the venue but the
    # flatten *intent* would otherwise be lost. Persisting this
    # boolean lets the boot path re-enter SOFT_FLATTENING immediately,
    # so the patient post-only close-out resumes without waiting for
    # another 30s drawdown-gate breach to re-trigger it.
    soft_flatten_active: bool = False
    soft_flatten_started_at_iso: Optional[str] = None
    # N12 analysis-day instrumentation. Snapshot of the basis-regime
    # classifier's terminal state at shutdown. On reload, the bot
    # seeds the classifier so it doesn't go through a 240-pair
    # warmup window after every restart. None when the classifier
    # had no verdict yet (first session, or fewer than
    # ``min_pair_samples`` paired observations).
    basis_regime_last_sign: Optional[float] = None
    basis_regime_last_ic: Optional[float] = None
    basis_regime_pair_count: Optional[int] = None


def _parse_date(v: Any) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v)
        except ValueError:
            return None
    return None


def _parse_datetime(v: Any) -> Optional[datetime]:
    if v is None:
        return None
    if isinstance(v, datetime):
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v
    if isinstance(v, str):
        try:
            # fromisoformat handles ...Z in 3.11+
            s = v.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


def _to_payload_dict(state: PersistentRuntimeState) -> dict[str, Any]:
    return {
        "day_anchor_utc": state.day_anchor_utc.isoformat(),
        "daily_realized_pnl": state.daily_realized_pnl,
        "daily_trade_count": state.daily_trade_count,
        "daily_traded_notional": state.daily_traded_notional,
        "last_fill_ts": state.last_fill_ts.isoformat() if state.last_fill_ts else None,
        "recent_buy_fill_count": state.recent_buy_fill_count,
        "recent_sell_fill_count": state.recent_sell_fill_count,
        "rolling_toxicity_markout_bps": state.rolling_toxicity_markout_bps,
        "rolling_one_sided_fill_ratio": state.rolling_one_sided_fill_ratio,
        "soft_flatten_active": bool(state.soft_flatten_active),
        "soft_flatten_started_at_iso": state.soft_flatten_started_at_iso,
        "basis_regime_last_sign": state.basis_regime_last_sign,
        "basis_regime_last_ic": state.basis_regime_last_ic,
        "basis_regime_pair_count": state.basis_regime_pair_count,
    }


def _coerce_int(v: Any) -> Optional[int]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    return None


def _coerce_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _from_payload_dict(data: dict[str, Any]) -> Optional[PersistentRuntimeState]:
    d_anchor = _parse_date(data.get("day_anchor_utc"))
    if d_anchor is None:
        return None

    drp = _coerce_float(data.get("daily_realized_pnl"))
    dtc = _coerce_int(data.get("daily_trade_count"))
    dtn = _coerce_float(data.get("daily_traded_notional"))
    if drp is None or dtc is None or dtn is None:
        return None
    if dtc < 0:
        return None

    rbc = _coerce_int(data.get("recent_buy_fill_count"))
    rsc = _coerce_int(data.get("recent_sell_fill_count"))
    if rbc is None or rsc is None or rbc < 0 or rsc < 0:
        return None

    last_ts = _parse_datetime(data.get("last_fill_ts"))

    rtm = data.get("rolling_toxicity_markout_bps")
    if rtm is not None:
        rtm_f = _coerce_float(rtm)
        if rtm_f is None:
            return None
        rtm = rtm_f

    ros = data.get("rolling_one_sided_fill_ratio")
    if ros is not None:
        ros_f = _coerce_float(ros)
        if ros_f is None:
            return None
        ros = ros_f

    sf_active = bool(data.get("soft_flatten_active", False))
    sf_started = data.get("soft_flatten_started_at_iso")
    if sf_started is not None and not isinstance(sf_started, str):
        sf_started = None  # bad type → silently drop the timer

    # N12: basis-regime classifier state. All optional; missing on
    # files written by v2-and-earlier runtime versions. Bad types are
    # silently dropped (None) so a corrupt regime field can't block
    # the rest of the operator-metrics restore.
    basis_sign = _coerce_float(data.get("basis_regime_last_sign"))
    basis_ic = _coerce_float(data.get("basis_regime_last_ic"))
    basis_pairs = _coerce_int(data.get("basis_regime_pair_count"))
    if basis_pairs is not None and basis_pairs < 0:
        basis_pairs = None

    return PersistentRuntimeState(
        day_anchor_utc=d_anchor,
        daily_realized_pnl=drp,
        daily_trade_count=dtc,
        daily_traded_notional=dtn,
        last_fill_ts=last_ts,
        recent_buy_fill_count=rbc,
        recent_sell_fill_count=rsc,
        rolling_toxicity_markout_bps=rtm,
        rolling_one_sided_fill_ratio=ros,
        soft_flatten_active=sf_active,
        soft_flatten_started_at_iso=sf_started,
        basis_regime_last_sign=basis_sign,
        basis_regime_last_ic=basis_ic,
        basis_regime_pair_count=basis_pairs,
    )


def load_persistent_runtime_state(path: str | Path) -> Optional[PersistentRuntimeState]:
    """
    Load operator metrics from path. Returns None if the file is missing, corrupt,
    not a dict, wrong version, or fails validation — never raises.
    """
    p = Path(path)
    try:
        if not p.is_file():
            return None
        raw = p.read_text(encoding="utf-8")
        obj = json.loads(raw)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None
    if obj.get("version") != PERSISTENT_RUNTIME_STATE_VERSION:
        return None
    data = obj.get("data")
    if not isinstance(data, dict):
        return None
    try:
        return _from_payload_dict(data)
    except Exception:
        return None


def save_persistent_runtime_state(path: str | Path, state: PersistentRuntimeState) -> None:
    """Atomic write: temp file in the same directory, then os.replace."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    envelope: dict[str, Any] = {
        "version": PERSISTENT_RUNTIME_STATE_VERSION,
        "data": _to_payload_dict(state),
    }
    text = json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=False)
    tmp = p.with_name(p.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8", newline="\n")
        os.replace(tmp, p)
    finally:
        try:
            if tmp.is_file():
                tmp.unlink(missing_ok=True)
        except OSError:
            pass
