"""TODO-001: inventory consistency watchdog.

Periodic three-way invariant check on the position state:

* **venue truth** — ``state.position.position_qty`` (refreshed each tick
  by ``refresh_account_only`` from the venue REST endpoint).
* **baseline + session fills** — ``inventory_baseline_qty`` (captured
  from the startup seed) plus ``session_signed_qty_total`` (signed sum
  of session-scoped fills).

In healthy operation the two views agree to within rounding. Divergence
beyond tolerance signals a class of bug where local state has drifted
from venue silently — the canonical example was BUG-005, where the
hardcoded ``hl_account_address`` made every Bluefin tick zero out the
position state. With the v1.0.8 fix that root cause is gone, but the
class of bug isn't closed; this watchdog is the generic safety net so
a future regression can't accumulate undetected.

On breach: log CRITICAL ``inventory_consistency_breach`` and return a
payload describing the gap. The caller is responsible for arming
``manual_pause`` and emitting Telegram. Storage / Telegram side effects
are kept out of this module to keep it pure for testing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from app import clock as _clock

if TYPE_CHECKING:
    from app.config import Settings
    from app.state import BotState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InventoryBreach:
    """Payload describing an inventory-consistency divergence."""

    venue_qty: float
    expected_qty: float
    baseline_qty: float
    session_signed_qty: float
    drift: float
    tolerance: float
    symbol: str

    def to_payload(self) -> dict[str, float | str]:
        return {
            "venue_qty": round(self.venue_qty, 6),
            "expected_qty": round(self.expected_qty, 6),
            "baseline_qty": round(self.baseline_qty, 6),
            "session_signed_qty": round(self.session_signed_qty, 6),
            "drift": round(self.drift, 6),
            "tolerance": round(self.tolerance, 6),
            "symbol": self.symbol,
        }


def _tolerance(settings: "Settings", expected_qty: float) -> float:
    return max(
        float(settings.inventory_consistency_tolerance_qty),
        float(settings.inventory_consistency_tolerance_pct) * abs(expected_qty),
    )


def check_inventory_consistency(
    state: "BotState", settings: "Settings"
) -> Optional[InventoryBreach]:
    """Run one consistency check; rate-limited internally so callers can
    invoke unconditionally per tick.

    Returns ``InventoryBreach`` on divergence, ``None`` otherwise (including
    rate-limited skips, disabled, or pre-baseline). Bumps
    ``state.inventory_consistency_breach_count`` and stores
    ``inventory_consistency_last_breach`` payload on breach.
    """
    if not bool(settings.inventory_consistency_enabled):
        return None
    interval = float(settings.inventory_consistency_check_seconds)
    if interval <= 0:
        return None

    now_m = _clock.monotonic()
    last = state.inventory_consistency_last_check_mono
    if last is not None and (now_m - last) < interval:
        return None
    state.inventory_consistency_last_check_mono = now_m

    if not state.inventory_baseline_set:
        # Baseline not captured yet — no comparison possible.
        return None

    venue_qty = float(getattr(state.position, "position_qty", 0.0) or 0.0)
    baseline = float(state.inventory_baseline_qty)
    sess_signed = float(state.session_signed_qty_total)
    expected = baseline + sess_signed
    drift = venue_qty - expected
    tol = _tolerance(settings, expected)

    if abs(drift) <= tol:
        # Clean read — reset the consecutive counter. A real divergence
        # has to be SUSTAINED across consecutive checks, not transient.
        state.inventory_consistency_consecutive_drift_count = 0
        return None

    # Drift exceeded tolerance. Bump the consecutive counter and only
    # declare a breach once it reaches the configured threshold. This
    # protects against transient ``fetch_position`` glitches that briefly
    # report a wrong qty (one such read on 2026-04-26 cost 1.5 h of
    # paused trading; see snap_20260426_103520 + the breach reproducer
    # in the doc-comment of the settings field).
    state.inventory_consistency_consecutive_drift_count += 1
    consec = state.inventory_consistency_consecutive_drift_count
    required = max(
        1,
        int(
            settings.inventory_consistency_consecutive_breaches_required
        ),
    )
    if consec < required:
        # Confirmed-divergent count not yet at threshold — log a WARNING
        # so the operator can see the suspicion building, but do NOT
        # declare a breach (don't pause, don't cancel resting orders).
        logger.warning(
            "inventory_consistency_drift_unconfirmed consec=%d/%d "
            "venue_qty=%.6f expected_qty=%.6f drift=%.6f tol=%.6f symbol=%s",
            consec,
            required,
            venue_qty,
            expected,
            drift,
            tol,
            str(getattr(state.position, "symbol", settings.symbol)),
        )
        return None

    breach = InventoryBreach(
        venue_qty=venue_qty,
        expected_qty=expected,
        baseline_qty=baseline,
        session_signed_qty=sess_signed,
        drift=drift,
        tolerance=tol,
        symbol=str(getattr(state.position, "symbol", settings.symbol)),
    )
    state.inventory_consistency_breach_count += 1
    state.inventory_consistency_last_breach = dict(breach.to_payload())
    # Reset the consecutive counter once we've fired so the breach
    # handler's actions (manual_pause + cancel resting + Telegram) are
    # the cool-down, not another threshold-reached fire.
    state.inventory_consistency_consecutive_drift_count = 0
    logger.critical(
        "inventory_consistency_breach venue_qty=%.6f expected_qty=%.6f "
        "drift=%.6f tolerance=%.6f symbol=%s consecutive_required=%d",
        breach.venue_qty,
        breach.expected_qty,
        breach.drift,
        breach.tolerance,
        breach.symbol,
        required,
        extra={"extra_data": breach.to_payload()},
    )
    return breach
