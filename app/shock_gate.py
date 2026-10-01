"""Shock-magnitude defence gate — Phase 1B (v1.4.107).

The "stronger shock rule" from Codex Section 6. Designed for the
06:50-style event in snapshot 260520-074215 where the mid snapped
-100 bp in ~30 seconds while the bot was already maxed long. Even
the post-Phase-1A `inventory_drift_gate` (15 bp / 10 s) widens —
it doesn't stand the bot down — and at SHOCK magnitudes a
wider-quote market is still a participating market.

**Binary by design.** Unlike `inventory_drift_gate`, this gate is
NOT a widening contributor. When it fires, the bot is *out* on the
suppressed side, full stop. Two reasons:

1. At SHOCK magnitudes (15-25 bp / 10 s OR 40-60 bp / 30 s) the
   classical adverse-selection model breaks down — the price move
   is fast enough that quote freshness is the dominant risk, not
   spread economics. Wider quotes don't fix freshness.
2. The bot's existing `quote_eligibility` pipeline already speaks
   binary overrides (`QUOTE_BUY_ONLY` / `QUOTE_SELL_ONLY` /
   `HOLD_ALL`) — using that channel keeps shock containment in
   the same code path that handles freshness holds, kill switches,
   and other hard-no signals.

Trigger conditions (all three must hold):

* Util ≥ `shock_inventory_pct_threshold` (default 0.80) — the gate
  only protects an already-loaded bot. At low inventory the bot
  has nothing to lose from a shock; the existing gates are
  sufficient.
* Drift sign × position sign < 0 — anti-aligned. The shock is
  hurting the held inventory.
* |drift_10s| ≥ shock_threshold_10s OR |drift_30s| ≥ shock_threshold_30s.
  Either window suffices; the magnitudes are tuned higher than
  `inventory_drift_gate`'s thresholds so this gate only fires on
  genuine shock events, not on the same slow grinds the widening
  gate handles.

Cooldown semantics (the gate is "sticky"):

* Once fired, the lock persists until **both** clear conditions
  are true simultaneously:
  - Util drops below `clear_util_threshold` (default 0.30), AND
  - Latest 30s drift magnitude < shock_threshold_30s / 3.
* A maximum-cooldown timestamp (`shock_max_cooldown_seconds`,
  default 300 s) is also tracked as a safety net so the bot is
  never stuck in shock-cooldown beyond a hard ceiling, even if a
  signal-feed glitch keeps the clear-conditions from triggering.

The cooldown direction (which side stays suppressed) is **frozen**
at the moment of fire — even if the position turns flat or flips
during cooldown, the original suppressed side stays suppressed.
This prevents fast-flip pattern: in 260520-074215 the bot opened
SHORT into the bounce within 60 s of unwinding its long; freezing
the lock direction prevents that.

Telemetry: every fire bumps `fire_count`, logs a WARNING via
`bot._log_event` (NO Telegram alert at Phase 1B — that lands in
Phase 2G). The `behavioural_gates.shock_gate` block on
`state_current.json` and live_stats heartbeat carries `active`,
`locked_side`, `seconds_in_lock`, `last_trigger_drift_bps`,
`last_trigger_util`, `fire_count`.

Config (TON profile):

    SHOCK_GATE_ENABLED=true
    SHOCK_INVENTORY_PCT_THRESHOLD=0.80
    SHOCK_THRESHOLD_BPS_10S=20.0
    SHOCK_THRESHOLD_BPS_30S=50.0
    SHOCK_CLEAR_UTIL_THRESHOLD=0.30
    SHOCK_MAX_COOLDOWN_SECONDS=300.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from app.enums import QuoteEligibility


@dataclass
class ShockGateState:
    """Per-bot mutable state, owned by ``BotState``.

    Single-threaded ownership: the bot's main quote loop is the
    sole writer (`observe()` mutates in place). The dashboard /
    state-publisher reads from it lock-free under the GIL — every
    field is either a primitive (int/float/bool) or an immutable
    enum value, so a torn read can produce only an inconsistent
    snapshot, never an undefined behaviour.
    """

    locked: bool = False
    """True while the gate's binary override is active."""

    locked_side: Optional[QuoteEligibility] = None
    """The override eligibility being applied while locked.
    Frozen at fire time — does not change during cooldown."""

    locked_at_mono: Optional[float] = None
    """Monotonic timestamp of the lock activation. Used to compute
    seconds-in-lock for telemetry and to enforce the safety-net
    max-cooldown."""

    max_cooldown_until_mono: float = 0.0
    """Hard ceiling on the cooldown duration — gate auto-clears at
    this timestamp even if the soft-clear conditions never trigger.
    Set at fire time to `locked_at_mono + shock_max_cooldown_seconds`."""

    last_trigger_drift_bps: float = 0.0
    last_trigger_util: float = 0.0
    last_trigger_window_label: str = ""
    """Diagnostics from the most recent fire. Surfaced in
    `behavioural_gates.shock_gate` for postmortem + live dashboard."""

    fire_count: int = 0


def observe(
    state: ShockGateState,
    *,
    now_mono: float,
    position_qty: float,
    effective_abs_cap: float,
    drift_bps_10s: Optional[float],
    drift_bps_30s: Optional[float],
    enabled: bool,
    shock_inventory_pct_threshold: float,
    shock_threshold_bps_10s: float,
    shock_threshold_bps_30s: float,
    clear_util_threshold: float,
    max_cooldown_seconds: float,
) -> tuple[Optional[QuoteEligibility], str]:
    """Apply the shock-gate logic to one quote-loop tick. Returns
    `(override_eligibility, reason)`.

    * `(None, "shock_gate_disabled")` — gate not enabled.
    * `(None, "shock_gate_idle")` — gate enabled but no lock.
    * `(QuoteEligibility.QUOTE_SELL_ONLY | QUOTE_BUY_ONLY, "shock_gate_...")`
      — gate locked, this tick is suppressed on one side.
    * `(None, "shock_gate_cleared")` — lock just cleared this tick.

    Mutates `state` in place when the lock-state changes. Idempotent
    when called multiple times per tick with the same inputs.

    Two code paths:

    1. **Currently locked** — check soft-clear conditions (util AND
       drift normalised) and hard-clear (max-cooldown timestamp).
       Either path returns to idle; otherwise emit the existing lock
       side.
    2. **Not locked** — check fire conditions. If all three hold
       (util high enough, drift anti-aligned, drift magnitude past
       threshold), activate the lock.
    """
    if not enabled:
        return None, "shock_gate_disabled"
    if effective_abs_cap <= 0 or not math.isfinite(effective_abs_cap):
        return None, "shock_gate_no_cap"
    if not math.isfinite(position_qty):
        return None, "shock_gate_no_position"

    util = abs(float(position_qty)) / float(effective_abs_cap)

    # ------------------------------------------------------------------
    # Path 1: currently locked. Check whether either clear condition
    # has triggered.
    # ------------------------------------------------------------------
    if state.locked:
        # Hard clear — max-cooldown ceiling. Safety net; we never want
        # a locked-forever state on a feed glitch.
        if now_mono >= state.max_cooldown_until_mono > 0:
            state.locked = False
            prior_side = state.locked_side
            state.locked_side = None
            state.locked_at_mono = None
            return None, f"shock_gate_cleared:max_cooldown(prior={prior_side})"

        # Soft clear — both conditions must hold simultaneously.
        latest_drift_mag = 0.0
        if drift_bps_30s is not None and math.isfinite(drift_bps_30s):
            latest_drift_mag = abs(float(drift_bps_30s))
        clear_drift_threshold = float(shock_threshold_bps_30s) / 3.0
        util_cleared = util < float(clear_util_threshold)
        drift_cleared = latest_drift_mag < clear_drift_threshold
        if util_cleared and drift_cleared:
            prior_side = state.locked_side
            state.locked = False
            state.locked_side = None
            state.locked_at_mono = None
            return (
                None,
                (
                    f"shock_gate_cleared:util={util:.3f},"
                    f"drift30s={latest_drift_mag:.2f}bps,prior={prior_side}"
                ),
            )

        # Still locked. Emit the existing override.
        elapsed = now_mono - (state.locked_at_mono or now_mono)
        return (
            state.locked_side,
            (
                f"shock_gate_locked:side={state.locked_side},"
                f"elapsed={elapsed:.1f}s,util={util:.3f},"
                f"drift30s={latest_drift_mag:.2f}bps"
            ),
        )

    # ------------------------------------------------------------------
    # Path 2: not locked. Check fire conditions.
    # ------------------------------------------------------------------
    if util < float(shock_inventory_pct_threshold):
        return None, f"shock_gate_idle:util_below({util:.3f})"

    pos_sign = 1.0 if position_qty > 0 else -1.0

    # Pick the strongest signal across the two windows.
    candidates: list[tuple[float, str, float]] = []
    if (
        drift_bps_10s is not None
        and math.isfinite(drift_bps_10s)
        and abs(drift_bps_10s) >= float(shock_threshold_bps_10s)
        and (drift_bps_10s * pos_sign) < 0
    ):
        candidates.append(
            (
                float(drift_bps_10s),
                "10s",
                float(shock_threshold_bps_10s),
            )
        )
    if (
        drift_bps_30s is not None
        and math.isfinite(drift_bps_30s)
        and abs(drift_bps_30s) >= float(shock_threshold_bps_30s)
        and (drift_bps_30s * pos_sign) < 0
    ):
        candidates.append(
            (
                float(drift_bps_30s),
                "30s",
                float(shock_threshold_bps_30s),
            )
        )

    if not candidates:
        return None, f"shock_gate_idle:no_shock(util={util:.3f})"

    # Fire — pick the largest-|drift| window for telemetry.
    chosen = max(candidates, key=lambda c: abs(c[0]))
    drift_used, window_label, threshold = chosen
    direction = "down" if drift_used < 0 else "up"
    locked_side = (
        QuoteEligibility.QUOTE_SELL_ONLY
        if drift_used < 0
        else QuoteEligibility.QUOTE_BUY_ONLY
    )

    state.locked = True
    state.locked_side = locked_side
    state.locked_at_mono = now_mono
    state.max_cooldown_until_mono = now_mono + max(0.0, float(max_cooldown_seconds))
    state.last_trigger_drift_bps = float(drift_used)
    state.last_trigger_util = float(util)
    state.last_trigger_window_label = window_label
    state.fire_count = int(state.fire_count) + 1

    reason = (
        f"shock_gate_fire:{direction},util={util:.3f},"
        f"drift_{window_label}={drift_used:+.2f}bps>={threshold:.2f}"
    )
    return locked_side, reason


def seconds_in_lock(state: ShockGateState, now_mono: float) -> Optional[float]:
    """Telemetry helper for the live dashboard / behavioural_gates
    snapshot."""
    if not state.locked or state.locked_at_mono is None:
        return None
    return max(0.0, float(now_mono) - float(state.locked_at_mono))


def snapshot_dict(
    state: ShockGateState, now_mono: float
) -> dict[str, object]:
    """JSON-safe payload for `behavioural_gates.shock_gate`.

    Fields render even when the gate is dormant — operator dashboard
    shows the block unconditionally per memory note
    `feedback_bot_stats_panels_always_render`.
    """
    locked_side = state.locked_side
    locked_side_str: Optional[str]
    if locked_side is None:
        locked_side_str = None
    else:
        locked_side_str = getattr(locked_side, "name", str(locked_side))
    in_lock = seconds_in_lock(state, now_mono)
    return {
        "active": bool(state.locked),
        "locked_side": locked_side_str,
        "seconds_in_lock": (
            None if in_lock is None else round(float(in_lock), 2)
        ),
        "last_trigger_drift_bps": float(state.last_trigger_drift_bps),
        "last_trigger_util": float(state.last_trigger_util),
        "last_trigger_window": state.last_trigger_window_label or None,
        "fire_count": int(state.fire_count),
    }
