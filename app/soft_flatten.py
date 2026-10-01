"""Soft-flatten pricing helpers.

Pure functions that decide what price the bot should rest its post-
only reduce order at. Lives in its own module so the staged-pricing
logic is unit-testable without spinning up a full Bot instance.

Staged model (per the operator spec 2026-05-06):
  * Phase 1 (default 5 s): rest at the near-touch -- best_ask for a
    SELL (close-long), best_bid for a BUY (close-short). Joining the
    queue at the maker side; earns rebate on fill.
  * Phase 2 (after phase 1 elapses, indefinite): rest at one tick
    INTO the spread from the far touch -- best_bid+1tick for SELL,
    best_ask-1tick for BUY. This is the most aggressive post-only
    price possible without crossing. On a 1-tick spread it collapses
    to phase 1 (best_ask == best_bid+1, etc.), so the worker stays at
    the near-touch and there is no escalation.
  * No further escalation. The bot stays at phase 2 until the order
    fills or the absolute drawdown / session-loss gates trip the
    hard kill (existing safety floor).

The pure ``compute_target_price`` lets tests cover the corner cases
(1-tick spread collapse, sign of position) without booting a Bot.

Phase 4D extension (v1.4.164) — adaptive aggressiveness ladder
==============================================================

Targets the v1.4.157-260520-213540 failure: a 14-second post-only
chase produced 1,443 place-cancels with 0 fills, then a single
``market_close`` taker that ate the worst spread of the rally. The
extension inserts cross-tick IOC ladders between the legacy 2-phase
post-only ladder and the terminal taker fallback:

  * Phase 0: legacy "phase 1" — post-only at near touch.
  * Phase 1: legacy "phase 2" — post-only 1 tick into the spread.
  * Phase 2 (NEW): IOC limit crossing 1 tick (taker).
  * Phase 3 (NEW): IOC limit crossing 2 ticks (taker).
  * Phase 4 (NEW): terminal ``market_close``.

Each phase has a time-budget; the per-tick evaluator can ALSO skip
ahead based on signal (ask running away fast, or N consecutive
post-only rejects).

These pure-function helpers are the LOGIC of the phase ladder. The
``_run_soft_flatten_tick`` wiring that consumes them and drives
``self._client.market_close`` / IOC placements lives in ``bot.py``
and is wired in a separate release — see Phase 4D in the defense
plan. This module's helpers are designed so that wiring step is a
mechanical consultation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.enums import Side


def compute_target_price(
    *,
    pos_qty: float,
    best_bid: float,
    best_ask: float,
    tick_size: float,
    in_phase_2: bool,
) -> tuple[Side, float]:
    """Returns ``(close_side, target_price)`` for the soft-flatten
    worker. ``pos_qty > 0`` means we're long and need to SELL;
    ``pos_qty < 0`` means short and need to BUY.

    Caller is expected to have already verified ``best_bid > 0`` and
    ``best_ask > 0``. Tick size 0 disables phase 2 (we stay at phase
    1; the venue's symbol_spec should always provide a positive tick
    in production).
    """
    if pos_qty > 0:
        side = Side.SELL
        phase_1_price = best_ask
        if not in_phase_2 or tick_size <= 0:
            return side, phase_1_price
        # Phase 2 (SELL): one tick above the bid -- max-aggressive
        # post-only without crossing. Cap at phase 1 so a degenerate
        # 1-tick spread doesn't accidentally raise the price.
        phase_2_price = min(phase_1_price, best_bid + tick_size)
        return side, phase_2_price
    else:
        side = Side.BUY
        phase_1_price = best_bid
        if not in_phase_2 or tick_size <= 0:
            return side, phase_1_price
        # Phase 2 (BUY): one tick below the ask.
        phase_2_price = max(phase_1_price, best_ask - tick_size)
        return side, phase_2_price


# ---------------------------------------------------------------------------
# Phase 4D — adaptive-aggressiveness ladder (pure helpers)
# ---------------------------------------------------------------------------


# Phase enum: integers so they're cheap to compare + serialise to
# SQLite + carry on BotState across ticks. The numeric ordering
# matches the aggressiveness ordering — higher phase = more aggressive.
PHASE_0_POST_ONLY_NEAR = 0
PHASE_1_POST_ONLY_FAR_PLUS_TICK = 1
PHASE_2_IOC_CROSS_1 = 2
PHASE_3_IOC_CROSS_2 = 3
PHASE_4_MARKET = 4


@dataclass(frozen=True)
class PhaseLadderDecision:
    """Per-tick decision output. The ``_run_soft_flatten_tick`` wiring
    in bot.py reads:

    * ``new_phase`` — write this back to ``state.sf_phase_ladder_phase``
      (and reset ``state.sf_phase_started_mono`` to ``now_mono`` if
      ``new_phase`` differs from the previous phase).
    * ``order_type`` — one of ``"post_only"``, ``"ioc"``, or
      ``"market"``. Drives which placement API to call.
    * ``target_price`` — the limit price for the order (``None`` for
      ``"market"``, which uses ``client.market_close``).
    * ``escalate_reason`` — diagnostic string for the operator log.
      ``""`` when no escalation this tick.
    """

    new_phase: int
    new_phase_started_mono: float
    order_type: str
    close_side: Side
    target_price: Optional[float]
    escalate_reason: str


def _initial_target(
    *,
    side: Side,
    best_bid: float,
    best_ask: float,
    tick_size: float,
    phase: int,
) -> Optional[float]:
    """Price for a given phase. Returns None for phase 4 (market_close
    has no limit price). Mirror-symmetric for BUY/SELL."""
    if phase == PHASE_0_POST_ONLY_NEAR:
        return best_ask if side == Side.SELL else best_bid
    if phase == PHASE_1_POST_ONLY_FAR_PLUS_TICK:
        if side == Side.SELL:
            # Phase 1 SELL: 1 tick INTO the spread from below the ask
            # (i.e., best_bid + 1 tick, capped at the ask).
            return min(best_ask, best_bid + tick_size)
        # Phase 1 BUY: 1 tick INTO the spread (best_ask - 1 tick,
        # capped at the bid).
        return max(best_bid, best_ask - tick_size)
    if phase == PHASE_2_IOC_CROSS_1:
        # IOC crossing by 1 tick. For SELL we sell ABOVE current bid
        # by 0 (i.e., AT the bid — crosses). For BUY we buy AT the
        # ask. Either way: the inside touch on the OPPOSITE side.
        return best_bid if side == Side.SELL else best_ask
    if phase == PHASE_3_IOC_CROSS_2:
        # IOC crossing by 2 ticks — one tick INSIDE the opposite
        # touch.
        if side == Side.SELL:
            return max(0.0, best_bid - tick_size)
        return best_ask + tick_size
    return None  # PHASE_4_MARKET — no limit price


def _order_type_for_phase(phase: int) -> str:
    if phase in (PHASE_0_POST_ONLY_NEAR, PHASE_1_POST_ONLY_FAR_PLUS_TICK):
        return "post_only"
    if phase in (PHASE_2_IOC_CROSS_1, PHASE_3_IOC_CROSS_2):
        return "ioc"
    return "market"


def _close_side_from_qty(pos_qty: float) -> Side:
    return Side.SELL if pos_qty > 0 else Side.BUY


def evaluate_sf_phase_ladder(
    *,
    pos_qty: float,
    best_bid: float,
    best_ask: float,
    tick_size: float,
    now_mono: float,
    current_phase: int,
    current_phase_started_mono: float,
    consecutive_rejects_in_phase: int,
    entry_mid_for_phase_ladder: Optional[float],
    # Per-phase max-dwell budgets (seconds). Caller passes from
    # Settings. Order matches phase number; index 4 unused.
    phase_durations_s: tuple[float, float, float, float],
    fast_escalate_ticks: float,
    consecutive_rejects_to_escalate: int,
    # v1.5.198 — episode-level hard timeout. ``episode_started_mono``
    # is the monotonic time when the SF episode FIRST entered (not
    # reset on phase changes). ``episode_max_duration_s`` is the hard
    # cap. When (now - episode_started) >= cap, force phase=4 so the
    # episode terminates via market_close. Default 0 disables.
    episode_started_mono: Optional[float] = None,
    episode_max_duration_s: float = 0.0,
) -> PhaseLadderDecision:
    """Per-tick phase decision. Combines time-based + signal-based
    escalation logic. Pure function — no side effects.

    Escalation rules (highest-priority first):

    0. **Episode hard-timeout** (v1.5.198): if ``episode_started_mono``
       is provided AND ``(now - episode_started_mono) >=
       episode_max_duration_s``, force phase to ``PHASE_4_MARKET``
       regardless of current phase or per-phase budget. Addresses the
       v1.5.195-260527-165258 30-min pause where SF hung in phase 2
       waiting for a post-only fill that never came; the per-phase
       timer kept resetting on re-entries so phase 4 was never reached.
       The episode-level bound is reset-resistant because it's
       anchored to the initial SF entry, not to the current phase.
    1. **Fast escalate via ask drift**: when the current adverse-side
       touch has moved away from the SF entry mid by
       ``fast_escalate_ticks`` ticks, jump to phase 2 immediately.
       (Heuristic: "market is running, post-only won't fill.")
    2. **Fast escalate via rejection rate**: when
       ``consecutive_rejects_in_phase >= consecutive_rejects_to_escalate``
       AND we're in a post-only phase (0 or 1), advance one phase.
       (Heuristic: "venue is rejecting everything; tighter prices
       won't help — change aggressiveness type.")
    3. **Time-based**: if ``now - phase_started_mono >=
       phase_durations_s[current_phase]``, advance one phase. Phase 4
       is terminal — no further advance.

    Returns the NEW phase + the target price + the order type. Caller
    persists the new phase to BotState and uses the returned price /
    order_type to place the order. ``new_phase_started_mono`` is
    ``now_mono`` if a transition occurred, else the input
    ``current_phase_started_mono`` (caller doesn't reset on no-op).
    """
    side = _close_side_from_qty(pos_qty)
    phase = int(current_phase)
    started = float(current_phase_started_mono)
    reason = ""

    # Already at terminal phase — stay.
    if phase >= PHASE_4_MARKET:
        return PhaseLadderDecision(
            new_phase=PHASE_4_MARKET,
            new_phase_started_mono=started,
            order_type=_order_type_for_phase(PHASE_4_MARKET),
            close_side=side,
            target_price=None,
            escalate_reason="",
        )

    # Rule 0 (v1.5.198): episode-level hard timeout. Force phase 4
    # if the episode has been running longer than the absolute cap.
    # Higher priority than per-phase rules so a single SF episode
    # CANNOT exceed ``episode_max_duration_s`` in total wall time.
    if (
        episode_started_mono is not None
        and episode_max_duration_s > 0.0
    ):
        episode_elapsed = float(now_mono) - float(episode_started_mono)
        if episode_elapsed >= float(episode_max_duration_s):
            phase = PHASE_4_MARKET
            started = float(now_mono)
            reason = (
                f"episode_hard_timeout:{episode_elapsed:.1f}s"
                f">={episode_max_duration_s:.1f}s"
            )
            return PhaseLadderDecision(
                new_phase=phase,
                new_phase_started_mono=started,
                order_type=_order_type_for_phase(phase),
                close_side=side,
                target_price=None,  # market_close has no limit price
                escalate_reason=reason,
            )

    # Rule 1: fast-escalate via ask drift past entry mid.
    if (
        entry_mid_for_phase_ladder is not None
        and tick_size > 0
        and fast_escalate_ticks > 0
        and phase < PHASE_2_IOC_CROSS_1
    ):
        adverse_touch = best_ask if side == Side.BUY else best_bid
        # For a BUY (closing short) adverse direction is "ask above
        # entry" — bigger ask = bot's close gets worse.
        # For a SELL (closing long) adverse direction is "bid below
        # entry".
        if side == Side.BUY:
            drift_ticks = (
                adverse_touch - float(entry_mid_for_phase_ladder)
            ) / float(tick_size)
        else:
            drift_ticks = (
                float(entry_mid_for_phase_ladder) - adverse_touch
            ) / float(tick_size)
        if drift_ticks >= float(fast_escalate_ticks):
            phase = PHASE_2_IOC_CROSS_1
            started = float(now_mono)
            reason = (
                f"fast_escalate_drift:{drift_ticks:.1f}ticks"
                f">={fast_escalate_ticks:.1f}ticks"
            )

    # Rule 2: fast-escalate via consecutive rejects (only in
    # post-only phases — IOC rejects are different signal).
    if (
        reason == ""
        and phase in (PHASE_0_POST_ONLY_NEAR, PHASE_1_POST_ONLY_FAR_PLUS_TICK)
        and consecutive_rejects_in_phase
        >= int(consecutive_rejects_to_escalate)
    ):
        phase = phase + 1
        started = float(now_mono)
        reason = (
            f"fast_escalate_rejects:"
            f"{consecutive_rejects_in_phase}"
            f">={consecutive_rejects_to_escalate}"
        )

    # Rule 3: time-based.
    if reason == "":
        try:
            dwell_budget = float(phase_durations_s[phase])
        except (IndexError, ValueError, TypeError):
            dwell_budget = 0.0
        elapsed = float(now_mono) - float(current_phase_started_mono)
        if dwell_budget > 0 and elapsed >= dwell_budget:
            phase = phase + 1
            started = float(now_mono)
            reason = f"phase_dwell_expired:{elapsed:.2f}s>={dwell_budget:.2f}s"

    # Clamp to terminal.
    if phase > PHASE_4_MARKET:
        phase = PHASE_4_MARKET

    target_price = _initial_target(
        side=side,
        best_bid=best_bid,
        best_ask=best_ask,
        tick_size=tick_size,
        phase=phase,
    )
    order_type = _order_type_for_phase(phase)
    return PhaseLadderDecision(
        new_phase=phase,
        new_phase_started_mono=started,
        order_type=order_type,
        close_side=side,
        target_price=target_price,
        escalate_reason=reason,
    )
