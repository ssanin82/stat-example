"""v1.5.198 — SF episode hard-time-bound tests.

Pre-v1.5.198 the SF episode could hang in any phase indefinitely if:
  * Post-only orders sat on the book without filling (market drifted
    away from the price).
  * Per-phase budgets were reset by SF re-entries from a tox-hard loop.

Snapshot `v1.5.195-260527-165258-prod.okx.ton.usdt.perp` showed a
30-minute total bot pause where SF entered phase 2 (post-only) at
12:18 and didn't exit until 12:48 when a leftover post-only BUY
filled by chance.

v1.5.198 adds two defenses:

1. **Episode hard-timeout**: ``evaluate_sf_phase_ladder`` takes
   ``episode_started_mono`` + ``episode_max_duration_s``. When the
   episode has been active longer than the cap, force-jump to
   PHASE_4_MARKET regardless of current phase or per-phase budget.

2. **Re-entry cooldown**: ``_enter_soft_flatten`` checks
   ``state.soft_flatten_last_exited_at_mono``. If less than
   ``soft_flatten_reentry_cooldown_seconds`` has elapsed, the entry
   is deferred (logs WARNING + returns). Eliminates the tox-hard
   loop pattern.

The two together guarantee:
  * NO single SF episode lasts more than ``EPISODE_MAX_DURATION``
    (default 60s).
  * NO toxicity-hard / drawdown loop can produce more than
    1 SF episode per ``REENTRY_COOLDOWN`` (default 30s) window.

Per CLAUDE.md: only this test file is run from the assistant.
"""

from __future__ import annotations

from app.enums import Side
from app.soft_flatten import (
    PHASE_0_POST_ONLY_NEAR,
    PHASE_1_POST_ONLY_FAR_PLUS_TICK,
    PHASE_2_IOC_CROSS_1,
    PHASE_3_IOC_CROSS_2,
    PHASE_4_MARKET,
    evaluate_sf_phase_ladder,
)


# ---------------------------------------------------------------------------
# Phase 1 — Episode hard-timeout via evaluate_sf_phase_ladder
# ---------------------------------------------------------------------------


def _eval(
    *,
    phase: int = PHASE_1_POST_ONLY_FAR_PLUS_TICK,
    now_mono: float = 100.0,
    phase_started: float = 100.0,
    episode_started: float | None = None,
    episode_max_duration: float = 60.0,
    pos_qty: float = 2.0,
):
    """Shortcut wrapper around evaluate_sf_phase_ladder with
    typical fixture inputs."""
    return evaluate_sf_phase_ladder(
        pos_qty=pos_qty,
        best_bid=1.890,
        best_ask=1.891,
        tick_size=0.001,
        now_mono=now_mono,
        current_phase=phase,
        current_phase_started_mono=phase_started,
        consecutive_rejects_in_phase=0,
        entry_mid_for_phase_ladder=1.890,
        phase_durations_s=(3.0, 4.0, 4.0, 2.0),
        fast_escalate_ticks=3.0,
        consecutive_rejects_to_escalate=10,
        episode_started_mono=episode_started,
        episode_max_duration_s=episode_max_duration,
    )


def test_episode_timeout_forces_phase_4_from_phase_1() -> None:
    """SF episode started 65s ago, currently in phase 1: episode-
    timeout rule (60s cap) fires → phase 4 (market_close)."""
    decision = _eval(
        phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
        now_mono=165.0,
        phase_started=160.0,    # only 5s in current phase
        episode_started=100.0,  # but 65s in episode → over the cap
        episode_max_duration=60.0,
    )
    assert decision.new_phase == PHASE_4_MARKET
    assert decision.order_type == "market"
    assert decision.target_price is None
    assert "episode_hard_timeout" in decision.escalate_reason


def test_episode_timeout_forces_phase_4_from_phase_0() -> None:
    """Even if SF only just started phase 0, the episode-level cap
    forces phase 4. The episode anchor doesn't reset on phase changes."""
    decision = _eval(
        phase=PHASE_0_POST_ONLY_NEAR,
        now_mono=200.0,
        phase_started=199.5,    # in phase 0 for 0.5s
        episode_started=100.0,  # but 100s in episode → way over cap
        episode_max_duration=60.0,
    )
    assert decision.new_phase == PHASE_4_MARKET
    assert decision.target_price is None
    assert "episode_hard_timeout" in decision.escalate_reason


def test_episode_timeout_inactive_under_cap() -> None:
    """SF episode 40s old, cap=60s — episode-timeout rule does NOT
    fire. Falls through to normal per-phase logic."""
    decision = _eval(
        phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
        now_mono=140.0,
        phase_started=139.0,   # 1s in phase 1 (under 4s budget)
        episode_started=100.0,  # 40s in episode (under 60s cap)
        episode_max_duration=60.0,
    )
    # Episode timeout didn't fire; per-phase didn't either → stay at phase 1.
    assert decision.new_phase == PHASE_1_POST_ONLY_FAR_PLUS_TICK
    assert "episode_hard_timeout" not in decision.escalate_reason


def test_episode_timeout_none_anchor_disabled() -> None:
    """When ``episode_started_mono`` is None, the episode-timeout rule
    is disabled — backward-compat for callers that don't pass it."""
    decision = _eval(
        phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
        now_mono=10_000.0,    # very large now
        phase_started=9_999.0,
        episode_started=None,  # disabled
        episode_max_duration=60.0,
    )
    assert "episode_hard_timeout" not in decision.escalate_reason


def test_episode_timeout_zero_max_disabled() -> None:
    """When ``episode_max_duration_s == 0``, the episode-timeout rule
    is disabled (operator opt-out)."""
    decision = _eval(
        phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
        now_mono=10_000.0,
        phase_started=9_999.0,
        episode_started=100.0,
        episode_max_duration=0.0,
    )
    assert "episode_hard_timeout" not in decision.escalate_reason


def test_episode_timeout_priority_over_per_phase_budget() -> None:
    """Episode timeout is rule 0 — higher priority than per-phase
    rule 3. Even if the per-phase budget would advance only one step,
    episode timeout jumps straight to phase 4."""
    decision = _eval(
        phase=PHASE_0_POST_ONLY_NEAR,
        now_mono=200.0,
        phase_started=190.0,    # 10s in phase 0 (way over 3s budget)
        episode_started=100.0,  # 100s in episode (over 60s cap)
        episode_max_duration=60.0,
    )
    # Without episode timeout, rule 3 would advance phase 0 → 1.
    # With episode timeout, it jumps phase 0 → 4 directly.
    assert decision.new_phase == PHASE_4_MARKET


def test_episode_timeout_terminal_phase_4_stays() -> None:
    """If already at phase 4, the timeout doesn't matter — stay at
    phase 4 (terminal). The early-return for terminal is still in
    force."""
    decision = _eval(
        phase=PHASE_4_MARKET,
        now_mono=1000.0,
        phase_started=999.0,
        episode_started=100.0,
        episode_max_duration=60.0,
    )
    assert decision.new_phase == PHASE_4_MARKET


def test_episode_timeout_buy_close_side() -> None:
    """Verify close_side is correct for a short position (BUY to close)."""
    decision = _eval(
        phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
        now_mono=200.0,
        phase_started=199.0,
        episode_started=100.0,
        episode_max_duration=60.0,
        pos_qty=-2.0,  # short → BUY to close
    )
    assert decision.new_phase == PHASE_4_MARKET
    assert decision.close_side == Side.BUY


def test_episode_timeout_exactly_at_threshold_fires() -> None:
    """Threshold is inclusive: elapsed == max → fires."""
    decision = _eval(
        phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
        now_mono=160.0,
        phase_started=159.0,
        episode_started=100.0,   # exactly 60s
        episode_max_duration=60.0,
    )
    assert decision.new_phase == PHASE_4_MARKET


def test_episode_timeout_one_second_before_threshold_no_fire() -> None:
    """Threshold strict: elapsed = max - 1s → does not fire."""
    decision = _eval(
        phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
        now_mono=159.0,
        phase_started=158.0,
        episode_started=100.0,   # 59s in episode
        episode_max_duration=60.0,
    )
    assert decision.new_phase == PHASE_1_POST_ONLY_FAR_PLUS_TICK
    assert "episode_hard_timeout" not in decision.escalate_reason


# ---------------------------------------------------------------------------
# Phase 2 — Re-entry cooldown contract (state-level field test)
# ---------------------------------------------------------------------------
#
# The re-entry cooldown is implemented inline in Bot._enter_soft_flatten
# (not a pure helper). The BotState field that backs it is what we
# pin here — the higher-level behavioural contract is exercised
# indirectly via the full integration tests.


def test_state_field_soft_flatten_last_exited_at_mono_exists() -> None:
    """``BotState.soft_flatten_last_exited_at_mono`` is the field
    Bot._enter_soft_flatten reads to gate re-entries. Pre-v1.5.198
    this field didn't exist; the cooldown couldn't be enforced.

    Pins:
      * The field is None on a fresh state (no prior SF exit).
      * The type is Optional[float] — monotonic time, not wall.
    """
    import os
    os.environ.setdefault("HL_SECRET_KEY", "x")
    os.environ.setdefault("HL_ACCOUNT_ADDRESS", "0xabc")
    from app.config import Settings
    from app.state import BotState
    s = Settings()
    bs = BotState(s)
    assert hasattr(bs, "soft_flatten_last_exited_at_mono")
    assert bs.soft_flatten_last_exited_at_mono is None


def test_config_knobs_present_with_safe_defaults() -> None:
    """v1.5.198 introduces two new config knobs. Verify both load
    with safe defaults from a fresh Settings()."""
    import os
    os.environ.setdefault("HL_SECRET_KEY", "x")
    os.environ.setdefault("HL_ACCOUNT_ADDRESS", "0xabc")
    from app.config import Settings
    s = Settings()
    # 60s episode timeout — generous but bounded.
    assert s.soft_flatten_episode_max_duration_seconds == 60.0
    # 30s re-entry cooldown — prevents tox-hard loops.
    assert s.soft_flatten_reentry_cooldown_seconds == 30.0
