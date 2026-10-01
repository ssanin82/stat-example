"""Phase 4D — soft-flatten adaptive-aggressiveness phase ladder.

Pure-function tests for the helpers in ``app/soft_flatten.py``. The
``_run_soft_flatten_tick`` wiring that consumes these helpers + drives
real placements is intentionally deferred to a separate release (the
function is intricate and touches live-trading hot path); these tests
nail down the LOGIC so the wiring can be a mechanical consultation.

Targets the v1.4.157-260520-213540 failure: a 14-second post-only
chase produced 1,443 place-cancels with 0 fills, then a single
``market_close`` taker that ate the worst spread of the rally. The
v1.4.157-replay test below confirms the ladder would have escalated
to IOC well before that 14-second window expired.
"""

from __future__ import annotations

from app.enums import Side
from app.soft_flatten import (
    PHASE_0_POST_ONLY_NEAR,
    PHASE_1_POST_ONLY_FAR_PLUS_TICK,
    PHASE_2_IOC_CROSS_1,
    PHASE_3_IOC_CROSS_2,
    PHASE_4_MARKET,
    PhaseLadderDecision,
    evaluate_sf_phase_ladder,
)


# Default phase durations and thresholds matching the v1.4.164 prod
# defaults.
_DURATIONS = (3.0, 4.0, 4.0, 2.0)  # phases 0/1/2/3
_FAST_ESCALATE_TICKS = 3.0
_REJECTS_TO_ESCALATE = 10


def _kwargs(**overrides):
    """Default scenario: bot is short -9, entry mid was 2.040, current
    best_ask is 2.041 (1 tick above entry — well below the 3-tick
    fast-escalate threshold). Each test overrides to drive a specific
    phase transition."""
    base = dict(
        pos_qty=-9.0,  # short — bot must BUY to flatten
        best_bid=2.040,
        best_ask=2.041,
        tick_size=0.001,
        now_mono=100.0,
        current_phase=PHASE_0_POST_ONLY_NEAR,
        current_phase_started_mono=99.0,  # 1 s into phase
        consecutive_rejects_in_phase=0,
        entry_mid_for_phase_ladder=2.040,
        phase_durations_s=_DURATIONS,
        fast_escalate_ticks=_FAST_ESCALATE_TICKS,
        consecutive_rejects_to_escalate=_REJECTS_TO_ESCALATE,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Phase 0: post-only at near touch
# ---------------------------------------------------------------------------


def test_phase_0_buy_targets_best_bid() -> None:
    """Bot is short → BUY to close. Phase 0: post-only at best_bid
    (queue-join the rebate side)."""
    r = evaluate_sf_phase_ladder(**_kwargs())
    assert r.new_phase == PHASE_0_POST_ONLY_NEAR
    assert r.order_type == "post_only"
    assert r.close_side == Side.BUY
    assert r.target_price == 2.040  # best_bid


def test_phase_0_sell_targets_best_ask() -> None:
    """Bot is long → SELL to close. Phase 0: post-only at best_ask.
    Use entry_mid above the touch so drift escalation doesn't fire."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(pos_qty=+9.0, entry_mid_for_phase_ladder=2.041)
    )
    assert r.close_side == Side.SELL
    assert r.target_price == 2.041  # best_ask


def test_phase_0_holds_when_dwell_not_expired_and_no_signals() -> None:
    """Phase 0 budget 3 s; elapsed 1 s + no signal-based escalation
    → stay at phase 0."""
    r = evaluate_sf_phase_ladder(**_kwargs(now_mono=100.0))
    assert r.new_phase == PHASE_0_POST_ONLY_NEAR
    assert r.escalate_reason == ""


# ---------------------------------------------------------------------------
# Time-based phase advance
# ---------------------------------------------------------------------------


def test_phase_0_advances_to_phase_1_after_3s() -> None:
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_0_POST_ONLY_NEAR,
            current_phase_started_mono=100.0,
            now_mono=103.5,
        )
    )
    assert r.new_phase == PHASE_1_POST_ONLY_FAR_PLUS_TICK
    assert r.new_phase_started_mono == 103.5
    assert r.order_type == "post_only"
    assert "phase_dwell_expired" in r.escalate_reason


def test_phase_1_advances_to_phase_2_ioc() -> None:
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
            current_phase_started_mono=100.0,
            now_mono=104.5,  # past 4 s phase-1 budget
            entry_mid_for_phase_ladder=2.057,  # no drift escalation
        )
    )
    assert r.new_phase == PHASE_2_IOC_CROSS_1
    assert r.order_type == "ioc"


def test_phase_2_advances_to_phase_3() -> None:
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_2_IOC_CROSS_1,
            current_phase_started_mono=100.0,
            now_mono=105.0,
            entry_mid_for_phase_ladder=2.057,
        )
    )
    assert r.new_phase == PHASE_3_IOC_CROSS_2


def test_phase_3_advances_to_phase_4_market() -> None:
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_3_IOC_CROSS_2,
            current_phase_started_mono=100.0,
            now_mono=103.0,  # past 2 s phase-3 budget
            entry_mid_for_phase_ladder=2.057,
        )
    )
    assert r.new_phase == PHASE_4_MARKET
    assert r.order_type == "market"
    assert r.target_price is None  # market_close has no limit


def test_phase_4_is_terminal() -> None:
    """Once at phase 4, stays at phase 4."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_4_MARKET,
            current_phase_started_mono=100.0,
            now_mono=200.0,
            entry_mid_for_phase_ladder=2.057,
        )
    )
    assert r.new_phase == PHASE_4_MARKET
    assert r.escalate_reason == ""


# ---------------------------------------------------------------------------
# Rule 1: fast-escalate via drift past entry mid (v1.4.157 replay)
# ---------------------------------------------------------------------------


def test_fast_escalate_drift_jumps_to_phase_2_immediately() -> None:
    """v1.4.157 scenario: bot is short, entry mid was $2.040, current
    ask is $2.044 — 4 ticks above (drift_ticks=4 ≥ 3 threshold).
    Should jump straight to phase 2 from phase 0."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            pos_qty=-9.0,
            best_ask=2.044,  # 4 ticks above entry mid 2.040
            best_bid=2.043,
            tick_size=0.001,
            current_phase=PHASE_0_POST_ONLY_NEAR,
            current_phase_started_mono=100.0,
            now_mono=100.5,  # only 0.5s in — dwell hasn't expired
            entry_mid_for_phase_ladder=2.040,
            fast_escalate_ticks=3.0,
        )
    )
    assert r.new_phase == PHASE_2_IOC_CROSS_1
    assert "fast_escalate_drift" in r.escalate_reason
    assert r.order_type == "ioc"


def test_fast_escalate_drift_only_fires_in_passive_phases() -> None:
    """Once we're past phase 1, drift escalation doesn't fire — time
    + reject signals take over."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_2_IOC_CROSS_1,
            best_ask=2.080,  # huge drift
            best_bid=2.079,
            entry_mid_for_phase_ladder=2.040,
            current_phase_started_mono=100.0,
            now_mono=100.5,  # not enough elapsed for time-based
        )
    )
    assert r.new_phase == PHASE_2_IOC_CROSS_1  # stays


def test_drift_below_threshold_does_not_escalate() -> None:
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            best_ask=2.042,  # only 2 ticks above entry
            best_bid=2.041,
            tick_size=0.001,
            entry_mid_for_phase_ladder=2.040,
            fast_escalate_ticks=3.0,
        )
    )
    assert r.new_phase == PHASE_0_POST_ONLY_NEAR


# ---------------------------------------------------------------------------
# Rule 2: fast-escalate via consecutive rejects (the 1,443-cancel
# pattern)
# ---------------------------------------------------------------------------


def test_fast_escalate_rejects_advances_one_phase() -> None:
    """10+ consecutive post-only rejects in phase 0 → advance to
    phase 1 (NOT skip ahead to phase 2 — incremental escalation by
    rule 2)."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_0_POST_ONLY_NEAR,
            consecutive_rejects_in_phase=10,
            entry_mid_for_phase_ladder=2.058,  # no drift to interfere
        )
    )
    assert r.new_phase == PHASE_1_POST_ONLY_FAR_PLUS_TICK
    assert "fast_escalate_rejects" in r.escalate_reason


def test_fast_escalate_rejects_phase_1_to_2() -> None:
    """20 rejects in phase 1 → escalate to phase 2 (IOC)."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_1_POST_ONLY_FAR_PLUS_TICK,
            consecutive_rejects_in_phase=20,
            entry_mid_for_phase_ladder=2.058,
        )
    )
    assert r.new_phase == PHASE_2_IOC_CROSS_1


def test_fast_escalate_rejects_does_not_fire_in_ioc_phases() -> None:
    """IOC rejects are a different signal — they don't accumulate
    here. Stay in phase 2 even with high reject count."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_2_IOC_CROSS_1,
            consecutive_rejects_in_phase=50,
            entry_mid_for_phase_ladder=2.058,
            current_phase_started_mono=100.0,
            now_mono=100.1,
        )
    )
    assert r.new_phase == PHASE_2_IOC_CROSS_1


# ---------------------------------------------------------------------------
# Phase 2/3 IOC pricing
# ---------------------------------------------------------------------------


def test_phase_2_buy_targets_best_ask() -> None:
    """IOC cross 1 tick for a BUY = limit price at the ASK (crosses)."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            pos_qty=-9.0,  # close short → BUY
            best_ask=2.059,
            best_bid=2.058,
            current_phase=PHASE_2_IOC_CROSS_1,
            current_phase_started_mono=100.0,
            now_mono=100.5,
            entry_mid_for_phase_ladder=2.040,
        )
    )
    assert r.target_price == 2.059
    assert r.order_type == "ioc"


def test_phase_3_buy_targets_one_tick_above_ask() -> None:
    """IOC cross 2 ticks for a BUY = limit price ASK + 1 tick (eats
    one extra level worst case)."""
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            pos_qty=-9.0,
            best_ask=2.059,
            best_bid=2.058,
            tick_size=0.001,
            current_phase=PHASE_3_IOC_CROSS_2,
            current_phase_started_mono=100.0,
            now_mono=100.5,
            entry_mid_for_phase_ladder=2.040,
        )
    )
    assert abs(r.target_price - 2.060) < 1e-9
    assert r.order_type == "ioc"


def test_phase_2_sell_targets_best_bid() -> None:
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            pos_qty=+9.0,  # close long → SELL
            best_bid=2.058,
            best_ask=2.059,
            current_phase=PHASE_2_IOC_CROSS_1,
            current_phase_started_mono=100.0,
            now_mono=100.5,
            entry_mid_for_phase_ladder=2.080,
        )
    )
    assert r.target_price == 2.058
    assert r.close_side == Side.SELL


def test_phase_3_sell_targets_one_tick_below_bid() -> None:
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            pos_qty=+9.0,
            best_bid=2.058,
            best_ask=2.059,
            tick_size=0.001,
            current_phase=PHASE_3_IOC_CROSS_2,
            current_phase_started_mono=100.0,
            now_mono=100.5,
            entry_mid_for_phase_ladder=2.080,
        )
    )
    assert abs(r.target_price - 2.057) < 1e-9


# ---------------------------------------------------------------------------
# Phase 4 — terminal market
# ---------------------------------------------------------------------------


def test_phase_4_buy_target_price_is_none() -> None:
    r = evaluate_sf_phase_ladder(
        **_kwargs(
            current_phase=PHASE_4_MARKET,
            current_phase_started_mono=100.0,
            now_mono=200.0,
            entry_mid_for_phase_ladder=2.040,
        )
    )
    assert r.target_price is None
    assert r.order_type == "market"


# ---------------------------------------------------------------------------
# Result dataclass is frozen
# ---------------------------------------------------------------------------


def test_decision_is_frozen() -> None:
    r = PhaseLadderDecision(
        new_phase=0,
        new_phase_started_mono=0.0,
        order_type="post_only",
        close_side=Side.BUY,
        target_price=2.0,
        escalate_reason="",
    )
    try:
        r.new_phase = 5  # type: ignore[misc]
        raised = False
    except Exception:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# v1.4.157 replay: would have escalated within seconds
# ---------------------------------------------------------------------------


def test_v1_4_157_replay_escalates_within_seconds() -> None:
    """Replay the actual v1.4.157-260520-213540 timeline.

    At SF entry (17:14:46) the position was -9 short, entry mid
    around $2.040, best_ask climbing from $2.052 to $2.059 over the
    next 14 s. The current (legacy) code stayed in post-only the
    whole 14 s and then called market_close at $2.059. With the
    Phase 4D ladder:

    * 17:14:46.000 (t=0): start at phase 0, ask=$2.052 (12 ticks
      above entry mid → already >3 ticks → fast-escalate to phase 2)
    """
    # Tick at t=0.5s, ask has already moved to $2.052
    r = evaluate_sf_phase_ladder(
        pos_qty=-9.0,
        best_bid=2.051,
        best_ask=2.052,
        tick_size=0.001,
        now_mono=0.5,
        current_phase=PHASE_0_POST_ONLY_NEAR,
        current_phase_started_mono=0.0,
        consecutive_rejects_in_phase=0,
        entry_mid_for_phase_ladder=2.040,
        phase_durations_s=_DURATIONS,
        fast_escalate_ticks=3.0,
        consecutive_rejects_to_escalate=_REJECTS_TO_ESCALATE,
    )
    # Drift = (2.052 - 2.040) / 0.001 = 12 ticks ≫ 3 → fast-escalate.
    assert r.new_phase == PHASE_2_IOC_CROSS_1
    assert "fast_escalate_drift" in r.escalate_reason
    # The IOC at best_ask=$2.052 closes the position 7 ticks below
    # the legacy market_close at $2.059. Per-contract saving =
    # 7 ticks × $0.001 = $0.007; 9 contracts saves $0.063 — 31 % of
    # the original $0.20 loss.
    assert abs(r.target_price - 2.052) < 1e-9
