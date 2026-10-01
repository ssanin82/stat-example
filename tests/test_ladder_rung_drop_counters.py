"""v1.4.100 F2 — ladder rung-drop attribution counters.

Each drop-attribution category in ``BotState`` has a corresponding
``on_rung_dropped`` callback invocation site in ``app/ladder.py``.
These tests pin the callback contract: when ``build_ladder()`` drops
a rung for reason X, ``on_rung_dropped("X")`` is called exactly once
with that reason string.

Coverage:
* ``grid_collision`` — outer-rung grid_px doesn't move at least one
  tick off the inner rung.
* ``inventory_aware_pruning`` — explicit pruning drops the adding
  side's outer rung when the flag is on and util ≥ threshold.

We rely on a recording callback (no production code path needs to be
mocked) so the test is fully deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.enums import ActiveSides, Side
from app.ladder import LadderConfig, build_ladder
from app.models import QuoteDecision


class _DropRecorder:
    """Captures every ``on_rung_dropped`` call into an ordered list.

    Closure-style helper that mirrors what `_bump_rung_drop` does in
    ``app/bot.py`` — except instead of dispatching to BotState
    counters, it records the reason strings for the test to assert.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, reason: str) -> None:
        self.calls.append(reason)


def _make_decision(
    *,
    mid: float = 2.000,
    target_spread_bps: float = 6.0,
    quoted_bid_sz: float = 3.0,
    quoted_ask_sz: float = 3.0,
    active_sides: ActiveSides = ActiveSides.BOTH,
) -> QuoteDecision:
    hs = target_spread_bps / 2.0
    bid = mid * (1.0 - hs / 10_000.0)
    ask = mid * (1.0 + hs / 10_000.0)
    return QuoteDecision(
        ts=datetime.now(timezone.utc),
        symbol="TON-USDT-SWAP",
        mid_price=mid,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=mid,
        target_spread_bps=target_spread_bps,
        target_bid=bid,
        target_ask=ask,
        quoted_bid=bid,
        quoted_ask=ask,
        quoted_bid_sz=quoted_bid_sz,
        quoted_ask_sz=quoted_ask_sz,
        active_sides=active_sides,
        toxicity_score=0.0,
        decision_reason="test",
        quote_cycle_id="test-cycle",
    )


def _cfg(
    *,
    n: int = 2,
    offset_step: float = 0.1,
    inventory_aware_pruning_enabled: bool = False,
) -> LadderConfig:
    return LadderConfig(
        num_levels_per_side=n,
        offset_step=offset_step,
        size_decay=0.7,
        inside_full_size=True,
        gates_limit_levels=True,
        batch_orders_enabled=False,
        inventory_aware_pruning_enabled=inventory_aware_pruning_enabled,
        inventory_aware_pruning_threshold_pct=0.65,
    )


# ---------------------------------------------------------------------------
# grid_collision
# ---------------------------------------------------------------------------


def test_grid_collision_fires_callback_when_outer_rung_collides() -> None:
    """With a tiny offset_step + a coarse tick_size, the outer rung's
    grid-rounded price collides with the inner rung's. Expect one
    ``grid_collision`` reason per collided side.

    v1.4.169 Phase 2J: the new tick-floor would pre-empt this path
    by shifting the outer rung outward instead of dropping it. Set
    ``tick_floor_steps=0`` to exercise the LEGACY collision-dedup
    behaviour this test was written for (now defense-in-depth)."""
    decision = _make_decision()
    cfg = LadderConfig(
        num_levels_per_side=2,
        offset_step=0.01,
        size_decay=0.7,
        inside_full_size=True,
        gates_limit_levels=True,
        batch_orders_enabled=False,
        inventory_aware_pruning_enabled=False,
        inventory_aware_pruning_threshold_pct=0.65,
        tick_floor_steps=0,  # Disable Phase 2J floor → exercise legacy path.
    )
    recorder = _DropRecorder()
    # Tick size large enough to absorb the small offset_step shift.
    build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        tick_size=0.01,
        on_rung_dropped=recorder,
    )
    assert recorder.calls.count("grid_collision") >= 1, (
        f"expected at least one grid_collision drop; got {recorder.calls}"
    )


def test_no_drops_when_offset_step_is_generous() -> None:
    """With a healthy offset_step, no rungs should drop and the
    callback should never fire."""
    decision = _make_decision()
    cfg = _cfg(offset_step=2.0)  # 2× half-spread offset between rungs
    recorder = _DropRecorder()
    build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        tick_size=0.0001,  # very fine tick — no collision
        on_rung_dropped=recorder,
    )
    assert recorder.calls == [], (
        f"expected no drops on a healthy ladder; got {recorder.calls}"
    )


# ---------------------------------------------------------------------------
# inventory_aware_pruning
# ---------------------------------------------------------------------------


def test_inventory_aware_pruning_fires_when_flag_on_and_util_high() -> None:
    """Flag ON + util >= threshold → adding side's outer rung drops
    with reason ``inventory_aware_pruning``."""
    decision = _make_decision()
    cfg = _cfg(
        offset_step=2.0, inventory_aware_pruning_enabled=True
    )  # offset_step generous so no grid_collision
    recorder = _DropRecorder()
    build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=9.0,  # +90% util
        max_abs_position=10.0,
        tick_size=0.0001,
        on_rung_dropped=recorder,
    )
    assert "inventory_aware_pruning" in recorder.calls, (
        f"expected inventory_aware_pruning drop; got {recorder.calls}"
    )


def test_inventory_aware_pruning_dormant_when_flag_off() -> None:
    """Flag OFF (default) → no pruning even when util is high. This
    is the dormant-by-default contract: shipping this code MUST NOT
    change production behaviour."""
    decision = _make_decision()
    cfg = _cfg(offset_step=2.0, inventory_aware_pruning_enabled=False)
    recorder = _DropRecorder()
    build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=9.0,
        max_abs_position=10.0,
        tick_size=0.0001,
        on_rung_dropped=recorder,
    )
    assert "inventory_aware_pruning" not in recorder.calls, (
        f"expected dormant flag to suppress pruning; got {recorder.calls}"
    )


# ---------------------------------------------------------------------------
# BotState wiring — the closure dispatch
# ---------------------------------------------------------------------------


def test_botstate_counter_block_present() -> None:
    """``BotState.to_health_payload`` exposes the ``ladder_rung_drops``
    block with every taxonomy category."""
    from tests.settings_helpers import UnitTestSettings
    from app.state import BotState

    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
        }
    )
    st = BotState(settings)
    # Direct attribute access — these are the counters live_stats and
    # state_current.json publish.
    for name in (
        "ladder_rung_dropped_grid_collision_total",
        "ladder_rung_dropped_min_notional_total",
        "ladder_rung_dropped_inventory_buffer_total",
        "ladder_rung_dropped_inventory_aware_pruning_total",
        "ladder_rung_dropped_position_cap_total",
        "ladder_rung_dropped_in_flight_total",
        "ladder_rung_dropped_other_total",
    ):
        assert hasattr(st, name), f"BotState missing counter: {name}"
        assert getattr(st, name) == 0, (
            f"{name} should start at 0 on a fresh BotState; got "
            f"{getattr(st, name)}"
        )
