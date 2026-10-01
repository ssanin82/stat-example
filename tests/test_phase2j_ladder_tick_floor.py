"""Phase 2J (v1.4.169) — tick-floored ladder offset.

STRUCTURAL fix in ``app/ladder.py`` for the case where the bps-math
rung-price computation underflows the venue's price tick. Without
the floor, two adjacent rungs whose ``half_spread × (1 + offset_step
× i)`` produces a sub-tick difference would both snap to the same
grid cell and the outer rung would be silently DROPPED by the legacy
``grid_collision`` dedup.

The tick-floor enforces a minimum gap (in ticks) between rung ``i``
and the inside rung's grid position, applied BEFORE the dedup check.
After the floor, ``grid_collision`` becomes defense-in-depth only.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.enums import ActiveSides, Side
from app.ladder import LadderConfig, build_ladder
from app.models import QuoteDecision


def _decision(
    *,
    mid: float = 2.040,
    reservation: float = 2.040,
    half_spread_bps: float = 4.0,
    bid_sz: float = 3.0,
    ask_sz: float = 3.0,
) -> tuple[QuoteDecision, float]:
    """TON-like default: mid $2.04, tick $0.001, half_spread 4 bps.
    At these values, 1 bp ≈ 0.00020 price units — i.e. ~5× SMALLER
    than 1 tick (0.001). Outer rungs at i=1 sit at half_spread × 2 =
    8 bps from mid (= 0.00163 price units = 1.6 ticks), so without
    the floor the grid would still allow it; but at offset_step=0.1
    the i=1 rung is at half_spread × 1.1 = 4.4 bps = 0.9 ticks → BELOW
    one full tick → grid collision. Triggers Phase 2J's path."""
    bid_px = reservation * (1.0 - half_spread_bps / 10_000.0)
    ask_px = reservation * (1.0 + half_spread_bps / 10_000.0)
    d = QuoteDecision(
        ts=datetime(2026, 5, 20, 12, 0, 0, tzinfo=timezone.utc),
        symbol="TON-USDT-SWAP",
        mid_price=mid,
        vol_estimate=0.0,
        inventory=0.0,
        reservation_price=reservation,
        target_spread_bps=half_spread_bps * 2.0,
        target_bid=bid_px,
        target_ask=ask_px,
        quoted_bid=bid_px,
        quoted_ask=ask_px,
        quoted_bid_sz=bid_sz,
        quoted_ask_sz=ask_sz,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="phase2j-test",
        quote_cycle_id="phase2j",
    )
    return d, half_spread_bps


# ---------------------------------------------------------------------------
# Floor enforces minimum gap regardless of offset_step
# ---------------------------------------------------------------------------


def test_tick_floor_shifts_outer_rung_when_bps_math_underflows() -> None:
    """At offset_step=0.1, the bps math gives rung 1 at half_spread ×
    1.1 = 4.4 bps from mid = ~0.9 ticks below inside_bid → would
    collide on grid. With ``tick_floor_steps=1`` the floor shifts
    the rung outward to exactly inside_bid - 1 tick."""
    d, hs = _decision(mid=2.040, half_spread_bps=4.0)
    cfg = LadderConfig(
        num_levels_per_side=2,
        offset_step=0.1,
        tick_floor_steps=1,
    )
    floor_calls: list[str] = []
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        tick_size=0.001,
        on_rung_floor_adjusted=floor_calls.append,
    )
    # The outer rung MUST land — floor pre-empted the collision-dedup.
    assert len(ladder.bids) == 2, f"expected 2 bid rungs, got {ladder.bids}"
    assert len(ladder.asks) == 2
    # Outer bid sits at inside_bid_grid - 1 tick.
    inside_bid_grid = round(d.quoted_bid / 0.001) * 0.001
    inside_ask_grid = round(d.quoted_ask / 0.001) * 0.001
    assert abs(ladder.bids[1].px - (inside_bid_grid - 0.001)) < 1e-9
    assert abs(ladder.asks[1].px - (inside_ask_grid + 0.001)) < 1e-9
    # Floor callback fired once per side.
    assert floor_calls.count("bid") == 1
    assert floor_calls.count("ask") == 1


def test_tick_floor_steps_2_gives_two_tick_gap() -> None:
    """``tick_floor_steps=2`` reserves 2 ticks of minimum gap per
    rung step. Rung 1 sits at inside − 2 ticks."""
    d, hs = _decision(mid=2.040, half_spread_bps=4.0)
    cfg = LadderConfig(
        num_levels_per_side=2,
        offset_step=0.1,
        tick_floor_steps=2,
    )
    ladder = build_ladder(
        decision=d, cfg=cfg, half_spread_bps=hs, tick_size=0.001,
    )
    inside_bid_grid = round(d.quoted_bid / 0.001) * 0.001
    inside_ask_grid = round(d.quoted_ask / 0.001) * 0.001
    assert abs(ladder.bids[1].px - (inside_bid_grid - 0.002)) < 1e-9
    assert abs(ladder.asks[1].px - (inside_ask_grid + 0.002)) < 1e-9


def test_tick_floor_zero_disables_floor() -> None:
    """``tick_floor_steps=0`` reverts to legacy behaviour: the bps
    math is honoured AS-IS, and the legacy collision-dedup drops the
    outer rung when it would land on the same grid cell."""
    d, hs = _decision(mid=2.040, half_spread_bps=4.0)
    cfg = LadderConfig(
        num_levels_per_side=2,
        offset_step=0.1,
        tick_floor_steps=0,
    )
    drop_calls: list[str] = []
    floor_calls: list[str] = []
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        tick_size=0.001,
        on_rung_dropped=drop_calls.append,
        on_rung_floor_adjusted=floor_calls.append,
    )
    # Legacy path: outer rung was DROPPED via collision-dedup; no
    # floor adjustments fired.
    assert len(ladder.bids) == 1
    assert len(ladder.asks) == 1
    assert drop_calls.count("grid_collision") >= 1
    assert floor_calls == []


def test_tick_floor_no_op_when_bps_math_already_far_enough() -> None:
    """At offset_step=2.0 the bps math gives a wide gap (rung 1 at
    half_spread × 3 = 12 bps = ~2.4 ticks). No floor adjustment
    needed — the rung sits where bps math said."""
    d, hs = _decision(mid=2.040, half_spread_bps=4.0)
    cfg = LadderConfig(
        num_levels_per_side=2,
        offset_step=2.0,
        tick_floor_steps=1,
    )
    floor_calls: list[str] = []
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        tick_size=0.001,
        on_rung_floor_adjusted=floor_calls.append,
    )
    assert len(ladder.bids) == 2
    assert len(ladder.asks) == 2
    # The bps math wins — rung 1 sits at half_spread × 3 = 12 bps
    # below mid (pre-rounding). Floor was NOT applied.
    assert floor_calls == []


def test_tick_floor_n1_is_no_op() -> None:
    """``num_levels_per_side=1`` short-circuits to single-rung output
    BEFORE the floor logic runs. No-op."""
    d, hs = _decision()
    cfg = LadderConfig(
        num_levels_per_side=1,
        offset_step=0.1,
        tick_floor_steps=1,
    )
    floor_calls: list[str] = []
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        tick_size=0.001,
        on_rung_floor_adjusted=floor_calls.append,
    )
    assert len(ladder.bids) == 1
    assert len(ladder.asks) == 1
    assert floor_calls == []


# ---------------------------------------------------------------------------
# Multi-rung walk: floor enforces monotone increasing distance
# ---------------------------------------------------------------------------


def test_tick_floor_4_rungs_monotone_increasing_distance() -> None:
    """At ``num_levels_per_side=4`` with tiny offset_step, EVERY outer
    rung gets floored. Floor distance scales with rung index:
      rung 1 → inside ± 1 tick
      rung 2 → inside ± 2 ticks
      rung 3 → inside ± 3 ticks
    """
    d, hs = _decision(mid=2.040, half_spread_bps=4.0)
    cfg = LadderConfig(
        num_levels_per_side=4,
        offset_step=0.01,  # very small; all outer rungs underflow
        tick_floor_steps=1,
    )
    floor_calls: list[str] = []
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        tick_size=0.001,
        on_rung_floor_adjusted=floor_calls.append,
    )
    assert len(ladder.bids) == 4
    assert len(ladder.asks) == 4
    inside_bid_grid = round(d.quoted_bid / 0.001) * 0.001
    inside_ask_grid = round(d.quoted_ask / 0.001) * 0.001
    for i in range(1, 4):
        # BID side: rung i at inside_bid_grid - i ticks.
        assert abs(
            ladder.bids[i].px - (inside_bid_grid - i * 0.001)
        ) < 1e-9, (
            f"bid rung {i} = {ladder.bids[i].px}, expected "
            f"{inside_bid_grid - i * 0.001}"
        )
        # ASK side mirror.
        assert abs(
            ladder.asks[i].px - (inside_ask_grid + i * 0.001)
        ) < 1e-9
    # 3 outer rungs on each side → 6 floor calls total.
    assert floor_calls.count("bid") == 3
    assert floor_calls.count("ask") == 3


# ---------------------------------------------------------------------------
# Floor + tick_size None / 0 = pass-through
# ---------------------------------------------------------------------------


def test_tick_floor_dormant_when_tick_size_missing() -> None:
    """No ``tick_size`` (degenerate market, missing symbol_spec) →
    floor is dormant (no shifts, no callback)."""
    d, hs = _decision()
    cfg = LadderConfig(
        num_levels_per_side=2,
        offset_step=0.1,
        tick_floor_steps=1,
    )
    floor_calls: list[str] = []
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        tick_size=None,
        on_rung_floor_adjusted=floor_calls.append,
    )
    # No grid → no collision check → both rungs land at bps-math px.
    assert len(ladder.bids) == 2
    assert floor_calls == []
