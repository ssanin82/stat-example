"""Tests for app.ladder.build_ladder() — Phase 1 (shadow mode).

The N=1 path must be bit-identical to the input QuoteDecision so
that flipping the LADDER_NUM_LEVELS_PER_SIDE knob to 1 reproduces
v1.2.13 behavior exactly. The N>1 path is tested for:
* correct geometric size decay
* correct linear price offset
* gate-cap clamping
* per-side suppression via ActiveSides
* JSON serialisation round-trip for storage
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.enums import ActiveSides, Side
from app.ladder import (
    LadderConfig,
    LadderDecision,
    LadderRung,
    build_ladder,
)
from app.models import QuoteDecision


def _decision(
    *,
    mid: float = 1.0,
    reservation: float = 1.0,
    half_spread_bps: float = 5.0,
    bid_sz: float = 10.0,
    ask_sz: float = 10.0,
    active: ActiveSides = ActiveSides.BOTH,
) -> tuple[QuoteDecision, float]:
    """Build a minimal QuoteDecision; returns (decision, half_spread_bps)
    so build_ladder() callers don't have to recompute."""
    bid_px = reservation * (1.0 - half_spread_bps / 10_000.0)
    ask_px = reservation * (1.0 + half_spread_bps / 10_000.0)
    d = QuoteDecision(
        ts=datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc),
        symbol="SUI-USDT-SWAP",
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
        active_sides=active,
        toxicity_score=0.0,
        decision_reason="test",
        quote_cycle_id="test-cycle",
    )
    return d, half_spread_bps


# -------------------------------------------------------------------
# N = 1 (default): bit-identical to input
# -------------------------------------------------------------------


def test_n1_returns_one_rung_per_side_matching_inside_scalar() -> None:
    """At N=1 the ladder must wrap the input quote without any
    arithmetic — the inside rung's px / sz exactly equal the
    decision's quoted_bid / quoted_bid_sz / etc."""
    d, hs = _decision(bid_sz=12.34, ask_sz=12.78)
    cfg = LadderConfig(num_levels_per_side=1)
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    assert len(ladder.bids) == 1
    assert len(ladder.asks) == 1
    assert ladder.bids[0].level_idx == 0
    assert ladder.bids[0].side == Side.BUY
    assert ladder.bids[0].px == d.quoted_bid
    assert ladder.bids[0].sz == d.quoted_bid_sz
    assert ladder.asks[0].level_idx == 0
    assert ladder.asks[0].side == Side.SELL
    assert ladder.asks[0].px == d.quoted_ask
    assert ladder.asks[0].sz == d.quoted_ask_sz
    assert ladder.requested_levels == 1
    assert ladder.effective_levels_buy == 1
    assert ladder.effective_levels_sell == 1


def test_n1_with_zero_size_omits_side() -> None:
    """When the upstream quote engine has set quoted_bid_sz=0 (e.g.
    inventory bias suppressing the bid side), the ladder shouldn't
    invent a rung. Empty bids list, ask still present."""
    d, hs = _decision(bid_sz=0.0, ask_sz=10.0)
    cfg = LadderConfig(num_levels_per_side=1)
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    assert ladder.bids == []
    assert len(ladder.asks) == 1


def test_n1_one_sided_active_suppresses_other_side() -> None:
    """ActiveSides.BID_ONLY → only bid rung; ASK_ONLY → only ask;
    NONE → empty."""
    d, hs = _decision(active=ActiveSides.BID_ONLY)
    cfg = LadderConfig(num_levels_per_side=1)
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    assert len(ladder.bids) == 1
    assert ladder.asks == []

    d2, hs2 = _decision(active=ActiveSides.ASK_ONLY)
    ladder2 = build_ladder(decision=d2, cfg=cfg, half_spread_bps=hs2)
    assert ladder2.bids == []
    assert len(ladder2.asks) == 1

    d3, hs3 = _decision(active=ActiveSides.NONE)
    ladder3 = build_ladder(decision=d3, cfg=cfg, half_spread_bps=hs3)
    assert ladder3.bids == []
    assert ladder3.asks == []


# -------------------------------------------------------------------
# N > 1: ladder shape
# -------------------------------------------------------------------


def test_n5_geometric_size_decay() -> None:
    """At N=5, decay=0.5, inside_full_size=True: rung sizes are
    base × 1.0, base × 0.5, base × 0.25, base × 0.125, base × 0.0625."""
    d, hs = _decision(bid_sz=10.0, ask_sz=10.0)
    cfg = LadderConfig(
        num_levels_per_side=5, size_decay=0.5, inside_full_size=True
    )
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    bid_sizes = [r.sz for r in ladder.bids]
    expected = [10.0, 5.0, 2.5, 1.25, 0.625]
    for got, exp in zip(bid_sizes, expected):
        assert got == pytest.approx(exp, rel=1e-9)


def test_n5_linear_price_offset() -> None:
    """At N=5, offset_step=1.0, half_spread=10 bps: rungs at
    1×, 2×, 3×, 4×, 5× half_spread away from reservation. Rung 0
    matches the inside quote (post-clamp from compute_quote_decision)."""
    d, hs = _decision(reservation=1.0, half_spread_bps=10.0)
    cfg = LadderConfig(num_levels_per_side=5, offset_step=1.0)
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    # inside rung == quoted_bid / quoted_ask exactly
    assert ladder.bids[0].px == d.quoted_bid
    assert ladder.asks[0].px == d.quoted_ask
    # outer rungs: bid_px[i] = reservation × (1 - hs × (1 + i) / 10000)
    for i in range(1, 5):
        expected_bid = 1.0 * (1.0 - 10.0 * (1.0 + 1.0 * i) / 10_000.0)
        expected_ask = 1.0 * (1.0 + 10.0 * (1.0 + 1.0 * i) / 10_000.0)
        assert ladder.bids[i].px == pytest.approx(expected_bid, rel=1e-12)
        assert ladder.asks[i].px == pytest.approx(expected_ask, rel=1e-12)


def test_n3_offset_step_half() -> None:
    """offset_step=0.5 produces tighter spacing: rungs at 1×, 1.5×, 2×."""
    d, hs = _decision(reservation=1.0, half_spread_bps=10.0)
    cfg = LadderConfig(num_levels_per_side=3, offset_step=0.5)
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    factors = [(d.quoted_bid - r.px) / d.quoted_bid * 10_000.0 for r in ladder.bids]
    # rung 0 = inside (factor 0 vs quoted_bid), rungs 1/2 outer
    assert factors[0] == pytest.approx(0.0, abs=1e-9)
    # bid_px[1] = reservation × (1 - hs × (1 + 0.5 × 1) / 10_000)
    #           = 1.0 × (1 - 10 × 1.5 / 10_000) = 1 - 0.0015 = 0.9985
    assert ladder.bids[1].px == pytest.approx(0.9985, rel=1e-9)
    # bid_px[2] = 1.0 × (1 - 10 × 2.0 / 10_000) = 0.998
    assert ladder.bids[2].px == pytest.approx(0.998, rel=1e-9)


def test_n5_default_decay_07() -> None:
    """Smoke check the documented default decay = 0.7, inside_full_size=True."""
    d, hs = _decision(bid_sz=10.0)
    cfg = LadderConfig(num_levels_per_side=5)  # all defaults
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    bid_sizes = [r.sz for r in ladder.bids]
    expected = [10.0, 7.0, 4.9, 3.43, 2.401]
    for got, exp in zip(bid_sizes, expected):
        assert got == pytest.approx(exp, rel=1e-9)


# -------------------------------------------------------------------
# Gate caps
# -------------------------------------------------------------------


def test_gate_cap_clamps_n() -> None:
    """A gate publishing cap=2 with cfg.num_levels_per_side=5 yields
    exactly 2 rungs per side."""
    d, hs = _decision()
    cfg = LadderConfig(num_levels_per_side=5, gates_limit_levels=True)
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        gate_caps={"vol_trend_gate": 2},
    )
    assert len(ladder.bids) == 2
    assert len(ladder.asks) == 2
    assert ladder.requested_levels == 5
    assert ladder.effective_levels_buy == 2
    assert ladder.effective_levels_sell == 2


def test_multiple_gate_caps_take_min() -> None:
    """When multiple gates fire, the most-restrictive cap wins."""
    d, hs = _decision()
    cfg = LadderConfig(num_levels_per_side=5)
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        gate_caps={"vol_trend_gate": 3, "post_swing": 1, "basis_regime": 2},
    )
    # min of (5, 3, 1, 2) = 1
    assert len(ladder.bids) == 1
    assert ladder.effective_levels_buy == 1


def test_gate_caps_ignored_when_flag_off() -> None:
    """Setting gates_limit_levels=False bypasses gate caps even when
    they're supplied — useful for shadow-mode unclamped calibration."""
    d, hs = _decision()
    cfg = LadderConfig(num_levels_per_side=5, gates_limit_levels=False)
    ladder = build_ladder(
        decision=d, cfg=cfg, half_spread_bps=hs, gate_caps={"x": 1}
    )
    assert len(ladder.bids) == 5
    assert len(ladder.asks) == 5


def test_none_cap_does_nothing() -> None:
    """A gate with cap=None (gate exists but isn't firing) shouldn't
    clamp."""
    d, hs = _decision()
    cfg = LadderConfig(num_levels_per_side=5)
    ladder = build_ladder(
        decision=d, cfg=cfg, half_spread_bps=hs, gate_caps={"x": None}
    )
    assert len(ladder.bids) == 5


# -------------------------------------------------------------------
# Serialisation
# -------------------------------------------------------------------


def test_ladder_to_dict_round_trip() -> None:
    """LadderDecision.to_dict() must produce a JSON-safe dict that
    survives ``json.dumps``. The DB writes it as a TEXT column."""
    import json
    d, hs = _decision()
    cfg = LadderConfig(num_levels_per_side=3)
    ladder = build_ladder(
        decision=d,
        cfg=cfg,
        half_spread_bps=hs,
        gate_caps={"vol_trend_gate": None, "basis_regime": 2},
    )
    out = ladder.to_dict()
    encoded = json.dumps(out)
    decoded = json.loads(encoded)
    # Round-trip preserves shape; we don't reconstruct LadderDecision
    # from JSON in production (it's read-only telemetry) so equality
    # at the dict level is sufficient. basis_regime cap=2 clamps the
    # configured 3 rungs down to 2.
    assert decoded["requested_levels"] == 3
    assert decoded["effective_levels_buy"] == 2
    assert decoded["effective_levels_sell"] == 2
    assert len(decoded["bids"]) == 2
    assert len(decoded["asks"]) == 2
    assert decoded["bids"][0]["level"] == 0
    assert decoded["bids"][0]["side"] == "BUY"
    assert decoded["asks"][0]["side"] == "SELL"
    assert decoded["gate_caps"] == {"vol_trend_gate": None, "basis_regime": 2}


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------


def test_n1_inside_px_unchanged_even_with_offset_step() -> None:
    """offset_step has no effect at N=1 because only rung 0 exists,
    and rung 0 always uses the inside scalar override."""
    d, hs = _decision()
    cfg_default = LadderConfig(num_levels_per_side=1, offset_step=1.0)
    cfg_alt = LadderConfig(num_levels_per_side=1, offset_step=10.0)
    l1 = build_ladder(decision=d, cfg=cfg_default, half_spread_bps=hs)
    l2 = build_ladder(decision=d, cfg=cfg_alt, half_spread_bps=hs)
    assert l1.bids[0].px == l2.bids[0].px == d.quoted_bid


def test_zero_decay_collapses_outer_sizes() -> None:
    """size_decay=0.1 (lowest allowed) shrinks aggressively. Validates
    the formula doesn't break at small ratios."""
    d, hs = _decision(bid_sz=100.0)
    cfg = LadderConfig(num_levels_per_side=3, size_decay=0.1)
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    sizes = [r.sz for r in ladder.bids]
    assert sizes[0] == pytest.approx(100.0)
    assert sizes[1] == pytest.approx(10.0)
    assert sizes[2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# v1.4.79 — grid-collision dedup.
#
# Snapshot v1.4.78-260519-081151 caught two same-side same-price orders
# on the book (BUY 2.010 ×2, SELL 2.013 ×2). With compressed half-spread,
# the outer-rung offset shrinks below the tick size, and two distinct
# pre-round prices collapse to the same grid value after rounding. Phase
# 4-prequel adds tick-size to ``build_ladder`` so colliding outer rungs
# are skipped.
# ---------------------------------------------------------------------------


def test_v1_4_79_outer_rung_dropped_when_grid_collision() -> None:
    """v1.4.79 (refined v1.4.86): when half_spread is compressed,
    both rungs round to the same grid value under ROUND_HALF_UP.
    With ``tick_size`` passed, the outer rung is dropped.

    v1.4.86 note: the dedup now uses ROUND_HALF_UP prediction to
    match ``round_price_and_size_to_grid``'s actual rounding. To
    trigger collision we need a case where both rungs ROUND TO THE
    SAME tick — at mid=2.0325 with hs=1.5, lvl 0 raw 2.0322 rounds
    to 2.032 and lvl 1 raw 2.0319 also rounds to 2.032. Collision.
    """
    d, hs = _decision(mid=2.0325, reservation=2.0325, half_spread_bps=1.5)
    # v1.4.169 Phase 2J: disable the new tick-floor for this test so
    # the LEGACY grid-collision dedup path is what's exercised. With
    # ``tick_floor_steps > 0``, the new floor would pre-empt this
    # path by shifting the outer rung outward instead of dropping it.
    # The new behaviour is BETTER; the legacy path is now
    # defense-in-depth only. Phase 2J's own tests cover the floor
    # behaviour.
    cfg = LadderConfig(num_levels_per_side=2, offset_step=1.0, tick_floor_steps=0)
    ladder = build_ladder(
        decision=d, cfg=cfg, half_spread_bps=hs, tick_size=0.001,
    )
    # Inside rung only; outer rung was a grid collision and dropped.
    assert len(ladder.bids) == 1, (
        f"outer BUY rung should be dropped on grid collision; got {ladder.bids}"
    )
    assert ladder.bids[0].level_idx == 0
    assert len(ladder.asks) == 1
    assert ladder.asks[0].level_idx == 0
    # effective_levels reflects the dedup.
    assert ladder.effective_levels_buy == 1
    assert ladder.effective_levels_sell == 1


def test_v1_4_86_dedup_uses_round_half_up_not_floor() -> None:
    """v1.4.86 fix: when raw outer-rung px sits in the round-up half
    of a tick, ``_tick_floor`` (pre-fix) predicted a lower tick than
    the actual ``ROUND_HALF_UP`` rounding produces. The dedup then
    failed to detect the collision and the outer rung landed on the
    book at the same price as the inside rung.

    Specifically: at mid=2.01 hs=1.5, lvl 0 raw is 2.0096985 (rounds
    UP to 2.010 under ROUND_HALF_UP) and lvl 1 raw is 2.009397
    (rounds DOWN to 2.009). They land on DIFFERENT ticks — no
    collision. Old `_tick_floor` dedup predicted both at 2.009 →
    falsely dropped the outer rung. New ROUND_HALF_UP dedup
    correctly keeps both.
    """
    d, hs = _decision(mid=2.01, reservation=2.01, half_spread_bps=1.5)
    cfg = LadderConfig(num_levels_per_side=2, offset_step=1.0)
    ladder = build_ladder(
        decision=d, cfg=cfg, half_spread_bps=hs, tick_size=0.001,
    )
    # Both rungs SHOULD be kept — under ROUND_HALF_UP they land on
    # distinct ticks (lvl 0 → 2.010, lvl 1 → 2.009 for BUY).
    assert len(ladder.bids) == 2, (
        f"both BUY rungs should be kept (distinct ticks under ROUND_HALF_UP); "
        f"got {ladder.bids}"
    )
    # Verify the predicted ticks match the actual rounding direction.
    from decimal import Decimal, ROUND_HALF_UP as RHU
    tick = Decimal("0.001")
    for rung in ladder.bids:
        rounded = float(Decimal(str(rung.px)).quantize(tick, rounding=RHU))
        # Each rung's raw px should be within half a tick of its rounded value.
        assert abs(rung.px - rounded) <= 0.0005 + 1e-9, (
            f"rung px {rung.px} should round to {rounded}"
        )


def test_v1_4_79_outer_rung_kept_when_grid_distinct() -> None:
    """v1.4.79: with normal half_spread (e.g. 13 bps at price 2.0),
    rungs are 2.6 ticks apart and grid-distinct. Both kept.
    """
    d, hs = _decision(mid=2.0, reservation=2.0, half_spread_bps=13.0)
    cfg = LadderConfig(num_levels_per_side=2, offset_step=1.0)
    ladder = build_ladder(
        decision=d, cfg=cfg, half_spread_bps=hs, tick_size=0.001,
    )
    # Both rungs kept — distinct grid values after rounding.
    assert len(ladder.bids) == 2
    assert ladder.bids[0].level_idx == 0
    assert ladder.bids[1].level_idx == 1
    assert len(ladder.asks) == 2
    # BID rungs descending (lvl 1 lower than lvl 0).
    assert ladder.bids[1].px < ladder.bids[0].px
    # ASK rungs ascending.
    assert ladder.asks[1].px > ladder.asks[0].px


def test_v1_4_79_default_no_tick_size_preserves_legacy_behavior() -> None:
    """v1.4.79 backward compat: when ``tick_size`` is None (default),
    the dedup logic is OFF and all configured rungs are emitted —
    same as v1.4.78 behavior.
    """
    # Compressed half_spread would normally cause collision.
    d, hs = _decision(mid=2.01, reservation=2.01, half_spread_bps=1.5)
    cfg = LadderConfig(num_levels_per_side=2, offset_step=1.0)
    ladder = build_ladder(decision=d, cfg=cfg, half_spread_bps=hs)
    # Without tick_size: both rungs emitted (collision detected
    # downstream, not here).
    assert len(ladder.bids) == 2
    assert len(ladder.asks) == 2


def test_v1_4_79_three_rung_drops_middle_collision_only() -> None:
    """v1.4.79 (refined v1.4.86): with 3 rungs and compressed
    half_spread, the dedup ensures no two rungs share a final grid
    value. The specific number kept depends on offsets, but the
    invariant — distinct grid values for all kept BUY rungs — holds.

    Note this can expose a discontinuity in level_idx (e.g. 0, 2) —
    the reconciler must handle that (it does; it iterates by slot
    key not consecutive integers).
    """
    d, hs = _decision(mid=2.01, reservation=2.01, half_spread_bps=2.5)
    cfg = LadderConfig(num_levels_per_side=3, offset_step=1.0)
    ladder = build_ladder(
        decision=d, cfg=cfg, half_spread_bps=hs, tick_size=0.001,
    )
    # Invariant: no two BUY rungs share a final grid value (under
    # ROUND_HALF_UP, matching actual placement).
    from decimal import Decimal, ROUND_HALF_UP
    tick = Decimal("0.001")
    bid_grids = [
        float(Decimal(str(r.px)).quantize(tick, rounding=ROUND_HALF_UP))
        for r in ladder.bids
    ]
    assert len(bid_grids) == len(set(bid_grids)), (
        f"all BUY rungs must have distinct ROUND_HALF_UP grid values; "
        f"got rung px={[r.px for r in ladder.bids]} → grids={bid_grids}"
    )


def test_v1_4_79_n1_path_unaffected() -> None:
    """v1.4.79: N=1 fast path unchanged — tick_size is irrelevant."""
    d, hs = _decision()
    cfg = LadderConfig(num_levels_per_side=1)
    ladder = build_ladder(
        decision=d, cfg=cfg, half_spread_bps=hs, tick_size=0.001,
    )
    assert len(ladder.bids) == 1
    assert len(ladder.asks) == 1
    assert ladder.bids[0].px == d.quoted_bid
    assert ladder.asks[0].px == d.quoted_ask
