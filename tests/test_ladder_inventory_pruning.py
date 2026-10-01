"""v1.4.99 — Inventory-aware rung pruning (DORMANT BY DEFAULT).

The pruning is shipped to the binary but disabled by default
(``LADDER_INVENTORY_AWARE_PRUNING_ENABLED=false``). These tests cover:

1. Dormant default — no behaviour change when the flag is False.
   Critical contract: pre-v1.4.99 snapshot data must be reproducible
   bit-identically when the new code is present but the flag is off.
2. Active enabled-state — when the flag is True AND position util
   exceeds the threshold, the adding side's outer rungs are dropped.
3. Reducing side untouched regardless of position direction.
4. Threshold logic — below the threshold = full ladder, at-or-above
   = adding-side prune.
5. Adding side determination — long position → BUY adds; short →
   SELL adds; flat → no prune (no adding side).
6. Defensive fallthroughs — missing inputs degrade to no-prune.
7. on_rung_dropped callback fires once per cut rung.

See ``plans/ladder-observability.md`` for the calibration framework
that should drive the decision to flip this on in production.
"""

from __future__ import annotations

from app.enums import ActiveSides, Side
from app.ladder import LadderConfig, build_ladder
from app.models import QuoteDecision


def _make_decision(
    *,
    quoted_bid_sz: float = 3.0,
    quoted_ask_sz: float = 3.0,
    target_spread_bps: float = 6.0,
    mid: float = 2.000,
    active_sides: ActiveSides = ActiveSides.BOTH,
) -> QuoteDecision:
    """Synthesize a healthy two-sided QuoteDecision."""
    hs = target_spread_bps / 2.0
    bid = mid * (1.0 - hs / 10_000.0)
    ask = mid * (1.0 + hs / 10_000.0)
    from datetime import datetime, timezone

    return QuoteDecision(
        ts=datetime.now(timezone.utc),
        symbol="TON-USDT-SWAP",
        mid_price=mid,
        vol_estimate=1.0,
        inventory=0.0,  # decoupled from build_ladder's position_qty input
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
    enabled: bool = False,
    threshold: float = 0.65,
) -> LadderConfig:
    return LadderConfig(
        num_levels_per_side=n,
        offset_step=1.0,
        size_decay=0.7,
        inside_full_size=True,
        gates_limit_levels=True,
        batch_orders_enabled=False,
        inventory_aware_pruning_enabled=enabled,
        inventory_aware_pruning_threshold_pct=threshold,
    )


# ---------------------------------------------------------------------------
# Dormant default — the critical "no behaviour change" contract
# ---------------------------------------------------------------------------


def test_dormant_default_full_ladder_when_flag_disabled_and_long() -> None:
    """Bot is loaded LONG to 90% util, but pruning_enabled=False.
    Result: full N=2 ladder on both sides (no prune).
    This is the contract that protects current snapshot analysis
    from the dormant code's presence."""
    cfg = _cfg(n=2, enabled=False)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=9.0,  # 90% of max
        max_abs_position=10.0,
    )
    assert len(result.bids) == 2, "BUY side should keep both rungs (dormant)"
    assert len(result.asks) == 2, "SELL side should keep both rungs"


def test_dormant_default_full_ladder_when_flag_disabled_and_short() -> None:
    """Same as above but short — pruning_enabled=False → no prune."""
    cfg = _cfg(n=2, enabled=False)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=-9.0,
        max_abs_position=10.0,
    )
    assert len(result.bids) == 2
    assert len(result.asks) == 2


def test_dormant_default_no_callback_fires() -> None:
    """on_rung_dropped MUST NOT fire when the feature is dormant.
    Even if max_abs_position is at extreme, the callback contract is
    'never invoke us when the feature is off'."""
    callback_calls: list[str] = []
    cfg = _cfg(n=2, enabled=False)
    decision = _make_decision()
    build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=9.5,
        max_abs_position=10.0,
        on_rung_dropped=callback_calls.append,
    )
    assert callback_calls == [], (
        f"Dormant feature must not invoke on_rung_dropped; got {callback_calls}"
    )


def test_dormant_default_omits_position_inputs() -> None:
    """Callers that don't pass position_qty / max_abs_position must
    still get full ladders. Backward-compat for test fixtures and
    scripts that pre-date v1.4.99."""
    cfg = _cfg(n=2, enabled=False)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        # no position_qty, no max_abs_position
    )
    assert len(result.bids) == 2
    assert len(result.asks) == 2


# ---------------------------------------------------------------------------
# Enabled-state behavior — the actual prune
# ---------------------------------------------------------------------------


def test_enabled_long_position_clamps_bid_side_to_single_rung() -> None:
    """Long > threshold → BUY (adding) cut to rung 0 only."""
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=8.0,  # 80% > 65% threshold
        max_abs_position=10.0,
    )
    assert len(result.bids) == 1, "BUY (adding when long) should be 1 rung"
    assert len(result.asks) == 2, "SELL (reducing when long) untouched"


def test_enabled_short_position_clamps_ask_side_to_single_rung() -> None:
    """Short > threshold → SELL (adding) cut to rung 0 only."""
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=-8.0,
        max_abs_position=10.0,
    )
    assert len(result.bids) == 2, "BUY (reducing when short) untouched"
    assert len(result.asks) == 1, "SELL (adding when short) should be 1 rung"


def test_enabled_below_threshold_no_prune() -> None:
    """Position util below threshold → full ladder both sides."""
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=5.0,  # 50% < 65% threshold
        max_abs_position=10.0,
    )
    assert len(result.bids) == 2
    assert len(result.asks) == 2


def test_enabled_flat_position_no_prune() -> None:
    """Position == 0 → no adding side → no prune even when enabled."""
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=0.0,
        max_abs_position=10.0,
    )
    assert len(result.bids) == 2
    assert len(result.asks) == 2


def test_enabled_at_threshold_engages_prune() -> None:
    """At the exact threshold value the prune SHOULD engage
    (``>=`` not ``>``)."""
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=6.5,  # exactly 65%
        max_abs_position=10.0,
    )
    assert len(result.bids) == 1
    assert len(result.asks) == 2


def test_enabled_n3_clamps_to_single_rung_not_two() -> None:
    """With LADDER_NUM_LEVELS_PER_SIDE=3, the prune cuts ADDING
    side to 1 (not 2). Verifies the prune is "single rung on
    adding side" not "reduce by one"."""
    cfg = _cfg(n=3, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=8.0,
        max_abs_position=10.0,
    )
    assert len(result.bids) == 1
    assert len(result.asks) == 3


# ---------------------------------------------------------------------------
# Drop-attribution callback
# ---------------------------------------------------------------------------


def test_callback_fires_with_correct_reason_when_prune_engages() -> None:
    """on_rung_dropped should be called once per cut rung with
    reason ``inventory_aware_pruning``. With N=3 and long, 2 rungs
    are cut on the BUY side."""
    callback_calls: list[str] = []
    cfg = _cfg(n=3, enabled=True, threshold=0.65)
    decision = _make_decision()
    build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=8.0,
        max_abs_position=10.0,
        on_rung_dropped=callback_calls.append,
    )
    # N=3, BUY clamped to 1 → 2 rungs cut on BUY side
    assert callback_calls == [
        "inventory_aware_pruning",
        "inventory_aware_pruning",
    ]


def test_callback_does_not_fire_when_no_prune_needed() -> None:
    """Below threshold → no rung drop → no callback invocation."""
    callback_calls: list[str] = []
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=3.0,  # below threshold
        max_abs_position=10.0,
        on_rung_dropped=callback_calls.append,
    )
    assert callback_calls == []


def test_callback_exception_does_not_break_build() -> None:
    """An observer callback that raises must not break the ladder
    build. The contract is 'best-effort observability, never blocks
    the hot path'."""

    def bad_callback(reason: str) -> None:
        raise RuntimeError("observer blew up")

    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    # Should NOT raise — the build_ladder side swallows callback
    # exceptions defensively.
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=8.0,
        max_abs_position=10.0,
        on_rung_dropped=bad_callback,
    )
    # Prune still happened correctly.
    assert len(result.bids) == 1


# ---------------------------------------------------------------------------
# Defensive fallthroughs — malformed inputs degrade to no-prune
# ---------------------------------------------------------------------------


def test_enabled_missing_position_qty_no_prune() -> None:
    """position_qty=None → no prune even when enabled."""
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=None,
        max_abs_position=10.0,
    )
    assert len(result.bids) == 2
    assert len(result.asks) == 2


def test_enabled_missing_max_abs_position_no_prune() -> None:
    """max_abs_position=None → no prune even when enabled."""
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=8.0,
        max_abs_position=None,
    )
    assert len(result.bids) == 2
    assert len(result.asks) == 2


def test_enabled_zero_max_abs_position_no_prune() -> None:
    """max_abs_position=0 → no prune (defensive against div-by-zero)."""
    cfg = _cfg(n=2, enabled=True, threshold=0.65)
    decision = _make_decision()
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=3.0,
        position_qty=8.0,
        max_abs_position=0.0,
    )
    assert len(result.bids) == 2
    assert len(result.asks) == 2
