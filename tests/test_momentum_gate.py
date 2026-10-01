"""Tests for ``app.momentum_gate`` — refuse to add to a position
already aligned with recent price drift.

Targets the snapshot 260510064549 pattern at 01:01-01:05 UTC where
the bot was long +64 SUI after a rebound and the next downtick
filled fresh BUYs at adverse prices. The gate fires on the
*aligned* case (long+uptrend / short+downtrend) on the assumption
that momentum just rewarded the existing position; adding more is
FOMO, not edge.
"""

from __future__ import annotations

from app.enums import QuoteEligibility
from app.momentum_gate import evaluate_momentum_gate


def _kw(**overrides):
    base = {
        "drift_threshold_bps": 2.0,
        "inventory_pct_threshold": 0.40,
        "enabled": True,
    }
    base.update(overrides)
    return base


def test_disabled_returns_no_override() -> None:
    """``enabled=False`` → gate never fires regardless of inputs."""
    override, _ = evaluate_momentum_gate(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=10.0,
        **_kw(enabled=False),
    )
    assert override is None


def test_long_uptrend_aligned_gates_buy_side() -> None:
    """Long position + positive drift = aligned. The gate should
    refuse to add to the long → QUOTE_SELL_ONLY (BUY adds to long
    so it's the side that's gated)."""
    override, reason = evaluate_momentum_gate(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert override == QuoteEligibility.QUOTE_SELL_ONLY
    assert "long_uptrend" in reason


def test_short_downtrend_aligned_gates_sell_side() -> None:
    """Symmetric: short + negative drift = aligned, refuse SELL."""
    override, reason = evaluate_momentum_gate(
        position_qty=-50.0,
        effective_abs_cap=100.0,
        drift_bps=-5.0,
        **_kw(),
    )
    assert override == QuoteEligibility.QUOTE_BUY_ONLY
    assert "short_downtrend" in reason


def test_long_downtrend_anti_aligned_does_not_fire() -> None:
    """Long + negative drift = ANTI-aligned (inventory is on the
    wrong side of momentum). The existing inventory_exec_bias and
    adverse-side-pause logic handle this case; the momentum gate
    deliberately stays out so we don't double-suppress."""
    override, _ = evaluate_momentum_gate(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=-5.0,  # against the long
        **_kw(),
    )
    assert override is None


def test_short_uptrend_anti_aligned_does_not_fire() -> None:
    """Symmetric: short + positive drift = anti-aligned, gate stays
    silent."""
    override, _ = evaluate_momentum_gate(
        position_qty=-50.0,
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert override is None


def test_inventory_below_threshold_does_not_fire() -> None:
    """Even if aligned, the gate should not fire when |position| is
    below the utilisation floor — the existing inventory bias is
    already lax in this regime."""
    override, _ = evaluate_momentum_gate(
        position_qty=20.0,  # 20/100 = 20% < 40% threshold
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert override is None


def test_drift_below_threshold_does_not_fire() -> None:
    """Drift < threshold means we're in noise territory; don't
    fire. The user's snapshot was -3 to -5 bp range during the
    actual whipsaw — a 2 bp threshold catches that without
    being trigger-happy on routine drift."""
    override, _ = evaluate_momentum_gate(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=1.0,  # below 2.0 threshold
        **_kw(),
    )
    assert override is None


def test_none_drift_does_not_fire() -> None:
    """Drift not yet computable (warmup, missing samples) → no fire,
    no error."""
    override, _ = evaluate_momentum_gate(
        position_qty=50.0,
        effective_abs_cap=100.0,
        drift_bps=None,
        **_kw(),
    )
    assert override is None


def test_zero_cap_safety_does_not_fire() -> None:
    """Defensive: cap=0 (mis-config / startup race) shouldn't
    divide-by-zero. Gate stays inactive."""
    override, _ = evaluate_momentum_gate(
        position_qty=50.0,
        effective_abs_cap=0.0,
        drift_bps=5.0,
        **_kw(),
    )
    assert override is None


def test_at_threshold_inclusive_fires() -> None:
    """The inventory_pct check is ``util >= threshold`` —
    inclusive at the boundary — so a position exactly at the
    threshold DOES fire."""
    override, _ = evaluate_momentum_gate(
        position_qty=40.0,  # 40/100 = 40% == threshold
        effective_abs_cap=100.0,
        drift_bps=5.0,
        **_kw(inventory_pct_threshold=0.40),
    )
    assert override == QuoteEligibility.QUOTE_SELL_ONLY
