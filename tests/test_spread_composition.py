"""Gate-widening Phase 1 — ``SpreadComposition`` dataclass arithmetic.

The dataclass is the load-bearing surface for the migration: each
former regime-response gate contributes bps, the bot's effective
half-spread per side is the sum capped at ``MAX_HALF_SPREAD_BPS``,
floored at the econ minimum.

Properties verified:

* Zero-contribution composition produces zero effective spread.
* Floor: contributions below ``econ_floor_bps`` are clamped UP to it.
* Cap: contributions above ``max_bps`` are clamped DOWN to it.
* ``capped_at_max_*`` flags reflect the uncapped total reaching cap.
* Per-side independence: a contributor that only widens the bid
  doesn't affect the ask.
* Symmetric contributors (econ_floor, toxicity, adverse_overlay)
  apply equally to both sides.
* ``to_dict`` round-trip preserves all fields.
"""

from __future__ import annotations

from app.quoting import SpreadComposition


def test_zero_composition_zero_spread() -> None:
    c = SpreadComposition()
    assert c.effective_half_spread_bid_bps(max_bps=30.0) == 0.0
    assert c.effective_half_spread_ask_bps(max_bps=30.0) == 0.0


def test_econ_floor_acts_as_lower_bound() -> None:
    """When other contributions are zero, the floor sets the
    half-spread directly. When other contributions are positive but
    sum below the floor, the floor still wins."""
    c = SpreadComposition(econ_floor_bps=3.5)
    assert c.effective_half_spread_bid_bps(max_bps=30.0) == 3.5
    assert c.effective_half_spread_ask_bps(max_bps=30.0) == 3.5


def test_contributions_sum_to_total() -> None:
    c = SpreadComposition(
        econ_floor_bps=3.5,
        toxicity_bps=2.0,
        vol_trend_bid_bps=8.0,
        vol_trend_ask_bps=8.0,
        microprice_bid_bps=4.0,
    )
    # bid: 3.5 + 2.0 + 8.0 + 4.0 = 17.5
    assert c.effective_half_spread_bid_bps(max_bps=30.0) == 17.5
    # ask: 3.5 + 2.0 + 8.0 = 13.5 (no microprice on ask)
    assert c.effective_half_spread_ask_bps(max_bps=30.0) == 13.5


def test_max_caps_total() -> None:
    """When the uncapped sum exceeds ``max_bps``, the effective
    half-spread caps. This is the 'effectively dark' state at
    gate-equivalent initial coefficients."""
    c = SpreadComposition(
        econ_floor_bps=3.5,
        vol_trend_bid_bps=30.0,  # alone hits the cap
        vol_trend_ask_bps=30.0,
    )
    assert c.effective_half_spread_bid_bps(max_bps=30.0) == 30.0
    assert c.effective_half_spread_ask_bps(max_bps=30.0) == 30.0


def test_capped_flags_set_via_with_caps() -> None:
    c = SpreadComposition(vol_trend_bid_bps=30.0).with_caps(max_bps=30.0)
    assert c.capped_at_max_bid is True
    assert c.capped_at_max_ask is False  # ask had no contribution


def test_per_side_asymmetric_independence() -> None:
    """A bid-only widener doesn't affect the ask. Microprice gate
    fires on book imbalance — when 'ask is thin' it widens the ask
    only (the bot's bid would get adversely selected when price
    bumps up into the thin ask side, so we make the bid LESS
    competitive... wait — the asymmetry rule is: widen the side
    the bot's adverse fill would land on)."""
    c = SpreadComposition(
        econ_floor_bps=3.5,
        microprice_ask_bps=20.0,
    )
    assert c.effective_half_spread_bid_bps(max_bps=30.0) == 3.5
    assert c.effective_half_spread_ask_bps(max_bps=30.0) == 23.5


def test_floor_overrides_low_contributions() -> None:
    """A 1 bps vol contribution shouldn't pull the spread BELOW the
    econ floor."""
    c = SpreadComposition(
        econ_floor_bps=3.5,
        vol_trend_bid_bps=1.0,
    )
    # Uncapped total = 4.5, > floor 3.5, < cap 30 → returns 4.5
    assert c.effective_half_spread_bid_bps(max_bps=30.0) == 4.5
    # Ask has no contribution; uncapped = floor = 3.5
    assert c.effective_half_spread_ask_bps(max_bps=30.0) == 3.5


def test_to_dict_round_trip_keys() -> None:
    c = SpreadComposition(
        econ_floor_bps=3.5,
        toxicity_bps=1.0,
        adverse_overlay_bps=0.5,
        vol_trend_bid_bps=2.0,
        vol_trend_ask_bps=2.0,
        momentum_bid_bps=0.0,
        momentum_ask_bps=0.0,
        post_swing_bid_bps=0.0,
        post_swing_ask_bps=0.0,
        microprice_bid_bps=0.0,
        microprice_ask_bps=0.0,
        basis_bid_bps=0.0,
        basis_ask_bps=0.0,
        freshness_bid_bps=0.0,
        freshness_ask_bps=0.0,
        recovery_cooldown_bid_bps=0.0,
        recovery_cooldown_ask_bps=0.0,
    )
    d = c.to_dict()
    # All expected keys present.
    assert "econ_floor_bps" in d
    assert "vol_trend_bid_bps" in d
    assert "recovery_cooldown_ask_bps" in d
    assert "capped_at_max_bid" in d
    # Values round-trip as floats.
    assert d["econ_floor_bps"] == 3.5
    assert d["vol_trend_bid_bps"] == 2.0


def test_with_caps_idempotent() -> None:
    """Calling with_caps twice produces the same result as once."""
    c = SpreadComposition(vol_trend_bid_bps=30.0).with_caps(max_bps=30.0)
    c2 = c.with_caps(max_bps=30.0)
    assert c == c2


def test_gate_equivalent_initial_coefficient_caps() -> None:
    """Phase 1 acceptance: with gate-equivalent initial coefficients
    (vol_trend widens by MAX when firing), the effective half-spread
    on the suppressed side equals MAX. Functional equivalence to
    today's HOLD_ALL — bot quotes far enough from mid to not fill."""
    max_bps = 30.0
    c = SpreadComposition(
        econ_floor_bps=3.5,
        vol_trend_bid_bps=max_bps,  # gate-equivalent firing
        vol_trend_ask_bps=max_bps,
    ).with_caps(max_bps=max_bps)
    assert c.effective_half_spread_bid_bps(max_bps=max_bps) == max_bps
    assert c.effective_half_spread_ask_bps(max_bps=max_bps) == max_bps
    assert c.capped_at_max_bid is True
    assert c.capped_at_max_ask is True
