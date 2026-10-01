"""Phase 4C.2.a + 4C.2.c (v1.5.146) — dampen band on the economics gate.

The refuse band (4C.1 + 4C.2, shipped v1.4.164) is binary: when the
adjusted per-side expected edge falls below
``MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE`` the side stops quoting. This
is correct in deeply adverse economics but binary-cuts the bot's
participation across the WHOLE marginal band — anywhere between the
refuse threshold and break-even, the bot can either quote at full
spread (probably losing) or not at all.

4C.2.a inserts a dampen band: when ``adjusted_edge ∈ (refuse, dampen_max]``
the bot KEEPS QUOTING that side with a widened spread. The
widening is emitted as a new ``SpreadComposition`` contributor
``negative_expectancy_dampen_{bid,ask}_bps`` (4C.2.c — the
contributor's presence in the composition IS the audit trail).

Tests pin:
* Pure-helper behaviour (band membership, disable conditions,
  precedence vs. refuse, confidence multiplier composition).
* ``SpreadComposition`` field plumbing (totals + to_dict).
"""

from __future__ import annotations

import pytest

from app.expected_edge import (
    ExpectedEdgeDampenResult,
    evaluate_per_side_dampen_band,
)
from app.quoting import SpreadComposition


# Default inputs used across the helper tests — refuse band at -1.0,
# dampen ceiling at 0.0 (the canonical operator-config pattern),
# widening 2.0 bp.
_REFUSE = -1.0
_DAMPEN_MAX = 0.0
_WIDEN = 2.0


def _make_inputs(
    *,
    target_hs_bps: float | None = 5.0,
    rebate: float = 1.0,
    adverse: float = 6.5,
    conf_mult: float = 1.0,
    already_refused: bool = False,
    refuse: float = _REFUSE,
    dampen_max: float = _DAMPEN_MAX,
    widen: float = _WIDEN,
) -> dict:
    """Helper that mirrors the bot's call-site keyword set. With the
    above defaults the raw edge is 5.0 + 1.0 - 6.5 = -0.5 bp — sits
    squarely in the (-1.0, 0.0] dampen band."""
    return dict(
        target_half_spread_bps=target_hs_bps,
        refuse_threshold_bps=refuse,
        dampen_max_bps=dampen_max,
        dampen_widen_bps=widen,
        maker_rebate_bps=rebate,
        typical_adverse_markout_bps=adverse,
        confidence_multiplier=conf_mult,
        already_refused=already_refused,
    )


# -------------------------------------------------------------------
# Band membership — three buckets: above, inside, below
# -------------------------------------------------------------------


def test_above_dampen_max_returns_zero_widening() -> None:
    """Edge = +1.0 (break-even or better): no widening, NOT armed.
    The bot quotes that side at its normal spread."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(target_hs_bps=7.5)
    )
    # 7.5 + 1.0 - 6.5 = 2.0 → above dampen_max
    assert r.armed is False
    assert r.widen_bps == 0.0
    assert r.expected_edge_bps == pytest.approx(2.0)


def test_inside_band_arms_with_full_widening() -> None:
    """Edge = -0.5 (marginally negative, above refuse): armed, full
    widen_bps returned. This is the core dampen-band fire path."""
    r = evaluate_per_side_dampen_band(**_make_inputs())
    assert r.armed is True
    assert r.widen_bps == _WIDEN
    assert r.expected_edge_bps == pytest.approx(-0.5)


def test_at_dampen_max_boundary_inclusive_arms() -> None:
    """Boundary at the upper edge of the band is INCLUSIVE — edge =
    exactly 0.0 → armed. This matches the half-open band notation in
    the docstring ``(refuse, dampen_max]``."""
    # 5.5 + 1.0 - 6.5 = 0.0 exactly
    r = evaluate_per_side_dampen_band(
        **_make_inputs(target_hs_bps=5.5)
    )
    assert r.armed is True
    assert r.widen_bps == _WIDEN
    assert r.expected_edge_bps == pytest.approx(0.0)


def test_at_refuse_boundary_exclusive_does_not_arm() -> None:
    """Boundary at the lower edge of the band is EXCLUSIVE — edge =
    exactly the refuse threshold is owned by the refuse evaluator,
    not the dampen band. Mirror of the docstring's ``(refuse, ...]``
    notation."""
    # 4.5 + 1.0 - 6.5 = -1.0 exactly
    r = evaluate_per_side_dampen_band(
        **_make_inputs(target_hs_bps=4.5)
    )
    assert r.armed is False
    assert r.widen_bps == 0.0
    assert r.expected_edge_bps == pytest.approx(-1.0)


def test_below_refuse_returns_zero_widening() -> None:
    """Edge < refuse threshold: the refuse evaluator would have
    fired; the dampen evaluator must NOT also pile on widening on
    top of the refusal."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(target_hs_bps=3.5)
    )
    # 3.5 + 1.0 - 6.5 = -2.0 (below refuse=-1.0)
    assert r.armed is False
    assert r.widen_bps == 0.0


# -------------------------------------------------------------------
# Disable conditions
# -------------------------------------------------------------------


def test_widen_bps_zero_is_dormant() -> None:
    """Default config (DAMPEN_WIDEN_BPS=0.0): helper returns the
    dormant result with ``armed=False, widen_bps=0`` regardless of
    edge value. The bot's existing refuse-only behaviour is
    preserved bit-identically."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(widen=0.0)
    )
    assert r.armed is False
    assert r.widen_bps == 0.0


def test_refuse_threshold_zero_dormant() -> None:
    """Operator set DAMPEN_WIDEN_BPS > 0 but kept the refuse
    threshold at 0.0 (=refuse band itself disabled). The dampen band
    is meaningless without a refuse anchor — must short-circuit to
    dormant."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(refuse=0.0, dampen_max=0.5)
    )
    assert r.armed is False
    assert r.widen_bps == 0.0


def test_pathological_band_inverted_dormant() -> None:
    """Operator misconfigured: DAMPEN_MAX_BPS <= refuse threshold.
    The band is empty / inverted; treat as dormant rather than
    firing on every tick."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(refuse=-1.0, dampen_max=-1.0)
    )
    assert r.armed is False
    assert r.widen_bps == 0.0


def test_missing_target_half_spread_dormant() -> None:
    """No edge signal yet (composition not ready / degenerate market
    state) → dormant. Defensive — caller may pass None during
    startup."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(target_hs_bps=None)
    )
    assert r.armed is False
    assert r.widen_bps == 0.0


# -------------------------------------------------------------------
# Composition with the refuse band
# -------------------------------------------------------------------


def test_already_refused_short_circuits() -> None:
    """When the refuse evaluator already fired for this side this
    tick, dampen returns zero widening — no point adding widening
    to a side that won't quote. The refuse precedence rule."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(already_refused=True)
    )
    assert r.armed is False
    assert r.widen_bps == 0.0


# -------------------------------------------------------------------
# Phase 4C.3 confidence multiplier composition
# -------------------------------------------------------------------


def test_confidence_multiplier_can_push_into_band() -> None:
    """Raw edge = +2.0 (healthy); multiplier = 0.1 (recent realised
    edge poor) → adjusted edge = 0.2. Above dampen_max → NOT armed.
    Confirms the multiplier is applied before band comparison."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(target_hs_bps=7.5, conf_mult=0.1)
    )
    assert r.expected_edge_bps == pytest.approx(0.2)
    assert r.armed is False


def test_confidence_multiplier_can_lift_out_of_band() -> None:
    """Raw edge = -0.5 (inside band); multiplier = 3.0 (recent
    realised edge very strong) → adjusted edge = -1.5. Below refuse
    threshold (-1.0) so NOT armed (would be refused)."""
    r = evaluate_per_side_dampen_band(
        **_make_inputs(target_hs_bps=5.0, conf_mult=3.0)
    )
    assert r.expected_edge_bps == pytest.approx(-1.5)
    assert r.armed is False


# -------------------------------------------------------------------
# Return type stability
# -------------------------------------------------------------------


def test_result_is_frozen_dataclass() -> None:
    """``ExpectedEdgeDampenResult`` is immutable — pinning the
    contract so callers can't accidentally mutate it."""
    r = evaluate_per_side_dampen_band(**_make_inputs())
    with pytest.raises((AttributeError, Exception)):
        r.widen_bps = 99.0  # type: ignore[misc]


# -------------------------------------------------------------------
# SpreadComposition plumbing
# -------------------------------------------------------------------


def test_composition_fields_default_zero() -> None:
    """Default-constructed composition has the new dampen fields at
    zero — pre-v1.5.146 callers that don't pass the new fields are
    unaffected."""
    c = SpreadComposition()
    assert c.negative_expectancy_dampen_bid_bps == 0.0
    assert c.negative_expectancy_dampen_ask_bps == 0.0


def test_composition_totals_include_dampen() -> None:
    """The new fields are folded into ``total_*_bps_uncapped`` so
    the effective half-spread reflects them. Bid-only dampen widens
    only the bid side; the ask side is unchanged."""
    c = SpreadComposition(
        econ_floor_bps=1.0,
        negative_expectancy_dampen_bid_bps=2.5,
    )
    # bid total = econ_floor + dampen = 1.0 + 2.5 = 3.5
    assert c.total_bid_bps_uncapped() == pytest.approx(3.5)
    # ask total = econ_floor only = 1.0
    assert c.total_ask_bps_uncapped() == pytest.approx(1.0)


def test_composition_to_dict_includes_dampen_fields() -> None:
    """The ``to_dict()`` payload (which goes into ``quote_decisions``
    and the dashboard publisher) carries the new fields. 4C.2.c —
    the audit trail entry IS the new keys in this dict."""
    c = SpreadComposition(
        negative_expectancy_dampen_bid_bps=1.0,
        negative_expectancy_dampen_ask_bps=2.0,
    )
    d = c.to_dict()
    assert d["negative_expectancy_dampen_bid_bps"] == pytest.approx(1.0)
    assert d["negative_expectancy_dampen_ask_bps"] == pytest.approx(2.0)


def test_composition_with_caps_after_dampen_addition() -> None:
    """After dataclass.replace() adds dampen widening, the
    ``capped_at_max_*`` flag is recomputed via ``with_caps``. When
    the dampen widening pushes the total over the cap, the flag
    flips to True. This is what the bot's wire-up does each tick."""
    from dataclasses import replace as _replace

    c0 = SpreadComposition(econ_floor_bps=5.0)
    c1 = _replace(
        c0, negative_expectancy_dampen_bid_bps=100.0
    ).with_caps(max_bps=10.0)
    # 5 + 100 = 105 > 10 → capped
    assert c1.capped_at_max_bid is True
    # Ask side untouched (no dampen on ask) → not capped
    assert c1.capped_at_max_ask is False
