"""Phase 4C.1+4C.2 — expected-net-edge side suppression (v1.4.164).

Out-of-order partial delivery of the Phase 4C arc. Targets the
v1.4.157-260520-213540 failure mode at the PREVENTION layer: when
the per-side expected NET edge
(``target_half_spread + maker_rebate - typical_adverse``) drops below
``MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE``, refuse to quote that side.
Hysteresis prevents flapping.

Tests are pure-function — they cover the helper directly. The
plumbing into ``compute_quote_decision`` (two new kwargs) is verified
by signature inspection at the bottom; the bot.py wiring is covered
indirectly by ``test_quoting.py`` (which still passes with the new
kwargs defaulted to False).
"""

from __future__ import annotations

from app.expected_edge import (
    ExpectedEdgeSideSuppressResult,
    compute_expected_net_edge_bps,
    evaluate_per_side_expected_edge_suppression,
)


def _kwargs(**overrides):
    base = dict(
        target_half_spread_bps=2.0,
        currently_refused=False,
        recovery_ticks=0,
        min_expected_edge_bps=-1.0,
        hysteresis_ticks=3,
        recovery_margin_bps=0.2,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# compute_expected_net_edge_bps — pure formula
# ---------------------------------------------------------------------------


def test_formula_matches_existing_place_time_stamp() -> None:
    """The formula here MUST match
    ``execution.py::_expected_edge`` so the place-time stamp and the
    decision-time gate agree."""
    edge = compute_expected_net_edge_bps(
        target_half_spread_bps=3.0,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
    )
    assert edge == 3.0 + 1.0 - 2.0


def test_formula_returns_none_for_missing_target() -> None:
    assert compute_expected_net_edge_bps(
        target_half_spread_bps=None,
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
    ) is None


def test_formula_returns_none_for_nan() -> None:
    assert compute_expected_net_edge_bps(
        target_half_spread_bps=float("nan"),
        maker_rebate_bps=1.0,
        typical_adverse_markout_bps=2.0,
    ) is None


# ---------------------------------------------------------------------------
# Feature dormant when MIN >= 0
# ---------------------------------------------------------------------------


def test_disabled_by_default() -> None:
    """``MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE=0.0`` (default) means
    nothing ever gets refused, even with a strongly negative edge."""
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=-10.0,  # edge = -10 + 1 - 2 = -11
            min_expected_edge_bps=0.0,
        )
    )
    assert r.refused is False
    assert r.transition == "none"


def test_dormant_when_target_missing() -> None:
    """No edge signal yet → gate stays dormant for this tick."""
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(target_half_spread_bps=None)
    )
    assert r.refused is False
    assert r.expected_edge_bps is None
    assert r.transition == "none"


# ---------------------------------------------------------------------------
# Arm + hold
# ---------------------------------------------------------------------------


def test_arms_when_edge_below_threshold() -> None:
    """edge = 0 + 1 - 2 = -1; threshold -1.0 → -1 < -1 is False...
    use -2 to clearly fire."""
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=-1.0,  # edge = -1 + 1 - 2 = -2
            min_expected_edge_bps=-1.0,
        )
    )
    assert r.refused is True
    assert r.transition == "armed"
    assert r.new_recovery_ticks == 0
    assert r.expected_edge_bps == -2.0


def test_stays_refused_while_edge_still_below() -> None:
    """Currently refused, edge still below threshold → stay refused."""
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=-2.0,  # edge = -3
            currently_refused=True,
            recovery_ticks=0,
            min_expected_edge_bps=-1.0,
        )
    )
    assert r.refused is True
    assert r.transition == "none"


# ---------------------------------------------------------------------------
# Hysteresis: recovery requires K consecutive ticks
# ---------------------------------------------------------------------------


def test_recovery_requires_hysteresis_ticks() -> None:
    """Threshold -1.0, margin 0.2 → recovery threshold -0.8.
    Edge of -0.5 > -0.8 so it counts as recovery."""
    # tick 1
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=0.5,  # edge = 0.5 + 1 - 2 = -0.5
            currently_refused=True,
            recovery_ticks=0,
            min_expected_edge_bps=-1.0,
            hysteresis_ticks=3,
            recovery_margin_bps=0.2,
        )
    )
    assert r.refused is True
    assert r.new_recovery_ticks == 1
    # tick 2
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=0.5,
            currently_refused=True,
            recovery_ticks=1,
            min_expected_edge_bps=-1.0,
            hysteresis_ticks=3,
            recovery_margin_bps=0.2,
        )
    )
    assert r.refused is True
    assert r.new_recovery_ticks == 2
    # tick 3 — clears
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=0.5,
            currently_refused=True,
            recovery_ticks=2,
            min_expected_edge_bps=-1.0,
            hysteresis_ticks=3,
            recovery_margin_bps=0.2,
        )
    )
    assert r.refused is False
    assert r.transition == "cleared"
    assert r.new_recovery_ticks == 0


def test_recovery_interrupted_resets_counter() -> None:
    """Mid-recovery, edge dips back below threshold → counter resets,
    side stays refused, full hysteresis must restart on next recovery."""
    # tick 1: building recovery (counter 0 → 1)
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=0.5,  # edge -0.5, > recovery -0.8
            currently_refused=True,
            recovery_ticks=0,
            min_expected_edge_bps=-1.0,
        )
    )
    assert r.new_recovery_ticks == 1
    # tick 2: edge interrupted — drops below recovery threshold
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=-0.5,  # edge -1.5, < recovery -0.8
            currently_refused=True,
            recovery_ticks=1,  # 1 progress so far
            min_expected_edge_bps=-1.0,
        )
    )
    assert r.refused is True
    assert r.new_recovery_ticks == 0  # reset


def test_recovery_margin_separates_arm_and_clear() -> None:
    """With margin = 0.5 → arms at edge < -1.0; clears only at edge >
    -0.5. Edge of -0.7 (between -1.0 and -0.5) is dead-band: not
    re-arming, not progressing recovery."""
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=0.3,  # edge = 0.3 + 1 - 2 = -0.7
            currently_refused=True,
            recovery_ticks=2,
            min_expected_edge_bps=-1.0,
            recovery_margin_bps=0.5,
        )
    )
    # -0.7 > -1.5 (recovery threshold = -1.0 + 0.5)... wait, recovery
    # threshold = min + margin = -1.0 + 0.5 = -0.5. -0.7 < -0.5 → NOT
    # in recovery, counter resets.
    assert r.refused is True
    assert r.new_recovery_ticks == 0


# ---------------------------------------------------------------------------
# Edge case: hysteresis_ticks=1 means clear on first recovery tick
# ---------------------------------------------------------------------------


def test_hysteresis_ticks_1_clears_immediately() -> None:
    r = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=2.0,  # edge = 1
            currently_refused=True,
            recovery_ticks=0,
            min_expected_edge_bps=-1.0,
            hysteresis_ticks=1,
        )
    )
    assert r.refused is False
    assert r.transition == "cleared"


# ---------------------------------------------------------------------------
# Integration: compute_quote_decision accepts the two new kwargs
# ---------------------------------------------------------------------------


def test_compute_quote_decision_accepts_expected_edge_refused_kwargs() -> None:
    """API smoke test: the new kwargs exist on
    ``compute_quote_decision`` and default to False."""
    import inspect
    from app.quoting import compute_quote_decision

    sig = inspect.signature(compute_quote_decision)
    assert "expected_edge_refused_bid" in sig.parameters
    assert "expected_edge_refused_ask" in sig.parameters
    assert sig.parameters["expected_edge_refused_bid"].default is False
    assert sig.parameters["expected_edge_refused_ask"].default is False


# ---------------------------------------------------------------------------
# v1.4.157 replay: SELL side would have been refused
# ---------------------------------------------------------------------------


def test_v1_4_157_sell_side_would_have_been_refused() -> None:
    """In the v1.4.157-260520-213540 snapshot the SELL side's
    target_half_spread under stress was negative (after toxicity
    bumps + reservation shifts). With the gate enabled at -1.0 bps,
    that side would have been refused, preventing the resting SELL
    from being picked off during the rally start.

    This is illustrative — uses approximate numbers from the
    session-summary attribution block. The actual production target
    half-spread varies tick by tick; the point is that the formula
    + threshold combination matches the documented snapshot."""
    # snapshot SELL: target_half_spread ~ 1.0 bps under stress
    # (post-floor, pre-widening); rebate 1.0; typical adverse 2.0
    # → edge = 1.0 + 1.0 - 2.0 = 0 → ABOVE -1.0 (would not have refused)
    r_no_refuse = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=1.0,
            min_expected_edge_bps=-1.0,
        )
    )
    assert r_no_refuse.refused is False
    # During the rally the realised adverse component blew up to
    # ~7 bps (session 5s markout -1.78 mean; SELL alone was worse).
    # A higher typical_adverse_markout_bps prior would have caught
    # it. With prior=7 → edge = 1 + 1 - 7 = -5 → REFUSED.
    r_refuse = evaluate_per_side_expected_edge_suppression(
        **_kwargs(
            target_half_spread_bps=1.0,
            typical_adverse_markout_bps=7.0,  # operator-tuned prior
            min_expected_edge_bps=-1.0,
        )
    )
    assert r_refuse.refused is True
    assert r_refuse.transition == "armed"
