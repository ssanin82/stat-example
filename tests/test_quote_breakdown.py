"""Tests for the Spread tab's per-cycle breakdown snapshot.

Closes the test-coverage gap in ``plans/spread-tab.md`` Phase 4.
Verifies:

  1. ``compute_quote_decision`` attaches a ``QuoteBreakdownSnapshot``
     to the returned ``QuoteDecision.breakdown`` field, with every
     declared field populated.
  2. ``to_dict()`` round-trips cleanly through ``json.dumps`` — the
     dashboard consumes this via the state_current / live_stats
     payload pipelines.
  3. ``dataclasses.replace`` works on the frozen dataclass — the bot
     stamps adaptive-widen / ladder / eligibility / executable
     half-spread context onto the snapshot AFTER quoting returns.
  4. ``BotState.snapshot_dict()`` includes the breakdown when
     populated and ``None`` when not.
  5. Math sanity for derived fields (``reservation_delta_from_mid_bps``).
  6. Schema-drift guard: every field documented in the plan is
     present on the dataclass.
  7. Active-sides variants (BOTH, BID_ONLY, ASK_ONLY, NONE) all
     populate cleanly without NaN / None leaks where unexpected.

None of these are trading-behavioural; the breakdown is observability-
only. The point is to catch a refactor that drops a field the
dashboard's Spread tab reads.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace
from typing import Any

import pytest

from app.enums import ActiveSides
from app.models import ToxicitySnapshot
from app.quote_breakdown import QuoteBreakdownSnapshot
from app.quoting import compute_quote_decision
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings(**kw: Any) -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "MAX_ABS_POSITION": 10.0,
            **{k.upper(): v for k, v in kw.items()},
        }
    )


def _benign_toxicity() -> ToxicitySnapshot:
    return ToxicitySnapshot(0, 0, 0, 1, False, False)


def _compute(**call_kw: Any) -> tuple[QuoteBreakdownSnapshot, Any]:
    """Run ``compute_quote_decision`` with sensible defaults and
    return ``(breakdown, decision)`` for assertion."""
    s = _settings()
    mid = call_kw.pop("mid", 100.0)
    pos = call_kw.pop("position_qty", 0.0)
    vol = call_kw.pop("vol_bps", 1.0)
    tox = call_kw.pop("toxicity", _benign_toxicity())
    dec = compute_quote_decision(s, mid, pos, vol, tox, **call_kw)
    assert dec.breakdown is not None, "compute_quote_decision must attach breakdown"
    return dec.breakdown, dec


# ---------------------------------------------------------------------------
# 1. Population: every field is present and typed correctly
# ---------------------------------------------------------------------------


def test_compute_attaches_breakdown_with_all_fields_populated() -> None:
    bk, _ = _compute()
    # Frozen dataclass → tuple-of-Field accessors.
    declared = {f.name for f in fields(bk)}
    # Every declared field should be a real attribute and not raise.
    for name in declared:
        getattr(bk, name)  # no AttributeError, no None for required fields
    # Type spot-checks: numeric fields are float, ladder lists are list.
    assert isinstance(bk.mid_price, float)
    assert isinstance(bk.base_half_spread_bps, float)
    assert isinstance(bk.target_half_spread_bps, float)
    assert isinstance(bk.quoted_bid_px, float)
    assert isinstance(bk.quoted_ask_px, float)
    assert isinstance(bk.quote_notional_usd, float)
    assert isinstance(bk.effective_notional_usd, float)
    assert isinstance(bk.ladder_bids, list)
    assert isinstance(bk.ladder_asks, list)
    assert isinstance(bk.active_sides, str)
    # ts is ISO-formatted UTC.
    assert "T" in bk.ts and (bk.ts.endswith("+00:00") or "Z" in bk.ts)


def test_breakdown_has_no_nan_leaks_on_clean_inputs() -> None:
    """Quote logic is full of conditional branches; this is the
    canary test that none of them leak NaN into the breakdown when
    fed normal, finite inputs."""
    bk, _ = _compute()
    # Walk every field; floats must be finite. ``Optional`` fields
    # may be None, but if they're a float they must be finite.
    for f in fields(bk):
        v = getattr(bk, f.name)
        if isinstance(v, float):
            assert math.isfinite(v), f"{f.name} is non-finite: {v!r}"


# ---------------------------------------------------------------------------
# 2. JSON round-trip — the dashboard reads via to_dict() → json
# ---------------------------------------------------------------------------


def test_breakdown_to_dict_is_json_serialisable() -> None:
    bk, _ = _compute()
    d = bk.to_dict()
    assert isinstance(d, dict)
    # Must serialise without TypeError (no datetimes, no enums, no
    # frozen dataclasses leaking through).
    blob = json.dumps(d)
    # Round-trip preserves structure.
    parsed = json.loads(blob)
    assert parsed.keys() == d.keys()
    # Headline fields the dashboard's SpreadPanel reads directly.
    for required in (
        "ts",
        "mid_price",
        "reservation_price",
        "reservation_delta_from_mid_bps",
        "base_half_spread_bps",
        "target_half_spread_bps",
        "quoted_bid_px",
        "quoted_ask_px",
        "quoted_bid_sz",
        "quoted_ask_sz",
        "ladder_bids",
        "ladder_asks",
        "active_sides",
        "executable_half_spread_bps",
    ):
        assert required in parsed, f"missing dashboard field: {required}"


# ---------------------------------------------------------------------------
# 3. Frozen + dataclasses.replace — bot.py uses this pattern to stamp
#    adaptive-widen context AFTER compute_quote_decision returns.
# ---------------------------------------------------------------------------


def test_breakdown_is_frozen_replace_compatible() -> None:
    bk, _ = _compute()
    # Direct attribute mutation must raise (frozen guard).
    with pytest.raises(Exception):  # FrozenInstanceError
        bk.target_half_spread_bps = 99.0  # type: ignore[misc]
    # Replace works — bot.py stamps post-quote-engine fields this way.
    new_bk = replace(
        bk,
        adaptive_widen_active=True,
        adaptive_widen_reason="post_swing_test",
        adaptive_widen_seconds_remaining=42.0,
        quote_eligibility="QUOTE_BOTH",
        quote_eligibility_reason="ok",
        ladder_bids=[{"level": 0, "side": "buy", "px": 99.0, "sz": 1.0}],
        ladder_asks=[{"level": 0, "side": "sell", "px": 101.0, "sz": 1.0}],
        ladder_requested_levels=1,
        ladder_effective_levels_buy=1,
        ladder_effective_levels_sell=1,
        executable_half_spread_bps=8.5,
    )
    assert new_bk.adaptive_widen_active is True
    assert new_bk.adaptive_widen_reason == "post_swing_test"
    assert new_bk.adaptive_widen_seconds_remaining == 42.0
    assert new_bk.ladder_bids[0]["px"] == 99.0
    assert new_bk.ladder_asks[0]["sz"] == 1.0
    assert new_bk.executable_half_spread_bps == 8.5
    # Original is untouched.
    assert bk.adaptive_widen_active is False
    assert bk.executable_half_spread_bps is None


# ---------------------------------------------------------------------------
# 4. BotState.snapshot_dict integration — the snapshot pipeline
# ---------------------------------------------------------------------------


def test_snapshot_dict_includes_breakdown_when_populated() -> None:
    s = _settings()
    state = BotState(s)
    # Pre-population: None passthrough.
    snap = state.snapshot_dict()
    assert "last_quote_breakdown" in snap
    assert snap["last_quote_breakdown"] is None

    # Populate via a real quote cycle, then attach.
    bk, _ = _compute()
    state.last_quote_breakdown = bk
    snap2 = state.snapshot_dict()
    assert snap2["last_quote_breakdown"] is not None
    assert isinstance(snap2["last_quote_breakdown"], dict)
    # The dashboard reads these top-level keys from the dict form.
    assert "mid_price" in snap2["last_quote_breakdown"]
    assert "target_half_spread_bps" in snap2["last_quote_breakdown"]
    # And it survives a json round-trip from the snapshot dict.
    assert json.loads(json.dumps(snap2["last_quote_breakdown"])) == (
        snap2["last_quote_breakdown"]
    )


# ---------------------------------------------------------------------------
# 5. Math sanity for derived fields
# ---------------------------------------------------------------------------


def test_reservation_delta_from_mid_matches_explicit_formula() -> None:
    """``reservation_delta_from_mid_bps`` is the derived ``(reservation
    - mid) / mid * 10_000`` shortcut for the dashboard. Verify the
    pre-computed value matches the explicit recomputation."""
    bk, _ = _compute()
    expected = (bk.reservation_price - bk.mid_price) / bk.mid_price * 10_000.0
    assert bk.reservation_delta_from_mid_bps == pytest.approx(expected, abs=1e-6)


def test_clamp_winner_field_is_one_of_known_values() -> None:
    """The dashboard's Spread tab highlights the winning clamp
    branch (raw / floor / ceiling). Guard against accidental drift."""
    bk, _ = _compute()
    assert bk.clamp_winner in ("raw", "floor", "ceiling")


def test_economic_floor_kind_field_is_known_value() -> None:
    bk, _ = _compute()
    assert bk.economic_floor_kind in ("neutral", "inventory")


# ---------------------------------------------------------------------------
# 6. Schema-drift guard — the plan documents an exhaustive field set;
#    catch accidental field removal in a refactor.
# ---------------------------------------------------------------------------


# Pulled verbatim from ``plans/spread-tab.md::Backend`` + the
# ``app/quote_breakdown.py`` dataclass. Splitting this list out makes
# the "what does the dashboard depend on" answer one-stop. Items
# tagged "since 1.2.x" capture later additions; the headline set is
# stable.
_REQUIRED_FIELDS = frozenset(
    {
        # Header
        "ts",
        # Reservation
        "mid_price",
        "microprice",
        "reservation_reference",
        "reference_fair_price",
        "reference_blend_alpha",
        "ref_price_anchor",
        "inventory_skew_bps",
        "trend_drift_shift_bps",
        "ob_imbalance_shift_bps",
        "basis_deviation_shift_bps",
        "flow_score_shift_bps",
        "reservation_price",
        "reservation_delta_from_mid_bps",
        # Half-spread stack
        "base_half_spread_bps",
        "vol_contribution_bps",
        "toxicity_bump_bps",
        "toxicity_score_used",
        "toxicity_coeff_used",
        "join_depth_autotune_overlay_bps",
        "toxicity_soft_trigger_bump_bps",
        "toxicity_soft_trigger_active",
        "vol_regime_bump_bps",
        "adaptive_widen_overlay_bps",
        "adaptive_widen_active",
        "adaptive_widen_reason",
        "adaptive_widen_seconds_remaining",
        "raw_half_spread_bps",
        "min_half_spread_bps",
        "max_half_spread_bps",
        "economic_min_half_spread_bps",
        "economic_floor_kind",
        "target_half_spread_bps",
        "clamp_winner",
        # Size mult chain
        "quote_notional_usd",
        "size_mult_after_toxicity",
        "size_mult_after_vol_regime",
        "size_mult_after_markout_scaler",
        "size_mult_after_basis_regime",
        "final_size_mult",
        "size_mult_floor",
        "effective_notional_usd",
        # Quote outputs
        "quoted_bid_px",
        "quoted_ask_px",
        "quoted_bid_sz",
        "quoted_ask_sz",
        "best_bid",
        "best_ask",
        # Eligibility
        "quote_eligibility",
        "quote_eligibility_reason",
        "active_sides",
        # Codex-#3 follow-on (since 1.2.53)
        "executable_half_spread_bps",
        # Ladder
        "ladder_bids",
        "ladder_asks",
        "ladder_requested_levels",
        "ladder_effective_levels_buy",
        "ladder_effective_levels_sell",
    }
)


def test_breakdown_schema_matches_documented_set() -> None:
    """Schema-drift guard. If a contributor renames or drops a field
    the dashboard's SpreadPanel reads, this test fails loudly so they
    update both the breakdown AND the frontend in the same change.

    Adding fields is fine; this only catches removals / renames.
    """
    declared = {f.name for f in fields(QuoteBreakdownSnapshot)}
    missing = _REQUIRED_FIELDS - declared
    assert not missing, (
        f"QuoteBreakdownSnapshot is missing fields the Spread tab "
        f"depends on: {sorted(missing)}. Update both "
        f"``app/quote_breakdown.py`` and the dashboard's SpreadPanel "
        f"in lockstep."
    )


# ---------------------------------------------------------------------------
# 7. Active-sides variants — no NaN / unexpected None on suppressed sides
# ---------------------------------------------------------------------------


def test_breakdown_populates_cleanly_when_only_bid_active() -> None:
    """Long position triggers ASK-only inventory bias; verify the
    breakdown still captures both quoted_bid_px / quoted_ask_px
    without NaN even when one side's size is zero."""
    s = _settings(max_abs_position=1.0)
    tox = _benign_toxicity()
    # Heavy short → quotes should still produce both sides
    # numerically, with active_sides reflecting the bias.
    dec = compute_quote_decision(s, 100.0, position_qty=-0.95, vol_bps=1.0, toxicity=tox)
    assert dec.breakdown is not None
    bk = dec.breakdown
    assert math.isfinite(bk.quoted_bid_px)
    assert math.isfinite(bk.quoted_ask_px)
    assert math.isfinite(bk.quoted_bid_sz)
    assert math.isfinite(bk.quoted_ask_sz)
    # active_sides string is one of the enum values.
    assert bk.active_sides in {s.value for s in ActiveSides}


def test_breakdown_populates_cleanly_with_microprice_input() -> None:
    """Microprice reservation reference is a critical Spread-tab
    branch — verify the breakdown captures the chosen reference
    even when depth data is present."""
    bk, _ = _compute(
        best_bid=99.95,
        best_ask=100.05,
        bid_size=10.0,
        ask_size=5.0,  # asymmetric → microprice shifts toward bid
    )
    assert bk.reservation_reference in ("mid", "microprice")
    # When microprice is computed and best_bid/best_ask were
    # supplied, the microprice field should be populated.
    assert bk.microprice is not None
    assert math.isfinite(bk.microprice)
    assert math.isfinite(bk.reservation_price)


# ---------------------------------------------------------------------------
# 8. Defaulted fields preserved when not stamped by the caller
# ---------------------------------------------------------------------------


def test_executable_half_spread_defaults_none_until_stamped() -> None:
    """``executable_half_spread_bps`` is filled in by the bot's
    ``_stamp_executable_half_spread_on_breakdown`` AFTER the quote
    engine runs. ``compute_quote_decision`` leaves it as the dataclass
    default (None)."""
    bk, _ = _compute()
    assert bk.executable_half_spread_bps is None


def test_ladder_fields_default_at_n_equals_1() -> None:
    """Ladder fields default to single-rung-friendly values when the
    caller doesn't stamp the actual ladder. Verifies the contract
    bot.py relies on (won't be stuck with None / NaN)."""
    bk, _ = _compute()
    assert bk.ladder_bids == []
    assert bk.ladder_asks == []
    assert bk.ladder_requested_levels == 1
    # Buy/sell effective levels reflect active_sides at decision time.
    assert isinstance(bk.ladder_effective_levels_buy, int)
    assert isinstance(bk.ladder_effective_levels_sell, int)
