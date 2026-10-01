"""Phase 4G.13 (v1.4.228) — structural-bias auto-throttle gate.

Driving snapshot: ``v1.4.219-260521-214353-prod.okx.ton.usdt.perp``.
4 SF events in 36 min. SF#11183 fired during a 285 s NORMAL window
where the forward classifier was blind to slow-building adverse drift.

The gate reads session-cumulative ``inventory_exec_bias`` suppression
counts per side and forces one-sided REDUCING eligibility when the
ratio crosses 5× (matching v1.4.220 Bias card's RED tier). Mirrors
the bias-direction logic of the Bias card:

  * BID-suppressed >> ASK-suppressed → LONG bias → QUOTE_SELL_ONLY
  * ASK-suppressed >> BID-suppressed → SHORT bias → QUOTE_BUY_ONLY

Composes via ``more_restrictive`` with the existing eligibility chain;
never widens, only narrows.

Test coverage:
  * disabled flag → no-op
  * below min-samples → no-op (avoid false-positive throttle on early ticks)
  * below ratio threshold → no-op
  * LONG bias above threshold → QUOTE_SELL_ONLY
  * SHORT bias above threshold → QUOTE_BUY_ONLY
  * already SELL_ONLY + LONG bias → still SELL_ONLY (intersection no-op)
  * already BUY_ONLY + LONG bias → HOLD_ALL (incompatible directions)
  * already HOLD_ALL → stays HOLD_ALL (gate doesn't widen)
  * fire counter increments per tick when throttled
  * direction + active flag set correctly on state
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.enums import QuoteEligibility
from app.quote_eligibility import QuoteEligibilityResult
from tests.settings_helpers import UnitTestSettings


def _bot(**overrides):
    """Build a minimal bot scaffold that the gate can mutate. Avoids the
    full bot-start path (DB, WS, etc.) by patching just enough state."""
    from app.state import BotState

    path = Path(tempfile.gettempdir()) / f"mm_4g13_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "STRUCTURAL_BIAS_AUTO_THROTTLE_ENABLED": True,
        "STRUCTURAL_BIAS_AUTO_THROTTLE_RATIO_THRESHOLD": 5.0,
        "STRUCTURAL_BIAS_AUTO_THROTTLE_MIN_SAMPLES": 20,
    }
    base.update(overrides)
    s = UnitTestSettings.model_validate(base)
    st = BotState(settings=s)

    # Test scaffold — minimal bot-like object the gate method needs.
    class _Bot:
        _state = st
        _settings = s

    # Bind the method to our scaffold.
    from app.bot import Bot

    _Bot._apply_structural_bias_throttle_gate = Bot._apply_structural_bias_throttle_gate
    return _Bot(), st, s


def _set_counts(state, bid: int, ask: int) -> None:
    """Seed the suppression counts inside the quote_quality module."""
    counts = state.quote_quality._suppression_counts
    counts["engine:inventory_exec_bias_bid"] = bid
    counts["engine:inventory_exec_bias_ask"] = ask


def _set_position(state, qty: float) -> None:
    """v1.5.151 BUG-029 — position-aware gate needs a current position.
    Seeds the PositionSnapshot so the gate's position-direction check
    sees the requested qty."""
    from app.models import PositionSnapshot
    state.position = PositionSnapshot(
        symbol="ETH",
        position_qty=float(qty),
        avg_entry_price=100.0 if qty != 0 else None,
        mark_price=100.0,
        position_notional=float(qty) * 100.0,
        unrealized_pnl_usd=0.0,
    )


def _baseline(elig=QuoteEligibility.QUOTE_BOTH, reason="ok"):
    """Construct a minimal ``QuoteEligibilityResult`` for tests. The
    dataclass has many required fields (book-staleness, jumps, etc.)
    that aren't relevant to the structural-bias gate's logic — all
    set to None / False here as inert placeholders."""
    return QuoteEligibilityResult(
        eligibility=elig,
        reason=reason,
        seconds_since_last_public_book_update=None,
        effective_staleness_ms=None,
        market_data_gap_p95_ms=None,
        market_data_gap_median_ms=None,
        mid_return_100ms_bps=None,
        mid_return_250ms_bps=None,
        mid_return_500ms_bps=None,
        jump_100ms_bps=None,
        jump_250ms_bps=None,
        jump_500ms_bps=None,
        in_cooldown=False,
    )


# -------------------------------------------------------------------
# Disabled / below-floor cases
# -------------------------------------------------------------------


def test_disabled_flag_no_op() -> None:
    bot, state, _ = _bot(STRUCTURAL_BIAS_AUTO_THROTTLE_ENABLED=False)
    _set_counts(state, bid=200, ask=10)  # 20× — would normally throttle
    eff_q = _baseline()
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False
    assert state.structural_bias_throttle_fire_count_total == 0


def test_below_min_samples_no_op() -> None:
    """Floor prevents false-positive on early-session ticks."""
    bot, state, _ = _bot()
    _set_counts(state, bid=15, ask=0)  # ratio = inf but total = 15 < 20 floor
    eff_q = _baseline()
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False


def test_below_ratio_threshold_no_op() -> None:
    bot, state, _ = _bot()
    _set_counts(state, bid=60, ask=20)  # 3× — below 5× threshold
    eff_q = _baseline()
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False


# -------------------------------------------------------------------
# Engagement cases
# -------------------------------------------------------------------


def test_long_bias_above_threshold_forces_sell_only() -> None:
    """BID-suppressed >> ASK-suppressed → LONG → can only sell.
    v1.5.151 BUG-029: position must be ACTUALLY LONG for the gate
    to fire (was: fired on session-cumulative bias regardless of
    current position; produced the wrong-direction force during
    rising trends — see v1.5.150-260525-213511 snapshot)."""
    bot, state, _ = _bot()
    _set_counts(state, bid=200, ask=20)  # 10× LONG
    _set_position(state, +3.0)  # v1.5.151 — bot is actually long
    eff_q = _baseline()
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.QUOTE_SELL_ONLY
    assert "structural_bias_throttle:LONG_10.0x" in result.reason
    assert state.structural_bias_throttle_active_last_tick is True
    assert state.structural_bias_throttle_direction == "LONG"
    assert state.structural_bias_throttle_fire_count_total == 1


def test_short_bias_above_threshold_forces_buy_only() -> None:
    """ASK-suppressed >> BID-suppressed → SHORT → can only buy.
    v1.5.151 BUG-029: position must be ACTUALLY SHORT for the gate
    to fire."""
    bot, state, _ = _bot()
    _set_counts(state, bid=20, ask=200)  # 10× SHORT
    _set_position(state, -3.0)  # v1.5.151 — bot is actually short
    eff_q = _baseline()
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.QUOTE_BUY_ONLY
    assert "structural_bias_throttle:SHORT_10.0x" in result.reason
    assert state.structural_bias_throttle_direction == "SHORT"


def test_v1_4_219_snapshot_pattern_engages_short_throttle() -> None:
    """Replay the exact bias counts from the v1.4.219-260521-214353
    snapshot. BID=94, ASK=237 → 2.5× — BELOW the 5× threshold so
    the gate should NOT have fired in that session. This pins
    the "5× isn't too sensitive for routine sessions" contract."""
    bot, state, _ = _bot()
    _set_counts(state, bid=94, ask=237)
    eff_q = _baseline()
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    # Ratio = 237 / 94 = 2.52 — below threshold 5.0
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH


def test_v1_4_187_snapshot_pattern_engages_long_throttle() -> None:
    """Replay the v1.4.187 session pattern. BID=1358, ASK=27 →
    ratio = 50.3× → SHOULD fire LONG throttle. This was the
    session where the absence of this gate let the bot bleed
    structurally for 41 minutes.

    v1.5.151 BUG-029: pin the test to the original intent — bot
    is ACTUALLY long-trapped (position > 0) and the throttle
    forces SELL_ONLY to unwind. Without the position assertion,
    the test would pass spuriously under the old logic against a
    flat bot."""
    bot, state, _ = _bot()
    _set_counts(state, bid=1358, ask=27)
    _set_position(state, +5.0)  # v1.5.151 — bot is actually long-trapped
    eff_q = _baseline()
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.QUOTE_SELL_ONLY
    assert state.structural_bias_throttle_direction == "LONG"


# -------------------------------------------------------------------
# Composition with existing eligibility
# -------------------------------------------------------------------


def test_already_sell_only_plus_long_bias_stays_sell_only() -> None:
    """No-op intersection."""
    bot, state, _ = _bot()
    _set_counts(state, bid=200, ask=10)
    _set_position(state, +3.0)  # v1.5.151 — bot actually long
    eff_q = _baseline(elig=QuoteEligibility.QUOTE_SELL_ONLY, reason="upstream_sell_only")
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.QUOTE_SELL_ONLY


def test_already_buy_only_plus_long_bias_collapses_to_hold_all() -> None:
    """BUY_ONLY ∩ SELL_ONLY = ∅ = HOLD_ALL. The gate forces
    SELL_ONLY because of LONG lean; existing eligibility is
    BUY_ONLY. Intersection = HOLD_ALL. Fail-safe direction —
    bot stops quoting entirely rather than picking a side that
    violates one of the gates."""
    bot, state, _ = _bot()
    _set_counts(state, bid=200, ask=10)
    _set_position(state, +3.0)  # v1.5.151 — bot actually long
    eff_q = _baseline(elig=QuoteEligibility.QUOTE_BUY_ONLY, reason="upstream_buy_only")
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.HOLD_ALL


def test_already_hold_all_stays_hold_all() -> None:
    """Gate never widens — HOLD_ALL stays HOLD_ALL."""
    bot, state, _ = _bot()
    _set_counts(state, bid=200, ask=10)
    _set_position(state, +3.0)  # v1.5.151 — bot actually long
    eff_q = _baseline(elig=QuoteEligibility.HOLD_ALL, reason="upstream_hold")
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.HOLD_ALL


# -------------------------------------------------------------------
# State tracking
# -------------------------------------------------------------------


def test_fire_counter_increments_per_tick() -> None:
    """Per-tick counter (not per-engagement) so the session counter
    reflects cumulative throttled time."""
    bot, state, _ = _bot()
    _set_counts(state, bid=200, ask=10)
    _set_position(state, +3.0)  # v1.5.151 — bot actually long
    eff_q = _baseline()
    bot._apply_structural_bias_throttle_gate(eff_q)
    bot._apply_structural_bias_throttle_gate(eff_q)
    bot._apply_structural_bias_throttle_gate(eff_q)
    assert state.structural_bias_throttle_fire_count_total == 3


def test_latched_flag_clears_when_gate_inactive() -> None:
    """When a tick doesn't fire the gate, the latched-active flag
    clears so the snapshot reflects last-tick truth."""
    bot, state, _ = _bot()
    # First tick: fire (200 / 10 = 20×).
    _set_counts(state, bid=200, ask=10)
    _set_position(state, +3.0)  # v1.5.151 — bot actually long
    bot._apply_structural_bias_throttle_gate(_baseline())
    assert state.structural_bias_throttle_active_last_tick is True

    # Reset counts to below threshold (counts can shrink in real life
    # only via session restart, but the gate must clear its flag
    # regardless — test by setting them down).
    _set_counts(state, bid=60, ask=20)  # 3× — below threshold
    bot._apply_structural_bias_throttle_gate(_baseline())
    assert state.structural_bias_throttle_active_last_tick is False
    assert state.structural_bias_throttle_direction is None


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------


def test_zero_counts_safe() -> None:
    """Total = 0 → no-op (covered by min_samples but defensive)."""
    bot, state, _ = _bot()
    eff_q = _baseline()
    result = bot._apply_structural_bias_throttle_gate(eff_q)
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH


def test_one_sided_zero_safe() -> None:
    """ASK = 0 with BID ≥ floor: ratio = BID / max(1, 0) = BID / 1
    = BID. Above 5× when BID ≥ 5. Combined with min_samples = 20,
    BID ≥ 20 → ratio ≥ 20 → throttle."""
    bot, state, _ = _bot()
    _set_counts(state, bid=25, ask=0)
    _set_position(state, +3.0)  # v1.5.151 — bot actually long
    result = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result.eligibility == QuoteEligibility.QUOTE_SELL_ONLY
    assert state.structural_bias_throttle_direction == "LONG"


def test_threshold_boundary_exactly_at_5x_fires() -> None:
    """Ratio = exactly 5.0 → engages (>= threshold)."""
    bot, state, _ = _bot()
    _set_counts(state, bid=100, ask=20)
    _set_position(state, +3.0)  # v1.5.151 — bot actually long
    result = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result.eligibility == QuoteEligibility.QUOTE_SELL_ONLY


def test_threshold_just_below_5x_does_not_fire() -> None:
    """Ratio = 4.99 → does not engage."""
    bot, state, _ = _bot()
    _set_counts(state, bid=499, ask=100)  # 4.99
    _set_position(state, +3.0)  # v1.5.151 — bot actually long; would fire if ratio met threshold
    result = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH


# -------------------------------------------------------------------
# v1.5.151 BUG-029 — position-aware gate
#
# These tests pin the new position-direction check. The OLD logic
# fired purely on the session-cumulative ratio (no position check).
# In a strongly trending market that ratio builds up via correct
# defensive suppressions while the bot's CURRENT position is flat
# or already in the opposite direction — the gate then forces the
# WRONG-DIRECTION eligibility (e.g., SELL_ONLY in an uptrend while
# the bot is flat), which is the exact mechanism observed in the
# v1.5.150-260525-213511 snapshot.
# -------------------------------------------------------------------


def test_flat_position_dormant_even_with_long_bias() -> None:
    """BUG-029 case 1: bias ratio above threshold but bot is FLAT.
    The bias is historical — the bot already unwound. Gate must
    stay dormant; no SELL_ONLY force on a flat bot."""
    bot, state, _ = _bot()
    _set_counts(state, bid=200, ask=10)  # 20× LONG
    _set_position(state, 0.0)  # FLAT
    result = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False
    assert state.structural_bias_throttle_direction is None
    assert state.structural_bias_throttle_fire_count_total == 0


def test_flat_position_dormant_even_with_short_bias() -> None:
    """BUG-029 symmetric: SHORT bias on flat bot also dormant."""
    bot, state, _ = _bot()
    _set_counts(state, bid=10, ask=200)
    _set_position(state, 0.0)  # FLAT
    result = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False


def test_short_position_dormant_with_long_bias() -> None:
    """BUG-029 case 2: LONG bias ratio + bot is SHORT. The bias is
    HISTORICAL (e.g., bot was long earlier, unwound, flipped short).
    Forcing SELL_ONLY here would deepen the short — exactly the
    wrong direction. Gate must stay dormant."""
    bot, state, _ = _bot()
    _set_counts(state, bid=200, ask=10)  # 20× LONG bias
    _set_position(state, -3.0)  # but bot is short
    result = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False


def test_long_position_dormant_with_short_bias() -> None:
    """BUG-029 mirror of case 2: SHORT bias ratio + bot is LONG.
    Gate stays dormant — bias direction doesn't match position."""
    bot, state, _ = _bot()
    _set_counts(state, bid=10, ask=200)
    _set_position(state, +3.0)  # but bot is long
    result = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False


def test_v1_5_150_260525_213511_snapshot_reproducer() -> None:
    """Direct reproducer of the v1.5.150-260525-213511 wedge pattern.
    Session counts: bid_suppressed=35, ask_suppressed=7 (ratio 5.0×,
    just at threshold). Bot's CURRENT position is 0 or briefly -3
    short while in a strong uptrend. The OLD logic fired LONG → SELL_ONLY
    every tick (91,501 fire-count) — forcing the bot to keep
    quoting the wrong side. The fix: dormant on flat/short bot."""
    bot, state, _ = _bot()
    _set_counts(state, bid=35, ask=7)  # ratio = 5.0× exactly

    # Sub-case (a): bot flat — gate dormant.
    _set_position(state, 0.0)
    result_flat = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result_flat.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False

    # Sub-case (b): bot short — gate still dormant (long bias on short
    # bot makes no sense; forcing SELL_ONLY would deepen the short).
    _set_position(state, -3.0)
    result_short = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result_short.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False

    # Sub-case (c): bot actually long — NOW the gate fires (the
    # original design intent: reduce a long-trap via SELL_ONLY).
    _set_position(state, +3.0)
    result_long = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result_long.eligibility == QuoteEligibility.QUOTE_SELL_ONLY
    assert state.structural_bias_throttle_active_last_tick is True
    assert state.structural_bias_throttle_direction == "LONG"


def test_position_deadzone_near_zero_treated_as_flat() -> None:
    """Tiny residual positions (floating-point fuzz, sub-tick
    rounding) treated as flat — don't fire the throttle on
    1e-12-level inventory artifacts."""
    bot, state, _ = _bot()
    _set_counts(state, bid=200, ask=10)
    _set_position(state, 1e-12)  # numerical residual
    result = bot._apply_structural_bias_throttle_gate(_baseline())
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH
    assert state.structural_bias_throttle_active_last_tick is False
