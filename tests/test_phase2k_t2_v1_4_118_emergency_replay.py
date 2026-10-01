"""Phase 2K.T2 (v1.4.169) — cross-cutting integration test for the
v1.4.118 emergency state shape.

Background. The v1.4.118 emergency snapshot showed a session where
util sat near 60 % with the ``post_reduction_cooldown`` armed and
``quoting:soft_skew_long`` firing every quote tick — for 25 min the
bot's BID stayed suppressed even though the inventory imbalance was
slowly resolving via ASK fills, because the cooldown was on a fixed
60 s timer that re-armed every time a fresh reduction landed. The
fix shipped in v1.4.144 (Phase 2K.1) was the **favorable-exit
predicate**: the cooldown clears EARLY when the bot's actual
utilisation drops below
``POST_REDUCTION_COOLDOWN_CLEAR_UTIL_PCT`` (default 0.30), regardless
of how much of the fixed-cooldown budget remains.

This test is the **cross-cutting integration regression** that the
defense plan calls out as ``2K.T2``: synthesise the v1.4.118 shape
end-to-end, drop util via a simulated ASK fill, and assert the
cooldown clears in the very next eligibility evaluation — NOT 60 s
later. It exercises the integration of:

* ``BotState._note_inventory_reduction_for_cooldown`` (Phase 1D arming)
* ``BotState.last_inventory_reduction_at_mono`` / ``_suppressed_side``
  (Phase 1D state machine)
* ``BotState.position`` mutation (the simulated ASK fill)
* The favorable-exit predicate (Phase 2K.1: ``util < clear_util_pct``)
* The MAX-ceiling preservation (``elapsed < cooldown_seconds``)

The test reproduces the gate-evaluation contract that the bot's hot
path enforces (``bot.py`` near the ``post_reduction_cooldown`` block,
lines 3487-3580); the goal is to catch any regression where the
favorable-exit clearing fails to fire even though the predicate's
inputs are all favourable.
"""

from __future__ import annotations

from app.config import Settings
from app.enums import QuoteEligibility, Side
from app.models import PositionSnapshot
from app.state import BotState


# ---------------------------------------------------------------------------
# Pure replay of the cooldown clamp decision
# ---------------------------------------------------------------------------
#
# This mirrors the inline logic in bot.py around lines 3503-3543. We
# replicate the decision here so the integration test catches any
# behavioural drift between the inline path and the docstrings of the
# pieces it composes. If bot.py ever extracts the logic into a helper
# function, this test should switch to call that helper directly.


def _evaluate_post_reduction_clamp(
    state: BotState,
    settings: Settings,
    *,
    now_mono: float,
    mode_is_normal: bool,
) -> dict:
    """Replays the bot's per-tick decision. Returns a dict with keys:

    * ``would_clamp`` (bool) — True iff the eligibility would be
      narrowed to the suppressed side this tick.
    * ``ceiling_satisfied`` (bool)
    * ``util_above_threshold`` (bool)
    * ``suppressed_side`` (QuoteEligibility | None)
    * ``elapsed_seconds`` (float)
    * ``current_util`` (float)
    """
    prc_seconds = float(
        getattr(settings, "post_reduction_cooldown_seconds", 0.0) or 0.0
    )
    clear_util_pct = float(
        getattr(
            settings, "post_reduction_cooldown_clear_util_pct", 0.30
        )
        or 0.0
    )
    sup_side = state.last_inventory_reduction_suppressed_side
    armed_at = state.last_inventory_reduction_at_mono
    if (
        prc_seconds <= 0.0
        or mode_is_normal
        or armed_at is None
        or sup_side is None
    ):
        return {
            "would_clamp": False,
            "ceiling_satisfied": False,
            "util_above_threshold": False,
            "suppressed_side": None,
            "elapsed_seconds": 0.0,
            "current_util": 0.0,
        }
    elapsed = float(now_mono) - float(armed_at)
    ceiling_satisfied = 0.0 <= elapsed < prc_seconds
    abs_cap = float(settings.max_abs_position)
    cur_util = (
        abs(float(state.position.position_qty)) / abs_cap
        if abs_cap > 1e-12
        else 0.0
    )
    util_above_threshold = cur_util >= clear_util_pct
    return {
        "would_clamp": ceiling_satisfied and util_above_threshold,
        "ceiling_satisfied": ceiling_satisfied,
        "util_above_threshold": util_above_threshold,
        "suppressed_side": sup_side,
        "elapsed_seconds": elapsed,
        "current_util": cur_util,
    }


def _build_state_in_v1_4_118_shape(
    *,
    cd_seconds: float = 60.0,
    clear_util_pct: float = 0.30,
    max_abs_position: float = 10.0,
) -> tuple[BotState, Settings]:
    """Synthesise the v1.4.118 emergency state: util ≈ 60 %, cooldown
    armed via a prior reducing fill, DEFENSIVE mode active."""
    s = Settings(
        POST_REDUCTION_COOLDOWN_SECONDS=cd_seconds,
        POST_REDUCTION_COOLDOWN_CLEAR_UTIL_PCT=clear_util_pct,
        MAX_ABS_POSITION=int(max_abs_position),
    )
    state = BotState(s)
    # Long inventory at 60 % util.
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=+6.0,  # |6| / 10 = 0.60 → "util=60%"
        avg_entry_price=2.040,
        mark_price=2.040,
        position_notional=6.0 * 2.040,
        unrealized_pnl_usd=0.0,
    )
    # Arm the cooldown: simulate a SELL fill that took +8 → +6
    # (reduction). ``_note_inventory_reduction_for_cooldown`` is the
    # arming entry point.
    state._note_inventory_reduction_for_cooldown(
        now_mono=100.0,
        fill_side=Side.SELL,
        prev_qty=+8.0,
        new_qty=+6.0,
    )
    return state, s


# ---------------------------------------------------------------------------
# Test 1 — armed cooldown clamps the BID (v1.4.118 baseline behaviour)
# ---------------------------------------------------------------------------


def test_cooldown_armed_clamps_bid_at_60pct_util() -> None:
    """At the moment the cooldown arms (util 60 %, DEFENSIVE), the
    bot SHOULD clamp to QUOTE_SELL_ONLY (suppress BUY/BID). This is
    the BEFORE state — the gate is doing its job."""
    state, s = _build_state_in_v1_4_118_shape()
    r = _evaluate_post_reduction_clamp(
        state, s, now_mono=100.5, mode_is_normal=False
    )
    assert r["would_clamp"] is True, r
    assert r["suppressed_side"] == QuoteEligibility.QUOTE_SELL_ONLY
    assert r["ceiling_satisfied"] is True
    assert r["util_above_threshold"] is True
    assert abs(r["current_util"] - 0.60) < 1e-9


# ---------------------------------------------------------------------------
# Test 2 — THE BUG: pre-2K.1 behaviour would have stayed clamped for
# 60 s regardless of util. After 2K.1, dropping util below the
# threshold lifts the clamp in the very next tick.
# ---------------------------------------------------------------------------


def test_cooldown_clears_within_one_tick_when_util_drops_below_threshold() -> None:
    """The v1.4.118 regression scenario, with the v1.4.144 fix:

    1. Cooldown armed at t=100 (util 60 %, position +6).
    2. ASK fill at t=101 takes position +6 → +2 (util 60 % → 20 %).
    3. NEXT tick at t=101.5 must see ``would_clamp = False`` because
       util (0.20) < clear_util_pct (0.30) — the favorable-exit
       predicate fires even though only 1.5 s of the 60 s timer
       has elapsed.

    Pre-2K.1 the cooldown would have stayed armed for the full 58.5 s
    of remaining budget."""
    state, s = _build_state_in_v1_4_118_shape()
    # Pre-fill: clamped.
    pre = _evaluate_post_reduction_clamp(
        state, s, now_mono=100.5, mode_is_normal=False
    )
    assert pre["would_clamp"] is True

    # Simulate an ASK fill that reduces +6 → +2. Just mutate
    # state.position directly (the integration test focuses on the
    # cooldown clamp logic; the fill ingestion path has its own
    # tests).
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=+2.0,
        avg_entry_price=2.040,
        mark_price=2.040,
        position_notional=2.0 * 2.040,
        unrealized_pnl_usd=0.0,
    )

    # NEXT tick: cooldown must clear.
    post = _evaluate_post_reduction_clamp(
        state, s, now_mono=101.5, mode_is_normal=False
    )
    assert post["would_clamp"] is False, (
        f"Cooldown failed to clear after util dropped from 0.60 to 0.20 "
        f"(below clear_util_pct=0.30). Result: {post}"
    )
    assert post["util_above_threshold"] is False
    assert abs(post["current_util"] - 0.20) < 1e-9
    # Ceiling is still in budget — proves it's the favorable-exit
    # predicate (not the timer) doing the work.
    assert post["ceiling_satisfied"] is True
    assert post["elapsed_seconds"] < 60.0


# ---------------------------------------------------------------------------
# Test 3 — Ceiling preserved: if util stays high, the timer remains
# the binding constraint (no regression to pre-1D behaviour)
# ---------------------------------------------------------------------------


def test_cooldown_ceiling_still_fires_if_util_stays_high() -> None:
    """If util stays at 60 % for the full cooldown window, the
    MAX-ceiling DOES eventually clear the clamp. Proves the timer
    didn't get removed entirely by the favorable-exit work."""
    state, s = _build_state_in_v1_4_118_shape()
    # Position unchanged at +6 (util 60 %) throughout.
    # Just before the ceiling (59.9 s elapsed).
    pre = _evaluate_post_reduction_clamp(
        state, s, now_mono=159.9, mode_is_normal=False
    )
    assert pre["would_clamp"] is True
    # Past the ceiling.
    post = _evaluate_post_reduction_clamp(
        state, s, now_mono=160.1, mode_is_normal=False
    )
    assert post["would_clamp"] is False
    assert post["ceiling_satisfied"] is False


# ---------------------------------------------------------------------------
# Test 4 — NORMAL mode never clamps regardless of state
# ---------------------------------------------------------------------------


def test_cooldown_does_not_clamp_in_normal_mode() -> None:
    """NORMAL mode keeps round-trip rebate capture intact: the
    cooldown's clamp only applies in DEFENSIVE/SHOCK."""
    state, s = _build_state_in_v1_4_118_shape()
    r = _evaluate_post_reduction_clamp(
        state, s, now_mono=100.5, mode_is_normal=True
    )
    assert r["would_clamp"] is False


# ---------------------------------------------------------------------------
# Test 5 — Threshold boundary semantics
# ---------------------------------------------------------------------------


def test_cooldown_boundary_util_strictly_below_threshold() -> None:
    """The condition is ``util >= clear_util_pct`` → util AT the
    threshold value still clamps; util STRICTLY BELOW clears."""
    state, s = _build_state_in_v1_4_118_shape(clear_util_pct=0.30)
    # util = 0.30 (right at the threshold) → still clamped.
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=+3.0,  # |3|/10 = 0.30
        avg_entry_price=2.040,
        mark_price=2.040,
        position_notional=3.0 * 2.040,
        unrealized_pnl_usd=0.0,
    )
    r = _evaluate_post_reduction_clamp(
        state, s, now_mono=101.0, mode_is_normal=False
    )
    assert r["util_above_threshold"] is True
    assert r["would_clamp"] is True
    # util = 0.29 (just below) → clears.
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=+2.9,
        avg_entry_price=2.040,
        mark_price=2.040,
        position_notional=2.9 * 2.040,
        unrealized_pnl_usd=0.0,
    )
    r2 = _evaluate_post_reduction_clamp(
        state, s, now_mono=101.0, mode_is_normal=False
    )
    assert r2["util_above_threshold"] is False
    assert r2["would_clamp"] is False


# ---------------------------------------------------------------------------
# Test 6 — Re-arm preserved
# ---------------------------------------------------------------------------


def test_cooldown_re_arms_on_subsequent_reduction() -> None:
    """If the cooldown clears via favorable-exit AND then a NEW
    reducing fill lands, the cooldown re-arms with a fresh
    timestamp. Proves the arming path still works after the
    favorable-exit fired (no stale-flag issue)."""
    state, s = _build_state_in_v1_4_118_shape()
    # Step 1: clear via favorable-exit.
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=+2.0,
        avg_entry_price=2.040,
        mark_price=2.040,
        position_notional=2.0 * 2.040,
        unrealized_pnl_usd=0.0,
    )
    cleared = _evaluate_post_reduction_clamp(
        state, s, now_mono=101.0, mode_is_normal=False
    )
    assert cleared["would_clamp"] is False
    # Step 2: another reducing fill at t=200, drops +2 → +1.
    # The arming side effect updates ``last_inventory_reduction_at_mono``
    # back to 200 — DEFENSIVE-mode clamp returns.
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=+1.0,
        avg_entry_price=2.040,
        mark_price=2.040,
        position_notional=1.0 * 2.040,
        unrealized_pnl_usd=0.0,
    )
    # But util is now 0.10 < 0.30 → still clears via favorable.
    # To prove the RE-ARM specifically, push position back up first.
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=+5.0,
        avg_entry_price=2.040,
        mark_price=2.040,
        position_notional=5.0 * 2.040,
        unrealized_pnl_usd=0.0,
    )
    state._note_inventory_reduction_for_cooldown(
        now_mono=200.0,
        fill_side=Side.SELL,
        prev_qty=+6.0,
        new_qty=+5.0,
    )
    # Re-armed at t=200, util=0.50 (above 0.30) → clamps again.
    rearmed = _evaluate_post_reduction_clamp(
        state, s, now_mono=200.5, mode_is_normal=False
    )
    assert rearmed["would_clamp"] is True
    assert rearmed["ceiling_satisfied"] is True
    # Elapsed since the re-arm should be ~0.5 s, NOT ~100 s.
    assert rearmed["elapsed_seconds"] < 1.0


# ---------------------------------------------------------------------------
# Test 7 — Inline source check: the inline logic in bot.py matches
# the replay in this test.
# ---------------------------------------------------------------------------


def test_inline_bot_logic_uses_clear_util_pct_setting() -> None:
    """Drift detector: if bot.py ever switches away from
    ``post_reduction_cooldown_clear_util_pct`` as the favorable-exit
    threshold (e.g. changes the formula or setting name), this test
    flags it loudly. Source-level inspection — cheap insurance."""
    import inspect
    import app.bot as bot_mod

    src = inspect.getsource(bot_mod)
    # Two structural markers the bot's post-reduction-clamp block
    # uses today. If either disappears, the integration test above
    # is checking the wrong logic and we want to be told.
    assert "post_reduction_cooldown_clear_util_pct" in src, (
        "bot.py no longer references the favorable-exit threshold "
        "setting. The 2K.T2 replay test may be stale — verify the "
        "extracted logic above still mirrors what the bot does."
    )
    assert "util_above_threshold" in src, (
        "bot.py no longer uses ``util_above_threshold`` as the "
        "Phase 2K.1 predicate variable. The 2K.T2 replay test may "
        "be stale."
    )
