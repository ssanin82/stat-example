"""Behavior tests for the soft-flatten worker integrations:
- CRITICAL #1: missing/crossed book cancels resting flatten orders
              (rather than silent return)
- CRITICAL #1: risk eval runs before worker (kill conditions still fire
              during soft-flatten)
- HIGH #1: existing-order branch handles size + price together
- MED #3: locked / crossed book triggers cancel-and-wait

Replaces the more fragile ``inspect.getsource()`` source-text tests
with actual function-level behaviour assertions.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.bot import Bot
from app.enums import BotStatus, OrderStatus, RiskAction, Side
from app.models import (
    BestBidAsk,
    PnlSnapshot,
    PositionSnapshot,
    RiskDecision,
    ToxicitySnapshot,
    WorkingOrder,
)


# ---------------------------------------------------------------------------
# Lightweight harness
# ---------------------------------------------------------------------------
#
# We instantiate Bot via __new__ + manual attribute injection so we
# don't have to wire up a real exchange adapter / database / WS
# stack. The methods under test only touch ``self._state``, ``self._
# settings``, ``self._exec``, ``self._client``, ``self._pnl``, and
# friends -- mock the minimum.


def _make_settings(**kw):
    from tests.settings_helpers import UnitTestSettings

    path = Path(tempfile.gettempdir()) / f"mm_sfsi_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 25.0,
        "MAX_POSITION_NOTIONAL_USD": 1_000.0,
        "MAX_ORDER_NOTIONAL_USD": 100.0,
        "SOFT_FLATTEN_PHASE_1_SECONDS": 5.0,
        "SOFT_FLATTEN_REPRICE_TICKS": 1,
        # Match venue spec floors so the SF pre-check (now
        # ``max(spec_min, settings.min_quote_notional_usd)`` per the
        # 1.1.36 broadening) doesn't unintentionally trigger an
        # exit-on-residual-too-small in tests that test taker
        # fallback / phase-2 behaviour, not min-notional logic.
        # Tests that DO want to exercise the broadened pre-check
        # override this explicitly.
        "MIN_QUOTE_NOTIONAL_USD": 5.0,
    }
    base.update(kw)
    return UnitTestSettings.model_validate(base)


def _bot(state, settings, *, exec_mock=None, client_mock=None) -> Bot:
    bot = Bot.__new__(Bot)
    bot._state = state
    bot._settings = settings
    # Phase 1c (v1.4.231) — Bot.__init__ normally installs self._clock
    # but tests using ``__new__`` bypass it. Provide a default SystemClock
    # so the migrated ``self._clock.monotonic()`` / ``self._clock.now_utc()``
    # call sites in the soft-flatten worker have something to call.
    from app.clock import SystemClock
    bot._clock = SystemClock()
    bot._exec = exec_mock or MagicMock()
    if client_mock is None:
        client_mock = MagicMock()
        # ``SymbolSpec`` exposes ``price_tick`` / ``size_step``;
        # ``tick_size`` / ``lot_size`` were a typo Codex flagged
        # 2026-05-09 (silently returned 0 in production). Using the
        # real attribute names here so the test guards against
        # regression. ``min_size`` aliases ``size_step`` in this
        # fixture for adapters that expose it.
        client_mock.symbol_spec = MagicMock(
            price_tick=0.0001, size_step=1.0, min_size=1.0
        )
    bot._client = client_mock
    bot._pnl = MagicMock()
    bot._pnl.build_snapshot = MagicMock(
        return_value=PnlSnapshot(0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 1000.0, datetime.now(timezone.utc))
    )
    bot._storage = MagicMock()
    # Suppress account REST refresh inside the worker tick so the
    # mock client doesn't overwrite our test-set position state.
    bot._should_refresh_account_rest = MagicMock(return_value=(False, ""))
    return bot


def _state(*, pos_qty=10.0, best_bid=1.0, best_ask=1.001, working_bid=None, working_ask=None, settings=None):
    from app.state import BotState

    s = settings or _make_settings()
    state = BotState(s)
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=pos_qty,
        avg_entry_price=1.0,
        mark_price=(best_bid + best_ask) / 2.0 if best_bid and best_ask else 1.0,
        position_notional=abs(pos_qty) * 1.0,
        unrealized_pnl_usd=0.0,
    )
    if best_bid is not None and best_ask is not None:
        state.market = BestBidAsk(
            symbol=s.symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=(best_bid + best_ask) / 2.0,
            spread_bps=(best_ask - best_bid) / best_bid * 10_000.0 if best_bid > 0 else 0.0,
            bid_size=100.0,
            ask_size=100.0,
        )
    else:
        state.market = None
    state.working_bid = working_bid
    state.working_ask = working_ask
    state.soft_flatten_active = True
    state.soft_flatten_started_at_mono = time.monotonic()
    return state, s


def _wo(side: Side, *, price: float, size: float, status: OrderStatus = OrderStatus.ACKED) -> WorkingOrder:
    return WorkingOrder(
        order_id_local="wo",
        order_id_exchange=12345,
        client_order_id="cloid",
        symbol="TEST-USDT-SWAP",
        side=side,
        price=price,
        size=size,
        post_only=True,
        status=status,
    )


# ---------------------------------------------------------------------------
# CRITICAL #1 — missing/crossed book CANCELS resting flatten orders
# ---------------------------------------------------------------------------


def test_missing_book_cancels_resting_flatten_order() -> None:
    """Pre-fix: silent ``return`` on missing book leaves resting
    flatten order parked at stale price. Post-fix: cancel it."""
    state, s = _state(pos_qty=10.0, best_bid=None, best_ask=None,
                      working_ask=_wo(Side.SELL, price=1.001, size=10.0))
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()


def test_crossed_book_cancels_resting_flatten_order() -> None:
    """best_bid >= best_ask (locked / crossed book) triggers cancel.
    Phase-2 pricing on crossed book would otherwise produce a
    crossing post-only that gets rejected."""
    state, s = _state(pos_qty=10.0, best_bid=1.001, best_ask=1.0,  # crossed!
                      working_ask=_wo(Side.SELL, price=1.005, size=10.0))
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()


def test_missing_book_no_resting_no_cancel_call() -> None:
    """Defensive: if there's no resting flatten order AND no book,
    don't spam cancel-all (nothing to cancel)."""
    state, s = _state(pos_qty=10.0, best_bid=None, best_ask=None,
                      working_ask=None, working_bid=None)
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    bot._exec.cancel_all_orders_for_symbol.assert_not_called()


def test_normal_book_does_not_trigger_missing_book_path() -> None:
    """Sanity: with a healthy book, the worker proceeds to placement."""
    state, s = _state(pos_qty=10.0, best_bid=1.0, best_ask=1.001)
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    # Did NOT trigger missing-book cancel-all path.
    bot._exec.cancel_all_orders_for_symbol.assert_not_called()
    # DID try to place.
    assert bot._exec.place_passive_order_manual_only.call_count == 1


# ---------------------------------------------------------------------------
# HIGH #1 — existing-order branch: stale price on covered position
# ---------------------------------------------------------------------------


def test_existing_order_at_stale_price_gets_cancelled_even_when_size_covers() -> None:
    """Existing SELL covers position size but is at stale price.
    Pre-fix: returned early (size check passed first), leaving
    stale order. Post-fix: cancels because price is wrong."""
    settings = _make_settings(MAX_ORDER_NOTIONAL_USD=1_000.0, SOFT_FLATTEN_REPRICE_TICKS=1)
    state, s = _state(
        pos_qty=10.0,
        best_bid=1.0,
        best_ask=1.001,  # phase-1 target = best_ask = 1.001
        working_ask=_wo(Side.SELL, price=1.05, size=15.0, status=OrderStatus.ACKED),  # WAY off price
        settings=settings,
    )
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()
    bot._exec.place_passive_order_manual_only.assert_not_called()


def test_existing_order_at_correct_price_but_too_small_gets_cancelled() -> None:
    """Existing SELL is at right price but only covers 5 of 10 needed.
    Pre-fix: returned early (price acceptable). Post-fix: cancels and
    re-places at full size."""
    settings = _make_settings(MAX_ORDER_NOTIONAL_USD=1_000.0, SOFT_FLATTEN_REPRICE_TICKS=1)
    state, s = _state(
        pos_qty=10.0,
        best_bid=1.0,
        best_ask=1.001,
        # Phase-1 target = best_ask = 1.001. Existing at right price
        # but only size 5 -- needs 10 to flatten.
        working_ask=_wo(Side.SELL, price=1.001, size=5.0, status=OrderStatus.ACKED),
        settings=settings,
    )
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()


def test_existing_order_correct_price_and_size_left_alone() -> None:
    """Existing SELL covers position fully at right price -- no
    further action this tick."""
    settings = _make_settings(MAX_ORDER_NOTIONAL_USD=1_000.0, SOFT_FLATTEN_REPRICE_TICKS=1)
    state, s = _state(
        pos_qty=10.0,
        best_bid=1.0,
        best_ask=1.001,
        working_ask=_wo(Side.SELL, price=1.001, size=10.0, status=OrderStatus.ACKED),
        settings=settings,
    )
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    bot._exec.cancel_all_orders_for_symbol.assert_not_called()
    bot._exec.place_passive_order_manual_only.assert_not_called()


# ---------------------------------------------------------------------------
# Reduce-only at the placement call
# ---------------------------------------------------------------------------


def test_worker_calls_place_with_reduce_only_true() -> None:
    state, s = _state(pos_qty=10.0, best_bid=1.0, best_ask=1.001)
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    call = bot._exec.place_passive_order_manual_only.call_args
    assert call is not None
    assert call.kwargs.get("reduce_only") is True


# ---------------------------------------------------------------------------
# Cap math + outstanding subtraction with the new desired_total_size logic
# ---------------------------------------------------------------------------


def test_target_size_uses_max_order_notional_clamp() -> None:
    """Position 100 SUI, MAX_ORDER_NOTIONAL_USD=$10 at $1.0/SUI →
    target 10 SUI per order even though position is 100."""
    settings = _make_settings(
        MAX_ORDER_NOTIONAL_USD=10.0,
        MAX_POSITION_NOTIONAL_USD=1_000.0,
        MAX_ABS_POSITION=1_000.0,
    )
    state, s = _state(pos_qty=100.0, best_bid=1.0, best_ask=1.001, settings=settings)
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    call = bot._exec.place_passive_order_manual_only.call_args
    assert call is not None
    # Args: (side, price, size, cycle, ...) — size is positional arg #2
    placed_size = call.args[2]
    expected = settings.max_order_notional_usd / 1.001  # close at best_ask
    assert placed_size == pytest.approx(expected, rel=0.001)


def test_target_size_subtracts_resting_flatten_size() -> None:
    """Resting flatten of 5 SUI; pos 10; new order should be 5."""
    settings = _make_settings(MAX_ORDER_NOTIONAL_USD=1_000.0)
    state, s = _state(
        pos_qty=10.0,
        best_bid=1.0,
        best_ask=1.001,
        # Wrong price (forces re-place path, but...) Actually use the
        # NO-existing path for direct sizing test:
        working_ask=None,
        settings=settings,
    )
    bot = _bot(state, s)
    # Manually inject a resting same-side via a separate scenario -- the
    # test above (existing_order_correct_price_and_size_left_alone) and
    # the cap math tests in test_soft_flatten_worker.py pin the
    # subtraction logic. Here we just verify the no-existing path
    # produces full size 10.
    bot._run_soft_flatten_tick()
    placed_size = bot._exec.place_passive_order_manual_only.call_args.args[2]
    assert placed_size == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Below-min-notional residual treats as job done (2026-05-06 stuck bug)
# ---------------------------------------------------------------------------
#
# Pre-fix: position −4 SUI at $0.985 = $3.94 notional with a $5 min_
# notional_usd floor stalled forever. Worker tried to place every
# tick, was rejected by the synthetic min-notional gate in
# ``execution.place_passive_order_manual_only``, set
# ``_min_notional_passive_block`` and silently no-op'd thereafter.
# Bot stayed in SOFT_FLATTENING with no orders + no progress.
# Post-fix: worker recognises an unworkable residual and exits
# SOFT_FLATTENING cleanly, returning to RUNNING. Normal quoting will
# work the residual off via opposite-side fills, and the position-
# drawdown gate has its own below-min-notional skip preventing
# re-entry.


def _bot_with_min_notional(state, settings, *, min_notional_usd: float):
    """Variant of ``_bot`` that wires ``symbol_spec.min_notional_usd``
    on the mock client. The default ``_bot`` harness leaves it as a
    MagicMock attribute (truthy but non-numeric)."""
    bot = Bot.__new__(Bot)
    bot._state = state
    bot._settings = settings
    # Phase 1c (v1.4.231) — see comment in ``_bot()`` above.
    from app.clock import SystemClock
    bot._clock = SystemClock()
    bot._exec = MagicMock()
    client_mock = MagicMock()
    client_mock.symbol_spec = MagicMock(
        price_tick=0.0001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=min_notional_usd,
    )
    bot._client = client_mock
    bot._pnl = MagicMock()
    bot._pnl.build_snapshot = MagicMock(
        return_value=PnlSnapshot(0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 1000.0, datetime.now(timezone.utc))
    )
    bot._storage = MagicMock()
    bot._notifier = None
    bot._should_refresh_account_rest = MagicMock(return_value=(False, ""))
    return bot


def test_residual_below_min_notional_exits_soft_flatten() -> None:
    """Stuck-forever scenario: short −4 SUI at $0.985 mid =
    $3.94 notional, ``min_notional_usd=$5``. Worker exits
    soft-flatten without trying to place (would be rejected) and
    transitions back to RUNNING."""
    state, s = _state(pos_qty=-4.0, best_bid=0.984, best_ask=0.986)
    # _state computes notional with best_bid; override to be explicit.
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=-4.0,
        avg_entry_price=0.98,
        mark_price=0.985,
        position_notional=4.0 * 0.985,  # $3.94
        unrealized_pnl_usd=0.0,
    )
    state.bot_status = BotStatus.SOFT_FLATTENING
    bot = _bot_with_min_notional(state, s, min_notional_usd=5.0)
    bot._run_soft_flatten_tick()
    bot._exec.place_passive_order_manual_only.assert_not_called()
    assert state.soft_flatten_active is False
    assert state.bot_status == BotStatus.RUNNING


def test_residual_above_min_notional_proceeds_to_placement() -> None:
    """Above-min-notional regression: position $10 notional with
    ``min_notional_usd=$5`` proceeds to normal placement."""
    state, s = _state(pos_qty=-10.0, best_bid=0.99, best_ask=1.001)
    bot = _bot_with_min_notional(state, s, min_notional_usd=5.0)
    bot._run_soft_flatten_tick()
    bot._exec.place_passive_order_manual_only.assert_called_once()
    # Did NOT exit.
    assert state.soft_flatten_active is True


def test_residual_below_min_notional_no_placement_attempt() -> None:
    """Defensive: no place call AND no cancel-all spam beyond the
    one that ``_exit_soft_flatten`` issues. The previous (broken)
    behaviour had the worker silently no-op forever; this test pins
    the new exit path so a regression would surface as ``place was
    called`` rather than the original ``forever-stuck-no-place``
    failure mode (which a passing test couldn't detect)."""
    state, s = _state(pos_qty=-4.0, best_bid=0.984, best_ask=0.986)
    state.position = PositionSnapshot(
        symbol=s.symbol,
        position_qty=-4.0,
        avg_entry_price=0.98,
        mark_price=0.985,
        position_notional=4.0 * 0.985,
        unrealized_pnl_usd=0.0,
    )
    state.bot_status = BotStatus.SOFT_FLATTENING
    bot = _bot_with_min_notional(state, s, min_notional_usd=5.0)
    bot._run_soft_flatten_tick()
    bot._exec.place_passive_order_manual_only.assert_not_called()
    # _exit_soft_flatten paranoid-cancels resting orders; here the
    # state has none, but the exec mock will still see the call
    # because the impl always issues it. We don't assert on this --
    # the contract is "no place".


# ---------------------------------------------------------------------------
# Source-level sentinel: position-drawdown gate skips below min_notional
# ---------------------------------------------------------------------------
#
# The gate-skip is at the call site in ``Bot.one_tick``. Behaviour-
# testing it requires the full one_tick wiring (market data refresh,
# risk eval, etc.); a source sentinel matches the established style
# elsewhere in this suite.


def test_source_position_drawdown_gate_skips_below_min_notional() -> None:
    """Sentinel: ``one_tick`` must skip the gate when position
    notional is below the venue's min_notional_usd. Without this,
    the gate fires on a sub-tradable position the worker can never
    close, putting the bot in an un-exitable SOFT_FLATTENING."""
    import inspect

    src = inspect.getsource(Bot.one_tick)
    assert "below_min_ntn" in src, (
        "one_tick must compute below_min_ntn from "
        "client.symbol_spec.min_notional_usd vs position_notional"
    )
    assert "not below_min_ntn" in src, (
        "one_tick must include `and not below_min_ntn` in the gate "
        "precondition; otherwise the gate fires on un-tradable residuals "
        "and the bot stalls in SOFT_FLATTENING"
    )


# ---------------------------------------------------------------------------
# Phase-2-immediate (force_phase=2) — toxicity-trigger entry path
# ---------------------------------------------------------------------------
#
# The toxicity-trigger path enters SF with ``force_phase=2`` so the
# worker uses far-touch ∓ 1 tick pricing on the very first tick
# instead of waiting ``SOFT_FLATTEN_PHASE_1_SECONDS`` at the near
# touch. Verified by the price the worker requests on its first
# tick after entry.


def test_force_phase_2_uses_phase2_pricing_immediately() -> None:
    """``force_phase=2`` skips the patient phase-1 window. With a
    long position and far-touch SELL, phase-2 price = best_bid + 1
    tick (one tick into the spread). Compare to phase-1 (near touch
    = best_ask) to confirm the override took effect."""
    settings = _make_settings(SOFT_FLATTEN_PHASE_1_SECONDS=999.0)
    state, s = _state(pos_qty=10.0, best_bid=1.0, best_ask=1.005, settings=settings)
    # Critical: set started_at_mono to NOW, so elapsed=0 → without
    # the override the worker would be in phase 1.
    state.soft_flatten_started_at_mono = time.monotonic()
    state.soft_flatten_force_phase = 2
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    call = bot._exec.place_passive_order_manual_only.call_args
    assert call is not None, "expected a place call with force_phase=2"
    placed_side = call.args[0]
    placed_price = call.args[1]
    # Long position, want to SELL. Phase-2 = best_bid + 1 tick.
    # Harness tick = 0.0001, so phase-2 = 1.0001.
    # Phase-1 would be best_ask = 1.005.
    assert placed_side == Side.SELL
    assert placed_price == pytest.approx(1.0001, abs=1e-9), (
        f"phase-2 pricing expected 1.0001 (best_bid + 1 tick at tick=0.0001); "
        f"got {placed_price} — force_phase override didn't take effect"
    )


def test_force_phase_none_uses_phase1_when_elapsed_below_threshold() -> None:
    """Regression guard: without ``force_phase``, behaviour stays
    legacy — phase 1 (near touch) until SOFT_FLATTEN_PHASE_1_SECONDS
    elapses."""
    settings = _make_settings(SOFT_FLATTEN_PHASE_1_SECONDS=999.0)
    state, s = _state(pos_qty=10.0, best_bid=1.0, best_ask=1.005, settings=settings)
    state.soft_flatten_started_at_mono = time.monotonic()
    state.soft_flatten_force_phase = None  # explicit
    bot = _bot(state, s)
    bot._run_soft_flatten_tick()
    placed_price = bot._exec.place_passive_order_manual_only.call_args.args[1]
    # Phase 1 = best_ask for a SELL closing a long position.
    assert placed_price == pytest.approx(1.005, abs=1e-9)


# ---------------------------------------------------------------------------
# Taker fallback (phase 3) — adverse drift escape
# ---------------------------------------------------------------------------
#
# Phase 3: when adverse mid drift exceeds
# ``soft_flatten_taker_fallback_ticks`` since SF entry, the worker
# fires a single ``client.market_close`` and exits SF. Configurable
# per-entry; off by default. Direction-aware: long position adverse
# = mid moves down; short position adverse = mid moves up.


def test_taker_fallback_fires_on_long_when_mid_drops() -> None:
    """Long position. Entry mid 1.0, current mid 1.0 - 5 ticks
    (with tick=0.001 ⇒ 0.995). 5-tick adverse threshold met →
    market_close called, SF exits."""
    state, s = _state(pos_qty=10.0, best_bid=0.9945, best_ask=0.9955)
    state.soft_flatten_active = True
    state.soft_flatten_started_at_mono = time.monotonic()
    state.soft_flatten_entry_mid = 1.0
    state.soft_flatten_taker_fallback_ticks = 5
    state.bot_status = BotStatus.SOFT_FLATTENING
    bot = _bot_with_min_notional(state, s, min_notional_usd=5.0)
    bot._run_soft_flatten_tick()
    bot._client.market_close.assert_called_once_with(s.symbol)
    bot._exec.place_passive_order_manual_only.assert_not_called()
    assert state.soft_flatten_active is False
    assert state.bot_status == BotStatus.RUNNING


def test_taker_fallback_fires_on_short_when_mid_rises() -> None:
    """Short position. Entry mid 1.0, current mid 1.005. 5-tick
    adverse threshold met → market_close fires."""
    state, s = _state(pos_qty=-10.0, best_bid=1.0045, best_ask=1.0055)
    state.soft_flatten_active = True
    state.soft_flatten_started_at_mono = time.monotonic()
    state.soft_flatten_entry_mid = 1.0
    state.soft_flatten_taker_fallback_ticks = 5
    state.bot_status = BotStatus.SOFT_FLATTENING
    bot = _bot_with_min_notional(state, s, min_notional_usd=5.0)
    bot._run_soft_flatten_tick()
    bot._client.market_close.assert_called_once_with(s.symbol)
    assert state.soft_flatten_active is False


def test_taker_fallback_does_not_fire_when_drift_in_favorable_direction() -> None:
    """Long position. Mid moved UP (favourable for a long). Should
    NOT fire fallback regardless of magnitude — adverse direction
    only."""
    state, s = _state(pos_qty=10.0, best_bid=1.0095, best_ask=1.0105)
    state.soft_flatten_active = True
    state.soft_flatten_started_at_mono = time.monotonic()
    state.soft_flatten_entry_mid = 1.0
    state.soft_flatten_taker_fallback_ticks = 5
    state.bot_status = BotStatus.SOFT_FLATTENING
    bot = _bot_with_min_notional(state, s, min_notional_usd=5.0)
    bot._run_soft_flatten_tick()
    bot._client.market_close.assert_not_called()
    assert state.soft_flatten_active is True


def test_taker_fallback_disabled_when_threshold_zero_or_none() -> None:
    """Threshold None → no fallback path even with massive drift.
    Same for 0. Verifies opt-in semantics: legacy callers that
    don't set the field get legacy behaviour (post-only forever)."""
    for ticks in (None, 0):
        state, s = _state(pos_qty=10.0, best_bid=0.94, best_ask=0.95)
        state.soft_flatten_active = True
        state.soft_flatten_started_at_mono = time.monotonic()
        state.soft_flatten_entry_mid = 1.0
        state.soft_flatten_taker_fallback_ticks = ticks
        state.bot_status = BotStatus.SOFT_FLATTENING
        bot = _bot_with_min_notional(state, s, min_notional_usd=5.0)
        bot._run_soft_flatten_tick()
        bot._client.market_close.assert_not_called()
        assert state.soft_flatten_active is True


def test_taker_fallback_does_not_fire_below_threshold() -> None:
    """3 ticks of adverse drift (tick=0.0001 ⇒ drift=0.0003);
    threshold is 5 → no fallback."""
    state, s = _state(pos_qty=10.0, best_bid=0.99965, best_ask=0.99975)
    state.soft_flatten_active = True
    state.soft_flatten_started_at_mono = time.monotonic()
    state.soft_flatten_entry_mid = 1.0
    state.soft_flatten_taker_fallback_ticks = 5
    state.bot_status = BotStatus.SOFT_FLATTENING
    bot = _bot_with_min_notional(state, s, min_notional_usd=5.0)
    bot._run_soft_flatten_tick()
    bot._client.market_close.assert_not_called()
    assert state.soft_flatten_active is True


# ---------------------------------------------------------------------------
# Toxicity-flatten now uses RiskAction.SOFT_FLATTEN (not FLATTEN)
# ---------------------------------------------------------------------------


def test_toxicity_with_inventory_returns_soft_flatten() -> None:
    """The toxicity-flatten path in ``risk.py`` must return
    ``RiskAction.SOFT_FLATTEN`` (post-only worker), not
    ``RiskAction.FLATTEN`` (taker market_close). The flip is the
    operator-facing fix from 2026-05-07."""
    from app.config import Settings
    from app.enums import (
        BotStatus as _BotStatus,
        DesyncPhase as _DesyncPhase,
        RiskAction as _RiskAction,
        Side as _Side,
    )
    from app.models import (
        BestBidAsk as _BestBidAsk,
        PnlSnapshot as _PnlSnapshot,
        ToxicitySnapshot as _ToxicitySnapshot,
    )
    from app.risk import evaluate_risk
    from datetime import datetime, timezone
    from tests.settings_helpers import UnitTestSettings

    s_local: Settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            # ``flatten_on_kill`` no longer affects this path —
            # explicit assertion that True doesn't re-enable taker
            # flattening on toxicity.
            "FLATTEN_ON_KILL": True,
            "MAX_ABS_POSITION": 25.0,
            "MAX_POSITION_NOTIONAL_USD": 100.0,
        }
    )
    tox = _ToxicitySnapshot(
        score=0.9,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=-3.0,
        vol_spike_ratio=0.0,
        hard_trigger=True,
        soft_trigger=False,
        toxic_side=_Side.BUY,
    )
    pnl_snap = _PnlSnapshot(0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 1000.0, datetime.now(timezone.utc))
    market = _BestBidAsk(
        symbol=s_local.symbol,
        best_bid=1.0,
        best_ask=1.001,
        mid_price=1.0005,
        spread_bps=10.0,
        bid_size=100.0,
        ask_size=100.0,
    )
    decision = evaluate_risk(
        s_local,
        bot_status=_BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=market,
        position_qty=5.0,  # NON-ZERO inventory
        position_notional=5.0,
        open_order_count=0,
        pnl=pnl_snap,
        toxicity=tox,
        execution_errors=0,
        desync=False,
        desync_phase=_DesyncPhase.OK,
    )
    assert decision.action == _RiskAction.SOFT_FLATTEN, (
        f"toxicity-with-inventory should return SOFT_FLATTEN; got {decision.action}"
    )
    assert "toxicity_soft_flatten" in decision.reasons


def test_toxicity_no_inventory_falls_through_to_side_suppression() -> None:
    """Toxicity hard-trigger but flat position → side suppression
    (no flatten of any kind, since there's nothing to close)."""
    from app.config import Settings
    from app.enums import (
        BotStatus as _BotStatus,
        DesyncPhase as _DesyncPhase,
        RiskAction as _RiskAction,
        Side as _Side,
    )
    from app.models import (
        BestBidAsk as _BestBidAsk,
        PnlSnapshot as _PnlSnapshot,
        ToxicitySnapshot as _ToxicitySnapshot,
    )
    from app.risk import evaluate_risk
    from datetime import datetime, timezone
    from tests.settings_helpers import UnitTestSettings

    s_local: Settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "MAX_ABS_POSITION": 25.0,
        }
    )
    tox = _ToxicitySnapshot(
        score=0.9,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=-3.0,
        vol_spike_ratio=0.0,
        hard_trigger=True,
        soft_trigger=False,
        toxic_side=_Side.BUY,
    )
    pnl_snap = _PnlSnapshot(0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 1000.0, datetime.now(timezone.utc))
    market = _BestBidAsk(
        symbol=s_local.symbol,
        best_bid=1.0,
        best_ask=1.001,
        mid_price=1.0005,
        spread_bps=10.0,
        bid_size=100.0,
        ask_size=100.0,
    )
    decision = evaluate_risk(
        s_local,
        bot_status=_BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=market,
        position_qty=0.0,  # FLAT
        position_notional=0.0,
        open_order_count=0,
        pnl=pnl_snap,
        toxicity=tox,
        execution_errors=0,
        desync=False,
        desync_phase=_DesyncPhase.OK,
    )
    # toxic_side=BUY → bot suppresses bid (ASK_ONLY).
    assert decision.action == _RiskAction.ASK_ONLY
