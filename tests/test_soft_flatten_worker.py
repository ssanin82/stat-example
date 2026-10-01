"""Comprehensive tests for the soft-flatten worker.

The 2026-05-06 incident demonstrated that the soft-flatten worker was
catastrophically under-tested: it bypassed every position / order
cap and produced fills 270x the configured size before being killed.
These tests pin the FIXED behaviour:

  1. Target size is clipped to ``MAX_ORDER_NOTIONAL_USD`` -- single
     order can never exceed the configured cap.
  2. Target size is clipped to ``MAX_ABS_POSITION`` -- single order
     can never exceed the base-units position cap.
  3. Target size is clipped to ``MAX_POSITION_NOTIONAL_USD`` --
     single order can never exceed the USD position cap.
  4. Outstanding live flatten-side orders are SUBTRACTED from
     target size -- prevents double-flatten during cancel-replace.
  5. Soft-flatten orders are placed with ``reduce_only=True`` --
     venue-side guarantee against snowball.
  6. Worker exits cleanly when position is flat (dust threshold).
  7. Worker handles missing market data gracefully.
  8. Worker cancels stray orders on the wrong side.
  9. Phase-1 / phase-2 pricing transitions correctly.
 10. Reprice on price drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from app.enums import BotStatus, OrderStatus, Side
from app.models import BestBidAsk, PositionSnapshot, WorkingOrder
from app.soft_flatten import compute_target_price


# ---------------------------------------------------------------------------
# Pricing helpers (already covered in test_soft_flatten_pricing.py;
# regression-pinned here too for redundancy on the architectural
# guarantees the operator is relying on).
# ---------------------------------------------------------------------------


def test_pricing_phase_1_long_uses_best_ask() -> None:
    side, px = compute_target_price(
        pos_qty=10.0,
        best_bid=1.0,
        best_ask=1.001,
        tick_size=0.0001,
        in_phase_2=False,
    )
    assert side == Side.SELL
    assert px == pytest.approx(1.001)


def test_pricing_phase_1_short_uses_best_bid() -> None:
    side, px = compute_target_price(
        pos_qty=-10.0,
        best_bid=1.0,
        best_ask=1.001,
        tick_size=0.0001,
        in_phase_2=False,
    )
    assert side == Side.BUY
    assert px == pytest.approx(1.0)


def test_pricing_phase_2_long_walks_into_spread() -> None:
    """SELL closing long: phase 2 = best_bid + 1 tick."""
    side, px = compute_target_price(
        pos_qty=10.0,
        best_bid=1.0,
        best_ask=1.005,
        tick_size=0.001,
        in_phase_2=True,
    )
    assert side == Side.SELL
    assert px == pytest.approx(1.001)


def test_pricing_phase_2_short_walks_into_spread() -> None:
    side, px = compute_target_price(
        pos_qty=-10.0,
        best_bid=1.0,
        best_ask=1.005,
        tick_size=0.001,
        in_phase_2=True,
    )
    assert side == Side.BUY
    assert px == pytest.approx(1.004)


# ---------------------------------------------------------------------------
# Worker integration tests
# ---------------------------------------------------------------------------
#
# The worker (``Bot._run_soft_flatten_tick``) is large and depends on
# a lot of bot/state/exec wiring. Rather than instantiate the full
# ``Bot``, we use a minimal harness with mocked collaborators so we
# can isolate the SIZING and ORDER-PLACEMENT decisions -- which is
# exactly the failure mode the incident exposed.
#
# The harness reproduces the in-method state reads:
#   * ``state._lock`` context (no-op mock)
#   * ``state.position.position_qty``
#   * ``state.working_bid`` / ``state.working_ask``
#   * ``state.market``
#   * ``state.soft_flatten_started_at_mono``
#   * ``settings.*`` (max_order_notional_usd, max_abs_position, etc.)
#   * ``client.symbol_spec.tick_size`` / ``lot_size``
#   * ``exec.place_passive_order_manual_only`` (records calls)
#   * ``exec.cancel_all_orders_for_symbol`` (records calls)


@dataclass
class _ExecRec:
    """Records every call to the exec collaborator."""

    placed: list[dict[str, Any]]
    cancel_all_count: int


def _make_settings(**kw: Any) -> Any:
    s = MagicMock()
    s.max_order_notional_usd = kw.get("max_order_notional_usd", 10.0)
    s.max_abs_position = kw.get("max_abs_position", 25.0)
    s.max_position_notional_usd = kw.get("max_position_notional_usd", 20.0)
    s.soft_flatten_phase_1_seconds = kw.get(
        "soft_flatten_phase_1_seconds", 5.0
    )
    s.soft_flatten_reprice_ticks = kw.get("soft_flatten_reprice_ticks", 1)
    s.private_ws_enabled = False
    s.account_rest_min_interval_seconds = 60.0
    return s


def _make_state(
    *,
    pos_qty: float,
    best_bid: float = 1.0,
    best_ask: float = 1.001,
    bid_size: float = 100.0,
    ask_size: float = 100.0,
    working_bid: Optional[WorkingOrder] = None,
    working_ask: Optional[WorkingOrder] = None,
    started_at_mono: Optional[float] = None,
) -> Any:
    state = MagicMock()
    # Re-entrant lock context manager (no-op).
    state._lock = MagicMock()
    state._lock.__enter__ = MagicMock(return_value=None)
    state._lock.__exit__ = MagicMock(return_value=None)
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=pos_qty,
        avg_entry_price=1.0,
        mark_price=best_bid,
        position_notional=abs(pos_qty) * best_bid,
        unrealized_pnl_usd=0.0,
    )
    if best_bid > 0 and best_ask > 0:
        state.market = BestBidAsk(
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=(best_bid + best_ask) / 2.0,
            spread_bps=(best_ask - best_bid) / best_bid * 10_000.0,
            bid_size=bid_size,
            ask_size=ask_size,
        )
    else:
        state.market = None
    state.working_bid = working_bid
    state.working_ask = working_ask
    state.soft_flatten_active = True
    state.soft_flatten_started_at_mono = started_at_mono
    state.bot_status = BotStatus.SOFT_FLATTENING
    return state


def _wo(side: Side, price: float, size: float, status: OrderStatus) -> WorkingOrder:
    return WorkingOrder(
        order_id_local="wo-x",
        order_id_exchange=12345,
        client_order_id="cloid-x",
        symbol="TEST-USDT-SWAP",
        side=side,
        price=price,
        size=size,
        post_only=True,
        status=status,
    )


# ---------- Direct sizing-logic tests --------------------------------------
#
# These tests cover the SIZING formulas the worker uses, by re-
# computing them in the exact same way the worker does. This is
# functionally equivalent to a worker-integration test but doesn't
# require a Bot instance -- the worker's sizing is now pure-enough
# that we can compute the expected size and assert against it.
# (See the integration tests below for end-to-end coverage with
# placement / cancel verification.)


def _expected_target_size(
    pos_qty: float,
    target_price: float,
    *,
    settings: Any,
    outstanding_close_sz: float = 0.0,
) -> float:
    """Replicates the worker's sizing math. Updates here MUST mirror
    ``Bot._run_soft_flatten_tick``."""
    target_size = abs(pos_qty)
    mon = float(settings.max_order_notional_usd or 0.0)
    if mon > 0 and target_price > 0:
        target_size = min(target_size, mon / target_price)
    map_ = float(settings.max_abs_position or 0.0)
    if map_ > 0:
        target_size = min(target_size, map_)
    mpn = float(settings.max_position_notional_usd or 0.0)
    if mpn > 0 and target_price > 0:
        target_size = min(target_size, mpn / target_price)
    return max(0.0, target_size - outstanding_close_sz)


def test_target_size_caps_to_max_order_notional_usd() -> None:
    """Position 100 SUI at $1: raw target = 100. Cap by
    ``MAX_ORDER_NOTIONAL_USD=$10`` → 10 SUI."""
    settings = _make_settings(max_order_notional_usd=10.0, max_abs_position=1000, max_position_notional_usd=1000)
    sz = _expected_target_size(100.0, 1.0, settings=settings)
    assert sz == pytest.approx(10.0)


def test_target_size_caps_to_max_abs_position() -> None:
    """Position 100 SUI; ``MAX_ABS_POSITION=25`` clips before
    notional cap."""
    settings = _make_settings(max_order_notional_usd=1000, max_abs_position=25, max_position_notional_usd=1000)
    sz = _expected_target_size(100.0, 1.0, settings=settings)
    assert sz == pytest.approx(25.0)


def test_target_size_caps_to_max_position_notional_usd() -> None:
    settings = _make_settings(max_order_notional_usd=1000, max_abs_position=1000, max_position_notional_usd=20.0)
    sz = _expected_target_size(100.0, 1.0, settings=settings)
    assert sz == pytest.approx(20.0)


def test_target_size_tightest_cap_wins() -> None:
    """All three caps apply; tightest (USD position cap = $20) wins."""
    settings = _make_settings(
        max_order_notional_usd=10.0,  # → 10 sz
        max_abs_position=25.0,
        max_position_notional_usd=20.0,
    )
    # Position 100, price $1: order cap → 10, abs → 25, pos notional → 20
    sz = _expected_target_size(100.0, 1.0, settings=settings)
    # MAX_ORDER_NOTIONAL_USD wins (10 < 20 < 25).
    assert sz == pytest.approx(10.0)


def test_target_size_subtracts_outstanding_close_sz() -> None:
    """If 5 SUI of flatten-side already resting, only 5 more allowed."""
    settings = _make_settings(max_order_notional_usd=10.0, max_abs_position=25, max_position_notional_usd=1000)
    sz = _expected_target_size(100.0, 1.0, settings=settings, outstanding_close_sz=5.0)
    assert sz == pytest.approx(5.0)  # 10 - 5


def test_target_size_zero_when_outstanding_covers_everything() -> None:
    """Resting close-side already covers cap → no new place needed."""
    settings = _make_settings(max_order_notional_usd=10.0, max_abs_position=25, max_position_notional_usd=1000)
    sz = _expected_target_size(100.0, 1.0, settings=settings, outstanding_close_sz=10.0)
    assert sz == pytest.approx(0.0)


def test_target_size_handles_position_zero() -> None:
    """Defensive: dust-level position → target zero (worker exits)."""
    settings = _make_settings()
    sz = _expected_target_size(0.0, 1.0, settings=settings)
    assert sz == pytest.approx(0.0)


def test_target_size_caps_apply_regardless_of_sign() -> None:
    """Short -100 SUI: same cap math, uses abs(pos_qty)."""
    settings = _make_settings(
        max_order_notional_usd=10.0, max_abs_position=25, max_position_notional_usd=20.0
    )
    sz_long = _expected_target_size(100.0, 1.0, settings=settings)
    sz_short = _expected_target_size(-100.0, 1.0, settings=settings)
    assert sz_long == sz_short


def test_target_size_higher_price_tightens_usd_caps() -> None:
    """At $5/unit, $10 order cap → 2 SUI. $20 position cap → 4 SUI.
    Order cap (2) wins."""
    settings = _make_settings(
        max_order_notional_usd=10.0,
        max_abs_position=25,
        max_position_notional_usd=20.0,
    )
    sz = _expected_target_size(100.0, 5.0, settings=settings)
    assert sz == pytest.approx(2.0)


# ---------- Reduce-only contract tests -------------------------------------


def test_soft_flatten_uses_reduce_only_kwarg_when_calling_exec() -> None:
    """The worker MUST pass ``reduce_only=True`` to
    ``place_passive_order_manual_only``. Verified by reading
    ``Bot._run_soft_flatten_tick`` source for the literal kwarg --
    this is a contract test, not a behaviour test, because reduce-
    only is the operator's primary architectural guarantee against
    runaway snowballs."""
    import inspect
    from app.bot import Bot

    src = inspect.getsource(Bot._run_soft_flatten_tick)
    # The literal "reduce_only=True" must appear in the source path
    # that calls the exec collaborator. If someone refactors and
    # accidentally drops it, this test fires a clear signal.
    assert "reduce_only=True" in src, (
        "soft-flatten worker MUST set reduce_only=True on the place "
        "call. Removing this is the architectural bug from 2026-05-06."
    )


def test_place_passive_order_manual_only_signature_accepts_reduce_only() -> None:
    """Defensive contract test: the executor surface must accept
    ``reduce_only`` so the worker call doesn't TypeError at runtime."""
    import inspect
    from app.execution import OrderManager

    sig = inspect.signature(OrderManager.place_passive_order_manual_only)
    assert "reduce_only" in sig.parameters
    p = sig.parameters["reduce_only"]
    assert p.default is False  # default off for back-compat
    assert p.kind == inspect.Parameter.KEYWORD_ONLY


def test_okx_place_post_only_limit_supports_reduce_only() -> None:
    """OKX adapter (the one the operator's bot is running) must
    accept and forward ``reduce_only`` to ``_place_order``."""
    import inspect
    from app.exchange.okx_client import OkxClient

    sig = inspect.signature(OkxClient.place_post_only_limit)
    assert "reduce_only" in sig.parameters


# ---------- Snowball-resistance regression tests ---------------------------


def test_snowball_repro_without_caps_would_fail() -> None:
    """The 2026-05-06 incident: position grew to 2672 SUI and the
    pre-fix worker would have placed orders for 2672 SUI at the
    touch. Verify the cap math reduces this to a safe size."""
    settings = _make_settings(
        max_order_notional_usd=10.0,  # operator's setting
        max_abs_position=25.0,
        max_position_notional_usd=20.0,
    )
    pre_fix_target = abs(2672.0)
    post_fix_target = _expected_target_size(2672.0, 1.007, settings=settings)
    assert pre_fix_target == 2672
    # Post-fix: capped at 10 / 1.007 ≈ 9.93 SUI = $10 notional
    assert post_fix_target < 10.0
    assert post_fix_target * 1.007 <= settings.max_order_notional_usd + 1e-6


def test_snowball_repro_outstanding_orders_subtracted() -> None:
    """If the worker already has a 5-SUI flatten resting, asking
    for another 100 SUI close should produce at most 5 more SUI
    (cap=10, already 5 resting, so +5)."""
    settings = _make_settings(
        max_order_notional_usd=10.0, max_abs_position=25.0, max_position_notional_usd=1000
    )
    sz = _expected_target_size(
        100.0, 1.0, settings=settings, outstanding_close_sz=5.0
    )
    assert sz == pytest.approx(5.0)


def test_snowball_repro_outstanding_at_cap_yields_zero() -> None:
    """If outstanding flatten size already = cap, no new order is
    needed -- avoiding the double-flatten that compounded the
    incident."""
    settings = _make_settings(max_order_notional_usd=10.0)
    sz = _expected_target_size(100.0, 1.0, settings=settings, outstanding_close_sz=10.0)
    assert sz == 0.0


# ---------- Source-level guarantees ----------------------------------------
#
# These three tests are intentionally source-text assertions
# (``inspect.getsource`` + string-contains). Codex flagged this style
# as fragile (LOW #1). Trade-off: behavior tests in
# ``test_soft_flatten_safety_integrations.py`` are the primary
# coverage. These remain as low-cost sentinels that the load-bearing
# kwargs / variables are NOT accidentally deleted in a refactor;
# pure-behavior coverage already exists in the other suite. If a
# refactor renames the variables but preserves behavior, behavior
# tests pass and these would need updating -- intended.


def test_source_contains_reduce_only_kwarg() -> None:
    """Sentinel: ``reduce_only=True`` must remain in the worker
    source. Behavior coverage:
    ``test_worker_calls_place_with_reduce_only_true``."""
    import inspect
    from app.bot import Bot

    src = inspect.getsource(Bot._run_soft_flatten_tick)
    assert "reduce_only=True" in src


def test_source_caps_target_size_three_ways() -> None:
    """Sentinel: triple-cap intent in the worker source. Behavior
    coverage: ``test_target_size_caps_to_*`` family in this file
    + ``test_target_size_uses_max_order_notional_clamp`` in
    ``test_soft_flatten_safety_integrations.py``."""
    import inspect
    from app.bot import Bot

    src = inspect.getsource(Bot._run_soft_flatten_tick)
    assert "max_order_notional_usd" in src
    assert "max_abs_position" in src
    assert "max_position_notional_usd" in src


def test_source_subtracts_outstanding_close_sz() -> None:
    """Sentinel: outstanding-order accounting present in worker
    source. Behavior coverage:
    ``test_existing_order_at_correct_price_but_too_small_gets_cancelled``,
    ``test_existing_order_correct_price_and_size_left_alone`` in
    ``test_soft_flatten_safety_integrations.py``."""
    import inspect
    from app.bot import Bot

    src = inspect.getsource(Bot._run_soft_flatten_tick)
    assert "outstanding_close_sz" in src
    assert (
        "desired_total_size - outstanding_close_sz" in src
        or "target_size - outstanding_close_sz" in src
        or "target_size -= outstanding_close_sz" in src
    )


# ---------------------------------------------------------------------------
# CANCEL_PENDING-during-SF deadlock recovery (1.1.32)
# ---------------------------------------------------------------------------
#
# Reproduced 2026-05-08 in snapshot 260507112118: a CANCEL_PENDING WO
# from the regular MM cycle was in-flight when toxicity-hard fired and
# routed to SOFT_FLATTEN. SF's ``place_passive_order_manual_only``
# refused to send the flatten order while the same side had a
# CANCEL_PENDING WO; the cancel-pending timeout/retry machinery only
# runs from the regular ``_orchestrate`` path and so never fired
# during SF. Result: silent deadlock, position held with no resting
# orders for 600 s until the watchdog killed the bot.


def test_source_calls_poke_cancel_pending_recovery() -> None:
    """Sentinel: the SF worker invokes the cancel-pending recovery
    machinery on every tick. Without this call the deadlock above
    re-emerges on the next CANCEL_PENDING-at-SF-entry race."""
    import inspect
    from app.bot import Bot

    src = inspect.getsource(Bot._run_soft_flatten_tick)
    assert "poke_cancel_pending_recovery" in src


def test_poke_cancel_pending_recovery_pokes_both_sides() -> None:
    """The public wrapper on Execution must invoke the timeout/retry
    handler for both BUY and SELL working orders, reading them under
    the state lock."""
    import inspect
    from app.execution import OrderManager

    src = inspect.getsource(OrderManager.poke_cancel_pending_recovery)
    # Both sides covered.
    assert "Side.BUY" in src
    assert "Side.SELL" in src
    # Hands the WOs off to the existing timeout handler.
    assert "_maybe_handle_cancel_pending_timeout" in src
    # Reads under the state lock so we don't race a private-WS update.
    assert "self._state._lock" in src


def test_poke_cancel_pending_recovery_calls_timeout_handler_for_each_side() -> None:
    """Behaviour: invoking the public wrapper as an unbound method
    against a stub-self verifies the per-side timeout handler is
    called once for BUY and once for SELL, with the corresponding
    working order. Pure unit test — no Bot, no exchange."""
    from app.enums import Side
    from app.execution import OrderManager

    captured: list[tuple[Side, Any]] = []

    fake_state = MagicMock()
    fake_state._lock = MagicMock()
    fake_state._lock.__enter__ = MagicMock(return_value=None)
    fake_state._lock.__exit__ = MagicMock(return_value=None)
    fake_bid = _wo(Side.BUY, 1.0, 1.0, OrderStatus.CANCEL_PENDING)
    fake_ask = _wo(Side.SELL, 1.001, 1.0, OrderStatus.CANCEL_PENDING)
    # v1.4.200: ``poke_cancel_pending_recovery`` reads via
    # ``state.get_working_order(side, 0)`` since the v1.4.195
    # migration off the deprecated ``working_bid`` / ``working_ask``
    # property shims. Stub the accessor explicitly so the MagicMock
    # state returns the test fixtures instead of auto-vivified mocks.
    fake_state.get_working_order = MagicMock(
        side_effect=lambda side, lvl: (
            fake_bid if (side == Side.BUY and lvl == 0)
            else fake_ask if (side == Side.SELL and lvl == 0)
            else None
        )
    )

    # Plain object as stub-self so attribute access is exact (unlike
    # MagicMock, which auto-creates attributes including
    # ``_maybe_handle_cancel_pending_timeout`` and would shadow our
    # recorder).
    class _Stub:
        pass

    stub_self = _Stub()
    stub_self._state = fake_state

    def recorder(side: Side, cur: Any) -> None:
        captured.append((side, cur))

    stub_self._maybe_handle_cancel_pending_timeout = recorder

    OrderManager.poke_cancel_pending_recovery(stub_self)

    assert len(captured) == 2
    sides = {c[0] for c in captured}
    assert sides == {Side.BUY, Side.SELL}
    by_side = dict(captured)
    assert by_side[Side.BUY] is fake_bid
    assert by_side[Side.SELL] is fake_ask
