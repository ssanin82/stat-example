"""Tests for 1.3.130 multi-rung Phase 2 — per-rung working-order
state on ``BotState``.

The backward-compat shims (``working_bid`` / ``working_ask`` as
properties backed by the per-rung dict) are the critical correctness
guarantee. These tests assert:

* Inside-rung (level_idx=0) round-trips through the shims identically
  to the pre-Phase-2 scalar attribute.
* Outer rungs (level_idx>0) coexist with inside rungs without
  affecting the shim view.
* ``set_working_order(side, level_idx, None)`` deletes the entry
  (no None-sentinel accumulation).
* ``iter_working_orders`` returns rungs in level_idx-ascending order.
* ``all_working_orders`` enumerates BUY then SELL, inside first.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.config import Settings
from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.state import BotState


def _settings() -> Settings:
    return Settings(
        EXCHANGE="okx",
        SYMBOL="SUI-USDT-SWAP",
        TRADING_ENABLED=False,
    )


def _wo(*, side: Side, level_idx: int = 0, oid_local: str = "X") -> WorkingOrder:
    return WorkingOrder(
        order_id_local=oid_local,
        order_id_exchange=None,
        client_order_id=None,
        symbol="SUI-USDT-SWAP",
        side=side,
        price=1.0,
        size=10.0,
        post_only=True,
        status=OrderStatus.NEW_LOCAL,
        ts_created=datetime(2026, 5, 17, 0, 0, 0, tzinfo=timezone.utc),
        level_idx=level_idx,
    )


def test_working_bid_shim_round_trips_to_inside_rung() -> None:
    """Setting the legacy ``working_bid`` field stores the WO at
    (BUY, 0). Reading ``working_bid`` returns it back. The shim is the
    backward-compat path for the 60+ scalar read sites that haven't
    been refactored to use ``get_working_order``."""
    s = BotState(_settings())
    assert s.working_bid is None
    wo = _wo(side=Side.BUY)
    s.working_bid = wo
    assert s.working_bid is wo
    # Also visible via the per-rung accessor.
    assert s.get_working_order(Side.BUY, 0) is wo
    # Setting to None deletes the entry.
    s.working_bid = None
    assert s.working_bid is None
    assert s.get_working_order(Side.BUY, 0) is None


def test_working_ask_shim_independent_from_working_bid() -> None:
    """The BUY and SELL slots are independent — setting working_bid
    doesn't touch working_ask."""
    s = BotState(_settings())
    bid = _wo(side=Side.BUY, oid_local="B")
    ask = _wo(side=Side.SELL, oid_local="A")
    s.working_bid = bid
    s.working_ask = ask
    assert s.working_bid is bid
    assert s.working_ask is ask
    s.working_bid = None
    assert s.working_bid is None
    assert s.working_ask is ask


def test_outer_rung_coexists_with_inside_rung() -> None:
    """At N=2 the inside rung (level_idx=0) and the outer rung
    (level_idx=1) live in distinct slots. The ``working_bid`` shim
    returns ONLY the inside rung — outer rungs are invisible to the
    legacy scalar interface, which is the intended backward-compat
    behaviour (the 60+ read sites care only about inside)."""
    s = BotState(_settings())
    inside = _wo(side=Side.BUY, level_idx=0, oid_local="I")
    outer = _wo(side=Side.BUY, level_idx=1, oid_local="O")
    s.set_working_order(Side.BUY, 0, inside)
    s.set_working_order(Side.BUY, 1, outer)
    # Scalar shim still sees the inside.
    assert s.working_bid is inside
    # Per-rung accessor sees both.
    assert s.get_working_order(Side.BUY, 0) is inside
    assert s.get_working_order(Side.BUY, 1) is outer


def test_iter_working_orders_returns_level_idx_sorted() -> None:
    """``iter_working_orders`` returns ``(level_idx, WO)`` tuples
    sorted ascending. Critical for the ladder orchestrator which
    relies on rung ordering for deterministic dispatch."""
    s = BotState(_settings())
    wo_outer = _wo(side=Side.BUY, level_idx=2, oid_local="2")
    wo_inside = _wo(side=Side.BUY, level_idx=0, oid_local="0")
    wo_mid = _wo(side=Side.BUY, level_idx=1, oid_local="1")
    # Insert in scrambled order.
    s.set_working_order(Side.BUY, 2, wo_outer)
    s.set_working_order(Side.BUY, 0, wo_inside)
    s.set_working_order(Side.BUY, 1, wo_mid)
    levels = [idx for idx, _ in s.iter_working_orders(Side.BUY)]
    assert levels == [0, 1, 2]


def test_set_working_order_none_removes_entry() -> None:
    """Setting a rung to None deletes the dict entry — the iter
    helpers shouldn't see a None sentinel."""
    s = BotState(_settings())
    s.set_working_order(Side.BUY, 1, _wo(side=Side.BUY, level_idx=1))
    assert len(s.iter_working_orders(Side.BUY)) == 1
    s.set_working_order(Side.BUY, 1, None)
    assert len(s.iter_working_orders(Side.BUY)) == 0


def test_all_working_orders_enumerates_buy_then_sell_inside_first() -> None:
    """``all_working_orders`` returns BUY rungs (inside first) then
    SELL rungs (inside first). Deterministic ordering simplifies
    telemetry and reconcile-loop semantics."""
    s = BotState(_settings())
    s.set_working_order(Side.SELL, 1, _wo(side=Side.SELL, level_idx=1, oid_local="s1"))
    s.set_working_order(Side.SELL, 0, _wo(side=Side.SELL, level_idx=0, oid_local="s0"))
    s.set_working_order(Side.BUY, 1, _wo(side=Side.BUY, level_idx=1, oid_local="b1"))
    s.set_working_order(Side.BUY, 0, _wo(side=Side.BUY, level_idx=0, oid_local="b0"))
    flat = s.all_working_orders()
    assert [w.order_id_local for w in flat] == ["b0", "b1", "s0", "s1"]


def test_open_order_count_includes_outer_rungs() -> None:
    """``open_order_count`` counts every non-terminal rung across
    both sides. At N=1 this is identical to the pre-Phase-2 behaviour
    (counts the inside rung if present); at N=2 the outer rungs are
    also counted so the residual-flatten / max-open-orders gates
    see the true count."""
    s = BotState(_settings())
    s.set_working_order(Side.BUY, 0, _wo(side=Side.BUY, level_idx=0, oid_local="b0"))
    s.set_working_order(Side.BUY, 1, _wo(side=Side.BUY, level_idx=1, oid_local="b1"))
    s.set_working_order(Side.SELL, 0, _wo(side=Side.SELL, level_idx=0, oid_local="s0"))
    # All three are NEW_LOCAL (non-terminal) → count is 3.
    assert s.open_order_count() == 3


def test_open_order_count_skips_terminal_rungs() -> None:
    """Terminal-status rungs (CANCELED / FILLED / REJECTED) are not
    counted. This matches the pre-Phase-2 scalar behaviour."""
    s = BotState(_settings())
    wo = _wo(side=Side.BUY, level_idx=0)
    wo.status = OrderStatus.CANCELED
    s.set_working_order(Side.BUY, 0, wo)
    assert s.open_order_count() == 0


def test_open_order_count_skips_DESYNC_rungs() -> None:
    """v1.5.136 (Codex bug #10) — DESYNC is documented as a terminal
    state in ``OrderStatus``'s docstring
    (``TERMINAL = {CANCELED, FILLED, REJECTED, DESYNC}``) and is
    already excluded by three other code paths (reconciler
    ``_TERMINAL`` set, ``_local_has_cancellable_wos`` skip list,
    ``cancel_all_orders_for_symbol`` skip list). Before this fix
    ``open_order_count`` was the only counter that disagreed — it
    over-counted DESYNC tombstones, which in turn inflated the
    per-tick ``evaluate_risk`` ``max_open_orders`` check at
    ``app/bot.py:6280`` / ``:7609`` and produced spurious
    ``CANCEL_ALL`` decisions during the 30 s window
    (``DESYNC_REAP_TIMEOUT_SECONDS``) before the reaper retired
    the tombstone. Net failure mode: a 30-second cancel-storm
    after every legitimate DESYNC event."""
    s = BotState(_settings())
    # One live order (ACKED) + one DESYNC tombstone on the same side.
    live = _wo(side=Side.BUY, level_idx=0, oid_local="live")
    live.status = OrderStatus.ACKED
    tombstone = _wo(side=Side.BUY, level_idx=1, oid_local="ghost")
    tombstone.status = OrderStatus.DESYNC
    s.set_working_order(Side.BUY, 0, live)
    s.set_working_order(Side.BUY, 1, tombstone)
    # Post-fix: only the live order counts. Pre-fix returned 2.
    assert s.open_order_count() == 1


def test_open_order_count_counts_in_flight_states() -> None:
    """v1.5.136 (Codex bug #10) companion — sanity check that the
    NON-terminal states from the docstring's ``IN_FLIGHT`` and
    ``LIVE_ON_VENUE`` partitions are all still counted. The fix
    only adds DESYNC to the exclusion set; it must not accidentally
    drop genuinely-live order states."""
    s = BotState(_settings())
    cases = [
        OrderStatus.NEW_LOCAL,
        OrderStatus.SENT,
        OrderStatus.ACKED,
        OrderStatus.PARTIAL,
        OrderStatus.AMEND_PENDING,
        OrderStatus.CANCEL_PENDING,
    ]
    for i, status in enumerate(cases):
        wo = _wo(side=Side.BUY, level_idx=i, oid_local=f"o{i}")
        wo.status = status
        s.set_working_order(Side.BUY, i, wo)
    assert s.open_order_count() == len(cases)


def test_has_resting_passive_order_sees_outer_rung() -> None:
    """A resting (ACKED) outer rung counts as a passive order on
    the book — the blind-resting watchdog and risk gates that
    consult this method must see every rung, not just the inside."""
    s = BotState(_settings())
    outer = _wo(side=Side.BUY, level_idx=1, oid_local="O")
    outer.status = OrderStatus.ACKED
    s.set_working_order(Side.BUY, 1, outer)
    assert s.has_resting_passive_order() is True
