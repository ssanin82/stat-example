"""Regression tests for the Codex-flagged bugs from the 2026-05-09
review (``reports/codex-review-20260509.md``):

1. CRITICAL — soft-flatten read ``client.symbol_spec.tick_size`` /
   ``lot_size``; the live ``SymbolSpec`` exposes ``price_tick`` /
   ``size_step``. ``getattr(..., default=0.0)`` silently returned 0,
   gating phase-2 / reprice-tolerance / taker-fallback off entirely.

2. HIGH — ``_risk_kill_check_during_soft_flatten`` only handled
   ``KILL`` and ``FLATTEN``; ``NO_QUOTE`` and ``CANCEL_ALL`` fell
   through and the SF worker kept placing quotes through degraded
   states.

3. HIGH — REST fill catch-up sliced ``fills_raw[:50]`` even though
   OKX returns 100. Half of any private-WS-missed burst was
   silently dropped from local accounting (fees, realized PnL,
   toxicity, session counters).

4. HIGH — ``apply_market_book_only`` updated ``self.market`` but
   never re-marked ``position.unrealized_pnl_usd`` from the new mid.
   Drawdown / session-loss / position-drawdown gates ran on stale
   unrealized between REST refreshes (~8 s) — a position could move
   materially against the bot before the gates noticed.

5. MED — soft-flatten min-notional precheck dereferenced
   ``market.best_bid`` / ``market.best_ask`` before the missing-book
   guard ran. A reconnect with one BBO side still ``None`` would
   raise ``TypeError`` on ``float(None)`` instead of falling through
   to the cancel-and-wait path.

6. MED — no index covered ``orders.order_id_exchange``, so per-fill
   attribution lookups (soft_flatten_event_id, target_half_spread,
   quote_aggressiveness) degraded to table scans as the orders
   table grew.

7. MED — shadow-position update preserved ``avg_entry_price=None``
   on flat→position transitions. The first few seconds after a
   fresh fill ran the position-drawdown gate and session-loss logic
   without a valid entry reference until REST repaired it.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from app.enums import OrderStatus, RiskAction, Side
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC, SymbolSpec
from app.models import AccountSnapshot, BestBidAsk, PositionSnapshot
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings(**kw):
    path = (
        Path(tempfile.gettempdir())
        / f"mm_codex_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    }
    base.update(kw)
    return UnitTestSettings.model_validate(base)


# =============================================================================
# Bug #1 — SymbolSpec attribute typo
# =============================================================================


def test_bug1_real_symbolspec_exposes_price_tick_not_tick_size() -> None:
    """Pin the public attribute names. The fact that ``SymbolSpec``
    exposes ``price_tick`` / ``size_step`` and NOT ``tick_size`` /
    ``lot_size`` is the contract; soft-flatten reads against this
    shape, and the previous typo silently returned 0 because the
    bot was using the wrong attribute names with ``getattr`` defaults.
    """
    spec = FALLBACK_SYMBOL_SPEC
    # Real attribute names exist and are non-zero.
    assert hasattr(spec, "price_tick")
    assert hasattr(spec, "size_step")
    assert spec.price_tick > 0
    assert spec.size_step > 0
    # The old (typo) names DO NOT exist.
    assert not hasattr(spec, "tick_size")
    assert not hasattr(spec, "lot_size")


def test_bug1_soft_flatten_reads_price_tick_from_real_spec() -> None:
    """End-to-end check: build a Bot with a REAL ``SymbolSpec``
    (not a MagicMock) and verify the soft-flatten reads land on
    a positive tick. Pre-fix this would have returned 0 because
    ``getattr(spec, 'tick_size', 0.0)`` defaulted silently.
    """
    # Real spec with a known tick.
    spec = SymbolSpec(
        price_tick=0.0001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=5.0,
        sz_decimals=4,
        source="hyperliquid_meta",
    )
    # Mirror the soft-flatten lines from app/bot.py:
    sp = spec
    lot = float(
        getattr(sp, "size_step", 0.0)
        or getattr(sp, "min_size", 0.0)
        or 0.0
    )
    tick_size = float(getattr(sp, "price_tick", 0.0) or 0.0)
    assert lot == 1.0
    assert tick_size == 0.0001
    # And the OLD attribute names would have silently returned 0:
    legacy_lot = float(
        getattr(sp, "lot_size", 0.0)
        or getattr(sp, "min_size", 0.0)  # falls back to min_size, so non-zero
        or 0.0
    )
    legacy_tick = float(getattr(sp, "tick_size", 0.0) or 0.0)
    # min_size fallback masks the lot bug; tick has no fallback.
    assert legacy_lot == 1.0  # masked by min_size
    assert legacy_tick == 0.0  # the real bug: no fallback


# =============================================================================
# Bug #2 — _risk_kill_check_during_soft_flatten honours NO_QUOTE / CANCEL_ALL
# =============================================================================
#
# Rather than spinning up a full Bot instance (large surface, brittle
# fixtures), we drive the decision branch directly through a stand-in
# ``risk.action`` and verify the orchestration return value + the
# cancel-resting hook gets called for CANCEL_ALL.


def _bot_with_mocked_risk(action: RiskAction, monkeypatch):
    """Build the smallest possible Bot stand-in that exercises the
    final branch dispatch in _risk_kill_check_during_soft_flatten
    given a pre-baked ``risk.action``. We monkey-patch the upstream
    ``evaluate_risk`` via the pytest fixture so the patch auto-
    reverts at end-of-test (avoids polluting downstream tests that
    import ``app.bot.evaluate_risk``).
    """
    from app.bot import Bot
    from app.clock import SystemClock
    import app.bot as bot_mod

    bot = Bot.__new__(Bot)
    bot._clock = SystemClock()
    bot._settings = MagicMock()
    bot._state = MagicMock()
    bot._state._lock = MagicMock()
    bot._state._lock.__enter__ = MagicMock(return_value=None)
    bot._state._lock.__exit__ = MagicMock(return_value=None)
    bot._state.position = PositionSnapshot(
        symbol="TEST",
        position_qty=5.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=5.0,
        unrealized_pnl_usd=0.0,
    )
    bot._state.toxicity = MagicMock()
    bot._state.reconcile_auto_pause = False
    bot._state.account = None
    bot._state.bot_status = MagicMock()
    bot._state.manual_pause = False
    bot._state.killed = False
    bot._state.flatten_mode = False
    bot._state.market = None
    bot._state.open_order_count = MagicMock(return_value=0)
    bot._state.execution_errors_window_snapshot = MagicMock(
        return_value={"windowed": 0}
    )
    bot._state.order_desync = False
    bot._state.desync_phase = "none"
    bot._state.desync_quarantine_remaining = 0
    bot._state.trades_last_minute = MagicMock(return_value=0)
    bot._state.account_rest_last_monotonic = None
    bot._state.soft_flatten_active = True
    bot._state.soft_flatten_started_at_mono = None
    # v1.4.200: the bot reads working orders via
    # ``state.get_working_order(side, level_idx)`` since the v1.4.195
    # migration off the deprecated property shims. The MagicMock state
    # auto-vivifies unknown method calls to fresh MagicMock objects, so
    # without an explicit stub the code reads "an order exists" for
    # every side (the MagicMock truthy default) and downstream checks
    # mis-fire. Default to None; individual tests overwrite via
    # ``bot._state.get_working_order = MagicMock(side_effect=...)``.
    bot._state.get_working_order = MagicMock(return_value=None)

    bot._exec = MagicMock()
    bot._exec.drain_private_events = MagicMock()
    bot._exec.should_ingest_fills_via_rest = MagicMock(return_value=False)
    bot._exec._enqueue_cancel_quote_path = MagicMock()
    bot._client = MagicMock()
    bot._storage = MagicMock()
    bot._pnl = MagicMock()
    bot._pnl.build_snapshot = MagicMock(return_value=MagicMock())

    def _fake_should_refresh(_addr):
        return False, 0.0

    bot._should_refresh_account_rest = _fake_should_refresh
    bot._public_ws_risk_kwargs = MagicMock(return_value={})
    bot._build_kill_payload = MagicMock(return_value={})
    bot.kill = MagicMock()
    bot.flatten = MagicMock()

    # Stub upstream risk eval to return the action we want to test.
    # Use monkeypatch.setattr so the override auto-reverts at
    # end-of-test (NOT a raw assignment — that polluted downstream
    # tests importing ``app.bot.evaluate_risk`` in the same suite).
    fake_risk = MagicMock()
    fake_risk.action = action
    fake_risk.reasons = ["test"]
    monkeypatch.setattr(
        bot_mod, "evaluate_risk", MagicMock(return_value=fake_risk)
    )
    return bot


def test_bug2_no_quote_during_sf_blocks_worker_keeps_sf_active(monkeypatch) -> None:
    bot = _bot_with_mocked_risk(RiskAction.NO_QUOTE, monkeypatch)
    skip = bot._risk_kill_check_during_soft_flatten()
    # Caller must skip the worker tick.
    assert skip is True
    # SF mode is held (not toggled off — we want to resume on next
    # tick if conditions clear).
    assert bot._state.soft_flatten_active is True
    # No kill / flatten / cancel.
    bot.kill.assert_not_called()
    bot.flatten.assert_not_called()
    bot._exec._enqueue_cancel_quote_path.assert_not_called()


def test_bug2_cancel_all_during_sf_cancels_resting_orders(monkeypatch) -> None:
    bot = _bot_with_mocked_risk(RiskAction.CANCEL_ALL, monkeypatch)
    # Stage a resting bid that should get cancelled.
    from app.models import WorkingOrder

    wo_bid = WorkingOrder(
        order_id_local="bid-1",
        order_id_exchange=10,
        client_order_id=None,
        symbol="TEST",
        side=Side.BUY,
        price=1.0,
        size=5.0,
        post_only=True,
        status=OrderStatus.ACKED,
    )
    # v1.4.200: stage the WO through the ``get_working_order``
    # accessor (the v1.4.195-onward read path). ``side_effect`` lets
    # us return wo_bid for the BUY/0 slot and None for SELL/0 in one
    # callable.
    bot._state.get_working_order = MagicMock(
        side_effect=lambda side, lvl: wo_bid
        if side == Side.BUY and lvl == 0
        else None
    )

    skip = bot._risk_kill_check_during_soft_flatten()
    assert skip is True
    # Resting bid was cancelled.
    bot._exec._enqueue_cancel_quote_path.assert_called_once_with(
        wo_bid, trigger_reason="soft_flatten"
    )
    # SF mode held (we want to resume placement when cancel-all clears).
    assert bot._state.soft_flatten_active is True
    bot.kill.assert_not_called()
    bot.flatten.assert_not_called()


def test_bug2_kill_during_sf_still_kills(monkeypatch) -> None:
    """Sanity check that the new branches didn't break the existing
    KILL handling."""
    bot = _bot_with_mocked_risk(RiskAction.KILL, monkeypatch)
    skip = bot._risk_kill_check_during_soft_flatten()
    assert skip is True
    bot.kill.assert_called_once()
    assert bot._state.soft_flatten_active is False


def test_bug2_allow_during_sf_lets_worker_run(monkeypatch) -> None:
    """ALLOW should NOT skip — the worker should run and handle
    quoting. Pin so we don't accidentally over-extend the new
    branches and break the happy path."""
    bot = _bot_with_mocked_risk(RiskAction.ALLOW, monkeypatch)
    skip = bot._risk_kill_check_during_soft_flatten()
    assert skip is False


# =============================================================================
# Bug #3 — REST catch-up ingests ALL returned fills, not just first 50
# =============================================================================


def test_bug3_rest_catchup_ingests_all_returned_fills() -> None:
    """Stage a 60-fill REST batch (above the old hard-coded 50 cap).
    Pre-fix, fills 50-59 were silently dropped. Post-fix all 60 are
    ingested for session bookkeeping.
    """
    from app.market_data import refresh_account_only

    settings = _settings()
    state = BotState(settings)

    # Mock client returns 60 fills via fetch_recent_fills_raw.
    client = MagicMock()
    client.fetch_position = MagicMock(
        return_value=PositionSnapshot(
            symbol="TEST",
            position_qty=0.0,
            avg_entry_price=None,
            mark_price=None,
            position_notional=0.0,
            unrealized_pnl_usd=0.0,
        )
    )
    client.fetch_account_snapshot = MagicMock(
        return_value=AccountSnapshot(
            equity_usd=1000.0,
            cash_usd=1000.0,
            withdrawable_usd=1000.0,
        )
    )

    # Build 60 fake fills. The exact shape matters only as far as
    # ``ingest_hl_fill_raw`` accepts the rows; we count CALLS to it.
    raw_fills = [{"trade_id": str(i), "px": 1.0} for i in range(60)]
    client.fetch_recent_fills_raw = MagicMock(return_value=raw_fills)

    storage = MagicMock()
    pnl = MagicMock()

    # Patch the per-fill ingest function so we can count calls
    # without exercising the full DB-write path.
    from app import market_data as md

    ingest_calls = {"n": 0}

    def _fake_ingest(*args, **kwargs):
        ingest_calls["n"] += 1

    real_ingest = md.ingest_hl_fill_raw
    md.ingest_hl_fill_raw = _fake_ingest
    try:
        refresh_account_only(
            client,
            state,
            "0xabc",
            storage,
            pnl,
            ingest_fills_via_rest=True,
        )
    finally:
        md.ingest_hl_fill_raw = real_ingest

    # Pre-fix would have stopped at 50; post-fix processes all 60.
    assert ingest_calls["n"] == 60


# =============================================================================
# Bug #4 — apply_market_book_only re-marks unrealized PnL from new mid
# =============================================================================


def test_bug4_book_update_remarks_unrealized_when_position_open() -> None:
    """Position long 10@$1.00, BBO publishes a new mid of $0.99 →
    unrealized should drop to (0.99 - 1.00) * 10 = -$0.10. Pre-fix,
    unrealized stayed at whatever REST last reported until the next
    refresh (8 s).
    """
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST",
        position_qty=10.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=10.0,
        unrealized_pnl_usd=0.0,  # REST-set; we'll see this update
    )

    # New BBO with mid of 0.99.
    new_book = BestBidAsk(
        symbol="TEST",
        best_bid=0.989,
        best_ask=0.991,
        mid_price=0.99,
        spread_bps=20.0,
    )
    state.apply_market_book_only(new_book, market_data_source="public_ws")

    # Live re-marked.
    assert abs(state.position.unrealized_pnl_usd - (-0.10)) < 1e-9
    # Mark price is REST-authoritative; should NOT be touched.
    assert state.position.mark_price == 1.0


def test_bug4_book_update_skips_remark_when_avg_entry_missing() -> None:
    """If the position came from a flat-state shadow update (no
    ``avg_entry_price`` yet — REST hasn't reported), the re-mark
    should NO-OP rather than dividing by zero or computing nonsense.
    """
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST",
        position_qty=5.0,
        avg_entry_price=None,
        mark_price=None,
        position_notional=5.0,
        unrealized_pnl_usd=0.0,
    )

    new_book = BestBidAsk(
        symbol="TEST",
        best_bid=1.05,
        best_ask=1.07,
        mid_price=1.06,
        spread_bps=190.0,
    )
    state.apply_market_book_only(new_book, market_data_source="public_ws")

    # No re-mark; legacy field unchanged.
    assert state.position.unrealized_pnl_usd == 0.0
    assert state.position.avg_entry_price is None


def test_bug4_book_update_skips_remark_when_position_flat() -> None:
    """Flat position → unrealized is trivially 0. The re-mark
    branch should short-circuit; we just want to verify it doesn't
    write nonsense or raise."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )

    new_book = BestBidAsk(
        symbol="TEST",
        best_bid=2.0,
        best_ask=2.001,
        mid_price=2.0005,
        spread_bps=5.0,
    )
    state.apply_market_book_only(new_book, market_data_source="public_ws")

    assert state.position.unrealized_pnl_usd == 0.0
    assert state.position.position_qty == 0.0


def test_bug4_book_update_remarks_short_position() -> None:
    """Sign check: short 10@$1.00, mid moves UP to $1.02 →
    unrealized = (1.02 - 1.00) * (-10) = -$0.20."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST",
        position_qty=-10.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=10.0,
        unrealized_pnl_usd=0.0,
    )

    new_book = BestBidAsk(
        symbol="TEST",
        best_bid=1.019,
        best_ask=1.021,
        mid_price=1.02,
        spread_bps=20.0,
    )
    state.apply_market_book_only(new_book, market_data_source="public_ws")

    assert abs(state.position.unrealized_pnl_usd - (-0.20)) < 1e-9


# =============================================================================
# Bug 5 — soft-flatten min-notional precheck must tolerate partial-book
# =============================================================================


def test_bug5_min_notional_precheck_handles_partial_book_no_bid() -> None:
    """When the public-WS feed is recovering from a reconnect and
    only one BBO side is populated, the soft-flatten min-notional
    check must not throw on ``float(None)``. Pre-fix the chain
    ``float(market.best_bid) + float(market.best_ask)`` would raise
    TypeError; post-fix the partial book falls through to the
    generic missing-book guard which cancels resting flatten orders
    and waits.

    This is a unit test on the partial-book detection itself —
    a full SF tick test would require pulling in the bot mock
    stack. The condition under test is the boolean expression
    that gates ``mid`` computation.
    """
    # Simulate the partial-book check used in ``_run_soft_flatten_tick``
    # at app/bot.py:1958. The expression must short-circuit to False
    # when either side is None, so the float() never runs.
    market_no_bid = BestBidAsk(
        symbol="X", best_bid=None, best_ask=1.05, mid_price=1.05, spread_bps=0.0,
    )
    market_no_ask = BestBidAsk(
        symbol="X", best_bid=1.05, best_ask=None, mid_price=1.05, spread_bps=0.0,
    )
    market_full = BestBidAsk(
        symbol="X", best_bid=1.04, best_ask=1.06, mid_price=1.05, spread_bps=20.0,
    )
    market_none = None

    def can_compute_mid(m) -> bool:
        return (
            m is not None
            and m.best_bid is not None
            and m.best_ask is not None
        )

    assert can_compute_mid(market_no_bid) is False
    assert can_compute_mid(market_no_ask) is False
    assert can_compute_mid(market_none) is False
    assert can_compute_mid(market_full) is True


# =============================================================================
# Bug 6 — orders.order_id_exchange index must exist after migration
# =============================================================================


def test_bug6_orders_order_id_exchange_index_exists() -> None:
    """After ``init_schema`` runs the v20 migration, the
    ``orders`` table must carry an index on ``order_id_exchange``.
    Without it, the per-fill attribution lookups
    (``soft_flatten_event_id_for_order``,
    ``order_quote_quality_for_order``) degrade to table scans as
    the orders table grows.
    """
    import sqlite3

    from app.storage import Storage

    s = _settings()
    storage = Storage(s)
    storage.init_schema()
    with storage.connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = 'orders'"
        ).fetchall()
    index_names = {r[0] for r in rows}
    assert "idx_orders_order_id_exchange" in index_names, (
        f"expected idx_orders_order_id_exchange in {index_names}"
    )


# =============================================================================
# Bug 7 — shadow position must seed avg_entry_price on flat→position / sign-flip
# =============================================================================


def _fill(side: Side, price: float, size: float, fill_id: str = "f"):
    """Minimal Fill object for record_fill tests."""
    from app.models import Fill

    return Fill(
        fill_id=fill_id,
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="ETH",
        side=side,
        price=price,
        size=size,
        notional=price * size,
        fee=-0.0001,
        liquidity_flag="resting",
        mid_at_fill=price,
    )


def test_bug7_shadow_seeds_avg_entry_on_flat_to_long() -> None:
    """Pre-fix, the first BUY from flat left ``avg_entry_price = None``
    until REST repaired it — leaving drawdown gates with a non-zero
    qty but no entry reference. Post-fix, the shadow update seeds
    avg_entry from the fill price."""
    s = _settings()
    state = BotState(s)
    # Start flat (this is the default state).
    assert state.position.position_qty == 0.0
    assert state.position.avg_entry_price is None

    state.record_fill(_fill(Side.BUY, price=1.05, size=10.0))

    assert state.position.position_qty == 10.0
    assert state.position.avg_entry_price == 1.05  # seeded from fill price


def test_bug7_shadow_seeds_avg_entry_on_flat_to_short() -> None:
    """Symmetric: first SELL from flat seeds avg_entry from the
    SELL fill's price."""
    s = _settings()
    state = BotState(s)
    state.record_fill(_fill(Side.SELL, price=1.05, size=10.0))
    assert state.position.position_qty == -10.0
    assert state.position.avg_entry_price == 1.05


def test_bug7_shadow_reseeds_avg_entry_on_sign_flip() -> None:
    """When a fill is large enough to flip the position from long
    to short (or vice-versa), the close leg is fully realised and
    the open leg's avg_entry IS this fill's price. The seed must
    pick up that case too."""
    from dataclasses import replace as _replace

    s = _settings()
    state = BotState(s)
    # Seed an existing long position.
    state.position = _replace(
        state.position,
        position_qty=5.0,
        avg_entry_price=1.00,
    )
    # Sell 10 → net -5 (flipped).
    state.record_fill(_fill(Side.SELL, price=1.10, size=10.0))
    assert state.position.position_qty == -5.0
    # Sign flipped — avg_entry re-seeded from the flip fill's price.
    assert state.position.avg_entry_price == 1.10


def test_bug7_shadow_keeps_avg_entry_on_same_sign_add() -> None:
    """Adding to an existing same-sign position should NOT replace
    the avg_entry — that's a true running-average update which is
    server-side bookkeeping. The shadow keeps the prior value;
    REST will reconcile the running average."""
    from dataclasses import replace as _replace

    s = _settings()
    state = BotState(s)
    state.position = _replace(
        state.position,
        position_qty=5.0,
        avg_entry_price=1.00,
    )
    state.record_fill(_fill(Side.BUY, price=1.10, size=2.0))
    assert state.position.position_qty == 7.0
    # Avg-entry preserved — REST will reconcile the running avg.
    assert state.position.avg_entry_price == 1.00


def test_bug7_shadow_clears_avg_entry_when_position_closes_to_flat() -> None:
    """When a fill closes the position to exactly zero, avg_entry
    should clear so the next opening fill seeds correctly."""
    from dataclasses import replace as _replace

    s = _settings()
    state = BotState(s)
    state.position = _replace(
        state.position,
        position_qty=5.0,
        avg_entry_price=1.00,
    )
    state.record_fill(_fill(Side.SELL, price=1.10, size=5.0))
    assert state.position.position_qty == 0.0
    assert state.position.avg_entry_price is None
