"""Tests for the fill-derived shadow position update.

Pre-2026-05-06, ``state.position.position_qty`` was only updated by
REST refresh (8s healthy / 12s unhealthy / 20s under uncertainty).
Between refreshes, the bot read STALE position state. The
2026-05-06 OKX SUI runaway used 7+ ticks of stale position to place
repeated BUY orders against a venue that had already filled them.

Fix: every session-scoped ``record_fill`` call now applies the
signed delta to ``state.position.position_qty`` in real time. REST
refresh remains AUTHORITATIVE; the shadow update is just the
"close the staleness gap between refreshes" layer.

These tests pin the contract:
  1. Every session-scoped fill applies a signed delta.
  2. Replay (non-session-scoped) fills do NOT apply.
  3. Idempotent fill_id de-dup prevents double-application.
  4. REST refresh overwrites the shadow.
  5. REST refresh logs divergence when shadow drifted from truth.
  6. Notional is recomputed from the new qty.
  7. Counter increments per applied fill.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.enums import Side
from app.models import AccountSnapshot, Fill, PositionSnapshot
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings(**kw) -> UnitTestSettings:
    path = Path(tempfile.gettempdir()) / f"mm_shadow_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
    }
    base.update(kw)
    return UnitTestSettings.model_validate(base)


def _make_fill(
    fill_id: str,
    side: Side,
    size: float,
    price: float = 1.0,
) -> Fill:
    return Fill(
        fill_id=fill_id,
        order_id_exchange=12345,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="TEST-USDT-SWAP",
        side=side,
        price=price,
        size=size,
        notional=size * price,
        fee=0.0,
        liquidity_flag="resting",
        mid_at_fill=price,
        best_bid_at_fill=price - 0.0001,
        best_ask_at_fill=price + 0.0001,
        book_snapshot_quality="full",
    )


# ---------------------------------------------------------------------------
# Core contract
# ---------------------------------------------------------------------------


def test_buy_fill_increases_position_qty() -> None:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    f = _make_fill("f1", Side.BUY, size=10.0, price=1.0)
    assert state.record_fill(f) is True
    assert state.position.position_qty == pytest.approx(10.0)
    assert state.shadow_position_apply_count == 1


def test_sell_fill_decreases_position_qty() -> None:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=10.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=10.0,
        unrealized_pnl_usd=0.0,
    )
    f = _make_fill("f1", Side.SELL, size=4.0, price=1.0)
    assert state.record_fill(f) is True
    assert state.position.position_qty == pytest.approx(6.0)


def test_fill_can_cross_zero_long_to_short() -> None:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=5.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=5.0,
        unrealized_pnl_usd=0.0,
    )
    # Big SELL: 5 long → 5 - 12 = -7 short
    f = _make_fill("f1", Side.SELL, size=12.0, price=1.0)
    state.record_fill(f)
    assert state.position.position_qty == pytest.approx(-7.0)


def test_replay_fill_does_not_apply_shadow() -> None:
    """``session_scoped=False`` is the historical replay path. It
    must not double-apply when a fill is being re-ingested at
    startup that the venue position already reflects."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    f = _make_fill("f1", Side.BUY, size=10.0)
    state.record_fill(f, session_scoped=False)
    assert state.position.position_qty == pytest.approx(0.0)
    assert state.shadow_position_apply_count == 0


def test_idempotent_fill_id_prevents_double_apply() -> None:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    f = _make_fill("dup-id", Side.BUY, size=10.0)
    assert state.record_fill(f) is True
    assert state.record_fill(f) is False  # second call is no-op
    assert state.position.position_qty == pytest.approx(10.0)
    assert state.shadow_position_apply_count == 1


# ---------------------------------------------------------------------------
# Multiple fills snowball reproduction
# ---------------------------------------------------------------------------


def test_multiple_buys_grow_position_each_fill() -> None:
    """The 2026-05-06 incident: 11 BUY 21 fills at ~0.5s intervals.
    Pre-fix, position state would have stayed at -21 across all of
    them. Post-fix, each fill increments shadow position so the
    bot's risk gates see growing position immediately."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=-21.0,  # short, soft-flatten about to fire BUY
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=21.0,
        unrealized_pnl_usd=0.0,
    )
    for i in range(11):
        f = _make_fill(f"f{i}", Side.BUY, size=21.0, price=1.0)
        state.record_fill(f)
    # Position should be -21 + 11×21 = 210 long
    assert state.position.position_qty == pytest.approx(210.0)
    assert state.shadow_position_apply_count == 11


def test_alternating_fills_track_each_other() -> None:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    # +5, -3, +7, -10
    state.record_fill(_make_fill("a", Side.BUY, 5.0))
    state.record_fill(_make_fill("b", Side.SELL, 3.0))
    state.record_fill(_make_fill("c", Side.BUY, 7.0))
    state.record_fill(_make_fill("d", Side.SELL, 10.0))
    # 0 + 5 - 3 + 7 - 10 = -1
    assert state.position.position_qty == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Notional recomputation
# ---------------------------------------------------------------------------


def test_notional_recomputed_after_fill_using_mark() -> None:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=2.0,  # different from fill price
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    state.record_fill(_make_fill("f1", Side.BUY, 10.0, price=1.0))
    # New qty = 10, notional uses mark (2.0): 10 × 2.0 = 20
    assert state.position.position_notional == pytest.approx(20.0)


def test_notional_falls_back_to_avg_entry_when_no_mark() -> None:
    """When the existing position is flat (qty=0) but carries a STALE
    ``avg_entry_price`` from a prior closed position, a new fill from
    flat re-seeds ``avg_entry`` from the fill price (Codex MED-7 fix,
    1.1.134). The notional therefore uses the fresh entry, not the
    stale one — pre-fix the old test asserted ``10 × 3 = 30`` which
    was exactly the blind-spot Codex flagged.
    """
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=3.0,  # stale leftover from a prior closed position
        mark_price=None,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    state.record_fill(_make_fill("f1", Side.BUY, 10.0, price=1.0))
    # Flat → long: avg_entry re-seeded from fill price (1.0).
    # No mark → notional uses the freshly-seeded avg_entry: 10 × 1 = 10.
    assert state.position.avg_entry_price == pytest.approx(1.0)
    assert state.position.position_notional == pytest.approx(10.0)


def test_notional_uses_avg_entry_when_adding_to_existing_position_no_mark() -> None:
    """When a same-sign fill ADDS to an existing position (not a
    flat→position transition), the prior ``avg_entry`` is preserved
    (server-side reconciles the running average; the bot doesn't
    mirror it). With no mark, notional uses that preserved avg_entry.
    """
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=5.0,  # already long
        avg_entry_price=3.0,
        mark_price=None,
        position_notional=15.0,
        unrealized_pnl_usd=0.0,
    )
    state.record_fill(_make_fill("f1", Side.BUY, 5.0, price=1.0))
    # Adding to long → avg_entry preserved at 3.0.
    # No mark → notional = 10 × 3 = 30.
    assert state.position.avg_entry_price == pytest.approx(3.0)
    assert state.position.position_notional == pytest.approx(30.0)


def test_notional_falls_back_to_fill_price_when_neither_set() -> None:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=None,
        mark_price=None,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    state.record_fill(_make_fill("f1", Side.BUY, 10.0, price=1.5))
    # Neither mark nor entry → use fill price (1.5): notional = 15
    assert state.position.position_notional == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# REST refresh authority + divergence detection
# ---------------------------------------------------------------------------


def test_rest_refresh_overwrites_shadow_position() -> None:
    """REST refresh is the AUTHORITATIVE truth; shadow is best-
    effort gap-closer. Test: simulate a fill that the bot processed,
    then a REST refresh that disagrees (e.g. funding adjustment, or
    a fill the bot missed). REST wins."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    state.record_fill(_make_fill("f1", Side.BUY, 10.0))
    assert state.position.position_qty == 10.0
    # REST says position is actually 7 (e.g. partial liquidation).
    state.apply_account_position_only(
        PositionSnapshot(
            symbol="TEST-USDT-SWAP",
            position_qty=7.0,
            avg_entry_price=1.0,
            mark_price=1.0,
            position_notional=7.0,
            unrealized_pnl_usd=0.0,
        ),
        AccountSnapshot(
            equity_usd=1000.0, cash_usd=1000.0, withdrawable_usd=1000.0
        ),
    )
    assert state.position.position_qty == 7.0


def test_divergence_counter_increments_on_drift() -> None:
    """When REST refresh disagrees with shadow by > 0.5 lots,
    divergence counter ticks up so operators can see it."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    state.record_fill(_make_fill("f1", Side.BUY, 10.0))  # shadow → 10
    # REST says 12 (we missed a fill of 2)
    state.apply_account_position_only(
        PositionSnapshot(
            symbol="TEST-USDT-SWAP",
            position_qty=12.0,
            avg_entry_price=1.0,
            mark_price=1.0,
            position_notional=12.0,
            unrealized_pnl_usd=0.0,
        ),
        AccountSnapshot(
            equity_usd=1000.0, cash_usd=1000.0, withdrawable_usd=1000.0
        ),
    )
    assert state.shadow_position_divergence_count == 1
    assert state.shadow_position_last_divergence_qty == pytest.approx(2.0)


def test_divergence_counter_does_not_increment_on_dust() -> None:
    """0.5-lot threshold avoids noise from rounding / timing."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    state.record_fill(_make_fill("f1", Side.BUY, 10.0))  # shadow → 10
    # REST says 10.3 (rounding)
    state.apply_account_position_only(
        PositionSnapshot(
            symbol="TEST-USDT-SWAP",
            position_qty=10.3,
            avg_entry_price=1.0,
            mark_price=1.0,
            position_notional=10.3,
            unrealized_pnl_usd=0.0,
        ),
        AccountSnapshot(
            equity_usd=1000.0, cash_usd=1000.0, withdrawable_usd=1000.0
        ),
    )
    assert state.shadow_position_divergence_count == 0


# ---------------------------------------------------------------------------
# Snowball-prevention regression
# ---------------------------------------------------------------------------


def test_snowball_scenario_position_visible_after_each_fill() -> None:
    """Replays the snowball mechanic: BUY fills land in rapid
    succession. Each fill is immediately reflected in
    ``state.position.position_qty`` so the next quote-eligibility
    check (or soft-flatten worker) sees correct position. Pre-fix,
    state would say -21 across all 11 fills. Post-fix, it grows
    monotonically."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=-21.0,
        avg_entry_price=1.0,
        mark_price=1.007,
        position_notional=21.0,
        unrealized_pnl_usd=0.0,
    )
    snapshot_qtys = []
    for i in range(11):
        state.record_fill(_make_fill(f"f{i}", Side.BUY, 21.0, price=1.007))
        snapshot_qtys.append(state.position.position_qty)
    # Position should monotonically increase: -21 → 0 → 21 → 42 → ... → 210
    expected = [-21 + 21 * (i + 1) for i in range(11)]
    assert snapshot_qtys == pytest.approx(expected)


# ---------------------------------------------------------------------------
# REST catch-up double-apply regression (Codex review 2026-05-08, 1.1.36)
# ---------------------------------------------------------------------------
#
# The shadow-position update was added to ``record_fill()`` and
# ``apply_account_position_only()`` was left unchanged. This left a
# correctness bug in ``refresh_account_only()``:
#
#   1. REST snapshot installs venue-truth position via
#      ``apply_account_position_only(pos, acct)``.
#   2. Then iterates the missed-fills list and calls ``record_fill()``
#      for each, which (pre-fix) shadow-updated ``position_qty`` by
#      the fill's signed delta.
#
# The position_qty installed in step 1 ALREADY reflects every fill
# on the venue (it's a snapshot of "where we are now"). Adding the
# fills again in step 2 double-counts them locally. Until the next
# refresh, position state is wrong.
#
# 1.1.36 fix: ``record_fill`` accepts ``shadow_update_position`` kwarg
# (default True for legacy callers). REST catch-up paths pass False
# so session metrics still update but the position delta is not
# re-applied.


def test_rest_catchup_does_not_double_apply_missed_fills() -> None:
    """REST catch-up scenario: bot's local position is stale (private
    WS missed a SELL 3). REST snapshot installs venue truth (-3), then
    iterates the missed fills and re-records them. Position should
    end up at -3 (matching venue), NOT -6 (double-applied).
    """
    state = BotState(_settings())
    # Local stale state: bot saw 0 last but actually got filled SELL 3.
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=None,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    # Step 1: REST snapshot lands the venue truth.
    venue_truth = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=-3.0,
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=3.0,
        unrealized_pnl_usd=0.0,
    )
    venue_acct = AccountSnapshot(
        equity_usd=1000.0, cash_usd=1000.0, withdrawable_usd=1000.0
    )
    state.apply_account_position_only(venue_truth, venue_acct)
    assert state.position.position_qty == pytest.approx(-3.0)
    # Step 2: REST catch-up replays the missed fill.
    # WITH the 1.1.36 fix: shadow_update_position=False prevents
    # double-apply. Position stays at -3.
    state.record_fill(
        _make_fill("missed-sell-1", Side.SELL, 3.0, price=1.0),
        shadow_update_position=False,
    )
    assert state.position.position_qty == pytest.approx(-3.0), (
        "REST catch-up double-applied a missed fill — pre-1.1.36 bug "
        "regressed (position would be -6.0)"
    )
    # Session metrics still updated (the fill was processed for
    # bookkeeping) — only the shadow-position write was suppressed.
    assert state.session_fill_count == 1


def test_legacy_record_fill_default_still_shadow_updates() -> None:
    """The 1.1.36 kwarg defaults to True so existing callers
    (private-WS event handler, all production paths except REST
    catch-up) keep their previous behaviour: shadow-update on every
    session-scoped fill."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=0.0,
        avg_entry_price=None,
        mark_price=1.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
    )
    # No kwarg → default True → shadow update applies.
    state.record_fill(_make_fill("legacy-buy", Side.BUY, 5.0, price=1.0))
    assert state.position.position_qty == pytest.approx(5.0)


def test_shadow_update_position_false_skips_only_position_not_metrics() -> None:
    """When ``shadow_update_position=False`` the fill is recorded for
    session metrics (fill_count, recent_fills, traded_notional) but
    the position is NOT mutated. Verifies the gate is surgical, not
    a full skip."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=10.0,  # arbitrary stale value
        avg_entry_price=1.0,
        mark_price=1.0,
        position_notional=10.0,
        unrealized_pnl_usd=0.0,
    )
    pre_traded_ntn = state.session_traded_notional_usd
    pre_fill_count = state.session_fill_count
    state.record_fill(
        _make_fill("rest-catchup-1", Side.BUY, 7.0, price=2.0),
        shadow_update_position=False,
    )
    # Position untouched.
    assert state.position.position_qty == pytest.approx(10.0)
    # Session metrics did update.
    assert state.session_fill_count == pre_fill_count + 1
    assert state.session_traded_notional_usd > pre_traded_ntn
    # recent_fills got the fill (visible to dashboards / UI).
    assert any(f.fill_id == "rest-catchup-1" for f in state.recent_fills)


# ---------------------------------------------------------------------------
# Shadow unrealized PnL (Codex review 2026-05-07, 1.1.37 HIGH-3)
# ---------------------------------------------------------------------------
#
# Pre-1.1.37, ``record_fill`` updated ``position_qty`` and
# ``position_notional`` but left ``unrealized_pnl_usd`` stale, so
# drawdown and session-loss gates ran against the unrealized number
# from the pre-fill snapshot until the next REST refresh — masking
# risk by up to the account-refresh interval under fast fill bursts.
# The fix is best-effort: needs both mark + avg_entry available.
# ``avg_entry_price`` is server-side; we leave it untouched and let
# REST refresh own it.


def test_shadow_unrealized_recomputed_on_buy_fill() -> None:
    """Long position grows on BUY: unrealized recomputed against
    new qty using existing mark + avg_entry."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=10.0,
        avg_entry_price=1.0,
        mark_price=1.10,
        position_notional=11.0,
        unrealized_pnl_usd=1.0,  # (1.10 - 1.0) * 10
    )
    # BUY 5 at any price; mark + avg_entry are what drive PnL.
    state.record_fill(_make_fill("f1", Side.BUY, 5.0, price=1.10))
    # New qty = 15. Expected unrealized = (1.10 - 1.0) * 15 = 1.5
    assert state.position.position_qty == pytest.approx(15.0)
    assert state.position.unrealized_pnl_usd == pytest.approx(1.5)


def test_shadow_unrealized_recomputed_on_short_close() -> None:
    """Short position partial-close on BUY: unrealized recomputed
    against the smaller residual."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=-10.0,
        avg_entry_price=2.0,
        mark_price=1.5,  # short position is in profit
        position_notional=15.0,
        unrealized_pnl_usd=5.0,  # (1.5 - 2.0) * -10 = +5
    )
    # BUY 4: short shrinks to -6.
    state.record_fill(_make_fill("f1", Side.BUY, 4.0, price=1.5))
    # New qty = -6. Expected unrealized = (1.5 - 2.0) * -6 = +3
    assert state.position.position_qty == pytest.approx(-6.0)
    assert state.position.unrealized_pnl_usd == pytest.approx(3.0)


def test_shadow_unrealized_zeroed_when_position_closes_to_zero() -> None:
    """Closing the position to exactly zero zeroes unrealized.
    Without this branch, a residual stale unrealized would leak
    into PnL reporting between the fill and the next REST refresh."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=5.0,
        avg_entry_price=1.0,
        mark_price=1.20,
        position_notional=6.0,
        unrealized_pnl_usd=1.0,  # (1.20 - 1.0) * 5
    )
    # SELL 5: position → 0.
    state.record_fill(_make_fill("f1", Side.SELL, 5.0, price=1.20))
    assert state.position.position_qty == pytest.approx(0.0)
    assert state.position.unrealized_pnl_usd == pytest.approx(0.0)


def test_shadow_unrealized_unchanged_when_mark_or_entry_missing() -> None:
    """Best-effort: if mark or avg_entry isn't known, leave the
    pre-fill unrealized in place and let REST repair it. Better than
    writing a wrong-zero or arbitrary number."""
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="TEST-USDT-SWAP",
        position_qty=10.0,
        avg_entry_price=None,  # no entry price yet
        mark_price=1.10,
        position_notional=11.0,
        unrealized_pnl_usd=1.0,  # legacy snapshot value
    )
    state.record_fill(_make_fill("f1", Side.BUY, 5.0, price=1.10))
    # qty updated, but unrealized untouched (preserve legacy value).
    assert state.position.position_qty == pytest.approx(15.0)
    assert state.position.unrealized_pnl_usd == pytest.approx(1.0)
