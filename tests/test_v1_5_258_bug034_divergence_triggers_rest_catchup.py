"""BUG-034 / v1.5.258 — shadow/REST divergence triggers REST fill catch-up.

Pre-fix, ``apply_account_position_only`` and ``apply_market_snapshot``
silently overwrote ``self.position`` from the REST snapshot whenever a
divergence was detected, without doing anything to recover the missing
fill record(s). The lost fills are permanently absent from
``fills.jsonl``, ``recent_fills``, ``session_fill_count``, markout
sampling, PnL attribution, the fill-burst detector, and the operator's
inventory chart — see issues/bug-034.md for the reproduction trace.

Post-fix, the same path sets ``private_ws_recovery_pending = True`` so
the next tick's ``OrderManager.should_ingest_fills_via_rest()`` returns
True and the bot fetches ``/fills`` from REST to catch up. The
``shadow_position_divergence_recovery_count`` counter increments
alongside ``shadow_position_divergence_count`` so the operator can
verify the trigger is firing.

These tests verify only the state-side trigger logic — that divergence
sets the recovery flag, increments both counters, and that small
sub-threshold divergences (≤ 0.5 lot) do NOT trip the trigger. The
end-to-end REST catch-up flow is tested by the existing
``test_market_data.py`` fixtures.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


def _make_state():
    """Construct a minimal BotState for the tests in this module."""
    from app.config import Settings
    from app.state import BotState
    s = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
    )
    return BotState(settings=s)


def _make_position(qty: float):
    """Construct a PositionSnapshot with the given qty (TON-USDT-SWAP)."""
    from app.models import PositionSnapshot
    return PositionSnapshot(
        symbol="TON-USDT-SWAP",
        position_qty=qty,
        avg_entry_price=1.744,
        mark_price=1.744,
        position_notional=abs(qty) * 1.744,
        unrealized_pnl_usd=0.0,
        ts_local=datetime.now(timezone.utc),
    )


def _make_market():
    """Construct a BestBidAsk with a tight book."""
    from app.models import BestBidAsk
    return BestBidAsk(
        symbol="TON-USDT-SWAP",
        best_bid=1.744,
        best_ask=1.745,
        mid_price=1.7445,
        spread_bps=5.73,
        ts_local=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# apply_account_position_only — the post-startup steady-state path
# ---------------------------------------------------------------------------

def test_account_position_only_no_divergence_does_not_trigger_recovery():
    """Sub-threshold drift (≤ 0.5 lot) is treated as rounding noise."""
    state = _make_state()
    state.position = _make_position(0.0)
    state.private_ws_recovery_pending = False

    # 0.3 lot drift — below the 0.5 threshold.
    state.apply_account_position_only(_make_position(0.3), None)

    assert state.shadow_position_divergence_count == 0
    assert state.shadow_position_divergence_recovery_count == 0
    assert state.private_ws_recovery_pending is False
    # Position still overwritten (REST is authoritative even on small drift).
    assert state.position.position_qty == pytest.approx(0.3)


def test_account_position_only_divergence_triggers_recovery():
    """Above-threshold divergence sets the recovery flag + counter."""
    state = _make_state()
    state.position = _make_position(0.0)
    state.private_ws_recovery_pending = False

    # The exact scenario from issues/bug-034.md (20:14:28 trace):
    # shadow=0, REST=-3 → delta=-3, definitely above the 0.5 threshold.
    state.apply_account_position_only(_make_position(-3.0), None)

    assert state.shadow_position_divergence_count == 1
    assert state.shadow_position_divergence_recovery_count == 1
    assert state.shadow_position_last_divergence_qty == pytest.approx(-3.0)
    assert state.private_ws_recovery_pending is True
    # REST authoritative — position overwritten.
    assert state.position.position_qty == pytest.approx(-3.0)


def test_account_position_only_divergence_in_opposite_direction():
    """Symmetric: shadow over-counted vs. REST under-counted both trigger."""
    state = _make_state()
    state.position = _make_position(-6.0)
    state.private_ws_recovery_pending = False

    # The second divergence from the bug trace (20:14:36):
    # shadow=-6, REST=-3 → delta=+3 (BUY 3 missed by WS).
    state.apply_account_position_only(_make_position(-3.0), None)

    assert state.shadow_position_divergence_count == 1
    assert state.shadow_position_divergence_recovery_count == 1
    assert state.shadow_position_last_divergence_qty == pytest.approx(3.0)
    assert state.private_ws_recovery_pending is True


def test_repeated_divergences_accumulate_counters():
    """Each divergence bumps both counters by exactly 1."""
    state = _make_state()
    state.position = _make_position(0.0)
    state.private_ws_recovery_pending = False

    state.apply_account_position_only(_make_position(-3.0), None)  # +1
    state.apply_account_position_only(_make_position(-6.0), None)  # +1
    state.apply_account_position_only(_make_position(-3.0), None)  # +1

    assert state.shadow_position_divergence_count == 3
    assert state.shadow_position_divergence_recovery_count == 3
    # Last delta is from the third call: shadow=-6, REST=-3 → +3.
    assert state.shadow_position_last_divergence_qty == pytest.approx(3.0)
    assert state.private_ws_recovery_pending is True


# ---------------------------------------------------------------------------
# apply_market_snapshot — the bootstrap/init path
# ---------------------------------------------------------------------------

def test_market_snapshot_divergence_triggers_recovery():
    """Same trigger fires on the market-snapshot path."""
    state = _make_state()
    state.position = _make_position(0.0)
    state.private_ws_recovery_pending = False

    state.apply_market_snapshot(
        _make_market(), _make_position(-3.0), None,
    )

    assert state.shadow_position_divergence_count == 1
    assert state.shadow_position_divergence_recovery_count == 1
    assert state.private_ws_recovery_pending is True


def test_market_snapshot_no_divergence_does_not_trigger():
    """Identical positions on both paths are a no-op for divergence."""
    state = _make_state()
    state.position = _make_position(-3.0)
    state.private_ws_recovery_pending = False

    state.apply_market_snapshot(
        _make_market(), _make_position(-3.0), None,
    )

    assert state.shadow_position_divergence_count == 0
    assert state.shadow_position_divergence_recovery_count == 0
    assert state.private_ws_recovery_pending is False


# ---------------------------------------------------------------------------
# Counter publishing — operator visibility via state_current.json
# ---------------------------------------------------------------------------

def test_divergence_counters_surfaced_in_snapshot_dict():
    """snapshot_dict (which generates state_current.json) publishes
    the new counters at the TOP LEVEL alongside the other connectivity
    counters — that's where the acceptance check reads them. v1.5.267
    fix: was originally in live_stats.py but state_current.json is
    generated by snapshot_dict, not live_stats.
    """
    state = _make_state()
    state.position = _make_position(0.0)

    # Fire one divergence so the counter is non-zero.
    state.apply_account_position_only(_make_position(-3.0), None)
    assert state.shadow_position_divergence_count == 1

    snap = state.snapshot_dict()

    # Counters live at the TOP LEVEL of snapshot_dict — same place
    # cancel_unexpected_gone_total and friends live. NOT in any
    # connectivity_counters sub-dict (snapshot_dict has no such key).
    assert "shadow_position_divergence_count" in snap
    assert snap["shadow_position_divergence_count"] == 1
    assert "shadow_position_divergence_recovery_count" in snap
    assert snap["shadow_position_divergence_recovery_count"] == 1
    assert "shadow_position_last_divergence_qty" in snap
    assert snap["shadow_position_last_divergence_qty"] == pytest.approx(-3.0)


def test_divergence_counters_surfaced_in_live_stats():
    """live_stats publishes the new counters so they show in the
    S3 live_stats payload (separate from state_current.json which
    snapshot_dict owns)."""
    from app.config import Settings
    from app.live_stats import LiveStatsPublisher
    state = _make_state()
    state.position = _make_position(0.0)

    # Fire one divergence so the counter is non-zero.
    state.apply_account_position_only(_make_position(-3.0), None)
    assert state.shadow_position_divergence_count == 1

    settings = Settings(
        VENUE="okx", SYMBOL="TON-USDT-SWAP",
        QUOTE_NOTIONAL_USD=7.0, MIN_QUOTE_NOTIONAL_USD=5.0,
        MAX_ABS_POSITION=6.0,
    )
    publisher = LiveStatsPublisher(
        settings=settings,
        state=state,
        bucket="test",
        profile_name="test",
    )
    snap = publisher._build_payload()

    conn = snap.get("connectivity_counters") or {}
    assert "shadow_position_divergence_count" in conn
    assert conn["shadow_position_divergence_count"] == 1
    assert "shadow_position_divergence_recovery_count" in conn
    assert conn["shadow_position_divergence_recovery_count"] == 1
    assert "shadow_position_last_divergence_qty" in conn
    assert conn["shadow_position_last_divergence_qty"] == pytest.approx(-3.0)
