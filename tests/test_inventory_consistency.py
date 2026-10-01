"""TODO-001: inventory consistency watchdog.

Three-way invariant: ``state.position.position_qty`` (venue truth) must
agree with ``inventory_baseline_qty + session_signed_qty_total`` (the
bot's view of what should have happened) within tolerance. Catches
BUG-005-class divergences (where local state silently drifted from
venue) before they accumulate.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from app.enums import Side
from app.inventory_consistency import (
    InventoryBreach,
    check_inventory_consistency,
)
from app.models import Fill, PositionSnapshot
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "EXCHANGE": "grvt",
        "SYMBOL": "ETH_USDT_Perp",
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_state(*, baseline_qty: float, venue_qty: float) -> BotState:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="ETH_USDT_Perp",
        position_qty=venue_qty,
        avg_entry_price=100.0,
        mark_price=100.0,
        position_notional=abs(venue_qty * 100.0),
        unrealized_pnl_usd=0.0,
    )
    state.set_inventory_baseline_at_session_start(baseline_qty)
    # Force the rate-limit anchor into the past so the first call evaluates.
    state.inventory_consistency_last_check_mono = None
    return state


def _bump(state: BotState, side: Side, size: float) -> None:
    """Drive ``session_signed_qty_total`` via the public API."""
    f = Fill(
        fill_id=f"f-{time.monotonic_ns()}",
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="ETH_USDT_Perp",
        side=side,
        price=100.0,
        size=size,
        notional=size * 100.0,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=100.0,
    )
    state.bump_operator_metrics_on_session_fill(f, 0.0)


def test_check_returns_none_when_disabled() -> None:
    state = _make_state(baseline_qty=0.0, venue_qty=100.0)
    settings = _settings(INVENTORY_CONSISTENCY_ENABLED=False)
    assert check_inventory_consistency(state, settings) is None


def test_check_returns_none_when_baseline_not_set() -> None:
    state = BotState(_settings())
    state.position = PositionSnapshot(
        symbol="ETH_USDT_Perp",
        position_qty=100.0,
        avg_entry_price=100.0,
        mark_price=100.0,
        position_notional=10000.0,
        unrealized_pnl_usd=0.0,
    )
    # No set_inventory_baseline_at_session_start call.
    settings = _settings()
    assert check_inventory_consistency(state, settings) is None


def test_check_returns_none_when_venue_matches_expected() -> None:
    """Healthy: venue_qty == baseline + session_signed_qty_total."""
    state = _make_state(baseline_qty=10.0, venue_qty=15.0)
    _bump(state, Side.BUY, 5.0)
    settings = _settings()
    breach = check_inventory_consistency(state, settings)
    assert breach is None


def test_check_within_tolerance_qty_returns_none() -> None:
    """Drift below the absolute-qty tolerance does not breach."""
    state = _make_state(baseline_qty=0.0, venue_qty=0.3)  # 0.3 SUI drift
    settings = _settings(INVENTORY_CONSISTENCY_TOLERANCE_QTY=0.5)
    assert check_inventory_consistency(state, settings) is None


def test_check_within_tolerance_pct_returns_none_on_large_book() -> None:
    """Drift below the percentage tolerance does not breach (large book)."""
    state = _make_state(baseline_qty=100.0, venue_qty=100.5)
    settings = _settings(
        INVENTORY_CONSISTENCY_TOLERANCE_QTY=0.0,
        INVENTORY_CONSISTENCY_TOLERANCE_PCT=0.01,  # 1% of 100 = 1.0 tolerance
    )
    assert check_inventory_consistency(state, settings) is None


def test_check_breach_returns_payload_on_drift() -> None:
    """BUG-005 reproducer: venue=200, expected=0 → breach.

    Uses ``CONSECUTIVE_BREACHES_REQUIRED=1`` to fire on the first sample;
    the default is 3 and the per-sample-count behaviour has its own
    dedicated tests below.
    """
    state = _make_state(baseline_qty=0.0, venue_qty=200.0)
    settings = _settings(
        INVENTORY_CONSISTENCY_CONSECUTIVE_BREACHES_REQUIRED=1,
    )
    breach = check_inventory_consistency(state, settings)
    assert isinstance(breach, InventoryBreach)
    assert breach.venue_qty == 200.0
    assert breach.expected_qty == 0.0
    assert breach.drift == 200.0
    assert state.inventory_consistency_breach_count == 1
    assert state.inventory_consistency_last_breach is not None
    assert state.inventory_consistency_last_breach["drift"] == 200.0


def test_check_rate_limited_within_interval() -> None:
    """Two back-to-back calls inside the check interval — only the first
    runs the comparison.

    NOTE: After the consecutive-breaches hardening the FIRST call now
    only bumps the consecutive counter (1 < 3 default); the breach
    payload returns once the threshold is met. The rate-limit semantics
    are independent of the threshold check, so we use
    ``CONSECUTIVE_BREACHES_REQUIRED=1`` here to keep the original
    invariant focused on the rate-limit gate.
    """
    state = _make_state(baseline_qty=0.0, venue_qty=200.0)
    settings = _settings(
        INVENTORY_CONSISTENCY_CHECK_SECONDS=60.0,
        INVENTORY_CONSISTENCY_CONSECUTIVE_BREACHES_REQUIRED=1,
    )
    b1 = check_inventory_consistency(state, settings)
    b2 = check_inventory_consistency(state, settings)
    assert b1 is not None
    assert b2 is None
    assert state.inventory_consistency_breach_count == 1


# --- consecutive-breaches hardening (post-snap_20260426_103520) -----------


def test_single_transient_divergence_does_not_breach() -> None:
    """A single bad ``fetch_position`` read (the snap_20260426_103520
    scenario where the venue briefly returned -8 instead of -28) must
    NOT fire the watchdog at the default 3-consecutive threshold.
    """
    state = _make_state(baseline_qty=-28.0, venue_qty=-8.0)  # +20 drift
    settings = _settings()  # default required=3
    b = check_inventory_consistency(state, settings)
    assert b is None
    assert state.inventory_consistency_consecutive_drift_count == 1
    assert state.inventory_consistency_breach_count == 0


def test_clean_read_resets_consecutive_counter() -> None:
    """One bad read followed by a good read does NOT carry the bad sample
    forward — only sustained divergence counts.
    """
    state = _make_state(baseline_qty=-28.0, venue_qty=-8.0)
    settings = _settings()
    # First: bad read, counter=1
    check_inventory_consistency(state, settings)
    assert state.inventory_consistency_consecutive_drift_count == 1
    # Recovery: good read clears the counter (allow next call past rate limit).
    state.inventory_consistency_last_check_mono = None
    state.position.position_qty = -28.0  # API now returns the real position
    check_inventory_consistency(state, settings)
    assert state.inventory_consistency_consecutive_drift_count == 0


def test_three_consecutive_divergent_reads_fires_breach() -> None:
    """Sustained divergence — three consecutive bad reads with default
    threshold = 3 — DOES declare a breach. This protects against the
    real BUG-005-class bugs the watchdog was built for.
    """
    state = _make_state(baseline_qty=-28.0, venue_qty=200.0)  # huge drift
    settings = _settings()  # required=3
    # Force past rate-limit between checks.
    for i in range(3):
        state.inventory_consistency_last_check_mono = None
        b = check_inventory_consistency(state, settings)
        if i < 2:
            assert b is None, f"breach fired prematurely at sample {i + 1}"
        else:
            assert b is not None, "breach did not fire after 3 consecutive samples"
    assert state.inventory_consistency_breach_count == 1
    # After fire, counter resets so the breach handler's pause/cancel is
    # the cooldown, not a threshold-reached re-fire.
    assert state.inventory_consistency_consecutive_drift_count == 0


def test_consecutive_threshold_one_restores_legacy_behaviour() -> None:
    """Setting the threshold to 1 restores the pre-hardening single-shot
    fire — useful for operators who'd rather over-fire than under-fire.
    """
    state = _make_state(baseline_qty=-28.0, venue_qty=-8.0)
    settings = _settings(
        INVENTORY_CONSISTENCY_CONSECUTIVE_BREACHES_REQUIRED=1,
    )
    b = check_inventory_consistency(state, settings)
    assert b is not None
    assert state.inventory_consistency_breach_count == 1


def test_resume_rebaseline_resets_consecutive_drift_counter() -> None:
    """The /resume rebaseline must also clear the consecutive counter —
    otherwise an unconfirmed-drift streak from before the operator's
    intervention would carry forward and skew the next check.
    """
    state = _make_state(baseline_qty=-28.0, venue_qty=-8.0)
    settings = _settings()
    # Build up a 2-sample drift streak.
    check_inventory_consistency(state, settings)
    state.inventory_consistency_last_check_mono = None
    check_inventory_consistency(state, settings)
    assert state.inventory_consistency_consecutive_drift_count == 2

    # Operator pauses, re-syncs, resumes.
    state.set_manual_pause(True)
    state.position.position_qty = -28.0  # operator confirmed the real position
    state.set_manual_pause(False)
    assert state.inventory_consistency_consecutive_drift_count == 0


def test_session_signed_qty_total_tracks_buy_and_sell_fills() -> None:
    """Underlying invariant: BUY adds, SELL subtracts."""
    state = _make_state(baseline_qty=0.0, venue_qty=0.0)
    _bump(state, Side.BUY, 7.0)
    _bump(state, Side.BUY, 3.0)
    _bump(state, Side.SELL, 4.0)
    assert abs(state.session_signed_qty_total - 6.0) < 1e-9


def test_baseline_set_is_idempotent() -> None:
    """Re-calling baseline setter must not reset session_signed_qty_total."""
    state = BotState(_settings())
    state.set_inventory_baseline_at_session_start(50.0)
    state.session_signed_qty_total = 12.0
    state.set_inventory_baseline_at_session_start(99.0)  # no-op
    assert state.inventory_baseline_qty == 50.0
    assert state.session_signed_qty_total == 12.0


# --- /resume rebaseline integration --------------------------------------


def test_set_manual_pause_resume_rebaselines_to_venue() -> None:
    """The natural operator flow — /pause → close manually on the venue UI →
    /resume — must NOT trip the inventory consistency watchdog. The
    True→False transition rebaselines ``inventory_baseline_qty`` to the
    current venue position and zeroes ``session_signed_qty_total`` so the
    next check sees expected == venue.
    """
    state = _make_state(baseline_qty=78.0, venue_qty=78.0)
    _bump(state, Side.BUY, 5.0)  # session_signed = +5
    state.session_signed_qty_total  # noqa: B018  — doc reader hint
    assert state.session_signed_qty_total == 5.0

    # Operator pauses, closes 78 SUI on the venue UI (venue_qty drops to 0).
    state.set_manual_pause(True)
    state.position.position_qty = 0.0  # simulate the venue close

    # Operator resumes. The True→False edge must rebaseline.
    rebaselined = state.set_manual_pause(False)
    assert rebaselined is True
    assert state.inventory_baseline_qty == 0.0
    assert state.session_signed_qty_total == 0.0
    # The rate-limit anchor is cleared so the next check evaluates promptly.
    assert state.inventory_consistency_last_check_mono is None

    # Watchdog now sees expected == venue == 0, no breach.
    settings = _settings()
    assert check_inventory_consistency(state, settings) is None


def test_set_manual_pause_unchanged_does_not_rebaseline() -> None:
    """Calling set_manual_pause(False) when already False is a no-op for
    the baseline (no transition fired). Likewise pause→pause."""
    state = _make_state(baseline_qty=10.0, venue_qty=15.0)
    _bump(state, Side.BUY, 5.0)
    assert state.session_signed_qty_total == 5.0

    # No transition (False stays False) → no rebaseline.
    rebaselined = state.set_manual_pause(False)
    assert rebaselined is False
    assert state.inventory_baseline_qty == 10.0
    assert state.session_signed_qty_total == 5.0

    # Pause → pause (True → True) is also not a transition.
    state.set_manual_pause(True)
    rebaselined2 = state.set_manual_pause(True)
    assert rebaselined2 is False
    assert state.inventory_baseline_qty == 10.0
    assert state.session_signed_qty_total == 5.0


def test_set_manual_pause_resume_clears_stale_breach_payload() -> None:
    """If a breach was recorded BEFORE the operator's manual close, the
    /resume rebaseline must clear the stale breach payload — otherwise the
    operator-facing status would keep showing the pre-resolution alarm.

    Uses ``CONSECUTIVE_BREACHES_REQUIRED=1`` so a single sample fires and
    populates the last-breach payload (the default 3-sample threshold has
    its own coverage in the consecutive-hardening tests).
    """
    state = _make_state(baseline_qty=0.0, venue_qty=200.0)
    settings = _settings(
        INVENTORY_CONSISTENCY_CONSECUTIVE_BREACHES_REQUIRED=1,
    )
    breach = check_inventory_consistency(state, settings)
    assert breach is not None
    assert state.inventory_consistency_last_breach is not None

    state.set_manual_pause(True)
    state.position.position_qty = 0.0  # operator flattens
    state.set_manual_pause(False)
    assert state.inventory_consistency_last_breach is None
