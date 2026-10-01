"""Regression test for the 2026-05-16 Codex review #1 fix.

Pre-fix bug: ``state.account_rest_last_monotonic`` was advanced on
BOTH success and failure paths (the failure-path stamp existed to
engage the retry-throttle on errors, and the same field was read by
the stale-account risk gate). On sustained REST failures the field
kept advancing every retry interval, so the gate measured "seconds
since last attempt" instead of "seconds since last success" — and
never tripped while account/position truth could be hours stale.

Fix: split the clock into two fields. ``account_rest_last_monotonic``
keeps the attempt semantics (throttle still engages on failure).
``account_rest_last_success_monotonic`` is the success-only anchor
the risk gate reads. This test exercises the full
``refresh_account_only`` failure path to ensure repeated failures
don't bump the success anchor — the same wiring the existing
``test_account_data_stale_gate`` suite explicitly didn't cover
(noted as the why-tests-missed-it gap in the Codex review).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from app.market_data import refresh_account_only
from app.state import BotState
from tests.settings_helpers import UnitTestSettings


def _state() -> BotState:
    """Bare BotState with minimum required wiring for refresh_account_only."""
    settings = UnitTestSettings.model_validate({
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
    })
    return BotState(settings)


def test_repeated_refresh_failures_do_not_advance_success_anchor() -> None:
    """The fix's core invariant: failing refreshes update the
    attempt clock (for throttle) but NOT the success clock (for the
    stale gate)."""
    state = _state()
    client = MagicMock()
    client.fetch_position.side_effect = Exception("rest 429")
    client.fetch_account_snapshot.side_effect = Exception("rest 429")

    # Before any refresh: both clocks unset.
    assert state.account_rest_last_monotonic is None
    assert state.account_rest_last_success_monotonic is None

    # First failure: attempt clock advances, success clock stays None.
    refresh_account_only(client, state, address="0xabc")
    after_first_attempt = state.account_rest_last_monotonic
    assert after_first_attempt is not None
    assert state.account_rest_last_success_monotonic is None

    # Pause and retry. Attempt clock should advance again; success
    # clock unchanged.
    time.sleep(0.02)
    refresh_account_only(client, state, address="0xabc")
    assert state.account_rest_last_monotonic > after_first_attempt
    assert state.account_rest_last_success_monotonic is None


def test_success_then_failures_pins_success_anchor() -> None:
    """The exact bug scenario: bot had a healthy refresh, then REST
    started failing. The success anchor must stay pinned to the
    successful refresh time so the stale-gate age grows as wall-clock
    advances."""
    state = _state()
    client = MagicMock()
    # First call succeeds.
    client.fetch_position.return_value = MagicMock(
        position_qty=0.0,
        symbol="ETH",
        avg_entry_price=0.0,
        mark_price=0.0,
        position_notional=0.0,
        unrealized_pnl_usd=0.0,
        timestamp=0,
    )
    client.fetch_account_snapshot.return_value = MagicMock(
        equity=1000.0, cash=1000.0, fees_usd=0.0, withdrawable_usd=1000.0,
        extra={}, account_value_usd=1000.0, total_margin_used_usd=0.0,
        available_margin_usd=1000.0,
    )
    refresh_account_only(client, state, address="0xabc", ingest_fills_via_rest=False)
    success_at = state.account_rest_last_success_monotonic
    assert success_at is not None

    # Now REST starts failing. Switch the mocks to raise.
    client.fetch_position.side_effect = Exception("rest 429")
    client.fetch_account_snapshot.side_effect = Exception("rest 429")
    # Reset the return values so they don't override side_effect (mock quirk).
    client.fetch_position.return_value = None
    client.fetch_account_snapshot.return_value = None

    # N retries. Attempt clock advances; success clock stays put.
    for _ in range(5):
        time.sleep(0.01)
        refresh_account_only(client, state, address="0xabc", ingest_fills_via_rest=False)

    assert state.account_rest_last_success_monotonic == success_at, (
        "Success anchor must NOT advance on failures — that's the whole "
        "point of the Codex #1 fix."
    )
    assert state.account_rest_last_monotonic is not None
    assert state.account_rest_last_monotonic > success_at, (
        "Attempt anchor MUST advance on failures so the throttle "
        "still engages and the bot doesn't tight-loop into 429s."
    )


def test_success_advances_both_anchors_together() -> None:
    """On a successful refresh, BOTH clocks advance — the success
    one for the stale gate, the attempt one so the throttle treats
    success as the most recent activity (no immediate re-fire)."""
    state = _state()
    client = MagicMock()
    client.fetch_position.return_value = MagicMock(
        position_qty=0.0, symbol="ETH", avg_entry_price=0.0,
        mark_price=0.0, position_notional=0.0, unrealized_pnl_usd=0.0,
        timestamp=0,
    )
    client.fetch_account_snapshot.return_value = MagicMock(
        equity=1000.0, cash=1000.0, fees_usd=0.0, withdrawable_usd=1000.0,
        extra={}, account_value_usd=1000.0, total_margin_used_usd=0.0,
        available_margin_usd=1000.0,
    )
    refresh_account_only(client, state, address="0xabc", ingest_fills_via_rest=False)
    assert state.account_rest_last_monotonic is not None
    assert state.account_rest_last_success_monotonic is not None
    # Both should be very close (set in the same monotonic.now() call).
    assert abs(
        state.account_rest_last_monotonic
        - state.account_rest_last_success_monotonic
    ) < 1e-6


def test_stale_age_anchored_on_success_after_repeated_failures() -> None:
    """End-to-end of the bug scenario: bot had one success, then
    repeated REST failures, then we compute the stale-age the way
    Bot.one_tick does. The age must reflect time-since-last-success,
    not time-since-last-attempt. Pre-fix this would have failed
    because the failure path advanced the same field the risk
    pipeline read."""
    state = _state()
    # Stamp a successful refresh 100 seconds ago into both clocks
    # (simulating "bot had a clean refresh 100s ago").
    fake_now = time.monotonic()
    success_at = fake_now - 100.0
    state.account_rest_last_monotonic = success_at
    state.account_rest_last_success_monotonic = success_at

    # Simulate N failed retries. Each call updates the attempt clock
    # but leaves the success clock alone (the fix's invariant).
    client = MagicMock()
    client.fetch_position.side_effect = Exception("rest 429")
    client.fetch_account_snapshot.side_effect = Exception("rest 429")
    for _ in range(5):
        refresh_account_only(client, state, address="0xabc", ingest_fills_via_rest=False)

    # This is the exact computation Bot.one_tick does
    # ([bot.py:4318] in the fix; previously read the attempt clock).
    age_s = (
        time.monotonic() - state.account_rest_last_success_monotonic
    )
    assert age_s >= 99.0, (
        f"After 100s of simulated staleness plus 5 failed retries, "
        f"the stale-gate age must be ~100s. Got {age_s:.3f}s. "
        f"If this is sub-second, the Codex #1 bug is back."
    )
    # And the attempt anchor MUST have moved past the 100s baseline,
    # confirming the throttle still functions.
    assert state.account_rest_last_monotonic > success_at + 1.0, (
        "Attempt anchor must advance on failures for the throttle "
        "to engage and prevent tight-loop retries."
    )
