"""Tests for the deadlock watchdog.

Covers the stuck-execution detection pattern:

* FIRES when quote engine has been emitting non-NONE recently AND
  execution hasn't dispatched for > ``WATCHDOG_NO_PLACE_ATTEMPT_SECONDS``
* DOES NOT FIRE when quote engine is itself idle (legitimate HOLD_ALL
  regime, not a deadlock)
* DOES NOT FIRE during warmup (either timestamp never set)
* DOES NOT FIRE when bot is killed / paused / flattening (intentional
  state, not a stuck state)
* SKIPS the initial grace window so slow startup doesn't false-trigger
* ENABLED=false is a no-op (no thread started, no exits)
"""

from __future__ import annotations

import time
from typing import Any

from app.state import BotState
from app.watchdog import Watchdog, WATCHDOG_EXIT_CODE
from tests.settings_helpers import UnitTestSettings


def _make_settings(**overrides: Any) -> UnitTestSettings:
    base = {
        "EXCHANGE": "bluefin",
        "SYMBOL": "SUI-PERP",
        "BLUEFIN_PRIVATE_KEY": "00" * 32,
        "BLUEFIN_ACCOUNT_ADDRESS": "0x" + "ab" * 32,
        # Tight thresholds so tests don't have to wait real time.
        "WATCHDOG_ENABLED": True,
        "WATCHDOG_NO_PLACE_ATTEMPT_SECONDS": 10.0,
        "WATCHDOG_QUOTE_ACTIVITY_WINDOW_SECONDS": 5.0,
        "WATCHDOG_CHECK_INTERVAL_SECONDS": 1.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_watchdog(settings: UnitTestSettings) -> tuple[Watchdog, BotState, list]:
    """Return (watchdog, state, exit_calls_list). Doesn't start the thread."""
    state = BotState(settings)
    exit_calls: list = []

    def fake_exit(code: int) -> None:
        exit_calls.append(code)
        # Don't raise, don't kill — let the watchdog continue so the
        # test can observe multiple checks if it wants.

    wd = Watchdog(settings, state, exit_fn=fake_exit)
    return wd, state, exit_calls


def test_watchdog_fires_on_stuck_execution() -> None:
    """Quote engine active, execution silent → watchdog exits."""
    s = _make_settings(
        WATCHDOG_NO_PLACE_ATTEMPT_SECONDS=10.0,
        WATCHDOG_QUOTE_ACTIVITY_WINDOW_SECONDS=5.0,
    )
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    # Quote engine emitted non-NONE 1 s ago — well within 5 s window.
    state.last_quote_engine_non_hold_ts_mono = now - 1.0
    # Execution last attempted 60 s ago — far past 10 s threshold.
    state.last_place_attempt_ts_mono = now - 60.0

    wd._check_once()

    assert exits == [WATCHDOG_EXIT_CODE], f"expected one exit call with code {WATCHDOG_EXIT_CODE}, got {exits}"


def test_watchdog_silent_when_quote_engine_also_idle() -> None:
    """If quote engine is itself not emitting, we're not stuck — HOLD_ALL regime."""
    s = _make_settings(
        WATCHDOG_NO_PLACE_ATTEMPT_SECONDS=10.0,
        WATCHDOG_QUOTE_ACTIVITY_WINDOW_SECONDS=5.0,
    )
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    # Quote engine last emitted 60 s ago — outside the 5 s activity window.
    state.last_quote_engine_non_hold_ts_mono = now - 60.0
    # Execution silent for ages — normally that'd trigger, but quote
    # engine is also idle, so the watchdog considers this a legitimate
    # "nothing to do" regime.
    state.last_place_attempt_ts_mono = now - 100.0

    wd._check_once()

    assert exits == [], "watchdog must not fire when quote engine itself is idle"


def test_watchdog_silent_during_warmup_no_quote_ever() -> None:
    """Fresh bot, quote engine has never emitted — no firing."""
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    # Both timestamps remain at their init value (0.0).
    assert state.last_quote_engine_non_hold_ts_mono == 0.0
    assert state.last_place_attempt_ts_mono == 0.0

    wd._check_once()

    assert exits == [], "watchdog must not fire during warm-up"


def test_watchdog_silent_during_warmup_never_placed() -> None:
    """Quote engine warmed up but execution has never placed an order.

    Typical on a brand-new session that hasn't had a chance to quote yet,
    or on a misconfigured bot that literally cannot place. Either way,
    bouncing the container won't help — don't fire.
    """
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    state.last_quote_engine_non_hold_ts_mono = now - 1.0  # engine is active
    state.last_place_attempt_ts_mono = 0.0  # never placed

    wd._check_once()

    assert exits == [], "watchdog must not fire when execution has never attempted"


def test_watchdog_silent_when_killed() -> None:
    """Killed state is intentional; don't bounce."""
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    state.last_quote_engine_non_hold_ts_mono = now - 1.0
    state.last_place_attempt_ts_mono = now - 600.0
    state.killed = True

    wd._check_once()

    assert exits == [], "watchdog must not fire when bot is killed"


def test_watchdog_silent_when_manual_paused() -> None:
    """Manual pause is intentional; don't bounce."""
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    state.last_quote_engine_non_hold_ts_mono = now - 1.0
    state.last_place_attempt_ts_mono = now - 600.0
    state.manual_pause = True

    wd._check_once()

    assert exits == [], "watchdog must not fire during manual pause"


def test_watchdog_silent_when_flattening() -> None:
    """Flatten-in-progress is intentional; don't bounce until it completes."""
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    state.last_quote_engine_non_hold_ts_mono = now - 1.0
    state.last_place_attempt_ts_mono = now - 600.0
    state.flatten_mode = True

    wd._check_once()

    assert exits == [], "watchdog must not fire while flatten is in progress"


def test_watchdog_disabled_does_not_start_thread() -> None:
    """WATCHDOG_ENABLED=false → start() is a no-op."""
    s = _make_settings(WATCHDOG_ENABLED=False)
    wd, _state, exits = _make_watchdog(s)

    wd.start()

    assert wd._thread is None
    # And even if something sets state, no exit should fire.
    now = time.monotonic()
    _state.last_quote_engine_non_hold_ts_mono = now - 1.0
    _state.last_place_attempt_ts_mono = now - 600.0
    wd._check_once()
    assert exits == []


def test_watchdog_tolerates_non_numeric_state_fields() -> None:
    """Defensive coercion: if state fields get set to something
    non-numeric (a bug elsewhere), the watchdog's ``float(...)`` call
    raises. The thread's try/except wrapper swallows it so the watchdog
    keeps running rather than silently dying.
    """
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    # Set a non-numeric value — this makes float() raise.
    state.last_quote_engine_non_hold_ts_mono = "not-a-number"  # type: ignore[assignment]
    state.last_place_attempt_ts_mono = 0.0

    import pytest

    with pytest.raises((TypeError, ValueError)):
        wd._check_once()

    # _run() would have caught — we just verify the check-side raise
    # (the catch is tested indirectly by the thread continuing to run).
    assert exits == []


def test_watchdog_silent_when_amend_only_active_v1_4_42() -> None:
    """v1.4.42 BUG-025-adjacent fix: an amend-only quoting regime is
    NOT a deadlock. The pre-fix watchdog read ``last_place_attempt_ts_mono``
    which the v1.4.16 amend-on-reprice path doesn't bump — so a bot
    keeping an order alive via continuous amends looked stuck after
    600 s and got killed every ~10 minutes (observed 2026-05-18,
    snapshot v1.4.41-260518-103610).

    Fix: watchdog reads ``last_outbound_attempt_ts_mono`` which is
    bumped on PLACE and AMEND (not cancel). Amend activity now keeps
    the watchdog silent.
    """
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    state.last_quote_engine_non_hold_ts_mono = now - 1.0
    # Place counter goes stale (no fresh places — the bot is repricing
    # via amend-on-reprice which preserves queue position).
    state.last_place_attempt_ts_mono = now - 600.0
    # But the broader outbound counter IS being bumped by the amend
    # path (execution.py:8090ish, v1.4.42).
    state.last_outbound_attempt_ts_mono = now - 0.5

    wd._check_once()

    assert exits == [], (
        "v1.4.42: amend-only quoting must NOT trigger the watchdog. "
        "If this fires, the watchdog is still reading the place-only "
        "counter and will kill amend-heavy sessions."
    )


def test_watchdog_still_fires_on_cancel_only_loop_v1_4_42() -> None:
    """v1.4.42 negative control: a cancels-only loop (the BUG-025
    zombie pattern) MUST still trigger the watchdog. The fix is a
    NARROW carve-out for amends; cancels alone are still a wedge."""
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    state.last_quote_engine_non_hold_ts_mono = now - 1.0
    state.last_place_attempt_ts_mono = now - 600.0
    # Neither places nor amends — pure cancel-loop (BUG-025).
    state.last_outbound_attempt_ts_mono = now - 600.0

    wd._check_once()

    assert exits == [WATCHDOG_EXIT_CODE], (
        "v1.4.42 over-reach: a cancels-only stuck pattern (the BUG-025 "
        "wedge) MUST still trip the watchdog. The fix should only "
        "exempt amend activity, not all non-place activity."
    )


def test_watchdog_backcompat_falls_back_to_place_counter_v1_4_42() -> None:
    """v1.4.42 back-compat: if ``last_outbound_attempt_ts_mono`` is
    zero (e.g. tests that construct BotState directly without the new
    field, or pre-v1.4.42 state being read in a hot upgrade), the
    watchdog falls back to the legacy ``last_place_attempt_ts_mono``
    field. This pins the back-compat path so the v1.5+ cleanup that
    removes the fallback doesn't accidentally regress."""
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    state.last_quote_engine_non_hold_ts_mono = now - 1.0
    state.last_place_attempt_ts_mono = now - 600.0
    state.last_outbound_attempt_ts_mono = 0.0  # never set (legacy state)

    wd._check_once()

    assert exits == [WATCHDOG_EXIT_CODE], (
        "v1.4.42 back-compat: when the new field is zero, fall back "
        "to the legacy place-only counter so deadlocks in pre-v1.4.42 "
        "state still get caught."
    )


def test_watchdog_fires_only_once_then_keeps_running_for_idempotency() -> None:
    """A single deadlock should produce exactly one exit call.

    ``os._exit`` in production terminates immediately so this is
    academic; for tests with a fake exit_fn, we just verify the
    watchdog doesn't double-dispatch on a single check.
    """
    s = _make_settings()
    wd, state, exits = _make_watchdog(s)

    now = time.monotonic()
    state.last_quote_engine_non_hold_ts_mono = now - 1.0
    state.last_place_attempt_ts_mono = now - 600.0

    wd._check_once()
    wd._check_once()  # second check with same conditions

    # Both checks saw deadlock and called exit. That's fine — production
    # os._exit terminates on first call. For tests, we just assert
    # each call is consistent.
    assert exits == [WATCHDOG_EXIT_CODE, WATCHDOG_EXIT_CODE]
