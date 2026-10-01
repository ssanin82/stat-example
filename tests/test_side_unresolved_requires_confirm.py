"""
Invariant tests for the side-unresolved `requires_confirm` latch
(`OrderManager._side_unresolved_requires_confirm` +
`_set_side_unresolved(..., requires_confirm=...)` +
`_is_side_unresolved` failsafe branch +
`_clear_side_unresolved`).

Context. `_is_side_unresolved` contains a failsafe: if the local slot is
empty AND `side_unresolved_suppression_timeout_seconds` has elapsed, the
unresolved state is auto-cleared so quoting can resume. That failsafe is
unsafe when we entered unresolved *because an orphan-cancel dispatch FAILED*
— the phantom is still live on the exchange, and auto-releasing would let
the next quote cycle place a fresh same-side order that the exchange
observes as a duplicate.

The `requires_confirm` latch prevents that race:

  - Set to True via `_set_side_unresolved(..., requires_confirm=True)`.
    Monotonic-up within an unresolved episode: re-entry with
    `requires_confirm=False` must NOT downgrade it.
  - Blocks the failsafe auto-release in `_is_side_unresolved` (the caller
    still sees True; state is NOT cleared).
  - Cleared only by `_clear_side_unresolved`, which is called from paths
    with positive exchange confirmation (`reconcile_*`, `ws_terminal_*`,
    `working_slot_released_*`).
  - Increments `_side_unresolved_confirm_block_count` once per blocked
    timeout window per side (the failsafe resets `since_mono` to throttle
    log spam to one-per-timeout-window).

These tests pin each of those four behaviors directly against the
`OrderManager` API.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from app.enums import Side
from app.execution import OrderManager
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[UnitTestSettings, Path, OrderManager]:
    path = Path(tempfile.gettempdir()) / f"mm_rc_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state)
    # Ensure both slots are empty so the failsafe timeout branch is eligible.
    om._state.working_bid = None
    om._state.working_ask = None
    return s, path, om


def _age_past_timeout(om: OrderManager, s: UnitTestSettings, side: Side) -> None:
    """Rewrite `_side_unresolved_since_mono` to well past the failsafe threshold."""
    timeout_s = float(s.side_unresolved_suppression_timeout_seconds)
    om._side_unresolved_since_mono[side] = time.monotonic() - timeout_s - 1.0


def test_requires_confirm_blocks_timeout_auto_release() -> None:
    """
    A. `requires_confirm=True` + empty slot + elapsed timeout →
       `_is_side_unresolved` returns True (state is NOT cleared), and the
       `_side_unresolved_confirm_block_count` counter increments.
    """
    s, path, om = _setup()

    om._set_side_unresolved(
        Side.BUY, reason="test_block_release", requires_confirm=True
    )
    assert om._side_unresolved_active[Side.BUY] is True
    assert om._side_unresolved_requires_confirm[Side.BUY] is True
    blocks_before = om._side_unresolved_confirm_block_count

    _age_past_timeout(om, s, Side.BUY)

    still_unresolved = om._is_side_unresolved(Side.BUY)

    assert still_unresolved is True, (
        "latch must block failsafe auto-release past timeout when orphan cancel unconfirmed"
    )
    assert om._side_unresolved_active[Side.BUY] is True, (
        "unresolved state must NOT be cleared by the failsafe when latch is set"
    )
    assert om._side_unresolved_requires_confirm[Side.BUY] is True, (
        "latch must remain True while blocking"
    )
    assert om._side_unresolved_confirm_block_count == blocks_before + 1, (
        f"block counter must increment on each blocked check "
        f"(before={blocks_before} after={om._side_unresolved_confirm_block_count})"
    )
    path.unlink(missing_ok=True)


def test_no_latch_allows_timeout_auto_release() -> None:
    """
    B. `requires_confirm=False` + empty slot + elapsed timeout →
       failsafe clears unresolved state and `_is_side_unresolved` returns False.
       This confirms the default liveness path is unchanged.
    """
    s, path, om = _setup()

    om._set_side_unresolved(
        Side.BUY, reason="test_normal_release", requires_confirm=False
    )
    assert om._side_unresolved_active[Side.BUY] is True
    assert om._side_unresolved_requires_confirm[Side.BUY] is False

    _age_past_timeout(om, s, Side.BUY)

    out = om._is_side_unresolved(Side.BUY)

    assert out is False, (
        "without latch, failsafe must clear unresolved state when slot empty + timeout"
    )
    assert om._side_unresolved_active[Side.BUY] is False, (
        "auto-release must flip the active flag off"
    )
    path.unlink(missing_ok=True)


def test_clear_side_unresolved_resets_latch_and_restores_failsafe() -> None:
    """
    C. `_clear_side_unresolved` (from a confirmed exchange-release path) must
       reset the latch. After clearing, a subsequent normal entry behaves like
       a fresh episode — the failsafe timeout releases it as usual.
    """
    s, path, om = _setup()

    om._set_side_unresolved(
        Side.BUY, reason="will_be_cleared", requires_confirm=True
    )
    assert om._side_unresolved_requires_confirm[Side.BUY] is True

    om._clear_side_unresolved(Side.BUY, reason="ws_terminal_canceled")
    assert om._side_unresolved_active[Side.BUY] is False, "clear deactivates state"
    assert om._side_unresolved_requires_confirm[Side.BUY] is False, (
        "clear must reset the `requires_confirm` latch"
    )
    assert om._side_unresolved_since_mono[Side.BUY] == 0.0, (
        "clear must reset since-mono"
    )

    # Re-enter with no latch and age past timeout — failsafe must now fire normally.
    om._set_side_unresolved(
        Side.BUY, reason="re_enter_no_confirm", requires_confirm=False
    )
    assert om._side_unresolved_requires_confirm[Side.BUY] is False, (
        "post-clear, a new episode without the latch must start clean"
    )
    _age_past_timeout(om, s, Side.BUY)
    assert om._is_side_unresolved(Side.BUY) is False, (
        "post-clear, failsafe timeout must release normally"
    )
    path.unlink(missing_ok=True)


def test_latch_is_monotonic_up_across_reentry() -> None:
    """
    D. Once the latch is True within an unresolved episode, a subsequent
       `_set_side_unresolved(..., requires_confirm=False)` MUST NOT downgrade
       it. The phantom risk from the earlier entry is still dominant.
       The failsafe must continue to block even after the re-entry call.
    """
    s, path, om = _setup()

    om._set_side_unresolved(
        Side.BUY, reason="first_entry", requires_confirm=True
    )
    assert om._side_unresolved_requires_confirm[Side.BUY] is True

    # Re-entry with requires_confirm=False. The latch must remain True.
    om._set_side_unresolved(
        Side.BUY, reason="second_entry_no_confirm", requires_confirm=False
    )
    assert om._side_unresolved_requires_confirm[Side.BUY] is True, (
        "latch is monotonic-up within an unresolved episode; "
        "False must NOT downgrade True"
    )

    # Aging past timeout must still block auto-release.
    _age_past_timeout(om, s, Side.BUY)
    assert om._is_side_unresolved(Side.BUY) is True, (
        "monotonic-up latch must continue to block failsafe auto-release"
    )
    assert om._side_unresolved_confirm_block_count >= 1
    path.unlink(missing_ok=True)


def test_latch_independent_per_side() -> None:
    """
    E. (defense-in-depth) The latch is per-side. Setting BUY's latch does not
       affect SELL's failsafe behavior.
    """
    s, path, om = _setup()

    om._set_side_unresolved(
        Side.BUY, reason="buy_needs_confirm", requires_confirm=True
    )
    om._set_side_unresolved(
        Side.SELL, reason="sell_no_confirm", requires_confirm=False
    )
    assert om._side_unresolved_requires_confirm[Side.BUY] is True
    assert om._side_unresolved_requires_confirm[Side.SELL] is False

    _age_past_timeout(om, s, Side.BUY)
    _age_past_timeout(om, s, Side.SELL)

    # BUY is blocked; SELL is released.
    assert om._is_side_unresolved(Side.BUY) is True, "BUY latch blocks release"
    assert om._is_side_unresolved(Side.SELL) is False, (
        "SELL without latch must be released by failsafe"
    )
    assert om._side_unresolved_active[Side.BUY] is True
    assert om._side_unresolved_active[Side.SELL] is False
    path.unlink(missing_ok=True)


def test_blocked_check_throttles_since_mono_to_avoid_log_spam() -> None:
    """
    F. When the failsafe is blocked by the latch, the code resets
       `_side_unresolved_since_mono` to the current monotonic time so the
       WARNING log fires at most once per timeout window. We verify that
       reset behavior directly by checking the `since_mono` field before
       vs. after the blocked check.
    """
    s, path, om = _setup()

    om._set_side_unresolved(
        Side.BUY, reason="throttle_test", requires_confirm=True
    )
    _age_past_timeout(om, s, Side.BUY)
    aged_since = om._side_unresolved_since_mono[Side.BUY]

    out = om._is_side_unresolved(Side.BUY)
    assert out is True

    refreshed_since = om._side_unresolved_since_mono[Side.BUY]
    assert refreshed_since > aged_since, (
        "blocked-release path must reset since-mono to throttle log spam "
        f"(aged={aged_since} refreshed={refreshed_since})"
    )
    # The refreshed value must be within epsilon of `now`.
    assert (time.monotonic() - refreshed_since) < 1.0
    path.unlink(missing_ok=True)
