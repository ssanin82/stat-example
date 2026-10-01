"""v1.4.203 — startup REST-only cleanup tests.

Pre-fix bug (snapshot v1.4.193-260521-162648):
``ws_event_unmatched_to_local_wo_total = 1`` flagged the wedge-
acceptance gate as FAIL even though the bot ran cleanly. Root cause:
``private_stream.start()`` subscribed BEFORE the startup cancel-all
REST request, so the cancel-terminals for pre-existing orphan
orders from the previous process arrived in the private queue with
no local working orders to match against.

Fix (v1.4.203): move the cancel-all + REST poll-until-empty BEFORE
the WS subscriptions, then reset the executor's startup-grace
counters at the session boundary. Operator directive:
*"validation counters start ONLY AFTER that bot starts."*

These tests cover the two halves of the fix:

1. ``wait_for_clean_book`` — the REST-poll-until-empty helper.
2. ``run_startup_cleanup`` — the orchestrator that calls cancel-all
   then ``wait_for_clean_book`` then ``reset_startup_grace_counters``.

Plus a focused test for the new ``reset_startup_grace_counters``
method on OrderManager (zeros the unmatched counter without
disturbing the genuinely-fatal-when-non-zero counters like
``wedge_episode_count_session``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.startup_cleanup import (
    CleanBookResult,
    run_startup_cleanup,
    wait_for_clean_book,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeOpenOrder:
    """Minimal stand-in for the venue's open-order row. Mirrors the
    REAL ``OpenOrderRaw`` (= ``HLOpenOrderRaw``), which carries the
    instrument under the venue-neutral ``coin`` attribute (NOT
    ``symbol``) -- the same field the live reconcile filters on
    (``_sync_open_orders_impl``: ``o.coin == settings.symbol``).

    This is a deliberate regression guard: an earlier
    ``wait_for_clean_book`` filtered on ``symbol``, a field the real
    row does NOT have, so the poll-until-empty fence matched nothing
    and reported the book clean on its first poll regardless of
    resting orders (a silent no-op)."""
    coin: str


@dataclass
class _FakeOpenOrderLegacySymbol:
    """A row exposing ONLY the legacy ``symbol`` attribute (no
    ``coin``). Pins the back-compat fallback: ``wait_for_clean_book``
    reads ``coin`` first but falls back to ``symbol`` so any older
    adapter / row shape is still counted."""
    symbol: str


class _SequencedClient:
    """A client whose ``fetch_open_orders_raw`` returns successive
    lists from a queue. Lets each test script the venue's drain
    timing precisely.

    ``raise_on`` lets the test inject exceptions at specific calls.
    """

    def __init__(
        self,
        responses: list[list[_FakeOpenOrder]],
        raise_on: tuple[int, ...] = (),
    ) -> None:
        self._responses = list(responses)
        self._raise_on = set(raise_on)
        self.calls = 0

    def fetch_open_orders_raw(self, address: str) -> list[_FakeOpenOrder]:
        self.calls += 1
        if self.calls in self._raise_on:
            raise RuntimeError(f"injected fault on call #{self.calls}")
        if not self._responses:
            return []
        return self._responses.pop(0)


@dataclass
class _FakeSettings:
    symbol: str = "TEST-USDT-SWAP"


# ---------------------------------------------------------------------------
# wait_for_clean_book — happy paths
# ---------------------------------------------------------------------------


def test_wait_for_clean_book_empty_on_first_poll() -> None:
    """Book already clean → single poll, return immediately."""
    client = _SequencedClient([[]])
    result = wait_for_clean_book(
        client=client,
        settings=_FakeSettings(),
        address="0xabc",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert result.success is True
    assert result.final_count == 0
    assert result.polls_taken == 1


def test_wait_for_clean_book_drains_after_a_few_polls() -> None:
    """Book has 3 orders that disappear over 3 polls (cancels
    landing asynchronously after the REST cancel-all returned)."""
    s = _FakeSettings(symbol="TEST-USDT-SWAP")
    client = _SequencedClient([
        [_FakeOpenOrder("TEST-USDT-SWAP")] * 3,
        [_FakeOpenOrder("TEST-USDT-SWAP")] * 1,
        [],
    ])
    result = wait_for_clean_book(
        client=client,
        settings=s,
        address="0xabc",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert result.success is True
    assert result.final_count == 0
    assert result.polls_taken == 3


def test_wait_for_clean_book_case_insensitive_symbol_match() -> None:
    """Venue may return symbol in different case. Match is upper-
    cased on both sides so e.g. ``ton-usdt-swap`` vs ``TON-USDT-SWAP``
    are treated as the same symbol."""
    client = _SequencedClient([
        [_FakeOpenOrder("ton-usdt-swap")],  # lowercase from venue
        [],
    ])
    result = wait_for_clean_book(
        client=client,
        settings=_FakeSettings(symbol="TON-USDT-SWAP"),
        address="0xabc",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert result.success is True


def test_wait_for_clean_book_ignores_other_symbols() -> None:
    """A second bot's orders on a different symbol are NOT counted —
    we only drain our own symbol."""
    client = _SequencedClient([
        [
            _FakeOpenOrder("OTHER-USDT-SWAP"),
            _FakeOpenOrder("OTHER-USDT-SWAP"),
        ],
    ])
    result = wait_for_clean_book(
        client=client,
        settings=_FakeSettings(symbol="TEST-USDT-SWAP"),
        address="0xabc",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert result.success is True
    assert result.final_count == 0


def test_wait_for_clean_book_counts_coin_field() -> None:
    """REGRESSION (the .coin/.symbol bug): orders are counted via the
    venue-neutral ``coin`` attribute. The real OpenOrderRaw row carries
    ``coin``, not ``symbol``; the pre-fix filter read ``symbol`` and so
    matched nothing -- the fence returned success on the FIRST poll even
    with orders resting. Here two ``coin`` orders MUST be seen (book NOT
    clean on poll 1) and only reported clean once they drain on poll 2.
    Under the old .symbol-only filter this would have returned
    polls_taken == 1 with success on the first read."""
    client = _SequencedClient([
        [_FakeOpenOrder("TEST-USDT-SWAP"), _FakeOpenOrder("TEST-USDT-SWAP")],
        [],
    ])
    result = wait_for_clean_book(
        client=client,
        settings=_FakeSettings(),
        address="0xabc",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert result.success is True
    assert result.final_count == 0
    assert result.polls_taken == 2  # would be 1 under the .symbol bug


def test_wait_for_clean_book_symbol_fallback_for_legacy_rows() -> None:
    """Back-compat: a row exposing only the legacy ``symbol`` attribute
    (no ``coin``) is still counted via the fallback, so the fence keeps
    working for any adapter / row shape that predates the ``coin``
    field."""
    client = _SequencedClient([
        [_FakeOpenOrderLegacySymbol("TEST-USDT-SWAP")],
        [],
    ])
    result = wait_for_clean_book(
        client=client,
        settings=_FakeSettings(),
        address="0xabc",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert result.success is True
    assert result.final_count == 0
    assert result.polls_taken == 2


# ---------------------------------------------------------------------------
# wait_for_clean_book — timeout + error paths
# ---------------------------------------------------------------------------


def test_wait_for_clean_book_timeout_when_orders_persist() -> None:
    """Cancel-all command failed silently (or matching engine stuck);
    the book never drains. We time out and report the residual count."""
    client = _SequencedClient([
        [_FakeOpenOrder("TEST-USDT-SWAP")] * 2
        for _ in range(50)  # plenty of stubs; loop times out first
    ])
    result = wait_for_clean_book(
        client=client,
        settings=_FakeSettings(),
        address="0xabc",
        timeout_seconds=0.05,
        poll_interval_seconds=0.02,
    )
    assert result.success is False
    assert result.final_count == 2


def test_wait_for_clean_book_swallows_rest_exceptions() -> None:
    """Transient REST errors during polling are caught + retried.
    A single transient hiccup doesn't crash startup."""
    client = _SequencedClient(
        responses=[[]],
        raise_on=(1,),  # first call throws; second returns []
    )
    result = wait_for_clean_book(
        client=client,
        settings=_FakeSettings(),
        address="0xabc",
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert result.success is True
    assert result.polls_taken >= 2


def test_wait_for_clean_book_records_polls_and_elapsed() -> None:
    """``CleanBookResult`` exposes poll count + elapsed time for
    operator log lines. Both must be sensible (polls >= 1; elapsed
    >= 0)."""
    client = _SequencedClient([[], [], []])
    t0 = time.monotonic()
    result = wait_for_clean_book(
        client=client,
        settings=_FakeSettings(),
        address="0xabc",
        timeout_seconds=5.0,
        poll_interval_seconds=0.001,
    )
    elapsed_real = time.monotonic() - t0
    assert result.polls_taken == 1
    assert result.elapsed_seconds >= 0.0
    assert result.elapsed_seconds <= elapsed_real + 0.5  # generous


# ---------------------------------------------------------------------------
# CleanBookResult is a frozen dataclass — defensive immutability check
# ---------------------------------------------------------------------------


def test_clean_book_result_is_frozen() -> None:
    """Result is frozen — callers can't mutate the outcome after
    the helper returns."""
    r = CleanBookResult(
        success=True, final_count=0, polls_taken=1, elapsed_seconds=0.5
    )
    with pytest.raises(Exception):
        r.success = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OrderManager.reset_startup_grace_counters
# ---------------------------------------------------------------------------


def test_reset_startup_grace_counters_zeros_unmatched() -> None:
    """The new method on OrderManager zeros the startup-noise
    counters but leaves the fatal-when-non-zero counters alone."""
    from app.execution import OrderManager

    class _ShellOM:
        pass

    om = _ShellOM()
    om._ws_event_unmatched_to_local_wo_total = 5  # type: ignore[attr-defined]
    # Bind the method.
    OrderManager.reset_startup_grace_counters(om)  # type: ignore[arg-type]
    assert om._ws_event_unmatched_to_local_wo_total == 0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# run_startup_cleanup — orchestration
# ---------------------------------------------------------------------------


def _make_orchestration_fixtures(
    *,
    cancel_outcome: str = "bulk_ok",
    book_responses: list[list[_FakeOpenOrder]] | None = None,
):
    """Build a bot stub + client stub + storage stub for the
    orchestration tests. Each test can override individual pieces."""
    if book_responses is None:
        book_responses = [[]]

    bot = MagicMock()
    bot._exec = MagicMock()
    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback = MagicMock(
        return_value=cancel_outcome
    )
    bot._exec.reset_startup_grace_counters = MagicMock()

    client = _SequencedClient(book_responses)
    settings = _FakeSettings()
    storage = MagicMock()
    storage.insert_bot_event = MagicMock()
    return bot, client, settings, storage


def test_run_startup_cleanup_invokes_each_step_in_order() -> None:
    """End-to-end orchestration: cancel-all → poll-until-empty →
    reset-counters. All three called exactly once on a clean run."""
    bot, client, settings, storage = _make_orchestration_fixtures()
    result = run_startup_cleanup(
        bot=bot,
        client=client,
        settings=settings,
        address="0xabc",
        storage=storage,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback.assert_called_once()
    assert client.calls == 1
    bot._exec.reset_startup_grace_counters.assert_called_once()
    assert result.success is True


def test_run_startup_cleanup_emits_executed_event() -> None:
    """Persists ``cancel_all_on_startup_executed`` for the dashboard."""
    bot, client, settings, storage = _make_orchestration_fixtures()
    run_startup_cleanup(
        bot=bot,
        client=client,
        settings=settings,
        address="0xabc",
        storage=storage,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    event_types = [
        call.args[2] for call in storage.insert_bot_event.call_args_list
    ]
    assert "cancel_all_on_startup_executed" in event_types


def test_run_startup_cleanup_emits_clean_event_on_success() -> None:
    """On a clean drain, persists ``startup_book_clean`` with the
    poll count + elapsed time."""
    bot, client, settings, storage = _make_orchestration_fixtures()
    run_startup_cleanup(
        bot=bot,
        client=client,
        settings=settings,
        address="0xabc",
        storage=storage,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    event_types = [
        call.args[2] for call in storage.insert_bot_event.call_args_list
    ]
    assert "startup_book_clean" in event_types
    assert "startup_book_drain_timeout" not in event_types


def test_run_startup_cleanup_emits_timeout_event_on_failure() -> None:
    """On a drain timeout, persists ``startup_book_drain_timeout``."""
    # Book never drains; helper times out.
    book = [[_FakeOpenOrder("TEST-USDT-SWAP")]] * 50
    bot, client, settings, storage = _make_orchestration_fixtures(
        book_responses=book,
    )
    result = run_startup_cleanup(
        bot=bot,
        client=client,
        settings=settings,
        address="0xabc",
        storage=storage,
        timeout_seconds=0.05,
        poll_interval_seconds=0.02,
    )
    assert result.success is False
    event_types = [
        call.args[2] for call in storage.insert_bot_event.call_args_list
    ]
    assert "startup_book_drain_timeout" in event_types
    assert "startup_book_clean" not in event_types


def test_run_startup_cleanup_resets_counters_even_on_timeout() -> None:
    """Even when the drain times out, the executor's startup-grace
    counters are reset — better to start the session with clean
    counters and accept the small risk of orphan events than to
    leave the wedge-acceptance gate failing forever from a single
    bad startup."""
    book = [[_FakeOpenOrder("TEST-USDT-SWAP")]] * 50
    bot, client, settings, storage = _make_orchestration_fixtures(
        book_responses=book,
    )
    run_startup_cleanup(
        bot=bot,
        client=client,
        settings=settings,
        address="0xabc",
        storage=storage,
        timeout_seconds=0.05,
        poll_interval_seconds=0.02,
    )
    bot._exec.reset_startup_grace_counters.assert_called_once()


def test_run_startup_cleanup_survives_cancel_all_exception() -> None:
    """If the cancel-all REST call throws, the orchestrator still
    runs the drain + reset — the goal is to leave the bot in the
    best state we can manage, not abort on a hiccup."""
    bot, client, settings, storage = _make_orchestration_fixtures()
    bot._exec.cancel_all_orders_for_symbol_bulk_or_fallback.side_effect = (
        RuntimeError("REST blew up")
    )
    result = run_startup_cleanup(
        bot=bot,
        client=client,
        settings=settings,
        address="0xabc",
        storage=storage,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    # The drain still runs (book happens to be empty in this fixture).
    assert client.calls >= 1
    bot._exec.reset_startup_grace_counters.assert_called_once()
    # The "executed" event carries the exception outcome marker.
    event_types_with_outcome = [
        (call.args[2], call.args[4].get("outcome"))
        for call in storage.insert_bot_event.call_args_list
        if call.args[2] == "cancel_all_on_startup_executed"
    ]
    assert event_types_with_outcome
    assert event_types_with_outcome[0][1] == "cancel_exception"


def test_run_startup_cleanup_returns_clean_book_result() -> None:
    """Return type is ``CleanBookResult`` so callers can inspect
    the outcome (success / polls / elapsed)."""
    bot, client, settings, storage = _make_orchestration_fixtures()
    result = run_startup_cleanup(
        bot=bot,
        client=client,
        settings=settings,
        address="0xabc",
        storage=storage,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert isinstance(result, CleanBookResult)
    assert result.success is True
    assert result.polls_taken >= 1
    assert result.elapsed_seconds >= 0.0


# ---------------------------------------------------------------------------
# Regression: the v1.4.193 snapshot scenario
# ---------------------------------------------------------------------------


def test_regression_v1_4_193_orphan_terminals_no_longer_tick() -> None:
    """Replays the v1.4.193 (2026-05-21 11:22 UTC) startup pattern
    that produced ``ws_event_unmatched_to_local_wo_total = 1``:

    - Process restart with 1 pre-existing order on the venue.
    - Pre-fix flow: WS subscribed, cancel-all REST fired, the WS
      cancel-terminal landed with no local WO → counter ticks.
    - Post-fix flow: cancel-all + drain BEFORE subscribe, then
      counter reset → counter stays at zero across the boundary.

    Asserts: after ``run_startup_cleanup``, the executor's
    ``reset_startup_grace_counters`` was called (zeroing the
    counter). The actual WS-subscribe-after-clean part is verified
    by the ``main.py`` reorder; this test pins the contract
    between the orchestration helper and the executor."""
    bot, client, settings, storage = _make_orchestration_fixtures(
        book_responses=[
            [_FakeOpenOrder("TEST-USDT-SWAP")],  # one pre-existing order
            [],                                  # gone after cancel landed
        ],
    )
    result = run_startup_cleanup(
        bot=bot,
        client=client,
        settings=settings,
        address="0xabc",
        storage=storage,
        timeout_seconds=5.0,
        poll_interval_seconds=0.01,
    )
    assert result.success is True
    assert result.final_count == 0
    assert result.polls_taken == 2
    bot._exec.reset_startup_grace_counters.assert_called_once()
