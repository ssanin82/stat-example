"""Phase 1b regression — opportunistic batch-cancel in the dispatcher.

When 1+ cancel(s) for distinct sides queue together, the dispatcher
should call ``execute_cancel_batch`` with the intents — routing
through the OKX ``cancel-batch-orders`` endpoint (300/2 s rate-limit
pool) instead of ``cancel-order`` (60/2 s pool).

v1.4.33 (2026-05-17) threshold change: lowered from "2+ cancels" to
"1+ cancel". The pre-v1.4.33 logic preferred the single-cancel path
for solo cancels on the assumption that 1-element batches save no
RTT and add overhead. Snapshot ``v1.4.32-260517-231426`` revealed
that solo cancels are the dominant case (every Binance-cross-venue
trigger fires one cancel at a time, never two together), so the
single-cancel endpoint was being hammered at 412 % of its 60/2 s
cap. The CANCEL_BATCH endpoint has a 300/2 s pool — 5× the budget —
and OKX accepts a 1-row body, so routing all cancels through batch
shifts the pressure to the larger pool without changing RTT or
behavior. The new contract: as long as ``batch_cancels_enabled`` is
true and the callback is wired, every cancel goes via batch.

The legacy per-intent path stays unchanged when
``batch_cancels_enabled=false`` or ``execute_cancel_batch=None``;
it's also still the fallback when same-rung or amend-inflight
invariants knock all cancels out of the batch (zero survivors after
filter → fall through to per-cancel loop).
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import uuid
from pathlib import Path

from app.enums import Side
from app.outbound_dispatch import (
    CancelTransportIntent,
    OutboundDispatchCoordinator,
    PlaceTransportIntent,
)
from tests.settings_helpers import UnitTestSettings


def _settings(*, batch_enabled: bool = True) -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_batch_cxl_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "ACTION_WS_ENABLED": False,
            "BATCH_CANCELS_ENABLED": batch_enabled,
        }
    )
    return s, path


def test_two_distinct_side_cancels_use_batch_path() -> None:
    """When BUY + SELL cancels queue together, the batch callback
    fires ONCE with both intents. The per-intent callback fires zero
    times."""
    s, path = _settings(batch_enabled=True)
    batch_calls: list[list[CancelTransportIntent]] = []
    single_calls: list[CancelTransportIntent] = []
    done = threading.Event()

    def exec_place(_p: PlaceTransportIntent) -> None:
        pass

    def exec_cancel(c: CancelTransportIntent) -> None:
        single_calls.append(c)

    def exec_cancel_batch(intents: list[CancelTransportIntent]) -> None:
        batch_calls.append(list(intents))
        done.set()

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
        execute_cancel_batch=exec_cancel_batch,
    )
    d.start()
    try:
        # Submit both cancels BEFORE the dispatcher wakes. The
        # ``submit_*`` calls notify the cond; we want them queued
        # before the worker takes its snapshot. Brief sleep at the
        # end gives the worker time to process.
        d.submit_cancel(CancelTransportIntent("a", Side.BUY, 1, time.monotonic()))
        d.submit_cancel(CancelTransportIntent("b", Side.SELL, 1, time.monotonic()))
        d.wait_until_idle(2.0)
        assert done.wait(timeout=1.0), "batch callback never fired"
        assert len(batch_calls) == 1, (
            f"Expected 1 batch call, got {len(batch_calls)}"
        )
        assert len(batch_calls[0]) == 2
        sides = {c.side for c in batch_calls[0]}
        assert sides == {Side.BUY, Side.SELL}
        assert single_calls == [], (
            f"Per-intent cancel should NOT fire when batch wins. "
            f"Got {len(single_calls)} single calls."
        )
        # Verify the dispatch counter incremented.
        stats = d.snapshot_stats()
        assert stats["batch_cancel_dispatch_count"] == 1
    finally:
        d.stop()
        path.unlink(missing_ok=True)


def test_single_cancel_uses_batch_path() -> None:
    """v1.4.33: a lone cancel now uses the batch path so its
    rate-limit pressure lands on CANCEL_BATCH (300/2 s) instead of
    CANCEL_SINGLE (60/2 s).

    Pre-v1.4.33 contract: single cancel → per-intent path (single
    endpoint). That was the right micro-optimisation when the
    dispatcher had no per-pool awareness, but it pegged the
    cancel-single pool at 412 % of cap in the v1.4.32 prod snapshot
    because solo cancels dominate (every Binance-cross-venue trigger
    is a single cancel). The new contract routes all cancels — solo
    or paired — through the batch callback.
    """
    s, path = _settings(batch_enabled=True)
    batch_calls: list[list[CancelTransportIntent]] = []
    single_calls: list[CancelTransportIntent] = []
    done = threading.Event()

    def exec_place(_p: PlaceTransportIntent) -> None:
        pass

    def exec_cancel(c: CancelTransportIntent) -> None:
        single_calls.append(c)

    def exec_cancel_batch(intents: list[CancelTransportIntent]) -> None:
        batch_calls.append(list(intents))
        done.set()

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
        execute_cancel_batch=exec_cancel_batch,
    )
    d.start()
    try:
        d.submit_cancel(CancelTransportIntent("a", Side.BUY, 1, time.monotonic()))
        assert done.wait(timeout=1.0), "batch callback never fired"
        d.wait_until_idle(2.0)
        assert len(batch_calls) == 1, (
            f"Expected 1 batch call, got {len(batch_calls)}"
        )
        assert len(batch_calls[0]) == 1
        assert batch_calls[0][0].wo_order_id_local == "a"
        assert single_calls == [], (
            f"Per-intent cancel should NOT fire when batch path is "
            f"available. Got {len(single_calls)} single calls."
        )
        stats = d.snapshot_stats()
        assert stats["batch_cancel_dispatch_count"] == 1
    finally:
        d.stop()
        path.unlink(missing_ok=True)


def test_batch_cancels_disabled_falls_back_to_per_intent() -> None:
    """``BATCH_CANCELS_ENABLED=false`` disables the batch path even
    when the callback is wired. Useful as a rollback hatch."""
    s, path = _settings(batch_enabled=False)
    batch_calls: list[list[CancelTransportIntent]] = []
    single_calls: list[CancelTransportIntent] = []

    def exec_place(_p: PlaceTransportIntent) -> None:
        pass

    def exec_cancel(c: CancelTransportIntent) -> None:
        single_calls.append(c)

    def exec_cancel_batch(intents: list[CancelTransportIntent]) -> None:
        batch_calls.append(list(intents))

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
        execute_cancel_batch=exec_cancel_batch,
    )
    d.start()
    try:
        d.submit_cancel(CancelTransportIntent("a", Side.BUY, 1, time.monotonic()))
        d.submit_cancel(CancelTransportIntent("b", Side.SELL, 1, time.monotonic()))
        d.wait_until_idle(2.0)
        # When the flag is off, all cancels go via the per-intent
        # path even if there are 2+ of them.
        assert len(single_calls) == 2
        assert batch_calls == []
    finally:
        d.stop()
        path.unlink(missing_ok=True)


def test_no_batch_callback_falls_back_to_per_intent() -> None:
    """When the dispatcher is constructed WITHOUT
    ``execute_cancel_batch=...`` (the legacy callsite shape — covered
    by other tests in this repo), the batch path is never used. The
    per-intent callback handles everything."""
    s, path = _settings(batch_enabled=True)
    single_calls: list[CancelTransportIntent] = []

    def exec_place(_p: PlaceTransportIntent) -> None:
        pass

    def exec_cancel(c: CancelTransportIntent) -> None:
        single_calls.append(c)

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
        # Note: NO execute_cancel_batch kwarg.
    )
    d.start()
    try:
        d.submit_cancel(CancelTransportIntent("a", Side.BUY, 1, time.monotonic()))
        d.submit_cancel(CancelTransportIntent("b", Side.SELL, 1, time.monotonic()))
        d.wait_until_idle(2.0)
        assert len(single_calls) == 2
        # batch_cancel_dispatch_count should stay 0.
        stats = d.snapshot_stats()
        assert stats["batch_cancel_dispatch_count"] == 0
    finally:
        d.stop()
        path.unlink(missing_ok=True)
