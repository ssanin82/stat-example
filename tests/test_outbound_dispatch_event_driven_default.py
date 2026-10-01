"""Phase 0a regression — ``ACTION_BATCH_INTERVAL_MS`` default is 0.

The pre-1.4.0 default was 6.0, adding 0-6 ms of idle wait per flush
cycle. On colo'd OKX where a single cancel RTT is ~5 ms, that 6 ms
micro-batch window was the same magnitude as the entire transport
leg — pure latency overhead since same-side coalescing already
happens at submit time. This test pins the new default so a future
config refactor doesn't silently revert.

Also: end-to-end assert that a cancel submitted while the worker
thread is parked in ``cond.wait()`` is dispatched within 50 ms (loose
enough to survive CI jitter, tight enough to catch a regression
where the worker only wakes on a periodic timeout).
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


def _settings_with_default_interval() -> tuple[UnitTestSettings, Path]:
    """Settings WITHOUT a manual override on ACTION_BATCH_INTERVAL_MS —
    the test inspects the default. UnitTestSettings inherits the
    pydantic default from ``app.config.Settings``."""
    path = Path(tempfile.gettempdir()) / f"mm_od_evt_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "ACTION_WS_ENABLED": False,
            # Intentionally NOT setting ACTION_BATCH_INTERVAL_MS — we
            # want to verify the new default behaviour.
        }
    )
    return s, path


def test_action_batch_interval_ms_default_is_zero() -> None:
    """Pin the post-1.4.0 default to 0.0. If this fails, the operator
    or a future refactor changed the default — verify the new value
    is intentional and update the test to match."""
    s, path = _settings_with_default_interval()
    try:
        assert s.action_batch_interval_ms == 0.0, (
            f"ACTION_BATCH_INTERVAL_MS default must be 0.0 for event-"
            f"driven dispatch (Phase 0a of cxl-optimize plan). "
            f"Got {s.action_batch_interval_ms}."
        )
    finally:
        path.unlink(missing_ok=True)


def test_cancel_dispatched_under_50ms_with_default_settings() -> None:
    """End-to-end: with the new default (interval=0), a cancel
    submitted while the worker is parked should reach
    ``execute_cancel`` within 50 ms. Loose threshold; the actual
    median on a quiet machine is sub-millisecond.

    Pre-1.4.0 with interval=6.0, the worst case was ~6 ms of idle
    wait; the new event-driven mode removes even that. The bound
    matters more as a regression detector — if someone reverts the
    default to a non-zero value, this test catches it on slow CI."""
    s, path = _settings_with_default_interval()
    executed_at: list[float] = []
    reached = threading.Event()

    def exec_place(_p: PlaceTransportIntent) -> None:
        pass

    def exec_cancel(_c: CancelTransportIntent) -> None:
        executed_at.append(time.perf_counter())
        reached.set()

    d = OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
    )
    d.start()
    try:
        # Let the worker park in cond.wait().
        time.sleep(0.05)
        t0 = time.perf_counter()
        d.submit_cancel(
            CancelTransportIntent("c1", Side.BUY, 1, time.monotonic())
        )
        assert reached.wait(timeout=1.0), "cancel never dispatched"
        notify_to_exec_ms = (executed_at[0] - t0) * 1000.0
        assert notify_to_exec_ms < 50.0, (
            f"cancel took {notify_to_exec_ms:.1f} ms to dispatch — "
            f"event-driven mode should be sub-millisecond; if this is "
            f"failing on a slow CI host, raise the bound but verify the "
            f"event-driven path is still firing on notify (not on a "
            f"periodic timeout)."
        )
    finally:
        d.stop()
        path.unlink(missing_ok=True)
