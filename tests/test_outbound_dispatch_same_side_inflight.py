"""
Invariant: the outbound dispatcher MUST refuse to execute a second same-side
place/cancel within one flush window or while a same-side callback is in
flight. It defers the duplicate back to the lane and increments
`same_side_inflight_deferrals`.

This is a dispatch-layer defense-in-depth. The normal submit_* path coalesces
same-side intents before they reach the lane, so deferrals should be rare in
practice — but if two intents do land (e.g. an external re-entrant producer),
the dispatcher must not send both concurrently. Same-side-exclusivity is the
fundamental strategy invariant ("at most one live order per side per symbol").
"""

from __future__ import annotations

import os
import tempfile
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


def _settings() -> UnitTestSettings:
    path = Path(tempfile.gettempdir()) / f"mm_od_same_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    return UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "ACTION_BATCH_INTERVAL_MS": 0,
            "ACTION_MAX_BATCH_SIZE": 8,
            "ACTION_WS_ENABLED": False,
        }
    )


def _place(local_id: str, side: Side, seq: int) -> PlaceTransportIntent:
    return PlaceTransportIntent(
        wo_order_id_local=local_id,
        side=side,
        intent_seq=seq,
        quote_cycle_id=f"cyc-{seq}",
        enqueued_mono=time.monotonic(),
        intent_created_perf=0.0,
    )


def _cancel(local_id: str, side: Side, seq: int) -> CancelTransportIntent:
    return CancelTransportIntent(
        wo_order_id_local=local_id,
        side=side,
        intent_seq=seq,
        enqueued_mono=time.monotonic(),
    )


def _coordinator(*, exec_place, exec_cancel) -> OutboundDispatchCoordinator:
    s = _settings()
    return OutboundDispatchCoordinator(
        s,
        execute_place=exec_place,
        execute_cancel=exec_cancel,
    )


def test_two_same_side_places_in_one_batch_defers_second() -> None:
    """
    Seed the place lane directly with 2 same-side intents (bypassing the
    submit_place coalescer), then run one flush — only one must execute; the
    other must be deferred and the counter must increment.
    """
    executed: list[PlaceTransportIntent] = []

    def exec_place(p: PlaceTransportIntent) -> None:
        executed.append(p)

    def exec_cancel(_c: CancelTransportIntent) -> None:
        pass

    d = _coordinator(exec_place=exec_place, exec_cancel=exec_cancel)
    # Seed directly (no start(); invoke _flush_one_batch synchronously).
    d._place_lane.append(_place("a", Side.BUY, 1))
    d._place_lane.append(_place("b", Side.BUY, 2))

    d._flush_one_batch()

    assert len(executed) == 1, (
        f"same-side exclusivity: only one place per side per flush; executed={len(executed)}"
    )
    assert d._same_side_inflight_deferrals == 1, (
        f"deferral counter must increment; got {d._same_side_inflight_deferrals}"
    )
    # Deferred intent must be back in the lane for the next flush.
    assert len(d._place_lane) == 1
    # Next flush drains the leftover.
    d._flush_one_batch()
    assert len(executed) == 2


def test_two_same_side_cancels_in_one_batch_defers_second() -> None:
    """Same invariant for the cancel lane."""
    executed: list[CancelTransportIntent] = []

    def exec_place(_p: PlaceTransportIntent) -> None:
        pass

    def exec_cancel(c: CancelTransportIntent) -> None:
        executed.append(c)

    d = _coordinator(exec_place=exec_place, exec_cancel=exec_cancel)
    d._cancel_lane.append(_cancel("a", Side.SELL, 1))
    d._cancel_lane.append(_cancel("b", Side.SELL, 2))

    d._flush_one_batch()

    assert len(executed) == 1
    assert d._same_side_inflight_deferrals == 1
    assert len(d._cancel_lane) == 1


def test_opposite_sides_both_execute_in_one_batch() -> None:
    """Opposite-side intents are independent — both execute in the same flush."""
    executed: list[PlaceTransportIntent] = []

    def exec_place(p: PlaceTransportIntent) -> None:
        executed.append(p)

    def exec_cancel(_c: CancelTransportIntent) -> None:
        pass

    d = _coordinator(exec_place=exec_place, exec_cancel=exec_cancel)
    d._place_lane.append(_place("a", Side.BUY, 1))
    d._place_lane.append(_place("b", Side.SELL, 2))

    d._flush_one_batch()

    assert len(executed) == 2
    assert d._same_side_inflight_deferrals == 0


def test_stats_snapshot_exposes_inflight_deferrals() -> None:
    """snapshot_stats must surface the counter for operator visibility."""
    d = _coordinator(
        exec_place=lambda _p: None,
        exec_cancel=lambda _c: None,
    )
    d._place_lane.append(_place("a", Side.BUY, 1))
    d._place_lane.append(_place("b", Side.BUY, 2))
    d._flush_one_batch()

    stats = d.snapshot_stats()
    assert "same_side_inflight_deferrals" in stats
    assert int(stats["same_side_inflight_deferrals"]) == 1
