"""Inbound WebSocket timing checkpoints (private + public paths). Monotonic perf_counter throughout."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass(frozen=True, slots=True)
class InboundPrivateTiming:
    """One private WS userFills/orderUpdates event lifecycle (monotonic + optional wall + exchange ts)."""

    ws_recv_mono: float = 0.0
    ws_recv_wall: Optional[datetime] = None
    parse_start_mono: float = 0.0
    parse_end_mono: float = 0.0
    enqueue_mono: float = 0.0
    dequeue_mono: float = 0.0
    handler_start_mono: float = 0.0
    handler_end_mono: float = 0.0
    state_apply_mono: float = 0.0
    exchange_event_ms: int = 0


@dataclass(frozen=True, slots=True)
class InboundPublicTiming:
    """One public BBO frame lifecycle (direct callback path — no app queue)."""

    ws_recv_mono: float = 0.0
    ws_recv_wall: Optional[datetime] = None
    parse_start_mono: float = 0.0
    parse_end_mono: float = 0.0
    callback_start_mono: float = 0.0
    apply_mono: float = 0.0
    exchange_ts_ms: Optional[int] = None


def _ms(a: float, b: float) -> Optional[float]:
    if a <= 0.0 or b <= 0.0:
        return None
    return max(0.0, (b - a) * 1000.0)


def private_timing_derived_ms(t: InboundPrivateTiming) -> dict[str, Any]:
    out: dict[str, Any] = {
        # Time from socket receive to start of JSON parse (thread scheduling / queue before parse).
        "private_ws_receive_to_parse_ms": _ms(t.ws_recv_mono, t.parse_start_mono),
        "private_ws_parse_ms": _ms(t.parse_start_mono, t.parse_end_mono),
        "private_ws_parse_to_queue_ms": _ms(t.parse_end_mono, t.enqueue_mono),
        "private_ws_queue_wait_ms": _ms(t.enqueue_mono, t.dequeue_mono),
        "private_ws_handler_ms": _ms(t.handler_start_mono, t.handler_end_mono),
        "private_ws_receive_to_state_apply_ms": _ms(t.ws_recv_mono, t.state_apply_mono),
    }
    if t.exchange_event_ms > 0 and t.ws_recv_wall is not None:
        try:
            ev_dt = datetime.fromtimestamp(t.exchange_event_ms / 1000.0, tz=timezone.utc)
            delta_s = (t.ws_recv_wall - ev_dt).total_seconds()
            out["exchange_to_local_private_receive_ms"] = float(delta_s * 1000.0)
        except Exception:
            out["exchange_to_local_private_receive_ms"] = None
    else:
        out["exchange_to_local_private_receive_ms"] = None
    return out


def public_timing_derived_ms(t: InboundPublicTiming) -> dict[str, Any]:
    out: dict[str, Any] = {
        "public_ws_receive_to_parse_ms": _ms(t.ws_recv_mono, t.parse_end_mono),
        "public_ws_parse_ms": _ms(t.parse_start_mono, t.parse_end_mono),
        "public_ws_callback_ms": _ms(t.parse_end_mono, t.callback_start_mono),
        "public_ws_handler_ms": _ms(t.callback_start_mono, t.apply_mono),
        "public_ws_receive_to_apply_ms": _ms(t.ws_recv_mono, t.apply_mono),
        "public_ws_queue_wait_ms": 0.0,
    }
    if t.exchange_ts_ms is not None and t.ws_recv_wall is not None:
        try:
            ev_dt = datetime.fromtimestamp(t.exchange_ts_ms / 1000.0, tz=timezone.utc)
            delta_s = (t.ws_recv_wall - ev_dt).total_seconds()
            out["public_ws_exchange_to_local_receive_ms"] = float(delta_s * 1000.0)
        except Exception:
            out["public_ws_exchange_to_local_receive_ms"] = None
    else:
        out["public_ws_exchange_to_local_receive_ms"] = None
    return out
