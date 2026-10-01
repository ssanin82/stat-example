"""Per-order outbound action tracing (timestamps + derived latency fields)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class OutboundActionTrace:
    """Mutable trace for one working order (place + cancel lifecycle)."""

    order_id_local: str
    side_value: str
    intent_created_perf: float = 0.0
    dispatcher_enqueue_perf: float = 0.0
    batch_selected_perf: float = 0.0
    sign_start_perf: float = 0.0
    sign_end_perf: float = 0.0
    transport_send_perf: float = 0.0
    transport_write_done_perf: float = 0.0
    exchange_ack_perf: float = 0.0
    first_private_ws_lifecycle_perf: Optional[float] = None
    local_order_open_perf: Optional[float] = None
    cancel_intent_perf: Optional[float] = None
    cancel_sent_perf: Optional[float] = None
    cancel_ack_or_closed_perf: Optional[float] = None
    quote_cycle_id: Optional[str] = None
    transport_mode: str = "http"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metrics_dict(self, *, now_perf: float) -> dict[str, Any]:
        """Derived ms fields (best-effort; monotonic-based intervals)."""
        ie = _ms(self.intent_created_perf, self.dispatcher_enqueue_perf)
        eb = _ms(self.dispatcher_enqueue_perf, self.batch_selected_perf)
        bw = _ms(self.batch_selected_perf, self.transport_send_perf)
        sg = _ms(self.sign_start_perf, self.sign_end_perf)
        sa = _ms(self.transport_send_perf, self.exchange_ack_perf)
        al = None
        if self.first_private_ws_lifecycle_perf is not None:
            al = _ms(self.exchange_ack_perf, self.first_private_ws_lifecycle_perf)
        pit = _ms(self.intent_created_perf, self.exchange_ack_perf)
        cc = None
        if self.cancel_intent_perf is not None and self.cancel_ack_or_closed_perf is not None:
            cc = _ms(self.cancel_intent_perf, self.cancel_ack_or_closed_perf)
        return {
            "intent_to_enqueue_ms": ie,
            "enqueue_to_batch_ms": eb,
            "batch_wait_ms": bw,
            "signing_ms": sg,
            "send_to_ack_ms": sa,
            "ack_to_first_lifecycle_ms": al,
            "place_intent_to_ack_ms": pit,
            "cancel_intent_to_closed_ms": cc,
        }


def _ms(a: float, b: float) -> Optional[float]:
    if a <= 0.0 or b <= 0.0:
        return None
    return max(0.0, (b - a) * 1000.0)
