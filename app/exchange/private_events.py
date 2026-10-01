"""Normalized internal events from Hyperliquid private websocket (no raw payloads here)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from app.inbound_timing import InboundPrivateTiming


class PrivateWsConnectionKind(str, Enum):
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECT_SCHEDULED = "reconnect_scheduled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PrivateWsConnectionEvent:
    kind: PrivateWsConnectionKind
    detail: str = ""
    backoff_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PrivateOrderUpdateEvent:
    """One orderUpdates entry (WsOrder) after parsing.

    ``cloid`` is the ``metadata.client_order_id`` carried by GRVT on every
    order update. It is optional because not every venue emits it (HL does
    not). When present it lets the execution layer bind a local
    ``WorkingOrder`` whose ``order_id_exchange`` is still ``None`` — the
    synchronous GRVT ``create_order`` response does not always populate the
    exchange oid, so order-state-machine events would otherwise arrive
    unmatchable.
    """

    oid: int
    coin: str
    status: str
    status_timestamp_ms: int
    side: str  # "B" / "A" or HL string
    limit_px: float
    remaining_sz: float
    orig_sz: float
    raw_status: str
    inbound_timing: Optional[InboundPrivateTiming] = None
    cloid: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PrivateFillEvent:
    """One userFills entry (WsFill) after parsing."""

    fill_id: str
    oid: Optional[int]
    coin: str
    px: float
    sz: float
    side: str
    time_ms: int
    fee: float
    closed_pnl: float
    crossed: bool
    is_snapshot: bool
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    inbound_timing: Optional[InboundPrivateTiming] = None


PrivateStreamEvent = Union[
    PrivateWsConnectionEvent,
    PrivateOrderUpdateEvent,
    PrivateFillEvent,
]
