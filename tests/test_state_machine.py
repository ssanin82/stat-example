import uuid

from app.enums import OrderStatus, Side
from app.execution import transition
from app.models import WorkingOrder


def test_order_lifecycle_ack_to_canceled() -> None:
    wo = WorkingOrder(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=None,
        client_order_id=None,
        symbol="ETH",
        side=Side.BUY,
        price=100.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.NEW_LOCAL,
    )
    transition(wo, OrderStatus.SENT)
    assert wo.status == OrderStatus.SENT
    transition(wo, OrderStatus.ACKED)
    assert wo.status == OrderStatus.ACKED
    transition(wo, OrderStatus.CANCEL_PENDING)
    assert wo.status == OrderStatus.CANCEL_PENDING
    transition(wo, OrderStatus.CANCELED, "test")
    assert wo.status == OrderStatus.CANCELED
    assert wo.ts_closed is not None


def test_rejected_sets_closed_timestamp() -> None:
    wo = WorkingOrder(
        order_id_local="a",
        order_id_exchange=None,
        client_order_id=None,
        symbol="ETH",
        side=Side.SELL,
        price=101.0,
        size=0.1,
        post_only=True,
        status=OrderStatus.SENT,
    )
    transition(wo, OrderStatus.REJECTED, "post_only_reject")
    assert wo.status == OrderStatus.REJECTED
    assert wo.ts_closed is not None
