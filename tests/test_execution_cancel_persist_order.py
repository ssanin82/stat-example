"""Phase 0b regression — ``submit_cancel`` runs BEFORE ``persist(wo)``.

The pre-1.4.0 ordering did ``transition → persist → submit_cancel``,
which serialized the cancel-on-the-wire moment behind a SQLite
write (median 0.5-3 ms, p99 ~30 ms under lock contention). With colo
cancel RTT at ~5 ms median, that DB write was ~10-60% of the path.

The new ordering does ``transition → submit_cancel → persist``. The
in-memory ``WorkingOrder`` already carries the CANCEL_PENDING state;
persist is only for restart recovery, which tolerates a millisecond-
scale lag.

Both cancel entry points changed:
  * ``cancel_order`` — synchronous, used by reconcile / risk-flatten
  * ``_enqueue_cancel_quote_path`` — hot quote path

The test patches ``self.persist`` and ``_outbound.submit_cancel``
with call-time recorders and asserts the relative order.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

from app.enums import OrderStatus, Side
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from app.execution import OrderManager
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings_db() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_persist_order_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "SYMBOL": "ETH_USDT_Perp",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    return s, path


def _bootstrap() -> tuple[OrderManager, Path]:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    # Return a benign cancel response so the synchronous path completes.
    client.cancel_order.return_value = {"result": {"ack": True}}
    client.cancel_order_by_cloid.return_value = {"result": {"ack": True}}
    client.interpret_cancel_response.return_value = ("success", "")
    om = OrderManager(settings, client, storage, state)
    return om, path


def _live_wo() -> WorkingOrder:
    """A WO that's been placed and acked (so the cancel-defer guard
    in ``_defer_cancel_if_unacked`` doesn't park the cancel)."""
    return WorkingOrder(
        order_id_local="L-cancel-order",
        order_id_exchange=1234567890,
        client_order_id="MM-cloid-1",
        symbol="ETH_USDT_Perp",
        side=Side.BUY,
        price=2400.00,
        size=0.01,
        post_only=True,
        # Status SENT + ts_ack set so the defer guard does NOT fire.
        status=OrderStatus.SENT,
        quote_cycle_id="qt",
    )


def test_cancel_order_submits_before_persist_sync_path() -> None:
    """The synchronous ``cancel_order`` path: ``_cancel_http_transport``
    (the actual HTTP) must run before ``persist(wo)``. We can't easily
    intercept ``_cancel_http_transport`` directly, but the client-side
    ``cancel_order`` mock fires inside it, so we record those call
    times against the persist call time."""
    om, path = _bootstrap()
    try:
        wo = _live_wo()
        # Mark the WO as ack-completed so the defer guard skips.
        from app.utils.time import utc_now
        wo.ts_sent = utc_now()
        wo.ts_ack = utc_now()
        with om._state._lock:
            om._state.working_bid = wo

        call_times: dict[str, float] = {}
        original_persist = om.persist
        original_cancel = om._client.cancel_order

        def spy_persist(target_wo: WorkingOrder) -> None:
            call_times.setdefault("persist", time.perf_counter())
            return original_persist(target_wo)

        def spy_cancel(*args, **kwargs):
            call_times.setdefault("http_cancel", time.perf_counter())
            return original_cancel(*args, **kwargs)

        with patch.object(om, "persist", side_effect=spy_persist):
            om._client.cancel_order.side_effect = spy_cancel
            om.cancel_order(wo, trigger_reason="test_phase_0b")

        assert "http_cancel" in call_times, "cancel HTTP was not called"
        assert "persist" in call_times, "persist was not called"
        # The HTTP cancel must have started BEFORE persist.
        assert call_times["http_cancel"] < call_times["persist"], (
            f"Phase 0b regression: cancel HTTP started at "
            f"{call_times['http_cancel']:.6f}, persist at "
            f"{call_times['persist']:.6f}. Persist must run AFTER the "
            f"HTTP cancel so SQLite I/O does not block the wire moment."
        )
    finally:
        path.unlink(missing_ok=True)


def test_enqueue_cancel_submits_before_persist_hot_path() -> None:
    """The hot quote path ``_enqueue_cancel_quote_path``: the
    dispatcher's ``submit_cancel`` must be called BEFORE ``persist``.
    """
    om, path = _bootstrap()
    try:
        wo = _live_wo()
        from app.utils.time import utc_now
        wo.ts_sent = utc_now()
        wo.ts_ack = utc_now()
        with om._state._lock:
            om._state.working_bid = wo

        call_times: dict[str, float] = {}
        original_persist = om.persist
        original_submit = om._outbound.submit_cancel

        def spy_persist(target_wo: WorkingOrder) -> None:
            call_times.setdefault("persist", time.perf_counter())
            return original_persist(target_wo)

        def spy_submit(intent) -> None:
            call_times.setdefault("submit_cancel", time.perf_counter())
            return original_submit(intent)

        with patch.object(om, "persist", side_effect=spy_persist):
            with patch.object(
                om._outbound, "submit_cancel", side_effect=spy_submit
            ):
                ok = om._enqueue_cancel_quote_path(
                    wo, trigger_reason="test_phase_0b"
                )
                assert ok is True

        assert "submit_cancel" in call_times, "submit_cancel was not called"
        assert "persist" in call_times, "persist was not called"
        assert call_times["submit_cancel"] < call_times["persist"], (
            f"Phase 0b regression: submit_cancel started at "
            f"{call_times['submit_cancel']:.6f}, persist at "
            f"{call_times['persist']:.6f}. The dispatcher enqueue "
            f"must happen BEFORE persist so the cancel goes onto the "
            f"queue without waiting for SQLite I/O."
        )
    finally:
        path.unlink(missing_ok=True)
