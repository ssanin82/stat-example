"""amend-prio Phase 4 (v1.4.17) — observability + reconcile awareness.

Counters: every amend outcome must bump exactly the right counter on
``BotState``. The dispatcher-side dispatch counters are tested in
test_outbound_dispatch_amend.py; this file covers the OrderManager-
level rollout counters (success / below_filled / order_gone /
post_only_cross / exchange_rejected_other / transport_rejected /
intents_emitted_total / pending_high_watermark).

Reconcile awareness: when a working order is in AMEND_PENDING and
the venue reports a different size on the same ordId (because the
amend has already been applied at the matching engine but the bot
hasn't seen the response yet), the legacy reconcile would treat
``remote.sz < wo.size`` as a partial fill and force the WO to
PARTIAL. That corrupts the amend lifecycle. The Phase 4 fix:
reconcile skips ALL mutations on AMEND_PENDING WOs — the amend
response handler is the only thing that should transition the WO
out of AMEND_PENDING.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.enums import OrderStatus, Side
from app.execution import OrderManager
from app.models import BestBidAsk, WorkingOrder
from app.outbound_dispatch import PlaceTransportIntent
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _setup() -> tuple[OrderManager, Path, Any]:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_amend_phase4_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "OKX_AMEND_ON_REPRICE_ENABLED": True,
            "STRICT_PLACE_UNCONFIRMED_KILL": False,
            # v1.4.169 Phase 2I — this test fixture issues back-to-back
            # amends on the same WO to verify the intent-emit counter
            # bumps. The new per-order amend rate-defence guard would
            # correctly suppress those rapid amends by design (its
            # whole purpose is to stop the v1.4.102 244-amends-in-1-s
            # runaway), which would defeat THIS test's purpose. Disable
            # the guard here so the counter-bump behaviour is testable.
            # Phase 2I itself is covered by
            # ``tests/test_phase2i_amend_rate_defence.py``.
            "AMEND_TICK_FLICKER_MIN_MS": 0.0,
            "AMEND_PER_ORDER_MAX_PER_SEC": 200,
        }
    )
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = BestBidAsk(
        symbol=s.symbol,
        best_bid=3000.0,
        best_ask=3001.0,
        mid_price=3000.5,
        spread_bps=10.0,
        ts_local=utc_now(),
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(s, client, storage, state, private_event_queue=None)
    return om, path, client


def _amend_pending(om: OrderManager) -> WorkingOrder:
    wo = WorkingOrder(
        order_id_local=f"local-{uuid.uuid4().hex[:8]}",
        order_id_exchange=12345,
        client_order_id="cl_" + uuid.uuid4().hex[:24],
        symbol=om._settings.symbol,
        side=Side.BUY,
        price=3000.0,
        size=0.02,
        post_only=True,
        status=OrderStatus.AMEND_PENDING,
        ts_created=datetime.now(timezone.utc),
        ts_sent=datetime.now(timezone.utc),
        ts_ack=datetime.now(timezone.utc),
    )
    wo.amend_intent_seq = 1
    wo.amend_target_px = 2999.5
    wo.amend_target_sz = 0.03
    with om._state._lock:
        om._state.set_working_order(Side.BUY, 0, wo)
    return wo


def _intent_for(wo: WorkingOrder) -> PlaceTransportIntent:
    import time as time_mod
    return PlaceTransportIntent(
        wo_order_id_local=wo.order_id_local,
        side=wo.side,
        intent_seq=int(wo.amend_intent_seq),
        quote_cycle_id="amend",
        enqueued_mono=time_mod.monotonic(),
        level_idx=0,
        kind="amend",
    )


# ----------------------------------------------------------------------
# Per-outcome counters
# ----------------------------------------------------------------------


def test_counter_success_increments() -> None:
    om, path, client = _setup()
    try:
        wo = _amend_pending(om)
        client.amend_batch_orders.return_value = {
            "code": "0", "msg": "", "data": [
                {"sCode": "0", "ordId": "12345",
                 "reqId": "a12345s1", "clOrdId": wo.client_order_id},
            ],
        }
        om._execute_amend_batch_intents([_intent_for(wo)])
        assert om._state.amend_success_total == 1
        assert om._state.amend_below_filled_total == 0
        assert om._state.amend_order_gone_total == 0
        assert om._state.amend_transport_rejected_total == 0
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_counter_below_filled_increments() -> None:
    om, path, client = _setup()
    try:
        wo = _amend_pending(om)
        client.amend_batch_orders.return_value = {
            "code": "0", "msg": "", "data": [
                {"sCode": "51016", "sMsg": "below filled",
                 "reqId": "a12345s1", "clOrdId": wo.client_order_id},
            ],
        }
        om._execute_amend_batch_intents([_intent_for(wo)])
        assert om._state.amend_below_filled_total == 1
        assert om._state.amend_success_total == 0
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_counter_order_gone_increments() -> None:
    om, path, client = _setup()
    try:
        wo = _amend_pending(om)
        client.amend_batch_orders.return_value = {
            "code": "0", "msg": "", "data": [
                {"sCode": "51400", "sMsg": "does not exist",
                 "reqId": "a12345s1", "clOrdId": wo.client_order_id},
            ],
        }
        om._execute_amend_batch_intents([_intent_for(wo)])
        assert om._state.amend_order_gone_total == 1
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_counter_post_only_cross_increments() -> None:
    om, path, client = _setup()
    try:
        wo = _amend_pending(om)
        client.amend_batch_orders.return_value = {
            "code": "0", "msg": "", "data": [
                {"sCode": "51604", "sMsg": "Post-only would cross",
                 "reqId": "a12345s1", "clOrdId": wo.client_order_id},
            ],
        }
        om._execute_amend_batch_intents([_intent_for(wo)])
        assert om._state.amend_post_only_cross_total == 1
        # The post_only_cross case is NOT counted as "exchange_rejected_other".
        assert om._state.amend_exchange_rejected_other_total == 0
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_counter_exchange_rejected_other_increments() -> None:
    """Any sCode other than 51016/51400/51401/51503/51604/50011 lands
    in exchange_rejected_other."""
    om, path, client = _setup()
    try:
        wo = _amend_pending(om)
        client.amend_batch_orders.return_value = {
            "code": "0", "msg": "", "data": [
                {"sCode": "51000", "sMsg": "parameter error",
                 "reqId": "a12345s1", "clOrdId": wo.client_order_id},
            ],
        }
        om._execute_amend_batch_intents([_intent_for(wo)])
        assert om._state.amend_exchange_rejected_other_total == 1
        assert om._state.amend_post_only_cross_total == 0
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_counter_transport_rejected_increments() -> None:
    om, path, client = _setup()
    try:
        wo = _amend_pending(om)
        client.amend_batch_orders.return_value = {
            "code": "0", "msg": "", "data": [
                {"sCode": "50011", "sMsg": "rate_limit",
                 "reqId": "a12345s1", "clOrdId": wo.client_order_id},
            ],
        }
        om._execute_amend_batch_intents([_intent_for(wo)])
        assert om._state.amend_transport_rejected_total == 1
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_counter_transport_envelope_bumps_all_wos() -> None:
    """An envelope-level transport failure reverts ALL WOs and bumps
    the transport counter by N."""
    om, path, client = _setup()
    try:
        wo1 = _amend_pending(om)
        # Build a second WO on the SELL side.
        wo2 = WorkingOrder(
            order_id_local=f"local-{uuid.uuid4().hex[:8]}",
            order_id_exchange=12346,
            client_order_id="cl_" + uuid.uuid4().hex[:24],
            symbol=om._settings.symbol,
            side=Side.SELL,
            price=3001.0, size=0.02, post_only=True,
            status=OrderStatus.AMEND_PENDING,
            ts_created=datetime.now(timezone.utc),
            ts_sent=datetime.now(timezone.utc),
            ts_ack=datetime.now(timezone.utc),
        )
        wo2.amend_intent_seq = 1
        wo2.amend_target_px = 3001.5
        wo2.amend_target_sz = 0.03
        with om._state._lock:
            om._state.set_working_order(Side.SELL, 0, wo2)
        client.amend_batch_orders.return_value = "garbage_envelope"
        om._execute_amend_batch_intents([_intent_for(wo1), _intent_for(wo2)])
        assert om._state.amend_transport_rejected_total == 2
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_counter_intents_emitted_bumps_on_enqueue() -> None:
    om, path, _client = _setup()
    try:
        om._outbound.stop()  # prevent dispatch race
        wo = WorkingOrder(
            order_id_local=f"local-{uuid.uuid4().hex[:8]}",
            order_id_exchange=12345,
            client_order_id="cl_" + uuid.uuid4().hex[:24],
            symbol=om._settings.symbol,
            side=Side.BUY,
            price=3000.0, size=0.02, post_only=True,
            status=OrderStatus.ACKED,
            ts_created=datetime.now(timezone.utc),
            ts_sent=datetime.now(timezone.utc),
            ts_ack=datetime.now(timezone.utc),
        )
        with om._state._lock:
            om._state.set_working_order(Side.BUY, 0, wo)
        from app.quote_engine import FinalQuoteOrder
        desired = FinalQuoteOrder(side=Side.BUY, price=2999.0, size=0.03)
        om._enqueue_amend_quote_path(wo, desired)
        assert om._state.amend_intents_emitted_total == 1
        assert om._state.amend_pending_high_watermark >= 1
        # Second amend on a fresh WO.
        wo.status = OrderStatus.ACKED  # reset for second emit
        wo.amend_target_px = None
        wo.amend_target_sz = None
        om._enqueue_amend_quote_path(wo, desired)
        assert om._state.amend_intents_emitted_total == 2
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_amend_to_cancel_fallback_total_is_sum() -> None:
    """``amend_to_cancel_fallback_total`` in the snapshot is the sum
    of below_filled + exchange_rejected_other (the two outcomes that
    trigger the cancel-then-place fallback path)."""
    om, path, _client = _setup()
    try:
        with om._state._lock:
            om._state.amend_below_filled_total = 3
            om._state.amend_exchange_rejected_other_total = 5
        snap = om._state.snapshot_dict()
        assert snap["amend_to_cancel_fallback_total"] == 8
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Reconcile awareness
# ----------------------------------------------------------------------


def test_reconcile_skips_amend_pending_size_mismatch() -> None:
    """The pivotal Phase 4 fix: reconcile must NOT force AMEND_PENDING
    → PARTIAL when ``remote.sz < wo.size``. Without the fix, the
    amend lifecycle gets clobbered before the response handler can
    commit the target."""
    from app.exchange.base import OpenOrderRaw as OpenOrder
    om, path, _client = _setup()
    try:
        wo = _amend_pending(om)
        # Pre-Phase-4 the legacy ``remote.sz < wo.size`` would force
        # PARTIAL when the amend was a size REDUCTION. Build that case
        # explicitly: original size 3.0, amend_target_sz 1.5.
        wo.size = 3.0
        wo.amend_target_sz = 1.5
        remote = OpenOrder(
            coin=om._settings.symbol,
            side=Side.BUY,
            oid=wo.order_id_exchange,
            limit_px=wo.price,
            sz=1.5,
            timestamp=0,
            cloid=wo.client_order_id,
        )
        # Call the reconcile-side branch directly.
        rest_request_dispatched_at = datetime.now(timezone.utc)
        result = om._reconcile_side(
            Side.BUY,
            remote,
            rest_request_dispatched_at=rest_request_dispatched_at,
        )
        # AMEND_PENDING must be preserved; size must NOT be clobbered.
        assert wo.status == OrderStatus.AMEND_PENDING
        assert wo.size == 3.0  # untouched — amend_target_sz commit is the response handler's job
        assert wo.amend_target_sz == 1.5  # still pending commit
        assert result is False  # not a mismatch
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_reconcile_acked_still_triggers_partial_on_size_mismatch() -> None:
    """Regression: the Phase 4 reconcile fix must ONLY skip
    AMEND_PENDING. A genuine partial fill on an ACKED order must
    still trigger PARTIAL transition."""
    from app.exchange.base import OpenOrderRaw as OpenOrder
    om, path, _client = _setup()
    try:
        wo = _amend_pending(om)
        # Switch to ACKED to simulate a real partial-fill scenario.
        wo.status = OrderStatus.ACKED
        wo.amend_target_px = None
        wo.amend_target_sz = None
        wo.size = 3.0
        remote = OpenOrder(
            coin=om._settings.symbol,
            side=Side.BUY,
            oid=wo.order_id_exchange,
            limit_px=wo.price,
            sz=1.5,  # partial fill: half filled
            timestamp=0,
            cloid=wo.client_order_id,
        )
        rest_request_dispatched_at = datetime.now(timezone.utc)
        result = om._reconcile_side(
            Side.BUY,
            remote,
            rest_request_dispatched_at=rest_request_dispatched_at,
        )
        assert wo.status == OrderStatus.PARTIAL
        assert wo.size == 1.5
        assert result is False
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Snapshot dict surface
# ----------------------------------------------------------------------


def test_snapshot_dict_includes_amend_counters() -> None:
    om, path, _client = _setup()
    try:
        snap = om._state.snapshot_dict()
        for key in (
            "amend_intents_emitted_total",
            "amend_success_total",
            "amend_below_filled_total",
            "amend_order_gone_total",
            "amend_post_only_cross_total",
            "amend_exchange_rejected_other_total",
            "amend_transport_rejected_total",
            "amend_pending_high_watermark",
            "amend_to_cancel_fallback_total",
        ):
            assert key in snap, f"snapshot_dict missing {key!r}"
            assert snap[key] == 0  # all zero by default
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_status_flags_dict_includes_amend_counters() -> None:
    om, path, _client = _setup()
    try:
        flags = om._state.status_flags_dict()
        for key in (
            "amend_intents_emitted_total",
            "amend_success_total",
            "amend_below_filled_total",
        ):
            assert key in flags
            assert flags[key] == 0
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)


def test_snapshot_dict_includes_okx_rate_window_per_pool() -> None:
    """rate-limit-observability Phase 2 (v1.4.20): the per-pool
    snapshot must be present in ``state_current.json``. Empty dict
    by default (the bot heartbeat handler populates it each tick
    from the OKX adapter's ``rest_runtime_counters()``)."""
    om, path, _client = _setup()
    try:
        snap = om._state.snapshot_dict()
        assert "okx_rate_window_per_pool" in snap
        assert snap["okx_rate_window_per_pool"] == {}
        # Simulate the bot heartbeat handler pushing live data.
        om._state.okx_rate_window_per_pool = {
            "aggregate": {
                "current_2s": 80, "peak_2s_60s": 384, "total": 3452,
                "cap": None, "pct_of_cap": None,
            },
            "place_batch": {
                "current_2s": 24, "peak_2s_60s": 180, "total": 1683,
                "cap": 300, "pct_of_cap": 0.6,
            },
        }
        snap2 = om._state.snapshot_dict()
        assert (
            snap2["okx_rate_window_per_pool"]["place_batch"]["pct_of_cap"]
            == 0.6
        )
        assert (
            snap2["okx_rate_window_per_pool"]["aggregate"]["peak_2s_60s"]
            == 384
        )
    finally:
        # Windows: SQLite holds a file handle until the Storage
        # instance is closed. Without an explicit close the unlink
        # below races the GC and raises PermissionError under load
        # (observed in CI 2026-05-21, full-suite 500s run).
        try:
            om._storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)
