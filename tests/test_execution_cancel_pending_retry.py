"""Cancel-pending watchdog must re-issue the cancel, rate-limited.

Regression guard for ``tmp/snap_20260417_173125`` where a GRVT BUY order
landed on the exchange (oid bound from the place response), the bot
transitioned it to CANCEL_PENDING locally, and then the first cancel
either was silently eaten by GRVT or the terminal WS event was dropped.
The old watchdog only set ``side_unresolved`` and requested a reconcile;
reconcile just observed the still-live order (oid match, status
CANCEL_PENDING is a no-op branch) and returned — so the BUY stayed
CANCEL_PENDING *forever*, ``has_order_state_uncertainty`` stayed True,
eligibility pinned HOLD_ALL, and the SELL side never replenished.

The retry closes the loop by re-dispatching the HTTP cancel; reconcile
then has a fresh chance to observe the order-gone state.
"""

from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from app.enums import OrderStatus, Side
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from app.execution import OrderManager
from app.models import WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings_db() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_cp_retry_{uuid.uuid4().hex}.db"
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


def _stuck_wo() -> WorkingOrder:
    return WorkingOrder(
        order_id_local="L-stuck",
        order_id_exchange=1334440892961932997811884864574694391,
        client_order_id="17376089106202547292",
        symbol="ETH_USDT_Perp",
        side=Side.BUY,
        price=2428.24,
        size=0.01,
        post_only=True,
        status=OrderStatus.CANCEL_PENDING,
        quote_cycle_id="qt",
    )


def _bootstrap(timeout_s: float = 0.01) -> tuple[OrderManager, Path]:
    settings, path = _settings_db()
    settings.cancel_pending_unresolved_timeout_seconds = timeout_s
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    client.cancel_order.return_value = {"result": {"ack": True}}
    client.interpret_cancel_response.return_value = ("success", "")
    om = OrderManager(settings, client, storage, state)
    return om, path


def test_timeout_dispatches_cancel_retry() -> None:
    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        # Age > timeout threshold.
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        # Retry dispatched: cancel_order called with the bound oid.
        om._client.cancel_order.assert_called_once_with(
            "ETH_USDT_Perp", 1334440892961932997811884864574694391
        )
        assert om._cancel_pending_retry_dispatch_count == 1
        # Retry throttle is keyed per-side (the other side's watchdog running
        # with cur=None used to wipe this entry and cause a cancel storm).
        assert om._cancel_pending_retry_last_mono[Side.BUY] > 0.0
        assert om._cancel_pending_retry_last_mono[Side.SELL] == 0.0
    finally:
        path.unlink(missing_ok=True)


def test_retry_is_rate_limited_on_subsequent_ticks() -> None:
    """The watchdog is called per ``_orchestrate`` (2×/tick). Rate-limit must
    prevent every call from firing an HTTP cancel."""
    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile"):
            for _ in range(10):
                om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        # Rate-limit is max(2s, timeout/3) — with timeout=0.01 this is 2s.
        # All 10 calls happen in the same monotonic instant, so only the
        # first should dispatch.
        assert om._client.cancel_order.call_count == 1
        assert om._cancel_pending_retry_dispatch_count == 1
    finally:
        path.unlink(missing_ok=True)


def test_retry_fires_again_after_rate_limit_interval() -> None:
    """After the rate-limit window passes, a second retry dispatches.
    With the new threshold=1 the second retry escalates to cloid, so
    we verify: 1 oid call + 1 cloid call (one dispatch per window)."""
    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 1
        assert om._client.cancel_order_by_cloid.call_count == 0
        # Rewind the retry bookkeeping past the rate-limit window to simulate
        # time passing without sleeping the test.
        om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        # Second retry escalates to cloid — so oid count stays at 1,
        # cloid count goes to 1. Total dispatches = 2.
        assert om._client.cancel_order.call_count == 1
        assert om._client.cancel_order_by_cloid.call_count == 1
    finally:
        path.unlink(missing_ok=True)


def test_no_retry_when_wo_not_cancel_pending() -> None:
    """If the WO has left CANCEL_PENDING (terminal, or replaced) the watchdog
    must NOT dispatch a cancel against it."""
    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        wo.status = OrderStatus.ACKED
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile") as req_m:
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        req_m.assert_not_called()
        om._client.cancel_order.assert_not_called()
        assert om._cancel_pending_retry_dispatch_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_no_retry_before_timeout_threshold() -> None:
    """Before the 15s timeout threshold, nothing retries — the WS CANCELED
    event is still expected to arrive on the normal path."""
    settings, path = _settings_db()
    # Big timeout — the retry block must be unreached.
    settings.cancel_pending_unresolved_timeout_seconds = 30.0
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state)
    try:
        wo = _stuck_wo()
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile") as req_m:
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        req_m.assert_not_called()
        assert om._client.cancel_order.call_count == 0
        assert om._cancel_pending_retry_dispatch_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_other_side_cleanup_does_not_reset_active_side_retry_throttle() -> None:
    """Regression guard for ``tmp/snap_20260417_180429``: the watchdog's
    cleanup branch on the OTHER side (``cur=None``) used to wipe the whole
    retry dict, resetting the active side's rate-limit to 0 and letting
    every tick fire a fresh cancel HTTP. 132 retries in 33 s of one stuck
    order. Keying the throttle per-side makes the cleanup side-local.
    """
    om, path = _bootstrap()
    try:
        # SELL is the stuck side, BUY is empty (typical after a BUY fill).
        sell_wo = _stuck_wo()
        sell_wo.side = Side.SELL
        with om._state._lock:
            om._state.working_ask = sell_wo
            om._state.working_bid = None
        om._cancel_pending_since_mono[Side.SELL] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile"):
            # SELL watchdog fires, retries, writes the throttle timestamp.
            om._maybe_handle_cancel_pending_timeout(Side.SELL, sell_wo)
        first_retry_ts = om._cancel_pending_retry_last_mono[Side.SELL]
        assert first_retry_ts > 0.0
        assert om._client.cancel_order.call_count == 1

        # Now the BUY-side watchdog runs with cur=None (slot empty). The
        # cleanup branch must ONLY reset the BUY throttle.
        om._maybe_handle_cancel_pending_timeout(Side.BUY, None)
        assert om._cancel_pending_retry_last_mono[Side.SELL] == first_retry_ts
        assert om._cancel_pending_retry_last_mono[Side.BUY] == 0.0

        # The immediately-following SELL tick within the rate-limit window
        # must NOT fire another retry.
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.SELL, sell_wo)
        assert om._client.cancel_order.call_count == 1
    finally:
        path.unlink(missing_ok=True)


def test_first_retry_uses_oid_cancel_then_escalates() -> None:
    """The first retry goes through ``cancel_order`` (by oid) — cheap
    idempotent path covers a one-off GRVT ack glitch. The SECOND retry
    escalates to ``cancel_order_by_cloid`` (with ``time_to_live_ms``)
    because ``_CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS = 1``.

    Lowered from 2 to 1 based on logs.1776583271526.json — two stuck
    orders required 3 attempts each (last being cloid-based) and stayed
    stuck ~2.5 minutes because oid-based cancels were 200-OK-but-no-op.
    """
    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile"):
            # 1st retry — oid path.
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 1
        assert om._client.cancel_order_by_cloid.call_count == 0
        om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
        with patch.object(om, "request_open_orders_reconcile"):
            # 2nd retry — escalated to cloid.
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 1
        assert om._client.cancel_order_by_cloid.call_count == 1
        assert om._cancel_pending_retry_attempts[Side.BUY] == 2
    finally:
        path.unlink(missing_ok=True)


def test_later_retries_stay_on_cloid_cancel_path() -> None:
    """After the first oid-cancel fails, subsequent retries all use
    ``cancel_order_by_cloid`` (carrying ``time_to_live_ms``). Regression
    guard for two failure modes:
      * ``tmp/snap_20260417_183547`` — GRVT acked oid-cancels without
        actually cancelling (~11 retries, position paralysed).
      * ``logs.1776583271526.json`` — 2.5-minute stuck cancel that only
        cleared on attempt 3 (the first cloid-based retry). Threshold
        lowered from 2 → 1 so cloid path kicks in one retry earlier.
    """
    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile"):
            # 1st — oid
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
            om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
            # 2nd — cloid (first escalated retry)
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
            om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
            # 3rd — cloid
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
            om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
            # 4th — cloid
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 1
        assert om._client.cancel_order_by_cloid.call_count == 3
        om._client.cancel_order_by_cloid.assert_called_with(
            "ETH_USDT_Perp", "17376089106202547292"
        )
        assert om._cancel_pending_retry_attempts[Side.BUY] == 4
    finally:
        path.unlink(missing_ok=True)


def test_escalated_retry_logs_escalation_flag(caplog) -> None:
    """The 2nd retry (first escalated one) logs
    ``escalated_to_cloid=True`` with ``attempts_this_episode=2``.
    Threshold lowered from 2 → 1 — see constant docstring in
    ``app/execution.py``.
    """
    import logging as _logging

    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile"):
            # 1st retry — oid path (silent on the escalation flag).
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
            om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
            # 2nd retry — escalated to cloid.
            with caplog.at_level(_logging.WARNING, logger="app.execution"):
                om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        recs = [r for r in caplog.records if r.getMessage() == "cancel_pending_retry_dispatched"]
        assert recs, "expected cancel_pending_retry_dispatched on the 2nd attempt"
        extra = getattr(recs[-1], "extra_data", {}) or {}
        assert extra.get("escalated_to_cloid") is True
        assert extra.get("attempts_this_episode") == 2
    finally:
        path.unlink(missing_ok=True)


def test_retry_attempts_counter_resets_when_wo_leaves_cancel_pending() -> None:
    """A brand-new CANCEL_PENDING episode must start at attempt 1 (not
    escalated). Otherwise a reprice after a clean cancel would skip the
    cheap oid path."""
    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
            om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
            om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)  # escalated
        assert om._cancel_pending_retry_attempts[Side.BUY] == 3

        # WO leaves CANCEL_PENDING (e.g. terminal, or replaced).
        wo.status = OrderStatus.CANCELED
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._cancel_pending_retry_attempts[Side.BUY] == 0
    finally:
        path.unlink(missing_ok=True)


def test_cloid_escalation_threshold_pins_at_one() -> None:
    """Pin the ``_CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS`` constant
    at 1. Any change to this value shifts real-world cancel latency
    during GRVT slow-ack episodes by tens of seconds per episode and
    should be deliberate. Lower = escalate sooner (less stuck time);
    higher = more oid-retries before switching.

    Evidence basis: logs.1776583271526.json showed 2.5-minute stuck
    cancels that cleared ONLY on the first cloid-based retry; oid-based
    retries returned 200 OK but were no-ops. Lowering from 2 → 1 cuts
    expected stuck time in half."""
    from app.execution import _CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS

    assert _CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS == 1, (
        "cloid escalation threshold changed — review cancel-latency impact "
        "(see constant docstring in app/execution.py for rationale)"
    )


def test_cancel_retry_transport_exception_does_not_break_the_watchdog() -> None:
    """A failing cancel HTTP on the retry must not crash the watchdog —
    the next tick must still be able to retry."""
    om, path = _bootstrap()
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
        om._client.cancel_order.side_effect = RuntimeError("transport boom")
        om._client.interpret_cancel_response.return_value = ("transport", "boom")
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        # Attempt was made; exception was swallowed into bump_execution_errors.
        assert om._client.cancel_order.call_count == 1
        assert om._cancel_pending_retry_dispatch_count == 1
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# 1.3.117: cloid-escalation disable flag (CANCEL_PENDING_CLOID_ESCALATION_ENABLED).
# Required on OKX colo deployments where the WS cancel-by-cloid frame
# triggers sCode 50014 "Parameter instIdCode can not be empty" because
# the colo endpoint uses numeric instIdCode for cloid lookup, not the
# string instId we send. See plans/_DONE/sbe.md §2.
# --------------------------------------------------------------------------


def test_cloid_escalation_disabled_keeps_retries_on_ordid_path() -> None:
    """When ``CANCEL_PENDING_CLOID_ESCALATION_ENABLED=false``, the
    second retry must NOT escalate to cancel-by-cloid — it must keep
    calling ``cancel_order`` (ordId path). This is the OKX-colo-safe
    mode: cloid-cancel WS frames fail on the colo endpoint, so the
    bot stays on ordId-only cancels which DO work on colo WS."""
    settings, path = _settings_db()
    settings.cancel_pending_unresolved_timeout_seconds = 0.01
    # Disable the escalation — simulates the OKX SUI profile.
    settings.cancel_pending_cloid_escalation_enabled = False
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    client.cancel_order.return_value = {"result": {"ack": True}}
    client.interpret_cancel_response.return_value = ("success", "")
    om = OrderManager(settings, client, storage, state)
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0

        # First retry — uses ordId path (attempts_this_episode=1,
        # below the cloid-escalation threshold of 1).
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 1
        assert om._client.cancel_order_by_cloid.call_count == 0

        # Second retry — WOULD escalate to cloid by default; with the
        # flag disabled, must stay on ordId.
        om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 2, (
            "second retry must hit cancel_order (ordId path) when "
            "escalation is disabled"
        )
        assert om._client.cancel_order_by_cloid.call_count == 0, (
            "cancel_order_by_cloid must NOT be called when escalation "
            "is disabled — that's the path that triggers OKX sCode "
            "50014 instIdCode-missing on colo WS"
        )

        # Third retry — same. Must NEVER escalate.
        om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 3
        assert om._client.cancel_order_by_cloid.call_count == 0
    finally:
        path.unlink(missing_ok=True)


def test_cloid_escalation_default_true_preserves_legacy_behavior() -> None:
    """When the flag is unset (or True), the legacy escalation path
    fires as before — second retry escalates to cloid. This protects
    GRVT and any non-OKX adapters that rely on the cloid-escalation
    workaround for their own ack-but-no-cancel quirks."""
    settings, path = _settings_db()
    settings.cancel_pending_unresolved_timeout_seconds = 0.01
    # Explicit True (matches the code default).
    settings.cancel_pending_cloid_escalation_enabled = True
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    client.cancel_order.return_value = {"result": {"ack": True}}
    client.cancel_order_by_cloid.return_value = {"result": {"ack": True}}
    client.interpret_cancel_response.return_value = ("success", "")
    om = OrderManager(settings, client, storage, state)
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0

        # First retry — ordId path.
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 1
        assert om._client.cancel_order_by_cloid.call_count == 0

        # Second retry — legacy escalates to cloid (above the
        # _CANCEL_PENDING_CLOID_ESCALATE_AFTER_ATTEMPTS=1 threshold).
        om._cancel_pending_retry_last_mono[Side.BUY] = time.monotonic() - 10.0
        with patch.object(om, "request_open_orders_reconcile"):
            om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
        assert om._client.cancel_order.call_count == 1
        assert om._client.cancel_order_by_cloid.call_count == 1
    finally:
        path.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# 1.3.121: CANCEL_TRUST_TRADE_WS_SUCCESS_TERMINAL — when True, the
# bot transitions the WO to CANCELED locally on trade-WS sCode 0
# success WITHOUT waiting for the inbound user-data WS event. Fixes
# the OKX-colo stall where user-data WS lags 90+ seconds.
# --------------------------------------------------------------------------


def test_trust_trade_ws_success_transitions_wo_locally() -> None:
    """With the flag enabled, a successful cancel response causes the
    WO to transition to CANCELED locally (not wait for inbound WS).
    working_bid/ask cleared, side_unresolved cleared, side resumes
    quoting in microseconds instead of 30-97 seconds."""
    settings, path = _settings_db()
    settings.cancel_trust_trade_ws_success_terminal = True
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client(symbol_spec=FALLBACK_SYMBOL_SPEC)
    client.has_write_access.return_value = True
    client.cancel_order.return_value = {"code": "0", "msg": "", "data": []}
    client.interpret_cancel_response.return_value = ("success", "")
    om = OrderManager(settings, client, storage, state)
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        # Manually mark side unresolved (as the cancel-pending path
        # would have).
        om._side_unresolved_active[Side.BUY] = True
        om._side_unresolved_reason[Side.BUY] = "cancel_pending_wait"
        om._side_unresolved_since_mono[Side.BUY] = time.monotonic()
        om._cancel_http_transport(wo)
        # WO transitioned to CANCELED locally.
        assert wo.status == OrderStatus.CANCELED, (
            f"WO must transition to CANCELED on trade-WS sCode 0 "
            f"when flag is on; got status={wo.status}"
        )
        # working_bid cleared.
        with om._state._lock:
            assert om._state.working_bid is None, (
                "working_bid must be cleared after local terminal "
                "transition"
            )
        # Side unresolved cleared.
        assert not om._is_side_unresolved(Side.BUY), (
            "side_unresolved must be cleared on trade-WS sCode 0 terminal"
        )
    finally:
        path.unlink(missing_ok=True)


def test_trust_trade_ws_success_default_off_preserves_legacy() -> None:
    """When the flag is off (default), trade-WS sCode 0 does NOT
    transition the WO locally — the bot still waits for the inbound
    user-data WS event. Preserves legacy GRVT behavior where the
    ack-but-no-cancel quirk made early local transition unsafe."""
    om, path = _bootstrap()  # default settings: flag is False
    try:
        wo = _stuck_wo()
        with om._state._lock:
            om._state.working_bid = wo
        om._side_unresolved_active[Side.BUY] = True
        om._side_unresolved_reason[Side.BUY] = "cancel_pending_wait"
        om._side_unresolved_since_mono[Side.BUY] = time.monotonic()
        # Mock a successful cancel response.
        om._client.cancel_order.return_value = {"code": "0", "msg": "", "data": []}
        om._client.interpret_cancel_response.return_value = ("success", "")
        om._cancel_http_transport(wo)
        # WO STAYS in CANCEL_PENDING (legacy path).
        assert wo.status == OrderStatus.CANCEL_PENDING, (
            f"With flag default-off, WO must NOT transition locally on "
            f"trade-WS success; got status={wo.status}"
        )
        # working_bid still points to the WO.
        with om._state._lock:
            assert om._state.working_bid is wo, (
                "working_bid must NOT be cleared until WS event arrives"
            )
    finally:
        path.unlink(missing_ok=True)
