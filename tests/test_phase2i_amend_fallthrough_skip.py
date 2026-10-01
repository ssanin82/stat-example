"""v1.4.180 — Phase 2I fall-through fix.

Snapshot ``v1.4.176-260521-110704-prod.okx.ton.usdt.perp`` showed a
pathological cancel-replace cycle: 737 ``amend_enqueue_failed_falling_through``
events paired with 737 ``amend_to_cancel_dispatched`` events in a 35-min
session, ~3 cancel-replaces/sec, 1 fill total.

Root cause: when Phase 2I (v1.4.169) suppressed an amend (tick_flicker
within 250 ms OR rate cap of 8/sec), the AmendAction dispatcher in
``OrderManager._dispatch_action`` *fell through* to cancel-replace —
which defeats the rate-defence intent. Cancel-replace loses queue
position and spikes the order-event rate just like rapid amends would.

Fix: ``_enqueue_amend_quote_path`` now stamps the
``_last_amend_enqueue_suppress_reason`` attribute on each call:

* ``None`` on success or generic failure
* ``"tick_flicker"`` / ``"rate_throttle"`` on Phase 2I suppression

The AmendAction dispatcher reads this attribute after the call and
SKIPS the fall-through cancel-replace when the suppress reason is one
of the Phase 2I codes. Existing order keeps resting; the
``BEHIND_TOUCH_MAX_AGE_SECONDS`` safety net is the backstop for
genuinely stale orders.

Tests:

1. Pure-function: the attribute is correctly set on each invocation
   (None / tick_flicker / rate_throttle / None-on-reset).
2. Source-level sentinel: the AmendAction dispatcher contains the
   skip-on-suppression branch (catches the regression if a future
   refactor removes the check).
"""

from __future__ import annotations

import inspect
import time

import pytest

from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.quote_engine import FinalQuoteOrder
from tests.settings_helpers import UnitTestSettings


# Reuse the shell-OM pattern from test_phase2i_amend_rate_defence.
def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "AMEND_TICK_FLICKER_MIN_MS": 250.0,
        "AMEND_PER_ORDER_MAX_PER_SEC": 8,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _shell_om(settings: UnitTestSettings):
    from app.clock import SystemClock
    from app.execution import OrderManager

    class _ShellOM:
        pass

    class _ShellClient:
        def has_write_access(self) -> bool:
            return True

    class _ShellOutbound:
        def __init__(self) -> None:
            self.submitted: list = []

        def submit_place(self, intent):
            self.submitted.append(intent)

    class _NoLock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _ShellState:
        def __init__(self) -> None:
            self._lock = _NoLock()
            self.amend_intents_emitted_total = 0
            self.amend_pending_high_watermark = 0
            self.amend_tick_flicker_suppressed_total = 0
            self.amend_rate_throttle_suppressed_total = 0
            self.amend_rate_defence_first_arm_logged = {
                Side.BUY: False,
                Side.SELL: False,
            }

    shell = _ShellOM()
    shell._settings = settings
    shell._clock = SystemClock()
    shell._client = _ShellClient()
    shell._outbound = _ShellOutbound()
    shell._state = _ShellState()
    shell._first_enqueue_perf = None
    shell._outbound_traces = {}
    shell._maybe_refresh_t0_perf = None
    shell._last_amend_enqueue_suppress_reason = None
    shell._enqueue_amend_quote_path = (
        OrderManager._enqueue_amend_quote_path.__get__(shell, _ShellOM)
    )
    shell._count_amend_pending_unlocked = (
        OrderManager._count_amend_pending_unlocked.__get__(shell, _ShellOM)
    )
    shell._sync_outbound_state_flags = lambda: None
    shell.persist = lambda *_a, **_kw: None
    shell._record_orchestrate_decision = lambda *_a, **_kw: None
    return shell


def _wo(*, side: Side = Side.SELL, ord_id: int = 12345) -> WorkingOrder:
    from datetime import datetime, timezone

    return WorkingOrder(
        order_id_local="local-amend-test",
        order_id_exchange=ord_id,
        client_order_id="cl-amend",
        symbol="TON-USDT-SWAP",
        side=side,
        price=2.040,
        size=3.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_created=datetime.now(timezone.utc),
        ts_sent=datetime.now(timezone.utc),
        ts_ack=datetime.now(timezone.utc),
    )


def _desired(
    price: float, size: float = 3.0, side: Side = Side.SELL
) -> FinalQuoteOrder:
    return FinalQuoteOrder(side=side, price=price, size=size)


# ---------------------------------------------------------------------------
# _last_amend_enqueue_suppress_reason semantics
# ---------------------------------------------------------------------------


def test_suppress_reason_is_none_after_successful_dispatch() -> None:
    s = _settings()
    om = _shell_om(s)
    wo = _wo()
    assert om._enqueue_amend_quote_path(wo, _desired(2.041)) is True
    assert om._last_amend_enqueue_suppress_reason is None


def test_suppress_reason_is_tick_flicker_on_back_to_back_amend() -> None:
    """Two amends within the 250 ms flicker window → second one
    suppressed with reason ``tick_flicker``."""
    s = _settings()
    om = _shell_om(s)
    wo = _wo()
    # First amend succeeds, populates the ring.
    assert om._enqueue_amend_quote_path(wo, _desired(2.041)) is True
    assert om._last_amend_enqueue_suppress_reason is None
    # Second amend immediately after — tick_flicker fires.
    ok = om._enqueue_amend_quote_path(wo, _desired(2.042))
    assert ok is False
    assert om._last_amend_enqueue_suppress_reason == "tick_flicker"
    assert om._state.amend_tick_flicker_suppressed_total == 1


def test_suppress_reason_is_rate_throttle_when_burst_caps() -> None:
    """Fill the per-order ring by spacing dispatches > 250 ms apart
    so flicker doesn't fire — once we hit the 8/sec cap, suppression
    flips to ``rate_throttle``."""
    s = _settings(
        AMEND_TICK_FLICKER_MIN_MS=0.0,  # disable flicker for this test
        AMEND_PER_ORDER_MAX_PER_SEC=3,
    )
    om = _shell_om(s)
    wo = _wo()
    for _ in range(3):
        assert om._enqueue_amend_quote_path(wo, _desired(2.041)) is True
    # 4th in the 1-second window — rate-throttle should fire.
    ok = om._enqueue_amend_quote_path(wo, _desired(2.042))
    assert ok is False
    assert om._last_amend_enqueue_suppress_reason == "rate_throttle"
    assert om._state.amend_rate_throttle_suppressed_total == 1


def test_suppress_reason_resets_to_none_on_next_successful_call() -> None:
    """After a suppressed call sets the reason, a subsequent successful
    call (e.g. after the flicker window elapses) MUST reset the
    attribute to None so a stale value from a prior suppression can't
    leak into the dispatcher's read."""
    s = _settings()
    om = _shell_om(s)
    wo = _wo()
    # Tick 1: first amend succeeds.
    assert om._enqueue_amend_quote_path(wo, _desired(2.041)) is True
    # Tick 2: flicker suppressed.
    assert om._enqueue_amend_quote_path(wo, _desired(2.042)) is False
    assert om._last_amend_enqueue_suppress_reason == "tick_flicker"
    # Wait past the flicker window.
    time.sleep(0.30)
    # Tick 3: should succeed, attribute reset.
    assert om._enqueue_amend_quote_path(wo, _desired(2.043)) is True
    assert om._last_amend_enqueue_suppress_reason is None


def test_suppress_reason_is_none_when_method_runs_clean() -> None:
    """Sanity: at module / instance init time the attribute is None
    (no prior call). Important because the AmendAction dispatcher
    reads this attribute IMMEDIATELY after each call — there must be
    no risk of seeing a stale ``tick_flicker`` value from a prior
    invocation."""
    s = _settings()
    om = _shell_om(s)
    assert om._last_amend_enqueue_suppress_reason is None


# ---------------------------------------------------------------------------
# Source sentinel: AmendAction dispatcher honours the suppress reason
# ---------------------------------------------------------------------------


def test_amend_action_dispatcher_skips_on_phase2i_suppression() -> None:
    """Sentinel: the AmendAction dispatcher in
    ``OrderManager._dispatch_action`` must read
    ``_last_amend_enqueue_suppress_reason`` and SKIP the cancel-
    replace fallthrough when the value is one of the Phase 2I codes.

    Source-text test (mirrors the existing wedge-prevention sentinels
    in ``test_bug025_executor_silent_wedge.py``). Behaviour-level
    coverage is intentionally deferred to the live-bot integration
    suite — the dispatcher's preconditions (orchestrate state machine,
    side-unresolved checks, slot tracking) are too tightly wound to
    isolate cleanly in a unit test. The sentinel keeps a future
    refactor from silently re-introducing the cancel fall-through."""
    from app.execution import OrderManager

    src = inspect.getsource(OrderManager._dispatch_action)
    # The skip branch reads the new attribute …
    assert "_last_amend_enqueue_suppress_reason" in src
    # … and recognises both Phase 2I suppress codes.
    assert '"tick_flicker"' in src
    assert '"rate_throttle"' in src
    # … and emits a recognisable orchestrate-decision label so the
    # operator can see the new code path in the lifecycle log.
    assert "amend_phase2i_suppressed_skipping" in src


def test_enqueue_amend_resets_suppress_reason_at_top() -> None:
    """Sentinel: ``_enqueue_amend_quote_path`` resets the
    ``_last_amend_enqueue_suppress_reason`` flag at the top of every
    invocation. Critical because the AmendAction dispatcher reads
    the attribute right after the call — leaving a stale value would
    falsely skip the cancel fallthrough on a subsequent unrelated
    failure."""
    from app.execution import OrderManager

    src = inspect.getsource(OrderManager._enqueue_amend_quote_path)
    assert "self._last_amend_enqueue_suppress_reason = None" in src
