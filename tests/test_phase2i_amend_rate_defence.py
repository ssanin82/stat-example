"""Phase 2I (v1.4.169) — per-order amend rate-defence guard.

STRUCTURAL guard at the top of
``OrderManager._enqueue_amend_quote_path`` against the snapshot
``v1.4.102-260520-120555`` pattern: a single SELL order received
244 amend dispatches in ~1 second as the bot's computed price
ping-ponged between two adjacent ticks (1.941 ↔ 1.942) every 5-7 ms
quote cycle. The per-account aggregate amend-rate counter showed
plenty of headroom (12/2s vs cap), so the breach was invisible to
existing pacing.

Two complementary guards:
  1. ``AMEND_TICK_FLICKER_MIN_MS`` (default 250) — minimum gap between
     consecutive amend dispatches on the SAME order. Catches the
     adjacent-tick ping-pong at the source.
  2. ``AMEND_PER_ORDER_MAX_PER_SEC`` (default 8) — cap on dispatches
     per order per 1-s window. Hard ceiling that bounds any pattern
     the flicker-gap doesn't catch.

Always-on (no enabled flag — these are structural). Suppressed
dispatches bump per-state counters + emit a first-arm WARNING log.

Tests use a shell-mock ``OrderManager`` like the existing
``test_post_only_cross_reject_cooldown.py`` pattern — binding only
the methods + state needed for the guard. Avoids spinning up the
full execution pipeline.
"""

from __future__ import annotations

import time
from typing import Optional

import pytest

from app.enums import OrderStatus, Side
from app.models import WorkingOrder
from app.quote_engine import FinalQuoteOrder
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "AMEND_TICK_FLICKER_MIN_MS": 250.0,
        "AMEND_PER_ORDER_MAX_PER_SEC": 8,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _shell_om(settings: UnitTestSettings):
    """Build a shell ``OrderManager`` instance with only the
    attributes the guard reads. Mirrors the pattern in
    ``test_post_only_cross_reject_cooldown.py``."""
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

    class _NoLock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    shell = _ShellOM()
    shell._settings = settings
    shell._clock = SystemClock()
    shell._client = _ShellClient()
    shell._outbound = _ShellOutbound()
    shell._state = _ShellState()
    shell._first_enqueue_perf = None
    shell._outbound_traces = {}
    shell._maybe_refresh_t0_perf = None
    # Bind the methods we need from OrderManager.
    shell._enqueue_amend_quote_path = (
        OrderManager._enqueue_amend_quote_path.__get__(shell, _ShellOM)
    )
    shell._count_amend_pending_unlocked = (
        OrderManager._count_amend_pending_unlocked.__get__(shell, _ShellOM)
    )
    shell._sync_outbound_state_flags = lambda: None
    shell.persist = lambda *_args, **_kw: None
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


def _desired(price: float, size: float = 3.0, side: Side = Side.SELL) -> FinalQuoteOrder:
    return FinalQuoteOrder(side=side, price=price, size=size)


# ---------------------------------------------------------------------------
# Default behaviour: first dispatch always allowed
# ---------------------------------------------------------------------------


def test_first_dispatch_succeeds() -> None:
    s = _settings()
    om = _shell_om(s)
    wo = _wo()
    assert om._enqueue_amend_quote_path(wo, _desired(2.041)) is True
    assert len(om._outbound.submitted) == 1
    assert om._state.amend_tick_flicker_suppressed_total == 0
    assert om._state.amend_rate_throttle_suppressed_total == 0


def test_ring_buffer_records_dispatch() -> None:
    s = _settings()
    om = _shell_om(s)
    wo = _wo()
    om._enqueue_amend_quote_path(wo, _desired(2.041))
    assert len(wo.amend_recent_dispatch_mono) == 1
    # The recorded value should be a recent monotonic time.
    now = time.monotonic()
    assert abs(wo.amend_recent_dispatch_mono[0] - now) < 1.0


# ---------------------------------------------------------------------------
# Tick-flicker suppression: too-close consecutive dispatches
# ---------------------------------------------------------------------------


def test_tick_flicker_suppression_blocks_too_close_dispatch() -> None:
    """Default flicker min = 250 ms. Two dispatches 50 ms apart should
    suppress the second."""
    s = _settings(AMEND_TICK_FLICKER_MIN_MS=250.0)
    om = _shell_om(s)
    wo = _wo()
    assert om._enqueue_amend_quote_path(wo, _desired(2.041)) is True
    # Sleep ~50 ms then attempt a second amend.
    time.sleep(0.05)
    assert om._enqueue_amend_quote_path(wo, _desired(2.042)) is False
    assert om._state.amend_tick_flicker_suppressed_total == 1
    assert om._state.amend_rate_throttle_suppressed_total == 0
    # Suppressed dispatch must NOT have called submit.
    assert len(om._outbound.submitted) == 1


def test_tick_flicker_threshold_is_inclusive_zero() -> None:
    """``AMEND_TICK_FLICKER_MIN_MS=0`` disables the flicker check."""
    s = _settings(AMEND_TICK_FLICKER_MIN_MS=0.0, AMEND_PER_ORDER_MAX_PER_SEC=200)
    om = _shell_om(s)
    wo = _wo()
    # Burst 5 amends immediately. None should be suppressed by flicker.
    for px in (2.041, 2.042, 2.043, 2.044, 2.045):
        assert om._enqueue_amend_quote_path(wo, _desired(px)) is True
    assert om._state.amend_tick_flicker_suppressed_total == 0


def test_dispatch_allowed_after_flicker_window_elapses() -> None:
    """After waiting more than ``MIN_MS``, the next dispatch goes."""
    s = _settings(AMEND_TICK_FLICKER_MIN_MS=50.0)
    om = _shell_om(s)
    wo = _wo()
    om._enqueue_amend_quote_path(wo, _desired(2.041))
    time.sleep(0.07)
    assert om._enqueue_amend_quote_path(wo, _desired(2.042)) is True
    assert len(om._outbound.submitted) == 2
    assert om._state.amend_tick_flicker_suppressed_total == 0


# ---------------------------------------------------------------------------
# Per-order rate throttle: cap on dispatches per 1-s window
# ---------------------------------------------------------------------------


def test_rate_throttle_caps_dispatches_per_second() -> None:
    """With rate cap = 4 and flicker disabled, 6 immediate amends
    on the same order: first 4 succeed, last 2 throttled."""
    s = _settings(
        AMEND_TICK_FLICKER_MIN_MS=0.0,
        AMEND_PER_ORDER_MAX_PER_SEC=4,
    )
    om = _shell_om(s)
    wo = _wo()
    results = []
    for px in (2.041, 2.042, 2.043, 2.044, 2.045, 2.046):
        results.append(om._enqueue_amend_quote_path(wo, _desired(px)))
    assert results == [True, True, True, True, False, False]
    assert om._state.amend_rate_throttle_suppressed_total == 2
    assert om._state.amend_tick_flicker_suppressed_total == 0
    assert len(om._outbound.submitted) == 4


def test_rate_throttle_recovers_after_1s_window() -> None:
    """After the 1-s window elapses the rate counter resets via
    pruning, so amends can resume."""
    s = _settings(
        AMEND_TICK_FLICKER_MIN_MS=0.0,
        AMEND_PER_ORDER_MAX_PER_SEC=2,
    )
    om = _shell_om(s)
    wo = _wo()
    om._enqueue_amend_quote_path(wo, _desired(2.041))
    om._enqueue_amend_quote_path(wo, _desired(2.042))
    # 3rd dispatch immediately → throttled (cap is 2).
    assert om._enqueue_amend_quote_path(wo, _desired(2.043)) is False
    # Sleep > 1 s — old entries prune, new dispatch goes.
    time.sleep(1.05)
    assert om._enqueue_amend_quote_path(wo, _desired(2.044)) is True


# ---------------------------------------------------------------------------
# Per-order isolation — guard is PER ORDER, not per side
# ---------------------------------------------------------------------------


def test_guard_does_not_cross_orders() -> None:
    """A burst on order A must not interfere with a separate amend
    on order B (same side or otherwise)."""
    s = _settings(
        AMEND_TICK_FLICKER_MIN_MS=250.0,
        AMEND_PER_ORDER_MAX_PER_SEC=8,
    )
    om = _shell_om(s)
    wo_a = _wo(ord_id=111)
    wo_b = _wo(ord_id=222)
    # Burst on A.
    om._enqueue_amend_quote_path(wo_a, _desired(2.041))
    assert om._enqueue_amend_quote_path(wo_a, _desired(2.042)) is False
    # Order B's first dispatch is unaffected.
    assert om._enqueue_amend_quote_path(wo_b, _desired(2.041)) is True
    assert om._state.amend_tick_flicker_suppressed_total == 1


# ---------------------------------------------------------------------------
# v1.4.102 reproduction: 244 amends in ~1 sec
# ---------------------------------------------------------------------------


def test_v1_4_102_replay_runaway_loop_caps_at_rate() -> None:
    """Replay of the snapshot v1.4.102-260520-120555 pattern: the
    bot's tick generates an amend every ~5 ms. With default
    ``AMEND_TICK_FLICKER_MIN_MS=250``, only every ~250 ms amend
    gets through.

    Bound is GENEROUS to absorb Windows sleep granularity. Python's
    ``time.sleep(0.005)`` on Windows is bounded by the system timer
    resolution (~15.6 ms by default), so 100 iterations can take
    anywhere from 500 ms (ideal) to ~2 s (Windows CI under load).
    At 250 ms throttle gap that's up to ~8 dispatches.

    The TEST'S PURPOSE is to prove the throttle stops the
    v1.4.102-style runaway of **244 amends in 1 s** — any bound
    well below that proves the throttle works. ≤ 12 dispatches
    represents a > 20× reduction; ≤ 4 was the original idealised-
    no-jitter bound and CI showed it's too tight (5 observed,
    2026-05-21).
    """
    s = _settings()
    om = _shell_om(s)
    wo = _wo()
    n_ok = 0
    n_suppressed = 0
    for px_int in range(100):
        px = 2.041 if (px_int % 2 == 0) else 2.042
        ok = om._enqueue_amend_quote_path(wo, _desired(px))
        if ok:
            n_ok += 1
        else:
            n_suppressed += 1
        time.sleep(0.005)
    # Bound covers a 2 s wall-clock window @ 250 ms gap = max 8
    # dispatches + 4-spot jitter slack = 12.
    assert n_ok <= 12, (
        f"throttle let through {n_ok} dispatches in 100 iterations — "
        f"either the throttle is broken or sleep jitter is "
        f"pathologically long on this host"
    )
    # n_suppressed = 100 - n_ok; if n_ok climbs to 12, suppressed
    # drops to 88. Lower bound of 88 still proves the throttle is
    # doing the dominant work.
    assert n_suppressed >= 88
    # The hard rate cap (8 in default) is never the binding constraint
    # here because flicker already throttles us; expect ZERO rate-cap
    # suppressions.
    assert om._state.amend_rate_throttle_suppressed_total == 0
    assert (
        om._state.amend_tick_flicker_suppressed_total == n_suppressed
    )


# ---------------------------------------------------------------------------
# Side first-arm log fires once per session per side
# ---------------------------------------------------------------------------


def test_first_arm_log_fires_once_per_side(caplog) -> None:
    s = _settings(AMEND_TICK_FLICKER_MIN_MS=250.0)
    om = _shell_om(s)
    wo = _wo(side=Side.SELL)
    caplog.set_level("WARNING", logger="app.execution")
    om._enqueue_amend_quote_path(wo, _desired(2.041, side=Side.SELL))
    time.sleep(0.05)
    om._enqueue_amend_quote_path(wo, _desired(2.042, side=Side.SELL))
    om._enqueue_amend_quote_path(wo, _desired(2.043, side=Side.SELL))
    om._enqueue_amend_quote_path(wo, _desired(2.044, side=Side.SELL))
    armed_msgs = [
        r for r in caplog.records if "amend_rate_defence_armed" in r.message
    ]
    assert len(armed_msgs) == 1
    # BUY side first-arm should still be untouched.
    assert om._state.amend_rate_defence_first_arm_logged[Side.BUY] is False
    assert om._state.amend_rate_defence_first_arm_logged[Side.SELL] is True


# ---------------------------------------------------------------------------
# WorkingOrder default — fresh WO starts with an empty buffer
# ---------------------------------------------------------------------------


def test_working_order_default_amend_buffer_empty() -> None:
    wo = _wo()
    assert wo.amend_recent_dispatch_mono == []
    assert isinstance(wo.amend_recent_dispatch_mono, list)
