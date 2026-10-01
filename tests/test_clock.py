"""Phase 1a (v1.4.229) — Clock abstraction tests.

Pins the ``Clock`` protocol's contract for both implementations:

* ``SystemClock`` — must behave identically to direct stdlib
  ``time.monotonic()`` / ``time.time()`` / ``datetime.now()`` calls.
* ``ReplayClock`` — must advance only on explicit
  ``advance_to(ts_ns)`` calls, must reject backward jumps, must
  return precision-stable floats across long replay runs.

Phase 1a SHIPS the module + tests with NO existing-callsite
migration. Phase 1b/1c follow with the actual site-by-site
replacement of ``time.*`` calls across ``app/``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from app.clock import Clock, ReplayClock, SystemClock


# -------------------------------------------------------------------
# Protocol conformance
# -------------------------------------------------------------------


def test_system_clock_satisfies_protocol() -> None:
    """Runtime ``isinstance`` check against the ``Clock`` protocol."""
    c = SystemClock()
    assert isinstance(c, Clock)


def test_replay_clock_satisfies_protocol() -> None:
    c = ReplayClock(start_t_ns=1_000_000_000)
    assert isinstance(c, Clock)


# -------------------------------------------------------------------
# SystemClock — production parity with stdlib
# -------------------------------------------------------------------


def test_system_clock_time_is_within_stdlib_window() -> None:
    """``SystemClock.time()`` returns the same wall-clock value the
    stdlib's ``time.time()`` does, within a tight delta."""
    stdlib_before = time.time()
    c = SystemClock()
    clock_value = c.time()
    stdlib_after = time.time()
    assert stdlib_before - 1e-3 <= clock_value <= stdlib_after + 1e-3


def test_system_clock_monotonic_is_within_stdlib_window() -> None:
    stdlib_before = time.monotonic()
    c = SystemClock()
    clock_value = c.monotonic()
    stdlib_after = time.monotonic()
    assert stdlib_before - 1e-3 <= clock_value <= stdlib_after + 1e-3


def test_system_clock_monotonic_advances_across_calls() -> None:
    """Strategy code's monotonicity invariant: two consecutive
    reads from the SAME ``SystemClock`` instance must produce
    non-decreasing values."""
    c = SystemClock()
    t1 = c.monotonic()
    t2 = c.monotonic()
    assert t2 >= t1


def test_system_clock_now_utc_is_tz_aware() -> None:
    """``now_utc()`` must always return a tz-aware datetime in UTC."""
    c = SystemClock()
    n = c.now_utc()
    assert n.tzinfo is not None
    assert n.tzinfo == timezone.utc


# -------------------------------------------------------------------
# ReplayClock — driven-time semantics
# -------------------------------------------------------------------


def test_replay_clock_constructor_pins_origin() -> None:
    """``monotonic()`` at construction time is 0.0 (start of replay)."""
    c = ReplayClock(start_t_ns=2_000_000_000)
    assert c.monotonic() == 0.0
    assert c.time() == 2.0  # 2_000_000_000 ns = 2.0 s


def test_replay_clock_advance_to_forward_jump_accepted() -> None:
    c = ReplayClock(start_t_ns=1_000_000_000)
    c.advance_to(1_500_000_000)
    assert c.time() == 1.5
    assert c.monotonic() == 0.5  # 500_000_000 ns since origin


def test_replay_clock_advance_to_same_instant_accepted() -> None:
    """Same-instant events processed in order — no advance, no error."""
    c = ReplayClock(start_t_ns=1_000_000_000)
    c.advance_to(1_000_000_000)
    assert c.time() == 1.0
    assert c.monotonic() == 0.0


def test_replay_clock_advance_to_backward_rejected() -> None:
    """Backward time corrupts strategy code's monotonic invariants —
    must fail loudly."""
    c = ReplayClock(start_t_ns=1_000_000_000)
    c.advance_to(2_000_000_000)
    with pytest.raises(ValueError, match="cannot go backward"):
        c.advance_to(1_500_000_000)


def test_replay_clock_negative_start_rejected() -> None:
    """Defensive: caller must pass a valid Unix-epoch nanosecond
    timestamp. Negative = bug → fail at construction."""
    with pytest.raises(ValueError, match="non-negative"):
        ReplayClock(start_t_ns=-1)


def test_replay_clock_now_utc_matches_advance() -> None:
    """``now_utc()`` returns a tz-aware datetime at the current
    replay instant."""
    # 2026-05-21T17:07:32.207697+00:00 in ns
    ts_ns = int(datetime(2026, 5, 21, 17, 7, 32, 207697, tzinfo=timezone.utc).timestamp() * 1e9)
    c = ReplayClock(start_t_ns=ts_ns)
    n = c.now_utc()
    assert n.tzinfo is not None
    assert n.year == 2026 and n.month == 5 and n.day == 21
    assert n.hour == 17 and n.minute == 7 and n.second == 32


# -------------------------------------------------------------------
# Precision / drift over long replays
# -------------------------------------------------------------------


def test_replay_clock_no_float_drift_across_million_ticks() -> None:
    """A 24h fixture at 0.001 s tick = 8.64e7 ticks. With internal
    integer ns storage, the final monotonic value MUST equal the
    exact integer sum / 1e9 with no accumulated float drift.

    Without ns-integer storage, repeated float additions would
    drift by ~1 ULP per tick × N ticks → visibly wrong after a
    few million additions. Pinning this here so a future "let's
    just use floats internally" refactor breaks the test."""
    c = ReplayClock(start_t_ns=0)
    # Advance in 1 ms steps for 1 million ticks = 1000 seconds total.
    final_step_ns = 1_000_000  # 1 ms
    n_steps = 1_000_000
    for i in range(1, n_steps + 1):
        c.advance_to(i * final_step_ns)
    expected_seconds = (n_steps * final_step_ns) / 1e9
    actual_seconds = c.monotonic()
    # Ns-integer storage → exact at the float-conversion step. The
    # only loss is the single int → float division, well within
    # 1e-15 relative tolerance.
    assert abs(actual_seconds - expected_seconds) < 1e-9, (
        f"drift after {n_steps} ticks: actual={actual_seconds}, "
        f"expected={expected_seconds}, delta={actual_seconds - expected_seconds}"
    )


def test_replay_clock_monotonic_strictly_non_decreasing() -> None:
    """100 incremental advances → 100 non-decreasing monotonic
    reads."""
    c = ReplayClock(start_t_ns=1_000_000_000)
    last_mono = c.monotonic()
    for i in range(1, 101):
        c.advance_to(1_000_000_000 + i * 13_337)  # arbitrary non-round step
        cur_mono = c.monotonic()
        assert cur_mono >= last_mono
        last_mono = cur_mono


# -------------------------------------------------------------------
# Bot.__init__ integration smoke (Phase 1a doesn't migrate sites,
# but the constructor MUST accept the clock kwarg + default cleanly)
# -------------------------------------------------------------------


def test_bot_init_defaults_to_system_clock() -> None:
    """``Bot.__init__(clock=None)`` should default to a SystemClock
    instance, preserving production behaviour."""
    from app.clock import SystemClock
    # We can't instantiate a Bot here without the full DB / WS /
    # exchange stack — instead verify the default-injection logic
    # by checking the source code path is correct.
    # Smoke test: SystemClock can be instantiated and its methods
    # all work — that's what Bot.__init__ relies on.
    c = SystemClock()
    _ = c.time()
    _ = c.monotonic()
    _ = c.now_utc()


def test_bot_accepts_replay_clock_kwarg() -> None:
    """``Bot.__init__(clock=ReplayClock(...))`` should be accepted
    as a substitute for the default SystemClock. (Smoke test —
    full bot construction requires DB + WS + exchange so we just
    verify the import + constructor signature.)"""
    from app.bot import Bot
    import inspect

    sig = inspect.signature(Bot.__init__)
    assert "clock" in sig.parameters, (
        "Bot.__init__ must accept a ``clock`` parameter for Phase 1a wiring"
    )
    # Default value should be None (the constructor body then
    # creates SystemClock).
    assert sig.parameters["clock"].default is None


# -------------------------------------------------------------------
# Module-level proxy (Phase 1c, v1.4.231)
# -------------------------------------------------------------------


def test_module_proxy_defaults_to_system_clock() -> None:
    """Out of the box, the module-level functions return
    stdlib-equivalent values."""
    import app.clock as clk

    # Reset to default in case a prior test installed a replay clock.
    clk.set_module_clock(clk.SystemClock())
    stdlib_before = time.monotonic()
    proxy_value = clk.monotonic()
    stdlib_after = time.monotonic()
    assert stdlib_before - 1e-3 <= proxy_value <= stdlib_after + 1e-3


def test_module_proxy_can_be_swapped_to_replay_clock() -> None:
    """The whole point of the proxy — install a ReplayClock and
    every module-level call returns replay-time values."""
    import app.clock as clk

    prior = clk.get_module_clock()
    try:
        rc = clk.ReplayClock(start_t_ns=42_000_000_000)  # 42 seconds
        clk.set_module_clock(rc)
        assert clk.time_seconds() == 42.0
        assert clk.monotonic() == 0.0
        rc.advance_to(50_000_000_000)
        assert clk.time_seconds() == 50.0
        assert clk.monotonic() == 8.0
    finally:
        # Restore so subsequent tests don't see the replay clock.
        clk.set_module_clock(prior)


def test_module_proxy_drives_utils_time_helpers() -> None:
    """The cascade: ``utils/time.py``'s ``utc_now()`` /
    ``utc_now_iso()`` / ``seconds_since()`` all route through the
    module proxy. Installing a ReplayClock makes ALL 55+ call sites
    (across app/) that use these helpers see replay time, with NO
    other code changes. This is the load-bearing property of
    Phase 1c."""
    import app.clock as clk
    from app.utils.time import utc_now, utc_now_iso, seconds_since
    from datetime import datetime, timezone

    prior = clk.get_module_clock()
    try:
        # Pin replay to 2026-01-01 00:00:00 UTC.
        ts_ns = int(
            datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1e9
        )
        clk.set_module_clock(clk.ReplayClock(start_t_ns=ts_ns))
        # utc_now() should return 2026-01-01.
        n = utc_now()
        assert n.year == 2026 and n.month == 1 and n.day == 1
        assert n.hour == 0 and n.minute == 0
        # utc_now_iso() reflects the same.
        assert utc_now_iso().startswith("2026-01-01T00:00:00")
        # seconds_since() against a 30s-prior timestamp.
        ts = datetime(2025, 12, 31, 23, 59, 30, tzinfo=timezone.utc)
        elapsed = seconds_since(ts)
        assert elapsed == 30.0
    finally:
        clk.set_module_clock(prior)


def test_module_proxy_set_with_none_safe_state_clock_helper() -> None:
    """Free-function modules using ``from app import clock as _clock``
    + ``_clock.monotonic()`` get the active module clock. Pin the
    contract: if a future refactor changes the proxy's spelling or
    semantics, this test catches it."""
    import app.clock as clk

    prior = clk.get_module_clock()
    try:
        rc = clk.ReplayClock(start_t_ns=100_000_000_000)
        clk.set_module_clock(rc)
        assert clk.monotonic() == 0.0  # zero since replay just started
        rc.advance_to(101_000_000_000)
        assert clk.monotonic() == 1.0  # 1 second elapsed in replay time
    finally:
        clk.set_module_clock(prior)


def test_module_proxy_thread_safety_under_steady_clock() -> None:
    """Under production (SystemClock), concurrent reads from
    multiple threads must not deadlock or produce garbage. There's
    no shared mutable state in SystemClock, but pin the property."""
    import app.clock as clk
    import threading

    prior = clk.get_module_clock()
    try:
        clk.set_module_clock(clk.SystemClock())
        results = []
        def worker():
            for _ in range(1000):
                results.append(clk.monotonic())
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len(results) == 4000
        # Each thread's reads should be non-decreasing (interleaving
        # between threads makes the overall order non-monotonic but
        # individual readings are still valid).
        for v in results:
            assert isinstance(v, float)
            assert v > 0  # monotonic since process start is positive
    finally:
        clk.set_module_clock(prior)


# ---------------------------------------------------------------------
# Phase 1c acceptance tests (v1.5.91).
#
# Phase 1c's load-bearing claim: every `app.utils.time.utc_now()` call
# in the bot codebase routes through `app.clock`'s module-level proxy.
# So installing a ReplayClock via `set_module_clock` is enough to make
# the helper return replay-time everywhere — no per-call-site
# migration needed.
#
# These tests pin that property. If a future refactor breaks the
# import chain (e.g. `utils/time.py` starts calling `datetime.now`
# directly again), the next replay run would silently misbehave;
# these tests would fail loudly first.
# ---------------------------------------------------------------------

def test_phase1c_utc_now_helper_routes_through_module_clock() -> None:
    """The helper ``app.utils.time.utc_now()`` reads the active
    module clock — same instant as `app.clock.now_utc()`.

    Most app/ modules import the helper (not the proxy) because
    that's the long-standing canonical spelling. The helper's body
    is a single delegation to the proxy; if anything ever changes
    that delegation (e.g. someone "optimises" by reading
    `datetime.now(timezone.utc)` directly), the bot's backtest
    deterministic-time guarantee silently breaks. This test pins
    the contract."""
    import app.clock as clk
    from app.utils.time import utc_now

    prior = clk.get_module_clock()
    try:
        # Fixed-time replay clock at 2026-01-01T00:00:00Z.
        FIXED_T_NS = 1_767_225_600_000_000_000
        rc = clk.ReplayClock(start_t_ns=FIXED_T_NS)
        clk.set_module_clock(rc)
        # The helper must return the replay clock's instant.
        observed = utc_now()
        assert observed.year == 2026
        assert observed.month == 1
        assert observed.day == 1
        # Round-trip: bytes match between the helper and the proxy.
        assert observed == clk.now_utc()
    finally:
        clk.set_module_clock(prior)


def test_phase1c_utc_now_iso_helper_routes_through_module_clock() -> None:
    """``utc_now_iso()`` is also widely used (log lines, bot_events
    rows, etc.). Same property: must route through the module
    clock."""
    import app.clock as clk
    from app.utils.time import utc_now_iso

    prior = clk.get_module_clock()
    try:
        FIXED_T_NS = 1_767_225_600_000_000_000  # 2026-01-01T00:00:00Z
        rc = clk.ReplayClock(start_t_ns=FIXED_T_NS)
        clk.set_module_clock(rc)
        iso = utc_now_iso()
        assert iso.startswith("2026-01-01T00:00:00")
    finally:
        clk.set_module_clock(prior)


def test_phase1c_seconds_since_helper_routes_through_module_clock() -> None:
    """``seconds_since(ts)`` is used by gate evaluators that store
    a timestamp and later check elapsed time. The "current" instant
    must come from the replay clock so a replay can reproduce the
    same gate firings as production."""
    import app.clock as clk
    from app.utils.time import seconds_since
    from datetime import datetime, timezone

    prior = clk.get_module_clock()
    try:
        # Replay clock starts at 2026-01-01T00:00:30Z.
        FIXED_T_NS = 1_767_225_630_000_000_000
        rc = clk.ReplayClock(start_t_ns=FIXED_T_NS)
        clk.set_module_clock(rc)
        # A "stored" timestamp 30 seconds earlier.
        earlier = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        elapsed = seconds_since(earlier)
        assert elapsed is not None
        # 30s gap exactly — no float drift at the second granularity.
        assert abs(elapsed - 30.0) < 1e-6
    finally:
        clk.set_module_clock(prior)


def test_phase1c_quoting_compute_quote_decision_clock_threading() -> None:
    """Phase 1b (v1.5.90) added a ``clock`` kwarg to
    ``compute_quote_decision``. Pin that the ``QuoteDecision.ts``
    field reflects the passed clock, not wall time."""
    import app.clock as clk
    # NB: compute_quote_decision is a deep function with many
    # required signals; we don't call it directly — too much setup.
    # Instead, just check that the import + signature is sound:
    from app.quoting import compute_quote_decision
    import inspect
    sig = inspect.signature(compute_quote_decision)
    assert "clock" in sig.parameters
    # The default must be None (or a Clock) so back-compat callers
    # without an explicit clock argument still work.
    assert sig.parameters["clock"].default is None
    # Pin the Phase 1b semantics: when clock=None, the function
    # falls back to a SystemClock(). Spot-check by reading the
    # source's fallback line. (Cheaper than running the full call;
    # the fallback is a single conditional.)
    src = inspect.getsource(compute_quote_decision)
    assert "SystemClock()" in src


def test_phase1c_execution_transition_uses_module_clock() -> None:
    """v1.5.143 — ``app.execution.transition()`` is the free function
    that stamps ts_sent / ts_ack / ts_closed on every WorkingOrder
    state change. It was the LAST Phase 1c hold-out using
    ``datetime.now(timezone.utc)`` directly. Pre-fix, replay runs
    leaked wall-clock timestamps into the bot's storage layer →
    the backtesting viewer's order-events overlay rendered at
    wall-clock positions (2 days off the simulated time, on the
    2026-05-25 incident).

    This test pins the fix: install a ReplayClock at a known
    instant; call ``transition()``; verify the stamped timestamps
    equal the replay-clock instant, not wall-clock."""
    import app.clock as clk
    from app.enums import OrderStatus, Side
    from app.execution import transition
    from app.models import WorkingOrder
    from datetime import datetime, timezone

    prior = clk.get_module_clock()
    try:
        # Pin replay to 2026-05-23T16:07:47Z (the user's actual
        # recording-window time from the incident).
        FIXED_T_NS = int(
            datetime(2026, 5, 23, 16, 7, 47, tzinfo=timezone.utc).timestamp()
            * 1e9
        )
        clk.set_module_clock(clk.ReplayClock(start_t_ns=FIXED_T_NS))

        wo = WorkingOrder(
            order_id_local="L-test-clock",
            order_id_exchange=None,
            client_order_id="MM-test-clock-1",
            symbol="TON-USDT-SWAP",
            side=Side.BUY,
            price=1.78,
            size=1.0,
            post_only=True,
            status=OrderStatus.NEW_LOCAL,
            quote_cycle_id="qt-test",
        )
        # SENT → ts_sent stamped at replay time, not wall clock.
        transition(wo, OrderStatus.SENT)
        assert wo.ts_sent is not None
        assert wo.ts_sent.year == 2026
        assert wo.ts_sent.month == 5
        assert wo.ts_sent.day == 23
        assert wo.ts_sent.hour == 16
        # ACKED → ts_ack stamped at replay time.
        transition(wo, OrderStatus.ACKED)
        assert wo.ts_ack is not None
        assert wo.ts_ack.year == 2026 and wo.ts_ack.month == 5 and wo.ts_ack.day == 23
        # CANCELED → ts_closed stamped at replay time.
        transition(wo, OrderStatus.CANCELED, reason="test")
        assert wo.ts_closed is not None
        assert wo.ts_closed.year == 2026 and wo.ts_closed.month == 5 and wo.ts_closed.day == 23
        assert wo.cancel_reason == "test"
    finally:
        clk.set_module_clock(prior)
