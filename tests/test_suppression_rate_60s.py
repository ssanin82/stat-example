"""Phase 5 — 60 s rolling suppression rate.

Used by the dashboard's "healthy but not trading" indicator chip
to distinguish "calm market, nothing to suppress" from "engine
deciding but every decision suppressed". The latter is the
silent-deadlock signature (snapshot 260507114312 spent 4.5 min
in that state with no operator-visible signal before the watchdog
killed the bot).

Pinned invariants:

1. ``None`` when no quote cycles are recorded yet.
2. 0.0 when cycles fired but no suppressors did.
3. >0.0 when at least one cycle was suppressed.
4. Exactly 1.0 when every cycle in the window was suppressed.
5. Multiple suppressions in a single cycle still count as one
   suppressed cycle (no double-counting).
6. Cycles older than 60 s are dropped from the window on read.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from app.quote_quality_telemetry import QuoteQualityRollup


def _rollup() -> QuoteQualityRollup:
    return QuoteQualityRollup(window_samples=400)


def _record_cycle(qq: QuoteQualityRollup, suppressed: bool) -> None:
    if suppressed:
        qq.record_suppression("test_reason")
    qq.record_quote_cycle(
        quoted_spread_bps=2.0,
        one_tick_wide=False,
        two_sided_effective=True,
        intended_two_sided=True,
    )


def test_returns_none_when_no_cycles() -> None:
    qq = _rollup()
    assert qq.suppression_rate_60s() is None


def test_zero_when_no_suppressions() -> None:
    qq = _rollup()
    for _ in range(10):
        _record_cycle(qq, suppressed=False)
    assert qq.suppression_rate_60s() == 0.0


def test_one_when_all_suppressed() -> None:
    qq = _rollup()
    for _ in range(10):
        _record_cycle(qq, suppressed=True)
    assert qq.suppression_rate_60s() == 1.0


def test_partial_rate() -> None:
    """Half suppressed, half not → 0.5."""
    qq = _rollup()
    for i in range(10):
        _record_cycle(qq, suppressed=(i % 2 == 0))
    rate = qq.suppression_rate_60s()
    assert rate is not None
    assert 0.49 <= rate <= 0.51


def test_multiple_suppressions_in_one_cycle_count_once() -> None:
    """A cycle with 5 suppression reasons fires still counts as ONE
    suppressed cycle in the rate. Prevents the rate from exceeding 1.0
    when multiple gates fire simultaneously (common: freshness +
    inventory bias + soft skew can all hit the same tick)."""
    qq = _rollup()
    qq.record_suppression("reason_a")
    qq.record_suppression("reason_b")
    qq.record_suppression("reason_c")
    qq.record_quote_cycle(
        quoted_spread_bps=2.0,
        one_tick_wide=False,
        two_sided_effective=True,
        intended_two_sided=True,
    )
    qq.record_quote_cycle(
        quoted_spread_bps=2.0,
        one_tick_wide=False,
        two_sided_effective=True,
        intended_two_sided=True,
    )
    # 1 suppressed cycle out of 2 → 0.5 (NOT 3/2 = 1.5).
    rate = qq.suppression_rate_60s()
    assert rate is not None
    assert 0.49 <= rate <= 0.51


def test_per_cycle_latch_resets_between_cycles() -> None:
    """If cycle N is suppressed and cycle N+1 isn't, only cycle N
    counts. The internal ``_last_cycle_was_suppressed`` latch must
    reset on every ``record_quote_cycle`` call."""
    qq = _rollup()
    qq.record_suppression("reason")  # marks cycle 1 as suppressed
    qq.record_quote_cycle(
        quoted_spread_bps=2.0,
        one_tick_wide=False,
        two_sided_effective=True,
        intended_two_sided=True,
    )
    # No suppression for cycle 2.
    qq.record_quote_cycle(
        quoted_spread_bps=2.0,
        one_tick_wide=False,
        two_sided_effective=True,
        intended_two_sided=True,
    )
    rate = qq.suppression_rate_60s()
    assert rate is not None
    assert 0.49 <= rate <= 0.51


def test_old_cycles_drop_from_window() -> None:
    """Cycles older than 60 s are discarded on read. Verify by
    monkeypatching ``time.monotonic`` to advance past the window."""
    qq = _rollup()
    base = 1_000_000.0
    # Record 4 suppressed cycles at t=0.
    with patch("app.quote_quality_telemetry.time.monotonic", return_value=base):
        for _ in range(4):
            _record_cycle(qq, suppressed=True)
        assert qq.suppression_rate_60s() == 1.0
    # Advance 70 s — all old cycles are now outside the 60 s window.
    with patch(
        "app.quote_quality_telemetry.time.monotonic", return_value=base + 70.0
    ):
        # Read alone should drop the old cycles → empty window → None.
        assert qq.suppression_rate_60s() is None
    # Add a fresh non-suppressed cycle at t=80; rate=0.0.
    with patch(
        "app.quote_quality_telemetry.time.monotonic", return_value=base + 80.0
    ):
        _record_cycle(qq, suppressed=False)
        assert qq.suppression_rate_60s() == 0.0


def test_clamp_when_suppressed_count_appears_to_exceed_total() -> None:
    """Defensive: if the deque pruning races somehow leave more
    suppressed-stamps than cycle-stamps in the window (shouldn't
    happen but the deques are independent), the result clamps at 1.0
    rather than overflowing."""
    qq = _rollup()
    # Manually prime the deques in an inconsistent state.
    qq._cycle_ts_mono.append(time.monotonic())
    qq._suppressed_cycle_ts_mono.append(time.monotonic())
    qq._suppressed_cycle_ts_mono.append(time.monotonic())
    rate = qq.suppression_rate_60s()
    assert rate is not None
    assert rate == 1.0
