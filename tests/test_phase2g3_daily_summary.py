"""Phase 2G.3 (v1.5.26) -- daily time-in-mode Telegram summary.

Pre-v1.5.26 this was deferred as "needs cron-style scheduling
out of scope for an inline tick check." v1.5.26 implements it
without cron: ``_maybe_send_daily_regime_summary`` is called once
per regime-controller tick and short-circuits cheaply when the
configured interval (default 24h) hasn't elapsed since the last
send. The cadence check is a single float subtract; the actual
send is fire-and-forget via ``notify_ops``.

Tested as a pure-function unit test against a shell ``Bot`` stub
-- no live bot, no exchange.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.regime_controller import Mode as RegimeMode


def _shell_bot(
    *,
    notifier_present: bool = True,
    enabled: bool = True,
    interval_s: float = 86400.0,
    last_sent_mono: float = 0.0,
    time_in_normal_s: float = 0.0,
    time_in_defensive_s: float = 0.0,
    time_in_shock_s: float = 0.0,
    time_in_calm_s: float = 0.0,
    time_in_cautious_s: float = 0.0,
    transitions: int = 0,
    mode: RegimeMode = RegimeMode.NORMAL,
) -> Any:
    """Build a minimal ``Bot`` stub with just the surface
    ``_maybe_send_daily_regime_summary`` reads."""
    from app.bot import Bot

    class _RC:
        def __init__(self) -> None:
            self.mode = mode
            self.time_in_normal_seconds = time_in_normal_s
            self.time_in_defensive_seconds = time_in_defensive_s
            self.time_in_shock_seconds = time_in_shock_s
            self.time_in_calm_seconds = time_in_calm_s
            self.time_in_cautious_seconds = time_in_cautious_s
            self.transition_count = transitions

    class _State:
        def __init__(self) -> None:
            self.regime_controller = _RC()
            self.regime_daily_summary_last_sent_mono = last_sent_mono
            self.regime_daily_summary_sent_total = 0

    class _Settings:
        regime_daily_summary_enabled = enabled
        regime_daily_summary_interval_seconds = interval_s

    class _Stub:
        pass

    stub = _Stub()
    stub._settings = _Settings()
    stub._state = _State()
    stub._notifier = MagicMock() if notifier_present else None
    # Bind the bot method to the stub via __get__.
    stub._maybe_send_daily_regime_summary = (
        Bot._maybe_send_daily_regime_summary.__get__(stub, type(stub))
    )
    return stub


# ---------------------------------------------------------------------------
# Dormant cases (no-op pass-through)
# ---------------------------------------------------------------------------


def test_disabled_setting_no_op():
    stub = _shell_bot(enabled=False, last_sent_mono=0.0)
    stub._maybe_send_daily_regime_summary(now_mono=200_000.0)
    stub._notifier.notify_ops.assert_not_called()
    assert stub._state.regime_daily_summary_last_sent_mono == 0.0
    assert stub._state.regime_daily_summary_sent_total == 0


def test_no_notifier_no_op():
    stub = _shell_bot(notifier_present=False, last_sent_mono=0.0)
    # Must not raise.
    stub._maybe_send_daily_regime_summary(now_mono=200_000.0)
    assert stub._state.regime_daily_summary_last_sent_mono == 0.0
    assert stub._state.regime_daily_summary_sent_total == 0


def test_zero_interval_no_op():
    """``regime_daily_summary_interval_seconds=0`` is treated as
    'disabled' (don't divide by zero, don't fire every tick)."""
    stub = _shell_bot(interval_s=0.0, last_sent_mono=0.0)
    stub._maybe_send_daily_regime_summary(now_mono=200_000.0)
    stub._notifier.notify_ops.assert_not_called()


def test_interval_not_elapsed_no_op():
    """now - last_sent < interval => no fire."""
    stub = _shell_bot(
        interval_s=86400.0,
        last_sent_mono=100_000.0,
    )
    # 12h since last send; interval is 24h.
    stub._maybe_send_daily_regime_summary(now_mono=143_200.0)
    stub._notifier.notify_ops.assert_not_called()
    # Counter untouched.
    assert stub._state.regime_daily_summary_sent_total == 0


# ---------------------------------------------------------------------------
# Active cases (fires + side effects)
# ---------------------------------------------------------------------------


def test_first_fire_at_exact_interval_boundary():
    """At now_mono == interval (inclusive), the summary fires.
    last_sent_mono=0 represents 'never sent in this process'; the
    first send fires at process start + interval. Validates the
    >= comparison is inclusive."""
    stub = _shell_bot(
        interval_s=86400.0,
        last_sent_mono=0.0,
        mode=RegimeMode.NORMAL,
        time_in_normal_s=86000.0,
        time_in_defensive_s=400.0,
    )
    stub._maybe_send_daily_regime_summary(now_mono=86400.0)
    stub._notifier.notify_ops.assert_called_once()
    assert stub._state.regime_daily_summary_sent_total == 1
    assert stub._state.regime_daily_summary_last_sent_mono == 86400.0


def test_fire_payload_includes_per_mode_counters():
    """Payload carries the per-mode time totals + transition count
    so the operator can read 'CALM 4.5h NORMAL 18h DEFENSIVE 12m
    transitions 7' at a glance."""
    stub = _shell_bot(
        interval_s=3600.0,  # 1h for test speed
        last_sent_mono=0.0,
        mode=RegimeMode.DEFENSIVE,
        time_in_normal_s=2700.0,
        time_in_defensive_s=600.0,
        time_in_shock_s=120.0,
        time_in_calm_s=180.0,
        time_in_cautious_s=300.0,
        transitions=7,
    )
    stub._maybe_send_daily_regime_summary(now_mono=3600.0)
    call = stub._notifier.notify_ops.call_args
    args, kwargs = call.args, call.kwargs
    # notify_ops(level, event_name, message, payload)
    assert args[0] == "INFO"
    assert args[1] == "regime_daily_summary"
    payload = args[3]
    assert payload["current_mode"] == "DEFENSIVE"
    assert payload["time_in_normal_s"] == 2700.0
    assert payload["time_in_defensive_s"] == 600.0
    assert payload["time_in_shock_s"] == 120.0
    assert payload["time_in_calm_s"] == 180.0
    assert payload["time_in_cautious_s"] == 300.0
    assert payload["transitions_total"] == 7
    # Message is the human-readable line; spot-check that mode tokens
    # appear (compact form: only non-zero modes listed).
    message = args[2]
    assert "DEFENSIVE" in message
    assert "transitions: 7" in message


def test_fire_message_compact_when_only_one_mode_active():
    """When the session sat in one mode the whole time (e.g. NORMAL),
    the message lists only that mode -- doesn't pad with zeros."""
    stub = _shell_bot(
        interval_s=3600.0,
        last_sent_mono=0.0,
        mode=RegimeMode.NORMAL,
        time_in_normal_s=3600.0,
        # All others = 0.
    )
    stub._maybe_send_daily_regime_summary(now_mono=3600.0)
    call = stub._notifier.notify_ops.call_args
    message = call.args[2]
    assert "NORMAL" in message
    # Modes with 0 seconds should NOT appear in the compact message.
    assert "CALM" not in message
    assert "DEFENSIVE" not in message
    assert "SHOCK" not in message
    assert "CAUTIOUS" not in message


def test_second_fire_only_after_full_interval():
    """After a first fire at T, the next fire requires another
    ``interval`` seconds. At T + interval/2 no fire; at T + interval fires."""
    stub = _shell_bot(
        interval_s=3600.0,
        last_sent_mono=0.0,
        time_in_normal_s=3600.0,
    )
    # First fire at T = 3600s.
    stub._maybe_send_daily_regime_summary(now_mono=3600.0)
    assert stub._state.regime_daily_summary_sent_total == 1
    last_sent = stub._state.regime_daily_summary_last_sent_mono

    # Half-interval later: no fire.
    stub._maybe_send_daily_regime_summary(now_mono=3600.0 + 1800.0)
    assert stub._state.regime_daily_summary_sent_total == 1
    assert stub._state.regime_daily_summary_last_sent_mono == last_sent

    # Full interval later: fires.
    stub._maybe_send_daily_regime_summary(now_mono=3600.0 + 3600.0)
    assert stub._state.regime_daily_summary_sent_total == 2
    assert stub._state.regime_daily_summary_last_sent_mono == 3600.0 + 3600.0


def test_notify_ops_failure_does_not_corrupt_state():
    """If notify_ops raises (e.g. Telegram API down), the helper
    catches the exception and leaves ``last_sent_mono`` UNCHANGED
    so the next tick retries. Counter also stays put."""
    stub = _shell_bot(
        interval_s=3600.0,
        last_sent_mono=0.0,
        time_in_normal_s=3600.0,
    )
    stub._notifier.notify_ops.side_effect = RuntimeError("telegram down")
    # Must not raise.
    stub._maybe_send_daily_regime_summary(now_mono=3600.0)
    # State remains as-if-not-sent so the next tick retries.
    assert stub._state.regime_daily_summary_last_sent_mono == 0.0
    assert stub._state.regime_daily_summary_sent_total == 0
