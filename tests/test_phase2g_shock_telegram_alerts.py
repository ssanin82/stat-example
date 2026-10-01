"""Phase 2G (v1.4.204) — SHOCK-mode Telegram alerts.

Two alert classes:

  * **Entry alert** — fires once per SHOCK episode on the
    transition into SHOCK (any → SHOCK).
  * **Persistence alert** — fires once per SHOCK episode when the
    bot has been stuck in SHOCK longer than
    ``REGIME_SHOCK_TELEGRAM_PERSISTENCE_SECONDS`` (default 300 s).
    Caught the v1.4.189 phase-ladder catastrophe in retrospect
    (13 min in SHOCK before SF triggered — a 5-min persistence
    alert would have flagged it ~8 min earlier).

The daily-summary message (Phase 2G.3 of the plan) is deferred —
cron-style scheduling is out of scope for an inline tick check.

Memory note ``feedback_postmortem_preferred_over_runtime_alerts``:
runtime alerts are reserved for rare + operator-actionable cases.
SHOCK qualifies (locks one side, persists until shock_gate clears,
indicates non-routine market conditions).

Tested as a pure-function unit test against a shell ``Bot`` stub —
no live bot, no exchange. The helper is small (~80 lines) but
load-bearing: it's the only Telegram-alert path in the
regime-controller surface.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from app.regime_controller import Mode as RegimeMode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _shell_bot(
    *,
    notifier_present: bool = True,
    alerts_enabled: bool = True,
    persistence_seconds: float = 300.0,
) -> Any:
    """Build a minimal ``Bot`` stub with just the surface
    ``_maybe_alert_shock_telegram`` reads. Plain class (not
    MagicMock) so attribute typos surface as AttributeError."""
    from app.bot import Bot

    class _Stub:
        pass

    class _ShellState:
        def __init__(self) -> None:
            self.shock_entry_telegram_alerted_for_episode: bool = False
            self.shock_persistence_telegram_alerted_for_episode: bool = (
                False
            )
            self.shock_telegram_entry_alerts_sent_total: int = 0
            self.shock_telegram_persistence_alerts_sent_total: int = 0

    class _ShellSettings:
        regime_shock_telegram_alerts_enabled: bool = alerts_enabled
        regime_shock_telegram_persistence_seconds: float = (
            persistence_seconds
        )

    bot = _Stub()
    bot._settings = _ShellSettings()
    bot._state = _ShellState()
    bot._notifier = MagicMock() if notifier_present else None
    # Bind the method.
    bot._maybe_alert_shock_telegram = (
        Bot._maybe_alert_shock_telegram.__get__(bot, _Stub)
    )
    return bot


# ---------------------------------------------------------------------------
# No-op exit paths
# ---------------------------------------------------------------------------


def test_no_alert_when_disabled() -> None:
    """``REGIME_SHOCK_TELEGRAM_ALERTS_ENABLED=false`` short-circuits."""
    bot = _shell_bot(alerts_enabled=False)
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.DEFENSIVE,
        transition_reason="shock_gate_locked",
        util=0.9,
        vol_ratio=2.5,
        now_mono=100.0,
        mode_since_mono=100.0,
    )
    bot._notifier.notify_ops.assert_not_called()


def test_no_alert_when_notifier_absent() -> None:
    """No Telegram bot configured → silent no-op."""
    bot = _shell_bot(notifier_present=False)
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.DEFENSIVE,
        transition_reason="shock_gate_locked",
        util=0.9,
        vol_ratio=2.5,
        now_mono=100.0,
        mode_since_mono=100.0,
    )
    # Counter does not increment when no notifier (no actual alert
    # fired).
    assert bot._state.shock_telegram_entry_alerts_sent_total == 0


def test_no_alert_when_not_in_shock() -> None:
    """When in NORMAL or DEFENSIVE, nothing fires + episode flags
    stay cleared so the next SHOCK entry alerts fresh."""
    bot = _shell_bot()
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.NORMAL,
        prior_mode=RegimeMode.DEFENSIVE,
        transition_reason="defensive_exit:...",
        util=0.2,
        vol_ratio=1.0,
        now_mono=100.0,
        mode_since_mono=100.0,
    )
    bot._notifier.notify_ops.assert_not_called()


# ---------------------------------------------------------------------------
# Entry alert
# ---------------------------------------------------------------------------


def test_entry_alert_fires_on_transition_into_shock() -> None:
    """DEFENSIVE → SHOCK transition → one notify_ops call with
    ``regime_shock_entered``."""
    bot = _shell_bot()
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.DEFENSIVE,
        transition_reason="shock_gate_locked",
        util=0.9,
        vol_ratio=2.5,
        now_mono=100.0,
        mode_since_mono=100.0,
    )
    bot._notifier.notify_ops.assert_called_once()
    args = bot._notifier.notify_ops.call_args.args
    # Signature is (severity, event_type, message, payload).
    assert args[0] == "WARNING"
    assert args[1] == "regime_shock_entered"
    assert "SHOCK entered" in args[2]
    assert args[3]["from"] == "DEFENSIVE"
    assert args[3]["to"] == "SHOCK"
    assert args[3]["util"] == pytest.approx(0.9)
    assert bot._state.shock_telegram_entry_alerts_sent_total == 1
    assert bot._state.shock_entry_telegram_alerted_for_episode is True


def test_entry_alert_one_shot_per_episode() -> None:
    """Second call with mode==SHOCK and no transition reason should
    NOT re-fire the entry alert (the FSM is observing mid-episode)."""
    bot = _shell_bot()
    # First call — entry.
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.DEFENSIVE,
        transition_reason="shock_gate_locked",
        util=0.9,
        vol_ratio=2.5,
        now_mono=100.0,
        mode_since_mono=100.0,
    )
    # Second call — same episode, mid-tick (no transition).
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.SHOCK,
        transition_reason=None,
        util=0.9,
        vol_ratio=2.5,
        now_mono=101.0,
        mode_since_mono=100.0,
    )
    assert bot._notifier.notify_ops.call_count == 1
    assert bot._state.shock_telegram_entry_alerts_sent_total == 1


def test_entry_flag_clears_on_shock_exit() -> None:
    """After exiting SHOCK, the per-episode flag clears so the
    next entry gets a fresh alert."""
    bot = _shell_bot()
    bot._state.shock_entry_telegram_alerted_for_episode = True
    # Exit to DEFENSIVE.
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.DEFENSIVE,
        prior_mode=RegimeMode.SHOCK,
        transition_reason="shock_gate_cleared",
        util=0.5,
        vol_ratio=1.5,
        now_mono=200.0,
        mode_since_mono=200.0,
    )
    assert bot._state.shock_entry_telegram_alerted_for_episode is False


# ---------------------------------------------------------------------------
# Persistence alert
# ---------------------------------------------------------------------------


def test_persistence_alert_does_not_fire_below_threshold() -> None:
    """Dwell < threshold → no persistence alert."""
    bot = _shell_bot(persistence_seconds=300.0)
    # Already alerted on entry — flip the flag so we focus on
    # persistence behaviour.
    bot._state.shock_entry_telegram_alerted_for_episode = True
    # 100s dwell (well below 300s threshold).
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.SHOCK,
        transition_reason=None,
        util=0.9,
        vol_ratio=2.5,
        now_mono=200.0,
        mode_since_mono=100.0,
    )
    bot._notifier.notify_ops.assert_not_called()


def test_persistence_alert_fires_when_threshold_exceeded() -> None:
    """Dwell >= threshold → persistence alert with
    ``regime_shock_persistent``."""
    bot = _shell_bot(persistence_seconds=300.0)
    bot._state.shock_entry_telegram_alerted_for_episode = True
    # 350s dwell (above 300s).
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.SHOCK,
        transition_reason=None,
        util=0.9,
        vol_ratio=2.5,
        now_mono=450.0,
        mode_since_mono=100.0,
    )
    bot._notifier.notify_ops.assert_called_once()
    args = bot._notifier.notify_ops.call_args.args
    assert args[1] == "regime_shock_persistent"
    assert "persistent" in args[2]
    assert args[3]["dwell_seconds"] == pytest.approx(350.0)
    assert args[3]["threshold_seconds"] == pytest.approx(300.0)
    assert bot._state.shock_telegram_persistence_alerts_sent_total == 1


def test_persistence_alert_one_shot_per_episode() -> None:
    """Repeated mid-episode ticks above the threshold fire the
    alert only once."""
    bot = _shell_bot(persistence_seconds=300.0)
    bot._state.shock_entry_telegram_alerted_for_episode = True
    for t in (450.0, 500.0, 600.0, 700.0):
        bot._maybe_alert_shock_telegram(
            mode=RegimeMode.SHOCK,
            prior_mode=RegimeMode.SHOCK,
            transition_reason=None,
            util=0.9,
            vol_ratio=2.5,
            now_mono=t,
            mode_since_mono=100.0,
        )
    assert bot._notifier.notify_ops.call_count == 1
    assert bot._state.shock_telegram_persistence_alerts_sent_total == 1


def test_persistence_flag_clears_on_shock_exit() -> None:
    """Persistence flag follows the same clear-on-exit contract as
    the entry flag."""
    bot = _shell_bot(persistence_seconds=300.0)
    bot._state.shock_persistence_telegram_alerted_for_episode = True
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.NORMAL,
        prior_mode=RegimeMode.SHOCK,
        transition_reason="defensive_exit:...",
        util=0.2,
        vol_ratio=1.0,
        now_mono=800.0,
        mode_since_mono=800.0,
    )
    assert (
        bot._state.shock_persistence_telegram_alerted_for_episode is False
    )


def test_persistence_threshold_zero_disables() -> None:
    """``REGIME_SHOCK_TELEGRAM_PERSISTENCE_SECONDS=0`` disables the
    persistence alert (entry alert still fires)."""
    bot = _shell_bot(persistence_seconds=0.0)
    bot._state.shock_entry_telegram_alerted_for_episode = True
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.SHOCK,
        transition_reason=None,
        util=0.9,
        vol_ratio=2.5,
        now_mono=10000.0,  # arbitrarily long dwell
        mode_since_mono=100.0,
    )
    bot._notifier.notify_ops.assert_not_called()


# ---------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------


def test_default_alerts_enabled_true() -> None:
    """Default is ON. Memory note: SHOCK is the only alert class
    that survived the postmortem-preferred filter."""
    from app.config import Settings

    s = Settings()
    assert s.regime_shock_telegram_alerts_enabled is True
    assert s.regime_shock_telegram_persistence_seconds == 300.0


def test_alerts_enabled_round_trip_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config import Settings

    monkeypatch.setenv("REGIME_SHOCK_TELEGRAM_ALERTS_ENABLED", "false")
    monkeypatch.setenv("REGIME_SHOCK_TELEGRAM_PERSISTENCE_SECONDS", "180.0")
    s = Settings()
    assert s.regime_shock_telegram_alerts_enabled is False
    assert s.regime_shock_telegram_persistence_seconds == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# Regression replay — the v1.4.189 phase-ladder catastrophe
# ---------------------------------------------------------------------------


def test_regression_v1_4_189_persistence_alert_would_have_fired() -> None:
    """The v1.4.189 (2026-05-21) phase-ladder catastrophe persisted
    13 min in SHOCK before SF cleared the position. With the
    default 5-min persistence threshold, the alert would have
    fired at ~5 min — giving the operator 8 min to react before
    the SF terminal-market_close blow.

    Replay: enter SHOCK at t=0; the bot ticks every ~500 ms; at
    t=300s the persistence alert fires; at t=780s (13 min) the
    real SF triggered (we don't simulate the SF here — just verify
    the alert was already sent)."""
    bot = _shell_bot(persistence_seconds=300.0)

    # Use a realistic ``time.monotonic()`` value as the SHOCK entry
    # baseline — production values are typically the system uptime
    # in seconds (large positive). The helper guards against the
    # 0.0 default with ``mode_since_mono > 0`` so an episode that
    # never had its since-time stamped doesn't spuriously fire the
    # persistence alert.
    SHOCK_ENTRY_MONO = 1000.0
    # Entry.
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.DEFENSIVE,
        transition_reason="shock_gate_locked",
        util=0.9,
        vol_ratio=2.5,
        now_mono=SHOCK_ENTRY_MONO,
        mode_since_mono=SHOCK_ENTRY_MONO,
    )
    # Tick mid-episode for 5 minutes.
    for offset in (60.0, 120.0, 180.0, 240.0, 295.0):
        bot._maybe_alert_shock_telegram(
            mode=RegimeMode.SHOCK,
            prior_mode=RegimeMode.SHOCK,
            transition_reason=None,
            util=0.9,
            vol_ratio=2.5,
            now_mono=SHOCK_ENTRY_MONO + offset,
            mode_since_mono=SHOCK_ENTRY_MONO,
        )
    # At t=295s (4m 55s) the persistence alert has NOT fired yet.
    assert bot._state.shock_telegram_persistence_alerts_sent_total == 0

    # Cross the 5-min boundary.
    bot._maybe_alert_shock_telegram(
        mode=RegimeMode.SHOCK,
        prior_mode=RegimeMode.SHOCK,
        transition_reason=None,
        util=0.9,
        vol_ratio=2.5,
        now_mono=SHOCK_ENTRY_MONO + 305.0,
        mode_since_mono=SHOCK_ENTRY_MONO,
    )
    # Persistence alert fired.
    assert bot._state.shock_telegram_persistence_alerts_sent_total == 1
    # Entry alert also counted (separately).
    assert bot._state.shock_telegram_entry_alerts_sent_total == 1
