"""Phase 4D.5 (v1.4.190) — SF action-rate throttle.

The SF tick worker runs inside the main quote loop, which wakes on
every public-WS book update (~100 Hz on active markets). Without a
throttle the worker re-evaluates the cancel-and-replace decision on
every wake; observed in snapshot
``v1.4.180-260521-141619-prod.okx.ton.usdt.perp``: SF#11171 placed
**4,902 orders in 56 s** (87/sec sustained) to produce 3 fills,
where ~85 of every 87 placements replaced an order at the SAME
price. Pure churn from WS-driven wake bursts.

The throttle suppresses the SF tick's cancel-replace and place
actions when less than ``SF_ACTION_MIN_GAP_MS`` ms have elapsed since
the previous action. Mirrors the Phase 2I per-order amend defence:
same pathology (rapid place/cancel-replace storm), different code
path (SF lives in ``Bot._run_soft_flatten_tick``, not
``OrderManager._enqueue_amend_quote_path``).

Tested via a shell-mock ``Bot`` instance that binds only
``_sf_action_throttled`` + ``_note_sf_throttle_suppression`` against
a synthetic state. This avoids spinning up the full bot pipeline
while exercising the exact predicate and counter logic used by
``_run_soft_flatten_tick``.
"""

from __future__ import annotations

from typing import Optional

import pytest

from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "SF_ACTION_MIN_GAP_MS": 250.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _shell_bot(settings: UnitTestSettings):
    """Build a shell ``Bot`` instance binding only
    ``_sf_action_throttled`` + ``_note_sf_throttle_suppression``
    against a synthetic state."""
    from app.bot import Bot

    class _ShellBot:
        pass

    class _ShellState:
        def __init__(self) -> None:
            self.sf_last_action_mono: Optional[float] = None
            self.sf_throttle_suppressed_total: int = 0
            self.sf_throttle_first_arm_logged: bool = False
            self.soft_flatten_event_id: Optional[int] = 12345

    shell = _ShellBot()
    shell._settings = settings
    shell._state = _ShellState()
    shell._sf_action_throttled = Bot._sf_action_throttled.__get__(
        shell, _ShellBot
    )
    shell._note_sf_throttle_suppression = (
        Bot._note_sf_throttle_suppression.__get__(shell, _ShellBot)
    )
    return shell


# ---------------------------------------------------------------------------
# Throttle predicate
# ---------------------------------------------------------------------------


def test_throttle_returns_false_when_no_prior_action() -> None:
    """First call of an SF episode has ``sf_last_action_mono=None`` —
    the throttle must allow the action."""
    bot = _shell_bot(_settings())
    assert bot._sf_action_throttled(now_mono=100.0) is False


def test_throttle_blocks_within_window() -> None:
    """A second action 100 ms after the first should be suppressed at
    the default 250 ms gap."""
    bot = _shell_bot(_settings(SF_ACTION_MIN_GAP_MS=250.0))
    bot._state.sf_last_action_mono = 100.0
    assert bot._sf_action_throttled(now_mono=100.10) is True  # 100 ms gap


def test_throttle_allows_after_window() -> None:
    """An action 300 ms after the first should be allowed at the
    250 ms gap."""
    bot = _shell_bot(_settings(SF_ACTION_MIN_GAP_MS=250.0))
    bot._state.sf_last_action_mono = 100.0
    assert bot._sf_action_throttled(now_mono=100.30) is False  # 300 ms gap


def test_throttle_disabled_when_gap_is_zero() -> None:
    """``SF_ACTION_MIN_GAP_MS=0`` disables the throttle entirely; any
    action passes through regardless of how recent the last one was."""
    bot = _shell_bot(_settings(SF_ACTION_MIN_GAP_MS=0.0))
    bot._state.sf_last_action_mono = 100.0
    # Even a 1-ms gap is allowed.
    assert bot._sf_action_throttled(now_mono=100.001) is False


def test_throttle_boundary_at_exact_gap() -> None:
    """An action exactly at the gap boundary should be allowed (the
    predicate uses ``<`` with an epsilon margin)."""
    bot = _shell_bot(_settings(SF_ACTION_MIN_GAP_MS=250.0))
    bot._state.sf_last_action_mono = 100.0
    # Exactly 250 ms = right at the threshold; allowed.
    assert bot._sf_action_throttled(now_mono=100.250) is False


# ---------------------------------------------------------------------------
# Counter + first-arm log
# ---------------------------------------------------------------------------


def test_suppression_bumps_counter() -> None:
    bot = _shell_bot(_settings())
    assert bot._state.sf_throttle_suppressed_total == 0
    bot._note_sf_throttle_suppression("cancel_reprice")
    assert bot._state.sf_throttle_suppressed_total == 1
    bot._note_sf_throttle_suppression("place")
    assert bot._state.sf_throttle_suppressed_total == 2


def test_first_arm_logged_only_once_per_episode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The first suppression in an SF episode emits a WARNING; the
    second and Nth are silent (counter still bumps)."""
    bot = _shell_bot(_settings())
    import logging

    with caplog.at_level(logging.WARNING, logger="app.bot"):
        bot._note_sf_throttle_suppression("cancel_reprice")
        bot._note_sf_throttle_suppression("place")
        bot._note_sf_throttle_suppression("cancel_reprice")
    # Counter records all three.
    assert bot._state.sf_throttle_suppressed_total == 3
    # First-arm flag flips to True after the first suppression.
    assert bot._state.sf_throttle_first_arm_logged is True
    # Exactly one WARNING line emitted (across the three suppressions).
    warn_lines = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "sf_action_throttle_armed" in r.getMessage()
    ]
    assert len(warn_lines) == 1


def test_first_arm_log_includes_sf_event_id_and_gap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The WARNING surfaces the SF event id + the configured gap so
    the operator can correlate with the postmortem episode table."""
    bot = _shell_bot(_settings(SF_ACTION_MIN_GAP_MS=125.0))
    bot._state.soft_flatten_event_id = 99001
    import logging

    with caplog.at_level(logging.WARNING, logger="app.bot"):
        bot._note_sf_throttle_suppression("cancel_reprice")

    warn_lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "sf_action_throttle_armed" in r.getMessage()
    ]
    assert len(warn_lines) == 1
    msg = warn_lines[0]
    assert "sf_event_id=99001" in msg
    assert "gap_ms=125" in msg
    assert "cancel_reprice" in msg


# ---------------------------------------------------------------------------
# Settings round-trip
# ---------------------------------------------------------------------------


def test_default_gap_is_250ms() -> None:
    """Default config value is 250 ms per the v1.4.190 plan entry."""
    from app.config import Settings

    s = Settings()
    assert s.sf_action_min_gap_ms == 250.0


def test_gap_setting_round_trips_via_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``SF_ACTION_MIN_GAP_MS`` env var maps onto the typed setting."""
    from app.config import Settings

    monkeypatch.setenv("SF_ACTION_MIN_GAP_MS", "175.5")
    s = Settings()
    assert s.sf_action_min_gap_ms == pytest.approx(175.5)


def test_negative_gap_rejected() -> None:
    """Pydantic ``ge=0.0`` constraint rejects negative values."""
    from pydantic import ValidationError

    from app.config import Settings

    with pytest.raises(ValidationError):
        Settings(SF_ACTION_MIN_GAP_MS=-1.0)


# ---------------------------------------------------------------------------
# Reset on SF enter (covered indirectly: bot._enter_soft_flatten resets
# sf_last_action_mono + sf_throttle_first_arm_logged so the first action
# of a new episode always passes).
# ---------------------------------------------------------------------------


def test_first_action_of_new_episode_after_reset_passes() -> None:
    """Simulates ``_enter_soft_flatten`` resetting throttle state on
    a new SF episode. The first action after reset MUST be allowed,
    even if a previous SF episode had set ``sf_last_action_mono`` to
    a near-recent value."""
    bot = _shell_bot(_settings())
    # Previous episode left state populated.
    bot._state.sf_last_action_mono = 100.0
    bot._state.sf_throttle_first_arm_logged = True
    # _enter_soft_flatten clears them:
    bot._state.sf_last_action_mono = None
    bot._state.sf_throttle_first_arm_logged = False
    # The next action must pass.
    assert bot._sf_action_throttled(now_mono=100.001) is False


# ---------------------------------------------------------------------------
# Action-kind label is recorded distinctly (postmortem channel)
# ---------------------------------------------------------------------------


def test_both_action_kinds_increment_same_counter() -> None:
    """``cancel_reprice`` and ``place`` are two label classes; they
    share the same counter (the operator-facing rate). The first-arm
    log distinguishes them via the ``action_kind`` field but the
    counter is single-valued."""
    bot = _shell_bot(_settings())
    bot._note_sf_throttle_suppression("cancel_reprice")
    bot._note_sf_throttle_suppression("place")
    bot._note_sf_throttle_suppression("cancel_reprice")
    bot._note_sf_throttle_suppression("place")
    assert bot._state.sf_throttle_suppressed_total == 4
