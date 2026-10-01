"""Post-only cross-reject arms a per-side cooldown.

With ``POST_ONLY_TOUCH_BUFFER_TICKS=1`` (BBO-join), a placement that lands at
BBO can cross-reject if the touch moved since the decision. The bot should:

1. Detect the cross-reject via ``is_post_only_immediate_match_rejection``.
2. Arm ``_post_only_cross_cooldown_until[side]`` for
   ``POST_ONLY_CROSS_COOLDOWN_SECONDS``.
3. Skip placements on that side while the cooldown is active.
4. Resume normally once the cooldown lapses.

This makes BBO-join safe: the worst-case churn is a single cross-reject
followed by a bounded wait, not a tight loop.
"""

from __future__ import annotations

import time

from app.enums import Side
from app.execution import is_post_only_immediate_match_rejection
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides) -> UnitTestSettings:
    base = {"TRADING_ENABLED": False, "POST_ONLY_CROSS_COOLDOWN_SECONDS": 2.5}
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _om():
    from app.clock import SystemClock
    from app.execution import OrderManager

    class _ShellOM:
        pass

    shell = _ShellOM()
    shell._settings = _settings()
    shell._clock = SystemClock()
    shell._post_only_cross_cooldown_until = {Side.BUY: 0.0, Side.SELL: 0.0}
    shell._post_only_cross_cooldown_active = (
        OrderManager._post_only_cross_cooldown_active.__get__(shell, _ShellOM)
    )
    shell._arm_post_only_cross_cooldown = (
        OrderManager._arm_post_only_cross_cooldown.__get__(shell, _ShellOM)
    )
    return shell


def test_detector_recognises_grvt_cross_reject_reasons() -> None:
    """The exchange-specific reject text must trip the detector."""
    assert is_post_only_immediate_match_rejection("Post-only order would cross the book")
    assert is_post_only_immediate_match_rejection("post only: immediately matched")
    assert is_post_only_immediate_match_rejection("post-only would have crossed")


def test_detector_rejects_non_cross_reasons() -> None:
    """Non-cross rejections must not arm the cross cooldown."""
    assert not is_post_only_immediate_match_rejection("insufficient balance")
    assert not is_post_only_immediate_match_rejection("")
    assert not is_post_only_immediate_match_rejection("price tick violation")


def test_arm_sets_cooldown_deadline() -> None:
    om = _om()
    om._arm_post_only_cross_cooldown(Side.BUY)
    assert om._post_only_cross_cooldown_active(Side.BUY)
    # Other side unaffected.
    assert not om._post_only_cross_cooldown_active(Side.SELL)


def test_cooldown_lapses_after_duration() -> None:
    om = _om()
    om._arm_post_only_cross_cooldown(Side.BUY)
    # Fast-forward the deadline.
    om._post_only_cross_cooldown_until[Side.BUY] = time.monotonic() - 0.1
    assert not om._post_only_cross_cooldown_active(Side.BUY)


def test_cooldown_zero_seconds_disables() -> None:
    om = _om()
    om._settings = _settings(POST_ONLY_CROSS_COOLDOWN_SECONDS=0.0)
    om._arm_post_only_cross_cooldown(Side.BUY)
    # Arm returns early when disabled → no deadline set → not active.
    assert not om._post_only_cross_cooldown_active(Side.BUY)


def test_per_side_independent() -> None:
    """Arming one side must not affect the other."""
    om = _om()
    om._arm_post_only_cross_cooldown(Side.SELL)
    assert om._post_only_cross_cooldown_active(Side.SELL)
    assert not om._post_only_cross_cooldown_active(Side.BUY)
    om._arm_post_only_cross_cooldown(Side.BUY)
    assert om._post_only_cross_cooldown_active(Side.BUY)
    assert om._post_only_cross_cooldown_active(Side.SELL)
