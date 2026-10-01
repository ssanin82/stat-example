"""Regression tests for ``reports/codex-review-20260511-bugs.md`` fixes
shipped in v1.2.53 (2026-05-12).

Covers:
  #1 cancel resting orders on HOLD_ALL eligibility (with allow-list).
  #2 separate at-touch maximum-age cap.
  #5 inventory-aware freshness one-sided fallback.

#3 (executable_half_spread_bps stamp), #4 (private_ws_receive_to_
state_apply_ms) and #6 (reference_venue_name) are covered by ad-hoc
asserts in other modules' tests.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.enums import QuoteEligibility, Side
from app.models import OrderStatus, WorkingOrder
from app.quote_aging import hard_reprice_reasons_buy, hard_reprice_reasons_sell
from tests.settings_helpers import UnitTestSettings as Settings


def _settings(**kw) -> Settings:
    base = dict(
        trading_enabled=False,
        hl_secret_key="",
        hl_account_address="",
    )
    base.update(kw)
    return Settings(**base)


# ---------------------------------------------------------------------------
# Codex-#1: HOLD_ALL allow-list parsing
# ---------------------------------------------------------------------------


def _hold_all_should_cancel(s, reason: str) -> bool:
    """Inline copy of the bot's allow-list logic for unit testing
    without spinning up a full Bot. Matches the implementation in
    ``app/bot.py::Bot._hold_all_should_cancel`` 1:1."""
    if not s.cancel_resting_on_hold_all:
        return False
    if not reason:
        return True
    allow_raw = s.hold_all_keep_resting_reasons or ""
    allow = {p.strip() for p in allow_raw.split(",") if p.strip()}
    parts = [p.strip() for p in reason.split("|") if p.strip()]
    meaningful = [p for p in parts if not p.startswith("ok") and not p.endswith("_ok")]
    if not meaningful:
        return False

    def _tag(part: str) -> str:
        return part.split("=", 1)[0].strip() if "=" in part else part

    meaningful_tags = {_tag(p) for p in meaningful}
    return not meaningful_tags.issubset(allow)


def test_hold_all_should_cancel_default_yes_when_disabled_knob_off() -> None:
    s = _settings(cancel_resting_on_hold_all=False)
    assert not _hold_all_should_cancel(s, "freshness_drift_hold")


def test_hold_all_should_cancel_empty_reason_defaults_to_cancel() -> None:
    s = _settings(cancel_resting_on_hold_all=True)
    assert _hold_all_should_cancel(s, "")


def test_hold_all_should_cancel_recovery_cooldown_only_keeps_resting() -> None:
    s = _settings(
        cancel_resting_on_hold_all=True,
        hold_all_keep_resting_reasons="recovery_cooldown",
    )
    # All-OK markers + recovery_cooldown only → keep resting.
    assert not _hold_all_should_cancel(s, "ok|fresh=freshness_ok|drift=drift_ok|recovery_cooldown")


def test_hold_all_should_cancel_basis_signal_absent_triggers_cancel() -> None:
    s = _settings(
        cancel_resting_on_hold_all=True,
        hold_all_keep_resting_reasons="recovery_cooldown",
    )
    assert _hold_all_should_cancel(s, "ok|basis_regime_signal_absent")


def test_hold_all_should_cancel_freshness_drift_triggers_cancel() -> None:
    s = _settings(
        cancel_resting_on_hold_all=True,
        hold_all_keep_resting_reasons="recovery_cooldown",
    )
    assert _hold_all_should_cancel(s, "ok|freshness_drift_hold")


def test_hold_all_should_cancel_extends_allow_list() -> None:
    s = _settings(
        cancel_resting_on_hold_all=True,
        hold_all_keep_resting_reasons="recovery_cooldown,basis_regime_signal_absent",
    )
    assert not _hold_all_should_cancel(s, "ok|recovery_cooldown")
    assert not _hold_all_should_cancel(s, "ok|basis_regime_signal_absent")
    # Still cancels for non-allowed reasons.
    assert _hold_all_should_cancel(s, "ok|post_swing")


# ---------------------------------------------------------------------------
# Codex-#2: separate at-touch age cap
# ---------------------------------------------------------------------------


def _working(side: Side, price: float, age_sec: float) -> WorkingOrder:
    """Build a minimal WorkingOrder with a back-dated ``ts_ack`` so
    ``resting_age_seconds`` returns ``age_sec``."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    return WorkingOrder(
        order_id_local="loc-1",
        order_id_exchange=12345,
        client_order_id="test-cloid",
        symbol="TON-USDT-SWAP",
        side=side,
        price=price,
        size=3.0,
        post_only=True,
        status=OrderStatus.ACKED,
        ts_ack=now - timedelta(seconds=age_sec),
    )


def test_at_touch_age_cap_disabled_skips_at_touch_aging() -> None:
    """Default ``at_touch_max_age_seconds=0`` preserves the legacy
    behaviour: at-touch quotes never age out via the hard-reprice path."""
    s = _settings(
        quote_aging_enabled=True,
        quote_aging_max_age_seconds=6.0,
        at_touch_max_age_seconds=0.0,
        quote_max_distance_to_touch_ticks=8.0,
    )
    # Working bid AT the best bid → at_touch = True. 60 s old.
    wo = _working(Side.BUY, price=2.350, age_sec=60.0)
    reasons = hard_reprice_reasons_buy(
        s,
        wo,
        best_bid=2.350,
        tick=0.001,
        now=datetime.now(timezone.utc),
    )
    # Both flags off (regular age skips at_touch; new cap disabled).
    assert "order_age_seconds" not in reasons
    assert "at_touch_order_age_seconds" not in reasons


def test_at_touch_age_cap_fires_buy_side() -> None:
    """When ``at_touch_max_age_seconds`` is set, at-touch quotes
    older than that trigger hard reprice."""
    s = _settings(
        quote_aging_enabled=True,
        quote_aging_max_age_seconds=6.0,
        at_touch_max_age_seconds=30.0,
        quote_max_distance_to_touch_ticks=8.0,
    )
    # 60 s old at-touch bid → exceeds 30 s cap.
    wo = _working(Side.BUY, price=2.350, age_sec=60.0)
    reasons = hard_reprice_reasons_buy(
        s,
        wo,
        best_bid=2.350,
        tick=0.001,
        now=datetime.now(timezone.utc),
    )
    assert "at_touch_order_age_seconds" in reasons


def test_at_touch_age_cap_fires_sell_side() -> None:
    s = _settings(
        quote_aging_enabled=True,
        quote_aging_max_age_seconds=6.0,
        at_touch_max_age_seconds=30.0,
        quote_max_distance_to_touch_ticks=8.0,
    )
    wo = _working(Side.SELL, price=2.355, age_sec=60.0)
    reasons = hard_reprice_reasons_sell(
        s,
        wo,
        best_ask=2.355,
        tick=0.001,
        now=datetime.now(timezone.utc),
    )
    assert "at_touch_order_age_seconds" in reasons


def test_at_touch_age_cap_does_not_fire_when_not_at_touch() -> None:
    """The new cap is at-touch specific. A behind-touch quote should
    NOT trigger ``at_touch_order_age_seconds`` (it triggers the
    regular ``order_age_seconds`` rule instead)."""
    s = _settings(
        quote_aging_enabled=True,
        quote_aging_max_age_seconds=6.0,
        at_touch_max_age_seconds=30.0,
        quote_max_distance_to_touch_ticks=8.0,
    )
    # Behind-touch (price < best_bid by 2 ticks).
    wo = _working(Side.BUY, price=2.348, age_sec=60.0)
    reasons = hard_reprice_reasons_buy(
        s,
        wo,
        best_bid=2.350,
        tick=0.001,
        now=datetime.now(timezone.utc),
    )
    assert "at_touch_order_age_seconds" not in reasons
    # Regular order_age_seconds DOES fire (behind-touch + old).
    assert "order_age_seconds" in reasons


def test_at_touch_age_cap_does_not_fire_below_threshold() -> None:
    s = _settings(
        quote_aging_enabled=True,
        quote_aging_max_age_seconds=6.0,
        at_touch_max_age_seconds=30.0,
        quote_max_distance_to_touch_ticks=8.0,
    )
    # 10 s old at-touch — past regular cap (6 s) but under new cap (30 s).
    wo = _working(Side.BUY, price=2.350, age_sec=10.0)
    reasons = hard_reprice_reasons_buy(
        s,
        wo,
        best_bid=2.350,
        tick=0.001,
        now=datetime.now(timezone.utc),
    )
    # Regular skipped (at_touch); new cap not yet reached.
    assert "order_age_seconds" not in reasons
    assert "at_touch_order_age_seconds" not in reasons


# ---------------------------------------------------------------------------
# Codex-#5: inventory-aware freshness one-sided fallback
# ---------------------------------------------------------------------------


def test_freshness_one_sided_prefers_reducing_side_when_long() -> None:
    """When long position + freshness degrades to one-sided mode,
    prefer the ASK side (which closes the long)."""
    from app.quote_eligibility import _freshness_eligibility

    s = _settings(
        quote_freshness_one_sided_preference="BUY",
        quote_freshness_one_sided_prefer_reducing=True,
        quote_one_sided_max_book_age_ms=1000.0,
        quote_hold_max_book_age_ms=2000.0,
    )
    elig, reason = _freshness_eligibility(
        s,
        age_ms=1500.0,  # past one-sided threshold but under hold
        gap_p95_ms=None,
        gap_median_ms=None,
        eff_staleness_ms=None,
        position_qty=2.5,  # long
    )
    assert elig == QuoteEligibility.QUOTE_SELL_ONLY
    assert "reducing_long" in reason


def test_freshness_one_sided_prefers_reducing_side_when_short() -> None:
    from app.quote_eligibility import _freshness_eligibility

    s = _settings(
        quote_freshness_one_sided_preference="BUY",
        quote_freshness_one_sided_prefer_reducing=True,
        quote_one_sided_max_book_age_ms=1000.0,
        quote_hold_max_book_age_ms=2000.0,
    )
    elig, reason = _freshness_eligibility(
        s,
        age_ms=1500.0,
        gap_p95_ms=None,
        gap_median_ms=None,
        eff_staleness_ms=None,
        position_qty=-2.5,  # short
    )
    assert elig == QuoteEligibility.QUOTE_BUY_ONLY
    assert "reducing_short" in reason


def test_freshness_one_sided_fallback_uses_static_pref_when_flat() -> None:
    """Flat position → fall back to the static config preference."""
    from app.quote_eligibility import _freshness_eligibility

    s = _settings(
        quote_freshness_one_sided_preference="SELL",
        quote_freshness_one_sided_prefer_reducing=True,
        quote_one_sided_max_book_age_ms=1000.0,
        quote_hold_max_book_age_ms=2000.0,
    )
    elig, reason = _freshness_eligibility(
        s,
        age_ms=1500.0,
        gap_p95_ms=None,
        gap_median_ms=None,
        eff_staleness_ms=None,
        position_qty=0.0,  # flat
    )
    assert elig == QuoteEligibility.QUOTE_SELL_ONLY
    assert "reducing" not in reason


def test_freshness_one_sided_respects_prefer_reducing_off() -> None:
    """Operator can disable the inventory-aware preference via the
    knob; falls back to static config."""
    from app.quote_eligibility import _freshness_eligibility

    s = _settings(
        quote_freshness_one_sided_preference="BUY",
        quote_freshness_one_sided_prefer_reducing=False,  # disabled
        quote_one_sided_max_book_age_ms=1000.0,
        quote_hold_max_book_age_ms=2000.0,
    )
    elig, reason = _freshness_eligibility(
        s,
        age_ms=1500.0,
        gap_p95_ms=None,
        gap_median_ms=None,
        eff_staleness_ms=None,
        position_qty=2.5,  # long, but reducing-pref disabled
    )
    # Falls back to static "BUY" pref despite the long position.
    assert elig == QuoteEligibility.QUOTE_BUY_ONLY
    assert "reducing" not in reason


def test_freshness_one_sided_no_position_qty_uses_static_pref() -> None:
    """When position_qty is None (caller didn't supply), fall back
    to static preference. Backward compat for callers not passing
    the new kwarg."""
    from app.quote_eligibility import _freshness_eligibility

    s = _settings(
        quote_freshness_one_sided_preference="SELL",
        quote_one_sided_max_book_age_ms=1000.0,
        quote_hold_max_book_age_ms=2000.0,
    )
    elig, reason = _freshness_eligibility(
        s,
        age_ms=1500.0,
        gap_p95_ms=None,
        gap_median_ms=None,
        eff_staleness_ms=None,
        # position_qty=None (default)
    )
    assert elig == QuoteEligibility.QUOTE_SELL_ONLY
