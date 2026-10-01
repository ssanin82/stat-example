"""v1.5.291 — residual_below_min_notional -> sf_fatigue_tier4 deadlock fix.

Two independent guards, each tested in isolation:

* **Fix B (toxicity markout-hard min-fill gate)** — pure
  ``ToxicityEngine`` unit tests. The markout hard-trigger path
  (``avg_adv <= -TOXICITY_MARKOUT_HARD_BPS``) must require at least
  ``TOXICITY_MARKOUT_HARD_MIN_FILLS`` *resolved-markout* fills before
  it can declare a hard trigger. Averaging 1-3 fills is statistical
  noise that should not drive a full SOFT_FLATTEN. Default 0 preserves
  the pre-v1.5.291 legacy behaviour (any single adverse fill trips).

* **Fix A (sub-min-notional SF entry guard)** — ``Bot._enter_soft_
  flatten`` must refuse to start an SF episode (and must NOT count it
  toward the sf_fatigue ladder) when the residual position notional is
  below the ``max(venue min_notional_usd, local min_quote_notional_
  usd)`` floor. Such a position cannot be flattened by a single closing
  order — the SF worker would immediately exit
  ``residual_below_min_notional`` — so the no-op episode must never
  start, and never be counted as fatigue.

Together these break the v1.5.290-260530-223859 tier-4 kill loop: a
2-fill -8 bp markout window tripped toxicity-hard on a sub-min-notional
residual, drove a no-op SOFT_FLATTEN that the fatigue ladder counted as
real churn, and 12 such re-fires escalated to KILL over a $0.006 loss.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.enums import BotStatus, Side
from app.models import Fill, PnlSnapshot, PositionSnapshot, BestBidAsk
from app.sf_fatigue_gate import event_count_in_window
from app.toxicity import ToxicityEngine


# ===========================================================================
# Fix B — toxicity markout-hard min-fill gate
# ===========================================================================


def _engine(min_fills: int) -> ToxicityEngine:
    """ToxicityEngine with the markout hard path gated at ``min_fills``."""
    s = Settings.model_construct(
        toxicity_enabled=True,
        toxicity_markout_soft_bps=3.0,
        toxicity_markout_hard_bps=12.0,
        toxicity_markout_hard_min_fills=min_fills,
        toxicity_one_sided_fill_ratio=0.75,
        toxicity_one_sided_min_fills=4,
        toxicity_recent_fills_max_age_seconds=600.0,
    )
    eng = ToxicityEngine(s)
    eng.set_baseline_vol(1.0)
    return eng


def _fill(fill_id: str, m5: float | None, side: Side = Side.BUY) -> Fill:
    """A fill with a resolved (or, if ``m5`` is None, unresolved) markout."""
    return Fill(
        fill_id=fill_id,
        order_id_exchange=1,
        client_order_id=None,
        ts_fill=datetime.now(timezone.utc),
        symbol="TON-USDT-SWAP",
        side=side,
        price=1.75,
        size=1.0,
        notional=1.75,
        fee=0.0,
        liquidity_flag="x",
        mid_at_fill=1.75,
        markout_1s_bps=m5,
        markout_3s_bps=m5,
        markout_5s_bps=m5,
    )


def test_markout_hard_gate_blocks_two_fill_hard() -> None:
    """The incident: 2 strongly-adverse fills (-100 bps, well past the
    12 bp hard line). With min_fills=4 the markout hard path is gated
    on sample size, so hard_trigger stays False. delayed_count is still
    reported (2) for observability."""
    eng = _engine(min_fills=4)
    fills = [_fill("a", -100.0), _fill("b", -100.0)]
    snap = eng.snapshot(mid=1.75, current_vol_bps=1.0, fills=fills)
    assert snap.delayed_markout_sample_count == 2
    assert snap.avg_adverse_markout_bps <= -12.0  # would have tripped legacy
    assert snap.hard_trigger is False


def test_markout_hard_gate_default_zero_preserves_legacy() -> None:
    """min_fills=0 (default) => gate disabled => legacy behaviour: even
    a 2-fill window past the hard threshold trips hard. This pins the
    backward-compat contract that protects existing tests / sessions."""
    eng = _engine(min_fills=0)
    fills = [_fill("a", -100.0), _fill("b", -100.0)]
    snap = eng.snapshot(mid=1.75, current_vol_bps=1.0, fills=fills)
    assert snap.delayed_markout_sample_count == 2
    assert snap.hard_trigger is True


def test_markout_hard_gate_allows_genuine_burst() -> None:
    """A genuine toxic burst (4 adverse fills at the min_fills=4 floor)
    still trips hard. The gate suppresses noise, not real toxicity."""
    eng = _engine(min_fills=4)
    fills = [_fill(str(i), -100.0) for i in range(4)]
    snap = eng.snapshot(mid=1.75, current_vol_bps=1.0, fills=fills)
    assert snap.delayed_markout_sample_count == 4
    assert snap.hard_trigger is True


def test_markout_hard_gate_boundary_three_blocked() -> None:
    """One below the floor (3 < 4) is still blocked — the markout path
    cannot trip hard, and with <8 fills the one-sided hard path is
    structurally unreachable either."""
    eng = _engine(min_fills=4)
    fills = [_fill(str(i), -100.0) for i in range(3)]
    snap = eng.snapshot(mid=1.75, current_vol_bps=1.0, fills=fills)
    assert snap.delayed_markout_sample_count == 3
    assert snap.hard_trigger is False


def test_markout_hard_gate_does_not_disable_one_sided_path() -> None:
    """The min-fill gate touches ONLY the markout leg. The independent
    one-sided hard path (one_sided >= 0.9 and len(fl) >= 8) must still
    fire even when no fill has a resolved markout (delayed_count=0)."""
    eng = _engine(min_fills=4)
    # 8 same-side fills, NONE with a resolved markout.
    fills = [_fill(str(i), None, side=Side.BUY) for i in range(8)]
    snap = eng.snapshot(mid=1.75, current_vol_bps=1.0, fills=fills)
    assert snap.delayed_markout_sample_count == 0
    assert snap.one_sided_fill_ratio >= 0.9
    assert snap.hard_trigger is True


# ===========================================================================
# Fix A — sub-min-notional SF entry guard
# ===========================================================================
#
# Harness mirrors tests/test_soft_flatten_safety_integrations.py: build
# Bot via __new__ + manual attribute injection so we exercise the real
# ``_enter_soft_flatten`` guard without a live exchange / DB / WS stack.


def _make_settings(**kw) -> Settings:
    from tests.settings_helpers import UnitTestSettings

    path = Path(tempfile.gettempdir()) / f"mm_v1591_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 1_000.0,
        "MAX_POSITION_NOTIONAL_USD": 10_000.0,
        "MAX_ORDER_NOTIONAL_USD": 1_000.0,
        # Local floor LARGER than the venue floor (the TON case: $5.5
        # local vs $5.0 venue) so the max() vs venue-only distinction
        # is testable.
        "MIN_QUOTE_NOTIONAL_USD": 5.5,
        # Re-entry cooldown is irrelevant here (no prior exit), but pin
        # it so a profile default change can't perturb these tests.
        "SOFT_FLATTEN_REENTRY_COOLDOWN_SECONDS": 30.0,
    }
    base.update(kw)
    return UnitTestSettings.model_validate(base)


def _state(*, pos_qty: float, best_bid: float, best_ask: float, settings: Settings):
    from app.state import BotState

    state = BotState(settings)
    mid = (best_bid + best_ask) / 2.0
    state.position = PositionSnapshot(
        symbol=settings.symbol,
        position_qty=pos_qty,
        avg_entry_price=mid,
        mark_price=mid,
        position_notional=abs(pos_qty) * mid,
        unrealized_pnl_usd=0.0,
    )
    state.market = BestBidAsk(
        symbol=settings.symbol,
        best_bid=best_bid,
        best_ask=best_ask,
        mid_price=mid,
        spread_bps=(best_ask - best_bid) / best_bid * 10_000.0 if best_bid > 0 else 0.0,
        bid_size=100.0,
        ask_size=100.0,
    )
    state.bot_status = BotStatus.RUNNING
    return state


def _bot(state, settings, *, venue_min_notional_usd: float):
    from app.bot import Bot
    from app.clock import SystemClock

    bot = Bot.__new__(Bot)
    bot._state = state
    bot._settings = settings
    bot._clock = SystemClock()
    bot._exec = MagicMock()
    client_mock = MagicMock()
    client_mock.symbol_spec = MagicMock(
        price_tick=0.0001,
        size_step=1.0,
        min_size=1.0,
        min_notional_usd=venue_min_notional_usd,
    )
    bot._client = client_mock
    bot._pnl = MagicMock()
    bot._pnl.build_snapshot = MagicMock(
        return_value=PnlSnapshot(
            0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 1000.0, datetime.now(timezone.utc)
        )
    )
    bot._storage = MagicMock()
    bot._notifier = None
    # Mock the event-log + persistence collaborators so the proceeds
    # path doesn't touch disk and so we can assert on event names.
    bot._log_event = MagicMock()
    bot._save_persistent_runtime_state_now = MagicMock()
    return bot


def _suppression_logged(bot) -> bool:
    return any(
        len(c.args) >= 2 and c.args[1] == "soft_flatten_suppressed_sub_min_notional"
        for c in bot._log_event.call_args_list
    )


def test_sub_min_notional_entry_suppressed() -> None:
    """Residual well below both floors ($2 notional vs $5.5 local /
    $5.0 venue). SF entry is suppressed BEFORE any state mutation or
    fatigue accounting: soft_flatten stays inactive, the no-op is NOT
    counted toward sf_fatigue, and cancel-all is not spammed."""
    s = _make_settings()
    state = _state(pos_qty=2.0, best_bid=1.0, best_ask=1.0, settings=s)
    bot = _bot(state, s, venue_min_notional_usd=5.0)

    bot._enter_soft_flatten(
        None,
        force_phase=2,
        trigger_reason="toxicity_hard",
        log_message_override="tox hard",
        log_payload_override={},
    )

    assert state.soft_flatten_active is False
    assert state.bot_status == BotStatus.RUNNING
    # The deadlock-defining property: the no-op episode is NOT counted.
    assert event_count_in_window(state.sf_fatigue) == 0
    assert _suppression_logged(bot) is True
    bot._exec.cancel_all_orders_for_symbol.assert_not_called()


def test_above_min_notional_entry_proceeds() -> None:
    """Residual comfortably above both floors ($20 notional) proceeds to
    a real SF entry: soft_flatten activates, the episode IS counted
    toward sf_fatigue, and the suppression event is NOT logged."""
    s = _make_settings()
    state = _state(pos_qty=20.0, best_bid=1.0, best_ask=1.0, settings=s)
    bot = _bot(state, s, venue_min_notional_usd=5.0)

    bot._enter_soft_flatten(
        None,
        force_phase=2,
        trigger_reason="toxicity_hard",
        log_message_override="tox hard",
        log_payload_override={},
    )

    assert state.soft_flatten_active is True
    assert state.bot_status == BotStatus.SOFT_FLATTENING
    assert event_count_in_window(state.sf_fatigue) == 1
    assert _suppression_logged(bot) is False
    bot._exec.cancel_all_orders_for_symbol.assert_called_once()


def test_between_venue_and_local_min_uses_max_floor() -> None:
    """The consistency fix: residual $5.2 is ABOVE the venue floor
    ($5.0) but BELOW the local floor ($5.5). Pre-v1.5.291 the entry
    guard used venue-only and would have let this through (then the
    worker would immediately exit residual_below_min_notional — the
    no-op loop). With max(venue, local) the entry is suppressed."""
    s = _make_settings()
    state = _state(pos_qty=5.2, best_bid=1.0, best_ask=1.0, settings=s)
    bot = _bot(state, s, venue_min_notional_usd=5.0)

    bot._enter_soft_flatten(
        None,
        force_phase=2,
        trigger_reason="toxicity_hard",
        log_message_override="tox hard",
        log_payload_override={},
    )

    assert state.soft_flatten_active is False
    assert event_count_in_window(state.sf_fatigue) == 0
    assert _suppression_logged(bot) is True


def test_drawdown_trigger_also_guarded() -> None:
    """The guard is trigger-agnostic — it sits at the universal SF entry
    chokepoint, so the drawdown-gate path (trigger_reason default) is
    suppressed on a sub-min-notional residual too."""
    s = _make_settings()
    state = _state(pos_qty=2.0, best_bid=1.0, best_ask=1.0, settings=s)
    bot = _bot(state, s, venue_min_notional_usd=5.0)

    # Drawdown path passes a real evaluation object; mock the minimum
    # the guard / downstream reads (guard returns before reading ev).
    ev = MagicMock()
    bot._enter_soft_flatten(ev, trigger_reason="position_drawdown_gate")

    assert state.soft_flatten_active is False
    assert event_count_in_window(state.sf_fatigue) == 0
    assert _suppression_logged(bot) is True
