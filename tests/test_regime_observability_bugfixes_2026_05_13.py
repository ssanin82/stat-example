"""Regression tests for the 2026-05-13 regime-observability bug fixes.

Bugs found in the first v1.3.0 production snapshot (260513-122839-colo)
and fixed before the overnight run:

Bug 1: ``inventory_utilization_before_fill`` was always null on fills.
       Root cause: ``state.settings`` wasn't a public attribute; the
       lookup silently returned 0 and skipped the calculation.
       Fix: added ``settings`` property on ``BotState``.

Bug 2 (multi-field): ~7 exposure-bar fields were always null.
       Root cause: wrong attribute paths on state (e.g. ``state.toxicity_snapshot``
       which doesn't exist — actual is ``state.toxicity``;
       ``state.last_quote_decision`` which doesn't exist — vol/active_sides
       live elsewhere). Plus ``bid_distance_ticks`` / ``ask_distance_ticks``
       were declared but never computed, and post_fill_cooldown /
       at_touch_adverse_pause fields were hardcoded None.
       Fix: corrected attribute paths; compute distance ticks from
       working_bid/ask vs best_bid/ask; added a state-level shared
       dict (``observability_gate_flags``) that the bot pushes per
       tick for the OrderManager-owned gate states.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.enums import OrderStatus, Side
from app.exposure_bar_emitter import ExposureBarEmitter
from app.models import BestBidAsk, WorkingOrder
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _make_settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "",
        "HL_ACCOUNT_ADDRESS": "",
        "SYMBOL": "TON-USDT-SWAP",
        "MAX_ABS_POSITION": 8.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _make_storage() -> Storage:
    path = (
        Path(tempfile.gettempdir())
        / f"mm_bf_{os.getpid()}_{uuid.uuid4().hex}.db"
    )
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        }
    )
    s = Storage(settings)
    s.init_schema()
    return s


# --------------------------------------------------------------------- #
# Bug 1: state.settings property                                         #
# --------------------------------------------------------------------- #


def test_bug1_state_exposes_settings_property() -> None:
    """The whole bug was that ``state.settings`` returned an attribute-
    miss → 0.0 fallback. Now it should return the actual Settings."""
    settings = _make_settings(MAX_ABS_POSITION=12.0)
    state = BotState(settings)
    assert state.settings is settings
    assert state.settings.max_abs_position == 12.0


def test_bug1_state_settings_is_read_only() -> None:
    """Sanity check: the public ``settings`` accessor is a read-only
    property, not a settable attribute. We don't want code to
    accidentally swap the bot's Settings mid-session."""
    settings = _make_settings()
    state = BotState(settings)
    try:
        state.settings = _make_settings(MAX_ABS_POSITION=999.0)
    except AttributeError:
        return  # Expected: properties without setters raise AttributeError
    # If we reach here, the assignment succeeded — bug.
    assert False, "state.settings should be read-only"


# --------------------------------------------------------------------- #
# Bug 2a: exposure-bar attribute paths                                   #
# --------------------------------------------------------------------- #


def test_bug2a_capture_bar_reads_state_toxicity() -> None:
    """Pre-fix: used ``state.toxicity_snapshot`` (doesn't exist) →
    always None. Fix: use ``state.toxicity``."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=99.9, best_ask=100.1, mid_price=100.0,
        spread_bps=20.0, bid_size=100.0, ask_size=80.0,
    )
    # state.toxicity exists by default; verify its score is readable.
    state.toxicity.score = 0.42
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    row = em._capture_bar()
    assert row is not None
    assert row["toxicity_score"] == 0.42


def test_bug2a_capture_bar_reads_vol_bps() -> None:
    """Pre-fix: tried ``state.last_quote_decision.vol_estimate``
    (doesn't exist). Fix: use ``state.vol_bps``."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=99.9, best_ask=100.1, mid_price=100.0,
        spread_bps=20.0,
    )
    state.vol_bps = 15.3
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    row = em._capture_bar()
    assert row is not None
    assert row["vol_estimate"] == 15.3


def test_bug2a_capture_bar_reads_last_active_sides() -> None:
    """Pre-fix: tried ``state.last_quote_decision.active_sides``
    (doesn't exist). Fix: use ``state.last_active_sides``."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=99.9, best_ask=100.1, mid_price=100.0,
        spread_bps=20.0,
    )
    state.last_active_sides = "BID_ONLY"
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    row = em._capture_bar()
    assert row is not None
    assert row["active_sides"] == "BID_ONLY"


def test_bug2a_capture_bar_derives_adaptive_widen_active() -> None:
    """Pre-fix: tried ``state.adaptive_widen_active`` (doesn't exist
    as a bool). Fix: derive from ``state.adaptive_spread_widen_until_mono``
    deadline vs ``time.monotonic()``."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=99.9, best_ask=100.1, mid_price=100.0,
        spread_bps=20.0,
    )
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    # Inactive case: deadline is 0 (never set) → not active.
    state.adaptive_spread_widen_until_mono = 0.0
    row = em._capture_bar()
    assert row["adaptive_widen_active"] == 0
    # Active case: deadline is in the future.
    state.adaptive_spread_widen_until_mono = time.monotonic() + 60.0
    row = em._capture_bar()
    assert row["adaptive_widen_active"] == 1


def test_bug2a_capture_bar_derives_recovery_cooldown_active() -> None:
    """Pre-fix: tried ``state.quote_eligibility_recovery.active``
    (doesn't exist). Fix: use
    ``state.quote_elig_recovery_remaining_ms > 0``."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=99.9, best_ask=100.1, mid_price=100.0,
        spread_bps=20.0,
    )
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    # Inactive.
    state.quote_elig_recovery_remaining_ms = 0.0
    row = em._capture_bar()
    assert row["recovery_cooldown_active"] == 0
    # Active.
    state.quote_elig_recovery_remaining_ms = 250.0
    row = em._capture_bar()
    assert row["recovery_cooldown_active"] == 1


# --------------------------------------------------------------------- #
# Bug 2b: bid/ask_distance_ticks computation                             #
# --------------------------------------------------------------------- #


class _MockSymbolSpec:
    def __init__(self, price_tick: float) -> None:
        self.price_tick = price_tick


def test_bug2b_distance_ticks_computed_correctly() -> None:
    """Pre-fix: these were declared but never assigned → always None.
    Fix: compute from working_bid/ask price vs best_bid/ask using
    symbol_spec.price_tick."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X",
        best_bid=100.00,
        best_ask=100.05,
        mid_price=100.025,
        spread_bps=5.0,
    )
    # Attach a symbol_spec the emitter can find.
    state.symbol_spec = _MockSymbolSpec(price_tick=0.01)  # type: ignore[attr-defined]
    # Place working orders 2 ticks behind touch.
    state.working_bid = WorkingOrder(
        order_id_local="wb", order_id_exchange=1, client_order_id="c",
        symbol="X", side=Side.BUY, price=99.98, size=1.0, post_only=True,
        status=OrderStatus.ACKED,
    )
    state.working_ask = WorkingOrder(
        order_id_local="wa", order_id_exchange=2, client_order_id="c",
        symbol="X", side=Side.SELL, price=100.07, size=1.0, post_only=True,
        status=OrderStatus.ACKED,
    )
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    row = em._capture_bar()
    assert row is not None
    # bid_distance_ticks = (100.00 - 99.98) / 0.01 = 2 ticks
    assert abs(row["bid_distance_ticks"] - 2.0) < 1e-6
    # ask_distance_ticks = (100.07 - 100.05) / 0.01 = 2 ticks
    assert abs(row["ask_distance_ticks"] - 2.0) < 1e-6
    # And bid/ask_live should also be 1.
    assert row["bid_live"] == 1
    assert row["ask_live"] == 1


def test_bug2b_distance_ticks_null_when_no_symbol_spec() -> None:
    """Defensive: when symbol_spec isn't available, fields are None
    (not 0 or wrong, just absent)."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=100.0, best_ask=100.05, mid_price=100.025,
        spread_bps=5.0,
    )
    # No state.symbol_spec attached.
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    row = em._capture_bar()
    assert row["bid_distance_ticks"] is None
    assert row["ask_distance_ticks"] is None


# --------------------------------------------------------------------- #
# Bug 2c: post_fill_cooldown / at_touch_adverse_pause gate flags          #
# --------------------------------------------------------------------- #


def test_bug2c_observability_gate_flags_round_trip() -> None:
    """Pre-fix: hardcoded None in capture_bar. Fix: state exposes a
    ``observability_gate_flags`` dict the bot pushes per tick; the
    emitter reads it lock-free."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=99.9, best_ask=100.1, mid_price=100.0,
        spread_bps=20.0,
    )
    state.set_observability_gate_flags(
        {
            "post_fill_cooldown_bid": True,
            "post_fill_cooldown_ask": False,
            "at_touch_adverse_pause_bid": False,
            "at_touch_adverse_pause_ask": True,
        }
    )
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    row = em._capture_bar()
    assert row["post_fill_cooldown_active_bid"] == 1
    assert row["post_fill_cooldown_active_ask"] == 0
    assert row["at_touch_adverse_pause_bid"] == 0
    assert row["at_touch_adverse_pause_ask"] == 1


def test_bug2c_observability_gate_flags_empty_returns_null() -> None:
    """Initial state (no push from bot) → fields should be NULL,
    not 0 or 1. Distinguishes 'gate inactive' from 'gate state
    unknown'."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=99.9, best_ask=100.1, mid_price=100.0,
        spread_bps=20.0,
    )
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    row = em._capture_bar()
    assert row["post_fill_cooldown_active_bid"] is None
    assert row["post_fill_cooldown_active_ask"] is None
    assert row["at_touch_adverse_pause_bid"] is None
    assert row["at_touch_adverse_pause_ask"] is None


# --------------------------------------------------------------------- #
# Bug 3: OKX-style QuoteEligibility values not recognized                #
# --------------------------------------------------------------------- #


def test_bug3_capture_bar_recognizes_okx_buy_only() -> None:
    """OKX bot emits ``QUOTE_BUY_ONLY`` / ``QUOTE_SELL_ONLY`` from the
    QuoteEligibility enum. Pre-fix: the emitter's hold_all_active
    fall-through only knew BID_ONLY / ASK_ONLY style → 35% of bars
    had ``hold_all_active = null``. Fix: add the BUY/SELL variants."""
    settings = _make_settings()
    state = BotState(settings)
    state.market = BestBidAsk(
        symbol="X", best_bid=99.9, best_ask=100.1, mid_price=100.0,
        spread_bps=20.0,
    )
    storage = _make_storage()
    em = ExposureBarEmitter(settings, state, storage)
    for elig in (
        "QUOTE_BUY_ONLY",
        "QUOTE_SELL_ONLY",
        "QUOTE_BID_ONLY",
        "QUOTE_ASK_ONLY",
        "QUOTE_BOTH",
        "BID_ONLY",
        "ASK_ONLY",
    ):
        state.set_quote_eligibility_snapshot_dict(
            {"quote_eligibility_state": elig}
        )
        row = em._capture_bar()
        assert row["hold_all_active"] == 0, (
            f"hold_all_active should be 0 for elig={elig}, got {row['hold_all_active']}"
        )
    # And HOLD_ALL → 1.
    state.set_quote_eligibility_snapshot_dict(
        {"quote_eligibility_state": "HOLD_ALL"}
    )
    row = em._capture_bar()
    assert row["hold_all_active"] == 1


def test_bug3_mode_bucket_handles_okx_naming() -> None:
    """Same root cause in ``scripts/regime_summary.py::mode_bucket`` —
    OKX BUY/SELL names weren't mapped to BID/ASK buckets."""
    import sys
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "scripts")
    )
    from regime_summary import mode_bucket as _mode_bucket

    assert _mode_bucket(None, "QUOTE_BUY_ONLY") == "BID_ONLY"
    assert _mode_bucket(None, "QUOTE_SELL_ONLY") == "ASK_ONLY"
    assert _mode_bucket(None, "QUOTE_BID_ONLY") == "BID_ONLY"
    assert _mode_bucket(None, "QUOTE_ASK_ONLY") == "ASK_ONLY"
    assert _mode_bucket(None, "QUOTE_BOTH") == "BOTH"
    assert _mode_bucket(None, "HOLD_ALL") == "HOLD_ALL"
