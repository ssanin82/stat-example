"""Tests for quote-cycle suppression-reason observability.

Covers:
- ``QuoteQualityRollup.record_suppression`` counter plumbing and ``to_dict`` surface.
- ``OrderManager._collect_quote_cycle_suppression_reasons`` classifying reasons from
  the three upstream sources (``decision.decision_reason``,
  ``decision.quote_eligibility_reason``, and engine telemetry).
- Event-emission rate limit: a persistent reason emits one event (on appearance),
  not one per tick. A newly-appearing reason triggers an event. A cleared reason
  does not re-emit until it reappears.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

from app.enums import ActiveSides
from app.exchange.symbol_spec import symbol_spec_from_hyperliquid_meta
from app.execution import OrderManager
from app.models import QuoteDecision
from app.quote_quality_telemetry import QuoteQualityRollup
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _eth_spec():
    meta = {
        "universe": [
            {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
        ],
    }
    return symbol_spec_from_hyperliquid_meta(meta, "ETH")


def _make_om() -> tuple[OrderManager, BotState, Storage]:
    path = Path(tempfile.gettempdir()) / f"mm_supp_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 1.0,
            "QUOTE_NOTIONAL_USD": 500.0,
            "QUOTE_QUALITY_WINDOW_SAMPLES": 100,
        }
    )
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.symbol_spec = _eth_spec()
    om = OrderManager(settings, client, storage, state)
    return om, state, storage


def _decision(
    *,
    decision_reason: str = "baseline",
    quote_eligibility_reason: str | None = None,
    active_sides: ActiveSides = ActiveSides.BOTH,
) -> QuoteDecision:
    return QuoteDecision(
        ts=utc_now(),
        symbol="ETH_USDT_Perp",
        mid_price=2345.0,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=2345.0,
        target_spread_bps=3.0,
        target_bid=2344.5,
        target_ask=2345.5,
        quoted_bid=2344.5,
        quoted_ask=2345.5,
        quoted_bid_sz=0.02,
        quoted_ask_sz=0.02,
        active_sides=active_sides,
        toxicity_score=0.0,
        decision_reason=decision_reason,
        quote_cycle_id=uuid.uuid4().hex,
        quote_eligibility_reason=quote_eligibility_reason,
    )


# ---------- QuoteQualityRollup.record_suppression ----------

def test_record_suppression_increments_monotonically() -> None:
    r = QuoteQualityRollup(window_samples=10)
    r.record_suppression("quoting:soft_skew_long")
    r.record_suppression("quoting:soft_skew_long")
    r.record_suppression("engine:inventory_exec_bias_bid")
    d = r.to_dict(realized_pnl_usd=0.0, fills_for_markout=[], markout_window=50)
    counts = d["suppression_reason_counts_session"]
    assert counts["quoting:soft_skew_long"] == 2
    assert counts["engine:inventory_exec_bias_bid"] == 1


def test_record_suppression_ignores_empty_reason() -> None:
    r = QuoteQualityRollup(window_samples=10)
    r.record_suppression("")
    d = r.to_dict(realized_pnl_usd=0.0, fills_for_markout=[], markout_window=50)
    assert d["suppression_reason_counts_session"] == {}


def test_to_dict_always_exposes_suppression_key_even_when_empty() -> None:
    r = QuoteQualityRollup(window_samples=10)
    d = r.to_dict(realized_pnl_usd=0.0, fills_for_markout=[], markout_window=50)
    assert "suppression_reason_counts_session" in d
    assert d["suppression_reason_counts_session"] == {}


# ---------- _collect_quote_cycle_suppression_reasons ----------

def test_collect_classifies_inventory_skew_from_decision_reason() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {}
    d = _decision(decision_reason="baseline,soft_skew_short", active_sides=ActiveSides.BID_ONLY)
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert "quoting:soft_skew_short" in reasons
    assert "quoting:soft_skew_long" not in reasons


def test_collect_classifies_toxicity_suppression() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {}
    d = _decision(decision_reason="baseline,toxic_bid", active_sides=ActiveSides.ASK_ONLY)
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert "quoting:toxic_bid" in reasons


def test_collect_classifies_freshness_hold() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {}
    d = _decision(quote_eligibility_reason="freshness_hold:gap_p95_ms>3000")
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert "eligibility:freshness_hold" in reasons


def test_collect_classifies_freshness_one_sided() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {}
    d = _decision(quote_eligibility_reason="freshness_one_sided:book_age_ms>500")
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert "eligibility:freshness_one_sided" in reasons


def test_collect_classifies_drift_one_sided() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {}
    d = _decision(quote_eligibility_reason="drift:mid_return_100ms_bps>=1.500")
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert "eligibility:drift_one_sided" in reasons


def test_collect_classifies_jump_hold() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {}
    d = _decision(quote_eligibility_reason="jump_250ms_bps>=10.000")
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert "eligibility:jump_hold" in reasons


def test_collect_classifies_inventory_exec_bias_from_engine_telemetry() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {
        "quote_engine_inventory_bias_suppressed_bid": True,
        "quote_engine_inventory_bias_suppressed_ask": False,
    }
    d = _decision()
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert "engine:inventory_exec_bias_bid" in reasons
    assert "engine:inventory_exec_bias_ask" not in reasons


def test_collect_ignores_baseline_and_non_suppression_parts() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {}
    d = _decision(decision_reason="baseline,adaptive_spread_widen")
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert reasons == frozenset()


def test_collect_multiple_simultaneous_reasons() -> None:
    om, _, _ = _make_om()
    om._quote_exec_telemetry = {"quote_engine_inventory_bias_suppressed_bid": True}
    d = _decision(
        decision_reason="baseline,soft_skew_short",
        quote_eligibility_reason="freshness_one_sided:book_age_ms>500",
    )
    reasons = om._collect_quote_cycle_suppression_reasons(d)
    assert reasons == frozenset(
        {
            "quoting:soft_skew_short",
            "eligibility:freshness_one_sided",
            "engine:inventory_exec_bias_bid",
        }
    )


# ---------- Event-emission rate limit via _record_quote_cycle_telemetry ----------

def _count_suppression_events(storage: Storage) -> int:
    rows = storage.recent_bot_events(limit=500)
    return sum(1 for r in rows if r.get("event_type") == "quote_side_suppressed")


def test_persistent_reason_emits_single_event_across_many_ticks() -> None:
    om, state, storage = _make_om()
    om._quote_exec_telemetry = {"quote_engine_inventory_bias_suppressed_bid": True}
    d = _decision()
    # Suppressor on for 5 ticks in a row — one event total.
    for _ in range(5):
        om._record_quote_cycle_telemetry(
            d, want_bid=True, want_ask=True, can_bid=False, can_ask=True, tick=0.01
        )
    assert _count_suppression_events(storage) == 1
    counts = state.quote_quality_dict()["suppression_reason_counts_session"]
    assert counts["engine:inventory_exec_bias_bid"] == 5


def test_new_reason_appearing_later_emits_new_event() -> None:
    om, state, storage = _make_om()
    om._quote_exec_telemetry = {"quote_engine_inventory_bias_suppressed_bid": True}
    d1 = _decision()
    om._record_quote_cycle_telemetry(
        d1, want_bid=True, want_ask=True, can_bid=False, can_ask=True, tick=0.01
    )
    # Now a second reason appears alongside the first — emit exactly one new event.
    d2 = _decision(quote_eligibility_reason="freshness_one_sided:book_age_ms>500")
    om._record_quote_cycle_telemetry(
        d2, want_bid=True, want_ask=True, can_bid=False, can_ask=True, tick=0.01
    )
    assert _count_suppression_events(storage) == 2


def test_cleared_reason_reappearing_emits_again() -> None:
    om, state, storage = _make_om()
    # Cycle 1: reason active.
    om._quote_exec_telemetry = {"quote_engine_inventory_bias_suppressed_bid": True}
    om._record_quote_cycle_telemetry(
        _decision(), want_bid=True, want_ask=True, can_bid=False, can_ask=True, tick=0.01
    )
    # Cycle 2: reason cleared.
    om._quote_exec_telemetry = {}
    om._record_quote_cycle_telemetry(
        _decision(), want_bid=True, want_ask=True, can_bid=True, can_ask=True, tick=0.01
    )
    # Cycle 3: reason reappears — should emit again (edge-triggered).
    om._quote_exec_telemetry = {"quote_engine_inventory_bias_suppressed_bid": True}
    om._record_quote_cycle_telemetry(
        _decision(), want_bid=True, want_ask=True, can_bid=False, can_ask=True, tick=0.01
    )
    assert _count_suppression_events(storage) == 2


def test_no_suppression_emits_no_events() -> None:
    om, _, storage = _make_om()
    om._quote_exec_telemetry = {}
    for _ in range(3):
        om._record_quote_cycle_telemetry(
            _decision(), want_bid=True, want_ask=True, can_bid=True, can_ask=True, tick=0.01
        )
    assert _count_suppression_events(storage) == 0
