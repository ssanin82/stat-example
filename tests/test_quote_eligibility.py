"""Quote eligibility: freshness, drift merge, apply to QuoteDecision."""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from app.enums import ActiveSides, QuoteEligibility
from app.models import QuoteDecision, ToxicitySnapshot
from app.quote_eligibility import (
    compute_quote_eligibility,
    merge_eligibility_freshness_drift,
    more_restrictive,
)
from app.quoting import apply_quote_eligibility_to_decision
from app.utils.time import utc_now
from tests.settings_helpers import UnitTestSettings


def _settings(**kw: object) -> UnitTestSettings:
    path = Path(tempfile.gettempdir()) / f"mm_qe_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "QUOTE_ELIGIBILITY_ENABLED": True,
        "QUOTE_HOLD_MAX_BOOK_AGE_MS": 500.0,
        "QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS": 200.0,
        "QUOTE_HOLD_MAX_GAP_P95_MS": 600.0,
        "QUOTE_ONE_SIDED_MAX_GAP_P95_MS": 350.0,
        "DRIFT_BLOCK_100MS_BPS": 12.0,
        "DRIFT_BLOCK_250MS_BPS": 22.0,
        "QUOTE_MID_HISTORY_MAX_SAMPLES": 256,
        "QUOTE_MID_HISTORY_MAX_AGE_MS": 2000.0,
    }
    for k, v in kw.items():
        data[k.upper() if k.islower() else k] = v
    return UnitTestSettings.model_validate(data)


def test_merge_conflict_one_sided() -> None:
    m, _ = merge_eligibility_freshness_drift(
        QuoteEligibility.QUOTE_BUY_ONLY,
        QuoteEligibility.QUOTE_SELL_ONLY,
    )
    assert m == QuoteEligibility.HOLD_ALL


def test_more_restrictive_order() -> None:
    assert more_restrictive(QuoteEligibility.QUOTE_BOTH, QuoteEligibility.QUOTE_BUY_ONLY) == (
        QuoteEligibility.QUOTE_BUY_ONLY
    )
    assert more_restrictive(QuoteEligibility.QUOTE_BUY_ONLY, QuoteEligibility.HOLD_ALL) == (
        QuoteEligibility.HOLD_ALL
    )


def test_freshness_hold_when_no_age_no_gap() -> None:
    s = _settings()
    r = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=3000.0,
        now_mono=time.monotonic(),
        mid_samples=(),
        seconds_since_public_bbo=None,
        gap_median_ms=None,
        gap_p95_ms=None,
        effective_staleness_ms=None,
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL


def test_apply_eligibility_zeros_both_sides_on_hold() -> None:
    d = QuoteDecision(
        ts=utc_now(),
        symbol="ETH",
        mid_price=100.0,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=100.0,
        target_spread_bps=10.0,
        target_bid=99.5,
        target_ask=100.5,
        quoted_bid=99.5,
        quoted_ask=100.5,
        quoted_bid_sz=0.02,
        quoted_ask_sz=0.02,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="test",
        quote_cycle_id="q1",
    )
    out = apply_quote_eligibility_to_decision(
        d,
        QuoteEligibility.HOLD_ALL,
        eligibility_reason="unit",
    )
    assert out.active_sides == ActiveSides.NONE
    assert out.quoted_bid_sz == 0.0 and out.quoted_ask_sz == 0.0
    assert out.quote_eligibility == QuoteEligibility.HOLD_ALL.value


def test_drift_upward_prefers_buy_only() -> None:
    s = _settings()
    now = time.monotonic()
    # Upward drift over 100/250ms above DRIFT_BLOCK_* but below JUMP_HOLD_250MS_BPS (default 75).
    # 3000 -> 3015 is 50 bps: one-sided BUY_ONLY, not jump HOLD.
    samples = [
        (now - 0.30, 3000.0),
        (now - 0.26, 3000.0),
        (now, 3015.0),
    ]
    r = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=3015.0,
        now_mono=now,
        mid_samples=samples,
        seconds_since_public_bbo=0.01,
        gap_median_ms=50.0,
        gap_p95_ms=80.0,
        effective_staleness_ms=None,
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BUY_ONLY
    assert r.mid_return_250ms_bps is not None


def test_order_uncertainty_forces_hold() -> None:
    s = _settings()
    now = time.monotonic()
    samples = [(now - 0.05, 100.0), (now, 100.0)]
    r = compute_quote_eligibility(
        s,
        order_state_uncertainty=True,
        mid_now=100.0,
        now_mono=now,
        mid_samples=samples,
        seconds_since_public_bbo=0.02,
        gap_median_ms=10.0,
        gap_p95_ms=15.0,
        effective_staleness_ms=None,
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL
    assert "order_state_uncertainty" in r.reason
