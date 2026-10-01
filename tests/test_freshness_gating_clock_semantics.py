"""
Invariant: _freshness_eligibility uses the LOCAL RECEIPT clock (age_ms) for gating
by default. effective_staleness_ms (= wall_now - exchange_timestamp) is a monitoring
metric only and MUST NOT influence gate thresholds unless the operator explicitly
opts in via QUOTE_FRESHNESS_USE_EXCHANGE_STALENESS_FOR_GATING.

Regression bug: before the fix, _freshness_eligibility compared `max(age_ms,
eff_staleness_ms)` to the thresholds. On hosts with ~220 ms one-way network
delay, every freshly-received book tick looked ~220 ms "stale" and tripped the
QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS threshold (250 ms default at the time), causing
the bot to spend 89.5% of cycles in one-sided BUY-only mode — a deployment-
geography artifact, not a market-data quality signal. (See forensic analysis
of snap_20260416_154159.)
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path

from app.enums import QuoteEligibility
from app.quote_eligibility import compute_quote_eligibility
from tests.settings_helpers import UnitTestSettings


def _settings(**kw: object) -> UnitTestSettings:
    path = Path(tempfile.gettempdir()) / f"mm_fresh_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    data: dict = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "QUOTE_ELIGIBILITY_ENABLED": True,
        # Tight thresholds so any accidental inclusion of eff_staleness would fail.
        "QUOTE_HOLD_MAX_BOOK_AGE_MS": 500.0,
        "QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS": 250.0,
        "QUOTE_HOLD_MAX_GAP_P95_MS": 600.0,
        "QUOTE_ONE_SIDED_MAX_GAP_P95_MS": 350.0,
    }
    for k, v in kw.items():
        data[k.upper() if k.islower() else k] = v
    return UnitTestSettings.model_validate(data)


def _fresh_samples(now: float) -> list[tuple[float, float]]:
    # Three stable mid samples across >=300ms — keeps drift/jump out of the picture.
    return [
        (now - 0.30, 3000.0),
        (now - 0.25, 3000.0),
        (now - 0.10, 3000.0),
        (now, 3000.0),
    ]


def test_default_gating_ignores_exchange_staleness() -> None:
    """Fresh local receipt + huge eff_staleness_ms → QUOTE_BOTH (gate only looks at age_ms)."""
    s = _settings()
    now = time.monotonic()
    r = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=3000.0,
        now_mono=now,
        mid_samples=_fresh_samples(now),
        seconds_since_public_bbo=0.010,  # age_ms=10 — well below 250/500 thresholds.
        gap_median_ms=50.0,
        gap_p95_ms=80.0,
        effective_staleness_ms=5000.0,  # huge network-delay signal; must be ignored.
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH, (
        f"default gate must ignore eff_staleness; got {r.eligibility} reason={r.reason!r}"
    )
    # Monitoring field still flows through to the result for observability.
    assert r.effective_staleness_ms == 5000.0


def test_default_gating_still_trips_on_real_local_staleness() -> None:
    """Local receipt age above threshold → gate still trips (reason tag=local_receipt_ms)."""
    s = _settings()
    now = time.monotonic()
    r = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=3000.0,
        now_mono=now,
        mid_samples=_fresh_samples(now),
        seconds_since_public_bbo=0.400,  # age_ms=400 > 250 one-sided; below 500 hold.
        gap_median_ms=50.0,
        gap_p95_ms=80.0,
        effective_staleness_ms=None,
    )
    # One-sided trip (BUY preferred per default).
    assert r.eligibility == QuoteEligibility.QUOTE_BUY_ONLY
    assert "local_receipt_ms" in r.reason, (
        f"gate reason must tag local_receipt_ms clock; got {r.reason!r}"
    )


def test_optin_exchange_staleness_gate_trips_on_network_lag_alone() -> None:
    """Legacy behavior opt-in: eff_staleness_ms enters the gate via max()."""
    s = _settings(QUOTE_FRESHNESS_USE_EXCHANGE_STALENESS_FOR_GATING=True)
    now = time.monotonic()
    r = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=3000.0,
        now_mono=now,
        mid_samples=_fresh_samples(now),
        seconds_since_public_bbo=0.010,  # local age 10ms — would pass default gate.
        gap_median_ms=50.0,
        gap_p95_ms=80.0,
        effective_staleness_ms=5000.0,  # huge — > HOLD threshold 500 ms.
    )
    # Under opt-in legacy, eff_staleness of 5000ms exceeds HOLD age threshold 500ms → HOLD_ALL.
    assert r.eligibility == QuoteEligibility.HOLD_ALL
    assert "receipt_or_eff_staleness_ms" in r.reason, (
        f"opt-in gate must tag receipt_or_eff_staleness_ms clock; got {r.reason!r}"
    )


def test_default_reason_tag_is_local_receipt_only() -> None:
    """Guard against regressions: the gate reason tag must never mention eff_staleness by default."""
    s = _settings()
    now = time.monotonic()
    r = compute_quote_eligibility(
        s,
        order_state_uncertainty=False,
        mid_now=3000.0,
        now_mono=now,
        mid_samples=_fresh_samples(now),
        seconds_since_public_bbo=0.600,  # age_ms=600 > hold → HOLD_ALL.
        gap_median_ms=50.0,
        gap_p95_ms=80.0,
        effective_staleness_ms=123.0,  # small but non-zero monitoring value.
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL
    # In default mode, tag is "local_receipt_ms"; it must NOT contain "eff_staleness".
    assert "local_receipt_ms" in r.reason
    assert "receipt_or_eff_staleness_ms" not in r.reason
