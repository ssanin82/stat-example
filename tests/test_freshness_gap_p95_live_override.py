"""Live-book-fresh override for the p95 freshness gate.

Context: ``tmp/snap_20260417_193813``. Session bot ran for 80 s. After the
cloid-escalation cleared the stuck CANCEL_PENDING at t=28 s, the bot was
held in HOLD_ALL for the remaining 49 s with reason
``freshness_hold:gap_p95_ms>1000.0`` — while ``book_age_seconds=0.0003``
(sub-ms fresh) and ``last_gap_ms=210``. One single ~1.9 s WS gap mid-session
pushed the rolling p95 above 1000 ms and the ring buffer kept it pinned.

The override: when book age is clearly live AND the median gap is healthy,
the p95 is stale statistics about the past — not a reason to keep quoting
paused. Invariants:

- p95 breach + fresh live signal → QUOTE_BOTH (override reason set)
- p95 breach + stale live signal → HOLD_ALL (do NOT override)
- p95 breach + high median → HOLD_ALL (median says feed is actually sick)
- age breach (independent of p95) → HOLD_ALL (override never fires for age)
- override disabled via setting → legacy HOLD_ALL behaviour
"""

from __future__ import annotations

from app.enums import QuoteEligibility
from app.quote_eligibility import compute_quote_eligibility
from tests.settings_helpers import UnitTestSettings


def _s(**overrides) -> UnitTestSettings:
    # Use the live-override defaults unless overridden.
    base = {"TRADING_ENABLED": False}
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _kwargs(**overrides) -> dict:
    base = dict(
        order_state_uncertainty=False,
        mid_now=100.0,
        now_mono=1000.0,
        mid_samples=[(999.5, 100.0), (999.0, 100.0)],
        seconds_since_public_bbo=0.05,
        gap_median_ms=50.0,
        gap_p95_ms=150.0,
        effective_staleness_ms=60.0,
    )
    base.update(overrides)
    return base


def test_p95_breach_with_live_fresh_book_overrides_to_quote_both() -> None:
    """The 183813 scenario: p95 elevated by an old outlier, current book is live."""
    r = compute_quote_eligibility(
        _s(QUOTE_HOLD_MAX_GAP_P95_MS=1000.0),
        **_kwargs(
            seconds_since_public_bbo=0.001,  # 1 ms — sub-ms fresh
            gap_median_ms=290.0,              # median healthy
            gap_p95_ms=1094.0,                 # p95 above threshold
        ),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH
    assert "p95_override_live" in r.reason


def test_p95_breach_with_stale_book_does_not_override() -> None:
    """Current feed is NOT live — override must not fire."""
    r = compute_quote_eligibility(
        _s(QUOTE_HOLD_MAX_GAP_P95_MS=1000.0),
        **_kwargs(
            seconds_since_public_bbo=5.0,   # 5 s — stale
            gap_median_ms=290.0,
            gap_p95_ms=1094.0,
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL
    assert "freshness_hold" in r.reason


def test_p95_breach_with_high_median_does_not_override() -> None:
    """Median is elevated → feed is systemically sick, not just one outlier."""
    r = compute_quote_eligibility(
        _s(QUOTE_HOLD_MAX_GAP_P95_MS=1000.0),
        **_kwargs(
            seconds_since_public_bbo=0.001,
            gap_median_ms=800.0,     # median above override threshold
            gap_p95_ms=1094.0,
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL


def test_p95_breach_with_missing_median_does_not_override() -> None:
    """No median data → we cannot vouch for feed health → do not override."""
    r = compute_quote_eligibility(
        _s(QUOTE_HOLD_MAX_GAP_P95_MS=1000.0),
        **_kwargs(
            seconds_since_public_bbo=0.001,
            gap_median_ms=None,
            gap_p95_ms=1094.0,
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL


def test_age_breach_never_overridden() -> None:
    """Age-based hold means the feed is stale RIGHT NOW. Override must NOT fire."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_BOOK_AGE_MS=500.0,
            QUOTE_HOLD_MAX_GAP_P95_MS=10_000.0,  # p95 inert for this test
        ),
        **_kwargs(
            seconds_since_public_bbo=2.0,   # 2000 ms — breaches age hold
            gap_median_ms=50.0,
            gap_p95_ms=100.0,
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL


def test_p95_breach_with_one_sided_gate_also_gets_overridden() -> None:
    """Override lifts the p95 one-sided cap too when conditions hold."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_ONE_SIDED_MAX_GAP_P95_MS=500.0,
            QUOTE_HOLD_MAX_GAP_P95_MS=10_000.0,
        ),
        **_kwargs(
            seconds_since_public_bbo=0.001,
            gap_median_ms=200.0,
            gap_p95_ms=700.0,   # above one-sided threshold, below hold
        ),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH


def test_override_disabled_via_setting_restores_legacy_behaviour() -> None:
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_ENABLED=False,
        ),
        **_kwargs(
            seconds_since_public_bbo=0.001,
            gap_median_ms=290.0,
            gap_p95_ms=1094.0,
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL
    assert "freshness_hold" in r.reason


def test_override_does_not_change_healthy_behaviour() -> None:
    """When nothing would hold anyway, override is a no-op → plain QUOTE_BOTH."""
    r = compute_quote_eligibility(
        _s(QUOTE_HOLD_MAX_GAP_P95_MS=1000.0),
        **_kwargs(
            seconds_since_public_bbo=0.001,
            gap_median_ms=50.0,
            gap_p95_ms=150.0,
        ),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH
    # The plain ok reason should be present (no override suffix since
    # override wasn't needed).
    assert r.reason.endswith("freshness_ok") or "freshness_ok|" in r.reason


def test_observed_session_regression() -> None:
    """Direct repro: the snap_20260417_193813 numbers must resolve to QUOTE_BOTH."""
    r = compute_quote_eligibility(
        # Mirrors config/profiles/prod.grvt.env active during that session.
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_ONE_SIDED_MAX_GAP_P95_MS=750.0,
            QUOTE_HOLD_MAX_BOOK_AGE_MS=1200.0,
            QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS=600.0,
        ),
        **_kwargs(
            # From state_current.json at the moment of snapshot:
            seconds_since_public_bbo=0.000336,  # 0.336 ms — book is live
            # From market-data_gap-stats.json:
            gap_median_ms=290.082,
            gap_p95_ms=1094.468,
            effective_staleness_ms=48.41,
        ),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH
    assert "p95_override_live" in r.reason
