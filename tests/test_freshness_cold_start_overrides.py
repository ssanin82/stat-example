"""Cold-start / ring-buffer-polluted overrides for the p95 freshness gate.

Two defenses against HOLD_ALL lingering past the point where the feed is
actually fine:

A. ``last_gap_ms`` fallback — if the most recent inter-update gap is healthy,
   allow the override to fire even when the rolling median is still polluted
   by an old outlier.

B. Warmup skip — when the gap-ring has fewer than ``QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES``
   samples, the p95 statistic is unreliable; skip the p95-hold gate entirely.
   Age-based holds remain active.

Both fire only on the p95 portion of the gates. Age-based gating is never
overridden — book age is a current-state read, not a rolling statistic.
Regression target: ``tmp/snap_20260418_132744`` — 66 s after a Railway redeploy,
a single 3.9 s startup outlier drove p95 to 1.8 s and held the bot in HOLD_ALL
for minutes despite median/last-gap both being healthy on the current feed.
"""

from __future__ import annotations

from app.enums import QuoteEligibility
from app.quote_eligibility import compute_quote_eligibility
from tests.settings_helpers import UnitTestSettings


def _s(**overrides) -> UnitTestSettings:
    base = {"TRADING_ENABLED": False}
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _kwargs(**overrides) -> dict:
    base = dict(
        order_state_uncertainty=False,
        mid_now=100.0,
        now_mono=1000.0,
        mid_samples=[(999.5, 100.0), (999.0, 100.0)],
        seconds_since_public_bbo=0.05,   # 50 ms — fresh
        gap_median_ms=50.0,
        gap_p95_ms=150.0,
        effective_staleness_ms=60.0,
    )
    base.update(overrides)
    return base


# ------------------- Fix A: last_gap_ms fallback ------------------------


def test_override_fires_on_healthy_last_gap_when_median_is_polluted() -> None:
    """The 132744 scenario: median is polluted by a startup outlier (748 ms)
    but last_gap_ms is fine (416 ms). Override should fire via the last-gap
    path even when median exceeds its threshold."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_BOOK_AGE_MS=150.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_GAP_MEDIAN_MS=500.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS=500.0,
        ),
        **_kwargs(
            seconds_since_public_bbo=0.050,   # 50 ms age
            gap_median_ms=748.0,              # polluted by old outlier
            gap_last_ms=416.0,                # current feed is FINE
            gap_p95_ms=1813.0,                # p95 breaches hold
            gap_sample_count=74,              # enough samples that warmup-skip doesn't apply — use sample>=200? No, for A's test, it's the last_gap path we want to trigger. Use count > min_samples default (200) so warmup doesn't mask the test.
        ),
    )
    # Override should fire via last_gap path → QUOTE_BOTH.
    # But wait: sample_count=74 < default 200, so warmup-skip ALSO applies.
    # That's fine — either path fires. Assert QUOTE_BOTH.
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH


def test_override_fires_on_last_gap_only_when_no_warmup_mask() -> None:
    """Explicit test that last_gap path is the thing firing, not warmup."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_BOOK_AGE_MS=150.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_GAP_MEDIAN_MS=500.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS=500.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=100,  # low enough to avoid warmup masking
        ),
        **_kwargs(
            seconds_since_public_bbo=0.050,
            gap_median_ms=748.0,
            gap_last_ms=416.0,
            gap_p95_ms=1813.0,
            gap_sample_count=500,  # well past warmup
        ),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH
    assert "p95_override_live" in r.reason


def test_override_does_not_fire_when_both_median_and_last_gap_are_high() -> None:
    """If both median AND last-gap are unhealthy, the feed is genuinely degraded
    and the override should NOT save us. Stay in HOLD_ALL."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_GAP_MEDIAN_MS=500.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS=500.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=100,
        ),
        **_kwargs(
            seconds_since_public_bbo=0.050,
            gap_median_ms=800.0,
            gap_last_ms=900.0,  # current gap ALSO bad
            gap_p95_ms=2000.0,
            gap_sample_count=500,
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL


def test_override_needs_both_age_and_gap_condition() -> None:
    """Age-live alone is not enough — we also need a healthy-gap signal."""
    # Age live, but median AND last_gap both missing → override doesn't fire.
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=100,
        ),
        **_kwargs(
            seconds_since_public_bbo=0.050,
            gap_median_ms=None,
            gap_last_ms=None,
            gap_p95_ms=1500.0,
            gap_sample_count=500,
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL


def test_missing_last_gap_does_not_break_median_path() -> None:
    """Back-compat: no last_gap reported, median-only override still works."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=100,
        ),
        **_kwargs(
            seconds_since_public_bbo=0.050,
            gap_median_ms=200.0,       # healthy
            gap_last_ms=None,           # missing
            gap_p95_ms=1500.0,
            gap_sample_count=500,
        ),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH
    assert "p95_override_live" in r.reason


# ------------------- Fix B: p95 warmup skip ------------------------


def test_warmup_skip_below_min_samples_with_polluted_p95() -> None:
    """With gap_sample_count below the threshold, p95-hold must be skipped
    regardless of the p95 value — the statistic is unreliable in small samples."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=200,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_ENABLED=False,  # isolate warmup path
        ),
        **_kwargs(
            seconds_since_public_bbo=0.050,
            gap_median_ms=800.0,          # polluted median too
            gap_p95_ms=3000.0,             # p95 way over
            gap_sample_count=74,            # below min 200
        ),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH
    assert "p95_warmup" in r.reason


def test_warmup_does_not_skip_once_enough_samples() -> None:
    """Past the min-sample threshold, p95-hold fires as before (assuming no
    override)."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=200,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_ENABLED=False,
        ),
        **_kwargs(
            seconds_since_public_bbo=0.050,
            gap_median_ms=800.0,
            gap_p95_ms=1500.0,
            gap_sample_count=500,         # past warmup
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL


def test_warmup_skip_does_not_affect_age_hold() -> None:
    """Age-based holds must still fire during warmup. The warmup guard ONLY
    covers p95. A stale book is a stale book regardless of sample count."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_BOOK_AGE_MS=1200.0,
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=200,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_ENABLED=False,
        ),
        **_kwargs(
            seconds_since_public_bbo=2.0,  # 2000 ms — stale
            gap_median_ms=50.0,
            gap_p95_ms=3000.0,
            gap_sample_count=50,            # below warmup
        ),
    )
    assert r.eligibility == QuoteEligibility.HOLD_ALL
    # Hold reason should cite age, NOT p95 (since p95 was warmup-skipped).
    assert "age" in r.reason.lower() or "local_receipt" in r.reason
    assert "gap_p95_ms" not in r.reason  # p95 WAS skipped


def test_warmup_skip_with_min_samples_zero_is_disabled() -> None:
    """Setting MIN_SAMPLES=0 disables the warmup guard (legacy behaviour)."""
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=0,  # disabled
            QUOTE_FRESHNESS_LIVE_OVERRIDE_ENABLED=False,
        ),
        **_kwargs(
            seconds_since_public_bbo=0.050,
            gap_median_ms=800.0,
            gap_p95_ms=3000.0,
            gap_sample_count=5,            # tiny sample
        ),
    )
    # MIN_SAMPLES=0 → no warmup skip → p95 gate fires → HOLD_ALL.
    assert r.eligibility == QuoteEligibility.HOLD_ALL


# ------------------- Regression: exact 132744 scenario ------------------------


def test_regression_20260418_132744_cold_start_scenario() -> None:
    """Exact numbers from the 132744 snapshot (feed polluted at session start).

    Before the fix: stuck in HOLD_ALL, user had to wait 5 min for gap-ring to
    rotate the outlier out before bot resumed quoting.

    After the fix: either the warmup guard (74 < 200 samples) OR the last-gap
    override (416 < 500) lifts the hold immediately. Result: QUOTE_BOTH.
    """
    r = compute_quote_eligibility(
        _s(
            QUOTE_HOLD_MAX_GAP_P95_MS=1000.0,
            QUOTE_ONE_SIDED_MAX_GAP_P95_MS=750.0,
            QUOTE_HOLD_MAX_BOOK_AGE_MS=1200.0,
            QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS=600.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_BOOK_AGE_MS=150.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_GAP_MEDIAN_MS=500.0,
            QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS=500.0,
            QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES=200,
        ),
        **_kwargs(
            # Exact values from state_current.json + market-data_gap-stats.json:
            seconds_since_public_bbo=0.050,   # book age ~50 ms at the quiet moments
            gap_median_ms=748.638,             # polluted by 3.9 s outlier
            gap_last_ms=416.408,                # feed alive right now
            gap_p95_ms=1813.822,                # p95 breaches threshold
            gap_sample_count=74,                 # below warmup threshold
        ),
    )
    assert r.eligibility == QuoteEligibility.QUOTE_BOTH
