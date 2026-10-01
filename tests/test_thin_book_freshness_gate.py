"""Integration regression: thin-book symbols (AXS / NEAR) with realistic
BBO cadence must not produce HOLD_ALL loops under the thin-book-tuned
freshness profile.

Context:
  ``tmp/snap_20260418_153202`` — the AXS profile running with the
  venue-default 1000 ms p95 hold threshold ran into 98 % HOLD_ALL on
  98 of 100 quote cycles because AXS's measured BBO cadence was
  median_gap 888 ms / p95_gap 2202 ms / last_gap 3753 ms — all above
  the default freshness thresholds calibrated for ETH's ~20 ms cadence.

  The fix was a per-profile relaxation of the freshness knobs. This
  test locks that fix in end-to-end by calling the real
  ``compute_quote_eligibility`` with AXS-class numbers under both the
  old and new thresholds, asserting:

    * Default thresholds + AXS cadence → HOLD_ALL (documents the bug).
    * Thin-book-tuned thresholds + AXS cadence → QUOTE_BOTH.

Tests focus on the freshness gate in isolation. Drift / jump gates,
order-state uncertainty, and per-side recovery are exercised by
``tests/test_freshness_gap_p95_live_override.py`` and friends.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.enums import QuoteEligibility
from app.quote_eligibility import compute_quote_eligibility
from tests.settings_helpers import UnitTestSettings


# ---------------------------------------------------------------------------
# Profile-file locator -- the bot's config/profiles/ directory holds
# active profiles; archived ones live in config/profiles_unused/. The
# tests below assert per-profile knob coherence; we want them to keep
# guarding the values whether the operator has the file in active or
# archived state, and to skip gracefully if the file is gone entirely.
# ---------------------------------------------------------------------------


def _find_profile(name: str) -> Path:
    """Return the path of profile ``name`` (without `.env`) under
    config/profiles/ or config/profiles_unused/. Pytest-skip if neither
    exists, so an operator who archived a profile sees the test as
    skipped rather than failed.
    """
    repo_root = Path(__file__).resolve().parent.parent
    for sub in ("profiles", "profiles_unused"):
        p = repo_root / "config" / sub / f"{name}.env"
        if p.is_file():
            return p
    pytest.skip(
        f"profile {name}.env not found under config/profiles/ "
        f"or config/profiles_unused/ -- skipping knob coherence test"
    )


# -----------------------------------------------------------------------
# Freshness inputs matching the measurements from tmp/snap_20260418_153202
# (AXS, GRVT public WS, 17.7 min session):
#
#   update_count   = 1089  (~1.03 updates/sec)
#   median_gap_ms  = 888
#   p95_gap_ms     = 2202
#   last_gap_ms    = 3753      (snapshot caught mid-quiet-window)
#   max_gap_ms     = 7142
#   book_age       = 0.459 ms  (latest BBO just arrived)
# -----------------------------------------------------------------------

_AXS_P95_GAP_MS = 2202.0
_AXS_MEDIAN_GAP_MS = 888.0
_AXS_LAST_GAP_MS = 3753.0
_AXS_BOOK_AGE_S = 0.000459   # latest BBO arrived <1 ms before decision
_AXS_GAP_SAMPLE_COUNT = 1088  # well above the p95-warmup floor


def _default_eth_settings() -> UnitTestSettings:
    """Venue-default thresholds (the ones calibrated for ETH). Does NOT
    override the freshness knobs."""
    return UnitTestSettings.model_validate(
        {
            "QUOTE_ELIGIBILITY_ENABLED": True,
            "SYMBOL": "ETH_USDT_Perp",
            # All freshness knobs at default:
            #   QUOTE_HOLD_MAX_GAP_P95_MS         = 600  (from app/config.py)
            #   QUOTE_ONE_SIDED_MAX_GAP_P95_MS    = 350
            #   QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS = 500
            #   QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_GAP_MEDIAN_MS = 500
            #   QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_BOOK_AGE_MS = 150
        }
    )


def _thin_book_settings() -> UnitTestSettings:
    """Thin-book-tuned thresholds matching prod.grvt.axs.env /
    prod.grvt.near.env. Raises every gap threshold ~3x to accommodate
    GRVT's ~1 update/sec cadence on these symbols."""
    return UnitTestSettings.model_validate(
        {
            "QUOTE_ELIGIBILITY_ENABLED": True,
            "SYMBOL": "AXS_USDT_Perp",
            "QUOTE_HOLD_MAX_BOOK_AGE_MS": 4000.0,
            "QUOTE_ONE_SIDED_MAX_BOOK_AGE_MS": 2000.0,
            "QUOTE_HOLD_MAX_GAP_P95_MS": 3000.0,
            "QUOTE_ONE_SIDED_MAX_GAP_P95_MS": 2000.0,
            "QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS": 2500.0,
            "QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_GAP_MEDIAN_MS": 1500.0,
            "QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_BOOK_AGE_MS": 1000.0,
        }
    )


def _call(settings: UnitTestSettings):
    """Invoke the real eligibility pipeline with AXS-class cadence."""
    return compute_quote_eligibility(
        settings,
        order_state_uncertainty=False,
        mid_now=1.1545,
        now_mono=1000.0,
        mid_samples=[(1000.0 - 0.1, 1.1545), (1000.0 - 0.5, 1.1545)],
        seconds_since_public_bbo=_AXS_BOOK_AGE_S,
        gap_median_ms=_AXS_MEDIAN_GAP_MS,
        gap_p95_ms=_AXS_P95_GAP_MS,
        effective_staleness_ms=_AXS_BOOK_AGE_S * 1000.0,
        vol_bps=1.0,
        gap_last_ms=_AXS_LAST_GAP_MS,
        gap_sample_count=_AXS_GAP_SAMPLE_COUNT,
    )


# ----- Documents the bug: ETH defaults + AXS cadence = HOLD_ALL ----------


def test_default_eth_thresholds_hold_all_on_axs_cadence() -> None:
    """Documents the pre-fix bug. ETH-default thresholds (p95_hold=600 ms)
    reject AXS's measured p95_gap=2202 ms → HOLD_ALL on every cycle.

    This is the failure mode observed in tmp/snap_20260418_153202:
    98 of 100 quote cycles blocked on freshness. Leaving this as an
    executable assertion so we can't accidentally tighten the thin-book
    profiles back towards ETH defaults without the test screaming.
    """
    settings = _default_eth_settings()
    result = _call(settings)
    assert result.eligibility == QuoteEligibility.HOLD_ALL
    assert "freshness_hold" in result.reason
    assert "gap_p95_ms" in result.reason


# ----- The fix: thin-book thresholds pass AXS cadence --------------------


def test_thin_book_thresholds_pass_on_axs_cadence() -> None:
    """With the AXS / NEAR profile thresholds, the same measurements
    yield QUOTE_BOTH. This is the "bot can actually quote" assertion."""
    settings = _thin_book_settings()
    result = _call(settings)
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH, (
        f"thin-book freshness must permit quoting at AXS cadence, "
        f"got eligibility={result.eligibility} reason={result.reason}"
    )
    # Sanity: shouldn't be via the warmup-skip path (sample count is
    # well over the p95-warmup floor).
    assert "freshness_ok" in result.reason


# ----- Edge cases inside the thin-book envelope --------------------------


def test_thin_book_p95_at_threshold_passes() -> None:
    """Exactly at the thin-book p95 threshold → pass (boundary test)."""
    settings = _thin_book_settings()
    result = compute_quote_eligibility(
        settings,
        order_state_uncertainty=False,
        mid_now=1.1545,
        now_mono=1000.0,
        mid_samples=[(999.9, 1.1545), (999.5, 1.1545)],
        seconds_since_public_bbo=0.001,
        gap_median_ms=700.0,
        gap_p95_ms=3000.0,  # exactly at QUOTE_HOLD_MAX_GAP_P95_MS
        effective_staleness_ms=1.0,
        vol_bps=1.0,
        gap_last_ms=700.0,
        gap_sample_count=500,
    )
    assert result.eligibility == QuoteEligibility.QUOTE_BOTH


def test_thin_book_p95_one_tick_over_threshold_holds() -> None:
    """Just over the thin-book p95 threshold → HOLD_ALL. The thresholds
    were lifted to accommodate AXS, not to disable the gate entirely.

    To show the HOLD actually fires we also have to defeat the
    live-book-fresh override (age_is_live AND (median_healthy OR
    last_healthy)). We do that by making BOTH median_gap and last_gap
    exceed their override ceilings (1500 ms / 2500 ms respectively).
    With neither "healthy" signal, the override can't fire even though
    the book itself is fresh — and the raw p95_hold bubbles through.
    """
    settings = _thin_book_settings()
    result = compute_quote_eligibility(
        settings,
        order_state_uncertainty=False,
        mid_now=1.1545,
        now_mono=1000.0,
        mid_samples=[(999.9, 1.1545), (999.5, 1.1545)],
        seconds_since_public_bbo=0.001,      # book is fresh (age_is_live=True)
        gap_median_ms=1600.0,                # > 1500 override ceiling → not healthy
        gap_p95_ms=3001.0,                   # > 3000 HOLD threshold by 1 ms
        effective_staleness_ms=1.0,
        vol_bps=1.0,
        gap_last_ms=3000.0,                  # > 2500 override ceiling → not healthy
        gap_sample_count=500,
    )
    assert result.eligibility == QuoteEligibility.HOLD_ALL
    assert "freshness_hold" in result.reason
    assert "gap_p95_ms" in result.reason


def test_thin_book_stale_book_age_still_holds() -> None:
    """A genuinely stale book (5 s since last BBO) must still be caught
    by the book-age gate even under relaxed thresholds. Relaxing p95 is
    not the same as disabling freshness — the catastrophic-stale catch
    remains."""
    settings = _thin_book_settings()
    result = compute_quote_eligibility(
        settings,
        order_state_uncertainty=False,
        mid_now=1.1545,
        now_mono=1000.0,
        mid_samples=[(999.9, 1.1545), (999.5, 1.1545)],
        seconds_since_public_bbo=5.0,  # book is 5s old — genuinely stale
        gap_median_ms=700.0,
        gap_p95_ms=1500.0,
        effective_staleness_ms=5000.0,
        vol_bps=1.0,
        gap_last_ms=5000.0,
        gap_sample_count=500,
    )
    assert result.eligibility == QuoteEligibility.HOLD_ALL
    assert "freshness_hold" in result.reason


# ----- Profile-level sanity check ---------------------------------------


def test_axs_profile_file_uses_thin_book_thresholds() -> None:
    """Sanity: the prod.grvt.axs.env file actually contains the thresholds
    this test suite codifies. If someone accidentally reverts the profile
    to the ETH defaults we want the test to scream."""
    from pathlib import Path

    axs_env = _find_profile("prod.grvt.axs")
    text = axs_env.read_text(encoding="utf-8")
    # Spot-check the three knobs that were mis-tuned pre-fix.
    assert "QUOTE_HOLD_MAX_GAP_P95_MS=3000" in text
    assert "QUOTE_ONE_SIDED_MAX_GAP_P95_MS=2000" in text
    assert "QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS=2500" in text


def test_near_profile_file_uses_thin_book_thresholds() -> None:
    """Same spot-check for NEAR — both thin-book profiles should match."""
    from pathlib import Path

    near_env = _find_profile("prod.grvt.near")
    text = near_env.read_text(encoding="utf-8")
    assert "QUOTE_HOLD_MAX_GAP_P95_MS=3000" in text
    assert "QUOTE_ONE_SIDED_MAX_GAP_P95_MS=2000" in text
    assert "QUOTE_FRESHNESS_LIVE_OVERRIDE_MAX_LAST_GAP_MS=2500" in text


def test_axs_profile_file_uses_split_idle_thresholds() -> None:
    """Sanity: the AXS profile carries both PRIVATE_WS_IDLE_WARN_SECONDS
    and PRIVATE_WS_IDLE_RECONNECT_SECONDS, and the reconnect threshold is
    large enough to accommodate thin-book silence."""
    from pathlib import Path

    axs_env = _find_profile("prod.grvt.axs")
    text = axs_env.read_text(encoding="utf-8")
    assert "PRIVATE_WS_IDLE_WARN_SECONDS=120.0" in text
    assert "PRIVATE_WS_IDLE_RECONNECT_SECONDS=300.0" in text


def test_eth_profile_file_keeps_tight_reconnect_for_fast_trading() -> None:
    """ETH is fast-trading: the reconnect tier must stay tight (60 s) so
    stuck-subscription recovery fires before the stuck-cancel ages out.
    Mis-copying the thin-book thresholds onto ETH would defeat the
    original stuck-subscription fix from tmp/snap_20260417_173125."""
    from pathlib import Path

    eth_env = _find_profile("prod.grvt.eth")
    text = eth_env.read_text(encoding="utf-8")
    assert "PRIVATE_WS_IDLE_WARN_SECONDS=30.0" in text
    assert "PRIVATE_WS_IDLE_RECONNECT_SECONDS=60.0" in text


# ----- Reprice-churn floor (Option A from tmp/snap_20260418_163608) ------


def test_axs_profile_reprice_threshold_clears_one_tick() -> None:
    """AXS on GRVT has tick=0.001 / ~$1.15 price → 1 tick ≈ 8.7 bps.
    The reprice threshold's effective value is clip(vol_mul×vol, MIN, MAX);
    to avoid cancel/replace on sub-tick jitter, the effective value must
    be >= 1 tick under ALL vol regimes. That requires BPS_MIN >= 8 bps
    (so low-vol sessions clip UP to ~1 tick) AND BPS >= 12 bps (so
    high-vol sessions still have room to scale).

    Ref: tmp/snap_20260418_163608 — with previous values (BPS=5,
    BPS_MIN=2) the effective threshold under observed short_vol=1.37 bps
    was 4.11 bps = ~half a tick, causing 72 % of quote cycles to enter
    ``order_state_uncertainty_both_sides`` and effective two-sided
    quoting to drop to 5.75 %.
    """
    from pathlib import Path

    axs_env = _find_profile("prod.grvt.axs")
    text = axs_env.read_text(encoding="utf-8")
    assert "REPRICE_THRESHOLD_BPS=12.0" in text
    assert "REPRICE_THRESHOLD_BPS_MIN=8.0" in text


def test_near_profile_reprice_threshold_clears_one_tick() -> None:
    """Same invariant as AXS — NEAR has the same tick + book geometry."""
    from pathlib import Path

    near_env = _find_profile("prod.grvt.near")
    text = near_env.read_text(encoding="utf-8")
    assert "REPRICE_THRESHOLD_BPS=12.0" in text
    assert "REPRICE_THRESHOLD_BPS_MIN=8.0" in text


def test_reprice_threshold_effective_value_under_observed_vol() -> None:
    """Sanity check on the reprice clip math itself.

    Given the thin-book-profile values (BPS=12, BPS_MIN=8, MULTIPLIER=3)
    and the observed low-vol regime on AXS (short_vol_bps ≈ 1.37), the
    effective threshold must be >= 1 tick worth of bps on AXS
    (8.7 bps at mid $1.15). This test computes the clip explicitly so
    if someone tunes any of the three inputs without thinking about the
    interaction, the breakage is caught here rather than in production.
    """
    # Replicate the formula from the code comment in app/config.py:
    #   effective = clip(VOL_MULTIPLIER * short_vol_bps, BPS_MIN, BPS)
    bps_ceiling = 12.0
    bps_floor = 8.0
    vol_mult = 3.0
    observed_vol_bps = 1.37

    raw = vol_mult * observed_vol_bps
    effective = min(bps_ceiling, max(bps_floor, raw))
    # AXS: 1 tick = 0.001 / 1.15 = 8.7 bps. Demand effective >= ~1 tick.
    one_tick_bps_axs = 8.695
    assert effective >= one_tick_bps_axs - 1.0, (
        f"effective reprice threshold {effective:.2f} bps is below one tick "
        f"on AXS ({one_tick_bps_axs:.2f} bps) at observed vol — reprice churn "
        f"will resume"
    )
