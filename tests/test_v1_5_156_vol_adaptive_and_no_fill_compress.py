"""v1.5.156 — tests for Option A (vol-adaptive base half-spread) and
Option B (no-fill–aware spread compression).

The driving observation (overnight ~9h session screenshot,
2026-05-26): in calm regimes the bot's quoted spread is structurally
wider than the natural touch → 56-min zero-fill windows. Option A
drops the base half-spread when vol is low; Option B gradually
compresses the spread after a configurable no-fill timeout, with
the first fill resetting the timer.

Both features are bounded by the existing
``min_half_spread_bps`` clamp, so safety floors still hold.

Per CLAUDE.md: only this test file is run from the assistant; full-
suite verification is the CI daemon's job.
"""

from __future__ import annotations

from app.models import ToxicitySnapshot
from app.quoting import compute_quote_decision
from tests.settings_helpers import UnitTestSettings as Settings


def _settings(**kw) -> Settings:
    return Settings(
        trading_enabled=False,
        hl_secret_key="",
        hl_account_address="",
        **kw,
    )


def _no_toxicity() -> ToxicitySnapshot:
    return ToxicitySnapshot(0, 0, 0, 1, False, False)


def _base_settings(**overrides):
    """Common test setup: minimal half-spread machinery so the
    feature being tested is the dominant contribution."""
    base = dict(
        max_abs_position=10.0,
        base_half_spread_bps=8.0,
        min_half_spread_bps=0.0,
        max_half_spread_bps=80.0,
        vol_multiplier=0.0,
        economic_min_half_spread_neutral_bps=0.0,
        economic_min_half_spread_inventory_bps=0.0,
        toxicity_score_half_spread_bps=0.0,
    )
    base.update(overrides)
    return _settings(**base)


# ---------------------------------------------------------------------------
# Option A — vol-adaptive base half-spread
# ---------------------------------------------------------------------------


def test_option_a_disabled_uses_legacy_base():
    """When ``BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=false`` the legacy
    ``BASE_HALF_SPREAD_BPS`` is used regardless of vol."""
    s = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=False,
        BASE_HALF_SPREAD_BPS=8.0,
        BASE_HALF_SPREAD_BPS_LOW_VOL=2.0,  # would be used if enabled
        BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S=10.0,
    )
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=1.0, toxicity=_no_toxicity())
    # vol=1 well below the 10 threshold; but flag is off → legacy base.
    assert q.target_spread_bps == 16.0  # 2 * 8


def test_option_a_low_vol_uses_low_vol_base():
    """When enabled AND vol < threshold, the low-vol base is used."""
    s = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=True,
        BASE_HALF_SPREAD_BPS=8.0,
        BASE_HALF_SPREAD_BPS_LOW_VOL=2.0,
        BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S=6.0,
    )
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=3.0, toxicity=_no_toxicity())
    # vol=3 below 6 → low-vol base 2 → total 4.
    assert q.target_spread_bps == 4.0  # 2 * 2


def test_option_a_high_vol_uses_normal_base():
    """When enabled BUT vol >= threshold, the normal base is used."""
    s = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=True,
        BASE_HALF_SPREAD_BPS=8.0,
        BASE_HALF_SPREAD_BPS_LOW_VOL=2.0,
        BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S=6.0,
    )
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=10.0, toxicity=_no_toxicity())
    # vol=10 above 6 → normal base 8 → total 16.
    assert q.target_spread_bps == 16.0


def test_option_a_at_threshold_uses_normal_base():
    """Strict inequality — vol exactly at threshold uses normal base."""
    s = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=True,
        BASE_HALF_SPREAD_BPS=8.0,
        BASE_HALF_SPREAD_BPS_LOW_VOL=2.0,
        BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S=6.0,
    )
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=6.0, toxicity=_no_toxicity())
    assert q.target_spread_bps == 16.0  # normal


def test_option_a_respects_min_half_spread_clamp():
    """Even if low-vol base is tiny, the existing min_half_spread_bps
    floor still binds."""
    s = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=True,
        BASE_HALF_SPREAD_BPS=8.0,
        BASE_HALF_SPREAD_BPS_LOW_VOL=0.5,
        BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S=10.0,
        MIN_HALF_SPREAD_BPS=3.0,  # hard floor wins
    )
    q = compute_quote_decision(s, 100.0, 0.0, vol_bps=1.0, toxicity=_no_toxicity())
    assert q.target_spread_bps == 6.0  # 2 * 3 (floored)


def test_option_a_overnight_screenshot_calibration():
    """Tuning sanity for the operator-chosen prod profile values
    (BASE=4.0, LOW=2.5, threshold=6.0) at the overnight session's
    vol=5 bp/s. Verifies the new total spread is materially tighter
    than legacy."""
    legacy = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=False,
        BASE_HALF_SPREAD_BPS=4.0,
    )
    new = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=True,
        BASE_HALF_SPREAD_BPS=4.0,
        BASE_HALF_SPREAD_BPS_LOW_VOL=2.5,
        BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S=6.0,
    )
    q_legacy = compute_quote_decision(
        legacy, 1.9, 0.0, vol_bps=5.0, toxicity=_no_toxicity()
    )
    q_new = compute_quote_decision(
        new, 1.9, 0.0, vol_bps=5.0, toxicity=_no_toxicity()
    )
    # Legacy: 2 * 4 = 8.  New: 2 * 2.5 = 5.  Saving: 3 bps total.
    assert q_legacy.target_spread_bps == 8.0
    assert q_new.target_spread_bps == 5.0
    assert q_new.target_spread_bps < q_legacy.target_spread_bps


# ---------------------------------------------------------------------------
# Option B — no-fill–aware spread compression
# ---------------------------------------------------------------------------


def test_option_b_disabled_does_not_compress():
    """When ``NO_FILL_COMPRESS_ENABLED=false`` no compression is
    applied regardless of how long since last fill."""
    s = _base_settings(
        NO_FILL_COMPRESS_ENABLED=False,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=300.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
    )
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=10_000.0,
    )
    assert q.target_spread_bps == 16.0  # 2 * 8, unchanged


def test_option_b_none_seconds_skips_compression():
    """When the caller passes ``None`` for seconds_since_last_fill
    (no fill yet this session), no compression applies."""
    s = _base_settings(
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=300.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
    )
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=None,
    )
    assert q.target_spread_bps == 16.0


def test_option_b_below_trigger_no_compression():
    """Within the trigger window, no compression is applied yet."""
    s = _base_settings(
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=600.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
    )
    # 500 s since last fill, below 600 trigger.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=500.0,
    )
    assert q.target_spread_bps == 16.0


def test_option_b_at_trigger_no_compression():
    """Exactly at trigger — strict greater-than, no compression yet."""
    s = _base_settings(
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=600.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
    )
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=600.0,
    )
    assert q.target_spread_bps == 16.0


def test_option_b_one_minute_past_trigger():
    """One minute past trigger at 1 bp/min → 1 bp compression on
    half-spread → 2 bp on total."""
    s = _base_settings(
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=600.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
    )
    # 660 s = 1 min past trigger.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=660.0,
    )
    # base 8 - 1 = 7 half-spread → 14 total.
    assert q.target_spread_bps == 14.0


def test_option_b_caps_at_max_compression():
    """Long no-fill stretch capped at NO_FILL_COMPRESS_MAX_BPS."""
    s = _base_settings(
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=600.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
    )
    # 6000 s = 90 min past trigger. Uncapped would be 90 bps; capped at 5.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=6_000.0,
    )
    # base 8 - 5 = 3 half-spread → 6 total.
    assert q.target_spread_bps == 6.0


def test_option_b_respects_min_half_spread_floor():
    """Compression cannot push below ``MIN_HALF_SPREAD_BPS``."""
    s = _base_settings(
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=600.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=10.0,  # aggressive
        NO_FILL_COMPRESS_MAX_BPS=20.0,
        BASE_HALF_SPREAD_BPS=8.0,
        MIN_HALF_SPREAD_BPS=3.0,
    )
    # 660 s past trigger * 10 bp/min = 100 bp compression — would
    # take half-spread to -92. Floor at 3.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=660.0 + 600.0,
    )
    assert q.target_spread_bps == 6.0  # 2 * 3


def test_option_b_overnight_screenshot_calibration():
    """Tuning sanity for the 56-min zero-fill stretch observed in the
    2026-05-26 overnight session: with trigger=600 (10 min), rate=1
    bp/min, max=5, the bot's spread would compress by:
      * 10 min in: 0 bp (still in trigger window)
      * 15 min in: 5 bp (5 min past trigger at 1/min)
      * 20 min in: 5 bp (capped at max)
      * 56 min in: 5 bp (capped at max)
    This is the operator-chosen prod profile calibration."""
    s = _base_settings(
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=600.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
        BASE_HALF_SPREAD_BPS=8.0,
        MIN_HALF_SPREAD_BPS=0.5,
    )
    tox = _no_toxicity()
    # At 10 min (trigger boundary): no compression.
    q_10m = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=tox,
        seconds_since_last_fill=600.0,
    )
    # At 15 min: 5 min past trigger × 1 = 5 bp compression.
    q_15m = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=tox,
        seconds_since_last_fill=900.0,
    )
    # At 56 min: capped at max=5 bp.
    q_56m = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=0.0, toxicity=tox,
        seconds_since_last_fill=3_360.0,
    )
    assert q_10m.target_spread_bps == 16.0  # 2 * 8
    assert q_15m.target_spread_bps == 6.0   # 2 * (8 - 5)
    assert q_56m.target_spread_bps == 6.0   # capped


# ---------------------------------------------------------------------------
# Option A + Option B together
# ---------------------------------------------------------------------------


def test_options_a_and_b_stack_in_low_vol_no_fill_regime():
    """Both features active simultaneously: the low-vol base is the
    starting half-spread, no-fill compression then subtracts from it.
    The combination should be very tight (the operator's intended
    state for calm regimes with no taker flow)."""
    s = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=True,
        BASE_HALF_SPREAD_BPS=8.0,
        BASE_HALF_SPREAD_BPS_LOW_VOL=2.5,
        BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S=6.0,
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=600.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
        MIN_HALF_SPREAD_BPS=0.5,
    )
    # Low-vol regime + 20 min no fill.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=3.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=1_200.0,
    )
    # base = 2.5 (low-vol), compress = min(10 * 1, 5) = 5.
    # half = max(2.5 - 5, 0.5) = 0.5 (floor binds).
    # total = 1.0.
    assert q.target_spread_bps == 1.0


def test_options_a_and_b_high_vol_no_fill():
    """In high vol + no fills, Option A keeps the normal base
    (defensive), Option B still compresses (slowly). Net result:
    moderate compression from the higher base."""
    s = _base_settings(
        BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=True,
        BASE_HALF_SPREAD_BPS=8.0,
        BASE_HALF_SPREAD_BPS_LOW_VOL=2.5,
        BASE_HALF_SPREAD_LOW_VOL_THRESHOLD_BPS_PER_S=6.0,
        NO_FILL_COMPRESS_ENABLED=True,
        NO_FILL_COMPRESS_TRIGGER_SECONDS=600.0,
        NO_FILL_COMPRESS_RATE_BPS_PER_MINUTE=1.0,
        NO_FILL_COMPRESS_MAX_BPS=5.0,
        MIN_HALF_SPREAD_BPS=0.5,
    )
    # High vol + 15 min no fill.
    q = compute_quote_decision(
        s, 100.0, 0.0, vol_bps=20.0, toxicity=_no_toxicity(),
        seconds_since_last_fill=900.0,
    )
    # base = 8 (high vol → normal base), compress = 5 min * 1 = 5.
    # half = 8 - 5 = 3.  total = 6.
    assert q.target_spread_bps == 6.0
