"""v1.4.102 regime-defense tuning verification.

The TON profile (``config/profiles/prod.okx.ton.usdt.perp.env``)
ships with the new defensive tuning. Tests here pin the integration:

1. TON profile values load cleanly (no validation errors).
2. Inventory-skew curve produces stronger response at mid utilisation
   than the pre-v1.4.102 defaults would have.
3. SOFT/HARD limit thresholds tightened to 0.50 / 0.75.
4. ``slow_trend_gate_enabled=True`` in the TON profile (it ships off
   by default but the production TON profile enables it).
5. ``slow_trend_widen_bps`` is on the NON-sentinel value of 10.0 bp
   (the gate ships live-from-day-one without coefficient iteration).

These tests guard against accidental regressions when the TON profile
is edited.

2026-05-20 v1.4.118 revision history (skew curve subset only):
==============================================================
The v1.4.102 skew tune (coeff=20, exp=1.5) compounded with the
``BASE_HALF_SPREAD_BPS`` bump to 5.0 (committed dddb109) pushed
the BID quote 14.8 bp from mid on a 1-tick-wide TON market —
2 ticks behind the touch, no fills for 28+ minutes (snapshot
``v1.4.117-260520-151247``). v1.4.118 dialled back: coeff 20→14,
exp 1.5→2.0 (revert to the concave-up shape). The new curve:

                       pre-v1.4.102    v1.4.102      v1.4.118
                       (12, exp 2.0)   (20, exp 1.5) (14, exp 2.0)
    util=0.3:           1.08 bp         3.30 bp       1.26 bp
    util=0.5:           3.00 bp         7.07 bp       3.50 bp
    util=0.7:           5.88 bp        11.71 bp       6.86 bp
    util=0.9:           9.72 bp        17.07 bp      11.34 bp

v1.4.118 is still STRONGER than the pre-v1.4.102 baseline (the
spirit of v1.4.102's defense) but no longer compounds with the
BASE_HALF_SPREAD_BPS=4.0 floor (also dialled back from 5.0 in
v1.4.118) into "quotes can't reach the touch" territory. The
defensive intent — "respond more strongly at mid util than the
old (12, exp 2.0) curve did" — is preserved.

If you tune this curve again, update both the assertion values
AND the table above so the rationale stays in sync with the file.

2026-05-20 v1.4.121 revision history (inventory limits + regime gate):
======================================================================
v1.4.118 fixed the skew-curve compounding but left a second
suppression layer in place: the v1.4.102 tightening of
INVENTORY_SOFT_LIMIT_PCT (0.65 → 0.50) and HARD_LIMIT_PCT
(0.85 → 0.75). Snapshot v1.4.118-260520-155640 showed why this
was load-bearing: at util=60% the bot tripped soft_skew_long
24.8/sec for the entire session, pct_time_two_sided_effective
was 0%, 3 fills in 25 min. Simultaneously the
REGIME_CONTROLLER (added v1.4.117) entered DEFENSIVE at
util_entry=0.5 and spent 99% of session in DEFENSIVE mode —
its 1.5× spread / 1.5× skew / +1-tick adding-side penalty
compounded with the soft-skew suppression.

v1.4.121 reverts both:
  * INVENTORY_SOFT_LIMIT_PCT: 0.50 → 0.65 (v1.4.92 baseline)
  * INVENTORY_HARD_LIMIT_PCT: 0.75 → 0.85 (v1.4.92 baseline)
  * REGIME_UTIL_ENTRY_THRESHOLD: 0.50 → 0.70
  * REGIME_UTIL_EXIT_THRESHOLD: 0.30 → 0.50

Net operating bands for the TON profile (post-v1.4.121):

    util          0.30      0.50      0.65      0.70      0.85      0.95
                  │         │         │         │         │         │
    calm quoting  ━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    soft_skew     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┃━━━━━━━━━━━┓
    DEFENSIVE     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┃━━━━━━━━┓
    hard_skew     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┃━━━━━

Soft layers stack from less restrictive to more. DEFENSIVE
sits BELOW hard_skew so the controller engages first; the
REGIME EXIT (0.50) sits at the soft limit so once a session
returns to 'calm quoting', the controller exits DEFENSIVE.

The v1.4.102 defensive INTENT (earlier one-sided arming) is
satisfied by the REGIME_CONTROLLER's DEFENSIVE mode at
util≥0.70 — same operator-visible behaviour (size halved,
spread widened, adding-side penalised) but through the
single FSM channel rather than three independent layers
stacking on top of each other.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def ton_profile_settings():
    """Load the TON profile env into a fresh Settings instance."""
    from app.config import Settings

    profile_path = (
        Path(__file__).parent.parent
        / "config"
        / "profiles"
        / "prod.okx.ton.usdt.perp.env"
    )
    assert profile_path.exists(), f"TON profile not found at {profile_path}"

    saved_env: dict[str, str | None] = {}
    try:
        # Parse the profile file and apply it.
        for line in profile_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            k = k.strip()
            v = v.strip()
            # Strip optional inline comments after the value (env-style).
            if " #" in v:
                v = v.split(" #")[0].strip()
            saved_env[k] = os.environ.get(k)
            os.environ[k] = v
        # Ensure trading-safety env that Settings demands.
        os.environ.setdefault("HL_SECRET_KEY", "")
        os.environ.setdefault("HL_ACCOUNT_ADDRESS", "")
        os.environ.setdefault("TRADING_ENABLED", "false")
        # Construct.
        s = Settings()
        yield s
    finally:
        # Restore environment so other tests aren't polluted.
        for k, prev in saved_env.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


# ---------------------------------------------------------------------------
# Inventory-skew tunes (#1 + #2)
# ---------------------------------------------------------------------------


def test_ton_profile_inventory_skew_coefficient_tuned(ton_profile_settings):
    """v1.4.102 → v1.4.118 → v1.5.276: coefficient 12 → 20 → 14 → 10.

    v1.4.118 dialled the v1.4.102 (20) back to (14) after observation
    that the 20-coeff/1.5-exp curve compounded with the 5-bp base half
    spread to push BID quotes 2 ticks behind touch on a 1-tick wide
    TON market. 14 is still stronger than the pre-v1.4.102 baseline
    of 12 (preserves the defensive spirit) but no longer leaves the
    bot unable to reach the touch.

    v1.5.276 (Weekend Aggression Pack A, 2026-05-30) — dropped from
    14 → 10 to unblock 2-sided weekend trading after the v1.5.275-
    260530-100531 session went near-dead (BUY=345 / SELL=0
    place_stage_returned_none asymmetry). The env file marks this as
    "Revert to 14 for weekday operation"; the test pin tracks
    whatever the prod profile currently ships. Weekday-revert PR
    will restore the 14.
    """
    assert ton_profile_settings.inventory_skew_coeff_bps == 10.0


def test_ton_profile_inventory_skew_exponent_tuned(ton_profile_settings):
    """Revision history: v1.4.102 reduced exp 2.0 → 1.5; v1.4.118 full
    revert to 2.0; v1.4.157 partial retune to 1.5.

    v1.4.157 (this pin) — snapshot v1.4.151-260520-203719 showed
    equity ↔ TON-mid Pearson correlation of −0.683 with inventory held
    nonzero 84.5 % of the time (one run lasted ~1 h 52 m) and SELL-side
    5 s markout of −1.86 bps. Root cause: at the typical 40 % operating
    utilisation, exp=2.0 produces only ~2.1 bps of skew — barely enough
    to shift fill probability toward the reducing side. The bot kept
    adding instead of reducing, then ate directional moves.

    The v1.4.157 tune drops EXP back to 1.5 — kicks in earlier at
    low/mid utilisation (3.5 bps at 40 % util vs the old 2.1 bps,
    +1.6×) without changing the high-util shape that the v1.4.92
    baseline operated under. **Deliberately not going to EXP=1.0** to
    avoid the v1.4.118 emergency, which was caused by EXP=1.5 combined
    with COEFF=20 AND SOFT=0.50 — three knobs at once. Only EXP moved
    here, COEFF stays at 14, SOFT stays at 0.65.

    Paired with the simultaneously-enabled trend_skew_amplifier
    (v1.4.157), which conditionally adds a 1.5× multiplier when
    inventory direction is aligned with adverse drift — kicks in
    exactly when carry is going adverse, dormant in calm markets.

    v1.5.276 (Weekend Aggression Pack A, 2026-05-30) — EXP 1.5 → 1.3.
    Flattens the high-util portion of the curve so weekend trading
    isn't strangled when inventory hits 50-70 % util. Companion to
    the COEFF 14 → 10 drop above. The env file marks this as
    "Revert to 1.5 for weekday operation"; weekday-revert PR will
    restore the 1.5 pin.
    """
    assert ton_profile_settings.inventory_skew_exponent == 1.3


def test_ton_profile_inventory_soft_limit_tightened(ton_profile_settings):
    """v1.4.102 → v1.4.121: SOFT 0.65 → 0.50 → 0.65 (full revert).

    v1.4.102 tightened to 0.50 for "earlier one-sided defence" but
    snapshot v1.4.118-260520-155640 showed the compounding effect:
    at util=60% (legitimate trading load), `soft_skew_long` fired
    37,549 times in 25 min (24.8/sec — every 40 ms BID suppressed),
    bot quoted both sides 0% of the time, 3 fills. v1.4.92 at the
    same 60% util with soft=0.65 was below the threshold and quoted
    two-sided 93.5% — 45 fills/hour. The "earlier defence" was
    suffocating routine operation.

    v1.4.121 reverts to 0.65 (v1.4.92 baseline). The hard_skew
    suppressor still fires above this threshold (138K fires in
    v1.4.92's 20h is the expected/normal high-util defence rate).
    """
    assert ton_profile_settings.inventory_soft_limit_pct == 0.65


def test_ton_profile_inventory_hard_limit_tightened(ton_profile_settings):
    """v1.4.102 → v1.4.121: HARD 0.85 → 0.75 → 0.85 (full revert).

    Companion to the soft-limit revert above. Tightening 0.85 → 0.75
    compressed the headroom between soft and hard limits — combined
    with the soft-limit drop to 0.50 it narrowed the "calm quoting"
    band from [0, 0.65] to [0, 0.50] and the "soft-defence" band
    from [0.65, 0.85] (20 pp wide) to [0.50, 0.75] (25 pp wide).
    Net effect at typical 30-60% util: bot spent most of its time
    in soft-defence rather than the calm-quoting band.

    Reverting to 0.85 restores v1.4.92's headroom. The hard limit
    sits above the regime_controller's util_entry threshold (now
    0.70, raised in v1.4.121 from 0.50) so the FSM enters DEFENSIVE
    BEFORE hard_skew engages — defence-in-depth preserved.
    """
    assert ton_profile_settings.inventory_hard_limit_pct == 0.85


def test_ton_skew_curve_yields_stronger_response_at_mid_util(
    ton_profile_settings,
):
    """Verify the current skew curve is STRICTLY STRONGER than the
    pre-v1.4.102 baseline at mid utilisation. The defensive intent
    of v1.4.102 ("Sunday-night accumulation" defence) is preserved
    across all subsequent revisions.

    pre-v1.4.102 (12 bp × util²):  util=0.5 → 3.00 bp
    v1.4.102     (20 bp × util^1.5): util=0.5 → 7.07 bp (2.4×, rolled
                                                          back — too
                                                          aggressive)
    v1.4.118     (14 bp × util²):  util=0.5 → 3.50 bp (1.17×)
    v1.4.157     (14 bp × util^1.5): util=0.5 → 4.95 bp (1.65× — kicks
                                                          in earlier
                                                          at low/mid
                                                          util than
                                                          v1.4.118)

    Bar: strictly stronger than the pre-v1.4.102 (12, exp 2.0)
    baseline. Each subsequent revision satisfies this.
    """
    coeff = ton_profile_settings.inventory_skew_coeff_bps
    exp = ton_profile_settings.inventory_skew_exponent
    util = 0.5
    skew_at_half_util = coeff * (util ** exp)
    pre_v1_4_102_baseline = 12.0 * (util ** 2.0)  # 3.00 bp
    assert skew_at_half_util > pre_v1_4_102_baseline, (
        f"Skew at util=0.5 must be > pre-v1.4.102 baseline "
        f"({pre_v1_4_102_baseline:.2f} bp); got {skew_at_half_util:.2f} bp"
    )


def test_ton_skew_curve_at_max_util(ton_profile_settings):
    """At util=0.9 (just below cap), the curve should still bite — but
    not so hard that quoting collapses on a 1-tick TON market.

    pre-v1.4.102 (12, exp 2.0):    util=0.9 →  9.72 bp
    v1.4.118     (14, exp 2.0):    util=0.9 → 11.34 bp
    v1.4.157     (14, exp 1.5):    util=0.9 → 11.95 bp (small bump
                                              at high util; the
                                              v1.4.157 change is
                                              mostly felt at low/mid
                                              util — see the mid-util
                                              test above)
    v1.4.102     (20, exp 1.5):    util=0.9 → 17.07 bp (rolled back —
                                              too aggressive on TON
                                              1-tick market)
    v1.5.276     (10, exp 1.3):    util=0.9 →  8.72 bp (Weekend
                                              Aggression Pack A —
                                              intentionally below the
                                              pre-v1.4.102 baseline
                                              at high util so the
                                              2-sided weekend quoting
                                              isn't strangled when
                                              inventory hits 50-70 %;
                                              reverts to v1.4.157
                                              shape on weekday)

    Bar: positive AND ≥ 5 bp (sanity — SOME defensive bite remains)
    AND ≤ v1.4.102 ceiling (17.07 — the documented "too aggressive
    on TON 1-tick market" line). Lower than pre-v1.4.102 (9.72) is
    allowed for the weekend-aggression regime; anything below 5 bp
    would mean inventory defence has effectively been disabled and
    is the actual regression we care about catching.
    """
    coeff = ton_profile_settings.inventory_skew_coeff_bps
    exp = ton_profile_settings.inventory_skew_exponent
    util = 0.9
    skew_at_high_util = coeff * (util ** exp)
    v1_4_102_ceiling = 20.0 * (util ** 1.5)  # 17.07 bp
    minimum_defensive_floor = 5.0  # below this, inventory defence is gone
    assert minimum_defensive_floor < skew_at_high_util < v1_4_102_ceiling, (
        f"Skew at util=0.9 should be between minimum defensive floor "
        f"({minimum_defensive_floor:.2f}) and v1.4.102 ceiling "
        f"({v1_4_102_ceiling:.2f}); got {skew_at_high_util:.2f}"
    )


# ---------------------------------------------------------------------------
# slow_trend_gate config (#3)
# ---------------------------------------------------------------------------


def test_ton_profile_slow_trend_gate_enabled(ton_profile_settings):
    """v1.4.102: TON profile enables slow_trend_gate."""
    assert ton_profile_settings.slow_trend_gate_enabled is True


def test_ton_profile_slow_trend_window_15min(ton_profile_settings):
    """15 minutes — longer than the 5-min long_drift window."""
    assert ton_profile_settings.slow_trend_window_seconds == 900.0


def test_ton_profile_slow_trend_threshold_25bp(ton_profile_settings):
    """25 bp — lower than the 50-bp long_drift threshold to catch
    slow grinds that average away in 5-min snapshots."""
    assert ton_profile_settings.slow_trend_threshold_bps == 25.0


def test_ton_profile_slow_trend_widen_bps_live_default(ton_profile_settings):
    """Critical: widen_bps must be NON-sentinel (positive value).
    The gate ships live from day one — sentinel -1.0 would make it
    gate-equivalent dark, defeating the whole point."""
    widen_bps = ton_profile_settings.slow_trend_widen_bps
    assert widen_bps > 0, (
        f"slow_trend_widen_bps must be > 0 for live-from-day-one "
        f"semantics; got {widen_bps}"
    )
    # Sanity bound: should be in the operator-usable range, not
    # accidentally near MAX (which would be effectively-dark).
    assert widen_bps <= 20.0, (
        f"slow_trend_widen_bps={widen_bps} is suspiciously large; "
        f"expected a moderate live value like 10 bp"
    )
