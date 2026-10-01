"""Tests for ``scripts/snapshot_acceptance.py`` — per-release
acceptance-criterion auditor.

Validates:
1. Each check's PASS / WARN / FAIL / N/A transitions on synthetic
   snapshot data.
2. The contradiction registry fires when paired criteria pull in
   opposite directions.
3. Version-filter prevents v1.5.155 checks from running on
   v1.5.154 snapshots.

Per CLAUDE.md: only this test file is run from the assistant; full-
suite verification is the CI daemon's job.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# scripts/ is not a package — add to sys.path so we can import.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "scripts"))

from snapshot_acceptance import (  # noqa: E402
    CONTRADICTIONS,
    CheckResult,
    SnapshotData,
    _evaluate_contradictions,
    check_v1_5_147_dampen_band_active,
    check_v1_5_147_sf_slice_active,
    check_v1_5_147_shock_full_dark_active,
    check_v1_5_149_publisher_blocks_emit,
    check_v1_5_151_position_aware_throttle,
    check_v1_5_154_cooldown_holds_all,
    check_v1_5_155_cautious_dwell,
    check_v1_5_155_fill_rate,
    check_v1_5_155_mae_gate_position_favorable,
    check_v1_5_155_markouts_balanced,
    check_v1_5_155_trend_skew_non_zero,
    check_v1_5_156_low_vol_fill_rate,
    check_v1_5_156_markout_stability,
    check_v1_5_156_min_floor_respected,
    check_v1_5_185_avellaneda_stoikov_active,
    check_v1_5_288_candidate_a_vol_warmstart,
    check_v1_5_288_candidate_b_microprice_z_fires,
    check_v1_5_294_book_age_hold_reverted,
    check_v1_5_294_combined_feed_gap_p95_healthy,
    check_v1_5_294_freshness_not_dominating_holds,
    check_v1_5_294_markout_coverage,
)


def _snap(
    *,
    bot_version: str = "1.5.156",
    state_current: Any = None,
    session_summary: Any = None,
    config: Any = None,
    inventory_since: Any = None,
    fills_since: Any = None,
    snapshot_dir: Any = None,
) -> SnapshotData:
    return SnapshotData(
        snapshot_dir=Path("/tmp/fake") if snapshot_dir is None else Path(snapshot_dir),
        snapshot_name="fake_snap",
        bot_version=bot_version,
        captured_at="2026-05-27T00:00:00Z",
        state_current=state_current,
        session_summary=session_summary,
        config=config,
        inventory_since=inventory_since,
        fills_since=fills_since,
    )


# ---------------------------------------------------------------------------
# Version filter
# ---------------------------------------------------------------------------


def test_v1_5_155_check_returns_na_on_v1_5_154_snapshot():
    r = check_v1_5_155_trend_skew_non_zero(_snap(bot_version="1.5.154"))
    assert r.status == "N/A"
    assert "predates v1.5.155" in r.detail


def test_v1_5_156_check_returns_na_on_v1_5_155_snapshot():
    r = check_v1_5_156_low_vol_fill_rate(_snap(bot_version="1.5.155"))
    assert r.status == "N/A"


# ---------------------------------------------------------------------------
# v1.5.151 — pre-publisher snapshot returns N/A (the 508K trap)
# ---------------------------------------------------------------------------


def test_v1_5_151_pre_publisher_returns_na():
    state = {
        "behavioural_gates": {
            "structural_bias_throttle": {
                "enabled": True,
                "fire_count": 508550,
                # changed_eligibility_count NOT present → pre-v1.5.155 publisher
            }
        }
    }
    r = check_v1_5_151_position_aware_throttle(
        _snap(bot_version="1.5.154", state_current=state)
    )
    assert r.status == "N/A"
    assert "pre-v1.5.155 publisher" in r.detail


def test_v1_5_151_modern_publisher_with_zero_engagement_passes():
    state = {
        "behavioural_gates": {
            "structural_bias_throttle": {
                "enabled": True,
                "fire_count": 508550,
                "changed_eligibility_count": 0,
            }
        }
    }
    ss = {"duration_seconds": 30000}
    r = check_v1_5_151_position_aware_throttle(
        _snap(bot_version="1.5.155", state_current=state, session_summary=ss)
    )
    assert r.status == "PASS"
    assert "changed_eligibility=0" in r.detail


def test_v1_5_151_high_engagement_fails():
    """Pre-v1.5.151 pattern: gate firing on misaligned position
    produces ~40 fires/sec = 144 K/hour. Synthesise that with
    600 K engagements over 30000s = 72 K/hour → FAIL."""
    state = {
        "behavioural_gates": {
            "structural_bias_throttle": {
                "enabled": True,
                "fire_count": 600_000,
                "changed_eligibility_count": 600_000,  # 72 K/hr
            }
        }
    }
    ss = {"duration_seconds": 30000}
    r = check_v1_5_151_position_aware_throttle(
        _snap(bot_version="1.5.155", state_current=state, session_summary=ss)
    )
    assert r.status == "FAIL"


def test_v1_5_151_continuous_engagement_passes():
    """Post-v1.5.151 pattern: gate fires every tick while position
    + bias are aligned. At 2 Hz cadence = 7,200/hr. v1.5.157-
    260526-113525 snapshot showed 7,240/hr in production — that's
    PASS, not FAIL (which is what the original threshold mistakenly
    flagged)."""
    state = {
        "behavioural_gates": {
            "structural_bias_throttle": {
                "enabled": True,
                "fire_count": 9956,
                "changed_eligibility_count": 9507,  # 7240/hr
            }
        }
    }
    ss = {"duration_seconds": 4727}  # 1.31 h
    r = check_v1_5_151_position_aware_throttle(
        _snap(bot_version="1.5.157", state_current=state, session_summary=ss)
    )
    assert r.status == "PASS"


# ---------------------------------------------------------------------------
# v1.5.154 — HOLD_ALL semantics
# ---------------------------------------------------------------------------


def test_v1_5_154_legacy_format_fails():
    state = {
        "last_quote_breakdown": {
            "quote_eligibility_reason": "ok|post_sf_cooldown:LONG_42s"
        }
    }
    r = check_v1_5_154_cooldown_holds_all(_snap(state_current=state))
    assert r.status == "FAIL"
    assert "LONG_" in r.detail


def test_v1_5_154_new_format_passes():
    state = {
        "last_quote_breakdown": {
            "quote_eligibility_reason": "ok|post_sf_cooldown:HOLD_ALL_prev=LONG_42s"
        }
    }
    r = check_v1_5_154_cooldown_holds_all(_snap(state_current=state))
    assert r.status == "PASS"


def test_v1_5_154_inactive_passes():
    state = {"last_quote_breakdown": {"quote_eligibility_reason": "ok"}}
    r = check_v1_5_154_cooldown_holds_all(_snap(state_current=state))
    assert r.status == "PASS"


# ---------------------------------------------------------------------------
# v1.5.155 — trend skew non-zero
# ---------------------------------------------------------------------------


def test_v1_5_155_trend_skew_quiet_market_insufficient_data():
    """Drift too small to verdict — INSUFFICIENT_DATA (the signal is
    present but ambiguous, not "doesn't apply")."""
    state = {
        "last_quote_breakdown": {"trend_drift_shift_bps": 0.3},
        "mid_drift_windows": {"drift_10s_bps": 0.5, "drift_5s_bps": 0.5},
    }
    r = check_v1_5_155_trend_skew_non_zero(_snap(state_current=state))
    assert r.status == "INSUFFICIENT_DATA"


def test_v1_5_155_trend_skew_zero_when_drift_high_fails():
    state = {
        "last_quote_breakdown": {"trend_drift_shift_bps": 0.0},
        "mid_drift_windows": {"drift_10s_bps": -10.0, "drift_5s_bps": -10.0},
    }
    r = check_v1_5_155_trend_skew_non_zero(_snap(state_current=state))
    assert r.status == "FAIL"


def test_v1_5_155_trend_skew_sign_mismatch_fails():
    state = {
        "last_quote_breakdown": {"trend_drift_shift_bps": -5.0},
        "mid_drift_windows": {"drift_10s_bps": +10.0, "drift_5s_bps": +10.0},
    }
    r = check_v1_5_155_trend_skew_non_zero(_snap(state_current=state))
    assert r.status == "FAIL"
    assert "sign mismatch" in r.detail


def test_v1_5_155_trend_skew_proportional_passes():
    state = {
        "last_quote_breakdown": {"trend_drift_shift_bps": -5.0},
        "mid_drift_windows": {"drift_10s_bps": -10.0, "drift_5s_bps": -10.0},
    }
    r = check_v1_5_155_trend_skew_non_zero(_snap(state_current=state))
    assert r.status == "PASS"


# ---------------------------------------------------------------------------
# v1.5.155 — per-side markouts balanced
# ---------------------------------------------------------------------------


def test_markouts_balanced_passes():
    ss = {
        "attribution": {
            "per_side": {
                "BUY": {"mean_markout_bps": -0.5, "fills": 50},
                "SELL": {"mean_markout_bps": -1.2, "fills": 50},
            }
        }
    }
    r = check_v1_5_155_markouts_balanced(_snap(session_summary=ss))
    assert r.status == "PASS"


def test_markouts_asymmetric_fails():
    ss = {
        "attribution": {
            "per_side": {
                "BUY": {"mean_markout_bps": +2.0, "fills": 50},
                "SELL": {"mean_markout_bps": -4.5, "fills": 50},
            }
        }
    }
    r = check_v1_5_155_markouts_balanced(_snap(session_summary=ss))
    assert r.status == "FAIL"


# ---------------------------------------------------------------------------
# v1.5.155 / v1.5.156 — fill rate
# ---------------------------------------------------------------------------


def test_v1_5_155_fill_rate_below_baseline_fails():
    ss = {"throughput": {"fill_rate_per_min": 0.18}}
    r = check_v1_5_155_fill_rate(_snap(session_summary=ss))
    assert r.status == "FAIL"


def test_v1_5_156_fill_rate_target_met_passes():
    ss = {"throughput": {"fill_rate_per_min": 0.60}}
    r = check_v1_5_156_low_vol_fill_rate(_snap(session_summary=ss))
    assert r.status == "PASS"


# ---------------------------------------------------------------------------
# v1.5.156 — markout stability
# ---------------------------------------------------------------------------


def test_v1_5_156_markout_stable_passes():
    ss = {"attribution": {"markout": {"mean_bps": -2.5, "sample_count": 100}}}
    r = check_v1_5_156_markout_stability(_snap(session_summary=ss))
    assert r.status == "PASS"


def test_v1_5_156_markout_regression_fails():
    ss = {"attribution": {"markout": {"mean_bps": -4.0, "sample_count": 100}}}
    r = check_v1_5_156_markout_stability(_snap(session_summary=ss))
    assert r.status == "FAIL"


# ---------------------------------------------------------------------------
# v1.5.156 — min half-spread floor
# ---------------------------------------------------------------------------


def test_v1_5_156_floor_respected_passes():
    state = {
        "last_quote_breakdown": {
            "target_half_spread_bps": 2.0,
            "raw_half_spread_bps": 1.0,
            "clamp_winner": "min",
        }
    }
    cfg = {"MIN_HALF_SPREAD_BPS": 1.5}
    r = check_v1_5_156_min_floor_respected(
        _snap(state_current=state, config=cfg)
    )
    assert r.status == "PASS"


def test_v1_5_156_floor_violated_fails():
    state = {
        "last_quote_breakdown": {
            "target_half_spread_bps": 0.5,  # below 1.5 floor
            "raw_half_spread_bps": 0.5,
            "clamp_winner": "raw",
        }
    }
    cfg = {"MIN_HALF_SPREAD_BPS": 1.5}
    r = check_v1_5_156_min_floor_respected(
        _snap(state_current=state, config=cfg)
    )
    assert r.status == "FAIL"


# ---------------------------------------------------------------------------
# v1.5.155 — mae_gate position-favorable clears
# ---------------------------------------------------------------------------


def test_mae_gate_no_fires_returns_na():
    state = {"behavioural_gates": {"mae_gate": {"fire_count": 0}}}
    r = check_v1_5_155_mae_gate_position_favorable(_snap(state_current=state))
    assert r.status == "N/A"


def test_mae_gate_only_ceiling_clears_warns():
    state = {
        "behavioural_gates": {
            "mae_gate": {
                "fire_count": 50,
                "cleared_via_position_favorable_total": 0,
                "cleared_via_ceiling_total": 50,
            }
        }
    }
    r = check_v1_5_155_mae_gate_position_favorable(_snap(state_current=state))
    assert r.status == "WARN"


def test_mae_gate_with_position_favorable_clears_passes():
    state = {
        "behavioural_gates": {
            "mae_gate": {
                "fire_count": 50,
                "cleared_via_position_favorable_total": 30,
                "cleared_via_ceiling_total": 20,
            }
        }
    }
    r = check_v1_5_155_mae_gate_position_favorable(_snap(state_current=state))
    assert r.status == "PASS"


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------


def test_fill_rate_up_vs_markout_stable_contradiction_fires():
    """v1.5.156 AC2 PASS with high fill rate + AC3 WARN with degraded
    markout → flagged contradiction with action."""
    results = [
        CheckResult(
            "v1.5.156", "AC2: low-vol fill rate >= 0.5/min",
            "PASS", "fill_rate = 0.65 /min",
            measured={"fill_rate_per_min": 0.65},
        ),
        CheckResult(
            "v1.5.156", "AC3: markout 5s mean stable",
            "WARN", "markout = -3.2 bps",
            measured={"markout_5s_mean": -3.2},
        ),
    ]
    cons = _evaluate_contradictions(results)
    assert len(cons) == 1
    con, detail = cons[0]
    assert con.name == "fill_rate_up_vs_markout_stable"
    assert "0.65/min" in detail
    assert "3.0" in detail  # action recommendation


def test_no_contradiction_when_both_pass():
    results = [
        CheckResult(
            "v1.5.156", "AC2: low-vol fill rate >= 0.5/min",
            "PASS", "0.55", measured={"fill_rate_per_min": 0.55},
        ),
        CheckResult(
            "v1.5.156", "AC3: markout 5s mean stable",
            "PASS", "-2.5", measured={"markout_5s_mean": -2.5},
        ),
    ]
    cons = _evaluate_contradictions(results)
    assert len(cons) == 0


def test_no_contradiction_when_one_is_na():
    """N/A status (insufficient data) should not trigger
    contradictions."""
    results = [
        CheckResult(
            "v1.5.156", "AC2: low-vol fill rate >= 0.5/min",
            "PASS", "0.65", measured={"fill_rate_per_min": 0.65},
        ),
        CheckResult(
            "v1.5.156", "AC3: markout 5s mean stable",
            "N/A", "insufficient samples", measured={},
        ),
    ]
    cons = _evaluate_contradictions(results)
    assert len(cons) == 0


def test_trend_skew_vs_markout_contradiction_fires():
    """Strong skew (>3 bps shift) AND asymmetric markouts
    (diff > 2 bps) → over-leaning suspicion."""
    results = [
        CheckResult(
            "v1.5.155", "AC1: trend_drift_shift_bps materially non-zero",
            "PASS", "shift 5 bps",
            measured={"shift_bps": -5.0, "drift_10s_bps": -10.0},
        ),
        CheckResult(
            "v1.5.155", "AC2: per-side markouts balanced",
            "WARN", "asymmetric",
            measured={"buy_mean": +1.5, "sell_mean": -3.5, "diff": 5.0},
        ),
    ]
    cons = _evaluate_contradictions(results)
    assert len(cons) == 1
    assert "TREND_DRIFT_RESERVATION_ALPHA" in cons[0][1]


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# v1.5.147 / v1.5.149 — Tier A publisher blocks
# ---------------------------------------------------------------------------


def test_v1_5_147_dampen_band_active_passes():
    state = {"behavioural_gates": {
        "negative_expectancy_dampen": {
            "enabled": True,
            "widen_bps_config": 2.0,
        }
    }}
    r = check_v1_5_147_dampen_band_active(_snap(state_current=state))
    assert r.status == "PASS"


def test_v1_5_147_dampen_block_missing_fails():
    state = {"behavioural_gates": {}}
    r = check_v1_5_147_dampen_band_active(_snap(state_current=state))
    assert r.status == "FAIL"
    assert "publisher emitter not running" in r.detail


def test_v1_5_147_dampen_enabled_but_widen_zero_warns():
    state = {"behavioural_gates": {
        "negative_expectancy_dampen": {"enabled": True, "widen_bps_config": 0.0}
    }}
    r = check_v1_5_147_dampen_band_active(_snap(state_current=state))
    assert r.status == "WARN"
    assert "effectively dormant" in r.detail


def test_v1_5_147_dampen_disabled_returns_na():
    state = {"behavioural_gates": {
        "negative_expectancy_dampen": {"enabled": False, "widen_bps_config": 2.0}
    }}
    r = check_v1_5_147_dampen_band_active(_snap(state_current=state))
    assert r.status == "N/A"


def test_v1_5_147_sf_slice_active_passes():
    state = {"behavioural_gates": {
        "sf_slice": {
            "enabled": True,
            "slice_notional_usd_config": 4.0,
            "dispatched_total": 16,
        }
    }}
    r = check_v1_5_147_sf_slice_active(_snap(state_current=state))
    assert r.status == "PASS"
    assert "dispatched_total=16" in r.detail


def test_v1_5_147_shock_full_dark_passes():
    state = {"behavioural_gates": {
        "shock_ladder_full_dark": {"enabled": True}
    }}
    r = check_v1_5_147_shock_full_dark_active(_snap(state_current=state))
    assert r.status == "PASS"


def test_v1_5_149_publisher_blocks_all_present_passes():
    state = {"behavioural_gates": {
        "negative_expectancy_dampen": {"enabled": True},
        "sf_slice": {"enabled": True},
        "shock_ladder_full_dark": {"enabled": True},
    }}
    r = check_v1_5_149_publisher_blocks_emit(_snap(state_current=state))
    assert r.status == "PASS"


def test_v1_5_149_publisher_missing_blocks_fails():
    state = {"behavioural_gates": {
        "negative_expectancy_dampen": {"enabled": True},
        # sf_slice and shock_ladder_full_dark missing
    }}
    r = check_v1_5_149_publisher_blocks_emit(_snap(state_current=state))
    assert r.status == "FAIL"
    assert "sf_slice" in r.detail
    assert "shock_ladder_full_dark" in r.detail


# ---------------------------------------------------------------------------
# Status taxonomy
# ---------------------------------------------------------------------------


def test_status_taxonomy_distinguishes_na_and_insufficient_data():
    """N/A = doesn't apply. INSUFFICIENT_DATA = applies but data
    missing. These must be distinct so the operator can tell why a
    check didn't produce a verdict."""
    # N/A: version doesn't apply
    r_na = check_v1_5_155_trend_skew_non_zero(_snap(bot_version="1.5.154"))
    assert r_na.status == "N/A"
    # INSUFFICIENT_DATA: applies but no state to compute from
    r_id = check_v1_5_155_trend_skew_non_zero(
        _snap(bot_version="1.5.155", state_current=None)
    )
    assert r_id.status == "INSUFFICIENT_DATA"


def test_all_contradictions_have_complete_definition():
    """Each Contradiction must have all required fields populated."""
    for c in CONTRADICTIONS:
        assert c.name
        assert c.ac_a
        assert c.ac_b
        assert c.rationale
        assert callable(c.evaluator)


# ---------------------------------------------------------------------------
# Regression: v1.5.150 BUG-027 check field-name bug (2026-05-27)
# ---------------------------------------------------------------------------


def test_v1_5_150_bug027_detects_event_type_field(tmp_path):
    """The snapshot's events_since.json uses ``event_type`` as the
    canonical key. Pre-fix the check looked for ``event``/``type``/
    ``name`` and silently reported 0 wedges on snapshots that had
    them — a false PASS. This regression test plants 3 synthetic
    wedge events with the actual production shape and verifies
    they're counted."""
    from snapshot_acceptance import check_v1_5_150_no_silent_wedge
    import json
    stats = tmp_path / "stats"
    stats.mkdir()
    # Real production-shape events (3 wedges + 2 unrelated).
    events = [
        {"ts": "2026-05-25T19:12:37Z", "event_type": "executor_silent_wedge_detected",
         "severity": "WARNING", "message": "executor stalled"},
        {"ts": "2026-05-25T19:19:34Z", "event_type": "executor_silent_wedge_detected",
         "severity": "WARNING", "message": "executor stalled"},
        {"ts": "2026-05-25T19:32:02Z", "event_type": "executor_silent_wedge_detected",
         "severity": "WARNING", "message": "executor stalled"},
        {"ts": "2026-05-25T19:00:00Z", "event_type": "bot_start",
         "severity": "INFO", "message": "ok"},
        {"ts": "2026-05-25T19:11:31Z", "event_type": "soft_flatten_started",
         "severity": "WARNING", "message": "ok"},
    ]
    (stats / "events_since.json").write_text(json.dumps(events))
    snap = SnapshotData(
        snapshot_dir=tmp_path,
        snapshot_name="regression",
        bot_version="1.5.154",
        captured_at="2026-05-26T06:04:30Z",
        state_current=None,
        session_summary=None,
        config=None,
        inventory_since=None,
        fills_since=None,
    )
    r = check_v1_5_150_no_silent_wedge(snap)
    assert r.status == "FAIL"
    assert r.measured.get("wedge_count") == 3


# ---------------------------------------------------------------------------
# Retirement plumbing
# ---------------------------------------------------------------------------


def test_retired_checks_list_exists_and_well_formed():
    """Every RetiredCheck must have name, release, retired_in, reason.
    Empty list is fine (nothing retired yet)."""
    from snapshot_acceptance import RETIRED_CHECKS, RetiredCheck
    assert isinstance(RETIRED_CHECKS, list)
    for rc in RETIRED_CHECKS:
        assert isinstance(rc, RetiredCheck)
        assert rc.name
        assert rc.release
        assert rc.retired_in
        assert rc.reason
        # superseded_by is optional


def test_no_check_appears_in_both_active_and_retired():
    """A check is either active or retired — never both. The name
    is the key (it's how operators look it up)."""
    from snapshot_acceptance import CHECKS, RETIRED_CHECKS
    # Active check names — invoke each function with an empty snap
    # just to read the name from the result.
    empty_snap = SnapshotData(
        snapshot_dir=Path("/tmp/empty"),
        snapshot_name="empty",
        bot_version="0.0.0",
        captured_at="",
        state_current=None,
        session_summary=None,
        config=None,
        inventory_since=None,
        fills_since=None,
    )
    active_names = {fn(empty_snap).name for fn in CHECKS}
    retired_names = {
        rc.name.split(" / ", 1)[-1]  # strip "v1.5.X / " prefix
        for rc in RETIRED_CHECKS
    }
    overlap = active_names & retired_names
    assert not overlap, (
        f"check name appears in both CHECKS and RETIRED_CHECKS: {overlap}. "
        f"A retired check must be removed from CHECKS."
    )


def test_retired_check_dataclass_round_trip():
    """The RetiredCheck dataclass shape must accept the documented
    fields (no surprises when an operator follows the template)."""
    from snapshot_acceptance import RetiredCheck
    rc = RetiredCheck(
        name="example / fake_check",
        release="v1.5.X",
        retired_in="v1.5.Y",
        reason="superseded by new mechanism",
        superseded_by="v1.5.Y / replacement_check",
    )
    assert rc.name == "example / fake_check"
    assert rc.superseded_by == "v1.5.Y / replacement_check"
    # Without superseded_by — also valid.
    rc2 = RetiredCheck(
        name="example / other",
        release="v1.5.X",
        retired_in="v1.5.Y",
        reason="feature removed entirely",
    )
    assert rc2.superseded_by is None


# ---------------------------------------------------------------------------
# v1.5.185 — Avellaneda-Stoikov active check
# ---------------------------------------------------------------------------


def test_as_check_na_when_flag_disabled():
    cfg = {"AVELLANEDA_STOIKOV_ENABLED": "false"}
    r = check_v1_5_185_avellaneda_stoikov_active(
        _snap(bot_version="1.5.189", config=cfg)
    )
    assert r.status == "N/A"
    assert "deliberately off" in r.detail


def test_as_check_na_when_version_too_old():
    cfg = {"AVELLANEDA_STOIKOV_ENABLED": "true"}
    r = check_v1_5_185_avellaneda_stoikov_active(
        _snap(bot_version="1.5.188", config=cfg)
    )
    assert r.status == "N/A"
    assert "predates v1.5.189" in r.detail


def test_as_check_fails_when_fire_count_zero():
    """Flag on but AS path never fired — the v1.5.186 silent-bug
    pattern."""
    cfg = {"AVELLANEDA_STOIKOV_ENABLED": "true"}
    state = {
        "as_path_fire_count": 0,
        "as_k_intensity_per_min": None,
    }
    ss = {"duration_seconds": 3600.0}  # 1 hour
    r = check_v1_5_185_avellaneda_stoikov_active(
        _snap(
            bot_version="1.5.189",
            config=cfg,
            state_current=state,
            session_summary=ss,
        )
    )
    assert r.status == "FAIL"
    assert "never fired" in r.detail


def test_as_check_fails_when_k_cache_stuck_none_after_grace():
    """Flag on, AS fired, but k-intensity cache still None after
    the 5-min grace — broken refresh path."""
    cfg = {"AVELLANEDA_STOIKOV_ENABLED": "true"}
    state = {
        "as_path_fire_count": 1000,
        "as_k_intensity_per_min": None,
    }
    ss = {"duration_seconds": 600.0}  # 10 min (> 5-min grace)
    r = check_v1_5_185_avellaneda_stoikov_active(
        _snap(
            bot_version="1.5.189",
            config=cfg,
            state_current=state,
            session_summary=ss,
        )
    )
    assert r.status == "FAIL"
    assert "cache refresh path may be broken" in r.detail


def test_as_check_pass_warmup_state():
    """Within the 5-min grace window, k=None is acceptable as long
    as the fire-count is positive."""
    cfg = {"AVELLANEDA_STOIKOV_ENABLED": "true"}
    state = {
        "as_path_fire_count": 60,
        "as_k_intensity_per_min": None,
    }
    ss = {"duration_seconds": 120.0}  # 2 min (still in grace)
    r = check_v1_5_185_avellaneda_stoikov_active(
        _snap(
            bot_version="1.5.189",
            config=cfg,
            state_current=state,
            session_summary=ss,
        )
    )
    assert r.status == "PASS"
    assert "warmup" in r.detail


def test_as_check_pass_steady_state():
    """AS firing, cache populated — happy path."""
    cfg = {"AVELLANEDA_STOIKOV_ENABLED": "true"}
    state = {
        "as_path_fire_count": 3600,
        "as_k_intensity_per_min": 0.523,
    }
    ss = {"duration_seconds": 3600.0}  # 1 hour
    r = check_v1_5_185_avellaneda_stoikov_active(
        _snap(
            bot_version="1.5.189",
            config=cfg,
            state_current=state,
            session_summary=ss,
        )
    )
    assert r.status == "PASS"
    assert "as_path_fire_count=3600" in r.detail
    assert "0.523" in r.detail


# ---------------------------------------------------------------------------
# v1.5.288 M8 Candidate A — vol warm-start seed (one-shot)
# ---------------------------------------------------------------------------

_FEED_ON = {"REGIME_USE_RUNTIME_RECORDER_FEED": "true"}
_FEED_OFF = {"REGIME_USE_RUNTIME_RECORDER_FEED": "false"}


def test_candidate_a_na_on_old_version():
    r = check_v1_5_288_candidate_a_vol_warmstart(
        _snap(bot_version="1.5.287", config=_FEED_ON)
    )
    assert r.status == "N/A"
    assert "predates v1.5.288" in r.detail


def test_candidate_a_na_when_feed_disabled():
    # The default production state: knob OFF → feed never opened →
    # nothing to seed from. Distinct from INSUFFICIENT_DATA.
    state = {"warmstart_vol_seeded_from_recorder_count": 0}
    r = check_v1_5_288_candidate_a_vol_warmstart(
        _snap(bot_version="1.5.288", config=_FEED_OFF, state_current=state)
    )
    assert r.status == "N/A"
    assert "REGIME_USE_RUNTIME_RECORDER_FEED disabled" in r.detail


def test_candidate_a_insufficient_when_config_missing():
    r = check_v1_5_288_candidate_a_vol_warmstart(
        _snap(bot_version="1.5.288", config=None)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "config.json missing" in r.detail


def test_candidate_a_insufficient_when_state_missing():
    r = check_v1_5_288_candidate_a_vol_warmstart(
        _snap(bot_version="1.5.288", config=_FEED_ON, state_current=None)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "state_current.json missing" in r.detail


def test_candidate_a_insufficient_when_counter_zero():
    # Feed ON but vol_bps_p95_24h is still dark today → seed no-ops →
    # counter stays 0. This is INSUFFICIENT_DATA (re-evaluate once the
    # recorder lights the field), NOT a failure.
    state = {"warmstart_vol_seeded_from_recorder_count": 0}
    r = check_v1_5_288_candidate_a_vol_warmstart(
        _snap(bot_version="1.5.288", config=_FEED_ON, state_current=state)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "dark" in r.detail


def test_candidate_a_pass_when_counter_one():
    state = {"warmstart_vol_seeded_from_recorder_count": 1}
    r = check_v1_5_288_candidate_a_vol_warmstart(
        _snap(bot_version="1.5.288", config=_FEED_ON, state_current=state)
    )
    assert r.status == "PASS"
    assert r.measured["warmstart_vol_seeded_from_recorder_count"] == 1


def test_candidate_a_fail_when_counter_above_one():
    # One-shot contract violated — the seed re-applied.
    state = {"warmstart_vol_seeded_from_recorder_count": 4}
    r = check_v1_5_288_candidate_a_vol_warmstart(
        _snap(bot_version="1.5.288", config=_FEED_ON, state_current=state)
    )
    assert r.status == "FAIL"
    assert "one-shot" in r.detail


# ---------------------------------------------------------------------------
# v1.5.288 M8 Candidate B — microprice-z widen fires >=5% of reads
# ---------------------------------------------------------------------------


def test_candidate_b_na_on_old_version():
    r = check_v1_5_288_candidate_b_microprice_z_fires(
        _snap(bot_version="1.5.287", config=_FEED_ON)
    )
    assert r.status == "N/A"
    assert "predates v1.5.288" in r.detail


def test_candidate_b_na_when_feed_disabled():
    state = {
        "runtime_feed_read_count": 0,
        "microprice_widen_z_runtime_feed_fires": 0,
    }
    r = check_v1_5_288_candidate_b_microprice_z_fires(
        _snap(bot_version="1.5.288", config=_FEED_OFF, state_current=state)
    )
    assert r.status == "N/A"
    assert "REGIME_USE_RUNTIME_RECORDER_FEED disabled" in r.detail


def test_candidate_b_insufficient_when_no_reads():
    state = {
        "runtime_feed_read_count": 0,
        "microprice_widen_z_runtime_feed_fires": 0,
    }
    r = check_v1_5_288_candidate_b_microprice_z_fires(
        _snap(bot_version="1.5.288", config=_FEED_ON, state_current=state)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "runtime_feed_read_count=0" in r.detail


def test_candidate_b_insufficient_below_reads_floor():
    state = {
        "runtime_feed_read_count": 50,  # < 200 floor
        "microprice_widen_z_runtime_feed_fires": 10,
    }
    r = check_v1_5_288_candidate_b_microprice_z_fires(
        _snap(bot_version="1.5.288", config=_FEED_ON, state_current=state)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "floor" in r.detail


def test_candidate_b_pass_at_or_above_5pct():
    state = {
        "runtime_feed_read_count": 1000,
        "microprice_widen_z_runtime_feed_fires": 100,  # 10%
    }
    r = check_v1_5_288_candidate_b_microprice_z_fires(
        _snap(bot_version="1.5.288", config=_FEED_ON, state_current=state)
    )
    assert r.status == "PASS"
    assert r.measured["fire_rate"] == 0.1


def test_candidate_b_warn_below_5pct():
    state = {
        "runtime_feed_read_count": 1000,
        "microprice_widen_z_runtime_feed_fires": 10,  # 1%
    }
    r = check_v1_5_288_candidate_b_microprice_z_fires(
        _snap(bot_version="1.5.288", config=_FEED_ON, state_current=state)
    )
    assert r.status == "WARN"
    assert "below the 5% demonstration threshold" in r.detail


# ---------------------------------------------------------------------------
# v1.5.294 BUG-028 — books5 liveness heartbeat
# ---------------------------------------------------------------------------


# --- AC1: book-age hold ceiling reverted to 2000 ---


def test_v1_5_294_book_age_na_on_old_version():
    r = check_v1_5_294_book_age_hold_reverted(
        _snap(bot_version="1.5.293", config={"QUOTE_HOLD_MAX_BOOK_AGE_MS": 2000})
    )
    assert r.status == "N/A"
    assert "predates v1.5.294" in r.detail


def test_v1_5_294_book_age_insufficient_when_config_missing():
    r = check_v1_5_294_book_age_hold_reverted(
        _snap(bot_version="1.5.294", config=None)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "config.json missing" in r.detail


def test_v1_5_294_book_age_insufficient_when_key_absent():
    r = check_v1_5_294_book_age_hold_reverted(
        _snap(bot_version="1.5.294", config={})
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "QUOTE_HOLD_MAX_BOOK_AGE_MS absent" in r.detail


def test_v1_5_294_book_age_passes_at_2000():
    r = check_v1_5_294_book_age_hold_reverted(
        _snap(bot_version="1.5.294", config={"QUOTE_HOLD_MAX_BOOK_AGE_MS": 2000})
    )
    assert r.status == "PASS"
    assert r.measured["quote_hold_max_book_age_ms"] == 2000.0


def test_v1_5_294_book_age_warns_when_band_aid_present():
    # The v1.5.80 band-aid 3500 ms should no longer be needed once the
    # heartbeat lands — flag it (WARN, not FAIL: operator owns tuning).
    r = check_v1_5_294_book_age_hold_reverted(
        _snap(bot_version="1.5.294", config={"QUOTE_HOLD_MAX_BOOK_AGE_MS": 3500})
    )
    assert r.status == "WARN"
    assert "3500" in r.detail


def test_v1_5_294_book_age_warns_when_tighter():
    r = check_v1_5_294_book_age_hold_reverted(
        _snap(bot_version="1.5.294", config={"QUOTE_HOLD_MAX_BOOK_AGE_MS": 1500})
    )
    assert r.status == "WARN"
    assert "1500" in r.detail


# --- AC2: combined-feed gap p95 healthy ---
#
# v1.5.294 fix repointed this check from a non-snapshotted
# ``quote_eligibility_guard`` block to ``stats/market-data_gap-stats.json``
# (``p95_gap_ms``) — the SAME ring the freshness gate consults. The
# helper writes that artifact into a real temp snapshot dir.


def _gap_snap(
    tmp_path: Path,
    *,
    bot_version: str = "1.5.294",
    p95: Any = 150.0,
    gap_count: Any = 5000,
    source_type: str = "public_ws",
    write_file: bool = True,
    config: Any = None,
) -> SnapshotData:
    if write_file:
        stats = tmp_path / "stats"
        stats.mkdir(parents=True, exist_ok=True)
        payload = {
            "p95_gap_ms": p95,
            "gap_count": gap_count,
            "source_type": source_type,
        }
        (stats / "market-data_gap-stats.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return _snap(bot_version=bot_version, config=config, snapshot_dir=tmp_path)


def test_v1_5_294_gap_p95_na_on_old_version(tmp_path):
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, bot_version="1.5.293")
    )
    assert r.status == "N/A"


def test_v1_5_294_gap_p95_insufficient_when_file_missing(tmp_path):
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, write_file=False)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "market-data_gap-stats.json" in r.detail


def test_v1_5_294_gap_p95_insufficient_when_null(tmp_path):
    # Ring not warmed up (session too short) → p95 is null.
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, p95=None)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "not warmed up" in r.detail


def test_v1_5_294_gap_p95_insufficient_when_few_samples(tmp_path):
    # Below the 200-sample floor → p95 noisy, gate ignores it.
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, p95=150.0, gap_count=50)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "ring not warmed" in r.detail


def test_v1_5_294_gap_p95_passes_when_low(tmp_path):
    # books5 ~100 ms → p95 well under the 600 ms healthy reference.
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, p95=150.0)
    )
    assert r.status == "PASS"
    assert r.measured["healthy_ref_ms"] == 600.0
    assert r.measured["in_force_hold_threshold_ms"] == 600.0
    assert r.measured["source_type"] == "public_ws"


def test_v1_5_294_gap_p95_passes_under_lenient_inforce(tmp_path):
    # The real v1.5.294 prod scenario: p95=305.7 ms, in-force hold
    # deliberately lenient at 10000 ms. 305.7 < 600 healthy ref → PASS,
    # and the lenient gate is nowhere near holding (~3% of it).
    cfg = {"QUOTE_HOLD_MAX_GAP_P95_MS": 10000}
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, p95=305.704, gap_count=65000, config=cfg)
    )
    assert r.status == "PASS"
    assert r.measured["in_force_hold_threshold_ms"] == 10000.0
    assert r.measured["p95_share_of_in_force_hold"] < 0.05


def test_v1_5_294_gap_p95_warns_when_elevated_but_lenient(tmp_path):
    # p95=800 ms exceeds the 600 ms healthy ref but the lenient 10000 ms
    # in-force gate won't hold → WARN (heartbeat not as tight as it should
    # be, but no hold).
    cfg = {"QUOTE_HOLD_MAX_GAP_P95_MS": 10000}
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, p95=800.0, config=cfg)
    )
    assert r.status == "WARN"
    assert "healthy-feed reference" in r.detail


def test_v1_5_294_gap_p95_fails_at_inforce_threshold(tmp_path):
    # With the default 600 ms in-force hold, p95=700 >= 600 → the gate is
    # actually holding → FAIL.
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, p95=700.0)
    )
    assert r.status == "FAIL"
    assert "going silent enough to hold" in r.detail


def test_v1_5_294_gap_p95_sample_floor_respects_config(tmp_path):
    # Operator can lower the min-samples floor; 100 >= 50 → not blocked.
    cfg = {"QUOTE_FRESHNESS_P95_MIN_GAP_SAMPLES": 50}
    r = check_v1_5_294_combined_feed_gap_p95_healthy(
        _gap_snap(tmp_path, p95=150.0, gap_count=100, config=cfg)
    )
    assert r.status == "PASS"


# --- AC3: freshness no longer dominates suppression ---
#
# v1.5.294 fix repointed this check from non-snapshotted
# ``quote_elig_hold_*`` counters to the snapshotted
# ``state_current['quote_quality']['suppression_reason_counts_session']``
# tally. Freshness's contribution is the ``eligibility:freshness_drift_
# hold`` key; the denominator is the sum of all suppression reasons.

_FRESH_KEY = "eligibility:freshness_drift_hold"


def _supp(counts: dict) -> dict:
    return {"quote_quality": {"suppression_reason_counts_session": counts}}


# A faithful copy of the v1.5.294 prod snapshot's tally (total 19261,
# freshness 40 → 0.2%). Used as a regression fixture.
_PROD_SUPP = {
    "engine:inventory_exec_bias_bid": 112,
    "quoting:post_fill_cooldown_ask": 833,
    "quoting:post_fill_cooldown_bid": 739,
    "quoting:soft_skew_long": 5666,
    "quoting:soft_skew_short": 2650,
    "quoting:at_touch_adverse_pause_bid": 564,
    "quoting:at_max_long": 2396,
    "eligibility:freshness_drift_hold": 40,
    "quoting:at_max_short": 3877,
    "quoting:at_touch_adverse_pause_ask": 2384,
}


def test_v1_5_294_freshness_share_na_on_old_version():
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.293", state_current=_supp(_PROD_SUPP))
    )
    assert r.status == "N/A"


def test_v1_5_294_freshness_share_insufficient_when_state_missing():
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.294", state_current=None)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "state_current.json missing" in r.detail


def test_v1_5_294_freshness_share_insufficient_when_quote_quality_missing():
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.294", state_current={})
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "quote_quality block missing" in r.detail


def test_v1_5_294_freshness_share_insufficient_when_counts_missing():
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.294", state_current={"quote_quality": {}})
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "suppression_reason_counts_session missing" in r.detail


def test_v1_5_294_freshness_share_insufficient_small_denominator():
    # Total 30 suppressions — too noisy to judge the share.
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.294", state_current=_supp({_FRESH_KEY: 20, "x": 10}))
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "too few" in r.detail


def test_v1_5_294_freshness_share_passes_on_prod_fixture():
    # Real prod data: freshness 40 / 19261 total = 0.2% — strong PASS.
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.294", state_current=_supp(_PROD_SUPP))
    )
    assert r.status == "PASS"
    assert r.measured["freshness_hold_count"] == 40
    assert r.measured["total_suppression_count"] == 19261
    assert r.measured["freshness_is_top_reason"] is False
    assert r.measured["top_reason"] == "quoting:soft_skew_long"


def test_v1_5_294_freshness_share_passes_when_key_absent():
    # Freshness gate never fired → key absent → 0 share → PASS.
    counts = {"quoting:soft_skew_long": 500, "quoting:at_max_short": 300}
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.294", state_current=_supp(counts))
    )
    assert r.status == "PASS"
    assert r.measured["freshness_hold_count"] == 0


def test_v1_5_294_freshness_share_warns_when_moderate():
    # 300/1000 = 30% — elevated (15% <= share < 35%).
    counts = {_FRESH_KEY: 300, "quoting:soft_skew_long": 700}
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.294", state_current=_supp(counts))
    )
    assert r.status == "WARN"


def test_v1_5_294_freshness_share_fails_when_dominant():
    # 700/1000 = 70% — the pre-fix BUG-028 signature.
    counts = {_FRESH_KEY: 700, "quoting:soft_skew_long": 300}
    r = check_v1_5_294_freshness_not_dominating_holds(
        _snap(bot_version="1.5.294", state_current=_supp(counts))
    )
    assert r.status == "FAIL"
    assert "signature" in r.detail
    assert r.measured["freshness_is_top_reason"] is True


# --- markout coverage (the audit §5 P0 #1 HEADLINE criterion) ---
#
# Encodes "markout coverage on the session's fills > 95%" — the outcome
# AQC's net-edge target (rebate + markout) actually depends on. Pre-fix
# the silent bbo-tbt feed left state.market stale so the per-tick
# delayed-markout sampler never resolved its horizons; v1.5.294's books5
# heartbeat keeps the live mid fresh. Measured 0%→100% across the deploy.


def _markout_fills(n_settled, *, coverage=1.0, miss_share=0.0, n_tail=5):
    """Build a synthetic ``fills_since`` list for the markout-coverage
    check. ``n_settled`` fills are spaced 1 s apart and sit comfortably
    behind the anchor (so all count as *settled*); ``n_tail`` recent fills
    sit within ``TAIL_S`` of the anchor (so all are *excluded*). ``coverage``
    is the fraction of SETTLED fills carrying a non-null ``markout_5s_bps``;
    ``miss_share`` the fraction tagged ``book_reference_quality ==
    'missing_reference'``."""
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)
    n_covered = round(n_settled * coverage)
    n_miss = round(n_settled * miss_share)
    fills = []
    for i in range(n_settled):
        fills.append({
            "ts_fill": (base + timedelta(seconds=i)).isoformat(),
            "markout_5s_bps": (0.5 if i < n_covered else None),
            "book_reference_quality": (
                "missing_reference" if i < n_miss else "exact_or_prior"),
        })
    # Tail fills 20 s after the last settled fill → settled fills are all
    # >= TAIL_S behind the anchor; the tail itself is within TAIL_S and
    # is excluded (genuinely-unresolved 5 s horizon).
    tail_base = base + timedelta(seconds=(n_settled - 1) + 20)
    for j in range(n_tail):
        fills.append({
            "ts_fill": (tail_base + timedelta(seconds=j)).isoformat(),
            "markout_5s_bps": None,
            "book_reference_quality": "exact_or_prior",
        })
    return fills


def test_v1_5_294_markout_coverage_na_on_old_version():
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.293", fills_since=_markout_fills(30))
    )
    assert r.status == "N/A"
    assert "predates v1.5.294" in r.detail


def test_v1_5_294_markout_coverage_insufficient_when_fills_missing():
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294", fills_since=None)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "empty / missing" in r.detail


def test_v1_5_294_markout_coverage_insufficient_when_no_parseable_ts():
    fills = [{"markout_5s_bps": 0.5}, {"markout_5s_bps": None}]
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294", fills_since=fills)
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "no parseable ts_fill" in r.detail


def test_v1_5_294_markout_coverage_insufficient_too_few_settled():
    # 10 settled fills < MIN_SETTLED (20) → can't judge.
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294", fills_since=_markout_fills(10))
    )
    assert r.status == "INSUFFICIENT_DATA"
    assert "settled fills" in r.detail
    assert r.measured is None or "settled_fills" not in (r.measured or {})


def test_v1_5_294_markout_coverage_passes_at_full_coverage():
    # The post-deploy reality: 100% coverage, low missing_reference share.
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294",
              fills_since=_markout_fills(30, coverage=1.0, miss_share=0.07))
    )
    assert r.status == "PASS"
    assert r.measured["markout_5s_coverage"] == 1.0
    assert r.measured["settled_fills"] == 30
    assert r.measured["missing_reference_share"] < 0.1


def test_v1_5_294_markout_coverage_tail_fills_excluded():
    # A burst of recent (unsettled) fills must NOT drag coverage down — all
    # within TAIL_S of the anchor are excluded even though markout is null.
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)
    # 30 settled, fully-covered fills 1 s apart.
    fills = [
        {"ts_fill": (base + timedelta(seconds=i)).isoformat(),
         "markout_5s_bps": 0.5, "book_reference_quality": "exact_or_prior"}
        for i in range(30)
    ]
    # 25 recent null-markout fills packed 0.2 s apart, all inside the last
    # ~5 s → all < TAIL_S behind the anchor → all excluded.
    burst0 = base + timedelta(seconds=49)
    fills += [
        {"ts_fill": (burst0 + timedelta(seconds=0.2 * k)).isoformat(),
         "markout_5s_bps": None, "book_reference_quality": "exact_or_prior"}
        for k in range(25)
    ]
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294", fills_since=fills)
    )
    assert r.status == "PASS"
    assert r.measured["settled_fills"] == 30  # 25 burst fills excluded


def test_v1_5_294_markout_coverage_warns_high_miss_is_bug028_signature():
    # 90% coverage with a high missing_reference share → WARN that points
    # at the BUG-028 book-staleness path.
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294",
              fills_since=_markout_fills(30, coverage=0.9, miss_share=0.5))
    )
    assert r.status == "WARN"
    assert "BUG-028 signature" in r.detail
    assert r.measured["missing_reference_share"] == 0.5


def test_v1_5_294_markout_coverage_warns_low_miss_points_elsewhere():
    # 90% coverage but LOW missing_reference share → WARN that explicitly
    # rules out BUG-028 and points at deque overflow / restart churn.
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294",
              fills_since=_markout_fills(30, coverage=0.9, miss_share=0.0))
    )
    assert r.status == "WARN"
    assert "NOT the BUG-028" in r.detail
    assert "deque overflow" in r.detail


def test_v1_5_294_markout_coverage_fails_when_sparse():
    # 50% coverage → markout signal too sparse for AQC net-edge calibration.
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294",
              fills_since=_markout_fills(30, coverage=0.5))
    )
    assert r.status == "FAIL"
    assert r.measured["markout_5s_coverage"] == 0.5
    assert "P0 #1" in r.detail


def test_v1_5_294_markout_coverage_uses_ts_fallback():
    # Rows without ts_fill fall back to the `ts` key.
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)
    fills = [
        {"ts": (base + timedelta(seconds=i)).isoformat(),
         "markout_5s_bps": 0.5, "book_reference_quality": "exact_or_prior"}
        for i in range(25)
    ]
    # one tail fill far ahead so the 25 above are all settled
    fills.append({"ts": (base + timedelta(seconds=60)).isoformat(),
                  "markout_5s_bps": None,
                  "book_reference_quality": "exact_or_prior"})
    r = check_v1_5_294_markout_coverage(
        _snap(bot_version="1.5.294", fills_since=fills)
    )
    assert r.status == "PASS"
    assert r.measured["settled_fills"] == 25
