"""Phase 2H per-deploy acceptance gate — synthetic-snapshot tests.

For each of the six metrics, a synthetic snapshot that trips the
metric asserts the gate fails, and a clean synthetic snapshot
asserts the gate passes. Pure-function tests against the helpers in
``tools.postmortem.sections.acceptance_gate`` — no snapshot files
on disk, no real bot, no exchange.

Background — why this test file exists:
=======================================

Phase 2H of the defense plan ships a numeric guard that catches
release regressions before merge. The guard exists BECAUSE we've
seen this class of regression slip through manual review twice:

* **v1.4.118 emergency** — config dial-down trapped the bot in
  DEFENSIVE 99% of session; fill rate collapsed.
* **v1.4.189 phase-ladder catastrophe** — SF dispatcher fired 1,249
  orders in 13s, ended with a terminal taker at the worst price.

The acceptance gate's six checks are designed to flag the exact
footprints of those two incidents (and similar ones). This test
file pins the check logic against synthesised fixtures that
reproduce the failure footprints — if a future refactor accidentally
defangs a check, the corresponding regression test fires.
"""

from __future__ import annotations

import pytest

from tools.postmortem.sections.acceptance_gate import (
    AcceptanceGateFindings,
    check_bounded_stale_quote_age,
    check_no_fills_during_forbidden_states,
    check_no_inventory_runaway,
    check_per_side_adverse_fill_rate,
    check_per_side_markout_floor,
    check_stable_account_fill_truth,
    detect_acceptance_gate_findings,
)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _clean_fill(
    *,
    side: str = "BUY",
    markout_5s_bps: float = -1.5,
    quote_age_at_fill_ms: float = 800.0,
    quote_eligibility_state: str = "QUOTE_BOTH",
    inventory_qty_before_fill: float = 0.0,
    size: float = 1.0,
    fill_id: str = "fill-x",
) -> dict:
    """One clean fill row — passes all metrics. Helper builds the
    minimal field set the check functions read. Real fills_since.json
    rows carry many more fields; tests only need the ones the gate
    consults."""
    return {
        "fill_id": fill_id,
        "side": side,
        "markout_5s_bps": markout_5s_bps,
        "quote_age_at_fill_ms": quote_age_at_fill_ms,
        "quote_eligibility_state": quote_eligibility_state,
        "inventory_qty_before_fill": inventory_qty_before_fill,
        "size": size,
    }


def _clean_fills(n: int = 60, side_split: float = 0.5) -> list[dict]:
    """``n`` fills total, split into BUY/SELL by ``side_split``.
    Markouts alternate between -1.5 and +1.0 bps so per-side mean
    is -0.25 (well above the -3.0 floor) and adverse rate is 50 %
    (below the 70 % ceiling). Quote ages all 800 ms (well below P99
    floor)."""
    n_buy = int(n * side_split)
    out: list[dict] = []
    for i in range(n_buy):
        out.append(_clean_fill(
            side="BUY",
            markout_5s_bps=-1.5 if i % 2 == 0 else 1.0,
            fill_id=f"b{i}",
        ))
    for i in range(n - n_buy):
        out.append(_clean_fill(
            side="SELL",
            markout_5s_bps=-1.5 if i % 2 == 0 else 1.0,
            fill_id=f"s{i}",
        ))
    return out


def _clean_config() -> dict:
    """Config with ``MAX_ABS_POSITION=10`` to anchor the runaway
    check's ceiling."""
    return {"MAX_ABS_POSITION": 10.0}


# ---------------------------------------------------------------------------
# Metric 1 — no fills during forbidden quote-eligibility states
# ---------------------------------------------------------------------------


def test_metric1_clean_session_passes() -> None:
    """A normal session with QUOTE_BOTH at every fill passes."""
    r = check_no_fills_during_forbidden_states(_clean_fills())
    assert r.passed is True
    assert r.severity == "fatal"


def test_metric1_hold_all_fill_fails() -> None:
    """A single fill with HOLD_ALL eligibility fails fatally."""
    fills = _clean_fills(60)
    # Trip one fill into HOLD_ALL state.
    fills[3]["quote_eligibility_state"] = (
        "HOLD_ALL|fresh=freshness_ok|drift=drift_ok"
    )
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is False
    assert r.severity == "fatal"
    assert "HOLD_ALL" in r.detail


def test_metric1_kill_state_fails() -> None:
    fills = [_clean_fill(quote_eligibility_state="KILL")]
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is False
    assert r.severity == "fatal"
    assert "KILL" in r.detail


def test_metric1_recovery_cooldown_fails() -> None:
    fills = [_clean_fill(quote_eligibility_state="RECOVERY_COOLDOWN")]
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is False


def test_metric1_quote_sell_only_passes() -> None:
    """SHOCK locks to SELL_ONLY or BUY_ONLY — those are NOT
    forbidden states (the bot legitimately quoted there)."""
    fills = [
        _clean_fill(
            side="SELL",
            quote_eligibility_state=(
                "QUOTE_SELL_ONLY|shock_gate_locked:..."
            ),
        )
    ]
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is True


def test_metric1_no_fills_returns_info_pass() -> None:
    """Zero fills → can't fail; returns info-tier pass."""
    r = check_no_fills_during_forbidden_states([])
    assert r.passed is True
    assert r.severity == "info"


def test_metric1_missing_fills_returns_info_pass() -> None:
    r = check_no_fills_during_forbidden_states(None)
    assert r.passed is True
    assert r.severity == "info"


# ---------------------------------------------------------------------------
# v1.5.28 -- SF taker closes during HOLD_ALL: informational, NOT fatal
# ---------------------------------------------------------------------------


def test_metric1_sf_taker_close_during_hold_all_does_not_fail() -> None:
    """v1.5.28: when multiple defensive gates (shock_gate + mae_gate +
    recovery_cooldown) compose at the same tick, side-set intersection
    collapses to HOLD_ALL. An SF emergency-close taker fill carrying
    that composed HOLD_ALL state is INTENDED behaviour -- SF bypasses
    quote eligibility by design so the bot can always flatten an
    adverse position. Pre-fix the gate falsely FAILed on every such
    SF episode. Both conditions required: ``liquidity_flag=crossed``
    AND ``soft_flatten_event_id`` non-null."""
    fills = [
        _clean_fill(
            quote_eligibility_state="HOLD_ALL|recovery_cooldown|mae_gate|shock_gate_locked",
            fill_id="sf-150879622",
        )
    ]
    # Add the SF taker-close discriminator fields.
    fills[0]["liquidity_flag"] = "crossed"
    fills[0]["soft_flatten_event_id"] = 11192

    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is True, (
        f"SF taker close during HOLD_ALL must NOT fail the gate. detail={r.detail}"
    )
    # But the report must still SURFACE the SF activity so the
    # operator can see it happened.
    assert "informational" in r.detail.lower()
    assert "SF#11192" in r.detail
    assert "1 SF taker close" in r.detail


def test_metric1_passive_resting_fill_during_hold_all_still_fails() -> None:
    """The cancel-chain-leak pattern stays a fatal failure: a
    PASSIVE (``liquidity_flag=resting``) fill during HOLD_ALL means
    the bot had a maker order on the book past a gate fire and the
    cancel chain failed to clear it before a taker hit. SF tagging
    is irrelevant here; resting + HOLD_ALL = real bug."""
    fills = [
        _clean_fill(
            quote_eligibility_state="HOLD_ALL|some_reason",
            fill_id="leak-fill",
        )
    ]
    fills[0]["liquidity_flag"] = "resting"  # passive maker
    fills[0]["soft_flatten_event_id"] = None
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is False
    assert r.severity == "fatal"
    assert "HOLD_ALL" in r.detail


def test_metric1_resting_fill_with_sf_tag_during_hold_all_still_fails() -> None:
    """Defence-in-depth: even when an SF tag is stamped on the fill,
    if liquidity_flag is ``resting`` (not ``crossed``) it's still a
    passive maker fill and the cancel-chain-leak interpretation
    applies. SF taker closes have ``crossed`` because they cross the
    spread; a resting+SF-tagged combination shouldn't happen at all,
    and if it does the gate is right to flag it for investigation."""
    fills = [
        _clean_fill(
            quote_eligibility_state="HOLD_ALL|composed_gates",
            fill_id="weird-fill",
        )
    ]
    fills[0]["liquidity_flag"] = "resting"
    fills[0]["soft_flatten_event_id"] = 99999
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is False, (
        "resting fill (passive maker) during HOLD_ALL must fail "
        "regardless of SF tagging"
    )


def test_metric1_sf_taker_without_sf_id_during_hold_all_still_fails() -> None:
    """Defence-in-depth: a taker fill (``liquidity_flag=crossed``)
    that LACKS an ``soft_flatten_event_id`` is not an SF-managed
    close. Could be a stray ``client.market_close`` call outside
    the SF flow OR a missed SF tagging. Either way the gate should
    surface it as an offender for the operator to investigate."""
    fills = [
        _clean_fill(
            quote_eligibility_state="HOLD_ALL|some_gate",
            fill_id="untagged-taker",
        )
    ]
    fills[0]["liquidity_flag"] = "crossed"
    fills[0]["soft_flatten_event_id"] = None
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is False
    assert r.severity == "fatal"


def test_metric1_mixed_genuine_offender_plus_sf_takers_fails() -> None:
    """When the offender list has BOTH a real leak AND SF taker
    closes, the gate must FAIL on the real leak and ALSO surface
    the SF count informationally. The SF count should never mask a
    genuine failure."""
    fills = [
        # Real leak: passive resting maker fill during HOLD_ALL.
        {
            **_clean_fill(
                quote_eligibility_state="HOLD_ALL|leak",
                fill_id="real-leak",
            ),
            "liquidity_flag": "resting",
            "soft_flatten_event_id": None,
        },
        # Informational: SF taker close.
        {
            **_clean_fill(
                quote_eligibility_state="HOLD_ALL|sf",
                fill_id="sf-fill-1",
            ),
            "liquidity_flag": "crossed",
            "soft_flatten_event_id": 7777,
        },
        {
            **_clean_fill(
                quote_eligibility_state="HOLD_ALL|sf",
                fill_id="sf-fill-2",
            ),
            "liquidity_flag": "crossed",
            "soft_flatten_event_id": 7777,
        },
    ]
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is False, (
        "real leak must still fail even when accompanied by SF takers"
    )
    assert r.severity == "fatal"
    # The detail mentions BOTH offender count and SF count.
    assert "forbidden=1" in r.detail
    assert "real-leak" in r.detail
    assert "2 SF taker closes" in r.detail


def test_metric1_clean_session_with_zero_sf_omits_info_suffix() -> None:
    """A normal clean session shows ``forbidden=0`` with NO informational
    SF suffix (suffix only renders when SF takers are present)."""
    r = check_no_fills_during_forbidden_states(_clean_fills())
    assert r.passed is True
    assert "forbidden=0" in r.detail
    assert "informational" not in r.detail.lower()


def test_metric1_clean_session_with_sf_taker_during_hold_renders_info() -> None:
    """A session with no genuine leaks BUT containing SF taker
    closes during HOLD_ALL passes and emits the informational suffix.
    This is the exact pattern from snapshot v1.5.25-260522-203815."""
    fills = _clean_fills(60)
    # Inject one SF taker close.
    fills.append({
        **_clean_fill(
            quote_eligibility_state="HOLD_ALL|recovery_cooldown|mae_gate",
            fill_id="sf-real-case",
        ),
        "liquidity_flag": "crossed",
        "soft_flatten_event_id": 11192,
    })
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is True
    assert "informational" in r.detail.lower()
    assert "SF#11192" in r.detail


# ---------------------------------------------------------------------------
# v1.5.33 — TP maker closes during HOLD_ALL: informational, NOT fatal
# ---------------------------------------------------------------------------


def test_metric1_tp_maker_close_during_hold_all_does_not_fail() -> None:
    """v1.5.33: take-profit (TP) is an overlay that, like SF, bypasses
    quote eligibility by design. A maker fill carrying a non-null
    ``tp_event_id`` during a forbidden state is the TP harvest path,
    NOT a cancel-chain leak. Unlike SF (taker), TP is post-only by
    design -- no ``crossed`` requirement. The fingerprint is simply
    the tp_event_id FK."""
    fills = [
        {
            **_clean_fill(
                quote_eligibility_state="HOLD_ALL|recovery_cooldown",
                fill_id="tp-fill-1",
            ),
            "liquidity_flag": "resting",  # maker, post-only
            "tp_event_id": 42,
        }
    ]
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is True, (
        f"TP maker close during HOLD_ALL must NOT fail the gate. "
        f"detail={r.detail}"
    )
    assert "informational" in r.detail.lower()
    assert "TP#42" in r.detail
    assert "1 TP maker close" in r.detail


def test_metric1_tp_during_hold_alongside_sf_renders_both_info() -> None:
    """Mixed informational case: a HOLD_ALL window contains BOTH an SF
    taker close and a TP maker close. Both surface informationally
    in separate sub-clauses; gate passes."""
    fills = [
        {
            **_clean_fill(
                quote_eligibility_state="HOLD_ALL|sf",
                fill_id="sf-fill",
            ),
            "liquidity_flag": "crossed",
            "soft_flatten_event_id": 7777,
        },
        {
            **_clean_fill(
                quote_eligibility_state="HOLD_ALL|tp",
                fill_id="tp-fill",
            ),
            "liquidity_flag": "resting",
            "tp_event_id": 8888,
        },
    ]
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is True
    assert "SF#7777" in r.detail
    assert "TP#8888" in r.detail


def test_metric1_passive_resting_fill_without_tp_id_still_fails() -> None:
    """Defence-in-depth: a resting fill during HOLD_ALL WITHOUT a
    tp_event_id (or sf_event_id) is the genuine cancel-chain leak.
    Must fail."""
    fills = [
        {
            **_clean_fill(
                quote_eligibility_state="HOLD_ALL|leak",
                fill_id="leak-fill",
            ),
            "liquidity_flag": "resting",
            "tp_event_id": None,
            "soft_flatten_event_id": None,
        }
    ]
    r = check_no_fills_during_forbidden_states(fills)
    assert r.passed is False
    assert r.severity == "fatal"


# ---------------------------------------------------------------------------
# Metric 2 — bounded stale-quote age distribution
# ---------------------------------------------------------------------------


def test_metric2_clean_quote_ages_passes() -> None:
    """All quote ages well below thresholds → pass."""
    fills = _clean_fills(60)  # all 800 ms
    r = check_bounded_stale_quote_age(fills)
    assert r.passed is True
    assert r.severity == "fatal"


def test_metric2_p99_breach_fails() -> None:
    """One fill with quote_age above the 5000 ms P99 floor fails
    fatally — but ONLY if it lands above P99. With 12 fills, the
    99th percentile lands near the max, so one large value trips it."""
    fills = [
        _clean_fill(quote_age_at_fill_ms=500.0, fill_id=f"f{i}")
        for i in range(11)
    ]
    fills.append(_clean_fill(quote_age_at_fill_ms=8000.0, fill_id="bad"))
    r = check_bounded_stale_quote_age(fills)
    assert r.passed is False
    assert r.severity == "fatal"


def test_metric2_p999_breach_fails() -> None:
    """A single fill with quote_age above the 30 s P99.9 floor on a
    large sample still fails (P99 stays within bounds; P99.9 spike
    catches it). 100 clean fills + 1 outlier → P99.9 ≈ outlier."""
    fills = [
        _clean_fill(quote_age_at_fill_ms=800.0, fill_id=f"f{i}")
        for i in range(100)
    ]
    fills.append(_clean_fill(quote_age_at_fill_ms=35000.0, fill_id="bad"))
    r = check_bounded_stale_quote_age(fills)
    assert r.passed is False


def test_metric2_insufficient_samples_returns_info_pass() -> None:
    """n < 10 fills can't yield a meaningful P99."""
    fills = [_clean_fill(quote_age_at_fill_ms=99999.0)] * 5
    r = check_bounded_stale_quote_age(fills)
    assert r.passed is True
    assert r.severity == "info"


def test_metric2_custom_thresholds_honoured() -> None:
    """Lower P99 ceiling forces a previously-passing session to fail.
    Useful for stricter operator profiles."""
    fills = [
        _clean_fill(quote_age_at_fill_ms=800.0, fill_id=f"f{i}")
        for i in range(60)
    ]
    # Default ceilings: passes.
    assert check_bounded_stale_quote_age(fills).passed is True
    # Tightened to 500 ms: fails.
    r = check_bounded_stale_quote_age(fills, p99_ms=500.0)
    assert r.passed is False


# ---------------------------------------------------------------------------
# Metric 3 — per-side mean markout floor
# ---------------------------------------------------------------------------


def test_metric3_clean_markouts_pass() -> None:
    """Per-side mean -0.25 bp > -3.0 floor → pass."""
    r = check_per_side_markout_floor(_clean_fills(60))
    assert r.passed is True
    assert r.severity == "warn"


def test_metric3_sell_side_breach_fails_warn() -> None:
    """Build a session where SELL markout averages -5 bp.
    Mirrors the v1.4.180 footprint (SELL avg -5.18 bp)."""
    fills = _clean_fills(60)
    # Overwrite SELL fills with -5 bp markouts.
    for f in fills:
        if f["side"] == "SELL":
            f["markout_5s_bps"] = -5.0
    r = check_per_side_markout_floor(fills)
    assert r.passed is False
    assert r.severity == "warn"
    assert "SELL=" in r.detail


def test_metric3_underpowered_side_returns_info_pass() -> None:
    """If neither side hits 30 fills → info-tier pass, no claim."""
    fills = _clean_fills(20)  # 10 BUY + 10 SELL
    r = check_per_side_markout_floor(fills)
    assert r.passed is True
    assert r.severity == "info"


def test_metric3_one_side_powered_other_not() -> None:
    """Mixed: BUY has 30+ fills, SELL has < 30. The check should
    still verdict on BUY but flag SELL as underpowered."""
    fills = (
        [_clean_fill(side="BUY", fill_id=f"b{i}") for i in range(40)]
        + [_clean_fill(side="SELL", fill_id=f"s{i}") for i in range(10)]
    )
    r = check_per_side_markout_floor(fills)
    # BUY is at -0.25 bp average → above floor → passes.
    assert r.passed is True


# ---------------------------------------------------------------------------
# Metric 4 — per-side adverse-fill rate ceiling
# ---------------------------------------------------------------------------


def test_metric4_clean_rates_pass() -> None:
    """50 % adverse rate is below the 70 % ceiling."""
    r = check_per_side_adverse_fill_rate(_clean_fills(60))
    assert r.passed is True
    assert r.severity == "warn"


def test_metric4_all_adverse_fails_warn() -> None:
    """100 % adverse on one side fails warn."""
    fills = _clean_fills(60)
    for f in fills:
        if f["side"] == "SELL":
            f["markout_5s_bps"] = -1.0  # always adverse
    r = check_per_side_adverse_fill_rate(fills)
    assert r.passed is False
    assert "SELL=" in r.detail


def test_metric4_underpowered_returns_info_pass() -> None:
    fills = _clean_fills(20)
    r = check_per_side_adverse_fill_rate(fills)
    assert r.severity == "info"


# ---------------------------------------------------------------------------
# Metric 5 — no inventory runaway
# ---------------------------------------------------------------------------


def test_metric5_within_cap_passes() -> None:
    """Max position = 9, cap = 10, ceiling = 10.5 → passes."""
    fills = _clean_fills(60)
    # Tweak one fill to have max position 9.
    fills[0]["inventory_qty_before_fill"] = 9.0
    fills[0]["size"] = 0.0
    inv = [{"position_qty": 9.0}]
    r = check_no_inventory_runaway(fills, inv, _clean_config())
    assert r.passed is True
    assert r.severity == "fatal"


def test_metric5_at_ceiling_passes() -> None:
    """Exactly at 5% overshoot — still passes."""
    inv = [{"position_qty": 10.5}]
    r = check_no_inventory_runaway([], inv, _clean_config())
    assert r.passed is True


def test_metric5_beyond_ceiling_fails() -> None:
    """11.0 > 10.5 ceiling → fail fatal. Position cap leak."""
    inv = [{"position_qty": 11.0}]
    r = check_no_inventory_runaway([], inv, _clean_config())
    assert r.passed is False
    assert r.severity == "fatal"


def test_metric5_picks_up_post_fill_state() -> None:
    """After-state computed from before + signed-size. A BUY fill of
    size 2 from before=10 gives after=12 → ceiling breach."""
    fills = [
        _clean_fill(side="BUY", inventory_qty_before_fill=10.0, size=2.0)
    ]
    r = check_no_inventory_runaway(fills, None, _clean_config())
    assert r.passed is False


def test_metric5_missing_config_returns_info_pass() -> None:
    """No MAX_ABS_POSITION in config → info-tier pass, no claim."""
    r = check_no_inventory_runaway(_clean_fills(60), None, None)
    assert r.severity == "info"
    assert r.passed is True


def test_metric5_no_data_returns_info_pass() -> None:
    r = check_no_inventory_runaway(None, None, _clean_config())
    assert r.severity == "info"


# ---------------------------------------------------------------------------
# Metric 6 — stable account/fill truth
# ---------------------------------------------------------------------------


def test_metric6_clean_events_pass() -> None:
    events = [
        {"event_type": "market_data_refresh_success"},
        {"event_type": "quote_side_suppressed"},
        {"event_type": "shock_gate_fired"},
    ]
    r = check_stable_account_fill_truth(events)
    assert r.passed is True
    assert r.severity == "fatal"


def test_metric6_account_data_stale_fails() -> None:
    events = [
        {"event_type": "account_data_stale"},
        {"event_type": "market_data_refresh_success"},
    ]
    r = check_stable_account_fill_truth(events)
    assert r.passed is False
    assert "account_data_stale" in r.detail


def test_metric6_order_desync_detected_fails() -> None:
    events = [{"event_type": "order_desync_detected"}]
    r = check_stable_account_fill_truth(events)
    assert r.passed is False


def test_metric6_desync_detected_alias_fails() -> None:
    """Two synonymous event names — both should trip the check."""
    events = [{"event_type": "desync_detected"}]
    r = check_stable_account_fill_truth(events)
    assert r.passed is False


def test_metric6_missing_events_returns_info_pass() -> None:
    r = check_stable_account_fill_truth(None)
    assert r.severity == "info"
    assert r.passed is True


# ---------------------------------------------------------------------------
# Integration — detect_acceptance_gate_findings + AcceptanceGateFindings
# ---------------------------------------------------------------------------


def test_clean_snapshot_all_six_pass() -> None:
    """End-to-end: a clean synthesised snapshot returns ok=True
    with zero fatal failures."""
    findings = detect_acceptance_gate_findings(
        snapshot_name="synthetic-clean",
        bot_version="x.y.z",
        captured_at="2026-05-21T12:00:00Z",
        fills=_clean_fills(60),
        inventory_history=[{"position_qty": 5.0}],
        config=_clean_config(),
        events=[{"event_type": "market_data_refresh_success"}],
    )
    assert isinstance(findings, AcceptanceGateFindings)
    assert findings.ok is True
    assert findings.fatal_failures == []


def test_findings_ok_property_short_circuits_on_fatal() -> None:
    """A single fatal failure flips ``ok`` to False even if everything
    else passes."""
    fills = _clean_fills(60)
    fills[0]["quote_eligibility_state"] = "HOLD_ALL"
    findings = detect_acceptance_gate_findings(
        snapshot_name="synthetic-1fatal",
        bot_version="x.y.z",
        captured_at="2026-05-21T12:00:00Z",
        fills=fills,
        inventory_history=[{"position_qty": 5.0}],
        config=_clean_config(),
        events=[{"event_type": "market_data_refresh_success"}],
    )
    assert findings.ok is False
    assert len(findings.fatal_failures) == 1


def test_findings_warn_failures_do_not_break_ok() -> None:
    """warn-tier failures (markout / adverse-rate) are surfaced via
    ``warn_failures`` but DO NOT flip ``ok`` to False — the gate's
    contract is ``ok = no fatal failures``."""
    fills = _clean_fills(60)
    # Drive SELL markout down to -5 bp → metric 3 warn fail.
    for f in fills:
        if f["side"] == "SELL":
            f["markout_5s_bps"] = -5.0
    findings = detect_acceptance_gate_findings(
        snapshot_name="synthetic-warn",
        bot_version="x.y.z",
        captured_at="2026-05-21T12:00:00Z",
        fills=fills,
        inventory_history=[{"position_qty": 5.0}],
        config=_clean_config(),
        events=[{"event_type": "market_data_refresh_success"}],
    )
    assert findings.ok is True  # no fatal failures
    assert len(findings.warn_failures) >= 1


def test_findings_total_passed_count() -> None:
    """All six metrics in a clean run → total_passed == 6."""
    findings = detect_acceptance_gate_findings(
        snapshot_name="synthetic-clean",
        bot_version="x.y.z",
        captured_at="2026-05-21T12:00:00Z",
        fills=_clean_fills(60),
        inventory_history=[{"position_qty": 5.0}],
        config=_clean_config(),
        events=[{"event_type": "market_data_refresh_success"}],
    )
    assert findings.total_passed == 6


# ---------------------------------------------------------------------------
# Regression replays of the two real-world incidents
# ---------------------------------------------------------------------------


def test_regression_v1_4_189_phase_ladder_catastrophe_caught() -> None:
    """v1.4.189 (2026-05-21 10:29 UTC) — phase-ladder fired a terminal
    market_close with quote_age=7984ms. Real snapshot reproducer.
    The acceptance gate flags it on the quote-age P99 check."""
    # Two regular fills + one bad terminal taker. Mimics the SF#11172
    # phase-4 fill landing well above 5 s.
    fills = [
        _clean_fill(quote_age_at_fill_ms=2350.0, fill_id=f"f{i}")
        for i in range(20)
    ]
    fills.append(
        _clean_fill(
            quote_age_at_fill_ms=7984.0,
            fill_id="sf-terminal",
            markout_5s_bps=-7.5,
        )
    )
    r = check_bounded_stale_quote_age(fills)
    assert r.passed is False
    assert r.severity == "fatal"


def test_regression_v1_4_118_defensive_lockout_caught() -> None:
    """v1.4.118 (2026-05-20 16:00 UTC) — DEFENSIVE 99% of session;
    fill rate collapsed to 7/h. Per-side markout was significantly
    negative because the rare fills were adverse-selected.

    Reproducer: 30 BUYs all at -3.5 bp (just under the floor), 30
    SELLs at -4.0 bp. Acceptance gate flags both sides on metric 3."""
    fills = (
        [
            _clean_fill(side="BUY", markout_5s_bps=-3.5, fill_id=f"b{i}")
            for i in range(30)
        ]
        + [
            _clean_fill(side="SELL", markout_5s_bps=-4.0, fill_id=f"s{i}")
            for i in range(30)
        ]
    )
    r = check_per_side_markout_floor(fills)
    assert r.passed is False
    assert "BUY=" in r.detail
    assert "SELL=" in r.detail
