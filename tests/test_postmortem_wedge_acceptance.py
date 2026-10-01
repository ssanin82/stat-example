"""v1.4.93 — postmortem wedge-acceptance section unit tests.

Same check logic backs both the postmortem report section AND the
``scripts/verify_snapshot.py`` standalone CLI. These tests exercise
the section module directly with inline-constructed payloads — no
filesystem dependency on snapshots/.
"""

from __future__ import annotations

from tools.postmortem.sections.wedge_acceptance import (
    CheckResult,
    WedgeAcceptanceFindings,
    check_desync_phase_ok,
    check_gate_phase2a_violations_zero,
    check_gone_on_exchange_zero,
    check_http_acked_no_ws_modest,
    check_hydration_merged_zero,
    check_no_desync_detected_events,
    check_no_duplicate_open_orders,
    check_reaper_total_modest,
    check_recent_events_no_critical,
    check_risk_state_normal,
    check_session_pnl_visible,
    check_wedge_episodes_zero,
    check_ws_arrived_late_modest,
    check_ws_unmatched_zero,
    detect_from_payloads,
    render_html_section,
    render_markdown_section,
)


# ---------------------------------------------------------------------------
# Individual checks — pass cases
# ---------------------------------------------------------------------------


def test_ws_unmatched_zero_passes_on_clean_state() -> None:
    r = check_ws_unmatched_zero({"executor_state": {"ws_event_unmatched_to_local_wo_total": 0}})
    assert r.passed
    assert r.severity == "fatal"


def test_ws_unmatched_zero_fails_on_nonzero() -> None:
    r = check_ws_unmatched_zero({"executor_state": {"ws_event_unmatched_to_local_wo_total": 4}})
    assert not r.passed
    assert "value=4" in r.detail


def test_wedge_episodes_zero_passes_on_clean() -> None:
    r = check_wedge_episodes_zero({"executor_state": {"wedge_episode_count_session": 0}})
    assert r.passed


def test_wedge_episodes_zero_fails_on_nonzero() -> None:
    r = check_wedge_episodes_zero({"executor_state": {"wedge_episode_count_session": 2}})
    assert not r.passed
    assert r.severity == "fatal"


def test_risk_state_normal_accepts_NORMAL_UNKNOWN_empty() -> None:
    for v in ("NORMAL", "UNKNOWN", "", None):
        r = check_risk_state_normal({"executor_state": {"risk_exec_state": v}})
        assert r.passed, f"expected pass for risk_exec_state={v!r}; got {r.detail}"


def test_risk_state_normal_fails_on_CANCELLING() -> None:
    r = check_risk_state_normal({"executor_state": {"risk_exec_state": "CANCELLING"}})
    assert not r.passed
    assert "CANCELLING" in r.detail


def test_desync_phase_ok_passes_on_OK_or_empty() -> None:
    for v in ("OK", "", None):
        sd = {} if v is None else {"desync_phase": v}
        r = check_desync_phase_ok(sd)
        assert r.passed, f"expected pass for desync_phase={v!r}"


def test_desync_phase_ok_fails_on_RECOVERED() -> None:
    r = check_desync_phase_ok({"desync_phase": "RECOVERED"})
    assert not r.passed


def test_hydration_merged_zero_is_warn_severity_when_nonzero() -> None:
    r = check_hydration_merged_zero({"executor_state": {"hydration_merged_existing_total": 3}})
    assert not r.passed
    assert r.severity == "warn"  # tolerated; not fatal


def test_reaper_total_modest_passes_below_threshold() -> None:
    r = check_reaper_total_modest({"executor_state": {
        "reaper_cancel_pending_reaped_total": 10,
        "reaper_desync_removed_total": 5,
        "reaper_sent_rejected_total": 0,
    }})
    assert r.passed
    assert "sum=15" in r.detail


def test_reaper_total_modest_fails_above_threshold() -> None:
    r = check_reaper_total_modest({"executor_state": {
        "reaper_cancel_pending_reaped_total": 100,
    }})
    assert not r.passed


def test_no_duplicate_open_orders_passes_on_distinct_prices() -> None:
    r = check_no_duplicate_open_orders({"data": [
        {"side": "buy", "px": "2.000"},
        {"side": "buy", "px": "2.001"},
        {"side": "sell", "px": "2.003"},
    ]})
    assert r.passed
    assert "no dupes" in r.detail


def test_no_duplicate_open_orders_fails_on_same_side_same_price() -> None:
    r = check_no_duplicate_open_orders({"data": [
        {"side": "buy", "px": "2.000"},
        {"side": "buy", "px": "2.000"},  # duplicate
    ]})
    assert not r.passed
    assert "dupes" in r.detail


def test_recent_events_no_critical_passes_on_info_only() -> None:
    r = check_recent_events_no_critical([
        {"severity": "INFO", "event_type": "foo"},
        {"severity": "WARNING", "event_type": "bar"},
    ])
    assert r.passed


def test_recent_events_no_critical_fails_on_critical() -> None:
    r = check_recent_events_no_critical([
        {"severity": "CRITICAL", "event_type": "kill_triggered"},
    ])
    assert not r.passed
    assert "critical=1" in r.detail


def test_no_desync_detected_events_passes_when_absent() -> None:
    r = check_no_desync_detected_events([
        {"severity": "INFO", "event_type": "other"},
    ])
    assert r.passed


def test_no_desync_detected_events_warns_when_present() -> None:
    r = check_no_desync_detected_events([
        {"severity": "WARNING", "event_type": "desync_detected"},
    ])
    assert not r.passed
    assert r.severity == "warn"


# ---------------------------------------------------------------------------
# Aggregated findings + rendering
# ---------------------------------------------------------------------------


def test_detect_from_payloads_returns_findings_with_14_checks() -> None:
    findings = detect_from_payloads(
        snapshot_name="test", bot_version="1.4.96",
        captured_at="2026-05-19T12:00:00Z",
        state={"executor_state": {}, "desync_phase": "OK"},
        open_orders={"data": []},
        events=[],
        session_summary={},
    )
    assert isinstance(findings, WedgeAcceptanceFindings)
    # v1.4.96: 14 checks total. Two new WARN gates added on top of
    # the v1.4.95 baseline:
    #   * check_http_acked_no_ws_modest (TIER 2)
    #   * check_ws_arrived_late_modest (TIER 3)
    assert len(findings.checks) == 14


# ---------------------------------------------------------------------------
# v1.4.96 TIER 1 (FATAL) — gone_on_exchange_total fatal gate
# ---------------------------------------------------------------------------


def test_gone_on_exchange_zero_passes_on_clean() -> None:
    r = check_gone_on_exchange_zero(
        {"executor_state": {"gone_on_exchange_total": 0}}
    )
    assert r.passed
    assert r.severity == "fatal"
    assert "value=0" in r.detail


def test_gone_on_exchange_zero_passes_when_field_absent() -> None:
    # Pre-v1.4.95 snapshots have no gone_on_exchange_total field —
    # the check should not fail those by accident. (Reads as 0.)
    r = check_gone_on_exchange_zero({"executor_state": {}})
    assert r.passed


def test_gone_on_exchange_zero_fails_on_acked_no_cancel() -> None:
    r = check_gone_on_exchange_zero(
        {
            "executor_state": {
                "gone_on_exchange_total": 3,
                "gone_on_exchange_acked_no_cancel_total": 3,
            }
        }
    )
    assert not r.passed
    assert r.severity == "fatal"
    assert "value=3" in r.detail
    assert "acked_no_cancel (BUG-023)=3" in r.detail


def test_gone_on_exchange_zero_fails_on_cancel_no_http_confirm() -> None:
    # v1.4.96 — cancel was sent but no HTTP success response. RED.
    r = check_gone_on_exchange_zero(
        {
            "executor_state": {
                "gone_on_exchange_total": 2,
                "gone_on_exchange_cancel_no_http_confirm_total": 2,
            }
        }
    )
    assert not r.passed
    assert "cancel_no_http_confirm=2" in r.detail


def test_gone_on_exchange_zero_reads_top_level_when_no_executor_state() -> None:
    r = check_gone_on_exchange_zero(
        {"gone_on_exchange_total": 1, "gone_on_exchange_phantom_no_ack_total": 1}
    )
    assert not r.passed
    assert "phantom_no_ack (BUG-024)=1" in r.detail


def test_findings_with_nonzero_gone_on_exchange_is_fatal() -> None:
    findings = detect_from_payloads(
        snapshot_name="snap-goe",
        bot_version="1.4.96",
        captured_at="2026-05-19T12:00:00Z",
        state={
            "executor_state": {
                "gone_on_exchange_total": 1,
                "gone_on_exchange_phantom_no_ack_total": 1,
            },
            "desync_phase": "OK",
        },
        open_orders={"data": []},
        events=[],
        session_summary={},
    )
    assert findings.ok is False
    assert any(
        c.name == "gone_on_exchange_total == 0" for c in findings.fatal_failures
    )


# ---------------------------------------------------------------------------
# v1.4.96 TIER 2 (WARN) — http_acked_no_ws_total threshold gate
# ---------------------------------------------------------------------------


def test_http_acked_no_ws_modest_passes_below_threshold() -> None:
    r = check_http_acked_no_ws_modest(
        {"executor_state": {"http_acked_no_ws_total": 6}}
    )
    assert r.passed
    assert r.severity == "warn"
    assert "value=6" in r.detail


def test_http_acked_no_ws_modest_fails_above_threshold() -> None:
    r = check_http_acked_no_ws_modest(
        {"executor_state": {"http_acked_no_ws_total": 75}}
    )
    assert not r.passed
    assert r.severity == "warn"


def test_http_acked_no_ws_modest_accepts_threshold_override() -> None:
    # Operator can tighten the threshold ad-hoc.
    r = check_http_acked_no_ws_modest(
        {"executor_state": {"http_acked_no_ws_total": 6}}, threshold=5
    )
    assert not r.passed


def test_http_acked_no_ws_does_not_count_as_fatal_failure() -> None:
    # Confirms tier-2 stays WARN — does NOT add to fatal_failures.
    findings = detect_from_payloads(
        snapshot_name="snap-warn",
        bot_version="1.4.96",
        captured_at="2026-05-19T12:00:00Z",
        state={
            "executor_state": {
                "gone_on_exchange_total": 0,
                "http_acked_no_ws_total": 100,  # > threshold, warns
            },
            "desync_phase": "OK",
        },
        open_orders={"data": []},
        events=[],
        session_summary={},
    )
    # ok=True because no fatal failure.
    assert findings.ok is True
    # But there's a warn failure for the http_acked_no_ws check.
    assert any(
        c.severity == "warn" and "http_acked_no_ws" in c.name
        for c in findings.warn_failures
    )


# ---------------------------------------------------------------------------
# v1.4.96 TIER 3 (WARN) — ws_arrived_late_total threshold gate
# ---------------------------------------------------------------------------


def test_ws_arrived_late_modest_passes_at_zero() -> None:
    r = check_ws_arrived_late_modest(
        {"executor_state": {"ws_arrived_late_total": 0}}
    )
    assert r.passed
    assert r.severity == "warn"


def test_ws_arrived_late_modest_passes_below_threshold() -> None:
    r = check_ws_arrived_late_modest(
        {"executor_state": {"ws_arrived_late_total": 12}}
    )
    assert r.passed


def test_ws_arrived_late_modest_fails_above_threshold() -> None:
    r = check_ws_arrived_late_modest(
        {"executor_state": {"ws_arrived_late_total": 200}}
    )
    assert not r.passed
    assert r.severity == "warn"
    assert "value=200" in r.detail


def test_findings_ok_property_reflects_fatal_failures() -> None:
    findings = detect_from_payloads(
        snapshot_name="test", bot_version="1.4.93",
        captured_at="2026-05-19T12:00:00Z",
        state={"executor_state": {}, "desync_phase": "OK"},
        open_orders={"data": []},
        events=[], session_summary={},
    )
    assert findings.ok is True
    # Now inject a fatal failure.
    findings2 = detect_from_payloads(
        snapshot_name="test", bot_version="1.4.93",
        captured_at="2026-05-19T12:00:00Z",
        state={"executor_state": {"ws_event_unmatched_to_local_wo_total": 5},
               "desync_phase": "OK"},
        open_orders={"data": []},
        events=[], session_summary={},
    )
    assert findings2.ok is False
    assert len(findings2.fatal_failures) == 1


def test_markdown_clean_renders_pass_one_liner() -> None:
    findings = detect_from_payloads(
        snapshot_name="snap-clean", bot_version="1.4.93",
        captured_at="2026-05-19T12:00:00Z",
        state={"executor_state": {}, "desync_phase": "OK"},
        open_orders={"data": []},
        events=[], session_summary={},
    )
    md = render_markdown_section(findings)
    assert "## Wedge acceptance" in md
    assert "**PASS**" in md
    assert "FAIL" not in md


def test_markdown_with_fatal_failure_lists_fatal_section() -> None:
    findings = detect_from_payloads(
        snapshot_name="snap-bad", bot_version="1.4.93",
        captured_at="2026-05-19T12:00:00Z",
        state={
            "executor_state": {"ws_event_unmatched_to_local_wo_total": 4},
            "desync_phase": "OK",
        },
        open_orders={"data": []},
        events=[], session_summary={},
    )
    md = render_markdown_section(findings)
    assert "**FAIL**" in md
    assert "### Fatal" in md
    assert "ws_event_unmatched_to_local_wo_total" in md
    assert "value=4" in md


def test_markdown_with_only_warn_failure_renders_pass_with_warnings() -> None:
    findings = detect_from_payloads(
        snapshot_name="snap-warn", bot_version="1.4.93",
        captured_at="2026-05-19T12:00:00Z",
        state={
            "executor_state": {"hydration_merged_existing_total": 3},
            "desync_phase": "OK",
        },
        open_orders={"data": []},
        events=[], session_summary={},
    )
    md = render_markdown_section(findings)
    assert "PASS (with 1 warning(s))" in md
    assert "### Warnings" in md
    assert "hydration_merged_existing_total" in md


def test_html_clean_renders_pass_status() -> None:
    findings = detect_from_payloads(
        snapshot_name="snap-clean", bot_version="1.4.93",
        captured_at="2026-05-19T12:00:00Z",
        state={"executor_state": {}, "desync_phase": "OK"},
        open_orders={"data": []},
        events=[], session_summary={},
    )
    html = render_html_section(findings)
    assert '<section id="wedge-acceptance">' in html
    assert "<h2>Wedge acceptance</h2>" in html
    assert "PASS" in html


def test_html_fatal_renders_red_FAIL() -> None:
    findings = detect_from_payloads(
        snapshot_name="snap-bad", bot_version="1.4.93",
        captured_at="2026-05-19T12:00:00Z",
        state={
            "executor_state": {"ws_event_unmatched_to_local_wo_total": 4},
            "desync_phase": "OK",
        },
        open_orders={"data": []},
        events=[], session_summary={},
    )
    html = render_html_section(findings)
    assert "FAIL" in html
    assert "Fatal" in html
    # Bad inputs should not appear unescaped (defensive HTML).
    assert "<script" not in html.lower()
