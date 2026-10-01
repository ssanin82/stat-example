"""v1.4.93 wedge-elimination-cleanup Phase 6D.2 —
silent_wedge detector payload assertion.

The bot's ``_maybe_emit_silent_wedge_diagnostic`` fires when the
conjoint condition (healthy engine + QUOTE_BOTH eligibility + BOTH
active_sides + no side_unresolved + idle > threshold) is met. The
payload it logs must include the operator-facing evidence the
postmortem tool consumes:

* `executor_state` — full snapshot with Phase 1/2/3 counters
* `eligibility` / `active_sides` — gate state at fire time
* `execution_idle_s` — how long the bot's been stuck
* Engine telemetry keys (mode, bid/ask reasons, final px/sz)

This test pins the payload contract — if a refactor accidentally
drops a key, the postmortem tool stops working.

Approach: invoke `_maybe_emit_silent_wedge_diagnostic` directly with
controlled inputs, monkeypatch `log_extra` to capture the payload,
and assert on the dict shape.
"""

from __future__ import annotations

import time

import pytest

from app.enums import Side
from tests.integration.full_flow_harness import harness_ctx


def test_phase6d2_silent_wedge_payload_includes_phase_1_evidence(
    monkeypatch,
) -> None:
    """When the silent_wedge detector fires, the payload includes:

    * ``executor_state`` dict with Phase 1 wedge counters
    * Phase 2A `side_unresolved` flags (zero — that's why we're wedged)
    * `execution_idle_s` field showing how stale the place attempt is
    * Engine telemetry context for the operator
    """
    captured_payloads: list[dict] = []

    # Intercept log_extra to capture payloads.
    import app.execution as _exec_mod

    real_log_extra = _exec_mod.log_extra

    def _spy_log_extra(logger, level, event_name, payload, *args, **kwargs):
        if event_name == "executor_silent_wedge_detected":
            captured_payloads.append(dict(payload))
        return real_log_extra(logger, level, event_name, payload, *args, **kwargs)

    monkeypatch.setattr(_exec_mod, "log_extra", _spy_log_extra)

    with harness_ctx(
        SILENT_WEDGE_DETECT_THRESHOLD_SECONDS=10.0,
        SILENT_WEDGE_DETECT_RELOG_SECONDS=30.0,
    ) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()

        # Drive a tick to populate state and exec_state snapshot.
        h.tick_once()

        # Simulate stale activity by setting last_outbound_attempt to
        # far past — this triggers the idle threshold (>10s).
        h.state.last_outbound_attempt_ts_mono = time.monotonic() - 30.0
        # Ensure side_unresolved is clean (no explicit justification).
        h.om._side_unresolved_active = {Side.BUY: False, Side.SELL: False}
        # Reset the relog timer so the firs-emission path executes.
        h.om._silent_wedge_diag_last_log_mono = 0.0
        # v1.4.206 BUG-025 false-positive fix added a "skip detector
        # if ACKED/PARTIAL working order exists" exemption (steady-
        # state resting orders are NOT a wedge). ``tick_once()`` above
        # populated working orders; clear them here so the detector
        # SEES the genuine wedge case it was built to catch. Without
        # this, the post-v1.4.206 exemption correctly suppresses the
        # fire and the payload-shape assertion below has nothing to
        # validate.
        with h.state._lock:
            h.state.set_working_order(Side.BUY, 0, None)
            h.state.set_working_order(Side.SELL, 0, None)

        # Build a synthetic telemetry dict that satisfies the conjoint
        # conditions (engine healthy = not no_quote).
        telemetry = {
            "quote_engine_mode": "two_sided",
            "quote_engine_normal_mode_requested": True,
            "quote_engine_bid_reason": "ok",
            "quote_engine_ask_reason": "ok",
            "quote_engine_bid_final_px": 1.999,
            "quote_engine_ask_final_px": 2.003,
            "quote_engine_bid_final_sz": 3.0,
            "quote_engine_ask_final_sz": 3.0,
        }

        # Invoke the detector directly.
        h.om._maybe_emit_silent_wedge_diagnostic(
            telemetry=telemetry,
            decision_quote_cycle_id="cycle-test-6d2",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )

    # The detector should have fired exactly once.
    assert len(captured_payloads) >= 1, (
        f"silent_wedge detector did not fire; captured={len(captured_payloads)}"
    )

    payload = captured_payloads[0]

    # === Core payload fields ===
    assert "quote_cycle_id" in payload
    assert payload["quote_cycle_id"] == "cycle-test-6d2"
    assert "execution_idle_s" in payload
    assert payload["execution_idle_s"] >= 10.0, (
        f"execution_idle_s should reflect the stale gap; "
        f"got {payload['execution_idle_s']}"
    )
    assert "threshold_s" in payload
    assert "eligibility" in payload
    assert payload["eligibility"] == "QUOTE_BOTH"
    assert "active_sides" in payload
    assert payload["active_sides"] == "BOTH"
    assert "first_emission" in payload
    assert payload["first_emission"] is True
    assert "position_qty" in payload
    assert "position_notional_usd" in payload

    # === executor_state must be present + carry Phase 1/2 counters ===
    assert "executor_state" in payload
    es = payload["executor_state"]
    assert isinstance(es, dict)
    # The Phase 1/2 wedge counters the postmortem depends on.
    expected_counters = {
        "ws_event_unmatched_to_local_wo_total",
        "hydration_merged_existing_total",
        "reaper_cancel_pending_reaped_total",
        "reaper_desync_removed_total",
        "reaper_sent_rejected_total",
        "gate_phase2a_invariant_violation_total",
        "risk_exec_state",
    }
    missing = expected_counters - set(es.keys())
    assert not missing, (
        f"silent_wedge executor_state payload missing required counters: "
        f"{sorted(missing)}. Got keys: {sorted(es.keys())}"
    )

    # === Engine telemetry context ===
    assert "quote_engine_mode" in payload
    assert payload["quote_engine_mode"] == "two_sided"


def test_phase6d2_silent_wedge_does_not_fire_when_idle_below_threshold(
    monkeypatch,
) -> None:
    """If idle time < threshold, the detector must NOT fire — even
    when all other conditions are met."""
    captured: list[dict] = []
    import app.execution as _exec_mod
    real_log_extra = _exec_mod.log_extra

    def _spy(logger, level, event_name, payload, *args, **kwargs):
        if event_name == "executor_silent_wedge_detected":
            captured.append(payload)
        return real_log_extra(logger, level, event_name, payload, *args, **kwargs)

    monkeypatch.setattr(_exec_mod, "log_extra", _spy)

    with harness_ctx(
        SILENT_WEDGE_DETECT_THRESHOLD_SECONDS=10.0,  # high threshold
    ) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()
        h.tick_once()

        # Idle of only 2 seconds — well below 10s threshold.
        h.state.last_outbound_attempt_ts_mono = time.monotonic() - 2.0
        h.om._side_unresolved_active = {Side.BUY: False, Side.SELL: False}

        h.om._maybe_emit_silent_wedge_diagnostic(
            telemetry={"quote_engine_mode": "two_sided"},
            decision_quote_cycle_id="cycle-test",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )

    assert len(captured) == 0, (
        f"silent_wedge fired with idle < threshold; got {len(captured)} payloads"
    )


def test_phase6d2_silent_wedge_does_not_fire_when_side_unresolved_active(
    monkeypatch,
) -> None:
    """If a side is in `side_unresolved`, that's an explicit justification
    for not placing — NOT a silent wedge."""
    captured: list[dict] = []
    import app.execution as _exec_mod
    real_log_extra = _exec_mod.log_extra

    def _spy(logger, level, event_name, payload, *args, **kwargs):
        if event_name == "executor_silent_wedge_detected":
            captured.append(payload)
        return real_log_extra(logger, level, event_name, payload, *args, **kwargs)

    monkeypatch.setattr(_exec_mod, "log_extra", _spy)

    with harness_ctx(
        SILENT_WEDGE_DETECT_THRESHOLD_SECONDS=10.0,
    ) as h:
        h.seed_market(best_bid=2.000, best_ask=2.002)
        h.seed_position()
        h.tick_once()

        h.state.last_outbound_attempt_ts_mono = time.monotonic() - 30.0
        # BUY side is unresolved → silent_wedge should NOT fire.
        h.om._side_unresolved_active = {Side.BUY: True, Side.SELL: False}

        h.om._maybe_emit_silent_wedge_diagnostic(
            telemetry={"quote_engine_mode": "two_sided"},
            decision_quote_cycle_id="cycle-test",
            eligibility="QUOTE_BOTH",
            active_sides="BOTH",
            position_qty=0.0,
            position_notional=0.0,
        )

    assert len(captured) == 0, (
        f"silent_wedge fired with side_unresolved active; got {len(captured)} payloads"
    )
