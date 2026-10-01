"""v1.4.115 Phase 1E.1 / 1F.4 — live_stats publisher contract test.

The dashboard's Alerts chip + Detectors card read three top-level
blocks from the live_stats payload: ``regime_mode``,
``shock_gate``, and ``post_reduction_cooldown``. This test pins
that the publisher emits each block with the expected shape, both
in the default (no-feature-enabled) case and after the
corresponding state has been mutated.

The matching ``SpreadComposition``-on-live_stats item (1E.1.e —
publish at 5 s cadence vs heartbeat's 30 s) is part of the larger
quote_breakdown publisher work that already ships
``spread_composition`` per cycle. Verified here by asserting
``quote_breakdown.spread_composition`` round-trips the new
``inventory_drift_bid_bps`` / ``inventory_drift_ask_bps`` fields
the Detectors card consumes for the inventory_drift row.
"""

from __future__ import annotations

from app.config import Settings
from app.live_stats import (
    _post_reduction_cooldown_block,
    _regime_mode_block,
    _shock_gate_block,
)
from app.quoting import SpreadComposition
from app.regime_controller import Mode
from app.shock_gate import observe as shock_observe
from app.state import BotState
from app.enums import QuoteEligibility, Side


def _bs() -> BotState:
    """Fresh BotState with the full Phase 1 stack enabled."""
    s = Settings(
        REGIME_CONTROLLER_ENABLED=True,
        SHOCK_GATE_ENABLED=True,
        INVENTORY_DRIFT_GATE_ENABLED=True,
        POST_REDUCTION_COOLDOWN_SECONDS=60.0,
    )
    return BotState(s)


# ---------------------------------------------------------------------------
# regime_mode block
# ---------------------------------------------------------------------------


def test_regime_mode_block_default_shape() -> None:
    bs = _bs()
    block = _regime_mode_block(bs)
    assert block is not None
    assert block["mode"] == "NORMAL"
    assert block["seconds_in_mode"] == 0.0
    assert "mode_since_iso" in block  # publisher adds wall-clock ISO
    assert block["transition_count"] == 0
    assert block["recent_transitions"] == []
    assert block["last_transition_reason"] is None
    assert block["entry_arming_seconds"] is None
    assert block["exit_arming_seconds"] is None
    # Session counters are floats.
    for k in (
        "time_in_normal_seconds",
        "time_in_defensive_seconds",
        "time_in_shock_seconds",
        # Phase 4G.2 (v1.4.209) — CALM / CAUTIOUS counters.
        "time_in_calm_seconds",
        "time_in_cautious_seconds",
    ):
        assert isinstance(block[k], float)
    # Phase 4G.5 (v1.4.211) — forward_signal block always renders.
    # Default: placeholder fields (forward layer disabled by default).
    fwd = block["forward_signal"]
    assert isinstance(fwd, dict)
    assert fwd["classification"] is None
    assert fwd["reason"] is None


def test_regime_mode_block_after_transition_to_defensive() -> None:
    bs = _bs()
    bs.regime_controller.mode = Mode.DEFENSIVE
    bs.regime_controller.mode_since_mono = 1000.0
    bs.regime_controller.last_transition_reason = (
        "defensive_entry:util=0.700,dwell=15.0s"
    )
    bs.regime_controller.transition_count = 1
    bs.regime_controller.last_transitions.append(
        ("NORMAL", "DEFENSIVE", 1000.0, "defensive_entry:util=0.700,dwell=15.0s")
    )
    block = _regime_mode_block(bs)
    assert block["mode"] == "DEFENSIVE"
    assert block["last_transition_reason"] == "defensive_entry:util=0.700,dwell=15.0s"
    assert block["transition_count"] == 1
    assert len(block["recent_transitions"]) == 1
    rec = block["recent_transitions"][0]
    assert rec["from"] == "NORMAL"
    assert rec["to"] == "DEFENSIVE"


def test_regime_mode_block_returns_none_when_controller_missing() -> None:
    """Back-compat for replay paths / pre-Phase-1C bots — publisher
    emits ``None`` rather than crashing."""

    class _Fake:
        pass

    fake = _Fake()
    assert _regime_mode_block(fake) is None


# ---------------------------------------------------------------------------
# shock_gate block
# ---------------------------------------------------------------------------


def test_shock_gate_block_default_dormant_shape() -> None:
    bs = _bs()
    block = _shock_gate_block(bs)
    assert block is not None
    assert block["active"] is False
    assert block["locked_side"] is None
    assert block["seconds_in_lock"] is None
    assert block["fire_count"] == 0


def test_shock_gate_block_after_fire() -> None:
    bs = _bs()
    # Drive shock_gate into locked state via its public observe API.
    shock_observe(
        bs.shock_gate,
        now_mono=1000.0,
        position_qty=+9.0,
        effective_abs_cap=10.0,
        drift_bps_10s=-25.0,
        drift_bps_30s=-60.0,
        enabled=True,
        shock_inventory_pct_threshold=0.80,
        shock_threshold_bps_10s=20.0,
        shock_threshold_bps_30s=50.0,
        clear_util_threshold=0.30,
        max_cooldown_seconds=300.0,
    )
    block = _shock_gate_block(bs)
    assert block["active"] is True
    assert block["locked_side"] == "QUOTE_SELL_ONLY"
    assert block["fire_count"] == 1


# ---------------------------------------------------------------------------
# post_reduction_cooldown block
# ---------------------------------------------------------------------------


def test_post_reduction_cooldown_block_dormant() -> None:
    bs = _bs()
    block = _post_reduction_cooldown_block(bs)
    assert block is not None
    assert block["enabled"] is True
    assert block["cooldown_seconds"] == 60.0
    assert block["active"] is False
    assert block["suppressed_side"] is None
    assert block["fire_count"] == 0


def test_post_reduction_cooldown_block_after_arming() -> None:
    """v1.5.191 BUG-032: ``fire_count`` no longer bumps in the
    state-arming method — only at the actual cooldown engagement edge
    in ``Bot._apply_eligibility_engine``. So a reducing fill alone
    sets timestamp + suppressed_side but leaves fire_count=0; the
    counter only increments when the cooldown enters its
    DEFENSIVE/SHOCK active window. Aligns ``fire_count`` semantics
    with ``cleared_via_*`` (both count engagements, not arms)."""
    bs = _bs()
    bs._note_inventory_reduction_for_cooldown(
        now_mono=1000.0,
        fill_side=Side.SELL,
        prev_qty=+5.0,
        new_qty=+3.0,
    )
    block = _post_reduction_cooldown_block(bs)
    assert block["enabled"] is True
    assert block["suppressed_side"] == "QUOTE_SELL_ONLY"
    # v1.5.191 BUG-032 — counter moved to engagement edge.
    assert block["fire_count"] == 0
    # ``active`` and ``seconds_remaining`` depend on the monotonic
    # clock; we just assert they exist with the right types.
    assert isinstance(block["active"], bool)
    assert isinstance(block["seconds_remaining"], float)


def test_post_reduction_cooldown_block_disabled_when_feature_off() -> None:
    s = Settings(POST_REDUCTION_COOLDOWN_SECONDS=0.0)
    bs = BotState(s)
    block = _post_reduction_cooldown_block(bs)
    assert block["enabled"] is False
    assert block["active"] is False


# ---------------------------------------------------------------------------
# spread_composition shape (1E.1.e — Detectors card reads
# inventory_drift_{bid,ask}_bps from the composition)
# ---------------------------------------------------------------------------


def test_spread_composition_to_dict_carries_inventory_drift_fields() -> None:
    """The Detectors card's inventory_drift row reads
    ``spread_composition.inventory_drift_bid_bps`` /
    ``inventory_drift_ask_bps`` from the live_stats payload's
    quote_breakdown. Pin the field names + presence."""
    sc = SpreadComposition(
        inventory_drift_bid_bps=15.0,
        inventory_drift_ask_bps=0.0,
    )
    d = sc.to_dict()
    assert d["inventory_drift_bid_bps"] == 15.0
    assert d["inventory_drift_ask_bps"] == 0.0
    # slow_trend fields also surfaced for the slow_trend row.
    assert "slow_trend_bid_bps" in d
    assert "slow_trend_ask_bps" in d


def test_spread_composition_to_dict_default_zero_fields() -> None:
    """A fresh SpreadComposition (no contributors firing) round-trips
    every field as zero — the Detectors card relies on this for the
    ``OFF`` rendering of the gate rows."""
    sc = SpreadComposition()
    d = sc.to_dict()
    assert d["inventory_drift_bid_bps"] == 0.0
    assert d["inventory_drift_ask_bps"] == 0.0
    assert d["slow_trend_bid_bps"] == 0.0
    assert d["slow_trend_ask_bps"] == 0.0
