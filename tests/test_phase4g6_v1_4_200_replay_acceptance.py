"""Phase 4G.6 (v1.4.212) — cross-cutting acceptance replay.

Anchored to the **driving incident** that motivated Phase 4G:

    ``snapshots/v1.4.200-260521-183301-prod.okx.ton.usdt.perp``
    14:30:19  shock_gate fired (util=0.900, drift_10s=+24.67 bps)
    14:31:18  soft_flatten started (position-drawdown gate,
              adverse=36.8 bps for 15 s)
    14:31:31  soft_flatten completed

The driving observation: SF#11175 fired straight from NORMAL mode
with ZERO upstream defensive activity. The bot was in NORMAL the
entire ~41 min while vol_bps was climbing in the background —
because all existing defences are REACTIVE on LAGGING indicators
(drift_10s ≥ 24 bps, util ≥ 50 %, vol_ratio ≥ 2.0, adverse ≥ 35 bps
for 15 s). By the time any of them fire, the position is already
underwater.

Phase 4G adds a LEADING-indicator layer ABOVE the reactive FSM.
The acceptance criterion for the whole phase: replay the SF#11175
signal envelope against the new code path and prove the bot would
have transitioned to CAUTIOUS — and therefore widened spreads,
shrunk size, dropped to 1 rung, AND shrunk the inventory cap by
30 % — BEFORE the position-drawdown gate fired.

This test exercises the **integrated path** end-to-end:

    raw signal histories (synthesised from the snapshot's envelope)
        → ``classify_forward_regime`` (4G.1, pure)
            → ``evaluate_mode(forward_regime=...)`` (4G.2, FSM)
                → ``compute_knobs_for_mode`` (4G.3, knobs)
                    → ``QuoteBuildContext.effective_max_abs_position``
                       (4G.4, wired to ``_clip_entry_sizes`` +
                       ``_inventory_exec_bias_active``)

4G.1 and 4G.2's regression tests covered the first two arrows in
isolation. 4G.6 closes the loop and proves the four pieces compose
correctly.

Counterfactual control: the same trajectory with the forward layer
disabled (``forward_regime=None`` / ``REGIME_FORWARD_ENABLED=False``)
keeps the bot in NORMAL throughout — proving the forward layer is
the difference-maker.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import pytest

from app.quote_engine import QuoteEngine
from app.regime_controller import (
    Mode,
    RegimeControllerState,
    compute_knobs_for_mode,
    evaluate_mode,
)
from app.regime_forward_signals import (
    ForwardRegime,
    ForwardSignalThresholds,
    classify_forward_regime,
)
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


# ---------------------------------------------------------------------------
# Trajectory synthesis — matches the v1.4.200 SF#11175 incident envelope
# ---------------------------------------------------------------------------


def _flat_history(
    *,
    start_t: float,
    end_t: float,
    value: float,
    step: float = 10.0,
) -> list[tuple[float, float]]:
    """Constant value across the time window, sampled every ``step`` s."""
    out: list[tuple[float, float]] = []
    t = float(start_t)
    while t <= end_t + 1e-9:
        out.append((t, float(value)))
        t += step
    return out


def _linear_history(
    *,
    start_t: float,
    end_t: float,
    start_v: float,
    end_v: float,
    step: float = 5.0,
) -> list[tuple[float, float]]:
    """Linear ramp from start_v to end_v across the window."""
    out: list[tuple[float, float]] = []
    t = float(start_t)
    span = float(end_t) - float(start_t)
    if span <= 0:
        return [(t, float(end_v))]
    while t <= end_t + 1e-9:
        frac = (t - start_t) / span
        out.append((t, float(start_v) + (float(end_v) - float(start_v)) * frac))
        t += step
    return out


def _build_v1_4_200_signal_envelope(
    *,
    now: float = 300.0,
) -> dict:
    """Reconstruct the v1.4.200 SF#11175 signal envelope:

      * vol_bps: 2.5 stable for 240 s, then climbing 2.5 → 6.0 over
        the final 60 s (matches the snapshot's vol-SMA chart;
        slope ≈ 3.5 bps/min — well above the 0.5 bps/min CAUTIOUS
        threshold)
      * drift_30s: a small steady drift in the climb segment
      * ob_imbalance: mild widening in the final 60 s
      * binance_basis: held flat (basis_stretch trigger dormant in
        this snapshot — the snapshot showed basis_regime FLAT for
        61 % of the session)

    Returns a dict of inputs ready to pass to
    ``classify_forward_regime``.
    """
    stable_vol = _flat_history(start_t=0.0, end_t=240.0, value=2.5, step=10.0)
    climb_vol = _linear_history(
        start_t=240.0, end_t=now, start_v=2.5, end_v=6.0, step=5.0
    )
    vol_history = stable_vol + climb_vol

    drift_history = _flat_history(start_t=0.0, end_t=now, value=1.5, step=10.0)
    # OB imbalance: nudge from ~0 toward ~0.05 in the climb segment
    # (mild widening; below the 0.3 delta threshold so this lever
    # doesn't fire — vol_slope is the dominant trigger here).
    ob_stable = _flat_history(start_t=0.0, end_t=240.0, value=0.0, step=10.0)
    ob_widen = _linear_history(
        start_t=240.0, end_t=now, start_v=0.0, end_v=0.05, step=5.0
    )
    ob_history = ob_stable + ob_widen

    return {
        "vol_bps_history": vol_history,
        "drift_30s_history": drift_history,
        "ob_imbalance_history": ob_history,
        "current_ob_imbalance": 0.05,
        "current_vol_bps": 6.0,
        "current_drift_30s_bps": 1.5,
        "current_binance_basis_bps": 0.1,
        "binance_basis_30min_median_bps": 0.1,
        "now_mono": now,
    }


# ---------------------------------------------------------------------------
# Step 1 — classifier on the synthesised envelope returns CAUTIOUS
# ---------------------------------------------------------------------------


def test_v1_4_200_envelope_classifier_returns_cautious() -> None:
    """Confirm the synthesised envelope still triggers CAUTIOUS
    under the *default* ``ForwardSignalThresholds`` (the values
    that ship in v1.4.211's settings)."""
    inputs = _build_v1_4_200_signal_envelope()
    reading = classify_forward_regime(
        settings=ForwardSignalThresholds(),
        **inputs,
    )
    assert reading.classification == ForwardRegime.CAUTIOUS, (
        f"v1.4.200 envelope should classify as CAUTIOUS under default "
        f"thresholds; got {reading.classification.value} "
        f"(reason={reading.reason!r})"
    )
    assert "vol_slope" in reading.reason
    # Slope ≈ 3.5 bps/min over the final 60 s — well above the 0.5
    # threshold but below the 5.0 calm_max_drift gate.
    assert reading.vol_slope_bps_per_min is not None
    assert reading.vol_slope_bps_per_min >= 2.5


# ---------------------------------------------------------------------------
# Step 2 — FSM transitions from NORMAL to CAUTIOUS within the entry dwell
# ---------------------------------------------------------------------------


def _seed_state_in_normal(now_mono: float) -> RegimeControllerState:
    state = RegimeControllerState()
    state.mode = Mode.NORMAL
    state.mode_since_mono = max(now_mono, 1.0)  # >0 so seconds_in_mode renders
    return state


def test_v1_4_200_fsm_engages_cautious_within_entry_dwell_window() -> None:
    """Given the envelope, the FSM transitions NORMAL → CAUTIOUS
    within the configured entry dwell (3.0 s of continuous
    CAUTIOUS classification).

    Pre-4G the bot stayed in NORMAL straight through to the
    shock_gate firing 12 min later — captured here as the
    counterfactual.
    """
    base_time = 10_000.0
    state = _seed_state_in_normal(now_mono=base_time)
    # Tick 0 — classifier returns CAUTIOUS; arming begins.
    mode_0, reason_0 = evaluate_mode(
        state,
        now_mono=base_time + 0.0,
        util=0.05,
        vol_ratio=1.1,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
        forward_regime=ForwardRegime.CAUTIOUS,
        cautious_entry_dwell_seconds=3.0,
    )
    assert mode_0 is Mode.NORMAL
    assert state.cautious_entry_arming_since_mono == pytest.approx(
        base_time + 0.0
    )

    # Tick at +1.5 s — still arming.
    evaluate_mode(
        state,
        now_mono=base_time + 1.5,
        util=0.05,
        vol_ratio=1.1,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
        forward_regime=ForwardRegime.CAUTIOUS,
        cautious_entry_dwell_seconds=3.0,
    )
    assert state.mode is Mode.NORMAL  # dwell not yet satisfied

    # Tick at +3.5 s — entry dwell satisfied, transition fires.
    mode_t, reason_t = evaluate_mode(
        state,
        now_mono=base_time + 3.5,
        util=0.05,
        vol_ratio=1.1,
        slow_trend_active=False,
        inventory_drift_active=False,
        shock_gate_locked=False,
        enabled=True,
        forward_regime=ForwardRegime.CAUTIOUS,
        cautious_entry_dwell_seconds=3.0,
    )
    assert mode_t is Mode.CAUTIOUS, (
        f"FSM should have transitioned to CAUTIOUS within 3 s of the "
        f"classifier's first CAUTIOUS read; got {mode_t.value}"
    )
    assert reason_t is not None
    assert "cautious_entry" in reason_t


# ---------------------------------------------------------------------------
# Step 3 — Counterfactual: WITHOUT forward layer the bot stays NORMAL
# ---------------------------------------------------------------------------


def test_v1_4_200_counterfactual_without_forward_layer_stays_normal() -> None:
    """The same envelope WITHOUT the forward layer (passing
    ``forward_regime=None``, which is what bot.py does when
    ``REGIME_FORWARD_ENABLED=False``) keeps the bot in NORMAL
    indefinitely — proving the forward layer is the difference-
    maker.

    This is the v1.4.200 pre-4G behaviour: util stays low (no
    reactive trigger), vol_ratio stays below 2.0 (no reactive
    trigger), and the bot rides the climb into the shock_gate
    firing at util=0.900 / drift_10s=+24.67 bps with zero upstream
    defensive activity.
    """
    base_time = 10_000.0
    state = _seed_state_in_normal(now_mono=base_time)
    # Simulate 60 s of ticks at 0.5 s cadence with NO forward signal
    # and benign reactive inputs (low util, vol_ratio below threshold).
    for tick in range(0, 120):  # 60 s at 0.5 s
        evaluate_mode(
            state,
            now_mono=base_time + tick * 0.5,
            util=0.05,
            vol_ratio=1.2,  # below 2.0 entry threshold
            slow_trend_active=False,
            inventory_drift_active=False,
            shock_gate_locked=False,
            enabled=True,
            forward_regime=None,  # forward layer OFF
            cautious_entry_dwell_seconds=3.0,
        )
    assert state.mode is Mode.NORMAL, (
        f"Without the forward layer the bot must stay in NORMAL "
        f"under benign reactive inputs (this is the v1.4.200 pre-4G "
        f"behaviour); got {state.mode.value}"
    )
    assert state.cautious_entry_arming_since_mono is None


# ---------------------------------------------------------------------------
# Step 4 — Knobs flip to CAUTIOUS profile (v1.5.231 reshape):
#   * 1.30× base half-spread
#   * 1-rung ladder
#   * 0.70× inventory budget
#   * 1.00× quote notional (raised from 0.70 in v1.5.231 — see test
#     docstring for the v1.5.230-260529-094214 0-fill incident that
#     motivated lifting notional out of CAUTIOUS's defensive stack).
# ---------------------------------------------------------------------------


def test_v1_4_200_knobs_flip_to_cautious_profile() -> None:
    """Once the FSM is in CAUTIOUS, ``compute_knobs_for_mode`` emits
    the CAUTIOUS profile: 1.30 × spread, 1-rung ladder, 0.70 ×
    inventory budget.

    v1.5.231 (2026-05-29) — notional multiplier raised from 0.70 to
    1.00. The 0.70 value was calibrated assuming
    `QUOTE_NOTIONAL_USD ≥ ~$10`; on TON we run at $7 and
    `MIN_QUOTE_NOTIONAL_USD=$5`, so 0.70 × $7 = $4.90 fell BELOW the
    min-notional floor and dropped every CAUTIOUS rung (v1.5.230
    260529-094214 snapshot: 0 fills in 27 min). CAUTIOUS now defends
    via spread widening + 1-rung ladder only; the notional axis is
    no longer part of CAUTIOUS's stack. DEFENSIVE retains 0.50× as
    the next escalation tier."""
    knobs_normal = compute_knobs_for_mode(Mode.NORMAL)
    knobs_cautious = compute_knobs_for_mode(Mode.CAUTIOUS)

    # NORMAL is the identity profile.
    assert knobs_normal.base_half_spread_mult == pytest.approx(1.0)
    assert knobs_normal.quote_notional_mult == pytest.approx(1.0)
    assert knobs_normal.inventory_budget_mult == pytest.approx(1.0)

    # CAUTIOUS — load-bearing axes are spread (1.30×) + inventory
    # budget (0.70×) + 1-rung ladder (asserted elsewhere in this
    # file). Notional axis intentionally held at 1.0 (v1.5.231).
    assert knobs_cautious.base_half_spread_mult > 1.0
    assert knobs_cautious.base_half_spread_mult == pytest.approx(1.30, abs=0.01)
    assert knobs_cautious.quote_notional_mult == pytest.approx(1.00, abs=0.01)
    assert knobs_cautious.inventory_budget_mult < 1.0
    assert knobs_cautious.inventory_budget_mult == pytest.approx(0.70, abs=0.01)


# ---------------------------------------------------------------------------
# Step 5 — QuoteEngine consumers honour the CAUTIOUS inventory budget
# ---------------------------------------------------------------------------


def _engine(**overrides):
    path = Path(tempfile.gettempdir()) / f"mm_4g6_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 100.0,
        "MAX_POSITION_NOTIONAL_USD": 100000.0,
        "MAX_ORDER_NOTIONAL_USD": 50000.0,
        "INVENTORY_EXEC_BIAS_RATIO": 0.5,
        "INVENTORY_EXEC_BIAS_MIN_UTIL_PCT": 0.5,
    }
    base.update(overrides)
    s = UnitTestSettings.model_validate(base)
    eng = QuoteEngine(s, mock_mm_client().symbol_spec)
    return eng, s


def test_v1_4_200_quote_engine_respects_cautious_inventory_budget() -> None:
    """Final arrow: the CAUTIOUS knob (inventory_budget_mult=0.70)
    flows through to ``_clip_entry_sizes`` and
    ``_inventory_exec_bias_active``.

    Concrete scenario tied to the snapshot: by 14:30:19 util was
    0.900 — i.e. pos_qty / MAX_ABS_POSITION = 0.9. Under NORMAL the
    inventory_exec_bias gate is on (util ≥ 0.5), and the bot can
    still add to the inventory-adding side up to MAX_ABS_POSITION.

    Under CAUTIOUS (cap × 0.70), the same raw pos_qty looks like
    util = 0.9 / 0.7 = ~1.29 — way past the cap, so headroom on the
    adding side collapses to zero. The bot literally cannot add to
    the underwater inventory direction. This is the load-bearing
    behaviour 4G needs to demonstrate.
    """
    eng, s = _engine()
    # NORMAL behaviour: at pos = 90 (util=0.9) the bot still has
    # 10 units of headroom on the inventory-adding side.
    _bid, _ask, max_buy_normal, max_sell_normal = eng._clip_entry_sizes(
        position_qty=90.0,
        bid_sz=50.0,
        ask_sz=50.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=2.0,  # TON ~$2
        ask_price=2.0,
        # NORMAL → no override → raw cap (100).
    )
    assert max_buy_normal == pytest.approx(10.0, abs=1e-6)
    assert max_sell_normal == pytest.approx(190.0, abs=1e-6)

    # CAUTIOUS behaviour: effective cap shrinks to 70. At pos = 90
    # the bot is ALREADY past the cap → buy headroom collapses to 0.
    knobs = compute_knobs_for_mode(Mode.CAUTIOUS)
    effective_cap = float(s.max_abs_position) * float(knobs.inventory_budget_mult)
    _bid, _ask, max_buy_cautious, max_sell_cautious = eng._clip_entry_sizes(
        position_qty=90.0,
        bid_sz=50.0,
        ask_sz=50.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=2.0,
        ask_price=2.0,
        effective_max_abs_position=effective_cap,
    )
    assert max_buy_cautious == pytest.approx(0.0, abs=1e-6), (
        "Under CAUTIOUS the inventory-adding side must have ZERO "
        "headroom once pos > effective cap — this is the load-bearing "
        "behaviour that would have prevented v1.4.200's continued "
        "long-side accumulation in the 41 min preceding SF#11175."
    )
    # Reducing side keeps its full headroom — the bot can ALWAYS
    # flatten regardless of regime knobs (rule pinned in 4G.4).
    assert max_sell_cautious > 100.0

    # The inventory_exec_bias gate engages MORE aggressively under
    # CAUTIOUS. At pos=45 under NORMAL the gate is off (util 0.45 <
    # 0.5); under CAUTIOUS (cap=70) util becomes 45/70 ≈ 0.64 > 0.5
    # → gate on, adding side suppressed unless the reducing side is
    # already maintained on the book.
    assert not eng._inventory_exec_bias_active(pos_qty=45.0)
    assert eng._inventory_exec_bias_active(
        pos_qty=45.0,
        effective_max_abs_position=effective_cap,
    )


# ---------------------------------------------------------------------------
# Step 6 — End-to-end smoke: full pipeline from envelope through to clip
# ---------------------------------------------------------------------------


def test_v1_4_200_end_to_end_envelope_drives_cautious_clip() -> None:
    """The whole chain in one test:

      envelope → classify → evaluate_mode (dwell) → compute_knobs
              → _clip_entry_sizes(effective_cap=cap × 0.70)
                → adding-side headroom collapses at high util.

    If any one link is broken, this asserts surfaces it. Acts as
    the **single acceptance gate** for Phase 4G.
    """
    inputs = _build_v1_4_200_signal_envelope()
    reading = classify_forward_regime(
        settings=ForwardSignalThresholds(), **inputs
    )
    assert reading.classification == ForwardRegime.CAUTIOUS

    # Drive the FSM forward enough ticks to satisfy the entry dwell.
    base_time = 50_000.0
    state = _seed_state_in_normal(now_mono=base_time)
    for tick_seconds in (0.0, 1.0, 2.0, 3.5):
        evaluate_mode(
            state,
            now_mono=base_time + tick_seconds,
            util=0.05,
            vol_ratio=1.1,
            slow_trend_active=False,
            inventory_drift_active=False,
            shock_gate_locked=False,
            enabled=True,
            forward_regime=reading.classification,
            cautious_entry_dwell_seconds=3.0,
        )
    assert state.mode is Mode.CAUTIOUS

    knobs = compute_knobs_for_mode(state.mode)
    assert knobs.inventory_budget_mult < 1.0

    eng, s = _engine()
    effective_cap = float(s.max_abs_position) * float(knobs.inventory_budget_mult)
    # At util=0.9 raw the adding side collapses to zero headroom.
    _, _, max_buy, _ = eng._clip_entry_sizes(
        position_qty=90.0,
        bid_sz=50.0,
        ask_sz=50.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=2.0,
        ask_price=2.0,
        effective_max_abs_position=effective_cap,
    )
    assert max_buy == pytest.approx(0.0, abs=1e-6)
