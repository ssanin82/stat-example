"""Phase 4G.4 — inventory_budget_mult wiring tests.

Phase 4G shipped a forward-looking regime classifier (CALM / NORMAL /
CAUTIOUS) on top of the existing reactive FSM (DEFENSIVE / SHOCK). The
mode → knobs overlay (``RegimeKnobs``) added a fourth axis,
``inventory_budget_mult``, so CAUTIOUS / SHOCK can shrink the effective
inventory cap independently of the spread / size / ladder axes.

4G.4 plumbs that multiplier from ``state.regime_knobs`` through the
``QuoteBuildContext.effective_max_abs_position`` field down to the two
place-time consumers in ``QuoteEngine``:

  * ``_clip_entry_sizes`` — caps new bid/ask sizes against the
    available headroom under the per-side cap. With the override,
    CAUTIOUS sees ``settings.max_abs_position × 0.7`` as the ceiling
    instead of the raw setting.
  * ``_inventory_exec_bias_active`` — gates adding-side suppression on
    the util = abs(pos_qty) / cap ratio. Shrinking the cap raises util
    at the SAME raw position, so the gate engages earlier under
    CAUTIOUS.

The SF clips and the hard risk-kill check are NOT consumed — they
keep the raw ``settings.max_abs_position`` so the bot can always
flatten regardless of the regime overlay.

Tests:
  * ``test_backward_compat_none_passes_raw_max_abs_position`` — the
    pre-4G call path (no override) keeps using ``settings.max_abs_position``.
  * ``test_cautious_multiplier_shrinks_clip_headroom`` — a 0.7 mult
    reduces ``_clip_entry_sizes`` headroom proportionally.
  * ``test_shock_multiplier_shrinks_clip_more_aggressively`` — 0.5
    mult (SHOCK default) cuts headroom in half.
  * ``test_inventory_exec_bias_engages_earlier_with_shrunken_cap`` —
    at a raw pos that would NOT engage the gate under NORMAL, the
    CAUTIOUS-shrunken cap pushes util past the threshold and the
    gate engages.
  * ``test_zero_or_negative_effective_cap_falls_back_to_setting`` —
    defensive: if a future caller passes 0 or a negative number, we
    fall back to the raw setting rather than producing nonsense.
  * ``test_regime_knobs_default_passes_none`` — ``RegimeKnobs()``
    default has ``inventory_budget_mult == 1.0``; sanity check that
    the default doesn't perturb the pre-4G path.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest

from app.quote_engine import QuoteEngine
from app.regime_controller import RegimeKnobs
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _engine(**overrides):
    path = Path(tempfile.gettempdir()) / f"mm_4g4_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    base = {
        "TRADING_ENABLED": True,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "DATABASE_URL": f"sqlite:///{path.as_posix()}",
        "MAX_ABS_POSITION": 100.0,
        "MAX_POSITION_NOTIONAL_USD": 100000.0,  # large; not the binding cap here
        "MAX_ORDER_NOTIONAL_USD": 50000.0,
        "INVENTORY_EXEC_BIAS_RATIO": 0.5,
        "INVENTORY_EXEC_BIAS_MIN_UTIL_PCT": 0.5,
    }
    base.update(overrides)
    s = UnitTestSettings.model_validate(base)
    eng = QuoteEngine(s, mock_mm_client().symbol_spec)
    return eng, s


# -------------------------------------------------------------------
# _clip_entry_sizes — the size clipper
# -------------------------------------------------------------------


def test_backward_compat_none_passes_raw_max_abs_position() -> None:
    """Pre-4G call path (no override) keeps the raw cap."""
    eng, s = _engine()
    # Position = 0, raw cap = 100. Bid headroom should be 100 base units
    # (modulo the 1e-9 epsilon). USD cap is huge so non-binding here.
    bid, ask, max_buy, max_sell = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=50.0,
        ask_sz=50.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
        # effective_max_abs_position omitted → backward-compat.
    )
    # Headroom = 100; requested 50 → unclipped.
    assert bid == pytest.approx(50.0)
    assert ask == pytest.approx(50.0)
    assert max_buy == pytest.approx(100.0, abs=1e-6)
    assert max_sell == pytest.approx(100.0, abs=1e-6)


def test_cautious_multiplier_shrinks_clip_headroom() -> None:
    """CAUTIOUS default mult 0.7 → headroom = 100 × 0.7 = 70."""
    eng, s = _engine()
    bid, ask, max_buy, max_sell = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
        effective_max_abs_position=70.0,
    )
    assert bid == pytest.approx(70.0, abs=1e-6)
    assert ask == pytest.approx(70.0, abs=1e-6)
    assert max_buy == pytest.approx(70.0, abs=1e-6)
    assert max_sell == pytest.approx(70.0, abs=1e-6)


def test_shock_multiplier_shrinks_clip_more_aggressively() -> None:
    """SHOCK default mult 0.5 → headroom = 100 × 0.5 = 50."""
    eng, s = _engine()
    bid, ask, max_buy, max_sell = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
        effective_max_abs_position=50.0,
    )
    assert bid == pytest.approx(50.0, abs=1e-6)
    assert ask == pytest.approx(50.0, abs=1e-6)


def test_zero_or_negative_effective_cap_falls_back_to_setting() -> None:
    """Defensive: 0 or negative → fall back to raw setting.

    The guard inside _clip_entry_sizes checks
    ``effective_max_abs_position is not None and > 0`` — otherwise
    raw ``settings.max_abs_position`` (=100) wins.
    """
    eng, s = _engine()
    bid, _ask, max_buy, _max_sell = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
        effective_max_abs_position=0.0,  # invalid → fallback
    )
    # Raw cap 100 wins → headroom 100, requested 100 → unclipped.
    assert bid == pytest.approx(100.0, abs=1e-6)
    assert max_buy == pytest.approx(100.0, abs=1e-6)

    bid2, _ask2, max_buy2, _max_sell2 = eng._clip_entry_sizes(
        position_qty=0.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
        effective_max_abs_position=-5.0,  # negative → fallback
    )
    assert bid2 == pytest.approx(100.0, abs=1e-6)
    assert max_buy2 == pytest.approx(100.0, abs=1e-6)


def test_long_position_with_cautious_cap_reduces_buy_headroom() -> None:
    """At pos = 50 long, raw cap 100 gives 50 buy headroom.
    Under CAUTIOUS (cap 70), buy headroom is only 20."""
    eng, s = _engine()
    # Baseline NORMAL behaviour.
    _bid_nb, _ask_nb, max_buy_n, _max_sell_n = eng._clip_entry_sizes(
        position_qty=50.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
    )
    assert max_buy_n == pytest.approx(50.0, abs=1e-6)

    # CAUTIOUS.
    _bid_c, _ask_c, max_buy_c, _max_sell_c = eng._clip_entry_sizes(
        position_qty=50.0,
        bid_sz=100.0,
        ask_sz=100.0,
        resting_bid_sz=0.0,
        resting_ask_sz=0.0,
        bid_price=1.0,
        ask_price=1.0,
        effective_max_abs_position=70.0,
    )
    assert max_buy_c == pytest.approx(20.0, abs=1e-6)


# -------------------------------------------------------------------
# _inventory_exec_bias_active — the adding-side suppression gate
# -------------------------------------------------------------------


def test_inventory_exec_bias_off_when_pos_low() -> None:
    """Sanity: at low utilisation, the gate is OFF regardless of cap."""
    eng, s = _engine(
        INVENTORY_EXEC_BIAS_RATIO=0.5,
        INVENTORY_EXEC_BIAS_MIN_UTIL_PCT=0.5,
    )
    # util = 10/100 = 0.10 → way below 0.5 threshold.
    assert not eng._inventory_exec_bias_active(pos_qty=10.0)
    # Even with CAUTIOUS cap (70), util = 10/70 ≈ 0.143 → still below.
    assert not eng._inventory_exec_bias_active(
        pos_qty=10.0,
        effective_max_abs_position=70.0,
    )


def test_inventory_exec_bias_engages_earlier_with_shrunken_cap() -> None:
    """The headline 4G.4 acceptance: a raw position that does NOT
    engage the gate under NORMAL DOES engage it under CAUTIOUS.

    util_threshold = 0.5. Raw cap = 100.
    Pick pos_qty = 40:
      * NORMAL: util = 40/100 = 0.40 → BELOW 0.5 → gate OFF.
      * CAUTIOUS (cap 70): util = 40/70 ≈ 0.571 → ABOVE 0.5 → gate ON.
    """
    eng, s = _engine(
        INVENTORY_EXEC_BIAS_RATIO=0.5,
        INVENTORY_EXEC_BIAS_MIN_UTIL_PCT=0.5,
    )
    # NORMAL — pre-4G call path.
    assert not eng._inventory_exec_bias_active(pos_qty=40.0)
    # CAUTIOUS — shrunken cap pushes util past the gate.
    assert eng._inventory_exec_bias_active(
        pos_qty=40.0,
        effective_max_abs_position=70.0,
    )


def test_inventory_exec_bias_shock_engages_at_even_lower_pos() -> None:
    """SHOCK mult 0.5 → cap 50. At pos = 30:
      * NORMAL: 30/100 = 0.30 → OFF.
      * CAUTIOUS: 30/70 ≈ 0.43 → still OFF.
      * SHOCK: 30/50 = 0.60 → ON.
    """
    eng, s = _engine(
        INVENTORY_EXEC_BIAS_RATIO=0.5,
        INVENTORY_EXEC_BIAS_MIN_UTIL_PCT=0.5,
    )
    assert not eng._inventory_exec_bias_active(pos_qty=30.0)
    assert not eng._inventory_exec_bias_active(
        pos_qty=30.0,
        effective_max_abs_position=70.0,
    )
    assert eng._inventory_exec_bias_active(
        pos_qty=30.0,
        effective_max_abs_position=50.0,
    )


def test_inventory_exec_bias_negative_position_uses_abs() -> None:
    """Short side is symmetric: pos_qty = -40, CAUTIOUS cap 70
    → util = 40/70 ≈ 0.571 → ON."""
    eng, s = _engine(
        INVENTORY_EXEC_BIAS_RATIO=0.5,
        INVENTORY_EXEC_BIAS_MIN_UTIL_PCT=0.5,
    )
    assert not eng._inventory_exec_bias_active(pos_qty=-40.0)
    assert eng._inventory_exec_bias_active(
        pos_qty=-40.0,
        effective_max_abs_position=70.0,
    )


def test_zero_effective_cap_falls_back_for_bias_gate() -> None:
    """Defensive: 0 / negative override → fall back to setting."""
    eng, s = _engine(
        INVENTORY_EXEC_BIAS_RATIO=0.5,
        INVENTORY_EXEC_BIAS_MIN_UTIL_PCT=0.5,
    )
    # Pos = 80, raw cap = 100 → util 0.80 → ON.
    assert eng._inventory_exec_bias_active(pos_qty=80.0)
    # With effective_cap=0 we should fall BACK to raw cap, not divide
    # by zero. Util = 80/100 = 0.80 → still ON.
    assert eng._inventory_exec_bias_active(
        pos_qty=80.0,
        effective_max_abs_position=0.0,
    )
    assert eng._inventory_exec_bias_active(
        pos_qty=80.0,
        effective_max_abs_position=-1.0,
    )


# -------------------------------------------------------------------
# RegimeKnobs default sanity
# -------------------------------------------------------------------


def test_regime_knobs_default_inventory_budget_mult_is_one() -> None:
    """Default knobs (NORMAL) must have ``inventory_budget_mult == 1.0``
    so the execution.py wire-up path passes ``None`` → backward compat.
    """
    k = RegimeKnobs()
    assert k.inventory_budget_mult == pytest.approx(1.0)


def test_regime_knobs_cautious_shock_mult_below_one() -> None:
    """CAUTIOUS / SHOCK must shrink the cap (< 1.0). NORMAL / CALM
    must NOT shrink (== 1.0). Pins the asymmetric design.

    Knobs come from ``compute_knobs_for_mode`` in app/regime_controller.py.
    """
    from app.regime_controller import Mode, compute_knobs_for_mode

    k_normal = compute_knobs_for_mode(Mode.NORMAL)
    assert k_normal.inventory_budget_mult == pytest.approx(1.0)

    k_calm = compute_knobs_for_mode(Mode.CALM)
    assert k_calm.inventory_budget_mult == pytest.approx(1.0)

    k_cautious = compute_knobs_for_mode(Mode.CAUTIOUS)
    assert k_cautious.inventory_budget_mult < 1.0
    assert k_cautious.inventory_budget_mult > 0.0

    k_defensive = compute_knobs_for_mode(Mode.DEFENSIVE)
    # DEFENSIVE was the pre-4G reactive layer; should also be <= 1.0
    # (not strictly required by 4G.4 but pinned for awareness).
    assert k_defensive.inventory_budget_mult <= 1.0

    k_shock = compute_knobs_for_mode(Mode.SHOCK)
    assert k_shock.inventory_budget_mult < k_cautious.inventory_budget_mult
    assert k_shock.inventory_budget_mult > 0.0
