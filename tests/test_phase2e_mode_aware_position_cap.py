"""Phase 2E (v1.5.26) -- mode-aware position cap.

The infrastructure already shipped at v1.4.210 (Phase 4G.4) via
``RegimeKnobs.inventory_budget_mult`` -> ``QuoteBuildContext
.effective_max_abs_position`` -> place-time consumer reads. 2E
finalises the per-mode VALUES to the operator's plan-spec:

    NORMAL    : 1.0  (full cap)
    CALM      : 1.0  (full cap; 4G layer)
    CAUTIOUS  : 0.7  (4G layer; pre-existing)
    DEFENSIVE : 0.6  (2E spec; was 1.0 pre-2E)
    SHOCK     : 0.3  (2E spec; was 0.5 pre-2E)

The cap is consulted at PLACE time only -- existing inventory
above the new cap reduces naturally without a forced flatten.
"""

from __future__ import annotations

import pytest

from app.regime_controller import (
    Mode,
    RegimeKnobs,
    compute_knobs_for_mode,
)


def test_normal_inventory_budget_full():
    """NORMAL = unrestricted cap (multiplier 1.0)."""
    k = compute_knobs_for_mode(Mode.NORMAL)
    assert k.inventory_budget_mult == 1.0


def test_calm_inventory_budget_full():
    """CALM = unrestricted cap (Phase 4G upshift layer; 2E doesn't
    change this)."""
    k = compute_knobs_for_mode(Mode.CALM)
    assert k.inventory_budget_mult == 1.0


def test_cautious_inventory_budget_seventy_percent():
    """CAUTIOUS = 70% cap (Phase 4G layer; predates 2E but matches
    the spirit of the 2E mode-ladder)."""
    k = compute_knobs_for_mode(Mode.CAUTIOUS)
    assert k.inventory_budget_mult == 0.70


def test_defensive_inventory_budget_sixty_percent_v1_5_26():
    """v1.5.26 Phase 2E: DEFENSIVE = 60% cap. Pre-fix this was 1.0
    (the default; no override). The plan-spec calls for 0.6."""
    k = compute_knobs_for_mode(Mode.DEFENSIVE)
    assert k.inventory_budget_mult == 0.60


def test_shock_inventory_budget_thirty_percent_v1_5_26():
    """v1.5.26 Phase 2E: SHOCK = 30% cap. Pre-2E was 0.50 (4G.4
    default at v1.4.210). The 2E plan-spec calls for 0.30 -- the
    tightest cap because the bot has already hit a hard signal
    (shock_gate fire) and we want minimal worst-case position size."""
    k = compute_knobs_for_mode(Mode.SHOCK)
    assert k.inventory_budget_mult == 0.30


def test_inventory_budget_mult_ordering():
    """Sanity: the cap tightens as the mode escalates (CALM/NORMAL
    full -> CAUTIOUS -> DEFENSIVE -> SHOCK)."""
    assert compute_knobs_for_mode(Mode.CALM).inventory_budget_mult == 1.0
    assert compute_knobs_for_mode(Mode.NORMAL).inventory_budget_mult == 1.0
    assert (
        compute_knobs_for_mode(Mode.NORMAL).inventory_budget_mult
        > compute_knobs_for_mode(Mode.CAUTIOUS).inventory_budget_mult
    )
    assert (
        compute_knobs_for_mode(Mode.CAUTIOUS).inventory_budget_mult
        > compute_knobs_for_mode(Mode.DEFENSIVE).inventory_budget_mult
    )
    assert (
        compute_knobs_for_mode(Mode.DEFENSIVE).inventory_budget_mult
        > compute_knobs_for_mode(Mode.SHOCK).inventory_budget_mult
    )


def test_regime_knobs_default_is_full_budget():
    """``RegimeKnobs()`` constructed without explicit override should
    default to ``inventory_budget_mult=1.0`` -- the back-compat
    behaviour that lets a caller pass an empty struct and get
    unconstrained behaviour."""
    k = RegimeKnobs()
    assert k.inventory_budget_mult == 1.0
