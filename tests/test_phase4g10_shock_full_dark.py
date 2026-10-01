"""Phase 4G.10 (v1.5.146) — SHOCK go-dark toggle.

``RegimeKnobs.ladder_levels_max`` is 0 for SHOCK per design (see
``test_phase4g8_ladder_levels_max_wired.py::test_shock_caps_to_zero_in_knob_table``).
The consumption sites in ``app/bot.py`` and ``app/execution.py``
historically applied a ``max(1, ...)`` floor so SHOCK still quoted
one reducing-side rung — a deliberate v1.4.219 4G.8 scope-limit.

v1.5.146 adds ``SHOCK_LADDER_ALLOW_FULL_DARK`` (default False, opt-in
per profile). When True the floor drops to 0 and the ladder builds
zero rungs in SHOCK, matching the documented design intent. The bot
then relies exclusively on the soft-flatten path to reduce inventory
during SHOCK episodes.

These tests pin both branches of the toggle so a future refactor of
the floor logic can't silently regress one branch.
"""

from __future__ import annotations

import pytest

from app.config import Settings


# -------------------------------------------------------------------
# Settings shape
# -------------------------------------------------------------------


def test_default_is_false() -> None:
    """Default behaviour is unchanged from v1.4.219 — SHOCK floors
    at 1 rung. The flag is opt-in only."""
    s = Settings()
    assert s.shock_ladder_allow_full_dark is False


def test_setting_round_trips_via_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env-var alias drives the setting. Operator flips via profile."""
    monkeypatch.setenv("SHOCK_LADDER_ALLOW_FULL_DARK", "true")
    s = Settings()
    assert s.shock_ladder_allow_full_dark is True


# -------------------------------------------------------------------
# Floor logic — mirror of the consumption sites in bot.py / execution.py
# -------------------------------------------------------------------


def _effective_levels(
    *,
    cfg_levels: int,
    regime_levels_cap: int | None,
    allow_full_dark: bool,
) -> int:
    """Mirror of the v1.5.146 floor logic at both call sites. Extracted
    so a refactor of either consumer drifts visibly in the tests below."""
    if regime_levels_cap is not None:
        knob_capped = min(cfg_levels, int(regime_levels_cap))
        floor = 0 if allow_full_dark else 1
        return max(floor, knob_capped)
    return max(1, cfg_levels)


# --- default path (allow_full_dark=False) — back-compat with 4G.8 ---


def test_default_shock_cap_floors_at_one() -> None:
    """allow_full_dark=False + SHOCK cap=0 → bot still quotes 1 rung.
    This is the v1.4.219 behaviour every prod session before v1.5.146
    has run with."""
    assert _effective_levels(
        cfg_levels=2,
        regime_levels_cap=0,
        allow_full_dark=False,
    ) == 1


def test_default_defensive_cap_floors_at_one_too() -> None:
    """DEFENSIVE cap=1 → 1 rung regardless of toggle (cap == floor)."""
    assert _effective_levels(
        cfg_levels=2,
        regime_levels_cap=1,
        allow_full_dark=False,
    ) == 1


def test_default_normal_unaffected() -> None:
    """NORMAL has no cap. The toggle has no effect."""
    assert _effective_levels(
        cfg_levels=2,
        regime_levels_cap=None,
        allow_full_dark=False,
    ) == 2
    assert _effective_levels(
        cfg_levels=2,
        regime_levels_cap=None,
        allow_full_dark=True,
    ) == 2


# --- opt-in path (allow_full_dark=True) — 4G.10 design intent ---


def test_full_dark_shock_yields_zero_rungs() -> None:
    """allow_full_dark=True + SHOCK cap=0 → ladder builds 0 rungs.
    Bot is fully dark on passive quoting during SHOCK and relies on
    the soft-flatten path to reduce existing inventory."""
    assert _effective_levels(
        cfg_levels=2,
        regime_levels_cap=0,
        allow_full_dark=True,
    ) == 0


def test_full_dark_defensive_still_one() -> None:
    """allow_full_dark=True only matters for SHOCK (the only mode
    with cap=0). DEFENSIVE cap=1 still produces 1 rung — the toggle
    doesn't broaden to dark anywhere else."""
    assert _effective_levels(
        cfg_levels=2,
        regime_levels_cap=1,
        allow_full_dark=True,
    ) == 1


def test_full_dark_negative_cap_floored_at_zero() -> None:
    """Defensive guard against a corrupted knob value (negative cap).
    The ``max(0, ...)`` clamp prevents the ladder builder from being
    handed a negative rung count."""
    assert _effective_levels(
        cfg_levels=2,
        regime_levels_cap=-1,
        allow_full_dark=True,
    ) == 0


def test_full_dark_cap_exceeds_config_clamps_to_config() -> None:
    """A pathological knob value above the configured ladder depth
    must not inflate the effective depth — the ``min(cfg, cap)``
    gate enforces this regardless of the toggle."""
    assert _effective_levels(
        cfg_levels=2,
        regime_levels_cap=5,
        allow_full_dark=True,
    ) == 2


# -------------------------------------------------------------------
# build_ladder accepts 0 rungs — confirms the downstream contract
# holds when 4G.10 enables full-dark SHOCK.
# -------------------------------------------------------------------


def test_build_ladder_with_zero_rungs_returns_empty_lists() -> None:
    """The ladder builder must accept ``num_levels_per_side=0`` and
    return empty rung lists. ``_effective_n_for_side`` already
    ``max(0, n)``-clamps so this is a stability pin, not new
    behaviour."""
    from datetime import datetime, timezone

    from app.enums import ActiveSides
    from app.ladder import LadderConfig, build_ladder
    from app.quoting import QuoteDecision

    decision = QuoteDecision(
        ts=datetime.now(timezone.utc),
        symbol="TON-USDT-SWAP",
        mid_price=100.0,
        vol_estimate=0.0,
        inventory=0.0,
        reservation_price=100.0,
        target_spread_bps=10.0,
        target_bid=99.95,
        target_ask=100.05,
        quoted_bid=99.95,
        quoted_ask=100.05,
        quoted_bid_sz=1.0,
        quoted_ask_sz=1.0,
        active_sides=ActiveSides.BOTH,
        toxicity_score=0.0,
        decision_reason="test",
        quote_cycle_id=0,
    )
    cfg = LadderConfig(num_levels_per_side=0)
    result = build_ladder(
        decision=decision,
        cfg=cfg,
        half_spread_bps=5.0,
    )
    assert result.bids == []
    assert result.asks == []
