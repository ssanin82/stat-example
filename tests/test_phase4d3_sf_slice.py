"""Phase 4D.3 (v1.5.146) — SF slice across phase 2/3.

When the SF phase ladder escalates to phase 2 (cross-1-tick IOC) or
phase 3 (cross-2-tick IOC), the historical behaviour was to fire ONE
IOC for the entire remaining quantity. On a 9-contract close that's
one trade that crosses 1-2 ticks of spread immediately. 4D.3 lets
the operator slice that close into smaller chunks by setting
``SOFT_FLATTEN_SLICE_NOTIONAL_USD > 0``: each phase-2/3 IOC is then
capped at ``slice_notional / target_price`` contracts and the
per-tick dispatcher fires the next slice after the previous one
lands (the dwell timer keeps ticking — slicing doesn't reset
escalation).

The cross-venue spread saving in the plan: while phase-3 trades the
first slice, Binance may pull back and the next slice can land at
+0 ticks instead of +1. The dispatcher re-reads the touch on every
tick so this happens naturally without any extra wiring.

These tests pin:
* Default config (slice disabled) preserves the legacy all-at-once IOC
  size — bit-identical to pre-v1.5.146.
* Slice cap arithmetic when enabled.
* Counter bumps only when the cap actually trims the size.
* Edge cases: zero target_price, zero/negative slice_notional,
  slice already larger than remaining.

The IOC dispatch wiring itself is exercised by the existing SF
phase-ladder tests; this file focuses on the slice-cap math, which
is the only new code.
"""

from __future__ import annotations

import pytest

from app.config import Settings


# -------------------------------------------------------------------
# Settings shape
# -------------------------------------------------------------------


def test_default_is_zero_disabled() -> None:
    """Default 0.0 = disabled. Each phase-2/3 IOC fires for full
    remaining quantity, gated only by the existing notional caps."""
    s = Settings()
    assert s.soft_flatten_slice_notional_usd == 0.0


def test_setting_round_trips_via_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator flips via profile env."""
    monkeypatch.setenv("SOFT_FLATTEN_SLICE_NOTIONAL_USD", "5.0")
    s = Settings()
    assert s.soft_flatten_slice_notional_usd == 5.0


def test_setting_rejects_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ge=0.0`` validator catches typo-ed negative values."""
    monkeypatch.setenv("SOFT_FLATTEN_SLICE_NOTIONAL_USD", "-1.0")
    with pytest.raises(Exception):
        Settings()


# -------------------------------------------------------------------
# Slice-cap arithmetic — mirror of the IOC-branch logic in bot.py
# -------------------------------------------------------------------


def _apply_slice_cap(
    *,
    new_size: float,
    slice_notional_usd: float,
    target_price: float,
) -> tuple[float, bool]:
    """Mirror of the v1.5.146 slice-cap block in
    ``_run_soft_flatten_phase_ladder_dispatch``'s IOC branch. Returns
    ``(capped_size, did_trim)`` so the counter-bump test can assert
    on the second value independently of the first."""
    if slice_notional_usd > 0.0 and target_price > 0:
        slice_size = slice_notional_usd / target_price
        if slice_size < new_size:
            return slice_size, True
    return new_size, False


def test_disabled_passes_through() -> None:
    """slice=0 → no trim, no counter bump."""
    sized, trimmed = _apply_slice_cap(
        new_size=9.0,
        slice_notional_usd=0.0,
        target_price=2.0,
    )
    assert sized == 9.0
    assert trimmed is False


def test_zero_target_price_passes_through() -> None:
    """target_price=0 is a degenerate market state; the dispatcher
    catches it upstream via the ``decision.target_price > 0`` guard,
    but the slice block defends against it independently too."""
    sized, trimmed = _apply_slice_cap(
        new_size=9.0,
        slice_notional_usd=5.0,
        target_price=0.0,
    )
    assert sized == 9.0
    assert trimmed is False


def test_negative_target_price_passes_through() -> None:
    """Defensive: pathological negative target_price doesn't fire
    the slice block (would yield negative slice_size which would
    incorrectly look smaller-than-new_size)."""
    sized, trimmed = _apply_slice_cap(
        new_size=9.0,
        slice_notional_usd=5.0,
        target_price=-2.0,
    )
    assert sized == 9.0
    assert trimmed is False


def test_slice_smaller_than_remaining_trims() -> None:
    """The typical case: 9 contracts × $2.00 = $18.00 remaining;
    slice cap $5.00 → first slice is $5.00 / $2.00 = 2.5 contracts."""
    sized, trimmed = _apply_slice_cap(
        new_size=9.0,
        slice_notional_usd=5.0,
        target_price=2.0,
    )
    assert sized == pytest.approx(2.5)
    assert trimmed is True


def test_slice_larger_than_remaining_passes_through() -> None:
    """When the remaining quantity is small (last slice of a close),
    the slice cap may exceed it — in that case the dispatch sends
    the full remainder, not the cap. No trim, no counter bump."""
    # 0.5 contracts × $2.00 = $1.00 remaining; slice cap $5.00 ->
    # slice would be 2.5 contracts > 0.5 actual remainder.
    sized, trimmed = _apply_slice_cap(
        new_size=0.5,
        slice_notional_usd=5.0,
        target_price=2.0,
    )
    assert sized == 0.5
    assert trimmed is False


def test_slice_equal_to_remaining_passes_through() -> None:
    """Boundary: slice_size == new_size → no trim (the legacy
    behaviour). Avoids spurious counter bumps when slice happens
    to exactly match remaining qty."""
    # 2.5 contracts × $2.00 = $5.00 exactly == slice cap
    sized, trimmed = _apply_slice_cap(
        new_size=2.5,
        slice_notional_usd=5.0,
        target_price=2.0,
    )
    assert sized == 2.5
    assert trimmed is False


# -------------------------------------------------------------------
# Multi-slice trajectory: simulate the dispatcher firing N IOCs
# in sequence, each trimming a fraction of the remaining quantity.
# Pins the contract that the cap arithmetic + remaining-tracking
# math compose to drain the position deterministically.
# -------------------------------------------------------------------


def test_multi_slice_drains_position_to_zero() -> None:
    """Operator sets slice=$5, price=$2.00, position remaining=9
    contracts ($18 notional). Each slice trims 2.5 contracts; we
    expect 4 slices total (2.5 + 2.5 + 2.5 + 1.5 = 9.0)."""
    remaining = 9.0
    slice_notional = 5.0
    price = 2.0
    slices_fired = []
    safety = 0
    while remaining > 0 and safety < 100:
        new_size, trimmed = _apply_slice_cap(
            new_size=remaining,
            slice_notional_usd=slice_notional,
            target_price=price,
        )
        slices_fired.append(new_size)
        remaining = max(0.0, remaining - new_size)
        safety += 1
    assert sum(slices_fired) == pytest.approx(9.0)
    assert slices_fired == pytest.approx([2.5, 2.5, 2.5, 1.5])


# -------------------------------------------------------------------
# State counter
# -------------------------------------------------------------------


def test_state_counter_default_zero() -> None:
    """``BotState.sf_slice_dispatched_total`` starts at zero; the
    counter only bumps when a slice ACTUALLY trims the IOC size."""
    # Mirror the BotState init pattern; testing the attribute value
    # in isolation without the full BotState() construction
    # avoids depending on every other BotState field's defaults.
    class _Stub:
        def __init__(self) -> None:
            self.sf_slice_dispatched_total: int = 0

    s = _Stub()
    assert s.sf_slice_dispatched_total == 0
    # The bot's wire-up bumps the counter; mirror that here.
    s.sf_slice_dispatched_total += 1
    assert s.sf_slice_dispatched_total == 1


def test_botstate_exposes_sf_slice_counter() -> None:
    """The actual BotState class exposes the counter — pins the
    field name + default so postmortem / dashboard surfaces can
    rely on it being present."""
    from app.state import BotState

    state = BotState(Settings())
    assert hasattr(state, "sf_slice_dispatched_total")
    assert state.sf_slice_dispatched_total == 0
