"""Tests for v1.4.83 Phase 4A — typed BuildCommand sum-type.

The legacy ``QuoteBuildResult`` is a single dataclass with optional
fields whose semantics depend on the string ``mode`` field. Phase 4A
introduces a typed sum-type (``QuoteBoth | QuoteOneSided | NoQuote |
ResidualFlatten``) and a ``to_build_command()`` converter so the
consumer can ``match`` exhaustively. The cutover at the consumer is
sequenced for after 4B/4C — Phase 4A is additive infrastructure.

Tests verify:

* The 4 variants are frozen dataclasses (immutability).
* ``QuoteBuildResult.to_build_command()`` maps every legacy ``mode``
  to the right variant, with telemetry / hard-cancel data preserved.
* The union alias ``BuildCommand`` matches all 4 variants.
* Inside-rung hard-cancel tuples are embedded into the multi-rung map
  (matches the v1.4.70 Phase 1C canonical-source invariant).
"""

from __future__ import annotations

import pytest

from app.enums import Side
from app.quote_engine import (
    BuildCommand,
    FinalQuoteOrder,
    NoQuote,
    QuoteBoth,
    QuoteBuildResult,
    QuoteOneSided,
    ResidualFlatten,
)


def _bid(price: float = 2.000, size: float = 1.0) -> FinalQuoteOrder:
    return FinalQuoteOrder(side=Side.BUY, price=price, size=size)


def _ask(price: float = 2.002, size: float = 1.0) -> FinalQuoteOrder:
    return FinalQuoteOrder(side=Side.SELL, price=price, size=size)


# ---------------------------------------------------------------------------
# Variant construction + immutability
# ---------------------------------------------------------------------------


def test_phase4a_quote_both_is_frozen() -> None:
    cmd = QuoteBoth(bid=_bid(), ask=_ask())
    assert cmd.bid.price == 2.000
    assert cmd.ask.price == 2.002
    with pytest.raises((AttributeError, Exception)):
        cmd.bid = _bid()  # type: ignore[misc]


def test_phase4a_quote_one_sided_is_frozen() -> None:
    cmd = QuoteOneSided(
        side=Side.BUY,
        order=_bid(),
        suppressed_side_reason="ask_inventory_bias",
    )
    assert cmd.side == Side.BUY
    assert cmd.suppressed_side_reason == "ask_inventory_bias"
    with pytest.raises((AttributeError, Exception)):
        cmd.order = _bid()  # type: ignore[misc]


def test_phase4a_no_quote_is_frozen() -> None:
    cmd = NoQuote(reason="bbo_missing")
    assert cmd.reason == "bbo_missing"
    with pytest.raises((AttributeError, Exception)):
        cmd.reason = "x"  # type: ignore[misc]


def test_phase4a_residual_flatten_is_frozen() -> None:
    cmd = ResidualFlatten(target_qty=0.0)
    assert cmd.target_qty == 0.0
    with pytest.raises((AttributeError, Exception)):
        cmd.target_qty = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Sum-type membership (the union alias)
# ---------------------------------------------------------------------------


def test_phase4a_build_command_union_includes_all_variants() -> None:
    """The union alias resolves to the 4 dataclass variants at runtime.

    ``isinstance(x, get_args(BuildCommand))`` would be the strict check
    but the simpler runtime check is just "are these all assignable to
    the union type" — which we verify via direct construction.
    """
    variants: list[BuildCommand] = [
        QuoteBoth(bid=_bid(), ask=_ask()),
        QuoteOneSided(side=Side.BUY, order=_bid(), suppressed_side_reason="r"),
        NoQuote(reason="r"),
        ResidualFlatten(target_qty=0.0),
    ]
    # All four should be acceptable as BuildCommand.
    for v in variants:
        assert isinstance(
            v,
            (QuoteBoth, QuoteOneSided, NoQuote, ResidualFlatten),
        )


# ---------------------------------------------------------------------------
# QuoteBuildResult.to_build_command() — legacy → typed conversion
# ---------------------------------------------------------------------------


def test_phase4a_converter_two_sided_maps_to_quote_both() -> None:
    legacy = QuoteBuildResult(
        bid_order=_bid(),
        ask_order=_ask(),
        mode="two_sided",
        telemetry={"quote_engine_mode": "two_sided"},
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, QuoteBoth)
    assert cmd.bid is legacy.bid_order
    assert cmd.ask is legacy.ask_order
    assert cmd.telemetry["quote_engine_mode"] == "two_sided"


def test_phase4a_converter_one_sided_bid_maps_to_quote_one_sided() -> None:
    legacy = QuoteBuildResult(
        bid_order=_bid(),
        ask_order=None,
        mode="one_sided",
        telemetry={"quote_engine_suppressed_ask_reason": "inventory_bias_ask"},
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, QuoteOneSided)
    assert cmd.side == Side.BUY
    assert cmd.order is legacy.bid_order
    assert cmd.suppressed_side_reason == "inventory_bias_ask"


def test_phase4a_converter_one_sided_ask_maps_to_quote_one_sided() -> None:
    legacy = QuoteBuildResult(
        bid_order=None,
        ask_order=_ask(),
        mode="one_sided",
        telemetry={"quote_engine_suppressed_bid_reason": "inventory_bias_bid"},
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, QuoteOneSided)
    assert cmd.side == Side.SELL
    assert cmd.order is legacy.ask_order
    assert cmd.suppressed_side_reason == "inventory_bias_bid"


def test_phase4a_converter_no_quote_maps_to_no_quote() -> None:
    legacy = QuoteBuildResult(
        bid_order=None,
        ask_order=None,
        mode="no_quote",
        telemetry={"quote_engine_no_quote_reason": "bbo_missing"},
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, NoQuote)
    assert cmd.reason == "bbo_missing"


def test_phase4a_converter_residual_flatten_maps_to_residual_flatten() -> None:
    legacy = QuoteBuildResult(
        bid_order=None,
        ask_order=None,
        mode="residual_flatten",
        residual_flatten_requested=True,
        telemetry={
            "quote_engine_mode": "residual_flatten",
            "residual_flatten_reason": "above_force_th_sub_spec",
        },
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, ResidualFlatten)
    assert cmd.target_qty == 0.0
    assert cmd.telemetry["residual_flatten_reason"] == "above_force_th_sub_spec"


def test_phase4a_converter_residual_flatten_via_requested_flag_only() -> None:
    """Even if ``mode`` says "no_quote", a ``residual_flatten_requested=True``
    flag still routes to the ResidualFlatten variant — the engine
    sometimes sets the flag without changing ``mode`` on the legacy path.
    """
    legacy = QuoteBuildResult(
        bid_order=None,
        ask_order=None,
        mode="no_quote",
        residual_flatten_requested=True,
        telemetry={},
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, ResidualFlatten)


# ---------------------------------------------------------------------------
# Hard-cancel propagation
# ---------------------------------------------------------------------------


def test_phase4a_converter_propagates_hard_cancel_by_slot_map() -> None:
    cancels = {
        (Side.BUY, 0): ("at_touch_order_age_seconds",),
        (Side.SELL, 1): ("behind_touch_order_age_seconds",),
    }
    legacy = QuoteBuildResult(
        bid_order=_bid(),
        ask_order=_ask(),
        mode="two_sided",
        telemetry={},
        hard_cancel_by_slot=cancels,
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, QuoteBoth)
    assert cmd.hard_cancel_by_slot == cancels


def test_phase4a_converter_embeds_legacy_inside_rung_tuples_into_map() -> None:
    """Legacy ``hard_cancel_bid_reasons`` / ``hard_cancel_ask_reasons``
    are inside-rung tuples. The converter should embed them into the
    map so post-conversion callers only need to look at the map.
    """
    legacy = QuoteBuildResult(
        bid_order=_bid(),
        ask_order=_ask(),
        mode="two_sided",
        telemetry={},
        hard_cancel_bid_reasons=("at_touch_order_age_seconds",),
        hard_cancel_ask_reasons=("distance_to_touch_ticks",),
        hard_cancel_by_slot={},  # legacy didn't populate the map
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, QuoteBoth)
    assert cmd.hard_cancel_by_slot[(Side.BUY, 0)] == ("at_touch_order_age_seconds",)
    assert cmd.hard_cancel_by_slot[(Side.SELL, 0)] == ("distance_to_touch_ticks",)


def test_phase4a_converter_map_takes_precedence_over_legacy_tuples() -> None:
    """If the multi-rung map already has the inside rung, the legacy
    tuple is NOT overwritten (the map is the canonical source per the
    v1.4.70 Phase 1C invariant)."""
    legacy = QuoteBuildResult(
        bid_order=_bid(),
        ask_order=_ask(),
        mode="two_sided",
        telemetry={},
        hard_cancel_bid_reasons=("OVERWRITE_ME",),
        hard_cancel_by_slot={(Side.BUY, 0): ("canonical_reason",)},
    )
    cmd = legacy.to_build_command()
    assert isinstance(cmd, QuoteBoth)
    assert cmd.hard_cancel_by_slot[(Side.BUY, 0)] == ("canonical_reason",)


# ---------------------------------------------------------------------------
# Exhaustive match — sketch of how the dispatcher will look post-cutover
# ---------------------------------------------------------------------------


def test_phase4a_match_statement_exhaustiveness_smoke() -> None:
    """Sketch of the exhaustive ``match`` pattern the dispatcher will
    use post-cutover. mypy / pyright catch missing cases at edit time;
    this runtime test just confirms each variant routes to its branch.
    """
    def dispatch(cmd: BuildCommand) -> str:
        match cmd:
            case QuoteBoth():
                return "quote_both"
            case QuoteOneSided():
                return "one_sided"
            case NoQuote():
                return "no_quote"
            case ResidualFlatten():
                return "residual_flatten"
        # Unreachable post-exhaustive match. mypy/pyright flag missing
        # cases; the runtime fall-through here is defensive only.
        return "unhandled"

    assert dispatch(QuoteBoth(bid=_bid(), ask=_ask())) == "quote_both"
    assert dispatch(
        QuoteOneSided(side=Side.BUY, order=_bid(), suppressed_side_reason="r")
    ) == "one_sided"
    assert dispatch(NoQuote(reason="r")) == "no_quote"
    assert dispatch(ResidualFlatten(target_qty=0.0)) == "residual_flatten"
