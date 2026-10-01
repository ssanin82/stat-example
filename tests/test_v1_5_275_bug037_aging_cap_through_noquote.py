"""v1.5.275 / BUG-037 — aging-cap signals must propagate through
``NoQuote`` and ``ResidualFlatten`` BuildCommand variants.

Pre-fix root cause (lines 377-379 / 418-420 of `app/quote_engine.py`,
pre v1.5.275): the ``hard_cancel_by_slot`` property on ``NoQuote``
and ``ResidualFlatten`` returned an empty dict. The executor in
``app/execution.py`` reads ``build.hard_cancel_by_slot`` to learn
which (side, level_idx) slots have aged past their cap and should
be cancelled. When the engine emitted ``NoQuote`` or
``ResidualFlatten``, those signals were silently dropped — a
resting order whose tick happened to be a NoQuote (one-sided
regime, CAUTIOUS suppression, eligibility gate, etc.) would
keep resting past ``BEHIND_TOUCH_MAX_AGE_SECONDS`` /
``AT_TOUCH_MAX_AGE_SECONDS``.

Reproduced on snapshot v1.5.271-260529-234638: SELL @ 1.762 on
(SELL, 1) lived 8.03 s past its 3.0 s behind-touch cap (and 5.0 s
at-touch cap). Lifecycle trace from the snapshot's log: 11
consecutive orchestrate_decision entries for (SELL, 1) showed
``cur_status=ACKED, desired_present=False,
action=noop:acked_no_reprice_needed``. The engine produced
``NoQuote`` for those 11 ticks, throwing away the aging cap
signals that ``compute_aging_signals`` had computed.

Fix: ``hard_cancel_by_slot`` is now a real dataclass field on
``NoQuote`` and ``ResidualFlatten`` (mirrors ``QuoteBoth`` /
``QuoteOneSided``); ``QuoteBuildResult.to_build_command()`` passes
the cancels dict through to all four variants.
"""

from __future__ import annotations

import pytest


def test_noquote_carries_hard_cancel_by_slot_field():
    """The NoQuote dataclass now has hard_cancel_by_slot as a real
    field (not a property returning empty dict)."""
    from app.enums import Side
    from app.quote_engine import NoQuote

    # Use rung-1 to verify the multi-rung map carries non-inside slots.
    cancels = {(Side.SELL, 1): ("behind_touch_order_age_seconds",)}
    nq = NoQuote(
        reason="test",
        hard_cancel_by_slot=cancels,
    )
    assert nq.hard_cancel_by_slot == cancels
    # Inside-rung legacy properties read inside-rung only — empty for
    # this rung-1 case. This matches QuoteBoth / QuoteOneSided semantics.
    assert nq.hard_cancel_ask_reasons == ()
    assert nq.hard_cancel_bid_reasons == ()

    # Inside-rung populated → legacy property surfaces it.
    inside = {(Side.BUY, 0): ("order_age_seconds",)}
    nq_inside = NoQuote(reason="test", hard_cancel_by_slot=inside)
    assert nq_inside.hard_cancel_bid_reasons == ("order_age_seconds",)
    assert nq_inside.hard_cancel_ask_reasons == ()


def test_noquote_default_empty_hard_cancel_by_slot():
    """Backwards-compat: existing call-sites that construct
    NoQuote without the cancels field still work — the default is
    an empty dict."""
    from app.quote_engine import NoQuote

    nq = NoQuote(reason="test")
    assert nq.hard_cancel_by_slot == {}
    assert nq.hard_cancel_bid_reasons == ()
    assert nq.hard_cancel_ask_reasons == ()


def test_residualflatten_carries_hard_cancel_by_slot_field():
    """ResidualFlatten parallel to NoQuote — inside-rung population."""
    from app.enums import Side
    from app.quote_engine import ResidualFlatten

    cancels = {(Side.BUY, 0): ("at_touch_order_age_seconds",)}
    rf = ResidualFlatten(
        target_qty=0.0,
        hard_cancel_by_slot=cancels,
    )
    assert rf.hard_cancel_by_slot == cancels
    # Inside-rung legacy property surfaces inside-rung tuple.
    assert rf.hard_cancel_bid_reasons == ("at_touch_order_age_seconds",)
    assert rf.hard_cancel_ask_reasons == ()


def test_residualflatten_default_empty_hard_cancel_by_slot():
    """Backwards-compat default."""
    from app.quote_engine import ResidualFlatten

    rf = ResidualFlatten(target_qty=0.0)
    assert rf.hard_cancel_by_slot == {}


def test_to_build_command_propagates_cancels_to_noquote():
    """The QuoteBuildResult → NoQuote conversion must carry the
    aging-cap signals through. This is the regression test that
    locks in the v1.5.275 fix."""
    from app.enums import Side
    from app.quote_engine import NoQuote, QuoteBuildResult

    cancels = {(Side.SELL, 1): ("behind_touch_order_age_seconds",)}
    result = QuoteBuildResult(
        bid_order=None,
        ask_order=None,
        mode="no_quote",
        telemetry={"quote_engine_no_quote_reason": "eligibility_gate"},
        hard_cancel_bid_reasons=(),
        hard_cancel_ask_reasons=(),
        hard_cancel_by_slot=cancels,
    )
    cmd = result.to_build_command()
    assert isinstance(cmd, NoQuote), (
        f"both orders None should produce NoQuote, got {type(cmd).__name__}"
    )
    assert cmd.hard_cancel_by_slot == cancels, (
        "BUG-037 regression: NoQuote.hard_cancel_by_slot was empty; "
        "aging-cap signals dropped during conversion."
    )
    assert cmd.reason == "eligibility_gate"


def test_to_build_command_propagates_cancels_to_residualflatten():
    """The QuoteBuildResult → ResidualFlatten conversion mirror of
    the above."""
    from app.enums import Side
    from app.quote_engine import QuoteBuildResult, ResidualFlatten

    cancels = {(Side.BUY, 0): ("order_age_seconds",)}
    result = QuoteBuildResult(
        bid_order=None,
        ask_order=None,
        mode="residual_flatten",
        telemetry={},
        hard_cancel_bid_reasons=(),
        hard_cancel_ask_reasons=(),
        hard_cancel_by_slot=cancels,
    )
    cmd = result.to_build_command()
    assert isinstance(cmd, ResidualFlatten), (
        f"residual_flatten mode should produce ResidualFlatten, "
        f"got {type(cmd).__name__}"
    )
    assert cmd.hard_cancel_by_slot == cancels


def test_to_build_command_propagates_legacy_inside_rung_to_noquote():
    """Inside-rung tuples (the legacy single-side cancel API)
    should also flow into NoQuote.hard_cancel_by_slot via the
    same conversion path that QuoteBoth uses."""
    from app.enums import Side
    from app.quote_engine import NoQuote, QuoteBuildResult

    # Legacy caller only sets the inside-rung tuples; per the
    # conversion logic these get folded into hard_cancel_by_slot
    # at (BUY, 0) and (SELL, 0).
    result = QuoteBuildResult(
        bid_order=None,
        ask_order=None,
        mode="no_quote",
        telemetry={},
        hard_cancel_bid_reasons=("order_age_seconds",),
        hard_cancel_ask_reasons=("behind_touch_order_age_seconds",),
        hard_cancel_by_slot={},  # legacy caller doesn't set this
    )
    cmd = result.to_build_command()
    assert isinstance(cmd, NoQuote)
    assert cmd.hard_cancel_by_slot.get((Side.BUY, 0)) == ("order_age_seconds",)
    assert cmd.hard_cancel_by_slot.get((Side.SELL, 0)) == (
        "behind_touch_order_age_seconds",
    )


def test_quoteboth_and_oneside_still_propagate_cancels_unchanged():
    """Regression guard: the existing QuoteBoth / QuoteOneSided
    behavior on hard_cancel_by_slot must NOT change. v1.5.275 only
    edits NoQuote / ResidualFlatten."""
    from app.enums import Side
    from app.quote_engine import (
        FinalQuoteOrder, QuoteBoth, QuoteBuildResult, QuoteOneSided,
    )

    cancels = {
        (Side.BUY, 0): ("order_age_seconds",),
        (Side.SELL, 1): ("behind_touch_order_age_seconds",),
    }
    # FinalQuoteOrder fields: side, price, size, target_half_spread_bps,
    # aging_tighten_applied. Minimal happy-path construction.
    bid = FinalQuoteOrder(side=Side.BUY, price=1.0, size=1.0)
    ask = FinalQuoteOrder(side=Side.SELL, price=1.1, size=1.0)
    result = QuoteBuildResult(
        bid_order=bid, ask_order=ask, mode="two_sided",
        telemetry={}, hard_cancel_bid_reasons=(),
        hard_cancel_ask_reasons=(), hard_cancel_by_slot=cancels,
    )
    cmd = result.to_build_command()
    assert isinstance(cmd, QuoteBoth)
    assert cmd.hard_cancel_by_slot == cancels

    result_one = QuoteBuildResult(
        bid_order=bid, ask_order=None, mode="one_sided",
        telemetry={"quote_engine_suppressed_ask_reason": "test"},
        hard_cancel_bid_reasons=(), hard_cancel_ask_reasons=(),
        hard_cancel_by_slot=cancels,
    )
    cmd_one = result_one.to_build_command()
    assert isinstance(cmd_one, QuoteOneSided)
    assert cmd_one.hard_cancel_by_slot == cancels
