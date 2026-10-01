"""QuoteEngine — single-model quote construction for placement-ready orders.

**One-model invariant**

Quoting in this bot is deliberately *not* layered. There is exactly one model
— :meth:`QuoteEngine.build_quotes` — that takes the strategy intent
(``QuoteDecision`` from :func:`app.quoting.compute_quote_decision`) plus
the live venue constraints (``symbol_spec``: tick, size_step, min_size,
min_notional_usd; settings: ``max_order_notional_usd``) and produces
exchange-ready :class:`FinalQuoteOrder` objects.

What the engine OWNS (and therefore cannot be re-done outside):

1. Tick/size_step rounding (via :func:`round_price_and_size_to_grid`).
2. Post-only price clamping against BBO (no crossing).
3. Economic / aging / spread-floor price adjustments (re-rounded after each).
4. Min-notional **self-heal**: if size_step rounding pushes the order below
   the venue minimum, the size is bumped to the smallest step-multiple
   that clears the minimum, capped at ``max_order_notional_usd``.
5. Inventory execution bias (suppress adding-side if reducing-side is not
   yet maintained on the book).

What the engine REJECTS are only genuine strategy outcomes — no BBO, caps in
conflict, intent too small vs the venue minimum (>4× bump required). It must
never reject for a reason a caller could paper over; that is layered
disagreement and a design bug. Any new constraint added to the venue
pipeline must be repairable in one of the phases above, not become a new
silent-rejection site downstream.

Callers of :meth:`build_quotes` submit the returned orders verbatim. There
is no secondary validation layer in execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from typing import Any, Optional

from app.config import Settings
from app.enums import ActiveSides, OrderStatus, RiskAction, Side
from app.exchange.hyperliquid_precision import (
    normalize_order_pair,
    round_price_and_size_to_grid,
    wire_format_preview_limit_px,
)
from app.exchange.symbol_spec import SymbolSpec
from app.models import BestBidAsk, QuoteDecision, WorkingOrder
from app.quoting import (
    apply_profitability_spread_floor,
    compute_effective_inventory_util_floor_pct,
    compute_effective_min_half_spread_bps,
)
from app.quote_aging import (
    MarkoutAdverseTracker,
    adjust_ask_target_for_aging,
    adjust_bid_target_for_aging,
    apply_inventory_high_adding_side_buffer,
    compute_normal_mm_market_capped_half_spread_bps,
    hard_reprice_reasons_buy,
    hard_reprice_reasons_sell,
    markout_adverse_bps_for_side,
    widen_two_sided_if_collapsed_to_one_tick,
)
from app.utils.math import clip
from app.utils.time import utc_now

_QE_POS_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class FinalQuoteOrder:
    """Grid-rounded, exchange-ready limit from QuoteEngine.build_quotes (do not reshape in execution)."""

    side: Side
    price: float
    size: float
    # todo-005 / todo-006 quality tags carried from the QuoteDecision +
    # quote_aging diagnostic so OrderManager can stamp them onto the
    # WorkingOrder at place-time. ``target_half_spread_bps`` is the
    # per-side half-spread the QuoteDecision asked for *before* aging
    # adjustments — the operator's "we configured X bps" reference for
    # the spread-quality report. ``aging_tighten_applied`` is True when
    # ``adjust_*_target_for_aging`` actively pushed the price tighter
    # than the model-computed target on this cycle (the
    # "aged_tightened" aggressiveness category). Both default-None so
    # non-MM placement paths (soft-flatten worker, manual placement)
    # don't have to plumb them.
    target_half_spread_bps: Optional[float] = None
    aging_tighten_applied: bool = False


@dataclass(frozen=True, slots=True)
class QuoteBuildContext:
    decision: QuoteDecision
    market: Optional[BestBidAsk]
    risk_action: RiskAction
    bid_mult: float
    ask_mult: float
    spread_add_bps: float
    position_qty: float
    """Signed position quantity (contracts)."""
    position_notional: float
    """Abs position notional USD (for residual / dust decisions)."""
    resting_bid: Optional[WorkingOrder]
    resting_ask: Optional[WorkingOrder]
    reprice_replace_pending_bid: bool = False
    reprice_replace_pending_ask: bool = False
    # v1.4.70 wedge-elimination-cleanup Phase 1C: per-slot working
    # order snapshot. Keyed by ``(side, level_idx)``. Used by the
    # hard age-cap evaluation in ``_build_side`` so EVERY live slot
    # gets its cap checked, not just the inside rung.
    #
    # Caller (``OrderManager.maybe_refresh_quotes``) snapshots
    # ``state.all_working_orders()`` once per tick into this dict.
    # When empty, hard-age behaves as pre-Phase-1C (inside-rung only,
    # from ``resting_bid`` / ``resting_ask``) — preserving backward
    # compat for any path that doesn't populate it.
    working_orders_by_slot: dict[tuple[Side, int], WorkingOrder] = field(
        default_factory=dict
    )
    # Phase 4G.4 (v1.4.210) — regime-conditional effective inventory
    # budget. ``None`` (the default) means "use ``settings.max_abs_position``
    # as-is" — pre-4G backward-compat path.
    #
    # When the regime controller's CAUTIOUS or SHOCK mode is active,
    # the bot computes ``settings.max_abs_position × inventory_budget_mult``
    # (0.7 for CAUTIOUS, 0.5 for SHOCK) and passes the result here.
    # The two place-time inventory consumers
    # (``_clip_entry_sizes`` + ``_inventory_exec_bias_active``) read this
    # override; the existing inventory-cap cascade (inventory_skew,
    # inventory_exec_bias, hard_skew_long) all see the shrunken cap
    # automatically because util = pos_qty / effective_max_abs_position
    # rises proportionally.
    #
    # NOT consumed by the SF flatten clips or the hard risk-kill
    # check — those keep the raw ``settings.max_abs_position`` so the
    # bot can ALWAYS flatten regardless of regime knobs.
    effective_max_abs_position: Optional[float] = None


@dataclass(frozen=True, slots=True)
class QuoteBuildResult:
    bid_order: Optional[FinalQuoteOrder]
    ask_order: Optional[FinalQuoteOrder]
    mode: str
    residual_flatten_requested: bool = False
    telemetry: dict[str, Any] = field(default_factory=dict)
    # 1.3.59: hard-cancel signals from the strict age caps. When
    # non-empty for a side, OrderManager MUST cancel the resting
    # order on that side this cycle even if the new ``bid_order`` /
    # ``ask_order`` has the same price as the existing one. The
    # reason strings come from ``hard_reprice_reasons_buy/sell`` in
    # ``quote_aging.py`` — typically ``behind_touch_order_age_seconds``
    # or ``at_touch_order_age_seconds``. Empty tuple = no hard
    # cancel needed.
    #
    # v1.4.70 Phase 1C deprecation note: these fields are kept for
    # backward compatibility but cover ONLY the inside rung
    # (level_idx=0). New code reads ``hard_cancel_by_slot`` which
    # carries reasons for every aged slot — that's what the
    # multi-rung wall-lifetime cap needs.
    hard_cancel_bid_reasons: tuple[str, ...] = field(default_factory=tuple)
    hard_cancel_ask_reasons: tuple[str, ...] = field(default_factory=tuple)
    # v1.4.70 wedge-elimination-cleanup Phase 1C — per-slot hard-cancel
    # signals. Keyed by ``(side, level_idx)``. Empty dict = no slot
    # needs a hard cancel this cycle.
    #
    # The reconciler's ``compute_desired_state`` iterates this dict
    # and empties every aged slot, NOT just the inside rung. This
    # closes Codex F6/B3: with ``LADDER_NUM_LEVELS_PER_SIDE>=2``, an
    # outer-rung order could age forever pre-Phase-1C because the
    # cap fold only touched ``(side, 0)``.
    #
    # Reason tuples are the same shape as the inside-rung fields:
    # tags from ``hard_reprice_reasons_buy/sell`` (e.g.
    # ``"order_age_seconds"``, ``"at_touch_order_age_seconds"``,
    # ``"behind_touch_order_age_seconds"``, ``"distance_to_touch_ticks"``).
    hard_cancel_by_slot: dict[tuple["Side", int], tuple[str, ...]] = field(
        default_factory=dict
    )

    # v1.4.83 wedge-elimination-cleanup Phase 4A — typed BuildCommand
    # sum-type converter. The legacy ``QuoteBuildResult`` is a single
    # loose dataclass with optional fields whose meaning depends on
    # ``mode``. Phase 2D shipped a runtime validator
    # (``quote_build_result_unconsumed_field_total``) that flagged the
    # v1.4.66-style regression where a new field is added but no
    # consumer reads it. Phase 4A replaces that runtime audit with a
    # typed sum type: every outcome is its own dataclass with exactly
    # the fields it needs, and the dispatcher's ``match`` statement is
    # exhaustively type-checked at edit time.
    #
    # The cutover is incremental: 4A defines the sum-type and the
    # ``to_build_command()`` converter; 4B/4C decompose the engine
    # internals; the final removal of ``QuoteBuildResult`` and the
    # consumer-audit infrastructure is sequenced for after 4B/4C ship
    # so each sub-phase is independently revertable.
    def to_build_command(self) -> "BuildCommand":
        """Convert this legacy result to the typed sum-type variant.

        Mapping:
          * ``mode == "residual_flatten"`` → ``ResidualFlatten``
          * both orders None → ``NoQuote`` (mode == "no_quote")
          * both orders present → ``QuoteBoth``
          * exactly one order present → ``QuoteOneSided``

        Telemetry, hard_cancel_by_slot, and the reducing-side tuples
        are carried through unchanged. The legacy
        ``hard_cancel_{bid,ask}_reasons`` inside-rung tuples are
        embedded into the ``hard_cancel_by_slot`` map on the new
        variants (matches the v1.4.70 Phase 1C invariant that the
        multi-rung map is the canonical source).
        """
        cancels = dict(self.hard_cancel_by_slot)
        # Preserve the inside-rung tuples in the map for callers that
        # only look at hard_cancel_by_slot post-conversion.
        if self.hard_cancel_bid_reasons and (Side.BUY, 0) not in cancels:
            cancels[(Side.BUY, 0)] = tuple(self.hard_cancel_bid_reasons)
        if self.hard_cancel_ask_reasons and (Side.SELL, 0) not in cancels:
            cancels[(Side.SELL, 0)] = tuple(self.hard_cancel_ask_reasons)

        if self.mode == "residual_flatten" or self.residual_flatten_requested:
            return ResidualFlatten(
                target_qty=0.0,  # Engine doesn't specify a target qty today; flatten means "close all".
                # v1.5.275 (BUG-037 fix): forward the aging-cap signals
                # so an aged resting order whose tick became a residual-
                # flatten still gets cancelled. Pre-fix the cancels dict
                # was dropped here, leaving resting orders aging past
                # their caps.
                hard_cancel_by_slot=cancels,
                telemetry=dict(self.telemetry),
            )
        if self.bid_order is None and self.ask_order is None:
            return NoQuote(
                reason=str(self.telemetry.get("quote_engine_no_quote_reason", "no_quote")),
                # v1.5.275 (BUG-037 fix): forward the aging-cap signals
                # so an aged resting order whose tick became a NoQuote
                # still gets cancelled. Pre-fix the cancels dict was
                # dropped here, which is the documented root cause of
                # the 8.03 s outlier on snapshot
                # v1.5.271-260529-234638 (see issues/done/bug-037).
                hard_cancel_by_slot=cancels,
                telemetry=dict(self.telemetry),
            )
        if self.bid_order is not None and self.ask_order is not None:
            return QuoteBoth(
                bid=self.bid_order,
                ask=self.ask_order,
                hard_cancel_by_slot=cancels,
                telemetry=dict(self.telemetry),
            )
        # Exactly one side present.
        if self.bid_order is not None:
            return QuoteOneSided(
                side=Side.BUY,
                order=self.bid_order,
                suppressed_side_reason=str(
                    self.telemetry.get("quote_engine_suppressed_ask_reason", "ask_suppressed")
                ),
                hard_cancel_by_slot=cancels,
                telemetry=dict(self.telemetry),
            )
        assert self.ask_order is not None  # mypy / readability
        return QuoteOneSided(
            side=Side.SELL,
            order=self.ask_order,
            suppressed_side_reason=str(
                self.telemetry.get("quote_engine_suppressed_bid_reason", "bid_suppressed")
            ),
            hard_cancel_by_slot=cancels,
            telemetry=dict(self.telemetry),
        )


# v1.4.83 wedge-elimination-cleanup Phase 4A — typed sum-type for
# quote-engine outcomes. Each variant carries EXACTLY the fields its
# downstream consumer needs. The dispatcher in
# ``OrderManager.compute_desired_state`` uses ``match`` to exhaustively
# handle every variant; adding a new variant to the union causes
# mypy / pyright to flag every match site missing a case.
#
# Phase 2D's runtime consumer audit
# (``quote_build_result_unconsumed_field_total``) becomes redundant
# once the cutover lands — the type system enforces the property at
# edit time instead of runtime.
@dataclass(frozen=True, slots=True)
class QuoteBoth:
    """Both sides priced. Standard two-sided MM tick output.

    v1.4.92 Phase 4A cutover — exposes legacy ``QuoteBuildResult``-
    compatible properties (``bid_order`` / ``ask_order`` / ``mode`` /
    ``residual_flatten_requested`` / ``hard_cancel_bid_reasons`` /
    ``hard_cancel_ask_reasons``) so existing consumers and tests keep
    working without per-variant ``isinstance`` branching. New code
    should prefer typed access (``cmd.bid`` / ``cmd.ask``).
    """
    bid: FinalQuoteOrder
    ask: FinalQuoteOrder
    hard_cancel_by_slot: dict[tuple[Side, int], tuple[str, ...]] = field(
        default_factory=dict
    )
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def bid_order(self) -> Optional[FinalQuoteOrder]:
        return self.bid

    @property
    def ask_order(self) -> Optional[FinalQuoteOrder]:
        return self.ask

    @property
    def mode(self) -> str:
        return "two_sided"

    @property
    def residual_flatten_requested(self) -> bool:
        return False

    @property
    def hard_cancel_bid_reasons(self) -> tuple[str, ...]:
        return self.hard_cancel_by_slot.get((Side.BUY, 0), ())

    @property
    def hard_cancel_ask_reasons(self) -> tuple[str, ...]:
        return self.hard_cancel_by_slot.get((Side.SELL, 0), ())


@dataclass(frozen=True, slots=True)
class QuoteOneSided:
    """One side priced, the other suppressed (risk action, inventory bias, etc.).

    v1.4.92 Phase 4A cutover — legacy-compat properties; see QuoteBoth."""
    side: Side
    order: FinalQuoteOrder
    suppressed_side_reason: str
    hard_cancel_by_slot: dict[tuple[Side, int], tuple[str, ...]] = field(
        default_factory=dict
    )
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def bid_order(self) -> Optional[FinalQuoteOrder]:
        return self.order if self.side == Side.BUY else None

    @property
    def ask_order(self) -> Optional[FinalQuoteOrder]:
        return self.order if self.side == Side.SELL else None

    @property
    def mode(self) -> str:
        return "one_sided"

    @property
    def residual_flatten_requested(self) -> bool:
        return False

    @property
    def hard_cancel_bid_reasons(self) -> tuple[str, ...]:
        return self.hard_cancel_by_slot.get((Side.BUY, 0), ())

    @property
    def hard_cancel_ask_reasons(self) -> tuple[str, ...]:
        return self.hard_cancel_by_slot.get((Side.SELL, 0), ())


@dataclass(frozen=True, slots=True)
class NoQuote:
    """Neither side priced. Reason carries the upstream suppression
    cause (eligibility gate, BBO missing, decision-empty, etc.).

    v1.4.92 Phase 4A cutover — legacy-compat properties.

    v1.5.275 (BUG-037 fix): the aging-cap signal map is now a
    real field (was a property returning the empty dict). Pre-fix
    every NoQuote tick silently dropped the per-slot hard_cancel
    reasons computed by ``compute_aging_signals``, so a resting
    order whose tick was a NoQuote (one-sided regime, CAUTIOUS
    suppression, eligibility gate, etc.) would age past
    ``AT_TOUCH_MAX_AGE_SECONDS`` /
    ``BEHIND_TOUCH_MAX_AGE_SECONDS`` without ever being cancelled.
    Reproduced on snapshot v1.5.271-260529-234638: SELL @ 1.762
    on (SELL, 1) lived 8.03 s past its 3.0 s behind-touch cap
    because every tick during its life was a NoQuote. See
    issues/done/bug-037-rung1-behind-touch-aging-overrun.md for
    the full lifecycle trace."""
    reason: str
    hard_cancel_by_slot: dict[tuple[Side, int], tuple[str, ...]] = field(
        default_factory=dict
    )
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def bid_order(self) -> Optional[FinalQuoteOrder]:
        return None

    @property
    def ask_order(self) -> Optional[FinalQuoteOrder]:
        return None

    @property
    def mode(self) -> str:
        return "no_quote"

    @property
    def residual_flatten_requested(self) -> bool:
        return False

    @property
    def hard_cancel_bid_reasons(self) -> tuple[str, ...]:
        return self.hard_cancel_by_slot.get((Side.BUY, 0), ())

    @property
    def hard_cancel_ask_reasons(self) -> tuple[str, ...]:
        return self.hard_cancel_by_slot.get((Side.SELL, 0), ())


@dataclass(frozen=True, slots=True)
class ResidualFlatten:
    """Force-close path. The dispatcher routes to the residual-flatten
    worker which closes the position via the appropriate venue path
    (passive replacement or taker, per config). ``target_qty`` is
    reserved for a future variant where the engine specifies a partial
    close target; today the worker uses the full position abs qty.

    v1.4.92 Phase 4A cutover — legacy-compat properties.

    v1.5.275 (BUG-037 fix): same fix as ``NoQuote`` — aging-cap
    signals are now a real field carried through to the executor
    rather than silently dropped during a residual-flatten tick."""
    target_qty: float
    hard_cancel_by_slot: dict[tuple[Side, int], tuple[str, ...]] = field(
        default_factory=dict
    )
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def bid_order(self) -> Optional[FinalQuoteOrder]:
        return None

    @property
    def ask_order(self) -> Optional[FinalQuoteOrder]:
        return None

    @property
    def mode(self) -> str:
        return "residual_flatten"

    @property
    def residual_flatten_requested(self) -> bool:
        return True

    @property
    def hard_cancel_bid_reasons(self) -> tuple[str, ...]:
        return self.hard_cancel_by_slot.get((Side.BUY, 0), ())

    @property
    def hard_cancel_ask_reasons(self) -> tuple[str, ...]:
        return self.hard_cancel_by_slot.get((Side.SELL, 0), ())


# Sum-type alias. Use ``BuildCommand`` in type hints; the union
# enforces exhaustiveness on ``match`` statements that consume it.
BuildCommand = QuoteBoth | QuoteOneSided | NoQuote | ResidualFlatten


# v1.4.92 Phase 4A cutover — navigation helpers on the BuildCommand
# union. The consumer code in ``maybe_refresh_quotes`` /
# ``compute_desired_state`` iterates both sides uniformly with
# ``None`` meaning "no order on this side". The typed sum-type
# splits this into 4 disjoint variants where ``bid_order`` /
# ``ask_order`` are no longer uniform fields — so the consumer
# would have to ``isinstance``-branch at every access. These helpers
# encapsulate that dispatch into one place per legacy field, keeping
# the consumer changes minimal (`build.bid_order` → `get_bid_order(build)`).
#
# Type-safe alternative would be a `match` statement at every read
# site, but that would multiply 60-line functions into 240-line ones.
# The helpers are the right granularity for the legacy consumer's
# uniform-iteration pattern; the new ``match`` dispatch at the
# top-level routing (residual_flatten vs no_quote vs quote) is where
# the typed sum-type pays off.


def get_bid_order(cmd: BuildCommand) -> Optional[FinalQuoteOrder]:
    """Return the bid order from ``cmd``, or None when no bid is
    present (NoQuote / ResidualFlatten / QuoteOneSided of SELL)."""
    if isinstance(cmd, QuoteBoth):
        return cmd.bid
    if isinstance(cmd, QuoteOneSided) and cmd.side == Side.BUY:
        return cmd.order
    return None


def get_ask_order(cmd: BuildCommand) -> Optional[FinalQuoteOrder]:
    """Return the ask order from ``cmd``, or None when no ask is
    present (NoQuote / ResidualFlatten / QuoteOneSided of BUY)."""
    if isinstance(cmd, QuoteBoth):
        return cmd.ask
    if isinstance(cmd, QuoteOneSided) and cmd.side == Side.SELL:
        return cmd.order
    return None


def get_hard_cancel_by_slot(
    cmd: BuildCommand,
) -> dict[tuple[Side, int], tuple[str, ...]]:
    """Return the hard-cancel-by-slot map. NoQuote and ResidualFlatten
    have no orders → no cancel signals; return an empty dict."""
    if isinstance(cmd, (QuoteBoth, QuoteOneSided)):
        return cmd.hard_cancel_by_slot
    return {}


def is_residual_flatten(cmd: BuildCommand) -> bool:
    """True iff ``cmd`` is the ``ResidualFlatten`` variant — the engine
    is requesting an execution-only flatten action (not a quoting cycle)."""
    return isinstance(cmd, ResidualFlatten)


@dataclass(frozen=True, slots=True)
class AgingSettings:
    """v1.4.85 Phase 4C — typed settings subset for the aging stage.

    The 5 operator-knobs that ``compute_aging_signals`` (and its
    transitive helpers ``hard_reprice_reasons_buy/sell``) actually
    read. Defining this as a frozen subset lets unit tests construct
    a minimal fixture instead of a full ``Settings`` object, and
    documents the stage's settings-API contract.

    Compatibility: the full ``Settings`` object exposes the same field
    names, so duck-typing lets the same function accept either the
    full object or this subset. ``from_full(settings)`` is the
    canonical factory for the partition.

    Why this matters: Phase 2D's runtime consumer audit shipped to
    catch the v1.4.66-style regression where a field is added but no
    consumer reads it. The Phase 4 type-system equivalent is the
    typed subset — declaring exactly what the stage consumes from
    Settings makes it impossible to silently grow the dependency.
    """

    quote_aging_enabled: bool
    quote_aging_max_age_seconds: float
    quote_max_distance_to_touch_ticks: float
    at_touch_max_age_seconds: float
    behind_touch_max_age_seconds: float

    @classmethod
    def from_full(cls, settings: Settings) -> "AgingSettings":
        """Partition the full Settings into the aging-stage subset.

        Reads only the fields the stage needs; everything else on
        Settings is intentionally NOT mirrored here (that's the point
        of the subset). When a new aging-relevant knob is added to
        Settings, it must be added here AND to the transitive helpers
        — the subset becomes the single point that catches missed
        consumers at edit time.
        """
        return cls(
            quote_aging_enabled=bool(settings.quote_aging_enabled),
            quote_aging_max_age_seconds=float(settings.quote_aging_max_age_seconds),
            quote_max_distance_to_touch_ticks=float(
                settings.quote_max_distance_to_touch_ticks
            ),
            at_touch_max_age_seconds=float(
                getattr(settings, "at_touch_max_age_seconds", 0.0)
            ),
            behind_touch_max_age_seconds=float(
                getattr(settings, "behind_touch_max_age_seconds", 0.0)
            ),
        )


def compute_aging_signals(
    settings: "Settings | AgingSettings",
    working_orders_by_slot: dict[tuple[Side, int], WorkingOrder],
    resting_bid: Optional[WorkingOrder],
    resting_ask: Optional[WorkingOrder],
    *,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
    now,
) -> dict[tuple[Side, int], tuple[str, ...]]:
    """v1.4.84 Phase 4B — pure function: compute hard-cancel reasons
    per (side, level_idx) slot.

    v1.4.85 Phase 4C: accepts either the full ``Settings`` object or
    the ``AgingSettings`` subset (Python duck typing). Tests should
    prefer the subset for minimal fixtures; production calls pass
    the full object for backward compatibility.

    Pre-Phase-4B this was an inline block at the bottom of
    ``QuoteEngine.build_quotes``. Extracted as a module-level pure
    function so:

    1. The stage is unit-testable without instantiating QuoteEngine.
    2. Phase 5B's per-slot diff struct can call it directly.
    3. The plan's Phase 1C invariant (every aged slot evaluated, NOT
       just inside-rung) is enforced in one place.

    Behaviour preserved exactly from the inline block:

    * When ``working_orders_by_slot`` is non-empty, evaluate the cap
      for EVERY (side, lvl) present. Outer rungs (level_idx >= 1) are
      invisible pre-Phase-1C; this path makes them visible.
    * When ``working_orders_by_slot`` is empty, fall back to
      ``resting_bid`` / ``resting_ask`` at level_idx=0. Preserves
      backward compat for callers that don't populate the per-slot
      snapshot.
    * The cap thresholds come from ``settings`` (BEHIND_TOUCH_*,
      AT_TOUCH_*). Both default to 0.0 = disabled; this function is
      a no-op unless the operator has opted in.

    Returns a dict mapping ``(side, level_idx) → tuple[reason, ...]``.
    Empty for slots whose order has not breached the cap.
    """
    out: dict[tuple[Side, int], tuple[str, ...]] = {}
    if working_orders_by_slot:
        for (side_key, lvl), wo in working_orders_by_slot.items():
            if wo is None:
                continue
            if side_key == Side.BUY:
                reasons = hard_reprice_reasons_buy(
                    settings,
                    wo,
                    best_bid=best_bid,
                    tick=tick,
                    now=now,
                    enforce_touch_distance_band=False,
                )
            else:
                reasons = hard_reprice_reasons_sell(
                    settings,
                    wo,
                    best_ask=best_ask,
                    tick=tick,
                    now=now,
                    enforce_touch_distance_band=False,
                )
            if reasons:
                out[(side_key, int(lvl))] = reasons
        return out
    # Backward-compat fallback: inside-rung only.
    if resting_bid is not None:
        reasons = hard_reprice_reasons_buy(
            settings,
            resting_bid,
            best_bid=best_bid,
            tick=tick,
            now=now,
            enforce_touch_distance_band=False,
        )
        if reasons:
            out[(Side.BUY, 0)] = reasons
    if resting_ask is not None:
        reasons = hard_reprice_reasons_sell(
            settings,
            resting_ask,
            best_ask=best_ask,
            tick=tick,
            now=now,
            enforce_touch_distance_band=False,
        )
        if reasons:
            out[(Side.SELL, 0)] = reasons
    return out


def _resting_clip_size_for_position_headroom(wo: Optional[WorkingOrder]) -> float:
    if wo is None:
        return 0.0
    if wo.status.value not in ("SENT", "CANCEL_PENDING"):
        return 0.0
    return max(0.0, float(wo.size))


def _violates_min_half_spread_floor(
    side: Side,
    px: float,
    *,
    mid_ref: Optional[float],
    min_half_spread_px: Optional[float],
) -> bool:
    """v1.4.34 (Codex #2 fix) — helper used by ``QuoteEngine._build_side``.

    Returns True when ``px`` sits inside the operator-configured
    economic spread floor:

    * BUY:  ``px > mid_ref - min_half_spread_px``  (too aggressive)
    * SELL: ``px < mid_ref + min_half_spread_px``  (too aggressive)

    Returns False when the floor isn't configured (``mid_ref`` or
    ``min_half_spread_px`` missing / zero / negative) — the floor
    only exists when both are present and positive.

    Module-level so the regression test can exercise it directly;
    the cross-retreat path inside ``_build_side`` invokes it post-
    retreat to refuse quotes that retreated to passive-but-still-
    too-aggressive prices.
    """
    if (
        min_half_spread_px is None
        or float(min_half_spread_px) <= 0
        or mid_ref is None
        or not math.isfinite(float(mid_ref))
        or float(mid_ref) <= 0
    ):
        return False
    m = float(mid_ref)
    mhs = float(min_half_spread_px)
    if side == Side.BUY:
        # BUY floor: must be at or below ``mid - min_half_spread``.
        # Strictly greater than that bound is a violation.
        return px > m - mhs + 1e-12
    # SELL floor: must be at or above ``mid + min_half_spread``.
    return px < m + mhs - 1e-12


class QuoteEngine:
    """Single quote-shaping layer: produces final passive orders for the runtime path.

    ``build_quotes`` returns prices/sizes that are already mode-final, grid-rounded,
    and economically valid for submission. Execution must not normalize, re-round, or
    reinterpret these orders—only diff/cancel/replace/place them via
    ``OrderManager._submit_passive_order_verbatim``.
    """

    def __init__(
        self,
        settings: Settings,
        spec: SymbolSpec,
        *,
        markout_adverse_tracker: Optional[MarkoutAdverseTracker] = None,
    ) -> None:
        """v1.4.93 wedge-elimination-cleanup Phase 4B.4 — the
        ``MarkoutAdverseTracker`` is now an injected dependency.
        Default behavior unchanged: when ``markout_adverse_tracker`` is
        None (the production call path), the engine constructs its own
        fresh instance, matching the pre-Phase-4B.4 behavior bit-for-bit.

        Tests can inject a pre-configured tracker (e.g., a
        ``MagicMock``, or a tracker with seeded per-side timer state)
        to validate the markout-adverse-cancel flow without going
        through the natural breach-then-elapse cycle. This is
        primarily a TESTABILITY improvement, not a behavioral change.
        """
        self._settings = settings
        self._spec = spec
        # Per-side timer for the markout-aging cancel signal. Kept on
        # the engine instance so the timer survives across cycles but
        # not across symbol switches (each symbol gets its own engine).
        self._markout_adverse_tracker = (
            markout_adverse_tracker
            if markout_adverse_tracker is not None
            else MarkoutAdverseTracker()
        )

    def _clip_entry_sizes(
        self,
        position_qty: float,
        bid_sz: float,
        ask_sz: float,
        *,
        resting_bid_sz: float,
        resting_ask_sz: float,
        bid_price: float = 0.0,
        ask_price: float = 0.0,
        effective_max_abs_position: Optional[float] = None,
    ) -> tuple[float, float, float, float]:
        """Clip new bid/ask sizes so an immediate full fill on either
        side cannot push position past either of the two configured caps:

        * ``MAX_ABS_POSITION`` — hard cap in BASE units (e.g. SUI)
        * ``MAX_POSITION_NOTIONAL_USD`` — hard cap in USD notional

        Both caps are applied; the tighter one wins. Pre-2026-05-06,
        only the base-units cap was applied, which leaves the USD
        cap unenforced when price moves significantly (at $5/SUI a
        25-SUI base cap = $125, way over the $20 USD cap).

        Caller passes ``bid_price`` / ``ask_price`` for the USD-cap
        translation. When 0 (e.g. tests that don't supply prices),
        the USD cap is skipped and only the base-unit cap applies.

        Returns ``(clipped_bid_sz, clipped_ask_sz, max_buy, max_sell)``.
        The last two are the position-cap-derived ceilings *before*
        intersecting with the caller's intended size — used by
        ``_build_side`` to know how far self-heal can safely bump a
        rounded-down candidate. Bug fix 2026-05-08 (snapshot
        260507123509): previously we returned only the post-clip
        sizes and the caller passed those as ``max_allowed_size``,
        which over-tightened self-heal whenever the candidate had
        been clipped by ``max_order_notional_usd`` rather than the
        position cap.
        """
        # Phase 4G.4: ``effective_max_abs_position`` (when not None)
        # overrides the raw setting. Used by the regime-controller-
        # driven CAUTIOUS / SHOCK modes to shrink the effective
        # inventory budget. Falls back to the raw setting in pre-4G
        # call paths.
        if (
            effective_max_abs_position is not None
            and effective_max_abs_position > 0
        ):
            mx = float(effective_max_abs_position)
        else:
            mx = float(self._settings.max_abs_position)
        if mx <= 0:
            return 0.0, 0.0, 0.0, 0.0
        max_buy = max(0.0, mx - position_qty - resting_bid_sz - 1e-9)
        max_sell = max(0.0, position_qty + mx - resting_ask_sz - 1e-9)

        # USD-notional cap. Only apply when prices are supplied AND
        # the cap is configured. Convert: max_position_notional_usd /
        # current_price -> base units allowed, then subtract resting
        # exposure on that side so the post-fill notional stays under
        # the cap.
        max_pos_usd = float(self._settings.max_position_notional_usd or 0.0)
        if max_pos_usd > 0:
            if bid_price > 0:
                allowed_long_base = max_pos_usd / bid_price
                max_buy_usd = max(
                    0.0,
                    allowed_long_base - position_qty - resting_bid_sz - 1e-9,
                )
                max_buy = min(max_buy, max_buy_usd)
            if ask_price > 0:
                allowed_short_base = max_pos_usd / ask_price
                max_sell_usd = max(
                    0.0,
                    position_qty + allowed_short_base - resting_ask_sz - 1e-9,
                )
                max_sell = min(max_sell, max_sell_usd)
        return (
            min(bid_sz, max_buy),
            min(ask_sz, max_sell),
            max_buy,
            max_sell,
        )

    def _inventory_exec_bias_active(
        self,
        pos_qty: float,
        *,
        effective_max_abs_position: Optional[float] = None,
        aqc_aggression_level: Optional[float] = None,
    ) -> bool:
        ratio = float(self._settings.inventory_execution_bias_ratio)
        if ratio <= 0:
            return False
        # Phase 4G.4: regime-overridden effective cap (CAUTIOUS / SHOCK).
        if (
            effective_max_abs_position is not None
            and effective_max_abs_position > 0
        ):
            mx = max(float(effective_max_abs_position), 1e-12)
        else:
            mx = max(float(self._settings.max_abs_position), 1e-12)
        util = abs(float(pos_qty)) / mx
        # v1.5.282 AQC Phase 3 — the util floor is modulated tighten-only
        # by the controller's aggression_level (no-op unless
        # AQC_WIRE_INVENTORY is on and the level is finite). The
        # max(ratio, floor) clamp below means the engagement util can
        # never drop below ``inventory_execution_bias_ratio``.
        eff_floor = compute_effective_inventory_util_floor_pct(
            self._settings,
            aqc_aggression_level=aqc_aggression_level,
        )
        gate = max(ratio, eff_floor)
        return util >= gate - 1e-15

    def _resting_side_maintained(self, side: Side, ctx: QuoteBuildContext) -> bool:
        if side == Side.BUY:
            if ctx.reprice_replace_pending_bid:
                return True
            wo = ctx.resting_bid
        else:
            if ctx.reprice_replace_pending_ask:
                return True
            wo = ctx.resting_ask
        if wo is None:
            return False
        return wo.status in (
            OrderStatus.SENT,
            OrderStatus.ACKED,
            OrderStatus.PARTIAL,
            OrderStatus.CANCEL_PENDING,
        )

    def _apply_inventory_execution_bias(
        self,
        ctx: QuoteBuildContext,
        bid_order: Optional[FinalQuoteOrder],
        ask_order: Optional[FinalQuoteOrder],
    ) -> tuple[Optional[FinalQuoteOrder], Optional[FinalQuoteOrder]]:
        """Suppress inventory-adding side until the reducing side is maintained on the book."""
        pq = float(ctx.position_qty)
        # Phase 4G.4: pass regime-overridden effective cap so CAUTIOUS /
        # SHOCK shrink the gate's util denominator → adding-side
        # suppression engages at a lower raw position than NORMAL.
        if not self._inventory_exec_bias_active(
            pq,
            effective_max_abs_position=ctx.effective_max_abs_position,
            aqc_aggression_level=getattr(
                ctx.decision, "aqc_aggression_level", None
            ),
        ):
            return bid_order, ask_order
        if pq > _QE_POS_EPS:
            if not self._resting_side_maintained(Side.SELL, ctx):
                return None, ask_order
        elif pq < -_QE_POS_EPS:
            if not self._resting_side_maintained(Side.BUY, ctx):
                return bid_order, None
        return bid_order, ask_order

    def _build_side(
        self,
        *,
        side: Side,
        want_side: bool,
        candidate_px: float,
        candidate_sz: float,
        best_bid: Optional[float],
        best_ask: Optional[float],
        tick: float,
        min_half_spread_px: Optional[float],
        mid_ref: Optional[float],
        max_allowed_size: Optional[float] = None,
    ) -> tuple[Optional[FinalQuoteOrder], str, Optional[float], Optional[float]]:
        """Single-model quote construction: rounding, min_notional, max_order_notional, all at once.

        **One-model invariant** (see module docstring): once this function is
        called with ``want_side=True`` and finite strategy intent, it must
        *either* return a placement-ready :class:`FinalQuoteOrder` *or* a
        rejection reason that reflects a real strategy/venue decision (no BBO,
        intent conflicts with risk caps, etc.). It must **never** reject for a
        reason a caller could paper over — in particular, below-min-notional
        from size_step rounding is *self-healed* here, not re-raised to the
        caller. Any new constraint added to the venue pipeline MUST be
        repairable inside this function; if it can only be diagnosed as an
        error, that is layered-disagreement and violates the contract.

        ``max_allowed_size``: upper bound on the final output size after any
        self-healing. Typically the caller has already clipped ``candidate_sz``
        against the position-cap; passing that same clipped value here
        prevents the min-notional self-heal from bumping the order back over
        the cap. When None, no per-side cap is enforced beyond the standard
        ``max_order_notional_usd`` risk cap. See ``tmp/snap_20260418_140635``
        for the observed bug: pos ran from −0.045 to −0.060 (exceeded
        ``MAX_ABS_POSITION=0.05`` by 20%) because self-heal bumped a
        position-clipped 0.005 size back up to 0.015 to meet min notional.
        """
        if not want_side:
            return None, "side_not_requested", None, None
        if candidate_px <= 0 or candidate_sz <= 0:
            return None, "non_positive_candidate", None, None

        px = float(candidate_px)
        if side == Side.BUY and best_bid is not None and math.isfinite(best_bid):
            px = min(px, float(best_bid))
        if side == Side.SELL and best_ask is not None and math.isfinite(best_ask):
            px = max(px, float(best_ask))

        if (
            min_half_spread_px is not None
            and min_half_spread_px > 0
            and mid_ref is not None
            and math.isfinite(float(mid_ref))
            and float(mid_ref) > 0
        ):
            m = float(mid_ref)
            if side == Side.BUY:
                px = min(px, m - float(min_half_spread_px))
                if best_bid is not None and math.isfinite(best_bid):
                    px = min(px, float(best_bid))
            else:
                px = max(px, m + float(min_half_spread_px))
                if best_ask is not None and math.isfinite(best_ask):
                    px = max(px, float(best_ask))

        rounded, rej = round_price_and_size_to_grid(self._spec, px, float(candidate_sz))
        if rounded is None:
            return None, f"normalize_rejected:{rej}", None, None
        npx, nsz = float(rounded[0]), float(rounded[1])
        if npx <= 0 or nsz <= 0:
            return None, "normalize_non_positive", None, None

        # Re-check economics after rounding; if tick rounding got too aggressive, retreat by one tick.
        if (
            min_half_spread_px is not None
            and min_half_spread_px > 0
            and mid_ref is not None
            and float(mid_ref) > 0
            and tick > 0
        ):
            m = float(mid_ref)
            if side == Side.BUY and npx > m - float(min_half_spread_px) + 1e-12:
                npx -= float(tick)
            elif side == Side.SELL and npx < m + float(min_half_spread_px) - 1e-12:
                npx += float(tick)
            rounded2, _rej2 = round_price_and_size_to_grid(self._spec, npx, nsz)
            if rounded2 is not None:
                npx, nsz = float(rounded2[0]), float(rounded2[1])

        # --- Unified notional handling (no layered disagreement) ---
        # Both floors below are things this function can SELF-HEAL by bumping
        # size to the next step that clears the floor, bounded by the risk cap
        # ``max_order_notional_usd``. The only legit rejects at this stage:
        #   (a) caps are contradictory — the venue minimum exceeds our per-order cap
        #   (b) intent was too small and the clamp would more than quadruple it
        #       (guard against config drift producing wildly-resized orders).
        #   (c) the position-cap-derived ``max_allowed_size`` is smaller than the
        #       minimum step-multiple that clears the venue floor — self-heal
        #       would exceed ``max_abs_position``. Correct answer: skip this
        #       placement, wait for position to drift back.
        venue_min_usd = float(self._spec.min_notional_usd)
        local_min_usd = float(self._settings.min_quote_notional_usd)
        required_min_usd = max(venue_min_usd, local_min_usd)
        max_order_usd = float(self._settings.max_order_notional_usd)
        step_dec = self._spec.size_step
        step = float(step_dec) if step_dec and float(step_dec) > 0 else 0.0
        ntn = npx * nsz
        if ntn + 1e-9 < required_min_usd and step > 0 and npx > 0:
            # Ceil to the smallest step-multiple that clears the required minimum.
            needed_sz = math.ceil((required_min_usd + 1e-9) / npx / step) * step
            # Quadruple-intent guard: if clearing the venue minimum requires
            # more than 4× the intent size, something is fundamentally
            # misconfigured (``QUOTE_NOTIONAL_USD`` too small for the venue).
            # Better to reject loudly than to silently quadruple an order.
            if needed_sz > max(4.0 * float(candidate_sz), step * 1.01):
                return None, (
                    f"below_min_notional_intent_too_small "
                    f"needed_sz={needed_sz} intent_sz={candidate_sz} "
                    f"required_min_usd={required_min_usd}"
                ), npx, nsz
            # Cap at max_order_notional_usd. If the minimum viable size already
            # exceeds the per-order cap, venue and risk configs conflict.
            if needed_sz * npx > max_order_usd + 1e-9:
                return None, (
                    f"min_notional_exceeds_max_order_notional "
                    f"needed_usd={needed_sz * npx:.4f} max_usd={max_order_usd}"
                ), npx, nsz
            # Position-cap guard. The caller passes ``max_allowed_size`` as the
            # position-cap-clipped value. Self-healing above this would silently
            # exceed ``max_abs_position``, which is the bug observed in
            # ``tmp/snap_20260418_140635`` (pos ran to −0.060 vs −0.050 cap).
            # A small ``+ step × 1e-9`` tolerance guards against fp noise.
            if (
                max_allowed_size is not None
                and max_allowed_size >= 0
                and needed_sz > float(max_allowed_size) + step * 1e-9
            ):
                return None, (
                    f"position_cap_forbids_min_notional "
                    f"needed_sz={needed_sz} max_allowed_size={max_allowed_size} "
                    f"required_min_usd={required_min_usd}"
                ), npx, nsz
            # Re-normalize to be safe (size_step alignment). Should be a no-op
            # since we ceil'd on the step, but defensive against fp drift.
            rounded3, rej3 = round_price_and_size_to_grid(self._spec, npx, needed_sz)
            if rounded3 is None:
                return None, f"normalize_rejected:{rej3}", npx, nsz
            npx, nsz = float(rounded3[0]), float(rounded3[1])
            ntn = npx * nsz
            if ntn + 1e-9 < required_min_usd:
                # Shouldn't happen after ceil; defensive.
                return None, "self_heal_still_below_min_notional", npx, nsz

        # Enforce the risk cap regardless of self-heal path.
        if ntn - 1e-9 > max_order_usd:
            return None, (
                f"above_max_order_notional notional_usd={ntn:.4f} "
                f"max_usd={max_order_usd}"
            ), npx, nsz

        # Enforce the position cap as a final defensive gate. This catches
        # the case where ``candidate_sz`` was above ``max_allowed_size`` on
        # entry (caller passed a larger candidate than the cap allows — e.g.
        # if the caller didn't clip and relies on us to respect the cap).
        # Together with the self-heal gate above, this is the single place
        # where the position cap is enforced in the build pipeline.
        if (
            max_allowed_size is not None
            and max_allowed_size >= 0
            and nsz > float(max_allowed_size) + (step * 1e-9 if step > 0 else 1e-12)
        ):
            return None, (
                f"above_position_cap_allowed_size "
                f"nsz={nsz} max_allowed_size={max_allowed_size}"
            ), npx, nsz

        # v1.4.26 crossing-place hot-loop fix (Fix A+B): final defensive
        # cross-check before emit. If the rounded + retreated price
        # would CROSS THE SPREAD at the venue (BUY >= best_ask or
        # SELL <= best_bid), retreat by one tick to keep it passive.
        # Diagnosed in snapshot v1.4.25-260517-202432: the engine's
        # earlier retreat-by-tick (line 340-355 above) checks against
        # ``mid - min_half_spread_px``; when ``normal_mm`` produces a
        # tiny min_half_spread (at-touch market spread = ~2.6 bps half),
        # the condition lets BUY sit at best_bid or even cross to
        # best_ask. Inventory skew + aging-tighten can also push the
        # price across in extreme regimes.
        #
        # OKX post-only orders at a crossing price get silently cancelled
        # by the matching engine within ms via user-data-WS — not via
        # a 51604 sCode in the place response — producing a hot
        # place-cancel-place-cancel loop at the WS wake rate (~100/sec
        # observed). The user-data WS cancel detector (execution.py
        # v1.4.26 Fix C) ALSO catches this pattern and arms a cooldown,
        # but preventing it at the engine layer eliminates the wasted
        # round-trip entirely.
        #
        # If retreating by one tick produces a non-positive price or
        # would drop below ``min_half_spread_px``, refuse the quote
        # rather than emitting a wider-than-intended price that the
        # caller didn't sanction.
        #
        # v1.4.34 (Codex #2 fix): the previous code only re-checked
        # "still crosses" after retreat, NOT "still respects
        # min_half_spread_px". A candidate retreated to one tick
        # below ``best_ask`` could still be tighter than the
        # operator's economic spread floor — the bot would emit a
        # passive-but-too-aggressive quote that violates the model's
        # spread discipline. Mirror the same floor check the
        # post-rounding retreat above uses, refusing the quote when
        # the retreated price still sits inside the floor.
        if tick > 0:
            if side == Side.BUY and best_ask is not None and math.isfinite(best_ask):
                if npx >= float(best_ask) - 1e-12:
                    retreated = npx - tick
                    if retreated > 0:
                        rounded_r, _rej_r = round_price_and_size_to_grid(
                            self._spec, retreated, nsz
                        )
                        if rounded_r is not None:
                            npx = float(rounded_r[0])
                            nsz = float(rounded_r[1])
                            # Re-check post-retreat — if still crossing
                            # (extreme: best_ask == best_bid + 1 tick →
                            # no passive room), refuse.
                            if npx >= float(best_ask) - 1e-12:
                                return None, (
                                    f"would_cross_best_ask_no_passive_room "
                                    f"npx={npx} best_ask={best_ask}"
                                ), npx, nsz
                            # v1.4.34 (Codex #2): the retreated price
                            # is now passive but may still violate the
                            # operator-configured min-half-spread
                            # economic floor.
                            if _violates_min_half_spread_floor(Side.BUY, npx, mid_ref=mid_ref, min_half_spread_px=min_half_spread_px):
                                return None, (
                                    f"cross_retreat_violates_min_half_spread "
                                    f"npx={npx} mid_ref={mid_ref} "
                                    f"min_half_spread_px={min_half_spread_px}"
                                ), npx, nsz
                        else:
                            return None, (
                                f"would_cross_best_ask_retreat_grid_rejected "
                                f"npx={npx} best_ask={best_ask}"
                            ), npx, nsz
                    else:
                        return None, (
                            f"would_cross_best_ask_retreat_non_positive "
                            f"npx={npx} best_ask={best_ask}"
                        ), npx, nsz
            elif side == Side.SELL and best_bid is not None and math.isfinite(best_bid):
                if npx <= float(best_bid) + 1e-12:
                    advanced = npx + tick
                    rounded_a, _rej_a = round_price_and_size_to_grid(
                        self._spec, advanced, nsz
                    )
                    if rounded_a is not None:
                        npx = float(rounded_a[0])
                        nsz = float(rounded_a[1])
                        if npx <= float(best_bid) + 1e-12:
                            return None, (
                                f"would_cross_best_bid_no_passive_room "
                                f"npx={npx} best_bid={best_bid}"
                            ), npx, nsz
                        # v1.4.34 (Codex #2): same floor check on
                        # the SELL side after advance.
                        if _violates_min_half_spread_floor(Side.SELL, npx, mid_ref=mid_ref, min_half_spread_px=min_half_spread_px):
                            return None, (
                                f"cross_advance_violates_min_half_spread "
                                f"npx={npx} mid_ref={mid_ref} "
                                f"min_half_spread_px={min_half_spread_px}"
                            ), npx, nsz
                    else:
                        return None, (
                            f"would_cross_best_bid_advance_grid_rejected "
                            f"npx={npx} best_bid={best_bid}"
                        ), npx, nsz

        if __debug__:
            assert math.isfinite(npx) and math.isfinite(nsz) and npx > 0 and nsz > 0
        return FinalQuoteOrder(side=side, price=npx, size=nsz), "ok", npx, nsz

    def build_quotes(self, ctx: QuoteBuildContext) -> BuildCommand:
        """Return the final desired bid/ask orders for this cycle, as a
        typed ``BuildCommand`` sum-type variant.

        v1.4.92 Phase 4A cutover: the return type is now ``BuildCommand``
        (one of ``QuoteBoth`` / ``QuoteOneSided`` / ``NoQuote`` /
        ``ResidualFlatten``). The legacy ``QuoteBuildResult`` dataclass
        is built internally as the staging type during the pricing
        pipeline (because the existing rounding / clamping / self-heal
        logic is bit-for-bit preserved) and then converted to
        ``BuildCommand`` at return via ``to_build_command()``. Consumers
        navigate the result via the module-level helpers
        ``get_bid_order(cmd)`` / ``get_ask_order(cmd)`` /
        ``get_hard_cancel_by_slot(cmd)`` / ``is_residual_flatten(cmd)``
        OR via exhaustive ``match`` for top-level routing.

        This is the only quoting layer: side activation, risk/degraded modes, rounding,
        min-notional / min-quote-notional handling, inventory bias, and residual-flatten
        *decisions* are resolved here. Callers pass the result to execution unchanged.

        Execution (`maybe_refresh_quotes`) must not repair or reshape these outputs.
        """
        t0 = time.perf_counter()
        decision = ctx.decision
        pn = float(ctx.position_notional)
        pq0 = float(ctx.position_qty)
        mn = float(self._spec.min_notional_usd)
        dust_th = float(self._settings.dust_position_notional_usd)
        force_th = float(self._settings.force_flatten_notional_usd)
        has_pos = abs(pq0) > _QE_POS_EPS and pn > _QE_POS_EPS
        is_dust = has_pos and pn + 1e-12 < dust_th
        stuck_sub_min = has_pos and pn + 1e-12 < mn
        # Path A: stuck mid-zone residual — too small to quote passively
        # (sub-spec) AND big enough to be worth flattening (above the
        # operator-set ``force_flatten_notional_usd`` ceiling). The
        # ``not is_dust`` clause here means "above operator's let-it-ride
        # dust threshold". Pre-existing behaviour (regression-pinned by
        # ``test_force_flatten_when_stuck_above_force_threshold``).
        if stuck_sub_min and not is_dust and pn + 1e-12 >= force_th:
            return QuoteBuildResult(
                bid_order=None,
                ask_order=None,
                mode="residual_flatten",
                residual_flatten_requested=True,
                telemetry={
                    "quote_engine_mode": "residual_flatten",
                    "residual_flatten_requested": True,
                    "residual_flatten_reason": "above_force_th_sub_spec",
                    "quote_contract_build_ms": (time.perf_counter() - t0) * 1000.0,
                },
            ).to_build_command()
        # Path B (added 2026-05-08, snapshot 260507114312; default-OFF
        # in 1.1.35 once the upstream wedge cause was identified):
        # operator-set dust threshold is AT OR BELOW the venue spec
        # floor. In this configuration ALL dust positions are by
        # definition sub-spec.
        #
        # 1.1.33 reasoning: the bot couldn't quote sub-spec residuals
        # out via normal MM (any order to close them would round below
        # spec), so we fall through to forced market_close as a safety
        # escape.
        #
        # 1.1.35 update: the actual wedge cause was the
        # ``max_allowed_size`` over-tightening bug in
        # ``_build_side`` self-heal (snapshot 260507123509). With that
        # fixed, the bot's normal QUOTE_NOTIONAL order overshoots the
        # residual and clears it at maker rebate. Path B's taker
        # market_close costs ~5 bp per residual versus +1 bp rebate
        # via passive — actively destructive in the typical case.
        # Default OFF; the operator can re-enable per-profile if the
        # upstream fix proves insufficient or as belt-and-suspenders.
        # Path A above (force-flatten when ``pn >= force_th``) is
        # unchanged and always active.
        if (
            bool(getattr(self._settings, "residual_flatten_dust_below_spec_enabled", False))
            and has_pos
            and stuck_sub_min
            and is_dust
            and dust_th + 1e-12 <= mn
        ):
            return QuoteBuildResult(
                bid_order=None,
                ask_order=None,
                mode="residual_flatten",
                residual_flatten_requested=True,
                telemetry={
                    "quote_engine_mode": "residual_flatten",
                    "residual_flatten_requested": True,
                    "residual_flatten_reason": "sub_spec_dust_unrideable_config",
                    "dust_position_notional_usd": float(dust_th),
                    "spec_min_notional_usd": float(mn),
                    "position_notional_usd": float(pn),
                    "quote_contract_build_ms": (time.perf_counter() - t0) * 1000.0,
                },
            ).to_build_command()

        mkt = ctx.market
        best_bid = mkt.best_bid if mkt else None
        best_ask = mkt.best_ask if mkt else None
        tick = float(self._spec.price_tick) if self._spec.price_tick and self._spec.price_tick > 0 else 0.0

        act = decision.active_sides
        want_bid = act in (ActiveSides.BOTH, ActiveSides.BID_ONLY) and ctx.risk_action in (
            RiskAction.ALLOW,
            RiskAction.BID_ONLY,
        )
        want_ask = act in (ActiveSides.BOTH, ActiveSides.ASK_ONLY) and ctx.risk_action in (
            RiskAction.ALLOW,
            RiskAction.ASK_ONLY,
        )
        if ctx.risk_action == RiskAction.BID_ONLY:
            want_ask = False
        if ctx.risk_action == RiskAction.ASK_ONLY:
            want_bid = False

        bid_px = decision.quoted_bid * (1.0 - float(ctx.spread_add_bps) / 10_000.0)
        ask_px = decision.quoted_ask * (1.0 + float(ctx.spread_add_bps) / 10_000.0)
        bid_sz = decision.quoted_bid_sz * float(ctx.bid_mult)
        ask_sz = decision.quoted_ask_sz * float(ctx.ask_mult)
        bid_sz = clip(bid_sz, 0.0, float(self._settings.max_order_notional_usd) / max(float(bid_px), 1e-12))
        ask_sz = clip(ask_sz, 0.0, float(self._settings.max_order_notional_usd) / max(float(ask_px), 1e-12))
        bid_sz, ask_sz, max_buy_position_cap, max_sell_position_cap = (
            self._clip_entry_sizes(
                float(ctx.position_qty),
                float(bid_sz),
                float(ask_sz),
                resting_bid_sz=_resting_clip_size_for_position_headroom(ctx.resting_bid),
                resting_ask_sz=_resting_clip_size_for_position_headroom(ctx.resting_ask),
                bid_price=float(bid_px),
                ask_price=float(ask_px),
                # Phase 4G.4: regime-overridden effective cap. None →
                # use ``settings.max_abs_position`` (pre-4G behaviour).
                effective_max_abs_position=ctx.effective_max_abs_position,
            )
        )

        book_fresh = (
            mkt is not None
            and mkt.ts_local is not None
            and (utc_now() - mkt.ts_local).total_seconds() < float(self._settings.stale_data_warn_seconds)
        )
        two_sided = bool(want_bid and want_ask)
        normal_mode = bool(
            two_sided
            and ctx.risk_action == RiskAction.ALLOW
            and self._settings.normal_mm_use_market_spread_anchor
            and book_fresh
        )
        min_half_spread_px: Optional[float] = None
        mid_ref = float(mkt.mid_price) if mkt and mkt.mid_price is not None else float(decision.mid_price)
        normal_diag: dict[str, Any] = {}
        if normal_mode and mid_ref > 0:
            eff_h = compute_effective_min_half_spread_bps(
                self._settings,
                decision.active_sides,
                decision.toxicity_score,
                spread_floor_overlay_half_spread_bps=decision.spread_floor_overlay_half_spread_bps,
                # 2026-05-13 todo-019 Part B: tick-aware floor in
                # normal-mode path too (two-sided competitive).
                price_tick=tick if tick > 0 else None,
                fair_value=mid_ref,
                # v1.5.281 AQC Phase 2 — PRIMARY wiring. In normal_mode
                # this econ floor is the binding tight-side term on a
                # wide book (market anchor + clip never bite below it
                # when NORMAL_MM_MAX_COMPETITIVE_HALF_SPREAD_BPS=0), so
                # tightening it here is where aggression earns fills.
                # No-op unless AQC_WIRE_MIN_HALF_SPREAD is on.
                aqc_aggression_level=getattr(
                    decision, "aqc_aggression_level", None
                ),
            )
            capped_h, normal_diag = compute_normal_mm_market_capped_half_spread_bps(
                self._settings,
                model_half_spread_bps=eff_h,
                best_bid=best_bid,
                best_ask=best_ask,
                mid=mid_ref,
                tick=tick if tick > 0 else 1e-12,
            )
            min_half_spread_px = (float(capped_h) / 10_000.0) * mid_ref
            bid_px = mid_ref - min_half_spread_px
            ask_px = mid_ref + min_half_spread_px

        bid_px_adj, bid_aging_diag = adjust_bid_target_for_aging(
            self._settings,
            decision,
            bid_px_model=float(bid_px),
            best_bid=best_bid,
            tick=tick if tick > 0 else 1e-12,
            working=ctx.resting_bid,
            now=utc_now(),
            position_qty=float(ctx.position_qty),
            enforce_touch_distance_band=True,
        )
        ask_px_adj, ask_aging_diag = adjust_ask_target_for_aging(
            self._settings,
            decision,
            ask_px_model=float(ask_px),
            best_ask=best_ask,
            tick=tick if tick > 0 else 1e-12,
            working=ctx.resting_ask,
            now=utc_now(),
            position_qty=float(ctx.position_qty),
            enforce_touch_distance_band=True,
        )
        # One-model invariant: when ``normal_mode`` is active, the market-anchored
        # spread from ``compute_normal_mm_market_capped_half_spread_bps`` IS the
        # authoritative spread-setter. The economic spread floor existed as a
        # fallback "don't quote at a loss" guard for one-sided / degraded modes;
        # firing it here overrides the market anchor with a wider
        # ``compute_effective_min_half_spread_bps`` value, which was exactly the
        # layered disagreement observed in ``tmp/snap_20260418_094415``: normal_mm
        # produced a 3-tick spread, the economic floor silently widened it to
        # 138 ticks, and the bot quoted invisibly for 17 minutes.
        #
        # Apply the floor ONLY when we're outside normal_mm (one-sided, stale book,
        # risk-capped). In that regime there is no market anchor and the floor is
        # the correct fallback.
        if not normal_mode:
            bid_px_adj, ask_px_adj, _ = apply_profitability_spread_floor(
                self._settings,
                decision,
                want_bid=want_bid,
                want_ask=want_ask,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_px=bid_px_adj,
                ask_px=ask_px_adj,
                # 2026-05-13 todo-019 Part B: thread tick so the
                # economic floor can compose a tick-denominated
                # floor on tight-tick symbols.
                price_tick=tick if tick > 0 else None,
            )
        bid_px_adj, ask_px_adj = widen_two_sided_if_collapsed_to_one_tick(
            bid_px_adj,
            ask_px_adj,
            want_bid=want_bid,
            want_ask=want_ask,
            tick=tick if tick > 0 else 1e-12,
            best_bid=best_bid,
            best_ask=best_ask,
        )

        # #1: behind-touch on the adding side at high inventory. No-op
        # under the default ``INVENTORY_HIGH_ADDING_SIDE_BUFFER_TICKS=0``
        # (current SUI profile). Applied AFTER aging tighten and AFTER
        # the collapsed-to-one-tick widening — both of those move
        # toward the touch on the reducing side; this only pushes the
        # ADDING side further away, so they don't fight.
        bid_px_adj, ask_px_adj, inv_buf_diag = apply_inventory_high_adding_side_buffer(
            settings=self._settings,
            position_qty=float(ctx.position_qty),
            bid_px=float(bid_px_adj),
            ask_px=float(ask_px_adj),
            best_bid=best_bid,
            best_ask=best_ask,
            tick=tick if tick > 0 else 1e-12,
        )

        # ``max_allowed_size`` is the position-cap-derived ceiling, NOT
        # the post-intent-clip size. This lets ``_build_side``'s
        # min-notional self-heal bump a rounded-down candidate up to
        # the position cap, while still preventing it from exceeding
        # ``max_abs_position`` / ``max_position_notional_usd``.
        #
        # Original 2026-04-18 bug (``tmp/snap_20260418_140635``): pos
        # ran from −0.045 to −0.060 because self-heal bumped 0.005 →
        # 0.015 with no upper bound. Position cap as the bound fixes
        # that AND the 2026-05-08 wedge from snapshot 260507123509
        # where the candidate had been clipped by max_order_notional
        # rather than the position cap, so passing the post-clip size
        # over-tightened self-heal: candidate=2.985 → rounded to 2 →
        # below min_notional → self-heal needed 3 → blocked because
        # max_allowed_size=2.985 < 3 → engine returned no_quote and
        # bot wedged.
        bid_order, bid_reason, bid_npx, bid_nsz = self._build_side(
            side=Side.BUY,
            want_side=want_bid,
            candidate_px=float(bid_px_adj),
            candidate_sz=float(bid_sz),
            best_bid=best_bid,
            best_ask=best_ask,
            tick=tick if tick > 0 else 1e-12,
            min_half_spread_px=min_half_spread_px,
            mid_ref=mid_ref,
            max_allowed_size=float(max_buy_position_cap),
        )
        ask_order, ask_reason, ask_npx, ask_nsz = self._build_side(
            side=Side.SELL,
            want_side=want_ask,
            candidate_px=float(ask_px_adj),
            candidate_sz=float(ask_sz),
            best_bid=best_bid,
            best_ask=best_ask,
            tick=tick if tick > 0 else 1e-12,
            min_half_spread_px=min_half_spread_px,
            mid_ref=mid_ref,
            max_allowed_size=float(max_sell_position_cap),
        )

        # todo-005 / todo-006: stamp the per-side quality tags onto the
        # FinalQuoteOrder so OrderManager can persist them on the
        # WorkingOrder at place-time. ``decision.target_spread_bps`` is
        # the full target spread (bid + ask); halve it for the per-side
        # half-spread reference shown in the spread-quality report.
        # ``aging_tighten_applied`` from the aging diagnostic flips True
        # whenever ``adjust_*_target_for_aging`` actively pushed the
        # price tighter than the model-computed target — the
        # "aged_tightened" aggressiveness category. Wrapping FinalQuote-
        # Order via ``replace`` keeps it frozen + slot-ed.
        target_half_spread_bps = float(decision.target_spread_bps) / 2.0
        if bid_order is not None:
            bid_order = replace(
                bid_order,
                target_half_spread_bps=target_half_spread_bps,
                aging_tighten_applied=bool(bid_aging_diag.tighten_applied),
            )
        if ask_order is not None:
            ask_order = replace(
                ask_order,
                target_half_spread_bps=target_half_spread_bps,
                aging_tighten_applied=bool(ask_aging_diag.tighten_applied),
            )

        bid_before_bias, ask_before_bias = bid_order, ask_order
        bid_order, ask_order = self._apply_inventory_execution_bias(ctx, bid_order, ask_order)
        if bid_before_bias != bid_order or ask_before_bias != ask_order:
            if bid_order is None and bid_before_bias is not None:
                bid_reason = "inventory_bias_suppressed_adding_side"
            if ask_order is None and ask_before_bias is not None:
                ask_reason = "inventory_bias_suppressed_adding_side"

        # #5: markout-based quote aging. Pre-cancel a resting quote
        # when current mid has drifted adverse to the resting price
        # for at least the configured duration. Adverse means: bid
        # above mid (counterparty hitting it has free money) or ask
        # below mid (same, opposite side). Both are the toxic-fill
        # setup; pulling the quote pre-emptively avoids the fill.
        #
        # Implemented as "set the order to None" — the orchestrator
        # in execution.py already cancels the resting working order
        # when ``desired is None`` and a working order exists. Next
        # cycle places a fresh quote from updated mid; if the drift
        # continues, the new quote will get cancelled too. The cycle
        # naturally throttles via ``min_replace_interval_ms``.
        #
        # Default off (threshold=0 / duration=0). Calibration target:
        # post-colo. Today's latency profile masks part of the signal
        # we're trying to surface.
        mk_threshold = float(self._settings.quote_aging_markout_adverse_bps_threshold)
        mk_duration = float(self._settings.quote_aging_markout_adverse_duration_seconds)
        bid_markout_adverse_bps = markout_adverse_bps_for_side(
            side=Side.BUY, working=ctx.resting_bid, mid=mid_ref
        )
        ask_markout_adverse_bps = markout_adverse_bps_for_side(
            side=Side.SELL, working=ctx.resting_ask, mid=mid_ref
        )
        bid_mk_cancel, bid_mk_elapsed = self._markout_adverse_tracker.evaluate(
            side=Side.BUY,
            working=ctx.resting_bid,
            adverse_bps=bid_markout_adverse_bps,
            threshold_bps=mk_threshold,
            duration_seconds=mk_duration,
            now=utc_now(),
        )
        ask_mk_cancel, ask_mk_elapsed = self._markout_adverse_tracker.evaluate(
            side=Side.SELL,
            working=ctx.resting_ask,
            adverse_bps=ask_markout_adverse_bps,
            threshold_bps=mk_threshold,
            duration_seconds=mk_duration,
            now=utc_now(),
        )
        if bid_mk_cancel and bid_order is not None:
            bid_order = None
            bid_reason = "markout_adverse_cancel"
        if ask_mk_cancel and ask_order is not None:
            ask_order = None
            ask_reason = "markout_adverse_cancel"

        mode = "no_quote"
        if bid_order and ask_order:
            mode = "two_sided"
        elif bid_order or ask_order:
            mode = "one_sided"

        # Telemetry: legacy keys (exec_*, finalize_ms, downgraded_*) are historical names for dashboards;
        # quote construction and normalization happen only in this module.
        telemetry: dict[str, Any] = {
            "quote_engine_mode": mode,
            "quote_engine_inventory_bias_active": self._inventory_exec_bias_active(
                float(ctx.position_qty),
                effective_max_abs_position=ctx.effective_max_abs_position,
                aqc_aggression_level=getattr(
                    ctx.decision, "aqc_aggression_level", None
                ),
            ),
            "quote_engine_inventory_bias_suppressed_bid": bool(
                bid_before_bias is not None and bid_order is None
            ),
            "quote_engine_inventory_bias_suppressed_ask": bool(
                ask_before_bias is not None and ask_order is None
            ),
            "quote_engine_normal_mode_requested": bool(normal_mode),
            "quote_engine_bid_reason": bid_reason,
            "quote_engine_ask_reason": ask_reason,
            "quote_engine_bid_candidate_px": float(bid_px_adj),
            "quote_engine_ask_candidate_px": float(ask_px_adj),
            "quote_engine_bid_candidate_sz": float(bid_sz),
            "quote_engine_ask_candidate_sz": float(ask_sz),
            "quote_engine_bid_final_px": bid_order.price if bid_order else None,
            "quote_engine_ask_final_px": ask_order.price if ask_order else None,
            "quote_engine_bid_final_sz": bid_order.size if bid_order else None,
            "quote_engine_ask_final_sz": ask_order.size if ask_order else None,
            "exec_raw_bid_px": float(bid_px_adj) if want_bid else None,
            "exec_raw_bid_sz": float(bid_sz) if want_bid else None,
            "exec_raw_ask_px": float(ask_px_adj) if want_ask else None,
            "exec_raw_ask_sz": float(ask_sz) if want_ask else None,
            # exec_norm_* / wire: mirror FinalQuoteOrder (same values execution submits verbatim).
            "exec_norm_bid_px": bid_order.price if bid_order else None,
            "exec_norm_bid_sz": bid_order.size if bid_order else None,
            "exec_norm_ask_px": ask_order.price if ask_order else None,
            "exec_norm_ask_sz": ask_order.size if ask_order else None,
            "exec_price_tick": float(self._spec.price_tick),
            "exec_size_step": float(self._spec.size_step),
            "exec_meta_decimal_grid_price_tick": float(self._spec.price_tick),
            "exec_meta_decimal_size_step": float(self._spec.size_step),
            "exec_hl_max_sig_figs_nonint": 5,
            "exec_price_normalize_pipeline": "hl_perp_limit_price_pipeline",
            "exec_wire_bid_limit_p": wire_format_preview_limit_px(bid_order.price)
            if bid_order
            else None,
            "exec_wire_ask_limit_p": wire_format_preview_limit_px(ask_order.price)
            if ask_order
            else None,
            "final_submitted_bid_px": bid_order.price if bid_order else None,
            "final_submitted_ask_px": ask_order.price if ask_order else None,
            "pre_finalize_norm_bid_px": bid_npx,
            "pre_finalize_norm_ask_px": ask_npx,
            "post_only_adjustment_applied_bid": False,
            "post_only_adjustment_applied_ask": False,
            "economic_floor_requested_bid": False,
            "economic_floor_requested_ask": False,
            "economic_floor_blocked_by_post_only_bid": False,
            "economic_floor_blocked_by_post_only_ask": False,
            "exec_normal_mm_contract_active": bool(normal_mode),
            "exec_normal_mm_contract_inactive_reason": None if normal_mode else "not_selected",
            "normal_mm_cycle_valid": bool((not normal_mode) or (bid_order is not None and ask_order is not None)),
            "normal_mm_downgrade_reason": "one_side_invalid_after_construction"
            if (normal_mode and not (bid_order and ask_order))
            else None,
            "normal_mm_bid_contract_ok": bool((not normal_mode) or (bid_order is not None)),
            "normal_mm_ask_contract_ok": bool((not normal_mode) or (ask_order is not None)),
            "downgraded_cycle_fallback_attempted": bool(normal_mode and not (bid_order and ask_order)),
            "downgraded_cycle_fallback_placed": bool(normal_mode and ((bid_order is not None) ^ (ask_order is not None))),
            "downgraded_cycle_skipped_reason": "no_executable_fallback_quote"
            if (normal_mode and bid_order is None and ask_order is None)
            else None,
            "downgraded_side_target": (
                Side.BUY.value if (normal_mode and bid_order and not ask_order)
                else Side.SELL.value if (normal_mode and ask_order and not bid_order)
                else None
            ),
            "downgraded_side_executable": bool(normal_mode and ((bid_order is not None) ^ (ask_order is not None))),
            "bid_executable": bid_order is not None,
            "ask_executable": ask_order is not None,
            "bid_executable_notional_usd": (bid_order.price * bid_order.size) if bid_order else None,
            "ask_executable_notional_usd": (ask_order.price * ask_order.size) if ask_order else None,
            "bid_min_notional_block": bool(want_bid and bid_order is None),
            "ask_min_notional_block": bool(want_ask and ask_order is None),
            "quote_contract_build_ms": (time.perf_counter() - t0) * 1000.0,
            "executable_size_check_ms": 0.0,
            "placement_mode_eval_ms": 0.0,
            "finalize_ms": 0.0,
            "order_submit_prep_ms": 0.0,
        }
        for k, v in normal_diag.items():
            telemetry[f"quote_engine_normal_{k}"] = v
        # #1 telemetry — surfaced regardless of feature on/off so the
        # dashboard can show "this profile would have buffered, but
        # the feature is off".
        for k, v in inv_buf_diag.items():
            telemetry[k] = v
        # #5 telemetry — current adverse_bps and breach-elapsed for
        # both sides; either may be None when no resting order or
        # the feature is off.
        telemetry["markout_adverse_bps_bid"] = bid_markout_adverse_bps
        telemetry["markout_adverse_bps_ask"] = ask_markout_adverse_bps
        telemetry["markout_adverse_breach_elapsed_s_bid"] = bid_mk_elapsed
        telemetry["markout_adverse_breach_elapsed_s_ask"] = ask_mk_elapsed
        telemetry["markout_adverse_cancel_bid"] = bool(bid_mk_cancel)
        telemetry["markout_adverse_cancel_ask"] = bool(ask_mk_cancel)
        # 1.3.59: strict age-cap evaluation. Returns non-empty when
        # the resting order has exceeded the operator-configured
        # ``BEHIND_TOUCH_MAX_AGE_SECONDS`` or
        # ``AT_TOUCH_MAX_AGE_SECONDS`` thresholds. Empty otherwise.
        # Both default to 0.0 = disabled, so this is a no-op unless
        # the operator has opted in. Pre-1.3.59 these functions
        # existed but were never called — the strict caps were
        # dead code. See ``app/quote_aging.py::hard_reprice_reasons_*``.
        #
        # v1.4.70 wedge-elimination-cleanup Phase 1C: evaluate the cap
        # for EVERY (side, level_idx) slot in ``ctx.working_orders_by_slot``,
        # not just the inside rung. Outer rungs (level_idx >= 1) were
        # invisible to the cap pre-Phase-1C (Codex F6/B3).
        #
        # Backward compat: when ``working_orders_by_slot`` is empty
        # (any caller that hasn't been updated), fall back to the old
        # inside-rung-only path via ``ctx.resting_bid`` /
        # ``ctx.resting_ask``.
        # v1.4.84 Phase 4B — aging-signals computation extracted to
        # ``compute_aging_signals`` module-level pure function.
        # Behaviour is byte-identical to the pre-Phase-4B inline block;
        # see the function docstring for the multi-rung / fallback
        # contract.
        _tick = self._spec.price_tick if self._spec.price_tick > 0 else 1e-12
        hard_cancel_by_slot = compute_aging_signals(
            self._settings,
            ctx.working_orders_by_slot,
            ctx.resting_bid,
            ctx.resting_ask,
            best_bid=best_bid,
            best_ask=best_ask,
            tick=_tick,
            now=utc_now(),
        )
        # Backward-compat tuples (inside rung only).
        hard_bid: tuple[str, ...] = hard_cancel_by_slot.get((Side.BUY, 0), ())
        hard_ask: tuple[str, ...] = hard_cancel_by_slot.get((Side.SELL, 0), ())
        # NOTE: ``hard_cancel_{bid,ask}_reasons`` deliberately NOT put
        # into ``telemetry`` — that dict gets merged into the
        # ``quote_decisions`` SQL row, and unknown columns there cause
        # an OperationalError. The canonical surface is the dedicated
        # fields on ``QuoteBuildResult`` below.
        return QuoteBuildResult(
            bid_order=bid_order,
            ask_order=ask_order,
            mode=mode,
            telemetry=telemetry,
            hard_cancel_bid_reasons=hard_bid,
            hard_cancel_ask_reasons=hard_ask,
            hard_cancel_by_slot=hard_cancel_by_slot,
        ).to_build_command()
