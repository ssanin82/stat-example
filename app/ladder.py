"""Multi-level (laddered) quoting — Phase 1 (shadow / instrumentation).

This module implements the data layer described in
``plans/multi-level.md``: given the bot's already-computed
``QuoteDecision`` (the inside bid / ask / sizes), produce a
``LadderDecision`` with N rungs per side at incrementally wider
prices and geometric-decay sizes. **At N = 1 the ladder is a
single rung that exactly matches the inside scalar quote** — no
behavioural drift vs v1.2.13.

This file is the COMPUTE side only. Phase 1 hooks the result into
``quote_decisions`` row logging so the operator can analyze what a
multi-level ladder would have looked like in a snapshot, without
the execution layer actually placing outer-rung orders. Phase 2
(separate plan + implementation) will wire the execution path so
outer rungs are real orders on the venue.

Why split that way:
* The ladder *shape* (offsets, sizes, rung counts) is the part the
  operator most needs to calibrate from snapshot data — the
  execution mechanics (batch endpoints, multi-rung diff, clOrdId
  encoding) are mechanical once the shape is right.
* The bot has been losing money under the current single-level
  path. Adding outer-rung exposure on top of that without first
  calibrating the shape on a snapshot run is uncomfortable. Shadow
  mode lets us pull a session's worth of "what would the ladder
  have done" data without taking the risk.

Public API:
* ``LadderConfig``     — settings dataclass; loaded from app.config
* ``LadderRung``       — one rung (level_idx, side, px, sz)
* ``LadderDecision``   — the full ladder for one cycle + diagnostics
* ``build_ladder()``   — pure function: QuoteDecision → LadderDecision

All ladder fields are JSON-serialisable for SQLite TEXT storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.enums import ActiveSides, Side
from app.models import QuoteDecision


@dataclass
class LadderConfig:
    """Operator-tunable ladder shape parameters. All knobs are no-ops
    when ``num_levels_per_side == 1`` — the build_ladder() function
    short-circuits at that case to return a single rung that exactly
    matches the input QuoteDecision's scalar values.

    Default values reproduce v1.2.13 behavior bit-for-bit:
    one rung per side equal to the inside quote.
    """

    # 1..5 inclusive. 1 = single-level (current behavior); higher
    # values activate multi-rung. Stored as int for clean comparison
    # in the bot's cycle code.
    num_levels_per_side: int = 1

    # Rung price spacing as a multiple of the cycle's
    # ``half_spread_bps``. Each rung i (i = 0..N-1) sits at:
    #
    #     half_spread × (1 + offset_step × i)
    #
    # away from the reservation price. ``offset_step = 0`` collapses
    # all rungs to the inside (degenerate; not blocked because the
    # validation belongs at config-load time, not here). ``1.0`` is
    # the recommended default — rung i sits at i+1 half-spreads from
    # the reservation, so 5 rungs cover the band from 1× to 5× half-
    # spread. Larger values produce sparser ladders, smaller values
    # tighter ladders.
    offset_step: float = 1.0

    # Geometric size decay ratio. Rung i's size before any other
    # multipliers is:
    #
    #     base_size × size_decay ^ i           (inside_full_size=False)
    #     base_size × size_decay ^ max(0, i-1) (inside_full_size=True; default)
    #
    # 1.0 = flat (every rung same size — discouraged because outer-
    # rung adverse selection on sweep days then maximises rather
    # than minimises losses). 0.7 is the plan's default; 0.5
    # aggressive shrinking. Range 0.1..1.0 enforced at config load.
    size_decay: float = 0.7

    # When True (default), rung 0 (inside) gets full base_size and
    # decay starts at rung 1. When False, decay starts at rung 0
    # (rung 0 size = base_size × 1.0 = unchanged anyway, but rung 1
    # = base × decay). The two modes diverge at rung 2+:
    #
    #   inside_full_size=True  → 1.0, decay, decay²,  decay³,  ...
    #   inside_full_size=False → 1.0, decay, decay²,  decay³,  ...  (same)
    #
    # NOTE: with the formula above, inside_full_size only changes
    # rung 1's exponent. Kept as a knob for symmetry with the plan
    # doc; the practical impact is small.
    inside_full_size: bool = True

    # When True (default), gates can publish ``max_levels_per_side``
    # caps that clamp the effective N below ``num_levels_per_side``.
    # When False, gate caps are ignored and the ladder always runs
    # at the configured depth — useful for shadow-mode calibration
    # where the operator wants to see the unclamped ladder shape.
    gates_limit_levels: bool = True

    # Phase 2J (v1.4.169) — tick-floored ladder offset. Minimum gap
    # between adjacent rungs measured in TICKS, applied BEFORE the
    # legacy grid-collision dedup check. The bps math
    # (``half_spread × (1 + offset_step × i)``) can underflow the
    # venue's price-tick when ``half_spread`` is small relative to
    # the tick (e.g. TON-USDT-SWAP with target_half_spread=4 bps and
    # tick=0.001 → 1 bp ≈ 0.00021 ticks at $2.04; rungs 1 + 2 can
    # both snap to the same grid cell and the outer rung gets
    # dropped silently as ``grid_collision``).
    #
    # Tick-floor enforcement: rung ``i`` (i ≥ 1) MUST sit at least
    # ``i * tick_floor_steps * tick_size`` away from the inside
    # rung (rung 0). When the bps math produces a closer price, the
    # rung is shifted OUTWARD to the floor; the legacy
    # ``grid_collision`` dedup then becomes a defense-in-depth
    # safety net (should never fire after the floor is applied, but
    # kept for the case where a future code path bypasses the floor).
    #
    # ``tick_floor_steps = 1`` (default) = "at least 1 tick gap per
    # rung step". ``0`` disables the floor (legacy pre-v1.4.169
    # behaviour — the bps math + collision-dedup combo is the only
    # safeguard). Higher values (2, 3) reserve more grid headroom
    # on symbols where 1-tick steps are too aggressive.
    #
    # Counter: ``BotState.ladder_rung_tick_floor_adjusted_total``
    # increments once per rung whose price was shifted by the floor
    # (so the operator sees how often the bps math was about to
    # underflow the tick).
    tick_floor_steps: int = 1

    # Phase 2 flag — wires multi-rung execution via OKX batch
    # endpoints. Defaults False so Phase 1 deploys can flip
    # ``num_levels_per_side`` to 5 without risking real outer-rung
    # orders (only inside rung is placed). Phase 2 lands this flag
    # together with the execution path; Phase 1 ignores it.
    #
    # **DEAD-FLAG NOTE (2026-05-19)**: this flag has no consumer in
    # the codebase. Multi-rung execution shipped UNCONDITIONALLY in
    # v1.4.0+; per-side rung places coalesce through the batch
    # endpoint via the dispatcher's ``BATCH_PLACES_ALWAYS=true``
    # path. The flag is kept here for backward-compatibility with
    # operator profiles that set it; safe to ignore.
    batch_orders_enabled: bool = False

    # ----------------------------------------------------------------
    # v1.4.99 — Inventory-aware rung pruning (DORMANT by default)
    # ----------------------------------------------------------------
    #
    # When ``inventory_aware_pruning_enabled=True``: at ladder-build
    # time, if ``|position_qty| / max_abs_position >= pruning_threshold_pct``,
    # the ADDING side's outer rungs (level ≥ 1) are dropped. The
    # adding side is the one that would INCREASE |position| if filled
    # (bid when long, ask when short).
    #
    # Rationale (see plans/multi-level.md Phase 2 residual): when the
    # bot is loaded directional, outer adding-side rungs are toxic —
    # they catch fills further from mid that systematically push
    # |position| past the soft cap, then drain at adverse markouts.
    # The existing ``apply_inventory_high_adding_side_buffer`` widens
    # the adding-side quote but doesn't drop rungs explicitly; this
    # is the deterministic hard-cap layer on top.
    #
    # **DORMANT BY DEFAULT** (``enabled=False``). When disabled the
    # pruning branch is a no-op: no computation, no counter bumps,
    # no behaviour change vs pre-v1.4.99. The 'enabled' flag is the
    # ONLY gate — flip the env knob when calibration data
    # (per-rung markouts at v1.4.98+ horizons; rung-drop attribution
    # counters from plans/ladder-observability.md F2) justifies it.
    #
    # The reducing side is always left untouched (its full ladder
    # accelerates inventory bleed-off — same intent as existing
    # inventory-skew machinery).
    #
    # See ``plans/ladder-observability.md`` for the calibration
    # framework that should drive the decision to flip this on.
    inventory_aware_pruning_enabled: bool = False
    # Threshold for engaging the prune (fraction of max_abs_position).
    # Default 0.65 matches ``INVENTORY_SOFT_LIMIT_PCT`` so the prune
    # engages at the same point as the existing one-sided clamps.
    # Below the threshold: full ladder both sides. Above: adding-side
    # outer rungs cut.
    inventory_aware_pruning_threshold_pct: float = 0.65


@dataclass
class LadderRung:
    """One rung in the ladder. Always carries an absolute price and
    size in venue units; downstream serializers can convert as
    needed. ``level_idx`` is 0-based with 0 = inside (closest to
    mid)."""

    level_idx: int
    side: Side
    px: float
    sz: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level_idx,
            "side": self.side.value,
            "px": float(self.px),
            "sz": float(self.sz),
        }


@dataclass
class LadderDecision:
    """Output of build_ladder(). Contains the per-rung ladder plus
    diagnostics (effective N per side after gate caps, the gate
    cap dict). Persisted to the ``quote_decisions`` row so a
    snapshot carries everything needed to analyze "what would the
    ladder have done."
    """

    bids: list[LadderRung]  # length 0 .. effective_levels_buy
    asks: list[LadderRung]  # length 0 .. effective_levels_sell
    # Configured N pre-cap (i.e. cfg.num_levels_per_side). Useful
    # in postmortem to see what was requested vs what gates allowed.
    requested_levels: int
    # Post-cap N actually emitted per side. When no gate fires:
    # equal to requested_levels. When a one-sided gate fires
    # (e.g. inventory bias or microprice gate suppressing the thin
    # side): one of these can be 0 while the other is full.
    effective_levels_buy: int
    effective_levels_sell: int
    # Gate name → cap (or None if gate published no cap). Only
    # populated when cfg.gates_limit_levels is True.
    gate_caps: dict[str, Optional[int]] = field(default_factory=dict)

    @property
    def inside_bid(self) -> Optional[LadderRung]:
        """Convenience accessor for the inside (level=0) bid rung,
        or None when buy side is fully suppressed.
        """
        return self.bids[0] if self.bids else None

    @property
    def inside_ask(self) -> Optional[LadderRung]:
        return self.asks[0] if self.asks else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict — what gets persisted to the
        ``quote_decisions`` row's TEXT columns. Compact: only the
        per-rung fields, no Python type info."""
        return {
            "bids": [r.to_dict() for r in self.bids],
            "asks": [r.to_dict() for r in self.asks],
            "requested_levels": int(self.requested_levels),
            "effective_levels_buy": int(self.effective_levels_buy),
            "effective_levels_sell": int(self.effective_levels_sell),
            "gate_caps": dict(self.gate_caps),
        }


def _effective_n_for_side(
    *,
    cfg: LadderConfig,
    gate_caps: dict[str, Optional[int]],
    side_active: bool,
) -> int:
    """Resolve the actual rung count for one side after applying
    gate caps. Returns 0 when the side is suppressed entirely
    (e.g. by inventory bias — caller passes side_active=False).
    """
    if not side_active:
        return 0
    n = int(cfg.num_levels_per_side)
    if cfg.gates_limit_levels and gate_caps:
        for cap in gate_caps.values():
            if cap is not None:
                n = min(n, int(cap))
    return max(0, n)


def _size_for_rung(
    *,
    base_size: float,
    level_idx: int,
    cfg: LadderConfig,
) -> float:
    """Geometric decay starting at rung 0 (or rung 1 if
    inside_full_size=True). Returns a non-negative float; clamped
    to 0 if upstream pathologically passed a negative base_size.
    """
    if base_size <= 0.0:
        return 0.0
    if level_idx <= 0:
        return float(base_size)
    if cfg.inside_full_size:
        # rung 1 multiplier = decay, rung 2 = decay², etc.
        # exponent = level_idx (since rung 1 → decay^1)
        exponent = level_idx
    else:
        exponent = level_idx
    return float(base_size) * (float(cfg.size_decay) ** int(exponent))


def _px_for_bid_rung(
    *,
    reservation: float,
    half_spread_bps: float,
    level_idx: int,
    cfg: LadderConfig,
    inside_px_override: Optional[float] = None,
) -> float:
    """Compute the price for a bid rung. Rung 0 returns the
    operator-supplied ``inside_px_override`` (the post-clamp
    inside quote from compute_quote_decision) so the inside rung
    is bit-identical to today's behavior. Outer rungs are
    derived from reservation + offset_step.
    """
    if level_idx <= 0 and inside_px_override is not None:
        return float(inside_px_override)
    factor = 1.0 + float(cfg.offset_step) * float(level_idx)
    return float(reservation) * (1.0 - float(half_spread_bps) * factor / 10_000.0)


def _px_for_ask_rung(
    *,
    reservation: float,
    half_spread_bps: float,
    level_idx: int,
    cfg: LadderConfig,
    inside_px_override: Optional[float] = None,
) -> float:
    if level_idx <= 0 and inside_px_override is not None:
        return float(inside_px_override)
    factor = 1.0 + float(cfg.offset_step) * float(level_idx)
    return float(reservation) * (1.0 + float(half_spread_bps) * factor / 10_000.0)


def _tick_floor(px: float, tick: float) -> float:
    """Round ``px`` down to the nearest multiple of ``tick``.

    v1.4.86 wedge-elimination-cleanup: this helper is kept for
    callers that explicitly want floor semantics. The v1.4.79 ladder
    dedup originally used ``_tick_floor`` / ``_tick_ceil`` to predict
    grid collisions, but the ACTUAL ``round_price_and_size_to_grid``
    uses ``Decimal.quantize(tick, rounding=ROUND_HALF_UP)`` — i.e.
    round-to-nearest, NOT floor. The mismatch let same-price
    duplicates slip through when the outer rung's raw px sat in the
    round-up half of the inside rung's tick. The dedup now uses
    ``_tick_round_half_up`` (below) to match the actual rounding.
    """
    if tick <= 0:
        return float(px)
    return (int(float(px) / float(tick))) * float(tick)


def _tick_ceil(px: float, tick: float) -> float:
    """Round ``px`` up to the nearest multiple of ``tick``.

    v1.4.86 note: same caveat as ``_tick_floor`` — kept for
    historical callers but the dedup now uses ``_tick_round_half_up``.
    """
    if tick <= 0:
        return float(px)
    p = float(px)
    t = float(tick)
    n = int(p / t)
    if abs(n * t - p) < 1e-12:
        return n * t
    if p > 0:
        return (n + 1) * t
    return n * t


def _tick_round_half_up(px: float, tick: float) -> float:
    """v1.4.86 wedge-elimination-cleanup — predict the actual venue
    grid rounding.

    ``round_price_and_size_to_grid`` (app/exchange/hyperliquid_precision.py)
    uses ``Decimal.quantize(tick, rounding=ROUND_HALF_UP)``. The
    ladder-collision dedup must use the SAME rounding semantic to
    correctly predict whether two rungs will collapse to the same
    grid value after the engine's downstream normalization.

    Pre-v1.4.86 the dedup used ``_tick_floor`` / ``_tick_ceil`` which
    are floor / ceil semantics. When an outer-rung raw px sat above
    the half-tick mark (e.g. 2.0312 with tick 0.001), floor gave
    2.031 while ROUND_HALF_UP gave 2.031 too — but the dedup also
    floored the inside override (e.g. 2.0317 → 2.031) and the
    collision SHOULD have fired. The actual failure mode: inside
    override is sometimes pre-rounded by upstream code to a
    DIFFERENT tick (e.g., to 2.032 via ROUND_HALF_UP on a 2.0317
    raw), so the dedup predicts 2.032 vs 2.031 (NOT a collision)
    while the actual placement rounds both to 2.031 (collision).

    Using ROUND_HALF_UP everywhere in the dedup eliminates the
    prediction-vs-actual rounding-mode drift.

    The implementation uses ``Decimal`` for exact rounding rather
    than ``round()`` (which uses banker's / ROUND_HALF_EVEN) so the
    semantic matches Decimal.quantize bit-for-bit.
    """
    if tick <= 0:
        return float(px)
    from decimal import Decimal, ROUND_HALF_UP
    try:
        px_d = Decimal(str(float(px)))
        tick_d = Decimal(str(float(tick)))
        return float(px_d.quantize(tick_d, rounding=ROUND_HALF_UP))
    except Exception:
        # Defensive fallback: use float round (banker's). The dedup
        # is best-effort; if Decimal fails on some pathological input,
        # we'd rather over-predict collisions than under-predict.
        t = float(tick)
        return round(float(px) / t) * t


def build_ladder(
    *,
    decision: QuoteDecision,
    cfg: LadderConfig,
    half_spread_bps: float,
    gate_caps: Optional[dict[str, Optional[int]]] = None,
    tick_size: Optional[float] = None,
    # v1.4.99 — inventory-aware rung pruning inputs. Both optional
    # for backward-compatibility with callers that don't pass them
    # (test fixtures, scripts) — when either is None, pruning is a
    # no-op regardless of the flag. Production callers always pass
    # both (position_qty from state, max_abs_position from settings).
    position_qty: Optional[float] = None,
    max_abs_position: Optional[float] = None,
    # v1.4.99 — optional drop-attribution callback. When provided,
    # invoked with a ``reason: str`` argument each time a rung is
    # dropped. Hook for the F2 counters in
    # ``plans/ladder-observability.md`` without making ladder.py
    # import-couple to BotState. None = no-op.
    on_rung_dropped: Optional[Callable[[str], None]] = None,
    # v1.4.169 Phase 2J — parallel callback fired when a rung's
    # price was SHIFTED OUTWARD by the tick-floor (so the bps math
    # was about to underflow the venue's price tick). Arg is the
    # side string ``"bid"`` / ``"ask"`` for per-side attribution.
    # None = no-op (default).
    on_rung_floor_adjusted: Optional[Callable[[str], None]] = None,
) -> LadderDecision:
    """Pure function. Convert a single-level QuoteDecision into a
    multi-rung LadderDecision.

    Contract at ``cfg.num_levels_per_side <= 1``:
    * Returns at most 1 rung per side.
    * The inside rung's px / sz exactly match the input
      decision's scalar fields (``quoted_bid``, ``quoted_bid_sz``,
      etc.). No new arithmetic — just a wrap.
    * This is the v1.2.13-equivalent path. By construction, no
      behavioural drift.

    Contract at ``cfg.num_levels_per_side > 1``:
    * Inside rung (level=0) preserves the input scalar values
      (so the bot's existing post-only / clamp logic still
      applies on the inside, unchanged).
    * Outer rungs (level=1..N-1) are computed from
      ``decision.reservation_price`` + offset_step × half_spread.
      Sizes follow geometric decay from the inside size.
    * Gate caps (when supplied + gates_limit_levels=True) clamp
      the effective rung count per side. Inventory bias / one-
      sided eligibility further suppress one whole side via
      ``decision.active_sides``.

    Args:
        decision: Output of compute_quote_decision. Carries
            mid, reservation, target spread, inside scalar quote.
        cfg: LadderConfig from settings.
        half_spread_bps: Effective half-spread in bps for THIS
            cycle (e.g. ``decision.target_spread_bps / 2.0``).
            Provided separately rather than recomputed because the
            cycle's effective half-spread already includes any
            adaptive widening / floor overlays — using
            decision.target_spread_bps / 2 is the right call site.
        gate_caps: Optional dict of gate name → max_levels_per_side
            cap (None = no cap from that gate). Pass {} or None to
            run uncapped. Only consulted when cfg.gates_limit_levels
            is True.

    Returns: LadderDecision with bids[] and asks[] populated.
    """
    if gate_caps is None:
        gate_caps = {}

    # Side activeness from the existing eligibility / inventory bias.
    bid_active = decision.active_sides in (ActiveSides.BOTH, ActiveSides.BID_ONLY)
    ask_active = decision.active_sides in (ActiveSides.BOTH, ActiveSides.ASK_ONLY)

    # Effective N per side post-gate-caps. ``effective_n`` is the
    # max for either side; per-side suppression then zeros out the
    # appropriate one.
    effective_n_buy = _effective_n_for_side(
        cfg=cfg, gate_caps=gate_caps, side_active=bid_active
    )
    effective_n_sell = _effective_n_for_side(
        cfg=cfg, gate_caps=gate_caps, side_active=ask_active
    )

    # v1.4.99 — Inventory-aware rung pruning. Dormant by default.
    # When enabled AND |position|/max_abs_position >= threshold,
    # clamp the ADDING side to a single rung (level 0). Reducing
    # side is untouched.
    #
    # ALL guards on the if-statement are checked together so the
    # entire pruning computation (threshold math, side determination)
    # is skipped when ``inventory_aware_pruning_enabled=False`` —
    # the disabled path is a TRUE no-op, no side effects, no
    # counter bumps, no behaviour change vs pre-v1.4.99. This is
    # the contract the snapshot-analysis depends on: enabling the
    # code path must not perturb anything during the dormant phase.
    #
    # When enabled, the prune fires only when both ``position_qty``
    # and ``max_abs_position`` are provided AND well-formed; missing
    # inputs degrade to no-prune rather than raising. The reducing-
    # side untouched policy mirrors the existing
    # ``apply_inventory_high_adding_side_buffer`` (which widens the
    # adding side without touching the reducing side).
    if (
        cfg.inventory_aware_pruning_enabled
        and position_qty is not None
        and max_abs_position is not None
        and max_abs_position > 0
    ):
        try:
            util_pct = abs(float(position_qty)) / float(max_abs_position)
            if util_pct >= float(cfg.inventory_aware_pruning_threshold_pct):
                # Adding side = the side that INCREASES |position| if
                # filled. Long (pos > 0) → BUY adds. Short (pos < 0)
                # → SELL adds. Pos == 0 → no adding side; skip.
                if position_qty > 0 and effective_n_buy > 1:
                    rungs_cut = effective_n_buy - 1
                    effective_n_buy = 1
                    if on_rung_dropped is not None:
                        for _ in range(rungs_cut):
                            try:
                                on_rung_dropped("inventory_aware_pruning")
                            except Exception:
                                # Never let observability callback
                                # break the ladder build.
                                pass
                elif position_qty < 0 and effective_n_sell > 1:
                    rungs_cut = effective_n_sell - 1
                    effective_n_sell = 1
                    if on_rung_dropped is not None:
                        for _ in range(rungs_cut):
                            try:
                                on_rung_dropped("inventory_aware_pruning")
                            except Exception:
                                pass
        except (TypeError, ValueError, ZeroDivisionError):
            # Defensive: malformed inputs fall through to no-prune.
            pass

    # Fast path: N=1 (or effective N=1 for both sides) — skip outer-
    # rung arithmetic entirely and wrap the scalar quote.
    if int(cfg.num_levels_per_side) <= 1:
        bids: list[LadderRung] = []
        asks: list[LadderRung] = []
        if effective_n_buy >= 1 and decision.quoted_bid_sz > 0.0:
            bids.append(
                LadderRung(
                    level_idx=0,
                    side=Side.BUY,
                    px=float(decision.quoted_bid),
                    sz=float(decision.quoted_bid_sz),
                )
            )
        if effective_n_sell >= 1 and decision.quoted_ask_sz > 0.0:
            asks.append(
                LadderRung(
                    level_idx=0,
                    side=Side.SELL,
                    px=float(decision.quoted_ask),
                    sz=float(decision.quoted_ask_sz),
                )
            )
        return LadderDecision(
            bids=bids,
            asks=asks,
            requested_levels=1,
            effective_levels_buy=len(bids),
            effective_levels_sell=len(asks),
            gate_caps=dict(gate_caps),
        )

    # Multi-rung path. Reservation is the anchor; outer rungs are
    # offset from there. Inside rung uses the existing post-clamp
    # quoted_bid/_ask so there's no double-clamping or drift.
    reservation = float(decision.reservation_price)
    base_bid_sz = float(decision.quoted_bid_sz)
    base_ask_sz = float(decision.quoted_ask_sz)

    # v1.4.79 grid-collision dedup. When the engine's half-spread is
    # compressed near MIN_HALF_SPREAD_BPS, the per-rung offset
    # ``half_spread_bps × offset_step`` can be a sub-tick value. After
    # grid-rounding downstream, the inside rung (lvl 0) and outer
    # rung (lvl ≥ 1) can collapse to the SAME grid price — visible
    # operator-facing symptom: two same-side same-price orders on the
    # book.
    #
    # v1.4.86 wedge-elimination-cleanup fix: use ``_tick_round_half_up``
    # to predict the grid result, matching the actual
    # ``round_price_and_size_to_grid`` semantic (ROUND_HALF_UP via
    # Decimal). Pre-v1.4.86 the dedup used ``_tick_floor`` / ``_tick_ceil``
    # which mismatched the actual rounding and let same-price
    # duplicates slip through when raw px sat in the round-up half
    # of a tick (snapshot 260519 BUY 2.031 ×2, SELL 2.034 ×2).
    #
    # The collision check is also tightened: for a BID, lvl i must
    # round to a tick STRICTLY BELOW the previous rung's (any equal-
    # or-greater tick is a collision). Same logic mirrored for ASK
    # (strictly above). The skipped rung's size is NOT merged into
    # the inside — the caller can opt-in to that policy via a higher-
    # layer change. For now: cleanly drop the redundant outer order
    # to avoid wasting a slot.
    bids = []
    # v1.4.169 Phase 2J — tick-floor inside-rung anchors. The
    # tick-floor enforces a minimum gap (in ticks) between rung i
    # and the INSIDE rung's grid position; applied BEFORE the legacy
    # ``grid_collision`` dedup so the dedup is defense-in-depth, not
    # the primary mechanism. ``tick_floor_steps = 0`` disables.
    _tick_floor_steps = int(getattr(cfg, "tick_floor_steps", 0))
    _inside_bid_grid: Optional[float] = None
    _inside_ask_grid: Optional[float] = None
    if tick_size is not None and float(tick_size) > 0:
        _inside_bid_grid = _tick_round_half_up(
            float(decision.quoted_bid), float(tick_size)
        )
        _inside_ask_grid = _tick_round_half_up(
            float(decision.quoted_ask), float(tick_size)
        )

    prev_grid_bid: Optional[float] = None
    for i in range(effective_n_buy):
        sz = _size_for_rung(base_size=base_bid_sz, level_idx=i, cfg=cfg)
        if sz <= 0.0:
            continue
        px = _px_for_bid_rung(
            reservation=reservation,
            half_spread_bps=float(half_spread_bps),
            level_idx=i,
            cfg=cfg,
            inside_px_override=float(decision.quoted_bid) if i == 0 else None,
        )
        # Phase 2J tick-floor: outer bid rung must sit at least
        # ``i * tick_floor_steps * tick`` BELOW the inside-bid grid.
        if (
            i > 0
            and _tick_floor_steps > 0
            and _inside_bid_grid is not None
            and tick_size is not None
            and float(tick_size) > 0
        ):
            tick_floor_px = (
                _inside_bid_grid
                - float(i)
                * float(_tick_floor_steps)
                * float(tick_size)
            )
            # For a BID we want lower-or-equal-to the floor; if the
            # bps math produced something HIGHER (closer to mid),
            # snap to the floor and attribute it.
            if px > tick_floor_px + 1e-12:
                px = tick_floor_px
                if on_rung_floor_adjusted is not None:
                    try:
                        on_rung_floor_adjusted("bid")
                    except Exception:
                        pass
        if tick_size is not None and float(tick_size) > 0:
            grid_px = _tick_round_half_up(px, float(tick_size))
            if i > 0 and prev_grid_bid is not None and (
                # Strict-less-than for BID: lvl i must be at least one
                # tick BELOW lvl i-1's grid.
                grid_px >= prev_grid_bid - 1e-12
            ):
                # Collision: outer rung doesn't move at least one tick
                # off the inner rung. Skip entirely.
                # v1.4.100 F2 — attribute the drop.
                # v1.4.169 Phase 2J: with ``tick_floor_steps >= 1``
                # this branch should be unreachable (the floor above
                # already guarantees ≥ 1-tick separation per step).
                # Kept as defense-in-depth.
                if on_rung_dropped is not None:
                    try:
                        on_rung_dropped("grid_collision")
                    except Exception:
                        pass
                continue
            prev_grid_bid = grid_px
        bids.append(
            LadderRung(level_idx=i, side=Side.BUY, px=px, sz=sz)
        )

    asks = []
    prev_grid_ask: Optional[float] = None
    for i in range(effective_n_sell):
        sz = _size_for_rung(base_size=base_ask_sz, level_idx=i, cfg=cfg)
        if sz <= 0.0:
            continue
        px = _px_for_ask_rung(
            reservation=reservation,
            half_spread_bps=float(half_spread_bps),
            level_idx=i,
            cfg=cfg,
            inside_px_override=float(decision.quoted_ask) if i == 0 else None,
        )
        # Phase 2J tick-floor mirror for asks: outer ask rung must
        # sit at least ``i * tick_floor_steps * tick`` ABOVE the
        # inside-ask grid.
        if (
            i > 0
            and _tick_floor_steps > 0
            and _inside_ask_grid is not None
            and tick_size is not None
            and float(tick_size) > 0
        ):
            tick_floor_px = (
                _inside_ask_grid
                + float(i)
                * float(_tick_floor_steps)
                * float(tick_size)
            )
            if px < tick_floor_px - 1e-12:
                px = tick_floor_px
                if on_rung_floor_adjusted is not None:
                    try:
                        on_rung_floor_adjusted("ask")
                    except Exception:
                        pass
        if tick_size is not None and float(tick_size) > 0:
            grid_px = _tick_round_half_up(px, float(tick_size))
            if i > 0 and prev_grid_ask is not None and (
                # Strict-greater-than for ASK: lvl i must be at least
                # one tick ABOVE lvl i-1's grid.
                grid_px <= prev_grid_ask + 1e-12
            ):
                # Collision: outer rung doesn't move at least one tick
                # off the inner rung. Skip entirely.
                # v1.4.100 F2 — attribute the drop.
                if on_rung_dropped is not None:
                    try:
                        on_rung_dropped("grid_collision")
                    except Exception:
                        pass
                continue
            prev_grid_ask = grid_px
        asks.append(
            LadderRung(level_idx=i, side=Side.SELL, px=px, sz=sz)
        )

    return LadderDecision(
        bids=bids,
        asks=asks,
        requested_levels=int(cfg.num_levels_per_side),
        effective_levels_buy=len(bids),
        effective_levels_sell=len(asks),
        gate_caps=dict(gate_caps),
    )
