"""Per-cycle quote decomposition — operator visibility.

A frozen dataclass capturing every input that contributes to the
current cycle's quote: reservation shifts (mid → reservation),
half-spread components (BASE → effective), size multiplier chain,
and the resulting bid/ask prices + sizes.

Populated by ``app.quoting.compute_quote_decision`` at the end of
each cycle. The bot stashes the snapshot onto
``state.last_quote_breakdown`` and serialises it into
``state_current.json`` for the dashboard's Spread tab to render.

Per-cycle cost is ~5 µs (one dataclass construction with ~30
float assignments); serialisation only fires on snapshot poll
(10 s cadence), not per cycle. No measurable bot impact.

DESIGN NOTE — KEEP IN SYNC WITH ``compute_quote_decision``:
the breakdown is meant to be EXHAUSTIVE for the operator. Every
contribution-bearing variable in ``compute_quote_decision``
should be represented here. When new shift / bump / mult terms
are added to the quote computation, add the corresponding fields
here AND populate them at the same call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class QuoteBreakdownSnapshot:
    """Single quote cycle's full decomposition.

    All fields are plain JSON-serialisable (float / int / str /
    None / list-of-dict) so ``dataclasses.asdict`` round-trips
    via the dashboard's S3 heartbeat + state_current pipelines.
    """

    # Captured at end of compute_quote_decision; ISO-UTC.
    ts: str

    # ─── Reservation decomposition (mid → reservation) ───
    mid_price: float
    microprice: Optional[float]
    # "mid" or "microprice" — which one served as the
    # reservation reference before any shift was applied.
    reservation_reference: str
    reference_fair_price: Optional[float]
    reference_blend_alpha: float
    # ref_price after any cross-venue reference blend, before
    # any local shifts. This is the "anchor" the shifts move from.
    ref_price_anchor: float
    # Per-shift contributions in BPS from the anchor. Positive
    # value = shift moves reservation UP from anchor.
    # ``inventory_skew_bps`` is negative when position is long
    # (skew pushes reservation DOWN to favour reducing-side asks).
    inventory_skew_bps: float
    trend_drift_shift_bps: float
    ob_imbalance_shift_bps: float
    basis_deviation_shift_bps: float
    flow_score_shift_bps: float
    # Final reservation = ref_price_anchor + sum-of-shifts.
    reservation_price: float
    # Helpful summary for the dashboard: bp distance from mid.
    reservation_delta_from_mid_bps: float

    # ─── Half-spread stack ───────────────────────────────
    base_half_spread_bps: float
    # Per-component bumps (each is the bp contribution before
    # the final min/max clamp). Negative means tightening
    # (shouldn't happen in current logic; reserved for safety).
    vol_contribution_bps: float        # vol_multiplier × vol_bps
    toxicity_bump_bps: float           # toxicity_coeff × tox_score
    toxicity_score_used: float
    toxicity_coeff_used: float
    join_depth_autotune_overlay_bps: float
    toxicity_soft_trigger_bump_bps: float  # bumped only when soft_trigger
    toxicity_soft_trigger_active: bool
    vol_regime_bump_bps: float         # vol_spike_window overlay
    adaptive_widen_overlay_bps: float  # spread_floor_overlay_half_spread_bps
    # Latched-overlay context (read from state, denormalised here
    # so the dashboard doesn't need a second lookup).
    adaptive_widen_active: bool
    adaptive_widen_reason: Optional[str]
    adaptive_widen_seconds_remaining: float
    # Raw and final half-spread.
    raw_half_spread_bps: float         # post-bumps, pre-clamp
    min_half_spread_bps: float         # hard config floor
    max_half_spread_bps: float         # hard config ceiling
    economic_min_half_spread_bps: float  # the floor used (post overlays)
    economic_floor_kind: str           # "neutral" or "inventory"
    # Model-level target half-spread (output of compute_quote_decision,
    # PRE-engine market cap). 2026-05-12 codex-#3: this is the model
    # target, NOT necessarily what gets placed — when normal-MM mode
    # caps to market spread + buffer, the actual placed half-spread
    # is stamped separately as ``executable_half_spread_bps`` below.
    target_half_spread_bps: float
    clamp_winner: str                  # "raw" / "floor" / "ceiling"

    # ─── Size mult chain ─────────────────────────────────
    quote_notional_usd: float
    # Each row in the chain = the cumulative size_mult AFTER that
    # adjustment. The final mult is the last value.
    size_mult_after_toxicity: float
    size_mult_after_vol_regime: float
    size_mult_after_markout_scaler: float
    size_mult_after_basis_regime: float
    final_size_mult: float
    size_mult_floor: float             # min-notional self-heal floor
    effective_notional_usd: float

    # ─── Quote outputs ───────────────────────────────────
    quoted_bid_px: float
    quoted_ask_px: float
    quoted_bid_sz: float
    quoted_ask_sz: float
    best_bid: Optional[float]
    best_ask: Optional[float]

    # ─── Eligibility / gate state (denormalised) ─────────
    quote_eligibility: str
    quote_eligibility_reason: str
    active_sides: str

    # 2026-05-12 codex-#3: actual half-spread between the placed bid
    # and ask, computed from quoted_bid_px / quoted_ask_px / mid_price.
    # Can differ from ``target_half_spread_bps`` (which is the model
    # output of compute_quote_decision) when the quote-engine's
    # normal-MM market cap tightens the model target to "market half
    # + buffer". When the engine doesn't cap, this equals the target.
    # ``None`` when one side is suppressed (active_sides != BOTH).
    # Stamped in bot.py AFTER quote_engine.build_quotes runs — the
    # ``compute_quote_decision`` site only sees the pre-cap value.
    executable_half_spread_bps: Optional[float] = None
    # ─── Ladder (length-1 at N=1, multi-rung at N>1) ─────
    # Per-rung dicts: {level, side, px, sz}
    ladder_bids: list = field(default_factory=list)
    ladder_asks: list = field(default_factory=list)
    ladder_requested_levels: int = 1
    ladder_effective_levels_buy: int = 0
    ladder_effective_levels_sell: int = 0

    # v1.5.209 Phase 8D — OFI alpha (5th reservation-alpha) shift.
    # Default 0.0 when alpha is dormant (OFI_RESERVATION_ALPHA=0.0)
    # or when the accumulator is in warmup. Same units as the other
    # 4 alpha shifts: bp contribution to reservation from anchor.
    ofi_shift_bps: float = 0.0

    # v1.5.229 — Per-side microprice gate widening (bps the gate
    # contributed to that side's half-spread when it fired this
    # tick; 0 otherwise). Sourced from the local
    # ``SpreadComposition`` inside compute_quote_decision. Drives
    # the Phase 4A per-fill stamping + acceptance check
    # (``check_v1_5_204_microprice_widen_markout_within_noise``).
    #
    # Bug history: v1.5.204 shipped the stamping logic in
    # execution.py reading ``bk.spread_composition.microprice_
    # {bid,ask}_bps`` — but ``spread_composition`` was never added
    # to this dataclass, so the lookup silently returned None and
    # every fill ended up with NULL microprice columns. v1.5.229
    # adds the two fields directly here + populates them at the
    # construction site. Operator caught 2026-05-28 on the v1.5.225
    # Phase 5 snapshot (38 fills, all NULL).
    microprice_bid_widen_bps: float = 0.0
    microprice_ask_widen_bps: float = 0.0

    # v1.5.230 — reservation-shift clamp telemetry. True when the
    # ``MAX_RESERVATION_SHIFT_BPS_FROM_MID`` knob clamped this
    # tick's reservation shift. Operator dashboard / snapshot
    # acceptance check can count how often the clamp fires —
    # frequent firing indicates the bot is regime-bound (alphas
    # wanting bigger shifts than the cap permits).
    reservation_clamp_active: bool = False

    # v1.5.209 Phase 8B — Queue-aware sizing + inside-post telemetry.
    # All four fields default to "feature disabled" semantics:
    # ratio=None (no signal), mult=1.0 (no shrink), step_bps=0.0
    # (no inside-post). Operator dashboard reads these to see when
    # the feature is biting.
    queue_position_ratio_bid: Optional[float] = None
    queue_position_ratio_ask: Optional[float] = None
    queue_size_mult_bid: float = 1.0
    queue_size_mult_ask: float = 1.0
    queue_inside_post_step_bid_bps: float = 0.0
    queue_inside_post_step_ask_bps: float = 0.0

    def to_dict(self) -> dict:
        """JSON-serialisable dict for snapshot inclusion. Plain
        ``dataclasses.asdict`` works because every field is
        scalar / list-of-scalar / list-of-dict-of-scalar."""
        from dataclasses import asdict
        return asdict(self)
