"""Phase 4C.1 + 4C.2 — Expected-net-edge side suppression (v1.4.164).

Out-of-order delivery of the core "economics-first quoting" arc
(maturity-doc's single biggest gap). Today the bot computes
``expected_net_edge_bps_at_decision`` per order but only LOGS it — the
quoting code never reads it back, so even when the per-side expected
edge is negative the bot still places that side.

This module ships the FOUNDATIONAL pieces of Phase 4C:

* **4C.1** — compute per-side ``expected_net_edge_bps`` from the
  ``target_half_spread_bps`` produced by ``SpreadComposition`` plus
  the configured maker-rebate / typical-adverse-markout priors.
* **4C.2** — refuse to quote a side when its expected edge is more
  negative than ``MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE`` (a small
  negative number; default 0.0 = disabled). Hysteresis: a refused
  side does not re-arm until expected edge > MIN + a recovery margin
  for K consecutive ticks (prevents flapping).
* **4C.2.a** (v1.5.146) — dampen band. When adjusted expected edge
  falls in the band ``(refuse_threshold, dampen_max_bps]`` (i.e.
  marginally negative but above the refuse threshold), emit a
  per-side widening contribution instead of refusing. The bot keeps
  quoting that side with a wider spread, which protects against the
  binary nature of refusal in marginal economics regimes. Composes
  with the refuse band — refusal takes precedence when both bands
  would arm at once.

Pure-function helpers; state lives on ``BotState`` (the per-side
refused flag + the per-side recovery counter). Bot calls
``evaluate_per_side_expected_edge_suppression`` once per tick before
``compute_quote_decision``; the resulting two booleans are passed to
``compute_quote_decision`` and consumed via the same ``active_sides``
mechanism the other side-suppression gates use
(``at_touch_adverse_pause``, ``realised_edge_side_suppress``). The
dampen helper ``evaluate_per_side_dampen_band`` is called in the
same block and its non-zero output is folded into ``SpreadComposition``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.enums import Side


def compute_expected_net_edge_bps(
    *,
    target_half_spread_bps: Optional[float],
    maker_rebate_bps: float,
    typical_adverse_markout_bps: float,
    stale_risk_penalty_bps: float = 0.0,
) -> Optional[float]:
    """Return the per-side expected NET edge in bps.

    Formula (matches the existing place-time stamp in
    ``execution.py:7388-7416``, extended in v1.5.205 with the
    stale-risk penalty term — Phase 4C.4):

        expected = target_half_spread
                 + maker_rebate
                 − typical_adverse
                 − stale_risk_penalty       ← v1.5.205

    Returns ``None`` when ``target_half_spread_bps`` is missing or
    non-finite — the caller treats that as "no info, gate dormant".

    ``stale_risk_penalty_bps`` (default 0.0 = no penalty) is the
    bps-per-second-overdue-of-median deduction the caller should
    have pre-computed via :func:`compute_stale_risk_penalty_bps`.
    Default zero preserves pre-v1.5.205 behaviour byte-for-byte.
    """
    if target_half_spread_bps is None:
        return None
    try:
        ths = float(target_half_spread_bps)
    except (TypeError, ValueError):
        return None
    if ths != ths:  # NaN guard
        return None
    return (
        ths
        + float(maker_rebate_bps)
        - float(typical_adverse_markout_bps)
        - float(stale_risk_penalty_bps)
    )


def compute_stale_risk_penalty_bps(
    *,
    current_quote_age_seconds: Optional[float],
    p50_quote_age_seconds: Optional[float],
    coeff_bps_per_sec: float,
) -> float:
    """v1.5.205 Phase 4C.4 — stale-resting-quote adverse-selection penalty.

    Microstructure intuition: the longer the bot's quote has rested
    untouched at a given price, the more the book has had time to
    update against it. By the time it gets hit, the hit is more
    likely to be by aggressive flow that's already priced in the
    move the bot hasn't reacted to. Per-fill markout history (P99
    quote_age_at_fill = 2.4 s on TON v1.5.200-260528) shows the
    tail of slow fills lines up with the worst markouts.

    The penalty subtracts ``(age − P50) × coeff`` bps from the
    side's expected net edge, where ``P50`` is the rolling median of
    decision-time quote ages over the recent window. When age is at
    or below the median, no penalty fires — only the upper tail of
    the distribution is taxed.

    Returns ``0.0`` (silent no-op) when:

    * ``coeff_bps_per_sec <= 0.0`` — feature disabled.
    * ``current_quote_age_seconds`` is None — no resting quote on
      this side (fresh place; no age to measure).
    * ``p50_quote_age_seconds`` is None — rolling window hasn't
      collected enough samples yet (warmup).
    * ``current_quote_age_seconds <= p50_quote_age_seconds`` —
      side is at or below median; no penalty.

    Returns a strictly positive bps deduction otherwise. The caller
    passes this directly into ``compute_expected_net_edge_bps``'s
    ``stale_risk_penalty_bps`` parameter — no further scaling.
    """
    if coeff_bps_per_sec <= 0.0:
        return 0.0
    if current_quote_age_seconds is None or p50_quote_age_seconds is None:
        return 0.0
    age = float(current_quote_age_seconds)
    p50 = float(p50_quote_age_seconds)
    if age <= p50:
        return 0.0
    return (age - p50) * float(coeff_bps_per_sec)


@dataclass(frozen=True)
class ExpectedEdgeSideSuppressResult:
    """One per-side evaluation result. ``refused`` is the flag the
    bot passes to ``compute_quote_decision`` (and which composes via
    ``active_sides``). ``new_recovery_ticks`` is the updated
    hysteresis counter that the caller writes back to BotState."""

    refused: bool
    new_recovery_ticks: int
    expected_edge_bps: Optional[float]
    transition: str
    """One of ``"none"``, ``"armed"`` (transitioned to refused this
    tick), ``"cleared"`` (transitioned out of refused this tick)."""


def evaluate_per_side_expected_edge_suppression(
    *,
    target_half_spread_bps: Optional[float],
    currently_refused: bool,
    recovery_ticks: int,
    min_expected_edge_bps: float,
    hysteresis_ticks: int,
    recovery_margin_bps: float,
    maker_rebate_bps: float,
    typical_adverse_markout_bps: float,
    confidence_multiplier: float = 1.0,
    stale_risk_penalty_bps: float = 0.0,
) -> ExpectedEdgeSideSuppressResult:
    """Per-side per-tick decision.

    Logic:
      * Compute current per-side expected edge.
      * v1.5.41 Phase 4C.3: scale the edge by ``confidence_multiplier``
        before comparing to the threshold. Multiplier > 1 (recent
        realised edge above break-even by stdev units) amplifies the
        edge → refusal less likely. Multiplier < 1 (recent edge
        adverse) discounts the edge → refusal sooner. Default 1.0
        (no scaling) preserves pre-4C.3 behaviour.
      * **If currently refused:**
          - If adjusted_edge > min + recovery_margin: increment the
            recovery counter.
          - Else (edge dropped back): reset the counter to 0.
          - If counter reached ``hysteresis_ticks``: clear refusal.
      * **If currently not refused:**
          - If adjusted_edge < min_expected_edge: arm refusal.
          - Else: leave alone.

    With ``min_expected_edge_bps=0.0`` (default) the gate is dormant
    — no side ever gets refused regardless of edge — preserving
    pre-v1.4.164 behaviour.

    When ``target_half_spread_bps`` is missing (e.g., the composition
    isn't ready yet or a degenerate market state), the gate stays
    dormant for THIS tick (no state change).

    The result's ``expected_edge_bps`` is the ADJUSTED edge (post-
    multiplier) so the bot's telemetry reflects the value the gate
    actually compared against. Callers that want the raw value can
    recompute via ``compute_expected_net_edge_bps`` directly.
    """
    edge = compute_expected_net_edge_bps(
        target_half_spread_bps=target_half_spread_bps,
        maker_rebate_bps=maker_rebate_bps,
        typical_adverse_markout_bps=typical_adverse_markout_bps,
        stale_risk_penalty_bps=stale_risk_penalty_bps,
    )
    # Apply Phase 4C.3 confidence multiplier. Default 1.0 is a no-op.
    if edge is not None:
        try:
            edge = float(edge) * float(confidence_multiplier)
        except (TypeError, ValueError):
            pass

    # Feature disabled (default) or no edge signal yet: pass through.
    if min_expected_edge_bps >= 0.0 or edge is None:
        return ExpectedEdgeSideSuppressResult(
            refused=currently_refused,
            new_recovery_ticks=recovery_ticks,
            expected_edge_bps=edge,
            transition="none",
        )

    if currently_refused:
        recovery_threshold = float(min_expected_edge_bps) + float(
            recovery_margin_bps
        )
        if edge > recovery_threshold:
            new_ticks = int(recovery_ticks) + 1
            if new_ticks >= int(hysteresis_ticks):
                return ExpectedEdgeSideSuppressResult(
                    refused=False,
                    new_recovery_ticks=0,
                    expected_edge_bps=edge,
                    transition="cleared",
                )
            return ExpectedEdgeSideSuppressResult(
                refused=True,
                new_recovery_ticks=new_ticks,
                expected_edge_bps=edge,
                transition="none",
            )
        # Recovery interrupted — reset counter, stay refused.
        return ExpectedEdgeSideSuppressResult(
            refused=True,
            new_recovery_ticks=0,
            expected_edge_bps=edge,
            transition="none",
        )

    # Currently not refused. Arm if edge dropped below threshold.
    if edge < float(min_expected_edge_bps):
        return ExpectedEdgeSideSuppressResult(
            refused=True,
            new_recovery_ticks=0,
            expected_edge_bps=edge,
            transition="armed",
        )
    return ExpectedEdgeSideSuppressResult(
        refused=False,
        new_recovery_ticks=0,
        expected_edge_bps=edge,
        transition="none",
    )


# ---------------------------------------------------------------------------
# Phase 4C.2.a (v1.5.146) — dampen band
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedEdgeDampenResult:
    """One per-side evaluation of the dampen band. ``widen_bps`` is
    the additive contribution the caller folds into the side's
    ``SpreadComposition`` (always non-negative). ``armed`` is True
    iff the band fired this tick (i.e. ``widen_bps > 0`` AND the
    inputs were in the dampen-band range)."""

    widen_bps: float
    armed: bool
    expected_edge_bps: Optional[float]


def evaluate_per_side_dampen_band(
    *,
    target_half_spread_bps: Optional[float],
    refuse_threshold_bps: float,
    dampen_max_bps: float,
    dampen_widen_bps: float,
    maker_rebate_bps: float,
    typical_adverse_markout_bps: float,
    confidence_multiplier: float = 1.0,
    already_refused: bool = False,
    stale_risk_penalty_bps: float = 0.0,
) -> ExpectedEdgeDampenResult:
    """Phase 4C.2.a — return the per-side dampen widening (bps) when
    the adjusted expected edge falls in the band
    ``(refuse_threshold_bps, dampen_max_bps]``.

    Inputs match :func:`evaluate_per_side_expected_edge_suppression`
    so the caller (bot.py) can reuse the same target half-spread +
    rebate + adverse priors + confidence multiplier without
    recomputation. The two evaluators are deliberately independent
    in their public surface — composition (refuse takes precedence)
    is handled via the ``already_refused`` kwarg.

    Returns zero widening when:

    * ``dampen_widen_bps <= 0`` — feature dormant by config.
    * ``refuse_threshold_bps >= 0`` — refuse band itself is disabled
      (default config). The dampen band is meaningless without a
      refuse threshold to anchor against.
    * ``dampen_max_bps <= refuse_threshold_bps`` — pathological
      operator config (band would be empty or inverted); treat as
      disabled rather than firing on every tick.
    * ``target_half_spread_bps`` is None — no edge signal this tick.
    * ``already_refused`` is True — the refuse band's side
      suppression takes precedence; no point adding widening to a
      side that won't quote.
    * Adjusted edge ≤ ``refuse_threshold_bps`` — would refuse, not
      dampen (the refuse evaluator handles this case).
    * Adjusted edge > ``dampen_max_bps`` — above the dampen band;
      economics are healthy enough that no widening is warranted.

    The Phase 4C.3 confidence multiplier is applied the same way as
    in the refuse evaluator — scale the raw edge by the multiplier
    before band comparison. Default 1.0 preserves the raw edge.

    Hysteresis is intentionally **not** modeled here because dampening
    is a continuous additive widening, not a binary on/off side
    suppression — per-tick flapping of the dampen contribution has
    no operator-visible cost (the side keeps quoting; the spread
    flexes a bit). If a future calibration shows otherwise, hysteresis
    can be added as a stateful wrapper without changing this helper.
    """
    # Disable conditions that bypass all math.
    if dampen_widen_bps <= 0.0:
        return ExpectedEdgeDampenResult(
            widen_bps=0.0, armed=False, expected_edge_bps=None
        )
    if refuse_threshold_bps >= 0.0:
        return ExpectedEdgeDampenResult(
            widen_bps=0.0, armed=False, expected_edge_bps=None
        )
    if dampen_max_bps <= refuse_threshold_bps:
        return ExpectedEdgeDampenResult(
            widen_bps=0.0, armed=False, expected_edge_bps=None
        )
    if already_refused:
        return ExpectedEdgeDampenResult(
            widen_bps=0.0, armed=False, expected_edge_bps=None
        )

    edge = compute_expected_net_edge_bps(
        target_half_spread_bps=target_half_spread_bps,
        maker_rebate_bps=maker_rebate_bps,
        typical_adverse_markout_bps=typical_adverse_markout_bps,
        stale_risk_penalty_bps=stale_risk_penalty_bps,
    )
    if edge is None:
        return ExpectedEdgeDampenResult(
            widen_bps=0.0, armed=False, expected_edge_bps=None
        )
    try:
        adjusted = float(edge) * float(confidence_multiplier)
    except (TypeError, ValueError):
        adjusted = float(edge)

    # Strictly above the refuse threshold (the refuse evaluator owns
    # the boundary at and below) and at-or-below the dampen ceiling.
    if adjusted > float(refuse_threshold_bps) and adjusted <= float(
        dampen_max_bps
    ):
        return ExpectedEdgeDampenResult(
            widen_bps=float(dampen_widen_bps),
            armed=True,
            expected_edge_bps=adjusted,
        )
    return ExpectedEdgeDampenResult(
        widen_bps=0.0, armed=False, expected_edge_bps=adjusted
    )
