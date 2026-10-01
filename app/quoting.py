"""Strategy-level quoting math: Avellaneda-Stoikov reservation + inventory + toxicity.

**Role in the one-model pipeline**

This module produces the *strategy intent*: the prices and sizes the strategy
*would like* to place if the venue had infinite precision. Nothing here knows
about ``price_tick``, ``size_step``, or ``min_notional_usd`` — those are
*venue constraints*, owned exclusively by :class:`app.quote_engine.QuoteEngine`.

A :class:`~app.models.QuoteDecision` from this module is therefore *not*
placement-ready. It flows into :meth:`QuoteEngine.build_quotes` which applies
rounding, post-only clamping, economic floor, and min-notional self-heal in a
single model pass — see ``app/quote_engine.py`` module docstring for the
invariant.

**Do not add a venue-constraint check here.** Anything downstream should not
need to re-negotiate what this module produced. If a new constraint comes up,
push it into the QuoteEngine's single pass, not into a new intermediate
validation layer — that was the layered-disagreement architecture that kept
the bot silently rejecting its own intent (``tmp/snap_20260417_195747``).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from app.config import Settings
from app.enums import ActiveSides, QuoteEligibility, Side
from app.models import QuoteDecision, ToxicitySnapshot
from app.quote_aging import clamp_buy_post_only, clamp_sell_post_only
from app.utils.math import clip
from app.clock import Clock, SystemClock
from app.utils.time import utc_now


# ============================================================
# Spread composition (gate-to-widening Phase 1, v1.4.8+)
# ============================================================
#
# The bot's effective half-spread per side is the SUM of contributions
# from multiple signals, capped at MAX_HALF_SPREAD_BPS. Replaces the
# pre-v1.5 binary gates that clamped eligibility (HOLD_ALL / one-sided)
# — see plans/gate-to-widening.md.
#
# Each former regime-response gate publishes ``widening_bps()`` →
# tuple of ``(bid_bps, ask_bps)``. Symmetric gates set both equal;
# asymmetric gates (microprice, momentum, freshness one-sided,
# recovery_cooldown) can widen only one side. Zero contribution when
# the signal is clear.
#
# Initial coefficients in Phase 1 are GATE-EQUIVALENT — when a gate's
# underlying signal triggers, widening_bps returns
# ``MAX_HALF_SPREAD_BPS`` so the effective half-spread caps out.
# Functionally identical to today's HOLD_ALL (the bot quotes too far
# from mid to fill in normal trading); difference is the orders are
# PRESENT at the venue at wide prices instead of absent.
#
# Phase 2 of the plan iterates the coefficients DOWN per gate to find
# operating points where the bot stays in the market under the gate's
# signal rather than going effectively dark.


@dataclass(frozen=True, slots=True)
class SpreadComposition:
    """Per-side half-spread contributions from each signal source.

    All fields are bps (a `vol_trend_bid_bps=12.4` means vol_trend is
    pushing the bid-side half-spread out by 12.4 bps). Always non-
    negative — widening contributors never tighten.

    The "always-on floor" contributors (econ_floor, toxicity_bump,
    adverse_overlay) are present every cycle even when no gate fires.
    The 7 regime contributors are zero when their underlying signal
    is clear.

    Per-side asymmetry: most contributors set bid == ask (symmetric).
    Asymmetric ones (microprice, momentum, freshness_one_sided,
    recovery_cooldown) widen only the side their signal threatens.
    """

    # --- Always-on floor contributors ------------------------------
    econ_floor_bps: float = 0.0
    toxicity_bps: float = 0.0
    adverse_overlay_bps: float = 0.0

    # --- Per-gate regime-response contributors ---------------------
    vol_trend_bid_bps: float = 0.0
    vol_trend_ask_bps: float = 0.0
    momentum_bid_bps: float = 0.0
    momentum_ask_bps: float = 0.0
    post_swing_bid_bps: float = 0.0
    post_swing_ask_bps: float = 0.0
    microprice_bid_bps: float = 0.0
    microprice_ask_bps: float = 0.0
    # M8 Candidate B — recorder-fed microprice-deviation z-score widen.
    # Additive sibling of the in-process ``microprice_*`` pair above,
    # sourced from the tape runtime feed's ``microprice_dev_z_24h``
    # instead of the in-process OB-imbalance EWMA. Zero (byte-identical
    # to pre-M8) whenever ``REGIME_USE_RUNTIME_RECORDER_FEED`` is off or
    # the feed field is dark / stale (the caller passes ``None`` then).
    microprice_runtime_bid_bps: float = 0.0
    microprice_runtime_ask_bps: float = 0.0
    basis_bid_bps: float = 0.0
    basis_ask_bps: float = 0.0
    freshness_bid_bps: float = 0.0
    freshness_ask_bps: float = 0.0
    recovery_cooldown_bid_bps: float = 0.0
    recovery_cooldown_ask_bps: float = 0.0
    # v1.4.102 — slow_trend gate. Catches sustained directional drift
    # at a longer window than ``long_drift_eligibility`` (15 min vs
    # 5 min) with a lower threshold (25 bp vs 50 bp). Asymmetric:
    # widens only the side that would add to a bad-trend position.
    slow_trend_bid_bps: float = 0.0
    slow_trend_ask_bps: float = 0.0
    # v1.4.106 Phase 1A — inventory_drift_gate. Position-aware short-
    # window (10s/30s) defence: fires only when bot inventory is
    # non-trivial AND the short-window drift is anti-aligned with the
    # held position. Asymmetric: widens only the side that would add
    # to the bad-side position.
    inventory_drift_bid_bps: float = 0.0
    inventory_drift_ask_bps: float = 0.0
    # v1.5.146 Phase 4C.2.a — dampen band on the economics gate.
    # Fires per-side when the adjusted expected net edge falls in the
    # dampen band `(refuse_threshold, dampen_max_bps]` (i.e. marginally
    # negative but above the refuse threshold). The composition's
    # other contributors are unchanged; this widens the bot's quote
    # on that side so it keeps participating with a wider spread
    # instead of being refused outright. Per-side asymmetric: only
    # the side with marginal economics gets the widening.
    # 4C.2.c — the SpreadComposition contributor itself serves as the
    # audit-trail entry (visible in `quote_decisions` rows + the
    # snapshot's spread-composition block).
    negative_expectancy_dampen_bid_bps: float = 0.0
    negative_expectancy_dampen_ask_bps: float = 0.0

    # --- Derived flags -----------------------------------------------
    # ``capped_at_max_*`` is True when the sum hit the
    # MAX_HALF_SPREAD_BPS cap; surfaces "the bot is effectively
    # dark on this side" on the operator dashboard.
    capped_at_max_bid: bool = False
    capped_at_max_ask: bool = False

    def total_bid_bps_uncapped(self) -> float:
        """Sum of all contributions to the bid-side half-spread,
        BEFORE clamping at MAX. Useful for telemetry / debugging —
        the operator can see how aggressive the widening would be
        absent the cap."""
        return (
            self.econ_floor_bps
            + self.toxicity_bps
            + self.adverse_overlay_bps
            + self.vol_trend_bid_bps
            + self.momentum_bid_bps
            + self.post_swing_bid_bps
            + self.microprice_bid_bps
            + self.microprice_runtime_bid_bps
            + self.basis_bid_bps
            + self.freshness_bid_bps
            + self.recovery_cooldown_bid_bps
            + self.slow_trend_bid_bps
            + self.inventory_drift_bid_bps
            + self.negative_expectancy_dampen_bid_bps
        )

    def total_ask_bps_uncapped(self) -> float:
        return (
            self.econ_floor_bps
            + self.toxicity_bps
            + self.adverse_overlay_bps
            + self.vol_trend_ask_bps
            + self.momentum_ask_bps
            + self.post_swing_ask_bps
            + self.microprice_ask_bps
            + self.microprice_runtime_ask_bps
            + self.basis_ask_bps
            + self.freshness_ask_bps
            + self.recovery_cooldown_ask_bps
            + self.slow_trend_ask_bps
            + self.inventory_drift_ask_bps
            + self.negative_expectancy_dampen_ask_bps
        )

    def effective_half_spread_bid_bps(self, max_bps: float) -> float:
        """Effective bid-side half-spread = sum of contributions,
        capped at ``max_bps`` (from ``MAX_HALF_SPREAD_BPS``).
        Floored at ``econ_floor_bps`` so the bot never quotes inside
        the rebate-economic minimum regardless of other contributors."""
        total = self.total_bid_bps_uncapped()
        return min(float(max_bps), max(self.econ_floor_bps, total))

    def effective_half_spread_ask_bps(self, max_bps: float) -> float:
        total = self.total_ask_bps_uncapped()
        return min(float(max_bps), max(self.econ_floor_bps, total))

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe payload for storage / dashboard publisher.
        Schema column ``spread_composition`` in ``quote_decisions``."""
        return {
            "econ_floor_bps": float(self.econ_floor_bps),
            "toxicity_bps": float(self.toxicity_bps),
            "adverse_overlay_bps": float(self.adverse_overlay_bps),
            "vol_trend_bid_bps": float(self.vol_trend_bid_bps),
            "vol_trend_ask_bps": float(self.vol_trend_ask_bps),
            "momentum_bid_bps": float(self.momentum_bid_bps),
            "momentum_ask_bps": float(self.momentum_ask_bps),
            "post_swing_bid_bps": float(self.post_swing_bid_bps),
            "post_swing_ask_bps": float(self.post_swing_ask_bps),
            "microprice_bid_bps": float(self.microprice_bid_bps),
            "microprice_ask_bps": float(self.microprice_ask_bps),
            "microprice_runtime_bid_bps": float(self.microprice_runtime_bid_bps),
            "microprice_runtime_ask_bps": float(self.microprice_runtime_ask_bps),
            "basis_bid_bps": float(self.basis_bid_bps),
            "basis_ask_bps": float(self.basis_ask_bps),
            "freshness_bid_bps": float(self.freshness_bid_bps),
            "freshness_ask_bps": float(self.freshness_ask_bps),
            "recovery_cooldown_bid_bps": float(self.recovery_cooldown_bid_bps),
            "recovery_cooldown_ask_bps": float(self.recovery_cooldown_ask_bps),
            "slow_trend_bid_bps": float(self.slow_trend_bid_bps),
            "slow_trend_ask_bps": float(self.slow_trend_ask_bps),
            "inventory_drift_bid_bps": float(self.inventory_drift_bid_bps),
            "inventory_drift_ask_bps": float(self.inventory_drift_ask_bps),
            "negative_expectancy_dampen_bid_bps": float(
                self.negative_expectancy_dampen_bid_bps
            ),
            "negative_expectancy_dampen_ask_bps": float(
                self.negative_expectancy_dampen_ask_bps
            ),
            "capped_at_max_bid": bool(self.capped_at_max_bid),
            "capped_at_max_ask": bool(self.capped_at_max_ask),
        }

    def with_caps(self, *, max_bps: float) -> "SpreadComposition":
        """Return a new SpreadComposition with ``capped_at_max_*``
        flags set based on whether the uncapped total reached
        ``max_bps``. Builder convenience — callers set the
        contribution fields, then call this to finalise the cap
        flags before the dataclass is returned to consumers."""
        eps = 1e-9
        max_f = float(max_bps)
        return replace(
            self,
            capped_at_max_bid=self.total_bid_bps_uncapped() + eps >= max_f,
            capped_at_max_ask=self.total_ask_bps_uncapped() + eps >= max_f,
        )


def build_spread_composition(
    *,
    settings: Settings,
    toxicity: ToxicitySnapshot,
    active_sides: ActiveSides,
    spread_floor_overlay_half_spread_bps: float,
    # Gate states / signals — passed by the caller (bot.py) which
    # already evaluates these for its own logic.
    vol_trend_state: Any,
    post_swing_state: Any,
    ob_imbalance_ewma: float | None,
    basis_regime_last_ic: float | None,
    basis_regime_pair_count: int,
    position_qty: float,
    effective_abs_cap: float,
    drift_bps: float | None,
    raw_eligibility: Any,
    effective_eligibility: Any,
    now_mono: float,
    price_tick: float | None = None,
    fair_value: float | None = None,
    # v1.4.102 — slow_trend_gate inputs. ``mid_samples_long`` is the
    # multi-minute mid-history deque (same source the
    # ``_long_drift_eligibility`` gate uses). ``mid_now`` is the
    # current mid price for the anchored-median computation. Both
    # default to None for backward compatibility — when missing, the
    # slow_trend contribution stays at zero.
    mid_samples_long: Any = None,
    mid_now: float | None = None,
    # v1.4.106 Phase 1A — inventory_drift_gate inputs. Short-window
    # drifts (10s/30s) computed from the same multi-second mid-history
    # deque slow_trend uses; bot.py computes them once via
    # ``inventory_drift_gate.compute_short_window_drifts``. Both
    # default to None for backward compatibility — when missing, the
    # inventory_drift contribution stays at zero.
    drift_bps_10s: float | None = None,
    drift_bps_30s: float | None = None,
    # M8 Candidate B — recorder microprice-deviation z-score. The caller
    # (bot.py) reads the tape runtime feed once per tick and passes the
    # live ``microprice_dev_z_24h`` here, or ``None`` when the feed is
    # off / dark / stale / coverage-invalid. ``None`` ⇒ zero contribution
    # (byte-identical to pre-M8). Default None keeps every existing caller
    # and test unchanged.
    microprice_dev_z_runtime: float | None = None,
) -> SpreadComposition:
    """Gather all contributors and build the per-side
    :class:`SpreadComposition` for the current tick.

    Phase 1 builder. Each contributor is queried via its
    ``widening_bps()`` (regime-response gates) or computed in-line
    (always-on floor contributors). Final composition is clamped
    via :meth:`SpreadComposition.with_caps` so ``capped_at_max_*``
    flags reflect the actual operator-facing cap.

    Per-gate contributions at PHASE 1 initial coefficients are
    gate-equivalent: when a gate's signal would have fired binary
    suppression today, its widening returns ``MAX_HALF_SPREAD_BPS``
    on the corresponding side(s). The composition's effective
    half-spread therefore caps out at the same magnitude as today's
    HOLD_ALL would have produced — functionally equivalent.

    Phase 2 of the plan iterates the per-gate coefficients DOWN
    based on observed markouts. The builder signature is stable
    across that iteration; only the gate modules' internals change.
    """
    # Lazy imports to avoid circular-dependency hairballs. The gate
    # modules don't import quoting, but quoting imports
    # SpreadComposition consumers from anywhere, so keep this loose.
    from app import (
        basis_regime_gate,
        inventory_drift_gate,
        microprice_gate,
        momentum_gate,
        post_swing_gate,
        slow_trend_gate,
        vol_trend_gate,
    )
    from app.quote_eligibility import (
        freshness_one_sided_widening_bps,
        recovery_cooldown_widening_bps,
    )

    max_bps = float(settings.max_half_spread_bps)

    # --- Always-on floor contributors ------------------------------
    # Econ floor: rebate-economic minimum (existing function).
    econ_floor = compute_effective_min_half_spread_bps(
        settings,
        active_sides,
        # Pass zero toxicity / zero overlay here so the floor is
        # only the econ minimum; toxicity and overlay are SEPARATE
        # contributors in the composition.
        0.0,
        spread_floor_overlay_half_spread_bps=0.0,
        price_tick=price_tick,
        fair_value=fair_value,
    )

    # Toxicity bump: continuous, existing.
    toxicity_score = float(getattr(toxicity, "score", 0.0) or 0.0)
    toxicity_bump = max(0.0, toxicity_score) * float(
        settings.economic_toxicity_score_half_spread_bps
    )

    # Adverse overlay: when armed by recent fill-quality signals
    # (the operator-tunable additive bps overlay).
    overlay = max(0.0, float(spread_floor_overlay_half_spread_bps))

    # --- Regime-response gate contributions ------------------------
    # Each gate reads its per-gate widening coefficient from settings
    # (Phase 2). Default sentinel ``-1.0`` falls back to
    # ``max_half_spread_bps`` (gate-equivalent magnitude), preserving
    # the cutover behaviour. Operators iterate one knob at a time
    # to find the operating spread per gate.
    vt_bid, vt_ask = vol_trend_gate.widening_bps(
        vol_trend_state, now_mono=now_mono,
        max_half_spread_bps=max_bps,
        widen_bps=float(settings.vol_trend_widen_bps),
    ) if bool(settings.vol_trend_gate_enabled) else (0.0, 0.0)

    ps_bid, ps_ask = post_swing_gate.widening_bps(
        post_swing_state, now_mono=now_mono,
        max_half_spread_bps=max_bps,
        widen_bps=float(settings.post_swing_widen_bps),
    ) if bool(settings.post_swing_enabled) else (0.0, 0.0)

    # v1.4.47 — BUG-026 structural: pass the raw config value
    # through to the gate. The gate itself interprets the sign:
    #   < 0  → direction-based structural suppression (default)
    #   == 0 → disabled
    #   > 0  → legacy threshold mode (back-compat)
    # See microprice_gate.widening_bps docstring for the full rationale.
    mp_bid, mp_ask = microprice_gate.widening_bps(
        ob_imbalance_ewma=ob_imbalance_ewma,
        threshold=float(settings.microprice_gate_ob_imbalance_threshold),
        max_half_spread_bps=max_bps,
        widen_bps=float(settings.microprice_widen_bps),
        enabled=bool(settings.microprice_gate_enabled),
        position_qty=float(position_qty),
        effective_abs_cap=float(effective_abs_cap),
        reducing_side_widen_suppress_pct=float(
            getattr(
                settings,
                "microprice_gate_reducing_side_widen_suppress_pct",
                -1.0,
            )
        ),
    )

    # M8 Candidate B — recorder microprice-deviation z-score widen.
    # Additive sibling of the OB-imbalance microprice gate above; fires
    # only when the caller passed a live z (feed on + field lit + fresh).
    # Reuses the same inventory-direction suppression sentinel as its
    # sibling for consistent behaviour. ``widen_bps`` is a real per-side
    # amount (default 4.0), not the full-cap sentinel.
    mpz_bid, mpz_ask = microprice_gate.microprice_z_widening_bps(
        microprice_dev_z=microprice_dev_z_runtime,
        z_threshold=float(settings.microprice_z_widen_threshold),
        widen_bps=float(settings.microprice_z_widen_bps),
        max_half_spread_bps=max_bps,
        enabled=bool(
            getattr(settings, "regime_use_runtime_recorder_feed", False)
        ),
        position_qty=float(position_qty),
        effective_abs_cap=float(effective_abs_cap),
        reducing_side_widen_suppress_pct=float(
            getattr(
                settings,
                "microprice_gate_reducing_side_widen_suppress_pct",
                -1.0,
            )
        ),
    )

    br_bid, br_ask = basis_regime_gate.widening_bps(
        last_ic=basis_regime_last_ic,
        pair_count=basis_regime_pair_count,
        ic_min_quote_threshold=float(
            settings.basis_regime_gate_ic_min_quote
        ),
        min_pair_samples=int(
            settings.basis_deviation_regime_min_pair_samples
        ),
        max_half_spread_bps=max_bps,
        widen_bps=float(settings.basis_regime_widen_bps),
        enabled=bool(settings.basis_regime_gate_enabled),
    )

    mom_bid, mom_ask = momentum_gate.widening_bps(
        position_qty=position_qty,
        effective_abs_cap=effective_abs_cap,
        drift_bps=drift_bps,
        drift_threshold_bps=float(
            settings.momentum_gate_drift_threshold_bps
        ),
        inventory_pct_threshold=float(
            settings.momentum_gate_inventory_pct
        ),
        max_half_spread_bps=max_bps,
        widen_bps=float(settings.momentum_widen_bps),
        enabled=bool(settings.momentum_gate_enabled),
    )

    fresh_bid, fresh_ask = freshness_one_sided_widening_bps(
        raw_eligibility,
        max_half_spread_bps=max_bps,
        widen_bps=float(settings.freshness_one_sided_widen_bps),
    )

    rec_bid, rec_ask = recovery_cooldown_widening_bps(
        effective_eligibility,
        max_half_spread_bps=max_bps,
        widen_bps=float(settings.recovery_cooldown_widen_bps),
    )

    # v1.4.102 — slow_trend gate. Live from day one with a NON-sentinel
    # widen_bps default. Only fires when both inputs are present (the
    # mid-samples deque and the current mid). Empty inputs degrade to
    # zero widening rather than raising.
    if (
        bool(settings.slow_trend_gate_enabled)
        and mid_samples_long is not None
        and mid_now is not None
        and mid_now > 0
    ):
        st_bid, st_ask = slow_trend_gate.widening_bps(
            samples_long=mid_samples_long,
            now_mono=now_mono,
            mid_now=float(mid_now),
            max_half_spread_bps=max_bps,
            enabled=True,
            window_seconds=float(settings.slow_trend_window_seconds),
            threshold_bps=float(settings.slow_trend_threshold_bps),
            min_samples=int(settings.slow_trend_min_samples),
            anchor_fraction=float(settings.slow_trend_anchor_fraction),
            widen_bps=float(settings.slow_trend_widen_bps),
        )
    else:
        st_bid, st_ask = (0.0, 0.0)

    # v1.4.106 Phase 1A — inventory_drift_gate. Position-aware, short-
    # window (10s/30s) anti-aligned drift defence. Live from day one
    # with a non-sentinel widen default (15 bp — wider than
    # slow_trend's 10 bp because the trigger is more specific). Gate
    # stays dormant when either drift input is missing.
    if bool(settings.inventory_drift_gate_enabled):
        id_bid, id_ask = inventory_drift_gate.widening_bps(
            position_qty=float(position_qty),
            effective_abs_cap=float(effective_abs_cap),
            drift_bps_10s=drift_bps_10s,
            drift_bps_30s=drift_bps_30s,
            inventory_pct_threshold=float(
                settings.inventory_drift_inventory_pct_threshold
            ),
            drift_threshold_bps_10s=float(
                settings.inventory_drift_threshold_bps_10s
            ),
            drift_threshold_bps_30s=float(
                settings.inventory_drift_threshold_bps_30s
            ),
            enabled=True,
            max_half_spread_bps=max_bps,
            widen_bps=float(settings.inventory_drift_widen_bps),
        )
    else:
        id_bid, id_ask = (0.0, 0.0)

    composition = SpreadComposition(
        econ_floor_bps=float(econ_floor),
        toxicity_bps=float(toxicity_bump),
        adverse_overlay_bps=float(overlay),
        vol_trend_bid_bps=float(vt_bid),
        vol_trend_ask_bps=float(vt_ask),
        momentum_bid_bps=float(mom_bid),
        momentum_ask_bps=float(mom_ask),
        post_swing_bid_bps=float(ps_bid),
        post_swing_ask_bps=float(ps_ask),
        microprice_bid_bps=float(mp_bid),
        microprice_ask_bps=float(mp_ask),
        microprice_runtime_bid_bps=float(mpz_bid),
        microprice_runtime_ask_bps=float(mpz_ask),
        basis_bid_bps=float(br_bid),
        basis_ask_bps=float(br_ask),
        freshness_bid_bps=float(fresh_bid),
        freshness_ask_bps=float(fresh_ask),
        recovery_cooldown_bid_bps=float(rec_bid),
        recovery_cooldown_ask_bps=float(rec_ask),
        slow_trend_bid_bps=float(st_bid),
        slow_trend_ask_bps=float(st_ask),
        inventory_drift_bid_bps=float(id_bid),
        inventory_drift_ask_bps=float(id_ask),
    )
    return composition.with_caps(max_bps=max_bps)


def adverse_spread_widen_arm(settings: Settings, toxicity: ToxicitySnapshot) -> bool:
    """
    True when recent fill-quality signals should arm (or extend) the adaptive spread widen cooldown.

    Uses the same thresholds as :class:`ToxicityEngine` for consistency (no duplicate tuning).
    """
    if float(settings.adaptive_spread_adverse_overlay_half_spread_bps) <= 0:
        return False
    if toxicity.hard_trigger or toxicity.soft_trigger:
        return True
    if toxicity.adverse_uses_delayed_markouts and toxicity.avg_adverse_markout_bps + 1e-12 <= -float(
        settings.toxicity_markout_soft_bps
    ):
        return True
    if toxicity.one_sided_fill_ratio + 1e-12 >= float(settings.toxicity_one_sided_fill_ratio):
        return True
    return False


def adaptive_spread_widen_signal_cleared(
    settings: Settings,
    reason: str,
    toxicity: ToxicitySnapshot,
    *,
    quote_quality_signal: bool,
    slow_trend_signal: bool,
) -> bool:
    """v1.4.155 Phase 2K.5 — per-reason favorable-exit predicate for
    the adaptive_spread_widen cooldown.

    Returns True when the signal that armed the overlay is no longer
    firing. Each of the six trigger reasons has its own predicate:

    * ``toxicity_hard``    — ``not toxicity.hard_trigger``.
      Toxicity engine recomputes ``hard_trigger`` per tick from a
      rolling adverse-markout window; clears as soon as the latest
      window's mean drops back above the hard threshold.
    * ``toxicity_soft``    — ``not toxicity.soft_trigger``.
      Same shape: ``soft_trigger`` is true while
      ``score >= 0.45``; clears when ``score`` drops below it.
    * ``markout_adverse``  — ``not (adverse_uses_delayed_markouts AND
      avg_adverse_markout_bps <= -toxicity_markout_soft_bps)``. The
      composite predicate already invertible per-tick.
    * ``one_sided_ratio``  — ``one_sided_fill_ratio < threshold``.
      Inversion of the arming check.
    * ``quote_quality``    — ``not quote_quality_signal`` (caller
      passes the result of ``quote_quality.spread_widen_signal()``).
    * ``slow_trend``       — ``not slow_trend_signal`` (caller passes
      whether ``evaluate_slow_trend_gate()`` currently fires).
    * Unknown / legacy reason → False (defaults to ceiling-only
      clearing for safety).

    Note: per-reason predicates do NOT include a built-in dwell
    timer. The caller wraps this in a shared dwell timer
    (``adaptive_spread_widen_favorable_dwell_started_mono``) so the
    full favorable-exit only fires when the predicate holds
    continuously for ``adaptive_spread_widen_favorable_exit_dwell_seconds``.
    """
    if reason == "toxicity_hard":
        return not toxicity.hard_trigger
    if reason == "toxicity_soft":
        return not toxicity.soft_trigger
    if reason == "markout_adverse":
        if not toxicity.adverse_uses_delayed_markouts:
            return True
        return toxicity.avg_adverse_markout_bps > -float(
            settings.toxicity_markout_soft_bps
        ) + 1e-12
    if reason == "one_sided_ratio":
        return toxicity.one_sided_fill_ratio < float(
            settings.toxicity_one_sided_fill_ratio
        ) - 1e-12
    if reason == "quote_quality":
        return not quote_quality_signal
    if reason == "slow_trend":
        return not slow_trend_signal
    return False


def compute_effective_min_half_spread_bps(
    settings: Settings,
    active_sides: ActiveSides,
    toxicity_score: float,
    *,
    spread_floor_overlay_half_spread_bps: float = 0.0,
    price_tick: float | None = None,
    fair_value: float | None = None,
    aqc_aggression_level: float | None = None,
) -> float:
    """
    Minimum half-spread (bps) the bot should respect for profitability / adverse-selection guardrails.

    Uses inventory mode (two-sided vs one-sided reduction), configured economic floors, and a linear
    bump from toxicity score (0..1).

    2026-05-13 todo-019 Part B: when ``price_tick`` and
    ``fair_value`` are both provided, a tick-denominated floor is
    composed alongside the bps floor. The effective floor becomes
    ``max(bps_floor, (1 + extra_ticks) * tick_bps_half)``. On
    tight-tick symbols this prevents the floor from rounding sub-
    tick (e.g. TON: 1 tick ≈ 1 bp half at $2.30; the
    INVENTORY_BPS=2.0 floor was only 2 ticks worth and didn't bind
    on the actual price grid when one-sided).

    Extra ticks are controlled by:
    - ``ONE_SIDED_EXTRA_TICK_NEUTRAL`` for two-sided mode (default 0)
    - ``ONE_SIDED_EXTRA_TICK_INVENTORY`` for one-sided mode (default 0)

    Defaults of 0.0 → tick-floor degenerates to ``tick_bps_half``
    which on most symbols is already smaller than the bps floor;
    behaviour is unchanged unless ``ONE_SIDED_EXTRA_TICK_*`` is
    configured. Set to 1.0 on tight-tick MM symbols to require at
    least one extra tick behind touch when one-sided.

    When ``price_tick`` or ``fair_value`` is missing / non-finite,
    the tick floor is skipped (pre-1.2.79 behaviour preserved).

    v1.5.281 AQC Phase 2: ``aqc_aggression_level`` (0..1, from the
    Active Quoting Controller) modulates the EFFECTIVE base econ floor
    TIGHTEN-ONLY. It is applied only when ``AQC_WIRE_MIN_HALF_SPREAD``
    is on AND the level is a finite number; both ``None`` and wire-off
    leave the result byte-identical to the pre-AQC computation. At
    aggression=0 the base is unchanged; at aggression=1 the base is
    lerped down toward ``AQC_MIN_HALF_SPREAD_FLOOR_AT_FULL_AGGRESSION_BPS``.
    The ``min(base, lerped)`` guard makes it impossible to WIDEN the
    floor (so a mis-set target above base is a no-op), and the lerp is
    applied BEFORE the tick floor so the sub-tick guard still clamps
    AQC tightening up to >=1 tick. ``tox_bump`` / ``overlay`` compose
    additively AFTER this, so adaptive-widening defenses are never
    undone. See plans/aqc-execute.md Phase 2.
    """
    inv = active_sides in (ActiveSides.BID_ONLY, ActiveSides.ASK_ONLY)
    econ = (
        float(settings.economic_min_half_spread_inventory_bps)
        if inv
        else float(settings.economic_min_half_spread_neutral_bps)
    )
    base = max(float(settings.min_half_spread_bps), econ)
    # v1.5.281 AQC Phase 2 — tighten-only aggression modulation of the
    # econ base floor (guarded; default-off wire → no-op).
    if (
        getattr(settings, "aqc_wire_min_half_spread", False)
        and aqc_aggression_level is not None
        and math.isfinite(float(aqc_aggression_level))
    ):
        aggr = min(1.0, max(0.0, float(aqc_aggression_level)))
        tight = float(
            getattr(
                settings,
                "aqc_min_half_spread_floor_at_full_aggression_bps",
                base,
            )
        )
        lerped = base + aggr * (tight - base)
        base = min(base, lerped)
    # todo-019 Part B: tick-aware floor.
    if (
        price_tick is not None
        and fair_value is not None
        and math.isfinite(float(price_tick))
        and math.isfinite(float(fair_value))
        and float(price_tick) > 0.0
        and float(fair_value) > 0.0
    ):
        tick_bps_half = (
            float(price_tick) / float(fair_value)
        ) * 10_000.0 / 2.0
        extra_ticks = float(
            getattr(settings, "one_sided_extra_tick_inventory", 0.0)
            if inv
            else getattr(settings, "one_sided_extra_tick_neutral", 0.0)
        )
        tick_floor = (1.0 + extra_ticks) * tick_bps_half
        base = max(base, tick_floor)
    tox_bump = max(0.0, float(toxicity_score)) * float(
        settings.economic_toxicity_score_half_spread_bps
    )
    overlay = max(0.0, float(spread_floor_overlay_half_spread_bps))
    out = base + tox_bump + overlay
    return min(out, float(settings.max_half_spread_bps))


def compute_effective_inventory_util_floor_pct(
    settings: Settings,
    *,
    aqc_aggression_level: float | None = None,
) -> float:
    """Effective inventory exec-bias util floor (fraction, 0..1).

    Returns ``INVENTORY_EXEC_BIAS_MIN_UTIL_PCT`` unchanged unless the
    AQC Phase 3 wire is active.

    v1.5.282 AQC Phase 3: ``aqc_aggression_level`` (0..1, from the
    Active Quoting Controller) modulates the util floor TIGHTEN-ONLY.
    Applied only when ``AQC_WIRE_INVENTORY`` is on AND the level is a
    finite number; both ``None`` and wire-off return the base floor
    byte-identical to the pre-AQC value. At aggression=0 the floor is
    unchanged; at aggression=1 the floor is lerped DOWN toward
    ``AQC_INVENTORY_UTIL_FLOOR_AT_FULL_AGGRESSION_PCT``. A lower floor
    makes the inventory exec-bias gate engage at a lower utilization →
    the adding side is suppressed earlier → faster inventory churn when
    the controller is aggressive about chasing fills.

    The ``min(base, lerped)`` guard makes it impossible to RAISE the
    floor, so a mis-set endpoint above the base is a silent no-op
    (exec-bias can never be made to engage LATER than configured). The
    caller's ``max(inventory_execution_bias_ratio, floor)`` clamp still
    applies downstream, so the engagement util can never fall below the
    ratio. See plans/aqc-execute.md Phase 3.
    """
    base = float(
        getattr(settings, "inventory_execution_bias_min_util_pct", 0.0)
    )
    if (
        getattr(settings, "aqc_wire_inventory", False)
        and aqc_aggression_level is not None
        and math.isfinite(float(aqc_aggression_level))
    ):
        aggr = min(1.0, max(0.0, float(aqc_aggression_level)))
        tight = float(
            getattr(
                settings,
                "aqc_inventory_util_floor_at_full_aggression_pct",
                base,
            )
        )
        lerped = base + aggr * (tight - base)
        base = min(base, lerped)
    return base


def compute_effective_inventory_skew_coeff_bps(
    settings: Settings,
    *,
    aqc_aggression_level: float | None = None,
) -> float:
    """Effective inventory skew coefficient (bps).

    Returns ``INVENTORY_SKEW_COEFF_BPS`` unchanged unless the AQC
    Phase 4 wire is active.

    v1.5.283 AQC Phase 4: ``aqc_aggression_level`` (0..1, from the
    Active Quoting Controller) modulates the inventory skew
    coefficient INCREASE-ONLY. Applied only when ``AQC_WIRE_SKEW`` is
    on AND the level is a finite number; both ``None`` and wire-off
    return the base coefficient byte-identical to the pre-AQC value.
    At aggression=0 the coefficient is unchanged; at aggression=1 it is
    lerped UP toward ``AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS``. A higher
    coefficient shifts the reservation further per unit of inventory →
    the inventory-REDUCING side quote pulls closer to touch and the
    adding side pushes away → faster, constructive inventory flattening
    when the controller is aggressive about chasing fills.

    The ``max(base, lerped)`` guard makes it impossible to LOWER the
    coefficient, so a mis-set endpoint below the base is a silent no-op
    (the bot can never be made to skew LESS aggressively than
    configured). This is the constructive / safe direction — it churns
    inventory DOWN, never holds risk longer (cf. CLAUDE.md Rule 0c).
    The trend-skew amplifier multiplier still composes on top, and the
    downstream ``MAX_RESERVATION_SHIFT_BPS_FROM_MID`` clamp still bounds
    the total reservation shift. See plans/aqc-execute.md Phase 4.
    """
    base = float(getattr(settings, "inventory_skew_coeff_bps", 0.0))
    if (
        getattr(settings, "aqc_wire_skew", False)
        and aqc_aggression_level is not None
        and math.isfinite(float(aqc_aggression_level))
    ):
        aggr = min(1.0, max(0.0, float(aqc_aggression_level)))
        higher = float(
            getattr(
                settings,
                "aqc_skew_coeff_at_full_aggression_bps",
                base,
            )
        )
        lerped = base + aggr * (higher - base)
        base = max(base, lerped)
    return base


def apply_profitability_spread_floor(
    settings: Settings,
    decision: QuoteDecision,
    *,
    want_bid: bool,
    want_ask: bool,
    best_bid: float | None,
    best_ask: float | None,
    bid_px: float,
    ask_px: float,
    price_tick: float | None = None,
) -> tuple[float, float, bool]:
    """
    After quote aging, widen targets if implied spread vs mid is below the effective economic floor.

    Returns (bid_px, ask_px, widened).

    2026-05-13 todo-019 Part B: ``price_tick`` (when provided) is
    threaded into the floor computation so the bps floor can be
    composed with a tick-denominated floor. See
    ``compute_effective_min_half_spread_bps`` for details. The
    parameter is optional for backwards-compat; pre-1.2.79 calls
    behave unchanged when omitted.
    """
    mid = decision.mid_price
    if not isinstance(mid, (int, float)) or not math.isfinite(float(mid)) or float(mid) <= 0:
        return bid_px, ask_px, False

    mid_f = float(mid)
    eff = compute_effective_min_half_spread_bps(
        settings,
        decision.active_sides,
        decision.toxicity_score,
        spread_floor_overlay_half_spread_bps=decision.spread_floor_overlay_half_spread_bps,
        price_tick=price_tick,
        fair_value=mid_f,
        # v1.5.281 AQC Phase 2 — keep the non-normal-mode re-widen
        # floor CONSISTENT with the aggression-modulated floor so a
        # quote already inside the tightened floor isn't re-widened.
        aqc_aggression_level=getattr(decision, "aqc_aggression_level", None),
    )
    half_px = eff / 10_000.0 * mid_f

    if want_bid and want_ask and bid_px > 0 and ask_px > 0:
        gross_bps = (float(ask_px) - float(bid_px)) / mid_f * 10_000.0
        need = 2.0 * eff
        if gross_bps + 1e-9 >= need:
            return bid_px, ask_px, False
        nb = clamp_buy_post_only(mid_f - half_px, best_bid)
        na = clamp_sell_post_only(mid_f + half_px, best_ask)
        return nb, na, True

    if want_bid and not want_ask and bid_px > 0:
        max_bid = mid_f - half_px
        widened = False
        if bid_px > max_bid + 1e-12:
            bid_px = max_bid
            widened = True
        bid_px = clamp_buy_post_only(bid_px, best_bid)
        return bid_px, ask_px, widened

    if want_ask and not want_bid and ask_px > 0:
        min_ask = mid_f + half_px
        widened = False
        if ask_px < min_ask - 1e-12:
            ask_px = min_ask
            widened = True
        ask_px = clamp_sell_post_only(ask_px, best_ask)
        return bid_px, ask_px, widened

    return bid_px, ask_px, False


def compute_microprice(
    best_bid: Optional[float],
    best_ask: Optional[float],
    bid_size: Optional[float],
    ask_size: Optional[float],
) -> Optional[float]:
    """Queue-imbalance-weighted mid.

    ``microprice = (bid_size * best_ask + ask_size * best_bid) / (bid_size + ask_size)``

    Intuition: a heavy bid stack means buyers are queued up. The next
    liquidity-taking trade is more likely to consume them, which moves
    price up — so the microprice pulls toward the ask side. Heavy ask
    stack: symmetric, pulls toward the bid side. Using microprice as
    the reservation centerpoint shifts our quotes in the direction price
    is statistically likely to move next, reducing adverse selection.

    Returns None when any input is missing, non-positive size totals,
    or the result isn't finite. The caller is expected to fall back to
    the midprice in those cases — the strategy must still produce a
    decision.
    """
    if best_bid is None or best_ask is None:
        return None
    if bid_size is None or ask_size is None:
        return None
    try:
        b_px = float(best_bid)
        a_px = float(best_ask)
        b_sz = float(bid_size)
        a_sz = float(ask_size)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(b_px) and math.isfinite(a_px)):
        return None
    if not (math.isfinite(b_sz) and math.isfinite(a_sz)):
        return None
    if b_sz < 0 or a_sz < 0:
        return None
    total = b_sz + a_sz
    if total <= 0:
        return None
    mp = (b_sz * a_px + a_sz * b_px) / total
    if not math.isfinite(mp) or mp <= 0:
        return None
    return mp


def compute_quote_decision(
    settings: Settings,
    mid: float,
    position_qty: float,
    vol_bps: float,
    toxicity: ToxicitySnapshot,
    *,
    spread_floor_overlay_half_spread_bps: float = 0.0,
    best_bid: Optional[float] = None,
    best_ask: Optional[float] = None,
    bid_size: Optional[float] = None,
    ask_size: Optional[float] = None,
    reference_fair_price: Optional[float] = None,
    short_term_drift_bps: Optional[float] = None,
    # v1.5.283 AQC Phase 4 — the controller's live aggression_level
    # (0..1) or None. When ``AQC_WIRE_SKEW`` is on, this lerps the
    # inventory skew coefficient UP toward
    # ``AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS`` (increase-only). None /
    # wire-off → the skew coefficient is the configured base, byte-
    # identical to pre-Phase-4. Caller (bot.py) passes the same value it
    # stamps onto ``decision.aqc_aggression_level``.
    aqc_aggression_level: Optional[float] = None,
    ob_imbalance_smoothed: Optional[float] = None,
    cross_venue_basis_now: Optional[float] = None,
    cross_venue_basis_ewma: Optional[float] = None,
    basis_deviation_regime_sign: float = -1.0,
    join_depth_overlay_bps: float = 0.0,
    flow_score_tfi_signed: Optional[float] = None,
    flow_score_streak_buy: int = 0,
    flow_score_streak_sell: int = 0,
    flow_score_streak_window_prints: int = 10,
    vol_regime_shrink_factor: float = 1.0,
    vol_regime_half_spread_bump_bps: float = 0.0,
    recent_markout_5s_median_bps: Optional[float] = None,
    # Phase 8A (v1.5.185) — Avellaneda-Stoikov k-intensity input.
    # Caller (bot.py) refreshes ``state.as_k_intensity_per_min``
    # every ``AS_K_INTENSITY_REFRESH_SECONDS`` and passes it here.
    # When AS is enabled and this is ``None`` (cold start), the
    # AS formula falls through to its k_floor — bot still produces
    # a sensible base half-spread on the first few ticks.
    as_k_intensity_per_min: Optional[float] = None,
    basis_regime_size_mult: float = 1.0,
    basis_regime_size_mult_reason: Optional[str] = None,
    # todo-011: post-fill replace cooldown. Caller (bot.py) computes
    # how many ms remain on the cooldown for each side, based on
    # ``state.last_fill_monotonic_ms_{buy,sell}`` and the configured
    # ``POST_FILL_REPLACE_COOLDOWN_MS``. Both default to 0 (no
    # cooldown active / feature disabled). Positive value on either
    # side suppresses placement on that side via active_sides.
    post_fill_cooldown_bid_remaining_ms: float = 0.0,
    post_fill_cooldown_ask_remaining_ms: float = 0.0,
    # 2026-05-12 codex-#1 narrow: at-touch adverse pause. When recent
    # at-touch fills cluster adverse on a given side, that side is
    # paused for the cooldown duration. Caller passes the per-side
    # ``True`` flag; we suppress the side via active_sides.
    at_touch_adverse_pause_bid: bool = False,
    at_touch_adverse_pause_ask: bool = False,
    # v1.4.161 Phase 4C.3 mini: per-side realised-edge suppression.
    # When the rolling N-fill mean of (markout_5s + rebate) on a
    # given side drops past threshold, that side is paused. Same
    # active_sides mechanism as at_touch_adverse_pause but a
    # different trigger signal (covers all fills, not just at_touch).
    realised_edge_suppress_bid: bool = False,
    realised_edge_suppress_ask: bool = False,
    # v1.4.164 Phase 4C.1+4C.2 — per-side expected-edge refusal.
    # When the per-side (target_half_spread + rebate - typical_adverse)
    # expected NET edge is below ``MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE``,
    # refuse to quote that side. Bot maintains the boolean flag with
    # hysteresis (see ``app/expected_edge.py``) and passes it in here.
    expected_edge_refused_bid: bool = False,
    expected_edge_refused_ask: bool = False,
    # 2026-05-12 codex-#3: fill-burst size shrink. 1.0 normally;
    # ``FILL_BURST_SIZE_MULT`` (default 0.5) during the cooldown
    # after a detected burst. Composes via ``min`` semantics with the
    # other size_mult chain components.
    fill_burst_size_mult: float = 1.0,
    # 1.4.9 gate-to-widening Phase 1: when ``composition`` is supplied,
    # the per-side half-spread is sourced from
    # ``composition.effective_half_spread_bid_bps(MAX)`` /
    # ``..._ask_bps(MAX)`` rather than the legacy symmetric
    # ``compute_effective_min_half_spread_bps`` path. When ``None``
    # (callers that haven't migrated yet), the legacy symmetric path
    # runs unchanged — preserves backward compatibility during the
    # cutover.
    composition: Optional["SpreadComposition"] = None,
    # v1.4.112 Phase 1C — regime_controller knob overlays. Both default
    # to 1.0 (no-op) so existing callers that haven't migrated keep
    # current behaviour exactly. Wired from bot.py's tick after the
    # FSM transitions for this tick.
    #
    # ``regime_base_half_spread_mult`` multiplies the post-vol baseline
    # half_spread BEFORE the eff_min/max clamps and the SpreadComposition
    # floor. The clamps and gate floors still bind: a too-low product
    # snaps up to eff_min; a too-high product snaps down to
    # MAX_HALF_SPREAD_BPS; gate widening (via composition) still wins
    # when higher.
    #
    # ``regime_quote_notional_mult`` composes multiplicatively with the
    # existing size_mult chain (toxicity / vol_regime / markout /
    # basis_regime / fill_burst), and is then re-clipped to the
    # MIN_QUOTE_NOTIONAL_USD floor so the venue spec still binds.
    regime_base_half_spread_mult: float = 1.0,
    regime_quote_notional_mult: float = 1.0,
    # v1.5.156 Option B — caller supplies seconds since the last
    # fill (any side); the function uses this for the no-fill spread-
    # compression mechanism. ``None`` (default) disables compression
    # entirely — preserves legacy behaviour for callers that don't
    # yet pass the timer (e.g. tests, replay driver tests). When the
    # bot has had no fills yet this session, callers should pass
    # ``None`` (not a huge number) to avoid triggering compression
    # before a baseline fill rate is established.
    seconds_since_last_fill: Optional[float] = None,
    # v1.5.248 — no-fill aggression escalator output. When the
    # escalator is enabled in settings, the caller pre-computes the
    # `EscalatorOutput` struct (via `compute_escalator_output` from
    # app/no_fill_escalator.py) and passes it here. When provided
    # AND aggression_level > 0, this overrides the legacy
    # NO_FILL_COMPRESS spread-only path and additionally attenuates
    # microprice widening, toxicity bump, and the reservation-shift
    # alpha sum per the escalator's per-tick multipliers.
    no_fill_escalator_output: Optional[Any] = None,
    # v1.5.158 Option A — caller supplies short-window vs long-
    # window MA ratio of realized vol_bps. When the ratio exceeds
    # the configured arm threshold the gate adds a small widening
    # to half-spread. Pure-function: caller computes the MAs from
    # ``state.vol_bps_history``. ``None`` = gate cannot fire this
    # tick (e.g. deque not yet warm).
    vol_climbing_ratio: Optional[float] = None,
    # v1.5.158 Option B — caller supplies the current UTC time-of-
    # day as (utc_hour, utc_minute). ``None`` disables the funding-
    # settle widen entirely.
    utc_hour_minute: Optional[tuple[int, int]] = None,
    # v1.5.158 Option C — caller can override MAX_ABS_POSITION for
    # this quote cycle (e.g. shrink the cap during high-vol regimes).
    # Falls back to ``settings.max_abs_position`` when ``None``.
    # Bounded by the settings static value at the caller's
    # discretion — pass a smaller value to reduce, pass the same
    # value to no-op.
    max_abs_position_override: Optional[float] = None,
    # v1.5.209 Phase 8D — OFI directional alpha (5th reservation-alpha).
    # Caller (bot.py) refreshes ``state.ofi.signal_5s_normalised()``
    # once per tick and passes it here. ``None`` (warmup or accumulator
    # disabled) → zero shift applied. ``OFI_RESERVATION_ALPHA=0.0``
    # default keeps the lever dormant even when a signal IS available
    # — both gates must be open for the shift to fire.
    ofi_signal_5s_normalised: Optional[float] = None,
    # v1.5.209 Phase 8B — Queue-position-aware sizing + inside-post.
    # Caller computes per-side ratio + arrival rate from
    # ``state.queue_arrival_rate`` + own resting WO sizes + inside L1
    # sizes. ``None`` (no own order this side, or feature disabled)
    # → no shrink + no inside-post. Both flags + the per-side ratio
    # are independent — sizing alone is the v1.5.209 default A/B
    # path, inside-post is the v1.5.210 A/B follow-up.
    queue_position_ratio_bid: Optional[float] = None,
    queue_position_ratio_ask: Optional[float] = None,
    # Phase 1b (v1.5.90) — clock-route the two ``utc_now()`` sites
    # inside this function so the replay driver can produce a
    # deterministic ``QuoteDecision.ts`` stream. Optional kwarg with a
    # default ``SystemClock()`` so production code paths that haven't
    # been updated yet continue to work unchanged.
    clock: Clock | None = None,
) -> QuoteDecision:
    """A-S style reservation price + half-spread, inventory and toxicity aware.

    Reservation reference: microprice when (a) the feature flag
    ``MICROPRICE_RESERVATION_ENABLED`` is true, AND (b) top-of-book
    depth is available (``best_bid``/``best_ask``/``bid_size``/``ask_size``
    all populated and finite). Otherwise falls back to ``mid``. The
    chosen reference is both used for the reservation calc and returned
    in ``QuoteDecision.microprice`` for offline analysis — we record
    the microprice value whenever depth is available, regardless of
    whether it was used (so the flag can be toggled for shadow-testing).
    """
    if not isinstance(mid, (int, float)) or not math.isfinite(float(mid)) or mid <= 0:
        raise ValueError("mid must be a finite positive number")
    if not isinstance(position_qty, (int, float)) or not math.isfinite(float(position_qty)):
        raise ValueError("position_qty must be finite")
    if not isinstance(vol_bps, (int, float)) or not math.isfinite(float(vol_bps)):
        raise ValueError("vol_bps must be finite")

    # Always compute microprice when depth is available, even if the
    # feature flag is off — we store it in the QuoteDecision telemetry
    # so shadow-mode comparisons are possible. The flag gates whether
    # we USE it for reservation, not whether we compute it.
    microprice = compute_microprice(best_bid, best_ask, bid_size, ask_size)
    if settings.microprice_reservation_enabled and microprice is not None:
        ref_price = microprice
    else:
        ref_price = float(mid)

    # Adaptive lever #7 — reference-venue fair-value blend.
    # Blend GRVT's local reference (microprice / mid) toward the
    # Bybit-anchored fair value passed in by the caller. At blend=0.0
    # this is a no-op (legacy). At blend=0.5 we land halfway between
    # GRVT and Bybit's view. When the reference feed is missing or
    # non-finite the blend is silently skipped.
    ref_blend_alpha = float(settings.reference_venue_fair_blend_alpha)
    if (
        ref_blend_alpha > 0.0
        and reference_fair_price is not None
        and isinstance(reference_fair_price, (int, float))
        and math.isfinite(float(reference_fair_price))
        and float(reference_fair_price) > 0.0
    ):
        ref_price = (1.0 - ref_blend_alpha) * ref_price + ref_blend_alpha * float(
            reference_fair_price
        )

    # 1.2.34: capture the anchor (post-blend, pre-shift) so the
    # spread-tab breakdown can show each shift's bp contribution.
    _breakdown_ref_price_anchor = float(ref_price)
    _breakdown_reservation_reference = (
        "microprice"
        if settings.microprice_reservation_enabled and microprice is not None
        else "mid"
    )

    # v1.5.158 Option C — vol-adaptive cap override. When the caller
    # determines vol is high enough to warrant a reduced cap, it
    # passes ``max_abs_position_override`` <= settings.max_abs_position.
    # The reduced value drives inventory_skew strength + the soft/
    # hard skew thresholds + active_sides at-max logic. Falls back
    # to the legacy ``settings.max_abs_position`` when override is
    # ``None`` or non-positive.
    if (
        max_abs_position_override is not None
        and isinstance(max_abs_position_override, (int, float))
        and float(max_abs_position_override) > 0.0
    ):
        max_pos = float(max_abs_position_override)
    else:
        max_pos = settings.max_abs_position
    norm_inv = clip(position_qty / max_pos if max_pos > 0 else 0.0, -1.0, 1.0)
    # v1.5.283 AQC Phase 4 — effective inventory skew coefficient. When
    # AQC_WIRE_SKEW is on this lerps the coefficient UP toward
    # AQC_SKEW_COEFF_AT_FULL_AGGRESSION_BPS as aggression rises
    # (increase-only). Wire-off / None → the configured base, byte-
    # identical to pre-Phase-4. The trend-skew amplifier below still
    # multiplies on top; the MAX_RESERVATION_SHIFT_BPS_FROM_MID clamp
    # downstream still bounds the total reservation shift.
    skew = compute_effective_inventory_skew_coeff_bps(
        settings, aqc_aggression_level=aqc_aggression_level
    )
    # Trend-aware skew amplifier (1.3.82). When the position is large
    # AND short-term drift is aligned with the position direction
    # (long+up or short+down), boost the skew coefficient so the
    # reducing-side quote pulls closer to touch. Pairs with
    # momentum_gate which blocks the adding side under the identical
    # trigger geometry. Returns 1.0 (no-op) when inactive. ``eff_cap``
    # mirrors momentum_gate's denominator so the two features arm
    # together. Pure function; safe to call every cycle.
    _trend_amp_mult = 1.0
    _trend_amp_reason = ""
    if settings.trend_skew_amplifier_enabled:
        from app.trend_skew_amplifier import compute_trend_skew_multiplier
        max_notional = float(getattr(settings, "max_position_notional_usd", 0.0) or 0.0)
        eff_cap = max_pos
        if max_notional > 0 and ref_price > 0:
            eff_cap = min(eff_cap, max_notional / ref_price)
        _trend_amp_mult, _trend_amp_reason = compute_trend_skew_multiplier(
            position_qty=position_qty,
            effective_abs_cap=eff_cap,
            drift_bps=short_term_drift_bps,
            drift_threshold_bps=float(settings.trend_skew_amplifier_drift_threshold_bps),
            inventory_pct_threshold=float(settings.trend_skew_amplifier_inventory_pct),
            amplification_factor=float(settings.trend_skew_amplifier_factor),
            enabled=True,
        )
        skew = skew * _trend_amp_mult
    # Apply shape exponent: sign(norm_inv) * |norm_inv|**exp. At exp=1
    # this is linear (legacy). At exp>1 the ramp is gentle near zero and
    # steep near ±1, so ordinary inventory oscillation doesn't push the
    # reducing-side quote off the book — see
    # ``tmp/snap_20260419_152651`` suppression counts for the motivation.
    skew_exp = float(settings.inventory_skew_exponent)
    if skew_exp != 1.0 and norm_inv != 0.0:
        signed_adj = math.copysign(abs(norm_inv) ** skew_exp, norm_inv)
    else:
        signed_adj = norm_inv
    reservation = ref_price - skew * signed_adj * ref_price / 10_000.0
    # 1.2.34: capture the inventory_skew bp shift for the breakdown.
    _breakdown_inventory_skew_bps = -float(skew) * float(signed_adj)

    # Adaptive lever #1 — short-term drift bias on the reservation.
    # Positive drift (market rising) shifts reservation up: ask widens
    # away from an incoming up-move, bid tracks toward the continuation.
    # Additive on top of inventory skew so the two compose naturally.
    # No-op when drift is missing / non-finite or the alpha is zero.
    drift_alpha = float(settings.trend_drift_reservation_alpha)
    # 1.2.34: track drift shift for the breakdown.
    _breakdown_trend_drift_shift_bps = 0.0
    if (
        drift_alpha > 0.0
        and short_term_drift_bps is not None
        and isinstance(short_term_drift_bps, (int, float))
        and math.isfinite(float(short_term_drift_bps))
    ):
        reservation += drift_alpha * float(short_term_drift_bps) * ref_price / 10_000.0
        _breakdown_trend_drift_shift_bps = drift_alpha * float(short_term_drift_bps)

    # Toxicity widening coefficient. Hardcoding this to 12 was calibrated for
    # Hyperliquid's 2-5 bps book; on a 1-tick venue (GRVT's 0.04 bps ETH book)
    # 12 bps/unit-score means a 0.25-score toxicity pushes half-spread 3 bps
    # which is 80 ticks behind the touch — invisible. Now a setting:
    # ``TOXICITY_SCORE_HALF_SPREAD_BPS`` (default 12 for HL back-compat;
    # GRVT profile lowers to something tight-book appropriate).
    # Adaptive join-depth overlay (``JoinDepthController``). Sums in
    # alongside vol + toxicity contributions; the controller is hard-
    # bounded by ``JOIN_DEPTH_AUTOTUNE_OVERLAY_{MIN,MAX}_BPS`` and the
    # final ``min/max_half_spread_bps`` clip below provides an
    # additional envelope. Off by default (overlay = 0.0).
    # 1.2.34: per-component bp tracking for the breakdown.
    # v1.5.156 Option A — vol-adaptive base half-spread. When
    # ``BASE_HALF_SPREAD_VOL_ADAPTIVE_ENABLED=true`` AND current vol
    # is below the configured threshold, use the (typically lower)
    # ``BASE_HALF_SPREAD_BPS_LOW_VOL`` instead of the legacy
    # ``BASE_HALF_SPREAD_BPS``. Addresses the participation gap in
    # calm regimes (overnight 2026-05-26 screenshot showed 56-min
    # zero-fill windows in vol=5 bp/s regimes where the natural
    # touch was ~5 bps but the bot quoted ~20 bps wide).
    #
    # Phase 8A (v1.5.185) — when ``AVELLANEDA_STOIKOV_ENABLED=true``,
    # neither of the legacy branches applies; the base half-spread
    # is computed from a closed-form AS-inspired formula driven by
    # current vol_bps + recent fill-rate (k_intensity_per_min).
    # See ``app/avellaneda_stoikov.py`` for the formula. Default
    # off; takes precedence over the vol-adaptive switch when on.
    _as_enabled = bool(
        getattr(settings, "avellaneda_stoikov_enabled", False)
    )
    if _as_enabled:
        from app.avellaneda_stoikov import compute_as_half_spread_bps

        _k = (
            float(as_k_intensity_per_min)
            if as_k_intensity_per_min is not None
            and as_k_intensity_per_min == as_k_intensity_per_min  # not NaN
            else 0.0  # caller hasn't populated cache yet → use floor
        )
        _breakdown_base_half = compute_as_half_spread_bps(
            vol_bps=float(vol_bps),
            k_intensity_per_min=_k,
            gamma_inv=float(getattr(settings, "as_gamma_inv", 0.05)),
            gamma_edge=float(getattr(settings, "as_gamma_edge", 1.0)),
            edge_alpha=float(getattr(settings, "as_edge_alpha", 1.0)),
            k_floor_per_min=float(getattr(settings, "as_k_floor_per_min", 0.1)),
            base_floor_bps=float(getattr(settings, "as_base_floor_bps", 1.0)),
            min_half_spread_bps=float(
                getattr(settings, "as_min_half_spread_bps", 1.5)
            ),
            max_half_spread_bps=float(
                getattr(settings, "as_max_half_spread_bps", 30.0)
            ),
        )
    elif (
        getattr(settings, "base_half_spread_vol_adaptive_enabled", False)
        and float(vol_bps) < float(getattr(
            settings,
            "base_half_spread_low_vol_threshold_bps_per_s",
            6.0,
        ))
    ):
        _breakdown_base_half = float(getattr(
            settings,
            "base_half_spread_bps_low_vol",
            settings.base_half_spread_bps,
        ))
    else:
        _breakdown_base_half = float(settings.base_half_spread_bps)
    _breakdown_vol_contrib = float(settings.vol_multiplier * vol_bps)
    _breakdown_tox_bump = float(
        toxicity.score * float(settings.toxicity_score_half_spread_bps)
    )
    _breakdown_tox_score_used = float(toxicity.score)
    _breakdown_tox_coeff_used = float(settings.toxicity_score_half_spread_bps)
    _breakdown_join_depth_overlay = float(join_depth_overlay_bps)
    half_spread = (
        _breakdown_base_half
        + _breakdown_vol_contrib
        + _breakdown_tox_bump
        + _breakdown_join_depth_overlay
    )
    _breakdown_raw_half_spread = float(half_spread)
    # v1.5.158 Option A — vol-climbing anticipatory widening.
    # Adds a small widening when realized vol's short-window MA
    # exceeds long-window MA by the configured ratio. Caller passes
    # the precomputed ratio (avoids re-walking the buffer here).
    # Hysteresis: gate stays armed until ratio drops below
    # ``clear_ratio`` — but that state machine lives in the caller;
    # this function only checks the live ratio against the active
    # threshold (arm if not_armed, clear if armed). For simplicity
    # the gate fires whenever ratio >= arm_ratio (caller's choice
    # to track was_armed for finer hysteresis if needed).
    _breakdown_vol_climbing_widen_bps = 0.0
    if (
        getattr(settings, "vol_climbing_widen_enabled", False)
        and vol_climbing_ratio is not None
        and isinstance(vol_climbing_ratio, (int, float))
        and math.isfinite(float(vol_climbing_ratio))
    ):
        _vcw_arm = float(getattr(
            settings, "vol_climbing_widen_arm_ratio", 1.5
        ))
        if float(vol_climbing_ratio) >= _vcw_arm:
            _breakdown_vol_climbing_widen_bps = float(getattr(
                settings, "vol_climbing_widen_bps", 1.5
            ))
            half_spread = half_spread + _breakdown_vol_climbing_widen_bps

    # v1.5.158 Option B — funding-settle anticipatory widening.
    # Adds a small widening within ±N min of any configured UTC
    # settle hour. Deterministic time-driven; bounded; recurring.
    _breakdown_funding_settle_widen_bps = 0.0
    if (
        getattr(settings, "funding_settle_widen_enabled", False)
        and utc_hour_minute is not None
    ):
        try:
            _h, _m = utc_hour_minute
            _now_min = int(_h) * 60 + int(_m)
            _settle_hours_str = str(getattr(
                settings, "funding_settle_widen_hours_utc", "0,8,16"
            ))
            _pre = float(getattr(
                settings, "funding_settle_widen_pre_minutes", 15.0
            ))
            _post = float(getattr(
                settings, "funding_settle_widen_post_minutes", 15.0
            ))
            for _sh_str in _settle_hours_str.split(","):
                try:
                    _sh = int(_sh_str.strip())
                except (ValueError, TypeError):
                    continue
                _settle_min = _sh * 60
                # Distance to nearest settle in minutes (handle midnight
                # wrap-around for the 00:00 settle).
                _d_forward = (_settle_min - _now_min) % 1440
                _d_backward = (_now_min - _settle_min) % 1440
                if _d_forward <= _pre or _d_backward <= _post:
                    _breakdown_funding_settle_widen_bps = float(getattr(
                        settings, "funding_settle_widen_bps", 1.5
                    ))
                    half_spread = (
                        half_spread + _breakdown_funding_settle_widen_bps
                    )
                    break  # one settle fires; no double-add
        except (TypeError, ValueError):
            pass

    # v1.5.156 Option B — no-fill–aware spread compression. Applied
    # BEFORE the min/max clip so the existing safety bounds still
    # win. The compression bps is subtracted; ``min_half_spread_bps``
    # provides the hard floor.
    #
    # v1.5.248 — when the no-fill escalator is active (caller passes
    # `no_fill_escalator_output` with aggression_level > 0), the
    # escalator's spread_compression_bps takes precedence and the
    # legacy NO_FILL_COMPRESS path is bypassed. This unifies the
    # spread-compression dimension under the escalator's continuous
    # 0→1 aggression level (which also drives microprice + toxicity
    # + reservation-shift attenuation in the blocks below).
    _breakdown_no_fill_compress_bps = 0.0
    _escalator_active = (
        no_fill_escalator_output is not None
        and float(getattr(no_fill_escalator_output, "aggression_level", 0.0))
        > 0.0
    )
    if _escalator_active:
        _breakdown_no_fill_compress_bps = float(
            getattr(no_fill_escalator_output, "spread_compression_bps", 0.0)
        )
        half_spread = half_spread - _breakdown_no_fill_compress_bps
    elif (
        getattr(settings, "no_fill_compress_enabled", False)
        and seconds_since_last_fill is not None
        and isinstance(seconds_since_last_fill, (int, float))
        and math.isfinite(float(seconds_since_last_fill))
    ):
        _trigger_s = float(getattr(
            settings, "no_fill_compress_trigger_seconds", 600.0
        ))
        _rate_bps_per_min = float(getattr(
            settings, "no_fill_compress_rate_bps_per_minute", 1.0
        ))
        _max_compress = float(getattr(
            settings, "no_fill_compress_max_bps", 5.0
        ))
        _ssf = float(seconds_since_last_fill)
        if _ssf > _trigger_s and _rate_bps_per_min > 0.0:
            _elapsed_min = (_ssf - _trigger_s) / 60.0
            _breakdown_no_fill_compress_bps = max(
                0.0, min(_elapsed_min * _rate_bps_per_min, _max_compress)
            )
            half_spread = half_spread - _breakdown_no_fill_compress_bps
    half_spread = clip(
        half_spread,
        settings.min_half_spread_bps,
        settings.max_half_spread_bps,
    )
    _breakdown_tox_soft_bump = 0.0
    _breakdown_tox_soft_active = bool(toxicity.soft_trigger)
    if toxicity.soft_trigger:
        _breakdown_tox_soft_bump = float(
            settings.toxicity_soft_trigger_half_spread_bump_bps
        )
        # v1.5.248 — escalator may attenuate the toxicity bump when
        # the bot has been starving for fills (typically because
        # toxicity widening is itself blocking fills). Mult of 1.0
        # = no change; 0.0 = fully suppressed.
        if _escalator_active:
            _tox_mult = float(getattr(
                no_fill_escalator_output,
                "toxicity_widen_multiplier",
                1.0,
            ))
            _breakdown_tox_soft_bump *= _tox_mult
        half_spread += _breakdown_tox_soft_bump
        half_spread = clip(
            half_spread,
            settings.min_half_spread_bps,
            settings.max_half_spread_bps,
        )

    util = abs(position_qty) / max_pos if max_pos > 0 else 0.0
    active = ActiveSides.BOTH
    reason_parts = ["baseline"]

    soft = settings.inventory_soft_limit_pct
    hard = settings.inventory_hard_limit_pct

    if util >= 1.0:
        if position_qty > 0:
            active = ActiveSides.ASK_ONLY
            reason_parts.append("at_max_long")
        else:
            active = ActiveSides.BID_ONLY
            reason_parts.append("at_max_short")
    elif util >= hard:
        if position_qty > 0:
            active = ActiveSides.ASK_ONLY
            reason_parts.append("hard_skew_long")
        elif position_qty < 0:
            active = ActiveSides.BID_ONLY
            reason_parts.append("hard_skew_short")
    elif util >= soft:
        if position_qty > 0:
            active = ActiveSides.ASK_ONLY
            reason_parts.append("soft_skew_long")
        elif position_qty < 0:
            active = ActiveSides.BID_ONLY
            reason_parts.append("soft_skew_short")

    if toxicity.hard_trigger and toxicity.toxic_side:
        if toxicity.toxic_side == Side.BUY and active == ActiveSides.BOTH:
            active = ActiveSides.ASK_ONLY
            reason_parts.append("toxic_bid")
        elif toxicity.toxic_side == Side.SELL and active == ActiveSides.BOTH:
            active = ActiveSides.BID_ONLY
            reason_parts.append("toxic_ask")

    # todo-011: post-fill replace cooldown. If a BUY fill just landed,
    # don't re-place a BID for ``POST_FILL_REPLACE_COOLDOWN_MS`` —
    # this dodges the 0-100 ms predator window where freshly-placed
    # quotes are filled at ~85 % adverse rate by informed flow. Per-
    # side scoped (BUY fill → suppress BID only; ASK side unaffected
    # so we still earn rebate where we can). When both sides have an
    # active cooldown simultaneously (rare — requires fills on both
    # sides within the cooldown window) active_sides collapses to
    # NONE. Caller (bot.py) computes remaining ms from
    # ``state.last_fill_monotonic_ms_{buy,sell}`` + settings.
    if post_fill_cooldown_bid_remaining_ms > 0.0:
        if active == ActiveSides.BOTH:
            active = ActiveSides.ASK_ONLY
            reason_parts.append("post_fill_cooldown_bid")
        elif active == ActiveSides.BID_ONLY:
            active = ActiveSides.NONE
            reason_parts.append("post_fill_cooldown_bid")
    if post_fill_cooldown_ask_remaining_ms > 0.0:
        if active == ActiveSides.BOTH:
            active = ActiveSides.BID_ONLY
            reason_parts.append("post_fill_cooldown_ask")
        elif active == ActiveSides.ASK_ONLY:
            active = ActiveSides.NONE
            reason_parts.append("post_fill_cooldown_ask")

    # 2026-05-12 codex-#1 narrow: at-touch adverse pause. Suppresses
    # the whole side (not just at_touch placements) when recent
    # at_touch fills on that side have median markout below the
    # configured threshold. Coarser action than the full 2D bucket
    # suppression (deferred — see codex review), but covers the
    # dominant pattern of "informed flow consistently picks off our
    # at-touch quotes on one side."
    if at_touch_adverse_pause_bid:
        if active == ActiveSides.BOTH:
            active = ActiveSides.ASK_ONLY
            reason_parts.append("at_touch_adverse_pause_bid")
        elif active == ActiveSides.BID_ONLY:
            active = ActiveSides.NONE
            reason_parts.append("at_touch_adverse_pause_bid")
    if at_touch_adverse_pause_ask:
        if active == ActiveSides.BOTH:
            active = ActiveSides.BID_ONLY
            reason_parts.append("at_touch_adverse_pause_ask")
        elif active == ActiveSides.ASK_ONLY:
            active = ActiveSides.NONE
            reason_parts.append("at_touch_adverse_pause_ask")

    # v1.4.161 Phase 4C.3 mini — realised-edge side suppression.
    # Same active_sides mechanic as at_touch_adverse_pause above; the
    # trigger is the per-side trailing realised-edge mean rather than
    # the at-touch median markout. The two gates compose by
    # short-circuit OR via active_sides.
    if realised_edge_suppress_bid:
        if active == ActiveSides.BOTH:
            active = ActiveSides.ASK_ONLY
            reason_parts.append("realised_edge_suppress_bid")
        elif active == ActiveSides.BID_ONLY:
            active = ActiveSides.NONE
            reason_parts.append("realised_edge_suppress_bid")
    if realised_edge_suppress_ask:
        if active == ActiveSides.BOTH:
            active = ActiveSides.BID_ONLY
            reason_parts.append("realised_edge_suppress_ask")
        elif active == ActiveSides.ASK_ONLY:
            active = ActiveSides.NONE
            reason_parts.append("realised_edge_suppress_ask")

    # v1.4.164 Phase 4C.1+4C.2 — expected-edge refusal. Same
    # active_sides mechanism. Trigger: bot-side hysteresis machine
    # that watches the per-side expected NET edge
    # (target_half_spread + rebate - typical_adverse). Refuses when
    # the edge falls below MIN_EXPECTED_NET_EDGE_BPS_PER_SIDE and
    # only re-arms after recovery for HYSTERESIS_TICKS ticks.
    if expected_edge_refused_bid:
        if active == ActiveSides.BOTH:
            active = ActiveSides.ASK_ONLY
            reason_parts.append("expected_edge_refused_bid")
        elif active == ActiveSides.BID_ONLY:
            active = ActiveSides.NONE
            reason_parts.append("expected_edge_refused_bid")
    if expected_edge_refused_ask:
        if active == ActiveSides.BOTH:
            active = ActiveSides.BID_ONLY
            reason_parts.append("expected_edge_refused_ask")
        elif active == ActiveSides.ASK_ONLY:
            active = ActiveSides.NONE
            reason_parts.append("expected_edge_refused_ask")

    eff_min = compute_effective_min_half_spread_bps(
        settings,
        active,
        toxicity.score,
        spread_floor_overlay_half_spread_bps=spread_floor_overlay_half_spread_bps,
    )
    # 1.2.34: track which floor kind applies (neutral vs inventory).
    # The util threshold below mirrors compute_effective_min_half_spread_bps's
    # internal logic (which is gated on active == BOTH).
    _breakdown_floor_kind = "neutral" if active == ActiveSides.BOTH else "inventory"
    _breakdown_economic_floor_pre_overlay = float(eff_min)
    # Vol-spike persistence overlay (BUGS/todo-009.md). Inside the
    # spike window the floor lifts by a configured bump so the bot
    # doesn't immediately re-tighten its quotes the moment vol calms;
    # captures the post-spike continuation risk (bounce-and-retest).
    # Zero when the feature is off or the window has expired.
    _breakdown_vol_regime_bump = 0.0
    if vol_regime_half_spread_bump_bps > 0.0:
        _breakdown_vol_regime_bump = float(vol_regime_half_spread_bump_bps)
        eff_min = eff_min + _breakdown_vol_regime_bump
        reason_parts.append("vol_spike_window")
    _breakdown_economic_floor_final = float(eff_min)
    # v1.4.112 Phase 1C — apply regime_controller base_half_spread_mult
    # AFTER vol modulation, BEFORE the eff_min floor and max ceiling.
    # In NORMAL the mult is 1.0 and this is a no-op. In DEFENSIVE / SHOCK
    # the mult widens the baseline by 1.5× / 2× respectively, then the
    # existing clamps and SpreadComposition floor still apply on top.
    if (
        math.isfinite(float(regime_base_half_spread_mult))
        and float(regime_base_half_spread_mult) > 0.0
        and abs(float(regime_base_half_spread_mult) - 1.0) > 1e-9
    ):
        half_spread = float(half_spread) * float(regime_base_half_spread_mult)
    _half_spread_pre_clamp = float(half_spread)
    if half_spread + 1e-12 < eff_min:
        half_spread = eff_min
        _breakdown_clamp_winner = "floor"
    else:
        _breakdown_clamp_winner = "raw"
    _half_spread_pre_ceiling = float(half_spread)
    half_spread = min(half_spread, float(settings.max_half_spread_bps))
    if half_spread < _half_spread_pre_ceiling - 1e-12:
        _breakdown_clamp_winner = "ceiling"
    if spread_floor_overlay_half_spread_bps > 1e-12:
        reason_parts.append("adaptive_spread_widen")

    # 1.4.9 gate-to-widening Phase 1: per-side half-spread floor from
    # the SpreadComposition when caller supplies one. The composition
    # is a FLOOR — it raises the half-spread, never lowers it. So
    # the half-spread used below is:
    #
    #   bid_half_spread = clamp(max_bps, max(legacy_half_spread,
    #                                        composition.bid_floor))
    #
    # The legacy ``half_spread`` carries the bot's intended baseline
    # (BASE_HALF_SPREAD_BPS + vol modulations + econ floor + toxicity
    # + overlay). The composition carries the SAME econ/tox/overlay
    # contributors PLUS the 7 gate widening contributions. Taking the
    # max gives gate-equivalent behaviour at firing (gate contributes
    # MAX → spread caps at MAX) while preserving the legacy baseline
    # in the common no-gate-firing case.
    if composition is not None:
        max_bps = float(settings.max_half_spread_bps)
        bid_floor = composition.effective_half_spread_bid_bps(max_bps)
        ask_floor = composition.effective_half_spread_ask_bps(max_bps)
        # v1.5.248 — escalator's microprice attenuation. When active,
        # subtract a fraction of the microprice contribution from
        # the per-side floor. mult=1.0 → no change; mult=0.0 →
        # fully suppress microprice's contribution to the floor.
        # Other composition contributors (vol_trend, momentum,
        # freshness, etc.) are NOT touched — only microprice, which
        # is the dominant fill-blocker on TON per v1.5.230-231 data.
        if _escalator_active:
            _mp_mult = float(getattr(
                no_fill_escalator_output,
                "microprice_widen_multiplier",
                1.0,
            ))
            _mp_suppress = max(0.0, 1.0 - _mp_mult)
            if _mp_suppress > 1e-9:
                bid_floor = bid_floor - _mp_suppress * float(
                    getattr(composition, "microprice_bid_bps", 0.0)
                )
                ask_floor = ask_floor - _mp_suppress * float(
                    getattr(composition, "microprice_ask_bps", 0.0)
                )
        bid_half_spread = min(max_bps, max(half_spread, bid_floor))
        ask_half_spread = min(max_bps, max(half_spread, ask_floor))
        # ``half_spread`` (legacy single-value) is updated to the
        # max of the two sides for downstream diagnostics. Per-side
        # values drive the actual bid_px / ask_px below.
        half_spread = max(bid_half_spread, ask_half_spread)
    else:
        bid_half_spread = half_spread
        ask_half_spread = half_spread

    # v1.5.256 — re-apply escalator spread compression to the
    # per-side spreads. Bug fix: v1.5.248 wired the compression
    # into the legacy `half_spread` variable BEFORE the composition
    # floor (line 1221), but the per-side floor max'es with the
    # composition's `econ_floor_bps` (which includes vol_contrib —
    # often 5+ bp on TON). So the compression was silently REVERTED
    # when composition floor exceeded the compressed `half_spread`.
    # v1.5.251 snapshot symptom: reservation_delta=0 (escalator's
    # reservation reset worked) but per-side spread = 4 bp despite
    # escalator at level=1 compressing by 15 bp. 98% of orders
    # behind_touch, fill rate dropped from v1.5.248's 0.76 to 0.57.
    #
    # Fix: subtract the escalator's spread_compression_bps from
    # EACH per-side spread AFTER the composition floor is taken,
    # floored by MIN_HALF_SPREAD_BPS. The MIN_HALF floor still
    # protects against sub-tick widths and post-only rejection.
    if _escalator_active:
        _bid_compress = float(getattr(
            no_fill_escalator_output, "spread_compression_bps", 0.0,
        ))
        _min_half = float(settings.min_half_spread_bps)
        bid_half_spread = max(_min_half, bid_half_spread - _bid_compress)
        ask_half_spread = max(_min_half, ask_half_spread - _bid_compress)
        # Update legacy diagnostic to reflect the post-escalator
        # per-side spreads.
        half_spread = max(bid_half_spread, ask_half_spread)

    # Priority #1 adaptive lever — order-book imbalance shift.
    # Applied AFTER half_spread is finalized (the shift magnitude is
    # capped at a fraction of half_spread, so we need its final value).
    # Caller supplies a pre-smoothed, finite ``ob_imbalance_smoothed`` in
    # [-1, 1]; this function clips it and translates into a price shift.
    # Composition intent (from deep-research consensus): the alpha term is
    # additive to the reservation — it does NOT replace the mid anchor.
    # A heavily bid-skewed book (I > 0) pushes reservation up so the bid
    # leans into the expected continuation and the ask widens away from
    # the imminent up-tick. Sign-symmetric on the short side.
    ob_alpha = float(settings.ob_imbalance_alpha)
    ob_shift_applied_bps = 0.0
    if (
        ob_alpha > 0.0
        and ob_imbalance_smoothed is not None
        and isinstance(ob_imbalance_smoothed, (int, float))
        and math.isfinite(float(ob_imbalance_smoothed))
    ):
        clip_bound = float(settings.ob_imbalance_clip)
        imb = clip(float(ob_imbalance_smoothed), -clip_bound, clip_bound)
        # Cap shift magnitude at ob_alpha × (half_spread / 2). With default
        # ob_alpha=0.3, the maximum |shift| is 15 % of half_spread — small
        # enough not to overwhelm skew / basis logic, large enough to
        # matter against a ~2 bps adverse-selection backdrop.
        ob_shift_applied_bps = ob_alpha * (half_spread / 2.0) * imb
        reservation += ob_shift_applied_bps * ref_price / 10_000.0
        if abs(ob_shift_applied_bps) > 1e-9:
            reason_parts.append("ob_imbalance_shift")

    # Priority #2 adaptive lever — cross-venue basis-deviation with regime
    # sign. Caller supplies instantaneous and smoothed basis (both in
    # price units) and a regime sign from the online IC classifier:
    #   sign = -1.0 → mean-reversion (legacy v1 behaviour; default)
    #   sign = +1.0 → trend-continuation (flip the shift direction)
    #   sign =  0.0 → undecided (skip the alpha regardless of alpha setting)
    # With ``sign`` defaulting to -1.0, existing call sites that don't
    # pass a regime sign preserve the v1 mean-reversion behaviour.
    basis_alpha = float(settings.basis_deviation_alpha)
    basis_shift_applied_bps = 0.0
    regime_sign = float(basis_deviation_regime_sign)
    if (
        basis_alpha > 0.0
        and regime_sign != 0.0
        and cross_venue_basis_now is not None
        and cross_venue_basis_ewma is not None
        and isinstance(cross_venue_basis_now, (int, float))
        and isinstance(cross_venue_basis_ewma, (int, float))
        and math.isfinite(float(cross_venue_basis_now))
        and math.isfinite(float(cross_venue_basis_ewma))
        and ref_price > 0
    ):
        dev_px = float(cross_venue_basis_now) - float(cross_venue_basis_ewma)
        dev_bps = dev_px / ref_price * 10_000.0
        clip_bps = float(settings.basis_deviation_clip_bps)
        if clip_bps > 0:
            dev_norm = clip(dev_bps / clip_bps, -1.0, 1.0)
            # shift = sign × alpha × (half/2) × dev_norm
            # With sign = sign(IC):
            #   IC < 0 → sign = -1 → shift opposes dev (mean-revert)
            #   IC > 0 → sign = +1 → shift aligns with dev (trend-follow)
            basis_shift_applied_bps = regime_sign * basis_alpha * (half_spread / 2.0) * dev_norm
            reservation += basis_shift_applied_bps * ref_price / 10_000.0
            if abs(basis_shift_applied_bps) > 1e-9:
                reason_parts.append("basis_deviation_shift")

    # Priority #3 v2 adaptive lever — flow-score reservation shift.
    # Combines TFI signed-imbalance (volume aggressor pressure over the
    # 1s window) and signed streak length (consecutive same-side
    # aggressors at the tail) into a single bounded composite, then
    # scales by half-spread the same way OB-imbalance does. Sign:
    # positive composite (buy-pressure / buy-streak) → reservation UP.
    # Caller passes the raw signals from ``FlowScoreAccumulator``;
    # this block does its own clipping and composition to keep the
    # accumulator a pure observation primitive.
    flow_alpha = float(settings.flow_score_reservation_alpha)
    flow_shift_applied_bps = 0.0
    if (
        flow_alpha > 0.0
        and flow_score_tfi_signed is not None
        and isinstance(flow_score_tfi_signed, (int, float))
        and math.isfinite(float(flow_score_tfi_signed))
    ):
        flow_clip = float(settings.flow_score_reservation_clip)
        tfi_clipped = clip(float(flow_score_tfi_signed), -flow_clip, flow_clip)
        streak_window = max(2, int(flow_score_streak_window_prints))
        # Net signed streak in [-1, +1]; only one side is non-zero by
        # construction in the accumulator.
        streak_signed = (
            float(flow_score_streak_buy) - float(flow_score_streak_sell)
        ) / float(streak_window)
        if streak_signed > 1.0:
            streak_signed = 1.0
        elif streak_signed < -1.0:
            streak_signed = -1.0
        composite = 0.5 * tfi_clipped + 0.5 * streak_signed
        # Re-clip composite — averaging two values each in [-1,1] is
        # already in [-1,1] but defensive against future feature adds.
        if composite > flow_clip:
            composite = flow_clip
        elif composite < -flow_clip:
            composite = -flow_clip
        flow_shift_applied_bps = flow_alpha * (half_spread / 2.0) * composite
        reservation += flow_shift_applied_bps * ref_price / 10_000.0
        if abs(flow_shift_applied_bps) > 1e-9:
            reason_parts.append("flow_score_shift")

    # Adaptive lever #5 (v1.5.209 Phase 8D) — OFI directional alpha.
    # Caller passes the pre-normalised signal in [-1, 1]; this block
    # applies the same ``alpha × half_spread/2 × signal`` shape as
    # the other 4 alphas. Off (alpha=0.0) by default. The OFI
    # accumulator runs unconditionally on the public-WS thread so the
    # signal is always available — flipping ``OFI_RESERVATION_ALPHA``
    # to a non-zero value is sufficient to arm the feature without
    # a restart-needed config reload (next tick picks it up).
    ofi_alpha = float(getattr(settings, "ofi_reservation_alpha", 0.0))
    ofi_shift_applied_bps = 0.0
    if (
        ofi_alpha > 0.0
        and bool(getattr(settings, "ofi_enabled", True))
        and ofi_signal_5s_normalised is not None
        and isinstance(ofi_signal_5s_normalised, (int, float))
        and math.isfinite(float(ofi_signal_5s_normalised))
    ):
        from app.ofi import compute_ofi_reservation_shift_bps
        ofi_shift_applied_bps = compute_ofi_reservation_shift_bps(
            ofi_signal_normalised=float(ofi_signal_5s_normalised),
            half_spread_bps=float(half_spread),
            alpha=ofi_alpha,
            clip_bound=float(getattr(settings, "ofi_signal_clip", 0.95)),
        )
        if abs(ofi_shift_applied_bps) > 1e-9:
            reservation += ofi_shift_applied_bps * ref_price / 10_000.0
            reason_parts.append("ofi_shift")

    # v1.5.248 — escalator's reservation-shift attenuation. When
    # active, pull the reservation back toward mid by
    # ``(1 - reservation_shift_multiplier)`` of the total alpha
    # contribution. mult=1.0 → no change; mult=0.5 → halve the
    # alpha pull; mult=0.0 → reservation reset to mid (no skew).
    # Applied AFTER all alpha contributions so it's a single uniform
    # attenuation of the net reservation shift, and BEFORE the v1.5.230
    # ``max_reservation_shift_bps_from_mid`` clamp so the clamp still
    # bounds any remaining shift.
    if _escalator_active and mid > 0:
        _res_mult = float(getattr(
            no_fill_escalator_output,
            "reservation_shift_multiplier",
            1.0,
        ))
        if abs(_res_mult - 1.0) > 1e-9:
            _shift_so_far_price = float(reservation) - float(mid)
            reservation = float(mid) + _res_mult * _shift_so_far_price

    # v1.5.230 — clamp total reservation shift from mid.
    #
    # All alpha shifts (inventory_skew + trend_drift + ob_imbalance +
    # basis_deviation + flow_score + ofi) have been summed into
    # ``reservation`` by this point. In strong-signal regimes (e.g.
    # heavy trend with trend_drift_alpha = 1.0) the cumulative
    # shift can exceed the per-side half_spread budget — then the
    # bid lands above best_ask or the ask lands below best_bid,
    # the post-only order is rejected at the venue, and that side
    # silently stops filling. MAX_HALF_SPREAD_BPS can't fix this
    # because it caps spread, not shift.
    #
    # This clamp keeps ``abs(reservation − mid) ≤ MAX_RESERVATION_
    # SHIFT_BPS_FROM_MID``. Default 0.0 = no clamp = legacy
    # behaviour (always safe to enable since the clamp can only
    # make the shift SMALLER than what the alphas wanted).
    #
    # Discovered 2026-05-29 on the v1.5.229 Phase 5-prime snapshot:
    # mid=1.7645, reservation=1.7659 (+8.1bp shift, trend_drift
    # dominated), MAX_HALF=4.0 capped the spread at 4 bp, bid
    # landed at 1.7652 > best_ask 1.765 → bid rejected → only 4
    # fills in 75 min. Clamping shift to ±3 bp would have kept
    # the bid at the inside.
    _max_shift_bps = float(
        getattr(settings, "max_reservation_shift_bps_from_mid", 0.0) or 0.0
    )
    _reservation_clamp_active = False
    if _max_shift_bps > 0.0 and mid > 0:
        _shift_bps = (float(reservation) - float(mid)) / float(mid) * 10_000.0
        if abs(_shift_bps) > _max_shift_bps:
            _sign = 1.0 if _shift_bps > 0 else -1.0
            reservation = float(mid) * (1.0 + _sign * _max_shift_bps / 10_000.0)
            _reservation_clamp_active = True
            reason_parts.append(
                f"reservation_clamp:{_shift_bps:+.2f}->{_sign*_max_shift_bps:+.2f}bp"
            )

    # v1.5.209 Phase 8B — Queue-aware inside-spread post. When the
    # bot is deep in the queue (ratio ≥ threshold) AND the operator
    # has flipped the flag, narrow the per-side half-spread by the
    # configured step. Capped at half_spread / 2 so we never cross
    # the reservation. Skips when ratio is None (no own order this
    # side, or accumulator in warmup).
    _queue_inside_post_step_bid = 0.0
    _queue_inside_post_step_ask = 0.0
    if bool(getattr(settings, "queue_aware_inside_post_enabled", False)):
        from app.queue_model import should_post_inside_spread
        _qip_step = float(getattr(settings, "queue_aware_inside_post_step_bps", 1.0))
        _qip_threshold = float(getattr(settings, "queue_aware_inside_post_threshold", 0.7))
        if should_post_inside_spread(
            queue_position_ratio=queue_position_ratio_bid,
            threshold=_qip_threshold,
            enabled=True,
        ):
            cap = max(0.0, bid_half_spread - 1e-6) / 2.0
            _queue_inside_post_step_bid = min(_qip_step, cap)
            bid_half_spread = max(0.0, bid_half_spread - _queue_inside_post_step_bid)
            if _queue_inside_post_step_bid > 1e-9:
                reason_parts.append("queue_inside_post_bid")
        if should_post_inside_spread(
            queue_position_ratio=queue_position_ratio_ask,
            threshold=_qip_threshold,
            enabled=True,
        ):
            cap = max(0.0, ask_half_spread - 1e-6) / 2.0
            _queue_inside_post_step_ask = min(_qip_step, cap)
            ask_half_spread = max(0.0, ask_half_spread - _queue_inside_post_step_ask)
            if _queue_inside_post_step_ask > 1e-9:
                reason_parts.append("queue_inside_post_ask")

    # 1.4.9 gate-to-widening: per-side half-spread. When composition
    # is provided, bid_half_spread / ask_half_spread can differ (the
    # asymmetric gates — microprice, momentum, freshness one-sided,
    # recovery_cooldown — widen only one side). Legacy path
    # (composition=None) has bid_half_spread == ask_half_spread ==
    # half_spread, preserving the pre-1.4.9 symmetric behaviour.
    bid_px = reservation * (1.0 - bid_half_spread / 10_000.0)
    ask_px = reservation * (1.0 + ask_half_spread / 10_000.0)

    notion = settings.quote_notional_usd
    # Toxicity-conditional size reduction. The lower clip floor must be
    # bounded BELOW by the bot's MIN_QUOTE_NOTIONAL_USD self-heal floor —
    # otherwise an aggressive coefficient (e.g. 1.0 on TON) can push the
    # requested notional below the venue spec floor, which the eligibility
    # path then suppresses entirely → execution deadlock → watchdog kill
    # at 600 s of execution_idle (incident 2026-05-08 with 1.1.30 on TON).
    #
    # Floor formula: size_mult >= MIN_QUOTE_NOTIONAL_USD / QUOTE_NOTIONAL_USD
    # so that ``size_mult * quote_notional >= min_quote_notional`` for any
    # toxicity score. The 0.2 absolute floor is kept as a defensive
    # backstop for venues / profiles that haven't set a sensible
    # MIN_QUOTE_NOTIONAL_USD.
    qn = float(settings.quote_notional_usd)
    mqn = float(settings.min_quote_notional_usd)
    fractional_floor = (mqn / qn) if qn > 0.0 else 1.0
    size_mult_floor = max(0.2, min(1.0, fractional_floor))
    size_mult = clip(
        1.0 - float(settings.toxicity_size_reduction_coeff) * toxicity.score,
        size_mult_floor,
        1.0,
    )
    # 1.2.34: track cumulative size_mult after each adjustment for the breakdown.
    _breakdown_size_mult_after_tox = float(size_mult)
    # Vol-spike sizing shrink (BUGS/todo-009.md). Composes
    # multiplicatively with the toxicity-conditional sizing above:
    # toxicity reacts to adverse fill flow, vol-shrink reacts to
    # price velocity — different signals, both wanting smaller
    # orders. Re-clipped to the venue self-heal floor so the
    # min-notional gate doesn't suppress us. Default ``1.0`` (off)
    # leaves size_mult untouched.
    if vol_regime_shrink_factor < 1.0 - 1e-12:
        size_mult = clip(
            size_mult * float(vol_regime_shrink_factor),
            size_mult_floor,
            1.0,
        )
        reason_parts.append("vol_shrink")
    _breakdown_size_mult_after_vol_regime = float(size_mult)
    # 1.2.3: markout-tier size scaler — direct rolling-median 5s
    # markout signal, bypasses the toxicity composite score so the
    # size shrinks track the dashboard's "markout" tier label
    # exactly. Take ``min(toxicity_size_mult, markout_size_mult)``:
    # either signal can shrink, neither can grow. See
    # ``app/markout_size_scaler.py`` for tier defaults.
    if (
        bool(getattr(settings, "markout_size_scaler_enabled", True))
        and recent_markout_5s_median_bps is not None
    ):
        from app.markout_size_scaler import markout_size_mult
        mk_mult = markout_size_mult(
            recent_markout_5s_median_bps,
            mild_threshold_bps=float(
                getattr(settings, "markout_size_scaler_mild_threshold_bps", 0.0)
            ),
            moderate_threshold_bps=float(
                getattr(settings, "markout_size_scaler_moderate_threshold_bps", -1.0)
            ),
            heavy_threshold_bps=float(
                getattr(settings, "markout_size_scaler_heavy_threshold_bps", -3.0)
            ),
            mild_mult=float(getattr(settings, "markout_size_scaler_mild_mult", 0.85)),
            moderate_mult=float(
                getattr(settings, "markout_size_scaler_moderate_mult", 0.5)
            ),
            heavy_mult=float(getattr(settings, "markout_size_scaler_heavy_mult", 0.25)),
            floor=size_mult_floor,
        )
        if mk_mult < size_mult:
            size_mult = clip(mk_mult, size_mult_floor, 1.0)
            reason_parts.append(
                f"markout_shrink:median_5s={recent_markout_5s_median_bps:+.2f}bp"
            )
    _breakdown_size_mult_after_markout = float(size_mult)
    # 1.2.8: basis_regime size shrink. Same ``min`` semantics as
    # the markout shrink above — only ever shrinks, never grows.
    # When ``basis_regime_gate_mode != size_shrink`` the caller
    # passes 1.0 here and this block is a no-op.
    if basis_regime_size_mult < 1.0 - 1e-12:
        clamped = clip(basis_regime_size_mult, size_mult_floor, 1.0)
        if clamped < size_mult:
            size_mult = clamped
            reason_parts.append(
                basis_regime_size_mult_reason
                or f"basis_regime_size_shrink:mult={basis_regime_size_mult:.2f}"
            )
    _breakdown_size_mult_after_basis_regime = float(size_mult)
    # 2026-05-12 codex-#3: fill-burst size shrink. Same ``min``
    # semantics as the other shrinks — clamped to the size_mult
    # floor so we don't pop below venue minimum notional. Active for
    # FILL_BURST_COOLDOWN_SECONDS after a detected burst.
    if fill_burst_size_mult < 1.0 - 1e-12:
        clamped = clip(fill_burst_size_mult, size_mult_floor, 1.0)
        if clamped < size_mult:
            size_mult = clamped
            reason_parts.append(
                f"fill_burst_shrink:mult={fill_burst_size_mult:.2f}"
            )
    # v1.4.112 Phase 1C — regime_controller quote_notional_mult.
    # DEFENSIVE halves the base size (0.5); NORMAL / SHOCK leave it
    # at 1.0. SHOCK keeps the reducing side full-size (the adding
    # side is suppressed by shock_gate's binary eligibility clamp,
    # not by shrinking notional). Min-semantic composition with the
    # other shrinks, clipped to the MIN_QUOTE_NOTIONAL_USD floor.
    if (
        math.isfinite(float(regime_quote_notional_mult))
        and float(regime_quote_notional_mult) < 1.0 - 1e-12
    ):
        rc_clamped = clip(
            size_mult * float(regime_quote_notional_mult),
            size_mult_floor,
            1.0,
        )
        if rc_clamped < size_mult:
            size_mult = rc_clamped
            reason_parts.append(
                f"regime_size_mult:{float(regime_quote_notional_mult):.2f}"
            )
    # v1.5.209 Phase 8B — Queue-aware per-side size shrink. Composes
    # multiplicatively on top of the symmetric size_mult chain above.
    # Per-side because queue position differs per side. Default-off
    # flag keeps multipliers at 1.0 (no shrink) when feature is off.
    # The size_mult_floor applies AFTER this so we don't pop below
    # MIN_QUOTE_NOTIONAL_USD.
    _queue_size_mult_bid = 1.0
    _queue_size_mult_ask = 1.0
    if bool(getattr(settings, "queue_aware_sizing_enabled", False)):
        from app.queue_model import queue_aware_size_multiplier
        _qa_floor = float(getattr(settings, "queue_aware_size_floor", 0.3))
        _qa_decay = float(getattr(settings, "queue_aware_size_decay", 0.7))
        _queue_size_mult_bid = queue_aware_size_multiplier(
            queue_position_ratio=queue_position_ratio_bid,
            floor=_qa_floor,
            decay=_qa_decay,
        )
        _queue_size_mult_ask = queue_aware_size_multiplier(
            queue_position_ratio=queue_position_ratio_ask,
            floor=_qa_floor,
            decay=_qa_decay,
        )
        if _queue_size_mult_bid < 1.0 - 1e-9:
            reason_parts.append("queue_size_shrink_bid")
        if _queue_size_mult_ask < 1.0 - 1e-9:
            reason_parts.append("queue_size_shrink_ask")
    # Per-side effective multipliers clip at size_mult_floor to keep
    # both sides above MIN_QUOTE_NOTIONAL_USD.
    _eff_bid_mult = max(size_mult_floor, size_mult * _queue_size_mult_bid)
    _eff_ask_mult = max(size_mult_floor, size_mult * _queue_size_mult_ask)
    bid_sz = (notion * _eff_bid_mult) / bid_px if bid_px > 0 else 0.0
    ask_sz = (notion * _eff_ask_mult) / ask_px if ask_px > 0 else 0.0

    if active == ActiveSides.ASK_ONLY:
        bid_sz = 0.0
    elif active == ActiveSides.BID_ONLY:
        ask_sz = 0.0
    elif active == ActiveSides.NONE:
        bid_sz = ask_sz = 0.0

    vol_est = vol_bps

    qid = str(uuid.uuid4())

    # 1.2.34: build the spread-tab breakdown — every contribution
    # to this cycle's quote, in one structured snapshot. ~5 µs of
    # work; no measurable bot impact.
    from app.quote_breakdown import QuoteBreakdownSnapshot
    _clock = clock if clock is not None else SystemClock()
    _breakdown = QuoteBreakdownSnapshot(
        ts=_clock.now_utc().isoformat(),
        # Reservation decomposition
        mid_price=float(mid),
        microprice=microprice,
        reservation_reference=_breakdown_reservation_reference,
        reference_fair_price=(
            float(reference_fair_price)
            if reference_fair_price is not None
            and isinstance(reference_fair_price, (int, float))
            and math.isfinite(float(reference_fair_price))
            else None
        ),
        reference_blend_alpha=float(settings.reference_venue_fair_blend_alpha),
        ref_price_anchor=_breakdown_ref_price_anchor,
        inventory_skew_bps=_breakdown_inventory_skew_bps,
        trend_drift_shift_bps=_breakdown_trend_drift_shift_bps,
        ob_imbalance_shift_bps=float(ob_shift_applied_bps),
        basis_deviation_shift_bps=float(basis_shift_applied_bps),
        flow_score_shift_bps=float(flow_shift_applied_bps),
        ofi_shift_bps=float(ofi_shift_applied_bps),
        # v1.5.229 — per-side microprice gate widening. ``composition``
        # is the SpreadComposition kwarg passed in by bot.py; when
        # ``None`` (legacy callers without gate composition) we
        # stamp 0.0/0.0.
        microprice_bid_widen_bps=float(
            getattr(composition, "microprice_bid_bps", 0.0) or 0.0
            if composition is not None else 0.0
        ),
        microprice_ask_widen_bps=float(
            getattr(composition, "microprice_ask_bps", 0.0) or 0.0
            if composition is not None else 0.0
        ),
        # v1.5.230 — reservation-shift clamp telemetry.
        reservation_clamp_active=bool(_reservation_clamp_active),
        reservation_price=float(reservation),
        reservation_delta_from_mid_bps=(
            (float(reservation) - float(mid)) / float(mid) * 10_000.0
            if mid > 0
            else 0.0
        ),
        # Half-spread stack
        base_half_spread_bps=_breakdown_base_half,
        vol_contribution_bps=_breakdown_vol_contrib,
        toxicity_bump_bps=_breakdown_tox_bump,
        toxicity_score_used=_breakdown_tox_score_used,
        toxicity_coeff_used=_breakdown_tox_coeff_used,
        join_depth_autotune_overlay_bps=_breakdown_join_depth_overlay,
        toxicity_soft_trigger_bump_bps=_breakdown_tox_soft_bump,
        toxicity_soft_trigger_active=_breakdown_tox_soft_active,
        vol_regime_bump_bps=_breakdown_vol_regime_bump,
        adaptive_widen_overlay_bps=float(spread_floor_overlay_half_spread_bps),
        # Latched-state context: read from caller's reason hints; the
        # bot.py side stamps the active/reason/remaining from
        # ``state.adaptive_spread_widen_*``. Defaults here are
        # "best-effort from settings" — bot.py replaces them after
        # return when fuller context is available.
        adaptive_widen_active=spread_floor_overlay_half_spread_bps > 1e-9,
        adaptive_widen_reason=None,  # bot.py fills in
        adaptive_widen_seconds_remaining=0.0,  # bot.py fills in
        raw_half_spread_bps=_breakdown_raw_half_spread,
        min_half_spread_bps=float(settings.min_half_spread_bps),
        max_half_spread_bps=float(settings.max_half_spread_bps),
        economic_min_half_spread_bps=_breakdown_economic_floor_final,
        economic_floor_kind=_breakdown_floor_kind,
        target_half_spread_bps=float(half_spread),
        clamp_winner=_breakdown_clamp_winner,
        # Size mult chain
        quote_notional_usd=float(notion),
        size_mult_after_toxicity=_breakdown_size_mult_after_tox,
        size_mult_after_vol_regime=_breakdown_size_mult_after_vol_regime,
        size_mult_after_markout_scaler=_breakdown_size_mult_after_markout,
        size_mult_after_basis_regime=_breakdown_size_mult_after_basis_regime,
        final_size_mult=float(size_mult),
        size_mult_floor=float(size_mult_floor),
        effective_notional_usd=float(notion) * float(size_mult),
        # Quote outputs
        quoted_bid_px=float(bid_px),
        quoted_ask_px=float(ask_px),
        quoted_bid_sz=float(bid_sz),
        quoted_ask_sz=float(ask_sz),
        best_bid=(float(best_bid) if best_bid is not None else None),
        best_ask=(float(best_ask) if best_ask is not None else None),
        # Eligibility — these get refined by the caller after
        # apply_quote_eligibility_to_decision; default to the
        # active_sides value here.
        quote_eligibility="QUOTE_BOTH",  # caller replaces
        quote_eligibility_reason="",  # caller replaces
        active_sides=active.value,
        # Ladder fields populated by the caller (bot.py) after
        # build_ladder() runs; default empty.
        ladder_bids=[],
        ladder_asks=[],
        ladder_requested_levels=1,
        ladder_effective_levels_buy=(0 if active == ActiveSides.ASK_ONLY or active == ActiveSides.NONE else 1),
        ladder_effective_levels_sell=(0 if active == ActiveSides.BID_ONLY or active == ActiveSides.NONE else 1),
        # v1.5.209 Phase 8B queue-aware telemetry
        queue_position_ratio_bid=(
            float(queue_position_ratio_bid)
            if queue_position_ratio_bid is not None
            and isinstance(queue_position_ratio_bid, (int, float))
            and math.isfinite(float(queue_position_ratio_bid))
            else None
        ),
        queue_position_ratio_ask=(
            float(queue_position_ratio_ask)
            if queue_position_ratio_ask is not None
            and isinstance(queue_position_ratio_ask, (int, float))
            and math.isfinite(float(queue_position_ratio_ask))
            else None
        ),
        queue_size_mult_bid=float(_queue_size_mult_bid),
        queue_size_mult_ask=float(_queue_size_mult_ask),
        queue_inside_post_step_bid_bps=float(_queue_inside_post_step_bid),
        queue_inside_post_step_ask_bps=float(_queue_inside_post_step_ask),
    )

    return QuoteDecision(
        ts=_clock.now_utc(),
        symbol=settings.symbol,
        mid_price=mid,
        vol_estimate=vol_est,
        inventory=position_qty,
        reservation_price=reservation,
        target_spread_bps=half_spread * 2.0,
        target_bid=bid_px,
        target_ask=ask_px,
        quoted_bid=bid_px,
        quoted_ask=ask_px,
        quoted_bid_sz=bid_sz,
        quoted_ask_sz=ask_sz,
        active_sides=active,
        toxicity_score=toxicity.score,
        decision_reason=",".join(reason_parts),
        quote_cycle_id=qid,
        spread_floor_overlay_half_spread_bps=float(spread_floor_overlay_half_spread_bps),
        # Record microprice whenever depth is available (regardless of
        # whether the flag caused us to USE it for reservation). Enables
        # shadow-analysis: "what would the reservation have been if the
        # flag were on/off for this cycle?"
        microprice=microprice,
        breakdown=_breakdown,
    )


def _intersect_active_sides_for_eligibility(
    base: ActiveSides,
    cap: QuoteEligibility,
) -> ActiveSides:
    """Inventory/toxicity already chose ``base``; apply eligibility cap (fail-safe: NONE on conflict)."""
    if cap == QuoteEligibility.QUOTE_BOTH:
        return base
    if cap == QuoteEligibility.HOLD_ALL:
        return ActiveSides.NONE
    if cap == QuoteEligibility.QUOTE_BUY_ONLY:
        if base == ActiveSides.ASK_ONLY:
            return ActiveSides.NONE
        if base == ActiveSides.BOTH:
            return ActiveSides.BID_ONLY
        return base
    if cap == QuoteEligibility.QUOTE_SELL_ONLY:
        if base == ActiveSides.BID_ONLY:
            return ActiveSides.NONE
        if base == ActiveSides.BOTH:
            return ActiveSides.ASK_ONLY
        return base
    return base


def apply_quote_eligibility_to_decision(
    decision: QuoteDecision,
    cap: QuoteEligibility,
    *,
    eligibility_reason: str,
) -> QuoteDecision:
    """
    Apply pre-inventory eligibility cap to sizes and active sides.

    QuoteEngine output is unchanged in price space; only sides/sizes are clipped to the cap.
    """
    new_active = _intersect_active_sides_for_eligibility(decision.active_sides, cap)
    bid_sz = decision.quoted_bid_sz
    ask_sz = decision.quoted_ask_sz
    if new_active == ActiveSides.NONE:
        bid_sz = 0.0
        ask_sz = 0.0
    elif new_active == ActiveSides.ASK_ONLY:
        bid_sz = 0.0
    elif new_active == ActiveSides.BID_ONLY:
        ask_sz = 0.0
    reason = decision.decision_reason
    if cap != QuoteEligibility.QUOTE_BOTH:
        reason = f"{reason}|eligibility:{cap.value}"
    return replace(
        decision,
        active_sides=new_active,
        quoted_bid_sz=bid_sz,
        quoted_ask_sz=ask_sz,
        decision_reason=reason,
        quote_eligibility=cap.value,
        quote_eligibility_reason=eligibility_reason[:2000],
    )
