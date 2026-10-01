"""
Quote aging / distance-to-touch helpers.

Runtime passive **quoting** is implemented only in ``app.quote_engine.QuoteEngine``.
``OrderManager`` no longer imports these for the active refresh path; helpers remain here
for ``tests/test_quote_aging.py`` and any future offline analysis.

Phase 0 contract (v1.4.68 wedge-elimination-cleanup) — age-cap semantics
------------------------------------------------------------------------

The settings ``BEHIND_TOUCH_MAX_AGE_SECONDS`` (default 1.5 s) and
``AT_TOUCH_MAX_AGE_SECONDS`` (default 2.5 s) are **wall-lifetime** caps,
not post-ACK caps. They guarantee:

    For every order placed by this bot, the time between the local
    place-dispatch (``ts_sent``) and the cancel-dispatch is bounded
    by the configured cap, irrespective of how slow the venue ACK is.

Pre-Phase-1C (v1.4.67 and earlier), ``resting_age_seconds()`` measured
from ``ts_ack``. That meant a slow ACK could consume the entire cap
window with the order live on the wire and the cap clock not started.
Snapshot v1.4.66-260518-192744 shows orders aging 8-27 s past the cap
because of this.

Phase 1C.4 changes the helper to prefer ``ts_sent`` over ``ts_ack``:

    wall_lifetime = max(now - ts_sent, now - ts_ack)

so hydrated orders (no ``ts_sent``) still age from venue ack-time, and
locally-placed orders age from the moment we hit the wire. The
``hard_cancel_*_reasons`` signal is emitted as soon as either clock
crosses the cap. Phase 1C.1 promotes the evaluation from inside-rung
only to every live slot.

Phase 0 contract — what callers must do
---------------------------------------

* The engine's ``build_quotes`` calls ``compute_aging_signals`` (Phase
  4B) which produces ``hard_cancel_by_slot: dict[(side, lvl), tuple[str, ...]]``.
* ``compute_desired_state`` empties each (side, lvl) entry that has
  non-empty reasons. Reason tag: ``"hard_age_cap:<reasons>:lvl=N"``.
* The reconciler emits ``CancelAction(trigger_reason="hard_age_cap")``
  with ``priority=URGENT``.
* ``app/outbound_dispatch.py`` (Phase 1C.5) lets URGENT cancels preempt
  same-slot amends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from app.config import Settings
from app.enums import ActiveSides, OrderStatus, RiskAction, Side, TouchPlacementMode
from app.models import QuoteDecision, WorkingOrder
from app.utils.math import clip


@dataclass(frozen=True, slots=True)
class SideAgingDiag:
    side: str
    order_age_seconds: Optional[float]
    distance_to_touch_ticks: Optional[float]
    at_touch: bool
    tighten_applied: bool
    reasons: tuple[str, ...]
    raw_target_before: float
    raw_target_after: float
    # When correction to policy target exceeds QUOTE_AGING_TIGHTEN_TICKS per cycle — execution must cancel/replace.
    aging_escalate_reprice: tuple[str, ...] = ()


def inventory_pressure_ratio(position_qty: float, settings: Settings) -> float:
    mx = settings.max_abs_position
    if mx <= 0:
        return 0.0
    return abs(position_qty) / mx


def inventory_pressure_active(settings: Settings, position_qty: float) -> bool:
    r = inventory_pressure_ratio(position_qty, settings)
    thr = float(settings.quote_inventory_pressure_pct)
    floor = float(settings.inventory_fair_touch_relax_min_util_pct)
    if floor > 0:
        thr = max(thr, floor)
    return r + 1e-12 >= thr


def resting_age_seconds(wo: WorkingOrder, now: datetime) -> Optional[float]:
    """Resting age of a working order in seconds.

    v1.4.70 wedge-elimination-cleanup Phase 1C (wall-lifetime cap):
    Returns the WALL LIFETIME — ``max(now - ts_sent, now - ts_ack)``.

    Pre-Phase-1C semantics measured from ``ts_ack`` only, which meant
    a slow venue ACK could consume the entire cap window with the
    order live on the wire and the cap clock not yet started. Snapshot
    v1.4.67-260518-195033 showed orders with 20-33 s sent→ack gaps
    aging past every cap because of this. Snapshot
    v1.4.69-260518-210941 still showed 6 orders with 6-68 s gaps
    despite Phase 1A's WS-buffer fix — the underlying HTTP response
    can simply be slow.

    Status partition:

      * ``ACKED`` / ``PARTIAL`` → resting on the venue. Standard case.
      * ``SENT`` → bot dispatched the place, no ACK yet. Phase 1C
        treats SENT as resting too, measured from ``ts_sent``, so the
        wall-lifetime cap fires even when ACK is missing.
      * ``CANCEL_PENDING`` → cancel issued; not aging (we're trying to
        get rid of it).
      * Terminal (CANCELED/FILLED/REJECTED/DESYNC) → no age.

    Anchor selection (per Phase 0.3 contract):

      * For locally-placed orders: ``ts_sent`` is set at HTTP dispatch.
        Use that as the primary anchor — it's the moment the order
        becomes a venue obligation.
      * For hydrated orders (no ``ts_sent``): ``ts_ack`` is set to the
        venue's ``cTime`` / ``uTime`` and represents the on-wire age
        from the venue's clock. Use that as the fallback.
      * If both are present, use ``max`` (whichever started earlier
        produces the larger age).

    Returns ``None`` when the WO is not in an age-relevant state.
    """
    if wo.status not in (
        OrderStatus.ACKED,
        OrderStatus.PARTIAL,
        OrderStatus.SENT,
    ):
        return None
    sent_age = None
    if wo.ts_sent is not None:
        sent_age = max(0.0, (now - wo.ts_sent).total_seconds())
    ack_age = None
    if wo.ts_ack is not None:
        ack_age = max(0.0, (now - wo.ts_ack).total_seconds())
    if sent_age is None and ack_age is None:
        return None
    if sent_age is None:
        return ack_age
    if ack_age is None:
        return sent_age
    # Both available — use the LARGER (older anchor). For a locally-
    # placed order this is normally ts_sent (slightly before ts_ack).
    return max(sent_age, ack_age)


def at_touch_bid(working_price: float, best_bid: Optional[float], tick: float) -> bool:
    if best_bid is None or not math.isfinite(best_bid) or tick <= 0:
        return False
    if working_price > best_bid + 1e-12:
        return False
    return (best_bid - working_price) <= tick * 0.75


def at_touch_ask(working_price: float, best_ask: Optional[float], tick: float) -> bool:
    if best_ask is None or not math.isfinite(best_ask) or tick <= 0:
        return False
    if working_price < best_ask - 1e-12:
        return False
    return (working_price - best_ask) <= tick * 0.75


def distance_buy_to_touch_ticks(working_price: float, best_bid: Optional[float], tick: float) -> Optional[float]:
    if best_bid is None or not math.isfinite(best_bid) or tick <= 0:
        return None
    if working_price >= best_bid:
        return 0.0
    return (best_bid - working_price) / tick


def distance_sell_to_touch_ticks(working_price: float, best_ask: Optional[float], tick: float) -> Optional[float]:
    if best_ask is None or not math.isfinite(best_ask) or tick <= 0:
        return None
    if working_price <= best_ask:
        return 0.0
    return (working_price - best_ask) / tick


def _bid_fair_cap(decision: QuoteDecision, best_bid: Optional[float], inv_pressure: bool, position_qty: float) -> float:
    """Max passive bid price (no cross); reservation caps fundamental aggression."""
    if best_bid is None or not math.isfinite(best_bid):
        return decision.reservation_price
    cap = min(best_bid, decision.reservation_price)
    if inv_pressure and position_qty < 0:
        cap = best_bid
    return cap


def _ask_fair_floor(decision: QuoteDecision, best_ask: Optional[float], inv_pressure: bool, position_qty: float) -> float:
    """Min passive ask price (no cross)."""
    if best_ask is None or not math.isfinite(best_ask):
        return decision.reservation_price
    floor = max(best_ask, decision.reservation_price)
    if inv_pressure and position_qty > 0:
        floor = best_ask
    return floor


def clamp_buy_post_only(px: float, best_bid: Optional[float]) -> float:
    if best_bid is not None and math.isfinite(best_bid):
        return min(px, best_bid)
    return px


def clamp_sell_post_only(px: float, best_ask: Optional[float]) -> float:
    if best_ask is not None and math.isfinite(best_ask):
        return max(px, best_ask)
    return px


def touch_placement_mode_for_side(
    *,
    risk_action: RiskAction,
    active_sides: ActiveSides,
    two_sided_quote: bool,
    book_fresh_for_placement: bool,
    skip_new_place_inventory_side: bool,
    normal_mm_contract_active: bool = True,
) -> TouchPlacementMode:
    """
    Placement / reprice mode: normal two-sided MM disables hard touch-band re-anchoring.

    ``skip_new_place_inventory_side`` is True when inventory bias deprioritizes this side
    while the other side is preferred (still two-sided quoting intent).
    """
    if risk_action in (RiskAction.FLATTEN, RiskAction.KILL, RiskAction.CANCEL_ALL):
        return TouchPlacementMode.EMERGENCY
    if not book_fresh_for_placement:
        return TouchPlacementMode.DEGRADED_FALLBACK
    if not two_sided_quote or active_sides != ActiveSides.BOTH:
        return TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION
    if risk_action != RiskAction.ALLOW:
        return TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION
    if skip_new_place_inventory_side:
        return TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION
    if not normal_mm_contract_active:
        return TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION
    return TouchPlacementMode.NORMAL_TWO_SIDED_MM


def widen_two_sided_if_collapsed_to_one_tick(
    bid_px: float,
    ask_px: float,
    *,
    want_bid: bool,
    want_ask: bool,
    tick: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
) -> tuple[float, float]:
    """
    When two-sided raw targets sit at the minimum one-tick spread (often after post-only clamp),
    step bid down and ask up by one tick so quotes rest slightly less aggressive.

    Re-applies post-only clamps so we never cross the touch; if clamps collapse spread again,
    returns the original prices.
    """
    if not (want_bid and want_ask and bid_px > 0 and ask_px > 0):
        return bid_px, ask_px
    if tick <= 0 or not math.isfinite(tick):
        return bid_px, ask_px
    spread = float(ask_px) - float(bid_px)
    if spread > tick * 1.0001:
        return bid_px, ask_px
    nb = clamp_buy_post_only(float(bid_px) - tick, best_bid)
    na = clamp_sell_post_only(float(ask_px) + tick, best_ask)
    if na - nb <= tick * 1.0001:
        return bid_px, ask_px
    return nb, na


def _bid_aging_trigger_targets(
    *,
    max_d: float,
    best_bid: float,
    tick: float,
    bid_cap: float,
    dist_violation: bool,
    age_violation: bool,
) -> float:
    """
    One-cycle join target: distance violations pull to band floor first; age-only pulls to touch.
    When both fire, band floor is used so a single QUOTE_AGING_TIGHTEN_TICKS step can suffice.
    """
    band_floor = best_bid - max_d * tick
    if dist_violation:
        return float(min(bid_cap, band_floor))
    if age_violation:
        return float(min(bid_cap, best_bid))
    return float(min(bid_cap, band_floor))


def _ask_aging_trigger_targets(
    *,
    max_d: float,
    best_ask: float,
    tick: float,
    ask_floor: float,
    reference_ask_px: float,
    dist_violation: bool,
    age_violation: bool,
) -> float:
    """
    Mirrored to bid distance handling: pull a high ask *down* toward best_ask + band, not up
    toward max(floor, band_ceil) which pinned far-above targets and skipped SELL placement.
    """
    band_ceil = best_ask + max_d * tick
    if dist_violation:
        return float(max(ask_floor, min(float(reference_ask_px), band_ceil)))
    if age_violation:
        return float(max(ask_floor, best_ask))
    return float(max(ask_floor, band_ceil))


def adjust_bid_target_for_aging(
    settings: Settings,
    decision: QuoteDecision,
    *,
    bid_px_model: float,
    best_bid: Optional[float],
    tick: float,
    working: Optional[WorkingOrder],
    now: datetime,
    position_qty: float,
    enforce_touch_distance_band: bool = True,
) -> tuple[float, SideAgingDiag]:
    """Raise bid toward a policy target; escalate to cancel/replace if one cycle cannot close the gap safely."""
    reasons: list[str] = []
    raw_after = bid_px_model
    age_s: Optional[float] = None
    dist_ticks: Optional[float] = None
    at = False
    tighten = False
    esc: tuple[str, ...] = ()

    if not settings.quote_aging_enabled or tick <= 0:
        return raw_after, SideAgingDiag(
            side="BUY",
            order_age_seconds=age_s,
            distance_to_touch_ticks=dist_ticks,
            at_touch=at,
            tighten_applied=False,
            reasons=(),
            raw_target_before=bid_px_model,
            raw_target_after=raw_after,
            aging_escalate_reprice=(),
        )

    inv_p = inventory_pressure_active(settings, position_qty)
    bid_cap = _bid_fair_cap(decision, best_bid, inv_p, position_qty)
    max_d = float(settings.quote_max_distance_to_touch_ticks)
    step_max = float(settings.quote_aging_tighten_ticks)
    max_age_s = float(settings.quote_aging_max_age_seconds)

    if working and working.side == Side.BUY and working.status in (OrderStatus.ACKED, OrderStatus.PARTIAL):
        age_s = resting_age_seconds(working, now)
        dist_ticks = distance_buy_to_touch_ticks(working.price, best_bid, tick)
        at = at_touch_bid(working.price, best_bid, tick)
        dist_viol = (
            dist_ticks is not None
            and dist_ticks > max_d
            and enforce_touch_distance_band
        )
        age_viol = (
            age_s is not None
            and age_s > max_age_s
            and not at
            and dist_ticks is not None
            and dist_ticks > 0.5
        )
        if not dist_viol and not age_viol:
            pass
        elif best_bid is None or not math.isfinite(best_bid):
            pass
        else:
            if dist_viol:
                reasons.append("distance_ticks")
            if age_viol:
                reasons.append("age_seconds")
            target_px = _bid_aging_trigger_targets(
                max_d=max_d,
                best_bid=float(best_bid),
                tick=tick,
                bid_cap=bid_cap,
                dist_violation=dist_viol,
                age_violation=age_viol,
            )
            target_px = clamp_buy_post_only(target_px, best_bid)
            need_px = target_px - working.price
            max_step_px = step_max * tick
            if need_px <= 1e-15:
                raw_after = bid_px_model
            elif need_px > max_step_px + tick * 1e-9:
                raw_after = bid_px_model
                esc = ("exceeds_safe_aging_step",)
            else:
                raw_after = min(bid_cap, target_px)
                raw_after = clamp_buy_post_only(raw_after, best_bid)
                tighten = raw_after >= bid_px_model + tick * 0.25 - 1e-12
    else:
        if best_bid is not None and math.isfinite(best_bid):
            dist_new = distance_buy_to_touch_ticks(bid_px_model, best_bid, tick)
            if (
                dist_new is not None
                and dist_new > max_d
                and enforce_touch_distance_band
            ):
                reasons.append("distance_ticks_new_order")
                target_px = _bid_aging_trigger_targets(
                    max_d=max_d,
                    best_bid=float(best_bid),
                    tick=tick,
                    bid_cap=bid_cap,
                    dist_violation=True,
                    age_violation=False,
                )
                target_px = clamp_buy_post_only(target_px, best_bid)
                need_px = target_px - bid_px_model
                max_step_px = step_max * tick
                if need_px > max_step_px + tick * 1e-9:
                    raw_after = min(bid_cap, bid_px_model + max_step_px)
                    raw_after = clamp_buy_post_only(raw_after, best_bid)
                    tighten = raw_after >= bid_px_model + tick * 0.25 - 1e-12
                else:
                    raw_after = target_px
                    tighten = raw_after >= bid_px_model + tick * 0.25 - 1e-12

    return raw_after, SideAgingDiag(
        side="BUY",
        order_age_seconds=age_s,
        distance_to_touch_ticks=(
            dist_ticks if working else distance_buy_to_touch_ticks(bid_px_model, best_bid, tick)
        ),
        at_touch=(at if working else at_touch_bid(bid_px_model, best_bid, tick)),
        tighten_applied=tighten,
        reasons=tuple(reasons),
        raw_target_before=bid_px_model,
        raw_target_after=raw_after,
        aging_escalate_reprice=esc,
    )


def adjust_ask_target_for_aging(
    settings: Settings,
    decision: QuoteDecision,
    *,
    ask_px_model: float,
    best_ask: Optional[float],
    tick: float,
    working: Optional[WorkingOrder],
    now: datetime,
    position_qty: float,
    enforce_touch_distance_band: bool = True,
) -> tuple[float, SideAgingDiag]:
    """Lower ask toward a policy target; escalate when the gap exceeds one safe step."""
    reasons: list[str] = []
    raw_after = ask_px_model
    age_s: Optional[float] = None
    dist_ticks: Optional[float] = None
    at = False
    tighten = False
    esc: tuple[str, ...] = ()

    if not settings.quote_aging_enabled or tick <= 0:
        return raw_after, SideAgingDiag(
            side="SELL",
            order_age_seconds=age_s,
            distance_to_touch_ticks=dist_ticks,
            at_touch=at,
            tighten_applied=False,
            reasons=(),
            raw_target_before=ask_px_model,
            raw_target_after=raw_after,
            aging_escalate_reprice=(),
        )

    inv_p = inventory_pressure_active(settings, position_qty)
    ask_floor = _ask_fair_floor(decision, best_ask, inv_p, position_qty)
    max_d = float(settings.quote_max_distance_to_touch_ticks)
    step_max = float(settings.quote_aging_tighten_ticks)
    max_age_s = float(settings.quote_aging_max_age_seconds)

    if working and working.side == Side.SELL and working.status in (OrderStatus.ACKED, OrderStatus.PARTIAL):
        age_s = resting_age_seconds(working, now)
        dist_ticks = distance_sell_to_touch_ticks(working.price, best_ask, tick)
        at = at_touch_ask(working.price, best_ask, tick)
        dist_viol = (
            dist_ticks is not None
            and dist_ticks > max_d
            and enforce_touch_distance_band
        )
        age_viol = (
            age_s is not None
            and age_s > max_age_s
            and not at
            and dist_ticks is not None
            and dist_ticks > 0.5
        )
        if not dist_viol and not age_viol:
            pass
        elif best_ask is None or not math.isfinite(best_ask):
            pass
        else:
            if dist_viol:
                reasons.append("distance_ticks")
            if age_viol:
                reasons.append("age_seconds")
            target_px = _ask_aging_trigger_targets(
                max_d=max_d,
                best_ask=float(best_ask),
                tick=tick,
                ask_floor=ask_floor,
                reference_ask_px=float(working.price),
                dist_violation=dist_viol,
                age_violation=age_viol,
            )
            target_px = clamp_sell_post_only(target_px, best_ask)
            need_px = working.price - target_px
            max_step_px = step_max * tick
            if need_px <= 1e-15:
                raw_after = ask_px_model
            elif need_px > max_step_px + tick * 1e-9:
                raw_after = ask_px_model
                esc = ("exceeds_safe_aging_step",)
            else:
                raw_after = max(ask_floor, target_px)
                raw_after = clamp_sell_post_only(raw_after, best_ask)
                tighten = raw_after <= ask_px_model - tick * 0.25 + 1e-12
    else:
        if best_ask is not None and math.isfinite(best_ask):
            dist_new = distance_sell_to_touch_ticks(ask_px_model, best_ask, tick)
            if (
                dist_new is not None
                and dist_new > max_d
                and enforce_touch_distance_band
            ):
                reasons.append("distance_ticks_new_order")
                target_px = _ask_aging_trigger_targets(
                    max_d=max_d,
                    best_ask=float(best_ask),
                    tick=tick,
                    ask_floor=ask_floor,
                    reference_ask_px=float(ask_px_model),
                    dist_violation=True,
                    age_violation=False,
                )
                target_px = clamp_sell_post_only(target_px, best_ask)
                need_px = ask_px_model - target_px
                max_step_px = step_max * tick
                if need_px > max_step_px + tick * 1e-9:
                    raw_after = max(ask_floor, ask_px_model - max_step_px)
                    raw_after = clamp_sell_post_only(raw_after, best_ask)
                    tighten = raw_after <= ask_px_model - tick * 0.25 + 1e-12
                else:
                    raw_after = target_px
                    tighten = raw_after <= ask_px_model - tick * 0.25 + 1e-12

    return raw_after, SideAgingDiag(
        side="SELL",
        order_age_seconds=age_s,
        distance_to_touch_ticks=(
            dist_ticks if working else distance_sell_to_touch_ticks(ask_px_model, best_ask, tick)
        ),
        at_touch=(at if working else at_touch_ask(ask_px_model, best_ask, tick)),
        tighten_applied=tighten,
        reasons=tuple(reasons),
        raw_target_before=ask_px_model,
        raw_target_after=raw_after,
        aging_escalate_reprice=esc,
    )


def apply_inventory_high_adding_side_buffer(
    *,
    settings: Settings,
    position_qty: float,
    bid_px: float,
    ask_px: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
) -> tuple[float, float, dict[str, Any]]:
    """Push the *adding-side* quote behind the touch when inventory pressure
    is high. Returns possibly-adjusted ``(bid_px, ask_px, telemetry)``.

    Adding side is the one that increases ``|position|`` if filled:
    bid when ``position > 0`` (long), ask when ``position < 0`` (short).
    No-op when ``inventory_high_adding_side_buffer_ticks <= 0`` (default
    off) or when ``|position|/max_abs_position`` is below
    ``quote_inventory_pressure_pct``.

    Symmetric to the existing inventory-skew machinery (which shifts the
    *reservation price*) but acts on the *placement* — once skew has
    pulled reservation away, this buffer ensures the actual quote is
    AT LEAST one tick (or N ticks) behind the touch, even if the
    market has tightened since reservation was computed. The reducing
    side is left untouched so it can still join the touch and fill
    out the inventory.
    """
    buf = float(getattr(settings, "inventory_high_adding_side_buffer_ticks", 0.0))
    if buf <= 0.0 or tick <= 0.0:
        return bid_px, ask_px, {}
    if not inventory_pressure_active(settings, position_qty):
        return bid_px, ask_px, {}

    diag: dict[str, Any] = {}
    new_bid = bid_px
    new_ask = ask_px
    offset = buf * tick

    # Bid is the adding side when position is long.
    if position_qty > 0 and best_bid is not None and math.isfinite(best_bid):
        target = float(best_bid) - offset
        if new_bid > target:
            new_bid = target
            diag["inventory_high_adding_side_bid_buffered"] = True
            diag["inventory_high_adding_side_bid_buffer_ticks"] = buf

    # Ask is the adding side when position is short.
    if position_qty < 0 and best_ask is not None and math.isfinite(best_ask):
        target = float(best_ask) + offset
        if new_ask < target:
            new_ask = target
            diag["inventory_high_adding_side_ask_buffered"] = True
            diag["inventory_high_adding_side_ask_buffer_ticks"] = buf

    return new_bid, new_ask, diag


def markout_adverse_bps_for_side(
    *,
    side: Side,
    working: Optional[WorkingOrder],
    mid: Optional[float],
) -> Optional[float]:
    """Bps of adverse drift between a resting quote and current mid.
    Positive = adverse (fill against us would be free money for the
    counterparty). ``None`` when there is no eligible resting order or
    mid is unusable.

    For a resting BID at ``px`` and current ``mid``:
      adverse_bps = (px - mid) / mid * 10_000
    A bid above mid is adverse to us — anyone selling to us collects
    the gap instantly. For a resting ASK at ``px``:
      adverse_bps = (mid - px) / mid * 10_000
    An ask below mid is adverse to us in the same way.
    """
    if working is None or mid is None or not math.isfinite(float(mid)) or float(mid) <= 0.0:
        return None
    if working.status not in (OrderStatus.ACKED, OrderStatus.PARTIAL):
        return None
    px = float(working.price)
    m = float(mid)
    if side == Side.BUY:
        return (px - m) / m * 10_000.0
    return (m - px) / m * 10_000.0


class MarkoutAdverseTracker:
    """Per-side timer for sustained adverse-markout drift on resting quotes.

    Returns ``True`` (cancel signal) when ``adverse_bps >= threshold``
    has persisted for ``>= duration_seconds`` on the *same* resting
    order (identified by ``order_id_local``). Resets when the resting
    order changes or when ``adverse_bps`` drops below threshold.

    State is per-side, not per-symbol — owned by ``QuoteEngine``
    (one instance per symbol). Threshold/duration come from settings
    on each evaluation, so config reloads take effect immediately.
    """

    __slots__ = ("_first_breach",)

    def __init__(self) -> None:
        self._first_breach: dict[Side, tuple[Optional[str], Optional[datetime]]] = {
            Side.BUY: (None, None),
            Side.SELL: (None, None),
        }

    def evaluate(
        self,
        *,
        side: Side,
        working: Optional[WorkingOrder],
        adverse_bps: Optional[float],
        threshold_bps: float,
        duration_seconds: float,
        now: datetime,
    ) -> tuple[bool, Optional[float]]:
        """Returns ``(should_cancel, breach_elapsed_seconds_or_None)``.

        Tuple form so callers can surface elapsed time in telemetry
        even when ``should_cancel`` is False (i.e. the timer is armed
        but hasn't yet hit duration).
        """
        oid = working.order_id_local if working is not None else None
        prev_oid, prev_ts = self._first_breach[side]

        # Order changed (or none) — reset and stop.
        if oid != prev_oid:
            self._first_breach[side] = (oid, None)
            prev_ts = None

        # Feature off, or no eligible resting order — clear timer, no fire.
        if oid is None or threshold_bps <= 0.0 or duration_seconds <= 0.0:
            self._first_breach[side] = (oid, None)
            return False, None

        # Below threshold — reset timer.
        if adverse_bps is None or adverse_bps < threshold_bps:
            self._first_breach[side] = (oid, None)
            return False, None

        # Above threshold — arm timer if not armed.
        if prev_ts is None:
            self._first_breach[side] = (oid, now)
            return False, 0.0

        elapsed = max(0.0, (now - prev_ts).total_seconds())
        return elapsed >= duration_seconds, elapsed


def aging_diag_to_dict(d: SideAgingDiag) -> dict[str, Any]:
    return {
        "side": d.side,
        "order_age_seconds": d.order_age_seconds,
        "distance_to_touch_ticks": d.distance_to_touch_ticks,
        "at_touch": d.at_touch,
        "tighten_applied": d.tighten_applied,
        "reasons": list(d.reasons),
        "raw_target_before": d.raw_target_before,
        "raw_target_after": d.raw_target_after,
        "aging_escalate_reprice": list(d.aging_escalate_reprice),
    }


def hard_reprice_reasons_buy(
    settings: Settings,
    wo: WorkingOrder,
    *,
    best_bid: Optional[float],
    tick: float,
    now: datetime,
    enforce_touch_distance_band: bool = True,
) -> tuple[str, ...]:
    """
    Mandatory cancel-and-replace triggers for a resting bid (OR semantics).
    Uses QUOTE_AGING_MAX_AGE_SECONDS and QUOTE_MAX_DISTANCE_TO_TOUCH_TICKS.

    v1.4.70 Phase 1C: SENT orders are now eligible for the wall-
    lifetime cap (via ``resting_age_seconds`` which uses ts_sent).
    Pre-Phase-1C this filter excluded SENT, which meant a slow-ACK
    order aged on the wire un-capped (the v1.4.67/v1.4.69
    snapshot-observed wedge). Distance-to-touch and at_touch checks
    still require a known price relative to the venue best — they
    apply naturally to ACKED/PARTIAL only.
    """
    if not settings.quote_aging_enabled or tick <= 0:
        return ()
    if wo.side != Side.BUY:
        return ()
    if wo.status not in (
        OrderStatus.ACKED,
        OrderStatus.PARTIAL,
        OrderStatus.SENT,
    ):
        return ()
    reasons: list[str] = []
    age_s = resting_age_seconds(wo, now)
    dist = distance_buy_to_touch_ticks(wo.price, best_bid, tick)
    at = at_touch_bid(wo.price, best_bid, tick)
    if age_s is not None and age_s > float(settings.quote_aging_max_age_seconds) and not at:
        reasons.append("order_age_seconds")
    # 2026-05-12 codex-#2: separate, longer age cap that DOES apply to
    # at-touch orders. The standard ``order_age_seconds`` rule (above)
    # exempts at-touch via ``and not at`` to preserve queue priority.
    # This second rule catches genuinely-stale at-touch orders (30+
    # seconds, etc.) before they get adversely selected when the market
    # walks past them. Default ``at_touch_max_age_seconds=0`` disables.
    at_touch_max_age = float(getattr(settings, "at_touch_max_age_seconds", 0.0))
    if (
        at_touch_max_age > 0.0
        and age_s is not None
        and age_s > at_touch_max_age
        and at
    ):
        reasons.append("at_touch_order_age_seconds")
    # 2026-05-13 todo-019 Part C: stricter BEHIND-touch age cap
    # mirrors the at-touch one above but with the at/not-at gate
    # inverted. The regular ``quote_aging_max_age_seconds`` (used
    # in ``order_age_seconds`` above) is the general behind-touch
    # cap, but at typical 6 s settings it doesn't catch the
    # 0.85-1.75 s adverse-selection window that dominates losses
    # on tight-tick symbols. This second cap fires hard-reprice on
    # any behind-touch order older than the threshold.
    # Default 0.0 = DISABLED → backwards-compatible.
    behind_touch_max_age = float(
        getattr(settings, "behind_touch_max_age_seconds", 0.0)
    )
    if (
        behind_touch_max_age > 0.0
        and age_s is not None
        and age_s > behind_touch_max_age
        and not at
    ):
        reasons.append("behind_touch_order_age_seconds")
    if (
        enforce_touch_distance_band
        and dist is not None
        and dist > float(settings.quote_max_distance_to_touch_ticks)
    ):
        reasons.append("distance_to_touch_ticks")
    return tuple(reasons)


def hard_reprice_reasons_sell(
    settings: Settings,
    wo: WorkingOrder,
    *,
    best_ask: Optional[float],
    tick: float,
    now: datetime,
    enforce_touch_distance_band: bool = True,
) -> tuple[str, ...]:
    """Mirror of ``hard_reprice_reasons_buy`` for the sell side.

    v1.4.70 Phase 1C: SENT orders eligible (wall-lifetime cap).
    """
    if not settings.quote_aging_enabled or tick <= 0:
        return ()
    if wo.side != Side.SELL:
        return ()
    if wo.status not in (
        OrderStatus.ACKED,
        OrderStatus.PARTIAL,
        OrderStatus.SENT,
    ):
        return ()
    reasons: list[str] = []
    age_s = resting_age_seconds(wo, now)
    dist = distance_sell_to_touch_ticks(wo.price, best_ask, tick)
    at = at_touch_ask(wo.price, best_ask, tick)
    if age_s is not None and age_s > float(settings.quote_aging_max_age_seconds) and not at:
        reasons.append("order_age_seconds")
    # 2026-05-12 codex-#2: separate at-touch age cap (mirror of the
    # BUY-side rule).
    at_touch_max_age = float(getattr(settings, "at_touch_max_age_seconds", 0.0))
    if (
        at_touch_max_age > 0.0
        and age_s is not None
        and age_s > at_touch_max_age
        and at
    ):
        reasons.append("at_touch_order_age_seconds")
    # 2026-05-13 todo-019 Part C: stricter BEHIND-touch age cap
    # (mirror of the BUY-side rule). See companion comment in
    # ``hard_reprice_reasons_buy``.
    behind_touch_max_age = float(
        getattr(settings, "behind_touch_max_age_seconds", 0.0)
    )
    if (
        behind_touch_max_age > 0.0
        and age_s is not None
        and age_s > behind_touch_max_age
        and not at
    ):
        reasons.append("behind_touch_order_age_seconds")
    if (
        enforce_touch_distance_band
        and dist is not None
        and dist > float(settings.quote_max_distance_to_touch_ticks)
    ):
        reasons.append("distance_to_touch_ticks")
    return tuple(reasons)


def _apply_min_half_spread_px_vs_mid(
    *,
    is_buy: bool,
    px: float,
    decision: QuoteDecision,
    min_half_spread_px: Optional[float],
    best_bid: Optional[float],
    best_ask: Optional[float],
    floor_mid_px: Optional[float] = None,
) -> float:
    """
    Cap aggressiveness vs mid so passive prices respect the effective economic half-spread
    width as :func:`app.quoting.compute_effective_min_half_spread_bps` (post-only clamps preserved).

    Used on normal two-sided finalize and on symmetric rescue ``out_r`` so working quotes
    do not sit inside the floor vs spread-floor mid.
    """
    if min_half_spread_px is None or min_half_spread_px <= 0:
        return float(px)
    mid = floor_mid_px if floor_mid_px is not None else decision.mid_price
    if not isinstance(mid, (int, float)) or not math.isfinite(float(mid)) or float(mid) <= 0:
        return float(px)
    m = float(mid)
    if is_buy:
        out = min(float(px), m - min_half_spread_px)
        return clamp_buy_post_only(out, best_bid)
    out = max(float(px), m + min_half_spread_px)
    return clamp_sell_post_only(out, best_ask)


def _economic_floor_target_px_vs_mid(
    *,
    is_buy: bool,
    px: float,
    decision: QuoteDecision,
    min_half_spread_px: Optional[float],
    floor_mid_px: Optional[float] = None,
) -> float:
    """
    Economic floor target before post-only clamp.
    """
    if min_half_spread_px is None or min_half_spread_px <= 0:
        return float(px)
    mid = floor_mid_px if floor_mid_px is not None else decision.mid_price
    if not isinstance(mid, (int, float)) or not math.isfinite(float(mid)) or float(mid) <= 0:
        return float(px)
    m = float(mid)
    if is_buy:
        return min(float(px), m - min_half_spread_px)
    return max(float(px), m + min_half_spread_px)


def _half_spread_bps_vs_mid_buy(bid_px: float, mid: Optional[float]) -> Optional[float]:
    if mid is None or not isinstance(mid, (int, float)) or not math.isfinite(float(mid)) or float(mid) <= 0:
        return None
    m = float(mid)
    return (m - float(bid_px)) / m * 10_000.0


def _half_spread_bps_vs_mid_ask(ask_px: float, mid: Optional[float]) -> Optional[float]:
    if mid is None or not isinstance(mid, (int, float)) or not math.isfinite(float(mid)) or float(mid) <= 0:
        return None
    m = float(mid)
    return (float(ask_px) - m) / m * 10_000.0


def _normal_mm_touch_distance_max_half_spread_bps(
    *,
    mid: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    tick: float,
    max_distance_to_touch_ticks: float,
) -> Optional[float]:
    """
    Max symmetric half-spread (bps vs mid) such that bid/ask at mid ± half_px are within
    max_distance_to_touch_ticks of best bid / best ask respectively.
    """
    if (
        best_bid is None
        or best_ask is None
        or not math.isfinite(best_bid)
        or not math.isfinite(best_ask)
        or tick <= 0
        or not math.isfinite(mid)
        or mid <= 0
    ):
        return None
    bb, ba = float(best_bid), float(best_ask)
    if ba <= bb + 1e-12:
        return None
    mx_px = max(0.0, float(max_distance_to_touch_ticks)) * float(tick)
    # bid at mid - h: need bb - (mid - h) <= mx_px  =>  h <= mid - bb + mx_px
    # ask at mid + h: need (mid + h) - ba <= mx_px  =>  h <= ba - mid + mx_px
    d_bid = float(mid) - bb + mx_px
    d_ask = ba - float(mid) + mx_px
    cap_px = min(d_bid, d_ask)
    if cap_px <= 0:
        return 0.0
    return cap_px / float(mid) * 10_000.0


def compute_normal_mm_market_capped_half_spread_bps(
    settings: Settings,
    *,
    model_half_spread_bps: float,
    best_bid: Optional[float],
    best_ask: Optional[float],
    mid: float,
    tick: float,
) -> tuple[float, dict[str, Any]]:
    """
    Tighten model half-spread for normal two-sided mode against live BBO microstructure.

    max_normal_half merges:
    - market anchor: max(market_half + buffer_bps, min_competitive), optional max-competitive ceiling
    - touch geometry: symmetric mid ± half must be within NORMAL_MM_MAX_DISTANCE_TO_TOUCH_TICKS of touch
    capped_half = min(model, max_normal), then clipped to global min/max half-spread rails.
    """
    mn = float(settings.min_half_spread_bps)
    mx = float(settings.max_half_spread_bps)
    m_mid = float(mid)
    if not math.isfinite(m_mid) or m_mid <= 0:
        c = float(clip(model_half_spread_bps, mn, mx))
        return c, {
            "live_market_spread_bps": None,
            "model_half_spread_bps": float(model_half_spread_bps),
            "market_capped_half_spread_bps": c,
            "normal_mm_touch_buffer_ticks": float(settings.normal_mm_touch_buffer_ticks),
            "was_market_spread_cap_applied": False,
            "normal_mm_market_cap_applied": False,
            "normal_mm_touch_distance_cap_applied": False,
            "normal_mm_touch_distance_max_half_spread_bps": None,
            "quoting_regime_reason": "invalid_mid",
            "max_normal_half_spread_bps": None,
        }

    live_spread_bps: Optional[float] = None
    market_half = 0.0
    if (
        best_bid is not None
        and best_ask is not None
        and math.isfinite(best_bid)
        and math.isfinite(best_ask)
        and float(best_ask) > float(best_bid) + 1e-12
    ):
        live_spread_bps = (float(best_ask) - float(best_bid)) / m_mid * 10_000.0
        market_half = live_spread_bps / 2.0

    buf_bps = 0.0
    if tick > 0 and math.isfinite(tick):
        buf_bps = float(settings.normal_mm_touch_buffer_ticks) * float(tick) / m_mid * 10_000.0

    max_normal_market = max(
        market_half + buf_bps,
        float(settings.normal_mm_min_competitive_half_spread_bps),
    )
    opt_top = float(settings.normal_mm_max_competitive_half_spread_bps)
    if opt_top > 0:
        max_normal_market = min(max_normal_market, opt_top)

    touch_bps_cap = _normal_mm_touch_distance_max_half_spread_bps(
        mid=m_mid,
        best_bid=best_bid,
        best_ask=best_ask,
        tick=tick,
        max_distance_to_touch_ticks=float(settings.normal_mm_max_distance_to_touch_ticks),
    )
    max_normal = float(max_normal_market)
    touch_cap_applied = False
    if touch_bps_cap is not None:
        before = max_normal
        max_normal = min(max_normal, float(touch_bps_cap))
        touch_cap_applied = max_normal + 1e-12 < before

    raw_capped = min(float(model_half_spread_bps), float(max_normal))
    was_market_cap = min(float(model_half_spread_bps), float(max_normal_market)) + 1e-9 < float(
        model_half_spread_bps
    )
    capped_final = float(clip(raw_capped, mn, mx))
    if touch_cap_applied:
        regime = "touch_distance_cap"
    elif was_market_cap:
        regime = "market_anchor_cap"
    else:
        regime = "model_half_spread"
    return capped_final, {
        "live_market_spread_bps": live_spread_bps,
        "model_half_spread_bps": float(model_half_spread_bps),
        "market_capped_half_spread_bps": raw_capped,
        "normal_mm_touch_buffer_ticks": float(settings.normal_mm_touch_buffer_ticks),
        "was_market_spread_cap_applied": was_market_cap,
        "normal_mm_market_cap_applied": was_market_cap,
        "normal_mm_touch_distance_cap_applied": touch_cap_applied,
        "normal_mm_touch_distance_max_half_spread_bps": touch_bps_cap,
        "quoting_regime_reason": regime,
        "max_normal_half_spread_bps": float(max_normal),
    }


def finalize_buy_placement_touch_distance(
    settings: Settings,
    decision: QuoteDecision,
    raw_px: float,
    norm_px: Optional[float],
    *,
    best_bid: Optional[float],
    tick: float,
    position_qty: float,
    best_ask: Optional[float] = None,
    symmetric_two_sided_rescue: bool = False,
    min_half_spread_px: Optional[float] = None,
    spread_floor_mid_px: Optional[float] = None,
    placement_mode: TouchPlacementMode = TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION,
    normal_mm_placement_telemetry: Optional[dict[str, Any]] = None,
) -> tuple[Optional[float], Optional[str], str, dict[str, Any]]:
    """
    Passive BUY placement transform: touch band (legacy) or execution-only (normal two-sided).

    Returns (px, skip_reason, rescue_tag, placement_diag).
    ``rescue_tag`` is ``symmetric_rescue`` when a two-sided rescue path produced ``px``
    after the primary anchor would have hit ``distance_guard`` (non-normal modes only).
    """
    rescue_tag = ""
    mid_ref = spread_floor_mid_px
    if mid_ref is None and isinstance(decision.mid_price, (int, float)):
        mid_ref = float(decision.mid_price) if math.isfinite(float(decision.mid_price)) else None

    def _diag(
        *,
        post_only_px: float,
        final_px: Optional[float],
        touch_band_applied: bool,
        touch_band_reason: str,
        post_only_adjustment_applied: Optional[bool] = None,
        economic_floor_requested: Optional[bool] = None,
        economic_floor_blocked_by_post_only: Optional[bool] = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "placement_mode": placement_mode.value,
            "side": "BUY",
            "raw_target_px": float(raw_px),
            "normalized_ref_px": float(norm_px) if norm_px is not None and math.isfinite(norm_px) else None,
            "post_only_adjusted_px": float(post_only_px),
            "final_submitted_px": float(final_px) if final_px is not None else None,
            "spread_floor_mid_px": float(mid_ref) if mid_ref is not None else None,
            "effective_half_spread_bps_vs_mid": _half_spread_bps_vs_mid_buy(
                float(final_px) if final_px is not None else float(post_only_px), mid_ref
            ),
            "touch_band_rule_applied": touch_band_applied,
            "touch_band_rule_reason": touch_band_reason,
        }
        if normal_mm_placement_telemetry:
            row.update(normal_mm_placement_telemetry)
        if placement_mode == TouchPlacementMode.NORMAL_TWO_SIDED_MM:
            row["touch_band_rule_applied"] = False
            row["touch_band_rule_reason"] = "normal_two_sided_mm_direct"
            row["post_only_adjustment_applied"] = bool(post_only_adjustment_applied)
            row["economic_floor_requested"] = bool(economic_floor_requested)
            row["economic_floor_blocked_by_post_only"] = bool(economic_floor_blocked_by_post_only)
            if bool(economic_floor_blocked_by_post_only):
                row["touch_band_rule_reason"] = "normal_two_sided_mm_post_only_blocked_economic_floor"
            if final_px is not None and best_bid is not None:
                row["final_distance_to_touch_ticks_bid"] = distance_buy_to_touch_ticks(
                    float(final_px), best_bid, tick
                )
            else:
                row["final_distance_to_touch_ticks_bid"] = None
            row["final_distance_to_touch_ticks_ask"] = None
        return row

    if not settings.quote_aging_enabled or tick <= 0:
        return raw_px, None, rescue_tag, _diag(
            post_only_px=float(raw_px),
            final_px=float(raw_px),
            touch_band_applied=False,
            touch_band_reason="quote_aging_disabled_or_tick",
        )
    ref = raw_px
    if norm_px is not None and math.isfinite(norm_px) and norm_px > 0:
        ref = float(norm_px)
    if best_bid is None or not math.isfinite(best_bid):
        return ref, None, rescue_tag, _diag(
            post_only_px=float(ref),
            final_px=float(ref),
            touch_band_applied=False,
            touch_band_reason="no_best_bid",
        )

    inv_p = inventory_pressure_active(settings, position_qty)
    cap = _bid_fair_cap(decision, best_bid, inv_p, position_qty)
    bb = float(best_bid)
    decede = max(0, int(settings.post_only_touch_buffer_ticks) - 1)
    aggr_bid_cap = bb - decede * tick

    if placement_mode == TouchPlacementMode.NORMAL_TWO_SIDED_MM:
        out = min(ref, cap, aggr_bid_cap)
        out_po = clamp_buy_post_only(out, bb)
        post_po = float(out_po)
        floor_target = _economic_floor_target_px_vs_mid(
            is_buy=True,
            px=out_po,
            decision=decision,
            min_half_spread_px=min_half_spread_px,
            floor_mid_px=spread_floor_mid_px,
        )
        out = clamp_buy_post_only(float(floor_target), bb)
        post_only_adjusted = out < floor_target - 1e-12
        economic_floor_requested = floor_target < out_po - 1e-12
        economic_floor_blocked = economic_floor_requested and out > floor_target + 1e-12
        return float(out), None, rescue_tag, _diag(
            post_only_px=post_po,
            final_px=float(out),
            touch_band_applied=False,
            touch_band_reason="normal_two_sided_mm_direct",
            post_only_adjustment_applied=post_only_adjusted,
            economic_floor_requested=economic_floor_requested,
            economic_floor_blocked_by_post_only=economic_floor_blocked,
        )

    mx = float(settings.quote_max_distance_to_touch_ticks)
    floor_touch = bb - mx * tick
    out = max(ref, floor_touch)
    out = min(out, cap, aggr_bid_cap)
    out = clamp_buy_post_only(out, bb)
    post_after_touch = float(out)
    d = distance_buy_to_touch_ticks(out, bb, tick)
    if d is not None and d > mx + 1e-9:
        if (
            symmetric_two_sided_rescue
            and best_ask is not None
            and math.isfinite(best_ask)
            and best_ask > bb + 1e-12
        ):
            out_r = max(ref, floor_touch)
            out_r = min(out_r, aggr_bid_cap)
            out_r = clamp_buy_post_only(out_r, bb)
            out_r = _apply_min_half_spread_px_vs_mid(
                is_buy=True,
                px=out_r,
                decision=decision,
                min_half_spread_px=min_half_spread_px,
                best_bid=best_bid,
                best_ask=best_ask,
                floor_mid_px=spread_floor_mid_px,
            )
            d2 = distance_buy_to_touch_ticks(out_r, bb, tick)
            if d2 is not None and d2 <= mx + 1e-9:
                return out_r, None, "symmetric_rescue", _diag(
                    post_only_px=float(out_r),
                    final_px=float(out_r),
                    touch_band_applied=True,
                    touch_band_reason="symmetric_two_sided_band_rescue",
                )
        return None, "distance_guard", rescue_tag, _diag(
            post_only_px=post_after_touch,
            final_px=None,
            touch_band_applied=True,
            touch_band_reason="distance_guard_max_touch_band",
        )
    return out, None, rescue_tag, _diag(
        post_only_px=post_after_touch,
        final_px=float(out),
        touch_band_applied=True,
        touch_band_reason="touch_band_primary_anchor",
    )


def finalize_ask_placement_touch_distance(
    settings: Settings,
    decision: QuoteDecision,
    raw_px: float,
    norm_px: Optional[float],
    *,
    best_ask: Optional[float],
    tick: float,
    position_qty: float,
    best_bid: Optional[float] = None,
    symmetric_two_sided_rescue: bool = False,
    min_half_spread_px: Optional[float] = None,
    spread_floor_mid_px: Optional[float] = None,
    placement_mode: TouchPlacementMode = TouchPlacementMode.ONE_SIDED_INVENTORY_REDUCTION,
    normal_mm_placement_telemetry: Optional[dict[str, Any]] = None,
) -> tuple[Optional[float], Optional[str], str, dict[str, Any]]:
    """
    Passive SELL placement transform: touch band (legacy) or execution-only (normal two-sided).
    """
    rescue_tag = ""
    mid_ref = spread_floor_mid_px
    if mid_ref is None and isinstance(decision.mid_price, (int, float)):
        mid_ref = float(decision.mid_price) if math.isfinite(float(decision.mid_price)) else None

    def _diag(
        *,
        post_only_px: float,
        final_px: Optional[float],
        touch_band_applied: bool,
        touch_band_reason: str,
        post_only_adjustment_applied: Optional[bool] = None,
        economic_floor_requested: Optional[bool] = None,
        economic_floor_blocked_by_post_only: Optional[bool] = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "placement_mode": placement_mode.value,
            "side": "SELL",
            "raw_target_px": float(raw_px),
            "normalized_ref_px": float(norm_px) if norm_px is not None and math.isfinite(norm_px) else None,
            "post_only_adjusted_px": float(post_only_px),
            "final_submitted_px": float(final_px) if final_px is not None else None,
            "spread_floor_mid_px": float(mid_ref) if mid_ref is not None else None,
            "effective_half_spread_bps_vs_mid": _half_spread_bps_vs_mid_ask(
                float(final_px) if final_px is not None else float(post_only_px), mid_ref
            ),
            "touch_band_rule_applied": touch_band_applied,
            "touch_band_rule_reason": touch_band_reason,
        }
        if normal_mm_placement_telemetry:
            row.update(normal_mm_placement_telemetry)
        if placement_mode == TouchPlacementMode.NORMAL_TWO_SIDED_MM:
            row["touch_band_rule_applied"] = False
            row["touch_band_rule_reason"] = "normal_two_sided_mm_direct"
            row["post_only_adjustment_applied"] = bool(post_only_adjustment_applied)
            row["economic_floor_requested"] = bool(economic_floor_requested)
            row["economic_floor_blocked_by_post_only"] = bool(economic_floor_blocked_by_post_only)
            if bool(economic_floor_blocked_by_post_only):
                row["touch_band_rule_reason"] = "normal_two_sided_mm_post_only_blocked_economic_floor"
            row["final_distance_to_touch_ticks_bid"] = None
            if final_px is not None and best_ask is not None:
                row["final_distance_to_touch_ticks_ask"] = distance_sell_to_touch_ticks(
                    float(final_px), best_ask, tick
                )
            else:
                row["final_distance_to_touch_ticks_ask"] = None
        return row

    if not settings.quote_aging_enabled or tick <= 0:
        return raw_px, None, rescue_tag, _diag(
            post_only_px=float(raw_px),
            final_px=float(raw_px),
            touch_band_applied=False,
            touch_band_reason="quote_aging_disabled_or_tick",
        )
    ref = raw_px
    if norm_px is not None and math.isfinite(norm_px) and norm_px > 0:
        ref = float(norm_px)
    if best_ask is None or not math.isfinite(best_ask):
        return ref, None, rescue_tag, _diag(
            post_only_px=float(ref),
            final_px=float(ref),
            touch_band_applied=False,
            touch_band_reason="no_best_ask",
        )

    inv_p = inventory_pressure_active(settings, position_qty)
    floor_fair = _ask_fair_floor(decision, best_ask, inv_p, position_qty)
    ba = float(best_ask)
    decede = max(0, int(settings.post_only_touch_buffer_ticks) - 1)
    aggr_ask_floor = ba + decede * tick

    if placement_mode == TouchPlacementMode.NORMAL_TWO_SIDED_MM:
        out = max(ref, floor_fair, aggr_ask_floor)
        out_po = clamp_sell_post_only(out, ba)
        post_po = float(out_po)
        floor_target = _economic_floor_target_px_vs_mid(
            is_buy=False,
            px=out_po,
            decision=decision,
            min_half_spread_px=min_half_spread_px,
            floor_mid_px=spread_floor_mid_px,
        )
        out = clamp_sell_post_only(float(floor_target), ba)
        post_only_adjusted = out > floor_target + 1e-12
        economic_floor_requested = floor_target > out_po + 1e-12
        economic_floor_blocked = economic_floor_requested and out < floor_target - 1e-12
        return float(out), None, rescue_tag, _diag(
            post_only_px=post_po,
            final_px=float(out),
            touch_band_applied=False,
            touch_band_reason="normal_two_sided_mm_direct",
            post_only_adjustment_applied=post_only_adjusted,
            economic_floor_requested=economic_floor_requested,
            economic_floor_blocked_by_post_only=economic_floor_blocked,
        )

    mx = float(settings.quote_max_distance_to_touch_ticks)
    ceil_touch = ba + mx * tick
    out = min(ref, ceil_touch)
    out = max(out, floor_fair, aggr_ask_floor)
    out = clamp_sell_post_only(out, ba)
    post_after_touch = float(out)
    d = distance_sell_to_touch_ticks(out, ba, tick)
    if d is not None and d > mx + 1e-9:
        if (
            symmetric_two_sided_rescue
            and best_bid is not None
            and math.isfinite(best_bid)
            and ba > float(best_bid) + 1e-12
        ):
            out_r = min(ref, ceil_touch)
            out_r = max(out_r, aggr_ask_floor)
            out_r = clamp_sell_post_only(out_r, ba)
            out_r = _apply_min_half_spread_px_vs_mid(
                is_buy=False,
                px=out_r,
                decision=decision,
                min_half_spread_px=min_half_spread_px,
                best_bid=best_bid,
                best_ask=best_ask,
                floor_mid_px=spread_floor_mid_px,
            )
            d2 = distance_sell_to_touch_ticks(out_r, ba, tick)
            if d2 is not None and d2 <= mx + 1e-9:
                return out_r, None, "symmetric_rescue", _diag(
                    post_only_px=float(out_r),
                    final_px=float(out_r),
                    touch_band_applied=True,
                    touch_band_reason="symmetric_two_sided_band_rescue",
                )
        return None, "distance_guard", rescue_tag, _diag(
            post_only_px=post_after_touch,
            final_px=None,
            touch_band_applied=True,
            touch_band_reason="distance_guard_max_touch_band",
        )
    return out, None, rescue_tag, _diag(
        post_only_px=post_after_touch,
        final_px=float(out),
        touch_band_applied=True,
        touch_band_reason="touch_band_primary_anchor",
    )


def preserve_bid_queue(
    wo: Optional[WorkingOrder],
    *,
    best_bid: Optional[float],
    tick: float,
    bid_target_norm: Optional[float],
    bid_px_model: float,
    mid: float,
    reprice_threshold_bps: float,
    aging_reasons: tuple[str, ...],
) -> bool:
    if wo is None or wo.status not in (OrderStatus.ACKED, OrderStatus.PARTIAL):
        return False
    if aging_reasons:
        return False
    if not at_touch_bid(wo.price, best_bid, tick):
        return False
    if bid_target_norm is None:
        return False
    if mid <= 0:
        return False
    # Model or norm may diverge from touch under skew/rounding; at-touch + no aging_reasons
    # is sufficient to preserve queue position.
    return True


def preserve_ask_queue(
    wo: Optional[WorkingOrder],
    *,
    best_ask: Optional[float],
    tick: float,
    ask_target_norm: Optional[float],
    ask_px_model: float,
    mid: float,
    reprice_threshold_bps: float,
    aging_reasons: tuple[str, ...],
) -> bool:
    if wo is None or wo.status not in (OrderStatus.ACKED, OrderStatus.PARTIAL):
        return False
    if aging_reasons:
        return False
    if not at_touch_ask(wo.price, best_ask, tick):
        return False
    if ask_target_norm is None:
        return False
    if mid <= 0:
        return False
    # Model or norm may diverge from touch under skew/rounding; at-touch + no aging_reasons
    # is sufficient to preserve queue position.
    return True
