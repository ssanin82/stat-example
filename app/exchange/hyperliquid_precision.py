"""
Hyperliquid order normalization (tick / lot / minimums).

Perpetuals: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size

Prices must satisfy **both**:
  - at most ``MAX_PERP_DECIMALS_BASE - szDecimals`` decimal places (decimal grid / ``price_tick``);
  - at most ``HL_PERP_MAX_SIG_FIGS`` significant figures, **unless** the price is an integer
    (integers are always allowed regardless of sig fig count).

The decimal grid alone (e.g. ``price_tick=0.01`` for ETH) is **not** sufficient: values like
``2244.95`` sit on the grid but have six significant figures and are rejected by the exchange.
"""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_FLOOR, ROUND_HALF_UP, localcontext
from typing import Optional

from app.exchange.symbol_spec import SymbolSpec

# Hyperliquid perp docs: max significant figures for non-integer prices.
HL_PERP_MAX_SIG_FIGS = 5
MAX_PERP_DECIMALS_BASE = 6

# Stable id for logs / DB: limit prices use meta decimal grid *and* sig-fig cap (not tick alone).
HL_PERP_LIMIT_PRICE_PIPELINE_ID = "hl_perp_decimal_grid_plus_max_sig_figs_nonint"

# Float tolerance after rounding (size path only; prices use Decimal).
_EPS = 1e-12


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def max_price_decimals_perp(sz_decimals: int) -> int:
    """``6 - szDecimals`` for perps (HL docs)."""
    return max(0, MAX_PERP_DECIMALS_BASE - int(sz_decimals))


def price_tick_decimal(sz_decimals: int) -> Decimal:
    """Decimal grid step ``10 ** -(6 - szDecimals)`` for perp limit prices."""
    d = max_price_decimals_perp(sz_decimals)
    return Decimal(1).scaleb(-d)


def decimal_size_step(sz_decimals: int) -> Decimal:
    return Decimal(1).scaleb(-int(sz_decimals))


def _count_sig_figs_positive_non_integer(d: Decimal) -> int:
    """
    Significant figures for positive, non-integral Decimal (HL price semantics).
    Integral fast path is handled by the caller.
    """
    if d <= 0:
        return 0
    t = d.normalize().as_tuple()
    exp = t.exponent
    nd = len(t.digits)
    if exp >= 0:
        return nd + exp
    return nd


def _round_positive_to_n_sig_figs(d: Decimal, n: int) -> Decimal:
    """Round ``d`` (positive) to ``n`` significant figures (half up)."""
    if d <= 0:
        return d
    adj = d.adjusted()
    qexp = adj - n + 1
    quant = Decimal(1).scaleb(qexp)
    return d.quantize(quant, rounding=ROUND_HALF_UP)


def normalize_hyperliquid_perp_limit_price(price: Decimal, sz_decimals: int) -> Decimal:
    """
    Full HL perp limit price: decimal grid + max 5 sig figs for non-integers.

    Source: Hyperliquid "Tick and lot size" (info endpoint ``szDecimals``).
    """
    tick = price_tick_decimal(sz_decimals)
    with localcontext() as ctx:
        ctx.prec = 40
        p = price.quantize(tick, rounding=ROUND_HALF_UP)
        for _ in range(24):
            if p <= 0:
                break
            if p == p.to_integral_value():
                break
            if _count_sig_figs_positive_non_integer(p) <= HL_PERP_MAX_SIG_FIGS:
                break
            p = _round_positive_to_n_sig_figs(p, HL_PERP_MAX_SIG_FIGS)
            p = p.quantize(tick, rounding=ROUND_HALF_UP)
        return p


def price_to_float_for_hyperliquid_wire(px: Decimal, sz_decimals: int) -> float:
    """
    Convert normalized Decimal price to float for SDK ``Exchange.order``.

    Uses string round-trip so binary float does not drift off the HL decimal grid.
    """
    max_d = max_price_decimals_perp(sz_decimals)
    s = format(px, f".{max_d}f").rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    return float(s)


def validate_hyperliquid_perp_limit_for_submit(
    px_float: float,
    sz_decimals: int,
    *,
    source_tick: float,
) -> None:
    """
    Pre-submit checks: Decimal grid divisibility, sig-fig rule, SDK wire serializer.

    Raises ``ValueError`` if the price is not admissible for Hyperliquid perp limits.
    """
    tick_dec = price_tick_decimal(sz_decimals)
    px_dec = Decimal(str(px_float))
    q = px_dec.quantize(tick_dec)
    if q != px_dec:
        raise ValueError(
            f"price not on HL decimal grid: px={px_float!r} tick={tick_dec} szDecimals={sz_decimals} "
            f"(startup decimal_grid_tick={source_tick!r} is only the max-decimals grid; "
            "HL also caps significant figures — see tick-and-lot-size)"
        )
    if px_dec > 0 and px_dec != px_dec.to_integral_value():
        sf = _count_sig_figs_positive_non_integer(px_dec)
        if sf > HL_PERP_MAX_SIG_FIGS:
            raise ValueError(
                f"price has too many significant figures: px={px_float!r} sig_figs={sf} "
                f"max={HL_PERP_MAX_SIG_FIGS} szDecimals={sz_decimals}"
            )
    try:
        from hyperliquid.utils.signing import float_to_wire
    except ImportError:
        return
    float_to_wire(px_float)


def normalize_price(price: float, tick: float) -> float:
    """
    Round limit price to an arbitrary tick (float).

    **Not** sufficient for Hyperliquid perp submission by itself; use
    :func:`normalize_hyperliquid_perp_limit_price` via :func:`normalize_order_pair`
    for orders (adds significant-figure rules).
    """
    if not _finite(price):
        return 0.0
    if not _finite(tick) or tick <= 0:
        return float(price)
    price = float(price)
    tick = float(tick)
    if price <= 0:
        return price
    n = round(price / tick)
    out = n * tick
    return float(f"{out:.12g}")


def normalize_size(size: float, step: float) -> float:
    """Round size down to lot step so we never exceed intended notional."""
    if not _finite(size) or size <= 0:
        return 0.0
    if not _finite(step) or step <= 0:
        return 0.0
    eps = max(1e-12, float(step) * 1e-6)
    n = math.floor(float(size) / float(step) + eps)
    if n < 0:
        n = 0
    return round(n * float(step), 12)


def normalize_size_decimal(size: Decimal, step: Decimal) -> Decimal:
    if size <= 0 or step <= 0:
        return Decimal(0)
    n = (size / step).to_integral_value(rounding=ROUND_FLOOR)
    out = n * step
    return out.quantize(step, rounding=ROUND_FLOOR)


def round_price_and_size_to_grid(
    spec: SymbolSpec,
    price: float,
    size: float,
) -> tuple[Optional[tuple[float, float]], Optional[str]]:
    """Round (price, size) to the venue's tick / size_step, no notional check.

    This is the low-level primitive. It applies only the venue's *structural*
    rules (tick, step, min_size). Min-notional enforcement is deliberately
    *not* here — it lives in :class:`QuoteEngine._build_side`, which is the
    single place that also does size self-heal, so the rounding layer never
    has to reject for an issue the engine could fix.

    Returns ``((px, sz), None)`` on success, or ``(None, reason)`` for
    structurally impossible inputs (NaN, zero tick, below min_size).
    """
    tick = spec.price_tick
    step = spec.size_step
    min_sz = spec.min_size

    if not _finite(price) or not _finite(size):
        return None, "non_finite_price_or_size"
    if not all(_finite(x) and x > 0 for x in (tick, step, min_sz)):
        return None, "invalid_symbol_spec_tick_step_or_min_size"

    # Fallback specs preserve HL semantics because they are only produced by the
    # HL adapter when the live meta fetch failed — treating them as a generic
    # venue would silently change HL price rules.
    is_hl = spec.source in ("hyperliquid_meta", "fallback")
    try:
        with localcontext() as ctx:
            ctx.prec = 40
            if is_hl:
                px_dec = normalize_hyperliquid_perp_limit_price(
                    Decimal(str(price)), spec.sz_decimals
                )
                step_dec = decimal_size_step(spec.sz_decimals)
            else:
                tick_dec = Decimal(str(tick))
                px_dec = Decimal(str(price)).quantize(tick_dec, rounding=ROUND_HALF_UP)
                step_dec = Decimal(str(step))
            sz_dec = normalize_size_decimal(Decimal(str(size)), step_dec)
    except Exception:
        return None, "decimal_normalize_failed"

    if is_hl:
        px = price_to_float_for_hyperliquid_wire(px_dec, spec.sz_decimals)
    else:
        px = float(px_dec)
    sz = float(sz_dec)
    if not _finite(px) or not _finite(sz) or px <= 0 or sz <= 0:
        return None, f"normalized_non_positive px={px} sz={sz}"
    if sz + _EPS < min_sz:
        return None, f"below_min_size normalized_sz={sz} min_size={min_sz}"
    return (px, sz), None


def normalize_order_pair(
    spec: SymbolSpec,
    price: float,
    size: float,
) -> tuple[Optional[tuple[float, float]], Optional[str]]:
    """Round + min-notional check. Kept for callers outside the quote engine.

    The quote engine itself uses :func:`round_price_and_size_to_grid` so it
    can self-heal size when notional rounds below the venue minimum. This
    function is the "strict" variant — it rejects with
    ``below_min_notional_usd`` rather than self-healing, and is appropriate
    for risk / reconcile paths that must not adjust size silently.
    """
    min_notional = spec.min_notional_usd
    if not _finite(min_notional) or min_notional < 0:
        return None, "invalid_min_notional_usd"
    rounded, rej = round_price_and_size_to_grid(spec, price, size)
    if rounded is None:
        return None, rej
    px, sz = rounded
    notion = px * sz
    if notion + _EPS < min_notional:
        return None, (
            f"below_min_notional_usd normalized_notional={notion} "
            f"min_notional_usd={min_notional}"
        )
    return (px, sz), None


def rejection_is_below_min_notional_usd(reason: str | None) -> bool:
    """True if :func:`normalize_order_pair` failed solely due to exchange min notional."""
    return bool(reason) and reason.startswith("below_min_notional_usd")


def normalize_order_for_symbol(
    spec: SymbolSpec,
    price: float,
    size: float,
) -> Optional[tuple[float, float]]:
    """
    Apply HL perp price rules, size step, min size / min notional.
    Returns None if the order is invalid after normalization.
    """
    out, _ = normalize_order_pair(spec, price, size)
    return out


def wire_format_preview_limit_px(px: float) -> str:
    """Exact ``p`` string shape the Python SDK sends (for tests / logs)."""
    try:
        from hyperliquid.utils.signing import float_to_wire

        return float_to_wire(px)
    except Exception:
        rounded = f"{px:.8f}"
        return str(Decimal(rounded).normalize())
