"""Basis-regime IC gate (1.2.2 — analysis-day 2026-05-10;
size-shrink mode added 1.2.8 after the 2026-05-10 wedge).

The bot already runs ``BasisRegimeClassifier`` on every quote
cycle, computing a rolling information coefficient (IC) of basis
deviation vs forward return. Today the classifier output (sign +
IC) is consumed only as a *quote skew* — it nudges
``BASIS_DEVIATION_ALPHA`` to lean reservation prices when a
mean-reversion regime is active.

This gate promotes the same signal to a defensive response: when
the IC magnitude is below ``ic_min_quote_threshold``, the regime
is signal-absent — the classifier sees no relationship between
basis deviation and forward return, which means the bot has no
information about direction either.

Two response modes (operator picks via ``BASIS_REGIME_GATE_MODE``):

* ``hold_all`` (legacy): clamp eligibility to HOLD_ALL — sit out
  entirely. Strictly safer than quoting blind, but on a passive
  market-maker whose income is mostly rebate, this throws away
  rebate income during quiet regimes that may not be truly toxic.
* ``size_shrink`` (default since 1.2.8): keep quoting at reduced
  size (e.g. 0.5×). Preserves rebate income while reducing
  exposure when the cross-venue signal is at noise level. The
  shrink composes multiplicatively with the existing
  toxicity / vol / markout shrinks via ``min`` semantics in
  ``app/quoting.py``.

Decision rule (cheap, stateless):

* ``last_ic`` is None or pair_count below warmup floor → don't gate
  (let the bot trade its existing baseline).
* ``|last_ic| < ic_min_quote_threshold`` → "signal absent" →
  ``hold_all`` mode returns a HOLD_ALL reason; ``size_shrink``
  mode returns ``size_mult_signal_absent``.
* Otherwise → don't gate (regime has detectable structure; the
  existing quote-skew handles the lean).

Symmetric — fires on negative IC (mean-reversion alpha absent) and
positive IC (trend-following alpha absent) the same. The asymmetry
between mean-reversion and trend-following alpha is captured by
the regime sign, which the existing quote-skew handles. This gate
is purely about "is there a signal at all."

Threshold default 0.05: empirically, the snapshot 260510082955
session showed |IC| of 0.41 at peak — a strong signal regime — so
0.05 is well below the typical-active range. Tune higher if the
gate ends up over-firing.
"""

from __future__ import annotations

import math
from typing import Optional


def _signal_absent(
    *,
    last_ic: Optional[float],
    pair_count: int,
    ic_min_quote_threshold: float,
    min_pair_samples: int,
) -> bool:
    """Stateless predicate shared by both response modes."""
    if last_ic is None:
        return False
    if pair_count < min_pair_samples:
        return False
    if not math.isfinite(last_ic):
        return False
    return abs(last_ic) < ic_min_quote_threshold


def evaluate_gate(
    *,
    last_ic: Optional[float],
    pair_count: int,
    ic_min_quote_threshold: float,
    min_pair_samples: int,
    enabled: bool = True,
) -> Optional[str]:
    """Return a non-empty reason string when the gate should
    HOLD_ALL; ``None`` otherwise.

    Stateless — re-evaluates each tick from current classifier
    state. Caller intersects the returned cap with its existing
    eligibility (HOLD_ALL is the most-restrictive cap, so the
    intersect always tightens). Used by ``hold_all`` mode only;
    ``size_shrink`` mode calls ``compute_size_mult`` instead.
    """
    if not enabled:
        return None
    if not _signal_absent(
        last_ic=last_ic,
        pair_count=pair_count,
        ic_min_quote_threshold=ic_min_quote_threshold,
        min_pair_samples=min_pair_samples,
    ):
        return None
    return (
        f"basis_regime_signal_absent|ic={last_ic:+.3f}|pairs={pair_count}"
    )


def compute_size_mult(
    *,
    last_ic: Optional[float],
    pair_count: int,
    ic_min_quote_threshold: float,
    min_pair_samples: int,
    size_mult_signal_absent: float,
    enabled: bool = True,
) -> tuple[float, Optional[str]]:
    """Return ``(size_mult, reason)`` — ``1.0, None`` when the gate
    is inactive, ``size_mult_signal_absent, reason`` when the
    signal is absent.

    Pure function. Caller composes the returned multiplier into the
    size_mult chain with ``min`` semantics (only shrinks, never
    grows), matching the convention used by
    ``markout_size_scaler`` and the toxicity/vol shrinks.

    ``size_mult_signal_absent`` of 0.0 is allowed and means
    "post zero size" — equivalent to HOLD_ALL but expressed in the
    size-shrink pathway. The caller's downstream
    ``MIN_QUOTE_NOTIONAL_USD`` floor will then suppress the side
    via the existing self-heal logic.
    """
    if not enabled:
        return 1.0, None
    if not _signal_absent(
        last_ic=last_ic,
        pair_count=pair_count,
        ic_min_quote_threshold=ic_min_quote_threshold,
        min_pair_samples=min_pair_samples,
    ):
        return 1.0, None
    # Clamp pathological config to [0.0, 1.0] — the gate is
    # supposed to shrink, never grow.
    mult = max(0.0, min(1.0, float(size_mult_signal_absent)))
    reason = (
        f"basis_regime_size_shrink|ic={last_ic:+.3f}|pairs={pair_count}"
        f"|mult={mult:.2f}"
    )
    return mult, reason


# ------------------------------------------------------------------
# Widening contribution (gate-to-widening Phase 1, v1.4.8+)
# ------------------------------------------------------------------

def widening_bps(
    *,
    last_ic: Optional[float],
    pair_count: int,
    ic_min_quote_threshold: float,
    min_pair_samples: int,
    max_half_spread_bps: float,
    widen_bps: float = -1.0,
    enabled: bool = True,
) -> tuple[float, float]:
    """Return ``(bid_bps, ask_bps)`` widening contribution.

    ``widen_bps`` sentinel ``-1.0`` falls back to
    ``max_half_spread_bps`` (gate-equivalent magnitude). Operator
    iterates DOWN per ``plans/gate-to-widening.md`` Phase 2.

    Symmetric — basis regime affects both sides equally (it's a
    reference-venue quality signal, not a directional one).
    """
    if not enabled:
        return (0.0, 0.0)
    if not _signal_absent(
        last_ic=last_ic,
        pair_count=pair_count,
        ic_min_quote_threshold=ic_min_quote_threshold,
        min_pair_samples=min_pair_samples,
    ):
        return (0.0, 0.0)
    cap = max(0.0, float(max_half_spread_bps))
    effective = cap if widen_bps < 0 else min(cap, max(0.0, widen_bps))
    return (effective, effective)
