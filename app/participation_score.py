"""v1.5.207 — Phase 4C.5 participation score.

A single continuous number in ``[0, 1]`` per side that summarises
how willing the bot is to quote that side given the current
expected-net-edge picture:

* **1.0** — full participation. Edge is healthy (≥ ``full_edge_bps``).
* **soft..1** — marginal-positive territory. Quote, but the next
  release will optionally dampen the spread based on score.
* **hard..soft** — dampen-band territory. Edge is marginally
  negative; bot keeps quoting with a widened spread.
* **0..hard** — refuse-band territory. Edge is too negative; bot
  withholds the side.
* **0.0** — fully refused.

Why this matters. The existing per-side refuse + dampen gates
(``expected_edge_side_refuse``, ``negative_expectancy_dampen``)
are discrete: a side is either refused, dampened, or quoting
normally. Three states. The participation score is the **continuous
analogue** of those three states: a single number that tells the
operator at a glance how the bot is treating each side. Useful for:

* Per-tick dashboard read — one number per side, color-coded.
* Postmortem: did the bot's participation correlate with realised
  markout? (Higher score → less adverse markout would mean the
  score is predictive — Phase 4C.5.d acceptance.)
* Future behavior: replace the discrete refuse/dampen with a
  continuous size+spread scaler driven by the score. **Not in
  v1.5.207** — initial release is observation-only. The bot's
  refuse/dampen decisions still come from the existing gates.

Pure-function helpers. State lives on ``BotState``
(``participation_score_bid`` / ``_ask`` per-tick scalars +
rolling-window deques for the dashboard mean).
"""

from __future__ import annotations

from typing import Optional


# Default thresholds. Operator-tunable via env knobs:
#   PARTICIPATION_SCORE_SOFT_THRESHOLD
#   PARTICIPATION_SCORE_HARD_THRESHOLD
#   PARTICIPATION_SCORE_FULL_EDGE_BPS
_DEFAULT_SOFT_THRESHOLD = 0.7
_DEFAULT_HARD_THRESHOLD = 0.3
_DEFAULT_FULL_EDGE_BPS = 3.0


def compute_participation_score(
    *,
    expected_edge_bps: Optional[float],
    refuse_floor_bps: float,
    dampen_floor_bps: float,
    full_edge_bps: float = _DEFAULT_FULL_EDGE_BPS,
    soft_threshold: float = _DEFAULT_SOFT_THRESHOLD,
    hard_threshold: float = _DEFAULT_HARD_THRESHOLD,
) -> Optional[float]:
    """Map per-side ``expected_edge_bps`` to a score in ``[0, 1]``.

    Piecewise-linear interpolation across three regions:

    * ``edge >= full_edge_bps``         → ``1.0``                full
    * ``dampen_floor < edge < full``    → ``soft..1``           marginal-positive
    * ``refuse_floor < edge <= dampen`` → ``hard..soft``        dampen band
    * ``edge <= refuse_floor``          → ``0.0``               refused

    The thresholds ``soft_threshold`` and ``hard_threshold`` are the
    score boundaries between regions — they're NOT edge bps. The
    edge-to-score mapping interpolates linearly between the bps
    boundaries (``refuse_floor`` / ``dampen_floor`` / ``full_edge``)
    and the corresponding score landmarks (``0`` / ``hard`` /
    ``soft`` / ``1.0``).

    Returns ``None`` when ``expected_edge_bps`` is ``None`` (caller
    treats as "no signal this tick"). Always returns finite ``[0, 1]``
    otherwise.

    Defensive against pathological config: if the bps thresholds are
    inverted (``refuse >= dampen`` or ``dampen >= full``), returns
    ``None`` and the caller surfaces "score unavailable".
    """
    if expected_edge_bps is None:
        return None
    try:
        e = float(expected_edge_bps)
    except (TypeError, ValueError):
        return None
    if e != e:  # NaN
        return None

    rf = float(refuse_floor_bps)
    df = float(dampen_floor_bps)
    fe = float(full_edge_bps)
    if not (rf < df < fe):
        # Inverted / collapsed thresholds — refuse to compute rather
        # than emit a garbage score.
        return None

    soft = float(soft_threshold)
    hard = float(hard_threshold)
    if not (0.0 <= hard < soft <= 1.0):
        return None

    if e >= fe:
        return 1.0
    if e > df:
        # marginal-positive: linear soft..1
        frac = (e - df) / (fe - df)
        return soft + (1.0 - soft) * frac
    if e >= rf:
        # dampen band: linear hard..soft. Note ``e >= rf`` (not ``>``)
        # so the boundary edge == refuse_floor maps to ``hard``, which
        # is consistent with the existing refuse evaluator semantics
        # in ``evaluate_per_side_expected_edge_suppression`` ("refused
        # only when edge STRICTLY less than refuse_threshold").
        frac = (e - rf) / (df - rf)
        return hard + (soft - hard) * frac
    # e < refuse_floor → refused
    return 0.0


def participation_action(
    score: Optional[float],
    *,
    soft_threshold: float = _DEFAULT_SOFT_THRESHOLD,
    hard_threshold: float = _DEFAULT_HARD_THRESHOLD,
) -> str:
    """Map score → discrete action label.

    * ``score >= soft_threshold`` → ``"quote"``  (full participation)
    * ``hard <= score < soft``    → ``"dampen"`` (marginal; widen spread)
    * ``score < hard_threshold``  → ``"refuse"`` (withhold side)
    * ``score is None``           → ``"unknown"``

    Pure mapping — no side effects.
    """
    if score is None:
        return "unknown"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "unknown"
    if s >= float(soft_threshold):
        return "quote"
    if s >= float(hard_threshold):
        return "dampen"
    return "refuse"


def is_score_consistent_with_existing_decision(
    score: Optional[float],
    *,
    was_refused: bool,
    was_dampened: bool,
    soft_threshold: float = _DEFAULT_SOFT_THRESHOLD,
    hard_threshold: float = _DEFAULT_HARD_THRESHOLD,
) -> bool:
    """Shadow-mode check: does the score's action match what the
    existing refuse/dampen gates decided?

    Returns ``True`` when they agree (or score is None — no opinion).
    Returns ``False`` when they disagree — operator should investigate
    (one of them is wrong about the regime). The bot's actual behavior
    in v1.5.207 is driven by the EXISTING gates; this check is
    diagnostic.

    Disagreement is rendered in the log so the operator can audit
    the score's calibration before flipping a future feature flag
    that makes the score drive behavior.
    """
    action = participation_action(
        score,
        soft_threshold=soft_threshold,
        hard_threshold=hard_threshold,
    )
    if action == "unknown":
        # No score this tick → can't check.
        return True
    existing = (
        "refuse" if was_refused
        else "dampen" if was_dampened
        else "quote"
    )
    return action == existing
