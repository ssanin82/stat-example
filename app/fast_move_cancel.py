"""Phase 4E (v1.4.165) — Target-venue fast-move cancel detector.

Symmetric defensive-cancel signal: when the **target venue** (the
trading venue the bot has resting orders on) shows a fast mid-price
move beyond a configured threshold, we cancel resting orders on the
side the mid is running AWAY from. Complementary to the existing
reference-venue cancel-on-move (which uses an external venue's
mid as a leading indicator): this trigger catches fast moves that
originate on the target venue itself and that the reference signal
may not lead.

Naming convention. The settings + state fields use the
``target_venue_*`` prefix to parallel the existing
``reference_venue_*`` settings. No specific exchange identifiers
appear in this module — the bot's bridge to a new trading venue or
new reference venue does not require any change here.

The detector is a pure function over the 500 ms mid-return
kinematic the bot already computes every tick. No new computation;
only a new consumption.
"""

from __future__ import annotations

import math
from typing import Optional

from app.enums import Side


def detect_target_venue_fast_move(
    *,
    mid_return_bps: Optional[float],
    threshold_bps: float,
) -> Optional[Side]:
    """Return the **side to cancel** (the side the local mid is
    running away from), or ``None`` if no fast move.

    Sign convention:
      * ``mid_return_bps > +threshold`` → target-venue mid is moving
        UP fast → ASKs are now stale-low → return ``Side.SELL`` so
        the caller cancels resting **asks**.
      * ``mid_return_bps < -threshold`` → mid moving DOWN fast → BIDs
        are stale-high → return ``Side.BUY`` so caller cancels
        resting **bids**.
      * Anything else → ``None`` (no fast move, no cancel).

    ``threshold_bps <= 0`` disables the detector (returns ``None``
    always). This is the default — feature is opt-in via a positive
    threshold in the profile.

    ``mid_return_bps`` is allowed to be ``None`` (kinematics signal
    not yet warmed up) — returns ``None``. Non-finite values (NaN /
    inf from a degenerate market) also return ``None``.

    Pure function: no side effects, no state. Caller dispatches the
    actual cancel.
    """
    if threshold_bps <= 0.0:
        return None
    if mid_return_bps is None:
        return None
    try:
        r = float(mid_return_bps)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(r):
        return None
    if r >= float(threshold_bps):
        return Side.SELL  # cancel ASKs (price running up past them)
    if r <= -float(threshold_bps):
        return Side.BUY   # cancel BIDs (price running down past them)
    return None
