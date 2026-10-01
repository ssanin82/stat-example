"""Phase 2C.3 (v1.5.26) -- regime-mode ladder-pruning attribution counter.

The regime_controller publishes a ``ladder_levels_max`` knob per mode
(NORMAL=None, CALM=None, CAUTIOUS=1, DEFENSIVE=1, SHOCK=0 -> floored
at 1). Phase 4G.8 (v1.4.219) already wired this into the ladder
builder's ``num_levels_per_side`` -- the ladder DOES collapse under
defensive modes. What was MISSING is the drop-attribution: rungs
that never get computed (because effective_levels < cfg_levels)
don't fire the ``on_rung_dropped`` callback, so they were invisible
in the snapshot's ladder_rung_drops counters.

2C.3 ships ``ladder_rung_dropped_regime_mode_total`` on ``BotState``,
bumped explicitly at the call site in bot.py whenever the regime
cap pulls effective_levels below cfg_levels (delta count per tick).

This test verifies the field is present and the snapshot exposes it.
The actual bump happens inside the bot's per-tick path which needs
a heavier harness; the per-tick logic is straightforward (one
conditional bump in the ladder-config-build block) and the
operator can verify in live snapshots that the counter moves
during DEFENSIVE / CAUTIOUS episodes.
"""

from __future__ import annotations

from app.config import Settings
from app.state import BotState


def test_ladder_rung_dropped_regime_mode_total_field_exists():
    """Default-init the counter is 0 and the attribute is accessible."""
    s = Settings()
    st = BotState(s)
    assert hasattr(st, "ladder_rung_dropped_regime_mode_total")
    assert st.ladder_rung_dropped_regime_mode_total == 0


def test_ladder_rung_drops_snapshot_includes_regime_mode_total():
    """``BotState.snapshot_dict()['ladder_rung_drops']`` surfaces
    the new counter so the dashboard / postmortem can read it
    alongside the existing drop categories."""
    s = Settings()
    st = BotState(s)
    snap = st.snapshot_dict()
    drops = snap.get("ladder_rung_drops") or {}
    assert "regime_mode_total" in drops
    assert drops["regime_mode_total"] == 0


def test_regime_mode_total_contributes_to_sum_total():
    """The aggregate ``sum_total`` count includes the regime_mode
    bump so a single sum-comparison covers all drop reasons."""
    s = Settings()
    st = BotState(s)
    st.ladder_rung_dropped_regime_mode_total = 17
    drops = (st.snapshot_dict() or {}).get("ladder_rung_drops") or {}
    # sum_total includes regime_mode_total now.
    assert drops["sum_total"] == 17  # other counters are 0
    assert drops["regime_mode_total"] == 17
