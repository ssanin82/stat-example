"""Phase 4C.3 mini (v1.4.161) — per-side realised-edge side suppression.

Out-of-order delivery of the maturity-doc's "single biggest gap"
(Phase 4C of the defense plan: economics-first quoting). This narrow
piece targets the failure mode from snapshot
``v1.4.157-260520-213540-prod.okx.ton.usdt.perp``: a resting SELL was
picked off as TON began a +88 bp rally, generating a -31.83 bps 5 s
markout that none of the util-gated existing defences could catch in
time.

Pure-function tests; integration with the bot's tick loop (via the
new ``state.realised_edge_side_suppress`` + the two new
``compute_quote_decision`` kwargs ``realised_edge_suppress_bid/ask``)
is covered by the regression sweep on ``test_quoting.py``.
"""

from __future__ import annotations

from app.enums import Side
from app.realised_edge_side_suppress import RealisedEdgeSideSuppressGate


def _gate(**overrides) -> RealisedEdgeSideSuppressGate:
    base = dict(
        threshold_bps=-4.0,
        cooldown_seconds=60.0,
        min_fills=4,
        window_size=20,
        favorable_exit_enabled=True,
        clear_band_mult=0.5,
        favorable_exit_dwell_seconds=10.0,
    )
    base.update(overrides)
    return RealisedEdgeSideSuppressGate(**base)


# ---------------------------------------------------------------------------
# Enable / disable + warmup
# ---------------------------------------------------------------------------


def test_disabled_when_threshold_zero() -> None:
    g = _gate(threshold_bps=0.0)
    assert not g.enabled()
    g.note_fill(side=Side.SELL, markout_5s_bps=-50.0,
                rebate_bps=1.0, now_mono=100.0)
    assert not g.is_suppressed(Side.SELL, 100.0)


def test_does_not_fire_below_min_fills() -> None:
    """Even a brutal single fill cannot arm before warmup window
    fills."""
    g = _gate(threshold_bps=-4.0, min_fills=4)
    g.note_fill(side=Side.SELL, markout_5s_bps=-30.0,
                rebate_bps=1.0, now_mono=100.0)
    assert not g.is_suppressed(Side.SELL, 100.0)


def test_ignores_missing_markout() -> None:
    g = _gate()
    for _ in range(5):
        g.note_fill(side=Side.SELL, markout_5s_bps=None,
                    rebate_bps=1.0, now_mono=100.0)
    assert not g.is_suppressed(Side.SELL, 100.0)


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


def test_fires_on_v1_4_157_kill_shot_pattern() -> None:
    """Replay of the v1.4.157-260520-213540 snapshot's SELL-side
    pattern. 4 SELL fills with markouts -2.46 / +2.46 / +2.45 /
    -31.83 (and 1.0 bp rebate) → net edges -1.46/+3.46/+3.45/-30.83
    → mean = -6.35 bps < -4.0 threshold → fires."""
    g = _gate(threshold_bps=-4.0, min_fills=4, cooldown_seconds=60.0)
    markouts = [-2.46, 2.46, 2.45, -31.83]
    for mk in markouts:
        g.note_fill(side=Side.SELL, markout_5s_bps=mk,
                    rebate_bps=1.0, now_mono=100.0)
    assert g.is_suppressed(Side.SELL, 100.0)
    # BUY unaffected.
    assert not g.is_suppressed(Side.BUY, 100.0)
    snap = g.snapshot_dict(100.0)
    assert snap["sell"]["fire_count"] == 1
    assert snap["sell"]["last_trigger_mean_bps"] < -4.0


def test_does_not_fire_when_trailing_mean_above_threshold() -> None:
    g = _gate(threshold_bps=-4.0, min_fills=4)
    # All modestly adverse but mean = -2.0 + 1.0 = -1.0 > -4.0.
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-2.0,
                    rebate_bps=1.0, now_mono=100.0)
    assert not g.is_suppressed(Side.SELL, 100.0)


def test_cooldown_expires_naturally() -> None:
    g = _gate(threshold_bps=-4.0, min_fills=4,
              cooldown_seconds=30.0,
              favorable_exit_enabled=False)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    assert g.is_suppressed(Side.SELL, 100.0)
    assert g.is_suppressed(Side.SELL, 129.9)
    assert not g.is_suppressed(Side.SELL, 130.5)


def test_per_side_independence() -> None:
    g = _gate(threshold_bps=-4.0, min_fills=4)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    assert g.is_suppressed(Side.SELL, 100.0)
    assert not g.is_suppressed(Side.BUY, 100.0)
    # Now arm BUY independently.
    for _ in range(4):
        g.note_fill(side=Side.BUY, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    assert g.is_suppressed(Side.BUY, 100.0)
    assert g.is_suppressed(Side.SELL, 100.0)


# ---------------------------------------------------------------------------
# Favorable-exit predicate
# ---------------------------------------------------------------------------


def test_favorable_exit_clears_when_mean_recovers() -> None:
    """threshold=-4, mult=0.5 → clear band -2.0. After arming with
    mean=-9, feed 4 fills at +2 each → mean=+2 > -2 → predicate
    holds; clears via favorable after the dwell."""
    g = _gate(threshold_bps=-4.0, clear_band_mult=0.5,
              favorable_exit_dwell_seconds=10.0,
              cooldown_seconds=120.0)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    assert g.is_suppressed(Side.SELL, 100.0)
    # Feed 4 favorable fills at t=110 (recovery).
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=+2.0,
                    rebate_bps=1.0, now_mono=110.0)
    # Dwell started at 110; predicate holds (mean = +3.0 > -2.0).
    assert g.is_suppressed(Side.SELL, 115.0)  # mid-dwell
    # Past the 10 s dwell.
    assert not g.is_suppressed(Side.SELL, 121.0)
    snap = g.snapshot_dict(121.0)
    assert snap["sell"]["cleared_via_favorable_total"] == 1
    assert snap["sell"]["cleared_via_ceiling_total"] == 0


def test_reflare_resets_dwell() -> None:
    """Predicate flipping back to not-holding mid-dwell resets the
    timer."""
    g = _gate(threshold_bps=-4.0, cooldown_seconds=600.0,
              favorable_exit_dwell_seconds=10.0)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    # Recovery: 4 favorable fills.
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=+2.0,
                    rebate_bps=1.0, now_mono=110.0)
    snap_mid = g.snapshot_dict(115.0)
    assert snap_mid["sell"]["favorable_dwell_active"] is True
    # Re-flare: one bad fill drags mean below clear band again. The
    # buffer is now [-10*4, +2*4]; after the next adverse fill the
    # last-4-fills window contains [+2, +2, +2, -30] → mean = -6 <
    # -2.0 clear band → predicate fails → dwell resets.
    g.note_fill(side=Side.SELL, markout_5s_bps=-31.0,
                rebate_bps=1.0, now_mono=115.0)
    snap_post = g.snapshot_dict(115.0)
    assert snap_post["sell"]["favorable_dwell_active"] is False
    # Even past where dwell would have completed, still suppressed.
    assert g.is_suppressed(Side.SELL, 125.0)


def test_ceiling_attribution_when_no_recovery() -> None:
    g = _gate(threshold_bps=-4.0, cooldown_seconds=30.0)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    assert g.is_suppressed(Side.SELL, 100.0)
    # No recovery fills. Ceiling expires at 130.
    assert not g.is_suppressed(Side.SELL, 131.0)
    snap = g.snapshot_dict(131.0)
    assert snap["sell"]["cleared_via_ceiling_total"] == 1
    assert snap["sell"]["cleared_via_favorable_total"] == 0


def test_ceiling_only_fires_once_per_arming() -> None:
    g = _gate(threshold_bps=-4.0, cooldown_seconds=30.0,
              favorable_exit_enabled=False)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    g.is_suppressed(Side.SELL, 100.0)
    g.is_suppressed(Side.SELL, 120.0)
    g.is_suppressed(Side.SELL, 131.0)  # ceiling
    g.is_suppressed(Side.SELL, 150.0)
    g.is_suppressed(Side.SELL, 200.0)
    snap = g.snapshot_dict(200.0)
    assert snap["sell"]["cleared_via_ceiling_total"] == 1


def test_partial_recovery_below_clear_band_no_favorable() -> None:
    """Mean recovers from -9 to -3 (above trigger -4, but BELOW clear
    band -2). Predicate doesn't hold → no favorable-exit dwell.

    Note on test mechanics: with min_fills=4 and a window of 20, each
    new recovery fill recomputes the trailing-4 mean as the deque
    slides through the boundary. Some of those intermediate sliding-
    window means re-fire the trigger and extend the cooldown — that's
    fine and expected; what we're verifying here is that the
    *favorable-exit predicate* never holds when the steady-state
    trailing mean is between trigger and clear band.
    """
    g = _gate(threshold_bps=-4.0, cooldown_seconds=600.0,
              clear_band_mult=0.5)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    # Push the trailing-4 mean to -3 (above trigger, below clear band).
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-4.0,
                    rebate_bps=1.0, now_mono=110.0)
    # Steady-state last-4 mean = -3, in the dead-band → predicate
    # doesn't hold → no favorable dwell.
    snap = g.snapshot_dict(110.0)
    assert snap["sell"]["favorable_dwell_active"] is False
    # No favorable-exit clear ever fires regardless of poll time.
    g.is_suppressed(Side.SELL, 200.0)
    g.is_suppressed(Side.SELL, 400.0)
    snap_end = g.snapshot_dict(400.0)
    assert snap_end["sell"]["cleared_via_favorable_total"] == 0


def test_disabled_favorable_exit_uses_pure_timer() -> None:
    g = _gate(threshold_bps=-4.0, cooldown_seconds=30.0,
              favorable_exit_enabled=False)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    # Even with strong recovery fills, the cooldown holds.
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=+10.0,
                    rebate_bps=1.0, now_mono=110.0)
    assert g.is_suppressed(Side.SELL, 120.0)
    assert not g.is_suppressed(Side.SELL, 131.0)
    snap = g.snapshot_dict(131.0)
    assert snap["sell"]["cleared_via_favorable_total"] == 0
    assert snap["sell"]["cleared_via_ceiling_total"] == 1


def test_clear_band_mult_zero_requires_positive_mean() -> None:
    """mult=0 → clear band = 0 → mean must be strictly positive."""
    g = _gate(threshold_bps=-4.0, clear_band_mult=0.0,
              cooldown_seconds=600.0)
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-10.0,
                    rebate_bps=1.0, now_mono=100.0)
    # Recovery to mean = -0.5 (not > 0). Doesn't clear.
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=-1.5,
                    rebate_bps=1.0, now_mono=110.0)
    assert g.is_suppressed(Side.SELL, 125.0)
    # Recovery to mean = +5 (positive). Clears after dwell.
    for _ in range(4):
        g.note_fill(side=Side.SELL, markout_5s_bps=+4.0,
                    rebate_bps=1.0, now_mono=130.0)
    assert not g.is_suppressed(Side.SELL, 141.0)
    snap = g.snapshot_dict(141.0)
    assert snap["sell"]["cleared_via_favorable_total"] == 1


# ---------------------------------------------------------------------------
# Snapshot shape
# ---------------------------------------------------------------------------


def test_snapshot_surfaces_phase4c3_fields() -> None:
    g = _gate()
    snap = g.snapshot_dict(0.0)
    for k in ("enabled", "threshold_bps", "min_fills",
              "cooldown_seconds",
              "favorable_exit_enabled", "clear_band_mult",
              "favorable_exit_dwell_seconds"):
        assert k in snap, f"missing top-level {k!r}"
    for side in ("buy", "sell"):
        for k in ("recent_count", "recent_mean_bps",
                  "suppressed", "seconds_remaining",
                  "fire_count", "last_trigger_mean_bps",
                  "favorable_dwell_active",
                  "cleared_via_favorable_total",
                  "cleared_via_ceiling_total"):
            assert k in snap[side], f"missing {side}.{k!r}"


# ---------------------------------------------------------------------------
# Integration with compute_quote_decision (smoke test the two new
# kwargs propagate correctly)
# ---------------------------------------------------------------------------


def test_compute_quote_decision_accepts_realised_edge_suppress_kwargs() -> None:
    """Smoke test: the two new parameters exist and default to False
    (no-op). The full quoting test suite covers the suppression
    semantics; here we just verify the API surface."""
    import inspect
    from app.quoting import compute_quote_decision

    sig = inspect.signature(compute_quote_decision)
    assert "realised_edge_suppress_bid" in sig.parameters
    assert "realised_edge_suppress_ask" in sig.parameters
    assert (
        sig.parameters["realised_edge_suppress_bid"].default is False
    )
    assert (
        sig.parameters["realised_edge_suppress_ask"].default is False
    )
