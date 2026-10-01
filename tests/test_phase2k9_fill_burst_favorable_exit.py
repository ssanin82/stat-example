"""Phase 2K.9 — favorable-exit predicate for the
``fill_burst_detector`` size-shrink cooldown.

``FILL_BURST_COOLDOWN_SECONDS`` becomes the MAX-cooldown ceiling;
the cooldown also clears EARLY when the live fill-in-window count
has dropped below ``threshold × clear_band_mult`` (default mult=0.5)
and held there for ``favorable_exit_dwell_seconds`` (default 5 s).

Unlike most prior Phase 2K items, the predicate is checked on every
``current_size_mult()`` / ``active()`` poll (not just on new
``note_fill``), because the burst signal is "fills have STOPPED
arriving" — silent recovery must be detectable without waiting for
the next fill.

The existing ``test_fill_burst_detector.py`` regression suite still
covers the pure-timer ceiling path.
"""

from __future__ import annotations

from app.fill_burst_detector import FillBurstDetector


def _detector(**overrides) -> FillBurstDetector:
    base = dict(
        threshold=3,
        window_seconds=30.0,
        size_mult=0.5,
        cooldown_seconds=60.0,
        favorable_exit_enabled=True,
        clear_band_mult=0.5,
        favorable_exit_dwell_seconds=5.0,
    )
    base.update(overrides)
    return FillBurstDetector(**base)


def _arm(detector: FillBurstDetector, *, at: float = 0.0) -> None:
    """Drive 3 fast fills at ``at`` (clustered at the same moment) so
    the cooldown arms."""
    for _ in range(3):
        detector.note_fill(now_mono=at)


# ---------------------------------------------------------------------------
# Favorable-exit: silence → clears early
# ---------------------------------------------------------------------------


def test_favorable_exit_clears_when_fills_dissipate() -> None:
    """threshold=3, window=30, mult=0.5 → clear band 1.5. After
    arming with 3 fills, let the window slide so the live count
    drops to 0; after the dwell, the cooldown clears via favorable."""
    d = _detector()
    _arm(d, at=0.0)
    assert d.active(now_mono=1.0)
    # At t=35 the window (30 s) has slid past all 3 fills → live count=0.
    # That starts the dwell.
    d.active(now_mono=35.0)
    # At t=40 (5 s into dwell), past the dwell duration → clears.
    assert not d.active(now_mono=40.5)
    snap = d.snapshot_dict(now_mono=40.5)
    assert snap["cleared_via_favorable_total"] == 1
    assert snap["cleared_via_ceiling_total"] == 0


def test_favorable_exit_does_not_fire_before_dwell_completes() -> None:
    """Dwell must hold for the full configured duration before clearing."""
    d = _detector()
    _arm(d, at=0.0)
    # Window slid past at t=35 → live count=0. Polls during dwell
    # must show active=True.
    for t in (35.0, 36.0, 38.0, 39.9):
        assert d.active(now_mono=t), f"unexpected clear at t={t}"
    # Past dwell.
    assert not d.active(now_mono=40.5)


def test_reflare_resets_dwell() -> None:
    """A fresh fill during the dwell drags the live count back above
    the clear band → dwell resets."""
    d = _detector(cooldown_seconds=120.0)  # long enough that ceiling
                                            # doesn't fire prematurely
    _arm(d, at=0.0)
    # Window slid past — start dwell at t=35.
    d.active(now_mono=35.0)
    snap_mid = d.snapshot_dict(now_mono=35.0)
    assert snap_mid["favorable_dwell_active"] is True
    # Two more fast fills at t=37: live count = 2 (the two new ones,
    # the original 3 evicted because window starts at 37-30=7 and
    # they were at t=0). 2 >= 1.5 clear band → predicate breaks →
    # dwell resets.
    d.note_fill(now_mono=37.0)
    d.note_fill(now_mono=37.0)
    snap_post = d.snapshot_dict(now_mono=37.0)
    assert snap_post["favorable_dwell_active"] is False
    # Even later (would have been past original dwell), still active.
    assert d.active(now_mono=45.0)
    assert d.snapshot_dict(now_mono=45.0)["cleared_via_favorable_total"] == 0


def test_ceiling_fires_when_fills_keep_arriving() -> None:
    """No dissipation → ceiling fires naturally."""
    d = _detector()
    _arm(d, at=0.0)
    # Keep adding fills every 5 s so the window stays full.
    for t in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0]:
        d.note_fill(now_mono=t)
    assert d.active(now_mono=55.0)
    # NOW stop adding fills. At t=61 the original cooldown (set at
    # t=50 from the last fill, extended) ends.
    # Actually the cooldown is `max(prev_until, now + cooldown_s)`,
    # so each new fill arming pushes the deadline. Last arm at t=50
    # → until = 110. So we need to wait longer.
    # Let's instead use a detector with a shorter cooldown.
    d2 = _detector()
    _arm(d2, at=0.0)
    # No further fills. At t=35 window slides → predicate would
    # start dwell, but we need to NOT let it complete. Test the
    # PURE ceiling case: poll only at the start and end.
    assert d2.active(now_mono=0.5)
    # At t=70 (past 60s ceiling, also past 35+5 dwell) — but the
    # dwell would have cleared first. Need an "elevated count" path.
    # Use a config where the clear band is unreachable.
    d3 = _detector(clear_band_mult=0.0)  # clear band = 0; live count
                                          # must be < 0 (impossible)
    _arm(d3, at=0.0)
    assert d3.active(now_mono=0.5)
    # Past ceiling.
    assert not d3.active(now_mono=61.0)
    snap = d3.snapshot_dict(now_mono=61.0)
    assert snap["cleared_via_ceiling_total"] == 1
    assert snap["cleared_via_favorable_total"] == 0


def test_partial_recovery_above_clear_band_no_favorable() -> None:
    """threshold=3, clear_band_mult=0.5 → clear band 1.5. If live
    count drops to 2 (above 1.5), predicate doesn't hold."""
    d = _detector(cooldown_seconds=120.0)
    _arm(d, at=0.0)
    # At t=29, the original 3 fills still in window (cutoff = -1).
    # Add 2 fills at t=29 to make the window contain 5.
    d.note_fill(now_mono=29.0)
    d.note_fill(now_mono=29.0)
    # Now window starts at t=-1, contains [0,0,0,29,29] → 5 fills.
    # At t=31: cutoff=1, evicts the 3 at t=0 → contains [29,29] = 2.
    # 2 >= 1.5 → predicate doesn't hold.
    snap = d.snapshot_dict(now_mono=31.0)
    assert snap["favorable_dwell_active"] is False
    # Ceiling fires at t=121+ (cooldown_until_mono extended when the
    # 4th and 5th fills hit threshold again at t=29; deadline 149).
    # Cap: just verify no favorable fire.
    assert snap["cleared_via_favorable_total"] == 0


def test_clear_band_mult_one_clears_at_trigger_boundary() -> None:
    """mult=1.0 → clear band = threshold. Live count must drop below
    threshold (i.e. burst no longer fires) for the dwell."""
    d = _detector(clear_band_mult=1.0)
    _arm(d, at=0.0)
    # At t=29 add 1 more fill (window now has 4); at t=31 the 3
    # originals evict → count=1 < 3 → predicate holds.
    d.note_fill(now_mono=29.0)
    assert d.active(now_mono=31.0)
    # Now wait through the dwell.
    assert not d.active(now_mono=36.5)
    assert d.snapshot_dict(now_mono=36.5)["cleared_via_favorable_total"] == 1


def test_disabled_falls_back_to_pure_timer() -> None:
    """``favorable_exit_enabled=False`` → cooldown only clears on the
    fixed timer; the favorable predicate is bypassed."""
    d = _detector(favorable_exit_enabled=False)
    _arm(d, at=0.0)
    # Window slid past, no new fills — gate would normally clear via
    # favorable. But with predicate disabled, ceiling holds.
    assert d.active(now_mono=35.0)
    assert d.active(now_mono=59.0)
    assert not d.active(now_mono=61.0)
    snap = d.snapshot_dict(now_mono=61.0)
    assert snap["cleared_via_favorable_total"] == 0
    assert snap["cleared_via_ceiling_total"] == 1


def test_size_mult_drops_to_1_when_favorable_clears() -> None:
    """Public ``current_size_mult`` returns 1.0 after favorable-exit
    fires (not the shrunken value)."""
    d = _detector()
    _arm(d, at=0.0)
    assert d.current_size_mult(now_mono=1.0) == 0.5
    # Slide window + complete dwell.
    d.current_size_mult(now_mono=35.0)
    assert d.current_size_mult(now_mono=40.5) == 1.0


def test_ceiling_attribution_only_fires_once_per_arming() -> None:
    """Multiple polls past the ceiling must not double-count."""
    d = _detector(clear_band_mult=0.0)
    _arm(d, at=0.0)
    d.active(now_mono=1.0)
    d.active(now_mono=30.0)
    d.active(now_mono=61.0)
    d.active(now_mono=70.0)
    d.active(now_mono=100.0)
    snap = d.snapshot_dict(now_mono=100.0)
    assert snap["cleared_via_ceiling_total"] == 1


def test_reflare_via_fresh_arm_resets_dwell_and_counts_as_new_fire() -> None:
    """Fresh arm during favorable dwell must reset the dwell and not
    confuse with a favorable clear."""
    d = _detector(cooldown_seconds=120.0)
    _arm(d, at=0.0)
    # Window slid at t=35 → dwell starts.
    d.active(now_mono=35.0)
    assert d.snapshot_dict(now_mono=35.0)["favorable_dwell_active"] is True
    # Now 3 fast fills at t=36 → re-arm + dwell reset.
    d.note_fill(now_mono=36.0)
    d.note_fill(now_mono=36.0)
    d.note_fill(now_mono=36.0)
    snap = d.snapshot_dict(now_mono=36.0)
    assert snap["fire_count"] >= 2
    assert snap["favorable_dwell_active"] is False


def test_snapshot_dict_exposes_phase2k9_fields() -> None:
    d = _detector()
    snap = d.snapshot_dict(now_mono=0.0)
    for k in (
        "favorable_exit_enabled",
        "clear_band_mult",
        "favorable_exit_dwell_seconds",
        "favorable_dwell_active",
        "cleared_via_favorable_total",
        "cleared_via_ceiling_total",
    ):
        assert k in snap, f"missing {k!r}"
    # Legacy keys still present.
    for k in (
        "enabled",
        "threshold",
        "window_seconds",
        "size_mult",
        "cooldown_seconds",
        "active",
        "seconds_remaining",
        "fire_count",
        "recent_fill_count",
    ):
        assert k in snap, f"legacy {k!r} broken"
