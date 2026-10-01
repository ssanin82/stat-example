"""Tests for v1.4.88 Phase 5B — SlotDiff struct unification.

Pre-Phase-5B the three materiality helpers
(``_should_replace_working_order``, ``_action_materiality_allows_replace``,
``_should_emit_fresh_place_intent``) each recomputed px/sz deltas
independently from the same ``(cur, desired)`` pair. Codex F11
flagged this as redundant work + a divergence risk.

Phase 5B introduces ``SlotDiff``: a frozen dataclass that captures
the three diff metrics (px_ticks, px_bps_vs_mid, sz_rel) computed
ONCE per slot per tick. ``_reconcile_materiality_check`` builds the
diff once and routes both helpers (``_replace_threshold_check_from_diff``
and ``_action_materiality_from_diff``) through it.

Tests pin:

* ``SlotDiff.from_pair`` arithmetic matches the legacy helpers
  bit-for-bit (golden-master regression).
* The new diff-driven helpers produce IDENTICAL results to the
  legacy helpers across a sweep of random inputs.
* Edge cases: zero-tick, near-zero-mid, near-zero-cur_size.
"""

from __future__ import annotations

import math
import random
from dataclasses import is_dataclass

import pytest

from app.execution import SlotDiff


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_phase5b_slot_diff_is_frozen_dataclass() -> None:
    assert is_dataclass(SlotDiff)
    diff = SlotDiff(px_ticks=1.0, px_bps_vs_mid=5.0, sz_rel=0.1)
    with pytest.raises((AttributeError, Exception)):
        diff.px_ticks = 99.0  # type: ignore[misc]


def test_phase5b_slot_diff_has_expected_fields() -> None:
    diff = SlotDiff(px_ticks=2.5, px_bps_vs_mid=10.0, sz_rel=0.2)
    assert diff.px_ticks == 2.5
    assert diff.px_bps_vs_mid == 10.0
    assert diff.sz_rel == 0.2


# ---------------------------------------------------------------------------
# from_pair arithmetic — matches the legacy helpers
# ---------------------------------------------------------------------------


def test_phase5b_from_pair_basic_case() -> None:
    """Simple sanity check with round numbers."""
    diff = SlotDiff.from_pair(
        cur_price=2.000, cur_size=10.0,
        desired_price=2.002, desired_size=11.0,
        tick=0.001, mid_price=2.001,
    )
    # px_diff = 0.002, in ticks = 2.0
    assert abs(diff.px_ticks - 2.0) < 1e-9
    # px_diff / mid * 10000 = 0.002 / 2.001 * 10000 = 9.995 bps
    assert abs(diff.px_bps_vs_mid - 9.9950024987506) < 1e-6
    # sz_diff / cur_size = 1.0 / 10.0 = 0.1
    assert abs(diff.sz_rel - 0.1) < 1e-9


def test_phase5b_from_pair_zero_diff() -> None:
    diff = SlotDiff.from_pair(
        cur_price=2.000, cur_size=10.0,
        desired_price=2.000, desired_size=10.0,
        tick=0.001, mid_price=2.001,
    )
    assert diff.px_ticks == 0.0
    assert diff.px_bps_vs_mid == 0.0
    assert diff.sz_rel == 0.0


def test_phase5b_from_pair_handles_tick_floor() -> None:
    """When tick is non-positive (degenerate), the floor of 1e-15
    prevents division by zero — matches legacy `_action_materiality_allows_replace`."""
    diff = SlotDiff.from_pair(
        cur_price=2.000, cur_size=10.0,
        desired_price=2.002, desired_size=10.0,
        tick=0.0, mid_price=2.001,
    )
    # px_ticks = 0.002 / 1e-15 = huge but finite.
    assert math.isfinite(diff.px_ticks)
    assert diff.px_ticks > 1e10


def test_phase5b_from_pair_handles_mid_floor() -> None:
    """When mid is zero/negative, the floor of 1e-12 prevents div0
    — matches legacy `_should_replace_working_order`."""
    diff = SlotDiff.from_pair(
        cur_price=2.000, cur_size=10.0,
        desired_price=2.002, desired_size=10.0,
        tick=0.001, mid_price=0.0,
    )
    assert math.isfinite(diff.px_bps_vs_mid)
    assert diff.px_bps_vs_mid > 1e10


def test_phase5b_from_pair_handles_cur_size_floor() -> None:
    """When cur_size is zero (impossible in production but defensive),
    the floor of 1e-15 prevents div0."""
    diff = SlotDiff.from_pair(
        cur_price=2.000, cur_size=0.0,
        desired_price=2.000, desired_size=1.0,
        tick=0.001, mid_price=2.001,
    )
    assert math.isfinite(diff.sz_rel)
    assert diff.sz_rel > 1e10


# ---------------------------------------------------------------------------
# Golden-master parity: SlotDiff produces same metrics as the
# legacy inline arithmetic.
# ---------------------------------------------------------------------------


def _legacy_action_materiality_px_ticks(cur_px: float, des_px: float, tick: float) -> float:
    """Mirrors the inline px_ticks computation from the pre-Phase-5B
    ``_action_materiality_allows_replace`` (line 10068)."""
    return abs(float(cur_px) - float(des_px)) / max(tick, 1e-15)


def _legacy_action_materiality_sz_rel(cur_sz: float, des_sz: float) -> float:
    """Mirrors the inline sz_rel from the same legacy helper (line 10069)."""
    return abs(float(cur_sz) - float(des_sz)) / max(float(cur_sz), 1e-15)


def _legacy_should_replace_diff_bps(cur_px: float, des_px: float, mid: float) -> float:
    """Mirrors the inline order_diff_bps from the legacy
    ``_should_replace_working_order`` (line 10125-10129)."""
    return abs(float(cur_px) - float(des_px)) / max(float(mid), 1e-12) * 10_000.0


def _legacy_should_replace_sz_rel(cur_sz: float, des_sz: float) -> float:
    """Mirrors the inline order_size_rel_diff (line 10130-10132).
    NOTE: legacy uses 1e-12 floor on cur_size; SlotDiff uses 1e-15.
    For all production-realistic cur_size values (> 1e-9), the
    difference is zero."""
    return abs(float(cur_sz) - float(des_sz)) / max(float(cur_sz), 1e-12)


def test_phase5b_golden_master_random_sweep() -> None:
    """Random inputs → assert SlotDiff matches legacy arithmetic.

    Tests 1000 random (cur, desired, tick, mid) tuples in realistic
    ranges. Validates the three SlotDiff fields against the legacy
    inline arithmetic used by the pre-Phase-5B helpers.
    """
    rng = random.Random(42)
    mismatches: list[tuple[str, float, float]] = []
    for _ in range(1000):
        cur_px = rng.uniform(0.001, 100_000.0)
        cur_sz = rng.uniform(0.001, 1000.0)
        des_px = cur_px * rng.uniform(0.95, 1.05)  # ±5% drift
        des_sz = cur_sz * rng.uniform(0.5, 2.0)    # 50%-200%
        tick = rng.choice([0.0001, 0.001, 0.01, 0.1, 1.0])
        mid = (cur_px + des_px) / 2

        diff = SlotDiff.from_pair(
            cur_price=cur_px, cur_size=cur_sz,
            desired_price=des_px, desired_size=des_sz,
            tick=tick, mid_price=mid,
        )

        # px_ticks parity
        expected = _legacy_action_materiality_px_ticks(cur_px, des_px, tick)
        if abs(diff.px_ticks - expected) > max(1e-9, abs(expected) * 1e-12):
            mismatches.append(("px_ticks", diff.px_ticks, expected))

        # px_bps_vs_mid parity
        expected = _legacy_should_replace_diff_bps(cur_px, des_px, mid)
        if abs(diff.px_bps_vs_mid - expected) > max(1e-9, abs(expected) * 1e-12):
            mismatches.append(("px_bps_vs_mid", diff.px_bps_vs_mid, expected))

        # sz_rel parity (1e-15 vs 1e-12 floor — irrelevant for production-realistic cur_size)
        expected = _legacy_action_materiality_sz_rel(cur_sz, des_sz)
        if abs(diff.sz_rel - expected) > max(1e-9, abs(expected) * 1e-12):
            mismatches.append(("sz_rel", diff.sz_rel, expected))

    assert not mismatches, (
        f"SlotDiff diverged from legacy arithmetic on {len(mismatches)} cases. "
        f"First mismatch: {mismatches[0]}"
    )


# ---------------------------------------------------------------------------
# Replace-threshold check parity
# ---------------------------------------------------------------------------


def _legacy_should_replace_working_order(
    cur_px: float, cur_sz: float, des_px: float, des_sz: float,
    *, replace_threshold_bps: float, mid_price: float,
) -> bool:
    """Bit-for-bit re-implementation of the pre-Phase-5B
    ``_should_replace_working_order`` static method (line 10112+)."""
    order_diff_bps = (
        abs(float(cur_px) - float(des_px))
        / max(float(mid_price), 1e-12)
        * 10_000.0
    )
    order_size_rel_diff = abs(float(cur_sz) - float(des_sz)) / max(
        float(cur_sz), 1e-12
    )
    return not (order_diff_bps < float(replace_threshold_bps) and order_size_rel_diff <= 0.15)


def test_phase5b_replace_threshold_check_matches_legacy() -> None:
    """Random sweep: ``_replace_threshold_check_from_diff`` produces
    identical bool result to the legacy ``_should_replace_working_order``.
    """
    from app.execution import OrderManager
    rng = random.Random(123)
    for _ in range(500):
        cur_px = rng.uniform(0.001, 100_000.0)
        cur_sz = rng.uniform(0.001, 1000.0)
        des_px = cur_px * rng.uniform(0.99, 1.01)
        des_sz = cur_sz * rng.uniform(0.7, 1.3)
        mid = (cur_px + des_px) / 2
        threshold_bps = rng.uniform(0.5, 20.0)

        legacy = _legacy_should_replace_working_order(
            cur_px, cur_sz, des_px, des_sz,
            replace_threshold_bps=threshold_bps, mid_price=mid,
        )
        diff = SlotDiff.from_pair(
            cur_price=cur_px, cur_size=cur_sz,
            desired_price=des_px, desired_size=des_sz,
            tick=0.001, mid_price=mid,
        )
        new = OrderManager._replace_threshold_check_from_diff(
            diff, replace_threshold_bps=threshold_bps
        )
        assert legacy == new, (
            f"_replace_threshold_check_from_diff diverged from legacy: "
            f"cur=({cur_px},{cur_sz}) des=({des_px},{des_sz}) "
            f"thr={threshold_bps} mid={mid} → legacy={legacy} new={new}"
        )
