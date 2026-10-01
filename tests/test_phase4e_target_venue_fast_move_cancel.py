"""Phase 4E (v1.4.165) — target-venue fast-move cancel detector.

Pure-function tests for ``detect_target_venue_fast_move``. The
``OrderManager`` wiring that consumes the helper and dispatches the
actual cancel is intentionally tested at integration level (existing
execution-test surface); these tests nail down the LOGIC + the
exchange-agnostic naming + the v1.4.157-224846 replay.

Replay context: BUY-side mean 5 s markout this session was −3.28 bps;
the worst behind-touch BUY fills had post-fill mid drops of 15-45 bps
in 5 seconds. With a 500 ms mid-return threshold of 10 bps, the
detector returns ``Side.BUY`` (cancel bids) on the leading edge of
those drops, before the resting bids get hit.
"""

from __future__ import annotations

import math

from app.enums import Side
from app.fast_move_cancel import detect_target_venue_fast_move


# ---------------------------------------------------------------------------
# Sign convention: detector returns the SIDE TO CANCEL
# ---------------------------------------------------------------------------


def test_up_move_cancels_asks() -> None:
    """Local mid moving UP fast → asks are stale-low → cancel ASKs.
    Helper returns ``Side.SELL`` (the side of asks)."""
    side = detect_target_venue_fast_move(
        mid_return_bps=15.0,
        threshold_bps=10.0,
    )
    assert side == Side.SELL


def test_down_move_cancels_bids() -> None:
    """Local mid moving DOWN fast → bids are stale-high → cancel BIDs.
    Helper returns ``Side.BUY`` (the side of bids)."""
    side = detect_target_venue_fast_move(
        mid_return_bps=-15.0,
        threshold_bps=10.0,
    )
    assert side == Side.BUY


def test_threshold_boundary_inclusive() -> None:
    """``>= threshold`` is true at the boundary itself."""
    assert detect_target_venue_fast_move(
        mid_return_bps=10.0, threshold_bps=10.0
    ) == Side.SELL
    assert detect_target_venue_fast_move(
        mid_return_bps=-10.0, threshold_bps=10.0
    ) == Side.BUY


def test_below_threshold_returns_none() -> None:
    assert detect_target_venue_fast_move(
        mid_return_bps=5.0, threshold_bps=10.0
    ) is None
    assert detect_target_venue_fast_move(
        mid_return_bps=-9.99, threshold_bps=10.0
    ) is None
    assert detect_target_venue_fast_move(
        mid_return_bps=0.0, threshold_bps=10.0
    ) is None


# ---------------------------------------------------------------------------
# Feature dormant: zero / negative threshold
# ---------------------------------------------------------------------------


def test_threshold_zero_returns_none() -> None:
    """Default ``threshold_bps=0.0`` keeps the gate dormant — opt-in."""
    assert detect_target_venue_fast_move(
        mid_return_bps=100.0,
        threshold_bps=0.0,
    ) is None


def test_negative_threshold_returns_none() -> None:
    """Negative thresholds are nonsensical; treat as dormant."""
    assert detect_target_venue_fast_move(
        mid_return_bps=100.0,
        threshold_bps=-5.0,
    ) is None


# ---------------------------------------------------------------------------
# Missing / degenerate inputs
# ---------------------------------------------------------------------------


def test_none_mid_return_returns_none() -> None:
    """Kinematics signal not yet warmed up → gate dormant."""
    assert detect_target_venue_fast_move(
        mid_return_bps=None,
        threshold_bps=10.0,
    ) is None


def test_nan_mid_return_returns_none() -> None:
    """Degenerate market data → no decision."""
    assert detect_target_venue_fast_move(
        mid_return_bps=float("nan"),
        threshold_bps=10.0,
    ) is None
    assert detect_target_venue_fast_move(
        mid_return_bps=float("inf"),
        threshold_bps=10.0,
    ) is None
    assert detect_target_venue_fast_move(
        mid_return_bps=float("-inf"),
        threshold_bps=10.0,
    ) is None


# ---------------------------------------------------------------------------
# Architecture / naming: exchange-agnostic
# ---------------------------------------------------------------------------


def test_helper_does_not_reference_any_specific_exchange() -> None:
    """The helper must work for any target/reference venue pair —
    no Binance / OKX / GRVT names in the module."""
    import app.fast_move_cancel as fmc
    import inspect

    src = inspect.getsource(fmc)
    forbidden = ("binance", "okx", "grvt", "hyperliquid", "bluefin")
    for name in forbidden:
        assert name not in src.lower(), (
            f"Exchange-specific name {name!r} leaked into "
            "fast_move_cancel.py — must stay exchange-agnostic."
        )


# ---------------------------------------------------------------------------
# v1.4.157-260520-224846 replay
# ---------------------------------------------------------------------------


def test_v1_4_157_replay_bid_cancel_on_down_move() -> None:
    """Replay the 17:36:22 sequence from the v1.4.157-260520-224846
    snapshot: bot's resting bid at $2.063 got hit, post-fill mid was
    $2.0595 (60 bps down in 1 second). With a 10 bp threshold on the
    500 ms mid-return, the detector should return Side.BUY (cancel
    bids) BEFORE the bid gets hit.

    The actual mid-return for the 500 ms window leading up to the
    fill is approximately −15 to −25 bps depending on the exact
    moment. Use −15 as the conservative replay value."""
    side = detect_target_venue_fast_move(
        mid_return_bps=-15.0,
        threshold_bps=10.0,
    )
    assert side == Side.BUY


def test_v1_4_157_replay_ask_cancel_on_up_move() -> None:
    """Mirror replay from the v1.4.157-260520-213540 snapshot: at
    17:14 TON ramped from $2.052 to $2.059 in ~14 s. The 500 ms
    leading-edge mid-return was approximately +12 to +20 bps. With
    threshold=10 bps the detector returns Side.SELL."""
    side = detect_target_venue_fast_move(
        mid_return_bps=+12.0,
        threshold_bps=10.0,
    )
    assert side == Side.SELL


# ---------------------------------------------------------------------------
# Pure-function: no side effects
# ---------------------------------------------------------------------------


def test_helper_is_idempotent() -> None:
    """Calling twice with same inputs returns same result; no hidden
    state."""
    for r in (-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0):
        r1 = detect_target_venue_fast_move(
            mid_return_bps=r, threshold_bps=10.0
        )
        r2 = detect_target_venue_fast_move(
            mid_return_bps=r, threshold_bps=10.0
        )
        assert r1 == r2
