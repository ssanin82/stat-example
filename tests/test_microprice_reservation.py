"""Microprice-based reservation for ``compute_quote_decision``.

The microprice is a queue-imbalance-weighted version of the midprice:

    microprice = (bid_size * best_ask + ask_size * best_bid) / (bid_size + ask_size)

Intuition:
  - Heavy bid stack (bid_size >> ask_size) → microprice shifts toward
    the ask (price likely to move up next as buyers consume asks).
  - Heavy ask stack → microprice shifts toward the bid.
  - Balanced stacks → microprice equals the midpoint.

Using microprice as the reservation centerpoint (instead of midpoint)
shifts our quotes in the direction price is statistically likely to
move next, reducing adverse selection. The overnight 2026-04-19 ETH
session showed a mean markout of −1.53 bps despite clean 1:1 BUY/SELL
symmetry; that residual adverse selection is what this refactor
targets.

Invariants pinned here:

  1. **Formula correctness.** Heavy bids → microprice near best_ask;
     heavy asks → near best_bid; balanced → the midpoint.
  2. **Graceful fallback.** Any missing / malformed / non-positive
     depth → ``compute_microprice`` returns None, and the reservation
     falls back to midprice transparently.
  3. **Feature flag gates USE, not compute.** When depth is available,
     the microprice is ALWAYS populated in ``QuoteDecision.microprice``
     for shadow-mode analysis, regardless of the flag — the flag only
     controls whether the value is applied to the reservation.
  4. **Reservation actually moves.** When the flag is on and depth
     is skewed, the reservation visibly differs from the mid-based
     reservation (controlled direction + sign).
"""

from __future__ import annotations

import math

import pytest

from app.quoting import compute_microprice, compute_quote_decision
from app.models import ToxicitySnapshot
from tests.settings_helpers import UnitTestSettings


# --------------------- Formula correctness -----------------------------


def test_microprice_balanced_stacks_equals_midpoint() -> None:
    """Equal bid/ask size → microprice == midpoint."""
    mp = compute_microprice(best_bid=99.0, best_ask=101.0, bid_size=5.0, ask_size=5.0)
    assert mp is not None
    assert mp == pytest.approx(100.0, abs=1e-9)


def test_microprice_heavy_bids_pulls_toward_ask() -> None:
    """bid_size >> ask_size → microprice near best_ask (price rising)."""
    # 9× more bid depth than ask. Expect microprice ~= best_ask.
    mp = compute_microprice(best_bid=99.0, best_ask=101.0, bid_size=90.0, ask_size=10.0)
    assert mp is not None
    # Formula: (90*101 + 10*99) / 100 = (9090 + 990)/100 = 100.8
    assert mp == pytest.approx(100.8, abs=1e-9)
    # Sanity: must lie in [best_bid, best_ask] and above midpoint.
    assert 99.0 <= mp <= 101.0
    assert mp > 100.0


def test_microprice_heavy_asks_pulls_toward_bid() -> None:
    """ask_size >> bid_size → microprice near best_bid (price falling)."""
    mp = compute_microprice(best_bid=99.0, best_ask=101.0, bid_size=10.0, ask_size=90.0)
    assert mp is not None
    # Formula: (10*101 + 90*99) / 100 = (1010 + 8910)/100 = 99.2
    assert mp == pytest.approx(99.2, abs=1e-9)
    assert 99.0 <= mp <= 101.0
    assert mp < 100.0


def test_microprice_monotonic_in_imbalance() -> None:
    """Increasing bid-side dominance must monotonically increase
    microprice (and vice versa). Guards against a sign-flip in the
    formula — a classic off-by-one mistake."""
    prev = None
    for bid_size in (1.0, 10.0, 50.0, 100.0, 1000.0):
        mp = compute_microprice(
            best_bid=99.0, best_ask=101.0, bid_size=bid_size, ask_size=10.0
        )
        assert mp is not None
        if prev is not None:
            assert mp >= prev, f"microprice should be non-decreasing as bid_size grows; {prev} -> {mp}"
        prev = mp


# --------------------- Graceful fallback -------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"best_bid": None, "best_ask": 101.0, "bid_size": 1.0, "ask_size": 1.0},
        {"best_bid": 99.0, "best_ask": None, "bid_size": 1.0, "ask_size": 1.0},
        {"best_bid": 99.0, "best_ask": 101.0, "bid_size": None, "ask_size": 1.0},
        {"best_bid": 99.0, "best_ask": 101.0, "bid_size": 1.0, "ask_size": None},
        # All None — e.g. early tick before first WS payload.
        {"best_bid": None, "best_ask": None, "bid_size": None, "ask_size": None},
    ],
)
def test_microprice_returns_none_on_missing_inputs(kwargs: dict) -> None:
    assert compute_microprice(**kwargs) is None


def test_microprice_returns_none_on_zero_total_depth() -> None:
    """Both sizes zero → division by zero guard → None."""
    assert compute_microprice(best_bid=99.0, best_ask=101.0, bid_size=0.0, ask_size=0.0) is None


def test_microprice_returns_none_on_negative_size() -> None:
    """Negative sizes are physically invalid; must be rejected."""
    assert compute_microprice(best_bid=99.0, best_ask=101.0, bid_size=-1.0, ask_size=5.0) is None
    assert compute_microprice(best_bid=99.0, best_ask=101.0, bid_size=5.0, ask_size=-1.0) is None


def test_microprice_returns_none_on_nan_or_inf() -> None:
    assert compute_microprice(
        best_bid=float("nan"), best_ask=101.0, bid_size=1.0, ask_size=1.0
    ) is None
    assert compute_microprice(
        best_bid=99.0, best_ask=float("inf"), bid_size=1.0, ask_size=1.0
    ) is None


# --------------------- compute_quote_decision integration --------------


def _settings(**overrides) -> UnitTestSettings:
    base = {
        "TRADING_ENABLED": False,
        "HL_SECRET_KEY": "x",
        "HL_ACCOUNT_ADDRESS": "0xabc",
        "SYMBOL": "ETH",
        "MAX_ABS_POSITION": 1.0,
        "INVENTORY_SKEW_COEFF_BPS": 0.0,  # isolate microprice effect from skew
        "BASE_HALF_SPREAD_BPS": 1.0,
        "MIN_HALF_SPREAD_BPS": 0.5,
        "MAX_HALF_SPREAD_BPS": 40.0,
        "ECONOMIC_MIN_HALF_SPREAD_NEUTRAL_BPS": 0.0,
        "ECONOMIC_MIN_HALF_SPREAD_INVENTORY_BPS": 0.0,
        "ECONOMIC_TOXICITY_SCORE_HALF_SPREAD_BPS": 0.0,
        "TOXICITY_SCORE_HALF_SPREAD_BPS": 0.0,
        "QUOTE_NOTIONAL_USD": 100.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _tox() -> ToxicitySnapshot:
    return ToxicitySnapshot(
        score=0.0,
        one_sided_fill_ratio=0.0,
        avg_adverse_markout_bps=0.0,
        vol_spike_ratio=0.0,
        hard_trigger=False,
        soft_trigger=False,
        toxic_side=None,
    )


def test_decision_uses_microprice_as_reservation_when_depth_balanced() -> None:
    """Flag on, balanced depth → microprice == mid → reservation == mid."""
    s = _settings(MICROPRICE_RESERVATION_ENABLED=True)
    d = compute_quote_decision(
        s, mid=100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(),
        best_bid=99.0, best_ask=101.0, bid_size=5.0, ask_size=5.0,
    )
    # microprice persisted (5,5) = 100 = mid
    assert d.microprice == pytest.approx(100.0, abs=1e-9)
    # reservation == ref_price * (1 - 0*norm_inv/1e4) = ref_price = microprice
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_decision_reservation_shifts_up_with_heavy_bids() -> None:
    """Flag on, heavy bid depth → reservation shifts toward best_ask.
    Both target_bid and target_ask move up vs their mid-based baseline
    by (microprice - mid), since the half-spread band is centered on
    reservation."""
    depth_kwargs = dict(best_bid=99.0, best_ask=101.0, bid_size=90.0, ask_size=10.0)

    d_on = compute_quote_decision(
        _settings(MICROPRICE_RESERVATION_ENABLED=True),
        mid=100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(), **depth_kwargs,
    )
    d_off = compute_quote_decision(
        _settings(MICROPRICE_RESERVATION_ENABLED=False),
        mid=100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(), **depth_kwargs,
    )

    # With inventory_skew=0, reservation_on == microprice == 100.8.
    assert d_on.microprice == pytest.approx(100.8, abs=1e-9)
    assert d_on.reservation_price == pytest.approx(100.8, abs=1e-9)
    # Flag-off: reservation follows mid.
    assert d_off.reservation_price == pytest.approx(100.0, abs=1e-9)
    # Quote band centered on reservation; both sides move together.
    shift_bid = d_on.target_bid - d_off.target_bid
    shift_ask = d_on.target_ask - d_off.target_ask
    assert shift_bid == pytest.approx(0.8, rel=1e-3), f"target_bid should shift +0.8; got {shift_bid}"
    assert shift_ask == pytest.approx(0.8, rel=1e-3), f"target_ask should shift +0.8; got {shift_ask}"
    # Microprice is persisted on BOTH decisions (flag gates use, not compute).
    assert d_off.microprice == pytest.approx(100.8, abs=1e-9)


def test_decision_reservation_shifts_down_with_heavy_asks() -> None:
    """Flag on, heavy ask depth → reservation shifts toward best_bid."""
    depth_kwargs = dict(best_bid=99.0, best_ask=101.0, bid_size=10.0, ask_size=90.0)

    d_on = compute_quote_decision(
        _settings(MICROPRICE_RESERVATION_ENABLED=True),
        mid=100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(), **depth_kwargs,
    )
    d_off = compute_quote_decision(
        _settings(MICROPRICE_RESERVATION_ENABLED=False),
        mid=100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(), **depth_kwargs,
    )

    assert d_on.microprice == pytest.approx(99.2, abs=1e-9)
    assert d_on.reservation_price == pytest.approx(99.2, abs=1e-9)
    assert d_off.reservation_price == pytest.approx(100.0, abs=1e-9)
    shift_bid = d_on.target_bid - d_off.target_bid
    shift_ask = d_on.target_ask - d_off.target_ask
    assert shift_bid == pytest.approx(-0.8, rel=1e-3), f"target_bid should shift -0.8; got {shift_bid}"
    assert shift_ask == pytest.approx(-0.8, rel=1e-3), f"target_ask should shift -0.8; got {shift_ask}"


def test_flag_disabled_uses_mid_despite_available_depth() -> None:
    """Flag off → reservation is mid-based even when depth is known.
    Microprice is STILL populated for shadow-mode analysis."""
    s = _settings(MICROPRICE_RESERVATION_ENABLED=False)
    d = compute_quote_decision(
        s, mid=100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(),
        best_bid=99.0, best_ask=101.0, bid_size=90.0, ask_size=10.0,
    )
    # microprice computed and persisted (for later measurement) …
    assert d.microprice == pytest.approx(100.8, abs=1e-9)
    # … but reservation used mid, not microprice.
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_missing_depth_falls_back_to_mid_even_with_flag_on() -> None:
    """Flag on, but depth unavailable → reservation uses mid. microprice
    is None in the decision (nothing to persist)."""
    s = _settings(MICROPRICE_RESERVATION_ENABLED=True)
    d = compute_quote_decision(
        s, mid=100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(),
        # No best_bid/best_ask/sizes — legacy call shape.
    )
    assert d.microprice is None
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_zero_depth_falls_back_to_mid() -> None:
    """Flag on, bid_size==ask_size==0 → microprice=None → fallback to mid."""
    s = _settings(MICROPRICE_RESERVATION_ENABLED=True)
    d = compute_quote_decision(
        s, mid=100.0, position_qty=0.0, vol_bps=0.0, toxicity=_tox(),
        best_bid=99.0, best_ask=101.0, bid_size=0.0, ask_size=0.0,
    )
    assert d.microprice is None
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


def test_legacy_three_arg_call_still_works() -> None:
    """Pre-refactor callers (and test fixtures) call
    ``compute_quote_decision(settings, mid, pos, vol, tox)`` without
    depth kwargs. That must keep working identically — microprice is
    just None."""
    s = _settings(MICROPRICE_RESERVATION_ENABLED=True)
    d = compute_quote_decision(s, 100.0, 0.0, 0.0, _tox())
    assert d.microprice is None
    assert d.reservation_price == pytest.approx(100.0, abs=1e-9)


# --------------------- Interaction with inventory skew -----------------


def test_inventory_skew_applies_to_microprice_reference() -> None:
    """When microprice is the reference, inventory skew is applied to
    microprice — NOT to mid. With a long position and heavy bids,
    skew lowers reservation and microprice raises it; the net effect
    depends on magnitudes but the computation must be consistent."""
    s = _settings(
        MICROPRICE_RESERVATION_ENABLED=True,
        INVENTORY_SKEW_COEFF_BPS=10.0,
        MAX_ABS_POSITION=1.0,
    )
    # Long half the cap; skew would lower reservation by 5 bps.
    # Heavy bids → microprice = 100.8.
    # reservation = 100.8 - 10 * 0.5 * 100.8 / 10000 = 100.8 - 0.0504 = 100.7496
    d = compute_quote_decision(
        s, mid=100.0, position_qty=0.5, vol_bps=0.0, toxicity=_tox(),
        best_bid=99.0, best_ask=101.0, bid_size=90.0, ask_size=10.0,
    )
    assert d.microprice == pytest.approx(100.8, abs=1e-9)
    expected = 100.8 - 10.0 * 0.5 * 100.8 / 10_000.0
    assert d.reservation_price == pytest.approx(expected, abs=1e-9)


# --------------------- Schema v13 sanity ------------------------------


def test_quote_decisions_table_has_microprice_column() -> None:
    """Schema v13 adds the column for offline analysis. Existing DBs
    are migrated via ``ALTER TABLE`` in the init path."""
    import os, tempfile, uuid
    from pathlib import Path
    from app.storage import Storage

    path = Path(tempfile.gettempdir()) / f"mm_v13_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {"DATABASE_URL": f"sqlite:///{path.as_posix()}"}
    )
    try:
        storage = Storage(s)
        storage.init_schema()
        with storage._lock:
            with storage.connection() as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(quote_decisions)").fetchall()]
        assert "microprice" in cols, f"microprice column missing; cols = {cols}"
    finally:
        try:
            storage.close()
        except Exception:
            pass
        path.unlink(missing_ok=True)
