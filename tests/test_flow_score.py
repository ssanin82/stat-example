"""Unit tests for the flow-direction score (Priority #3 v1).

Production code: ``app/flow_score.py::FlowScoreAccumulator``.

Design invariants pinned here:
- Empty accumulator returns 0.0 toxicity for both sides.
- A pure one-way stream (all BUY aggressors, or all SELL) drives the
  threatened side's score toward 1.0.
- The TFI window correctly prunes old entries via timestamps.
- Streak count is clipped at the configured streak-window length.
- Malformed or zero-size trades are dropped silently.
- Score is always in [0, 1]; clamps on floating-point drift.
"""

from __future__ import annotations

import pytest

from app.enums import Side
from app.flow_score import FlowScoreAccumulator
from app.models import TradePrint


def _trade(ts_ms: int, side: Side, size: float = 1.0, price: float = 100.0) -> TradePrint:
    return TradePrint(
        ts_exchange_ms=ts_ms,
        ts_local_ms=ts_ms,
        price=price,
        size=size,
        aggressor_side=side,
        trade_id=f"t_{ts_ms}",
    )


def _acc(**kw) -> FlowScoreAccumulator:
    defaults = {
        "tfi_window_seconds": 1.0,
        "streak_window_prints": 10,
        "recent_trades_maxlen": 100,
    }
    defaults.update(kw)
    return FlowScoreAccumulator(**defaults)


# ---------- Empty / malformed ----------

def test_empty_accumulator_returns_zero_scores() -> None:
    a = _acc()
    assert a.get_toxicity_score(Side.BUY) == 0.0
    assert a.get_toxicity_score(Side.SELL) == 0.0
    snap = a.snapshot()
    assert snap.buy_toxic_score == 0.0
    assert snap.sell_toxic_score == 0.0
    assert snap.tfi_signed_normalised == 0.0
    assert snap.tfi_window_trade_count == 0
    assert snap.last_trade_ts_ms is None


def test_record_trade_rejects_none_silently() -> None:
    a = _acc()
    a.record_trade(None)  # type: ignore[arg-type]
    assert a.get_toxicity_score(Side.BUY) == 0.0


def test_record_trade_rejects_zero_or_negative_size() -> None:
    a = _acc()
    # Zero size (bad) — should not be recorded.
    a.record_trade(_trade(1000, Side.BUY, size=0.0))
    a.record_trade(_trade(1000, Side.SELL, size=-5.0))
    # One valid trade so the accumulator isn't empty.
    a.record_trade(_trade(1500, Side.BUY, size=1.0))
    assert a.snapshot().tfi_window_trade_count == 1


def test_unknown_side_raises() -> None:
    a = _acc()
    with pytest.raises(ValueError):
        a.get_toxicity_score("NOT_A_SIDE")  # type: ignore[arg-type]


# ---------- One-way flow ----------

def test_pure_buy_flow_drives_sell_side_toxic_to_one() -> None:
    """10 BUY aggressors in a row → BUY pressure saturates → threat to
    SELL quote goes to 1.0 (TFI component 1.0, streak component 1.0)."""
    a = _acc()
    for i in range(10):
        a.record_trade(_trade(1000 + i * 10, Side.BUY, size=1.0))
    # Threat to SELL quote (we'd be selling into aggressive BUY flow).
    assert a.get_toxicity_score(Side.SELL) == pytest.approx(1.0, abs=1e-9)
    # Threat to BUY quote is zero.
    assert a.get_toxicity_score(Side.BUY) == 0.0


def test_pure_sell_flow_drives_buy_side_toxic_to_one() -> None:
    a = _acc()
    for i in range(10):
        a.record_trade(_trade(1000 + i * 10, Side.SELL, size=1.0))
    assert a.get_toxicity_score(Side.BUY) == pytest.approx(1.0, abs=1e-9)
    assert a.get_toxicity_score(Side.SELL) == 0.0


def test_balanced_flow_gives_moderate_streak_zero_tfi() -> None:
    """Alternating BUY/SELL → TFI=0 (cancels), streak broken repeatedly.
    Final print's side gets +0.05 from streak=1/10, no TFI component."""
    a = _acc()
    for i in range(10):
        side = Side.BUY if i % 2 == 0 else Side.SELL
        a.record_trade(_trade(1000 + i * 10, side, size=1.0))
    # Net TFI is 0; streak is 1 (just the last trade). Last trade
    # was sell (i=9), so sell-streak=1, score for BUY-quote side is
    # 0.5 * 0 + 0.5 * (1/10) = 0.05.
    buy_side_score = a.get_toxicity_score(Side.BUY)
    sell_side_score = a.get_toxicity_score(Side.SELL)
    assert buy_side_score == pytest.approx(0.05, abs=1e-9)
    assert sell_side_score == 0.0


# ---------- TFI window pruning ----------

def test_tfi_window_excludes_old_trades() -> None:
    """5 BUYs at t=0, then 5 SELLs at t=5000 (window=1s). TFI window
    only sees the 5 SELLs → threat to BUY side is 1.0."""
    a = _acc(tfi_window_seconds=1.0)
    for i in range(5):
        a.record_trade(_trade(0 + i * 10, Side.BUY, size=1.0))
    for i in range(5):
        a.record_trade(_trade(5000 + i * 10, Side.SELL, size=1.0))
    # Latest trade at 5040 ms; window is 1000 ms. Only the SELLs
    # should count.
    snap = a.snapshot()
    assert snap.tfi_signed_normalised == pytest.approx(-1.0, abs=1e-9)
    assert snap.tfi_window_trade_count == 5


def test_tfi_signed_normalised_between_minus_one_and_plus_one() -> None:
    a = _acc()
    # 7 BUYs + 3 SELLs in window → ratio = +0.4
    for i in range(7):
        a.record_trade(_trade(1000 + i, Side.BUY, size=1.0))
    for i in range(3):
        a.record_trade(_trade(1100 + i, Side.SELL, size=1.0))
    snap = a.snapshot()
    assert snap.tfi_signed_normalised == pytest.approx(0.4, abs=1e-9)
    assert -1.0 <= snap.tfi_signed_normalised <= 1.0


# ---------- Streak feature ----------

def test_streak_caps_at_streak_window_prints() -> None:
    """20 BUYs in a row → streak counter stops at window size (10)."""
    a = _acc(streak_window_prints=10)
    for i in range(20):
        a.record_trade(_trade(i * 10, Side.BUY, size=1.0))
    snap = a.snapshot()
    assert snap.streak_buy_count == 10
    assert snap.streak_sell_count == 0


def test_streak_broken_by_opposite_aggressor() -> None:
    """5 BUYs, then 1 SELL → streak flips to sell=1, buy=0."""
    a = _acc()
    for i in range(5):
        a.record_trade(_trade(i * 10, Side.BUY, size=1.0))
    a.record_trade(_trade(100, Side.SELL, size=1.0))
    snap = a.snapshot()
    assert snap.streak_buy_count == 0
    assert snap.streak_sell_count == 1


# ---------- Snapshot dict shape ----------

def test_snapshot_dict_has_expected_keys() -> None:
    a = _acc()
    a.record_trade(_trade(1000, Side.BUY, size=1.0))
    d = a.snapshot_dict()
    expected = {
        "buy_toxic_score",
        "sell_toxic_score",
        "tfi_signed_normalised",
        "tfi_window_seconds",
        "tfi_window_trade_count",
        "streak_buy_count",
        "streak_sell_count",
        "streak_window_prints",
        "last_trade_ts_ms",
        "trade_history_count",
    }
    assert expected <= d.keys()
    assert d["trade_history_count"] == 1
    assert d["last_trade_ts_ms"] == 1000


# ---------- Score bounds ----------

def test_score_always_in_unit_interval() -> None:
    """Pathological sizes shouldn't push the score outside [0, 1]."""
    a = _acc()
    a.record_trade(_trade(1000, Side.BUY, size=1e9))
    a.record_trade(_trade(1001, Side.BUY, size=1e9))
    score = a.get_toxicity_score(Side.SELL)
    assert 0.0 <= score <= 1.0


# ---------- Edge: maxlen enforcement ----------

def test_trades_deque_respects_maxlen() -> None:
    """When more trades are pushed than maxlen, oldest are evicted."""
    a = _acc(recent_trades_maxlen=50)
    for i in range(100):
        a.record_trade(_trade(i * 10, Side.BUY, size=1.0))
    # Deque capped at 50. Accumulator internal deque is accessible via
    # the snapshot's trade_history_count.
    assert a.snapshot_dict()["trade_history_count"] == 50
