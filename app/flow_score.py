"""Flow-direction / toxicity score from public trade prints (Priority #3 v1).

Background
----------
After the Priority #1 (OB imbalance) and Priority #2 (cross-venue basis
with regime detector) shipped, the remaining strategy-layer lever for
adverse-selection reduction is *flow direction*: classifying whether
current aggressor flow is informed (stay away — pre-empt the fill) or
uninformed (quote tight and collect rebate).

Per ``deep-research/flow-direction-chatgpt.md``, the canonical online
score blends five features (TFI, VPIN, streak, sign-ACF, volume
concentration). v1 of this module ships with the **two simplest and
most predictive** features: short-window TFI and streak length. The
other three are planned for v2 once v1 is validated against live data.

Usage
-----
Single-threaded assumption: the accumulator lives on ``BotState`` and
is updated from the public-WS message handler (hot path) and read from
the bot's quote loop. All operations are O(1) amortised in the sizes
of the bounded buffers. No external locking — BotState's existing
``_lock`` covers the combined update in the WS handler.

- ``record_trade(TradePrint)`` — O(1): append to the deque, evict
  stale entries from the TFI window head.
- ``get_score(side)`` — O(1) amortised: returns a composite toxicity
  score in ``[0, 1]`` for the specified side.

Score interpretation (for the integration layer in ``execution.py``):

- Score > ``FLOW_SCORE_PAUSE_THRESHOLD`` (default 1.1, unreachable in
  v1 — observability-only mode) → pause that side for the configured
  cooldown. Score of 1.0 is reached only when the TFI is fully one-way
  AND the streak hits the configured window length on the toxic side.
- Lower thresholds (0.5-0.75, per research) can be wired in v2 for
  gradient widening / size-shrinking before pausing.

Computing the score
-------------------
Two components, each mapped to ``[0, 1]`` and averaged:

1. **TFI component**: signed aggressor volume over
   ``FLOW_SCORE_TFI_WINDOW_SECONDS`` (default 1 s), normalised against
   the total volume in the window. Returns ``max(0, signed_tfi)`` for
   BUY-toxicity (i.e. buy pressure toxic to our ask) and
   ``max(0, -signed_tfi)`` for SELL-toxicity. Intuition: if 100% of
   the window's volume was BUY aggressors, BUY-toxic = 1.0; if 50/50,
   both sides get 0.0.

2. **Streak component**: ratio of consecutive same-side prints within
   the last ``FLOW_SCORE_STREAK_WINDOW_PRINTS`` (default 10). A full
   10-length BUY streak gives BUY-streak-component = 1.0; mixed
   prints give values below 1.0.

Final score per side: ``(tfi_component + streak_component) / 2``.

No sign-ACF, VPIN, or concentration in v1 — those require more print
history and are more computationally expensive. Revisit after
measuring v1.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from app.enums import Side
from app.models import TradePrint


@dataclass(slots=True)
class FlowScoreSnapshot:
    """Diagnostic snapshot for telemetry / logging. Non-strategy data."""

    buy_toxic_score: float          # [0, 1]
    sell_toxic_score: float         # [0, 1]
    tfi_signed_normalised: float    # [-1, 1]; + = buy-pressure
    tfi_window_seconds: float
    tfi_window_trade_count: int     # how many prints contributed to TFI
    streak_buy_count: int           # consecutive BUY aggressors
    streak_sell_count: int          # consecutive SELL aggressors
    streak_window_prints: int
    last_trade_ts_ms: Optional[int]


class FlowScoreAccumulator:
    """Online computation of the flow-direction toxicity score.

    v1: two features (TFI, streak). v2 will add VPIN, sign-ACF,
    volume concentration and expose a richer snapshot.
    """

    def __init__(
        self,
        *,
        tfi_window_seconds: float = 1.0,
        streak_window_prints: int = 10,
        recent_trades_maxlen: int = 500,
    ) -> None:
        if tfi_window_seconds <= 0:
            raise ValueError("tfi_window_seconds must be positive")
        if streak_window_prints < 2:
            raise ValueError("streak_window_prints must be >= 2")
        if recent_trades_maxlen < 50:
            raise ValueError("recent_trades_maxlen must be >= 50")
        self._tfi_window_ms = int(tfi_window_seconds * 1000.0)
        self._streak_window_prints = int(streak_window_prints)
        # Full deque of recent trades. Used both for streak (simple
        # tail scan) and for TFI window re-hydration when the head
        # ages out. Capacity comes from config so the operator can
        # tune retention depth without code changes.
        self._trades: deque[TradePrint] = deque(maxlen=int(recent_trades_maxlen))
        self._tfi_window_seconds = float(tfi_window_seconds)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def record_trade(self, trade: TradePrint) -> None:
        """Append one trade. Silently drops malformed entries."""
        if trade is None:
            return
        if not (isinstance(trade.size, (int, float)) and trade.size > 0):
            return
        self._trades.append(trade)

    def clear(self) -> None:
        self._trades.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _tfi_signed_normalised(
        self, *, now_ms: Optional[int] = None
    ) -> tuple[float, int]:
        """TFI over the configured window.

        Returns ``(signed_normalised, trade_count)``.
        ``signed_normalised`` is the net aggressor imbalance divided by
        the total volume in the window, bounded to ``[-1, 1]``.
        Positive = buy pressure; negative = sell pressure. Returns
        ``(0.0, 0)`` when the window is empty.
        """
        if not self._trades:
            return 0.0, 0
        ref_ts = now_ms if now_ms is not None else self._trades[-1].ts_local_ms
        cutoff = ref_ts - self._tfi_window_ms
        buy_vol = 0.0
        sell_vol = 0.0
        count = 0
        # Walk from the tail (most recent) backwards; break as soon as
        # we cross the cutoff. Avoids pruning the deque (keep history
        # for streak / future features).
        for t in reversed(self._trades):
            if t.ts_local_ms < cutoff:
                break
            if t.aggressor_side == Side.BUY:
                buy_vol += float(t.size)
            else:
                sell_vol += float(t.size)
            count += 1
        total = buy_vol + sell_vol
        if total <= 0:
            return 0.0, count
        ratio = (buy_vol - sell_vol) / total
        # Clamp defensively for any float drift.
        if ratio > 1.0:
            ratio = 1.0
        elif ratio < -1.0:
            ratio = -1.0
        return ratio, count

    def _streak_counts(self) -> tuple[int, int]:
        """Consecutive same-side aggressor count at the tail, clipped
        to the configured streak window. Returns
        ``(buy_streak, sell_streak)``; at most one of them is > 0 by
        construction (the current tail side). Useful to detect "5
        buys in a row" patterns that presage continuation.
        """
        if not self._trades:
            return 0, 0
        buy_run = 0
        sell_run = 0
        for t in reversed(self._trades):
            if t.aggressor_side == Side.BUY:
                if sell_run > 0:
                    break
                buy_run += 1
                if buy_run >= self._streak_window_prints:
                    break
            else:
                if buy_run > 0:
                    break
                sell_run += 1
                if sell_run >= self._streak_window_prints:
                    break
        return buy_run, sell_run

    def get_toxicity_score(self, side: Side) -> float:
        """Compute the toxicity score for the supplied side (BUY or
        SELL). Returns 0.0 when there is not enough history.

        Score = average of:
          - TFI component: positive fraction of the signed TFI in the
            direction that's toxic to ``side``.
          - Streak component: same-side streak length / streak window.
        """
        if side not in (Side.BUY, Side.SELL):
            raise ValueError(f"unknown side: {side!r}")
        if not self._trades:
            return 0.0
        tfi_signed, _ = self._tfi_signed_normalised()
        # Toxicity mapping: a BUY that lifts our ask is "BUY-toxic"
        # (adverse to our SELL quote). Similarly a SELL print that hits
        # our bid is "SELL-toxic" (adverse to our BUY quote). The
        # caller passes the side OF THEIR QUOTE; we return the threat
        # level to that quote.
        if side == Side.SELL:
            # Threat to our SELL quote = BUY aggressor pressure.
            tfi_component = max(0.0, tfi_signed)
        else:
            tfi_component = max(0.0, -tfi_signed)
        buy_run, sell_run = self._streak_counts()
        if side == Side.SELL:
            streak_component = buy_run / float(self._streak_window_prints)
        else:
            streak_component = sell_run / float(self._streak_window_prints)
        if streak_component > 1.0:
            streak_component = 1.0
        score = 0.5 * tfi_component + 0.5 * streak_component
        if score < 0.0:
            score = 0.0
        elif score > 1.0:
            score = 1.0
        return score

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def snapshot(self) -> FlowScoreSnapshot:
        tfi_signed, tfi_count = self._tfi_signed_normalised()
        buy_run, sell_run = self._streak_counts()
        last_ts = self._trades[-1].ts_local_ms if self._trades else None
        return FlowScoreSnapshot(
            buy_toxic_score=self.get_toxicity_score(Side.SELL),
            sell_toxic_score=self.get_toxicity_score(Side.BUY),
            tfi_signed_normalised=tfi_signed,
            tfi_window_seconds=self._tfi_window_seconds,
            tfi_window_trade_count=tfi_count,
            streak_buy_count=buy_run,
            streak_sell_count=sell_run,
            streak_window_prints=self._streak_window_prints,
            last_trade_ts_ms=last_ts,
        )

    def snapshot_dict(self) -> dict[str, object]:
        s = self.snapshot()
        return {
            "buy_toxic_score": s.buy_toxic_score,
            "sell_toxic_score": s.sell_toxic_score,
            "tfi_signed_normalised": s.tfi_signed_normalised,
            "tfi_window_seconds": s.tfi_window_seconds,
            "tfi_window_trade_count": s.tfi_window_trade_count,
            "streak_buy_count": s.streak_buy_count,
            "streak_sell_count": s.streak_sell_count,
            "streak_window_prints": s.streak_window_prints,
            "last_trade_ts_ms": s.last_trade_ts_ms,
            "trade_history_count": len(self._trades),
        }
