"""Per-tick state accumulator + summary metrics — Phase 4 (v1.4.235).

Phase 3 shipped a thin ``ReplayReport`` capturing event-flow counters
and paper-executor totals. Phase 4 adds the strategy-regression-
ready surface from ``backtesting.md §5.4``:

  * **Per-tick samples** (position, PnL, fees, book state) so a
    baseline-comparison test can detect drift on any of them.
  * **Summary metrics** that are stable enough to assert on across
    intentional and unintentional changes:

    - ``fills`` — total fill count
    - ``volume_usd`` — Σ(fill_sz × fill_px)
    - ``realized_pnl_usd`` — final realised PnL
    - ``max_drawdown_usd`` — max peak-to-trough excursion of
      (realised + unrealised − fees)
    - ``max_inventory_qty_long`` / ``max_inventory_qty_short``
    - ``fees_total_usd`` — signed sum (negative = net rebate)
    - ``time_with_orders_pct`` — % of ticks where any order rested
      (paper-only proxy for ``time_in_quote_pct``; the real metric
      requires Bot integration in Phase 4b).

  * **Gate-firing counters** (``gates_fired`` dict). Empty until
    Phase 4b wires the real ``Bot`` in — paper-only replay has no
    strategy to fire gates.

  * **Config hash** — deterministic SHA-256 of canonical config JSON,
    so baseline tests can refuse to compare across config drift.

Determinism contract: two replays of the same fixture + config
produce byte-identical reports (modulo ``wall_time_s``). Verified
by ``test_replay_report.py::test_two_runs_byte_identical_with_metrics``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Per-tick sample
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TickSample:
    """One per-tick state snapshot used by the report's per-tick
    detail mode (--verbose). Tracked even when per-tick output is
    disabled — the summary metrics are derived from these samples."""

    t_ns: int
    position_qty: float
    realized_pnl_usd: float
    unrealized_pnl_usd: float
    fees_total_usd: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    open_orders: int
    # Equity = realised + unrealised − fees. Used by max-drawdown.
    equity_usd: float
    # v1.5.304 (audit §4.1 / §4.4 — P0 #4+#5) — regime + AQC per-tick
    # context. Populated only in ``--with-bot`` mode (the regime FSM and
    # the AQC controller live on the bot's ``state``); ``None`` in
    # paper-only replay. These drive the report's ``per_regime`` rollup
    # and ``net_edge_per_min_series``. Source attrs (verified against
    # BotState, not the audit's prose names): ``regime_mode_label`` ←
    # ``state.regime_controller.mode.value``; ``aqc_aggression_level`` ←
    # ``state.active_quoting_controller.aggression_level``;
    # ``aqc_observed_net_edge_per_min_usd`` ←
    # ``state.active_quoting_controller.last_observed_net_edge_per_min_usd``.
    regime_mode_label: Optional[str] = None
    aqc_aggression_level: Optional[float] = None
    aqc_observed_net_edge_per_min_usd: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_ns": self.t_ns,
            "position_qty": self.position_qty,
            "realized_pnl_usd": self.realized_pnl_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "fees_total_usd": self.fees_total_usd,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "open_orders": self.open_orders,
            "equity_usd": self.equity_usd,
            "regime_mode_label": self.regime_mode_label,
            "aqc_aggression_level": self.aqc_aggression_level,
            "aqc_observed_net_edge_per_min_usd": self.aqc_observed_net_edge_per_min_usd,
        }


# ---------------------------------------------------------------------------
# Metrics accumulator
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MetricsAccumulator:
    """Running totals + extrema across the replay.

    Updated once per tick from the paper executor's state. Final
    summary computed via :meth:`finalize`.
    """

    ticks_total: int = 0
    ticks_with_orders: int = 0
    peak_equity_usd: float = 0.0
    max_drawdown_usd: float = 0.0
    max_inventory_qty_long: float = 0.0
    max_inventory_qty_short: float = 0.0
    # Filled by the paper executor's running counters at finalize.
    final_fills: int = 0
    final_volume_usd: float = 0.0
    final_realized_pnl_usd: float = 0.0
    final_fees_total_usd: float = 0.0

    # Running fills/volume sums — updated from paper executor between
    # ticks via the on_tick callback.
    cumulative_volume_usd: float = 0.0
    last_fills_seen: int = 0

    # v1.5.304 (audit §4.4 — P0 #4+#5). Per-regime rollup keyed by the
    # FSM mode label, and a downsampled net-edge-per-minute series. Both
    # are only populated in ``--with-bot`` mode (paper-only ticks carry
    # ``regime_mode_label is None`` and never touch these). Stored as
    # plain floats so the dataclass stays ``slots=True``-friendly and
    # JSON-trivial; integer-valued buckets are cast at finalize.
    per_regime: dict[str, dict[str, float]] = field(default_factory=dict)
    net_edge_per_min_series: list[dict[str, Any]] = field(default_factory=list)
    # Downsample cadence bookkeeping — last fixture-time we sampled the
    # net-edge series (ns). The controller's own metric is already a
    # per-minute rolling figure, so 1 sample / 60 s fixture-time keeps
    # the series compact + deterministic without losing signal.
    _net_edge_last_sample_t_ns: int = 0

    def observe_tick(
        self,
        sample: TickSample,
        fill_volume_usd_delta: float,
        *,
        fills_delta: int = 0,
        realized_pnl_delta: float = 0.0,
        fees_delta: float = 0.0,
    ) -> None:
        """Update the accumulator with one tick of state.

        The keyword deltas (``fills_delta`` / ``realized_pnl_delta`` /
        ``fees_delta``) default to 0.0 so paper-only callers and older
        tests that pass only ``fill_volume_usd_delta`` keep working;
        the ``--with-bot`` driver supplies them for the per-regime
        attribution.
        """
        self.ticks_total += 1
        if sample.open_orders > 0:
            self.ticks_with_orders += 1
        # Equity tracking + drawdown.
        if sample.equity_usd > self.peak_equity_usd:
            self.peak_equity_usd = sample.equity_usd
        drawdown = sample.equity_usd - self.peak_equity_usd
        if drawdown < self.max_drawdown_usd:
            self.max_drawdown_usd = drawdown
        # Inventory extrema (signed).
        if sample.position_qty > self.max_inventory_qty_long:
            self.max_inventory_qty_long = sample.position_qty
        if sample.position_qty < self.max_inventory_qty_short:
            self.max_inventory_qty_short = sample.position_qty
        # Volume.
        self.cumulative_volume_usd += fill_volume_usd_delta

        # --- Per-regime rollup (--with-bot only; label None otherwise) ---
        label = sample.regime_mode_label
        if label is not None:
            bucket = self.per_regime.get(label)
            if bucket is None:
                bucket = {
                    "ticks": 0.0,
                    "ticks_with_orders": 0.0,
                    "fills": 0.0,
                    "volume_usd": 0.0,
                    "realized_pnl_usd": 0.0,
                    "fees_total_usd": 0.0,
                    "aggression_sum": 0.0,
                    "aggression_n": 0.0,
                }
                self.per_regime[label] = bucket
            bucket["ticks"] += 1.0
            if sample.open_orders > 0:
                bucket["ticks_with_orders"] += 1.0
            bucket["fills"] += float(fills_delta)
            bucket["volume_usd"] += fill_volume_usd_delta
            bucket["realized_pnl_usd"] += realized_pnl_delta
            bucket["fees_total_usd"] += fees_delta
            if sample.aqc_aggression_level is not None:
                bucket["aggression_sum"] += float(sample.aqc_aggression_level)
                bucket["aggression_n"] += 1.0

        # --- Net-edge-per-minute series (downsample to >= 60 s) ---
        net_edge = sample.aqc_observed_net_edge_per_min_usd
        if net_edge is not None and (
            not self.net_edge_per_min_series
            or sample.t_ns - self._net_edge_last_sample_t_ns >= 60_000_000_000
        ):
            self.net_edge_per_min_series.append(
                {
                    "t_ns": int(sample.t_ns),
                    "net_edge_per_min_usd": round(float(net_edge), 6),
                }
            )
            self._net_edge_last_sample_t_ns = sample.t_ns

    def finalize(
        self,
        *,
        fills: int,
        realized_pnl_usd: float,
        fees_total_usd: float,
    ) -> dict[str, Any]:
        """Return the summary dict — frozen at the moment of call."""
        time_with_orders_pct = (
            (self.ticks_with_orders / self.ticks_total * 100.0)
            if self.ticks_total > 0 else 0.0
        )
        return {
            "fills": fills,
            "volume_usd": round(self.cumulative_volume_usd, 6),
            "realized_pnl_usd": round(realized_pnl_usd, 6),
            "fees_total_usd": round(fees_total_usd, 6),
            "max_drawdown_usd": round(self.max_drawdown_usd, 6),
            "max_inventory_qty_long": round(self.max_inventory_qty_long, 6),
            "max_inventory_qty_short": round(self.max_inventory_qty_short, 6),
            "peak_equity_usd": round(self.peak_equity_usd, 6),
            "ticks_total": self.ticks_total,
            "ticks_with_orders": self.ticks_with_orders,
            "time_with_orders_pct": round(time_with_orders_pct, 4),
        }

    def finalize_extras(self) -> dict[str, Any]:
        """Per-regime rollup + net-edge series — audit §4.4 (P0 #4+#5).

        Kept separate from :meth:`finalize` so the stable ``summary``
        contract (asserted on by baseline-comparison tests) is not
        disturbed. Deterministic: regime labels emitted in sorted
        order; the series is already in fixture-time order. Both blocks
        are empty in paper-only replay (no regime label / AQC reading
        was ever stamped on a tick).
        """
        per_regime_out: dict[str, dict[str, Any]] = {}
        for label in sorted(self.per_regime):
            b = self.per_regime[label]
            ticks = b["ticks"]
            time_in_mode_pct = (
                (ticks / self.ticks_total * 100.0) if self.ticks_total > 0 else 0.0
            )
            avg_aggr = (
                (b["aggression_sum"] / b["aggression_n"])
                if b["aggression_n"] > 0
                else None
            )
            per_regime_out[label] = {
                "ticks": int(ticks),
                "ticks_with_orders": int(b["ticks_with_orders"]),
                "fills": int(b["fills"]),
                "volume_usd": round(b["volume_usd"], 6),
                "realized_pnl_usd": round(b["realized_pnl_usd"], 6),
                "fees_total_usd": round(b["fees_total_usd"], 6),
                "time_in_mode_pct": round(time_in_mode_pct, 4),
                "avg_aggression_level": (
                    round(avg_aggr, 6) if avg_aggr is not None else None
                ),
            }
        return {
            "per_regime": per_regime_out,
            "net_edge_per_min_series": list(self.net_edge_per_min_series),
        }


# ---------------------------------------------------------------------------
# Config hashing — deterministic SHA-256 of canonical config JSON.
# ---------------------------------------------------------------------------


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Stable hash of the config payload. Sorted keys, no whitespace.

    Used by baseline-comparison tests to detect "you ran with a
    different config" cleanly — a config diff invalidates the
    comparison.
    """
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "TickSample",
    "MetricsAccumulator",
    "compute_config_hash",
]
