"""Replay driver — Phase 3 (v1.4.234).

Orchestrates the replay loop:

1. Open a fixture via :class:`~app.backtest.event_stream.SortedEventStream`.
2. Construct a :class:`~app.clock.ReplayClock` pinned to the first
   event timestamp.
3. Construct the public + private replay streams + paper executor,
   sharing the clock and a private-event queue.
4. Iterate events; for each: advance the clock, dispatch to the
   right stream, forward book/trade updates to the paper executor,
   then fire ``on_tick`` callback(s) for every tick interval that
   has elapsed.
5. Emit a deterministic :class:`ReplayReport` JSON summary.

Key invariants:

* The clock only ever moves forward. ``ReplayClock.advance_to``
  raises on backward jumps.
* Multiple events between ticks are processed before the tick fires
  (the bot sees the latest book state when it decides — same as
  production).
* Two consecutive runs of the same fixture + config produce
  byte-identical reports (modulo ``wall_time_s`` which is the only
  wall-clock-dependent field — tests exclude it from the diff).

Scope note (Phase 3): the driver runs against the **paper executor
only**. Wiring it through the full ``Bot.__init__`` requires state
seeding, dependency stubbing for daemon publishers, and a settings
overlay system — all Phase 4 work. The ``on_tick`` callback is the
hook a future caller will use to drive ``bot.one_tick()``.
"""

from __future__ import annotations

import json
import logging
import time as _stdlib_time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Optional

from app.backtest.event_stream import RecordedEvent, SortedEventStream
from app.backtest.paper_executor import PaperExecutor, PaperExecutorConfig
from app.backtest.report import MetricsAccumulator, TickSample, compute_config_hash
from app.backtest.streams import ReplayPrivateStream, ReplayPublicStream
from app.clock import ReplayClock
from app.models import BestBidAsk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    """Driver knobs. Defaults match the Phase 3 acceptance spec.

    ``tick_interval_s`` is the bot's quote-loop cadence — production
    default 0.5s. Increase for cheaper replays of bursty data;
    decrease to study sub-tick reaction behaviour.

    ``warmup_seconds`` skips the bot's tick callback for the first
    N seconds after replay start. Lets pre-seed data (basis EWMA,
    flow score deque, etc.) populate before any trading decisions
    are made.

    ``okx_symbol`` / ``binance_symbol`` are the contract / spot
    symbols expected in the fixture. The driver uses these only for
    decoration of the parsed events — the actual venue routing is
    fixed by the recorded data.

    ``allow_gaps`` propagates to the fixture loader; with gaps the
    replay still runs but the report flags the count so consumers
    can decide whether to trust it.
    """

    tick_interval_s: float = 0.5
    warmup_seconds: float = 0.0
    okx_symbol: str = "TON-USDT-SWAP"
    binance_symbol: str = "TONUSDT"
    allow_gaps: bool = False
    paper: PaperExecutorConfig = field(default_factory=PaperExecutorConfig)
    # Phase 4 (v1.4.235): per-tick detail emission. Off by default —
    # a 5-minute fixture × 0.5s tick interval = 600 entries, which is
    # fine in JSON, but at 50ms cadence × 1h = 72,000 entries the
    # output gets unwieldy. The Phase 4 CLI exposes ``--verbose`` for
    # this; baseline-comparison tests don't need per-tick.
    include_per_tick: bool = False

    def __post_init__(self) -> None:
        if self.tick_interval_s <= 0:
            raise ValueError("tick_interval_s must be > 0")
        if self.warmup_seconds < 0:
            raise ValueError("warmup_seconds must be >= 0")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReplayReport:
    """Deterministic summary of one replay run.

    Phase 3 fields: event-flow counters, stream counters, paper-
    executor totals, tick counts.

    Phase 4 additions: ``summary`` (fills, volume, PnL, drawdown,
    inventory extrema, time-with-orders %), ``gates_fired``
    (currently empty — Phase 4b will populate from real ``Bot``),
    optional ``per_tick`` detail array, and a ``config_hash`` so
    baseline-comparison tests can refuse to compare across config
    drift.

    Schema versioned via ``schema_version``. Bumped to **2** in
    Phase 4 to signal the added fields; baseline files written under
    v1 must be re-captured.
    """

    schema_version: int = 2
    bot_version: str = ""
    fixture_path: str = ""
    fixture_session_id: str = ""
    fixture_recorder_version: str = ""
    fixture_schema_version: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    config_hash: str = ""

    # Event flow
    first_t_ns: int = 0
    last_t_ns: int = 0
    total_events: int = 0
    events_by_source: dict[str, int] = field(default_factory=dict)
    gaps_reported: int = 0

    # Streams
    public_stream: dict[str, int] = field(default_factory=dict)
    private_stream: dict[str, int] = field(default_factory=dict)

    # Paper executor
    paper_executor: dict[str, float] = field(default_factory=dict)

    # Ticks
    ticks_scheduled: int = 0
    ticks_executed: int = 0
    ticks_skipped_warmup: int = 0

    # Phase 4 — strategy-regression-ready surface.
    summary: dict[str, Any] = field(default_factory=dict)
    gates_fired: dict[str, int] = field(default_factory=dict)
    per_tick: list[dict[str, Any]] = field(default_factory=list)

    # BUG-G (v1.5.303) — fill-attribution diagnostic. Lets the report
    # explain a low fill count (quiet market vs back-of-queue vs not
    # quoting) instead of leaving it a mystery. See
    # PaperExecutor.fill_attribution_snapshot.
    fill_attribution: dict[str, float] = field(default_factory=dict)

    # BUG-B + BUG-H (v1.5.303) — honesty caveats. Records the fill
    # source actually used (paper executor in --with-bot mode, with the
    # bot's own private-fill path suppressed) and the mark-price proxy
    # (mid, not a recorded mark feed) so drawdown figures aren't taken
    # as venue-exact. See _finalize_report.
    caveats: dict[str, Any] = field(default_factory=dict)

    # v1.5.304 (audit §4.4 — P0 #4+#5) — per-regime rollup keyed by the
    # FSM mode label, and a downsampled net-edge-per-minute series. Both
    # are only populated in ``--with-bot`` mode (the regime FSM and the
    # AQC controller live on the bot's ``state``); empty in paper-only
    # replay. Turns "7 fills over 24k ticks" into "which regime were we
    # in, and what net edge did the controller observe there."
    per_regime: dict[str, Any] = field(default_factory=dict)
    net_edge_per_min_series: list[dict[str, Any]] = field(default_factory=list)

    # Wall-clock observability (excluded from determinism check)
    wall_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministically-keyed dict for JSON dump."""
        return {
            "schema_version": self.schema_version,
            "bot_version": self.bot_version,
            "fixture": {
                "path": self.fixture_path,
                "session_id": self.fixture_session_id,
                "recorder_version": self.fixture_recorder_version,
                "schema_version": self.fixture_schema_version,
            },
            "config": self.config,
            "config_hash": self.config_hash,
            "events": {
                "first_t_ns": self.first_t_ns,
                "last_t_ns": self.last_t_ns,
                "total": self.total_events,
                "by_source": dict(sorted(self.events_by_source.items())),
                "gaps_reported": self.gaps_reported,
            },
            "streams": {
                "public": dict(sorted(self.public_stream.items())),
                "private": dict(sorted(self.private_stream.items())),
            },
            "paper_executor": dict(sorted(self.paper_executor.items())),
            "ticks": {
                "scheduled": self.ticks_scheduled,
                "executed": self.ticks_executed,
                "skipped_warmup": self.ticks_skipped_warmup,
            },
            "summary": dict(sorted(self.summary.items())),
            "gates_fired": dict(sorted(self.gates_fired.items())),
            "fill_attribution": dict(sorted(self.fill_attribution.items())),
            "caveats": dict(sorted(self.caveats.items())),
            "per_regime": dict(sorted(self.per_regime.items())),
            "net_edge_per_min_series": self.net_edge_per_min_series,
            "per_tick": self.per_tick,
            "wall_time_s": self.wall_time_s,
        }

    def to_json(
        self,
        *,
        exclude_wall_time: bool = False,
        exclude_per_tick: bool = False,
    ) -> str:
        d = self.to_dict()
        if exclude_wall_time:
            d.pop("wall_time_s", None)
        if exclude_per_tick:
            d["per_tick"] = []
        return json.dumps(d, indent=2, sort_keys=False)


# ---------------------------------------------------------------------------
# ReplayHarness
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReplayHarness:
    """The wired-together replay components. The driver owns one of
    these; tests can poke at the fields directly."""

    clock: ReplayClock
    paper: PaperExecutor
    public_stream: ReplayPublicStream
    private_stream: ReplayPrivateStream
    private_event_queue: "Queue[Any]"


def build_harness(
    *,
    start_t_ns: int,
    config: ReplayConfig,
    private_event_queue: Optional["Queue[Any]"] = None,
) -> ReplayHarness:
    """Wire a clock + paper executor + streams into one harness."""
    clock = ReplayClock(start_t_ns=start_t_ns)
    sink: "Queue[Any]" = private_event_queue if private_event_queue is not None else Queue()
    paper = PaperExecutor(
        clock=clock,
        private_event_sink=sink,
        config=config.paper,
        symbol=config.okx_symbol,
    )
    public = ReplayPublicStream(
        clock=clock,
        on_bbo=lambda _bba: None,  # forward via the driver loop
        on_trade=None,
        okx_symbol=config.okx_symbol,
        binance_symbol=config.binance_symbol,
    )
    private = ReplayPrivateStream(
        clock=clock,
        event_sink=sink,
        okx_symbol=config.okx_symbol,
    )
    return ReplayHarness(
        clock=clock,
        paper=paper,
        public_stream=public,
        private_stream=private,
        private_event_queue=sink,
    )


# ---------------------------------------------------------------------------
# Driver entry point
# ---------------------------------------------------------------------------


# Callback signature: ``(t_ns: int, harness: ReplayHarness) -> None``.
# Called at every tick-interval boundary AFTER all events at-or-before
# that boundary have been dispatched. Used by Phase 4+ to invoke
# ``bot.one_tick()``.
TickCallback = Callable[[int, ReplayHarness], None]


def replay(
    fixture_dir: Path,
    *,
    config: Optional[ReplayConfig] = None,
    on_tick: Optional[TickCallback] = None,
    bot_version: str = "",
    bot_runner: Optional[Any] = None,
) -> ReplayReport:
    """Run a replay over the given fixture.

    Parameters
    ----------
    fixture_dir:
        Directory containing ``manifest.json`` + per-source jsonl.gz.
    config:
        :class:`ReplayConfig` instance. Defaults preserve the Phase 3
        spec defaults.
    on_tick:
        Optional callback invoked at every tick interval. Phase 3 has
        no built-in caller — Phase 4 will use it to drive
        ``bot.one_tick()``.
    bot_version:
        Recorded into the report. Phase 3 callers pass
        ``app.__version__``; Phase 4+ may resolve from the Bot
        instance.
    bot_runner:
        Optional :class:`~app.backtest.bot_runner.BotRunner` (Phase 4b).
        When provided, the driver:

        - Uses the runner's harness instead of building a fresh one
          (so the bot's deps and the driver's harness are the same).
        - Calls ``bot_runner.one_tick()`` automatically at every tick
          interval (in addition to any user-supplied ``on_tick``).
        - Populates ``report.gates_fired`` from
          ``bot_runner.extract_gates_fired()`` at finalize.
        - Tracks ``time_in_quote_pct`` from ``bot_runner.is_quoting()``
          per-tick.

    Returns
    -------
    :class:`ReplayReport` — also writable to JSON via
    ``report.to_json()``.
    """
    config = config if config is not None else ReplayConfig()
    fixture_dir = Path(fixture_dir)

    events = SortedEventStream(fixture_dir, allow_gaps=config.allow_gaps)
    if bot_runner is not None:
        # Use the runner's pre-built harness — the bot is already
        # wired against it.
        harness = bot_runner.harness
        # Sanity: the runner's clock should start at the fixture's
        # first event timestamp. If it doesn't, the caller built the
        # runner against the wrong clock anchor — fail loud.
        if harness.clock._t_ns != events.first_t_ns:
            raise ValueError(
                f"bot_runner clock starts at {harness.clock._t_ns} ns but "
                f"fixture's first event is at {events.first_t_ns} ns — "
                f"build the runner with start_t_ns=events.first_t_ns"
            )
    else:
        harness = build_harness(
            start_t_ns=events.first_t_ns,
            config=config,
        )

    # v1.5.315 — install THIS replay's harness clock as the module-level
    # clock for the duration of the run. State-layer code reads monotonic
    # time through the module proxy (``app.clock.monotonic`` →
    # ``_module_clock``), not through the bot's injected ``self._clock``.
    # ``Bot.__init__`` sets the module clock to its own ``self._clock``,
    # but a caller that constructs MULTIPLE runners before replaying any
    # of them (e.g. the byte-identical determinism test builds r1 then r2)
    # leaves the module clock pointing at the LAST-built runner. Without
    # this re-install, ``replay(r1)`` drives r1's clock via ``advance_to``
    # while ``state.py`` stamps freshness / gap / decision-latency
    # receipts from r2's (never-advanced) clock — a mixed-origin delta
    # that makes the freshness gate fire non-deterministically and breaks
    # the determinism contract. Mirrors ``run_replay_with_bot``
    # (bot_runner.py), which already installs the harness clock the same
    # way before constructing its bot. Backtest-only; production has a
    # single bot whose clock is already the module clock.
    from app.clock import set_module_clock

    set_module_clock(harness.clock)

    # Wire the public-stream BBO + trade callbacks into the paper
    # executor so the paper book stays in sync. The BotRunner (if
    # present) installs its own callbacks that ALSO update BotState;
    # we only install the defaults below when no runner is active.
    paper = harness.paper

    if bot_runner is None:
        def _on_bbo(bba: BestBidAsk) -> None:
            # Paper executor only cares about the OKX (trading) venue's
            # touch — Binance is a basis input, not directly tradeable.
            if bba.symbol == config.okx_symbol:
                paper.process_book_event(
                    bid=bba.best_bid if bba.best_bid is not None else 0.0,
                    ask=bba.best_ask if bba.best_ask is not None else 0.0,
                    bid_size=bba.bid_size if bba.bid_size is not None else 0.0,
                    ask_size=bba.ask_size if bba.ask_size is not None else 0.0,
                )

        def _on_trade(price: float, size: float, side: str, venue: str) -> None:
            if venue == "okx":
                paper.process_trade_event(price=price, size=size, side=side)

        harness.public_stream._on_bbo = _on_bbo  # late-bind callbacks
        harness.public_stream._on_trade = _on_trade

    paper_echo = {
        "sim_place_latency_s": config.paper.sim_place_latency_s,
        "sim_cancel_latency_s": config.paper.sim_cancel_latency_s,
        "fee_maker_bps": config.paper.fee_maker_bps,
        "fee_taker_bps": config.paper.fee_taker_bps,
        "queue_policy": config.paper.queue_policy,
    }
    # Backtest-only fill-rate lever. Echoed ONLY when non-default (≠1.0)
    # so a default run's config block + config_hash stay BYTE-IDENTICAL
    # to every pre-knob replay — preserving the determinism contract and
    # keeping captured baselines valid (the golden config_hash regression
    # in tests/backtest/test_strategy_regressions.py compares against
    # baselines that predate this knob). When the lever IS engaged the
    # fraction is recorded for reproducibility + the Results panel, and
    # folds into config_hash so distinct fractions hash distinctly.
    # (Mirrors the ≠1.0-only summary line in scripts/backtest/replay.py.)
    if config.paper.queue_ahead_fraction != 1.0:
        paper_echo["queue_ahead_fraction"] = config.paper.queue_ahead_fraction
    config_dict = {
        "tick_interval_s": config.tick_interval_s,
        "warmup_seconds": config.warmup_seconds,
        "okx_symbol": config.okx_symbol,
        "binance_symbol": config.binance_symbol,
        "allow_gaps": config.allow_gaps,
        "include_per_tick": config.include_per_tick,
        "paper_executor": paper_echo,
    }
    report = ReplayReport(
        bot_version=bot_version,
        fixture_path=str(fixture_dir),
        fixture_session_id=events.session_id,
        fixture_recorder_version=events.recorder_version,
        fixture_schema_version=events.schema_version,
        config=config_dict,
        config_hash=compute_config_hash(config_dict),
        first_t_ns=events.first_t_ns,
        last_t_ns=events.last_t_ns,
        gaps_reported=events.total_gaps,
    )

    # Phase 4 metric accumulator — sampled once per tick from the
    # paper executor's state. Always populated; the summary derives
    # from it at finalize. Per-tick array only retained if the
    # config opts in.
    metrics = MetricsAccumulator()

    last_volume_seen = [0.0]  # mutable cell for closure
    # v1.5.304 (audit §4.4) — per-tick deltas for the per-regime rollup.
    # Same mutable-cell-over-closure pattern as ``last_volume_seen``.
    last_fills_seen = [0]
    last_realized_pnl_seen = [0.0]
    last_fees_seen = [0.0]

    def _sample_tick(t_ns: int) -> None:
        """Snapshot paper executor state at this tick boundary."""
        # Compute unrealised PnL from current mark + position.
        mark = harness.paper._mark_price()
        unrealised = 0.0
        if mark is not None and harness.paper._position_qty != 0.0:
            unrealised = (
                (mark - harness.paper._avg_entry_price)
                * harness.paper._position_qty
            )
        equity = (
            harness.paper._realized_pnl
            + unrealised
            - harness.paper._fees_total
        )
        # v1.5.304 (audit §4.4 — P0 #4+#5). In --with-bot mode read the
        # regime FSM label + AQC controller reading off the live bot
        # state so the tick can be attributed to a regime. Attr paths
        # verified against BotState / RegimeControllerState /
        # ActiveQuotingController (NOT the audit's prose ``state.regime_mode``,
        # which does not exist). All guarded — paper-only replay leaves
        # these None and the per_regime / net_edge blocks stay empty.
        regime_label: Optional[str] = None
        aqc_aggr: Optional[float] = None
        aqc_net_edge: Optional[float] = None
        if bot_runner is not None:
            bs = getattr(bot_runner, "state", None)
            if bs is not None:
                rc = getattr(bs, "regime_controller", None)
                mode = getattr(rc, "mode", None) if rc is not None else None
                if mode is not None:
                    regime_label = getattr(mode, "value", None)
                aqc = getattr(bs, "active_quoting_controller", None)
                if aqc is not None:
                    lvl = getattr(aqc, "aggression_level", None)
                    if lvl is not None:
                        try:
                            aqc_aggr = float(lvl)
                        except (TypeError, ValueError):
                            aqc_aggr = None
                    ne = getattr(aqc, "last_observed_net_edge_per_min_usd", None)
                    if ne is not None:
                        try:
                            aqc_net_edge = float(ne)
                        except (TypeError, ValueError):
                            aqc_net_edge = None
        sample = TickSample(
            t_ns=t_ns,
            position_qty=harness.paper._position_qty,
            realized_pnl_usd=harness.paper._realized_pnl,
            unrealized_pnl_usd=unrealised,
            fees_total_usd=harness.paper._fees_total,
            best_bid=harness.paper._best_bid,
            best_ask=harness.paper._best_ask,
            open_orders=len(harness.paper._orders),
            equity_usd=equity,
            regime_mode_label=regime_label,
            aqc_aggression_level=aqc_aggr,
            aqc_observed_net_edge_per_min_usd=aqc_net_edge,
        )
        # Delta of cumulative fill notional since the last sample —
        # exact, sourced from the paper executor's running counter.
        cur_volume = harness.paper.total_fill_volume_usd
        vol_delta = cur_volume - last_volume_seen[0]
        last_volume_seen[0] = cur_volume
        # Companion deltas for per-regime attribution (audit §4.4).
        cur_fills = harness.paper.fills_emitted
        fills_delta = cur_fills - last_fills_seen[0]
        last_fills_seen[0] = cur_fills
        cur_rpnl = harness.paper.current_realized_pnl()
        rpnl_delta = cur_rpnl - last_realized_pnl_seen[0]
        last_realized_pnl_seen[0] = cur_rpnl
        cur_fees = harness.paper.current_fees()
        fees_delta = cur_fees - last_fees_seen[0]
        last_fees_seen[0] = cur_fees
        metrics.observe_tick(
            sample,
            fill_volume_usd_delta=vol_delta,
            fills_delta=fills_delta,
            realized_pnl_delta=rpnl_delta,
            fees_delta=fees_delta,
        )
        if config.include_per_tick:
            report.per_tick.append(sample.to_dict())

    wall_started = _stdlib_time.perf_counter()
    next_tick_t_ns = events.first_t_ns  # tick at-or-after the first event

    # Tick interval in ns for integer comparison against clock state.
    tick_interval_ns = int(config.tick_interval_s * 1e9)
    warmup_ns = int(config.warmup_seconds * 1e9)
    warmup_end_t_ns = events.first_t_ns + warmup_ns

    def _maybe_fire_ticks(up_to_t_ns: int) -> None:
        """Fire every tick callback whose scheduled time is at or
        before ``up_to_t_ns``."""
        nonlocal next_tick_t_ns
        while next_tick_t_ns <= up_to_t_ns:
            report.ticks_scheduled += 1
            # Advance clock to the tick time so callback sees the
            # correct monotonic value.
            if next_tick_t_ns > harness.clock._t_ns:
                harness.clock.advance_to(next_tick_t_ns)
            if next_tick_t_ns < warmup_end_t_ns:
                report.ticks_skipped_warmup += 1
            else:
                # Phase 4b: drive the real bot's tick BEFORE the
                # user-supplied on_tick (the user callback observes
                # post-tick state).
                if bot_runner is not None:
                    bot_runner.one_tick()
                if on_tick is not None:
                    on_tick(next_tick_t_ns, harness)
                _sample_tick(next_tick_t_ns)
                report.ticks_executed += 1
            next_tick_t_ns += tick_interval_ns

    by_source = report.events_by_source
    total = 0
    last_seen_t_ns = events.first_t_ns - 1  # before any event

    for event in events:
        # Bug-022 safety: if a recorder file is somehow out-of-order,
        # `ReplayClock.advance_to` would raise — we'd rather log + skip.
        if event.t_recv_ns < last_seen_t_ns:
            logger.warning(
                "skipping out-of-order event (t=%d < prev=%d, source=%s)",
                event.t_recv_ns,
                last_seen_t_ns,
                event.source,
            )
            continue
        last_seen_t_ns = event.t_recv_ns
        # Fire any pending ticks at/before this event.
        _maybe_fire_ticks(event.t_recv_ns)
        # Advance the clock to the event instant.
        if event.t_recv_ns > harness.clock._t_ns:
            harness.clock.advance_to(event.t_recv_ns)
        # Dispatch.
        if event.source in ("okx_public", "binance_public"):
            harness.public_stream.deliver(event)
        elif event.source == "okx_private":
            harness.private_stream.deliver(event)
        elif event.source == "okx_rest":
            harness.paper.update_from_rest_snapshot(event.msg)
        # else: unknown source — counted only.
        by_source[event.source] = by_source.get(event.source, 0) + 1
        total += 1

    # Fire any straggler ticks at-or-before the last event timestamp.
    _maybe_fire_ticks(events.last_t_ns)

    report.total_events = total
    report.public_stream = {
        "events_seen": harness.public_stream.events_seen,
        "bbo_dispatched": harness.public_stream.bbo_dispatched,
        "trades_dispatched": harness.public_stream.trades_dispatched,
        "events_skipped": harness.public_stream.events_skipped,
        # BUG-E: per-reason breakdown of events_skipped (subscribe acks,
        # off-channel frames, …) so a big skip count is attributable.
        "skip_reasons": dict(sorted(harness.public_stream.skip_reasons.items())),
    }
    report.private_stream = {
        "events_seen": harness.private_stream.events_seen,
        "order_updates_dispatched": harness.private_stream.order_updates_dispatched,
        "fills_dispatched": harness.private_stream.fills_dispatched,
        # In --with-bot mode the recorded private stream is parsed but
        # NOT forwarded to the bot's queue (paper-exec is authoritative);
        # these count what was held back. Both 0 in paper-only mode.
        "order_updates_suppressed": harness.private_stream.order_updates_suppressed,
        "fills_suppressed": harness.private_stream.fills_suppressed,
        "events_skipped": harness.private_stream.events_skipped,
        # BUG-E: per-reason breakdown — confirms a large skip count is
        # benign off-channel filtering (unknown_channel:account, …).
        "skip_reasons": dict(sorted(harness.private_stream.skip_reasons.items())),
    }
    report.paper_executor = {
        "acks_emitted": float(harness.paper.acks_emitted),
        "fills_emitted": float(harness.paper.fills_emitted),
        "rejects_emitted": float(harness.paper.rejects_emitted),
        "cancels_emitted": float(harness.paper.cancels_emitted),
        "position_qty": harness.paper.current_position_qty(),
        "realized_pnl": harness.paper.current_realized_pnl(),
        "fees_total": harness.paper.current_fees(),
        "avg_entry_price": harness.paper.current_avg_entry(),
        "total_fill_volume_usd": harness.paper.total_fill_volume_usd,
    }

    # Phase 4: finalise summary + gates.
    report.summary = metrics.finalize(
        fills=harness.paper.fills_emitted,
        realized_pnl_usd=harness.paper.current_realized_pnl(),
        fees_total_usd=harness.paper.current_fees(),
    )
    # v1.5.304 (audit §4.4 — P0 #4+#5) — per-regime rollup + net-edge
    # series. Both empty in paper-only replay (no regime label / AQC
    # reading was stamped on any tick); populated in --with-bot mode.
    _extras = metrics.finalize_extras()
    report.per_regime = _extras["per_regime"]
    report.net_edge_per_min_series = _extras["net_edge_per_min_series"]
    # Phase 4b: populate gates_fired from the real bot's state when
    # a BotRunner is wired in; otherwise leave empty (paper-only).
    if bot_runner is not None:
        try:
            report.gates_fired = bot_runner.extract_gates_fired()
        except Exception:
            logger.exception("extract_gates_fired_failed")
            report.gates_fired = {}
    else:
        report.gates_fired = {}

    # BUG-G (v1.5.303) — fill-attribution diagnostic. Explains a low
    # fill count (quiet market vs back-of-queue vs not quoting) instead
    # of leaving the operator to guess.
    report.fill_attribution = harness.paper.fill_attribution_snapshot()

    # BUG-B + BUG-H (v1.5.303) — honesty caveats. Keep downstream readers
    # from mistaking paper-sim figures for venue-exact truth.
    report.caveats = {
        # BUG-B: which fill stream is authoritative. The paper executor
        # ALWAYS drives position/PnL in replay. In --with-bot mode the
        # recorded venue private-fill stream is deliberately suppressed
        # (parsed but not forwarded, and NOT cross-checked against paper
        # fills). A full shadow-mode diff of recorded-vs-simulated fills
        # is a separate P3 follow-up.
        "authoritative_fill_source": "paper_executor",
        "recorded_private_fills_cross_checked": False,
        "recorded_private_fills_suppressed": int(
            harness.private_stream.fills_suppressed
        ),
        "recorded_private_order_updates_suppressed": int(
            harness.private_stream.order_updates_suppressed
        ),
        "with_bot_mode": bool(bot_runner is not None),
        # BUG-H: mark price for unrealized-PnL / drawdown is the book mid
        # (or last trade if no book), NOT a recorded venue mark feed. So
        # max_drawdown_usd and any unrealized figures are mid-proxy
        # estimates, not venue-exact mark-to-market. Wiring a recorded
        # mark-price feed is a separate P3 follow-up.
        "mark_price_source": "mid_proxy",
        "drawdown_is_mid_proxy": True,
    }

    report.wall_time_s = round(_stdlib_time.perf_counter() - wall_started, 6)

    return report


def write_report(report: ReplayReport, report_path: Path) -> None:
    """Write the report JSON to ``report_path`` with sorted keys."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report.to_json())
        f.write("\n")


__all__ = [
    "ReplayConfig",
    "ReplayReport",
    "ReplayHarness",
    "TickCallback",
    "build_harness",
    "replay",
    "write_report",
]
