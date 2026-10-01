"""Backtesting subsystem (Phase 2 — v1.4.232).

This package contains the offline replay framework. The bot core
(``app.bot``, ``app.execution``, etc.) does not import anything from
here in production — the replay driver swaps these modules in only
when running a backtest.

Components:

* ``paper_executor`` — implements the ``PerpExchangeAdapter`` Protocol
  against recorded market data. Synthesizes fills using a back-of-
  queue model, emitting ``PrivateOrderUpdateEvent`` /
  ``PrivateFillEvent`` into the bot's private event queue. No threads,
  no network.

Future (Phase 3+):
* ``replay_driver`` — orchestrates events + tick cadence
* ``report`` — accumulates per-tick state for postmortem
* ``fixtures`` — loaders for recorder output

The design is documented in ``backtesting/docs/backtesting.md`` and
``backtesting/docs/execution-plan.md``.
"""

from app.backtest.bot_runner import (
    DEFAULT_REPLAY_WARMUP_SECONDS,
    BotRunner,
    build_bot_runner,
    make_replay_settings,
    seed_state_from_snapshot,
)
from app.backtest.driver import (
    ReplayConfig,
    ReplayHarness,
    ReplayReport,
    TickCallback,
    build_harness,
    replay,
    write_report,
)
from app.backtest.event_stream import RecordedEvent, SortedEventStream
from app.backtest.paper_executor import PaperExecutor, PaperExecutorConfig
from app.backtest.report import (
    MetricsAccumulator,
    TickSample,
    compute_config_hash,
)
from app.backtest.streams import ReplayPrivateStream, ReplayPublicStream

__all__ = [
    "PaperExecutor",
    "PaperExecutorConfig",
    "RecordedEvent",
    "SortedEventStream",
    "ReplayPublicStream",
    "ReplayPrivateStream",
    "ReplayConfig",
    "ReplayReport",
    "ReplayHarness",
    "TickCallback",
    "build_harness",
    "replay",
    "write_report",
    "MetricsAccumulator",
    "TickSample",
    "compute_config_hash",
    "DEFAULT_REPLAY_WARMUP_SECONDS",
    "BotRunner",
    "build_bot_runner",
    "make_replay_settings",
    "seed_state_from_snapshot",
]
