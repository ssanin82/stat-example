"""Latency-measurement toolkit tests.

Plan reference: latency-toolkit addition (after Phase 1-3 of
plans/20260420-binance-move/plan.md).

Two distributions to verify:

1. **L2 tick latency** — exchange-event-time → local-receive-time,
   captured via `PublicWsTimingTracker` and fed by the trading
   public-WS message handler.
2. **Order place-to-ack RTT** — captured by `OrderRttTracker`,
   ingested at the transport-completion point in the execution
   layer.

Both must:
- Preserve microsecond precision when the value is < 1 ms.
- Expose rolling distributional stats (min / median / p95 / p99 / max).
- Be displayed by the `/latency` Telegram command.
"""

from __future__ import annotations

import json
import queue
import time
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.market_data_timing import PublicWsTimingTracker
from app.order_rtt_tracker import OrderRttTracker
from tests.settings_helpers import UnitTestSettings


# ---------------------------------------------------------------------------
# OrderRttTracker
# ---------------------------------------------------------------------------


def test_order_rtt_tracker_empty_summary() -> None:
    t = OrderRttTracker(max_samples=64)
    summary = t.summary()
    assert summary["sample_count"] == 0
    assert summary["window_size"] == 64
    assert summary["total_observed_count"] == 0
    assert summary["min_ms"] is None
    assert summary["median_ms"] is None
    assert summary["p95_ms"] is None


def test_order_rtt_tracker_basic_percentiles() -> None:
    t = OrderRttTracker(max_samples=1000)
    for v in range(1, 101):  # 1.0, 2.0, ..., 100.0 ms
        t.ingest(rtt_ms=float(v))
    s = t.summary()
    assert s["sample_count"] == 100
    assert s["min_ms"] == 1.0
    assert s["max_ms"] == 100.0
    # p50 of 1..100 = ~50.5 (linear interpolation, percentile-by-rank
    # gives 50.5 with our nearest-rank-with-interp implementation)
    assert 49.0 <= s["median_ms"] <= 51.0
    # p95 of 1..100 ≈ 95.05
    assert 94.0 <= s["p95_ms"] <= 96.0


def test_order_rtt_tracker_microsecond_precision_preserved() -> None:
    """A < 1 ms RTT (rare but possible on a fast same-region run) must
    survive ingest with sub-millisecond precision intact.
    """
    t = OrderRttTracker(max_samples=64)
    t.ingest(rtt_ms=0.347)  # 347 μs
    t.ingest(rtt_ms=0.512)  # 512 μs
    s = t.summary()
    assert s["min_ms"] == 0.347
    assert s["max_ms"] == 0.512


def test_order_rtt_tracker_rejects_negative_and_huge() -> None:
    """Defensive bounds: negative RTT (clock skew) and 60s+ values
    (transport hang) get clamped + flagged with outcome=transport_error.
    """
    t = OrderRttTracker(max_samples=64)
    t.ingest(rtt_ms=-1.0)        # negative — clamped to 0
    t.ingest(rtt_ms=120_000.0)   # 2 minutes — clamped + flagged
    t.ingest(rtt_ms=5.0)         # normal
    # By default summary filters to outcome="accepted" — the two
    # outliers are flagged transport_error and excluded.
    s = t.summary()
    assert s["sample_count"] == 1
    assert s["min_ms"] == 5.0
    assert s["max_ms"] == 5.0
    # Including all outcomes shows the outliers
    s_all = t.summary(outcome_filter=None)
    assert s_all["sample_count"] == 3


def test_order_rtt_tracker_rolling_window_eviction() -> None:
    """Once the deque is full, old samples are evicted (FIFO)."""
    t = OrderRttTracker(max_samples=10)
    for v in range(1, 21):  # 20 samples into a window of 10
        t.ingest(rtt_ms=float(v))
    s = t.summary()
    assert s["sample_count"] == 10
    assert s["window_size"] == 10
    assert s["total_observed_count"] == 20
    # Only the last 10 (11..20) survived
    assert s["min_ms"] == 11.0
    assert s["max_ms"] == 20.0


def test_order_rtt_tracker_op_label_recorded() -> None:
    """The op label survives ingest so future per-op percentiles
    (e.g. cancel-RTT) can be split out without an additional tracker.
    """
    t = OrderRttTracker(max_samples=64)
    t.ingest(rtt_ms=5.0, op="place")
    t.ingest(rtt_ms=3.0, op="cancel")
    # Currently summary() doesn't filter by op (kept simple); op is
    # there for future expansion. Just verify ingest accepted both.
    assert t._samples[0].op == "place"
    assert t._samples[1].op == "cancel"


# ---------------------------------------------------------------------------
# Trading-public-WS feeds the timing tracker (BUG: previously didn't)
# ---------------------------------------------------------------------------


def test_binance_trading_public_ws_feeds_timing_tracker() -> None:
    """Phase-2 wiring fix: the trading public-WS module must feed
    `state.binance_public_ws_timing` so /latency can show L2 tick
    distribution.

    Pre-fix, my new `binance_trading_public_ws.py` parsed bookTicker
    messages and emitted BestBidAsk via on_bbo, but never called
    `tracker.ingest`. /latency would show 0 samples even after hours
    of running.
    """
    from app.exchange.binance_trading_public_ws import (
        BinanceTradingPublicStream,
    )
    from app.state import BotState

    s = UnitTestSettings.model_validate(
        {
            "EXCHANGE": "binance",
            "BINANCE_API_KEY": "k",
            "BINANCE_API_SECRET": "sec",
            "SYMBOL": "DOGEUSDT",
        }
    )
    state = BotState(s)
    # The state may or may not have a tracker pre-attached depending
    # on the bot init order; ensure one is there for this test.
    if state.binance_public_ws_timing is None:
        state.binance_public_ws_timing = PublicWsTimingTracker(
            symbol="DOGEUSDT", max_samples=128
        )

    captured_bbo: list = []

    def on_bbo(bbo) -> None:
        captured_bbo.append(bbo)

    stream = BinanceTradingPublicStream(
        s, state, "DOGEUSDT", on_bbo
    )
    # Synthetic bookTicker message with an exchange event time 5 ms
    # in the past (relative to wall clock at test execution).
    now_ms = int(time.time() * 1000)
    msg = {
        "e": "bookTicker",
        "u": 12345,
        "s": "DOGEUSDT",
        "b": "0.4231",
        "B": "1000",
        "a": "0.4232",
        "A": "1500",
        "T": now_ms - 5,
        "E": now_ms - 5,
    }
    stream._handle_raw_message(json.dumps(msg))

    # The on_bbo callback fired
    assert len(captured_bbo) == 1
    # The timing tracker captured the sample
    summary = state.binance_public_ws_timing.summary()
    assert summary["total_samples"] == 1
    one_way = summary["exchange_to_local_receive_ms"]
    # Expected one-way ≈ (now − (now − 5)) = 5 ms; allow some slack
    # for the wall-clock read between event-time synthesis and ingest.
    assert one_way["min"] is not None
    assert 0.0 <= one_way["min"] <= 50.0
    assert one_way["max"] is not None


def test_binance_trading_public_ws_skips_wrong_symbol() -> None:
    """Sanity: messages for a different symbol must not feed the
    tracker (each stream is a single-symbol subscription).
    """
    from app.exchange.binance_trading_public_ws import (
        BinanceTradingPublicStream,
    )
    from app.state import BotState

    s = UnitTestSettings.model_validate(
        {
            "EXCHANGE": "binance",
            "BINANCE_API_KEY": "k",
            "BINANCE_API_SECRET": "sec",
            "SYMBOL": "DOGEUSDT",
        }
    )
    state = BotState(s)
    state.binance_public_ws_timing = PublicWsTimingTracker(
        symbol="DOGEUSDT", max_samples=128
    )
    stream = BinanceTradingPublicStream(s, state, "DOGEUSDT", lambda b: None)
    msg = {
        "e": "bookTicker", "s": "BTCUSDT",
        "b": "75000", "B": "1", "a": "75001", "A": "1",
        "T": int(time.time() * 1000), "E": int(time.time() * 1000),
    }
    stream._handle_raw_message(json.dumps(msg))
    assert state.binance_public_ws_timing.summary()["total_samples"] == 0


# ---------------------------------------------------------------------------
# /latency Telegram command
# ---------------------------------------------------------------------------


def _make_poller_with_state(state, exec_layer=None):
    """Helper mirroring tests/test_telegram_commands.py shape."""
    from app.telegram_commands import TelegramCommandPoller
    from app.storage import Storage

    s = UnitTestSettings.model_validate(
        {
            "TELEGRAM_BOT_TOKEN": "t",
            "TELEGRAM_OPS_CHAT_ID": "-100111",
            "TELEGRAM_TRADES_CHAT_ID": "-100222",
            "TELEGRAM_ALLOWED_USER_IDS": "1001",
            "TELEGRAM_LONG_POLL_TIMEOUT_SECONDS": 1,
            "TELEGRAM_COMMAND_MIN_INTERVAL_SECONDS": 0.0,
        }
    )
    bot = MagicMock()
    bot._exec = exec_layer
    storage = MagicMock(recent_fills=lambda limit=100: [])
    posts: list = []

    def http_post(url: str, body: dict[str, Any]) -> tuple[int, str]:
        posts.append((url, dict(body)))
        return 200, '{"ok":true}'

    def http_get(url, params, timeout):
        return 200, '{"ok":true,"result":[]}'

    p = TelegramCommandPoller(
        s, bot=bot, state=state, storage=storage, notifier=None,
        http_post=http_post, http_get=http_get,
        exit_fn=lambda code: None,
    )
    return p, posts


def _msg(text: str = "/latency") -> dict:
    return {
        "update_id": 42,
        "message": {
            "from": {"id": 1001},
            "chat": {"id": 1001},
            "text": text,
        },
    }


def test_cmd_latency_with_no_data_shows_n_a() -> None:
    """Pre-warmup: no L2 ticks observed, no orders placed yet. The
    command must reply something useful (don't crash, don't lie).
    """
    from app.state import BotState

    s = UnitTestSettings.model_validate({})
    state = BotState(s)
    p, posts = _make_poller_with_state(state, exec_layer=None)
    p._handle_update(_msg("/latency"))
    assert posts, "expected a reply"
    text = posts[-1][1]["text"]
    # Both sections present
    assert "L2 tick" in text or "l2" in text.lower()
    assert "order RTT" in text or "place" in text.lower()


def test_cmd_latency_renders_sub_millisecond_in_microseconds() -> None:
    """A 0.347 ms RTT should display as ``347 μs`` (microsecond
    precision visible to the operator), not ``0 ms`` (millisecond
    rounding hides the data).
    """
    from app.state import BotState

    s = UnitTestSettings.model_validate({})
    state = BotState(s)
    exec_layer = MagicMock()
    exec_layer.order_rtt_summary.return_value = {
        "sample_count": 5,
        "window_size": 1024,
        "total_observed_count": 5,
        "min_ms": 0.347,
        "median_ms": 0.412,
        "p95_ms": 0.851,
        "p99_ms": 0.973,
        "max_ms": 1.234,
        "mean_ms": 0.500,
    }
    p, posts = _make_poller_with_state(state, exec_layer=exec_layer)
    p._handle_update(_msg("/latency"))
    text = posts[-1][1]["text"]
    # Sub-1ms values render in μs
    assert "347 μs" in text
    assert "412 μs" in text
    # Multi-ms value renders with 2 decimal places
    assert "1.23 ms" in text


def test_cmd_latency_renders_realistic_binance_tokyo_distribution() -> None:
    """Realistic shape: 5-15 ms median RTT on AWS Tokyo same-region.
    Verify the formatter picks the right scale.
    """
    from app.state import BotState

    s = UnitTestSettings.model_validate({})
    state = BotState(s)
    exec_layer = MagicMock()
    exec_layer.order_rtt_summary.return_value = {
        "sample_count": 100,
        "window_size": 1024,
        "total_observed_count": 100,
        "min_ms": 4.823,
        "median_ms": 7.142,
        "p95_ms": 12.480,
        "p99_ms": 18.301,
        "max_ms": 47.512,
        "mean_ms": 8.005,
    }
    p, posts = _make_poller_with_state(state, exec_layer=exec_layer)
    p._handle_update(_msg("/latency"))
    text = posts[-1][1]["text"]
    assert "7.14 ms" in text   # median
    assert "12.48 ms" in text  # p95
    assert "18.30 ms" in text  # p99


def test_cmd_latency_sample_count_reported() -> None:
    """Operator needs to know whether the distribution is meaningful
    (5 samples = noise) or settled (1000+ samples = signal).
    """
    from app.state import BotState

    s = UnitTestSettings.model_validate({})
    state = BotState(s)
    exec_layer = MagicMock()
    exec_layer.order_rtt_summary.return_value = {
        "sample_count": 753,
        "window_size": 1024,
        "total_observed_count": 1453,
        "min_ms": 4.0,
        "median_ms": 7.0,
        "p95_ms": 12.0,
        "p99_ms": 18.0,
        "max_ms": 47.0,
        "mean_ms": 8.0,
    }
    p, posts = _make_poller_with_state(state, exec_layer=exec_layer)
    p._handle_update(_msg("/latency"))
    text = posts[-1][1]["text"]
    assert "753" in text  # sample_count
    assert "1024" in text  # window_size
    assert "1453" in text  # total_observed_count
