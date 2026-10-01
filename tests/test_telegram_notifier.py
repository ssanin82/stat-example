"""Tests for the outbound Telegram notifier.

Coverage:
* disabled-when-no-token: every method is a cheap no-op, no thread, no IO
* notify_ops formats the severity prefix correctly
* notify_trade_fill enqueues; coalesce window batches multiple fills
* token-bucket throttle drops over-cap and reports the suppressed count
* notify_ops_blocking uses the sync HTTP path (for pre-exit hooks)
* HTTP errors are swallowed and logged
"""

from __future__ import annotations

import threading
import time
from typing import Any

from app.telegram_notifier import TelegramNotifier
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides: Any) -> UnitTestSettings:
    base = {
        "EXCHANGE": "bluefin",
        "SYMBOL": "SUI-PERP",
        "BLUEFIN_PRIVATE_KEY": "00" * 32,
        "BLUEFIN_ACCOUNT_ADDRESS": "0x" + "ab" * 32,
        "TELEGRAM_BOT_TOKEN": "TESTTOKEN",
        "TELEGRAM_OPS_CHAT_ID": "-100111",
        "TELEGRAM_TRADES_CHAT_ID": "-100222",
        # tight thresholds for fast tests
        "TELEGRAM_FILL_COALESCE_SECONDS": 0.2,
        "TELEGRAM_TRADE_RATE_PER_SECOND": 100.0,  # effectively unlimited
        "TELEGRAM_TRADE_BURST": 10,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


class _CapturingHttp:
    """Records all HTTP POSTs the notifier issues; mimics
    ``http_post`` injection signature."""

    def __init__(self, status: int = 200, body: str = '{"ok":true}') -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._status = status
        self._body = body
        self._lock = threading.Lock()

    def __call__(self, url: str, body: dict[str, Any]) -> tuple[int, str]:
        with self._lock:
            self.calls.append((url, dict(body)))
        return self._status, self._body


def _wait_until(predicate, timeout_s: float = 1.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def test_disabled_when_no_token_is_total_noop() -> None:
    s = _settings(TELEGRAM_BOT_TOKEN="")
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    assert n.enabled is False
    n.start()  # no-op when disabled
    n.notify_ops("INFO", "x", "y")
    n.notify_trade_fill({"side": "BUY", "symbol": "X"})
    n.notify_ops_blocking("CRITICAL", "x", "y")
    n.stop()
    assert http.calls == [], "no HTTP calls when disabled"


def test_disabled_when_no_chat_configured() -> None:
    s = _settings(TELEGRAM_OPS_CHAT_ID="", TELEGRAM_TRADES_CHAT_ID="")
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    assert n.enabled is False


def test_notify_ops_formats_severity_prefix() -> None:
    s = _settings()
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    n.start()
    try:
        n.notify_ops("CRITICAL", "watchdog_fired", "deadlock detected")
        assert _wait_until(lambda: len(http.calls) >= 1)
    finally:
        n.stop()
    url, body = http.calls[0]
    assert "sendMessage" in url
    assert body["chat_id"] == "-100111"
    text = body["text"]
    assert "CRITICAL" in text
    assert "watchdog_fired" in text
    assert "deadlock detected" in text


def test_notify_ops_unknown_severity_falls_back_to_info() -> None:
    s = _settings()
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    n.start()
    try:
        n.notify_ops("HOWDY", "evt", "msg")
        _wait_until(lambda: len(http.calls) >= 1)
    finally:
        n.stop()
    assert len(http.calls) == 1
    text = http.calls[0][1]["text"]
    assert "INFO" in text


def test_trade_fill_coalesce_batches_multiple_fills() -> None:
    s = _settings(TELEGRAM_FILL_COALESCE_SECONDS=0.3)
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    n.start()
    try:
        # Fire three fills inside the coalesce window.
        n.notify_trade_fill({"side": "BUY", "symbol": "SUI", "price": 1, "size": 10})
        n.notify_trade_fill({"side": "BUY", "symbol": "SUI", "price": 1.01, "size": 5})
        n.notify_trade_fill({"side": "SELL", "symbol": "SUI", "price": 1.02, "size": 7})
        # Wait for the window to elapse and the batch to flush.
        _wait_until(lambda: len(http.calls) >= 1, timeout_s=2.0)
    finally:
        n.stop()
    # The three fills should produce ONE message (a batch), not three.
    assert len(http.calls) == 1, f"expected 1 batched send; got {len(http.calls)}"
    text = http.calls[0][1]["text"]
    assert "FILLS x3" in text
    # All three sides represented.
    assert text.count("FILL ") == 3


def test_trade_fill_no_coalesce_emits_individually() -> None:
    s = _settings(TELEGRAM_FILL_COALESCE_SECONDS=0.0)
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    n.start()
    try:
        n.notify_trade_fill({"side": "BUY", "symbol": "SUI", "price": 1, "size": 10})
        n.notify_trade_fill({"side": "SELL", "symbol": "SUI", "price": 1.01, "size": 5})
        _wait_until(lambda: len(http.calls) >= 2, timeout_s=2.0)
    finally:
        n.stop()
    assert len(http.calls) == 2


def test_token_bucket_drops_over_cap_and_appends_suffix() -> None:
    # Cap at burst=2, refill at 0.5/s (slow). Send 4 trade fills back to
    # back; the first 2 should send, next 2 dropped (suppressed).
    s = _settings(
        TELEGRAM_FILL_COALESCE_SECONDS=0.0,  # immediate send
        TELEGRAM_TRADE_RATE_PER_SECOND=0.5,
        TELEGRAM_TRADE_BURST=2,
    )
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    n.start()
    try:
        for i in range(4):
            n.notify_trade_fill({"side": "BUY", "symbol": "SUI", "price": i, "size": 1})
        # Give worker time to process all 4.
        time.sleep(0.3)
        # Now emit a NEW notify_trade_event well after the bucket has time
        # to get one token back. Force a refill window:
        time.sleep(2.5)  # ~1.25 tokens accumulated -> 1 available
        n.notify_trade_event("synthetic")
        _wait_until(lambda: len(http.calls) >= 3, timeout_s=2.0)
    finally:
        n.stop()
    # At least 2 immediate fills landed; the 5th send carries the suffix.
    assert len(http.calls) >= 3
    # Find the message that has the suppressed-count suffix.
    suffix_messages = [
        body["text"] for _, body in http.calls if "prior fills suppressed" in body["text"]
    ]
    assert suffix_messages, "expected at least one message with suppressed-count suffix"


def test_notify_ops_blocking_bypasses_queue() -> None:
    """Sync path: HTTP call happens immediately, no thread needed."""
    s = _settings()
    http = _CapturingHttp()
    # Don't start the worker thread — verify blocking path works
    # standalone (which is the watchdog pre-exit use case).
    n = TelegramNotifier(s, http_post=http)
    n.notify_ops_blocking("CRITICAL", "watchdog_fired", "msg")
    # Synchronous: call should already be recorded.
    assert len(http.calls) == 1
    assert "CRITICAL" in http.calls[0][1]["text"]


def test_send_failure_is_swallowed() -> None:
    s = _settings()

    def bad_http(url: str, body: dict[str, Any]) -> tuple[int, str]:
        raise RuntimeError("network down")

    n = TelegramNotifier(s, http_post=bad_http)
    n.start()
    try:
        n.notify_ops("INFO", "x", "y")
        # Worker thread runs; exception should be caught, not propagate.
        time.sleep(0.1)
    finally:
        n.stop()
    # No assertion crash → test passes. No HTTP call recorded by capture.


def test_send_non_200_status_logs_but_does_not_raise() -> None:
    s = _settings()
    http = _CapturingHttp(status=400, body='{"ok":false,"description":"Bad Request"}')
    n = TelegramNotifier(s, http_post=http)
    n.start()
    try:
        n.notify_ops("INFO", "x", "y")
        _wait_until(lambda: len(http.calls) >= 1)
    finally:
        n.stop()
    # Telegram returned 400; notifier swallows it.
    assert len(http.calls) == 1


def test_payload_attached_to_ops_message() -> None:
    s = _settings()
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    n.start()
    try:
        n.notify_ops("INFO", "x", "y", {"session_id": "abcd1234", "fills": 3})
        _wait_until(lambda: len(http.calls) >= 1)
    finally:
        n.stop()
    text = http.calls[0][1]["text"]
    assert "session_id" in text
    assert "abcd1234" in text


def test_long_message_truncated() -> None:
    s = _settings()
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    n.start()
    try:
        # 10000-char message; should be truncated to <= 3800 + truncation marker.
        n.notify_ops("INFO", "big", "x" * 10000)
        _wait_until(lambda: len(http.calls) >= 1)
    finally:
        n.stop()
    text = http.calls[0][1]["text"]
    assert len(text) <= 3900
    assert "[truncated]" in text


def test_ops_chat_only_mode() -> None:
    """Token + ops chat, no trades chat → trades methods become no-ops."""
    s = _settings(TELEGRAM_TRADES_CHAT_ID="")
    http = _CapturingHttp()
    n = TelegramNotifier(s, http_post=http)
    assert n.enabled is True
    n.start()
    try:
        n.notify_trade_fill({"side": "BUY", "symbol": "X"})
        time.sleep(0.3)
        n.notify_ops("INFO", "evt", "msg")
        _wait_until(lambda: len(http.calls) >= 1)
    finally:
        n.stop()
    # Only the ops message went out.
    assert len(http.calls) == 1
    assert http.calls[0][1]["chat_id"] == "-100111"
