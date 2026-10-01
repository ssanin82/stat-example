"""Unit tests for the postmortem ``connectivity_health`` section.

Two coverage layers:

1. **Synthetic orders DataFrame** — verifies the three orders-table
   detectors (BUG-023 / BUG-024 / orphan fills) catch their target
   signatures and don't false-positive on benign rows. Fast,
   deterministic, no I/O.

2. **In-memory log scan** — writes a few synthetic JSON-formatted
   bot-log lines to a temp file, runs ``scan_log``, asserts the
   exception / connectivity-event / ack-timing counts match the
   hand-crafted log content.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from tools.postmortem.sections.connectivity_health import (
    ConnectivityFindings,
    detect_connectivity_findings,
    render_html_section,
    render_markdown_section,
    scan_log,
)


# ---------------------------------------------------------------------------
# Orders-table detectors
# ---------------------------------------------------------------------------


@dataclass
class _SnapStub:
    """Minimal SnapshotData-like object for detectors."""

    orders: pd.DataFrame
    events: pd.DataFrame
    state_current: dict


def _make_orders_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("ts_created", "ts_sent", "ts_ack", "ts_cancel_requested"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def test_bug_23_orphan_signature_detected() -> None:
    orders = _make_orders_df(
        [
            {
                # BUG-23 target: ts_ack set, ts_cancel_requested empty,
                # cancel_reason=gone_on_exchange.
                "ts_created": "2026-05-14T08:28:37.417Z",
                "ts_sent": "2026-05-14T08:28:37.418Z",
                "ts_ack": "2026-05-14T08:28:37.425Z",
                "ts_cancel_requested": None,
                "ts_closed": "2026-05-14T08:28:37.426Z",
                "order_id_exchange": "111",
                "side": "SELL",
                "price": 1.1958,
                "status": "CANCELED",
                "cancel_reason": "gone_on_exchange",
            },
            {
                # Benign: a normal cancel — has ts_cancel_requested
                # AND ts_ack, cancel_reason=ws:CANCELED. Must NOT be
                # picked up.
                "ts_created": "2026-05-14T08:28:38Z",
                "ts_sent": "2026-05-14T08:28:38Z",
                "ts_ack": "2026-05-14T08:28:38Z",
                "ts_cancel_requested": "2026-05-14T08:28:38Z",
                "ts_closed": "2026-05-14T08:28:38Z",
                "order_id_exchange": "222",
                "side": "BUY",
                "price": 1.1953,
                "status": "CANCELED",
                "cancel_reason": "ws:CANCELED",
            },
        ]
    )
    snap = _SnapStub(
        orders=orders, events=pd.DataFrame(), state_current={}
    )
    findings = detect_connectivity_findings(snap)
    assert findings.bug_23_count == 1
    assert findings.bug_23_examples[0].order_id_exchange == "111"


def test_bug_24_phantom_signature_detected() -> None:
    orders = _make_orders_df(
        [
            {
                # BUG-24 target: ts_ack empty, cancel_reason=gone_on_exchange.
                "ts_created": "2026-05-14T08:32:27.649Z",
                "ts_sent": "2026-05-14T08:32:27.650Z",
                "ts_ack": None,
                "ts_cancel_requested": "2026-05-14T08:32:27.653Z",
                "ts_closed": "2026-05-14T08:32:28.184Z",
                "order_id_exchange": None,
                "side": "SELL",
                "price": 1.1958,
                "status": "CANCELED",
                "cancel_reason": "gone_on_exchange",
            },
            {
                # Benign rejection: ts_ack empty, cancel_reason different.
                # Must NOT be picked up.
                "ts_created": "2026-05-14T08:32:28Z",
                "ts_sent": "2026-05-14T08:32:28Z",
                "ts_ack": None,
                "ts_cancel_requested": None,
                "ts_closed": "2026-05-14T08:32:28Z",
                "order_id_exchange": None,
                "side": "BUY",
                "price": 1.196,
                "status": "REJECTED",
                "cancel_reason": "post_only_would_cross:...",
            },
        ]
    )
    snap = _SnapStub(
        orders=orders, events=pd.DataFrame(), state_current={}
    )
    findings = detect_connectivity_findings(snap)
    assert findings.bug_24_count == 1


def test_orphan_fill_signature_detected() -> None:
    same_ts = "2026-05-14T08:28:40.747Z"
    orders = _make_orders_df(
        [
            {
                # Orphan fill: ts_created = ts_sent = ts_ack and FILLED.
                "ts_created": same_ts,
                "ts_sent": same_ts,
                "ts_ack": same_ts,
                "ts_cancel_requested": None,
                "ts_closed": "2026-05-14T08:28:40.884Z",
                "order_id_exchange": "999",
                "side": "SELL",
                "price": 1.1958,
                "status": "FILLED",
                "cancel_reason": None,
            },
            {
                # Normal fill: timestamps differ.
                "ts_created": "2026-05-14T08:00:00.000Z",
                "ts_sent": "2026-05-14T08:00:00.001Z",
                "ts_ack": "2026-05-14T08:00:00.010Z",
                "ts_cancel_requested": None,
                "ts_closed": "2026-05-14T08:00:01.000Z",
                "order_id_exchange": "888",
                "side": "BUY",
                "price": 1.20,
                "status": "FILLED",
                "cancel_reason": None,
            },
        ]
    )
    snap = _SnapStub(
        orders=orders, events=pd.DataFrame(), state_current={}
    )
    findings = detect_connectivity_findings(snap)
    assert findings.orphan_fill_count == 1
    assert findings.orphan_fill_examples[0].order_id_exchange == "999"


def test_clean_session_returns_empty() -> None:
    # Cancel-to-close latency 10 ms — well under the 200 ms threshold
    # in ``_detect_slow_cancel_tail`` so it stays out of findings.
    orders = _make_orders_df(
        [
            {
                "ts_created": "2026-05-14T08:00:00.000Z",
                "ts_sent": "2026-05-14T08:00:00.000Z",
                "ts_ack": "2026-05-14T08:00:00.000Z",
                "ts_cancel_requested": "2026-05-14T08:00:01.000Z",
                "ts_closed": "2026-05-14T08:00:01.010Z",
                "order_id_exchange": "1",
                "side": "BUY",
                "price": 1.20,
                "status": "CANCELED",
                "cancel_reason": "ws:CANCELED",
            }
        ]
    )
    snap = _SnapStub(
        orders=orders, events=pd.DataFrame(), state_current={}
    )
    findings = detect_connectivity_findings(snap)
    assert findings.has_any_findings() is False
    # Renderer returns empty when nothing to show.
    assert render_markdown_section(findings) == ""
    assert render_html_section(findings) == ""


def test_state_counters_surfaced_from_state_current() -> None:
    snap = _SnapStub(
        orders=pd.DataFrame(),
        events=pd.DataFrame(),
        state_current={
            "reconcile_skip_snapshot_stale_total": 7,
            "place_unconfirmed_critical_total": 0,
            "session_cross_venue_cancel_count": 14,
            # Other fields ignored.
            "unrelated_field": "foo",
        },
    )
    findings = detect_connectivity_findings(snap)
    assert findings.state_counters["reconcile_skip_snapshot_stale_total"] == 7
    assert findings.state_counters["place_unconfirmed_critical_total"] == 0
    assert findings.state_counters["session_cross_venue_cancel_count"] == 14
    assert "unrelated_field" not in findings.state_counters
    # Non-zero counter ⇒ section renders.
    assert findings.has_any_findings() is True
    md = render_markdown_section(findings)
    assert "reconcile_skip_snapshot_stale_total" in md
    assert "**7**" not in md  # value rendered without bold for counters


# ---------------------------------------------------------------------------
# Log scanner
# ---------------------------------------------------------------------------


def _write_log(lines: list[dict]) -> Path:
    """Write a synthetic bot log to a temp file in the same shape as
    journalctl-collected JSON lines: each line is
    ``Mmm DD HH:MM:SS host dtc-bot[123]: {json}``."""
    path = (
        Path(tempfile.gettempdir())
        / f"pm_conn_log_{os.getpid()}_{os.urandom(4).hex()}.txt"
    )
    with path.open("w", encoding="utf-8") as fh:
        for rec in lines:
            fh.write(
                "May 14 08:32:27 host dtc-bot[123]: " + json.dumps(rec) + "\n"
            )
    return path


def test_scan_log_picks_up_exceptions() -> None:
    log_path = _write_log(
        [
            {
                "ts": "2026-05-14 10:56:34,454",
                "level": "ERROR",
                "logger": "app.exchange.okx_public_ws",
                "msg": "okx_public_ws_subscribe_send_failed",
                "exc_info": (
                    "Traceback (most recent call last):\n"
                    '  File "app/x.py", line 10, in foo\n'
                    "WebSocketConnectionClosedException: Connection is already closed."
                ),
            },
            {
                "ts": "2026-05-14 10:56:35,000",
                "level": "INFO",
                "logger": "app.bot",
                "msg": "ordinary tick — no exception",
            },
        ]
    )
    try:
        (
            exc_n,
            exc_ex,
            conn_n,
            conn_ex,
            _conn_counts,
            _ack,
            _log_first_ts,
            _log_last_ts,
        ) = scan_log(log_path)
        assert exc_n == 1
        assert len(exc_ex) == 1
        # Last line of the traceback is the exception type + msg.
        assert (
            "WebSocketConnectionClosedException"
            in exc_ex[0].first_traceback_line
        )
        # Connectivity events ALSO matched the okx_public_ws_subscribe_send_failed
        # pattern. Verify cross-detection.
        assert conn_n >= 1
        labels = {c.label for c in conn_ex}
        assert "OKX public WS subscribe failed" in labels
    finally:
        log_path.unlink(missing_ok=True)


def test_scan_log_picks_up_connectivity_patterns() -> None:
    log_path = _write_log(
        [
            {
                "ts": "2026-05-14 10:00:00,000",
                "level": "CRITICAL",
                "logger": "app.execution",
                "msg": (
                    "place_response_unconfirmed_critical side=SELL"
                    " symbol=SUI-USDT-SWAP cloid=abc"
                ),
            },
            {
                "ts": "2026-05-14 10:00:01,000",
                "level": "WARNING",
                "logger": "app.execution",
                "msg": "okx_api_error code=50011",
            },
            {
                "ts": "2026-05-14 10:00:02,000",
                "level": "INFO",
                "logger": "app.bot",
                "msg": "unrelated info line",
            },
        ]
    )
    try:
        (
            _exc_n,
            _exc_ex,
            conn_n,
            conn_ex,
            conn_counts,
            _ack,
            _log_first_ts,
            _log_last_ts,
        ) = scan_log(log_path)
        assert conn_n == 2
        assert (
            conn_counts.get("Place response UNCONFIRMED (BUG-024)") == 1
        )
        assert conn_counts.get("OKX API error") == 1
        labels = {c.label for c in conn_ex}
        assert "Place response UNCONFIRMED (BUG-024)" in labels
    finally:
        log_path.unlink(missing_ok=True)


def test_scan_log_pairs_ws_bind_with_rest_response() -> None:
    """Two WS-bind / REST-response pairs in opposite order. The first
    pair has WS first (negative delta); the second has REST first
    (positive delta). Both within the 500ms window."""
    log_path = _write_log(
        [
            # Pair A: WS arrives 5ms BEFORE REST.
            {
                "ts": "2026-05-14 08:28:37,440",
                "level": "INFO",
                "logger": "app.execution",
                "msg": "private_order_update_bound_by_cloid side=BUY oid=1 cloid=aaa",
            },
            {
                "ts": "2026-05-14 08:28:37,445",
                "level": "INFO",
                "logger": "httpx",
                "msg": 'HTTP Request: POST https://www.okx.com/api/v5/trade/order "HTTP/1.1 200 "',
            },
            # Pair B: REST arrives 20ms BEFORE WS.
            {
                "ts": "2026-05-14 08:28:38,000",
                "level": "INFO",
                "logger": "httpx",
                "msg": 'HTTP Request: POST https://www.okx.com/api/v5/trade/order "HTTP/1.1 200 "',
            },
            {
                "ts": "2026-05-14 08:28:38,020",
                "level": "INFO",
                "logger": "app.execution",
                "msg": "private_order_update_bound_by_cloid side=SELL oid=2 cloid=bbb",
            },
        ]
    )
    try:
        (_e, _ex, _c, _ce, _cc, ack, _first, _last) = scan_log(log_path)
        assert ack is not None
        assert ack.paired_count == 2
        assert ack.ws_before_rest_count == 1
        assert ack.rest_before_ws_count == 1
        # Median |delta| = mean of {5, 20} = 12.5 (statistics.median
        # of an even-length list is the mean of the two middle).
        assert 5.0 <= ack.median_abs_delta_ms <= 20.0
        assert 5.0 <= ack.max_abs_delta_ms <= 50.0
    finally:
        log_path.unlink(missing_ok=True)


def test_scan_log_handles_empty_or_malformed_lines() -> None:
    path = (
        Path(tempfile.gettempdir())
        / f"pm_conn_log_{os.getpid()}_malformed.txt"
    )
    with path.open("w", encoding="utf-8") as fh:
        fh.write("not a json line at all\n")
        fh.write("\n")  # empty
        fh.write("May 14 host dtc-bot[1]: {invalid json}\n")
        fh.write(
            'May 14 host dtc-bot[1]: {"ts": "x", "level": "INFO", "msg": "ok"}\n'
        )
    try:
        (exc_n, _, conn_n, _, _, ack, first_ts, last_ts) = scan_log(path)
        assert exc_n == 0
        assert conn_n == 0
        assert ack is None
        # Only the well-formed line carries ts="x" — but that's
        # still a non-empty string, so log_first/log_last_ts get
        # populated with whatever the record provided.
        assert first_ts == "x"
        assert last_ts == "x"
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Integration: top-level detector with both data sources
# ---------------------------------------------------------------------------


def test_log_window_warning_when_log_starts_late() -> None:
    """When the snapshot's session started hours before the log's
    first timestamp, the connectivity section should warn that
    findings cover only the tail of the session."""
    # Log spans 11:00 → 11:01 (2 minutes), but session started at
    # 08:00 (3 hours earlier). The pre-log gap should be flagged.
    log_path = _write_log(
        [
            {
                "ts": "2026-05-14 11:00:00,000",
                "level": "INFO",
                "logger": "app.bot",
                "msg": "tick",
            },
            {
                "ts": "2026-05-14 11:01:00,000",
                "level": "INFO",
                "logger": "app.bot",
                "msg": "tick",
            },
        ]
    )
    try:
        snap = SimpleNamespace(
            orders=pd.DataFrame(),
            events=pd.DataFrame(),
            state_current={},
            session_summary={"session_started_at_utc": "2026-05-14T08:00:00Z"},
            meta={"captured_at_utc": "2026-05-14T11:01:30Z"},
        )
        findings = detect_connectivity_findings(snap, log_path=log_path)
        assert findings.log_scanned is True
        assert findings.session_start_iso == "2026-05-14T08:00:00Z"
        assert findings.session_end_iso == "2026-05-14T11:01:30Z"
        # Pre-log gap is ~3h — must trigger the warning.
        assert len(findings.log_window_warnings) >= 1
        joined = " ".join(findings.log_window_warnings)
        assert "NOT covered" in joined
        # has_any_findings should be true via the warning alone, even
        # with no bugs / events / exceptions.
        assert findings.has_any_findings() is True
        md = render_markdown_section(findings)
        assert "Log-window mismatch" in md
    finally:
        log_path.unlink(missing_ok=True)


def test_log_window_warning_quiet_when_log_brackets_session() -> None:
    """Log fully brackets the session → no warning."""
    log_path = _write_log(
        [
            {
                "ts": "2026-05-14 07:59:00,000",
                "level": "INFO",
                "logger": "app.bot",
                "msg": "tick",
            },
            {
                "ts": "2026-05-14 08:02:00,000",
                "level": "INFO",
                "logger": "app.bot",
                "msg": "tick",
            },
        ]
    )
    try:
        snap = SimpleNamespace(
            orders=pd.DataFrame(),
            events=pd.DataFrame(),
            state_current={},
            session_summary={"session_started_at_utc": "2026-05-14T08:00:00Z"},
            meta={"captured_at_utc": "2026-05-14T08:01:30Z"},
        )
        findings = detect_connectivity_findings(snap, log_path=log_path)
        assert findings.log_scanned is True
        assert findings.log_window_warnings == []
    finally:
        log_path.unlink(missing_ok=True)


def test_detect_with_orders_events_and_log() -> None:
    """End-to-end: orders + events + log produces a full findings
    object that ``has_any_findings`` for, and the markdown rendering
    contains the expected section headers."""
    orders = _make_orders_df(
        [
            {
                "ts_created": "2026-05-14T08:00:00Z",
                "ts_sent": "2026-05-14T08:00:00Z",
                "ts_ack": "2026-05-14T08:00:01Z",
                "ts_cancel_requested": None,
                "ts_closed": "2026-05-14T08:00:01Z",
                "order_id_exchange": "777",
                "side": "SELL",
                "price": 1.2,
                "status": "CANCELED",
                "cancel_reason": "gone_on_exchange",
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "ts": "2026-05-14T08:00:00Z",
                "severity": "CRITICAL",
                "event_type": "place_response_unconfirmed",
                "message": "unconfirmed place response — see payload",
            }
        ]
    )
    log_path = _write_log(
        [
            {
                "ts": "2026-05-14 08:00:00,000",
                "level": "ERROR",
                "logger": "app.exchange.okx_client",
                "msg": "okx_api_error",
                "exc_info": (
                    "Traceback (most recent call last):\n"
                    "RuntimeError: synthetic"
                ),
            }
        ]
    )
    try:
        snap = SimpleNamespace(
            orders=orders, events=events, state_current={}
        )
        findings = detect_connectivity_findings(snap, log_path=log_path)
        assert findings.has_any_findings() is True
        assert findings.bug_23_count == 1
        assert findings.error_event_count == 1
        assert findings.exceptions_count == 1
        assert findings.log_scanned is True

        md = render_markdown_section(findings)
        # Every sub-section we exercised should appear.
        assert "Connectivity & state-machine health" in md
        assert "BUG-023 orphan signatures" in md
        assert "ERROR / CRITICAL bot_events" in md
        assert "Caught exceptions with traceback" in md
        # HTML rendering produces something non-empty too.
        html = render_html_section(findings)
        assert "<section" in html
        assert "<h2>" in html
        assert "<table" in html
    finally:
        log_path.unlink(missing_ok=True)
