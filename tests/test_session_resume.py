"""Tests for best-effort session resume.

Coverage:
* disabled-by-default (returns ``fresh``)
* no prior bot_start in storage -> ``no_prior_session``
* prior session within window -> ``resumed`` with restored metrics
* prior session past continuity window -> ``expired``
* corrupted payload -> ``error``, state unchanged
* storage read error -> ``error``, state unchanged
* **regression guard**: behavioral state is NOT touched by the resume

The regression guard is the most important test in this file. It exists
because *restoring behavioral state defeats the deadlock watchdog* —
re-introducing the very latched gates the restart was meant to clear.
If you ever extend ``apply_session_resume`` to set additional fields,
audit them carefully against this test's expectations.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from app.session_resume import (
    SessionResumeResult,
    try_resume_session_from_storage,
    write_bot_start_event,
)
from app.state import BotState
from app.storage import Storage
from tests.settings_helpers import UnitTestSettings


def _settings(**overrides: Any) -> UnitTestSettings:
    base = {
        "EXCHANGE": "bluefin",
        "SYMBOL": "SUI-PERP",
        "BLUEFIN_PRIVATE_KEY": "00" * 32,
        "BLUEFIN_ACCOUNT_ADDRESS": "0x" + "ab" * 32,
        "BOT_RESUME_SESSION_ON_RESTART": True,
        "BOT_SESSION_CONTINUITY_MAX_HOURS": 24.0,
    }
    base.update(overrides)
    return UnitTestSettings.model_validate(base)


def _storage(tmp_path: Path) -> Storage:
    s = _settings(DATABASE_URL=f"sqlite:///{tmp_path / 'trading.db'}")
    st = Storage(s)
    st.init_schema()
    return st


# --- enablement ---


def test_disabled_returns_fresh() -> None:
    s = _settings(BOT_RESUME_SESSION_ON_RESTART=False)
    state = BotState(s)
    storage = _storage_in_tmp()
    pre_id = state.session_id
    pre_started = state.session_started_at_utc
    r = try_resume_session_from_storage(s, state, storage)
    assert r.status == "fresh"
    # Disabled path must not touch state.
    assert state.session_id == pre_id
    assert state.session_started_at_utc == pre_started


def test_no_prior_session_in_storage_returns_no_prior_session() -> None:
    s = _settings()
    state = BotState(s)
    storage = _storage_in_tmp()
    r = try_resume_session_from_storage(s, state, storage)
    assert r.status == "no_prior_session"


# --- happy path ---


def test_resume_within_window_restores_metrics(tmp_path: Path) -> None:
    s = _settings(DATABASE_URL=f"sqlite:///{tmp_path / 'trading.db'}")
    storage = Storage(s)
    storage.init_schema()

    # Synthesize a prior session 1 hour ago.
    prev_session_id = "abcdef1234567890"
    prev_start = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(microsecond=0)
    write_bot_start_event(
        storage,
        session_id=prev_session_id,
        session_started_at_utc=prev_start,
        version="1.0.0-test",
    )
    # Insert a few fills since prev_start.
    for i, side in enumerate(["BUY", "SELL", "BUY"]):
        storage.insert_fill_row(
            {
                "fill_id": f"f{i}",
                "order_id_exchange": f"o{i}",
                "client_order_id": None,
                "ts_fill": (prev_start + timedelta(minutes=i)).isoformat(),
                "symbol": "SUI-PERP",
                "side": side,
                "price": 2.10,
                "size": 10.0,
                "notional": 21.0,
                "fee": 0.0011,
                "liquidity_flag": "resting",
                "mid_at_fill": 2.10,
                "best_bid_at_fill": 2.099,
                "best_ask_at_fill": 2.101,
                "book_snapshot_quality": "full",
                "markout_1s_bps": -0.1,
                "markout_3s_bps": -0.2,
                "markout_5s_bps": -0.3,
                "book_reference_quality": "ok",
            }
        )
    # Equity snapshot for realized + peak.
    storage.insert_equity_snapshot(
        {
            "ts": (prev_start + timedelta(minutes=10)).isoformat(),
            "equity_usd": 422.10,
            "cash_usd": 422.10,
            "realized_pnl_usd": 0.0884,
            "unrealized_pnl_usd": 0.0,
            "fees_usd": 0.0033,
            "drawdown_usd": 0.0,
        }
    )

    state = BotState(s)
    pre_pending_cancels = (
        getattr(state, "_pending_cancels", None) or {}
    )
    r = try_resume_session_from_storage(s, state, storage)

    assert r.status == "resumed", f"reason={r.reason}"
    assert state.session_id == prev_session_id
    # session_started_at_utc was restored (within 1 second of prev_start)
    delta = abs((state.session_started_at_utc - prev_start).total_seconds())
    assert delta < 1.0
    assert state.session_fill_count == 3
    assert state.pnl.realized_pnl_usd == pytest.approx(0.0884)
    # fees aggregated from fills (3 * 0.0011)
    assert state.pnl.fees_usd == pytest.approx(0.0033)
    assert state.pnl.session_peak_equity_usd == pytest.approx(422.10)


# --- expired window ---


def test_expired_session_does_not_restore(tmp_path: Path) -> None:
    s = _settings(DATABASE_URL=f"sqlite:///{tmp_path / 'trading.db'}", BOT_SESSION_CONTINUITY_MAX_HOURS=1.0)
    storage = Storage(s)
    storage.init_schema()
    # Prior session 5 hours ago, max 1 hour.
    prev_start = (datetime.now(timezone.utc) - timedelta(hours=5)).replace(microsecond=0)
    write_bot_start_event(
        storage,
        session_id="oldsession",
        session_started_at_utc=prev_start,
        version="1.0.0-test",
    )
    state = BotState(s)
    pre_id = state.session_id
    r = try_resume_session_from_storage(s, state, storage)
    assert r.status == "expired"
    # state.session_id was NOT replaced.
    assert state.session_id == pre_id


# --- error / corruption fallback ---


def test_storage_read_failure_returns_error(monkeypatch) -> None:
    s = _settings()
    state = BotState(s)
    pre_id = state.session_id

    class BoomStorage:
        def recent_bot_events(self, limit: int = 100):
            raise RuntimeError("disk on fire")

    r = try_resume_session_from_storage(s, state, BoomStorage())
    assert r.status == "error"
    assert "disk on fire" in r.reason
    # State unchanged.
    assert state.session_id == pre_id


def test_corrupted_payload_returns_error(tmp_path: Path) -> None:
    s = _settings(DATABASE_URL=f"sqlite:///{tmp_path / 'trading.db'}")
    storage = Storage(s)
    storage.init_schema()
    # Write a bot_start event with a junk timestamp.
    storage.insert_bot_event(
        "not-a-timestamp",
        "INFO",
        "bot_start",
        "broken",
        {"session_id": "x", "session_started_at_utc": "not-a-timestamp"},
    )
    state = BotState(s)
    pre_id = state.session_id
    r = try_resume_session_from_storage(s, state, storage)
    assert r.status == "error"
    assert state.session_id == pre_id


def test_payload_without_session_id_returns_error(tmp_path: Path) -> None:
    s = _settings(DATABASE_URL=f"sqlite:///{tmp_path / 'trading.db'}")
    storage = Storage(s)
    storage.init_schema()
    # bot_start event missing session_id payload field.
    storage.insert_bot_event(
        datetime.now(timezone.utc).isoformat(),
        "INFO",
        "bot_start",
        "broken",
        {"version": "1.0"},
    )
    state = BotState(s)
    r = try_resume_session_from_storage(s, state, storage)
    assert r.status == "error"


# --- CRITICAL regression guard: behavioral state is NEVER touched ---


def test_apply_session_resume_does_not_touch_behavioral_state() -> None:
    """If this test fails, the resume feature is unsafe to ship.

    Restoring any of the behavioral fields below would re-introduce the
    very deadlock the watchdog was built to clear. Nothing in
    ``apply_session_resume`` should ever set these. If you intentionally
    extend the resume scope, you owe the next operator a paragraph on
    *why it's safe* and a deliberate update of the expected list here.
    """
    s = _settings()
    state = BotState(s)

    # Establish baseline behavioral state with non-default values, so
    # that "restore" would visibly mutate them if the implementation
    # were buggy.
    behavioral_before: dict[str, Any] = {}

    # 1. Quote-eligibility gates / counters.
    behavioral_before["quote_eligibility_snapshot_dict"] = dict(state.quote_eligibility_snapshot_dict)
    behavioral_before["quote_elig_recovery_until_mono"] = state.quote_elig_recovery_until_mono
    behavioral_before["quote_elig_hold_all_count"] = state.quote_elig_hold_all_count
    behavioral_before["adaptive_spread_widen_until_mono"] = getattr(
        state, "adaptive_spread_widen_until_mono", None
    )

    # 2. Order desync / market-data recovery state.
    behavioral_before["order_desync"] = getattr(state, "order_desync", None)
    behavioral_before["desync_phase"] = getattr(state, "desync_phase", None)
    behavioral_before["market_data_failed_refresh_streak"] = getattr(
        state, "market_data_failed_refresh_streak", None
    )

    # 3. WS connection / private-WS health.
    behavioral_before["private_ws_connected"] = state.private_ws_connected
    behavioral_before["private_ws_healthy"] = state.private_ws_healthy
    behavioral_before["private_ws_reconnect_count"] = state.private_ws_reconnect_count

    # 4. Position cache (must always reconcile from exchange).
    behavioral_before["position"] = state.position
    behavioral_before["account"] = state.account

    # 5. Working orders.
    behavioral_before["working_bid"] = state.working_bid
    behavioral_before["working_ask"] = state.working_ask

    # 6. Watchdog signals / timing.
    behavioral_before["last_quote_engine_non_hold_ts_mono"] = state.last_quote_engine_non_hold_ts_mono
    behavioral_before["last_place_attempt_ts_mono"] = state.last_place_attempt_ts_mono

    # 7. Toxicity rolling window.
    behavioral_before["operator_rolling_toxicity_markout_bps"] = (
        state.operator_rolling_toxicity_markout_bps
    )

    # 8. Kill / pause flags.
    behavioral_before["killed"] = state.killed
    behavioral_before["manual_pause"] = state.manual_pause
    behavioral_before["flatten_mode"] = state.flatten_mode

    # 9. Fills dedupe set (must keep new fills from being silently
    #    eaten as duplicates). Resume must NOT pre-populate this; the
    #    new process re-ingests via private WS / REST replay.
    behavioral_before["_seen_fill_ids"] = set(state._seen_fill_ids)

    # Now do the resume with realistic metric values.
    state.apply_session_resume(
        session_id="restored-session-id",
        session_started_at_utc=datetime(2026, 4, 25, 1, 0, 0, tzinfo=timezone.utc),
        session_fill_count=12,
        realized_pnl_usd=0.0884,
        fees_usd=0.0033,
        peak_equity_usd=422.10,
    )

    # METRIC fields DID change.
    assert state.session_id == "restored-session-id"
    assert state.session_fill_count == 12
    assert state.pnl.realized_pnl_usd == pytest.approx(0.0884)
    assert state.pnl.fees_usd == pytest.approx(0.0033)
    assert state.pnl.session_peak_equity_usd == pytest.approx(422.10)

    # BEHAVIORAL fields did NOT change.
    assert state.quote_eligibility_snapshot_dict == behavioral_before["quote_eligibility_snapshot_dict"]
    assert state.quote_elig_recovery_until_mono == behavioral_before["quote_elig_recovery_until_mono"]
    assert state.quote_elig_hold_all_count == behavioral_before["quote_elig_hold_all_count"]
    assert getattr(state, "adaptive_spread_widen_until_mono", None) == behavioral_before["adaptive_spread_widen_until_mono"]

    assert getattr(state, "order_desync", None) == behavioral_before["order_desync"]
    assert getattr(state, "desync_phase", None) == behavioral_before["desync_phase"]
    assert getattr(state, "market_data_failed_refresh_streak", None) == behavioral_before["market_data_failed_refresh_streak"]

    assert state.private_ws_connected == behavioral_before["private_ws_connected"]
    assert state.private_ws_healthy == behavioral_before["private_ws_healthy"]
    assert state.private_ws_reconnect_count == behavioral_before["private_ws_reconnect_count"]

    assert state.position == behavioral_before["position"]
    assert state.account == behavioral_before["account"]
    assert state.working_bid == behavioral_before["working_bid"]
    assert state.working_ask == behavioral_before["working_ask"]

    assert state.last_quote_engine_non_hold_ts_mono == behavioral_before["last_quote_engine_non_hold_ts_mono"]
    assert state.last_place_attempt_ts_mono == behavioral_before["last_place_attempt_ts_mono"]

    assert state.operator_rolling_toxicity_markout_bps == behavioral_before["operator_rolling_toxicity_markout_bps"]

    assert state.killed == behavioral_before["killed"]
    assert state.manual_pause == behavioral_before["manual_pause"]
    assert state.flatten_mode == behavioral_before["flatten_mode"]

    # Fill dedupe set unchanged.
    assert state._seen_fill_ids == behavioral_before["_seen_fill_ids"]


# --- helper ---


_TMP_DBS: list[Storage] = []


def _storage_in_tmp() -> Storage:
    """Storage in OS tempdir (per repo conftest rule: don't touch repo tmp/)."""
    d = tempfile.mkdtemp(prefix="dtc_test_")
    s = _settings(DATABASE_URL=f"sqlite:///{Path(d) / 'trading.db'}")
    st = Storage(s)
    st.init_schema()
    _TMP_DBS.append(st)
    return st


# --- telegram_summary ---


def test_telegram_summary_resumed() -> None:
    r = SessionResumeResult(
        status="resumed",
        reason="ok",
        session_id="abcd1234",
        fill_count=12,
        realized_pnl_usd=0.0884,
        fees_usd=0.0033,
        age_hours=1.5,
    )
    text = r.telegram_summary()
    assert "abcd1234" in text
    assert "12 fills" in text
    assert "1.5h" in text


def test_telegram_summary_expired() -> None:
    r = SessionResumeResult(status="expired", reason="age=37.4h > max=24.0h")
    assert "expired" in r.telegram_summary()


def test_telegram_summary_no_prior() -> None:
    r = SessionResumeResult(status="no_prior_session")
    assert "no prior session" in r.telegram_summary()


def test_telegram_summary_fresh() -> None:
    r = SessionResumeResult(status="fresh")
    assert "disabled" in r.telegram_summary()


def test_telegram_summary_error() -> None:
    r = SessionResumeResult(status="error", reason="disk full")
    assert "disk full" in r.telegram_summary()
