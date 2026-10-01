"""Per-tick eligibility log must carry enough fields to diagnose HOLD_ALL.

In `logs.1776440173816.log` the bot went silent for 52 s after a normal
cancel cycle. The existing ``quote_eligibility_transition`` log only said
"from=QUOTE_BOTH to=HOLD_ALL" with no attribution — so we couldn't tell
whether gap_p95, public-BBO staleness, kinematic drift/jump, or the
``order_state_uncertainty`` gate was doing the clamping. These tests lock
down the diagnostic field set on the transition log, the re-arm log, and
the new periodic tick sampler.

Intentionally not testing specific values — only presence — so the guard
survives future threshold tuning.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

from app.bot import Bot, _QUOTE_ELIGIBILITY_TICK_SAMPLE_EVERY_N
from app.enums import BotStatus
from app.models import BestBidAsk
from app.state import BotState
from app.storage import Storage
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


_EXPECTED_DIAG_FIELDS = {
    "raw_eligibility",
    "raw_reason",
    "counter_tags",
    "in_cooldown",
    "recovery_floor",
    "recovery_remaining_ms",
    "seconds_since_public_book_update",
    "effective_staleness_ms",
    "gap_p95_ms",
    "gap_median_ms",
    "mid_return_100ms_bps",
    "mid_return_250ms_bps",
    "mid_return_500ms_bps",
    "jump_100ms_bps",
    "jump_250ms_bps",
    "jump_500ms_bps",
    "order_state_uncertainty",
}


def _settings_db() -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_elig_log_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "QUOTE_ELIGIBILITY_ENABLED": True,
        }
    )
    return s, path


def _bootstrap_bot(settings: UnitTestSettings) -> tuple[Bot, Storage, BotState]:
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.bot_status = BotStatus.RUNNING
    state.market = BestBidAsk(
        symbol=settings.symbol,
        best_bid=99.0,
        best_ask=101.0,
        mid_price=100.0,
        spread_bps=200.0,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = {
        "status": "ok",
        "response": {
            "type": "order",
            "data": {"statuses": [{"resting": {"oid": 1}}]},
        },
    }
    bot = Bot(settings, state, client, storage)
    return bot, storage, state


def _run_one_tick(bot: Bot) -> None:
    with patch("app.bot.refresh_account_only"), patch.object(
        bot._exec, "maybe_sync_open_orders"
    ):
        bot.one_tick()


def _extra_of(record: logging.LogRecord) -> dict:
    """Our ``log_extra`` helper stores the structured payload on ``extra_data``."""
    return getattr(record, "extra_data", {}) or {}


def test_quote_eligibility_transition_log_carries_full_diag(caplog) -> None:
    settings, path = _settings_db()
    try:
        bot, _storage, _state = _bootstrap_bot(settings)
        with caplog.at_level(logging.INFO, logger="app.bot"):
            # First tick: eligibility transitions from default to some value,
            # which emits the transition log once.
            _run_one_tick(bot)
        recs = [r for r in caplog.records if r.getMessage() == "quote_eligibility_transition"]
        assert recs, "expected quote_eligibility_transition log on first healthy tick"
        extra = _extra_of(recs[0])
        missing = _EXPECTED_DIAG_FIELDS - set(extra.keys())
        assert not missing, (
            f"quote_eligibility_transition missing diagnostic fields: {sorted(missing)}"
        )
        # The existing from/to/reason fields must still be present.
        for legacy in ("from", "to", "reason"):
            assert legacy in extra, f"missing legacy field {legacy!r}"
    finally:
        path.unlink(missing_ok=True)


def test_quote_eligibility_transition_reports_order_state_uncertainty_when_desync(
    caplog,
) -> None:
    settings, path = _settings_db()
    try:
        bot, _storage, state = _bootstrap_bot(settings)
        state.order_desync = True  # forces has_order_state_uncertainty=True
        with caplog.at_level(logging.INFO, logger="app.bot"):
            _run_one_tick(bot)
        recs = [r for r in caplog.records if r.getMessage() == "quote_eligibility_transition"]
        assert recs
        extra = _extra_of(recs[0])
        assert extra.get("order_state_uncertainty") is True
        assert extra.get("raw_eligibility") == "HOLD_ALL"
        tags = extra.get("counter_tags")
        assert isinstance(tags, list) and "hold_order_uncertainty" in tags
    finally:
        path.unlink(missing_ok=True)


def test_quote_eligibility_recovery_armed_log_carries_counter_tags(caplog) -> None:
    from app.enums import QuoteEligibility

    path = Path(tempfile.gettempdir()) / f"mm_elig_arm_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    # Disable the eligibility engine so raw is guaranteed QUOTE_BOTH on every
    # tick; the recovery-arm logic is orthogonal to the rule machinery — it
    # fires purely on (last_effective, new_raw) rank comparison. This test
    # isolates the arming log without depending on market-data kinematics.
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "QUOTE_ELIGIBILITY_ENABLED": False,
        }
    )
    try:
        bot, _storage, state = _bootstrap_bot(settings)
        # Prime last_effective to HOLD_ALL so next tick's QUOTE_BOTH raw counts
        # as an improvement and arms the recovery cooldown.
        state.quote_eligibility_last_effective = QuoteEligibility.HOLD_ALL
        with caplog.at_level(logging.INFO, logger="app.bot"):
            _run_one_tick(bot)
        recs = [r for r in caplog.records if r.getMessage() == "quote_eligibility_recovery_armed"]
        assert recs, "expected recovery_armed log when raw eligibility improves"
        extra = _extra_of(recs[0])
        assert "counter_tags" in extra
        assert isinstance(extra["counter_tags"], list)
        # Legacy fields still present.
        for legacy in ("raw", "last_effective", "recovery_floor", "recovery_remaining_ms"):
            assert legacy in extra, f"missing legacy field {legacy!r}"
    finally:
        path.unlink(missing_ok=True)


def test_quote_eligibility_tick_sample_fires_on_cadence(caplog) -> None:
    settings, path = _settings_db()
    try:
        bot, _storage, _state = _bootstrap_bot(settings)
        with caplog.at_level(logging.INFO, logger="app.bot"):
            # Drive enough ticks that the mod-N sampler fires at least once.
            for _ in range(_QUOTE_ELIGIBILITY_TICK_SAMPLE_EVERY_N):
                _run_one_tick(bot)
        samples = [
            r for r in caplog.records if r.getMessage() == "quote_eligibility_tick_sample"
        ]
        assert samples, "expected at least one quote_eligibility_tick_sample on cadence"
        extra = _extra_of(samples[0])
        missing = _EXPECTED_DIAG_FIELDS - set(extra.keys())
        assert not missing, f"tick_sample missing fields: {sorted(missing)}"
        # Must also carry the effective cap and tick number for correlation.
        assert "effective" in extra
        assert "tick" in extra and int(extra["tick"]) > 0
    finally:
        path.unlink(missing_ok=True)
