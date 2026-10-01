"""Live-safety behaviors: sizing, sync desync, cancel path without mid, flatten/API."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.bot import Bot
from app.enums import BotStatus, DesyncPhase, OrderStatus, RiskAction, Side
from app.execution import OrderManager
from app.exchange.hyperliquid_types import HLOpenOrderRaw
from app.models import BestBidAsk, RiskDecision, WorkingOrder
from app.state import BotState
from app.storage import Storage
from app.exchange.symbol_spec import FALLBACK_SYMBOL_SPEC
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings


def _settings_db() -> tuple[UnitTestSettings, Path]:
    # v1.4.90 CI fix: every call gets a unique DB filename (uuid suffix)
    # so we never re-unlink a file that a previous test's SQLite
    # connection might still be holding open. Pre-v1.4.90 this used a
    # single shared filename `mm_exec_test_{pid}.db` and called
    # `path.unlink(missing_ok=True)` at the top of each call — which
    # races against Windows file-handle release (PermissionError
    # WinError 32 observed in CI 2026-05-19). Each test now creates
    # its own pristine file; no pre-unlink needed.
    path = Path(tempfile.gettempdir()) / f"mm_exec_test_{os.getpid()}_{uuid.uuid4().hex}.db"
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 0.05,
        }
    )
    return s, path


def test_clip_entry_sizes_long_inventory() -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.position.position_qty = 0.04
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state)
    b, a = om._clip_entry_sizes(0.04, 0.02, 0.02)
    assert abs(b - 0.01) < 1e-9
    assert abs(a - 0.02) < 1e-9
    path.unlink(missing_ok=True)


def test_sync_duplicate_open_buy_sets_desync() -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(1, settings.symbol, Side.BUY, 100.0, 0.1, 0),
        HLOpenOrderRaw(2, settings.symbol, Side.BUY, 100.0, 0.1, 0),
    ]
    om = OrderManager(settings, client, storage, state)
    om.sync_open_orders(force=True, emergency=True)
    assert state.order_desync is True
    assert state.desync_phase == DesyncPhase.DETECTED
    path.unlink(missing_ok=True)


def test_sync_duplicate_open_orders_logs_diagnostics_payload_once(caplog) -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.working_bid = WorkingOrder(
        order_id_local="L-bid",
        order_id_exchange=11,
        client_order_id="0x" + "ab" * 16,
        symbol=settings.symbol,
        side=Side.BUY,
        price=3000.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.SENT,
        quote_cycle_id="q-dup",
        transport_intent_seq=7,
    )
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(1, settings.symbol, Side.BUY, 3000.0, 0.01, 0),
        HLOpenOrderRaw(2, settings.symbol, Side.BUY, 3001.0, 0.01, 0),
    ]
    om = OrderManager(settings, client, storage, state)
    with caplog.at_level(logging.WARNING, logger="app.execution"):
        om.sync_open_orders(force=True, emergency=True)
    recs = [
        r for r in caplog.records if r.getMessage() == "multiple_open_orders_same_side"
    ]
    assert len(recs) == 1
    extra = getattr(recs[0], "extra_data", {})
    assert extra.get("dup_buy") is True
    assert extra.get("local_working_bid", {}).get("transport_intent_seq") == 7
    assert isinstance(extra.get("exchange_open_orders"), list) and extra["exchange_open_orders"]
    path.unlink(missing_ok=True)


def test_one_tick_no_mid_cancels_resting_when_running() -> None:
    """BUG-009: when mid is None we have no market data to evaluate
    against, so risk returns NO_QUOTE/no_market_data. Resting orders
    must be cancelled — leaving them up at potentially-stale prices is
    pick-off risk. Pre-fix the assertion was ``cancel_m.assert_not_called()``;
    that codified the bug.
    """
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.bot_status = BotStatus.RUNNING
    state.market = BestBidAsk(
        symbol=settings.symbol,
        best_bid=100.0,
        best_ask=101.0,
        mid_price=None,
        spread_bps=None,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)
    with patch("app.bot.refresh_account_only"), patch.object(
        bot._exec, "maybe_sync_open_orders"
    ) as sync_m, patch.object(bot._exec, "cancel_resting_for_risk") as cancel_m, patch.object(
        bot, "_persist_snapshots"
    ):
        bot.one_tick()
        sync_m.assert_called_once()
        cancel_m.assert_called_once()
    path.unlink(missing_ok=True)


def test_same_side_place_suppressed_when_cancel_unresolved_even_if_slot_clears() -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state)
    wo = WorkingOrder(
        order_id_local="L-bid-cp",
        order_id_exchange=1234,
        client_order_id="0x" + "ab" * 16,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.ACKED,
        quote_cycle_id="qcp",
    )
    state.working_bid = wo
    assert om._enqueue_cancel_quote_path(wo) is True
    state.working_bid = None
    blocked = om._stage_place_order_local(
        Side.BUY, price=99.0, size=0.01, quote_cycle_id="q-next"
    )
    assert blocked is None
    obs = om.get_reconcile_runtime_counters()
    assert obs["suppress_place_due_unresolved_count"] >= 1
    assert "BUY" in obs["active_unresolved_sides"]
    path.unlink(missing_ok=True)


def test_duplicate_open_detection_enters_side_unresolved_mode() -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = [
        HLOpenOrderRaw(1, settings.symbol, Side.BUY, 100.0, 0.1, 0),
        HLOpenOrderRaw(2, settings.symbol, Side.BUY, 101.0, 0.1, 0),
    ]
    om = OrderManager(settings, client, storage, state)
    om.sync_open_orders(force=True, emergency=True)
    obs = om.get_reconcile_runtime_counters()
    assert obs["duplicate_open_detect_count"] >= 1
    assert "BUY" in obs["active_unresolved_sides"]
    assert state.order_desync is True
    path.unlink(missing_ok=True)


def test_cancel_pending_timeout_requests_desync_recovery_reconcile() -> None:
    settings, path = _settings_db()
    settings.cancel_pending_unresolved_timeout_seconds = 0.01
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    om = OrderManager(settings, client, storage, state)
    wo = WorkingOrder(
        order_id_local="L-timeout",
        order_id_exchange=555,
        client_order_id="0x" + "cd" * 16,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.01,
        post_only=True,
        status=OrderStatus.CANCEL_PENDING,
        quote_cycle_id="qt",
    )
    state.working_bid = wo
    om._cancel_pending_since_mono[Side.BUY] = time.monotonic() - 1.0
    with patch.object(om, "request_open_orders_reconcile") as req_m:
        om._maybe_handle_cancel_pending_timeout(Side.BUY, wo)
    req_m.assert_called_once()
    _, kwargs = req_m.call_args
    assert kwargs["force"] is True and kwargs["emergency"] is True
    obs = om.get_reconcile_runtime_counters()
    assert obs["cancel_pending_timeout_count"] >= 1
    path.unlink(missing_ok=True)


def test_side_unresolved_suppression_emits_observability_payload(caplog) -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    om = OrderManager(settings, client, storage, state)
    om._set_side_unresolved(Side.BUY, reason="test_guard")
    with caplog.at_level(logging.INFO, logger="app.execution"):
        _ = om._stage_place_order_local(Side.BUY, price=100.0, size=0.01, quote_cycle_id="q-s")
    recs = [r for r in caplog.records if r.getMessage() == "same_side_place_suppressed"]
    assert recs
    extra = getattr(recs[0], "extra_data", {})
    assert extra.get("side") == "BUY"
    assert extra.get("reason") == "side_unresolved"
    path.unlink(missing_ok=True)


def test_account_refresh_suppressed_when_only_order_uncertainty_active() -> None:
    settings, path = _settings_db()
    settings.account_rest_min_interval_seconds = 1.0
    settings.order_state_uncertainty_account_rest_interval_seconds = 30.0
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.bot_status = BotStatus.RUNNING
    state.account_rest_last_monotonic = time.monotonic()
    state.exchange_snapshot_unhealthy_streak = 0
    state.private_ws_recovery_pending = False
    client = mock_mm_client()
    bot = Bot(settings, state, client, storage)
    with patch.object(bot._exec, "has_order_state_uncertainty", return_value=True), patch.object(
        bot._exec, "should_ingest_fills_via_rest", return_value=False
    ):
        bot._exec._private_ws_healthy = True  # noqa: SLF001 - test only
        should, reason = bot._should_refresh_account_rest(settings.hl_account_address)
    assert should is False
    assert reason == "suppressed_order_uncertainty"
    path.unlink(missing_ok=True)


def test_stale_warn_no_quote_cancels_resting_orders() -> None:
    """BUG-009: when market data goes stale (warn-level), the bot must
    cancel resting orders, not keep them advertising stale prices.
    Previously this asserted ``cancel_on_no_quote is False`` — that test
    codified the bug. Post-fix the same `stale_data_warn` reason MUST
    flip the cancel_on_no_quote flag to True.
    """
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.bot_status = BotStatus.RUNNING
    state.market = BestBidAsk(
        symbol=settings.symbol,
        best_bid=100.0,
        best_ask=101.0,
        mid_price=100.5,
        spread_bps=100.0,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)
    fake_risk = RiskDecision(action=RiskAction.NO_QUOTE, reasons=["stale_data_warn"])
    with patch("app.bot.refresh_account_only"), patch(
        "app.bot.evaluate_risk", return_value=fake_risk
    ), patch.object(bot._exec, "maybe_refresh_quotes") as refresh_m, patch.object(
        bot._exec, "maybe_sync_open_orders"
    ), patch.object(bot, "_persist_snapshots"):
        bot.one_tick()
    assert refresh_m.call_count == 1
    assert refresh_m.call_args.kwargs.get("cancel_on_no_quote") is True
    path.unlink(missing_ok=True)


def test_public_ws_stale_warn_cancels_resting_orders() -> None:
    """BUG-009 second variant: public-WS WARN must also cancel resting."""
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.bot_status = BotStatus.RUNNING
    state.market = BestBidAsk(
        symbol=settings.symbol,
        best_bid=100.0,
        best_ask=101.0,
        mid_price=100.5,
        spread_bps=100.0,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)
    fake_risk = RiskDecision(
        action=RiskAction.NO_QUOTE, reasons=["public_ws_stale_warn"]
    )
    with patch("app.bot.refresh_account_only"), patch(
        "app.bot.evaluate_risk", return_value=fake_risk
    ), patch.object(bot._exec, "maybe_refresh_quotes") as refresh_m, patch.object(
        bot._exec, "maybe_sync_open_orders"
    ), patch.object(bot, "_persist_snapshots"):
        bot.one_tick()
    assert refresh_m.call_args.kwargs.get("cancel_on_no_quote") is True
    path.unlink(missing_ok=True)


def test_no_quote_benign_reasons_keep_resting_orders() -> None:
    """Counter-test for BUG-009: NO_QUOTE on benign reasons (e.g. trade
    rate limiter) MUST NOT cancel resting orders — that's the intended
    'don't add new quotes but keep existing ones' semantics.

    2026-05-12 codex-#1 update: this test also implicitly verifies
    that the new ``CANCEL_RESTING_ON_HOLD_ALL`` knob does NOT trip
    when the only signal is a benign NO_QUOTE risk reason. We
    disable the new knob explicitly so the test stays focused on
    the NO_QUOTE risk path (the eligibility path is covered by its
    own tests in test_codex_review_20260511_bugs.py).
    """
    settings, path = _settings_db()
    # Disable the codex-#1 HOLD_ALL-cancel knob to isolate the
    # NO_QUOTE-benign-keep-resting variable. The new knob's behavior
    # is tested independently.
    settings = settings.model_copy(update={"cancel_resting_on_hold_all": False})
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.bot_status = BotStatus.RUNNING
    state.market = BestBidAsk(
        symbol=settings.symbol,
        best_bid=100.0,
        best_ask=101.0,
        mid_price=100.5,
        spread_bps=100.0,
    )
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)
    fake_risk = RiskDecision(action=RiskAction.NO_QUOTE, reasons=["trade_rate_limit"])
    with patch("app.bot.refresh_account_only"), patch(
        "app.bot.evaluate_risk", return_value=fake_risk
    ), patch.object(bot._exec, "maybe_refresh_quotes") as refresh_m, patch.object(
        bot._exec, "maybe_sync_open_orders"
    ), patch.object(bot, "_persist_snapshots"):
        bot.one_tick()
    assert refresh_m.call_args.kwargs.get("cancel_on_no_quote") is False
    path.unlink(missing_ok=True)


def test_one_tick_persists_quote_decision_when_trading_disabled() -> None:
    """Dry-run: quote is computed and stored for /quotes/recent; no sync/place."""
    path = Path(tempfile.gettempdir()) / f"mm_exec_dry_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 1.0,
            "QUOTE_ELIGIBILITY_ENABLED": False,
        }
    )
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
    bot = Bot(settings, state, client, storage)
    with patch("app.bot.refresh_account_only"), patch.object(
        bot._exec, "maybe_sync_open_orders"
    ) as sync_m, patch.object(bot._exec, "maybe_refresh_quotes") as maybe_q, patch.object(
        bot, "_persist_snapshots"
    ):
        bot.one_tick()
    sync_m.assert_not_called()
    maybe_q.assert_not_called()
    rows = storage.recent_quote_decisions(5)
    assert len(rows) == 1
    r = rows[0]
    assert float(r["mid_price"]) == 100.0
    assert r["exec_price_tick"] == float(FALLBACK_SYMBOL_SPEC.price_tick)
    assert r["exec_size_step"] == float(FALLBACK_SYMBOL_SPEC.size_step)
    assert r["exec_meta_decimal_grid_price_tick"] == r["exec_price_tick"]
    assert r["exec_meta_decimal_size_step"] == r["exec_size_step"]
    assert r["exec_hl_max_sig_figs_nonint"] == 5
    assert r["exec_price_normalize_pipeline"] == "hl_perp_decimal_grid_plus_max_sig_figs_nonint"
    assert r["exec_raw_bid_px"] is not None and r["exec_raw_ask_px"] is not None
    assert r["exec_norm_bid_px"] is not None and r["exec_norm_ask_px"] is not None
    assert r["exec_wire_bid_limit_p"] is not None and r["exec_wire_ask_limit_p"] is not None
    path.unlink(missing_ok=True)


def test_one_tick_persists_quote_when_trading_disabled_with_multiple_reasons() -> None:
    """Dry-run persistence if trading_disabled is among reasons (not reasons == exact list)."""
    path = Path(tempfile.gettempdir()) / f"mm_exec_dry2_{os.getpid()}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 1.0,
            "QUOTE_ELIGIBILITY_ENABLED": False,
        }
    )
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
    bot = Bot(settings, state, client, storage)
    fake_risk = RiskDecision(
        action=RiskAction.NO_QUOTE,
        reasons=["trading_disabled", "hypothetical_extra"],
    )
    with patch("app.bot.refresh_account_only"), patch(
        "app.bot.evaluate_risk", return_value=fake_risk
    ), patch.object(bot._exec, "maybe_sync_open_orders") as sync_m, patch.object(
        bot._exec, "maybe_refresh_quotes"
    ) as maybe_q, patch.object(bot, "_persist_snapshots"):
        bot.one_tick()
    sync_m.assert_not_called()
    maybe_q.assert_not_called()
    rows = storage.recent_quote_decisions(5)
    assert len(rows) == 1
    q = rows[0]
    assert q["exec_price_tick"] == float(FALLBACK_SYMBOL_SPEC.price_tick)
    assert q["exec_meta_decimal_grid_price_tick"] == q["exec_price_tick"]
    assert q["exec_price_normalize_pipeline"] == "hl_perp_decimal_grid_plus_max_sig_figs_nonint"
    assert q["exec_raw_bid_px"] is not None
    assert q["exec_norm_bid_px"] is not None
    assert q["exec_wire_bid_limit_p"] is not None
    path.unlink(missing_ok=True)


def test_quote_persist_always_includes_raw_and_spec_when_normalize_fails() -> None:
    """Tight max position clips size below HL min notional: raw + tick/step still stored."""
    path = Path(tempfile.gettempdir()) / f"mm_exec_submin_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": False,
            "HL_SECRET_KEY": "",
            "HL_ACCOUNT_ADDRESS": "",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 0.05,
            "QUOTE_ELIGIBILITY_ENABLED": False,
        }
    )
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
    bot = Bot(settings, state, client, storage)
    with patch("app.bot.refresh_account_only"), patch.object(
        bot._exec, "maybe_sync_open_orders"
    ), patch.object(bot._exec, "maybe_refresh_quotes"), patch.object(bot, "_persist_snapshots"):
        bot.one_tick()
    r = storage.recent_quote_decisions(1)[0]
    assert r["exec_price_tick"] == float(FALLBACK_SYMBOL_SPEC.price_tick)
    assert r["exec_size_step"] == float(FALLBACK_SYMBOL_SPEC.size_step)
    assert r["exec_meta_decimal_grid_price_tick"] == r["exec_price_tick"]
    assert r["exec_hl_max_sig_figs_nonint"] == 5
    assert r["exec_raw_bid_px"] is not None
    assert r["exec_norm_bid_px"] is None
    assert r["exec_wire_bid_limit_p"] is None
    path.unlink(missing_ok=True)


def test_one_tick_trading_enabled_allow_persists_same_exec_fields_as_dry_run() -> None:
    """Deferred quote row after maybe_refresh_quotes includes raw/norm/tick like dry-run."""
    path = Path(tempfile.gettempdir()) / f"mm_exec_liveq_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": 1.0,
            "QUOTE_ELIGIBILITY_ENABLED": False,
        }
    )
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
    with patch("app.bot.refresh_account_only"), patch.object(
        bot._exec, "maybe_sync_open_orders"
    ):
        bot.one_tick()
    rows = storage.recent_quote_decisions(5)
    assert len(rows) == 1
    r = rows[0]
    assert r["exec_price_tick"] == float(FALLBACK_SYMBOL_SPEC.price_tick)
    assert r["exec_size_step"] == float(FALLBACK_SYMBOL_SPEC.size_step)
    assert r["exec_meta_decimal_grid_price_tick"] == r["exec_price_tick"]
    assert r["exec_wire_bid_limit_p"] is not None
    assert r["exec_raw_bid_px"] is not None and r["exec_raw_ask_px"] is not None
    assert r["exec_norm_bid_px"] is not None and r["exec_norm_ask_px"] is not None
    path.unlink(missing_ok=True)


def test_flatten_restores_killed_status() -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    state.killed = True
    state.bot_status = BotStatus.KILLED
    client = mock_mm_client()
    client.has_write_access.return_value = True
    bot = Bot(settings, state, client, storage)
    with patch("app.bot.refresh_account_only"):
        state.position.position_qty = 0.0
        bot.flatten(blocking=True)
    assert state.bot_status == BotStatus.KILLED
    path.unlink(missing_ok=True)


def test_reconcile_clears_cancel_pending_when_order_gone() -> None:
    settings, path = _settings_db()
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.fetch_open_orders_raw.return_value = []
    wo = WorkingOrder(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=42,
        client_order_id=None,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.05,
        post_only=True,
        status=OrderStatus.CANCEL_PENDING,
        quote_cycle_id="c1",
    )
    state.working_bid = wo
    om = OrderManager(settings, client, storage, state)
    om.sync_open_orders(force=True, emergency=True)
    assert state.working_bid is None
    assert wo.status == OrderStatus.CANCELED
    path.unlink(missing_ok=True)


def test_cancel_failed_keeps_working_order_for_replace() -> None:
    settings, path = _settings_db()
    settings.max_abs_position = 1.0
    storage = Storage(settings)
    storage.init_schema()
    state = BotState(settings)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = {
        "status": "ok",
        "response": {"data": {"statuses": [{"resting": {"oid": 99}}]}},
    }
    wo = WorkingOrder(
        order_id_local=str(uuid.uuid4()),
        order_id_exchange=42,
        client_order_id=None,
        symbol=settings.symbol,
        side=Side.BUY,
        price=100.0,
        size=0.05,
        post_only=True,
        status=OrderStatus.ACKED,
        quote_cycle_id="c1",
    )
    state.working_bid = wo
    om = OrderManager(settings, client, storage, state)

    from app.models import QuoteDecision
    from app.enums import ActiveSides

    decision = QuoteDecision(
        ts=wo.ts_created,
        symbol=settings.symbol,
        mid_price=100.0,
        vol_estimate=1.0,
        inventory=0.0,
        reservation_price=100.0,
        target_spread_bps=5.0,
        target_bid=99.9,
        target_ask=100.1,
        quoted_bid=99.0,
        quoted_ask=101.0,
        quoted_bid_sz=0.2,
        quoted_ask_sz=0.05,
        active_sides=ActiveSides.BID_ONLY,
        toxicity_score=0.0,
        decision_reason="test",
        quote_cycle_id="c2",
    )
    client.cancel_order.side_effect = RuntimeError("network")
    om.maybe_refresh_quotes(decision, RiskAction.ALLOW, 1.0, 1.0, 0.0)
    assert state.working_bid is wo
    assert wo.status == OrderStatus.CANCEL_PENDING
    assert client.place_post_only_limit.call_count == 0
    path.unlink(missing_ok=True)
