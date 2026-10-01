"""Execution–quote boundary: runtime refresh must not reshape QuoteEngine output."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.enums import RiskAction, Side
from app.execution import OrderManager
from app.models import BestBidAsk
from app.quote_engine import FinalQuoteOrder, QuoteBuildResult
from app.state import BotState
from app.storage import Storage
from app.utils.time import utc_now
from tests.exchange_client_mocks import mock_mm_client
from tests.settings_helpers import UnitTestSettings
from tests.test_quote_reprice_maintenance import _decision, _fresh_market, _ok_place


def _db(max_pos: float = 10.0) -> tuple[UnitTestSettings, Path]:
    path = Path(tempfile.gettempdir()) / f"mm_bnd_{os.getpid()}_{uuid.uuid4().hex}.db"
    path.unlink(missing_ok=True)
    s = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": f"sqlite:///{path.as_posix()}",
            "MAX_ABS_POSITION": max_pos,
            # ETH-priced mock at $3000; central pre-send gate (Codex
            # MED #1, 2026-05-06) checks USD position cap. Raise it.
            "MAX_POSITION_NOTIONAL_USD": 100_000.0,
            "REPRICE_THRESHOLD_BPS": 20.0,
        }
    )
    return s, path


def test_maybe_refresh_submits_exact_quote_engine_prices() -> None:
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.12, size=0.011),
            ask_order=FinalQuoteOrder(side=Side.SELL, price=3002.34, size=0.019),
            mode="two_sided",
            telemetry={"quote_engine_mode": "two_sided"},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()

    assert client.place_post_only_limit.call_count == 2
    c0 = client.place_post_only_limit.call_args_list[0][0]
    c1 = client.place_post_only_limit.call_args_list[1][0]
    assert c0[1] is True  # buy
    assert c0[2] == 0.011
    assert c0[3] == 3000.12
    assert c1[1] is False  # sell
    assert c1[2] == 0.019
    assert c1[3] == 3002.34
    path.unlink(missing_ok=True)


def test_maybe_refresh_quotes_does_not_call_normalize_order_pair() -> None:
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=FinalQuoteOrder(side=Side.SELL, price=3001.0, size=0.01),
            mode="two_sided",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    with patch("app.execution.normalize_order_pair") as m_norm:
        m_norm.side_effect = AssertionError("normalize_order_pair must not run on quote-engine path")
        om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    m_norm.assert_not_called()
    path.unlink(missing_ok=True)


def test_maybe_refresh_placement_order_is_always_bid_then_ask() -> None:
    """Inventory does not change submit ordering (fixed BUY then SELL)."""
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    state.position.position_qty = 5.0
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=2999.0, size=0.02),
            ask_order=FinalQuoteOrder(side=Side.SELL, price=3002.0, size=0.02),
            mode="two_sided",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()
    assert client.place_post_only_limit.call_count == 2
    assert client.place_post_only_limit.call_args_list[0][0][1] is True
    assert client.place_post_only_limit.call_args_list[1][0][1] is False
    path.unlink(missing_ok=True)


def test_maybe_refresh_attempts_second_side_even_if_first_submit_fails() -> None:
    """No execution-side 'skip deprioritized side' when engine returned two-sided intent."""
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.side_effect = [
        Exception("wire fail bid"),
        _ok_place(2),
    ]
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=FinalQuoteOrder(side=Side.SELL, price=3001.0, size=0.01),
            mode="two_sided",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    om.wait_transport_idle()
    assert client.place_post_only_limit.call_count == 2
    path.unlink(missing_ok=True)


def test_maybe_refresh_quotes_routes_new_rests_through_stage_and_enqueue() -> None:
    """Runtime path stages SENT locally then enqueues transport (not sync submit_verbatim)."""
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=FinalQuoteOrder(side=Side.SELL, price=3001.0, size=0.01),
            mode="two_sided",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    stage = MagicMock(wraps=om._stage_place_order_local)
    enq = MagicMock(wraps=om._enqueue_place_transport)
    with patch.object(om, "_stage_place_order_local", stage), patch.object(
        om, "_enqueue_place_transport", enq
    ):
        om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    assert stage.call_count == 2
    assert enq.call_count == 2
    assert stage.call_args_list[0].kwargs["price"] == 3000.0
    assert stage.call_args_list[1].kwargs["price"] == 3001.0
    path.unlink(missing_ok=True)


def test_maybe_refresh_quotes_does_not_call_place_passive_order_manual_only() -> None:
    s, path = _db()
    storage = Storage(s)
    storage.init_schema()
    state = BotState(s)
    state.market = _fresh_market(s)
    client = mock_mm_client()
    client.has_write_access.return_value = True
    client.place_post_only_limit.return_value = _ok_place(1)
    om = OrderManager(s, client, storage, state, private_event_queue=None)

    def fake_build(_ctx):
        return QuoteBuildResult(
            bid_order=FinalQuoteOrder(side=Side.BUY, price=3000.0, size=0.01),
            ask_order=None,
            mode="one_sided",
            telemetry={},
        )

    om._quote_engine.build_quotes = fake_build  # type: ignore[method-assign]
    with patch.object(om, "place_passive_order_manual_only") as m_man:
        om.maybe_refresh_quotes(_decision(), RiskAction.ALLOW, 1.0, 1.0, 0.0)
    m_man.assert_not_called()
    path.unlink(missing_ok=True)
