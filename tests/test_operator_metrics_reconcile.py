"""Startup operator metrics reconciliation from user_fills (optional)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.enums import Side
from app.exchange.hyperliquid_types import HLFillRaw
from app.operator_metrics_reconcile import try_reconcile_operator_metrics_from_exchange_fills
from app.persistent_runtime_state import PersistentRuntimeState
from app.state import BotState


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _settings(**kwargs: object) -> Settings:
    base = dict(
        _env_file=None,
        HL_ACCOUNT_ADDRESS="0xabc",
        SYMBOL="ETH",
        PERSISTENT_RUNTIME_RECONCILE_FROM_FILLS=True,
        PERSISTENT_RUNTIME_RECONCILE_FILL_WINDOW_CAP=2000,
    )
    base.update(kwargs)
    return Settings(**base)


def test_reconcile_applies_when_window_covers_prior_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = date(2026, 4, 11)
    monkeypatch.setattr(
        "app.operator_metrics_reconcile.utc_now",
        lambda: datetime(2026, 4, 11, 15, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 11, 15, 0, 0, tzinfo=timezone.utc),
    )

    st = BotState(Settings(_env_file=None))
    st.apply_persistent_runtime_state(
        PersistentRuntimeState(
            day_anchor_utc=today,
            daily_realized_pnl=0.0,
            daily_trade_count=0,
            daily_traded_notional=0.0,
            last_fill_ts=None,
            recent_buy_fill_count=0,
            recent_sell_fill_count=0,
        )
    )

    y = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 4, 11, 10, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 4, 11, 11, 0, 0, tzinfo=timezone.utc)
    fills = [
        HLFillRaw(
            fill_id="a",
            oid=1,
            coin="ETH",
            side=Side.BUY,
            px=100.0,
            sz=1.0,
            fee=0.0,
            time_ms=_ms(t2),
            closed_pnl=0.2,
            raw={},
        ),
        HLFillRaw(
            fill_id="b",
            oid=2,
            coin="ETH",
            side=Side.SELL,
            px=101.0,
            sz=1.0,
            fee=0.0,
            time_ms=_ms(t1),
            closed_pnl=-0.1,
            raw={},
        ),
        HLFillRaw(
            fill_id="c",
            oid=3,
            coin="ETH",
            side=Side.BUY,
            px=99.0,
            sz=1.0,
            fee=0.0,
            time_ms=_ms(y),
            closed_pnl=0.0,
            raw={},
        ),
    ]
    client = MagicMock()
    client.fetch_recent_fills_raw.return_value = fills

    try_reconcile_operator_metrics_from_exchange_fills(_settings(), st, client)

    assert st.daily_trade_count == 2
    assert st.recent_buy_fill_count == 1
    assert st.recent_sell_fill_count == 1
    assert abs(st.daily_traded_notional - (100.0 + 101.0)) < 1e-9
    assert abs(st.daily_realized_pnl - 0.1) < 1e-9
    assert st.operator_last_fill_ts == t2


def test_reconcile_keeps_file_when_exchange_fewer_than_persisted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    today = date(2026, 4, 11)
    monkeypatch.setattr(
        "app.operator_metrics_reconcile.utc_now",
        lambda: datetime(2026, 4, 11, 15, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 11, 15, 0, 0, tzinfo=timezone.utc),
    )

    st = BotState(Settings(_env_file=None))
    st.apply_persistent_runtime_state(
        PersistentRuntimeState(
            day_anchor_utc=today,
            daily_realized_pnl=9.0,
            daily_trade_count=5,
            daily_traded_notional=5000.0,
            last_fill_ts=datetime(2026, 4, 11, 8, 0, 0, tzinfo=timezone.utc),
            recent_buy_fill_count=3,
            recent_sell_fill_count=2,
        )
    )

    t1 = datetime(2026, 4, 11, 9, 0, 0, tzinfo=timezone.utc)
    fills = [
        HLFillRaw(
            fill_id="x",
            oid=1,
            coin="ETH",
            side=Side.BUY,
            px=100.0,
            sz=1.0,
            fee=0.0,
            time_ms=_ms(t1),
            closed_pnl=1.0,
            raw={},
        ),
    ]
    client = MagicMock()
    client.fetch_recent_fills_raw.return_value = fills

    with caplog.at_level("WARNING"):
        try_reconcile_operator_metrics_from_exchange_fills(_settings(), st, client)

    assert st.daily_trade_count == 5
    assert st.daily_realized_pnl == 9.0
    assert st.daily_traded_notional == 5000.0
    assert "operator_metrics_reconcile incomplete" in caplog.text


def test_reconcile_skips_at_cap_all_today(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    today = date(2026, 4, 11)
    monkeypatch.setattr(
        "app.operator_metrics_reconcile.utc_now",
        lambda: datetime(2026, 4, 11, 20, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 11, 20, 0, 0, tzinfo=timezone.utc),
    )

    st = BotState(Settings(_env_file=None))
    st.apply_persistent_runtime_state(
        PersistentRuntimeState(
            day_anchor_utc=today,
            daily_realized_pnl=0.0,
            daily_trade_count=0,
            daily_traded_notional=0.0,
            last_fill_ts=None,
            recent_buy_fill_count=0,
            recent_sell_fill_count=0,
        )
    )

    base = datetime(2026, 4, 11, 1, 0, 0, tzinfo=timezone.utc)
    fills = [
        HLFillRaw(
            fill_id=f"id{i}",
            oid=i,
            coin="ETH",
            side=Side.BUY,
            px=100.0,
            sz=0.01,
            fee=0.0,
            time_ms=_ms(base) + i * 1000,
            closed_pnl=0.0,
            raw={},
        )
        for i in range(3)
    ]
    client = MagicMock()
    client.fetch_recent_fills_raw.return_value = fills

    with caplog.at_level("WARNING"):
        try_reconcile_operator_metrics_from_exchange_fills(
            _settings(PERSISTENT_RUNTIME_RECONCILE_FILL_WINDOW_CAP=3),
            st,
            client,
        )

    assert st.daily_trade_count == 0
    assert "operator_metrics_reconcile incomplete" in caplog.text


def test_reconcile_skips_on_fetch_error(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    today = date(2026, 4, 11)
    monkeypatch.setattr(
        "app.operator_metrics_reconcile.utc_now",
        lambda: datetime(2026, 4, 11, 15, 0, 0, tzinfo=timezone.utc),
    )
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 11, 15, 0, 0, tzinfo=timezone.utc),
    )

    st = BotState(Settings(_env_file=None))
    st.apply_persistent_runtime_state(
        PersistentRuntimeState(
            day_anchor_utc=today,
            daily_realized_pnl=2.0,
            daily_trade_count=1,
            daily_traded_notional=50.0,
            last_fill_ts=datetime(2026, 4, 11, 1, 0, 0, tzinfo=timezone.utc),
            recent_buy_fill_count=1,
            recent_sell_fill_count=0,
        )
    )
    client = MagicMock()
    client.fetch_recent_fills_raw.side_effect = RuntimeError("network")

    with caplog.at_level("WARNING"):
        try_reconcile_operator_metrics_from_exchange_fills(_settings(), st, client)

    assert st.daily_trade_count == 1
    assert st.daily_realized_pnl == 2.0
    assert "user_fills fetch failed" in caplog.text


def test_reconcile_disabled_does_not_call_client() -> None:
    st = BotState(Settings(_env_file=None))
    client = MagicMock()
    try_reconcile_operator_metrics_from_exchange_fills(
        Settings(
            _env_file=None,
            HL_ACCOUNT_ADDRESS="0xabc",
            PERSISTENT_RUNTIME_RECONCILE_FROM_FILLS=False,
        ),
        st,
        client,
    )
    client.fetch_recent_fills_raw.assert_not_called()


def test_apply_reconciliation_respects_update_realized_pnl_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 11, 15, 0, 0, tzinfo=timezone.utc),
    )
    st = BotState(Settings(_env_file=None))
    st.daily_realized_pnl = 7.5
    st.apply_operator_exchange_reconciliation(
        trade_count=2,
        traded_notional=200.0,
        realized_pnl=99.0,
        update_realized_pnl=False,
        last_fill_ts=datetime(2026, 4, 11, 14, 0, 0, tzinfo=timezone.utc),
        buy_count=1,
        sell_count=1,
    )
    assert st.daily_trade_count == 2
    assert st.daily_realized_pnl == 7.5
