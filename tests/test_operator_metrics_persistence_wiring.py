"""Operator metrics: restore from disk same UTC day, stale-day reset, fill bumps, save/load continuity."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app.config import Settings
from app.enums import Side
from app.exchange.hyperliquid_types import HLFillRaw
from app.fill_ingestion import ingest_hl_fill_raw
from app.persistent_runtime_state import (
    PersistentRuntimeState,
    load_persistent_runtime_state,
    save_persistent_runtime_state,
)
from app.state import BotState
from app.storage import Storage


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        DATABASE_URL=f"sqlite:///{tmp_path / 't.db'}",
        SQLITE_PATH=str(tmp_path / "t.db"),
    )


def test_apply_same_day_restores_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed = datetime(2026, 4, 11, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("app.state.utc_now", lambda: fixed)

    st = BotState(Settings(_env_file=None))
    loaded = PersistentRuntimeState(
        day_anchor_utc=date(2026, 4, 11),
        daily_realized_pnl=3.25,
        daily_trade_count=7,
        daily_traded_notional=1400.0,
        last_fill_ts=datetime(2026, 4, 11, 11, 0, 0, tzinfo=timezone.utc),
        recent_buy_fill_count=4,
        recent_sell_fill_count=3,
        rolling_toxicity_markout_bps=-0.5,
        rolling_one_sided_fill_ratio=0.55,
    )
    st.apply_persistent_runtime_state(loaded)
    assert st.operator_day_anchor_utc == date(2026, 4, 11)
    assert st.daily_realized_pnl == 3.25
    assert st.daily_trade_count == 7
    assert st.daily_traded_notional == 1400.0
    assert st.recent_buy_fill_count == 4
    assert st.recent_sell_fill_count == 3
    assert st.operator_rolling_toxicity_markout_bps == -0.5
    assert st.operator_rolling_one_sided_fill_ratio == 0.55


def test_apply_prior_utc_day_resets_day_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 12, 0, 1, 0, tzinfo=timezone.utc),
    )
    st = BotState(Settings(_env_file=None))
    loaded = PersistentRuntimeState(
        day_anchor_utc=date(2026, 4, 11),
        daily_realized_pnl=99.0,
        daily_trade_count=50,
        daily_traded_notional=50_000.0,
        last_fill_ts=datetime(2026, 4, 11, 23, 59, 0, tzinfo=timezone.utc),
        recent_buy_fill_count=30,
        recent_sell_fill_count=20,
        rolling_toxicity_markout_bps=-2.0,
        rolling_one_sided_fill_ratio=0.8,
    )
    st.apply_persistent_runtime_state(loaded)
    assert st.operator_day_anchor_utc == date(2026, 4, 12)
    assert st.daily_realized_pnl == 0.0
    assert st.daily_trade_count == 0
    assert st.daily_traded_notional == 0.0
    assert st.recent_buy_fill_count == 0
    assert st.recent_sell_fill_count == 0
    assert st.operator_last_fill_ts is None
    assert st.operator_rolling_toxicity_markout_bps is None
    assert st.operator_rolling_one_sided_fill_ratio is None


def test_wall_clock_midnight_rotates_without_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    st = BotState(Settings(_env_file=None))
    st.operator_day_anchor_utc = date(2026, 4, 11)
    st.daily_trade_count = 9
    st.recent_buy_fill_count = 5
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 12, 0, 5, 0, tzinfo=timezone.utc),
    )
    st.maybe_rotate_operator_day_to_wall_clock()
    assert st.operator_day_anchor_utc == date(2026, 4, 12)
    assert st.daily_trade_count == 0
    assert st.recent_buy_fill_count == 0


def test_session_fill_bumps_and_crosses_utc_day(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 11, 23, 59, 0, tzinfo=timezone.utc),
    )
    st = BotState(Settings(_env_file=None))
    st.session_started_at_utc = datetime(2026, 4, 11, 0, 0, 0, tzinfo=timezone.utc)

    raw = HLFillRaw(
        fill_id="f1",
        oid=1,
        coin="ETH",
        side=Side.BUY,
        px=3000.0,
        sz=0.01,
        fee=0.01,
        time_ms=int(datetime(2026, 4, 11, 23, 0, 0, tzinfo=timezone.utc).timestamp() * 1000),
        closed_pnl=0.5,
        raw={},
    )
    ingest_hl_fill_raw(
        state=st,
        storage=None,
        pnl=None,
        symbol="ETH",
        fr=raw,
        source="rest",
    )
    assert st.daily_trade_count == 1
    assert st.daily_realized_pnl == 0.5
    assert st.recent_buy_fill_count == 1

    raw2 = HLFillRaw(
        fill_id="f2",
        oid=2,
        coin="ETH",
        side=Side.SELL,
        px=3010.0,
        sz=0.01,
        fee=0.01,
        time_ms=int(datetime(2026, 4, 12, 0, 1, 0, tzinfo=timezone.utc).timestamp() * 1000),
        closed_pnl=-0.1,
        raw={},
    )
    ingest_hl_fill_raw(
        state=st,
        storage=None,
        pnl=None,
        symbol="ETH",
        fr=raw2,
        source="rest",
    )
    assert st.operator_day_anchor_utc == date(2026, 4, 12)
    assert st.daily_trade_count == 1
    assert abs(st.daily_realized_pnl - (-0.1)) < 1e-9
    assert st.recent_sell_fill_count == 1
    assert st.recent_buy_fill_count == 0


def test_restart_continuity_via_json_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.state.utc_now",
        lambda: datetime(2026, 4, 11, 18, 0, 0, tzinfo=timezone.utc),
    )
    settings = _settings(tmp_path)
    storage = Storage(settings)
    storage.init_schema()
    path = tmp_path / "op.json"

    s1 = BotState(settings)
    s1.session_started_at_utc = datetime(2026, 4, 11, 0, 0, 0, tzinfo=timezone.utc)
    raw = HLFillRaw(
        fill_id="persist1",
        oid=1,
        coin="ETH",
        side=Side.BUY,
        px=2000.0,
        sz=0.02,
        fee=0.02,
        time_ms=int(datetime(2026, 4, 11, 17, 0, 0, tzinfo=timezone.utc).timestamp() * 1000),
        closed_pnl=1.25,
        raw={},
    )
    ingest_hl_fill_raw(
        state=s1,
        storage=storage,
        pnl=None,
        symbol="ETH",
        fr=raw,
        source="rest",
    )
    s1.operator_rolling_toxicity_markout_bps = -1.0
    s1.operator_rolling_one_sided_fill_ratio = 0.66

    prs = s1.build_persistent_runtime_state()
    save_persistent_runtime_state(path, prs)

    s2 = BotState(settings)
    loaded = load_persistent_runtime_state(path)
    assert loaded is not None
    s2.apply_persistent_runtime_state(loaded)

    assert s2.daily_trade_count == 1
    assert s2.daily_realized_pnl == 1.25
    assert s2.recent_buy_fill_count == 1
    assert s2.operator_rolling_toxicity_markout_bps == -1.0
    assert s2.operator_rolling_one_sided_fill_ratio == 0.66

    outer = json.loads(path.read_text(encoding="utf-8"))
    assert outer["data"]["daily_trade_count"] == 1
