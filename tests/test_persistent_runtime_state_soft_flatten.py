"""Tests for soft-flatten persistence across restart (Codex HIGH #2).

A crash during ``SOFT_FLATTENING`` previously came back with the
adverse position seeded but the flatten *intent* lost -- the bot
would resume normal quoting on adverse inventory until the
drawdown gate re-fired (another 30+ s of breach). Persistence in
``PersistentRuntimeState`` closes this gap.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import date
from pathlib import Path

import pytest

from app.persistent_runtime_state import (
    PERSISTENT_RUNTIME_STATE_VERSION,
    PersistentRuntimeState,
    load_persistent_runtime_state,
    save_persistent_runtime_state,
)


def _tmp_path() -> Path:
    return Path(tempfile.gettempdir()) / f"prs_sf_{os.getpid()}_{uuid.uuid4().hex}.json"


def _base_state(**kw) -> PersistentRuntimeState:
    base = dict(
        day_anchor_utc=date.today(),
        daily_realized_pnl=0.0,
        daily_trade_count=0,
        daily_traded_notional=0.0,
    )
    base.update(kw)
    return PersistentRuntimeState(**base)


def test_default_state_is_soft_flatten_inactive() -> None:
    s = _base_state()
    assert s.soft_flatten_active is False
    assert s.soft_flatten_started_at_iso is None


def test_save_load_roundtrip_preserves_soft_flatten_active() -> None:
    p = _tmp_path()
    s = _base_state(
        soft_flatten_active=True,
        soft_flatten_started_at_iso="2026-05-06T10:23:11+00:00",
    )
    save_persistent_runtime_state(p, s)
    loaded = load_persistent_runtime_state(p)
    assert loaded is not None
    assert loaded.soft_flatten_active is True
    assert loaded.soft_flatten_started_at_iso == "2026-05-06T10:23:11+00:00"


def test_save_load_roundtrip_preserves_soft_flatten_inactive() -> None:
    p = _tmp_path()
    s = _base_state(soft_flatten_active=False)
    save_persistent_runtime_state(p, s)
    loaded = load_persistent_runtime_state(p)
    assert loaded.soft_flatten_active is False
    assert loaded.soft_flatten_started_at_iso is None


def test_load_pre_v2_file_is_rejected() -> None:
    """Schema version bumped to 2 because new fields. Old v1 files
    must be rejected (load returns None) so we don't load with
    soft_flatten_active=False unintentionally. The first run on the
    new bot will write v2."""
    p = _tmp_path()
    p.write_text(
        json.dumps(
            {
                "version": 1,  # old version
                "data": {
                    "day_anchor_utc": date.today().isoformat(),
                    "daily_realized_pnl": 0.0,
                    "daily_trade_count": 0,
                    "daily_traded_notional": 0.0,
                    "recent_buy_fill_count": 0,
                    "recent_sell_fill_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_persistent_runtime_state(p) is None


def test_load_v2_file_with_soft_flatten_loads() -> None:
    p = _tmp_path()
    p.write_text(
        json.dumps(
            {
                "version": PERSISTENT_RUNTIME_STATE_VERSION,
                "data": {
                    "day_anchor_utc": date.today().isoformat(),
                    "daily_realized_pnl": -1.5,
                    "daily_trade_count": 3,
                    "daily_traded_notional": 100.0,
                    "recent_buy_fill_count": 2,
                    "recent_sell_fill_count": 1,
                    "soft_flatten_active": True,
                    "soft_flatten_started_at_iso": "2026-05-06T08:00:00+00:00",
                },
            }
        ),
        encoding="utf-8",
    )
    s = load_persistent_runtime_state(p)
    assert s is not None
    assert s.soft_flatten_active is True
    assert s.soft_flatten_started_at_iso == "2026-05-06T08:00:00+00:00"


def test_load_v2_file_without_soft_flatten_field_defaults_inactive() -> None:
    """Backward-compat within v2: missing fields default to inactive."""
    p = _tmp_path()
    p.write_text(
        json.dumps(
            {
                "version": PERSISTENT_RUNTIME_STATE_VERSION,
                "data": {
                    "day_anchor_utc": date.today().isoformat(),
                    "daily_realized_pnl": 0.0,
                    "daily_trade_count": 0,
                    "daily_traded_notional": 0.0,
                    "recent_buy_fill_count": 0,
                    "recent_sell_fill_count": 0,
                    # no soft_flatten_* keys
                },
            }
        ),
        encoding="utf-8",
    )
    s = load_persistent_runtime_state(p)
    assert s is not None
    assert s.soft_flatten_active is False
    assert s.soft_flatten_started_at_iso is None


# ---------------------------------------------------------------------------
# State integration: apply_persistent_runtime_state restores soft-flatten
# ---------------------------------------------------------------------------


def test_apply_loaded_state_restores_soft_flatten_active() -> None:
    """``apply_persistent_runtime_state`` flips ``soft_flatten_active``
    on the BotState. main.py then sets bot_status accordingly."""
    from app.state import BotState
    from tests.settings_helpers import UnitTestSettings

    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": "sqlite:///:memory:",
        }
    )
    state = BotState(settings)
    assert state.soft_flatten_active is False

    state.apply_persistent_runtime_state(
        _base_state(
            soft_flatten_active=True,
            soft_flatten_started_at_iso="2026-05-06T10:00:00+00:00",
        )
    )
    assert state.soft_flatten_active is True
    assert state.soft_flatten_started_at_mono is not None


def test_apply_loaded_state_does_not_restore_when_inactive() -> None:
    from app.state import BotState
    from tests.settings_helpers import UnitTestSettings

    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": "sqlite:///:memory:",
        }
    )
    state = BotState(settings)
    state.apply_persistent_runtime_state(_base_state(soft_flatten_active=False))
    assert state.soft_flatten_active is False


def test_build_state_emits_soft_flatten_when_active() -> None:
    """Save path: when state has soft_flatten_active=True, the
    built ``PersistentRuntimeState`` carries it forward."""
    from app.state import BotState
    from tests.settings_helpers import UnitTestSettings
    import time

    settings = UnitTestSettings.model_validate(
        {
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "x",
            "HL_ACCOUNT_ADDRESS": "0xabc",
            "DATABASE_URL": "sqlite:///:memory:",
        }
    )
    state = BotState(settings)
    state.soft_flatten_active = True
    state.soft_flatten_started_at_mono = time.monotonic()
    built = state.build_persistent_runtime_state()
    assert built.soft_flatten_active is True
    # ISO is best-effort wall-clock; just assert it's non-None when active.
    assert built.soft_flatten_started_at_iso is not None
