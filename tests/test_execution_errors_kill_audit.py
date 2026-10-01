"""Tests for the windowed execution-errors kill audit.

Covers:
- per-source bump tagging and breakdown
- rolling-window decay (old bumps stop contributing)
- zero-window opt-out (legacy cumulative semantics)
- storm inside window still triggers kill via evaluate_risk
- state snapshot exposes `execution_errors_snapshot`
- ``Bot._build_kill_payload`` attaches a breakdown when killing on execution_errors
- all 12 call-site tags are non-empty strings (contract check)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from app.enums import BotStatus
from app.models import PnlSnapshot, ToxicitySnapshot
from app.risk import evaluate_risk
from app.state import BotState
from app.utils.time import utc_now

from tests.settings_helpers import UnitTestSettings


def _settings(window_seconds: float = 300.0, max_errors: int = 8) -> UnitTestSettings:
    return UnitTestSettings.model_validate(
        {
            "MAX_EXECUTION_ERRORS": max_errors,
            "EXECUTION_ERRORS_WINDOW_SECONDS": window_seconds,
            "TRADING_ENABLED": True,
            "HL_SECRET_KEY": "0x" + "11" * 32,
            "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
        }
    )


def _pnl_zero() -> PnlSnapshot:
    return PnlSnapshot(0.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 1000.0, utc_now())


# ---------------------------------------------------------------------------
# Windowing / snapshot
# ---------------------------------------------------------------------------

def test_bump_records_source_and_increments_lifetime() -> None:
    s = _settings()
    state = BotState(s)
    state.bump_execution_errors("cancel_http_exchange_reject")
    state.bump_execution_errors("orphan_cancel_transport_exception")
    state.bump_execution_errors("cancel_http_exchange_reject")
    snap = state.execution_errors_window_snapshot(60.0)
    assert snap["total"] == 3
    assert snap["windowed"] == 3
    assert snap["sources"] == {
        "cancel_http_exchange_reject": 2,
        "orphan_cancel_transport_exception": 1,
    }
    assert snap["window_seconds"] == 60.0


def test_window_decays_old_bumps(monkeypatch) -> None:
    state = BotState(_settings())
    real_mono = time.monotonic
    fake_now = [real_mono()]

    def fake_monotonic() -> float:
        return fake_now[0]

    monkeypatch.setattr("app.state.time.monotonic", fake_monotonic)
    state.bump_execution_errors("sync_open_orders_rate_limited")
    state.bump_execution_errors("cancel_http_exchange_reject")
    # Advance past the window.
    fake_now[0] += 61.0
    state.bump_execution_errors("orphan_cancel_transport_exception")
    snap = state.execution_errors_window_snapshot(60.0)
    assert snap["total"] == 3
    assert snap["windowed"] == 1
    assert snap["sources"] == {"orphan_cancel_transport_exception": 1}


def test_zero_window_opt_out_is_cumulative() -> None:
    state = BotState(_settings(window_seconds=0.0))
    for _ in range(5):
        state.bump_execution_errors("bot_tick_exception")
    snap = state.execution_errors_window_snapshot(0.0)
    assert snap["total"] == 5
    assert snap["windowed"] == 5
    assert snap["window_seconds"] == 0.0


# ---------------------------------------------------------------------------
# Risk gate
# ---------------------------------------------------------------------------

def test_evaluate_risk_kills_when_windowed_count_meets_threshold() -> None:
    s = _settings(max_errors=3)
    r = evaluate_risk(
        s,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=None,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=_pnl_zero(),
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=3,
        desync=False,
    )
    assert r.action.value == "KILL"
    assert "execution_errors" in r.reasons


def test_evaluate_risk_does_not_kill_when_windowed_below_threshold() -> None:
    s = _settings(max_errors=8)
    # Simulate windowed count below threshold even though lifetime could be higher.
    r = evaluate_risk(
        s,
        bot_status=BotStatus.RUNNING,
        manual_pause=False,
        killed=False,
        flatten_mode=False,
        market=None,
        position_qty=0.0,
        position_notional=0.0,
        open_order_count=0,
        pnl=_pnl_zero(),
        toxicity=ToxicitySnapshot(0, 0, 0, 1, False, False),
        execution_errors=3,
        desync=False,
    )
    # No other adverse signal present, so no kill.
    assert r.action.value != "KILL"


# ---------------------------------------------------------------------------
# State snapshot_dict exposure
# ---------------------------------------------------------------------------

def test_snapshot_dict_includes_execution_errors_snapshot() -> None:
    state = BotState(_settings())
    state.bump_execution_errors("cancel_http_exchange_reject")
    d = state.snapshot_dict()
    assert "execution_errors_snapshot" in d
    block = d["execution_errors_snapshot"]
    assert block["total"] == 1
    assert block["windowed"] == 1
    assert block["sources"] == {"cancel_http_exchange_reject": 1}


# ---------------------------------------------------------------------------
# Bot._build_kill_payload
# ---------------------------------------------------------------------------

def test_build_kill_payload_includes_breakdown_for_execution_errors() -> None:
    from app.bot import Bot  # local to avoid heavy import at module load

    state = BotState(_settings(max_errors=4))
    state.bump_execution_errors("cancel_http_exchange_reject")
    state.bump_execution_errors("orphan_cancel_exchange_reject")
    state.bump_execution_errors("cancel_http_exchange_reject")

    # Build a minimal Bot shell — we only need ``_state`` and ``_settings``.
    bot = Bot.__new__(Bot)
    bot._state = state
    bot._settings = state._settings

    payload = bot._build_kill_payload(["execution_errors"])
    assert payload is not None
    assert payload["reasons"] == ["execution_errors"]
    ee = payload["execution_errors"]
    assert ee["total"] == 3
    assert ee["windowed"] == 3
    assert ee["threshold"] == 4
    assert ee["sources"]["cancel_http_exchange_reject"] == 2
    assert ee["sources"]["orphan_cancel_exchange_reject"] == 1


def test_build_kill_payload_returns_none_for_other_reasons() -> None:
    from app.bot import Bot

    state = BotState(_settings())
    bot = Bot.__new__(Bot)
    bot._state = state
    bot._settings = state._settings
    assert bot._build_kill_payload(["max_session_loss"]) is None
    assert bot._build_kill_payload([]) is None


# ---------------------------------------------------------------------------
# Call-site tag contract
# ---------------------------------------------------------------------------

_EXPECTED_TAGS = {
    "bot_tick_exception",
    "account_refresh_exception",
    "cancel_all_fetch_exception",
    "cancel_all_cancel_exception",
    "cancel_all_exchange_reject",
    "sync_open_orders_rate_limited",
    "sync_open_orders_exception",
    "orphan_cancel_transport_exception",
    "orphan_cancel_exchange_reject",
    "place_order_exception",
    "cancel_http_transport_exception",
    "cancel_http_exchange_reject",
}


def test_all_known_tags_accepted_and_attributed() -> None:
    state = BotState(_settings())
    for tag in _EXPECTED_TAGS:
        state.bump_execution_errors(tag)
    snap = state.execution_errors_window_snapshot(60.0)
    assert snap["total"] == len(_EXPECTED_TAGS)
    assert set(snap["sources"].keys()) == _EXPECTED_TAGS
    assert all(v == 1 for v in snap["sources"].values())


def test_bump_sites_present_in_source_tree() -> None:
    """Guardrail: every bump in app/ must pass a non-empty source tag.

    If a new bump site is added without a tag, this test fails so the author
    is forced to classify it and update _EXPECTED_TAGS / the audit report.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "app"
    pattern = re.compile(r"bump_execution_errors\(\s*([^)]*)\)")
    bump_args: list[str] = []
    for py in root.rglob("*.py"):
        for m in pattern.finditer(py.read_text(encoding="utf-8")):
            arg = m.group(1).strip()
            # Skip the signature definition in state.py itself.
            if arg.startswith("self") or arg.startswith('source:'):
                continue
            bump_args.append(arg)
    # Every bump call must pass a quoted string literal tag (not empty, not None).
    assert bump_args, "no bump_execution_errors call sites found"
    for arg in bump_args:
        assert arg.startswith('"') and arg.endswith('"') and len(arg) > 2, (
            f"bump_execution_errors without source tag: {arg!r}"
        )
