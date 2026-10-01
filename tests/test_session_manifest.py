"""Phase F1 (v1.5.42) — tests for bot-side session forensic manifests.

See ``backtesting/docs/execution-plan.md`` §Phase F1 for the
contract these tests pin down (archived spec at
``plans/_DONE/forensics.md``).

The module's three responsibilities:

1. ``read_recorder_pointer`` — finds the recorder's session dir via
   ``/tmp/dtc-mm-as-recorder-active-session-{profile}.txt``; returns
   ``None`` cleanly when the pointer is absent OR points to a
   missing dir.
2. ``write_bot_manifest`` — composes 3 atomic-rename writes:
   ``bot_manifest.json`` + ``bot_config_resolved.json`` +
   ``bot_config_envfile.txt``. Redacts secrets in the env-file
   copy; sanitises secrets in the resolved config.
3. ``write_bot_shutdown`` — atomic-rename write of
   ``bot_shutdown.json``. Idempotent: second call no-ops if file
   exists.

Plus the second-start behaviour: if ``bot_manifest.json`` already
exists in the session dir, writes ``bot_manifest_2.json`` etc.
"""

from __future__ import annotations

import json
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from app.session_manifest import (
    read_recorder_pointer,
    write_bot_manifest,
    write_bot_shutdown,
    _sanitize_env_file,
    _next_manifest_path,
)


# ---------------------------------------------------------------------------
# read_recorder_pointer
# ---------------------------------------------------------------------------


def test_read_recorder_pointer_missing_returns_none(monkeypatch, tmp_path):
    """No pointer file → None, no exception. Used when recorder is
    disabled or hasn't started."""
    # Force-redirect /tmp probe to a non-existent path by patching Path.
    # Simpler: just call with a profile name that's not deployed.
    assert read_recorder_pointer("nonexistent-profile-xyz-12345") is None


def test_read_recorder_pointer_present_returns_path(tmp_path, monkeypatch):
    """Pointer file present + valid → returns the Path to session dir."""
    session_dir = tmp_path / "session-abc"
    session_dir.mkdir()
    pointer_path = tmp_path / "pointer.txt"
    pointer_path.write_text(str(session_dir), encoding="utf-8")

    # Patch the pointer-path resolution inside the helper.
    real_is_file = Path.is_file
    real_read_text = Path.read_text
    real_is_dir = Path.is_dir

    def fake_is_file(self):
        if "dtc-mm-as-recorder-active-session" in str(self):
            return pointer_path.is_file()
        return real_is_file(self)

    def fake_read_text(self, *args, **kwargs):
        if "dtc-mm-as-recorder-active-session" in str(self):
            return pointer_path.read_text(*args, **kwargs)
        return real_read_text(self, *args, **kwargs)

    def fake_is_dir(self):
        return real_is_dir(self)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(Path, "is_dir", fake_is_dir)

    result = read_recorder_pointer("test-profile")
    assert result == session_dir


def test_read_recorder_pointer_stale_dir_returns_none(
    tmp_path, monkeypatch
):
    """Pointer file present but the dir it points to has been deleted
    → None. Defensive against orphaned pointers."""
    pointer_path = tmp_path / "pointer.txt"
    pointer_path.write_text("/tmp/this-dir-does-not-exist-xyz", encoding="utf-8")

    real_is_file = Path.is_file
    real_read_text = Path.read_text

    def fake_is_file(self):
        if "dtc-mm-as-recorder-active-session" in str(self):
            return pointer_path.is_file()
        return real_is_file(self)

    def fake_read_text(self, *args, **kwargs):
        if "dtc-mm-as-recorder-active-session" in str(self):
            return pointer_path.read_text(*args, **kwargs)
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", fake_is_file)
    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert read_recorder_pointer("test-profile") is None


# ---------------------------------------------------------------------------
# _sanitize_env_file — secret redaction
# ---------------------------------------------------------------------------


def test_sanitize_env_file_redacts_api_key_lines(tmp_path):
    p = tmp_path / "test.env"
    p.write_text(
        "\n".join(
            [
                "# Header comment",
                "SYMBOL=TON-USDT-SWAP",
                "OKX_API_KEY=secret-key-123",
                "OKX_API_SECRET=secret-secret-456",
                "OKX_API_PASSPHRASE=secret-pass-789",
                "MAX_DRAWDOWN_USD=1.0",
                "",
                "# Trailing comment",
            ]
        ),
        encoding="utf-8",
    )
    sanitized = _sanitize_env_file(p)
    # Public keys preserved.
    assert "SYMBOL=TON-USDT-SWAP" in sanitized
    assert "MAX_DRAWDOWN_USD=1.0" in sanitized
    # Comments preserved.
    assert "# Header comment" in sanitized
    assert "# Trailing comment" in sanitized
    # Secret values redacted.
    assert "secret-key-123" not in sanitized
    assert "secret-secret-456" not in sanitized
    assert "secret-pass-789" not in sanitized
    assert "OKX_API_KEY={REDACTED}" in sanitized
    assert "OKX_API_SECRET={REDACTED}" in sanitized
    assert "OKX_API_PASSPHRASE={REDACTED}" in sanitized


def test_sanitize_env_file_redacts_vendor_prefixed_keys(tmp_path):
    """Substring match catches e.g. ``BLUEFIN_API_KEY`` too."""
    p = tmp_path / "test.env"
    p.write_text(
        "BLUEFIN_API_KEY=vendor-secret\n"
        "BINANCE_API_SECRET=binance-secret\n"
        "PRIVATE_KEY=sshhh\n",
        encoding="utf-8",
    )
    sanitized = _sanitize_env_file(p)
    assert "vendor-secret" not in sanitized
    assert "binance-secret" not in sanitized
    assert "sshhh" not in sanitized
    assert "{REDACTED}" in sanitized


def test_sanitize_env_file_handles_unreadable_file(tmp_path):
    """Pointer to a non-existent file → returns a stub, doesn't raise."""
    p = tmp_path / "nope.env"
    out = _sanitize_env_file(p)
    assert "env file read failed" in out


# ---------------------------------------------------------------------------
# _next_manifest_path — second-start no-overwrite
# ---------------------------------------------------------------------------


def test_next_manifest_path_returns_base_when_absent(tmp_path):
    p = _next_manifest_path(tmp_path, "bot_manifest.json")
    assert p == tmp_path / "bot_manifest.json"


def test_next_manifest_path_returns_2_when_base_present(tmp_path):
    (tmp_path / "bot_manifest.json").write_text("{}", encoding="utf-8")
    p = _next_manifest_path(tmp_path, "bot_manifest.json")
    assert p == tmp_path / "bot_manifest_2.json"


def test_next_manifest_path_returns_3_when_2_also_present(tmp_path):
    (tmp_path / "bot_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "bot_manifest_2.json").write_text("{}", encoding="utf-8")
    p = _next_manifest_path(tmp_path, "bot_manifest.json")
    assert p == tmp_path / "bot_manifest_3.json"


# ---------------------------------------------------------------------------
# write_bot_manifest — end-to-end
# ---------------------------------------------------------------------------


def _fake_state(
    *,
    position_qty: float = 0.0,
    position_notional: float = 0.0,
    avg_entry_price: float = 0.0,
    equity_usd: float = 1000.0,
    session_id: str = "test-session-12345",
    realized: float = 0.0,
    unrealized: float = 0.0,
    fees: float = 0.0,
    fills_total: int = 0,
    fills_buy: int = 0,
    fills_sell: int = 0,
    killed: bool = False,
) -> SimpleNamespace:
    """A minimal BotState-shaped object for the helpers."""
    return SimpleNamespace(
        position=SimpleNamespace(
            position_qty=position_qty,
            position_notional=position_notional,
            avg_entry_price=avg_entry_price,
        ),
        account=SimpleNamespace(equity_usd=equity_usd),
        pnl=SimpleNamespace(
            realized_pnl_usd=realized,
            unrealized_pnl_usd=unrealized,
            fees_usd=fees,
        ),
        session_id=session_id,
        session_started_at_utc=datetime(
            2026, 5, 23, 12, 0, 0, tzinfo=timezone.utc
        ),
        session_fill_count=fills_total,
        session_fill_count_by_side={"BUY": fills_buy, "SELL": fills_sell},
        killed=killed,
    )


def _fake_settings() -> SimpleNamespace:
    return SimpleNamespace(
        symbol="TON-USDT-SWAP",
        sanitized_dict=lambda: {
            "SYMBOL": "TON-USDT-SWAP",
            "MAX_DRAWDOWN_USD": 1.0,
            "OKX_API_KEY": "[REDACTED]",
        },
    )


def test_write_bot_manifest_produces_all_three_files(tmp_path):
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "SYMBOL=TON-USDT-SWAP\nMAX_DRAWDOWN_USD=1.0\nOKX_API_KEY=foo\n",
        encoding="utf-8",
    )
    write_bot_manifest(
        session_dir=tmp_path,
        state=_fake_state(position_qty=3.0, position_notional=5.85),
        settings=_fake_settings(),
        env_file_path=env_file,
        bot_version="1.5.42",
        bot_profile="prod.okx.ton.usdt.perp",
    )
    assert (tmp_path / "bot_manifest.json").is_file()
    assert (tmp_path / "bot_config_resolved.json").is_file()
    assert (tmp_path / "bot_config_envfile.txt").is_file()


def test_write_bot_manifest_redacts_secrets_in_envfile(tmp_path):
    env_file = tmp_path / "test.env"
    env_file.write_text(
        "SYMBOL=TON\nOKX_API_KEY=should-be-hidden\n", encoding="utf-8"
    )
    write_bot_manifest(
        session_dir=tmp_path,
        state=_fake_state(),
        settings=_fake_settings(),
        env_file_path=env_file,
        bot_version="1.5.42",
        bot_profile="test",
    )
    envfile_contents = (tmp_path / "bot_config_envfile.txt").read_text(
        encoding="utf-8"
    )
    assert "should-be-hidden" not in envfile_contents
    assert "{REDACTED}" in envfile_contents


def test_write_bot_manifest_includes_initial_inventory(tmp_path):
    env_file = tmp_path / "test.env"
    env_file.write_text("SYMBOL=TON\n", encoding="utf-8")
    write_bot_manifest(
        session_dir=tmp_path,
        state=_fake_state(
            position_qty=3.0,
            position_notional=5.85,
            avg_entry_price=1.957,
        ),
        settings=_fake_settings(),
        env_file_path=env_file,
        bot_version="1.5.42",
        bot_profile="test",
    )
    manifest = json.loads(
        (tmp_path / "bot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["initial_position_qty"] == 3.0
    assert manifest["initial_position_notional_usd"] == 5.85
    assert manifest["initial_avg_entry_price"] == 1.957
    assert manifest["bot_version"] == "1.5.42"
    assert manifest["bot_profile"] == "prod.okx.ton.usdt.perp" or \
        manifest["bot_profile"] == "test"  # whichever passed
    assert manifest["symbol"] == "TON-USDT-SWAP"
    assert manifest["session_id"] == "test-session-12345"
    # Hostname captured.
    assert manifest["hostname"] == socket.gethostname()


def test_write_bot_manifest_resolved_config_uses_sanitized_dict(tmp_path):
    """The resolved-config dump comes from ``Settings.sanitized_dict()``
    which already strips secrets by name pattern."""
    env_file = tmp_path / "test.env"
    env_file.write_text("SYMBOL=TON\n", encoding="utf-8")
    write_bot_manifest(
        session_dir=tmp_path,
        state=_fake_state(),
        settings=_fake_settings(),  # sanitized_dict has [REDACTED] for API_KEY
        env_file_path=env_file,
        bot_version="1.5.42",
        bot_profile="test",
    )
    resolved = json.loads(
        (tmp_path / "bot_config_resolved.json").read_text(encoding="utf-8")
    )
    assert resolved["OKX_API_KEY"] == "[REDACTED]"


def test_write_bot_manifest_second_start_does_not_overwrite(tmp_path):
    """Second bot start under same recorder → bot_manifest_2.json."""
    env_file = tmp_path / "test.env"
    env_file.write_text("SYMBOL=TON\n", encoding="utf-8")
    # First start.
    write_bot_manifest(
        session_dir=tmp_path,
        state=_fake_state(position_qty=3.0),
        settings=_fake_settings(),
        env_file_path=env_file,
        bot_version="1.5.42",
        bot_profile="test",
    )
    # Second start.
    write_bot_manifest(
        session_dir=tmp_path,
        state=_fake_state(position_qty=-3.0),
        settings=_fake_settings(),
        env_file_path=env_file,
        bot_version="1.5.42",
        bot_profile="test",
    )
    assert (tmp_path / "bot_manifest.json").is_file()
    assert (tmp_path / "bot_manifest_2.json").is_file()
    # First manifest preserved.
    first = json.loads(
        (tmp_path / "bot_manifest.json").read_text(encoding="utf-8")
    )
    second = json.loads(
        (tmp_path / "bot_manifest_2.json").read_text(encoding="utf-8")
    )
    assert first["initial_position_qty"] == 3.0
    assert second["initial_position_qty"] == -3.0


def test_write_bot_manifest_no_env_file_path_omits_envfile(tmp_path):
    """``env_file_path=None`` → still writes the two json files but
    no envfile.txt."""
    write_bot_manifest(
        session_dir=tmp_path,
        state=_fake_state(),
        settings=_fake_settings(),
        env_file_path=None,
        bot_version="1.5.42",
        bot_profile="test",
    )
    assert (tmp_path / "bot_manifest.json").is_file()
    assert (tmp_path / "bot_config_resolved.json").is_file()
    assert not (tmp_path / "bot_config_envfile.txt").exists()


# ---------------------------------------------------------------------------
# write_bot_shutdown
# ---------------------------------------------------------------------------


def test_write_bot_shutdown_creates_file(tmp_path):
    write_bot_shutdown(
        session_dir=tmp_path,
        state=_fake_state(
            position_qty=3.0,
            position_notional=5.85,
            equity_usd=976.7,
            realized=0.022,
            unrealized=-0.005,
            fees=-0.030,
            fills_total=78,
            fills_buy=39,
            fills_sell=39,
        ),
        bot_version="1.5.42",
        kill_reason=None,
    )
    assert (tmp_path / "bot_shutdown.json").is_file()
    payload = json.loads(
        (tmp_path / "bot_shutdown.json").read_text(encoding="utf-8")
    )
    assert payload["final_position_qty"] == 3.0
    assert payload["final_equity_usd"] == 976.7
    assert payload["realized_pnl_usd"] == 0.022
    assert payload["fills_total"] == 78
    assert payload["kill_reason"] is None


def test_write_bot_shutdown_with_kill_reason(tmp_path):
    write_bot_shutdown(
        session_dir=tmp_path,
        state=_fake_state(killed=True),
        bot_version="1.5.42",
        kill_reason="stale_data_kill_escalated",
    )
    payload = json.loads(
        (tmp_path / "bot_shutdown.json").read_text(encoding="utf-8")
    )
    assert payload["kill_reason"] == "stale_data_kill_escalated"
    assert payload["killed"] is True


def test_write_bot_shutdown_idempotent(tmp_path):
    """Second call no-ops if file exists. Paired stop()+kill() both
    invoke this; first write wins."""
    write_bot_shutdown(
        session_dir=tmp_path,
        state=_fake_state(position_qty=3.0),
        bot_version="1.5.42",
        kill_reason="first",
    )
    write_bot_shutdown(
        session_dir=tmp_path,
        state=_fake_state(position_qty=-99.0),  # would clobber
        bot_version="1.5.42",
        kill_reason="second",
    )
    payload = json.loads(
        (tmp_path / "bot_shutdown.json").read_text(encoding="utf-8")
    )
    # First-write-wins: position is 3.0, kill_reason "first".
    assert payload["final_position_qty"] == 3.0
    assert payload["kill_reason"] == "first"
