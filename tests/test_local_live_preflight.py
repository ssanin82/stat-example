"""scripts/local_live_preflight.py exit codes (no live network in CI)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_preflight(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    base = {k: v for k, v in os.environ.items() if not k.startswith("HLMMTEST_")}
    base.update(env)
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "local_live_preflight.py")],
        cwd=str(ROOT),
        env=base,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_local_live_preflight_fails_when_trading_disabled() -> None:
    r = _run_preflight(
        {
            "TRADING_ENABLED": "false",
            "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
            "HL_SECRET_KEY": "0x" + "11" * 32,
            "HL_SECRET_KEY_FILE": "",
        }
    )
    assert r.returncode == 1
    assert "TRADING_ENABLED must be true" in r.stderr


def test_local_live_preflight_fails_when_no_account() -> None:
    r = _run_preflight(
        {
            "TRADING_ENABLED": "true",
            "HL_ACCOUNT_ADDRESS": "",
            "HL_SECRET_KEY": "0x" + "11" * 32,
        }
    )
    assert r.returncode == 1
    assert "HL_ACCOUNT_ADDRESS" in r.stderr


def test_local_live_preflight_fails_when_no_secret() -> None:
    r = _run_preflight(
        {
            "TRADING_ENABLED": "true",
            "HL_ACCOUNT_ADDRESS": "0x" + "aa" * 20,
            "HL_SECRET_KEY": "",
            "HL_SECRET_KEY_FILE": "",
        }
    )
    assert r.returncode == 1
    assert "HL_SECRET_KEY" in r.stderr
