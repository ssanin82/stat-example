"""
Ensure core modules type-check against the exchange Protocol without importing the SDK.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_importing_execution_does_not_load_hyperliquid_sdk() -> None:
    root = Path(__file__).resolve().parents[1]
    code = r"""
import sys
import app.execution
import app.market_data
import app.bot
bad = [
    n
    for n in sys.modules
    if (
        n == "eth_account"
        or n.startswith("hyperliquid.")
        or n == "hyperliquid"
    )
]
assert not bad, "SDK modules loaded: " + repr(bad)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        check=True,
        env=env,
    )
