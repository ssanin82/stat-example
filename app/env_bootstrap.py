"""Optional profile env-file loading before Settings().

Precedence: ``--env-file`` (CLI, when ``argv`` is passed) over ``APP_ENV_FILE``.
Process environment always wins over file values (``load_dotenv(..., override=False)``).

Additionally always loads ``config/telegram.env`` (if present) so Telegram
routing values reach ``Settings`` without operator-side scripting on every
deploy target. The token itself stays in the deploy host's env-var
mechanism (systemd ``EnvironmentFile`` on the current colo box; PaaS
secret store on a container host) — never in that file. Process env
wins, so host-level overrides still take precedence.
"""

from __future__ import annotations

import os
from pathlib import Path

_bootstrap_done: bool = False
_loaded_abs_path: str | None = None
_configured_path: str | None = None
_missing_abs_path: str | None = None
_telegram_env_loaded: str | None = None  # absolute path or None
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TELEGRAM_ENV_PATH = _REPO_ROOT / "config" / "telegram.env"


def reset_env_bootstrap_for_tests() -> None:
    global _bootstrap_done, _loaded_abs_path, _configured_path, _missing_abs_path, _telegram_env_loaded
    _bootstrap_done = False
    _loaded_abs_path = None
    _configured_path = None
    _missing_abs_path = None
    _telegram_env_loaded = None


def extract_env_file_from_argv(argv: list[str]) -> tuple[str | None, list[str]]:
    """Return ``(path, argv_without_env_file_flags)``."""
    out: list[str] = []
    i = 0
    cli: str | None = None
    while i < len(argv):
        a = argv[i]
        if a == "--env-file":
            if i + 1 < len(argv):
                cli = argv[i + 1]
                i += 2
                continue
            i += 1
            continue
        if a.startswith("--env-file="):
            cli = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    return cli, out


def ensure_env_bootstrapped(*, argv: list[str] | None) -> None:
    """Load optional profile env file at most once. Idempotent."""
    global _bootstrap_done, _loaded_abs_path, _configured_path, _missing_abs_path, _telegram_env_loaded
    if _bootstrap_done:
        return

    cli_raw: str | None = None
    if argv is not None:
        cli_raw, _ = extract_env_file_from_argv(argv)
    cli = (cli_raw or "").strip() or None
    env_raw = os.environ.get("APP_ENV_FILE", "").strip() or None
    chosen = cli if cli is not None else env_raw
    _configured_path = chosen

    if chosen:
        p = Path(chosen).expanduser()
        if p.is_file():
            from dotenv import load_dotenv

            load_dotenv(p, override=False)
            _loaded_abs_path = str(p.resolve())
            _missing_abs_path = None
        else:
            _loaded_abs_path = None
            _missing_abs_path = str(p.resolve())
    else:
        _loaded_abs_path = None
        _missing_abs_path = None

    # Always load config/telegram.env if present. The file holds non-secret
    # routing (chat IDs, allowed user IDs); the token stays in the host's
    # env-var mechanism (systemd EnvironmentFile / PaaS secret store).
    # Existing process env wins (override=False), so host-level overrides
    # still take precedence.
    if _TELEGRAM_ENV_PATH.is_file():
        try:
            from dotenv import load_dotenv

            load_dotenv(_TELEGRAM_ENV_PATH, override=False)
            _telegram_env_loaded = str(_TELEGRAM_ENV_PATH.resolve())
        except Exception:
            _telegram_env_loaded = None

    _bootstrap_done = True


def env_file_startup_log_line() -> str:
    """Single-line status for logging (no secret values)."""
    if not _bootstrap_done:
        return "env_file: bootstrap not run (internal)"
    tg_part = (
        f" (+telegram.env: {_telegram_env_loaded})"
        if _telegram_env_loaded is not None
        else ""
    )
    if _configured_path is None:
        return (
            "env_file: none (--env-file and APP_ENV_FILE unset; "
            "using process environment only for Settings)" + tg_part
        )
    if _loaded_abs_path is not None:
        return f"env_file: loaded from {_loaded_abs_path}{tg_part}"
    miss = _missing_abs_path or _configured_path
    return (
        f"env_file: not loaded (missing or not a file: {miss}){tg_part}"
    )
