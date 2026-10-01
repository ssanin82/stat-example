"""Phase F1 (v1.5.42) — bot-side session-start + shutdown forensic manifests.

Writes structured metadata into the recorder's session dir at bot
startup so that recorded sessions are self-describing for backtest
reconstruction. Companion shutdown manifest on clean stop.

This module is the BOT side of the forensics contract. The recorder
side (market-data capture) is independent and lives under
``backtesting/recorder/``. The two meet at the recorder's session
dir on disk: the recorder creates it + writes its pointer file at
``ExecStartPre`` time; the bot finds the pointer at startup and
writes its artifacts into the same dir.

Files written (under the recorder's ``session_dir``):

* ``bot_manifest.json`` — bot session metadata (version, session_id,
  session_started_at_utc, initial inventory, hostname, resolved
  config sha256).
* ``bot_config_resolved.json`` — full resolved ``Settings`` dump
  (sanitised: secrets redacted by name pattern).
* ``bot_config_envfile.txt`` — verbatim copy of the env file at
  startup. Lines whose KEY matches ``*_API_KEY`` / ``*_API_SECRET``
  / ``*_PASSPHRASE`` substring get value replaced with
  ``{REDACTED}``.
* ``bot_shutdown.json`` *(clean stop only)* — final state at stop
  time: session_ended_at_utc, kill_reason if any, final position,
  final equity, fill totals.

Skip-when-recorder-disabled behaviour: if the recorder pointer
file is absent (``RECORDING_ENABLED=false`` or recorder failed to
start), this module is a no-op. The bot trades fine; the session
is intentionally non-forensic-grade.

Bot restart under same recorder: if ``bot_manifest.json`` already
exists in the session dir (second bot start writing into a recorder
session the first start initialised), the second start writes to
``bot_manifest_2.json`` etc., preserving the first start's
manifest as the canonical session-start record.

See ``backtesting/docs/execution-plan.md`` §Phase F (Forensics)
for the execution plan + design rationale. Original spec drafted
at ``plans/_DONE/forensics.md`` (archived 2026-05-23 after merge).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.config import Settings
    from app.state import BotState


logger = logging.getLogger(__name__)

# Substrings that mark an env line's KEY (not value) as secret.
# Mirrors the redaction rule in ``Settings.sanitized_dict()`` which
# checks the same substrings against the aliased UPPER_SNAKE_CASE
# key. Substring (not exact) so e.g. ``BLUEFIN_API_KEY`` matches.
_SECRET_KEY_SUBSTRINGS: tuple[str, ...] = (
    "API_KEY",
    "API_SECRET",
    "PASSPHRASE",
    "PRIVATE_KEY",
    "SECRET_KEY",
)

# Atomic-write convention: write to ``{path}.tmp`` then ``os.rename``.
# On POSIX rename is atomic within the same filesystem; on Windows it
# overwrites atomically (we call os.replace via Path.replace for cross-
# platform parity). The recorder's session dir is local — same FS as
# the temp file — so this is safe.


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically via tmp + rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, data: Any) -> None:
    _atomic_write_text(
        path, json.dumps(data, indent=2, default=str, ensure_ascii=False)
    )


def read_recorder_pointer(bot_profile: str) -> Optional[Path]:
    """Return the recorder's session dir, or ``None`` if absent.

    The recorder's ``ExecStartPre`` hook (``scripts/start_recorder_
    on_colo.sh``) writes ``/tmp/dtc-mm-as-recorder-active-session-
    {bot_profile}.txt`` containing the absolute path of the session
    dir as a single line. We read it lazily — if the file is absent
    OR the path it points to doesn't exist, we return ``None`` and
    the caller treats forensic-artifact writes as a no-op.

    Profile name should match the env-file basename (e.g.
    ``prod.okx.ton.usdt.perp``).
    """
    pointer_path = Path(
        f"/tmp/dtc-mm-as-recorder-active-session-{bot_profile}.txt"
    )
    try:
        if not pointer_path.is_file():
            return None
        raw = pointer_path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        session_dir = Path(raw)
        if not session_dir.is_dir():
            logger.warning(
                "recorder_pointer_session_dir_missing pointer=%s dir=%s",
                pointer_path,
                session_dir,
            )
            return None
        return session_dir
    except Exception:
        logger.exception(
            "recorder_pointer_read_failed pointer=%s", pointer_path
        )
        return None


def _sanitize_env_file(env_file_path: Path) -> str:
    """Read the env file and redact secret-pattern lines in place.

    A line is considered a secret if its KEY (the part before the
    first ``=``) contains any substring in ``_SECRET_KEY_SUBSTRINGS``.
    Comments + blank lines + non-KEY=VALUE lines are passed through
    unmodified. Value-side preserved as-is on non-secret lines.
    """
    try:
        lines = env_file_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        logger.exception("env_file_read_failed path=%s", env_file_path)
        return f"# env file read failed at {env_file_path}\n"
    out_lines: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        # Pass through comments + blanks.
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue
        if "=" not in line:
            out_lines.append(line)
            continue
        key_part = line.split("=", 1)[0]
        key_clean = key_part.strip().upper()
        if any(s in key_clean for s in _SECRET_KEY_SUBSTRINGS):
            out_lines.append(f"{key_part}={{REDACTED}}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if lines else "")


def _next_manifest_path(session_dir: Path, base_name: str) -> Path:
    """Resolve to a non-clobbering filename.

    ``base_name`` = ``"bot_manifest.json"`` → returns
    ``bot_manifest.json`` if absent, else ``bot_manifest_2.json``,
    ``bot_manifest_3.json``, ... Used when a second bot start lands
    on a recorder session whose first start already wrote a manifest.
    The first manifest stays canonical; subsequent restarts get
    suffixed.
    """
    p = session_dir / base_name
    if not p.exists():
        return p
    stem = p.stem  # e.g. "bot_manifest"
    suffix = p.suffix  # e.g. ".json"
    n = 2
    while True:
        candidate = session_dir / f"{stem}_{n}{suffix}"
        if not candidate.exists():
            return candidate
        n += 1
        if n > 1000:
            # Defensive: shouldn't happen, but don't infinite-loop.
            raise RuntimeError(
                f"too many bot_manifest_*.json files in {session_dir}"
            )


def write_bot_manifest(
    *,
    session_dir: Path,
    state: "BotState",
    settings: "Settings",
    env_file_path: Optional[Path],
    bot_version: str,
    bot_profile: str,
) -> None:
    """Write the three startup artifacts into ``session_dir``.

    All three are written atomically via ``tmp + os.replace``. On
    any single-file failure, the others still attempt to write —
    the artifacts are independent.

    First-start writes ``bot_manifest.json`` directly. Second start
    (manifest already exists from a prior bot session under the same
    recorder) writes ``bot_manifest_2.json`` etc. The config files
    follow the same suffixing for consistency.

    Best-effort: every step is wrapped in try/except so a forensic
    write failure never aborts bot startup.
    """
    # Resolved Settings dump (uses Settings.sanitized_dict() which
    # already strips secrets by name pattern).
    try:
        resolved_dict = settings.sanitized_dict()
    except Exception:
        logger.exception("settings_sanitized_dict_failed")
        resolved_dict = {}
    # Sha256 of the resolved dict — lets future tooling say "this
    # session used the same config as that one" by hash equality.
    try:
        resolved_blob = json.dumps(
            resolved_dict, sort_keys=True, default=str
        ).encode("utf-8")
        resolved_sha256 = hashlib.sha256(resolved_blob).hexdigest()
    except Exception:
        resolved_sha256 = ""

    # Initial inventory snapshot. ``state.position`` may be None at
    # very-early startup if BotState construction lands here before
    # the first reconcile; the caller is expected to invoke this
    # AFTER reconcile so the values are meaningful.
    try:
        pos = state.position
        initial_position_qty = float(getattr(pos, "position_qty", 0.0))
        initial_position_notional = float(
            getattr(pos, "position_notional", 0.0) or 0.0
        )
        initial_avg_entry_price = (
            float(getattr(pos, "avg_entry_price", 0.0) or 0.0) or None
        )
    except Exception:
        initial_position_qty = 0.0
        initial_position_notional = 0.0
        initial_avg_entry_price = None
    try:
        acct = state.account
        initial_equity_usd = (
            float(getattr(acct, "equity_usd", 0.0) or 0.0)
            if acct is not None
            else None
        )
    except Exception:
        initial_equity_usd = None

    manifest = {
        "schema_version": 1,
        "bot_version": bot_version,
        "bot_profile": bot_profile,
        "session_id": getattr(state, "session_id", None),
        "session_started_at_utc": (
            state.session_started_at_utc.isoformat()
            if getattr(state, "session_started_at_utc", None) is not None
            else _now_utc_iso()
        ),
        "manifest_written_at_utc": _now_utc_iso(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "symbol": getattr(settings, "symbol", None),
        "initial_position_qty": initial_position_qty,
        "initial_position_notional_usd": initial_position_notional,
        "initial_avg_entry_price": initial_avg_entry_price,
        "initial_equity_usd": initial_equity_usd,
        "resolved_config_sha256": resolved_sha256,
        "env_file_path": (
            str(env_file_path) if env_file_path is not None else None
        ),
    }

    manifest_path = _next_manifest_path(session_dir, "bot_manifest.json")
    config_resolved_path = (
        manifest_path.with_name(
            manifest_path.stem.replace("bot_manifest", "bot_config_resolved")
            + ".json"
        )
    )
    config_envfile_path = manifest_path.with_name(
        manifest_path.stem.replace("bot_manifest", "bot_config_envfile")
        + ".txt"
    )

    try:
        _atomic_write_json(manifest_path, manifest)
    except Exception:
        logger.exception(
            "bot_manifest_write_failed path=%s", manifest_path
        )
    try:
        _atomic_write_json(config_resolved_path, resolved_dict)
    except Exception:
        logger.exception(
            "bot_config_resolved_write_failed path=%s", config_resolved_path
        )
    if env_file_path is not None:
        try:
            envfile_sanitized = _sanitize_env_file(env_file_path)
            _atomic_write_text(config_envfile_path, envfile_sanitized)
        except Exception:
            logger.exception(
                "bot_config_envfile_write_failed path=%s",
                config_envfile_path,
            )
    logger.info(
        "wrote_bot_session_manifest session_dir=%s manifest=%s "
        "resolved_sha256=%s initial_pos_qty=%s",
        session_dir,
        manifest_path.name,
        resolved_sha256[:12] if resolved_sha256 else "?",
        initial_position_qty,
    )


def write_bot_shutdown(
    *,
    session_dir: Path,
    state: "BotState",
    bot_version: str,
    kill_reason: Optional[str] = None,
) -> None:
    """Write ``bot_shutdown.json`` on clean stop.

    Idempotent: if the file already exists (paired ``stop()`` +
    ``kill()`` paths can both call this), the first write wins. The
    second call no-ops. The first write captures session-end state
    closest to the actual halt; later mutations on ``state`` from
    background threads aren't relevant.
    """
    shutdown_path = session_dir / "bot_shutdown.json"
    if shutdown_path.exists():
        logger.info("bot_shutdown_already_written path=%s", shutdown_path)
        return

    try:
        pos = state.position
        final_position_qty = float(getattr(pos, "position_qty", 0.0))
        final_position_notional = float(
            getattr(pos, "position_notional", 0.0) or 0.0
        )
    except Exception:
        final_position_qty = 0.0
        final_position_notional = 0.0
    try:
        acct = state.account
        final_equity_usd = (
            float(getattr(acct, "equity_usd", 0.0) or 0.0)
            if acct is not None
            else None
        )
    except Exception:
        final_equity_usd = None
    try:
        pnl = state.pnl
        realized_pnl_usd = (
            float(getattr(pnl, "realized_pnl_usd", 0.0) or 0.0)
            if pnl is not None
            else 0.0
        )
        unrealized_pnl_usd = (
            float(getattr(pnl, "unrealized_pnl_usd", 0.0) or 0.0)
            if pnl is not None
            else 0.0
        )
        fees_cumulative_usd = (
            float(getattr(pnl, "fees_usd", 0.0) or 0.0)
            if pnl is not None
            else 0.0
        )
    except Exception:
        realized_pnl_usd = 0.0
        unrealized_pnl_usd = 0.0
        fees_cumulative_usd = 0.0
    fills_total = int(getattr(state, "session_fill_count", 0) or 0)
    fills_by_side = getattr(state, "session_fill_count_by_side", None)
    if isinstance(fills_by_side, dict):
        fills_buy = int(fills_by_side.get("BUY", 0) or 0)
        fills_sell = int(fills_by_side.get("SELL", 0) or 0)
    else:
        fills_buy = 0
        fills_sell = 0

    payload = {
        "schema_version": 1,
        "bot_version": bot_version,
        "session_ended_at_utc": _now_utc_iso(),
        "kill_reason": kill_reason,
        "killed": bool(getattr(state, "killed", False)),
        "final_position_qty": final_position_qty,
        "final_position_notional_usd": final_position_notional,
        "final_equity_usd": final_equity_usd,
        "realized_pnl_usd": realized_pnl_usd,
        "unrealized_pnl_usd": unrealized_pnl_usd,
        "fees_cumulative_usd": fees_cumulative_usd,
        "fills_total": fills_total,
        "fills_buy": fills_buy,
        "fills_sell": fills_sell,
    }
    try:
        _atomic_write_json(shutdown_path, payload)
    except Exception:
        logger.exception(
            "bot_shutdown_write_failed path=%s", shutdown_path
        )
        return
    logger.info(
        "wrote_bot_shutdown_manifest path=%s kill_reason=%s "
        "final_pos_qty=%s fills=%s",
        shutdown_path,
        kill_reason,
        final_position_qty,
        fills_total,
    )


__all__ = [
    "read_recorder_pointer",
    "write_bot_manifest",
    "write_bot_shutdown",
]
