"""Recording-status helper for the bot's /status endpoint.

v1.5.20 invariant (operator-stated): the recorder's lifetime is a
STRICT SUPERSET of the bot's. This is enforced structurally by the
systemd unit (``/etc/systemd/system/dtc-bot.service``):

* ``ExecStartPre=/opt/dtc/dtc-mm-as/scripts/colo_systemd_recorder_pre_start.sh``
  launches the recorder BEFORE the bot's ``ExecStart``. Non-zero
  exit fails the bot startup -- fail-CLOSED by design.
* ``ExecStopPost=/opt/dtc/dtc-mm-as/scripts/colo_systemd_recorder_post_stop.sh``
  stops the recorder AFTER the bot exits, so the bot's final
  cancel-all + WS-close events are captured.

Any code path that brings the bot up (``systemctl start``,
``systemctl restart``, ``Restart=on-failure`` auto-recovery, the
``ops.ps1 ... start`` orchestrator, dashboard buttons) goes through
the same unit and therefore the same recorder-launch step. Pre-v1.5.20
the recorder launch lived only in the laptop orchestrator
(``colo_start_bot.ps1``); a ``sudo systemctl restart dtc-bot``
on colo would bring up the bot WITHOUT the recorder. v1.5.20
moves the launch into the systemd unit itself so the invariant
holds regardless of the start path.

This module reads the pointer file at
``/tmp/dtc-mm-as-recorder-active-session-<bot-profile>.txt`` and
returns a small status dict for the dashboard. The dict shape is
fixed (always-render contract) -- the dashboard's
``RecordingIndicator`` reads:

* ``enabled=False`` -> "rec off" (operator opted out via config)
* ``enabled=True, active=True`` -> "REC <bytes>" (normal operation)
* ``enabled=True, active=False`` -> "REC ERROR" (defense-in-depth;
  per the invariant this state should NEVER fire during normal
  operation -- if it does, the systemd unit's ExecStartPre wasn't
  installed, OR the recorder died mid-session and didn't restart,
  OR the pointer file got deleted out from under a live recorder)

Profile resolution: same convention as ``app/main.py`` uses for
the heartbeat publisher -- strip ``APP_ENV_FILE``'s basename.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _resolve_bot_profile() -> str:
    """Derive the bot's profile name from ``APP_ENV_FILE``.

    Matches ``app/main.py:96-100``'s convention so the pointer-file
    lookup aligns with the orchestration scripts' filename pattern.
    Returns ``""`` when no env file is in scope (local dev).
    """
    env_path = os.environ.get("APP_ENV_FILE", "") or ""
    if not env_path:
        return ""
    return os.path.splitext(os.path.basename(env_path))[0] or ""


def _dir_size_bytes(path: Path) -> tuple[int, int]:
    """Recursive size of ``path``. Returns ``(total_bytes, file_count)``.

    Best-effort: any ``OSError`` during the walk is logged and the
    partial total returned. Status endpoints must NEVER raise — the
    operator's dashboard polls every 2s and a single failure cascading
    out would make the whole status page red.
    """
    total = 0
    count = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
                    count += 1
            except OSError:
                continue
    except OSError as e:
        logger.debug("recording_status_dir_walk_failed path=%s err=%s", path, e)
    return total, count


def recording_status(settings: Any) -> dict[str, Any]:
    """Build the ``recording`` sub-dict for the /status response.

    Shape (always present, fields vary by state):
        {
            "enabled": bool,            # RECORDING_ENABLED setting
            "active": bool,             # pointer file found + dir exists
            "session_name": str|None,   # basename of session dir
            "session_path": str|None,   # absolute path on colo
            "bytes": int,               # 0 when not active
            "files": int,               # 0 when not active
            "profile_resolved": str,    # bot profile used for lookup
        }

    Never raises — every failure path returns a sentinel shape that
    the dashboard can render as "off" / "unknown size".
    """
    enabled = bool(getattr(settings, "recording_enabled", False))
    bot_profile = _resolve_bot_profile()
    out: dict[str, Any] = {
        "enabled": enabled,
        "active": False,
        "session_name": None,
        "session_path": None,
        "bytes": 0,
        "files": 0,
        "profile_resolved": bot_profile,
    }
    if not enabled:
        return out
    if not bot_profile:
        # Recording wants to be on but we can't resolve the profile
        # (e.g. running locally without APP_ENV_FILE). Surface as
        # enabled-but-inactive so the dashboard can still show the
        # "recording configured" badge without claiming a session.
        return out

    pointer_path = Path(f"/tmp/dtc-mm-as-recorder-active-session-{bot_profile}.txt")
    if not pointer_path.exists():
        return out
    try:
        session_path_str = pointer_path.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.debug("recording_status_pointer_read_failed err=%s", e)
        return out
    if not session_path_str:
        return out
    session_path = Path(session_path_str)
    if not session_path.exists() or not session_path.is_dir():
        # Stale pointer (session moved/deleted) — recording isn't
        # actually active. Don't lie to the dashboard.
        return out

    bytes_total, files_total = _dir_size_bytes(session_path)
    out.update({
        "active": True,
        "session_name": session_path.name,
        "session_path": str(session_path),
        "bytes": bytes_total,
        "files": files_total,
    })
    return out


__all__ = ["recording_status"]
