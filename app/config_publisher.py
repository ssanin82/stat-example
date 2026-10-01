"""One-shot S3 publisher for the bot's effective configuration.

Writes a single JSON object describing the bot's live configuration —
both the raw ``APP_ENV_FILE`` contents (so the operator dashboard can
display the original ``KEY=value`` lines with their preceding ``#``
comment blocks as tooltips) and the resolved Pydantic ``Settings``
view (so the dashboard can show what the bot *actually* sees after
defaults / coercions / validators).

S3 path: ``s3://<logs_bucket>/config/<profile>.json``.

Cadence: ONE shot at bot startup. The config is static for the
lifetime of a deploy — there's no in-flight reload — so a periodic
publisher would be wasted PUTs. The Bot Stats panel's CONFIG tab
fetches once at page-load and doesn't poll.

Trading impact: zero. Best-effort like the other S3 publishers; any
failure (boto3 missing, IAM denial, file unreadable) is logged at
WARNING and the bot continues trading.

Secret handling: BOTH the raw env file content AND the resolved
settings dump are sanitised. The raw-file path scans each line for
``KEY=`` matching ``_is_sensitive_key_name`` and replaces the value
with ``***``; the resolved-settings dict reuses ``Settings.sanitized_dict()``.
This keeps ``HL_SECRET_KEY=…`` / ``BINANCE_API_KEY=…`` / etc. out of
the dashboard payload that lands in S3.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from app import __version__
from app.config import Settings, _is_sensitive_key_name
from app.state import BotState

from app import clock as _clock

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return _clock.now_utc().isoformat()


def _redact_env_file_text(raw: str) -> str:
    """Return ``raw`` with secret-line values masked.

    A "secret line" is one whose ``KEY=`` matches ``_is_sensitive_key_name``
    (the same predicate ``Settings.sanitized_dict`` uses). Comment
    lines, blank lines, and non-secret assignments pass through
    unchanged so the dashboard can still display the comments and
    structure.
    """
    out_lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        # Comments and blank lines pass through.
        if not stripped or stripped.startswith("#"):
            out_lines.append(line)
            continue
        # Find the first '=' for KEY=value parsing. Anything before
        # it is the key (potentially with leading whitespace).
        eq = stripped.find("=")
        if eq <= 0:
            out_lines.append(line)
            continue
        key = stripped[:eq].strip()
        if _is_sensitive_key_name(key.upper()):
            indent = line[: len(line) - len(stripped)]
            out_lines.append(f"{indent}{key}=***")
        else:
            out_lines.append(line)
    return "\n".join(out_lines)


class ConfigPublisher:
    """One-shot S3 writer for the bot's effective config.

    Unlike ``HeartbeatPublisher`` / ``LiveStatsPublisher`` /
    ``EquityHistoryPublisher`` (all daemon-thread loops), this publisher
    runs the upload exactly once on ``start()`` in a short-lived
    background thread, then the thread exits. Re-publishing on every
    poll would only churn S3 — the env file is frozen for the deploy.
    """

    def __init__(
        self,
        settings: Settings,
        state: BotState,
        bucket: str,
        profile_name: str,
        env_file_path: str,
    ) -> None:
        self._settings = settings
        self._state = state
        self._bucket = bucket.strip()
        self._profile_name = (profile_name or "unknown").strip() or "unknown"
        self._key = f"config/{self._profile_name}.json"
        self._env_file_path = env_file_path
        self._thread: Optional[threading.Thread] = None
        self._client: Any = None

    @classmethod
    def maybe_create(
        cls,
        settings: Settings,
        state: BotState,
        profile_name: str,
        env_file_path: str,
    ) -> Optional["ConfigPublisher"]:
        if not bool(settings.live_stats_enabled):
            logger.info(
                "config_publish_disabled reason=LIVE_STATS_ENABLED=false"
            )
            return None
        bucket = (settings.logs_bucket or "").strip()
        if not bucket:
            logger.info(
                "config_publish_disabled reason=no_logs_bucket "
                "set_LOGS_BUCKET_to_enable"
            )
            return None
        return cls(settings, state, bucket, profile_name, env_file_path)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import boto3  # type: ignore[import-untyped]

            self._client = boto3.client("s3")
        except Exception as e:
            logger.warning(
                "config_publish_disabled reason=boto3_init_failed err=%s",
                str(e)[:200],
            )
            return
        self._thread = threading.Thread(
            target=self._safe_publish_once,
            name="config_publish",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "config_publisher_started bucket=%s key=%s",
            self._bucket,
            self._key,
        )

    def _safe_publish_once(self) -> None:
        try:
            self._publish_once()
        except Exception as e:
            logger.warning(
                "config_publish_failed err=%s",
                str(e)[:200],
            )

    def _publish_once(self) -> None:
        if self._client is None:
            return
        body = self._build_payload()
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key,
            Body=json.dumps(body, indent=None, separators=(",", ":")).encode(
                "utf-8"
            ),
            ContentType="application/json",
            CacheControl="no-store, max-age=0",
        )

    def _build_payload(self) -> dict[str, Any]:
        # Read the raw env file (best-effort; missing file → empty
        # string and the dashboard falls back to the resolved view).
        env_file_raw = ""
        env_file_error: Optional[str] = None
        if self._env_file_path:
            try:
                with open(self._env_file_path, "r", encoding="utf-8") as fh:
                    env_file_raw = fh.read()
            except Exception as e:
                env_file_error = str(e)[:200]
                logger.warning(
                    "config_env_file_read_failed path=%s err=%s",
                    self._env_file_path,
                    env_file_error,
                )

        env_file_redacted = _redact_env_file_text(env_file_raw)
        resolved = self._settings.sanitized_dict()

        with self._state._lock:
            session_id = self._state.session_id
            session_started_at_utc = self._state.session_started_at_utc
            symbol = self._state.symbol

        return {
            "schema_version": 1,
            "profile": self._profile_name,
            "symbol": symbol,
            "version": __version__,
            "captured_at_utc": _now_iso(),
            "session_id": session_id,
            "session_started_at_utc": session_started_at_utc.isoformat(),
            "env_file_path": self._env_file_path,
            "env_file_raw": env_file_redacted,
            "env_file_read_error": env_file_error,
            "resolved_settings": resolved,
        }
