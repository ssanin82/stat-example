"""
Synchronous retry with exponential backoff for Hyperliquid HTTP calls (via SDK).

Keeps logic explicit; no async. Used only from ``hyperliquid_client``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar

from app.config import Settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """1.4.0 cancel-prio Phase 0c: per-op retry override.

    When passed to :func:`exchange_call_with_retry` via the ``policy``
    kwarg, these values supplant the shared
    ``settings.exchange_retry_*`` defaults for the duration of one
    call. Used by the OKX client to give cancels a more aggressive
    retry profile (more attempts, shorter base, lower cap) than
    places. The ``rate_extra`` delay on rate-limit hits is still
    sourced from settings — that's a venue-side budget concern, not a
    per-op concern.

    All fields validated by the caller; we clamp `max_attempts >= 1`
    and `base_seconds >= 0.01` inside the function for safety.
    """

    max_attempts: int
    base_seconds: float
    cap_seconds: float

try:
    from hyperliquid.utils.error import ClientError, ServerError
except ImportError:  # pragma: no cover

    class ClientError(Exception):  # type: ignore[no-redef]
        """Minimal stand-in when the SDK is not installed (``isinstance`` + attrs only)."""

        def __init__(
            self,
            status_code: int,
            error_code: str = "",
            error_message: str | None = None,
            header: Any = None,
            error_data: Any = None,
        ) -> None:
            super().__init__(error_message or "")
            self.status_code = status_code
            self.error_code = error_code
            self.error_message = error_message
            self.header = header
            self.error_data = error_data

    class ServerError(Exception):  # type: ignore[no-redef]
        """Minimal stand-in when the SDK is not installed (``isinstance`` + attrs only)."""

        def __init__(self, status_code: int, message: str = "") -> None:
            super().__init__(message)
            self.status_code = status_code
            self.message = message

try:
    import requests

    _REQUESTS_EXC = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    )
except ImportError:  # pragma: no cover
    _REQUESTS_EXC = ()


def _is_rate_limited(exc: BaseException) -> bool:
    if isinstance(exc, ClientError):
        if exc.status_code == 429:
            return True
        msg = f"{exc.error_message or ''} {exc.error_data or ''}".lower()
        if "rate" in msg and "limit" in msg:
            return True
        if "too many" in msg:
            return True
    return False


def _is_transient_exchange_error(exc: BaseException) -> bool:
    if isinstance(exc, ServerError):
        return True
    if isinstance(exc, ClientError):
        if exc.status_code == 429:
            return True
        if exc.status_code >= 500:
            return True
        msg = f"{exc.error_message or ''} {exc.error_data or ''}".lower()
        if "timeout" in msg:
            return True
        if "rate" in msg:
            return True
        return False
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if _REQUESTS_EXC and isinstance(exc, _REQUESTS_EXC):
        return True
    return False


def exchange_call_with_retry(
    operation: str,
    fn: Callable[[], T],
    settings: Settings,
    *,
    on_retry: Callable[[str, bool], None] | None = None,
    policy: Optional[RetryPolicy] = None,
) -> T:
    """
    Run ``fn`` with retries on transient / rate-limit style failures.
    Logs every retry attempt. Re-raises the last exception if all attempts fail.

    ``policy`` (1.4.0 cancel-prio Phase 0c): when provided, override
    ``max_attempts`` / ``base_seconds`` / ``cap_seconds`` from the
    shared ``settings.exchange_retry_*`` defaults. The rate-limit
    extra delay still comes from settings (per-venue budget). The OKX
    client uses this for cancel ops to retry sooner (50/100/200/400 ms
    vs the default 350/700/1400/2800 ms) and bound the worst case
    by 1.5 s read timeout rather than 8 s.
    """
    if policy is not None:
        max_tries = max(1, int(policy.max_attempts))
        base = max(0.01, float(policy.base_seconds))
        cap = max(base, float(policy.cap_seconds))
    else:
        max_tries = max(1, settings.exchange_retry_max_attempts)
        base = max(0.01, settings.exchange_retry_base_seconds)
        cap = max(base, settings.exchange_retry_max_backoff_seconds)
    rate_extra = max(0.0, settings.exchange_rate_limit_extra_delay_seconds)

    last_exc: BaseException | None = None
    for attempt in range(max_tries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt >= max_tries - 1 or not _is_transient_exchange_error(e):
                raise
            extra = rate_extra if _is_rate_limited(e) else 0.0
            is_rl = _is_rate_limited(e)
            if on_retry is not None:
                try:
                    on_retry(operation, is_rl)
                except Exception:
                    logger.debug("exchange_retry on_retry callback failed", exc_info=True)
            sleep_s = min(cap, base * (2**attempt)) + extra
            logger.warning(
                "exchange_retry operation=%s attempt=%s max_tries=%s sleep_s=%.3f "
                "rate_limited=%s err_type=%s err=%s",
                operation,
                attempt + 1,
                max_tries,
                sleep_s,
                is_rl,
                type(e).__name__,
                e,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc
