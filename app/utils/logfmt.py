from __future__ import annotations

import logging
from typing import Any


def log_extra(logger: logging.Logger, level: int, msg: str, data: dict[str, Any]) -> None:
    """Emit a structured log line when using StructuredFormatter (reads record.extra_data)."""
    logger.log(level, msg, extra={"extra_data": data})
