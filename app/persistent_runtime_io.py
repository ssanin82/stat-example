"""Best-effort load/save of operator metrics (path from Settings); logs failures, never raises."""

from __future__ import annotations

import logging

from app.config import Settings
from app.persistent_runtime_state import (
    load_persistent_runtime_state,
    save_persistent_runtime_state,
)
from app.state import BotState

logger = logging.getLogger(__name__)


def try_load_persistent_runtime_state(settings: Settings, state: BotState) -> None:
    path = (settings.persistent_runtime_state_path or "").strip()
    if not path:
        return
    loaded = load_persistent_runtime_state(path)
    if loaded is None:
        logger.info("persistent_runtime_state load skipped or failed path=%s", path)
        return
    state.apply_persistent_runtime_state(loaded)
    logger.info(
        "persistent_runtime_state loaded path=%s day_anchor_utc=%s",
        path,
        state.operator_day_anchor_utc.isoformat(),
    )


def try_save_persistent_runtime_state(settings: Settings, state: BotState) -> None:
    path = (settings.persistent_runtime_state_path or "").strip()
    if not path:
        return
    try:
        prs = state.build_persistent_runtime_state()
        save_persistent_runtime_state(path, prs)
    except Exception:
        logger.exception("persistent_runtime_state save failed path=%s", path)
