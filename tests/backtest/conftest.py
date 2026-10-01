"""Shared fixture helpers for the backtest regression suite — Phase 4.

The discovery logic centralises the "where do fixtures live?"
question so the test files don't duplicate it. If neither directory
exists or both are empty, ``list_fixtures()`` returns ``[]`` — which
the parametrised tests use to skip cleanly.
"""

from __future__ import annotations

from pathlib import Path

# Project root: tests/backtest/conftest.py → up 2 levels.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS_DIR = PROJECT_ROOT / "backtesting" / "data" / "sessions"
# Curated fixtures dir (Phase 4B+; may not exist yet).
FIXTURES_DIR = PROJECT_ROOT / "backtesting" / "data" / "fixtures"
BASELINES_DIR = Path(__file__).resolve().parent / "baselines"


def list_fixtures() -> list[Path]:
    """Return every fixture-shaped directory we know about.

    A "fixture-shaped" dir contains ``manifest.json``. We check the
    curated ``fixtures/`` dir first (preferred), then fall back to
    raw recordings under ``sessions/``. Result is sorted by directory
    name for deterministic test ordering.
    """
    candidates: list[Path] = []
    for parent in (FIXTURES_DIR, SESSIONS_DIR):
        if not parent.exists():
            continue
        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "manifest.json").exists():
                continue
            candidates.append(child)
    return candidates


def has_baseline(fixture: Path) -> bool:
    return (BASELINES_DIR / f"{fixture.name}.json").exists()


def baseline_path(fixture: Path) -> Path:
    return BASELINES_DIR / f"{fixture.name}.json"
