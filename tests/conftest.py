"""Per-test safety rails.

Enforces one hard rule: **the repository-root ``tmp/`` and
``snapshots/`` directories are for user artifacts only (stats
snapshots, DB downloads, the live trading SQLite, ad-hoc exports).
Tests MUST NEVER write, delete, or modify anything in either.**

This rule exists because both directories hold session data the
operator needs to analyse bot behaviour. A glob like
``rm -rf tmp/snap_*`` inside a cleanup hook would silently destroy
real snapshots, and such files are NOT recoverable from Windows
Recycle Bin (Git Bash ``rm`` skips it). See the AGENTS.md note at
the repo root.

History: ``tmp/`` was the original sole artifact directory. On
2026-05-08 snapshots were split into ``snapshots/`` for clearer
semantics (``tmp/`` continues to hold the bot's runtime
``trading.db`` and ad-hoc scratch). This guard now protects both.

If a test needs scratch space, use ``tempfile.gettempdir()`` —
cross-platform, system-managed, cleaned automatically on OS
cycles. Do not use ``tmp/`` or ``snapshots/`` at the repo root
for anything test-related.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_USER_ARTIFACT_DIRS: tuple[Path, ...] = (
    _REPO_ROOT / "tmp",
    _REPO_ROOT / "snapshots",
)

# Names that match the bot's external ``ops snapshot`` output pattern.
# When the operator runs ``ops snapshot`` while pytest is running, a new
# directory like ``v1.4.31-260517-225528-prod.okx.ton.usdt.perp`` lands in
# ``snapshots/`` mid-test, and the artifact-diff fixture would blame
# whichever test was in ``yield`` at that microsecond — a false-positive
# race (the test didn't write anything). The pattern below matches the
# operator-snapshot naming scheme exactly so the race-induced adds are
# ignored. Tests that genuinely write into ``snapshots/`` with any other
# name still trip the assertion. False-negative risk is bounded to "a
# test created a path named *exactly* like an ops-snapshot dir," which
# would itself be a bug worth surfacing via the assertion message.
#
# Pattern: ``v<MAJOR>.<MINOR>.<PATCH>-<YYMMDD>-<HHMMSS>-<profile>``
# Real example: ``v1.4.31-260517-225528-prod.okx.ton.usdt.perp``
_OPS_SNAPSHOT_NAME_RE = re.compile(
    r"^v\d+\.\d+\.\d+-\d{6}-\d{6}-[a-zA-Z0-9._-]+$"
)


def _listing(path: Path) -> frozenset[str]:
    """Set of direct-child names under ``path``. Empty set if missing.

    We only check the TOP-LEVEL listing (not recursively) because:
      1. That's where any test-induced corruption would first appear
         (a new folder, a deleted snap_<ts>/ directory, etc.).
      2. Recursing into real user snapshots would be slow and noisy.
      3. If a test somehow nested INTO an existing snap_<ts>/ folder,
         the top-level listing still flags it as modified (via mtime)
         in most cases, and in any case the user's snapshot integrity
         has already been violated.
    """
    if not path.exists():
        return frozenset()
    return frozenset(p.name for p in path.iterdir())


@pytest.fixture(autouse=True)
def _user_artifact_dirs_are_untouchable(request) -> "pytest.FixtureDef":
    """Fail any test that adds or removes entries at the top of
    ``tmp/`` or ``snapshots/``.

    Triggering condition: the set of direct-child names under either
    ``<repo>/tmp/`` or ``<repo>/snapshots/`` differs between test
    start and test end. That means a test either created a new path
    or deleted one. Both are violations of the "tests must never
    touch user artifacts" rule.

    To fix a failing test: move its scratch space to
    ``tempfile.gettempdir()``. Search the codebase for existing
    examples (every other test does this already). If you genuinely
    need a scratch folder shared across tests or observable from the
    repo, create one named anything OTHER than ``tmp/`` or
    ``snapshots/`` (e.g. ``_test_scratch/``) and delete it at
    session end.
    """
    before = {p: _listing(p) for p in _USER_ARTIFACT_DIRS}
    yield
    violations: list[str] = []
    for p in _USER_ARTIFACT_DIRS:
        after = _listing(p)
        if before[p] == after:
            continue
        added_raw = sorted(after - before[p])
        removed_raw = sorted(before[p] - after)
        # Strip operator-snapshot pattern matches from BOTH lists —
        # they're race-condition artifacts (operator ran ``ops
        # snapshot`` mid-pytest), not test violations. Both added
        # (snapshot appeared during test) and removed (snapshot was
        # cleaned up during test — rare but possible if the operator
        # ran ``ops clean``) are filtered to be symmetric.
        added = [n for n in added_raw if not _OPS_SNAPSHOT_NAME_RE.match(n)]
        removed = [n for n in removed_raw if not _OPS_SNAPSHOT_NAME_RE.match(n)]
        if not added and not removed:
            # Whatever changed was an operator-snapshot dir, race-induced.
            # Skip silently — no violation.
            continue
        # [DEBUG-CULPRIT-HUNT] Log every diff, even before raising, so we
        # catch culprits that touch the dir but happen to be blamed on a
        # different test (e.g. via module-fixture teardown order).
        import sys
        print(
            f"\n[ARTIFACT-DIFF] test={request.node.nodeid} dir={p.name}/ "
            f"added={added} removed={removed}",
            file=sys.stderr,
            flush=True,
        )
        parts: list[str] = []
        if added:
            parts.append(f"CREATED: {added}")
        if removed:
            parts.append(f"DELETED: {removed}")
        violations.append(f"{p.name}/ — {' | '.join(parts)}")
    if violations:
        raise AssertionError(
            "This test modified protected user-artifact directories. "
            "Both tmp/ and snapshots/ are for user artifacts ONLY "
            "(live trading.db, stats snapshots, DB downloads, etc.) "
            "and tests must never write, delete, or modify anything "
            f"inside them. Changes detected: {' ; '.join(violations)}. "
            "Fix: use tempfile.gettempdir() for scratch space (see "
            "any other test's ``_db()`` helper)."
        )


@pytest.fixture(autouse=True)
def _reset_module_clock_between_tests():
    """Reset ``app.clock._module_clock`` to a fresh ``SystemClock`` at
    the start of each test.

    Phase 1c (v1.4.231) shipped a module-level clock proxy in
    ``app/clock.py`` so free-function callers (``utc_now()`` etc.)
    can read time through a single configurable source. ``Bot.__init__``
    installs ``self._clock`` as the active proxy — so a test that
    constructs a ``Bot`` with a ``ReplayClock`` leaks that clock to
    EVERY subsequent test (the proxy stays set until something
    overwrites it).

    The Phase 4c backtesting tests (v1.4.237) construct Bots with
    ``ReplayClock`` instances anchored to 2023. Without this reset,
    later tests that call ``utc_now()`` see 2023 instead of real
    time, breaking time-comparison logic in session-resume,
    startup-cleanup, and market-data-recovery paths. (CI daemon
    flagged this 2026-05-22 with 4 failures across
    ``test_session_resume``, ``test_startup_cleanup``,
    ``test_startup_inventory_policy``.)

    Reset happens BEFORE the test runs so every test starts from a
    known state regardless of what the previous test did. Tests
    that need a different clock install one explicitly (e.g. via
    ``Bot.__init__(clock=...)``).
    """
    from app.clock import SystemClock, set_module_clock
    set_module_clock(SystemClock())
    yield
    # Restore once more on teardown as defence in depth — covers
    # the case where a test installs a non-SystemClock but doesn't
    # construct a fresh Bot to overwrite it.
    set_module_clock(SystemClock())
