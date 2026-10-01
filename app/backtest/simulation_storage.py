"""Storage primitives for Phase F7 simulate-tab variants (v1.5.93).

A "variant" is a named, non-destructive what-if config tied to a
specific recorded session. The simulator (Phase 3 replay driver,
shipped) consumes a variant's materialised full config + the source
session's market-data stream, and writes ``bot_db``-shape output
files under ``<session>/<variant>/results/``.

On-disk layout::

    backtesting/simulations/
      <session_id>/               ← matches the recording dir name
        <variant_slug>/
          manifest.json           ← name, created_at, bot_version, status
          config_delta.env        ← rows that differ from the session's
                                    recorded bot_config_envfile.txt
          config_full.env         ← delta layered on top of recorded
                                    config; rewritten on every save +
                                    before each simulator run
          simulation.lock         ← present while running
          stdout.log              ← simulator's captured stdout
          stderr.log              ← simulator's captured stderr
          results/                ← bot_db-shape outputs
            orders.jsonl.gz
            fills.jsonl.gz
            ...

This module is the canonical authority for that layout: every other
caller (Phase F7 API endpoints, ``scripts/backtest/run_variant.py``)
resolves variant paths through ``variant_paths()`` and validates
variant names through ``slugify_variant_name()``.

Path-traversal guard: every public function that accepts a
``variant_name`` validates it via ``slugify_variant_name`` first.
``session_id`` is similarly bounded — values containing ``..`` or
``/`` or ``\\`` are rejected. Same posture as the frontend's
``resolveUnderDataRoot``.

Phase F7 explicitly NEVER writes outside
``backtesting/simulations/``. The "Generate profile…" UI action
writes to ``config/profiles_pending/`` but that's a separate
operator-initiated path with its own confirmation dialog (F7.10) —
not part of this storage module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}[a-z0-9]$|^[a-z0-9]$")
_SLUG_MAX_LEN = 64

# session_id is the recording's directory name. Recorder uses
# ``<bot_version>-<date>-<hhmmss>-<profile>`` which is ASCII alnum +
# ``.`` + ``-``. Accept that shape; reject anything that could escape.
_SESSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def slugify_variant_name(raw: str) -> str:
    """Coerce a user-typed variant name into the canonical slug form.

    Lowercases, replaces spaces and ``.`` with ``_``, drops every
    character that isn't ``[a-z0-9_-]``, collapses runs of ``_``/``-``,
    and trims leading/trailing separators. Raises ``ValueError`` when
    the result is empty or exceeds ``_SLUG_MAX_LEN``.

    Note: this is BOTH the validator and the normaliser — callers
    that just want to *check* an already-slugged name can use
    ``is_valid_slug()`` for a yes/no answer. The motivation for
    auto-slugifying instead of rejecting is operator UX: typing
    "Microprice gate OFF" in the New Variant dialog and getting
    ``microprice_gate_off`` is friendlier than a validation error.
    """
    if not isinstance(raw, str):
        raise ValueError(f"variant name must be a string, got {type(raw)}")
    s = raw.strip().lower()
    # Collapse internal whitespace + dots into underscores.
    s = re.sub(r"[\s.]+", "_", s)
    # Drop anything outside the slug alphabet.
    s = re.sub(r"[^a-z0-9_-]", "", s)
    # Collapse runs of separators.
    s = re.sub(r"[_-]{2,}", "_", s)
    # Trim leading/trailing separators.
    s = s.strip("_-")
    if not s:
        raise ValueError(
            f"variant name {raw!r} slugified to empty string; "
            "use letters, digits, '_' or '-'"
        )
    if len(s) > _SLUG_MAX_LEN:
        raise ValueError(
            f"variant name {raw!r} slugified to {s!r}, length "
            f"{len(s)} > max {_SLUG_MAX_LEN}"
        )
    return s


def is_valid_slug(s: str) -> bool:
    """Return True when ``s`` is already in canonical slug form."""
    return isinstance(s, str) and bool(_SLUG_RE.match(s)) and len(s) <= _SLUG_MAX_LEN


def _validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str):
        raise ValueError(f"session_id must be a string, got {type(session_id)}")
    if not session_id or not _SESSION_RE.match(session_id):
        raise ValueError(
            f"session_id {session_id!r} invalid; expected the recording "
            "directory name (alphanumerics + . _ -)"
        )
    return session_id


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _repo_root_default() -> Path:
    """Resolve the repo root from this file's location.

    Layout::

        <repo>/app/backtest/simulation_storage.py     ← this file
        <repo>/backtesting/simulations/...            ← target
    """
    return Path(__file__).resolve().parent.parent.parent


def simulations_root(*, repo_root: Optional[Path] = None) -> Path:
    """Top-level ``backtesting/simulations/`` directory."""
    base = repo_root if repo_root is not None else _repo_root_default()
    return Path(base) / "backtesting" / "simulations"


def session_simulations_dir(
    session_id: str, *, repo_root: Optional[Path] = None
) -> Path:
    """Per-recording variant directory:
    ``backtesting/simulations/<session_id>/``."""
    _validate_session_id(session_id)
    return simulations_root(repo_root=repo_root) / session_id


def sessions_root(*, repo_root: Optional[Path] = None) -> Path:
    """Top-level ``backtesting/data/sessions/`` directory (recorder
    output root). Mirrors the frontend's ``sessionsRoot()`` in
    ``lib/simulation_storage.ts``."""
    base = repo_root if repo_root is not None else _repo_root_default()
    return Path(base) / "backtesting" / "data" / "sessions"


def session_recording_dir(
    session_id: str, *, repo_root: Optional[Path] = None
) -> Path:
    """Per-recording directory:
    ``backtesting/data/sessions/<session_id>/``. The variant's base
    config (``bot_config_envfile.txt``) lives here."""
    _validate_session_id(session_id)
    return sessions_root(repo_root=repo_root) / session_id


@dataclass(frozen=True)
class VariantPaths:
    """Concrete on-disk paths for one variant. All paths are absolute
    and rooted under ``simulations_root()`` — verified by
    ``variant_paths()``'s path-traversal guard."""

    session_id: str
    variant_slug: str
    root: Path
    manifest: Path
    config_delta: Path
    config_full: Path
    lock: Path
    stdout: Path
    stderr: Path
    results: Path


def variant_paths(
    session_id: str,
    variant_slug: str,
    *,
    repo_root: Optional[Path] = None,
) -> VariantPaths:
    """Compute on-disk paths for one variant. Validates inputs.

    Does NOT create any directories — purely path arithmetic. Use
    ``create_variant`` to actually scaffold the dir on disk.
    """
    _validate_session_id(session_id)
    if not is_valid_slug(variant_slug):
        raise ValueError(
            f"variant_slug {variant_slug!r} not in canonical slug form "
            f"(call slugify_variant_name first)"
        )
    sim_root = simulations_root(repo_root=repo_root).resolve()
    root = (sim_root / session_id / variant_slug).resolve()
    # Path-traversal guard: the resolved path must be under sim_root.
    try:
        root.relative_to(sim_root)
    except ValueError:
        raise ValueError(
            f"resolved variant root {root} escapes simulations_root "
            f"{sim_root}"
        )
    return VariantPaths(
        session_id=session_id,
        variant_slug=variant_slug,
        root=root,
        manifest=root / "manifest.json",
        config_delta=root / "config_delta.env",
        config_full=root / "config_full.env",
        lock=root / "simulation.lock",
        stdout=root / "stdout.log",
        stderr=root / "stderr.log",
        results=root / "results",
    )


# ---------------------------------------------------------------------------
# VariantManifest
# ---------------------------------------------------------------------------

@dataclass
class VariantManifest:
    """Schema for ``manifest.json``.

    Mirrors the F7.1 spec in execution-plan.md plus a few operational
    fields the UI uses: ``status`` (the state-machine label),
    ``error_message`` (last failure detail), and ``label`` (the
    operator's friendly free-form name; ``name`` stays the slug for
    routing).
    """

    name: str                       # slug (matches directory name)
    label: str                       # operator-typed friendly name
    base_session: str                # session_id this variant derives from
    created_at_utc: str              # ISO timestamp
    bot_version_at_creation: str     # app.__version__ at create time
    base_config_sha256: str          # sha256 of source session's bot_config_envfile.txt
    status: str = "fresh"            # fresh | running | done | failed
    last_run_at_utc: Optional[str] = None
    last_run_duration_s: Optional[float] = None
    last_run_exit_code: Optional[int] = None
    error_message: Optional[str] = None
    # v1.5.135 (Codex bug #8) — newest-mtime across the variant's
    # ``results/`` directory at the moment ``mark_done()`` succeeded.
    # Used by ``variant_status()``'s stale check; avoids re-scanning
    # ``results/`` (which can have many files) on every status read.
    # ``None`` on legacy manifests written before this field existed
    # — ``variant_status()`` falls back to the live scan in that case.
    results_mtime: Optional[float] = None
    # Schema version for forward compat — bump when adding required fields.
    schema_version: int = 1


_VALID_STATUSES = frozenset({"fresh", "running", "done", "failed", "cancelled"})


def load_manifest(paths: VariantPaths) -> VariantManifest:
    """Read and parse ``manifest.json``. Raises ``FileNotFoundError``
    when the variant doesn't exist."""
    raw = paths.manifest.read_text(encoding="utf-8")
    data = json.loads(raw)
    # Filter unknown keys defensively — forward-compat with older
    # variants that may lack newer fields.
    known = {f for f in VariantManifest.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in known}
    return VariantManifest(**filtered)


def save_manifest(paths: VariantPaths, manifest: VariantManifest) -> None:
    """Atomic write: tmp file → rename. Avoids torn writes if the
    process dies mid-save."""
    paths.root.mkdir(parents=True, exist_ok=True)
    tmp = paths.manifest.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(paths.manifest)


# ---------------------------------------------------------------------------
# Variant CRUD
# ---------------------------------------------------------------------------

def list_variants(
    session_id: str, *, repo_root: Optional[Path] = None
) -> list[str]:
    """Return the slugs of every variant under ``session_id``,
    sorted lexicographically. Empty list when the session has no
    variants (or doesn't exist as a sim dir yet)."""
    _validate_session_id(session_id)
    sess_dir = session_simulations_dir(session_id, repo_root=repo_root)
    if not sess_dir.is_dir():
        return []
    out: list[str] = []
    for child in sess_dir.iterdir():
        if not child.is_dir():
            continue
        if not is_valid_slug(child.name):
            # Skip junk directories — operator may have dropped files
            # in there manually. Don't surface them as variants.
            continue
        out.append(child.name)
    out.sort()
    return out


def create_variant(
    session_id: str,
    variant_label: str,
    *,
    base_session_dir: Path,
    bot_version: str,
    repo_root: Optional[Path] = None,
) -> VariantPaths:
    """Scaffold a new variant directory and seed an empty delta.

    Args:
        session_id: name of the recording dir (under ``backtesting/data/sessions/``).
        variant_label: operator-typed friendly name; auto-slugged for
            the directory name. The friendly form is preserved on the
            ``manifest.label`` field.
        base_session_dir: absolute path to the recording dir.
            ``base_session_dir/bot_config_envfile.txt`` is hashed and
            recorded as the base config's sha256.
        bot_version: ``app.__version__`` at creation time.

    Raises:
        FileExistsError: if the variant slug already exists.
        FileNotFoundError: if ``base_session_dir/bot_config_envfile.txt``
            is missing — operator must pick a real recording.
    """
    slug = slugify_variant_name(variant_label)
    paths = variant_paths(session_id, slug, repo_root=repo_root)
    if paths.root.exists():
        raise FileExistsError(
            f"variant {slug!r} already exists under session {session_id!r}"
        )
    base_env_path = Path(base_session_dir) / "bot_config_envfile.txt"
    if not base_env_path.is_file():
        raise FileNotFoundError(
            f"base session {base_session_dir} has no bot_config_envfile.txt; "
            "cannot create a variant against an unrecorded session"
        )
    base_text = base_env_path.read_text(encoding="utf-8")
    base_sha = hashlib.sha256(base_text.encode("utf-8")).hexdigest()

    # Create dirs.
    paths.root.mkdir(parents=True)
    paths.results.mkdir()
    # Seed an empty delta — same comment header makes the file look
    # like a normal env file to text editors.
    paths.config_delta.write_text(
        "# Variant delta — only rows that differ from\n"
        f"# {base_session_dir.name}/bot_config_envfile.txt go here.\n"
        "# Empty file = no overrides (replay uses the recorded config\n"
        "# verbatim).\n",
        encoding="utf-8",
    )
    # config_full is materialised on first save / first run — leave
    # absent at creation. The simulator wrapper (F7.3) writes it.
    manifest = VariantManifest(
        name=slug,
        label=variant_label.strip() or slug,
        base_session=session_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        bot_version_at_creation=bot_version,
        base_config_sha256=base_sha,
        status="fresh",
    )
    save_manifest(paths, manifest)
    return paths


def delete_variant(
    session_id: str,
    variant_slug: str,
    *,
    repo_root: Optional[Path] = None,
) -> None:
    """Remove the entire variant directory tree. Idempotent — silently
    returns if the variant doesn't exist.

    v1.5.135 (Codex bug #1): only LIVE locks block the delete. A stale
    lock (crashed wrapper, dead PID) is auto-cleared so the operator
    doesn't have to reach into the filesystem."""
    paths = variant_paths(session_id, variant_slug, repo_root=repo_root)
    if not paths.root.exists():
        return
    state = effective_lock_state(paths)
    if state == "live":
        raise RuntimeError(
            f"variant {variant_slug!r} has an active simulation lock; "
            "cancel or wait for the run to finish before deleting"
        )
    if state == "stale":
        _clear_stale_lock(paths)
    shutil.rmtree(paths.root)


def rename_variant(
    session_id: str,
    old_slug: str,
    new_label: str,
    *,
    repo_root: Optional[Path] = None,
) -> VariantPaths:
    """Rename in place: directory rename + manifest update. The
    new label is slugged for the directory, preserved verbatim on
    ``manifest.label``."""
    new_slug = slugify_variant_name(new_label)
    if new_slug == old_slug:
        # Trivial label-only rename — just update the manifest's
        # friendly name.
        paths = variant_paths(session_id, old_slug, repo_root=repo_root)
        if paths.root.exists():
            m = load_manifest(paths)
            m.label = new_label.strip() or m.label
            save_manifest(paths, m)
        return paths
    old_paths = variant_paths(session_id, old_slug, repo_root=repo_root)
    new_paths = variant_paths(session_id, new_slug, repo_root=repo_root)
    if not old_paths.root.exists():
        raise FileNotFoundError(
            f"variant {old_slug!r} not found under session {session_id!r}"
        )
    if new_paths.root.exists():
        raise FileExistsError(
            f"new variant slug {new_slug!r} already exists"
        )
    # v1.5.135 (Codex bug #1): only LIVE locks block the rename.
    # Stale lock (dead PID) gets auto-cleared.
    state = effective_lock_state(old_paths)
    if state == "live":
        raise RuntimeError(
            f"variant {old_slug!r} has an active simulation lock; cannot rename"
        )
    if state == "stale":
        _clear_stale_lock(old_paths)
    old_paths.root.rename(new_paths.root)
    # Update the manifest in the new location so ``name`` stays in sync.
    m = load_manifest(new_paths)
    m.name = new_slug
    m.label = new_label.strip() or new_slug
    save_manifest(new_paths, m)
    return new_paths


# ---------------------------------------------------------------------------
# Variant status state machine
# ---------------------------------------------------------------------------

def variant_status(paths: VariantPaths) -> str:
    """Compute the variant's effective status, factoring in the
    on-disk lock file + the delta-vs-results mtime check for ``stale``.

    Distinct from ``manifest.status`` because:
      * ``running`` is determined by lock-file presence (manifest may
        not have been updated if simulator crashed)
      * ``stale`` is computed from mtimes (manifest doesn't know if
        the operator edited the delta after the last run)

    Returns one of: ``fresh`` | ``running`` | ``done`` | ``failed`` |
    ``stale``. If the manifest is missing or malformed, returns
    ``"unknown"``.
    """
    if not paths.manifest.exists():
        return "unknown"
    # v1.5.135 (Codex bug #1): only LIVE locks force "running".
    # Stale locks fall through to the manifest's last-known status.
    lock_state = effective_lock_state(paths)
    if lock_state == "live":
        return "running"
    if lock_state == "stale":
        # Pre-fix code returned "running" here, leaving the variant
        # permanently stuck. Now we self-heal: clear the stale lock
        # AND, if the manifest still says "running" (because
        # mark_running() set it but the wrapper crashed before
        # mark_done() could write a terminal status), demote it to
        # "failed" with a synthetic error message so the UI shows
        # a diagnosable state.
        _clear_stale_lock(paths)
        try:
            m_pre = load_manifest(paths)
        except (json.JSONDecodeError, KeyError, TypeError):
            return "unknown"
        if m_pre.status == "running":
            m_pre.status = "failed"
            m_pre.error_message = (
                m_pre.error_message
                or "wrapper crashed (stale lock with dead PID detected)"
            )
            save_manifest(paths, m_pre)
    try:
        m = load_manifest(paths)
    except (json.JSONDecodeError, KeyError, TypeError):
        return "unknown"
    base_status = m.status if m.status in _VALID_STATUSES else "fresh"
    if base_status not in ("done",):
        # Only ``done`` variants can be ``stale``. fresh/failed stay
        # where they are.
        return base_status
    # Stale check: delta mtime > results mtime → operator edited
    # after the last successful run.
    try:
        delta_mtime = paths.config_delta.stat().st_mtime
    except FileNotFoundError:
        return base_status
    # v1.5.135 (Codex bug #8): use cached results_mtime from manifest
    # when available. Falls back to scanning the directory only on
    # legacy manifests written before this field existed.
    if m.results_mtime is not None:
        results_mtime = m.results_mtime
    else:
        results_mtime = _scan_results_mtime(paths)
    if delta_mtime > results_mtime:
        return "stale"
    return "done"


def mark_running(
    paths: VariantPaths,
    *,
    pid: Optional[int] = None,
    claim_token: Optional[str] = None,
) -> None:
    """Drop the lock file and set manifest.status = 'running'. Caller
    (the F7.3 ``run_variant.py`` wrapper, or the F7.4 ``/run`` API
    endpoint) calls this before spawning the simulator. Idempotent.

    The lock file is JSON-encoded
    ``{pid, started_at_utc, claim_token?}``. The ``/cancel`` endpoint
    uses ``pid`` to signal the process tree. The ``claim_token``
    (v1.5.122) is a UUID generated by ``/run`` BEFORE spawning the
    wrapper, passed to the wrapper via env var
    ``BACKTEST_CLAIM_TOKEN``, and matched against the lock to defeat
    Windows PID-recycling false positives — the 2026-05-25 ghost-
    lock problem where ``OpenProcess(pid)`` returned alive for a
    PID that had been recycled to a completely unrelated process.

    Args:
        pid: process id to record. When ``None``, uses ``os.getpid()``
            (right for the wrapper's self-marking).
        claim_token: optional UUID. When ``None``, omitted from the
            lock payload (standalone CLI mode). When set, written
            into the lock so the wrapper / cancel endpoint can
            distinguish "lock written for THIS run" from "lock from
            a previous attempt that happens to have a recycled
            alive PID".
    """
    paths.root.mkdir(parents=True, exist_ok=True)
    lock_pid = pid if pid is not None else os.getpid()
    payload: dict[str, Any] = {
        "pid": int(lock_pid),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if claim_token:
        payload["claim_token"] = claim_token
    paths.lock.write_text(
        json.dumps(payload), encoding="utf-8",
    )
    if paths.manifest.exists():
        m = load_manifest(paths)
        m.status = "running"
        m.last_run_at_utc = datetime.now(timezone.utc).isoformat()
        save_manifest(paths, m)


def read_lock_info(paths: VariantPaths) -> Optional[dict[str, Any]]:
    """Return the lock file's JSON payload, or ``None`` if no lock.

    Returns a best-effort dict even when the lock file is empty or
    unparseable (older zero-byte locks from pre-v1.5.96 callers) — in
    that case the dict is ``{"pid": None, "started_at_utc": None}``.
    The caller checks ``pid`` before attempting to signal anything.
    """
    if not paths.lock.exists():
        return None
    try:
        raw = paths.lock.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return {"pid": None, "started_at_utc": None}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"pid": None, "started_at_utc": None}
    if not isinstance(data, dict):
        return {"pid": None, "started_at_utc": None}
    pid_raw = data.get("pid")
    pid_val: Optional[int] = None
    if isinstance(pid_raw, int) and pid_raw > 0:
        pid_val = pid_raw
    # v1.5.122 — propagate the claim_token field through.
    token_raw = data.get("claim_token")
    token_val: Optional[str] = (
        token_raw if isinstance(token_raw, str) and token_raw else None
    )
    return {
        "pid": pid_val,
        "started_at_utc": data.get("started_at_utc"),
        "claim_token": token_val,
    }


def read_lock_pid(paths: VariantPaths) -> Optional[int]:
    """Convenience accessor: just the PID from the lock file, or
    ``None`` when the lock is missing / empty / unparseable / has
    no valid PID. Used by ``run_variant.py``'s self-PID check
    (v1.5.116) to distinguish "the lock the API endpoint wrote for
    ME" from "the lock another process owns"."""
    info = read_lock_info(paths)
    if info is None:
        return None
    pid = info.get("pid")
    return pid if isinstance(pid, int) and pid > 0 else None


def read_lock_claim_token(paths: VariantPaths) -> Optional[str]:
    """Read the claim token from the lock file (v1.5.122).
    Returns ``None`` when the lock is missing, has no token (older
    callers / standalone CLI mode), or the token field isn't a
    non-empty string. The wrapper uses this for the
    PID-recycling-immune ownership check."""
    info = read_lock_info(paths)
    if info is None:
        return None
    token = info.get("claim_token") if isinstance(info, dict) else None
    if isinstance(token, str) and token:
        return token
    return None


def is_pid_alive(pid: int) -> bool:
    """Cross-platform PID-liveness probe.

    Returns True if a process with the given PID is currently alive,
    False if it's gone (dead or never existed). Used by
    ``run_variant.py``'s ghost-lock takeover logic (v1.5.121) to
    distinguish "lock is held by a real concurrent run" from "lock
    is stale because the previous wrapper crashed".

    Mirrors the TS-side ``_isPidAlive`` in ``lib/simulation_storage.ts``.

    Windows: uses ``OpenProcess`` + ``GetExitCodeProcess`` via
    ``ctypes`` rather than ``os.kill(pid, 0)`` because Python's
    ``os.kill`` on Windows is mapped to ``TerminateProcess`` —
    signal-0 doesn't get the "probe only" semantics it has on POSIX,
    it would actually KILL the target.

    POSIX: ``os.kill(pid, 0)`` is the canonical no-op probe. Raises
    ``ProcessLookupError`` (errno ESRCH) when the PID is gone;
    ``PermissionError`` (EPERM) means the process exists but we
    can't signal it — treated as alive defensively.

    Treats negative / zero PIDs as not-alive (defensive — they
    have signal-broadcast semantics on POSIX that we'd never want
    to invoke accidentally).
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    import sys
    if sys.platform == "win32":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
        )
        if not handle:
            # OpenProcess returns 0 (NULL) when the PID is gone (or
            # when we lack permission — rare on the operator's
            # single-user laptop, ignored defensively).
            return False
        try:
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            if not ok:
                return False
            # STILL_ACTIVE (259) means the process is still running.
            # Anything else is its real exit code, meaning it exited.
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    # POSIX path.
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but owned by another user — defensively
        # treat as alive (we shouldn't kill it / take its lock).
        return True
    except OSError:
        # Anything else (shouldn't happen) — defensively assume alive.
        return True


def effective_lock_state(paths: VariantPaths) -> str:
    """Return one of ``"no_lock" | "live" | "stale"`` for ``paths.lock``.

    v1.5.135 (Codex bug #1). The pre-fix code in ``delete_variant``,
    ``rename_variant``, and ``variant_status`` treated
    ``paths.lock.exists()`` as proof of a live run. That left a
    crashed wrapper's variant permanently ``"running"`` — operator
    could not rename or delete without manual filesystem cleanup.

    This helper centralises the PID-liveness check (the same
    machinery ``run_variant.py``'s ghost-lock takeover uses) so all
    three call sites agree on what a "live" lock actually means.

      * ``"no_lock"``   — no lock file present.
      * ``"live"``      — lock has a valid PID and that PID is alive.
      * ``"stale"``     — lock present but the recorded PID is gone
                          (wrapper crashed), or no PID was recorded
                          (pre-v1.5.96 empty lock files / corrupt
                          payloads — same effective state because
                          there's no way to verify ownership).

    Important Windows caveat: PID liveness suffers from PID-recycling.
    The wrapper itself uses a ``claim_token`` to disambiguate; that
    check is for the wrapper's own self-recognition. Third-party
    callers (delete/rename/status) don't know the token and can only
    fall back to PID liveness — accepting the small chance that an
    alive recycled PID looks "live". Acceptable trade-off: the worst
    case is a falsely-blocked rename/delete on a session where
    Windows happened to recycle that exact PID since the wrapper
    died, vs the pre-fix worst case of "permanently stuck running".
    """
    info = read_lock_info(paths)
    if info is None:
        return "no_lock"
    pid = info.get("pid")
    if isinstance(pid, int) and pid > 0 and is_pid_alive(pid):
        return "live"
    return "stale"


def _clear_stale_lock(paths: VariantPaths) -> None:
    """Idempotently remove a stale lock file. Used by ``delete_variant``
    / ``rename_variant`` / ``variant_status`` after they've
    established via ``effective_lock_state`` that the lock is stale,
    so the next call sees a clean state."""
    try:
        paths.lock.unlink()
    except FileNotFoundError:
        pass


def mark_done(
    paths: VariantPaths,
    *,
    duration_s: float,
    exit_code: int,
    error_message: Optional[str] = None,
) -> None:
    """Write terminal manifest state, THEN clear the lock.
    ``exit_code == 0`` → ``done``; non-zero → ``failed`` (caller may
    pass an ``error_message`` scraped from stderr.log).

    ``error_message`` is honoured **unconditionally**. The
    F7.12.5 case is "replay succeeded but the post-run bot_db
    extract failed" — operator needs to see the extract error
    even though the variant ends up ``done``. Pass
    ``error_message=None`` (the default) to clear a stale error
    from a previous failed run.

    v1.5.135 (Codex bug #2) — write order REVERSED from the original
    "unlink lock first, then save manifest" sequence. If manifest
    save fails between the unlink and the write, the variant ends up
    in an inconsistent state: lock gone (UI says "not running") but
    terminal status not persisted (manifest still says "running"
    from ``mark_running()``). Saving the manifest first means a save
    failure leaves the lock intact — the variant stays visibly
    ``"running"`` (or transitions to ``"stale"`` via the new
    PID-liveness path once the wrapper dies), which is at least
    diagnosable.

    v1.5.135 (Codex bug #8) — also captures ``results_mtime`` at
    this point (newest mtime across the variant's ``results/``
    tree). ``variant_status()`` reads this cached value on its
    stale-check, avoiding a directory rescan on every status read.
    """
    # 1. Capture results_mtime BEFORE writing manifest so the cached
    #    value reflects the actual results dir at the moment the run
    #    finished.
    results_mtime = _scan_results_mtime(paths)
    # 2. Write terminal manifest state via the atomic tmp+rename
    #    pattern in save_manifest(). If THIS fails, the lock is
    #    still present — the variant stays "running" until next
    #    /cancel or /clear, which is the diagnosable state we want.
    m = load_manifest(paths)
    m.status = "done" if exit_code == 0 else "failed"
    m.last_run_duration_s = duration_s
    m.last_run_exit_code = exit_code
    m.error_message = error_message
    m.results_mtime = results_mtime
    save_manifest(paths, m)
    # 3. Only NOW unlink the lock. Terminal state is durably on
    #    disk; if the unlink fails (e.g. file already gone) we
    #    swallow it — the next status read sees the persisted
    #    terminal state regardless.
    try:
        paths.lock.unlink()
    except FileNotFoundError:
        pass


def _scan_results_mtime(paths: VariantPaths) -> float:
    """Compute the newest mtime across the variant's ``results/``
    directory. Returns 0.0 when the directory is missing or empty
    (no results files yet). Used by ``mark_done()`` to cache the
    value into the manifest (Codex bug #8) and by ``variant_status()``
    as a fallback for legacy manifests without the cached field."""
    if not paths.results.is_dir():
        return 0.0
    newest = 0.0
    for child in paths.results.iterdir():
        try:
            t = child.stat().st_mtime
        except FileNotFoundError:
            continue
        if t > newest:
            newest = t
    return newest


# ---------------------------------------------------------------------------
# Config materialisation (delta → config_full.env)
# ---------------------------------------------------------------------------

def _parse_env_kv(text: str) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` env file into an ordered dict.

    Tolerant of comments (``#``), blank lines, and inline trailing
    comments. Mirrors the parser in ``app.backtest.bot_runner.
    _parse_profile_env`` so that materialised configs round-trip
    identically through the bot's settings loader.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        hash_pos = line.find(" #")
        if hash_pos > 0:
            line = line[:hash_pos].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if key:
            out[key] = val
    return out


def materialize_full_config(
    paths: VariantPaths,
    *,
    repo_root: Optional[Path] = None,
) -> Path:
    """Merge ``<base_session>/bot_config_envfile.txt`` with the
    variant's ``config_delta.env`` and write the result to
    ``<variant>/config_full.env``. Returns the materialised path.

    Algorithm:
      1. Read base env line-by-line. For every ``KEY=VALUE`` line
         whose key appears in the delta, replace the value (preserving
         leading whitespace + inline ``# comment``).
      2. Append any delta-only keys at the tail under a
         ``# === variant overrides === `` header.
      3. Write atomically (tmp + rename).

    The bot's settings loader (``_parse_profile_env``) ignores
    comments + blank lines, so a delta that just overrides existing
    keys lands at the original line positions — diffability stays
    high. New delta keys land at the tail in delta-file order.

    Raises:
        FileNotFoundError: when the base session's env file is
            missing. (A missing delta is treated as empty — variants
            created via this module always have a delta file, but a
            hand-deleted one shouldn't crash the run path.)
    """
    base_path = session_recording_dir(
        paths.session_id, repo_root=repo_root
    ) / "bot_config_envfile.txt"
    if not base_path.is_file():
        raise FileNotFoundError(
            f"base session {paths.session_id!r} has no "
            f"{base_path.name}; cannot materialise full config"
        )
    base_text = base_path.read_text(encoding="utf-8")
    try:
        delta_text = paths.config_delta.read_text(encoding="utf-8")
    except FileNotFoundError:
        delta_text = ""
    delta = _parse_env_kv(delta_text)

    out_lines: list[str] = []
    applied: set[str] = set()
    for raw in base_text.splitlines():
        stripped = raw.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out_lines.append(raw)
            continue
        # Preserve leading whitespace (rare in env files but cheap).
        leading_len = len(raw) - len(stripped)
        leading = raw[:leading_len]
        key, _, rest = stripped.partition("=")
        key = key.strip()
        if key in delta:
            # Preserve inline trailing comment if present.
            hash_pos = rest.find(" #")
            inline_comment = rest[hash_pos:] if hash_pos >= 0 else ""
            out_lines.append(f"{leading}{key}={delta[key]}{inline_comment}")
            applied.add(key)
        else:
            out_lines.append(raw)

    # Append delta-only keys (in their delta-file order, which
    # ``dict`` preserves since 3.7).
    leftover = [(k, v) for k, v in delta.items() if k not in applied]
    if leftover:
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")
        out_lines.append("# === variant overrides ===")
        out_lines.append(f"# Source: {paths.config_delta.name}")
        for k, v in leftover:
            out_lines.append(f"{k}={v}")

    new_text = "\n".join(out_lines)
    # Preserve the trailing newline that's idiomatic for env files
    # (and that ``str.splitlines()`` ate above).
    if base_text.endswith("\n") or new_text and not new_text.endswith("\n"):
        new_text += "\n"

    paths.root.mkdir(parents=True, exist_ok=True)
    tmp = paths.config_full.with_suffix(".env.tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(paths.config_full)
    return paths.config_full
