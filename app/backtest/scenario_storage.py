"""Storage primitives for the Phase 4B scenario library (audit P1 #6).

A *scenario* is a named, archetype-tagged time-slice cut from a source
recording under ``backtesting/data/sessions/``. The probe runner
(``scripts/backtest/run_probe.py``, audit P1 #7) consumes scenarios by
archetype or id, replaying each under a matrix of config variants. The
scenario library is what turns "days of live tuning" into "one CLI =
one calibration cycle" (audit §5 P1).

On-disk layout (``backtesting/data/`` is gitignored — scenarios live on
local SSD, created on demand by :func:`create_scenario`)::

    backtesting/data/scenarios/
      <scenario_id>/
        manifest.json           ← ScenarioManifest (this module owns it)
        initial_state.json      ← optional pre-T0 state seed (audit P2 #14)
        okx_public.jsonl.gz     ← sliced feed files, named by recorder
        binance_public.jsonl.gz   `source` (okx_public / binance_public /
        okx_private.jsonl.gz      okx_private / okx_business / okx_rest).
        ...                       The frontend / CLI *cutter* writes these;
                                  this module only owns the manifest.

This module is the canonical authority for that layout: every caller
(the ``run_probe`` CLI, the future frontend cutter API) resolves
scenario paths through :func:`scenario_paths` and validates ids through
:func:`is_valid_scenario_id`.

Manifest schema mirrors ``execution-plan.md`` §4B.1, *reality-aligned*:
the plan sketch used ``.jsonl.zst`` and feeds named ``public_okx``; the
real recorder (see ``backtesting/data/sessions/*/manifest.json``) writes
``.jsonl.gz`` with feeds named ``okx_public`` etc. The manifest here is
format-agnostic — each :class:`ScenarioDataFile` records the feed name +
relative path verbatim, so it tracks whatever the cutter produced.

Path-traversal guard: :func:`scenario_paths` rejects any id whose
resolved directory escapes :func:`scenarios_root`. Same posture as
``app.backtest.simulation_storage`` and the frontend's
``resolveUnderDataRoot``.

This module performs NO data-file slicing and NO heavy metric
computation. Data-derived scenario metrics (SF count, max adverse bps,
mean vol_bps) come from the replay report (``app/backtest/report.py``);
if the cutter stamped a cheap summary at cut time it rides on
``ScenarioManifest.summary`` and :func:`scenario_summary` surfaces it
verbatim.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Id / archetype validation
# ---------------------------------------------------------------------------

# scenario_id is the directory name under scenarios/. Operator-chosen,
# typically ``<archetype>_<source-version>_<date>_<hhmmss>`` e.g.
# ``structural_bias_short_v1_4_219_260521_214353`` or ``low_vol_2026-05-15``.
# Lowercase slug alphabet (letters, digits, ``_``, ``-``); must start and
# end with an alnum; rejects ``..`` / ``/`` / ``\`` implicitly because
# those characters aren't in the class.
_SCENARIO_ID_MAX_LEN = 96
_SCENARIO_ID_RE = re.compile(
    r"^[a-z0-9][a-z0-9_-]{0,%d}[a-z0-9]$|^[a-z0-9]$" % (_SCENARIO_ID_MAX_LEN - 2)
)

# Archetype is a slug too (no path role) — same alphabet, shorter cap.
_ARCHETYPE_MAX_LEN = 64
_ARCHETYPE_RE = re.compile(
    r"^[a-z0-9][a-z0-9_-]{0,%d}[a-z0-9]$|^[a-z0-9]$" % (_ARCHETYPE_MAX_LEN - 2)
)


def is_valid_scenario_id(s: Any) -> bool:
    """Return True when ``s`` is already in canonical scenario-id form."""
    return (
        isinstance(s, str)
        and len(s) <= _SCENARIO_ID_MAX_LEN
        and bool(_SCENARIO_ID_RE.match(s))
    )


def is_valid_archetype(s: Any) -> bool:
    """Return True when ``s`` is a canonical archetype slug."""
    return (
        isinstance(s, str)
        and len(s) <= _ARCHETYPE_MAX_LEN
        and bool(_ARCHETYPE_RE.match(s))
    )


def _slugify(raw: str, *, max_len: int, kind: str) -> str:
    """Shared slug normaliser: lowercase, whitespace/``.`` → ``_``, drop
    everything outside ``[a-z0-9_-]``, collapse separator runs, trim.

    Raises ``ValueError`` when the result is empty or too long. Mirrors
    ``simulation_storage.slugify_variant_name`` so scenario ids and
    variant slugs share the same shape (operator muscle memory)."""
    if not isinstance(raw, str):
        raise ValueError(f"{kind} must be a string, got {type(raw)}")
    s = raw.strip().lower()
    s = re.sub(r"[\s.]+", "_", s)       # whitespace + dots → underscore
    s = re.sub(r"[^a-z0-9_-]", "", s)   # drop out-of-alphabet
    s = re.sub(r"[_-]{2,}", "_", s)     # collapse separator runs
    s = s.strip("_-")                   # trim leading/trailing separators
    if not s:
        raise ValueError(
            f"{kind} {raw!r} slugified to empty string; "
            "use letters, digits, '_' or '-'"
        )
    if len(s) > max_len:
        raise ValueError(
            f"{kind} {raw!r} slugified to {s!r}, length {len(s)} > max {max_len}"
        )
    return s


def slugify_scenario_id(raw: str) -> str:
    """Coerce a free-form name into a canonical scenario id."""
    return _slugify(raw, max_len=_SCENARIO_ID_MAX_LEN, kind="scenario_id")


def slugify_archetype(raw: str) -> str:
    """Coerce a free-form name into a canonical archetype slug."""
    return _slugify(raw, max_len=_ARCHETYPE_MAX_LEN, kind="archetype")


def _validate_scenario_id(scenario_id: str) -> str:
    if not is_valid_scenario_id(scenario_id):
        raise ValueError(
            f"scenario_id {scenario_id!r} not in canonical form "
            f"(lowercase alnum + '_'/'-', <= {_SCENARIO_ID_MAX_LEN} chars; "
            "call slugify_scenario_id first)"
        )
    return scenario_id


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _repo_root_default() -> Path:
    """Resolve the repo root from this file's location.

    Layout::

        <repo>/app/backtest/scenario_storage.py     ← this file
        <repo>/backtesting/data/scenarios/...        ← target
    """
    return Path(__file__).resolve().parent.parent.parent


def scenarios_root(*, repo_root: Optional[Path] = None) -> Path:
    """Top-level ``backtesting/data/scenarios/`` directory."""
    base = repo_root if repo_root is not None else _repo_root_default()
    return Path(base) / "backtesting" / "data" / "scenarios"


def sessions_root(*, repo_root: Optional[Path] = None) -> Path:
    """Top-level ``backtesting/data/sessions/`` directory (recorder
    output root — where scenarios are cut FROM). Mirrors
    ``simulation_storage.sessions_root``."""
    base = repo_root if repo_root is not None else _repo_root_default()
    return Path(base) / "backtesting" / "data" / "sessions"


@dataclass(frozen=True)
class ScenarioPaths:
    """Concrete on-disk paths for one scenario. All paths are absolute and
    rooted under :func:`scenarios_root` — verified by
    :func:`scenario_paths`'s path-traversal guard."""

    scenario_id: str
    root: Path
    manifest: Path
    initial_state: Path


def scenario_paths(
    scenario_id: str, *, repo_root: Optional[Path] = None
) -> ScenarioPaths:
    """Compute on-disk paths for one scenario. Validates the id and guards
    against path traversal. Does NOT create directories — pure arithmetic.
    Use :func:`create_scenario` to scaffold on disk."""
    _validate_scenario_id(scenario_id)
    base = scenarios_root(repo_root=repo_root).resolve()
    root = (base / scenario_id).resolve()
    try:
        root.relative_to(base)
    except ValueError:
        raise ValueError(
            f"resolved scenario root {root} escapes scenarios_root {base}"
        )
    return ScenarioPaths(
        scenario_id=scenario_id,
        root=root,
        manifest=root / "manifest.json",
        initial_state=root / "initial_state.json",
    )


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

@dataclass
class ScenarioDataFile:
    """One sliced feed file inside a scenario dir. ``feed`` matches the
    recorder's ``source`` tag (``okx_public`` / ``binance_public`` /
    ``okx_private`` / ``okx_business`` / ``okx_rest`` / ``trades``);
    ``path`` is the file's name relative to the scenario root."""

    feed: str
    path: str
    bytes: int = 0
    lines: Optional[int] = None
    first_t_recv_ns: Optional[int] = None
    last_t_recv_ns: Optional[int] = None


@dataclass
class ScenarioManifest:
    """Schema for a scenario's ``manifest.json`` (execution-plan §4B.1,
    reality-aligned). ``schema_version`` is bumped when a required field
    is added so :func:`load_manifest` can stay forward-compatible."""

    scenario_id: str                       # matches the directory name
    archetype: str                         # e.g. structural_bias_short, sf_storm
    symbol: str                            # e.g. TON-USDT-SWAP
    venue: str                             # e.g. okx
    source_recording_id: str               # session_id this was cut from
    cut_start_utc: str                     # ISO timestamp (inclusive)
    cut_end_utc: str                       # ISO timestamp (exclusive)
    cut_duration_seconds: float
    data_files: list[ScenarioDataFile] = field(default_factory=list)
    operator_tags: list[str] = field(default_factory=list)
    operator_notes: str = ""
    # Optional relative path (within the scenario dir) to an
    # ``initial_state.json`` pre-T0 state seed for warm-start replay
    # (audit P2 #14). Conventionally just ``"initial_state.json"``. The
    # cutter writes a snapshot of the source recording's bot state at the
    # cut boundary (T₀); the replay loader (``scripts/backtest/replay.py``
    # ``_load_initial_state``) reads the sidecar file and
    # ``app.backtest.seed_state_from_snapshot`` applies it so the scored
    # window ``[T₀, T_end]`` starts with already-warm estimators (no
    # warmup carve-out). Schema (all keys optional, mirrors
    # ``state_current.json``)::
    #
    #     {"binance_basis_ewma": <float>,   # OKX-minus-Binance mid EWMA
    #      "vol_bps": <float>,              # per-step sigma × 1e4
    #      "vol_sigma": <float>}            # per-step sigma scalar
    #
    # flow_score (self-warms in ~1 s) and recent_fills (belong to the live
    # bot's order flow, gated out of the replay queue) are deliberately
    # NOT seeded — see ``seed_state_from_snapshot`` for the rationale.
    # ``None`` ⇒ cold start; the warmup window warms the state instead.
    initial_state_snapshot: Optional[str] = None
    recorder_version: str = ""
    captured_at_utc: str = ""              # when the scenario was cut
    discarded_from_source: bool = False    # was the source recording freed?
    # Cheap cut-time metrics (SF count, max adverse bps, mean vol_bps, …)
    # stamped by the cutter when it has the data in hand. ``None`` means
    # "not computed" — the replay report is the authoritative source.
    summary: Optional[dict[str, Any]] = None
    schema_version: int = 1


def _data_file_from_obj(obj: Any) -> Optional[ScenarioDataFile]:
    """Coerce one ``data_files`` entry into a :class:`ScenarioDataFile`.
    Returns ``None`` for entries missing the required ``feed``/``path``
    keys (defensive against hand-edited manifests)."""
    if isinstance(obj, ScenarioDataFile):
        return obj
    if not isinstance(obj, dict):
        return None
    if "feed" not in obj or "path" not in obj:
        return None
    known = {f for f in ScenarioDataFile.__dataclass_fields__}
    return ScenarioDataFile(**{k: v for k, v in obj.items() if k in known})


def _manifest_from_dict(data: dict[str, Any]) -> ScenarioManifest:
    """Build a :class:`ScenarioManifest` from a parsed dict, filtering
    unknown keys (forward-compat) and reconstructing nested data files."""
    known = {f for f in ScenarioManifest.__dataclass_fields__}
    filtered = {k: v for k, v in data.items() if k in known}
    raw_files = filtered.get("data_files") or []
    files: list[ScenarioDataFile] = []
    if isinstance(raw_files, list):
        for item in raw_files:
            df = _data_file_from_obj(item)
            if df is not None:
                files.append(df)
    filtered["data_files"] = files
    return ScenarioManifest(**filtered)


def load_manifest(paths: ScenarioPaths) -> ScenarioManifest:
    """Read and parse ``manifest.json``. Raises ``FileNotFoundError`` when
    the scenario doesn't exist, ``json.JSONDecodeError`` on corrupt JSON,
    and ``TypeError`` when a required field is missing."""
    raw = paths.manifest.read_text(encoding="utf-8")
    data = json.loads(raw)
    return _manifest_from_dict(data)


def save_manifest(paths: ScenarioPaths, manifest: ScenarioManifest) -> None:
    """Atomic write: tmp file → rename. Avoids torn writes if the process
    dies mid-save. Creates the scenario dir if absent."""
    paths.root.mkdir(parents=True, exist_ok=True)
    tmp = paths.manifest.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(paths.manifest)


# ---------------------------------------------------------------------------
# Listing + archetype manifest loader
# ---------------------------------------------------------------------------

def list_scenarios(*, repo_root: Optional[Path] = None) -> list[str]:
    """Return the ids of every scenario under ``scenarios_root``, sorted.
    A directory counts as a scenario only when it has a valid id AND a
    ``manifest.json`` (junk dirs / half-cut scenarios are skipped)."""
    root = scenarios_root(repo_root=repo_root)
    if not root.is_dir():
        return []
    out: list[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if not is_valid_scenario_id(child.name):
            continue
        if not (child / "manifest.json").is_file():
            continue
        out.append(child.name)
    out.sort()
    return out


def load_all_manifests(
    *, repo_root: Optional[Path] = None
) -> list[ScenarioManifest]:
    """Load every scenario's manifest, best-effort. Malformed manifests
    (bad JSON, missing required field) are skipped rather than aborting
    the whole library load — one corrupt scenario shouldn't break a
    probe run across the rest. Sorted by ``scenario_id``."""
    out: list[ScenarioManifest] = []
    for sid in list_scenarios(repo_root=repo_root):
        paths = scenario_paths(sid, repo_root=repo_root)
        try:
            out.append(load_manifest(paths))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
    out.sort(key=lambda m: m.scenario_id)
    return out


def filter_by_archetype(
    manifests: list[ScenarioManifest], pattern: str
) -> list[ScenarioManifest]:
    """Filter manifests by archetype. ``pattern`` may be an exact
    archetype (``structural_bias_short``) or a glob
    (``structural_bias_*``) — the probe YAML uses both forms
    (``archetype:structural_bias_*`` in §4B.2). fnmatch handles ``*``,
    ``?`` and ``[seq]``."""
    return [
        m
        for m in manifests
        if m.archetype == pattern or fnmatch.fnmatch(m.archetype, pattern)
    ]


def filter_by_tag(
    manifests: list[ScenarioManifest], tag: str
) -> list[ScenarioManifest]:
    """Filter manifests carrying ``tag`` in ``operator_tags``."""
    return [m for m in manifests if tag in (m.operator_tags or [])]


def resolve_selector(
    manifests: list[ScenarioManifest], selector: str
) -> list[ScenarioManifest]:
    """Resolve one probe-YAML scenario selector to concrete manifests.

    Supported forms (execution-plan §4B.2)::

        scenario_id:<id>      → the one scenario with that exact id
        archetype:<pattern>   → archetype exact match OR glob (fnmatch)
        <archetype>           → bare form, exact archetype match
                                (the `- archetype: structural_bias_short`
                                list entry)

    Returns the (possibly empty) list of matching manifests. The
    run-probe CLL deduplicates across selectors."""
    sel = selector.strip()
    prefix_id = "scenario_id:"
    prefix_arch = "archetype:"
    if sel.startswith(prefix_id):
        sid = sel[len(prefix_id):].strip()
        return [m for m in manifests if m.scenario_id == sid]
    if sel.startswith(prefix_arch):
        pat = sel[len(prefix_arch):].strip()
        return filter_by_archetype(manifests, pat)
    return [m for m in manifests if m.archetype == sel]


# ---------------------------------------------------------------------------
# Scenario CRUD
# ---------------------------------------------------------------------------

def create_scenario(
    *,
    scenario_id: str,
    archetype: str,
    symbol: str,
    venue: str,
    source_recording_id: str,
    cut_start_utc: str,
    cut_end_utc: str,
    cut_duration_seconds: float,
    data_files: Optional[list[Any]] = None,
    operator_tags: Optional[list[str]] = None,
    operator_notes: str = "",
    initial_state_snapshot: Optional[str] = None,
    recorder_version: str = "",
    summary: Optional[dict[str, Any]] = None,
    discarded_from_source: bool = False,
    require_data_files: bool = True,
    repo_root: Optional[Path] = None,
) -> ScenarioPaths:
    """Scaffold a scenario directory and write its manifest.

    ``scenario_id`` and ``archetype`` are auto-slugged, so callers can
    pass friendly input. The *cutter* (frontend/CLI) is responsible for
    slicing the feed files into the scenario dir; this function only
    owns the manifest. It tolerates the cutter having pre-created the
    dir and dropped files in (``mkdir(exist_ok=True)``) but refuses to
    clobber an existing scenario that already has a ``manifest.json``.

    Args:
        require_data_files: when True (default), every entry in
            ``data_files`` must already exist on disk under the scenario
            root — guards against registering a manifest that points at
            missing feeds. Pass False to register the manifest first and
            slice the feeds afterward (or in tests).

    Raises:
        FileExistsError: a scenario with this id already has a manifest.
        FileNotFoundError: ``require_data_files`` and a listed feed file
            is absent.
    """
    sid = slugify_scenario_id(scenario_id)
    arch = slugify_archetype(archetype)
    paths = scenario_paths(sid, repo_root=repo_root)
    if paths.manifest.exists():
        raise FileExistsError(f"scenario {sid!r} already exists")

    norm_files: list[ScenarioDataFile] = []
    for f in data_files or []:
        df = _data_file_from_obj(f)
        if df is None:
            raise ValueError(f"invalid data_files entry: {f!r}")
        norm_files.append(df)

    paths.root.mkdir(parents=True, exist_ok=True)

    if require_data_files:
        for df in norm_files:
            if not (paths.root / df.path).is_file():
                raise FileNotFoundError(
                    f"scenario {sid!r} data file {df.path!r} not found under "
                    f"{paths.root}; the cutter must slice feeds before "
                    "registering the manifest (or pass require_data_files=False)"
                )

    manifest = ScenarioManifest(
        scenario_id=sid,
        archetype=arch,
        symbol=symbol,
        venue=venue,
        source_recording_id=source_recording_id,
        cut_start_utc=cut_start_utc,
        cut_end_utc=cut_end_utc,
        cut_duration_seconds=float(cut_duration_seconds),
        data_files=norm_files,
        operator_tags=list(operator_tags or []),
        operator_notes=operator_notes,
        initial_state_snapshot=initial_state_snapshot,
        recorder_version=recorder_version,
        captured_at_utc=datetime.now(timezone.utc).isoformat(),
        discarded_from_source=discarded_from_source,
        summary=summary,
    )
    save_manifest(paths, manifest)
    return paths


def delete_scenario(
    scenario_id: str, *, repo_root: Optional[Path] = None
) -> None:
    """Remove a scenario directory tree. Idempotent — silently returns if
    the scenario doesn't exist."""
    paths = scenario_paths(scenario_id, repo_root=repo_root)
    if not paths.root.exists():
        return
    shutil.rmtree(paths.root)


# ---------------------------------------------------------------------------
# Summary view (for `backtest scenario show <id>`)
# ---------------------------------------------------------------------------

def scenario_summary(paths: ScenarioPaths) -> dict[str, Any]:
    """Build a lightweight manifest + on-disk view for the
    ``backtest scenario show`` CLI.

    Data-derived metrics (SF count, max adverse bps, mean vol_bps) are
    NOT recomputed here — they're expensive and the replay report
    (``app/backtest/report.py``) is their authoritative source. If the
    cutter stamped a cheap summary at cut time it rides on
    ``manifest.summary`` and is surfaced verbatim under ``summary``.
    What this DOES add over the raw manifest: per-file presence + byte
    sizes as they actually are on disk right now (so the operator can
    spot a half-synced or partially-deleted scenario)."""
    m = load_manifest(paths)
    files_on_disk: list[dict[str, Any]] = []
    total_bytes = 0
    for df in m.data_files:
        fp = paths.root / df.path
        try:
            size: Optional[int] = fp.stat().st_size
        except OSError:
            size = None
        if size is not None:
            total_bytes += size
        files_on_disk.append(
            {
                "feed": df.feed,
                "path": df.path,
                "present": fp.is_file(),
                "bytes_on_disk": size,
                "bytes_manifest": df.bytes,
            }
        )
    return {
        "scenario_id": m.scenario_id,
        "archetype": m.archetype,
        "symbol": m.symbol,
        "venue": m.venue,
        "source_recording_id": m.source_recording_id,
        "cut_start_utc": m.cut_start_utc,
        "cut_end_utc": m.cut_end_utc,
        "cut_duration_seconds": m.cut_duration_seconds,
        "operator_tags": list(m.operator_tags or []),
        "operator_notes": m.operator_notes,
        "recorder_version": m.recorder_version,
        "captured_at_utc": m.captured_at_utc,
        "discarded_from_source": m.discarded_from_source,
        "data_files": files_on_disk,
        "total_data_bytes_on_disk": total_bytes,
        "has_initial_state": paths.initial_state.is_file(),
        "summary": m.summary,
    }
