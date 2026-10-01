"""Recorded-event loader for backtest fixtures — Phase 3 (v1.4.234).

Reads a fixture directory produced by ``backtesting/recorder/`` and
yields events in monotonically non-decreasing ``t_recv_ns`` order
using a deterministic k-way merge across the per-source jsonl.gz files.

Fixture layout::

    fixture_dir/
        manifest.json             # session metadata + file list
        okx_public.jsonl.gz       # one event per line, sorted by t_recv_ns
        okx_private.jsonl.gz      # (optional)
        binance_public.jsonl.gz   # (optional)
        okx_rest.jsonl.gz         # (optional)

Each event line has the schema::

    {"t_recv_ns": <int>, "source": "<str>", "msg": <object>,
     "kind": <str|null, optional>}

Determinism contract: two consecutive iterations of the same fixture
produce byte-identical event sequences. Tie-breaking when two files
have an event with the same ``t_recv_ns`` is by manifest file order
(stable across reads). ``heapq.merge`` with a key function gives us
this for free — it compares ``(key, iterator_index, value)``.
"""

from __future__ import annotations

import gzip
import heapq
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RecordedEvent:
    """One line out of a recorder jsonl.gz file."""

    t_recv_ns: int
    source: str
    msg: Any  # typically dict, but some recorder shapes use list/str
    kind: Optional[str] = None


def _iter_jsonl_gz(path: Path) -> Iterator[RecordedEvent]:
    """Yield ``RecordedEvent``s from one gzipped JSONL file."""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    "skipping malformed JSON line in %s: %s", path, e
                )
                continue
            t_ns = obj.get("t_recv_ns")
            src = obj.get("source")
            if not isinstance(t_ns, int) or not isinstance(src, str):
                logger.warning(
                    "skipping malformed event in %s (t_recv_ns=%r source=%r)",
                    path,
                    t_ns,
                    src,
                )
                continue
            yield RecordedEvent(
                t_recv_ns=t_ns,
                source=src,
                msg=obj.get("msg"),
                kind=obj.get("kind"),
            )


class SortedEventStream:
    """K-way merge over the per-source files of a recorder fixture.

    Iteration order: by ``t_recv_ns`` ascending. Ties broken by
    manifest file order — stable across re-iterations of the same
    instance (deterministic).

    ``first_t_ns`` / ``last_t_ns`` are computed from the manifest's
    per-file metadata at construction time — no full scan needed.

    The stream is *single-pass per __iter__ call*: each iteration
    reopens the underlying files. This is intentional — replays read
    each event at most once, and reopening keeps memory bounded.
    """

    def __init__(
        self,
        fixture_dir: Path,
        *,
        sources: Optional[set[str]] = None,
        allow_gaps: bool = False,
    ) -> None:
        fixture_dir = Path(fixture_dir)
        manifest_path = fixture_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"fixture {fixture_dir} missing manifest.json"
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest: dict[str, Any] = json.load(f)
        self._fixture_dir = fixture_dir
        self._files: list[tuple[str, Path]] = []
        first_candidates: list[int] = []
        last_candidates: list[int] = []
        total_lines = 0
        total_gaps = 0
        for entry in self.manifest.get("files", []):
            src = entry.get("source", "")
            if sources is not None and src not in sources:
                continue
            file_path = fixture_dir / entry["path"]
            if not file_path.exists():
                raise FileNotFoundError(
                    f"manifest references {file_path} which does not exist"
                )
            self._files.append((src, file_path))
            if "first_t_recv_ns" in entry:
                first_candidates.append(int(entry["first_t_recv_ns"]))
            if "last_t_recv_ns" in entry:
                last_candidates.append(int(entry["last_t_recv_ns"]))
            total_lines += int(entry.get("lines", 0))
            total_gaps += int(entry.get("gaps", 0))

        if not self._files:
            raise ValueError(
                f"fixture {fixture_dir} contains no event files matching "
                f"sources filter {sources!r}"
            )

        # Sort files by source name so iteration order is stable across
        # platforms (manifest may have OS-dependent ordering).
        self._files.sort(key=lambda t: t[0])

        self.first_t_ns: int = min(first_candidates) if first_candidates else 0
        self.last_t_ns: int = max(last_candidates) if last_candidates else 0
        self.total_lines: int = total_lines
        self.total_gaps: int = total_gaps

        if total_gaps > 0 and not allow_gaps:
            raise ValueError(
                f"fixture {fixture_dir} reports {total_gaps} sequence gap(s) "
                f"in the manifest; pass allow_gaps=True to proceed anyway"
            )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[RecordedEvent]:
        iters = [_iter_jsonl_gz(path) for _, path in self._files]
        # heapq.merge guarantees stable ordering — ties broken by
        # iterator index, which we've pinned via the file sort above.
        # The whole pipeline is therefore deterministic across runs.
        return heapq.merge(*iters, key=lambda e: e.t_recv_ns)

    # ------------------------------------------------------------------
    # Manifest accessors
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return str(self.manifest.get("session_id", ""))

    @property
    def recorder_version(self) -> str:
        return str(self.manifest.get("recorder_version", ""))

    @property
    def schema_version(self) -> int:
        return int(self.manifest.get("schema_version", 0))

    @property
    def sources(self) -> list[str]:
        return [src for src, _ in self._files]

    def file_metadata(self) -> list[dict[str, Any]]:
        """Per-file rows from the manifest (already filtered to the
        sources we'll iterate over)."""
        filtered_sources = {src for src, _ in self._files}
        return [
            entry
            for entry in self.manifest.get("files", [])
            if entry.get("source") in filtered_sources
        ]


__all__ = ["RecordedEvent", "SortedEventStream"]
