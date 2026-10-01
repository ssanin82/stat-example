"""Tests for app.backtest.simulation_storage (Phase F7.1, v1.5.93).

Coverage:
  * slugify_variant_name: normalisation + rejection
  * is_valid_slug: yes/no check
  * variant_paths: layout + path-traversal guard
  * VariantManifest: round-trip via load/save
  * list_variants / create_variant / delete_variant / rename_variant
  * variant_status state machine: fresh / running / done / failed / stale

All tests use a temp directory as ``repo_root`` so they never touch
the real ``backtesting/simulations/`` tree.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator

import pytest

from app.backtest import simulation_storage as ss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_repo(tmp_path: Path) -> Iterator[Path]:
    """Temp repo root with a stub recording dir already present."""
    sessions_dir = tmp_path / "backtesting" / "data" / "sessions"
    sessions_dir.mkdir(parents=True)
    yield tmp_path


@pytest.fixture
def stub_session(tmp_repo: Path) -> tuple[str, Path]:
    """Create a fake recording dir with a minimal bot_config_envfile."""
    session_id = "v1.5.99-260524-120000-prod.okx.ton.usdt.perp"
    session_dir = tmp_repo / "backtesting" / "data" / "sessions" / session_id
    session_dir.mkdir()
    (session_dir / "bot_config_envfile.txt").write_text(
        "# stub recorded env\n"
        "EXCHANGE=okx\n"
        "SYMBOL=TON-USDT-SWAP\n"
        "QUOTE_NOTIONAL_USD=10.0\n",
        encoding="utf-8",
    )
    return session_id, session_dir


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

class TestSlugify:
    @pytest.mark.parametrize("raw, expected", [
        ("microprice gate off", "microprice_gate_off"),
        ("Wide  Bands", "wide_bands"),
        ("v1_widen", "v1_widen"),
        ("BEHIND_TOUCH=3.0", "behind_touch3_0"),
        ("  trimmed  ", "trimmed"),
        ("with-dashes-ok", "with-dashes-ok"),
        ("mixed_-_separators", "mixed_separators"),
        ("a", "a"),
        ("UPPER_lower_123", "upper_lower_123"),
    ])
    def test_slugify_normalises(self, raw: str, expected: str) -> None:
        assert ss.slugify_variant_name(raw) == expected

    @pytest.mark.parametrize("raw", [
        "",
        "   ",
        "...",
        "!!!",
        " __ ",
    ])
    def test_slugify_rejects_empty_result(self, raw: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            ss.slugify_variant_name(raw)

    def test_slugify_rejects_overlong(self) -> None:
        with pytest.raises(ValueError, match="max"):
            ss.slugify_variant_name("a" * 100)

    def test_slugify_rejects_non_string(self) -> None:
        with pytest.raises(ValueError, match="string"):
            ss.slugify_variant_name(42)  # type: ignore[arg-type]

    @pytest.mark.parametrize("slug, valid", [
        ("ok", True),
        ("ok_too", True),
        ("with-dash", True),
        ("UPPER", False),
        ("under_", False),
        ("_under", False),
        ("-dash", False),
        ("dash-", False),
        ("", False),
        ("a" * 65, False),
    ])
    def test_is_valid_slug(self, slug: str, valid: bool) -> None:
        assert ss.is_valid_slug(slug) is valid


# ---------------------------------------------------------------------------
# Path resolution + traversal guard
# ---------------------------------------------------------------------------

class TestVariantPaths:
    def test_paths_layout(self, tmp_repo: Path) -> None:
        p = ss.variant_paths("session_a", "var_x", repo_root=tmp_repo)
        assert p.session_id == "session_a"
        assert p.variant_slug == "var_x"
        assert p.root == (tmp_repo / "backtesting" / "simulations" / "session_a" / "var_x").resolve()
        assert p.manifest == p.root / "manifest.json"
        assert p.config_delta == p.root / "config_delta.env"
        assert p.config_full == p.root / "config_full.env"
        assert p.lock == p.root / "simulation.lock"
        assert p.stdout == p.root / "stdout.log"
        assert p.stderr == p.root / "stderr.log"
        assert p.results == p.root / "results"

    def test_rejects_unsanitised_session_id(self, tmp_repo: Path) -> None:
        for bad in ["", "..", "../escape", "with/slash", "with\\back"]:
            with pytest.raises(ValueError):
                ss.variant_paths(bad, "ok", repo_root=tmp_repo)

    def test_rejects_unslugged_variant_name(self, tmp_repo: Path) -> None:
        with pytest.raises(ValueError, match="canonical slug"):
            ss.variant_paths("session_a", "Has Space", repo_root=tmp_repo)


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------

class TestManifest:
    def test_save_then_load_round_trips(self, tmp_repo: Path) -> None:
        paths = ss.variant_paths("session_a", "var_x", repo_root=tmp_repo)
        original = ss.VariantManifest(
            name="var_x",
            label="Friendly Name",
            base_session="session_a",
            created_at_utc="2026-05-23T00:00:00+00:00",
            bot_version_at_creation="1.5.93",
            base_config_sha256="abc123",
            status="fresh",
        )
        ss.save_manifest(paths, original)
        loaded = ss.load_manifest(paths)
        assert loaded == original

    def test_load_tolerates_extra_keys(self, tmp_repo: Path) -> None:
        """Forward-compat: a future manifest with extra fields must
        still load against the current dataclass (extras silently
        dropped). This is what lets us add fields without breaking
        old variants on disk."""
        paths = ss.variant_paths("session_a", "var_x", repo_root=tmp_repo)
        paths.root.mkdir(parents=True)
        paths.manifest.write_text(
            json.dumps({
                "name": "var_x",
                "label": "v",
                "base_session": "session_a",
                "created_at_utc": "2026-05-23T00:00:00+00:00",
                "bot_version_at_creation": "1.5.93",
                "base_config_sha256": "abc",
                "status": "done",
                "schema_version": 1,
                "future_field_we_dont_know_yet": "ignored",
            }),
            encoding="utf-8",
        )
        loaded = ss.load_manifest(paths)
        assert loaded.name == "var_x"
        assert loaded.status == "done"


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

class TestCreate:
    def test_create_variant_scaffolds_layout(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id,
            "My First Variant",
            base_session_dir=session_dir,
            bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        assert paths.variant_slug == "my_first_variant"
        assert paths.root.is_dir()
        assert paths.results.is_dir()
        assert paths.manifest.is_file()
        assert paths.config_delta.is_file()
        # config_full deferred to first run.
        assert not paths.config_full.exists()
        # No lock at creation.
        assert not paths.lock.exists()
        # Manifest sanity.
        m = ss.load_manifest(paths)
        assert m.name == "my_first_variant"
        assert m.label == "My First Variant"
        assert m.base_session == session_id
        assert m.bot_version_at_creation == "1.5.93"
        assert m.status == "fresh"
        assert len(m.base_config_sha256) == 64  # hex sha256

    def test_create_variant_refuses_existing_slug(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        with pytest.raises(FileExistsError, match="alpha"):
            ss.create_variant(
                session_id, "alpha",
                base_session_dir=session_dir, bot_version="1.5.93",
                repo_root=tmp_repo,
            )

    def test_create_variant_refuses_missing_base_envfile(
        self,
        tmp_repo: Path,
    ) -> None:
        # Session dir exists but no bot_config_envfile.txt
        session_id = "broken-session"
        session_dir = tmp_repo / "backtesting" / "data" / "sessions" / session_id
        session_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="bot_config_envfile"):
            ss.create_variant(
                session_id, "alpha",
                base_session_dir=session_dir, bot_version="1.5.93",
                repo_root=tmp_repo,
            )


class TestList:
    def test_list_variants_empty_when_no_session(self, tmp_repo: Path) -> None:
        assert ss.list_variants("nonexistent_session", repo_root=tmp_repo) == []

    def test_list_variants_sorted_and_filtered(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        for label in ["zebra", "alpha", "middle"]:
            ss.create_variant(
                session_id, label,
                base_session_dir=session_dir, bot_version="1.5.93",
                repo_root=tmp_repo,
            )
        # Drop a non-slug junk directory; it should be filtered out.
        junk = (
            tmp_repo / "backtesting" / "simulations" / session_id / "Bad Junk!"
        )
        junk.mkdir()
        # And a stray file (not a directory) — also filtered.
        (
            tmp_repo / "backtesting" / "simulations" / session_id / "stray.txt"
        ).write_text("x", encoding="utf-8")
        slugs = ss.list_variants(session_id, repo_root=tmp_repo)
        assert slugs == ["alpha", "middle", "zebra"]


class TestDelete:
    def test_delete_variant_removes_tree(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        ss.create_variant(
            session_id, "to_delete",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.delete_variant(session_id, "to_delete", repo_root=tmp_repo)
        paths = ss.variant_paths(session_id, "to_delete", repo_root=tmp_repo)
        assert not paths.root.exists()

    def test_delete_variant_idempotent(self, tmp_repo: Path) -> None:
        # Should silently succeed for nonexistent.
        ss.delete_variant(
            "v1.5.99-260524-120000-prod.okx.ton.usdt.perp", "ghost",
            repo_root=tmp_repo,
        )

    def test_delete_variant_refuses_during_run(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        # v1.5.135: "active simulation lock" now means PID-alive, not
        # just lock-file-present. mark_running() writes the test
        # process's PID, which IS alive — so this still raises.
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "locked",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        with pytest.raises(RuntimeError, match="lock"):
            ss.delete_variant(session_id, "locked", repo_root=tmp_repo)

    def test_delete_variant_auto_clears_stale_lock(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # v1.5.135 (Codex bug #1): a stale lock (dead PID) used to
        # permanently block deletion; the fix auto-cleans it.
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "ghost",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        # Simulate the wrapper having crashed: its PID is now dead.
        monkeypatch.setattr(ss, "is_pid_alive", lambda pid: False)
        ss.delete_variant(session_id, "ghost", repo_root=tmp_repo)
        assert not paths.root.exists()

    def test_delete_variant_auto_clears_empty_lock(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        # v1.5.135 (Codex bug #1): an empty/corrupt lock file (no PID
        # to verify) is treated as stale — same diagnosable state as
        # a dead-PID lock, since there's no way to prove ownership.
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "empty",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        paths.lock.touch()  # bare empty file (pre-v1.5.96 style)
        ss.delete_variant(session_id, "empty", repo_root=tmp_repo)
        assert not paths.root.exists()


class TestRename:
    def test_rename_directory_and_manifest(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        new_paths = ss.rename_variant(
            session_id, "alpha", "Beta Version",
            repo_root=tmp_repo,
        )
        assert new_paths.variant_slug == "beta_version"
        assert new_paths.root.is_dir()
        m = ss.load_manifest(new_paths)
        assert m.name == "beta_version"
        assert m.label == "Beta Version"
        # Old dir gone.
        old = ss.variant_paths(session_id, "alpha", repo_root=tmp_repo)
        assert not old.root.exists()

    def test_rename_to_same_slug_updates_label_only(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        # Different friendly name that slugifies to the SAME slug
        # ("ALPHA" → "alpha"). The rename path should detect the
        # equal-slug case and just refresh the manifest.label.
        ss.rename_variant(
            session_id, "alpha", "ALPHA",
            repo_root=tmp_repo,
        )
        paths = ss.variant_paths(session_id, "alpha", repo_root=tmp_repo)
        m = ss.load_manifest(paths)
        assert m.name == "alpha"
        assert m.label == "ALPHA"

    def test_rename_refuses_existing_target(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.create_variant(
            session_id, "beta",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        with pytest.raises(FileExistsError):
            ss.rename_variant(
                session_id, "alpha", "Beta",
                repo_root=tmp_repo,
            )

    def test_rename_refuses_during_active_run(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        # v1.5.135: same PID-aware semantics as delete. A live lock
        # (this test process's PID) blocks the rename.
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        with pytest.raises(RuntimeError, match="lock"):
            ss.rename_variant(
                session_id, "alpha", "Beta",
                repo_root=tmp_repo,
            )

    def test_rename_auto_clears_stale_lock(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # v1.5.135 (Codex bug #1): rename used to be stuck behind a
        # crashed-wrapper stale lock too. Fix auto-clears it.
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        monkeypatch.setattr(ss, "is_pid_alive", lambda pid: False)
        new_paths = ss.rename_variant(
            session_id, "alpha", "Beta",
            repo_root=tmp_repo,
        )
        assert new_paths.variant_slug == "beta"
        assert not paths.root.exists()


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_fresh_after_creation(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        assert ss.variant_status(paths) == "fresh"

    def test_status_running_when_lock_present(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        assert ss.variant_status(paths) == "running"
        assert paths.lock.exists()

    def test_status_done_after_successful_run(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        # Simulator wrote something to results/.
        (paths.results / "fills.jsonl.gz").write_bytes(b"\x1f\x8b")  # stub gzip
        ss.mark_done(paths, duration_s=12.5, exit_code=0)
        assert not paths.lock.exists()
        assert ss.variant_status(paths) == "done"

    def test_status_failed_after_non_zero_exit(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        ss.mark_done(paths, duration_s=3.0, exit_code=2)
        assert ss.variant_status(paths) == "failed"

    def test_status_stale_after_delta_edit(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        (paths.results / "fills.jsonl.gz").write_bytes(b"\x1f\x8b")
        # The done manifest is written AT or AFTER results/ mtime.
        # We need a results mtime that's strictly earlier than the
        # delta mtime to test "stale".
        old_time = time.time() - 60
        os.utime(paths.results / "fills.jsonl.gz", (old_time, old_time))
        os.utime(paths.results, (old_time, old_time))
        ss.mark_done(paths, duration_s=1.0, exit_code=0)
        # Touch the delta NOW — newer than results.
        paths.config_delta.write_text(
            paths.config_delta.read_text(encoding="utf-8") + "\nQUOTE_NOTIONAL_USD=20.0\n",
            encoding="utf-8",
        )
        assert ss.variant_status(paths) == "stale"

    def test_status_unknown_for_corrupt_manifest(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        paths.manifest.write_text("{ not valid json", encoding="utf-8")
        assert ss.variant_status(paths) == "unknown"

    def test_status_demotes_to_failed_on_stale_lock(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # v1.5.135 (Codex bug #1): pre-fix, a crashed wrapper's stale
        # lock left the variant permanently "running". The fix
        # self-heals: stale lock is cleared, and because the manifest
        # still says "running" (mark_running set it before the
        # wrapper crashed), it's demoted to "failed" with a
        # synthetic crash-message so the UI shows a diagnosable
        # state instead of a stuck "running".
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        monkeypatch.setattr(ss, "is_pid_alive", lambda pid: False)
        # Pre-fix: returns "running" indefinitely. Post-fix: detects
        # stale lock, demotes manifest, returns "failed".
        assert ss.variant_status(paths) == "failed"
        assert not paths.lock.exists()  # auto-cleared
        m = ss.load_manifest(paths)
        assert m.status == "failed"
        assert m.error_message is not None
        assert "stale lock" in m.error_message.lower()

    def test_mark_done_writes_manifest_before_unlinking_lock(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # v1.5.135 (Codex bug #2): if manifest save fails between
        # the unlink and the write (pre-fix order), the variant
        # ends up in a stuck state — lock gone but no terminal
        # status persisted. The fix orders save BEFORE unlink, so
        # a save failure leaves the lock intact (diagnosable).
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        # Inject a save failure.
        orig_save = ss.save_manifest

        def boom(*_a, **_k):
            raise OSError("simulated disk full")

        monkeypatch.setattr(ss, "save_manifest", boom)
        with pytest.raises(OSError, match="disk full"):
            ss.mark_done(paths, duration_s=1.0, exit_code=0)
        # Lock MUST still be present. The pre-fix sequence would have
        # unlinked it first, leaving an unrecoverable state.
        assert paths.lock.exists()
        # Restore + retry; the second attempt should succeed cleanly.
        monkeypatch.setattr(ss, "save_manifest", orig_save)
        ss.mark_done(paths, duration_s=1.0, exit_code=0)
        assert not paths.lock.exists()
        assert ss.load_manifest(paths).status == "done"

    def test_mark_done_caches_results_mtime(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        # v1.5.135 (Codex bug #8): mark_done() records the newest
        # results/ mtime into the manifest so variant_status() can
        # skip the directory rescan on subsequent reads.
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        (paths.results / "fills.jsonl.gz").write_bytes(b"\x1f\x8b")
        # Capture the actual results-file mtime so we can compare
        # against what mark_done caches.
        expected = (paths.results / "fills.jsonl.gz").stat().st_mtime
        ss.mark_done(paths, duration_s=1.0, exit_code=0)
        m = ss.load_manifest(paths)
        assert m.results_mtime is not None
        # Cached value should equal the newest file's mtime.
        assert abs(m.results_mtime - expected) < 0.001

    def test_status_uses_cached_results_mtime_when_present(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        # v1.5.135 (Codex bug #8): the stale-check fast path uses
        # m.results_mtime instead of walking results/.
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        (paths.results / "fills.jsonl.gz").write_bytes(b"\x1f\x8b")
        # Force results mtime into the past so we can test the stale
        # comparison against a delta edited NOW.
        old_time = time.time() - 60
        os.utime(paths.results / "fills.jsonl.gz", (old_time, old_time))
        os.utime(paths.results, (old_time, old_time))
        ss.mark_done(paths, duration_s=1.0, exit_code=0)
        # Manifest now carries old_time as results_mtime.
        m = ss.load_manifest(paths)
        assert m.results_mtime is not None and m.results_mtime <= old_time + 1
        # Touch the delta NOW → newer than cached results_mtime →
        # variant_status reports "stale" without rescanning results/.
        paths.config_delta.write_text(
            paths.config_delta.read_text(encoding="utf-8")
            + "\nQUOTE_NOTIONAL_USD=20.0\n",
            encoding="utf-8",
        )
        assert ss.variant_status(paths) == "stale"

    def test_status_falls_back_to_scan_for_legacy_manifest(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        # v1.5.135 (Codex bug #8): manifests written before the
        # results_mtime field existed (results_mtime is None) fall
        # back to the live directory scan. Confirms backward compat.
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.93",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        (paths.results / "fills.jsonl.gz").write_bytes(b"\x1f\x8b")
        ss.mark_done(paths, duration_s=1.0, exit_code=0)
        # Simulate a legacy manifest by clearing the cached field.
        m = ss.load_manifest(paths)
        m.results_mtime = None
        ss.save_manifest(paths, m)
        # variant_status falls back to the scan → still reports "done".
        assert ss.variant_status(paths) == "done"


# ---------------------------------------------------------------------------
# materialize_full_config (Phase F7.3)
# ---------------------------------------------------------------------------

class TestMaterializeFullConfig:
    """Acceptance tests for the delta → config_full.env merge.

    The materialised file is the single artefact the bot's settings
    loader reads at simulator launch — getting this right is what
    keeps non-destructive variants working.
    """

    def _make_variant(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> ss.VariantPaths:
        session_id, session_dir = stub_session
        return ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.95",
            repo_root=tmp_repo,
        )

    def test_empty_delta_emits_base_verbatim(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        # The seeded delta is comment-only → no overrides.
        out = ss.materialize_full_config(paths, repo_root=tmp_repo)
        assert out == paths.config_full
        full = out.read_text(encoding="utf-8")
        # Every base KEY=VAL line round-trips.
        assert "EXCHANGE=okx" in full
        assert "SYMBOL=TON-USDT-SWAP" in full
        assert "QUOTE_NOTIONAL_USD=10.0" in full
        # No "variant overrides" header when delta has no keys.
        assert "variant overrides" not in full

    def test_delta_overrides_existing_key(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        paths.config_delta.write_text(
            "# operator: widen the bands\n"
            "QUOTE_NOTIONAL_USD=20.0\n",
            encoding="utf-8",
        )
        ss.materialize_full_config(paths, repo_root=tmp_repo)
        full = paths.config_full.read_text(encoding="utf-8")
        # New value present.
        assert "QUOTE_NOTIONAL_USD=20.0" in full
        # Old value not present (replaced in place).
        assert "QUOTE_NOTIONAL_USD=10.0" not in full
        # Other base keys still present.
        assert "SYMBOL=TON-USDT-SWAP" in full

    def test_delta_appends_new_keys(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        paths.config_delta.write_text(
            "NEW_KNOB=42\n"
            "ANOTHER_KNOB=true\n",
            encoding="utf-8",
        )
        ss.materialize_full_config(paths, repo_root=tmp_repo)
        full = paths.config_full.read_text(encoding="utf-8")
        # Header banner present for new keys.
        assert "variant overrides" in full
        assert "NEW_KNOB=42" in full
        assert "ANOTHER_KNOB=true" in full
        # Order preserved (NEW_KNOB before ANOTHER_KNOB).
        assert full.index("NEW_KNOB=") < full.index("ANOTHER_KNOB=")

    def test_delta_mixed_override_and_new_keys(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        paths.config_delta.write_text(
            "QUOTE_NOTIONAL_USD=15.5\n"
            "NEW_KNOB=hello\n",
            encoding="utf-8",
        )
        ss.materialize_full_config(paths, repo_root=tmp_repo)
        full = paths.config_full.read_text(encoding="utf-8")
        assert "QUOTE_NOTIONAL_USD=15.5" in full
        assert "QUOTE_NOTIONAL_USD=10.0" not in full
        # Override stays in original position (above the header).
        override_idx = full.index("QUOTE_NOTIONAL_USD=15.5")
        header_idx = full.index("variant overrides")
        new_key_idx = full.index("NEW_KNOB=hello")
        assert override_idx < header_idx < new_key_idx

    def test_inline_comment_preserved_on_override(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        # Re-write base with an inline-comment row so we can verify it
        # survives.
        (session_dir / "bot_config_envfile.txt").write_text(
            "EXCHANGE=okx\n"
            "QUOTE_NOTIONAL_USD=10.0 # production minimum\n",
            encoding="utf-8",
        )
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.95",
            repo_root=tmp_repo,
        )
        paths.config_delta.write_text(
            "QUOTE_NOTIONAL_USD=20.0\n", encoding="utf-8"
        )
        ss.materialize_full_config(paths, repo_root=tmp_repo)
        full = paths.config_full.read_text(encoding="utf-8")
        assert "QUOTE_NOTIONAL_USD=20.0 # production minimum" in full

    def test_missing_delta_treated_as_empty(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        paths.config_delta.unlink()
        ss.materialize_full_config(paths, repo_root=tmp_repo)
        full = paths.config_full.read_text(encoding="utf-8")
        assert "EXCHANGE=okx" in full

    def test_missing_base_env_raises(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.95",
            repo_root=tmp_repo,
        )
        (session_dir / "bot_config_envfile.txt").unlink()
        with pytest.raises(FileNotFoundError, match="bot_config_envfile"):
            ss.materialize_full_config(paths, repo_root=tmp_repo)

    def test_atomic_write_no_tmp_left_behind(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        ss.materialize_full_config(paths, repo_root=tmp_repo)
        # The tmp suffix used during the atomic write.
        tmp = paths.config_full.with_suffix(".env.tmp")
        assert not tmp.exists()
        assert paths.config_full.exists()

    def test_round_trip_through_parser_idempotent(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        """Parsing the materialised config and the base+delta merge
        in-memory must agree. Catches any drift between the
        materialisation algorithm and the bot's settings loader."""
        paths = self._make_variant(tmp_repo, stub_session)
        paths.config_delta.write_text(
            "QUOTE_NOTIONAL_USD=20.0\nNEW_KEY=v\n", encoding="utf-8"
        )
        ss.materialize_full_config(paths, repo_root=tmp_repo)
        parsed = ss._parse_env_kv(
            paths.config_full.read_text(encoding="utf-8")
        )
        assert parsed["EXCHANGE"] == "okx"
        assert parsed["QUOTE_NOTIONAL_USD"] == "20.0"
        assert parsed["NEW_KEY"] == "v"


class TestMarkDoneErrorMessage:
    """``mark_done(error_message=...)`` round-trips through the
    manifest and is cleared on the next successful run."""

    def test_failure_stores_error_message(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.95",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        ss.mark_done(
            paths,
            duration_s=1.5,
            exit_code=2,
            error_message="bot crashed during warmup",
        )
        m = ss.load_manifest(paths)
        assert m.status == "failed"
        assert m.last_run_exit_code == 2
        assert m.error_message == "bot crashed during warmup"

    def test_success_clears_prior_error_message(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        session_id, session_dir = stub_session
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.95",
            repo_root=tmp_repo,
        )
        ss.mark_running(paths)
        ss.mark_done(
            paths, duration_s=1.0, exit_code=1, error_message="oops",
        )
        ss.mark_running(paths)
        ss.mark_done(paths, duration_s=2.0, exit_code=0)
        m = ss.load_manifest(paths)
        assert m.status == "done"
        assert m.error_message is None


# ---------------------------------------------------------------------------
# Lock-file PID payload (Phase F7.4)
# ---------------------------------------------------------------------------

class TestLockInfo:
    """``mark_running`` writes ``{pid, started_at_utc}`` JSON, and
    ``read_lock_info`` round-trips it. The /cancel endpoint relies on
    the PID landing in the lock file synchronously."""

    def _make_variant(
        self, tmp_repo: Path, stub_session: tuple[str, Path]
    ) -> ss.VariantPaths:
        session_id, session_dir = stub_session
        return ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.96",
            repo_root=tmp_repo,
        )

    def test_mark_running_writes_pid_payload(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        ss.mark_running(paths)
        info = ss.read_lock_info(paths)
        assert info is not None
        assert info["pid"] == os.getpid()
        assert isinstance(info["started_at_utc"], str)
        # ISO-8601-ish — at least the year matches.
        assert info["started_at_utc"].startswith("20")

    def test_mark_running_with_explicit_pid(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        ss.mark_running(paths, pid=99999)
        info = ss.read_lock_info(paths)
        assert info is not None
        assert info["pid"] == 99999

    def test_read_lock_info_missing_returns_none(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        assert ss.read_lock_info(paths) is None

    def test_read_lock_info_empty_lock_returns_none_pid(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        """A zero-byte lock (older callers, or torn write) is
        compatible: lock exists → variant is running, but no pid is
        available to signal."""
        paths = self._make_variant(tmp_repo, stub_session)
        paths.lock.touch()
        info = ss.read_lock_info(paths)
        assert info is not None
        assert info["pid"] is None

    def test_read_lock_info_unparseable_returns_none_pid(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        paths.lock.write_text("not json at all", encoding="utf-8")
        info = ss.read_lock_info(paths)
        assert info is not None
        assert info["pid"] is None

    def test_read_lock_info_rejects_zero_pid(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        """pid=0 means "no pid" semantically — most OSes won't return
        a real 0 PID and several APIs treat 0 as "signal all
        processes" which would be catastrophic. Defensive reject."""
        paths = self._make_variant(tmp_repo, stub_session)
        paths.lock.write_text('{"pid": 0}', encoding="utf-8")
        info = ss.read_lock_info(paths)
        assert info is not None
        assert info["pid"] is None

    def test_read_lock_info_after_mark_done(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
    ) -> None:
        paths = self._make_variant(tmp_repo, stub_session)
        ss.mark_running(paths)
        ss.mark_done(paths, duration_s=1.0, exit_code=0)
        # Lock cleared.
        assert ss.read_lock_info(paths) is None


# ---------------------------------------------------------------------------
# is_pid_alive (Phase F7 / v1.5.121)
# ---------------------------------------------------------------------------

class TestIsPidAlive:
    """Cross-platform PID-liveness probe used by the wrapper's
    ghost-lock takeover logic."""

    def test_own_pid_is_alive(self) -> None:
        assert ss.is_pid_alive(os.getpid()) is True

    def test_zero_pid_not_alive(self) -> None:
        # POSIX signal-broadcast PID; never a real process — defensive
        # reject so we never accidentally treat it as alive.
        assert ss.is_pid_alive(0) is False

    def test_negative_pid_not_alive(self) -> None:
        assert ss.is_pid_alive(-1) is False

    def test_huge_unlikely_pid_not_alive(self) -> None:
        # PID 2^31 - 1 — OS ceiling, nothing real ever has it.
        assert ss.is_pid_alive(2_147_483_647) is False

    def test_non_int_pid_not_alive(self) -> None:
        # Defensive — read_lock_pid is supposed to filter, but the
        # function should still be robust to bad input.
        assert ss.is_pid_alive("abc") is False  # type: ignore[arg-type]
        assert ss.is_pid_alive(None) is False  # type: ignore[arg-type]
