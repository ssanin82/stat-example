"""Tests for ``scripts/backtest/run_variant.py`` (Phase F7.3, v1.5.95).

End-to-end wrapper tests run a stub subprocess (``python -c ...``)
that exits with a chosen code after writing chosen content to its
captured stdout/stderr. This exercises:

  * Materialisation of ``config_full.env``.
  * Lock acquisition + manifest ``running`` transition.
  * Mark-done with the subprocess's actual exit code.
  * Error-message scraping from stderr.log on non-zero exit.
  * Refusal to spawn when the variant is already locked.

We DON'T exercise the real ``replay.py`` here — that's covered by
the existing replay test suite. The wrapper's only responsibilities
are the lifecycle + path arithmetic above.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterator

import pytest

from app.backtest import simulation_storage as ss


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RUN_VARIANT_PATH = (
    _PROJECT_ROOT / "scripts" / "backtest" / "run_variant.py"
)


def _import_run_variant():
    """Load run_variant.py as a module so its ``main()`` is callable
    from pytest without spawning a subprocess for the wrapper itself.

    Mutating ``sys.path`` + ``importlib`` is the cleanest way to import
    a script-style module that lives outside the package tree.
    """
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(
        "run_variant_under_test", _RUN_VARIANT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tmp_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A temp repo root the storage module resolves against.

    ``simulation_storage`` resolves paths relative to its own file's
    location (``app/backtest/simulation_storage.py`` → repo root via
    three ``parent`` hops). To redirect it at a temp tree, we patch
    its ``_repo_root_default`` shim.
    """
    sessions_dir = tmp_path / "backtesting" / "data" / "sessions"
    sessions_dir.mkdir(parents=True)
    monkeypatch.setattr(ss, "_repo_root_default", lambda: tmp_path)
    yield tmp_path


@pytest.fixture
def stub_session(tmp_repo: Path) -> tuple[str, Path]:
    session_id = "v1.5.99-260524-120000-prod.okx.ton.usdt.perp"
    session_dir = tmp_repo / "backtesting" / "data" / "sessions" / session_id
    session_dir.mkdir()
    (session_dir / "bot_config_envfile.txt").write_text(
        "EXCHANGE=okx\nSYMBOL=TON-USDT-SWAP\nQUOTE_NOTIONAL_USD=10.0\n",
        encoding="utf-8",
    )
    # The wrapper insists on a manifest.json in the recording dir.
    (session_dir / "manifest.json").write_text(
        '{"session_id": "stub", "events_count": 0}', encoding="utf-8"
    )
    return session_id, session_dir


@pytest.fixture
def stub_variant(
    tmp_repo: Path, stub_session: tuple[str, Path]
) -> tuple[str, str, ss.VariantPaths]:
    session_id, session_dir = stub_session
    paths = ss.create_variant(
        session_id, "alpha",
        base_session_dir=session_dir, bot_version="1.5.95",
    )
    return session_id, "alpha", paths


def _stub_replay_args(
    *, exit_code: int, stdout: str = "", stderr: str = "",
) -> tuple[str, list[str]]:
    """Build a tiny inline Python program that pretends to be replay.py.

    It writes ``stdout`` / ``stderr`` to its respective streams and
    exits with ``exit_code``. Used in place of the real replay.py by
    monkey-patching the wrapper's subprocess command list.
    """
    program = (
        "import sys\n"
        f"sys.stdout.write({stdout!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({exit_code})\n"
    )
    return sys.executable, ["-c", program]


def _build_stub_bot_sqlite(db_path: Path) -> None:
    """Create a minimal ``bot.sqlite`` with the 9 forensic tables the
    F7.12.5 extract reads, each with one row. Mirrors the real bot's
    schema closely enough that ``colo_bot_db_extract`` extracts it
    without skipping.

    Only the ts columns the extractor filters on are required — the
    rest of the row can be sparse. ``ts`` values are ISO-8601 strings
    so the extractor's BETWEEN clause matches.
    """
    import sqlite3
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        # Schema mirrors the extractor's TABLE_SPECS list. One row
        # per table is enough — the test cares that the extract path
        # works, not about realistic data shapes.
        ts = "2026-05-23T12:00:00+00:00"
        tables = [
            ("orders", "ts_created"),
            ("fills", "ts_fill"),
            ("quote_decisions", "ts"),
            ("bot_events", "ts"),
            ("position_snapshots", "ts"),
            ("equity_snapshots", "ts"),
            ("soft_flatten_events", "ts_start"),
            ("tp_events", "ts_start"),
            ("exposure_bars", "ts_bar"),
        ]
        for name, ts_col in tables:
            cur.execute(f"CREATE TABLE {name} ({ts_col} TEXT, payload TEXT)")
            cur.execute(
                f"INSERT INTO {name} ({ts_col}, payload) VALUES (?, ?)",
                (ts, f"stub row for {name}"),
            )
        conn.commit()
    finally:
        conn.close()


def _run_wrapper_with_stub(
    mod,
    *,
    session_id: str,
    variant_slug: str,
    stub_exit_code: int,
    stub_stdout: str = "",
    stub_stderr: str = "",
    monkeypatch: pytest.MonkeyPatch,
    write_sqlite_at: Path | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Drive ``run_variant.main()`` with the real ``Popen`` but swap
    the replay command for a no-op stub.

    Approach: intercept ``subprocess.Popen`` inside the wrapper module
    and rewrite ``cmd[0:N]`` to the stub program before delegating to
    the real ``Popen``. Cleanest mock for "real lifecycle, fake work".

    F7.12.5: when ``write_sqlite_at`` is provided, ``fake_popen``
    also writes a minimal stub bot.sqlite there BEFORE returning
    the Popen handle. This simulates the real replay's side effect
    (the bot writing its SQLite) so the wrapper's post-run extract
    has something to read.
    """
    import subprocess as real_subprocess

    real_popen = real_subprocess.Popen
    stub_py, stub_args = _stub_replay_args(
        exit_code=stub_exit_code, stdout=stub_stdout, stderr=stub_stderr,
    )

    def fake_popen(cmd, **kw):  # type: ignore[no-untyped-def]
        if write_sqlite_at is not None:
            _build_stub_bot_sqlite(write_sqlite_at)
        new_cmd = [stub_py, *stub_args]
        return real_popen(new_cmd, **kw)

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    argv = [
        "--session", session_id,
        "--variant", variant_slug,
    ]
    if extra_args:
        argv += extra_args
    return mod.main(argv)


class TestRunVariantHappyPath:
    def test_zero_exit_marks_done_and_materialises(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id, slug, paths = stub_variant
        # Add an override so we can verify materialisation actually ran.
        paths.config_delta.write_text(
            "QUOTE_NOTIONAL_USD=20.0\n", encoding="utf-8",
        )
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id, variant_slug=slug,
            stub_exit_code=0, stub_stdout="ok\n",
            monkeypatch=monkeypatch,
            # Skip extract for this test — it's covered separately by
            # TestRunVariantExtract. Keeps this test focused on the
            # config materialisation + manifest lifecycle.
            extra_args=["--skip-extract"],
        )
        assert rc == 0
        # Materialised full config exists and reflects the delta.
        assert paths.config_full.is_file()
        full = paths.config_full.read_text(encoding="utf-8")
        assert "QUOTE_NOTIONAL_USD=20.0" in full
        # Lock cleared.
        assert not paths.lock.exists()
        # Manifest reflects done.
        m = ss.load_manifest(paths)
        assert m.status == "done"
        assert m.last_run_exit_code == 0
        assert m.last_run_duration_s is not None
        assert m.last_run_duration_s >= 0.0
        assert m.error_message is None
        # Stdout captured to disk.
        assert paths.stdout.read_text(encoding="utf-8") == "ok\n"

    def test_results_dir_cleaned_before_run(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id, slug, paths = stub_variant
        # Pre-populate results from a "previous run".
        stale = paths.results / "stale_artefact.bin"
        stale.write_bytes(b"\x00\x01")
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id, variant_slug=slug,
            stub_exit_code=0,
            monkeypatch=monkeypatch,
            extra_args=["--skip-extract"],
        )
        assert rc == 0
        assert not stale.exists()


class TestRunVariantExtract:
    """F7.12.5 — post-run extract of bot_db forensic tables."""

    def test_successful_run_produces_jsonl_gz_files(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Happy path: replay exits 0 + bot.sqlite present → wrapper
        runs the extract → all 9 JSONL.gz tables land under results/."""
        session_id, slug, paths = stub_variant
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id, variant_slug=slug,
            stub_exit_code=0,
            monkeypatch=monkeypatch,
            write_sqlite_at=paths.results / "bot.sqlite",
        )
        assert rc == 0
        # All 9 forensic tables should be extracted.
        expected = [
            "orders.jsonl.gz",
            "fills.jsonl.gz",
            "quote_decisions.jsonl.gz",
            "bot_events.jsonl.gz",
            "position_snapshots.jsonl.gz",
            "equity_snapshots.jsonl.gz",
            "soft_flatten_events.jsonl.gz",
            "tp_events.jsonl.gz",
            "exposure_bars.jsonl.gz",
        ]
        for name in expected:
            assert (paths.results / name).is_file(), (
                f"missing extract file: {name}"
            )
        # Verify the row landed — pick orders.jsonl.gz and gunzip-check.
        import gzip
        with gzip.open(paths.results / "orders.jsonl.gz", "rt") as f:
            lines = [line for line in f if line.strip()]
        assert len(lines) == 1
        assert "stub row for orders" in lines[0]
        # Status stays "done" + no error.
        m = ss.load_manifest(paths)
        assert m.status == "done"
        assert m.error_message is None

    def test_skip_extract_flag_omits_jsonl_gz(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id, slug, paths = stub_variant
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id, variant_slug=slug,
            stub_exit_code=0,
            monkeypatch=monkeypatch,
            write_sqlite_at=paths.results / "bot.sqlite",
            extra_args=["--skip-extract"],
        )
        assert rc == 0
        # bot.sqlite is present but no JSONL.gz extracts.
        assert (paths.results / "bot.sqlite").is_file()
        assert not (paths.results / "orders.jsonl.gz").exists()

    def test_failed_replay_skips_extract(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replay exits non-zero → wrapper does NOT run the extract
        (the SQLite may be partial / corrupt)."""
        session_id, slug, paths = stub_variant
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id, variant_slug=slug,
            stub_exit_code=3, stub_stderr="boom\n",
            monkeypatch=monkeypatch,
            write_sqlite_at=paths.results / "bot.sqlite",
        )
        assert rc == 3
        # No JSONL.gz despite bot.sqlite being present — extract
        # skipped because the replay failed.
        assert not (paths.results / "orders.jsonl.gz").exists()
        m = ss.load_manifest(paths)
        assert m.status == "failed"

    def test_missing_sqlite_emits_warning_but_keeps_done(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Replay succeeded but produced no bot.sqlite (stub case) —
        wrapper records the missing-DB note in error_message but
        keeps status=done. The replay itself succeeded."""
        session_id, slug, paths = stub_variant
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id, variant_slug=slug,
            stub_exit_code=0,
            monkeypatch=monkeypatch,
            # No write_sqlite_at — replay "succeeds" but writes no DB.
        )
        assert rc == 0
        m = ss.load_manifest(paths)
        assert m.status == "done"
        # error_message carries the extract gripe + retry hint.
        assert m.error_message is not None
        assert "no bot.sqlite" in m.error_message
        assert "--extract-only" in m.error_message

    def test_extract_only_skips_replay(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """--extract-only on an existing bot.sqlite runs only the
        extract; manifest status is not touched."""
        session_id, slug, paths = stub_variant
        # Simulate an earlier run that left a bot.sqlite.
        paths.results.mkdir(parents=True, exist_ok=True)
        _build_stub_bot_sqlite(paths.results / "bot.sqlite")
        # ``--extract-only`` should not invoke Popen at all.
        import subprocess as real_subprocess
        popen_called = {"n": 0}
        real_popen = real_subprocess.Popen

        def guarded_popen(cmd, **kw):  # type: ignore[no-untyped-def]
            popen_called["n"] += 1
            return real_popen(cmd, **kw)

        mod = _import_run_variant()
        monkeypatch.setattr(mod.subprocess, "Popen", guarded_popen)
        rc = mod.main([
            "--session", session_id, "--variant", slug, "--extract-only",
        ])
        assert rc == 0
        assert popen_called["n"] == 0, "extract-only should not spawn replay"
        # JSONL.gz files appeared from the extract.
        assert (paths.results / "orders.jsonl.gz").is_file()

    def test_extract_only_without_sqlite_returns_2(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_id, slug, _paths = stub_variant
        mod = _import_run_variant()
        rc = mod.main([
            "--session", session_id, "--variant", slug, "--extract-only",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "needs an existing" in err

    def test_extract_failure_keeps_status_done(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A corrupt SQLite at extract-time produces an error_message
        but the variant stays ``done`` (the replay itself
        succeeded). Operator can retry with ``--extract-only`` after
        the underlying issue is fixed."""
        session_id, slug, paths = stub_variant
        mod = _import_run_variant()
        # Write garbage at the expected SQLite path so the extractor
        # fails when it opens the file.
        def write_bad_sqlite(db_path: Path) -> None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path.write_bytes(b"not a sqlite database")

        # Reuse _run_wrapper_with_stub's structure but with a custom
        # SQLite writer.
        import subprocess as real_subprocess

        real_popen = real_subprocess.Popen
        stub_py, stub_args = _stub_replay_args(exit_code=0)

        def fake_popen(cmd, **kw):  # type: ignore[no-untyped-def]
            write_bad_sqlite(paths.results / "bot.sqlite")
            return real_popen([stub_py, *stub_args], **kw)

        monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
        rc = mod.main(["--session", session_id, "--variant", slug])
        assert rc == 0
        m = ss.load_manifest(paths)
        # Replay succeeded → status stays done. Extract error lives
        # in error_message so the operator sees it.
        assert m.status == "done"
        assert m.error_message is not None
        assert (
            "extract" in m.error_message.lower()
            or "sqlite" in m.error_message.lower()
        )
        assert "--extract-only" in m.error_message


class TestRunVariantFailurePath:
    def test_nonzero_exit_marks_failed_and_scrapes_stderr(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session_id, slug, paths = stub_variant
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id, variant_slug=slug,
            stub_exit_code=7,
            stub_stderr="Traceback: BOOM\n",
            monkeypatch=monkeypatch,
        )
        assert rc == 7
        assert not paths.lock.exists()
        m = ss.load_manifest(paths)
        assert m.status == "failed"
        assert m.last_run_exit_code == 7
        assert m.error_message is not None
        assert "BOOM" in m.error_message


class TestRunVariantGuards:
    def test_missing_variant_returns_2(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_id, _ = stub_session
        mod = _import_run_variant()
        rc = mod.main([
            "--session", session_id,
            "--variant", "nonexistent",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err

    def test_invalid_slug_returns_2(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_id, _ = stub_session
        mod = _import_run_variant()
        rc = mod.main([
            "--session", session_id,
            "--variant", "Bad Slug!",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "canonical slug form" in err

    def test_already_locked_returns_2(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """v1.5.121: refuses only when the lock's PID is GENUINELY
        alive. Dead PIDs are taken over (covered in
        test_ghost_lock_with_dead_pid_does_NOT_block_wrapper). Use
        the current test runner's PID as a stand-in for a
        guaranteed-alive different process — fixtures spawn the
        wrapper in-process via main(), so os.getpid() returns the
        wrapper's pid (the test's own). To get a DIFFERENT alive
        PID we use the test process's parent — pytest's parent — or
        simply ourselves with the self-PID short-circuit bypassed
        by writing a different live PID. Simplest robust choice:
        spawn a stub subprocess just for its alive PID."""
        import subprocess
        # A subprocess that lives for a few seconds — long enough for
        # the wrapper's check to read the lock and decide.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        try:
            session_id, slug, paths = stub_variant
            ss.mark_running(paths, pid=proc.pid)
            assert paths.lock.exists()
            mod = _import_run_variant()
            rc = mod.main([
                "--session", session_id,
                "--variant", slug,
            ])
            assert rc == 2
            err = capsys.readouterr().err
            assert "genuinely running" in err
            # Lock still held — wrapper didn't touch it.
            assert paths.lock.exists()
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_claim_token_match_proceeds_even_when_lock_pid_is_alive(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """v1.5.122 regression: the API-path claim-token check is
        authoritative and bypasses the PID-liveness probe. When the
        lock's claim_token matches BACKTEST_CLAIM_TOKEN, the wrapper
        proceeds regardless of what PID the lock contains or whether
        that PID is alive (Windows PID-recycling false-positive
        defence).
        """
        session_id, slug, paths = stub_variant
        # Plant a lock with a real-alive other PID (the test runner
        # process itself + 1 is risky — use a real subprocess to be
        # safe) AND a claim token that matches what the wrapper sees.
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        try:
            token = "matching-uuid-1234"
            ss.mark_running(paths, pid=proc.pid, claim_token=token)
            assert ss.read_lock_claim_token(paths) == token
            assert ss.is_pid_alive(proc.pid)  # sanity

            monkeypatch.setenv("BACKTEST_CLAIM_TOKEN", token)
            mod = _import_run_variant()
            rc = _run_wrapper_with_stub(
                mod,
                session_id=session_id,
                variant_slug=slug,
                stub_exit_code=0,
                monkeypatch=monkeypatch,
                extra_args=["--skip-extract"],
            )
            assert rc == 0
            m = ss.load_manifest(paths)
            assert m.status == "done"
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_claim_token_mismatch_with_live_pid_refuses(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """v1.5.135 (Codex bug #4): when the lock's claim_token is from
        a DIFFERENT /run AND that other wrapper's PID is still ALIVE,
        the new wrapper must REFUSE to take over (exit 2).

        Pre-v1.5.135 the wrapper logged a warning and unconditionally
        took over on any token mismatch. That meant a duplicate /run
        (operator double-click, browser retry, racing API call, etc.)
        could trample a legitimately running variant. The fix:
        token mismatch is fail-closed unless the lock's PID is also
        demonstrably dead (handled by the companion test below)."""
        session_id, slug, paths = stub_variant
        import subprocess
        # Lock has alive PID + STALE token (from a previous /run).
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
        )
        try:
            stale_token = "stale-uuid-from-previous-run"
            ss.mark_running(paths, pid=proc.pid, claim_token=stale_token)
            # Wrapper has a DIFFERENT token in its env.
            monkeypatch.setenv("BACKTEST_CLAIM_TOKEN", "current-uuid-NEW")
            mod = _import_run_variant()
            rc = _run_wrapper_with_stub(
                mod,
                session_id=session_id,
                variant_slug=slug,
                stub_exit_code=0,
                monkeypatch=monkeypatch,
                extra_args=["--skip-extract"],
            )
            assert rc == 2  # refused — live wrapper still owns this slot
            # Lock should NOT have been touched.
            assert paths.lock.exists()
            m = ss.load_manifest(paths)
            # status stays "running" — the actual live wrapper is the
            # owner, and we didn't write a terminal state on its behalf.
            assert m.status == "running"
        finally:
            proc.terminate()
            proc.wait(timeout=5)

    def test_claim_token_mismatch_with_dead_pid_takes_over(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """v1.5.135 (Codex bug #4) companion to the live-PID refuse
        test: when the lock has a stale token AND the recorded PID is
        DEAD, the new wrapper takes over the abandoned lock and
        completes normally. Without this path, every crashed-wrapper
        scenario would leave the variant un-runnable until manual
        filesystem cleanup."""
        session_id, slug, paths = stub_variant
        # Lock has DEAD PID + stale token. Use the well-known
        # guaranteed-dead PID pattern (2^31 - 1) the ghost-lock
        # test already uses.
        dead_pid = (1 << 31) - 1
        ss.mark_running(
            paths,
            pid=dead_pid,
            claim_token="stale-uuid-from-previous-run",
        )
        monkeypatch.setenv("BACKTEST_CLAIM_TOKEN", "current-uuid-NEW")
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id,
            variant_slug=slug,
            stub_exit_code=0,
            monkeypatch=monkeypatch,
            extra_args=["--skip-extract"],
        )
        assert rc == 0  # took over and completed
        m = ss.load_manifest(paths)
        assert m.status == "done"

    def test_ghost_lock_with_dead_pid_does_NOT_block_wrapper(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """v1.5.121 regression: a stale lock with a PID that's no
        longer alive should be transparently taken over by a new
        wrapper run, not refused with 'already running'. The
        2026-05-25 incident: ghost lock from a pre-v1.5.120 crashed
        wrapper kept blocking every subsequent Run attempt because
        the v1.5.116 self-PID check was strict-equality and didn't
        verify whether the lock's PID was actually alive."""
        session_id, slug, paths = stub_variant
        # Plant a ghost lock with a PID that's guaranteed-dead.
        # PID 2^31 - 1 is the OS ceiling — nothing real ever has it.
        ss.mark_running(paths, pid=2_147_483_647)
        assert paths.lock.exists()
        mod = _import_run_variant()
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id,
            variant_slug=slug,
            stub_exit_code=0,
            monkeypatch=monkeypatch,
            extra_args=["--skip-extract"],
        )
        # Wrapper should have taken over the lock and completed.
        assert rc == 0
        # Lock cleaned up by mark_done.
        assert not paths.lock.exists()
        m = ss.load_manifest(paths)
        assert m.status == "done"

    def test_own_pid_in_lock_does_NOT_block_wrapper(
        self,
        tmp_repo: Path,
        stub_variant: tuple[str, str, ss.VariantPaths],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """v1.5.116 regression: the F7.4 /run endpoint writes the
        lock with the SPAWNED wrapper's PID before the wrapper has
        reached its variant_status check. Without the self-PID
        tolerance, the wrapper would see its own lock and refuse to
        run with "already running" — the 2026-05-24 ghost-lock
        symptom.

        This test plants a lock with the test process's own PID
        (which IS the wrapper's PID in test, since we invoke main()
        in-process) and verifies the wrapper proceeds past the check
        without bailing.
        """
        session_id, slug, paths = stub_variant
        # Pre-write the lock with OUR pid — simulates the /run
        # endpoint's pre-emptive write.
        ss.mark_running(paths, pid=os.getpid())
        assert paths.lock.exists()
        mod = _import_run_variant()
        # Stub replay subprocess so the wrapper completes without
        # actually doing replay work.
        rc = _run_wrapper_with_stub(
            mod,
            session_id=session_id,
            variant_slug=slug,
            stub_exit_code=0,
            stub_stdout="ok\n",
            monkeypatch=monkeypatch,
            extra_args=["--skip-extract"],
        )
        # Wrapper should have proceeded past the lock check + run
        # to completion (rc 0). If the self-PID check is broken,
        # we'd get rc 2 with "already running" on stderr.
        assert rc == 0
        # And mark_done should have been called, clearing the lock.
        assert not paths.lock.exists()
        m = ss.load_manifest(paths)
        assert m.status == "done"

    def test_missing_session_returns_2(
        self,
        tmp_repo: Path,
        stub_session: tuple[str, Path],
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session_id, session_dir = stub_session
        # Create a variant against this session...
        paths = ss.create_variant(
            session_id, "alpha",
            base_session_dir=session_dir, bot_version="1.5.95",
        )
        # ...then nuke the session dir to simulate a stale variant.
        import shutil
        shutil.rmtree(session_dir)
        mod = _import_run_variant()
        rc = mod.main([
            "--session", session_id,
            "--variant", "alpha",
        ])
        assert rc == 2
        err = capsys.readouterr().err
        assert "session" in err.lower()
        # The variant was never marked running because validation failed
        # before the mark_running call.
        m = ss.load_manifest(paths)
        assert m.status == "fresh"
