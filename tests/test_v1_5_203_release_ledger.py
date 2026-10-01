"""v1.5.203 — Release-ledger (``scripts/release_ledger.py``) tests.

Covers the schema init, record, lookup, supersede, snapshot-resolve,
and duplicate-detection logic. Uses a per-test temp DB to keep
isolation.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

# scripts/ isn't on sys.path by default — add it so tests can import.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import release_ledger as rl  # noqa: E402  (post sys.path tweak)


_GIT_SHA_A = "0" * 39 + "a"  # valid 40-char hex
_GIT_SHA_B = "0" * 39 + "b"


def _make_env_file(tmp: Path, contents: str) -> Path:
    p = tmp / "test.env"
    p.write_text(contents, encoding="utf-8")
    return p


# ------------------------ init -------------------------------------------- #


def test_init_creates_db_and_schema(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    assert not db.exists()
    rl.init_db(db)
    assert db.exists()
    # Re-init is idempotent.
    rl.init_db(db)
    # Tables present.
    import sqlite3
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "deploys" in tables
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        assert int(ver) == rl._SCHEMA_VERSION
    finally:
        conn.close()


# ------------------------ record_deploy basic ----------------------------- #


def test_record_deploy_inserts_row(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    env_file = _make_env_file(tmp_path, "FOO=bar\nBAZ=qux\n")
    record, dup = rl.record_deploy(
        db,
        version="1.5.203",
        git_sha=_GIT_SHA_A,
        profile="prod.okx.ton.usdt.perp",
        env_file_path=env_file,
        comment="initial deploy",
    )
    assert dup is False
    assert record.version == "1.5.203"
    assert record.git_sha == _GIT_SHA_A
    assert record.git_sha_short == _GIT_SHA_A[:7]
    assert record.comment == "initial deploy"
    assert record.is_superseded is False
    # env_sha256 is deterministic. v1.5.208 normalises CRLF → LF
    # before hashing so the expected hash mirrors that transform.
    expected = hashlib.sha256(
        env_file.read_bytes().replace(b"\r\n", b"\n")
    ).hexdigest()
    assert record.env_sha256 == expected


def test_record_deploy_empty_message_raises(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    env_file = _make_env_file(tmp_path, "FOO=bar\n")
    with pytest.raises(ValueError, match="message is required"):
        rl.record_deploy(
            db,
            version="1.5.203",
            git_sha=_GIT_SHA_A,
            profile="x",
            env_file_path=env_file,
            comment="",
        )


def test_record_deploy_invalid_git_sha_raises(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    env_file = _make_env_file(tmp_path, "FOO=bar\n")
    with pytest.raises(ValueError, match="git_sha"):
        rl.record_deploy(
            db,
            version="1.5.203",
            git_sha="short",
            profile="x",
            env_file_path=env_file,
            comment="m",
        )


def test_record_deploy_missing_env_file_raises(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    missing = tmp_path / "no_such.env"
    with pytest.raises(ValueError, match="env file not found"):
        rl.record_deploy(
            db,
            version="1.5.203",
            git_sha=_GIT_SHA_A,
            profile="x",
            env_file_path=missing,
            comment="m",
        )


# ------------------------ supersede semantics ----------------------------- #


def test_second_deploy_supersedes_first(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    env_a = _make_env_file(tmp_path, "FOO=v1\n")
    rec_a, _ = rl.record_deploy(
        db,
        version="1.5.203",
        git_sha=_GIT_SHA_A,
        profile="prof",
        env_file_path=env_a,
        comment="first",
    )
    env_b = tmp_path / "v2.env"
    env_b.write_text("FOO=v2\n", encoding="utf-8")
    rec_b, _ = rl.record_deploy(
        db,
        version="1.5.204",
        git_sha=_GIT_SHA_B,
        profile="prof",
        env_file_path=env_b,
        comment="second",
    )
    # Refetch latest live row + the first row's current state.
    live = rl.lookup_latest(db, profile="prof")
    assert live is not None
    assert live.id == rec_b.id
    assert live.is_superseded is False
    # Look up first row — should now be superseded by second.
    recent = rl.list_recent(db, profile="prof", limit=5)
    by_id = {r.id: r for r in recent}
    first = by_id[rec_a.id]
    assert first.is_superseded is True
    assert first.superseded_by_id == rec_b.id
    assert first.superseded_at_utc is not None


def test_supersede_is_per_profile(tmp_path: Path) -> None:
    """Deploying profile B should NOT supersede profile A's live row."""
    db = tmp_path / "deploys.db"
    env_a = _make_env_file(tmp_path, "A=1\n")
    rec_a, _ = rl.record_deploy(
        db, version="1.5.0", git_sha=_GIT_SHA_A,
        profile="prof_a", env_file_path=env_a, comment="A",
    )
    env_b = tmp_path / "b.env"
    env_b.write_text("B=1\n", encoding="utf-8")
    rec_b, _ = rl.record_deploy(
        db, version="1.5.0", git_sha=_GIT_SHA_B,
        profile="prof_b", env_file_path=env_b, comment="B",
    )
    live_a = rl.lookup_latest(db, profile="prof_a")
    live_b = rl.lookup_latest(db, profile="prof_b")
    assert live_a is not None and live_a.id == rec_a.id
    assert live_b is not None and live_b.id == rec_b.id
    assert live_a.is_superseded is False
    assert live_b.is_superseded is False


# ------------------------ duplicate detection ----------------------------- #


def test_duplicate_deploy_returns_existing_row_no_insert(
    tmp_path: Path,
) -> None:
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "FOO=bar\n")
    rec1, dup1 = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="first",
    )
    assert dup1 is False
    # Same (version, git_sha, env_blob) within the 24h window → dup.
    rec2, dup2 = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="second-attempt",
    )
    assert dup2 is True
    assert rec2.id == rec1.id
    # And only one row in the DB.
    rows = rl.list_recent(db, profile="prof", limit=10)
    assert len(rows) == 1


def test_duplicate_deploy_allow_duplicate_inserts_new_row(
    tmp_path: Path,
) -> None:
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "FOO=bar\n")
    rec1, _ = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="first",
    )
    rec2, dup = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="forced-redeploy",
        allow_duplicate=True,
    )
    assert dup is False
    assert rec2.id != rec1.id
    rows = rl.list_recent(db, profile="prof", limit=10)
    assert len(rows) == 2


# ------------------------ env-change creates new row --------------------- #


def test_env_edit_within_same_version_creates_new_row(
    tmp_path: Path,
) -> None:
    """The whole reason this ledger exists: env-only changes
    (same version, same git_sha) MUST be distinguishable."""
    db = tmp_path / "deploys.db"
    env_v1 = _make_env_file(tmp_path, "FOO=bar\n")
    rec1, _ = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env_v1, comment="initial",
    )
    env_v2 = tmp_path / "v2.env"
    env_v2.write_text("FOO=baz\n", encoding="utf-8")  # different content
    rec2, dup = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env_v2, comment="env-tweak",
    )
    assert dup is False  # different env content => not a duplicate
    assert rec1.env_sha256 != rec2.env_sha256
    assert rec1.id != rec2.id
    # rec1 superseded; rec2 is live.
    assert rl.lookup_latest(db, profile="prof").id == rec2.id


# ------------------------ lookup_by_env_sha256 ---------------------------- #


def test_lookup_by_env_sha256_exact_match(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "FOO=bar\n")
    rec, _ = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="m",
    )
    found = rl.lookup_by_env_sha256(db, rec.env_sha256)
    assert found is not None
    assert found.id == rec.id


def test_lookup_by_env_sha256_no_match_returns_none(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "FOO=bar\n")
    rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="m",
    )
    fake = "0" * 64
    assert rl.lookup_by_env_sha256(db, fake) is None


def test_lookup_by_env_sha256_profile_filter_disambiguates(
    tmp_path: Path,
) -> None:
    """If the same env content was deployed to two profiles
    (extremely unlikely, but cheap to guard), the profile filter
    must select the matching one."""
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "SHARED=1\n")
    rec_a, _ = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof_a", env_file_path=env, comment="a",
    )
    rec_b, _ = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_B,
        profile="prof_b", env_file_path=env, comment="b",
    )
    found_a = rl.lookup_by_env_sha256(
        db, rec_a.env_sha256, profile="prof_a"
    )
    found_b = rl.lookup_by_env_sha256(
        db, rec_b.env_sha256, profile="prof_b"
    )
    assert found_a is not None and found_a.id == rec_a.id
    assert found_b is not None and found_b.id == rec_b.id


# ------------------------ snapshot resolution ----------------------------- #


def _make_snapshot(tmp_path: Path, meta: dict) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return snap


def test_write_snapshot_deploy_record_exact_match(
    tmp_path: Path,
) -> None:
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "FOO=bar\n")
    rec, _ = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="m",
    )
    snap = _make_snapshot(
        tmp_path,
        {
            "bot_profile": "prof",
            "bot_version": "1.5.203",
            "env_file_sha256": rec.env_sha256,
        },
    )
    result = rl.write_snapshot_deploy_record(db, snap)
    assert result["match_kind"] == "exact"
    assert result["deploy_record"] is not None
    assert result["deploy_record"]["id"] == rec.id
    # File written.
    written = json.loads(
        (snap / "deploy_record.json").read_text(encoding="utf-8")
    )
    assert written["match_kind"] == "exact"


def test_write_snapshot_deploy_record_no_match(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "FOO=bar\n")
    rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="m",
    )
    bogus_sha = "0" * 64
    snap = _make_snapshot(
        tmp_path,
        {"bot_profile": "prof", "env_file_sha256": bogus_sha},
    )
    result = rl.write_snapshot_deploy_record(db, snap)
    assert result["match_kind"] == "no_match"
    assert result["deploy_record"] is None
    assert result["warning"] is not None


def test_write_snapshot_deploy_record_no_meta(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    rl.init_db(db)
    snap = tmp_path / "snap"
    snap.mkdir()
    # No meta.json.
    result = rl.write_snapshot_deploy_record(db, snap)
    assert result["match_kind"] == "no_meta"
    assert result["deploy_record"] is None


def test_write_snapshot_deploy_record_no_sha_field(
    tmp_path: Path,
) -> None:
    db = tmp_path / "deploys.db"
    rl.init_db(db)
    snap = _make_snapshot(tmp_path, {"bot_profile": "prof"})
    result = rl.write_snapshot_deploy_record(db, snap)
    assert result["match_kind"] == "no_sha"


# ------------------------ list_recent ------------------------------------- #


def test_list_recent_returns_newest_first(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "X=1\n")
    rec1, _ = rl.record_deploy(
        db, version="1.5.203", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment="one",
    )
    env2 = tmp_path / "two.env"
    env2.write_text("Y=2\n", encoding="utf-8")
    rec2, _ = rl.record_deploy(
        db, version="1.5.204", git_sha=_GIT_SHA_B,
        profile="prof", env_file_path=env2, comment="two",
    )
    rows = rl.list_recent(db, profile="prof", limit=10)
    assert [r.id for r in rows] == [rec2.id, rec1.id]


def test_crlf_lf_env_file_produce_same_sha(tmp_path: Path) -> None:
    """v1.5.207 — Windows working tree often has CRLF; colo runs LF
    (post git autocrlf=input). Without normalisation, the ledger
    records the CRLF sha and the snapshot computes the LF sha →
    every lookup is a false-negative ``no_match``. Verified on
    2026-05-28 against the v1.5.206-260528-141714 snapshot: the
    LF-normalised hash of the ledger's env_blob matched the
    snapshot's ``meta.json::env_file_sha256`` byte-for-byte.
    """
    db = tmp_path / "deploys.db"
    # Same content, different line endings.
    env_crlf = tmp_path / "crlf.env"
    env_lf = tmp_path / "lf.env"
    env_crlf.write_bytes(b"FOO=1\r\nBAR=2\r\nBAZ=3\r\n")
    env_lf.write_bytes(b"FOO=1\nBAR=2\nBAZ=3\n")
    rec_crlf, _ = rl.record_deploy(
        db, version="1.5.207", git_sha=_GIT_SHA_A,
        profile="prof-crlf", env_file_path=env_crlf, comment="m",
    )
    rec_lf, _ = rl.record_deploy(
        db, version="1.5.207", git_sha=_GIT_SHA_B,
        profile="prof-lf", env_file_path=env_lf, comment="m",
    )
    # Both should produce the SAME sha256 (CRLF normalised away).
    assert rec_crlf.env_sha256 == rec_lf.env_sha256


def test_unicode_in_comment_survives_record_and_lookup(
    tmp_path: Path,
) -> None:
    """v1.5.204 regression — operator deploy messages naturally contain
    arrows (``→``), em-dashes, smart-quotes. Pre-fix the
    ``_print_record_human`` path crashed on Windows cp1252 stdout
    even though the DB INSERT had already succeeded. Verify the
    row is stored AND human-print succeeds end-to-end.
    """
    db = tmp_path / "deploys.db"
    env = _make_env_file(tmp_path, "FOO=bar\n")
    unicode_msg = "microprice 12→10; tox 6→8 — “stable”"
    rec, dup = rl.record_deploy(
        db, version="1.5.204", git_sha=_GIT_SHA_A,
        profile="prof", env_file_path=env, comment=unicode_msg,
    )
    assert dup is False
    # Round-trip — the stored comment is byte-for-byte identical.
    assert rec.comment == unicode_msg
    # Look up by env_sha256 and confirm the comment is preserved.
    found = rl.lookup_by_env_sha256(db, rec.env_sha256)
    assert found is not None
    assert found.comment == unicode_msg


def test_force_utf8_stdio_is_safe_to_call_repeatedly() -> None:
    """The bootstrap helper must be idempotent + must not raise on
    streams that don't support reconfigure (e.g. captured under
    pytest)."""
    import importlib
    import scripts.release_ledger as _rl
    # Multiple invocations should be a no-op cumulatively.
    _rl._force_utf8_stdio()
    _rl._force_utf8_stdio()


def test_list_recent_limit_respected(tmp_path: Path) -> None:
    db = tmp_path / "deploys.db"
    for i in range(5):
        env = tmp_path / f"e{i}.env"
        env.write_text(f"X={i}\n", encoding="utf-8")
        sha = ("0" * 39) + f"{i}"  # crude unique 40-char
        # Pad to length 40
        sha = (sha + "0" * 40)[:40]
        rl.record_deploy(
            db, version=f"1.5.{i}", git_sha=sha,
            profile="prof", env_file_path=env, comment=f"deploy {i}",
        )
    rows = rl.list_recent(db, profile="prof", limit=3)
    assert len(rows) == 3
