"""Tests for app.backtest.scenario_storage (Phase 4B scenario library,
audit P1 #6).

Coverage:
  * slugify_scenario_id / slugify_archetype: normalisation + rejection
  * is_valid_scenario_id / is_valid_archetype: yes/no checks
  * scenario_paths: layout + path-traversal guard
  * ScenarioManifest round-trip via save/load (incl. nested data_files,
    summary dict, forward-compat unknown-key filtering)
  * list_scenarios / load_all_manifests (best-effort skip of malformed)
  * filter_by_archetype (exact + glob) / filter_by_tag / resolve_selector
  * create_scenario (auto-slug, require_data_files, clobber guard)
  * delete_scenario (idempotent)
  * scenario_summary (file presence + bytes, has_initial_state, summary)

All tests use a temp directory as ``repo_root`` so they never touch the
real ``backtesting/data/scenarios/`` tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pytest

from app.backtest import scenario_storage as ss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_repo(tmp_path: Path) -> Iterator[Path]:
    """Temp repo root with an empty scenarios dir already present."""
    (tmp_path / "backtesting" / "data" / "scenarios").mkdir(parents=True)
    yield tmp_path


def _create(
    repo: Path,
    scenario_id: str,
    archetype: str,
    *,
    tags: list[str] | None = None,
    summary: dict | None = None,
) -> ss.ScenarioPaths:
    """Helper: register a scenario with no data files (require_data_files
    False) so tests don't have to fabricate feed bytes."""
    return ss.create_scenario(
        scenario_id=scenario_id,
        archetype=archetype,
        symbol="TON-USDT-SWAP",
        venue="okx",
        source_recording_id="v1.5.148-260525-155330-prod.okx.ton.usdt.perp",
        cut_start_utc="2026-05-21T17:07:32+00:00",
        cut_end_utc="2026-05-21T17:43:39+00:00",
        cut_duration_seconds=2167.0,
        operator_tags=tags,
        summary=summary,
        require_data_files=False,
        repo_root=repo,
    )


# ---------------------------------------------------------------------------
# Slug validation
# ---------------------------------------------------------------------------

class TestSlugify:
    @pytest.mark.parametrize("raw, expected", [
        ("Structural Bias Short", "structural_bias_short"),
        ("low_vol 2026-05-15", "low_vol_2026-05-15"),
        ("v1.4.219", "v1_4_219"),
        ("SF Storm!!!", "sf_storm"),
        ("  trimmed  ", "trimmed"),
        ("mixed_-_separators", "mixed_separators"),
        ("with-dashes-ok", "with-dashes-ok"),
        ("a", "a"),
    ])
    def test_slugify_scenario_id_normalises(self, raw: str, expected: str) -> None:
        assert ss.slugify_scenario_id(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "...", "!!!", " __ "])
    def test_slugify_rejects_empty_result(self, raw: str) -> None:
        with pytest.raises(ValueError, match="empty"):
            ss.slugify_scenario_id(raw)

    def test_slugify_scenario_id_rejects_overlong(self) -> None:
        with pytest.raises(ValueError, match="max"):
            ss.slugify_scenario_id("a" * 200)

    def test_slugify_archetype_caps_shorter(self) -> None:
        # 64 is the archetype cap; 80 chars must reject.
        with pytest.raises(ValueError, match="max"):
            ss.slugify_archetype("a" * 80)

    def test_slugify_non_string(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            ss.slugify_scenario_id(None)  # type: ignore[arg-type]


class TestIsValid:
    @pytest.mark.parametrize("sid, ok", [
        ("structural_bias_short_v1_4_219_260521_214353", True),
        ("low_vol_2026-05-15", True),
        ("a", True),
        ("ab", True),
        ("", False),
        ("Upper", False),
        ("_leading", False),
        ("trailing_", False),
        ("has space", False),
        ("../evil", False),
        ("dir/sub", False),
        ("dot.in.id", False),
        ("a" * 97, False),
    ])
    def test_is_valid_scenario_id(self, sid: str, ok: bool) -> None:
        assert ss.is_valid_scenario_id(sid) is ok

    def test_is_valid_scenario_id_non_string(self) -> None:
        assert ss.is_valid_scenario_id(123) is False  # type: ignore[arg-type]

    @pytest.mark.parametrize("arch, ok", [
        ("structural_bias_short", True),
        ("sf_storm", True),
        ("low_vol", True),
        ("", False),
        ("Upper", False),
        ("a" * 65, False),
    ])
    def test_is_valid_archetype(self, arch: str, ok: bool) -> None:
        assert ss.is_valid_archetype(arch) is ok


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

class TestPaths:
    def test_scenarios_root(self, tmp_repo: Path) -> None:
        root = ss.scenarios_root(repo_root=tmp_repo)
        assert root == tmp_repo / "backtesting" / "data" / "scenarios"

    def test_scenario_paths_layout(self, tmp_repo: Path) -> None:
        p = ss.scenario_paths("low_vol_2026-05-15", repo_root=tmp_repo)
        assert p.scenario_id == "low_vol_2026-05-15"
        assert p.root == ss.scenarios_root(repo_root=tmp_repo) / "low_vol_2026-05-15"
        assert p.manifest == p.root / "manifest.json"
        assert p.initial_state == p.root / "initial_state.json"

    @pytest.mark.parametrize("bad", ["../evil", "dir/sub", "..", "a/../b"])
    def test_scenario_paths_traversal_guard(self, tmp_repo: Path, bad: str) -> None:
        with pytest.raises(ValueError):
            ss.scenario_paths(bad, repo_root=tmp_repo)

    def test_scenario_paths_stays_under_root(self, tmp_repo: Path) -> None:
        p = ss.scenario_paths("sf_storm_v1", repo_root=tmp_repo)
        base = ss.scenarios_root(repo_root=tmp_repo).resolve()
        # Must not raise.
        p.root.relative_to(base)


# ---------------------------------------------------------------------------
# Manifest round-trip
# ---------------------------------------------------------------------------

class TestManifestRoundTrip:
    def test_save_load_roundtrip(self, tmp_repo: Path) -> None:
        paths = ss.scenario_paths("rt_scenario", repo_root=tmp_repo)
        m = ss.ScenarioManifest(
            scenario_id="rt_scenario",
            archetype="sf_storm",
            symbol="TON-USDT-SWAP",
            venue="okx",
            source_recording_id="rec-1",
            cut_start_utc="2026-05-21T17:07:32+00:00",
            cut_end_utc="2026-05-21T17:43:39+00:00",
            cut_duration_seconds=2167.0,
            data_files=[
                ss.ScenarioDataFile(
                    feed="okx_public", path="okx_public.jsonl.gz",
                    bytes=12345678, lines=74229,
                    first_t_recv_ns=1779724411085020307,
                    last_t_recv_ns=1779725945252780149,
                ),
                ss.ScenarioDataFile(feed="binance_public",
                                    path="binance_public.jsonl.gz", bytes=42),
            ],
            operator_tags=["sf_storm", "TON-USDT-SWAP"],
            operator_notes="4 SF events in 36 min.",
            summary={"sf_event_count": 4, "max_adverse_bps": 51.2},
        )
        ss.save_manifest(paths, m)
        loaded = ss.load_manifest(paths)
        assert loaded == m
        # Nested data files reconstructed as dataclasses, not dicts.
        assert all(isinstance(d, ss.ScenarioDataFile) for d in loaded.data_files)
        assert loaded.data_files[0].lines == 74229
        assert loaded.summary == {"sf_event_count": 4, "max_adverse_bps": 51.2}

    def test_load_filters_unknown_keys(self, tmp_repo: Path) -> None:
        paths = ss.scenario_paths("fwd_compat", repo_root=tmp_repo)
        paths.root.mkdir(parents=True)
        paths.manifest.write_text(json.dumps({
            "scenario_id": "fwd_compat",
            "archetype": "low_vol",
            "symbol": "TON-USDT-SWAP",
            "venue": "okx",
            "source_recording_id": "rec-2",
            "cut_start_utc": "2026-05-21T17:07:32+00:00",
            "cut_end_utc": "2026-05-21T17:43:39+00:00",
            "cut_duration_seconds": 100.0,
            "data_files": [],
            # A future field this code version doesn't know about:
            "future_field_v999": {"nested": True},
        }), encoding="utf-8")
        loaded = ss.load_manifest(paths)
        assert loaded.scenario_id == "fwd_compat"
        assert not hasattr(loaded, "future_field_v999")

    def test_load_skips_malformed_data_file_entries(self, tmp_repo: Path) -> None:
        paths = ss.scenario_paths("partial_files", repo_root=tmp_repo)
        paths.root.mkdir(parents=True)
        paths.manifest.write_text(json.dumps({
            "scenario_id": "partial_files",
            "archetype": "low_vol",
            "symbol": "TON-USDT-SWAP",
            "venue": "okx",
            "source_recording_id": "rec-3",
            "cut_start_utc": "x",
            "cut_end_utc": "y",
            "cut_duration_seconds": 1.0,
            "data_files": [
                {"feed": "okx_public", "path": "okx_public.jsonl.gz"},
                {"feed": "missing_path_only"},     # no path → dropped
                {"path": "missing_feed_only.gz"},  # no feed → dropped
                "not_a_dict",                       # junk → dropped
            ],
        }), encoding="utf-8")
        loaded = ss.load_manifest(paths)
        assert len(loaded.data_files) == 1
        assert loaded.data_files[0].feed == "okx_public"


# ---------------------------------------------------------------------------
# Listing + archetype loader
# ---------------------------------------------------------------------------

class TestListing:
    def test_list_scenarios_empty(self, tmp_repo: Path) -> None:
        assert ss.list_scenarios(repo_root=tmp_repo) == []

    def test_list_scenarios_no_root(self, tmp_path: Path) -> None:
        # scenarios/ doesn't exist at all → empty, no crash.
        assert ss.list_scenarios(repo_root=tmp_path) == []

    def test_list_scenarios_sorted_and_filtered(self, tmp_repo: Path) -> None:
        _create(tmp_repo, "zeta_low_vol", "low_vol")
        _create(tmp_repo, "alpha_sf_storm", "sf_storm")
        # Junk dir with no manifest → excluded.
        (ss.scenarios_root(repo_root=tmp_repo) / "no_manifest_dir").mkdir()
        # Loose file → not a dir, excluded.
        (ss.scenarios_root(repo_root=tmp_repo) / "loose.txt").write_text("x")
        assert ss.list_scenarios(repo_root=tmp_repo) == [
            "alpha_sf_storm", "zeta_low_vol",
        ]

    def test_load_all_manifests_skips_malformed(self, tmp_repo: Path) -> None:
        _create(tmp_repo, "good_one", "low_vol")
        # Corrupt JSON.
        bad = ss.scenarios_root(repo_root=tmp_repo) / "bad_json"
        bad.mkdir()
        (bad / "manifest.json").write_text("{not json", encoding="utf-8")
        # Missing required field.
        miss = ss.scenarios_root(repo_root=tmp_repo) / "missing_field"
        miss.mkdir()
        (miss / "manifest.json").write_text(
            json.dumps({"scenario_id": "missing_field"}), encoding="utf-8")
        loaded = ss.load_all_manifests(repo_root=tmp_repo)
        assert [m.scenario_id for m in loaded] == ["good_one"]


class TestArchetypeLoader:
    @pytest.fixture
    def library(self, tmp_repo: Path) -> Path:
        _create(tmp_repo, "sbs_1", "structural_bias_short", tags=["audit"])
        _create(tmp_repo, "sbl_1", "structural_bias_long", tags=["audit", "v1_4"])
        _create(tmp_repo, "sf_1", "sf_storm")
        _create(tmp_repo, "lv_1", "low_vol", tags=["v1_4"])
        return tmp_repo

    def test_filter_by_archetype_exact(self, library: Path) -> None:
        ms = ss.load_all_manifests(repo_root=library)
        out = ss.filter_by_archetype(ms, "sf_storm")
        assert [m.scenario_id for m in out] == ["sf_1"]

    def test_filter_by_archetype_glob(self, library: Path) -> None:
        ms = ss.load_all_manifests(repo_root=library)
        out = ss.filter_by_archetype(ms, "structural_bias_*")
        assert sorted(m.scenario_id for m in out) == ["sbl_1", "sbs_1"]

    def test_filter_by_tag(self, library: Path) -> None:
        ms = ss.load_all_manifests(repo_root=library)
        out = ss.filter_by_tag(ms, "v1_4")
        assert sorted(m.scenario_id for m in out) == ["lv_1", "sbl_1"]

    def test_resolve_selector_scenario_id(self, library: Path) -> None:
        ms = ss.load_all_manifests(repo_root=library)
        out = ss.resolve_selector(ms, "scenario_id:sf_1")
        assert [m.scenario_id for m in out] == ["sf_1"]

    def test_resolve_selector_archetype_glob(self, library: Path) -> None:
        ms = ss.load_all_manifests(repo_root=library)
        out = ss.resolve_selector(ms, "archetype:structural_bias_*")
        assert sorted(m.scenario_id for m in out) == ["sbl_1", "sbs_1"]

    def test_resolve_selector_bare_archetype(self, library: Path) -> None:
        ms = ss.load_all_manifests(repo_root=library)
        out = ss.resolve_selector(ms, "low_vol")
        assert [m.scenario_id for m in out] == ["lv_1"]

    def test_resolve_selector_no_match(self, library: Path) -> None:
        ms = ss.load_all_manifests(repo_root=library)
        assert ss.resolve_selector(ms, "archetype:nonexistent_*") == []


# ---------------------------------------------------------------------------
# create / delete
# ---------------------------------------------------------------------------

class TestCreateDelete:
    def test_create_auto_slugs_id_and_archetype(self, tmp_repo: Path) -> None:
        paths = ss.create_scenario(
            scenario_id="Structural Bias Short v1",
            archetype="Structural Bias Short",
            symbol="TON-USDT-SWAP",
            venue="okx",
            source_recording_id="rec-1",
            cut_start_utc="a", cut_end_utc="b", cut_duration_seconds=1.0,
            require_data_files=False,
            repo_root=tmp_repo,
        )
        assert paths.scenario_id == "structural_bias_short_v1"
        m = ss.load_manifest(paths)
        assert m.archetype == "structural_bias_short"
        # captured_at_utc auto-stamped.
        assert m.captured_at_utc

    def test_create_refuses_clobber(self, tmp_repo: Path) -> None:
        _create(tmp_repo, "dup", "low_vol")
        with pytest.raises(FileExistsError):
            _create(tmp_repo, "dup", "low_vol")

    def test_create_require_data_files_missing(self, tmp_repo: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ss.create_scenario(
                scenario_id="needs_files",
                archetype="low_vol",
                symbol="TON-USDT-SWAP",
                venue="okx",
                source_recording_id="rec-1",
                cut_start_utc="a", cut_end_utc="b", cut_duration_seconds=1.0,
                data_files=[{"feed": "okx_public", "path": "okx_public.jsonl.gz"}],
                require_data_files=True,
                repo_root=tmp_repo,
            )

    def test_create_require_data_files_present(self, tmp_repo: Path) -> None:
        # Pre-create the dir and drop the feed file (cutter's job), then
        # register the manifest with validation on.
        paths = ss.scenario_paths("has_files", repo_root=tmp_repo)
        paths.root.mkdir(parents=True)
        (paths.root / "okx_public.jsonl.gz").write_bytes(b"\x1f\x8b stub")
        out = ss.create_scenario(
            scenario_id="has_files",
            archetype="low_vol",
            symbol="TON-USDT-SWAP",
            venue="okx",
            source_recording_id="rec-1",
            cut_start_utc="a", cut_end_utc="b", cut_duration_seconds=1.0,
            data_files=[ss.ScenarioDataFile(
                feed="okx_public", path="okx_public.jsonl.gz", bytes=8)],
            require_data_files=True,
            repo_root=tmp_repo,
        )
        m = ss.load_manifest(out)
        assert m.data_files[0].path == "okx_public.jsonl.gz"

    def test_create_invalid_data_file_entry(self, tmp_repo: Path) -> None:
        with pytest.raises(ValueError, match="invalid data_files"):
            ss.create_scenario(
                scenario_id="bad_df",
                archetype="low_vol",
                symbol="TON-USDT-SWAP",
                venue="okx",
                source_recording_id="rec-1",
                cut_start_utc="a", cut_end_utc="b", cut_duration_seconds=1.0,
                data_files=[{"feed": "okx_public"}],  # no path
                require_data_files=False,
                repo_root=tmp_repo,
            )

    def test_delete_scenario(self, tmp_repo: Path) -> None:
        paths = _create(tmp_repo, "to_delete", "low_vol")
        assert paths.root.is_dir()
        ss.delete_scenario("to_delete", repo_root=tmp_repo)
        assert not paths.root.exists()

    def test_delete_idempotent(self, tmp_repo: Path) -> None:
        # No such scenario → silent no-op.
        ss.delete_scenario("never_existed", repo_root=tmp_repo)


# ---------------------------------------------------------------------------
# Summary view
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_file_presence_and_bytes(self, tmp_repo: Path) -> None:
        paths = ss.scenario_paths("sum1", repo_root=tmp_repo)
        paths.root.mkdir(parents=True)
        (paths.root / "okx_public.jsonl.gz").write_bytes(b"x" * 100)
        ss.create_scenario(
            scenario_id="sum1",
            archetype="sf_storm",
            symbol="TON-USDT-SWAP",
            venue="okx",
            source_recording_id="rec-1",
            cut_start_utc="a", cut_end_utc="b", cut_duration_seconds=10.0,
            data_files=[
                ss.ScenarioDataFile(feed="okx_public",
                                    path="okx_public.jsonl.gz", bytes=100),
                ss.ScenarioDataFile(feed="binance_public",
                                    path="binance_public.jsonl.gz", bytes=50),
            ],
            summary={"sf_event_count": 8},
            require_data_files=False,
            repo_root=tmp_repo,
        )
        s = ss.scenario_summary(paths)
        files = {f["feed"]: f for f in s["data_files"]}
        assert files["okx_public"]["present"] is True
        assert files["okx_public"]["bytes_on_disk"] == 100
        # Listed in manifest but never sliced to disk.
        assert files["binance_public"]["present"] is False
        assert files["binance_public"]["bytes_on_disk"] is None
        assert files["binance_public"]["bytes_manifest"] == 50
        assert s["total_data_bytes_on_disk"] == 100
        assert s["has_initial_state"] is False
        assert s["summary"] == {"sf_event_count": 8}

    def test_summary_has_initial_state(self, tmp_repo: Path) -> None:
        paths = _create(tmp_repo, "seeded", "low_vol")
        paths.initial_state.write_text("{}", encoding="utf-8")
        s = ss.scenario_summary(paths)
        assert s["has_initial_state"] is True
