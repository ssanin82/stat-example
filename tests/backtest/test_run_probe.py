"""Tests for ``scripts/backtest/run_probe.py`` (Phase 4B, v1.5.308).

The probe runner's CLI machinery is fully unit-testable WITHOUT any
real fixture: the YAML loader, the flat metric extractor, the no-eval
assertion engine, the (scenario × config) matrix build with an
injectable runner, assertion evaluation, and the renderers all take
plain dicts. Tests feed canned replay reports through an injected
runner so nothing spawns a subprocess.

The one ``main()`` end-to-end test seeds a tiny scenario library on
disk (via ``scenario_storage.create_scenario``), monkeypatches the
subprocess runner with a canned-report stand-in, and asserts the exit
code is non-zero when an assertion fails / zero when they pass —
covering the §4B.4 acceptance criterion ("assertions fail loudly").
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from app.backtest import scenario_storage as ss
from app.backtest.scenario_storage import ScenarioManifest


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RUN_PROBE_PATH = _PROJECT_ROOT / "scripts" / "backtest" / "run_probe.py"


def _import_run_probe():
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(
        "run_probe_under_test", _RUN_PROBE_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution
    # (``from __future__ import annotations`` makes every annotation a
    # string) can find the module dict via ``sys.modules[__module__]``.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rp = _import_run_probe()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mani(scenario_id: str, archetype: str, tags: list[str] | None = None) -> ScenarioManifest:
    return ScenarioManifest(
        scenario_id=scenario_id,
        archetype=archetype,
        symbol="TON-USDT-SWAP",
        venue="okx",
        source_recording_id="src",
        cut_start_utc="",
        cut_end_utc="",
        cut_duration_seconds=0.0,
        operator_tags=tags or [],
    )


def _report(
    *,
    fills: float = 0.0,
    realized_pnl_usd: float = 0.0,
    gates: dict[str, Any] | None = None,
    paper: dict[str, Any] | None = None,
    summary_extra: dict[str, Any] | None = None,
    per_regime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = {
        "fills": fills,
        "volume_usd": 0.0,
        "realized_pnl_usd": realized_pnl_usd,
        "fees_total_usd": 0.0,
        "max_drawdown_usd": 0.0,
    }
    if summary_extra:
        summary.update(summary_extra)
    return {
        "summary": summary,
        "paper_executor": paper or {"fills_emitted": fills, "position_qty": 0.0},
        "gates_fired": gates or {},
        "fill_attribution": {},
        "ticks": {"executed": 100, "scheduled": 100, "skipped_warmup": 0},
        "events": {"total": 5000},
        "per_regime": per_regime or {},
    }


def _runner_from_map(mapping: dict[tuple[str, str], dict[str, Any]]):
    def runner(manifest: ScenarioManifest, config) -> dict[str, Any]:
        return mapping[(manifest.scenario_id, config.name)]
    return runner


# ---------------------------------------------------------------------------
# load_probe
# ---------------------------------------------------------------------------

_VALID_PROBE = """
probe: test_probe
description: |
  A test probe.
scenarios:
  - archetype: structural_bias_short
  - scenario_id: low_vol_2026_05_15
  - sf_storm
configs:
  baseline:
    env: {}
  variant_a:
    env:
      FOO_ENABLED: true
      FOO_THRESHOLD: 5.0
      BAR_COUNT: 3
assert:
  - on: "*"
    config: baseline
    check: "fills >= 0"
  - on: "archetype:structural_bias_*"
    check: "fills[variant_a] < fills[baseline]"
output:
  format: matrix
  include_metrics:
    - fills
    - realized_pnl_usd
"""


class TestLoadProbe:
    def test_valid(self, tmp_path):
        p = tmp_path / "probe.yaml"
        p.write_text(_VALID_PROBE, encoding="utf-8")
        spec = rp.load_probe(p)
        assert spec.probe == "test_probe"
        assert spec.scenarios == (
            "archetype:structural_bias_short",
            "scenario_id:low_vol_2026_05_15",
            "sf_storm",
        )
        assert [c.name for c in spec.configs] == ["baseline", "variant_a"]
        assert spec.configs[0].env == {}
        # bool -> lowercase string; numbers -> str.
        assert spec.configs[1].env == {
            "FOO_ENABLED": "true",
            "FOO_THRESHOLD": "5.0",
            "BAR_COUNT": "3",
        }
        assert spec.output.include_metrics == ("fills", "realized_pnl_usd")

    def test_on_key_yaml_bool_footgun(self, tmp_path):
        # PyYAML parses an unquoted ``on:`` as boolean True; the loader
        # must still recover the selector.
        p = tmp_path / "probe.yaml"
        p.write_text(_VALID_PROBE, encoding="utf-8")
        spec = rp.load_probe(p)
        assert spec.asserts[0].on == "*"
        assert spec.asserts[0].config == "baseline"
        assert spec.asserts[0].check == "fills >= 0"
        assert spec.asserts[1].on == "archetype:structural_bias_*"
        assert spec.asserts[1].config is None

    def test_missing_baseline_raises(self, tmp_path):
        p = tmp_path / "probe.yaml"
        p.write_text(
            "probe: x\nconfigs:\n  variant_a:\n    env: {}\n", encoding="utf-8"
        )
        with pytest.raises(rp.ProbeConfigError, match="baseline"):
            rp.load_probe(p)

    def test_missing_check_raises(self, tmp_path):
        p = tmp_path / "probe.yaml"
        p.write_text(
            "probe: x\nconfigs:\n  baseline:\n    env: {}\n"
            "assert:\n  - on: '*'\n",
            encoding="utf-8",
        )
        with pytest.raises(rp.ProbeConfigError, match="check"):
            rp.load_probe(p)

    def test_bad_scenario_entry_raises(self, tmp_path):
        p = tmp_path / "probe.yaml"
        p.write_text(
            "probe: x\nscenarios:\n  - 42\nconfigs:\n  baseline:\n    env: {}\n",
            encoding="utf-8",
        )
        with pytest.raises(rp.ProbeConfigError):
            rp.load_probe(p)

    def test_defaults(self, tmp_path):
        p = tmp_path / "noname.yaml"
        p.write_text("configs:\n  baseline:\n    env: {}\n", encoding="utf-8")
        spec = rp.load_probe(p)
        assert spec.probe == "noname"           # falls back to file stem
        assert spec.scenarios == ()
        assert spec.asserts == ()
        assert spec.output.format == "matrix"


# ---------------------------------------------------------------------------
# extract_metrics
# ---------------------------------------------------------------------------

class TestExtractMetrics:
    def test_bare_and_dotted(self):
        report = _report(
            fills=4.0,
            realized_pnl_usd=2.5,
            gates={"throttle_fire_count": 2, "some_flag": True},
            paper={"position_qty": -5.0, "fills_emitted": 4},
            per_regime={"CAUTIOUS": {"time_in_mode_pct": 30.0, "fills": 1}},
        )
        m = rp.extract_metrics(report)
        # bare convenience names
        assert m["fills"] == 4.0
        assert m["realized_pnl_usd"] == 2.5
        assert m["throttle_fire_count"] == 2.0
        assert m["position_qty"] == -5.0
        assert m["some_flag"] == 1.0           # bool -> 1.0
        # dotted escape hatches
        assert m["summary.fills"] == 4.0
        assert m["ticks.executed"] == 100.0
        assert m["events.total"] == 5000.0
        assert m["per_regime.CAUTIOUS.time_in_mode_pct"] == 30.0
        assert m["gates_fired.throttle_fire_count"] == 2.0

    def test_skips_non_numeric(self):
        report = {"summary": {"fills": 3, "label": "calm", "blob": None}}
        m = rp.extract_metrics(report)
        assert m["fills"] == 3.0
        assert "label" not in m and "blob" not in m


# ---------------------------------------------------------------------------
# assertion engine
# ---------------------------------------------------------------------------

class TestSplitCheck:
    @pytest.mark.parametrize(
        "expr,parts",
        [
            ("a < b", ("a", "<", "b")),
            ("a<=b", ("a", "<=", "b")),
            ("a >= b", ("a", ">=", "b")),
            ("a == b", ("a", "==", "b")),
            ("a != b", ("a", "!=", "b")),
            ("fills[v] == 0", ("fills[v]", "==", "0")),
        ],
    )
    def test_split(self, expr, parts):
        assert rp._split_check(expr) == parts

    def test_no_operator_raises(self):
        with pytest.raises(rp.ProbeAssertError, match="no comparison operator"):
            rp._split_check("fills_only")


class TestResolveOperand:
    def test_numeric_literal(self):
        assert rp._resolve_operand("5.0", {}, None) == (5.0, "5.0")
        assert rp._resolve_operand("-3", {}, None)[0] == -3.0

    def test_bare_metric_uses_default_config(self):
        sm = {"baseline": {"fills": 4.0}}
        val, label = rp._resolve_operand("fills", sm, "baseline")
        assert val == 4.0
        assert "fills[baseline]" in label

    def test_explicit_config(self):
        sm = {"variant_a": {"fills": 1.0}}
        val, _ = rp._resolve_operand("fills[variant_a]", sm, None)
        assert val == 1.0

    def test_bare_without_default_raises(self):
        with pytest.raises(rp.ProbeAssertError, match="without an explicit"):
            rp._resolve_operand("fills", {"baseline": {"fills": 1.0}}, None)

    def test_missing_metric_raises(self):
        with pytest.raises(rp.ProbeAssertError, match="not found"):
            rp._resolve_operand("nope", {"baseline": {}}, "baseline")

    def test_missing_config_raises(self):
        with pytest.raises(rp.ProbeAssertError, match="not in the matrix"):
            rp._resolve_operand("fills[ghost]", {"baseline": {"fills": 1}}, None)


class TestEvaluateCheck:
    def test_relative_pass(self):
        sm = {"variant_a": {"fills": 1.0}, "baseline": {"fills": 4.0}}
        passed, detail = rp.evaluate_check(
            "fills[variant_a] < fills[baseline]", sm, None
        )
        assert passed is True
        assert "<" in detail

    def test_relative_fail(self):
        sm = {"variant_a": {"fills": 9.0}, "baseline": {"fills": 4.0}}
        passed, _ = rp.evaluate_check(
            "fills[variant_a] < fills[baseline]", sm, None
        )
        assert passed is False

    def test_absolute_with_default_config(self):
        sm = {"baseline": {"fills": 0.0}}
        passed, _ = rp.evaluate_check("fills >= 0", sm, "baseline")
        assert passed is True


# ---------------------------------------------------------------------------
# matrix build + scenario resolution
# ---------------------------------------------------------------------------

class TestResolveScenarios:
    def test_dedup_preserves_order(self):
        manifests = [
            _mani("sb_short_1", "structural_bias_short"),
            _mani("sb_short_2", "structural_bias_short"),
            _mani("sf_1", "sf_storm"),
        ]
        spec = rp.ProbeSpec(
            probe="p", description="",
            scenarios=("archetype:structural_bias_*", "sb_short_1", "sf_storm"),
            configs=(rp.ProbeConfig("baseline"),),
            asserts=(), output=rp.ProbeOutput(),
        )
        resolved = rp.resolve_scenarios(spec, manifests)
        assert [m.scenario_id for m in resolved] == [
            "sb_short_1", "sb_short_2", "sf_1",
        ]


class TestBuildMatrix:
    def test_canned_runner(self):
        manifests = [_mani("a", "low_vol"), _mani("b", "sf_storm")]
        spec = rp.ProbeSpec(
            probe="p", description="", scenarios=(),
            configs=(rp.ProbeConfig("baseline"), rp.ProbeConfig("variant_a")),
            asserts=(), output=rp.ProbeOutput(),
        )
        mapping = {
            ("a", "baseline"): _report(fills=4),
            ("a", "variant_a"): _report(fills=1),
            ("b", "baseline"): _report(fills=8),
            ("b", "variant_a"): _report(fills=3),
        }
        results = rp.build_matrix(spec, manifests, runner=_runner_from_map(mapping))
        assert results["a"]["baseline"].metrics["fills"] == 4.0
        assert results["b"]["variant_a"].metrics["fills"] == 3.0
        assert results["a"]["variant_a"].error is None

    def test_runner_error_recorded_not_raised(self):
        manifests = [_mani("a", "low_vol")]
        spec = rp.ProbeSpec(
            probe="p", description="", scenarios=(),
            configs=(rp.ProbeConfig("baseline"),),
            asserts=(), output=rp.ProbeOutput(),
        )

        def boom(manifest, config):
            raise rp.ProbeRunError("replay blew up")

        results = rp.build_matrix(spec, manifests, runner=boom)
        cell = results["a"]["baseline"]
        assert cell.error is not None
        assert "replay blew up" in cell.error
        assert cell.metrics == {}


# ---------------------------------------------------------------------------
# evaluate_asserts
# ---------------------------------------------------------------------------

class TestEvaluateAsserts:
    def _setup(self, asserts):
        manifests = [
            _mani("low_1", "low_vol"),
            _mani("sb_1", "structural_bias_short"),
        ]
        spec = rp.ProbeSpec(
            probe="p", description="",
            scenarios=("archetype:low_vol", "archetype:structural_bias_short"),
            configs=(rp.ProbeConfig("baseline"), rp.ProbeConfig("variant_a")),
            asserts=tuple(asserts), output=rp.ProbeOutput(),
        )
        mapping = {
            ("low_1", "baseline"): _report(fills=4, realized_pnl_usd=1.0),
            ("low_1", "variant_a"): _report(fills=9, realized_pnl_usd=2.0),
            ("sb_1", "baseline"): _report(fills=4, realized_pnl_usd=-1.0),
            ("sb_1", "variant_a"): _report(fills=1, realized_pnl_usd=0.5),
        }
        results = rp.build_matrix(
            spec, manifests, runner=_runner_from_map(mapping)
        )
        return spec, results, manifests

    def test_pass_on_all(self):
        spec, results, manifests = self._setup(
            [rp.ProbeAssert(on="*", check="fills >= 0", config="baseline")]
        )
        outcomes = rp.evaluate_asserts(spec, results, manifests)
        assert len(outcomes) == 2          # one per matched scenario
        assert all(o.passed and not o.error for o in outcomes)

    def test_relative_fail_on_subset(self):
        # fills[variant_a] < fills[baseline]: true for sb_1 (1<4),
        # false for low_1 (9<4 is False).
        spec, results, manifests = self._setup(
            [rp.ProbeAssert(
                on="archetype:*", check="fills[variant_a] < fills[baseline]"
            )]
        )
        outcomes = rp.evaluate_asserts(spec, results, manifests)
        by_sid = {o.scenario_id: o for o in outcomes}
        assert by_sid["sb_1"].passed is True
        assert by_sid["low_1"].passed is False
        assert any(o.is_failure for o in outcomes)

    def test_no_match_is_skip_not_failure(self):
        spec, results, manifests = self._setup(
            [rp.ProbeAssert(on="archetype:nonexistent", check="fills >= 0",
                            config="baseline")]
        )
        outcomes = rp.evaluate_asserts(spec, results, manifests)
        assert len(outcomes) == 1
        assert outcomes[0].is_skip is True
        assert outcomes[0].is_failure is False

    def test_missing_metric_is_failure(self):
        spec, results, manifests = self._setup(
            [rp.ProbeAssert(on="archetype:low_vol", check="ghost_metric >= 0",
                            config="baseline")]
        )
        outcomes = rp.evaluate_asserts(spec, results, manifests)
        assert outcomes[0].error is not None
        assert outcomes[0].is_failure is True


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

class TestRender:
    def test_matrix_and_asserts(self):
        manifests = [_mani("a", "low_vol")]
        spec = rp.ProbeSpec(
            probe="p", description="", scenarios=(),
            configs=(rp.ProbeConfig("baseline"), rp.ProbeConfig("variant_a")),
            asserts=(), output=rp.ProbeOutput(include_metrics=("fills",)),
        )
        mapping = {
            ("a", "baseline"): _report(fills=4),
            ("a", "variant_a"): _report(fills=1),
        }
        results = rp.build_matrix(spec, manifests, runner=_runner_from_map(mapping))
        matrix = rp.render_matrix(spec, results, manifests)
        assert "Scenario" in matrix and "baseline" in matrix and "variant_a" in matrix
        assert "fills" in matrix
        outcomes = [
            rp.AssertOutcome(on="*", check="fills >= 0", scenario_id="a", passed=True, detail="ok"),
            rp.AssertOutcome(on="*", check="fills < 0", scenario_id="a", passed=False, detail="bad"),
            rp.AssertOutcome(on="archetype:x", check="fills >= 0", scenario_id=rp._NO_MATCH, passed=True, detail="skip"),
        ]
        text = rp.render_asserts(outcomes)
        assert "[PASS]" in text and "[FAIL]" in text and "[SKIP]" in text

    def test_error_cell_renders_err(self):
        manifests = [_mani("a", "low_vol")]
        spec = rp.ProbeSpec(
            probe="p", description="", scenarios=(),
            configs=(rp.ProbeConfig("baseline"),),
            asserts=(), output=rp.ProbeOutput(include_metrics=("fills",)),
        )

        def boom(manifest, config):
            raise rp.ProbeRunError("nope")

        results = rp.build_matrix(spec, manifests, runner=boom)
        matrix = rp.render_matrix(spec, results, manifests)
        assert "ERR" in matrix


# ---------------------------------------------------------------------------
# main() — skip-when-absent + fail-loudly (§4B.4)
# ---------------------------------------------------------------------------

class _CannedRunnerFactory:
    """Stand-in for ``SubprocessReplayRunner``: constructed with kwargs
    (ignored), instances callable as ``(manifest, config) -> report``."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __call__(self, **kwargs):  # mimics SubprocessReplayRunner(...)
        mapping = self._mapping

        def run(manifest, config):
            return mapping[(manifest.scenario_id, config.name)]

        return run


_MAIN_PROBE = """
probe: main_probe
scenarios:
  - archetype: low_vol
configs:
  baseline:
    env: {}
  variant_a:
    env:
      FOO: true
assert:
  - on: "archetype:low_vol"
    check: "fills[variant_a] < fills[baseline]"
output:
  include_metrics:
    - fills
"""


class TestMain:
    def test_skip_when_library_empty(self, tmp_path, capsys):
        p = tmp_path / "probe.yaml"
        p.write_text(_MAIN_PROBE, encoding="utf-8")
        rc = rp.main([str(p), "--repo-root", str(tmp_path)])
        assert rc == 0
        err = capsys.readouterr().err
        assert "no scenarios matched" in err

    def test_probe_file_not_found(self, tmp_path):
        rc = rp.main([str(tmp_path / "missing.yaml")])
        assert rc == 2

    def test_fail_loudly_on_assert(self, tmp_path, monkeypatch):
        ss.create_scenario(
            scenario_id="low_1", archetype="low_vol", symbol="TON-USDT-SWAP",
            venue="okx", source_recording_id="src", cut_start_utc="",
            cut_end_utc="", cut_duration_seconds=0.0,
            require_data_files=False, repo_root=tmp_path,
        )
        p = tmp_path / "probe.yaml"
        p.write_text(_MAIN_PROBE, encoding="utf-8")
        # variant_a fills(9) < baseline fills(4) -> False -> assertion fails.
        mapping = {
            ("low_1", "baseline"): _report(fills=4),
            ("low_1", "variant_a"): _report(fills=9),
        }
        monkeypatch.setattr(
            rp, "SubprocessReplayRunner", _CannedRunnerFactory(mapping)
        )
        rc = rp.main([str(p), "--repo-root", str(tmp_path)])
        assert rc == 1

    def test_pass_returns_zero(self, tmp_path, monkeypatch):
        ss.create_scenario(
            scenario_id="low_1", archetype="low_vol", symbol="TON-USDT-SWAP",
            venue="okx", source_recording_id="src", cut_start_utc="",
            cut_end_utc="", cut_duration_seconds=0.0,
            require_data_files=False, repo_root=tmp_path,
        )
        p = tmp_path / "probe.yaml"
        p.write_text(_MAIN_PROBE, encoding="utf-8")
        # variant_a fills(1) < baseline fills(4) -> True -> passes.
        mapping = {
            ("low_1", "baseline"): _report(fills=4),
            ("low_1", "variant_a"): _report(fills=1),
        }
        monkeypatch.setattr(
            rp, "SubprocessReplayRunner", _CannedRunnerFactory(mapping)
        )
        rc = rp.main([str(p), "--repo-root", str(tmp_path)])
        assert rc == 0

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        ss.create_scenario(
            scenario_id="low_1", archetype="low_vol", symbol="TON-USDT-SWAP",
            venue="okx", source_recording_id="src", cut_start_utc="",
            cut_end_utc="", cut_duration_seconds=0.0,
            require_data_files=False, repo_root=tmp_path,
        )
        p = tmp_path / "probe.yaml"
        p.write_text(_MAIN_PROBE, encoding="utf-8")
        mapping = {
            ("low_1", "baseline"): _report(fills=4),
            ("low_1", "variant_a"): _report(fills=1),
        }
        monkeypatch.setattr(
            rp, "SubprocessReplayRunner", _CannedRunnerFactory(mapping)
        )
        rc = rp.main([str(p), "--repo-root", str(tmp_path), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        import json
        payload = json.loads(out)
        assert payload["probe"] == "main_probe"
        assert "low_1" in payload["matrix"]
        assert payload["failures"] == 0


# ---------------------------------------------------------------------------
# main(--out-dir) — the artifact-persistence path the dashboard panel polls
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))


class TestOutDir:
    def _seed(self, tmp_path):
        ss.create_scenario(
            scenario_id="low_1", archetype="low_vol", symbol="TON-USDT-SWAP",
            venue="okx", source_recording_id="src", cut_start_utc="",
            cut_end_utc="", cut_duration_seconds=0.0,
            require_data_files=False, repo_root=tmp_path,
        )
        p = tmp_path / "probe.yaml"
        p.write_text(_MAIN_PROBE, encoding="utf-8")
        return p

    def test_pass_writes_done_status_and_artifacts(self, tmp_path, monkeypatch):
        p = self._seed(tmp_path)
        out_dir = tmp_path / "run01"
        mapping = {
            ("low_1", "baseline"): _report(fills=4),
            ("low_1", "variant_a"): _report(fills=1),   # 1<4 -> pass
        }
        monkeypatch.setattr(
            rp, "SubprocessReplayRunner", _CannedRunnerFactory(mapping)
        )
        rc = rp.main(
            [str(p), "--repo-root", str(tmp_path), "--out-dir", str(out_dir)]
        )
        assert rc == 0
        status = _read_json(out_dir / "status.json")
        assert status["state"] == "done"
        assert status["exit_code"] == 0
        assert status["failures"] == 0
        assert status["scenarios_matched"] == 1
        assert status["probe"] == "main_probe"
        assert status["finished_at_utc"] is not None
        # the machine-readable matrix + text render land beside it
        result = _read_json(out_dir / "result.json")
        assert "low_1" in result["matrix"]
        assert (out_dir / "matrix.txt").read_text(encoding="utf-8").strip()

    def test_fail_writes_failed_status(self, tmp_path, monkeypatch):
        p = self._seed(tmp_path)
        out_dir = tmp_path / "run02"
        mapping = {
            ("low_1", "baseline"): _report(fills=4),
            ("low_1", "variant_a"): _report(fills=9),   # 9<4 -> fail
        }
        monkeypatch.setattr(
            rp, "SubprocessReplayRunner", _CannedRunnerFactory(mapping)
        )
        rc = rp.main(
            [str(p), "--repo-root", str(tmp_path), "--out-dir", str(out_dir)]
        )
        assert rc == 1
        status = _read_json(out_dir / "status.json")
        assert status["state"] == "failed"
        assert status["exit_code"] == 1
        assert status["failures"] == 1

    def test_skip_writes_skipped_status(self, tmp_path):
        # empty library -> no scenarios match -> exit 0, state "skipped".
        p = tmp_path / "probe.yaml"
        p.write_text(_MAIN_PROBE, encoding="utf-8")
        out_dir = tmp_path / "run03"
        rc = rp.main(
            [str(p), "--repo-root", str(tmp_path), "--out-dir", str(out_dir)]
        )
        assert rc == 0
        status = _read_json(out_dir / "status.json")
        assert status["state"] == "skipped"
        assert status["exit_code"] == 0
        assert status["scenarios_matched"] == 0

    def test_probe_not_found_writes_error_status(self, tmp_path):
        out_dir = tmp_path / "run04"
        rc = rp.main(
            [str(tmp_path / "missing.yaml"), "--out-dir", str(out_dir)]
        )
        assert rc == 2
        status = _read_json(out_dir / "status.json")
        assert status["state"] == "error"
        assert status["exit_code"] == 2
        assert "not found" in (status["error"] or "")
