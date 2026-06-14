"""Tests for the verification scorecard generator (#388 Ask#4).

`scripts/validate/ocean_fidelity/build_fidelity_scorecard.py` aggregates the
existing `results/<component>/summary.json` matrix artifacts and applies the
truth-tier precedence gate to optional fidelity pass/fail records, emitting one
markdown + json scorecard.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = (_REPO_ROOT / "scripts" / "validate" / "ocean_fidelity"
           / "build_fidelity_scorecard.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("build_fidelity_scorecard",
                                                  str(_SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so the module's @dataclass can resolve __module__.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_module()


def _write_matrix_summary(results_root: Path, component: str, results: list[dict]):
    d = results_root / component
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(
        {"results": results, "n_pass": sum(1 for r in results
                                            if r.get("status") == "PASS")}))


class TestScorecardMatrix:
    def test_aggregates_matrix_rows(self, tmp_path):
        rr = tmp_path / "results"
        _write_matrix_summary(rr, "ocean", [
            {"test": "lock_exchange", "grid": "latlon", "resolution": "36x72",
             "tier": 1, "status": "PASS", "maturity": "run_tested"},
            {"test": "overflow", "grid": "cubed_sphere", "resolution": "C24",
             "tier": 2, "status": "FAIL", "maturity": "run_tested"},
        ])
        out = M.build_scorecard(rr, None)
        md = out["markdown"]
        assert "lock_exchange" in md and "overflow" in md
        assert "1/2 matrix cases PASS" in md
        assert out["json"]["matrix"][0]["component"] == "ocean"
        # no fidelity records -> precedence not evaluated
        assert "precedence gate not evaluated" in md
        assert out["verdict"] is None

    def test_missing_results_root_graceful(self, tmp_path):
        out = M.build_scorecard(tmp_path / "nope", None)
        assert "No `results/<component>/summary.json` found" in out["markdown"]


class TestScorecardPrecedence:
    def _fid(self, tmp_path, records):
        p = tmp_path / "fid.json"
        p.write_text(json.dumps(records))
        return p

    def test_oracle_trusted_when_truth_green(self, tmp_path):
        rr = tmp_path / "results"
        rr.mkdir()
        fp = self._fid(tmp_path, [
            {"tier": 0, "passed": True}, {"tier": 1, "passed": True},
            {"tier": 2, "passed": True}, {"tier": 3, "passed": True}])
        out = M.build_scorecard(rr, fp)
        assert out["verdict"].oracle_trusted is True
        assert "TRUSTED" in out["markdown"]
        assert out["json"]["fidelity_precedence"]["status"] == "trusted"

    def test_oracle_locked_when_truth_fails(self, tmp_path):
        rr = tmp_path / "results"
        rr.mkdir()
        fp = self._fid(tmp_path, [
            {"tier": 0, "passed": True}, {"tier": 2, "passed": False},
            {"tier": 3, "passed": True}])
        out = M.build_scorecard(rr, fp)
        assert out["verdict"].oracle_locked is True
        assert "LOCKED" in out["markdown"]

    def test_only_oracle_is_incomplete_not_trusted(self, tmp_path):
        rr = tmp_path / "results"
        rr.mkdir()
        fp = self._fid(tmp_path, [{"tier": 3, "passed": True}])
        out = M.build_scorecard(rr, fp)
        assert out["verdict"].status == "incomplete"
        assert out["verdict"].oracle_locked is False
        assert "INCOMPLETE" in out["markdown"]

    def test_junk_fidelity_records_graceful(self, tmp_path):
        for junk in ({"records": None}, [1, "x"], {"nope": 1}):
            p = tmp_path / "j.json"
            p.write_text(json.dumps(junk))
            assert M.load_fidelity_records(p) == []


class TestScorecardMain:
    def test_main_writes_files_and_exit0_when_clean(self, tmp_path):
        rr = tmp_path / "results"
        _write_matrix_summary(rr, "ocean", [
            {"test": "rest", "grid": "latlon", "tier": 0, "status": "PASS"}])
        out_md = tmp_path / "scorecard.md"
        rc = M.main(["--results-root", str(rr), "--output", str(out_md)])
        assert rc == 0
        assert out_md.is_file()
        assert out_md.with_suffix(".json").is_file()

    def test_main_exit1_on_precedence_violation(self, tmp_path):
        rr = tmp_path / "results"
        rr.mkdir()
        fp = tmp_path / "fid.json"
        fp.write_text(json.dumps([
            {"tier": 2, "passed": False}, {"tier": 3, "passed": True}]))
        rc = M.main(["--results-root", str(rr),
                     "--fidelity-results", str(fp),
                     "--output", str(tmp_path / "sc.md")])
        assert rc == 1  # oracle LOCKED by truth-tier failure
