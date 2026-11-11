"""Row gate: exactly the voted options differ, nothing else."""
from __future__ import annotations

import copy

import pytest

from scripts.validate.ocean_fidelity.omip_row_config_gate import gate


@pytest.fixture(scope="module")
def parser():
    from scripts.run.run_omip_core2 import _build_arg_parser
    return _build_arg_parser()


def _manifest(extra=(), cfg=None, out="/a"):
    argv = ["run_omip_core2.py", "--grid", "tripole", "--partial-cell",
            "--years", "0.04", "--output", out, *extra]
    return {"run": {"argv": argv},
            "config": {"resolved_config": cfg or {"output_path": out,
                                                  "runtime_config": {"x": 1}}}}


ROW = {"nemo_geometry": "domain_cfg"}


def test_exact_row_passes_and_paths_are_ignored(parser):
    base = _manifest()
    cand = _manifest(["--nemo-geometry", "domain_cfg"], out="/b")
    cand["run"]["argv"] += ["--restart-from", "/b/r.npz"]
    diff, problems = gate(base, cand, parser, ROW, [])
    assert problems == [] and list(diff) == ["nemo_geometry"]


def test_an_extra_option_fails(parser):
    cand = _manifest(["--nemo-geometry", "domain_cfg", "--A-h", "1e4"])
    _, problems = gate(_manifest(), cand, parser, ROW, [])
    assert any("A_h" in m for m in problems)


def test_a_different_duration_fails(parser):
    cand = _manifest(["--nemo-geometry", "domain_cfg"])
    cand["run"]["argv"][cand["run"]["argv"].index("0.04")] = "0.08"
    _, problems = gate(_manifest(), cand, parser, ROW, [])
    assert any("years" in m for m in problems)


def test_the_row_option_missing_fails(parser):
    _, problems = gate(_manifest(), _manifest(), parser, ROW, [])
    assert any("unchanged" in m for m in problems)


def test_wrong_row_value_fails(parser):
    _, problems = gate(_manifest(), _manifest(["--nemo-geometry", "mesh_mask"]),
                       parser, {"nemo_geometry": "domain_cfg"}, [])
    assert problems   # mesh_mask is the default: no option changed at all


def test_config_drift_fails_unless_allowed(parser):
    base = _manifest()
    cand = copy.deepcopy(_manifest(["--nemo-geometry", "domain_cfg"]))
    cand["config"]["resolved_config"]["runtime_config"]["x"] = 2
    _, problems = gate(base, cand, parser, ROW, [])
    assert any("runtime_config.x" in m for m in problems)
    _, problems = gate(base, cand, parser, ROW, ["runtime_config.x"])
    assert problems == []
