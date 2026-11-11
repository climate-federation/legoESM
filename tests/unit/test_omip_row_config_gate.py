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
    diff, problems = gate(base, cand, parser, parser, ROW, [])
    assert problems == [] and list(diff) == ["nemo_geometry"]


def test_an_extra_option_fails(parser):
    cand = _manifest(["--nemo-geometry", "domain_cfg", "--A-h", "1e4"])
    _, problems = gate(_manifest(), cand, parser, parser, ROW, [])
    assert any("A_h" in m for m in problems)


def test_a_different_duration_fails(parser):
    cand = _manifest(["--nemo-geometry", "domain_cfg"])
    cand["run"]["argv"][cand["run"]["argv"].index("0.04")] = "0.08"
    _, problems = gate(_manifest(), cand, parser, parser, ROW, [])
    assert any("years" in m for m in problems)


def test_the_row_option_missing_fails(parser):
    _, problems = gate(_manifest(), _manifest(), parser, parser, ROW, [])
    assert any("unchanged" in m for m in problems)


def test_wrong_row_value_fails(parser):
    _, problems = gate(_manifest(), _manifest(["--nemo-geometry", "mesh_mask"]),
                       parser, parser, {"nemo_geometry": "domain_cfg"}, [])
    assert problems   # mesh_mask is the default: no option changed at all


def test_config_drift_fails_unless_allowed(parser):
    base = _manifest()
    cand = copy.deepcopy(_manifest(["--nemo-geometry", "domain_cfg"]))
    cand["config"]["resolved_config"]["runtime_config"]["x"] = 2
    _, problems = gate(base, cand, parser, parser, ROW, [])
    assert any("runtime_config.x" in m for m in problems)
    _, problems = gate(base, cand, parser, parser, ROW, ["runtime_config.x"])
    assert problems == []


def test_a_default_that_changed_between_commits_fails(parser):
    import argparse
    old = argparse.ArgumentParser()
    old.add_argument("--grid"); old.add_argument("--partial-cell", action="store_true")
    old.add_argument("--years", type=float); old.add_argument("--output")
    old.add_argument("--nemo-geometry", default="mesh_mask")
    old.add_argument("--A-h", type=float, default=2e4)
    new = copy.deepcopy(old)
    new.set_defaults(A_h=1e4)                    # argv never names --A-h
    cand = _manifest(["--nemo-geometry", "domain_cfg"])
    _, problems = gate(_manifest(), cand, old, new, ROW, [])
    assert any("A_h" in m for m in problems)


@pytest.mark.parametrize("cfg_b", [{"output_path": "/a", "runtime_config": {"x": 1}},
                                   {"output_path": "/a", "runtime_config": {"x": 1, "y": {}}}])
def test_a_key_present_on_one_side_only_fails(parser, cfg_b):
    base = _manifest(cfg={"output_path": "/a", "runtime_config": {"x": 1, "physics": None}}
                     if "y" not in cfg_b["runtime_config"] else
                     {"output_path": "/a", "runtime_config": {"x": 1}})
    cand = _manifest(["--nemo-geometry", "domain_cfg"], cfg=cfg_b)
    _, problems = gate(base, cand, parser, parser, ROW, [])
    assert any("<absent>" in m for m in problems)


def test_parser_at_reads_the_runner_of_a_commit():
    from scripts.validate.ocean_fidelity.omip_row_config_gate import parser_at
    p = parser_at("HEAD")
    assert p.parse_args([]).nemo_geometry == "mesh_mask"
    old = parser_at("159773865e")               # before --nemo-geometry existed
    assert "nemo_geometry" not in vars(old.parse_args([]))
