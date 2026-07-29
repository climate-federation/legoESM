"""Unit tests for the shared OAT sweep planner (training/sweep_planner.py, D6)."""

import json
from pathlib import Path

import pytest
import yaml

from legoesm.training.sweep_planner import (
    build_oat_combos,
    combo_name,
    validate_sweep_baseline,
    write_sweep_plan,
)

_BASELINE = {
    "aimip_convection": "tiedtke",
    "aimip_turbulence": "louis",
}
_SPACE = {
    "aimip_convection": ["bechtold", "edmf"],
    "aimip_turbulence": ["ysu"],
}


def test_build_oat_combos_structure():
    combos = build_oat_combos(_BASELINE, _SPACE)
    assert combos[0]["name"] == "combo_baseline"
    assert combos[0]["scheme"] == "tiedtke+louis"
    assert [c["name"] for c in combos[1:]] == [
        "combo_conv_bechtold",
        "combo_conv_edmf",
        "combo_turb_ysu",
    ]
    # every OAT arm changes exactly one dimension from the baseline
    for c in combos[1:]:
        diff = [k for k, v in c["overrides"].items() if _BASELINE[k] != v]
        assert diff == [c["dim"]]


def test_unknown_dimension_raises():
    with pytest.raises(ValueError, match="Unknown sweep dimension"):
        combo_name("not_a_dim", "x")
    with pytest.raises(ValueError, match="Unknown sweep dimension"):
        build_oat_combos(_BASELINE, {"not_a_dim": ["x"]})


def _write_baseline(tmp_path, rel, text):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return rel


def test_validate_sweep_baseline_rrtmgp_gate(tmp_path):
    rel = Path("config/sweep/baseline.yaml")
    _write_baseline(tmp_path, rel, "n_max: 21\naimip_radiation: rrtmgp\n")
    assert validate_sweep_baseline(tmp_path, rel)["n_max"] == 21
    # absent key -> effective default rrtmgp -> passes
    _write_baseline(tmp_path, rel, "n_max: 21\n")
    validate_sweep_baseline(tmp_path, rel)
    # gray baseline -> hard error (classical sweeps hold radiation fixed)
    _write_baseline(tmp_path, rel, "aimip_radiation: gray\n")
    with pytest.raises(ValueError, match="rrtmgp"):
        validate_sweep_baseline(tmp_path, rel)
    with pytest.raises(FileNotFoundError):
        validate_sweep_baseline(tmp_path, Path("config/missing.yaml"))


def test_write_sweep_plan_roundtrip(tmp_path):
    sweep_rel = Path("config/sweep/stage1")
    results_rel = Path("results/sweep_stage1")
    baseline_rel = _write_baseline(
        tmp_path, sweep_rel / "baseline.yaml", "aimip_radiation: rrtmgp\n"
    )
    combos = build_oat_combos(_BASELINE, _SPACE)
    manifest_path = write_sweep_plan(
        tmp_path,
        campaign="testcamp",
        combos=combos,
        sweep_dir_rel=sweep_rel,
        results_dir_rel=results_rel,
        baseline=_BASELINE,
        baseline_rel=baseline_rel,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["campaign"] == "testcamp"
    assert manifest["n_combos"] == 4
    # per-combo YAML pair written, suite recorded repo-relative
    for c in manifest["combos"]:
        suite = yaml.safe_load((tmp_path / c["suite"]).read_text())
        assert suite["variants"] == ["classical"]
        assert suite["base"] == str(baseline_rel)
        overlay = yaml.safe_load(
            (tmp_path / sweep_rel / c["name"] / "variant_classical.yaml").read_text()
        )
        assert overlay == c["overrides"]
    # slurm log dir pre-created (SLURM opens --output before the body runs)
    assert (tmp_path / results_rel / "slurm_logs").is_dir()
    # idempotent: second write produces identical manifest
    combos2 = build_oat_combos(_BASELINE, _SPACE)
    manifest2 = json.loads(
        write_sweep_plan(
            tmp_path,
            campaign="testcamp",
            combos=combos2,
            sweep_dir_rel=sweep_rel,
            results_dir_rel=results_rel,
            baseline=_BASELINE,
            baseline_rel=baseline_rel,
        ).read_text()
    )
    assert manifest2 == manifest
