"""Test the WeatherBench Stage-1 sweep generator (config emitter). stdlib/yaml only."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_GEN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run" / "run_wb_sweep_stage1.py"
_spec = importlib.util.spec_from_file_location("run_wb_sweep_stage1", _GEN_PATH)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)


def test_combo_name():
    # naming now lives in the shared planner (D6); the script imports it
    from legoesm.training.sweep_planner import combo_name
    assert combo_name("aimip_convection", "bechtold") == "combo_conv_bechtold"
    assert combo_name("aimip_cloud", "sundqvist") == "combo_cloud_sundqvist"
    assert combo_name("aimip_microphysics", "thompson") == "combo_micro_thompson"
    with pytest.raises(ValueError, match="Unknown sweep dimension"):
        combo_name("aimip_convectoin", "bechtold")  # typo'd dim must raise


def test_generator_writes_expected_combos(tmp_path, monkeypatch):
    # fake baseline so the existence check passes
    (tmp_path / "config" / "wb" / "sweep" / "stage1").mkdir(parents=True)
    (tmp_path / "config" / "wb" / "sweep" / "stage1"
     / "baseline_classical_t63_rrtmgp.yaml").write_text("n_max: 63\n")
    (tmp_path / "scripts" / "run").mkdir(parents=True)

    monkeypatch.setattr(sys, "argv", ["gen", "--repo-root", str(tmp_path)])
    gen.main()

    manifest = json.loads(
        (tmp_path / "config" / "wb" / "sweep" / "stage1" / "manifest.json").read_text())
    names = {c["name"] for c in manifest["combos"]}
    assert "combo_baseline" in names
    assert {"combo_conv_bechtold", "combo_conv_edmf", "combo_turb_ysu", "combo_turb_edmf",
            "combo_gwd_hines", "combo_micro_kessler", "combo_micro_morrison",
            "combo_micro_thompson", "combo_cloud_sundqvist"} <= names
    assert manifest["n_combos"] == 10          # baseline + 9 curated alternatives
    assert manifest["campaign"] == "weatherbench"
    # all 5 families explicit in the baseline (incl. cloud, for manifest analysis)
    assert set(manifest["baseline"]) == {
        "aimip_convection", "aimip_turbulence", "aimip_gwd",
        "aimip_microphysics", "aimip_cloud"}

    # every OAT arm changes EXACTLY one family from the baseline (all-active, no 'none')
    base = manifest["baseline"]
    for c in manifest["combos"]:
        if c["dim"] == "baseline":
            continue
        assert "none" not in c["overrides"].values(), c["name"]
        diff = [k for k, v in c["overrides"].items() if base.get(k) != v]
        assert diff == [c["dim"]], c["name"]

    # runner sbatch emitted + points at the worktree
    runner = (tmp_path / "scripts" / "run" / "_wb_sweep_stage1_runner.sbatch").read_text()
    assert "run_aimip.py" in runner and "--variants classical" in runner
    assert "--resume" in runner                # walltime-kill-safe resubmit
    assert "scripts/cluster/wb_forecast/env.sh" in runner


def test_baseline_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gen", "--repo-root", str(tmp_path)])
    with pytest.raises(FileNotFoundError):
        gen.main()
