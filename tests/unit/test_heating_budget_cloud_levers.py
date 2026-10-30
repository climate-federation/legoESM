"""Pure helpers of the #1521 cloud-lever mode in heating_budget.py and the
relaunch helper it uses."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_P = Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias/heating_budget.py"
_spec = importlib.util.spec_from_file_location("heating_budget", _P)
hb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hb)


def test_scale_inputs_multiplies_only_named_and_refuses_missing():
    bound = {"q_cloud": np.ones(3), "n_cloud": 2 * np.ones(3), "T": np.ones(3)}
    out = hb.scale_inputs(bound, {"q_cloud": 2.0, "n_cloud": 2.0})
    assert np.all(out["q_cloud"] == 2.0) and np.all(out["n_cloud"] == 4.0)
    assert out["T"] is bound["T"] and np.all(bound["q_cloud"] == 1.0)
    with pytest.raises(SystemExit):
        hb.scale_inputs({"q_ice": None}, {"q_ice": 2.0})


def test_scale_inputs_clips_scaled_cloud_fraction():
    out = hb.scale_inputs({"cloud_fraction_override": np.array([0.5, 0.99])},
                          {"cloud_fraction_override": 1.07})
    np.testing.assert_allclose(out["cloud_fraction_override"], [0.535, 1.0])


def test_gate_aborts_on_failure():
    hb.gate(True, "passes")
    with pytest.raises(SystemExit, match="GATE FAILED"):
        hb.gate(False, "fails")


def test_launch_argv_drops_the_mpi_flag(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "mpas_onestep_param_grad", _P.parent / "mpas_onestep_param_grad.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    (tmp_path / "slurm_logs").mkdir()
    (tmp_path / "arm").mkdir()
    (tmp_path / "arm" / "checkpoint_day_0010.npz").write_bytes(b"")
    (tmp_path / "slurm_logs" / "arm-1.out").write_text(
        "[chain] launching on 4 GPU(s): srun python scripts/run/run_amip.py "
        "--config x.yaml --days 30 --output o --distributed\n")
    monkeypatch.setattr(H, "ROOT", tmp_path)
    argv = H.launch_argv("arm", 10, tmp_path / "out")
    assert "--distributed" not in argv
    assert argv[:2] == ["--config", "x.yaml"]
    assert argv[argv.index("--days") + 1] == "1"
