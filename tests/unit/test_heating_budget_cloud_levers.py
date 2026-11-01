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


def _helper():
    spec = importlib.util.spec_from_file_location(
        "mpas_onestep_param_grad", _P.parent / "mpas_onestep_param_grad.py")
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    return H


def test_launch_argv_falls_back_to_the_first_link_manifest(tmp_path, monkeypatch):
    import json
    H = _helper()
    run = tmp_path / "season"
    run.mkdir()
    (run / "checkpoint_day_0005.npz").write_bytes(b"")
    link1 = ["scripts/run/run_amip.py", "--config", "x.yaml", "--start-day", "0",
             "--days", "31", "--output", "o", "--distributed"]
    link2 = link1[:5] + ["--restart-from", "c.npz", "--days", "59"]
    (run / "run_manifest.link1.json").write_text(json.dumps({"run": {"argv": link1}}))
    (run / "run_manifest.json").write_text(json.dumps({"run": {"argv": link2}}))
    monkeypatch.setattr(H, "ROOT", tmp_path)
    argv = H.launch_argv("season", 5, tmp_path / "out")
    assert argv[:4] == ["--config", "x.yaml", "--start-day", "0"]
    assert "--distributed" not in argv and argv.count("--days") == 1
    assert argv[argv.index("--restart-from") + 1] == str(run / "checkpoint_day_0005.npz")


def test_launch_argv_day_zero_starts_from_the_initial_state(tmp_path, monkeypatch):
    import json
    H = _helper()
    run = tmp_path / "season"
    run.mkdir()
    (run / "run_manifest.json").write_text(json.dumps(
        {"run": {"argv": ["run_amip.py", "--config", "x.yaml", "--ic-path", "ic.zarr"]}}))
    monkeypatch.setattr(H, "ROOT", tmp_path)
    argv = H.launch_argv("season", 0, tmp_path / "out")
    assert "--restart-from" not in argv
    assert argv[argv.index("--ic-path") + 1] == "ic.zarr"
    assert argv[argv.index("--days") + 1] == "1"


def test_cadence_reads_the_write_step_multiplier():
    assert hb._cadence({}) == 1
    assert hb._cadence({"physics_cadence": "off", "physics_cadence_steps": 16}) == 1
    assert hb._cadence({"physics_cadence": "write", "physics_cadence_steps": 16}) == 16


def _window_case(tmp_path, monkeypatch, turb_calls=6):
    from types import SimpleNamespace as NS
    monkeypatch.setattr(hb.H, "ROOT", tmp_path)
    shp, n, dt, ncad, nmm = (3, 2), 4, 10.0, 2, 3
    rad, conv, turb = 1e-5 * np.ones(shp), 2e-5 * np.ones(shp), 4e-6 * np.ones(shp)
    # 4 steps: one full-radiation and one held-radiation physics call; turbulence
    # runs nmm macmic passes per call, each applied with weight 1/nmm
    acc = {"radiation": rad, "rad_held": rad, "convection": 2 * conv,
           "turbulence": 2 * nmm * turb, "applied": 2 * ncad * (rad + conv + turb)}
    ncall = {"applied": 2, "radiation": 1, "rad_held": 1, "convection": 2,
             "turbulence": turb_calls}
    T0 = 250.0 * np.ones(shp)
    mesh = NS(areaCell=np.ones(3), latCell=np.array([-0.5, 0.0, 0.5]))
    model = NS(mesh=mesh, sigma_coord=NS(sigma_full=np.array([0.5, 0.9])))
    win = {"n": n, "dt": dt, "T0": T0, "ps0": 1e5 * np.ones(3), "model": model,
           "last": NS(T=NS(data=T0 + n * dt * 5e-5))}
    builds = [({}, (), {"need_rad": True, "physics_cadence": "write",
                         "physics_cadence_steps": ncad, "cld_macmic_num_steps": nmm})]
    return NS(run="r", day=0, window_days=1), win, acc, ncall, builds


def test_window_budget_scales_by_cadence_macmic_and_held_radiation(tmp_path, monkeypatch):
    args, win, acc, ncall, builds = _window_case(tmp_path, monkeypatch)
    assert hb.window_budget(args, win, acc, ncall, builds, tmp_path, 0.0) == 0
    z = np.load(tmp_path / "heating_window_r_d0000_1d.npz")
    g = {k: dict(z[f"gm_{k}"])[850.0] for k in
         ("radiation", "convection", "turbulence", "physics_applied", "physics_closure",
          "total", "dynamics_and_fixers")}
    np.testing.assert_allclose(g["radiation"], 1e-5 * 86400)
    np.testing.assert_allclose(g["convection"], 2e-5 * 86400)
    np.testing.assert_allclose(g["turbulence"], 4e-6 * 86400)
    np.testing.assert_allclose(g["physics_applied"], 3.4e-5 * 86400)
    np.testing.assert_allclose(g["physics_closure"], 0.0, atol=1e-12)
    np.testing.assert_allclose(g["total"], 5e-5 * 86400)
    np.testing.assert_allclose(g["dynamics_and_fixers"], 1.6e-5 * 86400)


def test_window_budget_refuses_unexplained_call_counts(tmp_path, monkeypatch):
    args, win, acc, ncall, builds = _window_case(tmp_path, monkeypatch, turb_calls=5)
    with pytest.raises(SystemExit, match="GATE FAILED"):
        hb.window_budget(args, win, acc, ncall, builds, tmp_path, 0.0)
def test_replace_deck_swaps_the_one_config_value():
    argv = ["--config", "config/amip/amip_production.yaml", "--days", "1"]
    out = hb.replace_deck(argv, "config/amip/amip_production_prognostic_nc.yaml")
    assert out == ["--config", "config/amip/amip_production_prognostic_nc.yaml", "--days", "1"]
    assert argv[1] == "config/amip/amip_production.yaml"          # input untouched
    for bad in (["--days", "1"], ["--config", "a", "--config", "b"], ["--config"]):
        with pytest.raises(SystemExit, match="exactly one --config"):
            hb.replace_deck(bad, "x.yaml")
