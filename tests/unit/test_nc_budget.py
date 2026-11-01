"""Known-answer controls for the pure helpers of the #2216 droplet-number budget
probe (scripts/validate/amip_bias/nc_budget.py) and its relaunch-line parser."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_D = Path(__file__).resolve().parents[2] / "scripts/validate/amip_bias"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _D / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nb = _load("nc_budget")


def test_aggregate_known_answer():
    # two 1-kg cells at rho 1 (2 m^3): 1 and 3 per mg -> 2 cm^-3; a sink of
    # 1e6 /kg/day in both -> -1 cm^-3/day, sink time 2 days
    mass, rho, mk = np.ones((2, 1)), np.ones((2, 1)), np.ones((2, 1), bool)
    n0 = np.array([[1e6], [3e6]])
    rows = {"sink": np.full((2, 1), -1e6 / 86400.0), "src": np.full((2, 1), 2e6 / 86400.0)}
    inv, r, tau_all, tau = nb.aggregate(rows, n0, mass, rho, mk)
    assert inv == pytest.approx(2.0)
    assert r["sink"] == pytest.approx(-1.0) and r["src"] == pytest.approx(2.0)
    assert tau_all == pytest.approx(48.0) and tau["sink"] == pytest.approx(48.0)
    assert tau["src"] == np.inf
    # the mask excludes a cell
    inv1, r1, _, _ = nb.aggregate(rows, n0, mass, rho, np.array([[True], [False]]))
    assert inv1 == pytest.approx(1.0) and r1["sink"] == pytest.approx(-1.0)


def test_incloud_known_answer():
    # 100 cm^-3 grid-mean at half cloud cover -> 200 in-cloud; the clear cell
    # (q_c below the threshold) is excluded
    n = np.full((3, 1), 1e8)
    cf = np.array([[0.5], [0.5], [0.0]])
    qc = np.array([[1e-5], [1e-5], [0.0]])
    ic = nb.incloud(n, cf, qc, np.ones((3, 1)), np.ones((3, 1), bool))
    assert ic["n_cells"] == 2 and ic["n_cf_zero"] == 0
    assert ic["grid_mean"] == pytest.approx(100.0) and ic["in_cloud"] == pytest.approx(200.0)
    assert ic["median_in_cloud"] == pytest.approx(200.0) and ic["cloud_fraction"] == pytest.approx(0.5)


def test_incloud_median_skips_zero_cloud_fraction():
    ic = nb.incloud(np.full((2, 1), 1e8), np.array([[0.5], [0.0]]), np.full((2, 1), 1e-5),
                    np.ones((2, 1)), np.ones((2, 1), bool))
    assert ic["n_cf_zero"] == 1 and ic["median_in_cloud"] == pytest.approx(200.0)


def test_incloud_refuses_nonfinite():
    with pytest.raises(SystemExit, match="non-finite N_c"):
        nb.incloud(np.array([[np.nan]]), np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)),
                   np.ones((1, 1), bool))


def test_masks_regions_and_layers():
    m = nb.masks(np.array([-50.0, 0.0]), np.array([[800.0, 500.0], [800.0, 500.0]]))
    np.testing.assert_array_equal(m[("SO 40-65S", "column")], [[True, True], [False, False]])
    np.testing.assert_array_equal(m[("SO 40-65S", "950-700 hPa")], [[True, False], [False, False]])
    np.testing.assert_array_equal(m[("global", "950-700 hPa")], [[True, False], [True, False]])


def test_closes_catches_a_missing_term_and_mean_calls_weights():
    a, b = np.array([1.0, -2.0]), np.array([0.5, 0.25])
    assert nb.closes([a, b], a + b)[0]
    assert not nb.closes([a], a + b)[0]
    # a complete miss in a tiny cell next to a large one is still caught
    big, tiny = np.array([1e6, 1e-9]), np.array([0.0, 1e-9])
    assert not nb.closes([big - tiny], big)[0]
    np.testing.assert_allclose(nb.mean_calls([a, b, b]), (a + 2 * b) / 3)
    with pytest.raises(SystemExit, match="GATE FAILED"):
        nb.gate(False, "x")


def test_figure_writes(tmp_path):
    res = {(rn, ln): (1.0, {"a": -1.0, "b": 2.0}, 24.0, {}) for rn in nb.REGIONS for ln in nb.LAYERS}
    nb.figure(res, tmp_path / "f.png", "t")
    assert (tmp_path / "f.png").stat().st_size > 0


def test_launch_argv_reads_monthly_baseline_lines(tmp_path, monkeypatch):
    H = _load("mpas_onestep_param_grad")
    (tmp_path / "arm").mkdir()
    (tmp_path / "arm" / "checkpoint_day_0030.npz").write_bytes(b"")
    (tmp_path / "arm" / "slurm-1.out").write_text(
        "[jan] ARM ngpu=4\n[jan] python -u scripts/run/run_amip.py --config x.yaml "
        "--output o --distributed --days 30\n")
    monkeypatch.setattr(H, "ROOT", tmp_path)
    argv = H.launch_argv("arm", 30, tmp_path / "out")
    assert argv[:2] == ["--config", "x.yaml"] and "--distributed" not in argv
    assert argv[argv.index("--days") + 1] == "1"


def test_terrain_override_reports_only_small_magnitude_drift(monkeypatch):
    from types import SimpleNamespace
    from legoesm.driver.model_driver import ModelDriver
    msg = {"text": ""}

    def refuse(self):
        raise RuntimeError(msg["text"])
    monkeypatch.setattr(ModelDriver, "_check_restart_terrain", refuse)
    nb.report_terrain_drift()
    check = ModelDriver._check_restart_terrain

    def fake(drift):
        b = np.full(4, 5.0e4)
        return SimpleNamespace(_phis_data=b, state=SimpleNamespace(
            phis=SimpleNamespace(data=b + drift)))
    magnitude = "restart: the checkpoint's terrain differs from this run's terrain product (max |d| x)"
    msg["text"] = magnitude
    check(fake(5.0e4 * 1.4e-6))                       # measured size: reported
    with pytest.raises(RuntimeError):                 # above the cap: refused
        check(fake(5.0e4 * 2.0e-5))
    msg["text"] = "restart: the checkpoint's terrain has shape (3,), this run's terrain product (4,)"
    with pytest.raises(RuntimeError):                 # any other refusal: refused
        check(fake(0.0))
