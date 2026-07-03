"""Unit tests for scripts/validate/ocean_fidelity/veros_global_freerun.py.

The module lives outside ``src/`` so we add ``scripts/`` to sys.path (same
pattern as test_run_acc_freerun.py).  These lock the shared metrics / ratio
helpers that the 4°/1°/flexible global free-run drivers now route through.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def freerun_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import validate.ocean_fidelity.veros_global_freerun as mod  # type: ignore
        importlib.reload(mod)
        return mod
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def _fake_state(ny, nx, nz, *, u_val=0.0, T_val=4.0, S_val=34.0):
    """Build a duck-typed LatLonCGridOceanState-like object (wall rows incl.)."""
    def fld(arr):
        return SimpleNamespace(data=np.asarray(arr, dtype=np.float64))

    u = np.full((ny + 2, nx + 1, nz), u_val)
    v = np.full((ny + 3, nx, nz), 0.0)
    T = np.full((ny + 2, nx, nz), T_val)
    S = np.full((ny + 2, nx, nz), S_val)
    tke = np.full((ny + 2, nx, nz - 1), 1e-6)
    eke = np.full((ny + 2, nx, nz - 1), 1e-8)
    psi = np.zeros((ny + 3, nx + 1))
    return SimpleNamespace(
        u=fld(u), v=fld(v), T=fld(T), S=fld(S),
        tke=fld(tke), eke=fld(eke), psi=psi,
    )


def _fake_recipe(ny, nx, nz, rho0=1024.0):
    is_active = np.ones((ny + 2, nx, nz))
    grid = SimpleNamespace(lat=np.deg2rad(np.linspace(-80.0, 80.0, ny + 2)))
    return SimpleNamespace(
        z_coord=SimpleNamespace(is_active=is_active),
        grid=grid,
        model_config=SimpleNamespace(rho_0=rho0),
    )


def test_compute_metrics_rest_state_is_quiescent(freerun_module):
    ny, nx, nz = 4, 6, 3
    state = _fake_state(ny, nx, nz, u_val=0.0, T_val=4.0, S_val=34.0)
    recipe = _fake_recipe(ny, nx, nz)
    area = np.ones((ny, 1))
    dz = np.array([10.0, 20.0, 30.0])
    m = freerun_module.compute_metrics(state, recipe, area, dz)
    assert m["total_ke_j"] == 0.0          # u = v = 0
    assert m["max_abs_u"] == 0.0
    assert m["psi_range_sv"] == 0.0
    assert m["vol_mean_T"] == pytest.approx(4.0)
    assert m["vol_mean_S"] == pytest.approx(34.0)
    assert m["finite"] is True
    # tke/eke means == the seeded floors.
    assert m["mean_tke"] == pytest.approx(1e-6)
    assert m["mean_eke"] == pytest.approx(1e-8)


def test_compute_metrics_ke_scales_with_velocity(freerun_module):
    ny, nx, nz = 4, 6, 2
    recipe = _fake_recipe(ny, nx, nz, rho0=1024.0)
    area = np.ones((ny, 1))
    dz = np.array([10.0, 10.0])
    m0 = freerun_module.compute_metrics(_fake_state(ny, nx, nz, u_val=0.5),
                                        recipe, area, dz)
    m1 = freerun_module.compute_metrics(_fake_state(ny, nx, nz, u_val=1.0),
                                        recipe, area, dz)
    # KE ∝ u² ⇒ doubling u quadruples KE.
    assert m1["total_ke_j"] == pytest.approx(4.0 * m0["total_ke_j"])
    assert m1["max_abs_u"] == pytest.approx(1.0)


def test_read_nc_casts_and_transposes(freerun_module, tmp_path):
    h5netcdf = pytest.importorskip("h5netcdf")
    p = tmp_path / "f.nc"
    arr = np.arange(6, dtype="int32").reshape(2, 3)   # (x, y) on disk
    with h5netcdf.File(str(p), "w") as f:
        f.dimensions = {"x": 2, "y": 3}
        v = f.create_variable("foo", ("x", "y"), dtype="int32")
        v[:] = arr
    out = freerun_module.read_nc(str(p), "foo")
    assert out.dtype == np.float64                    # dtype="float" cast
    np.testing.assert_array_equal(out, arr.T.astype(float))


def _import_driver(name):
    sys.path.insert(0, str(SCRIPTS_DIR))
    try:
        import importlib
        return importlib.import_module(name)
    finally:
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass


def test_drivers_delegate_with_recipe_area_and_dz(freerun_module, monkeypatch):
    """Each migrated driver's ``compute_metrics`` wrapper must forward the
    recipe-specific ``area``/``dz`` into the shared helper unchanged — i.e. the
    only per-driver lines (area = veros_area_t_*(lat); dz = ...) reach the
    shared metrics intact.  A synthetic recipe (real grid.lat + dz arrays)
    exercises the wrapper without needing the Veros forcing assets.
    """
    captured = {}

    def spy(state, recipe, area, dz):
        captured["area"] = np.asarray(area)
        captured["dz"] = np.asarray(dz)
        return {"ok": True}

    def _lat_with_walls(n_interior):
        """Synthetic lat row (radians) with the 2 wall rows the wrappers
        strip via ``[1:-1]`` — interior length == ``n_interior``."""
        return np.deg2rad(np.linspace(-78.0, 78.0, n_interior + 2))

    monkeypatch.setattr(freerun_module, "compute_metrics", spy)

    # --- 1deg: area = veros_area_t_1deg(lat); dz = recipe.z_coord.dz_ref ---
    from legoesm.ocean.fidelity.veros_global_1deg_recipe import (
        NY as NY1,
        veros_area_t_1deg,
    )
    drv1 = _import_driver("validate.ocean_fidelity.run_global_1deg_freerun")
    lat1 = _lat_with_walls(NY1)
    dz1 = np.array([10.0, 20.0, 30.0])
    recipe1 = SimpleNamespace(
        grid=SimpleNamespace(lat=lat1),
        z_coord=SimpleNamespace(dz_ref=dz1))
    monkeypatch.setattr(drv1, "_shared_freerun", lambda: freerun_module)
    drv1.compute_metrics(object(), recipe1)
    np.testing.assert_array_equal(
        captured["area"], veros_area_t_1deg(np.degrees(lat1)[1:-1])[:, None])
    np.testing.assert_array_equal(captured["dz"], dz1)

    # --- 4deg: area = veros_area_t(lat); dz = GLOBAL4_DDZ ---
    from legoesm.ocean.fidelity.veros_global_4deg_recipe import (
        GLOBAL4_DDZ,
        NY as NY4,
        veros_area_t,
    )
    drv4 = _import_driver("validate.ocean_fidelity.run_global_4deg_freerun")
    lat4 = _lat_with_walls(NY4)
    recipe4 = SimpleNamespace(grid=SimpleNamespace(lat=lat4))
    monkeypatch.setattr(drv4, "_shared_freerun", lambda: freerun_module)
    drv4.compute_metrics(object(), recipe4)
    np.testing.assert_array_equal(
        captured["area"], veros_area_t(np.degrees(lat4)[1:-1])[:, None])
    np.testing.assert_array_equal(captured["dz"], GLOBAL4_DDZ)

    # --- flexible: area = veros_area_t_flexible(lat, dyt); dz = dzt[::-1] ---
    from legoesm.ocean.fidelity.veros_global_flexible_recipe import (
        NY as NYF,
        global_flexible_dyt_deg,
        global_flexible_dzt_veros,
        veros_area_t_flexible,
    )
    drvf = _import_driver("validate.ocean_fidelity.run_global_flexible_freerun")
    latf = _lat_with_walls(NYF)
    recipef = SimpleNamespace(grid=SimpleNamespace(lat=latf))
    monkeypatch.setattr(drvf, "_shared_freerun", lambda: freerun_module)
    drvf.compute_metrics(object(), recipef)
    np.testing.assert_array_equal(
        captured["area"],
        veros_area_t_flexible(
            np.degrees(latf)[1:-1], global_flexible_dyt_deg())[:, None])
    np.testing.assert_array_equal(captured["dz"], global_flexible_dzt_veros()[::-1])


def test_print_comparison_runs(freerun_module, tmp_path, capsys):
    ref = {"yearly": [{"year": 1, "total_ke_j": 2.0, "psi_min_sv": -1.0,
                       "psi_max_sv": 1.0, "psi_range_sv": 2.0, "vol_mean_T": 4.0,
                       "vol_mean_S": 34.0, "max_abs_u": 0.5, "mean_eke": 1e-8}]}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    yearly = [{"year": 1.0, "total_ke_j": 4.0, "psi_min_sv": -2.0,
               "psi_max_sv": 2.0, "psi_range_sv": 4.0, "vol_mean_T": 8.0,
               "vol_mean_S": 34.0, "max_abs_u": 1.0, "mean_eke": 2e-8}]
    freerun_module.print_comparison(yearly, str(p))
    out = capsys.readouterr().out
    assert "per-year ratios" in out
    assert "2.000" in out      # 4/2 = 2.0 ratio for total_ke_j
