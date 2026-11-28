"""Direct tests for the esoteric-climate checkpoint diagnostics
(``scripts/validate/esoteric/``): each reduction is checked on a synthetic field
with a known answer. The scripts are loaded from their files (scripts/ is not a
package); their directory goes on sys.path so their shared helper resolves.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pytest

_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate" / "esoteric"
# Tolerances from a measurement on the level-3 mesh (2026-11-28), with the
# pre-fix value each one rejects:
# band eastward wind, solid body U=20: reconstruction max err 0.055 m/s;
#   the old 2*mean(u_n*cos(angleEdge)) over edges erred up to 0.43 m/s.
ZONAL_ATOL = 0.06
# cell eastward wind |lat|<60: reconstruction max err 0.41 m/s (stale accumulator -99 m/s).
PANEL_ATOL = 0.5
# wave-1 snapshot RMS / (A/sqrt2), non-polar bands: within 2.5 %.
EDDY_RTOL = 0.03
_SBATCH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "cluster" / "esoteric" / "em_report.sbatch"


def _load(name):
    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))
    spec = importlib.util.spec_from_file_location(f"_esd_{name}", _DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mesh():
    from legoesm.grids.voronoi import create_voronoi_mesh
    return create_voronoi_mesh(3)


def _solid_body_edge_u(mesh, U, nlev=2):
    """Edge-normal components of u_east = U*cos(lat), v_north = 0."""
    un = U * np.cos(np.asarray(mesh.latEdge)) * np.cos(np.asarray(mesh.angleEdge))
    return np.repeat(un[:, None], nlev, axis=1)


# ---------------------------------------------------------------- zonal_state
def test_band_eastward_wind_matches_solid_body(mesh):
    zs, com = _load("zonal_state"), _load("_common")
    U = 20.0
    lat = np.degrees(np.asarray(mesh.latCell)); area = np.asarray(mesh.areaCell)
    got = com.band_means(zs.cell_u_east(_solid_body_edge_u(mesh, U), mesh), lat, area)[:, 0]
    want = com.band_means(U * np.cos(np.radians(lat)), lat, area)
    np.testing.assert_allclose(got, want, rtol=0, atol=ZONAL_ATOL)


def test_band_means_area_weighted_and_refuses_empty_band():
    com = _load("_common")
    lat = np.array([-80.0, -80.0, 0.0])
    got = com.band_means(np.array([1.0, 3.0, 5.0]), lat, np.array([3.0, 1.0, 1.0]),
                         bands=[(-90, -60), (-10, 10)])
    np.testing.assert_allclose(got, [1.5, 5.0])
    with pytest.raises(ValueError, match="no cells"):
        com.band_means(np.ones(3), lat, np.ones(3), bands=[(30, 60)])


# ---------------------------------------------------------------- ocean_heat
def test_storage_rate_uniform_warming():
    from legoesm import constants
    oh = _load("ocean_heat")
    h = np.full((3, 2), 50.0); T0 = np.full((3, 2), 10.0); area = np.array([1.0, 2.0, 3.0])
    days = 4
    want = constants.rho_ocean * constants.c_sw * 100.0 * 1.0 / (days * 86400.0)
    assert oh.storage_rate(T0, h, T0 + 1.0, h, area, days) == pytest.approx(want, rel=1e-12)


def test_storage_rate_counts_ssh_volume_change():
    """Same temperature, 1 m more water at 10 C: storage rises by rho*c*10*1."""
    from legoesm import constants
    oh = _load("ocean_heat")
    T = np.full((2, 2), 10.0); area = np.ones(2)
    h0 = np.full((2, 2), 50.0); h1 = h0 * (101.0 / 100.0)
    want = constants.rho_ocean * constants.c_sw * 10.0 * 1.0 / 86400.0
    assert oh.storage_rate(T, h0, T, h1, area, 1) == pytest.approx(want, rel=1e-12)


def test_find_key_nested():
    com = _load("_common")
    assert com.find_key({"a": {"b": {"ocean_nlev": 20}}}, "ocean_nlev") == 20
    assert com.find_key({"a": {}}, "x") is None


def test_ocean_heat_refuses_unrebuildable_runs():
    oh = _load("ocean_heat")
    oh.check_supported("qobs", np.full(4, 5200.0), 5200.0)
    with pytest.raises(ValueError, match="qobs"):
        oh.check_supported("rest", np.full(4, 5200.0), 5200.0)
    with pytest.raises(ValueError, match="partial-cell"):
        oh.check_supported("qobs", np.array([5200.0, 3000.0]), 5200.0)


def test_ocean_heat_refuses_mismatched_checkpoints():
    oh = _load("ocean_heat")
    atm = {"day": np.array(30.0), "step": np.array(11520)}
    oh.check_same_instant(atm, {"_atm_step": np.array(11520)}, 30)
    with pytest.raises(ValueError, match="disagree"):
        oh.check_same_instant(atm, {"_atm_step": np.array(11000)}, 30)
    with pytest.raises(ValueError, match="disagree"):
        oh.check_same_instant(atm, {"_atm_step": np.array(11520)}, 60)


# ------------------------------------------------------------- eddy_variance
def test_far_from_ridge_wraps_longitude():
    ev = _load("eddy_variance")
    got = ev.far_from_ridge(np.array([0.0, 29.0, 31.0, 331.0, 329.0, 180.0]), 0.0)
    np.testing.assert_array_equal(got, [False, False, True, False, True, True])
    np.testing.assert_array_equal(ev.far_from_ridge(np.array([350.0, 10.0, 60.0]), 20.0), [False, False, True])


def test_snapshot_rms_wave_one_and_zonally_uniform(mesh):
    ev = _load("eddy_variance")
    lat = np.degrees(np.asarray(mesh.latCell)); lon = np.asarray(mesh.lonCell)
    area = np.asarray(mesh.areaCell); sel = np.ones(lat.size, bool)
    assert np.all(ev.snapshot_spatial_rms(np.full(lat.size, 7.0), lat, area, sel) < 1e-9)
    A = 2.0
    got = ev.snapshot_spatial_rms(A * np.cos(lon), lat, area, sel)
    np.testing.assert_allclose(got[1:-1], A / np.sqrt(2.0), rtol=EDDY_RTOL)


def test_snapshot_rms_refuses_empty_band(mesh):
    ev = _load("eddy_variance")
    lat = np.degrees(np.asarray(mesh.latCell)); area = np.asarray(mesh.areaCell)
    with pytest.raises(ValueError, match="empty band"):
        ev.snapshot_spatial_rms(np.ones(lat.size), lat, area, lat < 0)


# ----------------------------------------------------------------- snap_maps
def test_panel_wind_is_snapshot_reconstruction(mesh):
    sm = _load("snap_maps")
    U, nlev = 15.0, 32
    lat = np.asarray(mesh.latCell)
    ck = {"T": np.full((lat.size, nlev), 280.0), "p_s": np.full(lat.size, 1.0e5),
          # only the LOWEST level carries the wind: a wrong level index maps zero
          "u": np.where(np.arange(nlev) == nlev - 1, _solid_body_edge_u(mesh, U, nlev), 0.0),
          # a stale accumulator that must NOT be what is mapped
          "land_win_n": np.array(1.0), "land_win_sum_u_lowest": np.full(lat.size, -99.0)}
    f = sm.panel_fields(ck, mesh)
    mid = np.abs(np.degrees(lat)) < 60
    np.testing.assert_allclose(f[2][mid], U * np.cos(lat[mid]), atol=PANEL_ATOL)
    ck["T"] = ck["T"].copy(); ck["T"][0, -1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sm.panel_fields(ck, mesh)


# ------------------------------------------------------------- report sbatch
def test_report_sbatch_maps_only_numbered_days_up_to_n(tmp_path):
    out = tmp_path / "run" / "out"; (out / "coupled").mkdir(parents=True)
    for d in ("0030", "0060", "0090", "final"):
        (out / f"checkpoint_day_{d}.npz").write_text("")
    wt = tmp_path / "wt"; (wt / "packages" / "core").mkdir(parents=True)
    diag = tmp_path / "diag"; diag.mkdir()
    for g in (wt, diag):
        subprocess.run(["git", "init", "-q", str(g)], check=True)
    (out / "coupled" / "run_manifest.json").write_text('{"run": {"cwd": "%s"}}' % wt)
    py = tmp_path / "py"   # stub interpreter: answers the manifest read, echoes every other call
    py.write_text('#!/bin/bash\nif [ "$1" = -c ]; then echo %s; else echo "CALL $*"; fi\n' % wt)
    py.chmod(0o755)
    r = subprocess.run(["bash", str(_SBATCH), str(out), "60", "5", "0.0", str(py)], capture_output=True,
                       text=True, env={"PATH": "/usr/bin:/bin", "DIAG_DIR": str(diag)})
    assert r.returncode == 0, r.stderr
    maps = [l for l in r.stdout.splitlines() if "snap_maps.py" in l]
    assert len(maps) == 1 and "day_0030" in maps[0] and "day_0060" in maps[0]
    assert "day_0090" not in maps[0] and "final" not in maps[0]


def test_report_sbatch_parses_and_requires_diag_dir():
    assert subprocess.run(["bash", "-n", str(_SBATCH)]).returncode == 0
    r = subprocess.run(["bash", str(_SBATCH), "/nonexistent", "30"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "DIAG_DIR" in r.stderr
