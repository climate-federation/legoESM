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


def test_zonal_state_cli_prints_band_wind_and_checks_day(tmp_path, mesh, capsys):
    zs = _load("zonal_state")
    n = np.asarray(mesh.latCell).size
    ck = tmp_path / "checkpoint_day_0030.npz"
    np.savez(ck, T=np.full((n, 2), 250.0), u=_solid_body_edge_u(mesh, 20.0), p_s=np.full(n, 1.0e5),
             day=np.array(30.0))
    zs.main([str(ck), "3", "30"])
    row = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("k= 1"))
    u_eq = float(row.split("| u")[1].split()[3])   # 10S-10N band
    assert u_eq == pytest.approx(19.88, abs=0.05)
    with pytest.raises(ValueError, match="stamped day 30"):
        zs.main([str(ck), "3", "24"])


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


def test_ocean_heat_cli_counts_ssh(tmp_path):
    """End to end on a level-2 mesh: day-N = rebuilt IC warmed 1 K everywhere with
    SSH +2 m. Fixed rate = rho*c*sum(h0)*1/t; live minus fixed = rho*c*sum(T1*dh)/t,
    globally and on a band row."""
    import json
    from legoesm import constants
    from legoesm.forcing.analytical import qobs_sst_C
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.init_mpas import REST_T_DEEP_C, rest_state_mpas_ocean
    from legoesm.ocean.vertical import compute_layer_thickness, create_ocean_z_star

    nlev, Hmax, day, step = 4, 4000.0, 2, 768
    out = tmp_path / "out"; (out / "coupled").mkdir(parents=True)
    (out / "coupled" / "run_manifest.json").write_text(json.dumps(
        {"config": {"precision": "fp64", "resolved_config": {"cli_args": {"dt": 225.0}},
                    "coupled": {"ocean_nlev": nlev, "ocean_H_max_m": Hmax, "ocean_ic": "qobs"}}}))
    mesh = create_voronoi_mesh(2); z = create_ocean_z_star(nlev, H_max=Hmax)
    Ts = qobs_sst_C(np.degrees(np.asarray(mesh.latCell)))
    s0 = rest_state_mpas_ocean(mesh, z, T_water_init_C=Ts, T_deep=np.minimum(REST_T_DEEP_C, Ts),
                               H_max=Hmax, land_lat_threshold=90.0)
    T0, S0, H, wet = (np.asarray(f.data) for f in (s0.T, s0.S, s0.H_bathy, s0.land_mask))
    eta1 = np.full(T0.shape[0], 2.0)
    k = lambda n: f"ocean3d_.{n}.[<flat index 0>]"
    T1 = T0 + 1.0
    np.savez(out / f"coupled_day_{day:04d}.npz", **{k("T"): T1, k("S"): S0, k("eta"): eta1, k("H_bathy"): H,
             k("land_mask"): wet, "_atm_step": np.array(step)})
    r = subprocess.run([sys.executable, str(_DIR / "ocean_heat.py"), str(out), "2", str(day)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    line = next(l for l in r.stdout.splitlines() if l.startswith("global ocean heat-storage"))
    fixed, live = (float(x.split()[0]) for x in line.split("thickness ")[1:])
    h0 = np.asarray(compute_layer_thickness(0 * eta1, H, z))
    dh = np.asarray(compute_layer_thickness(eta1, H, z)) - h0
    lat = np.degrees(np.asarray(mesh.latCell)); area = np.asarray(mesh.areaCell) * (wet > 0.5)
    rc = constants.rho_ocean * constants.c_sw / (day * 86400.0)
    want_fixed = rc * np.sum(h0.sum(1) * area) / area.sum()
    want_vol = rc * np.sum((T1 * dh).sum(1) * area) / area.sum()
    assert abs(want_fixed) > 10.0 and abs(want_vol) > 10.0
    # T1 is saved from a float32 rebuild, the script rebuilds T0 in float64: the
    # ~1e-6 K residual on a 4 km column gives ~3e-7 relative on the fixed rate
    assert fixed == pytest.approx(want_fixed, rel=2e-6)
    assert live - fixed == pytest.approx(want_vol, abs=0.02)
    row = next(l for l in r.stdout.splitlines() if l.startswith(" -30..-10"))
    rf, rl = (float(x) for x in row.split("|")[2].split("/"))
    m = (lat >= -30) & (lat < -10) & (wet > 0.5); am = area[m]
    assert rf == pytest.approx(rc * np.sum(h0[m].sum(1) * am) / am.sum(), rel=2e-6, abs=0.06)
    assert rl - rf == pytest.approx(rc * np.sum((T1[m] * dh[m]).sum(1) * am) / am.sum(), rel=2e-6, abs=0.11)


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


def test_ocean_heat_refuses_checkpoint_of_another_day():
    oh = _load("ocean_heat")
    oh.check_ocean_day({"_atm_step": np.array(11520)}, 30, 225.0)
    with pytest.raises(ValueError, match="not day 30"):
        oh.check_ocean_day({"_atm_step": np.array(11000)}, 30, 225.0)
    with pytest.raises(ValueError, match="not day 60"):
        oh.check_ocean_day({"_atm_step": np.array(11520)}, 60, 225.0)


# ------------------------------------------------------- snapshot_zonal_rms
def test_far_from_ridge_wraps_longitude():
    ev = _load("snapshot_zonal_rms")
    got = ev.far_from_ridge(np.array([0.0, 29.0, 31.0, 331.0, 329.0, 180.0]), 0.0)
    np.testing.assert_array_equal(got, [False, False, True, False, True, True])
    np.testing.assert_array_equal(ev.far_from_ridge(np.array([350.0, 10.0, 60.0]), 20.0), [False, False, True])


def test_snapshot_rms_wave_one_and_zonally_uniform(mesh):
    ev = _load("snapshot_zonal_rms")
    lat = np.degrees(np.asarray(mesh.latCell)); lon = np.asarray(mesh.lonCell)
    area = np.asarray(mesh.areaCell); sel = np.ones(lat.size, bool)
    assert np.all(ev.snapshot_spatial_rms(np.full(lat.size, 7.0), lat, area, sel) < 1e-9)
    A = 2.0
    got = ev.snapshot_spatial_rms(A * np.cos(lon), lat, area, sel)
    np.testing.assert_allclose(got[1:-1], A / np.sqrt(2.0), rtol=EDDY_RTOL)


def test_snapshot_rms_refuses_empty_band(mesh):
    ev = _load("snapshot_zonal_rms")
    lat = np.degrees(np.asarray(mesh.latCell)); area = np.asarray(mesh.areaCell)
    with pytest.raises(ValueError, match="empty band"):
        ev.snapshot_spatial_rms(np.ones(lat.size), lat, area, lat < 0)


# ----------------------------------------------------------------- snap_maps
def test_panel_wind_is_snapshot_reconstruction(mesh):
    sm = _load("snap_maps")
    U, nlev = 15.0, 32
    lat = np.asarray(mesh.latCell)
    ck = {"T": np.tile(200.0 + np.arange(nlev), (lat.size, 1)), "p_s": np.full(lat.size, 1.0e5),
          # only the LOWEST level carries the wind: a wrong level index maps zero
          "u": np.where(np.arange(nlev) == nlev - 1, _solid_body_edge_u(mesh, U, nlev), 0.0),
          # a stale accumulator that must NOT be what is mapped
          "land_win_n": np.array(1.0), "land_win_sum_u_lowest": np.full(lat.size, -99.0)}
    f = sm.panel_fields(ck, mesh)
    mid = np.abs(np.degrees(lat)) < 60
    np.testing.assert_allclose(f[2][mid], U * np.cos(lat[mid]), atol=PANEL_ATOL)
    np.testing.assert_array_equal(f[0], 200.0 + nlev - 1)   # lowest level
    np.testing.assert_array_equal(f[3], 220.0)              # level 20
    np.testing.assert_array_equal(f[1], 1000.0)             # Pa -> hPa
    ck["T"] = ck["T"].copy(); ck["T"][0, -1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        sm.panel_fields(ck, mesh)


# ------------------------------------------------------------- report sbatch
def test_report_sbatch_maps_only_numbered_days_up_to_n(tmp_path):
    out = tmp_path / "run" / "out"; (out / "coupled").mkdir(parents=True)
    for d in ("0030", "0060", "0090", "final", "0030_backup"):
        (out / f"checkpoint_day_{d}.npz").write_text("")
    wt = tmp_path / "wt"; (wt / "packages" / "core").mkdir(parents=True)
    diag = tmp_path / "diag"; diag.mkdir()
    for g in (wt, diag):
        subprocess.run(["git", "init", "-q", str(g)], check=True)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["-C", str(wt), "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    sha = subprocess.run(["git", "-C", str(wt), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    py = tmp_path / "py"   # stub interpreter: answers the manifest read, echoes every other call
    py.write_text('#!/bin/bash\nif [ "$1" = -c ]; then echo %s %s False; else echo "CALL $*"; fi\n' % (wt, sha))
    py.chmod(0o755)
    # "0060" must mean day 60 (bash would read a leading-zero number as octal)
    r = subprocess.run(["bash", str(_SBATCH), str(out), "0060", "5", "0.0", str(py)], capture_output=True,
                       text=True, env={"PATH": "/usr/bin:/bin", "DIAG_DIR": str(diag)})
    assert r.returncode == 0, r.stderr
    assert any("zonal_state.py" in l and "checkpoint_day_0060.npz 5 60" in l for l in r.stdout.splitlines())
    maps = [l for l in r.stdout.splitlines() if "snap_maps.py" in l]
    assert len(maps) == 1 and "day_0030" in maps[0] and "day_0060" in maps[0]
    assert "day_0090" not in maps[0] and "final" not in maps[0] and "backup" not in maps[0]
    # a run the manifest records as launched from a dirty tree is refused
    py.write_text(py.read_text().replace(" False;", " True;"))
    r = subprocess.run(["bash", str(_SBATCH), str(out), "60", "5", "0.0", str(py)], capture_output=True,
                       text=True, env={"PATH": "/usr/bin:/bin", "DIAG_DIR": str(diag)})
    assert r.returncode != 0 and "dirty tree" in r.stderr
    py.write_text(py.read_text().replace(" True;", " False;"))
    # a run worktree that moved off the run's commit is refused
    subprocess.run(git + ["-C", str(wt), "commit", "-q", "--allow-empty", "-m", "y"], check=True)
    r = subprocess.run(["bash", str(_SBATCH), str(out), "60", "5", "0.0", str(py)], capture_output=True,
                       text=True, env={"PATH": "/usr/bin:/bin", "DIAG_DIR": str(diag)})
    assert r.returncode != 0 and "not at the run's commit" in r.stderr


def test_report_sbatch_parses_and_requires_diag_dir():
    assert subprocess.run(["bash", "-n", str(_SBATCH)]).returncode == 0
    r = subprocess.run(["bash", str(_SBATCH), "/nonexistent", "30"], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin"})
    assert r.returncode != 0 and "DIAG_DIR" in r.stderr
