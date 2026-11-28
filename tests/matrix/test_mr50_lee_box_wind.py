"""Mountain Rossby 5-0 cross-grid lee-wind score: peak over matched output
times of the area-weighted mean lowest-level |V| over whole 10-degree boxes
of the lee region 0-45N x 90-150E (only the 0-40N boxes are whole)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")


@pytest.fixture(scope="module")
def M():
    spec = importlib.util.spec_from_file_location("_atmmatrix_mr50_lee",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _canvas(res=1.0):
    lat = np.arange(-90 + res / 2, 90, res)
    lon = np.arange(-180 + res / 2, 180, res)          # wraps through 360
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    area = np.cos(np.radians(LA))
    return LA, LO, area


def test_box_mean_peak(M):
    LA, LO, A = _canvas()
    lo = np.mod(LO, 360.0)
    spd = np.full(LA.shape, 5.0)
    # a strong jet OUTSIDE the region (west of 90E, and in the 40-45N sliver)
    spd[(LA > 10) & (LA < 20) & (lo > 60) & (lo < 89)] = 99.0
    spd[(LA > 40) & (LA < 45) & (lo > 100) & (lo < 140)] = 99.0
    # one box 10-20N x 110-120E: half 30, half 50 m/s (equal-area halves
    # in longitude) -> box mean 40
    box = (LA > 10) & (LA < 20) & (lo > 110) & (lo < 120)
    spd[box & (lo < 115)] = 30.0
    spd[box & (lo > 115)] = 50.0
    got = M._lee_box_wind(spd, LA, LO)
    assert got == pytest.approx(40.0, rel=1e-12)


def test_uniform_and_native_shapes(M):
    LA, LO, A = _canvas(2.5)
    spd = np.full(LA.shape, 7.0)
    assert M._lee_box_wind(spd, LA, LO) == pytest.approx(7.0)
    # flattened (unstructured) input gives the same value
    assert M._lee_box_wind(spd.ravel(), LA.ravel(),
                           LO.ravel()) == pytest.approx(7.0)


def test_cell_straddling_box_edge_is_split_by_overlap(M):
    """A 5-degree cell centred ON a box edge (lon 120E) counts half in each
    neighbouring box, not all in the box holding its centre."""
    lat = np.arange(-87.5, 90, 5.0)
    lon = np.arange(0.0, 360, 5.0)             # centres on 120E, 125E, ...
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    spd = np.zeros(LA.shape)
    spd[(np.abs(LA - 12.5) < 1) & (np.abs(LO - 120.0) < 1)] = 80.0
    got = M._lee_box_wind(spd, LA, LO)
    # box 10-20N x 110-120E and x 120-130E each hold a quarter-area
    # (half in lat-band x half in lon) of the fast cell: overlap
    # 2.5 deg x 5 deg of a 10 x 10 box ~ 1/8 of the box
    assert got == pytest.approx(80.0 / 8.0, rel=0.03)


def test_mr50_cadence_exact_on_row_steps(M):
    """9600 s is a whole number of steps on every row's dt (cube duo
    1920 s, lat-lon / ico 200 s, spectral 600 s): matched times, no offset."""
    for dt in (1920.0, 200.0, 600.0):
        n = int(2.0 * 86400 / dt)
        steps, off = M._matched_diag_steps(dt, n, M._MR50_DIAG_CADENCE_S,
                                           2.0 * 86400)
        assert off == 0.0 and len(steps) == 18


def test_every_mr50_lane_reports_lee_box_wind():
    src = _SCRIPT.read_text()
    assert src.count('out["lee_box_wind"] = _lee_box_wind(') == 4
    # the cube row scores on its native FV3 cells, not nearest centres
    assert ("duo.lon_deg, _duo_cell_corners_xyz(duo.grid.ctx_np))"
            in src)
    assert "dt, n_steps, _MR50_DIAG_CADENCE_S, days * 86400.0)" in src


class _Captured(Exception):
    pass


@pytest.mark.parametrize("grid", ["spectral", "latlon"])
def test_runner_row_wires_matched_lee_series(M, grid, tmp_path,
                                             monkeypatch):
    """The runner's own MR50 row: matched steps every 9600 s with no
    offset, the gated series' 6-hourly cadence untouched, and the matched
    callback giving the IC lee score (solid-body u0 cos(lat): the 0-10N
    boxes, area mean of 20 cos(lat) m/s, ~19.9)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    got = {}

    def _stop(step_fn, state, dt, n_steps, check_fn, scalar_fn, extract_fn,
              diag_every, key_array_fn, **kw):
        got.update(state=state, dt=dt, n=n_steps, every=diag_every,
                   scalar_fn=scalar_fn, **kw)
        raise _Captured
    monkeypatch.setattr(M, "_run_timeloop", _stop)
    row = [t for t in M._build_test_matrix()
           if t.case == "mountain_rossby_5_0" and t.grid_type == grid][0]
    with pytest.raises(_Captured):
        M.run_baroclinic(row, tmp_path, days=2.0)
    steps = sorted(got["matched_steps"])
    assert len(steps) == 18
    for k, st in enumerate(steps):              # exactly on the targets
        assert abs(st * got["dt"] - (k + 1) * 9600.0) < 1e-6
    assert got["every"] == max(1, int(6 * 3600 / got["dt"]))
    m0 = got["matched_fn"](got["state"])
    assert "lee_box_wind" not in got["scalar_fn"](got["state"])
    assert m0["lee_box_wind"] == pytest.approx(19.9, abs=0.3)


def test_native_containment_with_edge_off_the_centre_bisector(M):
    """Cells whose edges are NOT the bisectors of their centres (strip
    115-125E has its centre at 117E): with the corners the score uses the
    true cell overlap (box mean 5.0); nearest-centre assignment would give
    6.5 (bisector 113.5E)."""
    lon_e = np.arange(5.0, 366.0, 10.0)            # meridian edges
    lat_e = np.arange(-10.0, 51.0, 5.0)
    lon_c = 0.5 * (lon_e[:-1] + lon_e[1:])
    j117 = int(np.where(lon_c == 120.0)[0][0])
    lon_c[j117] = 117.0
    lat_c = 0.5 * (lat_e[:-1] + lat_e[1:])
    LA, LO = np.meshgrid(lat_c, lon_c, indexing="ij")
    spd = np.zeros(LA.shape)
    spd[:, j117] = 10.0
    corners = np.empty(LA.shape + (4, 3))
    for i in range(len(lat_c)):
        for j in range(len(lon_c)):
            q = [(lat_e[i], lon_e[j]), (lat_e[i], lon_e[j + 1]),
                 (lat_e[i + 1], lon_e[j + 1]), (lat_e[i + 1], lon_e[j])]
            corners[i, j] = M._sphere_xyz(np.array([p[0] for p in q]),
                                          np.array([p[1] for p in q]))
    native = M._lee_box_wind(spd, LA, LO, corners.reshape(-1, 4, 3))
    assert native == pytest.approx(5.0, abs=0.05)
    assert M._lee_box_wind(spd, LA, LO) == pytest.approx(6.5, abs=0.05)
