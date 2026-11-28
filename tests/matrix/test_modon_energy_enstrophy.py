"""Colliding-modons matrix rows: native total energy
E = sum area * (h |V|^2 / 2 + g h^2 / 2) and potential enstrophy
Z = sum area * (zeta + f)^2 / (2 h) on every grid, sampled at one common
cadence (1200 s) fine enough for the ~0.74 d antipodal refocus spike."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import jax
import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")

# Analytic IC integrals of the Lin et al. (2017) modon pair (U0 = 40 m/s,
# r0 = 500 km, h0 = 10 km, a = 6.371e6 m): fine-quadrature reference.
_Z0_ANALYTIC = 0.25223          # m/s^2
_KE0_ANALYTIC = 6.27996e18      # m^5/s^2


@pytest.fixture(scope="module")
def M():
    jax.config.update("jax_enable_x64", True)
    spec = importlib.util.spec_from_file_location("_atmmatrix_modon_ez",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def test_matched_steps_exact_and_nearest(M):
    steps, off = M._matched_diag_steps(300.0, 288, 1200.0, 86400.0)
    assert sorted(steps) == list(range(4, 289, 4)) and off == 0.0
    steps, off = M._matched_diag_steps(1200.0, 72, 1200.0, 86400.0)
    assert sorted(steps) == list(range(1, 73)) and off == 0.0
    dt = 9.68309152389369                       # lat-lon 72x144 pole CFL
    n = int(86400 / dt)
    steps, off = M._matched_diag_steps(dt, n, 1200.0, 86400.0)
    # interior targets within dt/2; only the final (1 d) target, past the
    # truncated last step, is clamped to it with offset < dt
    assert len(steps) == 72 and max(steps) == n
    assert 0.5 * dt < off < dt
    inner, off_in = M._matched_diag_steps(dt, n, 1200.0, 71 * 1200.0)
    assert len(inner) == 71 and off_in <= 0.5 * dt
    # never a target past the requested end
    assert len(M._matched_diag_steps(dt, n, 1200.0, 86000.0)[0]) == 71


def _loop(M, **kw):
    seen = []

    def scalar_fn(s, _matched=False):
        seen.append((s, _matched))
        return {"x": float(s), **({"y": -float(s)} if _matched else {})}
    out = M._run_timeloop(
        lambda s, dt: s + 1, 0, 1.0, 20, lambda s: (True, 0.0), scalar_fn,
        lambda s: {}, 7, lambda s: np.zeros(1), **kw)
    return out[2]


def test_matched_series_is_separate_and_gate_series_unchanged(M):
    base = _loop(M)
    diag = _loop(M, matched_steps=frozenset({3, 11, 20}),
                 matched_fn=lambda s: {"x": float(s), "y": -float(s)})
    assert base["steps"] == [0, 7, 14]          # legacy diag_every cadence
    for k in ("steps", "times", "x"):           # gate inputs identical
        assert diag[k] == base[k]
    m = diag["_matched"]
    assert m["steps"] == [0, 3, 11, 20] and m["y"] == [0.0, -3.0, -11.0, -20.0]
    assert "y" not in diag


def _check(d, z_tol=0.12):
    """IC enstrophy vs the analytic integral (the ~250-280 km grids resolve
    r0 = 500 km cores to within ~10 %)."""
    assert np.isfinite(d["total_energy"]) and d["total_energy"] > 0
    assert abs(d["potential_enstrophy"] / _Z0_ANALYTIC - 1.0) < z_tol, d


def test_latlon_ic(M):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm import constants
    from tests.test_cases.colliding_modons import (
        colliding_modons_latlon, _modon_winds_geo)
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterState)
    import jax.numpy as jnp
    grid = create_latlon_grid(72, 144, omega=0.0)
    cm = colliding_modons_latlon(grid)
    lon_f = grid.lon - 0.5 * grid.dlon
    lon_f = jnp.concatenate([lon_f, lon_f[:1] + 2 * jnp.pi])
    u, _ = _modon_winds_geo(lon_f[None, :], grid.lat[:, None], grid.radius)
    lat_f = jnp.linspace(-0.5 * jnp.pi, 0.5 * jnp.pi, grid.n_lat + 1)
    _, v = _modon_winds_geo(grid.lon[None, :], lat_f[:, None], grid.radius)
    st = CGridLatLonShallowWaterState(h=cm.h.data, u=u, v=v,
                                      h_s=jnp.zeros_like(cm.h.data))
    d = M._modon_energy_enstrophy_latlon(st, grid)
    _check(d)
    pe = 0.5 * constants.g * 1e4 ** 2 * 4 * np.pi * float(grid.radius) ** 2
    assert abs((d["total_energy"] - pe) / _KE0_ANALYTIC - 1.0) < 0.12


def test_mpas_ic(M):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm import constants
    from tests.test_cases.colliding_modons import colliding_modons_mpas
    mesh = create_voronoi_mesh(5, omega=0.0)
    d = M._modon_energy_enstrophy_mpas(colliding_modons_mpas(mesh), mesh)
    _check(d)
    pe = 0.5 * constants.g * 1e4 ** 2 * float(np.sum(mesh.areaCell))
    assert abs((d["total_energy"] - pe) / _KE0_ANALYTIC - 1.0) < 0.12


def test_spectral_ic(M):
    from legoesm.grids.gaussian import create_gaussian_grid
    from tests.test_cases.colliding_modons import colliding_modons_spectral
    grid = create_gaussian_grid(63, omega=0.0)
    d = M._modon_energy_enstrophy_spectral(colliding_modons_spectral(grid),
                                           grid)
    _check(d, z_tol=0.05)


def test_cube_ic(M):
    """Cube duo instrument: D-grid circulation vorticity (sw_core wk) gives
    the analytic IC enstrophy; KE from the c2l A-grid winds."""
    from scripts.matrix._cube_duo_adapters import build_duo_sw
    duo = build_duo_sw(8, 36)
    d = M._modon_energy_enstrophy_cube(duo.state, duo)
    _check(d)
    pe = 0.5 * duo.grav * 1e4 ** 2 * float(np.sum(duo.area))
    ke_ref = _KE0_ANALYTIC * (duo.radius / 6.371e6) ** 2
    assert abs((d["total_energy"] - pe) / ke_ref - 1.0) < 0.12


def test_every_modon_lane_reports_energy_enstrophy():
    src = _SCRIPT.read_text()
    for g in ("cube(s, duo)", "latlon(s, grid)", "mpas(s, mesh)",
              "spectral(s, grid)"):
        assert f"out.update(_modon_energy_enstrophy_{g})" in src, g
    assert ("_matched_diag_steps(\n        dt, n_steps, _MODON_DIAG_CADENCE_S, "
            "days * 86400.0)" in src)
    assert "matched_fn=lambda s: scalar_fn(s, _matched=True))" in src
