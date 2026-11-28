"""Hydrostatic 3-D dycore rows (lat-lon C-grid 72x144, MPAS ico5): the air
mass is conserved to roundoff WITHOUT a mass fixer, and every row reports
the scheme's total energy.

The DCMIP 2012 2-0-0 rest-over-terrain ICs (and the DCMIP 2008 dry ones)
inherited the fp32 storage dtype of the grids (``T``/``p_s``/``phis``).  An
fp32 ``p_s`` stays fp32 through the ico step: +9.8e-6 of the air mass per
day with the fixer off (the fixer hid it); lat-lon took a one-off +1.3e-8
on step 1.  ``_state_run_precision`` promotes the ICs under the x64 matrix;
the rows then run with no mass fixer (user directive 2026-11-28).
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")


@pytest.fixture(scope="module")
def x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


@pytest.fixture(scope="module")
def M(x64):
    spec = importlib.util.spec_from_file_location("_atmmatrix_pe_cons",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _rate(notes: str, key: str) -> float:
    m = re.search(rf"{key}=([-+0-9.e]+)", notes)
    assert m, f"{key} missing from notes: {notes}"
    return float(m.group(1))


def _capture_configs(monkeypatch, model_cls):
    seen = []
    orig = model_cls.__init__

    def init(self, *a, **kw):
        orig(self, *a, **kw)
        seen.append(self.config)
    monkeypatch.setattr(model_cls, "__init__", init)
    return seen


@pytest.mark.parametrize("grid_type,res", [("icosahedral", "ico5"),
                                           ("latlon", "72x144")])
def test_rest_state_topo_conserves_mass_without_fixer(M, tmp_path,
                                                      monkeypatch, grid_type,
                                                      res):
    """2-0-0 over a 2 km mountain, 0.25 d (one 6-hourly diagnostic sample
    after t = 0), fixer off: |dM/M| per day at roundoff.  fp32 ICs give
    ~1e-5/day (ico) and ~5e-8/day (lat-lon, the step-1 jump over 0.25 d)."""
    if grid_type == "icosahedral":
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
            MPASPrimitiveEquationModel as cls)
    else:
        from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
            CGridLatLonPrimitiveEquationModel as cls)
    configs = _capture_configs(monkeypatch, cls)
    tc = M.TestCase("hydrostatic", "rest_state_topo", grid_type, res,
                    "hybrid", 7, 1, {"h_0": 2000.0})
    status, _, notes = M.run_baroclinic(tc, tmp_path, 0.25)
    assert status == "PASS", notes
    assert configs and all(not c.fix_mass for c in configs)
    if grid_type == "latlon":
        assert all(not c.zero_mean_ps_tendency for c in configs)
    assert abs(_rate(notes, "dM/M/day")) < 1e-12, notes
    # the energy column is reported (no assertion on its sign here: the
    # lat-lon A_h T diffusion on pressure surfaces is a known heat source)
    assert np.isfinite(_rate(notes, "dE/E/day"))
    assert "mass fixer off" in notes


def test_pe_ics_promoted_to_fp64(M):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.test_cases.dcmip2012.rest_state_topography import (
        lapse_rate_state_topography_init_mpas)
    mesh = create_voronoi_mesh(4)
    sigma = M._create_vertical(10, "hybrid")
    raw = lapse_rate_state_topography_init_mpas(mesh, sigma, h_0=2000.0)
    assert jnp.dtype(jnp.float32) in {
        x.dtype for x in jax.tree.leaves(raw)}, "fixture no longer fp32"
    up = M._state_run_precision(raw)
    assert {x.dtype for x in jax.tree.leaves(up)
            if jnp.issubdtype(x.dtype, jnp.floating)} == {
                jnp.dtype(jnp.float64)}


def test_latlon_pe_energy_matches_hand_integral(M):
    """Both wind components, terrain and level-varying fields: the lat-lon
    energy equals a cell-by-cell loop over the four faces of every cell,
    K_c = (A_u[j](u_W^2 + u_E^2) + A_v[j] v_S^2 + A_v[j+1] v_N^2) / (4 A_c),
    E = sum_c A_c [sum_k (c_pd T + K) dp_k / g + phis p_s / g]."""
    from legoesm import constants as C
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import (
        latlon_face_metrics)
    from legoesm.grids.operator_adapters import latlon_sw_grid_operators
    grid = create_latlon_grid(12, 24)
    sigma = M._create_vertical(8, "hybrid")
    nj, ni = grid.n_lat, grid.n_lon
    nk = int(np.asarray(sigma.pressure_at_half(jnp.ones((1,)))).shape[-1]) - 1
    rng = np.random.default_rng(0)
    u = rng.normal(0.0, 10.0, (nj, ni + 1, nk))
    u[:, ni] = u[:, 0]                       # periodic duplicate face
    v = rng.normal(0.0, 10.0, (nj + 1, ni, nk))
    v[0] = v[-1] = 0.0                       # pole walls
    T = 220.0 + 60.0 * rng.random((nj, ni, nk))
    p_s = 9.0e4 + 1.0e4 * rng.random((nj, ni))
    phis = 2.0e4 * rng.random((nj, ni))
    s = CGridLatLonHydrostaticState(
        u=jnp.asarray(u), v=jnp.asarray(v), T=jnp.asarray(T),
        p_s=jnp.asarray(p_s), phis=jnp.asarray(phis), tracers={})
    e = M._latlon_pe_energy_fn(grid, sigma)(s)

    l_u, d_u, l_v, d_v = latlon_face_metrics(grid, latlon_sw_grid_operators(grid))
    A_u, A_v = np.asarray(l_u * d_u, np.float64), np.asarray(l_v * d_v, np.float64)
    area = np.asarray(grid.area, np.float64)
    ph = np.asarray(sigma.pressure_at_half(jnp.asarray(p_s)), np.float64)
    expect = 0.0
    for j in range(nj):
        for i in range(ni):
            K = (A_u[j] * (u[j, i] ** 2 + u[j, i + 1] ** 2)
                 + A_v[j] * v[j, i] ** 2 + A_v[j + 1] * v[j + 1, i] ** 2
                 ) / (4.0 * area[j, i])
            dp = ph[j, i, 1:] - ph[j, i, :-1]
            expect += area[j, i] * (np.sum((C.c_pd * T[j, i] + K) * dp)
                                    + phis[j, i] * p_s[j, i]) / C.g
    assert e == pytest.approx(expect, rel=1e-13)


def test_pe_conservation_note_rates(M):
    """Rates over the SAMPLED interval (first sample after t = 0), the mass
    excursion, and the flux over the given area."""
    diag = {"times": [0.5, 1.0, 2.5], "mass": [1.0, 1.5, 0.7],
            "total_energy": [1.0e9, 1.0e9, 1.0e9 + 4.0 * 86400.0]}
    note = M._pe_conservation_note(diag, total_area=2.0)
    assert "dM/M/day=-1.50e-01" in note          # -0.3 over 2 days
    assert "max |dM/M| 5.0e-01" in note
    assert "(+1 W/m^2)" in note                  # 2 W over 2 m^2
    assert M._pe_conservation_note({"times": [0.0], "mass": [1.0]}, 1.0) == ""


def test_mass_fixer_kept_without_x64(M):
    """fp32 matrix: ICs stay fp32 and do not conserve mass -> fixer kept."""
    assert M._pe_mass_fixer_kw("latlon") == dict(
        fix_mass=False, anchor_mass_to_initial=False,
        zero_mean_ps_tendency=False)
    jax.config.update("jax_enable_x64", False)
    try:
        assert M._pe_mass_fixer_kw("latlon") == dict(
            fix_mass=True, anchor_mass_to_initial=True)
        assert M._pe_mass_fixer_kw("mpas") == dict(
            fix_mass=True, anchor_mass_to_initial=True)
    finally:
        jax.config.update("jax_enable_x64", True)
    with pytest.raises(ValueError):
        M._pe_mass_fixer_kw("spectral")


def test_pe_conservation_note_fp32_energy_not_resolved(M):
    diag = {"times": [0.0, 1.0], "mass": [1.0, 1.0],
            "total_energy": [1.0e9, 2.0e9]}
    jax.config.update("jax_enable_x64", False)
    try:
        note = M._pe_conservation_note(diag, 1.0)
    finally:
        jax.config.update("jax_enable_x64", True)
    assert "energy rate not resolved (fp32 matrix)" in note
    assert "dE/E/day" not in note


@pytest.fixture(scope="module")
def jw_latlon_energy(M):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        hydrostatic_to_cgrid)
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon
    grid = create_latlon_grid(72, 144)
    sigma = M._create_vertical(M.DEFAULT_NLEV, "hybrid")
    s = M._state_run_precision(hydrostatic_to_cgrid(
        baroclinic_wave_init_latlon(grid, sigma, perturbed=True), grid))
    return M._latlon_pe_energy_fn(grid, sigma)(s)


def test_spectral_energy_step_zero(M, jw_latlon_energy):
    """Spectral JW IC energy (runner fields + Gaussian quadrature) matches the
    lat-lon one of the same analytic IC (measured 7e-6 apart): catches area /
    g / layer-mass scaling errors in the spectral wiring."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import spectral_pe_to_grid
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_spectral
    grid = create_gaussian_grid(21)
    sigma = M._create_vertical(M.DEFAULT_NLEV, "hybrid")
    f = spectral_pe_to_grid(baroclinic_wave_init_spectral(
        grid, sigma, perturbed=True), grid, sigma)
    e = M._pe_total_energy(f["T"], 0.5 * (f["u"] ** 2 + f["v"] ** 2),
                           f["p_s"], f["phis"], sigma, grid.grid_area)
    assert e == pytest.approx(jw_latlon_energy, rel=1e-4)


def test_duo_energy_step_zero(M, jw_latlon_energy):
    """FV3 duo te0_2d on the JW IC vs the lat-lon enthalpy-form energy: equal
    up to FV3's lid term -p_top Phi_top / g (measured 0.36 % on CAM6 L32);
    catches compute-window area weighting and the division by g (each an
    O(1) or larger factor)."""
    from scripts.matrix._cube_duo_adapters import build_duo_hydro
    duo = build_duo_hydro(36, perturbed=True,
                          **M._duo_vertical_kw("hybrid", M.DEFAULT_NLEV))
    e = M._duo_pe_energy_fn(duo)(duo.state)
    assert 0.990 < e / jw_latlon_energy < 0.999
