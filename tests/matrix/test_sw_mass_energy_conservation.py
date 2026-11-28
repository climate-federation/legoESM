"""Colliding-modon SW lanes (lat-lon C-grid 72x144, MPAS ico5): mass is
conserved to roundoff WITHOUT the mass fixer, and the lat-lon energy
diagnostic uses the C-grid's consistent (TRiSK / Arakawa-Lamb) kinetic energy.

The flux-form continuity equations of both cores telescope exactly; the
1.6e-8 (lat-lon, first step) and ~3e-9/day (ico) "leaks" seen with the fixer
off were fp32 arithmetic: the matrix ICs inherited the fp32 storage dtype of
the grids (lat-lon h/u fp32 with fp64 v, so part of step 1 ran in fp32
arithmetic; ico entirely fp32).  ``_sw_state_run_precision`` promotes them under the
x64 matrix.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
_KE0_ANALYTIC = 6.27996e18      # m^5/s^2, Lin et al. (2017) modon pair


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
    spec = importlib.util.spec_from_file_location("_atmmatrix_sw_cons",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def _float_dtypes(state):
    return {x.dtype for x in jax.tree.leaves(state)
            if jnp.issubdtype(x.dtype, jnp.floating)}


def _rel_mass_steps(model, state, mass, dt, n):
    out, m0 = [], mass(state)
    for _ in range(n):
        state = model.step(state, dt)
        m1 = mass(state)
        out.append(m1 / m0 - 1.0)
        m0 = m1
    return np.array(out)


@pytest.fixture(scope="module")
def latlon(M):
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from tests.test_cases.colliding_modons import _MODON_H0
    grid = create_latlon_grid(72, 144, omega=0.0)
    dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    dt = min(300.0, 0.5 * dx_pole / math.sqrt(constants.g * _MODON_H0))
    return grid, dt, M._latlon_sw_initial_state(grid, 8)


def test_sw_initial_states_are_fp64_under_x64(M, latlon):
    assert _float_dtypes(latlon[2]) == {jnp.dtype(jnp.float64)}
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(5, omega=0.0)
    assert _float_dtypes(M._ico_sw_initial_state(mesh, 8)) == {
        jnp.dtype(jnp.float64)}


def test_latlon_modon_mass_conserved_without_fixer(M, latlon):
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterConfig, CGridLatLonShallowWaterModel)
    grid, dt, state = latlon
    cfg = CGridLatLonShallowWaterConfig(
        nu_del4=M._biharmonic_visc_latlon(72, efold_hours=36.0),
        fix_mass=False)
    model = CGridLatLonShallowWaterModel(grid, cfg, dt=dt)
    area = np.asarray(grid.area, np.float64)
    d = _rel_mass_steps(
        model, state, lambda s: float(np.sum(np.asarray(s.h) * area)), dt, 3)
    # fp32 IC: step 1 jumps +1.6e-8; fp64: roundoff (<= a few 1e-17 / step)
    assert np.max(np.abs(d)) < 1e-14, d


def test_ico_modon_mass_conserved_without_fixer(M):
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
        MPASShallowWaterConfig, MPASShallowWaterModel)
    from legoesm.grids.voronoi import create_voronoi_mesh
    mesh = create_voronoi_mesh(5, omega=0.0)
    state = M._ico_sw_initial_state(mesh, 8)
    model = MPASShallowWaterModel(mesh, MPASShallowWaterConfig(
        nu_del4=M._hyperdiff_ico(mesh), fix_mass=False))
    area = np.asarray(mesh.areaCell, np.float64)
    d = _rel_mass_steps(
        model, state, lambda s: float(np.sum(np.asarray(s.h.data) * area)),
        300.0, 288)
    # fp32 IC/state: +-1e-10 per step, ~3e-9 net over the day
    assert np.max(np.abs(d)) < 1e-14, d
    assert abs(np.prod(1.0 + d) - 1.0) < 1e-14


def test_latlon_ic_kinetic_energy_is_the_cgrid_energy(M, latlon):
    """The matrix lat-lon KE is the face-weighted sum_f A_f hbar_f u_f^2 / 2
    and recovers the analytic modon IC KE; the |face-mean wind|^2 / 2 form
    misses 7 % of it."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import (
        latlon_face_metrics)
    from legoesm.grids.operator_adapters import latlon_sw_grid_operators
    grid, _, st = latlon
    d = M._modon_energy_enstrophy_latlon(st, grid)
    assert d["energy_kinetic_KE"] / _KE0_ANALYTIC == pytest.approx(1.0,
                                                                   abs=2e-3)
    # independent face sum (pole v faces carry zero weight)
    l_u, d_u, l_v, d_v = (np.asarray(a, np.float64) for a in
                          latlon_face_metrics(grid, latlon_sw_grid_operators(grid)))
    h = np.asarray(st.h, np.float64)
    u = np.asarray(st.u, np.float64)
    v = np.asarray(st.v, np.float64)
    h_u = 0.5 * (h + np.roll(h, 1, axis=1))            # face j: cells j-1, j
    ke_u = np.sum(0.5 * (l_u * d_u)[:, None] * h_u * u[:, :-1] ** 2)
    h_v = 0.5 * (h[:-1] + h[1:])
    ke_v = np.sum(0.5 * (l_v * d_v)[1:-1, None] * h_v * v[1:-1] ** 2)
    assert d["energy_kinetic_KE"] == pytest.approx(ke_u + ke_v, rel=1e-12)


def test_latlon_energy_diagnostic_meridional_term(M):
    """Nonzero interior v, unequal depths, zero polar v: the diagnostic KE
    matches a face sum on GEOMETRIC face areas (u face: its cell row; v face:
    half of each adjacent row), independent of ``latlon_face_metrics``
    (which agree with them to <1e-3 at 72x144)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterState)
    grid = create_latlon_grid(72, 144, omega=0.0)
    rng = np.random.default_rng(3)
    h = 1e4 + rng.normal(0.0, 50.0, (72, 144))
    u = rng.normal(0.0, 10.0, (72, 145))
    u[:, -1] = u[:, 0]                                   # periodic closure
    v = rng.normal(0.0, 10.0, (73, 144))
    v[0] = v[-1] = 0.0                                   # polar walls
    st = CGridLatLonShallowWaterState(h=jnp.asarray(h), u=jnp.asarray(u),
                                      v=jnp.asarray(v),
                                      h_s=jnp.zeros_like(jnp.asarray(h)))
    d = M._modon_energy_enstrophy_latlon(st, grid)
    A = np.asarray(grid.area, np.float64)
    h_u = 0.5 * (h + np.roll(h, 1, axis=1))
    ke_u = np.sum(0.5 * A * h_u * u[:, :-1] ** 2)
    A_v = 0.5 * (A[:-1] + A[1:])
    ke_v = np.sum(0.5 * A_v * 0.5 * (h[:-1] + h[1:]) * v[1:-1] ** 2)
    assert ke_v > 0.3 * ke_u                             # v carries real weight
    assert d["energy_kinetic_KE"] == pytest.approx(ke_u + ke_v, rel=1e-3)


def test_latlon_trisk_ke_matches_pe_closure_shapes(x64):
    """The hoisted helper handles 2-D (SW) and 3-D (PE, trailing nlev)."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import (
        latlon_trisk_kinetic_energy)
    rng = np.random.default_rng(0)
    nj, ni, nk = 4, 6, 3
    A_u, A_v = rng.uniform(1, 2, nj), rng.uniform(1, 2, nj + 1)
    area = rng.uniform(3, 4, (nj, ni))
    u3 = rng.normal(size=(nj, ni + 1, nk))
    v3 = rng.normal(size=(nj + 1, ni, nk))
    k3 = latlon_trisk_kinetic_energy(jnp.asarray(u3), jnp.asarray(v3),
                                     jnp.asarray(A_u), jnp.asarray(A_v),
                                     jnp.asarray(area))
    for k in range(nk):
        k2 = latlon_trisk_kinetic_energy(
            jnp.asarray(u3[..., k]), jnp.asarray(v3[..., k]),
            jnp.asarray(A_u), jnp.asarray(A_v), jnp.asarray(area))
        ref = (A_u[:, None] * (u3[:, :-1, k] ** 2 + u3[:, 1:, k] ** 2)
               + A_v[:-1, None] * v3[:-1, :, k] ** 2
               + A_v[1:, None] * v3[1:, :, k] ** 2) / (4 * area)
        np.testing.assert_allclose(np.asarray(k2), ref, rtol=1e-14)
        np.testing.assert_array_equal(np.asarray(k3[..., k]), np.asarray(k2))
