"""Conservation harness — exercise every (equation set × grid × dycore)
combination for a short integration and print mass + energy drift.

Default run: ``JAX_ENABLE_X64=1 PYTHONPATH=. .venv/bin/python scripts/check_conservation_all.py``

Use ``--probe NAME`` to exercise just one cell.  Use ``--with-fixers`` to
re-run with the on-by-default conservation fixers active (the harness
otherwise disables them so that intrinsic dycore drift is measured).
"""

from __future__ import annotations

import argparse
import time
import traceback

import jax
import jax.numpy as jnp

from legoesm import constants


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _rel_drift(initial: float, final: float) -> float:
    if abs(initial) < 1e-30:
        return abs(final - initial)
    return abs(final - initial) / abs(initial)


def _row(name, mass_drift, energy_drift, wall, status, notes=""):
    return dict(name=name, mass_drift=mass_drift, energy_drift=energy_drift,
                wall=wall, status=status, notes=notes)


# ---------------------------------------------------------------------------
# Atmosphere shallow water
# ---------------------------------------------------------------------------

def probe_sw_spectral(n_steps=20, dt=600.0, with_fixers=False):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.spectral_sw import (
        SpectralShallowWaterModel, SpectralSWConfig,
        williamson_test2_spectral, spectral_to_grid,
    )
    grid = create_gaussian_grid(n_max=21)
    state = williamson_test2_spectral(grid)
    model = SpectralShallowWaterModel(grid, SpectralSWConfig())
    area = grid.grid_area

    def diags(st):
        gp = spectral_to_grid(st, grid)
        h = gp['h']; u = gp['u']; v = gp['v']
        mass = float(jnp.sum(h * area))
        E = float(jnp.sum(0.5 * (u**2 + v**2) * h * area)
                  + 0.5 * constants.g * jnp.sum(h**2 * area))
        return mass, E

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("sw_spectral_w2", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"T21, n_steps={n_steps}")


def probe_sw_latlon(n_steps=20, dt=300.0, with_fixers=False):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterModel, CGridLatLonShallowWaterConfig,
        williamson_test2_cgrid,
    )
    grid = create_latlon_grid(n_lat=72, radius=constants.R_earth, omega=constants.Omega)
    state = williamson_test2_cgrid(grid)
    config = CGridLatLonShallowWaterConfig(fix_mass=with_fixers)
    model = CGridLatLonShallowWaterModel(grid, config)
    area = grid.area

    def diags(st):
        h = st.h; u = st.u; v = st.v
        mass = float(jnp.sum(h * area))
        u_h = 0.5 * (u[:, :-1] + u[:, 1:])
        v_h = 0.5 * (v[:-1, :] + v[1:, :])
        E = float(jnp.sum(0.5 * (u_h**2 + v_h**2) * h * area)
                  + 0.5 * constants.g * jnp.sum(h**2 * area))
        return mass, E

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("sw_latlon_w2", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"72x144, n_steps={n_steps}, fixers={with_fixers}")


def probe_sw_cubed_sphere(n_steps=20, dt=300.0, with_fixers=False):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.halo import pad_halo_vector
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterModel, CDGridShallowWaterConfig,
        CDGridShallowWaterState,
    )
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test2,
    )

    grid = create_cubed_sphere(24)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sw_state = williamson_test2(grid)
    h_a = sw_state.h.data
    u_a = sw_state.u.data
    v_a = sw_state.v.data
    h_s_a = sw_state.h_s.data
    u_pad, v_pad = pad_halo_vector(
        u_a, v_a,
        cdgrid.base.cos_angle, cdgrid.base.sin_angle,
        cdgrid.base.cos_angle_padded, cdgrid.base.sin_angle_padded,
        interp_offsets=cdgrid.base.halo_interp_offsets,
    )
    u_d = 0.25 * (u_pad[:, :-1, :-1] + u_pad[:, 1:, :-1] +
                  u_pad[:, :-1, 1:] + u_pad[:, 1:, 1:])
    v_d = 0.25 * (v_pad[:, :-1, :-1] + v_pad[:, 1:, :-1] +
                  v_pad[:, :-1, 1:] + v_pad[:, 1:, 1:])
    state = CDGridShallowWaterState(h=h_a, u_d=u_d, v_d=v_d, h_s=h_s_a)
    config = CDGridShallowWaterConfig(use_conservation_fixer=with_fixers)
    model = CDGridShallowWaterModel(grid, config)
    area = grid.area

    def diags(st):
        h = st.h; u_d_, v_d_ = st.u_d, st.v_d
        u_cc, v_cc = dgrid_to_center_vector(u_d_, v_d_)
        mass = float(jnp.sum(h * area))
        E = float(jnp.sum(0.5 * (u_cc**2 + v_cc**2) * h * area)
                  + 0.5 * constants.g * jnp.sum(h**2 * area))
        return mass, E

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("sw_cubed_sphere_w2", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"C24, n_steps={n_steps}, fixers={with_fixers}")


def probe_sw_mpas(n_steps=20, dt=600.0, with_fixers=False):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
        MPASShallowWaterModel, MPASShallowWaterConfig,
    )
    from legoesm.core.operators_voronoi import kinetic_energy_cell
    from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
        williamson_test2_mpas,
    )
    mesh = create_voronoi_mesh(subdivision_level=4)
    state = williamson_test2_mpas(mesh)
    model = MPASShallowWaterModel(mesh, MPASShallowWaterConfig(
        fix_mass=with_fixers, fix_energy=with_fixers))
    area = mesh.areaCell

    def diags(st):
        h = st.h.data; u = st.u.data; h_s = st.h_s.data
        mass = float(jnp.sum(h * area))
        ke_cell = kinetic_energy_cell(u, mesh)
        KE = jnp.sum(ke_cell * h * area)
        PE = 0.5 * constants.g * jnp.sum((h + h_s)**2 * area)
        return mass, float(KE + PE)

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("sw_mpas_w2", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"subdiv-4, fixers={with_fixers}")


# ---------------------------------------------------------------------------
# Atmosphere hydrostatic 3D PE
# ---------------------------------------------------------------------------

def probe_pe_spectral(n_steps=20, dt=600.0, with_fixers=False):
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel, SpectralPEConfig,
        isothermal_rest_state_spectral, spectral_pe_to_grid,
    )
    grid = create_gaussian_grid(n_max=21)
    sigma = create_sigma_coordinate(8)
    state = isothermal_rest_state_spectral(grid, sigma, perturbation_amplitude=0.1)
    model = SpectralPrimitiveEquationModel(grid, sigma, SpectralPEConfig())
    area = grid.grid_area
    dsigma = sigma.dsigma

    def diags(st):
        gp = spectral_pe_to_grid(st, grid, sigma)
        ps = gp['p_s']; u = gp['u']; v = gp['v']; T = gp['T']
        mass = float(jnp.sum(ps * area) / constants.g)
        dp = ps[..., None] * dsigma
        KE = 0.5 * (u**2 + v**2)
        IE = constants.c_vd * T
        E = float(jnp.sum(jnp.sum((KE + IE) * dp / constants.g, axis=-1) * area))
        return mass, E

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("pe_spectral_iso", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"T21/8L")


def probe_pe_cubed_sphere(n_steps=20, dt=300.0, with_fixers=False):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel, CDGridPrimitiveEquationConfig,
        hydrostatic_to_fv3, fv3_to_hydrostatic,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector

    grid = create_cubed_sphere(24)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    sigma = create_sigma_coordinate(8)
    state_cc = held_suarez_init(grid, sigma, T_init=280.0)
    state = hydrostatic_to_fv3(state_cc, cdgrid)
    config = CDGridPrimitiveEquationConfig(
        use_conservation_fixer=with_fixers,
        fix_mass=with_fixers,
        anchor_mass_to_initial=with_fixers,
    )
    model = CDGridPrimitiveEquationModel(grid, sigma, config)
    area = grid.area
    dsigma = sigma.dsigma

    def diags(st):
        cc = fv3_to_hydrostatic(st, cdgrid)
        u_cc, v_cc = dgrid_to_center_vector(st.u_d.data, st.v_d.data)
        ps = cc.p_s.data; T = cc.T.data
        mass = float(jnp.sum(ps * area) / constants.g)
        dp = ps[..., None] * dsigma
        KE = 0.5 * (u_cc**2 + v_cc**2)
        IE = constants.c_vd * T
        E = float(jnp.sum(jnp.sum((KE + IE) * dp / constants.g, axis=-1) * area))
        return mass, E

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("pe_cubed_sphere_hs", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"C24/8L, fixers={with_fixers}")


def probe_pe_latlon(n_steps=20, dt=30.0, with_fixers=False):
    # Lat-lon C-grid PE is CFL-limited at poles where dx → 0;
    # at 36×72 with the default polar filter dt=30 s is the stable bound.
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_latlon

    grid = create_latlon_grid(n_lat=36, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(8)
    state = held_suarez_init_latlon(grid, sigma, T_init=280.0)
    config = CGridLatLonPrimitiveEquationConfig(fix_mass=with_fixers)
    model = CGridLatLonPrimitiveEquationModel(grid, sigma, config)
    area = grid.area
    dsigma = sigma.dsigma

    def diags(st):
        ps = st.p_s.data; u = st.u.data; v = st.v.data; T = st.T.data
        mass = float(jnp.sum(ps * area) / constants.g)
        dp = ps[..., None] * dsigma
        u_h = 0.5 * (u[:, :-1] + u[:, 1:]) if u.shape[1] == ps.shape[1] + 1 else u
        v_h = 0.5 * (v[:-1, :] + v[1:, :]) if v.shape[0] == ps.shape[0] + 1 else v
        KE = 0.5 * (u_h**2 + v_h**2)
        IE = constants.c_vd * T
        E = float(jnp.sum(jnp.sum((KE + IE) * dp / constants.g, axis=-1) * area))
        return mass, E

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("pe_latlon_hs", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"36x72/8L, fixers={with_fixers}")


def probe_pe_mpas(n_steps=20, dt=300.0, with_fixers=False):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel, MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.held_suarez import held_suarez_init_mpas
    mesh = create_voronoi_mesh(subdivision_level=4)
    sigma = create_sigma_coordinate(8)
    state = held_suarez_init_mpas(mesh, sigma, T_init=280.0)
    model = MPASPrimitiveEquationModel(
        mesh, sigma, MPASPrimitiveEquationConfig(fix_mass=with_fixers))
    area = mesh.areaCell
    dsigma = sigma.dsigma

    def diags(st):
        ps = st.p_s.data; T = st.T.data
        mass = float(jnp.sum(ps * area) / constants.g)
        dp = ps[..., None] * dsigma
        IE = constants.c_vd * T
        E = float(jnp.sum(jnp.sum(IE * dp / constants.g, axis=-1) * area))
        return mass, E

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("pe_mpas_hs", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"subdiv-4/8L, fixers={with_fixers}")


# ---------------------------------------------------------------------------
# Atmosphere non-hydrostatic
# ---------------------------------------------------------------------------

def probe_nh_cubed_sphere(n_steps=10, dt=10.0, with_fixers=False):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_height_coordinate, compute_terrain_metric
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState

    N = 12
    grid = create_cubed_sphere(N)
    height = create_height_coordinate(8, 10000.0)
    z_s = jnp.zeros((6, N, N))
    terrain = compute_terrain_metric(z_s, height)
    nlev = height.n_levels
    shape_3d = (6, N, N, nlev)
    shape_w = (6, N, N, nlev + 1)
    z = jnp.zeros(shape_3d)
    state = NonHydrostaticState(
        u=Field(z, "u", ("face", "x", "y", "level"), "m/s"),
        v=Field(z, "v", ("face", "x", "y", "level"), "m/s"),
        w=Field(jnp.zeros(shape_w), "w", ("face", "x", "y", "level_half"), "m/s"),
        theta_prime=Field(z, "theta_prime", ("face", "x", "y", "level"), "K"),
        rho_prime=Field(z, "rho_prime", ("face", "x", "y", "level"), "kg/m^3"),
        tracers=Field(jnp.zeros((6, N, N, nlev, 0)), "tracers",
                      ("face", "x", "y", "level", "tracer"), "kg/kg"),
        phis=Field(jnp.zeros((6, N, N)), "phis", ("face", "x", "y"), "m^2/s^2"),
    )
    config = CDGridCompressibleEulerConfig(A_h=0.0, fix_mass=with_fixers)
    model = CDGridCompressibleEulerModel(grid, height, terrain, config)
    area = grid.area
    dz = height.dz

    def diags(st):
        rho = height.rho_ref + st.rho_prime.data
        theta = height.theta_ref + st.theta_prime.data
        u = st.u.data; v = st.v.data
        col_mass = jnp.sum(rho * dz, axis=-1)
        mass = float(jnp.sum(col_mass * area))
        KE = 0.5 * (u**2 + v**2)
        IE = constants.c_vd * theta
        E = float(jnp.sum(jnp.sum((KE + IE) * rho * dz, axis=-1) * area))
        return mass, E

    m0, e0 = diags(state)
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    m1, e1 = diags(state)
    return _row("nh_cubed_sphere_rest", _rel_drift(m0, m1), _rel_drift(e0, e1),
                wall, "OK", f"C12/8L, fixers={with_fixers}")


# ---------------------------------------------------------------------------
# Ocean 3D
# ---------------------------------------------------------------------------

def _ocean_diags_cell(state, mesh, z, area, mask_attr):
    from legoesm.ocean.vertical import compute_layer_thickness
    eta = state.eta.data; T = state.T.data; S = state.S.data
    mask = getattr(state, mask_attr).data
    H_bathy = state.H_bathy.data
    h_k = compute_layer_thickness(eta, H_bathy, z, min_water_column_m=1.0)
    if mask.ndim == 1:
        m_col = mask[:, None]; a_col = area[:, None]
    else:
        m_col = mask[..., None]; a_col = area[..., None]
    # Total volume (well-defined nonzero baseline) instead of eta-only,
    # so a rest-state divisor doesn't hit the zero-baseline branch in
    # ``_rel_drift`` and report misleading absolute m^3 numbers.
    vol = float(jnp.sum(h_k * m_col * a_col))
    heat = float(jnp.sum(T * h_k * m_col * a_col))
    salt = float(jnp.sum(S * h_k * m_col * a_col))
    return vol, heat + salt


def probe_ocean_cubed_sphere(n_steps=20, dt=600.0, with_fixers=False):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.state import OceanConfig

    grid = create_cubed_sphere(12)
    z = create_ocean_z_star(8, H_max=4000.0)
    state = rest_state_ocean(grid, z)
    config = OceanConfig(use_conservation_fixer=with_fixers)
    model = OceanModel(grid, z, config)
    area = grid.area

    v0, hs0 = _ocean_diags_cell(state, grid, z, area, "land_mask")
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    v1, hs1 = _ocean_diags_cell(state, grid, z, area, "land_mask")
    return _row("ocean_cubed_sphere_rest", _rel_drift(v0, v1),
                _rel_drift(hs0, hs1), wall, "OK",
                f"C12/8L, fixers={with_fixers}; mass=vol, energy=heat+salt")


def probe_ocean_latlon(n_steps=20, dt=600.0, with_fixers=False):
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.state import LatLonCGridOceanConfig

    grid = create_latlon_grid(n_lat=36, radius=constants.R_earth, omega=constants.Omega)
    z = create_ocean_z_star(8, H_max=4000.0)
    state = rest_state_latlon_cgrid_ocean(grid, z)
    config = LatLonCGridOceanConfig.from_flat(use_conservation_fixer=with_fixers)
    model = LatLonCGridOceanModel(grid, z, config)
    area = grid.area

    v0, hs0 = _ocean_diags_cell(state, grid, z, area, "land_mask")
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    v1, hs1 = _ocean_diags_cell(state, grid, z, area, "land_mask")
    return _row("ocean_latlon_rest", _rel_drift(v0, v1),
                _rel_drift(hs0, hs1), wall, "OK",
                f"36x72/8L, fixers={with_fixers}")


def probe_ocean_mpas(n_steps=20, dt=600.0, with_fixers=False):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.init_mpas import rest_state_mpas_ocean
    from legoesm.ocean.dynamics.ocean_model_mpas import MPASOceanModel
    from legoesm.ocean.mpas_config import MPASOceanConfig

    mesh = create_voronoi_mesh(subdivision_level=3)
    z = create_ocean_z_star(8, H_max=4000.0)
    state = rest_state_mpas_ocean(mesh, z)
    config = MPASOceanConfig(use_conservation_fixer=with_fixers)
    model = MPASOceanModel(mesh, z, config)
    area = mesh.areaCell

    v0, hs0 = _ocean_diags_cell(state, mesh, z, area, "land_mask")
    t0 = time.time()
    for _ in range(n_steps):
        state = model.step(state, dt)
    jax.block_until_ready(jax.tree.leaves(state))
    wall = time.time() - t0
    v1, hs1 = _ocean_diags_cell(state, mesh, z, area, "land_mask")
    return _row("ocean_mpas_rest", _rel_drift(v0, v1),
                _rel_drift(hs0, hs1), wall, "OK",
                f"subdiv-3/8L, fixers={with_fixers}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

PROBES = [
    ("sw_spectral", probe_sw_spectral),
    ("sw_latlon", probe_sw_latlon),
    ("sw_cubed_sphere", probe_sw_cubed_sphere),
    ("sw_mpas", probe_sw_mpas),
    ("pe_spectral", probe_pe_spectral),
    ("pe_cubed_sphere", probe_pe_cubed_sphere),
    ("pe_latlon", probe_pe_latlon),
    ("pe_mpas", probe_pe_mpas),
    ("nh_cubed_sphere", probe_nh_cubed_sphere),
    ("ocean_cubed_sphere", probe_ocean_cubed_sphere),
    ("ocean_latlon", probe_ocean_latlon),
    ("ocean_mpas", probe_ocean_mpas),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default=None)
    parser.add_argument("--with-fixers", action="store_true")
    args = parser.parse_args()
    print(f"{'name':28s}  {'mass_drift':>13s}  {'energy_drift':>13s}  "
          f"{'wall_s':>7s}  status   notes")
    print("-" * 110)
    for name, fn in PROBES:
        if args.probe is not None and args.probe != name:
            continue
        try:
            row = fn(with_fixers=args.with_fixers)
            print(f"{row['name']:28s}  {row['mass_drift']:13.3e}  "
                  f"{row['energy_drift']:13.3e}  {row['wall']:7.2f}  "
                  f"{row['status']:6s}   {row['notes']}")
        except Exception as exc:
            tb = traceback.format_exc().strip().splitlines()[-1]
            print(f"{name:28s}  {'-':>13s}  {'-':>13s}  {'-':>7s}  ERROR    {tb[:140]}")


if __name__ == "__main__":
    main()
