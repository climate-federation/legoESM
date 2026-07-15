"""Cross-grid SW regression test for iter-1..14 anchored mass fixers.

Parallel to ``test_mass_conservation_anchored.py`` for the hydrostatic
PE solvers: runs a short Williamson-5 integration on every shallow-
water dycore and asserts ``∫ h dA`` drifts at the fp64 floor (≤ 1e-10).

A regression here means one of the SW fixers (iter-4 lat-lon, iter-5
FV3 cube, iter-6 MPAS, iter-13 fp64 ``_total_area``) has been undone.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import math
import pytest

DRIFT_TOL = 1e-12
N_STEPS = 20


def _rel_drift(m0: float, m1: float) -> float:
    return abs(m1 - m0) / max(abs(m0), 1.0)


def test_sw_mass_conservation_fv3_cube():
    """FV3 cube SW: set_initial_mass + fp64 budget acc (iter-5)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
        CDGridShallowWaterConfig,
    )
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test5,
    )
    from legoesm import constants as _c

    grid = create_cubed_sphere(12)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        boundary_fix=True,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    cdgrid = model.cdgrid
    sw = williamson_test5(grid)
    u0 = 20.0
    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x
    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    model.set_initial_mass(state)

    def _mass(s):
        return float(jnp.sum(
            s.h.astype(jnp.float64) * grid.area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 300.0)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


def test_sw_mass_conservation_latlon():
    """C-grid lat-lon SW: anchor_mass_to_initial + fp64 acc (iter-4)."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterModel,
        CGridLatLonShallowWaterConfig,
        williamson_test5_cgrid,
    )
    from legoesm import constants as _c

    grid = create_latlon_grid(36, 72)
    dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    dt = min(300.0, 0.5 * dx_pole / math.sqrt(_c.g * 3000.0))
    cfg = CGridLatLonShallowWaterConfig(
        A_h=0.0, fix_mass=True, anchor_mass_to_initial=True,
    )
    model = CGridLatLonShallowWaterModel(grid, cfg, dt=dt)
    state = williamson_test5_cgrid(grid)

    def _mass(s):
        return float(jnp.sum(
            s.h.astype(jnp.float64) * grid.area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, dt)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


def test_sw_mass_conservation_mpas():
    """MPAS SW: anchor_mass_to_initial (iter-6)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
        MPASShallowWaterModel, MPASShallowWaterConfig,
    )
    from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
        williamson_test5_mpas,
    )

    mesh = create_voronoi_mesh(4)
    cfg = MPASShallowWaterConfig(
        nu_del4=0.0, fix_mass=True, anchor_mass_to_initial=True,
    )
    model = MPASShallowWaterModel(mesh, cfg)
    state = williamson_test5_mpas(mesh)

    def _mass(s):
        return float(jnp.sum(
            s.h.data.astype(jnp.float64)
            * mesh.areaCell.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 300.0)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


# ---------------------------------------------------------------------------
# iter-34: long-run drift check (parallel to iter-33 hydro PE).
# ---------------------------------------------------------------------------

def test_long_run_sw_mass_conservation_fv3_cube():
    """100-step FV3 cube SW: anchor must NOT random-walk."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        FV3EdgeShallowWaterModel, FV3EdgeShallowWaterState,
        CDGridShallowWaterConfig,
    )
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test5,
    )

    grid = create_cubed_sphere(12)
    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0, boundary_fix=True,
    )
    model = FV3EdgeShallowWaterModel(grid, cfg)
    cdgrid = model.cdgrid
    sw = williamson_test5(grid)
    u0 = 20.0
    u_east_x = u0 * jnp.cos(cdgrid.lat_edge_x)
    u_d = cdgrid.cos_angle_edge_x * u_east_x
    u_east_y = u0 * jnp.cos(cdgrid.lat_edge_y)
    v_d = -cdgrid.sin_angle_edge_y * u_east_y
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data,
    )
    model.set_initial_mass(state)

    def _mass(s):
        return float(jnp.sum(
            s.h.astype(jnp.float64) * grid.area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(100):
        state = model.step(state, 300.0)
    drift = _rel_drift(m0, _mass(state))
    assert drift < 1e-12, (
        f"cube SW 100-step drift {drift:.2e} exceeds 1e-12 — possible "
        f"per-step accumulation bug in the anchored fixer"
    )


def test_long_run_sw_mass_conservation_latlon():
    """iter-64: 100-step lat-lon SW long-run guard."""
    import math
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.atmosphere.dynamics.gcm.shallow_water_latlon_cgrid import (
        CGridLatLonShallowWaterModel,
        CGridLatLonShallowWaterConfig,
        williamson_test5_cgrid,
    )
    from legoesm import constants as _c

    grid = create_latlon_grid(36, 72)
    dx_pole = float(grid.radius) * grid.dlon * math.cos(
        math.pi / 2 - grid.dlat / 2)
    dt = min(300.0, 0.5 * dx_pole / math.sqrt(_c.g * 3000.0))
    cfg = CGridLatLonShallowWaterConfig(
        A_h=0.0, fix_mass=True, anchor_mass_to_initial=True,
    )
    model = CGridLatLonShallowWaterModel(grid, cfg, dt=dt)
    state = williamson_test5_cgrid(grid)

    def _mass(s):
        return float(jnp.sum(
            s.h.astype(jnp.float64) * grid.area.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(100):
        state = model.step(state, dt)
    drift = _rel_drift(m0, _mass(state))
    assert drift < 1e-12, (
        f"lat-lon SW 100-step drift {drift:.2e} exceeds 1e-12"
    )


def test_long_run_sw_mass_conservation_mpas():
    """iter-64: 100-step MPAS SW long-run guard."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.shallow_water_mpas import (
        MPASShallowWaterModel, MPASShallowWaterConfig,
    )
    from tests.atmosphere.shallow_water.test_cases.williamson_mpas import (
        williamson_test5_mpas,
    )

    mesh = create_voronoi_mesh(4)
    cfg = MPASShallowWaterConfig(
        nu_del4=0.0, fix_mass=True, anchor_mass_to_initial=True,
    )
    model = MPASShallowWaterModel(mesh, cfg)
    state = williamson_test5_mpas(mesh)

    def _mass(s):
        return float(jnp.sum(
            s.h.data.astype(jnp.float64)
            * mesh.areaCell.astype(jnp.float64),
        ))

    m0 = _mass(state)
    for _ in range(100):
        state = model.step(state, 300.0)
    drift = _rel_drift(m0, _mass(state))
    assert drift < 1e-12, (
        f"MPAS SW 100-step drift {drift:.2e} exceeds 1e-12"
    )
