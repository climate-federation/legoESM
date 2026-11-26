"""A uniform tracer stays uniform on the cubed-sphere ``OceanModel``.

The horizontal tracer operator is the FCT flux form ``-div_h(u q)`` on the
C-grid velocity, while the vertical one is advective ``-w dq/dz``.  Without the
``+ q div_h(u)`` term a uniform salinity picks up ``-q div_h(u)`` wherever the
flow is divergent: 35 psu drifted ~3e-2 psu in 10 steps of 300 s on this setup.
On z*, open coastal C-faces also carried land tracer into wet cells.
``OceanConfig.tracer_advection_constant_preserving`` (default True) adds the
term and closes coastal faces for the tracer fluxes; False keeps the old
operator.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

S0 = 35.0
N_STEPS = 10
DT = 300.0


def _igw(land=False):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.ocean.vertical import create_ocean_z_star
    from legoesm.ocean.experiments.inertia_gravity_wave import (
        InertiaGravityWaveConfig, create_initial_conditions)

    grid = create_cubed_sphere(8)
    z = create_ocean_z_star(n_levels=2, H_max=1000.0)
    s = create_initial_conditions("cubed_sphere", grid, z, InertiaGravityWaveConfig())
    if land:
        lat, lon = jnp.asarray(grid.lat), jnp.asarray(grid.lon)
        block = (jnp.abs(lat - 0.3) < 0.35) & (jnp.abs(lon - 1.2) < 0.5)
        mask = jnp.where(block, 0.0, 1.0).astype(s.land_mask.data.dtype)
        m3 = mask[..., None]
        s = s._replace(land_mask=s.land_mask.replace(data=mask),
                       u=s.u.replace(data=s.u.data * m3),
                       v=s.v.replace(data=s.v.data * m3),
                       eta=s.eta.replace(data=s.eta.data * mask))
    return grid, z, s


def _model(grid, z, constant_preserving, staggering="a_grid", rk3=False, K_h=0.0):
    from legoesm.ocean.state import OceanConfig
    from legoesm.ocean.dynamics.ocean_model import OceanModel

    cfg = OceanConfig(A_h=1e2, K_h=K_h, A_v=1e-4, K_v=1e-5,
                      n_barotropic_substeps=10, barotropic_staggering=staggering,
                      baroclinic_rk3=rk3,
                      tracer_advection_constant_preserving=constant_preserving)
    return OceanModel(grid, z, cfg)


def _uniform_drift(constant_preserving, staggering="a_grid", rk3=False, land=False,
                   K_h=0.0):
    """Max |S - 35| over wet cells after 10 steps; land cells hold S = 0."""
    grid, z, s = _igw(land)
    wet = s.land_mask.data[..., None]
    s = s._replace(S=s.S.replace(data=jnp.full_like(s.S.data, S0) * wet))
    m = _model(grid, z, constant_preserving, staggering, rk3, K_h)
    for _ in range(N_STEPS):
        s = m.step(s, DT)
    return float(jnp.max(jnp.abs(jnp.where(wet > 0.5, s.S.data - S0, 0.0))))


def test_default_on():
    from legoesm.ocean.state import OceanConfig
    assert OceanConfig().tracer_advection_constant_preserving is True


@pytest.mark.parametrize("staggering", ["a_grid", "c_grid"])
def test_uniform_salinity_stays_uniform(staggering):
    assert _uniform_drift(True, staggering) < 1e-12


def test_uniform_salinity_stays_uniform_rk3():
    assert _uniform_drift(True, rk3=True) < 1e-12


def test_uniform_salinity_stays_uniform_next_to_land():
    assert _uniform_drift(True, land=True) < 1e-12


def test_uniform_salinity_stays_uniform_next_to_land_with_diffusion():
    assert _uniform_drift(True, land=True, K_h=1e3) < 1e-12


def test_old_operator_still_drifts():
    assert _uniform_drift(False) > 1e-2
    assert _uniform_drift(False, land=True) > 1e-2


def test_dye_front_overshoot_bounded():
    """35->36 psu dye front in the divergent IGW flow: 1.6e-4 psu measured."""
    grid, z, s = _igw()
    lat, lon = jnp.asarray(grid.lat)[..., None], jnp.asarray(grid.lon)[..., None]
    dye = S0 + ((jnp.abs(lat) < 0.4) & (jnp.abs(lon - 1.0) < 0.6)).astype(s.S.data.dtype)
    s = s._replace(S=s.S.replace(data=jnp.broadcast_to(dye, s.S.data.shape)))
    m = _model(grid, z, True)
    worst = 0.0
    for _ in range(N_STEPS):
        s = m.step(s, DT)
        worst = max(worst, float(jnp.max(s.S.data)) - (S0 + 1.0),
                    S0 - float(jnp.min(s.S.data)))
    assert worst < 1e-3


def test_partial_cells_uniform_tendency_vanishes():
    """Sloped partial-cell bathymetry, below-seafloor cells at 0 psu: 6.3e-13 psu/s
    measured over active cells (2.8e-6 old)."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
    from legoesm.ocean.vertical import create_ocean_z_star, create_partial_cell_coordinate
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.state import OceanConfig
    from legoesm.ocean.dynamics.ocean_pe_cdgrid import ocean_baroclinic_tendencies_cdgrid

    n, nlev, hmax = 8, 5, 4000.0
    grid = create_cubed_sphere(n)
    cd = create_cubed_sphere_cdgrid(grid)
    zs = create_ocean_z_star(n_levels=nlev, H_max=hmax)
    lat = np.asarray(grid.lat)
    H = 1000.0 + (hmax - 1000.0) * 0.5 * (1.0 + np.sin(2.0 * lat))
    pc = create_partial_cell_coordinate(zs, jnp.asarray(H))
    st = rest_state_ocean(grid, pc, H_max=hmax)
    ones = jnp.ones_like(st.u.data)
    st = st._replace(
        land_mask=st.land_mask.replace(data=jnp.ones((6, n, n), st.land_mask.data.dtype)),
        H_bathy=st.H_bathy.replace(data=jnp.asarray(H)),
        S=st.S.replace(data=jnp.where(pc.is_active, S0, 0.0) * jnp.ones_like(st.S.data)),
        u=st.u.replace(data=0.1 * jnp.sin(jnp.asarray(grid.lon))[..., None] * ones),
        v=st.v.replace(data=-0.05 * jnp.cos(jnp.asarray(grid.lat))[..., None] * ones))
    cfg = OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0)
    t = ocean_baroclinic_tendencies_cdgrid(st, grid, pc, cd, cfg)
    active = jnp.asarray(pc.is_active)
    assert float(jnp.max(jnp.abs(jnp.where(active, t.dS_dt.data, 0.0)))) < 1e-10
